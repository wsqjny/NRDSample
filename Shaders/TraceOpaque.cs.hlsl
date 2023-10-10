/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"
#include "Include/RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float3>, gIn_PrevComposedDiff, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevComposedSpec_PrevViewZ, t, 1, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 2, 1 );
NRI_RESOURCE( Texture2D<uint3>, gIn_Scrambling_Ranking_1spp, t, 3, 1 );
NRI_RESOURCE( Texture2D<uint4>, gIn_Sobol, t, 4, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float3>, gOut_Mv, u, 0, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_ViewZ, u, 1, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Normal_Roughness, u, 2, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_BaseColor_Metalness, u, 3, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectLighting, u, 4, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_DirectEmission, u, 5, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_PsrThroughput, u, 6, 1 );
NRI_RESOURCE( RWTexture2D<float2>, gOut_ShadowData, u, 7, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Shadow_Translucency, u, 8, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 9, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 10, 1 );
#if( NRD_MODE == SH )
    NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffSh, u, 11, 1 );
    NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecSh, u, 12, 1 );
#endif

float2 GetBlueNoise( Texture2D<uint3> texScramblingRanking, uint2 pixelPos, bool isCheckerboard, uint seed, uint sampleIndex, uint sppVirtual = 1, uint spp = 1 )
{
    // Final SPP - total samples per pixel ( there is a different "gIn_Scrambling_Ranking" texture! )
    // SPP - samples per pixel taken in a single frame ( must be POW of 2! )
    // Virtual SPP - "Final SPP / SPP" - samples per pixel distributed in time ( across frames )

    // Based on:
    //     https://eheitzresearch.wordpress.com/772-2/
    // Source code and textures can be found here:
    //     https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html (but 2D only)

    // Sample index
    uint frameIndex = isCheckerboard ? ( gFrameIndex >> 1 ) : gFrameIndex;
    uint virtualSampleIndex = ( frameIndex + seed ) & ( sppVirtual - 1 );
    sampleIndex &= spp - 1;
    sampleIndex += virtualSampleIndex * spp;

    // The algorithm
    uint3 A = texScramblingRanking[ pixelPos & 127 ];
    uint rankedSampleIndex = sampleIndex ^ A.z;
    uint4 B = gIn_Sobol[ uint2( rankedSampleIndex & 255, 0 ) ];
    float4 blue = ( float4( B ^ A.xyxy ) + 0.5 ) * ( 1.0 / 256.0 );

    // Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = STL::Sequence::Bayer4x4ui( pixelPos, gFrameIndex );
    float2 dither = ( float2( d & 3, d >> 2 ) + 0.5 ) * ( 1.0 / 4.0 );
    blue += ( dither.xyxy - 0.5 ) * ( 1.0 / 256.0 );

    return saturate( blue.xy );
}

float4 GetRadianceFromPreviousFrame( GeometryProps geometryProps, MaterialProps materialProps, uint2 pixelPos, bool isDiffuse )
{
    // Reproject previous frame
    float3 prevLdiff, prevLspec;
    float prevFrameWeight = ReprojectIrradiance( true, false, gIn_PrevComposedDiff, gIn_PrevComposedSpec_PrevViewZ, geometryProps, pixelPos, prevLdiff, prevLspec );
    prevFrameWeight *= gPrevFrameConfidence; // see C++ code for details

    // Estimate how strong lighting at hit depends on the view direction
    float diffuseProbabilityBiased = EstimateDiffuseProbability( geometryProps, materialProps, true );
    float3 prevLsum = prevLdiff + prevLspec * diffuseProbabilityBiased;

    float diffuseLikeMotion = lerp( diffuseProbabilityBiased, 1.0, STL::Math::Sqrt01( materialProps.curvature ) );
    prevFrameWeight *= isDiffuse ? 1.0 : diffuseLikeMotion;

    float a = STL::Color::Luminance( prevLdiff );
    float b = STL::Color::Luminance( prevLspec );
    prevFrameWeight *= lerp( diffuseProbabilityBiased, 1.0, ( a + NRD_EPS ) / ( a + b + NRD_EPS ) );

    return float4( prevLsum, prevFrameWeight );
}

//========================================================================================
// TRACE OPAQUE
//========================================================================================

/*
"TraceOpaque" traces "pathNum" paths, doing up to "bounceNum" bounces. The function
has not been designed to trace primary hits. But still can be used to trace
direct and indirect lighting.

Prerequisites:
    STL::Rng::Hash::Initialize( )

Derivation:
    Lsum = L0 + BRDF0 * ( L1 + BRDF1 * ( L2 + BRDF2 * ( L3 +  ... ) ) )

    Lsum = L0 +
        L1 * BRDF0 +
        L2 * BRDF0 * BRDF1 +
        L3 * BRDF0 * BRDF1 * BRDF2 +
        ...
*/

struct TraceOpaqueDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Material properties
    MaterialProps materialProps;

    // Pixel position
    uint2 pixelPos;

    // Checkerboard
    uint checkerboard;

    // Number of paths to trace
    uint pathNum;

    // Number of bounces to trace ( up to )
    uint bounceNum;

    // Instance inclusion mask ( DXR )
    uint instanceInclusionMask;

    // Ray flags ( DXR )
    uint rayFlags;
};

struct TraceOpaqueResult
{
    float3 diffRadiance;
    float diffHitDist;

    float3 specRadiance;
    float specHitDist;

    #if( NRD_MODE == DIRECTIONAL_OCCLUSION || NRD_MODE == SH )
        float3 diffDirection;
        float3 specDirection;
    #endif
};

bool IsPsrAllowed( MaterialProps materialProps )
{
    return materialProps.roughness < 0.044 && ( materialProps.metalness > 0.941 || STL::Color::Luminance( materialProps.baseColor ) < 0.005 ); // TODO: tweaked for some content?
}

TraceOpaqueResult TraceOpaque( TraceOpaqueDesc desc )
{
    TraceOpaqueResult result = ( TraceOpaqueResult )0;

    GeometryProps geometryProps = desc.geometryProps;
    MaterialProps materialProps = desc.materialProps;

    float3 Lsum = 0.0;
    float3 pathThroughput = 1.0;

    float3x3 mLocalBasis = STL::Geometry::GetBasis(materialProps.N);
    float3 Vlocal = STL::Geometry::RotateVector(mLocalBasis, geometryProps.V);
    float3 ray = 0;
    {
        float2 rnd = STL::Rng::Hash::GetFloat2();

        float3 r;
        r = STL::ImportanceSampling::Cosine::GetRay(rnd);
        r = STL::Geometry::RotateVectorInverse(mLocalBasis, r);
        ray = r;
    }

    float2 mipAndCone = GetConeAngleFromRoughness(geometryProps.mip, 1.0);
    geometryProps = CastRay(geometryProps.GetXoffset(), ray, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags);
    

    float normHitDist = geometryProps.tmin;
    if (gDenoiserType != DENOISER_RELAX)
    {
        float viewZ = STL::Geometry::AffineTransform(gWorldToView, geometryProps.X).z;
        normHitDist = REBLUR_FrontEnd_GetNormHitDist(normHitDist, viewZ, gHitDistParams, 1.0);
    }

    
    if (geometryProps.IsSky())
    {
        Lsum = 1.0;
    }
    else
    {
        Lsum = 0.0;
    }

    result.diffRadiance = Lsum;
    result.diffHitDist = normHitDist;
    return result;
}

//========================================================================================
// MAIN
//========================================================================================

void WriteResult( uint checkerboard, uint2 outPixelPos, float4 diff, float4 spec, float4 diffSh, float4 specSh )
{
    if( gTracingMode == RESOLUTION_HALF )
    {
        if( checkerboard )
        {
            gOut_Diff[ outPixelPos ] = diff;
            #if( NRD_MODE == SH )
                gOut_DiffSh[ outPixelPos ] = diffSh;
            #endif
        }
        else
        {
            gOut_Spec[ outPixelPos ] = spec;
            #if( NRD_MODE == SH )
                gOut_SpecSh[ outPixelPos ] = specSh;
            #endif
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = diff;
        gOut_Spec[ outPixelPos ] = spec;
        #if( NRD_MODE == SH )
            gOut_DiffSh[ outPixelPos ] = diffSh;
            gOut_SpecSh[ outPixelPos ] = specSh;
        #endif
    }
}

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    const float NaN = sqrt( -1 );

    // Checkerboard
    uint2 outPixelPos = pixelPos;
    if( gTracingMode == RESOLUTION_HALF )
        outPixelPos.x >>= 1;

    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    // Do not generate NANs for unused threads
    if( pixelPos.x >= gRectSize.x || pixelPos.y >= gRectSize.y )
    {
        #if( USE_DRS_STRESS_TEST == 1 )
            WriteResult( checkerboard, outPixelPos, NaN, NaN, NaN, NaN );
        #endif

        return;
    }

    // Pixel and sample UV
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    //================================================================================================================================================================================
    // Primary rays
    //================================================================================================================================================================================

    // Initialize RNG
    STL::Rng::Hash::Initialize( pixelPos, gFrameIndex );

    // Primary ray
    float3 cameraRayOrigin = ( float3 )0;
    float3 cameraRayDirection = ( float3 )0;
    GetCameraRay( cameraRayOrigin, cameraRayDirection, sampleUv );

    GeometryProps geometryProps0 = CastRay( cameraRayOrigin, cameraRayDirection, 0.0, INF, GetConeAngleFromRoughness( 0.0, 0.0 ), gWorldTlas, ( gOnScreen == SHOW_INSTANCE_INDEX || gOnScreen == SHOW_NORMAL ) ? GEOMETRY_ALL : GEOMETRY_IGNORE_TRANSPARENT, 0 );
    MaterialProps materialProps0 = GetMaterialProps( geometryProps0 );

    // ViewZ
    float viewZ = STL::Geometry::AffineTransform( gWorldToView, geometryProps0.X ).z;
    viewZ = geometryProps0.IsSky( ) ? STL::Math::Sign( viewZ ) * INF : viewZ;
    gOut_ViewZ[ pixelPos ] = viewZ;

    // Motion
    float3 motion = GetMotion( geometryProps0.X, geometryProps0.Xprev );
    gOut_Mv[ pixelPos ] = motion;

    // Early out - sky
    if( geometryProps0.IsSky( ) )
    {
        gOut_ShadowData[ pixelPos ] = SIGMA_FrontEnd_PackShadow( viewZ, 0.0, 0.0 );
        gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi;

        return;
    }

    // G-buffer
    float diffuseProbability = EstimateDiffuseProbability( geometryProps0, materialProps0 );
    uint materialID = geometryProps0.IsHair( ) ? MATERIAL_ID_HAIR : ( diffuseProbability != 0.0 ? MATERIAL_ID_DEFAULT : MATERIAL_ID_METAL );

    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( gDebug == 0.0 )
            materialID *= float( frac( geometryProps0.X ).x < 0.05 );
    #endif

    gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( materialProps0.N, materialProps0.roughness, materialID );
    gOut_BaseColor_Metalness[ pixelPos ] = float4( STL::Color::LinearToSrgb( materialProps0.baseColor ), materialProps0.metalness );

    // Debug
    if( gOnScreen == SHOW_INSTANCE_INDEX )
    {
        STL::Rng::Hash::Initialize( geometryProps0.instanceIndex, 0 );

        uint checkerboard = STL::Sequence::CheckerBoard( pixelPos >> 2, 0 ) != 0;
        float3 color = STL::Rng::Hash::GetFloat4( ).xyz;
        color *= checkerboard && !geometryProps0.IsStatic( ) ? 0.5 : 1.0;

        materialProps0.Ldirect = color;
    }
    else if( gOnScreen == SHOW_UV )
        materialProps0.Ldirect = float3( frac( geometryProps0.uv ), 0 );
    else if( gOnScreen == SHOW_CURVATURE )
        materialProps0.Ldirect = sqrt( abs( materialProps0.curvature ) );
    else if( gOnScreen == SHOW_MIP_PRIMARY )
    {
        float mipNorm = STL::Math::Sqrt01( geometryProps0.mip / MAX_MIP_LEVEL );
        materialProps0.Ldirect = STL::Color::ColorizeZucconi( mipNorm );
    }

    // Unshadowed sun lighting and emission
    gOut_DirectLighting[ pixelPos ] = materialProps0.Ldirect;
    gOut_DirectEmission[ pixelPos ] = materialProps0.Lemi;

    // Sun shadow
    float2 rnd = GetBlueNoise( gIn_Scrambling_Ranking_1spp, pixelPos, false, 0, 0 );
    if( gDenoiserType == DENOISER_REFERENCE )
        rnd = STL::Rng::Hash::GetFloat2( );
    rnd = STL::ImportanceSampling::Cosine::GetRay( rnd ).xy;
    rnd *= gTanSunAngularRadius;

    float3x3 mSunBasis = STL::Geometry::GetBasis( gSunDirection_gExposure.xyz ); // TODO: move to CB
    float3 sunDirection = normalize( mSunBasis[ 0 ] * rnd.x + mSunBasis[ 1 ] * rnd.y + mSunBasis[ 2 ] );
    float3 Xoffset = geometryProps0.GetXoffset( );
    float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps0.mip, gTanSunAngularRadius );

    float shadowTranslucency = ( STL::Color::Luminance( materialProps0.Ldirect ) != 0.0 && !gDisableShadowsAndEnableImportanceSampling ) ? 1.0 : 0.0;
    float shadowHitDist = 0.0;

    while( shadowTranslucency > 0.01 )
    {
        GeometryProps geometryPropsShadow = CastRay( Xoffset, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_ALL, 0 );

        if( geometryPropsShadow.IsSky( ) )
        {
            shadowHitDist = shadowHitDist == 0.0 ? INF : shadowHitDist;
            break;
        }

        // ( Biased ) Cheap approximation of shadows through glass
        float NoV = abs( dot( geometryPropsShadow.N, sunDirection ) );
        shadowTranslucency *= lerp( geometryPropsShadow.IsTransparent( ) ? 0.9 : 0.0, 0.0, STL::Math::Pow01( 1.0 - NoV, 2.5 ) );

        // Go to the next hit
        Xoffset += sunDirection * ( geometryPropsShadow.tmin + 0.001 );
        shadowHitDist += geometryPropsShadow.tmin;
    }

    float4 shadowData1;
    float2 shadowData0 = SIGMA_FrontEnd_PackShadow( viewZ, shadowHitDist == INF ? NRD_FP16_MAX : shadowHitDist, gTanSunAngularRadius, shadowTranslucency, shadowData1 );

    gOut_ShadowData[ pixelPos ] = shadowData0;
    gOut_Shadow_Translucency[ pixelPos ] = shadowData1;

    // This code is needed to avoid self-intersections if tracing starts from crunched g-buffer coming from textures
    #if 0
    {
        float3 V = -cameraRayDirection;
        float3 N = materialProps0.N;
        float zScale = 0.0003 + abs( viewZ ) * 0.00005;
        float NoV0 = abs( dot( N, V ) );

        Xoffset = _GetXoffset( geometryProps0.X, N );
        Xoffset += V * zScale;
        Xoffset += N * STL::BRDF::Pow5( NoV0 ) * zScale;

        geometryProps0.X = Xoffset;
    }
    #endif

    //================================================================================================================================================================================
    // Secondary rays ( indirect and direct lighting from local light sources )
    //================================================================================================================================================================================

    TraceOpaqueDesc desc = ( TraceOpaqueDesc )0;
    desc.geometryProps = geometryProps0;
    desc.materialProps = materialProps0;
    desc.pixelPos = pixelPos;
    desc.checkerboard = checkerboard;
    desc.pathNum = gSampleNum;
    desc.bounceNum = gBounceNum; // TODO: adjust by roughness?
    desc.instanceInclusionMask = GEOMETRY_IGNORE_TRANSPARENT;
    desc.rayFlags = 0;

    TraceOpaqueResult result = TraceOpaque( desc );

    // Debug
    #if( USE_SIMULATED_MATERIAL_ID_TEST == 1 )
        if( frac( X ).x < 0.05 )
            result.diffRadiance = float3( 0, 10, 0 ) * STL::Color::Luminance( result.diffRadiance );
    #endif

    #if( USE_SIMULATED_FIREFLY_TEST == 1 )
        const float maxFireflyEnergyScaleFactor = 10000.0;
        result.diffRadiance /= lerp( 1.0 / maxFireflyEnergyScaleFactor, 1.0, STL::Rng::Hash::GetFloat( ) );
    #endif

    //================================================================================================================================================================================
    // Output
    //================================================================================================================================================================================

    float4 outDiff = 0.0;
    float4 outSpec = 0.0;
    float4 outDiffSh = 0.0;
    float4 outSpecSh = 0.0;

    if( gDenoiserType == DENOISER_RELAX )
    {
    #if( NRD_MODE == SH )
        outDiff = RELAX_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #else
        outDiff = RELAX_FrontEnd_PackRadianceAndHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = RELAX_FrontEnd_PackRadianceAndHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
    }
    else
    {
    #if( NRD_MODE == OCCLUSION )
        outDiff = result.diffHitDist;
        outSpec = result.specHitDist;
    #elif( NRD_MODE == SH )
        outDiff = REBLUR_FrontEnd_PackSh( result.diffRadiance, result.diffHitDist, result.diffDirection, outDiffSh, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackSh( result.specRadiance, result.specHitDist, result.specDirection, outSpecSh, USE_SANITIZATION );
    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
        outDiff = REBLUR_FrontEnd_PackDirectionalOcclusion( result.diffDirection, result.diffHitDist, USE_SANITIZATION );
    #else
        outDiff = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.diffRadiance, result.diffHitDist, USE_SANITIZATION );
        outSpec = REBLUR_FrontEnd_PackRadianceAndNormHitDist( result.specRadiance, result.specHitDist, USE_SANITIZATION );
    #endif
    }

    WriteResult( checkerboard, outPixelPos, outDiff, outSpec, outDiffSh, outSpecSh );
}
