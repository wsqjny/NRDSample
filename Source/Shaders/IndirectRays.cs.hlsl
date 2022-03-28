/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( !defined( COMPILER_FXC ) )

#include "Shared.hlsli"
#include "RaytracingShared.hlsli"

// Inputs
NRI_RESOURCE( Texture2D<float>, gIn_ViewZ, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Normal_Roughness, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_BaseColor_Metalness, t, 2, 1 );
NRI_RESOURCE( Texture2D<float>, gIn_PrimaryMip, t, 3, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_PrevFinalLighting_PrevViewZ, t, 4, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Ambient, t, 5, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_Motion, t, 6, 1 );

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gOut_Diff, u, 7, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Spec, u, 8, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_DiffDirectionPdf, u, 9, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_SpecDirectionPdf, u, 10, 1 );
NRI_RESOURCE( RWTexture2D<float>, gOut_Downsampled_ViewZ, u, 11, 1 );
NRI_RESOURCE( RWTexture2D<float3>, gOut_Downsampled_Motion, u, 12, 1 );
NRI_RESOURCE( RWTexture2D<float4>, gOut_Downsampled_Normal_Roughness, u, 13, 1 );


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Falcor Path tracer begain.

// win10 sdk corecrt_math_defines.h
#define M_E        2.71828182845904523536   // e
#define M_LOG2E    1.44269504088896340736   // log2(e)
#define M_LOG10E   0.434294481903251827651  // log10(e)
#define M_LN2      0.693147180559945309417  // ln(2)
#define M_LN10     2.30258509299404568402   // ln(10)
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)




#define MAX_SURFACE_BOUNCES 10
#define MAX_DIFFUSE_BOUNCES 10
#define MAX_SPECULAR_BOUNCES 10
#define MAX_TRANSMISSON_BOUNCES 10

// user config.
static const uint kMaxSurfaceBounces = MAX_SURFACE_BOUNCES;
static const uint kMaxDiffuseBounces = MAX_DIFFUSE_BOUNCES;
static const uint kMaxSpecularBounces = MAX_SPECULAR_BOUNCES;
static const uint kMaxTransmissionBounces = MAX_TRANSMISSON_BOUNCES;

static const bool kUseEnvLight = true;                                      // whether to enable env light.
static const bool kUseEmissiveLights = false;                               // not support now.
static const bool kUseAnalyticLights = true;                                // whether to enable analytic light, like point light, directional light ... local light. should prepare scene light first.

static const bool kDisableCaustics = true;
static const bool kUseRussianRoulette = true;

// should not change, for debug
static const bool kUseRTXDI = false;                                        // not support now.
static const bool kUseNEE = true;                                           // NEE
static const bool kUseMIS = true;                                           // MIS
static const bool kUseLightsInDielectricVolumes = false;                    //
static const bool kOutputNRDData = true;                                    // for denoise
static const bool kUseBSDFSampling = true;                                  // debug only.
static const uint misHeuristic = 1;                                         // MIS method.



/** Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
    \param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}




struct FalcorPayload
{
    // geometry
    float3 X;
    float3 rayDirection;
    float3 N;
    float3 faceN;

    // material
    float3 diffuse;
    float3 specular;
    float  roughness;
    float  metallic;
    float  ior;
    float3 transmission;
    float  diffuseTransmission;
    float  specularTransmission;
};



#include "FalcorSampleGenerator.hlsli"

#include "FalcorIBxDF.hlsli"
#include "FalcorLobeType.hlsli"
#include "FalcorShadingData.hlsli"
#include "FalcorMicrofacet.hlsli"
#include "FalcorFresnel.hlsli"
#include "FalcorBxDF.hlsli"
#include "FalcorStandardBSDF.hlsli"
#include "FalcorLoadShadingData.hlsli"
#include "FalcorPathTracerLightData.hlsli"
#include "FalcorPathTracerLightHelper.hlsli"




StandardBSDFData LoadStanderdBSDFData(FalcorPayload payload, ShadingData sd)
{
    StandardBSDFData data;

    data.diffuse = payload.diffuse;
    data.specular = payload.specular;
    data.roughness = payload.roughness;
    data.metallic = payload.metallic;
    data.eta = sd.frontFacing ? (sd.IoR / payload.ior) : (payload.ior / sd.IoR);
    data.transmission = payload.transmission;
    data.diffuseTransmission = payload.diffuseTransmission;
    data.specularTransmission = payload.specularTransmission;

    return data;
}

FalcorPayload PrepareForPrimaryRayPayload(uint2 pixelPos, float viewZ)
{
    float2 pixelUv = float2(pixelPos + 0.5) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[pixelPos]);
    float4 baseColorMetalness = gIn_BaseColor_Metalness[pixelPos];

    float3 Xv = STL::Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, viewZ, gOrthoMode);
    float3 X = STL::Geometry::AffineTransform(gViewToWorld, Xv);
    float3 V = GetViewVector(X);
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[pixelPos];

    float NoV0 = abs(dot(N, V));

    float3 albedo0, Rf00;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0(baseColorMetalness.xyz, baseColorMetalness.w, albedo0, Rf00);
    albedo0 = max(albedo0, 0.001);


    float mtlIoR = 1.1;
    float mtlTransmission = 0.0;
    float mtlDiffTrans = 0.0;
    float mtlSpecTrans = 0.0;

    if (0 && baseColorMetalness.w > 0.95)
    {
        mtlTransmission = 1.0;
        mtlSpecTrans = 1.0;
    }

    FalcorPayload payload;

    
    //geometry.
    {
        //assum primary is front face.
        payload.X = X;
        payload.rayDirection = -V;
        payload.N = N;
        payload.faceN = N;
    }

    // material
    {
        payload.diffuse = albedo0;
        payload.specular = Rf00;
        payload.roughness = normalAndRoughness.w;
        payload.metallic = baseColorMetalness.w;
        payload.ior = mtlIoR;
        payload.transmission = mtlTransmission;
        payload.diffuseTransmission = mtlDiffTrans;
        payload.specularTransmission = mtlSpecTrans;
    }

    return payload;
}
FalcorPayload FillFalcorPayloadAfterTrace(GeometryProps geometryProps, bool useSimplifiedModel = false)
{
    FalcorPayload payload = (FalcorPayload)0;

    // geometry
    {
        payload.X = geometryProps.X;
        payload.rayDirection = geometryProps.rayDirection;
        payload.N = geometryProps.N;
        payload.faceN = geometryProps.faceN;
    }   


    float sdIOR = 1.0;
    bool sdFrontFacing = dot(-payload.rayDirection, payload.faceN) >= 0.f;


    // material
    {
        [branch]
        if (geometryProps.IsSky())
        {
            return payload;
        }

        uint baseTexture = geometryProps.GetBaseTexture();
        float3 mips = GetRealMip(baseTexture, geometryProps.mip);

        // Base color
        float4 color = gIn_Textures[baseTexture].SampleLevel(gLinearMipmapLinearSampler, geometryProps.uv, mips.z);
        color.xyz *= geometryProps.IsTransparent() ? 1.0 : STL::Math::PositiveRcp(color.w); // Correct handling of BC1 with pre-multiplied alpha
        float3 baseColor = saturate(color.xyz);

        // Roughness and metalness
        float3 materialProps = gIn_Textures[baseTexture + 1].SampleLevel(gLinearMipmapLinearSampler, geometryProps.uv, mips.z).xyz;
        float roughness = materialProps.y;
        float metalness = materialProps.z;

        // Normal
        float2 packedNormal = gIn_Textures[baseTexture + 2].SampleLevel(gLinearMipmapLinearSampler, geometryProps.uv, mips.y).xy;
        packedNormal = gUseNormalMap ? packedNormal : (127.0 / 255.0);
        float3 N = STL::Geometry::TransformLocalNormal(packedNormal, geometryProps.T, geometryProps.N);
        N = useSimplifiedModel ? geometryProps.N : N;

        // Emission
        float3 Lemi = gIn_Textures[baseTexture + 3].SampleLevel(gLinearMipmapLinearSampler, geometryProps.uv, mips.x).xyz;
        Lemi *= (baseColor + 0.01) / (max(baseColor, max(baseColor, baseColor)) + 0.01);
        Lemi = geometryProps.IsForcedEmission() ? geometryProps.GetForcedEmissionColor() : Lemi;
        Lemi *= gEmissionIntensity * float(geometryProps.IsEmissive());

        // Override material
        [flatten]
        if (gForcedMaterial == MAT_GYPSUM)
        {
            roughness = 1.0;
            baseColor = 0.5;
            metalness = 0.0;
        }
        else if (gForcedMaterial == MAT_COBALT)
        {
            roughness = pow(saturate(baseColor.x * baseColor.y * baseColor.z), 0.33333);
            baseColor = float3(0.672411, 0.637331, 0.585456);
            metalness = 1.0;
        }

        metalness = gMetalnessOverride == 0.0 ? metalness : gMetalnessOverride;
        roughness = gRoughnessOverride == 0.0 ? roughness : gRoughnessOverride;


        // sample material   
        float3 diffuse, specular;
        STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0(baseColor, metalness, diffuse, specular);
        diffuse = max(diffuse, 0.001);

        float mtlIoR = 1.1;
        float mtlTransmission = 0.0;
        float mtlDiffTrans = 0.0;
        float mtlSpecTrans = 0.0;

        if (0 && metalness > 0.95)
        {
            mtlTransmission = 1.0;
            mtlSpecTrans = 1.0;
        }


        payload.diffuse = diffuse;
        payload.specular = specular;
        payload.roughness = roughness;
        payload.metallic = metalness;
        payload.ior = mtlIoR;
        payload.transmission = mtlTransmission;
        payload.diffuseTransmission = mtlDiffTrans;
        payload.specularTransmission = mtlSpecTrans;
    }

    return payload;
}


struct SceneLights
{
    uint LightCount;
    LightData LightArray[3];
};


static const float kNRDDepthRange = 10000.0f;
static const float kNRDInvalidPathLength = HLF_MAX;





struct VisibilityQuery
{
    bool traceVisibilityRay(const RayDesc ray)
    {
        float2 mipAndCone = float2(0.0, 1.0);

        return CastVisibilityRay_AnyHit(ray.Origin, ray.Direction, ray.TMin, ray.TMax, mipAndCone, gWorldTlas, 0xff, RAY_FLAG_NONE);
    }
};



#include "FalcorPathState.hlsli"
#include "FalcorPathTracer.hlsli"


void PrepareSceneLight(inout SceneLights sl)
{
    LightData sunlight;
    sunlight.dirW = -gSunDirection;
    sunlight.intensity = GetSunIntensity(gSunDirection, gSunDirection, gTanSunAngularRadius);
    sunlight.type = LightType_Directional;

    sl.LightCount = 1;
    sl.LightArray[0] = sunlight;    
}



[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    float2 rectSize = gRectSize;
    if( pixelPos.x >= rectSize.x || pixelPos.y >= rectSize.y )
        return;

    uint2 outPixelPos = pixelPos;

    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Early out
    float viewZ = gIn_ViewZ[ pixelPos ];
    if( abs( viewZ ) == INF )
    {
        gOut_Diff[ outPixelPos ] = 0;
        gOut_Spec[ outPixelPos ] = 0;
        return;
    }

   



    // Ambient
    //float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    //Lamb *= gAmbient;

    // Secondary rays
    STL::Rng::Initialize(pixelPos, gFrameIndex);


#if 1
    float4 diffIndirect = 0;
    float4 diffDirectionPdf = 0;
    float diffTotalWeight = 1e-6;

    float4 specIndirect = 0;
    float4 specDirectionPdf = 0;
    float specTotalWeight = 1e-6;

    uint sampleNum = gSampleNum << 1;
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;
    
    float mip = 0.0;
    float hitT = 0.0;
    



    uint pathID = pixelPos.x | (pixelPos.y << 12);

    PathTracer gPathTracer = (PathTracer)0.0;
    PathState path = (PathState)0.0;
    VisibilityQuery vq;

    PrepareSceneLight(gPathTracer.sceneLight);                                       // prepare scene light.

    gPathTracer.generatePath(pathID, path);                                          // generate path from camera
    //gPathTracer.setupPathLogging(path);


    // primay ray hit
    FalcorPayload primaryFalcorPayload = PrepareForPrimaryRayPayload(pixelPos, viewZ);

    gPathTracer.handleHit(path, primaryFalcorPayload, vq);                           // handle primary hit

    while (path.isActive())                                                          // handle path trace
    {
        // next hit
        {
            // Advance to next path vertex.
            path.incrementVertexIndex();

            // Trace ray.
            //logTraceRay(PixelStatsRayType::ClosestHit);
            const RayDesc ray = path.getScatterRay();

            //PathPayload payload = PathPayload::pack(path);
            //uint rayFlags = RAY_FLAG_NONE;
            //if (!kUseAlphaTest) rayFlags |= RAY_FLAG_FORCE_OPAQUE;    
            //TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kMissScatter /* missIdx */, ray.toRayDesc(), payload);
            //path = PathPayload::unpack(payload);

            float2 mipAndCone = GetConeAngleFromRoughness(path.mip, path.roughness);         ///!!!TODO
            GeometryProps geometryProps0 = CastRay(ray.Origin, ray.Direction, ray.TMin, ray.TMax, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0, 0);
            FalcorPayload falcorPayload = FillFalcorPayloadAfterTrace(geometryProps0);


            float hitT = geometryProps0.tmin;

            
            if (hitT == INF)
            {
                // handle miss
                
                path.clearHit();
                path.sceneLength = (kNRDInvalidPathLength);

                //gPathTracer.setupPathLogging(path);
                gPathTracer.handleMiss(path);

                //payload = PathPayload::pack(path);
            }
            else
            {
                // handle hit

                path.sceneLength += (hitT);

                //gPathTracer.setupPathLogging(path);
                //VisibilityQuery vq;
                gPathTracer.handleHit(path, falcorPayload, vq);
            }
        }
    }

    // output nrd
    {   
        float pathLengthMod = path.outSceneLength;
        bool isSample1Diffuse = path.outIsDiffsue;


        float3 irradiance = path.L;

        // De-modulate materials for denoising
        float3 V = GetViewVector(primaryFalcorPayload.X);
        float3 N = primaryFalcorPayload.N;
        float NoV0 = abs(dot(N, V));

        float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross(primaryFalcorPayload.specular, NoV0, primaryFalcorPayload.roughness);
        envBRDF0 = max(envBRDF0, 0.001);

        irradiance /= isSample1Diffuse ? primaryFalcorPayload.diffuse : envBRDF0;



        float normDist = REBLUR_FrontEnd_GetNormHitDist(pathLengthMod, viewZ, gHitDistParams, isSample1Diffuse ? 1.0 : primaryFalcorPayload.roughness);
        float4 nrdData = REBLUR_FrontEnd_PackRadianceAndHitDist(irradiance, normDist, USE_SANITIZATION);
        if (gDenoiserType != REBLUR)
            nrdData = RELAX_FrontEnd_PackRadianceAndHitDist(irradiance, pathLengthMod, USE_SANITIZATION);

        diffIndirect += nrdData * float(isSample1Diffuse);
        specIndirect += nrdData * float(!isSample1Diffuse);
    }

    {
        gOut_Diff[outPixelPos] = diffIndirect;
        gOut_DiffDirectionPdf[ outPixelPos ] = diffDirectionPdf;

        gOut_Spec[ outPixelPos ] = specIndirect;
        gOut_SpecDirectionPdf[ outPixelPos ] = specDirectionPdf;
    }

#endif


#if 0
    {
        float eta = 1.5;
        SampleGenerator sgg;
        float lobeSample = sampleNext1D(sgg);

        float3x3 mLocalBasis = STL::Geometry::GetBasis(N);         


        float3 spOrigin = X;
        float3 spNormal = N;

        float3 wiWorld = normalize(V);
        float3 wiLocal = STL::Geometry::RotateVector(mLocalBasis, wiWorld);                                   // world to local   
        float3 wi = wiLocal;
        

        float cosThetaT;
        float F = evalFresnelDielectric(eta, wi.z, cosThetaT);


        bool isReflection = lobeSample < F;

        //spOrigin = computeRayOrigin(spOrigin, -N);


        float4 result = 0.0;
        if (isReflection)
        {
            result = float4(1, 0, 0, 1);
        }
 


        gOut_Spec[outPixelPos] = result;
        return ;
    }
#endif


#if 0
    gOut_Spec[outPixelPos] = float4(worldPosition, 1);
    return;
#endif


#if 0
    {
        if (baseColorMetalness.w > 0.9)
        {
            float3 origin = _GetXoffset(X, N);// computeRayOrigin(X, N);
            float3 dir = reflect(-V, N);
            float2 mipAndCone = float2(0.0, 1.0);

            bool bIsMiss = CastVisibilityRay_AnyHit(origin, dir, 0.0, 10000.0, mipAndCone, gWorldTlas, 0xff, RAY_FLAG_NONE);

            float4 result = 0.0;
            if (!bIsMiss)
            {
                result = float4(1, 1, 0, 1);                
            }

            gOut_Spec[outPixelPos] = result;
            return;
        }
        else
        {
            gOut_Diff[outPixelPos] = 0;
            gOut_Spec[outPixelPos] = 0;
            return;
        }
    }
#endif



#if 0
    {
        if (baseColorMetalness.w > 0.9)
        {
            float3 origin = computeRayOrigin(worldPosition, N);
            V = normalize(-X);

            float3 dir = reflect(-V, N);
            float2 mipAndCone = float2(0.0, 1.0);

            bool bIsMiss = CastVisibilityRay_AnyHit(origin, dir, 0.0, 10000.0, mipAndCone, gWorldTlas, 0xff, RAY_FLAG_NONE);

            float4 result = 0.0;
            if (!bIsMiss)
            {
                result = float4(1, 1, 0, 1);
            }

            gOut_Spec[outPixelPos] = result;
            return;
        }
        else
        {
            gOut_Diff[outPixelPos] = 0;
            gOut_Spec[outPixelPos] = 0;
            return;
        }
    }
#endif


#if 0       // test for back or front N
    //gOut_Spec[outPixelPos] = float4(N, 1);

    float2 mipAndCone = GetConeAngleFromRoughness(0, 0);         ///!!!TODO

    float3 rayOrigin = computeRayOrigin(0, -N);
    GeometryProps geometryProps0 = CastRay(rayOrigin, -V, 0.f, kRayTMax, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0, 0);

    if (geometryProps0.tmin != INF)
    {
        gOut_Spec[outPixelPos] = float4(geometryProps0.N, 1);
    }
    else
    {
        gOut_Spec[outPixelPos] = float4(1,0,0, 1);
    }
    
#endif

#if 0
    {
        float2 mipAndCone = GetConeAngleFromRoughness(0, 0);         ///!!!TODO
        float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
        float eta = 1.0 / 1.5;
        SampleGenerator sgg;


        float3 rayOrigin = X;
        float3 rayDir = -V;
        float3 rayNormal = N;

        float4 result = 0.0;
        for (int i = 0; i < 4; i++)
        {
            float lobeSample = sampleNext1D(sgg);

            float3 wi = STL::Geometry::RotateVectorInverse(mLocalBasis, -rayDir);                     // local to world

            float cosThetaT;
            float F = evalFresnelDielectric(eta, wi.z, cosThetaT);

            bool isReflection = lobeSample < F;
            float3 wo = isReflection ? float3(-wi.x, -wi.y, wi.z) : float3(-wi.x * eta, -wi.y * eta, -cosThetaT);
            float3 woWorld = STL::Geometry::RotateVectorInverse(mLocalBasis, wo);

            
            rayOrigin = computeRayOrigin(rayOrigin, isReflection ? rayNormal : -rayNormal);

            GeometryProps geometryProps0 = CastRay(rayOrigin, woWorld, 0.f, kRayTMax, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0, 0);
            float hitT = geometryProps0.tmin;


            if (hitT == INF)   // miss
            {
                result.rgb = getSky(woWorld);
     
                break;
            }
            else
            {
                rayOrigin = rayOrigin + rayDir * hitT;
                rayDir = woWorld;
                rayNormal = geometryProps0.N;
            }
        }

        gOut_Spec[outPixelPos] = result;
        return;
    }
#endif
}

#else

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    // no TraceRayInline support, because of:
    //  - DXBC
    //  - SPIRV generation is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/4221
}

#endif
