/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if( !defined( COMPILER_FXC ) && !defined( VULKAN ) )

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

/*
"TracePath" continues tracing from a given bounce, to start from the camera do the following:
    tracePathPayload.materialProps = ( MaterialProps )0;
    tracePathPayload.geometryProps = ( GeometryProps )0;
    tracePathPayload.geometryProps.X = LinePlaneIntersection( cameraPos, cameraView, nearPlane );

Prerequisites:
    STL::Rng::Initialize( )

Derivation:
    Lsum = L0 + BRDF0 * ( L1 + BRDF1 * ( L2 + BRDF2 * ( L3 +  ... ) ) )

    Lsum = L0 +
        L1 * BRDF0 +
        L2 * BRDF0 * BRDF1 +
        L3 * BRDF0 * BRDF1 * BRDF2 +
        ...

    for each bounce
    {
        Lsum += L[i] * BRDF
        pathLength += F( tmin[i], ... )
        BRDF *= BRDF[i]
    }
*/

struct TracePathDesc
{
    // Non-jittered pixel UV
    float2 pixelUv;

    // BRDF energy threshold
    float threshold;

    // Bounces to trace
    uint bounceNum;

    // Instance inclusion mask ( DXR )
    uint instanceInclusionMask;

    // Ray flags ( DXR )
    uint rayFlags;

    // A hint to use simplified materials ( flat colors, flat normals, etc. )
    bool useSimplifiedModel;

    // Some global ambient to be applied at the end of the path
    float3 Lamb;
};

struct TracePathPayload
{
    // Geometry properties
    GeometryProps geometryProps;

    // Material properties
    MaterialProps materialProps;

    // Left by bounce preceding input bounce ( 1 if starting from primary hits or from the camera )
    float3 BRDF;

    // Left by input bounce or 0
    float3 Lsum;

    // Accumulated previous frame weight
    float accumulatedPrevFrameWeight;

    // Left by input bounce or 0
    float pathLength;

    // Input bounce index ( 0 if tracing starts from the camera )
    uint bounceIndex;

    // Diffuse or specular path ( at this event, next event will be stochastically estimated )
    bool isDiffuse;
};

// Taken out from NRD
float GetSpecMagicCurve( float roughness, float power = 0.25 )
{
    float f = 1.0 - exp2( -200.0 * roughness * roughness );
    f *= STL::Math::Pow01( roughness, power );

    return f;
}

float3 GetRadianceFromPreviousFrame( GeometryProps geometryProps, float2 pixelUv, inout float weight )
{
    float4 clipPrev = STL::Geometry::ProjectiveTransform( gWorldToClipPrev, geometryProps.X ); // Not Xprev because confidence is based on viewZ
    float2 uvPrev = ( clipPrev.xy / clipPrev.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
    float4 prevLsum = gIn_PrevFinalLighting_PrevViewZ.SampleLevel( gNearestMipmapNearestSampler, uvPrev * gRectSizePrev * gInvScreenSize, 0 );
    float prevViewZ = abs( prevLsum.w ) / NRD_FP16_VIEWZ_SCALE;

    // Clear out bad values
    weight *= all( !isnan( prevLsum ) && !isinf( prevLsum ) );

    // Fade-out on screen edges
    float2 f = STL::Math::LinearStep( 0.0, 0.1, uvPrev ) * STL::Math::LinearStep( 1.0, 0.9, uvPrev );
    weight *= f.x * f.y;
    weight *= float( pixelUv.x > gSeparator );
    weight *= float( uvPrev.x > gSeparator );

    // Confidence - viewZ
    // No "abs" for clipPrev.w, because if it's negative we have a back-projection!
    float err = abs( prevViewZ - clipPrev.w ) * STL::Math::PositiveRcp( min( prevViewZ, abs( clipPrev.w ) ) );
    weight *= STL::Math::LinearStep( 0.02, 0.005, err );

    // Confidence - ignore back-facing
    // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
    float NoL = dot( geometryProps.N, gSunDirection );
    weight *= float( NoL * STL::Math::Sign( prevLsum.w ) > 0.0 );

    // Confidence - ignore too short rays
    float4 clip = STL::Geometry::ProjectiveTransform( gWorldToClip, geometryProps.X );
    float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5 - gJitter;
    float d = length( ( uv - pixelUv ) * gRectSize );
    weight *= STL::Math::LinearStep( 1.0, 3.0, d );

    // Ignore sky
    weight *= float( !geometryProps.IsSky() );

    return weight ? prevLsum.xyz : 0;
}

float GetBasePrevFrameWeight( )
{
    // Avoid "stuck in history" effect
    float weight = 0.9;
    weight *= 1.0 - gAmbientAccumSpeed;

    // Don't use in reference mode
    weight *= 1.0 - gReference;

    return weight;
}

bool IsNextEventDiffuse( GeometryProps geometryProps, MaterialProps materialProps )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    float lumDiff = STL::Color::Luminance( albedo ) + 1e-6;
    float lumSpec = STL::Color::Luminance( F ) + 1e-6;
    float diffProbability = lumDiff / ( lumDiff + lumSpec );
    float rnd = STL::Rng::GetFloat2( ).x;

    return rnd < diffProbability;
}

float EstimateDiffuseProbability( GeometryProps geometryProps, MaterialProps materialProps )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float smc = GetSpecMagicCurve( materialProps.roughness );
    float NoV = abs( dot( materialProps.N, -geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    float lumDiff = lerp( STL::Color::Luminance( albedo ), 1.0, smc ); // boost diffuse if roughness is high
    float lumSpec = STL::Color::Luminance( F ) * lerp( 10.0, 1.0, smc ); // boost specular if roughness is low
    float diffProb = lumDiff / ( lumDiff + lumSpec + 1e-6 );

    return diffProb;
}

float4 TracePath( TracePathDesc desc, inout TracePathPayload payload, float primaryHitRoughness )
{
    float2 mipAndCone = GetConeAngleFromRoughness( payload.geometryProps.mip, payload.materialProps.roughness );
    float4 directionPdf = 0;

    [loop]
    for( uint i = 0; i < desc.bounceNum && !payload.geometryProps.IsSky(); i++ )
    {
        // Choose ray
        float3 rayDirection = 0;
        if( payload.bounceIndex != 0 )
        {
            // Not primary ray
            float3x3 mLocalBasis = STL::Geometry::GetBasis( payload.materialProps.N );
            float3 Vlocal = STL::Geometry::RotateVector( mLocalBasis, -payload.geometryProps.rayDirection );
            float trimmingFactor = NRD_GetTrimmingFactor( payload.materialProps.roughness, gTrimmingParams );

            float VoH = 0;
            float throughput = 0;
            float throughputWithImportanceSampling = 0;
            float pdf = 0;
            float sampleNum = 0;

            while( sampleNum < IMPORTANCE_SAMPLE_NUM && throughputWithImportanceSampling == 0 )
            {
                float2 rnd = STL::Rng::GetFloat2( );

                if( payload.isDiffuse )
                {
                    float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay( rnd );
                    rayDirection = STL::Geometry::RotateVectorInverse( mLocalBasis, rayLocal );

                    throughput = 1.0; // = [ albedo / PI ] / STL::ImportanceSampling::Cosine::GetPDF( NoL );

                    float NoL = saturate( dot( payload.materialProps.N, rayDirection ) );
                    pdf = STL::ImportanceSampling::Cosine::GetPDF( NoL );
                }
                else
                {
                    float3 Hlocal = STL::ImportanceSampling::VNDF::GetRay( rnd, payload.materialProps.roughness, Vlocal, trimmingFactor );
                    float3 H = STL::Geometry::RotateVectorInverse( mLocalBasis, Hlocal );
                    rayDirection = reflect( payload.geometryProps.rayDirection, H );

                    VoH = abs( dot( -payload.geometryProps.rayDirection, H ) );

                    // It's a part of VNDF sampling - see http://jcgt.org/published/0007/04/01/paper.pdf ( paragraph "Usage in Monte Carlo renderer" )
                    float NoL = saturate( dot( payload.materialProps.N, rayDirection ) );
                    throughput = STL::BRDF::GeometryTerm_Smith( payload.materialProps.roughness, NoL );

                    float NoV = abs( dot( payload.materialProps.N, -payload.geometryProps.rayDirection ) );
                    float NoH = saturate( dot( payload.materialProps.N, H ) );
                    pdf = STL::ImportanceSampling::VNDF::GetPDF( NoV, NoH, payload.materialProps.roughness );
                }

                throughputWithImportanceSampling = throughput;
                if( gDisableShadowsAndEnableImportanceSampling )
                {
                    bool isMiss = CastVisibilityRay_AnyHit( payload.geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gLightTlas, throughput != 0.0 ? GEOMETRY_ONLY_EMISSIVE : 0, desc.rayFlags );
                    throughputWithImportanceSampling *= float( !isMiss );
                }

                sampleNum += 1.0;
            }

            throughput /= sampleNum;
            directionPdf = payload.bounceIndex == 1 ? float4( rayDirection, pdf ) : directionPdf;

            // Update BRDF
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( payload.materialProps.baseColor, payload.materialProps.metalness, albedo, Rf0 );

            float3 F = STL::BRDF::FresnelTerm_Schlick( Rf0, VoH );
            payload.BRDF *= payload.isDiffuse ? albedo : F;
            payload.BRDF *= throughput;

            // Abort if expected contribution of the current bounce is low
            if( STL::Color::Luminance( payload.BRDF ) < desc.threshold )
                break;
        }
        else
        {
            // Primary ray
            rayDirection = -GetViewVector( payload.geometryProps.X );
        }

        float metalnessAtOrigin = payload.materialProps.metalness;
        float diffProbAtOrigin = EstimateDiffuseProbability( payload.geometryProps, payload.materialProps );

        // Cast ray and update payload ( i.e. jump to next point )
        payload.geometryProps = CastRay( payload.geometryProps.GetXoffset( ), rayDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags, desc.useSimplifiedModel );
        payload.materialProps = GetMaterialProps( payload.geometryProps, desc.useSimplifiedModel );
        mipAndCone = GetConeAngleFromRoughness( payload.geometryProps.mip, payload.isDiffuse ? 1.0 : payload.materialProps.roughness );

        // Compute lighting
        float3 L = payload.materialProps.Ldirect;
        if( STL::Color::Luminance( L ) != 0 && !gDisableShadowsAndEnableImportanceSampling )
            L *= CastVisibilityRay_AnyHit( payload.geometryProps.GetXoffset( ), gSunDirection, 0.0, INF, mipAndCone, gWorldTlas, desc.instanceInclusionMask, desc.rayFlags );
        L += payload.materialProps.Lemi;

        // Reuse previous frame ( carefully treating specular )
        float3 prevLsum = GetRadianceFromPreviousFrame( payload.geometryProps, desc.pixelUv, payload.accumulatedPrevFrameWeight );

        float diffProbAtHit = EstimateDiffuseProbability( payload.geometryProps, payload.materialProps ); // TODO: better split previous frame data into diffuse and specular and apply the logic to specular only
        payload.accumulatedPrevFrameWeight *= lerp( diffProbAtOrigin, diffProbAtHit, metalnessAtOrigin );

        float l1 = STL::Color::Luminance( L );
        float l2 = STL::Color::Luminance( prevLsum );
        payload.accumulatedPrevFrameWeight *= l2 / ( l1 + l2 + 1e-6 );

        L = lerp( L, prevLsum, payload.accumulatedPrevFrameWeight );

        // Accumulate lighting
        L *= payload.BRDF;
        payload.Lsum += L;

        // Reduce contribution of next samples
        payload.BRDF *= 1.0 - payload.accumulatedPrevFrameWeight;

        // Accumulate path length
        float a = STL::Color::Luminance( L ) + 1e-6;
        float b = STL::Color::Luminance( payload.Lsum ) + 1e-6;
        float importance = a / b;
        payload.pathLength += NRD_GetCorrectedHitDist( payload.geometryProps.tmin, payload.bounceIndex, primaryHitRoughness, importance );

        // Estimate next event and go to next bounce
        payload.isDiffuse = IsNextEventDiffuse( payload.geometryProps, payload.materialProps );
        payload.bounceIndex++;
    }

    // Ambient estimation at the end of the path
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( payload.materialProps.baseColor, payload.materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( payload.materialProps.N, -payload.geometryProps.rayDirection ) );
    float3 F = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, payload.materialProps.roughness );

    float scale = lerp( 1.0, 1.5, payload.materialProps.metalness );
    float3 BRDF = albedo * ( 1 - F ) + F / scale;
    BRDF *= float( !payload.geometryProps.IsSky() );

    float occlusion = REBLUR_FrontEnd_GetNormHitDist( payload.geometryProps.tmin, 0.0, gDiffHitDistParams, 1.0 );
    occlusion = lerp( 1.0 / STL::Math::Pi( 1.0 ), 1.0, occlusion );
    occlusion *= exp2( AMBIENT_FADE * STL::Math::LengthSquared( payload.geometryProps.X - gCameraOrigin ) );

    payload.Lsum += desc.Lamb * payload.BRDF * BRDF * occlusion;

    return directionPdf;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BSDF
#define sampleNext2D(sg) STL::Rng::GetFloat2()



#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi



#define SpecularMaskingFunctionSmithGGXSeparable    0       ///< Used by UE4.
#define SpecularMaskingFunctionSmithGGXCorrelated   1       ///< Used by Frostbite. This is the more accurate form (default).

#ifndef SpecularMaskingFunction
#define SpecularMaskingFunction SpecularMaskingFunctionSmithGGXCorrelated
#endif


// BxDF.slang
// Enable support for delta reflection/transmission.
#define EnableDeltaBSDF         1

// Enable GGX sampling using the distribution of visible normals (VNDF) instead of classic NDF sampling.
// This should be the default as it has lower variance, disable for testing only.
// TODO: Make default when transmission with VNDF sampling is properly validated
#define EnableVNDFSampling      1

// Enable explicitly computing sampling weights using eval(wi, wo) / evalPdf(wi, wo).
// This is for testing only, as many terms of the equation cancel out allowing to save on computation.
#define ExplicitSampleWeights   0
static float kMinCosTheta = 1e-6f;


///** Flags representing the various lobes of a BxDF.
//*/
//// TODO: Specify the backing type when Slang issue has been resolved
//enum class LobeType // : uint32_t
//{
//    None = 0x00,
//
//    DiffuseReflection = 0x01,
//    SpecularReflection = 0x02,
//    DeltaReflection = 0x04,
//
//    DiffuseTransmission = 0x10,
//    SpecularTransmission = 0x20,
//    DeltaTransmission = 0x40,
//
//    Diffuse = 0x11,
//    Specular = 0x22,
//    Delta = 0x44,
//    NonDelta = 0x33,
//
//    Reflection = 0x0f,
//    Transmission = 0xf0,
//
//    NonDeltaReflection = 0x03,
//    NonDeltaTransmission = 0x30,
//
//    All = 0xff,
//};



/** Describes a BSDF sample.
*/
struct BSDFSample
{
    float3  wo;             ///< Sampled direction in world space (normalized).
    float   pdf;            ///< pdf with respect to solid angle for the sampled direction (wo).
    float3  weight;         ///< Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
    uint    lobe;           ///< Sampled lobe. This is a combination of LobeType flags (see LobeType.slang).

    //bool isLobe(LobeType type)
    //{
    //    return (lobe & uint(type)) != 0;
    //}
};



struct DiffuseReflectionLambert
{
#if 0
    //float3 albedo;  ///< Diffuse albedo.

    float3 eval(float3 wi, float3 wo, float3 n, float3 albedo)
    {
        float NoL = max(dot(n, wo), 0.0);
        return M_1_PI * albedo * NoL;       // local wo.z -> NoL 
    }

    bool sample(out float3 wo, out float pdf, out float3 weight, float3 wi, float3 n, float3 albedo)        // n: world normal
    {
        float2 rnd = STL::Rng::GetFloat2();
        float3x3 mLocalBasis = STL::Geometry::GetBasis(n);        
        float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);
        float3 rayDirection = STL::Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

        // output
        wo = rayDirection;
        pdf = M_1_PI * rayLocal.z;
        weight = albedo;

        return true;
    }

    float evalPdf(float3 wi, float3 wo, float n)
    {
        float NoL = max(dot(n, wo), 0.0);
        return M_1_PI * NoL;
    }
#endif



    float3 albedo;  ///< Diffuse albedo.

    float3 eval(float3 wi, float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta) 
            return 0.0;

        return M_1_PI * albedo * wo.z;
    }

    bool sample(float3 wi, out float3 wo, out float pdf, out float3 weight/*, out uint lobe, inout S sg*/)
    {
        float2 rnd = STL::Rng::GetFloat2();

        wo = STL::ImportanceSampling::Cosine::GetRay(rnd); 
        //lobe = (uint)LobeType::DiffuseReflection;

        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            weight = 0.0;
            return false;
        }

        pdf = M_1_PI * wo.z;
        weight = albedo;
        return true;
    }

    float evalPdf(float3 wi, float3 wo)
    {
        if (min(wi.z, wo.z) < kMinCosTheta)
            return 0.f;

        return M_1_PI * wo.z;
    }
};



//#include "Microfacet.hlsli" -- BEGIN


/** Evaluates the GGX (Trowbridge-Reitz) normal distribution function (D).

    Introduced by Trowbridge and Reitz, "Average irregularity representation of a rough surface for ray reflection", Journal of the Optical Society of America, vol. 65(5), 1975.
    See the correct normalization factor in Walter et al. https://dl.acm.org/citation.cfm?id=2383874
    We use the simpler, but equivalent expression in Eqn 19 from http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf

    For microfacet models, D is evaluated for the direction h to find the density of potentially active microfacets (those for which microfacet normal m = h).
    The 'alpha' parameter is the standard GGX width, e.g., it is the square of the linear roughness parameter in Disney's BRDF.
    Note there is a singularity (0/0 = NaN) at NdotH = 1 and alpha = 0, so alpha should be clamped to some epsilon.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and half vector, in positive hemisphere.
    \return D(h)
*/
float evalNdfGGX(float alpha, float cosTheta)
{
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * M_PI);
}


/** Evaluates the Smith masking function (G1) for the GGX normal distribution.
    See Eq 34 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

    The evaluated direction is assumed to be in the positive hemisphere relative the half vector.
    This is the case when both incident and outgoing direction are in the same hemisphere, but care should be taken with transmission.

    \param[in] alphaSqr Squared GGX width parameter.
    \param[in] cosTheta Dot product between shading normal and evaluated direction, in the positive hemisphere.
*/
float evalG1GGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 2 / (1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/** Evaluates the Smith lambda function for the GGX normal distribution.
    See Eq 72 in http://jcgt.org/published/0003/02/03/paper.pdf

    \param[in] alphaSqr Squared GGX width parameter.
    \param[in] cosTheta Dot product between shading normal and the evaluated direction, in the positive hemisphere.
*/
float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}


/** Evaluates the PDF for sampling the GGX normal distribution function using Walter et al. 2007's method.
    See https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and half vector, in positive hemisphere.
    \return D(h) * cosTheta
*/
float evalPdfGGX_NDF(float alpha, float cosTheta)
{
    return evalNdfGGX(alpha, cosTheta) * cosTheta;
}

/** Samples the GGX (Trowbridge-Reitz) normal distribution function (D) using Walter et al. 2007's method.
    Note that the sampled half vector may lie in the negative hemisphere. Such samples should be discarded.
    See Eqn 35 & 36 in https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    See Listing A.1 in https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] u Uniform random number (2D).
    \param[out] pdf Sampling probability.
    \return Sampled half vector in local space.
*/
float3 sampleGGX_NDF(float alpha, float2 u, out float pdf)
{
    float alphaSqr = alpha * alpha;
    float phi = u.y * (2 * M_PI);
    float tanThetaSqr = alphaSqr * u.x / (1 - u.x);
    float cosTheta = 1 / sqrt(1 + tanThetaSqr);
    float r = sqrt(max(1 - cosTheta * cosTheta, 0));

    pdf = evalPdfGGX_NDF(alpha, cosTheta);
    return float3(cos(phi) * r, sin(phi) * r, cosTheta);
}

/** Evaluates the PDF for sampling the GGX distribution of visible normals (VNDF).
    See http://jcgt.org/published/0007/04/01/paper.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] h Half vector in local space, in the positive hemisphere.
    \return D_V(h) = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
*/
float evalPdfGGX_VNDF(float alpha, float3 wi, float3 h)
{
    float G1 = evalG1GGX(alpha * alpha, wi.z);
    float D = evalNdfGGX(alpha, h.z);
    return G1 * D * max(0.f, dot(wi, h)) / wi.z;
}

/** Samples the GGX (Trowbridge-Reitz) using the distribution of visible normals (VNDF).
    The GGX VDNF yields significant variance reduction compared to sampling of the GGX NDF.
    See http://jcgt.org/published/0007/04/01/paper.pdf

    \param[in] alpha Isotropic GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] wi Incident direction in local space, in the positive hemisphere.
    \param[in] u Uniform random number (2D).
    \param[out] pdf Sampling probability.
    \return Sampled half vector in local space, in the positive hemisphere.
*/
float3 sampleGGX_VNDF(float alpha, float3 wi, float2 u, out float pdf)
{
    float alpha_x = alpha, alpha_y = alpha;

    // Transform the view vector to the hemisphere configuration.
    float3 Vh = normalize(float3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

    // Construct orthonormal basis (Vh,T1,T2).
    float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(float3(0, 0, 1), Vh)) : float3(1, 0, 0); // TODO: fp32 precision
    float3 T2 = cross(Vh, T1);

    // Parameterization of the projected area of the hemisphere.
    float r = sqrt(u.x);
    float phi = (2.f * M_PI) * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    // Reproject onto hemisphere.
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    float3 h = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.f, Nh.z)));

    pdf = evalPdfGGX_VNDF(alpha, wi, h);
    return h;
}



/** Evaluates the separable form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.
    See Eq 98 in http://jcgt.org/published/0003/02/03/paper.pdf

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(cosThetaI, cosThetaO)
*/
float evalMaskingSmithGGXSeparable(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / ((1 + lambdaI) * (1 + lambdaO));
}

/** Evaluates the height-correlated form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.
    See Eq 99 in http://jcgt.org/published/0003/02/03/paper.pdf

    Eric Heitz recommends using it in favor of the separable form as it is more accurate and of similar complexity.
    The function is only valid for cosThetaI > 0 and cosThetaO > 0  and should be clamped to 0 otherwise.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(cosThetaI, cosThetaO)
*/
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}

/** Approximate pre-integrated specular BRDF. The approximation assumes GGX VNDF and Schlick's approximation.
    See Eq 4 in [Ray Tracing Gems, Chapter 32]

    \param[in] specularReflectance Reflectance from a direction parallel to the normal.
    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and evaluated direction, in the positive hemisphere.
*/
float3 approxSpecularIntegralGGX(float3 specularReflectance, float alpha, float cosTheta)
{
    cosTheta = abs(cosTheta);

    float4 X;
    X.x = 1.f;
    X.y = cosTheta;
    X.z = cosTheta * cosTheta;
    X.w = cosTheta * X.z;

    float4 Y;
    Y.x = 1.f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    // Select coefficients based on BRDF version being in use (either separable or correlated G term)
#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
    float2x2 M1 = float2x2(
        0.99044f, -1.28514f,
        1.29678f, -0.755907f
        );

    float3x3 M2 = float3x3(
        1.0f, 2.92338f, 59.4188f,
        20.3225f, -27.0302f, 222.592f,
        121.563f, 626.13f, 316.627f
        );

    float2x2 M3 = float2x2(
        0.0365463f, 3.32707f,
        9.0632f, -9.04756f
        );

    float3x3 M4 = float3x3(
        1.0f, 3.59685f, -1.36772f,
        9.04401f, -16.3174f, 9.22949f,
        5.56589f, 19.7886f, -20.2123f
        );
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
    float2x2 M1 = float2x2(
        0.995367f, -1.38839f,
        -0.24751f, 1.97442f
        );

    float3x3 M2 = float3x3(
        1.0f, 2.68132f, 52.366f,
        16.0932f, -3.98452f, 59.3013f,
        -5.18731f, 255.259f, 2544.07f
        );

    float2x2 M3 = float2x2(
        -0.0564526f, 3.82901f,
        16.91f, -11.0303f
        );

    float3x3 M4 = float3x3(
        1.0f, 4.11118f, -1.37886f,
        19.3254f, -28.9947f, 16.9514f,
        0.545386f, 96.0994f, -79.4492f
        );
#endif

    float bias = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));

    // This is a hack for specular reflectance of 0
    float specularReflectanceLuma = dot(specularReflectance, (1.f / 3.f));
    bias *= saturate(specularReflectanceLuma * 50.0f);

    return mad(specularReflectance, max(0.0, scale), max(0.0, bias));
}


//#include "Microfacet.hlsli" -- END



/** Evaluates the Fresnel term using Schlick's approximation.
    Introduced in http://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf

    The Fresnel term equals f0 at normal incidence, and approaches f90=1.0 at 90 degrees.
    The formulation below is generalized to allow both f0 and f90 to be specified.

    \param[in] f0 Specular reflectance at normal incidence (0 degrees).
    \param[in] f90 Reflectance at orthogonal incidence (90 degrees), which should be 1.0 for specular surface reflection.
    \param[in] cosTheta Cosine of angle between microfacet normal and incident direction (LdotH).
    \return Fresnel term.
*/
float3 evalFresnelSchlick(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

/** Specular reflection using microfacets.
*/
struct SpecularReflectionMicrofacet
{
    float3 albedo;      ///< Specular albedo.
    float alpha;        ///< GGX width parameter.
    uint activeLobes;   ///< BSDF lobes to include for sampling and evaluation. See LobeType.slang.


    bool sample(float3 wi, out float3 wo, out float pdf, out float3 weight/*, out uint lobe, inout S sg*/)
    {
        // Default initialization to avoid divergence at returns.
        wo = 0;
        weight = 0;
        pdf = 0.f;
        //lobe = (uint)LobeType::SpecularReflection;

        if (wi.z < kMinCosTheta) 
            return false;

//#if EnableDeltaBSDF
        // Handle delta reflection.
//        if (alpha == 0.f)
        {
            //if (!hasLobe(LobeType::DeltaReflection)) 
            //    return false;

            wo = float3(-wi.x, -wi.y, wi.z);
            pdf = 0.f;
            weight = evalFresnelSchlick(albedo, 1.f, wi.z);
            //lobe = (uint)LobeType::DeltaReflection;
            return true;
        }
//#endif

#if 0


        //if (!hasLobe(LobeType::SpecularReflection)) 
        //    return false;

        // Sample the GGX distribution to find a microfacet normal (half vector).
#if EnableVNDFSampling
        float3 h = sampleGGX_VNDF(alpha, wi, sampleNext2D(sg), pdf);    // pdf = G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
#else
        float3 h = sampleGGX_NDF(alpha, sampleNext2D(sg), pdf);         // pdf = D(h) * h.z
#endif

        // Reflect the incident direction to find the outgoing direction.
        float wiDotH = dot(wi, h);
        wo = 2.f * wiDotH * h - wi;
        if (wo.z < kMinCosTheta) 
            return false;

#if ExplicitSampleWeights
        // For testing.
        pdf = evalPdf(wi, wo);
        weight = eval(wi, wo) / pdf;
        //lobe = (uint)LobeType::SpecularReflection;
        return true;
#endif

#if SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXSeparable
        float G = evalMaskingSmithGGXSeparable(alpha, wi.z, wo.z);
        float GOverG1wo = evalG1GGX(alpha * alpha, wo.z);
#elif SpecularMaskingFunction == SpecularMaskingFunctionSmithGGXCorrelated
        float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        float GOverG1wo = G * (1.f + evalLambdaGGX(alpha * alpha, wi.z));
#endif
        float3 F = evalFresnelSchlick(albedo, 1.f, wiDotH);

        pdf /= (4.f * wiDotH); // Jacobian of the reflection operator.
#if EnableVNDFSampling
        weight = F * GOverG1wo;
#else
        weight = F * G * wiDotH / (wi.z * h.z);
#endif
        //lobe = (uint)LobeType::SpecularReflection;
        return true;


#endif
    }


};







struct StandardBSDF
{
    bool sample(out BSDFSample result, float3 wi, float3 n, float3 diffuseColor, float3 specularColor, float roughness)
    {
#if 0
        if (!useImportanceSampling) 
            return sampleReference(sd, sg, result);

        float3 wiLocal = sd.toLocal(sd.V);
        float3 woLocal = {};

        FalcorBSDF bsdf = FalcorBSDF(sd, data);

        bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, sg);
        result.wo = sd.fromLocal(woLocal);

        return valid;
#endif

#if 0
        //if (!useImportanceSampling) 
        //    return sampleReference(sd, sg, result);

        float3x3 mLocalBasis = STL::Geometry::GetBasis(n);
        float3 wiLocal = STL::Geometry::RotateVector(mLocalBasis, wi);                                       // world to local
        float3 woLocal = 0.0;

        //FalcorBSDF bsdf = FalcorBSDF(sd, data);
        DiffuseReflectionLambert lambert;
        lambert.albedo = albedo;

        bool valid = lambert.sample(wiLocal, woLocal, result.pdf, result.weight);
        //bool valid = bsdf.sample(wiLocal, woLocal, result.pdf, result.weight, result.lobe, sg);
        result.wo = STL::Geometry::RotateVectorInverse(mLocalBasis, woLocal);                               // local to world


        return valid;
#endif  



#if 1
        //if (!useImportanceSampling) 
        //    return sampleReference(sd, sg, result);

        float3x3 mLocalBasis = STL::Geometry::GetBasis(n);
        float3 wiLocal = STL::Geometry::RotateVector(mLocalBasis, wi);                                       // world to local
        float3 woLocal = 0.0;

        
        SpecularReflectionMicrofacet specularRM;
        float3 albedo = specularColor;
        float alpha = roughness;        

        bool valid = specularRM.sample(wiLocal, woLocal, result.pdf, result.weight);
        result.wo = STL::Geometry::RotateVectorInverse(mLocalBasis, woLocal);                               // local to world

        return valid;
#endif  
    }
};











struct RayDesc_NRD
{
    float3 origin;
    float3 direction;
    float tMin;
    float tMax;

    float3 throughput;
    float pdf;

    uint bounceIndex;
};





[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
#if 1
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

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ] );
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gIsOrtho );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = GetViewVector( X );
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[ pixelPos ];

    //float3 Xoffset = _GetXoffset( X, N );
    //Xoffset += V * ( 0.0003 + abs( viewZ ) * 0.00005 );

    //float NoV0 = abs( dot( N, V ) );
    //Xoffset += N * STL::BRDF::Pow5( NoV0 ) * ( 0.003 + abs( viewZ ) * 0.0005 );


    RayDesc_NRD ray;
    ray.origin = X;
    ray.direction = -V;
    ray.tMin = 0.0;
    ray.tMax = INF;
    ray.throughput = 1.0;
    ray.pdf = 1.0;
    ray.bounceIndex = 0;






   

    //float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross( Rf00, NoV0, materialProps0.roughness );
    //envBRDF0 = max( envBRDF0, 0.001 );

    // Ambient
    //float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    //Lamb *= gAmbient;

    // Secondary rays
    STL::Rng::Initialize( pixelPos, gFrameIndex );

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
    MaterialProps materialProps0 = (MaterialProps)0;
    // prepare for primary.
    {
        hitT = 0.0;
        mip = mip0* mip0* MAX_MIP_LEVEL;
        ray.bounceIndex += 1;

        materialProps0.N = N;
        materialProps0.baseColor = baseColorMetalness.xyz;
        materialProps0.roughness = normalAndRoughness.w;
        materialProps0.metalness = baseColorMetalness.w;  

        // sample light
        // primary sample light is done demodulated.
    }

    float primaryHitRoughness = materialProps0.roughness;



    float3 irradiance = 0;

    float hitTForNRD = 0.0;

    [loop]
    for (uint i = 1; i < gBounceNum; i++)
    {       
        bool bIsDiffuse = true;

        // sample material 
        BSDFSample result;   
        //{
            float3 albedo, Rf0;
            STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps0.baseColor, materialProps0.metalness, albedo, Rf0);


            StandardBSDF standerdBSDF;
            standerdBSDF.sample(result, -ray.direction, materialProps0.N, albedo, Rf0, materialProps0.roughness);
            //if (materialPdf == 0.0)
            //    break;
       // }
        


        // update ray
        ray.origin = _GetXoffset(ray.origin + ray.direction * hitT, materialProps0.N);
        ray.direction = result.wo;
        ray.throughput *= result.weight;
        ray.pdf *= result.pdf;

        // trace and decode.          
        float2 mipAndCone = GetConeAngleFromRoughness(mip, 1.0);
        GeometryProps geometryProps0 = CastRay(ray.origin, ray.direction, ray.tMin, ray.tMax, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0, bIsDiffuse);
        hitT = geometryProps0.tmin;
        mip = geometryProps0.mip;
        materialProps0 = GetMaterialProps(geometryProps0, bIsDiffuse);



        

        // calculate contribution
        float3 currentContribution = 0.0;
        if (hitT == INF)
        {
            float3 envRadiance = GetSkyIntensity(ray.direction, gSunDirection, gTanSunAngularRadius);
            currentContribution = envRadiance * ray.throughput;
            //break;
        }  

        irradiance += currentContribution;


        // Accumulate path length
        float a = STL::Color::Luminance(currentContribution) + 1e-6;
        float b = STL::Color::Luminance(irradiance) + 1e-6;
        float importance = a / b;
        hitTForNRD += NRD_GetCorrectedHitDist(hitT, ray.bounceIndex, primaryHitRoughness, importance);


        ray.bounceIndex += 1;
    }

    // output nrd
    {
        float pathLengthMod = NRD_GetCorrectedHitDist(hitTForNRD, 1, materialProps0.roughness, 1.0);
        float normDist = REBLUR_FrontEnd_GetNormHitDist(pathLengthMod, viewZ, gDiffHitDistParams, 1.0);
        float4 nrdData = REBLUR_FrontEnd_PackRadianceAndHitDist(irradiance, normDist, USE_SANITIZATION);
        diffIndirect += nrdData;
    }


    {
        gOut_Diff[ outPixelPos ] = diffIndirect;
        gOut_DiffDirectionPdf[ outPixelPos ] = diffDirectionPdf;

        gOut_Spec[ outPixelPos ] = specIndirect;
        gOut_SpecDirectionPdf[ outPixelPos ] = specDirectionPdf;
    }
#endif



#if 0
    // Do not generate NANs for unused threads
    float2 rectSize = gRectSize;
    if (pixelPos.x >= rectSize.x || pixelPos.y >= rectSize.y)
        return;

    uint2 outPixelPos = pixelPos;

    float2 pixelUv = float2(pixelPos + 0.5) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Early out
    float viewZ = gIn_ViewZ[pixelPos];
    if (abs(viewZ) == INF)
    {
        gOut_Diff[outPixelPos] = 0;
        gOut_Spec[outPixelPos] = 0;

        return;
    }

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness(gIn_Normal_Roughness[pixelPos]);
    float4 baseColorMetalness = gIn_BaseColor_Metalness[pixelPos];

    float3 Xv = STL::Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, viewZ, gIsOrtho);
    float3 X = STL::Geometry::AffineTransform(gViewToWorld, Xv);
    float3 V = GetViewVector(X);
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[pixelPos];

    float3 Xoffset = _GetXoffset(X, N);
    Xoffset += V * (0.0003 + abs(viewZ) * 0.00005);

    float NoV0 = abs(dot(N, V));
    Xoffset += N * STL::BRDF::Pow5(NoV0) * (0.003 + abs(viewZ) * 0.0005);

    GeometryProps geometryProps0 = (GeometryProps)0;
    geometryProps0.X = Xoffset;
    geometryProps0.rayDirection = -V;
    geometryProps0.N = N;
    geometryProps0.mip = mip0 * mip0 * MAX_MIP_LEVEL;

    MaterialProps materialProps0 = (MaterialProps)0;
    materialProps0.N = N;
    materialProps0.baseColor = baseColorMetalness.xyz;
    materialProps0.roughness = normalAndRoughness.w;
    materialProps0.metalness = baseColorMetalness.w;

    // Material de-modulation
    float3 albedo0, Rf00;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0(baseColorMetalness.xyz, baseColorMetalness.w, albedo0, Rf00);

    albedo0 = max(albedo0, 0.001);

    float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross(Rf00, NoV0, materialProps0.roughness);
    envBRDF0 = max(envBRDF0, 0.001);

    // Ambient
    float3 Lamb = gIn_Ambient.SampleLevel(gLinearSampler, float2(0.5, 0.5), 0);
    Lamb *= gAmbient;

    // Secondary rays
    STL::Rng::Initialize(pixelPos, gFrameIndex);

    float4 diffIndirect = 0;
    float4 diffDirectionPdf = 0;
    float diffTotalWeight = 1e-6;

    float4 specIndirect = 0;
    float4 specDirectionPdf = 0;
    float specTotalWeight = 1e-6;

    uint sampleNum = gSampleNum << (gTracingMode == RESOLUTION_HALF ? 0 : 1);
    uint checkerboard = STL::Sequence::CheckerBoard(pixelPos, gFrameIndex) != 0;

    float3x3 mLocalBasis = STL::Geometry::GetBasis(materialProps0.N);
    float2 mipAndCone = GetConeAngleFromRoughness(geometryProps0.mip, materialProps0.roughness);
    {
        float2 rnd = STL::Rng::GetFloat2();

        float3 rayLocal = STL::ImportanceSampling::Cosine::GetRay(rnd);
        float3 rayDirection = STL::Geometry::RotateVectorInverse(mLocalBasis, rayLocal);


        float3 result = 0.0;

        GeometryProps gpResult = CastRay(Xoffset, rayDirection, 0.0, INF, mipAndCone, gWorldTlas, GEOMETRY_IGNORE_TRANSPARENT, 0, 1);                    //desc.useSimplifiedModel
        if (gpResult.IsSky())
        {
            result = GetSkyIntensity(rayDirection, gSunDirection, gTanSunAngularRadius);
        }

        float pathLengthMod = NRD_GetCorrectedHitDist(gpResult.tmin, 1, materialProps0.roughness, 1.0);

        float normDist = REBLUR_FrontEnd_GetNormHitDist(pathLengthMod, viewZ, gDiffHitDistParams, 1.0);
        float4 nrdData = REBLUR_FrontEnd_PackRadianceAndHitDist(result, normDist, USE_SANITIZATION);

        diffIndirect += nrdData;
    }






    {
        gOut_Diff[outPixelPos] = diffIndirect;
        gOut_DiffDirectionPdf[outPixelPos] = diffDirectionPdf;

        gOut_Spec[outPixelPos] = specIndirect;
        gOut_SpecDirectionPdf[outPixelPos] = specDirectionPdf;
    }
#endif


#if 0
    // Do not generate NANs for unused threads
    float2 rectSize = gRectSize * ( gTracingMode == RESOLUTION_QUARTER ? 0.5 : 1.0 );
    if( pixelPos.x >= rectSize.x || pixelPos.y >= rectSize.y )
        return;

    uint2 outPixelPos = pixelPos;

    [branch]
    if( gTracingMode == RESOLUTION_QUARTER )
    {
        pixelPos = ( pixelPos << 1 ) + uint2( gFrameIndex & 0x1, ( gFrameIndex >> 1 ) & 0x1 );

        gOut_Downsampled_ViewZ[ outPixelPos ] = gIn_ViewZ[ pixelPos ];
        gOut_Downsampled_Motion[ outPixelPos ] = gIn_Motion[ pixelPos ];
        gOut_Downsampled_Normal_Roughness[ outPixelPos ] = gIn_Normal_Roughness[ pixelPos ];
    }

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

    // G-buffer
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPos ] );
    float4 baseColorMetalness = gIn_BaseColor_Metalness[ pixelPos ];

    float3 Xv = STL::Geometry::ReconstructViewPosition( sampleUv, gCameraFrustum, viewZ, gIsOrtho );
    float3 X = STL::Geometry::AffineTransform( gViewToWorld, Xv );
    float3 V = GetViewVector( X );
    float3 N = normalAndRoughness.xyz;
    float mip0 = gIn_PrimaryMip[ pixelPos ];

    float3 Xoffset = _GetXoffset( X, N );
    Xoffset += V * ( 0.0003 + abs( viewZ ) * 0.00005 );

    float NoV0 = abs( dot( N, V ) );
    Xoffset += N * STL::BRDF::Pow5( NoV0 ) * ( 0.003 + abs( viewZ ) * 0.0005 );

    GeometryProps geometryProps0 = ( GeometryProps )0;
    geometryProps0.X = Xoffset;
    geometryProps0.rayDirection = -V;
    geometryProps0.N = N;
    geometryProps0.mip = mip0 * mip0 * MAX_MIP_LEVEL;

    MaterialProps materialProps0 = ( MaterialProps )0;
    materialProps0.N = N;
    materialProps0.baseColor = baseColorMetalness.xyz;
    materialProps0.roughness = normalAndRoughness.w;
    materialProps0.metalness = baseColorMetalness.w;

    // Material de-modulation
    float3 albedo0, Rf00;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColorMetalness.xyz, baseColorMetalness.w, albedo0, Rf00 );

    albedo0 = max( albedo0, 0.001 );

    float3 envBRDF0 = STL::BRDF::EnvironmentTerm_Ross( Rf00, NoV0, materialProps0.roughness );
    envBRDF0 = max( envBRDF0, 0.001 );

    // Ambient
    float3 Lamb = gIn_Ambient.SampleLevel( gLinearSampler, float2( 0.5, 0.5 ), 0 );
    Lamb *= gAmbient;

    // Secondary rays
    STL::Rng::Initialize( pixelPos, gFrameIndex );

    float4 diffIndirect = 0;
    float4 diffDirectionPdf = 0;
    float diffTotalWeight = 1e-6;

    float4 specIndirect = 0;
    float4 specDirectionPdf = 0;
    float specTotalWeight = 1e-6;

    uint sampleNum = gSampleNum << ( gTracingMode == RESOLUTION_HALF ? 0 : 1 );
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex ) != 0;

    TracePathDesc tracePathDesc = ( TracePathDesc )0;
    tracePathDesc.pixelUv = pixelUv;
    tracePathDesc.bounceNum = gBounceNum; // TODO: adjust by roughness
    tracePathDesc.instanceInclusionMask = GEOMETRY_IGNORE_TRANSPARENT;
    tracePathDesc.rayFlags = 0;
    tracePathDesc.threshold = BRDF_ENERGY_THRESHOLD;

    for( uint i = 0; i < sampleNum; i++ )
    {
        bool isDiffuse = gTracingMode == RESOLUTION_HALF ? checkerboard : ( i < gSampleNum );

        // Trace
        tracePathDesc.useSimplifiedModel = isDiffuse; // TODO: adjust by roughness
        tracePathDesc.Lamb = Lamb * float( !isDiffuse );

        TracePathPayload tracePathPayload = ( TracePathPayload )0;
        tracePathPayload.BRDF = 1.0;
        tracePathPayload.Lsum = 0.0;
        tracePathPayload.accumulatedPrevFrameWeight = GetBasePrevFrameWeight( );
        tracePathPayload.pathLength = 0.0; // exclude primary ray length
        tracePathPayload.bounceIndex = 1; // starting from primary ray hit
        tracePathPayload.isDiffuse = isDiffuse;
        tracePathPayload.geometryProps = geometryProps0;
        tracePathPayload.materialProps = materialProps0;

        float4 directionPdf = TracePath( tracePathDesc, tracePathPayload, isDiffuse ? 1.0 : materialProps0.roughness );

        // De-modulate materials for denoising
        tracePathPayload.Lsum /= isDiffuse ? albedo0 : envBRDF0;

        // Convert for NRD
        directionPdf = NRD_FrontEnd_PackDirectionAndPdf( directionPdf.xyz, directionPdf.w );

        float normDist = REBLUR_FrontEnd_GetNormHitDist( tracePathPayload.pathLength, viewZ, isDiffuse ? gDiffHitDistParams : gSpecHitDistParams, isDiffuse ? 1.0 : materialProps0.roughness );
        float4 nrdData = REBLUR_FrontEnd_PackRadianceAndHitDist( tracePathPayload.Lsum, normDist, USE_SANITIZATION );
        if( gDenoiserType != REBLUR )
            nrdData = RELAX_FrontEnd_PackRadianceAndHitDist( tracePathPayload.Lsum, tracePathPayload.pathLength, USE_SANITIZATION );

        // Debug
        if( gOnScreen == SHOW_MIP_SPECULAR )
        {
            float mipNorm = STL::Math::Sqrt01( tracePathPayload.geometryProps.mip / MAX_MIP_LEVEL );
            nrdData.xyz = STL::Color::ColorizeZucconi( mipNorm );
        }

        // Accumulate
        float sampleWeight = NRD_GetSampleWeight( tracePathPayload.Lsum, USE_SANITIZATION );
        nrdData *= sampleWeight;
        directionPdf *= sampleWeight;

        diffIndirect += nrdData * float( isDiffuse );
        diffDirectionPdf += directionPdf * float( isDiffuse );
        diffTotalWeight += sampleWeight * float( isDiffuse );

        specIndirect += nrdData * float( !isDiffuse );
        specDirectionPdf += directionPdf * float( !isDiffuse );
        specTotalWeight += sampleWeight * float( !isDiffuse );
    }

    diffIndirect /= diffTotalWeight;
    diffDirectionPdf /= diffTotalWeight;

    specIndirect /= specTotalWeight;
    specDirectionPdf /= specTotalWeight;

    // Output
    [flatten]
    if( gOcclusionOnly )
    {
        diffIndirect = diffIndirect.wwww;
        specIndirect = specIndirect.wwww;
    }

    if( gTracingMode == RESOLUTION_HALF )
    {
        pixelPos.x >>= 1;

        if( checkerboard )
        {
            gOut_Diff[ pixelPos ] = diffIndirect;
            gOut_DiffDirectionPdf[ pixelPos ] = diffDirectionPdf;
        }
        else
        {
            gOut_Spec[ pixelPos ] = specIndirect;
            gOut_SpecDirectionPdf[ pixelPos ] = specDirectionPdf;
        }
    }
    else
    {
        gOut_Diff[ outPixelPos ] = diffIndirect;
        gOut_DiffDirectionPdf[ outPixelPos ] = diffDirectionPdf;

        gOut_Spec[ outPixelPos ] = specIndirect;
        gOut_SpecDirectionPdf[ outPixelPos ] = specDirectionPdf;
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
