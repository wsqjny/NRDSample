/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

// Inputs
NRI_RESOURCE(Texture2D<float4>,   sampleNRDRadiance,    t, 0, 1);   ///< Input per-sample NRD radiance data. Only valid if kOutputNRDData == true.
NRI_RESOURCE(Texture2D<float>,    sampleNRDHitDist,     t, 1, 1);   ///< Input per-sample NRD hit distance data. Only valid if kOutputNRDData == true.
NRI_RESOURCE(Texture2D<float4>,   sampleNRDEmission,    t, 2, 1);   ///< Input per-sample NRD emission data. Only valid if kOutputNRDData == true.
NRI_RESOURCE(Texture2D<float4>,   sampleNRDReflectance, t, 3, 1);  ///< Input per-sample NRD reflectance data. Only valid if kOutputNRDData == true.

// Outputs
NRI_RESOURCE(RWTexture2D<float4>, outputNRDDiffuseRadianceHitDist,          u, 4, 1);///< Output resolved diffuse radiance in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
NRI_RESOURCE(RWTexture2D<float4>, outputNRDSpecularRadianceHitDist,         u, 5, 1);///< Output resolved specular radiance in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
NRI_RESOURCE(RWTexture2D<float4>, outputNRDDeltaReflectionRadianceHitDist,  u, 6, 1);///< Output resolved delta reflection radiance in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
NRI_RESOURCE(RWTexture2D<float4>, outputNRDDeltaTransmissionRadianceHitDist,u, 7, 1);///< Output resolved delta transmission radiance in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.



//--------------------------------------------------------------------------- Pack Radiance Begin----------------------------------------------------------
// TODO: Falcor move this to a saperate pass.
inline float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

void clampRadiance(inout float3 diffuseRadiance, inout float3 specularRadiance)
{
    static const float kEpsilon = 1e-6f;
    static const float gMaxIntensity = 1000.f;

    float lDiff = luminance(diffuseRadiance);
    if (lDiff > kEpsilon)
    {
        diffuseRadiance *= min(gMaxIntensity / lDiff, 1.f);
    }

    float lSpec = luminance(specularRadiance);
    if (lSpec > kEpsilon)
    {
        specularRadiance *= min(gMaxIntensity / lSpec, 1.f);
    }
}

void PackRadiance(uint2 pixel)
{
    float4 diffuseRadianceHitDist               = outputNRDDiffuseRadianceHitDist[pixel];
    float4 specularRadianceHitDist              = outputNRDSpecularRadianceHitDist[pixel];
    float4 deltaReflectionRadianceHitDist       = outputNRDDeltaReflectionRadianceHitDist[pixel];
    float4 deltaTransmissionRadianceHitDist     = outputNRDDeltaTransmissionRadianceHitDist[pixel];
    
    clampRadiance(diffuseRadianceHitDist.rgb, specularRadianceHitDist.rgb);
    clampRadiance(deltaReflectionRadianceHitDist.rgb, deltaTransmissionRadianceHitDist.rgb);

    // Diffuse/Specular maybe REBLUR or RELAX. only for relax now.    
    if (gDenoiserType != REBLUR)    // RELAX
    {
        diffuseRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);
        specularRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);

        outputNRDDiffuseRadianceHitDist[pixel] = diffuseRadianceHitDist;
        outputNRDSpecularRadianceHitDist[pixel] = specularRadianceHitDist;
    }
    else
    {
#if 0
        float viewZ = gViewZ[ipos];
        float linearRoughness = gNormalRoughness[ipos].z;

        diffuseRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(diffuseRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
        REBLUR_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);

        specularRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(specularRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
        REBLUR_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);
#endif
    }

    // Delta Reflection/Transmission is RELAX_DIFFUSE
    {
        deltaReflectionRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(deltaReflectionRadianceHitDist.rgb, deltaReflectionRadianceHitDist.a);
        deltaTransmissionRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(deltaTransmissionRadianceHitDist.rgb, deltaTransmissionRadianceHitDist.a);

        outputNRDDeltaReflectionRadianceHitDist[pixel] = deltaReflectionRadianceHitDist;
        outputNRDDeltaTransmissionRadianceHitDist[pixel]= deltaTransmissionRadianceHitDist;
    }   
}
//--------------------------------------------------------------------------- Pack Radiance End--------------------------------------------------------------




[numthreads( 16, 16, 1)]
void main( int2 pixel : SV_DispatchThreadId )
{
    // Do not generate NANs for unused threads
    if(pixel.x >= gRectSize.x || pixel.y >= gRectSize.y )
        return;

    // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
    // const uint offset = params.getSampleOffset(pixel, sampleOffset);

    // Determine number of samples at the current pixel.
    // This is either a fixed number or loaded from the sample count texture.
    // TODO: We may want to use a nearest sampler to allow the texture to be of arbitrary dimension.
    const uint spp = 1;// kSamplesPerPixel > 0 ? kSamplesPerPixel : min(sampleCount[pixel], kMaxSamplesPerPixel);
    const float invSpp = spp > 0 ? 1.f / spp : 1.f; // Setting invSpp to 1.0 if we have no samples to avoid NaNs below.



    // NRD data is always written per-sample and needs to be resolved.
    //if (kOutputNRDData)
    {
        float3 diffuseRadiance = 0.f;
        float3 specularRadiance = 0.f;
        float3 deltaReflectionRadiance = 0.f;
        float3 deltaTransmissionRadiance = 0.f;
        float3 residualRadiance = 0.f;
        float hitDist = 0.f;

        //for (uint sampleIdx = 0; sampleIdx < spp; sampleIdx++)
        {
            const uint2 idx = pixel;// offset + sampleIdx;

            //const NRDRadiance radianceData = sampleNRDRadiance[idx];
            //const NRDPathType pathType = radianceData.getPathType();
            //const float3 radiance = radianceData.getRadiance();
            float4 radianceData = sampleNRDRadiance[idx];
            float  pathType = radianceData.a;
            float3 radiance = radianceData.rgb;

            float3 reflectance = sampleNRDReflectance[idx].rgb;
            float3 emission = sampleNRDEmission[idx].rgb;
            float3 demodulatedRadiance = max(0.f, (radiance - emission)) / reflectance;

#if 0
            switch (pathType)
            {
            case NRDPathType::Diffuse:
                diffuseRadiance += demodulatedRadiance;
                break;
            case NRDPathType::Specular:
                specularRadiance += demodulatedRadiance;
                break;
            case NRDPathType::DeltaReflection:
                deltaReflectionRadiance += demodulatedRadiance;
                break;
            case NRDPathType::DeltaTransmission:
                deltaTransmissionRadiance += demodulatedRadiance;
                break;
            default:
                // Do not demodulate residual.
                residualRadiance += radiance;
                break;
            }
#endif

            if (pathType < 0.05f)
            {
                diffuseRadiance += demodulatedRadiance;
            }
            else if (pathType < 0.15)
            {
                specularRadiance += demodulatedRadiance;
            }
            else if(pathType < 0.25)
            {
                deltaReflectionRadiance += demodulatedRadiance;
            }
            else if(pathType < 0.35)
            {
                deltaTransmissionRadiance += demodulatedRadiance;
            }
            else
            {
                residualRadiance += radiance;
            }     

            hitDist += sampleNRDHitDist[idx];
        }

        outputNRDDiffuseRadianceHitDist[pixel] = float4(invSpp * diffuseRadiance, invSpp * hitDist);
        outputNRDSpecularRadianceHitDist[pixel] = float4(invSpp * specularRadiance, invSpp * hitDist);
        outputNRDDeltaReflectionRadianceHitDist[pixel] = float4(invSpp * deltaReflectionRadiance, 0.f);
        outputNRDDeltaTransmissionRadianceHitDist[pixel] = float4(invSpp * deltaTransmissionRadiance, 0.f);
        //outputNRDResidualRadianceHitDist[pixel] = float4(invSpp * residualRadiance, hitDist);


        //-------------------- Pack Radiance
        PackRadiance(pixel);
    }
}