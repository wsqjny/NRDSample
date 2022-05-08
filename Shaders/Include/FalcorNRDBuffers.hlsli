#pragma once


struct NRDBuffers
{
    //RWStructuredBuffer<NRDRadiance> sampleRadiance;             ///< Output per-sample NRD radiance data. Only valid if kOutputNRDData == true.
    //RWStructuredBuffer<float>       sampleHitDist;              ///< Output per-sample NRD hit distance data. Only valid if kOutputNRDData == true.
    //RWStructuredBuffer<float4>      sampleEmission;             ///< Output per-sample NRD emission data. Only valid if kOutputNRDData == true.
    //RWStructuredBuffer<float4>      sampleReflectance;          ///< Output per-sample NRD reflectance data. Only valid if kOutputNRDData == true.

    //RWTexture2D<float4> primaryHitEmission;                     ///< Output per-pixel primary hit emission. Only valid if kOutputNRDData == true.
    //RWTexture2D<float4> primaryHitDiffuseReflectance;           ///< Output per-pixel primary hit diffuse reflectance. Only valid if kOutputNRDData == true.
    //RWTexture2D<float4> primaryHitSpecularReflectance;          ///< Output per-pixel primary hit specular reflectance. Only valid if kOutputNRDData == true.

    //RWTexture2D<float4> deltaReflectionReflectance;             ///< Output per-pixel delta reflection reflectance. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float4> deltaReflectionEmission;                ///< Output per-pixel delta reflection emission. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float4> deltaReflectionNormWRoughMaterialID;    ///< Output per-pixel delta reflection world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float>  deltaReflectionPathLength;              ///< Output per-pixel delta reflection path length. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float>  deltaReflectionHitDist;                 ///< Output per-pixel delta reflection hit distance. Only valid if kOutputNRDAdditionalData == true.

    //RWTexture2D<float4> deltaTransmissionReflectance;           ///< Output per-pixel delta transmission reflectance. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float4> deltaTransmissionEmission;              ///< Output per-pixel delta transmission emission. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float4> deltaTransmissionNormWRoughMaterialID;  ///< Output per-pixel delta transmission world normal, roughness, and material ID. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float>  deltaTransmissionPathLength;            ///< Output per-pixel delta transmission path length. Only valid if kOutputNRDAdditionalData == true.
    //RWTexture2D<float4> deltaTransmissionPosW;                  ///< Output per-pixel delta transmission world position. Only valid if kOutputNRDAdditionalData == true.
};

// Sample
NRI_RESOURCE(RWTexture2D<float4>,   gOut_SampleRadiance,                            u, 7, 1);
NRI_RESOURCE(RWTexture2D<float>,    gOut_SampleHitDist,                             u, 8, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_SampleEmission,                            u, 9, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_SampleReflectance,                         u, 10, 1);

// Primary
NRI_RESOURCE(RWTexture2D<float4>,   gOut_PrimaryHitEmission,                        u, 11, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_PrimaryHitDiffuseReflectance,              u, 12, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_PrimaryHitSpecularReflectance,             u, 13, 1);

// DeltaReflection
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaReflectionReflectance,                u, 14, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaReflectionEmission,                   u, 15, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaReflectionNormWRoughMaterialID,       u, 16, 1);
NRI_RESOURCE(RWTexture2D<float>,    gOut_DeltaReflectionPathLength,                 u, 17, 1);
NRI_RESOURCE(RWTexture2D<float>,    gOut_DeltaReflectionHitDist,                    u, 18, 1);

// DeltaTransmission
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaTransmissionReflectance,              u, 19, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaTransmissionEmission,                 u, 20, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaTransmissionNormWRoughMaterialID,     u, 21, 1);
NRI_RESOURCE(RWTexture2D<float>,    gOut_DeltaTransmissionPathLength,               u, 22, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gOut_DeltaTransmissionPosW,                     u, 23, 1);

// Outputs
NRI_RESOURCE(RWTexture2D<float4>,   outputNRDDiffuseRadianceHitDist,                u, 24, 1);
NRI_RESOURCE(RWTexture2D<float4>,   outputNRDSpecularRadianceHitDist,               u, 25, 1);
NRI_RESOURCE(RWTexture2D<float4>,   outputNRDDeltaReflectionRadianceHitDist,        u, 26, 1);
NRI_RESOURCE(RWTexture2D<float4>,   outputNRDDeltaTransmissionRadianceHitDist,      u, 27, 1);