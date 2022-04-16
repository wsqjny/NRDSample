/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
//import Rendering.Materials.IBSDF;
//import Rendering.Materials.Microfacet;
//import RenderPasses.Shared.Denoising.NRDBuffers;
//import Scene.ShadingData;
//import PathState;

#pragma once



/** Get material reflectance based on the metallic value.
*/
float3 getMaterialReflectanceForDeltaPaths(const bool hasDeltaLobes, const ShadingData sd, const BSDFProperties bsdfProperties, const FalcorPayload falcorPayload)
{
    //if (materialType == MaterialType::Standard)
    //{
        //const BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
    const float metallic = falcorPayload.metallic;// md.specular.b; // Blue component stores metallic in MetalRough mode.

    if (metallic == 0.f)
    {
        const float3 diffuseReflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo);
        return diffuseReflectance;
    }
    // Handle only non-delta specular lobes.
    else if (metallic == 1.f && !hasDeltaLobes)
    {
        const float NdotV = saturate(dot(sd.N, sd.V));
        const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
        float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
        specularReflectance = max(kNRDMinReflectance, specularReflectance);
        return specularReflectance;
    }
    //}
    //else if (materialType == MaterialType::Hair)
    //{
    //    const float3 reflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo);
    //    return reflectance;
    //}

    return 1.f;
}


bool isDeltaReflectionAllowedAlongDeltaTransmissionPath(const ShadingData sd, const FalcorPayload falcorPayload)
{
    // const BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
    const float metallic = falcorPayload.metallic; // md.specular.b; // Blue component stores metallic in MetalRough mode.
    const float insideIoR = falcorPayload.ior;// gScene.materials.evalIoR(sd.materialID);

    const float eta = sd.frontFacing ? (sd.IoR / insideIoR) : (insideIoR / sd.IoR);
    bool totalInternalReflection = evalFresnelDielectric(eta, sd.toLocal(sd.V).z) == 1.f;

    bool nonTransmissiveMirror = (falcorPayload.specularTransmission == 0.f) && (metallic == 1.f);

    return totalInternalReflection || nonTransmissiveMirror;
}


void setNRDPrimaryHitEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const float3 emission)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            gOut_PrimaryHitEmission[pixel] = float4(emission, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            gOut_PrimaryHitEmission[pixel] = 0.f;
        }
    }
}

void setNRDPrimaryHitReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            const float3 diffuseReflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo);
            gOut_PrimaryHitDiffuseReflectance[pixel] = float4(diffuseReflectance, 1.f);

            const float NdotV = saturate(dot(sd.N, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
            specularReflectance = max(kNRDMinReflectance, specularReflectance);
            gOut_PrimaryHitSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            gOut_PrimaryHitDiffuseReflectance[pixel] = 1.f;
            gOut_PrimaryHitSpecularReflectance[pixel] = 1.f;
        }
    }
}

void setNRDSampleHitDist(NRDBuffers outputNRD, const PathState path, const uint2 outSampleIdx)
{
    if (path.getVertexIndex() == 2)
    {
        gOut_SampleHitDist[outSampleIdx] = float(path.sceneLength);
    }
}

void setNRDSampleEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 outSampleIdx, const bool isPrimaryHit, const float3 emission, const bool wasDeltaOnlyPathBeforeScattering)
{
    if (useNRDDemodulation)
    {
        // Always demodulate emission on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            gOut_SampleEmission[outSampleIdx] = float4(emission, 1.f);
        }
        // Additionally, accumulate emission along the delta path.
        else if ((path.isDeltaReflectionPrimaryHit() || path.isDeltaTransmissionPath()) && any(emission > 0.f))
        {
            const bool demodulateDeltaReflectionEmission = path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering;
            const bool demodulateDeltaTransmissionEmission = path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering;
            if (demodulateDeltaReflectionEmission || demodulateDeltaTransmissionEmission)
            {
                float3 prevEmission = gOut_SampleEmission[outSampleIdx].rgb;
                gOut_SampleEmission[outSampleIdx] = float4(prevEmission + emission, 1.f);
            }
        }
    }
    else if (isPrimaryHit)
    {
        gOut_SampleEmission[outSampleIdx] = 0.f;
    }
}

void setNRDSampleReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 outSampleIdx, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties, const uint lobes, const bool wasDeltaOnlyPathBeforeScattering, const FalcorPayload falcorPayload)
{
    // Demodulate reflectance.
    if (useNRDDemodulation)
    {
        const bool hasDeltaLobes = (lobes & (uint)LobeType_Delta) != 0;
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType_NonDelta) != 0;

        // Always demodulate reflectance from diffuse and specular paths on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            if (path.isDiffusePrimaryHit())
            {
                const float3 diffuseReflectance = max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo);
                gOut_SampleReflectance[outSampleIdx] = float4(diffuseReflectance, 1.f);
            }
            else if (path.isSpecularPrimaryHit())
            {
                const float NdotV = saturate(dot(sd.N, sd.V));
                const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
                float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
                specularReflectance = max(kNRDMinReflectance, specularReflectance);
                gOut_SampleReflectance[outSampleIdx] = float4(specularReflectance, 1.f);
            }
            else
            {
                gOut_SampleReflectance[outSampleIdx] = 1.f;
            }
        }
        // Demodulate reflectance from the second vertex along delta reflection paths.
        else if (path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            //const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobes & (uint)LobeType_Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload);

            gOut_SampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
        // Demodulate reflectance from the first non-delta vertex along delta transmission paths.
        else if (path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            //const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobes & (uint)LobeType_Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload);

            gOut_SampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
    }
    else if (isPrimaryHit)
    {
        gOut_SampleReflectance[outSampleIdx] = 1.f;
    }
}

/** Write out delta reflection guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaReflectionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float hitDist)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    gOut_DeltaReflectionEmission[pixelPos] = float4(emission, 0.f);
    gOut_DeltaReflectionReflectance[pixelPos] = float4(reflectance, 0.f);
    gOut_DeltaReflectionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    gOut_DeltaReflectionPathLength[pixelPos] = pathLength;
    gOut_DeltaReflectionHitDist[pixelPos] = hitDist;
}

/** Write out delta transmission guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaTransmissionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float3 posW)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    gOut_DeltaTransmissionEmission[pixelPos] = float4(emission, 0.f);
    gOut_DeltaTransmissionReflectance[pixelPos] = float4(reflectance, 0.f);
    gOut_DeltaTransmissionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    gOut_DeltaTransmissionPathLength[pixelPos] = pathLength;
    gOut_DeltaTransmissionPosW[pixelPos] = float4(posW, 0.f);
}
