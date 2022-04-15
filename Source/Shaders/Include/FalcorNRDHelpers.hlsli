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



#if 0

void setNRDPrimaryHitEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const float3 emission)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        if (useNRDDemodulation)
        {
            outputNRD.primaryHitEmission[pixel] = float4(emission, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitEmission[pixel] = 0.f;
        }
    }
}

void setNRDPrimaryHitReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint2 pixel, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        if (useNRDDemodulation)
        {
            outputNRD.primaryHitDiffuseReflectance[pixel] = float4(bsdfProperties.diffuseReflectionAlbedo, 1.f);

            const float NdotV = saturate(dot(sd.N, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            const float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
            outputNRD.primaryHitSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitDiffuseReflectance[pixel] = 1.f;
            outputNRD.primaryHitSpecularReflectance[pixel] = 1.f;
        }
    }
}

void setNRDSampleHitDist(NRDBuffers outputNRD, const PathState path, const uint outSampleIdx)
{
    if (path.getVertexIndex() == 2)
    {
        outputNRD.sampleHitDist[outSampleIdx] = float(path.sceneLength);
    }
}

void setNRDSampleEmission(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint outSampleIdx, const bool isPrimaryHit, const float3 emission)
{
    if (useNRDDemodulation)
    {
        // Always demodulate emission on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            outputNRD.sampleEmission[outSampleIdx] = float4(emission, 1.f);
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleEmission[outSampleIdx] = 0.f;
    }
}

void setNRDSampleReflectance(NRDBuffers outputNRD, const bool useNRDDemodulation, const PathState path, const uint outSampleIdx, const bool isPrimaryHit, const ShadingData sd, const BSDFProperties bsdfProperties)
{
    if (useNRDDemodulation)
    {
        // Always demodulate reflectance from diffuse and specular paths on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            if (path.isDiffusePrimaryHit())
            {
                outputNRD.sampleReflectance[outSampleIdx] = float4(bsdfProperties.diffuseReflectionAlbedo, 1.f);
            }
            else if (path.isSpecularPrimaryHit())
            {
                const float NdotV = saturate(dot(sd.N, sd.V));
                const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
                const float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
                outputNRD.sampleReflectance[outSampleIdx] = float4(specularReflectance, 1.f);
            }
            else
            {
                outputNRD.sampleReflectance[outSampleIdx] = 1.f;
            }
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleReflectance[outSampleIdx] = 1.f;
    }
}

#endif



void setNRDSampleHitDist(inout PathState path)
{
    if (path.getVertexIndex() == 2)
    {
        path.outSceneLength = path.sceneLength;
    }
}

void setNRDPrimaryLobe(inout PathState path, bool isPrimaryHit)
{
    if (isPrimaryHit)
    {
        if (path.isDiffusePrimaryHit())
        {
            path.outIsDiffsue = true;
        }
        else if (path.isSpecularPrimaryHit())
        {
            path.outIsDiffsue = false;
        }
    }
}



/** Write out delta reflection guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaReflectionGuideBuffers(NRDBuffers outputNRD, const bool useNRDDemodulation, const uint2 pixelPos, float3 reflectance, float3 emission, float3 normal, float roughness, float pathLength, float hitDist)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    // float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    gOut_DeltaReflectionEmission[pixelPos] = float4(emission, 0.f);
    gOut_DeltaReflectionReflectance[pixelPos] = float4(reflectance, 0.f);
    gOut_DeltaReflectionNormWRoughMaterialID[pixelPos] = float4(normal, clampedRoughness);// float4(octNormal, clampedRoughness, materialID);
    gOut_DeltaReflectionPathLength[pixelPos] = pathLength;
    gOut_DeltaReflectionHitDist[pixelPos] = hitDist;
}

#if 0
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

    outputNRD.deltaTransmissionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaTransmissionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaTransmissionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaTransmissionPathLength[pixelPos] = pathLength;
    outputNRD.deltaTransmissionPosW[pixelPos] = float4(posW, 0.f);
}
#endif






