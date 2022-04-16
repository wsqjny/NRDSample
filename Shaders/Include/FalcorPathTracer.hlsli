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
#pragma once

#include "FalcorNRDHelpers.hlsli"
#include "FalcorEnvMapSampler.hlsli"

static const float specularRoughnessThreshold = 0.25f; ///< Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.



// TODO: Remove explicitly assigned values when we can use the enums as default initializer below
//enum class MISHeuristic
//{
static const uint MISHeuristic_Balance = 0;       ///< Balance heuristic.
static const uint MISHeuristic_PowerTwo = 1;      ///< Power heuristic (exponent = 2.0).
static const uint MISHeuristic_PowerExp = 2;      ///< Power heuristic (variable exponent).
//};



/** Types of samplable lights.
*/
//enum class LightType
//{
static const uint LightType_EnvMap = 0;
static const uint LightType_Emissive = 1;
static const uint LightType_Analytic = 2;
//};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  dir;        ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

    RayDesc getVisibilityRay() 
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.f;
        ray.TMax = distance;
        return ray;
    }
};


/** Describes a path vertex.
*/
struct PathVertex
{
    uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    float3 pos;         ///< Vertex position.
    float3 normal;      ///< Shading normal at the vertex (zero if not on a surface).
    float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).

    /** Initializes a path vertex.
        \param[in] index Vertex index.
        \param[in] pos Vertex position.
        \param[in] normal Shading normal.
        \param[in] faceNormal Geometry normal.
    */
    void __init(uint index, float3 pos, float3 normal, float3 faceNormal)
    {
        this.index = index;
        this.pos = pos;
        this.normal = normal;
        this.faceNormal = faceNormal;
    }

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
        for visibility rays.
        \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 rayDir)
    {
        return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
    }
};


struct PathTracer 
{
    SceneLights sceneLight;

    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    NRDBuffers outputNRD;                           ///< Output NRD data.

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType_Diffuse);
        const uint specularBounces = path.getBounces(BounceType_Specular);
        const uint transmissionBounces = path.getBounces(BounceType_Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }



    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = (PathState)0.0;
        path.setActive();
        path.id = pathID;
        path.thp = 1.f;

        //const uint2 pixel = path.getPixel();

        // Create primary ray.
        // Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        //path.origin = cameraOrigin;// cameraRay.origin;
        // path.dir = cameraDir;// cameraRay.dir;

        // Create sample generator.
        //const uint maxSpp = 1;
        //path.sg = SampleGenerator(pixel, params.seed * maxSpp + path.getSampleIdx());
        path.sg = (SampleGenerator)0.0;

        // Load the primary hit info from the V-buffer.
        //const HitInfo hit = HitInfo(vbuffer[pixel]);

        // If invalid, the path is still active and treated as a miss.
        //if (hit.isValid())
        {
            //path.setHit(hit);
            path.setVertexIndex(1);
        }
    }

    /** Set up path for logging and debugging.
    \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        //printSetPixel(path.getPixel());
        //logSetPixel(path.getPixel());
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        path.L += path.thp * radiance;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const StandardBSDF bsdf, inout PathState path, const FalcorPayload falcorPayload)
    {
        BSDFSample result;
        bool valid = bsdf.sample(sd, path.sg, result, kUseBSDFSampling);
        if (valid) valid = generateScatterRay(result, sd, bsdf, path);

        // Ignore valid on purpose for now.
        if (kOutputNRDData)
        {
            const uint lobes = bsdf.getLobes(sd);
            const bool hasDeltaTransmissionLobe = (lobes & (uint)LobeType_DeltaTransmission) != 0;
            const bool hasNonDeltaLobes = (lobes & (uint)LobeType_NonDelta) != 0;

            if (path.getVertexIndex() == 1)
            {
                path.setDiffusePrimaryHit(result.isLobe(LobeType_Diffuse));
                path.setSpecularPrimaryHit(result.isLobe(LobeType_Specular));

                if (kOutputNRDAdditionalData)
                {
                    // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                    path.setDeltaOnlyPath(result.isLobe(LobeType_DeltaReflection) || result.isLobe(LobeType_DeltaTransmission));

                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType_DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType_DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

                if (kOutputNRDAdditionalData && path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
                {
                    if (result.isLobe(LobeType_DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd, falcorPayload))
                    {
                        path.setDeltaTransmissionPath(false);
                    }
                }
            }
        }

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const StandardBSDF bsdf, inout PathState path)
    {
        path.dir = bs.wo;
        updatePathThroughput(path, bs.weight);
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType_Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = bsdf.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType_DiffuseReflection) || roughness > specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType_Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType_Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType_Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType_Transmission))
        {
            path.incrementBounces(BounceType_Transmission);
            path.setTransmission();

            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);

                // Update interior list and inside volume flag.
                /*if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }*/
            }
        }

        // Save the shading normal. This is needed for MIS.
        path.normal = sd.N;

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        //switch (MISHeuristic(kMISHeuristic))
        switch (misHeuristic)
        {
        case MISHeuristic_Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic_PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic_PowerExp:
        {
            // Power exp heuristic
           // float q0 = pow(n0 * p0, kMISPowerExponent);
           // float q1 = pow(n1 * p1, kMISPowerExponent);
            //return q0 / (q0 + q1);

            return 0.f;
        }
        default:
            return 0.f;
        }
    }


    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = (LightSample)0.0; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : 0.0;
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = (LightSample)0.0; // Default initialization to avoid divergence at returns.

        uint lightCount = sceneLight.LightCount;// gScene.getLightCount();                          
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, sceneLight.LightArray[lightIndex], sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        return false;
    };

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        //ls = LightSample(0.0);

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType_EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightType_Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            //bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            //valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType_Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float cosTheta = dot(vertex.normal, ls.dir);
        // Flip the face normal to point in the same hemisphere as the shading normal.
        float3 faceNormal = sign(dot(vertex.normal, vertex.faceNormal)) * vertex.faceNormal;
        float cosThetaFace = dot(faceNormal, ls.dir);
        if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) return false;
        if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        return false;
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit(inout PathState path, const FalcorPayload falcorPayload, const VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;

        // Load shading data. This is a long latency operation.
        //ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);
        ShadingData sd = loadShadingData(falcorPayload);

        // Create BSDF instance and query its properties.
        // const IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
        
        StandardBSDF bsdf = (StandardBSDF)0;        
        bsdf.data = LoadStanderdBSDFData(falcorPayload, sd);
        
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType_Diffuse) > 0 && isSpecular)
        {
            sd.mtlActiveLobe =(uint)LobeType_Diffuse;
        }

        // Optionally disable emission inside volumes.
        //if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        //{
        //    bsdfProperties.emission = float3(0.f);
        //}

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission or delta scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEmissive = false;

        float3 attenuatedEmission = 0.f;

        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
#if 0
            float misWeight = 1.f;
            if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            addToPathContribution(path, misWeight * bsdfProperties.emission);

            attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;
#endif
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        {
            path.origin = sd.computeNewRayOrigin();
        }

        // Determine if BSDF has non-delta lobes.
        const uint lobes = bsdf.getLobes(sd);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType_NonDelta) != 0;

        // Check if we should apply NEE.
        const bool applyNEE = kUseNEE && hasNonDeltaLobes;

        // Check if sample from RTXDI should be applied instead of NEE.
        const bool applyRTXDI = kUseRTXDI && isPrimaryHit && hasNonDeltaLobes;

        // TODO: Support multiple shadow rays.
        path.setLightSampled(false, false);
        if (applyNEE || applyRTXDI)
        {
            LightSample ls;
            bool validSample = false;

            if (applyRTXDI)
            {
                // Query final sample from RTXDI.
                //validSample = gRTXDI.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.Li);
                //ls.origin = sd.computeNewRayOrigin();
            }
            else
            {
                // Setup path vertex.
                PathVertex vertex; 
                vertex.__init(path.getVertexIndex(), sd.posW, sd.N, sd.faceN);

                // Determine if upper/lower hemispheres need to be sampled.
                bool sampleUpperHemisphere = ((lobes & (uint)LobeType_NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                bool sampleLowerHemisphere = ((lobes & (uint)LobeType_NonDeltaTransmission) != 0);

                // Sample a light.
                validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
                path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
            }

            if (validSample)
            {
                // Apply MIS weight.
                if (kUseMIS && !applyRTXDI && ls.lightType != (uint)LightType_Analytic)
                {
                    float scatterPdf = bsdf.evalPdf(sd, ls.dir, kUseBSDFSampling);
                    ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = bsdf.eval(sd, ls.dir);
                float3 Lr = weight * ls.Li;
                if (any(Lr > 0.f))
                {
                    const RayDesc ray = ls.getVisibilityRay();
                    //logTraceRay(PixelStatsRayType::Visibility);                    

                    bool visible = vq.traceVisibilityRay(ray);
                    if (visible) addToPathContribution(path, Lr);
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return;
        }

        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path, falcorPayload);

        // Output guide data.
        //if (path.getVertexIndex() == 1)
        //{
        //    setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);
        //}
        //if (path.getVertexIndex() == 2 && (path.getBounces(BounceType::Specular) == 1 || path.getBounces(BounceType::Transmission) == 1))
        //{
        //    setIndirectSurfaceGuideData(path.guideData, sd, bsdfProperties);
        //}

        if (kOutputNRDData)
        {
            const uint2 pixel = path.getPixel();
            //const uint outSampleIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

            setNRDPrimaryHitEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission);
            setNRDPrimaryHitReflectance(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, sd, bsdfProperties);

            setNRDSampleHitDist(outputNRD, path, pixel);
            setNRDSampleEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission, wasDeltaOnlyPathBeforeScattering);
            setNRDSampleReflectance(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, sd, bsdfProperties, lobes, wasDeltaOnlyPathBeforeScattering, falcorPayload);
        }

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType_Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission, delta or volume scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEnv = false;

        float3 emitterRadiance = 0.f;

        if (computeEnv)
        {
            //logPathVertex();

            float misWeight = 1.f;
            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            emitterRadiance = misWeight * Le;
            addToPathContribution(path, emitterRadiance);


#if 0
            if (kOutputGuideData && path.getVertexIndex() == 2
                && (path.getBounces(BounceType::Specular) == 1
                    || path.getBounces(BounceType::Transmission) == 1))
            {
                // Compress dynamic range similar to UE4.
                float3 compressedColor = pow(Le / (Le + 1.0f), 0.454545f);
                path.guideData.setIndirectAlbedo(compressedColor);
                path.guideData.setReflectionPos(path.dir * kEnvMapDepth);
            }
#endif
        }

#if 0
        if (kOutputGuideData && path.getVertexIndex() == 1)
        {
            path.guideData.setNormal(-path.dir);
        }
#endif

        if (kOutputNRDData)
        {
            const uint2 pixel = path.getPixel();
            //const uint outSampleIdx = params.getSampleOffset(path.getPixel()) + path.getSampleIdx();
            setNRDSampleHitDist(outputNRD, path, pixel);
        }

#if defined(DELTA_REFLECTION_PASS)
        if (path.isDeltaReflectionPrimaryHit())
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
        }
        else
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
        }
#elif defined(DELTA_TRANSMISSION_PASS)
        if (path.isDeltaTransmissionPath())
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, path.origin + path.dir * kNRDInvalidPathLength);
        }
        else
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, 0.f);
        }
#endif

        path.terminate();
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(const PathState path)
    {
        // assert(!any(isnan(path.L)));

        // Log path length.
        // logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        // const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

        //if (kSamplesPerPixel == 1)
        //{
        //    // Write color directly to frame buffer.
        //    outputColor[pixel] = float4(path.L, 1.f);
        //}
        //else
        //{
        //    // Write color to per-sample buffer.
        //    sampleColor[outIdx].set(path.L);
        //}

        //if (kOutputGuideData)
        //{
        //    sampleGuideData[outIdx] = path.guideData;
        //}

        if (kOutputNRDData)
        {
#if 0
            // TODO: Optimize this for 1 SPP. It doesn't have to go through resolve pass like the color above.
            NRDRadiance data = {};

            if (path.isDiffusePrimaryHit()) data.setPathType(NRDPathType::Diffuse);
            else if (path.isSpecularPrimaryHit()) data.setPathType(NRDPathType::Specular);
            else if (path.isDeltaReflectionPrimaryHit()) data.setPathType(NRDPathType::DeltaReflection);
            else if (path.isDeltaTransmissionPath()) data.setPathType(NRDPathType::DeltaTransmission);
            else data.setPathType(NRDPathType::Residual);

            data.setRadiance(path.L);

            outputNRD.sampleRadiance[outIdx] = data;
#endif
            
            float4 data = 0;
            data.rgb = path.L;
            
            if (path.isDiffusePrimaryHit()) data.a = 0.0;
            else if (path.isSpecularPrimaryHit()) data.a = 0.1;
            else if (path.isDeltaReflectionPrimaryHit()) data.a = 0.2;
            else if (path.isDeltaTransmissionPath()) data.a = 0.3;
            else data.a = 0.4;

            // TODO: 
            gOut_SampleRadiance[pixel] = data;
        }
    }





    ////- Extensions, PathTracerNRD.slang
    /** Handle hit on delta reflection materials.
    After handling the hit, the path is terminated.
    Executed only for guide paths.
    \param[in,out] path The path state.
    */
    void handleDeltaReflectionHit(inout PathState path, const FalcorPayload falcorPayload)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out reflectance/normalWRough/posW of the second path vertex
        // - Terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        //const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        //let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        //ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);
        ShadingData sd = loadShadingData(falcorPayload);

        // Reject false hits in nested dielectrics.
        // if (!handleNestedDielectrics(sd, path)) return;

        // Create BSDF instance and query its properties.
        // let bsdf = gScene.materials.getBSDF(sd, lod);
        // let bsdfProperties = bsdf.getProperties(sd);

        StandardBSDF bsdf = (StandardBSDF)0;
        bsdf.data = LoadStanderdBSDFData(falcorPayload, sd);

        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        // Query BSDF lobes.
        const uint lobes = bsdf.getLobes(sd);
        const bool hasDeltaLobes = (lobes & (uint)LobeType_Delta) != 0;

        // const MaterialType materialType = sd.mtl.getMaterialType();


        if (isPrimaryHit)
        {
            // Terminate without the write-out if the path doesn't start as delta reflection.
            bool hasDeltaReflectionLobe = ((lobes & (uint)LobeType_DeltaReflection) != 0);
            if (!hasDeltaReflectionLobe)
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
                path.terminate();
                return;
            }

            // Add primary ray length to the path length.
            float primaryHitDist = length(sd.posW - path.origin);
            path.sceneLength += float16_t(primaryHitDist);
            // Hijack pdf that we don't need.
            path.pdf += primaryHitDist;

            // Set the active lobes only to delta reflection on the first bounce.
            sd.mtlActiveLobe = (uint)LobeType_DeltaReflection;
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;

            // Terminate after scatter ray on last vertex has been processed or non-delta lobe exists.
            const bool lastVertex = hasFinishedSurfaceBounces(path);
            const bool hasNonDeltaLobes = (lobes & (uint)LobeType_NonDelta) != 0;
            const bool isEmissive = any(bsdfProperties.emission > 0.f);

            if (lastVertex || hasNonDeltaLobes || isEmissive)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), hitDist);

                path.terminate();
                return;
            }

            // For glass in reflections, force guide paths to always follow transmission/reflection based on albedos.
            // This is pretty hacky but works best our of the possible options.
            // Stable guide buffers are a necessity.
            if (bsdfProperties.isTransmissive() && all(bsdfProperties.specularReflectionAlbedo <= bsdfProperties.specularTransmissionAlbedo))
            {
                sd.mtlActiveLobe = (uint)LobeType_DeltaTransmission;
            }
            else
            {
                sd.mtlActiveLobe = (uint)LobeType_DeltaReflection;
            }
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Hijack pdf that we don't need.
        float primaryHitDist = path.pdf;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path, falcorPayload);

        path.pdf = primaryHitDist;

        if (!valid)
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            path.terminate();
            return;
        }

#if 0
        // Terminate if transmission lobe was chosen but volume absorption is too high
        // but store the previous vertex shading data.
        if (path.isTransmission())
        {
            // Fetch volume absorption from the material. This field only exist in basic materials for now.
            bool semiOpaque = false;
            if (gScene.materials.isBasicMaterial(sd.materialID))
            {
                BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
                // TODO: Expose this arbitrary value as a constant.
                semiOpaque = any(md.volumeAbsorption > 100.f);
            }

            if (semiOpaque)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), hitDist);

                path.terminate();
                return;
            }
        }
#endif
    }

    /** Handle hit on delta transmission materials.
        After handling the hit, a new scatter (delta transmission only) ray is generated or the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaTransmissionHit(inout PathState path, const FalcorPayload falcorPayload)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out albedo/normal/posW on the first hit of non delta transmission BSDF lobe
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        //const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        // let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        // ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);
        ShadingData sd = loadShadingData(falcorPayload);

        // Reject false hits in nested dielectrics.
        //if (!handleNestedDielectrics(sd, path)) return;

        // Create BSDF instance and query its properties.
        //let bsdf = gScene.materials.getBSDF(sd, lod);
        //let bsdfProperties = bsdf.getProperties(sd);
        StandardBSDF bsdf = (StandardBSDF)0;
        bsdf.data = LoadStanderdBSDFData(falcorPayload, sd);

        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        const uint lobes = bsdf.getLobes(sd);

        // Terminate without the write-out if the path doesn't start as delta transmission.
        const bool hasDeltaTransmissionLobe = ((lobes & (uint)LobeType_DeltaTransmission) != 0);
        if (isPrimaryHit && !hasDeltaTransmissionLobe)
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, 0.f);
            path.terminate();
            return;
        }

        if (isPrimaryHit)
        {
            // Add primary ray length to the path length.
            path.sceneLength += float16_t(length(sd.posW - path.origin));
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;
        }

        // Terminate the delta transmission path.
        const bool lastVertex = hasFinishedSurfaceBounces(path);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType_NonDelta) != 0;

        // Fetch volume absorption from the material. This field only exist in basic materials for now.
        bool semiOpaque = false;
        //if (gScene.materials.isBasicMaterial(sd.materialID))
        //{
        //    BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
            // TODO: Expose this arbitrary value as a constant.
         //   semiOpaque = any(md.volumeAbsorption > 100.f);
       // }

        //const MaterialType materialType = sd.mtl.getMaterialType();
        const bool hasDeltaLobes = (lobes & (uint)LobeType_Delta) != 0;

        if (lastVertex || semiOpaque)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload), emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Set the active lobes only to delta transmission.
        sd.mtlActiveLobe = (uint)LobeType_DeltaTransmission;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path, falcorPayload);

        // Delta transmission was not possible, fallback to delta reflection if it's allowed.
        if (!valid && isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd, falcorPayload))
        {
            sd.mtlActiveLobe = (uint)LobeType_DeltaTransmission | (uint)LobeType_DeltaReflection;
            valid = generateScatterRay(sd, bsdf, path, falcorPayload);
        }

        if (!valid)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(hasDeltaLobes, sd, bsdfProperties, falcorPayload), emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }
    }
};