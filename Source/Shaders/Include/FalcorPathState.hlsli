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
//import Utils.Math.PackedFormats;
//__exported import Scene.HitInfo;
//__exported import Utils.Math.Ray;
//__exported import Utils.Sampling.SampleGenerator;
//__exported import Rendering.Materials.InteriorList;
//__exported import GuideData;

#pragma once

static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

static const float kRayTMax = 1e30f;

// Be careful with changing these. PathFlags share 32-bit uint with vertexIndex. For now, we keep 10 bits for vertexIndex.
// PathFlags take higher bits, VertexIndex takes lower bits.
static const uint kVertexIndexBitCount = 10u;
static const uint kVertexIndexBitMask = (1u << kVertexIndexBitCount) - 1u;
static const uint kPathFlagsBitCount = 32u - kVertexIndexBitCount;
static const uint kPathFlagsBitMask = ((1u << kPathFlagsBitCount) - 1u) << kVertexIndexBitCount;


/** Path flags. The path flags are currently stored in kPathFlagsBitCount bits.
*/
static uint PathFlags_active = 0x0001;                      ///< Path is active/terminated.
static uint PathFlags_hit = 0x0002;                         ///< Result of the scatter ray (0 = miss, 1 = hit).

static uint PathFlags_transmission = 0x0004;                ///< Scatter ray went through a transmission event.
static uint PathFlags_specular = 0x0008;                    ///< Scatter ray went through a specular event.
static uint PathFlags_delta = 0x0010;                       ///< Scatter ray went through a delta event.

static uint PathFlags_insideDielectricVolume = 0x0020;      ///< Path vertex is inside a dielectric volume.
static uint PathFlags_lightSampledUpper = 0x0040;           ///< Last path vertex sampled lights using NEE (in upper hemisphere).
static uint PathFlags_lightSampledLower = 0x0080;           ///< Last path vertex sampled lights using NEE (in lower hemisphere).

static uint PathFlags_diffusePrimaryHit = 0x0100;           ///< Scatter ray went through a diffuse event on primary hit.
static uint PathFlags_specularPrimaryHit = 0x0200;          ///< Scatter ray went through a specular event on primary hit.
static uint PathFlags_deltaReflectionPrimaryHit = 0x0400;   ///< Primary hit was sampled as the delta reflection.
static uint PathFlags_deltaTransmissionPath = 0x0800;       ///< Path started with and followed delta transmission events (whenever possible - TIR could be an exception) until it hit the first non-delta event.
static uint PathFlags_deltaOnlyPath = 0x1000;               ///< There was no non-delta events along the path so far.

// Bits 14 to kPathFlagsBitCount are still unused.


/** Bounce types. We keep separate counters for all of these.
*/
static uint BounceType_Diffuse = 0;                         ///< Diffuse reflection.
static uint BounceType_Specular = 1;                        ///< Specular reflection (including delta).
static uint BounceType_Transmission = 2;                    ///< Transmission (all kinds).




struct PathState
{
    uint        id;                     ///< Path ID encodes (pixel, sampleIdx) with 12 bits each for pixel x|y and 8 bits for sample index.

    uint        flagsAndVertexIndex;    ///< Higher kPathFlagsBitCount bits: Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
                                        ///< Lower kVertexIndexBitCount bits: Current vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    //uint16_t    rejectedHits;           ///< Number of false intersections rejected along the path. This is used as a safeguard to avoid deadlock in pathological cases.
    float       sceneLength;            ///< Path length in scene units (0.f at primary hit).
    uint        bounceCounters;         ///< Packed counters for different types of bounces (see BounceType).

    // Scatter ray
    float3      origin;                 ///< Origin of the scatter ray.
    float3      dir;                    ///< Scatter ray normalized direction.
    float       pdf;                    ///< Pdf for generating the scatter ray.
    float3      normal;                 ///< Shading normal at the scatter ray origin.
    //HitInfo     hit;                    ///< Hit information for the scatter ray. This is populated at committed triangle hits.

    float3      thp;                    ///< Path throughput.
    float3      L;                      ///< Accumulated path contribution.

    //GuideData   guideData;              ///< Denoiser guide data.
    //InteriorList interiorList;          ///< Interior list. Keeping track of a stack of materials with medium properties.
    SampleGenerator sg;                 ///< Sample generator state. Typically 4-16B.

    float       mip;                      ///< jn temp
    float       roughness;                ///< jn temp 
    float       outSceneLength;           ///< jn temp
    bool        outIsDiffsue;             ///< jn temp


    // Accessors
    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags_active); }
    bool isHit() { return hasFlag(PathFlags_hit); }
    bool isTransmission() { return hasFlag(PathFlags_transmission); }
    bool isSpecular() { return hasFlag(PathFlags_specular); }
    bool isDelta() { return hasFlag(PathFlags_delta); }
    bool isInsideDielectricVolume() { return hasFlag(PathFlags_insideDielectricVolume); }

    bool isLightSampled()
    {
        const uint bits = (uint(PathFlags_lightSampledUpper) | uint(PathFlags_lightSampledLower)) << kVertexIndexBitCount;
        return flagsAndVertexIndex & bits;
    }

    bool isLightSampledUpper() { return hasFlag(PathFlags_lightSampledUpper); }
    bool isLightSampledLower() { return hasFlag(PathFlags_lightSampledLower); }

    bool isDiffusePrimaryHit() { return hasFlag(PathFlags_diffusePrimaryHit); }
    bool isSpecularPrimaryHit() { return hasFlag(PathFlags_specularPrimaryHit); }
    bool isDeltaReflectionPrimaryHit() { return hasFlag(PathFlags_deltaReflectionPrimaryHit); }
    bool isDeltaTransmissionPath() { return hasFlag(PathFlags_deltaTransmissionPath); }
    bool isDeltaOnlyPath() { return hasFlag(PathFlags_deltaOnlyPath); }


    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !isDelta(); }

    /*[mutating]*/ void terminate() { setFlag(PathFlags_active, false); }
    /*[mutating]*/ void setActive() { setFlag(PathFlags_active); }
    ///*[mutating]*/ void setHit(HitInfo hitInfo) { hit = hitInfo; setFlag(PathFlags_hit); }
    /*[mutating]*/ void clearHit() { setFlag(PathFlags_hit, false); }

    /*[mutating] */void clearEventFlags()
    {
        const uint bits = (uint(PathFlags_transmission) | uint(PathFlags_specular) | uint(PathFlags_delta)) << kVertexIndexBitCount;
        flagsAndVertexIndex &= ~bits;
    }

    /*[mutating]*/ void setTransmission(bool value = true) { setFlag(PathFlags_transmission, value); }
    /*[mutating]*/ void setSpecular(bool value = true) { setFlag(PathFlags_specular, value); }
    /*[mutating]*/ void setDelta(bool value = true) { setFlag(PathFlags_delta, value); }
    /*[mutating]*/ void setInsideDielectricVolume(bool value = true) { setFlag(PathFlags_insideDielectricVolume, value); }
    /*[mutating]*/ void setLightSampled(bool upper, bool lower) { setFlag(PathFlags_lightSampledUpper, upper); setFlag(PathFlags_lightSampledLower, lower); }
    /*[mutating]*/ void setDiffusePrimaryHit(bool value = true) { setFlag(PathFlags_diffusePrimaryHit, value); }
    /*[mutating]*/ void setSpecularPrimaryHit(bool value = true) { setFlag(PathFlags_specularPrimaryHit, value); }
    /*[mutating]*/ void setDeltaReflectionPrimaryHit(bool value = true) { setFlag(PathFlags_deltaReflectionPrimaryHit, value); }
    /*[mutating]*/ void setDeltaTransmissionPath(bool value = true) { setFlag(PathFlags_deltaTransmissionPath, value); }
    /*[mutating]*/ void setDeltaOnlyPath(bool value = true) { setFlag(PathFlags_deltaOnlyPath, value); }


    bool hasFlag(uint flag)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        return (flagsAndVertexIndex & bit) != 0;
    }

    /*[mutating]*/ void setFlag(uint flag, bool value = true)
    {
        const uint bit = uint(flag) << kVertexIndexBitCount;
        if (value) flagsAndVertexIndex |= bit;
        else flagsAndVertexIndex &= ~bit;
    }

    uint getBounces(uint type)
    {
        const uint shift = (uint)type << 3;
        return (bounceCounters >> shift) & 0xff;
    }

    /*[mutating]*/ void setBounces(uint type, uint bounces)
    {
        const uint shift = (uint)type << 3;
        bounceCounters = (bounceCounters & ~((uint)0xff << shift)) | ((bounces & 0xff) << shift);
    }

   /* [mutating]*/ void incrementBounces(uint type)
    {
        const uint shift = (uint)type << 3;
        // We assume that bounce counters cannot overflow.
        bounceCounters += ((uint)1 << shift);
    }

    uint2 getPixel() { return uint2(id, id >> 12) & 0xfff; }
    uint getSampleIdx() { return id >> 24; }

    // Unsafe - assumes that index is small enough.
    /*[mutating]*/ void setVertexIndex(uint index)
    {
        // Clear old vertex index.
        flagsAndVertexIndex &= kPathFlagsBitMask;
        // Set new vertex index (unsafe).
        flagsAndVertexIndex |= index;
    }

    uint getVertexIndex() { return flagsAndVertexIndex & kVertexIndexBitMask; }

    // Unsafe - assumes that vertex index never overflows.
    /*[mutating]*/ void incrementVertexIndex() { flagsAndVertexIndex += 1; }
    // Unsafe - assumes that vertex index will never be decremented below zero.
    /*[mutating]*/ void decrementVertexIndex() { flagsAndVertexIndex -= 1; }

    RayDesc getScatterRay()
    {
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.f;
        ray.TMax = kRayTMax;
        return ray;
    }
};