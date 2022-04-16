/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#if( !defined( COMPILER_FXC ) )

#include "Shared.hlsli"


#define NRD_METHOD_RELAX_DIFFUSE_SPECULAR 0
#define NRD_METHOD_REBLUR_DIFFUSE_SPECULAR 1

// Inputs
NRI_RESOURCE(Texture2D<float>,  gViewZ,                             t, 0, 1);
NRI_RESOURCE(Texture2D<float4>, gNormalRoughness,                   t, 1, 1);

// Outputs
NRI_RESOURCE(RWTexture2D<float4>, gDiffuseRadianceHitDist,          u, 2, 1);
NRI_RESOURCE(RWTexture2D<float4>, gSpecularRadianceHitDist,         u, 3, 1);


#if 0   // add to global cb.
NRI_RESOURCE(cbuffer, globalConstants, b, 0, 0)
{
    float4  gHitDistParams;
    float   gMaxIntensity;
    uint    gNRDMethod;
};
#endif

/** Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
    \param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
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


[numthreads( 16, 16, 1)]
void main( int2 dispatchThreadId : SV_DispatchThreadId )
{
    int2 ipos = dispatchThreadId.xy;

    float4 diffuseRadianceHitDist = gDiffuseRadianceHitDist[ipos];
    float4 specularRadianceHitDist = gSpecularRadianceHitDist[ipos];

    clampRadiance(diffuseRadianceHitDist.rgb, specularRadianceHitDist.rgb);

    if (gDenoiserType != REBLUR)
    {
        diffuseRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);
        specularRadianceHitDist = RELAX_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);
    }
    else
    {
        float viewZ = gViewZ[ipos];
        float linearRoughness = gNormalRoughness[ipos].z;

        diffuseRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(diffuseRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
        REBLUR_FrontEnd_PackRadianceAndHitDist(diffuseRadianceHitDist.rgb, diffuseRadianceHitDist.a);

        specularRadianceHitDist.a = REBLUR_FrontEnd_GetNormHitDist(specularRadianceHitDist.a, viewZ, gHitDistParams, linearRoughness);
        REBLUR_FrontEnd_PackRadianceAndHitDist(specularRadianceHitDist.rgb, specularRadianceHitDist.a);
    }

    gDiffuseRadianceHitDist[ipos] = diffuseRadianceHitDist;
    gSpecularRadianceHitDist[ipos] = specularRadianceHitDist;
}

#else

[numthreads(16, 16, 1)]
void main(uint2 pixelPos : SV_DispatchThreadId)
{
    // no TraceRayInline support, because of:
    //  - DXBC
    //  - SPIRV generation is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/4221
}

#endif