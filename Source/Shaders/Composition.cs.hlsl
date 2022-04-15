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
NRI_RESOURCE( Texture2D<float4>, gEmission,						t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gDiffuseReflectance,			t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gDiffuseRadiance,				t, 2, 1);
NRI_RESOURCE( Texture2D<float4>, gSpecularReflectance,			t, 3, 1);
NRI_RESOURCE( Texture2D<float4>, gSpecularRadiance,				t, 4, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaReflectionEmission,		t, 5, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaReflectionReflectance,	t, 6, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaReflectionRadiance,		t, 7, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaTransmissionEmission,	t, 8, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaTransmissionReflectance, t, 9, 1);
NRI_RESOURCE( Texture2D<float4>, gDeltaTransmissionRadiance,	t, 10, 1);

// Outputs
NRI_RESOURCE( RWTexture2D<float4>, gOutput, u, 11, 1 );


bool is_valid(Texture2D tex)
{
    return true;
}

[numthreads( 16, 16, 1)]
void main( int2 dispatchThreadId : SV_DispatchThreadId )
{
    const uint2 pixel = dispatchThreadId.xy;
 
    float4 outputColor = 0.0;

    if (is_valid(gEmission))
    {
        outputColor.rgb += gEmission[pixel].rgb;
    }

    if (is_valid(gDiffuseRadiance))
    {
        float3 diffuseColor = gDiffuseRadiance[pixel].rgb;

        if (is_valid(gDiffuseReflectance))
        {
            diffuseColor *= gDiffuseReflectance[pixel].rgb;
        }
        outputColor.rgb += diffuseColor;
    }

    if (is_valid(gSpecularRadiance))
    {
        float3 specularColor = gSpecularRadiance[pixel].rgb;

        if (is_valid(gSpecularReflectance))
        {
            specularColor *= gSpecularReflectance[pixel].rgb;
        }

        outputColor.rgb += specularColor;
    }

    if (is_valid(gDeltaReflectionEmission))
    {
        outputColor.rgb += gDeltaReflectionEmission[pixel].rgb;
    }

    if (is_valid(gDeltaReflectionRadiance))
    {
        float3 deltaReflectionColor = gDeltaReflectionRadiance[pixel].rgb;

        if (is_valid(gDeltaReflectionReflectance))
        {
            deltaReflectionColor *= gDeltaReflectionReflectance[pixel].rgb;
        }

        outputColor.rgb += deltaReflectionColor;
    }

    if (is_valid(gDeltaTransmissionEmission))
    {
        outputColor.rgb += gDeltaTransmissionEmission[pixel].rgb;
    }

    if (is_valid(gDeltaTransmissionRadiance))
    {
        float3 deltaTransmissionColor = gDeltaTransmissionRadiance[pixel].rgb;

        if (is_valid(gDeltaTransmissionReflectance))
        {
            deltaTransmissionColor *= gDeltaTransmissionReflectance[pixel].rgb;
        }

        outputColor.rgb += deltaTransmissionColor;
    }

    gOutput[pixel] = outputColor;
}
