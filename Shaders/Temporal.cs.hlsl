/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Shared.hlsli"

NRI_RESOURCE( Texture2D<float3>, gIn_ObjectMotion, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_ComposedLighting_ViewZ, t, 1, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_TransparentLayer, t, 2, 1 );
NRI_RESOURCE( Texture2D<float3>, gIn_History, t, 3, 1 );

NRI_RESOURCE( RWTexture2D<float3>, gOut_History, u, 4, 1 );

#define BORDER          1
#define GROUP_X         16
#define GROUP_Y         16
#define BUFFER_X        ( GROUP_X + BORDER * 2 )
#define BUFFER_Y        ( GROUP_Y + BORDER * 2 )

#define PRELOAD_INTO_SMEM \
    int2 groupBase = pixelPos - threadPos - BORDER; \
    uint stageNum = ( BUFFER_X * BUFFER_Y + GROUP_X * GROUP_Y - 1 ) / ( GROUP_X * GROUP_Y ); \
    [unroll] \
    for( uint stage = 0; stage < stageNum; stage++ ) \
    { \
        uint virtualIndex = threadIndex + stage * GROUP_X * GROUP_Y; \
        uint2 newId = uint2( virtualIndex % BUFFER_X, virtualIndex / BUFFER_Y ); \
        if( stage == 0 || virtualIndex < BUFFER_X * BUFFER_Y ) \
            Preload( newId, groupBase + newId ); \
    } \
    GroupMemoryBarrierWithGroupSync( )

groupshared float4 s_Data[ BUFFER_Y ][ BUFFER_X ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );

    float4 color_viewZ = gIn_ComposedLighting_ViewZ[ globalPos ];
    color_viewZ.xyz = ApplyPostLightingComposition( globalPos, color_viewZ.xyz, gIn_TransparentLayer );
    color_viewZ.w = abs( color_viewZ.w ) * STL::Math::Sign( gNearZ ) / NRD_FP16_VIEWZ_SCALE;

    s_Data[ sharedPos.y ][ sharedPos.x ] = color_viewZ;
}

#define MOTION_LENGTH_SCALE 16.0

[numthreads( GROUP_X, GROUP_Y, 1 )]
void main( int2 threadPos : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    float4 inColor = gIn_ComposedLighting_ViewZ[pixelPos];
    
    float3 Lsum = inColor.rgb;
    Lsum *= gExposure;

    // Tonemap
    if (gOnScreen == SHOW_FINAL)
        Lsum = STL::Color::HdrToLinear_Uncharted(Lsum);

    // Conversion
    if (gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR)
        Lsum = STL::Color::LinearToSrgb(Lsum);

    // Output
    gOut_History[pixelPos] = Lsum;
}
