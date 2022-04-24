#include "Shared.hlsli"

NRI_RESOURCE(Texture2D<float4>,     gIn_Input,          t, 0, 1);
NRI_RESOURCE(Texture2D<float4>,     gIn_InputHistory,   t, 1, 1);
NRI_RESOURCE(RWTexture2D<float4>,   gInOut_History,     u, 2, 1);

[numthreads(16, 16, 1)]
void main(uint2 pixelPos : SV_DispatchThreadId)
{
    float2 pixelUv = float2(pixelPos + 0.5) * gInvRectSize;

    float4 input = gIn_Input[pixelPos];
    float4 history = gIn_InputHistory[pixelPos];
    float4 result = lerp(history, input, gAccumSpeed.x);

    gInOut_History[pixelPos] = result;// pixelUv.x > gSplitScreen ? result : input;
}

