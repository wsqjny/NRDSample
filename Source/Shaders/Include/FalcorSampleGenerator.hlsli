#pragma once


struct SampleGenerator
{
    uint state;
};

float sampleNext1D(inout SampleGenerator sg)
{
    float2 rnd = STL::Rng::GetFloat2();
    return rnd.x;
}

float2 sampleNext2D(inout SampleGenerator sg)
{
    float2 rnd = STL::Rng::GetFloat2();
    return rnd;
}
