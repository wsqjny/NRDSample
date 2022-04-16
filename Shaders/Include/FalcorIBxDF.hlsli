#pragma once

// Minimum cos(theta) for the incident and outgoing vectors.
// Some BSDF functions are not robust for cos(theta) == 0.0,
// so using a small epsilon for consistency.
// TODO: Move into IBxDF if possible
static const float kMinCosTheta = 1e-6f;


/** Low-level interface for BxDF functions.

    Conventions:
    - All operations are done in a local coordinate frame.
    - The local frame has normal N=(0,0,1), tangent T=(1,0,0) and bitangent B=(0,1,0).
    - The incident and outgoing direction point away from the shading location.
    - The incident direction (wi) is always in the positive hemisphere.
    - The outgoing direction (wo) is sampled.
    - Evaluating the BxDF always includes the foreshortening term (dot(wo, n) = wo.z).
*/
interface IBxDF
{
    /** Evaluates the BxDF.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns f(wi, wo) * dot(wo, n).
    */
    float3 eval(const float3 wi, const float3 wo);

    /** Samples the BxDF.
        \param[in] wi Incident direction.
        \param[out] wo Outgoing direction.
        \param[out] pdf pdf with respect to solid angle for sampling outgoing direction wo (0 if a delta event is sampled).
        \param[out] weight Sample weight f(wi, wo) * dot(wo, n) / pdf(wo).
        \param[out] lobe Sampled lobe (see LobeType).
        \param[in,out] sg Sample generator.
        \return Returns true if successful.
    */
    bool sample(const float3 wi, out float3 wo, out float pdf, out float3 weight, out uint lobe, inout SampleGenerator sg);

    /** Evaluates the BxDF directional pdf for sampling outgoing direction wo.
        \param[in] wi Incident direction.
        \param[in] wo Outgoing direction.
        \return Returns the pdf with respect to solid angle for sampling outgoing direction wo (0 for delta events).
    */
    float evalPdf(const float3 wi, const float3 wo);
};