#pragma once


/** Computes new ray origin based on hit position to avoid self-intersections.
    The function assumes that the hit position has been computed by barycentric
    interpolation, and not from the ray t which is less accurate.

    The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
    Method for Avoiding Self-Intersection" by Carsten W?chter and Nikolaus Binder.

    \param[in] pos Ray hit position.
    \param[in] normal Face normal of hit surface (normalized). The offset will be in the positive direction.
    \return Ray origin of the new ray.
*/
float3 computeRayOrigin(float3 pos, float3 normal)
{
    const float origin = 1.f / 32.f;
    const float fScale = 1.f / 65536.f;
    const float iScale = 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + (pos < 0.f ? -iOff : iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    return abs(pos) < origin ? pos + fOff : iPos;
}





/** This struct holds information needed for shading a hit point.

    This includes:
    - Geometric properties of the surface.
    - Texture coordinates.
    - Material ID and header.
    - Opacity value for alpha testing.
    - Index of refraction of the surrounding medium.

    Based on a ShadingData struct, the material system can be queried
    for a BSDF instance at the hit using `gScene.materials.getBSDF()`.
    The BSDF has interfaces for sampling and evaluation, as well as for
    querying its properties at the hit.
*/
struct ShadingData
{
    // Geometry data
    float3  posW;                   ///< Shading hit position in world space.
    float3  V;                      ///< Direction to the eye at shading hit.
    float3  N;                      ///< Shading normal at shading hit.
    //float3  T;                      ///< Shading tangent at shading hit.
    //float3  B;                      ///< Shading bitangent at shading hit.
    //float2  uv;                     ///< Texture mapping coordinates.
    float3  faceN;                  ///< Face normal in world space, always on the front-facing side.
    bool    frontFacing;            ///< True if primitive seen from the front-facing side.
    //float   curveSphereRadius;      ///< Sphere radius for curve hits.

    // Material data
    // MaterialHeader mtl;             ///< Material header data.
    //uint    materialID;             ///< Material ID at shading location.    
    //float   opacity;                ///< Opacity value in [0,1]. This is used for alpha testing.
    float   IoR;                        ///< Index of refraction for the medium on the front-facing side (i.e. "outside" the material at the hit). 
    uint mtlActiveLobe;                 ///< JNChange : 


    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten W?chter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        //return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
        return computeRayOrigin(posW, viewside ? N : -N);
    }

    /** Transform vector from the local surface frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        //return T * v.x + B * v.y + N * v.z;

        float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
        return STL::Geometry::RotateVectorInverse(mLocalBasis, v);                     // local to world
    }

    /** Transform vector from world space to the local surface frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        //return float3(dot(v, T), dot(v, B), dot(v, N));

        float3x3 mLocalBasis = STL::Geometry::GetBasis(N);
        return STL::Geometry::RotateVector(mLocalBasis, v);                                   // world to local        
    }
};




