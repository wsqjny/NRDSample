#pragma once

/** Types of light sources. Used in LightData structure.
*/
//enum class LightType : uint32_t
//{
static const uint LightType_Point = 0;          ///< Point light source, can be a spot light if its opening angle is < 2pi
static const uint LightType_Directional = 1;    ///< Directional light source
static const uint LightType_Distant = 2;        ///< Distant light that subtends a non-zero solid angle
static const uint LightType_Rect = 3;           ///< Quad shaped area light source
static const uint LightType_Disc= 4;           ///< Disc shaped area light source
static const uint LightType_Sphere = 5;         ///< Spherical area light source
//};

/** This is a host/device structure that describes analytic light sources.
*/
struct LightData
{
    float3   posW;// = float3(0, 0, 0);            ///< World-space position of the center of a light source
    uint32_t type;// = uint(LightType_Point);      ///< Type of the light source (see above)
    float3   dirW;// = float3(0, -1, 0);           ///< World-space orientation of the light source (normalized).
    float    openingAngle;// = float(M_PI);        ///< For point (spot) light: Opening half-angle of a spot light cut-off, pi by default (full sphere).
    float3   intensity;// = float3(1, 1, 1);       ///< Emitted radiance of th light source
    float    cosOpeningAngle;// = -1.f;            ///< For point (spot) light: cos(openingAngle), -1 by default because openingAngle is pi by default
    float    cosSubtendedAngle;// = 0.9999893f;    ///< For distant light; cosine of the half-angle subtended by the light. Default corresponds to the sun as viewed from earth
    float    penumbraAngle;// = 0.f;               ///< For point (spot) light: Opening half-angle of penumbra region in radians, usually does not exceed openingAngle. 0.f by default, meaning a spot light with hard cut-off
    float2   _pad0;

    // Extra parameters for analytic area lights
    float3   tangent;// = float3(0);               ///< Tangent vector of the light shape
    float    surfaceArea;// = 0.f;                 ///< Surface area of the light shape
    float3   bitangent;// = float3(0);             ///< Bitangent vector of the light shape
    float    _pad1;//;
    float4x4 transMat;// = {};                     ///< Transformation matrix of the light shape, from local to world space.
    float4x4 transMatIT;// = {};                   ///< Inverse-transpose of transformation matrix of the light shape
};