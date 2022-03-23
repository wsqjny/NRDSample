#pragma once

// see MaterialFactory.slang
ShadingData loadShadingData(GeometryProps geometryProps)
{
	ShadingData sd;
	sd.posW = geometryProps.X;
	sd.V = -geometryProps.rayDirection;

	sd.N = geometryProps.N;
	sd.faceN = geometryProps.faceN;
	sd.frontFacing = sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;

	// Assume the default IoR for vacuum on the front-facing side.
	// The renderer may override this for nested dielectrics.
	sd.IoR = 1.0f;

	sd.mtlActiveLobe = LobeType_All;								// this will chagne when disableCaustics opt is enable.
	
	return sd;
}