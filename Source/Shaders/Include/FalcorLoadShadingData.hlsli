#pragma once

// see MaterialFactory.slang
ShadingData loadShadingData(FalcorPayload payload)
{
	ShadingData sd;
	sd.posW = payload.X;
	sd.V = -payload.rayDirection;

	sd.N = payload.N;
	sd.faceN = payload.faceN;
	sd.frontFacing = sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;

	// Assume the default IoR for vacuum on the front-facing side.
	// The renderer may override this for nested dielectrics.
	sd.IoR = 1.0f;

	sd.mtlActiveLobe = LobeType_All;								// this will chagne when disableCaustics opt is enable.
	
	return sd;
}