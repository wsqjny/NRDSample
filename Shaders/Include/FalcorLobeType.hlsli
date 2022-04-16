/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
//#include "Utils/HostDeviceShared.slangh"

//BEGIN_NAMESPACE_FALCOR

/** Flags representing the various lobes of a BxDF.
*/
// TODO: Specify the backing type when Slang issue has been resolved
//enum class LobeType // : uint32_t
//{
//    None                    = 0x00,
//
//    DiffuseReflection       = 0x01,
//    SpecularReflection      = 0x02,
//    DeltaReflection         = 0x04,
//
//    DiffuseTransmission     = 0x10,
//    SpecularTransmission    = 0x20,
//    DeltaTransmission       = 0x40,
//
//    Diffuse                 = 0x11,
//    Specular                = 0x22,
//    Delta                   = 0x44,
//    NonDelta                = 0x33,
//
//    Reflection              = 0x0f,
//    Transmission            = 0xf0,
//
//    NonDeltaReflection      = 0x03,
//    NonDeltaTransmission    = 0x30,
//
//    All                     = 0xff,
//};

//END_NAMESPACE_FALCOR


static uint LobeType_DiffuseReflection			= 0x01;
static uint LobeType_SpecularReflection			= 0x02;
static uint LobeType_DeltaReflection			= 0x04;

static uint LobeType_DiffuseTransmission		= 0x10;
static uint LobeType_SpecularTransmission		= 0x20;
static uint LobeType_DeltaTransmission			= 0x40;

static uint LobeType_Diffuse					= 0x11;
static uint LobeType_Specular					= 0x22;
static uint LobeType_Delta						= 0x44;
static uint LobeType_NonDelta					= 0x33;

static uint LobeType_Reflection					= 0x0fl;
static uint LobeType_Transmission				= 0xf0;

static uint LobeType_NonDeltaReflection			= 0x03;
static uint LobeType_NonDeltaTransmission		= 0x30;

static uint LobeType_All						= 0xff;