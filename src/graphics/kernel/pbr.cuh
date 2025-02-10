#pragma once

#include <algorithm>

#include "math/linalg.cuh"

namespace gph
{

__device__ float clamp(float value, float minVal, float maxVal);

__device__ float distributionGGX(vec3<float> N, vec3<float> H, float roughness);

__device__ float geometrySchlickGGX(float NdotV, float roughness);

__device__ float geometrySmith(vec3<float> N, vec3<float> V, vec3<float> L, float roughness);

__device__ vec3<float> fresnelSchlick(float cosTheta, vec3<float> F0);

__device__ vec3<float> specularCookTorrance(vec3<float> H, vec3<float> normal, vec3<float> wo, vec3<float> wi, vec3<float> F, float roughness);

}