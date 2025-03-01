#pragma once

#include <algorithm>

#include <curand_kernel.h>

#include "math/linalg.cuh"
#include "math/geometry.cuh"

namespace gph
{

/**
 * @brief Clamps a value between minVal and maxVal
 * 
 * @param value the value.
 * @param minVal the min value.
 * @param maxVal the max value.
 * @return float 
 */
__device__ float clamp(float value, float minVal, float maxVal);

/**
 * @brief Constructs an orthonormal system from N.
 * 
 * @param N the normal vector.
 * @param tangent the tangent vector.
 * @param bitangent the bitangent vector.
 * @return __device__ 
 */
__device__ void orthonormalBasis(vec3<float> N, vec3<float>& tangent, vec3<float>& bitangent);

/**
 * @brief Microfacet-BRDF hemisphere sampling using GGX distribution function.
 * 
 * @param N the normal vector.
 * @param V the outoging direction.
 * @param roughness the roughness of the material.
 * @param state CUDA rand state.
 * @return vec3<float> 
 */
__device__ vec3<float> sampleGGX(vec3<float> N, vec3<float> V, float roughness, curandState& state);

/**
 * @brief Distribution GGX.
 * 
 * @param N the normal vector.
 * @param H the halfway vector.
 * @param roughness the roughness of the material.
 * @return float 
 */
__device__ float distributionGGX(vec3<float> N, vec3<float> H, float roughness);

/**
 * @brief Geometry SchlickGGX.
 * 
 * @param NdotV the dot product between the normal and the outgoing direction.
 * @param roughness the roughness of the material.
 * @return float 
 */
__device__ float geometrySchlickGGX(float NdotV, float roughness);

/**
 * @brief Geometry Smith function.
 * 
 * @param N the normal vector.
 * @param V the outgoing direction.
 * @param L the incoming direction.
 * @param roughness the roughness of the material.
 * @return float 
 */
__device__ float geometrySmith(vec3<float> N, vec3<float> V, vec3<float> L, float roughness);

/**
 * @brief Fresnel Schlick approximation.
 * 
 * @param cosTheta cos theta.
 * @param F0 F0.
 * @return vec3<float> 
 */
__device__ vec3<float> fresnelSchlick(float cosTheta, vec3<float> F0);

/**
 * @brief Cook-Torrance BRDF.
 * 
 * @param H the halfway vector.
 * @param normal the normal vector.
 * @param wo the outgoing direction.
 * @param wi the incoming direction.
 * @param F Fresnel.
 * @param roughness the roughness of the material.
 * @return __device__ 
 */
__device__ vec3<float> specularCookTorrance(vec3<float> H, vec3<float> normal, vec3<float> wo, vec3<float> wi, vec3<float> F, float roughness);

}