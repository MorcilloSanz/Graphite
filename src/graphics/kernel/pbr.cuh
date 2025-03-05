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
 * @brief Reflects a vector `wo` across a given normal vector.
 * 
 * This function computes the reflection of an incident vector `wo` relative to a normal vector `normal`.
 * The formula used is: 
 *      reflected = wo - 2 * dot(wo, normal) * normal
 * where `dot(wo, normal)` is the dot product of the incident vector and the normal vector.
 * This is commonly used in shading and ray tracing to simulate reflection of light on surfaces.
 * 
 * @param wo The incident vector (e.g., the outgoing ray direction or the view direction).
 * @param normal The surface normal vector to reflect the incident vector against.
 * @return vec3<float>
 */
__device__ vec3<float> reflect(vec3<float> wo, vec3<float> normal);

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

/**
 * @brief Monte Carlo GGX BRDF estimator. IMPORTANT: monteCarloGGX = (BRDF * wi.dot(normal)) / PDF
 * 
 * Take a look at:
 * https://jcgt.org/published/0007/04/01/paper.pdf
 * 
 * @param H 
 * @param normal 
 * @param wo 
 * @param wi 
 * @param F 
 * @param roughness 
 * @return __device__ 
 */
__device__ vec3<float> monteCarloGGX(vec3<float> H, vec3<float> normal, vec3<float> wo, vec3<float> wi, vec3<float> F0, float roughness);

}