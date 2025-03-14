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
__device__ vec3<float> reflect(const vec3<float>& wo, const vec3<float>& normal);

/**
 * @brief Computes the refraction direction using Snell's law.
 *
 * This function calculates the refracted direction given an incident direction, 
 * a surface normal, and the relative index of refraction (eta).
 *
 * @param wo The incident direction vector (normalized).
 * @param normal The surface normal vector (normalized).
 * @param eta The ratio of the indices of refraction (η = IOR_in / IOR_out).
 * @return The refracted direction vector. If total internal reflection occurs, returns a zero vector.
 *
 * @note Ensure that `wo` and `normal` are normalized before calling this function.
 * @note The function assumes that `wo` points away from the surface.
 */
__device__ vec3<float> refract(const vec3<float>& wo, const vec3<float>& normal, float eta);

/**
 * @brief Distribution GGX.
 * 
 * @param N the normal vector.
 * @param H the halfway vector.
 * @param roughness the roughness of the material.
 * @return float 
 */
__device__ float distributionGGX(const vec3<float>& N, const vec3<float>& H, float roughness);

/**
 * @brief Fresnel Schlick approximation.
 * 
 * @param cosTheta cos theta.
 * @param F0 F0.
 * @return vec3<float> 
 */
__device__ vec3<float> fresnelSchlick(float cosTheta, const vec3<float>& F0);

/**
 * @brief Sampling hemisphere normals using a GGX Distribution .
 * 
 * Take a look at:
 * https://jcgt.org/published/0007/04/01/paper.pdf
 * 
 * @param wo 
 * @param roughness 
 * @param U1 
 * @param U2 
 * @return vec3<float> 
 */
__device__ vec3<float> sampleGGXVNDF(
    const vec3<float>& wo, 
    float roughness, 
    float U1, 
    float U2
);

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
 * @return vec3<float> 
 */
__device__ vec3<float> monteCarloGGX(
    const vec3<float>& normal, 
    const vec3<float>& wo, 
    const vec3<float>& wi, 
    const vec3<float>& F0, 
    float roughness
);

}