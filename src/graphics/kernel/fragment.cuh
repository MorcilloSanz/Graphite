#pragma once

#include <cmath>
#include <cstdlib>

#include <curand_kernel.h>

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"
#include "graphics/material.cuh"

#include "kernel.cuh"
#include "attributes.cuh"
#include "pbr.cuh"

#define SEED 1234
#define GAMMA 2.2

namespace gph 
{

template <typename T>
__device__ vec3<T> lerp(vec3<T> u, vec3<T> v, T t) {
    return u + (v - u) * t;
}

/**
 * @brief Get the barycentric interpolation (vec3) of a point inside a triangle.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @param attribute the attribute associated with the vertex.
 * @return vec3<float>
 */
template <typename T>
__device__ vec3<T> getBarycentricInterpolation3(KernelFragmentParams params, int i, vec3<T> barycentricCoords, int attribute) {

    vec3<T> A1 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute);     // v1x v2x v3x
    vec3<T> A2 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 1); // v1y v2y v3y
    vec3<T> A3 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 2); // v1z v2z v3z

    vec3<T> a1 = { A1.x, A2.x, A3.x };
    vec3<T> a2 = { A1.y, A2.y, A3.y };
    vec3<T> a3 = { A1.z, A2.z, A3.z };

    return a1 * barycentricCoords.x + a2 * barycentricCoords.y + a3 * barycentricCoords.z;
}

/**
 * @brief Get the barycentric interpolation (vec2) of a point inside a triangle.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @param attribute the attribute associated with the vertex.
 * @return vec2<float>
 */
template <typename T>
__device__ vec2<T> getBarycentricInterpolation2(KernelFragmentParams params, int i, vec3<T> barycentricCoords, int attribute) {

    vec3<T> A1 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute);     // uv1x uv2x uv3x
    vec3<T> A2 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 1); // uv1y uv2y uv3y

    vec2<T> a1 = { A1.x, A2.x };
    vec2<T> a2 = { A1.y, A2.y };
    vec2<T> a3 = { A1.z, A2.z };

    return a1 * barycentricCoords.x + a2 * barycentricCoords.y + a3 * barycentricCoords.z;
}

/**
 * @brief Get the corresponding color of a point inside a triangle given the barycentric coordinates.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @return vec3<float> 
 */
__device__ vec3<float> getBarycentricColor(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3<float>(params, i, barycentricCoords, ATTRIBUTE_R);
}

/**
 * @brief Get the corresponding normal of a point inside a triangle given the barycentric coordinates.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @return vec3<float> 
 */
__device__ vec3<float> getBarycentricNormal(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3<float>(params, i, barycentricCoords, ATTRIBUTE_NX).normalize();
}

/**
 * @brief Get the corresponding UVs of a point inside a triangle given the barycentric coordinates.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @return vec2<float> 
 */
__device__ vec2<float> getBarycentricUVs(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation2<float>(params, i, barycentricCoords, ATTRIBUTE_UVX);
}

/**
 * @brief Get the corresponding tangent of a point inside a triangle given the barycentric coordinates.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @return vec3<float> 
 */
__device__ vec3<float> getBarycentricTangent(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3<float>(params, i, barycentricCoords, ATTRIBUTE_TANX).normalize();
}

/**
 * @brief Get the corresponding bitangent of a point inside a triangle given the barycentric coordinates.
 * 
 * @param params KernelFragmentParams struct.
 * @param i index of the indices of the IndexBuffer.
 * @param barycentricCoords the barycentric coordinates of the triangle.
 * @return vec3<float> 
 */
__device__ vec3<float> getBarycentricBitangent(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3<float>(params, i, barycentricCoords, ATTRIBUTE_BITANX).normalize();
}

__device__ vec3<float> reflect(vec3<float> wo, vec3<float> normal) {
    float dot = normal.dot(wo);
    vec3<float> wi = wo - normal * 2.f * dot;
    return wi.normalize();
}

/**
 * @brief Computes the corresponding UVs of an image mapped to a sphere depending on the ray direction.
 * 
 * @param ray the ray.
 * @return vec2<float>
 */
__device__ vec2<float> getSkyUVs(Ray<float> ray) {

    float theta = acosf(fmaxf(-1.0f, fminf(ray.direction.y, 1.0f)));
    float phi = atan2f(ray.direction.z, ray.direction.x);

    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = 1.0f - (theta / M_PI);

    return { u, v };
}

__device__ vec3<float> perpendicular(vec3<float> v) {
    if (fabs(v.x) > fabs(v.z))
        return vec3<float>(-v.y, v.x, 0.0f).normalize();
    else
        return vec3<float>(0.0f, -v.z, v.y).normalize();
}

__device__ vec3<float> sampleHemisphereCosine(float u1, float u2, vec3<float> normal) {
    float r = sqrt(u1);
    float theta = 2.0f * M_PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u1));

    vec3<float> tangent = perpendicular(normal).normalize();
    vec3<float> bitangent = normal.cross(tangent);

    return (tangent * x + bitangent * y + normal * z).normalize();
}

__device__ vec3<float> toLocalFrame(float x, float y, float z, vec3<float> normal) {
    // Construcción de un sistema ortonormal (TBN)
    vec3<float> up = (fabs(normal.z) > 0.999f) ? vec3<float>(1, 0, 0) : vec3<float>(0, 0, 1);
    vec3<float> tangent = up.cross(normal).normalize();
    vec3<float> bitangent = normal.cross(tangent).normalize();

    // Transformación del vector (x, y, z) desde el espacio local al global
    return (tangent * x) + (bitangent * y) + (normal * z);
}

// Genera una dirección de reflexión especular siguiendo la distribución GGX
__device__ vec3<float> sampleGGX(float u1, float u2, float alpha, vec3<float> normal, vec3<float> wo) {
    float phi = 2.0f * M_PI * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

    vec3<float> H = toLocalFrame(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta, normal);
    vec3<float> wi = reflect(wo * -1, H);

    return (wi.dot(normal) > 0.f) ? wi : wi * -1;
}

__device__ vec3<float> generateWi(vec3<float> wo, Ray<float>::HitInfo hitInfo, vec3<float> metallicRoughness, curandState& randState) {
    float u1 = curand_uniform(&randState);
    float u2 = curand_uniform(&randState);

    vec3<float> diffuseDir = sampleHemisphereCosine(u1, u2, hitInfo.normal);
    vec3<float> specularDir = sampleGGX(u1, u2, metallicRoughness.g * metallicRoughness.g, hitInfo.normal, wo);

    float metallic = metallicRoughness.b;
    float roughness = metallicRoughness.g;

    // Mezcla entre los dos métodos según el metalizado
    float mixFactor = (1.0f - roughness) * metallic;
    vec3<float> wi = lerp(diffuseDir, specularDir, mixFactor).normalize();

    return (wi.dot(hitInfo.normal) > 0.f) ? wi : wi * -1;
}

/**
 * @brief Returns the value of a texture for a given UVs.
 * 
 * @param texObj cudaTextureObject_t of the texture.
 * @param u the u coordinate.
 * @param v the v coordinate.
 * @return vec3<float> 
 */
__device__ vec3<float> tex(cudaTextureObject_t texObj, float u, float v) {
    float4 texValue = tex2D<float4>(texObj, u, v);
    return vec3<float>(texValue.x, texValue.y, texValue.z);
}

/**
 * @brief Sets the pixel color of a pixel in the FrameBuffer.
 * 
 * @param frameBuffer the FrameBuffer that contains the final image.
 * @param x the x coordinate.
 * @param y the y coordinate.
 * @param width the with of the FrameBuffer image.
 * @param color the color of the pixel
 * @return void 
 */
__device__ void setPixel(uint8_t* frameBuffer, int x, int y, int width, const vec3<unsigned char>& color) {
    frameBuffer[3 * (x + y * width)    ] = color.r;
    frameBuffer[3 * (x + y * width) + 1] = color.g;
    frameBuffer[3 * (x + y * width) + 2] = color.b;
}

__device__ vec3<float> missFunction(KernelFragmentParams params, Ray<float> ray) {

    vec2<float> uvs = getSkyUVs(ray);
    vec3<float> sky = tex(params.sky.texture, uvs.u, uvs.v);

    return sky;
}

/**
 * @brief The program which solves the rendering equation computing the output radiance in the eye direction.
 * 
 * @param params KernelFragmentParams struct.
 * @param x the x coordinate.
 * @param y the y coordinate.
 * @return void 
 */
__device__ vec3<float> castRay(KernelFragmentParams params, Ray<float> ray, int samples, int bounces, curandState randState) {

    vec3<float> Lo;
    float distance = INFINITY;
    bool missed = true;

    // Ray intersections
    int count = params.indexBuffer.size / sizeof(unsigned int);
    for(int i = 0; i < count; i += 3) {

        unsigned int materialIndex = static_cast<unsigned int>(getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, ATTRIBUTE_MATERIAL_INDEX));

        vec3<float> X = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, ATTRIBUTE_X); // v1x v2x v3x
        vec3<float> Y = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, ATTRIBUTE_Y); // v1y v2y v3y
        vec3<float> Z = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, ATTRIBUTE_Z); // v1z v2z v3z

        vec3<float> v1 = { X.x, Y.x, Z.x };
        vec3<float> v2 = { X.y, Y.y, Z.y };
        vec3<float> v3 = { X.z, Y.z, Z.z };

        Triangle<float> triangle (v1, v2, v3);
        Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

        if(hitInfo.hit && hitInfo.distance < distance) {

            missed = false;
            distance = hitInfo.distance;

            vec3<float> barycentricCoords = barycentric<float>(hitInfo.intersection, triangle);

            vec3<float> c = getBarycentricColor(params, i, barycentricCoords);
            vec3<float> n = getBarycentricNormal(params, i, barycentricCoords).normalize();
            vec2<float> uvs = getBarycentricUVs(params, i, barycentricCoords);
            vec3<float> tan = getBarycentricTangent(params, i, barycentricCoords).normalize();
            vec3<float> bitan = getBarycentricBitangent(params, i, barycentricCoords).normalize();

            // Compute TBN matrix: transforms from tangent space to world space.
            mat3<float> TBN;
            TBN.row1 = { tan.x, bitan.x, n.x };
            TBN.row2 = { tan.y, bitan.y, n.y };
            TBN.row3 = { tan.z, bitan.z, n.z };

            hitInfo.normal = n; // Consider interpolated normal instead of the actual normal of the triangle

            vec3<float> albedo, metallicRoughness, normal, ambientOcclusion, emission;
            if(params.materialsCount > 0) {

                if(params.materials[materialIndex].albedo.hasTexture) {
                    albedo = tex(params.materials[materialIndex].albedo.texture, uvs.u, uvs.v);
                    albedo = { pow(albedo.r, GAMMA), pow(albedo.g, GAMMA), pow(albedo.b, GAMMA) };  // Convert from sRGB to lineal
                }
                    
                if(params.materials[materialIndex].metallicRoughness.hasTexture)
                    metallicRoughness = tex(params.materials[materialIndex].metallicRoughness.texture, uvs.u, uvs.v);

                if(params.materials[materialIndex].normal.hasTexture && tan.x != 0 && tan.y != 0 && tan.z != 0) {
                    normal = tex(params.materials[materialIndex].normal.texture, uvs.u, uvs.v);
                    normal = normal * 2.0 - 1.0f;
                    normal = TBN.transform(normal).normalize();
                    hitInfo.normal = normal; // Consider normal from normal mapping
                }

                if(params.materials[materialIndex].ambientOcclusion.hasTexture)
                    ambientOcclusion = tex(params.materials[materialIndex].ambientOcclusion.texture, uvs.u, uvs.v);

                if(params.materials[materialIndex].emission.hasTexture) {
                    emission = tex(params.materials[materialIndex].emission.texture, uvs.u, uvs.v);
                    emission = { pow(emission.r, GAMMA), pow(emission.g, GAMMA), pow(emission.b, GAMMA) }; // Convert from sRGB to lineal
                }

                c = emission + c * albedo;
            }

            vec3<float> wo = ray.direction.normalize() * -1;

            vec3<float> integral(0.0f);
            for(int i = 0; i < samples; i ++) {

                // Compute new direction
                float u1 = curand_uniform(&randState);
                float u2 = curand_uniform(&randState);

                vec3<float> diffuseDir = sampleHemisphereCosine(u1, u2, hitInfo.normal);
                vec3<float> specularDir = reflect(wo, hitInfo.normal);

                float metallic = metallicRoughness.b;
                float roughness = metallicRoughness.g;

                vec3<float> wi = lerp(diffuseDir, specularDir, 1.f - roughness).normalize();
                if(wi.dot(hitInfo.normal) < 0.f) wi = wi * -1;

                wi = generateWi(wo, hitInfo, metallicRoughness, randState);
                wi = vec3<float>(0.5f, -0.75f, -1.f).normalize() * -1;

                // Rendering equation
                vec3<float> Li(0.0);

                const float epsilon = 1e-4;
                Ray<float> newRay(hitInfo.intersection + hitInfo.normal * epsilon, wi); // Desplazar un poco el origen para que no intersecte con si mismo

                if(bounces > 0) {
                    Li = castRay(params, newRay, samples, bounces - 1, randState);
                }else {
                    if(params.sky.hasTexture)
                        Li = missFunction(params, newRay);
                }

                //wi = vec3<float>(0.5f, -0.75f, -1.f).normalize() * -1;

                vec3<float> H = (wi + wo).normalize();
                // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
                // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow) 
                vec3<float> F0 = vec3<float>(0.04f);
                F0 = lerp<float>(F0, albedo, metallic);
                vec3<float> F = fresnelSchlick(max(H.dot(wo), 0.0), F0);
                
                // Diffuse
                vec3<float> fLambert = albedo / M_PI;
                
                // Specular
                vec3<float> specular = specularCookTorrance(H, hitInfo.normal, wo, wi, F, roughness);
                
                // Energy ratios
                vec3<float> kS = F;
                vec3<float> kD = vec3<float>(1.0f) - kS;

                // BRDF
                vec3<float> fr = kD * fLambert * ambientOcclusion + specular;

                // Integral
                integral = integral + fr * Li * max(0.f, wi.dot(hitInfo.normal));
            }

            // Rendering equation
            Lo = emission + integral * (2.f * M_PI / samples);
        }
    }

    // Miss function
    if(missed) {
        if(params.sky.hasTexture)
            Lo = missFunction(params, ray);
    }

    return Lo;
}

/**
 * @brief the CUDA kernel that executes the program for each pixel of the FrameBuffer image.
 * 
 * @param params KernelFragmentParams struct.
 * @return void 
 */
__global__ void kernel_fragment(KernelFragmentParams params) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= params.frameBuffer.width || y >= params.frameBuffer.height)
        return;

    // Inicializar el estado de curand con una semilla única por píxel
    curandState randState;
    int pixelIndex = y * params.frameBuffer.width + x;
    curand_init(SEED, pixelIndex, 0, &randState);

    // Compute outgoing radiance
    const int samples = 100;
    const int bounces = 1;

    Ray<float> ray = Ray<float>::castRayPerspective(x, y, params.frameBuffer.width, params.frameBuffer.height, 60);
    vec3<float> outputColor = castRay(params, ray, samples, bounces, randState);

    // HDR (Reinhard tone mapping)
    outputColor = outputColor / (vec3<float>(1.0f) + outputColor);  // Reinhard Tone Mapping

    // Gamma correction
    float power = 1.0 / GAMMA;
    outputColor = { pow(outputColor.r, power), pow(outputColor.g, power), pow(outputColor.b, power) };

    // Write pixel into frame buffer
    vec3<unsigned char> pixelColor = {
        static_cast<unsigned char>(outputColor.x * 255),
        static_cast<unsigned char>(outputColor.y * 255),
        static_cast<unsigned char>(outputColor.z * 255),
    };

    setPixel(params.frameBuffer.buffer, x, y, params.frameBuffer.width, pixelColor);
}

}