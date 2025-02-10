#pragma once

#include <cmath>

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"
#include "graphics/material.cuh"

#include "kernel.cuh"
#include "attributes.cuh"
#include "pbr.cuh"

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

/**
 * @brief The program which solves the rendering equation computing the output radiance in the eye direction.
 * 
 * @param params KernelFragmentParams struct.
 * @param x the x coordinate.
 * @param y the y coordinate.
 * @return void 
 */
__device__ void program(KernelFragmentParams params, int x, int y) {

    if (x >= params.frameBuffer.width || y >= params.frameBuffer.height) {
        return;
    }

    // Ray casting
    Ray<float> ray = Ray<float>::castRayPerspective(x, y, params.frameBuffer.width, params.frameBuffer.height, 60);
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

            vec3<float> c = getBarycentricColor(params, i, barycentricCoords);  // Color interpolation
            vec3<float> n = getBarycentricNormal(params, i, barycentricCoords); // Normal interpolation
            vec2<float> uvs = getBarycentricUVs(params, i, barycentricCoords);  // UVs interpolation

            hitInfo.normal = n; // Consider interpolated normal instead of the actual normal of the triangle

            vec3<float> albedo, metallicRoughness, normal, ambientOcclusion, emission;
            if(params.materialsCount > 0) {

                if(params.materials[materialIndex].albedo.hasTexture)
                    albedo = tex(params.materials[materialIndex].albedo.texture, uvs.u, uvs.v);
                
                if(params.materials[materialIndex].metallicRoughness.hasTexture)
                    metallicRoughness = tex(params.materials[materialIndex].metallicRoughness.texture, uvs.u, uvs.v);

                if(params.materials[materialIndex].normal.hasTexture)
                    normal = tex(params.materials[materialIndex].normal.texture, uvs.u, uvs.v);

                if(params.materials[materialIndex].ambientOcclusion.hasTexture)
                    ambientOcclusion = tex(params.materials[materialIndex].ambientOcclusion.texture, uvs.u, uvs.v);

                if(params.materials[materialIndex].emission.hasTexture)
                    emission = tex(params.materials[materialIndex].emission.texture, uvs.u, uvs.v);

                c = emission + c * albedo;
            }

            vec3<float> lightColor = vec3<float>(1.5f, 1.5f, 1.4f);
            vec3<float> lightDirection = vec3<float>(0.5f, 0.75f, -1.f).normalize();

            // Rendering equation
            vec3<float> Li = lightColor;
            vec3<float> wi = lightDirection.normalize();

            vec3<float> wo = ray.direction.normalize() * -1;
            vec3<float> H = (wi + wo).normalize();

            // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
            // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow) 
            float metallic = metallicRoughness.b;
            vec3<float> F0 = vec3<float>(0.04f);
            F0 = lerp<float>(F0, albedo, metallic);
            vec3<float> F = fresnelSchlick(max(H.dot(wo), 0.0), F0);
            
            // Diffuse
            vec3<float> fLambert = albedo / M_PI;
            
            // Specular
            float roughness = metallicRoughness.g;
            vec3<float> specular = specularCookTorrance(H, hitInfo.normal, wo, wi, F, roughness);
            
            // Energy ratios
            vec3<float> kS = F;
            vec3<float> kD = vec3<float>(1.0f) - kS;

            // BRDF
            vec3<float> fr = kD * fLambert * ambientOcclusion + specular;
            vec3<float> Lo = emission + fr * Li * max(0.f, lightDirection.dot(hitInfo.normal * -1));
            
            // HDR (Reinhard tone mapping)
            vec3<float> outputColor = Lo / (vec3<float>(1.0f) + Lo);  // Reinhard Tone Mapping

            // Gamma correction
            float gamma = 2.2;
            float power = 1.0 / gamma;
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

    // Miss function
    if(missed) {

        vec3<unsigned char> pixelColor(0, 0, 0);

        if(params.sky.hasTexture) {

            vec2<float> uvs = getSkyUVs(ray);
            vec3<float> sky = tex(params.sky.texture, uvs.u, uvs.v);

            pixelColor = { static_cast<unsigned char>(sky.r * 255), static_cast<unsigned char>(sky.g * 255), static_cast<unsigned char>(sky.b * 255) };
        }

        setPixel(params.frameBuffer.buffer, x, y, params.frameBuffer.width, pixelColor);
    }
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

    program(params, x, y);
}

}