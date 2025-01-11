#pragma once

#include <cmath>

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"
#include "graphics/material.cuh"

#include "kernel.cuh"
#include "attributes.cuh"

namespace gph 
{

__device__ vec3<float> getBarycentricInterpolation3(KernelFragmentParams params, int i, vec3<float> barycentricCoords, int attribute) {

    vec3<float> A1 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute);     // v1x v2x v3x
    vec3<float> A2 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 1); // v1y v2y v3y
    vec3<float> A3 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 2); // v1z v2z v3z

    vec3<float> a1 = { A1.x, A2.x, A3.x };
    vec3<float> a2 = { A1.y, A2.y, A3.y };
    vec3<float> a3 = { A1.z, A2.z, A3.z };

    return a1 * barycentricCoords.x + a2 * barycentricCoords.y + a3 * barycentricCoords.z;
}

__device__ vec2<float> getBarycentricInterpolation2(KernelFragmentParams params, int i, vec3<float> barycentricCoords, int attribute) {

    vec3<float> A1 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute);     // uv1x uv2x uv3x
    vec3<float> A2 = getAttributes3(params.vertexBuffer.buffer, params.indexBuffer.buffer, i, attribute + 1); // uv1y uv2y uv3y

    vec2<float> a1 = { A1.x, A2.x };
    vec2<float> a2 = { A1.y, A2.y };
    vec2<float> a3 = { A1.z, A2.z };

    return a1 * barycentricCoords.x + a2 * barycentricCoords.y + a3 * barycentricCoords.z;
}

__device__ vec3<float> getBarycentricColor(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3(params, i, barycentricCoords, ATTRIBUTE_R);
}

__device__ vec3<float> getBarycentricNormal(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation3(params, i, barycentricCoords, ATTRIBUTE_NX).normalize();
}

__device__ vec2<float> getBarycentricUVs(KernelFragmentParams params, int i, vec3<float> barycentricCoords) {
    return getBarycentricInterpolation2(params, i, barycentricCoords, ATTRIBUTE_UVX);
}

__device__ vec2<float> getSkyUVs(Ray<float> ray) {

    float theta = acosf(fmaxf(-1.0f, fminf(ray.direction.y, 1.0f)));
    float phi = atan2f(ray.direction.z, ray.direction.x);

    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = 1.0f - (theta / M_PI);

    return { u, v };
}

__device__ vec3<float> tex(cudaTextureObject_t texObj, float u, float v) {
    float4 texValue = tex2D<float4>(texObj, u, v);
    return vec3<float>(texValue.x, texValue.y, texValue.z);
}

__device__ void setPixel(uint8_t* frameBuffer, int x, int y, int width, const vec3<unsigned char>& color) {
    frameBuffer[3 * (x + y * width)    ] = color.r;
    frameBuffer[3 * (x + y * width) + 1] = color.g;
    frameBuffer[3 * (x + y * width) + 2] = color.b;
}

__global__ void kernel_fragment(KernelFragmentParams params) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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
            vec3<float> n = getBarycentricNormal(params, i, barycentricCoords); // Not used for the moment
            vec2<float> uvs = getBarycentricUVs(params, i, barycentricCoords);

            vec3<float> lightDirection = vec3<float>(-0.5f, 1.0f, -1.f).normalize();
            float intensity = max(0.f, lightDirection.dot(hitInfo.normal * -1));

            vec3<float> lightColor = vec3<float>(1.0f, 0.9f, 0.9f);
            vec3<float> outputColor = c * intensity * lightColor;

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

}