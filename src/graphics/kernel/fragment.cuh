#pragma once

#include <cmath>

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"
#include "attributes.cuh"


namespace gph 
{

struct KernelFragmentParams {

    uint8_t* frameBuffer;
    unsigned int width;
    unsigned int height;

    float* vertexBuffer;
    size_t vertexSize;
    
    unsigned int* indexBuffer;
    size_t indexSize;

    cudaTextureObject_t skyTextureObj;
    bool hasSky;
};

__device__ vec3<float> getBarycentricColor(float* vertexBuffer, unsigned int* indexBuffer, int i, vec3<float> barycentricCoords) {

    vec3<float> R = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_R); // c1r c2r c3r
    vec3<float> G = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_G); // c1g c2g c3g
    vec3<float> B = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_B); // c1b c2b c3b

    vec3<float> c1 = { R.x, G.x, B.x };
    vec3<float> c2 = { R.y, G.y, B.y };
    vec3<float> c3 = { R.z, G.z, B.z };

    return c1 * barycentricCoords.x + c2 * barycentricCoords.y + c3 * barycentricCoords.z;
}

__device__ vec3<float> getBarycentricNormal(float* vertexBuffer, unsigned int* indexBuffer, int i, vec3<float> barycentricCoords) {

    vec3<float> NX = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_NX); // n1x n2x n3x
    vec3<float> NY = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_NY); // n1y n2y n3y
    vec3<float> NZ = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_NZ); // n1z n2z n3z

    vec3<float> n1 = { NX.x, NY.x, NZ.x };
    vec3<float> n2 = { NX.y, NY.y, NZ.y };
    vec3<float> n3 = { NX.z, NY.z, NZ.z };

    vec3<float> n = n1 * barycentricCoords.x + n2 * barycentricCoords.y + n3 * barycentricCoords.z;

    return n.normalize();
}

__device__ vec2<float> getBarycentricUVs(float* vertexBuffer, unsigned int* indexBuffer, int i, vec3<float> barycentricCoords) {

    vec3<float> UVX = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_UVX); // uv1x uv2x uv3x
    vec3<float> UVY = getAttributes3(vertexBuffer, indexBuffer, i, ATTRIBUTE_UVY); // uv1y uv2y uv3y

    vec2<float> uv1 = { UVX.x, UVY.x };
    vec2<float> uv2 = { UVX.y, UVY.y };
    vec2<float> uv3 = { UVX.z, UVY.z };

    return uv1 * barycentricCoords.x + uv2 * barycentricCoords.y + uv3 * barycentricCoords.z;
}

__device__ vec2<float> getSkyUVs(Ray<float> ray) {

    float theta = acosf(fmaxf(-1.0f, fminf(ray.direction.y, 1.0f))); // Clampea el valor entre [-1, 1]
    float phi = atan2f(ray.direction.z, ray.direction.x);

    // Ajustar el rango de phi de [-π, π] a [0, 1]
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

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Ray casting
    Ray<float> ray = Ray<float>::castRayPerspective(x, y, params.width, params.height, 60);
    float distance = INFINITY;
    bool missed = true;

    // Ray intersections
    int count = params.indexSize / sizeof(unsigned int);
    for(int i = 0; i < count; i += 3) {

        vec3<float> X = getAttributes3(params.vertexBuffer, params.indexBuffer, i, ATTRIBUTE_X); // v1x v2x v3x
        vec3<float> Y = getAttributes3(params.vertexBuffer, params.indexBuffer, i, ATTRIBUTE_Y); // v1y v2y v3y
        vec3<float> Z = getAttributes3(params.vertexBuffer, params.indexBuffer, i, ATTRIBUTE_Z); // v1z v2z v3z

        vec3<float> v1 = { X.x, Y.x, Z.x };
        vec3<float> v2 = { X.y, Y.y, Z.y };
        vec3<float> v3 = { X.z, Y.z, Z.z };

        Triangle<float> triangle (v1, v2, v3);
        Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

        if(hitInfo.hit && hitInfo.distance < distance) {

            missed = false;
            distance = hitInfo.distance;

            vec3<float> barycentricCoords = barycentric<float>(hitInfo.intersection, triangle);

            vec3<float> c = getBarycentricColor(params.vertexBuffer, params.indexBuffer, i, barycentricCoords);
            vec3<float> n = getBarycentricNormal(params.vertexBuffer, params.indexBuffer, i, barycentricCoords); // Not used for the moment
            vec2<float> uvs = getBarycentricUVs(params.vertexBuffer, params.indexBuffer, i, barycentricCoords);

            vec3<float> lightDirection = vec3<float>(-0.5f, 1.0f, -1.f).normalize();
            float intensity = max(0.f, lightDirection.dot(hitInfo.normal * -1));

            vec3<float> lightColor = vec3<float>(1.0f, 0.9f, 0.9f);
            vec3<float> outputColor = c * intensity * lightColor;

            vec3<unsigned char> pixelColor = {
                static_cast<unsigned char>(outputColor.x * 255),
                static_cast<unsigned char>(outputColor.y * 255),
                static_cast<unsigned char>(outputColor.z * 255),
            };

            setPixel(params.frameBuffer, x, y, params.width, pixelColor);
        }
    }

    // Miss function
    if(missed) {

        vec3<unsigned char> pixelColor(0, 0, 0);

        if(params.hasSky) {

            vec2<float> uvs = getSkyUVs(ray);
            vec3<float> sky = tex(params.skyTextureObj, uvs.u, uvs.v);

            pixelColor = { static_cast<unsigned char>(sky.r * 255), static_cast<unsigned char>(sky.g * 255), static_cast<unsigned char>(sky.b * 255) };
        }

        setPixel(params.frameBuffer, x, y, params.width, pixelColor);
    }
}

}