#pragma once

#include <math/linalg.cuh>
#include <math/geometry.cuh>

/**
 * 1) Join all the meshes of a scene.
 * 2) Send them to GPU using VertexBuffer.
 * 3) Build BVH of the whole scene.
 * 5) Pass the BVH as a parameter to the kernel
 */

namespace gph 
{

__device__ void setPixel(uint8_t* frameBuffer, int x, int y, unsigned int width, 
    const vec3<unsigned char>& color) {

    frameBuffer[3 * (x + y * width)    ] = color.r;
    frameBuffer[3 * (x + y * width) + 1] = color.g;
    frameBuffer[3 * (x + y * width) + 2] = color.b;
}

__global__ void kernel(uint8_t* frameBuffer, float* vertexBuffer, 
    unsigned int* indexBuffer, unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    vec3<float> origin = {
        2.0 * x / width - 1.0, 
        2.0 * y / height - 1.0, 
        0.01f 
    };
    vec3<float> direction = { 0.0f, 0.0f, -1.0f };
    Ray<float> ray(origin, direction);

    vec3<float> v1 = { vertexBuffer[0], vertexBuffer[1], vertexBuffer[2] };
    vec3<float> v2 = { vertexBuffer[6], vertexBuffer[7], vertexBuffer[8] };
    vec3<float> v3 = { vertexBuffer[12], vertexBuffer[13], vertexBuffer[14] };

    Triangle<float> triangle (v1, v2, v3);
    Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

    if(hitInfo.hit) {

        vec3<float> c1 = { vertexBuffer[3], vertexBuffer[4], vertexBuffer[5] };
        vec3<float> c2 = { vertexBuffer[9], vertexBuffer[10], vertexBuffer[11] };
        vec3<float> c3 = { vertexBuffer[15], vertexBuffer[16], vertexBuffer[17] };

        vec3<float> barycentricCoords = barycentric<float>(hitInfo.intersection, triangle);
        vec3 colorInterpolation = c1 * barycentricCoords.x + c2 * barycentricCoords.y + c3 * barycentricCoords.z;

        vec3<unsigned char> pixelColor = {
            static_cast<unsigned char>(colorInterpolation.x * 255),
            static_cast<unsigned char>(colorInterpolation.y * 255),
            static_cast<unsigned char>(colorInterpolation.z * 255),
        };

        setPixel(frameBuffer, x, y, width, pixelColor);
    }else {
        setPixel(frameBuffer, x, y, width, vec3<unsigned char>(0, 0, 0));
    }
}

}