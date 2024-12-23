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
    const uchar3& color) {

    frameBuffer[3 * (x + y * width)    ] = color.x;
    frameBuffer[3 * (x + y * width) + 1] = color.y;
    frameBuffer[3 * (x + y * width) + 2] = color.z;
}

__global__ void kernel(uint8_t* frameBuffer, float* vertexBuffer, 
    unsigned int* indexBuffer, unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    vec3<float> origin = {
        2.0 * x / width - 1.0, 
        2.0 * y / height - 1.0, 
        1.0f 
    };
    vec3<float> direction = { 0.0f, 0.0f, -1.0f };
    Ray<float> ray(origin, direction);

    vec3<float> v1 = { vertexBuffer[0], vertexBuffer[1], vertexBuffer[2] };
    vec3<float> v2 = { vertexBuffer[6], vertexBuffer[7], vertexBuffer[8] };
    vec3<float> v3 = { vertexBuffer[12], vertexBuffer[13], vertexBuffer[14] };
    Triangle<float> triangle (v1, v2, v3);

    Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

    if(hitInfo.hit) {
        setPixel(frameBuffer, x, y, width, make_uchar3(255, 255, 255));
    }else {
        setPixel(frameBuffer, x, y, width, make_uchar3(128, 50, 255));
    }
}

}