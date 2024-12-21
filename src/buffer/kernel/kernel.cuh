#pragma once

/**
 * 1) Join all the meshes of a scene.
 * 2) Send them to GPU using VertexBuffer.
 * 3) Build BVH of the whole scene.
 * 5) Pass the BVH as a parameter to the kernel
 */

namespace ghp 
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

    setPixel(frameBuffer, x, y, width, make_uchar3(128, 50, 255));
}

}