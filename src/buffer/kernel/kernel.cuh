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

struct KernelFrameBuffer {

    uint8_t* buffer;
    unsigned int width, height;

    KernelFrameBuffer(uint8_t* _buffer, unsigned int _width, unsigned int _height)
        : buffer(_buffer), width(_width), height(_height) {
    }

    KernelFrameBuffer() = default;
    ~KernelFrameBuffer() = default;
};

struct KernelBuffer {

    void* buffer;
    size_t count;

    KernelBuffer(void* _buffer, size_t _count)
        : buffer(_buffer), count(_count) {
    }

    KernelBuffer() = default;
    ~KernelBuffer() = default;
};

__device__ void setPixel(KernelFrameBuffer frameBuffer, int x, int y, 
    const vec3<unsigned char>& color) {

    frameBuffer.buffer[3 * (x + y * frameBuffer.width)    ] = color.r;
    frameBuffer.buffer[3 * (x + y * frameBuffer.width) + 1] = color.g;
    frameBuffer.buffer[3 * (x + y * frameBuffer.width) + 2] = color.b;
}

__global__ void kernel(KernelFrameBuffer kernelFrameBuffer, KernelBuffer kernelVertexBuffer, 
    KernelBuffer kernelIndexBuffer) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float* vertexBuffer = (float*) kernelVertexBuffer.buffer;
    unsigned int* indexBuffer = (unsigned int*) kernelIndexBuffer.buffer;

    vec3<float> origin = {
        2.0 * x / kernelFrameBuffer.width - 1.0, 
        2.0 * y / kernelFrameBuffer.height - 1.0, 
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

        setPixel(kernelFrameBuffer, x, y, pixelColor);
    }else {
        setPixel(kernelFrameBuffer, x, y, vec3<unsigned char>(0, 0, 0));
    }
}

}