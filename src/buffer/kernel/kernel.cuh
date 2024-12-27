#pragma once

#include <math/linalg.cuh>
#include <math/geometry.cuh>

/**
 * 1) Join all the meshes of a scene.
 * 2) Send them to GPU using VertexBuffer.
 * 3) Build BVH of the whole scene.
 * 5) Pass the BVH as a parameter to the kernel
 */

#define ATTRIBUTE_X 0
#define ATTRIBUTE_Y 1
#define ATTRIBUTE_Z 2
#define ATTRIBUTE_R 3
#define ATTRIBUTE_G 4
#define ATTRIBUTE_B 5

#define ATTRIBUTE_STRIDE 6

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

template <typename T>
__device__ Ray<T> castRay(int x, int y, unsigned int width, unsigned int height) {

    vec3<float> origin = {
        2.0 * x / width - 1.0, 
        2.0 * y / height - 1.0, 
        0.01f 
    };

    vec3<float> direction = { 0.0f, 0.0f, -1.0f };
    Ray<float> ray(origin, direction);

    return ray;
}

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

    Ray<float> ray = castRay<float>(x, y, kernelFrameBuffer.width, kernelFrameBuffer.height);

    for(int i = 0; i < kernelIndexBuffer.count; i += 3) {

        float x1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_X];
        float x2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_X];
        float x3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_X];

        float y1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_Y];
        float y2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_Y];
        float y3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_Y];

        float z1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_Z];
        float z2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_Z];
        float z3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_Z];

        vec3<float> v1 = { x1, y1, z1 };
        vec3<float> v2 = { x2, y2, z2 };
        vec3<float> v3 = { x3, y3, z3 };

        Triangle<float> triangle (v1, v2, v3);

        float distance = INFINITY;
        Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

        if(hitInfo.hit && hitInfo.distance < distance) {

            distance = hitInfo.distance;

            float r1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_R];
            float r2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_R];
            float r3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_R];

            float g1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_G];
            float g2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_G];
            float g3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_G];

            float b1 = vertexBuffer[indexBuffer[i + 0] * ATTRIBUTE_STRIDE + ATTRIBUTE_B];
            float b2 = vertexBuffer[indexBuffer[i + 1] * ATTRIBUTE_STRIDE + ATTRIBUTE_B];
            float b3 = vertexBuffer[indexBuffer[i + 2] * ATTRIBUTE_STRIDE + ATTRIBUTE_B];

            vec3<float> c1 = { r1, g1, b1 };
            vec3<float> c2 = { r2, g2, b2 };
            vec3<float> c3 = { r3, g3, b3 };

            vec3<float> barycentricCoords = barycentric<float>(hitInfo.intersection, triangle);
            vec3<float> colorInterpolation = c1 * barycentricCoords.x + c2 * barycentricCoords.y + c3 * barycentricCoords.z;

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

}