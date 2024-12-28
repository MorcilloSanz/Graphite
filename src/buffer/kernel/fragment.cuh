#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "kernelbuffer.cuh"
#include "attributes.cuh"

/**
 * 1) Join all the meshes of a scene.
 * 2) Send them to GPU using VertexBuffer.
 * 3) Build BVH of the whole scene.
 * 5) Pass the BVH as a parameter to the kernel
 */
namespace gph 
{

/**
 * Casts a ray from a screen coordinate (x, y) into 3D space.
 *
 * @tparam T Data type for the ray components.
 * @param x Horizontal screen coordinate.
 * @param y Vertical screen coordinate.
 * @param width Screen width in pixels.
 * @param height Screen height in pixels.
 * @return A Ray<T> representing the ray's origin and direction.
 */
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

/**
 * Sets the color of a specific pixel in the framebuffer.
 *
 * @param frameBuffer The framebuffer to modify.
 * @param x Horizontal coordinate of the pixel.
 * @param y Vertical coordinate of the pixel.
 * @param color A vec3 containing the RGB color values for the pixel.
 */
__device__ void setPixel(KernelFrameBuffer frameBuffer, int x, int y, 
    const vec3<unsigned char>& color) {

    frameBuffer.buffer[3 * (x + y * frameBuffer.width)    ] = color.r;
    frameBuffer.buffer[3 * (x + y * frameBuffer.width) + 1] = color.g;
    frameBuffer.buffer[3 * (x + y * frameBuffer.width) + 2] = color.b;
}

__global__ void kernel_fragment(KernelFrameBuffer kernelFrameBuffer, KernelBuffer kernelVertexBuffer, 
    KernelBuffer kernelIndexBuffer) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Buffers
    float* vertexBuffer = (float*) kernelVertexBuffer.buffer;
    unsigned int* indexBuffer = (unsigned int*) kernelIndexBuffer.buffer;

    // Ray casting
    Ray<float> ray = castRay<float>(x, y, kernelFrameBuffer.width, kernelFrameBuffer.height);
    float distance = INFINITY;
    bool missed = true;

    // Ray intersections
    for(int i = 0; i < kernelIndexBuffer.count; i += 3) {

        vec3<float> X = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_X); // v1x v2x v3x
        vec3<float> Y = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_Y); // v1y v2y v3y
        vec3<float> Z = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_Z); // v1z v2z v3z

        vec3<float> v1 = { X.x, Y.x, Z.x };
        vec3<float> v2 = { X.y, Y.y, Z.y };
        vec3<float> v3 = { X.z, Y.z, Z.z };

        Triangle<float> triangle (v1, v2, v3);
        Ray<float>::HitInfo hitInfo = ray.intersects(triangle);

        if(hitInfo.hit && hitInfo.distance < distance) {

            missed = false;
            distance = hitInfo.distance;

            vec3<float> R = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_R); // c1r c2r c3r
            vec3<float> G = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_G); // c1g c2g c3g
            vec3<float> B = getAttributes(vertexBuffer, indexBuffer, i, ATTRIBUTE_B); // c1b c2b c3b

            vec3<float> c1 = { R.x, G.x, B.x };
            vec3<float> c2 = { R.y, G.y, B.y };
            vec3<float> c3 = { R.z, G.z, B.z };

            vec3<float> barycentricCoords = barycentric<float>(hitInfo.intersection, triangle);
            vec3<float> colorInterpolation = c1 * barycentricCoords.x + c2 * barycentricCoords.y + c3 * barycentricCoords.z;

            vec3<unsigned char> pixelColor = {
                static_cast<unsigned char>(colorInterpolation.x * 255),
                static_cast<unsigned char>(colorInterpolation.y * 255),
                static_cast<unsigned char>(colorInterpolation.z * 255),
            };

            setPixel(kernelFrameBuffer, x, y, pixelColor);
        }
    }

    // Miss function
    if(missed) {
        setPixel(kernelFrameBuffer, x, y, vec3<unsigned char>(0, 0, 0));
    }
}

}