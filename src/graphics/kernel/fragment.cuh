#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "attributes.cuh"


namespace gph 
{

__device__ void setPixel(uint8_t* frameBuffer, int x, int y, int width, const vec3<unsigned char>& color) {

    frameBuffer[3 * (x + y * width)    ] = color.r;
    frameBuffer[3 * (x + y * width) + 1] = color.g;
    frameBuffer[3 * (x + y * width) + 2] = color.b;
}

__global__ void kernel_fragment(uint8_t* frameBuffer, unsigned int width, unsigned int height, 
    float* vertexBuffer, size_t vertexSize, unsigned int *indexBuffer, size_t indexSize) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Ray casting
    Ray<float> ray = Ray<float>::castRay(x, y, width, height);
    float distance = INFINITY;
    bool missed = true;

    // Ray intersections
    int count = indexSize / sizeof(unsigned int);
    for(int i = 0; i < count; i += 3) {

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

            setPixel(frameBuffer, x, y, width, pixelColor);
        }
    }

    // Miss function
    if(missed) {
        setPixel(frameBuffer, x, y, width, vec3<unsigned char>(0, 0, 0));
    }
}

}