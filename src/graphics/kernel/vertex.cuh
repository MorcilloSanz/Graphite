#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "attributes.cuh"

namespace gph
{

__global__ void kernel_vertex(float* vertexBuffer, size_t vertexSize, unsigned int* indexBuffer, 
    size_t indexSize, mat4<float> modelviewMatrix, mat3<float> normalMatrix) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int count = indexSize / sizeof(unsigned int);
    if (idx >= count)
        return;
    
    // transform vertex position
    vec4<float> position = {
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_X),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Y),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Z),
        1.0f
    };

    vec4<float> positionPrime = modelviewMatrix * position;

    // transform normal
    vec3<float> normal = {
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_NX),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_NY),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_NZ)
    };

    vec3<float> normalPrime = normalMatrix * normal;

    // update position 
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_X)] = positionPrime.x;
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_Y)] = positionPrime.y;
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_Z)] = positionPrime.z;

    // update normal
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_NX)] = normalPrime.x;
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_NY)] = normalPrime.y;
    vertexBuffer[getAttributeIndex(indexBuffer, idx, ATTRIBUTE_NZ)] = normalPrime.z;
}

}