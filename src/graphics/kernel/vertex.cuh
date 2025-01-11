#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"

#include "kernel.cuh"
#include "attributes.cuh"

namespace gph
{

__global__ void kernel_vertex(KernelVertexParams params) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int count = params.indexBuffer.size / sizeof(unsigned int);
    if (idx >= count)
        return;
    
    // transform vertex position
    vec4<float> position = {
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_X),
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_Y),
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_Z),
        1.0f
    };

    vec4<float> positionPrime = params.modelviewMatrix * position;

    // transform normal
    vec3<float> normal = {
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_NX),
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_NY),
        getAttribute(params.vertexBuffer.buffer, params.indexBuffer.buffer, idx, ATTRIBUTE_NZ)
    };

    vec3<float> normalPrime = params.normalMatrix * normal;

    // update position 
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_X)] = positionPrime.x;
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_Y)] = positionPrime.y;
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_Z)] = positionPrime.z;

    // update normal
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_NX)] = normalPrime.x;
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_NY)] = normalPrime.y;
    params.vertexBuffer.buffer[getAttributeIndex(params.indexBuffer.buffer, idx, ATTRIBUTE_NZ)] = normalPrime.z;
}

}