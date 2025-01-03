#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "graphics/buffer.cuh"
#include "attributes.cuh"

namespace gph
{

__global__ void kernel_vertex(float* vertexBuffer, size_t vertexSize, unsigned int* indexBuffer, 
    size_t indexSize, mat4<float> modelview) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int count = indexSize / sizeof(unsigned int);
    if (idx >= count)
        return;
    
    // transform vertex
    vec4<float> vertex = {
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_X),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Y),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Z),
        1.0f
    };

    vec4<float> transformed = modelview * vertex;

    // update vertex buffer
    unsigned int attributeIndexX = getAttributeIndex(indexBuffer, idx, ATTRIBUTE_X);
    unsigned int attributeIndexY = getAttributeIndex(indexBuffer, idx, ATTRIBUTE_Y);
    unsigned int attributeIndexZ = getAttributeIndex(indexBuffer, idx, ATTRIBUTE_Z);

    vertexBuffer[attributeIndexX] = transformed.x;
    vertexBuffer[attributeIndexY] = transformed.y;
    vertexBuffer[attributeIndexZ] = transformed.z;
}

}