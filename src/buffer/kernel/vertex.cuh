#pragma once

/*
#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "kernelbuffer.cuh"
#include "attributes.cuh"
*/

namespace gph
{

__global__ void kernel_vertex(KernelBuffer kernelVertexBuffer, KernelBuffer kernelIndexBuffer, mat4<float> modelview) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= kernelIndexBuffer.count)
        return;
    
    float* vertexBuffer = (float*) kernelVertexBuffer.buffer;
    unsigned int* indexBuffer = (unsigned int*) kernelIndexBuffer.buffer;

    // transform vertex
    vec4<float> vertex = {
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_X),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Y),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Z),
        1.0f
    };

    vec4<float> transformed = modelview.transform(vertex);

    // update vertex buffer
    unsigned int attributeIndex = getAttributeIndex(indexBuffer, idx, ATTRIBUTE_X);
    vertexBuffer[attributeIndex    ] = transformed.x;
    vertexBuffer[attributeIndex + 1] = transformed.y;
    vertexBuffer[attributeIndex + 2] = transformed.z;
}

}