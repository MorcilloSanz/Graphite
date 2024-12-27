#pragma once

#include "math/linalg.cuh"
#include "math/geometry.cuh"

#include "kernelbuffer.cuh"
#include "attributes.cuh"

namespace gph
{

__global__ void kernel_vertex(KernelBuffer kernelVertexBuffer, KernelBuffer kernelIndexBuffer, mat4<float> modelview) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kernelIndexBuffer.count)
        return;
    
    float* vertexBuffer = (float*) kernelVertexBuffer.buffer;
    unsigned int* indexBuffer = (unsigned int*) kernelIndexBuffer.buffer;

    vec4<float> vertex = {
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_X),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Y),
        getAttribute(vertexBuffer, indexBuffer, idx, ATTRIBUTE_Z),
        1.0f
    };

    // transform vertex

    // update vertex buffer

}

}