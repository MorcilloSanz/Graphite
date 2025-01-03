#include "renderer.cuh"

#include "buffer.cuh"
#include "texture.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

Renderer::Renderer(unsigned int width, unsigned int height) 
    : frameBuffer(width, height) {
}

void Renderer::vertexShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer) {

    mat4<float> modelViewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;

    int threadsPerBlock = 256;
    int count = indexBuffer.size / sizeof(unsigned int);
    int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    kernel_vertex<<<numBlocks, threadsPerBlock>>>(vertexBuffer.buff, vertexBuffer.size, 
        indexBuffer.buff, indexBuffer.size, modelViewMatrix);

    cudaDeviceSynchronize();
}

void Renderer::fragmentShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer) {

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((frameBuffer.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (frameBuffer.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(frameBuffer.buff, frameBuffer.width, frameBuffer.height, vertexBuffer.buff, 
        vertexBuffer.size, indexBuffer.buff, indexBuffer.size);
        
    cudaDeviceSynchronize();
}

void Renderer::draw(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer) {
    vertexShader(vertexBuffer, indexBuffer);
    fragmentShader(vertexBuffer, indexBuffer);
}

void Renderer::clear() {
    frameBuffer.clear();
}

}