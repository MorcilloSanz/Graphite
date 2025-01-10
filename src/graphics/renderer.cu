#include "renderer.cuh"

#include "buffer.cuh"
#include "texture.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

Renderer::Renderer(unsigned int width, unsigned int height) 
    : frameBuffer(FrameBuffer::New(width, height)), hasSky(false) {
}

void Renderer::vertexShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    mat4<float> modelviewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;

    mat3<float> normalMatrix;
    normalMatrix.row1 = uniforms.modelMatrix.row1.xyz();
    normalMatrix.row2 = uniforms.modelMatrix.row2.xyz();
    normalMatrix.row3 = uniforms.modelMatrix.row3.xyz();
    normalMatrix = normalMatrix.inverse().transpose();

    int threadsPerBlock = 256;
    int count = indexBuffer->size / sizeof(unsigned int);
    int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    kernel_vertex<<<numBlocks, threadsPerBlock>>>(vertexBuffer->buff, vertexBuffer->size, 
        indexBuffer->buff, indexBuffer->size, modelviewMatrix, normalMatrix);

    cudaDeviceSynchronize();
}

void Renderer::fragmentShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    KernelFragmentParams params;
    params.frameBuffer = frameBuffer->buff;
    params.width = frameBuffer->width;
    params.height = frameBuffer->height;
    params.vertexBuffer = vertexBuffer->buff;
    params.vertexSize = vertexBuffer->size;
    params.indexBuffer = indexBuffer->buff;
    params.indexSize = indexBuffer->size;
    
    params.hasSky = hasSky;
    if(hasSky) params.skyTextureObj = sky.getTextureObject();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((frameBuffer->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (frameBuffer->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();
}

void Renderer::setSky(const Texture& sky) {
    this->sky = sky;
    hasSky = true;
}

void Renderer::draw(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {
    vertexShader(vertexBuffer, indexBuffer);
    fragmentShader(vertexBuffer, indexBuffer);
}

void Renderer::clear() {
    frameBuffer->clear();
}

}