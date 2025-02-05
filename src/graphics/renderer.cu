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

KernelVertexParams Renderer::getKernelVertexParams(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    KernelVertexParams params;

    KernelVertexBuffer kernelVertexBuffer(vertexBuffer->buff, vertexBuffer->size);
    params.vertexBuffer = kernelVertexBuffer;

    KernelIndexBuffer kernelIndexBuffer(indexBuffer->buff, indexBuffer->size);
    params.indexBuffer = kernelIndexBuffer;

    mat4<float> modelviewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;
    params.modelviewMatrix = modelviewMatrix;

    mat3<float> normalMatrix;
    normalMatrix.row1 = uniforms.modelMatrix.row1.xyz();
    normalMatrix.row2 = uniforms.modelMatrix.row2.xyz();
    normalMatrix.row3 = uniforms.modelMatrix.row3.xyz();
    normalMatrix = normalMatrix.inverse().transpose();
    params.normalMatrix = normalMatrix;

    return params;
}

void Renderer::vertexShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    KernelVertexParams params = getKernelVertexParams(vertexBuffer, indexBuffer);

    int threadsPerBlock = 256;
    int count = indexBuffer->size / sizeof(unsigned int);
    int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    kernel_vertex<<<numBlocks, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();
}

KernelFragmentParams Renderer::getKernelFragmentParams(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    KernelFragmentParams params;

    KernelFrameBuffer kernelFrameBuffer(frameBuffer->buff, frameBuffer->width, frameBuffer->height);
    params.frameBuffer = kernelFrameBuffer;

    KernelVertexBuffer kernelVertexBuffer(vertexBuffer->buff, vertexBuffer->size);
    params.vertexBuffer = kernelVertexBuffer;

    KernelIndexBuffer kernelIndexBuffer(indexBuffer->buff, indexBuffer->size);
    params.indexBuffer = kernelIndexBuffer;

    KernelTexture kernelSky(sky->getTextureObject(), hasSky);
    params.sky = kernelSky;

    params.materialsCount = 0;

    return params;
}

void Renderer::fragmentShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer) {

    KernelFragmentParams params = getKernelFragmentParams(vertexBuffer, indexBuffer);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((frameBuffer->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (frameBuffer->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();
}

void Renderer::setSky(Texture::Ptr sky) {
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