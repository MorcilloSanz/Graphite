#include "renderer.cuh"

#include "buffer.cuh"
#include "texture.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

KernelBuffer Renderer::getKernelVertexBuffer() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    size_t vertexBufferSize = 0;
    void* vertexBuffer = nullptr;

    if(bufferRegister->getBindedVertexBufferID() > 0) {
        Ptr<VertexBuffer> bindedVertexBuffer = bufferRegister->getBindedVertexBuffer();
        vertexBuffer = bindedVertexBuffer->getBuffer();
        vertexBufferSize = bindedVertexBuffer->getSize();
    }

    return KernelBuffer(vertexBuffer, vertexBufferSize / sizeof(float));
}

KernelBuffer Renderer::getKernelIndexBuffer() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    size_t indexBufferSize = 0;
    void* indexBuffer = nullptr;

    if(bufferRegister->getBindedIndexBufferID() > 0) {
        Ptr<IndexBuffer> bindedIndexBuffer = bufferRegister->getBindedIndexBuffer();
        indexBuffer = bindedIndexBuffer->getBuffer();
        indexBufferSize = bindedIndexBuffer->getSize();
    }

    return KernelBuffer(indexBuffer, indexBufferSize / sizeof(unsigned int));
}

void Renderer::vertexShader(const KernelBuffer& kernelVertexBuffer, const KernelBuffer& kernelIndexBuffer) {
    
    mat4<float> modelViewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;

    int threadsPerBlockVertex = 256;
    int numBlocksVertex = (kernelIndexBuffer.count + threadsPerBlockVertex - 1) / threadsPerBlockVertex;

    kernel_vertex<<<numBlocksVertex, threadsPerBlockVertex>>>(kernelVertexBuffer, kernelIndexBuffer, modelViewMatrix);
    cudaDeviceSynchronize();
}

void Renderer::fragmentShader(const KernelFrameBuffer& kernelFrameBuffer, const KernelBuffer& kernelVertexBuffer, 
    const KernelBuffer& kernelIndexBuffer) {

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((kernelFrameBuffer.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (kernelFrameBuffer.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(kernelFrameBuffer, kernelVertexBuffer, kernelIndexBuffer);
    cudaDeviceSynchronize();
}

void Renderer::init() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
}

void Renderer::destroy() {
    BufferRegister::destroyInstance();
}

void Renderer::draw() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();
    if(bufferRegister->getBindedFrameBufferID() > 0) {

        // Kernel Buffers
        Ptr<FrameBuffer> bindedFrameBuffer = bufferRegister->getBindedFrameBuffer();

        KernelFrameBuffer kernelFrameBuffer(
            (uint8_t*) bindedFrameBuffer->getBuffer(), 
            bindedFrameBuffer->getWidth(), 
            bindedFrameBuffer->getHeight()
        );
        
        KernelBuffer kernelVertexBuffer = getKernelVertexBuffer();
        KernelBuffer kernelIndexBuffer = getKernelIndexBuffer();

        // Graphics pipeline
        vertexShader(kernelVertexBuffer, kernelIndexBuffer);
        fragmentShader(kernelFrameBuffer, kernelVertexBuffer, kernelIndexBuffer);
    }
}

void Renderer::clear() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    if(bufferRegister->getBindedFrameBufferID() > 0) {

        Ptr<FrameBuffer> bindedFrameBuffer = bufferRegister->getBindedFrameBuffer();
        bindedFrameBuffer->clear();
    }
}

}