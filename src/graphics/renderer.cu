#include "renderer.cuh"

#include "buffer.cuh"
#include "texture.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

void Renderer::init() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
}

void Renderer::destroy() {
    BufferRegister::destroyInstance();
}

void Renderer::draw() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();
    if(bufferRegister->getBindedFrameBufferID() > 0) {

        // Kernel Frame Buffer
        Ptr<FrameBuffer> bindedFrameBuffer = bufferRegister->getBindedFrameBuffer();
        const unsigned int width = bindedFrameBuffer->getWidth();
        const unsigned int height = bindedFrameBuffer->getHeight();

        KernelFrameBuffer kernelFrameBuffer((uint8_t*)bindedFrameBuffer->getBuffer(), width, height);
        
        // Kernel Vertex Buffer
        size_t vertexBufferSize = 0;
        void* vertexBuffer = nullptr;

        if(bufferRegister->getBindedVertexBufferID() > 0) {
            Ptr<VertexBuffer> bindedVertexBuffer = bufferRegister->getBindedVertexBuffer();
            vertexBuffer = bindedVertexBuffer->getBuffer();
            vertexBufferSize = bindedVertexBuffer->getSize();
        }

        KernelBuffer kernelVertexBuffer(vertexBuffer, vertexBufferSize / sizeof(float));

        // Index Buffer
        size_t indexBufferSize = 0;
        void* indexBuffer = nullptr;

        if(bufferRegister->getBindedIndexBufferID() > 0) {
            Ptr<IndexBuffer> bindedIndexBuffer = bufferRegister->getBindedIndexBuffer();
            indexBuffer = bindedIndexBuffer->getBuffer();
            indexBufferSize = bindedIndexBuffer->getSize();
        }

        KernelBuffer kernelIndexBuffer(indexBuffer, indexBufferSize / sizeof(unsigned int));

        // Vertex kernel -> transform each vertex
        mat4<float> modelViewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;

        int threadsPerBlockVertex = 256;
        int numBlocksVertex = (kernelIndexBuffer.count + threadsPerBlockVertex - 1) / threadsPerBlockVertex;

        kernel_vertex<<<numBlocksVertex, threadsPerBlockVertex>>>(kernelVertexBuffer, kernelIndexBuffer, modelViewMatrix);
        cudaDeviceSynchronize();
  
        // Fragment kernel -> compute each fragment
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(kernelFrameBuffer, kernelVertexBuffer, kernelIndexBuffer);
        cudaDeviceSynchronize();
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