#include "buffer.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

//------------------//
//  BufferRegister  //
//------------------//

BufferRegister* BufferRegister::getInstance() {
    if(instance == nullptr)
        instance = new BufferRegister();

    return instance;
}

void BufferRegister::destroyInstance() {
    if (instance != nullptr) {
        delete instance;
        instance = nullptr;
    }
}

BufferRegister* BufferRegister::instance = nullptr;

//-----------------//
//     Renderer    //
//-----------------//

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


//-------------//
//   Buffer    //
//-------------//

Buffer::Buffer(unsigned int _id, size_t _size)
    : id(_id), size(_size) {
    cudaMalloc((void**)&buffer, size);
    check_cuda_error("Buffer::Buffer cudaMalloc");
    cudaMemset(buffer, 0, size);
    check_cuda_error("Buffer::Buffer cudaMemset");
}

Buffer::~Buffer() {
    cudaFree(buffer);
    check_cuda_error("Buffer::~Buffer cudaFree");
}

Buffer::Buffer(const Buffer& buff)
    : id(buff.id), size(buff.size) {
    if (buff.buffer) {
        cudaMalloc((void**)&buffer, size);
        check_cuda_error("Buffer::Buffer (copy) cudaMalloc");
        cudaMemcpy(buffer, buff.buffer, size, cudaMemcpyDeviceToDevice);
        check_cuda_error("Buffer::Buffer (copy) cudaMemcpy");
    }
}

Buffer::Buffer(Buffer&& buff) noexcept
    : buffer(buff.buffer), id(buff.id), size(buff.size) {
    buff.buffer = nullptr;
    buff.size = 0;
    buff.id = 0;
}

Buffer& Buffer::operator=(const Buffer& buff) {

    if (this != &buff) {
 
        cudaFree(buffer);
        check_cuda_error("Buffer::operator= cudaFree");


        id = buff.id;
        size = buff.size;

        if (buff.buffer) {
            cudaMalloc((void**)&buffer, size);
            check_cuda_error("Buffer::operator= cudaMalloc");
            cudaMemcpy(buffer, buff.buffer, size, cudaMemcpyDeviceToDevice);
            check_cuda_error("Buffer::operator= cudaMemcpy");
        } else {
            buffer = nullptr;
        }
    }

    return *this;
}

Buffer& Buffer::operator=(Buffer&& buff) noexcept {

    if (this != &buff) {

        cudaFree(buffer);
        check_cuda_error("Buffer::operator= (move) cudaFree");

        buffer = buff.buffer;
        id = buff.id;
        size = buff.size;

        buff.buffer = nullptr;
        buff.size = 0;
        buff.id = 0;
    }

    return *this;
}

//-----------------//
//   FrameBuffer   //
//-----------------//

FrameBuffer::FrameBuffer(unsigned int id, unsigned int _width, unsigned int _height)
    : Buffer(id, 1 * width * height * 3), width(_width), height(_height) {
}

FrameBuffer::FrameBuffer(unsigned int _width, unsigned int _height)
    : FrameBuffer(0, width, height) {
}

FrameBuffer::FrameBuffer(const FrameBuffer& frameBuffer)
    : Buffer(frameBuffer), width(frameBuffer.width),
    height(frameBuffer.height) {
}

FrameBuffer::FrameBuffer(FrameBuffer&& frameBuffer) noexcept
    : Buffer(std::move(frameBuffer)), width(frameBuffer.width),
    height(frameBuffer.height) {
    frameBuffer.width = 0;
    frameBuffer.height = 0;
}

FrameBuffer& FrameBuffer::operator=(const FrameBuffer& frameBuffer) {

    if (this != &frameBuffer) {

        Buffer::operator=(frameBuffer);

        width = frameBuffer.width;
        height = frameBuffer.height;
    }

    return *this;
}

FrameBuffer& FrameBuffer::operator=(FrameBuffer&& frameBuffer) noexcept {

    if (this != &frameBuffer) {

        Buffer::operator=(std::move(frameBuffer));

        width = frameBuffer.width;
        height = frameBuffer.height;

        frameBuffer.width = 0;
        frameBuffer.height = 0;
    }

    return *this;
}

Ptr<FrameBuffer> FrameBuffer::New(unsigned int width, unsigned int height) {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    int id = bufferRegister->getFrameBuffers().size() + 1;
    Ptr<FrameBuffer> frameBuffer = std::make_shared<FrameBuffer>(id, width, height);
    bufferRegister->addFrameBuffer(frameBuffer);

    return frameBuffer;
}

void FrameBuffer::bind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindFbo(id);
}

void FrameBuffer::unbind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindFbo(0);
}

void FrameBuffer::clear() {
    cudaMemset(buffer, 0, size);
}

//------------------//
//   VertexBuffer   //
//------------------//

VertexBuffer::VertexBuffer(unsigned int id, float* data, size_t size) 
    : Buffer(id, size) {
    cudaMemcpy(buffer, data, size, cudaMemcpyHostToDevice);
    check_cuda_error("VertexBuffer::VertexBuffer cudaMemcpy");
}

VertexBuffer::VertexBuffer(const VertexBuffer& vertexBuffer) 
    : Buffer(vertexBuffer) {
}

VertexBuffer::VertexBuffer(VertexBuffer&& vertexBuffer) noexcept 
    : Buffer(std::move(vertexBuffer)) {
}

VertexBuffer& VertexBuffer::operator=(const VertexBuffer& vertexBuffer) {

    if(this != &vertexBuffer) {
        Buffer::operator=(vertexBuffer);
    }

    return *this;
}

VertexBuffer& VertexBuffer::operator=(VertexBuffer&& vertexBuffer) noexcept {

    if(this != &vertexBuffer) {
        Buffer::operator=(std::move(vertexBuffer));
    }

    return *this;
}

Ptr<VertexBuffer> VertexBuffer::New(float* data, size_t size) {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    int id = bufferRegister->getVertexBuffers().size() + 1;
    Ptr<VertexBuffer> vertexBuffer = std::make_shared<VertexBuffer>(id, data, size);
    bufferRegister->addVertexBuffer(vertexBuffer);

    return vertexBuffer;
}

void VertexBuffer::bind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindVbo(id);
}

void VertexBuffer::unbind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindVbo(0);
}

//-----------------//
//   IndexBuffer   //
//-----------------//

IndexBuffer::IndexBuffer(unsigned int id, unsigned int* indices, size_t size) 
    : Buffer(id, size) {
    cudaMemcpy(buffer, indices, size, cudaMemcpyHostToDevice);
    check_cuda_error("IndexBuffer::IndexBuffer cudaMemcpy");
}

IndexBuffer::IndexBuffer(const IndexBuffer& indexBuffer) 
    : Buffer(indexBuffer) {
}

IndexBuffer::IndexBuffer(IndexBuffer&& indexBuffer) noexcept 
    : Buffer(std::move(indexBuffer)) {
}

IndexBuffer& IndexBuffer::operator=(const IndexBuffer& indexBuffer) {

    if(this != &indexBuffer) {
        Buffer::operator=(indexBuffer);
    }

    return *this;
}

IndexBuffer& IndexBuffer::operator=(IndexBuffer&& indexBuffer) noexcept {

    if(this != &indexBuffer) {
        Buffer::operator=(std::move(indexBuffer));
    }

    return *this;
}

Ptr<IndexBuffer> IndexBuffer::New(unsigned int* indices, size_t size) {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    int id = bufferRegister->getIndexBuffers().size() + 1;
    Ptr<IndexBuffer> indexBuffer = std::make_shared<IndexBuffer>(id, indices, size);
    bufferRegister->addIndexBuffer(indexBuffer);

    return indexBuffer;
}

void IndexBuffer::bind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindIbo(id);
}

void IndexBuffer::unbind() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
    bufferRegister->bindIbo(0);
}

//------------//
//    CUDA    //
//------------//

void check_cuda_error(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after " << message << ": " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

}