#include "buffer.cuh"

#include <kernel/kernel.cuh>

namespace ghp
{

//----------------//
//    Graphite    //
//----------------//

void initGraphite() {
    BufferRegister* bufferRegister = BufferRegister::getInstance();
}

void destroyGraphite() {
    BufferRegister::destroyInstance();
}

void draw() {

    BufferRegister* bufferRegister = BufferRegister::getInstance();

    if(bufferRegister->getBindedFrameBufferID() > 0) {
        Ptr<FrameBuffer> bindedFrameBuffer = bufferRegister->getBindedFrameBuffer();
        bindedFrameBuffer->draw();
    }
}

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

//-------------//
//   Buffer    //
//-------------//

Buffer::Buffer(unsigned int _id, size_t _size)
    : id(_id), size(_size) {
    cudaMalloc((void**)&buffer, size);
    check_cuda_error("Buffer::Buffer cudaMalloc");
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
    : Buffer(id, size()), width(_width), height(_height) {
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

void FrameBuffer::draw() {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel<<<blocksPerGrid, threadsPerBlock>>>((uint8_t*)buffer, width, height);
    cudaDeviceSynchronize();
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