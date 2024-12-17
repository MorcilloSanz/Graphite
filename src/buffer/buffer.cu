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

//-----------------//
//   FrameBuffer   //
//-----------------//

FrameBuffer::FrameBuffer(int _id, unsigned int _width, unsigned int _height)
    : id(_id), width(_width), height(_height) {
    cudaMalloc((void **)&buffer, size());
    check_cuda_error("FrameBuffer::FrameBuffer cudaMalloc");
}

FrameBuffer::FrameBuffer(unsigned int _width, unsigned int _height)
    : FrameBuffer(0, width, height) {
}

FrameBuffer::~FrameBuffer() {
    cudaFree(buffer);
    check_cuda_error("FrameBuffer::~FrameBuffer cudaFree");
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

    kernel<<<blocksPerGrid, threadsPerBlock>>>(buffer, width, height);
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