#include "buffer.cuh"

namespace gph
{

void check_cuda_error() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

template <typename T>
Buffer<T>::Buffer(size_t _size)
    : size(_size) {
    cudaMalloc((void**)&buff, size);
    cudaMemset(buff, 0, size);
}

template <typename T>
Buffer<T>::Buffer(T* data, size_t size) 
    : Buffer(size) {
    cudaMemcpy(buff, data, size, cudaMemcpyHostToDevice);
}

template <typename T>
Buffer<T>::~Buffer() {
    if(buff) cudaFree(buff);
}

FrameBuffer::FrameBuffer(unsigned int _width, unsigned int _height)
    : Buffer<uint8_t>(_width * _height * 3), width(_width), height(_height) {
}

void FrameBuffer::clear() {
    if(buff) {
        cudaMemset(buff, 0, size);
    }
}

template class Buffer<uint8_t>;
template class Buffer<char>;
template class Buffer<float>;
template class Buffer<double>;
template class Buffer<unsigned int>;
template class Buffer<int>;
template class Buffer<long>;

}