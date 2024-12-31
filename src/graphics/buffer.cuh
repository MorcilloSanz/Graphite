#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include <cuda_runtime.h>

namespace gph
{

template <typename T>
struct Buffer {

    T* buff;
    size_t size;

    __host__ __device__ Buffer(size_t _size);
    __host__ __device__ Buffer(T* data, size_t _size);
    __host__ __device__ Buffer() = default;

    __host__ __device__ ~Buffer();
};

struct FrameBuffer : public Buffer<uint8_t> {

    unsigned int width, height;

    __host__ __device__ FrameBuffer(unsigned int _width, unsigned int _height);
    __host__ __device__ FrameBuffer() = default;

    __host__ __device__ ~FrameBuffer() = default;

    __host__ __device__ void clear();
};

void check_cuda_error(const char* message);

}
