#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include <cuda_runtime.h>

#include "shared.cuh"

namespace gph
{

template <typename T>
struct Buffer {
    SHARED_PTR(Buffer)

    T* buff;
    size_t size;

    Buffer(size_t _size);
    Buffer(T* data, size_t _size);
    Buffer() = default;
    ~Buffer();
};

struct FrameBuffer : public Buffer<uint8_t> {
    SHARED_PTR(FrameBuffer)

    unsigned int width, height;

    FrameBuffer(unsigned int _width, unsigned int _height);
    FrameBuffer() = default;
    ~FrameBuffer() = default;

    void clear();
};

void check_cuda_error(const char* message);

}
