#pragma once

#include "math/linalg.cuh"
#include "attributes.cuh"

namespace gph
{

struct KernelFrameBuffer {

    uint8_t* buffer;
    unsigned int width, height;

    KernelFrameBuffer(uint8_t* _buffer, unsigned int _width, unsigned int _height)
        : buffer(_buffer), width(_width), height(_height) {
    }

    KernelFrameBuffer() = default;
    ~KernelFrameBuffer() = default;
};

struct KernelBuffer {

    void* buffer;
    size_t count;

    KernelBuffer(void* _buffer, size_t _count)
        : buffer(_buffer), count(_count) {
    }

    KernelBuffer() = default;
    ~KernelBuffer() = default;
};

}