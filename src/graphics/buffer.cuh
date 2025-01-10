#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

#include <cuda_runtime.h>

namespace gph
{

#define SHARED_PTR(clazz) public: \
                            using Ptr = std::shared_ptr<clazz>; \
                            template<class... Args> \
                            inline static Ptr New(Args&&... args) { \
                                return std::make_shared<clazz>(std::forward<Args>(args)...); \
                            }\

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
