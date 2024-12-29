#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

namespace gph
{

class Texture {
private:
    uint8_t* data;
    size_t width, height;
    size_t pitch;
public:
    Texture(uint8_t* hData, size_t _width, size_t _height);
    Texture() = default;
    ~Texture() = default;
public:
    void bind();
    void unbind();
public:
    inline size_t getWidth() const { return width; }
    inline size_t getHeight() const { return height; }
    inline size_t getPitch() const { return pitch; }

    inline uint8_t* getData() { return data; }
};

}