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
    
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    cudaTextureObject_t texObj;
public:
    Texture(uint8_t* hData, size_t _width, size_t _height);
    Texture() = default;
    ~Texture();
public:
    inline size_t getWidth() const { return width; }
    inline size_t getHeight() const { return height; }

    inline uint8_t* getData() { return data; }

    inline cudaResourceDesc& getResourceDesc() { return resDesc; }
    inline cudaTextureDesc& getTextureDesc() { return texDesc; }
    inline cudaTextureObject_t getTextureObject() { return texObj; }
};

}