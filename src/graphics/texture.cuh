#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

#include "shared.cuh"

namespace gph
{

template <typename T>
class TextureBase {
protected:
    T* data;
    size_t width, height;

    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
    cudaTextureObject_t texObj;
public:
    TextureBase(size_t _width, size_t _height)
        : width(_width), height(_height) {
    }

    TextureBase() = default;
    ~TextureBase() = default;
public:
    inline size_t getWidth() const { return width; }
    inline size_t getHeight() const { return height; }

    inline T* getData() { return data; }

    inline cudaResourceDesc& getResourceDesc() { return resDesc; }
    inline cudaTextureDesc& getTextureDesc() { return texDesc; }
    inline cudaTextureObject_t getTextureObject() { return texObj; }
};

class Texture : public TextureBase<uint8_t> {
    SHARED_PTR(Texture)
public:
    Texture(uint8_t* hData, size_t width, size_t height);
    Texture() = default;
    ~Texture();
};

class TextureHDR : public TextureBase<float> {
    SHARED_PTR(TextureHDR)
public:
    TextureHDR(float* hData, size_t width, size_t height);
    TextureHDR() = default;
    ~TextureHDR();
};

}