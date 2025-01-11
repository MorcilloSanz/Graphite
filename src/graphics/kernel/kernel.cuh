#pragma once

#include "math/linalg.cuh"

namespace gph
{

struct KernelFrameBuffer {

    uint8_t* buffer;
    unsigned int width;
    unsigned int height;

    KernelFrameBuffer(uint8_t* _buffer, unsigned int _width, unsigned int _height)
        : buffer(_buffer), width(_width), height(_height) {
    }

    KernelFrameBuffer() = default;
    ~KernelFrameBuffer() = default;
};

struct KernelVertexBuffer {

    float* buffer;
    size_t size;

    KernelVertexBuffer(float* _buffer, size_t _size)
        : buffer(_buffer), size(_size) {
    }

    KernelVertexBuffer() = default;
    ~KernelVertexBuffer() = default;
};

struct KernelIndexBuffer {

    unsigned int* buffer;
    size_t size;

    KernelIndexBuffer(unsigned int* _buffer, size_t _size)
        : buffer(_buffer), size(_size) {
    }

    KernelIndexBuffer() = default;
    ~KernelIndexBuffer() = default;
};

struct KernelTexture {

    cudaTextureObject_t texture;
    bool hasTexture;

    KernelTexture(cudaTextureObject_t _texture, bool _hasTexture)
        : texture(_texture), hasTexture(_hasTexture) {
    }

    KernelTexture(cudaTextureObject_t texture)
        : KernelTexture(texture, true) {
    }

    KernelTexture() = default;
    ~KernelTexture() = default;
};

struct KernelMaterial {
    KernelTexture albedo;
    KernelTexture roughness;
    KernelTexture metallic;
    KernelTexture normal;
    KernelTexture emission;
};

struct KernelFragmentParams {

    KernelFrameBuffer frameBuffer;
    KernelVertexBuffer vertexBuffer;
    KernelIndexBuffer indexBuffer;

    KernelTexture sky;
    KernelMaterial material;
};

struct KernelVertexParams {

    KernelVertexBuffer vertexBuffer;
    KernelIndexBuffer indexBuffer;

    mat4<float> modelviewMatrix;
    mat3<float> normalMatrix;
};

}
