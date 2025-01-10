#pragma once

#include "math/linalg.cuh"
#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"

namespace gph
{

template <typename T>
struct Uniforms {

    mat4<T> modelMatrix;
    mat4<T> viewMatrix;

    Uniforms(const mat4<T>& _modelMatrix, const mat4<T>& _viewMatrix)
        : modelMatrix(_modelMatrix), viewMatrix(_viewMatrix) {
    }

    Uniforms()
        : modelMatrix(mat4<T>(1.0)), viewMatrix(mat4<T>(1.0)) {
    }

    ~Uniforms() = default;
};

class Renderer {
private:
    FrameBuffer::Ptr frameBuffer;
    Uniforms<float> uniforms;
    Texture sky;
    bool hasSky;
public:
    Renderer(unsigned int width, unsigned int height);
    Renderer() = default;
    ~Renderer() = default;
private:
    void vertexShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer);
    void fragmentShader(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer);
public:
    void setSky(const Texture& sky);
    void draw(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer);
    void clear();
public:
    inline void setUniforms(const Uniforms<float>& uniforms) { this->uniforms = uniforms; }
    inline const Uniforms<float>& getUniforms() { return uniforms; }

    inline FrameBuffer::Ptr getFrameBuffer() { return frameBuffer; }
};

}