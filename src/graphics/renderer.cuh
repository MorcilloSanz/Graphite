#pragma once

#include "math/linalg.cuh"
#include "graphics/buffer.cuh"

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
    FrameBuffer frameBuffer;
    Uniforms<float> uniforms;
public:
    Renderer() = default;
    ~Renderer() = default;
private:
    void vertexShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
    void fragmentShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
public:
    void draw(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
    void clear();
public:
    inline void setUniforms(const Uniforms<float>& uniforms) { this->uniforms = uniforms; }
    inline const Uniforms<float>& getUniforms() { return uniforms; }

    inline void setFrameBuffer(const FrameBuffer& frameBuffer) { this->frameBuffer = frameBuffer; };
    inline FrameBuffer& getFrameBuffer() { return frameBuffer; }
};

}