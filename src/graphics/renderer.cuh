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
    FrameBuffer frameBuffer;
    Uniforms<float> uniforms;
    Texture sky;
    bool hasSky;
public:
    Renderer(unsigned int width, unsigned int height);
    Renderer() = default;
    ~Renderer() = default;
private:
    void vertexShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
    void fragmentShader(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
public:
    void setSky(const Texture& sky);
    void draw(const Buffer<float>& vertexBuffer, const Buffer<unsigned int>& indexBuffer);
    void clear();
public:
    inline void setUniforms(const Uniforms<float>& uniforms) { this->uniforms = uniforms; }
    inline const Uniforms<float>& getUniforms() { return uniforms; }

    inline FrameBuffer& getFrameBuffer() { return frameBuffer; }
};

}