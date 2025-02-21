#pragma once

#include "math/linalg.cuh"
#include "graphics/buffer.cuh"
#include "graphics/texture.cuh"
#include "kernel/kernel.cuh"
#include "scene/scene.cuh"

#include "material.cuh"

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
    TextureHDR::Ptr sky;
    bool hasSky;
public:
    Renderer(unsigned int width, unsigned int height);
    Renderer() = default;
    ~Renderer() = default;
private:
    KernelVertexParams getKernelVertexParams(Scene::Ptr scene);
    void vertexShader(Scene::Ptr scene);

    KernelFragmentParams getKernelFragmentParams(Scene::Ptr scene);
    void fragmentShader(Scene::Ptr scene);
public:
    void setSky(TextureHDR::Ptr sky);
    void draw(Scene::Ptr scene);
    void clear();
public:
    inline void setUniforms(const Uniforms<float>& uniforms) { this->uniforms = uniforms; }
    inline const Uniforms<float>& getUniforms() { return uniforms; }

    inline FrameBuffer::Ptr getFrameBuffer() { return frameBuffer; }
};

}