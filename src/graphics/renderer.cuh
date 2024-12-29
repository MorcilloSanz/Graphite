#pragma once

#include "math/linalg.cuh"

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

class KernelBuffer;
class KernelFrameBuffer;

class Renderer {
private:
    Uniforms<float> uniforms;
public:
    Renderer() = default;
    ~Renderer() = default;
private:
    KernelBuffer getKernelVertexBuffer();
    KernelBuffer getKernelIndexBuffer();

    void vertexShader(const KernelBuffer& kernelVertexBuffer, const KernelBuffer& kernelIndexBuffer);
    void fragmentShader(const KernelFrameBuffer& kernelFrameBuffer, const KernelBuffer& kernelVertexBuffer, 
        const KernelBuffer& kernelIndexBuffer);
public:
    void init();
    void destroy();
    void draw();
    void clear();
public:
    inline void setUniforms(const Uniforms<float>& uniforms) { this->uniforms = uniforms; }
    inline const Uniforms<float>& getUniforms() { return uniforms; }
};

}