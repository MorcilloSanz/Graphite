#pragma once

#include <iostream>
#include <vector>

#include "graphics/buffer.cuh"
#include "graphics/material.cuh"

namespace gph
{

class Scene {
    SHARED_PTR(Scene)
private:
    Buffer<float>::Ptr vertexBuffer;
    Buffer<unsigned int>::Ptr indexBuffer;
    Material material;
    std::vector<Scene> children;
public:
    Scene(Buffer<float>::Ptr _vertexBuffer, Buffer<unsigned int>::Ptr _indexBuffer, const Material& _material);
    Scene() = default;
    ~Scene() = default;
public:
    inline Buffer<float>::Ptr getVertexBuffer() { return vertexBuffer; }
    inline Buffer<unsigned int>::Ptr getIndexBuffer() { return indexBuffer; }

    inline Material getMaterial() { return material; }

    inline std::vector<Scene>& getChildren() { return children; }
};

}
