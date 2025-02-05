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
    std::vector<Buffer<float>::Ptr> vertexBuffers;
    std::vector<Buffer<unsigned int>::Ptr> indexBuffers;
    std::vector<Material> materials;
public:
    Scene() = default;
    ~Scene() = default;
public:
    inline void addVertexBuffer(Buffer<float>::Ptr vertexBuffer) { vertexBuffers.push_back(vertexBuffer); }
    inline void addIndexBuffer(Buffer<unsigned int>::Ptr indexBuffer) { indexBuffers.push_back(indexBuffer); }
    inline void addMaterial(const Material& material) { materials.push_back(material); }
public:
    inline std::vector<Buffer<float>::Ptr>& getVertexBuffers() { return vertexBuffers; }
    inline std::vector<Buffer<unsigned int>::Ptr>& getIndexBuffers() { return indexBuffers; }
    inline std::vector<Material>& getMaterials() { return materials; }
};

}
