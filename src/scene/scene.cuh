#pragma once

#include <iostream>
#include <vector>

#include "graphics/buffer.cuh"
#include "graphics/material.cuh"

namespace gph
{

struct Scene {
    SHARED_PTR(Scene)

    Buffer<float>::Ptr vertexBuffer;
    Buffer<unsigned int>::Ptr indexBuffer;
    std::vector<Material> materials;

    Scene(Buffer<float>::Ptr _vertexBuffer, Buffer<unsigned int>::Ptr _indexBuffer, 
        const std::vector<Material>& _materials) : vertexBuffer(_vertexBuffer), 
        indexBuffer(_indexBuffer), materials(_materials) {
    }

    Scene() = default;
    ~Scene() = default;
};

}