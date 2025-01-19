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
    std::vector<Buffer<float>> vertexBuffers;
    std::vector<Buffer<unsigned int>> indexBuffers;
    std::vector<Material> materials;
public:
    
};

class SceneGraph {
    SHARED_PTR(Scene)
private:
    std::vector<Scene> scenes;
    std::vector<SceneGraph> children;
};

}
