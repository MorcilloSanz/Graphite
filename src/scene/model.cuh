#pragma once

#include <iostream>

#include "scene.cuh"

namespace gph
{

class Model : public Scene {
    SHARED_PTR(Model)
public:
    Model(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer, const std::vector<Material>& materials);
    Model() = default;
    ~Model() = default;
public:
    static Model::Ptr fromFile(const std::string& path);
};

}