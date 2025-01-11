#include "scene.cuh"

namespace gph
{

Scene::Scene(Buffer<float>::Ptr _vertexBuffer, Buffer<unsigned int>::Ptr _indexBuffer, const Material& _material)
    : vertexBuffer(_vertexBuffer), indexBuffer(_indexBuffer), material(_material) {
}

}