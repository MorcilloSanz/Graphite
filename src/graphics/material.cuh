#pragma once

#include "graphics/texture.cuh"
#include "math/linalg.cuh"

namespace gph
{

struct Material {
    Texture::Ptr albedo;
    Texture::Ptr roughness;
    Texture::Ptr metallic;
    Texture::Ptr emission;
};

}