#pragma once

#include "graphics/texture.cuh"
#include "math/linalg.cuh"

namespace gph
{

struct Material {
    Texture::Ptr albedo;
    Texture::Ptr metallicRoughness;
    Texture::Ptr normal;
    Texture::Ptr ambientOcclusion;
    Texture::Ptr emission;
};

}