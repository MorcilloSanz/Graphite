#pragma once

#include "graphics/texture.cuh"
#include "math/linalg.cuh"

namespace gph
{

struct Material {
    vec3<float> albedo;
    float roughness;
    float metallic;
    vec3<float> emission;
};

struct TextureMaterial {
    Texture albedo;
    Texture roughness;
    Texture metallic;
    Texture emission;
};

}