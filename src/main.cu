#include <iostream>
#include <cstdint>

//#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <vendor/stb_image_write.h>

#include "graphics/buffer.cuh"
#include "graphics/renderer.cuh"

#include "math/linalg.cuh"
#include "math/transform.cuh"

#include "scene/scene.cuh"
#include "scene/model.cuh"

using namespace gph;

int main() {

    cudaDeviceReset();

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free Memory (GPU): " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Memory (GPU): " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // Renderer
    constexpr unsigned int width = 1600;
    constexpr unsigned int height = 900;
    
    Renderer renderer(width, height);

    // Sky
    int skyWidth, skyHeight, skyChannels;

    stbi_set_flip_vertically_on_load(1);
    float* skyData = stbi_loadf("C:/Users/amorc/Documents/Dev/3D/360images/aerodynamics_workshop_2k.hdr", &skyWidth, &skyHeight, &skyChannels, STBI_rgb_alpha);

    TextureHDR::Ptr sky = TextureHDR::New(skyData, skyWidth, skyHeight);
    renderer.setSky(sky);

    stbi_image_free(skyData);

    // Model
    Model::Ptr model = Model::fromFile("c:/Users/amorc/Documents/Dev/3D/models/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf");

    // Draw call
    renderer.clear();

    mat4<float> modelMatrix = rotationX<float>(M_PI / 2.0f) * rotationZ<float>(M_PI / 4.0f);
    mat4<float> viewMatrix = scale<float>(vec3<float>(0.75f));

    Uniforms uniforms(modelMatrix, viewMatrix);

    renderer.setUniforms(uniforms);
    renderer.draw(model);

    // CPU image
    uint8_t* bufferCPU = new uint8_t[renderer.getFrameBuffer()->size];

    cudaMemcpy(bufferCPU, renderer.getFrameBuffer()->buff, renderer.getFrameBuffer()->size, cudaMemcpyDeviceToHost);
    stbi_write_png("output.png", width, height, STBI_rgb, bufferCPU, width * STBI_rgb);
    
    delete[] bufferCPU;

    return 0;
}