#include <iostream>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <vendor/stb_image_write.h>

#include "graphics/buffer.cuh"
#include "graphics/renderer.cuh"

#include "math/linalg.cuh"
#include "math/transform.cuh"

using namespace gph;

int main() {

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free Memory (GPU): " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Memory (GPU): " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // Renderer
    constexpr unsigned int width = 1080;
    constexpr unsigned int height = 720;
    
    Renderer renderer(width, height);

    size_t framebufferMem = sizeof(uint8_t) * width * height * 3;
    std::cout << "Frame Buffer Mem (GPU): " << static_cast<float>(framebufferMem) / (1024 * 1024) << "MB" << std::endl;

    // Vertex Buffer: x y z r g b nx ny nz uvx uvy
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,  0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,  0.0f, 0.0f, -1.0f,  0.0f, 1.0f,
         0.0f,  0.5f,  0.0f,  1.0f, 0.5f, 0.5f,  0.0f, 1.0f,  0.0f,  0.5f, 0.5f
    };

    Buffer<float> vertexBuffer(vertices, sizeof(vertices));

    // Index buffer
    unsigned int indices[] = {
        0, 1, 4, 1, 2, 4,
        2, 3, 4, 3, 0, 4,
        0, 1, 2, 0, 2, 3
    };
    
    Buffer<unsigned int> indexBuffer(indices, sizeof(indices));

    // Sky
    int skyWidth, skyHeight, skyChannels;

    stbi_set_flip_vertically_on_load(1);
    unsigned char* skyData = stbi_load("C:/Users/amorc/Desktop/sky.png", &skyWidth, &skyHeight, &skyChannels, STBI_rgb_alpha);

    Texture sky(skyData, skyWidth, skyHeight);
    renderer.setSky(sky);

    stbi_image_free(skyData);

    // Draw call
    renderer.clear();

    mat4<float> model = rotationX<float>(M_PI / 5) * rotationY<float>(M_PI / 5);
    mat4<float> view = scale<float>(vec3<float>(0.9f));

    Uniforms uniforms(model, view);
    renderer.setUniforms(uniforms);

    renderer.draw(vertexBuffer, indexBuffer);

    // CPU image
    uint8_t* bufferCPU = new uint8_t[renderer.getFrameBuffer().size];

    cudaMemcpy(bufferCPU, renderer.getFrameBuffer().buff, renderer.getFrameBuffer().size, cudaMemcpyDeviceToHost);
    stbi_write_png("output.png", width, height, STBI_rgb, bufferCPU, width * STBI_rgb);
    
    delete[] bufferCPU;

    return 0;
}