#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <vendor/stb_image_write.h>

#include "buffer/buffer.cuh"

using namespace ghp;

int main() {

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free Memory (GPU): " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total Memory (GPU): " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // Graphite
    initGraphite();

    // Frame buffer
    constexpr unsigned int width = 500;
    constexpr unsigned int height = 500;

    size_t framebufferMem = sizeof(uint8_t) * width * height * 3;
    std::cout << "Frame Buffer Mem (GPU): " << static_cast<float>(framebufferMem) / (1024 * 1024) << "MB" << std::endl;

    Ptr<FrameBuffer> frameBuffer = FrameBuffer::New(width, height);
    frameBuffer->bind();

    // Vertex Buffer
    float data[18] = {
         0.0f,  0.5f, 1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, 1.0f,  0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, 1.0f,  0.0f, 0.0f, 1.0f
    };

    VertexBuffer::Attributes attributes;
    attributes.insert(VertexBuffer::Attribute(0, 3)); // (0) Position: x, y, z
    attributes.insert(VertexBuffer::Attribute(1, 3)); // (1) Color: r, g, b
    
    Ptr<VertexBuffer> vertexBuffer = VertexBuffer::New(data, sizeof(data), attributes);
    vertexBuffer->bind();

    unsigned int indices[3] = { 0, 1, 2 };
    Ptr<IndexBuffer> indexBuffer = IndexBuffer::New(indices, sizeof(indices));
    indexBuffer->bind();

    // Draw call
    draw();

    // CPU
    uint8_t* bufferCPU = new uint8_t[frameBuffer->getSize()];
    cudaMemcpy(bufferCPU, frameBuffer->getBuffer(), frameBuffer->getSize(), cudaMemcpyDeviceToHost);
    
    stbi_write_png("output.png", width, height, STBI_rgb, bufferCPU, width * STBI_rgb);
    
    delete[] bufferCPU;

    // Destroy
    ghp::destroyGraphite();

    return 0;
}