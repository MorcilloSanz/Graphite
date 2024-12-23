#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <vendor/stb_image_write.h>

#include "buffer/buffer.cuh"
#include "math/linalg.cuh"

using namespace gph;

int main() {


    mat4<float> matrix(1.0f);
    matrix.row1 = { 1.0f, 2.0f, 3.0f, -0.3f };
    matrix.row2 = { 3.0f, -1.0f, 0.5f, 2.0f };
    matrix.row3 = { -1.0f, 1.0f, -5.5f, 4.0f };
    matrix.row4 = { -2.0f, 3.0f, -3.5f, 1.0f };

    matrix = matrix.inverse();
    matrix = matrix.inverse();

    std::cout << matrix.row1.x << " " << matrix.row1.y << " " << matrix.row1.z << " " << matrix.row1.w << std::endl;
    std::cout << matrix.row2.x << " " << matrix.row2.y << " " << matrix.row2.z << " " << matrix.row2.w << std::endl;
    std::cout << matrix.row3.x << " " << matrix.row3.y << " " << matrix.row3.z << " " << matrix.row3.w << std::endl;
    std::cout << matrix.row4.x << " " << matrix.row4.y << " " << matrix.row4.z << " " << matrix.row4.w << std::endl;


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

    // Vertex Buffer. x y z r g b nx ny nz u v tanx tany tanz bitanx bitany bitanz
    float data[18] = {
         0.0f,  0.5f, -1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -1.0f,  0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, -1.0f,  0.0f, 0.0f, 1.0f
    };
    
    Ptr<VertexBuffer> vertexBuffer = VertexBuffer::New(data, sizeof(data));
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
    destroyGraphite();

    return 0;
}