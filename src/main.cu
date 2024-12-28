#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <vendor/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <vendor/stb_image_write.h>

#include "buffer/buffer.cuh"
#include "math/linalg.cuh"
#include "math/transform.cuh"

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
    constexpr unsigned int width = 1080;
    constexpr unsigned int height = 720;

    size_t framebufferMem = sizeof(uint8_t) * width * height * 3;
    std::cout << "Frame Buffer Mem (GPU): " << static_cast<float>(framebufferMem) / (1024 * 1024) << "MB" << std::endl;

    Ptr<FrameBuffer> frameBuffer = FrameBuffer::New(width, height);
    
    // Vertex Buffer: x y z r g b
    float vertices[] = {
        // Front square
        -0.5, -0.5,  0.5,  0.0f, 0.0f, 1.0f,
         0.5, -0.5,  0.5,  1.0f, 0.0f, 1.0f,
         0.5,  0.5,  0.5,  0.0f, 1.0f, 1.0f,
        -0.5,  0.5,  0.5,  0.0f, 1.0f, 0.5f,
        // Back square
        -0.5, -0.5, -0.5,  0.0f, 0.0f, 1.0f,
         0.5, -0.5, -0.5,  1.0f, 0.0f, 1.0f,
         0.5,  0.5, -0.5,  0.0f, 1.0f, 1.0f,
        -0.5,  0.5, -0.5,  0.0f, 1.0f, 0.5f
    };

    Ptr<VertexBuffer> vertexBuffer = VertexBuffer::New(vertices, sizeof(vertices));

    mat4<float> model = rotationX<float>(M_PI / 3) * rotationY<float>(M_PI / 4) * scale<float>(vec3<float>(1.25f));
    vertexBuffer->setModelMatrix(model);

    // Index buffer
    unsigned int indices[] = { 
        //front   //right   //back
        0, 1, 2,  1, 5, 6,  7, 6, 5,
        2, 3, 0,  6, 2, 1,  5, 4, 7,
        //left    //bottom  //top
        4, 0, 3,  4, 5, 1,  3, 2, 6,
        3, 7, 4,  1, 0, 4,  6, 7, 3 
    };

    Ptr<IndexBuffer> indexBuffer = IndexBuffer::New(indices, sizeof(indices));
    
    // Draw call
    frameBuffer->bind();
    vertexBuffer->bind();
    indexBuffer->bind();

    clear();
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