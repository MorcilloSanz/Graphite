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

    // Graphite
    ghp::initGraphite();

    constexpr unsigned int width = 500;
    constexpr unsigned int height = 500;

    ghp::Ptr<FrameBuffer> frameBuffer = ghp::FrameBuffer::New(width, height);
    frameBuffer->bind();

    ghp::draw();

    // CPU
    uint8_t* bufferCPU = new uint8_t[frameBuffer->size()];
    cudaMemcpy(bufferCPU, frameBuffer->getBuffer(), frameBuffer->size(), cudaMemcpyDeviceToHost);
    
    stbi_write_png("output.png", width, height, STBI_rgb, bufferCPU, width * STBI_rgb);
    
    delete[] bufferCPU;

    // Destroy
    ghp::destroyGraphite();

    return 0;
}