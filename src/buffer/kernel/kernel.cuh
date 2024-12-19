#pragma once

struct KernelParams {
    
};

__global__ void kernel(uint8_t* buffer, unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    buffer[3 * (x + y * width)    ] = 128;
    buffer[3 * (x + y * width) + 1] = 50;
    buffer[3 * (x + y * width) + 2] = 255;
}