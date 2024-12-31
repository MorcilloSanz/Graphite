#include "texture.cuh"

namespace gph
{

Texture::Texture(uint8_t* hData, size_t _width, size_t _height) 
    : width(_width), height(_height) {

    size_t imageSize = width * height * 4 * sizeof(uint8_t);
    cudaMalloc(&data, imageSize);
    cudaMemcpy(data, hData, imageSize, cudaMemcpyHostToDevice);

    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = data;
    resDesc.res.linear.sizeInBytes = imageSize;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 8;
    resDesc.res.linear.desc.y = 8;
    resDesc.res.linear.desc.z = 8;
    resDesc.res.linear.desc.w = 8;

    texDesc.readMode = cudaReadModeNormalizedFloat; // Leer como flotante normalizado
    texDesc.normalizedCoords = 1; // Coordenadas normalizadas    
}

void Texture::bind() {
    
}

void Texture::unbind() {
    
}

}