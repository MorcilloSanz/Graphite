#include "texture.cuh"

namespace gph
{

Texture::Texture(uint8_t* hData, size_t _width, size_t _height) 
    : width(_width), height(_height) {

    size_t imageSize = width * height * 4 * sizeof(uint8_t);

    // 1. Crear un Array 2D CUDA
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 2. Copiar los datos desde el host al Array CUDA
    cudaMemcpy2DToArray(cuArray, 0, 0, hData, width * 4 * sizeof(uint8_t),
                        width * 4 * sizeof(uint8_t), height, cudaMemcpyHostToDevice);

    // 3. Configurar Resource Descriptor
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 4. Configurar Texture Descriptor
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Coordenadas fuera del rango se envuelven
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;    // Interpolación lineal
    texDesc.readMode = cudaReadModeNormalizedFloat; // Leer como flotantes normalizados
    texDesc.normalizedCoords = 1; // Coordenadas normalizadas

    // 4. Crear el objeto de textura
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        printf("Couldn't create cudaTextureObject: %s\n", cudaGetErrorString(err));
    }  
}

}