#include "texture.cuh"

namespace gph
{

Texture::Texture(uint8_t* hData, size_t width, size_t height) 
    : TextureBase(width, height) {

    size_t imageSize = width * height * 4 * sizeof(uint8_t);

    // 2D CUDA Array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, hData, width * 4 * sizeof(uint8_t),
                        width * 4 * sizeof(uint8_t), height, cudaMemcpyHostToDevice);

    // Resource Descriptor
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Configure Texture Descriptor
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        printf("Couldn't create cudaTextureObject: %s\n", cudaGetErrorString(err));
    }  
}

Texture::~Texture() {

    if (texObj)
        cudaDestroyTextureObject(texObj);

    if (resDesc.resType == cudaResourceTypeArray && resDesc.res.array.array)
        cudaFreeArray(resDesc.res.array.array);
}

TextureHDR::TextureHDR(float* hData, size_t width, size_t height) 
    : TextureBase(width, height) {

    size_t imageSize = width * height * 4 * sizeof(float);

    // 2D CUDA Array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, hData, width * 4 * sizeof(float),
                        width * 4 * sizeof(float), height, cudaMemcpyHostToDevice);

    // Resource Descriptor
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Configure Texture Descriptor
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;

    // **Cambio importante:** Leer como float directamente
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        printf("Couldn't create cudaTextureObject: %s\n", cudaGetErrorString(err));
    }
}

TextureHDR::~TextureHDR() {

    if (texObj)
        cudaDestroyTextureObject(texObj);

    if (resDesc.resType == cudaResourceTypeArray && resDesc.res.array.array)
        cudaFreeArray(resDesc.res.array.array);
}

}