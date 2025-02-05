#include "renderer.cuh"

#include "buffer.cuh"
#include "texture.cuh"

#include "kernel/fragment.cuh"
#include "kernel/vertex.cuh"

namespace gph
{

Renderer::Renderer(unsigned int width, unsigned int height) 
    : frameBuffer(FrameBuffer::New(width, height)), hasSky(false) {
}

KernelVertexParams Renderer::getKernelVertexParams(Scene::Ptr scene) {

    KernelVertexParams params;

    Buffer<float>::Ptr vertexBuffer = scene->vertexBuffer;
    Buffer<unsigned int>::Ptr indexBuffer = scene->indexBuffer;

    KernelVertexBuffer kernelVertexBuffer(vertexBuffer->buff, vertexBuffer->size);
    params.vertexBuffer = kernelVertexBuffer;

    KernelIndexBuffer kernelIndexBuffer(indexBuffer->buff, indexBuffer->size);
    params.indexBuffer = kernelIndexBuffer;

    mat4<float> modelviewMatrix = uniforms.viewMatrix * uniforms.modelMatrix;
    params.modelviewMatrix = modelviewMatrix;

    mat3<float> normalMatrix;
    normalMatrix.row1 = uniforms.modelMatrix.row1.xyz();
    normalMatrix.row2 = uniforms.modelMatrix.row2.xyz();
    normalMatrix.row3 = uniforms.modelMatrix.row3.xyz();
    normalMatrix = normalMatrix.inverse().transpose();
    params.normalMatrix = normalMatrix;

    return params;
}

void Renderer::vertexShader(Scene::Ptr scene) {

    // Params
    KernelVertexParams params = getKernelVertexParams(scene);

    // Kernel
    int threadsPerBlock = 256;
    int count = scene->indexBuffer->size / sizeof(unsigned int);
    int numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    kernel_vertex<<<numBlocks, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();
}

KernelFragmentParams Renderer::getKernelFragmentParams(Scene::Ptr scene) {

    KernelFragmentParams params;

    Buffer<float>::Ptr vertexBuffer = scene->vertexBuffer;
    Buffer<unsigned int>::Ptr indexBuffer = scene->indexBuffer;

    KernelFrameBuffer kernelFrameBuffer(frameBuffer->buff, frameBuffer->width, frameBuffer->height);
    params.frameBuffer = kernelFrameBuffer;

    KernelVertexBuffer kernelVertexBuffer(vertexBuffer->buff, vertexBuffer->size);
    params.vertexBuffer = kernelVertexBuffer;

    KernelIndexBuffer kernelIndexBuffer(indexBuffer->buff, indexBuffer->size);
    params.indexBuffer = kernelIndexBuffer;

    KernelTexture kernelSky(sky->getTextureObject(), hasSky);
    params.sky = kernelSky;

    return params;
}

void Renderer::fragmentShader(Scene::Ptr scene) {

    // Params
    KernelFragmentParams params = getKernelFragmentParams(scene);

    // Materials
    KernelMaterial* kernelMaterialsGPU;
    params.materialsCount = scene->materials.size();

    if(params.materialsCount > 0) {

        KernelMaterial* kernelMaterials = new KernelMaterial[params.materialsCount];
        cudaMalloc((void**)&kernelMaterialsGPU, sizeof(KernelMaterial) * params.materialsCount);

        unsigned int index = 0;
        for(auto& material : scene->materials) {

            KernelMaterial kernelMaterial;

            KernelTexture albedo(0, false);
            KernelTexture metallicRoughness(0, false);
            KernelTexture normal(0, false);
            KernelTexture ambientOcclusion(0, false);
            KernelTexture emission(0, false);
            
            if(material.albedo) albedo = KernelTexture(material.albedo->getTextureObject());
            if(material.metallicRoughness) metallicRoughness = KernelTexture(material.metallicRoughness->getTextureObject());
            if(material.normal) normal = KernelTexture(material.normal->getTextureObject());
            if(material.ambientOcclusion) ambientOcclusion = KernelTexture(material.ambientOcclusion->getTextureObject());
            if(material.emission) emission = KernelTexture(material.emission->getTextureObject());

            kernelMaterial.albedo = albedo;
            kernelMaterial.metallicRoughness = metallicRoughness;
            kernelMaterial.normal = normal;
            kernelMaterial.ambientOcclusion = ambientOcclusion;
            kernelMaterial.emission = emission;

            kernelMaterials[index] = kernelMaterial;
            index ++;
        }

        for(int batch = 0; batch < params.materialsCount; batch ++) {
            std::cout << "batch " << batch << " albedo " << kernelMaterials[batch].albedo.hasTexture << std::endl;
            std::cout << "batch " << batch << " metallicRoughness " << kernelMaterials[batch].metallicRoughness.hasTexture << std::endl;
            std::cout << "batch " << batch << " normal " << kernelMaterials[batch].normal.hasTexture << std::endl;
            std::cout << "batch " << batch << " ambientOcclusion " << kernelMaterials[batch].ambientOcclusion.hasTexture << std::endl;
            std::cout << "batch " << batch << " emission " << kernelMaterials[batch].emission.hasTexture << std::endl;
        }

        cudaMemcpy(kernelMaterialsGPU, kernelMaterials, sizeof(KernelMaterial) * params.materialsCount, cudaMemcpyHostToDevice);
        params.materials = kernelMaterialsGPU;

        delete[] kernelMaterials;
    }

    // Kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((frameBuffer->width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (frameBuffer->height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    kernel_fragment<<<blocksPerGrid, threadsPerBlock>>>(params);
    cudaDeviceSynchronize();

    if(params.materialsCount > 0) 
        cudaFree(kernelMaterialsGPU);
}

void Renderer::setSky(Texture::Ptr sky) {
    this->sky = sky;
    hasSky = true;
}

void Renderer::draw(Scene::Ptr scene) {
    vertexShader(scene);
    fragmentShader(scene);
}

void Renderer::clear() {
    frameBuffer->clear();
}

}