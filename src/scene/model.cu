#include "model.cuh"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include "vendor/tiny_gltf.h"

namespace gph
{

Model::Model(Buffer<float>::Ptr vertexBuffer, Buffer<unsigned int>::Ptr indexBuffer, const std::vector<Material>& materials)
    : Scene(vertexBuffer, indexBuffer, materials) {
}

bool loadGLTFModel(const std::string& filename, tinygltf::Model& model) {

    tinygltf::TinyGLTF loader;
    std::string err, warn;
    
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    if (!ret) {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    }
    
    if (!warn.empty()) std::cerr << "Warning: " << warn << std::endl;
    if (!err.empty()) std::cerr << "Error: " << err << std::endl;
    
    return ret;
}

size_t getNumComponents(int type) {
    switch (type) {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2: return 2;
        case TINYGLTF_TYPE_VEC3: return 3;
        case TINYGLTF_TYPE_VEC4: return 4;
        case TINYGLTF_TYPE_MAT2: return 4;
        case TINYGLTF_TYPE_MAT3: return 9;
        case TINYGLTF_TYPE_MAT4: return 16;
        default: return 1; // Fallback en caso de error
    }
}

void extractMeshData(const tinygltf::Model& gltfModel, Model::Ptr model) {

    std::vector<float> batchVertices;
    std::vector<unsigned int> batchIndices;

    for (const auto& mesh : gltfModel.meshes) {
        for (const auto& primitive : mesh.primitives) {

            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) 
                continue;

            int materialIndex = 0;

            // Obtener el índice del material
            int matIndex = primitive.material;
            if (matIndex >= 0 && matIndex < gltfModel.materials.size()) {
                const tinygltf::Material& material = gltfModel.materials[matIndex];
                materialIndex = material.pbrMetallicRoughness.baseColorTexture.index;
            }

            // Función para extraer atributos correctamente respetando el byteStride
            auto extractAttribute = [&](const std::string& name, std::vector<float>& buffer) {

                auto it = primitive.attributes.find(name);
                if (it == primitive.attributes.end()) 
                    return false;

                const tinygltf::Accessor& accessor = gltfModel.accessors[it->second];
                const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = gltfModel.buffers[bufferView.buffer];

                const uint8_t* dataPtr = bufferData.data.data() + bufferView.byteOffset + accessor.byteOffset;
                size_t componentCount = getNumComponents(accessor.type);
                size_t elementSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);

                buffer.resize(accessor.count * componentCount);

                for (size_t i = 0; i < accessor.count; ++i) {
                    const uint8_t* src = dataPtr + i * (bufferView.byteStride > 0 ? bufferView.byteStride : componentCount * elementSize);
                    float* dst = &buffer[i * componentCount];

                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        memcpy(dst, src, componentCount * sizeof(float));
                    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        for (size_t j = 0; j < componentCount; ++j) {
                            dst[j] = src[j] / 255.0f;  // Normalizar si es UNSIGNED_BYTE
                        }
                    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
                        for (size_t j = 0; j < componentCount; ++j) {
                            dst[j] = src16[j] / 65535.0f;  // Normalizar si es UNSIGNED_SHORT
                        }
                    }
                }

                return true;
            };

            std::vector<float> positions, normals, texcoords;
            extractAttribute("POSITION", positions);
            extractAttribute("NORMAL", normals);
            extractAttribute("TEXCOORD_0", texcoords);

            std::vector<float> vertex;
            size_t vertexCount = positions.size() / 3;

            for (size_t i = 0; i < vertexCount; i++) {
                vertex.push_back(positions[i * 3 + 0]); 
                vertex.push_back(positions[i * 3 + 1]); 
                vertex.push_back(positions[i * 3 + 2]); // Posición

                vertex.push_back(1.f); vertex.push_back(1.0f); vertex.push_back(1.0f); // Color (dummy)

                if (normals.size() >= (i * 3 + 3)) {
                    vertex.push_back(normals[i * 3 + 0]); 
                    vertex.push_back(normals[i * 3 + 1]); 
                    vertex.push_back(normals[i * 3 + 2]); // Normal
                } else {
                    vertex.push_back(0.f); vertex.push_back(0.f); vertex.push_back(0.f); // Normal por defecto
                }

                if (texcoords.size() >= (i * 2 + 2)) {
                    vertex.push_back(texcoords[i * 2 + 0]); 
                    vertex.push_back(1.0f - texcoords[i * 2 + 1]); // Invertir V para OpenGL
                } else {
                    vertex.push_back(0.0f); vertex.push_back(0.0f); // UV por defecto
                }

                vertex.push_back(static_cast<float>(materialIndex)); // Índice de material
            }

            size_t n = batchVertices.size() / 12; // Cada vértice tiene 12 atributos

            batchVertices.insert(batchVertices.end(), vertex.begin(), vertex.end());

            // Obtener los índices
            std::vector<uint32_t> indices;
            if (primitive.indices >= 0) {
                const tinygltf::Accessor& accessor = gltfModel.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = gltfModel.buffers[bufferView.buffer];

                const void* dataPtr = bufferData.data.data() + bufferView.byteOffset + accessor.byteOffset;

                for (size_t i = 0; i < accessor.count; i++) {
                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        indices.push_back(static_cast<const uint16_t*>(dataPtr)[i] + n);
                    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        indices.push_back(static_cast<const uint32_t*>(dataPtr)[i] + n);
                    }
                }
            }

            batchIndices.insert(batchIndices.end(), indices.begin(), indices.end());
        }
    }

    Buffer<float>::Ptr vertexBuffer = Buffer<float>::New(batchVertices.data(), sizeof(float) * batchVertices.size());
    Buffer<unsigned int>::Ptr indexBuffer = Buffer<unsigned int>::New(batchIndices.data(), sizeof(unsigned int) * batchIndices.size());

    model->vertexBuffer = vertexBuffer;
    model->indexBuffer = indexBuffer;
}

Texture::Ptr loadTexture(const tinygltf::Model& model, int textureIndex) {

    Texture::Ptr texture;
    if (textureIndex < 0) return nullptr;

    const tinygltf::Texture& tex = model.textures[textureIndex];
    const tinygltf::Image& image = model.images[tex.source];

    uint8_t* data = (uint8_t*)&image.image[0];
    size_t width = static_cast<size_t>(image.width);
    size_t height = static_cast<size_t>(image.height);

    texture = Texture::New(data, width, height);
    return texture;
}

void extractMaterials(const tinygltf::Model& gltfModel, Model::Ptr model) {

    for (const auto& mat : gltfModel.materials) {

        Material material;

        material.albedo = loadTexture(gltfModel, mat.pbrMetallicRoughness.baseColorTexture.index);
        material.metallicRoughness = loadTexture(gltfModel, mat.pbrMetallicRoughness.metallicRoughnessTexture.index);
        material.normal = loadTexture(gltfModel, mat.normalTexture.index);
        material.ambientOcclusion = loadTexture(gltfModel, mat.occlusionTexture.index);
        material.emission = loadTexture(gltfModel, mat.emissiveTexture.index);

        model->materials.push_back(material);
    }
}

Model::Ptr Model::fromFile(const std::string& path) {

    tinygltf::Model gltfModel;
    if (!loadGLTFModel(path, gltfModel))
        std::cerr << "Couldn't load model" << std::endl;

    Model::Ptr model = Model::New();

    extractMeshData(gltfModel, model);
    extractMaterials(gltfModel, model);
    
    return model;
}

}