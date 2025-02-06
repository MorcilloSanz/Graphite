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

void extractMeshData(const tinygltf::Model& gltfModel, Model::Ptr model) {

    std::vector<float> batchVertices;
    std::vector<unsigned int> batchIndices;

    for (const auto& mesh : gltfModel.meshes) {
        for (const auto& primitive : mesh.primitives) {

            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) 
                continue;

            int materialIndex = 0;

            // Get material index
            int matIndex = primitive.material;
            if (matIndex >= 0 && matIndex < gltfModel.materials.size()) {
                const tinygltf::Material& material = gltfModel.materials[matIndex];
                materialIndex = material.pbrMetallicRoughness.baseColorTexture.index;
            }

            // Get pos, normals and uvs
            auto extractAttribute = [&](const std::string& name, std::vector<float>& buffer) {

                auto it = primitive.attributes.find(name);
                if (it == primitive.attributes.end()) 
                    return false;

                const tinygltf::Accessor& accessor = gltfModel.accessors[it->second];
                const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = gltfModel.buffers[bufferView.buffer];

                const float* data = reinterpret_cast<const float*>(&bufferData.data[bufferView.byteOffset + accessor.byteOffset]);
                buffer.assign(data, data + accessor.count * accessor.type);

                return true;
            };

            std::vector<float> positions, normals, texcoords;
            extractAttribute("POSITION", positions);
            extractAttribute("NORMAL", normals);
            extractAttribute("TEXCOORD_0", texcoords);

            std::vector<float> vertex;

            unsigned int length = positions.size() / 3;
            for(int i = 0; i < length; i ++) {
                vertex.push_back(positions[0 + i * 3]); vertex.push_back(positions[1 + i * 3]); vertex.push_back(positions[2 + i * 3]); // Position
                vertex.push_back(1.f); vertex.push_back(1.0f); vertex.push_back(1.0f);                                                  // Color
                vertex.push_back(normals[0 + i * 3]); vertex.push_back(normals[1 + i * 3]); vertex.push_back(normals[2 + i * 3]);       // Normals
                vertex.push_back(texcoords[0 + i * 2]); vertex.push_back(texcoords[1 + i * 2]);                                         // Uvs
                vertex.push_back(materialIndex);                                                                                        // Material index
            }

            size_t n = batchVertices.size();
            batchVertices.insert(batchVertices.end(), vertex.begin(), vertex.end());

            // Get indices
            std::vector<uint32_t> indices;
            if (primitive.indices >= 0) {

                const tinygltf::Accessor& accessor = gltfModel.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = gltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = gltfModel.buffers[bufferView.buffer];

                const void* dataPtr = &bufferData.data[bufferView.byteOffset + accessor.byteOffset];
                for (size_t i = 0; i < accessor.count; i++) {

                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                        indices.push_back(static_cast<const uint16_t*>(dataPtr)[i] + n);
                    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                        indices.push_back(static_cast<const uint32_t*>(dataPtr)[i] + n);
                }
            }

            batchIndices.insert(batchIndices.end(), indices.begin(), indices.end());
        }
    }

    Buffer<float>::Ptr vertexBuffer = Buffer<float>::New(&batchVertices[0], sizeof(float) * batchVertices.size());
    Buffer<unsigned int>::Ptr indexBuffer = Buffer<unsigned int>::New(&batchIndices[0], sizeof(unsigned int) * batchIndices.size());

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