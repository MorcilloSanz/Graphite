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

struct Vertex {
    float position[3];
    float normal[3];
    float texcoord[2];
};

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

void extractMeshData(const tinygltf::Model& model) {

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {

            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) 
                continue;

            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
            unsigned int materialIndex = -1;

            // Obtener índice de material
            int matIndex = primitive.material;
            if (matIndex >= 0 && matIndex < model.materials.size()) {
                const tinygltf::Material& material = model.materials[matIndex];
                materialIndex = material.pbrMetallicRoughness.baseColorTexture.index;
            }

            // Extraer posiciones, normales y coordenadas UV
            auto extractAttribute = [&](const std::string& name, std::vector<float>& buffer) {

                auto it = primitive.attributes.find(name);
                if (it == primitive.attributes.end()) 
                    return false;

                const tinygltf::Accessor& accessor = model.accessors[it->second];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = model.buffers[bufferView.buffer];

                const float* data = reinterpret_cast<const float*>(&bufferData.data[bufferView.byteOffset + accessor.byteOffset]);
                buffer.assign(data, data + accessor.count * accessor.type);

                return true;
            };

            std::vector<float> positions, normals, texcoords;
            extractAttribute("POSITION", positions);
            extractAttribute("NORMAL", normals);
            extractAttribute("TEXCOORD_0", texcoords);

            // Almacenar vértices
            for (size_t i = 0; i < positions.size() / 3; i++) {

                Vertex v{};
                std::memcpy(v.position, &positions[i * 3], sizeof(float) * 3);

                if (!normals.empty()) std::memcpy(v.normal, &normals[i * 3], sizeof(float) * 3);
                if (!texcoords.empty()) std::memcpy(v.texcoord, &texcoords[i * 2], sizeof(float) * 2);

                vertices.push_back(v);
            }

            // Extraer índices
            if (primitive.indices >= 0) {

                const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                const tinygltf::Buffer& bufferData = model.buffers[bufferView.buffer];

                const void* dataPtr = &bufferData.data[bufferView.byteOffset + accessor.byteOffset];

                for (size_t i = 0; i < accessor.count; i++) {

                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        indices.push_back(static_cast<const uint16_t*>(dataPtr)[i]);
                    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        indices.push_back(static_cast<const uint32_t*>(dataPtr)[i]);
                    }
                }
            }

            std::cout << "Mesh: " << mesh.name << " - Vertices: " << vertices.size() << ", Indices: " << indices.size() << ", Material index: " << materialIndex << std::endl;
        }
    }
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

void extractMaterials(const tinygltf::Model& model) {

    unsigned int materialIndex = 0;
    for (const auto& mat : model.materials) {

        Material material;

        material.albedo = loadTexture(model, mat.pbrMetallicRoughness.baseColorTexture.index);
        material.metallicRoughness = loadTexture(model, mat.pbrMetallicRoughness.metallicRoughnessTexture.index);
        material.normal = loadTexture(model, mat.normalTexture.index);
        material.ambientOcclusion = loadTexture(model, mat.occlusionTexture.index);
        material.emission = loadTexture(model, mat.emissiveTexture.index);

        std::cout << "Material " << materialIndex << std::endl;
        std::cout << "  Albedo: " << (material.albedo != nullptr ? "Loaded" : "Not loaded") << std::endl;
        std::cout << "  Roughness/Metallic: " << (material.metallicRoughness != nullptr ? "Loaded" : "Not loaded") << std::endl;
        std::cout << "  Normal: " << (material.normal != nullptr ? "Loaded" : "Not loaded") << std::endl;
        std::cout << "  AO: " << (material.ambientOcclusion != nullptr ? "Loaded" : "Not loaded") << std::endl;
        std::cout << "  Emission: " << (material.emission != nullptr ? "Loaded" : "Not loaded") << std::endl;

        materialIndex ++;
    }
}

Model::Ptr Model::fromFile(const std::string& path) {

    tinygltf::Model tinygltfModel;
    if (!loadGLTFModel(path, tinygltfModel)) {
        std::cerr << "Error al cargar el modelo" << std::endl;
    }

    extractMeshData(tinygltfModel);
    extractMaterials(tinygltfModel);

    Buffer<float> vertexBuffer;

    Model::Ptr model = Model::New();
    return model;
}

}