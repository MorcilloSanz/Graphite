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
#include "math/linalg.cuh"

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

            materialIndex = (matIndex >= 0 && matIndex < gltfModel.materials.size()) ? matIndex : 0;

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

                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
                        memcpy(dst, src, componentCount * sizeof(float));
                    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        for (size_t j = 0; j < componentCount; ++j)
                            dst[j] = src[j] / 255.0f;  // Normalizar si es UNSIGNED_BYTE
                    } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
                        for (size_t j = 0; j < componentCount; ++j)
                            dst[j] = src16[j] / 65535.0f;  // Normalizar si es UNSIGNED_SHORT
                    }
                }

                return true;
            };

            std::vector<float> positions, normals, texcoords, tangents;
            extractAttribute("POSITION", positions);
            extractAttribute("NORMAL", normals);
            extractAttribute("TEXCOORD_0", texcoords);
            extractAttribute("TANGENT", tangents);

            std::vector<float> vertex;
            bool hasTangents = true;
            size_t vertexCount = positions.size() / 3;

            for (size_t i = 0; i < vertexCount; i++) {

                // Position
                vertex.push_back(positions[i * 3 + 0]); 
                vertex.push_back(positions[i * 3 + 1]); 
                vertex.push_back(positions[i * 3 + 2]);                                      

                // Color (dummy)
                vertex.push_back(1.f); vertex.push_back(1.0f); vertex.push_back(1.0f);       

                // Normal
                vec3<float> normal(0.f);
                if (normals.size() >= (i * 3 + 3)) {
                    normal = { normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2] };
                    vertex.push_back(normal.x); vertex.push_back(normal.y); vertex.push_back(normal.z);                                    
                } else {
                    vertex.push_back(0.f); vertex.push_back(0.f); vertex.push_back(0.f);
                }

                // UVs
                if (texcoords.size() >= (i * 2 + 2)) {
                    vertex.push_back(texcoords[i * 2 + 0]); 
                    vertex.push_back(1.0f - texcoords[i * 2 + 1]); // Invertir V para OpenGL
                } else {
                    vertex.push_back(0.0f); vertex.push_back(0.0f);
                }

                // Tangents and bitangents
                vec4<float> tangent(0.0f);
                float bitangentSign = 0.0f;

                if(tangents.size() >= (i * 4 + 4)) {
                    // Tangents
                    tangent = { tangents[i * 4 + 0], tangents[i * 4 + 1], tangents[i * 4 + 2], tangents[i * 4 + 3] };
                    bitangentSign = tangent.w;
                    vertex.push_back(tangent.x); vertex.push_back(tangent.y); vertex.push_back(tangent.z);
                }else {
                    vertex.push_back(0.0f); vertex.push_back(0.0f); vertex.push_back(0.0f);
                    hasTangents = false;
                }

                // Bitangents
                vec3<float> bitangent = normal.cross(tangent.xyz()) * bitangentSign;
                vertex.push_back(bitangent.x); vertex.push_back(bitangent.y); vertex.push_back(bitangent.z);

                // Material index
                vertex.push_back(static_cast<float>(materialIndex));
            }

            size_t n = batchVertices.size() / 18; // Cada vértice tiene 18 atributos
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

            // Compute tangents and bitangents
            if (!hasTangents) {

                for (size_t i = 0; i < batchIndices.size(); i += 3) {

                    uint32_t i0 = batchIndices[i];
                    uint32_t i1 = batchIndices[i + 1];
                    uint32_t i2 = batchIndices[i + 2];
            
                    vec3<float> p0(batchVertices[i0 * 18], batchVertices[i0 * 18 + 1], batchVertices[i0 * 18 + 2]);
                    vec3<float> p1(batchVertices[i1 * 18], batchVertices[i1 * 18 + 1], batchVertices[i1 * 18 + 2]);
                    vec3<float> p2(batchVertices[i2 * 18], batchVertices[i2 * 18 + 1], batchVertices[i2 * 18 + 2]);
            
                    vec2<float> uv0(batchVertices[i0 * 18 + 9], batchVertices[i0 * 18 + 10]);
                    vec2<float> uv1(batchVertices[i1 * 18 + 9], batchVertices[i1 * 18 + 10]);
                    vec2<float> uv2(batchVertices[i2 * 18 + 9], batchVertices[i2 * 18 + 10]);
            
                    vec3<float> edge1 = p1 - p0;
                    vec3<float> edge2 = p2 - p0;
                    vec2<float> deltaUV1 = uv1 - uv0;
                    vec2<float> deltaUV2 = uv2 - uv0;
            
                    float det = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                    float f = 1.0f / (det + 1e-6);
            
                    vec3<float> tangent = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * f;
                    tangent = tangent.normalize();
            
                    vec3<float> bitangent = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) * f;
                    bitangent = bitangent.normalize();
            
                    for (uint32_t idx : {i0, i1, i2}) {
                        batchVertices[idx * 18 + 11] = tangent.x;
                        batchVertices[idx * 18 + 12] = tangent.y;
                        batchVertices[idx * 18 + 13] = tangent.z;
            
                        batchVertices[idx * 18 + 14] = bitangent.x;
                        batchVertices[idx * 18 + 15] = bitangent.y;
                        batchVertices[idx * 18 + 16] = bitangent.z;
                    }
                }
            }
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