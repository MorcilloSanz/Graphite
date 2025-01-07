#pragma once

#define ATTRIBUTE_X 0
#define ATTRIBUTE_Y 1
#define ATTRIBUTE_Z 2

#define ATTRIBUTE_R 3
#define ATTRIBUTE_G 4
#define ATTRIBUTE_B 5

#define ATTRIBUTE_NX 6
#define ATTRIBUTE_NY 7
#define ATTRIBUTE_NZ 8

#define ATTRIBUTE_UVX 9
#define ATTRIBUTE_UVY 10

#define ATTRIBUTE_STRIDE 11

namespace gph 
{

/**
 * Retrieves a specific attribute index from a vertex in the vertex buffer.
 *
 * @param indexBuffer Pointer to the index buffer mapping vertices.
 * @param i Index of the vertex in the index buffer.
 * @param attribute Offset of the desired attribute within the vertex data.
 * @return The value of the specified attribute for the given vertex.
 */
__device__ unsigned int getAttributeIndex(unsigned int* indexBuffer, int i, int attribute) {
    return indexBuffer[i] * ATTRIBUTE_STRIDE + attribute;
}

/**
 * Retrieves a specific attribute from a vertex in the vertex buffer.
 *
 * @param vertexBuffer Pointer to the vertex buffer containing vertex data.
 * @param indexBuffer Pointer to the index buffer mapping vertices.
 * @param i Index of the vertex in the index buffer.
 * @param attribute Offset of the desired attribute within the vertex data.
 * @return The value of the specified attribute for the given vertex.
 */
__device__ float getAttribute(float* vertexBuffer, unsigned int* indexBuffer, int i, int attribute) {
    return vertexBuffer[getAttributeIndex(indexBuffer, i, attribute)];
}

/**
 * Retrieves three consecutive vertex attributes as a vec3.
 *
 * @param vertexBuffer Pointer to the vertex buffer containing vertex data.
 * @param indexBuffer Pointer to the index buffer mapping vertices.
 * @param i Starting index of the vertex in the index buffer.
 * @param attribute Offset of the desired attribute within the vertex data.
 * @return A vec3<float> containing the three specified attributes.
 */
__device__ vec3<float> getAttributes3(float* vertexBuffer, unsigned int* indexBuffer, int i, int attribute) {
    return {
        getAttribute(vertexBuffer, indexBuffer, i + 0, attribute),
        getAttribute(vertexBuffer, indexBuffer, i + 1, attribute),
        getAttribute(vertexBuffer, indexBuffer, i + 2, attribute)
    };
}

}