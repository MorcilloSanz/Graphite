#pragma once

#include "linalg.cuh"

namespace gph
{

template <typename T>
__host__ __device__ static mat4<T> rotationX(T angle) {
    
    T c = cos(angle);
    T s = sin(angle);

    mat4<T> matrix(1);
    matrix.row2.y = c;
    matrix.row2.z = -s;
    matrix.row3.y = s;
    matrix.row3.z = c;

    return matrix;
}

template <typename T>
__host__ __device__ static mat4<T> rotationY(T angle) {

    T c = cos(angle);
    T s = sin(angle);

    mat4<T> matrix(1);
    matrix.row1.x = c;
    matrix.row1.z = s;
    matrix.row3.x = -s;
    matrix.row3.z = c;

    return matrix;
}

template <typename T>
__host__ __device__ static mat4<T> rotationZ(T angle) {

    T c = cos(angle);
    T s = sin(angle);

    mat4<T> matrix(1);
    matrix.row1.x = c;
    matrix.row1.y = -s;
    matrix.row2.x = s;
    matrix.row2.y = c;

    return matrix;
}

template <typename T>
__host__ __device__ static mat4<T> translation(const vec3<T>& v) {

    mat4<T> matrix(1.0f);

    matrix.row1.w = v.x;
    matrix.row2.w = v.y;
    matrix.row3.w = v.z;

    return matrix;
}

template <typename T>
__host__ __device__ static mat4<T> scale(const vec3<T>& v) {

    mat4<float> matrix(1.0f);
    
    matrix.row1.x = v.x;
    matrix.row2.y = v.y;
    matrix.row3.z = v.z;

    return matrix;
}

}