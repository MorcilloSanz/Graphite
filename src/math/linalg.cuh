#pragma once

#include <cmath>

#include <cuda_runtime.h>

namespace gph
{

template <typename T>
struct vec2 {

    union { T x; T u; };
    union { T y; T v; };

    __host__ __device__ vec2(T _x, T _y)
        : x(_x), y(_y) {
    }

    __host__ __device__ vec2(T value)
        : vec2(value, value) {
    }

    __host__ __device__ vec2() = default;
    __host__ __device__ ~vec2() = default;

    // Operations
    __host__ __device__ vec2 sum(const vec2& vec) const {
        return vec2(x + vec.x, y + vec.y);
    }

    __host__ __device__ vec2 subtract(const vec2& vec) const {
        return vec2(x - vec.x, y - vec.y);
    }

    __host__ __device__ vec2 hadamard(const vec2& vec) const {
        return vec2(x * vec.x, y * vec.y);
    }

    __host__ __device__ vec2 div(const vec2& vec) const {
        return vec2(x / vec.x, y / vec.y);
    }

    __host__ __device__ T dot(const vec2& vec) const {
        return x * vec.x + y * vec.y;
    }

    __host__ __device__ T cross(const vec2& vec) const {
        return x * vec.y - y * vec.x;
    }

    __host__ __device__ T module() const {
        return sqrt(x * x + y * y);
    }

    // Operators
    __host__ __device__ vec2 operator+(const vec2& vec) const {
        return sum(vec);
    }

    __host__ __device__ vec2 operator+(T value) const {
        return sum(vec2(value));
    }

    __host__ __device__ vec2 operator-(const vec2& vec) const {
        return subtract(vec);
    }

    __host__ __device__ vec2 operator-(T value) const {
        return subtract(vec2(value));
    }

    __host__ __device__ vec2 operator*(const vec2& vec) const {
        return hadamard(vec);
    }

    __host__ __device__ vec2 operator*(T value) const {
        return hadamard(vec2(value));
    }

    __host__ __device__ vec2 operator/(const vec2& vec) const {
        return div(vec);
    }

    __host__ __device__ vec2 operator/(T value) const {
        return div(vec2(value));
    }
};

using vec2i = vec2<int>;
using vec2f = vec2<float>;
using vec2d = vec2<double>;

template <typename T>
struct vec3 {

    union { T x; T r; };
    union { T y; T g; };
    union { T z; T b; };

    __host__ __device__ vec3(T _x, T _y, T _z)
        : x(_x), y(_y), z(_z) {
    }

    __host__ __device__ vec3(T value)
        : vec3(value, value, value) {
    }

    __host__ __device__ vec3() = default;
    __host__ __device__ ~vec3() = default;

    // Operations
    __host__ __device__ vec3 sum(const vec3& vec) const {
        return vec3(x + vec.x, y + vec.y, z + vec.z);
    }

    __host__ __device__ vec3 subtract(const vec3& vec) const {
        return vec3(x - vec.x, y - vec.y, z - vec.z);
    }

    __host__ __device__ vec3 hadamard(const vec3& vec) const {
        return vec3(x * vec.x, y * vec.y, z * vec.z);
    }

    __host__ __device__ vec3 div(const vec3& vec) const {
        return vec3(x / vec.x, y / vec.y, z / vec.z);
    }

    __host__ __device__ T dot(const vec3& vec) const {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    __host__ __device__ vec3 cross(const vec3& vec) const {
        return vec3(y * vec.z - z * vec.y, -x * vec.z + z * vec.x, x * vec.y - y * vec.x);
    }

    __host__ __device__ T module() const {
        return sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ vec2<T> xy() const {
        return vec2<T>(x, y);
    }

    // Operators
    __host__ __device__ vec3 operator+(const vec3& vec) const {
        return sum(vec);
    }

    __host__ __device__ vec3 operator+(T value) const {
        return sum(vec3(value));
    }

    __host__ __device__ vec3 operator-(const vec3& vec) const {
        return subtract(vec);
    }

    __host__ __device__ vec3 operator-(T value) const {
        return subtract(vec3(value));
    }

    __host__ __device__ vec3 operator*(const vec3& vec) const {
        return hadamard(vec);
    }

    __host__ __device__ vec3 operator*(T value) const {
        return hadamard(vec3(value));
    }

    __host__ __device__ vec3 operator/(const vec3& vec) const {
        return div(vec);
    }

    __host__ __device__ vec3 operator/(T value) const {
        return div(vec3(value));
    }
};

using vec3i = vec3<int>;
using vec3f = vec3<float>;
using vec3d = vec3<double>;

template <typename T>
struct vec4 {

    union { T x; T r; };
    union { T y; T g; };
    union { T z; T b; };
    union { T w; T a; };

    __host__ __device__ vec4(T _x, T _y, T _z, T _w)
        : x(_x), y(_y), z(_z), w(_w) {
    }

    __host__ __device__ vec4(T value)
        : vec4(value, value, value, value) {
    }

    __host__ __device__ vec4() = default;
    __host__ __device__ ~vec4() = default;

    // Operations
    __host__ __device__ vec4 sum(const vec4& vec) const {
        return vec4(x + vec.x, y + vec.y, z + vec.z, w + vec.w);
    }

    __host__ __device__ vec4 subtract(const vec4& vec) const {
        return vec4(x - vec.x, y - vec.y, z - vec.z, w - vec.w);
    }

    __host__ __device__ vec4 hadamard(const vec4& vec) const {
        return vec4(x * vec.x, y * vec.y, z * vec.z, w * vec.w);
    }

    __host__ __device__ vec4 div(const vec4& vec) const {
        return vec4(x / vec.x, y / vec.y, z / vec.z, w / vec.w);
    }

    __host__ __device__ T dot(const vec4& vec) const {
        return x * vec.x + y * vec.y + z * vec.z, w * vec.w;
    }

    __host__ __device__ T module() const {
        return sqrt(x * x + y * y + z * z + w * w);
    }

    __host__ __device__ vec3<T> xyz() const {
        return vec3<T>(x, y, z);
    }

    // Operators
    __host__ __device__ vec4 operator+(const vec4& vec) const {
        return sum(vec);
    }

    __host__ __device__ vec4 operator+(T value) const {
        return sum(vec4(value));
    }

    __host__ __device__ vec4 operator-(const vec4& vec) const {
        return subtract(vec);
    }

    __host__ __device__ vec4 operator-(T value) const {
        return subtract(vec4(value));
    }

    __host__ __device__ vec4 operator*(const vec4& vec) const {
        return hadamard(vec);
    }

    __host__ __device__ vec4 operator*(T value) const {
        return hadamard(vec4(value));
    }

    __host__ __device__ vec4 operator/(const vec4& vec) const {
        return div(vec);
    }

    __host__ __device__ vec4 operator/(T value) const {
        return div(vec4(value));
    }
};

using vec4i = vec4<int>;
using vec4f = vec4<float>;
using vec4d = vec4<double>;

template <typename T>
struct mat2 {

    vec2<T> row1, row2;

    __host__ __device__ mat2(T value) {
        row1 = vec2<T>(value, 0);
        row2 = vec2<T>(0, value);
    }   

    __host__ __device__ mat2() {
        row1 = vec2<T>(0);
        row2 = vec2<T>(0);
    }

    __host__ __device__ ~mat2() = default;

    __host__ __device__ static mat2 full(T value) {

        mat2 matrix(value);

        matrix.row1.y = value;
        matrix.row2.x = value;

        return matrix;
    }

    // Operations
    __host__ __device__ mat2 sum(const mat2& matrix) {

        mat2<T> result;

        result.row1 = row1 + matrix.row1;
        result.row2 = row2 + matrix.row2;

        return result;
    }

    __host__ __device__ mat2 subtract(const mat2& matrix) {

        mat2<T> result;

        result.row1 = row1 - matrix.row1;
        result.row2 = row2 - matrix.row2;

        return result;
    }

    __host__ __device__ mat2 hadamard(const mat2& matrix) {

        mat2<T> result;

        result.row1 = row1 * matrix.row1;
        result.row2 = row2 * matrix.row2;

        return result;
    }

    __host__ __device__ mat2 div(const mat2& matrix) {

        mat2<T> result;

        result.row1 = row1 / matrix.row1;
        result.row2 = row2 / matrix.row2;

        return result;
    }

    __host__ __device__ mat2 product(const mat2& matrix) {

        mat2<T> result;

        result.row1.x = row1.x * matrix.row1.x + row1.y * matrix.row2.x;
        result.row1.y = row1.x * matrix.row1.y + row1.y * matrix.row2.y;
        result.row2.x = row2.x * matrix.row1.x + row2.y * matrix.row2.x;
        result.row2.y = row2.x * matrix.row1.y + row2.y * matrix.row2.y;

        return result;
    }

    __host__ __device__ vec2<T> transform(const vec2<T>& v) {

        vec2<T> result;

        result.x = row1.x * v.x + row1.y * v.y;
        result.y = row2.x * v.x + row2.y * v.y;

        return result;
    }

    __host__ __device__ T determinant() {
        return row1.x * row2.y - row1.y * row2.x;
    }

    __host__ __device__ mat2 inverse() {

        mat2<T> result;

        T det = determinant();
        T inv_det = 1 / det;

        if (det == 0) return mat2();
        
        result.row1.x = row2.y * inv_det;
        result.row1.y = -row1.y * inv_det;
        result.row2.x = -row2.x * inv_det;
        result.row2.y = row1.x * inv_det;

        return result;
    }

    __host__ __device__ mat2 transpose() {

        mat2<T> result;

        result.row1.x = row1.x;
        result.row1.y = row2.x;
        result.row2.x = row1.y;
        result.row2.y = row2.y;

        return result;
    }

    // Operators
    __host__ __device__ mat2 operator+(const mat2& matrix) const {
        return sum(matrix);
    }

    __host__ __device__ mat2 operator+(T value) const {
        return sum(mat2<T>::full(value));
    }

    __host__ __device__ mat2 operator-(const mat2& matrix) const {
        return subtract(matrix);
    }

    __host__ __device__ mat2 operator-(T value) const {
        return subtract(mat2<T>::full(value));
    }

    __host__ __device__ mat2 operator*(const mat2& matrix) const {
        return hadamard(matrix);
    }

    __host__ __device__ mat2 operator*(T value) const {
        return hadamard(mat2<T>::full(value));
    }

    __host__ __device__ mat2 operator/(const mat2& matrix) const {
        return div(matrix);
    }

    __host__ __device__ mat2 operator/(T value) const {
        return div(mat2<T>::full(value));
    }
};

using mat2i = mat2<int>;
using mat2f = mat2<float>;
using mat2d = mat2<double>;

template <typename T>
struct mat3 {

    vec3<T> row1, row2, row3;

    __host__ __device__ mat3(T value) {
        row1 = vec3<T>(value, 0, 0);
        row2 = vec3<T>(0, value, 0);
        row3 = vec3<T>(0, 0, value);
    }

    __host__ __device__ mat3() {
        row1 = vec3<T>(0);
        row2 = vec3<T>(0);
        row3 = vec3<T>(0);
    }

    __host__ __device__ ~mat3() = default;

    __host__ __device__ static mat3 full(T value) {

        mat3 matrix(value);

        matrix.row1.y = value;
        matrix.row1.z = value;
        matrix.row2.x = value;
        matrix.row2.z = value;
        matrix.row3.x = value;
        matrix.row3.y = value;

        return matrix;
    }

    // Operations
    __host__ __device__ mat3 sum(const mat3& matrix) {

        mat3<T> result;

        result.row1 = row1 + matrix.row1;
        result.row2 = row2 + matrix.row2;
        result.row3 = row3 + matrix.row3;

        return result;
    }

    __host__ __device__ mat3 subtract(const mat3& matrix) {

        mat3<T> result;

        result.row1 = row1 - matrix.row1;
        result.row2 = row2 - matrix.row2;
        result.row3 = row3 - matrix.row3;

        return result;
    }

    __host__ __device__ mat3 hadamard(const mat3& matrix) {

        mat3<T> result;

        result.row1 = row1 * matrix.row1;
        result.row2 = row2 * matrix.row2;
        result.row3 = row3 * matrix.row3;

        return result;
    }

    __host__ __device__ mat3 div(const mat3& matrix) {

        mat3<T> result;

        result.row1 = row1 / matrix.row1;
        result.row2 = row2 / matrix.row2;
        result.row3 = row3 / matrix.row3;

        return result;
    }

    __host__ __device__ mat3 product(const mat3& matrix) {

        mat3<T> result;

        result.row1.x = row1.x * matrix.row1.x + row1.y * matrix.row2.x + row1.z * matrix.row3.x;
        result.row1.y = row1.x * matrix.row1.y + row1.y * matrix.row2.y + row1.z * matrix.row3.y;
        result.row1.z = row1.x * matrix.row1.z + row1.y * matrix.row2.z + row1.z * matrix.row3.z;

        result.row2.x = row2.x * matrix.row1.x + row2.y * matrix.row2.x + row2.z * matrix.row3.x;
        result.row2.y = row2.x * matrix.row1.y + row2.y * matrix.row2.y + row2.z * matrix.row3.y;
        result.row2.z = row2.x * matrix.row1.z + row2.y * matrix.row2.z + row2.z * matrix.row3.z;

        result.row3.x = row3.x * matrix.row1.x + row3.y * matrix.row2.x + row3.z * matrix.row3.x;
        result.row3.y = row3.x * matrix.row1.y + row3.y * matrix.row2.y + row3.z * matrix.row3.y;
        result.row3.z = row3.x * matrix.row1.z + row3.y * matrix.row2.z + row3.z * matrix.row3.z;

        return result;
    }

    __host__ __device__ vec3<T> transform(const vec3<T>& v) {

        vec3<T> result;

        result.x = row1.x * v.x + row1.y * v.y + row1.z * v.z;
        result.y = row2.x * v.x + row2.y * v.y + row2.z * v.z;
        result.z = row3.x * v.x + row3.y * v.y + row3.z * v.z;

        return result;
    }

    __host__ __device__ T determinant() {
        return row1.x * (row2.y * row3.z - row2.z * row3.y) -
               row1.y * (row2.x * row3.z - row2.z * row3.x) +
               row1.z * (row2.x * row3.y - row2.y * row3.x);
    }

    __host__ __device__ mat3 inverse() {

        mat3<T> result;

        T det = determinant();
        T inv_det = 1 / det;

        if (det == 0) return mat3();

        result.row1.x = (row2.y * row3.z - row2.z * row3.y) * inv_det;
        result.row1.y = (row1.z * row3.y - row1.y * row3.z) * inv_det;
        result.row1.z = (row1.y * row2.z - row1.z * row2.y) * inv_det;

        result.row2.x = (row2.z * row3.x - row2.x * row3.z) * inv_det;
        result.row2.y = (row1.x * row3.z - row1.z * row3.x) * inv_det;
        result.row2.z = (row1.z * row2.x - row1.x * row2.z) * inv_det;

        result.row3.x = (row2.x * row3.y - row2.y * row3.x) * inv_det;
        result.row3.y = (row1.y * row3.x - row1.x * row3.y) * inv_det;
        result.row3.z = (row1.x * row2.y - row1.y * row2.x) * inv_det;

        return result;
    }

    __host__ __device__ mat3 transpose() {

        mat3<T> result;

        result.row1.x = row1.x; result.row1.y = row2.x; result.row1.z = row3.x;
        result.row2.x = row1.y; result.row2.y = row2.y; result.row2.z = row3.y;
        result.row3.x = row1.z; result.row3.y = row2.z; result.row3.z = row3.z;

        return result;
    }

    // Operators
    __host__ __device__ mat3 operator+(const mat3& matrix) const {
        return sum(matrix);
    }

    __host__ __device__ mat3 operator+(T value) const {
        return sum(mat3<T>::full(value));
    }

    __host__ __device__ mat3 operator-(const mat3& matrix) const {
        return subtract(matrix);
    }

    __host__ __device__ mat3 operator-(T value) const {
        return subtract(mat3<T>::full(value));
    }

    __host__ __device__ mat3 operator*(const mat3& matrix) const {
        return hadamard(matrix);
    }

    __host__ __device__ mat3 operator*(T value) const {
        return hadamard(mat3<T>::full(value));
    }

    __host__ __device__ mat3 operator/(const mat3& matrix) const {
        return div(matrix);
    }

    __host__ __device__ mat3 operator/(T value) const {
        return div(mat3<T>::full(value));
    }
};

using mat3i = mat3<int>;
using mat3f = mat3<float>;
using mat3d = mat3<double>;

template <typename T>
struct mat4 {

    vec4<T> row1, row2, row3, row4;

    __host__ __device__ mat4(T value) {
        row1 = vec4<T>(value, 0, 0, 0);
        row2 = vec4<T>(0, value, 0, 0);
        row3 = vec4<T>(0, 0, value, 0);
        row4 = vec4<T>(0, 0, 0, value);
    }

    __host__ __device__ mat4() {
        row1 = vec4<T>(0);
        row2 = vec4<T>(0);
        row3 = vec4<T>(0);
        row4 = vec4<T>(0);
    }

    __host__ __device__ ~mat4() = default;

    __host__ __device__ static mat4 full(T value) {

        mat4 matrix(value);

        matrix.row1.y = value;
        matrix.row1.z = value;
        matrix.row1.w = value;
        matrix.row2.x = value;
        matrix.row2.z = value;
        matrix.row2.w = value;
        matrix.row3.x = value;
        matrix.row3.y = value;
        matrix.row3.w = value;
        matrix.row4.x = value;
        matrix.row4.y = value;
        matrix.row4.z = value;

        return matrix;
    }

    // Operations
    __host__ __device__ mat4 sum(const mat4& matrix) const {

        mat4<T> result;

        result.row1 = row1 + matrix.row1;
        result.row2 = row2 + matrix.row2;
        result.row3 = row3 + matrix.row3;
        result.row4 = row4 + matrix.row4;

        return result;
    }

    __host__ __device__ mat4 subtract(const mat4& matrix) const {

        mat4<T> result;

        result.row1 = row1 - matrix.row1;
        result.row2 = row2 - matrix.row2;
        result.row3 = row3 - matrix.row3;
        result.row4 = row4 - matrix.row4;

        return result;
    }

    __host__ __device__ mat4 hadamard(const mat4& matrix) const {

        mat4<T> result;

        result.row1 = row1 * matrix.row1;
        result.row2 = row2 * matrix.row2;
        result.row3 = row3 * matrix.row3;
        result.row4 = row4 * matrix.row4;

        return result;
    }

    __host__ __device__ mat4 div(const mat4& matrix) const {

        mat4<T> result;

        result.row1 = row1 / matrix.row1;
        result.row2 = row2 / matrix.row2;
        result.row3 = row3 / matrix.row3;
        result.row4 = row4 / matrix.row4;

        return result;
    }

    __host__ __device__ mat4 product(const mat4& matrix) const {

        mat4<T> result;

        result.row1.x = row1.x * matrix.row1.x + row1.y * matrix.row2.x + row1.z * matrix.row3.x + row1.w * matrix.row4.x;
        result.row1.y = row1.x * matrix.row1.y + row1.y * matrix.row2.y + row1.z * matrix.row3.y + row1.w * matrix.row4.y;
        result.row1.z = row1.x * matrix.row1.z + row1.y * matrix.row2.z + row1.z * matrix.row3.z + row1.w * matrix.row4.z;
        result.row1.w = row1.x * matrix.row1.w + row1.y * matrix.row2.w + row1.z * matrix.row3.w + row1.w * matrix.row4.w;

        result.row2.x = row2.x * matrix.row1.x + row2.y * matrix.row2.x + row2.z * matrix.row3.x + row2.w * matrix.row4.x;
        result.row2.y = row2.x * matrix.row1.y + row2.y * matrix.row2.y + row2.z * matrix.row3.y + row2.w * matrix.row4.y;
        result.row2.z = row2.x * matrix.row1.z + row2.y * matrix.row2.z + row2.z * matrix.row3.z + row2.w * matrix.row4.z;
        result.row2.w = row2.x * matrix.row1.w + row2.y * matrix.row2.w + row2.z * matrix.row3.w + row2.w * matrix.row4.w;

        result.row3.x = row3.x * matrix.row1.x + row3.y * matrix.row2.x + row3.z * matrix.row3.x + row3.w * matrix.row4.x;
        result.row3.y = row3.x * matrix.row1.y + row3.y * matrix.row2.y + row3.z * matrix.row3.y + row3.w * matrix.row4.y;
        result.row3.z = row3.x * matrix.row1.z + row3.y * matrix.row2.z + row3.z * matrix.row3.z + row3.w * matrix.row4.z;
        result.row3.w = row3.x * matrix.row1.w + row3.y * matrix.row2.w + row3.z * matrix.row3.w + row3.w * matrix.row4.w;

        result.row4.x = row4.x * matrix.row1.x + row4.y * matrix.row2.x + row4.z * matrix.row3.x + row4.w * matrix.row4.x;
        result.row4.y = row4.x * matrix.row1.y + row4.y * matrix.row2.y + row4.z * matrix.row3.y + row4.w * matrix.row4.y;
        result.row4.z = row4.x * matrix.row1.z + row4.y * matrix.row2.z + row4.z * matrix.row3.z + row4.w * matrix.row4.z;
        result.row4.w = row4.x * matrix.row1.w + row4.y * matrix.row2.w + row4.z * matrix.row3.w + row4.w * matrix.row4.w;

        return result;
    }

    __host__ __device__ vec4<T> transform(const vec4<T>& v) const {

        vec4<T> result;

        result.x = row1.x * v.x + row1.y * v.y + row1.z * v.z + row1.w * v.w;
        result.y = row2.x * v.x + row2.y * v.y + row2.z * v.z + row2.w * v.w;
        result.z = row3.x * v.x + row3.y * v.y + row3.z * v.z + row3.w * v.w;
        result.w = row4.x * v.x + row4.y * v.y + row4.z * v.z + row4.w * v.w;

        return result;
    }

    __host__ __device__ T determinant() {

        T det1 = row1.x * (row2.y * row3.z * row4.w - row2.y * row3.w * row4.z - row3.y * row2.z * row4.w + row3.y * row2.w * row4.z + row4.y * row2.z * row3.w - row4.y * row2.w * row3.z);
        T det2 = row1.y * (row2.x * row3.z * row4.w - row2.x * row3.w * row4.z - row3.x * row2.z * row4.w + row3.x * row2.w * row4.z + row4.x * row2.z * row3.w - row4.x * row2.w * row3.z);
        T det3 = row1.z * (row2.x * row3.y * row4.w - row2.x * row3.w * row4.y - row3.x * row2.y * row4.w + row3.x * row2.w * row4.y + row4.x * row2.y * row3.w - row4.x * row2.w * row3.y);
        T det4 = row1.w * (row2.x * row3.y * row4.z - row2.x * row3.z * row4.y - row3.x * row2.y * row4.z + row3.x * row2.z * row4.y + row4.x * row2.y * row3.z - row4.x * row2.z * row3.y);

        return det1 - det2 + det3 - det4;
    }

    __host__ __device__ mat4 inverse() {

        mat4<T> result;

        T det = determinant();
        T inv_det = 1 / det;

        if (det == 0) return mat4<T>();

        // Cofactor matrix
        result.row1.x =  (row2.y * row3.z * row4.w - row2.y * row3.w * row4.z - row3.y * row2.z * row4.w + row3.y * row2.w * row4.z + row4.y * row2.z * row3.w - row4.y * row2.w * row3.z);
        result.row1.y = -(row2.x * row3.z * row4.w - row2.x * row3.w * row4.z - row3.x * row2.z * row4.w + row3.x * row2.w * row4.z + row4.x * row2.z * row3.w - row4.x * row2.w * row3.z);
        result.row1.z =  (row2.x * row3.y * row4.w - row2.x * row3.w * row4.y - row3.x * row2.y * row4.w + row3.x * row2.w * row4.y + row4.x * row2.y * row3.w - row4.x * row2.w * row3.y);
        result.row1.w = -(row2.x * row3.y * row4.z - row2.x * row3.z * row4.y - row3.x * row2.y * row4.z + row3.x * row2.z * row4.y + row4.x * row2.y * row3.z - row4.x * row2.z * row3.y);

        result.row2.x = -(row1.y * row3.z * row4.w - row1.y * row3.w * row4.z - row3.y * row1.z * row4.w + row3.y * row1.w * row4.z + row4.y * row1.z * row3.w - row4.y * row1.w * row3.z);
        result.row2.y =  (row1.x * row3.z * row4.w - row1.x * row3.w * row4.z - row3.x * row1.z * row4.w + row3.x * row1.w * row4.z + row4.x * row1.z * row3.w - row4.x * row1.w * row3.z);
        result.row2.z = -(row1.x * row3.y * row4.w - row1.x * row3.w * row4.y - row3.x * row1.y * row4.w + row3.x * row1.w * row4.y + row4.x * row1.y * row3.w - row4.x * row1.w * row3.y);
        result.row2.w =  (row1.x * row3.y * row4.z - row1.x * row3.z * row4.y - row3.x * row1.y * row4.z + row3.x * row1.z * row4.y + row4.x * row1.y * row3.z - row4.x * row1.z * row3.y);

        result.row3.x =  (row1.y * row2.z * row4.w - row1.y * row2.w * row4.z - row2.y * row1.z * row4.w + row2.y * row1.w * row4.z + row4.y * row1.z * row2.w - row4.y * row1.w * row2.z);
        result.row3.y = -(row1.x * row2.z * row4.w - row1.x * row2.w * row4.z - row2.x * row1.z * row4.w + row2.x * row1.w * row4.z + row4.x * row1.z * row2.w - row4.x * row1.w * row2.z);
        result.row3.z =  (row1.x * row2.y * row4.w - row1.x * row2.w * row4.y - row2.x * row1.y * row4.w + row2.x * row1.w * row4.y + row4.x * row1.y * row2.w - row4.x * row1.w * row2.y);
        result.row3.w = -(row1.x * row2.y * row4.z - row1.x * row2.z * row4.y - row2.x * row1.y * row4.z + row2.x * row1.z * row4.y + row4.x * row1.y * row2.z - row4.x * row1.z * row2.y);

        result.row4.x = -(row1.y * row2.z * row3.w - row1.y * row2.w * row3.z - row2.y * row1.z * row3.w + row2.y * row1.w * row3.z + row3.y * row1.z * row2.w - row3.y * row1.w * row2.z);
        result.row4.y =  (row1.x * row2.z * row3.w - row1.x * row2.w * row3.z - row2.x * row1.z * row3.w + row2.x * row1.w * row3.z + row3.x * row1.z * row2.w - row3.x * row1.w * row2.z);
        result.row4.z = -(row1.x * row2.y * row3.w - row1.x * row2.w * row3.y - row2.x * row1.y * row3.w + row2.x * row1.w * row3.y + row3.x * row1.y * row2.w - row3.x * row1.w * row2.y);
        result.row4.w =  (row1.x * row2.y * row3.z - row1.x * row2.z * row3.y - row2.x * row1.y * row3.z + row2.x * row1.z * row3.y + row3.x * row1.y * row2.z - row3.x * row1.z * row2.y);

        // Now, take the transpose of the cofactor matrix to get the adjugate
        result = result.transpose();

        // Multiply the adjugate matrix by the inverse of the determinant
        result.row1 = result.row1 * inv_det;
        result.row2 = result.row2 * inv_det;
        result.row3 = result.row3 * inv_det;
        result.row4 = result.row4 * inv_det;

        return result;
    }

    __host__ __device__ mat4 transpose() {

        mat4<T> result;

        result.row1.x = row1.x; result.row1.y = row2.x; result.row1.z = row3.x; result.row1.w = row4.x;
        result.row2.x = row1.y; result.row2.y = row2.y; result.row2.z = row3.y; result.row2.w = row4.y;
        result.row3.x = row1.z; result.row3.y = row2.z; result.row3.z = row3.z; result.row3.w = row4.z;
        result.row4.x = row1.w; result.row4.y = row2.w; result.row4.z = row3.w; result.row4.w = row4.w;
        
        return result;
    }

    // Operators
    __host__ __device__ mat4 operator+(const mat4& matrix) const {
        return sum(matrix);
    }

    __host__ __device__ mat4 operator+(T value) const {
        return sum(mat4<T>::full(value));
    }

    __host__ __device__ mat4 operator-(const mat4& matrix) const {
        return subtract(matrix);
    }

    __host__ __device__ mat4 operator-(T value) const {
        return subtract(mat4<T>::full(value));
    }

    __host__ __device__ mat4 operator*(const mat4& matrix) const {
        return product(matrix);
    }

    __host__ __device__ mat4 operator*(T value) const {
        return hadamard(mat4<T>::full(value));
    }

    __host__ __device__ mat4 operator/(const mat4& matrix) const {
        return div(matrix);
    }

    __host__ __device__ mat4 operator/(T value) const {
        return div(mat4<T>::full(value));
    }
};

using mat4i = mat4<int>;
using mat4f = mat4<float>;
using mat4d = mat4<double>;

}