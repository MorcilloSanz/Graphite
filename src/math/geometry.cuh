#pragma once

#include "linalg.cuh"

#define EPSILON 0.0000001

namespace gph
{

template <typename T>
struct Triangle {

    vec3<T> v1, v2, v3;

    __host__ __device__ Triangle(const vec3<T>& _v1, const vec3<T>& _v2, const vec3<T>& _v3)
        : v1(_v1), v2(_v2), v3(_v3) {
    }

    __host__ __device__ Triangle() = default;
    __host__ __device__ ~Triangle() = default;
};

template <typename T>
__host__ __device__ vec3<T> barycentric(vec3<T> p, Triangle<T> triangle) {

    T denom = (triangle.v2.y - triangle.v3.y) * (triangle.v1.x - triangle.v3.x) + (triangle.v3.x - triangle.v2.x) * (triangle.v1.y - triangle.v3.y);
    T alpha = (triangle.v2.y - triangle.v3.y) * (p.x - triangle.v3.x) + (triangle.v3.x - triangle.v2.x) * (p.y - triangle.v3.y);
    T beta = (triangle.v3.y - triangle.v1.y) * (p.x - triangle.v3.x) + (triangle.v1.x - triangle.v3.x) * (p.y - triangle.v3.y);
    
    alpha /= denom;
    beta /= denom;
    
    T gamma = 1.0 - alpha - beta;

    return vec3<T>(alpha, beta, gamma);
}

template <typename T>
struct Ray {

    struct HitInfo {

        vec3<T> intersection;
        vec3<T> normal;
        T distance;
        bool hit;

        __host__ __device__ HitInfo(const vec3<T>& _intersection, const vec3<T>& _normal, 
            T _distance, bool _hit) : intersection(_intersection), normal(_normal), 
            distance(_distance), hit(_hit) {
        }

        __host__ __device__ HitInfo() = default;
        __host__ __device__ ~HitInfo() = default;
    };

    vec3<T> origin;
    vec3<T> direction;

    __host__ __device__ Ray(const vec3<T>& _origin, const vec3<T>& _direction)
        : origin(_origin), direction(_direction) {
    }

    __host__ __device__ Ray() = default;
    __host__ __device__ ~Ray() = default;

    __host__ __device__ vec3<T> evaluate(T lambda) {
        return origin + lambda * direction;
    }

    __host__ __device__ HitInfo intersects(const Triangle<T>& triangle) {

        HitInfo hitInfo;
        hitInfo.intersection = vec3<T>(0.0);
        hitInfo.hit = false;

        vec3<T> edge1 = triangle.v2 - triangle.v1;
        vec3<T> edge2 = triangle.v3 - triangle.v1;
        vec3<T> ray_cross_e2 = direction.cross(edge2);

        float det = edge1.dot(ray_cross_e2);
        if (det > -EPSILON && det < EPSILON) 
            return hitInfo;

        float inv_det = 1.0 / det;
        vec3<T> s = origin - triangle.v1;

        float u = inv_det * s.dot(ray_cross_e2);
        if (u < 0 || u > 1) 
            return hitInfo;

        vec3<T> s_cross_e1 = s.cross(edge1);
        float v = direction.dot(s_cross_e1) * inv_det;
        if (v < 0 || u + v > 1) 
            return hitInfo;

        float t = edge2.dot(s_cross_e1) * inv_det;
        if (t > EPSILON) {
            hitInfo.intersection = vec3<T>(origin + direction * t);
            hitInfo.distance = t;

            vec3<T> normal = edge1.cross(edge2); // CCW
            if (normal.dot(direction) > 0)
                normal = normal * -1;

            hitInfo.normal = normal / normal.module();
            hitInfo.hit = true;
        }

        return hitInfo;
    }

    /**
     * Casts a ray from a screen coordinate (x, y) into 3D space.
     *
     * @tparam T Data type for the ray components.
     * @param x Horizontal screen coordinate.
     * @param y Vertical screen coordinate.
     * @param width Screen width in pixels.
     * @param height Screen height in pixels.
     * @return A Ray<T> representing the ray's origin and direction.
     */
    __device__ static Ray<T> castRayOrtho(int x, int y, unsigned int width, unsigned int height) {

        T aspectRatio = static_cast<float>(width) / static_cast<float>(height);

        vec3<T> origin = {
            (2.0f * x / width - 1.0f) * aspectRatio, // Escalar x por el aspect ratio
            1.0f - 2.0f * y / height,                 // Invertir y para que vaya de arriba a abajo
            1.f 
        };

        vec3<T> direction = { 0.0f, 0.0f, -1.0f };
        Ray<T> ray(origin, direction);

        return ray;
    }

    /**
     * Casts a ray from a screen coordinate (x, y) into 3D space.
     *
     * @tparam T Data type for the ray components.
     * @param x Horizontal screen coordinate.
     * @param y Vertical screen coordinate.
     * @param width Screen width in pixels.
     * @param height Screen height in pixels.
     * @param fovy fovy in degrees.
     * @return A Ray<T> representing the ray's origin and direction.
     */
    __device__ static Ray<T> castRayPerspective(int x, int y, unsigned int width, unsigned int height, float fovy) {

        vec3<T> cameraPos = { 0.0, 0.0, 2.0 };
        vec3<T> cameraTarget = { 0.0, 0.0, 0.0 };
        vec3<T> cameraUp = { 0.0, 1.0, 0.0 };

        vec3<T> forward = (cameraTarget - cameraPos).normalize();
        vec3<T> right = forward.cross(cameraUp).normalize();
        vec3<T> up = right.cross(forward);

        vec2<T> imageSize = { static_cast<T>(width), static_cast<T>(height) };
        vec2<T> ndc = { 2.0 * x / width - 1.0, 1.0 - 2.0 * y / height };

        T fov = fovy * M_PI / 180;
        T aspectRatio = static_cast<T>(width) / static_cast<T>(height);

        T imagePlaneX = ndc.x * aspectRatio * tan(fov / 2.0);
        T imagePlaneY = ndc.y * tan(fov / 2.0);

        vec3<T> origin = cameraPos;
        vec3<T> direction = (right * imagePlaneX + up * imagePlaneY + forward).normalize();
        Ray<T> ray(origin, direction);

        return ray;
    }
};

}