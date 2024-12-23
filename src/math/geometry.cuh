#pragma once

#include "linalg.cuh"

#define EPSILON 0.0000001

namespace gph
{

template <typename T>
struct Triangle {

    vec3<T> v1, v2, v3;

    __gph__ Triangle(const vec3<T>& _v1, const vec3<T>& _v2, const vec3<T>& _v3)
        : v1(_v1), v2(_v2), v3(_v3) {
    }

    __gph__ Triangle() = default;
    __gph__ ~Triangle() = default;
};

template <typename T>
struct Ray {

    struct HitInfo {

        vec3<T> intersection;
        vec3<T> normal;
        T distance;
        bool hit;

        __gph__ HitInfo(const vec3<T>& _intersection, const vec3<T>& _normal, 
            T _distance, bool _hit) : intersection(_intersection), normal(_normal), 
            distance(_distance), hit(_hit) {
        }

        __gph__ HitInfo() = default;
        __gph__ ~HitInfo() = default;
    };

    vec3<T> origin;
    vec3<T> direction;

    __gph__ Ray(const vec3<T>& _origin, const vec3<T>& _direction)
        : origin(_origin), direction(_direction) {
    }

    __gph__ Ray() = default;
    __gph__ ~Ray() = default;

    __gph__ vec3<T> evaluate(T lambda) {
        return origin + lambda * direction;
    }

    __gph__ HitInfo intersects(const Triangle<T>& triangle) {

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

        float u = inv_det * dot(s, ray_cross_e2);
        if (u < 0 || u > 1) return hitInfo;

        vec3 s_cross_e1 = s.cross(edge1);
        float v = inv_det * direction.cross(s_cross_e1);
        if (v < 0 || u + v > 1) return hitInfo;

        float t = inv_det * edge2.cross(s_cross_e1);
        if (t > EPSILON) {
            hitInfo.intersection = vec3(origin + direction * t);
            hitInfo.dist = t;
            hitInfo.normal = normalize(cross(edge2, edge1));
            hitInfo.hit = true;
        }

        return hitInfo;
    }
};

}