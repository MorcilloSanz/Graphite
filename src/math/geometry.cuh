#pragma once

#include "linalg.cuh"

namespace ghp
{
    
template <typename T>
struct Ray {

    vec3<T> origin;
    vec3<T> direction;

    __ghp__ Ray(const vec3<T>& _origin, const vec3<T>& _direction)
        : origin(_origin), direction(_direction) {
    }

    __ghp__ Ray() = default;
    __ghp__ ~Ray() = default;

    __ghp__ vec3<T> evaluate(T lambda) {
        return origin + lambda * direction;
    }
};

}