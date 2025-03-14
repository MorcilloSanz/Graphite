#include "pbr.cuh"

namespace gph
{

__device__ float clamp(float value, float minVal, float maxVal) {
    return max(minVal, min(value, maxVal));
}

__device__ vec3<float> reflect(const vec3<float>& wo, const vec3<float>& normal) {

    float dot = normal.dot(wo);
    vec3<float> wi = wo - normal * 2.f * dot;
    
    return wi.normalize();
}

__device__ vec3<float> refract(const vec3<float>& wo, const vec3<float>& normal, float eta) {

    float dotNI = normal.dot(wo);
    float k = 1.f - eta * eta * (1.f - dotNI * dotNI);
    
    if(k < 0.f)
        return vec3<float>(0.f);

    return wo * eta - normal * (eta * dotNI + sqrt(k));
}

__device__ float distributionGGX(const vec3<float>& N, const vec3<float>& H, float roughness) {

    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(N.dot(H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = M_PI * denom * denom;

    return nom / denom;
}

__device__ vec3<float> fresnelSchlick(float cosTheta, const vec3<float>& F0) {
    return F0 + (vec3<float>(1.0f) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

__device__ float G1(const vec3<float>& V, float roughness) {

    float alpha2 = roughness * roughness;
    float lenqs = V.x * V.x + V.y * V.y;
    float lambda = 0.5 * (-1 + sqrt(1 + (alpha2 * lenqs) / (V.z * V.z)));

    return 1 / (1 + lambda);
}

__device__ float G2(const vec3<float>& V, const vec3<float>& L, float roughness) {
    return G1(V, roughness) * G1(L, roughness);
}

__device__ vec3<float> sampleGGXVNDF(const vec3<float>& wo, float roughness, float U1, float U2) {

    vec3<float> vh = (vec3<float>(roughness, roughness, 1.f) * wo).normalize();
    float lensq = vh.x * vh.x + vh.y * vh.y;
    float invSqrt = 1.f / sqrt(lensq);

    vec3<float> T1 = lensq > 0 ? vec3<float>(-vh.y, vh.x, 0) * invSqrt : vec3<float>(1.f, 0.f, 0.f);
    vec3<float> T2 = vh.cross(T1);

    float r = sqrt(U1);
    float phi = 2.0 * M_PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);

    t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;

    vec3<float> nh = T1 * t1 + T2 * t2 + vh * sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2));
    vec3<float> ne = vec3<float>(roughness, roughness, 1.f) * vec3<float>(nh.x, nh.y, max(0.f, nh.z));

    return ne.normalize();
}

__device__ vec3<float> monteCarloGGX(
    const vec3<float>& normal, 
    const vec3<float>& wo, 
    const vec3<float>& wi, 
    const vec3<float>& F0, 
    float roughness) {

    vec3<float> F = fresnelSchlick(max(wo.dot(wi), 0.0), F0);
    return F * G2(wo, wi, roughness) / G1(wo, roughness);
}
    
}