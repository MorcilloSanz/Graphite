#include "pbr.cuh"

namespace gph
{

__device__ float clamp(float value, float minVal, float maxVal) {
    return max(minVal, min(value, maxVal));
}

__device__ void orthonormalBasis(vec3<float> N, vec3<float>& tangent, vec3<float>& bitangent) {
    if (fabs(N.x) > fabs(N.z))
        tangent = vec3<float>(-N.y, N.x, 0.0f).normalize();
    else
        tangent = vec3<float>(0.0f, -N.z, N.y).normalize();

    bitangent = N.cross(tangent);
}

__device__ vec3<float> sampleGGX(vec3<float> N, vec3<float> V, float roughness, curandState& state) {
    float alpha = roughness * roughness;

    // Generamos dos números aleatorios en [0,1]
    float xi1 = curand_uniform(&state);
    float xi2 = curand_uniform(&state);

    // Muestreo de GGX para theta_h y phi_h
    float theta_h = atan(sqrt(alpha * alpha * xi1) / sqrt(1.0 - xi1));
    float phi_h = 2.0 * M_PI * xi2;

    // Convertir a coordenadas cartesianas
    float sinTheta = sin(theta_h);
    float cosTheta = cos(theta_h);
    float sinPhi = sin(phi_h);
    float cosPhi = cos(phi_h);

    // Vector mitad H en espacio local
    vec3<float> H = vec3<float>(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

    // Convertir H al sistema de referencia de la normal N
    vec3<float> tangent, bitangent;
    orthonormalBasis(N, tangent, bitangent);
    H = (tangent * H.x + bitangent * H.y + N * H.z).normalize();

    // Reflexión especular para obtener L
    vec3<float> L = (H * V.dot(H) * 2.0f - V).normalize();

    return L;
}

__device__ float distributionGGX(vec3<float> N, vec3<float> H, float roughness) {

    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(N.dot(H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = M_PI * denom * denom;

    return nom / denom;
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {

    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

__device__ float geometrySmith(vec3<float> N, vec3<float> V, vec3<float> L, float roughness) {

    float NdotV = max(N.dot(V), 0.0);
    float NdotL = max(N.dot(L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

__device__ vec3<float> fresnelSchlick(float cosTheta, vec3<float> F0) {
    return F0 + (vec3<float>(1.0f) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

__device__ vec3<float> specularCookTorrance(vec3<float> H, vec3<float> normal, vec3<float> wo, vec3<float> wi, vec3<float> F, float roughness) {
    
    float NDF = distributionGGX(normal, H, roughness);   
    float G = geometrySmith(normal, wo, wi, roughness);      
    
    vec3<float> numerator = F * NDF * G; 
    float denominator = 4.0 * normal.dot(wo) * normal.dot(wi) + 0.0001;
    vec3<float> specular = numerator / denominator;
    
    return specular;
}
    
}