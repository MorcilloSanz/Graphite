# Graphite
Ray tracing and path tracing API written in CUDA from scratch for research purposes.

> [!IMPORTANT] 
This project is still under development

## Features
* **Linear algebra library:** for both, host and device.
* **CUDA Path tracing API:** vertex buffer, index buffer, frame buffer, textures...
* **Fixed graphics pipeline:** CUDA Kernels for Vertex and Fragment shaders.
* **Physically based rendering (PBR):** Metallic-Roughness workflow.
* **Load 3D models:** render glTF 2.0 3D models.
* **High dynamic range (HDR):** Reinhard tone mapping.
* **360 HDRI skybox.**
* **Gamma correction.**

# Dependencies
* **stb_image:** for reading and writing images.
* **tinygltf:** for reading glTF 2.0 models.
* **CUDA Toolkit:** nvcc for compiling CUDA code.

# Output
![](img/output.png)
