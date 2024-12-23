#[[
    MIT License

    Copyright (c) 2024 Alberto Morcillo Sanz

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
]]

cmake_minimum_required(VERSION 3.10.0...3.22)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(Graphite LANGUAGES CXX CUDA)

# std c++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda std c++
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include src
include_directories(src)

# Files
set(CXX_HEADERS 
    src/vendor/stb_image.h
    src/vendor/stb_image_write.h
)

set(CUDA_HEADERS
    src/buffer/buffer.cuh
    src/buffer/kernel/kernel.cuh
    src/math/linalg.cuh
    src/math/geometry.cuh
)

# Source Files
set(CXX_SOURCES

)

set(CUDA_SOURCES
	src/main.cu
    src/buffer/buffer.cu
)

# Add CUDA executable
add_executable(${PROJECT_NAME} ${CXX_SOURCES} ${CXX_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA executable with CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})