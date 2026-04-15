# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# cmake_minimum_required(VERSION 2.8.7)
# project(pointpillar)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


# set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
# set(CMAKE_BUILD_TYPE "Release")
# # set(CMAKE_BUILD_TYPE "Debug")
# set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

# find_package(CUDA REQUIRED)

# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3")
# set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g")

# set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3")
# set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
#     -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
# )

# include_directories(
#     $ENV{CUDA_Inc}
#     $ENV{TensorRT_Inc}
#     src
# 	src/pointpillar
# )

# link_directories(
#     $ENV{CUDA_Lib}
#     $ENV{TensorRT_Lib}
#     build
# )

# file(GLOB_RECURSE CORE_FILES 
#     src/pointpillar/*.cu 
#     src/pointpillar/*.cpp
#     src/common/tensor.cu
#     src/common/tensorrt.cpp
# )

# ####################### libpointpillar_core.so ##########################
# cuda_add_library(pointpillar_core SHARED 
#   	${CORE_FILES}
# )

# target_link_libraries(pointpillar_core
# 	libnvinfer.so
# )
# #######################################################################


# ####################### pointpillar program #############################
# cuda_add_executable(pointpillar 
# 	src/main.cpp
# )

# target_link_libraries(pointpillar
# 	pointpillar_core
# )
# #######################################################################
cmake_minimum_required(VERSION 3.18)

project(pointpillar LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++ / CUDA standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Jetson Orin NX = SM87
set(CMAKE_CUDA_ARCHITECTURES 87)

# 환경변수 사용
set(TensorRT_Lib $ENV{TensorRT_Lib})
set(TensorRT_Inc $ENV{TensorRT_Inc})
set(CUDA_Lib     $ENV{CUDA_Lib})
set(CUDA_Inc     $ENV{CUDA_Inc})

include_directories(
    ${CUDA_Inc}
    ${TensorRT_Inc}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/pointpillar
)

link_directories(
    ${CUDA_Lib}
    ${TensorRT_Lib}
    ${PROJECT_BINARY_DIR}
)

file(GLOB_RECURSE CORE_FILES
    src/pointpillar/*.cu
    src/pointpillar/*.cpp
    src/common/tensor.cu
    src/common/tensorrt.cpp
)

add_library(pointpillar_core SHARED ${CORE_FILES})

set_target_properties(pointpillar_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(pointpillar_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wno-deprecated-declarations -Wno-missing-field-initializers>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -Xcompiler=-fPIC>
)

target_link_libraries(pointpillar_core
    nvinfer
    cudart
)

add_executable(pointpillar src/main.cpp)
target_link_libraries(pointpillar pointpillar_core)