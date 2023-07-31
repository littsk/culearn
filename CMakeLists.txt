cmake_minimum_required(VERSION 3.16)
project(culearn)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release) 
endif()

# 设置c++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDA REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "sm80")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>: -G -g -O0>)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
endif()



add_subdirectory(cpp_programming_guide)
add_subdirectory(dnn)
