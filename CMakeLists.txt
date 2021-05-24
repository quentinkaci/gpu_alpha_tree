cmake_minimum_required(VERSION 3.19)
project(gpu_alpha_tree CXX CUDA)

find_package(PNG REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(gpu_alpha_tree main.cu cc_labelling.cu cc_labelling.cuh)

set_target_properties(
        gpu_alpha_tree
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gpu_alpha_tree PRIVATE
        ${PNG_LIBRARY}
        )