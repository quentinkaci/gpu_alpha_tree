cmake_minimum_required(VERSION 3.19)
project(gpu_alpha_tree CXX CUDA)

# Set compile flags and sources paths
set(SRC
    src/cpu_implem/alpha_tree_cpu.cc
    src/gpu_implem/alpha_tree_gpu.cu
    src/gpu_implem/cc_labelling.cu
    src/gpu_implem/graph_creation.cu
    src/utils/image.cu)

set(TESTS
    tests/test_example.cc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)
set(OPT_FLAGS "-Ofast -march=native -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address")

# Temp to work on OpenStack
set(CMAKE_PREFIX_PATH "$HOME/afs/libs/")
include_directories("/home/theo.lepage/afs/libs/")

# Find packages: PNG, GoogleBenchmark, CLI11, spdlog
set(Thrust_DIR ".")
find_package(Thrust REQUIRED)
find_package(PNG REQUIRED)

include(FetchContent)
FetchContent_Declare(
        GoogleTest
        URL https://github.com/google/googletest/archive/refs/tags/v1.10.x.tar.gz)
FetchContent_Declare(
        GoogleBenchmark
        URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz)
FetchContent_Declare(
        CLI11
        URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz)
FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz)

if (NOT GoogleBenchmark_POPULATED)
        FetchContent_Populate(GoogleBenchmark)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
        add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(GoogleTest)
enable_testing()
FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(spdlog)

# Create target containing all source files
add_library(gpu_alpha_tree)
target_sources(gpu_alpha_tree PRIVATE ${SRC})
set_target_properties(gpu_alpha_tree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_alpha_tree PUBLIC Thrust::Thrust Thrust ${PNG_LIBRARY} spdlog::spdlog)
target_include_directories(gpu_alpha_tree PUBLIC src/)

# Create target `main`
add_executable(main src/main.cc)
target_link_libraries(main PRIVATE gpu_alpha_tree CLI11::CLI11)

# Create target `bench`
add_executable(bench src/bench.cc)
target_link_libraries(bench PRIVATE gpu_alpha_tree benchmark)

# Create target `tests`
add_executable(tests ${TESTS})
target_link_libraries(tests PRIVATE gpu_alpha_tree gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)