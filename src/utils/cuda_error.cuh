#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <spdlog/spdlog.h>

#define checkCudaError()                                                               \
    if (cudaGetLastError() != cudaSuccess)                                             \
    {                                                                                  \
        cudaError_t err = cudaGetLastError();                                          \
        spdlog::error("({}, line: {})", __FUNCTION__, __LINE__);                       \
        spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err)); \
        std::exit(1);                                                                  \
    }