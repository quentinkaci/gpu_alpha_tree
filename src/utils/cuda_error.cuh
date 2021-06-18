#pragma once

#include <cuda_runtime.h>
#include <iostream>

static cudaError_t err;

#define checkCudaError()                                                        \
    if ((err = cudaGetLastError()) != cudaSuccess)                              \
    {                                                                           \
        printf("(%s, line: %d)", __FUNCTION__, __LINE__);                       \
        printf("Error %s: %s", cudaGetErrorName(err), cudaGetErrorString(err)); \
        std::exit(1);                                                           \
    }
