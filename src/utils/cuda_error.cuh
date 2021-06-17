#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define checkCudaError()                                                        \
    if (cudaGetLastError() != cudaSuccess)                                      \
    {                                                                           \
        cudaError_t err = cudaGetLastError();                                   \
        printf("(%s, line: %d)", __FUNCTION__, __LINE__);                       \
        printf("Error %s: %s", cudaGetErrorName(err), cudaGetErrorString(err)); \
        std::exit(1);                                                           \
    }
