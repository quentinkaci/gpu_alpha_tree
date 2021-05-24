#pragma once

#include <iostream>

inline void _abortError(const char* msg, const char* fname)
{
    cudaError_t err = cudaGetLastError();
    std::cerr << "Error: " << err << std::endl;
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__)