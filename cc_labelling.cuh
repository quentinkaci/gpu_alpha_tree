#pragma once

#include <cuda_runtime.h>

__global__ void initialization_step(const int* nn_list, int max_len, int* residual_list, int* labels, int height, int width);

__global__ void analysis_step(int* labels, int height, int width);

__global__ void reduction_step(const int* residual_list, int max_len, int* labels, int height, int width);