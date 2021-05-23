#pragma once

#include <cuda_runtime.h>

void initialization_step(const int* nn_list, int max_len, int* residual_list, int* labels, int i);

void analysis_step(int* labels, int i);

__global__ void reduction_step(const int* residual_list, int max_len, int* labels, int height, int width);