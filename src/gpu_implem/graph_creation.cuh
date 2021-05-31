#pragma once

#include "utils/image.cuh"

#include <cuda_runtime.h>

using namespace utils;

__global__ void create_graph_4(RGBPixel* pixels, int* nn_list, int height, int width);