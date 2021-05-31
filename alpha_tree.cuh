#pragma once

#include "image.hh"

#include <thrust/sort.h>

using namespace utils;

inline __global__ void init_parent(int* parent, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    parent[x + y * width] = x + y * width;
}

inline __device__ double l2_dist(RGBPixel src, RGBPixel dst)
{
    return sqrt(pow(dst.r - src.r, 2) + pow(dst.g - src.g, 2) + pow(dst.b - src.b, 2));
}

inline __device__ int find(const int* parent, int val)
{
    int p = val;
    int q = parent[val];

    while (p != q)
    {
        p = q;
        q = parent[p];
    }

    return p;
}

template <unsigned int BlockHeight>
inline __global__ void build_alpha_tree_col(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
        return;

    double weights[BlockHeight - 1];
    int sources[BlockHeight - 1];

    int node_offset = (2 * BlockHeight - 1) * (x + blockIdx.y);

    for (int i = 0; i < BlockHeight - 1; ++i)
    {
        double dist = l2_dist(image[x + (y + i) * width], image[x + (y + i + 1) * width]);
        weights[i] = dist;

        sources[i] = i + node_offset;
    }
    thrust::sort_by_key(thrust::device, weights, weights + (BlockHeight - 1), sources);

    int nb_pix_col = min(y - height, BlockHeight - 1);
    for (int i = 0; i < nb_pix_col; ++i)
    {
        int p = sources[i];
        int q = p + 1;
        double w = weights[i];

        int rp = find(parent, p);
        int rq = find(parent, q);

        if (rp != rq)
        {
            int new_node = node_offset + BlockHeight + i;

            parent[rp] = new_node;
            parent[rq] = new_node;

            levels[new_node] = w;
        }
    }
}