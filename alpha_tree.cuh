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
inline __global__ void build_alpha_tree_col(RGBPixel* image, int* parent, double* levels, unsigned int height, unsigned int width)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = BlockHeight * blockIdx.y;

    if (x >= width || y >= height)
        return;

    //    if (x != 1 || y != 6)
    //        return;

    double weights[BlockHeight - 1];
    int sources[BlockHeight - 1];

    int nb_pix_col = min(height - y, BlockHeight);

    int leaf_offset;
    int parent_offset = width * height;
    if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
    {
        leaf_offset = BlockHeight * width * blockIdx.y + nb_pix_col * x;
        parent_offset += (BlockHeight - 1) * width * blockIdx.y + (nb_pix_col - 1) * x;
    }
    else
    {
        leaf_offset = BlockHeight * (x + blockIdx.y * width);
        parent_offset += (BlockHeight - 1) * (x + blockIdx.y * width);
    }

    //    printf("Leaf offset: %d\n", leaf_offset);
    //    printf("Parent offset: %d\n", parent_offset);

    for (int i = 0; i < (nb_pix_col - 1); ++i)
    {
        double dist = l2_dist(image[x + (y + i) * width], image[x + (y + i + 1) * width]);
        weights[i] = dist;

        //        printf("(src: %d, dst: %d, w: %f)\n", x + (y + i) * width, x + (y + i + 1) * width, dist);

        sources[i] = leaf_offset + i;
    }
    thrust::sort_by_key(thrust::device, weights, weights + (nb_pix_col - 1), sources);

    for (int i = 0; i < (nb_pix_col - 1); ++i)
    {
        int p = sources[i];
        int q = p + 1;
        double w = weights[i];

        //        printf("(%d, %d, %f), %d\n", p, q, w, i);

        int rp = find(parent, p);
        int rq = find(parent, q);

        if (rp != rq)
        {
            int new_node = parent_offset + i;

            parent[rp] = new_node;
            parent[rq] = new_node;

            levels[new_node] = w;
        }
    }
}