#pragma once

#include "utils/image.cuh"

#include <thrust/sort.h>

using namespace utils;

__global__ void init_parent(int* parent, int height, int width)
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

template <int BlockHeight>
__global__ void build_alpha_tree_col(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    if (x >= width || y >= height)
        return;

    double weights[BlockHeight - 1];
    int sources[BlockHeight - 1];

    int nb_pix_col = min(height - y, BlockHeight);

    int leaf_offset;
    if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
        leaf_offset = BlockHeight * width * blockIdx.y + nb_pix_col * x;
    else
        leaf_offset = BlockHeight * (x + blockIdx.y * width);
    int parent_offset = width * height + 2 * BlockHeight * (x + blockIdx.y * width);

    for (int i = 0; i < (nb_pix_col - 1); ++i)
    {
        double dist = l2_dist(image[x + (y + i) * width], image[x + (y + i + 1) * width]);
        weights[i] = dist;

        sources[i] = leaf_offset + i;
    }
    thrust::sort_by_key(thrust::device, weights, weights + (nb_pix_col - 1), sources);

    for (int i = 0; i < (nb_pix_col - 1); ++i)
    {
        int p = sources[i];
        int q = p + 1;
        double w = weights[i];

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

inline __device__ int find_intersection(const int* parent, const double* levels, int node, double val)
{
    int p = node;

    while (levels[parent[p]] <= val)
    {
        if (parent[p] == p)
            return p;

        p = parent[p];
    }

    return p;
}

inline __device__ int merge(int* parent, const double* levels, int p, int q)
{
    if (levels[q] > levels[p])
    {
        parent[p] = q;
        return q;
    }
    else
    {
        parent[q] = p;
        return p;
    }
}

inline __device__ void canonize_tree(int* parent, const double* levels, int leaves_offset, int nb_leaves)
{
    for (int i = leaves_offset; i < leaves_offset + nb_leaves; ++i)
    {
        int node = i;
        while (parent[node] != node)
        {
            while (parent[node] != parent[parent[node]] && levels[parent[node]] == levels[parent[parent[node]]])
                parent[node] = parent[parent[node]];

            node = parent[node];
        }
    }
}

template <int BlockHeight>
__global__ void merge_alpha_tree_col(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    if (x >= width || y >= height)
        return;

    int nb_pix_col = min(height - y, BlockHeight);

    int leaf_offset;
    if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
        leaf_offset = BlockHeight * width * blockIdx.y + nb_pix_col * x;
    else
        leaf_offset = BlockHeight * (x + blockIdx.y * width);
    int parent_offset = width * height + 2 * BlockHeight * (x + blockIdx.y * width);

    // Merge with column on the right
    int rl = find(parent, leaf_offset);
    int rr = find(parent, leaf_offset + BlockHeight);

    // Merge root node
    merge(parent, levels, rl, rr);

    // Iterate on border edges
    for (int i = leaf_offset; i < leaf_offset + nb_pix_col; ++i)
    {
        // Merge with column on the right
        int p = i;
        int q = i + BlockHeight;
        double dist = l2_dist(image[x + (y + i) * width], image[(x + 1) + (y + i) * width]);

        int c1 = find_intersection(parent, levels, p, dist);
        int c2 = find_intersection(parent, levels, q, dist);

        // FIXME Maybe wrong if the edge has a higher weight than the root of sub-trees: units tests
        int p1 = parent[c1];
        int p2 = parent[c2];

        int n = parent_offset + BlockHeight + i;
        parent[c1] = n;
        parent[c2] = n;
        levels[n] = dist;

        if (levels[p1] > levels[p2])
        {
            int tmp = p1;
            p1 = p2;
            p2 = tmp;
        }

        parent[n] = p1;

        while (p1 != p2)
        {
            if (levels[p1] == levels[p2])
            {
                int p1_ = parent[p1];
                int p2_ = parent[p2];
                n = merge(parent, levels, p1, p2);
                p1 = p1_;
                p2 = p2_;
            }
            else
                p1 = parent[p1];

            if (levels[p1] > levels[p2])
            {
                parent[n] = p2;

                int tmp = p1;
                p1 = p2;
                p2 = tmp;
            }
        }
    }
}