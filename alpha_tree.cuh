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

template <int BlockHeight>
inline __global__ void build_alpha_tree_col(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    if (x >= width || y >= height)
        return;

    //    if (x != 1 || y != 0)
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

inline __device__ int find_last_leq(const int* parent, const double* levels, int node, double val)
{
    int p = node;

    while (levels[parent[p]] <= val)
        p = parent[p];

    return p;
}

inline __device__ int merge(int* parent, const double* levels, int p, int q)
{
    // FIXME Dirty trick
    if (levels[p] == 0 && levels[q] == 0)
    {
        parent[q] = parent[p];
        return parent[p];
    }

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

template <int BlockHeight>
inline __global__ void merge_alpha_tree_col(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    if (x >= width || y >= height)
        return;

    //    printf("x: %d, y: %d\n", x, y);

    int nb_pix_col = min(height - y, BlockHeight);

    int leaf_offset;
    if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
        leaf_offset = BlockHeight * width * blockIdx.y + nb_pix_col * x;
    else
        leaf_offset = BlockHeight * (x + blockIdx.y * width);

    //    printf("Leaf offset: %d\n", leaf_offset);

    // Merge with column on the right
    int rl = find(parent, leaf_offset);
    int rr = find(parent, leaf_offset + BlockHeight);

    // Merge root node
    merge(parent, levels, rl, rr);

    // Iterate on border edges
    for (int i = leaf_offset; i < leaf_offset + nb_pix_col; ++i)
    {
        // Merge with column on the right
        int p1 = i;
        int p2 = i + BlockHeight;
        double dist = l2_dist(image[x + (y + i) * width], image[(x + 1) + (y + i) * width]);

        //        if (i == 1)
        //            printf("src: %d, dst: %d, w: %f\n", p1, p2, dist);

        //        printf("(src: %d, dst: %d, w: %f)\n", p1, p2, dist);

        int n1 = find_last_leq(parent, levels, p1, dist);
        int n2 = find_last_leq(parent, levels, p2, dist);

        //        if (i == 1)
        //            printf("n1: %d, n2: %d\n", n1, n2);

        // FIXME Wrong if the edge has a higher weight than the root of sub-trees.
        //       In this case we have to create a new node: Where to store it ?
        int a = parent[n1];
        int b = parent[n2];

        //        if (i == 1)
        //            printf("a: %d, b: %d\n", a, b);

        int n = merge(parent, levels, n1, n2);

        if (levels[a] > levels[b])
        {
            int tmp = a;
            a = b;
            b = tmp;
        }

        parent[n] = a;

        while (a != b)
        {
            if (levels[a] == levels[b])
            {
                b = parent[b];
                n = merge(parent, levels, a, b);
                a = parent[a];
            }
            else
                a = parent[a];

            if (levels[a] > levels[b])
            {
                parent[n] = b;

                int tmp = a;
                a = b;
                b = tmp;
            }
        }
    }
}