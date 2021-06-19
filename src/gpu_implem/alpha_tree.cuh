#pragma once

#include "utils/image.cuh"

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
    thrust::sort_by_key(thrust::seq, weights, weights + (nb_pix_col - 1), sources);

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

inline __device__ bool pre_check_cycle(const int* parent, const double* levels, int p, int q)
{
    if (parent[p] == q)
        return true;

    while (true)
    {
        if (p == q)
            return true;

        p = parent[p];

        if (parent[p] == p or levels[p] > levels[q])
            break;
    }

    return p == q;
}

enum MergeDirection
{
    RIGHT = 0,
    DOWN
};

template <int BlockHeight>
inline __device__ void
merge_alpha_tree(RGBPixel* image, int* parent, double* levels, int height, int width, int first_node,
                 int parent_offset, MergeDirection dir, int blockIdx_y)
{
    int x, y;
    if ((blockIdx_y + 1) * BlockHeight >= height) // Last Line
    {
        x = (first_node - blockIdx_y * BlockHeight * width) / (height - blockIdx_y * BlockHeight);
        y = blockIdx_y * BlockHeight + (first_node - blockIdx_y * BlockHeight * width) % (height - blockIdx_y * BlockHeight);
    }
    else
    {
        x = static_cast<int>(first_node / BlockHeight) % width;
        y = first_node % BlockHeight + static_cast<int>(first_node / (width * BlockHeight)) * BlockHeight;
    }

    int nb_merge, offset;
    if (dir == MergeDirection::RIGHT)
    {
        nb_merge = offset = min(height - y, BlockHeight);
    }
    else // DOWN Merge
    {
        nb_merge = width;
        offset = BlockHeight * width - (BlockHeight - 1);
    }

    int rl = find(parent, first_node);
    int rr = find(parent, first_node + offset);

    // Merge root node
    merge(parent, levels, rl, rr);

    // Iterate on border edges
    for (int i = 0; i < nb_merge; ++i)
    {
        int p, q, n;
        double dist;
        if (dir == MergeDirection::RIGHT)
        {
            p = i + first_node;
            q = i + first_node + offset;
            dist = l2_dist(image[x + (y + i) * width], image[(x + 1) + (y + i) * width]);
            n = parent_offset + BlockHeight + i;
        }
        else // DOWN Merge
        {
            p = BlockHeight * i + first_node;
            q = BlockHeight * i + first_node + BlockHeight * (width - i) + min(height - (y + 1), BlockHeight) * i;
            dist = l2_dist(image[(x + i) + y * width], image[(x + i) + (y + 1) * width]);
            n = 2 * BlockHeight * i + parent_offset + BlockHeight - 1;
        }

        int c1 = find_intersection(parent, levels, p, dist);
        int c2 = find_intersection(parent, levels, q, dist);

        int p1 = parent[c1];
        int p2 = parent[c2];

        parent[c1] = n;
        parent[c2] = n;
        levels[n] = dist;

        if (levels[p1] > levels[p2])
        {
            int tmp = p1;
            p1 = p2;
            p2 = tmp;
        }

        merge(parent, levels, n, p1);

        while (p1 != p2)
        {
            if (levels[p1] == levels[p2] && !pre_check_cycle(parent, levels, p1, p2))
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
                if (!pre_check_cycle(parent, levels, p2, n))
                    parent[n] = p2;

                int tmp = p1;
                p1 = p2;
                p2 = tmp;
            }
        }
    }
}

template <int BlockHeight>
inline __global__ void merge_alpha_tree_cols(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    int nb_pix_col = min(height - y, BlockHeight);

    // Merge in blocks

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % (2 * stride) == 0 && (threadIdx.x + stride) < blockDim.x && (x + stride) < width && y < height)
        {
            int leaf_offset = nb_pix_col * (stride - 1) + BlockHeight * width * blockIdx.y + nb_pix_col * x;
            int parent_offset = width * height + 2 * BlockHeight * ((x + (stride - 1)) + blockIdx.y * width);

            merge_alpha_tree<BlockHeight>(image, parent, levels, height, width, leaf_offset, parent_offset, MergeDirection::RIGHT, blockIdx.y);
        }

        __syncthreads();
    }
}

template <int BlockHeight>
inline __global__ void merge_alpha_tree_blocks_h(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    int y = BlockHeight * blockIdx.y;

    int nb_pix_col = min(height - y, BlockHeight);

    // Merge between blocks horizontally
    // The first thread of first block of each line make the merge of all line

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int stride = 1; stride < gridDim.x; stride += 1)
        {
            int column_stride = blockDim.x * stride;

            int leaf_offset = nb_pix_col * (column_stride - 1) + BlockHeight * width * blockIdx.y;
            int parent_offset = width * height + 2 * BlockHeight * ((column_stride - 1) + blockIdx.y * width);

            merge_alpha_tree<BlockHeight>(image, parent, levels, height, width, leaf_offset, parent_offset, MergeDirection::RIGHT, blockIdx.y);
        }
    }
}

template <int BlockHeight>
inline __global__ void merge_alpha_tree_blocks_v(RGBPixel* image, int* parent, double* levels, int height, int width)
{
    // Merge between blocks vertically
    // The first thread make the merge of all lines

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        for (int stride = 1; stride < gridDim.y; stride += 1)
        {
            int leaf_offset = (stride - 1) * BlockHeight * width + (BlockHeight - 1);
            int parent_offset = width * height + (stride - 1) * 2 * BlockHeight * width;

            merge_alpha_tree<BlockHeight>(image, parent, levels, height, width, leaf_offset, parent_offset, MergeDirection::DOWN, blockIdx.y);
        }
    }
}