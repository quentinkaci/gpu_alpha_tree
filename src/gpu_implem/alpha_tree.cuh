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

template <int BlockHeight>
inline __device__ void merge_left_right(RGBPixel* image, int* parent, double* levels, int height, int width, int x, int y, int left_offset, int parent_offset)
{
    int nb_pix_col = min(height - y, BlockHeight);

    // Merge with column on the right
    int rl = find(parent, left_offset);
    int rr = find(parent, left_offset + BlockHeight);

    // Merge root node
    merge(parent, levels, rl, rr);

    // Iterate on border edges
    for (int i = 0; i < nb_pix_col; ++i)
    {
        // Merge with column on the right
        int p = i + left_offset;
        int q = i + left_offset + BlockHeight;
        double dist = l2_dist(image[x + (y + i) * width], image[(x + 1) + (y + i) * width]);

        int c1 = find_intersection(parent, levels, p, dist);
        int c2 = find_intersection(parent, levels, q, dist);

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
inline __global__ void merge_alpha_tree_cols(RGBPixel* image, int* parent, double* levels, int height, int width, volatile bool* block_mask)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = BlockHeight * blockIdx.y;

    int nb_pix_col = min(height - y, BlockHeight);

    // Merge in blocks

    for (int stride = 1; (threadIdx.x + stride) < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % (2 * stride) == 0 && y < height)
        {
            int leaf_offset = BlockHeight * (stride - 1);
            if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
                leaf_offset += BlockHeight * width * blockIdx.y + nb_pix_col * x;
            else
                leaf_offset += BlockHeight * (x + blockIdx.y * width);
            int parent_offset = width * height + 2 * BlockHeight * ((x + (stride - 1)) + blockIdx.y * width);

            merge_left_right<BlockHeight>(image, parent, levels, height, width, x, y, leaf_offset, parent_offset);
        }

        __syncthreads();
    }

    // Merge between blocks

    //    if (threadIdx.x == 0)
    //    {
    //        for (int stride = 1; blockIdx.x + stride < gridDim.x; stride *= 2)
    //        {
    //            // Reset all values on the same line
    //
    //            block_mask[blockIdx.x + gridDim.x * blockIdx.y] = false;
    //
    //            bool line_has_finished = false;
    //            while (!line_has_finished)
    //            {
    //                line_has_finished = true;
    //                for (int i = 0; i < gridDim.x && line_has_finished; ++i)
    //                {
    //                    if (i + stride < gridDim.x)
    //                        line_has_finished &= !block_mask[i + gridDim.x * blockIdx.y];
    //                }
    //            }
    //
    //            if (blockIdx.x % (2 * stride) == 0 && y < height)
    //            {
    //                int column_stride = blockDim.x * stride;
    //
    //                int leaf_offset = BlockHeight * (column_stride - 1);
    //                if ((blockIdx.y + 1) * BlockHeight >= height) // Last line
    //                    leaf_offset += BlockHeight * width * blockIdx.y + nb_pix_col * x;
    //                else
    //                    leaf_offset += BlockHeight * (x + blockIdx.y * width);
    //                int parent_offset = width * height + 2 * BlockHeight * ((x + (column_stride - 1)) + blockIdx.y * width);
    //
    //                merge_left_right<BlockHeight>(image, parent, levels, height, width, x, y, leaf_offset, parent_offset);
    //            }
    //
    //            // Current block has finished
    //            block_mask[blockIdx.x + gridDim.x * blockIdx.y] = true;
    //
    //            // Waiting for all blocks in the same line
    //            line_has_finished = false;
    //            while (!line_has_finished)
    //            {
    //                line_has_finished = true;
    //                for (int i = 0; i < gridDim.x && line_has_finished; ++i)
    //                {
    //                    if (i + stride < gridDim.x)
    //                        line_has_finished &= block_mask[i + gridDim.x * blockIdx.y];
    //                }
    //            }
    //        }
    //    }
}
