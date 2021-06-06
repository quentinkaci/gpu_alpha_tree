#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "src/gpu_implem/alpha_tree.cuh"
#include "utils/cuda_error.cuh"
#include "utils/image.cuh"

#include <algorithm>

using namespace utils;

constexpr int BlockHeight = 6;

void assert_alpha_tree_eq(RGBPixel* image, int height, int width, const int* expected_parent, const double* expected_levels, dim3 mergeDimGrid, dim3 mergeDimBlock)
{
    int new_height = (height + 2 * BlockHeight * (int)std::ceil((float)height / BlockHeight));

    int nb_nodes = width * new_height;

    cudaError_t rc = cudaSuccess;

    RGBPixel* m_image;
    rc = cudaMallocManaged(&m_image, height * width * sizeof(RGBPixel));
    if (rc)
        abortError("Fail M_IMAGE allocation");
    rc = cudaMemcpy(m_image, image, height * width * sizeof(RGBPixel), cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail M_IMAGE memcpy");

    int* m_parent;
    rc = cudaMallocManaged(&m_parent, nb_nodes * sizeof(int));
    if (rc)
        abortError("Fail M_IMAGE allocation");

    double* m_levels;
    rc = cudaMallocManaged(&m_levels, nb_nodes * sizeof(double));
    if (rc)
        abortError("Fail M_LEVELS allocation");
    rc = cudaMemset(m_levels, 0, nb_nodes * sizeof(double));
    if (rc)
        abortError("Fail M_LEVELS memset");

    int w = std::ceil((float)width / BlockHeight);
    int h = std::ceil((float)new_height / BlockHeight);

    dim3 dimBlock(BlockHeight, BlockHeight);
    dim3 dimGrid(w, h);

    init_parent<<<dimGrid, dimBlock>>>(m_parent, new_height, width);
    cudaDeviceSynchronize();

    w = std::ceil((float)width / BlockHeight);
    h = std::ceil((float)height / BlockHeight);

    dimBlock = dim3(BlockHeight, 1);
    dimGrid = dim3(w, h);

    build_alpha_tree_col<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    cudaDeviceSynchronize();

    merge_alpha_tree_col<BlockHeight><<<mergeDimGrid, mergeDimBlock>>>(m_image, m_parent, m_levels, height, width);
    cudaDeviceSynchronize();

    for (int i = 0; i < nb_nodes; ++i)
    {
        ASSERT_EQ(m_parent[i], expected_parent[i]) << i;
        ASSERT_NEAR(m_levels[i], expected_levels[i], 0.01) << i;
    }

    cudaFree(m_image);
    cudaFree(m_parent);
    cudaFree(m_levels);
}

TEST(test_basic_merge, MergeTwoColumns)
{
    // clang-format off
    RGBPixel image[] = {
        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

    };

    int expected_parent[] = {12, 19, 14, 13, 22, 23, 18, 19, 24, 26, 22, 25, 18, 14, 20, 16, 16, 17, 20, 12, 21, 23, 13, 16, 12, 13, 18, 15, 15, 29, 30, 31, 32, 33, 34, 35, };
    double expected_levels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 99.3277, 120.275, 120.275, 0, 99.3277, 0, 99.3277, 99.3277, 0, 99.3277, 50, 50, 99.3277, 120.275, 120.275, 0, 0, 0, 0, 0, 0, 0, };
    // clang-format on

    assert_alpha_tree_eq(image, 6, 2, expected_parent, expected_levels, 1, 1);
}

TEST(test_basic_merge, MergeTwoColumnsOnTwoLines)
{
    // clang-format off
    RGBPixel image[] = {
        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

        //

        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 150},
        {100, 100, 150},

        {20, 45, 79},
        {100, 100, 100},

    };

    int expected_parent[] = {24, 31, 26, 25, 34, 35, 30, 31, 36, 38, 34, 37, 48, 55, 50, 49, 58, 59, 54, 55, 60, 62, 58, 61, 30, 26, 32, 28, 28, 29, 32, 24, 33, 35, 25, 28, 24, 25, 30, 27, 27, 41, 42, 43, 44, 45, 46, 47, 54, 50, 56, 52, 52, 53, 56, 48, 57, 59, 49, 52, 48, 49, 54, 51, 51, 65, 66, 67, 68, 69, 70, 71, };
    double expected_levels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 99.3277, 120.275, 120.275, 0, 99.3277, 0, 99.3277, 99.3277, 0, 99.3277, 50, 50, 99.3277, 120.275, 120.275, 0, 0, 0, 0, 0, 0, 0, 50, 50, 99.3277, 120.275, 120.275, 0, 99.3277, 0, 99.3277, 99.3277, 0, 99.3277, 50, 50, 99.3277, 120.275, 120.275, 0, 0, 0, 0, 0, 0, 0, };
    // clang-format on

    assert_alpha_tree_eq(image, 12, 2, expected_parent, expected_levels, dim3(1, 2, 1), 1);
}

TEST(test_basic_merge, LeftColumnFlatZone)
{
    // clang-format off
    RGBPixel image[] = {
        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 100},

        {100, 100, 100},
        {20, 45, 79},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 100},
    };

    int expected_parent[] = {12, 12, 13, 14, 15, 16, 18, 24, 20, 26, 25, 23, 13, 14, 15, 16, 20, 17, 21, 22, 23, 28, 18, 19, 19, 19, 18, 28, 28, 29, 30, 31, 32, 33, 34, 35, };
    double expected_levels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99.3277, 50, 0, 99.3277, 50, 0, 50, 50, 99.3277, 120.275, 120.275, 0, 0, 0, 0, 0, 0, 0, };
    // clang-format on

    assert_alpha_tree_eq(image, 6, 2, expected_parent, expected_levels, 1, 1);
}

TEST(test_basic_merge, BothColumnFlatZone)
{
    // clang-format off
    RGBPixel image[] = {
        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 150},

        {100, 100, 100},
        {100, 100, 150},
    };

    int expected_parent[] = {12, 12, 13, 14, 15, 16, 24, 24, 25, 26, 27, 28, 13, 14, 15, 16, 18, 17, 19, 20, 21, 22, 23, 23, 25, 26, 27, 28, 16, 29, 30, 31, 32, 33, 34, 35, };
    double expected_levels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
    // clang-format on

    assert_alpha_tree_eq(image, 6, 2, expected_parent, expected_levels, 1, 1);
}
