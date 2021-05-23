#include "cc_labelling.cuh"
#include "image.hh"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using namespace utils;

// 4 connectivity
constexpr int connectivity = 4;

[[gnu::noinline]] void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::cerr << "Error: " << err << std::endl;
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

float gradient(RGBPixel src, RGBPixel dst)
{
    return sqrt(pow(dst.r - src.r, 2) + pow(dst.g - src.g, 2) + pow(dst.b - src.b, 2));
}

void add_neighbour(std::vector<int>& nn_list, int site, int nn)
{
    int i = 0;
    while (nn_list[connectivity * site + i] != -1)
        ++i;
    nn_list[connectivity * site + i] = nn;
}

std::vector<int> create_graph_4(std::shared_ptr<RGBImage> image)
{
    std::vector<int> nn_list(connectivity * image->width * image->height, -1);

    for (int j = 0; j < image->height; ++j)
    {
        for (int i = 0; i < image->width; ++i)
        {
            int src_pos = i + j * image->width;

            if (j != image->height - 1)
            {
                int dst_pos = i + (j + 1) * image->width;
                if (gradient(image->pixels[src_pos], image->pixels[dst_pos]) == 0.f)
                {
                    add_neighbour(nn_list, src_pos, dst_pos);
                    add_neighbour(nn_list, dst_pos, src_pos);
                }
            }

            if (i != image->width - 1)
            {
                int dst_pos = (i + 1) + j * image->width;
                if (gradient(image->pixels[src_pos], image->pixels[dst_pos]) == 0.f)
                {
                    add_neighbour(nn_list, src_pos, dst_pos);
                    add_neighbour(nn_list, dst_pos, src_pos);
                }
            }
        }
    }

    return nn_list;
}

int main()
{
    // clang-format off
    //    int nn_list[] = {
    //        3, 10, -1, -1,
    //        4, 5, 15, -1,
    //        17, 18, -1, -1,
    //        0, 8, 9, 21,
    //        1, 11, -1, -1,
    //        1, 13, 20, -1,
    //        8, 9, -1, -1,
    //        20, -1, -1, -1,
    //        3, 6, -1, -1,
    //        3, 6, 10, -1,
    //        0, 9, -1, -1,
    //        4, 20, -1, -1,
    //        16, 19, 21, -1,
    //        5, 15, -1, -1,
    //        19, 21, -1, -1,
    //        1, 13, -1, -1,
    //        12, 21, -1, -1,
    //        2, 18, -1, -1,
    //        2, 17, -1, -1,
    //        12, 14, -1, -1,
    //        5, 7, 11, -1,
    //        3, 12, 14, 16
    //    };
    // clang-format on

    auto image = RGBImage::load("../batiment.png");

    auto nn_list_vect = create_graph_4(image);
    int nb_site = nn_list_vect.size() / connectivity;
    auto nn_list = nn_list_vect.data();

    cudaError_t rc = cudaSuccess;

    int* m_nn_list;
    rc = cudaMallocManaged(&m_nn_list, connectivity * nb_site * sizeof(int));
    if (rc)
        abortError("Fail M_NN_LIST allocation");

    cudaMemcpy(m_nn_list, nn_list, connectivity * nb_site * sizeof(int), cudaMemcpyHostToHost);

    int* m_labels;
    rc = cudaMallocManaged(&m_labels, nb_site * sizeof(int));
    if (rc)
        abortError("Fail M_LABELS allocation");

    int* m_residual_list;
    rc = cudaMallocManaged(&m_residual_list, connectivity * nb_site * sizeof(int));
    if (rc)
        abortError("Fail M_RESIDUAL_LIST allocation");
    cudaMemset(m_residual_list, -1, connectivity * nb_site * sizeof(int));

    for (int i = 0; i < nb_site; ++i)
        initialization_step(m_nn_list, connectivity, m_residual_list, m_labels, i);

    for (int i = 0; i < nb_site; ++i)
        anylisis_step(m_labels, i);

    for (int i = 0; i < nb_site; ++i)
    {
        reduction_step<<<1, 1>>>(m_residual_list, connectivity, m_labels, i);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < nb_site; ++i)
        anylisis_step(m_labels, i);

    for (int j = 0; j < image->height; ++j)
    {
        for (int i = 0; i < image->width; ++i)
        {
            int site = i + j * image->width;
            image->pixels[site] = image->pixels[m_labels[site]];
        }
    }

    //    for (int j = 0; j < 50; ++j)
    //    {
    //        for (int i = 0; i < 50; ++i)
    //        {
    //            int site = i + j * image->width;
    //            std::cout << m_labels[site] << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    std::vector<int> unique;
    for (int i = 0; i < nb_site; ++i)
        unique.push_back(m_labels[i]);
    auto ip = std::unique(unique.begin(), unique.end());
    unique.resize(std::distance(unique.begin(), ip));

    std::cout
        << (double)unique.size() / (double)nb_site << std::endl;

    image->save("flatzone_labelling.png");

    //    for (int i = 0; i < nb_site; ++i)
    //    {
    //        std::cout << "SITE: " << i << ", LABEL: " << m_labels[i] << ", RESIDUAL: ";
    //        for (uint j = i * connectivity; j - i * connectivity < connectivity; ++j)
    //        {
    //            std::cout << m_residual_list[j] << " ";
    //        }
    //        std::cout << std::endl;
    //    }

    return 0;
}