#include "cc_labelling.cuh"
#include "graph_creation.cuh"
#include "utils/cuda_error.cuh"
#include "utils/image.cuh"

#include <algorithm>

constexpr int connectivity = 4;

void debug_display(int nb_site, int* labels, int* residual_list, bool verbose = false)
{
    if (verbose)
    {
        for (int i = 0; i < nb_site; ++i)
        {
            std::cout << "SITE: " << i << ", LABEL: " << labels[i] << ", RESIDUAL: ";
            for (uint j = i * connectivity; j - i * connectivity < connectivity; ++j)
            {
                std::cout << residual_list[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<int> unique;
    for (int i = 0; i < nb_site; ++i)
        unique.push_back(labels[i]);
    auto ip = std::unique(unique.begin(), unique.end());
    unique.resize(std::distance(unique.begin(), ip));

    std::cout << "Number of flat zone / Number of pixels: " << (double)unique.size() << " / " << (double)nb_site
              << std::endl;
}

void cc_labeling_gpu(const std::shared_ptr<utils::RGBImage>& image)
{
    int nb_site = image->height * image->width;

    // Memory allocation

    int* m_nn_list;
    cudaMallocManaged(&m_nn_list, connectivity * nb_site * sizeof(int));
    checkCudaError();
    cudaMemset(m_nn_list, -1, connectivity * nb_site * sizeof(int));
    checkCudaError();

    int* m_labels;
    cudaMallocManaged(&m_labels, nb_site * sizeof(int));
    checkCudaError();

    int* m_residual_list;
    cudaMallocManaged(&m_residual_list, connectivity * nb_site * sizeof(int));
    checkCudaError();
    cudaMemset(m_residual_list, -1, connectivity * nb_site * sizeof(int));
    checkCudaError();

    // Kernel setup

    int BlockHeight = 32;
    int w = std::ceil((float)image->width / BlockHeight);
    int h = std::ceil((float)image->height / BlockHeight);

    dim3 dimBlock(BlockHeight, BlockHeight);
    dim3 dimGrid(w, h);

    // Kernel launch

    // Graph creation
    {
        create_graph_4<<<dimGrid, dimBlock>>>(image->pixels, m_nn_list, image->height, image->width);
        checkCudaError();
        cudaDeviceSynchronize();
        checkCudaError();
    }

    // Flat zone labeling
    {
        initialization_step<<<dimGrid, dimBlock>>>(m_nn_list, m_residual_list, m_labels, image->height, image->width);
        checkCudaError();
        cudaDeviceSynchronize();
        checkCudaError();

        analysis_step<<<dimGrid, dimBlock>>>(m_labels, image->height, image->width);
        checkCudaError();
        cudaDeviceSynchronize();
        checkCudaError();

        reduction_step<<<dimGrid, dimBlock>>>(m_residual_list, m_labels, image->height, image->width);
        checkCudaError();
        cudaDeviceSynchronize();
        checkCudaError();

        analysis_step<<<dimGrid, dimBlock>>>(m_labels, image->height, image->width);
        checkCudaError();
        cudaDeviceSynchronize();
        checkCudaError();
    }

    // Image reconstruction

    for (int j = 0; j < image->height; ++j)
    {
        for (int i = 0; i < image->width; ++i)
        {
            int site = i + j * image->width;
            image->pixels[site] = image->pixels[m_labels[site]];
        }
    }

    // Sanity checks

    image->save("flatzone_labelling.png");

    debug_display(nb_site, m_labels, m_residual_list);

    // Free memory

    cudaFree(m_nn_list);
    cudaFree(m_labels);
    cudaFree(m_residual_list);
}