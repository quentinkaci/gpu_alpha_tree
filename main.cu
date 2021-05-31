#include "alpha_tree.cuh"
#include "cc_labelling.cuh"
#include "cuda_error.cuh"
#include "graph_creation.cuh"

#include "image.hh"

#include <algorithm>
#include <cmath>

using namespace utils;

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

    std::cout << "Number of flatzone / Number of pixels: " << (double)unique.size() << " / " << (double)nb_site << std::endl;
}

int main()
{
    // // Image loading

    //    auto image = RGBImage::load("../batiment.png");

    //    int nb_site = image->height * image->width;
    //
    //    // Memory allocation
    //
    //    cudaError_t rc = cudaSuccess;
    //
    //    int* m_nn_list;
    //    rc = cudaMallocManaged(&m_nn_list, connectivity * nb_site * sizeof(int));
    //    if (rc)
    //        abortError("Fail M_NN_LIST allocation");
    //    rc = cudaMemset(m_nn_list, -1, connectivity * nb_site * sizeof(int));
    //    if (rc)
    //        abortError("Fail M_NN_LIST memset");
    //
    //    int* m_labels;
    //    rc = cudaMallocManaged(&m_labels, nb_site * sizeof(int));
    //    if (rc)
    //        abortError("Fail M_LABELS allocation");
    //
    //    int* m_residual_list;
    //    rc = cudaMallocManaged(&m_residual_list, connectivity * nb_site * sizeof(int));
    //    if (rc)
    //        abortError("Fail M_RESIDUAL_LIST allocation");
    //    rc = cudaMemset(m_residual_list, -1, connectivity * nb_site * sizeof(int));
    //    if (rc)
    //        abortError("Fail M_RESIDUAL_LIST memset");
    //
    //    // Kernel setup
    //
    //    int bsize = 32;
    //    int w = std::ceil((float)image->width / bsize);
    //    int h = std::ceil((float)image->height / bsize);
    //
    //    dim3 dimBlock(bsize, bsize);
    //    dim3 dimGrid(w, h);
    //
    //    // Kernel launch
    //
    //    // Graph creation
    //    {
    //        create_graph_4<<<dimGrid, dimBlock>>>(image->pixels, m_nn_list, image->height, image->width);
    //        cudaDeviceSynchronize();
    //    }
    //
    //    // Flat zone labelization
    //    {
    //        initialization_step<<<dimGrid, dimBlock>>>(m_nn_list, m_residual_list, m_labels, image->height, image->width);
    //        cudaDeviceSynchronize();
    //
    //        analysis_step<<<dimGrid, dimBlock>>>(m_labels, image->height, image->width);
    //        cudaDeviceSynchronize();
    //
    //        reduction_step<<<dimGrid, dimBlock>>>(m_residual_list, m_labels, image->height, image->width);
    //        cudaDeviceSynchronize();
    //
    //        analysis_step<<<dimGrid, dimBlock>>>(m_labels, image->height, image->width);
    //        cudaDeviceSynchronize();
    //    }
    //
    //    if (cudaPeekAtLastError())
    //        abortError("Computation Error");
    //
    //    // Image reconstruction
    //
    //    for (int j = 0; j < image->height; ++j)
    //    {
    //        for (int i = 0; i < image->width; ++i)
    //        {
    //            int site = i + j * image->width;
    //            image->pixels[site] = image->pixels[m_labels[site]];
    //        }
    //    }
    //
    //    // Validity checks
    //
    //    image->save("flatzone_labelling.png");
    //
    //    debug_display(nb_site, m_labels, m_residual_list);
    //
    //    // Free memory
    //
    //    cudaFree(m_nn_list);
    //    cudaFree(m_labels);
    //    cudaFree(m_residual_list);

    RGBPixel image[6] = {
        {100, 100, 100}, {100, 100, 150}, {20, 45, 79}, {100, 100, 100}, {100, 100, 150}, {20, 45, 79}};
    int height = 6;
    int width = 1;

    cudaError_t rc = cudaSuccess;

    RGBPixel* m_image;
    rc = cudaMallocManaged(&m_image, sizeof(image));
    if (rc)
        abortError("Fail M_IMAGE allocation");
    rc = cudaMemcpy(m_image, image, sizeof(image), cudaMemcpyHostToDevice);
    if (rc)
        abortError("Fail M_IMAGE memcpy");

    int* m_parent;
    rc = cudaMallocManaged(&m_parent, (2 * height * width - 1) * sizeof(int));
    if (rc)
        abortError("Fail M_IMAGE allocation");

    double* m_levels;
    rc = cudaMallocManaged(&m_levels, (2 * height * width - 1) * sizeof(double));
    if (rc)
        abortError("Fail M_LEVELS allocation");

    int bsize = 32;
    int w = std::ceil((float)(width + bsize) / bsize);
    int h = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    init_parent<<<dimGrid, dimBlock>>>(m_parent, height, width);

    cudaDeviceSynchronize();

    for (int i = 0; i < 2 * height * width - 1; ++i)
        std::cout << m_parent[i] << ", ";
    std::cout << std::endl;

    //    w = std::ceil((float)width / bsize);
    //    h = std::ceil((float)height / bsize);
    //
    //    dimBlock = dim3(bsize, bsize);
    //    dimGrid = dim3(w, h);
    //
    //    build_alpha_tree_col<32><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    //
    //    cudaDeviceSynchronize();
    //
    //    for (int i = 0; i < 2 * height * width - 1; ++i)
    //        std::cout << m_parent[i] << ", ";
    //    std::cout << std::endl;

    cudaFree(m_image);
    cudaFree(m_parent);
    cudaFree(m_levels);

    return 0;
}