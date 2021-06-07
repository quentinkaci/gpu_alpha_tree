#include "alpha_tree.cuh"
#include "cc_labelling.cuh"
#include "graph_creation.cuh"
#include "utils/cuda_error.cuh"

#include "utils/image.cuh"

#include <algorithm>
#include <cmath>

#include <fstream>

using namespace utils;

constexpr int connectivity = 4;
constexpr int BlockHeight = 6;

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

    std::cout << "Number of flatzone / Number of pixels: " << (double)unique.size() << " / " << (double)nb_site
              << std::endl;
}

void dfs_dot(std::ofstream& out, const std::vector<std::vector<int>>& children, const double* levels, int node)
{
    out << node << "-> {";

    for (std::size_t i = 0; i < children[node].size(); ++i)
    {
        out << children[node][i];
        if (i != children[node].size() - 1)
            out << ", ";
    }

    out << "}" << std::endl;
    out << node << " [label=\"" << node << " [" << levels[node] << "]\"]" << std::endl;

    for (const auto& child : children[node])
        dfs_dot(out, children, levels, child);
}

void save_alpha_tree_dot(const std::string& filename, const int* parent, const double* levels, int nb_nodes)
{
    std::ofstream file(filename);

    std::vector<int> roots;
    std::vector<std::vector<int>> children(nb_nodes, std::vector<int>());
    for (int i = 0; i < nb_nodes; ++i)
    {
        int p = parent[i];
        if (p == i)
            roots.push_back(i);
        else
            children[p].push_back(i);
    }

    file << "digraph D {" << std::endl;

    for (const auto& root : roots)
        dfs_dot(file, children, levels, root);

    file << "}" << std::endl;
}

void alpha_tree_gpu(const std::shared_ptr<utils::RGBImage>&)
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
    //    int BlockHeight = 32;
    //    int w = std::ceil((float)image->width / BlockHeight);
    //    int h = std::ceil((float)image->height / BlockHeight);
    //
    //    dim3 dimBlock(BlockHeight, BlockHeight);
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

    RGBPixel image[24] = {
        {100, 100, 100},
        {100, 100, 150},
        {100, 100, 200},
        {100, 200, 200},

        {100, 100, 100},
        {100, 100, 150},
        {100, 100, 200},
        {100, 200, 200},

        {100, 100, 100},
        {250, 100, 150},
        {250, 100, 200},
        {100, 200, 200},

        {100, 100, 100},
        {250, 100, 150},
        {250, 100, 200},
        {100, 200, 200},

        {100, 100, 100},
        {100, 100, 150},
        {100, 100, 200},
        {100, 200, 200},

        {100, 100, 100},
        {100, 100, 150},
        {100, 100, 200},
        {100, 200, 200},
    };
    int height = 6;
    int width = 4;

    int new_height = (height + 2 * BlockHeight * (int)std::ceil((float)height / BlockHeight));

    int nb_nodes = width * new_height;

    cudaError_t rc = cudaSuccess;

    RGBPixel* m_image;
    rc = cudaMallocManaged(&m_image, sizeof(image));
    if (rc)
        abortError("Fail M_IMAGE allocation");
    rc = cudaMemcpy(m_image, image, sizeof(image), cudaMemcpyHostToDevice);
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

    std::cout << std::endl
              << "Build an alpha tree per column:" << std::endl;

    build_alpha_tree_col<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);

    cudaDeviceSynchronize();

    for (int i = 0; i < nb_nodes; ++i)
        std::cout << "Node: " << i << ", Parent: " << m_parent[i] << ", Level: " << m_levels[i] << std::endl;

    save_alpha_tree_dot("before_merge.dot", m_parent, m_levels, nb_nodes);

    std::cout << std::endl
              << "Merge alpha tree per column:" << std::endl;

    // FIXME Modify block and grid dim according to the reduce
    merge_alpha_tree_col<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);

    cudaDeviceSynchronize();

    for (int i = 0; i < nb_nodes; ++i)
        std::cout << "Node: " << i << ", Parent: " << m_parent[i] << ", Level: " << m_levels[i] << std::endl;

    save_alpha_tree_dot("after_merge.dot", m_parent, m_levels, nb_nodes);

    cudaFree(m_image);
    cudaFree(m_parent);
    cudaFree(m_levels);
}
