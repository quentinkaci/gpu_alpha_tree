#include "alpha_tree.cuh"
#include "cc_labelling.cuh"
#include "graph_creation.cuh"
#include "utils/cuda_error.cuh"

#include "utils/image.cuh"

#include <algorithm>
#include <cmath>

#include <fstream>

using namespace utils;

constexpr int BlockHeight = 6;

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

void alpha_tree_gpu(const std::shared_ptr<utils::RGBImage>& image)
{
    int height = image->height;
    int width = image->width;

    int new_height = (height + 2 * BlockHeight * (int)std::ceil((float)height / BlockHeight));

    int nb_nodes = width * new_height;

    RGBPixel* m_image = image->pixels;

    int* m_parent;
    cudaMallocManaged(&m_parent, nb_nodes * sizeof(int));
    checkCudaError();

    double* m_levels;
    cudaMallocManaged(&m_levels, nb_nodes * sizeof(double));
    checkCudaError();
    cudaMemset(m_levels, 0, nb_nodes * sizeof(double));
    checkCudaError();

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

    //    std::cout << std::endl
    //              << "Build an alpha tree per column:" << std::endl;

    build_alpha_tree_col<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    //    for (int i = 0; i < nb_nodes; ++i)
    //        std::cout << "Node: " << i << ", Parent: " << m_parent[i] << ", Level: " << m_levels[i] << std::endl;
    //
    //    save_alpha_tree_dot("before_merge.dot", m_parent, m_levels, nb_nodes);
    //
    //    std::cout << std::endl
    //              << "Merge alpha tree per column:" << std::endl;

    merge_alpha_tree_cols<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    merge_alpha_tree_blocks_h<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    merge_alpha_tree_blocks_v<BlockHeight><<<dimGrid, dimBlock>>>(m_image, m_parent, m_levels, height, width);
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();

    //    for (int i = 0; i < nb_nodes; ++i)
    //        std::cout << "Node: " << i << ", Parent: " << m_parent[i] << ", Level: " << m_levels[i] << std::endl;

    //    save_alpha_tree_dot("after_merge.dot", m_parent, m_levels, nb_nodes);

    cudaFree(m_parent);
    cudaFree(m_levels);
    checkCudaError();
}
