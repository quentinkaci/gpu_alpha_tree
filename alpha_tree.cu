#include "alpha_tree.cuh"

//__global__ void merge_alpha_tree_col(int origin_width, int previous_block_width)
//{
//    const int nb_succ_merges = static_cast<int>(ceilf(log2(static_cast<float>(origin_width / previous_block_width))));
//    const int merging_width = 1 << static_cast<int>(log2(static_cast<float>(origin_width)));
//    const int power_indice = static_cast<int>(log2(static_cast<float>(previous_block_width)));
//    unsigned int merging_indice;
//
//    for (int i = 0; i < nb_succ_merges; ++i)
//    {
//        merging_indice = (((1 << (i + power_indice)) - 1) + (threadIdx.x * (1 << (i + power_indice + 1))));
//        if (merging_indice < merging_width)
//            merge_columns<int16_t>(d_image, merging_indice, merging_indice + 1, parent, origin_height, origin_height, image_cols_pitched, parent_cols_pitched);
//        else
//            return; // This thread is done, return to help the the warp scheduler
//
//        __syncthreads();
//    }
//}

//template <unsigned int BlockHeight, unsigned int BlockWidth>
//__device__ void build_alpha_tree()
//{
//}