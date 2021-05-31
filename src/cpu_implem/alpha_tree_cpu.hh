#pragma once

#include "utils/image.cuh"

struct GraphEdge
{
    uint src;
    uint dst;
    float weight;

    bool operator<(const GraphEdge& rhs) const
    {
        return weight < rhs.weight;
    }
};

struct AlphaTree
{
    std::vector<uint> par;
    std::vector<uint> levels;
};

void alpha_tree_cpu(const std::shared_ptr<utils::RGBImage>& image);