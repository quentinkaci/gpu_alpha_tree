#include "alpha_tree_cpu.hh"

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace utils;

static float gradient(const RGBPixel src, const RGBPixel dst)
{
    return sqrt(pow(dst.r - src.r, 2) + pow(dst.g - src.g, 2) + pow(dst.b - src.b, 2));
}

static std::vector<GraphEdge> create_graph(const std::shared_ptr<RGBImage>& image)
{
    std::vector<GraphEdge> edges;

    for (uint y = 0; y < image->height; y++)
    {
        for (uint x = 0; x < image->width; x++)
        {
            const RGBPixel src = image->pixels[x + y * image->width];

            if (y != image->height - 1)
            {
                const RGBPixel dst = image->pixels[x + (y + 1) * image->width];
                const float w = gradient(src, dst);
                edges.emplace_back(x + y * image->width, x + (y + 1) * image->width, w);
            }

            if (x != image->width - 1)
            {
                const RGBPixel dst = image->pixels[(x + 1) + y * image->width];
                const float w = gradient(src, dst);
                edges.emplace_back(x + y * image->width, (x + 1) + y * image->width, w);
            }
        }
    }

    return edges;
}

static uint find(const std::vector<uint>& par, const uint val)
{
    uint p = val;
    uint q = par[val];

    while (p != q)
    {
        p = q;
        q = par[p];
    }
    
    const uint r = p;
    return r;
}

static std::vector<uint> compute_flatzones(const std::shared_ptr<RGBImage>& image,
                                           const std::vector<GraphEdge>& edges)
{
    std::vector<uint> zpar(image->width * image->height);
    std::iota(zpar.begin(), zpar.end(), 0);

    uint i = 0;
    while (i < edges.size() && edges[i].weight == 0)
    {
        const uint p = edges[i].src;
        const uint q = edges[i].dst;

        const uint rp = find(zpar, p);
        const uint rq = find(zpar, q);

        if (rp != rq)
            zpar[rq] = rp;

        i++;
    }

    return zpar;
}

static std::vector<int> create_node_map(const std::vector<uint>& zpar, uint& flatzones_count)
{
    flatzones_count = 0;
    
    std::vector<int> node_map(zpar.size(), -1);

    for (uint p = 0; p < zpar.size(); p++)
    {
        const uint rp = find(zpar, p);

        if (node_map[rp] < 0)
        {
            node_map[rp] = flatzones_count;
            flatzones_count++;
        }

        node_map[p] = node_map[rp];
    }

    return node_map;
}

static AlphaTree compute_hierarchy(const std::vector<GraphEdge>& edges,
                                   const std::vector<int>& node_map,
                                   uint node_count)
{
    std::vector<uint> par(2 * node_count - 1);
    std::iota(par.begin(), par.end(), 0);

    std::vector<uint> levels(2 * node_count - 1, 0);

    for (uint i = 0; i < edges.size() && edges[i].weight != 0; i++)
    {
        const uint p = edges[i].src;
        const uint q = edges[i].dst;
        const float weight = edges[i].weight;

        uint rp = find(par, node_map[p]);
        uint rq = find(par, node_map[q]);

        if (rp != rq)
        {
            const uint new_root_id = node_count;
            node_count++;

            levels[new_root_id] = weight;
            par[rp] = new_root_id;
            par[rq] = new_root_id;
        }
    }

    return AlphaTree{par, levels};
}

void alpha_tree_cpu(const std::shared_ptr<RGBImage>& image)
{
    // 1. Create 4 connectivity graph
    std::vector<GraphEdge> edges = create_graph(image);
    
    // 2. Sort edges per weight
    std::sort(edges.begin(), edges.end());

    // 3. Compute flat zones
    uint flatzones_count = 0;
    auto zpar = compute_flatzones(image, edges);
    auto node_map = create_node_map(zpar, flatzones_count);

    // 4. Create α-tree
    auto res = compute_hierarchy(edges, node_map, flatzones_count);
}