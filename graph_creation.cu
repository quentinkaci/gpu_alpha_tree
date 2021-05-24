#include "graph_creation.cuh"

constexpr int connectivity = 4;

__device__ float gradient(RGBPixel src, RGBPixel dst)
{
    return sqrt(pow(dst.r - src.r, 2) + pow(dst.g - src.g, 2) + pow(dst.b - src.b, 2));
}

__global__ void create_graph_4(RGBPixel* pixels, int* nn_list, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int src = x + y * width;

    if (x - 1 >= 0) // Left neighbour
    {
        int dst = (x - 1) + y * width;
        if (gradient(pixels[src], pixels[dst]) == 0.f)
            nn_list[src * connectivity + 0] = dst;
    }

    if (x + 1 < width) // Right neighbour
    {
        int dst = (x + 1) + y * width;
        if (gradient(pixels[src], pixels[dst]) == 0.f)
            nn_list[src * connectivity + 1] = dst;
    }

    if (y - 1 >= 0) // Up neighbour
    {
        int dst = x + (y - 1) * width;
        if (gradient(pixels[src], pixels[dst]) == 0.f)
            nn_list[src * connectivity + 2] = dst;
    }

    if (y + 1 < height) // Down neighbour
    {
        int dst = x + (y + 1) * width;
        if (gradient(pixels[src], pixels[dst]) == 0.f)
            nn_list[src * connectivity + 3] = dst;
    }
}