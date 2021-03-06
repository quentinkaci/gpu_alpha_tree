#include "cc_labelling.cuh"

constexpr int connectivity = 4;

__global__ void initialization_step(const int* nn_list, int* residual_list, int* labels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = x + y * width;

    int label = i;

    for (int j = i * connectivity; (j - i * connectivity) < connectivity; ++j)
    {
        int nn = nn_list[j];
        if (nn == -1)
            continue;

        if (nn < label)
            label = nn;
    }

    int pos = i * connectivity;
    for (int j = i * connectivity; (j - i * connectivity) < connectivity; ++j)
    {
        int nn = nn_list[j];
        if (nn == -1)
            continue;

        if (nn < i && nn != label)
            residual_list[pos++] = nn;
    }

    // Assign label
    labels[i] = label;
}

__global__ void analysis_step(int* labels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = x + y * width;

    int last_label;
    do
    {
        last_label = labels[i];
        labels[i] = labels[labels[i]];
    } while (labels[i] != last_label);
}

__global__ void reduction_step(const int* residual_list, int* labels, int height, int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = x + y * width;

    for (int j = i * connectivity; (j - i * connectivity) < connectivity && residual_list[j] != -1; ++j)
    {
        int label_1 = labels[i];
        while (label_1 != labels[label_1])
            label_1 = labels[label_1];

        int residual_element = residual_list[j];
        int label_2 = labels[residual_element];
        while (label_2 != labels[label_2])
            label_2 = labels[label_2];

        bool flag = label_1 == label_2;

        if (label_1 < label_2)
        {
            int tmp = label_1;
            label_1 = label_2;
            label_2 = tmp;
        }

        while (!flag)
        {
            int label_3 = atomicMin(&labels[label_1], label_2);
            if (label_3 == label_2)
                flag = true;
            else if (label_3 > label_2)
                label_1 = label_3;
            else if (label_3 < label_2)
            {
                label_1 = label_2;
                label_2 = label_3;
            }
        }
    }
}