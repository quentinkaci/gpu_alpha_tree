#include "cc_labelling.cuh"

void initialization_step(const int* nn_list, int max_len, int* residual_list, int* labels, int i)
{
    // FIXME int i = 0;

    int min = i;
    bool found_min = false;

    for (int j = i * max_len; (j - i * max_len) < max_len && nn_list[j] != -1; ++j)
    {
        if (nn_list[j] < min)
        {
            min = nn_list[j];
            found_min = true;
        }
    }

    int pos = i * max_len;
    for (int j = i * max_len; (j - i * max_len) < max_len && nn_list[j] != -1; ++j)
    {
        if (nn_list[j] < i && nn_list[j] != min)
            residual_list[pos++] = nn_list[j];
    }

    // Assign label
    labels[i] = found_min ? min : i;
}

void anylisis_step(int* labels, int i)
{
    // FIXME int i = 0;

    int last_label;
    do
    {
        last_label = labels[i];
        labels[i] = labels[labels[i]];
    } while (labels[i] != last_label);
}

#include <cstdio>

__global__ void reduction_step(const int* residual_list, int max_len, int* labels, int i)
{
    if (threadIdx.x != 0 && threadIdx.y != 0)
        return;

    // FIXME int i = 0;

    for (int j = i * max_len; (j - i * max_len) < max_len && residual_list[j] != -1; ++j)
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