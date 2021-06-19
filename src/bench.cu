#include "cpu_implem/alpha_tree_cpu.hh"
#include "gpu_implem/alpha_tree_gpu.cuh"
#include "utils/cuda_error.cuh"
#include "utils/image.cuh"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

static std::vector<std::string> bench_images = {"../resources/plane.png", "../resources/nature.png", "../resources/mountains.png", "../resources/hong_kong.png"};

void bench_gpu(benchmark::State& st)
{
    int image_id = st.range(0);

    while (st.KeepRunning())
    {
        st.PauseTiming();
        auto image = utils::RGBImage::load(bench_images[image_id]);
        st.ResumeTiming();
        alpha_tree_gpu(image);
    }
}

void bench_cpu(benchmark::State& st)
{
    int image_id = st.range(0);

    while (st.KeepRunning())
    {
        st.PauseTiming();
        auto image = utils::RGBImage::load(bench_images[image_id]);
        st.ResumeTiming();
        alpha_tree_cpu(image);
    }
}

BENCHMARK(bench_cpu)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(bench_gpu)->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Unit(benchmark::kMillisecond)->UseRealTime();

int main(int argc, char** argv)
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 1024);
    checkCudaError();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}