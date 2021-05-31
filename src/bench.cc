#include <benchmark/benchmark.h>

#include "utils/image.cuh"
#include "cpu_implem/alpha_tree_cpu.hh"
#include "gpu_implem/alpha_tree_gpu.cuh"

void bench_cpu(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto image = utils::RGBImage::load("../resources/batiment.png");

        alpha_tree_cpu(image);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void bench_gpu(benchmark::State& st)
{
    for (auto _ : st)
    {
        auto image = utils::RGBImage::load("../resources/batiment.png");

        alpha_tree_gpu(image);
    }

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(bench_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(bench_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_MAIN();