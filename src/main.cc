#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "cpu_implem/alpha_tree_cpu.hh"
#include "gpu_implem/alpha_tree_gpu.cuh"
#include "utils/image.cuh"

int main(int argc, char** argv)
{
    std::string path = "../resources/hong_kong.png";
    std::string mode = "GPU";

    CLI::App app("gpu_alpha_tree");
    app.add_option("-i", path, "Input image");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");
    CLI11_PARSE(app, argc, argv);

    spdlog::info("Running on {} implementation on {}", mode, path);

    auto image = utils::RGBImage::load(path);
    if (mode == "GPU")
        alpha_tree_gpu(image);
    else if (mode == "CPU")
        alpha_tree_cpu(image);
    image->save("output.png");

    return 0;
}
