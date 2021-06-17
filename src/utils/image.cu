#include "utils/image.cuh"

#include "cuda_error.cuh"

namespace utils
{
    RGBImage::RGBImage(uint height, uint width)
        : height(height), width(width)
    {
        cudaMallocManaged(&pixels, height * width * sizeof(RGBPixel));
        checkCudaError();
    }

    RGBImage::~RGBImage()
    {
        cudaFree(pixels);
        checkCudaError();
    }

    std::shared_ptr<RGBImage> RGBImage::load(const std::string& filename)
    {
        png::image<png::rgb_pixel> image(filename);
        std::shared_ptr<RGBImage> res = std::make_shared<RGBImage>(image.get_height(), image.get_width());

        auto height = image.get_height();
        auto width = image.get_width();

        for (size_t j = 0; j < height; ++j)
        {
            for (size_t i = 0; i < width; ++i)
            {
                auto pix = image[j][i];
                res->pixels[i + j * width] = {pix.red, pix.green, pix.blue};
            }
        }

        return res;
    }

    void RGBImage::save(const std::string& filename) const
    {
        png::image<png::rgb_pixel> image(width, height);

        for (size_t j = 0; j < height; ++j)
        {
            for (size_t i = 0; i < width; ++i)
            {
                auto pix = pixels[i + j * width];
                image[j][i] = png::rgb_pixel(pix.r, pix.g, pix.b);
            }
        }

        image.write(filename);
    }
} // namespace utils