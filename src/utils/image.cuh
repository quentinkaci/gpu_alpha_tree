#pragma once

#include <memory>
#include <png++/png.hpp>

namespace utils
{
    struct RGBPixel
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
    };

    struct RGBImage
    {
        RGBImage(uint height, uint width);

        ~RGBImage();

        static std::shared_ptr<RGBImage> load(const std::string& filename);
        
        void save(const std::string& filename) const;

        uint height;
        uint width;
        RGBPixel* pixels;
    };
} // namespace utils