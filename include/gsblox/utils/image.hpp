#pragma once

#include <nvblox/sensors/image.h>

#include <filesystem>

namespace gsblox::utils {

bool load_16bit_depth_image(
    const std::filesystem::path& file,
    nvblox::DepthImage* out_depth,
    float depth_value_multiplier);

bool load_8bit_color_image(
    const std::filesystem::path& file,
    nvblox::ColorImage* out_color);

} // namespace gsblox::utils
