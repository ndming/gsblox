#include "gsblox/utils/image.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

bool gsblox::utils::load_16bit_depth_image(
    const std::filesystem::path& file,
    nvblox::DepthImage* out_depth,
    const float depth_value_multiplier)
{
    if (!out_depth) {
        spdlog::error("out_depth must be non-null");
        return false;
    }

    // Load depth image as-is (keeps uint16)
    const auto depth_raw = cv::imread(file.string(), cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) {
        spdlog::error("Failed to load depth image at: {}", file.string());
        return false;
    }

    // Validate format
    if (depth_raw.type() != CV_16UC1) {
        spdlog::error("Expected CV_16UC1 depth image, got type {} from file: {}", depth_raw.type(), file.string());
        return false;
    }

    // Convert to float with scaling (uint16 -> float)
    cv::Mat depth_float;
    depth_raw.convertTo(depth_float, CV_32FC1, depth_value_multiplier);

    // Copy to nvblox-owned image
    out_depth->copyFrom(depth_raw.rows, depth_raw.cols, reinterpret_cast<float*>(depth_float.data));
    return true;
}

bool gsblox::utils::load_8bit_color_image(
    const std::filesystem::path& file,
    nvblox::ColorImage* out_color)
{
    if (!out_color) {
        spdlog::error("out_color must be non-null");
        return false;
    }

    // Load image (force 3 channels, 8-bit)
    const auto color_bgr = cv::imread(file.string(), cv::IMREAD_COLOR);
    if (color_bgr.empty()) {
        spdlog::error("Failed to load color image at: {}", file.string());
        return false;
    }

    // Validate format
    if (color_bgr.type() != CV_8UC3) {
        spdlog::error("Expected CV_8UC3 image, got type {} from file: {}", color_bgr.type(), file.string());
        return false;
    }

    // Convert BGR -> RGB
    cv::Mat color_rgb;
    cv::cvtColor(color_bgr, color_rgb, cv::COLOR_BGR2RGB);

    // Copy to nvblox-owned image
    out_color->copyFrom(color_rgb.rows, color_rgb.cols, reinterpret_cast<nvblox::Color*>(color_rgb.data));
    return true;
}
