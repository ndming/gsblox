#include <gsblox/readers/tum_rgbd.hpp>
#include <gsblox/utils/path.hpp>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char* argv[]) {
    auto config = gsblox::ReaderConfig{};
    config.scene_dir = gsblox::utils::make_norm(argv[1]);
    config.depth_scale = 1.0f / 5000.f;
    config.type = gsblox::ReaderType::TumRgbD;

    auto reader = gsblox::TumRgbDReader{ config };
    spdlog::info("Frame count: {}", reader.count());

    const auto color_image = std::make_shared<nvblox::ColorImage>(nvblox::MemoryType::kHost);
    const auto depth_image = std::make_shared<nvblox::DepthImage>(nvblox::MemoryType::kHost);
    auto c2w = Eigen::Isometry3f::Identity();

    while (!reader.exhausted()) {
        reader.next(color_image.get(), depth_image.get(), &c2w);
        // spdlog::info("Pose: {}", c2w.matrix());

        cv::Mat color(color_image->height(), color_image->width(),  CV_8UC3, color_image->dataPtr());
        cv::Mat depth(depth_image->height(), depth_image->width(), CV_32FC1, depth_image->dataPtr());

        constexpr float max_depth = 5.0f; // meters

        // Clamp depth values
        cv::Mat depth_clamped;
        cv::min(depth, max_depth, depth_clamped);

        // Convert to 8-bit WITHOUT normalization
        cv::Mat depth_u8;
        depth_clamped.convertTo(depth_u8, CV_8UC1, 255.0f / max_depth);

        // Optional colormap
        cv::Mat depth_color;
        cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_TURBO);

        cv::Mat combined;
        cv::hconcat(color, depth_color, combined);

        cv::imshow("Color | Depth", combined);

        if (int key = cv::waitKey(0); key == 27) {
            break;
        }
    }

    return 0;
}
