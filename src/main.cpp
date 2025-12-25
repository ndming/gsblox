#include "gsblox/readers/replica.hpp"
#include "gsblox/readers/tum_rgbd.hpp"
#include "gsblox/utils/path.hpp"
#include "gsblox/utils/sensor.hpp"

#include <nvblox/mapper/multi_mapper.h>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char* argv[]) {
    // Redirect nvblox log to files
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_log_dir = "output/nvblox/replica/log";

    if (argc < 2) {
        spdlog::error("Please provide path to a config file at arg 1");
        return EXIT_FAILURE;
    }

    const auto config_file = gsblox::utils::make_norm(argv[1]);
    spdlog::info("Config file: {}", config_file.string());

    const auto reader = gsblox::ReplicaReader::create(config_file);
    spdlog::info("Frame count: {}", reader->count());
    if (reader->count() == 0) {
        spdlog::error("Could NOT create reader");
        return EXIT_FAILURE;
    }

    const auto sensors = gsblox::utils::read_sensors(config_file);
    spdlog::info("Found {} sensor(s):", sensors.size());
    for (const auto& sensor : sensors) {
        spdlog::info(
            "- {}: w={} h={}, fx={}, fy={}, cx={}, cy={}",
            sensor.type, sensor.width, sensor.height, sensor.fx, sensor.fy, sensor.cx, sensor.cy);
    }

    const auto color_image = std::make_shared<nvblox::ColorImage>(nvblox::MemoryType::kUnified);
    const auto depth_image = std::make_shared<nvblox::DepthImage>(nvblox::MemoryType::kUnified);
    auto c2w = Eigen::Isometry3f::Identity();

    const auto multi_mapper = std::make_unique<nvblox::MultiMapper>(
        0.02, // voxel size
        nvblox::MappingType::kStaticTsdf,
        nvblox::EsdfMode::k3D
    );

    const auto camera = nvblox::Camera{
        sensors[0].fx, sensors[0].fy, sensors[0].cx, sensors[0].cy,
        static_cast<int>(sensors[0].width), static_cast<int>(sensors[0].height) };

    while (!reader->exhausted()) {
        if (const auto read_status = reader->next(color_image.get(), depth_image.get(), &c2w);
            read_status == gsblox::Reader::ReadStatus::Skipped) {
            continue;
        }

        multi_mapper->integrateDepth(*depth_image, c2w, camera);
        multi_mapper->integrateColor(*color_image, c2w, camera);
        multi_mapper->updateColorMesh();
        multi_mapper->updateEsdf();

        cv::Mat color(color_image->height(), color_image->width(),  CV_8UC3, color_image->dataPtr());
        cv::Mat depth(depth_image->height(), depth_image->width(), CV_32FC1, depth_image->dataPtr());

        constexpr float max_depth = 5.0f; // meters

        // RGB -> BGR
        cv::Mat color_bgr;
        cv::cvtColor(color, color_bgr, cv::COLOR_RGB2BGR);

        // Clamp depth values
        cv::Mat depth_clamped;
        cv::min(depth, max_depth, depth_clamped);

        // Convert to 8-bit WITHOUT normalization
        cv::Mat depth_u8;
        depth_clamped.convertTo(depth_u8, CV_8UC1, 255.0f / max_depth);

        // Optional colormap
        cv::Mat depth_color;
        cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_CIVIDIS);

        cv::Mat combined;
        cv::hconcat(color_bgr, depth_color, combined);

        cv::imshow("Color | Depth", combined);
        cv::waitKey(1);
    }

    const auto mesh_output_file = gsblox::utils::make_norm("output/nvblox/replica/office0.ply");
    if (!multi_mapper->background_mapper()->saveColorMeshAsPly(mesh_output_file.string())) {
        spdlog::error("Failed to save color mesh: {}", mesh_output_file.string());
    }

    return 0;
}
