#include "gsblox/reader.hpp"
#include "gsblox/sensor.hpp"
#include "gsblox/fuser.hpp"
#include "gsblox/viewer.hpp"

#include "gsblox/utils/path.hpp"

#include <pangolin/gl/glcuda.h>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

[[nodiscard]] std::filesystem::path prepare_out_dir(const std::filesystem::path& config_file);
void prepare_log_dir(const char* program, const std::filesystem::path& output_dir);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        spdlog::error("Please provide path to a config file at arg 1");
        return EXIT_FAILURE;
    }

    const auto config_file = gsblox::utils::make_norm(argv[1]);
    spdlog::info("Config file: {}", config_file.string());

    const auto out_dir = prepare_out_dir(config_file);
    if (out_dir.empty()) {
        spdlog::error("Could NOT prepare output directory");
        return EXIT_FAILURE;
    }

    prepare_log_dir(argv[0], out_dir);

    const auto reader = gsblox::reader::create(config_file);
    if (!reader) {
        spdlog::error("Could NOT create reader");
        return EXIT_FAILURE;
    }

    const auto sensors = gsblox::sensor::read(config_file);
    spdlog::info("Found {} sensor(s):", sensors.size());
    for (const auto&[type, width, height, fx, fy, cx, cy] : sensors) {
        spdlog::info(
            "- {}: w={} h={}, fx={}, fy={}, cx={}, cy={}",
            type, width, height, fx, fy, cx, cy);
    }
    if (sensors.empty()) {
        spdlog::error("Could NOT find any sensors");
        return EXIT_FAILURE;
    }

    const auto color_image = std::make_shared<nvblox::ColorImage>(nvblox::MemoryType::kUnified);
    const auto depth_image = std::make_shared<nvblox::DepthImage>(nvblox::MemoryType::kUnified);
    auto c2w = Eigen::Isometry3f::Identity();

    const auto fuser = gsblox::fuser::create(config_file);
    if (!fuser) {
        spdlog::error("Could NOT create fuser");
        return EXIT_FAILURE;
    }

    const auto max_depth = fuser->get_background_mapper_config().max_integration_distance;
    const auto copy_stream = std::make_shared<nvblox::CudaStreamOwning>();

    auto viewer = gsblox::viewer::create(config_file);
    if (!viewer) {
        spdlog::error("Could NOT create viewer");
        return EXIT_FAILURE;
    }

    spdlog::info("Frame count: {}", reader->count());
    auto n_indices = 0; // to draw triangle mesh

    while (!gsblox::Viewer::should_quit()) {
        if (!reader->exhausted()) {
            uint32_t sensor_id = 0;
            if (const auto read_status = reader->next(color_image.get(), depth_image.get(), &c2w, &sensor_id);
                read_status == gsblox::Reader::ReadStatus::Skipped) [[unlikely]] {
                    continue;
                }

            fuser->integrate_depth(*depth_image, c2w, sensors[sensor_id]);
            fuser->integrate_color(*color_image, c2w, sensors[sensor_id]);
            fuser->update_mesh();
            fuser->update_esdf();
            fuser->finish_frame();

            int n_vertices;
            fuser->get_background_color_mesh(&n_vertices, &n_indices);

            viewer->update_vbo(n_vertices);
            viewer->update_cbo(n_vertices);
            viewer->update_ibo(n_indices);

            {
                auto vbo = pangolin::CudaScopedMappedPtr{ viewer->get_mesh_vbo() };
                auto cbo = pangolin::CudaScopedMappedPtr{ viewer->get_mesh_cbo() };
                auto ibo = pangolin::CudaScopedMappedPtr{ viewer->get_mesh_ibo() };
                fuser->get_background_color_mesh(
                    static_cast<Eigen::Vector3f*>(*vbo),
                    static_cast<int*>(*ibo),
                    nullptr,
                    static_cast<nvblox::Color*>(*cbo),
                    *copy_stream
                );
            }
        }

        viewer->draw_mesh(n_indices, true);
        gsblox::Viewer::finish_draw();

        cv::Mat color(color_image->height(), color_image->width(),  CV_8UC3, color_image->dataPtr());
        cv::Mat depth(depth_image->height(), depth_image->width(), CV_32FC1, depth_image->dataPtr());

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

    copy_stream->synchronize();
    cudaDeviceSynchronize();

    viewer.reset();
    cv::destroyAllWindows();

    const auto mesh_output_file = out_dir / "tsdf.ply";
    spdlog::info("Saving color mesh to: {}", mesh_output_file.string());
    fuser->save_background_color_mesh(mesh_output_file);

    spdlog::info("Done");
    return 0;
}

std::filesystem::path prepare_default_output_dir() {
    constexpr auto output_path = "output";
    auto output_dir = gsblox::utils::make_norm(output_path);
    spdlog::info("Creating default output directory at: {}", output_dir.string());

    auto ec = std::error_code{};
    if (std::filesystem::create_directory(output_dir, ec) || !ec) [[likely]] {
        return output_dir; // succeed or already created
    }
    spdlog::error("Failed to create default output directory, due to: {}", output_dir.string(), ec.message());
    return {};
}

std::filesystem::path prepare_out_dir(const std::filesystem::path& config_file) {
    // Find output_dir node in the config file. If successful, create a directory there,
    // otherwise, create a default output directory prepare_default_output_dir
    auto root = YAML::Node{};
    try {
        root = YAML::LoadFile(config_file.string());
    } catch (const YAML::ParserException& e) {
        spdlog::warn("Could NOT load YAML file at: {}, due to: {}", config_file.string(), e.what());
        return prepare_default_output_dir();
    }

    const auto output_node = root["output_dir"];
    if (!output_node || !output_node.IsScalar()) [[unlikely]] {
        spdlog::warn("Could NOT find \'output_dir\' node in the config file, or the node is not a string");
        return prepare_default_output_dir();
    }

    try {
        const auto output_path = output_node.as<std::string>();
        auto output_dir = gsblox::utils::make_norm(output_path);

        auto ec = std::error_code{};
        if (std::filesystem::create_directories(output_dir, ec) || !ec) [[likely]] {
            return output_dir; // succeed or already created
        }
        spdlog::error("Failed to create output directory {}, due to: {}", output_dir.string(), ec.message());
        return {};
    } catch (const YAML::ParserException& e) {
        spdlog::error("Failed to parse output_dir, due to: {}", e.what());
        return prepare_default_output_dir();
    }
}

void prepare_log_dir(const char* program, const std::filesystem::path& output_dir) {
    // Redirect nvblox log to files
    google::InitGoogleLogging(program);
    FLAGS_logtostderr = false;
    // Where to dump glog files
    const auto log_dir = output_dir / "log";
    std::filesystem::create_directory(log_dir);
    FLAGS_log_dir = log_dir.string();
}
