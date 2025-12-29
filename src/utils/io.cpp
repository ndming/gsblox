#include "gsblox/utils/io.hpp"
#include "gsblox/utils/path.hpp"

#include <glog/logging.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

std::filesystem::path prepare_default_output_dir() {
    constexpr auto output_path = "output";
    const auto output_dir = gsblox::utils::make_norm(output_path);

    auto ec = std::error_code{};
    if (std::filesystem::create_directory(output_dir, ec) || !ec) [[likely]] {
        return output_dir; // succeed or already created
    }
    spdlog::error("Failed to create default output directory, due to: {}", output_dir.string(), ec.message());
    return {};
}

std::filesystem::path gsblox::utils::prepare_out_dir(const std::filesystem::path& config_file) {
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
        const auto output_dir  = make_norm(output_path);

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

void gsblox::utils::prepare_log_dir(const char* program, const std::filesystem::path& output_dir) {
    // Redirect nvblox log to files
    google::InitGoogleLogging(program);
    FLAGS_logtostderr = false;
    // Where to dump glog files
    const auto log_dir = output_dir / "log";
    std::filesystem::create_directory(log_dir);
    FLAGS_log_dir = log_dir.string();
}
