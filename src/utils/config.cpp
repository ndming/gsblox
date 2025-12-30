#include "gsblox/utils/config.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

bool gsblox::utils::cmp_str_key(const std::string_view str, const std::string_view key, const bool ignore_case) noexcept {
    if (str.size() != key.size()) {
        return false;
    }
    for (size_t i = 0; i < str.size(); ++i) {
        if (ignore_case && std::tolower(static_cast<unsigned char>(str[i])) != std::tolower(static_cast<unsigned char>(key[i]))) {
            return false;
        }
        if (!ignore_case && str[i] != key[i]) {
            return false;
        }
    }
    return true;
}

YAML::Node gsblox::utils::load_yaml(const std::filesystem::path& config_file) {
    auto root = YAML::Node{};
    try {
        root = YAML::LoadFile(config_file.string());
        return root;
    } catch (const YAML::ParserException& e) {
        spdlog::error("Could NOT load YAML file {}, due to: {}", config_file.string(), e.what());
        return {};
    }
}
