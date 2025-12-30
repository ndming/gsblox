#pragma once

#include <filesystem>

namespace YAML {
class Node;
}

namespace gsblox::utils {

[[nodiscard]] bool cmp_str_key(std::string_view str, std::string_view key, bool ignore_case = true) noexcept;

[[nodiscard]] YAML::Node load_yaml(const std::filesystem::path& config_file);

} // gsblox::utils
