#pragma once

#include <filesystem>

namespace gsblox::utils {

[[nodiscard]] std::filesystem::path normalize(const std::filesystem::path& path);
[[nodiscard]] std::filesystem::path make_path(const std::string& str_path);
[[nodiscard]] std::filesystem::path make_norm(const std::string& str_path);

} // namespace gsblox::utils
