#pragma once

#include <filesystem>

namespace gsblox::utils {

[[nodiscard]] std::filesystem::path prepare_out_dir(const std::filesystem::path& config_file);
void prepare_log_dir(const char* program, const std::filesystem::path& output_dir);

[[nodiscard]] std::size_t peak_lines(const std::filesystem::path& file, char ignore_symbol = '#');

} // namespace gsblox::utils
