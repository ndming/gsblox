#pragma once

#include "gsblox/readers/common.hpp"

namespace gsblox::reader {

[[nodiscard]] std::unique_ptr<Reader> create(const std::filesystem::path& config_file);

} // namespace gsblox::reader
