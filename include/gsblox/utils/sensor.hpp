#pragma once

#include <filesystem>
#include <vector>

namespace gsblox {

struct Sensor {
    std::string type;
    uint32_t width;
    uint32_t height;
    float fx;
    float fy;
    float cx;
    float cy;
};

namespace utils {

[[nodiscard]] std::vector<Sensor> read_sensors(const std::filesystem::path& yaml_file);

} // namespace utils
} // namespace gsblox
