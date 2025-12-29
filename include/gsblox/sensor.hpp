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

namespace sensor {

[[nodiscard]] std::vector<Sensor> read(const std::filesystem::path& config_file);

} // namespace sensor
} // namespace gsblox