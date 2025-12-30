#include "gsblox/reader.hpp"

#include "gsblox/readers/replica.hpp"
#include "gsblox/readers/tum_rgbd.hpp"

#include <spdlog/spdlog.h>

std::unique_ptr<gsblox::Reader> gsblox::reader::create(const std::filesystem::path& config_file) {
    const auto config = ReaderConfig::from_yaml(config_file);
    if (!config.valid()) {
        spdlog::error("Reader config is invalid");
        return nullptr;
    }

    const auto reader_type = config.reader_type;
    spdlog::info("Creating reader: {}", utils::to_string(reader_type));
    switch (reader_type) {
        case ReaderType::Replica: return ReplicaReader::create(config_file);
        case ReaderType::TumRgbD: return TumRgbDReader::create(config_file);
        default:
            spdlog::error("Unknown reader type {}", utils::to_string(reader_type));
            return nullptr;
    }
}
