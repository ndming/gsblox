#include "gsblox/readers/common.hpp"

#include "gsblox/utils/config.hpp"
#include "gsblox/utils/path.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <chrono>

gsblox::ReaderType gsblox::utils::get_reader_type(const std::string_view key) noexcept {
    if (cmp_str_key(key, "replica")) {
        return ReaderType::Replica;
    }
    if (cmp_str_key(key, "tum_rgbd")) {
        return ReaderType::TumRgbD;
    }
    return ReaderType::Unknown;
}

std::string gsblox::utils::to_string(const ReaderType type) noexcept {
    switch (type) {
        case ReaderType::Replica: return "replica";
        case ReaderType::TumRgbD: return "tum_rgbd";
        default: return "unknown";
    }
}

bool gsblox::ReaderConfig::valid() const {
    if (!std::filesystem::exists(scene_dir)) {
        spdlog::warn("Invalid reader config: non-existing scene_dir path {}", scene_dir.string());
        return false;
    }
    if (reader_type == ReaderType::Unknown) {
        spdlog::warn("Invalid reader config: unknown reader_type");
        return false;
    }
    if (std::isnan(depth_multiplier)) {
        spdlog::warn("Invalid reader config: depth_multiplier is NaN");
        return false;
    }
    return true;
}

template <typename T>
bool read_yaml_node(const YAML::Node& node, const std::string_view key, T* out) {
    if (!node[key.data()]) [[unlikely]] {
        return false;
    }

    try {
        *out = node[key.data()].as<T>();
        return true;
    } catch (const YAML::Exception& e) {
        spdlog::error("Failed to parse key {}, due to: {}", key, e.what());
        return false;
    }
}

gsblox::ReaderConfig gsblox::ReaderConfig::from_yaml(const std::filesystem::path& file) {
    const auto root = utils::load_yaml(file);

    const auto reader_node = root["reader"];
    if (!reader_node || !reader_node.IsMap()) [[unlikely]] {
        spdlog::error("Could NOT find reader node in: {}", file.string());
        return {};
    }

    // Start populating config
    auto config = ReaderConfig{};

    // Get reader type
    auto reader_type_str = std::string{};
    read_yaml_node(reader_node, "reader_type", &reader_type_str);
    config.reader_type = utils::get_reader_type(reader_type_str);

    // Get scene dir
    auto scene_dir_str = std::string{};
    read_yaml_node(reader_node, "scene_dir", &scene_dir_str);
    config.scene_dir = utils::make_norm(scene_dir_str);

    // Depth scale -> depth multiplier
    auto depth_scale = 1.0f;
    read_yaml_node(reader_node, "depth_scale", &depth_scale);
    config.depth_multiplier = 1.0f / depth_scale;

    // Get the remaining config vars
    read_yaml_node(reader_node, "fps", &config.fps);
    read_yaml_node(reader_node, "drop_frames", &config.drop_frames);
    read_yaml_node(reader_node, "is_live", &config.is_live);

    return config;
}

gsblox::Reader::Reader(ReaderConfig config)
    : _config{ std::move(config) }
    , _num_frames{ 0 }
    , _curr_frame{ 0 }
    , _last_frame_time{ std::chrono::steady_clock::now() }
{
}

void gsblox::Reader::wait_then_increment(const ReadStatus status) {
    if (status == ReadStatus::Failed) {
        spdlog::error("Failed to read frame at index {}, halting frame increment", _curr_frame);
        return;
    }

    if (_config.fps == 0.0f || _config.is_live) {
        ++_curr_frame;
        return;
    }

    using namespace std::chrono;
    const auto spf = 1.0f / _config.fps;
    const auto now = steady_clock::now();

    if (const auto elapsed = duration<float>(now - _last_frame_time).count(); elapsed <= spf) {
        const auto seconds_to_sleep = spf - elapsed;
        std::this_thread::sleep_for(duration<float>(seconds_to_sleep));
    } else if (_config.drop_frames && _curr_frame != 0) {
        const auto frames_to_skip = static_cast<uint32_t>(std::ceil(elapsed * _config.fps));
        _curr_frame += frames_to_skip;
    } else {
        ++_curr_frame;
    }

    _last_frame_time = steady_clock::now();
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::ColorImage* color, uint32_t* sensor_id) {
    const auto status = read_color(color) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::Transform* c2w, uint32_t* sensor_id) {
    const auto status = read_c2w_color(c2w) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::ColorImage* color, nvblox::Transform* c2w, uint32_t* sensor_id) {
    const auto status = read_color(color) + read_c2w_color(c2w) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::DepthImage* depth, uint32_t* sensor_id) {
    const auto status = read_depth(depth) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::Transform* c2w_color, nvblox::Transform* c2w_depth, uint32_t* sensor_id) {
    const auto status = read_c2w_color(c2w_color) + read_c2w_depth(c2w_depth) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(nvblox::ColorImage* color, nvblox::DepthImage* depth, uint32_t* sensor_id) {
    const auto status = read_color(color) + read_depth(depth) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(
    nvblox::ColorImage* color,
    nvblox::DepthImage* depth,
    nvblox::Transform* c2w,
    uint32_t* sensor_id)
{
    const auto status = read_color(color) + read_depth(depth) + read_c2w_color(c2w) + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::next(
    nvblox::ColorImage* color, nvblox::Transform* c2w_color,
    nvblox::DepthImage* depth, nvblox::Transform* c2w_depth,
    uint32_t* sensor_id)
{
    const auto status = read_color(color) + read_c2w_color(c2w_color)
                      + read_depth(depth) + read_c2w_depth(c2w_depth)
                      + read_sensor(sensor_id);
    wait_then_increment(status);
    return status;
}

gsblox::Reader::ReadStatus gsblox::Reader::read_color(nvblox::ColorImage* color) {
    spdlog::error("Failed to read color images: it's likely that the underlying dataset reader does not support this operation");
    return ReadStatus::Failed;
}

gsblox::Reader::ReadStatus gsblox::Reader::read_depth(nvblox::DepthImage* depth) {
    spdlog::error("Failed to read color images: it's likely that the underlying dataset reader does not support this operation");
    return ReadStatus::Failed;
}

gsblox::Reader::ReadStatus gsblox::Reader::read_c2w_color(nvblox::Transform* c2w) {
    // Default to identity (no transform from cam to world)
    if (c2w) *c2w = nvblox::Transform::Identity();
    return ReadStatus::Consumed;
}

gsblox::Reader::ReadStatus gsblox::Reader::read_c2w_depth(nvblox::Transform* c2w) {
    // Steal the default behavior of c2w color
    return read_c2w_color(c2w);
}

gsblox::Reader::ReadStatus gsblox::Reader::read_sensor(uint32_t* sensor_id) {
    // Default to the first (and only) sensor
    if (sensor_id) *sensor_id = 0;
    return ReadStatus::Consumed;
}
