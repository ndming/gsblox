#include "gsblox/sensor.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

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

std::vector<gsblox::Sensor> gsblox::sensor::read(const std::filesystem::path& config_file) {
    auto root = YAML::Node{};
    try {
        root = YAML::LoadFile(config_file.string());
    } catch (const YAML::ParserException& e) {
        spdlog::error("Could NOT load YAML file at: {}, due to: {}", config_file.string(), e.what());
        return {};
    }

    const auto sensor_list = root["sensors"];
    if (!sensor_list || !sensor_list.IsSequence()) [[unlikely]] {
        spdlog::error("Could NOT find \'sensors\' node in the config file, or the node is not a list of sensors");
        return {};
    }

    auto sensors = std::vector<Sensor>(sensor_list.size());
    for (auto i = 0u; i < sensors.size(); i++) {
        auto sensor = Sensor{};
        read_yaml_node(sensor_list[i], "type", &sensor.type);
        read_yaml_node(sensor_list[i], "width", &sensor.width);
        read_yaml_node(sensor_list[i], "height", &sensor.height);
        read_yaml_node(sensor_list[i], "fx", &sensor.fx);
        read_yaml_node(sensor_list[i], "fy", &sensor.fy);
        read_yaml_node(sensor_list[i], "cx", &sensor.cx);
        read_yaml_node(sensor_list[i], "cy", &sensor.cy);
        sensors[i] = std::move(sensor);
    }

    return sensors;
}
