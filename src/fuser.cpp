#include "gsblox/fuser.hpp"
#include "gsblox/sensor.hpp"

#include "gsblox/utils/config.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

bool gsblox::FuserConfig::valid() const {
    if (voxel_size <= 0.0f) {
        spdlog::warn("Invalid fuser config: non-positive voxel_size {}", voxel_size);
        return false;
    }
    if (mapping_type == nvblox::MappingType::kUnset) {
        spdlog::warn("Invalid fuser config: mapping_type is unset");
        return false;
    }
    if (esdf_mode == nvblox::EsdfMode::kUnset) {
        spdlog::warn("Invalid fuser config: esdf_mode is unset");
        return false;
    }
    return true;
}

bool gsblox::MapperConfig::valid() const {
    if (max_integration_distance <= 0.0f) {
        spdlog::warn("Invalid mapper config: non-positive max_integration_distance {}", max_integration_distance);
        return false;
    }
    return true;
}

gsblox::Fuser::Fuser(const FuserConfig& config) : _mapper{ config.voxel_size, config.mapping_type, config.esdf_mode } {
}

void gsblox::Fuser::finish_frame() noexcept {
    ++_curr_frame;
}

void gsblox::Fuser::integrate_depth(
    const nvblox::DepthImage& depth,
    const Eigen::Isometry3f& c2w,
    const Sensor& sensor)
{
    if (_curr_frame % _projective_subsampling == 0) {
        const auto camera = nvblox::Camera{
            sensor.fx, sensor.fy, sensor.cx, sensor.cy,
            static_cast<int>(sensor.width), static_cast<int>(sensor.height) };
        _mapper.integrateDepth(depth, c2w, camera);
    }
}

void gsblox::Fuser::integrate_color(
    const nvblox::ColorImage& color,
    const Eigen::Isometry3f& c2w,
    const Sensor& sensor)
{
    if (_curr_frame % _feat_frame_subsampling == 0) {
        const auto camera = nvblox::Camera{
            sensor.fx, sensor.fy, sensor.cx, sensor.cy,
            static_cast<int>(sensor.width), static_cast<int>(sensor.height) };
        _mapper.integrateColor(color, c2w, camera);
    }
}

void gsblox::Fuser::update_mesh() {
    if (_mesh_frame_subsampling > 0 && _curr_frame % _mesh_frame_subsampling == 0) {
        _mapper.updateColorMesh();
    }
}

void gsblox::Fuser::update_esdf() {
    if (_esdf_frame_subsampling > 0 && _curr_frame % _esdf_frame_subsampling == 0) {
        _mapper.updateEsdf();
    }
}

std::shared_ptr<nvblox::ColorMesh> gsblox::Fuser::get_background_color_mesh(const nvblox::CudaStream& stream) const {
    return _mapper.background_mapper().color_mesh_layer().getMesh(stream);
}

void gsblox::Fuser::get_background_color_mesh(int* n_vertices, int* n_indices) const {
    if (!n_vertices || !n_indices) {
        spdlog::warn("Querying number of color mesh's vertices / triangles with null pointers, doing nothing");
        return;
    }
    _mapper.background_mapper().color_mesh_layer().getMesh(n_vertices, n_indices);
}

void gsblox::Fuser::get_background_color_mesh(
    Eigen::Vector3f* vbo, int* ibo, Eigen::Vector3f* nbo, nvblox::Color* cbo,
    const nvblox::CudaStream& cuda_stream) const
{
    _mapper.background_mapper().color_mesh_layer().getMesh(vbo, ibo, nbo, cbo, cuda_stream);
}

void gsblox::Fuser::save_background_color_mesh(const std::filesystem::path& mesh_file) const {
    if (!_mapper.background_mapper().saveColorMeshAsPly(mesh_file.string())) {
        spdlog::error("Failed to save background color mesh: {}", mesh_file.string());
    }
}

void gsblox::Fuser::save_foreground_color_mesh(const std::filesystem::path& mesh_file) const {
    if (!_mapper.foreground_mapper().saveColorMeshAsPly(mesh_file.string())) {
        spdlog::error("Failed to save foreground color mesh: {}", mesh_file.string());
    }
}

gsblox::MapperConfig gsblox::Fuser::get_background_mapper_config() const noexcept {
    auto config = MapperConfig{};
    config.max_integration_distance = _mapper.background_mapper().tsdf_integrator().max_integration_distance_m();
    return config;
}

gsblox::MapperConfig gsblox::Fuser::get_foreground_mapper_config() const noexcept {
    auto config = MapperConfig{};
    config.max_integration_distance = _mapper.foreground_mapper().tsdf_integrator().max_integration_distance_m();
    return config;
}

void gsblox::Fuser::set_background_mapper_config(const MapperConfig& config) noexcept {
    if (!config.valid()) {
        spdlog::warn("Skipping parameter settings for background mapper due to invalid config");
        return;
    }

    _mapper.background_mapper()->tsdf_integrator().max_integration_distance_m(config.max_integration_distance);
    _mapper.background_mapper()->color_integrator().max_integration_distance_m(config.max_integration_distance);
}

void gsblox::Fuser::set_foreground_mapper_config(const MapperConfig& config) noexcept {
    if (!config.valid()) {
        spdlog::warn("Skipping parameter settings for foreground mapper due to invalid config");
        return;
    }

    _mapper.foreground_mapper()->tsdf_integrator().max_integration_distance_m(config.max_integration_distance);
    _mapper.foreground_mapper()->color_integrator().max_integration_distance_m(config.max_integration_distance);
}

nvblox::MappingType gsblox::utils::get_mapping_type(const std::string_view key) noexcept {
    if (cmp_str_key("static_tsdf",            key, false)) return nvblox::MappingType::kStaticTsdf;
    if (cmp_str_key("static_occupancy",       key, false)) return nvblox::MappingType::kStaticOccupancy;
    if (cmp_str_key("dynamic",                key, false)) return nvblox::MappingType::kDynamic;
    if (cmp_str_key("human_static_tsdf",      key, false)) return nvblox::MappingType::kHumanWithStaticTsdf;
    if (cmp_str_key("human_static_occupancy", key, false)) return nvblox::MappingType::kHumanWithStaticOccupancy;
    return nvblox::MappingType::kUnset;
}

nvblox::EsdfMode gsblox::utils::get_esdf_mode(const std::string_view key) noexcept {
    if (cmp_str_key("3D", key)) return nvblox::EsdfMode::k3D;
    if (cmp_str_key("2D", key)) return nvblox::EsdfMode::k2D;
    return nvblox::EsdfMode::kUnset;
}

std::string gsblox::utils::to_string(const nvblox::MappingType type) noexcept {
    switch (type) {
        case nvblox::MappingType::kStaticTsdf: return "static TSDF";
        case nvblox::MappingType::kStaticOccupancy : return "static occupancy";
        case nvblox::MappingType::kDynamic: return "static TSDF and dynamic occupancy";
        case nvblox::MappingType::kHumanWithStaticTsdf: return "static TSDF and human occupancy";
        case nvblox::MappingType::kHumanWithStaticOccupancy: return "static occupancy and static occupancy";
        default: return "unset";
    }
}

std::string gsblox::utils::to_string(const nvblox::EsdfMode mode) noexcept {
    switch (mode) {
        case nvblox::EsdfMode::k3D: return "3D";
        case nvblox::EsdfMode::k2D: return "2D";
        default: return "unset";
    }
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

std::unique_ptr<gsblox::Fuser> gsblox::fuser::create(const std::filesystem::path& config_file) {
    const auto root = utils::load_yaml(config_file);
    if (!root) {
        spdlog::error("Could NOT create Fuser: empty YAML root");
        return nullptr;
    }

    const auto fuser_node = root["fuser"];
    if (!fuser_node) {
        spdlog::error("Could NOT find fuser node in the config file");
        return nullptr;
    }

    auto config = FuserConfig{};
    read_yaml_node(fuser_node, "voxel_size", &config.voxel_size);

    std::string mapping_type_str;
    read_yaml_node(fuser_node, "mapping_type", &mapping_type_str);
    config.mapping_type = utils::get_mapping_type(mapping_type_str);

    std::string esdf_mode_str;
    read_yaml_node(fuser_node, "esdf_mode", &esdf_mode_str);
    config.esdf_mode = utils::get_esdf_mode(esdf_mode_str);

    if (!config.valid()) {
        spdlog::error("Fuser config is invalid, please check the provided config file: {}", config_file.string());
        return nullptr;
    }

    spdlog::info("Creating fuser:");
    spdlog::info(" - voxel size: {}", config.voxel_size);
    spdlog::info(" - mapping type: {}", utils::to_string(config.mapping_type));
    spdlog::info(" - esdf mode: {}", utils::to_string(config.esdf_mode));

    auto fuser = std::make_unique<Fuser>(config);

    // Set subsampling rates, if any found
    if (uint32_t subsampling = 1; read_yaml_node(fuser_node, "projective_subsampling", &subsampling) && subsampling > 0) {
        fuser->set_projective_subsampling(subsampling);
    } else if (subsampling == 0) {
        spdlog::warn("Invalid fuser config: projective_subsampling must be non-zero, overriding with 1");
    }
    if (uint32_t subsampling = 1; read_yaml_node(fuser_node, "feat_frame_subsampling", &subsampling) && subsampling > 0) {
        fuser->set_feat_frame_subsampling(subsampling);
    } else if (subsampling == 0) {
        spdlog::warn("Invalid fuser config: feat_frame_subsampling must be non-zero, overriding with 1");
    }
    uint32_t subsampling;
    if (read_yaml_node(fuser_node, "mesh_frame_subsampling", &subsampling)) {
        fuser->set_mesh_frame_subsampling(subsampling);
    }
    if (read_yaml_node(fuser_node, "esdf_frame_subsampling", &subsampling)) {
        fuser->set_esdf_frame_subsampling(subsampling);
    }

    // Set background node params
    if (const auto background_node = fuser_node["background"]; background_node && background_node.IsMap()) {
        auto mapper_config = MapperConfig{};
        read_yaml_node(background_node, "max_integration_distance", &mapper_config.max_integration_distance);

        fuser->set_background_mapper_config(mapper_config);
    }

    // Set foreground node params
    if (const auto foreground_node = fuser_node["foreground"]; foreground_node && foreground_node.IsMap()) {
        auto mapper_config = MapperConfig{};
        read_yaml_node(foreground_node, "max_integration_distance", &mapper_config.max_integration_distance);

        fuser->set_foreground_mapper_config(mapper_config);
    }

    return fuser;
}
