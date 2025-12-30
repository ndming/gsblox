#pragma once

#include <nvblox/mapper/multi_mapper.h>

#include <filesystem>

namespace gsblox {

struct FuserConfig {
    float voxel_size; // in meter
    nvblox::MappingType mapping_type;
    nvblox::EsdfMode esdf_mode;

    [[nodiscard]] bool valid() const;
};

struct MapperConfig {
    float max_integration_distance{ 7.0f };

    [[nodiscard]] bool valid() const;
};

struct Sensor;

class Fuser final {
public:
    explicit Fuser(const FuserConfig& config);
    void finish_frame() noexcept;

    void integrate_depth(const nvblox::DepthImage& depth, const Eigen::Isometry3f& c2w, const Sensor& sensor);
    void integrate_color(const nvblox::ColorImage& color, const Eigen::Isometry3f& c2w, const Sensor& sensor);

    void update_mesh();
    void update_esdf();

    void save_background_mesh(const std::filesystem::path& mesh_file) const;
    void save_foreground_mesh(const std::filesystem::path& mesh_file) const;

    [[nodiscard]] uint32_t get_projective_subsampling() const noexcept { return _projective_subsampling; }
    [[nodiscard]] uint32_t get_feat_frame_subsampling() const noexcept { return _feat_frame_subsampling; }
    [[nodiscard]] uint32_t get_mesh_frame_subsampling() const noexcept { return _mesh_frame_subsampling; }
    [[nodiscard]] uint32_t get_esdf_frame_subsampling() const noexcept { return _esdf_frame_subsampling; }

    void set_projective_subsampling(const uint32_t rate) noexcept { _projective_subsampling = rate; }
    void set_feat_frame_subsampling(const uint32_t rate) noexcept { _feat_frame_subsampling = rate; }
    void set_mesh_frame_subsampling(const uint32_t rate) noexcept { _mesh_frame_subsampling = rate; }
    void set_esdf_frame_subsampling(const uint32_t rate) noexcept { _esdf_frame_subsampling = rate; }

    [[nodiscard]] MapperConfig get_background_mapper_config() const noexcept;
    [[nodiscard]] MapperConfig get_foreground_mapper_config() const noexcept;

    void set_background_mapper_config(const MapperConfig& config) noexcept;
    void set_foreground_mapper_config(const MapperConfig& config) noexcept;

private:
    nvblox::MultiMapper _mapper;
    uint32_t _curr_frame{ 0 };

    uint32_t _projective_subsampling{ 1 };
    uint32_t _feat_frame_subsampling{ 1 };
    uint32_t _mesh_frame_subsampling{ 1 };
    uint32_t _esdf_frame_subsampling{ 1 };
};

namespace fuser {

[[nodiscard]] std::unique_ptr<Fuser> create(const std::filesystem::path& config_file);

} // namespace fuser

namespace utils {

[[nodiscard]] nvblox::MappingType get_mapping_type(std::string_view key) noexcept;
[[nodiscard]] nvblox::EsdfMode get_esdf_mode(std::string_view key) noexcept;

[[nodiscard]] std::string to_string(nvblox::MappingType type) noexcept;
[[nodiscard]] std::string to_string(nvblox::EsdfMode mode) noexcept;

} // namespace utils
} // namespace gsblox
