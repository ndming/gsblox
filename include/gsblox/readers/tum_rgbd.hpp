#pragma once

#include "gsblox/readers/common.hpp"

namespace gsblox {

class TumRgbDReader final : public RgbDReader {
public:
    explicit TumRgbDReader(const ReaderConfig& config, float max_timestamp_difference = 0.02f);

    static std::unique_ptr<TumRgbDReader> create(const std::filesystem::path& yaml_file) {
        return std::make_unique<TumRgbDReader>(ReaderConfig::from_yaml(yaml_file));
    }

private:
    ReadStatus read_color(nvblox::ColorImage*  color) override;
    ReadStatus read_depth(nvblox::DepthImage*  depth) override;
    ReadStatus read_c2w_color(nvblox::Transform* c2w) override;

    struct Frame {
        std::string color_path{};
        std::string depth_path{};
        nvblox::Transform c2w{ Eigen::Isometry3f::Identity() };
    };

    std::vector<Frame> _frames{};
};

} // namespace gsblox
