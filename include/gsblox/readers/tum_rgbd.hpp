#pragma once

#include "gsblox/readers/common.hpp"

namespace gsblox {

class TumRgbDReader final : public Reader {
public:
    explicit TumRgbDReader(const ReaderConfig& config, double max_timestamp_difference = DEFAULT_MAX_TIMESTAMP_DIFFERENCE);

    static std::unique_ptr<TumRgbDReader> create(const std::filesystem::path& yaml_file);
    static constexpr auto DEFAULT_MAX_TIMESTAMP_DIFFERENCE = 0.02;

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
