#pragma once

#include "gsblox/readers/common.hpp"

#include <fstream>

namespace gsblox {

class ReplicaReader final : public RgbDReader {
public:
    explicit ReplicaReader(const ReaderConfig& config);

    static std::unique_ptr<ReplicaReader> create(const std::filesystem::path& yaml_file) {
        return std::make_unique<ReplicaReader>(ReaderConfig::from_yaml(yaml_file));
    }

    ~ReplicaReader() override { _traj_file.close(); }

private:
    ReadStatus read_color(nvblox::ColorImage*  color) override;
    ReadStatus read_depth(nvblox::DepthImage*  depth) override;
    ReadStatus read_c2w_color(nvblox::Transform* c2w) override;

    std::ifstream _traj_file;
};

} // namespace gsblox
