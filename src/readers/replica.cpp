#include "gsblox/readers/replica.hpp"

#include <spdlog/spdlog.h>

#include "gsblox/utils/image.hpp"

gsblox::ReplicaReader::ReplicaReader(const ReaderConfig& config)
    : Reader{ config }
    , _traj_file{ std::ifstream{ config.scene_dir / "traj.txt" } }
{
    if (!_traj_file.is_open()) {
        spdlog::error("Could NOT open trajectory file at: {}", (config.scene_dir / "traj.txt").string());
        return; // empty reader
    }

    auto line = std::string{};
    while (std::getline(_traj_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        ++_num_frames;
    }

    // Reset stream to initial state
    _traj_file.clear(); // clear eof/fail bit
    _traj_file.seekg(0, std::ios::beg);
}

gsblox::Reader::ReadStatus gsblox::ReplicaReader::read_color(nvblox::ColorImage* color) {
    auto ss = std::stringstream{};
    ss << "frame" << std::setfill('0') << std::setw(6) << _curr_frame << ".jpg";
    const auto file = _config.scene_dir / "results" / ss.str();
    return utils::load_8bit_color_image(file, color) ? ReadStatus::Consumed : ReadStatus::Failed;
}

gsblox::Reader::ReadStatus gsblox::ReplicaReader::read_depth(nvblox::DepthImage* depth) {
    auto ss = std::stringstream{};
    ss << "depth" << std::setfill('0') << std::setw(6) << _curr_frame << ".png";
    const auto file = _config.scene_dir / "results" / ss.str();
    return utils::load_16bit_depth_image(file, depth, _config.depth_multiplier) ? ReadStatus::Consumed : ReadStatus::Failed;
}

template <typename Derived>
void read_matrix_from_line(const std::string& line, Eigen::DenseBase<Derived>* mat_ptr) {
    std::istringstream iss(line);
    for (int row = 0; row < mat_ptr->rows(); row++) {
        for (int col = 0; col < mat_ptr->cols(); col++) {
            float item = 0.0;
            iss >> item;
            (*mat_ptr)(row, col) = item;
        }
    }
}

gsblox::Reader::ReadStatus gsblox::ReplicaReader::read_c2w_color(nvblox::Transform* c2w) {
    // Note: here we're not synchronizing the filestream state with _curr_frame,
    // which shouldn't be a major concern given the simple nature of the dataset
    auto line = std::string{};
    if (std::getline(_traj_file, line)) {
        Eigen::Matrix4f T_c2w;
        read_matrix_from_line(line, &T_c2w);
        *c2w = Eigen::Isometry3f(T_c2w);
    } else {
        spdlog::error("Could NOT read c2w line: {}", line);
        return ReadStatus::Failed;
    }

    // Check that the loaded data doesn't contain NaNs or a faulty rotation matrix
    if (constexpr auto R_det_esp = 1e-4f;
        !c2w->matrix().allFinite() || std::abs(c2w->matrix().block<3, 3>(0, 0).determinant() - 1.0f) > R_det_esp) {
        spdlog::warn("Bad c2w matrix: {}", line);
        return ReadStatus::Skipped; // bad data, but keep going
    }

    return ReadStatus::Consumed;
}
