#include "gsblox/readers/tum_rgbd.hpp"

#include "gsblox/utils/config.hpp"
#include "gsblox/utils/image.hpp"
#include "gsblox/utils/io.hpp"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

/// Reads rgb.txt or depth.txt and returns a vector of timestamp-path
std::vector<std::pair<double, std::string>> read_list_file(const std::filesystem::path& file) {
    const auto num_lines = gsblox::utils::peak_lines(file, '#');
    if (num_lines == 0) [[unlikely]] {
        return {};
    }

    // Pre-allocate with peaked number of lines
    auto images = std::vector<std::pair<double, std::string>>{};
    images.reserve(num_lines);

    // Consume lines
    auto txt_file = std::ifstream{ file };
    std::string line;
    while (std::getline(txt_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto iss = std::istringstream{ line };
        auto t = 0.0;
        auto p = std::string{};
        iss >> t >> p;
        images.emplace_back(t, p);
    }

    return images;
}

struct Pose {
    double timestamp{ 0.0f };
    Eigen::Vector3f t{};
    Eigen::Quaternionf q{};
};

/// Reads groundtruth.txt and returns a vector of timestamp-pose
std::vector<Pose> read_pose_file(const std::filesystem::path& file) {
    const auto num_lines = gsblox::utils::peak_lines(file, '#');
    if (num_lines == 0) [[unlikely]] {
        return {};
    }

    // Pre-allocate with peaked number of lines
    auto poses = std::vector<Pose>{};
    poses.reserve(num_lines);

    // Consume poses
    auto txt_file = std::ifstream{ file };
    std::string line;
    while (std::getline(txt_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto iss = std::istringstream{ line };
        auto p = Pose{};
        // timestamp tx ty tz qx qy qz qw
        iss >> p.timestamp
            >> p.t.x() >> p.t.y() >> p.t.z()
            >> p.q.x() >> p.q.y() >> p.q.z() >> p.q.w();
        p.q.normalize();
        poses.push_back(std::move(p));
    }

    return poses;
}

std::pair<Pose, Pose> find_surrounding_poses(const std::vector<Pose>& poses, const double t) {
    // TODO: use binary search to reduce the query time
    for (auto i = 0u; i + 1 < poses.size(); ++i) {
        if (poses[i].timestamp == t) {
            return { poses[i], poses[i] };
        }
        if (poses[i + 1].timestamp == t) {
            return { poses[i + 1], poses[i + 1] };
        }
        if (poses[i].timestamp < t && t < poses[i + 1].timestamp) {
            return { poses[i], poses[i + 1] };
        }
    }
    return {};
}

std::vector<int> find_associated_indices(
    const std::vector<std::pair<double, std::string>>& depth_images,
    const std::vector<std::pair<double, std::string>>& color_images,
    const double max_timestamp_difference)
{
    auto indices = std::vector(depth_images.size(), -1);
    auto used = std::vector(color_images.size(), false);

    for (uint32_t i = 0; i < depth_images.size(); ++i) {
        double best = max_timestamp_difference;
        auto best_j = -1;
        for (auto j = 0; j < static_cast<int>(color_images.size()); ++j) {
            if (const auto dt = std::fabs(depth_images[i].first - color_images[j].first);
                dt < best && !used[j]) {
                best = dt;
                best_j = j;
            }
        }
        if (best_j >= 0) {
            used[best_j] = true;
            indices[i] = best_j;
        } else {
            spdlog::warn(
                "Could NOT find an associated color image for depth at timestamp {}, using max difference: {}",
                depth_images[i].first, max_timestamp_difference);
        }
    }

    return indices;
}

gsblox::TumRgbDReader::TumRgbDReader(
    const ReaderConfig& config,
    const double max_timestamp_difference)
    : Reader{ config }
{
    // The main driving forces are depth and poses: rgb images selected based on their timestamps
    auto depth_images = read_list_file(config.scene_dir / "depth.txt");
    if (depth_images.empty()) [[unlikely]] {
        spdlog::error("Empty depth list at: {}", (config.scene_dir / "depth.txt").string());
        return; // empty reader
    }

    // Use as many depth images as we have read
    _frames.resize(depth_images.size());

    // Process ground-truth poses
    const auto pose_file = config.scene_dir / "groundtruth.txt";
    if (!std::filesystem::exists(pose_file)) [[unlikely]] {
        spdlog::error("Could NOT find ground-truth pose file: {}", pose_file.string());
        return; // empty reader
    }

    const auto poses = read_pose_file(pose_file);
    if (poses.empty()) [[unlikely]] {
        spdlog::error("Empty pose file: {}", pose_file.string());
        return; // empty reader
    }

    auto first_inv = Eigen::Matrix4f{};
    bool first = true;

    for (auto i = 0u; i < depth_images.size(); ++i) {
        const auto depth_timestamp = depth_images[i].first;
        const auto [p0, p1] = find_surrounding_poses(poses, depth_timestamp);
        if (p0.timestamp == 0.0 && p1.timestamp == 0.0) {
            spdlog::warn("Could NOT find surrounding poses for depth image at timestamp: {}, discarding it", depth_timestamp);
            continue;
        }

        auto pose_t = p0.t;
        auto pose_q = p0.q;

        // Interpolate poses surrounding the depth image's timestamp,
        // only do this if the 2 timestamps are different
        if (const auto dt = p1.timestamp - p0.timestamp; dt > 0.0) {
            const auto alpha = (depth_timestamp - p0.timestamp) / dt;
            pose_t = (1.f - alpha) * p0.t + alpha * p1.t;
            pose_q = p0.q.slerp(static_cast<float>(alpha), p1.q);
        }
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3,3>(0,0) = pose_q.toRotationMatrix();
        T.block<3,1>(0,3) = pose_t;

        // Transform the whole trajectory so that the first pose is identity
        if (first) {
            first_inv = T.inverse();
            T.setIdentity();
            first = false;
        } else {
            T = first_inv * T;
        }

        auto frame = Frame{};
        frame.depth_path = std::move(depth_images[i].second);
        frame.c2w = Eigen::Isometry3f(T);
        _frames[i] = std::move(frame);
    }

    // We may not have color images at all, check if a rgb.txt presents first
    if (const auto color_list_file = config.scene_dir / "rgb.txt"; std::filesystem::exists(color_list_file)) {
        if (auto color_images = read_list_file(color_list_file); !color_images.empty()) {
            // indices has the same size as depth_images (hence _frames), whose values indexing into color_images,
            // where values of -1 indicate that there was not a color image satisfying max_timestamp_difference
            const auto indices = find_associated_indices(depth_images, color_images, max_timestamp_difference);
            for (auto i = 0u; i < indices.size(); ++i) {
                if (indices[i] < 0) continue;
                _frames[i].color_path = std::move(color_images[i].second);
            }
        } else [[unlikely]] {
            spdlog::warn("Found an rgb.txt: {}, yet we could NOT read it, skipping color images", color_list_file.string());
        }
    }

    _num_frames = _frames.size();
}

std::unique_ptr<gsblox::TumRgbDReader> gsblox::TumRgbDReader::create(const std::filesystem::path& yaml_file) {
    const auto config = ReaderConfig::from_yaml(yaml_file);
    auto max_timestamp_difference = DEFAULT_MAX_TIMESTAMP_DIFFERENCE;

    // Find a few more params specific to TUM-RGBD
    const auto root = utils::load_yaml(yaml_file);

    const auto reader_node = root["reader"];
    if (!reader_node || !reader_node.IsMap()) [[unlikely]] {
        // Use the default max_timestamp_difference
        return std::make_unique<TumRgbDReader>(config, max_timestamp_difference);
    }

    try {
        // Read max_timestamp_difference from config file if it presents
        max_timestamp_difference = reader_node["max_timestamp_difference"].as<double>();
    } catch (const YAML::Exception& _) {
        // Use the default max_timestamp_difference
        return std::make_unique<TumRgbDReader>(config, max_timestamp_difference);
    }

    // Use max_timestamp_difference in the config file
    return std::make_unique<TumRgbDReader>(config, max_timestamp_difference);
}

gsblox::Reader::ReadStatus gsblox::TumRgbDReader::read_color(nvblox::ColorImage* color) {
    if (_frames[_curr_frame].color_path.empty()) {
        spdlog::warn("There was not a color image associated with depth frame {}, skipping the whole frame", _curr_frame);
        return ReadStatus::Skipped;
    }
    const auto file = _config.scene_dir / _frames[_curr_frame].color_path;
    return utils::load_8bit_color_image(file, color) ? ReadStatus::Consumed : ReadStatus::Failed;
}

gsblox::Reader::ReadStatus gsblox::TumRgbDReader::read_depth(nvblox::DepthImage* depth) {
    if (_frames[_curr_frame].depth_path.empty()) {
        spdlog::warn("There was not a pose associated with depth frame {}, skipping the whole frame", _curr_frame);
        return ReadStatus::Skipped;
    }
    const auto file = _config.scene_dir / _frames[_curr_frame].depth_path;
    return utils::load_16bit_depth_image(file, depth, _config.depth_multiplier) ? ReadStatus::Consumed : ReadStatus::Failed;
}

gsblox::Reader::ReadStatus gsblox::TumRgbDReader::read_c2w_color(nvblox::Transform* c2w) {
    *c2w = _frames[_curr_frame].c2w;
    return ReadStatus::Consumed;
}
