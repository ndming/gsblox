#pragma once

#include <nvblox/sensors/image.h>

#include <filesystem>

namespace gsblox {

enum class ReaderType {
    Replica,
    TumRgbD,
    Unknown,
};

ReaderType get_reader_type(std::string_view str);

std::string to_string(ReaderType reader);

struct ReaderConfig {
    /// It should be the path to a single-scene directory in the dataset
    std::filesystem::path scene_dir;
    /// The underlying reader implementation, print it with to_string(type)
    ReaderType type = ReaderType::Unknown;
    /// Depth values are multiplied by this factor before returning to the call site
    float depth_scale = 1.0f;
    /// Reader will consume frames at this FPS rate, or unlimited if set to 0
    float fps = 0.0f;
    /// Drop frames when processing cannot keep up with the target FPS
    bool drop_frames = false;
    /// Signal that this reader type is producing frames from a live sensor
    bool is_live = false;

    [[nodiscard]] static ReaderConfig from_yaml(const std::filesystem::path& file);
};

class Reader {
public:
    enum class ReadStatus : uint32_t {
        Consumed,
        Skipped,
        Failed,
    };

    explicit Reader(const ReaderConfig& config)
        : _config{ config }
        , _num_frames{ 0 }
        , _curr_frame{ 0 }
        , _last_frame_time{ std::chrono::steady_clock::now() }
    {
    }

    ReadStatus next(nvblox::ColorImage* color);
    ReadStatus next(nvblox::Transform* c2w);
    ReadStatus next(nvblox::ColorImage* color, nvblox::Transform* c2w);

    [[nodiscard]] uint32_t count() const { return _num_frames; }
    [[nodiscard]] bool exhausted() const { return _curr_frame >= _num_frames; }

    virtual ~Reader() = default;

protected:
    virtual ReadStatus read_color(nvblox::ColorImage* color) = 0;

    virtual ReadStatus read_c2w_color(nvblox::Transform* c2w) {
        *c2w = nvblox::Transform::Identity();
        return ReadStatus::Consumed;
    }

    // Advance to the next frame, wait if necessary
    // before loading (due to constraints like FPS)
    void wait_then_increment(ReadStatus status);

    ReaderConfig _config;
    uint32_t _num_frames; // initialize to 0 and children must re-assign it to the proper value
    uint32_t _curr_frame; // the frame index that is going to be read on the next next() call

private:
    std::chrono::steady_clock::time_point _last_frame_time;
};

constexpr Reader::ReadStatus operator+(const Reader::ReadStatus lhs, const Reader::ReadStatus rhs) {
    const auto _lhs = static_cast<uint32_t>(lhs);
    const auto _rhs = static_cast<uint32_t>(rhs);
    return static_cast<Reader::ReadStatus>(std::max(_lhs, _rhs));
}

class RgbDReader : public Reader {
public:
    explicit RgbDReader(const ReaderConfig& config) : Reader{ config } {}

    using Reader::next;
    ReadStatus next(nvblox::DepthImage* depth);
    ReadStatus next(nvblox::Transform* c2w_color, nvblox::Transform* c2w_depth);
    ReadStatus next(nvblox::ColorImage* color, nvblox::DepthImage* depth);
    ReadStatus next(nvblox::ColorImage* color, nvblox::DepthImage* depth, nvblox::Transform* c2w);
    ReadStatus next(
        nvblox::ColorImage* color, nvblox::Transform* c2w_color,
        nvblox::DepthImage* depth, nvblox::Transform* c2w_depth);

protected:
    virtual ReadStatus read_depth(nvblox::DepthImage* depth) = 0;

    virtual ReadStatus read_c2w_depth(nvblox::Transform* c2w) { return read_c2w_color(c2w); }
};

} // namespace gsblox
