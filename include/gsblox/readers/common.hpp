#pragma once

#include <nvblox/sensors/image.h>

#include <filesystem>

namespace gsblox {

enum class ReaderType {
    Replica,
    TumRgbD,
    Unknown,
};

namespace utils {

[[nodiscard]] ReaderType get_reader_type(std::string_view key) noexcept;

[[nodiscard]] std::string to_string(ReaderType type) noexcept;

} // namespace utils

struct ReaderConfig {
    /// It should be the path to a single-scene directory in the dataset
    std::filesystem::path scene_dir;
    /// The underlying reader implementation, print it with to_string(type)
    ReaderType reader_type = ReaderType::Unknown;
    /// Depth values are multiplied by this factor before returning to the call site
    float depth_multiplier = 1.0f;
    /// Reader will consume frames at this FPS rate, or unlimited if set to 0
    float fps = 0.0f;
    /// Drop frames when processing cannot keep up with the target FPS
    bool drop_frames = false;
    /// Signal that this reader type is producing frames from a live sensor
    bool is_live = false;

    [[nodiscard]] bool valid() const;

    [[nodiscard]] static ReaderConfig from_yaml(const std::filesystem::path& file);
};

class Reader {
public:
    enum class ReadStatus : uint32_t {
        Consumed,
        Skipped,
        Failed,
    };

    explicit Reader(ReaderConfig config);

    // RGB datasets
    ReadStatus next(nvblox::ColorImage* color, uint32_t* sensor_id = nullptr);
    ReadStatus next(nvblox::Transform* c2w, uint32_t* sensor_id = nullptr);
    ReadStatus next(nvblox::ColorImage* color, nvblox::Transform* c2w, uint32_t* sensor_id = nullptr);

    // RGB-D datasets
    ReadStatus next(nvblox::DepthImage* depth, uint32_t* sensor_id = nullptr);
    ReadStatus next(nvblox::Transform* c2w_color, nvblox::Transform* c2w_depth, uint32_t* sensor_id = nullptr);
    ReadStatus next(nvblox::ColorImage* color, nvblox::DepthImage* depth, uint32_t* sensor_id = nullptr);
    ReadStatus next(nvblox::ColorImage* color, nvblox::DepthImage* depth, nvblox::Transform* c2w, uint32_t* sensor_id = nullptr);
    ReadStatus next(
        nvblox::ColorImage* color, nvblox::Transform* c2w_color,
        nvblox::DepthImage* depth, nvblox::Transform* c2w_depth,
        uint32_t* sensor_id = nullptr);

    [[nodiscard]] uint32_t count() const noexcept { return _num_frames; }
    [[nodiscard]] bool exhausted() const noexcept { return _curr_frame >= _num_frames; }

    virtual ~Reader() noexcept = default;

protected:
    virtual ReadStatus read_color(nvblox::ColorImage* color);
    virtual ReadStatus read_depth(nvblox::DepthImage* depth);

    virtual ReadStatus read_c2w_color(nvblox::Transform* c2w);
    virtual ReadStatus read_c2w_depth(nvblox::Transform* c2w);

    virtual ReadStatus read_sensor(uint32_t* sensor_id);

    ReaderConfig _config;
    uint32_t _num_frames; // initialize to 0 and children must re-assign it to the proper value
    uint32_t _curr_frame; // the frame index that is going to be read on the next next() call

private:
    // Advance to the next frame, wait before loading
    // if necessary (due to constraints like FPS)
    void wait_then_increment(ReadStatus status);
    std::chrono::steady_clock::time_point _last_frame_time;
};

constexpr Reader::ReadStatus operator+(const Reader::ReadStatus lhs, const Reader::ReadStatus rhs) noexcept {
    const auto _lhs = static_cast<uint32_t>(lhs);
    const auto _rhs = static_cast<uint32_t>(rhs);
    return static_cast<Reader::ReadStatus>(std::max(_lhs, _rhs));
}

} // namespace gsblox
