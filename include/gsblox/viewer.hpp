#pragma once

#include <filesystem>

namespace pangolin {
struct GlBufferCudaPtr;
struct HandlerBase3D;
class OpenGlRenderState;
}

namespace gsblox {

struct ViewerConfig {
    int width{ 1280 };
    int height{ 720 };
    int cam_focal_length_u{ 420 };
    int cam_focal_length_v{ 420 };
    int ui_width{ 180 };
    int init_vertex_buffer_capacity{ 200'000 };
    int init_index_buffer_capacity{ 500'000 };
    std::string window_title{ "gsblox" };
};

class Viewer final {
public:
    explicit Viewer(const ViewerConfig& config);

    [[nodiscard]] const pangolin::GlBufferCudaPtr& get_mesh_vbo() const noexcept { return *_mesh_vbo; }
    [[nodiscard]] const pangolin::GlBufferCudaPtr& get_mesh_nbo() const noexcept { return *_mesh_nbo; }
    [[nodiscard]] const pangolin::GlBufferCudaPtr& get_mesh_cbo() const noexcept { return *_mesh_cbo; }
    [[nodiscard]] const pangolin::GlBufferCudaPtr& get_mesh_ibo() const noexcept { return *_mesh_ibo; }

    void update_vbo(int n_vertices) const;
    void update_nbo(int n_vertices) const;
    void update_cbo(int n_vertices) const;
    void update_ibo(int n_indices) const;

    void draw_mesh(int n_indices, bool draw_color = false, bool draw_normal = false) const;

    static bool should_quit();
    static void finish_draw();

    ~Viewer() noexcept;

private:
    // UI
    std::string _window_title;
    pangolin::OpenGlRenderState* _camera_state;
    pangolin::HandlerBase3D* _camera_handler;

    // Mesh resources
    pangolin::GlBufferCudaPtr* _mesh_vbo;
    pangolin::GlBufferCudaPtr* _mesh_nbo;
    pangolin::GlBufferCudaPtr* _mesh_cbo;
    pangolin::GlBufferCudaPtr* _mesh_ibo;
};

namespace viewer {

[[nodiscard]] std::unique_ptr<Viewer> create(const std::filesystem::path& config_file);

} // namespace viewer
} // namespace gsblox
