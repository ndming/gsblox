#include "gsblox/viewer.hpp"
#include "gsblox/utils/config.hpp"

#include <nvblox/core/color.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

gsblox::Viewer::Viewer(const ViewerConfig& config) : _window_title{ config.window_title } {
    // Set up pangolin
    pangolin::CreateWindowAndBind(_window_title, config.width, config.height);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Define Camera Render Object (for view / scene browsing)
    _camera_state = new pangolin::OpenGlRenderState{
        pangolin::ProjectionMatrix(
            config.width, config.height,
            config.cam_focal_length_u, config.cam_focal_length_v,
            config.width / 2.0f, config.height / 2.0f,
            0.1, 100.0),
        pangolin::ModelViewLookAt(4, -4, -4, 0, 0, 0, pangolin::AxisNegY)
    };

    // Add named OpenGL viewport to window and provide 3D Handler
    _camera_handler = new pangolin::HandlerBase3D{ *_camera_state };
    pangolin::Display("camera")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(config.ui_width), 1.0, -static_cast<float>(config.width) / config.height)
        .SetHandler(_camera_handler);

    // Add named Panel and bind to variables beginning 'ui'
    // A Panel is just a View with a default layout and input handling
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(config.ui_width));

    // Create buffers for rendering the fused mesh with OpenGL
    _mesh_vbo = new pangolin::GlBufferCudaPtr{
        pangolin::GlArrayBuffer, static_cast<uint32_t>(config.init_vertex_buffer_capacity),
        GL_FLOAT, 3, // each vertex is a float3
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    };
    _mesh_nbo = new pangolin::GlBufferCudaPtr{
        pangolin::GlArrayBuffer, static_cast<uint32_t>(config.init_vertex_buffer_capacity),
        GL_FLOAT, 3, // each normal is a float3
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    };
    _mesh_cbo = new pangolin::GlBufferCudaPtr{
        pangolin::GlArrayBuffer, static_cast<uint32_t>(config.init_vertex_buffer_capacity),
        GL_UNSIGNED_BYTE, nvblox::kRgbNumElements,
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    };
    _mesh_ibo = new pangolin::GlBufferCudaPtr{
        pangolin::GlElementArrayBuffer, static_cast<uint32_t>(config.init_index_buffer_capacity),
        GL_UNSIGNED_INT, 1, // each index is an unsigned int
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    };
}

void gsblox::Viewer::update_vbo(const int n_vertices) const {
    if (n_vertices <= _mesh_vbo->num_elements) [[likely]] {
        return;
    }
    spdlog::info("Growing vertex buffer to make room for {} vertices", n_vertices);
    auto n_elems = _mesh_vbo->num_elements;
    while (n_vertices > n_elems) {
        n_elems *= 2;
    }
    _mesh_vbo->Reinitialise(pangolin::GlArrayBuffer, n_elems, GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
}

void gsblox::Viewer::update_nbo(const int n_vertices) const {
    if (n_vertices <= _mesh_nbo->num_elements) [[likely]] {
        return;
    }
    spdlog::info("Growing normal buffer to make room for {} vertices", n_vertices);
    auto n_elems = _mesh_nbo->num_elements;
    while (n_vertices > n_elems) {
        n_elems *= 2;
    }
    _mesh_nbo->Reinitialise(pangolin::GlArrayBuffer, n_elems, GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
}

void gsblox::Viewer::update_cbo(const int n_vertices) const {
    if (n_vertices <= _mesh_cbo->num_elements) [[likely]] {
        return;
    }
    spdlog::info("Growing color buffer to make room for {} vertices", n_vertices);
    auto n_elems = _mesh_cbo->num_elements;
    while (n_vertices > n_elems) {
        n_elems *= 2;
    }
    _mesh_cbo->Reinitialise(
        pangolin::GlArrayBuffer, n_elems, GL_UNSIGNED_BYTE, nvblox::kRgbNumElements,
        cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW
    );
}

void gsblox::Viewer::update_ibo(const int n_indices) const {
    if (n_indices <= _mesh_ibo->num_elements) [[likely]] {
        return;
    }
    spdlog::info("Growing index buffer to make room for {} indices", n_indices);
    auto n_elems = _mesh_ibo->num_elements;
    while (n_indices > n_elems) {
        n_elems *= 2;
    }
    _mesh_ibo->Reinitialise(
        pangolin::GlElementArrayBuffer, n_elems, GL_UNSIGNED_INT, 1, cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
}

void gsblox::Viewer::draw_mesh(const int n_indices, const bool draw_color, const bool draw_normal) const {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Which camera to use for this pass
    pangolin::Display("camera").Activate(*_camera_state);
    glColor3f(1.0, 1.0, 1.0);

    if (draw_color) {
        _mesh_cbo->Bind();
        glColorPointer(_mesh_cbo->count_per_element, _mesh_cbo->datatype, 0, nullptr);
        glEnableClientState(GL_COLOR_ARRAY);
    }

    if (draw_normal) {
        _mesh_nbo->Bind();
        glNormalPointer(_mesh_nbo->datatype, static_cast<GLsizei>(_mesh_nbo->count_per_element * pangolin::GlDataTypeBytes(_mesh_nbo->datatype)), nullptr);
        glEnableClientState(GL_NORMAL_ARRAY);
    }

    _mesh_vbo->Bind();
    glVertexPointer(_mesh_vbo->count_per_element, _mesh_vbo->datatype, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    _mesh_ibo->Bind();
    glDrawElements(GL_TRIANGLES, n_indices, _mesh_ibo->datatype, nullptr);
    _mesh_ibo->Unbind();

    if (draw_color) {
        glDisableClientState(GL_COLOR_ARRAY);
        _mesh_cbo->Unbind();
    }

    if (draw_normal) {
        glDisableClientState(GL_NORMAL_ARRAY);
        _mesh_nbo->Unbind();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    _mesh_vbo->Unbind();
}

bool gsblox::Viewer::should_quit() {
    return pangolin::ShouldQuit();
}

void gsblox::Viewer::finish_draw() {
    pangolin::FinishFrame();
}

gsblox::Viewer::~Viewer() noexcept {
    delete _mesh_ibo;
    delete _mesh_cbo;
    delete _mesh_nbo;
    delete _mesh_vbo;

    delete _camera_handler;
    delete _camera_state;

    pangolin::DestroyWindow(_window_title);
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

std::unique_ptr<gsblox::Viewer> gsblox::viewer::create(const std::filesystem::path& config_file) {
    // const auto root = utils::load_yaml(config_file);
    // if (!root) {
    //     spdlog::error("Could NOT create Viewer: empty YAML root");
    //     return nullptr;
    // }
    //
    // // Default viewer config
    // auto viewer_config = ViewerConfig{};
    //
    // const auto viewer_node = root["viewer"];
    // if (!viewer_node) {
    //     // No viewer setting found, use default
    //     return std::make_unique<Viewer>(viewer_config);
    // }
    return std::make_unique<Viewer>(ViewerConfig{});
}
