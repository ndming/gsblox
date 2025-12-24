#include "gsblox/utils/path.hpp"

#include <spdlog/spdlog.h>

std::filesystem::path gsblox::utils::normalize(const std::filesystem::path& path) {
    auto n_path = path;
    if (n_path.is_relative()) {
        n_path = std::filesystem::current_path() / n_path;
    }
    return std::filesystem::weakly_canonical(n_path);
}

std::filesystem::path gsblox::utils::make_path(const std::string& str_path) {
    if (str_path.empty()) [[unlikely]] {
        return {};
    }
    if (str_path[0] != '~') {
        return str_path;
    }
#if defined(_WIN32)
    const auto home = getenv("USERPROFILE");
#else
    const auto home = std::getenv("HOME");
#endif
    if (!home) [[unlikely]] {
        spdlog::warn("Could NOT expand ~ to home directory");
        return str_path;
    }

    // The path is only ~
    if (str_path.size() == 1) [[unlikely]] {
        return home;
    }

    // Handles ~/file or ~\file (Windows)
    if (str_path[1] == '/' || str_path[1] == '\\') [[likely]] {
        return std::filesystem::path(home) / str_path.substr(2);
    }

    // "~username" not supported
    return str_path;
}

std::filesystem::path gsblox::utils::make_norm(const std::string& str_path) {
    return normalize(make_path(str_path));
}
