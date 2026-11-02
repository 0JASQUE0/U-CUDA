#pragma once
#include <cstdlib>
#include <filesystem>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace platform {
inline std::string executable_dir() {
#ifdef _WIN32
    char path[32768];
    const auto n = GetModuleFileNameA(nullptr, path, sizeof(path));
    if (!n || n == sizeof(path)) return ".";
#else
    char path[4096];
    const auto n = readlink("/proc/self/exe", path, sizeof(path));
    if (n <= 0 || n == sizeof(path)) return ".";
#endif
    return std::filesystem::path(std::string(path, n)).parent_path().string();
}
inline std::string cuda_include_dir() {
    if (const char* p = std::getenv("CUDA_PATH"))
        if (*p) return (std::filesystem::path(p) / "include").string();
#ifdef U_CUDA_TOOLKIT_INCLUDE
    return U_CUDA_TOOLKIT_INCLUDE;
#else
    return {};
#endif
}
}
