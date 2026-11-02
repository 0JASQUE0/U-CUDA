#pragma once
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace platform {
inline std::string executable_dir() {
    std::vector<wchar_t> path(32768);
    const auto length = GetModuleFileNameW(nullptr, path.data(), (DWORD)path.size());
    if (!length || length >= path.size()) throw std::runtime_error("Cannot locate reference executable");
    return std::filesystem::path(std::wstring(path.data(), length)).parent_path().u8string();
}
}
