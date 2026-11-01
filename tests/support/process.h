#pragma once
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include "ucuda/platform/platform_posix.h"
#endif

namespace regression {
inline int process(const std::vector<std::string>& args, const std::filesystem::path& log) {
    if (args.empty()) throw std::runtime_error("Empty command");
#ifdef _WIN32
    auto wide = [](const std::string& s) {
        const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0);
        if (!n && !s.empty()) throw std::runtime_error("Invalid UTF-8 argument");
        std::wstring w(n, L'\0');
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), w.data(), n);
        return w;
    };
    const auto executable = wide(args[0]);
    auto quote = [](const std::wstring& value) {
        std::wstring out = L"\""; size_t slashes = 0;
        for (wchar_t c : value) {
            if (c == L'\\') { ++slashes; continue; }
            out.append(c == L'\"' ? slashes * 2 + 1 : slashes, L'\\');
            slashes = 0; out += c;
        }
        out.append(slashes * 2, L'\\'); out += L'\"'; return out;
    };
    std::vector<std::wstring> values;
    for (const auto& arg : args) values.push_back(quote(wide(arg)));
    std::vector<const wchar_t*> argv;
    for (const auto& arg : values) argv.push_back(arg.c_str());
    argv.push_back(nullptr);
    const int fd = _wopen(log.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd < 0) throw std::runtime_error("Cannot open process log");
    std::fflush(nullptr);
    const int out = _dup(1), err = _dup(2);
    if (out < 0 || err < 0) { _close(fd); if (out >= 0) _close(out); if (err >= 0) _close(err); throw std::runtime_error("Cannot duplicate output"); }
    _dup2(fd, 1); _dup2(fd, 2); _close(fd);
    const auto rc = _wspawnv(_P_WAIT, executable.c_str(), argv.data());
    _dup2(out, 1); _dup2(err, 2); _close(out); _close(err);
    if (rc < 0) throw std::runtime_error("Cannot launch " + args[0]);
    return (int)rc;
#else
    std::string output;
    const int rc = platform::run(args, output, {}, true);
    std::ofstream stream(log, std::ios::binary);
    if (!(stream << output)) throw std::runtime_error("Cannot write process log");
    return rc;
#endif
}
inline std::string text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}
