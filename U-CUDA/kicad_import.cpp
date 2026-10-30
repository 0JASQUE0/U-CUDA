#include "kicad_import.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

// Старшая версия среди <root>\<версия>\bin\kicad-cli.exe ("9.0" > "8.0" > "7.0").
std::string newest_under(const fs::path& root) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) return {};
    std::vector<std::pair<std::vector<int>, fs::path>> found;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        const fs::path cli = e.path() / "bin" / "kicad-cli.exe";
        if (!fs::exists(cli, ec)) continue;
        std::vector<int> ver;
        int cur = -1;
        for (wchar_t c : e.path().filename().wstring()) {
            if (c >= L'0' && c <= L'9') cur = (cur < 0 ? 0 : cur * 10) + (c - L'0');
            else if (cur >= 0) { ver.push_back(cur); cur = -1; }
        }
        if (cur >= 0) ver.push_back(cur);
        found.push_back({ ver, cli });
    }
    if (found.empty()) return {};
    std::sort(found.begin(), found.end());
    return narrow(found.back().second.wstring());
}

std::wstring env(const wchar_t* name) {
    wchar_t buf[32768];
    const DWORD n = GetEnvironmentVariableW(name, buf, 32768);
    return (n > 0 && n < 32768) ? std::wstring(buf, n) : std::wstring();
}

std::string lower_ext(const std::string& path) {
    std::string e = fs::u8path(path).extension().u8string();
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    return e;
}

}  // namespace

std::string kicad_find_cli() {
    std::error_code ec;
    const std::wstring user = env(L"UCUDA_KICAD_CLI");
    if (!user.empty() && fs::exists(user, ec)) return narrow(user);
    wchar_t buf[MAX_PATH];
    if (SearchPathW(nullptr, L"kicad-cli.exe", nullptr, MAX_PATH, buf, nullptr)) return narrow(buf);
    for (const wchar_t* var : { L"ProgramFiles", L"ProgramW6432", L"LOCALAPPDATA" }) {
        const std::wstring base = env(var);
        if (base.empty()) continue;
        fs::path root = fs::path(base);
        root /= (std::wstring(var) == L"LOCALAPPDATA") ? fs::path(L"Programs") / L"KiCad" : fs::path(L"KiCad");
        const std::string cli = newest_under(root);
        if (!cli.empty()) return cli;
    }
    return {};
}

bool circuit_source_is_kicad(const std::string& path) {
    return lower_ext(path) == ".kicad_sch";
}

long long circuit_source_mtime(const std::string& path) {
    std::error_code ec;
    const auto t = fs::last_write_time(fs::u8path(path), ec);
    if (ec) return -1;
    return (long long)t.time_since_epoch().count();
}

static bool read_file(const fs::path& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    if (out.size() >= 3 && (unsigned char)out[0] == 0xEF && (unsigned char)out[1] == 0xBB && (unsigned char)out[2] == 0xBF)
        out.erase(0, 3);
    return true;
}

bool circuit_source_load(const std::string& path, std::string& netlist,
                         std::string* log, std::string* err) {
    const fs::path src = fs::u8path(path);
    std::error_code ec;
    if (!fs::exists(src, ec)) { if (err) *err = "file not found: " + path; return false; }
    if (!circuit_source_is_kicad(path)) {
        if (!read_file(src, netlist)) { if (err) *err = "cannot read " + path; return false; }
        return true;
    }

    const std::string cli = kicad_find_cli();
    if (cli.empty()) {
        if (err) *err = "kicad-cli.exe not found (KiCad 7 or newer). Install KiCad, add its bin folder to PATH "
                        "or set UCUDA_KICAD_CLI; or export the netlist in KiCad (Simulator -> Export SPICE netlist) "
                        "and open the .cir file.";
        return false;
    }
    static std::atomic<int> seq{ 0 };
    const fs::path out = fs::temp_directory_path(ec) /
        (L"ucuda_kicad_" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(++seq) + L".cir");
    std::wstring cmd = L"\"" + widen(cli) + L"\" sch export netlist --format spice -o \"" + out.wstring()
                     + L"\" \"" + src.wstring() + L"\"";

    SECURITY_ATTRIBUTES sa{ sizeof(sa), nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { if (err) *err = "CreatePipe failed"; return false; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> line(cmd.begin(), cmd.end());
    line.push_back(L'\0');
    const BOOL ok = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                   nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        CloseHandle(rd);
        if (err) *err = "cannot start " + cli;
        return false;
    }
    std::string text;
    char buf[4096];
    DWORD n = 0;
    while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0) text.append(buf, n);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, 120000);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (log) *log = text;
    const bool have = read_file(out, netlist);
    fs::remove(out, ec);
    if (code != 0 || !have) {
        if (err) *err = "kicad-cli failed (exit code " + std::to_string((int)code) + ")" + (text.empty() ? "" : ": " + text);
        return false;
    }
    return true;
}
