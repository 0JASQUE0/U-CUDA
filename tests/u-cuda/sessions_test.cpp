// JSON библиотеки (этап 11): каждый library/*/system.json и каждая сессия library/*/sessions/*.json
// разбирается своим разборщиком (как при загрузке в GUI, по имени файла: _last_<вкладка>, остальное —
// Analysis) и проходит круг "разбор -> запись -> разбор -> запись": вторая запись равна первой
// (запись идемпотентна, ни одно поле не теряется и не искажается при повторном сохранении).
#include "test_util.h"
#include "session_io.h"
#include "system_library.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <functional>
#include <algorithm>

namespace fs = std::filesystem;

namespace {

std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF) s.erase(0, 3);
    return s;
}

// Разбор + запись для типа сессии T.
template <class T>
std::function<bool(const std::string&, std::string&)> rt(bool (*from)(const std::string&, T&), std::string (*to)(const T&)) {
    return [from, to](const std::string& j, std::string& out) {
        T s;
        if (!from(j, s)) return false;
        out = to(s);
        return true;
    };
}

}  // namespace

int main() {
    return ut::run_main("sessions", []() -> int {
        const fs::path lib = fs::u8path(exe_dir()) / ".." / ".." / ".." / "U-CUDA" / "library";
        std::map<std::string, std::function<bool(const std::string&, std::string&)>> by_name = {
            { "_last_parametric", rt(session_from_json_parametric, session_to_json_parametric) },
            { "_last_lle", rt(session_from_json_lle, session_to_json_lle) },
            { "_last_ls", rt(session_from_json_ls, session_to_json_ls) },
            { "_last_metrics", rt(session_from_json_metrics, session_to_json_metrics) },
            { "_last_dft1d", rt(session_from_json_dft1d, session_to_json_dft1d) },
            { "_last_basins", rt(session_from_json_basins, session_to_json_basins) },
            { "_last_fastsync", rt(session_from_json_fastsync, session_to_json_fastsync) },
            { "_last_network", rt(session_from_json_network, session_to_json_network) },
            { "_last_order", rt(session_from_json_order, session_to_json_order) },
            { "_last_custom", rt(session_from_json_custom, session_to_json_custom) },
            { "_last_parametric_windows", rt(session_from_json_parametric_windows, session_to_json_parametric_windows) },
            { "_last_order_windows", rt(session_from_json_order_windows, session_to_json_order_windows) },
            { "_last_dft1d_windows", rt(session_from_json_dft1d_windows, session_to_json_dft1d_windows) },
        };
        const auto phase = rt(session_from_json, session_to_json);
        int nsys = 0, nses = 0;
        std::vector<std::string> bad;
        std::vector<fs::path> dirs;
        for (auto& e : fs::directory_iterator(lib)) if (e.is_directory()) dirs.push_back(e.path());
        std::sort(dirs.begin(), dirs.end());
        for (const auto& d : dirs) {
            const fs::path sj = d / "system.json";
            if (fs::exists(sj)) {
                ++nsys;
                try {
                    const SystemRecord r1 = record_from_json(read(sj));
                    const std::string j1 = record_to_json(r1);
                    const std::string j2 = record_to_json(record_from_json(j1));
                    if (j1 != j2) bad.push_back(sj.u8string() + ": system record round trip changes the JSON");
                } catch (const std::exception& e) { bad.push_back(sj.u8string() + ": " + e.what()); }
            }
            const fs::path sd = d / "sessions";
            if (!fs::exists(sd)) continue;
            for (const auto& e : fs::directory_iterator(sd)) {
                if (e.path().extension() != ".json") continue;
                ++nses;
                const std::string stem = e.path().stem().u8string();
                auto it = by_name.find(stem);
                const auto& fn = it != by_name.end() ? it->second : phase;
                std::string j1, j2;
                try {
                    if (!fn(read(e.path()), j1)) { bad.push_back(e.path().u8string() + ": not parsed"); continue; }
                    if (!fn(j1, j2)) { bad.push_back(e.path().u8string() + ": own output not parsed"); continue; }
                    if (j1 != j2) {
                        size_t k = 0;
                        while (k < j1.size() && k < j2.size() && j1[k] == j2[k]) ++k;
                        bad.push_back(e.path().u8string() + ": round trip changes the JSON near '"
                                      + j1.substr(k > 40 ? k - 40 : 0, 80) + "'");
                    }
                } catch (const std::exception& ex) { bad.push_back(e.path().u8string() + ": " + ex.what()); }
            }
        }
        std::printf("  %d system records, %d sessions, %zu problems\n", nsys, nses, bad.size());
        for (const auto& b : bad) std::printf("  BAD %s\n", b.c_str());
        ut::require(bad.empty(), std::to_string(bad.size()) + " JSON files fail the round trip");
        return 0;
    });
}
