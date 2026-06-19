#include "app_config.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace {

// Находит начало значения после "key": в JSON. Пропускает пробелы. Возвращает
// std::string::npos если ключ не найден или после двоеточия ничего нет.
size_t find_value_pos(const std::string& s, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = s.find(needle);
    if (pos == std::string::npos) return std::string::npos;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return std::string::npos;
    ++pos;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
    return (pos < s.size()) ? pos : std::string::npos;
}

bool parse_float_field(const std::string& s, const std::string& key, float& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + pos, &end);
    if (end == s.c_str() + pos) return false;
    out = (float)v;
    return true;
}

bool parse_bool_field(const std::string& s, const std::string& key, bool& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    if (s.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (s.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

} // namespace

bool load_app_config(const std::string& dir, AppConfig& out) {
    std::ifstream f(dir + "_app_config.json", std::ios::binary);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    std::string body = ss.str();
    float fv = 0.0f;
    if (parse_float_field(body, "ui_scale_override", fv))
        out.ui_scale_override = fv;
    bool bv = false;
    if (parse_bool_field(body, "use_builtin_font", bv))
        out.use_builtin_font = bv;
    return true;
}

bool save_app_config(const std::string& dir, const AppConfig& cfg) {
    std::string final_path = dir + "_app_config.json";
    std::string tmp_path   = final_path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << "{\n";
        f << "  \"ui_scale_override\": " << cfg.ui_scale_override << ",\n";
        f << "  \"use_builtin_font\": "  << (cfg.use_builtin_font ? "true" : "false") << "\n";
        f << "}\n";
        if (!f) return false;
    }
    // std::remove + std::rename: на Windows std::rename падает если target существует.
    std::remove(final_path.c_str());
    return std::rename(tmp_path.c_str(), final_path.c_str()) == 0;
}
