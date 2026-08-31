#include "app_config.h"
#include <fstream>
#include <iomanip>
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

// Двойник parse_float_field без сужения до float: peak-пороги живут в диапазоне
// 1e-14 .. -1e25, и float потерял бы и точность, и сам порядок величины.
bool parse_double_field(const std::string& s, const std::string& key, double& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    char* end = nullptr;
    double v = std::strtod(s.c_str() + pos, &end);
    if (end == s.c_str() + pos) return false;
    out = v;
    return true;
}

bool parse_bool_field(const std::string& s, const std::string& key, bool& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    if (s.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (s.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

bool parse_int_field(const std::string& s, const std::string& key, int& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    char* end = nullptr;
    long v = std::strtol(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) return false;
    out = (int)v;
    return true;
}

// Минимальный JSON-строковый парсер: ожидает открывающую кавычку сразу после
// двоеточия, читает до закрывающей, поддерживает \" \\ \/ \n \r \t и \uXXXX
// (последнее — только для ASCII: 0x00..0x7F, чего достаточно для имён
// систем в library). Сложнее не тянем — конфиг пишем сами, экраним только то,
// что реально может встретиться в имени системы.
bool parse_string_field(const std::string& s, const std::string& key, std::string& out) {
    size_t pos = find_value_pos(s, key);
    if (pos == std::string::npos) return false;
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    std::string acc;
    while (pos < s.size() && s[pos] != '"') {
        char c = s[pos++];
        if (c == '\\' && pos < s.size()) {
            char e = s[pos++];
            switch (e) {
                case '"':  acc.push_back('"');  break;
                case '\\': acc.push_back('\\'); break;
                case '/':  acc.push_back('/');  break;
                case 'n':  acc.push_back('\n'); break;
                case 'r':  acc.push_back('\r'); break;
                case 't':  acc.push_back('\t'); break;
                case 'b':  acc.push_back('\b'); break;
                case 'f':  acc.push_back('\f'); break;
                case 'u': {
                    if (pos + 4 > s.size()) return false;
                    int code = 0;
                    for (int i = 0; i < 4; ++i) {
                        char h = s[pos + i];
                        int d = (h >= '0' && h <= '9') ? h - '0'
                              : (h >= 'a' && h <= 'f') ? h - 'a' + 10
                              : (h >= 'A' && h <= 'F') ? h - 'A' + 10
                              : -1;
                        if (d < 0) return false;
                        code = code * 16 + d;
                    }
                    pos += 4;
                    if (code < 0x80) acc.push_back((char)code);
                    // Non-ASCII \uXXXX — silently skipped; system names are ASCII.
                    break;
                }
                default: return false;
            }
        } else {
            acc.push_back(c);
        }
    }
    if (pos >= s.size()) return false; // unterminated string
    out = std::move(acc);
    return true;
}

// Экранирует строку для записи в JSON (кавычки, слэши, управляющие символы).
std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)(unsigned char)c);
                    o += buf;
                } else {
                    o.push_back(c);
                }
        }
    }
    return o;
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
    int iv = 0;
    if (parse_int_field(body, "heatmap_colormap", iv))
        out.heatmap_colormap = iv;
    int bcm = 0;
    if (parse_int_field(body, "basins_colormap", bcm))
        out.basins_colormap = bcm;
    if (parse_int_field(body, "basins_avgpk_colormap", bcm))
        out.basins_avgpk_colormap = bcm;
    if (parse_int_field(body, "basins_avgint_colormap", bcm))
        out.basins_avgint_colormap = bcm;
    if (parse_int_field(body, "basins_states_colormap", bcm))
        out.basins_states_colormap = bcm;
    int tp = 0;
    parse_string_field(body, "slancm_enabled", out.slancm_enabled);
    if (parse_int_field(body, "tick_precision", tp))
        out.tick_precision = tp;
    bool dark = true;
    if (parse_bool_field(body, "dark_theme", dark))
        out.dark_theme = dark;
    int lam = 0;
    if (parse_int_field(body, "last_app_mode", lam))
        out.last_app_mode = lam;
    std::string lsn;
    if (parse_string_field(body, "last_system_name", lsn))
        out.last_system_name = std::move(lsn);

    // Peak-knobs: каждое поле опционально — отсутствие оставляет дефолт из
    // configCUDA.h, поэтому старый _app_config.json читается без миграции.
    parse_bool_field  (body, "peak_do_calculate",    out.peak.do_calculate_peaks);
    parse_bool_field  (body, "peak_do_interpolate",  out.peak.do_interpolate_peaks);
    parse_double_field(body, "peak_eps_fixed_point", out.peak.eps_fixed_point);
    parse_double_field(body, "peak_eps_peak_delta",  out.peak.eps_peak_delta);
    parse_double_field(body, "peak_eps_interpeak",   out.peak.eps_interPeak_delta);
    parse_double_field(body, "peak_threshold",       out.peak.peak_threshold);
    parse_int_field   (body, "peak_max_amount",      out.peak.max_amount_of_peaks);

    // Отсутствие ключа оставляет дефолт true (= дефолт NVRTC), поэтому конфиг,
    // записанный до появления настройки, читается без изменения поведения.
    parse_bool_field  (body, "nvrtc_fmad",           out.nvrtc_fmad);
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
        f << "  \"use_builtin_font\": "  << (cfg.use_builtin_font ? "true" : "false") << ",\n";
        f << "  \"heatmap_colormap\": "  << cfg.heatmap_colormap << ",\n";
        f << "  \"basins_colormap\": "        << cfg.basins_colormap << ",\n";
        f << "  \"basins_avgpk_colormap\": "  << cfg.basins_avgpk_colormap << ",\n";
        f << "  \"basins_avgint_colormap\": " << cfg.basins_avgint_colormap << ",\n";
        f << "  \"basins_states_colormap\": " << cfg.basins_states_colormap << ",\n";
        // Маска из '0'/'1' — экранировать нечего, но json_escape держит формат
        // единообразным с last_system_name.
        f << "  \"slancm_enabled\": \""       << json_escape(cfg.slancm_enabled) << "\",\n";
        f << "  \"tick_precision\": "         << cfg.tick_precision << ",\n";
        f << "  \"dark_theme\": "             << (cfg.dark_theme ? "true" : "false") << ",\n";
        f << "  \"last_app_mode\": "          << cfg.last_app_mode << ",\n";
        f << "  \"last_system_name\": \""     << json_escape(cfg.last_system_name) << "\",\n";
        // Пороги пишем с максимальной точностью double: 1e-14 и -1e25 должны
        // пережить round-trip без изменения значения.
        f << std::setprecision(17);
        f << "  \"peak_do_calculate\": "    << (cfg.peak.do_calculate_peaks   ? "true" : "false") << ",\n";
        f << "  \"peak_do_interpolate\": "  << (cfg.peak.do_interpolate_peaks ? "true" : "false") << ",\n";
        f << "  \"peak_eps_fixed_point\": " << cfg.peak.eps_fixed_point     << ",\n";
        f << "  \"peak_eps_peak_delta\": "  << cfg.peak.eps_peak_delta      << ",\n";
        f << "  \"peak_eps_interpeak\": "   << cfg.peak.eps_interPeak_delta << ",\n";
        f << "  \"peak_threshold\": "       << cfg.peak.peak_threshold      << ",\n";
        f << "  \"peak_max_amount\": "      << cfg.peak.max_amount_of_peaks << ",\n";
        f << "  \"nvrtc_fmad\": "           << (cfg.nvrtc_fmad ? "true" : "false") << "\n";
        f << "}\n";
        if (!f) return false;
    }
    // std::remove + std::rename: на Windows std::rename падает если target существует.
    std::remove(final_path.c_str());
    return std::rename(tmp_path.c_str(), final_path.c_str()) == 0;
}
