#include "session_io.h"
#include <sstream>
#include <cctype>
#include <stdexcept>

// ---------- сериализация ----------
namespace {

    std::string esc(const std::string& s) {
        std::string o; o.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
            }
        }
        return o;
    }

    void jstr(std::ostringstream& o, const std::string& v) { o << '"' << esc(v) << '"'; }

    void jmap(std::ostringstream& o, const std::map<std::string, std::string>& m) {
        o << "{"; bool first = true;
        for (auto& p : m) { if (!first)o << ","; o << '"' << esc(p.first) << "\":\"" << esc(p.second) << '"'; first = false; }
        o << "}";
    }

    // ---------- мини-парсер JSON ----------
    struct JP {
        std::string s; size_t i = 0;
        JP(std::string src) :s(std::move(src)) {}
        void ws() { while (i < s.size() && std::isspace((unsigned char)s[i]))++i; }
        char peek() { ws(); return i < s.size() ? s[i] : '\0'; }
        void expect(char c) { ws(); if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("expected ") + c); ++i; }
        bool opt(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }

        std::string str() {
            ws(); if (s[i] != '"') throw std::runtime_error("expected string"); ++i;
            std::string o;
            while (i < s.size() && s[i] != '"') {
                char c = s[i++];
                if (c == '\\' && i < s.size()) {
                    char e = s[i++];
                    switch (e) {
                    case 'n':o += '\n'; break; case 'r':o += '\r'; break; case 't':o += '\t'; break;
                    case '"':o += '"'; break; case '\\':o += '\\'; break; case '/':o += '/'; break;
                    case 'u': if (i + 4 <= s.size()) { int cd = std::stoi(s.substr(i, 4), nullptr, 16); i += 4; if (cd < 0x80)o += (char)cd; } break;
                    default:o += e;
                    }
                }
                else o += c;
            }
            if (i < s.size())++i; return o;
        }
        bool boolean() { ws(); if (s.compare(i, 4, "true") == 0) { i += 4; return true; } if (s.compare(i, 5, "false") == 0) { i += 5; return false; } throw std::runtime_error("expected bool"); }
        // читает число (целое/дробное) и возвращает как строку
        std::string str_or_num() {
            ws(); size_t st = i; if (i < s.size() && (s[i] == '-' || s[i] == '+'))++i;
            while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+'))++i;
            return s.substr(st, i - st);
        }
        std::map<std::string, std::string> map_ss() {
            std::map<std::string, std::string> m; expect('{'); ws();
            if (opt('}')) return m;
            while (true) { std::string k = str(); expect(':'); std::string v = str(); m[k] = v; ws(); if (opt(','))continue; expect('}'); break; }
            return m;
        }
        void skip_value() { // на случай неизвестных полей
            ws(); char c = peek();
            if (c == '"') { str(); }
            else if (c == '{') { expect('{'); if (!opt('}')) { while (true) { str(); expect(':'); skip_value(); if (opt(','))continue; expect('}'); break; } } }
            else if (c == '[') { expect('['); if (!opt(']')) { while (true) { skip_value(); if (opt(','))continue; expect(']'); break; } } }
            else if (c == 't' || c == 'f') { boolean(); }
            else { while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i; }
        }
    };

} // namespace

std::string session_to_json(const PhaseAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"step_h\":"; jstr(o, s.step_h); o << ",\n";
    o << "  \"sim_time\":"; jstr(o, s.sim_time); o << ",\n";
    o << "  \"skip_time\":"; jstr(o, s.skip_time); o << ",\n";
    o << "  \"scheme\":"; jstr(o, s.scheme); o << ",\n";
    o << "  \"symmetry_s\":"; jstr(o, s.symmetry_s); o << ",\n";
    o << "  \"decimation\":"; jstr(o, s.decimation); o << ",\n";
    o << "  \"auto_recompute\":" << (s.auto_recompute ? "true" : "false") << ",\n";
    o << "  \"legend_show_ic\":" << (s.legend_show_ic ? "true" : "false") << ",\n";
    o << "  \"use_gpu\":" << (s.use_gpu ? "true" : "false") << ",\n";
    o << "  \"param_values\":"; jmap(o, s.param_values); o << ",\n";
    // ic_sets: [ {label, visible, values{...}} ]
    o << "  \"ic_sets\":[";
    for (size_t k = 0; k < s.ic_sets.size(); ++k) {
        if (k)o << ","; const auto& ic = s.ic_sets[k];
        o << "{\"label\":"; jstr(o, ic.label);
        o << ",\"visible\":" << (ic.visible ? "true" : "false");
        o << ",\"values\":"; jmap(o, ic.values); o << "}";
    }
    o << "],\n";
    // projections: [ {label,type,ax,ay,az,show_var[...]} ]
    o << "  \"projections\":[";
    for (size_t k = 0; k < s.projections.size(); ++k) {
        if (k)o << ","; const auto& p = s.projections[k];
        o << "{\"label\":"; jstr(o, p.label);
        o << ",\"type\":" << (int)p.type;
        o << ",\"ax\":" << p.axis_x << ",\"ay\":" << p.axis_y << ",\"az\":" << p.axis_z;
        o << ",\"cls\":" << (p.custom_line_style ? "true" : "false");
        o << ",\"lw\":" << p.line_width << ",\"al\":" << p.alpha;
        o << ",\"show_var\":[";
        for (size_t v = 0; v < p.show_var.size(); ++v) { if (v)o << ","; o << (p.show_var[v] ? "true" : "false"); }
        o << "]}";
    }
    o << "],\n";
    o << "  \"layout\":\"\"\n"; // зарезервировано
    o << "}\n";
    return o.str();
}

bool session_from_json(const std::string& json, PhaseAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "step_h")     s.step_h = p.str();
            else if (key == "sim_time")   s.sim_time = p.str();
            else if (key == "skip_time")  s.skip_time = p.str();
            else if (key == "scheme")     s.scheme = p.str();
            else if (key == "symmetry_s") s.symmetry_s = p.str();
            else if (key == "decimation") s.decimation = p.str();
            else if (key == "auto_recompute") s.auto_recompute = p.boolean();
            else if (key == "legend_show_ic") s.legend_show_ic = p.boolean();
            else if (key == "use_gpu")        s.use_gpu = p.boolean();
            else if (key == "param_values")   s.param_values = p.map_ss();
            else if (key == "ic_sets") {
                s.ic_sets.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        InitialConditionSet ic;
                        while (true) {
                            std::string k = p.str(); p.expect(':');
                            if (k == "label")   ic.label = p.str();
                            else if (k == "visible") ic.visible = p.boolean();
                            else if (k == "values")  ic.values = p.map_ss();
                            else p.skip_value();
                            if (p.opt(',')) continue;
                            p.expect('}'); break;
                        }
                        s.ic_sets.push_back(std::move(ic));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "projections") {
                s.projections.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        Projection pr;
                        while (true) {
                            std::string k = p.str(); p.expect(':');
                            if (k == "label") pr.label = p.str();
                            else if (k == "type") { pr.type = (ProjType)std::stoi(p.str_or_num()); }
                            else if (k == "ax")    pr.axis_x = std::stoi(p.str_or_num());
                            else if (k == "ay")    pr.axis_y = std::stoi(p.str_or_num());
                            else if (k == "az")    pr.axis_z = std::stoi(p.str_or_num());
                            else if (k == "cls")   pr.custom_line_style = p.boolean();
                            else if (k == "lw")    pr.line_width = (float)std::stod(p.str_or_num());
                            else if (k == "al")    pr.alpha      = (float)std::stod(p.str_or_num());
                            else if (k == "show_var") {
                                pr.show_var.clear();
                                p.expect('[');
                                if (!p.opt(']')) { while (true) { pr.show_var.push_back(p.boolean()); if (p.opt(','))continue; p.expect(']'); break; } }
                            }
                            else p.skip_value();
                            if (p.opt(',')) continue;
                            p.expect('}'); break;
                        }
                        s.projections.push_back(std::move(pr));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        // КРС зависит от scheme: после загрузки сессии перегенерируем, иначе
        // GPU-путь считал бы по старому методу (combo показывает новый scheme,
        // а krs_code остался прежним).
        s.regenerate_krs();
        return true;
    }
    catch (...) {
        return false;
    }
}
// ============================================================================
// BifurcationAnalysisSession
// ============================================================================

// Сериализует одну БД в JSON-объект (без обёртки фигурными). Используется
// внутри массива "diagrams".
static void write_diagram(std::ostringstream& o, const BifurcationDiagramConfig& bd) {
    o << "{";
    o << "\"label\":";            jstr(o, bd.label);
    o << ",\"visible\":"          << (bd.visible ? "true" : "false");
    o << ",\"scheme\":";          jstr(o, bd.scheme);
    o << ",\"symmetry_s\":";      jstr(o, bd.symmetry_s);
    o << ",\"param_index\":"      << bd.param_index;
    o << ",\"sweep_over_var\":"   << (bd.sweep_over_var ? "true" : "false");
    o << ",\"var_sweep_index\":"  << bd.var_sweep_index;
    o << ",\"continuation\":"     << (bd.continuation ? "true" : "false");
    o << ",\"continuation_reverse\":" << (bd.continuation_reverse ? "true" : "false");
    o << ",\"param_lo_text\":";   jstr(o, bd.param_lo_text);
    o << ",\"param_hi_text\":";   jstr(o, bd.param_hi_text);
    o << ",\"n_pts_text\":";      jstr(o, bd.n_pts_text);
    o << ",\"writable_var\":"     << bd.writable_var;
    o << ",\"h_text\":";          jstr(o, bd.h_text);
    o << ",\"t_max_text\":";      jstr(o, bd.t_max_text);
    o << ",\"transient_text\":";  jstr(o, bd.transient_text);
    o << ",\"pre_scaller_text\":";jstr(o, bd.pre_scaller_text);
    o << ",\"max_value_text\":";  jstr(o, bd.max_value_text);
    o << ",\"param_values\":";    jmap(o, bd.param_values);
    o << ",\"initial_conditions\":"; jmap(o, bd.initial_conditions);
    o << ",\"csv_save_enabled\":" << (bd.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, bd.csv_output_path);
    o << ",\"plot_inter_peaks\":" << (bd.plot_inter_peaks ? "true" : "false");
    // 2D-mode config: result_2d not persisted (too heavy), user runs again.
    o << ",\"mode_2d\":"           << (bd.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"     << bd.param_index_2;
    o << ",\"sweep_over_var_2\":"  << (bd.sweep_over_var_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":" << bd.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";  jstr(o, bd.param_lo_2_text);
    o << ",\"param_hi_2_text\":";  jstr(o, bd.param_hi_2_text);
    o << ",\"eps_dbscan_text\":";  jstr(o, bd.eps_dbscan_text);
    o << "}";
}

// Читает одно поле в БД. Возвращает true, если ключ распознан.
// Используется и при разборе нового формата (внутри "diagrams"), и при
// разборе legacy-формата (плоские ключи на верхнем уровне = одна БД).
static bool read_diagram_field(JP& p, BifurcationDiagramConfig& bd, const std::string& key) {
    if      (key == "label")              bd.label             = p.str();
    else if (key == "visible")            bd.visible           = p.boolean();
    else if (key == "scheme")             bd.scheme            = p.str();
    else if (key == "symmetry_s")         bd.symmetry_s        = p.str();
    else if (key == "param_index")        bd.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     bd.sweep_over_var    = p.boolean();
    else if (key == "var_sweep_index")    bd.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "continuation")       bd.continuation      = p.boolean();
    else if (key == "continuation_reverse") bd.continuation_reverse = p.boolean();
    else if (key == "param_lo_text")      bd.param_lo_text     = p.str();
    else if (key == "param_hi_text")      bd.param_hi_text     = p.str();
    else if (key == "n_pts_text")         bd.n_pts_text        = p.str();
    else if (key == "writable_var")       bd.writable_var      = std::stoi(p.str_or_num());
    else if (key == "h_text")             bd.h_text            = p.str();
    else if (key == "t_max_text")         bd.t_max_text        = p.str();
    else if (key == "transient_text")     bd.transient_text    = p.str();
    else if (key == "pre_scaller_text")   bd.pre_scaller_text  = p.str();
    else if (key == "max_value_text")     bd.max_value_text    = p.str();
    else if (key == "param_values")       bd.param_values      = p.map_ss();
    else if (key == "initial_conditions") bd.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   bd.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    bd.csv_output_path   = p.str();
    else if (key == "plot_inter_peaks")   bd.plot_inter_peaks  = p.boolean();
    // 2D-mode: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора.
    else if (key == "mode_2d")            bd.mode_2d           = p.boolean();
    else if (key == "param_index_2")      bd.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   bd.sweep_over_var_2  = p.boolean();
    else if (key == "var_sweep_index_2")  bd.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    bd.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    bd.param_hi_2_text   = p.str();
    else if (key == "eps_dbscan_text")    bd.eps_dbscan_text   = p.str();
    else return false;
    return true;
}

std::string session_to_json_parametric(const BifurcationAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_diagram_index\":" << s.active_diagram_index << ",\n";
    o << "  \"diagrams\":[";
    for (size_t i = 0; i < s.diagrams.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_diagram(o, s.diagrams[i]);
    }
    if (!s.diagrams.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}

bool session_from_json_parametric(const std::string& json, BifurcationAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;

        // Legacy (старые _last_parametric.json без ключа "diagrams") — собираем
        // плоские поля в одну БД. Если встречается ключ "diagrams" — переключаемся
        // на новый формат и старый buffer отбрасываем.
        BifurcationDiagramConfig legacy;
        bool legacy_has_fields = false;
        bool new_format = false;

        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "diagrams") {
                new_format = true;
                s.diagrams.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        BifurcationDiagramConfig bd;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_diagram_field(p, bd, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.diagrams.push_back(std::move(bd));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_diagram_index") {
                s.active_diagram_index = std::stoi(p.str_or_num());
            }
            else if (read_diagram_field(p, legacy, key)) {
                legacy_has_fields = true;
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }

        if (!new_format && legacy_has_fields) {
            // Старое сохранение: оборачиваем в одну БД и заменяем существующий
            // дефолт (load_from_record уже положил BD 1 — заменим его).
            if (legacy.label.empty()) legacy.label = "BD 1";
            s.diagrams.clear();
            s.diagrams.push_back(std::move(legacy));
        }

        if (s.diagrams.empty()) {
            // Сейв был пустой — добавим хотя бы дефолтную БД.
            s.add_diagram();
        }
        if (s.active_diagram_index < 0 || s.active_diagram_index >= (int)s.diagrams.size())
            s.active_diagram_index = 0;
        s.running_diagram_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}

// ============================================================================
// LLEAnalysisSession — отдельный JSON-файл `_last_lle.json`. Структура та же
// что у parametric (массив объектов), отличаются только поля per-«прогон».
// ============================================================================

namespace {

void write_lle_curve(std::ostringstream& o, const LLECurveConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);
    o << ",\"visible\":"          << (c.visible ? "true" : "false");
    o << ",\"scheme\":";          jstr(o, c.scheme);
    o << ",\"symmetry_s\":";      jstr(o, c.symmetry_s);
    o << ",\"param_index\":"      << c.param_index;
    o << ",\"sweep_over_var\":"   << (c.sweep_over_var ? "true" : "false");
    o << ",\"var_sweep_index\":"  << c.var_sweep_index;
    o << ",\"param_lo_text\":";   jstr(o, c.param_lo_text);
    o << ",\"param_hi_text\":";   jstr(o, c.param_hi_text);
    o << ",\"n_pts_text\":";      jstr(o, c.n_pts_text);
    o << ",\"h_text\":";          jstr(o, c.h_text);
    o << ",\"t_max_text\":";      jstr(o, c.t_max_text);
    o << ",\"transient_text\":";  jstr(o, c.transient_text);
    o << ",\"max_value_text\":";  jstr(o, c.max_value_text);
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    o << ",\"param_values\":";    jmap(o, c.param_values);
    o << ",\"initial_conditions\":"; jmap(o, c.initial_conditions);
    o << ",\"csv_save_enabled\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, c.csv_output_path);
    // 2D-режим (heatmap): persistим только конфиг, result_2d не храним
    // (на 200x200 это ~40k чисел в JSON — слишком тяжело). После load
    // пользователь нажмёт Run заново.
    o << ",\"mode_2d\":"           << (c.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"     << c.param_index_2;
    o << ",\"sweep_over_var_2\":"  << (c.sweep_over_var_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":" << c.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";  jstr(o, c.param_lo_2_text);
    o << ",\"param_hi_2_text\":";  jstr(o, c.param_hi_2_text);
    o << "}";
}

bool read_lle_curve_field(JP& p, LLECurveConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "visible")            c.visible           = p.boolean();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "param_index")        c.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     c.sweep_over_var    = p.boolean();
    else if (key == "var_sweep_index")    c.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "param_lo_text")      c.param_lo_text     = p.str();
    else if (key == "param_hi_text")      c.param_hi_text     = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    // 2D-режим: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора (mode_2d=false, ...).
    else if (key == "mode_2d")            c.mode_2d           = p.boolean();
    else if (key == "param_index_2")      c.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   c.sweep_over_var_2  = p.boolean();
    else if (key == "var_sweep_index_2")  c.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    c.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    c.param_hi_2_text   = p.str();
    else return false;
    return true;
}

} // namespace

std::string session_to_json_lle(const LLEAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_curve_index\":" << s.active_curve_index << ",\n";
    o << "  \"curves\":[";
    for (size_t i = 0; i < s.curves.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_lle_curve(o, s.curves[i]);
    }
    if (!s.curves.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}

bool session_from_json_lle(const std::string& json, LLEAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "curves") {
                s.curves.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        LLECurveConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_lle_curve_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.curves.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_curve_index") {
                s.active_curve_index = std::stoi(p.str_or_num());
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.curves.empty()) s.add_curve();
        if (s.active_curve_index < 0 || s.active_curve_index >= (int)s.curves.size())
            s.active_curve_index = 0;
        s.running_curve_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}

// ============================================================================
// LyapunovSpectrumAnalysisSession — `_last_ls.json`. Поля LSCurveConfig
// идентичны LLECurveConfig — копия LLE-сериализатора.
// ============================================================================

namespace {

void write_ls_curve(std::ostringstream& o, const LSCurveConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);
    o << ",\"visible\":"          << (c.visible ? "true" : "false");
    o << ",\"scheme\":";          jstr(o, c.scheme);
    o << ",\"symmetry_s\":";      jstr(o, c.symmetry_s);
    o << ",\"param_index\":"      << c.param_index;
    o << ",\"sweep_over_var\":"   << (c.sweep_over_var ? "true" : "false");
    o << ",\"var_sweep_index\":"  << c.var_sweep_index;
    o << ",\"param_lo_text\":";   jstr(o, c.param_lo_text);
    o << ",\"param_hi_text\":";   jstr(o, c.param_hi_text);
    o << ",\"n_pts_text\":";      jstr(o, c.n_pts_text);
    o << ",\"h_text\":";          jstr(o, c.h_text);
    o << ",\"t_max_text\":";      jstr(o, c.t_max_text);
    o << ",\"transient_text\":";  jstr(o, c.transient_text);
    o << ",\"max_value_text\":";  jstr(o, c.max_value_text);
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    o << ",\"param_values\":";    jmap(o, c.param_values);
    o << ",\"initial_conditions\":"; jmap(o, c.initial_conditions);
    o << ",\"csv_save_enabled\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, c.csv_output_path);
    // 2D-mode config: result_2d not persisted (too heavy), user runs again.
    o << ",\"mode_2d\":"             << (c.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"       << c.param_index_2;
    o << ",\"sweep_over_var_2\":"    << (c.sweep_over_var_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":"   << c.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";    jstr(o, c.param_lo_2_text);
    o << ",\"param_hi_2_text\":";    jstr(o, c.param_hi_2_text);
    o << ",\"display_exponent_idx\":" << c.display_exponent_idx;
    o << "}";
}

bool read_ls_curve_field(JP& p, LSCurveConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "visible")            c.visible           = p.boolean();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "param_index")        c.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     c.sweep_over_var    = p.boolean();
    else if (key == "var_sweep_index")    c.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "param_lo_text")      c.param_lo_text     = p.str();
    else if (key == "param_hi_text")      c.param_hi_text     = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    // 2D-mode: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора.
    else if (key == "mode_2d")            c.mode_2d           = p.boolean();
    else if (key == "param_index_2")      c.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   c.sweep_over_var_2  = p.boolean();
    else if (key == "var_sweep_index_2")  c.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    c.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    c.param_hi_2_text   = p.str();
    else if (key == "display_exponent_idx") c.display_exponent_idx = std::stoi(p.str_or_num());
    else return false;
    return true;
}

} // namespace

std::string session_to_json_ls(const LyapunovSpectrumAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_curve_index\":" << s.active_curve_index << ",\n";
    o << "  \"curves\":[";
    for (size_t i = 0; i < s.curves.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_ls_curve(o, s.curves[i]);
    }
    if (!s.curves.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}

bool session_from_json_ls(const std::string& json, LyapunovSpectrumAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "curves") {
                s.curves.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        LSCurveConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_ls_curve_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.curves.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_curve_index") {
                s.active_curve_index = std::stoi(p.str_or_num());
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.curves.empty()) s.add_curve();
        if (s.active_curve_index < 0 || s.active_curve_index >= (int)s.curves.size())
            s.active_curve_index = 0;
        s.running_curve_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}

// ============================================================================
// BasinsAnalysisSession — `_last_basins.json`. Один config на сессию
// (без curves-vector). Result не сохраняется.
// ============================================================================

// Запись одного BasinsConfig в JSON (без внешних { } — пишет голый объект).
// Используется внутри массива "configs".
static void write_basins_config(std::ostringstream& o, const BasinsConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
    o << "\"scheme\":";           jstr(o, c.scheme);           o << ",";
    o << "\"symmetry_s\":";       jstr(o, c.symmetry_s);       o << ",";
    o << "\"axis_x_var\":"        << c.axis_x_var            << ",";
    o << "\"axis_y_var\":"        << c.axis_y_var            << ",";
    o << "\"axis_x_lo_text\":";   jstr(o, c.axis_x_lo_text);   o << ",";
    o << "\"axis_x_hi_text\":";   jstr(o, c.axis_x_hi_text);   o << ",";
    o << "\"axis_y_lo_text\":";   jstr(o, c.axis_y_lo_text);   o << ",";
    o << "\"axis_y_hi_text\":";   jstr(o, c.axis_y_hi_text);   o << ",";
    o << "\"n_pts_text\":";       jstr(o, c.n_pts_text);       o << ",";
    o << "\"writable_var\":"      << c.writable_var          << ",";
    o << "\"h_text\":";           jstr(o, c.h_text);           o << ",";
    o << "\"t_max_text\":";       jstr(o, c.t_max_text);       o << ",";
    o << "\"transient_text\":";   jstr(o, c.transient_text);   o << ",";
    o << "\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text); o << ",";
    o << "\"max_value_text\":";   jstr(o, c.max_value_text);   o << ",";
    o << "\"eps_dbscan_text\":";  jstr(o, c.eps_dbscan_text);  o << ",";
    o << "\"csv_save_enabled\":"  << (c.csv_save_enabled ? "true" : "false") << ",";
    o << "\"csv_output_path\":";  jstr(o, c.csv_output_path);  o << ",";
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";
    o << "\"param_values\":";     jmap(o, c.param_values);     o << ",";
    o << "\"feature1\":"          << c.feature1            << ",";
    o << "\"feature2\":"          << c.feature2            << ",";
    o << "\"mult_feature1_text\":"; jstr(o, c.mult_feature1_text); o << ",";
    o << "\"mult_feature2_text\":"; jstr(o, c.mult_feature2_text); o << ",";
    o << "\"active_plot_tab\":"   << c.active_plot_tab;
    o << "}";
}

// Чтение одного поля BasinsConfig. Возвращает true если ключ распознан и
// значение прочитано из p. Иначе false — caller должен сам сделать skip_value.
// Используется и при разборе нового формата (внутри "configs"), и при
// разборе legacy плоского формата.
static bool read_basins_field(JP& p, BasinsConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "axis_x_var")         c.axis_x_var        = std::stoi(p.str_or_num());
    else if (key == "axis_y_var")         c.axis_y_var        = std::stoi(p.str_or_num());
    else if (key == "axis_x_lo_text")     c.axis_x_lo_text    = p.str();
    else if (key == "axis_x_hi_text")     c.axis_x_hi_text    = p.str();
    else if (key == "axis_y_lo_text")     c.axis_y_lo_text    = p.str();
    else if (key == "axis_y_hi_text")     c.axis_y_hi_text    = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "writable_var")       c.writable_var      = std::stoi(p.str_or_num());
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "pre_scaller_text")   c.pre_scaller_text  = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_dbscan_text")    c.eps_dbscan_text   = p.str();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "feature1")           c.feature1           = std::stoi(p.str_or_num());
    else if (key == "feature2")           c.feature2           = std::stoi(p.str_or_num());
    else if (key == "mult_feature1_text") c.mult_feature1_text = p.str();
    else if (key == "mult_feature2_text") c.mult_feature2_text = p.str();
    else if (key == "active_plot_tab")    c.active_plot_tab   = std::stoi(p.str_or_num());
    else return false;
    return true;
}

std::string session_to_json_basins(const BasinsAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_basins_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}

// ============================================================================
// FastSyncAnalysisSession JSON — multi-config layout как у basins.
// ============================================================================
static void write_fastsync_config(std::ostringstream& o, const FastSyncConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
    o << "\"scheme\":";           jstr(o, c.scheme);           o << ",";
    o << "\"symmetry_s\":";       jstr(o, c.symmetry_s);       o << ",";
    o << "\"mode\":"              << c.mode                  << ",";
    o << "\"h_text\":";           jstr(o, c.h_text);           o << ",";
    o << "\"iter_of_synchr_text\":"; jstr(o, c.iter_of_synchr_text); o << ",";
    o << "\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text); o << ",";
    o << "\"max_value_text\":";   jstr(o, c.max_value_text);   o << ",";
    o << "\"t_max_text\":";       jstr(o, c.t_max_text);       o << ",";
    o << "\"transient_text\":";   jstr(o, c.transient_text);   o << ",";
    o << "\"window_text\":";      jstr(o, c.window_text);      o << ",";
    o << "\"axis_x_var\":"        << c.axis_x_var            << ",";
    o << "\"axis_y_var\":"        << c.axis_y_var            << ",";
    o << "\"axis_x_lo_text\":";   jstr(o, c.axis_x_lo_text);   o << ",";
    o << "\"axis_x_hi_text\":";   jstr(o, c.axis_x_hi_text);   o << ",";
    o << "\"axis_y_lo_text\":";   jstr(o, c.axis_y_lo_text);   o << ",";
    o << "\"axis_y_hi_text\":";   jstr(o, c.axis_y_hi_text);   o << ",";
    o << "\"n_pts_text\":";       jstr(o, c.n_pts_text);       o << ",";
    o << "\"type_of_synch\":"     << c.type_of_synch         << ",";
    o << "\"error_estim\":"       << c.error_estim           << ",";
    o << "\"fs_error_trs_text\":"; jstr(o, c.fs_error_trs_text); o << ",";
    o << "\"colormap_idx\":"      << c.colormap_idx          << ",";
    o << "\"autoscale_color\":"   << (c.autoscale_color ? "true" : "false") << ",";
    o << "\"c_min_text\":";       jstr(o, c.c_min_text);       o << ",";
    o << "\"c_max_text\":";       jstr(o, c.c_max_text);       o << ",";
    o << "\"line_width\":"        << c.line_width            << ",";
    o << "\"alpha\":"             << c.alpha                 << ",";
    o << "\"swap_axes\":"         << (c.swap_axes ? "true" : "false") << ",";
    o << "\"invert_depth\":"      << (c.invert_depth ? "true" : "false") << ",";
    o << "\"ic_master\":";        jmap(o, c.ic_master);        o << ",";
    o << "\"ic_slave\":";         jmap(o, c.ic_slave);         o << ",";
    o << "\"k_forward\":";        jmap(o, c.k_forward);        o << ",";
    o << "\"k_backward\":";       jmap(o, c.k_backward);       o << ",";
    o << "\"param_values\":";     jmap(o, c.param_values);
    o << "}";
}

static bool read_fastsync_field(JP& p, FastSyncConfig& c, const std::string& key) {
    if      (key == "label")               c.label               = p.str();
    else if (key == "scheme")              c.scheme              = p.str();
    else if (key == "symmetry_s")          c.symmetry_s          = p.str();
    else if (key == "mode")                c.mode                = std::stoi(p.str_or_num());
    else if (key == "h_text")              c.h_text              = p.str();
    else if (key == "iter_of_synchr_text") c.iter_of_synchr_text = p.str();
    else if (key == "pre_scaller_text")    c.pre_scaller_text    = p.str();
    else if (key == "max_value_text")      c.max_value_text      = p.str();
    else if (key == "t_max_text")          c.t_max_text          = p.str();
    else if (key == "transient_text")      c.transient_text      = p.str();
    else if (key == "window_text")         c.window_text         = p.str();
    // Backward-compat: старые сессии могли называть это поле "n_time_text".
    else if (key == "n_time_text")         c.window_text         = p.str();
    else if (key == "axis_x_var")          c.axis_x_var          = std::stoi(p.str_or_num());
    else if (key == "axis_y_var")          c.axis_y_var          = std::stoi(p.str_or_num());
    else if (key == "axis_x_lo_text")      c.axis_x_lo_text      = p.str();
    else if (key == "axis_x_hi_text")      c.axis_x_hi_text      = p.str();
    else if (key == "axis_y_lo_text")      c.axis_y_lo_text      = p.str();
    else if (key == "axis_y_hi_text")      c.axis_y_hi_text      = p.str();
    else if (key == "n_pts_text")          c.n_pts_text          = p.str();
    else if (key == "type_of_synch")       c.type_of_synch       = std::stoi(p.str_or_num());
    else if (key == "error_estim")         c.error_estim         = std::stoi(p.str_or_num());
    else if (key == "fs_error_trs_text")   c.fs_error_trs_text   = p.str();
    else if (key == "colormap_idx")        c.colormap_idx        = std::stoi(p.str_or_num());
    else if (key == "autoscale_color")     c.autoscale_color     = p.boolean();
    else if (key == "c_min_text")          c.c_min_text          = p.str();
    else if (key == "c_max_text")          c.c_max_text          = p.str();
    else if (key == "line_width")          c.line_width          = (float)std::stod(p.str_or_num());
    else if (key == "alpha")               c.alpha               = (float)std::stod(p.str_or_num());
    else if (key == "swap_axes")           c.swap_axes           = p.boolean();
    else if (key == "invert_depth")        c.invert_depth        = p.boolean();
    else if (key == "decimator_view")      p.skip_value();   // legacy, не используется
    else if (key == "ic_master")           c.ic_master           = p.map_ss();
    else if (key == "ic_slave")            c.ic_slave            = p.map_ss();
    else if (key == "k_forward")           c.k_forward           = p.map_ss();
    else if (key == "k_backward")          c.k_backward          = p.map_ss();
    else if (key == "param_values")        c.param_values        = p.map_ss();
    else return false;
    return true;
}

std::string session_to_json_fastsync(const FastSyncAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_fastsync_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}

bool session_from_json_fastsync(const std::string& json, FastSyncAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                s.configs.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        FastSyncConfig fc;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_fastsync_field(p, fc, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.configs.push_back(std::move(fc));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") s.active_config_index = std::stoi(p.str_or_num());
            else                                    p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
            s.active_config_index = 0;
        s.running_config_index = -1;
        return true;
    } catch (...) {
        return false;
    }
}

bool session_from_json_basins(const std::string& json, BasinsAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;

        // Legacy (старые _last_basins.json без ключа "configs") — собираем
        // плоские поля в один config. Если встречается ключ "configs" —
        // переключаемся на новый формат и legacy-buffer отбрасываем.
        BasinsConfig legacy;
        bool legacy_has_fields = false;
        bool new_format = false;

        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                new_format = true;
                s.configs.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        BasinsConfig bc;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_basins_field(p, bc, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.configs.push_back(std::move(bc));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") {
                s.active_config_index = std::stoi(p.str_or_num());
            }
            else if (read_basins_field(p, legacy, key)) {
                legacy_has_fields = true;
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }

        if (!new_format && legacy_has_fields) {
            // Старое сохранение: оборачиваем в один config (заменяет
            // дефолт, который положил load_from_record).
            if (legacy.label.empty()) legacy.label = "Basins 1";
            s.configs.clear();
            s.configs.push_back(std::move(legacy));
        }

        if (s.configs.empty()) {
            // JSON был пустой или только non-config поля — оставим хотя бы
            // один config (load_from_record уже положил дефолт).
            return true;
        }
        if (s.active_config_index < 0 ||
            s.active_config_index >= (int)s.configs.size()) {
            s.active_config_index = 0;
        }
        s.running_config_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}
