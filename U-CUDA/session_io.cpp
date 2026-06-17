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
// ParametricAnalysisSession
// ============================================================================

std::string session_to_json_parametric(const ParametricAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"scheme\":";          jstr(o, s.scheme);          o << ",\n";
    o << "  \"param_index\":"    << s.param_index            << ",\n";
    o << "  \"param_lo_text\":";   jstr(o, s.param_lo_text);   o << ",\n";
    o << "  \"param_hi_text\":";   jstr(o, s.param_hi_text);   o << ",\n";
    o << "  \"n_pts_text\":";      jstr(o, s.n_pts_text);      o << ",\n";
    o << "  \"writable_var\":"   << s.writable_var           << ",\n";
    o << "  \"h_text\":";          jstr(o, s.h_text);          o << ",\n";
    o << "  \"t_max_text\":";      jstr(o, s.t_max_text);      o << ",\n";
    o << "  \"transient_text\":";  jstr(o, s.transient_text);  o << ",\n";
    o << "  \"pre_scaller_text\":";jstr(o, s.pre_scaller_text);o << ",\n";
    o << "  \"max_value_text\":";  jstr(o, s.max_value_text);  o << ",\n";
    o << "  \"param_values\":";    jmap(o, s.param_values);    o << ",\n";
    o << "  \"initial_conditions\":"; jmap(o, s.initial_conditions); o << ",\n";
    o << "  \"csv_save_enabled\":" << (s.csv_save_enabled ? "true" : "false") << ",\n";
    o << "  \"csv_output_path\":"; jstr(o, s.csv_output_path); o << "\n";
    o << "}\n";
    return o.str();
}

bool session_from_json_parametric(const std::string& json, ParametricAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if      (key == "scheme")             s.scheme           = p.str();
            else if (key == "param_index")        s.param_index      = std::stoi(p.str_or_num());
            else if (key == "param_lo_text")      s.param_lo_text    = p.str();
            else if (key == "param_hi_text")      s.param_hi_text    = p.str();
            else if (key == "n_pts_text")         s.n_pts_text       = p.str();
            else if (key == "writable_var")       s.writable_var     = std::stoi(p.str_or_num());
            else if (key == "h_text")             s.h_text           = p.str();
            else if (key == "t_max_text")         s.t_max_text       = p.str();
            else if (key == "transient_text")     s.transient_text   = p.str();
            else if (key == "pre_scaller_text")   s.pre_scaller_text = p.str();
            else if (key == "max_value_text")     s.max_value_text   = p.str();
            else if (key == "param_values")       s.param_values        = p.map_ss();
            else if (key == "initial_conditions") s.initial_conditions  = p.map_ss();
            else if (key == "csv_save_enabled")   s.csv_save_enabled = p.boolean();
            else if (key == "csv_output_path")    s.csv_output_path  = p.str();
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        // КРС перегенерируем под загруженный scheme
        s.regenerate_krs();
        return true;
    }
    catch (...) {
        return false;
    }
}
