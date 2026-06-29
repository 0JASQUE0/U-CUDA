#include "system_library.h"
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

// папка сессий внутри папки системы (определение ниже)
static std::string sessions_dir(const std::string& sysdir);

// ==================== JSON helpers ====================
namespace {

    // Экранирование строки для JSON.
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
                if ((unsigned char)c < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                }
                else o += c;
            }
        }
        return o;
    }

    void kv(std::ostringstream& o, const std::string& key, const std::string& val, bool last = false) {
        o << "  \"" << key << "\": \"" << esc(val) << "\"" << (last ? "\n" : ",\n");
    }
    void kvbool(std::ostringstream& o, const std::string& key, bool val, bool last = false) {
        o << "  \"" << key << "\": " << (val ? "true" : "false") << (last ? "\n" : ",\n");
    }
    void kvmap(std::ostringstream& o, const std::string& key,
        const std::map<std::string, std::string>& m, bool last = false) {
        o << "  \"" << key << "\": {";
        bool first = true;
        for (auto& p : m) {
            if (!first) o << ", ";
            o << "\"" << esc(p.first) << "\": \"" << esc(p.second) << "\"";
            first = false;
        }
        o << "}" << (last ? "\n" : ",\n");
    }

    // ---- Минимальный JSON-парсер (под нашу плоскую структуру) ----
    struct JParser {
        std::string s; size_t i = 0;
        JParser(std::string src) : s(std::move(src)) {}
        void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
        char peek() { ws(); return i < s.size() ? s[i] : '\0'; }

        std::string parse_string() {
            ws();
            if (s[i] != '"') throw std::runtime_error("JSON: expected string");
            ++i; std::string o;
            while (i < s.size() && s[i] != '"') {
                char c = s[i++];
                if (c == '\\' && i < s.size()) {
                    char e = s[i++];
                    switch (e) {
                    case 'n': o += '\n'; break;
                    case 'r': o += '\r'; break;
                    case 't': o += '\t'; break;
                    case '"': o += '"'; break;
                    case '\\': o += '\\'; break;
                    case '/': o += '/'; break;
                    case 'u': {
                        if (i + 4 <= s.size()) {
                            int code = std::stoi(s.substr(i, 4), nullptr, 16);
                            i += 4;
                            if (code < 0x80) o += (char)code;
                            // (для наших данных >0x80 в \u не ожидается)
                        }
                        break;
                    }
                    default: o += e;
                    }
                }
                else o += c;
            }
            if (i < s.size()) ++i; // закрывающая "
            return o;
        }

        bool parse_bool() {
            ws();
            if (s.compare(i, 4, "true") == 0) { i += 4; return true; }
            if (s.compare(i, 5, "false") == 0) { i += 5; return false; }
            throw std::runtime_error("JSON: expected bool");
        }

        // map<string,string> : { "k": "v", ... }
        std::map<std::string, std::string> parse_map() {
            std::map<std::string, std::string> m;
            ws();
            if (s[i] != '{') throw std::runtime_error("JSON: expected {");
            ++i; ws();
            if (peek() == '}') { ++i; return m; }
            while (true) {
                std::string k = parse_string();
                ws(); if (s[i] == ':') ++i;
                std::string v = parse_string();
                m[k] = v;
                ws();
                if (peek() == ',') { ++i; continue; }
                if (peek() == '}') { ++i; break; }
                break;
            }
            return m;
        }
    };

} // namespace

// ==================== record <-> JSON ====================
std::string record_to_json(const SystemRecord& r) {
    std::ostringstream o;
    o << "{\n";
    kv(o, "name", r.name);
    kv(o, "note", r.note);
    kv(o, "mode", r.mode);
    kv(o, "latex_text", r.latex_text);
    kv(o, "plain_text", r.plain_text);
    kv(o, "alphabet_text", r.alphabet_text);
    kv(o, "vars_text", r.vars_text);
    kv(o, "params_text", r.params_text);
    kvbool(o, "use_aux_funcs", r.use_aux_funcs);
    kv(o, "func_defs_text", r.func_defs_text);
    kv(o, "param_order", r.param_order);
    kvbool(o, "scheme_euler", r.scheme_euler);
    kvbool(o, "scheme_cromer", r.scheme_cromer);
    kvbool(o, "scheme_midpoint", r.scheme_midpoint);
    kvbool(o, "scheme_rk4", r.scheme_rk4);
    kvbool(o, "scheme_dopri78", r.scheme_dopri78);
    kvbool(o, "scheme_cd", r.scheme_cd);
    kv(o, "symmetry_s", r.symmetry_s);
    kv(o, "step_h", r.step_h);
    kvmap(o, "init_conditions", r.init_conditions);
    kvmap(o, "param_values", r.param_values);
    // custom_schemes: [ {"name": "...", "body": "..."}, ... ]
    o << "  \"custom_schemes\": [";
    for (size_t k = 0; k < r.custom_schemes.size(); ++k) {
        const auto& cs = r.custom_schemes[k];
        if (k) o << ", ";
        o << "{\"name\": \"" << esc(cs.name)
          << "\", \"body\": \"" << esc(cs.body) << "\"}";
    }
    o << "]\n";
    o << "}\n";
    return o.str();
}

SystemRecord record_from_json(const std::string& json) {
    SystemRecord r;
    JParser p(json);
    p.ws();
    if (p.peek() != '{') throw std::runtime_error("JSON: expected object");
    ++p.i;
    while (true) {
        if (p.peek() == '}') { ++p.i; break; }
        std::string key = p.parse_string();
        p.ws(); if (p.s[p.i] == ':') ++p.i;
        p.ws();
        // значение: bool, map или string
        if (key == "use_aux_funcs") r.use_aux_funcs = p.parse_bool();
        else if (key == "scheme_euler") r.scheme_euler = p.parse_bool();
        else if (key == "scheme_cromer") r.scheme_cromer = p.parse_bool();
        else if (key == "scheme_midpoint") r.scheme_midpoint = p.parse_bool();
        else if (key == "scheme_rk4") r.scheme_rk4 = p.parse_bool();
        else if (key == "scheme_dopri78") r.scheme_dopri78 = p.parse_bool();
        else if (key == "scheme_cd") r.scheme_cd = p.parse_bool();
        else if (key == "init_conditions") r.init_conditions = p.parse_map();
        else if (key == "param_values") r.param_values = p.parse_map();
        else if (key == "custom_schemes") {
            p.ws();
            if (p.s[p.i] != '[') throw std::runtime_error("JSON: expected [ for custom_schemes");
            ++p.i; p.ws();
            if (p.peek() != ']') {
                while (true) {
                    p.ws();
                    if (p.s[p.i] != '{') throw std::runtime_error("JSON: expected { in custom_schemes");
                    ++p.i;
                    CustomScheme cs;
                    while (true) {
                        std::string ck = p.parse_string();
                        p.ws(); if (p.s[p.i] == ':') ++p.i;
                        std::string cv = p.parse_string();
                        if (ck == "name") cs.name = cv;
                        else if (ck == "body") cs.body = cv;
                        p.ws();
                        if (p.peek() == ',') { ++p.i; continue; }
                        if (p.peek() == '}') { ++p.i; break; }
                        break;
                    }
                    r.custom_schemes.push_back(std::move(cs));
                    p.ws();
                    if (p.peek() == ',') { ++p.i; continue; }
                    if (p.peek() == ']') { ++p.i; break; }
                    break;
                }
            } else { ++p.i; }
        }
        else {
            std::string val = p.parse_string();
            if (key == "name") r.name = val;
            else if (key == "note") r.note = val;
            else if (key == "mode") r.mode = val;
            else if (key == "latex_text") r.latex_text = val;
            else if (key == "plain_text") r.plain_text = val;
            else if (key == "alphabet_text") r.alphabet_text = val;
            else if (key == "vars_text") r.vars_text = val;
            else if (key == "params_text") r.params_text = val;
            else if (key == "func_defs_text") r.func_defs_text = val;
            else if (key == "param_order") r.param_order = val;
            else if (key == "step_h") r.step_h = val;
            else if (key == "symmetry_s") r.symmetry_s = val;
        }
        p.ws();
        if (p.peek() == ',') { ++p.i; continue; }
        if (p.peek() == '}') { ++p.i; break; }
    }
    return r;
}

// ==================== SystemLibrary ====================
SystemLibrary::SystemLibrary(std::string dir) : dir_(std::move(dir)) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
}

std::string SystemLibrary::sanitize(const std::string& name) const {
    std::string o;
    for (char c : name) {
        if (std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == ' ') o += c;
        else o += '_';
    }
    if (o.empty()) o = "untitled";
    return o;
}

// Папка системы: dir_/<sanitized name>/
std::string SystemLibrary::dir_for(const std::string& name) const {
    return (fs::path(dir_) / sanitize(name)).string();
}
// Определение системы: dir_/<name>/system.json
std::string SystemLibrary::path_for(const std::string& name) const {
    return (fs::path(dir_for(name)) / "system.json").string();
}

bool SystemLibrary::exists(const std::string& name) const {
    return fs::exists(path_for(name));
}

std::vector<std::string> SystemLibrary::list() const {
    std::vector<std::string> names;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return names;
    // сканируем подпапки, где есть system.json
    for (auto& e : fs::directory_iterator(dir_, ec)) {
        if (!e.is_directory()) continue;
        fs::path sj = e.path() / "system.json";
        if (!fs::exists(sj)) continue;
        try {
            std::ifstream f(sj);
            std::stringstream ss; ss << f.rdbuf();
            SystemRecord r = record_from_json(ss.str());
            names.push_back(r.name.empty() ? e.path().filename().string() : r.name);
        }
        catch (...) {
            names.push_back(e.path().filename().string());
        }
    }
    return names;
}

std::string SystemLibrary::save(const SystemRecord& rec) {
    SystemRecord r = rec;
    if (r.name.empty()) {
        int n = 1;
        while (exists("Untitled " + std::to_string(n))) ++n;
        r.name = "Untitled " + std::to_string(n);
    }
    std::error_code ec;
    fs::create_directories(dir_for(r.name), ec); // папка системы
    std::ofstream f(path_for(r.name), std::ios::binary);
    if (!f) throw std::runtime_error("cannot write system file for: " + r.name);
    f << record_to_json(r);
    return r.name;
}

SystemRecord SystemLibrary::load(const std::string& name) const {
    std::ifstream f(path_for(name));
    if (!f) throw std::runtime_error("system not found: " + name);
    std::stringstream ss; ss << f.rdbuf();
    return record_from_json(ss.str());
}

bool SystemLibrary::remove(const std::string& name) {
    std::error_code ec;
    // удаляем папку системы целиком (система + НУ + прочее)
    return fs::remove_all(dir_for(name), ec) > 0;
}

bool SystemLibrary::rename(const std::string& old_name, const std::string& new_name) {
    if (old_name == new_name) return true;
    std::error_code ec;
    std::string olddir = dir_for(old_name);
    std::string newdir = dir_for(new_name);
    if (!fs::exists(olddir)) return false;
    if (fs::exists(newdir)) return false; // имя занято
    fs::rename(olddir, newdir, ec);        // переносит всё содержимое
    if (ec) return false;
    // обновим поле name внутри system.json, чтобы оно совпадало с папкой
    try {
        SystemRecord r = load(new_name);
        r.name = new_name;
        std::ofstream f(path_for(new_name), std::ios::binary);
        if (f) f << record_to_json(r);
    }
    catch (...) {}
    return true;
}

std::string SystemLibrary::duplicate(const std::string& name) {
    SystemRecord r = load(name);
    std::string base = r.name + " (copy)";
    std::string newname = base;
    int n = 2;
    while (exists(newname)) { newname = base + " " + std::to_string(n); ++n; }
    r.name = newname;
    std::string saved = save(r);
    // копируем папку sessions/ целиком, если есть
    std::error_code ec;
    std::string src = sessions_dir(dir_for(name));
    if (fs::exists(src, ec))
        fs::copy(src, sessions_dir(dir_for(saved)),
            fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    return saved;
}

// --- сессии: sessions/<name>.json в папке системы ---
static std::string sessions_dir(const std::string& sysdir) {
    return (fs::path(sysdir) / "sessions").string();
}

void SystemLibrary::save_session(const std::string& sysname, const std::string& session,
    const std::string& json) const {
    std::error_code ec;
    std::string sd = sessions_dir(dir_for(sysname));
    fs::create_directories(sd, ec);
    std::ofstream f((fs::path(sd) / (sanitize(session) + ".json")).string(), std::ios::binary);
    if (f) f << json;
}

std::string SystemLibrary::load_session(const std::string& sysname,
    const std::string& session) const {
    std::ifstream f((fs::path(sessions_dir(dir_for(sysname))) / (sanitize(session) + ".json")).string());
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

bool SystemLibrary::has_session(const std::string& sysname, const std::string& session) const {
    return fs::exists((fs::path(sessions_dir(dir_for(sysname))) / (sanitize(session) + ".json")).string());
}

bool SystemLibrary::remove_session(const std::string& sysname, const std::string& session) const {
    std::error_code ec;
    return fs::remove((fs::path(sessions_dir(dir_for(sysname))) / (sanitize(session) + ".json")).string(), ec);
}

std::vector<std::string> SystemLibrary::list_sessions(const std::string& sysname) const {
    std::vector<std::string> names;
    std::error_code ec;
    std::string sd = sessions_dir(dir_for(sysname));
    if (!fs::exists(sd, ec)) return names;
    for (auto& e : fs::directory_iterator(sd, ec)) {
        if (e.path().extension() != ".json") continue;
        std::string stem = e.path().stem().string();
        // авто-сессии не показываем в списке именованных
        if (stem == "_last" || stem == "_last_parametric") continue;
        names.push_back(stem);
    }
    return names;
}