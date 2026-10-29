#include "spice_parse.h"
#include "dae_devices.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

const double kPi = 3.14159265358979323846;

struct SpiceError : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] void fail(const std::string& m) { throw SpiceError(m); }

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Число для текста ДАУ: парсер уравнений читает "1e-12" как 1*e - 12, поэтому показатель —
// через 10^(k).
std::string fmt(double v) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.17g", v);
    std::string s = b;
    const size_t e = s.find('e');
    if (e == std::string::npos) return s;
    return "(" + s.substr(0, e) + "*10^(" + std::to_string(std::atoi(s.c_str() + e + 1)) + "))";
}

// Короткая запись для номиналов в записи: 1e-08, а не 1.0000000000000001e-08.
std::string fmt_short(double v) {
    char b[40];
    for (int p = 6; p <= 17; ++p) {
        std::snprintf(b, sizeof(b), "%.*g", p, v);
        if (std::strtod(b, nullptr) == v) break;
    }
    return b;
}

// --- числа и выражения -----------------------------------------------------------

// Число SPICE с множителем: 10k, 4.7u, 1meg, 2mil. Буквы после множителя — единицы
// (10nF, 1kOhm) и пропускаются, как в SPICE.
bool read_number(const std::string& s, size_t& i, double& v) {
    if (i >= s.size()) return false;
    const char c0 = s[i];
    if (!(std::isdigit((unsigned char)c0) ||
          (c0 == '.' && i + 1 < s.size() && std::isdigit((unsigned char)s[i + 1]))))
        return false;
    const char* b = s.c_str() + i;
    char* e = nullptr;
    v = std::strtod(b, &e);
    size_t k = i + (size_t)(e - b);
    std::string suf;
    while (k < s.size() && std::isalpha((unsigned char)s[k])) suf += (char)std::tolower((unsigned char)s[k++]);
    double m = 1.0;
    if (suf.rfind("meg", 0) == 0) m = 1e6;
    else if (suf.rfind("mil", 0) == 0) m = 25.4e-6;
    else if (!suf.empty()) {
        switch (suf[0]) {
        case 'f': m = 1e-15; break;
        case 'p': m = 1e-12; break;
        case 'n': m = 1e-9;  break;
        case 'u': m = 1e-6;  break;
        case 'm': m = 1e-3;  break;
        case 'k': m = 1e3;   break;
        case 'g': m = 1e9;   break;
        case 't': m = 1e12;  break;
        case 'a': m = 1e-18; break;
        default: break;
        }
    }
    v *= m;
    i = k;
    return true;
}

struct Scope {
    std::map<std::string, double> v;
    const Scope* parent = nullptr;
    bool get(const std::string& n, double& out) const {
        for (const Scope* s = this; s; s = s->parent) {
            auto it = s->v.find(n);
            if (it != s->v.end()) { out = it->second; return true; }
        }
        return false;
    }
};

bool ident_start(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool ident_char(char c)  { return std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == '$'; }

// Числовое значение выражения .param / {..}: + - * / ^ **, скобки, функции, параметры.
class Eval {
public:
    Eval(const std::string& s, const Scope& sc) : s_(s), sc_(sc) {}
    double run() {
        const double r = expr();
        skip();
        if (i_ != s_.size()) fail("cannot evaluate '" + s_ + "'");
        return r;
    }
private:
    const std::string& s_;
    const Scope& sc_;
    size_t i_ = 0;
    void skip() {
        while (i_ < s_.size() && (std::isspace((unsigned char)s_[i_]) || s_[i_] == '{' || s_[i_] == '}' || s_[i_] == '\''))
            ++i_;
    }
    bool at(char c) { skip(); return i_ < s_.size() && s_[i_] == c; }
    double expr() {
        double a = term();
        for (;;) {
            if (at('+')) { ++i_; a += term(); }
            else if (at('-')) { ++i_; a -= term(); }
            else return a;
        }
    }
    double term() {
        double a = unary();
        for (;;) {
            if (at('*') && !(i_ + 1 < s_.size() && s_[i_ + 1] == '*')) { ++i_; a *= unary(); }
            else if (at('/')) { ++i_; a /= unary(); }
            else return a;
        }
    }
    double unary() {
        if (at('-')) { ++i_; return -unary(); }
        if (at('+')) { ++i_; return unary(); }
        return power();
    }
    double power() {
        const double a = primary();
        if (at('^')) { ++i_; return std::pow(a, unary()); }
        if (at('*') && i_ + 1 < s_.size() && s_[i_ + 1] == '*') { i_ += 2; return std::pow(a, unary()); }
        return a;
    }
    double primary() {
        skip();
        if (i_ >= s_.size()) fail("unexpected end of '" + s_ + "'");
        if (s_[i_] == '(') {
            ++i_;
            const double r = expr();
            if (!at(')')) fail("missing ')' in '" + s_ + "'");
            ++i_;
            return r;
        }
        double v;
        if (read_number(s_, i_, v)) return v;
        if (!ident_start(s_[i_])) fail("unexpected '" + std::string(1, s_[i_]) + "' in '" + s_ + "'");
        size_t j = i_;
        while (j < s_.size() && ident_char(s_[j])) ++j;
        const std::string id = lower(s_.substr(i_, j - i_));
        i_ = j;
        if (at('(')) {
            ++i_;
            std::vector<double> a;
            if (!at(')')) {
                for (;;) {
                    a.push_back(expr());
                    if (at(',')) { ++i_; continue; }
                    break;
                }
            }
            if (!at(')')) fail("missing ')' after " + id + "(");
            ++i_;
            return call(id, a);
        }
        if (id == "pi") return kPi;
        if (sc_.get(id, v)) return v;
        fail("unknown parameter '" + id + "'");
    }
    double call(const std::string& f, const std::vector<double>& a) {
        auto need = [&](size_t n) { if (a.size() != n) fail(f + "() takes " + std::to_string(n) + " argument(s)"); };
        if (f == "sqrt") { need(1); return std::sqrt(a[0]); }
        if (f == "exp")  { need(1); return std::exp(a[0]); }
        if (f == "ln" || f == "log") { need(1); return std::log(a[0]); }
        if (f == "log10") { need(1); return std::log10(a[0]); }
        if (f == "sin")  { need(1); return std::sin(a[0]); }
        if (f == "cos")  { need(1); return std::cos(a[0]); }
        if (f == "tan")  { need(1); return std::tan(a[0]); }
        if (f == "asin") { need(1); return std::asin(a[0]); }
        if (f == "acos") { need(1); return std::acos(a[0]); }
        if (f == "atan") { need(1); return std::atan(a[0]); }
        if (f == "sinh") { need(1); return std::sinh(a[0]); }
        if (f == "cosh") { need(1); return std::cosh(a[0]); }
        if (f == "tanh") { need(1); return std::tanh(a[0]); }
        if (f == "abs")  { need(1); return std::fabs(a[0]); }
        if (f == "floor") { need(1); return std::floor(a[0]); }
        if (f == "ceil") { need(1); return std::ceil(a[0]); }
        if (f == "int")  { need(1); return std::trunc(a[0]); }
        if (f == "pow" || f == "pwr") { need(2); return std::pow(a[0], a[1]); }
        if (f == "atan2") { need(2); return std::atan2(a[0], a[1]); }
        if (f == "min")  { need(2); return std::min(a[0], a[1]); }
        if (f == "max")  { need(2); return std::max(a[0], a[1]); }
        fail("unknown function " + f + "()");
    }
};

double eval_value(const std::string& tok, const Scope& sc) {
    size_t i = 0;
    double v;
    if (read_number(tok, i, v) && i == tok.size()) return v;
    return Eval(tok, sc).run();
}

// --- карточки ----------------------------------------------------------------------

struct Card {
    std::string text;                 // строка целиком (продолжения склеены, комментарии сняты)
    std::vector<std::string> tok;     // ( ) , — разделители, '=' — отдельный токен, {..} и '..' целиком
    int line = 0;
};

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&] { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '{' || c == '\'' || c == '"') {
            const char close = (c == '{') ? '}' : c;
            flush();
            size_t j = i + 1;
            int depth = 1;
            while (j < s.size()) {
                if (c == '{' && s[j] == '{') ++depth;
                else if (s[j] == close && --depth == 0) break;
                ++j;
            }
            out.push_back(c == '"' ? s.substr(i + 1, j - i - 1) : s.substr(i, j - i + 1));
            i = j;
        } else if (std::isspace((unsigned char)c) || c == '(' || c == ')' || c == ',') {
            flush();
        } else if (c == '=') {
            flush();
            out.push_back("=");
        } else cur += c;
    }
    flush();
    return out;
}

// Строки текста -> карточки: первая строка — заголовок, '*' — комментарий, ';' и ' $' —
// комментарий до конца строки, '+' — продолжение предыдущей.
std::vector<Card> read_cards(const std::string& text, bool first_is_title, std::string* title) {
    std::vector<Card> cards;
    std::istringstream in(text);
    std::string ln;
    int no = 0;
    bool first = first_is_title;
    while (std::getline(in, ln)) {
        ++no;
        if (!ln.empty() && ln.back() == '\r') ln.pop_back();
        if (first) {
            first = false;
            std::string t = ln;
            const size_t a = t.find_first_not_of(" \t*");
            t = (a == std::string::npos) ? std::string() : t.substr(a);
            if (lower(t).rfind(".title", 0) == 0) t = t.substr(6);
            const size_t b = t.find_first_not_of(" \t");
            if (title) *title = (b == std::string::npos) ? std::string() : t.substr(b);
            continue;
        }
        size_t cut = ln.find(';');
        for (size_t k = 1; k < ln.size(); ++k)
            if (ln[k] == '$' && std::isspace((unsigned char)ln[k - 1])) { cut = std::min(cut, k); break; }
        if (cut != std::string::npos) ln = ln.substr(0, cut);
        const size_t a = ln.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        if (ln[a] == '*') continue;
        if (ln[a] == '+') {
            if (cards.empty()) fail("line " + std::to_string(no) + ": '+' continuation without a card");
            cards.back().text += " " + ln.substr(a + 1);
            continue;
        }
        Card c;
        c.text = ln.substr(a);
        c.line = no;
        cards.push_back(c);
    }
    for (Card& c : cards) c.tok = tokenize(c.text);
    return cards;
}

bool is_ground(const std::string& n) {
    const std::string l = lower(n);
    return l == "0" || l == "gnd";
}

// ОУ, которые заменяются идеальным нуллором (таблица замен плана).
// Выводы: 3 (in+, in-, out) или 5 (in+, in-, V+, V-, out — порядок моделей TI/ADI).
bool opamp_table(const std::string& name) {
    static const std::set<std::string> t = {
        "opamp", "op_amp", "ideal_opamp", "tl071", "tl072", "tl074", "tl081", "tl082", "tl084",
        "lm741", "ua741", "lm358", "lm324", "ne5532", "ne5534", "opa2134", "opa134", "lf356",
        "lf353", "ad8065", "ad711", "ad712", "op07", "op27", "ca3140",
    };
    return t.count(lower(name)) > 0;
}

struct Subckt {
    std::string name;
    std::vector<std::string> pins;                         // в нижнем регистре
    std::vector<std::pair<std::string, std::string>> defaults;
    std::vector<Card> cards;
};

std::string sanitize(const std::string& s) {
    std::string o;
    for (char c : lower(s)) o += std::isalnum((unsigned char)c) ? c : '_';
    while (!o.empty() && o.back() == '_') o.pop_back();
    size_t a = 0;
    while (a < o.size() && o[a] == '_') ++a;
    o = o.substr(a);
    if (o.empty()) o = "n";
    if (std::isdigit((unsigned char)o[0])) o = "n" + o;
    return o;
}

struct Model {
    std::string name, type;                     // type: d, npn, pnp, ...
    std::map<std::string, double> p;            // параметры в нижнем регистре
};

// Всё, что читает файл: подсхемы, модели, параметры, директивы. Общее для импорта в ДАУ
// и для кругового теста.
struct Deck {
    std::string title;
    std::vector<Card> top;                      // элементы верхнего уровня
    std::map<std::string, Subckt> subckts;      // ключ — имя в нижнем регистре
    std::map<std::string, Model> models;        // ключ — имя в нижнем регистре
    bool uic = false;                           // .tran ... UIC: без рабочей точки
    Scope global;
    double tstep = 0.0, tstop = 0.0;
    std::vector<std::pair<std::string, double>> ic_nodes;   // .ic V(узел) = ...
    std::set<std::string> ignored;
    std::string base_dir;
    int include_depth = 0;

    void load(const std::string& text, bool title_line) {
        std::vector<Card> cards = read_cards(text, title_line, title_line ? &title : nullptr);
        std::vector<Subckt*> open;
        bool control = false;
        for (Card& c : cards) {
            if (c.tok.empty()) continue;
            const std::string k = lower(c.tok[0]);
            if (control) { if (k == ".endc") control = false; continue; }
            if (k == ".control") { control = true; continue; }
            if (k == ".end") break;
            if (k == ".subckt") {
                if (c.tok.size() < 2) fail("line " + std::to_string(c.line) + ": .subckt without a name");
                Subckt s;
                s.name = c.tok[1];
                size_t i = 2;
                for (; i < c.tok.size(); ++i) {
                    const std::string t = lower(c.tok[i]);
                    if (t == "params:") { ++i; break; }
                    if (i + 1 < c.tok.size() && c.tok[i + 1] == "=") break;
                    s.pins.push_back(t);
                }
                for (; i + 2 < c.tok.size(); i += 3) {
                    if (c.tok[i + 1] != "=") fail("line " + std::to_string(c.line) + ": bad .subckt parameter list");
                    s.defaults.push_back({ lower(c.tok[i]), c.tok[i + 2] });
                }
                Subckt& slot = subckts[lower(s.name)];
                slot = s;
                open.push_back(&slot);
                continue;
            }
            if (k == ".ends") {
                if (open.empty()) fail("line " + std::to_string(c.line) + ": .ends without .subckt");
                open.pop_back();
                continue;
            }
            // .model внутри .subckt (так пишут вендорские макромодели) — в общую таблицу моделей:
            // область видимости по имени не различается.
            if (!open.empty()) {
                if (k == ".model") directive(c, k);
                else open.back()->cards.push_back(c);
                continue;
            }
            if (k[0] == '.') { directive(c, k); continue; }
            top.push_back(c);
        }
        if (!open.empty()) fail(".subckt " + open.back()->name + " has no .ends");
    }

    void directive(const Card& c, const std::string& k) {
        const auto& t = c.tok;
        const std::string at = "line " + std::to_string(c.line) + ": ";
        if (k == ".param") {
            for (size_t i = 1; i + 2 < t.size(); i += 3) {
                if (t[i + 1] != "=") fail(at + "bad .param");
                global.v[lower(t[i])] = eval_value(t[i + 2], global);
            }
        } else if (k == ".tran") {
            if (t.size() < 3) fail(at + ".tran needs TSTEP TSTOP");
            tstep = eval_value(t[1], global);
            tstop = eval_value(t[2], global);
            for (size_t i = 3; i < t.size(); ++i) if (lower(t[i]) == "uic") uic = true;
        } else if (k == ".ic") {
            for (size_t i = 1; i + 3 < t.size(); i += 4) {
                if (lower(t[i]) != "v" || t[i + 2] != "=") fail(at + ".ic expects V(node)=value");
                ic_nodes.push_back({ t[i + 1], eval_value(t[i + 3], global) });
            }
        } else if (k == ".model") {
            if (t.size() < 3) fail(at + ".model needs a name and a type");
            Model& m = models[lower(t[1])];
            m.name = t[1];
            m.type = lower(t[2]);
            for (size_t i = 3; i + 2 < t.size(); i += 3) {
                if (t[i + 1] != "=") fail(at + "bad .model parameter list");
                m.p[lower(t[i])] = eval_value(t[i + 2], global);
            }
        } else if (k == ".include" || k == ".inc" || k == ".lib") {
            if (t.size() < 2) fail(at + k + " without a file");
            std::filesystem::path p = std::filesystem::u8path(t[1]);
            if (p.is_relative() && !base_dir.empty()) p = std::filesystem::u8path(base_dir) / p;
            std::ifstream f(p, std::ios::binary);
            if (!f) fail(at + "cannot open " + p.u8string());
            if (++include_depth > 16) fail(at + ".include nested too deep");
            std::stringstream ss;
            ss << f.rdbuf();
            Deck sub;
            sub.base_dir = p.parent_path().u8string();
            sub.include_depth = include_depth;
            sub.global.v = global.v;
            sub.load(ss.str(), false);
            for (auto& kv : sub.subckts) subckts[kv.first] = kv.second;
            for (auto& kv : sub.models) models[kv.first] = kv.second;
            for (auto& kv : sub.global.v) global.v[kv.first] = kv.second;
            for (const Card& e : sub.top) top.push_back(e);
            --include_depth;
        } else if (k == ".title") {
            title = c.text.size() > 6 ? c.text.substr(7) : std::string();
        } else if (k == ".endl" || k == ".global") {
        } else ignored.insert(k);
    }
};

// --- сборка ДАУ ----------------------------------------------------------------------

struct Builder {
    const SpiceImportOptions& opt;
    Deck& deck;
    explicit Builder(const SpiceImportOptions& o, Deck& d) : opt(o), deck(d) {}

    std::set<std::string> used;
    std::vector<std::string> node_keys, node_disp, node_var;
    std::map<std::string, int> node_idx;
    std::vector<std::vector<std::string>> kcl;      // вытекающие токи по узлу

    std::vector<std::string> xs, x_rhs, zs, gs;
    std::vector<std::string> params;
    std::map<std::string, std::string> pvals, ic;
    std::map<std::string, std::string> branch;       // ключ элемента -> переменная тока
    std::vector<std::string> notes;

    struct Edge { int a, b; std::string name; };
    std::vector<Edge> v_edges;                        // C и ветви, заданные напряжением

    int n_r = 0, n_c = 0, n_l = 0, n_src = 0, n_ctl = 0, n_nullor = 0, n_b = 0, n_x = 0;

    std::string uniq(const std::string& base) {
        static const std::set<std::string> reserved = {
            "t", "pi", "e", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
            "exp", "log", "log2", "log10", "sqrt", "cbrt", "fabs", "abs", "pow", "atan2", "fmod",
            "floor", "ceil", "copysign", "fmin", "fmax", "min", "max", "h", "a", "x", "X",
        };
        std::string n = base;
        for (int k = 2; used.count(n) || reserved.count(n); ++k) n = base + "_" + std::to_string(k);
        used.insert(n);
        return n;
    }

    struct Ctx {
        std::string prefix;                           // "x1." внутри подсхемы
        std::map<std::string, std::string> pins;      // вывод подсхемы -> ключ внешнего узла
        const Scope* scope = nullptr;
        int depth = 0;
    };

    // Ключ узла: "" — земля.
    std::string node_key(const std::string& tok, const Ctx& c) {
        if (is_ground(tok)) return {};
        const std::string l = lower(tok);
        auto it = c.pins.find(l);
        if (it != c.pins.end()) return it->second;
        return c.prefix + l;
    }
    int node(const std::string& key, const std::string& disp) {
        if (key.empty()) return -1;
        auto it = node_idx.find(key);
        if (it != node_idx.end()) return it->second;
        const int i = (int)node_keys.size();
        node_idx[key] = i;
        node_keys.push_back(key);
        node_disp.push_back(disp);
        node_var.push_back(uniq("v_" + sanitize(key)));
        kcl.emplace_back();
        return i;
    }
    int node_of(const std::string& tok, const Ctx& c) {
        const std::string k = node_key(tok, c);
        return node(k, c.prefix.empty() ? tok : c.prefix + tok);
    }
    std::string V(int n) const { return n < 0 ? std::string("0") : node_var[(size_t)n]; }
    // Va - Vb без нулевых слагаемых; в скобках, если это разность.
    std::string vdiff(int a, int b) const {
        if (a < 0 && b < 0) return "0";
        if (b < 0) return V(a);
        if (a < 0) return "(-" + V(b) + ")";
        return "(" + V(a) + " - " + V(b) + ")";
    }
    std::string lin(int a, int b) const {
        if (a < 0 && b < 0) return "0";
        if (b < 0) return V(a);
        if (a < 0) return "-" + V(b);
        return V(a) + " - " + V(b);
    }
    // Ток cur течёт из a через элемент в b: из узла a вытекает, в b втекает.
    void current(int a, int b, const std::string& cur) {
        if (a >= 0) kcl[(size_t)a].push_back("+ " + cur);
        if (b >= 0) kcl[(size_t)b].push_back("- " + cur);
    }
    std::string param(const std::string& name, double v) {
        const std::string p = uniq(sanitize(name));
        params.push_back(p);
        pvals[p] = fmt_short(v);
        return p;
    }
    std::string branch_ref(const std::string& key) { return "\x01" + key + "\x02"; }

    // Выражение B / VALUE= в синтаксис U-CUDA: V(a), V(a,b) -> потенциалы, I(Vx) -> ток
    // ветви, time -> t, ** -> ^, ln -> log; параметры и числа с множителями — числами.
    std::string translate(const std::string& s, const Ctx& c, const std::string& at) {
        static const std::set<std::string> funcs = {
            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log",
            "log10", "sqrt", "abs", "pow", "atan2", "min", "max",
        };
        std::string o;
        size_t i = 0;
        while (i < s.size()) {
            const char ch = s[i];
            double v;
            if (std::isspace((unsigned char)ch) || ch == '{' || ch == '}' || ch == '\'') { ++i; continue; }
            if (read_number(s, i, v)) { o += fmt(v); continue; }
            if (ch == '*' && i + 1 < s.size() && s[i + 1] == '*') { o += '^'; i += 2; continue; }
            if (std::strchr("+-*/^(),", ch)) { o += ch; ++i; continue; }
            if (!ident_start(ch))
                fail(at + "'" + std::string(1, ch) + "' in an expression is not supported "
                     "(conditionals and comparisons are not)");
            size_t j = i;
            while (j < s.size() && ident_char(s[j])) ++j;
            const std::string id = lower(s.substr(i, j - i));
            size_t k = j;
            while (k < s.size() && std::isspace((unsigned char)s[k])) ++k;
            const bool call = k < s.size() && s[k] == '(';
            if (call && (id == "v" || id == "i")) {
                size_t e = s.find(')', k);
                if (e == std::string::npos) fail(at + "missing ')' after " + id + "(");
                std::vector<std::string> args = tokenize(s.substr(k + 1, e - k - 1));
                if (id == "v") {
                    if (args.empty() || args.size() > 2) fail(at + "V() takes one or two nodes");
                    const int a = node_of(args[0], c);
                    const int b = args.size() == 2 ? node_of(args[1], c) : -1;
                    o += vdiff(a, b);
                } else {
                    if (args.size() != 1) fail(at + "I() takes one source name");
                    o += branch_ref(c.prefix + lower(args[0]));
                }
                i = e + 1;
                continue;
            }
            if (call) {
                std::string f = id == "ln" ? "log" : id == "pwr" ? "pow" : id;
                if (!funcs.count(f)) fail(at + "function " + id + "() is not supported");
                o += f;
                i = j;
                continue;
            }
            if (id == "time") o += "t";
            else if (id == "pi") o += fmt(kPi);
            else if (c.scope->get(id, v)) o += "(" + fmt(v) + ")";
            else fail(at + "unknown symbol '" + id + "'");
            i = j;
        }
        return "(" + o + ")";
    }

    // Независимый источник: выражение по t. DC и параметры SIN — параметры a[].
    std::string source_expr(const Card& cd, size_t from, const Ctx& c, const std::string& name,
                            const std::string& at) {
        const auto& t = cd.tok;
        auto is_kw = [](const std::string& s) {
            const std::string l = lower(s);
            return l == "dc" || l == "ac" || l == "sin" || l == "pulse" || l == "pwl" || l == "exp"
                || l == "distof1" || l == "distof2" || l == "sffm" || l == "am" || l == "trrandom";
        };
        double dc = 0.0;
        std::string kind;
        std::vector<double> a;
        for (size_t i = from; i < t.size();) {
            const std::string l = lower(t[i]);
            if (!is_kw(t[i])) { dc = eval_value(t[i], *c.scope); ++i; continue; }
            ++i;
            std::vector<double> nums;
            while (i < t.size() && !is_kw(t[i])) nums.push_back(eval_value(t[i++], *c.scope));
            if (l == "dc") { if (!nums.empty()) dc = nums[0]; }
            else if (l == "ac" || l == "distof1" || l == "distof2") {}
            else if (l == "sin" || l == "pulse" || l == "pwl" || l == "exp") { kind = l; a = nums; }
            else fail(at + l + " sources are not supported");
        }
        auto arg = [&](size_t k, double def) { return k < a.size() ? a[k] : def; };
        const double tstep = deck.tstep, tstop = deck.tstop;
        auto clamp = [](const std::string& u) { return "min(max(" + u + ", 0), 1)"; };
        if (kind.empty()) return param(name, dc);
        if (kind == "sin") {
            if (a.size() < 2) fail(at + "SIN needs at least VO and VA");
            double f = arg(2, 0.0);
            if (f == 0.0) {
                if (tstop <= 0.0) fail(at + "SIN without FREQ needs .tran (FREQ = 1/TSTOP)");
                f = 1.0 / tstop;
            }
            const double td = arg(3, 0.0), th = arg(4, 0.0), ph = arg(5, 0.0);
            const std::string vo = param(name + "_vo", a[0]), va = param(name + "_va", a[1]),
                              fr = param(name + "_f", f);
            const std::string tt = td == 0.0 ? std::string("t") : "max(t - " + fmt(td) + ", 0)";
            std::string e = "(" + vo + " + " + va;
            if (th != 0.0) e += "*exp(-" + fmt(th) + "*" + tt + ")";
            e += "*sin(" + fmt(2.0 * kPi) + "*" + fr + "*" + tt;
            if (ph != 0.0) e += " + " + fmt(ph * kPi / 180.0);
            return e + "))";
        }
        if (kind == "pulse") {
            if (a.size() < 2) fail(at + "PULSE needs at least V1 and V2");
            const double dflt = tstep > 0.0 ? tstep : 1e-9;
            const double v1 = a[0], v2 = a[1], td = arg(2, 0.0);
            double tr = arg(3, dflt), tf = arg(4, dflt);
            if (tr <= 0.0) tr = dflt;
            if (tf <= 0.0) tf = dflt;
            const double pw = arg(5, tstop > 0.0 ? tstop : 1e30), per = arg(6, 0.0);
            int n = 1;
            if (per > 0.0 && tstop > td) n = (int)std::ceil((tstop - td) / per);
            if (per > 0.0 && tstop <= 0.0)
                notes.push_back(name + ": periodic PULSE without .tran - only the first pulse is kept");
            if (n > 200) fail(at + "PULSE has " + std::to_string(n) + " periods up to TSTOP (limit 200); use PWL "
                              "or a shorter .tran");
            if (n > 1) notes.push_back(name + ": PULSE unrolled into " + std::to_string(n)
                                       + " pulses up to TSTOP; past TSTOP the source stays at V1");
            std::string sum;
            for (int k = 0; k < n; ++k) {
                const double s0 = td + k * per;
                sum += (k ? " + " : "") + clamp("(t - " + fmt(s0) + ")/" + fmt(tr)) + " - "
                     + clamp("(t - " + fmt(s0 + tr + pw) + ")/" + fmt(tf));
            }
            return "(" + fmt(v1) + " + " + fmt(v2 - v1) + "*(" + sum + "))";
        }
        if (kind == "pwl") {
            if (a.size() < 2 || a.size() % 2) fail(at + "PWL needs time-value pairs");
            std::string e = "(" + fmt(a[1]);
            for (size_t k = 2; k + 1 < a.size(); k += 2) {
                const double dt = a[k] - a[k - 2];
                if (dt <= 0.0) fail(at + "PWL times must increase");
                e += " + " + fmt(a[k + 1] - a[k - 1]) + "*" + clamp("(t - " + fmt(a[k - 2]) + ")/" + fmt(dt));
            }
            return e + ")";
        }
        // EXP(V1 V2 TD1 TAU1 TD2 TAU2)
        if (a.size() < 2) fail(at + "EXP needs at least V1 and V2");
        const double dflt = tstep > 0.0 ? tstep : 1e-9;
        const double v1 = a[0], v2 = a[1], td1 = arg(2, 0.0), tau1 = arg(3, dflt),
                     td2 = arg(4, td1 + dflt), tau2 = arg(5, dflt);
        return "(" + fmt(v1) + " + " + fmt(v2 - v1) + "*(1 - exp(-max(t - " + fmt(td1) + ", 0)/" + fmt(tau1)
             + ")) + " + fmt(v1 - v2) + "*(1 - exp(-max(t - " + fmt(td2) + ", 0)/" + fmt(tau2) + ")))";
    }

    // POLY(n): коэффициенты в порядке SPICE 2 — по степеням, внутри степени неубывающие
    // наборы индексов в лексикографическом порядке (c0, c1 x1, c2 x2, c3 x1^2, c4 x1 x2, ...).
    std::string poly(const std::vector<std::string>& xs_in, const std::vector<double>& coef) {
        std::string e;
        size_t k = 0;
        const int n = (int)xs_in.size();
        for (int deg = 0; k < coef.size(); ++deg) {
            std::vector<int> idx((size_t)deg, 0);
            for (;;) {
                if (k >= coef.size()) break;
                const double cf = coef[k++];
                if (cf != 0.0) {
                    std::string m = fmt(cf);
                    for (int q : idx) m += "*" + xs_in[(size_t)q];
                    e += (e.empty() ? "" : " + ") + m;
                }
                // следующий неубывающий набор
                int p = deg - 1;
                while (p >= 0 && idx[(size_t)p] == n - 1) --p;
                if (p < 0) break;
                const int v = idx[(size_t)p] + 1;
                for (int q = p; q < deg; ++q) idx[(size_t)q] = v;
            }
        }
        return "(" + (e.empty() ? std::string("0") : e) + ")";
    }

    // Ветвь, заданная напряжением: z = ток i, связь lin(a, b) - u = 0.
    std::string vbranch(const std::string& key, const std::string& name, int a, int b, const std::string& u) {
        const std::string i = uniq("i_" + sanitize(name));
        zs.push_back(i);
        gs.push_back(lin(a, b) + " - " + u);
        current(a, b, i);
        branch[key] = i;
        v_edges.push_back({ a, b, name });
        vsrc_edges.push_back({ a, b, name });
        conduct(a, b);
        return i;
    }

    void nullor(const std::string& key, const std::string& name, int out_p, int out_n, int inp, int inn) {
        const std::string i = uniq("i_" + sanitize(name));
        zs.push_back(i);
        gs.push_back(lin(inp, inn));
        current(out_p, out_n, i);
        branch[key] = i;
        conduct(out_p, out_n);
        ++n_nullor;
    }

    // Конденсатор с номиналом p (выражение): x = vc, z = ic, vc' = ic/p, Va - Vb - vc = 0.
    // НУ: IC= элемента, иначе .ic V(a) - V(b) (неназванный узел — 0, как в SPICE с UIC).
    void capacitor(const std::string& name, int a, int b, const std::string& p, bool has_ic, double v0) {
        const std::string x = uniq("vc_" + sanitize(name)), i = uniq("ic_" + sanitize(name));
        xs.push_back(x);
        x_rhs.push_back(i + "/" + p);
        zs.push_back(i);
        gs.push_back(lin(a, b) + " - " + x);
        current(a, b, i);
        if (!has_ic) {
            double va = 0.0, vb = 0.0;
            for (const auto& kv : deck.ic_nodes) {
                const std::string k = node_key(kv.first, Ctx{});
                if (a >= 0 && k == node_keys[(size_t)a]) { va = kv.second; has_ic = true; }
                if (b >= 0 && k == node_keys[(size_t)b]) { vb = kv.second; has_ic = true; }
            }
            v0 = va - vb;
        }
        if (has_ic) any_ic = true;
        ic[x] = fmt_short(v0);
        v_edges.push_back({ a, b, name });
        touch(a);
        touch(b);
    }

    // Модель прибора по имени и проверка типа: "d" — D, "npn" — NPN/PNP, "njf" — NJF/PJF.
    const Model& model(const std::string& name, const char* kind, const std::string& at) {
        auto it = deck.models.find(lower(name));
        if (it == deck.models.end()) fail(at + "unknown model " + name);
        const std::string& ty = it->second.type;
        const std::string k = kind;
        const bool ok = k == "d" ? ty == "d" : k == "npn" ? (ty == "npn" || ty == "pnp") : (ty == "njf" || ty == "pjf");
        if (!ok) fail(at + "model " + name + " has type " + ty);
        return it->second;
    }

    // Параметры модели -> параметры a[] "<модель>_<имя>", один раз на модель. Нулевое умолчание
    // значит "нет" (RS, CJO, VAF, ...): такой параметр создаётся, только если модель его задала.
    // Заданные, но не моделируемые параметры — в отчёт.
    std::map<std::string, std::map<std::string, std::string>> model_cache;
    const std::map<std::string, std::string>& model_params(const Model& m, const std::vector<std::string>& keys,
                                                           const std::vector<double>& defs,
                                                           const std::vector<std::string>& ignored) {
        auto it = model_cache.find(lower(m.name));
        if (it != model_cache.end()) return it->second;
        std::map<std::string, std::string>& out = model_cache[lower(m.name)];
        for (size_t i = 0; i < keys.size(); ++i) {
            auto v = m.p.find(keys[i]);
            const double val = v != m.p.end() ? v->second : defs[i];
            if (defs[i] == 0.0 && val == 0.0) continue;
            out[keys[i]] = param(m.name + "_" + keys[i], val);
        }
        std::string skip;
        for (const auto& kv : m.p)
            if (std::find(ignored.begin(), ignored.end(), kv.first) != ignored.end()) skip += " " + kv.first;
            else if (std::find(keys.begin(), keys.end(), kv.first) == keys.end()) skip += " " + kv.first + "(?)";
        if (!skip.empty()) notes.push_back("model " + m.name + ": not modelled:" + skip);
        if (m.p.count("cjo") || m.p.count("cje") || m.p.count("cjc"))
            notes.push_back("model " + m.name + ": junction capacitances are taken as constant");
        return out;
    }

    // Внутренний узел за последовательным сопротивлением r/area (RS диода, RB/RC/RE транзистора).
    int internal(int n, const std::string& r, const std::string& name, const char* suf, double area) {
        if (r.empty()) return n;
        const std::string key = lower(name) + "#" + suf;
        const int j = node(key, name + "#" + suf);
        const std::string rr = area == 1.0 ? r : "(" + r + "/" + fmt(area) + ")";
        current(n, j, vdiff(n, j) + "/" + rr);
        conduct(n, j);
        return j;
    }

    // Топология: ветви с путём по постоянному току и узлы, к которым подключено что-то кроме
    // источников тока.
    std::vector<std::pair<int, int>> dc_edges;
    std::vector<Edge> vsrc_edges;
    std::set<int> solid;
    bool any_ic = false;
    int n_dev = 0;
    void touch(int n) { if (n >= 0) solid.insert(n); }
    void conduct(int a, int b) { dc_edges.push_back({ a, b }); touch(a); touch(b); }

    // Текст после имени и двух узлов: "V = ..." у B, "VALUE = ..." у E/G.
    static std::string rest_after_words(const std::string& text, int words) {
        size_t i = 0;
        for (int w = 0; w < words; ++w) {
            while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
            while (i < text.size() && !std::isspace((unsigned char)text[i])) ++i;
        }
        while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
        return text.substr(i);
    }
    static bool take_assign(std::string& r, const std::string& kw) {
        if (lower(r.substr(0, kw.size())) != kw) return false;
        size_t i = kw.size();
        while (i < r.size() && std::isspace((unsigned char)r[i])) ++i;
        if (i >= r.size() || r[i] != '=') return false;
        r = r.substr(i + 1);
        return true;
    }

    void element(const Card& cd, const Ctx& c) {
        const auto& t = cd.tok;
        const std::string at = "line " + std::to_string(cd.line) + " (" + t[0] + "): ";
        const char type = (char)std::tolower((unsigned char)t[0][0]);
        const std::string name = c.prefix + t[0];
        const std::string key = c.prefix + lower(t[0]);
        auto need = [&](size_t n) { if (t.size() < n) fail(at + "too few fields"); };
        auto value_at = [&](size_t i, const char* kw) {
            need(i + 1);
            if (lower(t[i]) == kw && i + 2 < t.size() && t[i + 1] == "=") return eval_value(t[i + 2], *c.scope);
            return eval_value(t[i], *c.scope);
        };
        auto ic_of = [&](double* v) {
            for (size_t i = 3; i + 2 < t.size(); ++i)
                if (lower(t[i]) == "ic" && t[i + 1] == "=") { *v = eval_value(t[i + 2], *c.scope); return true; }
            return false;
        };
        switch (type) {
        case 'r': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            const double r = value_at(3, "r");
            if (r == 0.0) fail(at + "zero resistance");
            current(a, b, vdiff(a, b) + "/" + param(name, r));
            conduct(a, b);
            ++n_r;
            break;
        }
        case 'c': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            const double cv = value_at(3, "c");
            if (cv == 0.0) fail(at + "zero capacitance");
            double v0 = 0.0;
            const bool has = ic_of(&v0);
            capacitor(name, a, b, param(name, cv), has, v0);
            ++n_c;
            break;
        }
        case 'd': {
            need(4);
            const int a = node_of(t[1], c), k = node_of(t[2], c);
            const Model& m = model(t[3], "d", at);
            const double area = t.size() > 4 && t[4] != "=" && lower(t[4]) != "off" && lower(t[4]) != "ic"
                              ? eval_value(lower(t[4]) == "area" && t.size() > 6 ? t[6] : t[4], *c.scope) : 1.0;
            const auto& mp = model_params(m, { "is", "n", "rs", "cjo" }, { 1e-14, 1, 0, 0 },
                                          { "bv", "ibv", "tt", "vj", "m", "eg", "xti", "kf", "af", "fc" });
            auto sc = [&](const std::string& p) { return area == 1.0 ? p : "(" + fmt(area) + "*" + p + ")"; };
            const int j = internal(a, mp.count("rs") ? mp.at("rs") : std::string(), name, "a", area);
            current(j, k, diode_current(vdiff(j, k), sc(mp.at("is")), mp.at("n")));
            conduct(j, k);
            if (mp.count("cjo")) capacitor(name + "_cj", j, k, sc(mp.at("cjo")), false, 0.0);
            ++n_dev;
            break;
        }
        case 'q': {
            need(5);
            // Q c b e [s] model [area]
            size_t mi = 4;
            if (!deck.models.count(lower(t[4])) && t.size() > 5) mi = 5;
            const Model& m = model(t[mi], "npn", at);
            const bool pnp = m.type == "pnp";
            if (!pnp && m.type != "npn") fail(at + "model " + m.name + " is not NPN/PNP");
            const double area = t.size() > mi + 1 && t[mi + 1] != "=" && lower(t[mi + 1]) != "off"
                              ? eval_value(lower(t[mi + 1]) == "area" && t.size() > mi + 3 ? t[mi + 3] : t[mi + 1], *c.scope) : 1.0;
            const auto& mp = model_params(m, { "is", "bf", "br", "nf", "nr", "vaf", "var", "rb", "rc", "re", "cje", "cjc" },
                                          { 1e-16, 100, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 },
                                          { "ikf", "ikr", "ise", "isc", "ne", "nc", "tf", "tr", "vje", "vjc", "mje", "mjc",
                                            "xtf", "itf", "vtf", "ptf", "eg", "xti", "xtb", "irb", "rbm", "cjs", "kf", "af",
                                            "fc", "nk", "iss", "ns", "bvcbo" });
            auto sc = [&](const std::string& p) { return area == 1.0 ? p : "(" + fmt(area) + "*" + p + ")"; };
            auto opt_p = [&](const char* k) { return mp.count(k) ? mp.at(k) : std::string(); };
            const int nc = internal(node_of(t[1], c), opt_p("rc"), name, "c", area);
            const int nb = internal(node_of(t[2], c), opt_p("rb"), name, "b", area);
            const int ne = internal(node_of(t[3], c), opt_p("re"), name, "e", area);
            auto inv = [&](const char* k) { return mp.count(k) ? "(1/" + mp.at(k) + ")" : std::string(); };
            // NPN: vbe, vbc; PNP: veb, vcb и токи обратного знака (из коллектора и базы).
            const std::string v1 = pnp ? vdiff(ne, nb) : vdiff(nb, ne), v2 = pnp ? vdiff(nc, nb) : vdiff(nb, nc);
            const BjtCurrents q = bjt_currents(v1, v2, sc(mp.at("is")), mp.at("bf"), mp.at("br"), mp.at("nf"),
                                               mp.at("nr"), inv("vaf"), inv("var"));
            if (pnp) { current(ne, nc, q.ic); current(ne, nb, q.ib); }
            else     { current(nc, ne, q.ic); current(nb, ne, q.ib); }
            conduct(nc, ne);
            conduct(nb, ne);
            if (mp.count("cje")) capacitor(name + "_cje", nb, ne, sc(mp.at("cje")), false, 0.0);
            if (mp.count("cjc")) capacitor(name + "_cjc", nb, nc, sc(mp.at("cjc")), false, 0.0);
            ++n_dev;
            break;
        }
        case 'l': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            const double lv = value_at(3, "l");
            if (lv == 0.0) fail(at + "zero inductance");
            const std::string p = param(name, lv);
            const std::string x = uniq("il_" + sanitize(name));
            xs.push_back(x);
            x_rhs.push_back(vdiff(a, b) + "/" + p);
            current(a, b, x);
            double v0 = 0.0;
            if (ic_of(&v0)) any_ic = true;
            else v0 = 0.0;
            ic[x] = fmt_short(v0);
            branch[key] = x;
            conduct(a, b);
            ++n_l;
            break;
        }
        case 'v': {
            need(3);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            vbranch(key, name, a, b, source_expr(cd, 3, c, name, at));
            ++n_src;
            break;
        }
        case 'i': {
            need(3);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            current(a, b, source_expr(cd, 3, c, name, at));
            ++n_src;
            break;
        }
        case 'e':
        case 'g': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            std::string u;
            std::string rest = rest_after_words(cd.text, 3);
            if (take_assign(rest, "value")) u = translate(rest, c, at);
            else if (lower(t[3]) == "poly") {
                need(5);
                const int n = (int)eval_value(t[4], *c.scope);
                if (n < 1 || t.size() < (size_t)(5 + 2 * n)) fail(at + "bad POLY");
                std::vector<std::string> ctl;
                for (int k = 0; k < n; ++k)
                    ctl.push_back(vdiff(node_of(t[5 + 2 * k], c), node_of(t[6 + 2 * k], c)));
                std::vector<double> coef;
                for (size_t k = (size_t)(5 + 2 * n); k < t.size(); ++k) coef.push_back(eval_value(t[k], *c.scope));
                u = poly(ctl, coef);
            } else {
                need(6);
                const int cp = node_of(t[3], c), cn = node_of(t[4], c);
                const double gain = eval_value(t[5], *c.scope);
                if (type == 'e' && opt.vcvs_as_nullor && std::fabs(gain) >= opt.nullor_gain_min) {
                    // V(out) = A (V+ - V-) при A -> inf: V+ = V-, ток выхода свободен.
                    nullor(key, name, a, b, cp, cn);
                    break;
                }
                u = param(name, gain) + "*" + vdiff(cp, cn);
            }
            if (type == 'e') vbranch(key, name, a, b, u);
            else current(a, b, u);
            ++n_ctl;
            break;
        }
        case 'f':
        case 'h': {
            need(5);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            std::string u;
            if (lower(t[3]) == "poly") {
                // POLY(n) Vc1 .. Vcn c0 c1 ...: управляющие — токи источников напряжения
                const int n = (int)eval_value(t[4], *c.scope);
                if (n < 1 || t.size() < (size_t)(5 + n)) fail(at + "bad POLY");
                std::vector<std::string> ctl;
                for (int k = 0; k < n; ++k) ctl.push_back(branch_ref(c.prefix + lower(t[(size_t)(5 + k)])));
                std::vector<double> coef;
                for (size_t k = (size_t)(5 + n); k < t.size(); ++k) coef.push_back(eval_value(t[k], *c.scope));
                u = poly(ctl, coef);
            } else u = param(name, eval_value(t[4], *c.scope)) + "*" + branch_ref(c.prefix + lower(t[3]));
            if (type == 'h') vbranch(key, name, a, b, u);
            else current(a, b, u);
            ++n_ctl;
            break;
        }
        case 'b': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            std::string rest = rest_after_words(cd.text, 3);
            if (take_assign(rest, "v")) vbranch(key, name, a, b, translate(rest, c, at));
            else if (take_assign(rest, "i")) current(a, b, translate(rest, c, at));
            else fail(at + "B source needs V = ... or I = ...");
            ++n_b;
            break;
        }
        case 'x': instance(cd, c, at); break;
        case 'j': {
            need(5);
            // J d g s model [area]
            const Model& m = model(t[4], "njf", at);
            const bool pch = m.type == "pjf";
            const double area = t.size() > 5 && t[5] != "=" && lower(t[5]) != "off"
                              ? eval_value(lower(t[5]) == "area" && t.size() > 7 ? t[7] : t[5], *c.scope) : 1.0;
            const auto& mp = model_params(m, { "vto", "beta", "lambda", "is", "rd", "rs", "cgs", "cgd" },
                                          { -2.0, 1e-4, 0, 1e-14, 0, 0, 0, 0 },
                                          { "b", "pb", "fc", "kf", "af", "n", "isr", "nr", "alpha", "vk", "m",
                                            "tnom", "xti", "vtotc", "betatce" });
            auto sc = [&](const std::string& p) { return area == 1.0 ? p : "(" + fmt(area) + "*" + p + ")"; };
            auto opt_p = [&](const char* k) { return mp.count(k) ? mp.at(k) : std::string(); };
            const int nd = internal(node_of(t[1], c), opt_p("rd"), name, "d", area);
            const int ng = node_of(t[2], c);
            const int ns = internal(node_of(t[3], c), opt_p("rs"), name, "s", area);
            // P-канал: те же формулы от vsg, vsd, vdg, токи обратного знака.
            const std::string vgs = pch ? vdiff(ns, ng) : vdiff(ng, ns), vds = pch ? vdiff(ns, nd) : vdiff(nd, ns),
                              vgd = pch ? vdiff(nd, ng) : vdiff(ng, nd);
            const std::string id = jfet_drain_current(vgs, vds, vgd, mp.at("vto"), sc(mp.at("beta")), opt_p("lambda"));
            const std::string dgs = diode_current(vgs, sc(mp.at("is")), "1"), dgd = diode_current(vgd, sc(mp.at("is")), "1");
            if (pch) { current(ns, nd, id); current(ns, ng, dgs); current(nd, ng, dgd); }
            else     { current(nd, ns, id); current(ng, ns, dgs); current(ng, nd, dgd); }
            conduct(nd, ns);
            conduct(ng, ns);
            if (mp.count("cgs")) capacitor(name + "_cgs", ng, ns, sc(mp.at("cgs")), false, 0.0);
            if (mp.count("cgd")) capacitor(name + "_cgd", ng, nd, sc(mp.at("cgd")), false, 0.0);
            ++n_dev;
            break;
        }
        case 'm': case 'z':
            fail(at + "MOSFET / MESFET are not supported yet");
        default:
            fail(at + "element type '" + std::string(1, type) + "' is not supported");
        }
    }

    void instance(const Card& cd, const Ctx& c, const std::string& at) {
        const auto& t = cd.tok;
        size_t end = t.size();
        for (size_t i = 1; i < t.size(); ++i) {
            if (lower(t[i]) == "params:") { end = i; break; }
            if (i + 1 < t.size() && t[i + 1] == "=") { end = i; break; }
        }
        if (end < 3) fail(at + "subcircuit instance needs nodes and a name");
        const std::string sub = t[end - 1];
        std::vector<std::string> nodes(t.begin() + 1, t.begin() + (long long)(end - 1));
        size_t pi = end < t.size() && lower(t[end]) == "params:" ? end + 1 : end;
        std::vector<std::pair<std::string, std::string>> over;
        for (; pi + 2 < t.size(); pi += 3) {
            if (t[pi + 1] != "=") fail(at + "bad instance parameter list");
            over.push_back({ lower(t[pi]), t[pi + 2] });
        }
        const std::string name = c.prefix + t[0];
        const std::string key = c.prefix + lower(t[0]);
        // ОУ: "как есть" — подсхема из netlist'а разворачивается по элементам; идеальным он
        // становится, если так просили (ideal_opamps) или подсхемы в файле нет.
        const bool defined = deck.subckts.count(lower(sub)) > 0;
        const bool is_opamp = opamp_table(sub) || lower(sub) == "uc_opamp";
        if (is_opamp && (opt.ideal_opamps || !defined)) {
            if (nodes.size() != 3 && nodes.size() != 5)
                fail(at + sub + " as an op-amp needs 3 (in+ in- out) or 5 (in+ in- V+ V- out) pins");
            const int inp = node_of(nodes[0], c), inn = node_of(nodes[1], c), out = node_of(nodes.back(), c);
            nullor(key, name, out, -1, inp, inn);
            notes.push_back(name + ": " + sub + " replaced by an ideal op-amp (nullor)"
                            + std::string(defined ? "" : ", no .subckt for it in the netlist")
                            + (nodes.size() == 5 ? "; supply pins ignored" : ""));
            ++n_x;
            return;
        }
        if (is_opamp) notes.push_back(name + ": " + sub + " imported as is (its .subckt model)");
        auto it = deck.subckts.find(lower(sub));
        if (it == deck.subckts.end()) fail(at + "unknown subcircuit " + sub);
        const Subckt& s = it->second;
        if (s.pins.size() != nodes.size())
            fail(at + sub + " has " + std::to_string(s.pins.size()) + " pins, the instance gives "
                 + std::to_string(nodes.size()));
        if (c.depth > 20) fail(at + "subcircuits nested too deep");
        Ctx in;
        in.prefix = key + ".";
        in.depth = c.depth + 1;
        for (size_t k = 0; k < nodes.size(); ++k) in.pins[s.pins[k]] = node_key(nodes[k], c);
        // узлы-выводы должны существовать снаружи
        for (size_t k = 0; k < nodes.size(); ++k) node_of(nodes[k], c);
        auto sc = std::make_unique<Scope>();
        sc->parent = &deck.global;
        for (const auto& kv : s.defaults) sc->v[kv.first] = eval_value(kv.second, *sc);
        for (const auto& kv : over) sc->v[kv.first] = eval_value(kv.second, *c.scope);
        in.scope = sc.get();
        for (const Card& e : s.cards) {
            if (e.tok.empty()) continue;
            const std::string k = lower(e.tok[0]);
            if (k == ".param") {
                for (size_t i = 1; i + 2 < e.tok.size(); i += 3)
                    if (e.tok[i + 1] == "=") sc->v[lower(e.tok[i])] = eval_value(e.tok[i + 2], *sc);
                continue;
            }
            if (k[0] == '.') { deck.ignored.insert(k); continue; }
            element(e, in);
        }
        scopes.push_back(std::move(sc));
        ++n_x;
    }
    std::vector<std::unique_ptr<Scope>> scopes;

    // Контур из конденсаторов и ветвей, заданных напряжением, — индекс 2.
    void loops() {
        std::vector<int> par(node_keys.size() + 1);
        for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
        std::function<int(int)> find = [&](int x) { return par[(size_t)x] == x ? x : par[(size_t)x] = find(par[(size_t)x]); };
        for (const Edge& e : v_edges) {
            const int a = find(e.a + 1), b = find(e.b + 1);
            if (a == b) notes.push_back("loop of capacitors / voltage sources closed by " + e.name
                                        + ": index 2 (their voltages are not independent)");
            else par[(size_t)a] = b;
        }
        // Контур из одних источников напряжения — матрица вырождена при любом методе.
        for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
        for (const Edge& e : vsrc_edges) {
            const int a = find(e.a + 1), b = find(e.b + 1);
            if (a == b) fail("loop of voltage sources closed by " + e.name + ": their voltages contradict each other "
                             "or the branch currents are undetermined");
            par[(size_t)a] = b;
        }
        // Узел, к которому подходят только источники тока: потенциал не определён.
        for (size_t n = 0; n < node_keys.size(); ++n)
            if (!solid.count((int)n)) fail("node " + node_disp[n] + " is connected only to current sources");
        // Путь к земле по постоянному току: без него рабочая точка не определена (переходный
        // процесс при этом считается — потенциал держат конденсаторы).
        for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
        for (const auto& e : dc_edges) par[(size_t)find(e.first + 1)] = find(e.second + 1);
        std::string floating;
        for (size_t n = 0; n < node_keys.size(); ++n)
            if (find((int)n + 1) != find(0)) floating += " " + node_disp[n];
        if (!floating.empty())
            notes.push_back("no DC path to ground:" + floating + " (the DC operating point is undefined)");
    }

    std::string resolve(std::string s) const {
        for (;;) {
            const size_t a = s.find('\x01');
            if (a == std::string::npos) return s;
            const size_t b = s.find('\x02', a);
            const std::string k = s.substr(a + 1, b - a - 1);
            auto it = branch.find(k);
            if (it == branch.end()) fail("I(" + k + "): no voltage source, inductor or voltage-controlled branch with this name");
            s = s.substr(0, a) + it->second + s.substr(b + 1);
        }
    }

    void finish(SpiceImportResult& r) {
        loops();
        std::ostringstream eq;
        for (size_t i = 0; i < xs.size(); ++i) eq << xs[i] << "' = " << resolve(x_rhs[i]) << "\n";
        for (size_t n = 0; n < node_keys.size(); ++n) {
            std::string s;
            for (const std::string& term : kcl[n]) s += (s.empty() ? (term[0] == '+' ? term.substr(2) : "-" + term.substr(2))
                                                                   : " " + term);
            if (s.empty()) fail("node " + node_disp[n] + " is not connected");
            eq << "0 = " << resolve(s) << "\n";
        }
        for (const std::string& g : gs) eq << "0 = " << resolve(g) << "\n";
        r.plain_text = eq.str();

        auto join = [](const std::vector<std::string>& v) {
            std::string s;
            for (const std::string& x : v) s += (s.empty() ? "" : ", ") + x;
            return s;
        };
        std::vector<std::string> alg = node_var;
        alg.insert(alg.end(), zs.begin(), zs.end());
        r.vars_text = join(xs);
        r.alg_text = join(alg);
        r.params_text = join(params);
        r.param_values = pvals;
        r.init_conditions = ic;
        for (const auto& kv : deck.ic_nodes) {
            auto it = node_idx.find(node_key(kv.first, Ctx{}));
            if (it != node_idx.end()) r.init_conditions[node_var[(size_t)it->second]] = fmt_short(kv.second);
        }
        for (const std::string& z : alg) r.init_conditions.emplace(z, "0");
        if (deck.tstep > 0.0) r.step_h = fmt_short(deck.tstep);
        if (deck.tstop > 0.0) r.t_stop = fmt_short(deck.tstop);
        for (size_t n = 0; n < node_keys.size(); ++n) r.node_vars.push_back({ node_disp[n], node_var[n] });
        r.notes = notes;
        if (!deck.ignored.empty()) {
            std::string s = "ignored directives:";
            for (const std::string& d : deck.ignored) s += " " + d;
            r.notes.push_back(s);
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "%d nodes; R %d, C %d, L %d, sources %d, controlled %d, B %d, op-amps as nullors %d, "
                      "semiconductors %d, subcircuits %d; DAE: %d differential, %d algebraic, %d parameters",
                      (int)node_keys.size(), n_r, n_c, n_l, n_src, n_ctl, n_b, n_nullor, n_dev, n_x,
                      (int)xs.size(), (int)alg.size(), (int)params.size());
        r.summary = buf;
        r.title = deck.title;
        r.uic = deck.uic;
        r.has_ic = any_ic || !deck.ic_nodes.empty();
    }
};

}  // namespace

bool spice_import(const std::string& netlist, const SpiceImportOptions& opt,
                  SpiceImportResult& out, std::string* err) {
    try {
        Deck deck;
        deck.base_dir = opt.base_dir;
        deck.load(netlist, true);
        Builder b(opt, deck);
        Builder::Ctx top;
        top.scope = &deck.global;
        for (const Card& c : deck.top) b.element(c, top);
        if (b.node_keys.empty()) throw SpiceError("the netlist has no elements");
        if (b.xs.empty()) throw SpiceError("the circuit has no capacitors or inductors - nothing to integrate");
        out = SpiceImportResult{};
        b.finish(out);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// --- обратный путь для кругового теста ------------------------------------------------

namespace {

// "V(a)" | "(-V(b))" | "(V(a)-V(b))" | "0" (так пишет ndiff эмиттера) -> пара узлов.
bool parse_ndiff(const std::string& s, size_t& i, std::string& a, std::string& b) {
    auto take_v = [&](std::string& n) {
        if (s.compare(i, 2, "V(") && s.compare(i, 2, "v(")) return false;
        const size_t e = s.find(')', i);
        if (e == std::string::npos) return false;
        n = s.substr(i + 2, e - i - 2);
        i = e + 1;
        return true;
    };
    a = b = "0";
    if (s.compare(i, 1, "0") == 0) { ++i; return true; }
    if (s.compare(i, 3, "(-V") == 0 || s.compare(i, 3, "(-v") == 0) {
        ++i; ++i;
        if (!take_v(b) || i >= s.size() || s[i] != ')') return false;
        ++i;
        return true;
    }
    if (s[i] == '(') {
        ++i;
        if (!take_v(a) || i >= s.size() || s[i] != '-') return false;
        ++i;
        if (!take_v(b) || i >= s.size() || s[i] != ')') return false;
        ++i;
        return true;
    }
    return take_v(a);
}

}  // namespace

bool spice_to_circuit_graph(const std::string& netlist, CircuitGraph& g, std::string* err) {
    try {
        Deck deck;
        deck.load(netlist, true);
        g = CircuitGraph{};
        g.add_node("gnd");
        std::map<std::string, int> idx;
        auto nd = [&](const std::string& n) {
            if (is_ground(n)) return 0;
            const std::string k = lower(n);
            auto it = idx.find(k);
            if (it != idx.end()) return it->second;
            const int i = g.add_node(n);
            idx[k] = i;
            return i;
        };
        for (const Card& c : deck.top) {
            const auto& t = c.tok;
            const std::string at = "line " + std::to_string(c.line) + " (" + t[0] + "): ";
            const char type = (char)std::tolower((unsigned char)t[0][0]);
            if (t.size() < 4) fail(at + "too few fields");
            if (type == 'r') {
                CircuitResistor r;
                r.a = nd(t[1]); r.b = nd(t[2]);
                r.siemens = 1.0 / eval_value(t[3], deck.global);
                g.resistors.push_back(r);
            } else if (type == 'c') {
                CircuitCapacitor cp;
                cp.a = nd(t[1]); cp.b = nd(t[2]);
                cp.farad = eval_value(t[3], deck.global);
                g.capacitors.push_back(cp);
            } else if (type == 'v') {
                if (!is_ground(t[2])) fail(at + "only sources to ground");
                CircuitVSource v;
                v.node = nd(t[1]);
                v.volt = eval_value(lower(t[3]) == "dc" && t.size() > 4 ? t[4] : t[3], deck.global);
                g.vsources.push_back(v);
            } else if (type == 'x') {
                if (t.size() != 5 || (lower(t[4]) != "uc_opamp" && !opamp_table(t[4])))
                    fail(at + "only 3-pin op-amps");
                CircuitOpAmp o;
                o.in_plus = nd(t[1]); o.in_minus = nd(t[2]); o.out = nd(t[3]);
                g.opamps.push_back(o);
            } else if (type == 'b' || (type == 'e' && lower(t[3]) == "poly")) {
                if (!is_ground(t[2])) fail(at + "multiplier output must be referenced to ground");
                CircuitMultiplier m;
                m.out = nd(t[1]);
                if (type == 'e') {
                    const int n = std::atoi(t[4].c_str());
                    if ((n != 2 && n != 3) || t.size() < (size_t)(5 + 2 * n)) fail(at + "bad POLY multiplier");
                    m.x1 = nd(t[5]); m.x2 = nd(t[6]); m.y1 = nd(t[7]); m.y2 = nd(t[8]);
                    if (n == 3) m.z = nd(t[9]);
                } else {
                    std::string e;
                    for (char ch : Builder::rest_after_words(c.text, 3)) if (!std::isspace((unsigned char)ch)) e += ch;
                    if (!Builder::take_assign(e, "v")) fail(at + "not a multiplier");
                    size_t i = 0;
                    std::string x1, x2, y1, y2;
                    if (!parse_ndiff(e, i, x1, x2) || i >= e.size() || e[i] != '*') fail(at + "not a multiplier");
                    ++i;
                    if (!parse_ndiff(e, i, y1, y2) || e.compare(i, 3, "/10")) fail(at + "not a multiplier");
                    i += 3;
                    m.x1 = nd(x1); m.x2 = nd(x2); m.y1 = nd(y1); m.y2 = nd(y2);
                    if (i < e.size()) {
                        std::string z, z2;
                        if (e[i] != '+') fail(at + "not a multiplier");
                        ++i;
                        if (!parse_ndiff(e, i, z, z2) || i != e.size()) fail(at + "not a multiplier");
                        m.z = nd(z);
                    }
                }
                g.multipliers.push_back(m);
            } else fail(at + "element is not part of a synthesized circuit");
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}
