#include "spice_parse.h"
#include "dae_devices.h"
#include "num_parse.h"

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

// Имя узла без префикса KiCad для неназванных цепей: "net-_led1-k_" (так kicad-cli пишет
// Net-(LED1-K)) -> "_led1-k_", sanitize дальше даёт led1_k. Ключи — в нижнем регистре; префикс
// подсхемы "x1." остаётся.
std::string net_short(const std::string& key) {
    const size_t dot = key.rfind('.');
    const std::string pre = dot == std::string::npos ? std::string() : key.substr(0, dot + 1);
    std::string n = dot == std::string::npos ? key : key.substr(dot + 1);
    if (n.size() > 4 && n.compare(0, 4, "net-") == 0) n = n.substr(4);
    return pre + n;
}

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

// Параметры модели хранятся текстом и вычисляются при использовании (Builder::model) в области
// видимости экземпляра: вендорские подсхемы пишут Is={Ices/2} с параметрами .subckt, а библиотеки
// KiCad держат модели XSPICE и с текстовыми полями (mfg=Diotec) — неиспользуемая модель не мешает.
struct Model {
    std::string name, type;                     // type: d, npn, pnp, ...
    std::map<std::string, double> p;            // вычисленные параметры (у экземпляра Builder::model)
    std::map<std::string, std::string> raw;     // как в файле, ключи в нижнем регистре
    std::set<std::string> flags;                // слова без значения (nchan, pchan)
    std::set<std::string> bad;                  // не вычислились (у экземпляра)
    bool in_subckt = false;
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
                if (k == ".model") {
                    directive(c, k);
                    if (c.tok.size() > 1) models[lower(c.tok[1])].in_subckt = true;
                }
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
            m = Model{};
            m.name = t[1];
            m.type = lower(t[2]);
            for (size_t i = 3; i < t.size();) {
                if (i + 2 < t.size() && t[i + 1] == "=") { m.raw[lower(t[i])] = t[i + 2]; i += 3; }
                else m.flags.insert(lower(t[i++]));   // nchan / pchan у VDMOS, скобки сняты токенизатором
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

// --- условия в выражениях B / VALUE= ----------------------------------------------------
// Сравнения, && || !, c ? a : b, if(c, a, b), u(), uramp(), limit(), sgn() — без ветвлений: истина — 1,
// ложь — 0 через ступеньку st(u) = (1 + copysign(1, u))/2 (st(+0) = 1). a >= b: st(a - b); a > b:
// 1 - st(b - a); a == b: st(a - b) st(b - a) (x - x = +0, равенство точное); c ? a : b: b + (a - b) c.
// Производная ступеньки — 0 (неявные методы видят выбранную ветвь), сама функция разрывна, как в SPICE.
// Вход — токены translate() (атомы уже в синтаксисе U-CUDA, операторы — отдельными токенами).
struct CondRewriter {
    const std::vector<std::string>& tk;
    const std::string& at;
    size_t p = 0;
    struct V { std::string s; bool boolean; };

    static std::string step(const std::string& u) { return "(0.5 + 0.5*copysign(1, " + u + "))"; }
    static std::string truth(const V& v) {   // число -> истинность (не ноль)
        return v.boolean ? v.s : "(1 - " + step(v.s) + "*" + step("-" + v.s) + ")";
    }
    bool is(const char* op) const { return p < tk.size() && tk[p] == op; }
    void expect(const char* op) {
        if (!is(op)) fail(at + "expected '" + op + "' in a conditional expression");
        ++p;
    }
    V expr() {
        V c = lor();
        if (!is("?")) return c;
        ++p;
        const V a = expr();
        expect(":");
        const V b = expr();
        return { "(" + b.s + " + (" + a.s + " - " + b.s + ")*" + truth(c) + ")", a.boolean && b.boolean };
    }
    V lor() {
        V a = land();
        while (is("||")) { ++p; const V b = land(); a = { "max(" + truth(a) + ", " + truth(b) + ")", true }; }
        return a;
    }
    V land() {
        V a = cmp();
        while (is("&&")) { ++p; const V b = cmp(); a = { "(" + truth(a) + "*" + truth(b) + ")", true }; }
        return a;
    }
    V cmp() {
        const V a = add();
        for (const char* op : { "<=", ">=", "==", "!=", "<", ">" })
            if (is(op)) {
                ++p;
                const V b = add();
                const std::string d = "(" + a.s + " - " + b.s + ")", r = "(" + b.s + " - " + a.s + ")";
                const std::string o = op;
                std::string s;
                if (o == ">=") s = step(d);
                else if (o == "<=") s = step(r);
                else if (o == ">") s = "(1 - " + step(r) + ")";
                else if (o == "<") s = "(1 - " + step(d) + ")";
                else if (o == "==") s = "(" + step(d) + "*" + step(r) + ")";
                else s = "(1 - " + step(d) + "*" + step(r) + ")";
                return { s, true };
            }
        return a;
    }
    V add() {
        V a = mul();
        while (is("+") || is("-")) { const std::string op = tk[p++]; const V b = mul(); a = { "(" + a.s + " " + op + " " + b.s + ")", false }; }
        return a;
    }
    V mul() {
        V a = unary();
        while (is("*") || is("/")) { const std::string op = tk[p++]; const V b = unary(); a = { "(" + a.s + op + b.s + ")", false }; }
        return a;
    }
    V unary() {
        if (is("-")) { ++p; const V a = unary(); return { "(-" + a.s + ")", false }; }
        if (is("+")) { ++p; return unary(); }
        if (is("!")) { ++p; const V a = unary(); return { "(1 - " + truth(a) + ")", true }; }
        return power();
    }
    V power() {
        const V a = primary();
        if (!is("^")) return a;
        ++p;
        const V b = unary();
        return { "(" + a.s + "^" + b.s + ")", false };
    }
    std::vector<V> args() {
        expect("(");
        std::vector<V> v;
        if (is(")")) { ++p; return v; }
        for (;;) {
            v.push_back(expr());
            if (is(",")) { ++p; continue; }
            expect(")");
            return v;
        }
    }
    V primary() {
        if (p >= tk.size()) fail(at + "unexpected end of expression");
        const std::string t = tk[p++];
        if (t == "(") { const V v = expr(); expect(")"); return { "(" + v.s + ")", v.boolean }; }
        static const std::set<std::string> ops = { ")", ",", "?", ":", "*", "/", "^", "&&", "||", "<", ">", "<=", ">=",
                                                   "==", "!=" };
        if (ops.count(t)) fail(at + "unexpected '" + t + "'");
        if (!is("(")) return { t, false };   // атом: число, параметр, V(..), ток ветви, t
        const std::vector<V> a = args();
        auto need = [&](size_t n) { if (a.size() != n) fail(at + t + "() takes " + std::to_string(n) + " argument(s)"); };
        if (t == "if") { need(3); return { "(" + a[2].s + " + (" + a[1].s + " - " + a[2].s + ")*" + truth(a[0]) + ")", false }; }
        if (t == "u") { need(1); return { step(a[0].s), true }; }
        if (t == "uramp") { need(1); return { "max(" + a[0].s + ", 0)", false }; }
        if (t == "limit") { need(3); return { "min(max(" + a[0].s + ", " + a[1].s + "), " + a[2].s + ")", false }; }
        if (t == "sgn") { need(1); return { "copysign(1, " + a[0].s + ")", false }; }
        std::string s = t + "(";
        for (size_t i = 0; i < a.size(); ++i) s += (i ? ", " : "") + a[i].s;
        return { s + ")", false };
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
    // Индуктивности (для K): строка x_rhs, узлы, параметр L; связи K — после всех элементов (couple()).
    struct Inductor { size_t row; int a, b; std::string p, name; };
    std::map<std::string, Inductor> inductors;
    struct Coupling { std::string name, at, k; std::vector<std::string> inductors; };
    std::vector<Coupling> couplings;
    std::set<size_t> l_dc_edges;   // номера ветвей индуктивностей в dc_edges (сечения L-I в loops())

    // Взаимная индуктивность: у связанных катушек il_i' — алгебраические z dl_i (il_i' = dl_i) с
    // строками Va - Vb = L_i dl_i + sum_j k_ij sqrt(L_i L_j) dl_j. Матрица индуктивностей невырождена
    // при |k| < 1, поэтому индекс 1 — без обращения матрицы в тексте и для любых групп катушек.
    void couple() {
        if (couplings.empty()) return;
        std::map<std::string, std::vector<std::pair<std::string, std::string>>> mutual;   // L -> (L', k)
        for (const Coupling& cp : couplings) {
            for (const std::string& l : cp.inductors)
                if (!inductors.count(l)) fail(cp.at + "no inductor " + l);
            for (size_t i = 0; i < cp.inductors.size(); ++i)
                for (size_t j = 0; j < cp.inductors.size(); ++j)
                    if (i != j) {
                        if (cp.inductors[i] == cp.inductors[j]) fail(cp.at + "an inductor is coupled with itself");
                        mutual[cp.inductors[i]].push_back({ cp.inductors[j], cp.k });
                    }
        }
        std::map<std::string, std::string> dl;
        for (const auto& kv : mutual) {
            const Inductor& L = inductors.at(kv.first);
            dl[kv.first] = uniq("dl_" + sanitize(L.name));
            labels[dl[kv.first]] = var_tex("dl", L.name);
            zs.push_back(dl[kv.first]);
            x_rhs[L.row] = dl[kv.first];
        }
        for (const auto& kv : mutual) {
            const Inductor& L = inductors.at(kv.first);
            std::string g = lin(L.a, L.b) + " - " + L.p + "*" + dl[kv.first];
            for (const auto& mk : kv.second) {
                const Inductor& L2 = inductors.at(mk.first);
                g += " - " + mk.second + "*sqrt(" + L.p + "*" + L2.p + ")*" + dl[mk.first];
            }
            gs.push_back(g);
        }
        notes.push_back(std::to_string(couplings.size()) + " mutual inductance(s) K: inductor derivatives are "
                        "algebraic variables dl_* (index 1)");
    }

    struct Edge { int a, b; std::string name; };
    std::vector<Edge> v_edges;                        // C и ветви, заданные напряжением

    int n_r = 0, n_c = 0, n_l = 0, n_src = 0, n_ctl = 0, n_nullor = 0, n_b = 0, n_x = 0;
    int n_ideal_d = 0, n_ideal_q = 0, n_nocap = 0;   // упрощения приборов (SpiceImportOptions)

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
        std::string group;                            // группа параметров подсхемы "OPA2134/XU1"
    };

    // Группа, в которую param() кладёт параметр (только интерфейс, SpiceImportResult::param_groups).
    std::string cur_group;
    std::vector<std::pair<std::string, std::string>> pgroups;
    // Группы начальных условий: x — по элементу (cur_elem — имя элемента, как в netlist'е), z — в finish.
    std::string cur_elem;
    // Подписи полей (SpiceImportResult::field_labels).
    std::map<std::string, std::string> labels;
    static std::string tex_list(const std::string& s) {   // '.' и '_' имени -> запятые индекса
        std::string o = s;
        for (char& ch : o) if (ch == '.' || ch == '_') ch = ',';
        return o;
    }
    // Номинал элемента: буква типа заглавная, остальное имени — индекс (R1 -> R_{1}, x1.Rb -> R_{b,x1}).
    static std::string elem_tex(const std::string& full) {
        const size_t dot = full.rfind('.');
        const std::string last = dot == std::string::npos ? full : full.substr(dot + 1);
        std::string sub = tex_list(last.substr(1));
        if (dot != std::string::npos) sub += (sub.empty() ? "" : ",") + tex_list(full.substr(0, dot));
        const std::string letter(1, (char)std::toupper((unsigned char)last[0]));
        return sub.empty() ? letter : letter + "_{" + sub + "}";
    }
    // Переменная элемента: vc_{C1}, il_{L1}, i_{V1}.
    static std::string var_tex(const char* kind, const std::string& full) {
        return std::string(kind) + "_{" + tex_list(full) + "}";
    }
    std::string elem_param(const std::string& name, double v) {
        const std::string p = param(name, v);
        labels[p] = elem_tex(name);
        pvals[p] = fmt_num_si(v);   // номинал как на схеме: 15k, 10n (поля понимают приставки СИ)
        return p;
    }
    std::vector<std::pair<std::string, std::string>> icgroups;
    std::string state_group() const { return cur_group == "Devices" ? "Devices/" + cur_elem : cur_group; }
    // Порядок групп на экране: по виду элемента, подсхемы после приборов, модели и z в конце.
    static int group_rank(const std::string& path) {
        static const char* const order[] = { "Resistors", "Capacitors", "Inductors", "Coupling (K)", "Sources",
                                             "Controlled sources", "Devices" };
        const std::string top = path.substr(0, path.find('/'));
        for (int i = 0; i < 7; ++i) if (top == order[i]) return i;
        if (top == "Models") return 9;
        if (top == "Algebraic (computed)") return 10;
        return 8;
    }
    static void sort_groups(std::vector<std::pair<std::string, std::string>>& v) {
        std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return group_rank(a.second) < group_rank(b.second);
        });
    }
    // Верхний уровень — по виду элемента; внутри подсхемы — группа экземпляра.
    static std::string element_group(char type, const Ctx& c) {
        if (!c.group.empty()) return c.group;
        switch (type) {
        case 'r': return "Resistors";
        case 'c': return "Capacitors";
        case 'l': return "Inductors";
        case 'k': return "Coupling (K)";
        case 'v': case 'i': return "Sources";
        case 'e': case 'f': case 'g': case 'h': case 'b': return "Controlled sources";
        default:  return "Devices";
        }
    }

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
        node_var.push_back(uniq("v_" + sanitize(net_short(key))));
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
        pgroups.push_back({ p, cur_group });
        return p;
    }
    std::string branch_ref(const std::string& key) { return "\x01" + key + "\x02"; }

    // Выражение B / VALUE= в синтаксис U-CUDA: V(a), V(a,b) -> потенциалы, I(Vx) -> ток
    // ветви, time -> t, ** -> ^, ln -> log; параметры и числа с множителями — числами.
    // Условия и сравнения — CondRewriter (выражение без них печатается как раньше).
    std::string translate(const std::string& s, const Ctx& c, const std::string& at) {
        static const std::set<std::string> funcs = {
            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log",
            "log10", "sqrt", "abs", "pow", "atan2", "min", "max", "if", "u", "uramp", "limit", "sgn",
        };
        static const std::set<std::string> cond = { "<", ">", "<=", ">=", "==", "!=", "&&", "||", "!", "?", ":",
                                                    "if", "u", "uramp", "limit", "sgn" };
        std::vector<std::string> tk;
        bool has_cond = false;
        auto emit = [&](const std::string& x) { tk.push_back(x); if (cond.count(x)) has_cond = true; };
        size_t i = 0;
        while (i < s.size()) {
            const char ch = s[i];
            double v;
            if (std::isspace((unsigned char)ch) || ch == '{' || ch == '}' || ch == '\'') { ++i; continue; }
            if (read_number(s, i, v)) { emit(fmt(v)); continue; }
            if (ch == '*' && i + 1 < s.size() && s[i + 1] == '*') { emit("^"); i += 2; continue; }
            if (std::strchr("+-*/^(),", ch)) { emit(std::string(1, ch)); ++i; continue; }
            if (std::strchr("<>=!&|?:", ch)) {
                const std::string two = s.substr(i, 2);
                if (two == "<=" || two == ">=" || two == "==" || two == "!=" || two == "&&" || two == "||") {
                    emit(two); i += 2; continue;
                }
                if (ch == '=' || ch == '&' || ch == '|') fail(at + "'" + std::string(1, ch) + "' in an expression: use ==, &&, ||");
                emit(std::string(1, ch)); ++i; continue;
            }
            if (!ident_start(ch))
                fail(at + "'" + std::string(1, ch) + "' in an expression is not supported");
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
                    emit(vdiff(a, b));
                } else {
                    if (args.size() != 1) fail(at + "I() takes one source name");
                    emit(branch_ref(c.prefix + lower(args[0])));
                }
                i = e + 1;
                continue;
            }
            if (call) {
                std::string f = id == "ln" ? "log" : id == "pwr" ? "pow" : id;
                if (!funcs.count(f)) fail(at + "function " + id + "() is not supported");
                emit(f);
                i = j;
                continue;
            }
            if (id == "time") emit("t");
            else if (id == "pi") emit(fmt(kPi));
            else if (c.scope->get(id, v)) emit("(" + fmt(v) + ")");
            else fail(at + "unknown symbol '" + id + "'");
            i = j;
        }
        if (has_cond) {
            CondRewriter r{ tk, at };
            const std::string e = r.expr().s;
            if (r.p != tk.size()) fail(at + "unexpected '" + tk[r.p] + "' in an expression");
            return "(" + e + ")";
        }
        std::string o;
        for (const std::string& x : tk) o += x;
        return "(" + o + ")";
    }

    // Независимый источник: выражение по t. DC и параметры SIN — параметры a[].
    std::string source_expr(const Card& cd, size_t from, const Ctx& c, const std::string& name,
                            const std::string& at) {
        const auto& t = cd.tok;
        auto is_kw = [](const std::string& s) {
            const std::string l = lower(s);
            return l == "dc" || l == "ac" || l == "sin" || l == "sine" || l == "pulse" || l == "pwl" || l == "exp"
                || l == "distof1" || l == "distof2" || l == "sffm" || l == "am" || l == "trrandom" || l == "trnoise";
        };
        double dc = 0.0;
        std::string kind;
        std::vector<double> a;
        for (size_t i = from; i < t.size();) {
            const std::string l = lower(t[i]);
            // key = value (KiCad пишет "DC ac=1 AC 1"): DC — значение, остальное (ac, ...) — без влияния
            if (i + 2 < t.size() && t[i + 1] == "=") {
                if (l == "dc") dc = eval_value(t[i + 2], *c.scope);
                i += 3;
                continue;
            }
            if (!is_kw(t[i])) { dc = eval_value(t[i], *c.scope); ++i; continue; }
            ++i;
            std::vector<double> nums;
            while (i < t.size() && !is_kw(t[i]) && !(i + 1 < t.size() && t[i + 1] == "="))
                nums.push_back(eval_value(t[i++], *c.scope));
            if (l == "dc") { if (!nums.empty()) dc = nums[0]; }
            else if (l == "ac" || l == "distof1" || l == "distof2") {}
            else if (l == "sin" || l == "sine" || l == "pulse" || l == "pwl" || l == "exp" || l == "am" || l == "sffm") {
                kind = l == "sine" ? "sin" : l;
                a = nums;
            }
            else if (l == "trnoise" || l == "trrandom")
                fail(at + l + ": noise / random sources are not supported (the DAE is deterministic)");
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
        if (kind == "am") {   // AM(VA VO MF FC TD): VA (VO + sin(2 pi MF tt)) sin(2 pi FC tt), tt = t - TD, 0 до TD
            if (a.size() < 4) fail(at + "AM needs VA VO MF FC");
            const double td = arg(4, 0.0);
            const std::string tt = td == 0.0 ? std::string("t") : "max(t - " + fmt(td) + ", 0)";
            return "(" + param(name + "_va", a[0]) + "*(" + param(name + "_vo", a[1]) + " + sin(" + fmt(2.0 * kPi * a[2]) + "*"
                 + tt + "))*sin(" + fmt(2.0 * kPi * a[3]) + "*" + tt + "))";
        }
        if (kind == "sffm") {   // SFFM(VO VA FC MDI FS): VO + VA sin(2 pi FC t + MDI sin(2 pi FS t))
            if (a.size() < 2) fail(at + "SFFM needs VO and VA");
            const double fc = arg(2, tstop > 0.0 ? 1.0 / tstop : 0.0), mdi = arg(3, 0.0), fs = arg(4, fc);
            return "(" + param(name + "_vo", a[0]) + " + " + param(name + "_va", a[1]) + "*sin(" + fmt(2.0 * kPi * fc)
                 + "*t + " + fmt(mdi) + "*sin(" + fmt(2.0 * kPi * fs) + "*t)))";
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
        labels[i] = var_tex("i", name);
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
        labels[i] = var_tex("i", name);
        zs.push_back(i);
        gs.push_back(lin(inp, inn));
        current(out_p, out_n, i);
        branch[key] = i;
        conduct(out_p, out_n);
        ++n_nullor;
    }

    // Конденсатор с номиналом p (выражение): x = vc, z = ic, vc' = ic/p, Va - Vb - vc = 0.
    // НУ: IC= элемента, иначе .ic V(a) - V(b) (неназванный узел — 0, как в SPICE с UIC).
    // p — ёмкость: имя параметра или выражение, где "@V" — напряжение на ней (ёмкость перехода C(v)).
    void capacitor(const std::string& name, int a, int b, const std::string& p, bool has_ic, double v0) {
        const std::string x = uniq("vc_" + sanitize(name)), i = uniq("ic_" + sanitize(name));
        xs.push_back(x);
        icgroups.push_back({ x, state_group() });
        labels[x] = var_tex("vc", name);
        labels[i] = var_tex("ic", name);
        std::string pc = p;
        for (size_t k = pc.find("@V"); k != std::string::npos; k = pc.find("@V", k + x.size())) pc.replace(k, 2, x);
        x_rhs.push_back(i + "/" + (pc == p ? p : "(" + pc + ")"));
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
    // Параметры вычисляются в области видимости элемента; модель из .subckt — своя у каждого экземпляра
    // (имя с префиксом экземпляра: параметры a[] тоже свои), общая — одна на весь netlist.
    std::map<std::string, Model> model_inst;
    const Model& model(const std::string& name, const char* kind, const std::string& at, const Ctx& c) {
        auto it = deck.models.find(lower(name));
        if (it == deck.models.end()) fail(at + "unknown model " + name);
        const Model& src = it->second;
        const std::string& ty = src.type;
        const std::string k = kind;
        if (ty == "vdmos") fail(at + "model " + name + ": VDMOS (power MOSFET) models are not supported");
        const bool ok = k == "d" ? ty == "d" : k == "npn" ? (ty == "npn" || ty == "pnp")
                      : k == "nmos" ? (ty == "nmos" || ty == "pmos") : (ty == "njf" || ty == "pjf");
        if (!ok) fail(at + "model " + name + " has type " + ty);
        const std::string key = (src.in_subckt ? c.prefix : std::string()) + lower(name);
        auto mi = model_inst.find(key);
        if (mi != model_inst.end()) return mi->second;
        Model m = src;
        if (src.in_subckt) m.name = c.prefix + src.name;
        for (const auto& kv : src.raw) {
            try { m.p[kv.first] = eval_value(kv.second, *c.scope); }
            catch (const std::exception&) { m.bad.insert(kv.first); }
        }
        return model_inst[key] = m;
    }

    // Число из .model экземпляра (умолчание SPICE, если не задано).
    static double mpv(const Model& m, const char* k, double def) {
        auto it = m.p.find(k);
        return it != m.p.end() ? it->second : def;
    }
    // Параметры упрощённой модели (SpiceImportOptions::ideal_*) — a[] "<модель>_<имя>" с вычисленными значениями,
    // один раз на модель, в той же группе, что параметры .model.
    std::map<std::string, std::map<std::string, std::string>> ideal_cache;
    const std::map<std::string, std::string>& ideal_params(const Model& m,
                                                           const std::vector<std::pair<std::string, double>>& kv) {
        auto it = ideal_cache.find(lower(m.name));
        if (it != ideal_cache.end()) return it->second;
        std::map<std::string, std::string>& out = ideal_cache[lower(m.name)];
        const std::string saved = cur_group;
        if (!m.in_subckt) cur_group = "Models/" + m.name;
        for (const auto& p : kv) out[p.first] = param(m.name + "_" + p.first, p.second);
        cur_group = saved;
        return out;
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
        // Модель из подсхемы — своя у экземпляра (его группа); общая .model — своя группа.
        const std::string saved_group = cur_group;
        if (!m.in_subckt) cur_group = "Models/" + m.name;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (m.bad.count(keys[i]))
                fail("model " + m.name + ": cannot evaluate " + keys[i] + " = " + m.raw.at(keys[i]));
            auto v = m.p.find(keys[i]);
            const double val = v != m.p.end() ? v->second : defs[i];
            if (defs[i] == 0.0 && val == 0.0) continue;
            out[keys[i]] = param(m.name + "_" + keys[i], val);
        }
        cur_group = saved_group;
        std::string skip;
        for (const auto& kv : m.raw)
            if (kv.first == "mfg" || kv.first == "type" || kv.first == "icrating" || kv.first == "vceo"
                || kv.first == "vds" || kv.first == "iave" || kv.first == "vpk") continue;   // справочные поля
            else if (std::find(ignored.begin(), ignored.end(), kv.first) != ignored.end()) skip += " " + kv.first;
            else if (std::find(keys.begin(), keys.end(), kv.first) == keys.end()) skip += " " + kv.first + "(?)";
        if (!skip.empty()) notes.push_back("model " + m.name + ": not modelled:" + skip);
        return out;
    }
    // Параметр модели, заданный в .model, иначе — литерал умолчания SPICE (без параметра a[]).
    static std::string pd(const std::map<std::string, std::string>& mp, const char* k, const char* def) {
        auto it = mp.find(k);
        return it != mp.end() ? it->second : std::string(def);
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
        if (i < r.size() && r[i] == '{' && kw == "value") { r = r.substr(i); return true; }   // PSpice: VALUE {..}
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
        cur_group = element_group(type, c);
        cur_elem = t[0];
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
            // Значение-выражение от времени / напряжений (KiCad: r='TIME > 350m ? 8 : 89') — проводимость
            // как у B: I = V / R(t, V); число — как раньше, параметр a[].
            const std::string rv = lower(t[3]) == "r" && t.size() > 5 && t[4] == "=" ? t[5] : t[3];
            double r = 0.0;
            bool num_ok = true;
            try { r = eval_value(rv, *c.scope); } catch (const std::exception&) { num_ok = false; }
            if (!num_ok) {
                current(a, b, vdiff(a, b) + "/" + translate(rv, c, at));
                conduct(a, b);
                notes.push_back(name + ": resistance is an expression (" + rv + ")");
                ++n_r;
                break;
            }
            if (r == 0.0) fail(at + "zero resistance");
            current(a, b, vdiff(a, b) + "/" + elem_param(name, r));
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
            capacitor(name, a, b, elem_param(name, cv), has, v0);
            ++n_c;
            break;
        }
        case 'd': {
            need(4);
            const int a = node_of(t[1], c), k = node_of(t[2], c);
            const Model& m = model(t[3], "d", at, c);
            const double area = t.size() > 4 && t[4] != "=" && lower(t[4]) != "off" && lower(t[4]) != "ic"
                              ? eval_value(lower(t[4]) == "area" && t.size() > 6 ? t[6] : t[4], *c.scope) : 1.0;
            auto sc = [&](const std::string& p) { return area == 1.0 ? p : "(" + fmt(area) + "*" + p + ")"; };
            const bool caps = !opt.no_junction_caps;
            int j = a;
            std::map<std::string, std::string> mp;
            if (opt.ideal_diodes) {   // порог и Ron (касательная при 10 мА), RS — в Ron, без внутреннего узла
                const IdealJunction f = ideal_junction_fit(mpv(m, "is", 1e-14), mpv(m, "n", 1), mpv(m, "rs", 0));
                const auto& ip = ideal_params(m, { { "vf", f.vf }, { "ron", f.ron } });
                current(a, k, ideal_diode_current(vdiff(a, k), ip.at("vf"),
                                                  area == 1.0 ? ip.at("ron") : "(" + ip.at("ron") + "/" + fmt(area) + ")"));
                if (caps) mp = model_params(m, { "cjo", "vj", "m", "fc" }, { 0, 0, 0, 0 },
                                            { "is", "n", "rs", "bv", "ibv", "tt", "eg", "xti", "kf", "af" });
                ++n_ideal_d;
            } else {
                mp = caps ? model_params(m, { "is", "n", "rs", "cjo", "vj", "m", "fc" }, { 1e-14, 1, 0, 0, 0, 0, 0 },
                                         { "bv", "ibv", "tt", "eg", "xti", "kf", "af" })
                          : model_params(m, { "is", "n", "rs" }, { 1e-14, 1, 0 },
                                         { "cjo", "vj", "m", "fc", "bv", "ibv", "tt", "eg", "xti", "kf", "af" });
                j = internal(a, mp.count("rs") ? mp.at("rs") : std::string(), name, "a", area);
                current(j, k, diode_current(vdiff(j, k), sc(mp.at("is")), mp.at("n")));
            }
            conduct(j, k);
            if (caps && mp.count("cjo"))
                capacitor(name + "_cj", j, k, junction_capacitance("@V", sc(mp.at("cjo")), pd(mp, "vj", "1"),
                                                                   pd(mp, "m", "0.5"), pd(mp, "fc", "0.5")), false, 0.0);
            if (!caps && m.p.count("cjo")) ++n_nocap;
            ++n_dev;
            break;
        }
        case 'q': {
            need(5);
            // Q c b e [s] model [area]
            size_t mi = 4;
            if (!deck.models.count(lower(t[4])) && t.size() > 5) mi = 5;
            const Model& m = model(t[mi], "npn", at, c);
            const bool pnp = m.type == "pnp";
            if (!pnp && m.type != "npn") fail(at + "model " + m.name + " is not NPN/PNP");
            const double area = t.size() > mi + 1 && t[mi + 1] != "=" && lower(t[mi + 1]) != "off"
                              ? eval_value(lower(t[mi + 1]) == "area" && t.size() > mi + 3 ? t[mi + 3] : t[mi + 1], *c.scope) : 1.0;
            const bool caps = !opt.no_junction_caps, ideal = opt.ideal_bjts;
            static const std::vector<std::string> kQIgn = { "ikf", "ikr", "ise", "isc", "ne", "nc", "tf", "tr",
                "xtf", "itf", "vtf", "ptf", "eg", "xti", "xtb", "irb", "rbm", "cjs", "kf", "af", "nk", "iss", "ns", "bvcbo" };
            static const std::vector<std::string> kQDc = { "is", "bf", "br", "nf", "nr", "vaf", "var", "rb", "rc", "re" };
            static const std::vector<std::string> kQCap = { "cje", "cjc", "vje", "vjc", "mje", "mjc", "fc" };
            std::vector<std::string> keys, ign = kQIgn;
            std::vector<double> defs;
            if (!ideal) { keys = kQDc; defs = { 1e-16, 100, 1, 1, 1, 0, 0, 0, 0, 0 }; }
            else ign.insert(ign.end(), kQDc.begin(), kQDc.end());
            if (caps) { keys.insert(keys.end(), kQCap.begin(), kQCap.end()); defs.insert(defs.end(), kQCap.size(), 0.0); }
            else ign.insert(ign.end(), kQCap.begin(), kQCap.end());
            const auto& mp = model_params(m, keys, defs, ign);
            auto sc = [&](const std::string& p) { return area == 1.0 ? p : "(" + fmt(area) + "*" + p + ")"; };
            auto opt_p = [&](const char* k) { return mp.count(k) ? mp.at(k) : std::string(); };
            // Идеальный BJT — без RB / RC / RE: внутренних узлов нет.
            const int nc = ideal ? node_of(t[1], c) : internal(node_of(t[1], c), opt_p("rc"), name, "c", area);
            const int nb = ideal ? node_of(t[2], c) : internal(node_of(t[2], c), opt_p("rb"), name, "b", area);
            const int ne = ideal ? node_of(t[3], c) : internal(node_of(t[3], c), opt_p("re"), name, "e", area);
            auto inv = [&](const char* k) { return mp.count(k) ? "(1/" + mp.at(k) + ")" : std::string(); };
            // NPN: vbe, vbc; PNP: veb, vcb и токи обратного знака (из коллектора и базы).
            const std::string v1 = pnp ? vdiff(ne, nb) : vdiff(nb, ne), v2 = pnp ? vdiff(nc, nb) : vdiff(nb, nc);
            BjtCurrents q;
            if (ideal) {   // Эберс–Молл на идеальных переходах: Ic = Dbe - Dbc - Dbc/BR, Ib = Dbe/BF + Dbc/BR
                const double is = mpv(m, "is", 1e-16);
                const IdealJunction fe = ideal_junction_fit(is, mpv(m, "nf", 1), 0), fc_ = ideal_junction_fit(is, mpv(m, "nr", 1), 0);
                const auto& ip = ideal_params(m, { { "vfbe", fe.vf }, { "ronbe", fe.ron }, { "vfbc", fc_.vf },
                                                   { "ronbc", fc_.ron }, { "bf", mpv(m, "bf", 100) }, { "br", mpv(m, "br", 1) } });
                auto ron = [&](const char* k) { return area == 1.0 ? ip.at(k) : "(" + ip.at(k) + "/" + fmt(area) + ")"; };
                const std::string dbe = ideal_diode_current(v1, ip.at("vfbe"), ron("ronbe"));
                const std::string dbc = ideal_diode_current(v2, ip.at("vfbc"), ron("ronbc"));
                q.ic = "(" + dbe + " - " + dbc + "*(1 + 1/" + ip.at("br") + "))";
                q.ib = "(" + dbe + "/" + ip.at("bf") + " + " + dbc + "/" + ip.at("br") + ")";
                ++n_ideal_q;
            } else
                q = bjt_currents(v1, v2, sc(mp.at("is")), mp.at("bf"), mp.at("br"), mp.at("nf"),
                                 mp.at("nr"), inv("vaf"), inv("var"));
            if (pnp) { current(ne, nc, q.ic); current(ne, nb, q.ib); }
            else     { current(nc, ne, q.ic); current(nb, ne, q.ib); }
            conduct(nc, ne);
            conduct(nb, ne);
            // Напряжение на ёмкости — vb - ve (vb - vc); прямое смещение перехода у PNP — обратного знака.
            const std::string vj = pnp ? "(-@V)" : "@V", fc = pd(mp, "fc", "0.5");
            if (caps && mp.count("cje"))
                capacitor(name + "_cje", nb, ne, junction_capacitance(vj, sc(mp.at("cje")), pd(mp, "vje", "0.75"),
                                                                     pd(mp, "mje", "0.33"), fc), false, 0.0);
            if (caps && mp.count("cjc"))
                capacitor(name + "_cjc", nb, nc, junction_capacitance(vj, sc(mp.at("cjc")), pd(mp, "vjc", "0.75"),
                                                                     pd(mp, "mjc", "0.33"), fc), false, 0.0);
            if (!caps && (m.p.count("cje") || m.p.count("cjc"))) ++n_nocap;
            ++n_dev;
            break;
        }
        case 'l': {
            need(4);
            const int a = node_of(t[1], c), b = node_of(t[2], c);
            const double lv = value_at(3, "l");
            if (lv == 0.0) fail(at + "zero inductance");
            const std::string p = elem_param(name, lv);
            const std::string x = uniq("il_" + sanitize(name));
            xs.push_back(x);
            icgroups.push_back({ x, state_group() });
            labels[x] = var_tex("il", name);
            inductors[key] = Inductor{ x_rhs.size(), a, b, p, name };
            x_rhs.push_back(vdiff(a, b) + "/" + p);
            current(a, b, x);
            double v0 = 0.0;
            if (ic_of(&v0)) any_ic = true;
            else v0 = 0.0;
            ic[x] = fmt_short(v0);
            branch[key] = x;
            l_dc_edges.insert(dc_edges.size());
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
            const Model& m = model(t[4], "njf", at, c);
            const bool pch = m.type == "pjf";
            const double area = t.size() > 5 && t[5] != "=" && lower(t[5]) != "off"
                              ? eval_value(lower(t[5]) == "area" && t.size() > 7 ? t[7] : t[5], *c.scope) : 1.0;
            const auto& mp = model_params(m, { "vto", "beta", "lambda", "is", "rd", "rs", "cgs", "cgd", "pb", "m", "fc" },
                                          { -2.0, 1e-4, 0, 1e-14, 0, 0, 0, 0, 0, 0, 0 },
                                          { "b", "kf", "af", "n", "isr", "nr", "alpha", "vk",
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
            const std::string vj = pch ? "(-@V)" : "@V", pb = pd(mp, "pb", "1"), mj = pd(mp, "m", "0.5"),
                              fc = pd(mp, "fc", "0.5");
            if (!opt.no_junction_caps && mp.count("cgs"))
                capacitor(name + "_cgs", ng, ns, junction_capacitance(vj, sc(mp.at("cgs")), pb, mj, fc), false, 0.0);
            if (!opt.no_junction_caps && mp.count("cgd"))
                capacitor(name + "_cgd", ng, nd, junction_capacitance(vj, sc(mp.at("cgd")), pb, mj, fc), false, 0.0);
            ++n_dev;
            break;
        }
        case 'm': {
            if (t.size() == 5 && deck.models.count(lower(t[4]))) model(t[4], "nmos", at, c);   // 3 вывода: VDMOS — его отказ
            need(6);
            // M d g s b model [L=..] [W=..] [M=..] [AD= AS= PD= PS= NRD= NRS= OFF IC=]: level 1
            const Model& m = model(t[5], "nmos", at, c);
            const bool pch = m.type == "pmos";
            auto lv = m.p.find("level");
            if (lv != m.p.end() && lv->second != 1.0)
                fail(at + "MOSFET model " + m.name + ": LEVEL " + fmt_short(lv->second) + " is not supported (level 1 only)");
            double W = 100e-6, L = 100e-6, mult = 1.0;
            for (size_t i = 6; i + 2 < t.size(); ++i) {
                if (t[i + 1] != "=") continue;
                const std::string k = lower(t[i]);
                if (k == "w") W = eval_value(t[i + 2], *c.scope);
                else if (k == "l") L = eval_value(t[i + 2], *c.scope);
                else if (k == "m") mult = eval_value(t[i + 2], *c.scope);
            }
            if (!(W > 0) || !(L > 0)) fail(at + "W and L must be > 0");
            const auto& mp = model_params(m, { "vto", "kp", "lambda", "gamma", "phi", "rd", "rs", "cgso", "cgdo" },
                                          { 0, 2e-5, 0, 0, 0, 0, 0, 0, 0 },
                                          { "level", "tox", "u0", "uo", "ld", "nsub", "nss", "nfs", "tpg", "xj", "js", "is",
                                            "pb", "cj", "mj", "cjsw", "mjsw", "fc", "cbd", "cbs", "cgbo", "rsh", "kf", "af",
                                            "tnom" });
            auto opt_p = [&](const char* k) { return mp.count(k) ? mp.at(k) : std::string(); };
            const int nd = internal(node_of(t[1], c), opt_p("rd"), name, "d", 1.0);
            const int ng = node_of(t[2], c);
            const int ns = internal(node_of(t[3], c), opt_p("rs"), name, "s", 1.0);
            const int nb = node_of(t[4], c);
            // PMOS: те же формулы от vsg, vsd, vdg, vsb, порог -VTO (SPICE хранит type*VTO), ток обратного знака.
            const std::string vgs = pch ? vdiff(ns, ng) : vdiff(ng, ns), vds = pch ? vdiff(ns, nd) : vdiff(nd, ns),
                              vgd = pch ? vdiff(nd, ng) : vdiff(ng, nd), vbs = pch ? vdiff(ns, nb) : vdiff(nb, ns),
                              vbd = pch ? vdiff(nd, nb) : vdiff(nb, nd);
            const std::string vt0 = mp.count("vto") ? (pch ? "(-" + mp.at("vto") + ")" : mp.at("vto")) : std::string("0");
            std::string vts = vt0, vtd = vt0;
            if (mp.count("gamma")) {
                const std::string phi = pd(mp, "phi", "0.6");
                vts = mosfet_threshold(vt0, mp.at("gamma"), phi, vbs);
                vtd = mosfet_threshold(vt0, mp.at("gamma"), phi, vbd);
            }
            const std::string beta = "(" + fmt(mult * W / (2 * L)) + "*" + mp.at("kp") + ")";
            const std::string id = mosfet_drain_current(vgs, vds, vgd, vts, vtd, beta, opt_p("lambda"));
            if (pch) current(ns, nd, id); else current(nd, ns, id);
            conduct(nd, ns);
            // Затвор изолирован: тока нет, но ёмкости перекрытия CGSO / CGDO (на единицу ширины) — постоянные.
            if (!opt.no_junction_caps && mp.count("cgso")) capacitor(name + "_cgs", ng, ns, "(" + fmt(mult * W) + "*" + mp.at("cgso") + ")", false, 0.0);
            if (!opt.no_junction_caps && mp.count("cgdo")) capacitor(name + "_cgd", ng, nd, "(" + fmt(mult * W) + "*" + mp.at("cgdo") + ")", false, 0.0);
            touch(ng);
            touch(nb);
            ++n_dev;
            break;
        }
        case 'k': {
            // K name L1 L2 [L3 ...] k: взаимная индуктивность M_ij = k sqrt(Li Lj) (см. couple())
            need(4);
            Coupling cp;
            cp.name = name;
            cp.at = at;
            for (size_t i = 1; i + 1 < t.size(); ++i) cp.inductors.push_back(c.prefix + lower(t[i]));
            const double kv = eval_value(t.back(), *c.scope);
            if (!(std::fabs(kv) <= 1.0)) fail(at + "|k| must be <= 1");
            // |k| = 1 (идеальная связь, так пишут трансформаторы KiCad): матрица индуктивностей вырождена,
            // напряжения обмоток связаны алгебраически — индекс 2 (Radau IIA / BDF его держат).
            if (std::fabs(kv) == 1.0)
                notes.push_back(name + ": k = 1 (ideal coupling): the inductance matrix is singular - index 2");
            cp.k = param(name, kv);
            couplings.push_back(cp);
            break;
        }
        case 'z':
            fail(at + "MESFET is not supported yet");
        case 'a':
            fail(at + "XSPICE code models (A elements: digital, potentiometer, ...) are not supported");
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
        in.group = c.group.empty() ? sub + "/" + t[0] : c.group + "/" + t[0];
        for (size_t k = 0; k < nodes.size(); ++k) in.pins[s.pins[k]] = node_key(nodes[k], c);
        // узлы-выводы должны существовать снаружи
        for (size_t k = 0; k < nodes.size(); ++k) node_of(nodes[k], c);
        auto sc = std::make_unique<Scope>();
        sc->parent = &deck.global;
        for (const auto& kv : s.defaults) sc->v[kv.first] = eval_value(kv.second, *sc);
        for (const auto& kv : over) sc->v[kv.first] = eval_value(kv.second, *c.scope);
        in.scope = sc.get();
        // .param подсхемы — до её элементов: PSpice-модели (LM317) пишут .PARAM после строк, где он нужен
        for (const Card& e : s.cards)
            if (!e.tok.empty() && lower(e.tok[0]) == ".param")
                for (size_t i = 1; i + 2 < e.tok.size(); i += 3)
                    if (e.tok[i + 1] == "=") sc->v[lower(e.tok[i])] = eval_value(e.tok[i + 2], *sc);
        for (const Card& e : s.cards) {
            if (e.tok.empty()) continue;
            const std::string k = lower(e.tok[0]);
            if (k == ".param") continue;
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
        // Сечение из индуктивностей и источников тока (двойственно контуру C-V): стянуть все ветви, кроме L
        // и источников тока (у тех нет записи в dc_edges / v_edges); индуктивность между разными
        // стянутыми узлами лежит в таком сечении — KCL связывает токи катушек без z, индекс 2.
        for (size_t i = 0; i < par.size(); ++i) par[i] = (int)i;
        for (size_t k = 0; k < dc_edges.size(); ++k)
            if (!l_dc_edges.count(k)) par[(size_t)find(dc_edges[k].first + 1)] = find(dc_edges[k].second + 1);
        for (const Edge& e : v_edges) par[(size_t)find(e.a + 1)] = find(e.b + 1);
        std::string cut;
        for (const auto& kv : inductors)
            if (find(kv.second.a + 1) != find(kv.second.b + 1)) cut += " " + kv.second.name;
        if (!cut.empty())
            notes.push_back("cutset of inductors / current sources through" + cut
                            + ": index 2 (the inductor currents are not independent)");
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
        couple();
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
        // .ic без UIC (SPICE): рабочая точка считается с узлами .ic, удержанными на заданных напряжениях, потом
        // узлы отпускаются. Для неё — те же уравнения, где KCL удержанного узла заменён на v - V = 0 (ток
        // удерживающего источника свободен).
        if (!deck.uic && !deck.ic_nodes.empty()) {
            std::map<size_t, double> pin;
            for (const auto& kv : deck.ic_nodes) {
                auto it = node_idx.find(node_key(kv.first, Ctx{}));
                if (it != node_idx.end()) pin[(size_t)it->second] = kv.second;
            }
            std::ostringstream op;
            for (size_t i = 0; i < xs.size(); ++i) op << xs[i] << "' = " << resolve(x_rhs[i]) << "\n";
            for (size_t n = 0; n < node_keys.size(); ++n) {
                auto p = pin.find(n);
                if (p != pin.end()) { op << "0 = " << node_var[n] << " - (" << fmt(p->second) << ")\n"; continue; }
                std::string s;
                for (const std::string& term : kcl[n]) s += (s.empty() ? (term[0] == '+' ? term.substr(2) : "-" + term.substr(2))
                                                                       : " " + term);
                op << "0 = " << resolve(s) << "\n";
            }
            for (const std::string& g : gs) op << "0 = " << resolve(g) << "\n";
            if (!pin.empty()) r.op_plain_text = op.str();
        }

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
        r.param_groups = pgroups;
        sort_groups(r.param_groups);
        r.ic_groups = icgroups;
        for (const std::string& v : node_var) r.ic_groups.push_back({ v, "Algebraic (computed)/Node potentials" });
        for (const std::string& z : zs) r.ic_groups.push_back({ z, "Algebraic (computed)/Branch currents" });
        sort_groups(r.ic_groups);
        r.field_labels = labels;
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
        if (n_ideal_d) r.notes.push_back(std::to_string(n_ideal_d) + " diode(s) / LED(s) ideal: threshold Vf + Ron (tangent of "
                                         "the exponential at 10 mA, parameters <model>_vf / _ron)");
        if (n_ideal_q) r.notes.push_back(std::to_string(n_ideal_q) + " BJT(s) ideal: Ebers-Moll on Vf + Ron junctions "
                                         "(<model>_vfbe/_ronbe/_vfbc/_ronbc, _bf, _br), no Early effect, no RB / RC / RE");
        if (n_nocap) r.notes.push_back(std::to_string(n_nocap) + " device(s) without junction capacitances");
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
