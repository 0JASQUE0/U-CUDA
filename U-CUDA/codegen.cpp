// Реализация парсера (обычный синтаксис + LaTeX) и кодгена схем.
#include "codegen.hpp"
#include <map>
#include <set>
#include <memory>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <vector>

namespace { // внутренняя линковка: всё ниже не видно из других .cpp/.cu

    // ---------- Мини-AST ----------
    struct Node {
        enum Kind { Num, Sym, Call, Add, Sub, Mul, Div, Pow, Neg } kind;
        double num = 0; std::string name;
        std::shared_ptr<Node> a, b; std::vector<std::shared_ptr<Node>> args;
    };
    using PN = std::shared_ptr<Node>;
    PN mk(Node::Kind k) { auto n = std::make_shared<Node>(); n->kind = k; return n; }

    const std::set<std::string>& known_funcs() {
        static const std::set<std::string> f = {
            "sin","cos","tan","asin","acos","atan","sinh","cosh","tanh",
            "exp","log","log2","log10","sqrt","cbrt","fabs","abs",
            "pow","atan2","fmod","floor","ceil"
        };
        return f;
    }
    const std::map<std::string, std::string>& latex_funcs() {
        static const std::map<std::string, std::string> m = {
            {"sin","sin"},{"cos","cos"},{"tan","tan"},{"arcsin","asin"},
            {"arccos","acos"},{"arctan","atan"},{"sinh","sinh"},{"cosh","cosh"},
            {"tanh","tanh"},{"exp","exp"},{"ln","log"},{"log","log"},{"sqrt","sqrt"}
        };
        return m;
    }
    const std::map<std::string, std::string>& latex_symbols() {
        static const std::map<std::string, std::string> m = {
            {"alpha","alpha"},{"beta","beta"},{"gamma","gamma"},{"delta","delta"},
            {"epsilon","epsilon"},{"varepsilon","epsilon"},{"zeta","zeta"},{"eta","eta"},
            {"theta","theta"},{"vartheta","theta"},{"iota","iota"},{"kappa","kappa"},
            {"lambda","lambda"},{"mu","mu"},{"nu","nu"},{"xi","xi"},{"pi","pi"},
            {"rho","rho"},{"sigma","sigma"},{"tau","tau"},{"phi","phi"},{"varphi","phi"},
            {"chi","chi"},{"psi","psi"},{"omega","omega"},{"Gamma","Gamma"},
            {"Delta","Delta"},{"Theta","Theta"},{"Lambda","Lambda"},{"Sigma","Sigma"},
            {"Phi","Phi"},{"Psi","Psi"},{"Omega","Omega"}
        };
        return m;
    }

    // ---------- Парсер ----------
    struct Parser {
        std::string s; size_t i = 0; bool latex;
        Parser(std::string src, bool lx = false) :s(std::move(src)), latex(lx) {}
        void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
        char peek() { ws(); return i < s.size() ? s[i] : '\0'; }
        bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }

        std::string latex_command() {
            size_t j = i; while (j < s.size() && std::isalpha((unsigned char)s[j])) ++j;
            std::string cmd = s.substr(i, j - i); i = j; return cmd;
        }
        PN braced_arg() {
            ws();
            if (i < s.size() && s[i] == '{') {
                ++i; PN e = expr(); ws();
                if (i >= s.size() || s[i] != '}') throw std::runtime_error("expected } "); ++i; return e;
            }
            return primary_single();
        }

        PN parse() { PN e = expr(); ws(); if (i != s.size()) throw std::runtime_error("trailing input at pos " + std::to_string(i)); return e; }

        PN expr() {
            PN n = term();
            for (;;) {
                char c = peek();
                if (c == '+') { eat('+'); auto t = mk(Node::Add); t->a = n; t->b = term(); n = t; }
                else if (c == '-') { eat('-'); auto t = mk(Node::Sub); t->a = n; t->b = term(); n = t; }
                else break;
            }
            return n;
        }
        bool at_primary_start() {
            ws(); if (i >= s.size()) return false; char c = s[i];
            if (c == '(' || c == '{') return true;
            if (std::isdigit((unsigned char)c) || c == '.') return true;
            if (std::isalpha((unsigned char)c) || c == '_') return true;
            if (latex && c == '\\') {
                if (s.compare(i, 5, "\\cdot") == 0) return false;
                if (s.compare(i, 6, "\\right") == 0) return false;
                return true;
            }
            return false;
        }
        PN term() {
            PN n = powr();
            for (;;) {
                if (latex) {
                    ws();
                    if (i + 5 <= s.size() && s.compare(i, 5, "\\cdot") == 0) { i += 5; auto t = mk(Node::Mul); t->a = n; t->b = powr(); n = t; continue; }
                }
                char c = peek();
                if (c == '*') { eat('*'); auto t = mk(Node::Mul); t->a = n; t->b = powr(); n = t; }
                else if (c == '/') { eat('/'); auto t = mk(Node::Div); t->a = n; t->b = powr(); n = t; }
                else if (latex && at_primary_start()) { auto t = mk(Node::Mul); t->a = n; t->b = powr(); n = t; }
                else break;
            }
            return n;
        }
        PN powr() {
            PN base = unary();
            if (peek() == '^') {
                eat('^'); auto t = mk(Node::Pow); t->a = base;
                t->b = latex ? braced_arg() : powr(); return t;
            }
            return base;
        }
        PN unary() {
            char c = peek();
            if (c == '-') { eat('-'); auto t = mk(Node::Neg); t->a = unary(); return t; }
            if (c == '+') { eat('+'); return unary(); }
            return primary();
        }
        PN primary_single() {
            ws();
            char c = i < s.size() ? s[i] : '\0';
            if (std::isdigit((unsigned char)c) || c == '.') {
                auto n = mk(Node::Num);
                size_t j = i;
                while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
                if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                    ++j; if (j < s.size() && (s[j] == '+' || s[j] == '-')) ++j;
                    while (j < s.size() && std::isdigit((unsigned char)s[j])) ++j;
                }
                std::string numstr = s.substr(i, j - i);
                try { n->num = std::stod(numstr); }
                catch (...) { throw std::runtime_error("bad number: '" + numstr + "'"); }
                i = j; return n;
            }
            if (latex && c == '\\') {
                ++i; std::string cmd = latex_command();
                auto sy = latex_symbols().find(cmd);
                if (sy != latex_symbols().end()) { auto n = mk(Node::Sym); n->name = sy->second; return n; }
                throw std::runtime_error("unexpected latex command \\" + cmd + " as single arg");
            }
            if (std::isalpha((unsigned char)c) || c == '_') {
                auto n = mk(Node::Sym);
                size_t j = i; while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '_'))++j;
                n->name = s.substr(i, j - i); i = j; return n;
            }
            return primary();
        }
        PN primary() {
            ws();
            char c = i < s.size() ? s[i] : '\0';
            if (c == '(') { ++i; PN e = expr(); if (!eat(')')) throw std::runtime_error("expected )"); return e; }
            if (latex) {
                if (i + 5 <= s.size() && s.compare(i, 5, "\\left") == 0) {
                    i += 5; ws();
                    if (i < s.size() && (s[i] == '(' || s[i] == '[')) ++i;
                    PN e = expr(); ws();
                    if (i + 6 <= s.size() && s.compare(i, 6, "\\right") == 0) {
                        i += 6; ws();
                        if (i < s.size() && (s[i] == ')' || s[i] == ']')) ++i;
                    }
                    return e;
                }
                if (c == '{') {
                    ++i; PN e = expr(); ws();
                    if (i >= s.size() || s[i] != '}') throw std::runtime_error("expected }"); ++i; return e;
                }
                if (c == '\\') {
                    ++i; std::string cmd = latex_command();
                    if (cmd == "frac") {
                        PN num = braced_arg(); PN den = braced_arg();
                        auto t = mk(Node::Div); t->a = num; t->b = den; return t;
                    }
                    auto fn = latex_funcs().find(cmd);
                    if (fn != latex_funcs().end()) {
                        // степень на функции: \sin^{2} x -> pow(sin(x), 2)
                        PN exponent;
                        if (peek() == '^') {
                            eat('^');
                            exponent = braced_arg();
                        }
                        auto n = mk(Node::Call); n->name = fn->second;
                        n->args.push_back(braced_arg());
                        if (exponent) {
                            auto p = mk(Node::Pow); p->a = n; p->b = exponent; return p;
                        }
                        return n;
                    }
                    auto sy = latex_symbols().find(cmd);
                    if (sy != latex_symbols().end()) { auto n = mk(Node::Sym); n->name = sy->second; return n; }
                    throw std::runtime_error("unknown latex command \\" + cmd);
                }
            }
            if (std::isdigit((unsigned char)c) || c == '.') {
                size_t j = i; while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.'))++j;
                if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                    ++j; if (j < s.size() && (s[j] == '+' || s[j] == '-'))++j;
                    while (j < s.size() && std::isdigit((unsigned char)s[j]))++j;
                }
                std::string numstr = s.substr(i, j - i);
                auto n = mk(Node::Num);
                try { n->num = std::stod(numstr); }
                catch (...) { throw std::runtime_error("bad number: '" + numstr + "'"); }
                i = j; return n;
            }
            if (std::isalpha((unsigned char)c) || c == '_') {
                size_t j = i; while (j < s.size() && (std::isalnum((unsigned char)s[j]) || s[j] == '_'))++j;
                std::string id = s.substr(i, j - i); i = j;
                if (peek() == '(') {
                    if (!known_funcs().count(id)) throw std::runtime_error("unknown function: " + id);
                    eat('('); auto n = mk(Node::Call); n->name = id;
                    if (peek() != ')') {
                        n->args.push_back(expr());
                        while (peek() == ',') { eat(','); n->args.push_back(expr()); }
                    }
                    if (!eat(')')) throw std::runtime_error("expected ) after " + id); return n;
                }
                auto n = mk(Node::Sym); n->name = id; return n;
            }
            throw std::runtime_error(std::string("unexpected char '") + c + "' at pos " + std::to_string(i));
        }
    };

    // ---------- Печать ----------
    struct NameMap {
        std::map<std::string, std::string> m;
        std::string resolve(const std::string& nm) const {
            auto it = m.find(nm); if (it == m.end()) throw std::runtime_error("unknown symbol: " + nm);
            return it->second;
        }
    };

    int prec(Node::Kind k) {
        switch (k) {
        case Node::Add: case Node::Sub: return 1;
        case Node::Mul: case Node::Div: return 2;
        case Node::Neg: return 3; case Node::Pow: return 4; default: return 5;
        }
    }
    std::string fmtnum(double v) {
        // целые числа в разумном диапазоне печатаем как "N.0", без научной формы
        // (иначе 10 -> "1e+01"). Большие/дробные -> кратчайший round-trip.
        if (v == (double)(long long)v) {
            long long iv = (long long)v;
            if (iv > -1000000000000LL && iv < 1000000000000LL) {
                return std::to_string(iv) + ".0";
            }
        }
        char buf[64];
        for (int p = 1; p <= 17; ++p) {
            std::snprintf(buf, sizeof(buf), "%.*g", p, v);
            if (std::strtod(buf, nullptr) == v) break;
        }
        std::string s(buf);
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos
            && s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) s += ".0";
        return s;
    }
    void emit(const PN& n, const NameMap& nm, std::ostream& o);
    void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) {
        bool paren = prec(n->kind) < pp;
        if (paren) { o << "("; emit(n, nm, o); o << ")"; }
        else { emit(n, nm, o); }
    }
    void emit(const PN& n, const NameMap& nm, std::ostream& o) {
        switch (n->kind) {
        case Node::Num:o << fmtnum(n->num); break;
        case Node::Sym:o << nm.resolve(n->name); break;
        case Node::Neg:o << "-"; emit_child(n->a, nm, o, prec(Node::Neg)); break;
        case Node::Add:emit_child(n->a, nm, o, 1); o << " + "; emit_child(n->b, nm, o, 1); break;
        case Node::Sub:emit_child(n->a, nm, o, 1); o << " - "; emit_child(n->b, nm, o, 2); break;
        case Node::Mul:emit_child(n->a, nm, o, 2); o << " * "; emit_child(n->b, nm, o, 2); break;
        case Node::Div:emit_child(n->a, nm, o, 2); o << " / "; emit_child(n->b, nm, o, 3); break;
        case Node::Pow: {
            if (n->b->kind == Node::Num) {
                double e = n->b->num; int ie = (int)e;
                if (e == (double)ie && ie >= 2 && ie <= 4) { o << "("; for (int k = 0; k < ie; ++k) { if (k)o << " * "; emit_child(n->a, nm, o, 2); }o << ")"; break; }
            }
            o << "pow("; emit(n->a, nm, o); o << ", "; emit(n->b, nm, o); o << ")"; break;
        }
        case Node::Call:o << n->name << "("; for (size_t k = 0; k < n->args.size(); ++k) { if (k)o << ", "; emit(n->args[k], nm, o); }o << ")"; break;
        }
    }

    // Зарезервированные математические константы: остаются именем в коде,
    // не идут в a[]. Определяются в шапке ядра отдельно (напр. const double pi = M_PI;).
    const std::set<std::string>& math_constants() {
        static const std::set<std::string> c = { "pi" };
        return c;
    }
    NameMap build_namemap(const System& s, const std::string& st) {
        NameMap nm;
        for (size_t i = 0; i < s.vars.size(); ++i) nm.m[s.vars[i]] = st + "[" + std::to_string(i) + "]";
        for (size_t j = 0; j < s.params.size(); ++j) nm.m[s.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
        // константы маппятся сами на себя
        for (const auto& c : math_constants()) nm.m[c] = c;
        return nm;
    }
    std::vector<std::string> rhs_over(const System& s, const std::string& st) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        NameMap nm = build_namemap(s, st); std::vector<std::string> out;
        for (size_t i = 0; i < s.rhs.size(); ++i) {
            Parser p(s.rhs[i], s.latex); PN ast = p.parse();
            std::ostringstream o; emit(ast, nm, o); out.push_back(o.str());
        }
        return out;
    }

    // ---------- emit_plain: AST -> plain-выражение (имена как есть) ----------
    // Используется CD-генератором как нормализация: парсер AST понимает и LaTeX,
    // и обычный синтаксис; emit_plain даёт каноничную plain-строку, по которой
    // дальше работает регекс-логика разложения rhs на (sign, coef, remainder).
    void emit_plain(const PN& n, std::ostream& o);
    void emit_plain_child(const PN& n, std::ostream& o, int pp) {
        bool paren = prec(n->kind) < pp;
        if (paren) { o << "("; emit_plain(n, o); o << ")"; }
        else { emit_plain(n, o); }
    }
    void emit_plain(const PN& n, std::ostream& o) {
        switch (n->kind) {
        case Node::Num: o << fmtnum(n->num); break;
        case Node::Sym: o << n->name; break;
        case Node::Neg: o << "-"; emit_plain_child(n->a, o, prec(Node::Neg)); break;
        case Node::Add: emit_plain_child(n->a, o, 1); o << " + "; emit_plain_child(n->b, o, 1); break;
        case Node::Sub: emit_plain_child(n->a, o, 1); o << " - "; emit_plain_child(n->b, o, 2); break;
        case Node::Mul: emit_plain_child(n->a, o, 2); o << " * "; emit_plain_child(n->b, o, 2); break;
        case Node::Div: emit_plain_child(n->a, o, 2); o << " / "; emit_plain_child(n->b, o, 3); break;
        case Node::Pow:
            // Печатаем как X^N, чтобы CD-регекс мог распознать степень переменной.
            emit_plain_child(n->a, o, 4); o << "^"; emit_plain_child(n->b, o, 4); break;
        case Node::Call:
            o << n->name << "(";
            for (size_t k = 0; k < n->args.size(); ++k) { if (k) o << ", "; emit_plain(n->args[k], o); }
            o << ")";
            break;
        }
    }

    // ---------- CD helpers (порт Python-скрипта; работают по plain-строкам) ----------
    const std::set<std::string>& cd_known_functions() {
        static const std::set<std::string> f = {
            "sin","cos","tan","exp","log","ln","sqrt","abs","fabs",
            "asin","acos","atan","sinh","cosh","tanh","pow"
        };
        return f;
    }

    std::string cd_strip(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\n\r");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\n\r");
        return s.substr(a, b - a + 1);
    }

    bool cd_contains_variable(const std::string& expr, const std::string& var) {
        try {
            std::regex re("\\b" + var + "\\b");
            return std::regex_search(expr, re);
        } catch (...) { return false; }
    }

    enum class CdNonlin { None, Linear, Nonlinear };

    CdNonlin cd_get_nonlinearity_type(const std::string& expr, const std::string& var) {
        if (!cd_contains_variable(expr, var)) return CdNonlin::None;
        // var^N (N >= 2)
        try {
            std::regex pow_re("\\b" + var + "\\s*\\^\\s*(\\d+)");
            std::smatch m;
            if (std::regex_search(expr, m, pow_re)) {
                int p = std::stoi(m[1].str());
                if (p >= 2) return CdNonlin::Nonlinear;
            }
        } catch (...) {}
        // var * var
        try {
            std::regex mm_re("\\b" + var + "\\s*\\*\\s*" + var + "\\b");
            if (std::regex_search(expr, mm_re)) return CdNonlin::Nonlinear;
        } catch (...) {}
        // var внутри функции f(... var ...)
        for (const auto& fn : cd_known_functions()) {
            try {
                std::regex f_re(fn + "\\s*\\([^)]*\\b" + var + "\\b[^)]*\\)");
                if (std::regex_search(expr, f_re)) return CdNonlin::Nonlinear;
            } catch (...) {}
        }
        return CdNonlin::Linear;
    }

    // Разбивает выражение по знакам +/- (на верхнем уровне Python это не учитывает,
    // мы тоже не учитываем). Не разрезает e+5 / e-5 у чисел.
    std::vector<std::string> cd_split_terms(const std::string& expr) {
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t i = 0; i < expr.size(); ++i) {
            if (i > 0 && (expr[i] == '+' || expr[i] == '-')) {
                char prev = expr[i - 1];
                if (prev == 'e' || prev == 'E') continue;
                std::string tok = cd_strip(expr.substr(start, i - start));
                if (!tok.empty()) out.push_back(tok);
                start = i;
            }
        }
        if (start < expr.size()) {
            std::string tok = cd_strip(expr.substr(start));
            if (!tok.empty()) out.push_back(tok);
        }
        return out;
    }

    struct CdExpand { std::string sign; std::string coef; std::string remainder; };

    // По набору токенов tok с outer_coef-обёрткой раскладывает на (var_sign, remainder).
    // Используется в case 1 и case 2 (paren_mult и simple_paren).
    void cd_expand_paren_tokens(const std::vector<std::string>& tokens,
                                const std::string& var,
                                const std::string& outer_coef,
                                std::string& out_var_sign,
                                std::string& out_remainder) {
        out_var_sign = "+";
        std::vector<std::string> rem_parts;
        for (const auto& token : tokens) {
            std::string token_sign = "+";
            std::string token_body = token;
            if (!token.empty() && token[0] == '-') { token_sign = "-"; token_body = cd_strip(token.substr(1)); }
            else if (!token.empty() && token[0] == '+') { token_body = cd_strip(token.substr(1)); }

            try {
                std::regex var_re("^" + var + "$");
                if (std::regex_match(token_body, var_re)) { out_var_sign = token_sign; continue; }
            } catch (...) {}
            if (token_sign == "-") rem_parts.push_back("- " + outer_coef + " * " + token_body);
            else rem_parts.push_back("+ " + outer_coef + " * " + token_body);
        }
        std::string rem;
        for (auto& p : rem_parts) rem += (rem.empty() ? "" : " ") + p;
        if (!rem.empty() && rem[0] == '+') rem = cd_strip(rem.substr(1));
        out_remainder = rem;
    }

    CdExpand cd_expand_expression(const std::string& expr_in, const std::string& var) {
        std::string expr = cd_strip(expr_in);
        CdExpand r{ "+", "1", "" };

        // Case 1: (expr1) * (expr2), переменная только в одной из скобок.
        try {
            std::regex re(R"(^\(([^)]+)\)\s*\*\s*\(([^)]+)\)$)");
            std::smatch m;
            if (std::regex_match(expr, m, re)) {
                std::string left = m[1].str();
                std::string right = m[2].str();
                bool right_has = cd_contains_variable(right, var);
                bool left_has = cd_contains_variable(left, var);
                if (right_has && !left_has) {
                    std::string outer = "(" + left + ")";
                    std::string inner = right;
                    if (!inner.empty() && inner[0] != '+' && inner[0] != '-') inner = "+" + inner;
                    auto tokens = cd_split_terms(inner);
                    std::string vs, rem;
                    cd_expand_paren_tokens(tokens, var, outer, vs, rem);
                    r.sign = vs; r.coef = outer; r.remainder = rem;
                    return r;
                }
            }
        } catch (...) {}

        // Case 2: coef * (expr), где coef — простой идентификатор.
        try {
            std::regex re(R"(^([a-zA-Z_]\w*)\s*\*\s*\((.+)\)$)");
            std::smatch m;
            if (std::regex_match(expr, m, re)) {
                std::string outer = m[1].str();
                std::string inner = m[2].str();
                if (cd_contains_variable(inner, var)) {
                    if (!inner.empty() && inner[0] != '+' && inner[0] != '-') inner = "+" + inner;
                    auto tokens = cd_split_terms(inner);
                    std::string vs, rem;
                    cd_expand_paren_tokens(tokens, var, outer, vs, rem);
                    r.sign = vs; r.coef = outer; r.remainder = rem;
                    return r;
                }
            }
        } catch (...) {}

        // Case 3: обычная сумма членов.
        std::string expr2 = expr;
        if (!expr2.empty() && expr2[0] != '+' && expr2[0] != '-') expr2 = "+" + expr2;
        auto tokens = cd_split_terms(expr2);
        std::string var_sign = "+";
        std::string var_coef = "1";
        std::vector<std::string> rem_parts;
        bool found = false;
        for (const auto& token : tokens) {
            std::string token_sign = "+";
            std::string token_body = token;
            if (!token.empty() && token[0] == '-') { token_sign = "-"; token_body = cd_strip(token.substr(1)); }
            else if (!token.empty() && token[0] == '+') { token_body = cd_strip(token.substr(1)); }

            bool has = false;
            std::smatch m;
            try {
                std::regex r1("^([a-zA-Z_]\\w*)\\s*\\*\\s*" + var + "$");
                if (std::regex_match(token_body, m, r1)) { has = true; var_coef = m[1].str(); }
            } catch (...) {}
            if (!has) try {
                std::regex r2("^" + var + "\\s*\\*\\s*([a-zA-Z_]\\w*)$");
                if (std::regex_match(token_body, m, r2)) { has = true; var_coef = m[1].str(); }
            } catch (...) {}
            if (!has) try {
                std::regex r3("^" + var + "$");
                if (std::regex_match(token_body, r3)) { has = true; var_coef = "1"; }
            } catch (...) {}

            if (has && !found) { found = true; var_sign = token_sign; }
            else { rem_parts.push_back(token); }
        }
        std::string rem;
        for (auto& p : rem_parts) rem += (rem.empty() ? "" : " ") + p;
        if (!rem.empty() && rem[0] == '+') rem = cd_strip(rem.substr(1));
        r.sign = var_sign; r.coef = var_coef; r.remainder = rem;
        return r;
    }

    // Превращает plain-выражение с символическими именами в C-код:
    //   var^N  -> pow(var, N)
    //   ln(.)  -> log(.)
    //   var    -> X[i]
    //   param  -> a[1+j]
    std::string cd_expr_to_code(const std::string& expr_in,
                                const std::vector<std::string>& vars,
                                const std::vector<std::string>& params) {
        std::string code = expr_in;
        // 1) var^N -> pow(var, N) (до подстановки X[i], чтобы регекс распознал имя)
        for (size_t i = 0; i < vars.size(); ++i) {
            try {
                std::regex re("\\b" + vars[i] + "\\s*\\^\\s*(\\d+)");
                code = std::regex_replace(code, re, "pow(" + vars[i] + ", $1)");
            } catch (...) {}
        }
        // 2) X[i]^N -> pow(X[i], N) (на случай если уже было)
        try {
            std::regex re(R"((X\[\d+\])\s*\^\s*(\d+))");
            code = std::regex_replace(code, re, "pow($1, $2)");
        } catch (...) {}
        // 3) ln(  -> log(  (страховка; парсер уже мапит ln→log)
        try {
            std::regex re("\\bln\\s*\\(");
            code = std::regex_replace(code, re, "log(");
        } catch (...) {}
        // 4) переменные -> X[i]
        for (size_t i = 0; i < vars.size(); ++i) {
            try {
                std::regex re("\\b" + vars[i] + "\\b");
                code = std::regex_replace(code, re, "X[" + std::to_string(i) + "]");
            } catch (...) {}
        }
        // 5) параметры -> a[1+j]
        for (size_t j = 0; j < params.size(); ++j) {
            try {
                std::regex re("\\b" + params[j] + "\\b");
                code = std::regex_replace(code, re, "a[" + std::to_string(1 + (int)j) + "]");
            } catch (...) {}
        }
        return code;
    }

    // ---------- Схемы ----------
    std::string scheme_euler(const System& s) {
        int N = s.vars.size(); auto f = rhs_over(s, "X"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }
    std::string scheme_euler_cromer(const System& s) {
        auto f = rhs_over(s, "X"); std::ostringstream o;
        for (size_t i = 0; i < f.size(); ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n"; return o.str();
    }
    std::string scheme_midpoint(const System& s) {
        int N = s.vars.size();
        auto f0 = rhs_over(s, "X"); auto f1 = rhs_over(s, "X1"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + 0.5 * h * (" << f0[i] << ");\n";
        for (int i = 0; i < N; ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f1[i] << ");\n"; return o.str();
    }
    std::string scheme_rk4(const System& s) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    numb k[" << N << "][4];\n    int N = " << N << ";\n    int i, j;\n";
        o << "    for (i = 0; i < N; i++) {\n        X1[i] = X[i];\n    }\n";
        o << "    for (j = 0; j < 4; j++) {\n";
        for (int i = 0; i < N; ++i)o << "        k[" << i << "][j] = (" << k[i] << ");\n";
        o << "        if (j == 3) {\n            for (i = 0; i < N; i++) {\n            X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;\n            }\n        }\n";
        o << "        else if (j == 2) {\n            for (i = 0; i < N; i++) {\n            X1[i] = X[i] + h * k[i][j];\n            }\n        }\n";
        o << "        else {\n            for (i = 0; i < N; i++) {\n                X1[i] = X[i] + 0.5 * h * k[i][j];\n            }\n        }\n    }\n"; return o.str();
    }
    // Dormand-Prince 8(7), 13-стадийный. Матрицы M[13][12] и B[2][13] — static
    // const, компилятор положит в constant memory. Считаем y (8-й, b[0]) и z
    // (7-й, b[1]) — z пока не используется, но оставлен под будущий адаптивный
    // шаг по |y - z|. На каждый step X := y.
    std::string scheme_dopri78(const System& s) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;

        o << "    static const numb M[13][12] = {\n";
        o << "        {0,0,0,0,0,0,0,0,0,0,0,0},\n";
        o << "        {0.05555555555556,0,0,0,0,0,0,0,0,0,0,0},\n";
        o << "        {0.02083333333333,0.0625,0,0,0,0,0,0,0,0,0,0},\n";
        o << "        {0.03125,0,0.09375,0,0,0,0,0,0,0,0,0},\n";
        o << "        {0.3125,0,-1.171875,1.171875,0,0,0,0,0,0,0,0},\n";
        o << "        {0.0375,0,0,0.1875,0.15,0,0,0,0,0,0,0},\n";
        o << "        {0.04791013711111,0,0,0.1122487127778,-0.02550567377778,0.01284682388889,0,0,0,0,0,0},\n";
        o << "        {0.01691798978729,0,0,0.387848278486,0.0359773698515,0.1969702142157,-0.1727138523405,0,0,0,0,0},\n";
        o << "        {0.06909575335919,0,0,-0.6342479767289,-0.1611975752246,0.1386503094588,0.9409286140358,0.2116363264819,0,0,0,0},\n";
        o << "        {0.183556996839,0,0,-2.468768084316,-0.2912868878163,-0.02647302023312,2.847838764193,0.2813873314699,0.1237448998633,0,0,0},\n";
        o << "        {-1.215424817396,0,0,16.67260866595,0.9157418284168,-6.056605804357,-16.00357359416,14.8493030863,-13.37157573529,5.13418264818,0,0},\n";
        o << "        {0.2588609164383,0,0,-4.774485785489,-0.435093013777,-3.049483332072,5.577920039936,6.155831589861,-5.062104586737,2.193926173181,0.1346279986593,0},\n";
        o << "        {0.8224275996265,0,0,-11.65867325728,-0.7576221166909,0.7139735881596,12.07577498689,-2.12765911392,1.990166207049,-0.234286471544,0.1758985777079,0}\n";
        o << "    };\n";
        o << "    static const numb B[2][13] = {\n";
        o << "        {0.04174749114153,0,0,0,0,-0.05545232861124,0.2393128072012,0.7035106694034,-0.7597596138145,0.6605630309223,0.1581874825101,-0.2381095387529,0.25},\n";
        o << "        {0.02955321367635,0,0,0,0,-0.8286062764878,0.3112409000511,2.4673451906,-2.546941651842,1.443548583677,0.07941559588113,0.04444444444444,0}\n";
        o << "    };\n";

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb y[" << N << "];\n";
        o << "    numb z[" << N << "];\n";
        o << "    numb k[" << N << "][13];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < 13; ++i) {\n";
        for (int v = 0; v < N; ++v)
            o << "        k[" << v << "][i] = (" << k[v] << ");\n";
        o << "        if (i != 12) {\n";
        o << "            for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "            for (j = 0; j < i + 1; ++j)\n";
        o << "                for (l = 0; l < N; ++l)\n";
        o << "                    X2[l] += M[i + 1][j] * k[l][j];\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X1[l] = X[l] + h * X2[l];\n";
        o << "        }\n";
        o << "    }\n";

        // 8-й порядок: y
        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < 13; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[0][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        y[l] = X[l] + h * X2[l];\n";

        // 7-й порядок: z (future-proof для адаптивного шага)
        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < 13; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[1][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        z[l] = X[l] + h * X2[l];\n";
        o << "    (void)z[0];\n";

        o << "    for (l = 0; l < N; ++l) X[l] = y[l];\n";
        return o.str();
    }

    // ---------- CD: Composition D-method (диагонально-неявная схема) ----------
    // Из теории (см. PDF, формулы 8–9): Ψ_h,s = Φ_h1 ∘ Φ*_h2, где
    //   h1 = h * s, h2 = h * (1 - s), s — коэффициент симметрии (a[0]).
    // Φ_h1 — явный полушаг прямого порядка; Φ*_h2 — неявный полушаг обратного
    // порядка. Диагональная неявность по var_i решается аналитически когда RHS
    // линеен по var_i (формула X = (X + h2*rem) / (1 ± h2*coef)), и итерациями
    // (4 шага) когда RHS нелинеен. Структурно совпадает с Python-прототипом
    // в @CD.txt; здесь работает по plain-выражению, полученному из AST через
    // emit_plain (что даёт корректную работу и с LaTeX-входом).
    std::string scheme_cd(const System& s) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();
        if (N < 2) throw std::runtime_error("CD method requires N >= 2");

        // Парсим каждый RHS и эмитим в plain (нормализация LaTeX -> plain).
        std::vector<std::string> rhs_plain(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            std::ostringstream pp;
            emit_plain(ast, pp);
            rhs_plain[i] = pp.str();
        }

        std::ostringstream o;
        o << "    numb h1 = h * a[0];\n";
        o << "    numb h2 = h * (1 - a[0]);\n";

        // Явный полушаг: прямой порядок переменных.
        for (int i = 0; i < N; ++i) {
            std::string code = cd_expr_to_code(rhs_plain[i], s.vars, s.params);
            o << "    X[" << i << "] = X[" << i << "] + h1 * (" << code << ");\n";
        }

        // Неявный полушаг: обратный порядок.
        for (int i = N - 1; i >= 0; --i) {
            const std::string& var = s.vars[i];
            const std::string& rhs = rhs_plain[i];
            std::string x = "X[" + std::to_string(i) + "]";
            CdNonlin t = cd_get_nonlinearity_type(rhs, var);

            if (t == CdNonlin::None) {
                std::string rhs_code = cd_expr_to_code(rhs, s.vars, s.params);
                o << "    " << x << " = " << x << " + h2 * (" << rhs_code << ");\n";
            }
            else if (t == CdNonlin::Linear) {
                CdExpand e = cd_expand_expression(rhs, var);
                std::string rem_code = e.remainder.empty()
                    ? std::string()
                    : cd_expr_to_code(e.remainder, s.vars, s.params);
                std::string coef_code = (e.coef == "1")
                    ? std::string("1")
                    : cd_expr_to_code(e.coef, s.vars, s.params);
                // sign — знак при var в RHS. f = sign*coef*var + rem  =>
                //   var_new = (var + h2*rem) / (1 - sign*h2*coef)
                std::string denom_sign = (e.sign == "-") ? "+" : "-";
                std::string denom = (coef_code == "1")
                    ? std::string("(1 ") + denom_sign + " h2)"
                    : std::string("(1 ") + denom_sign + " h2 * " + coef_code + ")";
                if (!rem_code.empty())
                    o << "    " << x << " = (" << x << " + h2 * (" << rem_code << ")) / " << denom << ";\n";
                else
                    o << "    " << x << " = " << x << " / " << denom << ";\n";
            }
            else {
                // Нелинейный случай — 4 итерации простой итерации с фиксированным
                // "стартовым" значением переменной (одномерная неподвижная точка).
                std::string temp = "x" + std::to_string(i) + "_cd";
                std::string rhs_code = cd_expr_to_code(rhs, s.vars, s.params);
                o << "    numb " << temp << " = " << x << ";\n";
                for (int k = 0; k < 4; ++k)
                    o << "    " << x << " = " << temp << " + h2 * (" << rhs_code << ");\n";
            }
        }

        return o.str();
    }

    // CPU-equivalent of scheme_cd: all variables use 4 simple iterations,
    // no analytic Linear branch. Mirrors integrator.cpp::step_cd exactly so the
    // debug panel can show what the CPU integrator computes per step.
    std::string scheme_cd_iter_only(const System& s) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();
        if (N < 2) throw std::runtime_error("CD method requires N >= 2");

        std::vector<std::string> rhs_plain(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            std::ostringstream pp;
            emit_plain(ast, pp);
            rhs_plain[i] = pp.str();
        }

        std::ostringstream o;
        o << "    numb h1 = h * a[0];\n";
        o << "    numb h2 = h * (1 - a[0]);\n";

        // Explicit half-step, forward order (same as GPU).
        for (int i = 0; i < N; ++i) {
            std::string code = cd_expr_to_code(rhs_plain[i], s.vars, s.params);
            o << "    X[" << i << "] = X[" << i << "] + h1 * (" << code << ");\n";
        }

        // Implicit half-step, reverse order: 4 simple iterations for every var.
        for (int i = N - 1; i >= 0; --i) {
            std::string x = "X[" + std::to_string(i) + "]";
            std::string temp = "x" + std::to_string(i) + "_cd";
            std::string rhs_code = cd_expr_to_code(rhs_plain[i], s.vars, s.params);
            o << "    numb " << temp << " = " << x << ";\n";
            for (int k = 0; k < 4; ++k)
                o << "    " << x << " = " << temp << " + h2 * (" << rhs_code << ");\n";
        }

        return o.str();
    }

    // ---------- Байткод-интерпретатор (для CPU-расчёта без компиляции) ----------
    // Дерево выражения компилируется в постфиксную программу; вычисление идёт
    // по плоскому массиву инструкций на стеке — быстро и кэш-френдли.
    enum OpCode : int {
        OP_PUSH_CONST, OP_PUSH_VAR, OP_PUSH_PARAM,
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG,
        OP_FUNC1, OP_FUNC2
    };
    // id функций (унарные < 100, бинарные >= 100)
    enum FuncId : int {
        F_SIN, F_COS, F_TAN, F_ASIN, F_ACOS, F_ATAN,
        F_SINH, F_COSH, F_TANH, F_EXP, F_LOG, F_LOG2, F_LOG10,
        F_SQRT, F_CBRT, F_FABS,
        F_POW = 100, F_ATAN2, F_FMOD
    };
    struct Instr { int op; int idx; double val; };

    int func_id(const std::string& nm) {
        if (nm == "sin")return F_SIN; if (nm == "cos")return F_COS; if (nm == "tan")return F_TAN;
        if (nm == "asin")return F_ASIN; if (nm == "acos")return F_ACOS; if (nm == "atan")return F_ATAN;
        if (nm == "sinh")return F_SINH; if (nm == "cosh")return F_COSH; if (nm == "tanh")return F_TANH;
        if (nm == "exp")return F_EXP; if (nm == "log")return F_LOG; if (nm == "log2")return F_LOG2;
        if (nm == "log10")return F_LOG10; if (nm == "sqrt")return F_SQRT; if (nm == "cbrt")return F_CBRT;
        if (nm == "fabs" || nm == "abs")return F_FABS;
        if (nm == "pow")return F_POW; if (nm == "atan2")return F_ATAN2; if (nm == "fmod")return F_FMOD;
        throw std::runtime_error("eval: unknown function " + nm);
    }

    // Компиляция AST в постфиксный байткод.
    // var_index: имя переменной -> индекс в X. param_index: имя параметра -> индекс j (a[1+j]).
    struct ByteCompiler {
        std::vector<Instr>& out;
        const std::map<std::string, int>& var_index;
        const std::map<std::string, int>& param_index;

        void compile(const PN& n) {
            switch (n->kind) {
            case Node::Num: out.push_back({ OP_PUSH_CONST, 0, n->num }); break;
            case Node::Sym: {
                auto v = var_index.find(n->name);
                if (v != var_index.end()) { out.push_back({ OP_PUSH_VAR, v->second, 0 }); break; }
                auto p = param_index.find(n->name);
                if (p != param_index.end()) { out.push_back({ OP_PUSH_PARAM, p->second, 0 }); break; }
                if (n->name == "pi") { out.push_back({ OP_PUSH_CONST, 0, 3.14159265358979323846 }); break; }
                throw std::runtime_error("eval: unknown symbol " + n->name);
            }
            case Node::Add: compile(n->a); compile(n->b); out.push_back({ OP_ADD,0,0 }); break;
            case Node::Sub: compile(n->a); compile(n->b); out.push_back({ OP_SUB,0,0 }); break;
            case Node::Mul: compile(n->a); compile(n->b); out.push_back({ OP_MUL,0,0 }); break;
            case Node::Div: compile(n->a); compile(n->b); out.push_back({ OP_DIV,0,0 }); break;
            case Node::Pow: compile(n->a); compile(n->b); out.push_back({ OP_POW,0,0 }); break;
            case Node::Neg: compile(n->a); out.push_back({ OP_NEG,0,0 }); break;
            case Node::Call: {
                int fid = func_id(n->name);
                for (auto& arg : n->args) compile(arg);
                if (fid >= F_POW) {
                    if (n->args.size() != 2) throw std::runtime_error("eval: " + n->name + " needs 2 args");
                    out.push_back({ OP_FUNC2, fid, 0 });
                }
                else {
                    if (n->args.size() != 1) throw std::runtime_error("eval: " + n->name + " needs 1 arg");
                    out.push_back({ OP_FUNC1, fid, 0 });
                }
                break;
            }
            }
        }
    };

    double apply_func1(int fid, double x) {
        switch (fid) {
        case F_SIN:return std::sin(x); case F_COS:return std::cos(x); case F_TAN:return std::tan(x);
        case F_ASIN:return std::asin(x); case F_ACOS:return std::acos(x); case F_ATAN:return std::atan(x);
        case F_SINH:return std::sinh(x); case F_COSH:return std::cosh(x); case F_TANH:return std::tanh(x);
        case F_EXP:return std::exp(x); case F_LOG:return std::log(x); case F_LOG2:return std::log2(x);
        case F_LOG10:return std::log10(x); case F_SQRT:return std::sqrt(x); case F_CBRT:return std::cbrt(x);
        case F_FABS:return std::fabs(x);
        } return 0;
    }
    double apply_func2(int fid, double a, double b) {
        switch (fid) {
        case F_POW:return std::pow(a, b); case F_ATAN2:return std::atan2(a, b); case F_FMOD:return std::fmod(a, b);
        } return 0;
    }

    // Выполнить программу на стеке. a — параметры со сдвигом (a[0] reserved).
    double run_program(const std::vector<Instr>& prog, const double* X, const double* a,
        double* stack) {
        int sp = 0;
        for (const Instr& in : prog) {
            switch (in.op) {
            case OP_PUSH_CONST: stack[sp++] = in.val; break;
            case OP_PUSH_VAR:   stack[sp++] = X[in.idx]; break;
            case OP_PUSH_PARAM: stack[sp++] = a[1 + in.idx]; break; // сдвиг: a[0] reserved
            case OP_ADD: stack[sp - 2] = stack[sp - 2] + stack[sp - 1]; --sp; break;
            case OP_SUB: stack[sp - 2] = stack[sp - 2] - stack[sp - 1]; --sp; break;
            case OP_MUL: stack[sp - 2] = stack[sp - 2] * stack[sp - 1]; --sp; break;
            case OP_DIV: stack[sp - 2] = stack[sp - 2] / stack[sp - 1]; --sp; break;
            case OP_POW: stack[sp - 2] = std::pow(stack[sp - 2], stack[sp - 1]); --sp; break;
            case OP_NEG: stack[sp - 1] = -stack[sp - 1]; break;
            case OP_FUNC1: stack[sp - 1] = apply_func1(in.idx, stack[sp - 1]); break;
            case OP_FUNC2: stack[sp - 2] = apply_func2(in.idx, stack[sp - 2], stack[sp - 1]); --sp; break;
            }
        }
        return stack[0];
    }

} // anonymous namespace

// ---------- Реализация SystemEvaluator ----------
struct SystemEvaluator::Impl {
    int dim = 0;
    std::vector<std::vector<Instr>> programs; // по одной на уравнение
    int max_stack = 16;                        // глубина стека (с запасом)
    mutable std::vector<double> stack;
};

SystemEvaluator::SystemEvaluator(const System& sys) : impl_(new Impl) {
    if (sys.vars.size() != sys.rhs.size())
        throw std::runtime_error("SystemEvaluator: vars/rhs size mismatch");
    impl_->dim = (int)sys.vars.size();
    // индексы имён
    std::map<std::string, int> var_index, param_index;
    for (size_t i = 0; i < sys.vars.size(); ++i) var_index[sys.vars[i]] = (int)i;
    for (size_t j = 0; j < sys.params.size(); ++j) param_index[sys.params[j]] = (int)j;
    // парсим и компилируем каждое уравнение
    impl_->programs.resize(impl_->dim);
    int maxdepth = 8;
    for (int i = 0; i < impl_->dim; ++i) {
        Parser p(sys.rhs[i], sys.latex);
        PN ast = p.parse();
        ByteCompiler bc{ impl_->programs[i], var_index, param_index };
        bc.compile(ast);
        // оценка глубины стека: число push не превышает длину программы
        int depth = (int)impl_->programs[i].size() + 4;
        if (depth > maxdepth) maxdepth = depth;
    }
    impl_->max_stack = maxdepth;
    impl_->stack.resize(maxdepth);
}

SystemEvaluator::~SystemEvaluator() = default;
SystemEvaluator::SystemEvaluator(SystemEvaluator&&) noexcept = default;
SystemEvaluator& SystemEvaluator::operator=(SystemEvaluator&&) noexcept = default;

int SystemEvaluator::dim() const { return impl_->dim; }

void SystemEvaluator::eval(const double* X, const double* a, double* deriv) const {
    double* st = impl_->stack.data();
    for (int i = 0; i < impl_->dim; ++i)
        deriv[i] = run_program(impl_->programs[i], X, a, st);
}

// ---------- Публичная функция ----------
std::string codegen_scheme(const System& s, Scheme sch) {
    switch (sch) {
    case Scheme::Euler:            return scheme_euler(s);
    case Scheme::EulerCromer:      return scheme_euler_cromer(s);
    case Scheme::ExplicitMidpoint: return scheme_midpoint(s);
    case Scheme::RK4:              return scheme_rk4(s);
    case Scheme::DOPRI78:          return scheme_dopri78(s);
    case Scheme::CD:               return scheme_cd(s);
    }
    throw std::runtime_error("unknown scheme");
}

Scheme scheme_from_name(const std::string& name) {
    if (name == "Euler-Cromer")      return Scheme::EulerCromer;
    if (name == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (name == "RK4")               return Scheme::RK4;
    if (name == "DOPRI78")           return Scheme::DOPRI78;
    if (name == "CD")                return Scheme::CD;
    return Scheme::Euler;
}

std::string codegen_scheme_cpu_equivalent(const System& s, Scheme sch) {
    // For non-CD schemes the CPU integrator evaluates the same AST through a
    // bytecode interpreter; the resulting algorithm matches the codegen output
    // (same expression, same operation order). So we just return codegen_scheme.
    // Only CD has a genuinely different CPU algorithm — 4 simple iterations per
    // variable instead of the analytic linear solve used on GPU.
    if (sch == Scheme::CD) return scheme_cd_iter_only(s);
    return codegen_scheme(s, sch);
}


// ---------- Нормализация значения параметра ----------
std::string normalize_value(const std::string& value) {
    // trim
    size_t a = value.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return ""; // пустое = не задано
    size_t b = value.find_last_not_of(" \t\n\r");
    std::string v = value.substr(a, b - a + 1);

    // парсим как обычное выражение (не LaTeX) и печатаем в C-синтаксис.
    // числа получат .0, "8/3" -> "8.0 / 3.0".
    Parser p(v, false);
    PN ast = p.parse();
    NameMap empty; // значения не содержат имён переменных/параметров
    std::ostringstream o;
    emit(ast, empty, o);
    return o.str();
}