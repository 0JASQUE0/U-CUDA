// Реализация парсера (обычный синтаксис + LaTeX) и кодгена схем.
#include "codegen.hpp"
#include <map>
#include <set>
#include <memory>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <climits>
#include <limits>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

namespace { // внутренняя линковка: всё ниже не видно из других .cpp/.cu

    // Мини-AST
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

    // Парсер
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

    // Печать
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

    // CD helpers (AST-based).
    // Composition D-method (see chapter 1.1 of the theory PDF) assembles the
    // implicit half-step by trying to write the RHS f_i as coef*v + rem where
    // v == vars[i] and both coef, rem are v-free. On success the diagonal
    // implicit equation X = X_saved + h2*f admits the analytic solution
    //     X = (X_saved + h2*rem) / (1 - h2*coef)
    // (one division, no iterations). When v enters non-linearly -- through a
    // product v*v, division by v, or any pow/call argument -- the extraction
    // fails and we fall back to simple iterations, matching the CPU integrator
    // in integrator.cpp::step_cd exactly. This replaces an earlier regex-based
    // pass that miscounted parenthesised subtractions, missed numeric/compound
    // coefficients, and dropped repeated linear-in-v terms into the remainder.

    // Number of simple iterations used in the fallback path. Must stay in sync
    // with integrator.cpp::step_cd so that CPU and GPU trajectories match.
    constexpr int CD_ITERS = 4;

    // Complex CD: imaginary part of the half-step coefficients, sqrt(3)/6.
    // h1 = h*(0.5 + i*CCD_IMAG), h2 = conj(h1). Mirrored in
    // integrator.cpp::step_complex_cd — the CPU path must use the same value.
    constexpr double CCD_IMAG = 0.28867513459481288225;

    // В тело печатается ФОРМУЛА, а не это число: sqrt(3)/6 иррационален, и
    // десятичный литерал нёс бы только 17 цифр независимо от того, в какой
    // точности собрано тело. На dd/qd-ветке вкладки Order это держало бы
    // полку округления на уровне double, то есть сводило бы расширенную
    // точность к нулю на всех схемах семейства Complex CD.
    // Один sqrt на шаг против четырёх обращений к константе — цена мелкая на
    // фоне комплексной арифметики самой схемы.
    const char* const kCcdImagDecl =
        "    const numb ccd_im = sqrt((numb)3) / (numb)6;\n";

    // Коэффициент палиндромной тройки CCD4 (o4s3): alpha — комплексный корень
    // условий 2*alpha + beta = 1 и 2*alpha^3 + beta^3 = 0, то есть
    // alpha = 1/(2 - 2^(1/3)*exp(2*pi*i/3)). Раскрыв знаменатель:
    //   c = 2^(1/3),  D = 4 + 2c + c^2,
    //   Re alpha = (2 + c/2) / D,   Im alpha = (c*sqrt(3)/2) / D.
    // Считается в рантайме по той же причине, что и ccd_im: десятичный литерал
    // обрезал бы константу до 17 цифр независимо от точности numb.
    const char* const kCcd3Decl =
        "    const numb ccd3_c  = cbrt((numb)2);\n"
        "    const numb ccd3_d  = (numb)4 + (numb)2 * ccd3_c + ccd3_c * ccd3_c;\n"
        "    const numb ccd3_re = ((numb)2 + (numb)0.5 * ccd3_c) / ccd3_d;\n"
        "    const numb ccd3_im = (numb)0.5 * ccd3_c * sqrt((numb)3) / ccd3_d;\n";

    bool pn_contains_var(const PN& n, const std::string& v) {
        if (!n) return false;
        switch (n->kind) {
        case Node::Sym:  return n->name == v;
        case Node::Num:  return false;
        case Node::Neg:  return pn_contains_var(n->a, v);
        case Node::Add: case Node::Sub:
        case Node::Mul: case Node::Div:
        case Node::Pow:
            return pn_contains_var(n->a, v) || pn_contains_var(n->b, v);
        case Node::Call:
            for (auto& c : n->args) if (pn_contains_var(c, v)) return true;
            return false;
        }
        return false;
    }

    PN pn_num(double v) { auto n = mk(Node::Num); n->num = v; return n; }
    bool pn_is_zero(const PN& n) { return n && n->kind == Node::Num && n->num == 0.0; }
    bool pn_is_one (const PN& n) { return n && n->kind == Node::Num && n->num == 1.0; }

    // Peephole constructors: fold trivial identities so the generated C code
    // stays readable. Only algebraic no-ops (x*1, 1*x, x*0, 0*x, --x, -(Num)),
    // never re-associates. Correctness is unchanged.
    PN pn_neg(PN a) {
        if (a && a->kind == Node::Neg) return a->a;                       // -(-x) -> x
        if (a && a->kind == Node::Num) return pn_num(-a->num);            // -(c)  -> (-c)
        auto n = mk(Node::Neg); n->a = std::move(a); return n;
    }
    PN pn_add(PN a, PN b) { auto n = mk(Node::Add); n->a = std::move(a); n->b = std::move(b); return n; }
    PN pn_sub(PN a, PN b) { auto n = mk(Node::Sub); n->a = std::move(a); n->b = std::move(b); return n; }
    // Parser emits literal -1 as Neg(Num(1)) rather than Num(-1), so catch
    // both shapes here (my own pn_neg fold produces Num(-1), the parser does
    // not).
    bool pn_is_neg_one(const PN& n) {
        if (!n) return false;
        if (n->kind == Node::Num && n->num == -1.0) return true;
        if (n->kind == Node::Neg && pn_is_one(n->a)) return true;
        return false;
    }
    PN pn_mul(PN a, PN b) {
        if (pn_is_zero(a) || pn_is_zero(b)) return pn_num(0);
        if (pn_is_one(a)) return b;
        if (pn_is_one(b)) return a;
        if (pn_is_neg_one(a)) return pn_neg(std::move(b));                // (-1)*x -> -x
        if (pn_is_neg_one(b)) return pn_neg(std::move(a));                // x*(-1) -> -x
        auto n = mk(Node::Mul); n->a = std::move(a); n->b = std::move(b); return n;
    }
    PN pn_div(PN a, PN b) { auto n = mk(Node::Div); n->a = std::move(a); n->b = std::move(b); return n; }

    // Try to split n = coef*v + rem with coef, rem both free of v.
    // Returns false when v enters non-linearly and the analytic solve is
    // unsafe: product v*v, division by an expression that contains v, or v
    // appearing inside a pow-exponent or function call.
    bool cd_try_extract_linear(const PN& n, const std::string& v, PN& coef, PN& rem) {
        if (!n) { coef = pn_num(0); rem = pn_num(0); return true; }
        switch (n->kind) {
        case Node::Num:
            coef = pn_num(0); rem = n; return true;
        case Node::Sym:
            if (n->name == v) { coef = pn_num(1); rem = pn_num(0); }
            else              { coef = pn_num(0); rem = n; }
            return true;
        case Node::Neg: {
            PN ca, ra;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            coef = pn_is_zero(ca) ? pn_num(0) : pn_neg(ca);
            rem  = pn_is_zero(ra) ? pn_num(0) : pn_neg(ra);
            return true;
        }
        case Node::Add: {
            PN ca, ra, cb, rb;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            if (!cd_try_extract_linear(n->b, v, cb, rb)) return false;
            coef = pn_is_zero(ca) ? cb : (pn_is_zero(cb) ? ca : pn_add(ca, cb));
            rem  = pn_is_zero(ra) ? rb : (pn_is_zero(rb) ? ra : pn_add(ra, rb));
            return true;
        }
        case Node::Sub: {
            PN ca, ra, cb, rb;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            if (!cd_try_extract_linear(n->b, v, cb, rb)) return false;
            coef = pn_is_zero(cb) ? ca : (pn_is_zero(ca) ? pn_neg(cb) : pn_sub(ca, cb));
            rem  = pn_is_zero(rb) ? ra : (pn_is_zero(ra) ? pn_neg(rb) : pn_sub(ra, rb));
            return true;
        }
        case Node::Mul: {
            bool la = pn_contains_var(n->a, v);
            bool lb = pn_contains_var(n->b, v);
            if (la && lb) return false;                    // v*v-like coupling
            if (!la && !lb) { coef = pn_num(0); rem = n; return true; }
            PN linear_side = la ? n->a : n->b;
            PN const_side  = la ? n->b : n->a;
            PN ci, ri;
            if (!cd_try_extract_linear(linear_side, v, ci, ri)) return false;
            coef = pn_is_zero(ci) ? pn_num(0) : pn_mul(const_side, ci);
            rem  = pn_is_zero(ri) ? pn_num(0) : pn_mul(const_side, ri);
            return true;
        }
        case Node::Div: {
            if (pn_contains_var(n->b, v)) return false;    // v in denominator
            PN ci, ri;
            if (!cd_try_extract_linear(n->a, v, ci, ri)) return false;
            coef = pn_is_zero(ci) ? pn_num(0) : pn_div(ci, n->b);
            rem  = pn_is_zero(ri) ? pn_num(0) : pn_div(ri, n->b);
            return true;
        }
        case Node::Pow:
            if (pn_contains_var(n->a, v) || pn_contains_var(n->b, v)) return false;
            coef = pn_num(0); rem = n; return true;
        case Node::Call:
            for (auto& c : n->args) if (pn_contains_var(c, v)) return false;
            coef = pn_num(0); rem = n; return true;
        }
        return false;
    }

    std::string emit_to_str(const PN& n, const NameMap& nm) {
        std::ostringstream o; emit(n, nm, o); return o.str();
    }

    // Complex CD only: fmod and atan2 have no complex counterpart (both are
    // defined through the sign/magnitude of REAL arguments), so configCUDA.h
    // deliberately provides no ucmplx overload for them. Catch them here, at
    // codegen time, so the user gets the reason instead of an NVRTC template
    // error from the middle of a generated kernel.
    void cd_check_complex_safe(const PN& n) {
        if (!n) return;
        if (n->kind == Node::Call && (n->name == "fmod" || n->name == "atan2"))
            throw std::runtime_error("Complex CD: function " + n->name +
                " is not defined in complex arithmetic - pick another scheme");
        cd_check_complex_safe(n->a);
        cd_check_complex_safe(n->b);
        for (const auto& c : n->args) cd_check_complex_safe(c);
    }

    // ---------- Symbolic differentiation (implicit schemes) ----------
    //
    // d/dv over the same 9-kind AST the rest of codegen uses. Everything is built
    // through the pn_* peephole constructors above so that x*0, x*1 and -(-x) fold
    // on the way out -- without that a 3x3 Jacobian prints as unreadable noise.
    // pn_add / pn_sub deliberately do NOT fold (the CD emitter depends on their exact
    // shape), so zero-folding for the chain rule lives in the jd_* wrappers instead.

    constexpr double JD_LN2  = 0.69314718055994530942;
    constexpr double JD_LN10 = 2.30258509299404568402;

    PN pn_call1(const char* f, PN u) {
        auto n = mk(Node::Call); n->name = f; n->args.push_back(std::move(u)); return n;
    }
    PN pn_call2(const char* f, PN u, PN w) {
        auto n = mk(Node::Call); n->name = f;
        n->args.push_back(std::move(u)); n->args.push_back(std::move(w)); return n;
    }
    PN pn_pow(PN a, PN b) {
        // a^0 -> 1 and a^1 -> a, so the power rule never leaves pow(x, 1.0) behind
        if (b && b->kind == Node::Num) {
            if (b->num == 0.0) return pn_num(1);
            if (b->num == 1.0) return a;
        }
        auto n = mk(Node::Pow); n->a = std::move(a); n->b = std::move(b); return n;
    }
    PN jd_add(PN a, PN b) {
        if (pn_is_zero(a)) return b;
        if (pn_is_zero(b)) return a;
        return pn_add(std::move(a), std::move(b));
    }
    PN jd_sub(PN a, PN b) {
        if (pn_is_zero(b)) return a;
        if (pn_is_zero(a)) return pn_neg(std::move(b));
        return pn_sub(std::move(a), std::move(b));
    }

    PN pn_diff(const PN& n, const std::string& v);

    PN pn_diff_call(const PN& n, const std::string& v) {
        const std::string& f = n->name;
        // pow / atan2 / fmod are the only binary calls the parser produces
        if (f == "pow") {
            if (n->args.size() != 2) throw std::runtime_error("jacobian: pow needs 2 args");
            return pn_diff(pn_pow(n->args[0], n->args[1]), v);
        }
        if (f == "atan2") {
            if (n->args.size() != 2) throw std::runtime_error("jacobian: atan2 needs 2 args");
            const PN& u = n->args[0]; const PN& w = n->args[1];
            PN num = jd_sub(pn_mul(w, pn_diff(u, v)), pn_mul(u, pn_diff(w, v)));
            if (pn_is_zero(num)) return pn_num(0);
            return pn_div(num, pn_add(pn_mul(u, u), pn_mul(w, w)));
        }
        if (n->args.size() != 1) throw std::runtime_error("jacobian: " + f + " needs 1 arg");
        const PN& u = n->args[0];
        PN du = pn_diff(u, v);
        if (pn_is_zero(du)) return pn_num(0);

        PN g;  // outer derivative, d f / d u
        if      (f == "sin")   g = pn_call1("cos", u);
        else if (f == "cos")   g = pn_neg(pn_call1("sin", u));
        else if (f == "tan")   g = pn_div(pn_num(1), pn_mul(pn_call1("cos", u), pn_call1("cos", u)));
        else if (f == "asin")  g = pn_div(pn_num(1), pn_call1("sqrt", pn_sub(pn_num(1), pn_mul(u, u))));
        else if (f == "acos")  g = pn_neg(pn_div(pn_num(1), pn_call1("sqrt", pn_sub(pn_num(1), pn_mul(u, u)))));
        else if (f == "atan")  g = pn_div(pn_num(1), pn_add(pn_num(1), pn_mul(u, u)));
        else if (f == "sinh")  g = pn_call1("cosh", u);
        else if (f == "cosh")  g = pn_call1("sinh", u);
        else if (f == "tanh")  g = pn_sub(pn_num(1), pn_mul(pn_call1("tanh", u), pn_call1("tanh", u)));
        else if (f == "exp")   g = pn_call1("exp", u);
        else if (f == "log")   g = pn_div(pn_num(1), u);
        else if (f == "log2")  g = pn_div(pn_num(1), pn_mul(u, pn_num(JD_LN2)));
        else if (f == "log10") g = pn_div(pn_num(1), pn_mul(u, pn_num(JD_LN10)));
        else if (f == "sqrt")  g = pn_div(pn_num(1), pn_mul(pn_num(2), pn_call1("sqrt", u)));
        else if (f == "cbrt")  g = pn_div(pn_num(1), pn_mul(pn_num(3), pn_mul(pn_call1("cbrt", u), pn_call1("cbrt", u))));
        // sysparse rewrites every |...| as fabs(), so the Chua-family piecewise systems
        // in this project hang on this line. copysign rather than the device-side sign():
        // it is a builtin for nvcc, NVRTC and MSVC alike, so the same emitted text
        // compiles on every codegen consumer with no extra header. At u == 0 fabs has no
        // derivative anyway -- copysign(1,-0.0) is -1 there, sign(0) would be 0.
        else if (f == "fabs" || f == "abs") g = pn_call2("copysign", pn_num(1), u);
        else throw std::runtime_error("jacobian: no derivative for function " + f);

        return pn_mul(g, du);
    }

    PN pn_diff(const PN& n, const std::string& v) {
        if (!n) return pn_num(0);
        switch (n->kind) {
        case Node::Num: return pn_num(0);
        // params are Sym as well, and correctly differentiate to 0
        case Node::Sym: return pn_num(n->name == v ? 1.0 : 0.0);
        case Node::Neg: {
            PN d = pn_diff(n->a, v);
            return pn_is_zero(d) ? pn_num(0) : pn_neg(d);
        }
        case Node::Add: return jd_add(pn_diff(n->a, v), pn_diff(n->b, v));
        case Node::Sub: return jd_sub(pn_diff(n->a, v), pn_diff(n->b, v));
        case Node::Mul: return jd_add(pn_mul(pn_diff(n->a, v), n->b),
                                      pn_mul(n->a, pn_diff(n->b, v)));
        case Node::Div: {
            PN da = pn_diff(n->a, v), db = pn_diff(n->b, v);
            // v-free denominator is the common case; skip the quotient rule there
            if (pn_is_zero(db)) return pn_is_zero(da) ? pn_num(0) : pn_div(da, n->b);
            PN num = jd_sub(pn_mul(da, n->b), pn_mul(n->a, db));
            if (pn_is_zero(num)) return pn_num(0);
            return pn_div(num, pn_mul(n->b, n->b));
        }
        case Node::Pow: {
            PN da = pn_diff(n->a, v), db = pn_diff(n->b, v);
            if (pn_is_zero(da) && pn_is_zero(db)) return pn_num(0);
            if (pn_is_zero(db)) {
                // Power rule whenever the exponent is v-free (literal OR parameter).
                // The general form below needs log(a), undefined for a < 0, and
                // negative bases are ordinary here.
                PN em1 = (n->b->kind == Node::Num) ? pn_num(n->b->num - 1.0)
                                                   : pn_sub(n->b, pn_num(1));
                return pn_mul(pn_mul(n->b, pn_pow(n->a, em1)), da);
            }
            // general: a^b * (db*log(a) + b*da/a)
            PN t1 = pn_mul(db, pn_call1("log", n->a));
            PN t2 = pn_is_zero(da) ? pn_num(0) : pn_div(pn_mul(n->b, da), n->a);
            return pn_mul(pn_pow(n->a, n->b), jd_add(t1, t2));
        }
        case Node::Call: return pn_diff_call(n, v);
        }
        return pn_num(0);
    }

    // Same role as cd_check_complex_safe: refuse at codegen time, with a readable
    // reason, instead of letting NVRTC fail inside a generated kernel. These three
    // parse fine but have no usable derivative (0 almost everywhere / undefined).
    void jac_check_differentiable(const PN& n) {
        if (!n) return;
        if (n->kind == Node::Call &&
            (n->name == "fmod" || n->name == "floor" || n->name == "ceil"))
            throw std::runtime_error("Implicit scheme: function " + n->name +
                " is not differentiable, no Jacobian can be built -- pick an explicit scheme");
        jac_check_differentiable(n->a);
        jac_check_differentiable(n->b);
        for (const auto& c : n->args) jac_check_differentiable(c);
    }

    // Jacobian entries as C strings, row-major: out[i*N + j] = d f_i / d vars[j],
    // emitted over state array st. Counterpart of rhs_over above.
    std::vector<std::string> jac_over(const System& s, const std::string& st) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        NameMap nm = build_namemap(s, st);
        std::vector<std::string> out((size_t)N * N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            jac_check_differentiable(ast);
            for (int j = 0; j < N; ++j)
                out[(size_t)i * N + j] = emit_to_str(pn_diff(ast, s.vars[j]), nm);
        }
        return out;
    }

    // Схемы
    // Discrete map x_{n+1} = f(x_n): the right-hand side is the step, h unused.
    // The temp buffer is required - updating in place would be Gauss-Seidel.
    std::string scheme_map(const System& s) {
        int N = (int)s.vars.size(); auto f = rhs_over(s, "X"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i) o << "    X1[" << i << "] = (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }
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
    std::string scheme_dopri78(const System& s, bool legacy = false) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;

        // Коэффициент печатается суммой "(numb)(c0) + (numb)(c1) + ...": в double
        // она сворачивается при компиляции в c0, в dd/qd CPU-ветки вкладки Order
        // складывается точно (two_sum) и несёт 32/62 знака коэффициента.
        auto dlit = [](double v) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", v);
            return std::string(buf);
        };
        auto coef = [&](const MultiDoubleCoef& r) -> std::string {
            if (r.c[0] == 0.0) return "0";
            std::string t;
            for (int m = 0; m < 4 && r.c[m] != 0.0; ++m)
                t += (m ? " + (numb)(" : "(numb)(") + dlit(r.c[m]) + ")";
            return t;
        };
        auto table = [&](const char* decl, const MultiDoubleCoef* v, int rows, int cols) {
            o << "    static const numb " << decl << " = {\n";
            for (int i = 0; i < rows; ++i) {
                o << "        {";
                for (int j = 0; j < cols; ++j) o << (j ? ", " : "") << coef(v[i * cols + j]);
                o << "}" << (i + 1 < rows ? "," : "") << "\n";
            }
            o << "    };\n";
        };
        table("M[13][12]", legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12);
        table("B[2][13]",  legacy ? &kDopri78LegacyB[0][0] : &kDopri78B[0][0], 2, 13);

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

    // ---------- Implicit schemes: Newton on a symbolic Jacobian ----------
    //
    // ImplicitEuler:    F(Xn) = Xn - X - h*f(Xn) = 0,       X_next = Xn
    // ImplicitMidpoint: solved for the stage value Y = (X + X_next)/2,
    //                   F(Y)  = Y  - X - (h/2)*f(Y) = 0,    X_next = 2*Y - X
    // so both are the same emitted code with hc = h or h/2, differing only in the
    // final write-back. Unlike CD, which solves each equation for its own variable
    // and never sees the cross terms, this solves the FULL coupled system -- that is
    // what makes the two methods A-stable and worth their cost on stiff problems.
    //
    // Newton starts from an explicit-Euler predictor, which lands within O(h^2) of
    // the root, so 2-3 iterations are typical. System::newton_full picks the variant:
    //   false -- Jacobian and LU built once per step at the predictor and REUSED
    //            while the correction keeps shrinking; rebuilt as soon as it does
    //            not (modified Newton with a refresh). The refresh is not a
    //            refinement: a permanently frozen Jacobian diverges outright on a
    //            stiff nonlinear system - measured on Van der Pol at mu = 100,
    //            where h >= 0.01 blew up without it and reproduces the full-Newton
    //            answer with it, at about half the Jacobian evaluations.
    //   true  -- rebuilt every iteration (full Newton): quadratic convergence and
    //            the most robust at large h, at roughly k times the cost.
    // Iteration stops on ||dX||^2 < tol^2 or after newton_max_iters passes. tol and
    // the iteration cap are printed as literals, so they land in krs_body and thus in
    // the NVRTC cache key by themselves -- no extra placeholder, no manual eviction.
    //
    // The Gauss/LU solver is printed INTO the body instead of being shared from a
    // header, on purpose: the phase-portrait path hand-assembles its NVRTC source in
    // nvrtc_engine.cpp and pulls in configCUDA.h only when the body mentions ucmplx,
    // so a header helper would simply not be found there. DOPRI78 inlines its Butcher
    // tables for the same reason. Cost: numb Am[N*N] in local memory per thread --
    // 72 bytes at the usual N = 3, growing as N^2.
    // ImplicitKind::ComplexEuler — Complex Implicit Euler (см. комментарий к
    // Scheme в codegen.hpp): тот же ньютоновский блок, но прогоняется ДВАЖДЫ,
    // с tau1 = h*(a[0] + i*CIE_IMAG) и tau2 = h*(1 - a[0] - i*CIE_IMAG), и вся
    // арифметика комплексная. Общий код с вещественными схемами не ради
    // экономии строк: разъехавшись, две копии ньютона начали бы давать разные
    // ответы на одной и той же задаче, и это заметили бы не сразу.
    enum class ImplicitKind { Euler, Midpoint, ComplexEuler };

    // Мнимая часть полушага Complex Implicit Euler. Не настройка: 1/2 —
    // единственное значение, при котором tau1^2 + tau2^2 = 0 (см. codegen.hpp).
    constexpr double CIE_IMAG = 0.5;

    std::string scheme_implicit_common(const System& s, ImplicitKind kind) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();

        const bool  cx  = (kind == ImplicitKind::ComplexEuler);
        // Массивы состояния, из которых читают сгенерированные выражения:
        // X/Xn у вещественных схем (как было), Z/Zn у комплексной.
        const char* stv = cx ? "Z"  : "X";
        const char* stn = cx ? "Zn" : "Xn";
        const char* sty = cx ? "ucmplx" : "numb";

        auto f0 = rhs_over(s, stv);    // predictor, evaluated at the old state
        auto fn = rhs_over(s, stn);    // residual, evaluated at the Newton iterate
        auto Jn = jac_over(s, stn);    // Jacobian, same point as the residual

        if (cx) {
            // Те же запреты, что у Complex CD: у fabs/floor/fmod нет
            // аналитического продолжения в комплексную плоскость, и шаг с
            // мнимой частью по ним считать нечем.
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                cd_check_complex_safe(p.parse());
            }
        }

        const bool   full  = s.newton_full;
        const double tol   = s.newton_tol > 0.0 ? s.newton_tol : 1e-10;
        const int    maxit = s.newton_max_iters > 0 ? s.newton_max_iters : 1;

        // Компаунд-операторов у ucmplx нет (см. configCUDA.h), поэтому в
        // комплексном режиме печатаем развёрнутую форму. У вещественных схем
        // текст обязан остаться ПОБАЙТОВО прежним: он уходит в ключ кэша PTX
        // и в отладочную панель как «что считает CPU».
        auto sub_eq = [cx](const std::string& lhs, const std::string& rhs) {
            return cx ? (lhs + " = " + lhs + " - " + rhs + ";")
                      : (lhs + " -= " + rhs + ";");
        };
        auto div_eq = [cx](const std::string& lhs, const std::string& rhs) {
            return cx ? (lhs + " = " + lhs + " / " + rhs + ";")
                      : (lhs + " /= " + rhs + ";");
        };
        // Вещественный литерал в матрицу: у ucmplx конструктор из numb ЯВНЫЙ,
        // так что Am[k] = 1.0 в комплексном режиме просто не скомпилируется.
        auto lit = [cx](const char* v) {
            return cx ? ("ucmplx(" + std::string(v) + ", 0.0)") : std::string(v);
        };
        const std::string absf = cx ? "ucmplx_abs" : "fabs";

        std::ostringstream o;
        if (cx) {
            o << "    ucmplx Z[" << N << "];\n";
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = ucmplx(X[" << i << "], 0.0);\n";
        }
        o << "    " << sty << " " << stn << "[" << N << "];\n";
        o << "    " << sty << " Fv[" << N << "];\n";
        o << "    " << sty << " Am[" << N * N << "];\n";
        o << "    int  piv[" << N << "];\n";
        o << "    const int  ndim = " << N << ";\n";
        if (cx)
            // Переприсваивается между проходами, поэтому не const.
            o << "    ucmplx hc = ucmplx(a[0] * h, h * " << fmtnum(CIE_IMAG) << ");\n";
        else
            o << "    const numb hc = "
              << (kind == ImplicitKind::Euler ? "h" : "(0.5 * h)") << ";\n";
        o << "    const numb ntol = " << fmtnum(tol) << ";\n";
        o << "    int it, ik, ir, ic, ip, sing, refresh;\n";
        if (cx) {
            o << "    numb mx, av, nrm, prevn;\n";
            o << "    ucmplx dgn, mlt;\n";
        } else {
            o << "    numb mx, av, dgn, mlt, nrm, prevn;\n";
        }

        // Один ньютоновский проход целиком. У вещественных схем печатается
        // один раз, у комплексной — дважды, с разными tau (см. ниже).
        auto emit_pass = [&]() {
        // Predictor: explicit Euler over the same hc, so the Newton start is already
        // second-order accurate for the midpoint variant.
        o << "\n    /* predictor: explicit Euler */\n";
        for (int i = 0; i < N; ++i)
            o << "    " << stn << "[" << i << "] = " << stv << "[" << i
              << "] + hc * (" << f0[i] << ");\n";

        o << "\n    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < " << maxit << "; it++) {\n";

        // One emission of the Jacobian + LU, entered on iteration 0 and again
        // whenever `refresh` is set below. Full Newton sets it every iteration;
        // modified Newton only when the frozen matrix stops contracting.
        o << "        if (refresh) {\n";
        o << "            /* Am = I - hc * J(" << stn << ") */\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Am[" << (i * N + j) << "] = ";
                if (d == "0.0") o << lit(i == j ? "1.0" : "0.0") << ";\n";
                else if (i == j) o << "1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
        }
        o << "            /* LU with partial pivoting */\n";
        o << "            for (ik = 0; ik < ndim; ik++) {\n";
        o << "                ip = ik; mx = " << absf << "(Am[ik * ndim + ik]);\n";
        o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
        o << "                    av = " << absf << "(Am[ir * ndim + ik]);\n";
        o << "                    if (av > mx) { mx = av; ip = ir; }\n";
        o << "                }\n";
        o << "                piv[ik] = ip;\n";
        o << "                if (ip != ik)\n";
        o << "                    for (ic = 0; ic < ndim; ic++) {\n";
        o << "                        mlt = Am[ik * ndim + ic];\n";
        o << "                        Am[ik * ndim + ic] = Am[ip * ndim + ic];\n";
        o << "                        Am[ip * ndim + ic] = mlt;\n";
        o << "                    }\n";
        o << "                dgn = Am[ik * ndim + ik];\n";
        // Singular pivot: flag and bail instead of dividing. Nudging the pivot
        // would only turn the correction into inf and then nan, killing a
        // trajectory that is otherwise alive; leaving the step on the predictor
        // costs one order locally and nothing globally.
        o << "                if (" << absf << "(dgn) < 1e-30) { sing = 1; break; }\n";
        o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
        o << "                    mlt = Am[ir * ndim + ik] / dgn;\n";
        o << "                    Am[ir * ndim + ik] = mlt;\n";
        o << "                    for (ic = ik + 1; ic < ndim; ic++)\n";
        o << "                        " << sub_eq("Am[ir * ndim + ic]", "mlt * Am[ik * ndim + ic]") << "\n";
        o << "                }\n";
        o << "            }\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        o << "        if (sing) break;\n";

        o << "        /* residual F(" << stn << ") = " << stn << " - " << stv
          << " - hc * f(" << stn << ") */\n";
        for (int i = 0; i < N; ++i)
            o << "        Fv[" << i << "] = " << stn << "[" << i << "] - " << stv << "[" << i
              << "] - hc * (" << fn[i] << ");\n";
        o << "        /* solve Am * dX = Fv in place */\n";
        o << "        for (ik = 0; ik < ndim; ik++) {\n";
        o << "            ip = piv[ik];\n";
        o << "            if (ip != ik) { mlt = Fv[ik]; Fv[ik] = Fv[ip]; Fv[ip] = mlt; }\n";
        o << "            for (ir = ik + 1; ir < ndim; ir++)\n";
        o << "                " << sub_eq("Fv[ir]", "Am[ir * ndim + ik] * Fv[ik]") << "\n";
        o << "        }\n";
        o << "        for (ik = ndim - 1; ik >= 0; ik--) {\n";
        o << "            for (ic = ik + 1; ic < ndim; ic++)\n";
        o << "                " << sub_eq("Fv[ik]", "Am[ik * ndim + ic] * Fv[ic]") << "\n";
        o << "            " << div_eq("Fv[ik]", "Am[ik * ndim + ik]") << "\n";
        o << "        }\n";
        o << "        nrm = 0.0;\n";
        for (int i = 0; i < N; ++i) {
            if (cx)
                // |dX|^2 в комплексном случае — по модулю, а не по квадрату
                // самого числа: Fv[i]*Fv[i] у комплексного не вещественно.
                o << "        Zn[" << i << "] = Zn[" << i << "] - Fv[" << i << "];"
                  << " nrm += Fv[" << i << "].re * Fv[" << i << "].re + Fv[" << i
                  << "].im * Fv[" << i << "].im;\n";
            else
                o << "        Xn[" << i << "] -= Fv[" << i << "]; nrm += Fv[" << i
                  << "] * Fv[" << i << "];\n";
        }
        o << "        if (nrm < ntol * ntol) break;\n";
        if (full)
            o << "        refresh = 1;\n";
        else
            // nrm and prevn are SQUARED norms, so 0.25 is the 0.5 contraction
            // factor. Without this a frozen Jacobian diverges outright on a stiff
            // nonlinear system (measured on Van der Pol, mu = 100, h >= 0.01);
            // refreshing recovers the full-Newton answer at half the Jacobians.
            o << "        if (nrm > 0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n\n";

        if (kind == ImplicitKind::Euler)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Xn[" << i << "];\n";
        else if (kind == ImplicitKind::Midpoint)
            // Xn holds the stage value Y; recover the endpoint from it.
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = 2.0 * Xn[" << i << "] - X[" << i << "];\n";
        else
            // Комплексный проход кладёт результат обратно в Z: второй проход
            // стартует с него, а после второго Re Z уходит в X.
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = Zn[" << i << "];\n";
        };  // emit_pass

        emit_pass();
        if (cx) {
            // Второй полушаг — сопряжённый: tau2 = h*(1 - a[0]) - i*h*CIE_IMAG.
            // tau1 + tau2 = h при любом a[0], но h^2-член гасится только при
            // a[0] = 1/2, и только там метод второго порядка.
            o << "\n    hc = ucmplx((1 - a[0]) * h, -h * " << fmtnum(CIE_IMAG) << ");\n";
            emit_pass();
            // Наружу — только Re, как у Complex CD: всё вокруг
            // calculateDiscreteModel остаётся вещественным.
            o << "\n";
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";
        }

        return o.str();
    }
    std::string scheme_implicit_euler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::Euler); }
    std::string scheme_implicit_midpoint(const System& s) { return scheme_implicit_common(s, ImplicitKind::Midpoint); }
    std::string scheme_complex_ieuler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::ComplexEuler); }

    // Общий кирпич диагонально-неявного полушага: одно уравнение
    // X[i] = X_saved + hs * f_i(X), решённое относительно своей переменной.
    // Им пользуются Phi* в CD (hs = h2, обратный порядок) и стадия SIMP
    // (hs = h1, прямой порядок) — сам способ решения у них общий.
    //
    // Wrap a subterm in parentheses only when its top-level operator binds
    // less tightly than '*'/'/' (i.e. Add or Sub); otherwise emit it raw so
    // that `h2 * X[0] * X[1]` stays a left-associative multiplication chain
    // instead of turning into `h2 * (X[0] * X[1])`. Under FMA both forms
    // are algebraically equal but not bit-identical, and a chaotic system
    // (Lorenz, Rossler, ...) amplifies that ULP-level split over ~10^3-10^4
    // steps into visibly different trajectories.
    bool needs_paren_after_mul(const PN& n) {
        return n && (n->kind == Node::Add || n->kind == Node::Sub);
    }
    std::string wrap_paren(const std::string& s, bool w) {
        return w ? "(" + s + ")" : s;
    }
    // Absorb a leading minus of a rem/coef into the operator sign: peel one
    // Neg or a negative Num off the node, so that `- h2 * -a[1]` and
    // `+ h2 * -a[1]` come out as `+ h2 * a[1]` and `- h2 * a[1]` -- the
    // shape a person would write by hand, and one less negation for the
    // compiler to chase. Returns the sign character to emit before "h2 * ".
    char peel_sign(PN& n, char pos, char neg) {
        if (!n) return pos;
        if (n->kind == Node::Neg) { n = n->a; return neg; }
        if (n->kind == Node::Num && n->num < 0.0) { n = pn_num(-n->num); return neg; }
        return pos;
    }
    // "h2 * factor" -> "h2" when factor == 1 (post-peel-sign), same reason
    // as the pn_mul(x, Num(1)) fold: the multiplication reads like noise
    // both to a human and to the peephole compiler expects.
    std::string mul_step(const PN& factor, const std::string& factor_c, const char* hs) {
        return pn_is_one(factor) ? std::string(hs) : hs + (" * " + factor_c);
    }
    // saved — имя временной под X_saved в итерационной ветке: у Complex CD4 два
    // прохода живут в одной области видимости, поэтому имя приходит снаружи.
    void emit_diag_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                               const std::string& v, const std::string& x,
                               const NameMap& nm, const char* hs, const char* sty,
                               const std::string& saved) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);

        if (linear && pn_is_zero(coef)) {
            // v absent from f_i -> explicit one-shot update.
            char sgn = peel_sign(rem, '+', '-');
            std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
            o << "    " << x << " = " << x
              << " " << sgn << " " << mul_step(rem, rem_c, hs) << ";\n";
        }
        else if (linear) {
            // Denominator: 1 (- | +) hs * |coef|.
            char dsgn = peel_sign(coef, '-', '+');
            std::string coef_c = wrap_paren(emit_to_str(coef, nm), needs_paren_after_mul(coef));
            if (pn_is_zero(rem))
                o << "    " << x << " = " << x
                  << " / (1 " << dsgn << " " << mul_step(coef, coef_c, hs) << ");\n";
            else {
                // Numerator: X_saved (+ | -) hs * |rem|.
                char nsgn = peel_sign(rem, '+', '-');
                std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
                o << "    " << x << " = (" << x
                  << " " << nsgn << " " << mul_step(rem, rem_c, hs)
                  << ") / (1 " << dsgn << " " << mul_step(coef, coef_c, hs) << ");\n";
            }
        }
        else {
            // v enters non-linearly -> fixed-point iterations from X_saved.
            std::string rhs_c = emit_to_str(rhs_ast, nm);
            bool w = needs_paren_after_mul(rhs_ast);
            o << "    " << sty << " " << saved << " = " << x << ";\n";
            for (int k = 0; k < CD_ITERS; ++k)
                o << "    " << x << " = " << saved
                  << " + " << hs << " * " << wrap_paren(rhs_c, w) << ";\n";
        }
    }

    // CD: Composition D-method (diagonally-implicit symplectic composition).
    // Theory (PDF chapter 1.1): Psi_{h,s} = Phi_{h1} o Phi*_{h2} with
    // h1 = h*s and h2 = h*(1 - s), s = a[0] symmetry coefficient. Phi is an
    // explicit half-step in forward variable order (Euler-Cromer flavour);
    // Phi* is a diagonally-implicit half-step in reverse order, where each
    // equation X[i] = X_saved + h2 * f_i(X) is solved for its own X[i].
    // For every equation we ask cd_try_extract_linear whether f_i can be
    // written as coef*v + rem with v = vars[i] and coef, rem free of v:
    //   - success  -> analytic step X[i] = (X_saved + h2*rem) / (1 - h2*coef);
    //   - failure  -> CD_ITERS simple iterations, identical to the CPU path.
    // Extraction, unlike the previous regex pipeline, walks the AST directly
    // so parenthesised sub-expressions, numeric/compound coefficients and
    // repeated linear-in-v terms are handled uniformly.
    //
    // complex_mode == false -> classic CD, output unchanged.
    // complex_mode == true  -> Complex CD: same composition, same linear
    // extraction, same iteration fallback, but the half-steps carry an
    // imaginary part: h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6,
    // with s = a[0], the same symmetry slot CD uses (default 0.5). The whole
    // step runs over a local ucmplx Z[] copied from X[] on entry; only Re Z[i]
    // goes back into X[i], so everything outside calculateDiscreteModel stays
    // real and untouched. The imaginary part is NOT carried into the next
    // step: it lives inside one step only.
    //
    // Order stays 2 at s = 1/2, same as CD -- the gain is in the error
    // constant, not the order. With g1 = 1/2 + i*sqrt(3)/6, g2 = conj(g1): the
    // h^2 term of the composition carries g1^2 - g2^2 = i*sqrt(3)/3, i.e. it
    // is purely imaginary and Re drops it; the h^3 term keeps g1*g2 = 1/3,
    // which is real and survives. Measured against an RK4 reference
    // (h = 1e-6): Lorenz -- 2.00, error ~2.4x below CD at equal h; pendulum
    // with sin -- 2.00, ~7x below CD. Per-step cost is roughly 3x the real CD.
    // Away from s = 1/2 that h^2 factor picks up a real part (2s - 1), which Re
    // no longer removes, and the method drops to first order -- exactly what
    // real CD does for s != 1/2. So s is an experiment knob, not a free
    // parameter: 0.5 is the value the scheme is built around.
    //
    // CdKind::Cx4 -- Complex CD4, the same CD block emitted TWICE with the
    // complex coefficients moved one level up: pass 1 runs the whole symmetric
    // CD with step gamma*h, pass 2 with conj(gamma)*h, gamma = 1/2 + i*sqrt(3)/6.
    // That is the order-4 construction: a symmetric method (s = 1/2 makes the CD
    // block self-adjoint, so its expansion has only odd powers) composed with
    // alpha + beta = 1 and alpha^3 + beta^3 = 0; those two conditions have
    // exactly gamma, conj(gamma) as their solution. The h^3 term dies by the
    // second condition, h^4 is imaginary by the conjugate symmetry and goes with
    // Re, so the first surviving real term is h^5 -> global order 4 in two
    // sub-steps instead of the three a real triple jump needs.
    // Measured (reference DOPRI78 h = 1e-3, converged to ~1e-14): order 4.00 on
    // both Rossler (T = 10) and Lorenz (T = 2); at equal h the error sits within
    // ~1.3x of RK4 either way, at ~2.7x the operation count (205 vs 76 for a
    // 3D system). As with CD itself, s != 1/2 breaks the symmetry of the inner
    // block and drops the whole thing to first order.
    //
    // CdKind::Cx4s3 / Cx4s4 -- СИММЕТРИЧНЫЕ варианты того же построения.
    // Complex CD4 симметричной НЕ является: у композиции из двух стадий
    // палиндромность требует g1 = g2, а тогда sum g = 1 и sum g^3 = 0
    // несовместны и порядок падает до второго. Её четвёртый порядок держится
    // не на симметрии, а на том, что нарушенное условие веса 4 (координата при
    // [E1,E3]) чисто мнимое и его снимает Re. Замеренный дефект симметрии
    // ||Psi_{-h}(Psi_h(x)) - x|| у неё O(h^8) — ниже собственной ошибки, но не
    // ноль. Настоящая самосопряжённость начинается с трёх стадий:
    //
    //   o4s3: (alpha, 1 - 2*alpha, alpha), alpha = 1/(2 - 2^(1/3)*exp(2*pi*i/3))
    //         = 0.3243964040201712 + 0.1345862724908067i. Комплексная ветвь
    //         того же кубического уравнения, что даёт вещественный тройной
    //         прыжок Йошиды (1.3512, -1.7024, 1.3512), но без шага назад:
    //         Re у всех трёх коэффициентов положительна (min Re = 0.3244), а
    //         константа ошибки в 215 раз меньше (||err||2 = 0.0247 против 5.29).
    //   o4s4: (g/2, gc/2, gc/2, g/2) — это Complex CD4, симметризованная
    //         композицией с собственной сопряжённой на половинном шаге. Решение
    //         условий для четырёх палиндромных стадий единственно с точностью
    //         до сопряжения и выходит ровно этим. ||err||2 = 0.00759.
    //
    // Обе палиндромны, то есть самосопряжены как КОМПЛЕКСНЫЕ композиции; взятое
    // в конце шага Re оставляет дефект симметрии O(h^10) против O(h^8) у
    // Complex CD4. Разложение ошибки по чётным степеням h замерено у обеих
    // (экстраполяция Ричардсона по h^2 даёт наклон 6.0), поэтому в
    // builtin_scheme_traits они помечены symmetric = true.
    // При РАВНОЙ работе (одинаковом суммарном числе проходов CD) все три схемы
    // дают одну и ту же ошибку с точностью до 4%: симметрия здесь достаётся
    // даром, но и выигрыша в точности не приносит — она нужна ради обратимости
    // отображения и законной h^2-экстраполяции.
    // Обе, как и Complex CD4, требуют s = a[0] = 1/2.
    //
    // CdKind::Cx4ss01 -- CCD4 (s-split)_{0-1}, двойственная к Complex CD4: там
    // комплексны внешние шаги, а CD внутри симметричен; здесь внешние шаги
    // вещественные (h/2 и h/2), а комплексно само разбиение CD:
    //   проход 1: h1 = (1-i)h/6, h2 = (2+i)h/6    (s1 = (1-i)/3)
    //   проход 2: h1 = (2+i)h/6, h2 = (1-i)h/6    (s2 = 1 - s1)
    // Полушаги a, b, b, a с a + b = 1/2 -- палиндром в смысле Phi/Phi*, поэтому
    // КОМПЛЕКСНАЯ композиция самосопряжена и в log Psi нет h^2 и h^4. Оба
    // коэффициента при h^3 чисто мнимые: 2(a^3 + b^3) = i/12 при N3 и
    // -7i/72 при [N1, N2] (N_k -- ряд полушага Эйлера; знак -- для записи
    // M = E*_a E_b E*_b E_a, правый множитель применяется первым),
    // Re их снимает, первый вещественный член -- h^5, глобальный порядок 4.
    // Среди двухпроходных CD со своими комплексными s_k решений порядка 4
    // ровно две сопряжённые пары: Complex CD4 и эта.
    // Замерено 4.0 на Лоренце и Рёсслере. Константа ошибки против Complex CD4
    // зависит от задачи: Лоренц -- в 2.9 раза меньше, Рёсслер -- в 3.1 раза
    // больше; по норме главного члена (веса 5) 0.0242 против 0.0134.
    // Симметричной НЕ помечена, хотя комплексная композиция самосопряжена:
    // мнимая часть появляется уже при h^3, и Re посреди пары шагов даёт дефект
    // симметрии O(h^6) (у Complex CD4 -- O(h^8)); в глобальной ошибке есть h^5,
    // экстраполяция Ричардсона по h^2 даёт 5, а не 6.
    // Коэффициенты фиксированы, a[0] не читается: s здесь -- часть метода, а
    // не ручка, и любое другое разбиение теряет порядок. Перестановка
    // b, a, a, b тоже: коммутаторный член получает вещественную часть 1/12,
    // и порядок падает до 2 -- на краях обязан стоять (1-i)/6.
    //
    // CdKind::Cx4ss10 -- CCD4 (s-split)_{1-0}: те же полушаги a, b, b, a, но в
    // каждом проходе ПЕРВОЙ идёт неявная половина (обратный порядок), затем
    // явная: E*_a, E_b | E*_b, E_a. Это та же композиция над сопряжённым
    // базовым методом: N_2 меняет знак, член при N3 остаётся i/12, при
    // [N1, N2] знак меняется на +7i/72 -- оба по-прежнему мнимые, порядок 4
    // (замерено 4.0 на Лоренце и Рёсслере). Главный член ошибки на
    // осцилляторе зеркален {0-1}, на нелинейных задачах выигрыш разный:
    // Лоренц -- {0-1} в 10 раз точнее, Рёсслер -- {1-0} в 2.8 раза точнее.
    enum class CdKind { Real, Cx, Cx4, Cx4s3, Cx4s4, Cx4ss01, Cx4ss10 };

    // core_only -- ядро для ExtrZ: только проходы над Z[], объявленным снаружи,
    // без перехода X -> Z на входе и без Re на выходе (см. codegen.hpp).
    // flip_halves -- вариант {1-0} любой CD-схемы: в каждом проходе сначала
    // неявная половина (h1, обратный порядок), потом явная (h2, прямой). Это
    // та же композиция над сопряжённым базовым методом: условия порядка
    // переходят в себя заменой N_k -> (-1)^(k+1) N_k, порядок и симметрия не
    // меняются. Замерено (Лоренц, Рёсслер): CD и Complex CD -- 2, Complex CD4,
    // o4s3, o4s4 -- 4; у CD4-семейства {1-0} точнее {0-1} в 1.5-4 раза.
    std::string scheme_cd_common(const System& s, CdKind kind, bool core_only = false,
                                 bool flip_halves = false) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();
        if (N < 2) throw std::runtime_error("CD method requires N >= 2");

        const bool cx = (kind != CdKind::Real);
        // State array the generated expressions read from: X[] for the real
        // scheme (in-place, as before), Z[] for the complex ones.
        const char* stv = cx ? "Z" : "X";
        const char* sty = cx ? "ucmplx" : "numb";

        NameMap nm = build_namemap(s, stv);
        std::vector<PN> rhs_ast(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            rhs_ast[i] = p.parse();
        }
        if (cx)
            for (int i = 0; i < N; ++i) cd_check_complex_safe(rhs_ast[i]);

        std::ostringstream o;
        if (cx && !core_only) {
            o << "    ucmplx Z[" << N << "];\n";
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = ucmplx(X[" << i << "], 0.0);\n";
        }
        // Полушаги h1/h2 каждого прохода композиции — выражениями над
        // объявленными выше переменными. Пусто у CD и Complex CD: у них ровно
        // один проход на весь h.
        std::vector<std::string> pass_h1, pass_h2;
        if (kind == CdKind::Real) {
            o << "    numb h1 = h * a[0];\n";
            o << "    numb h2 = h * (1 - a[0]);\n";
        }
        else if (kind == CdKind::Cx) {
            o << kCcdImagDecl;
            o << "    ucmplx h1 = ucmplx(a[0] * h,  h * ccd_im);\n";
            o << "    ucmplx h2 = ucmplx((1 - a[0]) * h, -h * ccd_im);\n";
        }
        else if (kind == CdKind::Cx4ss01 || kind == CdKind::Cx4ss10) {
            // Разбиение комплексное и фиксированное, a[0] не участвует:
            // проход 1 -- (p, q), проход 2 -- (q, p). У {1-0} h1 -- это
            // неявная половина, она идёт первой (см. emit_pass).
            o << "    ucmplx cp = ucmplx(h / (numb)6, -h / (numb)6);   // (1-i)h/6\n";
            o << "    ucmplx cq = ucmplx(h / (numb)3,  h / (numb)6);   // (2+i)h/6\n";
            pass_h1 = { "cp", "cq" };
            pass_h2 = { "cq", "cp" };
            o << "    ucmplx h1 = " << pass_h1[0] << ";\n";
            o << "    ucmplx h2 = " << pass_h2[0] << ";\n";
        }
        else {
            // Комплексная композиция: s делит уже СВОЙ комплексный подшаг
            // внутри прохода, поэтому здесь объявляются только сами подшаги,
            // а h1/h2 перевыставляются перед каждым проходом.
            // ccd_im нужен всем, кроме o4s3: у той свой коэффициент, и лишняя
            // константа в теле была бы неиспользованной.
            std::vector<std::string> cgs;
            if (kind != CdKind::Cx4s3) o << kCcdImagDecl;
            if (kind == CdKind::Cx4) {
                // gamma*h and conj(gamma)*h.
                o << "    ucmplx g  = ucmplx(0.5 * h,  h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx(0.5 * h, -h * ccd_im);\n";
                cgs = { "g", "gc" };
            }
            else if (kind == CdKind::Cx4s4) {
                // (gamma/2, conj/2, conj/2, gamma/2) — палиндром.
                o << "    ucmplx g  = ucmplx(0.25 * h,  0.5 * h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx(0.25 * h, -0.5 * h * ccd_im);\n";
                cgs = { "g", "gc", "gc", "g" };
            }
            else {
                // (alpha, 1 - 2*alpha, alpha) — палиндром. Средний коэффициент
                // выражен через alpha, а не литералом: тогда сумма подшагов
                // равна h ТОЧНО и в плавающей арифметике тоже.
                o << kCcd3Decl;
                o << "    ucmplx ga = ucmplx(ccd3_re * h, ccd3_im * h);\n";
                o << "    ucmplx gb = ucmplx(((numb)1 - (numb)2 * ccd3_re) * h,"
                     " -(numb)2 * ccd3_im * h);\n";
                cgs = { "ga", "gb", "ga" };
            }
            for (const std::string& g : cgs) {
                pass_h1.push_back(g + " * a[0]");
                pass_h2.push_back(g + " * (1 - a[0])");
            }
            o << "    ucmplx h1 = " << pass_h1[0] << ";\n";
            o << "    ucmplx h2 = " << pass_h2[0] << ";\n";
        }

        // Phi*_{h2}: diagonally-implicit half-step, reverse order. Сам разбор
        // уравнения — в emit_diag_implicit_eq выше: стадия SIMP гоняет тот же
        // код с h1 и в прямом порядке.

        // Один проход CD целиком: явный полушаг h1 вперёд, неявный h2 назад.
        // Complex CD4 зовёт его дважды с разными h1/h2, поэтому имена временных
        // переменных неявной ветки получают суффикс — иначе второй проход
        // переобъявил бы x0_cd в той же области видимости.
        // У вариантов {1-0} (CCD4 (s-split)_{1-0} и flip_halves) половины
        // меняются местами: неявная (h1, обратный порядок) первой, явная (h2,
        // прямой порядок) второй.
        const bool implicit_first = flip_halves || (kind == CdKind::Cx4ss10);
        auto emit_pass = [&](const char* sfx) {
        // Phi_{h1}: explicit half-step, forward order. Each X[i] update sees
        // the just-written values of X[0..i-1] (Euler-Cromer coupling).
        auto emit_explicit = [&](const char* step) {
            for (int i = 0; i < N; ++i)
                o << "    " << stv << "[" << i << "] = " << stv << "[" << i << "] + "
                  << step << " * (" << emit_to_str(rhs_ast[i], nm) << ");\n";
        };
        auto emit_implicit = [&](const char* step) {
            for (int i = N - 1; i >= 0; --i)
                emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i],
                                      stv + ("[" + std::to_string(i) + "]"), nm,
                                      step, sty, "x" + std::to_string(i) + "_cd" + sfx);
        };
        if (implicit_first) { emit_implicit("h1"); emit_explicit("h2"); }
        else                { emit_explicit("h1"); emit_implicit("h2"); }
        };  // emit_pass

        if (pass_h1.empty()) {
            emit_pass("");
        }
        else {
            for (size_t k = 0; k < pass_h1.size(); ++k) {
                if (k) {
                    o << "    h1 = " << pass_h1[k] << ";\n";
                    o << "    h2 = " << pass_h2[k] << ";\n";
                }
                // Суффикс временных неявной ветки: без него второй проход
                // переобъявил бы x0_cd в той же области видимости.
                const std::string sfx = k ? ("_" + std::to_string(k)) : std::string();
                emit_pass(sfx.c_str());
            }
        }

        // Наружу — только действительная часть: X[] вещественный и на входе, и
        // на выходе, поэтому вся обвязка (ядра, LLE/LS, бассейны, рендер) о
        // комплексности не знает.
        if (cx && !core_only)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";

        return o.str();
    }

    std::string scheme_cd(const System& s)          { return scheme_cd_common(s, CdKind::Real); }
    std::string scheme_complex_cd(const System& s)  { return scheme_cd_common(s, CdKind::Cx); }
    std::string scheme_complex_cd4(const System& s) { return scheme_cd_common(s, CdKind::Cx4); }
    std::string scheme_complex_cd4_s3(const System& s) { return scheme_cd_common(s, CdKind::Cx4s3); }
    std::string scheme_complex_cd4_s4(const System& s) { return scheme_cd_common(s, CdKind::Cx4s4); }
    std::string scheme_complex_cd4_ss01(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss01); }
    std::string scheme_complex_cd4_ss10(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss10); }

    // SEMP / SIMP — методы средней точки, у которых СТАДИЯ считается
    // последовательно по компонентам (Гаусс-Зейдель) вместо полной неявной
    // системы Implicit Midpoint:
    //
    //   X1 = X                                  — база для корректора
    //   стадия, шаг h1 = s*h, прямой порядок, in-place по X:
    //     SEMP: X[i] = X[i] + h1 * f_i(X)       — явно, i-е уравнение уже видит
    //           новые X[0..i-1] (связка как у Euler-Cromer);
    //     SIMP: X[i] = X_saved + h1 * f_i(X)    — решено относительно своей
    //           переменной, той же аналитикой/итерациями, что Phi* в CD.
    //   корректор — полный шаг от X1, все уравнения читают стадию:
    //     X1[i] = X1[i] + h * f_i(X);  затем X = X1.
    //
    // Порядок 2 достигается ТОЛЬКО при s = 1/2: разложение даёт
    // X_next = X + h*F + s*h^2*F'F против h^2/2*F'F у точного решения, так что
    // при s != 1/2 обе схемы падают до первого — ровно как CD. Перестановка
    // аргументов и диагональная неявность порядок не портят: стадия
    // возмущается на O(h^2), а входит в корректор с множителем h, то есть
    // локально это O(h^3).
    // Замерено на эмитируемом коде (эталон RK4 h = 1e-6, s = 0.5): Rossler
    // T = 10 — 2.00 у обеих схем, ошибка примерно вчетверо ниже, чем у явной
    // средней точки при равном h; Lorenz T = 2 — 2.00 у обеих, причём SEMP на
    // этой системе точнее SIMP примерно на порядок. При s = 0.3 обе дают 1.00.
    std::string scheme_semi_midpoint(const System& s, bool implicit_stage) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        NameMap nm = build_namemap(s, "X");
        std::vector<PN> rhs_ast(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            rhs_ast[i] = p.parse();
        }

        std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        o << "    numb h1 = h * a[0];\n";
        for (int i = 0; i < N; ++i)
            o << "    X1[" << i << "] = X[" << i << "];\n";

        for (int i = 0; i < N; ++i) {
            const std::string x = "X[" + std::to_string(i) + "]";
            const std::string saved = "x" + std::to_string(i) + "_si";
            if (!implicit_stage)
                o << "    " << x << " = " << x << " + h1 * ("
                  << emit_to_str(rhs_ast[i], nm) << ");\n";
            else
                emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i], x, nm,
                                      "h1", "numb", saved);
        }

        for (int i = 0; i < N; ++i)
            o << "    X1[" << i << "] = X1[" << i << "] + h * ("
              << emit_to_str(rhs_ast[i], nm) << ");\n";
        for (int i = 0; i < N; ++i)
            o << "    X[" << i << "] = X1[" << i << "];\n";
        return o.str();
    }
    std::string scheme_semp(const System& s) { return scheme_semi_midpoint(s, false); }
    std::string scheme_simp(const System& s) { return scheme_semi_midpoint(s, true); }

    // D — диагонально-неявный метод первого порядка, он же Phi* из CD и он же
    // стадия SIMP, взятая как самостоятельный шаг:
    //   для i = 0..N-1:  X[i] = X_saved + h * f_i(X),
    // решённое относительно своей переменной (аналитически, если f_i линейна по
    // ней, иначе CD_ITERS простых итераций), в прямом порядке — i-е уравнение
    // уже видит новые X[0..i-1]. Диагонально-неявный близнец Euler-Cromer:
    // та же связка по компонентам, но каждое уравнение решается, а не считается.
    // Перекрёстные члены (как и у CD) в решение не входят — это НЕ полный
    // неявный Эйлер, здесь нет ни якобиана, ни Ньютона.
    // Коэффициент симметрии a[0] не используется: шаг не делится пополам.
    // Замерено (эталон RK4 h = 1e-6): Rossler T = 10 — 1.00, Lorenz T = 2 — 1.00.
    std::string scheme_d(const System& s) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        NameMap nm = build_namemap(s, "X");
        std::ostringstream o;
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            const std::string x = "X[" + std::to_string(i) + "]";
            const std::string saved = "x" + std::to_string(i) + "_d";
            emit_diag_implicit_eq(o, ast, s.vars[i], x, nm, "h", "numb", saved);
        }
        return o.str();
    }

    // Имя-маркер множителя шага в AST (опкод OP_PUSH_STEP ниже). Парсер такое
    // имя выдать не может — символы систем это идентификаторы, — поэтому
    // столкновение с переменной или параметром исключено.
    const char* const kStepSym = "@h";

    // Собирает выражение "hs * expr" в ТОЙ ЖЕ группировке, в какой его получит
    // компилятор из текста, который печатает emit_diag_implicit_eq. mul_step
    // выводит "hs * " + текст множителя, а скобки вокруг множителя ставятся
    // только при Add/Sub сверху (needs_paren_after_mul), поэтому "hs * a * b"
    // разбирается как ((hs*a)*b): множитель шага заходит в САМЫЙ ЛЕВЫЙ конец
    // цепочки умножений и делений. Интерпретатор, посчитав hs*(a*b), разошёлся
    // бы с GPU на последний бит — в хаотической системе это видно уже через
    // тысячу шагов, ровно как с --fmad.
    PN pn_mul_step(const PN& e) {
        if (e && (e->kind == Node::Mul || e->kind == Node::Div)) {
            PN n = mk(e->kind);
            n->a = pn_mul_step(e->a);
            n->b = e->b;
            return n;
        }
        PN step = mk(Node::Sym); step->name = kStepSym;
        return pn_mul(std::move(step), e);
    }

    // Байткод-интерпретатор (для CPU-расчёта без компиляции)
    // Дерево выражения компилируется в постфиксную программу; вычисление идёт
    // по плоскому массиву инструкций на стеке — быстро и кэш-френдли.
    enum OpCode : int {
        OP_PUSH_CONST, OP_PUSH_VAR, OP_PUSH_PARAM,
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW, OP_NEG,
        OP_FUNC1, OP_FUNC2,
        OP_PUSH_STEP   // множитель шага (h, h1, h2) — приходит в run_program
    };
    // id функций (унарные < 100, бинарные >= 100)
    enum FuncId : int {
        F_SIN, F_COS, F_TAN, F_ASIN, F_ACOS, F_ATAN,
        F_SINH, F_COSH, F_TANH, F_EXP, F_LOG, F_LOG2, F_LOG10,
        F_SQRT, F_CBRT, F_FABS,
        // F_COPYSIGN is emitted only by pn_diff (derivative of fabs); the user-facing
        // parser does not accept it, so it is absent from known_funcs().
        F_POW = 100, F_ATAN2, F_FMOD, F_COPYSIGN
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
        if (nm == "copysign")return F_COPYSIGN;
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
                if (n->name == kStepSym) { out.push_back({ OP_PUSH_STEP,0,0 }); break; }
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
        case F_COPYSIGN:return std::copysign(a, b);
        } return 0;
    }

    // Комплексные версии — те же id, те же формулы, что и в device-коде: обе
    // ветки зовут функции из configCUDA.h, поэтому CPU-траектория Complex CD
    // повторяет GPU-шную операция в операцию (в пределах разной группировки FMA).
    // atan2/fmod комплексного смысла не имеют и до сюда не доходят: система с
    // ними отсекается в cd_check_complex_safe ещё на кодгене.
    ucmplx apply_func1(int fid, ucmplx x) {
        switch (fid) {
        case F_SIN:return sin(x); case F_COS:return cos(x); case F_TAN:return tan(x);
        case F_ASIN:return asin(x); case F_ACOS:return acos(x); case F_ATAN:return atan(x);
        case F_SINH:return sinh(x); case F_COSH:return cosh(x); case F_TANH:return tanh(x);
        case F_EXP:return exp(x); case F_LOG:return log(x); case F_LOG2:return log2(x);
        case F_LOG10:return log10(x); case F_SQRT:return sqrt(x); case F_CBRT:return cbrt(x);
        case F_FABS:return fabs(x);
        } return ucmplx(0, 0);
    }
    ucmplx apply_func2(int fid, ucmplx a, ucmplx b) {
        switch (fid) {
        case F_POW:return pow(a, b);
        } return ucmplx(0, 0);
    }

    // Степень отдельной функцией: у double это std::pow, у ucmplx — перегрузка
    // из configCUDA.h (со спрямлением целых показателей). Нужна, чтобы
    // run_program остался одним шаблоном на оба типа состояния.
    double  op_pow(double a, double b) { return std::pow(a, b); }
    ucmplx  op_pow(ucmplx a, ucmplx b) { return pow(a, b); }

    // Выполнить программу на стеке. a — параметры со сдвигом (a[0] reserved).
    // T — тип состояния: double для обычных схем, ucmplx для Complex CD.
    // Параметры a[] в обоих случаях вещественные и поднимаются в T на push'е.
    template <class T>
    T run_program(const std::vector<Instr>& prog, const T* X, const double* a,
        T* stack, T step = T(0)) {
        int sp = 0;
        for (const Instr& in : prog) {
            switch (in.op) {
            case OP_PUSH_CONST: stack[sp++] = T(in.val); break;
            case OP_PUSH_VAR:   stack[sp++] = X[in.idx]; break;
            case OP_PUSH_PARAM: stack[sp++] = T(a[1 + in.idx]); break; // сдвиг: a[0] reserved
            case OP_PUSH_STEP:  stack[sp++] = step; break;
            case OP_ADD: stack[sp - 2] = stack[sp - 2] + stack[sp - 1]; --sp; break;
            case OP_SUB: stack[sp - 2] = stack[sp - 2] - stack[sp - 1]; --sp; break;
            case OP_MUL: stack[sp - 2] = stack[sp - 2] * stack[sp - 1]; --sp; break;
            case OP_DIV: stack[sp - 2] = stack[sp - 2] / stack[sp - 1]; --sp; break;
            case OP_POW: stack[sp - 2] = op_pow(stack[sp - 2], stack[sp - 1]); --sp; break;
            case OP_NEG: stack[sp - 1] = -stack[sp - 1]; break;
            case OP_FUNC1: stack[sp - 1] = apply_func1(in.idx, stack[sp - 1]); break;
            case OP_FUNC2: stack[sp - 2] = apply_func2(in.idx, stack[sp - 2], stack[sp - 1]); --sp; break;
            }
        }
        return stack[0];
    }

} // anonymous namespace

// Реализация SystemEvaluator
struct SystemEvaluator::Impl {
    int dim = 0;
    std::vector<std::vector<Instr>> programs; // по одной на уравнение
    // Jacobian, row-major dim*dim, from the same symbolic derivatives the GPU
    // schemes emit. Empty when the system has no derivative (floor/ceil/fmod) --
    // that must not break explicit schemes, which never ask for it.
    std::vector<std::vector<Instr>> jac_programs;
    // Диагональное разложение f_i = coef_i * x_i + rem_i для схем, решающих
    // каждое уравнение относительно своей переменной (CD, SIMP, D). Строится
    // тем же cd_try_extract_linear, что и GPU-ветка, поэтому решение "линейно
    // / нелинейно" у CPU и GPU совпадает по построению. diag_linear[i] == 0 —
    // разложения нет, программы пустые, шаг уходит на итерации.
    std::vector<char> diag_linear;
    // Знак, вынесенный из coef/rem наружу (peel_sign в кодогене): программы
    // считают hs * |rem| и hs * |coef|, а знак попадает в оператор формулы.
    // Побитово это нейтрально — смена знака в IEEE точна.
    std::vector<char> diag_rem_neg, diag_coef_neg;
    std::vector<std::vector<Instr>> diag_coef, diag_rem;
    bool   newton_full = false;
    double newton_tol = 1e-10;
    int    newton_max_iters = 8;
    int max_stack = 16;                        // глубина стека (с запасом)
    mutable std::vector<double> stack;
    // Отдельный стек под комплексный проход: eval и eval_complex не зовутся
    // одновременно (обе — из шага одного интегратора), но держать один буфер
    // на два типа нельзя. Аллоцируется по требованию — обычные схемы за него
    // не платят.
    mutable std::vector<ucmplx> stack_c;
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
    impl_->diag_linear.assign((size_t)impl_->dim, 0);
    impl_->diag_rem_neg.assign((size_t)impl_->dim, 0);
    impl_->diag_coef_neg.assign((size_t)impl_->dim, 0);
    impl_->diag_coef.resize(impl_->dim);
    impl_->diag_rem.resize(impl_->dim);
    int maxdepth = 8;
    for (int i = 0; i < impl_->dim; ++i) {
        Parser p(sys.rhs[i], sys.latex);
        PN ast = p.parse();
        ByteCompiler bc{ impl_->programs[i], var_index, param_index };
        bc.compile(ast);
        // оценка глубины стека: число push не превышает длину программы
        int depth = (int)impl_->programs[i].size() + 4;
        if (depth > maxdepth) maxdepth = depth;

        // Разложение по своей переменной — для диагонально-неявных схем.
        PN coef, rem;
        if (cd_try_extract_linear(ast, sys.vars[i], coef, rem)) {
            // Зеркалим emit_diag_implicit_eq вплоть до формы выражения: сперва
            // тот же вынос знака, затем тот же порядок умножений.
            impl_->diag_rem_neg[(size_t)i]  = (peel_sign(rem,  '+', '-') == '-');
            impl_->diag_coef_neg[(size_t)i] = (peel_sign(coef, '-', '+') == '+');
            ByteCompiler bcc{ impl_->diag_coef[i], var_index, param_index };
            bcc.compile(pn_mul_step(coef));
            ByteCompiler bcr{ impl_->diag_rem[i], var_index, param_index };
            bcr.compile(pn_mul_step(rem));
            impl_->diag_linear[(size_t)i] = 1;
            for (const auto* prog : { &impl_->diag_coef[i], &impl_->diag_rem[i] }) {
                int d = (int)prog->size() + 4;
                if (d > maxdepth) maxdepth = d;
            }
        }
    }
    // Jacobian programs. Built eagerly (N*N tiny programs, negligible at the usual
    // N = 3) but tolerantly: a non-differentiable system must still work with the
    // explicit schemes, so failure leaves jac_programs empty and is reported by
    // has_jacobian() rather than thrown here.
    try {
        std::vector<std::vector<Instr>> jp((size_t)impl_->dim * impl_->dim);
        for (int i = 0; i < impl_->dim; ++i) {
            Parser p(sys.rhs[i], sys.latex);
            PN ast = p.parse();
            jac_check_differentiable(ast);
            for (int j = 0; j < impl_->dim; ++j) {
                ByteCompiler bc{ jp[(size_t)i * impl_->dim + j], var_index, param_index };
                bc.compile(pn_diff(ast, sys.vars[j]));
                int depth = (int)jp[(size_t)i * impl_->dim + j].size() + 4;
                if (depth > maxdepth) maxdepth = depth;
            }
        }
        impl_->jac_programs = std::move(jp);
    }
    catch (const std::exception&) {
        impl_->jac_programs.clear();
    }

    impl_->newton_full      = sys.newton_full;
    impl_->newton_tol       = sys.newton_tol;
    impl_->newton_max_iters = sys.newton_max_iters;

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

bool SystemEvaluator::has_jacobian() const { return !impl_->jac_programs.empty(); }

void SystemEvaluator::eval_jacobian(const double* X, const double* a, double* J) const {
    const int n = impl_->dim;
    if (impl_->jac_programs.empty()) {
        for (int k = 0; k < n * n; ++k) J[k] = 0.0;
        return;
    }
    double* st = impl_->stack.data();
    for (int k = 0; k < n * n; ++k)
        J[k] = run_program(impl_->jac_programs[(size_t)k], X, a, st);
}

void SystemEvaluator::eval_jacobian_complex(const ucmplx* Z, const double* a, ucmplx* J) const {
    const int n = impl_->dim;
    if (impl_->jac_programs.empty()) {
        for (int k = 0; k < n * n; ++k) J[k] = ucmplx(0.0, 0.0);
        return;
    }
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    for (int k = 0; k < n * n; ++k)
        J[k] = run_program(impl_->jac_programs[(size_t)k], Z, a, st);
}

bool   SystemEvaluator::newton_full() const      { return impl_->newton_full; }
double SystemEvaluator::newton_tol() const       { return impl_->newton_tol; }
int    SystemEvaluator::newton_max_iters() const { return impl_->newton_max_iters; }

// Сборка результата повторяет emit_diag_implicit_eq оператор в оператор:
//   X[i] = (X[i] <+|-> hs*|rem|) / (1 <-|+> hs*|coef|).
// Случаи coef == 0 и rem == 0 отдельно не разбираются: на GPU они лишь убирают
// из текста деление на 1 и прибавление нуля, а это побитово тождественные
// операции (деление на 1.0 и сложение с 0.0 в IEEE точны).
bool SystemEvaluator::solve_diag_implicit(int i, double* X, const double* a, double hs) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    double* st = impl_->stack.data();
    const double num = run_program(impl_->diag_rem[(size_t)i],  X, a, st, hs);
    const double den = run_program(impl_->diag_coef[(size_t)i], X, a, st, hs);
    X[i] = (impl_->diag_rem_neg[(size_t)i]  ? X[i] - num : X[i] + num)
         / (impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den);
    return true;
}

bool SystemEvaluator::solve_diag_implicit_complex(int i, ucmplx* Z, const double* a,
                                                  ucmplx hs) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    const ucmplx num = run_program(impl_->diag_rem[(size_t)i],  Z, a, st, hs);
    const ucmplx den = run_program(impl_->diag_coef[(size_t)i], Z, a, st, hs);
    Z[i] = (impl_->diag_rem_neg[(size_t)i]  ? Z[i] - num : Z[i] + num)
         / (impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den);
    return true;
}

void SystemEvaluator::eval_complex(const ucmplx* X, const double* a, ucmplx* deriv) const {
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    for (int i = 0; i < impl_->dim; ++i)
        deriv[i] = run_program(impl_->programs[i], X, a, st);
}

bool scheme_has_complex_core(const std::string& name) {
    return name == "Complex CD" || name == "Complex CD4" || name == "CCD4 (o4s3)"
        || name == "CCD4 (o4s4)" || name == "CCD4 (s-split)_{0-1}"
        || name == "CCD4 (s-split)_{1-0}" || name == "Complex CD_{1-0}"
        || name == "Complex CD4_{1-0}" || name == "CCD4 (o4s3)_{1-0}"
        || name == "CCD4 (o4s4)_{1-0}";
}

std::string codegen_scheme_complex_core(const System& s, Scheme sch) {
    switch (sch) {
    case Scheme::ComplexCD:    return scheme_cd_common(s, CdKind::Cx,    true);
    case Scheme::ComplexCD4:   return scheme_cd_common(s, CdKind::Cx4,   true);
    case Scheme::ComplexCD4S3: return scheme_cd_common(s, CdKind::Cx4s3, true);
    case Scheme::ComplexCD4S4: return scheme_cd_common(s, CdKind::Cx4s4, true);
    case Scheme::ComplexCD4SS01: return scheme_cd_common(s, CdKind::Cx4ss01, true);
    case Scheme::ComplexCD4SS10: return scheme_cd_common(s, CdKind::Cx4ss10, true);
    case Scheme::ComplexCD10:     return scheme_cd_common(s, CdKind::Cx,    true, true);
    case Scheme::ComplexCD4_10:   return scheme_cd_common(s, CdKind::Cx4,   true, true);
    case Scheme::ComplexCD4S3_10: return scheme_cd_common(s, CdKind::Cx4s3, true, true);
    case Scheme::ComplexCD4S4_10: return scheme_cd_common(s, CdKind::Cx4s4, true, true);
    default: break;
    }
    throw std::runtime_error("scheme has no complex core (ExtrZ needs a built-in complex CD scheme)");
}

// RK8(7)13M, уточнённые коэффициенты (см. codegen.hpp): x = c[0] + c[1] + c[2] + c[3].
// Сгенерировано из 70-значных значений; руками не править.
const MultiDoubleCoef kDopri78A[13][12] = {
    { // row 1
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 2
      { 0.05555555555555555, 3.093290075265715e-18, -1.1547725396663453e-34, -1.0237840688138976e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 3
      { 0.020833333333333332, 1.1596595694305568e-18, -3.949890252538956e-35, 1.1525193213706288e-51 },
      { 0.0625, 1.1477009102573743e-20, -4.404924089131691e-37, 3.2917127145845097e-53 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 4
      { 0.03125, 5.49534795572217e-21, -1.4143058530497092e-37, -5.339649673618338e-54 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.09375, 1.6486043867166522e-20, 7.375601105664188e-37, 3.277824392235864e-53 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 5
      { 0.3125, -9.254030713740753e-20, 1.8209627900960226e-36, 7.84288127831444e-53 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -1.171875, 3.9730812450208377e-19, 3.361953464094195e-36, -2.08180363371821e-52 },
      { 1.171875, -2.8557040730918284e-19, 6.707779166838405e-36, -5.497205027421227e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 6
      { 0.0375, 1.4225247854593709e-18, -6.165335569208968e-35, 4.077445886605457e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.1875, -4.4558093993658926e-20, -2.8966368003730405e-36, -1.4450187914661259e-53 },
      { 0.15, 5.632889430634718e-18, -6.645840907664647e-35, 3.588162896942603e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 7
      { 0.04791013711111111, -7.259747678910195e-19, 3.121431990604312e-35, 1.789740628240766e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.11224871277777777, 4.570684995097771e-18, 1.7064340048750203e-36, 3.1923929805042253e-53 },
      { -0.02550567377777778, 1.4251552616654307e-18, -4.496688518765216e-36, -1.2146812140527791e-52 },
      { 0.012846823888888888, 7.810234878340747e-19, 2.2390341576581862e-35, 6.544222636989059e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 8
      { 0.01691798978729228, -2.0540039544109118e-20, -3.727035489163214e-37, 3.8964829998468045e-54 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.38784827848604314, 2.733755715791944e-17, 1.4675591184534425e-33, -5.945917286616092e-50 },
      { 0.03597736985150033, -2.826141523945096e-18, -1.5405419792955276e-35, -1.0831426136454798e-52 },
      { 0.19697021421566607, -1.2816423652749336e-17, -7.02488878143139e-35, 1.874751876772036e-51 },
      { -0.17271385234050185, 1.2508053007794974e-17, 3.1953680005449514e-34, 2.0516119268381452e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 9
      { 0.0690957533591923, 3.95539563801157e-18, -7.60017896720407e-35, 1.1280325137149748e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.6342479767288541, -1.9182849195068947e-17, -8.841268402445377e-34, -2.557766257365555e-50 },
      { -0.16119757522460407, -1.3490375049608381e-17, 6.404921539646194e-34, 7.877501818115059e-51 },
      { 0.13865030945882525, 3.4138747885767947e-19, 8.705119596063811e-39, 4.195159148916378e-55 },
      { 0.9409286140357562, 4.4244914236133045e-17, 4.93059768016733e-34, -2.2445639624434776e-50 },
      { 0.21163632648194397, 1.1655192226572519e-17, 5.473920128706212e-34, 1.333485207038265e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 10
      { 0.1835569968390454, -3.469167714708232e-18, -6.797552562728481e-35, 3.886476806985195e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -2.4687680843155926, 1.2916691550344442e-16, -8.702027542677447e-33, -2.0583588139265426e-49 },
      { -0.29128688781630047, 9.468516750986405e-18, -1.6536736150169545e-34, -6.93176645379474e-51 },
      { -0.026473020233117376, 2.839706335212075e-19, 1.8546990030578558e-35, -9.336051665621152e-52 },
      { 2.8478387641928005, -1.5885094707574538e-17, 1.167408071483644e-33, -7.602835084357443e-50 },
      { 0.2813873314698498, 4.319575416077761e-18, 1.3752417580078162e-34, 8.090733073219804e-51 },
      { 0.12374489986331466, -3.7559814193933234e-18, -1.1466145616872957e-36, -8.016150451327041e-53 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 11
      { -1.2154248173958881, 9.060225652364193e-17, -3.9181659732551024e-33, 2.3341868673266386e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 16.672608665945774, -1.1094896268896866e-15, -2.817280137065733e-32, -2.5056604153028628e-48 },
      { 0.915741828416818, 1.0541856202026875e-17, -6.715510784332172e-34, -3.899842211334372e-51 },
      { -6.056605804357471, -2.8957576613790487e-16, -7.617897739487842e-34, 1.9279008971304112e-50 },
      { -16.00357359415618, 1.5559490448823435e-15, 6.372484001938011e-33, 5.3373470501071936e-49 },
      { 14.849303086297663, -6.357153739164178e-16, 3.387190718545068e-32, -2.520920498532447e-48 },
      { -13.371575735289849, -7.338567119819293e-16, -3.161251839075618e-34, -1.9399031749479233e-51 },
      { 5.134182648179638, -6.119053789436656e-17, 5.043897697840891e-33, 2.056546293882501e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 12
      { 0.25886091643826425, 2.739166755748268e-17, -3.348521212692587e-34, -1.1507795625259842e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -4.774485785489205, -4.4194601010926526e-16, 1.0306828690387583e-32, -3.1799895794904816e-49 },
      { -0.4350930137770325, 1.3621859043699128e-17, 2.0486624908685586e-34, 1.4535647966514852e-50 },
      { -3.0494833320722416, 1.4162969938663227e-16, -1.1586619973724037e-32, 3.666601675115431e-49 },
      { 5.5779200399360995, -4.2236307920525115e-16, 1.7203934769168018e-32, 7.588689577335947e-49 },
      { 6.15583158986104, 2.610284349193258e-17, 9.215792480937486e-34, -2.8491166473898953e-50 },
      { -5.062104586736939, 3.870297949115654e-16, 4.449195490101153e-33, -1.3438534921622063e-49 },
      { 2.193926173180679, 1.074963043753374e-16, 1.948021075934798e-33, 1.446596938541905e-49 },
      { 0.13462799865933495, -5.642292882422152e-18, -6.713529738070735e-35, 3.389053151037805e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 13
      { 0.8224275996265075, 9.124149362657711e-18, 7.904957679810968e-35, -2.4780478669227778e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -11.658673257277664, 8.110019339956859e-17, -2.2184667201377284e-33, -2.608922261652751e-50 },
      { -0.7576221166909362, -4.140072001753216e-17, 1.6264874178091684e-33, 7.107391589207899e-50 },
      { 0.7139735881595816, -3.5635339181609017e-17, -1.9788528073755075e-33, 4.220119599163384e-50 },
      { 12.075774986890057, -7.159632054588964e-17, -2.103822950489072e-34, -2.1361447702687554e-50 },
      { -2.127659113920403, 2.0638499378742928e-16, -1.3772597620887114e-33, -3.135820897519478e-51 },
      { 1.9901662070489554, -7.537862754363986e-19, 3.4451122019900287e-35, 9.788608365667116e-52 },
      { -0.23428647154404028, -1.2249151372755581e-17, 4.90710018978847e-34, -4.193217626835576e-51 },
      { 0.17589857770794226, 3.681026522615018e-18, 8.918111840741425e-35, 3.389748116240012e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
};
const MultiDoubleCoef kDopri78B[2][13] = {
    { // b, 8th order
      { 0.041747491141530244, 2.69031714934492e-18, 1.2670094674040199e-34, 5.906307657088468e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.05545232861123931, 2.90573943993639e-18, -5.266359783563116e-35, -2.6440657700898857e-51 },
      { 0.2393128072011801, 8.162102731025973e-18, 3.2875430670994776e-34, -1.8636590513910183e-50 },
      { 0.703510669403443, 4.256752126511246e-17, -1.0523525810623621e-33, -3.6525890916753464e-50 },
      { -0.7597596138144609, -3.9709909607487243e-17, -2.386360797892327e-33, -1.3209742825705595e-49 },
      { 0.6605630309222863, 4.211110342468121e-17, 1.6292361215956356e-33, -1.0109105476848643e-49 },
      { 0.15818748251012332, 1.2772213847110806e-17, 1.5600770150795557e-34, -4.9176612144785693e-51 },
      { -0.2381095387528628, 4.896006341639819e-18, 2.1593227161375922e-35, -1.1528683429180865e-51 },
      { 0.25, -6.726164838481741e-20, 1.304333528140994e-36, -5.053479069967406e-54 }
    },
    { // b, 7th order
      { 0.029553213676353503, -1.4971536072406993e-18, 6.58751221158917e-35, -2.3871729553979215e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.8286062764877969, 3.5341669811058704e-17, -1.119866351476355e-33, -3.6700667553332564e-50 },
      { 0.3112409000511183, -1.017338805788491e-17, -2.251263048940433e-34, -4.03802187815624e-52 },
      { 2.467345190599887, -1.465434075784604e-16, 2.8743803814327853e-33, -1.1874361585430109e-50 },
      { -2.5469416518419083, -1.9557984225562319e-16, 4.5464712634926405e-33, 2.1458732205172338e-51 },
      { 1.4435485836767752, -1.404909149434919e-17, 9.56267944325026e-34, -8.454842072541161e-51 },
      { 0.07941559588112729, 1.3754984390730884e-18, 7.656505392426402e-35, -2.7194651917628785e-51 },
      { 0.044444444444444446, -1.9411926441203856e-18, 1.439666797512871e-34, -3.7519147164496724e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
};

// DOPRI78 (legacy): рациональные коэффициенты статьи Prince & Dormand (1981)
// как есть, суммой 4 double (сгенерировано; руками не править).
const MultiDoubleCoef kDopri78LegacyA[13][12] = {
    { // row 1
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 2
      { 0.05555555555555555, 3.0839528461809902e-18, 1.7119377283442096e-34, 9.50316341366114e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 3
      { 0.020833333333333332, 1.1564823173178713e-18, 6.419766481290786e-35, 3.5636862801229273e-51 },
      { 0.0625, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 4
      { 0.03125, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.09375, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 5
      { 0.3125, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -1.171875, 0.0, 0.0, 0.0 },
      { 1.171875, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 6
      { 0.0375, 1.3877787807814458e-18, -7.703719777548944e-35, 4.276423536147513e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.1875, 0.0, 0.0, 0.0 },
      { 0.15, 5.551115123125783e-18, -3.0814879110195775e-34, 1.7105694144590053e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 7
      { 0.04791013711111111, -1.4526756652194954e-18, 4.3381891074120376e-35, -2.015804424501742e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.11224871277777777, 3.744872947541009e-18, -1.0080550407220715e-34, -9.7149306512512e-52 },
      { -0.02550567377777778, 1.443098874965977e-18, -2.839638496440861e-35, 1.2381508917501458e-51 },
      { 0.012846823888888888, 7.761529690558038e-19, -3.2964251166886494e-36, -1.6085700609074975e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 8
      { 0.01691798978729228, -6.368397251761342e-19, 2.4357902274338865e-35, -5.66939475143029e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.3878482784860432, -2.6885265415831007e-17, -1.5355505417054944e-33, -5.498438070979638e-50 },
      { 0.03597736985150033, -2.4970669000735346e-18, 1.94895358171659e-35, -5.6253689112715186e-52 },
      { 0.19697021421566607, -1.3131178979550301e-17, 6.742526492605492e-34, 1.8259858136689701e-50 },
      { -0.17271385234050185, 1.205371316696555e-17, -2.3917418433697972e-34, -4.594926922398343e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 9
      { 0.0690957533591923, 4.339884607225479e-18, -2.7084324871328984e-34, -9.776406066899341e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.6342479767288541, -1.5999891229660937e-17, 8.422617006140311e-34, -6.07234692902508e-50 },
      { -0.16119757522460407, -1.2715151291274541e-17, 4.85465875946948e-34, -2.2349921146113426e-50 },
      { 0.13865030945882525, -6.436577938232034e-19, 2.444236523466998e-35, 1.1537054287203372e-51 },
      { 0.9409286140357562, 4.042088530185426e-17, 3.0245749894551174e-33, -1.3255731094231783e-49 },
      { 0.21163632648194397, 1.0949420370553082e-17, -6.655351832524269e-34, -9.5203414909633e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 10
      { 0.1835569968390454, -3.523409866966572e-18, 3.694251671251376e-34, 1.774788884449841e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -2.4687680843155926, 1.3217967229249085e-16, 3.0971283409380165e-33, 7.014621606130137e-50 },
      { -0.29128688781630047, 1.0482551751731106e-17, -6.961211663043419e-34, 2.6537436149930247e-50 },
      { -0.026473020233117376, 2.1605579436016833e-19, -4.222838510956721e-36, 2.997600533679052e-52 },
      { 2.8478387641928005, -2.1488428969104426e-17, -9.34090861305883e-34, -5.38142807830714e-50 },
      { 0.2813873314698498, 3.3102934900943334e-18, -1.0671519788801398e-34, -1.930698594614613e-51 },
      { 0.12374489986331466, -4.229816804811852e-18, 2.434549826224879e-34, -1.6961483182662875e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 11
      { -1.2154248173958881, 8.88099435870682e-17, -1.5021781724544412e-33, 3.018475938162135e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 16.672608665945774, -1.1104383600928815e-15, 2.994638260841533e-32, -1.6799335965366346e-48 },
      { 0.915741828416818, 8.970509643411096e-18, 3.8506437729383177e-34, 1.0003793274530859e-50 },
      { -6.056605804357471, -2.9127895008046388e-16, -2.149597400438059e-32, 8.955504976033509e-49 },
      { -16.00357359415618, 1.5542545815775273e-15, -9.75712082114111e-32, -7.358891771789301e-49 },
      { 14.849303086297663, -6.384404166415637e-16, -4.5730910529414385e-32, -2.3419647001024056e-48 },
      { -13.371575735289849, -7.352619849870231e-16, -1.1037820706946624e-32, -3.0882888977925673e-49 },
      { 5.134182648179638, -6.205101409963547e-17, 1.633864331906019e-33, 1.0977717568283333e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 12
      { 0.25886091643826425, 2.7350750012505298e-17, 1.215434577262574e-34, -6.509497285934003e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -4.774485785489205, -4.4338582381881673e-16, -4.757578289796903e-33, 2.6177563356835204e-49 },
      { -0.4350930137770325, 1.3639107949486849e-17, 5.789433839745416e-34, -3.51172590656963e-50 },
      { -3.0494833320722416, 1.425074887461722e-16, -9.258508856826952e-33, -5.583788087306652e-49 },
      { 5.5779200399360995, -4.238620854736817e-16, -1.821385015123591e-32, -8.979540058249146e-49 },
      { 6.15583158986104, 2.847407335763832e-17, 2.7561054222824837e-33, 3.0305760897956624e-50 },
      { -5.062104586736939, 3.8925787100792194e-16, -4.2484458437723856e-33, -1.9248662385821055e-49 },
      { 2.193926173180679, 1.0724478602788584e-16, -2.5847552501537252e-33, -4.609996126150793e-50 },
      { 0.13462799865933495, -5.039794795402269e-18, 3.550306599504452e-35, 1.5233674682291436e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 13
      { 0.8224275996265075, 9.857639838863826e-18, -6.767453150994087e-34, 8.633663719943064e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -11.658673257277664, 8.334298163044945e-17, 5.241507322802345e-33, -3.3795951502960358e-49 },
      { -0.7576221166909362, -4.0626828536037304e-17, -1.1329139394269249e-34, 7.346525104702948e-51 },
      { 0.7139735881595816, -3.574462166732446e-17, -1.5213318981416248e-33, -4.425655509470321e-50 },
      { 12.075774986890057, -6.928407964604289e-17, -4.1464622411062394e-33, 2.935028686976042e-49 },
      { -2.127659113920403, 2.0473700870867944e-16, 5.68640963819146e-33, 4.338973590926975e-50 },
      { 1.9901662070489554, -2.2189013401207363e-18, -6.314399949059115e-35, 2.1195748992020166e-52 },
      { -0.23428647154404028, -1.108305257591422e-17, -5.649965828191292e-34, 7.049018009643217e-51 },
      { 0.17589857770794226, 4.040345125515876e-18, -2.9167288527880757e-34, 2.9153636471623585e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
};
const MultiDoubleCoef kDopri78LegacyB[2][13] = {
    { // b, 8th order
      { 0.041747491141530244, 2.5003377343256745e-18, 1.16407987548751e-35, 6.009804155247104e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.05545232861123931, 1.8848658149163404e-18, 5.565485578607775e-35, 3.9320554913302793e-51 },
      { 0.2393128072011801, 7.166985282873301e-18, 6.405946585392943e-34, 2.0430746972466673e-50 },
      { 0.703510669403443, 4.1866092586930134e-17, 2.590545295744438e-34, 8.129096637065185e-51 },
      { -0.7597596138144609, -3.9890577688618486e-17, -2.6957037062523015e-33, -6.711016926335441e-50 },
      { 0.6605630309222863, 4.1333074232293545e-17, 2.958848910597238e-33, 1.141115475830444e-49 },
      { 0.15818748251012332, 1.21756483733966e-17, 6.535093822724954e-34, -3.5986118847169934e-50 },
      { -0.2381095387528628, 5.606091933880033e-18, 1.2259724943251947e-34, 5.315401390514777e-51 },
      { 0.25, 0.0, 0.0, 0.0 }
    },
    { // b, 7th order
      { 0.0295532136763535, -1.6826969615593314e-18, -6.016629474430411e-35, -4.687144773281858e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.828606276487797, 1.726634266453714e-17, 1.1770344648705815e-33, 3.2811278142500814e-50 },
      { 0.3112409000511183, 8.172424645811795e-18, 5.866102878703906e-35, -5.263634954777557e-51 },
      { 2.467345190599887, 9.399457860675346e-17, -4.733760868975007e-34, -3.598093534437834e-50 },
      { -2.546941651841909, 6.252291117664108e-17, -1.2047691986372572e-33, -8.014147352453355e-51 },
      { 1.4435485836767752, 4.940219249902469e-17, 2.652447677982459e-33, -1.9071076767243827e-51 },
      { 0.07941559588112729, -1.6944294213357039e-18, 8.234117767009426e-36, -5.204136108664492e-52 },
      { 0.044444444444444446, -1.6961740653995447e-18, -5.563797617118681e-35, -5.226739877513627e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
};

// Определена ниже, рядом с весами экстраполяции, которыми пользуется.
namespace { std::string scheme_gbs(const System& s, int K); }

int gbs_stage_count(Scheme sch) {
    switch (sch) {
    case Scheme::GBS:         return 1;
    case Scheme::GBS24:       return 2;
    case Scheme::GBS246:      return 3;
    case Scheme::GBS2468:     return 4;
    case Scheme::GBS246810:   return 5;
    case Scheme::GBS24681012: return 6;
    default:                  return 0;
    }
}

std::vector<int> gbs_substeps(int K) {
    std::vector<int> n;
    for (int k = 1; k <= K; ++k) n.push_back(2 * k);
    return n;
}

// Публичная функция
std::string codegen_scheme(const System& s, Scheme sch) {
    if (const int K = gbs_stage_count(sch)) return scheme_gbs(s, K);
    switch (sch) {
    case Scheme::Euler:            return scheme_euler(s);
    case Scheme::EulerCromer:      return scheme_euler_cromer(s);
    case Scheme::ExplicitMidpoint: return scheme_midpoint(s);
    case Scheme::RK4:              return scheme_rk4(s);
    case Scheme::DOPRI78:          return scheme_dopri78(s);
    case Scheme::DOPRI78Legacy:    return scheme_dopri78(s, true);
    case Scheme::CD:               return scheme_cd(s);
    case Scheme::ComplexCD:        return scheme_complex_cd(s);
    case Scheme::ComplexCD4:       return scheme_complex_cd4(s);
    case Scheme::ComplexCD4S3:     return scheme_complex_cd4_s3(s);
    case Scheme::ComplexCD4S4:     return scheme_complex_cd4_s4(s);
    case Scheme::ComplexCD4SS01:   return scheme_complex_cd4_ss01(s);
    case Scheme::ComplexCD4SS10:   return scheme_complex_cd4_ss10(s);
    case Scheme::CD10:             return scheme_cd_common(s, CdKind::Real,  false, true);
    case Scheme::ComplexCD10:      return scheme_cd_common(s, CdKind::Cx,    false, true);
    case Scheme::ComplexCD4_10:    return scheme_cd_common(s, CdKind::Cx4,   false, true);
    case Scheme::ComplexCD4S3_10:  return scheme_cd_common(s, CdKind::Cx4s3, false, true);
    case Scheme::ComplexCD4S4_10:  return scheme_cd_common(s, CdKind::Cx4s4, false, true);
    case Scheme::ImplicitEuler:    return scheme_implicit_euler(s);
    case Scheme::ImplicitMidpoint: return scheme_implicit_midpoint(s);
    case Scheme::SEMP:             return scheme_semp(s);
    case Scheme::SIMP:             return scheme_simp(s);
    case Scheme::D:                return scheme_d(s);
    case Scheme::ComplexIEuler:    return scheme_complex_ieuler(s);
    case Scheme::Map:              return scheme_map(s);
    default:                       break;   // ГБШ разобран выше
    }
    throw std::runtime_error("unknown scheme");
}

std::string codegen_coupling(const System& s, const std::vector<std::string>& exprs,
                             const std::string& self_state, const std::string& nbr_state,
                             const std::string& self_par, const std::string& nbr_par,
                             const std::string& weight, const std::string& dst,
                             const std::string& indent) {
    if (exprs.size() != s.vars.size())
        throw std::runtime_error("coupling: expression count != number of variables");
    NameMap nm;
    // Вес кладём ПЕРВЫМ: имя из системы должно его перекрывать, а не наоборот.
    nm.m["K"] = weight;
    nm.m["w"] = weight;
    for (const auto& c : math_constants()) nm.m[c] = c;
    for (size_t i = 0; i < s.vars.size(); ++i) {
        const std::string idx = "[" + std::to_string(i) + "]";
        nm.m[s.vars[i]]        = self_state + idx;
        nm.m[s.vars[i] + "_i"] = self_state + idx;
        nm.m[s.vars[i] + "_j"] = nbr_state + idx;
    }
    for (size_t j = 0; j < s.params.size(); ++j) {
        const std::string idx = "[" + std::to_string(1 + (int)j) + "]";
        nm.m[s.params[j]]        = self_par + idx;
        nm.m[s.params[j] + "_i"] = self_par + idx;
        nm.m[s.params[j] + "_j"] = nbr_par + idx;
    }
    std::ostringstream o;
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (exprs[i].empty()) continue;
        Parser p(exprs[i], false);
        PN ast = p.parse();
        o << indent << dst << "[" << i << "] += (";
        emit(ast, nm, o);
        o << ");\n";
    }
    return o.str();
}

Scheme scheme_from_name(const std::string& name) {
    if (name == "Euler-Cromer")      return Scheme::EulerCromer;
    if (name == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (name == "RK4")               return Scheme::RK4;
    if (name == "DOPRI78")           return Scheme::DOPRI78;
    if (name == "DOPRI78 (legacy)")  return Scheme::DOPRI78Legacy;
    if (name == "CD")                return Scheme::CD;
    if (name == "Complex CD")        return Scheme::ComplexCD;
    if (name == "Complex CD4")       return Scheme::ComplexCD4;
    if (name == "CCD4 (o4s3)")       return Scheme::ComplexCD4S3;
    if (name == "CCD4 (o4s4)")       return Scheme::ComplexCD4S4;
    if (name == "CCD4 (s-split)_{0-1}") return Scheme::ComplexCD4SS01;
    if (name == "CCD4 (s-split)_{1-0}") return Scheme::ComplexCD4SS10;
    if (name == "CD_{1-0}")          return Scheme::CD10;
    if (name == "Complex CD_{1-0}")  return Scheme::ComplexCD10;
    if (name == "Complex CD4_{1-0}") return Scheme::ComplexCD4_10;
    if (name == "CCD4 (o4s3)_{1-0}") return Scheme::ComplexCD4S3_10;
    if (name == "CCD4 (o4s4)_{1-0}") return Scheme::ComplexCD4S4_10;
    if (name == "Implicit Euler")    return Scheme::ImplicitEuler;
    if (name == "Implicit Midpoint") return Scheme::ImplicitMidpoint;
    if (name == "SEMP")              return Scheme::SEMP;
    if (name == "SIMP")              return Scheme::SIMP;
    if (name == "D")                 return Scheme::D;
    if (name == "Complex Implicit Euler") return Scheme::ComplexIEuler;
    if (name == "GBS (n=2)")         return Scheme::GBS;
    if (name == "GBS 2-4")           return Scheme::GBS24;
    if (name == "GBS 2-4-6")         return Scheme::GBS246;
    if (name == "GBS 2-4-6-8")       return Scheme::GBS2468;
    if (name == "GBS 2-4-6-8-10")    return Scheme::GBS246810;
    if (name == "GBS 2-4-6-8-10-12") return Scheme::GBS24681012;
    if (name == "Map")               return Scheme::Map;
    return Scheme::Euler;
}

// --- Паспорт встроенных схем -----------------------------------------------
// Порядки — те, что замерены и записаны в подсказках чекбоксов (см. gui.cpp).
// Напоминание про a[0] = 1/2: у CD, Complex CD, Complex CD4, CCD4 (o4s3),
// CCD4 (o4s4) (и их вариантов _{1-0}), SEMP и SIMP
// заявленный порядок достигается только при этом значении, иначе все они
// падают до первого. Здесь стоит паспортный (то есть при a[0] = 1/2) —
// предупредить пользователя обязан UI.
namespace {
// Порядок строк = порядок в таблице схем UI (kBuiltinSchemes в gui.cpp): по
// нему sort_wrapper_schemes ставит опорники обёрток, и список обёрток
// читается теми же блоками, что и список встроенных.
struct SchemeRow { const char* name; int order; bool symmetric; };
const SchemeRow kSchemeRows[] = {
    { "Euler",                  1, false },
    { "Euler-Cromer",           1, false },
    { "D",                      1, false },
    { "Implicit Euler",         1, false },
    { "Explicit Midpoint",      2, false },
    { "Implicit Midpoint",      2, true  },  // Phi* ∘ Phi, самосопряжённая
    { "CD",                     2, true  },  // самосопряжённая при a[0] = 1/2
    // {1-0}: неявная половина первой; порядок и симметрия как у исходных.
    { "CD_{1-0}",               2, true  },
    { "Complex CD",             2, false },
    { "Complex CD_{1-0}",       2, false },
    { "Complex Implicit Euler", 2, false },
    { "SEMP",                   2, false },  // предиктор-корректор, не Phi* ∘ Phi
    { "SIMP",                   2, false },
    { "RK4",                    4, false },
    { "Complex CD4",            4, false },
    { "Complex CD4_{1-0}",      4, false },
    { "CCD4 (o4s3)",            4, true  },  // палиндром (a, 1-2a, a)
    { "CCD4 (o4s3)_{1-0}",      4, true  },
    { "CCD4 (o4s4)",            4, true  },  // палиндром (g/2, gc/2, gc/2, g/2)
    { "CCD4 (o4s4)_{1-0}",      4, true  },
    { "CCD4 (s-split)_{0-1}",   4, false },  // самосопряжена до Re, после Re дефект O(h^6)
    { "CCD4 (s-split)_{1-0}",   4, false },  // то же над сопряжённым базовым методом
    { "DOPRI78",                8, false },
    { "DOPRI78 (legacy)",       8, false },  // дроби статьи без уточнения
    // ГБШ: опорная GBS (n=2) не самосопряжена как одношаговый метод
    // (дефект O(h^6)), поэтому Extr над ней на трёх стадиях даёт 5, а не 6.
    // Экстраполяторы — не база для дальнейших обёрток.
    { "GBS (n=2)",              2, false },
    { "GBS 2-4",                4, false },
    { "GBS 2-4-6",              6, false },
    { "GBS 2-4-6-8",            8, false },
    { "GBS 2-4-6-8-10",        10, false },
    { "GBS 2-4-6-8-10-12",     12, false },
};

// Номер строки паспорта, -1 — не встроенная схема.
int builtin_scheme_rank(const std::string& name) {
    const int n = (int)(sizeof(kSchemeRows) / sizeof(kSchemeRows[0]));
    for (int i = 0; i < n; ++i)
        if (name == kSchemeRows[i].name) return i;
    return -1;
}
} // namespace

bool builtin_scheme_traits(const std::string& name, int* order, bool* symmetric) {
    const int i = builtin_scheme_rank(name);
    if (i < 0) return false;
    if (order)     *order     = kSchemeRows[i].order;
    if (symmetric) *symmetric = kSchemeRows[i].symmetric;
    return true;
}

// --- Имя обёртки: необязательная метка --------------------------------------

namespace {

    std::string wrap_trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    // Голова имени — то, что стоит до ПОСЛЕДНЕЙ '|', то есть до коэффициентов.
    // Метка, если она есть, идёт первой: "s17o8|Short CD". Режем по ПЕРВОЙ
    // черте, тогда как база берётся по последней, — старое имя без метки
    // ("Short CD") головы с чертой не имеет и разбирается ровно как раньше.
    bool wrap_split_head(const std::string& head, std::string* label,
                         std::string* base, std::string* err) {
        const size_t bar = head.find('|');
        if (bar == std::string::npos) { label->clear(); *base = wrap_trim(head); return true; }
        *label = wrap_trim(head.substr(0, bar));
        *base  = wrap_trim(head.substr(bar + 1));
        if (label->empty()) { if (err) *err = "empty scheme label"; return false; }
        if (!wrapper_label_ok(*label)) {
            if (err) *err = "scheme label must not contain '|', ',', '(' or ')'";
            return false;
        }
        return true;
    }

} // namespace

bool wrapper_label_ok(const std::string& label) {
    if (label.empty()) return false;
    if (wrap_trim(label) != label) return false;
    for (char c : label)
        if (c == '|' || c == ',' || c == '(' || c == ')') return false;
    return true;
}

std::string wrapper_sanitize_label(const std::string& label) {
    std::string out;
    for (char c : label)
        if (c != '|' && c != ',' && c != '(' && c != ')') out += c;
    return wrap_trim(out);
}

// --- Экстраполяция Ричардсона ----------------------------------------------

static const char* const kExtrPrefix  = "Extr(";
static const char* const kExtrZPrefix = "ExtrZ(";   // Re только на выходе

static bool starts_with(const std::string& s, const char* pref) {
    const size_t n = std::strlen(pref);
    return s.size() >= n && s.compare(0, n, pref) == 0;
}

std::string make_extrapolation_name(const std::string& base, const std::vector<int>& n,
                                    const std::string& label, bool re_at_output) {
    std::string s = re_at_output ? kExtrZPrefix : kExtrPrefix;
    if (!label.empty()) s += label + "|";
    s += base + "|";
    for (size_t k = 0; k < n.size(); ++k) {
        if (k) s += ",";
        s += std::to_string(n[k]);
    }
    return s + ")";
}

// "12" -> 12. Отказывает на пустой строке, нецифрах и выходе за диапазон,
// чтобы кривое имя из старого JSON не превратилось молча в рабочую схему.
static bool extr_parse_substeps(const std::string& s, int* out) {
    if (s.empty() || s.size() > 9) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    const long v = std::strtol(s.c_str(), nullptr, 10);
    if (v < 1 || v > kExtrMaxSubsteps) return false;
    *out = (int)v;
    return true;
}

bool parse_extrapolation_name(const std::string& name, ExtrapolationSpec* out,
                              std::string* err) {
    auto fail = [&](const char* msg) { if (err) *err = msg; return false; };
    auto fail_s = [&](const std::string& msg) { if (err) *err = msg; return false; };

    const bool re_out = starts_with(name, kExtrZPrefix);
    const std::string pref = re_out ? kExtrZPrefix : kExtrPrefix;
    if (name.size() <= pref.size() || name.compare(0, pref.size(), pref) != 0)
        return fail("not an extrapolated scheme name");
    if (name.back() != ')')
        return fail("missing closing ')'");

    const std::string inner = name.substr(pref.size(), name.size() - pref.size() - 1);
    // rfind, а не find: имя базы по правилам валидации '|' не содержит, но если
    // в старом JSON такое всё же лежит — резать надо по последней черте, иначе
    // список n разберётся как часть имени и схема тихо станет другой.
    const size_t bar = inner.rfind('|');
    if (bar == std::string::npos) return fail("missing '|' between base and substep list");

    ExtrapolationSpec spec;
    spec.re_at_output = re_out;
    // Голова — "<база>" либо "<метка>|<база>": метку добавили позже, и имена
    // без неё обязаны разбираться как прежде.
    if (!wrap_split_head(inner.substr(0, bar), &spec.label, &spec.base, err)) return false;
    if (spec.base.empty()) return fail("empty base scheme name");
    // Вложенность запрещена: порядок обёртки над обёрткой считается не как
    // p + K - 1, и заодно это отрезает бесконечную рекурсию в резолвере.
    if (starts_with(spec.base, kExtrPrefix) || starts_with(spec.base, kExtrZPrefix))
        return fail("base scheme cannot itself be extrapolated");

    const std::string tail = inner.substr(bar + 1);
    size_t pos = 0;
    while (true) {
        const size_t comma = tail.find(',', pos);
        const std::string tok = tail.substr(pos, comma == std::string::npos
                                                ? std::string::npos : comma - pos);
        int v = 0;
        if (!extr_parse_substeps(tok, &v))
            return fail_s("substep counts must be integers in 1.."
                          + std::to_string(kExtrMaxSubsteps));
        spec.n.push_back(v);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    const int K = (int)spec.n.size();
    // Сообщения печатаются из констант, а не из букв: потолок стадий уже
    // поднимали, и расхождение текста с проверкой врёт пользователю.
    if (K < kExtrMinStages)
        return fail_s("at least " + std::to_string(kExtrMinStages) + " stages are needed");
    if (K > kExtrMaxStages)
        return fail_s("at most " + std::to_string(kExtrMaxStages) + " stages are supported");
    for (int k = 1; k < K; ++k)
        if (spec.n[k] <= spec.n[k - 1]) return fail("substep counts must strictly increase");

    if (out) *out = spec;
    return true;
}

int extrapolation_order(int stages, int p, bool symmetric) {
    if (stages < 1) return p;
    return symmetric ? p + 2 * (stages - 1) : p + stages - 1;
}

// Точная рациональная арифметика для весов экстраполяции.
//
// Веса — отношения небольших целых, но напечатанные десятичным литералом они
// несут ошибку ~1e-16 ОТ СЕБЯ: литерал читается как double, в какой бы
// точности ни считала схема. Экстраполяция складывает почти равные величины с
// коэффициентами порядка единиц (у Extr(CD|1,2,3,4) это 16/45, -729/280,
// 1024/315), так что ошибка коэффициентов не сокращается, а садится прямо в
// результат макрошага. На dd-ветке вкладки Order это держало полку на уровне
// double и сводило расширенную точность к нулю: p валился с 8 до -1 там, где
// с точными дробями он остаётся 8.01 при E1 = 2e-18.
//
// Поэтому печатаем дробь, а не число: деление выполняется уже в numb.
namespace {

struct ExRat { long long n, d; };

long long ex_gcd(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { const long long t = a % b; a = b; b = t; }
    return a ? a : 1;
}

// false — переполнение; вызывающий откатывается на печать литералом.
bool ex_mul(long long a, long long b, long long& r) {
    if (a == 0 || b == 0) { r = 0; return true; }
    const long long q = a * b;
    if (q / b != a) return false;
    r = q;
    return true;
}

bool ex_norm(ExRat& x) {
    if (x.d == 0) return false;
    if (x.d < 0) { x.d = -x.d; x.n = -x.n; }
    const long long g = ex_gcd(x.n, x.d);
    x.n /= g;
    x.d /= g;
    return true;
}

bool ex_mul_rat(ExRat a, ExRat b, ExRat& out) {
    // Сокращаем крест-накрест ДО умножения: так переполнение наступает много
    // позже, а знаменатели тут растут как произведение разностей n^q.
    const long long g1 = ex_gcd(a.n, b.d), g2 = ex_gcd(b.n, a.d);
    a.n /= g1; b.d /= g1;
    b.n /= g2; a.d /= g2;
    if (!ex_mul(a.n, b.n, out.n)) return false;
    if (!ex_mul(a.d, b.d, out.d)) return false;
    return ex_norm(out);
}

bool ex_add_rat(ExRat a, ExRat b, ExRat& out) {
    const long long g = ex_gcd(a.d, b.d);
    long long den, t1, t2;
    if (!ex_mul(a.d, b.d / g, den)) return false;
    if (!ex_mul(a.n, b.d / g, t1)) return false;
    if (!ex_mul(b.n, a.d / g, t2)) return false;
    if ((t2 > 0 && t1 > LLONG_MAX - t2) || (t2 < 0 && t1 < LLONG_MIN - t2)) return false;
    out.n = t1 + t2;
    out.d = den;
    return ex_norm(out);
}

bool ex_pow(long long base, int e, long long& out) {
    out = 1;
    for (int i = 0; i < e; ++i)
        if (!ex_mul(out, base, out)) return false;
    return true;
}

// Те же alpha, что в extrapolation_weights, но без единого округления.
// Подстановка u_k = 1/n_k^q в alpha_k ~ n_k^p * prod 1/(u_k - u_m) даёт
// gamma_k = n_k^(p + q(K-2)) / prod_{m!=k} (n_m^q - n_k^q); общий множитель
// сокращается нормировкой суммы в единицу.
bool extrapolation_weights_rational(const std::vector<int>& n, int p, bool symmetric,
                                    std::vector<ExRat>& out) {
    const int K = (int)n.size();
    if (K < 2) return false;
    const int q = symmetric ? 2 : 1;
    const int e = p + q * (K - 2);
    if (e < 0) return false;

    std::vector<ExRat> g((size_t)K);
    ExRat sum{ 0, 1 };
    for (int k = 0; k < K; ++k) {
        long long num = 0, den = 1, pk = 1;
        if (!ex_pow((long long)n[k], e, num)) return false;
        if (!ex_pow((long long)n[k], q, pk))  return false;
        for (int m = 0; m < K; ++m) {
            if (m == k) continue;
            long long pm = 1;
            if (!ex_pow((long long)n[m], q, pm)) return false;
            if (!ex_mul(den, pm - pk, den))      return false;
        }
        g[(size_t)k] = ExRat{ num, den };
        if (!ex_norm(g[(size_t)k]))               return false;
        if (!ex_add_rat(sum, g[(size_t)k], sum))  return false;
    }
    if (sum.n == 0) return false;

    out.assign((size_t)K, ExRat{ 0, 1 });
    for (int k = 0; k < K; ++k) {
        const ExRat inv{ sum.d, sum.n };
        if (!ex_mul_rat(g[(size_t)k], inv, out[(size_t)k])) return false;
        // Числитель и знаменатель обязаны быть точны в double, иначе печать
        // дробью ничего не меняет: литерал снова округлится.
        constexpr long long kExact = 9007199254740992LL;   // 2^53
        if (out[(size_t)k].n >  kExact || out[(size_t)k].n < -kExact ||
            out[(size_t)k].d >  kExact) return false;
    }
    return true;
}

std::string ex_rat_expr(const ExRat& r) {
    std::string s = "(numb)(" + std::to_string(r.n) + ".0)";
    if (r.d != 1) s = "(" + s + "/(numb)(" + std::to_string(r.d) + ".0))";
    return s;
}

} // namespace

std::vector<double> extrapolation_weights(const std::vector<int>& n, int p, bool symmetric) {
    const int K = (int)n.size();
    std::vector<double> u((size_t)K), alpha((size_t)K);
    for (int k = 0; k < K; ++k) {
        const double inv = 1.0 / (double)n[k];
        u[k] = symmetric ? inv * inv : inv;
    }
    // alpha_k ~ n_k^p * prod_{m != k} 1/(u_k - u_m). Вектор обратных произведений
    // разностей — это коэффициенты (K-1)-й разделённой разности, то есть ровно
    // тот единственный (с точностью до масштаба) вектор, который ортогонален
    // 1, u, ..., u^(K-2). Нормировка суммы в 1 закрывает условие состоятельности.
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        double prod = 1.0;
        for (int m = 0; m < K; ++m)
            if (m != k) prod *= (u[k] - u[m]);
        alpha[k] = std::pow((double)n[k], (double)p) / prod;
        sum += alpha[k];
    }
    for (int k = 0; k < K; ++k) alpha[k] /= sum;
    return alpha;
}

bool extrapolation_weights_exact(const std::vector<int>& n, int p, bool symmetric,
                                 std::vector<std::string>* out) {
    std::vector<ExRat> r;
    if (!extrapolation_weights_rational(n, p, symmetric, r)) return false;
    if (!out) return true;
    out->clear();
    out->reserve(r.size());
    for (const ExRat& v : r)
        out->push_back(v.d == 1 ? std::to_string(v.n)
                                : std::to_string(v.n) + "/" + std::to_string(v.d));
    return true;
}

// ГБШ (см. Scheme::GBS в codegen.hpp). Не обёртка над готовым телом, как
// wrap_extrapolation, а свой эмиттер: стадия держит ДВА состояния (z_{m-1},
// z_m) через все подшаги, и повтором одношаговой базы её не выразить.
// Правые части печатаются по одному разу на каждое из трёх мест (старт,
// leapfrog, сглаживание), а номер стадии идёт рантайм-циклом, поэтому тело не
// растёт с K. Веса — те же, что у Extr с симметричной базой порядка 2, точными
// дробями, когда сходятся.
namespace {
std::string scheme_gbs(const System& s, int K) {
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size();
    if (N < 1) throw std::runtime_error("GBS needs a non-empty system");
    const std::vector<int> n = gbs_substeps(K);
    const auto f0 = rhs_over(s, "X0_gbs");
    const auto f1 = rhs_over(s, "Z1_gbs");

    std::vector<double> alpha(1, 1.0);
    std::vector<ExRat>  arat(1, ExRat{ 1, 1 });
    bool alpha_exact = true;
    if (K > 1) {
        alpha       = extrapolation_weights(n, 2, true);
        alpha_exact = extrapolation_weights_rational(n, 2, true, arat);
    }
    int cost = 0;
    for (int v : n) cost += v + 1;   // n подшагов + сглаживание

    const std::string Ns = std::to_string(N), Ks = std::to_string(K);
    std::ostringstream o;
    o << "    // --- GBS";
    if (K == 1) o << " (n=2)";
    else for (int k = 0; k < K; ++k) o << (k ? "-" : " ") << n[k];
    o << ": Gragg modified midpoint, " << K << " stage" << (K == 1 ? "" : "s")
      << ", order " << 2 * K << ", " << cost << " RHS evaluations per step ---\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    o << "    numb X0_gbs[" << Ns << "], AC_gbs[" << Ns << "], Z0_gbs[" << Ns
      << "], Z1_gbs[" << Ns << "], ZT_gbs[" << Ns << "];\n";
    o << "    const numb W_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) {
        if (k) o << ", ";
        o << (alpha_exact ? ex_rat_expr(arat[(size_t)k]) : ("(numb)(" + fmtnum(alpha[k]) + ")"));
    }
    o << " };\n";
    o << "    int i_gbs, m_gbs, k_gbs;\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { X0_gbs[i_gbs] = X[i_gbs]; AC_gbs[i_gbs] = (numb)0; }\n";
    o << "    for (k_gbs = 0; k_gbs < " << Ks << "; ++k_gbs) {\n";
    o << "        const int  n_gbs  = 2 * (k_gbs + 1);\n";
    o << "        const numb hs_gbs = h / (numb)n_gbs;\n";
    o << "        // старт: явный Эйлер на первый подшаг\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) Z0_gbs[i_gbs] = X0_gbs[i_gbs];\n";
    for (int i = 0; i < N; ++i)
        o << "        Z1_gbs[" << i << "] = X0_gbs[" << i << "] + hs_gbs * (" << f0[i] << ");\n";
    o << "        // leapfrog: z_{m+1} = z_{m-1} + 2*hs*f(z_m), всего n подшагов\n";
    o << "        for (m_gbs = 1; m_gbs < n_gbs; ++m_gbs) {\n";
    for (int i = 0; i < N; ++i)
        o << "            ZT_gbs[" << i << "] = Z0_gbs[" << i << "] + (numb)2 * hs_gbs * (" << f1[i] << ");\n";
    o << "            for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { Z0_gbs[i_gbs] = Z1_gbs[i_gbs]; Z1_gbs[i_gbs] = ZT_gbs[i_gbs]; }\n";
    o << "        }\n";
    o << "        // сглаживание Грэгга: (z_{n-1} + z_n + hs*f(z_n)) / 2\n";
    for (int i = 0; i < N; ++i)
        o << "        AC_gbs[" << i << "] += W_gbs[k_gbs] * (numb)0.5 * (Z0_gbs[" << i << "] + Z1_gbs["
          << i << "] + hs_gbs * (" << f1[i] << "));\n";
    o << "    }\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) X[i_gbs] = AC_gbs[i_gbs];\n";
    return o.str();
}

// --- Подстановка тела базы в обёртки ------------------------------------------
//
// Обёртки (Extr, ExtrZ, Comp) печатают тело базы прямо в каждую стадию, блоком
// с собственным "const numb h = <подшаг>": внутренний h затеняет макрошаг, и
// тело по-прежнему не правится ни в одном символе. Раньше тело уходило один раз
// в локальную лямбду step_ex(h), и это решал компилятор: NVRTC её встраивал
// всегда, а MSVC на CPU-пути — не всегда. У ExtrZ(Complex CD4|1,2,3) он оставил
// шесть настоящих вызовов на шаг, и шаг выходил на треть медленнее (354 против
// 268 нс, побитово тот же результат). Текст растёт в K раз, но ровно столько
// копий NVRTC делал и сам.
//
// Лямбда остаётся запасным путём для тела с return или goto: return внутри
// подставленного тела вышел бы из всего calculateDiscreteModel, а метка,
// размноженная по стадиям, не скомпилировалась бы (метки видны во всей функции).
bool wrapper_can_inline(const std::string& body) {
    for (const char* kw : { "return", "goto" }) {
        const size_t n = std::strlen(kw);
        for (size_t p = body.find(kw); p != std::string::npos; p = body.find(kw, p + 1)) {
            const bool left  = p == 0 || !(std::isalnum((unsigned char)body[p - 1]) || body[p - 1] == '_');
            const bool right = p + n >= body.size()
                               || !(std::isalnum((unsigned char)body[p + n]) || body[p + n] == '_');
            if (left && right) return false;
        }
    }
    return true;
}

// Тело построчно с дополнительным отступом: панель KRS показывает текст как есть.
void emit_indented(std::ostringstream& o, const std::string& body, const char* indent) {
    size_t pos = 0;
    while (pos < body.size()) {
        size_t eol = body.find('\n', pos);
        if (eol == std::string::npos) eol = body.size();
        const std::string line = body.substr(pos, eol - pos);
        if (!line.empty()) o << indent << line;
        o << "\n";
        pos = eol + 1;
    }
}
} // namespace

std::string wrap_extrapolation(const std::string& base_body, int N,
                               const std::vector<int>& n, int p, bool symmetric,
                               const std::string& base_name) {
    const int K = (int)n.size();
    if (K < 1) throw std::runtime_error("extrapolation needs at least one stage");
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");

    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    // Точные дроби — если сошлись (см. extrapolation_weights_rational).
    std::vector<ExRat> arat;
    const bool alpha_exact = extrapolation_weights_rational(n, p, symmetric, arat);
    long long cost = 0;
    for (int k = 0; k < K; ++k) cost += n[k];

    const std::string Ns = std::to_string(N);
    std::ostringstream o;

    o << "    // --- " << make_extrapolation_name(base_name, n) << " ---\n";
    o << "    // base: order " << p << ", " << (symmetric ? "symmetric" : "non-symmetric")
      << "  ->  extrapolated order " << extrapolation_order(K, p, symmetric) << "\n";
    o << "    // " << K << " stages, " << cost << " base steps per macro-step\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n[" << k << "] = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    o << "    numb X0_ex[" << Ns << "], AC_ex[" << Ns << "];\n";
    o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { X0_ex[i_ex] = X[i_ex]; AC_ex[i_ex] = (numb)0; }\n\n";

    // Тело базы — в каждую стадию, блоком со своим h (см. wrapper_can_inline).
    // Внутренний h затеняет макрошаг, поэтому любое "h" в базе становится
    // подшагом без единой правки текста — это же позволяет обернуть кастомную
    // КРС, содержимое которой нам непрозрачно.
    const bool inl = wrapper_can_inline(base_body);
    if (inl) {
        o << "    const numb h_ex = h;\n\n";
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_ex = [&](const numb h) {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
    }

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << "\n";
        if (inl) {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) {\n";
            o << "        const numb h = h_ex / (numb)" << fmtnum((double)n[k]) << ";\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        } else {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) step_ex(h / (numb)"
              << fmtnum((double)n[k]) << ");\n";
        }
        o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { AC_ex[i_ex] += "
          << (alpha_exact ? ex_rat_expr(arat[(size_t)k])
                          : ("(numb)(" + fmtnum(alpha[k]) + ")"))
          << " * X[i_ex];";
        // После последней стадии откатывать нечего — X сразу перезаписывается.
        if (k + 1 < K) o << " X[i_ex] = X0_ex[i_ex];";
        o << " }\n";
    }

    o << "\n    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) X[i_ex] = AC_ex[i_ex];\n";
    return o.str();
}

// ExtrZ. Отличие от wrap_extrapolation ровно одно: состояние стадий,
// начальная точка и накопитель — ucmplx, ядро базы работает прямо на Z[] без
// Re, а Re берётся один раз после суммы. Тогда стадия из n подшагов —
// это exp(n * log Psi_{h/n}) целиком, её разложение по 1/n идёт с
// вещественными членами только при чётных степенях, и симметричные веса
// гасят их через одну (см. ExtrapolationSpec в codegen.hpp).
std::string wrap_extrapolation_complex(const std::string& core_body, int N,
                                       const std::vector<int>& n, int p,
                                       const std::string& base_name) {
    const int K = (int)n.size();
    if (K < 1) throw std::runtime_error("extrapolation needs at least one stage");
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");

    const bool symmetric = extrapolation_symmetric(true, false);
    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    std::vector<ExRat> arat;
    const bool alpha_exact = extrapolation_weights_rational(n, p, symmetric, arat);
    long long cost = 0;
    for (int k = 0; k < K; ++k) cost += n[k];

    const std::string Ns = std::to_string(N);
    std::ostringstream o;

    o << "    // --- " << make_extrapolation_name(base_name, n, {}, true) << " ---\n";
    o << "    // base: order " << p << ", complex state across substeps, Re only on output\n";
    o << "    //   ->  extrapolated order " << extrapolation_order(K, p, symmetric) << "\n";
    o << "    // " << K << " stages, " << cost << " base steps per macro-step\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n[" << k << "] = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    // Z — то самое имя, над которым работает ядро базы.
    o << "    ucmplx Z[" << Ns << "], Z0_ex[" << Ns << "], AC_ex[" << Ns << "];\n";
    o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) {\n"
      << "        Z0_ex[i_ex] = ucmplx(X[i_ex], 0.0); Z[i_ex] = Z0_ex[i_ex];"
      << " AC_ex[i_ex] = ucmplx(0.0, 0.0);\n    }\n\n";

    // Ядро — встроенная CD-схема, return/goto в нём нет: подставляется всегда.
    o << "    const numb h_ex = h;\n\n";

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << ", no Re in between\n";
        o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) {\n";
        o << "        const numb h = h_ex / (numb)" << fmtnum((double)n[k]) << ";\n";
        emit_indented(o, core_body, "    ");
        o << "    }\n";
        o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { AC_ex[i_ex] = AC_ex[i_ex] + "
          << (alpha_exact ? ex_rat_expr(arat[(size_t)k])
                          : ("(numb)(" + fmtnum(alpha[k]) + ")"))
          << " * Z[i_ex];";
        if (k + 1 < K) o << " Z[i_ex] = Z0_ex[i_ex];";
        o << " }\n";
    }

    o << "\n    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) X[i_ex] = AC_ex[i_ex].re;\n";
    return o.str();
}

// --- Композиция -------------------------------------------------------------

static const char* const kCompPrefix = "Comp(";

std::string make_composition_name(const std::string& base,
                                  const std::vector<std::string>& gammas,
                                  const std::string& label) {
    std::string s = kCompPrefix;
    if (!label.empty()) s += label + "|";
    s += base + "|";
    for (size_t k = 0; k < gammas.size(); ++k) {
        if (k) s += ",";
        s += gammas[k];
    }
    return s + ")";
}

namespace {

    std::string comp_trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    // Режем только по запятым ВЕРХНЕГО уровня: коэффициент — выражение, и
    // запятая внутри pow(g,2) не должна заканчивать токен.
    bool comp_split(const std::string& tail, std::vector<std::string>& out) {
        int depth = 0;
        std::string cur;
        for (char c : tail) {
            if (c == '(') ++depth;
            else if (c == ')' && --depth < 0) return false;
            if (c == ',' && depth == 0) { out.push_back(cur); cur.clear(); continue; }
            cur += c;
        }
        if (depth != 0) return false;
        out.push_back(cur);
        return true;
    }

    // Сворачивает коэффициент в число. false на любом символе — тогда значение
    // известно только в рантайме, и звать нас незачем.
    bool comp_fold(const PN& n, double* v) {
        if (!n) return false;
        double a = 0, b = 0;
        switch (n->kind) {
        case Node::Num: *v = n->num; return true;
        case Node::Neg: if (!comp_fold(n->a, &a)) return false; *v = -a; return true;
        case Node::Add: case Node::Sub: case Node::Mul: case Node::Div: case Node::Pow:
            if (!comp_fold(n->a, &a) || !comp_fold(n->b, &b)) return false;
            switch (n->kind) {
            case Node::Add: *v = a + b; break;
            case Node::Sub: *v = a - b; break;
            case Node::Mul: *v = a * b; break;
            case Node::Div: if (b == 0.0) return false; *v = a / b; break;
            default:        *v = std::pow(a, b); break;
            }
            return true;
        default: return false;   // Sym / Call
        }
    }

} // namespace

bool parse_composition_name(const std::string& name, CompositionSpec* out,
                            std::string* err) {
    auto fail = [&](const char* msg) { if (err) *err = msg; return false; };
    auto fail_s = [&](const std::string& msg) { if (err) *err = msg; return false; };

    const std::string pref = kCompPrefix;
    if (name.size() <= pref.size() || name.compare(0, pref.size(), pref) != 0)
        return fail("not a composed scheme name");
    if (name.back() != ')')
        return fail("missing closing ')'");

    const std::string inner = name.substr(pref.size(), name.size() - pref.size() - 1);
    // rfind, как у Extr: имя базы по правилам валидации '|' не содержит, но из
    // старого JSON могло приехать что угодно — резать надо по последней черте.
    const size_t bar = inner.rfind('|');
    if (bar == std::string::npos) return fail("missing '|' between base and coefficients");

    CompositionSpec spec;
    // Голова — "<база>" либо "<метка>|<база>", как у Extr.
    if (!wrap_split_head(inner.substr(0, bar), &spec.label, &spec.base, err)) return false;
    if (spec.base.empty()) return fail("empty base scheme name");
    // Обёртка над обёрткой отрезает рекурсию в резолвере, как и у Extr.
    if (spec.base.compare(0, pref.size(), pref) == 0 ||
        spec.base.compare(0, 5, "Extr(") == 0 || spec.base.compare(0, 6, "ExtrZ(") == 0)
        return fail("base scheme cannot itself be a wrapper");

    std::vector<std::string> toks;
    if (!comp_split(inner.substr(bar + 1), toks)) return fail("unbalanced parentheses");

    for (const std::string& t : toks) {
        const std::string g = comp_trim(t);
        if (g.empty()) return fail("empty coefficient");
        // Синтаксис проверяем здесь, чтобы кривое имя не доехало до Run: имена
        // резолвятся позже, в wrap_composition, где уже известна система.
        try { Parser(g, false).parse(); }
        catch (const std::exception& e) {
            if (err) *err = "coefficient \"" + g + "\": " + e.what();
            return false;
        }
        spec.gammas.push_back(g);
    }

    const int K = (int)spec.gammas.size();
    if (K < kCompMinStages)
        return fail_s("at least " + std::to_string(kCompMinStages) + " stages are needed");
    if (K > kCompMaxStages)
        return fail_s("at most " + std::to_string(kCompMaxStages) + " stages are supported");

    if (out) *out = spec;
    return true;
}

namespace {

    // Чистый числовой литерал: знак, цифры, точка, экспонента — и ничего
    // больше. Нужен ровно одному месту: решить, был ли первый столбец строки
    // нумерующим. Поэтому и такая строгость — "- 2*g1" числом не считается,
    // и "1 - 2*g1" не теряет свою единицу.
    bool comp_is_plain_number(const std::string& s) {
        if (s.empty()) return false;
        size_t i = 0;
        if (s[i] == '+' || s[i] == '-') ++i;
        bool digits = false, dot = false;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.')) {
            if (s[i] == '.') { if (dot) return false; dot = true; }
            else digits = true;
            ++i;
        }
        if (!digits) return false;
        if (i == s.size()) return true;
        if (s[i] != 'e' && s[i] != 'E') return false;
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        if (i == s.size()) return false;
        while (i < s.size()) { if (!std::isdigit((unsigned char)s[i])) return false; ++i; }
        return true;
    }

    // Срезает нумерующий первый столбец: "3<tab>-0.3825" -> "-0.3825".
    // Разделителем считаются пробелы/табы и ')' ';' ':' — но НЕ точка (иначе
    // "1.35" распалось бы на индекс 1 и коэффициент 35) и НЕ запятая (её
    // разбирает разделение строки на коэффициенты).
    std::string comp_strip_index(const std::string& line) {
        size_t i = 0;
        while (i < line.size() && std::isdigit((unsigned char)line[i])) ++i;
        if (i == 0 || i == line.size()) return line;
        size_t j = i;
        while (j < line.size() && (std::isspace((unsigned char)line[j])
                                   || line[j] == ')' || line[j] == ';' || line[j] == ':')) ++j;
        if (j == i || j == line.size()) return line;
        const std::string rest = comp_trim(line.substr(j));
        return comp_is_plain_number(rest) ? rest : line;
    }

} // namespace

bool parse_composition_coeff_file(const std::string& text,
                                  std::vector<std::string>* out, std::string* err) {
    auto fail = [&](const std::string& msg) { if (err) *err = msg; return false; };

    std::vector<std::string> coeffs;
    size_t pos = 0;
    int line_no = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const std::string raw = text.substr(pos, eol == std::string::npos
                                                     ? std::string::npos : eol - pos);
        pos = (eol == std::string::npos) ? text.size() + 1 : eol + 1;
        ++line_no;

        // comp_trim снимает и '\r' от CRLF, и табуляцию в конце строки —
        // именно в таком виде таблицы коэффициентов обычно и лежат.
        std::string line = comp_trim(raw);
        if (line.empty()) continue;
        if (line[0] == '#' || line[0] == '%' || line.compare(0, 2, "//") == 0) continue;

        line = comp_strip_index(line);

        // Запятые верхнего уровня — тем же расщепителем, что у имени схемы:
        // так "g1, 1-2*g1, g1" в одну строку работает, а pow(g,2) не рвётся.
        std::vector<std::string> toks;
        if (!comp_split(line, toks))
            return fail("line " + std::to_string(line_no) + ": unbalanced parentheses");
        for (const std::string& t : toks) {
            const std::string g = comp_trim(t);
            if (g.empty()) continue;   // хвостовая запятая — не ошибка
            try { Parser(g, false).parse(); }
            catch (const std::exception& e) {
                return fail("line " + std::to_string(line_no) + ": \"" + g + "\": " + e.what());
            }
            coeffs.push_back(g);
        }
    }

    if ((int)coeffs.size() < kCompMinStages)
        return fail("found " + std::to_string(coeffs.size()) + " coefficient(s), at least "
                    + std::to_string(kCompMinStages) + " are needed");
    if ((int)coeffs.size() > kCompMaxStages)
        return fail("found " + std::to_string(coeffs.size()) + " coefficients, at most "
                    + std::to_string(kCompMaxStages) + " stages are supported");

    if (out) *out = coeffs;
    return true;
}

bool composition_sums(const CompositionSpec& spec, double* sum, double* cube_sum) {
    double s = 0.0, c = 0.0;
    for (const std::string& g : spec.gammas) {
        double v = 0.0;
        PN ast;
        try { ast = Parser(g, false).parse(); } catch (...) { return false; }
        if (!comp_fold(ast, &v)) return false;
        s += v;
        c += v * v * v;
    }
    if (sum)      *sum = s;
    if (cube_sum) *cube_sum = c;
    return true;
}

namespace {

    // comp_fold с окружением: Sym берётся из env, Call считается для функций,
    // которые в коэффициенте шага вообще осмысленны (sqrt(2), cbrt(2) в
    // классических композициях). Неизвестное имя — честный false, а не ноль:
    // подставить ноль значило бы показать пользователю число, которого схема
    // считать не будет.
    bool comp_fold_env(const PN& n, const std::map<std::string, double>& env, double* v) {
        if (!n) return false;
        double a = 0, b = 0;
        switch (n->kind) {
        case Node::Num: *v = n->num; return true;
        case Node::Sym: {
            auto it = env.find(n->name);
            if (it == env.end()) return false;
            *v = it->second; return true;
        }
        case Node::Neg:
            if (!comp_fold_env(n->a, env, &a)) return false;
            *v = -a; return true;
        case Node::Add: case Node::Sub: case Node::Mul: case Node::Div: case Node::Pow:
            if (!comp_fold_env(n->a, env, &a) || !comp_fold_env(n->b, env, &b)) return false;
            switch (n->kind) {
            case Node::Add: *v = a + b; break;
            case Node::Sub: *v = a - b; break;
            case Node::Mul: *v = a * b; break;
            case Node::Div: if (b == 0.0) return false; *v = a / b; break;
            default:        *v = std::pow(a, b); break;
            }
            return true;
        case Node::Call: {
            if (n->args.size() == 1) {
                if (!comp_fold_env(n->args[0], env, &a)) return false;
                const std::string& f = n->name;
                if (f == "sqrt") { if (a < 0) return false; *v = std::sqrt(a); return true; }
                if (f == "cbrt")               { *v = std::cbrt(a); return true; }
                if (f == "exp")                { *v = std::exp(a);  return true; }
                if (f == "log")  { if (a <= 0) return false; *v = std::log(a); return true; }
                if (f == "sin")                { *v = std::sin(a);  return true; }
                if (f == "cos")                { *v = std::cos(a);  return true; }
                if (f == "tan")                { *v = std::tan(a);  return true; }
                if (f == "fabs" || f == "abs") { *v = std::fabs(a); return true; }
                return false;
            }
            if (n->args.size() == 2 && n->name == "pow") {
                if (!comp_fold_env(n->args[0], env, &a)) return false;
                if (!comp_fold_env(n->args[1], env, &b)) return false;
                *v = std::pow(a, b); return true;
            }
            return false;
        }
        default: return false;
        }
    }

    // Округляет КАЖДЫЙ числовой литерал в строке, не трогая имена и операторы.
    // Работает и на цельном числе, и на выражении: "-1.2599210498948732*g1"
    // укорачивается так же, как "1.3512071919596578".
    std::string comp_round_literals(const std::string& s, int digits) {
        std::string out;
        for (size_t i = 0; i < s.size(); ) {
            const char c = s[i];
            // Идентификатор забирает свои цифры целиком, иначе g1 стало бы g1.
            if (std::isalpha((unsigned char)c) || c == '_') {
                while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_'))
                    out += s[i++];
                continue;
            }
            const bool starts_num = std::isdigit((unsigned char)c)
                || (c == '.' && i + 1 < s.size() && std::isdigit((unsigned char)s[i + 1]));
            if (!starts_num) { out += c; ++i; continue; }

            const char* start = s.c_str() + i;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start) { out += c; ++i; continue; }
            const size_t len = (size_t)(end - start);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
            // Короче из двух: у "2" и "1e-05" %g даёт ровно исходное.
            if (std::strlen(buf) < len) out += buf;
            else                        out.append(start, len);
            i += len;
        }
        return out;
    }

} // namespace

bool composition_gamma_values(const CompositionSpec& spec,
                              const std::map<std::string, std::string>& param_values,
                              std::vector<double>* out, std::string* err) {
    // Окружение: сначала константы, затем параметры. Значение параметра само
    // разбирается выражением (в библиотеке лежат "8/3" и "1e-5"), но уже БЕЗ
    // окружения: параметр, сосланный на другой параметр, — это цепочка, и
    // раскрывать её молча значило бы угадывать порядок вычисления.
    std::map<std::string, double> env;
    env["pi"] = 3.14159265358979323846;
    for (const auto& kv : param_values) {
        if (kv.second.empty()) continue;
        double v = 0.0;
        PN ast;
        try { ast = Parser(kv.second, false).parse(); } catch (...) { continue; }
        if (comp_fold(ast, &v)) env[kv.first] = v;
    }

    const int K = (int)spec.gammas.size();
    if (out) out->assign((size_t)K, std::numeric_limits<double>::quiet_NaN());
    bool all_ok = true;
    for (int k = 0; k < K; ++k) {
        const std::string& g = spec.gammas[(size_t)k];
        double v = 0.0;
        PN ast;
        try { ast = Parser(g, false).parse(); }
        catch (...) {
            if (all_ok && err) *err = "coefficient \"" + g + "\" does not parse";
            all_ok = false; continue;
        }
        if (!comp_fold_env(ast, env, &v)) {
            if (all_ok && err)
                *err = "coefficient \"" + g + "\" needs a parameter value that is not set";
            all_ok = false; continue;
        }
        if (out) (*out)[(size_t)k] = v;
    }
    return all_ok;
}

std::string wrapper_display_name(const std::string& name, int digits) {
    // Метка пользователя ЗАМЕНЯЕТ описание целиком: её задавали ровно затем,
    // чтобы в списке и в комбо стояло "S17o8", а не семнадцать коэффициентов.
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return esp.label.empty() ? name : esp.label;

    CompositionSpec spec;
    if (!parse_composition_name(name, &spec)) return name;
    if (!spec.label.empty()) return spec.label;
    if (digits < 1)  digits = 1;
    if (digits > 17) digits = 17;

    std::vector<std::string> shown;
    shown.reserve(spec.gammas.size());
    for (const std::string& g : spec.gammas)
        shown.push_back(comp_round_literals(g, digits));
    return make_composition_name(spec.base, shown);
}

// --- Порядок списка обёрток --------------------------------------------------

namespace {

// Ключ сортировки обёртки. Метку не содержит: переименование строку не двигает.
struct WrapperKey {
    int              kind = 3;      // 0 Extr, 1 ExtrZ, 2 Comp, 3 не разобралось
    int              base_rank = 0; // строка паспорта; кастомные — после всех встроенных
    std::string      base;          // для кастомных опорников — алфавит
    int              stages = 0;
    std::vector<int> n;             // подшаги Extr
    std::string      canon;         // имя без метки — последний разделитель
};

WrapperKey wrapper_key(const std::string& name) {
    WrapperKey k;
    ExtrapolationSpec esp;
    CompositionSpec   csp;
    if (parse_extrapolation_name(name, &esp)) {
        k.kind   = esp.re_at_output ? 1 : 0;
        k.base   = esp.base;
        k.stages = (int)esp.n.size();
        k.n      = esp.n;
    } else if (parse_composition_name(name, &csp)) {
        k.kind   = 2;
        k.base   = csp.base;
        k.stages = (int)csp.gammas.size();
    }
    const int r = builtin_scheme_rank(k.base);
    k.base_rank = r >= 0 ? r : INT_MAX;
    k.canon     = wrapper_canonical_name(name);
    return k;
}

bool wrapper_key_less(const WrapperKey& x, const WrapperKey& y) {
    if (x.kind != y.kind)           return x.kind < y.kind;
    if (x.base_rank != y.base_rank) return x.base_rank < y.base_rank;
    if (x.base != y.base)           return x.base < y.base;
    if (x.stages != y.stages)       return x.stages < y.stages;
    if (x.n != y.n)                 return x.n < y.n;
    return x.canon < y.canon;
}

} // namespace

void sort_wrapper_schemes(std::vector<std::string>& names) {
    std::vector<std::pair<WrapperKey, std::string>> v;
    v.reserve(names.size());
    for (auto& nm : names) v.emplace_back(wrapper_key(nm), std::move(nm));
    std::stable_sort(v.begin(), v.end(), [](const auto& x, const auto& y) {
        return wrapper_key_less(x.first, y.first);
    });
    for (size_t i = 0; i < v.size(); ++i) names[i] = std::move(v[i].second);
}

std::string wrapper_group_title(const std::string& name) {
    static const char* const kKind[] = { "Extr", "ExtrZ", "Comp" };
    const WrapperKey k = wrapper_key(name);
    if (k.kind > 2) return {};
    return std::string(kKind[k.kind]) + " over " + k.base;
}

std::string wrapper_canonical_name(const std::string& name) {
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return make_extrapolation_name(esp.base, esp.n, {}, esp.re_at_output);
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return make_composition_name(csp.base, csp.gammas);
    return name;
}

std::string wrapper_relabel(const std::string& name, const std::string& label) {
    const std::string lb = wrapper_sanitize_label(label);
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return make_extrapolation_name(esp.base, esp.n, lb, esp.re_at_output);
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return make_composition_name(csp.base, csp.gammas, lb);
    return name;
}

std::string wrapper_label(const std::string& name) {
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp)) return esp.label;
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return csp.label;
    return {};
}

std::string wrap_composition(const std::string& base_body, const System& sys,
                             const std::vector<std::string>& gammas,
                             int p, bool symmetric, const std::string& base_name) {
    const int K = (int)gammas.size();
    if (K < 1) throw std::runtime_error("composition needs at least one stage");
    if (base_body.empty()) throw std::runtime_error("composition needs a base body");

    // Коэффициенты видят ТОЛЬКО параметры и константы: переменная состояния
    // здесь означала бы шаг, зависящий от X посреди шага.
    NameMap nm;
    for (size_t j = 0; j < sys.params.size(); ++j)
        nm.m[sys.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
    for (const auto& c : math_constants()) nm.m[c] = c;

    std::vector<std::string> gc((size_t)K);
    for (int k = 0; k < K; ++k) {
        try { gc[(size_t)k] = emit_to_str(Parser(gammas[(size_t)k], false).parse(), nm); }
        catch (const std::exception& e) {
            throw std::runtime_error("composition coefficient \"" + gammas[(size_t)k]
                                     + "\": " + e.what()
                                     + " (only system parameters and pi are allowed)");
        }
    }

    std::ostringstream o;
    o << "    // --- " << make_composition_name(base_name, gammas) << " ---\n";
    o << "    // base: order " << p << ", " << (symmetric ? "symmetric" : "non-symmetric")
      << "; " << K << " stages, " << K << " base steps per macro-step\n";

    CompositionSpec probe;
    probe.gammas = gammas;
    double gsum = 0.0, gcube = 0.0;
    if (!composition_sums(probe, &gsum, &gcube)) {
        o << "    // coefficients are symbolic -- the order follows their run-time\n"
             "    // values; sweep them in the Order tab to see it\n";
    } else {
        o << "    // sum(gamma) = " << fmtnum(gsum)
          << (std::fabs(gsum - 1.0) < 1e-12
                  ? "  (consistent)\n"
                  : "  -- NOT 1: this integrates the field scaled by that factor\n");
        o << "    // sum(gamma^3) = " << fmtnum(gcube);
        if (symmetric && std::fabs(gcube) < 1e-12) o << "  ->  expected order " << (p + 2) << "\n";
        else if (symmetric)                        o << "  -- not 0, order stays " << p << "\n";
        else                                       o << "  (base is not symmetric: order not implied)\n";
    }
    for (int k = 0; k < K; ++k)
        o << "    //   gamma[" << k << "] = " << gammas[(size_t)k]
          << (gc[(size_t)k] == gammas[(size_t)k] ? "" : "  ->  " + gc[(size_t)k]) << "\n";

    // Та же подстановка, что у wrap_extrapolation, и по той же причине:
    // внутренний h затеняет макрошаг, поэтому база не правится ни в одном символе.
    if (wrapper_can_inline(base_body)) {
        o << "    const numb h_co = h;\n";
        for (int k = 0; k < K; ++k) {
            o << "    {   // stage " << k << "\n";
            o << "        const numb h = h_co * (" << gc[(size_t)k] << ");\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        }
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_co = [&](const numb h) {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
        for (int k = 0; k < K; ++k)
            o << "    step_co(h * (" << gc[(size_t)k] << "));\n";
    }

    return o.str();
}

std::string codegen_scheme_cpu_equivalent(const System& s, Scheme sch) {
    // Одна форма на оба пути. CPU-интегратор считает тот же AST через
    // байткод-интерпретатор, а диагонально-неявные схемы (CD, Complex CD, SIMP,
    // D) решают каждое уравнение той же аналитической формулой — коэффициенты
    // даёт SystemEvaluator::solve_diag_implicit, построенный тем же
    // cd_try_extract_linear, что и кодоген. Итерации остаются только там, где
    // их эмитит и GPU: уравнение нелинейно по своей переменной.
    // Расхождение возможно лишь в группировке FMA у компилятора.
    return codegen_scheme(s, sch);
}

std::vector<std::string> codegen_rhs(const System& s) {
    return rhs_over(s, "X");
}

// Нормализация значения параметра
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