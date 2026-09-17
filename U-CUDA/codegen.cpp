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
#include <vector>

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
    enum class CdKind { Real, Cx, Cx4 };

    std::string scheme_cd_common(const System& s, CdKind kind) {
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
        if (cx) {
            o << "    ucmplx Z[" << N << "];\n";
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = ucmplx(X[" << i << "], 0.0);\n";
        }
        if (kind == CdKind::Real) {
            o << "    numb h1 = h * a[0];\n";
            o << "    numb h2 = h * (1 - a[0]);\n";
        }
        else if (kind == CdKind::Cx) {
            o << "    ucmplx h1 = ucmplx(a[0] * h,  h * " << fmtnum(CCD_IMAG) << ");\n";
            o << "    ucmplx h2 = ucmplx((1 - a[0]) * h, -h * " << fmtnum(CCD_IMAG) << ");\n";
        }
        else {
            // gamma*h and conj(gamma)*h; s splits each of them inside its pass.
            o << "    ucmplx g  = ucmplx(0.5 * h,  h * " << fmtnum(CCD_IMAG) << ");\n";
            o << "    ucmplx gc = ucmplx(0.5 * h, -h * " << fmtnum(CCD_IMAG) << ");\n";
            o << "    ucmplx h1 = g * a[0];\n";
            o << "    ucmplx h2 = g * (1 - a[0]);\n";
        }

        // Phi*_{h2}: diagonally-implicit half-step, reverse order. Сам разбор
        // уравнения — в emit_diag_implicit_eq выше: стадия SIMP гоняет тот же
        // код с h1 и в прямом порядке.

        // Один проход CD целиком: явный полушаг h1 вперёд, неявный h2 назад.
        // Complex CD4 зовёт его дважды с разными h1/h2, поэтому имена временных
        // переменных неявной ветки получают суффикс — иначе второй проход
        // переобъявил бы x0_cd в той же области видимости.
        auto emit_pass = [&](const char* sfx) {
        // Phi_{h1}: explicit half-step, forward order. Each X[i] update sees
        // the just-written values of X[0..i-1] (Euler-Cromer coupling).
        for (int i = 0; i < N; ++i)
            o << "    " << stv << "[" << i << "] = " << stv << "[" << i << "] + h1 * ("
              << emit_to_str(rhs_ast[i], nm) << ");\n";

        for (int i = N - 1; i >= 0; --i)
            emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i],
                                  stv + ("[" + std::to_string(i) + "]"), nm,
                                  "h2", sty, "x" + std::to_string(i) + "_cd" + sfx);
        };  // emit_pass

        emit_pass("");
        if (kind == CdKind::Cx4) {
            o << "    h1 = gc * a[0];\n";
            o << "    h2 = gc * (1 - a[0]);\n";
            emit_pass("_b");
        }

        // Наружу — только действительная часть: X[] вещественный и на входе, и
        // на выходе, поэтому вся обвязка (ядра, LLE/LS, бассейны, рендер) о
        // комплексности не знает.
        if (cx)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";

        return o.str();
    }

    std::string scheme_cd(const System& s)          { return scheme_cd_common(s, CdKind::Real); }
    std::string scheme_complex_cd(const System& s)  { return scheme_cd_common(s, CdKind::Cx); }
    std::string scheme_complex_cd4(const System& s) { return scheme_cd_common(s, CdKind::Cx4); }

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

// Публичная функция
std::string codegen_scheme(const System& s, Scheme sch) {
    switch (sch) {
    case Scheme::Euler:            return scheme_euler(s);
    case Scheme::EulerCromer:      return scheme_euler_cromer(s);
    case Scheme::ExplicitMidpoint: return scheme_midpoint(s);
    case Scheme::RK4:              return scheme_rk4(s);
    case Scheme::DOPRI78:          return scheme_dopri78(s);
    case Scheme::CD:               return scheme_cd(s);
    case Scheme::ComplexCD:        return scheme_complex_cd(s);
    case Scheme::ComplexCD4:       return scheme_complex_cd4(s);
    case Scheme::ImplicitEuler:    return scheme_implicit_euler(s);
    case Scheme::ImplicitMidpoint: return scheme_implicit_midpoint(s);
    case Scheme::SEMP:             return scheme_semp(s);
    case Scheme::SIMP:             return scheme_simp(s);
    case Scheme::D:                return scheme_d(s);
    case Scheme::ComplexIEuler:    return scheme_complex_ieuler(s);
    case Scheme::Map:              return scheme_map(s);
    }
    throw std::runtime_error("unknown scheme");
}

Scheme scheme_from_name(const std::string& name) {
    if (name == "Euler-Cromer")      return Scheme::EulerCromer;
    if (name == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (name == "RK4")               return Scheme::RK4;
    if (name == "DOPRI78")           return Scheme::DOPRI78;
    if (name == "CD")                return Scheme::CD;
    if (name == "Complex CD")        return Scheme::ComplexCD;
    if (name == "Complex CD4")       return Scheme::ComplexCD4;
    if (name == "Implicit Euler")    return Scheme::ImplicitEuler;
    if (name == "Implicit Midpoint") return Scheme::ImplicitMidpoint;
    if (name == "SEMP")              return Scheme::SEMP;
    if (name == "SIMP")              return Scheme::SIMP;
    if (name == "D")                 return Scheme::D;
    if (name == "Complex Implicit Euler") return Scheme::ComplexIEuler;
    if (name == "Map")               return Scheme::Map;
    return Scheme::Euler;
}

// --- Паспорт встроенных схем -----------------------------------------------
// Порядки — те, что замерены и записаны в подсказках чекбоксов (см. gui.cpp).
// Напоминание про a[0] = 1/2: у CD, Complex CD, Complex CD4, SEMP и SIMP
// заявленный порядок достигается только при этом значении, иначе все они
// падают до первого. Здесь стоит паспортный (то есть при a[0] = 1/2) —
// предупредить пользователя обязан UI.
bool builtin_scheme_traits(const std::string& name, int* order, bool* symmetric) {
    struct Row { const char* name; int order; bool symmetric; };
    static const Row kRows[] = {
        { "Euler",                  1, false },
        { "Euler-Cromer",           1, false },
        { "D",                      1, false },
        { "Implicit Euler",         1, false },
        { "Explicit Midpoint",      2, false },
        { "Implicit Midpoint",      2, true  },  // Phi* ∘ Phi, самосопряжённая
        { "CD",                     2, true  },  // самосопряжённая при a[0] = 1/2
        { "Complex CD",             2, false },
        { "Complex Implicit Euler", 2, false },
        { "SEMP",                   2, false },  // предиктор-корректор, не Phi* ∘ Phi
        { "SIMP",                   2, false },
        { "RK4",                    4, false },
        { "Complex CD4",            4, false },
        { "DOPRI78",                8, false },
    };
    for (const Row& r : kRows) {
        if (name == r.name) {
            if (order)     *order     = r.order;
            if (symmetric) *symmetric = r.symmetric;
            return true;
        }
    }
    return false;
}

// --- Экстраполяция Ричардсона ----------------------------------------------

static const char* const kExtrPrefix = "Extr(";

std::string make_extrapolation_name(const std::string& base, const std::vector<int>& n) {
    std::string s = kExtrPrefix + base + "|";
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

    const std::string pref = kExtrPrefix;
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
    spec.base = inner.substr(0, bar);
    if (spec.base.empty()) return fail("empty base scheme name");
    // Вложенность запрещена: порядок обёртки над обёрткой считается не как
    // p + K - 1, и заодно это отрезает бесконечную рекурсию в резолвере.
    if (spec.base.compare(0, pref.size(), pref) == 0)
        return fail("base scheme cannot itself be extrapolated");

    const std::string tail = inner.substr(bar + 1);
    size_t pos = 0;
    while (true) {
        const size_t comma = tail.find(',', pos);
        const std::string tok = tail.substr(pos, comma == std::string::npos
                                                ? std::string::npos : comma - pos);
        int v = 0;
        if (!extr_parse_substeps(tok, &v)) return fail("substep counts must be integers in 1..1024");
        spec.n.push_back(v);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    const int K = (int)spec.n.size();
    if (K < kExtrMinStages) return fail("at least 2 stages are needed");
    if (K > kExtrMaxStages) return fail("at most 6 stages are supported");
    for (int k = 1; k < K; ++k)
        if (spec.n[k] <= spec.n[k - 1]) return fail("substep counts must strictly increase");

    if (out) *out = spec;
    return true;
}

int extrapolation_order(int stages, int p, bool symmetric) {
    if (stages < 1) return p;
    return symmetric ? p + 2 * (stages - 1) : p + stages - 1;
}

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

std::string wrap_extrapolation(const std::string& base_body, int N,
                               const std::vector<int>& n, int p, bool symmetric,
                               const std::string& base_name) {
    const int K = (int)n.size();
    if (K < 1) throw std::runtime_error("extrapolation needs at least one stage");
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");

    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    long long cost = 0;
    for (int k = 0; k < K; ++k) cost += n[k];

    const std::string Ns = std::to_string(N);
    std::ostringstream o;

    o << "    // --- " << make_extrapolation_name(base_name, n) << " ---\n";
    o << "    // base: order " << p << ", " << (symmetric ? "symmetric" : "non-symmetric")
      << "  ->  extrapolated order " << extrapolation_order(K, p, symmetric) << "\n";
    o << "    // " << K << " stages, " << cost << " base steps per macro-step\n";
    for (int k = 0; k < K; ++k)
        o << "    //   n[" << k << "] = " << n[k] << "   alpha = " << fmtnum(alpha[k]) << "\n";

    o << "    numb X0_ex[" << Ns << "], AC_ex[" << Ns << "];\n";
    o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { X0_ex[i_ex] = X[i_ex]; AC_ex[i_ex] = (numb)0; }\n\n";

    // Тело базы вставляется дословно и ровно один раз. Параметр лямбды назван h
    // и затеняет макрошаг из сигнатуры calculateDiscreteModel, поэтому любое
    // "h" внутри базы становится подшагом без единой правки текста — это же и
    // позволяет обернуть кастомную КРС, содержимое которой нам непрозрачно.
    o << "    auto step_ex = [&](const numb h) {\n";
    {
        // Сдвигаем базу на уровень вложенности: панель KRS показывает этот текст
        // как есть, и без отступа он читается хуже.
        size_t pos = 0;
        while (pos < base_body.size()) {
            size_t eol = base_body.find('\n', pos);
            if (eol == std::string::npos) eol = base_body.size();
            const std::string line = base_body.substr(pos, eol - pos);
            if (!line.empty()) o << "    " << line;
            o << "\n";
            pos = eol + 1;
        }
    }
    o << "    };\n\n";

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << "\n";
        o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) step_ex(h / (numb)"
          << fmtnum((double)n[k]) << ");\n";
        o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { AC_ex[i_ex] += (numb)("
          << fmtnum(alpha[k]) << ") * X[i_ex];";
        // После последней стадии откатывать нечего — X сразу перезаписывается.
        if (k + 1 < K) o << " X[i_ex] = X0_ex[i_ex];";
        o << " }\n";
    }

    o << "\n    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) X[i_ex] = AC_ex[i_ex];\n";
    return o.str();
}

// --- Композиция -------------------------------------------------------------

static const char* const kCompPrefix = "Comp(";

std::string make_composition_name(const std::string& base,
                                  const std::vector<std::string>& gammas) {
    std::string s = kCompPrefix + base + "|";
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
    spec.base = comp_trim(inner.substr(0, bar));
    if (spec.base.empty()) return fail("empty base scheme name");
    // Обёртка над обёрткой отрезает рекурсию в резолвере, как и у Extr.
    if (spec.base.compare(0, pref.size(), pref) == 0 ||
        spec.base.compare(0, 5, "Extr(") == 0)
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
    if (K < kCompMinStages) return fail("at least 2 stages are needed");
    if (K > kCompMaxStages) return fail("at most 9 stages are supported");

    if (out) *out = spec;
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

    // Та же лямбда, что у wrap_extrapolation, и ровно по той же причине:
    // параметр h затеняет макрошаг, поэтому база не правится ни в одном символе.
    o << "    auto step_co = [&](const numb h) {\n";
    {
        size_t pos = 0;
        while (pos < base_body.size()) {
            size_t eol = base_body.find('\n', pos);
            if (eol == std::string::npos) eol = base_body.size();
            const std::string line = base_body.substr(pos, eol - pos);
            if (!line.empty()) o << "    " << line;
            o << "\n";
            pos = eol + 1;
        }
    }
    o << "    };\n\n";

    for (int k = 0; k < K; ++k)
        o << "    step_co(h * (" << gc[(size_t)k] << "));\n";

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