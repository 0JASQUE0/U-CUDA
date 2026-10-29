// Реализация парсера (обычный синтаксис + LaTeX) и кодгена схем.
#include "codegen.hpp"
#include "sparse_lu.h"
#include <cstdlib>
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
#include <functional>

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
            "pow","atan2","fmod","floor","ceil",
            "copysign",  // производная fabs (pn_diff): её печатает pn_print, и текст читается обратно
            "fmin", "fmax", "min", "max"   // min/max разбираются в fmin/fmax (так их печатает emit)
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
        const std::vector<std::string>* extra_funcs = nullptr;   // только превью LaTeX: вызовы функций пользователя
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
            PN n = unary();
            for (;;) {
                if (latex) {
                    ws();
                    if (i + 5 <= s.size() && s.compare(i, 5, "\\cdot") == 0) { i += 5; auto t = mk(Node::Mul); t->a = n; t->b = unary(); n = t; continue; }
                }
                char c = peek();
                if (c == '*') { eat('*'); auto t = mk(Node::Mul); t->a = n; t->b = unary(); n = t; }
                else if (c == '/') { eat('/'); auto t = mk(Node::Div); t->a = n; t->b = unary(); n = t; }
                else if (latex && at_primary_start()) { auto t = mk(Node::Mul); t->a = n; t->b = unary(); n = t; }
                else break;
            }
            return n;
        }
        // Унарный минус слабее степени, как в математике: -x^2 = -(x^2),
        // 2*-x^2 = 2*(-(x^2)). Раньше основание степени бралось через unary(),
        // и -x^2 молча превращался в (-x)^2.
        PN unary() {
            char c = peek();
            if (c == '-') { eat('-'); auto t = mk(Node::Neg); t->a = unary(); return t; }
            if (c == '+') { eat('+'); return unary(); }
            return powr();
        }
        PN powr() {
            PN base = primary();
            if (peek() == '^') {
                eat('^'); auto t = mk(Node::Pow); t->a = base;
                t->b = exponent(); return t;
            }
            return base;
        }
        // Показатель: в LaTeX один токен или группа {..}, в обычном синтаксисе —
        // снова powr(), поэтому ^ правоассоциативна. Знак перед показателем
        // разрешён в обоих режимах: x^-1, x^-y^2 = x^(-(y^2)).
        PN exponent() {
            char c = peek();
            if (c == '-') { eat('-'); auto t = mk(Node::Neg); t->a = exponent(); return t; }
            if (c == '+') { eat('+'); return exponent(); }
            return latex ? braced_arg() : powr();
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
                    if (!known_funcs().count(id)
                        && !(extra_funcs && std::find(extra_funcs->begin(), extra_funcs->end(), id) != extra_funcs->end()))
                        throw std::runtime_error("unknown function: " + id);
                    eat('('); auto n = mk(Node::Call);
                    n->name = id == "min" ? std::string("fmin") : id == "max" ? std::string("fmax") : id;
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
    // "u / c" с числовым c печатается как "u * ((numb)1 / c)": обратная сворачивается
    // компилятором в литерал (double/float), а деление в FP64 — это ~10 инструкций на
    // конвейере, который у ядер и так узкое место. На dd/qd-ветке (numb = dd/qd)
    // (numb)1 / c считается в рантайме в полной точности — литерал 1/c в double срезал
    // бы её. Расхождение с честным делением — 1 ulp. "c1 / c2" не трогаем: это и так
    // константа, а правильно округлённое c1/c2 точнее c1 * (1/c2). Байткод CPU делает то же.
    bool div_by_reciprocal(const PN& n) {
        if (n->kind != Node::Div || n->b->kind != Node::Num || n->a->kind == Node::Num) return false;
        const double c = std::fabs(n->b->num);
        return std::isfinite(c) && c >= 1e-30 && c <= 1e30;
    }
    void emit(const PN& n, const NameMap& nm, std::ostream& o);
    void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) {
        bool paren = prec(n->kind) < pp;
        if (paren) { o << "("; emit(n, nm, o); o << ")"; }
        else { emit(n, nm, o); }
    }
    void emit(const PN& n, const NameMap& nm, std::ostream& o) {
        switch (n->kind) {
        // Литерал всегда через (numb): голый "0.5" — double, и при numb = float всё
        // выражение уходило в FP64 (F2F + DFMA). Отрицательный — в скобках, чтобы
        // остаться атомом. Текст "(numb)0.0" сравнивается в jac_emit (Am = I - hc*J).
        case Node::Num:
            if (std::signbit(n->num)) o << "((numb)" << fmtnum(n->num) << ")";
            else                      o << "(numb)" << fmtnum(n->num);
            break;
        case Node::Sym:o << nm.resolve(n->name); break;
        case Node::Neg:
            o << "-";
            // "- -x" и "-(-x)^2" дают Neg(Neg(..)); без скобок вышло бы "--X[0]" —
            // в C это декремент состояния. То же для отрицательного литерала.
            if (n->a->kind == Node::Neg || (n->a->kind == Node::Num && std::signbit(n->a->num))) {
                o << "("; emit(n->a, nm, o); o << ")";
            }
            else emit_child(n->a, nm, o, prec(Node::Neg));
            break;
        case Node::Add:emit_child(n->a, nm, o, 1); o << " + "; emit_child(n->b, nm, o, 1); break;
        case Node::Sub:emit_child(n->a, nm, o, 1); o << " - "; emit_child(n->b, nm, o, 2); break;
        case Node::Mul:emit_child(n->a, nm, o, 2); o << " * "; emit_child(n->b, nm, o, 2); break;
        case Node::Div:
            emit_child(n->a, nm, o, 2);
            if (div_by_reciprocal(n)) { o << " * ((numb)1 / "; emit_child(n->b, nm, o, 3); o << ")"; }
            else                      { o << " / ";            emit_child(n->b, nm, o, 3); }
            break;
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
    // tname — во что печатать явное время t (время стадии: "ucuda_t", "ucuda_ts",
    // "(ucuda_t + (numb)0.5 * h)"). Пусто — t не определено, как раньше: схема,
    // которая не печатает время стадий, на t упадёт с "unknown symbol", а не
    // молча посчитает все стадии в один момент. Переменную или параметр с именем
    // t не перекрывает.
    NameMap build_namemap(const System& s, const std::string& st, const std::string& tname = {}) {
        NameMap nm;
        for (size_t i = 0; i < s.vars.size(); ++i) nm.m[s.vars[i]] = st + "[" + std::to_string(i) + "]";
        for (size_t j = 0; j < s.params.size(); ++j) nm.m[s.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
        // константы маппятся сами на себя
        for (const auto& c : math_constants()) nm.m[c] = c;
        if (!tname.empty() && !nm.m.count("t")) nm.m["t"] = tname;
        // Уровень B: z — решение Ньютона текущего вычисления f (ucuda_zb_v), не элемент массива стадии.
        for (size_t k = 0; k < s.zb_names.size(); ++k) nm.m[s.zb_names[k]] = "ucuda_zb_v[" + std::to_string(k) + "]";
        return nm;
    }
    std::vector<std::string> rhs_over(const System& s, const std::string& st, const std::string& tname = {}) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        NameMap nm = build_namemap(s, st, tname); std::vector<std::string> out;
        for (size_t i = 0; i < s.rhs.size(); ++i) {
            Parser p(s.rhs[i], s.latex); PN ast = p.parse();
            std::ostringstream o;
            // Уровень B: f над массивом стадии st — сначала z = phi(st) (ucuda_zb, запятая задаёт порядок;
            // повтор с тем же x и t берёт кэш), потом само выражение.
            if (!s.zb_g.empty()) o << "(ucuda_zb(" << st << ", " << (tname.empty() ? "(numb)0" : tname) << "), (";
            emit(ast, nm, o);
            if (!s.zb_g.empty()) o << "))";
            out.push_back(o.str());
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
    // "CD FP": число итераций из System (UI), не меньше одной. Сверху
    // ограничено, чтобы опечатка не развернула тело на мегабайты.
    // Само число итераций приходит в рантайме слотом a[M+1] (cd_fp_slot); это
    // значение — только оценка цены шага для scheme_rhs_per_step.
    int cd_fp_iters_of(const System& s) {
        return s.cd_fp_iters < 1 ? 1 : s.cd_fp_iters;
    }

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
    // The rule itself lives in cd_find_complex_unsafe: SystemEvaluator asks the same
    // question for the CPU path, which never goes through codegen.
    const std::string* cd_find_complex_unsafe(const PN& n) {
        if (!n) return nullptr;
        if (n->kind == Node::Call && (n->name == "fmod" || n->name == "atan2" || n->name == "fmin" || n->name == "fmax"))
            return &n->name;
        if (const std::string* f = cd_find_complex_unsafe(n->a)) return f;
        if (const std::string* f = cd_find_complex_unsafe(n->b)) return f;
        for (const auto& c : n->args)
            if (const std::string* f = cd_find_complex_unsafe(c)) return f;
        return nullptr;
    }
    void cd_check_complex_safe(const PN& n) {
        if (const std::string* f = cd_find_complex_unsafe(n))
            throw std::runtime_error("Complex CD: function " + *f +
                " is not defined in complex arithmetic - pick another scheme");
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
        if (f == "fmin" || f == "fmax") {
            // ветвь выбирает знак u - w: s = copysign(1, u - w); fmin = u при s < 0
            if (n->args.size() != 2) throw std::runtime_error("jacobian: " + f + " needs 2 args");
            const PN& u = n->args[0]; const PN& w = n->args[1];
            const PN du = pn_diff(u, v), dw = pn_diff(w, v);
            if (pn_is_zero(du) && pn_is_zero(dw)) return pn_num(0);
            const PN s = pn_call2("copysign", pn_num(1), pn_sub(u, w));
            const PN pu = pn_mul(pn_num(0.5), f == "fmin" ? pn_sub(pn_num(1), s) : pn_add(pn_num(1), s));
            const PN pw = pn_mul(pn_num(0.5), f == "fmin" ? pn_add(pn_num(1), s) : pn_sub(pn_num(1), s));
            if (pn_is_zero(dw)) return pn_mul(pu, du);
            if (pn_is_zero(du)) return pn_mul(pw, dw);
            return pn_add(pn_mul(pu, du), pn_mul(pw, dw));
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
    std::vector<std::string> jac_over(const System& s, const std::string& st, const std::string& tname = {}) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        NameMap nm = build_namemap(s, st, tname);
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
    // Явное время (этап 2): у неавтономной системы шаг получает t_n аргументом
    // kStepTimeArg, а схема печатает время каждой стадии. У автономной tname
    // пуст, и текст тела прежний до байта.
    std::string time_name(const System& s, const char* stage_expr = nullptr) {
        if (!system_uses_time(s)) return {};
        return stage_expr ? std::string(stage_expr) : std::string(kStepTimeArg);
    }
    std::string scheme_euler(const System& s) {
        int N = s.vars.size(); auto f = rhs_over(s, "X", time_name(s)); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }
    // Эйлер-Кромер обновляет X по месту, все уравнения — в момент t_n.
    std::string scheme_euler_cromer(const System& s) {
        auto f = rhs_over(s, "X", time_name(s)); std::ostringstream o;
        for (size_t i = 0; i < f.size(); ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n"; return o.str();
    }
    std::string scheme_midpoint(const System& s) {
        int N = s.vars.size();
        auto f0 = rhs_over(s, "X", time_name(s));
        auto f1 = rhs_over(s, "X1", time_name(s, "(ucuda_t + (numb)0.5 * h)")); std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + (numb)0.5 * h * (" << f0[i] << ");\n";
        for (int i = 0; i < N; ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f1[i] << ");\n"; return o.str();
    }
    std::string scheme_rk4(const System& s) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    numb k[" << N << "][4];\n    int N = " << N << ";\n    int i, j;\n";
        o << "    for (i = 0; i < N; i++) {\n        X1[i] = X[i];\n    }\n";
        o << "    for (j = 0; j < 4; j++) {\n";
        // Стадии в моменты t, t + h/2, t + h/2, t + h.
        if (!tn.empty())
            o << "        const numb ucuda_ts = ucuda_t + (j == 0 ? (numb)0 : (j == 3 ? h : (numb)0.5 * h));\n";
        for (int i = 0; i < N; ++i)o << "        k[" << i << "][j] = (" << k[i] << ");\n";
        o << "        if (j == 3) {\n            for (i = 0; i < N; i++) {\n            X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) * ((numb)1 / (numb)6);\n            }\n        }\n";
        o << "        else if (j == 2) {\n            for (i = 0; i < N; i++) {\n            X1[i] = X[i] + h * k[i][j];\n            }\n        }\n";
        o << "        else {\n            for (i = 0; i < N; i++) {\n                X1[i] = X[i] + (numb)0.5 * h * k[i][j];\n            }\n        }\n    }\n"; return o.str();
    }
    // Dormand-Prince 8(7), 13-стадийный. Матрицы M[13][12] и B[2][13] — static
    // const, компилятор положит в constant memory. Считаем y (8-й, b[0]) и z
    // (7-й, b[1]) — z пока не используется, но оставлен под будущий адаптивный
    // шаг по |y - z|. На каждый step X := y.
    //
    // Коэффициент печатается суммой "(numb)(c0) + (numb)(c1) + ...": в double
    // она сворачивается при компиляции в c0, в dd/qd CPU-ветки вкладки Order
    // складывается точно (two_sum) и несёт 32/62 знака коэффициента.
    std::string md_literal(const MultiDoubleCoef& r) {
        if (r.c[0] == 0.0) return "0";
        std::string t;
        for (int m = 0; m < 4 && r.c[m] != 0.0; ++m) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", r.c[m]);
            t += (m ? " + (numb)(" : "(numb)(") + std::string(buf) + ")";
        }
        return t;
    }
    // Квалификатор таблиц коэффициентов. При numb = float хвосты ~1e-51 ниже денормалей float,
    // NVRTC перестаёт считать инициализатор константным, а static __device__ с динамической
    // инициализацией запрещён — поэтому там таблица обычная локальная. Хвосты остаются: их
    // складывают dd/qd CPU-ветки вкладки Order. При double текст прежний.
    const char* md_table_qual() {
        return sizeof(numb) < sizeof(double) ? "    const numb " : "    static const numb ";
    }
    // "static const numb <decl> = {...};" — rows x cols из таблицы с шагом
    // строки ld (ld > cols — берётся левый верхний угол).
    void md_table(std::ostringstream& o, const char* decl, const MultiDoubleCoef* v,
                  int rows, int cols, int ld) {
        o << md_table_qual() << decl << " = {\n";
        for (int i = 0; i < rows; ++i) {
            o << "        {";
            for (int j = 0; j < cols; ++j) o << (j ? ", " : "") << md_literal(v[i * ld + j]);
            o << "}" << (i + 1 < rows ? "," : "") << "\n";
        }
        o << "    };\n";
    }

    // Узлы c_i = sum_j a_ij явного РК (S стадий, строка i с шагом ld) — таблица Cs[S]
    // для времени стадии ucuda_ts = ucuda_t + Cs[i]*h. Печатается только у
    // неавтономной системы. Сумма — в long double по всем частям коэффициента:
    // узлы у этих таблиц — простые дроби, и ulp-ошибка суммы на порядок не влияет.
    void stage_time_table(std::ostringstream& o, const MultiDoubleCoef* A, int S, int ld) {
        o << md_table_qual() << "Cs[" << S << "] = { ";
        for (int i = 0; i < S; ++i) {
            long double c = 0;
            for (int j = 0; j < i && j < ld; ++j)
                for (int m = 0; m < 4; ++m) c += (long double)A[i * ld + j].c[m];
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", (double)c);
            o << (i ? ", " : "") << "(numb)(" << buf << ")";
        }
        o << " };\n";
    }

    std::string scheme_dopri78(const System& s, bool legacy = false) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        md_table(o, "M[13][12]", legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12, 12);
        md_table(o, "B[2][13]",  legacy ? &kDopri78LegacyB[0][0] : &kDopri78B[0][0], 2, 13, 13);
        if (!tn.empty()) stage_time_table(o, legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb y[" << N << "];\n";
        o << "    numb z[" << N << "];\n";
        o << "    numb k[" << N << "][13];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < 13; ++i) {\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + Cs[i] * h;\n";
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

    // Явный РК с постоянным шагом по таблице Бутчера: RK45 (S = 6) и DOP853
    // (S = 12). A — строки стадий 0..S-1 с шагом ld (строка 0 нулевая,
    // печатаются столбцы 0..S-2), b — S весов старшего решения. Форма кода
    // та же, что у DOPRI78: стадии в цикле, таблицы static const. Оценщики и
    // стадия FSAL здесь не нужны — их считает адаптивный вариант.
    std::string scheme_erk(const System& s, const MultiDoubleCoef* A, int ld,
                           const MultiDoubleCoef* b, int S) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        const std::string sm1 = std::to_string(S - 1), ss = std::to_string(S);
        md_table(o, ("M[" + ss + "][" + sm1 + "]").c_str(), A, S, S - 1, ld);
        md_table(o, ("B[1][" + ss + "]").c_str(), b, 1, S, S);
        if (!tn.empty()) stage_time_table(o, A, S, ld);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb k[" << N << "][" << S << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < " << S << "; ++i) {\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + Cs[i] * h;\n";
        for (int v = 0; v < N; ++v)
            o << "        k[" << v << "][i] = (" << k[v] << ");\n";
        o << "        if (i != " << S - 1 << ") {\n";
        o << "            for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "            for (j = 0; j < i + 1; ++j)\n";
        o << "                for (l = 0; l < N; ++l)\n";
        o << "                    X2[l] += M[i + 1][j] * k[l][j];\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X1[l] = X[l] + h * X2[l];\n";
        o << "        }\n";
        o << "    }\n";

        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < " << S << "; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[0][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        X[l] = X[l] + h * X2[l];\n";
        return o.str();
    }

    // ---------- Адаптивный шаг: тела функций (см. AdaptiveCode в codegen.hpp) ----------
    //
    // Встроенная пара явного РК. A — стадии 0..S-1 с шагом строки ld, b — S весов
    // старшего решения. Оценки ошибки — строки весов lo[m] = b - b^_m длины lc по стадиям
    // 0..lc-1, где стадия S (если lc > S) — f(Y). Пишется сразу разность
    // E_m = Y - Yhat_m = h*sum(lo_m*K), как у Хайрера и scipy, а не само оценочное решение:
    // вычитать из Y близкое к нему Yhat значило бы терять в E ошибку округления eps*|y|,
    // а так она eps*h*|f|.
    // Стадия 0 не считается: это F0 = f(X), принятая с прошлого шага (FSAL у
    // RK45 и DOP853; у DOPRI78 первая стадия тоже f(X)). Суммы идут в том же
    // порядке, что у тела постоянного шага, поэтому Y побитово совпадает с ним.
    // Стадии развёрнуты, и в суммы печатаются только ненулевые коэффициенты: умножение
    // на 0 компилятор в FP64 выбросить не вправе (inf, NaN, -0), а ядра упираются в FP64.
    // При конечных стадиях результат тот же до бита: x + 0*w = x, порядок остальных
    // слагаемых прежний (у DOP853 нулей 16 из 66 в A, 4 из 12 в b, по 5 из 13 в оценках —
    // ~30% умножений комбинаций стадий).
    struct EmbSpec {
        const MultiDoubleCoef* A; int ld;
        const MultiDoubleCoef* b; int S;
        const MultiDoubleCoef* lo[2]; int nlo; int lc;
    };

    std::string adaptive_emb_body(const System& s, const EmbSpec& e) {
        const int N = (int)s.vars.size(), S = e.S;
        // Явное время: стадия i — в момент ucuda_t + Cs[i]*h, f(Y) — в ucuda_t + h.
        const std::string tn = time_name(s, "ucuda_ts");
        const auto kx = rhs_over(s, "X1", tn), ky = rhs_over(s, "Y", time_name(s, "(ucuda_t + h)"));
        std::ostringstream o;
        md_table(o, ("M[" + std::to_string(S) + "][" + std::to_string(S - 1) + "]").c_str(), e.A, S, S - 1, e.ld);
        md_table(o, ("B[1][" + std::to_string(S) + "]").c_str(), e.b, 1, S, S);
        if (!tn.empty()) stage_time_table(o, e.A, S, e.ld);
        o << md_table_qual() << "L[" << e.nlo << "][" << e.lc << "] = {\n";
        for (int m = 0; m < e.nlo; ++m) {
            o << "        {";
            for (int i = 0; i < e.lc; ++i) o << (i ? ", " : "") << md_literal(e.lo[m][i]);
            o << "}" << (m + 1 < e.nlo ? "," : "") << "\n";
        }
        o << "    };\n";
        auto nz = [](const MultiDoubleCoef& c) { return c.c[0] != 0.0; };
        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int l;\n";
        if (!tn.empty()) o << "    numb ucuda_ts;\n";
        o << "    for (l = 0; l < N; ++l) W[l] = F0[l];\n";
        for (int i = 1; i < S; ++i) {
            o << "    for (l = 0; l < N; ++l) {\n";
            o << "        X2[l] = 0;\n";
            for (int j = 0; j < i; ++j)
                if (nz(e.A[i * e.ld + j]))
                    o << "        X2[l] += M[" << i << "][" << j << "] * W[" << j << " * N + l];\n";
            o << "        X1[l] = X[l] + h * X2[l];\n";
            o << "    }\n";
            if (!tn.empty()) o << "    ucuda_ts = ucuda_t + Cs[" << i << "] * h;\n";
            for (int v = 0; v < N; ++v)
                o << "    W[" << i << " * N + " << v << "] = (" << kx[v] << ");\n";
        }
        o << "    for (l = 0; l < N; ++l) {\n";
        o << "        X2[l] = 0;\n";
        for (int i = 0; i < S; ++i)
            if (nz(e.b[i])) o << "        X2[l] += B[0][" << i << "] * W[" << i << " * N + l];\n";
        o << "        Y[l] = X[l] + h * X2[l];\n";
        o << "    }\n";
        for (int v = 0; v < N; ++v)
            o << "    F1[" << v << "] = (" << ky[v] << ");\n";
        o << "    for (l = 0; l < N; ++l) W[" << S << " * N + l] = F1[l];\n";
        for (int m = 0; m < e.nlo; ++m) {
            o << "    for (l = 0; l < N; ++l) {\n";
            o << "        X2[l] = 0;\n";
            for (int i = 0; i < e.lc; ++i)
                if (nz(e.lo[m][i])) o << "        X2[l] += L[" << m << "][" << i << "] * W[" << i << " * N + l];\n";
            o << "        E[" << m << " * N + l] = h * X2[l];\n";
            o << "    }\n";
        }
        return o.str();
    }

    std::string adaptive_rhs_body(const System& s) {
        const auto k = rhs_over(s, "X", time_name(s));   // f в момент аргумента ucuda_t
        std::ostringstream o;
        for (size_t v = 0; v < k.size(); ++v) o << "    F[" << v << "] = (" << k[v] << ");\n";
        return o.str();
    }

    // Плотный выход RK45 — полином Шампайна 4-й степени, форма scipy:
    // y(th) = X + h * sum_j Q_j th^(j+1), Q = K^T P. D = [X, hQ_0..hQ_3].
    std::string rk45_dprep_body(const System& s) {
        const int N = (int)s.vars.size();
        std::ostringstream o;
        md_table(o, "P[7][4]", &kRK45P[0][0], 7, 4, 4);
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";
        o << "    (void)Y; (void)F0; (void)F1; (void)a;\n";
        o << "    for (l = 0; l < N; ++l) D[l] = X[l];\n";
        o << "    for (j = 0; j < 4; ++j)\n";
        o << "        for (l = 0; l < N; ++l) {\n";
        o << "            numb acc = 0;\n";
        o << "            for (i = 0; i < 7; ++i) acc += W[i * N + l] * P[i][j];\n";
        o << "            D[(1 + j) * N + l] = h * acc;\n";
        o << "        }\n";
        return o.str();
    }
    std::string rk45_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l)\n";
        o << "        Yo[l] = D[l] + th * (D[N + l] + th * (D[2 * N + l] + th * (D[3 * N + l] + th * D[4 * N + l])));\n";
        return o.str();
    }

    // Плотный выход DOP853 — полином 7-й степени Хайрера, форма scipy
    // (Dop853DenseOutput): три доп. стадии 13..15, затем D = [X, F0..F6].
    std::string dop853_dprep_body(const System& s) {
        const int N = (int)s.vars.size();
        const std::string tn = time_name(s, "ucuda_ts");
        const auto kx = rhs_over(s, "X1", tn);
        std::ostringstream o;
        md_table(o, "AE[3][15]", &kDop853AExt[0][0], 3, 15, 15);
        md_table(o, "DD[4][16]", &kDop853D[0][0], 4, 16, 16);
        if (!tn.empty()) {
            // Узлы доп. стадий 13..15 — суммы строк AE (у стадии 13 + i — первые 13 + i).
            o << md_table_qual() << "CE[3] = { ";
            for (int i = 0; i < 3; ++i) {
                long double c = 0;
                for (int j = 0; j < 13 + i; ++j)
                    for (int m = 0; m < 4; ++m) c += (long double)kDop853AExt[i][j].c[m];
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", (double)c);
                o << (i ? ", " : "") << "(numb)(" << buf << ")";
            }
            o << " };\n";
        }
        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l, r;\n";
        o << "    for (i = 0; i < 3; ++i) {\n";
        o << "        for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "        for (j = 0; j < 13 + i; ++j)\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X2[l] += AE[i][j] * W[j * N + l];\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X1[l] = X[l] + h * X2[l];\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + CE[i] * h;\n";
        for (int v = 0; v < N; ++v)
            o << "        W[(13 + i) * N + " << v << "] = (" << kx[v] << ");\n";
        o << "    }\n";
        o << "    for (l = 0; l < N; ++l) {\n";
        o << "        numb dy = Y[l] - X[l];\n";
        o << "        D[l] = X[l];\n";
        o << "        D[N + l] = dy;\n";
        o << "        D[2 * N + l] = h * F0[l] - dy;\n";
        o << "        D[3 * N + l] = 2 * dy - h * (F1[l] + F0[l]);\n";
        o << "    }\n";
        o << "    for (r = 0; r < 4; ++r)\n";
        o << "        for (l = 0; l < N; ++l) {\n";
        o << "            numb acc = 0;\n";
        o << "            for (i = 0; i < 16; ++i) acc += DD[r][i] * W[i * N + l];\n";
        o << "            D[(4 + r) * N + l] = h * acc;\n";
        o << "        }\n";
        return o.str();
    }
    std::string dop853_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    numb u = 1 - th;\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        numb y = D[7 * N + l] * th;\n";
        o << "        y = (y + D[6 * N + l]) * u;\n";
        o << "        y = (y + D[5 * N + l]) * th;\n";
        o << "        y = (y + D[4 * N + l]) * u;\n";
        o << "        y = (y + D[3 * N + l]) * th;\n";
        o << "        y = (y + D[2 * N + l]) * u;\n";
        o << "        y = (y + D[N + l]) * th;\n";
        o << "        Yo[l] = y + D[l];\n";
        o << "    }\n";
        return o.str();
    }

    // Кубический Эрмит по концам шага (DOPRI78): D = [X, Y, hF0, hF1],
    // y(th) = X + th*hF0 + th^2*(3d - 2hF0 - hF1) + th^3*(hF0 + hF1 - 2d), d = Y - X.
    std::string hermite_dprep_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    (void)a; (void)W;\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        D[l] = X[l];\n";
        o << "        D[N + l] = Y[l];\n";
        o << "        D[2 * N + l] = h * F0[l];\n";
        o << "        D[3 * N + l] = h * F1[l];\n";
        o << "    }\n";
        return o.str();
    }
    std::string hermite_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        numb d = D[N + l] - D[l], f0 = D[2 * N + l], f1 = D[3 * N + l];\n";
        o << "        Yo[l] = D[l] + th * (f0 + th * ((3 * d - 2 * f0 - f1) + th * (f0 + f1 - 2 * d)));\n";
        o << "    }\n";
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
    // the root, so 2-3 iterations are typical. The "full" setting (a[M+4], see
    // NewtonSettings in codegen.hpp) picks the variant:
    //   false -- Jacobian and LU built once per step at the predictor and REUSED
    //            while the correction keeps shrinking; rebuilt as soon as it does
    //            not (modified Newton with a refresh). The refresh is not a
    //            refinement: a permanently frozen Jacobian diverges outright on a
    //            stiff nonlinear system - measured on Van der Pol at mu = 100,
    //            where h >= 0.01 blew up without it and reproduces the full-Newton
    //            answer with it, at about half the Jacobian evaluations.
    //   true  -- rebuilt every iteration (full Newton): quadratic convergence and
    //            the most robust at large h, at roughly k times the cost.
    // Iteration stops on ||dX||^2 < tol^2 (in float also at the precision floor, see
    // below) or after max_iters passes. tol, max_iters and full are read from a[M+2..M+4]
    // at run time, like s and the CD FP n: changing them does not rebuild the module,
    // and the step body reads them every call.
    //
    // The Gauss/LU solver is printed INTO the body instead of being shared from a
    // header, on purpose: the phase-portrait path hand-assembles its NVRTC source in
    // nvrtc_engine.cpp and pulls in configCUDA.h only when the body mentions ucmplx,
    // so a header helper would simply not be found there. DOPRI78 inlines its Butcher
    // tables for the same reason. LU and solve are printed unrolled, so numb Am[N*N]
    // lives in registers (N^2 of them per thread).
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

    // ДАУ (этап 4 плана DAE): dae_nx >= 0 — система в форме DaeSystem::as_codegen_system,
    // X = [x (dae_nx штук); z], rhs = [f; g]. Тот же Ньютон по всему X_{n+1}, но строки
    // связей — g(X_{n+1}, t_{n+1}) = 0 (в Am — dg/dX), а не X - X_n - hc*f. Предиктор:
    // X_n целиком (x и z). Явный Эйлер по x на жёсткой ДАУ уводил старт Ньютона далеко от
    // корня: PUT-нейрон (library/DAE-1 PUT burst neuron) при h = 5e-4 на фронте спайка
    // получал Vn ~ 1.4 В, токи ~e^40, и 25 итераций полного Ньютона не сходились; с X_n
    // тот же шаг сходится. ImplicitKind::Euler — BDF1 (неявный Эйлер на ДАУ),
    // dae_trap — трапеции: hc = h/2 и к невязке строк x добавлено -hc*f(X_n, t_n).
    // Время стадий: f(X_n) в ucuda_t, невязка и якобиан в ucuda_t + h. Без dae_nx текст
    // прежний до байта.
    // dae_hc — своё выражение для hc (стадия BDF2 у TR-BDF2), иначе h или h/2.
    // ---------- Этап 6: разреженная LU по структуре (план — sparse_lu.h) ----------
    // Матрица — массив Asp[nnz] с постоянными индексами (клетки портрета после fill-in), без перестановок
    // во время счёта: ведущие элементы выбраны планом по структуре. Множители L ложатся на место
    // A[r][pc] (их уже никто не перезапишет: столбец pc после шага закрыт).
    // Гибрид (так у ДАУ): по плану исключаются только строки diag_first по своим диагоналям (x-строки
    // I - h J, диагональ ~1); остаток — блок Rb x Cb (дополнение Шура по z) — плотная LU с выбором
    // главного во время счёта. Статический выбор в блоке связей промахивался: строки индекса 3 после
    // исключения x масштабированы как h^2 (Andrews squeezer: порог не проходил ни разу). Блок лежит в
    // Asp[0 .. nb*nb) по строкам, остальные клетки — после.
    struct SparseLu {
        EliminationPlan plan;
        std::map<std::pair<int, int>, int> id;
        std::set<std::pair<int, int>> orig;
        int nnz = 0, n1 = 0, nb = 0;
        double ops1 = 0;   // умножений-сложений первой фазы
        std::vector<int> Rb, Cb;
        std::string at(int r, int c) const { return "Asp[" + std::to_string(id.at({ r, c })) + "]"; }
    };
    bool sparse_lu_build(const std::vector<MatEntry>& pat, int n, const std::vector<char>& df, SparseLu& S) {
        // UCUDA_DAE_DENSE_LU=1 — плотная LU, как до этапа 6 (сравнение скорости и результатов).
        static const bool dense_forced = [] { const char* e = std::getenv("UCUDA_DAE_DENSE_LU"); return e && *e == '1'; }();
        if (dense_forced) return false;
        S = SparseLu{};
        for (const MatEntry& e : pat) S.orig.insert({ e.row, e.col });
        for (int i = 0; i < n; ++i) {
            if (df[(size_t)i]) ++S.n1;
            else { S.Rb.push_back(i); S.Cb.push_back(i); }
        }
        S.nb = (int)S.Rb.size();
        std::vector<MatEntry> full = pat;
        for (int r : S.Rb) for (int c : S.Cb) if (!S.orig.count({ r, c })) full.push_back(MatEntry{ r, c });
        std::string err;
        if (!plan_elimination(full, n, S.plan, &err, &df)) return false;
        for (int k = 0; k < S.n1; ++k)   // первая фаза — ровно диагонали строк df
            if (S.plan.steps[(size_t)k].pivot_row != S.plan.steps[(size_t)k].pivot_col || !df[(size_t)S.plan.steps[(size_t)k].pivot_row])
                return false;
        // Модель стоимости: умножения-сложения первой фазы (по плану) + nb^3/3 блока против n^3/3 плотной;
        // разреженная — только при выигрыше хотя бы вдвое. И первая фаза не длиннее 4000 операций: это
        // прямолинейный код, и его объём стоит дороже арифметики. Andrews, Radau + GGL: 7884 операции первой
        // фазы + блок 57 x 57 — по модели в 4.6 раза дешевле плотной 99 x 99, на деле в 1.4 раза медленнее
        // (CPU, cl /O2); Radau без GGL (909 операций, блок 39) — в 3.9 раза быстрее.
        {
            for (int k = 0; k < S.n1; ++k)
                S.ops1 += (double)S.plan.steps[(size_t)k].rows.size() * (double)S.plan.steps[(size_t)k].cols.size();
            if (S.ops1 > 4000.0) return false;
            if (2.0 * (S.ops1 + (double)S.nb * S.nb * S.nb / 3.0) > (double)n * n * n / 3.0) return false;
        }
        for (int a = 0; a < S.nb; ++a)
            for (int b = 0; b < S.nb; ++b) S.id[{ S.Rb[(size_t)a], S.Cb[(size_t)b] }] = a * S.nb + b;
        for (const MatEntry& e : S.plan.filled)
            if (!S.id.count({ e.row, e.col })) S.id[{ e.row, e.col }] = (int)S.id.size();
        S.nnz = (int)S.id.size();
        return true;
    }
    // Факторизация. Первая фаза (x-диагонали по плану) проверяет порог: |ведущий| >= 1e-3 * max|столбца|
    // среди исключаемых строк и >= 1e-30; не прошёл — fallback = 1 (вызывающий решает всю матрицу плотной
    // LU). Затем блок — плотная LU с выбором главного (перестановки только от столбца k, как везде) в
    // ucuda_pv; вырожденный — sing = 1. Нужны int ik, ir, ic, ip и numb mx, av, dgn, mlt.
    void sparse_lu_factor(std::ostream& o, const SparseLu& S, const char* ind, const char* fallback) {
        for (int k = 0; k < S.n1; ++k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            o << ind << "if (!" << fallback << ") { dgn = " << S.at(st.pivot_row, st.pivot_col) << "; mx = fabs(dgn) * (numb)1e3;";
            for (int r : st.rows) o << " av = fabs(" << S.at(r, st.pivot_col) << "); if (av > mx) mx = av;";
            o << " if (fabs(dgn) < (numb)1e-30 || fabs(dgn) * (numb)1e3 < mx) " << fallback << " = 1;";
            if (!st.rows.empty()) {
                o << " else {\n";
                for (int r : st.rows) {
                    o << ind << "    mlt = " << S.at(r, st.pivot_col) << " / dgn; " << S.at(r, st.pivot_col) << " = mlt;";
                    for (int c : st.cols) o << " " << S.at(r, c) << " -= mlt * " << S.at(st.pivot_row, c) << ";";
                    o << "\n";
                }
                o << ind << "} }\n";
            } else o << " }\n";
        }
        if (S.nb == 0) return;
        const std::string nb = std::to_string(S.nb);
        o << ind << "if (!" << fallback << ") {   /* dense block " << nb << " x " << nb << ", partial pivoting */\n";
        o << ind << "    for (ik = 0; ik < " << nb << "; ik++) {\n";
        o << ind << "        ip = ik; mx = fabs(Asp[ik * " << nb << " + ik]);\n";
        o << ind << "        for (ir = ik + 1; ir < " << nb << "; ir++) { av = fabs(Asp[ir * " << nb << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n";
        o << ind << "        ucuda_pv[ik] = ip;\n";
        o << ind << "        if (ip != ik) for (ic = ik; ic < " << nb << "; ic++) { mlt = Asp[ik * " << nb << " + ic]; Asp[ik * " << nb
          << " + ic] = Asp[ip * " << nb << " + ic]; Asp[ip * " << nb << " + ic] = mlt; }\n";
        o << ind << "        dgn = Asp[ik * " << nb << " + ik];\n";
        o << ind << "        if (fabs(dgn) < (numb)1e-30) { sing = 1; break; }\n";
        o << ind << "        for (ir = ik + 1; ir < " << nb << "; ir++) {\n";
        o << ind << "            mlt = Asp[ir * " << nb << " + ik] / dgn; Asp[ir * " << nb << " + ik] = mlt;\n";
        o << ind << "            for (ic = ik + 1; ic < " << nb << "; ic++) Asp[ir * " << nb << " + ic] -= mlt * Asp[ik * " << nb << " + ic];\n";
        o << ind << "        }\n";
        o << ind << "    }\n";
        o << ind << "}\n";
    }
    // Объявления гибрида: Asp, перестановки блока, номера строк блока.
    void sparse_lu_decl(std::ostream& o, const SparseLu& S, int nsol) {
        o << "    numb Asp[" << S.nnz << "], Dv[" << nsol << "];   /* sparse LU: " << S.n1 << " pivots by plan ("
          << (long long)S.ops1 << " multiply-adds), dense block " << S.nb << " x " << S.nb << "; dense fallback */\n";
        o << "    int  ucuda_dn = 0, ucuda_pv[" << (S.nb > 0 ? S.nb : 1) << "];\n";
        o << "    const int ucuda_rb[" << (S.nb > 0 ? S.nb : 1) << "] = { ";
        for (int a = 0; a < S.nb; ++a) o << (a ? ", " : "") << S.Rb[(size_t)a];
        if (S.nb == 0) o << "0";
        o << " };\n";
    }
    // Решение: правая часть rhs[] по строкам (портится), ответ sol[] по столбцам.
    void sparse_lu_solve(std::ostream& o, const SparseLu& S, const std::string& rhs, const std::string& sol, const char* ind) {
        for (int k = 0; k < S.n1; ++k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            for (int r : st.rows)
                o << ind << rhs << "[" << r << "] -= " << S.at(r, st.pivot_col) << " * " << rhs << "[" << st.pivot_row << "];\n";
        }
        if (S.nb > 0) {
            const std::string nb = std::to_string(S.nb);
            o << ind << "for (ik = 0; ik < " << nb << "; ik++) {\n";
            o << ind << "    ip = ucuda_pv[ik]; if (ip != ik) { mlt = " << rhs << "[ucuda_rb[ik]]; " << rhs << "[ucuda_rb[ik]] = " << rhs
              << "[ucuda_rb[ip]]; " << rhs << "[ucuda_rb[ip]] = mlt; }\n";
            o << ind << "    for (ir = ik + 1; ir < " << nb << "; ir++) " << rhs << "[ucuda_rb[ir]] -= Asp[ir * " << nb << " + ik] * "
              << rhs << "[ucuda_rb[ik]];\n";
            o << ind << "}\n";
            o << ind << "for (ik = " << S.nb - 1 << "; ik >= 0; ik--) {\n";
            o << ind << "    mlt = " << rhs << "[ucuda_rb[ik]];\n";
            o << ind << "    for (ic = ik + 1; ic < " << nb << "; ic++) mlt -= Asp[ik * " << nb << " + ic] * " << sol << "[ucuda_rb[ic]];\n";
            o << ind << "    " << sol << "[ucuda_rb[ik]] = mlt / Asp[ik * " << nb << " + ik];\n";
            o << ind << "}\n";
        }
        for (int k = S.n1 - 1; k >= 0; --k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            o << ind << sol << "[" << st.pivot_col << "] = (" << rhs << "[" << st.pivot_row << "]";
            for (int c : st.cols) o << " - " << S.at(st.pivot_row, c) << " * " << sol << "[" << c << "]";
            o << ") / " << S.at(st.pivot_row, st.pivot_col) << ";\n";
        }
    }

    // Множитель h^(k-1) переменной i индекса k >= 2 в норме поправки Ньютона ДАУ (как в RADAU5);
    // "" — индекс 0/1 (тогда текст тела прежний до байта).
    std::string dae_norm_w(const System& s, int i) {
        const int k = i < (int)s.dae_index.size() ? s.dae_index[(size_t)i] : 1;
        std::string w;
        for (int p = 1; p < k; ++p) w += "h * ";
        return w;
    }

    std::string scheme_implicit_common(const System& s, ImplicitKind kind, int dae_nx = -1,
                                       bool dae_trap = false, const char* dae_hc = nullptr) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        const bool dae = dae_nx >= 0;

        const bool  cx  = (kind == ImplicitKind::ComplexEuler);
        // Массивы состояния, из которых читают сгенерированные выражения:
        // X/Xn у вещественных схем (как было), Z/Zn у комплексной.
        const char* stv = cx ? "Z"  : "X";
        const char* stn = cx ? "Zn" : "Xn";
        const char* sty = cx ? "ucmplx" : "numb";

        // Время: предиктор в t_n, невязка и якобиан — в момент неизвестной (Euler, ДАУ: t + h;
        // Midpoint: стадия Y в t + h/2). Комплексный Эйлер времени не поддерживает (codegen_scheme
        // откажет раньше). Без t в системе time_name пуст и текст прежний.
        const std::string t_new = cx ? std::string()
            : time_name(s, kind == ImplicitKind::Midpoint ? "(ucuda_t + (numb)0.5 * h)" : "(ucuda_t + h)");
        auto f0 = rhs_over(s, stv, cx ? std::string() : time_name(s));   // predictor, evaluated at the old state
        auto fn = rhs_over(s, stn, t_new);    // residual, evaluated at the Newton iterate
        auto Jn = jac_over(s, stn, t_new);    // Jacobian, same point as the residual

        if (cx) {
            // Те же запреты, что у Complex CD: у atan2/fmod нет комплексного
            // смысла, и шаг с мнимой частью по ним считать нечем (floor/ceil/fmod
            // уже отсёк jac_over). fabs разрешён: |z| и copysign(1, Re u) в
            // якобиане — см. configCUDA.h.
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                cd_check_complex_safe(p.parse());
            }
        }

        // Допуск, предел итераций и режим якобиана — из a[] (см. NewtonSettings в codegen.hpp).
        const int nslot = newton_slot(s);

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
        // Этап 6: ДАУ больше 8 переменных — разреженная LU по структуре (плотная в форме цикла уходит в
        // локальную память и стоит N^3); до 8 — плотная раскатанная, текст прежний.
        SparseLu SL;
        bool sparse = false;
        if (dae && N > 8) {
            std::vector<MatEntry> pat;
            std::vector<char> df((size_t)N, 0);
            for (int i = 0; i < N; ++i) {
                if (i < dae_nx) { df[(size_t)i] = 1; pat.push_back(MatEntry{ i, i }); }
                for (int j = 0; j < N; ++j)
                    if (Jn[(size_t)i * N + j] != "(numb)0.0" && !(i < dae_nx && i == j)) pat.push_back(MatEntry{ i, j });
            }
            sparse = sparse_lu_build(pat, N, df, SL);
        }
        if (sparse) sparse_lu_decl(o, SL, N);
        o << "    " << sty << " Am[" << N * N << "];\n";
        o << "    int  piv[" << N << "];\n";
        if (cx)
            // Переприсваивается между проходами, поэтому не const.
            o << "    ucmplx hc = ucmplx(a[0] * h, h * (numb)" << fmtnum(CIE_IMAG) << ");\n";
        else if (dae_hc)
            o << "    const numb hc = " << dae_hc << ";\n";
        else
            o << "    const numb hc = "
              << (kind == ImplicitKind::Euler && !dae_trap ? "h" : "((numb)0.5 * h)") << ";\n";
        if (dae_trap) {
            // f(X_n, t_n) — постоянное слагаемое невязки трапеций.
            o << "    numb Fd[" << (dae_nx > 0 ? dae_nx : 1) << "];\n";
            for (int i = 0; i < dae_nx; ++i) o << "    Fd[" << i << "] = (" << f0[i] << ");\n";
        }
        o << "    const numb ntol  = a[" << nslot << "];\n";
        o << "    const int  nmax  = (int)a[" << nslot + 1 << "];\n";
        o << "    const int  nfull = (int)a[" << nslot + 2 << "];\n";
        // LU/solve are printed unrolled up to 8 variables; beyond that the unrolled text
        // grows as N^3 and Am (N^2 values) would not fit in registers anyway, so the
        // loop form stays.
        const bool unroll = N <= 8;
        o << "    int it, ip, sing, refresh;\n";
        if (!unroll) {
            o << "    const int  ndim = " << N << ";\n";
            o << "    int ik, ir, ic;\n";
        }
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
        o << (dae ? "\n    /* predictor: X_n */\n" : "\n    /* predictor: explicit Euler */\n");
        for (int i = 0; i < N; ++i) {
            if (dae)
                o << "    " << stn << "[" << i << "] = " << stv << "[" << i << "];\n";
            else
                o << "    " << stn << "[" << i << "] = " << stv << "[" << i
                  << "] + hc * (" << f0[i] << ");\n";
        }

        o << "\n    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < nmax; it++) {\n";

        // One emission of the Jacobian + LU, entered on iteration 0 and again
        // whenever `refresh` is set below. Full Newton sets it every iteration;
        // modified Newton only when the frozen matrix stops contracting.
        o << "        if (refresh) {\n";
        o << "            /* Am = I - hc * J(" << stn << ") */\n";
        if (sparse) {
            for (const auto& kv : SL.id) {
                const int i = kv.first.first, j = kv.first.second;
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Asp[" << kv.second << "] = ";
                if (!SL.orig.count(kv.first)) o << "(numb)0.0;\n";   // fill-in
                else if (i >= dae_nx) o << "(" << d << ");\n";
                else if (d == "(numb)0.0") o << "(numb)1.0;\n";
                else if (i == j) o << "(numb)1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
            o << "            /* sparse LU, pivots chosen by structure (sparse_lu.h); a failed threshold -> dense */\n";
            o << "            ucuda_dn = 0;\n";
            sparse_lu_factor(o, SL, "            ", "ucuda_dn");
            o << "            if (ucuda_dn) {\n";
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Am[" << (i * N + j) << "] = ";
                if (dae && i >= dae_nx) o << (d == "(numb)0.0" ? std::string("0.0") : "(" + d + ")") << ";\n";
                else if (d == "(numb)0.0") o << lit(i == j ? "1.0" : "0.0") << ";\n";
                else if (i == j) o << "(numb)1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
        }
        // LU with partial pivoting, printed unrolled: every index is a compile-time
        // constant and the row swap is a chain of `if (ip == r)` swaps, so Am stays in
        // registers. The loop form indexed Am[ip * ndim + ic] with a runtime pivot and put
        // the whole matrix in local memory (BD 2D Lorenz 256^2, float: 1002 -> 410 ms;
        // double: +2 %, it is FP64-bound). Same operations in the same order -- results
        // are bit-identical to the loop form.
        auto A = [N](int r, int c) { return "Am[" + std::to_string(r * N + c) + "]"; };
        if (!unroll) {
            o << "            /* LU with partial pivoting */\n";
            o << "            for (ik = 0; ik < ndim; ik++) {\n";
            o << "                ip = ik; mx = " << absf << "(Am[ik * ndim + ik]);\n";
            o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
            o << "                    av = " << absf << "(Am[ir * ndim + ik]);\n";
            o << "                    if (av > mx) { mx = av; ip = ir; }\n";
            o << "                }\n";
            o << "                piv[ik] = ip;\n";
            // Переставляются только столбцы от ik (LINPACK dgefa): множители L прежних шагов остаются на
            // своих строках, иначе прямой ход ниже, повторяющий перестановки по одной, берёт чужие
            // множители. Полная перестановка строк давала неверную поправку Ньютона, как только выбор
            // главного срабатывал после первого столбца (строки связей ДАУ) — см. CLAUDE.md, DAE.
            o << "                if (ip != ik)\n";
            o << "                    for (ic = ik; ic < ndim; ic++) {\n";
            o << "                        mlt = Am[ik * ndim + ic];\n";
            o << "                        Am[ik * ndim + ic] = Am[ip * ndim + ic];\n";
            o << "                        Am[ip * ndim + ic] = mlt;\n";
            o << "                    }\n";
            o << "                dgn = Am[ik * ndim + ik];\n";
            o << "                if (" << absf << "(dgn) < (numb)1e-30) { sing = 1; break; }\n";
            o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
            o << "                    mlt = Am[ir * ndim + ik] / dgn;\n";
            o << "                    Am[ir * ndim + ik] = mlt;\n";
            o << "                    for (ic = ik + 1; ic < ndim; ic++)\n";
            o << "                        " << sub_eq("Am[ir * ndim + ic]", "mlt * Am[ik * ndim + ic]") << "\n";
            o << "                }\n";
            o << "            }\n";
        }
        if (unroll) o << "            /* LU with partial pivoting, unrolled */\n";
        for (int k = 0; unroll && k < N; ++k) {
            o << "            if (!sing) {\n";
            o << "                ip = " << k << "; mx = " << absf << "(" << A(k, k) << ");\n";
            for (int r = k + 1; r < N; ++r)
                o << "                av = " << absf << "(" << A(r, k) << "); if (av > mx) { mx = av; ip = " << r << "; }\n";
            o << "                piv[" << k << "] = ip;\n";
            for (int r = k + 1; r < N; ++r) {
                o << "                if (ip == " << r << ") {";
                for (int c = k; c < N; ++c)   // от k, см. форму цикла выше
                    o << " mlt = " << A(k, c) << "; " << A(k, c) << " = " << A(r, c) << "; " << A(r, c) << " = mlt;";
                o << " }\n";
            }
            o << "                dgn = " << A(k, k) << ";\n";
            // Singular pivot: flag and bail instead of dividing. Nudging the pivot
            // would only turn the correction into inf and then nan, killing a
            // trajectory that is otherwise alive; leaving the step on the predictor
            // costs one order locally and nothing globally.
            o << "                if (" << absf << "(dgn) < (numb)1e-30) sing = 1;\n";
            if (k + 1 < N) {
                o << "                else {\n";
                for (int r = k + 1; r < N; ++r) {
                    o << "                    mlt = " << A(r, k) << " / dgn;\n";
                    o << "                    " << A(r, k) << " = mlt;\n";
                    for (int c = k + 1; c < N; ++c)
                        o << "                    " << sub_eq(A(r, c), "mlt * " + A(k, c)) << "\n";
                }
                o << "                }\n";
            }
            o << "            }\n";
        }
        if (sparse) o << "            }   /* ucuda_dn */\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        o << "        if (sing) break;\n";

        o << "        /* residual F(" << stn << ") = " << stn << " - " << stv
          << " - hc * f(" << stn << ") */\n";
        for (int i = 0; i < N; ++i) {
            if (dae && i >= dae_nx) { o << "        Fv[" << i << "] = (" << fn[i] << ");\n"; continue; }
            o << "        Fv[" << i << "] = " << stn << "[" << i << "] - " << stv << "[" << i
              << "] - hc * (" << fn[i] << ")";
            if (dae_trap) o << " - hc * Fd[" << i << "]";
            o << ";\n";
        }
        if (sparse) {
            o << "        if (!ucuda_dn) {\n";
            o << "        /* sparse solve: Dv = A^-1 Fv */\n";
            sparse_lu_solve(o, SL, "Fv", "Dv", "        ");
            for (int i = 0; i < N; ++i) o << "        Fv[" << i << "] = Dv[" << i << "];\n";
            o << "        } else {\n";
        }
        if (!unroll) {
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
        }
        if (sparse) o << "        }   /* ucuda_dn */\n";
        if (unroll) o << "        /* solve Am * dX = Fv in place, unrolled */\n";
        auto F = [](int i) { return "Fv[" + std::to_string(i) + "]"; };
        for (int k = 0; unroll && k < N; ++k) {
            o << "        ip = piv[" << k << "];\n";
            for (int r = k + 1; r < N; ++r)
                o << "        if (ip == " << r << ") { mlt = " << F(k) << "; " << F(k) << " = " << F(r)
                  << "; " << F(r) << " = mlt; }\n";
            for (int r = k + 1; r < N; ++r)
                o << "        " << sub_eq(F(r), A(r, k) + " * " + F(k)) << "\n";
        }
        for (int k = N - 1; unroll && k >= 0; --k) {
            for (int c = k + 1; c < N; ++c)
                o << "        " << sub_eq(F(k), A(k, c) + " * " + F(c)) << "\n";
            o << "        " << div_eq(F(k), A(k, k)) << "\n";
        }
        o << "        nrm = 0.0;\n";
        for (int i = 0; i < N; ++i) {
            if (cx)
                // |dX|^2 в комплексном случае — по модулю, а не по квадрату
                // самого числа: Fv[i]*Fv[i] у комплексного не вещественно.
                o << "        Zn[" << i << "] = Zn[" << i << "] - Fv[" << i << "];"
                  << " nrm += Fv[" << i << "].re * Fv[" << i << "].re + Fv[" << i
                  << "].im * Fv[" << i << "].im;\n";
            else if (dae && !dae_norm_w(s, i).empty())
                // индекс k >= 2: поправка в норме с множителем h^(k-1) (как RADAU5), см. System::dae_index
                o << "        Xn[" << i << "] -= Fv[" << i << "]; mlt = " << dae_norm_w(s, i) << "Fv[" << i
                  << "]; nrm += mlt * mlt;\n";
            else
                o << "        Xn[" << i << "] -= Fv[" << i << "]; nrm += Fv[" << i
                  << "] * Fv[" << i << "];\n";
        }
        o << "        if (nrm < ntol * ntol) break;\n";
        if (dae) {
            // ДАУ: ещё и относительный критерий. На индексе 2-3 обусловленность ~1/h^k, а
            // множитель Лагранжа порядка десятков: шум поправки ~eps*cond*|X| не опускается
            // ниже абсолютного tol, и каждый шаг впустую шёл до max_iters (двойной маятник,
            // h = 1e-3: ||dX||^2 ~ 5e-20 при tol^2 = 1e-20, связи уже до 1e-16).
            o << "        if (nrm < ntol * ntol * (";
            for (int i = 0; i < N; ++i) {
                const std::string w = dae_norm_w(s, i);
                if (w.empty()) o << (i ? " + " : "") << stn << "[" << i << "] * " << stn << "[" << i << "]";
                else o << (i ? " + " : "") << "(" << w << stn << "[" << i << "]) * (" << w << stn << "[" << i << "])";
            }
            o << ")) break;\n";
        }
        // Precision floor, float only: there ||dX|| never drops below ~eps*|X| ~ 1e-6, so
        // at the usual tol = 1e-10 every step ran all max_iters passes and, the correction
        // no longer shrinking, rebuilt the Jacobian and LU on almost every one. Stop at
        // ||dX||^2 < (16 eps)^2 |Xn|^2 instead (BD 2D Lorenz 256^2: 410 -> 68 ms with the
        // unrolled LU). It changes float results at round-off level; in double (and dd/qd
        // on the CPU) the constant condition drops the branch and nothing changes. A plain
        // `if`, not `if constexpr`: cl.exe builds the CPU DLL from this text in C++14.
        o << "        if (sizeof(numb) == 4) {\n";
        o << "            if (nrm < (numb)3.6e-12f * (";
        for (int i = 0; i < N; ++i) {
            if (i) o << " + ";
            if (cx) o << stn << "[" << i << "].re * " << stn << "[" << i << "].re + "
                      << stn << "[" << i << "].im * " << stn << "[" << i << "].im";
            else    o << stn << "[" << i << "] * " << stn << "[" << i << "]";
        }
        o << ")) break;\n";
        o << "        }\n";
        // full Newton refreshes every iteration. Otherwise nrm and prevn are SQUARED
        // norms, so 0.25 is the 0.5 contraction factor. Without this a frozen Jacobian
        // diverges outright on a stiff nonlinear system (measured on Van der Pol,
        // mu = 100, h >= 0.01); refreshing recovers the full-Newton answer at half the
        // Jacobians.
        o << "        if (nfull || nrm > (numb)0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n\n";

        if (kind == ImplicitKind::Euler)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Xn[" << i << "];\n";
        else if (kind == ImplicitKind::Midpoint)
            // Xn holds the stage value Y; recover the endpoint from it.
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = (numb)2.0 * Xn[" << i << "] - X[" << i << "];\n";
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
            o << "\n    hc = ucmplx((1 - a[0]) * h, -h * (numb)" << fmtnum(CIE_IMAG) << ");\n";
            emit_pass();
            // Наружу — только Re, как у Complex CD: всё вокруг
            // calculateDiscreteModel остаётся вещественным.
            o << "\n";
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";
        }

        return o.str();
    }
    // Методы, решающие z из связи (явные со связью, CD), — только индекс 1. Связь, в которой
    // символьно нет ни одной z (dg/dz — нулевая строка), z не определяет: Ньютон дал бы NaN.
    void dae_require_index1(const std::vector<std::string>& J, int N, int nx, const char* what) {
        for (int i = nx; i < N; ++i) {
            bool any = false;
            for (int j = nx; j < N; ++j) any = any || J[(size_t)i * N + j] != "(numb)0.0";
            if (!any)
                throw std::runtime_error(std::string(what) + " need index 1, but constraint " + std::to_string(i - nx + 1)
                    + " contains no algebraic variable (index >= 2): use DAE Radau IIA, BDF1, SDIRK2, TR-BDF2 or a Rosenbrock method");
        }
    }

    // ---------- ДАУ: TR-BDF2 и явные схемы с решением связи на стадиях ----------
    //
    // TR-BDF2 (Bank et al., gamma = 2 - sqrt 2): трапеции до t + gamma*h, затем стадия BDF2
    //   x+ - c1*x_g + c2*x_n - c3*h*f(x+) = 0,  0 = g(x+, z+),
    //   c1 = 1/(g(2-g)), c2 = (1-g)^2/(g(2-g)), c3 = (1-g)/(2-g).
    // L-устойчив, второй порядок. Оба прохода — тот же Ньютон (scheme_implicit_common), каждый в
    // своём блоке; у первого h затенён на gamma*h (время стадии t + gamma*h выходит само),
    // второй решает как неявный Эйлер от комбинации c1*x_g - c2*x_n с hc = c3*h.
    std::string dae_tr_bdf2(const System& s) {
        const int nx = s.dae_nx;
        const double g = 2.0 - std::sqrt(2.0);
        const double c1 = 1.0 / (g * (2.0 - g)), c2 = (1.0 - g) * (1.0 - g) / (g * (2.0 - g)),
                     c3 = (1.0 - g) / (2.0 - g);
        char hc3[64];
        std::snprintf(hc3, sizeof(hc3), "((numb)%.17g * h)", c3);
        std::ostringstream o;
        o << "    /* TR-BDF2: trapezoid to t + gamma*h, then BDF2 stage */\n";
        o << "    const numb ucuda_h0 = h;\n";
        o << "    numb ucuda_xn[" << (nx > 0 ? nx : 1) << "];\n";
        for (int l = 0; l < nx; ++l) o << "    ucuda_xn[" << l << "] = X[" << l << "];\n";
        o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(g) << ";\n"
          << scheme_implicit_common(s, ImplicitKind::Euler, nx, true) << "    }\n";
        for (int l = 0; l < nx; ++l)
            o << "    X[" << l << "] = (numb)" << fmtnum(c1) << " * X[" << l << "] - (numb)" << fmtnum(c2)
              << " * ucuda_xn[" << l << "];\n";
        o << "    {\n" << scheme_implicit_common(s, ImplicitKind::Euler, nx, false, hc3) << "    }\n";
        return o.str();
    }

    // SDIRK2 Александера (gamma = 1 - 1/sqrt 2, L-устойчив, жёстко точен, порядок 2):
    //   Y1 = x_n + gamma h f(Y1),  Y2 = x_n + (1 - gamma) h f(Y1) + gamma h f(Y2),  x_{n+1} = Y2;
    //   связи g(Y_i) = 0 на обеих стадиях. Обе стадии — неявный Эйлер с hc = gamma*h (тот же Ньютон):
    //   первая — от x_n (h затенён на gamma*h, время стадии t + gamma*h), вторая — от
    //   x_n + (1 - gamma)/gamma (Y1 - x_n), т.е. x_n + (1 - gamma) h f(Y1), время t + h.
    std::string dae_sdirk2(const System& s) {
        const int nx = s.dae_nx;
        const double g = 1.0 - std::sqrt(0.5), c = (1.0 - g) / g;
        char hcg[64];
        std::snprintf(hcg, sizeof(hcg), "((numb)%.17g * h)", g);
        std::ostringstream o;
        o << "    /* SDIRK2 (Alexander): two implicit Euler stages with gamma*h */\n";
        o << "    const numb ucuda_h0 = h;\n";
        o << "    numb ucuda_xn[" << (nx > 0 ? nx : 1) << "];\n";
        for (int l = 0; l < nx; ++l) o << "    ucuda_xn[" << l << "] = X[" << l << "];\n";
        o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(g) << ";\n"
          << scheme_implicit_common(s, ImplicitKind::Euler, nx, false) << "    }\n";
        for (int l = 0; l < nx; ++l)
            o << "    X[" << l << "] = ucuda_xn[" << l << "] + (numb)" << fmtnum(c) << " * (X[" << l
              << "] - ucuda_xn[" << l << "]);\n";
        o << "    {\n" << scheme_implicit_common(s, ImplicitKind::Euler, nx, false, hcg) << "    }\n";
        return o.str();
    }

    // Radau IIA, 3 стадии, порядок 5 (жёстко точен, L-устойчив; индексы 1-3, Hairer-Wanner IV.8, VII.4).
    // Неизвестные W = [Y1; Y2; Y3] (по N), x_{n+1} = Y3. Строки x стадии i:
    //   Y_i - x_n - h sum_j a_ij f(Y_j, t + c_j h) = 0,  строки связей: g(Y_i, t + c_i h) = 0
    // (у ДАУ M y' = F с A невырожденной это то же, что sum_j a_ij g(Y_j) = 0). Полный Ньютон по 3N
    // неизвестным: якобианы f и g в каждой стадии, LU 3N x 3N с выбором главного (цикл). Норма поправки —
    // с множителями h^(k-1) по индексу переменной, как у scheme_implicit_common. Предиктор: x — явный
    // Эйлер до t + c_i h, z — прежние. Упрощённый Ньютон RADAU5 (одна J, разложение A^-1) — потом.
    std::string dae_radau5(const System& s) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, M = 3 * N;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const double r6 = std::sqrt(6.0);
        const double A[9] = { (88 - 7 * r6) / 360, (296 - 169 * r6) / 1800, (-2 + 3 * r6) / 225,
                              (296 + 169 * r6) / 1800, (88 + 7 * r6) / 360, (-2 - 3 * r6) / 225,
                              (16 - r6) / 36, (16 + r6) / 36, 1.0 / 9 };
        const double C[3] = { (4 - r6) / 10, (4 + r6) / 10, 1.0 };
        std::ostringstream o;
        o << "    /* Radau IIA (3 stages, order 5), full Newton on " << M << " unknowns */\n";
        o << "    const numb RA[9] = { ";
        for (int k = 0; k < 9; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(A[k]);
        o << " };\n    const numb RC[3] = { ";
        for (int k = 0; k < 3; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(C[k]);
        o << " };\n";
        // Этап 6: матрица 3N x 3N блочная (стадии связаны только через x-строки с ненулевым J) — разреженная
        // LU по структуре; плотная (форма цикла) — только если план не построился.
        SparseLu SL;
        bool sparse = false;
        {
            std::vector<MatEntry> pat;
            std::vector<char> df((size_t)M, 0);
            for (int bi = 0; bi < 3; ++bi)
                for (int l = 0; l < N; ++l) {
                    const int r = bi * N + l;
                    if (l < nx) df[(size_t)r] = 1;
                    for (int bj = 0; bj < 3; ++bj)
                        for (int m = 0; m < N; ++m) {
                            const bool jz = JS[(size_t)l * N + m] == "(numb)0.0";
                            const bool live = l < nx ? ((bi == bj && l == m) || !jz) : (bi == bj && !jz);
                            if (live) pat.push_back(MatEntry{ r, bj * N + m });
                        }
                }
            sparse = sparse_lu_build(pat, M, df, SL);
        }
        o << "    numb Sv[" << N << "], W[" << M << "], Fs[" << M << "], Js[" << 3 * N * N << "], Rv[" << M << "], Wt[" << N << "];\n";
        if (sparse) sparse_lu_decl(o, SL, M);
        o << "    numb Am[" << M * M << "];\n    int  piv[" << M << "];\n";
        o << "    const numb ntol  = a[" << nslot << "];\n";
        o << "    const int  nmax  = (int)a[" << nslot + 1 << "];\n";
        o << "    const int  nfull = (int)a[" << nslot + 2 << "];\n";
        o << "    int it, bi, bj, l, m, ik, ir, ic, ip, sing, refresh;\n";
        o << "    numb mx, av, dgn, mlt, nrm, xx, prevn;\n";
        if (ht) o << "    numb ucuda_ts = ucuda_t;\n";
        // множители нормы h^(k-1)
        for (int i = 0; i < N; ++i) {
            const std::string w = dae_norm_w(s, i);
            o << "    Wt[" << i << "] = " << (w.empty() ? std::string("(numb)1.0") : w.substr(0, w.size() - 3)) << ";\n";
        }
        // предиктор
        for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
        for (int l = 0; l < nx; ++l) o << "    Rv[" << l << "] = (" << fS[l] << ");\n";
        o << "    for (bi = 0; bi < 3; bi++) {\n";
        o << "        for (l = 0; l < " << N << "; l++) W[bi * " << N << " + l] = X[l];\n";
        o << "        for (l = 0; l < " << nx << "; l++) W[bi * " << N << " + l] += RC[bi] * h * Rv[l];\n";
        o << "    }\n";
        o << "    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < nmax; it++) {\n";
        // f, g и (при обновлении) якобианы в каждой стадии
        o << "        for (bj = 0; bj < 3; bj++) {\n";
        o << "            for (l = 0; l < " << N << "; l++) Sv[l] = W[bj * " << N << " + l];\n";
        if (ht) o << "            ucuda_ts = ucuda_t + RC[bj] * h;\n";
        for (int i = 0; i < N; ++i) o << "            Fs[bj * " << N << " + " << i << "] = (" << fS[i] << ");\n";
        o << "            if (refresh) {\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = JS[(size_t)i * N + j];
                o << "                Js[bj * " << N * N << " + " << i * N + j << "] = "
                  << (d == "(numb)0.0" ? std::string("(numb)0.0") : "(" + d + ")") << ";\n";
            }
        o << "            }\n";
        o << "        }\n";
        o << "        if (refresh) {\n";
        if (sparse) {
            for (const auto& kv : SL.id) {
                const int r = kv.first.first, c = kv.first.second;
                const int bi = r / N, l = r % N, bj = c / N, m = c % N;
                o << "            Asp[" << kv.second << "] = ";
                if (!SL.orig.count(kv.first)) { o << "(numb)0.0;\n"; continue; }
                const std::string js = "Js[" + std::to_string(bj * N * N + l * N + m) + "]";
                const bool jz = JS[(size_t)l * N + m] == "(numb)0.0";
                if (l < nx) {
                    if (bi == bj && l == m) o << "(numb)1.0" << (jz ? "" : " - h * RA[" + std::to_string(bi * 3 + bj) + "] * " + js);
                    else o << "-h * RA[" << bi * 3 + bj << "] * " << js;
                } else o << js;
                o << ";\n";
            }
            o << "            ucuda_dn = 0;\n";
            sparse_lu_factor(o, SL, "            ", "ucuda_dn");
            o << "            if (ucuda_dn) {\n";
        }
        {
        o << "            for (bi = 0; bi < 3; bi++) for (l = 0; l < " << N << "; l++)\n";
        o << "            for (bj = 0; bj < 3; bj++) for (m = 0; m < " << N << "; m++) {\n";
        o << "                ir = bi * " << N << " + l; ic = bj * " << N << " + m;\n";
        o << "                if (l < " << nx << ") Am[ir * " << M << " + ic] = (bi == bj && l == m ? (numb)1.0 : (numb)0.0)"
             " - h * RA[bi * 3 + bj] * Js[bj * " << N * N << " + l * " << N << " + m];\n";
        o << "                else Am[ir * " << M << " + ic] = bi == bj ? Js[bi * " << N * N << " + l * " << N << " + m] : (numb)0.0;\n";
        o << "            }\n";
        o << "            for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "                ip = ik; mx = fabs(Am[ik * " << M << " + ik]);\n";
        o << "                for (ir = ik + 1; ir < " << M << "; ir++) { av = fabs(Am[ir * " << M << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n";
        o << "                piv[ik] = ip;\n";
        o << "                if (ip != ik) for (ic = ik; ic < " << M << "; ic++) { mlt = Am[ik * " << M << " + ic]; Am[ik * " << M
          << " + ic] = Am[ip * " << M << " + ic]; Am[ip * " << M << " + ic] = mlt; }\n";
        o << "                dgn = Am[ik * " << M << " + ik];\n";
        o << "                if (fabs(dgn) < (numb)1e-30) { sing = 1; break; }\n";   // 1e-300 в float — ноль
        o << "                for (ir = ik + 1; ir < " << M << "; ir++) {\n";
        o << "                    mlt = Am[ir * " << M << " + ik] / dgn; Am[ir * " << M << " + ik] = mlt;\n";
        o << "                    for (ic = ik + 1; ic < " << M << "; ic++) Am[ir * " << M << " + ic] -= mlt * Am[ik * " << M << " + ic];\n";
        o << "                }\n";
        o << "            }\n";
        if (sparse) o << "            }   /* ucuda_dn */\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        }
        o << "        if (sing) break;\n";
        // невязка
        o << "        for (bi = 0; bi < 3; bi++) for (l = 0; l < " << N << "; l++) {\n";
        o << "            ir = bi * " << N << " + l;\n";
        o << "            if (l < " << nx << ") Rv[ir] = W[ir] - X[l] - h * (RA[bi * 3] * Fs[l] + RA[bi * 3 + 1] * Fs["
          << N << " + l] + RA[bi * 3 + 2] * Fs[" << 2 * N << " + l]);\n";
        o << "            else Rv[ir] = Fs[ir];\n";
        o << "        }\n";
        // решение
        if (sparse) {
            o << "        if (!ucuda_dn) {\n";
            sparse_lu_solve(o, SL, "Rv", "Dv", "        ");
            o << "        for (ik = 0; ik < " << M << "; ik++) Rv[ik] = Dv[ik];\n";
            o << "        } else {\n";
        }
        {
        o << "        for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "            ip = piv[ik]; if (ip != ik) { mlt = Rv[ik]; Rv[ik] = Rv[ip]; Rv[ip] = mlt; }\n";
        o << "            for (ir = ik + 1; ir < " << M << "; ir++) Rv[ir] -= Am[ir * " << M << " + ik] * Rv[ik];\n";
        o << "        }\n";
        o << "        for (ik = " << M - 1 << "; ik >= 0; ik--) {\n";
        o << "            for (ic = ik + 1; ic < " << M << "; ic++) Rv[ik] -= Am[ik * " << M << " + ic] * Rv[ic];\n";
        o << "            Rv[ik] /= Am[ik * " << M << " + ik];\n";
        o << "        }\n";
        if (sparse) o << "        }   /* ucuda_dn */\n";
        }
        o << "        nrm = 0; xx = 0;\n";
        o << "        for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "            W[ik] -= Rv[ik];\n";
        o << "            mlt = Wt[ik % " << N << "] * Rv[ik]; nrm += mlt * mlt;\n";
        o << "            mlt = Wt[ik % " << N << "] * W[ik];  xx += mlt * mlt;\n";
        o << "        }\n";
        o << "        if (nrm < ntol * ntol || nrm < ntol * ntol * xx) break;\n";
        o << "        if (sizeof(numb) == 4 && nrm < (numb)3.6e-12f * xx) break;\n";
        o << "        if (nfull || nrm > (numb)0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n";
        o << "    for (l = 0; l < " << N << "; l++) X[l] = W[" << 2 * N << " + l];\n";
        return o.str();
    }

    // Явный РК (Бутчер A, b, c; S стадий) с решением связи на каждой стадии: x стадии — как
    // у явной схемы, z стадии — Ньютон по z на g(x_s, z, t_s) = 0 (старт — z прошлой стадии),
    // k_s = f(x_s, z_s, t_s). В конце z_{n+1} из g(x_{n+1}, z) = 0. Индекс 1 (dg/dz невырожден).
    // Euler, Midpoint (полуявный РК2 отчёта CD-ДАУ), RK4.
    std::string dae_explicit_rk(const System& s, int S, const double* A, const double* b, const double* c) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const std::string tn = system_uses_time(s) ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        dae_require_index1(JS, N, nx, "explicit DAE methods");
        const int nslot = newton_slot(s);
        std::ostringstream o;
        o << "    numb Sv[" << N << "];\n";
        o << "    numb Kx[" << (nx * S > 0 ? nx * S : 1) << "];\n";
        o << "    numb Gz[" << (nz > 0 ? nz : 1) << "], Jz[" << (nz * nz > 0 ? nz * nz : 1) << "];\n";
        o << "    const numb ntol = a[" << nslot << "];\n";
        o << "    const int  nmax = (int)a[" << nslot + 1 << "];\n";
        o << "    int it, ik, ir, ic, ip;\n";
        o << "    numb mx, av, mlt, nrm, zz;\n";
        if (!tn.empty()) o << "    numb ucuda_ts;\n";
        for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
        // Ньютон по z над Sv (x фиксированы); Gauss с выбором главного в форме цикла.
        auto z_solve = [&](const std::string& tstage) {
            if (!tn.empty()) o << "    ucuda_ts = " << tstage << ";\n";
            o << "    for (it = 0; it < nmax; it++) {\n";
            for (int k = 0; k < nz; ++k) o << "        Gz[" << k << "] = (" << fS[nx + k] << ");\n";
            for (int i = 0; i < nz; ++i)
                for (int j = 0; j < nz; ++j)
                    o << "        Jz[" << i * nz + j << "] = (" << JS[(size_t)(nx + i) * N + nx + j] << ");\n";
            o << "        for (ik = 0; ik < " << nz << "; ik++) {\n"
                 "            ip = ik; mx = fabs(Jz[ik * " << nz << " + ik]);\n"
                 "            for (ir = ik + 1; ir < " << nz << "; ir++) { av = fabs(Jz[ir * " << nz << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
                 "            if (ip != ik) { for (ic = 0; ic < " << nz << "; ic++) { mlt = Jz[ik * " << nz << " + ic]; Jz[ik * " << nz << " + ic] = Jz[ip * " << nz << " + ic]; Jz[ip * " << nz << " + ic] = mlt; }"
                 " mlt = Gz[ik]; Gz[ik] = Gz[ip]; Gz[ip] = mlt; }\n"
                 "            if (fabs(Jz[ik * " << nz << " + ik]) < (numb)1e-300) break;\n"
                 "            for (ir = ik + 1; ir < " << nz << "; ir++) {\n"
                 "                mlt = Jz[ir * " << nz << " + ik] / Jz[ik * " << nz << " + ik];\n"
                 "                for (ic = ik; ic < " << nz << "; ic++) Jz[ir * " << nz << " + ic] -= mlt * Jz[ik * " << nz << " + ic];\n"
                 "                Gz[ir] -= mlt * Gz[ik];\n"
                 "            }\n"
                 "        }\n"
                 "        nrm = 0; zz = 0;\n"
                 "        for (ik = " << nz - 1 << "; ik >= 0; ik--) {\n"
                 "            for (ic = ik + 1; ic < " << nz << "; ic++) Gz[ik] -= Jz[ik * " << nz << " + ic] * Gz[ic];\n"
                 "            Gz[ik] /= Jz[ik * " << nz << " + ik];\n"
                 "            Sv[" << nx << " + ik] -= Gz[ik]; nrm += Gz[ik] * Gz[ik]; zz += Sv[" << nx << " + ik] * Sv[" << nx << " + ik];\n"
                 "        }\n"
                 "        if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n"
                 "    }\n";
        };
        char buf[64];
        auto cnum = [&](double v) { std::snprintf(buf, sizeof(buf), "(numb)%.17g", v); return std::string(buf); };
        for (int st = 0; st < S; ++st) {
            o << "    /* stage " << st << " */\n";
            if (st > 0)
                for (int l = 0; l < nx; ++l) {
                    o << "    Sv[" << l << "] = X[" << l << "]";
                    for (int j = 0; j < st; ++j)
                        if (A[st * S + j] != 0.0) o << " + h * " << cnum(A[st * S + j]) << " * Kx[" << j * nx + l << "]";
                    o << ";\n";
                }
            z_solve(c[st] == 0.0 ? std::string("ucuda_t") : "ucuda_t + " + cnum(c[st]) + " * h");
            for (int l = 0; l < nx; ++l) o << "    Kx[" << st * nx + l << "] = (" << fS[l] << ");\n";
        }
        o << "    /* x_{n+1}, then z_{n+1} from the constraint */\n";
        for (int l = 0; l < nx; ++l) {
            o << "    X[" << l << "] = X[" << l << "]";
            for (int j = 0; j < S; ++j) if (b[j] != 0.0) o << " + h * " << cnum(b[j]) << " * Kx[" << j * nx + l << "]";
            o << ";\n    Sv[" << l << "] = X[" << l << "];\n";
        }
        z_solve("ucuda_t + h");
        for (int k = 0; k < nz; ++k) o << "    X[" << nx + k << "] = Sv[" << nx + k << "];\n";
        return o.str();
    }

    // Rosenbrock ROS2 (Verwer, Spee, Blom, Hundsdorfer 1999), gamma = 1 + 1/sqrt 2, для M y' = F(y),
    // M = diag(I_x, 0_z):
    //   (M - gamma*h*J) k1 = h F(y_n),
    //   (M - gamma*h*J) k2 = h F(y_n + k1) - 2 M k1,
    //   y_{n+1} = y_n + 3/2 k1 + 1/2 k2.
    // Линейно-неявный: одно LU на шаг, без итераций Ньютона. L-устойчив, порядок 2. Якобиан —
    // в y_n. Неавтономные системы требуют dF/dt в стадиях — пока отказ.
    // Явное время (Verwer et al.): к правым частям стадий добавляется +-gamma*h^2*F_t(t_n, y_n),
    //   k1: h F(t_n, y_n) + gamma h^2 F_t,   k2: h F(t_n + h, y_n + k1) - 2 M k1 - gamma h^2 F_t.
    // F_t — символьная производная по t (pn_diff), как якобиан.
    std::string dae_ros2(const System& s) {
        const bool ht = system_uses_time(s);
        const int N = (int)s.vars.size(), nx = s.dae_nx;
        const auto F0 = rhs_over(s, "X", time_name(s)), FY = rhs_over(s, "Yv", time_name(s, "(ucuda_t + h)"));
        const auto J = jac_over(s, "X", time_name(s));
        std::vector<std::string> Ft;
        if (ht) {
            NameMap nm = build_namemap(s, "X", kStepTimeArg);
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                Ft.push_back(emit_to_str(pn_diff(p.parse(), "t"), nm));
            }
        }
        const double gam = 1.0 + 1.0 / std::sqrt(2.0);
        std::ostringstream o;
        o << "    /* Rosenbrock ROS2, gamma = 1 + 1/sqrt(2) */\n";
        o << "    const numb gh = (numb)" << fmtnum(gam) << " * h;\n";
        o << "    numb Am[" << N * N << "], K1[" << N << "], K2[" << N << "], Yv[" << N << "];\n";
        o << "    int piv[" << N << "];\n    int ik, ir, ic, ip;\n    numb mx, av, mlt;\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = J[(size_t)i * N + j];
                const bool diag_m = (i == j && i < nx);
                o << "    Am[" << i * N + j << "] = " << (diag_m ? "(numb)1.0" : "(numb)0.0");
                if (d != "(numb)0.0") o << " - gh * (" << d << ")";
                o << ";\n";
            }
        // LU с выбором главного (форма цикла: N мало, матрица одна на шаг)
        o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
             "        ip = ik; mx = fabs(Am[ik * " << N << " + ik]);\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) { av = fabs(Am[ir * " << N << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
             "        piv[ik] = ip;\n"
             "        if (ip != ik) for (ic = ik; ic < " << N << "; ic++) { mlt = Am[ik * " << N << " + ic]; Am[ik * " << N << " + ic] = Am[ip * " << N << " + ic]; Am[ip * " << N << " + ic] = mlt; }\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) {\n"
             "            mlt = Am[ir * " << N << " + ik] / Am[ik * " << N << " + ik];\n"
             "            Am[ir * " << N << " + ik] = mlt;\n"
             "            for (ic = ik + 1; ic < " << N << "; ic++) Am[ir * " << N << " + ic] -= mlt * Am[ik * " << N << " + ic];\n"
             "        }\n"
             "    }\n";
        auto solve = [&](const char* v) {
            o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
                 "        ip = piv[ik]; if (ip != ik) { mlt = " << v << "[ik]; " << v << "[ik] = " << v << "[ip]; " << v << "[ip] = mlt; }\n"
                 "        for (ir = ik + 1; ir < " << N << "; ir++) " << v << "[ir] -= Am[ir * " << N << " + ik] * " << v << "[ik];\n"
                 "    }\n"
                 "    for (ik = " << N - 1 << "; ik >= 0; ik--) {\n"
                 "        for (ic = ik + 1; ic < " << N << "; ic++) " << v << "[ik] -= Am[ik * " << N << " + ic] * " << v << "[ic];\n"
                 "        " << v << "[ik] /= Am[ik * " << N << " + ik];\n"
                 "    }\n";
        };
        if (ht) o << "    numb Ftv[" << N << "];\n";
        for (int i = 0; i < N && ht; ++i) o << "    Ftv[" << i << "] = gh * h * (" << Ft[i] << ");\n";
        for (int i = 0; i < N; ++i)
            o << "    K1[" << i << "] = h * (" << F0[i] << ")" << (ht ? " + Ftv[" + std::to_string(i) + "]" : "") << ";\n";
        solve("K1");
        for (int i = 0; i < N; ++i) o << "    Yv[" << i << "] = X[" << i << "] + K1[" << i << "];\n";
        for (int i = 0; i < N; ++i) {
            o << "    K2[" << i << "] = h * (" << FY[i] << ")";
            if (i < nx) o << " - (numb)2.0 * K1[" << i << "]";
            if (ht) o << " - Ftv[" << i << "]";
            o << ";\n";
        }
        solve("K2");
        for (int i = 0; i < N; ++i)
            o << "    X[" << i << "] = X[" << i << "] + (numb)1.5 * K1[" << i << "] + (numb)0.5 * K2[" << i << "];\n";
        return o.str();
    }

    // Rosenbrock для M y' = F(t, y), M = diag(I_x, 0_z), в форме Hairer-Wanner (rodas.f) — без
    // умножений на J в правых частях стадий:
    //   (M/(gamma h) - J) u_i = F(t_n + alpha_i h, y_n + sum_{j<i} a_ij u_j) + M sum_{j<i} (c_ij/h) u_j
    //                           + gt_i h F_t,
    //   y_{n+1} = y_n + sum m_i u_i.
    // J и F_t — в (t_n, y_n): одно LU на шаг, без Ньютона. Коэффициенты: RosTab ниже.
    struct RosTab {
        const char* title;
        int S;
        double gamma;
        std::vector<double> alpha, gt, m;
        std::vector<std::vector<double>> a, c;   // строго нижнетреугольные, a[i][j], j < i
        // Вложенная пара (адаптивный шаг): E = sum (m - mhat) u; пусто — оценки нет.
        // Плотный выход rodas.f: y(th) = (1-th) y0 + th (y1 + (1-th) (d2.u + th d3.u)).
        std::vector<double> mhat, d2, d3;
        int q = 0;                               // порядок оценщика
    };
    // ROS3P (Lang, Verwer 2001): порядок 3 и по x, и по z индекса 1, A-устойчив, gamma = 1/2 + sqrt(3)/6.
    const RosTab& ros3p_tab() {
        static const RosTab t = { "ROS3P (Lang-Verwer), order 3", 3, 7.886751345948129e-01,
            { 0.0, 1.0, 1.0 },
            { 7.886751345948129e-01, -2.113248654051871e-01, -1.077350269189626e+00 },
            { 2.0, 5.773502691896258e-01, 4.226497308103742e-01 },
            { {}, { 1.267949192431123e+00 }, { 1.267949192431123e+00, 0.0 } },
            { {}, { -1.607695154586736e+00 }, { -3.464101615137755e+00, -1.732050807568877e+00 } } };
        return t;
    }
    // Rodas4 (Hairer, Wanner; rodas.f, METH = 1): порядок 4, жёстко точный (последние две стадии
    // в t_n + h), L-устойчив. y_{n+1} = вход 6-й стадии + u_6.
    const RosTab& rodas4_tab() {
        static const double A51 = 0.1221224509226641e+01, A52 = 0.6019134481288629e+01,
                            A53 = 0.1253708332932087e+02, A54 = -0.6878860361058950e+00;
        static const RosTab t = { "Rodas4 (Hairer-Wanner), order 4", 6, 0.25,
            { 0.0, 0.386, 0.21, 0.63, 1.0, 1.0 },
            { 0.25, -0.1043, 0.1035, -0.362e-01, 0.0, 0.0 },
            { A51, A52, A53, A54, 1.0, 1.0 },
            { {},
              { 0.1544000000000000e+01 },
              { 0.9466785280815826e+00, 0.2557011698983284e+00 },
              { 0.3314825187068521e+01, 0.2896124015972201e+01, 0.9986419139977817e+00 },
              { A51, A52, A53, A54 },
              { A51, A52, A53, A54, 1.0 } },
            { {},
              { -0.5668800000000000e+01 },
              { -0.2430093356833875e+01, -0.2063599157091915e+00 },
              { -0.1073529058151375e+00, -0.9594562251023355e+01, -0.2047028614809616e+02 },
              { 0.7496443313967647e+01, -0.1024680431464352e+02, -0.3399990352819905e+02, 0.1170890893206160e+02 },
              { 0.8083246795921522e+01, -0.7981132988064893e+01, -0.3152159432874371e+02, 0.1631930543123136e+02,
                -0.6058818238834054e+01 } },
            // вложенное решение порядка 3 — вход 6-й стадии, E = u_6 (rodas.f: ERR = AK6)
            { A51, A52, A53, A54, 1.0, 0.0 },
            { 0.1012623508344586e+02, -0.7487995877610167e+01, -0.3480091861555747e+02,
              -0.7992771707568823e+01, 0.1025137723295662e+01, 0.0 },
            { -0.6762803392801253e+00, 0.6087714651680015e+01, 0.1643084320892478e+02,
              0.2476722511418386e+02, -0.6594389125716872e+01, 0.0 },
            3 };
        return t;
    }
    // emb = true — тело попытки адаптивного шага (ucuda_krs_emb): X константный вход, пишутся
    // Y, E = sum (m - mhat) u, стадии для плотного выхода в W и F1 = f(t + h, Y).
    std::string dae_rosenbrock(const System& s, const RosTab& T, bool emb = false) {
        const bool ht = system_uses_time(s);
        const int N = (int)s.vars.size(), nx = s.dae_nx, S = T.S;
        const auto J = jac_over(s, "X", time_name(s));
        std::vector<std::string> Ft;
        if (ht) {
            NameMap nm = build_namemap(s, "X", kStepTimeArg);
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                Ft.push_back(emit_to_str(pn_diff(p.parse(), "t"), nm));
            }
        }
        std::ostringstream o;
        o << "    /* Rosenbrock " << T.title << " */\n";
        o << "    const numb ihg = (numb)1 / ((numb)" << fmtnum(T.gamma) << " * h), ih = (numb)1 / h;\n";
        o << "    numb Am[" << N * N << "], U[" << S * N << "], Yv[" << N << "];\n";
        o << "    int piv[" << N << "];\n    int ik, ir, ic, ip;\n    numb mx, av, mlt;\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = J[(size_t)i * N + j];
                o << "    Am[" << i * N + j << "] = " << ((i == j && i < nx) ? "ihg" : "(numb)0.0");
                if (d != "(numb)0.0") o << " - (" << d << ")";
                o << ";\n";
            }
        o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
             "        ip = ik; mx = fabs(Am[ik * " << N << " + ik]);\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) { av = fabs(Am[ir * " << N << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
             "        piv[ik] = ip;\n"
             "        if (ip != ik) for (ic = ik; ic < " << N << "; ic++) { mlt = Am[ik * " << N << " + ic]; Am[ik * " << N << " + ic] = Am[ip * " << N << " + ic]; Am[ip * " << N << " + ic] = mlt; }\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) {\n"
             "            mlt = Am[ir * " << N << " + ik] / Am[ik * " << N << " + ik];\n"
             "            Am[ir * " << N << " + ik] = mlt;\n"
             "            for (ic = ik + 1; ic < " << N << "; ic++) Am[ir * " << N << " + ic] -= mlt * Am[ik * " << N << " + ic];\n"
             "        }\n"
             "    }\n";
        if (ht) {
            o << "    numb Ftv[" << N << "];\n";
            for (int i = 0; i < N; ++i) o << "    Ftv[" << i << "] = h * (" << Ft[i] << ");\n";
        }
        for (int st = 0; st < S; ++st) {
            const std::string base = "U + " + std::to_string(st * N);
            o << "    /* stage " << st + 1 << " */\n";
            for (int i = 0; i < N; ++i) {
                o << "    Yv[" << i << "] = X[" << i << "]";
                for (int j = 0; j < st; ++j)
                    if (T.a[st][j] != 0.0) o << " + (numb)(" << fmtnum(T.a[st][j]) << ") * U[" << j * N + i << "]";
                o << ";\n";
            }
            const std::string tst = T.alpha[st] == 0.0 ? std::string() : "(ucuda_t + (numb)(" + fmtnum(T.alpha[st]) + ") * h)";
            const auto F = rhs_over(s, "Yv", tst.empty() ? time_name(s) : time_name(s, tst.c_str()));
            for (int i = 0; i < N; ++i) {
                o << "    U[" << st * N + i << "] = (" << F[i] << ")";
                if (i < nx)
                    for (int j = 0; j < st; ++j)
                        if (T.c[st][j] != 0.0) o << " + (numb)(" << fmtnum(T.c[st][j]) << ") * ih * U[" << j * N + i << "]";
                if (ht && T.gt[st] != 0.0) o << " + (numb)(" << fmtnum(T.gt[st]) << ") * Ftv[" << i << "]";
                o << ";\n";
            }
            o << "    {\n        numb* v = " << base << ";\n";
            o << "        for (ik = 0; ik < " << N << "; ik++) {\n"
                 "            ip = piv[ik]; if (ip != ik) { mlt = v[ik]; v[ik] = v[ip]; v[ip] = mlt; }\n"
                 "            for (ir = ik + 1; ir < " << N << "; ir++) v[ir] -= Am[ir * " << N << " + ik] * v[ik];\n"
                 "        }\n"
                 "        for (ik = " << N - 1 << "; ik >= 0; ik--) {\n"
                 "            for (ic = ik + 1; ic < " << N << "; ic++) v[ik] -= Am[ik * " << N << " + ic] * v[ic];\n"
                 "            v[ik] /= Am[ik * " << N << " + ik];\n"
                 "        }\n    }\n";
        }
        for (int i = 0; i < N; ++i) {
            o << "    " << (emb ? "Y[" : "X[") << i << "] = X[" << i << "]";
            for (int j = 0; j < S; ++j)
                if (T.m[j] != 0.0) o << " + (numb)(" << fmtnum(T.m[j]) << ") * U[" << j * N + i << "]";
            o << ";\n";
        }
        if (emb) {
            o << "    (void)F0;\n";
            for (int i = 0; i < N; ++i) {
                o << "    E[" << i << "] = (numb)0";
                for (int j = 0; j < S; ++j)
                    if (T.m[j] != T.mhat[j]) o << " + (numb)(" << fmtnum(T.m[j] - T.mhat[j]) << ") * U[" << j * N + i << "]";
                o << ";\n";
            }
            o << "    for (ik = 0; ik < " << S * N << "; ik++) W[ik] = U[ik];\n";
            const auto fy = rhs_over(s, "Y", time_name(s, "(ucuda_t + h)"));
            for (int i = 0; i < N; ++i) o << "    F1[" << i << "] = (" << fy[(size_t)i] << ");\n";
        }
        return o.str();
    }

    // Плотный выход Rosenbrock (rodas.f, CONTRO): D = [y0, y1, d2.u, d3.u].
    std::string ros_dprep_body(const System& s, const RosTab& T) {
        const int N = (int)s.vars.size();
        std::ostringstream o;
        o << "    (void)F0; (void)F1; (void)a; (void)h;\n";
        for (int i = 0; i < N; ++i) {
            o << "    D[" << i << "] = X[" << i << "]; D[" << N + i << "] = Y[" << i << "];\n";
            o << "    D[" << 2 * N + i << "] = (numb)0";
            for (int j = 0; j < T.S; ++j)
                if (T.d2[j] != 0.0) o << " + (numb)(" << fmtnum(T.d2[j]) << ") * W[" << j * N + i << "]";
            o << ";\n    D[" << 3 * N + i << "] = (numb)0";
            for (int j = 0; j < T.S; ++j)
                if (T.d3[j] != 0.0) o << " + (numb)(" << fmtnum(T.d3[j]) << ") * W[" << j * N + i << "]";
            o << ";\n";
        }
        return o.str();
    }
    std::string ros_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        const numb s1 = (numb)1 - th;\n";
        o << "        Yo[l] = D[l] * s1 + th * (D[N + l] + s1 * (D[2 * N + l] + th * D[3 * N + l]));\n";
        o << "    }\n";
        return o.str();
    }

    // CD для ДАУ индекса 1 (уровень B, отчёт CD-ДАУ): связь решается всякий раз, когда меняется x.
    //   D  (шаг h1 = a[0]*h, прямой порядок x): x_i += h1 f_i(x, z, t_n), затем z из g(x, z) = 0;
    //   D* (шаг h2 = h - h1, обратный порядок): x_i - x_i^old - h2 f_i(x, z, t_{n+1}) = 0 и
    //       g(x, z, t_{n+1}) = 0 — одним Ньютоном по (x_i, z).
    // Замороженная z дала бы первый порядок (правило отчёта). yoshida — композиция трёх CD-шагов
    // (gamma1, gamma2, gamma1), порядок 4 при a[0] = 0.5; каждый шаг — в своём блоке с затенённым h
    // (и ucuda_t, если система неавтономна).
    std::string dae_cd(const System& s, bool yoshida) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const int MM = 1 + nz;
        // Ньютон по неизвестным U (индексы в Sv): невязки R, якобиан Jm (|U| x |U|), форма цикла.
        auto newton = [&](std::ostringstream& o, const std::vector<int>& U, const std::vector<std::string>& R,
                          const std::vector<std::string>& Jm) {
            const int M = (int)U.size();
            o << "        for (it = 0; it < nmax; it++) {\n";
            for (int k = 0; k < M; ++k) o << "            Gm[" << k << "] = (" << R[k] << ");\n";
            for (int k = 0; k < M * M; ++k) o << "            Jmm[" << k << "] = (" << Jm[k] << ");\n";
            o << "            for (ik = 0; ik < " << M << "; ik++) {\n"
                 "                ip = ik; mx = fabs(Jmm[ik * " << M << " + ik]);\n"
                 "                for (ir = ik + 1; ir < " << M << "; ir++) { av = fabs(Jmm[ir * " << M << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
                 "                if (ip != ik) { for (ic = 0; ic < " << M << "; ic++) { mlt = Jmm[ik * " << M << " + ic]; Jmm[ik * " << M << " + ic] = Jmm[ip * " << M << " + ic]; Jmm[ip * " << M << " + ic] = mlt; }"
                 " mlt = Gm[ik]; Gm[ik] = Gm[ip]; Gm[ip] = mlt; }\n"
                 "                if (fabs(Jmm[ik * " << M << " + ik]) < (numb)1e-300) break;\n"
                 "                for (ir = ik + 1; ir < " << M << "; ir++) {\n"
                 "                    mlt = Jmm[ir * " << M << " + ik] / Jmm[ik * " << M << " + ik];\n"
                 "                    for (ic = ik; ic < " << M << "; ic++) Jmm[ir * " << M << " + ic] -= mlt * Jmm[ik * " << M << " + ic];\n"
                 "                    Gm[ir] -= mlt * Gm[ik];\n"
                 "                }\n"
                 "            }\n"
                 "            for (ik = " << M - 1 << "; ik >= 0; ik--) {\n"
                 "                for (ic = ik + 1; ic < " << M << "; ic++) Gm[ik] -= Jmm[ik * " << M << " + ic] * Gm[ic];\n"
                 "                Gm[ik] /= Jmm[ik * " << M << " + ik];\n"
                 "            }\n"
                 "            nrm = 0; zz = 0;\n";
            for (int k = 0; k < M; ++k)
                o << "            Sv[" << U[k] << "] -= Gm[" << k << "]; nrm += Gm[" << k << "] * Gm[" << k << "]; zz += Sv["
                  << U[k] << "] * Sv[" << U[k] << "];\n";
            o << "            if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n"
                 "        }\n";
        };
        dae_require_index1(JS, N, nx, "DAE CD");
        std::vector<int> Uz;
        std::vector<std::string> Rz, Jz;
        for (int k = 0; k < nz; ++k) { Uz.push_back(nx + k); Rz.push_back(fS[nx + k]); }
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nz; ++j) Jz.push_back(JS[(size_t)(nx + i) * N + nx + j]);

        auto cd_step = [&](std::ostringstream& o) {
            o << "    numb Sv[" << N << "], Gm[" << MM << "], Jmm[" << MM * MM << "];\n";
            o << "    numb ucuda_xo, mx, av, mlt, nrm, zz;\n    int it, ik, ir, ic, ip;\n";
            if (ht) o << "    numb ucuda_ts;\n";
            o << "    const numb h1 = a[0] * h, h2 = h - h1;\n";
            o << "    const numb ntol = a[" << nslot << "];\n    const int  nmax = (int)a[" << nslot + 1 << "];\n";
            for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
            o << "    /* D: explicit, forward order; z re-solved after every update */\n";
            for (int i = 0; i < nx; ++i) {
                if (ht) o << "    ucuda_ts = ucuda_t;\n";
                o << "    Sv[" << i << "] = Sv[" << i << "] + h1 * (" << fS[i] << ");\n";
                newton(o, Uz, Rz, Jz);
            }
            o << "    /* D*: implicit, reverse order; (x_i, z) by one Newton */\n";
            for (int i = nx - 1; i >= 0; --i) {
                if (ht) o << "    ucuda_ts = ucuda_t + h;\n";
                o << "    ucuda_xo = Sv[" << i << "];\n";
                std::vector<int> U = { i };
                std::vector<std::string> R = { "Sv[" + std::to_string(i) + "] - ucuda_xo - h2 * (" + fS[i] + ")" };
                for (int k = 0; k < nz; ++k) { U.push_back(nx + k); R.push_back(fS[nx + k]); }
                std::vector<std::string> Jm;
                for (int r = 0; r < MM; ++r)
                    for (int c = 0; c < MM; ++c) {
                        const int row = r == 0 ? i : nx + r - 1, col = U[c];
                        const std::string& d = JS[(size_t)row * N + col];
                        if (r == 0) Jm.push_back(std::string(c == 0 ? "(numb)1.0" : "(numb)0.0") + " - h2 * (" + d + ")");
                        else        Jm.push_back(d);
                    }
                newton(o, U, R, Jm);
            }
            for (int i = 0; i < N; ++i) o << "    X[" << i << "] = Sv[" << i << "];\n";
        };
        std::ostringstream o;
        if (!yoshida) { cd_step(o); return o.str(); }
        const double cb = std::cbrt(2.0), g1 = 1.0 / (2.0 - cb), g2 = -cb / (2.0 - cb);
        o << "    /* CD + Yoshida-4: CD(g1 h), CD(g2 h), CD(g1 h); 4th order at a[0] = 0.5 */\n";
        o << "    const numb ucuda_h0 = h;\n";
        if (ht) o << "    const numb ucuda_t0 = ucuda_t;\n";
        const double gs[3] = { g1, g2, g1 }, off[3] = { 0.0, g1, g1 + g2 };
        for (int k = 0; k < 3; ++k) {
            o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(gs[k]) << ";\n";
            if (ht) o << "    const numb ucuda_t = ucuda_t0 + ucuda_h0 * (numb)" << fmtnum(off[k]) << ";\n";
            std::ostringstream b;
            cd_step(b);
            o << b.str() << "    }\n";
        }
        return o.str();
    }

    // CD для индекса 2 (отчёт CD-ДАУ, «как SHAKE»): на каждом полушаге z постоянна и подбирается так,
    // чтобы связь выполнялась в КОНЦЕ полушага: g(x(h1; z1), z1) = 0 после D, g(x(h; z2), z2) = 0 после D*.
    //   D  (h1 = a[0] h, прямой порядок): x_i += h1 f_i(x, z1, t_n);
    //   D* (h2 = h - h1, обратный порядок): x_i - x_i^old - h2 f_i(x, z2, t_{n+1}) = 0 скалярным Ньютоном.
    // z полушага — Ньютон с разностным якобианом (nz + 1 проходов на итерацию; «секущие» отчёта). В X
    // пишется (z1 + z2)/2 — z середины шага (сравнивать с точным решением при t - h/2). Работает и на
    // индексе 1 (g с z), но там обычный DAE CD точнее (z после каждого x_i).
    std::string dae_cd2(const System& s) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const int NZ = nz > 0 ? nz : 1;
        std::ostringstream o;
        o << "    /* DAE CD, index 2: z constant per half-step, chosen so that g holds at its end */\n";
        o << "    numb Sv[" << N << "], Xb[" << (nx > 0 ? nx : 1) << "], zA[" << NZ << "], zB[" << NZ << "], Rv[" << NZ
          << "], R0[" << NZ << "], Jf[" << NZ * NZ << "];\n";
        o << "    numb ucuda_xo, ucuda_r, ucuda_d, mx, mlt, zt, dk, nrm, zz;\n    int it, jt, ik, ir, ic, ip;\n";
        if (ht) o << "    numb ucuda_ts;\n";
        o << "    const numb h1 = a[0] * h, h2 = h - h1;\n";
        o << "    const numb ntol = a[" << nslot << "];\n    const int  nmax = (int)a[" << nslot + 1 << "];\n";
        o << "    const numb ucuda_fd = sizeof(numb) == 4 ? (numb)3e-4 : (numb)1.5e-8;\n";
        // проход полушага при данной z: x из Xb, результат в Sv, невязка связи в Rv
        o << "    auto ucuda_sweep = [&](int star, const numb* zv) {\n";
        o << "        for (ik = 0; ik < " << nx << "; ik++) Sv[ik] = Xb[ik];\n";
        o << "        for (ik = 0; ik < " << nz << "; ik++) Sv[" << nx << " + ik] = zv[ik];\n";
        o << "        if (!star) {\n";
        if (ht) o << "            ucuda_ts = ucuda_t;\n";
        for (int i = 0; i < nx; ++i) o << "            Sv[" << i << "] = Sv[" << i << "] + h1 * (" << fS[i] << ");\n";
        if (ht) o << "            ucuda_ts = ucuda_t + h1;\n";
        o << "        } else {\n";
        if (ht) o << "            ucuda_ts = ucuda_t + h;\n";
        for (int i = nx - 1; i >= 0; --i) {
            const std::string& d = JS[(size_t)i * N + i];
            o << "            ucuda_xo = Sv[" << i << "];\n";
            o << "            for (jt = 0; jt < nmax; jt++) {\n";
            o << "                ucuda_r = Sv[" << i << "] - ucuda_xo - h2 * (" << fS[i] << ");\n";
            o << "                ucuda_d = (numb)1.0" << (d == "(numb)0.0" ? std::string() : " - h2 * (" + d + ")") << ";\n";
            o << "                ucuda_r /= ucuda_d; Sv[" << i << "] -= ucuda_r;\n";
            o << "                if (fabs(ucuda_r) <= ntol * ((numb)1.0 + fabs(Sv[" << i << "]))) break;\n";
            o << "            }\n";
        }
        o << "        }\n";
        for (int k = 0; k < nz; ++k) o << "        Rv[" << k << "] = (" << fS[(size_t)(nx + k)] << ");\n";
        o << "    };\n";
        // Ньютон по z полушага с разностным якобианом
        o << "    auto ucuda_zsolve = [&](int star, numb* zv) {\n";
        o << "        for (it = 0; it < nmax; it++) {\n";
        o << "            ucuda_sweep(star, zv);\n";
        o << "            for (ik = 0; ik < " << nz << "; ik++) R0[ik] = Rv[ik];\n";
        o << "            for (ic = 0; ic < " << nz << "; ic++) {\n";
        o << "                zt = zv[ic]; dk = ucuda_fd * ((numb)1.0 + fabs(zt)); zv[ic] = zt + dk;\n";
        o << "                ucuda_sweep(star, zv);\n";
        o << "                for (ir = 0; ir < " << nz << "; ir++) Jf[ir * " << nz << " + ic] = (Rv[ir] - R0[ir]) / dk;\n";
        o << "                zv[ic] = zt;\n";
        o << "            }\n";
        o << "            for (ik = 0; ik < " << nz << "; ik++) {\n"
             "                ip = ik; mx = fabs(Jf[ik * " << nz << " + ik]);\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) if (fabs(Jf[ir * " << nz << " + ik]) > mx) { mx = fabs(Jf[ir * " << nz << " + ik]); ip = ir; }\n"
             "                if (ip != ik) { for (ic = ik; ic < " << nz << "; ic++) { mlt = Jf[ik * " << nz << " + ic]; Jf[ik * " << nz << " + ic] = Jf[ip * " << nz << " + ic]; Jf[ip * " << nz << " + ic] = mlt; } mlt = R0[ik]; R0[ik] = R0[ip]; R0[ip] = mlt; }\n"
             "                if (!(mx > (numb)0)) break;\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) {\n"
             "                    mlt = Jf[ir * " << nz << " + ik] / Jf[ik * " << nz << " + ik];\n"
             "                    for (ic = ik; ic < " << nz << "; ic++) Jf[ir * " << nz << " + ic] -= mlt * Jf[ik * " << nz << " + ic];\n"
             "                    R0[ir] -= mlt * R0[ik];\n"
             "                }\n"
             "            }\n";
        o << "            nrm = 0; zz = 0;\n";
        o << "            for (ik = " << nz - 1 << "; ik >= 0; ik--) {\n"
             "                for (ic = ik + 1; ic < " << nz << "; ic++) R0[ik] -= Jf[ik * " << nz << " + ic] * R0[ic];\n"
             "                R0[ik] /= Jf[ik * " << nz << " + ik];\n"
             "                zv[ik] -= R0[ik]; nrm += R0[ik] * R0[ik]; zz += zv[ik] * zv[ik];\n"
             "            }\n";
        o << "            if (!(nrm == nrm)) break;\n";
        o << "            if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n";
        o << "        }\n";
        o << "        ucuda_sweep(star, zv);\n";
        o << "    };\n";
        for (int l = 0; l < nx; ++l) o << "    Xb[" << l << "] = X[" << l << "];\n";
        for (int k = 0; k < nz; ++k) o << "    zA[" << k << "] = X[" << nx + k << "];\n";
        o << "    ucuda_zsolve(0, zA);\n";
        for (int l = 0; l < nx; ++l) o << "    Xb[" << l << "] = Sv[" << l << "];\n";
        for (int k = 0; k < nz; ++k) o << "    zB[" << k << "] = zA[" << k << "];\n";
        o << "    ucuda_zsolve(1, zB);\n";
        for (int l = 0; l < nx; ++l) o << "    X[" << l << "] = Sv[" << l << "];\n";
        for (int k = 0; k < nz; ++k)
            o << "    X[" << nx + k << "] = (numb)0.5 * (zA[" << k << "] + zB[" << k << "]);\n";
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
    // Предрасчёт ("CD fast"): константа шага — имя и выражение над a[] и h.
    using PrepList = std::vector<std::pair<std::string, std::string>>;
    bool pn_free_of_vars(const PN& n, const std::vector<std::string>& vars) {
        for (const std::string& v : vars) if (pn_contains_var(n, v)) return false;
        return true;
    }

    // prep != nullptr -- вариант "CD fast": если коэффициент при своей
    // переменной не зависит ни от одной переменной состояния (state_vars),
    // обратный знаменатель 1/(1 - hs*coef) уходит в prep под именем prep_name,
    // а шаг умножает на него вместо деления.
    void emit_diag_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                               const std::string& v, const std::string& x,
                               const NameMap& nm, const char* hs, const char* sty,
                               const std::string& saved,
                               PrepList* prep = nullptr,
                               const std::vector<std::string>* state_vars = nullptr,
                               const std::string& prep_name = std::string()) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);

        if (linear && prep && !pn_is_zero(coef) && pn_free_of_vars(coef, *state_vars)) {
            char dsgn = peel_sign(coef, '-', '+');
            std::string coef_c = wrap_paren(emit_to_str(coef, nm), needs_paren_after_mul(coef));
            prep->emplace_back(prep_name, "(numb)1 / (1 " + std::string(1, dsgn) + " "
                                          + mul_step(coef, coef_c, hs) + ")");
            if (pn_is_zero(rem))
                o << "    " << x << " = " << x << " * " << prep_name << ";\n";
            else {
                char nsgn = peel_sign(rem, '+', '-');
                std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
                o << "    " << x << " = (" << x
                  << " " << nsgn << " " << mul_step(rem, rem_c, hs)
                  << ") * " << prep_name << ";\n";
            }
            return;
        }

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

    // --- "CD FP" / "CD fast FP": неявная половина простыми итерациями ---------
    // Слагаемые верхней суммы f_i, не содержащие своей переменной v, за время
    // итераций не меняются (остальные X[] и a[] итерации не трогают), поэтому
    // идут в начальное значение один раз:
    //   x_fp = X[i] + hs * (r);          r -- слагаемые без v
    //   for (k < cdfp_n) X[i] = x_fp + hs * (g);   g -- слагаемые с v
    // Если f_i линейна по v (coef*v + rem), итерация -- X[i] = x_fp + c*X[i]
    // с c = hs * coef, посчитанным тоже один раз. Первое приближение то же,
    // что у CD: X[i] до полушага. Число итераций n -- рантайм-значение, как s:
    // a[] = [s, p_1..p_M, n], тело читает его в cdfp_n (kCdFpIterDecl).
    void fp_collect_terms(const PN& n, bool neg, std::vector<std::pair<bool, PN>>& out) {
        if (n && n->kind == Node::Add) { fp_collect_terms(n->a, neg, out); fp_collect_terms(n->b, neg, out); return; }
        if (n && n->kind == Node::Sub) { fp_collect_terms(n->a, neg, out); fp_collect_terms(n->b, !neg, out); return; }
        if (n && n->kind == Node::Neg) { fp_collect_terms(n->a, !neg, out); return; }
        out.emplace_back(neg, n);
    }
    PN fp_sum_terms(const std::vector<std::pair<bool, PN>>& t) {
        PN acc;
        for (const auto& e : t) {
            if (!acc) acc = e.first ? pn_neg(e.second) : e.second;
            else      acc = e.first ? pn_sub(acc, e.second) : pn_add(acc, e.second);
        }
        return acc;   // nullptr -- слагаемых нет
    }
    // f = r + g по слагаемым верхней суммы; r == nullptr, если все слагаемые с v.
    void fp_split_terms(const PN& f, const std::string& v, PN& r, PN& g) {
        std::vector<std::pair<bool, PN>> terms, tr, tg;
        fp_collect_terms(f, false, terms);
        for (const auto& e : terms) (pn_contains_var(e.second, v) ? tg : tr).push_back(e);
        r = fp_sum_terms(tr);
        g = fp_sum_terms(tg);
    }
    // Множитель после "hs * ": скобки всегда, кроме атомов. CPU-зеркало
    // (SystemEvaluator::solve_diag_fp) считает hs * eval(e) -- та же группировка.
    std::string fp_factor(const PN& e, const NameMap& nm) {
        const bool atom = e && (e->kind == Node::Sym || e->kind == Node::Call
                                || (e->kind == Node::Num && e->num >= 0.0));
        return wrap_paren(emit_to_str(e, nm), !atom);
    }

    // Одно уравнение неявной половины "CD FP" / "CD fast FP". fast -- линейные
    // уравнения решаются как в CD fast (предрасчитанный 1/d или деление),
    // итерации только у нелинейных; иначе итерации у всех, кроме coef == 0
    // (там одна явная формула и есть точное решение).
    void emit_fp_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                             const std::string& v, const std::string& x,
                             const NameMap& nm, const char* hs,
                             const std::string& sfx, bool fast, PrepList* prep,
                             const std::vector<std::string>* state_vars,
                             const std::string& prep_name) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);
        if (linear && (fast || pn_is_zero(coef))) {
            emit_diag_implicit_eq(o, rhs_ast, v, x, nm, hs, "numb", "x" + sfx + "_cd",
                                  fast ? prep : nullptr, state_vars, prep_name);
            return;
        }
        PN r, g;
        if (linear) { r = pn_is_zero(rem) ? nullptr : rem; g = coef; }
        else        fp_split_terms(rhs_ast, v, r, g);
        const std::string xs = "x" + sfx + "_fp";
        o << "    numb " << xs << " = " << x;
        if (r) o << " + " << hs << " * " << fp_factor(r, nm);
        o << ";\n";
        if (linear) {
            const std::string cs = "c" + sfx + "_fp";
            o << "    numb " << cs << " = " << hs << " * " << fp_factor(g, nm) << ";\n";
            o << "    for (int cdfp_k = 0; cdfp_k < cdfp_n; ++cdfp_k)\n";
            o << "        " << x << " = " << xs << " + " << cs << " * " << x << ";\n";
        }
        else {
            o << "    for (int cdfp_k = 0; cdfp_k < cdfp_n; ++cdfp_k)\n";
            o << "        " << x << " = " << xs << " + " << hs << " * " << fp_factor(g, nm) << ";\n";
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
                                 bool flip_halves = false, bool fast = false,
                                 bool fp = false) {
        // fast -- "CD fast": только вещественный CD. h1, h2 и обратные
        // знаменатели, не зависящие от состояния, идут в блок предрасчёта
        // (см. krs_split_prep в codegen.hpp).
        // fp -- "CD FP" / "CD fast FP": неявная половина простыми итерациями
        // (emit_fp_implicit_eq), их число -- a[M+1] (cd_fp_slot).
        if (fast && kind != CdKind::Real)
            throw std::runtime_error("CD fast: only the real CD is supported");
        if (fp && (kind != CdKind::Real || flip_halves))
            throw std::runtime_error("CD FP: only the real CD {0-1} is supported");
        PrepList prep;
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
        // Явное время (вещественный CD, CD {1-0}, CD FP): явный полушаг — в начале своего
        // отрезка, неявный — в конце. {0-1}: t_n и t_n + h; {1-0}: оба в t_n + h1. Без t
        // в системе обе таблицы имён совпадают с nm, и текст прежний.
        const bool time_cd = !cx && !fast && system_uses_time(s);
        const bool flip_cd = flip_halves || (kind == CdKind::Cx4ss10);
        const NameMap nm_exp = time_cd ? build_namemap(s, stv, flip_cd ? "(ucuda_t + h1)" : "ucuda_t") : nm;
        const NameMap nm_imp = time_cd ? build_namemap(s, stv, flip_cd ? "(ucuda_t + h1)" : "(ucuda_t + h)") : nm;
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
        if (kind == CdKind::Real && fast) {
            prep.emplace_back("h1", "h * a[0]");
            prep.emplace_back("h2", "h * (1 - a[0])");
        }
        else if (kind == CdKind::Real) {
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
            o << "    ucmplx cp = ucmplx(h * ((numb)1 / (numb)6), -h * ((numb)1 / (numb)6));   // (1-i)h/6\n";
            o << "    ucmplx cq = ucmplx(h * ((numb)1 / (numb)3),  h * ((numb)1 / (numb)6));   // (2+i)h/6\n";
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
                o << "    ucmplx g  = ucmplx((numb)0.5 * h,  h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx((numb)0.5 * h, -h * ccd_im);\n";
                cgs = { "g", "gc" };
            }
            else if (kind == CdKind::Cx4s4) {
                // (gamma/2, conj/2, conj/2, gamma/2) — палиндром.
                o << "    ucmplx g  = ucmplx((numb)0.25 * h,  (numb)0.5 * h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx((numb)0.25 * h, -(numb)0.5 * h * ccd_im);\n";
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
        // "CD FP": число итераций из a[M+1]; объявляется, только если хоть одно
        // уравнение действительно уходит на итерации (у CD fast FP на системе,
        // линейной по своим переменным, тело совпадает с CD fast).
        if (fp) {
            bool need = false;
            for (int i = 0; i < N && !need; ++i) {
                PN coef, rem;
                const bool lin = cd_try_extract_linear(rhs_ast[i], s.vars[i], coef, rem);
                need = !(lin && (fast || pn_is_zero(coef)));
            }
            if (need)
                o << "    const int cdfp_n = (int)a[" << cd_fp_slot(s) << "];\n";
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
                  << step << " * (" << emit_to_str(rhs_ast[i], nm_exp) << ");\n";
        };
        auto emit_implicit = [&](const char* step) {
            for (int i = N - 1; i >= 0; --i)
                if (fp)
                    emit_fp_implicit_eq(o, rhs_ast[i], s.vars[i],
                                        stv + ("[" + std::to_string(i) + "]"), nm_imp,
                                        step, std::to_string(i) + sfx,
                                        fast, &prep, &s.vars,
                                        "cdf_r" + std::to_string(i) + sfx);
                else
                    emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i],
                                          stv + ("[" + std::to_string(i) + "]"), nm_imp,
                                          step, sty, "x" + std::to_string(i) + "_cd" + sfx,
                                          fast ? &prep : nullptr, &s.vars,
                                          "cdf_r" + std::to_string(i) + sfx);
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

        if (fast) {
            std::ostringstream full;
            full << kKrsPrepBegin << "\n";
            for (const auto& c : prep)
                full << "    const numb " << c.first << " = " << c.second << ";\n";
            full << kKrsPrepEnd << "\n" << o.str();
            return full.str();
        }
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

        // Явное время: явная стадия (Эйлер-Кромер на h1) — в t, диагонально-неявная — в t + h1
        // (неявный Эйлер на h1); корректор берёт f по значениям стадии — тоже в t + h1.
        NameMap nm = build_namemap(s, "X", time_name(s, implicit_stage ? "(ucuda_t + h1)" : nullptr));
        NameMap nm_corr = build_namemap(s, "X", time_name(s, "(ucuda_t + h1)"));
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
              << emit_to_str(rhs_ast[i], nm_corr) << ");\n";
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

        NameMap nm = build_namemap(s, "X", time_name(s, "(ucuda_t + h)"));   // неявный: f в t + h
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
        F_SQRT, F_CBRT, F_FABS, F_FLOOR, F_CEIL,
        // F_COPYSIGN is emitted by pn_diff (derivative of fabs); known_funcs() accepts it
        // so that pn_print text (z outputs: dz/dt, sys_augmented) parses back.
        F_POW = 100, F_ATAN2, F_FMOD, F_COPYSIGN, F_FMIN, F_FMAX
    };
    struct Instr { int op; int idx; double val; };

    int func_id(const std::string& nm) {
        if (nm == "sin")return F_SIN; if (nm == "cos")return F_COS; if (nm == "tan")return F_TAN;
        if (nm == "asin")return F_ASIN; if (nm == "acos")return F_ACOS; if (nm == "atan")return F_ATAN;
        if (nm == "sinh")return F_SINH; if (nm == "cosh")return F_COSH; if (nm == "tanh")return F_TANH;
        if (nm == "exp")return F_EXP; if (nm == "log")return F_LOG; if (nm == "log2")return F_LOG2;
        if (nm == "log10")return F_LOG10; if (nm == "sqrt")return F_SQRT; if (nm == "cbrt")return F_CBRT;
        if (nm == "fabs" || nm == "abs")return F_FABS;
        if (nm == "floor")return F_FLOOR; if (nm == "ceil")return F_CEIL;
        if (nm == "pow")return F_POW; if (nm == "atan2")return F_ATAN2; if (nm == "fmod")return F_FMOD;
        if (nm == "copysign")return F_COPYSIGN; if (nm == "fmin")return F_FMIN; if (nm == "fmax")return F_FMAX;
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
            case Node::Div:
                // Как emit: u / c -> u * (1 / c), чтобы CPU совпадал с ядром.
                compile(n->a);
                if (div_by_reciprocal(n)) { out.push_back({ OP_PUSH_CONST, 0, 1.0 / n->b->num }); out.push_back({ OP_MUL,0,0 }); }
                else                      { compile(n->b); out.push_back({ OP_DIV,0,0 }); }
                break;
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
        case F_FLOOR:return std::floor(x); case F_CEIL:return std::ceil(x);
        } return 0;
    }
    double apply_func2(int fid, double a, double b) {
        switch (fid) {
        case F_POW:return std::pow(a, b); case F_ATAN2:return std::atan2(a, b); case F_FMOD:return std::fmod(a, b);
        case F_COPYSIGN:return std::copysign(a, b);
        case F_FMIN:    return std::fmin(a, b);
        case F_FMAX:    return std::fmax(a, b);
        } return 0;
    }

    // Комплексные версии — те же id, те же формулы, что и в device-коде: обе
    // ветки зовут функции из configCUDA.h, поэтому CPU-траектория Complex CD
    // повторяет GPU-шную операция в операцию (в пределах разной группировки FMA).
    // atan2/fmod комплексного смысла не имеют и до сюда не доходят: на GPU систему
    // с ними отсекает cd_check_complex_safe на кодгене, а CPU-путь кодген не
    // зовёт и спрашивает complex_unsafe_function() перед комплексной схемой.
    // floor/ceil кусочно-постоянны, и их продолжение — константа floor(Re z)
    // (см. configCUDA.h).
    ucmplx apply_func1(int fid, ucmplx x) {
        switch (fid) {
        case F_SIN:return sin(x); case F_COS:return cos(x); case F_TAN:return tan(x);
        case F_ASIN:return asin(x); case F_ACOS:return acos(x); case F_ATAN:return atan(x);
        case F_SINH:return sinh(x); case F_COSH:return cosh(x); case F_TANH:return tanh(x);
        case F_EXP:return exp(x); case F_LOG:return log(x); case F_LOG2:return log2(x);
        case F_LOG10:return log10(x); case F_SQRT:return sqrt(x); case F_CBRT:return cbrt(x);
        case F_FABS:return fabs(x);
        case F_FLOOR:return floor(x); case F_CEIL:return ceil(x);
        } return ucmplx(0, 0);
    }
    // F_COPYSIGN приходит только из якобиана (производная fabs) и всегда с
    // литералом 1 первым аргументом — поэтому a.re. atan2/fmod сюда не доходят:
    // CPU-путь отказывает им заранее (SystemEvaluator::complex_unsafe_function).
    ucmplx apply_func2(int fid, ucmplx a, ucmplx b) {
        switch (fid) {
        case F_POW:return pow(a, b);
        case F_COPYSIGN:return copysign(a.re, b);
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
    // coef не зависит от переменных состояния (и не ноль): "CD fast" умножает
    // на 1/(1 -+ hs*|coef|) вместо деления, как его GPU-предрасчёт.
    std::vector<char> diag_coef_free;
    std::vector<std::vector<Instr>> diag_coef, diag_rem;
    // "CD FP": программы итераций emit_fp_implicit_eq. Линейная f_i: fp_r = rem,
    // fp_g = coef; нелинейная: слагаемые без x_i и с x_i. fp_has_r == 0 —
    // вынесенной части нет. diag_coef_zero — coef == 0 (явная формула).
    std::vector<char> fp_has_r, diag_coef_zero;
    std::vector<std::vector<Instr>> fp_r, fp_g;
    int fp_slot = 1;   // a[fp_slot] -- число итераций CD FP (cd_fp_slot)
    // Первая функция правых частей без комплексного смысла (cd_find_complex_unsafe),
    // пусто — таких нет.
    std::string complex_unsafe;
    int newton_slot = 2;   // a[newton_slot..+2] -- tol, max_iters, full (newton_slot)
    int max_stack = 16;                        // глубина стека (с запасом)
    mutable std::vector<double> stack;
    // Отдельный стек под комплексный проход: eval и eval_complex не зовутся
    // одновременно (обе — из шага одного интегратора), но держать один буфер
    // на два типа нельзя. Аллоцируется по требованию — обычные схемы за него
    // не платят.
    mutable std::vector<ucmplx> stack_c;
};

SystemEvaluator::SystemEvaluator(const System& sys) : impl_(new Impl) {
    if (!sys.out_exprs.empty() || !sys.zb_g.empty())   // z-выходы считает шаг КРС (krs_cpu), не вычислитель
        throw std::runtime_error("SystemEvaluator: z outputs need the compiled step (krs_cpu)");
    if (sys.vars.size() != sys.rhs.size())
        throw std::runtime_error("SystemEvaluator: vars/rhs size mismatch");
    // ДАУ: строки связей 0 = g тут посчитались бы как производные — тихо неверный ответ.
    // Все CPU-пути через вычислитель правых частей для ДАУ закрыты, пока не проведены.
    if (sys.dae_nx >= 0)
        throw std::runtime_error("DAE systems run on the GPU only so far (turn on GPU)");
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
    impl_->diag_coef_free.assign((size_t)impl_->dim, 0);
    impl_->diag_coef.resize(impl_->dim);
    impl_->diag_rem.resize(impl_->dim);
    impl_->fp_has_r.assign((size_t)impl_->dim, 0);
    impl_->diag_coef_zero.assign((size_t)impl_->dim, 0);
    impl_->fp_r.resize(impl_->dim);
    impl_->fp_g.resize(impl_->dim);
    int maxdepth = 8;
    for (int i = 0; i < impl_->dim; ++i) {
        Parser p(sys.rhs[i], sys.latex);
        PN ast = p.parse();
        ByteCompiler bc{ impl_->programs[i], var_index, param_index };
        bc.compile(ast);
        // оценка глубины стека: число push не превышает длину программы
        int depth = (int)impl_->programs[i].size() + 4;
        if (depth > maxdepth) maxdepth = depth;
        if (impl_->complex_unsafe.empty())
            if (const std::string* f = cd_find_complex_unsafe(ast)) impl_->complex_unsafe = *f;

        // Разложение по своей переменной — для диагонально-неявных схем.
        PN coef, rem;
        const bool lin = cd_try_extract_linear(ast, sys.vars[i], coef, rem);
        {
            // До peel_sign ниже: он меняет coef/rem на месте.
            PN r, g;
            if (lin) { r = pn_is_zero(rem) ? nullptr : rem; g = coef; }
            else     fp_split_terms(ast, sys.vars[i], r, g);
            impl_->diag_coef_zero[(size_t)i] = lin && pn_is_zero(coef);
            impl_->fp_has_r[(size_t)i] = r ? 1 : 0;
            if (r) { ByteCompiler b{ impl_->fp_r[i], var_index, param_index }; b.compile(r); }
            if (g) { ByteCompiler b{ impl_->fp_g[i], var_index, param_index }; b.compile(g); }
            for (const auto* prog : { &impl_->fp_r[i], &impl_->fp_g[i] }) {
                int d = (int)prog->size() + 4;
                if (d > maxdepth) maxdepth = d;
            }
        }
        if (lin) {
            // Зеркалим emit_diag_implicit_eq вплоть до формы выражения: сперва
            // тот же вынос знака, затем тот же порядок умножений.
            // "CD fast" берёт обратный знаменатель из предрасчёта только там,
            // где coef не зависит от состояния, — то же условие, что в кодогене.
            impl_->diag_coef_free[(size_t)i] = !pn_is_zero(coef) && pn_free_of_vars(coef, sys.vars);
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

    impl_->newton_slot      = newton_slot(sys);
    impl_->fp_slot          = cd_fp_slot(sys);

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

const std::string& SystemEvaluator::complex_unsafe_function() const { return impl_->complex_unsafe; }

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

bool   SystemEvaluator::newton_full(const double* a) const      { return (int)a[impl_->newton_slot + 2] != 0; }
double SystemEvaluator::newton_tol(const double* a) const       { return a[impl_->newton_slot]; }
int    SystemEvaluator::newton_max_iters(const double* a) const { return (int)a[impl_->newton_slot + 1]; }

// Сборка результата повторяет emit_diag_implicit_eq оператор в оператор:
//   X[i] = (X[i] <+|-> hs*|rem|) / (1 <-|+> hs*|coef|).
// Случаи coef == 0 и rem == 0 отдельно не разбираются: на GPU они лишь убирают
// из текста деление на 1 и прибавление нуля, а это побитово тождественные
// операции (деление на 1.0 и сложение с 0.0 в IEEE точны).
bool SystemEvaluator::solve_diag_implicit(int i, double* X, const double* a, double hs,
                                          bool recip) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    double* st = impl_->stack.data();
    const double num = run_program(impl_->diag_rem[(size_t)i],  X, a, st, hs);
    const double den = run_program(impl_->diag_coef[(size_t)i], X, a, st, hs);
    const double xn  = impl_->diag_rem_neg[(size_t)i]  ? X[i] - num : X[i] + num;
    const double d   = impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den;
    // "CD fast": X * (1/d), как предрасчитанный cdf_r<i> на GPU.
    X[i] = (recip && impl_->diag_coef_free[(size_t)i]) ? xn * (1.0 / d) : xn / d;
    return true;
}

// Зеркало emit_fp_implicit_eq оператор в оператор:
//   x_fp = X[i] (+ hs * r);  X[i] = x_fp + c * X[i] (c = hs * coef) | x_fp + hs * g.
void SystemEvaluator::solve_diag_fp(int i, double* X, const double* a, double hs,
                                    bool fast) const {
    if (i < 0 || i >= impl_->dim) return;
    const int iters = (int)a[impl_->fp_slot];
    const size_t k = (size_t)i;
    const bool lin = impl_->diag_linear[k] != 0;
    if (lin && (fast || impl_->diag_coef_zero[k])) {
        solve_diag_implicit(i, X, a, hs, fast);
        return;
    }
    double* st = impl_->stack.data();
    const double xs = impl_->fp_has_r[k] ? X[i] + hs * run_program(impl_->fp_r[k], X, a, st)
                                         : X[i];
    if (lin) {
        const double c = hs * run_program(impl_->fp_g[k], X, a, st);
        for (int it = 0; it < iters; ++it) X[i] = xs + c * X[i];
    }
    else {
        for (int it = 0; it < iters; ++it) X[i] = xs + hs * run_program(impl_->fp_g[k], X, a, st);
    }
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
    if (!s.out_exprs.empty() || !s.zb_g.empty())
        throw std::runtime_error("z outputs (Solve as ODE): ExtrZ (complex extrapolation core) is not available - pick another scheme");
    if (system_uses_time(s)) throw std::runtime_error(kTimeUnsupportedMsg);
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
// Сгенерировано из 110-значных значений; руками не править.
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
      { 0.04791013711111111, -7.29488963063179e-19, 9.668991420483712e-36, -4.91512430690202e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.11224871277777777, 4.5470748154406866e-18, 3.417281894288947e-34, -5.2363920505296145e-51 },
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
      { 0.01691798978729228, -1.7105251427459866e-19, 3.723864394551783e-36, -1.7113106107946274e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.3878482784860432, -2.625122524707441e-17, 1.3798255563461028e-33, 4.33301542036532e-50 },
      { 0.03597736985150033, -2.6352822909269973e-18, -1.265760980031232e-34, 4.5171609366277505e-51 },
      { 0.19697021421566607, -1.2819222913547043e-17, -7.061913383153047e-35, 2.002908946794796e-51 },
      { -0.17271385234050185, 1.0512982520162375e-17, -6.0568631067324936e-34, 1.524236178214315e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 9
      { 0.0690957533591923, 3.863111725481836e-18, -1.7407741023710015e-34, -3.25021439083372e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.6342479767288541, -1.6863489844299227e-17, 3.935229095221158e-34, -1.7608658229824102e-50 },
      { -0.16119757522460407, -1.3139696561013005e-17, -7.105891745354981e-34, 3.016509112462971e-50 },
      { 0.13865030945882525, 4.711321025650907e-19, -4.675780921083092e-36, -2.229895658603977e-52 },
      { 0.9409286140357562, 4.16831424134591e-17, 1.642410055084629e-33, 4.90712843935553e-50 },
      { 0.21163632648194397, 1.1398660449338332e-17, 1.0527436920176656e-35, 6.24544966083015e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 10
      { 0.1835569968390454, -3.1942841978022834e-18, -1.3268440419462044e-34, 5.736184287696735e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -2.4687680843155926, 1.2724357279252323e-16, -3.362901558325251e-33, -1.9029338935070292e-49 },
      { -0.29128688781630047, 9.309140152127047e-18, 5.809644193915391e-34, -1.84029143274582e-51 },
      { -0.026473020233117376, 4.3247465966218974e-19, -1.4318273955012294e-35, 2.0103502354023667e-52 },
      { 2.8478387641928005, -1.4120901429993237e-17, 6.329904845219972e-34, 1.1069853675291458e-50 },
      { 0.2813873314698498, 4.038267296750919e-18, 5.427826689908849e-36, -5.669842308851793e-53 },
      { 0.12374489986331466, -3.66540473567281e-18, 3.4533226204033037e-34, -1.2177499957454098e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 11
      { -1.2154248173958881, 8.968229594217733e-17, 1.6952470734594256e-33, -1.4877950832063884e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 16.672608665945774, -1.1083793675005522e-15, -7.322952761458726e-32, 4.4270793377731104e-48 },
      { 0.915741828416818, 1.2485087919209452e-17, -1.7421391855387534e-34, -3.0008141483532916e-52 },
      { -6.056605804357471, -2.891617388202682e-16, -1.4083223380834355e-32, 1.88193857642937e-49 },
      { -16.00357359415618, 1.5553731265449219e-15, -2.3812136541864074e-32, 2.461922912163167e-49 },
      { 14.849303086297663, -6.321689735901507e-16, -3.4122843390402117e-32, 4.6689952822979985e-49 },
      { -13.371575735289849, -7.37359270841095e-16, 3.4934551269462483e-32, 7.987441748064901e-49 },
      { 5.134182648179638, -6.330688849723096e-17, 1.3235238896928598e-33, 4.423271541705328e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 12
      { 0.2588609164382643, -2.6268870059543483e-17, -6.439309016894105e-34, 2.831289883929919e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -4.7744857854892055, 4.340456147864016e-16, -1.8442642918772008e-32, 8.198619232330141e-50 },
      { -0.4350930137770325, 1.2736059730365673e-17, 2.6723735367257606e-34, -1.3030652979191969e-50 },
      { -3.0494833320722416, 1.4224644883124253e-16, 7.946136856325412e-33, -6.5311548544352845e-49 },
      { 5.5779200399360995, -4.09867930090708e-16, -2.1958536752647138e-32, 1.3461742764371556e-48 },
      { 6.15583158986104, 2.3922810708867838e-17, -6.036239333443348e-34, 1.6824825318891742e-51 },
      { -5.062104586736939, 3.877507422185132e-16, -1.1179244274002401e-32, -5.0163262177943216e-49 },
      { 2.193926173180679, 1.0624101686315504e-16, -7.191616504281861e-36, 2.8253697948929832e-52 },
      { 0.13462799865933495, -4.672078213200417e-18, -5.977852282228673e-35, 4.680873556205574e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // row 13
      { 0.8224275996265075, 1.0151978295805894e-17, 4.604538681537877e-35, 1.9714276153047397e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -11.658673257277664, 8.554164470660502e-17, -5.924989052354781e-33, 3.2503729327807256e-49 },
      { -0.7576221166909362, -4.215643190258618e-17, 2.997932603330136e-33, 8.349359205113489e-50 },
      { 0.7139735881595816, -3.6754038519363424e-17, -2.6491816035664892e-33, 3.428073442723074e-50 },
      { 12.075774986890057, -7.618015642748613e-17, 2.4209871553989445e-33, 9.098950830350395e-50 },
      { -2.127659113920403, 2.0707574856902548e-16, -2.239007847713876e-34, -1.1637665555803625e-50 },
      { 1.9901662070489554, -2.545308073809086e-18, -1.0676137332590066e-34, 3.9907320255645996e-51 },
      { -0.23428647154404028, -1.0930898288012542e-17, -5.898859853756288e-34, 3.518079794017134e-50 },
      { 0.17589857770794226, 4.575339717965544e-18, -2.072922238021916e-34, -7.371360386002512e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
};
const MultiDoubleCoef kDopri78B[2][13] = {
    { // b, 8th order
      { 0.041747491141530244, 2.68471419286714e-18, 1.1942512924970686e-34, -9.693064295023475e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.05545232861123931, 2.746985496909287e-18, 7.659704894661783e-35, -5.235058878512545e-51 },
      { 0.2393128072011801, 8.151420286401371e-18, -4.75395880585749e-34, 2.6532168477385627e-50 },
      { 0.703510669403443, 4.328150648852634e-17, -1.4387132649350449e-33, -1.4095769282261243e-50 },
      { -0.7597596138144609, -4.045011591994921e-17, -5.094952270492675e-34, -3.8070368182808807e-50 },
      { 0.6605630309222863, 4.223901749300027e-17, 3.628686398066598e-34, -1.8094877808475783e-50 },
      { 0.15818748251012332, 1.2674959227707835e-17, -1.3245867758797916e-34, -4.727318535457591e-51 },
      { -0.2381095387528628, 4.999345677516475e-18, -3.139437011096268e-34, 1.2052364377887758e-51 },
      { 0.25, 0.0, 0.0, 0.0 }
    },
    { // b, 7th order
      { 0.029553213676353496, 1.1771998029592105e-18, -4.20811520858817e-35, 1.490241321136727e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.828606276487797, 1.7451000794233212e-17, -4.548900457482285e-34, 2.105183309710463e-50 },
      { 0.3112409000511183, 8.968321405231069e-18, -2.707502434470271e-34, -1.403198687797432e-50 },
      { 2.467345190599887, 9.401071971453215e-17, -3.0725728593195607e-33, 1.322946812272192e-49 },
      { -2.546941651841909, 5.879884287695143e-17, -2.7363425814001714e-33, -8.920354321076386e-50 },
      { 1.4435485836767752, 5.0586152005861605e-17, 1.6969079348257847e-33, 1.3879510383066166e-49 },
      { 0.07941559588112729, -3.125637054305847e-19, -2.3902683450861054e-35, -7.488245135088957e-52 },
      { 0.044444444444444446, -1.6961740653995447e-18, -5.563797617118681e-35, -5.226739877513627e-51 },
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

// RK45 и DOP853: см. codegen.hpp. Сгенерировано скриптом уточнения
// (точные дроби RK45; DOP853 — Ньютон по условиям порядка); руками не править.
const MultiDoubleCoef kRK45A[7][6] = {
    { // stage 0
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 1
      { 0.2, -1.1102230246251566e-17, 6.162975822039155e-34, -3.4211388289180106e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 2
      { 0.075, 2.7755575615628915e-18, -1.5407439555097888e-34, 8.552847072295027e-51 },
      { 0.225, -5.551115123125783e-18, 3.0814879110195775e-34, -1.7105694144590053e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 3
      { 0.9777777777777777, 3.20731096002823e-17, 3.013010401885809e-33, 9.883289950207585e-50 },
      { -3.7333333333333334, 5.921189464667501e-17, 8.217301096052206e-34, 1.1403796096393368e-50 },
      { 3.5555555555555554, 1.9737298215558337e-16, 1.0956401461402941e-32, 6.082024584743129e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 4
      { 2.9525986892242035, 8.440470121673953e-17, 2.7240710080648605e-34, -1.602762608655195e-50 },
      { -11.595793324188385, -8.739643160469453e-16, -1.1993427114128323e-32, -6.882949632939756e-51 },
      { 9.822892851699436, 2.382554517104436e-16, -3.8174567506122735e-33, 2.1045139938291557e-49 },
      { -0.2908093278463649, -9.975254885178018e-18, -2.113503368326185e-34, 7.274026316629514e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 5
      { 2.8462752525252526, -7.177199351112124e-17, 6.1007235410084565e-33, -2.8336705451644127e-49 },
      { -10.757575757575758, 2.153159805333637e-16, -5.976218978947059e-33, 1.6587339776572172e-49 },
      { 8.906422717743473, -2.562801843769753e-16, -1.955426367168055e-32, -1.2732608929123482e-48 },
      { 0.2784090909090909, 2.5232341468753557e-17, 8.404057939144301e-34, 6.220252416214565e-50 },
      { -0.2735313036020583, -2.3518446576536335e-17, 1.0042584958725895e-34, 2.860729295193019e-51 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 6 = b (FSAL)
      { 0.09114583333333333, 4.625929269271485e-18, 2.5679065925163143e-34, 1.425474512049171e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.44923629829290207, -3.341641628476437e-18, 8.340505239844094e-35, 1.9211246793115513e-52 },
      { 0.6510416666666666, 3.700743415417188e-17, 2.0543252740130515e-33, 1.1403796096393367e-49 },
      { -0.322376179245283, -1.7805463602478925e-17, -8.721192200998804e-34, -2.259242622870384e-50 },
      { 0.13095238095238096, -6.6084703846735505e-18, -3.6684379893090205e-34, -2.0363921600702442e-50 }
    }
};

const MultiDoubleCoef kRK45B[2][7] = {
    { // b, 5th order
      { 0.09114583333333333, 4.625929269271485e-18, 2.5679065925163143e-34, 1.425474512049171e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.44923629829290207, -3.341641628476437e-18, 8.340505239844094e-35, 1.9211246793115513e-52 },
      { 0.6510416666666666, 3.700743415417188e-17, 2.0543252740130515e-33, 1.1403796096393367e-49 },
      { -0.322376179245283, -1.7805463602478925e-17, -8.721192200998804e-34, -2.259242622870384e-50 },
      { 0.13095238095238096, -6.6084703846735505e-18, -3.6684379893090205e-34, -2.0363921600702442e-50 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // b^, 4th order
      { 0.08991319444444444, 3.0222737892573704e-18, 1.8317733693283044e-34, -3.516170463054622e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.4534890685834082, 2.4605122438532968e-19, 1.5657193273552415e-35, 1.3129286245861693e-51 },
      { 0.6140625, 4.4408920985006264e-17, -2.465190328815662e-33, 1.3684555315672042e-49 },
      { -0.2715123820754717, -1.7093245058379767e-17, -1.3302725170590176e-33, 8.094543568798463e-50 },
      { 0.08904761904761904, 5.4982473600483945e-18, -1.8782402505262186e-34, -1.6291137280561954e-52 },
      { 0.025, -1.3877787807814458e-18, 7.703719777548944e-35, -4.276423536147513e-51 }
    }
};

const MultiDoubleCoef kRK45P[7][4] = {
    { // stage 0
      { 1.0, 0.0, 0.0, 0.0 },
      { -2.8535800653862835, 3.68427257320943e-17, -2.4976913004863847e-33, -9.960466584444954e-50 },
      { 3.0717434641059005, -2.2171518808087614e-16, -3.2219184950794364e-33, -2.569425121668356e-49 },
      { -1.1270175653862835, 3.68427257320943e-17, -2.4976913004863847e-33, -9.960466584444954e-50 }
    },
    { // stage 1
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 2
      { 0.0, 0.0, 0.0, 0.0 },
      { 4.023133379230305, -4.167878105681829e-16, -4.406275417401954e-33, 2.0532854406233223e-49 },
      { -6.249321565289, -2.900139700026965e-16, 1.9931378732966193e-32, -1.2651733454824425e-48 },
      { 2.675424484351598, -1.8184827063948802e-17, -4.194446969453574e-34, -9.068970148836875e-51 }
    },
    { // stage 3
      { 0.0, 0.0, 0.0, 0.0 },
      { -3.7324019615885042, 2.1710335226270377e-16, -9.750291501216877e-34, -8.01719707531775e-50 },
      { 10.068970589843675, 1.579122419413426e-16, 1.0167359396295582e-32, -4.098458633133134e-49 },
      { -5.685526961588504, 2.1710335226270377e-16, -9.750291501216877e-34, -8.01719707531775e-50 }
    },
    { // stage 4
      { 0.0, 0.0, 0.0, 0.0 },
      { 2.5548038301849423, 3.8217959547573824e-17, -1.0077827653079225e-33, 2.0872818676513115e-50 },
      { -6.399112377351017, 7.438683141996796e-17, -1.4729113497836767e-33, 3.894159917805892e-50 },
      { 3.5219323679207912, 1.4714550158626843e-16, -7.635888838067013e-33, 8.865009736262464e-50 }
    },
    { // stage 5
      { 0.0, 0.0, 0.0, 0.0 },
      { -1.3744241142186024, -9.763400827762438e-17, -4.011545171095598e-33, -9.58069897082707e-50 },
      { 3.272657752246729, -1.6423277237099238e-16, -1.1933212319649876e-32, 4.5227217590553265e-49 },
      { -1.7672812570757455, 6.096928095454083e-17, 4.7927060032460515e-33, -2.9130063707501414e-49 }
    },
    { // stage 6
      { 0.0, 0.0, 0.0, 0.0 },
      { 1.3824689317781436, 2.131763784040944e-19, -5.4068229809333387e-36, -6.388356965324184e-53 },
      { -3.764937863556287, -4.263527568081888e-19, 1.0813645961866677e-35, 1.2776713930648367e-52 },
      { 2.382468931778144, -2.218314285466272e-16, -5.590603661703917e-33, -2.9273161932474868e-49 }
    }
};

const MultiDoubleCoef kDop853A[12][11] = {
    { // stage 0
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
    { // stage 1
      { 0.05260015195876773, 2.2355514829149407e-18, -1.5346033954957451e-34, 2.0824181446335972e-51 },
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
    { // stage 2
      { 0.0197250569845379, -8.963916698837045e-19, 8.689711849795225e-35, -1.8918579058545967e-51 },
      { 0.0591751709536137, -2.6891750096511132e-18, 6.809836105513315e-35, 5.0154851228049927e-51 },
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
    { // stage 3
      { 0.02958758547680685, -1.3445875048255566e-18, 3.4049180527566575e-35, 2.5077425614024963e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.08876275643042054, 2.9051313894305585e-18, 1.0214754158269974e-34, -3.1678311561612936e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 4
      { 0.2413651341592667, -6.624152203917114e-18, 3.7195947639330404e-34, 2.8658886821464074e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.8845494793282861, 7.016246096573064e-18, -6.865175212618015e-34, -2.342299703753789e-50 },
      { 0.924834003261792, 1.0537371249772059e-17, -1.2978746834372656e-34, 6.249714573633984e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 5
      { 0.037037037037037035, 2.05596856412066e-18, 1.1412918188961397e-34, 6.335442275774093e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.17082860872947386, 8.175883949864987e-18, -9.48970137532143e-35, -1.2025200548370281e-51 },
      { 0.12546768756682242, 8.271864563100296e-18, -5.328134866396626e-34, -1.226029478118292e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 6
      { 0.037109375, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.17025221101954405, -8.13951006236759e-18, -2.2418086556732926e-34, 3.0131460987444247e-51 },
      { 0.06021653898045596, 1.2006161584603617e-18, 3.158787112860567e-35, 2.3323833214399666e-51 },
      { -0.017578125, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 7
      { 0.03709200011850479, 3.4253328451426237e-18, 7.018827961872075e-35, 2.924305353552585e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.17038392571223998, 1.296714833161161e-17, -4.472680244810507e-35, 1.9919810634508965e-51 },
      { 0.10726203044637328, 4.772385209398961e-18, -3.3102162741455476e-34, 1.8843663222867612e-50 },
      { -0.015319437748624402, 2.8475137341764543e-19, 1.2518929741958412e-35, 4.4563732100976746e-52 },
      { 0.008273789163814023, -3.660555130834923e-19, -1.0663116882160401e-35, 6.307946796208615e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 8
      { 0.6241109587160757, 2.492689915630563e-17, -8.354066106086308e-34, -3.155547974346869e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -3.3608926294469414, 1.5456576283941592e-16, 1.6727450922588503e-33, 1.3660069404578421e-49 },
      { -0.868219346841726, -3.7185825878655457e-17, -1.8328556600571346e-33, 1.5924328294219774e-49 },
      { 27.59209969944671, -1.2881343059163851e-15, 5.743358655564372e-32, -2.3936136714692698e-49 },
      { 20.154067550477894, -4.800642775601842e-16, 2.5367606342703624e-32, 9.323950035508549e-49 },
      { -43.48988418106996, 1.9413089553812144e-15, -1.3218405243958356e-31, -2.7222788833363994e-48 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 9
      { 0.47766253643826434, 2.572426291265571e-17, 7.455879655083232e-34, 2.0855900906272878e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -2.4881146199716677, 7.616377660551136e-17, -2.141782295916322e-33, -9.286696546849666e-51 },
      { -0.590290826836843, -2.1870450269553572e-17, -6.96431429491923e-34, 1.722876240336273e-50 },
      { 21.230051448181193, 1.1824507909843233e-15, -8.699853450050349e-32, 1.381819029792882e-48 },
      { 15.279233632882423, 1.6175344620753758e-16, -2.9754929303213652e-33, -2.789104185536866e-50 },
      { -33.28821096898486, 3.943900496164246e-16, -4.235359212016942e-33, -4.798614723362035e-50 },
      { -0.020331201708508627, 7.661055475764539e-19, 2.4774482823797087e-35, 5.73441695382292e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 10
      { -0.9371424300859873, -2.6601638178283907e-17, 1.2101135652694934e-33, 1.8149204463773746e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 5.186372428844064, -8.94257554040937e-17, 6.0950838910364114e-33, 4.522106025438129e-50 },
      { 1.0914373489967295, 7.207295585395831e-17, -1.1574799412418217e-33, -3.1006832542073335e-50 },
      { -8.149787010746927, 6.712606849485751e-16, -4.6760313000172983e-32, -1.1351785616858875e-48 },
      { -18.52006565999696, -7.798463984469645e-16, 3.3191525688987114e-32, -1.5206489600118135e-48 },
      { 22.739487099350505, -8.796132218075435e-16, -2.2632263384641856e-33, 4.178624486663252e-50 },
      { 2.4936055526796523, 5.248197812836309e-17, 2.1273204336628297e-33, 2.6092507566695615e-50 },
      { -3.0467644718982196, 1.3907396197551349e-16, -1.1372164323911402e-32, -2.0576243196696017e-49 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 11
      { 2.273310147516538, 1.7007512072370436e-16, -9.623762376420605e-33, -4.418859932203722e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -10.53449546673725, -7.400733880608563e-16, 2.8578654784978565e-32, -1.3296293721646003e-48 },
      { -2.0008720582248625, -2.82313110997892e-17, -1.3563867990410358e-33, -1.766869940206593e-50 },
      { -17.9589318631188, 3.818960836761564e-16, -9.018534674593708e-33, -9.73010227056346e-50 },
      { 27.94888452941996, 1.1856623515434772e-16, -9.085146062912505e-33, -4.9612011171824415e-49 },
      { -2.8589982771350235, -1.7007630722635566e-16, 3.736902005291964e-33, -7.737552357969245e-50 },
      { -8.87285693353063, -2.245736640821424e-16, -1.5996375490302907e-32, -7.88542117966847e-49 },
      { 12.360567175794303, -1.3158566470101767e-16, -6.360109834503327e-33, 4.8796725825069295e-49 },
      { 0.6433927460157636, -4.21309191591411e-17, 6.358309813860974e-34, 2.3644519372355375e-50 }
    }
};

const MultiDoubleCoef kDop853B[12] = { // b, 8th order
    { 0.054293734116568765, -2.5645259176078743e-18, 1.1726163637567311e-34, 1.0112299441244706e-50 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 4.450312892752409, -2.7370927755819525e-16, -2.3348564287333256e-32, -8.75536358529298e-49 },
    { 1.8915178993145003, 6.581122499260077e-17, 4.460977827996999e-33, 5.241959163583826e-50 },
    { -5.801203960010585, 1.3317442893008658e-16, -1.2093375898820709e-32, -9.207245908385643e-50 },
    { 0.3111643669578199, -1.0104455449341981e-17, 6.025163981682236e-34, -4.1345362442930254e-50 },
    { -0.1521609496625161, 8.776696451847702e-18, 4.6991965332279296e-35, -4.460101458911855e-52 },
    { 0.20136540080403034, 6.1225721600273116e-18, 9.70517520995187e-35, -7.284356622076802e-51 },
    { 0.04471061572777259, 3.1043973515104815e-18, 7.263347374039298e-35, 2.6484189541480008e-51 }
};

const MultiDoubleCoef kDop853E5[13] = { // b - b^(5), stage 12 = f(y_{n+1})
    { 0.01312004499419488, 6.877543760739182e-19, 4.6937030111314477e-35, 2.5300343862993303e-51 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { -1.2251564463762044, -8.518996614596023e-17, -4.5966459348612166e-33, -2.934588664047051e-49 },
    { -0.4957589496572502, 1.1503308488869991e-17, -2.148491805765752e-34, 1.1176774457940957e-52 },
    { 1.6643771824549864, 8.898233545173895e-17, 3.439017373455576e-33, 1.2048277004127268e-49 },
    { -0.35032884874997366, -1.6932962897432775e-17, 1.2032700631929538e-33, 8.717125495163365e-51 },
    { 0.3341791187130175, 3.412709883408671e-18, 8.381013260313865e-35, 2.7944043278423933e-51 },
    { 0.08192320648511571, 2.5584664709960105e-18, -1.5134756359128273e-35, -4.733368283134708e-52 },
    { -0.022355307863886294, -1.5521986757409153e-18, 5.359527243393703e-35, -1.0697813676703599e-51 },
    { 0.0, 0.0, 0.0, 0.0 }
};

const MultiDoubleCoef kDop853E3[13] = { // b - b^(3)
    { -0.18980075407240762, 7.925770378062896e-18, 2.628437424080941e-34, -9.249460663360176e-51 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0 },
    { 4.450312892752409, -2.7370927755819525e-16, -2.3348564287333256e-32, -8.75536358529298e-49 },
    { 1.8915178993145003, 6.581122499260077e-17, 4.460977827996999e-33, 5.241959163583826e-50 },
    { -5.801203960010585, 1.3317442893008658e-16, -1.2093375898820709e-32, -9.207245908385643e-50 },
    { -0.42268232132379197, 2.2627811149169426e-18, 3.774305221158832e-36, -2.0843102375072065e-52 },
    { -0.1521609496625161, 8.776696451847702e-18, 4.6991965332279296e-35, -4.460101458911855e-52 },
    { 0.20136540080403034, 6.1225721600273116e-18, 9.70517520995187e-35, -7.284356622076802e-51 },
    { 0.022651792198360825, 1.0635462033024733e-18, -5.198552266113404e-35, -1.7537817448273804e-51 },
    { 0.0, 0.0, 0.0, 0.0 }
};

const MultiDoubleCoef kDop853AExt[3][15] = {
    { // stage 13
      { 0.056167502283047954, -1.88644555359609e-18, -1.6920115343993582e-34, 9.68242491378623e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.25350021021662483, -2.1693122810297358e-17, -1.239382791072797e-33, -6.198500802155098e-50 },
      { -0.2462390374708025, 1.2121494224161007e-17, -3.8466471017763857e-34, -1.917614504596044e-52 },
      { -0.12419142326381637, 6.293125460594507e-18, -2.7490349886147472e-34, 2.0967920186003992e-50 },
      { 0.15329179827876568, 1.3218335854066822e-17, -4.580995107693475e-34, -1.5324828580326159e-52 },
      { 0.00820105229563469, -7.959644410071743e-19, -3.9992017842713925e-35, -1.680959455753614e-51 },
      { 0.007567897660545699, 3.8130025355849374e-19, 1.9780236896081142e-35, 4.1661475186279074e-52 },
      { -0.008298, -1.794120407794253e-19, 3.542170353717004e-36, 2.6436850300463926e-52 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 14
      { 0.03183464816350214, -1.9866654610511737e-18, -6.02278263700697e-35, -2.3455217573569616e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.028300909672366776, -4.0327406802622954e-19, -2.037231005004932e-35, -1.0652471098183434e-51 },
      { 0.053541988307438566, 1.2699877652684217e-18, -3.7263579579533447e-35, -3.2615764046129827e-52 },
      { -0.05492374857139099, 3.015077888147616e-18, 1.9859055406731e-35, 8.203128195515007e-54 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.00010834732869724932, -1.6268392119904694e-21, -3.513511088573477e-38, -3.2106222422273074e-55 },
      { 0.0003825710908356584, 2.070642294190223e-20, 4.924651597261943e-38, -4.9845919691181606e-54 },
      { -0.00034046500868740456, -4.8166271401306394e-21, -3.7554939433508125e-37, 6.971512692250605e-54 },
      { 0.1413124436746325, 1.7199776914665478e-18, -3.851253230740418e-37, -2.242689014111791e-53 },
      { 0.0, 0.0, 0.0, 0.0 }
    },
    { // stage 15
      { -0.42889630158379194, 1.5618605757625628e-17, -1.3388987343533495e-33, 7.294060384342185e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -4.697621415361164, 1.1142707166495955e-16, -5.034038440225204e-33, -1.046642734477419e-50 },
      { 7.683421196062599, -5.92973577603934e-17, 2.1981638180371223e-35, -6.639532153233303e-52 },
      { 4.06898981839711, -1.8994109982627546e-16, 4.615241470610001e-33, 2.5397468482741046e-49 },
      { 0.3567271874552811, 1.7527083212753864e-17, 1.3167323146705442e-33, 2.1902256185714365e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -0.0013990241651590145, -1.0435935893777387e-19, -3.345319876888507e-36, 1.1106105363673204e-52 },
      { 2.9475147891527724, -9.080327027287263e-17, -1.9556240698256254e-33, 1.4285730057139738e-49 },
      { -9.15095847217987, -4.3692612746328837e-16, -2.0377933238709514e-32, 2.6190013324322404e-49 }
    }
};

const MultiDoubleCoef kDop853D[4][16] = {
    { // d4
      { -8.428938276109013, 6.07001772311276e-16, -2.79600512078912e-32, 2.6753601505775603e-48 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.5667149535193777, 4.085083132246998e-17, -1.536760991785711e-33, 6.522637683150082e-50 },
      { -3.0689499459498917, -1.3480870726737455e-17, 6.014161631421085e-34, -7.945132898667396e-51 },
      { 2.38466765651207, -8.217286869152e-17, 3.6833605963618406e-33, -3.3666995012098066e-50 },
      { 2.117034582445028, 2.086504301138e-16, 4.5588142576489396e-33, 2.7449128394218643e-50 },
      { -0.871391583777973, -3.1344855853238036e-18, -7.063319657002988e-35, -1.5006609418827384e-51 },
      { 2.2404374302607883, -6.705197588280659e-17, 3.0481089014366786e-33, -2.0792213163274346e-50 },
      { 0.6315787787694688, 4.4331899416815995e-17, 2.5332815965847335e-34, -4.797502770838777e-51 },
      { -0.08899033645133331, -3.5435308668958646e-18, -2.816230536026599e-34, -4.64078362308914e-52 },
      { 18.148505520854727, 6.873877271126774e-16, 3.329853995378474e-32, 4.041229803579634e-49 },
      { -9.194632392478356, 5.533128061827524e-16, -2.837817638602394e-32, -1.7006518243778594e-48 },
      { -4.436036387594894, -3.6232834900443103e-16, 1.96828290931694e-32, -1.1633832275374736e-48 }
    },
    { // d5
      { 10.427508642579134, 2.5756180718540317e-16, -1.748464374438624e-33, 5.7905349840125045e-50 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 242.28349177525817, 1.2619950687919397e-14, 4.16110250999749e-31, 3.987825855976933e-47 },
      { 165.20045171727028, -1.3399247147160801e-15, 3.822994159642477e-32, 2.2562096134893693e-48 },
      { -374.5467547226902, -7.145990220191793e-15, -6.083979438829103e-31, -2.9498418868562617e-47 },
      { -22.113666853125306, -2.5837691116326833e-16, 7.969892121609336e-33, 8.090024269101403e-50 },
      { 7.733432668472264, -1.000846077165442e-17, -4.942337847361957e-34, 7.849221018844824e-51 },
      { -30.674084731089398, -5.89065475830282e-16, -1.4521852951233064e-32, -4.109794835106153e-49 },
      { -9.332130526430229, 6.835214192289661e-16, -3.678555838006444e-32, -1.4113566434494886e-48 },
      { 15.697238121770845, -6.572034421526076e-16, 2.058225310402236e-32, 4.817808686893378e-49 },
      { -31.139403219565178, 5.273923982966446e-16, 7.158019339286387e-33, 2.3097329315110793e-49 },
      { -9.35292435884448, 8.712970565344274e-16, -4.010679981507997e-32, -2.344112127322613e-48 },
      { 35.81684148639408, 1.2580947935545244e-15, 2.838990758337474e-32, -7.97234128833612e-49 }
    },
    { // d6
      { 19.985053242002433, 8.826808429334292e-16, 2.468409885016677e-33, -7.533968562901507e-51 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -387.0373087493518, 1.4528391673698836e-14, 2.5387690338938347e-31, -1.6478157545323074e-47 },
      { -189.17813819516758, 8.06767469151173e-15, -5.435801746678478e-31, 7.87672875107738e-49 },
      { 527.8081592054236, 1.3720495297384635e-14, -1.4378797374563408e-31, -1.0356855154786328e-47 },
      { -11.57390253995963, -7.608570167042261e-16, 4.3151302051853423e-32, -1.9126979537264915e-48 },
      { 6.8812326946963, -2.6131419487549096e-16, 2.3575990142282554e-32, 4.573735233540581e-49 },
      { -1.0006050966910838, -6.736550672641282e-17, 2.9872592920328013e-33, 2.5974675841319056e-50 },
      { 0.7777137798053443, 3.572781834715124e-17, -2.4865273282172878e-33, 1.079929461660864e-49 },
      { -2.778205752353508, -2.110530566458063e-16, -9.038086788251233e-33, 3.463680566590541e-49 },
      { -60.19669523126412, 1.8248028942949262e-15, -9.26242617193829e-35, 2.5441813942970323e-51 },
      { 84.32040550667716, -5.929910464094051e-16, -2.1605818150308992e-32, -5.761405417587309e-49 },
      { 11.99229113618279, 2.6278928135977475e-17, 8.389406245414651e-34, 7.565195008579152e-50 }
    },
    { // d7
      { -25.69393346270375, 1.3333738395969295e-15, -1.974157555796191e-32, -8.805035974120892e-49 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { 0.0, 0.0, 0.0, 0.0 },
      { -154.18974869023643, -5.9211399582358015e-15, -1.699409170471317e-31, 3.9588271345895877e-48 },
      { -231.5293791760455, -5.608383362985495e-16, 1.9709252144284378e-32, -1.090169682164949e-48 },
      { 357.6391179106141, 5.737366523638755e-15, 1.698464450440262e-31, 2.932630865023776e-48 },
      { 93.40532418362432, -5.988439273072516e-15, 3.0586093184750795e-31, -1.484652723395078e-47 },
      { -37.45832313645163, -9.054105230765388e-16, -9.521683672187523e-32, -1.030057140489342e-48 },
      { 104.0996495089623, -6.184417850693204e-16, 2.9667108345788673e-32, -2.380920927037188e-48 },
      { 29.8402934266605, 1.5710388591321508e-15, -6.682199069182589e-32, -3.417978986666703e-48 },
      { -43.53345659001114, -2.325046131602548e-15, -1.3172005033134027e-31, -8.421612394642856e-48 },
      { 96.32455395918828, 4.052941400678487e-15, 2.7224728323752824e-31, -1.152583365091759e-47 },
      { -39.17726167561544, 1.4295794480996634e-15, 1.2080410994477676e-32, 2.643345418872729e-49 },
      { -149.72683625798564, 1.2853156972603791e-14, -3.6872758261522275e-31, 7.841312822002661e-48 }
    }
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
static const char* kDaeNeedsDaeScheme =
    "this is a DAE system: pick a DAE method (or tick 'Solve as ODE' in Library -> DAE)";

static std::string dae_scheme_body(const System& s, DaeScheme sch) {
    switch (sch) {
    case DaeScheme::BDF1:        return scheme_implicit_common(s, ImplicitKind::Euler, s.dae_nx, false);
    case DaeScheme::Trapezoidal: return scheme_implicit_common(s, ImplicitKind::Euler, s.dae_nx, true);
    case DaeScheme::TRBDF2:      return dae_tr_bdf2(s);
    case DaeScheme::SDIRK2:      return dae_sdirk2(s);
    case DaeScheme::ImplicitMidpoint: return scheme_implicit_common(s, ImplicitKind::Midpoint, s.dae_nx, false);
    case DaeScheme::Radau5:      return dae_radau5(s);
    case DaeScheme::CDIndex2:    return dae_cd2(s);
    case DaeScheme::BDF1GGL: case DaeScheme::Radau5GGL: break;   // codegen_dae_scheme
    case DaeScheme::ROS2:        return dae_ros2(s);
    case DaeScheme::ROS3P:       return dae_rosenbrock(s, ros3p_tab());
    case DaeScheme::Rodas4:      return dae_rosenbrock(s, rodas4_tab());
    case DaeScheme::CD:          return dae_cd(s, false);
    case DaeScheme::CDYoshida4:  return dae_cd(s, true);
    case DaeScheme::ExplicitEuler: {
        static const double A[] = { 0 }, b[] = { 1 }, c[] = { 0 };
        return dae_explicit_rk(s, 1, A, b, c);
    }
    case DaeScheme::ExplicitMidpoint: {
        static const double A[] = { 0, 0, 0.5, 0 }, b[] = { 0, 1 }, c[] = { 0, 0.5 };
        return dae_explicit_rk(s, 2, A, b, c);
    }
    case DaeScheme::RK4: {
        static const double A[] = { 0, 0, 0, 0,  0.5, 0, 0, 0,  0, 0.5, 0, 0,  0, 0, 1, 0 };
        static const double b[] = { 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 }, c[] = { 0, 0.5, 0.5, 1 };
        return dae_explicit_rk(s, 4, A, b, c);
    }
    }
    throw std::runtime_error("unknown DAE scheme");
}

std::string codegen_dae_scheme(const System& s, DaeScheme sch) {
    if (s.dae_nx < 0 || s.dae_nx > (int)s.vars.size())
        throw std::runtime_error("DAE method on a system without a DAE split");
    // h == 0 — проекция на связь: x не меняется, z решает g(x, z) = 0 (BDF1 при h = 0, Ньютон
    // стартует с z из X). Её зовут LLE/LS (ucudaDaeProject): клон x + eps*v получает
    // согласованную z, иначе у методов, берущих f(x_n, z_n) (трапеция, TR-BDF2, ROS2, CD), первый
    // шаг блока нёс бы дефект O(h*eps) и показатели смещались на O(h/NT). Индекс >= 2: z из связи
    // при фиксированном x не определяется (dg/dz = 0) — проекции нет, h == 0 ничего не делает.
    const int N = (int)s.vars.size();
    std::string proj;
    try {
        dae_require_index1(jac_over(s, "X", system_uses_time(s) ? kStepTimeArg : ""), N, s.dae_nx, "projection");
        proj = scheme_implicit_common(s, ImplicitKind::Euler, s.dae_nx, false);
    } catch (const std::runtime_error&) {}
    // Индекс 3 (этап 5): трапеции и средняя точка не L-устойчивы и расходятся (двойной маятник), Rosenbrock —
    // методы индекса 1 (теряют порядок уже на индексе 2). Для индекса 3 — Radau IIA, BDF-подобные или GGL.
    if (sch == DaeScheme::Trapezoidal || sch == DaeScheme::ImplicitMidpoint || sch == DaeScheme::ROS2
        || sch == DaeScheme::ROS3P || sch == DaeScheme::Rodas4 || sch == DaeScheme::CDIndex2) {
        const int idx = dae_structural_index(s);
        if (idx >= 3)
            throw std::runtime_error("this method does not handle index 3 (constraint on positions): use DAE Radau "
                                     "IIA, BDF1, SDIRK2, TR-BDF2 or a '+ GGL' method");
        // На индексе 1 z полушага постоянна и связь в конце шага держится лишь в среднем (Kepler: |g| ~ 3e-3)
        if (sch == DaeScheme::CDIndex2 && idx == 1)
            throw std::runtime_error("DAE CD (index 2) is for constraints without z; on index 1 use DAE CD");
    }
    std::string body;
    if (sch == DaeScheme::BDF1GGL || sch == DaeScheme::Radau5GGL) {
        // GGL: решается система индекса 2 над [x; z; mu]; mu — локальные (0 в начале шага, на решении 0),
        // X снаружи прежний. Тело метода над расширенной системой — в блоке с X, затенённым на копию.
        System g;
        std::string why;
        if (!dae_ggl_system(s, g, &why)) throw std::runtime_error(why);
        const int K = (int)g.vars.size() - N;
        std::ostringstream o;
        o << "    /* GGL: index 3 -> 2, " << K << " local multiplier(s) mu */\n";
        o << "    numb ucuda_xg[" << N + K << "];\n";
        for (int i = 0; i < N; ++i) o << "    ucuda_xg[" << i << "] = X[" << i << "];\n";
        for (int i = N; i < N + K; ++i) o << "    ucuda_xg[" << i << "] = (numb)0.0;\n";
        // Тело над [x; z; mu] пишет в ucuda_xg[], а не в затенённый X: CPU-путь (krs_cpu_check_indices)
        // сверяет индексы X[i] с размерностью состояния, и mu за её пределами он отверг бы.
        std::string inner = dae_scheme_body(g, sch == DaeScheme::BDF1GGL ? DaeScheme::BDF1 : DaeScheme::Radau5);
        {
            std::string r;
            auto idc = [](char c) { return std::isalnum((unsigned char)c) || c == '_'; };
            for (size_t i = 0; i < inner.size(); ++i) {
                if (inner[i] == 'X' && (i == 0 || !idc(inner[i - 1])) && i + 1 < inner.size() && inner[i + 1] == '[') {
                    r += "ucuda_xg";
                    continue;
                }
                r += inner[i];
            }
            inner.swap(r);
        }
        o << "    {\n" << inner << "    }\n";
        for (int i = 0; i < N; ++i) o << "    X[" << i << "] = ucuda_xg[" << i << "];\n";
        body = o.str();
    } else body = dae_scheme_body(s, sch);
    return "if (h == (numb)0) {\n" + proj + "\n} else {\n" + body + "\n}\n"
           + dae_nx_tag(s.dae_nx);
}

namespace {
    // AST -> текст правой части, который снова прочтёт Parser (с явными скобками и
    // операциями; степень — ^{..} в LaTeX-режиме, ^(..) в обычном).
    std::string pn_print(const PN& n, bool latex) {
        auto p = [latex](const PN& x) { return pn_print(x, latex); };
        switch (n->kind) {
        case Node::Num: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", std::fabs(n->num));
            return std::signbit(n->num) ? "(-" + std::string(buf) + ")" : std::string(buf);
        }
        case Node::Sym: return n->name;
        case Node::Neg: return "(-" + p(n->a) + ")";
        case Node::Add: return "(" + p(n->a) + " + " + p(n->b) + ")";
        case Node::Sub: return "(" + p(n->a) + " - " + p(n->b) + ")";
        case Node::Mul: return "(" + p(n->a) + " * " + p(n->b) + ")";
        case Node::Div: return "(" + p(n->a) + " / " + p(n->b) + ")";
        case Node::Pow: return "(" + p(n->a) + ")^" + (latex ? "{" + p(n->b) + "}" : "(" + p(n->b) + ")");
        case Node::Call: {
            std::string o = n->name + "(";
            for (size_t k = 0; k < n->args.size(); ++k) o += (k ? ", " : "") + p(n->args[k]);
            return o + ")";
        }
        }
        return "0";
    }
    // Копия дерева с заменой символов из m.
    PN pn_subst(const PN& n, const std::map<std::string, PN>& m) {
        if (!n) return n;
        if (n->kind == Node::Sym) { auto it = m.find(n->name); return it != m.end() ? it->second : n; }
        auto c = std::make_shared<Node>(*n);
        c->a = pn_subst(n->a, m);
        c->b = pn_subst(n->b, m);
        for (auto& x : c->args) x = pn_subst(x, m);
        return c;
    }

    // ---------- Этап 5: массив производных связей (скрытые связи индекса 2-3) ----------
    // Полная производная выражения c вдоль x' = f: sum_i dc/dx_i f_i + dc/dt. Берётся только от
    // выражений без z (z тогда входят лишь через f).
    PN pn_total_dt(const PN& c, const System& s, const std::vector<PN>& f) {
        PN r = pn_num(0.0);
        for (int i = 0; i < s.dae_nx; ++i)
            if (pn_contains_var(c, s.vars[(size_t)i])) r = jd_add(r, pn_mul(pn_diff(c, s.vars[(size_t)i]), f[(size_t)i]));
        if (pn_contains_var(c, "t")) r = jd_add(r, pn_diff(c, "t"));
        return r;
    }
    // Для связи k: lv[k] = { g_k, d/dt g_k, ... } до первого выражения с z (оно последнее). Длина — индекс
    // связи: 1 — z прямо в g, 2 — скорость (маятник индекса 2), 3 — позиция (маятник в декартовых).
    // Больше трёх производных — отказ (why).
    bool dae_deriv_array(const System& s, std::vector<std::vector<PN>>& lv, std::string* why) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        if (nx < 0 || (int)s.rhs.size() != N) { if (why) *why = "not a DAE system"; return false; }
        std::vector<PN> f((size_t)nx);
        for (int i = 0; i < nx; ++i) { Parser p(s.rhs[(size_t)i], s.latex); f[(size_t)i] = p.parse(); }
        auto has_z = [&](const PN& c) {
            for (int j = nx; j < N; ++j) if (pn_contains_var(c, s.vars[(size_t)j])) return true;
            return false;
        };
        lv.assign((size_t)nz, {});
        for (int k = 0; k < nz; ++k) {
            Parser p(s.rhs[(size_t)(nx + k)], s.latex);
            PN c = p.parse();
            lv[(size_t)k].push_back(c);
            while (!has_z(c)) {
                if ((int)lv[(size_t)k].size() > 3) {
                    if (why) *why = "constraint " + std::to_string(k + 1) + " still has no algebraic variable after 3 "
                                    "time derivatives (index > 3, or a constraint that does not involve z)";
                    return false;
                }
                c = pn_total_dt(c, s, f);
                if (pn_is_zero(c)) {
                    if (why) *why = "constraint " + std::to_string(k + 1) + " has a zero time derivative: it does not "
                                    "determine any algebraic variable";
                    return false;
                }
                lv[(size_t)k].push_back(c);
            }
        }
        return true;
    }
    // Плотная система A x = b (n x n, по строкам) Гауссом с выбором главного; false — вырождена.
    bool dense_solve(std::vector<double> A, std::vector<double>& b, int n) {
        double amax = 0;
        for (double v : A) amax = std::fmax(amax, std::fabs(v));
        if (!(amax <= 1e300)) return false;
        for (int k = 0; k < n; ++k) {
            int p = k;
            for (int i = k + 1; i < n; ++i) if (std::fabs(A[(size_t)i * n + k]) > std::fabs(A[(size_t)p * n + k])) p = i;
            if (!(std::fabs(A[(size_t)p * n + k]) > 1e-13 * (amax > 0 ? amax : 1.0))) return false;
            if (p != k) { for (int j = 0; j < n; ++j) std::swap(A[(size_t)k * n + j], A[(size_t)p * n + j]); std::swap(b[(size_t)k], b[(size_t)p]); }
            for (int i = k + 1; i < n; ++i) {
                const double m = A[(size_t)i * n + k] / A[(size_t)k * n + k];
                for (int j = k; j < n; ++j) A[(size_t)i * n + j] -= m * A[(size_t)k * n + j];
                b[(size_t)i] -= m * b[(size_t)k];
            }
        }
        for (int k = n - 1; k >= 0; --k) {
            double v = b[(size_t)k];
            for (int j = k + 1; j < n; ++j) v -= A[(size_t)k * n + j] * b[(size_t)j];
            b[(size_t)k] = v / A[(size_t)k * n + k];
        }
        return true;
    }
}

int dae_structural_index(const System& s) {
    std::vector<std::vector<PN>> lv;
    if (s.dae_nx < 0) return -1;
    try { if (!dae_deriv_array(s, lv, nullptr)) return 0; } catch (const std::exception&) { return 0; }
    int m = 1;
    for (const auto& l : lv) m = std::max(m, (int)l.size());
    return m;
}

bool dae_stabilized_index1(const System& s, double alpha, System& out, std::string* why) {
    std::vector<std::vector<PN>> lv;
    if (!dae_deriv_array(s, lv, why)) return false;
    out = s;
    const int nx = s.dae_nx;
    for (size_t k = 0; k < lv.size(); ++k) {
        const int m = (int)lv[k].size() - 1;   // число производных
        if (m == 0) continue;
        // (d/dt + alpha)^m g = sum_j C(m, j) alpha^(m-j) c_j = 0: связь затухает как e^(-alpha t) (кратный корень)
        PN e = lv[k][(size_t)m];
        double binom = 1.0;
        for (int j = m - 1; j >= 0; --j) {
            binom = binom * (j + 1) / (m - j);   // C(m, j) из C(m, j + 1)
            e = jd_add(e, pn_mul(pn_num(binom * std::pow(alpha, m - j)), lv[k][(size_t)j]));
        }
        out.rhs[(size_t)nx + k] = pn_print(e, s.latex);
    }
    out.dae_index.clear();
    return true;
}

bool dae_ggl_system(const System& s, System& out, std::string* why) {
    std::vector<std::vector<PN>> lv;
    if (!dae_deriv_array(s, lv, why)) return false;
    const int N = (int)s.vars.size(), nx = s.dae_nx;
    std::vector<int> pos;   // связи уровня позиций (индекс 3)
    for (size_t k = 0; k < lv.size(); ++k) if (lv[k].size() == 3) pos.push_back((int)k);
    if (pos.empty()) { if (why) *why = "GGL is for index 3: no constraint needs two time derivatives to reach z"; return false; }
    out = s;
    std::vector<std::string> mu;
    for (size_t q = 0; q < pos.size(); ++q) {
        std::string nm = "ucuda_ggl_mu" + std::to_string(q + 1);
        mu.push_back(nm);
        out.vars.push_back(nm);
    }
    // x' = f - (dg/dx)^T mu (только связи уровня позиций), 0 = g, 0 = d/dt g (вдоль исходного f)
    for (int i = 0; i < nx; ++i) {
        Parser p(s.rhs[(size_t)i], s.latex);
        PN e = p.parse();
        for (size_t q = 0; q < pos.size(); ++q) {
            const PN& g = lv[(size_t)pos[q]][0];
            if (!pn_contains_var(g, s.vars[(size_t)i])) continue;
            PN m = mk(Node::Sym); m->name = mu[q];
            e = pn_sub(e, pn_mul(pn_diff(g, s.vars[(size_t)i]), m));
        }
        out.rhs[(size_t)i] = pn_print(e, s.latex);
    }
    for (size_t q = 0; q < pos.size(); ++q) out.rhs.push_back(pn_print(lv[(size_t)pos[q]][1], s.latex));
    out.dae_index.assign(out.vars.size(), 1);
    for (size_t j = (size_t)nx; j < out.vars.size(); ++j) out.dae_index[j] = 2;
    (void)N;
    return true;
}

bool dae_index1_structural(const System& s) {
    const int N = (int)s.vars.size(), nx = s.dae_nx;
    if (nx < 0 || (int)s.rhs.size() != N) return false;
    for (int k = nx; k < N; ++k) {
        Parser p(s.rhs[k], s.latex);
        const PN g = p.parse();
        bool any = false;
        for (int j = nx; j < N && !any; ++j) any = pn_contains_var(g, s.vars[j]);
        if (!any) return false;
    }
    return true;
}

bool dae_reduce_to_ode(const System& s, System& ode, std::vector<std::string>* z_exprs, std::string* why) {
    const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
    if (nx < 0 || (int)s.rhs.size() != N) { if (why) *why = "not a DAE system"; return false; }
    std::vector<PN> g(nz);
    for (int k = 0; k < nz; ++k) { Parser p(s.rhs[nx + k], s.latex); g[k] = p.parse(); }
    std::vector<std::string> zn(s.vars.begin() + nx, s.vars.end());
    std::map<std::string, PN> sol;            // z -> выражение через x
    std::vector<char> used(nz, 0);
    for (int round = 0; round < nz; ++round) {
        bool progress = false;
        for (int k = 0; k < nz; ++k) {
            if (used[k]) continue;
            const PN gk = pn_subst(g[k], sol);
            int cnt = 0, which = -1;
            for (int j = 0; j < nz; ++j)
                if (!sol.count(zn[j]) && pn_contains_var(gk, zn[j])) { ++cnt; which = j; }
            if (cnt == 0) {
                if (why) *why = "constraint " + std::to_string(k + 1) + " contains no algebraic variable "
                                "(index >= 2): it cannot be solved for z";
                return false;
            }
            if (cnt != 1) continue;
            PN coef, rem;
            if (!cd_try_extract_linear(gk, zn[which], coef, rem) || pn_is_zero(coef)) continue;
            // g = coef*z + rem = 0  =>  z = -rem / coef
            sol[zn[which]] = pn_div(pn_neg(rem), coef);
            used[k] = 1;
            progress = true;
        }
        if (!progress) break;
    }
    if ((int)sol.size() != nz) {
        for (int j = 0; j < nz; ++j)
            if (!sol.count(zn[j])) {
                if (why) *why = "'" + zn[j] + "' enters its constraint nonlinearly (or with other unknowns): "
                                "no explicit z(x); use a DAE method";
                return false;
            }
    }
    // Выражения z ссылаются только на x: решённые раньше уже подставлены.
    ode = System{};
    ode.vars.assign(s.vars.begin(), s.vars.begin() + nx);
    ode.params = s.params;
    ode.latex = s.latex;
    ode.cd_fp_iters = s.cd_fp_iters;
    for (int i = 0; i < nx; ++i) {
        Parser p(s.rhs[i], s.latex);
        ode.rhs.push_back(pn_print(pn_subst(p.parse(), sol), s.latex));
    }
    if (z_exprs) {
        z_exprs->clear();
        for (int j = 0; j < nz; ++j) z_exprs->push_back(pn_print(sol[zn[j]], s.latex));
    }
    return true;
}

// Индекс >= 2 (этап 5): скрытые связи из массива производных. Связи без z (сама g и её производные до
// предпоследней) ограничивают x: x0 проецируется на них Гаусс-Ньютоном с минимальной нормой сдвига
// (x <- x - C^T (C C^T)^-1 c(x), C = dc/dx); последние элементы (с z) решаются Ньютоном по z при этом x.
// Маятник индекса 3: позиция на окружность, скорость по касательной, lambda из второй производной.
static bool dae_consistent_hidden(const System& s, const std::vector<std::vector<PN>>& lv, const double* a, int na,
                                  double t0, double* X, std::string* why) {
    const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
    std::vector<std::string> xc, zc;   // тексты связей уровня x и уровня z
    for (const auto& l : lv) {
        for (size_t j = 0; j + 1 < l.size(); ++j) xc.push_back(pn_print(l[j], s.latex));
        zc.push_back(pn_print(l.back(), s.latex));
    }
    // Вычислитель выражений над X кусками по N строк (у SystemEvaluator rhs и vars одной длины).
    System e;
    e.vars = s.vars;
    e.params = s.params;
    e.latex = s.latex;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + (na < M + 1 ? na : M + 1));
    ae.resize((size_t)M + 1, 0.0);
    if (system_uses_time(s)) { e.params.push_back("t"); ae.push_back(t0); }
    struct Group { std::vector<SystemEvaluator> ev; int K = 0; };
    auto make = [&](const std::vector<std::string>& ex) {
        Group g;
        g.K = (int)ex.size();
        for (int b = 0; b < g.K; b += N) {
            System c = e;
            c.rhs.assign((size_t)N, "0");
            for (int i = 0; i < N && b + i < g.K; ++i) c.rhs[(size_t)i] = ex[(size_t)(b + i)];
            g.ev.emplace_back(c);
        }
        return g;
    };
    std::vector<double> buf((size_t)N), jb((size_t)N * N);
    auto val = [&](Group& g, double* out) {
        for (size_t b = 0; b < g.ev.size(); ++b) {
            g.ev[b].eval(X, ae.data(), buf.data());
            for (int i = 0; i < N && (int)b * N + i < g.K; ++i) out[b * N + i] = buf[(size_t)i];
        }
    };
    auto jac = [&](Group& g, double* J) {   // J[K x N]
        for (size_t b = 0; b < g.ev.size(); ++b) {
            g.ev[b].eval_jacobian(X, ae.data(), jb.data());
            for (int i = 0; i < N && (int)b * N + i < g.K; ++i)
                for (int j = 0; j < N; ++j) J[(b * N + i) * N + j] = jb[(size_t)i * N + j];
        }
    };
    Group gx = make(xc), gz = make(zc);
    const std::vector<double> X0(X, X + N);
    // 1. x на связи уровня x
    const int m = gx.K;
    double shift = 0;
    if (m > 0) {
        std::vector<double> r((size_t)m), J((size_t)m * N), A((size_t)m * m);
        bool ok = false;
        for (int it = 0; it < 60; ++it) {
            val(gx, r.data());
            double rn = 0, xn = 0;
            for (double v : r) rn = std::fmax(rn, std::fabs(v));
            for (int i = 0; i < nx; ++i) xn = std::fmax(xn, std::fabs(X[i]));
            if (!(rn <= 1e300)) break;
            if (rn <= 1e-14 * (1.0 + xn)) { ok = true; break; }
            jac(gx, J.data());
            for (int p = 0; p < m; ++p)
                for (int q = 0; q < m; ++q) {
                    double v = 0;
                    for (int i = 0; i < nx; ++i) v += J[(size_t)p * N + i] * J[(size_t)q * N + i];
                    A[(size_t)p * m + q] = v;
                }
            if (!dense_solve(A, r, m)) {
                if (why) *why = "hidden constraints on x are dependent or singular at x0 (d c/dx C C^T singular)";
                std::copy(X0.begin(), X0.end(), X);
                return false;
            }
            for (int i = 0; i < nx; ++i) {
                double d = 0;
                for (int p = 0; p < m; ++p) d += J[(size_t)p * N + i] * r[(size_t)p];
                X[i] -= d;
            }
        }
        if (!ok) {
            if (why) *why = "projecting x0 onto the hidden constraints did not converge";
            std::copy(X0.begin(), X0.end(), X);
            return false;
        }
        for (int i = 0; i < nx; ++i) shift = std::fmax(shift, std::fabs(X[i] - X0[(size_t)i]));
    }
    // 2. z из последних элементов (Ньютон по z при найденном x)
    std::vector<double> r((size_t)nz), J((size_t)nz * N), A((size_t)nz * nz);
    for (int it = 0; it < 60; ++it) {
        val(gz, r.data());
        double rn = 0;
        for (double v : r) rn = std::fmax(rn, std::fabs(v));
        if (!(rn <= 1e300)) break;
        jac(gz, J.data());
        for (int p = 0; p < nz; ++p)
            for (int q = 0; q < nz; ++q) A[(size_t)p * nz + q] = J[(size_t)p * N + nx + q];
        if (!dense_solve(A, r, nz)) {
            if (why) *why = "the differentiated constraints do not determine z (singular): index above 3 or a "
                            "degenerate configuration";
            std::copy(X0.begin(), X0.end(), X);
            return false;
        }
        double dn = 0, zn = 0;
        for (int q = 0; q < nz; ++q) {
            X[nx + q] -= r[(size_t)q];
            dn = std::fmax(dn, std::fabs(r[(size_t)q]));
            zn = std::fmax(zn, std::fabs(X[nx + q]));
        }
        if (dn <= 1e-13 * (1.0 + zn)) {
            if (why) {
                char b[160];
                std::snprintf(b, sizeof(b), "hidden constraints (index %d): ", (int)std::max_element(lv.begin(), lv.end(),
                    [](const std::vector<PN>& p, const std::vector<PN>& q) { return p.size() < q.size(); })->size());
                *why = b;
                if (shift > 0) { std::snprintf(b, sizeof(b), "x0 projected (max shift %.2e), ", shift); *why += b; }
                *why += "z from the differentiated constraints";
            }
            return true;
        }
    }
    if (why) *why = "Newton for z on the differentiated constraints did not converge";
    std::copy(X0.begin(), X0.end(), X);
    return false;
}

bool dae_consistent_z(const System& s, const double* a, int na, double t0, double* X, std::string* why) {
    const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
    if (nx < 0 || nz <= 0) return true;
    {
        // Индекс >= 2 — через скрытые связи (dg/dz вырожден, обычный путь ниже их не видит).
        std::vector<std::vector<PN>> lv;
        bool deep = false;
        try {
            if (dae_deriv_array(s, lv, nullptr))
                for (const auto& l : lv) deep = deep || l.size() > 1;
        } catch (const std::exception&) {}
        if (deep) return dae_consistent_hidden(s, lv, a, na, t0, X, why);
    }
    // Вычислитель правых частей отказывает ДАУ (не путать g с производной) — здесь он
    // и нужен как набор выражений [f; g]. Явное время — лишним параметром в конце.
    System e = s;
    e.dae_nx = -1;
    const int M = (int)s.params.size();
    std::vector<double> ae(a, a + (na < M + 1 ? na : M + 1));
    ae.resize((size_t)M + 1, 0.0);
    if (system_uses_time(s)) { e.params.push_back("t"); ae.push_back(t0); }
    SystemEvaluator ev(e);
    if (!ev.has_jacobian()) { if (why) *why = "dg/dz is not available (non-differentiable constraint)"; return false; }

    std::vector<double> z0(X + nx, X + N), F(N), J((size_t)N * N), A((size_t)nz * nz), r(nz), zt(nz);
    // max|g| в текущем X (только строки связей)
    auto gnorm = [&]() {
        ev.eval(X, ae.data(), F.data());
        double gn = 0;
        for (int k = 0; k < nz; ++k) gn = std::fmax(gn, std::fabs(F[nx + k]));
        return gn;
    };
    for (int it = 0; it < 100; ++it) {
        const double gn = gnorm();
        double zn = 0;
        for (int k = 0; k < nz; ++k) { r[k] = F[nx + k]; zn = std::fmax(zn, std::fabs(X[nx + k])); }
        if (!(gn <= 1e300)) break;   // NaN/inf в самой точке: дальше нечего уточнять
        ev.eval_jacobian(X, ae.data(), J.data());
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nz; ++j) A[(size_t)i * nz + j] = J[(size_t)(nx + i) * N + nx + j];
        // Гаусс с выбором главного элемента; масштаб для порога вырожденности — max|A|.
        double amax = 0;
        bool finite = true;
        for (double v : A) { amax = std::fmax(amax, std::fabs(v)); finite = finite && (v == v) && std::fabs(v) <= 1e300; }
        if (!finite) break;
        for (int k = 0; k < nz; ++k) {
            int p = k;
            for (int i = k + 1; i < nz; ++i) if (std::fabs(A[(size_t)i * nz + k]) > std::fabs(A[(size_t)p * nz + k])) p = i;
            if (!(std::fabs(A[(size_t)p * nz + k]) > 1e-12 * (amax > 0 ? amax : 1.0))) {
                for (int q = 0; q < nz; ++q) X[nx + q] = z0[q];
                if (why) *why = "dg/dz is singular (index >= 2): z0 kept as given";
                return false;
            }
            if (p != k) { for (int j = 0; j < nz; ++j) std::swap(A[(size_t)k * nz + j], A[(size_t)p * nz + j]); std::swap(r[k], r[p]); }
            for (int i = k + 1; i < nz; ++i) {
                const double m = A[(size_t)i * nz + k] / A[(size_t)k * nz + k];
                for (int j = k; j < nz; ++j) A[(size_t)i * nz + j] -= m * A[(size_t)k * nz + j];
                r[i] -= m * r[k];
            }
        }
        double dn = 0;
        for (int k = nz - 1; k >= 0; --k) {
            double v = r[k];
            for (int j = k + 1; j < nz; ++j) v -= A[(size_t)k * nz + j] * r[j];
            r[k] = v / A[(size_t)k * nz + k];
            dn = std::fmax(dn, std::fabs(r[k]));
        }
        // Демпфирование: шаг Ньютона делится пополам, пока max|g| не уменьшится. Без него
        // экспоненциальная связь (диод, sinh) из далёкого старта перелетала в cosh = inf.
        for (int k = 0; k < nz; ++k) zt[k] = X[nx + k];
        double lam = 1.0;
        for (; lam > 1e-8; lam *= 0.5) {
            for (int k = 0; k < nz; ++k) X[nx + k] = zt[k] - lam * r[k];
            const double gt = gnorm();
            // У корня max|g| стоит на уровне округления и строго не убывает: полный шаг
            // принимается и тогда, иначе дробление не давало выйти по сходимости ниже.
            if (gt <= 1e300 && (gt < gn || gn == 0 || dn <= 1e-12 * (1.0 + zn))) break;
        }
        if (!(lam > 1e-8)) { for (int k = 0; k < nz; ++k) X[nx + k] = zt[k] - r[k]; }
        if (lam == 1.0 && dn <= 1e-14 * (1.0 + zn)) return true;
        if (gn == 0) return true;
    }
    for (int q = 0; q < nz; ++q) X[nx + q] = z0[q];
    if (why) *why = "Newton on g(x0, z) = 0 did not converge: z0 kept as given";
    return false;
}

AdaptiveCode codegen_adaptive_dae(const System& s, DaeScheme sch) {
    if (s.dae_nx < 0) throw std::runtime_error("adaptive DAE method on a system that is not a DAE");
    if (sch != DaeScheme::Rodas4)
        throw std::runtime_error("adaptive step on a DAE: only DAE Rodas4 has an embedded error estimate");
    // Норма ошибки драйвера одна на все переменные; у индекса >= 2 её надо масштабировать h^(k-1)
    // по индексу переменной (как в RADAU5) — этого в драйвере нет, а Rodas4 — метод индекса 1.
    if (s.dae_nx < (int)s.vars.size() && dae_structural_index(s) != 1)
        throw std::runtime_error("adaptive DAE Rodas4: index 1 only (the constraints must determine z)");
    const RosTab& T = rodas4_tab();
    AdaptiveCode c;
    c.rhs = adaptive_rhs_body(s);   // [f; g]: g-строки нужны лишь стартовой оценке шага
    c.emb = dae_rosenbrock(s, T, true);
    c.dprep = ros_dprep_body(s, T);
    c.deval = ros_deval_body(s);
    c.p = 4; c.q = T.q; c.nlo = 1;
    c.wstages = T.S; c.dsize = 4;
    c.emb_rhs = T.S + 1; c.dprep_rhs = 0; c.dense_order = 3;
    return c;
}

const std::vector<std::string>& dae_scheme_names() {
    // Порядок — как в комбо: неявные (любой индекс), затем явные с решением связи (индекс 1).
    static const std::vector<std::string> n = { "DAE BDF1", "DAE Trapezoidal", "DAE TR-BDF2",
        "DAE SDIRK2", "DAE Implicit Midpoint", "DAE Radau IIA (5)", "DAE BDF1 + GGL", "DAE Radau IIA + GGL",
        "DAE Rosenbrock ROS2", "DAE Rosenbrock ROS3P", "DAE Rodas4", "DAE CD", "DAE CD + Yoshida-4", "DAE CD (index 2)",
        "DAE Explicit Euler", "DAE Explicit Midpoint", "DAE RK4" };
    return n;
}

bool dae_scheme_from_name(const std::string& name, DaeScheme* out) {
    static const DaeScheme v[] = { DaeScheme::BDF1, DaeScheme::Trapezoidal, DaeScheme::TRBDF2,
        DaeScheme::SDIRK2, DaeScheme::ImplicitMidpoint, DaeScheme::Radau5, DaeScheme::BDF1GGL, DaeScheme::Radau5GGL,
        DaeScheme::ROS2, DaeScheme::ROS3P, DaeScheme::Rodas4, DaeScheme::CD, DaeScheme::CDYoshida4, DaeScheme::CDIndex2,
        DaeScheme::ExplicitEuler, DaeScheme::ExplicitMidpoint, DaeScheme::RK4 };
    const auto& n = dae_scheme_names();
    for (size_t i = 0; i < n.size(); ++i)
        if (name == n[i]) { if (out) *out = v[i]; return true; }
    return false;
}

// --- Выходы z (System::out_exprs): «Solve as ODE» несёт z = phi(x) в состоянии после x ---
namespace {
    // ОДУ по x: z сняты (их строк в rhs и нет). Уровень B: имена z переезжают в zb_names — f ссылается
    // на них, а build_namemap отдаёт их решению Ньютона.
    System sys_core(const System& s) {
        System c = s;
        if (!s.zb_g.empty() && c.zb_names.empty())
            c.zb_names.assign(s.vars.begin() + (long long)s.rhs.size(), s.vars.end());
        c.vars.resize(s.rhs.size());
        c.out_exprs.clear();
        return c;
    }

    // --- Уровень B: z из g(x, z, t) = 0 Ньютоном (System::zb_g) ---
    // Только схемы, у которых каждое вычисление f идёт через rhs_over (там z решается перед f):
    // прочие (CD, SEMP/SIMP/D, неявные, комплексные) печатают f своим путём и увидели бы старую z.
    bool zb_scheme_ok(Scheme sch) {
        switch (sch) {
        case Scheme::Euler: case Scheme::ExplicitMidpoint: case Scheme::RK4: case Scheme::DOPRI78:
        case Scheme::DOPRI78Legacy: case Scheme::RK45: case Scheme::DOP853:
        case Scheme::GBS: case Scheme::GBS24: case Scheme::GBS246: case Scheme::GBS2468:
        case Scheme::GBS246810: case Scheme::GBS24681012:
            return true;
        default:
            return false;
        }
    }
    const char* const kZbSchemeMsg =
        "Solve as ODE with a nonlinear constraint (z by Newton): explicit RK / DOPRI / GBS and Extr/Comp over "
        "them only - CD, SEMP/SIMP/D, implicit and complex schemes print f their own way; use a DAE method";
    // Преамбула тела: лямбда ucuda_zb(xs, t) — Ньютон по z (полный, якобиан dg/dz символьный, LU с
    // выбором главного; допуск и предел итераций — Ньютоновские слоты a[]). Тёплый старт — z входа
    // X[nx + k]; повторный вызов с тем же x и t — без пересчёта.
    std::string zb_prelude(const System& c, int nx) {
        const int nz = (int)c.zb_g.size();
        System sx = c;
        sx.vars.resize((size_t)nx);
        NameMap nm = build_namemap(sx, "xs", system_uses_time(c) ? "ucuda_tz" : "");
        std::vector<std::string> G(nz), Jz((size_t)nz * nz);
        for (int k = 0; k < nz; ++k) {
            Parser p(c.zb_g[k], c.latex);
            const PN g = p.parse();
            jac_check_differentiable(g);
            G[k] = emit_to_str(g, nm);
            for (int m = 0; m < nz; ++m) Jz[(size_t)k * nz + m] = emit_to_str(pn_diff(g, c.zb_names[m]), nm);
        }
        const int slot = newton_slot(c);
        std::ostringstream o;
        o << "    /* Solve as ODE, level B: z from g(x, z, t) = 0 by Newton before every f */\n";
        o << "    numb ucuda_zb_v[" << nz << "], ucuda_zb_c[" << nx << "], ucuda_zb_tc = (numb)0;\n";
        o << "    bool ucuda_zb_has = false;\n";
        for (int k = 0; k < nz; ++k) o << "    ucuda_zb_v[" << k << "] = X[" << nx + k << "];\n";
        o << "    auto ucuda_zb = [&](const numb* xs, const numb ucuda_tz) {\n"
             "        bool same = ucuda_zb_has && ucuda_tz == ucuda_zb_tc;\n";
        o << "        for (int i = 0; i < " << nx << " && same; ++i) same = xs[i] == ucuda_zb_c[i];\n"
             "        if (same) return;\n";
        o << "        for (int i = 0; i < " << nx << "; ++i) ucuda_zb_c[i] = xs[i];\n"
             "        ucuda_zb_tc = ucuda_tz; ucuda_zb_has = true;\n"
             "        (void)ucuda_tz;\n";
        o << "        const numb zb_tol = a[" << slot << "];\n"
             "        const int zb_max = (int)a[" << slot + 1 << "];\n";
        o << "        numb Gm[" << nz << "], Jmm[" << nz * nz << "];\n"
             "        int ik, ir, ic, ip; numb mx, av, mlt, nrm, zz;\n"
             "        for (int it = 0; it < zb_max; ++it) {\n";
        for (int k = 0; k < nz; ++k) o << "            Gm[" << k << "] = (" << G[k] << ");\n";
        for (int k = 0; k < nz * nz; ++k) o << "            Jmm[" << k << "] = (" << Jz[(size_t)k] << ");\n";
        o << "            for (ik = 0; ik < " << nz << "; ik++) {\n"
             "                ip = ik; mx = fabs(Jmm[ik * " << nz << " + ik]);\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) { av = fabs(Jmm[ir * " << nz << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
             "                if (ip != ik) { for (ic = 0; ic < " << nz << "; ic++) { mlt = Jmm[ik * " << nz << " + ic]; Jmm[ik * " << nz << " + ic] = Jmm[ip * " << nz << " + ic]; Jmm[ip * " << nz << " + ic] = mlt; }"
             " mlt = Gm[ik]; Gm[ik] = Gm[ip]; Gm[ip] = mlt; }\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) {\n"
             "                    mlt = Jmm[ir * " << nz << " + ik] / Jmm[ik * " << nz << " + ik];\n"
             "                    for (ic = ik; ic < " << nz << "; ic++) Jmm[ir * " << nz << " + ic] -= mlt * Jmm[ik * " << nz << " + ic];\n"
             "                    Gm[ir] -= mlt * Gm[ik];\n"
             "                }\n"
             "            }\n"
             "            for (ik = " << nz - 1 << "; ik >= 0; ik--) {\n"
             "                for (ic = ik + 1; ic < " << nz << "; ic++) Gm[ik] -= Jmm[ik * " << nz << " + ic] * Gm[ic];\n"
             "                Gm[ik] /= Jmm[ik * " << nz << " + ik];\n"
             "            }\n"
             "            nrm = 0; zz = 0;\n";
        for (int k = 0; k < nz; ++k)
            o << "            ucuda_zb_v[" << k << "] -= Gm[" << k << "]; nrm += Gm[" << k << "] * Gm[" << k
              << "]; zz += ucuda_zb_v[" << k << "] * ucuda_zb_v[" << k << "];\n";
        o << "            if (nrm < zb_tol * zb_tol || nrm < zb_tol * zb_tol * zz) break;\n"
             "        }\n"
             "    };\n";
        return o.str();
    }
    // После шага: z нового состояния st (момент tname) — в st[nx + k].
    std::string zb_tail(const System& c, const char* st, const std::string& tname, int nx) {
        std::ostringstream o;
        o << "    ucuda_zb(" << st << ", " << (tname.empty() ? "(numb)0" : tname) << ");\n";
        for (size_t k = 0; k < c.zb_g.size(); ++k)
            o << "    " << st << "[" << nx + (int)k << "] = ucuda_zb_v[" << k << "];\n";
        return o.str();
    }
    // Адаптивный шаг: строки z с dz_k/dt = -(sum_i dg_k/dx_i f_i + dg_k/dt) / (dg_k/dz_k) — если каждая
    // связь содержит только свою z (dg/dz диагонален), иначе отказ. Выражения ссылаются на z,
    // rhs_over подставит решение Ньютона в том же x.
    System sys_augmented_b(const System& s) {
        System a = sys_core(s);
        const int nx = (int)s.rhs.size(), nz = (int)s.zb_g.size();
        std::vector<PN> f(nx);
        for (int i = 0; i < nx; ++i) { Parser p(s.rhs[i], s.latex); f[i] = p.parse(); }
        const bool ut = system_uses_time(s);
        for (int k = 0; k < nz; ++k) {
            Parser p(s.zb_g[k], s.latex);
            const PN g = p.parse();
            for (int m = 0; m < nz; ++m)
                if (m != k && pn_contains_var(g, a.zb_names[m]))
                    throw std::runtime_error("adaptive step, Solve as ODE with a nonlinear constraint: constraint "
                        + std::to_string(k + 1) + " couples several z - use a fixed step or a DAE method");
            PN num = pn_num(0.0);
            for (int i = 0; i < nx; ++i) {
                const PN di = pn_diff(g, s.vars[i]);
                if (!pn_is_zero(di)) num = pn_is_zero(num) ? pn_mul(di, f[i]) : pn_add(num, pn_mul(di, f[i]));
            }
            if (ut) {
                const PN dt = pn_diff(g, "t");
                if (!pn_is_zero(dt)) num = pn_is_zero(num) ? dt : pn_add(num, dt);
            }
            a.vars.push_back(a.zb_names[(size_t)k]);
            a.rhs.push_back(pn_print(pn_neg(pn_div(num, pn_diff(g, a.zb_names[(size_t)k]))), s.latex));
        }
        return a;
    }
    void zb_adaptive_finish(AdaptiveCode& c, const System& s) {
        const int nx = (int)s.rhs.size();
        System core = sys_core(s);
        const std::string pre = zb_prelude(core, nx);
        for (std::string* b : { &c.rhs, &c.emb, &c.dprep })
            if (b->find("ucuda_zb(") != std::string::npos) *b = pre + *b;
        c.emb += zb_tail(core, "Y", time_name(s, "(ucuda_t + h)"), nx);
        for (size_t k = 0; k < s.zb_g.size(); ++k)
            c.emb += "    E[" + std::to_string(nx + (int)k) + "] = (numb)0;\n";
    }
    // st[nx + j] = phi_j(st[0..nx-1]) — пересчёт выходов по новому состоянию (время — tname).
    std::string outputs_tail(const System& s, const char* st, const std::string& tname) {
        const int nx = (int)s.rhs.size();
        const NameMap nm = build_namemap(sys_core(s), st, tname);
        std::ostringstream o;
        o << "    // z outputs: phi(x) of the new state (passive: not in f)\n";
        for (size_t j = 0; j < s.out_exprs.size(); ++j) {
            Parser p(s.out_exprs[j], s.latex);
            o << "    " << st << "[" << nx + (int)j << "] = " << emit_to_str(p.parse(), nm) << ";\n";
        }
        return o.str();
    }
    // Для адаптивного шага z — строки ОДУ с производной вдоль потока dz/dt = sum dphi/dx_i f_i
    // + dphi/dt: так у них есть F для плотного выхода (Эрмит, стадии схемы). Значение z в узле
    // всё равно перезаписывается phi(Y), а оценка ошибки по z — нулём (adaptive_outputs_tail).
    System sys_augmented(const System& s) {
        System a = sys_core(s);
        const int nx = (int)s.rhs.size();
        std::vector<PN> f(nx);
        for (int i = 0; i < nx; ++i) { Parser p(s.rhs[i], s.latex); f[i] = p.parse(); }
        const bool ut = system_uses_time(s);
        for (size_t j = 0; j < s.out_exprs.size(); ++j) {
            Parser p(s.out_exprs[j], s.latex);
            const PN phi = p.parse();
            jac_check_differentiable(phi);
            PN d = pn_num(0.0);
            for (int i = 0; i < nx; ++i) {
                const PN di = pn_diff(phi, s.vars[i]);
                if (!pn_is_zero(di)) d = pn_is_zero(d) ? pn_mul(di, f[i]) : pn_add(d, pn_mul(di, f[i]));
            }
            if (ut) {
                const PN dt = pn_diff(phi, "t");
                if (!pn_is_zero(dt)) d = pn_is_zero(d) ? dt : pn_add(d, dt);
            }
            a.vars.push_back(s.vars[(size_t)nx + j]);
            a.rhs.push_back(pn_print(d, s.latex));
        }
        return a;
    }
    // Хвост emb: z в Y — phi(Y) (точно на связи), ошибка по z — 0 (регулятор ведёт по x).
    void adaptive_outputs_tail(AdaptiveCode& c, const System& s) {
        const int nx = (int)s.rhs.size();
        c.emb += outputs_tail(s, "Y", time_name(s, "(ucuda_t + h)"));
        for (size_t j = 0; j < s.out_exprs.size(); ++j)
            c.emb += "    E[" + std::to_string(nx + (int)j) + "] = (numb)0;\n";
    }
    const char* const kOutputsNoComplex =
        "z outputs (Solve as ODE): ExtrZ (complex extrapolation core) is not available - pick another scheme";
}

std::string codegen_scheme(const System& s, Scheme sch) {
    if (s.dae_nx >= 0) throw std::runtime_error(kDaeNeedsDaeScheme);
    // Уровень B: преамбула с Ньютоном по z, шаг по x (каждое f сначала решает z), z нового
    // состояния в X[nx + k]; метка nx — как у выходов (проекция клона = пересчёт его z).
    if (!s.zb_g.empty() && s.vars.size() > s.rhs.size()) {
        if (!zb_scheme_ok(sch)) throw std::runtime_error(kZbSchemeMsg);
        const int nx = (int)s.rhs.size();
        const System core = sys_core(s);
        return zb_prelude(core, nx) + codegen_scheme(core, sch)
             + zb_tail(core, "X", time_name(s, "(ucuda_t + h)"), nx) + dae_nx_tag(nx);
    }
    // Выходы z: шаг по x, затем z = phi(x нового состояния); метка nx — LLE/LS возмущают только x,
    // а проекция клона (шаг с h = 0) просто пересчитывает его z.
    if (!s.out_exprs.empty())
        return codegen_scheme(sys_core(s), sch) + outputs_tail(s, "X", time_name(s, "(ucuda_t + h)"))
             + dae_nx_tag((int)s.rhs.size());
    // CD fast выносит в предрасчёт всё, что не зависит от X, — с явным t это были бы константы
    // одного момента. Схема та же, что CD (FP), только без предрасчёта: с t она им и считается.
    if (system_uses_time(s) && (sch == Scheme::CDFast || sch == Scheme::CDFastFP))
        sch = (sch == Scheme::CDFast) ? Scheme::CD : Scheme::CDFP;
    if (!scheme_supports_time(sch) && system_uses_time(s))
        throw std::runtime_error(kTimeUnsupportedMsg);
    if (const int K = gbs_stage_count(sch)) return scheme_gbs(s, K);
    switch (sch) {
    case Scheme::Euler:            return scheme_euler(s);
    case Scheme::EulerCromer:      return scheme_euler_cromer(s);
    case Scheme::ExplicitMidpoint: return scheme_midpoint(s);
    case Scheme::RK4:              return scheme_rk4(s);
    case Scheme::DOPRI78:          return scheme_dopri78(s);
    case Scheme::DOPRI78Legacy:    return scheme_dopri78(s, true);
    case Scheme::RK45:             return scheme_erk(s, &kRK45A[0][0], 6, &kRK45B[0][0], 6);
    case Scheme::DOP853:           return scheme_erk(s, &kDop853A[0][0], 11, kDop853B, 12);
    case Scheme::CD:               return scheme_cd(s);
    case Scheme::CDFast:           return scheme_cd_common(s, CdKind::Real, false, false, true);
    case Scheme::CDFP:             return scheme_cd_common(s, CdKind::Real, false, false, false, true);
    case Scheme::CDFastFP:         return scheme_cd_common(s, CdKind::Real, false, false, true,  true);
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

void adaptive_err_weights(const MultiDoubleCoef* b, int S, const MultiDoubleCoef* bh, int lc,
                          MultiDoubleCoef* w) {
    auto two_sum = [](double a, double c, double& err) {
        const double s = a + c, bb = s - a;
        err = (a - (s - bb)) + (c - bb);
        return s;
    };
    for (int j = 0; j < lc; ++j) {
        const MultiDoubleCoef z = {};
        const MultiDoubleCoef& x = j < S ? b[j] : z;
        double e0 = 0, e1 = 0;
        const double s = two_sum(x.c[0], -bh[j].c[0], e0);
        double lo = e0;
        for (int k = 1; k < 4; ++k) lo += x.c[k] - bh[j].c[k];
        const double hi = two_sum(s, lo, e1);
        w[j] = MultiDoubleCoef{ { hi, e1, 0.0, 0.0 } };
    }
}

bool scheme_supports_adaptive(Scheme sch) {
    return sch == Scheme::RK45 || sch == Scheme::DOP853
        || sch == Scheme::DOPRI78 || sch == Scheme::DOPRI78Legacy;
}

AdaptiveCode codegen_adaptive(const System& s, Scheme sch) {
    if (!s.zb_g.empty() && s.vars.size() > s.rhs.size()) {   // уровень B: dz/dt строками, z(Y) Ньютоном
        AdaptiveCode c = codegen_adaptive(sys_augmented_b(s), sch);
        zb_adaptive_finish(c, s);
        return c;
    }
    if (!s.out_exprs.empty()) {   // выходы z: строки с dz/dt, затем z = phi(Y), E_z = 0
        AdaptiveCode c = codegen_adaptive(sys_augmented(s), sch);
        adaptive_outputs_tail(c, s);
        return c;
    }
    if (s.vars.size() != s.rhs.size())
        throw std::runtime_error("vars/rhs size mismatch");
    if (s.dae_nx >= 0) throw std::runtime_error("adaptive step for DAE systems is not implemented yet");
    // Пары RK45 / DOP853 / DOPRI78 печатают время стадий сами (adaptive_emb_body,
    // dop853_dprep_body); сигнатуры функций с ucuda_t ставит модуль по
    // UCUDA_NONAUTONOMOUS (признак — step_body_uses_time(rhs)).
    AdaptiveCode c;
    c.rhs = adaptive_rhs_body(s);
    EmbSpec e{};
    MultiDoubleCoef w[14];   // веса оценки ошибки b - b^ (RK45, DOPRI78); живут до adaptive_emb_body
    switch (sch) {
    case Scheme::RK45:
        adaptive_err_weights(&kRK45B[0][0], 6, &kRK45B[1][0], 7, w);
        e = { &kRK45A[0][0], 6, &kRK45B[0][0], 6, { w, nullptr }, 1, 7 };
        c.p = 5; c.q = 4; c.nlo = 1; c.wstages = 7; c.dsize = 5;
        c.emb_rhs = 6; c.dprep_rhs = 0; c.dense_order = 4;
        c.dprep = rk45_dprep_body(s); c.deval = rk45_deval_body(s);
        break;
    case Scheme::DOP853:
        e = { &kDop853A[0][0], 11, kDop853B, 12, { kDop853E5, kDop853E3 }, 2, 13 };   // E5, E3 — уже b - b^
        c.p = 8; c.q = 7; c.nlo = 2; c.wstages = 16; c.dsize = 8;
        c.emb_rhs = 12; c.dprep_rhs = 3; c.dense_order = 7;
        c.dprep = dop853_dprep_body(s); c.deval = dop853_deval_body(s);
        break;
    case Scheme::DOPRI78:
    case Scheme::DOPRI78Legacy: {
        const bool legacy = (sch == Scheme::DOPRI78Legacy);
        const MultiDoubleCoef (*A)[12] = legacy ? kDopri78LegacyA : kDopri78A;
        const MultiDoubleCoef (*B)[13] = legacy ? kDopri78LegacyB : kDopri78B;
        // f(Y) у DOPRI78 не стадия метода: оценщик z её не весит (lc = 13), а
        // считается она ради плотного выхода и как F0 следующего шага.
        adaptive_err_weights(&B[0][0], 13, &B[1][0], 13, w);
        e = { &A[0][0], 12, &B[0][0], 13, { w, nullptr }, 1, 13 };
        c.p = 8; c.q = 7; c.nlo = 1; c.wstages = 14; c.dsize = 4;
        c.emb_rhs = 13; c.dprep_rhs = 0; c.dense_order = 3;
        c.dprep = hermite_dprep_body(s); c.deval = hermite_deval_body(s);
        break;
    }
    default:
        throw std::runtime_error("scheme has no embedded error estimate (adaptive step needs RK45, DOPRI78 or DOP853)");
    }
    c.emb = adaptive_emb_body(s, e);
    return c;
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
    if (name == "RK45")              return Scheme::RK45;
    if (name == "DOP853")            return Scheme::DOP853;
    if (name == "CD")                return Scheme::CD;
    if (name == "CD fast")           return Scheme::CDFast;
    if (name == "CD FP")             return Scheme::CDFP;
    if (name == "CD fast FP")        return Scheme::CDFastFP;
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
    { "CD fast",                2, true  },  // CD с предрасчётом констант шага
    // Неявная половина простыми итерациями: самосопряжённость лишь до
    // невязки итераций (O(h^(n+1)) при n итерациях), поэтому false.
    // Порядок 2 -- при n >= 2.
    { "CD FP",                  2, false },
    { "CD fast FP",             2, false },
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
    { "RK45",                   5, false },  // Dormand-Prince 5(4), шаг решением 5-го порядка
    { "DOPRI78",                8, false },
    { "DOPRI78 (legacy)",       8, false },  // дроби статьи без уточнения
    { "DOP853",                 8, false },
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
    // Явное время: старт в t, leapfrog z_m — в t + m*hs, сглаживание f(z_n) — в t + h.
    const auto f0 = rhs_over(s, "X0_gbs", time_name(s));
    const auto f1 = rhs_over(s, "Z1_gbs", time_name(s, "(ucuda_t + (numb)m_gbs * hs_gbs)"));
    const auto f1s = rhs_over(s, "Z1_gbs", time_name(s, "(ucuda_t + h)"));

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
    o << "        const numb hs_gbs = h * ((numb)1 / (numb)n_gbs);\n";
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
          << i << "] + hs_gbs * (" << f1s[i] << "));\n";
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
    // Неавтономная база: каждый подшаг получает своё время t0 + s*(h/n) — затенённым
    // ucuda_t (inline) или вторым аргументом лямбды. Иначе все подшаги считались бы в t_n.
    const bool ht = step_body_uses_time(base_body);
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
    if (ht) o << "    const numb ucuda_t_ex = ucuda_t;\n";
    if (inl) {
        o << "    const numb h_ex = h;\n\n";
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_ex = [&](const numb h" << (ht ? ", const numb ucuda_t" : "") << ") {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
    }

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << "\n";
        if (inl) {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) {\n";
            o << "        const numb h = h_ex * ((numb)1 / (numb)" << fmtnum((double)n[k]) << ");\n";
            if (ht) o << "        const numb ucuda_t = ucuda_t_ex + (numb)s_ex * h;\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        } else if (ht) {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) step_ex(h * ((numb)1 / (numb)"
              << fmtnum((double)n[k]) << "), ucuda_t_ex + (numb)s_ex * h * ((numb)1 / (numb)"
              << fmtnum((double)n[k]) << "));\n";
        } else {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) step_ex(h * ((numb)1 / (numb)"
              << fmtnum((double)n[k]) << "));\n";
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
        o << "        const numb h = h_ex * ((numb)1 / (numb)" << fmtnum((double)n[k]) << ");\n";
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

// --- Адаптивный шаг экстраполяторов ----------------------------------------
//
// Вложенная пара из одних и тех же стадий T_1..T_K: старшее решение Y = sum alpha_k T_k
// (веса и порядок — как у постоянного шага), младшее — по первым K-1 стадиям с весами
// beta (порядок extrapolation_order(K-1)). Оценка ошибки E = Y - Yhat = sum (alpha_k -
// beta_k) T_k, beta_K = 0 — по разности весов, как у RK-пар. Это обычная локальная
// экстраполяция: регулятор ведёт по ошибке Yhat (показатель 1/(q+1)), а шаг продолжает Y.
// Y печатается тем же выражением, что в wrap_extrapolation / scheme_gbs, поэтому
// принятый шаг с тем же h совпадает с шагом постоянного Extr / GBS (GBS берёт f(X) из F0
// — то же значение, на вычисление f меньше на стадию). F1 = f(Y): FSAL драйвера и
// эрмитов плотный выход 3-го порядка (как у DOPRI78).
namespace {

// Веса alpha (по всем K стадиям) и разности alpha - beta литералами: alpha — точными
// дробями, когда сходятся (как у постоянного шага), разности — double: это только оценка.
struct ExtrAdWeights {
    std::vector<std::string> alpha, diff;
};

ExtrAdWeights extr_ad_weights(const std::vector<int>& n, int p, bool symmetric) {
    const int K = (int)n.size();
    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    const std::vector<int> nlo(n.begin(), n.end() - 1);
    const std::vector<double> beta = extrapolation_weights(nlo, p, symmetric);
    std::vector<ExRat> arat;
    const bool exact = extrapolation_weights_rational(n, p, symmetric, arat);
    ExtrAdWeights w;
    for (int k = 0; k < K; ++k) {
        w.alpha.push_back(exact ? ex_rat_expr(arat[(size_t)k]) : ("(numb)(" + fmtnum(alpha[k]) + ")"));
        const double b = k < K - 1 ? beta[(size_t)k] : 0.0;
        w.diff.push_back("(numb)(" + fmtnum(alpha[k] - b) + ")");
    }
    return w;
}

// Общая часть AdaptiveCode экстраполятора: правая часть, плотный выход, порядки.
AdaptiveCode extr_ad_code(const System& s, int p, int q, int emb_rhs) {
    AdaptiveCode c;
    c.rhs = adaptive_rhs_body(s);
    c.p = p; c.q = q; c.nlo = 1; c.wstages = 0; c.dsize = 4;
    c.emb_rhs = emb_rhs; c.dprep_rhs = 0; c.dense_order = 3;
    c.dprep = hermite_dprep_body(s); c.deval = hermite_deval_body(s);
    return c;
}

// Хвост emb: Y, E и F1 = f(Y).
void extr_ad_finish(std::ostringstream& o, const System& s, const char* ac, const char* ec) {
    const int N = (int)s.vars.size();
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { Y[i_ea] = " << ac << "[i_ea]; E[i_ea] = "
      << ec << "[i_ea]; }\n";
    const auto fy = rhs_over(s, "Y", time_name(s, "(ucuda_t + h)"));   // F1 = f(t + h, Y)
    for (int v = 0; v < N; ++v) o << "    F1[" << v << "] = (" << fy[(size_t)v] << ");\n";
}

} // namespace

AdaptiveCode codegen_adaptive_extrapolation(const System& s, const std::string& base_body,
                                            const std::vector<int>& n, int p, bool symmetric,
                                            const std::string& base_name, double base_rhs) {
    if (!s.zb_g.empty() && s.vars.size() > s.rhs.size()) {   // база уже несёт свой Ньютон по z
        AdaptiveCode c = codegen_adaptive_extrapolation(sys_augmented_b(s), base_body, n, p, symmetric,
                                                        base_name, base_rhs);
        zb_adaptive_finish(c, s);
        return c;
    }
    if (!s.out_exprs.empty()) {   // база (codegen_scheme) уже пересчитывает z после каждого подшага
        AdaptiveCode c = codegen_adaptive_extrapolation(sys_augmented(s), base_body, n, p, symmetric,
                                                        base_name, base_rhs);
        adaptive_outputs_tail(c, s);
        return c;
    }
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size(), K = (int)n.size();
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs an extrapolation with at least two stages "
                                        "(the lower-order solution comes from the first K-1)");
    if (base_body.empty()) throw std::runtime_error("extrapolation base '" + base_name + "' has no step body");
    const ExtrAdWeights w = extr_ad_weights(n, p, symmetric);
    long long cost = 0;
    for (int v : n) cost += v;
    const int P = extrapolation_order(K, p, symmetric), Q = extrapolation_order(K - 1, p, symmetric);

    std::ostringstream o;
    o << "    // --- adaptive " << make_extrapolation_name(base_name, n) << ": Y over " << K
      << " stages (order " << P << "), estimate over the first " << K - 1 << " (order " << Q << ") ---\n";
    o << "    numb X0_ea[" << N << "], AC_ea[" << N << "], EC_ea[" << N << "];\n";
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { X0_ea[i_ea] = X[i_ea]; AC_ea[i_ea] = (numb)0; "
         "EC_ea[i_ea] = (numb)0; }\n";
    o << "    (void)F0; (void)W;\n";
    o << "    {\n";
    // База шагает по X на месте: локальная копия затеняет константный вход emb.
    o << "    numb XS_ea[" << N << "];\n";
    o << "    numb* const X = XS_ea;\n";
    const bool inl = wrapper_can_inline(base_body);
    // Явное время — как у wrap_extrapolation: подшаг s начинается в t + s*h/n (t попытки —
    // аргумент emb от драйвера), внутренний ucuda_t затеняет его для тела базы.
    const bool ht = step_body_uses_time(base_body);
    if (ht) o << "    const numb ucuda_t_ea = ucuda_t;\n";
    if (inl) {
        o << "    const numb h_ea = h;\n";
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_ea = [&](const numb h" << (ht ? ", const numb ucuda_t" : "") << ") {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n";
    }
    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s") << " of h/" << n[k] << "\n";
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) X[i_ea] = X0_ea[i_ea];\n";
        if (inl) {
            o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) {\n";
            o << "        const numb h = h_ea * ((numb)1 / (numb)" << fmtnum((double)n[k]) << ");\n";
            if (ht) o << "        const numb ucuda_t = ucuda_t_ea + (numb)s_ea * h;\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        } else if (ht) {
            o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) step_ea(h * ((numb)1 / (numb)" << fmtnum((double)n[k])
              << "), ucuda_t_ea + (numb)s_ea * h * ((numb)1 / (numb)" << fmtnum((double)n[k]) << "));\n";
        } else {
            o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) step_ea(h * ((numb)1 / (numb)" << fmtnum((double)n[k]) << "));\n";
        }
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { AC_ea[i_ea] += " << w.alpha[(size_t)k]
          << " * X[i_ea]; EC_ea[i_ea] += " << w.diff[(size_t)k] << " * X[i_ea]; }\n";
    }
    o << "    }\n";
    extr_ad_finish(o, s, "AC_ea", "EC_ea");

    // Вызовов f на попытку: все подшаги всех стадий + F1 = f(Y). Дробная f базы (неявный
    // полушаг CD по компонентам) округляется уже в произведении.
    AdaptiveCode c = extr_ad_code(s, P, Q, base_rhs > 0 ? (int)std::lround((double)cost * base_rhs) + 1 : 1);
    c.emb = o.str();
    return c;
}

// ExtrZ: стадии и обе суммы — ucmplx, ядро базы шагает по Z[] без Re между подшагами, Re
// берётся на выходе (как wrap_extrapolation_complex). Веса — симметричные: разложение Re стадии
// идёт по чётным степеням, поэтому Re младшей комбинации — решение порядка
// extrapolation_order(K-1, p, true), и E = Re sum (alpha - beta) Z — его оценка.
AdaptiveCode codegen_adaptive_extrapolation_complex(const System& s, const std::string& core_body,
                                                    const std::vector<int>& n, int p,
                                                    const std::string& base_name, double base_rhs) {
    if (!s.out_exprs.empty()) throw std::runtime_error(kOutputsNoComplex);
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    if (system_uses_time(s)) throw std::runtime_error(kTimeUnsupportedMsg);
    const int N = (int)s.vars.size(), K = (int)n.size();
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs an extrapolation with at least two stages "
                                        "(the lower-order solution comes from the first K-1)");
    if (core_body.empty()) throw std::runtime_error("ExtrZ base '" + base_name + "' has no complex core");
    const bool symmetric = extrapolation_symmetric(true, false);
    const ExtrAdWeights w = extr_ad_weights(n, p, symmetric);
    const int P = extrapolation_order(K, p, symmetric), Q = extrapolation_order(K - 1, p, symmetric);

    std::ostringstream o;
    o << "    // --- adaptive " << make_extrapolation_name(base_name, n, {}, true) << ": Y = Re over " << K
      << " stages (order " << P << "), estimate over the first " << K - 1 << " (order " << Q << ") ---\n";
    // Z — имя, над которым работает ядро базы.
    o << "    ucmplx Z[" << N << "], Z0_ea[" << N << "], AC_ea[" << N << "], EC_ea[" << N << "];\n";
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) {\n"
      << "        Z0_ea[i_ea] = ucmplx(X[i_ea], 0.0); AC_ea[i_ea] = ucmplx(0.0, 0.0); EC_ea[i_ea] = ucmplx(0.0, 0.0);\n"
      << "    }\n";
    o << "    (void)F0; (void)W;\n";
    // Ядро — встроенная CD-схема, return/goto в нём нет: подставляется всегда.
    o << "    const numb h_ea = h;\n";
    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s") << " of h/" << n[k]
          << ", no Re in between\n";
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) Z[i_ea] = Z0_ea[i_ea];\n";
        o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) {\n";
        o << "        const numb h = h_ea * ((numb)1 / (numb)" << fmtnum((double)n[k]) << ");\n";
        emit_indented(o, core_body, "    ");
        o << "    }\n";
        // Сумма — тем же выражением, что у постоянного ExtrZ (AC = AC + alpha * Z).
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { AC_ea[i_ea] = AC_ea[i_ea] + " << w.alpha[(size_t)k]
          << " * Z[i_ea]; EC_ea[i_ea] = EC_ea[i_ea] + " << w.diff[(size_t)k] << " * Z[i_ea]; }\n";
    }
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { Y[i_ea] = AC_ea[i_ea].re; E[i_ea] = EC_ea[i_ea].re; }\n";
    const auto fy = rhs_over(s, "Y");
    for (int v = 0; v < N; ++v) o << "    F1[" << v << "] = (" << fy[(size_t)v] << ");\n";

    // Как у Extr: подшаги стадий x f базы (комплексное ядро делает те же вызовы) + F1.
    long long cost = 0;
    for (int v : n) cost += v;
    AdaptiveCode c = extr_ad_code(s, P, Q, base_rhs > 0 ? (int)std::lround((double)cost * base_rhs) + 1 : 1);
    c.emb = o.str();
    return c;
}

AdaptiveCode codegen_adaptive_gbs(const System& s, int K) {
    if (!s.zb_g.empty() && s.vars.size() > s.rhs.size()) {
        AdaptiveCode c = codegen_adaptive_gbs(sys_augmented_b(s), K);
        zb_adaptive_finish(c, s);
        return c;
    }
    if (!s.out_exprs.empty()) {
        AdaptiveCode c = codegen_adaptive_gbs(sys_augmented(s), K);
        adaptive_outputs_tail(c, s);
        return c;
    }
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size();
    if (N < 1) throw std::runtime_error("GBS needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs GBS 2-4 or longer (the lower-order solution comes "
                                        "from the first K-1 stages)");
    const std::vector<int> n = gbs_substeps(K);
    const ExtrAdWeights w = extr_ad_weights(n, 2, true);
    // Явное время — как у scheme_gbs: leapfrog z_m в t + m*hs, сглаживание f(z_n) в t + h.
    const auto f1  = rhs_over(s, "Z1_gbs", time_name(s, "(ucuda_t + (numb)m_gbs * hs_gbs)"));
    const auto f1s = rhs_over(s, "Z1_gbs", time_name(s, "(ucuda_t + h)"));
    int cost = 0;
    for (int v : n) cost += v;   // n - 1 leapfrog + сглаживание; f(X) — из F0

    const std::string Ns = std::to_string(N), Ks = std::to_string(K);
    std::ostringstream o;
    o << "    // --- adaptive GBS";
    for (int k = 0; k < K; ++k) o << (k ? "-" : " ") << n[k];
    o << ": Y order " << 2 * K << ", estimate over the first " << K - 1 << " stages (order " << 2 * (K - 1) << ") ---\n";
    o << "    numb X0_gbs[" << Ns << "], AC_gbs[" << Ns << "], EC_gbs[" << Ns << "], Z0_gbs[" << Ns
      << "], Z1_gbs[" << Ns << "], ZT_gbs[" << Ns << "], T_gbs[" << Ns << "];\n";
    o << "    const numb W_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) o << (k ? ", " : "") << w.alpha[(size_t)k];
    o << " };\n";
    o << "    const numb V_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) o << (k ? ", " : "") << w.diff[(size_t)k];
    o << " };\n";
    o << "    (void)W;\n";
    o << "    int i_gbs, m_gbs, k_gbs;\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { X0_gbs[i_gbs] = X[i_gbs]; AC_gbs[i_gbs] = (numb)0; "
         "EC_gbs[i_gbs] = (numb)0; }\n";
    o << "    for (k_gbs = 0; k_gbs < " << Ks << "; ++k_gbs) {\n";
    o << "        const int  n_gbs  = 2 * (k_gbs + 1);\n";
    o << "        const numb hs_gbs = h * ((numb)1 / (numb)n_gbs);\n";
    o << "        // старт: явный Эйлер на первый подшаг, f(X) — F0 драйвера\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) {\n";
    o << "            Z0_gbs[i_gbs] = X0_gbs[i_gbs];\n";
    o << "            Z1_gbs[i_gbs] = X0_gbs[i_gbs] + hs_gbs * F0[i_gbs];\n";
    o << "        }\n";
    o << "        // leapfrog: z_{m+1} = z_{m-1} + 2*hs*f(z_m), всего n подшагов\n";
    o << "        for (m_gbs = 1; m_gbs < n_gbs; ++m_gbs) {\n";
    for (int i = 0; i < N; ++i)
        o << "            ZT_gbs[" << i << "] = Z0_gbs[" << i << "] + (numb)2 * hs_gbs * (" << f1[(size_t)i] << ");\n";
    o << "            for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { Z0_gbs[i_gbs] = Z1_gbs[i_gbs]; Z1_gbs[i_gbs] = ZT_gbs[i_gbs]; }\n";
    o << "        }\n";
    // (numb)0.5 * (...) и W * (0.5 * x) == (W * 0.5) * x точно: множитель — степень двойки.
    o << "        // сглаживание Грэгга: (z_{n-1} + z_n + hs*f(z_n)) / 2\n";
    for (int i = 0; i < N; ++i)
        o << "        T_gbs[" << i << "] = (numb)0.5 * (Z0_gbs[" << i << "] + Z1_gbs[" << i << "] + hs_gbs * ("
          << f1s[(size_t)i] << "));\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { AC_gbs[i_gbs] += W_gbs[k_gbs] * T_gbs[i_gbs]; "
         "EC_gbs[i_gbs] += V_gbs[k_gbs] * T_gbs[i_gbs]; }\n";
    o << "    }\n";
    extr_ad_finish(o, s, "AC_gbs", "EC_gbs");

    AdaptiveCode c = extr_ad_code(s, 2 * K, 2 * (K - 1), cost + 1);
    c.emb = o.str();
    return c;
}

int builtin_scheme_rhs_per_step(const std::string& name) {
    static const struct { const char* name; int k; } kTab[] = {
        { "Euler", 1 }, { "Explicit Midpoint", 2 }, { "RK4", 4 }, { "RK45", 6 },
        { "DOP853", 12 }, { "DOPRI78", 13 }, { "DOPRI78 (legacy)", 13 },
    };
    for (const auto& t : kTab) if (name == t.name) return t.k;
    return 0;
}

// Неявный полушаг CD (и стадия SIMP, и шаг D): по компоненте — одно вычисление f_i, если она
// линейна по своей переменной (аналитика), иначе CD_ITERS простых итераций; в единицах полной f.
// iters -- число итераций нелинейной компоненты (у "CD FP" -- cd_fp_iters; линейная
// у FP тоже одно вычисление: rem и coef считаются один раз).
static double cd_implicit_sweep_rhs(const System& s, int iters = CD_ITERS) {
    const int N = (int)s.vars.size();
    if (N < 1 || (int)s.rhs.size() != N) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        Parser p(s.rhs[(size_t)i], s.latex);
        PN coef, rem;
        sum += cd_try_extract_linear(p.parse(), s.vars[(size_t)i], coef, rem) ? 1.0 : (double)iters;
    }
    return sum / (double)N;
}

double scheme_rhs_per_step(const System& s, const std::string& name) {
    if ((!s.out_exprs.empty() || !s.zb_g.empty()) && s.vars.size() > s.rhs.size())
        return scheme_rhs_per_step(sys_core(s), name);   // z: пересчёт выходов / Ньютон в цене f не учтён
    if (const int k = builtin_scheme_rhs_per_step(name); k > 0) return (double)k;
    ExtrapolationSpec es;
    if (parse_extrapolation_name(name, &es)) {
        long long cost = 0;
        for (int v : es.n) cost += v;
        return (double)cost * scheme_rhs_per_step(s, es.base);   // ExtrZ: ядро базы — те же f
    }
    CompositionSpec cs;
    if (parse_composition_name(name, &cs))
        return (double)cs.gammas.size() * scheme_rhs_per_step(s, cs.base);
    const Scheme sch = scheme_from_name(name);
    if (const int K = gbs_stage_count(sch); K >= 1) {
        int cost = 1;                                 // f(X) на всех стадиях одна
        for (int k = 1; k <= K; ++k) cost += 2 * k;   // n - 1 leapfrog + сглаживание
        return (double)cost;
    }
    try {
        switch (sch) {
        case Scheme::EulerCromer:  return 1.0;
        case Scheme::SEMP:         return 2.0;
        case Scheme::SIMP:         return cd_implicit_sweep_rhs(s) + 1.0;
        case Scheme::D:            return cd_implicit_sweep_rhs(s);
        case Scheme::CD: case Scheme::CD10: case Scheme::CDFast:
        case Scheme::ComplexCD: case Scheme::ComplexCD10:
            return 1.0 + cd_implicit_sweep_rhs(s);
        case Scheme::CDFP: case Scheme::CDFastFP:
            return 1.0 + cd_implicit_sweep_rhs(s, cd_fp_iters_of(s));
        case Scheme::ComplexCD4: case Scheme::ComplexCD4_10:
        case Scheme::ComplexCD4SS01: case Scheme::ComplexCD4SS10:
            return 2.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::ComplexCD4S3: case Scheme::ComplexCD4S3_10:
            return 3.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::ComplexCD4S4: case Scheme::ComplexCD4S4_10:
            return 4.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::Map:          return 1.0;
        default:                   return 0.0;   // Ньютон (Implicit Euler/Midpoint, Complex IE), кастомная КРС
        }
    } catch (const std::exception&) {
        return 0.0;   // правая часть не разбирается — счёт неизвестен, а не ошибка
    }
}

const char* const kBdfSchemeOde = "BDF 1-5";
const char* const kBdfSchemeDae = "DAE BDF 1-5";
bool scheme_is_multistep(const std::string& name) { return name == kBdfSchemeOde || name == kBdfSchemeDae; }

bool adaptive_scheme_name_supported(const std::string& name) {
    if (name == "DAE Rodas4") return true;   // ДАУ: codegen_adaptive_dae
    if (scheme_is_multistep(name)) return true;   // свой драйвер (multistep_bdf.h)
    if (scheme_supports_adaptive(scheme_from_name(name)) && name != "Euler") return true;
    if (gbs_stage_count(scheme_from_name(name)) >= 2) return true;
    ExtrapolationSpec spec;
    return parse_extrapolation_name(name, &spec) && spec.n.size() >= 2;   // Extr и ExtrZ
}

int adaptive_scheme_q_guess(const std::string& name) {
    if (name == "RK45") return 4;
    if (name == "DAE Rodas4") return 3;
    const int K = gbs_stage_count(scheme_from_name(name));
    if (K >= 2) return 2 * (K - 1);
    ExtrapolationSpec spec;
    int p = 0; bool sym = false;
    if (parse_extrapolation_name(name, &spec) && spec.n.size() >= 2 && builtin_scheme_traits(spec.base, &p, &sym))
        return extrapolation_order((int)spec.n.size() - 1, p, extrapolation_symmetric(spec.re_at_output, sym));
    return 7;
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
    // Неавтономная база: стадия k начинается в t0 + h*(gamma_0 + ... + gamma_{k-1}).
    const bool ht = step_body_uses_time(base_body);
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
    // Начало стадии k в долях h: (gamma_0 + ... + gamma_{k-1}), только при явном времени.
    auto t_off = [&](int k) {
        std::string acc = "(numb)0";
        for (int j = 0; j < k; ++j) acc += " + (" + gc[(size_t)j] + ")";
        return acc;
    };
    if (ht) o << "    const numb ucuda_t_co = ucuda_t;\n";
    if (wrapper_can_inline(base_body)) {
        o << "    const numb h_co = h;\n";
        for (int k = 0; k < K; ++k) {
            o << "    {   // stage " << k << "\n";
            o << "        const numb h = h_co * (" << gc[(size_t)k] << ");\n";
            if (ht) o << "        const numb ucuda_t = ucuda_t_co + h_co * (" << t_off(k) << ");\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        }
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_co = [&](const numb h" << (ht ? ", const numb ucuda_t" : "") << ") {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
        for (int k = 0; k < K; ++k) {
            o << "    step_co(h * (" << gc[(size_t)k] << ")";
            if (ht) o << ", ucuda_t_co + h * (" << t_off(k) << ")";
            o << ");\n";
        }
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
    // Превью: явное время печатается как есть, "t" (в шаге оно станет временем стадии).
    return rhs_over(s, "X", system_uses_time(s) ? "t" : "");
}

// --- LaTeX для превью уравнений (редактор ДАУ) ---
std::string name_to_latex(const std::string& n) {
    static const char* const greek[] = { "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
        "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega" };
    // хвост из цифр — индекс: x1 -> x_{1}, A11 -> A_{11}; буквенная основа — греческая, если совпала
    size_t k = n.size();
    while (k > 0 && std::isdigit((unsigned char)n[k - 1])) --k;
    std::string base = n.substr(0, k), idx = n.substr(k);
    if (base.empty()) return n;
    for (const char* g : greek) if (base == g) { base = std::string("\\") + g; break; }
    return idx.empty() ? base : base + "_{" + idx + "}";
}

std::string expr_to_latex(const std::string& expr, bool latex_in, const std::vector<std::string>& user_funcs) {
    Parser p(expr, latex_in);
    if (!user_funcs.empty()) p.extra_funcs = &user_funcs;
    const PN root = p.parse();
    // приоритет: 1 — сумма, 2 — произведение, 3 — унарный минус, 4 — степень, 5 — атом
    std::function<int(const PN&)> prec = [&](const PN& n) {
        switch (n->kind) {
        case Node::Add: case Node::Sub: return 1;
        case Node::Mul: return 2;
        case Node::Neg: return 3;
        case Node::Pow: return 4;
        case Node::Num: return n->num < 0 ? 3 : 5;
        default: return 5;   // Sym, Call, Div (\frac — сама себе скобки)
        }
    };
    std::function<std::string(const PN&)> L;
    auto paren = [&](const PN& n, int need) {
        const std::string s = L(n);
        return prec(n) < need ? "\\left(" + s + "\\right)" : s;
    };
    L = [&](const PN& n) -> std::string {
        switch (n->kind) {
        case Node::Num: {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.10g", n->num);
            return buf;
        }
        case Node::Sym: return n->name == "t" ? "t" : name_to_latex(n->name);
        case Node::Neg: return "-" + paren(n->a, 3);
        case Node::Add: {
            if (n->b->kind == Node::Neg) return L(n->a) + " - " + paren(n->b->a, 2);
            return L(n->a) + " + " + paren(n->b, 1);
        }
        case Node::Sub: return L(n->a) + " - " + paren(n->b, 2);
        case Node::Mul: {
            const std::string a = paren(n->a, 2), b = paren(n->b, 3);
            // число на символ — без точки: 2x; иначе \cdot
            const bool juxt = n->a->kind == Node::Num && (n->b->kind == Node::Sym || n->b->kind == Node::Call
                                                           || n->b->kind == Node::Pow);
            return juxt ? a + " " + b : a + " \\cdot " + b;
        }
        case Node::Div: return "\\frac{" + L(n->a) + "}{" + L(n->b) + "}";
        case Node::Pow: return paren(n->a, 5) + "^{" + L(n->b) + "}";
        case Node::Call: {
            const std::string& f = n->name;
            auto arg = [&](size_t i) { return i < n->args.size() ? L(n->args[i]) : std::string(); };
            if (f == "sqrt") return "\\sqrt{" + arg(0) + "}";
            if (f == "fabs" || f == "abs") return "\\left|" + arg(0) + "\\right|";
            // e^{u} только для короткого показателя: дробь или вызов в степени нечитаемы — тогда \exp(u)
            if (f == "exp" && !n->args.empty() && n->args[0]->kind != Node::Div && n->args[0]->kind != Node::Call
                && n->args[0]->kind != Node::Pow)
                return "e^{" + arg(0) + "}";
            if (f == "pow") return (n->args.empty() ? std::string() : paren(n->args[0], 5)) + "^{" + arg(1) + "}";
            // стандартные функции — командами LaTeX (\sinh, \min), fmin/fmax парсера — обратно в min/max
            static const char* const std_fns[] = { "sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log" };
            std::string head = "\\mathrm{" + f + "}";
            if (f == "fmin" || f == "fmax") head = f == "fmin" ? "\\min" : "\\max";
            else if (f == "log") head = "\\ln";
            else for (const char* s : std_fns) if (f == s) { head = std::string("\\") + s; break; }
            std::string o = head + "\\left(";
            for (size_t i = 0; i < n->args.size(); ++i) o += (i ? ", " : "") + L(n->args[i]);
            return o + "\\right)";
        }
        }
        return "";
    };
    return L(root);
}

bool system_uses_time(const System& s) {
    if (std::find(s.vars.begin(), s.vars.end(), "t") != s.vars.end()
        || std::find(s.params.begin(), s.params.end(), "t") != s.params.end())
        return false;   // t — обычная переменная или параметр
    std::function<bool(const PN&)> has_t = [&](const PN& n) -> bool {
        if (!n) return false;
        if (n->kind == Node::Sym) return n->name == "t";
        if (has_t(n->a) || has_t(n->b)) return true;
        for (const auto& c : n->args) if (has_t(c)) return true;
        return false;
    };
    std::vector<std::string> all = s.rhs;   // и выходы z (их пересчёт берёт t конца шага)
    all.insert(all.end(), s.out_exprs.begin(), s.out_exprs.end());
    all.insert(all.end(), s.zb_g.begin(), s.zb_g.end());   // и связи уровня B
    for (const auto& r : all) {
        Parser p(r, s.latex);
        if (has_t(p.parse())) return true;
    }
    return false;
}

bool scheme_supports_time(Scheme sch) {
    switch (sch) {
    case Scheme::Euler: case Scheme::EulerCromer: case Scheme::ExplicitMidpoint: case Scheme::RK4:
    case Scheme::DOPRI78: case Scheme::DOPRI78Legacy: case Scheme::RK45: case Scheme::DOP853:
    case Scheme::ImplicitEuler: case Scheme::ImplicitMidpoint:
    case Scheme::CD: case Scheme::CD10: case Scheme::CDFP:
    case Scheme::GBS: case Scheme::GBS24: case Scheme::GBS246: case Scheme::GBS2468:
    case Scheme::GBS246810: case Scheme::GBS24681012:
    case Scheme::SEMP: case Scheme::SIMP: case Scheme::D:
        return true;
    default:
        return false;
    }
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