#pragma once
#include "ucuda/equations/codegen.hpp"
#include "ucuda/equations/adaptive_codegen.h"
#include <iomanip>
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

namespace ucuda::equations_detail {
// Реализация парсера (обычный синтаксис + LaTeX) и кодгена схем.


 // внутренняя линковка: всё ниже не видно из других .cpp/.cu

    // Мини-AST
    struct Node {
        enum Kind { Num, Sym, Call, Add, Sub, Mul, Div, Pow, Neg } kind;
        double num = 0; std::string name;
        std::shared_ptr<Node> a, b; std::vector<std::shared_ptr<Node>> args;
    };

    using PN = std::shared_ptr<Node>;

    PN mk(Node::Kind k) ;


    const std::set<std::string>& known_funcs() ;

    const std::map<std::string, std::string>& latex_funcs() ;

    const std::map<std::string, std::string>& latex_symbols() ;


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


    int prec(Node::Kind k) ;

    std::string fmtnum(double v) ;

    void emit(const PN& n, const NameMap& nm, std::ostream& o);

    void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) ;

    void emit(const PN& n, const NameMap& nm, std::ostream& o) ;


    // Зарезервированные математические константы: остаются именем в коде,
    // не идут в a[]. Определяются в шапке ядра отдельно (напр. const double pi = M_PI;).
    const std::set<std::string>& math_constants() ;

    NameMap build_namemap(const System& s, const std::string& st) ;

    std::vector<std::string> rhs_over(const System& s, const std::string& st) ;


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


    bool pn_contains_var(const PN& n, const std::string& v) ;


    PN pn_num(double v) ;

    bool pn_is_zero(const PN& n) ;

    bool pn_is_one (const PN& n) ;


    // Peephole constructors: fold trivial identities so the generated C code
    // stays readable. Only algebraic no-ops (x*1, 1*x, x*0, 0*x, --x, -(Num)),
    // never re-associates. Correctness is unchanged.
    PN pn_neg(PN a) ;

    PN pn_add(PN a, PN b) ;

    PN pn_sub(PN a, PN b) ;

    // Parser emits literal -1 as Neg(Num(1)) rather than Num(-1), so catch
    // both shapes here (my own pn_neg fold produces Num(-1), the parser does
    // not).
    bool pn_is_neg_one(const PN& n) ;

    PN pn_mul(PN a, PN b) ;

    PN pn_div(PN a, PN b) ;


    // Try to split n = coef*v + rem with coef, rem both free of v.
    // Returns false when v enters non-linearly and the analytic solve is
    // unsafe: product v*v, division by an expression that contains v, or v
    // appearing inside a pow-exponent or function call.
    bool cd_try_extract_linear(const PN& n, const std::string& v, PN& coef, PN& rem) ;


    std::string emit_to_str(const PN& n, const NameMap& nm) ;


    // Complex CD only: fmod and atan2 have no complex counterpart (both are
    // defined through the sign/magnitude of REAL arguments), so configCUDA.h
    // deliberately provides no ucmplx overload for them. Catch them here, at
    // codegen time, so the user gets the reason instead of an NVRTC template
    // error from the middle of a generated kernel.
    void cd_check_complex_safe(const PN& n) ;


    // ---------- Symbolic differentiation (implicit schemes) ----------
    //
    // d/dv over the same 9-kind AST the rest of codegen uses. Everything is built
    // through the pn_* peephole constructors above so that x*0, x*1 and -(-x) fold
    // on the way out -- without that a 3x3 Jacobian prints as unreadable noise.
    // pn_add / pn_sub deliberately do NOT fold (the CD emitter depends on their exact
    // shape), so zero-folding for the chain rule lives in the jd_* wrappers instead.

    constexpr double JD_LN2  = 0.69314718055994530942;

    constexpr double JD_LN10 = 2.30258509299404568402;


    PN pn_call1(const char* f, PN u) ;

    PN pn_call2(const char* f, PN u, PN w) ;

    PN pn_pow(PN a, PN b) ;

    PN jd_add(PN a, PN b) ;

    PN jd_sub(PN a, PN b) ;


    PN pn_diff(const PN& n, const std::string& v);


    PN pn_diff_call(const PN& n, const std::string& v) ;


    PN pn_diff(const PN& n, const std::string& v) ;


    // Same role as cd_check_complex_safe: refuse at codegen time, with a readable
    // reason, instead of letting NVRTC fail inside a generated kernel. These three
    // parse fine but have no usable derivative (0 almost everywhere / undefined).
    void jac_check_differentiable(const PN& n) ;


    // Jacobian entries as C strings, row-major: out[i*N + j] = d f_i / d vars[j],
    // emitted over state array st. Counterpart of rhs_over above.
    std::vector<std::string> jac_over(const System& s, const std::string& st) ;


    // Схемы
    std::string scheme_euler(const System& s) ;

    std::string scheme_euler_cromer(const System& s) ;

    std::string scheme_midpoint(const System& s) ;

    std::string scheme_rk4(const System& s) ;

    // Dormand-Prince 8(7), 13-стадийный. Матрицы M[13][12] и B[2][13] — // const, компилятор положит в constant memory. Считаем y (8-й, b[0]) и z
    // (7-й, b[1]) — z пока не используется, но оставлен под будущий адаптивный
    // шаг по |y - z|. На каждый step X := y.
    std::string scheme_dopri78(const System& s) ;


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


    std::string scheme_implicit_common(const System& s, ImplicitKind kind) ;

    std::string scheme_implicit_euler(const System& s)    ;

    std::string scheme_implicit_midpoint(const System& s) ;

    std::string scheme_complex_ieuler(const System& s)    ;


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
    bool needs_paren_after_mul(const PN& n) ;

    std::string wrap_paren(const std::string& s, bool w) ;

    // Absorb a leading minus of a rem/coef into the operator sign: peel one
    // Neg or a negative Num off the node, so that `- h2 * -a[1]` and
    // `+ h2 * -a[1]` come out as `+ h2 * a[1]` and `- h2 * a[1]` -- the
    // shape a person would write by hand, and one less negation for the
    // compiler to chase. Returns the sign character to emit before "h2 * ".
    char peel_sign(PN& n, char pos, char neg) ;

    // "h2 * factor" -> "h2" when factor == 1 (post-peel-sign), same reason
    // as the pn_mul(x, Num(1)) fold: the multiplication reads like noise
    // both to a human and to the peephole compiler expects.
    std::string mul_step(const PN& factor, const std::string& factor_c, const char* hs) ;

    // saved — имя временной под X_saved в итерационной ветке: у Complex CD4 два
    // прохода живут в одной области видимости, поэтому имя приходит снаружи.
    void emit_diag_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                               const std::string& v, const std::string& x,
                               const NameMap& nm, const char* hs, const char* sty,
                               const std::string& saved) ;


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


    std::string scheme_cd_common(const System& s, CdKind kind) ;


    std::string scheme_cd(const System& s)          ;

    std::string scheme_complex_cd(const System& s)  ;

    std::string scheme_complex_cd4(const System& s) ;


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
    std::string scheme_semi_midpoint(const System& s, bool implicit_stage) ;

    std::string scheme_semp(const System& s) ;

    std::string scheme_simp(const System& s) ;


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
    std::string scheme_d(const System& s) ;


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
    PN pn_mul_step(const PN& e) ;


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


    int func_id(const std::string& nm) ;


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


    double apply_func1(int fid, double x) ;

    double apply_func2(int fid, double a, double b) ;


    // Комплексные версии — те же id, те же формулы, что и в device-коде: обе
    // ветки зовут функции из configCUDA.h, поэтому CPU-траектория Complex CD
    // повторяет GPU-шную операция в операцию (в пределах разной группировки FMA).
    // atan2/fmod комплексного смысла не имеют и до сюда не доходят: система с
    // ними отсекается в cd_check_complex_safe ещё на кодгене.
    ucmplx apply_func1(int fid, ucmplx x) ;

    ucmplx apply_func2(int fid, ucmplx a, ucmplx b) ;


    // Степень отдельной функцией: у double это std::pow, у ucmplx — перегрузка
    // из configCUDA.h (со спрямлением целых показателей). Нужна, чтобы
    // run_program остался одним шаблоном на оба типа состояния.
    double  op_pow(double a, double b) ;

    ucmplx  op_pow(ucmplx a, ucmplx b) ;


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
}
