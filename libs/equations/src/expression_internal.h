#pragma once
#include "ucuda/equations/codegen.hpp"
#include "ucuda/domain/sparse_lu.h"
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
#include <complex>
#include <array>
namespace ucuda::equations_detail {
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


    int prec(Node::Kind k) ;

    std::string fmtnum(double v) ;

    // "u / c" с числовым c печатается как "u * ((numb)1 / c)": обратная сворачивается
    // компилятором в литерал (double/float), а деление в FP64 — это ~10 инструкций на
    // конвейере, который у ядер и так узкое место. На dd/qd-ветке (numb = dd/qd)
    // (numb)1 / c считается в рантайме в полной точности — литерал 1/c в double срезал
    // бы её. Расхождение с честным делением — 1 ulp. "c1 / c2" не трогаем: это и так
    // константа, а правильно округлённое c1/c2 точнее c1 * (1/c2). Байткод CPU делает то же.
    bool div_by_reciprocal(const PN& n) ;

    void emit(const PN& n, const NameMap& nm, std::ostream& o);

    void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) ;

    void emit(const PN& n, const NameMap& nm, std::ostream& o) ;


    // Зарезервированные математические константы: остаются именем в коде,
    // не идут в a[]. Определяются в шапке ядра отдельно (напр. const double pi = M_PI;).
    const std::set<std::string>& math_constants() ;

    // tname — во что печатать явное время t (время стадии: "ucuda_t", "ucuda_ts",
    // "(ucuda_t + (numb)0.5 * h)"). Пусто — t не определено, как раньше: схема,
    // которая не печатает время стадий, на t упадёт с "unknown symbol", а не
    // молча посчитает все стадии в один момент. Переменную или параметр с именем
    // t не перекрывает.
    NameMap build_namemap(const System& s, const std::string& st, const std::string& tname = {}) ;

    std::vector<std::string> rhs_over(const System& s, const std::string& st, const std::string& tname = {}) ;


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
    int cd_fp_iters_of(const System& s) ;


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
    // The rule itself lives in cd_find_complex_unsafe: SystemEvaluator asks the same
    // question for the CPU path, which never goes through codegen.
    const std::string* cd_find_complex_unsafe(const PN& n) ;

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
    std::vector<std::string> jac_over(const System& s, const std::string& st, const std::string& tname = {}) ;


    // Схемы
    // Discrete map x_{n+1} = f(x_n): the right-hand side is the step, h unused.
    // The temp buffer is required - updating in place would be Gauss-Seidel.
    std::string scheme_map(const System& s) ;

    // Явное время (этап 2): у неавтономной системы шаг получает t_n аргументом
    // kStepTimeArg, а схема печатает время каждой стадии. У автономной tname
    // пуст, и текст тела прежний до байта.
    std::string time_name(const System& s, const char* stage_expr = nullptr) ;

    std::string scheme_euler(const System& s) ;

    // Эйлер-Кромер обновляет X по месту, все уравнения — в момент t_n.
    std::string scheme_euler_cromer(const System& s) ;

    std::string scheme_midpoint(const System& s) ;

    std::string scheme_rk4(const System& s) ;

    // Dormand-Prince 8(7), 13-стадийный. Матрицы M[13][12] и B[2][13] — static
    // const, компилятор положит в constant memory. Считаем y (8-й, b[0]) и z
    // (7-й, b[1]) — z пока не используется, но оставлен под будущий адаптивный
    // шаг по |y - z|. На каждый step X := y.
    //
    // Коэффициент печатается суммой "(numb)(c0) + (numb)(c1) + ...": в double
    // она сворачивается при компиляции в c0, в dd/qd CPU-ветки вкладки Order
    // складывается точно (two_sum) и несёт 32/62 знака коэффициента.
    std::string md_literal(const MultiDoubleCoef& r) ;

    // Квалификатор таблиц коэффициентов. При numb = float хвосты ~1e-51 ниже денормалей float,
    // NVRTC перестаёт считать инициализатор константным, а static __device__ с динамической
    // инициализацией запрещён — поэтому там таблица обычная локальная. Хвосты остаются: их
    // складывают dd/qd CPU-ветки вкладки Order. При double текст прежний.
    const char* md_table_qual() ;

    // "static const numb <decl> = {...};" — rows x cols из таблицы с шагом
    // строки ld (ld > cols — берётся левый верхний угол).
    void md_table(std::ostringstream& o, const char* decl, const MultiDoubleCoef* v,
                  int rows, int cols, int ld) ;


    // Узлы c_i = sum_j a_ij явного РК (S стадий, строка i с шагом ld) — таблица Cs[S]
    // для времени стадии ucuda_ts = ucuda_t + Cs[i]*h. Печатается только у
    // неавтономной системы. Сумма — в long double по всем частям коэффициента:
    // узлы у этих таблиц — простые дроби, и ulp-ошибка суммы на порядок не влияет.
    void stage_time_table(std::ostringstream& o, const MultiDoubleCoef* A, int S, int ld) ;


    std::string scheme_dopri78(const System& s, bool legacy = false) ;


    // Явный РК с постоянным шагом по таблице Бутчера: RK45 (S = 6) и DOP853
    // (S = 12). A — строки стадий 0..S-1 с шагом ld (строка 0 нулевая,
    // печатаются столбцы 0..S-2), b — S весов старшего решения. Форма кода
    // та же, что у DOPRI78: стадии в цикле, таблицы static const. Оценщики и
    // стадия FSAL здесь не нужны — их считает адаптивный вариант.
    std::string scheme_erk(const System& s, const MultiDoubleCoef* A, int ld,
                           const MultiDoubleCoef* b, int S) ;


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


    std::string adaptive_emb_body(const System& s, const EmbSpec& e) ;


    std::string adaptive_rhs_body(const System& s) ;


    // Плотный выход RK45 — полином Шампайна 4-й степени, форма scipy:
    // y(th) = X + h * sum_j Q_j th^(j+1), Q = K^T P. D = [X, hQ_0..hQ_3].
    std::string rk45_dprep_body(const System& s) ;

    std::string rk45_deval_body(const System& s) ;


    // Плотный выход DOP853 — полином 7-й степени Хайрера, форма scipy
    // (Dop853DenseOutput): три доп. стадии 13..15, затем D = [X, F0..F6].
    std::string dop853_dprep_body(const System& s) ;

    std::string dop853_deval_body(const System& s) ;


    // Кубический Эрмит по концам шага (DOPRI78): D = [X, Y, hF0, hF1],
    // y(th) = X + th*hF0 + th^2*(3d - 2hF0 - hF1) + th^3*(hF0 + hF1 - 2d), d = Y - X.
    std::string hermite_dprep_body(const System& s) ;

    std::string hermite_deval_body(const System& s) ;


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

    bool sparse_lu_build(const std::vector<MatEntry>& pat, int n, const std::vector<char>& df, SparseLu& S) ;

    // Факторизация. Первая фаза (x-диагонали по плану) проверяет порог: |ведущий| >= 1e-3 * max|столбца|
    // среди исключаемых строк и >= 1e-30; не прошёл — fallback = 1 (вызывающий решает всю матрицу плотной
    // LU). Затем блок — плотная LU с выбором главного (перестановки только от столбца k, как везде) в
    // ucuda_pv; вырожденный — sing = 1. Нужны int ik, ir, ic, ip и numb mx, av, dgn, mlt.
    void sparse_lu_factor(std::ostream& o, const SparseLu& S, const char* ind, const char* fallback) ;

    // Объявления гибрида: Asp, перестановки блока, номера строк блока.
    void sparse_lu_decl(std::ostream& o, const SparseLu& S, int nsol) ;

    // Решение: правая часть rhs[] по строкам (портится), ответ sol[] по столбцам.
    void sparse_lu_solve(std::ostream& o, const SparseLu& S, const std::string& rhs, const std::string& sol, const char* ind) ;


    // Множитель h^(k-1) переменной i индекса k >= 2 в норме поправки Ньютона ДАУ (как в RADAU5);
    // "" — индекс 0/1 (тогда текст тела прежний до байта).
    std::string dae_norm_w(const System& s, int i) ;


    std::string scheme_implicit_common(const System& s, ImplicitKind kind, int dae_nx = -1,
                                       bool dae_trap = false, const char* dae_hc = nullptr) ;

    // Методы, решающие z из связи (явные со связью, CD), — только индекс 1. Связь, в которой
    // символьно нет ни одной z (dg/dz — нулевая строка), z не определяет: Ньютон дал бы NaN.
    void dae_require_index1(const std::vector<std::string>& J, int N, int nx, const char* what) ;


    // ---------- ДАУ: TR-BDF2 и явные схемы с решением связи на стадиях ----------
    //
    // TR-BDF2 (Bank et al., gamma = 2 - sqrt 2): трапеции до t + gamma*h, затем стадия BDF2
    //   x+ - c1*x_g + c2*x_n - c3*h*f(x+) = 0,  0 = g(x+, z+),
    //   c1 = 1/(g(2-g)), c2 = (1-g)^2/(g(2-g)), c3 = (1-g)/(2-g).
    // L-устойчив, второй порядок. Оба прохода — тот же Ньютон (scheme_implicit_common), каждый в
    // своём блоке; у первого h затенён на gamma*h (время стадии t + gamma*h выходит само),
    // второй решает как неявный Эйлер от комбинации c1*x_g - c2*x_n с hc = c3*h.
    std::string dae_tr_bdf2(const System& s) ;


    // SDIRK2 Александера (gamma = 1 - 1/sqrt 2, L-устойчив, жёстко точен, порядок 2):
    //   Y1 = x_n + gamma h f(Y1),  Y2 = x_n + (1 - gamma) h f(Y1) + gamma h f(Y2),  x_{n+1} = Y2;
    //   связи g(Y_i) = 0 на обеих стадиях. Обе стадии — неявный Эйлер с hc = gamma*h (тот же Ньютон):
    //   первая — от x_n (h затенён на gamma*h, время стадии t + gamma*h), вторая — от
    //   x_n + (1 - gamma)/gamma (Y1 - x_n), т.е. x_n + (1 - gamma) h f(Y1), время t + h.
    std::string dae_sdirk2(const System& s) ;


    // Radau IIA, 3 стадии, порядок 5 (жёстко точен, L-устойчив; индексы 1-3, Hairer-Wanner IV.8, VII.4).
    // Неизвестные W = [Y1; Y2; Y3] (по N), x_{n+1} = Y3. Строки x стадии i:
    //   Y_i - x_n - h sum_j a_ij f(Y_j, t + c_j h) = 0,  строки связей: g(Y_i, t + c_i h) = 0
    // (у ДАУ M y' = F с A невырожденной это то же, что sum_j a_ij g(Y_j) = 0). Полный Ньютон по 3N
    // неизвестным: якобианы f и g в каждой стадии, LU 3N x 3N с выбором главного (цикл). Норма поправки —
    // с множителями h^(k-1) по индексу переменной, как у scheme_implicit_common. Предиктор: x — явный
    // Эйлер до t + c_i h, z — прежние. Упрощённый Ньютон RADAU5 (одна J, разложение A^-1) — потом.
    std::string dae_radau5(const System& s) ;


    // Явный РК (Бутчер A, b, c; S стадий) с решением связи на каждой стадии: x стадии — как
    // у явной схемы, z стадии — Ньютон по z на g(x_s, z, t_s) = 0 (старт — z прошлой стадии),
    // k_s = f(x_s, z_s, t_s). В конце z_{n+1} из g(x_{n+1}, z) = 0. Индекс 1 (dg/dz невырожден).
    // Euler, Midpoint (полуявный РК2 отчёта CD-ДАУ), RK4.
    std::string dae_explicit_rk(const System& s, int S, const double* A, const double* b, const double* c) ;


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
    std::string dae_ros2(const System& s) ;


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
    const RosTab& ros3p_tab() ;

    // Rodas4 (Hairer, Wanner; rodas.f, METH = 1): порядок 4, жёстко точный (последние две стадии
    // в t_n + h), L-устойчив. y_{n+1} = вход 6-й стадии + u_6.
    const RosTab& rodas4_tab() ;

    // emb = true — тело попытки адаптивного шага (ucuda_krs_emb): X константный вход, пишутся
    // Y, E = sum (m - mhat) u, стадии для плотного выхода в W и F1 = f(t + h, Y).
    std::string dae_rosenbrock(const System& s, const RosTab& T, bool emb = false) ;


    // Плотный выход Rosenbrock (rodas.f, CONTRO): D = [y0, y1, d2.u, d3.u].
    std::string ros_dprep_body(const System& s, const RosTab& T) ;

    std::string ros_deval_body(const System& s) ;


    // CD для ДАУ индекса 1 (уровень B, отчёт CD-ДАУ): связь решается всякий раз, когда меняется x.
    //   D  (шаг h1 = a[0]*h, прямой порядок x): x_i += h1 f_i(x, z, t_n), затем z из g(x, z) = 0;
    //   D* (шаг h2 = h - h1, обратный порядок): x_i - x_i^old - h2 f_i(x, z, t_{n+1}) = 0 и
    //       g(x, z, t_{n+1}) = 0 — одним Ньютоном по (x_i, z).
    // Замороженная z дала бы первый порядок (правило отчёта). yoshida — композиция трёх CD-шагов
    // (gamma1, gamma2, gamma1), порядок 4 при a[0] = 0.5; каждый шаг — в своём блоке с затенённым h
    // (и ucuda_t, если система неавтономна).
    std::string dae_cd(const System& s, bool yoshida) ;


    // CD для индекса 2 (отчёт CD-ДАУ, «как SHAKE»): на каждом полушаге z постоянна и подбирается так,
    // чтобы связь выполнялась в КОНЦЕ полушага: g(x(h1; z1), z1) = 0 после D, g(x(h; z2), z2) = 0 после D*.
    //   D  (h1 = a[0] h, прямой порядок): x_i += h1 f_i(x, z1, t_n);
    //   D* (h2 = h - h1, обратный порядок): x_i - x_i^old - h2 f_i(x, z2, t_{n+1}) = 0 скалярным Ньютоном.
    // z полушага — Ньютон с разностным якобианом (nz + 1 проходов на итерацию; «секущие» отчёта). В X
    // пишется (z1 + z2)/2 — z середины шага (сравнивать с точным решением при t - h/2). Работает и на
    // индексе 1 (g с z), но там обычный DAE CD точнее (z после каждого x_i).
    std::string dae_cd2(const System& s) ;


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
    // Предрасчёт ("CD fast"): константа шага — имя и выражение над a[] и h.
    using PrepList = std::vector<std::pair<std::string, std::string>>;

    bool pn_free_of_vars(const PN& n, const std::vector<std::string>& vars) ;


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
                               const std::string& prep_name = std::string()) ;


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
    void fp_collect_terms(const PN& n, bool neg, std::vector<std::pair<bool, PN>>& out) ;

    PN fp_sum_terms(const std::vector<std::pair<bool, PN>>& t) ;

    // f = r + g по слагаемым верхней суммы; r == nullptr, если все слагаемые с v.
    void fp_split_terms(const PN& f, const std::string& v, PN& r, PN& g) ;

    // Множитель после "hs * ": скобки всегда, кроме атомов. CPU-зеркало
    // (SystemEvaluator::solve_diag_fp) считает hs * eval(e) -- та же группировка.
    std::string fp_factor(const PN& e, const NameMap& nm) ;


    // Одно уравнение неявной половины "CD FP" / "CD fast FP". fast -- линейные
    // уравнения решаются как в CD fast (предрасчитанный 1/d или деление),
    // итерации только у нелинейных; иначе итерации у всех, кроме coef == 0
    // (там одна явная формула и есть точное решение).
    void emit_fp_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                             const std::string& v, const std::string& x,
                             const NameMap& nm, const char* hs,
                             const std::string& sfx, bool fast, PrepList* prep,
                             const std::vector<std::string>* state_vars,
                             const std::string& prep_name) ;


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
                                 bool fp = false) ;


    std::string scheme_cd(const System& s)          ;

    std::string scheme_complex_cd(const System& s)  ;

    std::string scheme_complex_cd4(const System& s) ;

    std::string scheme_complex_cd4_s3(const System& s) ;

    std::string scheme_complex_cd4_s4(const System& s) ;

    std::string scheme_complex_cd4_ss01(const System& s) ;

    std::string scheme_complex_cd4_ss10(const System& s) ;


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
        F_SQRT, F_CBRT, F_FABS, F_FLOOR, F_CEIL,
        // F_COPYSIGN is emitted by pn_diff (derivative of fabs); known_funcs() accepts it
        // so that pn_print text (z outputs: dz/dt, sys_augmented) parses back.
        F_POW = 100, F_ATAN2, F_FMOD, F_COPYSIGN, F_FMIN, F_FMAX
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


    double apply_func1(int fid, double x) ;

    double apply_func2(int fid, double a, double b) ;


    // Комплексные версии — те же id, те же формулы, что и в device-коде: обе
    // ветки зовут функции из configCUDA.h, поэтому CPU-траектория Complex CD
    // повторяет GPU-шную операция в операцию (в пределах разной группировки FMA).
    // atan2/fmod комплексного смысла не имеют и до сюда не доходят: на GPU систему
    // с ними отсекает cd_check_complex_safe на кодгене, а CPU-путь кодген не
    // зовёт и спрашивает complex_unsafe_function() перед комплексной схемой.
    // floor/ceil кусочно-постоянны, и их продолжение — константа floor(Re z)
    // (см. configCUDA.h).
    ucmplx apply_func1(int fid, ucmplx x) ;

    // F_COPYSIGN приходит только из якобиана (производная fabs) и всегда с
    // литералом 1 первым аргументом — поэтому a.re. atan2/fmod сюда не доходят:
    // CPU-путь отказывает им заранее (SystemEvaluator::complex_unsafe_function).
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
 std::string scheme_gbs(const System& s, int K);


// Публичная функция
static const char* kDaeNeedsDaeScheme =
    "this is a DAE system: pick a DAE method (or tick 'Solve as ODE' in Library -> DAE)";


std::string dae_scheme_body(const System& s, DaeScheme sch) ;

    // AST -> текст правой части, который снова прочтёт Parser (с явными скобками и
    // операциями; степень — ^{..} в LaTeX-режиме, ^(..) в обычном).
    std::string pn_print(const PN& n, bool latex) ;

    // Копия дерева с заменой символов из m.
    PN pn_subst(const PN& n, const std::map<std::string, PN>& m) ;


    // ---------- Этап 5: массив производных связей (скрытые связи индекса 2-3) ----------
    // Полная производная выражения c вдоль x' = f: sum_i dc/dx_i f_i + dc/dt. Берётся только от
    // выражений без z (z тогда входят лишь через f).
    PN pn_total_dt(const PN& c, const System& s, const std::vector<PN>& f) ;

    // Для связи k: lv[k] = { g_k, d/dt g_k, ... } до первого выражения с z (оно последнее). Длина — индекс
    // связи: 1 — z прямо в g, 2 — скорость (маятник индекса 2), 3 — позиция (маятник в декартовых).
    // Больше трёх производных — отказ (why).
    bool dae_deriv_array(const System& s, std::vector<std::vector<PN>>& lv, std::string* why) ;

    // Плотная система A x = b (n x n, по строкам) Гауссом с выбором главного; false — вырождена.
    bool dense_solve(std::vector<double> A, std::vector<double>& b, int n) ;


// Индекс >= 2 (этап 5): скрытые связи из массива производных. Связи без z (сама g и её производные до
// предпоследней) ограничивают x: x0 проецируется на них Гаусс-Ньютоном с минимальной нормой сдвига
// (x <- x - C^T (C C^T)^-1 c(x), C = dc/dx); последние элементы (с z) решаются Ньютоном по z при этом x.
// Маятник индекса 3: позиция на окружность, скорость по касательной, lambda из второй производной.
bool dae_consistent_hidden(const System& s, const std::vector<std::vector<PN>>& lv, const double* a, int na,
                                  double t0, double* X, std::string* why) ;

    // ОДУ по x: z сняты (их строк в rhs и нет). Уровень B: имена z переезжают в zb_names — f ссылается
    // на них, а build_namemap отдаёт их решению Ньютона.
    System sys_core(const System& s) ;


    // --- Уровень B: z из g(x, z, t) = 0 Ньютоном (System::zb_g) ---
    // Только схемы, у которых каждое вычисление f идёт через rhs_over (там z решается перед f):
    // прочие (CD, SEMP/SIMP/D, неявные, комплексные) печатают f своим путём и увидели бы старую z.
    bool zb_scheme_ok(Scheme sch) ;

    const char* const kZbSchemeMsg =
        "Solve as ODE with a nonlinear constraint (z by Newton): explicit RK / DOPRI / GBS and Extr/Comp over "
        "them only - CD, SEMP/SIMP/D, implicit and complex schemes print f their own way; use a DAE method";

    // Преамбула тела: лямбда ucuda_zb(xs, t) — Ньютон по z (полный, якобиан dg/dz символьный, LU с
    // выбором главного; допуск и предел итераций — Ньютоновские слоты a[]). Тёплый старт — z входа
    // X[nx + k]; повторный вызов с тем же x и t — без пересчёта.
    std::string zb_prelude(const System& c, int nx) ;

    // После шага: z нового состояния st (момент tname) — в st[nx + k].
    std::string zb_tail(const System& c, const char* st, const std::string& tname, int nx) ;

    // Адаптивный шаг: строки z с dz_k/dt = -(sum_i dg_k/dx_i f_i + dg_k/dt) / (dg_k/dz_k) — если каждая
    // связь содержит только свою z (dg/dz диагонален), иначе отказ. Выражения ссылаются на z,
    // rhs_over подставит решение Ньютона в том же x.
    System sys_augmented_b(const System& s) ;

    void zb_adaptive_finish(AdaptiveCode& c, const System& s) ;

    // st[nx + j] = phi_j(st[0..nx-1]) — пересчёт выходов по новому состоянию (время — tname).
    std::string outputs_tail(const System& s, const char* st, const std::string& tname) ;

    // Для адаптивного шага z — строки ОДУ с производной вдоль потока dz/dt = sum dphi/dx_i f_i
    // + dphi/dt: так у них есть F для плотного выхода (Эрмит, стадии схемы). Значение z в узле
    // всё равно перезаписывается phi(Y), а оценка ошибки по z — нулём (adaptive_outputs_tail).
    System sys_augmented(const System& s) ;

    // Хвост emb: z в Y — phi(Y) (точно на связи), ошибка по z — 0 (регулятор ведёт по x).
    void adaptive_outputs_tail(AdaptiveCode& c, const System& s) ;

    const char* const kOutputsNoComplex =
        "z outputs (Solve as ODE): ExtrZ (complex extrapolation core) is not available - pick another scheme";

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
int builtin_scheme_rank(const std::string& name) ;


    std::string wrap_trim(const std::string& s) ;


    // Голова имени — то, что стоит до ПОСЛЕДНЕЙ '|', то есть до коэффициентов.
    // Метка, если она есть, идёт первой: "s17o8|Short CD". Режем по ПЕРВОЙ
    // черте, тогда как база берётся по последней, — старое имя без метки
    // ("Short CD") головы с чертой не имеет и разбирается ровно как раньше.
    bool wrap_split_head(const std::string& head, std::string* label,
                         std::string* base, std::string* err) ;


// --- Экстраполяция Ричардсона ----------------------------------------------

static const char* const kExtrPrefix  = "Extr(";

static const char* const kExtrZPrefix = "ExtrZ(";
   // Re только на выходе

bool starts_with(const std::string& s, const char* pref) ;


// "12" -> 12. Отказывает на пустой строке, нецифрах и выходе за диапазон,
// чтобы кривое имя из старого JSON не превратилось молча в рабочую схему.
bool extr_parse_substeps(const std::string& s, int* out) ;


struct ExRat { long long n, d; };


long long ex_gcd(long long a, long long b) ;


// false — переполнение; вызывающий откатывается на печать литералом.
bool ex_mul(long long a, long long b, long long& r) ;


bool ex_norm(ExRat& x) ;


bool ex_mul_rat(ExRat a, ExRat b, ExRat& out) ;


bool ex_add_rat(ExRat a, ExRat b, ExRat& out) ;


bool ex_pow(long long base, int e, long long& out) ;


// Те же alpha, что в extrapolation_weights, но без единого округления.
// Подстановка u_k = 1/n_k^q в alpha_k ~ n_k^p * prod 1/(u_k - u_m) даёт
// gamma_k = n_k^(p + q(K-2)) / prod_{m!=k} (n_m^q - n_k^q); общий множитель
// сокращается нормировкой суммы в единицу.
bool extrapolation_weights_rational(const std::vector<int>& n, int p, bool symmetric,
                                    std::vector<ExRat>& out) ;


std::string ex_rat_expr(const ExRat& r) ;

std::string scheme_gbs(const System& s, int K) ;


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
bool wrapper_can_inline(const std::string& body) ;


// Тело построчно с дополнительным отступом: панель KRS показывает текст как есть.
void emit_indented(std::ostringstream& o, const std::string& body, const char* indent) ;


// Веса alpha (по всем K стадиям) и разности alpha - beta литералами: alpha — точными
// дробями, когда сходятся (как у постоянного шага), разности — double: это только оценка.
struct ExtrAdWeights {
    std::vector<std::string> alpha, diff;
};


ExtrAdWeights extr_ad_weights(const std::vector<int>& n, int p, bool symmetric) ;


// Общая часть AdaptiveCode экстраполятора: правая часть, плотный выход, порядки.
AdaptiveCode extr_ad_code(const System& s, int p, int q, int emb_rhs) ;


// Хвост emb: Y, E и F1 = f(Y).
void extr_ad_finish(std::ostringstream& o, const System& s, const char* ac, const char* ec) ;


// Неявный полушаг CD (и стадия SIMP, и шаг D): по компоненте — одно вычисление f_i, если она
// линейна по своей переменной (аналитика), иначе CD_ITERS простых итераций; в единицах полной f.
// iters -- число итераций нелинейной компоненты (у "CD FP" -- cd_fp_iters; линейная
// у FP тоже одно вычисление: rem и coef считаются один раз).
double cd_implicit_sweep_rhs(const System& s, int iters = CD_ITERS) ;


// --- Композиция -------------------------------------------------------------

static const char* const kCompPrefix = "Comp(";


    std::string comp_trim(const std::string& s) ;


    // Режем только по запятым ВЕРХНЕГО уровня: коэффициент — выражение, и
    // запятая внутри pow(g,2) не должна заканчивать токен.
    bool comp_split(const std::string& tail, std::vector<std::string>& out) ;


    // Сворачивает коэффициент в число. false на любом символе — тогда значение
    // известно только в рантайме, и звать нас незачем.
    bool comp_fold(const PN& n, double* v) ;


    // Чистый числовой литерал: знак, цифры, точка, экспонента — и ничего
    // больше. Нужен ровно одному месту: решить, был ли первый столбец строки
    // нумерующим. Поэтому и такая строгость — "- 2*g1" числом не считается,
    // и "1 - 2*g1" не теряет свою единицу.
    bool comp_is_plain_number(const std::string& s) ;


    // Срезает нумерующий первый столбец: "3<tab>-0.3825" -> "-0.3825".
    // Разделителем считаются пробелы/табы и ')' ';' ':' — но НЕ точка (иначе
    // "1.35" распалось бы на индекс 1 и коэффициент 35) и НЕ запятая (её
    // разбирает разделение строки на коэффициенты).
    std::string comp_strip_index(const std::string& line) ;


    // comp_fold с окружением: Sym берётся из env, Call считается для функций,
    // которые в коэффициенте шага вообще осмысленны (sqrt(2), cbrt(2) в
    // классических композициях). Неизвестное имя — честный false, а не ноль:
    // подставить ноль значило бы показать пользователю число, которого схема
    // считать не будет.
    bool comp_fold_env(const PN& n, const std::map<std::string, double>& env, double* v) ;


    // Округляет КАЖДЫЙ числовой литерал в строке, не трогая имена и операторы.
    // Работает и на цельном числе, и на выражении: "-1.2599210498948732*g1"
    // укорачивается так же, как "1.3512071919596578".
    std::string comp_round_literals(const std::string& s, int digits) ;


// Ключ сортировки обёртки. Метку не содержит: переименование строку не двигает.
struct WrapperKey {
    int              kind = 3;      // 0 Extr, 1 ExtrZ, 2 Comp, 3 не разобралось
    int              base_rank = 0; // строка паспорта; кастомные — после всех встроенных
    std::string      base;          // для кастомных опорников — алфавит
    int              stages = 0;
    std::vector<int> n;             // подшаги Extr
    std::string      canon;         // имя без метки — последний разделитель
};


WrapperKey wrapper_key(const std::string& name) ;


bool wrapper_key_less(const WrapperKey& x, const WrapperKey& y) ;

}
