#include "expression_internal.h"
using namespace ucuda::equations_detail;
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
    return Scheme::Euler;
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

