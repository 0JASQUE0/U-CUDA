#pragma once
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <system_error>

// ============================================================================
// Единственный парсер числового поля во всём проекте.
//
// Все текстовые поля UI (диапазоны свипов, начальные условия, пороги, шаг)
// хранятся строками и разбираются в double в момент использования. Поле
// принимает не просто число, а АРИФМЕТИЧЕСКОЕ ВЫРАЖЕНИЕ:
//
//     8/3        1/3        2*pi       pi/4       0.5*1e-3
//     1e-3/2     100-1      (1+2)*3     -2^ нет, степени нет
//
// Из именованных констант есть pi (регистр не важен: pi, Pi, PI). Умножение
// перед ней обязательно: "2*pi", а не "2pi" — неявное умножение не поддержано
// намеренно, чтобы не гадать за пользователя; "2pi" честно подсветится как
// невалидный ввод, а не посчитается молча.
//
// Поддержаны + - * / , унарный минус/плюс и скобки, с обычным приоритетом
// (* / сильнее + -). Дробь "a/b" — частный случай, ради которого всё и
// начиналось: пользователи пишут "8/3" в диапазонах свипа. Остальное добавлено
// затем, что серии экспериментов удобно задавать относительно уже введённого
// числа ("тот же диапазон, но вдвое шире" — это *2, а не пересчёт в уме).
//
// Числа читаются через strtod, поэтому научная запись остаётся целой: "1e-14"
// это одно число, а не "1e" минус "14".
//
// Пустая строка и мусор дают `def`. Отличие от прежней версии: раньше parse_num
// стоял на atof и на "5asdf" молча возвращал 5, хотя поле рядом писало
// "invalid number, using default" — сообщение врало. Теперь хвост, который не
// разобрался, делает ВСЮ строку невалидной, и поведение сходится с подписью.
// Невалидны также деление на ноль и не-finite результат ("1e400").
//
// Зачем один на всех: реализаций было пять, и три из них (на std::stod) дробей
// НЕ понимали. Из-за расхождения "8/3" в поле границы свипа Custom давал 2.667
// в ядре и 8 в интерфейсе — диапазон ползунка, кламп fix-значения и привязка
// крестика к сетке уезжали за пределы реального свипа. Один и тот же класс
// бага чинили дважды в разных местах, поэтому теперь точка одна.
//
// Для вопроса «это вообще число?» есть parse_num_checked ниже — та же
// грамматика, что и у parse_num, поэтому подсветка невалидного ввода и
// фактический разбор не могут разойтись (а раньше расходились: is_numeric_string
// в gui.cpp несла собственную копию грамматики).
// ============================================================================
namespace num_parse_detail {

// ЗНАЧЕНИЕ ОБЯЗАНО СОВПАДАТЬ с constexpr numb pi из configCUDA.h — иначе поле
// ввода и ядро считали бы разное pi. Здесь литерал, а не #include configCUDA.h:
// тот тянет <math_constants.h> и, главное, содержит `#ifndef AMOUNTOFX` —
// включённый рано из этого лёгкого заголовка, он зафиксировал бы размерность
// по умолчанию в TU, который собирался её переопределить.
// Расхождение ловится static_assert'ом в gui.cpp, где видны оба определения.
inline constexpr double kPi = 3.1415926535897932384626433832795;

inline void skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t') ++p;
}

inline bool parse_expr(const char*& p, double& out);   // взаимная рекурсия со скобками

inline bool parse_primary(const char*& p, double& out) {
    skip_ws(p);
    if (*p == '(') {
        ++p;
        if (!parse_expr(p, out)) return false;
        skip_ws(p);
        if (*p != ')') return false;
        ++p;
        return true;
    }
    // Именованные константы разбираются ДО strtod. Это не только про pi: strtod
    // распознаёт слова "inf" и "nan", и без этой ветки они молча просачивались
    // бы в поле как значения. Любое неизвестное имя делает ввод невалидным.
    if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
        const char* const start = p;
        while ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) ++p;
        const size_t len = (size_t)(p - start);
        const bool is_pi = (len == 2)
                        && (start[0] == 'p' || start[0] == 'P')
                        && (start[1] == 'i' || start[1] == 'I');
        if (!is_pi) return false;
        out = kPi;
        return true;
    }

    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) return false;        // ни одного символа числа не съедено
    p = end;
    out = v;
    return true;
}

inline bool parse_factor(const char*& p, double& out) {
    skip_ws(p);
    if (*p == '+' || *p == '-') {
        const bool negate = (*p == '-');
        ++p;
        double v = 0.0;
        if (!parse_factor(p, v)) return false;   // допускает "--1" и "-(2+3)"
        out = negate ? -v : v;
        return true;
    }
    return parse_primary(p, out);
}

inline bool parse_term(const char*& p, double& out) {
    if (!parse_factor(p, out)) return false;
    for (;;) {
        skip_ws(p);
        const char op = *p;
        if (op != '*' && op != '/') return true;
        ++p;
        double rhs = 0.0;
        if (!parse_factor(p, rhs)) return false;
        if (op == '/') {
            if (rhs == 0.0) return false;   // "8/0" — невалидно, как и раньше
            out /= rhs;
        } else {
            out *= rhs;
        }
    }
}

inline bool parse_expr(const char*& p, double& out) {
    if (!parse_term(p, out)) return false;
    for (;;) {
        skip_ws(p);
        const char op = *p;
        if (op != '+' && op != '-') return true;
        ++p;
        double rhs = 0.0;
        if (!parse_term(p, rhs)) return false;
        out = (op == '+') ? (out + rhs) : (out - rhs);
    }
}

} // namespace num_parse_detail

// Разбор с ответом «получилось или нет». Пустая строка — НЕ валидна здесь
// (вызывающий сам решает, что значит пустое поле); см. parse_num ниже.
inline bool parse_num_checked(const std::string& s, double& out) {
    const char* p = s.c_str();
    double v = 0.0;
    if (!num_parse_detail::parse_expr(p, v)) return false;
    num_parse_detail::skip_ws(p);
    if (*p != '\0') return false;                 // неразобранный хвост
    if (!std::isfinite(v)) return false;          // overflow / nan / inf
    out = v;
    return true;
}

inline double parse_num(const std::string& s, double def) {
    if (s.empty()) return def;
    double v = 0.0;
    return parse_num_checked(s, v) ? v : def;
}

// ============================================================================
// Целочисленные поля: Resolution, decimator, iter of synchr, число бинов, N
// сетки. Тот же разбор, что и у вещественных, плюс округление и кламп в
// диапазон int.
//
// Зачем отдельная функция, а не (int)parse_num на месте: копий было две, и
// расходились они молча. Поле запроса читал std::stoi, а он не кидает на
// "512/2" — парсит ведущее "512" и тихо выбрасывает остальное. То есть
// Resolution "512/2" давал сетку 512x512 вместо 256x256, "100*2" — 100 вместо
// 200, а "1e3" — 1 вместо 1000, и всё это без единого предупреждения: подсветка
// поля идёт через parse_num_checked и такой ввод считает валидным. Дроби в
// целочисленных полях по той же причине не работали никогда.
//
// Округление, а не усечение: "100/3" — это 33, а не 33.333 и не 33 по обрезке
// вниз от 33.999 при "101/3".
// ============================================================================
inline int parse_num_int(const std::string& s, int def) {
    if (s.empty()) return def;
    double v = 0.0;
    if (!parse_num_checked(s, v)) return def;
    const double lo = -2147483648.0;
    const double hi =  2147483647.0;
    if (v < lo) return (int)lo;
    if (v > hi) return (int)hi;
    return (int)(v < 0.0 ? -std::floor(-v + 0.5) : std::floor(v + 0.5));
}

// ============================================================================
// Обратная к parse_num: КРАТЧАЙШИЙ текст, который парсится обратно в РОВНО ТО
// ЖЕ double (shortest round-trip). Нужен там, где в текстовое поле уходит
// значение, посчитанное движком — узел параметрической сетки под пикселем
// карты (см. ucuda_node_value в configCUDA.h).
//
// Зачем не %.6g, которым это делалось раньше: шесть значащих цифр отрезают
// ~10 знаков, и фазовый портрет считался НЕ в том параметре, в котором
// посчитана ячейка (для узла 0.301751884422111 в поле уходило 0.301752).
// В периодическом окне это незаметно, в хаотической полосе сдвиг 4e-7 даёт
// другую выборку аттрактора — то есть пиксель и портрет расходились по режиму.
//
// Короткие значения короткими и остаются: 0.352 -> "0.352", а не "0.35200000000000004"
// — to_chars печатает ровно столько цифр, сколько нужно для round-trip'а, и не
// больше. atof в parse_num корректно округляет, поэтому обратный переход точен.
// ============================================================================
inline std::string fmt_num_shortest(double v) {
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    if (r.ec != std::errc{}) {          // не влезло — падаем на %.17g (тоже round-trip)
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    }
    return std::string(buf, r.ptr);
}
