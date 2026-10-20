#pragma once
// Стадия предрасчёта КРС ("CD fast", пользовательские КРС с блоком предрасчёта).
//
// Тело шага может начинаться блоком между строками-маркерами kKrsPrepBegin /
// kKrsPrepEnd (перед маркером допускаются только пробелы). В блоке — то, что не
// зависит от состояния X: выражения над a[] и h. Вставленное как есть, тело считает
// блок на каждом шаге; так оно и работает везде, где о предрасчёте не знают
// (CPU-DLL, обёртки Extr/Comp, FastSync, устойчивость в Order).
//
// Две формы блока:
//   1) константы — каждая строка "const numb <имя> = <выражение>;" (так пишет codegen
//      для CD fast). Шаг пользуется именами;
//   2) массив — первая строка блока "numb P[<K>];", дальше произвольный C-код, который
//      заполняет P[0..K-1] (циклы, целые, временные — что угодно, но без X[]). Шаг читает
//      P[]. Так пишутся пользовательские КРС (Matreshka 4.0 v3).
// Обе сводятся к одному: код предрасчёта пишет в P[], шаг читает из const numb* P.
//
// Горячие циклы ядер (loopCalculateDiscreteModel*, Metrics, Network, Order ->
// Performance, фазовый портрет Analysis) зовут ucudaKrsPrep(pre, a, h) один раз до
// траектории и calculateDiscreteModelP на каждом шаге. Определения для них движок
// печатает из этого разбора (krs_prep_inline_defs / krs_prep_lib_defs). Без
// предрасчёта (все прочие схемы) cudaLibrary.cu определяет их пустой структурой и
// прямым вызовом calculateDiscreteModel — код тот же, что был до предрасчёта.
// Арифметика в обоих путях одна и та же, результаты побитово равны.
//
// Header-only: его читает и codegen, и движки, а пробники, линкуемые с
// parametric_engine.obj, не должны тянуть за собой codegen.obj.
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

// Маркеры — комментарии, поэтому тело компилируется и без разбора. Без двойных
// фигурных скобок: тело проходит через replace_all плейсхолдеров шаблона.
inline const char* const kKrsPrepBegin = "    // @ucuda-krs-prep-begin";
inline const char* const kKrsPrepEnd   = "    // @ucuda-krs-prep-end";

struct KrsPrepSplit {
    int n = 0;            // число слотов P[]
    std::string prep;     // код предрасчёта: пишет P[0..n-1], видит a и h
    std::string step;     // тело шага: читает P[] (const numb* const P)
};

namespace krs_prep_detail {
inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
// Строки текста без '\r' (тело пользовательской КРС может прийти с CRLF).
inline std::vector<std::string> lines_of(const std::string& body) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : body) {
        if (c == '\r') continue;
        if (c == '\n') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}
}  // namespace krs_prep_detail

// false — в теле нет блока предрасчёта (или он не той формы).
inline bool krs_split_prep(const std::string& body, KrsPrepSplit& out) {
    using namespace krs_prep_detail;
    out = KrsPrepSplit();
    const std::vector<std::string> L = lines_of(body);
    // Блок обязан стоять В НАЧАЛЕ тела: обёртка (Extr, Comp) печатает тело базы
    // внутри своего блока или лямбды, и там оно должно остаться как есть.
    size_t i = 0;
    while (i < L.size() && trim(L[i]).empty()) ++i;
    if (i >= L.size() || trim(L[i]) != trim(kKrsPrepBegin)) return false;
    size_t e = i + 1;
    while (e < L.size() && trim(L[e]) != trim(kKrsPrepEnd)) ++e;
    if (e >= L.size() || e == i + 1) return false;

    std::ostringstream prep, step;
    const std::string first = trim(L[i + 1]);
    if (first.compare(0, 7, "numb P[") == 0) {
        // Форма 2: "numb P[K];" и произвольный код.
        const size_t rb = first.find(']');
        if (rb == std::string::npos || first.substr(rb) != "];") return false;
        const std::string k = first.substr(7, rb - 7);
        if (k.empty() || k.size() > 4) return false;
        for (char c : k) if (!std::isdigit((unsigned char)c)) return false;
        out.n = std::stoi(k);
        if (out.n <= 0) return false;
        for (size_t j = i + 2; j < e; ++j) prep << L[j] << "\n";
    } else {
        // Форма 1: константы, по одной на строку; каждая сразу в свой слот, шаг
        // получает их обратно под теми же именами.
        const std::string decl = "const numb ";
        for (size_t j = i + 1; j < e; ++j) {
            const std::string t = trim(L[j]);
            if (t.compare(0, decl.size(), decl) != 0 || t.back() != ';') return false;
            const size_t eq = t.find(" = ", decl.size());
            if (eq == std::string::npos) return false;
            const std::string name = t.substr(decl.size(), eq - decl.size());
            const std::string expr = t.substr(eq + 3, t.size() - eq - 4);
            prep << "    const numb " << name << " = " << expr << ";\n"
                 << "    P[" << out.n << "] = " << name << ";\n";
            step << "    const numb " << name << " = P[" << out.n << "];\n";
            ++out.n;
        }
    }
    for (size_t j = e + 1; j < L.size(); ++j) step << L[j] << "\n";
    out.prep = prep.str();
    out.step = step.str();
    return true;
}

// Единая единица трансляции: #define UCUDA_KRS_HAS_PREP, struct UcudaKrsPre,
// ucudaKrsPrep и calculateDiscreteModelP. Нужен numb (configCUDA.h) до них.
// device_only — для исходника без __host__ (фазовый портрет Analysis).
inline std::string krs_prep_inline_defs(const KrsPrepSplit& sp, bool device_only = false) {
    const char* q = device_only ? "__device__ __forceinline__\n"
                                : "__device__ __host__ __forceinline__\n";
    std::ostringstream o;
    o << "#define UCUDA_KRS_HAS_PREP 1\n"
      << "struct UcudaKrsPre { numb c[" << sp.n << "]; };\n"
      << q << "void ucudaKrsPrep(UcudaKrsPre& ucuda_pre, const numb* a, const numb h) {\n"
      << "    numb* const P = ucuda_pre.c;\n"
      << sp.prep
      << "}\n"
      << q << "void calculateDiscreteModelP(numb* X, const numb* a, const numb h, const UcudaKrsPre& ucuda_pre) {\n"
      << "    const numb* const P = ucuda_pre.c;\n"
      << sp.step
      << "}\n";
    return o.str();
}

// Раздельная сборка (rdc): тела внешних функций модуля КРС
//   krs_prep(numb* P, const numb* a, const numb h)
//   krs_step_p(numb* X, const numb* a, const numb h, const numb* P)
// и обёртки библиотечной половины над ними (с extern-объявлениями).
inline const char* krs_prep_fn_params()   { return "numb* P, const numb* a, const numb h"; }
inline const char* krs_step_p_fn_params() { return "numb* X, const numb* a, const numb h, const numb* P"; }
inline std::string krs_prep_fn_body(const KrsPrepSplit& sp)   { return sp.prep; }
inline std::string krs_step_p_fn_body(const KrsPrepSplit& sp) { return sp.step; }

inline std::string krs_prep_lib_defs(const KrsPrepSplit& sp) {
    std::ostringstream o;
    o << "\nextern \"C\" __device__ void krs_prep(" << krs_prep_fn_params() << ");"
      << "\nextern \"C\" __device__ void krs_step_p(" << krs_step_p_fn_params() << ");\n"
      << "#define UCUDA_KRS_HAS_PREP 1\n"
      << "struct UcudaKrsPre { numb c[" << sp.n << "]; };\n"
      << "__device__ __host__ __forceinline__\n"
      << "void ucudaKrsPrep(UcudaKrsPre& P, const numb* a, const numb h) { krs_prep(P.c, a, h); }\n"
      << "__device__ __host__ __forceinline__\n"
      << "void calculateDiscreteModelP(numb* X, const numb* a, const numb h, const UcudaKrsPre& P) {\n"
      << "    krs_step_p(X, a, h, P.c);\n"
      << "}\n";
    return o.str();
}

// Вставить text в исходник src сразу после строки, начинающейся с line
// (первое вхождение в начале строки — комментарии с тем же текстом не в счёт).
// false — такой строки нет.
inline bool krs_insert_after_line(std::string& src, const std::string& line, const std::string& text) {
    size_t p = 0;
    for (;;) {
        p = src.find(line, p);
        if (p == std::string::npos) return false;
        if (p == 0 || src[p - 1] == '\n') break;
        p += line.size();
    }
    const size_t eol = src.find('\n', p);
    const size_t at = (eol == std::string::npos) ? src.size() : eol + 1;
    src.insert(at, (eol == std::string::npos ? "\n" : "") + text);
    return true;
}
// То же, но перед строкой.
inline bool krs_insert_before_line(std::string& src, const std::string& line, const std::string& text) {
    size_t p = 0;
    for (;;) {
        p = src.find(line, p);
        if (p == std::string::npos) return false;
        if (p == 0 || src[p - 1] == '\n') break;
        p += line.size();
    }
    src.insert(p, text);
    return true;
}
