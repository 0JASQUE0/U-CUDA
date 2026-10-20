#pragma once
// Стадия предрасчёта КРС ("CD fast").
//
// Тело шага, как его выдаёт codegen_scheme, самодостаточно: в его начале, между
// строками-маркерами kKrsPrepBegin / kKrsPrepEnd, стоят константы вида
//     const numb <имя> = <выражение>;
// — выражения над a[] и h (и над предыдущими константами), без X[]. Вставленное
// как есть, тело считает их на каждом шаге; так оно и работает везде, где о
// предрасчёте не знают (CPU-DLL, обёртки Extr/Comp, FastSync, устойчивость в Order).
//
// Горячие циклы ядер (loopCalculateDiscreteModel*, Metrics, Network, Order ->
// Performance, фазовый портрет Analysis) зовут ucudaKrsPrep(P, a, h) один раз до
// траектории и calculateDiscreteModelP на каждом шаге. Определения для них движок
// печатает из этого разбора (krs_prep_inline_defs / krs_prep_lib_defs). Без
// предрасчёта (все прочие схемы) cudaLibrary.cu определяет их пустой структурой и
// прямым вызовом calculateDiscreteModel — код тот же, что был до предрасчёта.
// Арифметика в обоих путях одна и та же, результаты побитово равны.
//
// Header-only: его читает и codegen, и движки, а пробники, линкуемые с
// parametric_engine.obj, не должны тянуть за собой codegen.obj.
#include <sstream>
#include <string>
#include <vector>

// Маркеры — комментарии, поэтому тело компилируется и без разбора. Без двойных
// фигурных скобок: тело проходит через replace_all плейсхолдеров шаблона.
inline const char* const kKrsPrepBegin = "    // @ucuda-krs-prep-begin";
inline const char* const kKrsPrepEnd   = "    // @ucuda-krs-prep-end";

struct KrsPrepSplit {
    std::vector<std::string> names, exprs;   // константы в порядке объявления
    std::string step;                        // тело шага без блока констант
};

// false — в теле нет блока предрасчёта (или он пуст / не той формы).
inline bool krs_split_prep(const std::string& body, KrsPrepSplit& out) {
    out = KrsPrepSplit();
    // Блок обязан стоять В НАЧАЛЕ тела: обёртка (Extr, Comp) печатает тело
    // базы с отступом внутри своего блока, и там оно должно остаться как есть.
    const std::string beg = std::string(kKrsPrepBegin) + "\n";
    const std::string end = std::string(kKrsPrepEnd) + "\n";
    if (body.compare(0, beg.size(), beg) != 0) return false;
    const size_t e = body.find(end, beg.size());
    if (e == std::string::npos) return false;
    const std::string decl = "    const numb ";
    size_t p = beg.size();
    while (p < e) {
        const size_t nl = body.find('\n', p);
        if (nl == std::string::npos || nl > e) return false;
        const std::string line = body.substr(p, nl - p);
        p = nl + 1;
        if (line.size() < decl.size() + 5 || line.compare(0, decl.size(), decl) != 0
            || line.back() != ';') return false;
        const size_t eq = line.find(" = ", decl.size());
        if (eq == std::string::npos) return false;
        out.names.push_back(line.substr(decl.size(), eq - decl.size()));
        out.exprs.push_back(line.substr(eq + 3, line.size() - eq - 4));
    }
    if (out.names.empty()) return false;
    out.step = body.substr(e + end.size());
    return true;
}

// "const numb <имя> = <arr>[k];" для каждой константы — шапка тела шага.
inline std::string krs_prep_loads(const KrsPrepSplit& sp, const char* arr) {
    std::ostringstream o;
    for (size_t k = 0; k < sp.names.size(); ++k)
        o << "    const numb " << sp.names[k] << " = " << arr << "[" << k << "];\n";
    return o.str();
}
// Тело предрасчёта: константы по порядку (следующая может читать предыдущую),
// каждая сразу в свой слот.
inline std::string krs_prep_stores(const KrsPrepSplit& sp, const char* arr) {
    std::ostringstream o;
    for (size_t k = 0; k < sp.names.size(); ++k)
        o << "    const numb " << sp.names[k] << " = " << sp.exprs[k] << ";\n"
          << "    " << arr << "[" << k << "] = " << sp.names[k] << ";\n";
    return o.str();
}

// Единая единица трансляции: #define UCUDA_KRS_HAS_PREP, struct UcudaKrsPre,
// ucudaKrsPrep и calculateDiscreteModelP. Нужен numb (configCUDA.h) до них.
// device_only — для исходника без __host__ (фазовый портрет Analysis).
inline std::string krs_prep_inline_defs(const KrsPrepSplit& sp, bool device_only = false) {
    const char* q = device_only ? "__device__ __forceinline__\n"
                                : "__device__ __host__ __forceinline__\n";
    std::ostringstream o;
    o << "#define UCUDA_KRS_HAS_PREP 1\n"
      << "struct UcudaKrsPre { numb c[" << sp.names.size() << "]; };\n"
      << q << "void ucudaKrsPrep(UcudaKrsPre& P, const numb* a, const numb h) {\n"
      << krs_prep_stores(sp, "P.c")
      << "}\n"
      << q << "void calculateDiscreteModelP(numb* X, const numb* a, const numb h, const UcudaKrsPre& P) {\n"
      << krs_prep_loads(sp, "P.c") << sp.step
      << "}\n";
    return o.str();
}

// Раздельная сборка (rdc): тела внешних функций модуля КРС
//   krs_prep(numb* P, const numb* a, const numb h)
//   krs_step_p(numb* X, const numb* a, const numb h, const numb* P)
// и обёртки библиотечной половины над ними (с extern-объявлениями).
inline const char* krs_prep_fn_params()   { return "numb* P, const numb* a, const numb h"; }
inline const char* krs_step_p_fn_params() { return "numb* X, const numb* a, const numb h, const numb* P"; }
inline std::string krs_prep_fn_body(const KrsPrepSplit& sp)   { return krs_prep_stores(sp, "P"); }
inline std::string krs_step_p_fn_body(const KrsPrepSplit& sp) { return krs_prep_loads(sp, "P") + sp.step; }

inline std::string krs_prep_lib_defs(const KrsPrepSplit& sp) {
    std::ostringstream o;
    o << "\nextern \"C\" __device__ void krs_prep(" << krs_prep_fn_params() << ");"
      << "\nextern \"C\" __device__ void krs_step_p(" << krs_step_p_fn_params() << ");\n"
      << "#define UCUDA_KRS_HAS_PREP 1\n"
      << "struct UcudaKrsPre { numb c[" << sp.names.size() << "]; };\n"
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
