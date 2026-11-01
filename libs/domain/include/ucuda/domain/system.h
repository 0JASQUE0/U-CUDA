#pragma once
#include <string>
#include <vector>
// Описание динамической системы для кодгена.
//   vars   — переменные состояния, маппятся на X[0..N-1]
//   params — параметры системы, маппятся на a[1..M] (a[0] зарезервирован под коэф. схемы)
//   rhs    — правые части dX[i]/dt, по одной строке на переменную
//   latex  — если true, rhs трактуются как LaTeX, иначе как обычный синтаксис
struct System {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    std::vector<std::string> rhs;
    bool latex = false;

    // Discrete map: rhs is x_{n+1} directly, not a derivative. It lives here for
    // the same reason as the Newton knobs below - build_system() is the single
    // System factory and its result is copied into every session. The config's
    // scheme is then ignored, so a stale session JSON holding some older scheme
    // cannot substitute a different step for the map.
    bool is_map = false;

    // ДАУ (вкладка DAE): >= 0 — первые dae_nx переменных дифференциальные (x), остальные
    // алгебраические (z), rhs = [f; g] со связями 0 = g. -1 — обычная ОДУ. Схемы ОДУ на
    // такой System не строятся (codegen_scheme откажет), только codegen_dae_scheme.
    int dae_nx = -1;
    // Индекс каждой переменной ДАУ (как nind1/2/3 в RADAU5; пусто или 0/1 — индекс 1). Норма поправки
    // Ньютона берёт переменную индекса k с множителем h^(k-1): без этого на индексе 3 поправка
    // множителя Лагранжа (~1/h^2) решала, когда остановиться, а не позиции.
    std::vector<int> dae_index;

    // Выходы z («Solve as ODE», уровень A): vars = [x; z], rhs = f (только x), out_exprs[j] = phi_j(x)
    // — z после шага пересчитываются по новому x и в f не входят. Так z видят все вкладки
    // (выход БД, метрики, оси) как переменные состояния. codegen_scheme / codegen_adaptive и
    // обёртки поддерживают это сами; точки входа, которые не умеют, отказывают явно.
    std::vector<std::string> out_exprs;
    // Уровень B («Solve as ODE» при связях, не разрешимых явно): vars = [x; z], rhs = f(x, z, t) (только
    // x), zb_g[k] = g_k(x, z, t), z — из g = 0 Ньютоном в каждом вычислении f (лямбда ucuda_zb в
    // начале тела, тёплый старт — z прошлого шага, кэш по x и t), после шага z пишутся в X[nx + k].
    // zb_names — имена z (заполняет внутренний sys_core: у ядра схемы vars только x).
    std::vector<std::string> zb_g;
    std::vector<std::string> zb_names;

    // Ньютон неявных схем здесь не живёт: допуск, предел итераций и режим якобиана —
    // рантайм-слоты a[] каждой вкладки (NewtonSettings ниже).
    // "CD FP" / "CD fast FP": число простых итераций. Сам расчёт берёт его в
    // рантайме из a[] (см. cd_fp_slot ниже); здесь — только оценка цены шага
    // для scheme_rhs_per_step (Order -> Performance кладёт сюда n своего конфига).
    int    cd_fp_iters      = 4;
};

// "CD FP" / "CD fast FP": число простых итераций n — рантайм-значение, как
// коэффициент симметрии s. Раскладка a[] во всех вкладках: [s, p_1..p_M, n],
// то есть n лежит в a[M+1]; тело КРС читает его в cdfp_n. Слот есть у всех
// схем (остальные его не читают), поэтому размер a[] от схемы не зависит.
inline int cd_fp_slot(const System& s) { return (int)s.params.size() + 1; }
// Текст поля "FP iterations" -> n (не меньше 1; мусор -> 4).
inline int cd_fp_iters_from_text(const std::string& t) {
    int v = 4;
    try { v = std::stoi(t); } catch (...) { v = 4; }
    return v < 1 ? 1 : v;
}
// Дописать слот n в конец собранного a[] = [s, p_1..p_M].
template <class Vec>
inline void cd_fp_append_slot(Vec& values, const std::string& n_text) {
    values.push_back((typename Vec::value_type)cd_fp_iters_from_text(n_text));
}

// Ньютон неявных схем (Implicit Euler, Implicit Midpoint, Complex Implicit Euler): допуск,
// предел итераций и режим якобиана — рантайм-значения, как s и n, со своими полями в блоке
// Integration каждой вкладки (System::newton_* — только значения по умолчанию из записи
// системы). Раскладка хвоста a[] после параметров: [n, tol, max_iters, full], то есть
// a[M+1] = n, a[M+2] = tol, a[M+3] = max_iters, a[M+4] = full (0 — модифицированный Ньютон,
// 1 — полный). Слоты есть у всех схем, размер a[] от схемы не зависит. Тело КРС читает их
// из a[], поэтому смена допуска не пересобирает модуль.
struct NewtonSettings {
    bool        full      = false;
    std::string tol       = "1e-10";
    std::string max_iters = "8";
};
constexpr int kSchemeTailSlots = 4;   // [n, tol, max_iters, full] после p_M
inline int newton_slot(const System& s) { return (int)s.params.size() + 2; }   // a[] допуска
inline double newton_tol_from_text(const std::string& t) {
    double v = 1e-10;
    try { v = std::stod(t); } catch (...) { v = 1e-10; }
    return v > 0.0 ? v : 1e-10;
}
inline int newton_iters_from_text(const std::string& t) {
    int v = 8;
    try { v = std::stoi(t); } catch (...) { v = 8; }
    return v < 1 ? 1 : v;
}
// Записать хвост [n, tol, max_iters, full] начиная с tail = &a[M+1].
template <class T>
inline void scheme_fill_tail(T* tail, const std::string& n_text, const NewtonSettings& nw) {
    tail[0] = (T)cd_fp_iters_from_text(n_text);
    tail[1] = (T)newton_tol_from_text(nw.tol);
    tail[2] = (T)newton_iters_from_text(nw.max_iters);
    tail[3] = (T)(nw.full ? 1 : 0);
}
// Дописать хвост в конец собранного a[] = [s, p_1..p_M].
template <class Vec>
inline void scheme_append_tail(Vec& values, const std::string& n_text, const NewtonSettings& nw) {
    const size_t b = values.size();
    values.resize(b + kSchemeTailSlots);
    scheme_fill_tail(values.data() + b, n_text, nw);
}

