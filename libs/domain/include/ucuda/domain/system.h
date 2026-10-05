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

    // Newton knobs for the implicit schemes. They live here (and not in every
    // per-analysis config) because AppModel::build_system() is the single System
    // factory and its result is copied into every session, so codegen sees them
    // without touching analysis_session.h / session_io.cpp.
    // newton_full: false = modified Newton (Jacobian and LU reused across
    // iterations, refreshed when the correction stops shrinking), true = full
    // Newton (rebuilt every iteration).
    bool   newton_full      = false;
    double newton_tol       = 1e-10;
    int    newton_max_iters = 8;
};

