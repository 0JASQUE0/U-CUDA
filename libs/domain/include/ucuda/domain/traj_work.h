#pragma once
// Оценка времени счёта ТРАЕКТОРИЙ — без пост-обработки (пики, DBSCAN, Грам-Шмидт, редукции):
//   T = steps * ops * max(w * ns_thr, ns_lat),
// steps — вызовов шага КРС на нить, ops — цена шага по PTX (krs_step_cost), w — варпов на SM
// при раскладке нитей блоками по варпу, ns_* — калибровка карты (gpu_calibrate). Калибровка
// делается один раз на карту и тип numb и хранится в <exe>\gpu_calibration.txt.
// Адаптивный шаг и CPU-ветки не оцениваются: число шагов там заранее неизвестно, а CPU — другая
// машина счёта.
#include "ucuda/domain/analysis_types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Объём работы одного расчёта. Заполняют сессии (traj_work) из того же запроса, что уходит в движок.
struct TrajWork {
    bool        valid = false;
    std::string why;              // почему оценки нет, если valid == false
    std::string krs_body;
    int         amountOfX = 0;
    double      threads = 0.0;    // параллельных нитей (точек сетки; 1 у continuation на GPU)
    double      steps   = 0.0;    // вызовов шага КРС на нить, с клонами LLE/LS
};

// Среднее 1/h по оси свипа h: число шагов у каждой точки своё, ~ T/h.
inline double traj_mean_inv_h(double lo, double hi, int n, bool log_scale) {
    if (n <= 1) return lo > 0.0 ? 1.0 / lo : 0.0;
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f = (double)i / (double)(n - 1);
        const double h = (log_scale && lo > 0.0 && hi > 0.0) ? lo * std::pow(hi / lo, f)
                                                               : lo + (hi - lo) * f;
        if (h <= 0.0) return 0.0;
        s += 1.0 / h;
    }
    return s / (double)n;
}

// Шагов на отрезке t при (среднем) 1/h.
inline double traj_steps(double t, double inv_h) {
    return (t > 0.0 && inv_h > 0.0) ? t * inv_h : 0.0;
}

// Подпись времени: 850 ms, 12.3 s, 4m 05s, 2h 10m.
inline std::string traj_format_seconds(double s) {
    char b[64];
    if (!(s >= 0.0) || !std::isfinite(s)) return "?";
    if (s < 1.0)          std::snprintf(b, sizeof b, "%.0f ms", std::max(1.0, s * 1e3));
    else if (s < 60.0)    std::snprintf(b, sizeof b, "%.1f s", s);
    else if (s < 3600.0)  std::snprintf(b, sizeof b, "%dm %02ds", (int)(s / 60.0), (int)std::fmod(s, 60.0));
    else if (s < 86400.0) std::snprintf(b, sizeof b, "%dh %02dm", (int)(s / 3600.0), (int)std::fmod(s / 60.0, 60.0));
    else                  std::snprintf(b, sizeof b, "%.1f days", s / 86400.0);
    return b;
}
