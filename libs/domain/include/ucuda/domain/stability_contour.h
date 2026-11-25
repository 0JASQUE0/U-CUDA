#pragma once
// Границы области устойчивости и области предпочтительности как ломаные —
// изолинии по сетке StabilityResult (marching squares с линейной интерполяцией
// вдоль рёбер ячеек). Нужны окну «Stability bounds» вкладки Order, где границы
// нескольких схем лежат на одной диаграмме, и экспорту этих границ.
//
// Поле, чей нуль — граница («внутри» — значения <= 0):
//   устойчивость:        f = rho - 1;
//   предпочтительность:  f = max(rho - 1, log10(err / tol)) — пересечение двух
//                        областей; ошибка в логарифме, потому что идёт на
//                        порядки, и линейная интерполяция по ней самой клала бы
//                        границу почти всегда в узел.
// Непосчитанная ячейка (status != OK, NaN) рвёт линию: ячейки сетки, задевшие
// её углом, не дают отрезков. Граница, уходящая за край сетки, обрывается на
// краю, а не замыкается по нему.
#include "ucuda/domain/analysis_types.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr int kStabBoundStability  = 0;
constexpr int kStabBoundPreference = 1;

// Ломаные подряд, точки (sigma, omega), между ломаными — пара NaN (так их
// принимает Plot2DView). n_lines — число ломаных.
struct StabilityContour {
    std::vector<double> xy;
    int n_lines = 0;
};

inline double stability_bound_field(const StabilityResult& r, size_t i, int which, double tol) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (i >= r.rho.size()) return nan;
    if (i < r.status.size() && r.status[i] != STAB_ST_OK) return nan;
    const double f_rho = r.rho[i] - 1.0;
    if (std::isnan(f_rho)) return nan;
    if (which != kStabBoundPreference) return f_rho;
    if (i >= r.err.size() || std::isnan(r.err[i]) || !(tol > 0.0)) return nan;
    const double e = r.err[i];
    // 0 -> -inf и +inf (экспонента выродилась) — законные значения: max и
    // клэмп ниже делают из них «глубоко внутри» и «глубоко снаружи».
    const double f_err = (e <= 0.0) ? -1e300 : std::log10(e / tol);
    return std::max(f_rho, f_err);
}

inline StabilityContour stability_contour(const StabilityResult& r, int which, double tol) {
    StabilityContour out;
    const int nx = r.n_pts_x, ny = r.n_pts_y;
    if (nx < 2 || ny < 2 || r.rho.size() < (size_t)nx * (size_t)ny
        || r.sigma_vals.size() < (size_t)nx || r.omega_vals.size() < (size_t)ny)
        return out;

    // Бесконечности клэмпятся: интерполяция с ними даёт точку ровно в узле
    // конечной стороны, а NaN — пропуск ячейки.
    std::vector<double> f((size_t)nx * (size_t)ny);
    for (size_t i = 0; i < f.size(); ++i) {
        const double v = stability_bound_field(r, i, which, tol);
        f[i] = std::isnan(v) ? v : std::clamp(v, -1e6, 1e6);
    }
    auto at = [&](int ix, int iy) { return f[(size_t)iy * (size_t)nx + (size_t)ix]; };

    // Рёбра сетки: горизонтальное (ix,iy)-(ix+1,iy) — 2k, вертикальное
    // (ix,iy)-(ix,iy+1) — 2k+1, k = iy*nx + ix. Соседние ячейки делят ребро, и по
    // его номеру отрезки сшиваются в ломаные.
    auto eid_h = [nx](int ix, int iy) { return 2LL * ((long long)iy * nx + ix); };
    auto eid_v = [nx](int ix, int iy) { return 2LL * ((long long)iy * nx + ix) + 1; };
    std::unordered_map<long long, std::pair<double, double>> pt;
    auto cross = [](double va, double vb) {
        const double t = va / (va - vb);
        return std::isfinite(t) ? std::clamp(t, 0.0, 1.0) : 0.5;
    };
    auto put_h = [&](int ix, int iy) {
        const long long id = eid_h(ix, iy);
        if (!pt.count(id)) {
            const double t = cross(at(ix, iy), at(ix + 1, iy));
            const double x0 = r.sigma_vals[(size_t)ix], x1 = r.sigma_vals[(size_t)ix + 1];
            pt[id] = { x0 + t * (x1 - x0), r.omega_vals[(size_t)iy] };
        }
        return id;
    };
    auto put_v = [&](int ix, int iy) {
        const long long id = eid_v(ix, iy);
        if (!pt.count(id)) {
            const double t = cross(at(ix, iy), at(ix, iy + 1));
            const double y0 = r.omega_vals[(size_t)iy], y1 = r.omega_vals[(size_t)iy + 1];
            pt[id] = { r.sigma_vals[(size_t)ix], y0 + t * (y1 - y0) };
        }
        return id;
    };

    std::vector<std::pair<long long, long long>> seg;
    for (int iy = 0; iy + 1 < ny; ++iy) {
        for (int ix = 0; ix + 1 < nx; ++ix) {
            const double v0 = at(ix, iy), v1 = at(ix + 1, iy),
                         v2 = at(ix + 1, iy + 1), v3 = at(ix, iy + 1);
            if (std::isnan(v0) || std::isnan(v1) || std::isnan(v2) || std::isnan(v3)) continue;
            const int c = (v0 <= 0.0 ? 1 : 0) | (v1 <= 0.0 ? 2 : 0)
                        | (v2 <= 0.0 ? 4 : 0) | (v3 <= 0.0 ? 8 : 0);
            if (c == 0 || c == 15) continue;
            // Рёбра ячейки: низ, право, верх, лево (углы 0-1, 1-2, 3-2, 0-3).
            auto e = [&](int k) {
                switch (k) {
                    case 0:  return put_h(ix, iy);
                    case 1:  return put_v(ix + 1, iy);
                    case 2:  return put_h(ix, iy + 1);
                    default: return put_v(ix, iy);
                }
            };
            auto add = [&](int a, int b) { seg.emplace_back(e(a), e(b)); };
            const bool center_in = 0.25 * (v0 + v1 + v2 + v3) <= 0.0;
            switch (c) {
                case 1:  case 14: add(3, 0); break;
                case 2:  case 13: add(0, 1); break;
                case 3:  case 12: add(3, 1); break;
                case 4:  case 11: add(1, 2); break;
                case 6:  case 9:  add(0, 2); break;
                case 7:  case 8:  add(3, 2); break;
                case 5:   // углы 0 и 2 внутри: седло решает центр ячейки
                    if (center_in) { add(0, 1); add(2, 3); } else { add(3, 0); add(1, 2); }
                    break;
                case 10:  // углы 1 и 3 внутри
                    if (center_in) { add(3, 0); add(1, 2); } else { add(0, 1); add(2, 3); }
                    break;
                default: break;
            }
        }
    }
    if (seg.empty()) return out;

    // Сшивка: у каждого ребра не больше двух отрезков (по одному с каждой ячейки).
    std::unordered_map<long long, std::vector<int>> by_edge;
    for (int s = 0; s < (int)seg.size(); ++s) {
        by_edge[seg[(size_t)s].first].push_back(s);
        by_edge[seg[(size_t)s].second].push_back(s);
    }
    std::vector<char> used(seg.size(), 0);
    auto next_from = [&](long long edge, long long& other) {
        for (int s : by_edge[edge]) {
            if (used[(size_t)s]) continue;
            used[(size_t)s] = 1;
            other = (seg[(size_t)s].first == edge) ? seg[(size_t)s].second : seg[(size_t)s].first;
            return true;
        }
        return false;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<long long> fwd, back;
    for (int s0 = 0; s0 < (int)seg.size(); ++s0) {
        if (used[(size_t)s0]) continue;
        used[(size_t)s0] = 1;
        fwd.assign({ seg[(size_t)s0].first, seg[(size_t)s0].second });
        long long o = 0;
        while (next_from(fwd.back(), o)) fwd.push_back(o);
        back.clear();
        while (next_from(back.empty() ? fwd.front() : back.back(), o)) back.push_back(o);
        if (out.n_lines > 0) { out.xy.push_back(nan); out.xy.push_back(nan); }
        for (auto it = back.rbegin(); it != back.rend(); ++it) {
            const auto& p = pt[*it];
            out.xy.push_back(p.first); out.xy.push_back(p.second);
        }
        for (long long id : fwd) {
            const auto& p = pt[id];
            out.xy.push_back(p.first); out.xy.push_back(p.second);
        }
        ++out.n_lines;
    }
    return out;
}
