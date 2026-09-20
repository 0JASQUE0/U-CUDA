#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Генератор эталонных данных для tests/fixtures/lorenz_reference.cir.

Оракул фазы 0: схема на ИДЕАЛЬНЫХ компонентах тождественно реализует
масштабированную систему Лоренца (вывод — в README.md рядом). Поэтому эталон
считается прямым интегрированием этой системы независимой реализацией (RK4
здесь, в Python), а не снимается со схемы. Неидеальности ОУ и AD633 этим НЕ
проверяются — это задача фазы 2.5 и внешнего симулятора.

Запуск:  python gen_lorenz_reference.py
Пишет рядом: lorenz_reference_trajectory.csv, lorenz_reference_invariants.json
"""

import json
import math
import os

# --- система (Cuomo & Oppenheim 1993, масштаб u=x/10, v=y/10, w=z/20) --------
SIGMA, RHO, BETA = 16.0, 45.6, 4.0

# Временной масштаб схемы: 1/(R_unit*C) = 1/(100k * 10n) = 1000 1/с.
# Одна единица времени ОДУ = 1 мс схемного времени.
RATE = 1000.0

IC = (0.1, 0.1, 0.1)          # (u,v,w) в вольтах; = (x,y,z) = (1,1,2)
H = 1.0e-5                    # шаг RK4 в единицах времени ОДУ
T_END = 50.0                  # длина эталонной записи
OUT_EVERY = 0.01              # шаг выдачи в CSV


def rhs(s):
    u, v, w = s
    return (SIGMA * (v - u),
            RHO * u - v - 20.0 * u * w,
            5.0 * u * v - BETA * w)


def rk4(s, h):
    k1 = rhs(s)
    s1 = (s[0] + h / 2 * k1[0], s[1] + h / 2 * k1[1], s[2] + h / 2 * k1[2])
    k2 = rhs(s1)
    s2 = (s[0] + h / 2 * k2[0], s[1] + h / 2 * k2[1], s[2] + h / 2 * k2[2])
    k3 = rhs(s2)
    s3 = (s[0] + h * k3[0], s[1] + h * k3[1], s[2] + h * k3[2])
    k4 = rhs(s3)
    return (s[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            s[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            s[2] + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))


def integrate(s, h, n):
    for _ in range(n):
        s = rk4(s, h)
    return s


def trajectory():
    """Пишет CSV и попутно собирает статистику по аттрактору."""
    steps = int(round(OUT_EVERY / H))
    npts = int(round(T_END / OUT_EVERY))
    s = IC
    rows = [(0.0, 0.0) + s]
    for i in range(1, npts + 1):
        s = integrate(s, H, steps)
        rows.append((i * OUT_EVERY, i * OUT_EVERY / RATE) + s)
    return rows


def convergence_check():
    """Самооценка точности: та же траектория шагом h и h/2 на коротком горизонте.

    Лоренц хаотичен, поэтому число растёт с горизонтом — оно характеризует не
    'ошибку эталона вообще', а точность на участке, где поточечное сравнение
    ещё осмысленно.
    """
    out = {}
    for t in (1.0, 5.0, 10.0):
        a = integrate(IC, H, int(round(t / H)))
        b = integrate(IC, H / 2, int(round(t / (H / 2))))
        out["t=%g" % t] = max(abs(a[i] - b[i]) for i in range(3))
    return out


def lle(t_transient=20.0, t_run=200.0, renorm_every=0.5, d0=1e-8):
    """Старший ляпуновский показатель по Беннеттину, в единицах времени ОДУ."""
    s = integrate(IC, H, int(round(t_transient / H)))
    p = (s[0] + d0, s[1], s[2])
    nsteps = int(round(renorm_every / H))
    nren = int(round(t_run / renorm_every))
    acc = 0.0
    for _ in range(nren):
        s = integrate(s, H, nsteps)
        p = integrate(p, H, nsteps)
        d = math.sqrt(sum((p[i] - s[i]) ** 2 for i in range(3)))
        acc += math.log(d / d0)
        k = d0 / d
        p = tuple(s[i] + (p[i] - s[i]) * k for i in range(3))
    return acc / (nren * renorm_every)


def ranges(rows, skip_frac=0.2):
    """Статистика установившегося режима.

    min/max — вход для масштабирования фазы 5, но СРАВНИВАТЬ по ним нельзя:
    на хаотическом аттракторе это выборочная величина, а не инвариант. Замер:
    возмущение НУ на 1e-9 сдвигает min/max на 0.19 В, mean на 0.055 В, а std —
    всего на 0.004 В. Для сверки решателей годится только std.
    """
    tail = rows[int(len(rows) * skip_frac):]
    out = {}
    for j, name in enumerate(("u", "v", "w")):
        col = [r[2 + j] for r in tail]
        mean = sum(col) / len(col)
        var = sum((c - mean) ** 2 for c in col) / len(col)
        out[name] = {"min": min(col), "max": max(col),
                     "mean": mean, "std": math.sqrt(var)}
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    print("integrating reference trajectory ...")
    rows = trajectory()
    csv_path = os.path.join(here, "lorenz_reference_trajectory.csv")
    with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("t_ode,t_circuit_s,u_V,v_V,w_V\n")
        for r in rows:
            f.write("%.6f,%.9e,%.12e,%.12e,%.12e\n" % r)
    print("  %s (%d rows)" % (csv_path, len(rows)))

    print("convergence check ...")
    conv = convergence_check()
    print("largest Lyapunov exponent ...")
    l1 = lle()

    inv = {
        "system": "Lorenz, Cuomo-Oppenheim scaling u=x/10, v=y/10, w=z/20",
        "parameters": {"sigma": SIGMA, "r": RHO, "b": BETA},
        "initial_conditions_uvw": list(IC),
        "integrator": {"method": "RK4", "h_ode": H, "t_end_ode": T_END},
        "circuit_time_scale_hz": RATE,
        "lle_per_ode_time": l1,
        "lle_per_second_circuit": l1 * RATE,
        "self_convergence_max_abs_diff_h_vs_h2": conv,
        "steady_state_ranges_V": ranges(rows),
        "comparison_note": ("compare solvers by std only: a 1e-9 change of the initial "
                            "condition moves min/max by 0.19 V and mean by 0.055 V, "
                            "while std moves by 0.004 V"),
    }
    json_path = os.path.join(here, "lorenz_reference_invariants.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("  %s" % json_path)
    print("  LLE = %.6f per ODE time unit (%.1f per second of circuit time)"
          % (l1, l1 * RATE))


if __name__ == "__main__":
    main()
