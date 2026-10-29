#pragma once
// Модели полупроводниковых приборов для импорта схем и рабочая точка ДАУ (этап 9, часть a).
//
// Приборы, как и всё в импорте схем, — ВЫРАЖЕНИЯ в синтаксисе U-CUDA: ток прибора по
// напряжениям на нём. Параметры моделей (.model) — параметры a[] (свип без перекомпиляции).
//   Диод (SPICE level 1, только постоянный ток):  I = IS (e^(Vd/(N Vt)) - 1) + GMIN Vd.
//   BJT — Эберс–Молл в транспортной форме с эффектом Эрли (Гуммель–Пун при IKF = IKR = inf,
//   ISE = ISC = 0):  Ict = IS (e^(Vbe/(NF Vt)) - e^(Vbc/(NR Vt))) (1 - Vbc/VAF - Vbe/VAR),
//   Ic = Ict - IS/BR (e^(Vbc/(NR Vt)) - 1),  Ib = IS/BF (e^(Vbe/(NF Vt)) - 1) + IS/BR (e^(Vbc/(NR Vt)) - 1).
// Экспонента — с линейным продолжением выше 40 (dev_explim): Ньютон с далёкого старта иначе
// уходит в inf (у SPICE ту же роль играет pnjlim). Ёмкости переходов CJO / CJE / CJC — постоянные
// (у SPICE они зависят от напряжения); TT, BV, IKF, ISE и прочее не моделируются — импорт
// перечисляет их в отчёте.

#include <string>
#include "codegen.hpp"

// kT/q при 27 °C (TNOM SPICE), В.
constexpr double kDeviceVt = 0.025864186384551;
constexpr double kDeviceGmin = 1e-12;

// exp(u) при u <= 40, выше — касательная: exp(40) (1 + u - 40). Непрерывна вместе с производной.
std::string dev_explim(const std::string& u);

// Ток диода от анода к катоду. vd, is, n — выражения (имена параметров a[] или числа).
std::string diode_current(const std::string& vd, const std::string& is, const std::string& n);

// Токи BJT NPN: ic втекает в коллектор, ib — в базу (эмиттер отдаёт ic + ib). Для PNP вызывать
// с veb, vcb и менять знак токов. vaf_inv / var_inv — 1/VAF, 1/VAR (пусто — эффекта Эрли нет).
struct BjtCurrents { std::string ic, ib; };
BjtCurrents bjt_currents(const std::string& vbe, const std::string& vbc,
                         const std::string& is, const std::string& bf, const std::string& br,
                         const std::string& nf, const std::string& nr,
                         const std::string& vaf_inv, const std::string& var_inv);

// Рабочая точка ДАУ (s.dae_nx >= 0): f(x, z, t0) = 0 и g(x, z, t0) = 0 по ВСЕМ переменным —
// у схемы это расчёт по постоянному току (токи конденсаторов 0, напряжения индуктивностей 0).
// Ньютон с дроблением шага (Армихо по max|F|) и ограничением поправки; X — старт и результат.
// false + why: вырожденный якобиан (узел без пути к земле по постоянному току, контур из
// источников напряжения, равновесие не изолировано) или нет сходимости.
bool dae_dc_operating_point(const System& s, const double* a, int na, double t0, double* X,
                            int* iters, std::string* why);
