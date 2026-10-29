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
// уходит в inf (у SPICE ту же роль играет pnjlim). Ёмкости переходов CJO / CJE / CJC / CGS / CGD —
// барьерные, от напряжения (junction_capacitance); TT, BV, IKF, ISE и прочее не моделируются —
// импорт перечисляет их в отчёте. MOSFET — level 1 (mosfet_drain_current).

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

// Ток стока N-JFET (Шихман–Ходжес, SPICE level 1), втекает в сток: при vds >= 0
//   Id = BETA vm (2 vov - vm)(1 + LAMBDA vds), vov = max(vgs - VTO, 0), vm = min(vds, vov)
// (vm = vds — линейная область, vm = vov — насыщение, vov = 0 — отсечка); при vds < 0 сток и
// исток меняются ролями (вычитается то же от vgd, -vds). Ветвлений нет — только min/max.
// P-JFET: вызывать с vsg, vsd, vdg и менять знак тока.
std::string jfet_drain_current(const std::string& vgs, const std::string& vds, const std::string& vgd,
                               const std::string& vto, const std::string& beta, const std::string& lambda);

// Ток стока MOSFET level 1 (Шихман–Ходжес) — та же формула, BETA = KP W / (2 L), но порог свой у
// каждой половины: vt_s — от vbs (прямой режим), vt_d — от vbd (сток и исток поменялись ролями).
// Без эффекта подложки vt_s = vt_d = VTO (с учётом типа). PMOS: vsg, vsd, vdg и знак тока, как у P-JFET.
std::string mosfet_drain_current(const std::string& vgs, const std::string& vds, const std::string& vgd,
                                 const std::string& vt_s, const std::string& vt_d,
                                 const std::string& beta, const std::string& lambda);

// Порог с эффектом подложки: VT0 + GAMMA (sqrt(PHI - vbs) - sqrt(PHI)); под корнем — не меньше
// 0.01 PHI (прямое смещение подложки: у SPICE там своё сглаживание, здесь — излом).
std::string mosfet_threshold(const std::string& vt0, const std::string& gamma, const std::string& phi,
                             const std::string& vbs);

// Барьерная ёмкость перехода по формуле SPICE от напряжения v (прямое — положительно):
//   v <= FC VJ:  C = CJ0 (1 - v/VJ)^(-M);  выше — касательная: CJ0/(1 - FC)^M + CJ0 M/(VJ (1 - FC)^(1+M)) (v - FC VJ).
// Записана через min/max, без ветвлений. Диффузионная (TT) не моделируется.
std::string junction_capacitance(const std::string& v, const std::string& cj0, const std::string& vj,
                                 const std::string& m, const std::string& fc);

// Рабочая точка ДАУ (s.dae_nx >= 0): f(x, z, t0) = 0 и g(x, z, t0) = 0 по ВСЕМ переменным —
// у схемы это расчёт по постоянному току (токи конденсаторов 0, напряжения индуктивностей 0).
// Ньютон с дроблением шага (Армихо по max|F|) и ограничением поправки; X — старт и результат.
// false + why: вырожденный якобиан (узел без пути к земле по постоянному току, контур из
// источников напряжения, равновесие не изолировано) или нет сходимости.
bool dae_dc_operating_point(const System& s, const double* a, int na, double t0, double* X,
                            int* iters, std::string* why);
