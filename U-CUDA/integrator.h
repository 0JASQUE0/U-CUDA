#pragma once
#include "codegen.hpp"
#include "configCUDA.h"   // typedef numb — состояние считается в точности GPU
#include "adaptive_settings.h"
#include <atomic>
#include <vector>
#include <string>

// Расширяемый набор схем интегрирования поверх SystemEvaluator.
// Добавление новой схемы = добавить enum + одну функцию шага в .cpp,
// не трогая расчёт траектории и остальные схемы.
// Map is not a scheme but the discrete-map step itself (h is unused).
enum class IntScheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD, ComplexCD, ComplexCD4,
                       ComplexCD4S3, ComplexCD4S4, ComplexCD4SS01, ComplexCD4SS10,
                       CD10, ComplexCD10, ComplexCD4_10, ComplexCD4S3_10, ComplexCD4S4_10,
                       ImplicitEuler, ImplicitMidpoint, SEMP, SIMP, D, ComplexIEuler,
                       GBS, GBS24, GBS246, GBS2468, GBS246810, GBS24681012, DOPRI78Legacy,
                       RK45, DOP853, Map };

IntScheme int_scheme_from_string(const std::string& s);

// Шаг, скомпилированный из пользовательской КРС (см. krs_cpu.h). Сигнатура
// совпадает с calculateDiscreteModel на GPU — включая тип numb: тело мутирует
// X[] в той же точности, в какой считал бы kernel.
using CustomStepFn = void (*)(numb* X, const numb* a, numb h);

// Считает одну траекторию на CPU через интерпретатор (быстро, смена системы на лету).
//   ev        — интерпретатор системы (уже распарсенный)
//   scheme    — схема интегрирования
//   ic        — начальные условия [dim]
//   a         — параметры со сдвигом [>= nparams+1], a[0] не используется
//   h         — шаг
//   total     — число записываемых точек
//   skip      — число шагов transient (без записи)
//   out       — [total][dim] результат
// Возвращает false при расходимости (nan/inf).
bool computePhasePortraitCPU(
    const SystemEvaluator& ev,
    IntScheme scheme,
    const double* ic, int dim,
    const double* a, int amountOfValues,
    double h, int total, int skip,
    std::vector<std::vector<double>>& out);

// То же самое, но шаг задаётся готовой нативной функцией, а не парой
// (интерпретатор + встроенная схема). Используется для custom КРС: их тело —
// сырой C, SystemEvaluator его не понимает. Transient, запись точек и
// проверка на расходимость — тот же код, что и выше.
//
// ic/a/out остаются double: это интерфейс и хранилище, общее с GPU-путём
// (оттуда результат тоже приезжает расширенным до double). Само интегрирование
// внутри идёт в numb.
bool computePhasePortraitCPU_custom(
    CustomStepFn step,
    const double* ic, int dim,
    const double* a, int amountOfValues,
    double h, int total, int skip,
    std::vector<std::vector<double>>& out);

// Схема со встроенной оценкой ошибки (RK45, DOPRI78 обоих видов, DOP853).
bool int_scheme_supports_adaptive(IntScheme s);

// Фазовый портрет с адаптивным шагом на CPU: тот же драйвер
// (kernels/ucuda_adaptive.cuh), что у ядра phase_kernel_ad, функции схемы — через
// вычислитель правых частей. Транзиент кончается ровно в t_skip; дальше либо
// равномерная сетка (total точек через dt, первая в t_skip), либо узлы шага (raw,
// не больше max_pts; times — их моменты). log — попытки {t, h, err, код}.
// ctrl_fn, prep_fn — пользовательский регулятор и его раздел подготовки (CtrlCpuFn);
// ctrl_fn обязателен при P.ctrl = CUSTOM.
// false — схема без оценки ошибки или решение разошлось (traj тогда короче).
bool computePhasePortraitCPU_adaptive(
    const SystemEvaluator& ev, IntScheme scheme,
    const double* ic, int dim, const double* a, const UcudaAdaptParams& P,
    bool raw, double t_skip, double t_rec, double dt, int total, int max_pts, int log_cap,
    std::vector<std::vector<double>>& traj, std::vector<double>& times,
    std::vector<double>& log, AdaptiveStats& stats, double& final_h,
    UcudaCtrlCustomFn ctrl_fn = nullptr, UcudaCtrlPrepFn prep_fn = nullptr,
    const std::atomic<bool>* cancel = nullptr);   // отмена: смотрится раз в 1024 принятых шага
