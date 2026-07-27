#pragma once
#include "codegen.hpp"
#include "configCUDA.h"   // typedef numb — состояние считается в точности GPU
#include <vector>
#include <string>

// Расширяемый набор схем интегрирования поверх SystemEvaluator.
// Добавление новой схемы = добавить enum + одну функцию шага в .cpp,
// не трогая расчёт траектории и остальные схемы.
enum class IntScheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78, CD };

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
