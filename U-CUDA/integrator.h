#pragma once
#include "codegen.hpp"
#include <vector>
#include <string>

// Расширяемый набор схем интегрирования поверх SystemEvaluator.
// Добавление новой схемы = добавить enum + одну функцию шага в .cpp,
// не трогая расчёт траектории и остальные схемы.
enum class IntScheme { Euler, EulerCromer, ExplicitMidpoint, RK4, DOPRI78 };

IntScheme int_scheme_from_string(const std::string& s);

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
