#pragma once
// Синтез аналоговой схемы на ОУ по системе ОДУ (фаза 1 плана): функции. Типы (PolyRhs,
// ScalePlan, CircuitGraph, SynthesisConfig) — в ucuda/domain/circuit_synth.h.
#include <string>
#include <vector>
#include "ucuda/domain/circuit_synth.h"

// --- Разложение правой части на мономы ---------------------------------------

// Снимает мономы правых частей при ОДНОМ наборе параметров.
//   values — массив в раскладке a[]: values[0] зарезервирован, values[1..M] — параметры.
// false + err, если правая часть не полином степени <= 2 (невязка выше порога)
// либо если система содержит недифференцируемые функции.
bool extract_quadratic(const System& sys,
                       const std::vector<double>& values,
                       const PolyExtractConfig& cfg,
                       std::vector<PolyRhs>& out,
                       std::string* err);

// То же, но по НЕСКОЛЬКИМ наборам параметров: структура берётся объединением
// ненулевых мономов, числа — из values_sets[0].
//
// Зачем: коэффициент, обнулившийся именно на текущем наборе (sigma = 0 и т.п.),
// иначе выкинул бы элемент из топологии, и правка параметра в UI его бы не
// вернула — пересинтез topology_generation не бампается.
bool extract_quadratic_union(const System& sys,
                             const std::vector<std::vector<double>>& values_sets,
                             const PolyExtractConfig& cfg,
                             std::vector<PolyRhs>& out,
                             std::string* err);

// --- Масштабирование по амплитуде ---------------------------------------------

// Масштабы из размахов траектории: traj[шаг][координата], переходный участок
// вызывающий отбрасывает сам. Вырожденную переменную (размах ~0) оставляет
// единичной, иначе делили бы на ноль.
bool plan_amplitude_scaling(const std::vector<std::vector<double>>& traj,
                            double target_volt, ScalePlan& out, std::string* err);

// Пересчёт коэффициентов под масштаб: du_i/dt = f_i(S*u) / S_i.
void apply_amplitude_scaling(const std::vector<PolyRhs>& in, const ScalePlan& sp,
                             std::vector<PolyRhs>& out);

// --- Граф схемы ---------------------------------------------------------------

// Строит граф по разложенным правым частям. false + err на нереализуемом
// мономе (степень 3 появиться не может — её отсеивает extract_*).
bool synthesize_circuit(const System& sys,
                        const std::vector<PolyRhs>& rhs,
                        const SynthesisConfig& cfg,
                        CircuitGraph& out,
                        std::string* err);

// --- Диагностика и сравнение --------------------------------------------------

// Канонический текстовый слепок топологии: имена узлов нормализованы, элементы
// отсортированы. Две схемы одинаковы тогда и только тогда, когда слепки равны.
std::string circuit_topology_signature(const CircuitGraph& g);

// Человекочитаемый дамп со всеми номиналами — для отладочной панели и логов.
std::string circuit_dump(const CircuitGraph& g);

// Проверки, которые дешевле сделать здесь, чем ловить вырожденной матрицей в
// решателе: висячий узел, узел без пути на землю, выход ОУ, никуда не идущий.
bool circuit_check(const CircuitGraph& g, std::vector<std::string>& problems);

// Прогон синтезатора на масштабированном Лоренце с проверкой против
// tests/fixtures/lorenz_reference.cir. Возвращает отчёт; ok — сошлось ли.
std::string circuit_synth_selftest(bool* ok);
