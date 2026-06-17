#pragma once
#include "analysis_session.h"
#include <string>

// Сериализация рабочего состояния сессии анализа в JSON и обратно.
// Сохраняется в sessions/<name>.json внутри папки системы.
// Эталон системы (system.json) при этом не трогается — session отдельно,
// чтобы был возможен "сброс до базовых".
//
// Сохраняем: step_h, sim_time, skip_time, scheme, decimation,
//            auto_recompute, legend_show_ic, use_gpu,
//            param_values, ic_sets, projections.
// Лейаут окон — зарезервировано (поле layout пустое, заполним позже).

// Сериализовать состояние сессии в JSON-строку.
std::string session_to_json(const PhaseAnalysisSession& s);

// Восстановить состояние сессии из JSON. Применяет только поля состояния
// (vars/params/sys НЕ трогает — они приходят из системы). Возвращает false
// при ошибке разбора (s остаётся как есть).
bool session_from_json(const std::string& json, PhaseAnalysisSession& s);

// То же самое для ParametricAnalysisSession.
// Сохраняем: scheme, диапазон/n_pts, writable_var, integration-поля,
// param_values, initial_conditions, csv_output_path + csv_save_enabled.
// Не сохраняем результат и runtime-флаги.
std::string session_to_json_parametric(const ParametricAnalysisSession& s);
bool session_from_json_parametric(const std::string& json, ParametricAnalysisSession& s);
