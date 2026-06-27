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

// То же самое для BifurcationAnalysisSession (1D-бифуркация — список БД).
// Сохраняем: scheme, диапазон/n_pts, writable_var, integration-поля,
// param_values, initial_conditions, csv_output_path + csv_save_enabled.
// Не сохраняем результат и runtime-флаги.
std::string session_to_json_parametric(const BifurcationAnalysisSession& s);
bool session_from_json_parametric(const std::string& json, BifurcationAnalysisSession& s);

// LLE-сессия — список кривых λ(param). Каждая со своим scheme/range/IC/params,
// плюс eps/NT. Хранится отдельным файлом (`_last_lle.json`).
std::string session_to_json_lle(const LLEAnalysisSession& s);
bool session_from_json_lle(const std::string& json, LLEAnalysisSession& s);

// LS-сессия — список спектров (полей exponents). Те же поля что у LLE — отличается
// только Result, который JSON не хранит.
std::string session_to_json_ls(const LyapunovSpectrumAnalysisSession& s);
bool session_from_json_ls(const std::string& json, LyapunovSpectrumAnalysisSession& s);

// Basins-сессия — один config (без curves-vector). Сохраняем axes/ranges,
// integration, DBSCAN eps, IC, params, CSV-настройки. Result в JSON не пишется.
std::string session_to_json_basins(const BasinsAnalysisSession& s);
bool session_from_json_basins(const std::string& json, BasinsAnalysisSession& s);
