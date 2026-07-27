#pragma once
#include "analysis_session.h"
#include "custom_session.h"
#include "app_model.h"
#include <string>
#include <vector>

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

// Dft1D-сессия — multi-config layout как у Basins/FastSync. Сохраняем
// sweep/integration/IC/params/frequency-range/display/normalize/CSV.
// Result и display_cache* в JSON не пишутся.
std::string session_to_json_dft1d(const Dft1DAnalysisSession& s);
bool session_from_json_dft1d(const std::string& json, Dft1DAnalysisSession& s);

// Basins-сессия — один config (без curves-vector). Сохраняем axes/ranges,
// integration, DBSCAN eps, IC, params, CSV-настройки. Result в JSON не пишется.
std::string session_to_json_basins(const BasinsAnalysisSession& s);
bool session_from_json_basins(const std::string& json, BasinsAnalysisSession& s);

// FastSync-сессия — multi-config layout как у basins. Result в JSON не пишется.
std::string session_to_json_fastsync(const FastSyncAnalysisSession& s);
bool session_from_json_fastsync(const std::string& json, FastSyncAnalysisSession& s);

// Parametric plot windows (AppModel::parametric_plot_windows) — the dynamic
// window list, spans all 3 sessions so it's its own file (`_last_parametric_windows.json`)
// rather than living inside one of the per-kind files above.
std::string session_to_json_parametric_windows(const std::vector<ParametricPlotWindow>& wins);
bool session_from_json_parametric_windows(const std::string& json, std::vector<ParametricPlotWindow>& wins);

// DFT1D plot windows (AppModel::dft1d_plot_windows) — same idea, its own file
// (`_last_dft1d_windows.json`), minus the kind/mode_2d/colored_1d fields.
std::string session_to_json_dft1d_windows(const std::vector<Dft1DPlotWindow>& wins);
bool session_from_json_dft1d_windows(const std::string& json, std::vector<Dft1DPlotWindow>& wins);

// Custom-tab bundle (`_last_custom.json`) — wraps shared pipeline config plus
// each of the five owned sub-sessions in one file. Sub-sessions are embedded
// as nested objects using the existing per-mode writers, so future fields on
// any sub-session flow into the bundle automatically.
std::string session_to_json_custom(const CustomSession& s);
bool session_from_json_custom(const std::string& json, CustomSession& s);
