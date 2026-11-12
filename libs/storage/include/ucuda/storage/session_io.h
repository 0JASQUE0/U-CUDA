#pragma once
#include "ucuda/domain/analysis_session.h"
#include "ucuda/domain/custom_session.h"
#include "ucuda/domain/network_session.h"
#include "ucuda/domain/neural_session.h"
#include "ucuda/domain/order_session.h"
#include "ucuda/domain/plot_window.h"
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
std::string session_to_json(const PhaseAnalysisState& s);

// Восстановить состояние сессии из JSON. Применяет только поля состояния
// (vars/params/sys НЕ трогает — они приходят из системы). Возвращает false
// при ошибке разбора (s остаётся как есть).
bool session_from_json(const std::string& json, PhaseAnalysisState& s);

// То же самое для BifurcationAnalysisSession (1D-бифуркация — список БД).
// Сохраняем: scheme, диапазон/n_pts, writable_var, integration-поля,
// param_values, initial_conditions, csv_output_path + csv_save_enabled.
// Не сохраняем результат и runtime-флаги.
std::string session_to_json_parametric(const BifurcationAnalysisState& s);
bool session_from_json_parametric(const std::string& json, BifurcationAnalysisState& s);

// LLE-сессия — список кривых λ(param). Каждая со своим scheme/range/IC/params,
// плюс eps/NT. Хранится отдельным файлом (`_last_lle.json`).
std::string session_to_json_lle(const LLEAnalysisState& s);
bool session_from_json_lle(const std::string& json, LLEAnalysisState& s);

// LS-сессия — список спектров (полей exponents). Те же поля что у LLE — отличается
// только Result, который JSON не хранит.
std::string session_to_json_ls(const LyapunovSpectrumAnalysisState& s);
bool session_from_json_ls(const std::string& json, LyapunovSpectrumAnalysisState& s);

// Metrics-сессия (`_last_metrics.json`) — список конфигов с маской метрик.
// Result не хранится.
std::string session_to_json_metrics(const SignalMetricsAnalysisState& s);
bool session_from_json_metrics(const std::string& json, SignalMetricsAnalysisState& s);

// Dft1D-сессия — multi-config layout как у Basins/FastSync. Сохраняем
// sweep/integration/IC/params/frequency-range/display/normalize/CSV.
// Result и display_cache* в JSON не пишутся.
std::string session_to_json_dft1d(const Dft1DAnalysisState& s);
bool session_from_json_dft1d(const std::string& json, Dft1DAnalysisState& s);

// Basins-сессия — один config (без curves-vector). Сохраняем axes/ranges,
// integration, DBSCAN eps, IC, params, CSV-настройки. Result в JSON не пишется.
// Проекции фазовых портретов живут в слотах сессии (analysis), поэтому идут отдельным
// списком: на запись — по config'у (nullptr = "[]"), на чтение — в phase_projections, и
// slots говорит владельцу слотов, что с ними сделать. Обёртки над сессией целиком —
// session_to_json_basins / session_from_json_basins в application/session_persistence.h.
enum class BasinsSlotsAfterLoad {
    Keep,     // JSON без полей: слоты не трогать
    Ensure,   // конфигов нет: подогнать слоты под configs
    Reset,    // пересоздать слоты и разложить phase_projections
};
std::string basins_state_to_json(const BasinsAnalysisState& s,
                                 const std::vector<const std::vector<Projection>*>& phase_projections);
bool basins_state_from_json(const std::string& json, BasinsAnalysisState& s,
                            std::vector<std::vector<Projection>>& phase_projections,
                            BasinsSlotsAfterLoad& slots);

// FastSync-сессия — multi-config layout как у basins. Result в JSON не пишется.
std::string session_to_json_fastsync(const FastSyncAnalysisState& s);
bool session_from_json_fastsync(const std::string& json, FastSyncAnalysisState& s);

// Network-сессия — конфиги вместе с топологией (узлы, рёбра, законы связи).
// Result не пишется: он на порядки больше сессии и восстанавливается одним Run.
std::string session_to_json_network(const NetworkState& s);
bool session_from_json_network(const std::string& json, NetworkState& s);
// Neural-сессия: рецепт сети (популяции, проекции, seed) с хешем графа; рёбер в файле нет.
std::string session_to_json_neural(const NeuralState& s);
// Runtime comparison only: excludes output paths, last-run fields and unused saved models.
std::string neural_settings_key(const NeuralConfig& c);
bool session_from_json_neural(const std::string& json, NeuralState& s);
// Обученный выход Neural отдельным JSON с формулой применения, для обработки вне U-CUDA.
void neural_readout_export(const std::string& path, const NeuralReadoutSaved& r);
// Обратное к экспорту; бросает на чужом формате или несогласованных размерах.
NeuralReadoutSaved neural_readout_import(const std::string& path);

// Parametric plot windows (AppModel::parametric_plot_windows) — the dynamic
// window list, spans all 3 sessions so it's its own file (`_last_parametric_windows.json`)
// rather than living inside one of the per-kind files above.
std::string session_to_json_parametric_windows(const std::vector<ParametricPlotWindow>& wins);
bool session_from_json_parametric_windows(const std::string& json, std::vector<ParametricPlotWindow>& wins);

// Order-сессия (`_last_order.json`) — multi-config layout как у DFT1D.
// Настройки отображения в неё не идут: они на окнах графика (ниже).
std::string session_to_json_order(const OrderAnalysisState& s);
bool session_from_json_order(const std::string& json, OrderAnalysisState& s);

// Окна графиков Order (`_last_order_windows.json`) — вид окна, его члены и
// настройки отображения. Отдельный файл по той же причине, что у параметрики:
// это не часть ни одного конфига.
std::string session_to_json_order_windows(const std::vector<OrderPlotWindow>& wins);
bool session_from_json_order_windows(const std::string& json, std::vector<OrderPlotWindow>& wins);

// DFT1D plot windows (AppModel::dft1d_plot_windows) — same idea, its own file
// (`_last_dft1d_windows.json`), minus the kind/mode_2d/colored_1d fields.
std::string session_to_json_dft1d_windows(const std::vector<Dft1DPlotWindow>& wins);
bool session_from_json_dft1d_windows(const std::string& json, std::vector<Dft1DPlotWindow>& wins);

// Проекции Phase analysis отдельно от сессии — массив в формате блока "projections".
// Нужны сохранённым раскладкам окон (layouts/*.layout): по ним недостающая проекция
// создаётся заново.
std::string session_to_json_projections(const std::vector<Projection>& projs);
std::string session_to_json_projection(const Projection& proj);   // массив из одной проекции
bool session_from_json_projections(const std::string& json, std::vector<Projection>& projs);

// Custom-tab bundle (`_last_custom.json`) — wraps shared pipeline config plus
// each of the five owned sub-sessions in one file. Sub-sessions are embedded
// as nested objects using the existing per-mode writers, so future fields on
// any sub-session flow into the bundle automatically. Sub-sessions are owned by
// CustomSession (analysis), so storage gets their states explicitly; whole-session
// wrappers live in application/session_persistence.h.
struct CustomSubStates {
    const BifurcationAnalysisState&      bif;
    const LLEAnalysisState&              lle;
    const LyapunovSpectrumAnalysisState& ls;
    const PhaseAnalysisState&            phase;
    const BasinsAnalysisState&           basins;
    std::vector<const std::vector<Projection>*> basins_phase_projections;
};
struct CustomSubStatesMut {
    BifurcationAnalysisState&      bif;
    LLEAnalysisState&              lle;
    LyapunovSpectrumAnalysisState& ls;
    PhaseAnalysisState&            phase;
    BasinsAnalysisState&           basins;
    std::vector<std::vector<Projection>> basins_phase_projections;
    BasinsSlotsAfterLoad basins_slots = BasinsSlotsAfterLoad::Keep;
};
std::string custom_state_to_json(const CustomState& s, const CustomSubStates& sub);
bool custom_state_from_json(const std::string& json, CustomState& s, CustomSubStatesMut& sub);

// Библиотека регуляторов шага (adaptive_ctrl_library() в adaptive_settings.h): файл
// step_controllers.json в каталоге library. load — при старте, запоминает путь (нет
// файла — пустая библиотека, не ошибка); save — после правки в редакторе и при импорте
// регулятора из сессии (write_adaptive кладёт определение в сессию целиком).
bool load_ctrl_library(const std::string& path, std::string* err = nullptr);
bool save_ctrl_library(std::string* err = nullptr);
