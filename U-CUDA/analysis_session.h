#pragma once
#include "system_record.h"
#include "codegen.hpp"
#include "parametric_engine.h"
#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <vector>
#include <map>
#include "plot_view_2d.h"
#include "plot_view_3d.h"
#include <memory>


// Одно начальное условие: имя (для легенды) + значения по переменным.
// values[varName] = строка ("" = 0). Храним строки для единообразия с GUI.
struct InitialConditionSet {
    std::string label;                          // "IC 1", можно переименовать
    std::map<std::string, std::string> values;  // var -> значение
    bool visible = true;                        // показывать ли эту траекторию
};

// Одна проекция (окно-график): какие оси показывать.
// Пока только 2D: ось X = переменная axis_x, ось Y = axis_y.
// Тип проекции/графика.
enum class ProjType {
    Phase2D,    // фазовая плоскость: ось X = var, ось Y = var
    TimeDomain, // временная развёртка: ось X = время, Y = выбранные переменные
    Phase3D     // 3D фазовый портрет (нужен ImPlot3D)
};

struct Projection {
    std::string label;       // заголовок окна
    ProjType type = ProjType::Phase2D;
    int axis_x = 0;          // индекс переменной по X (Phase2D/3D)
    int axis_y = 1;          // индекс переменной по Y
    int axis_z = 2;          // индекс переменной по Z (Phase3D)
    // для TimeDomain: какие переменные выводить (по индексу). Размер синхр. с vars.
    std::vector<bool> show_var;
    bool open = true;        // окно открыто (крестик)

    // Опционально: per-projection line styling. По дефолту false → быстрый
    // GL shader-line путь (glLineWidth клампится драйвером, alpha = 1). При
    // true → ImDrawList::AddLine per segment с заданными толщиной + α (живой
    // line_width, alpha — но медленнее на длинных траекториях). Toolbar над
    // плотом переключает + показывает слайдеры.
    bool  custom_line_style = false;
    float line_width        = 1.5f;
    float alpha             = 1.0f;

    std::unique_ptr<Plot2DView> view2d;
    std::unique_ptr<Plot3DView> view3d;
    int prev_ax = -1, prev_ay = -1;

    Projection() = default;
    Projection(Projection&&) = default;
    Projection& operator=(Projection&&) = default;
    Projection(const Projection&) = delete;
    Projection& operator=(const Projection&) = delete;
};

// Результат расчёта: по одной траектории на каждое НУ.
// trajectories[k] = траектория для k-го НУ (точки {x,y,z,...}).
struct AnalysisResult {
    std::vector<std::vector<std::vector<double>>> trajectories; // [ic][step][coord]
    std::vector<std::string> labels;   // подписи (из НУ)
    std::vector<bool> visible;         // видимость (из НУ)
    std::vector<std::string> ic_text;  // текстовое представление НУ (для легенды)
    bool ok = false;
    std::string error;
    int generation = 0;

    // Snapshot of the inputs that produced this trajectory set — used by
    // right-click "Export data..." in the GUI to write the same scheme/IC
    // metadata into <path>_config.csv that drove the compute.
    data_export::PhaseSnapshot snapshot;
};

// Сессия анализа фазовых портретов: общие параметры системы +
// список НУ + список проекций. Параметры общие на все проекции.
// Это "песочница": при загрузке из библиотеки сюда копируются настройки,
// изменения тут НЕ сохраняются обратно в библиотеку.
struct PhaseAnalysisSession {
    // система (для расчёта нужны имена переменных/параметров и КРС-движок)
    std::vector<std::string> vars;     // имена переменных (порядок = индексы X[])
    std::vector<std::string> params;   // имена параметров (порядок = a[1..])

    // Пользовательские КРС из текущей системы (копия из SystemRecord).
    // Доступны в scheme combo вместе с built-in именами.
    std::vector<CustomScheme> custom_schemes;

    // общие параметры системы (строки, "" = не задано/0)
    std::map<std::string, std::string> param_values; // param -> значение
    std::string step_h = "0.01";
    std::string sim_time = "250";       // время моделирования (с)
    std::string skip_time = "100";      // transient (с)
    std::string scheme = "Euler";      // выбор метода (Euler/EulerCromer/Midpoint/RK4)
    std::string symmetry_s = "0.5";    // коэф. симметрии s для CD (a[0])
    std::string decimation = "1";      // выводить каждую N-ю точку
    bool auto_recompute = false;       // пересчитывать сразу при изменении
    bool legend_show_ic = false;       // в легенде показывать НУ вместо имён графиков

    // система (уравнения) для генерации КРС под NVRTC, и сама готовая КРС.
    System sys;                        // правые части и т.д.
    std::string krs_code;              // тело calculateDiscreteModel (для NVRTC)
    bool use_gpu = true;               // расчёт через NVRTC на GPU (CPU оставлен в коде)

    // Перегенерировать krs_code из sys по текущему scheme. Зовётся при загрузке
    // системы и при смене метода. Дёшево (codegen быстрый).
    void regenerate_krs();

    // несколько начальных условий (мультистабильность)
    std::vector<InitialConditionSet> ic_sets;

    // несколько проекций
    std::vector<Projection> projections;

    // Имя системы, под которую сессия инициализирована (через load_from_record).
    // Используется в GUI: при переключении радио-кнопок в один и тот же режим
    // не сбрасывать пользовательские правки, если система не менялась.
    std::string loaded_system_name;

    // последний результат расчёта
    AnalysisResult result;

    // запрос автоскейла осей после пересчёта (взводится в recompute,
    // сбрасывается в GUI после применения)
    bool fit_request = false;

    // --- async-расчёт (как у BifurcationAnalysisSession) ---
    std::future<AnalysisResult> recompute_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // session не копируется (содержит future) — только move.
    PhaseAnalysisSession() = default;
    PhaseAnalysisSession(PhaseAnalysisSession&&) = default;
    PhaseAnalysisSession& operator=(PhaseAnalysisSession&&) = default;
    PhaseAnalysisSession(const PhaseAnalysisSession&) = delete;
    PhaseAnalysisSession& operator=(const PhaseAnalysisSession&) = delete;

    // поколение раскладки окон: увеличивается кнопкой "Reset layout",
    // входит в ID окон проекций -> docking забывает их позиции, окна
    // появляются заново (можно сделать новую раскладку).
    int layout_generation = 0;

    // --- операции ---
    void add_ic();              // добавить НУ (пустое)
    void remove_ic(int i);
    void add_projection();      // добавить проекцию (x-y по умолчанию)
    void remove_projection(int i);

    // Загрузить систему/настройки из записи библиотеки (копия, не ссылка).
    void load_from_record(const SystemRecord& r,
        const std::vector<std::string>& vars_,
        const std::vector<std::string>& params_);

    // Пересчитать все траектории (по кнопке или авто). Sync — блокирует поток.
    void recompute();

    // Async-аналог: снапшотит входы на главном потоке, спавнит worker.
    // Результат применяется в poll() позже. Возвращает false если уже идёт расчёт.
    bool recompute_async();

    // Раз в кадр из GUI. Если worker готов — забирает result, выставляет
    // fit_request и data_generation, сбрасывает in_flight. Возвращает true
    // в кадр завершения (для авто-сохранения).
    bool poll();

    int data_generation = 0;
};

// Одна бифуркационная диаграмма (БД) на параметрическом графике:
// независимый свип, свои IC/scheme/range/CSV/peaks-vs-inter-peaks.
// Несколько БД оверлеятся на один график с легендой (см. BifurcationAnalysisSession).
struct BifurcationDiagramConfig {
    std::string label   = "BD 1";   // подпись для легенды, редактируется
    bool        visible = true;     // toggle в легенде (как visible у IC)

    // Схема интегрирования: built-in имя или имя custom-схемы (resolve через
    // session.custom_schemes на момент Run).
    std::string scheme         = "Euler";
    std::string symmetry_s     = "0.5";    // a[0] для CD

    // Свип: какой параметр, его диапазон, число точек.
    int         param_index    = 0;
    // Если sweep_over_var=true — БД строится по начальному условию переменной
    // vars[var_sweep_index] вместо параметра. Дефолт false → param-свип (старое
    // поведение). NonLinAnal-kernel принимает par_or_var runtime-аргументом,
    // переключаем только индекс + флаг в engine — никакой пересборки PTX.
    bool        sweep_over_var = false;
    int         var_sweep_index = 0;
    // Continuation: следующая точка параметра стартует с конечного x[]
    // предыдущей. Требует sweep_over_var=false. Reverse — направление
    // обхода (forward lo→hi vs backward hi→lo) для visualisation of
    // hysteresis (forward и reverse BD на одном плоте).
    bool        continuation = false;
    bool        continuation_reverse = false;
    std::string param_lo_text  = "0";
    std::string param_hi_text  = "1";
    std::string n_pts_text     = "500";

    // По какой переменной строим диаграмму.
    int         writable_var   = 0;

    // Интегрирование.
    std::string h_text             = "0.01";
    std::string t_max_text         = "100";
    std::string transient_text     = "100";
    std::string pre_scaller_text   = "1";
    std::string max_value_text     = "1e6";

    // CSV — путь хранится отдельно от флага, чтобы можно было выключить запись,
    // не удаляя сам путь (для повторного включения).
    bool        csv_save_enabled = false;
    std::string csv_output_path;

    // false → точки = значения пиков (bifurcation_points). true → межпиковые
    // интервалы (peak_times). Колонка 3 vs 2 в CSV.
    bool        plot_inter_peaks = false;

    // Per-БД базовые значения параметров и НУ.
    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    // Per-БД результат + GUI-флаги.
    Bifurcation1DResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;

    // ---- 2D-режим (хитмап «период»(p1, p2) через DBSCAN) ----
    bool        mode_2d           = false;
    int         param_index_2     = 0;
    bool        sweep_over_var_2  = false;
    int         var_sweep_index_2 = 0;
    std::string param_lo_2_text   = "0";
    std::string param_hi_2_text   = "1";
    std::string eps_dbscan_text   = "0.1";

    Bifurcation2DResult result_2d;
    bool        last_run_2d_ok    = false;
    int         data_generation_2d = 0;
    bool        fit_request_2d    = false;
};

// Сессия параметрического анализа: одна система + список БД, каждая со своим
// конфигом и результатом. Async-очередь одна на сессию (одна БД в полёте за раз).
struct BifurcationAnalysisSession {
    // Идентичность системы — общая на все БД.
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::string loaded_system_name;

    // Список бифуркационных диаграмм. После load_from_record содержит как
    // минимум один дефолтный элемент.
    std::vector<BifurcationDiagramConfig> diagrams;

    // Какая вкладка сейчас открыта — для Ctrl+R (последняя видимая).
    int active_diagram_index = 0;
    // Индекс БД, чей расчёт сейчас идёт в worker'е; -1 = ничего.
    int running_diagram_index = -1;

    // --- async-расчёт ---
    // Все мутации session происходят на главном потоке (run_async/poll),
    // worker трогает только engine + собственную копию Request.
    std::future<Bifurcation1DResult> run_future;
    std::future<Bifurcation2DResult> run_future_2d;
    bool is_2d_run = false;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // Cooperative cancellation. Token is created in run_async() and shared
    // with the worker via Request::cancel; request_cancel() sets it to true,
    // engine bails out at the next cuDeviceSynchronize and returns
    // cancelled=true. Token is reset after poll() consumes the result.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    // Progress (0..1) written by engine on each chunk, read by GUI each frame.
    std::shared_ptr<std::atomic<float>> progress_token;
    // Persistent info about the last finished run — shown in the busy bar
    // after completion until the next run_async clears it. last_run_label
    // empty == no completion to display.
    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    // session не копируется (содержит future) — только move.
    BifurcationAnalysisSession() = default;
    BifurcationAnalysisSession(BifurcationAnalysisSession&&) = default;
    BifurcationAnalysisSession& operator=(BifurcationAnalysisSession&&) = default;
    BifurcationAnalysisSession(const BifurcationAnalysisSession&) = delete;
    BifurcationAnalysisSession& operator=(const BifurcationAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    // Добавить БД: глубокая копия последней (если есть), иначе дефолт из vars/params.
    // Label автоматически "BD <N+1>", результат и флаги обнуляются.
    void add_diagram();
    void remove_diagram(int i);

    // Старый синхронный Run для конкретной БД (по индексу).
    bool run(ParametricEngine& engine, int diagram_idx);

    // Async-Run для конкретной БД. Возвращает false если очередь занята.
    bool run_async(ParametricEngine& engine, int diagram_idx);

    // Cooperative cancel of the in-flight run. No-op if nothing is in flight.
    void request_cancel();

    // Вызывается каждый кадр из GUI-потока. Если future готова — применяет
    // результат к diagrams[running_diagram_index], сбрасывает in_flight.
    // Возвращает true в кадр завершения.
    bool poll();
};

// Одна кривая LLE(param) на параметрическом LLE-графике — структурно
// зеркалит BifurcationDiagramConfig (тот же UX). Отличия:
//   - нет writable_var (LLE — скаляр на точку параметра),
//   - нет plot_inter_peaks,
//   - + eps (возмущение Wolf/Benettin) и NT (длина блока интегрирования между
//     ренормализациями, в единицах времени).
struct LLECurveConfig {
    std::string label   = "LLE 1";
    bool        visible = true;

    std::string scheme         = "Euler";
    std::string symmetry_s     = "0.5";    // a[0] для CD

    int         param_index    = 0;
    // IC sweep: если sweep_over_var=true, кривая считается по нач. условию
    // переменной vars[var_sweep_index] вместо параметра. Engine выберет
    // соответствующий par_or_var compile-time через template substitution.
    bool        sweep_over_var = false;
    int         var_sweep_index = 0;
    std::string param_lo_text  = "0";
    std::string param_hi_text  = "1";
    std::string n_pts_text     = "500";

    std::string h_text             = "0.01";
    std::string t_max_text         = "100";
    std::string transient_text     = "100";
    std::string max_value_text     = "1e6";

    // LLE-специфика.
    std::string eps_text       = "1e-4";
    std::string nt_text        = "1";       // NT (в единицах времени)

    bool        csv_save_enabled = false;
    std::string csv_output_path;

    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    LLE1DResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;

    // ---- 2D-режим (LLE по двум параметрам/IC, рисуется хитмапой) ----
    // mode_2d=true → Run собирает LLE2DRequest вместо LLE1DRequest, результат
    // ложится в result_2d (не затирает result, чтобы переключение туда-обратно
    // не теряло последний 1D-расчёт). См. analysis_session.cpp:run_async.
    // Смешанный свип (одна ось param, другая IC) разруливается engine'ом
    // автоматически по sweep_over_var/_2 — отдельного флага не требуется.
    bool        mode_2d = false;
    // Sweep target для второй оси (структурно зеркалит param_index/sweep_over_var).
    int         param_index_2 = 0;
    bool        sweep_over_var_2 = false;
    int         var_sweep_index_2 = 0;
    std::string param_lo_2_text = "0";
    std::string param_hi_2_text = "1";
    // n_pts_2_text не нужен: kernel-getValueByIdx работает только на квадратной
    // сетке (cudaLibrary.cu:1276), используется n_pts_text для обеих осей.

    LLE2DResult result_2d;
    bool        last_run_2d_ok = false;
    int         data_generation_2d = 0;
    bool        fit_request_2d = false;
};

// Сессия LLE-анализа. Структура и API — копия BifurcationAnalysisSession
// (другие request/result и другая терминология «curve» вместо «diagram»).
struct LLEAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::string loaded_system_name;

    std::vector<LLECurveConfig> curves;
    int active_curve_index  = 0;
    int running_curve_index = -1;

    std::future<LLE1DResult> run_future;
    // Параллельный future для 2D-режима. in_flight (общий) гарантирует, что
    // в полёте одновременно только одна задача — 1D или 2D, не обе. Какая
    // именно — определяет is_2d_run (выставляется в run_async, читается poll).
    std::future<LLE2DResult> run_future_2d;
    bool is_2d_run = false;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    LLEAnalysisSession() = default;
    LLEAnalysisSession(LLEAnalysisSession&&) = default;
    LLEAnalysisSession& operator=(LLEAnalysisSession&&) = default;
    LLEAnalysisSession(const LLEAnalysisSession&) = delete;
    LLEAnalysisSession& operator=(const LLEAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void add_curve();
    void remove_curve(int i);
    bool run(ParametricEngine& engine, int curve_idx);
    bool run_async(ParametricEngine& engine, int curve_idx);
    void request_cancel();
    bool poll();
};

// ============================================================================
// Basins of Attraction — карта классификации траекторий на сетке IC.
// Одна конфигурация на сессию (без inner tab-bar — basin расчёт тяжёлый).
// 5 плотов в окне результата переключаются внутренним tab-bar'ом.
// ============================================================================

// Mirror BF_* кодов из configCUDA.h — для type-safe использования в GUI/host
// коде. Значения должны совпадать с BF_* (static_assert'ы в analysis_session.cpp
// ловят drift). Передаётся в kernel как plain int.
enum class BasinFeature : int {
    AvgPeaks            = 0,
    AvgIntervals        = 1,
    RMSPeaks            = 2,
    RMSIntervals        = 3,
    StDevPeaks          = 4,
    StDevIntervals      = 5,
    LogAvgPeaks         = 6,
    LogAvgIntervals     = 7,
    LogRMSPeaks         = 8,
    LogRMSIntervals     = 9,
    LogStDevPeaks       = 10,
    LogStDevIntervals   = 11,
};

struct BasinsConfig {
    std::string label = "Basins";
    std::string scheme = "Euler";
    std::string symmetry_s = "0.5";    // a[0] для CD

    int axis_x_var = 0;
    int axis_y_var = 1;
    std::string axis_x_lo_text = "-10";
    std::string axis_x_hi_text = "10";
    std::string axis_y_lo_text = "-10";
    std::string axis_y_hi_text = "10";
    std::string n_pts_text = "100";

    int writable_var = 0;

    std::string h_text           = "0.01";
    std::string t_max_text       = "200";
    std::string transient_text   = "50";
    std::string pre_scaller_text = "1";
    std::string max_value_text   = "1e6";
    std::string eps_dbscan_text  = "0.5";

    bool        csv_save_enabled = false;
    std::string csv_output_path;

    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    // Какие фичи пишутся в outAvgPeaks / AvgTimeOfPeaks из avgPeakFinderCUDA.
    // Значения 0..11 (см. enum BasinFeature / BF_* в configCUDA.h). Множители
    // mult_feature*_text применяются к финальному значению фичи (для подстройки
    // масштаба DBSCAN-кластеризации). Хранятся как text для единообразия с
    // остальными числовыми полями BasinsConfig.
    int         feature1 = BF_FEATURE1_DEFAULT;
    int         feature2 = BF_FEATURE2_DEFAULT;
    std::string mult_feature1_text = "1.0";
    std::string mult_feature2_text = "1.0";

    BasinsResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;

    // Активный внутренний таб плотов (0..4 = Basins/AvgPk/AvgInt/States/Scatter).
    int         active_plot_tab = 0;
};

struct BasinsAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::string loaded_system_name;

    // Список Basins-конфигов. После load_from_record содержит как минимум
    // один дефолтный элемент. Структура зеркалит BifurcationAnalysisSession::diagrams.
    std::vector<BasinsConfig> configs;

    // Какая вкладка сейчас открыта — для Ctrl+R (последняя видимая).
    int active_config_index = 0;
    // Индекс config'а, чей расчёт сейчас идёт в worker'е; -1 = ничего.
    int running_config_index = -1;

    std::future<BasinsResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    // Two-phase progress: 1 = sim, 2 = cluster. GUI uses this to render
    // "(1/2 sim)" or "(2/2 cluster)" suffix and reset the bar between phases.
    std::shared_ptr<std::atomic<int>>   progress_phase_token;
    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    BasinsAnalysisSession() = default;
    BasinsAnalysisSession(BasinsAnalysisSession&&) = default;
    BasinsAnalysisSession& operator=(BasinsAnalysisSession&&) = default;
    BasinsAnalysisSession(const BasinsAnalysisSession&) = delete;
    BasinsAnalysisSession& operator=(const BasinsAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    // Добавить config: глубокая копия последнего (если есть), иначе дефолт.
    // Label автоматически "Basins <N+1>", результат и флаги обнуляются.
    void add_config();
    void remove_config(int i);

    bool run(ParametricEngine& engine, int config_idx);
    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
};

// ============================================================================
// Fast Synchro — анализ возвратной (recurrent) синхронизации. Два режима:
//   mode 0 (On Attractor) — синхро-ошибка вдоль одной мастер-траектории;
//   mode 1 (On Grid)      — синхро-ошибка на 2D-сетке slave IC.
// Зеркалит BasinsConfig в стиле (multi-config, text-storage, futures worker).
// ============================================================================
struct FastSyncConfig {
    std::string label  = "FastSync 1";
    std::string scheme = "Euler";
    std::string symmetry_s = "0.5"; // a[0] для CD

    // Режим:
    int mode = 0;   // 0 = On Attractor, 1 = On Grid

    // ---- Общие параметры алгоритма (text-storage; parse при build_request) ----
    std::string h_text              = "0.01";
    std::string iter_of_synchr_text = "100";
    std::string pre_scaller_text    = "1";
    std::string max_value_text      = "1e6";

    // Per-var значения. Все 4 секции инициализируются нулями в load_from_record.
    std::map<std::string, std::string> ic_master;
    std::map<std::string, std::string> ic_slave;
    std::map<std::string, std::string> k_forward;
    std::map<std::string, std::string> k_backward;
    std::map<std::string, std::string> param_values;

    // ---- mode == 0 (On Attractor) ----
    std::string t_max_text         = "100";
    std::string transient_text     = "0";
    // Окно синхронизации (раньше называлось n_time). Хранится как text;
    // парсится в numb при build_request.
    std::string window_text        = "50";

    // ---- mode == 1 (On Grid) ----
    int         axis_x_var       = 0;
    int         axis_y_var       = 1;
    std::string axis_x_lo_text   = "-10";
    std::string axis_x_hi_text   = "10";
    std::string axis_y_lo_text   = "-10";
    std::string axis_y_hi_text   = "10";
    std::string n_pts_text       = "200";
    // false — grid перебирает НУ мастера (legacy default).
    // true  — grid перебирает НУ слейва, мастер фикс.
    bool        grid_swap_master_slave = false;

    // ---- Runtime knobs (substituted в NVRTC #define) ----
    int         type_of_synch    = 0;     // 0=unidir, 1=bidir
    int         error_estim      = 2;     // 0|1|2
    std::string fs_error_trs_text = "1e-12";

    // ---- CSV export ----
    // mode 0 (On Attractor): один файл, строки = decimated trajectory points,
    //   столбцы = vars[0..N-1] + sync_error.
    // mode 1 (On Grid): один файл, две строки заголовка (X/Y ranges) + матрица
    //   ошибок n_pts × n_pts row-major.
    bool        csv_save_enabled = false;
    std::string csv_output_path;

    // ---- Визуализация ----
    int         colormap_idx     = 2;     // Turbo
    bool        autoscale_color  = true;  // true → cmin/cmax = result.min_val/max_val
    std::string c_min_text       = "-12";
    std::string c_max_text       = "0";
    float       line_width       = 1.0f;
    // α=0.5 по дефолту — overlapping segments дают illusion глубины
    // (при α=1 далёкие/близкие проекции сливаются в одну плотную линию).
    float       alpha            = 0.5f;
    bool        swap_axes        = false; // grid mode: transpose heatmap
    // Painter's algorithm для attractor-mode: сегменты сортируются по
    // средней координате НЕпоказываемой оси (z = первая var, не равная
    // axis_x_var / axis_y_var). invert_depth=false → дальние (малый z)
    // рисуются первыми, ближние сверху (взгляд "сверху", z↑). При true —
    // взгляд "снизу" (z↓).
    bool        invert_depth     = false;

    // ---- Состояние ----
    FastSyncResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;
};

struct FastSyncAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::string loaded_system_name;

    std::vector<FastSyncConfig> configs;
    int active_config_index = 0;
    int running_config_index = -1;

    std::future<FastSyncResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    FastSyncAnalysisSession() = default;
    FastSyncAnalysisSession(FastSyncAnalysisSession&&) = default;
    FastSyncAnalysisSession& operator=(FastSyncAnalysisSession&&) = default;
    FastSyncAnalysisSession(const FastSyncAnalysisSession&) = delete;
    FastSyncAnalysisSession& operator=(const FastSyncAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    void add_config();
    void remove_config(int i);

    bool run(ParametricEngine& engine, int config_idx);
    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
};

// Одна спектр-кривая на параметрическом LS-графике — конфиг идентичен
// LLECurveConfig (тот же UX), но результат — матрица: N экспонент на
// каждую точку параметра, где N == |vars|.
struct LSCurveConfig {
    std::string label   = "LS 1";
    bool        visible = true;

    std::string scheme         = "Euler";
    std::string symmetry_s     = "0.5";    // a[0] для CD

    int         param_index    = 0;
    // IC sweep: если sweep_over_var=true, кривая считается по нач. условию
    // переменной vars[var_sweep_index] вместо параметра. Engine выберет
    // соответствующий par_or_var compile-time через template substitution.
    bool        sweep_over_var = false;
    int         var_sweep_index = 0;
    std::string param_lo_text  = "0";
    std::string param_hi_text  = "1";
    std::string n_pts_text     = "500";

    std::string h_text             = "0.01";
    std::string t_max_text         = "100";
    std::string transient_text     = "100";
    std::string max_value_text     = "1e6";

    std::string eps_text       = "1e-4";
    std::string nt_text        = "1";

    bool        csv_save_enabled = false;
    std::string csv_output_path;

    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    LS1DResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;

    // ---- 2D-режим (LS по двум параметрам/IC, хитмап одной экспоненты) ----
    // mode_2d=true → Run собирает LS2DRequest вместо LS1DRequest, результат
    // (полный спектр N экспонент на каждую ячейку) ложится в result_2d.
    // Combo над хитмапой переключает display_exponent_idx без повторного Run.
    bool        mode_2d = false;
    int         param_index_2 = 0;
    bool        sweep_over_var_2 = false;
    int         var_sweep_index_2 = 0;
    std::string param_lo_2_text = "0";
    std::string param_hi_2_text = "1";

    LS2DResult result_2d;
    bool        last_run_2d_ok = false;
    int         data_generation_2d = 0;
    bool        fit_request_2d = false;

    // Какую экспоненту показывать в heatmap (0-based, default λ₁).
    int         display_exponent_idx = 0;
};

// LS-сессия — структура та же что у LLE-сессии, но другая Request/Result и
// per-«прогон» — это «спектр» (curves).
struct LyapunovSpectrumAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::string loaded_system_name;

    std::vector<LSCurveConfig> curves;
    int active_curve_index  = 0;
    int running_curve_index = -1;

    std::future<LS1DResult> run_future;
    // Параллельный future для 2D-режима — гарантия: одновременно в полёте только
    // одна задача (1D или 2D); какая именно — is_2d_run.
    std::future<LS2DResult> run_future_2d;
    bool is_2d_run = false;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    LyapunovSpectrumAnalysisSession() = default;
    LyapunovSpectrumAnalysisSession(LyapunovSpectrumAnalysisSession&&) = default;
    LyapunovSpectrumAnalysisSession& operator=(LyapunovSpectrumAnalysisSession&&) = default;
    LyapunovSpectrumAnalysisSession(const LyapunovSpectrumAnalysisSession&) = delete;
    LyapunovSpectrumAnalysisSession& operator=(const LyapunovSpectrumAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void add_curve();
    void remove_curve(int i);
    bool run(ParametricEngine& engine, int curve_idx);
    bool run_async(ParametricEngine& engine, int curve_idx);
    void request_cancel();
    bool poll();
};