#pragma once
#include "system_record.h"
#include "codegen.hpp"
#include "parametric_engine.h"
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

    // Свип: какой параметр, его диапазон, число точек.
    int         param_index    = 0;
    // Если sweep_over_var=true — БД строится по начальному условию переменной
    // vars[var_sweep_index] вместо параметра. Дефолт false → param-свип (старое
    // поведение). NonLinAnal-kernel принимает par_or_var runtime-аргументом,
    // переключаем только индекс + флаг в engine — никакой пересборки PTX.
    bool        sweep_over_var = false;
    int         var_sweep_index = 0;
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
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

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
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

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
    bool poll();
};

// Одна спектр-кривая на параметрическом LS-графике — конфиг идентичен
// LLECurveConfig (тот же UX), но результат — матрица: N экспонент на
// каждую точку параметра, где N == |vars|.
struct LSCurveConfig {
    std::string label   = "LS 1";
    bool        visible = true;

    std::string scheme         = "Euler";

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
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

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
    bool poll();
};