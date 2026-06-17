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

    // --- async-расчёт (как у ParametricAnalysisSession) ---
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

// Сессия параметрического анализа (1D-бифуркация).
// Изолирована от PhaseAnalysisSession — своя копия системы, своя интегрирующая схема.
struct ParametricAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::string krs_code;

    std::map<std::string, std::string> param_values;
    std::map<std::string, std::string> initial_conditions;

    // выбор изменяемого параметра и диапазон
    int param_index = 0;              // 0-индекс в params (внутри run сдвигается на +1)
    std::string param_lo_text = "0";
    std::string param_hi_text = "1";
    std::string n_pts_text    = "500";

    int writable_var = 0;

    std::string scheme         = "Euler";
    std::string h_text         = "0.01";
    std::string t_max_text     = "100";
    std::string transient_text = "100";
    std::string pre_scaller_text = "1";
    std::string max_value_text = "1e6";

    // CSV-вывод. Путь хранится отдельно от флага, чтобы можно было выключить
    // запись, не удаляя сам путь из поля (для повторного включения).
    bool csv_save_enabled = false;
    std::string csv_output_path;

    // Если true — на графике рисуются межпиковые интервалы (peak_times),
    // иначе значения пиков (bifurcation_points). Колонка 3 vs 2 в CSV.
    bool plot_inter_peaks = false;

    Bifurcation1DResult result;
    bool last_run_ok = false;
    std::string last_error;
    int data_generation = 0;
    bool fit_request = false;

    // Имя системы, под которую сессия инициализирована (см. PhaseAnalysisSession).
    std::string loaded_system_name;

    // --- async-расчёт ---
    // future, в которую std::async кладёт результат с worker-потока.
    // Все мутации session происходят ТОЛЬКО на главном потоке (в run_async/poll),
    // worker трогает только engine.
    std::future<Bifurcation1DResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    // session не копируется (содержит future) — только move.
    ParametricAnalysisSession() = default;
    ParametricAnalysisSession(ParametricAnalysisSession&&) = default;
    ParametricAnalysisSession& operator=(ParametricAnalysisSession&&) = default;
    ParametricAnalysisSession(const ParametricAnalysisSession&) = delete;
    ParametricAnalysisSession& operator=(const ParametricAnalysisSession&) = delete;

    void regenerate_krs();
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    // Старый синхронный Run. Заблокирует поток до конца расчёта.
    bool run(ParametricEngine& engine);

    // Async: запускает расчёт в std::async-потоке. Возвращает false если
    // уже что-то считается. Результат применяется в poll() позже.
    bool run_async(ParametricEngine& engine);

    // Вызывается каждый кадр из GUI-потока. Если future готова — забирает
    // результат, применяет его к session (result, flags, data_generation,
    // fit_request), сбрасывает in_flight. Возвращает true, если только что
    // завершилось (GUI может среагировать, например авто-сохранить).
    bool poll();
};