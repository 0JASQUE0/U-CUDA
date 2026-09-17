#pragma once
#include "analysis_session.h"
#include "system_record.h"
#include "parametric_engine.h"
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Order — вкладка оценки порядка точности схемы.
//
// Оценщик (см. kernels/order.template.cu): три решения одной задачи шагами
// h, h/2, h/4, сравниваемые на грубой сетке, дают
//   E1 = max|y_h - y_h/2|,  E2 = max|y_h/2 - y_h/4|,  p = log2(E1/E2),
// что при y_h = y* + C h^p равно ровно p. Опорное решение не участвует.
//
// Три диаграммы вкладки — это один и тот же расчёт с разным выбором осей:
//   p(h)            — ось X = шаг;
//   p(параметр)     — ось X = элемент a[], h фиксирован;
//   p(пар1, пар2)   — обе оси; любая из них может быть и шагом.
//
// Параметры МЕТОДА (для композиционных схем их больше одного) отдельного
// механизма не имеют: они объявляются обычными параметрами системы, в правых
// частях не используются и читаются телом кастомной КРС как a[k]. Поэтому
// селектор оси ниже — единый список «h + все элементы a[]», без деления на
// параметры системы и параметры метода.

// Что перебирается по оси. Значение поля axis_*_target:
//   kOrderTargetH (-1) — шаг интегрирования;
//   0                  — a[0], коэффициент симметрии s;
//   1..M               — a[i], i-й параметр системы.
constexpr int kOrderTargetH = -1;

// Что запускает Run на этой вкладке. Живёт в конфиге, а не в окне графика:
// "Run all" обязан знать, что именно считать для каждой вкладки, а окна
// графиков к моменту запуска могут быть вообще закрыты.
//   kOrderCalcOrder — порядок/ошибка по сетке (order.template.cu);
//   kOrderCalcPerf  — время счёта vs достигнутая ошибка (perfIntegrateKernel).
constexpr int kOrderCalcOrder = 0;
constexpr int kOrderCalcPerf  = 1;

struct OrderConfig {
    std::string label  = "Order";
    std::string scheme = "Euler";
    std::string symmetry_s = "0.5";        // a[0], когда s не свипуется

    // ---- Оси ----
    bool        two_d = false;             // false = 1D-кривая, true = 2D-карта

    int         axis_x_target   = kOrderTargetH;
    std::string axis_x_lo_text  = "1e-4";
    std::string axis_x_hi_text  = "1e-1";
    bool        axis_x_log      = true;    // лог-сетка УЗЛОВ по оси
    // Число узлов. В 2D — ОБЩЕЕ на обе оси, отдельного axis_y_n_text нет:
    // карта всегда N x N. Разные N по осям давали растянутые ячейки, из-за
    // чего тонкая структура (например луч порядка у композиций) читалась
    // по-разному вдоль X и вдоль Y.
    std::string axis_x_n_text   = "200";

    int         axis_y_target   = 0;
    std::string axis_y_lo_text  = "0";
    std::string axis_y_hi_text  = "1";
    bool        axis_y_log      = false;

    // Устройство расчёта. GPU — прежний путь через NVRTC и order.template.cu.
    // CPU — та же арифметика тем же телом КРС, скомпилированным cl.exe
    // (krs_cpu.h), но прогоны ПОСЛЕДОВАТЕЛЬНЫЕ: меряется время одной
    // траектории, а не пропускная способность, поэтому replicas там не значат
    // ничего и в UI гасятся. Формулы, пороги и коды статусов у обеих веток
    // общие — см. run_order_cpu в order_session.cpp.
    bool        use_gpu        = true;

    // ---- Интегрирование ----
    std::string h_text         = "0.01";   // базовый шаг, когда h не свипуется
    std::string t_max_text     = "10";
    std::string max_value_text = "1e6";

    // Подгонять h к t_max/N с целым N. По умолчанию ВКЛЮЧЕНО: без этого число
    // шагов = floor(t_max/h), фактическое конечное время гуляет от узла к узлу,
    // и на кривую садится паразитный вклад O(h) — для схем порядка выше первого
    // он забивает измеряемую величину целиком. Выключатель оставлен, чтобы
    // можно было посмотреть на поведение «как на остальных вкладках».
    bool        snap_steps    = true;
    // Сравнивать только в t = t_max вместо максимума по всей траектории.
    bool        endpoint_only = false;

    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    // ---- Результат ----
    // Настройки ОТОБРАЖЕНИЯ (масштаб осей, вторая кривая, колормапа) живут не
    // здесь, а на окне графика (OrderPlotWindow): в одном окне лежат кривые
    // нескольких вкладок, и масштаб оси у них обязан быть один.
    OrderResult result;
    bool        last_run_ok = false;
    std::string last_error;
    int         data_generation = 0;
    bool        fit_request = false;

    // ---- Performance ----
    // Замер времени счёта по той же оси X. Ось Y (2D-карта) здесь не
    // участвует: диаграмма «время vs ошибка» одномерна по построению.
    int         calc_kind = kOrderCalcOrder;

    std::string perf_repeats_text  = "20";   // засекаемых запусков на узел
    std::string perf_warmup_text   = "2";    // прогревочных запусков вне зачёта
    // Одинаковых потоков в замеряемом запуске. 1 = латентность одного расчёта;
    // большое число = пропускная способность загруженного GPU. Умолчание 1,
    // потому что вопрос «сколько стоит ОДИН расчёт» задают чаще.
    std::string perf_replicas_text = "1";

    // Эталонный метод для третьей величины по оси X: E_ref = max|y_h - y_ref| —
    // не ричардсоновская оценка, а сама ошибка схемы на шаге h. По умолчанию
    // DOPRI78: восьмой порядок, и на рабочих шагах его собственная ошибка
    // уходит под машинную точность решения, то есть он и есть «точный ответ».
    std::string perf_ref_scheme = "DOPRI78";
    // Шагов эталона на один шаг испытуемой схемы. Единицы хватает, пока методы
    // разные; когда эталон совпадает с испытуемым, при равном шаге это была бы
    // та же арифметика и разность тождественно нулевая — отсюда умолчание 4.
    std::string perf_ref_substeps_text = "4";

    PerfResult  perf_result;
    bool        perf_last_run_ok = false;
    int         perf_data_generation = 0;
    bool        perf_fit_request = false;
};

struct OrderAnalysisSession {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    // Wrapper names for the combo; the resolver builds the body from the name.
    std::vector<std::string>  wrapper_schemes;
    std::vector<std::string> enabled_builtin_schemes;
    std::string loaded_system_name;

    std::vector<OrderConfig> configs;

    int active_config_index  = 0;
    int running_config_index = -1;

    std::future<OrderResult> run_future;
    // Замер времени возвращает другой тип, поэтому у него своё future;
    // running_is_perf говорит poll'у, какое из них забирать.
    std::future<PerfResult>  perf_future;
    bool running_is_perf = false;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;

    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;

    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool   last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    OrderAnalysisSession() = default;
    OrderAnalysisSession(OrderAnalysisSession&&) = default;
    OrderAnalysisSession& operator=(OrderAnalysisSession&&) = default;
    OrderAnalysisSession(const OrderAnalysisSession&) = delete;
    OrderAnalysisSession& operator=(const OrderAnalysisSession&) = delete;

    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);

    void add_config();
    void remove_config(int i);

    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();

    // Подписи для селектора оси: "h" + "s (a[0])" + параметры системы.
    // Индекс в возвращаемом векторе = target + 1 (target -1 идёт первым).
    std::vector<std::string> axis_target_names() const;
    // target -> человекочитаемое имя (для подписей осей графика).
    std::string axis_target_label(int target) const;
};

// Сколько грубых шагов сделает ячейка при данных h и t_max — та же формула,
// что в ядре. Нужна UI, чтобы показать N и предупредить о нецелом t_max/h.
long long order_steps_for_h(double h, double t_max, bool snap);
