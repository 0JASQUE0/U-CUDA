#pragma once
#include "ucuda/domain/order_session.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/equations/adaptive_params.h"

// Сессия = сохраняемое состояние (OrderAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только OrderAnalysisState.
struct OrderAnalysisSession : OrderAnalysisState {
    std::future<OrderResult> run_future;
    // Каждый расчёт вкладки возвращает свой тип, поэтому у каждого своё
    // future; running_kind (kOrderCalc*) говорит poll'у, какое забирать.
    std::future<PerfResult>       perf_future;
    std::future<StabilityResult>  stab_future;
    std::future<ErrTimeResult>    errt_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
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

// Вспомогательные функции сессий (реализация в analysis; типы — в ucuda/domain).

// Сколько грубых шагов сделает ячейка при данных h и t_max — та же формула,
// что в ядре. Нужна UI, чтобы показать N и предупредить о нецелом t_max/h.
long long order_steps_for_h(double h, double t_max, bool snap);

// Допуск Ньютона, с которым вкладка собирает неявные схемы при выбранной
// точности; 0 — системная настройка не трогается (режим double). Неявная
// схема останавливает Ньютон по ||dX|| < newton_tol, и системные 1e-10 стали
// бы потолком раньше арифметики. Функция публичная, потому что UI показывает
// подставленное значение: молча менять чужую настройку нельзя.
double order_newton_tol_for_precision(int prec);

// Error(t): ошибка схемы во времени против референса (см. ErrTimeRequest).
// Однопоточный CPU-расчёт тем же телом КРС, что у остальных расчётов вкладки.
ErrTimeResult run_order_error_time(const ErrTimeRequest& req);

// Паспортный порядок схемы для Ричардсона: встроенная — из таблицы кодогена,
// кастомная — со слов автора; обёртки (Extr/Comp) и неизвестные — 0.
int order_scheme_passport_order(const OrderAnalysisSession& s, const std::string& scheme);

// Порядок, с которым Error(t) считает Ричардсона: errt_order_text, если задан,
// иначе паспортный; 0 — оценка не считается.
int order_errt_effective_order(const OrderAnalysisSession& s, const OrderConfig& c);
