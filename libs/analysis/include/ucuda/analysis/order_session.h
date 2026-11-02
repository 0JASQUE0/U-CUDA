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
