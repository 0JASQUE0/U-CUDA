#pragma once
#include "ucuda/domain/network_session.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/cuda/parametric_engine.h"

// Сессия = сохраняемое состояние (NetworkState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только NetworkState.
struct NetworkSession : NetworkState {
    std::future<NetworkResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::chrono::steady_clock::time_point last_run_completed_at;
    NetworkSession() = default;
    NetworkSession(NetworkSession&&) = default;
    NetworkSession& operator=(NetworkSession&&) = default;
    NetworkSession(const NetworkSession&) = delete;
    NetworkSession& operator=(const NetworkSession&) = delete;
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void add_config();
    void remove_config(int i);
    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
};
