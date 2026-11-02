#pragma once
#include "ucuda/domain/network_session.h"
#include "ucuda/analysis/analysis_session.h"

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

// Вспомогательные функции сессий (реализация в analysis; типы — в ucuda/domain).

// Перестраивает nodes/edges под текущие настройки генератора. Существующие
// переопределения параметров сохраняются для узлов, которые остались (по
// индексу): смена топологии не должна стирать расстройку ансамбля.
// Возвращает текст ошибки (пусто = успех).
std::string net_generate_topology(NetworkConfig& c, const std::vector<std::string>& vars);

// Раскладывает рёбра в дуги и считает CSR по узлу-приёмнику. Веса уже
// домножены на общий множитель связи. Используется и движком, и редактором
// (чтобы показать степень узла).
void net_build_csr(const NetworkConfig& c, double coupling,
                   std::vector<int>& edge_start, std::vector<int>& edge_src,
                   std::vector<double>& edge_w, std::vector<int>& edge_law);

// Готовое тело switch(law) для ядра. Бросает std::runtime_error с текстом от
// кодогена, если выражение закона не разобралось.
std::string net_coupling_body(const System& sys, const std::vector<NetCouplingLaw>& laws);

// Человекочитаемое имя топологии (для комбо и подписей).
const char* net_topology_name(NetTopology t);

// Independent ODE snapshot. Coupling is part of the RHS and therefore every
// stage of a generated scheme. Throws on unsupported input or engine limits.
NetworkMonolithic net_monolithic_snapshot(const NetworkState& s, const NetworkConfig& c);
