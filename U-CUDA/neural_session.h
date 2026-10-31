#pragma once
#include "analysis_session.h"
#include "network_session.h"
#include "neural_engine.h"
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Neural — большая сеть: популяции и проекции вместо узлов и рёбер. Сессия хранит рецепт (правила, seed),
// граф строится заново при запуске; хеш ловит тихую смену генератора.
// Network остаётся отдельной вкладкой со своим ядром; отсюда берутся законы связи и codegen.

// Версия генератора входит в рецепт: новая версия меняет графы, и старые сессии должны это увидеть.
constexpr int kNeuralGraphVersion = 1;

enum class NeuralRule { Probability = 0, FixedDegree, Chain, Ring, Grid, King, SmallWorld };
// Деление веса дуги: без него, на степень этой проекции, на всю входящую степень узла (как в Network).
enum class NeuralNorm { None = 0, Projection, Total };

struct NeuralPopulation {
    std::string name = "pop";
    std::string size_text = "1000";
    // Значение и половина ширины равномерного разброса вокруг него; пустое = 0.
    std::map<std::string, std::string> param_values, param_spread;
    std::map<std::string, std::string> initial_conditions, ic_spread;
};

struct NeuralProjection {
    int from = 0, to = 0;
    NeuralRule rule = NeuralRule::FixedDegree;
    std::string p_text = "0.01";       // Probability
    std::string degree_text = "100";   // FixedDegree: входящая степень
    std::string k_text = "1";          // Ring, SmallWorld: соседей в каждую сторону
    std::string width_text = "10";     // Grid, King: высота = размер / ширина
    std::string rewire_text = "0.1";   // SmallWorld
    bool periodic = false;             // Grid, King
    bool directed = false;             // регулярные правила: дуги вместо рёбер
    std::string weight_text = "1";
    std::string weight_spread_text = "0";  // половина ширины относительного разброса
    NeuralNorm norm = NeuralNorm::Total;
    bool norm_expected = false;        // делить на ожидаемую степень, а не на фактическую
    int law = 0;
};

struct NeuralConfig {
    std::string label = "Experiment";
    NeuralSchedule schedule = NeuralSchedule::Stage;
    std::string scheme = "RK4";        // шаг узла post/heun; stage всегда RK4
    std::string symmetry_s = "0.5";
    std::string cd_fp_iters = "4";
    NewtonSettings newton;

    std::string h_text = "0.01";
    std::string t_max_text = "100";
    std::string transient_text = "0";
    std::string pre_scaller_text = "10";
    std::string max_value_text = "1e6";

    std::string coupling_text = "0.1";
    std::string seed_text = "1";
    std::string observed_per_pop_text = "8";

    std::vector<NetCouplingLaw> laws;
    std::vector<NeuralPopulation> populations;
    std::vector<NeuralProjection> projections;

    // Рецепт, по которому получен graph_hash, и система на момент расчёта.
    std::string graph_recipe, graph_hash, model_text;

    NeuralResult result;
    long long arcs = 0;
    int degree_min = 0, degree_max = 0;
    double build_seconds = 0.0;
    bool last_run_ok = false;
    std::string last_error;
    bool hash_mismatch = false;
    int data_generation = 0;
};

// CSR по приёмнику в канонической нумерации: популяции подряд, в популяции по номеру.
struct NeuralGraph {
    int N = 0;
    std::vector<int> pop_start;
    std::vector<int> start, src, law;
    std::vector<double> w;
    std::string hash;
    int degree_min = 0, degree_max = 0;
};

struct NeuralOutcome {
    NeuralResult result;
    std::string hash, recipe;
    long long arcs = 0;
    int degree_min = 0, degree_max = 0;
    double build_seconds = 0.0;
    bool hash_mismatch = false;
};

struct NeuralSession {
    std::vector<std::string> vars, params;
    System sys;
    std::vector<CustomScheme> custom_schemes;
    std::vector<std::string>  wrapper_schemes;
    std::vector<std::string>  enabled_builtin_schemes;
    std::string loaded_system_name;

    std::vector<NeuralConfig> configs;
    int active_config_index = 0;
    int running_config_index = -1;

    std::future<NeuralOutcome> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;

    std::string last_run_label;
    double last_run_seconds = 0.0;
    bool   last_run_succeeded = false;
    std::chrono::steady_clock::time_point last_run_completed_at;

    NeuralSession() = default;
    NeuralSession(NeuralSession&&) = default;
    NeuralSession& operator=(NeuralSession&&) = default;
    NeuralSession(const NeuralSession&) = delete;
    NeuralSession& operator=(const NeuralSession&) = delete;

    void load_from_record(const SystemRecord& r, const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void add_config();
    void remove_config(int i);
    bool run_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
};

const char* neural_rule_name(NeuralRule r);
const char* neural_schedule_name(NeuralSchedule s);
NeuralPopulation neural_default_population(const SystemRecord* r, const std::vector<std::string>& vars,
                                           const std::vector<std::string>& params);

// Бросает std::runtime_error на неверных настройках. Случайность только из своего кода, по потоку на
// (seed, проекция, узел): граф не зависит от порядка обхода и от стандартной библиотеки.
NeuralGraph neural_build_graph(const NeuralConfig& c);
std::string neural_graph_recipe(const NeuralConfig& c);
std::string neural_model_text(const System& sys);
// Запрос к движку поверх готового графа. Бросает на неверных настройках и на отказе кодгена.
NeuralRequest neural_build_request(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                                   const std::vector<std::string>& vars, const std::vector<std::string>& params,
                                   const NeuralConfig& c, const NeuralGraph& g);
