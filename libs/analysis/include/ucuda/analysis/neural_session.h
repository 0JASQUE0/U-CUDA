#pragma once
// Neural: сохраняемое состояние (NeuralState) — в domain/neural_session.h, здесь — выполнение
// (future, отмена, прогресс, запуск, опрос) и функции, которые строят граф и запрос (через кодоген).
#include "ucuda/domain/neural_session.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/cuda/parametric_engine.h"
#include <atomic>
#include <chrono>
#include <future>
#include <memory>

// Сессия = сохраняемое состояние (NeuralState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только NeuralState.
struct NeuralSession : NeuralState {
    std::future<NeuralOutcome> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
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
    // resume: продолжить с контрольной точки конфигурации до её T max.
    bool run_async(ParametricEngine& engine, int config_idx, bool resume = false);
    void request_cancel();
    bool poll();
};

const char* neural_rule_name(NeuralRule r);
const char* neural_schedule_name(NeuralSchedule s);
const char* neural_sweep_name(NeuralSweepKind k);
const char* neural_source_name(NeuralSourceKind k);
// Снимок системы библиотеки в источник. Бросает на отображениях и ДАУ.
void neural_source_set_system(NeuralSource& src, const std::string& name, const SystemRecord& r, const System& sys);
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
// Значения свипа, пусто = свип выключен. Бросает на неверном диапазоне.
std::vector<double> neural_sweep_values(const NeuralConfig& c);
// То же для второй оси: пусто, если она выключена или выключен свип.
std::vector<double> neural_sweep2_values(const NeuralConfig& c);
// Сводка свипа строками: значения осей, популяция, R, разброс, показатель, шаг остановки. Пустая при отсутствии свипа.
std::string neural_sweep_csv(const NeuralConfig& c);
// Длительность записи с учётом протокола: при попытках это попытки * (пауза + стимул), а не t_max.
double neural_record_time(const NeuralConfig& c);
// Переменная снимка системы на тех же отсчётах и той же траектории, что и вход; без нормировки. Цель задачи.
std::vector<double> neural_source_samples(const NeuralSource& s, int variable, size_t n, double dt);
// Настройки запуска под задачу: что писать, с каким шагом, по какому отрезку нормировать вход. Бросает на несовместимых настройках.
NeuralConfig neural_task_config(const NeuralConfig& c);
// Классификация: класс каждой попытки (номер в task.classes). Бросает, если попыток мало для разбиения.
std::vector<int> neural_class_labels(const NeuralConfig& c);
// Обучение выхода по готовому расчёту. c из neural_task_config, r его запрос.
NeuralTaskResult neural_task_evaluate(const NeuralConfig& c, const NeuralRequest& r, const NeuralResult& res);
// Прогноз: подключённые входы по номеру источника, они же цели.
std::vector<int> neural_forecast_channels(const NeuralConfig& c);
// Обучение по первому прогону (r1, res1) и замыкание второго: r2 = тот же план до конца прогноза, с res1.checkpoint.
// Первая строка прогноза считается здесь и пишется в r2, остальные считает выход на GPU (r2.feedback); авторегрессия тут же.
void neural_forecast_prepare(const NeuralConfig& c, const NeuralRequest& r1, const NeuralResult& res1, NeuralRequest& r2, NeuralTaskResult& out);
// Прогноз сети из второго прогона, ошибка и время верного прогноза.
void neural_forecast_finish(const NeuralResult& res2, NeuralTaskResult& out);
// Спайки строками: шаг, время, узел. Время спайка шага n = (n + 1) h.
std::string neural_spike_csv(const NeuralConfig& c);
// Контрольная точка в файле. Бросает на ошибке чтения и записи.
void neural_checkpoint_save(const std::string& path, const NeuralCheckpoint& c);
std::shared_ptr<NeuralCheckpoint> neural_checkpoint_load(const std::string& path);
// Ранняя часть записи из прошлого расчёта: продолжение пишет только свою часть.
void neural_merge_resumed(const NeuralResult& before, NeuralResult& after);
// Конфигурация одного значения: в эксперименте меняется только то, что свипается.
NeuralConfig neural_sweep_instance(const NeuralConfig& c, int index, double value);
NeuralPlan neural_plan(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                       const std::vector<std::string>& vars, const std::vector<std::string>& params,
                       const NeuralConfig& c);
