#pragma once
// Сессии анализа: сохраняемое состояние (*State) — в domain/analysis_session.h, здесь —
// выполнение (future, отмена, прогресс, запуск, опрос) и резолвер имени схемы.
#include "ucuda/domain/analysis_session.h"
#include "ucuda/equations/adaptive_params.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>

class ParametricEngine;   // ucuda/cuda/parametric_engine.h — только в .cpp analysis

// Тело calculateDiscreteModel для выбранной схемы: сперва ищется среди
// пользовательских КРС по имени, затем имя пробуется как экстраполяционная
// обёртка "Extr(<база>|n...)", иначе генерируется кодогеном. Пустая строка =
// нет валидной системы или схема не разобралась. Живёт здесь, а не в
// analysis_session.cpp приватно, потому что тем же путём КРС резолвит
// order_session.cpp.
// ЕДИНСТВЕННЫЙ резолвер имени схемы: сюда же ходит PhaseAnalysisSession::
// regenerate_krs, которая наполняет отладочную панель KRS.
// err (если не nullptr) получает причину пустой КРС: текст отказа кодгена
// (cd_check_complex_safe, jac_check_differentiable, ...) или резолвера
// обёрток; пусто — причина не известна (нет системы). Сама КРС от err не зависит.
std::string compute_krs_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                   const System& sys,
                                   const std::string& scheme,
                                   std::string* err = nullptr);

// Текст ошибки для запуска с пустой КРС: причина из compute_krs_for_scheme,
// иначе общая фраза. Только для пути отказа — КРС генерируется заново.
std::string krs_empty_error(const std::vector<CustomScheme>& custom_schemes,
                            const System& sys, const std::string& scheme);

// То же имя, но CPU-форма шага (codegen_scheme_cpu_equivalent под тем же
// разбором имени). Только для отладочной панели: этот текст нигде не
// компилируется, CPU-интегратор считает по байткоду AST.
std::string compute_krs_cpu_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                       const System& sys,
                                       const std::string& scheme);

// Тела адаптивного шага по имени схемы — тот же разбор имени, что у compute_krs_for_scheme:
// RK-пары (codegen_adaptive), GBS 2-4 ... (codegen_adaptive_gbs), Extr(база|n1..nK), K >= 2
// (codegen_adaptive_extrapolation; база — встроенная или кастомная КРС). Схема без оценки
// ошибки (и кастомная КРС с тем же именем) — std::runtime_error с текстом для UI.
AdaptiveCode adaptive_code_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                      const System& sys, const std::string& scheme);

// Сессия = сохраняемое состояние (PhaseAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только PhaseAnalysisState.
struct PhaseAnalysisSession : PhaseAnalysisState {
    std::chrono::steady_clock::time_point continuation_last_frame;
    // Drop the oldest pairs above continuation_peaks_cap in every buffer. Called
    // after each chunk and when the cap is lowered, so a paused loop shrinks too.
    void trim_continuation_peaks();
    // Append result.features' peaks with time offset `base` (continuation chunk).
    void append_result_peaks(double base);
    // Continuation off: the diagram shows the peaks of the last single run,
    // rebuilt from result (not accumulated). Called after every result.
    void rebuild_peaks_from_result();
    // Перегенерировать krs_code из sys по текущему scheme. Зовётся при загрузке
    // системы и при смене метода. Дёшево (codegen быстрый).
    void regenerate_krs();
    // Фоновая сборка GPU-ядра под текущие систему, схему и шаг: адаптивное (phase_kernel_ad),
    // если адаптивный шаг включён, иначе ядро постоянного шага. Зовётся из regenerate_krs и при
    // правке блока адаптивного шага; занятый прогрев не дублируется.
    void prewarm_gpu();
    // async-расчёт (как у BifurcationAnalysisSession)
    std::future<AnalysisResult> recompute_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Отмена идущего расчёта: токен создаёт recompute_async, request_cancel взводит. Слушает
    // его ветка адаптивного шага (её траектория может идти сколько угодно шагов); постоянный
    // шаг с известным числом шагов доходит до конца. Отменённый результат не заменяет
    // прежний: poll() оставляет графики и пишет причину в result.error.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    void request_cancel();
    // Фоновый прогрев NVRTC-кэша сразу при смене system/scheme (см.
    // regenerate_krs) — к моменту реального Run модуль уже готов. Хранится
    // тут же: деструктор future (async-политика) сам дождётся завершения,
    // если сессия закрывается посреди прогрева.
    std::future<void> prewarm_future;
    // session не копируется (содержит future) — только move.
    PhaseAnalysisSession() = default;
    PhaseAnalysisSession(PhaseAnalysisSession&&) = default;
    PhaseAnalysisSession& operator=(PhaseAnalysisSession&&) = default;
    PhaseAnalysisSession(const PhaseAnalysisSession&) = delete;
    PhaseAnalysisSession& operator=(const PhaseAnalysisSession&) = delete;
    // операции
    void add_ic();              // добавить НУ (пустое)
    void remove_ic(int i);
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
};

// Сессия = сохраняемое состояние (BifurcationAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только BifurcationAnalysisState.
struct BifurcationAnalysisSession : BifurcationAnalysisState {
    // async-расчёт
    // Все мутации session происходят на главном потоке (run_async/poll),
    // worker трогает только engine + собственную копию Request.
    std::future<Bifurcation1DResult> run_future;
    std::future<Bifurcation2DResult> run_future_2d;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Cooperative cancellation. Token is created in run_async() and shared
    // with the worker via Request::cancel; request_cancel() sets it to true,
    // engine bails out at the next cuDeviceSynchronize and returns
    // cancelled=true. Token is reset after poll() consumes the result.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    // Progress (0..1) written by engine on each chunk, read by GUI each frame.
    std::shared_ptr<std::atomic<float>> progress_token;
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
    void remove_diagram(int i);
    // Старый синхронный Run для конкретной БД (по индексу).
    bool run(ParametricEngine& engine, int diagram_idx);
    // Async-Run для конкретной БД. Возвращает false если очередь занята.
    bool run_async(ParametricEngine& engine, int diagram_idx);
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int diagram_idx) const;
    // Задача фонового прогрева: копирует запрос НА ВЫЗЫВАЮЩЕМ потоке (сессию читать из фонового
    // нельзя — её правит UI) и компилирует модуль, ничего не считая. Пустая функция, если
    // индекс плохой или КРС не собрался.
    std::function<void(ParametricEngine&)> prewarm_task(int diagram_idx) const;
    // Cooperative cancel of the in-flight run. No-op if nothing is in flight.
    void request_cancel();
    // Вызывается каждый кадр из GUI-потока. Если future готова — применяет
    // результат к diagrams[running_diagram_index], сбрасывает in_flight.
    // Возвращает true в кадр завершения.
    bool poll();
};

// Сессия = сохраняемое состояние (LLEAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только LLEAnalysisState.
struct LLEAnalysisSession : LLEAnalysisState {
    std::future<LLE1DResult> run_future;
    // Параллельный future для 2D-режима. in_flight (общий) гарантирует, что
    // в полёте одновременно только одна задача — 1D или 2D, не обе. Какая
    // именно — определяет is_2d_run (выставляется в run_async, читается poll).
    std::future<LLE2DResult> run_future_2d;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::chrono::steady_clock::time_point last_run_completed_at;
    LLEAnalysisSession() = default;
    LLEAnalysisSession(LLEAnalysisSession&&) = default;
    LLEAnalysisSession& operator=(LLEAnalysisSession&&) = default;
    LLEAnalysisSession(const LLEAnalysisSession&) = delete;
    LLEAnalysisSession& operator=(const LLEAnalysisSession&) = delete;
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void remove_curve(int i);
    bool run(ParametricEngine& engine, int curve_idx);
    bool run_async(ParametricEngine& engine, int curve_idx);
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int curve_idx) const;
    // Задача фонового прогрева: копирует запрос НА ВЫЗЫВАЮЩЕМ потоке (сессию читать из фонового
    // нельзя — её правит UI) и компилирует модуль, ничего не считая. Пустая функция, если
    // индекс плохой или КРС не собрался.
    std::function<void(ParametricEngine&)> prewarm_task(int curve_idx) const;
    void request_cancel();
    bool poll();
};

// Сессия = сохраняемое состояние (Dft1DAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только Dft1DAnalysisState.
struct Dft1DAnalysisSession : Dft1DAnalysisState {
    std::future<Dft1DResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::chrono::steady_clock::time_point last_run_completed_at;
    Dft1DAnalysisSession() = default;
    Dft1DAnalysisSession(Dft1DAnalysisSession&&) = default;
    Dft1DAnalysisSession& operator=(Dft1DAnalysisSession&&) = default;
    Dft1DAnalysisSession(const Dft1DAnalysisSession&) = delete;
    Dft1DAnalysisSession& operator=(const Dft1DAnalysisSession&) = delete;
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    // Добавить config: глубокая копия последнего (если есть), иначе дефолт.
    // Label автоматически "DFT <N+1>", результат и флаги обнуляются.
    void add_config();
    void remove_config(int i);
    bool run(ParametricEngine& engine, int config_idx);
    bool run_async(ParametricEngine& engine, int config_idx);
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int config_idx) const;
    void request_cancel();
    bool poll();
};

// Фазовые портреты по бассейнам — один слот на BasinsConfig, позиционно синхронный с
// BasinsAnalysisSession::configs. Живёт ОТДЕЛЬНО от config'а, а не внутри него, потому что
// BasinsConfig обязан оставаться копируемым (add_config делает `c = configs.back()`), а
// PhaseAnalysisSession — move-only (держит std::future).
struct BasinsPhaseSlot {
    PhaseAnalysisSession phase;

    // Ячейки сетки (row-major iy*n + ix), по одной на бассейн; параллельно
    // phase.ic_sets. Номер и цвет бассейна выводятся из ячейки на лету, чтобы
    // toggle «Renumber (spiral)» менял подписи/цвета без пересчёта траекторий.
    std::vector<int> cells;

    // Подпись входов, породивших ТЕКУЩИЙ результат (полная — для индикации
    // «stale»), и подпись только тех входов, что реально влияют на форму
    // траекторий (см. build_basins_phase_run_signature): если она не менялась,
    // а набор ячеек тот же, пересчёт на GPU не нужен — хватит переподписать.
    std::string run_signature;
    std::string run_inputs_signature;
    // Подпись входов на последний кадр — для debounce автозапуска.
    std::string pending_signature;
    double      pending_since = 0.0;   // ImGui::GetTime() смены pending_signature
    bool        has_result = false;

    BasinsPhaseSlot() = default;
    BasinsPhaseSlot(BasinsPhaseSlot&&) = default;
    BasinsPhaseSlot& operator=(BasinsPhaseSlot&&) = default;
    BasinsPhaseSlot(const BasinsPhaseSlot&) = delete;
    BasinsPhaseSlot& operator=(const BasinsPhaseSlot&) = delete;
};

// Сессия = сохраняемое состояние (BasinsAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только BasinsAnalysisState; проекции портретов
// слотов передаются ей отдельным списком (phase_projections).
struct BasinsAnalysisSession : BasinsAnalysisState {
    // Фазовые портреты по бассейнам. Слот на каждый config (позиционно). Не сериализуются
    // целиком: в _last_basins.json уходят только настройки config'а и список проекций.
    std::vector<std::unique_ptr<BasinsPhaseSlot>> phase_slots;
    // Подогнать phase_slots под configs (ленивое создание). Безопасно звать каждый кадр.
    void ensure_phase_slots();
    // Проекции портретов по слотам (nullptr, где слота нет) — для сохранения сессии.
    std::vector<const std::vector<Projection>*> phase_projections() const;
    // Слоты заново под configs и в них сохранённые проекции (пустой список у слота = дефолт).
    void restore_phase_slots(std::vector<std::vector<Projection>>&& projs);
    std::future<BasinsResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    // Two-phase progress: 1 = sim, 2 = cluster. GUI uses this to render
    // "(1/2 sim)" or "(2/2 cluster)" suffix and reset the bar between phases.
    std::shared_ptr<std::atomic<int>>   progress_phase_token;
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
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int config_idx) const;
    // Reclustering: запуск DBSCAN-only прогона поверх уже сохранённых фич
    // configs[config_idx].result.{avg_peaks,avg_intervals,helpful_array}. Использует ту же
    // in_flight/cancel/future-машинерию, что и run_async, поэтому polls/Stop/queue-логика в GUI
    // обрабатывают его прозрачно. Возвращает false, если уже идёт другой расчёт, индекс невалиден
    // или нет валидного предыдущего результата.
    bool run_recluster_async(ParametricEngine& engine, int config_idx);
    void request_cancel();
    bool poll();
    BasinsPhaseSlot& phase_slot(int i);
    // Пересобрать НУ слота из результата бассейнов: по одной случайной ячейке
    // на бассейн (фиксированный seed), 0-бассейны отброшены, срез по
    // pp_max_attractors_text. Возвращает true, если НАБОР ячеек изменился —
    // только тогда нужен пересчёт траекторий; иначе обновились лишь подписи.
    bool rebuild_phase_ics(int config_idx);
    // Поставить портреты config'а в очередь пересчёта (дубликаты схлопываются).
    void request_phase_run(int config_idx);
    // Снять следующий элемент очереди и стартовать. Безопасно звать каждый кадр.
    bool start_next_phase();
    // Раз в кадр: забрать готовые результаты. true в кадр любого завершения.
    bool poll_phase();
};

// Сессия = сохраняемое состояние (FastSyncAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только FastSyncAnalysisState.
struct FastSyncAnalysisSession : FastSyncAnalysisState {
    std::future<FastSyncResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
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

// Сессия = сохраняемое состояние (LyapunovSpectrumAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только LyapunovSpectrumAnalysisState.
struct LyapunovSpectrumAnalysisSession : LyapunovSpectrumAnalysisState {
    std::future<LS1DResult> run_future;
    // Параллельный future для 2D-режима — гарантия: одновременно в полёте только
    // одна задача (1D или 2D); какая именно — is_2d_run.
    std::future<LS2DResult> run_future_2d;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    // Cooperative cancellation — see BifurcationAnalysisSession::cancel_token.
    std::shared_ptr<std::atomic<bool>> cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::chrono::steady_clock::time_point last_run_completed_at;
    LyapunovSpectrumAnalysisSession() = default;
    LyapunovSpectrumAnalysisSession(LyapunovSpectrumAnalysisSession&&) = default;
    LyapunovSpectrumAnalysisSession& operator=(LyapunovSpectrumAnalysisSession&&) = default;
    LyapunovSpectrumAnalysisSession(const LyapunovSpectrumAnalysisSession&) = delete;
    LyapunovSpectrumAnalysisSession& operator=(const LyapunovSpectrumAnalysisSession&) = delete;
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void remove_curve(int i);
    bool run(ParametricEngine& engine, int curve_idx);
    bool run_async(ParametricEngine& engine, int curve_idx);
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int curve_idx) const;
    // Задача фонового прогрева: копирует запрос НА ВЫЗЫВАЮЩЕМ потоке (сессию читать из фонового
    // нельзя — её правит UI) и компилирует модуль, ничего не считая. Пустая функция, если
    // индекс плохой или КРС не собрался.
    std::function<void(ParametricEngine&)> prewarm_task(int curve_idx) const;
    void request_cancel();
    bool poll();
};

// Сессия = сохраняемое состояние (SignalMetricsAnalysisState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только SignalMetricsAnalysisState.
struct SignalMetricsAnalysisSession : SignalMetricsAnalysisState {
    std::future<SignalMetricsResult> run_future;
    bool in_flight = false;
    std::chrono::steady_clock::time_point compute_start_time;
    std::shared_ptr<std::atomic<bool>>  cancel_token;
    std::shared_ptr<std::atomic<float>> progress_token;
    std::chrono::steady_clock::time_point last_run_completed_at;
    SignalMetricsAnalysisSession() = default;
    SignalMetricsAnalysisSession(SignalMetricsAnalysisSession&&) = default;
    SignalMetricsAnalysisSession& operator=(SignalMetricsAnalysisSession&&) = default;
    SignalMetricsAnalysisSession(const SignalMetricsAnalysisSession&) = delete;
    SignalMetricsAnalysisSession& operator=(const SignalMetricsAnalysisSession&) = delete;
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    void remove_config(int i);
    bool run_async(ParametricEngine& engine, int config_idx);
    // Объём работы для оценки времени счёта траекторий (time_estimate.h).
    TrajWork traj_work(int config_idx) const;
    std::function<void(ParametricEngine&)> prewarm_task(int config_idx) const;
    void request_cancel();
    bool poll();
};

// Вспомогательные функции сессий (реализация в analysis; типы — в ucuda/domain).

// Names of built-in schemes enabled in SystemRecord (order matches kBuiltinSchemeNames).
// Empty when nothing is ticked; combo callers fall back to "show all built-ins".
std::vector<std::string> enabled_builtins_from_record(const SystemRecord& r);

// Резолвер имени схемы (compute_krs_for_scheme и соседи) идёт через кодоген и живёт в
// analysis/analysis_session.h.

// Session start step. Hard-pinned to 1 for a map: the engine derives "time" as
// t/h, so with h = 1 the computing time is exactly the iteration count.
std::string default_h_from_record(const SystemRecord& r);

// Session start scheme: "Map" for a map, otherwise left untouched.
std::string default_scheme_from_record(const SystemRecord& r, std::string current);

// Start value for BifurcationDiagramConfig::plot_all_iterates: a map's diagram
// is built from every iterate, a flow's from peaks (a Poincare section).
bool default_all_iterates_from_record(const SystemRecord& r);

// Найти результат под конкретный job. nullptr, если такого расчёта в результате нет
// (конфиг поменяли после Run, или расчёт пропустили).
const rqa::Result* find_rqa(const AnalysisResult& res, const RqaJob& job);

// Собрать RqaJob из проекции: разбор текстовых полей + перевод RR из процентов в долю.
// Возвращает false, если проекция не RecurrencePlot или источник ещё не выбран.
bool make_rqa_job(const Projection& pr, RqaJob& out);

// Перенумерация бассейнов (spiral-from-center, порт MATLAB)
// Положительные id → 1,2,3... в порядке первого появления при обходе спиралью
// из центра, отрицательные → -1,-2,-3... Нули (diverged) не трогаются.
std::vector<int> renumber_basins_spiral(const std::vector<int>& src, int N,
                                        int& out_n_pos, int& out_n_neg);
// Лениво (пере)заполнить c.basin_idx_spiral для текущего data_generation.
// Безопасно звать каждый кадр — пересчёт только при смене поколения данных.
void ensure_basins_spiral_cache(BasinsConfig& c);
// Массив id'ов в ТЕКУЩЕЙ нумерации (учитывает renumber_spiral).
// nullptr, если результата ещё нет.
const int* basins_display_ids(BasinsConfig& c);
// Подпись только тех входов, что влияют на ФОРМУ траекторий (интегратор,
// время, параметры, НУ не-осевых переменных, устройство). Сюда намеренно НЕ
// входят data_generation / renumber_spiral / max_attractors: они меняют лишь
// НАБОР представительных ячеек, а он сравнивается напрямую.
std::string build_basins_phase_run_signature(const BasinsConfig& c);
// Полная подпись (run-подпись + то, что влияет на набор ячеек). Равные строки
// ⇔ портреты актуальны.
std::string build_basins_phase_signature(const BasinsConfig& c);
