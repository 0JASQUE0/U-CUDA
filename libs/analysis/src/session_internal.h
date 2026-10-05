#pragma once
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/domain/circuit_synth.h"
#include "ucuda/domain/circuit_solver.h"
#include "ucuda/cuda/phase_portrait_nvrtc.h"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <memory>
#include <chrono>
#include <thread>   // наблюдатель отмены CPU-ветки адаптивного шага
#include <random>   // выбор представительной ячейки бассейна (mt19937)

namespace ucuda::analysis_detail {
















// Слот в a[] для цели свипа по параметру. param_index конфига — 0-based индекс
// в списке параметров системы, а в a[] параметры сдвинуты на единицу, потому
// что a[0] занят коэффициентом симметрии s. Отдельное значение -1 означает
// "свипать сам s": он и есть a[0], поэтому движку хватает индекса и никакого
// отдельного флага (в отличие от sweep_over_h / sweep_over_var) не нужно.
// Пункт "s (a[0])" в UI показывается только у схем, которые a[0] читают.
int sweep_param_slot(int idx, int nparams) ;


// Mirrors the top-bar indicator text to the console on completion so users
// who keep stdout visible can see timings without watching the title bar.
void log_run_completed(const char* label, bool ok, double secs) ;


// Host-порт peakFinder из cudaLibrary.cu.
//
// Зачем копия, а не вызов оригинала: peakFinder объявлен как __device__ __host__ в
// cudaLibrary.cuh, который парсит только nvcc, а этот файл собирает MSVC. Тот же приём уже
// применён для getValueByIdx (getValueByIdx_local в parametric_engine.cpp).
//
// Отличия от оригинала — только в способе доступа к настройкам: пороги берутся из PeakConfig (GUI
// Settings), а не из constexpr-дефолтов configCUDA.h, потому что на GPU те же значения инжектятся
// как #define перед компиляцией (peak_config_defines). Логика поиска пиков, интерполяция вершины
// и фильтр по eps_interPeak_delta повторены один в один — иначе Feature diagram показывала бы не
// те точки, что реально уходят в DBSCAN 2D-бифуркации.
//
// dt — шаг ПО ВРЕМЕНИ между соседними отсчётами `data`. Вызывающий передаёт h*decimator и сам ряд,
// прореженный тем же шагом: признаки должны считаться по тому же сигналу, что уходит в
// peakFinderCUDA. Возвращает пары (значение пика; интервал от предыдущего удержанного пика).
FeaturePoints find_peaks_host(const std::vector<double>& data, double dt) ;


// Host-порт PeakStreamNU из kernels/adaptive_part.cu: пики на узлах адаптивного шага.
// Копия по той же причине, что и find_peaks_host выше, и повторяет оригинал один в один:
// пик — смена знака производной сигнала с + на - между соседними узлами; принимается,
// если вершина поднялась над минимумом сигнала после предыдущего пика не меньше чем на
// eps_peak_delta и выше peak_threshold. Вершина — по interp: 0 — больший из двух узлов,
// 1 — парабола по трём последним узлам с неравными шагами, 2 — кубический Эрмит на шаге
// со сменой знака по значениям и точным производным. Время пика абсолютное, интервалы и
// их фильтр eps_interPeak_delta — как у find_peaks_host.
//
// x, d, t — значение сигнала, его производная и время узлов; прореживание (каждый dec-й
// узел и последний) уже сделано вызывающим — тем же правилом, что в ucudaAdRecordNodes.
FeaturePoints find_peaks_host_nodes(const std::vector<double>& x, const std::vector<double>& d,
                                           const std::vector<double>& t, int interp) ;


// Парсит значение (число или дробь a/b) в double; пусто -> fallback.
inline double parse_val(const std::string& s, double fallback) ;


// Снапшот всех входов phase-расчёта. Создаётся на главном потоке, передаётся
// worker'у при async-режиме. Все поля — by value, никакой ссылки на session.

struct PhaseRunInputs {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    std::string step_h, sim_time, skip_time;
    std::string scheme, decimation;
    std::string symmetry_s = "0.5"; // a[0] для CD
    AdaptiveSettings adaptive;
    std::shared_ptr<std::atomic<bool>> cancel;   // отмена (адаптивный шаг), nullptr — нет
    bool        use_gpu = true;
    System      sys;
    std::string krs_code;
    // krs_code для custom схемы содержит тело дословно (см. regenerate_krs).
    // Флаг нужен, чтобы CPU-ветка знала: это сырой C, а не codegen-вывод.
    bool        krs_is_custom = false;
    std::vector<CustomScheme> custom_schemes;   // база Extr у адаптивного шага (adaptive_code_for_scheme)
    std::map<std::string, std::string>  param_values;
    std::vector<InitialConditionSet>    ic_sets;
    // Continuation mode plumbing. When ic_override is populated it takes
    // precedence over parsing ic_sets, and skip_transient=false zeroes the
    // in-kernel transient loop so the resumed chunk starts recording immediately.
    bool                              skip_transient = true;
    std::vector<std::vector<double>>  ic_override;   // [ic][coord]; empty → use ic_sets text
    // RQA-расчёты, запрошенные окнами проекций. Уже дедуплицированы по (ic, cfg).
    std::vector<RqaJob>               rqa_jobs;
    // Схемная траектория (фаза 4a).
    bool        circuit_show        = false;
    bool        circuit_ideal_opamp = true;
    std::string circuit_target_volt, circuit_gbw_mhz, circuit_vsat, circuit_substeps;
};


// Чистая функция: входы → AnalysisResult. Не трогает session, поэтому её
// безопасно звать с любого потока.
// Схемная траектория поверх ОДУ. Кладётся ещё одной строкой в trajectories: оттуда
// она сама доедет до всех проекций, легенды и экспорта, без правок в отрисовке.
//
// Масштаб ИЗМЕРЯЕТСЯ по уже посчитанной траектории ОДУ, а не подбирается: схема
// живёт в вольтах, и у Лоренца из библиотеки без этого вышло бы +-45 В. Наружу
// траектория домножается на масштаб обратно — иначе две кривые в одном окне не
// с чем было бы сравнивать.
void append_circuit_trajectory(const PhaseRunInputs& in,
                                      const std::vector<double>& a,
                                      double h, double tsim, int dec,
                                      AnalysisResult& result) ;


// Лог попыток шага на траекторию: хватает на наглядный h(t), а память
// ограничена (4 числа на запись, 6.4 МБ на НУ).
inline constexpr int kAdaptLogCap = 200000;


// Ветка адаптивного шага для compute_phase_portrait: GPU (ядро phase_kernel_ad) или
// CPU (computePhasePortraitCPU_adaptive) — один и тот же драйвер. raw[k] — траектория
// НУ k, times[k] — моменты её точек на узлах шага (на равномерной сетке пусто).
// false — ошибка настройки или компиляции, текст уже в result.error; расхождение
// траектории ошибкой не считается и уходит в calc_ok / calc_err, как у постоянного шага.
bool run_adaptive_phase(const PhaseRunInputs& in, int dim, int N,
                               const std::vector<double>& ic_flat, const std::vector<double>& a,
                               double h, double tsim, double tskip, int total,
                               std::vector<std::vector<std::vector<double>>>& raw,
                               std::vector<std::vector<double>>& times,
                               AnalysisResult& result, bool& calc_ok, std::string& calc_err) ;


AnalysisResult compute_phase_portrait(const PhaseRunInputs& in) ;


// Снапшот текущей session в PhaseRunInputs. Делается на главном потоке.
PhaseRunInputs snapshot_phase(PhaseAnalysisSession& s) ;


// генерация КРС из системы по выбранному методу (для NVRTC)
Scheme scheme_from_string(const std::string& s) ;


// BifurcationAnalysisSession

// Резолвит КРС по имени scheme: сперва среди custom_schemes (имя имеет
// приоритет над built-in, что блокируется в System tab), затем разбирает имя
// как экстраполяционную обёртку "Extr(<база>|n...)", иначе генерирует через
// codegen_scheme. Чистая функция — зовётся при сборке Request в момент Run
// (не персистится). Переиспользуется bifurcation и LLE.
//
// Тело, порядок и симметричность базы обёртки (Extr, Comp, адаптивный Extr): у кастомной
// КРС их объявил автор, у встроенной берём из паспорта. Неизвестная база — отказ, а не
// тихий откат на Эйлер. Одна функция на все обёртки — иначе они разъехались бы по тому,
// что считают допустимой базой.
bool resolve_wrapper_base(const std::vector<CustomScheme>& custom_schemes, const System& sys,
                                 const std::string& base_name, std::string (*gen)(const System&, Scheme),
                                 std::string& body, int& p, bool& sym) ;


// gen отличает GPU-форму от CPU-эквивалента: обе проходят ОДИН И ТОТ ЖЕ разбор
// имени, поэтому панель KRS и движок не могут понять имя по-разному.
std::string krs_for_scheme_impl(const std::vector<CustomScheme>& custom_schemes,
                                       const System& sys,
                                       const std::string& scheme,
                                       std::string (*gen)(const System&, Scheme)) ;


// Парсит double, понимая дробь "a/b" — поведение симметрично с parse_val
// (см. строки 10–18 этого же файла, которой пользуется Phase). Иначе
// Parametric получал бы "8/3" как 8, а Phase как 2.667.
inline double parse_d(const std::string& s, double def) ;

int parse_i(const std::string& s, int def) ;


// Адаптивный шаг вкладки -> AdaptiveRequest движка. Ошибка настройки (схема без
// оценки ошибки, неразборный rtol...) не бросается, а уходит в setup_error: движок
// вернёт её вместо расчёта, как любую другую ошибку запуска. span — длина всего
// интервала интегрирования (транзиент + запись), нужна выбору h0. ax0 / ax1 —
// коды адаптивных осей (AdaptiveAxisKind), 0 — обычная цель свипа.
void fill_adaptive_request(const std::vector<CustomScheme>& custom_schemes,
                                  const System& sys, const std::string& scheme,
                                  const AdaptiveSettings& st, int n, double span,
                                  int ax0, int ax1, AdaptiveRequest& out) ;


// Снапшот текущих GUI-полей конкретной БД в Bifurcation1DRequest. Делается на
// главном потоке перед стартом async-расчёта, чтобы worker работал со
// стабильной копией и пользователь мог в это время менять поля без race.
// KRS резолвится здесь же, не персистится в session.
Bifurcation1DRequest build_bif1d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) ;


// Применить только что готовый Bifurcation1DResult к конкретной БД.
void apply_bif1d_result(BifurcationDiagramConfig& bd, Bifurcation1DResult&& r) ;


Bifurcation2DRequest build_bif2d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) ;


void apply_bif2d_result(BifurcationDiagramConfig& bd, Bifurcation2DResult&& r) ;


LLE1DRequest build_lle1d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) ;


void apply_lle1d_result(LLECurveConfig& c, LLE1DResult&& r) ;


// Снапшот для 2D-режима. Те же общие поля, что у 1D, + поля второй оси и
// mixed_mode. Несимметричный режим (одна param, другая IC без mixed_mode)
// engine отвергает в run_lle_2d — здесь только пробрасываем поля.
LLE2DRequest build_lle2d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) ;


void apply_lle2d_result(LLECurveConfig& c, LLE2DResult&& r) ;


Dft1DRequest build_dft1d_request(const Dft1DAnalysisSession& s,
                                        const Dft1DConfig& c) ;


void apply_dft1d_result(Dft1DConfig& c, Dft1DResult&& r) ;


BasinsRequest build_basins_request(const BasinsAnalysisSession& s,
                                          const BasinsConfig& c) ;


// Гарантия синка enum class BasinFeature ↔ BF_* кодов из configCUDA.h.
// Если кто-то изменит одну сторону без второй — фейл компиляции, а не runtime.
static_assert((int)BasinFeature::AvgPeaks            == BF_AVG_PEAKS,            "BasinFeature drift: AvgPeaks");

static_assert((int)BasinFeature::AvgIntervals        == BF_AVG_INTERVALS,        "BasinFeature drift: AvgIntervals");

static_assert((int)BasinFeature::RMSPeaks            == BF_RMS_PEAKS,            "BasinFeature drift: RMSPeaks");

static_assert((int)BasinFeature::RMSIntervals        == BF_RMS_INTERVALS,        "BasinFeature drift: RMSIntervals");

static_assert((int)BasinFeature::StDevPeaks          == BF_STDEV_PEAKS,          "BasinFeature drift: StDevPeaks");

static_assert((int)BasinFeature::StDevIntervals      == BF_STDEV_INTERVALS,      "BasinFeature drift: StDevIntervals");

static_assert((int)BasinFeature::LogAvgPeaks         == BF_LOG_AVG_PEAKS,        "BasinFeature drift: LogAvgPeaks");

static_assert((int)BasinFeature::LogAvgIntervals     == BF_LOG_AVG_INTERVALS,    "BasinFeature drift: LogAvgIntervals");

static_assert((int)BasinFeature::LogRMSPeaks         == BF_LOG_RMS_PEAKS,        "BasinFeature drift: LogRMSPeaks");

static_assert((int)BasinFeature::LogRMSIntervals     == BF_LOG_RMS_INTERVALS,    "BasinFeature drift: LogRMSIntervals");

static_assert((int)BasinFeature::LogStDevPeaks       == BF_LOG_STDEV_PEAKS,      "BasinFeature drift: LogStDevPeaks");

static_assert((int)BasinFeature::LogStDevIntervals   == BF_LOG_STDEV_INTERVALS,  "BasinFeature drift: LogStDevIntervals");


void apply_basins_result(BasinsConfig& c, BasinsResult&& r) ;


// Basins → фазовые портреты: по одной случайной ячейке сетки на каждый найденный бассейн. Из
// ячейки восстанавливаются НУ двух осевых переменных (той же формулой, что в ядре — getValueByIdx
// в cudaLibrary.cu), остальные берутся из config'а. Бассейны с id == 0 (расходящиеся)
// пропускаются.

// Seed фиксированный: один и тот же результат бассейнов всегда даёт одни и те
// же представительные точки (в т.ч. после перезапуска приложения). Домешивание
// id бассейна делает выбор независимым — появление/исчезновение одного
// бассейна не сдвигает точки всех остальных.
inline const unsigned kBasinsPhaseSeed = 12345u;


// Координаты обхода NxN-сетки по спирали из центра (порт MATLAB spiral_coords_from_center).
// Стартовая клетка — округление к верху центра: для N=5 это (2,2), для N=4 — (1,1), 0-based.
// Дальше right→down→left→up с увеличением шага на каждом цикле полуоборотов; точки за границей
// сетки пропускаются, поэтому набирается ровно N*N валидных индексов. Возвращает row-major
// индексы row*N + col.
std::vector<int> spiral_coords_from_center(int N) ;


// Значение узла сетки — зеркало getValueByIdx (cudaLibrary.cu): та же
// lerp-форма и те же вырожденные случаи, иначе восстановленное НУ не совпадёт
// с тем, что реально считало ядро для этой ячейки.
double basins_grid_value(int point_idx, int n_pts, double lo, double hi) ;


std::string basins_fmt_num(double v) ;


// Перенос настроек basins-config'а в фазовую подсессию. Зовётся ровно перед
// запуском (не каждый кадр), как apply_shared_to_* в Custom-табе.
void apply_basins_config_to_phase(const BasinsAnalysisSession& s,
                                         const BasinsConfig& c,
                                         PhaseAnalysisSession& ph) ;


FastSyncRequest build_fastsync_request(const FastSyncAnalysisSession& s,
                                              const FastSyncConfig& c) ;


void apply_fastsync_result(FastSyncConfig& c, FastSyncResult&& r) ;


LS1DRequest build_ls1d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) ;


void apply_ls1d_result(LSCurveConfig& c, LS1DResult&& r) ;


// 2D-Request: те же общие поля + поля второй оси (param_index_2 / var_sweep_index_2).
LS2DRequest build_ls2d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) ;


void apply_ls2d_result(LSCurveConfig& c, LS2DResult&& r) ;


SignalMetricsRequest build_metrics_request(const SignalMetricsAnalysisSession& s,
                                                  const SignalMetricsConfig& c) ;


void apply_metrics_result(SignalMetricsConfig& c, SignalMetricsResult&& r) ;


}
