#pragma once
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/domain/circuit_synth.h"
#include "ucuda/domain/circuit_solver.h"
#include "ucuda/cuda/phase_portrait_nvrtc.h"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/numerics/multistep_bdf.h"   // BDF 1-5: свой драйвер многошагового метода
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <memory>
#include <chrono>
#include <thread>   // наблюдатель отмены CPU-ветки адаптивного шага
#include <random>   // выбор представительной ячейки бассейна (mt19937)


#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/equations/adaptive_params.h"
namespace ucuda::analysis_detail {
int sweep_param_slot(int idx, int nparams) ;
void log_run_completed(const char* label, bool ok, double secs) ;
std::string stop_note_for(bool cancelled, bool ok, const std::vector<int>& flags) ;
FeaturePoints find_peaks_host(const std::vector<double>& data, double dt) ;
FeaturePoints find_peaks_host_nodes(const std::vector<double>& x, const std::vector<double>& d,
                                           const std::vector<double>& t, int interp) ;
double parse_val(const std::string& s, double fallback) ;

struct PhaseRunInputs {
    std::vector<std::string> vars;
    std::vector<std::string> params;
    std::string step_h, sim_time, skip_time;
    std::string scheme, decimation;
    std::string symmetry_s = "0.5"; // a[0] для CD
    std::string cd_fp_iters = "4";   // CD FP: число простых итераций, a[M+1]
    NewtonSettings newton;          // Ньютон неявных схем: a[M+2..M+4] (см. codegen.hpp)
    AdaptiveSettings adaptive;
    std::shared_ptr<std::atomic<bool>> cancel;   // отмена (адаптивный шаг), nullptr — нет
    bool        use_gpu = true;
    System      sys;
    std::vector<std::string> obs_exprs;   // наблюдаемые z(x) (PhaseAnalysisSession::obs_exprs)
    std::string krs_code;
    std::string krs_error;   // причина пустой krs_code (отказ кодгена), пусто — не известна
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
    // Номер шага, с которого стартует траектория (явное t = n * h): у continuation-куска —
    // final_step прошлого куска, иначе 0. Автономным системам безразличен.
    long long                         step0 = 0;
    // RQA-расчёты, запрошенные окнами проекций. Уже дедуплицированы по (ic, cfg).
    std::vector<RqaJob>               rqa_jobs;
    // Схемная траектория (фаза 4a).
    bool        circuit_show        = false;
    bool        circuit_ideal_opamp = true;
    std::string circuit_target_volt, circuit_gbw_mhz, circuit_vsat, circuit_substeps;
};
void append_circuit_trajectory(const PhaseRunInputs& in,
                                      const std::vector<double>& a,
                                      double h, double tsim, int dec,
                                      AnalysisResult& result) ;


// Лог попыток шага на траекторию: хватает на наглядный h(t), а память
// ограничена (4 числа на запись, 6.4 МБ на НУ).
static constexpr int kAdaptLogCap = 200000;
void trim_step_log_to_record(AnalysisResult& result, double t_skip) ;
bool run_adaptive_phase(const PhaseRunInputs& in, int dim, int N,
                               const std::vector<double>& ic_flat, const std::vector<double>& a,
                               double h, double tsim, double tskip, int total,
                               std::vector<std::vector<std::vector<double>>>& raw,
                               std::vector<std::vector<double>>& times,
                               AnalysisResult& result, bool& calc_ok, std::string& calc_err) ;
AnalysisResult compute_phase_portrait(const PhaseRunInputs& in) ;
PhaseRunInputs snapshot_phase(PhaseAnalysisSession& s) ;
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
// Кодген отказывает исключением с читаемой причиной (cd_check_complex_safe,
// jac_check_differentiable, ...). Наверх уходит пустая КРС, а причина — в *err:
// без неё пользователь видел только "КРС пуста" и гадал, что не так с системой.
template <class F>
static std::string krs_try(std::string* err, F&& f) {
    try { return f(); }
    catch (const std::exception& e) { if (err) *err = e.what(); }
    catch (...) { if (err) *err = "code generation failed"; }
    return {};
}
bool resolve_wrapper_base(const std::vector<CustomScheme>& custom_schemes, const System& sys,
                                 const std::string& base_name, std::string (*gen)(const System&, Scheme),
                                 std::string& body, int& p, bool& sym, std::string* err = nullptr) ;
std::string krs_for_scheme_impl(const std::vector<CustomScheme>& custom_schemes,
                                       const System& sys,
                                       const std::string& scheme,
                                       std::string (*gen)(const System&, Scheme),
                                       std::string* err) ;
double parse_d(const std::string& s, double def) ;
int parse_i(const std::string& s, int def) ;
std::string sweep_krs(const std::vector<CustomScheme>& cs, const System& sys, const std::string& scheme,
                             const AdaptiveSettings& ad) ;
void fill_adaptive_request(const std::vector<CustomScheme>& custom_schemes,
                                  const System& sys, const std::string& scheme,
                                  const AdaptiveSettings& st_in, int n, double span,
                                  int ax0, int ax1, AdaptiveRequest& out, double h_fixed) ;


// Тела свипа добавили переменные (BDF индекса 3: множители GGL) — модуль считает amountOfX + extra_x,
// хвост НУ — нули (mu = 0 на решении). Результаты читаются по прежним индексам x / z.
template <class R>
static void sweep_extend_state(R& req) {
    if (req.adaptive.extra_x <= 0 || !req.adaptive.setup_error.empty()) return;
    req.amountOfX += req.adaptive.extra_x;
    req.initial_conditions.resize((size_t)req.amountOfX, 0.0);
}
Bifurcation1DRequest build_bif1d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) ;
void apply_bif1d_result(BifurcationDiagramConfig& bd, Bifurcation1DResult&& r) ;
Bifurcation2DRequest build_bif2d_request(const BifurcationAnalysisSession& s,
                                                const BifurcationDiagramConfig& bd) ;
void apply_bif2d_result(BifurcationDiagramConfig& bd, Bifurcation2DResult&& r) ;
LLE1DRequest build_lle1d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) ;
void apply_lle1d_result(LLECurveConfig& c, LLE1DResult&& r) ;
LLE2DRequest build_lle2d_request(const LLEAnalysisSession& s,
                                        const LLECurveConfig& c) ;
void apply_lle2d_result(LLECurveConfig& c, LLE2DResult&& r) ;
Dft1DRequest build_dft1d_request(const Dft1DAnalysisSession& s,
                                        const Dft1DConfig& c) ;
void apply_dft1d_result(Dft1DConfig& c, Dft1DResult&& r) ;
BasinsRequest build_basins_request(const BasinsAnalysisSession& s,
                                          const BasinsConfig& c) ;
void apply_basins_result(BasinsConfig& c, BasinsResult&& r) ;


// Basins → фазовые портреты: по одной случайной ячейке сетки на каждый найденный бассейн. Из
// ячейки восстанавливаются НУ двух осевых переменных (той же формулой, что в ядре — getValueByIdx
// в cudaLibrary.cu), остальные берутся из config'а. Бассейны с id == 0 (расходящиеся)
// пропускаются.

// Seed фиксированный: один и тот же результат бассейнов всегда даёт одни и те
// же представительные точки (в т.ч. после перезапуска приложения). Домешивание
// id бассейна делает выбор независимым — появление/исчезновение одного
// бассейна не сдвигает точки всех остальных.
static const unsigned kBasinsPhaseSeed = 12345u;
std::vector<int> spiral_coords_from_center(int N) ;
double basins_grid_value(int point_idx, int n_pts, double lo, double hi, bool symlog) ;
std::string basins_fmt_num(double v) ;
void apply_basins_config_to_phase(const BasinsAnalysisSession& s,
                                         const BasinsConfig& c,
                                         PhaseAnalysisSession& ph) ;
FastSyncRequest build_fastsync_request(const FastSyncAnalysisSession& s,
                                              const FastSyncConfig& c) ;
void apply_fastsync_result(FastSyncConfig& c, FastSyncResult&& r) ;
LS1DRequest build_ls1d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) ;
void apply_ls1d_result(LSCurveConfig& c, LS1DResult&& r) ;
LS2DRequest build_ls2d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) ;
void apply_ls2d_result(LSCurveConfig& c, LS2DResult&& r) ;
SignalMetricsRequest build_metrics_request(const SignalMetricsAnalysisSession& s,
                                                  const SignalMetricsConfig& c) ;
void apply_metrics_result(SignalMetricsConfig& c, SignalMetricsResult&& r) ;
double work_inv_h(double h, bool sweep_x, double lo, double hi, int n, bool log_x,
                  bool sweep_y = false, double lo2 = 0.0, double hi2 = 0.0, bool log_y = false) ;
TrajWork make_traj_work(const std::string& body, int n, bool adaptive, bool cpu, bool continuation,
                        double points, double per_point) ;
double lyap_steps(double inv_h, double transient, double t_max, double vtr, double trajectories) ;
}
