#pragma once
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/platform/platform_paths.h"
#include <cstdlib>
#include <cstring>

// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>


#include "ucuda/numerics/krs_cpu.h"   // CPU-ветка continuation: КРС -> нативная функция шага
#include "ucuda/equations/krs_prep.h"  // стадия предрасчёта КРС (CD fast)
#include "ucuda/equations/codegen.hpp" // step_body_uses_time: явное время t в свипах пока не поддержано
#include "module_lru.h"
#include "perf_warmup.h"   // прогрев GPU перед замером Performance

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>


namespace ucuda::cuda_detail {
inline unsigned long env_value(const char* name, char* buf, size_t cap) { const char* v=std::getenv(name); if(!v)return 0; const size_t n=std::strlen(v); if(n>=cap)return (unsigned long)(n+1); std::memcpy(buf,v,n+1); return (unsigned long)n; }
int launch_block_size(size_t sharedPerThread) ;
void add_profiling_nvrtc_opts(std::vector<const char*>& opts, std::string& storage) ;
bool nvrtc_prune_kernels() ;
std::string prune_kernel_defines(const std::vector<const char*>& name_exprs, const std::string& src) ;
void dump_module_ptx(const char* src_name, const std::string& src, const std::string& ptx) ;
const char* nvrtc_fmad_opt() ;
bool gpu_free_budget(double reserve, size_t& out_bytes) ;
int par_or_var_2d(bool sweep_h_x, bool sweep_h_y, bool sweep_var_x, bool sweep_var_y) ;


constexpr int kBlockSize         = 32;
     // как в NonLinAnal::bifurcation1D
constexpr int kMaxAmountOfX      = 32;

constexpr int kMaxAmountOfValues = 64;


// Потолки вкладки Network. Сеть считается ОДНИМ блоком (узел = поток), её
// состояние лежит в динамической shared — отсюда и лимит на узлы, и лимит на
// shared. 32 КБ взяты с запасом к 48 КБ на блок: остальное нужно самому ядру
// под локальные массивы схемы.
constexpr size_t kNetworkSharedCap = 32u * 1024u;

// Потолок записанной траектории. Точки режутся прореживанием, но при большой
// сети и большой размерности упереться можно и им — лучше честная ошибка, чем
// cudaMalloc на несколько гигабайт.
constexpr size_t kNetworkOutCap = 512u * 1024u * 1024u;

// Узло-шагов на один запуск ядра. Блок один, то есть работает один SM, и
// запуск на весь расчёт легко перевалил бы за watchdog драйвера.
constexpr double kNetworkWorkBudget = 1.0e7;


// Зеркало constexpr-констант configCUDA.h. Сам заголовок сюда не включается — он читается как
// ТЕКСТ и уходит в NVRTC (src_configCUDA_h), поэтому значения дублируются литералами, как уже
// сделано в run_bif1d. При правке configCUDA.h эти значения надо править вместе с ним.
// Peak-knobs (doCalculatePeaks / eps_* / peak_threshold / max_amount_of_peaks) здесь больше НЕ
// дублируются: они настраиваются из GUI, и CPU-ветки читают их через get_peak_config(). Заодно
// ушло расхождение — kEpsFixedPoint был 1e-8 против 1e-6 в configCUDA.h, из-за чего CPU-ветка
// ловила fixed point не там, где GPU.
constexpr int    kCheckInterval      = 100;
      // CHECK_INTERVAL: циклы с записью траектории, метрики
constexpr int    kLoopCheckBlock     = 1000;
     // UCUDA_LOOP_CHECK_BLOCK: транзиент, LLE/LS, цикл пиков
constexpr double kPi    = 3.1415926535897932384626433832795;

constexpr double kEuler = 2.7182818284590452353602874713527;
int lyap_warm_blocks(double vt, double NT)
;
bool cancel_requested(const std::shared_ptr<std::atomic<bool>>& c)
;
void mark_cancelled_tail(std::vector<int>& flags, size_t j)
;
void fill_nan_tail(std::vector<double>& v, size_t from)
;
double lyap_value_from_kernel(double v, int& flag)
;
size_t steps_from_time_size_t(double t, double h)
;
int steps_from_time_int(double t, double h)
;


// Порт loopCalculateDiscreteModel_int (cudaLibrary.cu) на CPU. Отличие одно: там размерность —
// compile-time AMOUNTOFX, здесь она приходит параметром (в NVRTC-сборке AMOUNTOFX как раз
// раскрывается в amountOfX, т.е. численно это одно и то же).
// Возврат — REGIME_* из configCUDA.h (1 = oscillation, -1 = fixed point, 0 = unbound). Проверка
// расходимости — с тем же шагом, что у GPU-двойника: без записи траектории kLoopCheckBlock,
// с записью kCheckInterval; БД 1D с записью передаёт kLoopCheckBlock явно (на GPU это цикл пиков).
// ДАУ (этап 7): z0 точки свипа — на связь при её параметрах (шаг метода ДАУ при h = 0; у ОДУ — ничего).
// Двойник согласования в ucudaSetupSweepPoint и ядрах продолжения (cudaLibrary.cu, kernels/*_cont).
// Ось/свип по НУ алгебраической переменной z нативной ДАУ: z каждой точки решается из связи, так что
// заданное значение — лишь стартовое приближение Ньютона (выбор ветви), а не координата.
static const char* const kDaeZAxisMsg = "DAE: an initial-condition axis must be a differential variable "
    "(algebraic z is solved from the constraint at every point)";
bool dae_z_axis(const std::string& body, int var) ;
void dae_point_start(const KrsCpuStep& step, const std::string& body, numb* x, const numb* a) ;
int cpu_loop_model(KrsCpuStep::StepFn step,
                   numb* x, const numb* a, numb h,
                   size_t iterations, int amountOfX, int preScaller,
                   int writableVar, numb maxValue,
                   numb* data,
                   numb* boxMin = nullptr, numb* boxMax = nullptr,
                   int checkEvery = 0)
;
int cpu_peak_finder(const numb* data, size_t amountOfPoints,
                    numb* outPeaks, numb* timeOfPeaks, numb h,
                    bool emit_all)
;
double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) ;
std::vector<numb> to_numb(const std::vector<double>& v) ;


struct CpuRandState { unsigned long long state; };
void cpu_curand_init(unsigned long long seed, unsigned long long sequence,
                     unsigned long long offset, CpuRandState* s) ;
float cpu_curand_uniform(CpuRandState* s) ;
void fill_bif1d_snapshot(data_export::Bif1DSnapshot& sn, const Bifurcation1DRequest& req,
                                bool continuation, bool on_cpu) ;
Bifurcation1DResult run_bif1d_continuation_cpu(const Bifurcation1DRequest& req) ;
Bifurcation1DResult run_bif1d_cpu(const Bifurcation1DRequest& req) ;
std::string exe_dir() ;
std::string read_text_file(const std::string& path, std::string& err) ;
std::string replace_all(std::string s, const std::string& from, const std::string& to) ;
std::string lyap_nx_prefix(const std::string& krs_body) ;
bool ad_is_bdf(const AdaptiveRequest& ad) ;


// Явное время t (этап 2): шаг calculateDiscreteModel(X, ucuda_t, a, h), циклы cudaLibrary.cu
// ведут номер шага (UCUDA_STEP / UCUDA_NA в cudaLibrary.cuh); адаптивные — время драйвера S.t. Ставят только модули, чьи ядра
// это проводят (БД, LLE/LS, Metrics, бассейны, DFT, Order; фиксированный и адаптивный шаг); build_module отказывает
// модулю без макроса. У автономного тела пусто — исходник модуля прежний.
static const char* const kNonautonomousDefine = "#define UCUDA_NONAUTONOMOUS\n";
std::string time_prefix(const std::string& krs_body) ;
int lyap_dim(const std::string& krs_body, int N) ;
std::string peak_config_defines() ;
std::string cu_err(CUresult r) ;
double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) ;
double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) ;


// run_lle1d_cpu — LLE(param) на CPU. Один код обслуживает два режима.
//
// continuation = false (классический свип): построчный порт LLEKernelCUDA — то, что GPU делает в
//   каждом потоке. Точки независимы: x сбрасывается на initial_conditions, щуп берётся из RNG с
//   subsequence = индекс точки (ровно как idx в kernel'е), прогревается transient_time, дальше
//   цикл Бенеттина. Нужен для сверки CPU-ядра с GPU и для счёта без GPU.
//
// continuation = true: точки выстроены в цепочку, переносится не только траектория x[], но и САМ
//   щуп y[] — к концу точки он уже развёрнут вдоль направления максимального растяжения, и
//   следующей точке не надо разворачивать его заново из случайного направления, отчего кривая
//   lambda(param) выходит глаже. GPU-двойника у режима нет и не будет: цепочка последовательна по
//   построению и выродилась бы в один поток. Прогрев отрабатывается на КАЖДОЙ точке, но начиная
//   со второй идёт NT-блоками с перенормировкой — щуп остаётся прицепленным и не теряет
//   ориентацию, поэтому transient_time сохраняет прежний смысл, а результаты сравнимы с классикой.
//
// Внутри точки алгоритм в обоих режимах одинаков: Бенеттин с одним вектором возмущения,
// перенормировка каждые k = ucuda_steps_per_block(NT, h) шагов,
// lambda = sum(log(|dX|/eps)) / (nBlocks*k*h) — делится на проинтегрированное время, не на tMax.
// Снимок кривой LLE/LS 1D для экспорта (правый клик, export_lle1d / export_ls1d) и CSV —
// один на все ветки: CPU, continuation на GPU, адаптивная цепочка. GPU-классика заполняет
// свой снимок сама, по своим локальным копиям. on_cpu — строка "device = CPU" вместо fmad.
template <class Snap, class Req>
static void fill_curve1d_snapshot(Snap& sn, const Req& req, bool continuation, bool on_cpu) {
    sn.values.assign(req.base_values.begin(), req.base_values.end());
    sn.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
    sn.tMax            = req.t_max;
    sn.NT              = req.NT;
    sn.vectorTransient = req.vector_transient;
    sn.transientTime   = req.transient_time;
    sn.h               = req.h;
    sn.eps             = req.eps;
    sn.gpu_fmad        = get_nvrtc_fmad();
    sn.on_cpu          = on_cpu;
    sn.indexOfMutVar   = req.sweep_over_var ? req.var_sweep_index : req.param_index;
    sn.range_lo        = req.param_lo;
    sn.range_hi        = req.param_hi;
    sn.log_scale       = req.log_scale;
    sn.continuation    = continuation;
    sn.continuation_reverse = continuation && req.continuation_reverse;
    sn.step_control    = req.adaptive.enabled ? req.adaptive.desc : std::string();
}
void write_curve1d_csv(const LLE1DResult& r, const std::string& path) ;
void write_curve1d_csv(const LS1DResult& r, const std::string& path, int K) ;
void dae_project(const KrsCpuStep& step, const numb* x, numb* v, const numb* a, int k, int n) ;
LLE1DResult run_lle1d_cpu(const LLE1DRequest& req, bool continuation) ;
void cpu_build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) ;
void cpu_dft_block(const numb* data, int sizeOfBlock, const numb* window,
                   int nFreq, numb freq_lo, numb freq_hi, bool logFreqAxis,
                   numb h, numb* akcos, numb* bksin)
;
void cpu_projection_operator(const numb* a, const numb* b, numb* minuend, int n) ;
void cpu_gram_schmidt(const numb* a, numb* b, int n, numb* denominators = nullptr) ;
void cpu_gram_schmidt_dae(const numb* a, numb* b, int k, int n, numb* denominators = nullptr) ;
LS1DResult run_ls1d_cpu(const LS1DRequest& req, bool continuation) ;
Dft1DResult run_dft1d_cpu(const Dft1DRequest& req, bool continuation) ;
void metrics_finish(SignalMetricsResult& res, const SignalMetricsRequest& req) ;


// Host-двойник MetricsAccum из signal_metrics.template.cu: те же суммы в том же
// порядке, поэтому CPU и GPU расходятся только арифметикой шага, не метрик.
struct CpuMetricsAccum {
    size_t n = 0;
    numb   shift = 0, s1 = 0, s2 = 0;
    numb   mn = 0, mx = 0;
    numb   yPrev = 0, dPrev = 0;
    numb   sd = 0, sd2 = 0, sdd = 0, sdd2 = 0;

    void push(numb y) {
        if (n == 0) { shift = y; mn = y; mx = y; }
        const numb c = y - shift;
        s1 += c; s2 += c * c;
        if (y < mn) mn = y;
        if (y > mx) mx = y;
        if (n >= 1) {
            const numb d = y - yPrev;
            sd += d; sd2 += d * d;
            if (n >= 2) {
                const numb dd = d - dPrev;
                sdd += dd; sdd2 += dd * dd;
            }
            dPrev = d;
        }
        yPrev = y;
        ++n;
    }
};


// Host-двойник ExtremaInterp (signal_metrics.template.cu, SIGM_MINMAX_INTERP): вершина
// параболы через три соседних отсчёта для максимумов и минимумов. Без него галка Metrics
// "min/max by interpolated extrema in Fixed" на CPU молча ничего не делала.
struct CpuExtremaInterp {
    numb y0 = 0, y1 = 0;
    size_t n = 0;
    numb mx = 0, mn = 0;
    bool haveMax = false, haveMin = false;

    void push(numb y2) {
        if (n >= 2) {
            const bool isMax = y1 >= y0 && y1 >= y2 && (y1 > y0 || y1 > y2);
            const bool isMin = y1 <= y0 && y1 <= y2 && (y1 < y0 || y1 < y2);
            if (isMax || isMin) {
                const numb denom = y0 - (numb)2.0 * y1 + y2;
                numb v = y1;
                if (std::fabs(denom) > 1e-12) {
                    const numb delta = (numb)0.5 * (y0 - y2) / denom;
                    v = y1 - (numb)0.25 * (y0 - y2) * delta;
                }
                if (isMax && (!haveMax || v > mx)) { mx = v; haveMax = true; }
                if (isMin && (!haveMin || v < mn)) { mn = v; haveMin = true; }
            }
        }
        y0 = y1; y1 = y2; ++n;
    }
};
numb cpu_sm_variance(numb s1, numb s2, size_t n) ;
SignalMetricsResult run_metrics_cpu(const SignalMetricsRequest& req, bool continuation) ;
bool nvrtc_compile_ptx(int cc_major, int cc_minor,
                              const std::string& src, const char* src_name,
                              const std::vector<const char*>& headers_src,
                              const std::vector<const char*>& headers_name,
                              const std::vector<const char*>& name_exprs,
                              bool rdc, std::string& out_ptx, std::vector<std::string>& lowered,
                              std::string& err) ;
void trim_ls_spectrum(LS1DResult& r, int K) ;
void trim_ls_spectrum(LS2DResult& r, int K) ;
double stab_norm2(double m00, double m01, double m10, double m11) ;
const char* ptx_numb_type() ;
double ptx_op_weight(const std::string& op) ;
double ptx_function_cost(const std::string& ptx, const std::string& fn, bool& loops) ;
bool estimate_device(CUdevice& dev, int& cc_major, int& cc_minor, std::string& err) ;


// Калибровочные шаги на системе Лоренца: дешёвый (Эйлер) и дорогой (RK4). Их цену считает тот же
// krs_step_cost, что и цену пользовательской КРС, поэтому систематика счёта (свёртка констант,
// вынос инвариантов из цикла) сокращается в отношении время / операции. Два шага разной цены
// разделяют время на операции и постоянную добавку шага (цикл, shared, запись траектории): у
// Эйлера Лоренца она больше половины времени.
const char* const kCalibBodyCheap =
    "    numb k[3];\n"
    "    k[0] = a[1] * (X[1] - X[0]); k[1] = X[0] * (a[2] - X[2]) - X[1]; k[2] = X[0] * X[1] - a[3] * X[2];\n"
    "    for (int i = 0; i < 3; ++i) X[i] = X[i] + h * k[i];\n";

const char* const kCalibBody =
    "    numb k1[3], k2[3], k3[3], k4[3], Y[3];\n"
    "    k1[0] = a[1] * (X[1] - X[0]); k1[1] = X[0] * (a[2] - X[2]) - X[1]; k1[2] = X[0] * X[1] - a[3] * X[2];\n"
    "    for (int i = 0; i < 3; ++i) Y[i] = X[i] + (numb)0.5 * h * k1[i];\n"
    "    k2[0] = a[1] * (Y[1] - Y[0]); k2[1] = Y[0] * (a[2] - Y[2]) - Y[1]; k2[2] = Y[0] * Y[1] - a[3] * Y[2];\n"
    "    for (int i = 0; i < 3; ++i) Y[i] = X[i] + (numb)0.5 * h * k2[i];\n"
    "    k3[0] = a[1] * (Y[1] - Y[0]); k3[1] = Y[0] * (a[2] - Y[2]) - Y[1]; k3[2] = Y[0] * Y[1] - a[3] * Y[2];\n"
    "    for (int i = 0; i < 3; ++i) Y[i] = X[i] + h * k3[i];\n"
    "    k4[0] = a[1] * (Y[1] - Y[0]); k4[1] = Y[0] * (a[2] - Y[2]) - Y[1]; k4[2] = Y[0] * Y[1] - a[3] * Y[2];\n"
    "    for (int i = 0; i < 3; ++i)\n"
    "        X[i] = X[i] + h * (k1[i] + (numb)2 * (k2[i] + k3[i]) + k4[i]) * ((numb)1 / (numb)6);\n";


// Длина кольца записи на нить в ядре калибровки.
constexpr int kCalibRing = 64;
std::string calib_source() ;
}
