#include "parametric_engine.h"
// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <windows.h>

#include "krs_cpu.h"   // CPU-ветка continuation: КРС -> нативная функция шага
#include "module_lru.h"

#include <algorithm>
#include <atomic>
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

// Глобальный PeakConfig — единственный источник истины для GUI-настраиваемых knobs
// configCUDA.h. Пишет UI-поток (Settings), читают worker-потоки при компиляции NVRTC и в
// CPU-ветках, поэтому под мьютексом. Epoch бампается на каждую запись и входит в cache-key
// модулей (hash_key) — иначе после смены настройки переиспользовался бы старый PTX.
namespace {
std::mutex g_peak_mu;
PeakConfig g_peak_cfg;
uint64_t   g_peak_epoch = 1;
}  // namespace

void set_peak_config(const PeakConfig& c) {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    g_peak_cfg = c;
    ++g_peak_epoch;
}
PeakConfig get_peak_config() {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    return g_peak_cfg;
}
uint64_t peak_config_epoch() {
    std::lock_guard<std::mutex> lk(g_peak_mu);
    return g_peak_epoch;
}

// FMA-контракция для NVRTC (развёрнуто в parametric_engine.h). Флаг всегда передаётся ЯВНО,
// даже совпадая с дефолтом NVRTC: молчаливое «по умолчанию» уже один раз разъехалось между
// картами и портретами. Хранится атомарно, а не под мьютексом: одно bool-поле, читателям нужен
// только свежий снимок.
namespace {
std::atomic<bool> g_nvrtc_fmad{ true };
}  // namespace

void set_nvrtc_fmad(bool enabled) { g_nvrtc_fmad.store(enabled, std::memory_order_relaxed); }
bool get_nvrtc_fmad()             { return g_nvrtc_fmad.load(std::memory_order_relaxed); }

// Раздельная компиляция (см. build_module). Отключается на случай драйвера, на котором линковка
// поведёт себя не так, как на проверенных: тогда движок собирает всё одной единицей трансляции,
// как раньше. Хранится атомарно по той же причине, что и fmad.
namespace {
std::atomic<bool> g_nvrtc_rdc{ false };
}  // namespace

void set_nvrtc_rdc(bool enabled) { g_nvrtc_rdc.store(enabled, std::memory_order_relaxed); }
bool get_nvrtc_rdc()             { return g_nvrtc_rdc.load(std::memory_order_relaxed); }

// Ширина блока запуска (см. parametric_engine.h). Тоже atomic и по той же причине: одно
// int-поле, читателю нужен только свежий снимок. В hash_key НЕ входит — на PTX не влияет.
namespace {
std::atomic<int> g_gpu_block_size{ kGpuBlockSizeDefault };
}  // namespace

void set_gpu_block_size(int threads) {
    g_gpu_block_size.store(clamp_gpu_block_size(threads), std::memory_order_relaxed);
}
int get_gpu_block_size() { return g_gpu_block_size.load(std::memory_order_relaxed); }

namespace {
// Ширина блока, ужатая под бюджет динамической shared-памяти этого запуска. 48 КБ — потолок
// на блок без opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize); широкая система (много X
// и a[]) при 128 потоках в него не влезет, и cuLaunchKernel вернёт CUDA_ERROR_INVALID_VALUE.
// Делим пополам, а не вычитаем: кратность варпу обязана сохраниться.
// sharedPerThread == 0 (ядро без динамической shared) — настройка применяется как есть.
int launch_block_size(size_t sharedPerThread) {
    int b = get_gpu_block_size();
    while (b > kGpuBlockSizeMin && sharedPerThread * (size_t)b > 48u * 1024u)
        b /= 2;
    return b;
}

// Строка опции для nvrtcCompileProgram. Литералы статические, поэтому указатель
// живёт дольше вызова.
const char* nvrtc_fmad_opt() {
    return get_nvrtc_fmad() ? "--fmad=true" : "--fmad=false";
}

// gpu_free_budget — свободная память GPU с применённым запасом; общий первый шаг всех расчётов
// чанкования. Возвращает false, если cudaMemGetInfo недоступен (вызывающие превращают это в
// свой fail("cudaMemGetInfo failed")).
//
// `reserve` — доля свободной памяти, которую анализу разрешено занять; единого значения нет и
// сведено оно намеренно НЕ было:
//     0.92  — bifurcation 1D/2D, DFT и h-свипы (сверху ещё SAFETY_FACTOR 0.9 и вычет
//             memConstants у вызывающего, т.е. фактически ~0.83)
//     0.9   — basins
//     0.5   — LLE 1D/2D и fastsync-сетка
//     1/16  — LS 1D/2D: per-system память там ~N (буферы возмущённых траекторий), но 1/16 всё
//             равно агрессивнее, чем следует из одной этой оценки.
// Происхождение 0.5 и 1/16 не восстановлено — это следы отладки проблем с памятью, а не расчёт,
// поэтому числа сохранены как есть. Менять только после замера на реальных сетках; выигрыш
// реальный (LS сейчас берёт шестнадцатую часть памяти и делает в разы больше чанков).
bool gpu_free_budget(double reserve, size_t& out_bytes) {
    size_t freeMemory = 0, totalMemory = 0;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess) return false;
    out_bytes = (size_t)((double)freeMemory * reserve);
    return true;
}
}  // namespace

namespace {

// par_or_var for a 2D sweep: 1 = both axes are parameters, 0 = both are initial conditions,
// 2 = mixed. Kept as one helper because run_* and prewarm() must derive the same cache key.
inline int par_or_var_2d(bool sweep_h_x, bool sweep_h_y, bool sweep_var_x, bool sweep_var_y) {
    if (sweep_h_x || sweep_h_y) return (sweep_h_x ? sweep_var_y : sweep_var_x) ? 0 : 1;
    if (sweep_var_x == sweep_var_y) return sweep_var_x ? 0 : 1;
    return 2;
}

constexpr int kBlockSize         = 32;     // как в NonLinAnal::bifurcation1D
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
constexpr int    kCheckInterval      = 100;      // CHECK_INTERVAL
constexpr double kPi    = 3.1415926535897932384626433832795;
constexpr double kEuler = 2.7182818284590452353602874713527;

// Число шагов интегрирования из времени: transient_time / h, NT / h и т.п.
//
// Прямой (int)(t / h) — UB, когда частное не влезает в int: TT=1e5 при h=1e-5 дают 1e10. На
// практике получался мусор или отрицательное число, т.е. транзиент молча не отрабатывал вовсе —
// худший вид ошибки, потому что расчёт выглядел успешным.
//
// Две версии, и выбор между ними НЕ косметический: он определяется типом приёмника. Значение
// уходит в ядро через void*[] в cuLaunchKernel, где тип обязан совпадать с параметром БАЙТ В
// БАЙТ — компилятор там ничего не проверит, рассогласование соберётся молча и развалит буфер
// аргументов на запуске.
//   ..._size_t — число ШАГОВ интегрирования; вся цепочка расширена до size_t (параметры ядер
//                amountOfPointsForSkip, device-функции loopCalculateDiscreteModel / _int со
//                счётчиками циклов, CPU-порт cpu_loop_model). Потолка нет — рабочая версия.
//   ..._int    — только под settleBlocks = TT / NT: это число NT-БЛОКОВ, а не шагов, оно живёт в
//                host-side счётчиках и на порядки меньше. INT_MAX здесь недостижим на осмысленных
//                входах, но каст всё равно идёт через проверку, а не вслепую.
static inline size_t steps_from_time_size_t(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (size_t)s;
}

static inline int steps_from_time_int(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (s >= 2147483647.0) ? 2147483647 : (int)s;
}

// Порт loopCalculateDiscreteModel_int (cudaLibrary.cu) на CPU. Отличие одно: там размерность —
// compile-time AMOUNTOFX, здесь она приходит параметром (в NVRTC-сборке AMOUNTOFX как раз
// раскрывается в amountOfX, т.е. численно это одно и то же).
// Возврат — REGIME_* из configCUDA.h (1 = oscillation, -1 = fixed point, 0 = unbound). Проверка
// расходимости — раз в kCheckInterval итераций, как на GPU.
int cpu_loop_model(KrsCpuStep::StepFn step,
                   numb* x, const numb* a, numb h,
                   size_t iterations, int amountOfX, int preScaller,
                   int writableVar, numb maxValue,
                   numb* data)
{
    for (size_t i = 0; i < iterations; ++i) {
        if (data != nullptr) {
            // writableVar < 0 -> комбинация первых (до трёх) переменных.
            if (writableVar < 0) {
                if      (amountOfX >= 3) data[i] = x[0] + (numb)kPi * x[1] + (numb)kEuler * x[2];
                else if (amountOfX == 2) data[i] = x[0] + (numb)kPi * x[1];
                else                     data[i] = x[0];
            } else {
                data[i] = x[writableVar];
            }
            for (int j = 0; j < preScaller; ++j) step(x, a, h);
        }
        else {
            step(x, a, h);
        }

        if (i % (size_t)kCheckInterval == 0) {
            numb checker = (numb)0;
            for (int j = 0; j < amountOfX; ++j) checker += std::fabs(x[j]);
            if (std::isnan(checker) || std::isinf(checker)) return REGIME_UNBOUND;
            if (maxValue != (numb)0 && std::fabs(checker) > maxValue) return REGIME_UNBOUND;
        }
    }

    // Лишний preScaller-шаг после основного цикла — он нужен GPU-версии для
    // проверки на неподвижную точку И заодно сдвигает x[], который в
    // continuation переносится в следующую точку параметра. Без него CPU-ветка
    // разошлась бы с GPU уже на второй точке свипа.
    static thread_local std::vector<numb> xPrev;
    xPrev.assign(x, x + amountOfX);
    for (int j = 0; j < preScaller; ++j) step(x, a, h);

    numb tempResult = (numb)0;
    for (int j = 0; j < amountOfX; ++j) tempResult += std::fabs(x[j] - xPrev[j]);
    if (std::fabs(tempResult) < (numb)get_peak_config().eps_fixed_point)
        return REGIME_FIXED_POINT;
    return REGIME_OSCILLATION;
}

// Порт peakFinder (cudaLibrary.cu) на один блок данных. Индексация здесь блок-локальная
// (startDataIndex == 0): CPU-ветка обрабатывает точки свипа потоково и держит один блок, а не
// матрицу nPts x record_steps. Логика — построчная копия, включая параболическую интерполяцию,
// сдвиг пиков влево на один и пересчёт timeOfPeaks в межпиковые интервалы.
int cpu_peak_finder(const numb* data, size_t amountOfPoints,
                    numb* outPeaks, numb* timeOfPeaks, numb h,
                    bool emit_all)
{
    // Снимок knobs на весь вызов: GPU-двойник видит их как compile-time
    // константы, поэтому в пределах одной точки свипа они меняться не должны.
    const PeakConfig pc = get_peak_config();

    if (emit_all || !pc.do_calculate_peaks) {
        int n = (int)amountOfPoints;
        if (n >= pc.max_amount_of_peaks) n = pc.max_amount_of_peaks;
        for (int i = 0; i < n; ++i) { outPeaks[i] = data[i]; timeOfPeaks[i] = (numb)0; }
        return n - 1;
    }
    // Границы циклов на GPU считаются в size_t как amountOfPoints - 2; при
    // крошечном блоке это ушло бы в переполнение. Отсекаем такие блоки здесь —
    // пиков в них всё равно быть не может.
    if (amountOfPoints < 5) return 0;

    int amountOfPeaks = 0;
    for (size_t i = 2; i + 2 < amountOfPoints; ++i) {
        if (data[i] - data[i - 1] > (numb)pc.eps_peak_delta &&
            data[i] > (numb)pc.peak_threshold &&
            data[i] >= data[i + 1])
        {
            for (size_t j = i; j + 2 < amountOfPoints; ++j) {
                // Наткнулись на точку строго больше — это был не пик.
                if (data[j] < data[j + 1]) { i = j + 1; break; }
                if (data[j] - data[j + 1] > (numb)pc.eps_peak_delta) {
                    if (pc.do_interpolate_peaks) {
                        const numb denom = data[j - 1] - (numb)2.0 * data[j] + data[j + 1];
                        numb delta = (numb)0;
                        if (std::fabs(denom) > (numb)1e-12)
                            delta = (numb)0.5 * (data[j - 1] - data[j + 1]) / denom;
                        outPeaks[amountOfPeaks]    = data[j] - (numb)0.25 * (data[j - 1] - data[j + 1]) * delta;
                        // Здесь индекс, а не время: на время умножаем ниже.
                        timeOfPeaks[amountOfPeaks] = (numb)(j - 1) + delta;
                    } else {
                        outPeaks[amountOfPeaks]    = data[j];
                        timeOfPeaks[amountOfPeaks] = (numb)(j - 1);
                    }
                    ++amountOfPeaks;
                    i = j + 1;   // два пика подряд невозможны
                    break;
                }
            }
        }
    }

    // Anchor-фильтр по eps_interPeak_delta — построчная копия GPU-peakFinder.
    // При eps_interPeak_delta == 0 (дефолт) вырождается в прежний сдвиг влево.
    if (amountOfPeaks > 1) {
        int  writeIdx   = 0;
        numb anchorTime = timeOfPeaks[0];
        for (int i = 1; i < amountOfPeaks; ++i) {
            const numb currentTime = timeOfPeaks[i];
            const numb delta = (currentTime - anchorTime) * h;
            if (delta >= (numb)pc.eps_interPeak_delta) {
                outPeaks[writeIdx]    = outPeaks[i];
                timeOfPeaks[writeIdx] = delta;
                ++writeIdx;
                anchorTime = currentTime;
            }
        }
        amountOfPeaks = writeIdx;
    } else {
        amountOfPeaks = 0;
    }
    if (amountOfPeaks >= pc.max_amount_of_peaks) amountOfPeaks = pc.max_amount_of_peaks;
    return amountOfPeaks;
}

// Генератор случайных чисел — побайтовая копия stub'а из шаблонов
// kernels/lle1d.template.cu (splitmix64-инициализация + LCG). Копия, а не
// повторная реализация: направления начального возмущения обязаны совпадать
// с GPU-веткой, иначе первые точки λ разошлись бы из-за разного «щупа».
// Значение свипа в точке j continuation-цепочки. Одна формула на все три
// анализа (Bif/LLE/LS) и на оба устройства — GPU-ядра считают её же.
// reverse — идём от hi к lo (гистерезис), log_scale — логарифмическая сетка.
// Тело — общая с ядрами реализация (ucuda_node_value_cont в configCUDA.h):
// та же функция, которую после правки зовут kernels/*_cont.template.cu.
// Раньше здесь стояла копия в double, а ядро считало в numb — в double-режиме
// одно и то же, при numb = float расходились.
inline double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) {
    return (double)ucuda_node_value_cont(j, nPts, (numb)lo, (numb)hi, log_scale, reverse);
}

// Request/Result — публичный интерфейс в double, а ядра работают в numb.
// Конвертация живёт ровно на границе: перед H2D сужаем, после D2H расширяем.
// Когда numb == double это no-op копия, поэтому обе конфигурации ведут себя
// одинаково и без #ifdef.
std::vector<numb> to_numb(const std::vector<double>& v) {
    return std::vector<numb>(v.begin(), v.end());
}

struct CpuRandState { unsigned long long state; };

void cpu_curand_init(unsigned long long seed, unsigned long long sequence,
                     unsigned long long offset, CpuRandState* s) {
    unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s->state = z ^ (z >> 31);
}

float cpu_curand_uniform(CpuRandState* s) {
    s->state = s->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(((s->state >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}

// run_bif1d_continuation_cpu — CPU-двойник run_bif1d_continuation.
//
// Зачем: GPU-версия гоняет ВЕСЬ свип в ОДНОМ потоке (gridDim = blockDim = 1, см.
// bifurcation1d_cont.template.cu) — continuation последователен по построению, каждая точка
// стартует с конечного x[] предыдущей, и распараллелить это нельзя, не сломав гистерезис.
// Одиночный GPU-поток на зависимой FP64-цепочке — заведомо медленное железо.
// КРС компилируется в нативную функцию шага через krs_cpu.h: это работает и для custom, и для
// встроенных схем (req.krs_body в обоих случаях обычный C — у встроенных вывод codegen_scheme).
//
// Два отличия от GPU-двойника, оба в лучшую сторону: память — один блок траектории вместо
// матрицы nPts x record_steps (GPU держит её трижды: d_data + d_outPeaks + d_timeOfPeaks); отмена
// и прогресс работают на каждой точке свипа, а не только до запуска монолитного kernel'а.
Bifurcation1DResult run_bif1d_continuation_cpu(const Bifurcation1DRequest& req) {
    Bifurcation1DResult res;
    auto fail = [&](const std::string& msg) -> Bifurcation1DResult& {
        res.error = msg; return res;
    };

    // Та же валидация, что и у GPU-двойника.
    if (req.krs_body.empty())                                 return fail("krs_body is empty");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("too many base_values");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index out of range");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var out of range");
    if (req.n_pts <= 0)         return fail("n_pts must be > 0");
    if (req.h <= 0.0)           return fail("h must be > 0");
    if (req.t_max <= 0.0)       return fail("t_max must be > 0");
    if (req.transient_time < 0) return fail("transient_time must be >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    // При h-свипе шаг свой в каждой точке, поэтому буфер блока выделяем под
    // ХУДШИЙ случай (минимальный h => больше всего записанных точек), а
    // реальную длину считаем внутри цикла. Так же поступает классический путь.
    // (std::min в скобках: windows.h тянет макрос min, а NOMINMAX тут не задан)
    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts = req.n_pts;
    res.n_pts        = nPts;
    res.record_steps = maxPointsInBlock;
    res.param_lo     = req.param_lo;
    res.param_hi     = req.param_hi;
    res.continuation_reverse = req.continuation_reverse;
    res.flags.assign(nPts, 0);
    res.bifurcation_points.assign(nPts, {});
    res.peak_times.assign(nPts, {});

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> peaks((size_t)maxPointsInBlock);
    std::vector<numb> times((size_t)maxPointsInBlock);

    numb h_local = (numb)req.h;

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          req.continuation_reverse, req.log_scale);
        if (req.sweep_over_h) h_local = (numb)p;
        else                  a[(size_t)req.param_index] = p;

        // При h-свипе число записываемых точек своё в каждой точке; буфер
        // выделен под худший случай, поэтому clamp'им к нему.
        int pointsInBlock = (h_local > 0.0)
            ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, h_local);
        const numb   timeStep   = h_local * (numb)req.pre_scaller;

        // Вырожденный шаг — траектории нет: это unbound, а не fixed point.
        if (h_local <= 0.0 || pointsInBlock <= 0) {
            res.flags[j] = REGIME_UNBOUND;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        // Transient: x[] НЕ сбрасываем — это и есть перенос с прошлой точки.
        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        if (flag == REGIME_UNBOUND) {
            res.flags[j] = REGIME_UNBOUND;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                              pointsInBlock, req.amountOfX,
                              req.pre_scaller, req.writable_var,
                              req.max_value, block.data());

        // Зеркалит continuation-ядро (bifurcation1d_cont.template.cu): в
        // d_amountOfPeaks уходит СЫРОЙ REGIME_*, и peakFinderCUDA пропускает и
        // FP, и unbound. Раньше unbound на основном участке проходил дальше, и
        // пики искались в уже разошедшемся блоке.
        // Emitting every sample keeps the fixed point: for a map a period-1
        // orbit is a real branch, and the block already holds its iterates.
        if (flag != REGIME_OSCILLATION &&
            !(req.emit_all_samples && flag == REGIME_FIXED_POINT)) {
            res.flags[j] = flag;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        const int n = cpu_peak_finder(block.data(), (size_t)pointsInBlock,
                                      peaks.data(), times.data(), timeStep,
                                      req.emit_all_samples);
        res.flags[j] = n;
        if (n > 0) {
            res.bifurcation_points[j].assign(peaks.begin(),  peaks.begin() + n);
            res.peak_times[j].assign        (times.begin(),  times.begin() + n);
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}

// run_bif1d_cpu — CPU-порт КЛАССИЧЕСКОГО (без continuation) 1D-свипа, двойник run_bif1d. Точки
// независимы: каждая стартует с одних и тех же НУ и базовых параметров, цепочки здесь нет.
// Флаги повторяют связку calculateDiscreteModelCUDA + peakFinderCUDA: kernel пишет
// checker = REGIME_* транзиента, основной участок считается только при OSCILLATION или
// FIXED_POINT, а peakFinderCUDA пропускает FP и UNBOUND, оставляя их коды в массиве.
// Continuation-ветка теперь ведёт себя так же (раньше метила unbound как -1).
// Свипуемая величина — параметр, НУ или сам шаг h, как на GPU; сетка берётся из cont_sweep_value
// (reverse только у continuation), как у CPU-веток LLE/LS/DFT.
Bifurcation1DResult run_bif1d_cpu(const Bifurcation1DRequest& req) {
    Bifurcation1DResult res;
    auto fail = [&](const std::string& msg) -> Bifurcation1DResult& {
        res.error = msg; return res;
    };

    // Диспетчер уже провалидировал req целиком; здесь — только то, от чего
    // зависит своя разметка буферов, плюс дешёвая страховка на прямой вызов.
    if (req.krs_body.empty())                                 return fail("krs_body is empty");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("too many base_values");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var out of range");
    if (req.n_pts <= 0)         return fail("n_pts must be > 0");
    if (req.h <= 0.0)           return fail("h must be > 0");
    if (req.t_max <= 0.0)       return fail("t_max must be > 0");
    if (req.transient_time < 0) return fail("transient_time must be >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    // При h-свипе шаг свой в каждой точке — буфер под худший случай
    // (минимальный h => больше всего записей), как в continuation-ветке.
    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts = req.n_pts;
    res.n_pts        = nPts;
    res.record_steps = maxPointsInBlock;
    res.param_lo     = req.param_lo;
    res.param_hi     = req.param_hi;
    res.flags.assign(nPts, 0);
    res.bifurcation_points.assign(nPts, {});
    res.peak_times.assign(nPts, {});

    // Снимок для right-click экспорта — те же поля, что заполняет run_bif1d.
    res.snapshot.values.assign(req.base_values.begin(), req.base_values.end());
    res.snapshot.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
    res.snapshot.tMax          = req.t_max;
    res.snapshot.transientTime = req.transient_time;
    res.snapshot.h             = req.h;
    res.snapshot.gpu_fmad      = get_nvrtc_fmad();
    res.snapshot.preScaller    = req.pre_scaller;
    res.snapshot.writableVar   = req.writable_var;
    res.snapshot.indexOfMutVar = req.sweep_over_var ? req.var_sweep_index : req.param_index;
    res.snapshot.range_lo      = req.param_lo;
    res.snapshot.range_hi      = req.param_hi;

    const std::string& OUT_FILE_PATH = req.csv_output_path;
    constexpr int set_precision = 15;
    if (!OUT_FILE_PATH.empty()) {
        std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
        data_export::write_bif1d_config(cfg, res.snapshot);
        std::ofstream trunc(OUT_FILE_PATH);   // обнуляем основной файл данных
        trunc.close();
    }
    std::ofstream out;
    if (!OUT_FILE_PATH.empty()) {
        out.open(OUT_FILE_PATH, std::ios::app);
        if (out.is_open()) out << std::setprecision(set_precision);
    }

    std::vector<numb> x((size_t)req.amountOfX);
    std::vector<numb> a(req.base_values.size());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> peaks((size_t)maxPointsInBlock);
    std::vector<numb> times((size_t)maxPointsInBlock);

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          /*reverse*/ false, req.log_scale);

        // Каждая точка независима — состояние и параметры восстанавливаем.
        x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
        a.assign(req.base_values.begin(), req.base_values.end());

        numb h_local = (numb)req.h;
        if (req.sweep_over_h)        h_local = (numb)p;
        else if (req.sweep_over_var) x[(size_t)req.var_sweep_index] = (numb)p;
        else                         a[(size_t)req.param_index]     = (numb)p;

        int pointsInBlock = (h_local > (numb)0)
            ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, (double)h_local);
        // Шаг между записанными сэмплами: h*preScaller. Передаём его в
        // peak-finder напрямую, поэтому поправка time_scale из run_bif1d
        // (там peakFinderCUDA получает один h на весь запуск) здесь не нужна.
        const numb timeStep = h_local * (numb)req.pre_scaller;

        auto finish_point = [&](int flag) {
            res.flags[j] = flag;
            if (out.is_open())
                data_export::write_bif1d_rows(out, p, flag, nullptr, nullptr);
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
        };

        // Вырожденный шаг — траектории нет: unbound, а не fixed point.
        if (h_local <= (numb)0 || pointsInBlock <= 0) { finish_point(REGIME_UNBOUND); continue; }

        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        // Ровно как в kernel'е: основной участок считается только при
        // OSCILLATION и FIXED_POINT.
        if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT)
            flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsInBlock, req.amountOfX,
                                  req.pre_scaller, req.writable_var,
                                  req.max_value, block.data());
        // Same reasoning as above: a map's period-1 orbit is a branch, not a blank.
        if (flag != REGIME_OSCILLATION &&
            !(req.emit_all_samples && flag == REGIME_FIXED_POINT)) { finish_point(flag); continue; }

        const int n = cpu_peak_finder(block.data(), (size_t)pointsInBlock,
                                      peaks.data(), times.data(), timeStep,
                                      req.emit_all_samples);
        res.flags[j] = n;
        if (n > 0) {
            res.bifurcation_points[j].assign(peaks.begin(), peaks.begin() + n);
            res.peak_times[j].assign        (times.begin(), times.begin() + n);
        }
        if (out.is_open())
            data_export::write_bif1d_rows(out, p, n,
                                          res.bifurcation_points[j].data(),
                                          res.peak_times[j].data());
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    if (out.is_open()) out.close();
    res.ok = true;
    return res;
}

std::string exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    std::string p(buf, n);
    auto pos = p.find_last_of("\\/");
    return (pos == std::string::npos) ? std::string(".") : p.substr(0, pos);
}

std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "failed to open " + path; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    // UTF-8 BOM: 0xEF 0xBB 0xBF в начале — NVRTC от него спотыкается.
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        s.erase(0, 3);
    }
    // NVRTC не любит не-ASCII символы даже внутри комментариев (компилит как
    // device code, не предполагает многобайтовых юникод-точек). Заменяем все
    // байты со старшим битом на пробел — это безвредно для комментариев и
    // не задевает строковые литералы из ASCII.
    for (char& c : s) {
        if ((unsigned char)c >= 0x80) c = ' ';
    }
    return s;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string hash_key(const std::string& krs_body, int amountOfX) {
    // peak_config_epoch(): knobs configCUDA.h уходят в NVRTC как #define, т.е. при их смене тот
    // же КРС даёт другой PTX. Без epoch в ключе кэши модулей отдали бы старый модуль, и настройка
    // не применилась бы до перезапуска приложения.
    // fmad — то же самое, но через опцию компиляции. Здесь в ключ идёт САМО значение, а не счётчик
    // поколений: флаг бинарный, поэтому возврат к прежнему положению переиспользует уже собранный
    // модуль вместо лишней перекомпиляции.
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" +
           std::to_string(amountOfX) + ":pk" + std::to_string(peak_config_epoch()) +
           ":fm" + (get_nvrtc_fmad() ? "1" : "0") +
           ":rdc" + (get_nvrtc_rdc() ? "1" : "0");
}

// Блок #define'ов, дописываемый ПЕРЕД текстом виртуального configCUDA.h (тот оборачивает свои
// дефолты в #ifndef). Один инжект покрывает все 14 шаблонов — они тянут configCUDA.h транзитивно
// через cudaLibrary.cuh. Приведение к (numb) внутри макроса безопасно: разворачивается он только
// в местах использования (cudaLibrary.cu), т.е. заведомо после typedef numb.
std::string peak_config_defines() {
    const PeakConfig c = get_peak_config();
    std::ostringstream o;
    o << std::setprecision(17);
    o << "// --- injected by parametric_engine (GUI Settings) ---\n"
      << "#define doCalculatePeaks "    << (c.do_calculate_peaks   ? 1 : 0) << "\n"
      << "#define doInterpolatePeaks "  << (c.do_interpolate_peaks ? 1 : 0) << "\n"
      << "#define eps_fixed_point ((numb)"     << c.eps_fixed_point     << ")\n"
      << "#define eps_peak_delta ((numb)"      << c.eps_peak_delta      << ")\n"
      << "#define eps_interPeak_delta ((numb)" << c.eps_interPeak_delta << ")\n"
      << "#define peak_threshold ((numb)"      << c.peak_threshold      << ")\n"
      << "#define max_amount_of_peaks "        << c.max_amount_of_peaks << "\n";
    return o.str();
}

std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}

// Локальная host-копия __device__ __host__ getValueByIdx из cudaLibrary.cu:1266.
// Для 1D-бифуркации valueNumber всегда 0, поэтому просто линейная интерполяция.
// Нужна для расчёта значения параметра в CSV (host-side).
inline double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    // Одна арифметика с ядром — ucuda_node_value (configCUDA.h). Раньше здесь стояло
    // lo + (hi-lo)*idx/(n-1): при idx = nPts-1 это не давало РОВНО hi, и CSV сообщал параметр, в
    // котором точка не считалась. Вырожденный nPts == 1 теперь тоже как в ядре (hi, а не lo).
    // Касты явные: при numb = float сетка обязана считаться во float — ровно так, как её увидит
    // ядро (ranges уходят туда через to_numb).
    return (double)ucuda_node_value((int)idx, nPts, (numb)lo, (numb)hi);
}

// Host-копия getValueByIdx_log из cudaLibrary.cu -- та же формула, что кернел
// реально использовал для этой точки при log_scale=true. Нужна, чтобы CSV
// (см. getValueByIdx_local) сообщал то самое значение параметра, что было
// просимулировано, а не линейную интерполяцию поверх log-распределённой сетки.
inline double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) {
    // См. getValueByIdx_local: общая формула вместо копии. Здесь копия ещё и
    // умножала до деления, а ядро (getValueByIdxLog) — наоборот.
    return (double)ucuda_node_value_log((int)idx, nPts, (numb)lo, (numb)hi);
}

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
// перенормировка каждые NT единиц времени, lambda = sum(log(|dX|/eps)) / tMax.
LLE1DResult run_lle1d_cpu(const LLE1DRequest& req, bool continuation) {
    LLE1DResult res;
    auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body is empty");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("too many base_values");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index out of range");
    if (req.n_pts <= 0)         return fail("n_pts must be > 0");
    if (req.h <= 0.0)           return fail("h must be > 0");
    if (req.t_max <= 0.0)       return fail("t_max must be > 0");
    if (req.transient_time < 0) return fail("transient_time must be >= 0");
    if (req.NT <= 0.0)          return fail("NT must be > 0");
    if (req.eps <= 0.0)         return fail("eps must be > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const int N            = req.amountOfX;
    const int nBlocks      = (int)(req.t_max / req.NT);   // NT-блоков; от h не зависит
    const int settleBlocks = steps_from_time_int(req.transient_time, req.NT);
    if (nBlocks <= 0) return fail("computed t_max / NT <= 0");
    // Число шагов в NT-блоке и в прогреве зависит от h, а при h-свипе h своё в
    // каждой точке — считаем их внутри цикла.
    int ntSteps   = (int)(req.NT / req.h);
    size_t skipSteps = steps_from_time_size_t(req.transient_time, req.h);
    if (!req.sweep_over_h && ntSteps <= 0) return fail("computed NT / h <= 0");

    const int nPts = req.n_pts;
    res.n_pts    = nPts;
    res.param_lo = req.param_lo;
    res.param_hi = req.param_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.lyapunov.assign(nPts, std::numeric_limits<double>::quiet_NaN());
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> y((size_t)N, (numb)0);

    // Случайное единичное направление щупа. subsequence = индекс точки: в
    // kernel'е это idx потока, поэтому классический CPU-свип получает те же
    // направления, что и GPU. В continuation цепочка одна на весь свип, и
    // subsequence используется только при разрыве.
    CpuRandState rng;
    auto seed_probe_direction = [&](unsigned long long subsequence) {
        cpu_curand_init(1234567891ULL, subsequence, 0ULL, &rng);
        double zPower = 0.0;
        for (int i = 0; i < N; ++i) { y[i] = cpu_curand_uniform(&rng) - 0.5; zPower += y[i] * y[i]; }
        zPower = std::sqrt(zPower);
        for (int i = 0; i < N; ++i) y[i] = (zPower == 0.0) ? 0.0 : y[i] / zPower;
    };
    seed_probe_direction(0ULL);

    bool probe_attached = false;   // true => y[] уже в абсолютных координатах
    numb h_local = (numb)req.h;    // при h-свипе меняется от точки к точке

    // Один NT-блок: продвинуть x и щуп, вернуть log(|dX|/eps) и вернуть щуп на
    // расстояние eps. Точная копия тела цикла LLEKernelCUDA.
    auto advance_block = [&](numb& out_log) -> bool {
        if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;
        if (cpu_loop_model(step.fn(), y.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;

        double d = 0.0;
        for (int l = 0; l < N; ++l) {
            const double t = (x[l] - y[l]) / req.eps;
            d += t * t;
        }
        d = std::sqrt(d);
        if (d <= 1e-14) d = 1e-14;
        out_log = std::log(d);

        const double inv = 1.0 / d;
        for (int l = 0; l < N; ++l)
            y[l] = x[l] - ((x[l] - y[l] + 1e-14) * inv);
        return true;
    };

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        // Свипуемая величина. Классика: направление и log берутся из запроса,
        // reverse не применяется (точки независимы).
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);
        if (req.sweep_over_h) {
            // Свипуем сам шаг: a[] не трогаем, пересчитываем число шагов.
            h_local   = p;
            ntSteps   = (h_local > 0.0) ? (int)(req.NT / h_local) : 0;
            skipSteps = steps_from_time_size_t(req.transient_time, h_local);
        } else {
            a[(size_t)req.param_index] = p;
        }

        // В классическом режиме состояние не переносится: сбрасываем всё, как
        // делает kernel в начале каждого потока.
        if (!continuation) {
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            seed_probe_direction((unsigned long long)j);
            probe_attached = false;
        }

        if (h_local <= 0.0 || ntSteps <= 0) {
            res.flags[j] = REGIME_UNBOUND;     // вырожденный шаг — точки нет
            if (continuation) probe_attached = false;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        bool alive = true;
        if (!probe_attached) {
            // Первая точка (и любая после разрыва цепочки): выходим на аттрактор
            // одной траекторией, как классический свип, затем цепляем щуп.
            if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, skipSteps, N,
                               1, 0, req.max_value, nullptr) == 0) {
                alive = false;
            } else {
                for (int i = 0; i < N; ++i) y[i] = y[i] * req.eps + x[i];
                probe_attached = true;
            }
        } else {
            numb dummy;
            for (int b = 0; b < settleBlocks && alive; ++b)
                if (!advance_block(dummy)) alive = false;
        }

        numb sum = (numb)0;
        for (int b = 0; alive && b < nBlocks; ++b) {
            numb lg;
            if (!advance_block(lg)) alive = false;
            else                    sum += lg;
        }

        if (alive) {
            res.lyapunov[j] = (double)(sum / (numb)req.t_max);
            res.flags[j]    = REGIME_OSCILLATION;
        } else {
            // LLE не различает fixed point: единственная причина «не посчиталось»
            // — расходимость. lyapunov[j] остаётся NaN.
            res.flags[j] = REGIME_UNBOUND;
            // Continuation: цепочку рвём — сбрасываем траекторию на IC и берём
            // новое направление щупа, чтобы следующая точка стартовала с
            // чистого листа, а не с разошедшегося состояния.
            if (continuation) {
                probe_attached = false;
                x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
                seed_probe_direction((unsigned long long)j + 1ULL);
            }
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}

// Оконная функция для DFT. Все варианты — косинусные суммы
//   w(n) = a0 - a1*cos(g) + a2*cos(2g) - a3*cos(3g),   g = 2*pi*n/(N-1),
// так что таблицы коэффициентов хватает на все типы:
//   0 = None (rectangular)
//   1 = Hanning (default)        -31 дБ, спад -18 дБ/окт
//   2 = Hamming                  -43 дБ, спад  -6 дБ/окт
//   3 = Blackman                 -58 дБ, спад -18 дБ/окт
//   4 = Blackman-Harris (4 чл.)  -92 дБ, спад  -6 дБ/окт
// Уровень боковых лепестков — это и есть динамический диапазон: на Hanning'е
// субгармоники слабее -31 дБ тонут в утечке от основной частоты, что в каскаде
// удвоений периода отрезает всё после третьего-четвёртого удвоения.
//
// Окно нормируется на единичное среднее. Все четыре пути DFT (DFT_custom,
// cpu_dft_block и оба template-ядра) делят сумму на длину блока, а не на
// sum(w), поэтому без нормировки абсолютная амплитуда спектра зависела бы от
// выбора окна — когерентное усиление тут 1.0 / 0.5 / 0.54 / 0.42 / 0.36.
// Форма спектра от нормировки не зависит.
//
// Дублируется в dft_window (dft1d_cont.template.cu) и dft_hsweep_window
// (dft1d_hsweep.template.cu): при h-свипе и в continuation-режиме длина блока
// своя в каждой точке, и окно приходится строить прямо на GPU.
void cpu_build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) {
    out.resize((size_t)sizeOfBlock);
    if (window_type <= 0 || window_type > 4 || sizeOfBlock < 2) {
        std::fill(out.begin(), out.end(), (numb)1);
        return;
    }
    static const double kCoef[5][4] = {
        { 1.0,     0.0,     0.0,     0.0     },   // 0 = None (сюда не доходим)
        { 0.5,     0.5,     0.0,     0.0     },   // 1 = Hanning
        { 0.53836, 0.46164, 0.0,     0.0     },   // 2 = Hamming
        { 0.42,    0.5,     0.08,    0.0     },   // 3 = Blackman
        { 0.35875, 0.48829, 0.14128, 0.01168 },   // 4 = Blackman-Harris
    };
    const double* c = kCoef[window_type];
    const numb gamma = (numb)2.0 * (numb)kPi / (numb)(sizeOfBlock - 1);
    numb sum = (numb)0;
    for (int n = 0; n < sizeOfBlock; ++n) {
        // cos(2g) и cos(3g) через кратные углы, а не тремя вызовами cos:
        // так же считает GPU-версия, иначе пути разошлись бы в последнем бите.
        const numb c1 = std::cos(gamma * (numb)n);
        const numb c2 = (numb)2.0 * c1 * c1 - (numb)1.0;
        const numb c3 = ((numb)4.0 * c1 * c1 - (numb)3.0) * c1;
        const numb w  = (numb)c[0] - (numb)c[1] * c1 + (numb)c[2] * c2 - (numb)c[3] * c3;
        out[(size_t)n] = w;
        sum += w;
    }
    const numb mean = sum / (numb)sizeOfBlock;
    if (mean > (numb)0)
        for (int n = 0; n < sizeOfBlock; ++n) out[(size_t)n] /= mean;
}

// Порт DFT_custom (cudaLibrary.cu) на один блок. Рекуррентный поворот вектора (cos_n, sin_n)
// вместо cos/sin на каждой итерации, со сбросом на точное значение раз в RESET_INTERVAL — как на
// GPU, включая float-точность в самом сбросе (там cosf/sinf).
void cpu_dft_block(const numb* data, int sizeOfBlock, const numb* window,
                   int nFreq, numb freq_lo, numb freq_hi, bool logFreqAxis,
                   numb h, numb* akcos, numb* bksin)
{
    constexpr int RESET_INTERVAL = 1000;
    const numb f_step = (nFreq > 1) ? (freq_hi - freq_lo) / (numb)(nFreq - 1) : (numb)0;
    const numb psi    = (numb)2.0 * (numb)kPi * h;

    for (int k = 0; k < nFreq; ++k) {
        numb ak = (numb)0, bk = (numb)0;
        const numb f_k = logFreqAxis
            ? std::pow((numb)10.0, std::log10(freq_lo) + (nFreq > 1
                  ? (std::log10(freq_hi) - std::log10(freq_lo)) * (numb)k / (numb)(nFreq - 1) : (numb)0))
            : (freq_lo + (numb)k * f_step);
        const numb cos_theta = std::cos((numb)2.0 * (numb)kPi * h * f_k);
        const numb sin_theta = std::sin((numb)2.0 * (numb)kPi * h * f_k);
        numb cos_n = (numb)1, sin_n = (numb)0;
        int reset_counter = RESET_INTERVAL;

        for (int n = 0; n < sizeOfBlock; ++n) {
            const numb wd = window[n] * data[n];
            ak += wd * cos_n;
            bk += wd * sin_n;
            const numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
            const numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
            cos_n = new_cos;
            sin_n = new_sin;
            if (--reset_counter == 0) {
                reset_counter = RESET_INTERVAL;
                if (n + 1 < sizeOfBlock) {
                    const numb exact = psi * f_k * (numb)(n + 1);
                    // cosf/sinf, как в kernel'е: сброс идёт через float.
                    cos_n = (numb)std::cos((float)exact);
                    sin_n = (numb)std::sin((float)exact);
                }
            }
        }
        akcos[k] = ak / (numb)sizeOfBlock;
        bksin[k] = bk / (numb)sizeOfBlock;
    }
}

// Порты projectionOperator и gramSchmidtProcess (cudaLibrary.cu:2832 и :2848).
// Оригиналы объявлены __device__ __host__ и уже работают с runtime-размерностью,
// но живут в .cu под nvcc — из обычного .cpp их не подключить, поэтому копия.
void cpu_projection_operator(const numb* a, const numb* b, numb* minuend, int n) {
    numb numerator = (numb)0, denominator = (numb)0;
    for (int i = 0; i < n; ++i) {
        numerator   += a[i] * b[i];
        denominator += b[i] * b[i];
    }
    const numb fraction = (denominator == (numb)0) ? (numb)0 : numerator / denominator;
    for (int i = 0; i < n; ++i) minuend[i] -= fraction * b[i];
}

// a — входные векторы (n штук по n компонент, row-major), b — ортонормированный
// результат. denominators (опц.) получает длины ДО нормировки — из них LS и
// собирает показатели.
void cpu_gram_schmidt(const numb* a, numb* b, int n, numb* denominators = nullptr) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) b[j + i * n] = a[j + i * n];
        for (int j = 0; j < i; ++j)
            cpu_projection_operator(a + i * n, b + j * n, b + i * n, n);
    }
    for (int i = 0; i < n; ++i) {
        numb denominator = (numb)0;
        for (int j = 0; j < n; ++j) denominator += b[i * n + j] * b[i * n + j];
        denominator = std::sqrt(denominator);
        for (int j = 0; j < n; ++j)
            b[i * n + j] = (denominator == (numb)0) ? (numb)0 : b[i * n + j] / denominator;
        if (denominators != nullptr) denominators[i] = denominator;
    }
}

// run_ls1d_cpu — спектр Ляпунова LS(param) на CPU, оба режима (см. run_lle1d_cpu).
// Внутри точки — построчный порт LSKernelCUDA: Бенеттин с N векторами возмущения и
// ортогонализацией Грама-Шмидта каждые NT единиц времени,
// lambda_k = sum(log(denominators[k]/eps)) / tMax.
// В continuation переносятся траектория x[] и ВСЕ N щупов y[] — к концу точки они уже выстроены
// вдоль собственных направлений растяжения. Отличие от LLE только в размере состояния (там один
// щуп, здесь N плюс ортогонализация), поэтому точка дороже примерно в N раз.
LS1DResult run_ls1d_cpu(const LS1DRequest& req, bool continuation) {
    LS1DResult res;
    auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body is empty");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("too many base_values");
    if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                              return fail("param_index out of range");
    if (req.n_pts <= 0)         return fail("n_pts must be > 0");
    if (req.h <= 0.0)           return fail("h must be > 0");
    if (req.t_max <= 0.0)       return fail("t_max must be > 0");
    if (req.transient_time < 0) return fail("transient_time must be >= 0");
    if (req.NT <= 0.0)          return fail("NT must be > 0");
    if (req.eps <= 0.0)         return fail("eps must be > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const int N            = req.amountOfX;
    const int nBlocks      = (int)(req.t_max / req.NT);
    const int settleBlocks = steps_from_time_int(req.transient_time, req.NT);
    if (nBlocks <= 0) return fail("computed t_max / NT <= 0");
    // См. run_lle1d_cpu: при h-свипе шаг свой в каждой точке.
    int ntSteps   = (int)(req.NT / req.h);
    size_t skipSteps = steps_from_time_size_t(req.transient_time, req.h);
    if (!req.sweep_over_h && ntSteps <= 0) return fail("computed NT / h <= 0");

    const int nPts = req.n_pts;
    res.n_pts       = nPts;
    res.n_exponents = N;
    res.param_lo    = req.param_lo;
    res.param_hi    = req.param_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.spectrum.assign(nPts, std::vector<double>((size_t)N,
                        std::numeric_limits<double>::quiet_NaN()));
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> y((size_t)N * N, (numb)0);   // N щупов, абсолютные координаты
    std::vector<numb> z((size_t)N * N, (numb)0);   // рабочий буфер Грама-Шмидта
    std::vector<numb> denominators((size_t)N, (numb)0);
    std::vector<numb> sum((size_t)N, (numb)0);

    CpuRandState rng;
    // Случайный базис из N векторов — та же процедура и тот же RNG, что в
    // kernel'е (subsequence = idx потока).
    auto seed_probe_basis = [&](unsigned long long subsequence) {
        cpu_curand_init(1234567891ULL, subsequence, 0ULL, &rng);
        for (int j = 0; j < N; ++j) {
            double zPower = 0.0;
            for (int i = 0; i < N; ++i) {
                y[(size_t)j * N + i] = cpu_curand_uniform(&rng) - 0.5;
                zPower += y[(size_t)j * N + i] * y[(size_t)j * N + i];
            }
            zPower = std::sqrt(zPower);
            for (int i = 0; i < N; ++i)
                y[(size_t)j * N + i] = (zPower == 0.0) ? 0.0 : y[(size_t)j * N + i] / zPower;
        }
    };
    seed_probe_basis(0ULL);

    bool probes_attached = false;
    numb h_local = (numb)req.h;

    // Один NT-блок: продвинуть траекторию и все щупы, ортогонализовать, вернуть
    // log(denominators[k]/eps) по каждому направлению и вернуть щупы на eps.
    // Точная копия тела цикла LSKernelCUDA.
    auto advance_block = [&](numb* out_logs) -> bool {
        if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, ntSteps, N,
                           1, 0, req.max_value, nullptr) == 0) return false;
        for (int j = 0; j < N; ++j)
            if (cpu_loop_model(step.fn(), y.data() + (size_t)j * N, a.data(), h_local,
                               ntSteps, N, 1, 0, req.max_value, nullptr) == 0) return false;

        for (int k = 0; k < N; ++k)
            for (int l = 0; l < N; ++l)
                y[(size_t)k * N + l] -= x[l];

        cpu_gram_schmidt(y.data(), z.data(), N, denominators.data());

        for (int k = 0; k < N; ++k) {
            if (out_logs) out_logs[k] = std::log(denominators[k] / req.eps);
            for (int j = 0; j < N; ++j)
                y[(size_t)k * N + j] = x[j] + z[(size_t)k * N + j] * req.eps;
        }
        return true;
    };

    std::vector<numb> logs((size_t)N, (numb)0);

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);
        if (req.sweep_over_h) {
            h_local   = p;
            ntSteps   = (h_local > 0.0) ? (int)(req.NT / h_local) : 0;
            skipSteps = steps_from_time_size_t(req.transient_time, h_local);
        } else {
            a[(size_t)req.param_index] = p;
        }

        if (!continuation) {
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            seed_probe_basis((unsigned long long)j);
            probes_attached = false;
        }

        if (h_local <= 0.0 || ntSteps <= 0) {
            res.flags[j] = REGIME_UNBOUND;     // вырожденный шаг — точки нет
            if (continuation) probes_attached = false;
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
            continue;
        }

        bool alive = true;
        if (!probes_attached) {
            if (cpu_loop_model(step.fn(), x.data(), a.data(), h_local, skipSteps, N,
                               1, 0, req.max_value, nullptr) == 0) {
                alive = false;
            } else {
                // z = ортонормированный базис из y, затем щупы ставятся на eps
                // вокруг текущей точки траектории (порядок как в kernel'е).
                cpu_gram_schmidt(y.data(), z.data(), N);
                for (int k = 0; k < N; ++k)
                    for (int i = 0; i < N; ++i)
                        y[(size_t)k * N + i] = z[(size_t)k * N + i] * req.eps + x[i];
                probes_attached = true;
            }
        } else {
            for (int b = 0; b < settleBlocks && alive; ++b)
                if (!advance_block(nullptr)) alive = false;
        }

        std::fill(sum.begin(), sum.end(), 0.0);
        for (int b = 0; alive && b < nBlocks; ++b) {
            if (!advance_block(logs.data())) { alive = false; break; }
            for (int k = 0; k < N; ++k) sum[k] += logs[k];
        }

        if (alive) {
            for (int k = 0; k < N; ++k) res.spectrum[j][k] = (double)(sum[k] / (numb)req.t_max);
            res.flags[j] = REGIME_OSCILLATION;
        } else {
            // Как и LLE, LS не различает fixed point — только расходимость.
            res.flags[j] = REGIME_UNBOUND;      // spectrum[j] остаётся NaN
            if (continuation) {
                probes_attached = false;
                x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
                seed_probe_basis((unsigned long long)j + 1ULL);
            }
        }
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}

// run_dft1d_cpu — 1D DFT на CPU, оба режима (см. run_lle1d_cpu).
// continuation = false: точки независимы, x сбрасывается на initial_conditions (при IC-свипе
//   свипуемая координата подменяется). continuation = true: x[] переносится с предыдущей точки,
//   как в Bifurcation1D — тангенциальных векторов здесь нет.
// Внутри точки: transient, запись блока writable_var, окно, DFT (cpu_dft_block — порт
// DFT_custom). При h-свипе число сэмплов и окно пересчитываются под шаг точки; буфер выделен под
// худший случай.
Dft1DResult run_dft1d_cpu(const Dft1DRequest& req, bool continuation) {
    Dft1DResult res;
    auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

    if (req.krs_body.empty())                                 return fail("krs_body is empty");
    if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
    if ((int)req.initial_conditions.size() != req.amountOfX)  return fail("initial_conditions.size() != amountOfX");
    if ((int)req.base_values.size() > kMaxAmountOfValues)     return fail("too many base_values");
    if (!req.sweep_over_h && !req.sweep_over_var &&
        (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
                                                              return fail("param_index out of range");
    if (req.sweep_over_var &&
        (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX))
                                                              return fail("var_sweep_index out of range");
    if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                              return fail("writable_var out of range");
    if (req.n_pts <= 0)         return fail("n_pts must be > 0");
    if (req.n_freq <= 0)        return fail("n_freq must be > 0");
    if (req.h <= 0.0)           return fail("h must be > 0");
    if (req.t_max <= 0.0)       return fail("t_max must be > 0");
    if (req.transient_time < 0) return fail("transient_time must be >= 0");
    if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
    if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
                                return fail("a log grid over frequency requires freq lo/hi > 0");

    KrsCpuStep step;
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(req.krs_body, req.amountOfX, (int)req.base_values.size(), diags)) {
        std::string msg = "CPU KRS:";
        for (const auto& d : diags) {
            msg += "\n";
            if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
            msg += d.message;
        }
        return fail(msg);
    }

    const double worstCaseH = req.sweep_over_h
                            ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                            : req.h;
    if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
    const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
    if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

    const int nPts  = req.n_pts;
    const int nFreq = req.n_freq;
    res.n_pts   = nPts;
    res.n_freq  = nFreq;
    res.param_lo = req.param_lo;
    res.param_hi = req.param_hi;
    res.freq_lo  = req.freq_lo;
    res.freq_hi  = req.freq_hi;
    res.continuation_reverse = continuation && req.continuation_reverse;
    res.ak_cos.assign((size_t)nPts * nFreq, 0.0);
    res.bk_sin.assign((size_t)nPts * nFreq, 0.0);
    res.flags.assign(nPts, 0);

    std::vector<numb> x(req.initial_conditions.begin(), req.initial_conditions.end());
    std::vector<numb> a(req.base_values.begin(), req.base_values.end());
    std::vector<numb> block((size_t)maxPointsInBlock);
    std::vector<numb> window;
    std::vector<numb> ak_tmp((size_t)nFreq), bk_tmp((size_t)nFreq);
    int window_len = -1;                 // окно перестраиваем только при смене длины

    numb h_local = (numb)req.h;

    for (int j = 0; j < nPts; ++j) {
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true;
            res.error = "Cancelled by user";
            return res;
        }
        const double p = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                          continuation && req.continuation_reverse,
                                          req.log_scale);

        if (!continuation)
            x.assign(req.initial_conditions.begin(), req.initial_conditions.end());

        if (req.sweep_over_h)        h_local = p;
        else if (req.sweep_over_var) x[(size_t)req.var_sweep_index] = p;
        else                         a[(size_t)req.param_index] = p;

        int pointsInBlock = (h_local > 0.0) ? (int)(req.t_max / h_local / req.pre_scaller) : 0;
        if (pointsInBlock > maxPointsInBlock) pointsInBlock = maxPointsInBlock;
        const size_t pointsForSkip = steps_from_time_size_t(req.transient_time, h_local);

        auto mark_dead = [&](int flag) {
            res.flags[j] = flag;
            // Как DFT_custom: FIXED_POINT -> -1.0, UNBOUND -> 0.0.
            const double fill = (flag == REGIME_FIXED_POINT) ? -1.0 : 0.0;
            for (int k = 0; k < nFreq; ++k) {
                res.ak_cos[(size_t)j * nFreq + k] = fill;
                res.bk_sin[(size_t)j * nFreq + k] = fill;
            }
            if (continuation) x.assign(req.initial_conditions.begin(), req.initial_conditions.end());
            if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
        };

        // Вырожденный шаг — траектории нет: unbound, а не fixed point.
        if (h_local <= 0.0 || pointsInBlock <= 2) { mark_dead(REGIME_UNBOUND); continue; }

        int flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                                  pointsForSkip, req.amountOfX,
                                  /*preScaller*/ 1, /*writableVar*/ 0,
                                  req.max_value, nullptr);
        if (flag == REGIME_UNBOUND) { mark_dead(REGIME_UNBOUND); continue; }

        flag = cpu_loop_model(step.fn(), x.data(), a.data(), h_local,
                              pointsInBlock, req.amountOfX,
                              req.pre_scaller, req.writable_var,
                              req.max_value, block.data());
        // Сырой REGIME_* и для continuation, и для классики: раньше в
        // continuation unbound подменялся на 1 и спектр считался по
        // разошедшемуся блоку (см. dft1d_cont.template.cu — там та же правка).
        if (flag != REGIME_OSCILLATION) { mark_dead(flag); continue; }

        if (window_len != pointsInBlock) {
            cpu_build_window(window, pointsInBlock, req.window_type);
            window_len = pointsInBlock;
        }
        // Считаем в numb (как GPU), затем расширяем в double-хранилище результата.
        cpu_dft_block(block.data(), pointsInBlock, window.data(),
                      nFreq, (numb)req.freq_lo, (numb)req.freq_hi, req.freq_log_scale,
                      h_local * (numb)req.pre_scaller,
                      ak_tmp.data(), bk_tmp.data());
        for (int k = 0; k < nFreq; ++k) {
            res.ak_cos[(size_t)j * nFreq + k] = (double)ak_tmp[(size_t)k];
            res.bk_sin[(size_t)j * nFreq + k] = (double)bk_tmp[(size_t)k];
        }
        res.flags[j] = REGIME_OSCILLATION;
        if (req.progress) req.progress->store((float)(j + 1) / (float)nPts, std::memory_order_relaxed);
    }

    res.ok = true;
    return res;
}

}  // namespace

struct ParametricEngine::Impl {
    bool       inited   = false;
    CUcontext  context  = nullptr;
    CUdevice   device   = 0;
    int        cc_major = 0;
    int        cc_minor = 0;

    // Сигнальная страница расчёта: [0] — запрос отмены, [1] — тики прогресса.
    //
    // Mapped host memory, а НЕ память устройства. На WDDM любой cudaMemcpy,
    // выданный пока работает ядро, встаёт за ним в очередь и блокирует хост:
    // цикл опроса делал ровно одну итерацию, и ни прогресс, ни Cancel не
    // работали. Через mapped-страницу хост читает и пишет обычными load/store,
    // без единого вызова CUDA.
    struct RunSignals {
        int* host = nullptr;   // страница на хосте
        int* dev  = nullptr;   // её же device-проекция для ядра

        bool alloc(std::string& err) {
            CUresult r = cuMemHostAlloc((void**)&host, 2 * sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP);
            if (r != CUDA_SUCCESS) { err = "cuMemHostAlloc(signals): " + cu_err(r); return false; }
            r = cuMemHostGetDevicePointer((CUdeviceptr*)&dev, host, 0);
            if (r != CUDA_SUCCESS) { err = "cuMemHostGetDevicePointer: " + cu_err(r); release(); return false; }
            host[0] = 0; host[1] = 0;
            return true;
        }
        void release()          { if (host) { cuMemFreeHost(host); host = nullptr; dev = nullptr; } }
        int* cancelArg()  const { return dev; }
        int* progressArg()const { return dev ? dev + 1 : nullptr; }
        void resetTicks()       { if (host) ((volatile int*)host)[1] = 0; }
        int  ticks()      const { return host ? ((volatile int*)host)[1] : 0; }
        void raiseCancel()      { if (host) ((volatile int*)host)[0] = 1; }
    };

    // Шаг тика прогресса. Кратен CHECK_INTERVAL, потому что тики ставятся в уже
    // существующей проверке расходимости, и подобран так, чтобы точка отчиталась
    // около 64 раз за свою работу: этого хватает на гладкий бар и не создаёт
    // давки на одном адресе.
    static int progress_stride_for(size_t stepsPerPoint) {
        int s = (int)(stepsPerPoint / 64);
        s -= s % kCheckInterval;
        if (s < kCheckInterval) s = kCheckInterval;
        return s;
    }

    // Ждём, пока отданная в stream работа закончится, обновляя прогресс из тиков
    // и передавая Cancel в ядро. В теле цикла не должно быть вызовов CUDA, кроме
    // cudaStreamQuery — см. комментарий к RunSignals.
    bool wait_with_signals(cudaStream_t stream, RunSignals& sig,
                           const std::shared_ptr<std::atomic<bool>>& cancel,
                           const std::shared_ptr<std::atomic<float>>& progress,
                           double ticksBefore, double ticksTotal, std::string& err) const
    {
        bool cancelSent = false;
        // Публикуем долю и ДО первого опроса, и ПЕРЕД выходом. Раньше обновление стояло
        // только в середине цикла, между «не готово» и сном, из-за чего:
        //   - значение, выставленное вызывающим перед запуском, висело до первого сна,
        //   - а если stream успевал закончиться до первого опроса, цикл выходил по
        //     `break` вообще ни разу не обновив прогресс, и на баре до конца чанка
        //     оставалось предыдущее значение.
        // Пока чанк считался секундами, оба окна были незаметны; после ускорения dbscan
        // (в 20-100 раз) чанки стали короткими, и стыки полезли наружу.
        auto publish = [&]() {
            if (!progress || ticksTotal <= 0.0) return;
            double f = (ticksBefore + (double)sig.ticks()) / ticksTotal;
            if (f > 1.0) f = 1.0;
            progress->store((float)f, std::memory_order_relaxed);
        };
        for (;;) {
            publish();
            cudaError_t q = cudaStreamQuery(stream);
            if (q == cudaSuccess) { publish(); break; }
            if (q != cudaErrorNotReady) {
                err = std::string("CUDA stream query: ") + cudaGetErrorString(q);
                return false;
            }
            if (!cancelSent && cancel && cancel->load(std::memory_order_relaxed)) {
                sig.raiseCancel();
                cancelSent = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        return true;
    }

    // Закэшированные тексты NonLinAnal headers (читаются один раз)
    std::string src_cudaLibrary_cu;
    std::string src_cudaLibrary_cuh;
    std::string src_cudaMacros_cuh;
    // curand_kernel.h перехвачен inline-stub'ом в каждом template'е (kernels/*.cu), а
    // `#define CURAND_KERNEL_H_` блокирует реальный header; virtual header'а здесь больше нет —
    // иначе он повторно объявлял бы curandState_t.
    // src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw: сырой текст держим отдельно,
    // чтобы пересборка при смене Settings не лезла на диск.
    std::string src_configCUDA_h_raw;
    std::string src_configCUDA_h;
    std::string src_template;        // bifurcation1d.template.cu
    std::string src_template_cont;       // bifurcation1d_cont.template.cu
    std::string src_template_lle_cont;   // lle1d_cont.template.cu
    std::string src_template_ls_cont;    // ls1d_cont.template.cu
    std::string src_template_dft_cont;   // dft1d_cont.template.cu
    std::string src_template_dft_hsweep; // dft1d_hsweep.template.cu
    std::string src_template_lle;    // lle1d.template.cu
    std::string src_template_lle_2d; // lle2d.template.cu
    std::string src_template_ls;     // ls1d.template.cu
    std::string src_template_ls_2d;  // ls2d.template.cu
    std::string src_template_bif2d;  // bifurcation2d.template.cu
    std::string src_template_basins; // basins.template.cu
    std::string src_template_fs_attr; // fastsync_attr.template.cu (mode 0)
    std::string src_template_fs_grid; // fastsync_grid.template.cu (mode 1)
    std::string src_template_order;   // order.template.cu
    std::string src_template_network; // network.template.cu
    bool     srcs_loaded     = false;
    uint64_t srcs_peak_epoch = 0;   // != peak_config_epoch() -> пересобрать configCUDA.h

    struct CachedModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_traj = nullptr;  // calculateDiscreteModelCUDA
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
        CUfunction  kernel_fused = nullptr; // calculateDiscreteModelPeaksCUDA
        CUfunction  kernel_dft  = nullptr;  // DFT_custom (used by run_dft_1d classical branch)
    };
    CachedModule cached;          // bif1d kernels (traj + peak + dft)

    // Отдельный модуль для LLE — другой шаблон, другой kernel.
    struct CachedLleModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_lle  = nullptr;  // LLEKernelCUDA
    };
    CachedLleModule cached_lle;

    // LLE-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLleModule cached_lle_2d;

    // LS — отдельный модуль/кэш (третий слот PTX).
    struct CachedLsModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_ls  = nullptr;   // LSKernelCUDA
    };
    CachedLsModule cached_ls;

    // LS-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLsModule cached_ls_2d;

    // Continuation bif1d — четвёртый слот. Кернел single-thread, плюс
    // peakFinderCUDA из того же модуля (он определён в cudaLibrary.cu).
    struct CachedContModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_cont = nullptr;  // bifurcation1dContinuationKernel
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
        CUfunction  kernel_dft  = nullptr;  // DFT_custom (used by run_dft_1d continuation branch)
    };
    CachedContModule cached_cont;

    // Continuation LLE-1D / LS-1D — те же single-thread ядра по идее, но каждое
    // в своём модуле (у них свой шаблон и свой KRS-инстанс). Peak-finder им не
    // нужен: результат — сразу число (lambda) или N чисел (спектр).
    struct CachedSimpleContModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;
    };
    CachedSimpleContModule cached_lle_cont;   // lle1dContinuationKernel
    CachedSimpleContModule cached_ls_cont;    // ls1dContinuationKernel
    CachedSimpleContModule cached_dft_cont;   // dft1dContinuationKernel
    // Классический (без continuation) DFT-свип по h — тоже одноядерный модуль,
    // но ядро параллельное: поток на точку (см. dft1d_hsweep.template.cu).
    CachedSimpleContModule cached_dft_hsweep; // dft1dHSweepKernel

    // Bif-2D — отдельный шаблон (bifurcation2d.template.cu), три kernel'а:
    // calculateDiscreteModelPeaksCUDA + dbscanCUDA. Траектория больше не хранится,
    // поэтому traj и peak — одно ядро (см. PeakStream в cudaLibrary.cu).
    struct CachedBif2dModule {
        std::string  key;
        CUmodule     module        = nullptr;
        CUfunction   kernel_fused  = nullptr;   // calculateDiscreteModelPeaksCUDA
        CUfunction   kernel_dbscan = nullptr;   // dbscanCUDA
    };
    CachedBif2dModule cached_bif2d;

    // Basins — отдельный шаблон, пять kernel'ов (трое для DBSCAN host-цикла).
    struct CachedBasinsModule {
        std::string  key;
        CUmodule     module                  = nullptr;
        CUfunction   kernel_fused            = nullptr;  // calculateDiscreteModelAvgPeaksCUDA
        CUfunction   kernel_dbscan           = nullptr;  // CUDA_dbscan_kernel
        CUfunction   kernel_search_fixed     = nullptr;  // CUDA_dbscan_search_fixed_points_kernel
        CUfunction   kernel_search_clear     = nullptr;  // CUDA_dbscan_search_clear_points_kernel
    };
    CachedBasinsModule cached_basins;

    // Fast Synchro — два модуля (mode 0 = on attractor, mode 1 = on grid).
    // Каждый кэширует свой PTX, ключ = hash(krs_body)+amountOfX+":fs_attr"/":fs_grid"
    // + (type_of_synch, error_estim, fs_error_trs) — все три substituted в #define
    // перед include configCUDA.h, поэтому смена любого требует recompile.
    struct CachedFastSyncModule {
        std::string  key;
        CUmodule     module          = nullptr;
        // FS-specific. НЕ используем calculateDiscreteModelCUDA (та пишет
        // скалярную сводку, не полный X[]).
        CUfunction   kernel_fs_fill  = nullptr;  // fillFSMasterTrajectory (template, mode 0)
        CUfunction   kernel_fs_traj  = nullptr;  // calculateDiscreteModelforFastSynchroCUDA (mode 0)
        CUfunction   kernel_fs_grid  = nullptr;  // calculateDiscreteModelICCforFastSynchro (mode 1)
    };
    CachedFastSyncModule cached_fs_attr;
    CachedFastSyncModule cached_fs_grid;

    // Order — одно ядро, но ключ кэша включает ещё и РАЗМЕР a[]: шаблон
    // объявляет локальный numb a[AMOUNTOFVALUES], а добавление системе
    // неиспользуемого в правых частях параметра (именно так задаются
    // параметры метода для кастомных КРС) krs_body не меняет.
    struct CachedOrderModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;        // orderEstimateKernel
        CUfunction  kernel_perf = nullptr;   // perfIntegrateKernel (вкладка Performance)
        CUfunction  kernel_stab = nullptr;   // stabilityRegionKernel (области устойчивости)
    };
    CachedOrderModule cached_order;

    // Network — ключ включает размер a[] (шаблон объявляет его константой) и
    // тело связи: оно подставляется в switch внутри ядра, и смена закона
    // связи требует перекомпиляции ровно так же, как смена КРС.
    struct CachedNetworkModule {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;        // networkIntegrateKernel
    };
    CachedNetworkModule cached_network;

    // Every cached_* above is just a view on the active entry of its pool; the pool owns the
    // modules. One slot per analysis type meant recompiling on every switch back to a scheme that
    // had already been built minutes ago -- 6 seconds of NVRTC for nothing on an implicit scheme.
    static constexpr std::size_t kModuleCacheCapacity = 4;

    // Guards the pools and the active slots: background prewarming will make compile_* concurrent.
    std::recursive_mutex cache_mu;

    ModuleLru<CachedModule>            pool_bif1d     { kModuleCacheCapacity };
    ModuleLru<CachedLleModule>         pool_lle       { kModuleCacheCapacity };
    ModuleLru<CachedLleModule>         pool_lle_2d    { kModuleCacheCapacity };
    ModuleLru<CachedLsModule>          pool_ls        { kModuleCacheCapacity };
    ModuleLru<CachedLsModule>          pool_ls_2d     { kModuleCacheCapacity };
    ModuleLru<CachedContModule>        pool_cont      { kModuleCacheCapacity };
    ModuleLru<CachedSimpleContModule>  pool_lle_cont  { kModuleCacheCapacity };
    ModuleLru<CachedSimpleContModule>  pool_ls_cont   { kModuleCacheCapacity };
    ModuleLru<CachedSimpleContModule>  pool_dft_cont  { kModuleCacheCapacity };
    ModuleLru<CachedSimpleContModule>  pool_dft_hsweep{ kModuleCacheCapacity };
    ModuleLru<CachedBif2dModule>       pool_bif2d     { kModuleCacheCapacity };
    ModuleLru<CachedBasinsModule>      pool_basins    { kModuleCacheCapacity };
    ModuleLru<CachedFastSyncModule>    pool_fs_attr   { kModuleCacheCapacity };
    ModuleLru<CachedFastSyncModule>    pool_fs_grid   { kModuleCacheCapacity };
    ModuleLru<CachedOrderModule>       pool_order     { kModuleCacheCapacity };
    ModuleLru<CachedNetworkModule>     pool_network   { kModuleCacheCapacity };

    // Activates the module already built for this key, if the pool still holds it.
    template <class T>
    bool cache_activate(ModuleLru<T>& pool, const std::string& key, T& active) {
        if (active.module && active.key == key) return true;
        return pool.take(key, active);
    }

    // Prewarm path: park the module in the pool WITHOUT touching the active slot -- a running
    // task reads that slot from its own thread and must not see it change under it.
    template <class T>
    void cache_store(ModuleLru<T>& pool, const T& fresh, const std::string& pinned) {
        std::vector<T> evicted;
        pool.insert(fresh, pinned, evicted);
        for (const T& e : evicted)
            if (e.module) cuModuleUnload(e.module);
    }

    // Keys being compiled right now, so two threads asking for the same module (a prewarm and the
    // Run that overtakes it) do not both pay NVRTC: the second waits and takes the pooled result.
    std::set<std::string>            in_flight_keys;
    std::condition_variable_any      compile_done;

    // The shared body of every compile_*_if_needed. `build` does the slow NVRTC work and runs
    // WITHOUT the lock held; `activate` distinguishes a real Run (needs the module in the active
    // slot) from a prewarm (pool only).
    template <class T, class Build>
    bool compile_into(ModuleLru<T>& pool, const std::string& key, T& active,
                      bool activate, Build&& build, std::string& err) {
        std::unique_lock<std::recursive_mutex> lk(cache_mu);
        for (;;) {
            if (activate ? cache_activate(pool, key, active) : pool.contains(key)) return true;
            if (in_flight_keys.insert(key).second) break;   // nobody else is building it: we do
            // wait() drops ONE level of a recursive_mutex, so compile_* must never be reached
            // with cache_mu already held -- it would wait here holding the lock and deadlock.
            compile_done.wait(lk);                          // someone is: take their result
        }
        if (!load_sources(err)) {
            in_flight_keys.erase(key);
            compile_done.notify_all();
            return false;
        }

        T fresh;
        lk.unlock();
        const bool ok = build(fresh);   // seconds of NVRTC, lock released
        lk.lock();

        in_flight_keys.erase(key);
        compile_done.notify_all();
        if (!ok) return false;
        if (activate) cache_publish(pool, fresh, active);
        else          cache_store(pool, fresh, active.key);
        return true;
    }

    // Pins the PREVIOUS active entry while inserting: a running task may still hold its kernels,
    // and evicting it would unload code from under a live launch.
    template <class T>
    void cache_publish(ModuleLru<T>& pool, const T& fresh, T& active) {
        std::vector<T> evicted;
        pool.insert(fresh, active.key, evicted);
        for (const T& e : evicted)
            if (e.module) cuModuleUnload(e.module);
        active = fresh;
    }

    template <class T>
    void drain_pool(ModuleLru<T>& pool) {
        for (const T& e : pool.drain())
            if (e.module) cuModuleUnload(e.module);
    }

    ~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            drain_pool(pool_bif1d);
            drain_pool(pool_lle);
            drain_pool(pool_lle_2d);
            drain_pool(pool_ls);
            drain_pool(pool_ls_2d);
            drain_pool(pool_cont);
            drain_pool(pool_lle_cont);
            drain_pool(pool_ls_cont);
            drain_pool(pool_dft_cont);
            drain_pool(pool_dft_hsweep);
            drain_pool(pool_bif2d);
            drain_pool(pool_basins);
            drain_pool(pool_fs_attr);
            drain_pool(pool_fs_grid);
            drain_pool(pool_order);
            drain_pool(pool_network);
            cuCtxDestroy(context);
        }
    }












    bool ensure_init(std::string& err) {
        if (inited) return true;
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { err = "cuInit: " + cu_err(r); return false; }
        r = cuDeviceGet(&device, 0);
        if (r != CUDA_SUCCESS) { err = "cuDeviceGet: " + cu_err(r); return false; }
#if CUDA_VERSION >= 13000
        r = cuCtxCreate(&context, nullptr, 0, device);
#else
        r = cuCtxCreate(&context, 0, device);
#endif
        if (r != CUDA_SUCCESS) { err = "cuCtxCreate: " + cu_err(r); return false; }
        cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        inited = true;
        return true;
    }

    bool load_sources(std::string& err) {
        // Перечитывать файлы с диска надо один раз, а вот текст configCUDA.h
        // пересобирать — на каждую смену peak-настроек (см. peak_config_defines).
        const uint64_t ep = peak_config_epoch();
        if (srcs_loaded && srcs_peak_epoch == ep) return true;
        if (srcs_loaded) {
            src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
            srcs_peak_epoch  = ep;
            return true;
        }
        std::string root = exe_dir() + "\\kernels\\";
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu",      e); if (!e.empty()) { err = e; return false; }
        src_template_cont     = read_text_file(root + "bifurcation1d_cont.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_template_lle_cont = read_text_file(root + "lle1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_ls_cont  = read_text_file(root + "ls1d_cont.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_dft_cont = read_text_file(root + "dft1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_dft_hsweep = read_text_file(root + "dft1d_hsweep.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_lle      = read_text_file(root + "lle1d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_lle_2d   = read_text_file(root + "lle2d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_ls       = read_text_file(root + "ls1d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_ls_2d    = read_text_file(root + "ls2d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_bif2d    = read_text_file(root + "bifurcation2d.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_basins   = read_text_file(root + "basins.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_fs_attr  = read_text_file(root + "fastsync_attr.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_fs_grid  = read_text_file(root + "fastsync_grid.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_order    = read_text_file(root + "order.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_network  = read_text_file(root + "network.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h_raw  = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
        srcs_peak_epoch  = ep;
        srcs_loaded = true;
        return true;
    }

    // Copy of everything build_module reads, taken under cache_mu. NVRTC then runs unlocked:
    // load_sources() rewrites configCUDA.h whenever the peak settings change, and reading a
    // std::string while another thread assigns it is a race no matter how rare.
    struct SrcSnapshot {
        std::string tmpl, lib_cu, lib_cuh, macros_cuh, config_h;
    };

    SrcSnapshot snapshot_sources(const std::string& tmpl) const {
        return { tmpl, src_cudaLibrary_cu, src_cudaLibrary_cuh, src_cudaMacros_cuh, src_configCUDA_h };
    }

    // Кэш PTX библиотечной половины: ключ — всё, кроме КРС. Именно это и есть смысл затеи:
    // при смене схемы 1.2 МБ PTX библиотеки переиспользуются, а заново собирается только шаг.
    struct CachedLibPtx {
        std::string              key;
        std::string              ptx;
        std::vector<std::string> lowered;   // mangled-имена ядер: они из библиотечной половины
    };
    struct CachedKrsPtx {
        std::string key;
        std::string ptx;
    };

    // Свой мьютекс: build_module работает с ОТПУЩЕННЫМ cache_mu (иначе NVRTC сериализовался бы),
    // поэтому эти кэши защищаются отдельно. Вложенности нет — cache_mu здесь не берётся.
    std::mutex                  ptx_mu;
    std::set<std::string>       lib_in_flight;
    std::condition_variable     lib_ptx_done;
    ModuleLru<CachedLibPtx>     pool_lib_ptx{ 4 };   // ~1.2 МБ на запись
    ModuleLru<CachedKrsPtx>     pool_krs_ptx{ 8 };   // ~50 КБ на запись

    // Компиляция одной единицы трансляции в PTX. rdc=true — для половинок раздельной сборки.
    bool compile_ptx(const std::string& src, const char* src_name,
                     const std::vector<const char*>& headers_src,
                     const std::vector<const char*>& headers_name,
                     const std::vector<const char*>& name_exprs,
                     bool rdc, std::string& out_ptx, std::vector<std::string>& lowered,
                     std::string& err) {
        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), src_name,
                                            (int)headers_src.size(),
                                            headers_src.empty() ? nullptr : headers_src.data(),
                                            headers_name.empty() ? nullptr : headers_name.data());
        if (nr != NVRTC_SUCCESS) {
            err = std::string("nvrtcCreateProgram(") + src_name + "): " + nvrtcGetErrorString(nr);
            return false;
        }
        for (const char* sym : name_exprs) nvrtcAddNameExpression(prog, sym);

        char arch[64];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
        }
        if (cuda_include_opt.empty()) {
            err = "the CUDA_PATH environment variable is not set - NVRTC will not find math_constants.h "
                  "(install the CUDA Toolkit or set CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        std::vector<const char*> opts = { arch, std_opt.c_str(), "-default-device",
                                          cuda_include_opt.c_str(), nvrtc_fmad_opt() };
        if (rdc) opts.push_back("--relocatable-device-code=true");

        nr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = std::string("NVRTC compile failed (") + src_name + "):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        out_ptx.assign(ptxsz, '\0');
        nvrtcGetPTX(prog, &out_ptx[0]);

        // Mangled-имена копируем ДО destroy — после него указатели невалидны.
        lowered.clear();
        lowered.reserve(name_exprs.size());
        for (const char* sym : name_exprs) {
            const char* lo = nullptr;
            nvrtcGetLoweredName(prog, sym, &lo);
            lowered.push_back(lo ? lo : sym);
        }
        nvrtcDestroyProgram(&prog);
        return true;
    }

    // Библиотечная половина: тот же шаблон, но вместо тела шага — вызов внешнего krs_step.
    // Объявление вставляется сразу после #include "cudaLibrary.cuh" (он тянет numb из
    // configCUDA.h, а сам КРС определяется четырьмя строками ниже — структура одинакова у всех
    // 15 шаблонов).
    bool lib_ptx_for(const SrcSnapshot& snap, const char* src_name, const std::string& lib_key,
                     const std::vector<std::pair<std::string, std::string>>& subs,
                     const std::vector<const char*>& name_exprs,
                     CachedLibPtx& out, std::string& err) {
        std::unique_lock<std::mutex> lk(ptx_mu);
        for (;;) {
            if (pool_lib_ptx.take(lib_key, out)) return true;
            if (lib_in_flight.insert(lib_key).second) break;
            lib_ptx_done.wait(lk);
        }
        lk.unlock();

        std::string src = snap.tmpl;
        for (const auto& sub : subs) {
            if (sub.first == "{{KRS_BODY}}") continue;
            src = replace_all(src, sub.first, sub.second);
        }
        src = replace_all(src, "{{KRS_BODY}}", "    krs_step(X, a, h);");
        src = replace_all(src, "#include \"cudaLibrary.cuh\"",
                          "#include \"cudaLibrary.cuh\"\n"
                          "extern \"C\" __device__ void krs_step(numb* X, const numb* a, const numb h);");

        const std::vector<const char*> hs = { snap.lib_cu.c_str(), snap.lib_cuh.c_str(),
                                              snap.macros_cuh.c_str(), snap.config_h.c_str() };
        const std::vector<const char*> hn = { "cudaLibrary.cu", "cudaLibrary.cuh",
                                              "cudaMacros.cuh", "configCUDA.h" };
        CachedLibPtx fresh;
        fresh.key = lib_key;
        const bool ok = compile_ptx(src, src_name, hs, hn, name_exprs, true,
                                    fresh.ptx, fresh.lowered, err);

        lk.lock();
        lib_in_flight.erase(lib_key);
        lib_ptx_done.notify_all();
        if (!ok) return false;
        std::vector<CachedLibPtx> evicted;
        pool_lib_ptx.insert(fresh, std::string(), evicted);
        out = fresh;
        return true;
    }

    // Половина с шагом: отдельная единица трансляции на одном configCUDA.h (нужен и numb, и
    // ucmplx для комплексных схем).
    bool krs_ptx_for(const SrcSnapshot& snap, const std::string& krs_body,
                     const std::string& amount_of_x, const std::string& krs_key,
                     std::string& out_ptx, std::string& err) {
        {
            std::lock_guard<std::mutex> lk(ptx_mu);
            CachedKrsPtx hit;
            if (pool_krs_ptx.take(krs_key, hit)) { out_ptx = hit.ptx; return true; }
        }

        std::string src = "#define AMOUNTOFX " + amount_of_x + "\n"
                          "#include \"configCUDA.h\"\n"
                          "extern \"C\" __device__ void krs_step(numb* X, const numb* a, const numb h) {\n"
                          + krs_body + "\n}\n";
        const std::vector<const char*> hs = { snap.config_h.c_str() };
        const std::vector<const char*> hn = { "configCUDA.h" };
        std::vector<std::string> ignored;
        if (!compile_ptx(src, "krs.cu", hs, hn, {}, true, out_ptx, ignored, err)) return false;

        std::lock_guard<std::mutex> lk(ptx_mu);
        CachedKrsPtx fresh;
        fresh.key = krs_key;
        fresh.ptx = out_ptx;
        std::vector<CachedKrsPtx> evicted;
        pool_krs_ptx.insert(fresh, std::string(), evicted);
        return true;
    }

    // Линкует половинки в cubin и грузит модуль. Указатель из cuLinkComplete живёт до
    // cuLinkDestroy — модуль обязан загрузиться ДО разрушения линкера.
    bool link_and_load(const std::string& lib_ptx, const std::string& krs_ptx,
                       CUmodule& out_module, std::string& err) {
        CUlinkState st = nullptr;
        CUresult r = cuLinkCreate(0, nullptr, nullptr, &st);
        if (r != CUDA_SUCCESS) { err = "cuLinkCreate: " + cu_err(r); return false; }
        auto fail = [&](const std::string& what, CUresult rr) {
            err = what + ": " + cu_err(rr);
            cuLinkDestroy(st);
            return false;
        };
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)lib_ptx.data(), lib_ptx.size() + 1,
                          "lib", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(lib)", r);
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)krs_ptx.data(), krs_ptx.size() + 1,
                          "krs", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(krs)", r);

        void* cubin = nullptr; size_t cubin_size = 0;
        r = cuLinkComplete(st, &cubin, &cubin_size);
        if (r != CUDA_SUCCESS) return fail("cuLinkComplete", r);

        CUmodule mod = nullptr;
        r = cuModuleLoadData(&mod, cubin);
        cuLinkDestroy(st);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadData(linked): " + cu_err(r); return false; }
        out_module = mod;
        return true;
    }

    // build_module — общая часть всех compile_*_if_needed: подстановка плейсхолдеров в шаблон,
    // NVRTC-компиляция с едиными опциями, добыча mangled-имён и загрузка PTX. У вызывающих
    // различаются только шаблон, имя исходника и набор символов.
    // Ради этого всё и сведено: раньше блок был скопирован под каждый анализ, и когда --fmad стал
    // настройкой, флаг проставили в nvrtc_engine.cpp, а копии здесь остались на дефолте NVRTC —
    // карта и фазовый портрет считались разной арифметикой, и на фрактальной границе траектория
    // уходила в другой аттрактор (см. nvrtc_fmad_opt).
    // Требует выставленного контекста и уже загруженных источников (load_sources). При успехе
    // out_module загружен, а lowered содержит по одному имени на каждый вход name_exprs в том же
    // порядке. Символы, объявленные extern "C", в name_exprs передавать не нужно.
    bool build_module(const SrcSnapshot& snap,
                      const char* src_name,
                      const std::vector<std::pair<std::string, std::string>>& subs,
                      const std::vector<const char*>& name_exprs,
                      CUmodule& out_module,
                      std::vector<std::string>& lowered,
                      std::string& err) {
        // Раздельная сборка: библиотека без КРС компилируется один раз на (шаблон, размерность,
        // настройки) и переиспользуется, шаг живёт в своём модуле, склейка — cuLink. Замерено на
        // bifurcation2d + implicit midpoint: смена КРС 0.25 с против ~15 с одной единицей
        // трансляции. Цена — вызов вместо инлайна: на RTX 2060 SUPER от -1% до +7% времени счёта,
        // результаты побитово те же (FP64 идёт 1/32 скорости, накладные прячутся в её тени).
        if (get_nvrtc_rdc()) {
            std::string krs_body, amount_of_x = "3";
            std::string lib_key = src_name;
            for (const auto& sub : subs) {
                if (sub.first == "{{KRS_BODY}}") { krs_body = sub.second; continue; }
                if (sub.first == "{{AMOUNT_OF_X}}") amount_of_x = sub.second;
                lib_key += ''; lib_key += sub.first; lib_key += '='; lib_key += sub.second;
            }
            lib_key += get_nvrtc_fmad() ? "|fm1" : "|fm0";
            lib_key += "|pk" + std::to_string(peak_config_epoch());

            std::string krs_err;
            std::string krs_ptx;
            const std::string krs_key = std::to_string(std::hash<std::string>{}(krs_body))
                                      + ":" + amount_of_x
                                      + (get_nvrtc_fmad() ? ":fm1" : ":fm0")
                                      + ":pk" + std::to_string(peak_config_epoch());
            if (!krs_ptx_for(snap, krs_body, amount_of_x, krs_key, krs_ptx, krs_err)) {
                // Не компилируется САМ шаг — это ошибка пользователя, и монолитный путь выдал бы
                // ту же самую. Отдаём как есть, без второго захода на те же грабли.
                err = krs_err;
                return false;
            }

            CachedLibPtx lib;
            std::string lib_err;
            if (lib_ptx_for(snap, src_name, lib_key, subs, name_exprs, lib, lib_err)
                && link_and_load(lib.ptx, krs_ptx, out_module, lib_err)) {
                lowered = lib.lowered;
                return true;
            }
            // Библиотека или линковка — не пользовательская ошибка: собираем по-старому, одной
            // единицей трансляции. Медленно, но работает везде, где работало раньше.
            std::fprintf(stderr, "[nvrtc] rdc split failed (%s), falling back to one TU\n",
                         lib_err.c_str());
        }

        std::string src = snap.tmpl;
        for (const auto& sub : subs) src = replace_all(src, sub.first, sub.second);

        // Виртуальные заголовки для NVRTC. curand_kernel.h-stub НЕ нужен:
        // inline-stub в шаблонах + `#define CURAND_KERNEL_H_` блокируют как
        // реальный header (по -I path), так и любой повторный inject.
        const char* header_sources[] = {
            snap.lib_cu.c_str(),
            snap.lib_cuh.c_str(),
            snap.macros_cuh.c_str(),
            snap.config_h.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
        };
        constexpr int n_headers = 4;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), src_name,
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) {
            err = std::string("nvrtcCreateProgram(") + src_name + "): " + nvrtcGetErrorString(nr);
            return false;
        }

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        for (const char* sym : name_exprs) nvrtcAddNameExpression(prog, sym);

        char arch[64];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h,
        // curand_kernel.h и т.п.). Берём CUDA_PATH из окружения — его проставляет
        // CUDA Toolkit installer.
        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
        }
        if (cuda_include_opt.empty()) {
            err = "the CUDA_PATH environment variable is not set - NVRTC will not find math_constants.h "
                  "(install the CUDA Toolkit or set CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device",
                               cuda_include_opt.c_str(), nvrtc_fmad_opt() };

        nr = nvrtcCompileProgram(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = std::string("NVRTC compile failed (") + src_name + "):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);

        // Mangled-имена копируем в свои строки ДО nvrtcDestroyProgram — после
        // destroy указатели становятся невалидны.
        lowered.clear();
        lowered.reserve(name_exprs.size());
        for (const char* sym : name_exprs) {
            const char* lo = nullptr;
            nvrtcGetLoweredName(prog, sym, &lo);
            lowered.push_back(lo ? lo : sym);
        }
        nvrtcDestroyProgram(&prog);

        // Грузим во временную переменную: при неудаче у вызывающего слот кэша
        // остаётся нетронутым, и release_*_module() не получит мусорный handle.
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) {
            err = std::string("cuModuleLoadDataEx(") + src_name + "): " + cu_err(r);
            return false;
        }
        out_module = mod;
        return true;
    }

    // Достаёт функцию из загруженного модуля. Отдельная обёртка — чтобы у
    // вызывающих не размножалось одинаковое сообщение об ошибке.
    bool module_fn(CUmodule mod, const std::string& name, CUfunction& out, std::string& err) {
        CUresult r = cuModuleGetFunction(&out, mod, name.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + name + "): " + cu_err(r);
            return false;
        }
        return true;
    }

    bool compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err, bool activate = true) {
        // Если worker-thread унаследовал чужой контекст (NvrtcEngine, например),
        // компиляция и загрузка модуля прицепят символы не в тот контекст. Жёстко
        // выставляем наш перед NVRTC/CU-вызовами.
        cuCtxSetCurrent(context);
        // par_or_var в kernel'е cudaLibrary.cu — compile-time макрос. Поэтому
        // включаем его в hash-key: param-sweep и IC-sweep кешируются отдельно.
        std::string key = hash_key(krs_body, amountOfX) + ":pov" + std::to_string(par_or_var);
        return compile_into(pool_bif1d, key, cached, activate, [&](CachedModule& fresh) {
            // DFT_custom уже присутствует в этом же модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) — регистрируем его тоже, чтобы run_dft_1d
            // мог переиспользовать этот кэш без отдельной компиляции.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template), "bifurcation1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelCUDA", "peakFinderCUDA", "DFT_custom",
                                "calculateDiscreteModelPeaksCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_traj,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_peak,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_dft,   err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_fused, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_bif1d — порт NonLinAnal::bifurcation1D из hostLibrary.cu. Идея: брать оригинальный код
    // почти как есть, чтобы при обновлениях NonLinAnal перенос был механическим diff → patch.
    // ОБЯЗАТЕЛЬНЫЕ изменения (помечены комментарием [ADAPT]):
    //   - <<<grid, block, shared>>>(...) → cuLaunchKernel(CUfunction, ...): наш kernel-модуль
    //     скомпилирован NVRTC'ом во время работы и недоступен по compile-time символу
    //   - gpuErrorCheck(...) у NonLinAnal зовёт exit(); здесь локальный BIF_CHECK, который пишет
    //     в res.error и возвращает результат с cleanup'ом
    //   - OUT_FILE_PATH приходит из req.csv_output_path (пусто = CSV не пишем)
    //   - ветка continuation_bif1D == 1 отключена: она использует host-side
    //     calculateDiscreteModel (default Lorenz), а не user's KRS
    //   - calculate_mean_med_freq / calculate_mean_and_variance отключены — их kernel-ы не входят
    //     в NVRTC-bundle
    //   - результат пишется в Bifurcation1DResult (память + CSV), не в файл
    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) {
        // Continuation требует sequential x-carry — это совсем другой путь
        // (single-thread kernel). Отказываем при IC-sweep (не имеет смысла:
        // continuation подразумевает param как непрерывный параметр).
        if (req.continuation) {
            if (req.sweep_over_var) {
                Bifurcation1DResult r;
                r.error = "continuation requires a param sweep, not an IC sweep";
                return r;
            }
            // Continuation-ветка возвращается ДО общего блока валидации ниже,
            // поэтому log-проверку дублируем здесь: без неё log10(lo<=0) дал бы
            // NaN-сетку молча. У LLE/LS эта проверка отрабатывает раньше
            // диспетчера, там дублировать не нужно.
            if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
                Bifurcation1DResult r;
                r.error = "log scale requires param lo/hi > 0";
                return r;
            }
            // h-свип теперь поддержан обеими ветками: шаг пересчитывается в
            // каждой точке, буфер блока выделен под худший случай.
            // CPU-ветка — до ensure_init: она не трогает CUDA вообще, так что
            // считает и на машине без работающего GPU-контекста.
            if (req.use_cpu) return run_bif1d_continuation_cpu(req);
            return run_bif1d_continuation(req);
        }

        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        // base_values уже идёт со сдвигом +1 (a[0] зарезервирован):
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.sweep_over_h) {
            if (req.param_lo <= 0.0 || req.param_hi <= 0.0)
                return fail("h lo/hi must be > 0 with sweep_over_h");
        } else if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index out of range");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index out of range");
        }
        // writable_var == -1 — sentinel "combination of first vars" (порт MATLAB
        // выбора x[0]+pi*x[1]+euler*x[2] в loopCalculateDiscreteModel_int).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0");

        // Классический свип на CPU — для счёта без GPU и для сверки ядра с GPU
        // (то же место в конвейере, что у run_lle_1d / run_ls_1d). Стоит после
        // общей валидации и до ensure_init: CUDA этой ветке не нужна вовсе.
        if (req.use_cpu) return run_bif1d_cpu(req);

        // dt-sweep: t_max/transient_time фиксированы, число шагов на GPU
        // пересчитывается из h per-thread (см. hSweepAxis в
        // calculateDiscreteModelCUDA). hSweepAxis=0 -- в 1D всегда единственная
        // ось X.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        // Log-масштаб сетки (любой sweep target) -- бит 0, т.к. в 1D одна ось.
        const int logAxisMask = req.log_scale ? 1 : 0;

        // init + контекст
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // компиляция или cache hit
        if (!compile_if_needed(req.krs_body, req.amountOfX,
                               req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ПОРТ NonLinAnal::bifurcation1D (hostLibrary.cu:165-655).
        // Локальные имена мапятся на аргументы функции NonLinAnal для удобства
        // дифа — слева name из req, справа name как в hostLibrary.
        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target:
        //   param-sweep → indicesOfMutVars[0] = 1-based индекс параметра (a[]),
        //                 par_or_var_arg = true  → kernel пишет в localValues
        //   var-sweep   → indicesOfMutVars[0] = 0-based индекс переменной (X[]),
        //                 par_or_var_arg = false → kernel пишет в localX
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // Константы из configCUDA.h — у NonLinAnal они constexpr,
        // здесь литералы (значения те же).
        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // [ADAPT] continuation_bif1D, calculate_mean_med_freq отключены —
        // см. комментарий в шапке функции.

        // amountOfPointsInBlock / amountOfPointsForSkip — порт строк 196-200 NL
        // dt-sweep: буфер/sizeOfBlock должен вмещать худший случай (минимальный
        // h в диапазоне = param_lo после lo<=hi нормализации = больше всего
        // шагов); per-thread реальное число шагов пересчитывается в кернеле и
        // не превышает эту аллокацию (см. actualIterations).
        double worstCaseH = (hSweepAxis != -1) ? ranges[0] : h;
        int amountOfPointsInBlock = (int)std::ceil(tMax / worstCaseH / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Пики живут в собственных коротких строках, а не в буферах длиной
        // в траекторию — см. run_bif2d. Для 1D это важнее, чем для 2D: здесь оба
        // буфера пиков едут на хост целиком каждый чанк (2D возвращал по int на ячейку).
        const size_t peakStride   = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                  ? (size_t)amountOfPointsInBlock
                                  : (size_t)max_amount_of_peaks + 1;
        const int    peakCapacity = (int)peakStride;

        // Memory budget (порт строк 202-244 NL)
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        size_t memPerSystem =
            2 * peakStride * sizeof(numb) +                     // d_outPeaks, d_timeOfPeaks
            sizeof(int);                                          // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(numb) +
            sizeof(int) +
            (size_t)amountOfInitialConditions * sizeof(numb) +
            (size_t)amountOfValues * sizeof(numb);
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        // Округления вниз до кратного blockSize_setup (nPtsLimiter / 32 * 32) здесь больше нет:
        // ядра сами отсекают лишние потоки через `if (idx >= nPtsLimiter) return`, а последний чанк
        // кратным 32 не бывает и всегда считался нормально. Зато при n_pts < 32 округление давало
        // 0, и Run падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("n_pts must be > 0");
        size_t originalNPtsLimiter = nPtsLimiter;

        // Host buffers (порт строк 257-264 NL)
        // h_data/h_meanFreq/h_medianFreq/h_localX/h_localValues нужны только для
        // continuation_bif1D и mean/median — мы их не используем.
        std::vector<numb> h_outPeaks   (nPtsLimiter * peakStride);
        std::vector<numb> h_timeOfPeaks(nPtsLimiter * peakStride);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);

        // Device buffers (порт строк 297-306 NL, без d_meanFreq/d_medianFreq)
        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        RunSignals sig;                       // прогресс и отмена в mapped-памяти
        numb* d_outPeaks          = nullptr;
        numb* d_timeOfPeaks       = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            sig.release();
            if (d_outPeaks)          cudaFree(d_outPeaks);
            if (d_timeOfPeaks)       cudaFree(d_timeOfPeaks);
        };

        // [ADAPT] gpuErrorCheck → BIF_CHECK: пишем в res.error и выходим с cleanup
        #define BIF_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        // Cooperative cancellation between kernel launches.
        #define BIF_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BIF_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                          "cudaMalloc d_ranges");
        BIF_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        BIF_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),          "cudaMalloc d_initialConditions");
        BIF_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                     "cudaMalloc d_values");
        BIF_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_outPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_timeOfPeaks,       nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_timeOfPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        // H2D констант (порт строк 314-319 NL)
        BIF_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        BIF_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot of CSV-relevant request fields, captured BEFORE any disk
        // write so it can be reused by the GUI right-click export (which runs
        // after `req` is gone). Engine and GUI share the same writers in
        // data_export, so the on-disk format is identical byte-for-byte.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        // Config CSV (порт строк 331-376 NL — упрощённо, только если путь задан)
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_bif1d_config(cfg, res.snapshot);
            // обнуляем основной файл данных
            std::ofstream trunc(OUT_FILE_PATH);
            trunc.close();
        }

        // результат-аккумулятор (для GUI)
        res.n_pts        = nPts;
        res.record_steps = (int)peakStride;   // ёмкость строки пиков, а не длина траектории
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});

        // Главный цикл (порт строк 396-630 NL)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BIF_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            // последний чанк может быть меньше
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // Ширина блока — настройка (Settings -> GPU launch), ужатая под shared этого ядра.
            int blockSize = launch_block_size(
                (size_t)ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb));
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // [ADAPT] <<<>>> → cuLaunchKernel. Траектория и поиск пиков — одно
            // ядро, то же, что у run_bif2d. Continuation отключена.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            numb h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = !req.sweep_over_var; // true=param, false=IC
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int    logAxisMask_arg           = logAxisMask;
            size_t peakStride_arg            = peakStride;
            int    peakCapacity_arg          = peakCapacity;
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            bool   emitAllSamples_arg        = req.emit_all_samples;
            sig.resetTicks();

            void* args_fused[] = {
                &nPts_int, &nPtsLimiter_int, &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s, &dimension, &d_ranges, &h_arg,
                &d_indicesOfMutVars, &d_initialConditions, &amountOfInitialConditions_int,
                &d_values, &amountOfValues_int, &amountOfIterations_arg,
                &preScaller_int, &writableVar_int, &maxValue_arg,
                &d_outPeaks, &d_timeOfPeaks, &d_amountOfPeaks,
                &par_or_var_arg, &hSweepAxis_arg, &transientTime_arg, &tMax_arg,
                &logAxisMask_arg, &peakStride_arg, &peakCapacity_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg,
                &emitAllSamples_arg
            };

            unsigned int shared = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);

            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_fused,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_fused, nullptr),
                         "cuLaunchKernel(bif1d traj+peaks)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BIF_CHECK(cudaDeviceSynchronize(), "sync after traj+peaks");
            BIF_CANCEL_CHECK();

            // D2H (порт строк 538-540 NL)
            BIF_CHECK(cudaMemcpy(h_outPeaks.data(),       d_outPeaks,       nPtsLimiter * peakStride * sizeof(numb),                    cudaMemcpyDeviceToHost), "memcpy h_outPeaks");
            BIF_CHECK(cudaMemcpy(h_amountOfPeaks.data(),  d_amountOfPeaks,  nPtsLimiter * sizeof(int),                                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            BIF_CHECK(cudaMemcpy(h_timeOfPeaks.data(),    d_timeOfPeaks,    nPtsLimiter * peakStride * sizeof(numb),                    cudaMemcpyDeviceToHost), "memcpy h_timeOfPeaks");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // CSV + аккумуляция результата (порт строк 574-608 NL)
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }

            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                int    npeaks     = h_amountOfPeaks[k];

                // Поправки на h-свип здесь больше нет. Раньше peakFinderCUDA получал общий
                // на запуск h, и хост домножал времена на param_val/h; слитое ядро
                // сразу масштабирует интервалы шагом своей точки, и вторая поправка
                // применилась бы дважды (см. commit-message к слиянию ядер bif2d).

                int n = npeaks;
                if (n > (int)peakStride) n = (int)peakStride;
                const numb* peakRow = h_outPeaks.data()    + k * peakStride;
                const numb* timeRow = h_timeOfPeaks.data() + k * peakStride;

                // Экспорт и результат работают в double — расширяем numb-ряды
                // здесь, один раз, и переиспользуем для CSV и для памяти.
                std::vector<double> widePeaks, scaledTimes;
                if (n > 0) {
                    widePeaks.assign(peakRow, peakRow + n);
                    scaledTimes.assign(timeRow, timeRow + n);
                }

                if (out.is_open())
                    data_export::write_bif1d_rows(out, param_val, npeaks,
                                                  widePeaks.data(), scaledTimes.data());

                // В память для GUI: значения пиков и межпиковые интервалы
                res.flags[global_idx] = npeaks;
                auto& dst_peaks = res.bifurcation_points[global_idx];
                auto& dst_times = res.peak_times[global_idx];
                if (n > 0) {
                    dst_peaks = std::move(widePeaks);
                    dst_times = std::move(scaledTimes);
                } else {
                    dst_peaks.clear();
                    dst_times.clear();
                }
            }

            if (out.is_open()) out.close();
        }

        cleanup();
        #undef BIF_CHECK
        #undef BIF_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_lle_if_needed — отдельная компиляция для LLE-шаблона.
    // Ключ кэша = hash(krs_body + amountOfX + "lle"), чтобы PTX от bif1d
    // не путался с LLE даже при одной и той же KRS.
    bool compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        // par_or_var в kernel — compile-time макрос (см. bif1d). Два PTX-модуля
        // кешируются отдельно: param-sweep и IC-sweep.
        std::string key = hash_key(krs_body, amountOfX) + ":lle:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle, key, cached_lle, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_lle), "lle1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_lle_1d — порт NonLinAnal::LLE1D из hostLibrary.cu, той же diff-friendly стратегией, что
    // run_bif1d (оригинальные имена слева, req-имена справа). [ADAPT] — отличия от NonLinAnal:
    // <<<>>> → cuLaunchKernel (NVRTC-модуль); gpuErrorCheck → локальный LLE_CHECK с cleanup'ом,
    // без exit(); OUT_FILE_PATH — req.csv_output_path (пусто = без файла); результат пишется в
    // LLE1DResult (память + опц. CSV).
    LLE1DResult run_lle_1d(const LLE1DRequest& req) {
        LLE1DResult res;
        auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.sweep_over_h) {
            if (req.param_lo <= 0.0 || req.param_hi <= 0.0)
                return fail("h lo/hi must be > 0 with sweep_over_h");
        } else if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index out of range");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index out of range");
        }
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.NT <= 0.0)          return fail("NT must be > 0");
        if (req.eps <= 0.0)         return fail("eps must be > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0");

        // Continuation: точки выстроены в цепочку, поэтому IC-свип несовместим
        // (как в run_bif1d). h-свип и log-сетка поддержаны на обоих устройствах.
        if (req.continuation) {
            if (req.sweep_over_var) return fail("continuation requires a param sweep, not an IC sweep");
            return req.use_cpu ? run_lle1d_cpu(req, /*continuation*/ true)
                               : run_lle1d_continuation_gpu(req);
        }
        // Классический свип на CPU — для счёта без GPU и для сверки ядра с GPU.
        if (req.use_cpu) {
            if (req.sweep_over_var) return fail("the CPU branch supports a param sweep only");
            return run_lle1d_cpu(req, /*continuation*/ false);
        }

        // dt-sweep: t_max/transient_time/NT остаются фиксированными (см.
        // sweep_over_h в analysis_session.h), число шагов пересчитывается на GPU
        // из h per-thread -- kernel получает hSweepAxis вместо compile-time флага.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        const int logAxisMask = req.log_scale ? 1 : 0;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_if_needed(req.krs_body, req.amountOfX,
                                   req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ПОРТ NonLinAnal::LLE1D (hostLibrary.cu:2261-2511)
        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target: см. bif1d. При IC-свипе индекс — 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        // amountOfPointsInBlock = tMax / NT — число NT-блоков интегрирования.
        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT too small)");

        // Memory budget — мирор NonLinAnal LLE1D:2291-2299 (консервативно).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lleResult (одно numb): траектория
        // не хранится, ядро зовёт цикл интегрирования с data = nullptr. Прежняя
        // формула делила память на длину блока, которой здесь нет.
        size_t nPtsLimiter = freeMemory / sizeof(numb);
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_lleResult(nPtsLimiter);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lleResult         = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };
            sig.release();

        #define LLE_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LLE_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LLE_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),"cudaMalloc d_initialConditions");
        LLE_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LLE_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(numb)),                      "cudaMalloc d_lleResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        LLE_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot CSV-relevant request fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_lle1d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // Результат-аккумулятор
        res.n_pts    = nPts;
        res.param_lo = ranges[0];
        res.param_hi = ranges[1];
        res.lyapunov.assign(nPts, 0.0);
        res.flags.assign(nPts, 0);

        // Главный цикл (порт NonLinAnal LLE1D:2403-2496)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LLE_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize по формуле NonLinAnal (hostLibrary.cu:2419). Верхний cap теперь не 32,
            // а настройка (Settings -> GPU launch); формула остаётся вторым потолком — при
            // широкой системе она даёт МЕНЬШЕ варпа, и терять эту защиту нельзя.
            const size_t sharedPerThread = (size_t)(3 * amountOfInitialConditions + amountOfValues) * sizeof(numb);
            int blockSize = (int)std::ceil((1024.0 * 32.0) / (double)sharedPerThread);
            if (blockSize < 1) blockSize = 1;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap) blockSize = blockSizeCap;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA (cudaLibrary.cu:2379)
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult,
                &hSweepAxis_arg,
                &transientTime_arg,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            // Shared = (3 * amountOfIC + amountOfValues) * sizeof(numb) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LLE_CHECK_CU(cuLaunchKernel(cached_lle.kernel_lle,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(lle)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LLE_CHECK(cudaDeviceSynchronize(), "sync after lle");

            LLE_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, nPtsLimiter * sizeof(numb), cudaMemcpyDeviceToHost),
                      "memcpy h_lleResult");
            LLE_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // Аккумулируем + опциональный CSV (порт NonLinAnal LLE1D:2478-2492)
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                double v          = h_lleResult[k];

                // 999 / -999 — спец-флаги из kernel'а (нет аттрактора / разошлось). Наружу отдаём
                // NaN, а не сырой sentinel: 999 — легитимное по типу значение λ, поэтому любой
                // потребитель, забывший сверить flags[], молча рисовал выброс на 999. NaN
                // отсекается через !isfinite (и в GUI, и в min/max), т.е. безопасен по умолчанию;
                // flags[] остаётся источником истины о ПРИЧИНЕ отсутствия точки.
                const bool diverged = (v == 999.0 || v == -999.0);
                if (diverged) v = std::numeric_limits<double>::quiet_NaN();

                res.lyapunov[global_idx] = v;
                // LLE не детектирует fixed point (ветка в LLEKernelCUDA выключена),
                // поэтому единственный «плохой» код — REGIME_UNBOUND.
                res.flags[global_idx] = diverged ? REGIME_UNBOUND : REGIME_OSCILLATION;

                if (out.is_open()) data_export::write_lle1d_row(out, param_val, v);
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LLE_CHECK
        #undef LLE_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_lle_2d_if_needed — отдельный PTX-кэш для LLE-2D. Структура та
    // же, что у compile_lle_if_needed; отличия только в исходнике шаблона
    // (lle2d.template.cu) и в маркере ключа кэша (":lle2d"). Сам kernel
    // (LLEKernelCUDA) — тот же, ловится по тому же имени.
    bool compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":lle2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle_2d, key, cached_lle_2d, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_lle_2d), "lle2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_lle_2d — λ(p1, p2) на квадратной сетке, порт NonLinAnal::LLE2D на NVRTC-engine.
    // Отличия от LLE1D: ranges[4], indicesOfMutVars[2], dimension=2, chunking по ячейкам
    // (n_pts × n_pts может не влезть в память за раз), результат — плоский row-major массив.
    // par_or_var (compile-time): mixed_mode=true → 2; иначе оба свипа одного типа — 1 если обе оси
    // param, 0 если обе IC. Смешанные комбинации без mixed_mode не поддержаны (validator
    // отказывает — kernel-ветки под это нет).
    LLE2DResult run_lle_2d(const LLE2DRequest& req) {
        LLE2DResult res;
        auto fail = [&](const std::string& msg) -> LLE2DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of the allowed range");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.NT <= 0.0)          return fail("NT must be > 0");
        if (req.eps <= 0.0)         return fail("eps must be > 0");

        // par_or_var — compile-time. Логика маппинга см. parametric_engine.h.
        // Дополнительно нужен swap_xy: если X=param, Y=IC, передаём в kernel с
        // X↔Y и потом транспонируем результат на хосте.
        auto check_param = [&](int p1based) -> bool {
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;   // что передаём в indicesOfMutVars[0/1]
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;
        int  hSweepAxis = -1;          // -1=off, 0=X свипует h, 1=Y свипует h
        // log_axis_x/y следуют тому же swap, что и ranges_lo_x/y ниже -- в
        // кернел они идут как биты того же физического слота (0=X,1=Y), а не
        // пользовательской оси. Валидация (lo/hi>0) проверяется по
        // ПОЛЬЗОВАТЕЛЬСКИМ req.log_scale/_2 отдельно, ниже, до свопа.
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0 (X axis)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale requires param lo/hi > 0 (Y axis)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h and sweep_over_h_2 cannot both be true");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // Ровно одна ось — h, другая param либо IC. Кернел-слоты X/Y совпадают с
            // пользовательскими напрямую, swap_xy тут не нужен: в смешанном param/IC случае ниже
            // swap существует только потому, что ветка par_or_var==2 захардкожена под одну
            // конкретную пару слотов, а здесь par_or_var симметричен по слотам.
            hSweepAxis = req.sweep_over_h ? 0 : 1;
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);

            if (req.sweep_over_h) {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index_2))   return fail("param_index_2 (Y axis) out of range");
                    idx_axis_y = req.param_index_2;
                } else {
                    if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
                    idx_axis_y = req.var_sweep_index_2;
                }
                idx_axis_x = 0; // dummy -- слот X пропускается в кернеле (i == hSweepAxis)
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0; // dummy
            }
            // ranges для h-оси переиспользуют тот же param_lo/hi(_2), что и для
            // param/IC -- семантика диапазона просто меняется на "значения h".
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (par_or_var == 1) {
                if (!check_param(req.param_index))    return fail("param_index (X axis) out of range");
                if (!check_param(req.param_index_2))  return fail("param_index_2 (Y axis) out of range");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))    return fail("var_sweep_index (X axis) out of range");
                if (!check_var(req.var_sweep_index_2))  return fail("var_sweep_index_2 (Y axis) out of range");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            // X=IC, Y=param — нативно соответствует ветке par_or_var=2 kernel'а.
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
            if (!check_param(req.param_index_2))   return fail("param_index_2 (Y axis) out of range");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            // X=param, Y=IC — kernel напрямую не поддерживает, свопаем оси
            // под капотом и транспонируем результат при выгрузке.
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
            if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
            // В kernel: ось 1 (IC) = пользовательский Y, ось 2 (param) = X.
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;  // тот же своп, что и ranges выше
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;       // сторона сетки
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x,
                                                    (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT too small)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — мирор NonLinAnal LLE2D (hostLibrary.cu:2535-2547).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lleResult (одно numb): траектория
        // не хранится, ядро зовёт цикл интегрирования с data = nullptr. Прежняя
        // формула делила память на длину блока, которой здесь нет.
        size_t nPtsLimiter = freeMemory / sizeof(numb);
        if (nPtsLimiter == 0)                  nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)         nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_lleResult(nPtsLimiter);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lleResult         = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };
            sig.release();

        #define LLE2_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE2_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE2_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LLE2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LLE2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),"cudaMalloc d_initialConditions");
        LLE2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LLE2_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(numb)),                      "cudaMalloc d_lleResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerPoint;

        LLE2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // В result диапазоны храним всегда «пользовательские» (X = первая ось
        // запроса), не свопнутые ranges[] — чтобы GUI рисовал оси правильно
        // независимо от того, делал ли engine внутренний свап.
        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        // Snapshot of CSV-relevant fields in USER ordering — engine + GUI
        // share the writer, so both files agree on axis ordering even when
        // the engine has internally swapped X/Y for kernel dispatch.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts     = nPts;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_lle2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LLE2_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // Cap — настройка, формула остаётся вторым потолком (см. run_lle_1d).
            const size_t sharedPerThread = (size_t)(3 * amountOfInitialConditions + amountOfValues) * sizeof(numb);
            int blockSize = (int)std::ceil((1024.0 * 32.0) / (double)sharedPerThread);
            if (blockSize < 1) blockSize = 1;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap) blockSize = blockSizeCap;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA — те же 21 параметр, что и в LLE1D,
            // но dimension=2 и амбулатура индекса nPts остаётся "сторона сетки".
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult,
                &hSweepAxis_arg,
                &transientTime_arg,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LLE2_CHECK_CU(cuLaunchKernel(cached_lle_2d.kernel_lle,
                                         gridSize, 1, 1, blockSize, 1, 1,
                                         shared, nullptr, args, nullptr),
                          "cuLaunchKernel(lle2d)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after lle2d");

            LLE2_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, cur_limiter * sizeof(numb), cudaMemcpyDeviceToHost),
                       "memcpy h_lleResult");
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                double v          = h_lleResult[k];
                // Транспонирование при свопе: kernel_idx = kix + n*kiy, в
                // системе пользователя ix=kiy, iy=kix → user_idx = kix*n + kiy.
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                res.values[out_idx] = v;
                // 999/-999 — kernel-sentinel расходимости; FP LLE не различает.
                res.flags[out_idx]  = (v == 999.0 || v == -999.0) ? REGIME_UNBOUND
                                                                  : REGIME_OSCILLATION;
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_lle2d_grid(out, nPts, res.values.data());
            }
        }

        // Авто-нормализация для colormap'а: min/max по валидным значениям.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < total_cells; ++k) {
            int f = res.flags[k];
            double v = res.values[k];
            if (!regime_is_oscillation(f)) continue;   // FP и unbound вне шкалы
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        if (std::isfinite(vmin) && std::isfinite(vmax)) {
            res.min_val = vmin;
            res.max_val = vmax;
        } else {
            res.min_val = 0.0;
            res.max_val = 0.0;
        }

        cleanup();
        #undef LLE2_CHECK
        #undef LLE2_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_ls_if_needed — третий шаблон. Ключ кэша помечен ":ls".
    bool compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls, key, cached_ls, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_ls), "ls1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_ls_1d — порт NonLinAnal::LS1D той же стратегией, что run_bif1d / run_lle_1d.
    // Per-system результат — вектор длины amountOfX (один Ляпунов на переменную).
    // Memory budget на поток у LS значительно больше: shared = (3*N + 2*N^2 + nValues) *
    // sizeof(numb) * blockSize, и blockSize подбирается из 32K shared-лимита (как в NonLinAnal).
    LS1DResult run_ls_1d(const LS1DRequest& req) {
        LS1DResult res;
        auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.sweep_over_h) {
            if (req.param_lo <= 0.0 || req.param_hi <= 0.0)
                return fail("h lo/hi must be > 0 with sweep_over_h");
        } else if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index out of range");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index out of range");
        }
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.NT <= 0.0)          return fail("NT must be > 0");
        if (req.eps <= 0.0)         return fail("eps must be > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0");

        // CPU-ветки — до ensure_init, CUDA им не нужна. Ограничения те же, что
        // и у LLE (см. run_lle_1d).
        if (req.continuation) {
            if (req.sweep_over_var) return fail("continuation requires a param sweep, not an IC sweep");
            return req.use_cpu ? run_ls1d_cpu(req, /*continuation*/ true)
                               : run_ls1d_continuation_gpu(req);
        }
        if (req.use_cpu) {
            if (req.sweep_over_var) return fail("the CPU branch supports a param sweep only");
            return run_ls1d_cpu(req, /*continuation*/ false);
        }

        // dt-sweep: см. run_lle_1d -- t_max/transient_time/NT фиксированы, число
        // шагов пересчитывается на GPU из h per-thread.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        const int logAxisMask = req.log_scale ? 1 : 0;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_ls_if_needed(req.krs_body, req.amountOfX,
                                  req.sweep_over_var ? 0 : 1, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target: param-sweep → 1-based в localValues; IC-sweep → 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT too small)");

        // Memory budget — мирор NonLinAnal LS1D:2719-2727 (агрессивно делит /16,
        // т.к. per-system memory ~ N).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lsResult (N numb): траектория
        // не хранится. Прежняя формула умножала это на длину блока и брала
        // 1/16 свободной памяти — вместе это резало чанк без причины.
        size_t perSystemBytes = sizeof(numb) * (size_t)amountOfInitialConditions;
        if (perSystemBytes == 0) perSystemBytes = sizeof(numb);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult хранит nPtsLimiter × N row-major.
        std::vector<numb> h_lsResult(nPtsLimiter * (size_t)amountOfInitialConditions);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lsResult          = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };
            sig.release();

        #define LS_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LS_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                                          "cudaMalloc d_ranges");
        LS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                                             "cudaMalloc d_indicesOfMutVars");
        LS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),                          "cudaMalloc d_initialConditions");
        LS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                                     "cudaMalloc d_values");
        LS_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(numb)),            "cudaMalloc d_lsResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        LS_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot CSV-relevant request fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_ls1d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        res.n_pts       = nPts;
        res.n_exponents = amountOfInitialConditions;
        res.param_lo    = ranges[0];
        res.param_hi    = ranges[1];
        res.spectrum.assign(nPts, std::vector<double>(amountOfInitialConditions, 0.0));
        res.flags.assign(nPts, 0);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LS_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize: 32K shared / per-thread (порт LS1D:2821-2824). Cap — настройка
            // (Settings -> GPU launch); формула остаётся вторым потолком, см. run_lle_1d.
            const size_t sharedPerThread = (size_t)(3 * amountOfInitialConditions
                                + 2 * amountOfInitialConditions * amountOfInitialConditions
                                + amountOfValues) * sizeof(numb);
            int blockSizeMax = (int)(32000 / (double)sharedPerThread);
            int blockSize = blockSizeMax;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap) blockSize = blockSizeCap;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult, &hSweepAxis_arg, &transientTime_arg, &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            // Shared = (3N + 2N² + nValues) * sizeof(numb) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions
                                                 + 2 * amountOfInitialConditions * amountOfInitialConditions
                                                 + amountOfValues)
                                                * sizeof(numb) * blockSize);

            LS_CHECK_CU(cuLaunchKernel(cached_ls.kernel_ls,
                                       gridSize, 1, 1, blockSize, 1, 1,
                                       shared, nullptr, args, nullptr),
                        "cuLaunchKernel(ls)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LS_CHECK(cudaDeviceSynchronize(), "sync after ls");

            LS_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(numb),
                                cudaMemcpyDeviceToHost),
                     "memcpy h_lsResult");
            LS_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);

                // первая экспонента используется как ground-truth для флага
                double first = h_lsResult[k * (size_t)amountOfInitialConditions + 0];
                // 999/-999 — kernel-sentinel расходимости; FP LS не различает.
                res.flags[global_idx] = (first == 999.0 || first == -999.0) ? REGIME_UNBOUND
                                                                            : REGIME_OSCILLATION;

                auto& row = res.spectrum[global_idx];
                for (int j = 0; j < amountOfInitialConditions; ++j) {
                    row[j] = h_lsResult[k * (size_t)amountOfInitialConditions + j];
                }

                if (out.is_open())
                    data_export::write_ls1d_row(out, param_val, row.data(),
                                                amountOfInitialConditions);
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LS_CHECK
        #undef LS_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_ls_2d_if_needed — отдельный шаблон ls2d.template.cu, тот же kernel
    // LSKernelCUDA. Ключ кэша помечен ":ls2d:" — изолирован от ":ls:" slot'а.
    bool compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls_2d, key, cached_ls_2d, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_ls_2d), "ls2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_ls_2d — спектр Ляпунова на сетке n_pts × n_pts: гибрид run_ls_1d (per-system буфер на N
    // экспонент) и run_lle_2d (swap_xy, chunking по ячейкам). Kernel возвращает N значений на
    // ячейку; D2H копирует cur_limiter * N doubles, host распаковывает по плоскостям —
    // values[k*n*n + iy*n + ix] = k-я экспонента в ячейке (ix, iy).
    LS2DResult run_ls_2d(const LS2DRequest& req) {
        LS2DResult res;
        auto fail = [&](const std::string& msg) -> LS2DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of the allowed range");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.NT <= 0.0)          return fail("NT must be > 0");
        if (req.eps <= 0.0)         return fail("eps must be > 0");

        // par_or_var + swap_xy: точная копия логики из run_lle_2d.
        auto check_param = [&](int p1based) -> bool {
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;
        int  hSweepAxis = -1;
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;  // см. run_lle_2d

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0 (X axis)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale requires param lo/hi > 0 (Y axis)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h and sweep_over_h_2 cannot both be true");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // См. run_lle_2d -- симметрично по слотам, swap_xy не нужен.
            hSweepAxis = req.sweep_over_h ? 0 : 1;
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);

            if (req.sweep_over_h) {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index_2))   return fail("param_index_2 (Y axis) out of range");
                    idx_axis_y = req.param_index_2;
                } else {
                    if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
                    idx_axis_y = req.var_sweep_index_2;
                }
                idx_axis_x = 0;
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (par_or_var == 1) {
                if (!check_param(req.param_index))    return fail("param_index (X axis) out of range");
                if (!check_param(req.param_index_2))  return fail("param_index_2 (Y axis) out of range");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))    return fail("var_sweep_index (X axis) out of range");
                if (!check_var(req.var_sweep_index_2))  return fail("var_sweep_index_2 (Y axis) out of range");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
            if (!check_param(req.param_index_2))   return fail("param_index_2 (Y axis) out of range");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
            if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_ls_2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x,
                                                    (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT too small)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;
        const int N = amountOfInitialConditions;

        // Memory budget — мирор run_ls_1d (агрессивно делит /16, т.к. per-system
        // память ~N). total_cells заменяет nPts.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lsResult (N numb): траектория
        // не хранится. Прежняя формула умножала это на длину блока и брала
        // 1/16 свободной памяти — вместе это резало чанк без причины.
        size_t perSystemBytes = sizeof(numb) * (size_t)N;
        if (perSystemBytes == 0) perSystemBytes = sizeof(numb);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)             nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)    nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult — nPtsLimiter × N row-major (как в run_ls_1d).
        std::vector<numb> h_lsResult(nPtsLimiter * (size_t)N);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lsResult          = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };
            sig.release();

        #define LS2_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS2_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS2_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LS2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LS2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LS2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)N * sizeof(numb)),                        "cudaMalloc d_initialConditions");
        LS2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LS2_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)N * sizeof(numb)),          "cudaMalloc d_lsResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerPoint;

        LS2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                       cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                          cudaMemcpyHostToDevice), "memcpy d_indices");
        LS2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)N * sizeof(numb),               cudaMemcpyHostToDevice), "memcpy d_ic");
        LS2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_values");
        LS2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        res.n_pts       = nPts;
        res.n_exponents = N;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.param_lo_2  = req.param_lo_2;
        res.param_hi_2  = req.param_hi_2;
        res.values.assign((size_t)N * total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + N);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts       = nPts;
        res.snapshot.n_exponents = N;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_ls2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LS2_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: тот же расчёт что в run_ls_1d (32K shared / per-thread), cap — настройка.
            const size_t sharedPerThread = (size_t)(3 * N + 2 * N * N + amountOfValues) * sizeof(numb);
            int blockSizeMax = (int)(32000 / (double)sharedPerThread);
            int blockSize = blockSizeMax;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap) blockSize = blockSizeCap;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = N;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult, &hSweepAxis_arg, &transientTime_arg, &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)((3 * N + 2 * N * N + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LS2_CHECK_CU(cuLaunchKernel(cached_ls_2d.kernel_ls,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(ls2d)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LS2_CHECK(cudaDeviceSynchronize(), "sync after ls2d");

            LS2_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                 cur_limiter * (size_t)N * sizeof(numb),
                                 cudaMemcpyDeviceToHost),
                      "memcpy h_lsResult");
            LS2_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // Распаковка: для каждой ячейки чанка — N экспонент. Layout
            // values[k * total_cells + out_idx] — k-я плоскость contiguous.
            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                // Первая экспонента — ground-truth для флага (как в run_ls_1d).
                double first = h_lsResult[k * (size_t)N + 0];
                int flag = (first == 999.0 || first == -999.0) ? REGIME_UNBOUND
                                                               : REGIME_OSCILLATION;
                res.flags[out_idx] = flag;
                for (int j = 0; j < N; ++j) {
                    double v = h_lsResult[k * (size_t)N + j];
                    res.values[(size_t)j * total_cells + out_idx] = v;
                }
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_ls2d_cells(out, nPts, N, res.values.data());
            }
        }

        // Per-plane min/max — autoscale colormap'а в GUI выбирает их по
        // активной экспоненте. Diverged-ячейки (flag<0) и 999/-999/NaN
        // исключаются.
        res.min_val.assign((size_t)N, 0.0);
        res.max_val.assign((size_t)N, 0.0);
        for (int j = 0; j < N; ++j) {
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < total_cells; ++c) {
                if (!regime_is_oscillation(res.flags[c])) continue;
                double v = res.values[(size_t)j * total_cells + c];
                if (!std::isfinite(v))         continue;
                if (v == 999.0 || v == -999.0) continue;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            if (std::isfinite(vmin) && std::isfinite(vmax)) {
                res.min_val[j] = vmin;
                res.max_val[j] = vmax;
            }
        }

        cleanup();
        #undef LS2_CHECK
        #undef LS2_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_bif1d_cont_if_needed — отдельный модуль (single-thread sequential
    // continuation kernel + peakFinderCUDA). Cache key с суффиксом :cont.
    bool compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":cont";
        return compile_into(pool_cont, key, cached_cont, activate, [&](CachedContModule& fresh) {
            // bifurcation1dContinuationKernel — extern "C" (имя не мангается), поэтому в name_exprs
            // не идёт и берётся из модуля напрямую. peakFinderCUDA и DFT_custom — обычные C++
            // символы, нужны mangled-варианты; DFT_custom уже в этом модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) и нужен continuation-ветке run_dft_1d.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_cont), "bifurcation1d_cont.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "peakFinderCUDA", "DFT_custom" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "bifurcation1dContinuationKernel", fresh.kernel_cont, err))
                { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[0], fresh.kernel_peak, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dft,  err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // compile_simple_cont_if_needed — общий компилятор для одноядерных
    // continuation-модулей (LLE и LS). От compile_bif1d_cont_if_needed
    // отличается только тем, что регистрировать mangled-имена не нужно:
    // единственный нужный символ объявлен extern "C".
    bool compile_simple_cont_if_needed(ModuleLru<CachedSimpleContModule>& pool,
                                       CachedSimpleContModule& slot,
                                       const std::string& tmpl,
                                       const char* kernel_name,
                                       const char* src_name,
                                       const char* key_suffix,
                                       const std::string& krs_body, int amountOfX,
                                       std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + key_suffix;
        return compile_into(pool, key, slot, true, [&](CachedSimpleContModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;   // пуст: единственный символ — extern "C"
            if (!build_module(snapshot_sources(tmpl), src_name,
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              {},
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, kernel_name, fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_lle1d_continuation_gpu / run_ls1d_continuation_gpu — GPU-двойники
    // CPU-веток (см. run_lle1d_cpu / run_ls1d_cpu). Тот же «костыль», что у
    // Bifurcation1D: single-thread kernel, потому что continuation — цепочка.
    // Медленнее CPU, нужны для сверки и как привычный способ считать на GPU.
    LLE1DResult run_lle1d_continuation_gpu(const LLE1DRequest& req) {
        LLE1DResult res;
        auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_lle_cont, cached_lle_cont, src_template_lle_cont,
                                           "lle1dContinuationKernel", "lle1d_cont.cu", ":llecont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int nPts = req.n_pts;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_result     = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_result)     cudaFree(d_result);
            sig.release();
        };
        #define LC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define LC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        LC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        LC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),  "cudaMalloc baseX");
        LC_CHECK(cudaMalloc((void**)&d_result,     (size_t)nPts * sizeof(numb)),           "cudaMalloc result");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        LC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        LC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = req.amountOfX;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, NT_arg = req.NT, tMax_arg = req.t_max;
        numb transientTime_arg = req.transient_time, eps_arg = req.eps, maxValue_arg = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &NT_arg, &tMax_arg, &transientTime_arg,
            &eps_arg, &maxValue_arg, &d_result
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        LC_CHECK_CU(cuLaunchKernel(cached_lle_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(lle cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        LC_CHECK(cudaDeviceSynchronize(), "sync after lle cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> host((size_t)nPts);
        LC_CHECK(cudaMemcpy(host.data(), d_result, (size_t)nPts * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy result");
        cleanup();
        #undef LC_CHECK
        #undef LC_CHECK_CU

        res.n_pts    = nPts;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.lyapunov.assign(host.begin(), host.end());   // numb -> double
        res.flags.assign(nPts, 0);
        // NaN у continuation-ядра означает расходимость (FP оно не различает).
        for (int j = 0; j < nPts; ++j)
            res.flags[j] = std::isfinite(res.lyapunov[j]) ? REGIME_OSCILLATION : REGIME_UNBOUND;
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }

    LS1DResult run_ls1d_continuation_gpu(const LS1DRequest& req) {
        LS1DResult res;
        auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_ls_cont, cached_ls_cont, src_template_ls_cont,
                                           "ls1dContinuationKernel", "ls1d_cont.cu", ":lscont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int nPts = req.n_pts;
        const int N    = req.amountOfX;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_result     = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_result)     cudaFree(d_result);
            sig.release();
        };
        #define SC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        SC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)N * sizeof(numb)),              "cudaMalloc baseX");
        SC_CHECK(cudaMalloc((void**)&d_result,     (size_t)nPts * N * sizeof(numb)),       "cudaMalloc result");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        SC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        SC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)N * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = N;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, NT_arg = req.NT, tMax_arg = req.t_max;
        numb transientTime_arg = req.transient_time, eps_arg = req.eps, maxValue_arg = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &NT_arg, &tMax_arg, &transientTime_arg,
            &eps_arg, &maxValue_arg, &d_result
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SC_CHECK_CU(cuLaunchKernel(cached_ls_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(ls cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SC_CHECK(cudaDeviceSynchronize(), "sync after ls cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> host((size_t)nPts * N);
        SC_CHECK(cudaMemcpy(host.data(), d_result, host.size() * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy result");
        cleanup();
        #undef SC_CHECK
        #undef SC_CHECK_CU

        res.n_pts       = nPts;
        res.n_exponents = N;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.spectrum.assign(nPts, std::vector<double>((size_t)N, 0.0));
        res.flags.assign(nPts, 0);
        for (int j = 0; j < nPts; ++j) {
            bool ok = true;
            for (int k = 0; k < N; ++k) {
                const double v = host[(size_t)j * N + k];
                res.spectrum[j][k] = v;
                if (!std::isfinite(v)) ok = false;
            }
            // NaN у continuation-ядра = расходимость (FP оно не различает).
            res.flags[j] = ok ? REGIME_OSCILLATION : REGIME_UNBOUND;
        }
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }

    // run_dft1d_continuation_gpu — выделенное ядро (dft1d_cont.template.cu),
    // а не связка bifurcation1dContinuationKernel + DFT_custom: под h-свипом у
    // каждой точки своя длина блока, свой шаг и своё окно, чего DFT_custom
    // выразить не может (у него один sizeOfBlock/h/window на запуск).
    Dft1DResult run_dft1d_continuation_gpu(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_dft_cont, cached_dft_cont, src_template_dft_cont,
                                           "dft1dContinuationKernel", "dft1d_cont.cu", ":dftcont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
        const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts  = req.n_pts;
        const int nFreq = req.n_freq;

        numb* d_baseValues = nullptr; numb* d_baseX = nullptr;
        numb* d_data = nullptr; numb* d_ak = nullptr; numb* d_bk = nullptr;
        int*    d_flags = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_data)       cudaFree(d_data);
            if (d_ak)         cudaFree(d_ak);
            if (d_bk)         cudaFree(d_bk);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define DC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define DC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        const size_t freqCells = (size_t)nPts * (size_t)nFreq;
        DC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        DC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),  "cudaMalloc baseX");
        // Один блок, а не nPts x block: kernel однопоточный, больше не нужно.
        DC_CHECK(cudaMalloc((void**)&d_data,       (size_t)maxPointsInBlock * sizeof(numb)), "cudaMalloc data");
        DC_CHECK(cudaMalloc((void**)&d_ak,         freqCells * sizeof(numb)),                "cudaMalloc AkCOS");
        DC_CHECK(cudaMalloc((void**)&d_bk,         freqCells * sizeof(numb)),                "cudaMalloc BkSIN");
        DC_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                "cudaMalloc flags");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        DC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        DC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = req.amountOfX;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, tMax_arg = req.t_max, transientTime_arg = req.transient_time;
        int    preScaller_arg = req.pre_scaller, writableVar_arg = req.writable_var;
        numb maxValue_arg = req.max_value;
        int    nFreq_arg = nFreq;
        numb freqLo_arg = req.freq_lo, freqHi_arg = req.freq_hi;
        int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
        int    windowType_arg = req.window_type;
        int    maxBlock_arg = maxPointsInBlock;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &preScaller_arg, &writableVar_arg, &maxValue_arg,
            &nFreq_arg, &freqLo_arg, &freqHi_arg, &logFreqAxis_arg, &windowType_arg,
            &maxBlock_arg, &d_data, &d_ak, &d_bk, &d_flags
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        DC_CHECK_CU(cuLaunchKernel(cached_dft_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(dft cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        DC_CHECK(cudaDeviceSynchronize(), "sync after dft cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> ak_host(freqCells), bk_host(freqCells);
        res.flags.assign(nPts, 0);
        DC_CHECK(cudaMemcpy(ak_host.data(), d_ak, freqCells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy AkCOS");
        DC_CHECK(cudaMemcpy(bk_host.data(), d_bk, freqCells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy BkSIN");
        res.ak_cos.assign(ak_host.begin(), ak_host.end());   // numb -> double
        res.bk_sin.assign(bk_host.begin(), bk_host.end());
        DC_CHECK(cudaMemcpy(res.flags.data(),  d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        cleanup();
        #undef DC_CHECK
        #undef DC_CHECK_CU

        res.n_pts   = nPts;
        res.n_freq  = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = req.freq_lo;
        res.freq_hi  = req.freq_hi;
        res.continuation_reverse = req.continuation_reverse;
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }

    // run_dft1d_hsweep_gpu — классический (без continuation) свип по шагу h. Отдельно от
    // run_dft1d_classical, потому что связка calculateDiscreteModelCUDA + DFT_custom тут
    // неприменима: DFT_custom берёт одну длину блока, одно готовое окно и один шаг дискретизации
    // на весь запуск, а под h-свипом все три — per-point. Ядро dft1dHSweepKernel делает траекторию
    // и DFT в одном месте, поток на точку (в отличие от однопоточного dft1dContinuationKernel:
    // здесь точки независимы).
    // Буферы считаются по worst case — наименьшему h диапазона, дающему больше всего сэмплов; так
    // же поступают run_dft1d_cpu и run_dft1d_continuation_gpu. Chunking по nPtsLimiter — как в
    // run_dft1d_classical / run_bif1d.
    Dft1DResult run_dft1d_hsweep_gpu(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        // Валидация — как в run_dft1d_classical, минус param_index/var_sweep_index:
        // свипуемая величина здесь h, индексы параметра/НУ не участвуют вовсе
        // (именно поэтому старый путь ломался на системах без параметров).
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        // param_lo/hi (= границы h) проверены на положительность в run_dft_1d.

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_dft_hsweep, cached_dft_hsweep, src_template_dft_hsweep,
                                           "dft1dHSweepKernel", "dft1d_hsweep.cu", ":dfthsweep",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int    nPts  = req.n_pts;
        const int    nFreq = req.n_freq;
        const double worstCaseH = (req.param_lo < req.param_hi) ? req.param_lo : req.param_hi;
        if (worstCaseH <= 0.0) return fail("h lo/hi must be > 0 when sweeping over dt (h)");

        // Самый мелкий шаг диапазона задаёт длину буфера. Считаем в double и
        // проверяем до сужения в int: t_max/h_lo легко перевалит за 2^31, и без
        // этой проверки пользователь получил бы отрицательный размер вместо
        // внятного сообщения.
        const double maxBlockD = std::ceil(req.t_max / worstCaseH / (double)req.pre_scaller);
        if (!(maxBlockD >= 1.0))
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");
        if (maxBlockD > 2.0e9)
            return fail("h lo too small: block > 2e9 samples (raise h lo, t_max or pre_scaller)");
        const int maxPointsInBlock = (int)maxBlockD;

        // --- Memory budget: как в run_dft1d_classical, но на систему берётся
        // worst-case длина блока. d_window тут нет вовсе — окно ядро считает
        // на лету, его длина у каждой точки своя.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        const size_t memPerSystem =
            (size_t)maxPointsInBlock * sizeof(numb) +   // d_data
            2 * (size_t)nFreq * sizeof(numb) +           // d_AkCOS + d_BkSIN
            sizeof(int);                                   // d_flags
        const size_t memConstants =
            (size_t)req.amountOfX * sizeof(numb) +
            req.base_values.size() * sizeof(numb);
        constexpr double SAFETY_FACTOR = 0.9;
        const size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        const size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter == 0)
            return fail("not enough GPU memory even for a single h-sweep point "
                        "(block " + std::to_string(maxPointsInBlock) + " samples)");
        if (nPtsLimiter > (size_t)nPts) nPtsLimiter = (size_t)nPts;
        const size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_AkCOS(nPtsLimiter * (size_t)nFreq);
        std::vector<numb> h_BkSIN(nPtsLimiter * (size_t)nFreq);
        std::vector<int>  h_flags(nPtsLimiter);

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_data       = nullptr;
        numb* d_AkCOS      = nullptr;
        numb* d_BkSIN      = nullptr;
        int*  d_flags      = nullptr;

        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_data)       cudaFree(d_data);
            if (d_AkCOS)      cudaFree(d_AkCOS);
            if (d_BkSIN)      cudaFree(d_BkSIN);
            if (d_flags)      cudaFree(d_flags);
        };

        #define DH_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DH_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DH_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        const int amountOfValues = (int)req.base_values.size();

        DH_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)),        "cudaMalloc baseValues");
        DH_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),         "cudaMalloc baseX");
        DH_CHECK(cudaMalloc((void**)&d_data,       nPtsLimiter * (size_t)maxPointsInBlock * sizeof(numb)), "cudaMalloc d_data");
        DH_CHECK(cudaMalloc((void**)&d_AkCOS,      nPtsLimiter * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_AkCOS");
        DH_CHECK(cudaMalloc((void**)&d_BkSIN,      nPtsLimiter * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_BkSIN");
        DH_CHECK(cudaMalloc((void**)&d_flags,      nPtsLimiter * sizeof(int)),                      "cudaMalloc d_flags");

        DH_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        DH_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice),  "memcpy baseX");
        DH_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Снимок — как у classical; indexOfMutVar = -1 помечает "свипуется h,
        // а не элемент a[]/X[]" (data_export пишет его как есть).
        res.snapshot.values.assign(req.base_values.begin(), req.base_values.end());
        res.snapshot.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
        res.snapshot.tMax          = req.t_max;
        res.snapshot.transientTime = req.transient_time;
        res.snapshot.h             = req.h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = req.pre_scaller;
        res.snapshot.writableVar   = req.writable_var;
        res.snapshot.indexOfMutVar = -1;
        res.snapshot.range_lo      = req.param_lo;
        res.snapshot.range_hi      = req.param_hi;
        res.snapshot.n_freq        = nFreq;
        res.snapshot.freq_lo       = req.freq_lo;
        res.snapshot.freq_hi       = req.freq_hi;
        res.snapshot.window_type   = req.window_type;

        const std::string& OUT_FILE_PATH = req.csv_output_path;
        constexpr int set_precision = 15;
        std::ofstream akFile, bkFile;
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            akFile.open(OUT_FILE_PATH + "_AkCOS.csv");
            bkFile.open(OUT_FILE_PATH + "_BkSIN.csv");
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            akFile.close();
            bkFile.close();
        }

        res.n_pts    = nPts;
        res.n_freq   = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = req.freq_lo;
        res.freq_hi  = req.freq_hi;
        res.flags.assign(nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        const size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            DH_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // Ядро h-свипа динамическую shared не берёт (launch с 0) — настройка идёт как есть.
            const int blockSize = launch_block_size(0);
            const int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_arg                 = nPts;
            int    nPtsLimiter_arg          = (int)nPtsLimiter;
            int    amountOfCalculatedPoints = (int)(iter * originalNPtsLimiter);
            numb   lo_arg                   = (numb)req.param_lo;
            numb   hi_arg                   = (numb)req.param_hi;
            int    logScale_arg             = req.log_scale ? 1 : 0;
            int    amountOfValues_arg       = amountOfValues;
            int    amountOfX_arg            = req.amountOfX;
            numb   tMax_arg                 = (numb)req.t_max;
            numb   transientTime_arg        = (numb)req.transient_time;
            int    preScaller_arg           = req.pre_scaller;
            int    writableVar_arg          = req.writable_var;
            numb   maxValue_arg             = (numb)req.max_value;
            int    nFreq_arg                = nFreq;
            numb   freqLo_arg               = (numb)req.freq_lo;
            numb   freqHi_arg               = (numb)req.freq_hi;
            int    logFreqAxis_arg          = req.freq_log_scale ? 1 : 0;
            int    windowType_arg           = req.window_type;
            int    maxBlock_arg             = maxPointsInBlock;

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &amountOfCalculatedPoints,
                &lo_arg, &hi_arg, &logScale_arg,
                &d_baseValues, &amountOfValues_arg,
                &d_baseX, &amountOfX_arg,
                &tMax_arg, &transientTime_arg,
                &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &nFreq_arg, &freqLo_arg, &freqHi_arg, &logFreqAxis_arg, &windowType_arg,
                &maxBlock_arg, &d_data, &d_AkCOS, &d_BkSIN, &d_flags
            };

            DH_CHECK_CU(cuLaunchKernel(cached_dft_hsweep.kernel,
                                       gridSize, 1, 1, blockSize, 1, 1,
                                       0, nullptr, args, nullptr),
                        "cuLaunchKernel(dft h-sweep)");
            DH_CHECK(cudaDeviceSynchronize(), "sync after dft h-sweep");

            DH_CHECK(cudaMemcpy(h_AkCOS.data(), d_AkCOS, nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_AkCOS");
            DH_CHECK(cudaMemcpy(h_BkSIN.data(), d_BkSIN, nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_BkSIN");
            DH_CHECK(cudaMemcpy(h_flags.data(), d_flags, nPtsLimiter * sizeof(int),                    cudaMemcpyDeviceToHost), "memcpy h_flags");
            DH_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            if (!OUT_FILE_PATH.empty()) {
                akFile.open(OUT_FILE_PATH + "_AkCOS.csv", std::ios::app);
                bkFile.open(OUT_FILE_PATH + "_BkSIN.csv", std::ios::app);
                if (akFile.is_open()) akFile << std::setprecision(set_precision);
                if (bkFile.is_open()) bkFile << std::setprecision(set_precision);
                const std::vector<double> akWide(h_AkCOS.begin(), h_AkCOS.end());
                const std::vector<double> bkWide(h_BkSIN.begin(), h_BkSIN.end());
                data_export::write_dft1d_matrix(akFile, akWide.data(), 0, (int)nPtsLimiter, nFreq);
                data_export::write_dft1d_matrix(bkFile, bkWide.data(), 0, (int)nPtsLimiter, nFreq);
                akFile.close();
                bkFile.close();
            }

            const size_t global_offset = originalNPtsLimiter * iter;
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                const size_t global_idx = global_offset + k;
                res.flags[global_idx] = h_flags[k];
                std::copy(h_AkCOS.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_AkCOS.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.ak_cos.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
                std::copy(h_BkSIN.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_BkSIN.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.bk_sin.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
            }
        }

        cleanup();
        #undef DH_CHECK
        #undef DH_CHECK_CU
        #undef DH_CANCEL_CHECK
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }

    // run_bif1d_continuation — single-thread sequential. Каждый параметр
    // стартует с конечного x[] предыдущего. Direction (forward/reverse)
    // обрабатывается в kernel'е. PeakFinderCUDA вызывается из того же модуля.
    Bifurcation1DResult run_bif1d_continuation(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)      return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                                     return fail("param_index out of range");
        // writable_var == -1 — sentinel "combination" (см. loopCalculateDiscreteModel_int).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                     return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err))
            return fail(err);

        // При h-свипе шаг меняется от точки к точке: буфер блока — под худший
        // случай (минимальный h), реальную длину каждой точки kernel сообщает
        // через d_actualIterations, и peakFinderCUDA сканирует только её.
        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
        const int amountOfPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (amountOfPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts = req.n_pts;
        numb* d_data       = nullptr;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        int*    d_amountOfPeaks = nullptr;
        numb* d_outPeaks   = nullptr;
        numb* d_timeOfPeaks= nullptr;
        int*    d_actualIterations = nullptr;

        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_data)       cudaFree(d_data);
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_amountOfPeaks) cudaFree(d_amountOfPeaks);
            if (d_outPeaks)   cudaFree(d_outPeaks);
            if (d_timeOfPeaks)cudaFree(d_timeOfPeaks);
            if (d_actualIterations) cudaFree(d_actualIterations);
            sig.release();
        };
        #define C_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define C_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)
        // Continuation kernel is monolithic (single launch sweeping all nPts),
        // so this can only catch cancellation BEFORE launch. After launch the
        // kernel runs to completion regardless.
        #define C_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const size_t dataBytes  = (size_t)nPts * (size_t)amountOfPointsInBlock * sizeof(numb);
        C_CHECK(cudaMalloc((void**)&d_data,         dataBytes),                                       "cudaMalloc d_data");
        C_CHECK(cudaMalloc((void**)&d_baseValues,   (size_t)req.base_values.size() * sizeof(numb)), "cudaMalloc d_baseValues");
        C_CHECK(cudaMalloc((void**)&d_baseX,        (size_t)req.amountOfX * sizeof(numb)),          "cudaMalloc d_baseX");
        C_CHECK(cudaMalloc((void**)&d_amountOfPeaks,(size_t)nPts * sizeof(int)),                      "cudaMalloc d_amountOfPeaks");
        C_CHECK(cudaMalloc((void**)&d_outPeaks,     dataBytes),                                       "cudaMalloc d_outPeaks");
        C_CHECK(cudaMalloc((void**)&d_timeOfPeaks,  dataBytes),                                       "cudaMalloc d_timeOfPeaks");
        C_CHECK(cudaMalloc((void**)&d_actualIterations, (size_t)nPts * sizeof(int)),                  "cudaMalloc d_actualIterations");

        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        C_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                           req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseValues");
        C_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                           (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseX");
        C_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Launch continuation kernel
        int    nPts_arg              = nPts;
        numb lo_arg                = req.param_lo;
        numb hi_arg                = req.param_hi;
        bool   reverse_arg           = req.continuation_reverse;
        int    logScale_arg          = req.log_scale ? 1 : 0;
        int    sweepIsH_arg          = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg       = req.param_index;
        int    amountOfValues_arg    = (int)req.base_values.size();
        int    amountOfX_arg         = req.amountOfX;
        numb h_arg                 = req.h;
        numb tMax_arg              = req.t_max;
        numb transientTime_arg     = req.transient_time;
        int    sizeOfBlock_arg       = amountOfPointsInBlock;
        int    preScaller_arg        = req.pre_scaller;
        int    writableVar_arg       = req.writable_var;
        numb maxValue_arg          = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* cont_args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg,
            &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfValues_arg,
            &d_baseX,      &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_data, &d_amountOfPeaks, &d_actualIterations,
            &d_cancel_arg, &d_progress_arg
        };
        C_CANCEL_CHECK();
        // Continuation is monolithic: progress jumps 0 -> 0.5 around the sweep
        // kernel, then -> 1.0 after the peak-finder below. Mid-kernel reporting
        // isn't possible without restructuring the kernel itself.
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_cont,
                                  1, 1, 1, 1, 1, 1, 0, nullptr, cont_args, nullptr),
                   "cuLaunchKernel(cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        C_CHECK(cudaDeviceSynchronize(), "sync after cont kernel");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        if (req.progress) req.progress->store(0.5f, std::memory_order_relaxed);

        // Launch peakFinderCUDA на полученные данные
        // Сигнатура: (numb* data, size_t sizeOfBlock, int amountOfBlocks,
        //             int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
        size_t sizeOfBlock_s = (size_t)amountOfPointsInBlock;
        int    nBlocks       = nPts;
        // numb, а не double: cuLaunchKernel копирует аргумент побайтово, а
        // peakFinderCUDA ждёт numb h (см. сигнатуру выше). При numb=float из
        // double уехали бы первые 4 байта, и межпиковые интервалы разъехались
        // бы на порядки. В классическом пути тот же аргумент объявлен numb.
        numb timeStep        = (numb)(req.h * (double)req.pre_scaller);
        // actualIterations — 8-й параметр peakFinderCUDA: реальная длина блока каждой точки (при
        // h-свипе она своя, буфер выделен под худший случай). Раньше аргумент не передавался вовсе:
        // cuLaunchKernel читал 8-й слот за концом peak_args[], kernel получал мусорный указатель и
        // разыменовывал его — "illegal memory access" сразу после peak-kernel'а.
        // 9-й и 10-й параметры peakFinderCUDA (peakStride/peakCapacity): 0/0 = legacy-раскладка.
        // Передаём явно по той же причине, что и actualIterations выше.
        size_t peakStride_arg   = 0;
        int    peakCapacity_arg = 0;
        // 11th parameter: emit every iterate instead of peaks (discrete maps).
        bool   emitAllSamples_arg = req.emit_all_samples;
        void* peak_args[] = {
            &d_data, &sizeOfBlock_s, &nBlocks,
            &d_amountOfPeaks, &d_outPeaks, &d_timeOfPeaks, &timeStep,
            &d_actualIterations,
            &peakStride_arg, &peakCapacity_arg, &emitAllSamples_arg
        };
        int    peakBlock = 32;
        int    peakGrid  = (nPts + peakBlock - 1) / peakBlock;
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_peak,
                                  peakGrid, 1, 1, peakBlock, 1, 1,
                                  0, nullptr, peak_args, nullptr),
                   "cuLaunchKernel(peak)");
        C_CHECK(cudaDeviceSynchronize(), "sync after peak");

        // D2H
        std::vector<numb> h_outPeaks   ((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<numb> h_timeOfPeaks((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks((size_t)nPts);
        C_CHECK(cudaMemcpy(h_outPeaks.data(),    d_outPeaks,    dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_outPeaks");
        C_CHECK(cudaMemcpy(h_timeOfPeaks.data(), d_timeOfPeaks, dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_timeOfPeaks");
        C_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy out h_amountOfPeaks");
        C_CHECK(cudaDeviceSynchronize(), "sync after D2H");

        // Заполнение Result
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.param_lo     = req.param_lo;
        res.param_hi     = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});
        for (int j = 0; j < nPts; ++j) {
            int n = h_amountOfPeaks[j];
            res.flags[j] = n;
            if (n > 0) {
                if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
                const numb* pr = h_outPeaks.data()    + (size_t)j * amountOfPointsInBlock;
                const numb* tr = h_timeOfPeaks.data() + (size_t)j * amountOfPointsInBlock;
                res.bifurcation_points[j].assign(pr, pr + n);
                res.peak_times[j].assign(tr, tr + n);
                // h-свип: peakFinderCUDA умножал разности индексов на общий h,
                // а шаг этой точки — cont_sweep_value. Интервал линеен по h,
                // поэтому поправка точная (см. тот же приём в run_bif1d).
                if (req.sweep_over_h) {
                    const double h_point = cont_sweep_value(j, nPts, req.param_lo, req.param_hi,
                                                            req.continuation_reverse, req.log_scale);
                    const double scale = h_point / req.h;
                    if (scale != 1.0)
                        for (double& v : res.peak_times[j]) v *= scale;
                }
            }
        }

        cleanup();
        #undef C_CHECK
        #undef C_CHECK_CU
        res.ok = true;
        return res;
    }

    // run_dft_1d — 1D DFT (порт bifurcation_DFT_1D из hostLibrary.cu:4900-5315).
    // Диспетчер: continuation требует param-sweep (та же причина, что и у
    // run_bif1d) — делегирует в run_dft1d_continuation, иначе classical.
    Dft1DResult run_dft_1d(const Dft1DRequest& req) {
        // Ограничения и порядок — как у Bif/LLE/LS 1D: continuation требует
        // param- или h-свипа (цепочка по IC бессмысленна), log-сетка по
        // параметру требует положительных границ. CPU-ветки не трогают CUDA,
        // поэтому стоят до ensure_init внутри самих функций.
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
            Dft1DResult r;
            r.error = "log scale requires param lo/hi > 0";
            return r;
        }
        // h-свип: границы — это сам шаг, нулевой/отрицательный шаг не считается
        // ни на одном из путей. Проверка здесь, а не в каждой ветке, как у
        // log_scale выше (см. те же проверки в run_bif1d / run_lle_1d).
        if (req.sweep_over_h && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
            Dft1DResult r;
            r.error = "h lo/hi must be > 0 when sweeping over dt (h)";
            return r;
        }
        if (req.continuation) {
            if (req.sweep_over_var) {
                Dft1DResult r;
                r.error = "continuation requires a param sweep, not an IC sweep";
                return r;
            }
            return req.use_cpu ? run_dft1d_cpu(req, /*continuation*/ true)
                               : run_dft1d_continuation_gpu(req);
        }
        if (req.use_cpu) return run_dft1d_cpu(req, /*continuation*/ false);
        // Классический GPU-свип по h идёт отдельным ядром: связка
        // calculateDiscreteModelCUDA + DFT_custom тут не работает, потому что
        // DFT_custom принимает одну длину блока, одно готовое окно и один шаг
        // дискретизации на весь запуск, а под h-свипом все три — per-point.
        if (req.sweep_over_h) return run_dft1d_hsweep_gpu(req);
        return run_dft1d_classical(req);
    }

    // Общая для classical/continuation: строит оконную функцию длиной sizeOfBlock. DFT_custom
    // принимает готовое окно аргументом и НЕ считает его сам. Формулы и нумерация типов — в
    // cpu_build_window выше (была отдельная копия тех же формул; после добавления Blackman и
    // Blackman-Harris держать две копии в одном файле смысла нет).
    static void build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) {
        cpu_build_window(out, sizeOfBlock, window_type);
    }

    // run_dft1d_classical — порт classical-ветки bifurcation_DFT_1D. Реюзает
    // cached.kernel_traj (calculateDiscreteModelCUDA, тот же PTX-кэш что и
    // run_bif1d) для генерации сырых траекторий, затем cached.kernel_dft
    // (DFT_custom) вместо peakFinderCUDA. Chunked по nPtsLimiter как run_bif1d.
    Dft1DResult run_dft1d_classical(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        // валидация (как run_bif1d + n_freq/freq range)
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index out of range");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index out of range");
        }
        // writable_var == -1 — sentinel "combination" (см. loopCalculateDiscreteModel_int
        // / Bifurcation1DRequest::writable_var) — тот же calculateDiscreteModelCUDA,
        // так что DFT1D поддерживает её точно так же.
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_if_needed(req.krs_body, req.amountOfX,
                               req.sweep_over_var ? 0 : 1, err)) return fail(err);

        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const int    nFreq                      = req.n_freq;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        numb   rangesFreq[2]                    = { (numb)req.freq_lo, (numb)req.freq_hi };
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Memory budget: как run_bif1d, но выход DFT (AkCOS/BkSIN, nPtsLimiter*n_freq каждый)
        // обычно намного меньше, чем outPeaks/timeOfPeaks (nPtsLimiter*amountOfPointsInBlock) —
        // n_freq почти всегда << amountOfPointsInBlock. d_window константен и с nPtsLimiter не
        // масштабируется.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        size_t memPerSystem =
            (size_t)amountOfPointsInBlock * sizeof(numb) +      // d_data
            2 * (size_t)nFreq * sizeof(numb) +                   // d_AkCOS + d_BkSIN (per-point share)
            sizeof(int);                                           // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(numb) +                                    // d_ranges
            2 * sizeof(numb) +                                    // d_rangesFreq
            sizeof(int) +                                           // d_indicesOfMutVars
            (size_t)amountOfInitialConditions * sizeof(numb) +
            (size_t)amountOfValues * sizeof(numb) +
            (size_t)amountOfPointsInBlock * sizeof(numb);         // d_window
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        // Округления вниз до кратного blockSize_setup здесь больше нет — по той же причине, что в
        // run_bif1d: ядра сами отсекают лишние потоки, а при n_pts < 32 округление давало 0, и Run
        // падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("n_pts must be > 0");
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_AkCOS(nPtsLimiter * (size_t)nFreq);
        std::vector<numb> h_BkSIN(nPtsLimiter * (size_t)nFreq);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);
        std::vector<numb> h_window;
        build_window(h_window, amountOfPointsInBlock, req.window_type);

        numb* d_data              = nullptr;
        numb* d_ranges            = nullptr;
        numb* d_rangesFreq        = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        numb* d_AkCOS             = nullptr;
        numb* d_BkSIN             = nullptr;
        numb* d_window            = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_data)              cudaFree(d_data);
            if (d_ranges)            cudaFree(d_ranges);
            if (d_rangesFreq)        cudaFree(d_rangesFreq);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_AkCOS)             cudaFree(d_AkCOS);
            if (d_BkSIN)             cudaFree(d_BkSIN);
            if (d_window)            cudaFree(d_window);
        };
            sig.release();

        #define DFT_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DFT_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DFT_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        DFT_CHECK(cudaMalloc((void**)&d_data,              nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(numb)), "cudaMalloc d_data");
        DFT_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                          "cudaMalloc d_ranges");
        DFT_CHECK(cudaMalloc((void**)&d_rangesFreq,        2 * sizeof(numb)),                                          "cudaMalloc d_rangesFreq");
        DFT_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        DFT_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),          "cudaMalloc d_initialConditions");
        DFT_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                     "cudaMalloc d_values");
        DFT_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");
        DFT_CHECK(cudaMalloc((void**)&d_AkCOS,             nPtsLimiter * (size_t)nFreq * sizeof(numb)),                "cudaMalloc d_AkCOS");
        DFT_CHECK(cudaMalloc((void**)&d_BkSIN,             nPtsLimiter * (size_t)nFreq * sizeof(numb)),                "cudaMalloc d_BkSIN");
        DFT_CHECK(cudaMalloc((void**)&d_window,            (size_t)amountOfPointsInBlock * sizeof(numb)),             "cudaMalloc d_window");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        DFT_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        DFT_CHECK(cudaMemcpy(d_rangesFreq,        rangesFreq,         2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_rangesFreq");
        DFT_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        DFT_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        DFT_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        DFT_CHECK(cudaMemcpy(d_window,            h_window.data(),   (size_t)amountOfPointsInBlock * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy d_window");
        DFT_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];
        res.snapshot.n_freq        = nFreq;
        res.snapshot.freq_lo       = rangesFreq[0];
        res.snapshot.freq_hi       = rangesFreq[1];
        res.snapshot.window_type   = req.window_type;

        std::ofstream akFile, bkFile;
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            akFile.open(OUT_FILE_PATH + "_AkCOS.csv");
            bkFile.open(OUT_FILE_PATH + "_BkSIN.csv");
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            akFile.close();
            bkFile.close();
        }

        res.n_pts  = nPts;
        res.n_freq = nFreq;
        res.param_lo = ranges[0];
        res.param_hi = ranges[1];
        res.freq_lo  = rangesFreq[0];
        res.freq_hi  = rangesFreq[1];
        res.flags.assign(nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            DFT_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // Обе стадии (траектория и DFT_custom) запускаются одной шириной, поэтому режем её
            // по shared более требовательной из них — траектории.
            int blockSize = launch_block_size(
                (size_t)ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb));
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t sizeOfBlock_s             = (size_t)amountOfPointsInBlock;  // for kernel_traj (size_t param)
            int    sizeOfBlock_i             = amountOfPointsInBlock;          // for DFT_custom (int param — see gotcha #3)
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            numb h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = !req.sweep_over_var;
            // h-свип сюда не доходит: run_dft_1d уводит его в run_dft1d_hsweep_gpu (DFT_custom
            // берёт одну длину блока, одно окно и один шаг на весь запуск — под per-point h этого
            // мало). Здесь ось всегда param/IC, поэтому hSweepAxis выключен, а actualIterations не
            // нужен: длина блока одна на весь запуск, в отличие от run_bif1d, где его читает
            // peakFinderCUDA.
            int    hSweepAxis_arg            = -1;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int*   d_actualIterations        = nullptr;
            // Лог-сетка по оси параметра — бит 0 (в 1D ось одна), как в run_bif1d.
            int    logAxisMask_arg           = req.log_scale ? 1 : 0;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args_traj[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &sizeOfBlock_s,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfInitialConditions_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_data,
                &d_amountOfPeaks,
                &par_or_var_arg,
                &hSweepAxis_arg,
                &transientTime_arg,
                &tMax_arg,
                &d_actualIterations,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);

            DFT_CHECK_CU(cuLaunchKernel(cached.kernel_traj,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_traj, nullptr),
                         "cuLaunchKernel(traj)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            DFT_CHECK(cudaDeviceSynchronize(), "sync after traj");
            DFT_CANCEL_CHECK();

            // DFT_custom(data, sizeOfBlock, amountOfBlocks, checkerArray, AkCOS,
            // BkSIN, rangesFreq, window, nFreq, h, logFreqAxis) — h здесь ШАГ
            // МЕЖДУ decimated сэмплами (h*preScaller), как и у peakFinderCUDA
            // (см. run_bif1d).
            int    nFreq_int      = nFreq;
            numb timeStep_arg   = h * (double)preScaller;
            int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
            void* args_dft[] = {
                &d_data,
                &sizeOfBlock_i,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_AkCOS,
                &d_BkSIN,
                &d_rangesFreq,
                &d_window,
                &nFreq_int,
                &timeStep_arg,
                &logFreqAxis_arg
            };
            DFT_CHECK_CU(cuLaunchKernel(cached.kernel_dft,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args_dft, nullptr),
                         "cuLaunchKernel(dft)");
            DFT_CHECK(cudaDeviceSynchronize(), "sync after dft");

            DFT_CHECK(cudaMemcpy(h_AkCOS.data(),         d_AkCOS,          nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_AkCOS");
            DFT_CHECK(cudaMemcpy(h_BkSIN.data(),         d_BkSIN,          nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_BkSIN");
            DFT_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks,  nPtsLimiter * sizeof(int),                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            DFT_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            if (!OUT_FILE_PATH.empty()) {
                akFile.open(OUT_FILE_PATH + "_AkCOS.csv", std::ios::app);
                bkFile.open(OUT_FILE_PATH + "_BkSIN.csv", std::ios::app);
                if (akFile.is_open()) akFile << std::setprecision(set_precision);
                if (bkFile.is_open()) bkFile << std::setprecision(set_precision);
                // Экспорт принимает double* — расширяем numb-буфер на месте.
                const std::vector<double> akWide(h_AkCOS.begin(), h_AkCOS.end());
                const std::vector<double> bkWide(h_BkSIN.begin(), h_BkSIN.end());
                data_export::write_dft1d_matrix(akFile, akWide.data(), 0, (int)nPtsLimiter, nFreq);
                data_export::write_dft1d_matrix(bkFile, bkWide.data(), 0, (int)nPtsLimiter, nFreq);
                akFile.close();
                bkFile.close();
            }

            size_t global_offset = originalNPtsLimiter * iter;
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = global_offset + k;
                res.flags[global_idx] = h_amountOfPeaks[k];
                std::copy(h_AkCOS.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_AkCOS.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.ak_cos.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
                std::copy(h_BkSIN.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_BkSIN.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.bk_sin.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
            }
        }

        cleanup();
        #undef DFT_CHECK
        #undef DFT_CHECK_CU
        #undef DFT_CANCEL_CHECK
        res.ok = true;
        return res;
    }

    // run_dft1d_continuation — реюзает bifurcation1dContinuationKernel как есть: он уже пишет
    // полную decimated-траекторию в d_data и флаги в d_amountOfPeaks для ВСЕГО nPts за один
    // монолитный запуск (kernels/bifurcation1d_cont.template.cu). Второй проход — DFT_custom вместо
    // peakFinderCUDA над теми же буферами. Монолитно, без chunking, как run_bif1d_continuation.
    Dft1DResult run_dft1d_continuation(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)      return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                                     return fail("param_index out of range");
        // writable_var == -1 — sentinel "combination" (см. run_bif1d_continuation).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                     return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err))
            return fail(err);

        const int amountOfPointsInBlock = (int)(req.t_max / req.h / req.pre_scaller);
        // amountOfPointsForSkip здесь не нужен: в этой ветке транзиент уходит в
        // ядро как время (transientTime_arg), и число шагов ядро считает само —
        // по своему hLocal, который в h-свипе меняется от точки к точке.
        if (amountOfPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts  = req.n_pts;
        const int nFreq = req.n_freq;
        numb   rangesFreq[2] = { (numb)req.freq_lo, (numb)req.freq_hi };

        std::vector<numb> h_window;
        build_window(h_window, amountOfPointsInBlock, req.window_type);

        numb* d_data       = nullptr;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        int*    d_amountOfPeaks = nullptr;
        numb* d_AkCOS      = nullptr;
        numb* d_BkSIN      = nullptr;
        numb* d_rangesFreq = nullptr;
        numb* d_window     = nullptr;

        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_data)       cudaFree(d_data);
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_amountOfPeaks) cudaFree(d_amountOfPeaks);
            if (d_AkCOS)      cudaFree(d_AkCOS);
            if (d_BkSIN)      cudaFree(d_BkSIN);
            if (d_rangesFreq) cudaFree(d_rangesFreq);
            if (d_window)     cudaFree(d_window);
            sig.release();
        };
        #define DFTC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define DFTC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)
        #define DFTC_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const size_t dataBytes = (size_t)nPts * (size_t)amountOfPointsInBlock * sizeof(numb);
        DFTC_CHECK(cudaMalloc((void**)&d_data,         dataBytes),                                       "cudaMalloc d_data");
        DFTC_CHECK(cudaMalloc((void**)&d_baseValues,   (size_t)req.base_values.size() * sizeof(numb)), "cudaMalloc d_baseValues");
        DFTC_CHECK(cudaMalloc((void**)&d_baseX,        (size_t)req.amountOfX * sizeof(numb)),          "cudaMalloc d_baseX");
        DFTC_CHECK(cudaMalloc((void**)&d_amountOfPeaks,(size_t)nPts * sizeof(int)),                      "cudaMalloc d_amountOfPeaks");
        DFTC_CHECK(cudaMalloc((void**)&d_AkCOS,        (size_t)nPts * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_AkCOS");
        DFTC_CHECK(cudaMalloc((void**)&d_BkSIN,        (size_t)nPts * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_BkSIN");
        DFTC_CHECK(cudaMalloc((void**)&d_rangesFreq,   2 * sizeof(numb)),                               "cudaMalloc d_rangesFreq");
        DFTC_CHECK(cudaMalloc((void**)&d_window,       (size_t)amountOfPointsInBlock * sizeof(numb)),  "cudaMalloc d_window");

        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        DFTC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                           req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseValues");
        DFTC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                           (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseX");
        DFTC_CHECK(cudaMemcpy(d_rangesFreq, rangesFreq, 2 * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_rangesFreq");
        DFTC_CHECK(cudaMemcpy(d_window, h_window.data(),
                           (size_t)amountOfPointsInBlock * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_window");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Launch continuation kernel (тот же, что у run_bif1d_continuation)
        // У DFT1D нет h-свипа и log-сетки по параметру, поэтому соответствующие
        // флаги ядра выключены, а actualIterations не нужен (длина блока одна
        // на все точки) — передаём nullptr, ядро это допускает.
        int    nPts_arg              = nPts;
        numb lo_arg                = req.param_lo;
        numb hi_arg                = req.param_hi;
        bool   reverse_arg           = req.continuation_reverse;
        int    logScale_arg          = 0;
        int    sweepIsH_arg          = 0;
        int    mutParamIdx_arg       = req.param_index;
        int    amountOfValues_arg    = (int)req.base_values.size();
        int    amountOfX_arg         = req.amountOfX;
        numb h_arg                 = req.h;
        numb tMax_arg              = req.t_max;
        numb transientTime_arg     = req.transient_time;
        int    sizeOfBlock_arg       = amountOfPointsInBlock;
        int    preScaller_arg        = req.pre_scaller;
        int    writableVar_arg       = req.writable_var;
        numb maxValue_arg          = req.max_value;
        int*   d_actualIterations    = nullptr;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* cont_args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg,
            &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfValues_arg,
            &d_baseX,      &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_data, &d_amountOfPeaks, &d_actualIterations,
            &d_cancel_arg, &d_progress_arg
        };
        DFTC_CANCEL_CHECK();
        DFTC_CHECK_CU(cuLaunchKernel(cached_cont.kernel_cont,
                                  1, 1, 1, 1, 1, 1, 0, nullptr, cont_args, nullptr),
                   "cuLaunchKernel(cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after cont kernel");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        if (req.progress) req.progress->store(0.5f, std::memory_order_relaxed);

        // DFT_custom вместо peakFinderCUDA, над теми же d_data/d_amountOfPeaks
        int    sizeOfBlock_i   = amountOfPointsInBlock;  // int, не size_t — см. gotcha #3
        int    nFreq_int       = nFreq;
        // numb, а не double — DFT_custom ждёт numb h; см. пояснение в
        // run_bif1d_continuation.
        numb timeStep          = (numb)(req.h * (double)req.pre_scaller);
        int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
        void* dft_args[] = {
            &d_data, &sizeOfBlock_i, &nPts_arg,
            &d_amountOfPeaks, &d_AkCOS, &d_BkSIN, &d_rangesFreq, &d_window,
            &nFreq_int, &timeStep, &logFreqAxis_arg
        };
        int    dftBlock = 32;
        int    dftGrid  = (nPts + dftBlock - 1) / dftBlock;
        DFTC_CHECK_CU(cuLaunchKernel(cached_cont.kernel_dft,
                                  dftGrid, 1, 1, dftBlock, 1, 1,
                                  0, nullptr, dft_args, nullptr),
                   "cuLaunchKernel(dft)");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after dft");

        res.n_pts  = nPts;
        res.n_freq = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = rangesFreq[0];
        res.freq_hi  = rangesFreq[1];
        res.continuation_reverse = req.continuation_reverse;
        res.flags.assign((size_t)nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        std::vector<int> h_amountOfPeaks((size_t)nPts);
        {   // Приёмник D2H — numb, наружу расширяем в double-хранилище Result.
            const size_t cells_ = (size_t)nPts * (size_t)nFreq;
            std::vector<numb> ak_(cells_), bk_(cells_);
            DFTC_CHECK(cudaMemcpy(ak_.data(), d_AkCOS, cells_ * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy out d_AkCOS");
            DFTC_CHECK(cudaMemcpy(bk_.data(), d_BkSIN, cells_ * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy out d_BkSIN");
            res.ak_cos.assign(ak_.begin(), ak_.end());
            res.bk_sin.assign(bk_.begin(), bk_.end());
        }
        DFTC_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy out d_amountOfPeaks");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after D2H");
        for (int j = 0; j < nPts; ++j) res.flags[(size_t)j] = h_amountOfPeaks[(size_t)j];
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);

        if (!req.csv_output_path.empty()) {
            res.snapshot.values = req.base_values;
            res.snapshot.initial_conditions = req.initial_conditions;
            res.snapshot.tMax          = req.t_max;
            res.snapshot.transientTime = req.transient_time;
            res.snapshot.h             = req.h;
            res.snapshot.gpu_fmad      = get_nvrtc_fmad();
            res.snapshot.preScaller    = req.pre_scaller;
            res.snapshot.writableVar   = req.writable_var;
            res.snapshot.indexOfMutVar = req.param_index;
            res.snapshot.range_lo      = req.param_lo;
            res.snapshot.range_hi      = req.param_hi;
            res.snapshot.n_freq        = nFreq;
            res.snapshot.freq_lo       = rangesFreq[0];
            res.snapshot.freq_hi       = rangesFreq[1];
            res.snapshot.window_type   = req.window_type;

            std::ofstream cfg(req.csv_output_path + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            std::ofstream akFile(req.csv_output_path + "_AkCOS.csv");
            std::ofstream bkFile(req.csv_output_path + "_BkSIN.csv");
            akFile << std::setprecision(15);
            bkFile << std::setprecision(15);
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            data_export::write_dft1d_matrix(akFile, res.ak_cos.data(), 0, nPts, nFreq);
            data_export::write_dft1d_matrix(bkFile, res.bk_sin.data(), 0, nPts, nFreq);
        }

        cleanup();
        #undef DFTC_CHECK
        #undef DFTC_CHECK_CU
        #undef DFTC_CANCEL_CHECK
        res.ok = true;
        return res;
    }

    // compile_bif2d_if_needed — шаблон bifurcation2d.template.cu, два kernel'а:
    // calculateDiscreteModelPeaksCUDA + dbscanCUDA.
    // Cache key: ":bif2d:" + par_or_var.
    bool compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":bif2d:" + std::to_string(par_or_var);
        return compile_into(pool_bif2d, key, cached_bif2d, activate, [&](CachedBif2dModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_bif2d), "bifurcation2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelPeaksCUDA", "dbscanCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dbscan, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_bif2d — порт NonLinAnal::bifurcation2D (hostLibrary.cu:898-1724).
    // Три kernel'а: calculateDiscreteModelCUDA → peakFinderCUDA → dbscanCUDA.
    // Результат: число DBSCAN-кластеров на ячейку = период системы.
    // [ADAPT]: те же адаптации, что у run_bif1d + run_lle_2d (см. их комментарии).
    Bifurcation2DResult run_bif2d(const Bifurcation2DRequest& req) {
        Bifurcation2DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation2DResult& { res.error = msg; return res; };

        // валидация
        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.n_pts <= 0)          return fail("n_pts must be > 0");
        if (req.h <= 0.0)            return fail("h must be > 0");
        if (req.t_max <= 0.0)        return fail("t_max must be > 0");
        if (req.transient_time < 0)  return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)    return fail("pre_scaller must be > 0");
        if (req.eps_dbscan <= 0.0)   return fail("eps_dbscan must be > 0");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");

        // par_or_var + swap_xy (та же логика что у run_lle_2d)
        auto check_param = [&](int p1based) -> bool {
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int    idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool   swap_xy = false;
        int    hSweepAxis = -1;
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;  // см. run_lle_2d

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale requires param lo/hi > 0 (X axis)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale requires param lo/hi > 0 (Y axis)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h and sweep_over_h_2 cannot both be true");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // См. run_lle_2d -- симметрично по слотам, swap_xy не нужен.
            hSweepAxis = req.sweep_over_h ? 0 : 1;
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);

            if (req.sweep_over_h) {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index_2))   return fail("param_index_2 (Y axis) out of range");
                    idx_axis_y = req.param_index_2;
                } else {
                    if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
                    idx_axis_y = req.var_sweep_index_2;
                }
                idx_axis_x = 0;
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0;
            }
            // Требование "> 0" относится ТОЛЬКО к оси, которая свипает h: это шаг интегрирования,
            // на него ниже делит worstCaseH, и нулевой либо отрицательный шаг не считается. Вторая
            // ось — параметр или НУ, где отрицательные значения совершенно законны (свип sigma от
            // -5 до 5 к шагу отношения не имеет). Раньше условие требовало > 0 от ОБЕИХ осей и
            // заворачивало такой прогон сообщением про h. У run_lle_2d / run_ls_2d проверки нет вовсе.
            const double h_lo = req.sweep_over_h ? req.param_lo : req.param_lo_2;
            const double h_hi = req.sweep_over_h ? req.param_hi : req.param_hi_2;
            if (h_lo <= 0.0 || h_hi <= 0.0)
                return fail(std::string("h lo/hi must be > 0 when sweeping over dt (h) (axis ")
                            + (req.sweep_over_h ? "X" : "Y") + ")");
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (par_or_var == 1) {
                if (!check_param(req.param_index))   return fail("param_index (X axis) out of range");
                if (!check_param(req.param_index_2)) return fail("param_index_2 (Y axis) out of range");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (X axis) out of range");
                if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index))  return fail("var_sweep_index (X axis) out of range");
            if (!check_param(req.param_index_2))  return fail("param_index_2 (Y axis) out of range");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            par_or_var = par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                       req.sweep_over_var, req.sweep_over_var_2);
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (Y axis) out of range");
            if (!check_param(req.param_index))     return fail("param_index (X axis) out of range");
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        // локальные переменные (порт hostLibrary.cu:bifurcation2D)
        const int    nPts                       = req.n_pts;
        const double tMax                       = req.t_max;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x, (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        // Множители осей DBSCAN (пик / межпиковый интервал) — per-diagram.
        const double mult_peak_arg              = req.mult_peak;
        const double mult_interval_arg          = req.mult_interval;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // Ширина запуска переехала в настройку (Settings -> GPU launch, launch_block_size).
        // Локальное 128 здесь было прототипом этой идеи; замер показал, что на крупной 2D-сетке
        // ширина безразлична (+-7%), а на мелкой 128 проигрывает — дефолт поэтому 32.

        // dt-sweep: буфер должен вмещать худший случай (минимальный h в
        // диапазоне h-оси) -- см. run_bif1d.
        double worstCaseH = (hSweepAxis == 0) ? ranges[0] : (hSweepAxis == 1) ? ranges[2] : h;
        int amountOfPointsInBlock = (int)std::ceil(tMax / worstCaseH / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Буферы пиков больше НЕ длиной в траекторию: peakFinder всё равно отдаёт
        // не больше max_amount_of_peaks. "+1" — стадия межпиковых интервалов съедает
        // один пик, поэтому для полной строки нужно max_amount_of_peaks + 1 сырых.
        // При eps_interPeak_delta == 0 (дефолт) результат бит-в-бит такой же, как
        // у прежнего неограниченного скана (сжатие сдвигает ряд ровно на 1).
        const size_t peakStride    = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)amountOfPointsInBlock
                                   : (size_t)max_amount_of_peaks + 1;
        // dbscan держит в своей строке и метки кластеров, и стек обхода (см. dbscan()).
        const size_t helpfulStride = 2 * peakStride;
        const int    peakCapacity  = (int)peakStride;

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — порт hostLibrary.cu:930-950.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        // Бюджет больше не зависит от длины траектории вообще: сэмплы
        // потребляются на лету (PeakStream), в памяти остаются только пики.
        // tMax/h теперь влияет на время счёта, но не на число ячеек в чанке.
        size_t baseMemPerSystem = peakStride * 2 * sizeof(numb)                  // d_outPeaks + d_intervals
                                + helpfulStride * sizeof(numb)                   // d_helpfulArray
                                + 2 * sizeof(int);                               // d_amountOfPeaks + d_dbscanResult
        size_t memConstants     = (4 + (size_t)amountOfInitialConditions + (size_t)amountOfValues) * sizeof(numb) + 2 * sizeof(int);
        if (memConstants >= freeMemory) return fail("not enough GPU memory for constants");

        size_t nPtsLimiter = (freeMemory - memConstants) / baseMemPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)             nPtsLimiter = total_cells;
        // Округления вниз до кратного blockSize_setup здесь больше нет — по той же причине, что в
        // run_bif1d: ядра сами отсекают лишние потоки, а при числе ячеек сетки < 32 округление
        // давало 0, и Run падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("the grid is empty (n_pts must be > 0)");

        size_t originalNPtsLimiter = nPtsLimiter;

        // Прогресс считается по СЧЁТУ, а не по готовым ячейкам: все ячейки
        // делают одну и ту же работу и финишируют пачкой, так что счётчик
        // готовых ячеек при сетке в одну волну прыгал бы сразу с 0 на 100%.
        // Шаг кратен CHECK_INTERVAL (тики ставятся в уже существующей проверке)
        // и выбран так, чтобы ячейка отчиталась около 64 раз за всю работу:
        // этого хватает для гладкого бара и не создаёт давки на одном адресе.
        const size_t stepsPerCell = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;

        // host buffers
        std::vector<int> h_dbscanResult(nPtsLimiter);

        // device buffers
        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        numb* d_outPeaks          = nullptr;
        numb* d_intervals         = nullptr;
        numb* d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        RunSignals sig;   // прогресс и отмена в mapped-памяти

        // Dedicated stream for traj→peak→dbscan within each chunk. Avoids
        // per-kernel cudaDeviceSynchronize, which on Windows/WDDM lets the GPU
        // downclock between launches; on the same stream kernels are ordered
        // implicitly and the GPU stays under continuous load.
        CUstream stream = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_outPeaks)          cudaFree(d_outPeaks);
            if (d_intervals)         cudaFree(d_intervals);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            sig.release();
            if (stream)              cuStreamDestroy(stream);
        };

        #define BIF2D_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF2D_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF2D_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BIF2D_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                           "cudaMalloc d_ranges");
        BIF2D_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BIF2D_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),           "cudaMalloc d_initialConditions");
        BIF2D_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                      "cudaMalloc d_values");
        BIF2D_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BIF2D_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_outPeaks");
        BIF2D_CHECK(cudaMalloc((void**)&d_intervals,         nPtsLimiter * peakStride * sizeof(numb)),                    "cudaMalloc d_intervals");
        BIF2D_CHECK(cudaMalloc((void**)&d_helpfulArray,      nPtsLimiter * helpfulStride * sizeof(numb)),                 "cudaMalloc d_helpfulArray");
        BIF2D_CHECK(cudaMalloc((void**)&d_dbscanResult,      nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_dbscanResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        BIF2D_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF2D_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF2D_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF2D_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),             cudaMemcpyHostToDevice), "memcpy d_values");
        BIF2D_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        BIF2D_CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells,  0);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.eps_dbscan    = eps_dbscan;
        res.snapshot.mult_peak     = mult_peak_arg;
        res.snapshot.mult_interval = mult_interval_arg;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts     = nPts;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_bif2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // Главный цикл по чанкам (порт hostLibrary.cu:1212-1692)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BIF2D_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // Ширина блока — настройка, ужатая под 48 KB shared на блок (launch_block_size):
            // широкая система (много X и a[]) при 128 потоках в потолок не влезет, и запуск
            // упал бы на CUDA_ERROR_INVALID_VALUE.
            const size_t sharedPerThread = (size_t)ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb);
            int blockSize = launch_block_size(sharedPerThread);
            int gridSize  = (int)((cur_limiter + blockSize - 1) / blockSize);

            // 1. calculateDiscreteModelPeaksCUDA (dimension=2): интегрирование и поиск
            // пиков в одном ядре — траектория никуда не пишется.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = (par_or_var != 0);  // true=param, false=IC (runtime hint, kernel использует compile-time макрос)
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int    logAxisMask_arg           = logAxisMask;
            size_t peakStride_arg            = peakStride;
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            int    peakCapacity_arg          = peakCapacity;
            bool   emitAllSamples_arg        = req.emit_all_samples;

            void* args_fused[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_outPeaks,
                &d_intervals,
                &d_amountOfPeaks,
                &par_or_var_arg,
                &hSweepAxis_arg,
                &transientTime_arg,
                &tMax_arg,
                &logAxisMask_arg,
                &peakStride_arg,
                &peakCapacity_arg,
                &d_cancel_arg,
                &d_progress_arg,
                &progressStride_arg,
                &emitAllSamples_arg
            };
            sig.resetTicks();
            unsigned int shared_traj = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_fused,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          shared_traj, stream, args_fused, nullptr),
                           "cuLaunchKernel(bif2d traj+peaks)");

            // 3. dbscanCUDA — same stream, ordered after peak.
            // Множители осей признаков передаём явно: при запуске через driver
            // API дефолты из объявления не подставляются (см. cudaLibrary.cuh).
            numb eps_arg      = eps_dbscan;
            numb mult_pk_arg  = mult_peak_arg;
            numb mult_int_arg = mult_interval_arg;
            size_t helpfulStride_arg = helpfulStride;
            // sizeOfBlock у dbscanCUDA остаётся ради legacy-вызовов; здесь раскладку
            // задают peakStride/helpfulStride, и этот аргумент не читается.
            size_t sizeOfBlock_s = peakStride;
            void* args_dbscan[] = {
                &d_outPeaks,          // кластеризуем пики, а не сырую траекторию
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_intervals,
                &d_helpfulArray,
                &eps_arg,
                &d_dbscanResult,
                &mult_pk_arg,
                &mult_int_arg,
                &peakStride_arg,
                &helpfulStride_arg
            };
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_dbscan,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          0, stream, args_dbscan, nullptr),
                           "cuLaunchKernel(bif2d dbscan)");

            // D2H: только d_dbscanResult (число кластеров = период).
            // Async on the same stream + single sync — keeps the GPU continuously
            // loaded across the whole chunk instead of inserting 3 sync gaps.
            if (!wait_with_signals(stream, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BIF2D_CHECK(cudaStreamSynchronize(stream), "sync stream before host loop");
            BIF2D_CANCEL_CHECK();

            // Копируем ПОСЛЕ опроса, а не до него: device -> pageable копирование
            // асинхронным не бывает (h_dbscanResult — std::vector), драйвер возвращает
            // управление лишь после завершения копии, т.е. всей очереди потока. Стоя
            // перед циклом, оно блокировало хост до конца ядра, и цикл не выполнялся
            // ни разу — ни прогресса, ни отмены.
            BIF2D_CHECK(cudaMemcpy(h_dbscanResult.data(), d_dbscanResult,
                                   cur_limiter * sizeof(int), cudaMemcpyDeviceToHost),
                        "memcpy h_dbscanResult");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                int    period     = h_dbscanResult[k];
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                // period из dbscanCUDA — сырой REGIME_*/период: -1 = fixed
                // point, 0 = unbound, N > 0 = число кластеров пиков. Раньше
                // unbound (0) получал flag 1, т.е. считался валидной ячейкой и
                // попадал в автошкалу colormap'а как «период 0».
                res.values[out_idx] = (double)period;
                res.flags[out_idx]  = period;
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_bif2d_grid(out, nPts, res.values.data());
            }
        }

        // Авто-нормализация colormap. Верх шкалы — по осцилляционным ячейкам (там значение =
        // период), а низ опускаем до кода режима, если такие ячейки на карте есть: -1 (fixed point)
        // и 0 (unbound) — это не «период -1/0», а отдельные состояния, и в res.values они лежат как
        // есть. Раньше в шкалу шли только осцилляции, поэтому vmin был >= 1 и оба режима прижимались
        // к самому дну — неотличимо от периода 1.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        bool has_fp = false, has_unbound = false;
        for (size_t k = 0; k < total_cells; ++k) {
            const int f = res.flags[k];
            if (f == REGIME_FIXED_POINT) { has_fp      = true; continue; }
            if (f == REGIME_UNBOUND)     { has_unbound = true; continue; }
            if (!regime_is_oscillation(f)) continue;
            double v = res.values[k];
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        res.min_val = std::isfinite(vmin) ? vmin : 0.0;
        res.max_val = std::isfinite(vmax) ? vmax : 0.0;
        // Приоритет у fixed point: он ниже unbound, и одной нижней границы
        // хватает на оба режима сразу.
        if      (has_fp)      res.min_val = -1.0;
        else if (has_unbound) res.min_val =  0.0;

        cleanup();
        #undef BIF2D_CHECK
        #undef BIF2D_CHECK_CU
        res.ok = true;
        return res;
    }

    // compile_basins_if_needed — отдельный шаблон basins.template.cu, регистрирует 5 kernel'ов:
    // calculateDiscreteModelCUDA, avgPeakFinderCUDA и три DBSCAN-kernel'а (cluster,
    // search_fixed_points, search_clear_points). Cache-ключ помечен `:basins`; par_or_var=0
    // захардкожен в шаблоне, поэтому ключа не требует — только хэш krs_body/amountOfX.
    bool compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err, bool activate = true) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":basins";
        return compile_into(pool_basins, key, cached_basins, activate, [&](CachedBasinsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(src_template_basins), "basins.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "calculateDiscreteModelAvgPeaksCUDA",
                                "CUDA_dbscan_kernel",
                                "CUDA_dbscan_search_fixed_points_kernel",
                                "CUDA_dbscan_search_clear_points_kernel" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused,        err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dbscan,       err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_search_fixed, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_search_clear, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // run_basins — порт NonLinAnal::basinsOfAttraction + CUDA_dbscan host-loop (hostLibrary.cu).
    // Структура: 1) traj+avg_peak chunked loop по ячейкам; 2) host-DBSCAN — цикл по точкам, на
    // каждой итерации search_fixed/search_clear + cluster_kernel с расширением через
    // neighbors-список. Возвращает basin_idx (cluster IDs), avg_peaks, avg_intervals, helpful_array.
    BasinsResult run_basins(const BasinsRequest& req) {
        BasinsResult res;
        auto fail = [&](const std::string& msg) -> BasinsResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                  return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of the allowed range");
        if ((int)req.initial_conditions.size() != req.amountOfX)   return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)      return fail("too many base_values");
        if (req.axis_x_var < 0 || req.axis_x_var >= req.amountOfX) return fail("axis_x_var out of range");
        if (req.axis_y_var < 0 || req.axis_y_var >= req.amountOfX) return fail("axis_y_var out of range");
        if (req.axis_x_var == req.axis_y_var)                      return fail("axis_x_var == axis_y_var (pick different variables)");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        if (req.eps_dbscan <= 0.0)  return fail("eps_dbscan must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_basins_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);

        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)req.axis_x_lo, (numb)req.axis_x_hi,
                                                    (numb)req.axis_y_lo, (numb)req.axis_y_hi };
        int    indicesOfMutVars[2]              = { req.axis_x_var, req.axis_y_var };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double tMax                       = req.t_max;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // blockSize_setup здесь больше нет: ширина запуска берётся из настройки
        // (launch_block_size), а полом бюджета памяти у бассейнов он не служил.
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // По памяти чанк ничем не ограничен: буферы, росшие вместе с ним,
        // ушли вместе с траекторией, а выходные массивы выделяются на всю сетку.
        size_t nPtsLimiter = total_cells;

        size_t originalNPtsLimiter = nPtsLimiter;

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        int*    d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        RunSignals sig;   // прогресс и отмена в mapped-памяти
        numb* d_avgPeaks          = nullptr;
        numb* d_avgIntervals      = nullptr;
        int*    d_amountOfNeighbors = nullptr;
        int*    d_neighbors         = nullptr;
        int*    d_clearIdx          = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            sig.release();
            if (d_avgPeaks)          cudaFree(d_avgPeaks);
            if (d_avgIntervals)      cudaFree(d_avgIntervals);
            if (d_amountOfNeighbors) cudaFree(d_amountOfNeighbors);
            if (d_neighbors)         cudaFree(d_neighbors);
            if (d_clearIdx)          cudaFree(d_clearIdx);
        };

        #define BAS_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BAS_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                           "cudaMalloc d_ranges");
        BAS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BAS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),           "cudaMalloc d_initialConditions");
        BAS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                      "cudaMalloc d_values");
        BAS_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_helpfulArray,      total_cells * sizeof(int)),                                    "cudaMalloc d_helpfulArray");
        BAS_CHECK(cudaMalloc((void**)&d_dbscanResult,      total_cells * sizeof(int)),                                    "cudaMalloc d_dbscanResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        const size_t stepsPerCell   = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;
        BAS_CHECK(cudaMalloc((void**)&d_avgPeaks,          total_cells * sizeof(numb)),                                 "cudaMalloc d_avgPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_avgIntervals,      total_cells * sizeof(numb)),                                 "cudaMalloc d_avgIntervals");
        BAS_CHECK(cudaMalloc((void**)&d_amountOfNeighbors, sizeof(int)),                                                  "cudaMalloc d_amountOfNeighbors");
        BAS_CHECK(cudaMalloc((void**)&d_neighbors,         total_cells * sizeof(int)),                                    "cudaMalloc d_neighbors");
        BAS_CHECK(cudaMalloc((void**)&d_clearIdx,          sizeof(int)),                                                  "cudaMalloc d_clearIdx");

        BAS_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BAS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BAS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BAS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),             cudaMemcpyHostToDevice), "memcpy d_values");
        BAS_CHECK(cudaMemset(d_dbscanResult, 0, total_cells * sizeof(int)),  "memset d_dbscanResult");
        BAS_CHECK(cudaMemset(d_helpfulArray, 0, total_cells * sizeof(int)),  "memset d_helpfulArray");
        BAS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Snapshot CSV-relevant fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.eps_dbscan    = eps_dbscan;
        res.snapshot.writableVar   = req.writable_var;
        res.snapshot.axis_x_var    = req.axis_x_var;
        res.snapshot.axis_y_var    = req.axis_y_var;
        res.snapshot.axis_x_lo     = req.axis_x_lo;
        res.snapshot.axis_x_hi     = req.axis_x_hi;
        res.snapshot.axis_y_lo     = req.axis_y_lo;
        res.snapshot.axis_y_hi     = req.axis_y_hi;
        res.snapshot.n_pts         = nPts;
        res.snapshot.feature1      = req.feature1;
        res.snapshot.feature2      = req.feature2;
        res.snapshot.mult1         = (double)req.mult1;
        res.snapshot.mult2         = (double)req.mult2;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_basins_config(cfg, res.snapshot);
        }

        // Pre-write the 2-line ranges header into each of the 4 data files.
        // Body is appended after compute (see the post-loop block below).
        if (!OUT_FILE_PATH.empty()) {
            auto write_ranges = [&](const std::string& path) {
                std::ofstream o(path);
                data_export::write_basins_ranges(o, res.snapshot);
            };
            write_ranges(OUT_FILE_PATH);
            write_ranges(OUT_FILE_PATH + "_1.csv");
            write_ranges(OUT_FILE_PATH + "_2.csv");
            write_ranges(OUT_FILE_PATH + "_3.csv");
        }

        // 1. Цикл по chunk'ам: traj-kernel + avg-peak-kernel
        // Two-phase progress: phase 1 here (sim), phase 2 (DBSCAN) below. Each
        // phase reports its own 0..1 fraction; GUI shows the phase label.
        if (req.progress_phase) req.progress_phase->store(1, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BAS_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: ceil(48K / ((N + nValues) * sizeof(numb))), cap — настройка
            // (Settings -> GPU launch); формула остаётся вторым потолком, см. run_lle_1d.
            const size_t sharedPerThread = (size_t)(amountOfInitialConditions + amountOfValues) * sizeof(numb);
            int blockSize = (int)std::ceil((1024.0 * 48.0) / (double)sharedPerThread);
            if (blockSize < 1)                blockSize = 1;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap)     blockSize = blockSizeCap;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // calculateDiscreteModelAvgPeaksCUDA: интегрирование и фичи в одном ядре.
            // h-свипа и лог-осей у бассейнов нет в запросе вовсе, ядро зашивает их
            // константами -1 / 0 — ровно тем, что сюда передавалась раздельная пара.
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = req.writable_var;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = false;   // compile-time par_or_var=0 в шаблоне
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int  feature1_int = req.feature1;
            int  feature2_int = req.feature2;
            numb mult1_v      = req.mult1;
            numb mult2_v      = req.mult2;
            // Сигналы пока выключены: прогресс и Cancel у бассейнов ещё по чанкам.
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            sig.resetTicks();

            // Offset-указатели: чанк пишет в свою часть общей сетки.
            int*  d_helpful_chunk    = d_helpfulArray  + iter * originalNPtsLimiter;
            numb* d_avg_peak_chunk   = d_avgPeaks      + iter * originalNPtsLimiter;
            numb* d_avg_interv_chunk = d_avgIntervals  + iter * originalNPtsLimiter;

            void* args_basins[] = {
                &nPts_arg, &nPtsLimiter_int, &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s, &dimension_arg, &d_ranges, &h_arg,
                &d_indicesOfMutVars, &d_initialConditions, &amountOfIC_int,
                &d_values, &amountOfValues_int, &amountOfIterations_arg,
                &preScaller_int, &writableVar_int, &maxValue_arg,
                &d_avg_peak_chunk, &d_avg_interv_chunk, &d_helpful_chunk,
                &par_or_var_arg, &transientTime_arg, &tMax_arg,
                &feature1_int, &feature2_int, &mult1_v, &mult2_v,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };
            unsigned int shared_traj = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);
            BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_fused,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared_traj, nullptr, args_basins, nullptr),
                         "cuLaunchKernel(basins traj+features)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BAS_CHECK(cudaDeviceSynchronize(), "sync after traj+features");
            BAS_CANCEL_CHECK();
        }

        // 2. Host-DBSCAN: порт hostLibrary.cu:3066 (CUDA_dbscan).
        // Динамической shared у dbscan-ядер нет — настройка идёт как есть.
        int blockSize_db = launch_block_size(0);
        int gridSize_db  = (int)((total_cells + blockSize_db - 1) / blockSize_db);
        int amountOfData_int = (int)total_cells;

        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        std::vector<int> h_neighbors(total_cells, 0);
        int h_amountOfNeighbors      = 0;

        // Прогресс DBSCAN-фазы: внешний цикл идёт ОДИН РАЗ НА КЛАСТЕР (с ранним выходом, когда все
        // ячейки классифицированы), поэтому main_iter — плохая метрика: он доходит до ~10 против
        // миллионов ячеек. Вместо него после расширения каждого кластера считаем классифицированные
        // ячейки в d_dbscanResult и репортим долю. Троттлинг >200 мс между сканами держит накладные
        // расходы ограниченными на больших сетках (скан ~4 МБ D2H + линейный проход на 1М ячеек,
        // ~3-5 мс; с троттлингом это максимум ~5 сканов в секунду).
        std::vector<int> h_dbscan_check(total_cells, 0);
        auto last_progress_scan = std::chrono::steady_clock::now() - std::chrono::seconds(1);

        // Phase 2: DBSCAN expansion. Reset bar to 0..1, bump phase to 2.
        if (req.progress_phase) req.progress_phase->store(2, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        for (size_t main_iter = 0; main_iter < total_cells; ++main_iter) {
            BAS_CANCEL_CHECK();
            int clearIdx_init = -1;
            BAS_CHECK(cudaMemcpy(d_clearIdx, &clearIdx_init, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_clearIdx init");

            // search_fixed_points: проверяет, остались ли -1-точки без cluster'а.
            void* args_search[] = { &d_avgPeaks, &d_avgIntervals, &d_helpfulArray, &d_dbscanResult,
                                    &amountOfData_int, &d_clearIdx };
            BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_fixed,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_search, nullptr),
                         "cuLaunchKernel(search_fixed)");
            BAS_CHECK(cudaDeviceSynchronize(), "sync search_fixed");

            int clearIdx = -1;
            BAS_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H");

            int resultClusters = 0;
            if (clearIdx == -1) {
                // FP-точек не осталось — search_clear_points (Osc).
                BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_clear,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_search, nullptr),
                             "cuLaunchKernel(search_clear)");
                BAS_CHECK(cudaDeviceSynchronize(), "sync search_clear");
                BAS_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H 2");
                // Все ячейки классифицированы — нового seed'а нет, останавливаемся.
                // NB: инкремент идёт ПОСЛЕ проверки clearIdx. Референс NonLinAnal инкрементирует до
                // неё и сообщает на один кластер больше, но он возвращает void и счётчик никто не
                // читает; мы отдаём n_clusters в UI, где off-by-one был виден как «4 кластера» при
                // трёх реальных.
                if (clearIdx == -1) break;
                ++amountOfClusters;
                resultClusters = amountOfClusters;
            } else {
                --amountOfNegativeClusters;
                resultClusters = amountOfNegativeClusters;
            }

            // Reset amountOfNeighbors на каждой итерации.
            h_amountOfNeighbors = 0;
            BAS_CHECK(cudaMemcpy(d_amountOfNeighbors, &h_amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_amountOfNeighbors=0");

            // CUDA_dbscan_kernel — расширение cluster'а от clearIdx.
            numb eps_arg = eps_dbscan;
            void* args_db[] = {
                &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                &amountOfData_int, &eps_arg, &resultClusters,
                &d_amountOfNeighbors, &d_neighbors, &clearIdx, &d_helpfulArray
            };
            BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_db, nullptr),
                         "cuLaunchKernel(dbscan expand init)");
            BAS_CHECK(cudaDeviceSynchronize(), "sync dbscan init");

            BAS_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors D2H");
            if (h_amountOfNeighbors > 0)
                BAS_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                          "memcpy neighbors D2H");

            // Цикл расширения cluster'а по всем найденным соседям.
            for (size_t ni = 0; ni < (size_t)h_amountOfNeighbors; ++ni) {
                int neighbor_idx = h_neighbors[ni];
                void* args_db2[] = {
                    &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                    &amountOfData_int, &eps_arg, &resultClusters,
                    &d_amountOfNeighbors, &d_neighbors, &neighbor_idx, &d_helpfulArray
                };
                BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_db2, nullptr),
                             "cuLaunchKernel(dbscan expand neighbor)");
                BAS_CHECK(cudaDeviceSynchronize(), "sync dbscan neighbor");

                BAS_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors loop");
                if (h_amountOfNeighbors > 0)
                    BAS_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy neighbors loop");
            }

            // Report progress by counting classified cells. Throttled so the
            // D2H+scan happens at most ~5x/sec regardless of cluster count.
            if (req.progress) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_progress_scan >= std::chrono::milliseconds(200)) {
                    last_progress_scan = now;
                    BAS_CHECK(cudaMemcpy(h_dbscan_check.data(), d_dbscanResult,
                                         total_cells * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy dbscanResult D2H (progress)");
                    size_t classified = 0;
                    for (size_t k = 0; k < total_cells; ++k)
                        if (h_dbscan_check[k] != 0) ++classified;
                    req.progress->store(float(classified) / float(total_cells),
                                        std::memory_order_relaxed);
                }
            }
        }

        // 3. D2H результаты
        res.n_pts       = nPts;
        res.axis_x_lo   = req.axis_x_lo;
        res.axis_x_hi   = req.axis_x_hi;
        res.axis_y_lo   = req.axis_y_lo;
        res.axis_y_hi   = req.axis_y_hi;
        res.axis_x_var  = req.axis_x_var;
        res.axis_y_var  = req.axis_y_var;
        res.basin_idx.assign(total_cells, 0);
        res.avg_peaks.assign(total_cells, 0.0);
        res.avg_intervals.assign(total_cells, 0.0);
        res.helpful_array.assign(total_cells, 0);

        BAS_CHECK(cudaMemcpy(res.basin_idx.data(),     d_dbscanResult, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy basin_idx");
        {   // numb на устройстве -> double в Result.
            std::vector<numb> ap_(total_cells), ai_(total_cells);
            BAS_CHECK(cudaMemcpy(ap_.data(), d_avgPeaks,     total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_peaks");
            BAS_CHECK(cudaMemcpy(ai_.data(), d_avgIntervals, total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_intervals");
            res.avg_peaks.assign(ap_.begin(), ap_.end());
            res.avg_intervals.assign(ai_.begin(), ai_.end());
        }
        BAS_CHECK(cudaMemcpy(res.helpful_array.data(), d_helpfulArray, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy helpful_array");

        // 4. Сводки + CSV
        res.n_clusters       = amountOfClusters;
        res.min_cluster_idx  = amountOfNegativeClusters;

        // Per-field min/max (исключая 999/-999/NaN).
        auto compute_minmax = [&](const std::vector<double>& v, double& mn, double& mx) {
            mn = std::numeric_limits<double>::infinity();
            mx = -std::numeric_limits<double>::infinity();
            for (double x : v) {
                if (!std::isfinite(x)) continue;
                if (x == 999.0 || x == -999.0) continue;
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            if (!std::isfinite(mn) || !std::isfinite(mx)) { mn = 0.0; mx = 0.0; }
        };
        compute_minmax(res.avg_peaks,     res.avg_peaks_min,     res.avg_peaks_max);
        compute_minmax(res.avg_intervals, res.avg_intervals_min, res.avg_intervals_max);

        // Append the n_pts × n_pts grid for each of the 4 fields, after the
        // 2-line ranges header already written above. Layout matches the
        // shared writers data_export uses for the GUI right-click export.
        if (!OUT_FILE_PATH.empty()) {
            auto append_int = [&](const std::string& path, const int* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_int(o, nPts, data);
            };
            auto append_double = [&](const std::string& path, const double* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_double(o, nPts, data);
            };
            append_int   (OUT_FILE_PATH,             res.basin_idx.data());
            append_double(OUT_FILE_PATH + "_1.csv",  res.avg_peaks.data());
            append_double(OUT_FILE_PATH + "_2.csv",  res.avg_intervals.data());
            append_int   (OUT_FILE_PATH + "_3.csv",  res.helpful_array.data());
        }

        cleanup();
        #undef BAS_CHECK
        #undef BAS_CHECK_CU
        res.ok = true;
        return res;
    }

    // run_basins_recluster — DBSCAN-only прогон поверх кэшированных фич. Зеркалит фазу 2
    // (host-DBSCAN) из run_basins, но пропускает sim-фазу: d_avgPeaks/d_avgIntervals/d_helpfulArray
    // заливаются из request'а вместо повторного прогона avgPeakFinderCUDA. Мульты уже применены к
    // загруженным фичам (run_basins, фаза 1), поэтому здесь не нужны.
    BasinsReclusterResult run_basins_recluster(const BasinsReclusterRequest& req) {
        BasinsReclusterResult res;
        auto fail = [&](const std::string& msg) -> BasinsReclusterResult& {
            res.error = msg; return res;
        };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of the allowed range");
        if (req.n_pts <= 0)                                      return fail("n_pts must be > 0");
        if (req.eps_dbscan <= 0.0)                               return fail("eps_dbscan must be > 0");
        const size_t total_cells = (size_t)req.n_pts * (size_t)req.n_pts;
        if (req.avg_peaks.size()     != total_cells)             return fail("avg_peaks: size does not match n_pts²");
        if (req.avg_intervals.size() != total_cells)             return fail("avg_intervals: size does not match n_pts²");
        if (req.helpful_array.size() != total_cells)             return fail("helpful_array: size does not match n_pts²");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_basins_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);

        // Ширина запуска dbscan-ядер — из настройки (launch_block_size ниже).
        const double eps_dbscan = req.eps_dbscan;

        int*    d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        numb* d_avgPeaks          = nullptr;
        numb* d_avgIntervals      = nullptr;
        int*    d_amountOfNeighbors = nullptr;
        int*    d_neighbors         = nullptr;
        int*    d_clearIdx          = nullptr;

        auto cleanup = [&]() {
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            if (d_avgPeaks)          cudaFree(d_avgPeaks);
            if (d_avgIntervals)      cudaFree(d_avgIntervals);
            if (d_amountOfNeighbors) cudaFree(d_amountOfNeighbors);
            if (d_neighbors)         cudaFree(d_neighbors);
            if (d_clearIdx)          cudaFree(d_clearIdx);
        };

        #define BRC_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BRC_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BRC_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BRC_CHECK(cudaMalloc((void**)&d_helpfulArray,      total_cells * sizeof(int)),    "cudaMalloc d_helpfulArray");
        BRC_CHECK(cudaMalloc((void**)&d_dbscanResult,      total_cells * sizeof(int)),    "cudaMalloc d_dbscanResult");
        BRC_CHECK(cudaMalloc((void**)&d_avgPeaks,          total_cells * sizeof(numb)), "cudaMalloc d_avgPeaks");
        BRC_CHECK(cudaMalloc((void**)&d_avgIntervals,      total_cells * sizeof(numb)), "cudaMalloc d_avgIntervals");
        BRC_CHECK(cudaMalloc((void**)&d_amountOfNeighbors, sizeof(int)),                  "cudaMalloc d_amountOfNeighbors");
        BRC_CHECK(cudaMalloc((void**)&d_neighbors,         total_cells * sizeof(int)),    "cudaMalloc d_neighbors");
        BRC_CHECK(cudaMalloc((void**)&d_clearIdx,          sizeof(int)),                  "cudaMalloc d_clearIdx");

        BRC_CHECK(cudaMemcpy(d_avgPeaks,     to_numb(req.avg_peaks).data(),     total_cells * sizeof(numb), cudaMemcpyHostToDevice), "memcpy avg_peaks H2D");
        BRC_CHECK(cudaMemcpy(d_avgIntervals, to_numb(req.avg_intervals).data(), total_cells * sizeof(numb), cudaMemcpyHostToDevice), "memcpy avg_intervals H2D");
        BRC_CHECK(cudaMemcpy(d_helpfulArray, req.helpful_array.data(), total_cells * sizeof(int),    cudaMemcpyHostToDevice), "memcpy helpful_array H2D");
        BRC_CHECK(cudaMemset(d_dbscanResult, 0, total_cells * sizeof(int)), "memset d_dbscanResult");
        BRC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        const int blockSize_db = launch_block_size(0);   // dbscan-ядра без динамической shared
        const int gridSize_db  = (int)((total_cells + blockSize_db - 1) / blockSize_db);
        const int amountOfData_int = (int)total_cells;

        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        std::vector<int> h_neighbors(total_cells, 0);
        int h_amountOfNeighbors      = 0;
        std::vector<int> h_dbscan_check(total_cells, 0);
        auto last_progress_scan = std::chrono::steady_clock::now() - std::chrono::seconds(1);

        if (req.progress) req.progress->store(0.0f, std::memory_order_relaxed);

        for (size_t main_iter = 0; main_iter < total_cells; ++main_iter) {
            BRC_CANCEL_CHECK();
            int clearIdx_init = -1;
            BRC_CHECK(cudaMemcpy(d_clearIdx, &clearIdx_init, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_clearIdx init");

            void* args_search[] = { &d_avgPeaks, &d_avgIntervals, &d_helpfulArray, &d_dbscanResult,
                                    (void*)&amountOfData_int, &d_clearIdx };
            BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_fixed,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_search, nullptr),
                         "cuLaunchKernel(search_fixed)");
            BRC_CHECK(cudaDeviceSynchronize(), "sync search_fixed");

            int clearIdx = -1;
            BRC_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H");

            int resultClusters = 0;
            if (clearIdx == -1) {
                BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_search_clear,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_search, nullptr),
                             "cuLaunchKernel(search_clear)");
                BRC_CHECK(cudaDeviceSynchronize(), "sync search_clear");
                BRC_CHECK(cudaMemcpy(&clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost), "memcpy clearIdx D2H 2");
                if (clearIdx == -1) break;
                ++amountOfClusters;
                resultClusters = amountOfClusters;
            } else {
                --amountOfNegativeClusters;
                resultClusters = amountOfNegativeClusters;
            }

            h_amountOfNeighbors = 0;
            BRC_CHECK(cudaMemcpy(d_amountOfNeighbors, &h_amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice), "memcpy d_amountOfNeighbors=0");

            numb eps_arg = eps_dbscan;
            void* args_db[] = {
                &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                (void*)&amountOfData_int, &eps_arg, &resultClusters,
                &d_amountOfNeighbors, &d_neighbors, &clearIdx, &d_helpfulArray
            };
            BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                        gridSize_db, 1, 1, blockSize_db, 1, 1,
                                        0, nullptr, args_db, nullptr),
                         "cuLaunchKernel(dbscan expand init)");
            BRC_CHECK(cudaDeviceSynchronize(), "sync dbscan init");

            BRC_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors D2H");
            if (h_amountOfNeighbors > 0)
                BRC_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                          "memcpy neighbors D2H");

            for (size_t ni = 0; ni < (size_t)h_amountOfNeighbors; ++ni) {
                int neighbor_idx = h_neighbors[ni];
                void* args_db2[] = {
                    &d_avgPeaks, &d_avgIntervals, &d_dbscanResult,
                    (void*)&amountOfData_int, &eps_arg, &resultClusters,
                    &d_amountOfNeighbors, &d_neighbors, &neighbor_idx, &d_helpfulArray
                };
                BRC_CHECK_CU(cuLaunchKernel(cached_basins.kernel_dbscan,
                                            gridSize_db, 1, 1, blockSize_db, 1, 1,
                                            0, nullptr, args_db2, nullptr),
                             "cuLaunchKernel(dbscan expand neighbor)");
                BRC_CHECK(cudaDeviceSynchronize(), "sync dbscan neighbor");

                BRC_CHECK(cudaMemcpy(&h_amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost), "memcpy amountOfNeighbors loop");
                if (h_amountOfNeighbors > 0)
                    BRC_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, (size_t)h_amountOfNeighbors * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy neighbors loop");
            }

            if (req.progress) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_progress_scan >= std::chrono::milliseconds(200)) {
                    last_progress_scan = now;
                    BRC_CHECK(cudaMemcpy(h_dbscan_check.data(), d_dbscanResult,
                                         total_cells * sizeof(int), cudaMemcpyDeviceToHost),
                              "memcpy dbscanResult D2H (progress)");
                    size_t classified = 0;
                    for (size_t k = 0; k < total_cells; ++k)
                        if (h_dbscan_check[k] != 0) ++classified;
                    req.progress->store(float(classified) / float(total_cells),
                                        std::memory_order_relaxed);
                }
            }
        }

        res.n_pts = req.n_pts;
        res.basin_idx.assign(total_cells, 0);
        BRC_CHECK(cudaMemcpy(res.basin_idx.data(), d_dbscanResult, total_cells * sizeof(int), cudaMemcpyDeviceToHost), "memcpy basin_idx D2H");
        res.n_clusters      = amountOfClusters;
        res.min_cluster_idx = amountOfNegativeClusters;

        cleanup();
        #undef BRC_CHECK
        #undef BRC_CHECK_CU
        #undef BRC_CANCEL_CHECK
        res.ok = true;
        return res;
    }

    // FastSynchro — два режима.

    // Один helper для обоих режимов — параметризован шаблоном и списком
    // kernel-symbols. Ключ кэша включает все substituted-параметры
    // (type_of_synch, error_estim, fs_error_trs), чтобы при их смене recompile
    // действительно случился.
    bool compile_fs_module(const std::string& src_template,
                           const std::string& tag, // ":fs_attr" или ":fs_grid"
                           int amountOfX,
                           const std::string& krs_body,
                           int type_of_synch_v, int error_estim_v, double fs_error_trs_v,
                           const std::vector<const char*>& expr_kernels,
                           ModuleLru<CachedFastSyncModule>& pool,
                           CachedFastSyncModule& slot,
                           std::string& err,
                           bool activate = true)
    {
        cuCtxSetCurrent(context);
        char trs_buf[64];
        std::snprintf(trs_buf, sizeof(trs_buf), "%.17g", fs_error_trs_v);
        std::string key = hash_key(krs_body, amountOfX) + tag
                        + ":t" + std::to_string(type_of_synch_v)
                        + ":e" + std::to_string(error_estim_v)
                        + ":r" + trs_buf;
        return compile_into(pool, key, slot, activate, [&](CachedFastSyncModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template), "fastsync.cu",
                              { { "{{AMOUNT_OF_X}}",   std::to_string(amountOfX) },
                                { "{{TYPE_OF_SYNCH}}", std::to_string(type_of_synch_v) },
                                { "{{ERROR_ESTIM}}",   std::to_string(error_estim_v) },
                                { "{{FS_ERROR_TRS}}",  std::string(trs_buf) },
                                { "{{KRS_BODY}}",      krs_body } },
                              expr_kernels,
                              mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;

            // Связываем по expr_kernels индексу.
            for (size_t i = 0; i < expr_kernels.size(); ++i) {
                CUfunction f = nullptr;
                if (!module_fn(mod, lowered[i], f, err)) { cuModuleUnload(mod); return false; }
                const std::string sym_name = expr_kernels[i];
                if      (sym_name == "fillFSMasterTrajectory")                   fresh.kernel_fs_fill = f;
                else if (sym_name == "calculateDiscreteModelforFastSynchroCUDA") fresh.kernel_fs_traj = f;
                else if (sym_name == "calculateDiscreteModelICCforFastSynchro")  fresh.kernel_fs_grid = f;
            }
            return true;
        }, err);
    }

    bool compile_order_module(bool activate, int amountOfX, int amountOfValues,
                              const std::string& krs_body, const std::string& ref_body,
                              std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело эталона входит в ключ: с ним и без него это РАЗНЫЙ модуль (ветка
        // эталона вырезается препроцессором), да и сам эталонный метод меняется.
        const std::string key = hash_key(krs_body, amountOfX) + ":order:v"
                              + std::to_string(amountOfValues)
                              + ":ref" + std::to_string(std::hash<std::string>{}(ref_body));
        return compile_into(pool_order, key, cached_order, activate, [&](CachedOrderModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            // orderEstimateKernel объявлено extern "C" — в name_exprs не нужно.
            if (!build_module(snapshot_sources(src_template_order), "order.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{REF_ENABLED}}",      ref_body.empty() ? "0" : "1" },
                                { "{{KRS_REF_BODY}}",     ref_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "orderEstimateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            // Ядро замера лежит в том же модуле: Performance нужны ОБА прохода,
            // и второй компиляции ради этого быть не должно.
            if (!module_fn(mod, "perfIntegrateKernel", fresh.kernel_perf, err)) { cuModuleUnload(mod); return false; }
            // И ядро областей устойчивости — оттуда же. Отдельного модуля ему не
            // нужно: тело КРС в этом уже лежит, а регистров чужим расчётам оно
            // не стоит, регистровый бюджет считается на ядро.
            if (!module_fn(mod, "stabilityRegionKernel", fresh.kernel_stab, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    // prewarm_* — компиляция без расчёта. Ключи считаются ровно так же, как в соответствующем
    // run_*, иначе прогретый модуль не был бы найден. Ошибки не всплывают: не прогрелось — Run
    // скомпилирует сам и покажет ошибку уже там.
    void prewarm_bif1d(const Bifurcation1DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        if (req.continuation && !req.use_cpu) compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err, false);
        else compile_if_needed(req.krs_body, req.amountOfX, req.sweep_over_var ? 0 : 1, err, false);
    }
    void prewarm_bif2d(const Bifurcation2DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_bif2d_if_needed(req.krs_body, req.amountOfX,
                                par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                              req.sweep_over_var, req.sweep_over_var_2), err, false);
    }
    void prewarm_lle1d(const LLE1DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_lle_if_needed(req.krs_body, req.amountOfX, req.sweep_over_var ? 0 : 1, err, false);
    }
    void prewarm_lle2d(const LLE2DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_lle_2d_if_needed(req.krs_body, req.amountOfX,
                                 par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                               req.sweep_over_var, req.sweep_over_var_2), err, false);
    }
    void prewarm_ls1d(const LS1DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_ls_if_needed(req.krs_body, req.amountOfX, req.sweep_over_var ? 0 : 1, err, false);
    }
    void prewarm_ls2d(const LS2DRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_ls_2d_if_needed(req.krs_body, req.amountOfX,
                                par_or_var_2d(req.sweep_over_h, req.sweep_over_h_2,
                                              req.sweep_over_var, req.sweep_over_var_2), err, false);
    }
    void prewarm_basins(const BasinsRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        compile_basins_if_needed(req.krs_body, req.amountOfX, err, false);
    }

    // Узлы одной оси. Лог-сетка требует строго положительных границ: по шагу
    // это всегда так, по параметру — нет, поэтому проверка, а не clamp.
    static bool order_axis_nodes(const OrderAxis& ax, const char* what,
                                 std::vector<double>& out, std::string& err)
    {
        const int n = (ax.kind == OrderAxisKind::None) ? 1 : ax.n_pts;
        if (n <= 0)    { err = std::string(what) + ": the number of points must be > 0"; return false; }
        if (n > 100000){ err = std::string(what) + ": the number of points is too large"; return false; }
        out.assign((size_t)n, 0.0);
        if (ax.kind == OrderAxisKind::None) { out[0] = 0.0; return true; }
        if (n == 1) { out[0] = ax.lo; return true; }
        if (ax.log_scale) {
            if (!(ax.lo > 0.0) || !(ax.hi > 0.0)) {
                err = std::string(what) + ": log scale requires both bounds > 0";
                return false;
            }
            const double k = std::log(ax.hi / ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo * std::exp(k * (double)i);
        } else {
            const double d = (ax.hi - ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo + d * (double)i;
        }
        return true;
    }

    // Число грубых шагов ячейки — ТА ЖЕ формула, что в ядре. Хосту она нужна,
    // чтобы посчитать полную работу для прогресс-бара: при свипе по h она
    // отличается от узла к узлу на порядки, и «среднее по сетке» давало бы
    // бар, который стоит на месте, а потом прыгает в конец.
    static long long order_steps_for(double h, double tMax, bool snap) {
        if (!(h > 0.0)) return 1;
        long long N = snap ? (long long)(tMax / h + 0.5) : (long long)(tMax / h);
        return N < 1 ? 1 : N;
    }

    struct OrderDevBuf {
        void* p = nullptr;
        ~OrderDevBuf() { if (p) cudaFree(p); }
        bool alloc(size_t bytes, const char* what, std::string& err) {
            cudaError_t e = cudaMalloc(&p, bytes ? bytes : 1);
            if (e != cudaSuccess) { p = nullptr; err = std::string("cudaMalloc ") + what + ": " + cudaGetErrorString(e); return false; }
            return true;
        }
        template <class T> T* as() const { return (T*)p; }
    };

    OrderResult run_order(const OrderRequest& req) {
        OrderResult res;
        res.axis_x = req.axis_x;
        res.axis_y = req.axis_y;
        auto fail = [&](const std::string& msg) -> OrderResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX) return fail("initial_conditions.size() != amountOfX");
        if (req.values.empty())                                  return fail("values is empty (at least a[0] is required)");
        if ((int)req.values.size() > kMaxAmountOfValues)         return fail("too many values");
        if (!(req.h > 0.0))                                      return fail("h must be > 0");
        if (!(req.t_max > 0.0))                                  return fail("t_max must be > 0");
        if (req.axis_x.kind == OrderAxisKind::None)              return fail("the X axis is not set");

        const int amountOfValues = (int)req.values.size();
        auto check_axis_index = [&](const OrderAxis& ax, const char* what) -> bool {
            if (ax.kind != OrderAxisKind::Value) return true;
            if (ax.index < 0 || ax.index >= amountOfValues) {
                res.error = std::string(what) + ": parameter index outside a[]";
                return false;
            }
            return true;
        };
        if (!check_axis_index(req.axis_x, "X axis")) return res;
        if (!check_axis_index(req.axis_y, "Y axis")) return res;
        if (req.axis_x.kind == OrderAxisKind::H && req.axis_y.kind == OrderAxisKind::H)
            return fail("both axes cannot sweep h");
        if (req.axis_x.kind == OrderAxisKind::Value && req.axis_y.kind == OrderAxisKind::Value
            && req.axis_x.index == req.axis_y.index)
            return fail("both axes sweep the same parameter");

        std::string err;
        if (!order_axis_nodes(req.axis_x, "X axis", res.axis_x_vals, err)) return fail(err);
        if (!order_axis_nodes(req.axis_y, "Y axis", res.axis_y_vals, err)) return fail(err);

        res.n_pts_x = (int)res.axis_x_vals.size();
        res.n_pts_y = (req.axis_y.kind == OrderAxisKind::None) ? 1 : (int)res.axis_y_vals.size();
        if (req.axis_y.kind == OrderAxisKind::None) res.axis_y_vals.assign(1, 0.0);

        const size_t total_cells = (size_t)res.n_pts_x * (size_t)res.n_pts_y;
        if (total_cells == 0) return fail("empty grid");

        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        // Эталон включается только когда есть И тело, И положительное число
        // подшагов: одно без другого — это просьба посчитать пустоту.
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);

        // Работа по ячейкам: h ячейки зависит только от той оси, что свипует h.
        auto cell_h = [&](int ix, int iy) -> double {
            double h = req.h;
            if (req.axis_x.kind == OrderAxisKind::H) h = res.axis_x_vals[(size_t)ix];
            if (req.axis_y.kind == OrderAxisKind::H) h = res.axis_y_vals[(size_t)iy];
            return h;
        };
        long long maxStepsPerCell = 1;
        double    totalSteps      = 0.0;
        for (int iy = 0; iy < res.n_pts_y; ++iy)
            for (int ix = 0; ix < res.n_pts_x; ++ix) {
                const long long N = order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps);
                if (N > maxStepsPerCell) maxStepsPerCell = N;
                totalSteps += (double)N;
            }
        // 1e15 шагов — это уже «никогда не досчитается», и (long long) ниже
        // всё равно переполнится на произведении с числом ячеек.
        if (totalSteps > 1.0e15) return fail("t_max / h too large: the work does not fit into a reasonable time");

        const int    progressStride = progress_stride_for((size_t)maxStepsPerCell);
        const double ticksTotal     = totalSteps / (double)progressStride;

        // Чанкование по ячейкам — не ради памяти (выход 4 числа на ячейку), а
        // ради TDR: один запуск на всю сетку при большом t_max/h легко
        // перевалит за watchdog. Бюджет — примерно столько вычислений правой
        // части, сколько GPU успевает за доли секунды; 7 вызовов КРС на
        // грубый шаг (1 + 2 + 4).
        const double kWorkBudget = 2.0e8;
        size_t cellsPerLaunch = (size_t)(kWorkBudget / (7.0 * (double)maxStepsPerCell));
        if (cellsPerLaunch < 256)          cellsPerLaunch = 256;
        if (cellsPerLaunch > total_cells)  cellsPerLaunch = total_cells;

        OrderDevBuf d_axisX, d_axisY, d_X0, d_values;
        OrderDevBuf d_p, d_e1, d_e2, d_eref, d_h, d_status;
        if (!d_axisX .alloc(res.axis_x_vals.size() * sizeof(numb), "axisXVals", err)) return fail(err);
        if (!d_axisY .alloc(res.axis_y_vals.size() * sizeof(numb), "axisYVals", err)) return fail(err);
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",        err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",    err)) return fail(err);
        if (!d_p     .alloc(total_cells * sizeof(numb), "outP",      err)) return fail(err);
        if (!d_e1    .alloc(total_cells * sizeof(numb), "outE1",     err)) return fail(err);
        if (!d_e2    .alloc(total_cells * sizeof(numb), "outE2",     err)) return fail(err);
        if (!d_eref  .alloc(total_cells * sizeof(numb), "outERef",   err)) return fail(err);
        if (!d_h     .alloc(total_cells * sizeof(numb), "outH",      err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);

        {
            const std::vector<numb> ax = to_numb(res.axis_x_vals);
            const std::vector<numb> ay = to_numb(res.axis_y_vals);
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_axisX.p,  ax.data(), ax.size() * sizeof(numb), "axisXVals")) return fail(err);
            if (!up(d_axisY.p,  ay.data(), ay.size() * sizeof(numb), "axisYVals")) return fail(err);
            if (!up(d_X0.p,     x0.data(), x0.size() * sizeof(numb), "X0"))        return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Множитель порога полки округления. Само сравнение в ядре —
        // floorEps * |решение| * sqrt(4N): шум вычитания копится по шагам, так
        // что абсолютный порог обязан расти вместе с их числом (см. коммент в
        // order.template.cu). Здесь только константа при нём.
        const numb floorEps = (numb)2 * std::numeric_limits<numb>::epsilon();

        double ticksDone = 0.0;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int    nPtsX_arg  = res.n_pts_x;
            int    nPtsY_arg  = res.n_pts_y;
            int    nCells_arg = (int)count;
            int    offset_arg = (int)offset;
            numb*  axX_arg    = d_axisX.as<numb>();
            numb*  axY_arg    = d_axisY.as<numb>();
            int    axXKind    = (int)req.axis_x.kind;
            int    axXIndex   = req.axis_x.index;
            int    axYKind    = (int)req.axis_y.kind;
            int    axYIndex   = req.axis_y.index;
            numb*  X0_arg     = d_X0.as<numb>();
            numb*  values_arg = d_values.as<numb>();
            numb   hBase_arg  = (numb)req.h;
            numb   tMax_arg   = (numb)req.t_max;
            int    snap_arg   = req.snap_steps    ? 1 : 0;
            int    endp_arg   = req.endpoint_only ? 1 : 0;
            numb   maxV_arg   = (numb)req.max_value;
            numb   feps_arg   = floorEps;
            int    refSub_arg = ref_on ? req.ref_substeps : 0;
            numb*  outP_arg   = d_p.as<numb>();
            numb*  outE1_arg  = d_e1.as<numb>();
            numb*  outE2_arg  = d_e2.as<numb>();
            numb*  outERef_arg = d_eref.as<numb>();
            numb*  outH_arg   = d_h.as<numb>();
            int*   outSt_arg  = d_status.as<int>();
            int*   cancel_arg = sig.cancelArg();
            int*   prog_arg   = sig.progressArg();
            int    stride_arg = progressStride;

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &axX_arg, &axY_arg,
                &axXKind, &axXIndex, &axYKind, &axYIndex,
                &X0_arg, &values_arg,
                &hBase_arg, &tMax_arg, &snap_arg, &endp_arg, &maxV_arg, &feps_arg,
                &refSub_arg,
                &outP_arg, &outE1_arg, &outE2_arg, &outERef_arg, &outH_arg, &outSt_arg,
                &cancel_arg, &prog_arg, &stride_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            sig.resetTicks();
            CUresult r = cuLaunchKernel(cached_order.kernel, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(order): " + cu_err(r));
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("order kernel: ") + cudaGetErrorString(ce));

            // Работа чанка в тиках — сумма по его ячейкам, а не count*средняя:
            // при свипе по h соседние ячейки различаются на порядки.
            for (size_t c = offset; c < offset + count; ++c) {
                const int ix = (int)(c % (size_t)res.n_pts_x);
                const int iy = (res.n_pts_y > 1) ? (int)(c / (size_t)res.n_pts_x) : 0;
                ticksDone += (double)order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps)
                           / (double)progressStride;
            }

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hp(total_cells), he1(total_cells), he2(total_cells),
                              herf(total_cells), hh(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_p.p,      hp.data(),         total_cells * sizeof(numb), "p"))      return fail(err);
            if (!dn(d_e1.p,     he1.data(),        total_cells * sizeof(numb), "e1"))     return fail(err);
            if (!dn(d_e2.p,     he2.data(),        total_cells * sizeof(numb), "e2"))     return fail(err);
            if (!dn(d_eref.p,   herf.data(),       total_cells * sizeof(numb), "eRef"))   return fail(err);
            if (!dn(d_h.p,      hh.data(),         total_cells * sizeof(numb), "h"))      return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);

            res.p.resize(total_cells); res.e1.resize(total_cells);
            res.e2.resize(total_cells); res.h_eff.resize(total_cells);
            res.e_ref.resize(total_cells);
            bool first_p = true, first_e = true, first_er = true;
            for (size_t i = 0; i < total_cells; ++i) {
                res.p[i]     = (double)hp[i];
                res.e1[i]    = (double)he1[i];
                res.e2[i]    = (double)he2[i];
                res.e_ref[i] = (double)herf[i];
                res.h_eff[i] = (double)hh[i];
                switch (res.status[i]) {
                    case ORDER_ST_DIVERGED:   ++res.n_diverged;   break;
                    case ORDER_ST_FLOOR:      ++res.n_floor;      break;
                    case ORDER_ST_NOCONTRACT: ++res.n_nocontract; break;
                    default:                  ++res.n_ok;         break;
                }
                // Диапазоны — только по чистым ячейкам (см. OrderResult::p_min).
                if (res.status[i] != ORDER_ST_OK) continue;
                if (std::isfinite(res.p[i])) {
                    if (first_p) { res.p_min = res.p_max = res.p[i]; first_p = false; }
                    else { if (res.p[i] < res.p_min) res.p_min = res.p[i];
                           if (res.p[i] > res.p_max) res.p_max = res.p[i]; }
                }
                if (std::isfinite(res.e1[i]) && res.e1[i] > 0.0) {
                    if (first_e) { res.e1_min = res.e1_max = res.e1[i]; first_e = false; }
                    else { if (res.e1[i] < res.e1_min) res.e1_min = res.e1[i];
                           if (res.e1[i] > res.e1_max) res.e1_max = res.e1[i]; }
                }
                if (std::isfinite(res.e_ref[i]) && res.e_ref[i] > 0.0) {
                    if (first_er) { res.eref_min = res.eref_max = res.e_ref[i]; first_er = false; }
                    else { if (res.e_ref[i] < res.eref_min) res.eref_min = res.e_ref[i];
                           if (res.e_ref[i] > res.eref_max) res.eref_max = res.e_ref[i]; }
                }
            }
        }

        res.ok = true;
        return res;
    }

    // Stability — область устойчивости схемы на двумерной задаче Дальквиста.
    //
    // Считать тут почти нечего: ячейка — это ДВА шага схемы (по одному от
    // каждого базисного вектора), поэтому вся карта 512x512 стоит столько же,
    // сколько один узел p(h) при t_max/h = 1000. Чанкование оставлено только
    // ради отмены и прогресса: экстраполяционная обёртка с десятком стадий на
    // мелкой сетке всё-таки набирает заметное время.
    StabilityResult run_stability(const StabilityRequest& req) {
        StabilityResult res;
        auto fail = [&](const std::string& msg) -> StabilityResult& { res.error = msg; return res; };

        const std::string bad = stability_validate(req);
        if (!bad.empty()) return fail(bad);
        stability_fill_axes(req, res);

        const int    amountOfValues = (int)req.values.size();
        const size_t total_cells    = (size_t)res.n_pts_x * (size_t)res.n_pts_y;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_order

        // Модуль общий с Order: тот же ключ, та же компиляция, эталон выключен.
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  std::string(), err)) return fail(err);
        if (cached_order.kernel_stab == nullptr)
            return fail("stabilityRegionKernel not found in the module");

        OrderDevBuf d_sig, d_om, d_values, d_rho, d_status, d_R;
        if (!d_sig   .alloc(res.sigma_vals.size() * sizeof(numb), "sigmaVals", err)) return fail(err);
        if (!d_om    .alloc(res.omega_vals.size() * sizeof(numb), "omegaVals", err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",   err)) return fail(err);
        if (!d_rho   .alloc(total_cells * sizeof(numb), "outRho",    err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);
        if (!d_R     .alloc(4 * total_cells * sizeof(numb), "outR",  err)) return fail(err);

        {
            const std::vector<numb> hs = to_numb(res.sigma_vals);
            const std::vector<numb> ho = to_numb(res.omega_vals);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_sig.p,    hs.data(), hs.size() * sizeof(numb), "sigmaVals")) return fail(err);
            if (!up(d_om.p,     ho.data(), ho.size() * sizeof(numb), "omegaVals")) return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        const size_t cellsPerLaunch = 1u << 16;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int   nPtsX_arg  = res.n_pts_x;
            int   nPtsY_arg  = res.n_pts_y;
            int   nCells_arg = (int)count;
            int   offset_arg = (int)offset;
            numb* sig_arg    = d_sig.as<numb>();
            numb* om_arg     = d_om.as<numb>();
            numb* values_arg = d_values.as<numb>();
            int   ia_arg = req.idx_a, ib_arg = req.idx_b, ic_arg = req.idx_c, id_arg = req.idx_d;
            numb  k_arg = (numb)req.k, r_arg = (numb)req.r, h_arg = (numb)req.h;
            numb* rho_arg = d_rho.as<numb>();
            int*  st_arg  = d_status.as<int>();
            numb* R_arg   = d_R.as<numb>();

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &sig_arg, &om_arg, &values_arg,
                &ia_arg, &ib_arg, &ic_arg, &id_arg,
                &k_arg, &r_arg, &h_arg,
                &rho_arg, &st_arg, &R_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            CUresult r = cuLaunchKernel(cached_order.kernel_stab, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(stability): " + cu_err(r));
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("stability kernel: ") + cudaGetErrorString(ce));

            if (req.progress)
                req.progress->store((float)((double)(offset + count) / (double)total_cells),
                                    std::memory_order_relaxed);
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hrho(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_rho.p,    hrho.data(),       total_cells * sizeof(numb), "rho"))    return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);
            std::vector<numb> hR(4 * total_cells);
            if (!dn(d_R.p, hR.data(), hR.size() * sizeof(numb), "R")) return fail(err);

            res.rho.resize(total_cells);
            for (size_t i = 0; i < total_cells; ++i) res.rho[i] = (double)hrho[i];

            // Ошибка шага — на хосте, той же функцией, что в CPU-ветке.
            res.err.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < total_cells; ++i) {
                if (res.status[i] != STAB_ST_OK) continue;
                const size_t ix = i % (size_t)res.n_pts_x, iy = i / (size_t)res.n_pts_x;
                res.err[i] = stability_step_error(res.sigma_vals[ix], res.omega_vals[iy],
                                                  req.k, req.r, req.h,
                                                  (double)hR[i],                   (double)hR[total_cells + i],
                                                  (double)hR[2 * total_cells + i], (double)hR[3 * total_cells + i]);
            }
            stability_summarize(res);
        }

        res.ok = true;
        return res;
    }

    // Performance — «время счёта vs достигнутая ошибка» по одномерной сетке.
    //
    // Два прохода. Первый — обычный run_order по той же оси: он даёт E1/E2,
    // статусы и фактический шаг узла и НЕ засекается. Второй — собственно
    // замер: на каждом узле warmup холостых запусков, затем repeats запусков
    // под cudaEvent'ами, стоящими вплотную вокруг cuLaunchKernel. Всё, что
    // запуском ядра не является — компиляция модуля, H2D значений, выделение
    // выходного буфера, D2H, — из измеряемого интервала вынесено; кэш модуля
    // к этому моменту уже прогрет order-проходом.
    PerfResult run_performance(const PerfRequest& req) {
        PerfResult res;
        res.axis = req.axis;
        auto fail = [&](const std::string& msg) -> PerfResult& { res.error = msg; return res; };

        if (req.axis.kind == OrderAxisKind::None) return fail("the axis is not set");
        if (req.repeats  < 1) return fail("the number of measurements must be >= 1");
        if (req.replicas < 1) return fail("the number of replicas must be >= 1");
        if (req.warmup   < 0) return fail("the number of warmup launches cannot be negative");

        // ---- Проход 1: ошибки (не засекается) ----
        OrderRequest oreq;
        oreq.krs_body           = req.krs_body;
        oreq.amountOfX          = req.amountOfX;
        oreq.initial_conditions = req.initial_conditions;
        oreq.values             = req.values;
        oreq.axis_x             = req.axis;
        oreq.axis_y             = OrderAxis{};          // kind == None -> одномерная сетка
        oreq.h                  = req.h;
        oreq.t_max              = req.t_max;
        oreq.snap_steps         = req.snap_steps;
        oreq.endpoint_only      = req.endpoint_only;
        oreq.max_value          = req.max_value;
        oreq.ref_krs_body       = req.ref_krs_body;
        oreq.ref_substeps       = req.ref_substeps;
        oreq.cancel             = req.cancel;

        // Прогресс order-прохода отдаётся в [0, 0.5]: без пересчёта бар
        // пробежал бы от нуля до конца дважды. run_order синхронный, поэтому
        // масштабированием занимается отдельный поток, живущий ровно на время
        // вызова.
        std::shared_ptr<std::atomic<float>> sub_prog;
        std::atomic<bool> watch_stop{ false };
        std::thread watcher;
        if (req.progress) {
            sub_prog = std::make_shared<std::atomic<float>>(0.0f);
            oreq.progress = sub_prog;
            watcher = std::thread([&req, &sub_prog, &watch_stop]() {
                while (!watch_stop.load(std::memory_order_relaxed)) {
                    req.progress->store(0.5f * sub_prog->load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
        OrderResult ores = run_order(oreq);
        if (watcher.joinable()) { watch_stop.store(true, std::memory_order_relaxed); watcher.join(); }

        if (ores.cancelled) { res.cancelled = true; return res; }
        if (!ores.ok) return fail(ores.error.empty() ? std::string("the order pass failed") : ores.error);

        res.n_pts     = ores.n_pts_x;
        res.axis_vals = ores.axis_x_vals;
        res.e1        = ores.e1;
        res.e2        = ores.e2;
        res.e_ref     = ores.e_ref;
        res.p         = ores.p;
        res.h_eff     = ores.h_eff;
        res.status    = ores.status;
        res.n_ok         = ores.n_ok;
        res.n_diverged   = ores.n_diverged;
        res.n_floor      = ores.n_floor;
        res.n_nocontract = ores.n_nocontract;
        res.repeats   = req.repeats;
        res.warmup    = req.warmup;
        res.replicas  = req.replicas;

        const int n = res.n_pts;
        if (n <= 0) return fail("empty grid");
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        res.t_min.assign((size_t)n, qnan);
        res.t_max.assign((size_t)n, qnan);
        res.t_avg.assign((size_t)n, qnan);
        res.n_steps.assign((size_t)n, 0);

        // ---- Проход 2: замер ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки, см. run_order

        const int amountOfValues = (int)req.values.size();
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);
        if (cached_order.kernel_perf == nullptr) return fail("perfIntegrateKernel not found in the module");

        OrderDevBuf d_X0, d_values, d_out;
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",      err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",  err)) return fail(err);
        if (!d_out   .alloc((size_t)req.replicas * (size_t)req.amountOfX * sizeof(numb), "perfOut", err)) return fail(err);

        std::vector<numb> h_vals = to_numb(req.values);
        {
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            cudaError_t e = cudaMemcpy(d_X0.p, x0.data(), x0.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy X0: ") + cudaGetErrorString(e));
            e = cudaMemcpy(d_values.p, h_vals.data(), h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
        }

        struct EventPair {
            cudaEvent_t a = nullptr, b = nullptr;
            ~EventPair() { if (a) cudaEventDestroy(a); if (b) cudaEventDestroy(b); }
        } ev;
        if (cudaEventCreate(&ev.a) != cudaSuccess || cudaEventCreate(&ev.b) != cudaSuccess)
            return fail("cudaEventCreate: could not create the timing events");

        // Ширина блока — 32: измерено, что дальше упирается в регистры, а не в
        // лимит блоков на SM. При replicas == 1 это один активный warp.
        const int blockSize = 32;
        const int gridSize  = (req.replicas + blockSize - 1) / blockSize;

        for (int i = 0; i < n; ++i) {
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            const double h_node = res.h_eff[(size_t)i];
            if (!(h_node > 0.0) || !std::isfinite(h_node)) continue;   // узел без времени: см. status

            // Свип по параметру меняет a[] от узла к узлу. Загрузка делается
            // здесь, ДО прогрева, и в замер не попадает.
            if (req.axis.kind == OrderAxisKind::Value) {
                const int vi = req.axis.index;
                if (vi >= 0 && vi < amountOfValues) {
                    h_vals[(size_t)vi] = (numb)res.axis_vals[(size_t)i];
                    cudaError_t e = cudaMemcpy(d_values.p, h_vals.data(),
                                               h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
                    if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
                }
            }

            const long long N = order_steps_for(h_node, req.t_max, req.snap_steps);
            res.n_steps[(size_t)i] = N;

            int       nt_arg  = req.replicas;
            numb*     X0_arg  = d_X0.as<numb>();
            numb*     val_arg = d_values.as<numb>();
            numb      h_arg   = (numb)h_node;
            long long n_arg   = N;
            numb*     out_arg = d_out.as<numb>();
            void* args[] = { &nt_arg, &X0_arg, &val_arg, &h_arg, &n_arg, &out_arg };

            auto launch = [&](const char* what) -> bool {
                CUresult r = cuLaunchKernel(cached_order.kernel_perf, gridSize, 1, 1, blockSize, 1, 1,
                                            0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = std::string("cuLaunchKernel(") + what + "): " + cu_err(r); return false; }
                return true;
            };

            for (int w = 0; w < req.warmup; ++w) {
                if (!launch("perf warmup")) return fail(err);
                cudaError_t ce = cudaDeviceSynchronize();
                if (ce != cudaSuccess) return fail(std::string("perf kernel: ") + cudaGetErrorString(ce));
            }

            double tmin = 0.0, tmax = 0.0, tsum = 0.0;
            int got = 0;
            for (int rep = 0; rep < req.repeats; ++rep) {
                cudaEventRecord(ev.a, 0);
                if (!launch("perf")) return fail(err);
                cudaEventRecord(ev.b, 0);
                cudaError_t ce = cudaEventSynchronize(ev.b);
                if (ce != cudaSuccess) return fail(std::string("perf kernel: ") + cudaGetErrorString(ce));
                float ms = 0.0f;
                ce = cudaEventElapsedTime(&ms, ev.a, ev.b);
                if (ce != cudaSuccess) return fail(std::string("cudaEventElapsedTime: ") + cudaGetErrorString(ce));
                const double us = (double)ms * 1000.0;
                if (got == 0) { tmin = tmax = us; }
                else { if (us < tmin) tmin = us; if (us > tmax) tmax = us; }
                tsum += us;
                ++got;

                if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                    res.cancelled = true;
                    return res;
                }
            }
            if (got > 0) {
                res.t_min[(size_t)i] = tmin;
                res.t_max[(size_t)i] = tmax;
                res.t_avg[(size_t)i] = tsum / (double)got;
            }

            if (req.progress)
                req.progress->store(0.5f + 0.5f * (float)(i + 1) / (float)n, std::memory_order_relaxed);
        }

        // Диапазоны для автоскейла: только узлы, у которых есть И время, И
        // конечная положительная ошибка — точка графика существует лишь тогда,
        // когда есть обе координаты.
        bool first_t = true, first_e1 = true, first_e2 = true, first_er = true;
        for (int i = 0; i < n; ++i) {
            const double ta = res.t_avg[(size_t)i];
            if (!std::isfinite(ta)) continue;
            const double lo = res.t_min[(size_t)i], hi = res.t_max[(size_t)i];
            if (first_t) { res.t_lo = lo; res.t_hi = hi; first_t = false; }
            else { if (lo < res.t_lo) res.t_lo = lo; if (hi > res.t_hi) res.t_hi = hi; }
            const double e1 = res.e1[(size_t)i], e2 = res.e2[(size_t)i];
            if (std::isfinite(e1) && e1 > 0.0) {
                if (first_e1) { res.e1_min = res.e1_max = e1; first_e1 = false; }
                else { if (e1 < res.e1_min) res.e1_min = e1; if (e1 > res.e1_max) res.e1_max = e1; }
            }
            if (std::isfinite(e2) && e2 > 0.0) {
                if (first_e2) { res.e2_min = res.e2_max = e2; first_e2 = false; }
                else { if (e2 < res.e2_min) res.e2_min = e2; if (e2 > res.e2_max) res.e2_max = e2; }
            }
            const double er = ((size_t)i < res.e_ref.size())
                                  ? res.e_ref[(size_t)i]
                                  : std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(er) && er > 0.0) {
                if (first_er) { res.eref_min = res.eref_max = er; first_er = false; }
                else { if (er < res.eref_min) res.eref_min = er; if (er > res.eref_max) res.eref_max = er; }
            }
        }

        res.ok = true;
        return res;
    }

    // run_fastsync: dispatch по req.mode
    FastSyncResult run_fastsync(const FastSyncRequest& req) {
        FastSyncResult res;
        res.mode = req.mode;
        auto fail = [&](const std::string& msg) -> FastSyncResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                  return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of range");
        if ((int)req.ic_master.size()  != req.amountOfX)           return fail("ic_master.size() != amountOfX");
        if ((int)req.ic_slave.size()   != req.amountOfX)           return fail("ic_slave.size() != amountOfX");
        if ((int)req.k_forward.size()  != req.amountOfX)           return fail("k_forward.size() != amountOfX");
        if ((int)req.k_backward.size() != req.amountOfX)           return fail("k_backward.size() != amountOfX");
        if ((int)req.values.size() > kMaxAmountOfValues)           return fail("too many values");
        if (req.h <= 0.0)             return fail("h must be > 0");
        if (req.iter_of_synchr <= 0)  return fail("iter_of_synchr must be > 0");
        if (req.pre_scaller <= 0)     return fail("pre_scaller must be > 0");

        // Snapshot CSV-relevant request fields for GUI right-click export.
        // FastSync has no engine-side CSV writer, so this is consumed only by
        // data_export::export_fastsync; engine writes nothing to disk.
        res.snapshot.mode           = req.mode;
        res.snapshot.values         = req.values;
        res.snapshot.ic_master      = req.ic_master;
        res.snapshot.ic_slave       = req.ic_slave;
        res.snapshot.k_forward      = req.k_forward;
        res.snapshot.k_backward     = req.k_backward;
        res.snapshot.h              = req.h;
        res.snapshot.gpu_fmad       = get_nvrtc_fmad();
        res.snapshot.iter_of_synchr = req.iter_of_synchr;
        res.snapshot.preScaller     = req.pre_scaller;
        res.snapshot.window         = (double)req.window;
        res.snapshot.type_of_synch  = req.type_of_synch;
        res.snapshot.error_estim    = req.error_estim;
        res.snapshot.fs_error_trs   = req.fs_error_trs;
        res.snapshot.tMax           = req.t_max;
        res.snapshot.transientTime  = req.transient_time;
        res.snapshot.transientTimeSlave = req.transient_time_slave;
        res.snapshot.axis_x_var     = req.axis_x_var;
        res.snapshot.axis_y_var     = req.axis_y_var;
        res.snapshot.axis_x_lo      = req.axis_x_lo;
        res.snapshot.axis_x_hi      = req.axis_x_hi;
        res.snapshot.axis_y_lo      = req.axis_y_lo;
        res.snapshot.axis_y_hi      = req.axis_y_hi;
        res.snapshot.n_pts          = req.n_pts;
        res.snapshot.grid_swap_master_slave = req.grid_swap_master_slave;
        res.snapshot.ic_random_offset = req.ic_random_offset;
        res.snapshot.ic_eps           = req.ic_eps;
        res.snapshot.ic_seed          = req.ic_seed;
        res.snapshot.gs_warmup        = req.gs_warmup;
        res.snapshot.var_names      = req.var_names;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // Сбрасываем sticky-ошибку от прошлого неудачного запуска (например OOB в shared memory):
        // без этого cudaMalloc на старте мгновенно падает с "illegal memory access", что является
        // эхом прошлой ошибки, а не текущей. Реально повреждённый контекст восстановится только
        // рестартом приложения — здесь лишь сброс флага для случаев, когда GPU ещё работоспособен.
        cudaGetLastError();

        if (req.mode == 0) {
            // On Attractor
            if (req.t_max <= 0.0)         return fail("t_max must be > 0");
            if (req.transient_time < 0)   return fail("transient_time must be >= 0");
            if (req.window <= (numb)0.0)  return fail("window must be > 0");

            // Two FS-specific kernels — НЕ calculateDiscreteModelCUDA:
            //   fillFSMasterTrajectory (template, single-thread) — фильтрует
            //   timeDomain полным X[] на каждом шаге через FS device function.
            //   calculateDiscreteModelforFastSynchroCUDA — параллельный синхро-проход.
            std::vector<const char*> exprs = {
                "fillFSMasterTrajectory",
                "calculateDiscreteModelforFastSynchroCUDA"
            };
            if (!compile_fs_module(src_template_fs_attr, ":fs_attr", req.amountOfX, req.krs_body,
                                   req.type_of_synch, req.error_estim, req.fs_error_trs,
                                   exprs, pool_fs_attr, cached_fs_attr, err)) return fail(err);

            const int amountOfNTPoints      = (int)(req.window / req.h);
            const int amountOfCTPoints      = (int)(req.t_max / req.h);
            // size_t + потолок — как в On Grid: (int)(TT/h) при мелком h это UB.
            size_t amountOfPointsForSkip = 0;
            {
                const double skip_d = req.transient_time / req.h;   // h > 0 проверен выше
                if (!std::isfinite(skip_d) || skip_d > 1.0e15)
                    return fail("transient_time / h too large");
                amountOfPointsForSkip = (size_t)skip_d;
            }
            const int nPts                  = amountOfCTPoints / (req.pre_scaller > 0 ? req.pre_scaller : 1);
            if (nPts <= 0) return fail("computed nPts <= 0 (t_max/h/preScaller)");
            // FS-kernel читает timeDomain[idx*preScaller*amountOfX + i*amountOfX + j]
            // где idx ∈ [0..nPts), i ∈ [0..amountOfNTPoints). Поэтому буфер должен
            // вмещать amountOfCTPoints + amountOfNTPoints точек (с padding'ом).
            const int traj_len_pts          = amountOfCTPoints + amountOfNTPoints;

            int amountOfIC_int     = req.amountOfX;
            int amountOfValues_int = (int)req.values.size();

            numb* d_timeDomain = nullptr; numb* d_output = nullptr;
            numb* d_Xs = nullptr;   numb* d_X0 = nullptr;
            numb* d_values = nullptr;
            numb* d_kF = nullptr;   numb* d_kB = nullptr;
            RunSignals sig;   // прогресс и отмена в mapped-памяти
            #define FS_CHECK(x, m) do { cudaError_t _e = (x); if (_e != cudaSuccess) { err = std::string(m) + ": " + cudaGetErrorString(_e); goto FS0_FAIL; } } while(0)

            size_t traj_bytes = (size_t)traj_len_pts * (size_t)amountOfIC_int * sizeof(numb);
            size_t out_bytes  = (size_t)nPts * sizeof(numb);
            FS_CHECK(cudaMalloc((void**)&d_timeDomain, traj_bytes), "cudaMalloc d_timeDomain");
            FS_CHECK(cudaMalloc((void**)&d_output,     out_bytes),  "cudaMalloc d_output");
            FS_CHECK(cudaMalloc((void**)&d_Xs,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_Xs");
            FS_CHECK(cudaMalloc((void**)&d_X0,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_X0");
            FS_CHECK(cudaMalloc((void**)&d_values,     amountOfValues_int * sizeof(numb)), "cudaMalloc d_values");
            FS_CHECK(cudaMalloc((void**)&d_kF,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_kF");
            FS_CHECK(cudaMalloc((void**)&d_kB,         amountOfIC_int * sizeof(numb)), "cudaMalloc d_kB");
            if (!sig.alloc(err)) goto FS0_FAIL;
            // Бар делится между двумя фазами по их доле работы. Заполнение
            // окна идёт в один поток и стоит skip + pts шагов; проходы вперёд-назад
            // идут параллельно по точкам, так что по времени это 2 * iterOfSynchr * pts.
            // Первая версия отдавала весь бар первой фазе — он доходил до 90%
            // и там вставал.
            const double fsW1 = (double)amountOfPointsForSkip + (double)traj_len_pts;
            const double fsW2 = 2.0 * (double)req.iter_of_synchr * (double)traj_len_pts;
            const double fsW  = (fsW1 + fsW2) > 0 ? (fsW1 + fsW2) : 1.0;
            const double fsT1 = (double)traj_len_pts;                                 // тиков в фазе 1
            const double fsT2 = (double)nPts * (double)req.iter_of_synchr;       // тиков в фазе 2

            // См. пояснение в grid-ветке: Request хранит их как vector<double>,
            // а буферы — numb, поэтому конверсия обязательна для всех, не
            // только для values.
            FS_CHECK(cudaMemcpy(d_X0,     to_numb(req.ic_master).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy X0");
            FS_CHECK(cudaMemcpy(d_Xs,     to_numb(req.ic_slave).data(),   amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy Xs");
            FS_CHECK(cudaMemcpy(d_values, to_numb(req.values).data(),     amountOfValues_int * sizeof(numb), cudaMemcpyHostToDevice), "memcpy values");
            FS_CHECK(cudaMemcpy(d_kF,     to_numb(req.k_forward).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kF");
            FS_CHECK(cudaMemcpy(d_kB,     to_numb(req.k_backward).data(), amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kB");

            // Шаг 1: fillFSMasterTrajectory — single-thread, заливает d_timeDomain
            // полным X[] на каждом шаге (через FS device function с K=0).
            if (req.progress) req.progress->store(0.05f);
            {
                numb  h_arg     = req.h;
                size_t  skip_arg  = amountOfPointsForSkip;
                int     pts_arg   = traj_len_pts;
                int* d_cancel_arg   = sig.cancelArg();
                int* d_progress_arg = sig.progressArg();
                sig.resetTicks();
                void* args_fill[] = {
                    &d_values, &h_arg, &d_X0, &skip_arg, &pts_arg, &d_timeDomain,
                    &d_cancel_arg, &d_progress_arg
                };
                CUresult r = cuLaunchKernel(cached_fs_attr.kernel_fs_fill,
                                            1, 1, 1, 1, 1, 1,
                                            0, nullptr, args_fill, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_attr fill): " + cu_err(r); goto FS0_FAIL; }
                // Заполнение окна мастера идёт в один поток и занимает
                // основную часть прогона, поэтому отдаём ему весь бар.
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       0.0, fsT1 * fsW / (fsW1 > 0 ? fsW1 : 1.0), err)) goto FS0_FAIL;
                cudaDeviceSynchronize();
            }

            // Шаг 2: calculateDiscreteModelforFastSynchroCUDA — nPts threads, читает
            // timeDomain, пишет d_output[idx] = ошибка синхронизации на точке idx.
            {
                int    nPts_int           = nPts;
                int    nPtsLimiter_int    = nPts;
                int    amountOfNTPoints_i = amountOfNTPoints;
                numb h_arg              = req.h;
                int    iterOfSynchr_i     = req.iter_of_synchr;
                int    preScaller_i       = req.pre_scaller;
                numb maxValue_arg       = req.max_value;
                int    amountOfValues_i   = amountOfValues_int;
                // d_Xs остаётся заполненным и в random-режиме: ядро просто не
                // читает его, когда ic_random_offset взведён.
                int    icRandom_i         = req.ic_random_offset ? 1 : 0;
                numb   icEps_arg          = (numb)req.ic_eps;
                unsigned long long icSeed_arg = req.ic_seed;
                int    gsWarmup_i         = req.gs_warmup;

                int* d_cancel_arg   = sig.cancelArg();
                int* d_progress_arg = sig.progressArg();
                void* args_fs[] = {
                    &nPts_int, &nPtsLimiter_int, &amountOfNTPoints_i, &h_arg,
                    &d_Xs, &amountOfIC_int,
                    &d_values, &d_kF, &d_kB,
                    &iterOfSynchr_i, &amountOfValues_i, &amountOfNTPoints_i,
                    &maxValue_arg,
                    &d_timeDomain, &d_output, &preScaller_i,
                    &icRandom_i, &icEps_arg, &icSeed_arg, &gsWarmup_i,
                    &d_cancel_arg, &d_progress_arg
                };
                int blockSize = launch_block_size(0);   // fs_attr без динамической shared
                int gridSize  = (nPts + blockSize - 1) / blockSize;
                CUresult r = cuLaunchKernel(cached_fs_attr.kernel_fs_traj,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            0, nullptr, args_fs, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_attr): " + cu_err(r); goto FS0_FAIL; }
                sig.resetTicks();
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       fsT2 * fsW1 / (fsW2 > 0 ? fsW2 : 1.0),
                                       fsT2 * fsW  / (fsW2 > 0 ? fsW2 : 1.0), err)) goto FS0_FAIL;
                cudaDeviceSynchronize();
            }
            if (req.progress) req.progress->store(1.0f);

            // Шаг 3: D2H — trajectory + errors. timeDomain хранит RAW точки
            // (без decimator'а); для визуализации выбираем точки с шагом preScaller.
            {
                std::vector<numb> h_traj((size_t)traj_len_pts * (size_t)amountOfIC_int);
                FS_CHECK(cudaMemcpy(h_traj.data(), d_timeDomain, traj_bytes, cudaMemcpyDeviceToHost), "memcpy traj D2H");

                std::vector<numb> h_out(nPts);
                FS_CHECK(cudaMemcpy(h_out.data(), d_output, out_bytes, cudaMemcpyDeviceToHost), "memcpy out D2H");

                // Сохраняем полную (decimated) траекторию — все каналы. GUI
                // выберет axis_x_var / axis_y_var при отрисовке (re-render
                // при смене combo'ов без повторного Run).
                res.traj_full.resize((size_t)nPts * (size_t)amountOfIC_int);
                for (int i = 0; i < nPts; ++i) {
                    size_t row = (size_t)i * (size_t)req.pre_scaller;
                    if (row >= (size_t)traj_len_pts) row = (size_t)traj_len_pts - 1;
                    for (int j = 0; j < amountOfIC_int; ++j)
                        res.traj_full[(size_t)i * amountOfIC_int + j]
                            = h_traj[row * (size_t)amountOfIC_int + j];
                }
                res.sync_error.assign(h_out.begin(), h_out.end());   // numb -> double
                res.n_pts_traj    = nPts;
                res.amountOfX_traj = amountOfIC_int;
            }

            // min/max sync_error для autoscale.
            {
                double vmin = std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (double v : res.sync_error) {
                    if (!std::isfinite(v)) continue;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                res.min_val = std::isfinite(vmin) ? vmin : 0.0;
                res.max_val = std::isfinite(vmax) ? vmax : 0.0;
            }

            cudaFree(d_timeDomain); cudaFree(d_output);
            cudaFree(d_Xs); cudaFree(d_X0);
            cudaFree(d_values); cudaFree(d_kF); cudaFree(d_kB);
            #undef FS_CHECK

            // CSV (On Attractor): delegate to data_export so engine and the
            // GUI right-click export produce byte-identical files.
            if (!req.csv_output_path.empty()) {
                std::ofstream cfg(req.csv_output_path + "_config.csv");
                if (cfg.is_open()) {
                    cfg << std::setprecision(set_precision);
                    data_export::write_fastsync_config(cfg, res.snapshot);
                }
                std::ofstream csv(req.csv_output_path);
                if (csv.is_open()) {
                    csv << std::setprecision(set_precision);
                    data_export::write_fastsync_attractor(csv, res, req.var_names);
                }
            }

            res.ok = true;
            return res;
            FS0_FAIL:
            sig.release();
                if (d_timeDomain) cudaFree(d_timeDomain);
                if (d_output)     cudaFree(d_output);
                if (d_Xs)         cudaFree(d_Xs);
                if (d_X0)         cudaFree(d_X0);
                if (d_values)     cudaFree(d_values);
                if (d_kF)         cudaFree(d_kF);
                if (d_kB)         cudaFree(d_kB);
            return fail(err);
        }
        else {
            // On Grid
            if (req.n_pts <= 0)         return fail("n_pts must be > 0");
            if (req.axis_x_var < 0 || req.axis_x_var >= req.amountOfX) return fail("axis_x_var out of range");
            if (req.axis_y_var < 0 || req.axis_y_var >= req.amountOfX) return fail("axis_y_var out of range");
            if (req.axis_x_var == req.axis_y_var) return fail("axis_x_var == axis_y_var");
            if (req.transient_time < 0)        return fail("transient_time must be >= 0");
            if (req.transient_time_slave < 0)  return fail("transient_time_slave must be >= 0");

            std::vector<const char*> exprs = { "calculateDiscreteModelICCforFastSynchro" };
            if (!compile_fs_module(src_template_fs_grid, ":fs_grid", req.amountOfX, req.krs_body,
                                   req.type_of_synch, req.error_estim, req.fs_error_trs,
                                   exprs, pool_fs_grid, cached_fs_grid, err)) return fail(err);

            // Non-const: их адреса попадают в void*[] для cuLaunchKernel.
            int amountOfPointsInBlock = (int)(req.window / req.h / req.pre_scaller);
            if (amountOfPointsInBlock <= 0) return fail("computed amountOfPointsInBlock <= 0");

            // Transient (TT) — число шагов интегрирования на досадку, отдельно для master и для
            // slave. Уходят в grid-ядро, которое досаживает каждую систему per-cell после
            // grid-override (свипуемая сторона — из затравки своей ячейки, фиксированная — из одной
            // точки). size_t, а не int: TT/h легко перерастает 2^31 (TT=1e5 при h=1e-5 даёт 1e10) —
            // на int это UB, на практике мусор или отрицательное, и транзиент молча пропадал.
            // h > 0 уже проверен выше.
            auto skip_steps = [&](double tt, const char* what, size_t& out) -> bool {
                out = 0;
                if (tt <= 0.0) return true;
                const double skip_d = tt / req.h;
                if (!std::isfinite(skip_d) || skip_d > 1.0e15) {
                    err = std::string(what) + " / h too large";
                    return false;
                }
                out = (size_t)skip_d;
                return true;
            };
            size_t amountOfPointsForSkipMaster = 0;
            size_t amountOfPointsForSkipSlave  = 0;
            if (!skip_steps(req.transient_time,       "transient_time (master)", amountOfPointsForSkipMaster)) return fail(err);
            if (!skip_steps(req.transient_time_slave, "transient_time (slave)",  amountOfPointsForSkipSlave))  return fail(err);
            const size_t total_cells = (size_t)req.n_pts * (size_t)req.n_pts;
            int amountOfIC_int     = req.amountOfX;
            int amountOfValues_int = (int)req.values.size();

            // Memory budget — per-cell trajectory buffer = sizeOfBlock * amountOfX * sizeof(numb).
            size_t freeMemory = 0;
            if (!gpu_free_budget(0.5, freeMemory)) return fail("cudaMemGetInfo failed");
            size_t perCellBytes = (size_t)amountOfPointsInBlock * (size_t)amountOfIC_int * sizeof(numb);
            if (perCellBytes == 0) perCellBytes = sizeof(numb);
            size_t nPtsLimiter = freeMemory / perCellBytes;
            if (nPtsLimiter == 0) nPtsLimiter = 32;
            if (nPtsLimiter > total_cells) nPtsLimiter = total_cells;
            RunSignals sig;   // прогресс и отмена в mapped-памяти
            const size_t originalNPtsLimiter = nPtsLimiter;

            numb* d_data    = nullptr; numb* d_ranges = nullptr;
            int*    d_idx_mv  = nullptr; numb* d_ic_m   = nullptr; numb* d_ic_s = nullptr;
            numb* d_values  = nullptr; numb* d_kF     = nullptr; numb* d_kB   = nullptr;
            int*    d_helpful = nullptr; numb* d_fs_err = nullptr;

            #define FS_GCHECK(x, m) do { cudaError_t _e = (x); if (_e != cudaSuccess) { err = std::string(m) + ": " + cudaGetErrorString(_e); goto FS1_FAIL; } } while(0)

            FS_GCHECK(cudaMalloc((void**)&d_data,    (size_t)nPtsLimiter * perCellBytes), "cudaMalloc d_data");
            FS_GCHECK(cudaMalloc((void**)&d_ranges,  4 * sizeof(numb)),                  "cudaMalloc d_ranges");
            FS_GCHECK(cudaMalloc((void**)&d_idx_mv,  2 * sizeof(int)),                     "cudaMalloc d_idx_mv");
            FS_GCHECK(cudaMalloc((void**)&d_ic_m,    amountOfIC_int * sizeof(numb)),     "cudaMalloc d_ic_m");
            FS_GCHECK(cudaMalloc((void**)&d_ic_s,    amountOfIC_int * sizeof(numb)),     "cudaMalloc d_ic_s");
            FS_GCHECK(cudaMalloc((void**)&d_values,  amountOfValues_int * sizeof(numb)), "cudaMalloc d_values");
            FS_GCHECK(cudaMalloc((void**)&d_kF,      amountOfIC_int * sizeof(numb)),     "cudaMalloc d_kF");
            FS_GCHECK(cudaMalloc((void**)&d_kB,      amountOfIC_int * sizeof(numb)),     "cudaMalloc d_kB");
            FS_GCHECK(cudaMalloc((void**)&d_helpful, nPtsLimiter * sizeof(int)),           "cudaMalloc d_helpful");
            FS_GCHECK(cudaMalloc((void**)&d_fs_err,  total_cells * sizeof(numb)),         "cudaMalloc d_fs_err");
            // Тики ставит цикл заполнения окна мастера — это amountOfPointsInBlock
            // шагов на ячейку; проходы вперёд-назад после него не тикают, поэтому
            // бар доходит до ста на хвосте чанка и ждёт там.
            if (!sig.alloc(err)) goto FS1_FAIL;
            const int    progressStride = progress_stride_for((size_t)amountOfPointsInBlock);
            const double ticksPerCell   = (double)((size_t)amountOfPointsInBlock / (size_t)progressStride);
            const double ticksTotal     = (double)total_cells * ticksPerCell;

            {
                numb   ranges_arr[4] = { (numb)req.axis_x_lo, (numb)req.axis_x_hi, (numb)req.axis_y_lo, (numb)req.axis_y_hi };
                int    idx_mv_arr[2] = { req.axis_x_var, req.axis_y_var };
                FS_GCHECK(cudaMemcpy(d_ranges, ranges_arr, 4 * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy ranges");
                FS_GCHECK(cudaMemcpy(d_idx_mv, idx_mv_arr, 2 * sizeof(int),     cudaMemcpyHostToDevice), "memcpy idx_mv");
                // Все четыре — vector<double> в Request, а буферы типа numb:
                // без to_numb при numb=float копировались бы половинки double.
                // values рядом уже конвертировался, остальные — нет.
                FS_GCHECK(cudaMemcpy(d_ic_m,   to_numb(req.ic_master).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy ic_m");
                FS_GCHECK(cudaMemcpy(d_ic_s,   to_numb(req.ic_slave).data(),   amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy ic_s");
                FS_GCHECK(cudaMemcpy(d_values, to_numb(req.values).data(),     amountOfValues_int * sizeof(numb), cudaMemcpyHostToDevice), "memcpy values");
                FS_GCHECK(cudaMemcpy(d_kF,     to_numb(req.k_forward).data(),  amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kF");
                FS_GCHECK(cudaMemcpy(d_kB,     to_numb(req.k_backward).data(), amountOfIC_int * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy kB");
            }

            // Transient выполняется внутри grid-ядра, per-cell и своим временем для каждой системы
            // (см. amountOfPointsForSkip* выше и комментарий в
            // calculateDiscreteModelICCforFastSynchro). Раньше здесь стоял однопоточный пре-пасс
            // fillFSTransientIC с ОДНИМ TT на обоих и только для нефиксируемой IC: свипуемая сторона
            // стартовала сырой, а прогон в один поток на длинных TT ловил TDR.

            size_t amountOfIteration = (total_cells + nPtsLimiter - 1) / nPtsLimiter;
            for (size_t i = 0; i < amountOfIteration; ++i) {
                size_t cur_limiter = nPtsLimiter;
                if (i == amountOfIteration - 1) cur_limiter = total_cells - originalNPtsLimiter * i;

                int    nPts_arg                  = req.n_pts;
                int    nPtsLimiter_int           = (int)cur_limiter;
                // sizeOfBlock — РАЗМЕР per-thread слота в `data` (в numb-элементах).
                // loopCalculateDiscreteModelForFastSynchro_2 индексирует
                // data[idx*sizeOfBlock + i*amountOfX + j] → нужно вместить
                // amountOfIterations кадров по amountOfX компонент каждый.
                int    sizeOfBlock_int           = amountOfPointsInBlock * amountOfIC_int;
                int    amountOfCalculatedPoints  = (int)(i * originalNPtsLimiter);
                int    dimension_arg             = 2;
                numb h_arg                     = req.h;
                int    amountOfIterations_int    = amountOfPointsInBlock;
                int    preScaller_int            = req.pre_scaller;
                numb maxValue_arg              = req.max_value;
                int    iterOfSynchr_int          = req.iter_of_synchr;
                numb* d_fs_err_chunk           = d_fs_err + i * originalNPtsLimiter;
                int    swap_role_int             = req.grid_swap_master_slave ? 1 : 0;
                size_t skip_master_arg           = amountOfPointsForSkipMaster;
                size_t skip_slave_arg            = amountOfPointsForSkipSlave;
                int    icRandom_int              = req.ic_random_offset ? 1 : 0;
                numb   icEps_arg                 = (numb)req.ic_eps;
                unsigned long long icSeed_arg    = req.ic_seed;
                int    gsWarmup_int              = req.gs_warmup;

                int*   d_cancel_arg       = sig.cancelArg();
                int*   d_progress_arg     = sig.progressArg();
                int    progressStride_arg = progressStride;
                sig.resetTicks();
                void* args_grid[] = {
                    &nPts_arg, &nPtsLimiter_int, &sizeOfBlock_int, &amountOfCalculatedPoints,
                    &dimension_arg, &d_ranges, &h_arg, &d_idx_mv,
                    &d_ic_m, &d_ic_s, &amountOfIC_int,
                    &d_values, &amountOfValues_int,
                    &amountOfIterations_int, &preScaller_int, &maxValue_arg, &iterOfSynchr_int,
                    &d_kF, &d_kB,
                    &d_data, &d_helpful, &d_fs_err_chunk,
                    &swap_role_int, &skip_master_arg, &skip_slave_arg,
                    &icRandom_int, &icEps_arg, &icSeed_arg, &gsWarmup_int
                    ,&d_cancel_arg, &d_progress_arg, &progressStride_arg
                };
                int blockSize = launch_block_size((size_t)(amountOfIC_int + amountOfValues_int) * sizeof(numb));
                int gridSize  = (int)((cur_limiter + blockSize - 1) / blockSize);
                // Shared memory: kernel объявляет `extern __shared__ numb s[]` и кладёт туда
                // localX[amountOfIC] + localValues[amountOfValues] на поток, т.е. blockSize × (IC +
                // values) × 8 байт. Без правильного размера — OOB в shared, illegal memory access и
                // повреждённый (sticky) CUDA-контекст.
                unsigned int shared_grid = (unsigned int)((amountOfIC_int + amountOfValues_int)
                                                          * sizeof(numb) * blockSize);
                CUresult r = cuLaunchKernel(cached_fs_grid.kernel_fs_grid,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            shared_grid, nullptr, args_grid, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(fs_grid): " + cu_err(r); goto FS1_FAIL; }
                if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                       (double)(originalNPtsLimiter * i) * ticksPerCell,
                                       ticksTotal, err)) goto FS1_FAIL;
                cudaDeviceSynchronize();

                // Диверг-флаги чанка: ядро пишет -1 в разлетевшиеся ячейки и 0
                // в остальные, так что буфер определён целиком и читать его
                // безопасно без предварительного обнуления.
                {
                    std::vector<int> h_flags(cur_limiter);
                    if (cudaMemcpy(h_flags.data(), d_helpful, cur_limiter * sizeof(int),
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        for (int f : h_flags) if (f == -1) ++res.diverged_cells;
                    }
                }

                if (req.cancel && req.cancel->load()) { res.cancelled = true; goto FS1_CLEANUP; }
            }

            {
                std::vector<numb> h_out(total_cells);
                FS_GCHECK(cudaMemcpy(h_out.data(), d_fs_err, total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy fs_err D2H");
                res.heatmap.assign(h_out.begin(), h_out.end());      // numb -> double
                res.n_pts_grid = req.n_pts;
                res.total_cells = (int)total_cells;
                res.axis_x_lo = req.axis_x_lo; res.axis_x_hi = req.axis_x_hi;
                res.axis_y_lo = req.axis_y_lo; res.axis_y_hi = req.axis_y_hi;
                res.axis_x_var = req.axis_x_var; res.axis_y_var = req.axis_y_var;

                double vmin = std::numeric_limits<double>::infinity();
                double vmax = -std::numeric_limits<double>::infinity();
                for (double v : res.heatmap) {
                    if (!std::isfinite(v)) continue;
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
                res.min_val = std::isfinite(vmin) ? vmin : 0.0;
                res.max_val = std::isfinite(vmax) ? vmax : 0.0;

                // CSV (On Grid): delegate to data_export so engine and the
                // GUI right-click export share the same writer (and stay
                // byte-identical). Layout: 2-line ranges header + n×n grid.
                if (!req.csv_output_path.empty()) {
                    std::ofstream cfg(req.csv_output_path + "_config.csv");
                    if (cfg.is_open()) {
                        cfg << std::setprecision(set_precision);
                        data_export::write_fastsync_config(cfg, res.snapshot);
                    }
                    std::ofstream csv(req.csv_output_path);
                    if (csv.is_open()) {
                        csv << std::setprecision(set_precision);
                        data_export::write_fastsync_grid(csv, res,
                            req.axis_x_lo, req.axis_x_hi,
                            req.axis_y_lo, req.axis_y_hi);
                    }
                }
            }

            FS1_CLEANUP:
            sig.release();
            cudaFree(d_data); cudaFree(d_ranges); cudaFree(d_idx_mv);
            cudaFree(d_ic_m); cudaFree(d_ic_s); cudaFree(d_values);
            cudaFree(d_kF); cudaFree(d_kB); cudaFree(d_helpful); cudaFree(d_fs_err);
            #undef FS_GCHECK
            if (!res.cancelled) res.ok = true;
            return res;

            FS1_FAIL:
            sig.release();
            if (d_data)    cudaFree(d_data);
            if (d_ranges)  cudaFree(d_ranges);
            if (d_idx_mv)  cudaFree(d_idx_mv);
            if (d_ic_m)    cudaFree(d_ic_m);
            if (d_ic_s)    cudaFree(d_ic_s);
            if (d_values)  cudaFree(d_values);
            if (d_kF)      cudaFree(d_kF);
            if (d_kB)      cudaFree(d_kB);
            if (d_helpful) cudaFree(d_helpful);
            if (d_fs_err)  cudaFree(d_fs_err);
            return fail(err);
        }
    }

    // -----------------------------------------------------------------------
    // Network — сеть связанных осцилляторов (kernels/network.template.cu).

    bool compile_network_module(bool activate, int amountOfX, int amountOfValues,
                                const std::string& krs_body, const std::string& coupling_body,
                                std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело связи входит в ключ: оно подставляется в switch внутри ядра, и
        // смена закона связи — такая же перекомпиляция, как смена КРС.
        const std::string key = hash_key(krs_body, amountOfX) + ":net:v"
                              + std::to_string(amountOfValues)
                              + ":c" + std::to_string(std::hash<std::string>{}(coupling_body));
        return compile_into(pool_network, key, cached_network, activate, [&](CachedNetworkModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template_network), "network.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{COUPLING_BODY}}",    coupling_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "networkIntegrateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

    NetworkResult run_network(const NetworkRequest& req) {
        NetworkResult res;
        auto fail = [&](const std::string& msg) -> NetworkResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if (req.n_nodes <= 0)                                    return fail("the network has no nodes");
        if (req.n_nodes > kMaxNetworkNodes)
            return fail("too many nodes: the whole network runs in one block, the cap is "
                        + std::to_string(kMaxNetworkNodes));
        if (req.amountOfValues <= 0 || req.amountOfValues > kMaxAmountOfValues)
            return fail("amountOfValues out of range");
        if ((int)req.values.size() != req.n_nodes * req.amountOfValues)
            return fail("values.size() != n_nodes * amountOfValues");
        if ((int)req.initial_conditions.size() != req.n_nodes * req.amountOfX)
            return fail("initial_conditions.size() != n_nodes * amountOfX");
        if ((int)req.edge_start.size() != req.n_nodes + 1)
            return fail("edge_start.size() != n_nodes + 1");
        const int n_arcs = req.edge_start.back();
        if (n_arcs < 0 || (int)req.edge_src.size() != n_arcs
            || (int)req.edge_w.size() != n_arcs || (int)req.edge_law.size() != n_arcs)
            return fail("the CSR arrays disagree on the number of arcs");
        for (int i = 0; i < req.n_nodes; ++i)
            if (req.edge_start[(size_t)i] > req.edge_start[(size_t)i + 1])
                return fail("edge_start is not monotonic");
        for (int e = 0; e < n_arcs; ++e)
            if (req.edge_src[(size_t)e] < 0 || req.edge_src[(size_t)e] >= req.n_nodes)
                return fail("an arc points outside the node list");
        if (!(req.h > 0.0))     return fail("h must be > 0");
        if (!(req.t_max > 0.0)) return fail("t_max must be > 0");
        if (req.transient < 0.0) return fail("transient must be >= 0");
        if (req.pre_scaller < 1) return fail("preScaller must be >= 1");
        if (req.max_points < 2)  return fail("max_points must be >= 2");

        // Состояние всей сети живёт в динамической shared — отсюда потолок.
        const size_t shared_bytes = (size_t)req.n_nodes * (size_t)req.amountOfX * sizeof(numb);
        if (shared_bytes > kNetworkSharedCap)
            return fail("the network state does not fit into shared memory ("
                        + std::to_string(shared_bytes) + " B > " + std::to_string(kNetworkSharedCap)
                        + " B): reduce the node count or the system dimension");

        const long long skipSteps  = (long long)steps_from_time_size_t(req.transient, req.h);
        const long long workSteps  = (long long)steps_from_time_size_t(req.t_max, req.h);
        const long long totalSteps = skipSteps + workSteps;
        if (totalSteps <= 0) return fail("t_max / h gives no steps");
        if ((double)totalSteps > 1.0e12)
            return fail("t_max / h too large: the work does not fit into a reasonable time");

        // Прореживание: пользовательское, но поднятое до того, при котором
        // записанное влезает в max_points. Поднимать молча нельзя — фактическое
        // значение уезжает в результат и показывается в UI.
        int preScaller = req.pre_scaller;
        {
            const long long need = workSteps / preScaller + 1;
            if (need > (long long)req.max_points) {
                const long long k = (workSteps + (long long)req.max_points - 2)
                                  / ((long long)req.max_points - 1);
                if (k > preScaller) preScaller = (int)k;
            }
        }
        const long long nPointsLL = workSteps / preScaller + 1;
        if (nPointsLL <= 0 || nPointsLL > 2147483647LL) return fail("the recorded point count is out of range");
        const int nPoints = (int)nPointsLL;

        const size_t out_elems = (size_t)nPoints * (size_t)req.n_nodes * (size_t)req.amountOfX;
        const size_t out_bytes = out_elems * sizeof(numb);
        if (out_bytes > kNetworkOutCap)
            return fail("the recorded trajectory would take " + std::to_string(out_bytes >> 20)
                        + " MB: raise the decimation or lower t_max");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        if (!compile_network_module(true, req.amountOfX, req.amountOfValues,
                                    req.krs_body, req.coupling_body, err)) return fail(err);

        OrderDevBuf d_values, d_state, d_eStart, d_eSrc, d_eW, d_eLaw, d_out, d_status;
        if (!d_values.alloc(req.values.size() * sizeof(numb), "values", err)) return fail(err);
        if (!d_state .alloc(req.initial_conditions.size() * sizeof(numb), "state", err)) return fail(err);
        if (!d_eStart.alloc(req.edge_start.size() * sizeof(int), "edgeStart", err)) return fail(err);
        if (!d_eSrc  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeSrc", err)) return fail(err);
        if (!d_eW    .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(numb), "edgeW",   err)) return fail(err);
        if (!d_eLaw  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeLaw", err)) return fail(err);
        if (!d_out   .alloc(out_bytes, "out", err)) return fail(err);
        if (!d_status.alloc((size_t)req.n_nodes * sizeof(int), "status", err)) return fail(err);

        {
            const std::vector<numb> hv  = to_numb(req.values);
            const std::vector<numb> hic = to_numb(req.initial_conditions);
            const std::vector<numb> hw  = to_numb(req.edge_w);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                if (bytes == 0) return true;
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_values.p, hv.data(),  hv.size()  * sizeof(numb), "values"))    return fail(err);
            if (!up(d_state.p,  hic.data(), hic.size() * sizeof(numb), "state"))     return fail(err);
            if (!up(d_eStart.p, req.edge_start.data(), req.edge_start.size() * sizeof(int), "edgeStart")) return fail(err);
            if (!up(d_eSrc.p,   req.edge_src.data(),   (size_t)n_arcs * sizeof(int),  "edgeSrc")) return fail(err);
            if (!up(d_eW.p,     hw.data(),             (size_t)n_arcs * sizeof(numb), "edgeW"))   return fail(err);
            if (!up(d_eLaw.p,   req.edge_law.data(),   (size_t)n_arcs * sizeof(int),  "edgeLaw")) return fail(err);
        }
        // 0xFF по всем байтам — это NaN и для float, и для double: точки после
        // разлёта обязаны читаться как «не посчитано», а не как ноль.
        if (cudaMemset(d_out.p, 0xFF, out_bytes) != cudaSuccess) return fail("memset out");
        if (cudaMemset(d_status.p, 0, (size_t)req.n_nodes * sizeof(int)) != cudaSuccess)
            return fail("memset status");

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Один блок = одна сеть, поток = узел. Идущие сверх узлов потоки стоят
        // на тех же __syncthreads, что и рабочие, и ничего не считают.
        int blockSize = ((req.n_nodes + 31) / 32) * 32;
        if (blockSize > 1024) blockSize = 1024;

        // Чанкование против watchdog'а (TDR): расчёт последователен по времени,
        // поэтому режем его по шагам, а состояние переносится через d_state.
        const double workPerStep = (double)req.n_nodes * (1.0 + (double)n_arcs / (double)req.n_nodes);
        long long stepsPerLaunch = (long long)(kNetworkWorkBudget / (workPerStep > 1.0 ? workPerStep : 1.0));
        if (stepsPerLaunch < 1)          stepsPerLaunch = 1;
        if (stepsPerLaunch > totalSteps) stepsPerLaunch = totalSteps;

        const int    progressStride = progress_stride_for((size_t)totalSteps);
        const double ticksTotal     = (double)totalSteps / (double)progressStride;
        double ticksDone = 0.0;

        for (long long base = 0; base < totalSteps; base += stepsPerLaunch) {
            const long long left  = totalSteps - base;
            const long long count = (stepsPerLaunch < left) ? stepsPerLaunch : left;

            numb*     values_arg = d_values.as<numb>();
            numb*     state_arg  = d_state.as<numb>();
            int*      eStart_arg = d_eStart.as<int>();
            int*      eSrc_arg   = d_eSrc.as<int>();
            numb*     eW_arg     = d_eW.as<numb>();
            int*      eLaw_arg   = d_eLaw.as<int>();
            int       nNodes_arg = req.n_nodes;
            numb      h_arg      = (numb)req.h;
            long long base_arg   = base;
            long long count_arg  = count;
            long long skip_arg   = skipSteps;
            int       pre_arg    = preScaller;
            int       nPts_arg   = nPoints;
            numb      maxV_arg   = (numb)req.max_value;
            numb*     out_arg    = d_out.as<numb>();
            int*      status_arg = d_status.as<int>();
            int*      cancel_arg = sig.cancelArg();
            int*      prog_arg   = sig.progressArg();
            int       stride_arg = progressStride;

            void* args[] = {
                &values_arg, &state_arg, &eStart_arg, &eSrc_arg, &eW_arg, &eLaw_arg,
                &nNodes_arg, &h_arg, &base_arg, &count_arg, &skip_arg, &pre_arg, &nPts_arg,
                &maxV_arg, &out_arg, &status_arg, &cancel_arg, &prog_arg, &stride_arg
            };

            sig.resetTicks();
            CUresult r = cuLaunchKernel(cached_network.kernel, 1, 1, 1, blockSize, 1, 1,
                                        (unsigned)shared_bytes, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(network): " + cu_err(r));
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("network kernel: ") + cudaGetErrorString(ce));

            ticksDone += (double)count / (double)progressStride;

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            // Разлетелась сеть — остальные чанки досчитывать нечего: ядро на
            // них всё равно сразу выйдет, а пользователь ждал бы впустую.
            std::vector<int> st((size_t)req.n_nodes, 0);
            if (cudaMemcpy(st.data(), d_status.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H status");
            bool diverged = false;
            for (int v : st) if (v != NET_ST_OK) diverged = true;
            if (diverged) break;
        }

        res.status.assign((size_t)req.n_nodes, 0);
        if (cudaMemcpy(res.status.data(), d_status.p, res.status.size() * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H status");
        for (int v : res.status) if (v != NET_ST_OK) ++res.n_diverged;

        {
            std::vector<numb> host(out_elems);
            if (cudaMemcpy(host.data(), d_out.p, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H out");
            res.data.resize(out_elems);
            res.vmin.assign((size_t)req.amountOfX, 0.0);
            res.vmax.assign((size_t)req.amountOfX, 0.0);
            std::vector<bool> seen((size_t)req.amountOfX, false);
            for (size_t i = 0; i < out_elems; ++i) {
                const double v = (double)host[i];
                res.data[i] = v;
                if (!std::isfinite(v)) continue;
                const size_t k = i % (size_t)req.amountOfX;
                if (!seen[k]) { res.vmin[k] = res.vmax[k] = v; seen[k] = true; }
                else { if (v < res.vmin[k]) res.vmin[k] = v; if (v > res.vmax[k]) res.vmax[k] = v; }
            }
        }

        res.n_nodes     = req.n_nodes;
        res.n_vars      = req.amountOfX;
        res.n_points    = nPoints;
        res.h           = req.h;
        res.t0          = (double)skipSteps * req.h;
        res.dt          = (double)preScaller * req.h;
        res.pre_scaller = preScaller;
        res.ok          = true;
        return res;
    }
};

ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
}

Dft1DResult ParametricEngine::run_dft_1d(const Dft1DRequest& req) {
    return impl_->run_dft_1d(req);
}

Bifurcation2DResult ParametricEngine::run_bifurcation_2d(const Bifurcation2DRequest& req) {
    return impl_->run_bif2d(req);
}

LLE1DResult ParametricEngine::run_lle_1d(const LLE1DRequest& req) {
    return impl_->run_lle_1d(req);
}

LS1DResult ParametricEngine::run_ls_1d(const LS1DRequest& req) {
    return impl_->run_ls_1d(req);
}

LLE2DResult ParametricEngine::run_lle_2d(const LLE2DRequest& req) {
    return impl_->run_lle_2d(req);
}

LS2DResult ParametricEngine::run_ls_2d(const LS2DRequest& req) {
    return impl_->run_ls_2d(req);
}

BasinsResult ParametricEngine::run_basins(const BasinsRequest& req) {
    return impl_->run_basins(req);
}

BasinsReclusterResult ParametricEngine::run_basins_recluster(const BasinsReclusterRequest& req) {
    return impl_->run_basins_recluster(req);
}

PerfResult ParametricEngine::run_performance(const PerfRequest& req) {
    return impl_->run_performance(req);
}

OrderResult ParametricEngine::run_order(const OrderRequest& req) {
    return impl_->run_order(req);
}

StabilityResult ParametricEngine::run_stability(const StabilityRequest& req) {
    return impl_->run_stability(req);
}

std::string stability_validate(const StabilityRequest& req) {
    if (req.krs_body.empty()) return "krs_body is empty";
    if (req.amountOfX != 2)
        return "the stability region needs the two-dimensional Dahlquist test system "
               "(Library -> Dahlquist 2D): this one has " + std::to_string(req.amountOfX)
               + " state variables";
    if (req.values.empty())                          return "values is empty (at least a[0] is required)";
    if ((int)req.values.size() > kMaxAmountOfValues) return "too many values";

    const int   amountOfValues = (int)req.values.size();
    const int   idx[4]      = { req.idx_a, req.idx_b, req.idx_c, req.idx_d };
    const char* idx_name[4] = { "a", "b", "c", "d" };
    for (int i = 0; i < 4; ++i) {
        // a[0] занят коэффициентом симметрии s во всех расчётах проекта, и
        // положить туда элемент матрицы значило бы молча сломать схему.
        if (idx[i] < 1 || idx[i] >= amountOfValues)
            return std::string("matrix element ") + idx_name[i]
                 + ": a[" + std::to_string(idx[i]) + "] is outside the parameters "
                   "(a[0] is the symmetry s and cannot hold a matrix element)";
        for (int j = 0; j < i; ++j)
            if (idx[i] == idx[j])
                return std::string("matrix elements ") + idx_name[j] + " and " + idx_name[i]
                     + " point at the same a[" + std::to_string(idx[i]) + "]";
    }
    if (req.k == -1.0) return "k = -1 makes the matrix undefined: d = 2*sigma/(1+k)";
    if (!(req.r < 0.0))
        return "r must be negative: with r >= 0 the radicand of c is not positive and "
               "the test matrix has no real form";
    if (!(req.h > 0.0))   return "h must be > 0";
    if (req.n_pts <= 0)   return "the number of points must be > 0";
    if (req.n_pts > 8192) return "the number of points is too large";
    return {};
}

void stability_fill_axes(const StabilityRequest& req, StabilityResult& res) {
    res.n_pts_x = res.n_pts_y = req.n_pts;
    res.sigma_lo = req.sigma_lo; res.sigma_hi = req.sigma_hi;
    res.omega_lo = req.omega_lo; res.omega_hi = req.omega_hi;
    res.k = req.k; res.r = req.r; res.h = req.h;

    auto nodes = [](double lo, double hi, int n, std::vector<double>& out) {
        out.assign((size_t)n, lo);
        if (n < 2) return;
        const double d = (hi - lo) / (double)(n - 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = lo + d * (double)i;
    };
    nodes(req.sigma_lo, req.sigma_hi, res.n_pts_x, res.sigma_vals);
    nodes(req.omega_lo, req.omega_hi, res.n_pts_y, res.omega_vals);
}

void stability_summarize(StabilityResult& res) {
    res.n_ok = res.n_bad = res.n_stable = 0;
    res.rho_min = res.rho_max = 0.0;
    res.err_min = res.err_max = 0.0;
    bool first = true, first_err = true;
    for (size_t i = 0; i < res.rho.size(); ++i) {
        if (i < res.status.size() && res.status[i] != STAB_ST_OK) { ++res.n_bad; continue; }
        if (!std::isfinite(res.rho[i])) { ++res.n_bad; continue; }
        ++res.n_ok;
        if (res.rho[i] <= 1.0) ++res.n_stable;
        if (first) { res.rho_min = res.rho_max = res.rho[i]; first = false; }
        else {
            if (res.rho[i] < res.rho_min) res.rho_min = res.rho[i];
            if (res.rho[i] > res.rho_max) res.rho_max = res.rho[i];
        }
        if (i < res.err.size() && std::isfinite(res.err[i])) {
            const double e = res.err[i];
            if (first_err) { res.err_min = res.err_max = e; first_err = false; }
            else {
                if (e < res.err_min) res.err_min = e;
                if (e > res.err_max) res.err_max = e;
            }
        }
    }
}

// Спектральная норма 2x2 в замкнутом виде: sigma_max = hypot(E, H) + hypot(F, G)
// с E, F — полусуммой и полуразностью диагонали, G, H — то же для
// недиагонали. В отличие от sqrt((|M|_F^2 + sqrt(|M|_F^4 - 4 det^2)) / 2) здесь
// нет вычитания близких чисел.
static double stab_norm2(double m00, double m01, double m10, double m11) {
    const double E = 0.5 * (m00 + m11), F = 0.5 * (m00 - m11);
    const double G = 0.5 * (m10 + m01), H = 0.5 * (m10 - m01);
    return std::hypot(E, H) + std::hypot(F, G);
}

double stability_step_error(double sigma, double omega, double k, double r, double h,
                            double R00, double R01, double R10, double R11) {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(R00) && std::isfinite(R01) && std::isfinite(R10) && std::isfinite(R11)))
        return qnan;

    // Матрица — ровно та, что в stabilityRegionKernel.
    const double kp1 = k + 1.0, km1 = k - 1.0;
    if (kp1 == 0.0 || r == 0.0) return qnan;
    const double rad = -(1.0 / r) * (sigma * sigma * km1 * km1 / (kp1 * kp1) + omega * omega);
    if (!(rad >= 0.0)) return qnan;
    const double dEl = 2.0 * sigma / kp1;
    const double cEl = -std::sqrt(rad);
    const double bEl = r * cEl;
    const double aEl = k * dEl;

    // (A - sigma I)^2 = -omega^2 I (Кэли-Гамильтон при tr A = 2 sigma,
    // det A = sigma^2 + omega^2), отсюда замкнутая форма экспоненты. При
    // omega -> 0 множитель sin(h omega)/omega -> h; ряд до x^4 держит там
    // полную точность double.
    const double x = h * omega;
    const double sinc = (std::fabs(x) < 1e-4) ? 1.0 - x * x / 6.0 * (1.0 - x * x / 20.0)
                                              : std::sin(x) / x;
    const double co = std::cos(x);
    const double sc = h * sinc;
    const double g  = std::exp(h * sigma);
    const double E00 = g * (co + sc * (aEl - sigma));
    const double E01 = g * (sc * bEl);
    const double E10 = g * (sc * cEl);
    const double E11 = g * (co + sc * (dEl - sigma));

    const double ne = stab_norm2(E00, E01, E10, E11);
    if (!(ne > 0.0) || !std::isfinite(ne)) return std::numeric_limits<double>::infinity();
    const double nd = stab_norm2(E00 - R00, E01 - R01, E10 - R10, E11 - R11);
    if (!std::isfinite(nd)) return std::numeric_limits<double>::infinity();
    return nd / ne;
}

int stability_count_preferred(const StabilityResult& res, double tol) {
    int n = 0;
    for (size_t i = 0; i < res.rho.size(); ++i)
        if (stability_cell_preferred(res, i, tol)) ++n;
    return n;
}

NetworkResult ParametricEngine::run_network(const NetworkRequest& req) {
    return impl_->run_network(req);
}

FastSyncResult ParametricEngine::run_fastsync(const FastSyncRequest& req) {
    return impl_->run_fastsync(req);
}

void ParametricEngine::prewarm(const Bifurcation1DRequest& req) { impl_->prewarm_bif1d(req); }
void ParametricEngine::prewarm(const Bifurcation2DRequest& req) { impl_->prewarm_bif2d(req); }
void ParametricEngine::prewarm(const LLE1DRequest& req)         { impl_->prewarm_lle1d(req); }
void ParametricEngine::prewarm(const LLE2DRequest& req)         { impl_->prewarm_lle2d(req); }
void ParametricEngine::prewarm(const LS1DRequest& req)          { impl_->prewarm_ls1d(req); }
void ParametricEngine::prewarm(const LS2DRequest& req)          { impl_->prewarm_ls2d(req); }
void ParametricEngine::prewarm(const BasinsRequest& req)        { impl_->prewarm_basins(req); }
