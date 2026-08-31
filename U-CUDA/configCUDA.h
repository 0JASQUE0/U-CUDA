#pragma once
#include <math_constants.h>

typedef float numb;

// AMOUNTOFX оборачивается в #ifndef, чтобы NVRTC-вызывающие проекты могли
// переопределить размерность системы через #define AMOUNTOFX N перед include.
// Обычная nvcc-сборка ничего не определяет — поведение прежнее.
#ifndef AMOUNTOFX
constexpr int AMOUNTOFX = 3;
#endif
constexpr int CHECK_INTERVAL = 100;

// ---------------------------------------------------------------------------
// --- REGIME CODES — единая индикация режима траектории для ВСЕХ расчётов ---
//   -1 — fixed point (схлопнулась в неподвижную точку, |Δx| < eps_fixed_point)
//    0 — unbound     (nan/inf или |x| > maxValue)
//    1 — oscillation (нормальный колебательный режим)
// Источник — loopCalculateDiscreteModel_int; коды живут в maxValueCheckerArray /
// helpfulArray / *Result::flags. Там, где flags[] хранит СЫРОЙ выход
// peakFinder / DBSCAN, N > 0 — это число пиков (период), т.е. OSCILLATION;
// нормализация — regime_code() в parametric_engine.h.
// ---------------------------------------------------------------------------
constexpr int REGIME_FIXED_POINT = -1;
constexpr int REGIME_UNBOUND     =  0;
constexpr int REGIME_OSCILLATION =  1;

// --- Calculate peaks ---
// 1 -- Yes;
// 0 -- No
// Здесь и ниже (все knobs, настраиваемые из GUI/Settings) — #ifndef, чтобы
// NVRTC-сборка могла переопределить через #define перед include configCUDA.h.
// Обычная nvcc-сборка (Debug / legacy main_NonLinAnal.cu) ничего не определяет
// и получает прежние дефолты.
#ifndef doCalculatePeaks
constexpr bool doCalculatePeaks = 1;
#endif

// --- data[startDataIndex + i] = xDataMultiplier * (x[writableVar]); ---
constexpr numb xDataMultiplier = 1.0;

// --- Parabolic interpolation of Peaks and InterPeaks
// 1 -- Yes
// 0 -- No
#ifndef doInterpolatePeaks
constexpr bool doInterpolatePeaks = 1;
#endif

// --- CHOOSE CLASSICAL OR CONTINUATION BIFURCATION DIAGRAM ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation and DFT diagram)
constexpr bool continuation_bif1D = 0;

// --- CHOOSE PARAMETER OR VARIABLE ANALYSIS ---
// 2 -- variable and parameter analysis (first variable, second parameter)
// 1 -- parameter analysis;
// 0 -- variable analysis
// Обёрнуто в #ifndef, чтобы NVRTC-template мог переопределить через #define.
#ifndef par_or_var
constexpr int par_or_var = 1;
#endif

// --- CHOOSE LIN OR LOG variable/parameter distribution ---
// 1 -- lin; [Xmin,Xmax] or [Xmin,Xmax,Ymin,Ymax]
// 0 -- log; [log10(Xmin)
constexpr bool LINEAR_OR_LOG_DISTRIB = 1;


// --- CHOOSE AVG OR LOG10(AVG) ANALYSIS ---
// 1 -- avg peak analysis;
// 0 -- log10(avg peak) analysis
constexpr bool lin_or_log = 0;

// --- FAST SYNCHRO knobs (NVRTC-overridable) ---
// Wrapped в #ifndef чтобы NVRTC-template мог переопределить через #define
// перед #include "configCUDA.h" — даём GUI настраивать per-run. Без override
// (Debug / legacy main_NonLinAnal.cu) — старые дефолты.
#ifndef type_of_synch
// 0 = unidirectional synchro; 1 = bidirectional synchro
constexpr int type_of_synch = 0;
#endif
#ifndef error_estim
// 0 = RMS(error) on last iter; 1 = #iters to reach FS_error_trs; 2 = RMS at last point on last iter
constexpr int error_estim = 2;
#endif
#ifndef FS_error_trs
constexpr numb FS_error_trs = 1e-12;
#endif

// --- CALCULATE ADDITIONALLY MEAN AND MEDIAN FREQUENCY? ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation diagram)
constexpr bool calculate_mean_med_freq = 0; 

// --- CALCULATE ADDITIONALLY MEAN AND VARIANCE OF CHOOSEN VARIABLE? ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation diagram)
constexpr bool calculate_mean_and_variance = 0; 

// --- CALCULATE GLOBAL PEAK VALUE? ---
// 1 -- yes; 
// 0 -- no (only for 2D bifurcation diagram)
constexpr bool calculate_global_peak = 0;

// --- THRESHOLDs FOR ESTIMATION ---
// Все пять настраиваются из GUI (Settings) — см. #ifndef-комментарий выше.
#ifndef eps_fixed_point
constexpr numb eps_fixed_point = 1e-6;		// Minimal diffrence between two last points to mark it fixed point regime
#endif
#ifndef eps_peak_delta
constexpr numb eps_peak_delta = 1e-14;		// minimal differnece between two neibor points to mark it peak
#endif
#ifndef eps_interPeak_delta
constexpr numb eps_interPeak_delta = 0.0;	// minimal interspike interval between two peak points (0 = фильтр выключен)
#endif
#ifndef peak_threshold
constexpr numb peak_threshold = -1e25;	// threshold for peakfinder
#endif
#ifndef max_amount_of_peaks
// Потолок числа пиков, которые peakFinder оставляет на одну точку свипа.
// Раньше здесь стояло предупреждение про local memory: значение задавало
// размер per-thread массивов next[] / labels[] в dbscan_optimized. Те ядра
// удалены как мёртвые, рабочий путь (dbscanCUDA -> dbscan) локальных массивов
// не держит, так что occupancy это больше не задевает — влияние осталось
// только на размер device-буферов пиков и интервалов.
constexpr int max_amount_of_peaks = 2500;	// max amount of peaks for estimation for peakfinder
#endif

// --- MULTIPLIERS FOR DBSCAN (2D bifurcation diagrams) ---
constexpr numb mult_peak  = 1.0; // multiplier for peak values for DBSCAN //1000
constexpr numb mult_interval = 0.0; // multiplier for interval values for DBSCAN //4

// --- MULTIPLIERS FOR DBSCAN (Basins of attraction) ---
// Дефолты для mult1/mult2 в avgPeakFinderCUDA. Используются как default args
// kernel'а, в headless call site (hostLibrary.cu) и как initial value для
// UI-полей mult_feature1_text / mult_feature2_text в BasinsConfig.
constexpr numb mult_avg_peak = 1.0; // multiplier for average peak values for DBSCAN (basins of attraction)
constexpr numb mult_avg_interval = 1.0; // multiplier for average interval values for DBSCAN (basins of attraction)

// ---------------------------------------------------------------------------
// avgPeakFinderCUDA — feature codes. Источник истины для switch'а внутри
// kernel'а И для headless callers (без UI), которые должны явно указать,
// какие фичи писать в outAvgPeaks / AvgTimeOfPeaks. Host-side mirror —
// enum class BasinFeature в analysis_session.h (значения совпадают;
// static_assert'ы в analysis_session.cpp ловят drift).
//
// Значения features:
//   AVG     — mean(peaks/intervals)
//   RMS     — sqrt(mean(x²))
//   STDEV   — sqrt(mean(x²) - mean(x)²)
//   LOG_AVG — sign(mean) * log10(|mean|), 0 при mean==0
//   LOG_RMS — log10(RMS), -999 при RMS==0
//   LOG_STDEV — log10(StDev), -999 при StDev==0
// ---------------------------------------------------------------------------
constexpr int BF_AVG_PEAKS            = 0;
constexpr int BF_AVG_INTERVALS        = 1;
constexpr int BF_RMS_PEAKS            = 2;
constexpr int BF_RMS_INTERVALS        = 3;
constexpr int BF_STDEV_PEAKS          = 4;
constexpr int BF_STDEV_INTERVALS      = 5;
constexpr int BF_LOG_AVG_PEAKS        = 6;
constexpr int BF_LOG_AVG_INTERVALS    = 7;
constexpr int BF_LOG_RMS_PEAKS        = 8;
constexpr int BF_LOG_RMS_INTERVALS    = 9;
constexpr int BF_LOG_STDEV_PEAKS      = 10;
constexpr int BF_LOG_STDEV_INTERVALS  = 11;
constexpr int BF_FEATURE_COUNT        = 12;

// Дефолты — семантически равны pre-feature-selection поведению (mean пиков,
// mean интервалов). Подставляются как default args kernel'а, в UI и в
// initial value BasinsConfig::feature1 / feature2.
constexpr int BF_FEATURE1_DEFAULT = BF_AVG_PEAKS;
constexpr int BF_FEATURE2_DEFAULT = BF_AVG_INTERVALS;

constexpr int blockSize_setup = 32; // default blockSize value
constexpr int set_precision  = 15; // precision of numbers in writng final csv files

// ---  ---  ---  ---  --- Fast Synchro ---  ---  ---  ---  ---
// type_of_synch / error_estim / FS_error_trs объявлены выше (в #ifndef-блоках),
// чтобы NVRTC-template мог их переопределить через #define перед include.

constexpr int amount_GPU = 1920; // precision of numbers in writng final csv files
constexpr numb pi	 = 3.1415926535897932384626433832795;
constexpr numb euler = 2.7182818284590452353602874713527;
//constexpr int blockSize_fixed = 1024;

// ---------------------------------------------------------------------------
// --- ЗНАЧЕНИЕ УЗЛА ПАРАМЕТРИЧЕСКОЙ СЕТКИ — ЕДИНСТВЕННАЯ РЕАЛИЗАЦИЯ ---
//
// Ровно эта арифметика определяет, в какой точке параметра посчитана ячейка
// карты. Раньше формула была написана от руки в ВОСЬМИ местах и в ЧЕТЫРЁХ
// алгебраически эквивалентных, но численно различных формах:
//   ядро (getValueByIdx)                   (1-t)*lo + t*hi
//   host-копия для CSV (parametric_engine) lo + (hi-lo)*k/(n-1)
//   heatmap (snap/tooltip/drill-down)      lo + k*step, step=(hi-lo)/(n-1)
//   grid_snap, sweep_value_at              снова свои
// и в логарифмических вариантах ещё и порядок умножения/деления отличался:
// ядро делит шаг ДО умножения на k, копии — после.
//
// Формы расходятся не только в последнем бите: при k = n-1 lerp даёт РОВНО hi,
// а lo + (hi-lo)*k/(n-1) — не обязательно. Практическое следствие: snap по
// пикселю отдавал в фазовый портрет и в 1D-срезы значение параметра, в котором
// эта ячейка никогда не считалась.
//
// Конвенция continuation-свипов (lo + (hi-lo)*t, плюс reverse) — ДРУГАЯ и
// остаётся своей: она живёт в kernels/*_cont.template.cu и в cont_sweep_value
// (parametric_engine.cpp). Те шаблоны configCUDA.h не включают, поэтому здесь
// её нет — вызывающий обязан выбрать функцию под то ядро, которое считало данные.
//
// UCUDA_HD: MSVC не понимает __host__ __device__, а nvcc/NVRTC обязаны их
// видеть. Этот заголовок парсится и тем, и другим (и копируется в kernels/
// post-build'ом), поэтому квалификаторы — через макрос.
// ---------------------------------------------------------------------------
// NVRTC определяет __CUDACC_RTC__ автоматически (см. cudaMacros.cuh); проверяем
// оба макроса, чтобы квалификаторы гарантированно попали и в runtime-сборку —
// иначе device-код звал бы host-функцию, и это вылезло бы только при Run.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define UCUDA_HD __host__ __device__
#else
#define UCUDA_HD
#endif
// pow/log10: под NVRTC они builtin, файловые заголовки там не нужны и подтянуть
// CRT-шный math.h в device-режиме нельзя — тот же гард, что у host-only
// includes в cudaLibrary.cu / cudaMacros.cuh.
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

// Линейная сетка: узел k из nPts на [lo, hi], inclusive по обоим концам.
// Вырожденные случаи повторяют прежний getValueByIdx (nPts == 1 -> hi): это то,
// что ядро реально симулирует. Host-копии возвращали здесь lo и расходились
// с расчётом на односегментной сетке.
UCUDA_HD inline numb ucuda_node_value(int k, int nPts, numb lo, numb hi) {
    if (nPts <= 0) return lo;
    if (nPts == 1) return hi;
    const numb t = (numb)k / (numb)(nPts - 1);
    return ((numb)1.0 - t) * lo + t * hi;
}

// Внутренняя лог-арифметика — ВСЕГДА в double, независимо от numb. Дело не в
// точности, а в согласованности host/device: при numb = float перегрузка
// log10/pow на host-стороне (MSVC math.h) выбирается double'ная, а в device-коде
// — float'овая, и одна и та же формула давала РАЗНЫЕ значения узла (замерено:
// ~1 float-ulp расхождения). В double-режиме — а это текущий numb — здесь ровно
// те же операции, что стояли в getValueByIdxLog, и значения не меняются ни на бит.
// Порядок операций тот же: шаг делится ДО умножения на k.
UCUDA_HD inline double ucuda_node_log10_d(int k, int nPts, double lo, double hi) {
    const double l0 = log10(lo);
    if (nPts <= 1) return l0;
    return l0 + ((double)k * ((log10(hi) - l0) / (double)(nPts - 1)));
}

// Лог-сетка, значение В log10-МАСШТАБЕ — в нём работает getValueByIdxLog и в
// нём же лежат X-координаты лог-осей в VBO.
UCUDA_HD inline numb ucuda_node_value_log10(int k, int nPts, numb lo, numb hi) {
    return (numb)ucuda_node_log10_d(k, nPts, (double)lo, (double)hi);
}

// Лог-сетка, значение в исходном масштабе. Требует lo, hi > 0 — проверяется на
// host-стороне до запуска ядра (при lo <= 0 вызывающий деградирует на линейную
// сетку, см. grid_snap.h / heatmap_view.cpp).
UCUDA_HD inline numb ucuda_node_value_log(int k, int nPts, numb lo, numb hi) {
    if (nPts <= 1) return lo;
    return (numb)pow(10.0, ucuda_node_log10_d(k, nPts, (double)lo, (double)hi));
}

// Continuation-свип: у него СВОЯ конвенция узлов, не совпадающая с классической
// (там lerp, здесь lo + (hi-lo)*t). Менять её нельзя — по ней посчитаны все
// continuation-диаграммы. Точка k цепочки: forward — lo + (hi-lo)*t,
// reverse (гистерезисный проход от hi к lo) — hi - (hi-lo)*t, где
// t = k/denom, denom = nPts > 1 ? nPts-1 : 1 (при nPts == 1 даёт lo/hi).
//
// Была тремя рукописными копиями: kernels/*_cont.template.cu (device),
// cont_sweep_value в parametric_engine.cpp (CPU-continuation + CSV) и
// cont-ветка sweep_value_at в gui.cpp (оси графиков). Копии совпадали в
// double-режиме и расходились во float: host считал в double, device — в numb.
// Линейная ветка идёт в numb (как в ядре), лог-ветка внутри в double — по той
// же причине, что ucuda_node_log10_d.
UCUDA_HD inline numb ucuda_node_value_cont(int k, int nPts, numb lo, numb hi,
                                           bool log_scale, bool reverse) {
    const numb denom = (numb)(nPts > 1 ? nPts - 1 : 1);
    const numb t = (numb)k / denom;
    if (log_scale) {
        const double l0 = log10((double)lo), l1 = log10((double)hi);
        const double td = (double)t;
        return (numb)pow(10.0, reverse ? (l1 - (l1 - l0) * td) : (l0 + (l1 - l0) * td));
    }
    return reverse ? (hi - (hi - lo) * t) : (lo + (hi - lo) * t);
}