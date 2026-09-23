// signal_metrics.template.cu
// NVRTC-шаблон вкладки Parametric -> Metrics: на каждой точке свипа (1D-отрезок
// или 2D-сетка) — числовые характеристики записанного сигнала writable_var:
// максимум, минимум, размах, среднее, средняя и медианная частота по пикам и
// параметры Хьорта (activity / mobility / complexity).
//
// Траектория не хранится: ядро потребляет сэмплы по мере интегрирования — как
// calculateDiscreteModelPeaksCUDA (bifurcation2d.template.cu). Пики ищет тот же
// PeakStream, что у бифуркационных диаграмм, поэтому частоты считаются ровно по
// тем пикам, которые видны на БД при тех же настройках Settings -> Peaks.
// В глобальной памяти остаются только межпиковые интервалы — и только когда
// нужна медианная частота (медиане нужен весь ряд, остальному хватает сумм).
//
// Плейсхолдеры (без двойных фигурных скобок в комментариях — replace_all
// заменит их прямо тут):
//   AMOUNT_OF_X — размерность системы (число переменных, int)
//   KRS_BODY    — тело calculateDiscreteModel из codegen
//   PAR_OR_VAR  — 1 (оси по param), 0 (по IC), 2 (X = IC, Y = param)

#define AMOUNTOFX {{AMOUNT_OF_X}}

#define par_or_var {{PAR_OR_VAR}}

#ifdef __CUDACC_RTC__
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#endif

// curand_kernel.h: перехват для CUDA 13+ — см. bifurcation1d.template.cu.
#ifdef __CUDACC_RTC__
#define CURAND_KERNEL_H_
#define CURAND_KERNEL_H
typedef struct { unsigned long long state; } curandState_t;
typedef curandState_t curandStateXORWOW_t;
__device__ __forceinline__ void curand_init(
    unsigned long long seed, unsigned long long, unsigned long long, curandState_t* s) {
    s->state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
}
__device__ __forceinline__ float curand_uniform(curandState_t* s) {
    s->state = s->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(((s->state >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}
#endif

#include "cudaLibrary.cuh"

__device__ __host__ __forceinline__
void calculateDiscreteModel(numb* X, const numb* a, const numb h) {
{{KRS_BODY}}
}

#include "cudaLibrary.cu"

// Номера метрик — строки выходного буфера. Обязаны совпадать с SignalMetric
// в parametric_engine.h: хост читает строку m как метрику m.
#define SIGM_MAX               0
#define SIGM_MIN               1
#define SIGM_RANGE             2
#define SIGM_MEAN              3
#define SIGM_MEAN_FREQ         4
#define SIGM_MEDIAN_FREQ       5
#define SIGM_VARIANCE          6
#define SIGM_HJORTH_MOBILITY   7
#define SIGM_HJORTH_COMPLEXITY 8
#define SIGM_INT_MAX           9
#define SIGM_INT_MIN           10
#define SIGM_INT_RANGE         11
#define SIGM_INT_MEAN          12
#define SIGM_COUNT             13

// Потоковые суммы для экстремумов, среднего и параметров Хьорта.
//
// Производные — конечные разности записанного ряда: d1 = Δy, d2 = Δ²y. Шаг
// сэмплов dt в суммы не входит: в mobility он даёт множитель 1/dt, в complexity
// сокращается, поэтому делить на него один раз в конце и точнее, и дешевле.
//
// Дисперсия y считается от сдвига на первый сэмпл: сумма квадратов в лоб
// теряет знаки, когда колебание мало по сравнению со средним (x ~ 1e3 ± 1).
// У d1 и d2 среднее почти ноль само по себе, им сдвиг не нужен.
struct MetricsAccum
{
	size_t n;
	numb   shift, s1, s2;
	numb   mn, mx;
	numb   yPrev, dPrev;
	numb   sd, sd2;
	numb   sdd, sdd2;

	__device__ __host__ void init()
	{
		n = 0;
		shift = s1 = s2 = (numb)0;
		mn = mx = (numb)0;
		yPrev = dPrev = (numb)0;
		sd = sd2 = sdd = sdd2 = (numb)0;
	}

	__device__ __host__ void push(numb y)
	{
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

// Дисперсия по суммам (смещённая, 1/n — как var() на скриншоте Хьорта).
__device__ __host__ __forceinline__ numb smVariance(numb s1, numb s2, size_t n)
{
	if (n == 0) return (numb)0;
	const numb inv = (numb)1 / (numb)n;
	const numb v = (s2 - s1 * s1 * inv) * inv;
	return v < (numb)0 ? (numb)0 : v;
}

// Суммы по межпиковым интервалам, принятым PeakStream в счёт (count()): для
// средней частоты (sum 1/T) и для статистик самих интервалов. k — номер
// интервала, первый задаёт начальные min/max.
struct IntervalStats
{
	numb sumInv, sum, mn, mx;

	__device__ __host__ void init() { sumInv = sum = mn = mx = (numb)0; }

	__device__ __host__ void push(numb T, int k)
	{
		if (k == 0) { mn = T; mx = T; }
		else {
			if (T < mn) mn = T;
			if (T > mx) mx = T;
		}
		sum += T;
		if (T > (numb)0) sumInv += (numb)1 / T;
	}
};

// loopCalculateDiscreteModelPeaks_int + MetricsAccum. Тот же сэмпл, те же
// проверки расходимости и тот же вердикт fixed point, что у БД.
//
// ist — статистики межпиковых интервалов, которые PeakStream принял в счёт
// (count()). Интервал восстанавливается из сдвига опорного пика ровно той же
// формулой, что в PeakStream::emitInterval, поэтому совпадает побитово, и ни
// средней частоте, ни min/max/mean интервалов буфер не нужен.
__device__ int loopCalculateDiscreteModelMetrics_int(
	numb* x, const numb* values,
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, PeakStream& peaks, MetricsAccum& acc, IntervalStats& ist,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported)
{
	numb xPrev[AMOUNTOFX];
	numb checker;
	size_t sinceReport = 0;

	for (size_t i = 0; i < amountOfIterations; ++i)
	{
		numb sample;
		if (writableVar < 0) {
			if constexpr (AMOUNTOFX >= 3)
				sample = x[0] + pi*x[1] + euler*x[2];
			else if constexpr (AMOUNTOFX == 2)
				sample = x[0] + pi*x[1];
			else
				sample = x[0];
		} else {
			sample = x[writableVar];
		}

		acc.push(sample);

		if (peaks.emitAll) {
			peaks.pushRaw(sample);
		} else {
			const int  emitted0 = peaks.emitted;
			const numb anchor0  = peaks.anchorTime;
			peaks.push(sample);
			if (peaks.emitted != emitted0 && emitted0 < max_amount_of_peaks) {
				const numb delta = (peaks.anchorTime - anchor0) * peaks.sampleStep;
				ist.push(delta, emitted0);
			}
		}

		for (int j = 0; j < preScaller; ++j)
			calculateDiscreteModel(x, values, h);

		if (i % CHECK_INTERVAL == 0) {
			if (progressCounter != nullptr && progressStride > 0) {
				sinceReport += CHECK_INTERVAL;
				if (sinceReport >= (size_t)progressStride) {
					atomicAdd(progressCounter, 1);
					if (ticksReported != nullptr) ++(*ticksReported);
					sinceReport = 0;
				}
			}
			if (cancelFlag != nullptr && *cancelFlag != 0) return REGIME_UNBOUND;
			checker = 0;
			for (int j = 0; j < AMOUNTOFX; ++j)
				checker = checker + abs(x[j]);

			if (isnan(checker) || isinf(checker))
				return REGIME_UNBOUND;

			if (maxValue != 0)
				if (abs(checker) > maxValue)
					return REGIME_UNBOUND;
		}
	}

	for (int j = 0; j < AMOUNTOFX; ++j)
		xPrev[j] = x[j];

	for (int j = 0; j < preScaller; ++j)
		calculateDiscreteModel(x, values, h);

	numb tempResult = 0;
	for (int j = 0; j < AMOUNTOFX; ++j)
		tempResult += abs(x[j] - xPrev[j]);

	if (abs(tempResult) < eps_fixed_point)
		return REGIME_FIXED_POINT;

	return REGIME_OSCILLATION;
}

// Итог одной точки свипа: из сумм и пиков — значения метрик в res[] (NaN, где
// величину не измерить). Общая для обоих ядер, чтобы классический свип и
// continuation считали метрики одной и той же арифметикой. T — строка
// межпиковых интервалов этой точки или nullptr (медиана не нужна).
//
// Что пишется, когда величину измерить нельзя (NaN, а не заглушка вроде 999:
// NaN отсекается штатным !isfinite и на графике, и в автошкале):
//   UNBOUND (разошлось / отмена / h <= 0) — NaN во всех метриках;
//   нет ни одного межпикового интервала: FIXED_POINT — частота 0 (колебаний
//     нет, это измеренный факт), OSCILLATION — NaN (период длиннее окна);
//   нулевая дисперсия — mobility/complexity NaN (0/0).
// Частоты — в единицах 1/время (для отображения: 1/итерация), mobility — в
// рад/время, как в определении Хьорта. Возвращает итоговый флаг режима.
__device__ int smFinalize(int flag, const MetricsAccum& acc, const PeakStream& peaks,
	const IntervalStats& ist, numb* T, numb dt, numb* res)
{
	const numb NaN = (numb)nan("");
	for (int m = 0; m < SIGM_COUNT; ++m) res[m] = NaN;

	if (!((flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT) && acc.n > 0))
		return REGIME_UNBOUND;

	res[SIGM_MAX]   = acc.mx;
	res[SIGM_MIN]   = acc.mn;
	res[SIGM_RANGE] = acc.mx - acc.mn;
	res[SIGM_MEAN]  = acc.shift + acc.s1 / (numb)acc.n;

	const numb varY  = smVariance(acc.s1,  acc.s2,   acc.n);
	const numb varD  = smVariance(acc.sd,  acc.sd2,  acc.n > 1 ? acc.n - 1 : 0);
	const numb varDD = smVariance(acc.sdd, acc.sdd2, acc.n > 2 ? acc.n - 2 : 0);
	res[SIGM_VARIANCE] = varY;
	if (varY > (numb)0 && varD > (numb)0) {
		const numb mobRaw = sqrt(varD / varY);          // за один сэмпл
		res[SIGM_HJORTH_MOBILITY]   = mobRaw / dt;
		res[SIGM_HJORTH_COMPLEXITY] = sqrt(varDD / varD) / mobRaw;
	}

	// Частоты. peaks.emitAll = Settings -> Peaks выключен: пиков нет вовсе,
	// есть только сырые сэмплы — частоту по ним не измерить.
	if (!peaks.emitAll) {
		const int nI = peaks.count();
		if (nI > 0) {
			res[SIGM_MEAN_FREQ] = ist.sumInv / (numb)nI;
			res[SIGM_INT_MAX]   = ist.mx;
			res[SIGM_INT_MIN]   = ist.mn;
			res[SIGM_INT_RANGE] = ist.mx - ist.mn;
			res[SIGM_INT_MEAN]  = ist.sum / (numb)nI;
			if (T != nullptr) {
				ucuda_heapsort(T, nI);
				const numb med = (nI & 1) ? T[nI / 2]
				                          : (numb)0.5 * (T[nI / 2 - 1] + T[nI / 2]);
				if (med > (numb)0) res[SIGM_MEDIAN_FREQ] = (numb)1 / med;
			}
		} else if (flag == REGIME_FIXED_POINT) {
			res[SIGM_MEAN_FREQ]   = (numb)0;
			res[SIGM_MEDIAN_FREQ] = (numb)0;
		}
	}
	return flag;
}

// Одна точка свипа = один поток. Выход — SoA: outMetrics[m * metricStride + idx],
// строка m заполняется, только если бит m стоит в metricMask.
__global__ void calculateDiscreteModelMetricsCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const size_t	amountOfCalculatedPoints,
	const size_t	amountOfPointsForSkip,
	const int		dimension,
	numb* __restrict__			ranges,
	const numb		h,
	int* __restrict__			indicesOfMutVars,
	numb* __restrict__			initialConditions,
	const int		amountOfInitialConditions,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const size_t	amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const numb		maxValue,
	numb*			intervals,       // nullptr = медиана не нужна
	const size_t	peakStride,
	const int		peakCapacity,
	numb*			outMetrics,
	const size_t	metricStride,
	const int		metricMask,
	int*			flags,
	const int		hSweepAxis,
	const numb		transientTime,
	const numb		tMax,
	const int		logAxisMask,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + ( threadIdx.x * sharedStride );
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	numb res[SIGM_COUNT];
	for (int m = 0; m < SIGM_COUNT; ++m) res[m] = (numb)nan("");

	int    ticks = 0;
	numb   h_local;
	size_t skip_local;
	size_t iters_local;
	int    flag = REGIME_UNBOUND;

	if (ucudaSetupSweepPoint(nPts, amountOfCalculatedPoints, idx, dimension, ranges, h,
			indicesOfMutVars, initialConditions, amountOfInitialConditions,
			values, amountOfValues, hSweepAxis, logAxisMask,
			transientTime, tMax, preScaller,
			amountOfPointsForSkip, amountOfIterations,
			localX, localValues, h_local, skip_local, iters_local))
	{
		flag = loopCalculateDiscreteModel_int(localX, localValues, h_local, skip_local,
			amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, 0, 1,
			cancelFlag, progressCounter, progressStride, &ticks);

		if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT) {
			const numb dt = h_local * (numb)preScaller;
			PeakStream peaks;
			peaks.init(nullptr, intervals, (size_t)idx * peakStride,
				dt, iters_local, peakCapacity, false);
			MetricsAccum acc;
			acc.init();
			IntervalStats ist;
			ist.init();

			flag = loopCalculateDiscreteModelMetrics_int(localX, localValues, h_local, iters_local,
				amountOfInitialConditions, preScaller, writableVar, maxValue, peaks, acc, ist,
				cancelFlag, progressCounter, progressStride, &ticks);

			flag = smFinalize(flag, acc, peaks, ist,
				intervals != nullptr ? intervals + (size_t)idx * peakStride : nullptr, dt, res);
		}
	}

	if (flags != nullptr) flags[idx] = flag;
	for (int m = 0; m < SIGM_COUNT; ++m)
		if ((metricMask >> m) & 1)
			outMetrics[(size_t)m * metricStride + idx] = res[m];

	// Добивка до ожидаемого числа тиков — см. calculateDiscreteModelPeaksCUDA.
	if (progressCounter != nullptr && progressStride > 0) {
		const size_t totalSteps = amountOfPointsForSkip + amountOfIterations;
		const int expected = (int)(totalSteps / (size_t)progressStride);
		if (expected > ticks) atomicAdd(progressCounter, expected - ticks);
	}
}

// Continuation: точки идут цепочкой, каждая стартует с конечного x[] предыдущей
// (гистерезис, forward/backward). Однопоточное ядро — устроено как
// bifurcation1dContinuationKernel (bifurcation1d_cont.template.cu): тот же узел
// цепочки ucuda_node_value_cont, тот же транзиент с preScaller = 1, те же длины
// блока. Разошедшаяся точка не сбрасывает x[] — как и у БД, хвост цепочки после
// неё тоже уходит в unbound.
//
// intervals — ОДНА строка на peakStride (точки последовательны, строку
// переиспользуем) или nullptr. Тик прогресса — точка, отмена — между точками
// и внутри интегрирования.
__global__ void signalMetricsContinuationKernel(
	int nPts,
	numb lo,
	numb hi,
	int reverse,
	int logScale,
	int sweepIsH,
	int mutParamIdx,                 // 1-based индекс в a[]; не читается при sweepIsH
	const numb* baseValues,
	int amountOfValues,
	const numb* baseX,
	int amountOfX,
	numb hBase,
	numb tMax,
	numb transientTime,
	int sizeOfBlock,                 // длина блока под худший (минимальный) h
	int preScaller,
	int writableVar,
	numb maxValue,
	numb* intervals,
	int peakCapacity,
	numb* outMetrics,                // [SIGM_COUNT * nPts]
	int metricMask,
	int* flags,
	const volatile int* cancelFlag,
	int* progressCounter)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	numb x[AMOUNTOFX];
	numb a[64];   // kMaxAmountOfValues в engine
	for (int i = 0; i < amountOfX; ++i)      x[i] = baseX[i];
	for (int i = 0; i < amountOfValues; ++i) a[i] = baseValues[i];

	numb res[SIGM_COUNT];
	for (int j = 0; j < nPts; ++j) {
		if (cancelFlag != nullptr && *cancelFlag != 0) return;
		if (progressCounter != nullptr) atomicAdd(progressCounter, 1);

		const numb p = ucuda_node_value_cont(j, nPts, lo, hi, logScale != 0, reverse != 0);
		numb hLocal = hBase;
		if (sweepIsH) hLocal = p;
		else          a[mutParamIdx] = p;

		int blockLen = (hLocal > 0) ? (int)(tMax / hLocal / (numb)preScaller) : 0;
		if (blockLen > sizeOfBlock) blockLen = sizeOfBlock;
		const numb transient_f = (hLocal > 0) ? (transientTime / hLocal) : (numb)0;
		const int transient_steps = (transient_f >= (numb)2147483647.0) ? 2147483647 : (int)transient_f;

		int flag = REGIME_UNBOUND;
		for (int m = 0; m < SIGM_COUNT; ++m) res[m] = (numb)nan("");

		if (hLocal > 0 && blockLen > 0) {
			flag = loopCalculateDiscreteModel_int(x, a, hLocal, (size_t)transient_steps,
				amountOfX, 1, 0, maxValue, nullptr, 0, 1, cancelFlag);
			if (flag != REGIME_UNBOUND) {
				const numb dt = hLocal * (numb)preScaller;
				PeakStream peaks;
				peaks.init(nullptr, intervals, 0, dt, (size_t)blockLen, peakCapacity, false);
				MetricsAccum acc;
				acc.init();
				IntervalStats ist;
				ist.init();
				flag = loopCalculateDiscreteModelMetrics_int(x, a, hLocal, (size_t)blockLen,
					amountOfX, preScaller, writableVar, maxValue, peaks, acc, ist,
					cancelFlag, nullptr, 0, nullptr);
				flag = smFinalize(flag, acc, peaks, ist, intervals, dt, res);
			}
		}

		if (flags != nullptr) flags[j] = flag;
		for (int m = 0; m < SIGM_COUNT; ++m)
			if ((metricMask >> m) & 1)
				outMetrics[(size_t)m * (size_t)nPts + j] = res[m];
	}
}
