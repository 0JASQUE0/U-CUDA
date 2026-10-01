// metrics_adaptive_part.cu — метрики сигнала с адаптивным шагом (Parametric -> Metrics).
//
// Не самостоятельный шаблон: движок склеивает
//   #define SIGM_MINMAX_INTERP 1 + #define UCUDA_AD_NO_SWEEP_KERNELS 1
//   + signal_metrics.template.cu + adaptive_part.cu + этот файл,
// так что здесь доступны и MetricsAccum / IntervalStats / smFinalize шаблона метрик,
// и драйвер с помощниками адаптивного шага из adaptive_part.cu.
//
// Два режима вывода (AdaptiveRequest::raw_nodes, ucudaAdMetricsRun):
//   равномерная сетка — отсчёты в T0 + i*dt плотным выходом, дальше всё как у постоянного
//     шага; min/max — по интерполированным экстремумам (SIGM_MINMAX_INTERP);
//   узлы шага — каждый dec-й принятый узел (MetricsAccumNU): средние по времени, экстремумы
//     и пики — интерполяцией между узлами. Нет плотного выхода на каждом отсчёте и нет
//     расхождения варпа на сетке — в свипах в разы быстрее.
// Параметры Хьорта при адаптивном шаге не считаются (NaN): вторые разности отсчётов плотного
// выхода несут изломы интерполянта в узлах шага, а на узлах шаг неравномерен.

// Отсчёт сетки -> метрики: тот же сэмпл, те же суммы и интервалы, что у
// loopCalculateDiscreteModelMetrics_int.
struct UcudaAdPushMetrics {
	PeakStream&    peaks;
	MetricsAccum&  acc;
	IntervalStats& ist;
	int            writableVar;
	__device__ void operator()(const numb* y)
	{
		const numb sample = ucudaAdObs(y, writableVar);
		if (acc.box) acc.pushBox(y);
		acc.push(sample);
		const int  emitted0 = peaks.emitted;
		const numb anchor0  = peaks.anchorTime;
		peaks.push(sample);
		if (peaks.emitted != emitted0 && emitted0 < max_amount_of_peaks) {
			const numb delta = (peaks.anchorTime - anchor0) * peaks.sampleStep;
			ist.push(delta, emitted0);
		}
	}
};

// Транзиент и запись метрик одной точки (ucudaAdUniformPoint); UNBOUND smFinalize
// сам сводит к NaN.
__device__ __forceinline__ int ucudaAdMetricsPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb transientTime, const numb dt, const size_t iters,
	const int writableVar, const numb maxValue, PeakStream& peaks, MetricsAccum& acc,
	IntervalStats& ist, const volatile int* cancelFlag, UcudaAdProgress& prog)
{
	UcudaAdPushMetrics push{ peaks, acc, ist, writableVar };
	return ucudaAdUniformPoint(S, K, a, P, transientTime, dt, iters, maxValue, push, cancelFlag, prog);
}

// Метрики на узлах шага. Узлы неравномерны, поэтому среднее и дисперсия — средние по ВРЕМЕНИ
// на [первый узел, последний]: интегралы y и y^2 эрмитовой квадратурой — трапеция с поправкой
// h^2 (g'(t_a) - g'(t_b)) / 12 по точным производным в узлах (y' = f, (y^2)' = 2 y f): точна для
// кубики, 4-й порядок по h. Простое среднее по узлам смещено туда, где шаг мелкий. Суммы — от
// сдвига на первый узел, как у MetricsAccum. max/min — по узлам и по вершинам между ними (смена
// знака y'), вершина — ucudaAdNodeVertex тем же способом interp, что пики на узлах. Volume —
// коробка по узлам и по вершинам каждой переменной (кубический Эрмит; при interp == 0 — узлы).
struct MetricsAccumNU
{
	size_t n;
	int    interp;
	numb   shift, i1, i2, span;
	numb   t0, y0;          // узел n-2 (парабола)
	numb   t1, y1, d1;      // узел n-1
	numb   mn, mx;
	bool   box;
	numb   bMin[AMOUNTOFX], bMax[AMOUNTOFX];
	numb   xp[AMOUNTOFX], fp[AMOUNTOFX];   // узел n-1 по всем переменным (Volume)

	__device__ void init(bool trackBox, int interpMode)
	{
		n = 0; interp = interpMode;
		shift = i1 = i2 = span = (numb)0;
		t0 = y0 = t1 = y1 = d1 = (numb)0;
		mn = mx = (numb)0;
		box = trackBox;
		for (int j = 0; j < AMOUNTOFX; ++j) bMin[j] = bMax[j] = xp[j] = fp[j] = (numb)0;
	}

	// y, d — наблюдаемая и её производная в узле t; X, F — состояние и f(X) (Volume).
	__device__ void push(numb t, numb y, numb d, const numb* X, const numb* F)
	{
		if (n == 0) {
			shift = y; mn = y; mx = y;
			if (box) for (int j = 0; j < AMOUNTOFX; ++j) { bMin[j] = X[j]; bMax[j] = X[j]; }
		} else {
			const numb h  = t - t1;
			const numb c0 = y1 - shift, c1 = y - shift;
			i1 += (numb)0.5 * h * (c0 + c1) + h * h * (d1 - d) / (numb)12;
			i2 += (numb)0.5 * h * (c0 * c0 + c1 * c1) + h * h * (c0 * d1 - c1 * d) / (numb)6;
			span += h;
			if (y > mx) mx = y;
			if (y < mn) mn = y;
			numb pv, pt;
			if (d1 > 0 && d <= 0) {
				ucudaAdNodeVertex(interp, (int)n, t0, y0, t1, y1, d1, t, y, d, pv, pt);
				if (pv > mx) mx = pv;
			} else if (d1 < 0 && d >= 0) {
				ucudaAdNodeVertex(interp, (int)n, t0, -y0, t1, -y1, -d1, t, -y, -d, pv, pt);
				if (-pv < mn) mn = -pv;
			}
			if (box) {
				for (int j = 0; j < AMOUNTOFX; ++j) {
					const numb x = X[j], f = F[j];
					if (x < bMin[j]) bMin[j] = x;
					if (x > bMax[j]) bMax[j] = x;
					if (interp == 0) continue;
					if (fp[j] > 0 && f <= 0) {
						ucudaAdNodeVertex(2, 1, t1, xp[j], t1, xp[j], fp[j], t, x, f, pv, pt);
						if (pv > bMax[j]) bMax[j] = pv;
					} else if (fp[j] < 0 && f >= 0) {
						ucudaAdNodeVertex(2, 1, t1, -xp[j], t1, -xp[j], -fp[j], t, -x, -f, pv, pt);
						if (-pv < bMin[j]) bMin[j] = -pv;
					}
				}
			}
		}
		if (box) for (int j = 0; j < AMOUNTOFX; ++j) { xp[j] = X[j]; fp[j] = F[j]; }
		t0 = t1; y0 = y1;
		t1 = t;  y1 = y;  d1 = d;
		++n;
	}
};

// Узел -> метрики: суммы, пики и межпиковые интервалы (как UcudaAdPushMetrics на сетке).
struct UcudaAdPushMetricsNU {
	PeakStreamNU&   peaks;
	MetricsAccumNU& acc;
	IntervalStats&  ist;
	int             writableVar;
	__device__ void operator()(const numb t, const numb* X, const numb* F)
	{
		const numb y = ucudaAdObs(X, writableVar), d = ucudaAdObs(F, writableVar);
		acc.push(t, y, d, X, F);
		const int  emitted0 = peaks.emitted;
		const numb anchor0  = peaks.anchorTime;
		peaks.pushNode(t, y, d);
		if (peaks.emitted != emitted0 && emitted0 < max_amount_of_peaks)
			ist.push(peaks.anchorTime - anchor0, emitted0);   // = delta в PeakStreamNU::emitInterval
	}
};

// Итог точки на узлах — как smFinalize, но среднее и дисперсия по времени, экстремумы уже с
// вершинами, пики — PeakStreamNU (всегда ищутся).
__device__ int smFinalizeNU(int flag, const MetricsAccumNU& acc, const PeakStreamNU& peaks,
	const IntervalStats& ist, numb* T, numb* res)
{
	const numb NaN = (numb)nan("");
	for (int m = 0; m < SIGM_COUNT; ++m) res[m] = NaN;

	if (!((flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT) && acc.n > 0))
		return REGIME_UNBOUND;

	res[SIGM_MAX]   = acc.mx;
	res[SIGM_MIN]   = acc.mn;
	res[SIGM_RANGE] = acc.mx - acc.mn;
	if (acc.span > (numb)0) {
		const numb m1 = acc.i1 / acc.span;
		const numb v  = acc.i2 / acc.span - m1 * m1;
		res[SIGM_MEAN]     = acc.shift + m1;
		res[SIGM_VARIANCE] = v < (numb)0 ? (numb)0 : v;
	} else {
		res[SIGM_MEAN]     = acc.shift;
		res[SIGM_VARIANCE] = (numb)0;
	}
	if (acc.box) {
		numb vol = (numb)1;
		for (int j = 0; j < AMOUNTOFX; ++j) vol *= acc.bMax[j] - acc.bMin[j];
		res[SIGM_VOLUME] = vol;
	}

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
	return flag;
}

// Одна точка метрик целиком: транзиент, запись и итог в res[] (Хьорт — NaN). raw == 0 —
// равномерная сетка (ucudaAdMetricsPoint, smFinalize: арифметика прежняя), иначе — узлы шага
// (каждый dec-й, ucudaAdNodesPointT, MetricsAccumNU, smFinalizeNU) до transientTime + tRec.
// T — строка межпиковых интервалов точки или nullptr (медиана не нужна); skip — точка уже
// разошлась (continuation: X вне maxValue), счёт не нужен. Общая для ядер и CPU-DLL.
__device__ __forceinline__ int ucudaAdMetricsRun(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb transientTime, const numb tRec, const numb dt, const size_t iters,
	const int raw, const int dec, const int peakInterp, const int writableVar, const numb maxValue,
	numb* T, const int peakCapacity, const int metricMask, const bool skip,
	const volatile int* cancelFlag, UcudaAdProgress& prog, numb* res)
{
	const bool box = ((metricMask >> SIGM_VOLUME) & 1) != 0;
	IntervalStats ist;
	ist.init();
	int flag;
	if (!raw) {
		PeakStream peaks;
		peaks.init(nullptr, T, 0, dt, iters, peakCapacity, false);
		MetricsAccum acc;
		acc.init(box);
		flag = skip ? REGIME_UNBOUND
		     : ucudaAdMetricsPoint(S, K, a, P, transientTime, dt, iters, writableVar, maxValue,
		                           peaks, acc, ist, cancelFlag, prog);
		flag = smFinalize(flag, acc, peaks, ist, T, dt, res);
	} else {
		PeakStreamNU peaks;
		peaks.init(nullptr, T, 0, peakCapacity, peakInterp);
		MetricsAccumNU acc;
		acc.init(box, peakInterp);
		UcudaAdPushMetricsNU push{ peaks, acc, ist, writableVar };
		flag = skip ? REGIME_UNBOUND
		     : ucudaAdNodesPointT(S, K, a, P, transientTime, transientTime + tRec, dec, maxValue,
		                          push, cancelFlag, prog);
		flag = smFinalizeNU(flag, acc, peaks, ist, T, res);
	}
	res[SIGM_HJORTH_MOBILITY]   = (numb)nan("");
	res[SIGM_HJORTH_COMPLEXITY] = (numb)nan("");
	return flag;
}

// Ядра — не для CPU-DLL (krs_cpu.cpp, AdaptiveCpuModule): она берёт отсюда только
// точку (ucudaAdMetricsPoint), а свои входы — построчные копии ядер ниже — пишет сама.
#ifndef UCUDA_AD_NO_METRICS_KERNELS

// То же, что calculateDiscreteModelMetricsCUDA (раскладка выхода SoA одна и та же),
// шагом управляет регулятор. adStats — см. calculateDiscreteModelPeaksAdCUDA.
__global__ void calculateDiscreteModelMetricsAdCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const size_t	amountOfCalculatedPoints,
	const int		dimension,
	const numb* __restrict__	ranges,
	const int* __restrict__		indicesOfMutVars,
	const numb* __restrict__	initialConditions,
	const int		amountOfInitialConditions,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const int		writableVar,
	const numb		maxValue,
	numb*			intervals,       // nullptr = медиана не нужна
	const size_t	peakStride,
	const int		peakCapacity,
	numb*			outMetrics,
	const size_t	metricStride,
	const int		metricMask,
	int*			flags,
	const int		logAxisMask,
	const UcudaAdaptParams* __restrict__ Pbase,
	const int* __restrict__		axisKind,
	const numb		tolRatio,
	const numb		transientTime,
	const numb		tRec,
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
	const int		raw,             // узлы шага (иначе сетка); preScaller — их прореживание
	const int		peakInterp,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride,
	const numb		progressDt,
	const size_t	progressUnits,
	numb*			adStats)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + (threadIdx.x * sharedStride);
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	numb res[SIGM_COUNT];
	for (int m = 0; m < SIGM_COUNT; ++m) res[m] = (numb)nan("");

	UcudaAdaptParams P = *Pbase;
	ucudaSetupSweepPointAd(nPts, amountOfCalculatedPoints, idx, dimension, ranges, indicesOfMutVars,
		initialConditions, values, amountOfValues, logAxisMask, axisKind, tolRatio,
		localX, localValues, P);

	const UcudaKrsFns K{};
	UcudaAdaptState S;
	ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);
	UcudaAdProgress prog;
	prog.init(progressCounter, progressStride, progressDt);

	const numb dt = dtOut * (numb)preScaller;
	// UNBOUND итог сам сводит к NaN во всех метриках.
	const int flag = ucudaAdMetricsRun(S, K, localValues, P, transientTime, tRec, dt, iters, raw, preScaller,
		peakInterp, writableVar, maxValue, intervals != nullptr ? intervals + (size_t)idx * peakStride : nullptr,
		peakCapacity, metricMask, false, cancelFlag, prog, res);

	if (flags != nullptr) flags[idx] = flag;
	for (int m = 0; m < SIGM_COUNT; ++m)
		if ((metricMask >> m) & 1)
			outMetrics[(size_t)m * metricStride + idx] = res[m];
	ucudaAdWriteStats(adStats, idx, S);
	prog.top_up((int)(progressUnits / (size_t)(progressStride > 0 ? progressStride : 1)));
}

// Continuation метрик: цепочка точек, как у signalMetricsContinuationKernel, но
// вместе с X переносятся предложенный шаг и память регулятора (ucuda_ad_restart,
// см. calculateDiscreteModelPeaksAdContCUDA). intervals — одна строка на
// peakStride или nullptr: точки последовательны, строка переиспользуется. Выход —
// [SIGM_COUNT * nPts], adStats — по 4 числа на точку. Тик прогресса — точка.
__global__ void signalMetricsContinuationAdKernel(
	const int		nPts,
	const numb		lo,
	const numb		hi,
	const int		reverse,
	const int		logScale,
	const int		mutParamIdx,
	const numb* __restrict__	baseValues,
	const int		amountOfValues,
	const numb* __restrict__	baseX,
	const int		writableVar,
	const numb		maxValue,
	numb*			intervals,
	const int		peakCapacity,
	numb*			outMetrics,
	const int		metricMask,
	int*			flags,
	const UcudaAdaptParams* __restrict__ Pbase,
	const int* __restrict__		axisKind,
	const numb		tolRatio,
	const numb		transientTime,
	const numb		tRec,
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
	const int		raw,
	const int		peakInterp,
	const volatile int* cancelFlag,
	int*			progressCounter,
	numb*			adStats)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	numb x[AMOUNTOFX];
	numb a[64];   // kMaxAmountOfValues в движке
	for (int i = 0; i < AMOUNTOFX; ++i) x[i] = baseX[i];
	for (int i = 0; i < amountOfValues && i < 64; ++i) a[i] = baseValues[i];

	UcudaAdaptParams P = *Pbase;
	const int kind = axisKind[0];
	const UcudaKrsFns K{};
	UcudaAdaptState S;
	UcudaAdProgress prog;
	prog.init(nullptr, 0, (numb)0);

	numb res[SIGM_COUNT];
	for (int j = 0; j < nPts; ++j) {
		if (cancelFlag != nullptr && *cancelFlag != 0) return;
		if (progressCounter != nullptr) atomicAdd(progressCounter, 1);
		const numb v = ucuda_node_value_cont(j, nPts, lo, hi, logScale != 0, reverse != 0);
		if (kind == UCUDA_AXIS_SYSTEM) a[mutParamIdx] = v;
		else ucudaAdApplyStepAxis(kind, v, tolRatio, P);

		if (j == 0) ucuda_ad_init(S, K, AMOUNTOFX, x, (numb)0, a, P);
		else        ucuda_ad_restart(S, K, (numb)0, a, P);

		const numb dt = dtOut * (numb)preScaller;
		const int flag = ucudaAdMetricsRun(S, K, a, P, transientTime, tRec, dt, iters, raw, preScaller, peakInterp,
			writableVar, maxValue, intervals, peakCapacity, metricMask, ucudaAdOut(S.X, maxValue), cancelFlag,
			prog, res);

		if (flags != nullptr) flags[j] = flag;
		for (int m = 0; m < SIGM_COUNT; ++m)
			if ((metricMask >> m) & 1)
				outMetrics[(size_t)m * (size_t)nPts + j] = res[m];
		ucudaAdWriteStats(adStats, j, S);
	}
}

#endif // UCUDA_AD_NO_METRICS_KERNELS
