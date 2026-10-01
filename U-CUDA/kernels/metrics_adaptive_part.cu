// metrics_adaptive_part.cu — метрики сигнала с адаптивным шагом (Parametric -> Metrics).
//
// Не самостоятельный шаблон: движок склеивает
//   #define SIGM_MINMAX_INTERP 1 + #define UCUDA_AD_NO_SWEEP_KERNELS 1
//   + signal_metrics.template.cu + adaptive_part.cu + этот файл,
// так что здесь доступны и MetricsAccum / IntervalStats / smFinalize шаблона метрик,
// и драйвер с помощниками адаптивного шага из adaptive_part.cu.
//
// Только равномерная сетка: отсчёты в T0 + i*dt плотным выходом, дальше всё
// как у постоянного шага. Параметры Хьорта при адаптивном шаге не считаются
// (NaN): вторые разности отсчётов плотного выхода несут изломы интерполянта в
// узлах шага. min/max — по интерполированным экстремумам (SIGM_MINMAX_INTERP).

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
	PeakStream peaks;
	peaks.init(nullptr, intervals, (size_t)idx * peakStride, dt, iters, peakCapacity, false);
	MetricsAccum acc;
	acc.init(((metricMask >> SIGM_VOLUME) & 1) != 0);
	IntervalStats ist;
	ist.init();
	int flag = ucudaAdMetricsPoint(S, K, localValues, P, transientTime, dt, iters, writableVar, maxValue,
		peaks, acc, ist, cancelFlag, prog);
	// UNBOUND smFinalize сам сводит к NaN во всех метриках.
	flag = smFinalize(flag, acc, peaks, ist,
		intervals != nullptr ? intervals + (size_t)idx * peakStride : nullptr, dt, res);
	res[SIGM_HJORTH_MOBILITY]   = (numb)nan("");
	res[SIGM_HJORTH_COMPLEXITY] = (numb)nan("");

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
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
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
		PeakStream peaks;
		peaks.init(nullptr, intervals, 0, dt, iters, peakCapacity, false);
		MetricsAccum acc;
		acc.init(((metricMask >> SIGM_VOLUME) & 1) != 0);
		IntervalStats ist;
		ist.init();
		int flag = ucudaAdOut(S.X, maxValue) ? REGIME_UNBOUND
		         : ucudaAdMetricsPoint(S, K, a, P, transientTime, dt, iters, writableVar, maxValue,
		                               peaks, acc, ist, cancelFlag, prog);
		flag = smFinalize(flag, acc, peaks, ist, intervals, dt, res);
		res[SIGM_HJORTH_MOBILITY]   = (numb)nan("");
		res[SIGM_HJORTH_COMPLEXITY] = (numb)nan("");

		if (flags != nullptr) flags[j] = flag;
		for (int m = 0; m < SIGM_COUNT; ++m)
			if ((metricMask >> m) & 1)
				outMetrics[(size_t)m * (size_t)nPts + j] = res[m];
		ucudaAdWriteStats(adStats, j, S);
	}
}
