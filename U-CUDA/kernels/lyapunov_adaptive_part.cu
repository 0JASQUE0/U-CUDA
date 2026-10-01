// lyapunov_adaptive_part.cu — старший показатель (LLE) и спектр Ляпунова (LS) с
// адаптивным шагом: GPU-ядра.
//
// Не самостоятельный шаблон: движок приклеивает его после adaptive_part.cu к
// bifurcation2d.template.cu с UCUDA_AD_NO_SWEEP_KERNELS (ядра пиков не нужны),
// UCUDA_AD_NO_DENSE (плотный выход не нужен: перенормировка только в узлах шага) и
// UCUDA_AD_LYAPUNOV. Сам алгоритм — клоны, перенормировка, точка, цепочка continuation —
// в разделе UCUDA_AD_LYAPUNOV файла ucuda_adaptive.cuh: его же собирает CPU-ветка
// (krs_cpu.cpp), поэтому GPU и CPU считают одним текстом.
//
// Два набора ядер, по макросу модуля:
//   без UCUDA_LYAP_AD_CONT_ONLY — классический свип, нить на точку (lyapunovLLEAdCUDA /
//     lyapunovLSAdCUDA), раскладка результата — как у LLEKernelCUDA / LSKernelCUDA
//     (NC чисел на точку, 999 — разлёт);
//   с UCUDA_LYAP_AD_CONT_ONLY — continuation: одна нить идёт по цепочке точек
//     (lyapunovLLEAdContCUDA / lyapunovLSAdContCUDA), NaN — разлёт, как у
//     lle1dContinuationKernel / ls1dContinuationKernel.
//
// Время сборки: транзиент, блоки и перенормировка — в одном цикле, чтобы в ядре была
// одна копия шага (см. шапку adaptive_part.cu).

#ifndef UCUDA_LYAP_AD_CONT_ONLY

// Общее тело ядер: точка свипа (система или настройка шага по осям), расчёт, запись
// NC чисел в resultArray[idx*NC ..] (999 — разлёт) и статистики шага.
template <int NC>
__device__ __forceinline__ void ucudaLyapAdKernel(
	const int nPts, const int nPtsLimiter, const size_t amountOfCalculatedPoints, const int dimension,
	const numb* ranges, const int* indicesOfMutVars, const numb* initialConditions,
	const int amountOfInitialConditions, const numb* values, const int amountOfValues,
	const numb maxValue, numb* resultArray, const int logAxisMask,
	const UcudaAdaptParams* Pbase, const int* axisKind, const numb tolRatio,
	const numb transientTime, const numb tMax, const numb NT, const int nBlocks, const int nWarm, const numb eps,
	const int renorm, const volatile int* cancelFlag, int* progressCounter, const int progressStride,
	const numb progressDt, const size_t progressUnits, numb* adStats)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + (threadIdx.x * sharedStride);
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	UcudaAdaptParams P = *Pbase;
	ucudaSetupSweepPointAd(nPts, amountOfCalculatedPoints, idx, dimension, ranges, indicesOfMutVars,
		initialConditions, values, amountOfValues, logAxisMask, axisKind, tolRatio,
		localX, localValues, P);

	const UcudaKrsFns K{};
	UcudaAdaptState S;
	ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);
	UcudaAdProgress prog;
	prog.init(progressCounter, progressStride, progressDt);

	(void)tMax;
	UcudaLyapClones<NC> cl;
	cl.active = false;
	numb res[NC];
	const int ok = ucuda_lyap_point<NC>(S, K, localValues, P, cl, false, transientTime, NT, nBlocks, nWarm, eps,
		renorm, maxValue, idx, res, cancelFlag, prog);
	for (int m = 0; m < NC; ++m) resultArray[(size_t)idx * NC + m] = ok ? res[m] : (numb)999;
	ucudaAdWriteStats(adStats, idx, S);
	prog.top_up((int)(progressUnits / (size_t)(progressStride > 0 ? progressStride : 1)));
}

#define UCUDA_LYAP_AD_PARAMS \
	const int nPts, const int nPtsLimiter, const size_t amountOfCalculatedPoints, const int dimension, \
	const numb* __restrict__ ranges, const int* __restrict__ indicesOfMutVars, \
	const numb* __restrict__ initialConditions, const int amountOfInitialConditions, \
	const numb* __restrict__ values, const int amountOfValues, const numb maxValue, numb* resultArray, \
	const int logAxisMask, const UcudaAdaptParams* __restrict__ Pbase, const int* __restrict__ axisKind, \
	const numb tolRatio, const numb transientTime, const numb tMax, const numb NT, const int nBlocks, const int nWarm, \
	const numb eps, const int renorm, const volatile int* cancelFlag, int* progressCounter, \
	const int progressStride, const numb progressDt, const size_t progressUnits, numb* adStats
#define UCUDA_LYAP_AD_ARGS \
	nPts, nPtsLimiter, amountOfCalculatedPoints, dimension, ranges, indicesOfMutVars, initialConditions, \
	amountOfInitialConditions, values, amountOfValues, maxValue, resultArray, logAxisMask, Pbase, axisKind, \
	tolRatio, transientTime, tMax, NT, nBlocks, nWarm, eps, renorm, cancelFlag, progressCounter, progressStride, \
	progressDt, progressUnits, adStats

// LLE: один клон, одно число на точку (как LLEKernelCUDA).
__global__ void lyapunovLLEAdCUDA(UCUDA_LYAP_AD_PARAMS)
{
	ucudaLyapAdKernel<1>(UCUDA_LYAP_AD_ARGS);
}

// LS: AMOUNTOFX клонов, спектр из AMOUNTOFX чисел на точку (как LSKernelCUDA).
__global__ void lyapunovLSAdCUDA(UCUDA_LYAP_AD_PARAMS)
{
	ucudaLyapAdKernel<AMOUNTOFX>(UCUDA_LYAP_AD_ARGS);
}

#else // UCUDA_LYAP_AD_CONT_ONLY

// Continuation: одна нить, точки цепочки подряд (ucuda_lyap_chain). Ось axisKind —
// параметр a[mutParamIdx] (0) или настройка шага. Тик прогресса — точка.
#define UCUDA_LYAP_AD_CONT_PARAMS \
	const int nPts, const numb lo, const numb hi, const int reverse, const int logScale, \
	const int mutParamIdx, const numb* __restrict__ baseValues, const int amountOfValues, \
	const numb* __restrict__ baseX, const UcudaAdaptParams* __restrict__ Pbase, const int axisKind, \
	const numb tolRatio, const numb transientTime, const numb NT, const int nBlocks, const int nWarm, \
	const numb eps, const int renorm, const numb maxValue, numb* result, numb* adStats, \
	const volatile int* cancelFlag, int* progressCounter
#define UCUDA_LYAP_AD_CONT_ARGS \
	K, 1, nPts, lo, hi, reverse, logScale, mutParamIdx, baseValues, amountOfValues, baseX, *Pbase, axisKind, \
	tolRatio, transientTime, NT, nBlocks, nWarm, eps, renorm, maxValue, result, adStats, cancelFlag, progressCounter

__global__ void lyapunovLLEAdContCUDA(UCUDA_LYAP_AD_CONT_PARAMS)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	const UcudaKrsFns K{};
	ucuda_lyap_chain<1>(UCUDA_LYAP_AD_CONT_ARGS);
}

__global__ void lyapunovLSAdContCUDA(UCUDA_LYAP_AD_CONT_PARAMS)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	const UcudaKrsFns K{};
	ucuda_lyap_chain<AMOUNTOFX>(UCUDA_LYAP_AD_CONT_ARGS);
}

#endif // UCUDA_LYAP_AD_CONT_ONLY
