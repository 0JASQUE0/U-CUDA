__device__  bool loopCalculateDiscreteModel(numb* x, const numb* values, 
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, numb* data,
	const int startDataIndex, const int writeStep UCUDA_NP)
{

	numb xPrev[AMOUNTOFX];
	UcudaKrsPre krsPre;
	ucudaKrsPrep(krsPre, values, h);
	UCUDA_NBEGIN
	// Глобальный цикл, который производит вычисления заданные amountOfIterations раз
	// Счётчик size_t под стать amountOfIterations: при int-счётчике широкий
	// параметр не давал бы ничего, цикл переполнялся бы на 2^31-м шаге.
	for ( size_t i = 0; i < amountOfIterations; ++i )
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// Если все-таки передали массив для записи - записываем значение переменной
		if ( data != nullptr )
			data[startDataIndex + i * writeStep] = x[writableVar];

		// Моделируем систему preScaller раз ( то есть если preScaller > 1, то мы пропустим ( preScaller - 1 ) в смоделированной траектории )
		for ( int j = 0; j < preScaller; ++j )
			UCUDA_STEP(x, values, h, krsPre);

		// Если isnan или isinf - возвращаем false, ибо это нежелательное поведение системы
		if ( isnan( x[writableVar] ) || isinf( x[writableVar] ) )
		{
			UCUDA_NEND
			return false;
		}

		// Если maxValue == 0, это значит пользователь не выставил ограничение, иначе требуется его проверить
		if ( maxValue != 0 )
			if ( fabs( x[writableVar] ) > maxValue )
			{
				UCUDA_NEND
				return false;
			}
	}

	UCUDA_NEND
	return true;
}

// x[w] без индекса во время выполнения: маска q == w считается до цикла, в цикле —
// только выборы. С x[w] весь массив состояния уходил бы в локальную память.
__device__ __host__ __forceinline__ void ucudaPickMask(bool* sel, const int w)
{
	for (int q = 0; q < AMOUNTOFX; ++q) sel[q] = (q == w);
}
__device__ __host__ __forceinline__ numb ucudaPickVar(const numb* x, const bool* sel)
{
	numb s = x[0];
	for (int q = 1; q < AMOUNTOFX; ++q) s = sel[q] ? x[q] : s;
	return s;
}

// Шаг проверки (прогресс, Stop, расходимость) в циклах без записи траектории: транзиент,
// блоки LLE/LS и цикл с пиками. Проверка — после шагов 0, 1000, 2000, ..., а шаги между
// ними идут внутренним циклом с постоянным числом итераций. С проверкой через счётчик на
// каждом шаге цикл был «расходящимся» (своё число шагов у нити), и компилятор ставил
// на каждую итерацию барьеры и голосование варпа: во float это стоило вдвое больше шага
// Эйлера (транзиент x3.1 после правки; в double шаг дороже, и выигрыша нет). Циклы с
// записью траектории и метрики проверяют по-прежнему раз в CHECK_INTERVAL (100);
// CPU-двойники — с тем же шагом (cpu_loop_model). Шаг прогресса в движке кратен обоим.
constexpr int UCUDA_LOOP_CHECK_BLOCK = 1000;

// Проверка раз в UCUDA_LOOP_CHECK_BLOCK шагов: прогресс, Stop, расходимость. Возвращает
// REGIME_CANCELLED / REGIME_UNBOUND, если цикл надо прервать, иначе UCUDA_LOOP_GO.
constexpr int UCUDA_LOOP_GO = 2;
__device__ __host__ __forceinline__ int ucudaLoopCheck(const numb* x, const numb maxValue,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported,
	size_t& sinceReport)
{
	if (progressCounter != nullptr && progressStride > 0) {
		sinceReport += UCUDA_LOOP_CHECK_BLOCK;
		if (sinceReport >= (size_t)progressStride) {
#ifdef __CUDA_ARCH__
			atomicAdd(progressCounter, 1);
#endif
			if (ticksReported != nullptr) ++(*ticksReported);
			sinceReport = 0;
		}
	}
	if (ucudaCancelRequested(cancelFlag)) return REGIME_CANCELLED;
	numb checker = 0;
	for (int j = 0; j < AMOUNTOFX; ++j)
		checker = checker + abs(x[j]);
	if (isnan(checker) || isinf(checker))
		return REGIME_UNBOUND;
	if (maxValue != 0)
		if (abs(checker) > maxValue)
			return REGIME_UNBOUND;
	return UCUDA_LOOP_GO;
}

// Прогресс и Stop за короткие куски работы — NT-блоки LLE/LS. Блоки зовутся без
// счётчика и без флага отмены, а шаги копятся здесь:
//  - цикл шагов засчитывает себе UCUDA_LOOP_CHECK_BLOCK шагов уже на первой проверке, и
//    при блоке в 100 шагов тиков выходило в разы больше ожидаемого — бар упирался в 100 %
//    задолго до конца;
//  - флаг отмены лежит в памяти хоста, и его чтение на каждом блоке (x и каждый клон)
//    шло через PCIe по одному: LLE 2D 256², float, NT = 0.1 — 105 мс против 13 без него.
// Отмена проверяется раз в UCUDA_LOOP_CHECK_BLOCK шагов ведущей траектории, как в
// остальных циклах. Возвращает true, если пришёл Stop.
__device__ __forceinline__ bool ucudaBlockSignals(const volatile int* cancelFlag,
	int* progressCounter, int progressStride, size_t steps,
	size_t& sinceTick, size_t& sinceCancel, int& ticks)
{
	if (progressCounter != nullptr && progressStride > 0) {
		sinceTick += steps;
		if (sinceTick >= (size_t)progressStride) {
			const int n = (int)(sinceTick / (size_t)progressStride);
			sinceTick -= (size_t)n * (size_t)progressStride;
			atomicAdd(progressCounter, n);
			ticks += n;
		}
	}
	sinceCancel += steps;
	if (sinceCancel < (size_t)UCUDA_LOOP_CHECK_BLOCK) return false;
	sinceCancel = 0;
	return ucudaCancelRequested(cancelFlag);
}

// Тело loopCalculateDiscreteModel_int (обёртка ниже). x — копия состояния в регистрах:
// с x в shared каждый шаг читал и писал его заново (3 LDS + 3 STS), и дешёвые схемы во
// float упирались в это, а не в арифметику. Отсюда и выбор x[writableVar] без индекса
// во время выполнения — с ним компилятор кладёт весь массив в локальную память.
__device__ __host__ __forceinline__ int loopCalculateDiscreteModel_int_core(
	numb* __restrict__ x, const numb* __restrict__ values,
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, numb* data,
	const size_t startDataIndex, const int writeStep,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported UCUDA_NREF)
{
	//numb* xPrev = new numb[amountOfX];
	numb xPrev[AMOUNTOFX];
	numb checker;
	UcudaKrsPre krsPre;
	ucudaKrsPrep(krsPre, values, h);
	bool pick[AMOUNTOFX];
	ucudaPickMask(pick, writableVar);
	size_t sinceReport = 0;
	// Обратный отсчёт до проверки вместо i % CHECK_INTERVAL: деление size_t на
	// каждом шаге стоило ~12% времени ядра. Проверки — те же i = 0, 100, 200, ...
	int untilCheck = 0;

	// Без записи траектории — блоками (см. UCUDA_LOOP_CHECK_BLOCK): шаг i, кратный блоку,
	// проверка, затем блок - 1 шагов внутренним циклом без проверок.
	if (data == nullptr) {
		size_t i = 0;
		while (i < amountOfIterations) {
			UCUDA_STEP(x, values, h, krsPre);
			++i;
			const int r = ucudaLoopCheck(x, maxValue, cancelFlag, progressCounter, progressStride, ticksReported, sinceReport);
			if (r != UCUDA_LOOP_GO) return r;
			if (amountOfIterations - i >= (size_t)(UCUDA_LOOP_CHECK_BLOCK - 1)) {
				#pragma unroll 4
				for (int k = 0; k < UCUDA_LOOP_CHECK_BLOCK - 1; ++k)
					UCUDA_STEP(x, values, h, krsPre);
				i += UCUDA_LOOP_CHECK_BLOCK - 1;
			} else {
				for (; i < amountOfIterations; ++i)
					UCUDA_STEP(x, values, h, krsPre);
			}
		}
	}
	else
	// Глобальный цикл, который производит вычисления заданные amountOfIterations раз
	for (size_t i = 0; i < amountOfIterations; ++i)
	{
		//#pragma unroll
		//for (int j = 0; j < AMOUNTOFX; ++j)
		//{
		//	xPrev[j] = x[j];
		//}
		// Если все-таки передали массив для записи - записываем значение переменной

		if (data != nullptr) {
			// writableVar < 0 -> комбинация первых до 3 переменных (port MATLAB).
			// AMOUNTOFX в NVRTC раскрывается через #define {{AMOUNT_OF_X}},
			// в обычной nvcc-сборке — constexpr int из configCUDA.h. В обоих
			// случаях `if constexpr` выкидывает мёртвые ветки на этапе компиляции.
			if (writableVar < 0) {
				if constexpr (AMOUNTOFX >= 3)
					data[startDataIndex + i] = x[0] + pi*x[1] + euler*x[2];
				else if constexpr (AMOUNTOFX == 2)
					data[startDataIndex + i] = x[0] + pi*x[1];
				else
					data[startDataIndex + i] = x[0];
			} else {
				data[startDataIndex + i] = ucudaPickVar(x, pick);
			}

			//#pragma unroll
			for (int j = 0; j < preScaller; ++j) {
				//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
				UCUDA_STEP(x, values, h, krsPre);
			}
		}
		else
			UCUDA_STEP(x, values, h, krsPre);

		if (--untilCheck < 0) {
			untilCheck = CHECK_INTERVAL - 1;
			if (progressCounter != nullptr && progressStride > 0) {
				sinceReport += CHECK_INTERVAL;
				if (sinceReport >= (size_t)progressStride) {
#ifdef __CUDA_ARCH__
					atomicAdd(progressCounter, 1);
#endif
					if (ticksReported != nullptr) ++(*ticksReported);
					sinceReport = 0;
				}
			}
			// Cancel отдаёт REGIME_CANCELLED: по Stop хост показывает посчитанные точки,
			// а прерванные — пустыми, не путая их с расходимостью.
			if (ucudaCancelRequested(cancelFlag)) return REGIME_CANCELLED;
			checker = 0;
			//#pragma unroll
			for (int j = 0; j < AMOUNTOFX; ++j) {
				checker = checker + abs(x[j]);
			}

			// Если isnan или isinf - возвращаем false, ибо это нежелательное поведение системы
			if (isnan(checker) || isinf(checker))
			{
				//delete[] xPrev;
				return 0;
			}

			// Если maxValue == 0, это значит пользователь не выставил ограничение, иначе требуется его проверить
			if (maxValue != 0)
				if (abs(checker) > maxValue)
				{
					//delete[] xPrev;
					return 0;
				}
		}
	}

	//#pragma unroll
	for (int j = 0; j < AMOUNTOFX; ++j)
	{
		xPrev[j] = x[j];
	}
	// Если все-таки передали массив для записи - записываем значение переменной

	//#pragma unroll
	for (int j = 0; j < preScaller; ++j) {
		//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
		UCUDA_STEP(x, values, h, krsPre);
	}

	// Возврат — REGIME_* из configCUDA.h:
	//    1 (REGIME_OSCILLATION) — норма
	//   -1 (REGIME_FIXED_POINT) — сваливание в неподвижную точку
	//    0 (REGIME_UNBOUND)     — расходимость (возвращается выше, по checker)
	//// --- Проверка на сваливание в точку ---
	numb tempResult = 0;

	//for (int j = 0; j < 1; ++j)
	//#pragma unroll
	for (int j = 0; j < AMOUNTOFX; ++j)
	{
		tempResult += abs(x[j] - xPrev[j]);
	}

	//if (tempResult == 0)
	//{
	//	//delete[] xPrev;
	//	return -1;
	//}

	if (abs(tempResult) < eps_fixed_point)
	{
		//delete[] xPrev;
		return -1;
	}

	//delete[] xPrev;
	return 1;
}

// Тот же сэмпл, что записался бы в data[] (см. loopCalculateDiscreteModel_int).
__device__ __host__ __forceinline__ numb ucudaSampleOf(const numb* x, const int writableVar, const bool* pick)
{
	if (writableVar < 0) {
		if constexpr (AMOUNTOFX >= 3)
			return x[0] + pi*x[1] + euler*x[2];
		else if constexpr (AMOUNTOFX == 2)
			return x[0] + pi*x[1];
		else
			return x[0];
	}
	return ucudaPickVar(x, pick);
}

// Сколько сэмплов цикл с пиками обрабатывает за раз: сначала столько шагов (сэмплы — в
// регистры), потом PeakStream по ним развёрнуто. Окно w0/w1/w2 тогда не перекладывается
// с регистра на регистр каждый сэмпл. Замерено (БД 2D, float): 4 — лучше 1 и 8.
constexpr int UCUDA_PEAK_CHUNK = 4;

// loopCalculateDiscreteModelPeaks_int -- sibling of loopCalculateDiscreteModel_int
// that feeds each sample to a PeakStream instead of storing it. Everything else
// (preScaller, the divergence test every UCUDA_LOOP_CHECK_BLOCK steps, the trailing
// fixed-point check) is the same code path, so the returned REGIME_* is the same too.
//
// Body (wrapper below): x is the register copy of the state. EA = peaks.emitAll,
// PS1 = (preScaller == 1) -- template parameters, so neither is tested per sample.
// The chunked part reorders only the pushes: every step, and the divergence test
// after it, happens at the same i as in the plain loop; on an early exit the point
// is discarded, so the samples not yet pushed do not matter.
template <bool EA, bool PS1>
__device__ __host__ __forceinline__ int loopCalculateDiscreteModelPeaks_int_core(
	numb* __restrict__ x, const numb* __restrict__ values,
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, PeakStream& peaks,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported UCUDA_NREF)
{
	numb xPrev[AMOUNTOFX];
	size_t sinceReport = 0;
	// Обратный отсчёт до проверки (в хвосте и при emitAll) вместо i % блок: деление size_t
	// на каждом шаге стоило ~12% времени ядра. Проверки — после шагов 0, 1000, 2000, ...
	int untilCheck = 0;
	UcudaKrsPre krsPre;
	ucudaKrsPrep(krsPre, values, h);
	bool pick[AMOUNTOFX];
	ucudaPickMask(pick, writableVar);

	size_t i = 0;
	if (!EA) {
		const size_t chunked = amountOfIterations - amountOfIterations % UCUDA_PEAK_CHUNK;
		// Проверка — только в первой порции блока из UCUDA_LOOP_CHECK_BLOCK сэмплов (после шага
		// с номером, кратным блоку), остальные порции блока идут циклом без неё.
		static_assert(UCUDA_LOOP_CHECK_BLOCK % UCUDA_PEAK_CHUNK == 0, "the check must land on a chunk start");
		auto chunk = [&](bool check) -> int {
			numb samples[UCUDA_PEAK_CHUNK];
			#pragma unroll
			for (int k = 0; k < UCUDA_PEAK_CHUNK; ++k) {
				samples[k] = ucudaSampleOf(x, writableVar, pick);
				if (PS1) UCUDA_STEP(x, values, h, krsPre);
				else for (int j = 0; j < preScaller; ++j) UCUDA_STEP(x, values, h, krsPre);
				if (k == 0 && check) {
					const int r = ucudaLoopCheck(x, maxValue, cancelFlag, progressCounter, progressStride, ticksReported, sinceReport);
					if (r != UCUDA_LOOP_GO) return r;
				}
			}
			#pragma unroll
			for (int k = 0; k < UCUDA_PEAK_CHUNK; ++k)
				peaks.pushAt(samples[k], i + k);
			i += UCUDA_PEAK_CHUNK;
			return UCUDA_LOOP_GO;
		};
		constexpr int kChunksPerBlock = UCUDA_LOOP_CHECK_BLOCK / UCUDA_PEAK_CHUNK;
		while (i < chunked) {   // здесь i кратно блоку
			const int r = chunk(true);
			if (r != UCUDA_LOOP_GO) return r;
			if (chunked - i >= (size_t)(kChunksPerBlock - 1) * UCUDA_PEAK_CHUNK) {
				for (int c = 0; c < kChunksPerBlock - 1; ++c) chunk(false);
			} else {
				while (i < chunked) chunk(false);
			}
		}
		// Хвост ниже ведёт отсчёт сам: следующая проверка — на ближайшем i, кратном блоку.
		untilCheck = (i % UCUDA_LOOP_CHECK_BLOCK == 0) ? 0 : (int)(UCUDA_LOOP_CHECK_BLOCK - i % UCUDA_LOOP_CHECK_BLOCK);
	}

	for (; i < amountOfIterations; ++i)
	{
		const numb sample = ucudaSampleOf(x, writableVar, pick);
		if (EA) peaks.pushRaw(sample);
		else    peaks.pushAt(sample, i);

		if (PS1) UCUDA_STEP(x, values, h, krsPre);
		else for (int j = 0; j < preScaller; ++j) UCUDA_STEP(x, values, h, krsPre);

		if (--untilCheck < 0) {
			untilCheck = UCUDA_LOOP_CHECK_BLOCK - 1;
			const int r = ucudaLoopCheck(x, maxValue, cancelFlag, progressCounter, progressStride, ticksReported, sinceReport);
			if (r != UCUDA_LOOP_GO) return r;
		}
	}

	for (int j = 0; j < AMOUNTOFX; ++j)
		xPrev[j] = x[j];

	for (int j = 0; j < preScaller; ++j)
		UCUDA_STEP(x, values, h, krsPre);

	numb tempResult = 0;
	for (int j = 0; j < AMOUNTOFX; ++j)
		tempResult += abs(x[j] - xPrev[j]);

	if (abs(tempResult) < eps_fixed_point)
		return -1;

	return 1;
}

// Обёртки с прежними сигнатурами: состояние копируется из shared в регистры на входе
// и обратно на выходе (см. loopCalculateDiscreteModel_int_core).
__device__ __host__ int loopCalculateDiscreteModel_int(
	numb* xs, const numb* values,
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, numb* data,
	const size_t startDataIndex, const int writeStep,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported UCUDA_NP)
{
	numb x[AMOUNTOFX];
	UCUDA_NBEGIN
	for (int j = 0; j < AMOUNTOFX; ++j) x[j] = xs[j];
	const int r = loopCalculateDiscreteModel_int_core(x, values, h, amountOfIterations, amountOfX, preScaller,
		writableVar, maxValue, data, startDataIndex, writeStep, cancelFlag, progressCounter, progressStride, ticksReported UCUDA_NA(ucudaN));
	for (int j = 0; j < AMOUNTOFX; ++j) xs[j] = x[j];
	UCUDA_NEND
	return r;
}

__device__ __host__ int loopCalculateDiscreteModelPeaks_int(
	numb* xs, const numb* values,
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, PeakStream& peaks,
	const volatile int* cancelFlag, int* progressCounter, int progressStride, int* ticksReported UCUDA_NP)
{
	numb x[AMOUNTOFX];
	UCUDA_NBEGIN
	for (int j = 0; j < AMOUNTOFX; ++j) x[j] = xs[j];
	int r;
	if (peaks.emitAll)
		r = loopCalculateDiscreteModelPeaks_int_core<true, false>(x, values, h, amountOfIterations, amountOfX,
			preScaller, writableVar, maxValue, peaks, cancelFlag, progressCounter, progressStride, ticksReported UCUDA_NA(ucudaN));
	else if (preScaller == 1)
		r = loopCalculateDiscreteModelPeaks_int_core<false, true>(x, values, h, amountOfIterations, amountOfX,
			preScaller, writableVar, maxValue, peaks, cancelFlag, progressCounter, progressStride, ticksReported UCUDA_NA(ucudaN));
	else
		r = loopCalculateDiscreteModelPeaks_int_core<false, false>(x, values, h, amountOfIterations, amountOfX,
			preScaller, writableVar, maxValue, peaks, cancelFlag, progressCounter, progressStride, ticksReported UCUDA_NA(ucudaN));
	for (int j = 0; j < AMOUNTOFX; ++j) xs[j] = x[j];
	UCUDA_NEND
	return r;
}

#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_distributedCalculateDiscreteModelCUDA)
__global__ void distributedCalculateDiscreteModelCUDA(
	const size_t		amountOfPointsForSkip,
	const int		amountOfThreads,
	const numb	h,
	const numb	hSpecial,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar,
	numb*			data)
{
	extern __shared__ numb s[];
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfThreads)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime )
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0);

	loopCalculateDiscreteModel(localX, localValues, h, idx,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0, 0);

	loopCalculateDiscreteModel(localX, localValues, hSpecial, amountOfIterations,
		amountOfInitialConditions, 1, writableVar, 0, data, idx, amountOfThreads);

	return;
}
#endif

// Глобальная функция, которая вычисляет траекторию нескольких систем

// ucudaSetupSweepPoint -- which point of the sweep this thread is, extracted so
// that the trajectory kernel and its fused (trajectory-free) sibling cannot
// drift apart. The node values, the log axes and the dt-sweep bookkeeping are
// the delicate part; configCUDA.h already documents what happens when one
// formula gets copied by hand into several places.
//
// Returns false if the point is degenerate (h <= 0) and must be skipped.
__device__ __forceinline__ bool ucudaSetupSweepPoint(
	const int nPts, const size_t amountOfCalculatedPoints, const int idx,
	const int dimension, const numb* ranges, const numb h,
	const int* indicesOfMutVars,
	const numb* initialConditions, const int amountOfInitialConditions,
	const numb* values, const int amountOfValues,
	const int hSweepAxis, const int logAxisMask,
	const numb transientTime, const numb tMax, const int preScaller,
	const size_t amountOfPointsForSkip, const size_t amountOfIterations,
	numb* localX, numb* localValues,
	numb& h_local, size_t& skip_local, size_t& iters_local)
{
	// Per-thread h override for a dt-sweep axis; falls back to the uniform
	// launch-wide h otherwise (identical to the pre-dt-sweep behaviour).
	h_local = h;
	if (hSweepAxis == 0)
		h_local = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
	else if (hSweepAxis == 1)
		h_local = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);

	if (h_local <= 0)
		return false;

	// Buffers are sized for the worst case (smallest h in the swept range ->
	// most steps) when hSweepAxis != -1; the actual per-thread step counts are
	// recomputed here from h_local.
	skip_local  = (hSweepAxis != -1) ? (size_t)ceil(transientTime / h_local)             : amountOfPointsForSkip;
	iters_local = (hSweepAxis != -1) ? (size_t)ceil(tMax / h_local / (numb)preScaller)   : amountOfIterations;
	if (iters_local > amountOfIterations) iters_local = amountOfIterations;  // defensive clamp to the allocated buffer

	// Определяем localX[] начальными условиями
	#pragma unroll
	for ( int i = 0; i < AMOUNTOFX; ++i )
		localX[i] = initialConditions[i];

	// Определяем localValues[] начальными параметрами
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// Меняем значение изменяемых параметров на результат функции getValueByIdx
	if (par_or_var == 1) {
		for (int i = 0; i < dimension; ++i) {
			if (i == hSweepAxis) continue;
			bool isLog = (logAxisMask >> i) & 1;
			localValues[indicesOfMutVars[i]] = isLog ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i)
			                                          : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 0) {
		for (int i = 0; i < dimension; ++i) {
			if (i == hSweepAxis) continue;
			bool isLog = (logAxisMask >> i) & 1;
			// ±log — только у бассейнов (оси НУ); остальные свипы этот бит не ставят.
			if ((logAxisMask >> (UCUDA_SYMLOG_SHIFT + i)) & 1)
				localX[indicesOfMutVars[i]] = getValueByIdx_symlog(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
			else
			localX[indicesOfMutVars[i]] = isLog ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i)
			                                     : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 2) {
		// Host never combines mixed param/IC (par_or_var==2) with an h-axis in
		// the same request (see run_bif2d) -- this branch is unaffected.
		localX[indicesOfMutVars[0]]      = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
		                                                             : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
		localValues[indicesOfMutVars[1]] = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
		                                                             : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}
#ifdef UCUDA_LYAP_NX
	// ДАУ (этап 7): связь g(x, z; p) = 0 зависит от параметров точки, поэтому z согласуется здесь, для
	// каждой точки: шаг метода ДАУ при h = 0 — Ньютон по z при фиксированном x (индекс 1; на индексе >= 2
	// пусто). Старт — z из настроек: это и есть правило выбора ветви при нескольких корнях.
	calculateDiscreteModel(localX, UCUDA_TA((numb)0) localValues, (numb)0);
#endif
	return true;
}

#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_calculateDiscreteModelCUDA)
__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const size_t	sizeOfBlock,
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
	const size_t		amountOfIterations, 
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			data,
	int*			maxValueCheckerArray,
	const bool		Par_or_Var,
	const int		hSweepAxis,
	const numb	transientTime,
	const numb	tMax,
	int*			actualIterations,
	const int		logAxisMask,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride)
{
	// Общая память в рамках одного блока
	// Строение памяти (per-thread slice, шаг ucuda_shared_stride):
	// {localX_0..localX_n, localValues_0..localValues_m, [pad], следуюший поток...}

	extern __shared__ numb s[];
	//////// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	// Odd stride keeps both arrays bank-conflict-free; the launch allocates
	// exactly ucuda_shared_stride * blockDim.x numb's (see configCUDA.h).
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + ( threadIdx.x * sharedStride );
	numb* localValues = localX + amountOfInitialConditions;

	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	numb   h_local;
	size_t skip_local;
	size_t iters_local;
	if (!ucudaSetupSweepPoint(nPts, amountOfCalculatedPoints, idx, dimension, ranges, h,
			indicesOfMutVars, initialConditions, amountOfInitialConditions,
			values, amountOfValues, hSweepAxis, logAxisMask,
			transientTime, tMax, preScaller,
			amountOfPointsForSkip, amountOfIterations,
			localX, localValues, h_local, skip_local, iters_local)) {
		// Degenerate/zero step: skip entirely, same as an unbound/diverged solution
		// (flag=-1) -- downstream (peakFinderCUDA, host unpack) already treats -1
		// as "no data for this point", so no separate NaN-fill is needed here.
		if (maxValueCheckerArray != nullptr) maxValueCheckerArray[idx] = -1;
		if (actualIterations != nullptr)     actualIterations[idx] = 0;
		ucudaProgressTopUp(progressCounter, progressStride, amountOfPointsForSkip + amountOfIterations, 0);
		return;
	}
	if (actualIterations != nullptr) actualIterations[idx] = (int)iters_local;

	// flag — REGIME_* из configCUDA.h: 1 = OSCILLATION, -1 = FIXED_POINT,
	// 0 = UNBOUND. (Раньше здесь был комментарий с перепутанными 0 и -1.)
	int ticks = 0;
	UCUDA_NVAR(ucudaStep, 0)   // номер шага траектории: t = n*h (UCUDA_NONAUTONOMOUS)
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h_local, skip_local,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, (size_t)idx * sizeOfBlock, 1,
		cancelFlag, progressCounter, progressStride, &ticks UCUDA_NA(&ucudaStep));

	// Теперь уже по-взрослому моделируем систему
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h_local, iters_local,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, (size_t)idx * sizeOfBlock, 1,
			cancelFlag, progressCounter, progressStride, &ticks UCUDA_NA(&ucudaStep));

	// Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}
	ucudaProgressTopUp(progressCounter, progressStride, skip_local + iters_local, ticks);
	//delete[] localX;
	//delete[] localValues;
	return;
}
#endif

// calculateDiscreteModelPeaksCUDA -- calculateDiscreteModelCUDA fused with
// peakFinderCUDA. Same sweep point (shared ucudaSetupSweepPoint), same
// integrator, same REGIME_* semantics; the difference is that the trajectory is
// consumed by a PeakStream as it is produced instead of being written to a
// per-thread block of global memory. That block is the term that dominated the
// 2D memory budget, so dropping it is what lets the whole grid run at once.
//
// Inter-peak intervals are scaled by h_local, not by the launch-wide h the
// split path could only pass: under a dt-sweep every point has its own step.
#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_calculateDiscreteModelPeaksCUDA)
__global__ void calculateDiscreteModelPeaksCUDA(
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
	numb*			outPeaks,
	numb*			timeOfPeaks,
	int*			maxValueCheckerArray,
	const bool		Par_or_Var,
	const int		hSweepAxis,
	const numb		transientTime,
	const numb		tMax,
	const int		logAxisMask,
	const size_t	peakStride,
	const int		peakCapacity,
	const volatile int* cancelFlag,   // device-side, polled once per CHECK_INTERVAL steps
	int*			progressCounter,  // device-side; one atomic per progressStride steps of work
	const int		progressStride,   // 0 = не считать прогресс
	const bool		emitAllSamples)   // true = every iterate, not just the peaks
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + ( threadIdx.x * sharedStride );
	numb* localValues = localX + amountOfInitialConditions;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	int    ticks = 0;   // сколько эта ячейка уже отчитала
	numb   h_local;
	size_t skip_local;
	size_t iters_local;
	if (!ucudaSetupSweepPoint(nPts, amountOfCalculatedPoints, idx, dimension, ranges, h,
			indicesOfMutVars, initialConditions, amountOfInitialConditions,
			values, amountOfValues, hSweepAxis, logAxisMask,
			transientTime, tMax, preScaller,
			amountOfPointsForSkip, amountOfIterations,
			localX, localValues, h_local, skip_local, iters_local)) {
		if (maxValueCheckerArray != nullptr) maxValueCheckerArray[idx] = -1;
		if (progressCounter != nullptr && progressStride > 0)
			atomicAdd(progressCounter, (int)((amountOfPointsForSkip + amountOfIterations) / (size_t)progressStride));
		return;
	}

	UCUDA_NVAR(ucudaStep, 0)
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h_local, skip_local,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, 0, 1,
		cancelFlag, progressCounter, progressStride, &ticks UCUDA_NA(&ucudaStep));

	if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT) {
		PeakStream peaks;
		peaks.init(outPeaks, timeOfPeaks, (size_t)idx * peakStride,
			h_local * (numb)preScaller, iters_local, peakCapacity, emitAllSamples);

		flag = loopCalculateDiscreteModelPeaks_int(localX, localValues, h_local, iters_local,
			amountOfInitialConditions, preScaller, writableVar, maxValue, peaks,
			cancelFlag, progressCounter, progressStride, &ticks UCUDA_NA(&ucudaStep));

		// peakFinderCUDA leaves FP and UNBOUND codes in the array untouched and
		// only replaces OSCILLATION with the peak count -- same rule here.
		//
		// Emitting every sample takes the fixed point too: for a map a period-1
		// orbit is a legitimate single branch, and the samples are already in the
		// buffer (the FP verdict is decided after the loop has run in full).
		// Dropping it is what leaves the periodic windows blank.
		if (flag == REGIME_OSCILLATION || (peaks.emitAll && flag == REGIME_FIXED_POINT)) {
			if (maxValueCheckerArray != nullptr)
				maxValueCheckerArray[idx] = peaks.emitAll ? peaks.countRaw() : peaks.count();
			// Добивка до ожидаемого числа тиков — см. тот же блок ниже.
			if (progressCounter != nullptr && progressStride > 0) {
				const size_t totalSteps = skip_local + iters_local;
				const int expected = (int)(totalSteps / (size_t)progressStride);
				if (expected > ticks) atomicAdd(progressCounter, expected - ticks);
			}
			return;
		}
	}

	if (maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = flag;
	// Добивка до ожидаемого числа тиков: ячейка, упавшая в расходимость
	// на первом шагу, иначе недосчитала бы свою долю, и бар застрял бы.
	if (progressCounter != nullptr && progressStride > 0) {
		const size_t totalSteps = skip_local + iters_local;
		const int expected = (int)(totalSteps / (size_t)progressStride);
		if (expected > ticks) atomicAdd(progressCounter, expected - ticks);
	}
}
#endif

#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_calculateDiscreteModelCUDA_H)
__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const numb	transientTime,
	const int		dimension,
	numb*			ranges,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const numb	tMax,
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			data,
	int*			maxValueCheckerArray)
{
	// Общая память в рамках одного блока
	// Строение памяти:
	// {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...}
	extern __shared__ numb s[];

	// В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Определяем localX[] начальными условиями
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// Определяем localValues[] начальными параметрами
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	//for (int i = 0; i < dimension; ++i)
	//	localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
	//		nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	numb h = (numb)pow((numb)10.0, getValueByIdxLog(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0));

	// Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime )
	loopCalculateDiscreteModel(localX, localValues, h, transientTime / h,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// Теперь уже по-взрослому моделируем систему
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, tMax / h / preScaller,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;
	else
		maxValueCheckerArray[idx] = tMax / h / preScaller;

	return;
}
#endif

#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_calculateDiscreteModelICCUDA)
__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const size_t		amountOfPointsForSkip,
	const int		dimension, 
	numb*			ranges, 
	const numb	h,
	int*			indicesOfMutVars, 
	numb*			initialConditions,
	const int		amountOfInitialConditions, 
	const numb*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const numb	maxValue, 
	numb*			data, 
	int*			maxValueCheckerArray)
{
	// Общая память в рамках одного блока
	// Строение памяти:
	// {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...}
	extern __shared__ numb s[];

	// В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами
	numb* localX = s + ( threadIdx.x * amountOfInitialConditions );
	numb* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	//numb* localX = new numb[amountOfInitialConditions];
	//numb* localValues = new numb[amountOfValues];

	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Определяем localX[] начальными условиями
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// Определяем localValues[] начальными параметрами
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// Меняем значение изменяемых параметров на результат функции getValueByIdx
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	//__device__ __host__ numb getValueByIdx(const int idx, const int nPts,
	//	const numb startRange, const numb finishRange, const int valueNumber)
	//{
	//	return startRange + (((int)((int)idx / powf((numb)nPts, (numb)valueNumber)) % nPts)
	//		* ((numb)(finishRange - startRange) / (numb)(nPts - 1)));
	//}

	//// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	//bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
	//	amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	//// --- Теперь уже по-взрослому моделируем систему --- 
	//if (flag)
	//	flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
	//	amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	//// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	//if (!flag && maxValueCheckerArray != nullptr)
	//	maxValueCheckerArray[idx] = -1;
	//else
	//	maxValueCheckerArray[idx] = 1;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime )

	// flag — REGIME_* из configCUDA.h: 1 = OSCILLATION, -1 = FIXED_POINT, 0 = UNBOUND
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// Теперь уже по-взрослому моделируем систему
	if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	//delete[] localX;
	//delete[] localValues;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}
#endif

#if !defined(UCUDA_PRUNE_KERNELS) || defined(UCUDA_KEEP_calculateDiscreteModelICCUDA_logAxes)
__global__ void calculateDiscreteModelICCUDA_logAxes(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb* ranges,
	const numb	h,
	int* indicesOfMutVars,
	numb* initialConditions,
	const int		amountOfInitialConditions,
	const numb* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb* data,
	int* maxValueCheckerArray)
{
	// Общая память в рамках одного блока
	// Строение памяти:
	// {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...}
	extern __shared__ numb s[];

	// В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Определяем localX[] начальными условиями
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// Определяем localValues[] начальными параметрами
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// Меняем значение изменяемых параметров на результат функции getValueByIdx
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime )

	// flag — REGIME_* из configCUDA.h: 1 = OSCILLATION, -1 = FIXED_POINT, 0 = UNBOUND
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// Теперь уже по-взрослому моделируем систему
	if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations, amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}
	return;
}
#endif

// Функция, которая находит индекс в последовательности значений

//__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
//	const numb startRange, const numb finishRange, const int valueNumber)
//{
//	return startRange + (numb)( ( (int64_t)( (int64_t)idx / pow( (numb)nPts, (numb)valueNumber) ) % (int64_t)nPts ) * ( (numb)( finishRange - startRange ) / (numb)( nPts - 1.0) ) );
//}

//__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
//	const numb startRange, const numb finishRange,
//	const int valueNumber)
//{
//	if (nPts <= 1) return startRange; // Или (startRange + finishRange) / 2, в зависимости от логики
//
//	// 1. Целочисленное вычисление индекса точки по оси (без pow и float-деления)
//	// Если valueNumber == 0 (ось X), делитель равен 1.
//	// Если valueNumber == 1 (ось Y), делитель равен nPts.
//	const int64_t divisor = (valueNumber == 0) ? 1 : nPts;
//	const int64_t pointIdx = (idx / divisor) % nPts;
//
//	// 2. Вычисление шага сетки. 
//	// Приводим (nPts - 1) к типу numb, чтобы избежать неявного повышения до double
//	const numb step = (finishRange - startRange) / static_cast<numb>(nPts - 1);
//
//	// 3. Итоговое значение
//	return startRange + static_cast<numb>(pointIdx) * step;
//}

__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
	const numb startRange, const numb finishRange,
	const int valueNumber)
{
	// 1. Обработка вырожденных случаев
	if (nPts <= 0) return startRange;
	if (nPts == 1) return finishRange;

	// 2. Целочисленное вычисление индекса точки по оси
	const int64_t divisor = (valueNumber == 0) ? 1 : nPts;
	const int64_t pointIdx = (idx / divisor) % nPts;

	// 3. Значение узла — ЕДИНСТВЕННАЯ реализация формулы (configCUDA.h).
	// Прежняя арифметика (lerp, «обеспечивает строгую симметрию») сохранена
	// внутри ucuda_node_value без изменений; сюда её вынесли, чтобы UI и CSV
	// считали узел ТОЙ ЖЕ формулой, а не одной из четырёх похожих.
	return ucuda_node_value((int)pointIdx, nPts, startRange, finishRange);
}

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	numb log10_xstart = log10(startRange);
	numb log10_xstop = log10(finishRange);
	int n = ((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);
	//0 1 2 3 4 5	n = 6
	if (n < nPts / 2) {
		return -pow(10, log10_xstart + (numb)n * (log10_xstop - log10_xstart) / (numb)(nPts / 2 - 1));
	}
	else {
		return pow(10, log10_xstop + (numb)(n - nPts / 2) * (log10_xstart - log10_xstop) / (numb)(nPts / 2 - 1));
	}

}

// Функция, которая находит индекс в последовательности значений
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	// Извлечение индекса узла оставлено как было (pow-делитель по valueNumber);
	// сама формула значения — общая, см. ucuda_node_value_log10 в configCUDA.h.
	// Порядок операций там тот же, что стоял здесь: шаг делится до умножения.
	const int n = (int)((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);
	return ucuda_node_value_log10(n, nPts, startRange, finishRange);
}

// getValueByIdx для log-равномерной сетки (drop-in замена getValueByIdx, тот
// же calling convention) -- требует startRange, finishRange > 0 (проверяется
// на host-стороне до запуска kernel'а).
//
// Композиция pow(10, log10-узел) раньше стояла здесь отдельной строкой, т.е.
// была ещё одной копией формулы: UI считал её у себя, и в float-режиме обе
// стороны выбирали разные перегрузки pow. Теперь и то, и другое — одна
// ucuda_node_value_log (configCUDA.h). Извлечение индекса узла оставлено
// как было (pow-делитель по valueNumber, см. getValueByIdxLog).
__device__ __host__ __forceinline__ numb getValueByIdx_log(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	const int n = (int)((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);
	return ucuda_node_value_log(n, nPts, startRange, finishRange);
}

// getValueByIdx для симметричной лог-сетки бассейнов (±log): узел — ucuda_node_value_symlog
// (configCUDA.h), извлечение индекса — как у getValueByIdx_log.
__device__ __host__ __forceinline__ numb getValueByIdx_symlog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	const int n = (int)((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);
	return ucuda_node_value_symlog(n, nPts, startRange, finishRange);
}

// Находит пики в интервале [startDataIndex; startDataIndex + amountOfPoints] в "data" массиве
//peakFinder(data, idx* sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

__device__ __host__ void MeanAndMedianFreq(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq)
{
	numb MeanFreq = 0;

	for (size_t i = 0; i < amountOfPeaks; i++) {
		MeanFreq = MeanFreq + ((numb)1.0 / timeOfPeaks[startDataIndex + i]);
	}
	meanFreq[idx] = MeanFreq / (numb)amountOfPeaks;

	medianFreq[idx] = 1.0 / timeOfPeaks[startDataIndex];

}

__device__ __host__ void MeanAndVariance(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval)
{
	numb localValue = 0;

	for (size_t i = 0; i < amountOfPeaks; i++) {
		localValue = localValue + outPeaks[startDataIndex + i];
	}
	meanPeak[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; i++) {
		localValue = localValue + timeOfPeaks[startDataIndex + i];
	}
	meanInterval[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; ++i)
		localValue = localValue + (outPeaks[startDataIndex + i] - meanPeak[idx]) * (outPeaks[startDataIndex + i] - meanPeak[idx]);

	variancePeak[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; ++i)
		localValue = localValue + (timeOfPeaks[startDataIndex + i] - meanInterval[idx]) * (timeOfPeaks[startDataIndex + i] - meanInterval[idx]);

	varianceInterval[idx] = localValue / (numb)amountOfPeaks;

	localValue = -9999999999;

	for (size_t i = 0; i < amountOfPeaks; ++i)
		if (outPeaks[startDataIndex + i] > localValue)
			localValue = outPeaks[startDataIndex + i];

	maxPeak[idx] = localValue;

	localValue = -9999999999;

	for (size_t i = 0; i < amountOfPeaks; ++i)
		if (timeOfPeaks[startDataIndex + i] > localValue)
			localValue = timeOfPeaks[startDataIndex + i];

	maxInterval[idx] = localValue;
}

__device__ __host__ numb globalPeakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints)
{
		// Переменная для хранения найденных пиков
		numb maxValue = -1e25;
		// Начинаем просматривать заданных интервал на наличие пиков
		for (size_t i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i)
		{
			// Если текущая точка больше предыдущей и больше ИЛИ РАВНА следующей, то... ( не факт, что это пик ( например: 2 3 3 4 ) )
			if (data[i] >= maxValue) //
			{
				maxValue = data[i];
			}

		}

		return maxValue;
}

