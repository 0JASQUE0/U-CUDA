__device__ __host__ void calculateDiscreteModelforFastSynchro(numb* X, numb* S1, numb* K, const numb* a, const numb h, const bool directionOfintegration)
{
	// Generic FastSynchro step. Раньше тело было ЗАХАРДКОЖЕНО под Lorenz CD;
	// теперь делегируем сам ОДУ-шаг в user's `calculateDiscreteModel` (KRS_BODY
	// от NVRTC template). FS-семантика:
	//
	//   • directionOfintegration = 1 (forward, h>0)  → caller передаёт K_forward
	//   • directionOfintegration = 0 (backward, h<0) → caller передаёт K_backward
	//     (см. loopCalculateDiscreteModelForFastSynchro — K_local меняется
	//      между прямым и обратным проходами).
	//
	//   1. ОДУ-шаг с инвертированным знаком h на обратном проходе.
	//   2. Coupling-тяга slave→master: ВСЕГДА с положительным h, независимо
	//      от direction. Исходная конвенция FS-алгоритма (Lorenz-CD реализация
	//      делала forward: X += N*h; backward: X -= (-N)*h — двойное отрицание
	//      = +N*h). Так slave притягивается к master И на forward, И на backward
	//      проходах. Если использовать eff_h здесь — backward пасс будет
	//      отталкивать → расходимость.
	const numb eff_h = directionOfintegration ? h : -h;
	numb N[AMOUNTOFX];

	for (int i = 0; i < AMOUNTOFX; ++i)
		N[i] = K[i] * (S1[i] - X[i]);

	calculateDiscreteModel(X, a, eff_h);

	for (int i = 0; i < AMOUNTOFX; ++i)
		X[i] += h * N[i];
}

// Детектор разлёта для FastSynchro On Grid
// Соглашение то же, что в loopCalculateDiscreteModel: maxValue == 0 означает
// "пользователь не выставил ограничение", nan/inf ловятся всегда. В отличие от
// тех функций здесь проверяется ВЕСЬ вектор состояния, а не один writableVar:
// у FS нет выделенной наблюдаемой переменной, ошибка синхронизации считается
// по всем компонентам сразу.
__device__ __forceinline__ bool fsStateDiverged(const numb* X, const int amountOfX, const numb maxValue)
{
	for (int j = 0; j < amountOfX; ++j) {
		if (isnan(X[j]) || isinf(X[j])) return true;
		if (maxValue != 0 && fabs(X[j]) > maxValue) return true;
	}
	return false;
}

// Что записать в ячейку, которая разлетелась. Именно NaN, а не насыщенное
// значение вроде 999 или maxValue: ошибка синхронизации здесь не «очень
// большая», а неизмеримая, и подмена числом врала бы в статистике и в
// автошкале colormap'а. Хост уже отфильтровывает non-finite при поиске
// min/max, а HeatmapView::upload_data красит такие ячейки тёмно-серым.
__device__ __forceinline__ numb fsDivergedError()
{
	return (numb)nan("");
}

// splitmix64 — детерминированный хэш-ГПСЧ для рандомных НУ второй системы
// в FastSynchro (icRandomOffset).
//
// Почему не curand: под NVRTC curand_kernel.h недоступен и подменён
// заглушкой-LCG в kernels/fastsync_*.template.cu, а у неё соседние seed'ы
// дают коррелированные первые выборки — при seed = индекс ячейки это рисует
// на карте регулярный узор вместо шума. splitmix64 перемешивает соседние
// счётчики полностью и компилируется одинаково обоими путями (nvcc и NVRTC),
// без внешних зависимостей.
__device__ __host__ __forceinline__
unsigned long long fsSplitMix64(unsigned long long x)
{
	unsigned long long z = x + 0x9E3779B97F4A7C15ULL;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

// Равномерное значение в [-1, 1) по счётчику (seed, ячейка, компонента).
// Не хранит состояние: одна и та же тройка всегда даёт одно и то же число,
// поэтому расчёт воспроизводится бит-в-бит независимо от разбиения на чанки
// и от порядка запуска блоков.
__device__ __host__ __forceinline__
numb fsRandSymmetric(unsigned long long seed, unsigned long long cell, int comp)
{
	unsigned long long h = fsSplitMix64(seed
		^ (cell * 0x9E3779B97F4A7C15ULL)
		^ ((unsigned long long)comp * 0xD1B54A32D192ED03ULL));
	// Старшие 53 бита → [0, 1), затем в [-1, 1).
	numb u = (numb)(h >> 11) * (numb)(1.0 / 9007199254740992.0);
	return (numb)2 * u - (numb)1;
}

// Добивка тиков прогресса перед выходом из ядра: точка, ушедшая
// с дистанции раньше срока (расходимость, отмена), иначе недосчитала бы
// свою долю, и бар застрял бы не дойдя до ста.
__device__ __forceinline__ void ucudaProgressTopUp(int* progressCounter, int progressStride,
	size_t totalSteps, int ticks)
{
	if (progressCounter == nullptr || progressStride <= 0) return;
	const int expected = (int)(totalSteps / (size_t)progressStride);
	if (expected > ticks) atomicAdd(progressCounter, expected - ticks);
}

__global__ void calculateDiscreteModelICCforFastSynchro(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		dimension,
	numb* ranges,
	const numb	h,
	int* indicesOfMutVars,
	numb* initialConditions,
	numb* initialConditionsSlave,
	const int		amountOfInitialConditions,
	const numb* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const numb	maxValue,
	const int		iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	numb* data,
	int* maxValueCheckerArray,
	numb* FastSynchroError,
	int   swapRole,
	const size_t amountOfPointsForSkipMaster,
	const size_t amountOfPointsForSkipSlave,
	const int    icRandomOffset,
	const numb   icEps,
	const unsigned long long icSeed,
	const int    gsWarmup,
	const volatile int* cancelFlag,
	int* progressCounter,
	const int progressStride)
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

	// Определяем localX[] начальными условиями master + lokalSlave для slave
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];
	numb localSlave[AMOUNTOFX];
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localSlave[i] = initialConditionsSlave[i];

	// Определяем localValues[] начальными параметрами
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Применяем grid-override либо на master, либо на slave (по swapRole).
	// swapRole=0 → varies master (legacy default), slave фиксирован.
	// swapRole=1 → varies slave, master фиксирован.
	for (int i = 0; i < dimension; ++i) {
		numb v = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		if (swapRole == 0) localX[indicesOfMutVars[i]]     = v;
		else               localSlave[indicesOfMutVars[i]] = v;
	}

	// Транзиент (TT): у master и slave он СВОЙ и независимый
	// Считается per-cell и уже ПОСЛЕ grid-override, поэтому расклад получается
	// ровно такой, какой нужен:
	//   swapRole==0 (сетка свипует master) — master досаживается в КАЖДОЙ
	//     ячейке из её собственной затравки, slave — из одной фиксированной
	//     точки (его старт от ячейки не зависит, путь один и тот же);
	//   swapRole==1 — симметрично наоборот.
	// Никакого ветвления по swapRole здесь не нужно: разница целиком в том,
	// какой стороне grid-override уже подменил координаты.
	//
	// Фиксированная сторона проходит один и тот же детерминированный путь в
	// каждом потоке — результат побитово тот же, что дала бы однопоточная
	// досадка на хосте, но без её главной беды: последовательный прогон в
	// 1 поток на длинных TT упирался в WDDM-watchdog (TDR) и ронял контекст.
	//
	// Uncoupled: K=0 → N[i] = K[i]*(S1[i]-X[i]) = 0, связь занулена, поэтому
	// передавать сам X как S1 безопасно (алиасинг ни на что не влияет).
	//
	// icRandomOffset: НУ фиксированной стороны берутся не из своей затравки, а
	// из точки свипуемой стороны ПОСЛЕ её транзиента, со случайным отступом в
	// eps-окрестности (см. блок сразу после транзиента). Собственный транзиент
	// фиксированной стороны в этом режиме смысла не имеет — он увёл бы её из
	// этой самой окрестности, — поэтому обнуляется.
	size_t skipMaster = amountOfPointsForSkipMaster;
	size_t skipSlave  = amountOfPointsForSkipSlave;
	if (icRandomOffset) {
		if (swapRole == 0) skipSlave = 0;
		else               skipMaster = 0;
	}

	if (skipMaster > 0 || skipSlave > 0) {
		numb transientZeros[AMOUNTOFX];
		for (int i = 0; i < AMOUNTOFX; ++i)
			transientZeros[i] = 0;

		// Разлёт проверяется на каждом шаге, как в loopCalculateDiscreteModel:
		// траектория может выскочить за maxValue и вернуться, и это уже повод
		// не считать ячейку. Заодно диверг-ячейки выходят из цикла рано, так
		// что проверка себя окупает.
		bool diverged = false;

		for (size_t step = 0; step < skipMaster && !diverged; ++step) {
			calculateDiscreteModelforFastSynchro(localX, localX, transientZeros, localValues, h, 1);
			diverged = fsStateDiverged(localX, amountOfInitialConditions, maxValue);
		}

		for (size_t step = 0; step < skipSlave && !diverged; ++step) {
			calculateDiscreteModelforFastSynchro(localSlave, localSlave, transientZeros, localValues, h, 1);
			diverged = fsStateDiverged(localSlave, amountOfInitialConditions, maxValue);
		}

		if (diverged) {
			// Синхро-цикл на разлетевшемся состоянии считать нечего.
			FastSynchroError[idx] = fsDivergedError();
			if (maxValueCheckerArray != nullptr) maxValueCheckerArray[idx] = -1;
			return;
		}
	}

	// --- Рандомные НУ второй системы (icRandomOffset == 1).
	// Свипуемая сторона уже стоит в точке своего аттрактора (grid-override +
	// её транзиент отработали выше); вторая стартует от ЭТОЙ точки со
	// случайным отступом в кубе [-eps, +eps] по каждой координате. Кто из
	// двух свипуемый — решает swapRole, ровно как и для grid-override.
	//
	// Счётчик ГПСЧ — ГЛОБАЛЬНЫЙ индекс ячейки (amountOfCalculatedPoints + idx),
	// а не idx внутри чанка: иначе отступ повторялся бы от чанка к чанку и
	// результат зависел бы от того, на сколько запусков разбилась сетка.
	if (icRandomOffset) {
		const unsigned long long cell = (unsigned long long)(amountOfCalculatedPoints + idx);
		if (swapRole == 0)
			for (int i = 0; i < amountOfInitialConditions; ++i)
				localSlave[i] = localX[i] + icEps * fsRandSymmetric(icSeed, cell, i);
		else
			for (int i = 0; i < amountOfInitialConditions; ++i)
				localX[i] = localSlave[i] + icEps * fsRandSymmetric(icSeed, cell, i);
	}

	// Возврат здесь — не REGIME_*, а RMS ошибки синхронизации (numb).
	// NaN означает, что ячейка разлетелась (по maxValue или по nan/inf).
	int ticks = 0;
	const numb fsError = loopCalculateDiscreteModelForFastSynchro_2(localX, localSlave, localValues, h, amountOfIterations,
		amountOfInitialConditions, maxValue, iterOfSynchr, kForward, kBackward, data, idx * sizeOfBlock, 1, icEps, gsWarmup,
		cancelFlag, progressCounter, progressStride, &ticks);

	FastSynchroError[idx] = fsError;
	ucudaProgressTopUp(progressCounter, progressStride, (size_t)amountOfIterations, ticks);
	if (maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = (isnan(fsError) || isinf(fsError)) ? -1 : 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}

__device__ numb loopCalculateDiscreteModelForFastSynchro_2(
	numb* x,
	const numb* initialConditionsSlave,
	const numb* values,
	const numb h,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	const int	iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	numb* data,
	const int startDataIndex,
	const int writeStep,
	const numb icEps,
	const int gsWarmup,
	const volatile int* cancelFlag,
	int* progressCounter,
	const int progressStride,
	int* ticksReported)
{
	size_t sinceReport = 0;
	//numb* Xm = new numb[amountOfX];
	//numb* Xs = new numb[amountOfX];
	//numb* arrayZeros = new numb[amountOfX];
	//numb* K_local = new numb[amountOfX];

	numb Xm[AMOUNTOFX];
	numb Xs[AMOUNTOFX];
	numb X_prev[AMOUNTOFX];
	numb arrayZeros[AMOUNTOFX];
	numb K_local[AMOUNTOFX];

	numb rms_error = 0;

	// Инициализация состояния — БЕЗУСЛОВНО. Раньше она жила внутри
	// `if (data != nullptr)`, из-за чего условие «писать траекторию» и условие
	// «инициализировать Xm/Xs/arrayZeros» были склеены: вызов с data == nullptr
	// прогонял бы весь синхро-цикл на мусоре из регистров. Сейчас такой вызов
	// не встречается (grid всегда передаёт буфер), но мина была настоящая.
	for (int w = 0; w < amountOfX; w++) {
		Xs[w] = initialConditionsSlave[w];
		Xm[w] = x[w];
		arrayZeros[w] = 0;
	}

	// error_estim 4/5 — сжатие ошибки за проходы синхронизации:
	//   err_start — ‖Xm(t0) − Xs(t0)‖ в начале окна ДО синхронизации;
	//   err_stop  — ‖Xm(t0) − Xs(t0)‖ в ТОЙ ЖЕ точке после iterOfSynchr проходов
	//               вперёд-назад (последовательность кончается обратным проходом,
	//               лишнего прямого в конце нет);
	//   4 → err_stop / err_start, 5 → log10(err_stop / err_start).
	// В отличие от оценщиков 0..3 последний обратный проход идёт в зачёт: именно
	// он возвращает slave в начало окна, где и сравниваем.
	//
	// Считаем здесь: Xm ещё равен старту мастера при обоих type_of_synch —
	// пред-пасс unidir, прокручивающий Xm вперёд для заливки data, идёт ниже.
	numb err_start = 0;
	if (error_estim == 4 || error_estim == 5 || error_estim == 6) {
		for (int j = 0; j < amountOfX; ++j) {
			numb e = Xm[j] - Xs[j];
			err_start += e * e;
		}
		err_start = sqrt(err_start);
	}

	if (data != nullptr) {
		for (int w = 0; w < amountOfX; w++)
			data[startDataIndex + w] = x[w];
	}
	if (type_of_synch == 0) {
		for (int i = 1; i < amountOfIterations; i++) {
			// Та же схема, что и в loopCalculateDiscreteModel_int: раз в CHECK_INTERVAL шагов.
			if ((i % CHECK_INTERVAL) == 0) {
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
				// Отмена отдаёт тот же sentinel, что и разлёт: результат
				// отменённого прогона хост выбрасывает целиком.
				if (cancelFlag != nullptr && *cancelFlag != 0) return fsDivergedError();
			}
			calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h, 1);

			// Разлёт master'а на пред-проходе: дальше в fwd/bwd циклах Xm
			// читается из data, так что ловить его надо здесь.
			if (fsStateDiverged(Xm, amountOfX, maxValue))
				return fsDivergedError();

			for (int w = 0; w < amountOfX; w++)
				data[startDataIndex + w + i * amountOfX] = Xm[w];
		}
	}

	// error_estim 7 — Беннеттин поверх того же цикла вперёд-назад; окно мастера
	// к этому моменту уже залито. Только unidirectional: при bidir мастер сам
	// подтягивается к слейву, и ρ означала бы не ту величину (см. комментарий
	// у fsBenettinCycleExponent).
	if (error_estim == 7) {
		if (type_of_synch != 0 || data == nullptr)
			return fsDivergedError();
		return fsBenettinCycleExponent(data + startDataIndex, values, h, amountOfIterations,
			amountOfX, maxValue, iterOfSynchr, kForward, kBackward, icEps, gsWarmup);
	}

	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = kForward[j];

		// Глобальный цикл, который производит вычисления заданные amountOfIterations раз
		for (int i = 0; i < amountOfIterations - 1; ++i) {

			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j) {
					Xm[j] = data[startDataIndex + i * amountOfX + j];
					//Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
				}
			}

			if (error_estim == 0) {
				if (m == iterOfSynchr - 1) {
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 1) {
				if (i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 2) {
				if (m == iterOfSynchr - 1 && i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 3) {
				rms_error = 0;
				for (int j = 0; j < amountOfX; ++j)
					rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				rms_error = (sqrt(rms_error));
				if (rms_error <= FS_error_trs) {
					return (numb)(i + 1) * h;
				}

			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 1);

			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 1);
			}

			// При type_of_synch == 0 Xm берётся из уже проверенного data —
			// достаточно смотреть на slave; при bidir разлететься может и master.
			if (fsStateDiverged(Xs, amountOfX, maxValue) ||
				(type_of_synch == 1 && fsStateDiverged(Xm, amountOfX, maxValue)))
				return fsDivergedError();

		}

		if (error_estim == 1) {
			rms_error = (sqrt(rms_error));
			if (rms_error <= FS_error_trs) {
				//delete[] Xm;
				//delete[] Xs;
				//delete[] X_prev;
				//delete[] K_local;
				//delete[] norm_error;

				return (numb)(m + 1);
			}
			else
				rms_error = 0;
		}

		for (int j = 0; j < amountOfX; ++j)
			K_local[j] = kBackward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = data[startDataIndex + i * amountOfX + j];
			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 0);

			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 0);
			}

			if (fsStateDiverged(Xs, amountOfX, maxValue) ||
				(type_of_synch == 1 && fsStateDiverged(Xm, amountOfX, maxValue)))
				return fsDivergedError();
		}
	}

	// err_stop — та же стартовая точка окна, но после всех проходов.
	// Опорная точка мастера: при unidir мастер задан заранее и не менялся —
	// берём x[] (входные НУ мастера, loop_2 их не трогает; читать то же самое из
	// data[startDataIndex] было бы лишней зависимостью от ненулевого буфера).
	// При bidir мастер тоже интегрировался и последним обратным проходом
	// вернулся в начало окна — там берём текущий Xm.
	//
	// Деление на ноль НЕ глушим: err_start == 0 (slave стартовал ровно из точки
	// мастера) даёт inf либо NaN, и это честное «не измерено», а не нулевая
	// ошибка. Хост выбрасывает non-finite из min/max, такие ячейки красятся
	// тёмно-серым, как и разлетевшиеся.
	if (error_estim == 4 || error_estim == 5 || error_estim == 6) {
		numb err_stop = 0;
		for (int j = 0; j < amountOfX; ++j) {
			numb e = (type_of_synch == 1 ? Xm[j] : x[j]) - Xs[j];
			err_stop += e * e;
		}
		err_stop = sqrt(err_stop);
		const numb ratio = err_stop / err_start;
		if (error_estim == 4)
			return ratio;
		const numb lg = log10(ratio);
		// 6 — та же величина, делённая на число циклов: log10(ratio) растёт как
		// m*log10 ρ, и только после деления на m получается скорость ЗА ЦИКЛ,
		// сравнимая с log10 ρ из линеаризованного расчёта.
		return (error_estim == 5) ? lg : lg / (numb)iterOfSynchr;
	}

	if (error_estim == 0)
		rms_error = sqrt(rms_error / (numb)(amountOfIterations - 1));

	if (error_estim == 2)
		rms_error = sqrt(rms_error);

	if (rms_error <= FS_error_trs)
		rms_error = log10(FS_error_trs);
	else
		rms_error = log10(rms_error);

	if (isinf(rms_error) || isnan(rms_error) || rms_error >= 3)
		rms_error = 3;

	//delete[] Xm;
	//delete[] Xs;
	//delete[] X_prev;
	//delete[] K_local;
	//delete[] norm_error;

	if (error_estim == 0)
		return rms_error;
	if (error_estim == 1)
		return (numb)iterOfSynchr;
	if (error_estim == 2)
		return rms_error;
	if (error_estim == 3)
		return (numb)amountOfIterations * h;
	return fsDivergedError();   // недостижимо: все error_estim разобраны выше
}

// ---------------------------------------------------------------------------
// PeakStream -- sample-by-sample equivalent of peakFinder().
//
// peakFinder() needs the whole trajectory resident in global memory, and THAT
// buffer is what caps how many sweep points fit on the GPU at once (see the
// budget in run_bif2d). PeakStream consumes the same samples as the integrator
// produces them, so the trajectory never has to be stored.
//
// Equivalence with the batch version is the whole point -- the diagram must not
// move when the buffer goes away -- so the correspondence is spelled out here:
//
//   batch                                   stream
//   outer loop index i (from 2, i < L-2)    j_or_i = n-1, gated by nextI
//   inner plateau walk over j               `walking` state, same window test
//   i = j + 1; break;  then ++i             nextI = j + 2
//   second pass: inter-peak interval filter emitInterval(), anchor carried live
//
// Index convention is the batch one: samples numbered from 0 inside the block,
// a peak at sample j is reported at time (j - 1), samples 0..1 are history only,
// and the last sample is never examined (the batch bound is i < L-2 and it
// reads x[j+1]).
//
// The one batch path with no stream analogue is an inner loop that runs off the
// end of the block: there the batch resumes its outer loop from the stale i.
// That cannot produce another peak -- exhaustion means the tail is non-increasing
// with steps <= eps_peak_delta, so no later i can satisfy x[i]-x[i-1] > eps --
// so the stream just stops.
struct PeakStream
{
	numb*  outPeaks;
	numb*  timeOfPeaks;
	size_t peakBase;
	numb   sampleStep;    // time between consecutive samples (preScaller included)
	size_t scanLen;       // block length the batch version would have scanned
	int    capacity;      // cap on RAW peaks; 0 = uncapped (peakFinder's peakCapacity)

	numb   w0, w1, w2;    // rolling window x[n-2], x[n-1], x[n]
	size_t n;             // samples consumed
	size_t nextI;         // smallest outer index still to be tested
	bool   walking;       // inside a plateau walk
	bool   done;          // capacity reached or tail exhausted
	int    raw;           // raw peaks seen
	int    emitted;       // peaks written after the interval filter
	numb   anchorTime;    // time (in samples) of the current anchor peak
	bool   haveAnchor;

	// Running sums over emitted peaks and intervals. Every basin feature
	// (avg / rms / stdev) is a function of these, so that path needs no peak
	// arrays at all -- and `last` is the one sample it still needs, for the
	// fixed-point case avgPeakFinderCUDA read from the end of the trajectory.
	numb   sumP, sumP2, sumI, sumI2;
	numb   last;

	// Emit every sample instead of peaks. A discrete map's bifurcation diagram
	// plots each iterate x_n, not the local maxima of the sequence: the peak
	// filter would drop the lower branch of every period-2 orbit and clip the
	// bottom of the chaotic band. Feeds pushRaw() instead of push().
	bool   emitAll;

	__device__ __host__ void init(numb* peaks, numb* times, size_t base,
		numb step, size_t len, int cap, bool all = false)
	{
		outPeaks = peaks; timeOfPeaks = times; peakBase = base;
		sampleStep = step; scanLen = len; capacity = cap;
		emitAll = all || !doCalculatePeaks;
		w0 = w1 = w2 = (numb)0;
		n = 0; nextI = 2; walking = false; done = false;
		raw = 0; emitted = 0; anchorTime = (numb)0; haveAnchor = false;
		sumP = sumP2 = sumI = sumI2 = (numb)0;
		last = (numb)0;
	}

	// Inter-peak interval filter, live. The batch version anchors on the FIRST
	// raw peak and never emits it, measuring every interval from the last
	// emitted peak -- hence haveAnchor.
	__device__ __host__ void emitInterval(numb value, numb time)
	{
		if (!haveAnchor) { anchorTime = time; haveAnchor = true; return; }

		const numb delta = (time - anchorTime) * sampleStep;
		if (delta < eps_interPeak_delta) return;

		// The batch version clamps its count to max_amount_of_peaks at the very
		// end; capping the writes here is the same thing for every consumer,
		// which only ever reads the first `count` entries.
		if (emitted < max_amount_of_peaks) {
			if (outPeaks != nullptr)    outPeaks[peakBase + emitted]    = value;
			if (timeOfPeaks != nullptr) timeOfPeaks[peakBase + emitted] = delta;
			// Same cap the batch path applied: avgPeakFinderCUDA summed exactly the
			// peaks that made it into the array and into the clamped count.
			sumP += value;   sumP2 += value * value;
			sumI += delta;   sumI2 += delta * delta;
		}
		++emitted;
		anchorTime = time;
	}

	// Peak confirmed at sample j, window (w0, w1, w2) = x[j-1], x[j], x[j+1].
	__device__ __host__ void recordPeak(size_t j)
	{
		numb value, time;
		if (doInterpolatePeaks) {
			const numb denom = w0 - (numb)2.0 * w1 + w2;
			numb delta = 0.0;
			if (fabs(denom) > 1e-12)
				delta = 0.5 * (w0 - w2) / denom;
			value = w1 - 0.25 * (w0 - w2) * delta;
			time  = (numb)(j - 1) + delta;
		}
		else {
			value = w1;
			time  = (numb)(j - 1);
		}

		emitInterval(value, time);
		++raw;
		// Batch checks the cap at the top of the outer loop, i.e. after the
		// record -- so raw peaks never exceed peakCapacity.
		if (capacity > 0 && raw >= capacity) done = true;
	}

	// Feed one trajectory sample.
	__device__ __host__ void push(numb x)
	{
		last = x;   // before the done check: the scan stops short of the block end
		if (done) return;

		w0 = w1; w1 = w2; w2 = x;
		const size_t idxSample = n;   // index of the sample just pushed
		++n;
		if (idxSample < 2) return;    // history only; batch starts at i = 2

		const size_t j = idxSample - 1;          // outer i, or the walk's j
		if (scanLen < 3 || j + 2 >= scanLen) { done = true; return; }  // batch bound: j < L-2

		if (!walking) {
			if (j < nextI) return;
			if (!(w1 - w0 > eps_peak_delta && w1 > peak_threshold && w1 >= w2))
				return;
			walking = true;   // batch enters the inner loop with j = i
		}

		// Inner-loop body at j.
		if (w1 < w2) {                       // rise -> it was not a peak
			walking = false; nextI = j + 2; return;
		}
		if (w1 - w2 > eps_peak_delta) {      // fall -> peak at j
			recordPeak(j);
			walking = false; nextI = j + 2; return;
		}
		// plateau: keep walking
	}

	// Value peakFinder() would have returned.
	__device__ __host__ int count() const
	{
		return (emitted >= max_amount_of_peaks) ? max_amount_of_peaks : emitted;
	}

	// doCalculatePeaks == 0: the batch version copies raw samples instead of
	// peaks and returns count - 1. Kept here so the fused kernel covers both
	// settings of the knob (the engine injects it as a #define).
	__device__ __host__ void pushRaw(numb x)
	{
		int limit = (int)((scanLen < (size_t)max_amount_of_peaks) ? scanLen : (size_t)max_amount_of_peaks);
		if (capacity > 0 && limit > capacity) limit = capacity;
		if (raw < limit) {
			if (outPeaks != nullptr)    outPeaks[peakBase + raw]    = x;
			if (timeOfPeaks != nullptr) timeOfPeaks[peakBase + raw] = 0;
			++raw;
		}
		emitted = raw;
	}

	__device__ __host__ int countRaw() const { return raw - 1; }
};
