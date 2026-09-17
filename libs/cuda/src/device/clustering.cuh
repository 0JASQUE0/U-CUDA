__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData,
	const numb multPeak, const numb multInterval, const size_t helpfulStartIndex)
{
	// helpfulArray may have its own per-thread stride: the peak buffers are
	// sized by max_amount_of_peaks in the 2D path, the scratch needs twice that
	// (labels + traversal stack). (size_t)-1 = legacy, shares startDataIndex.
	const size_t hBase = (helpfulStartIndex == (size_t)-1) ? startDataIndex : helpfulStartIndex;
	// Если пиков 0 или 1 - даже не обрабатываем эти случаи

	if (amountOfPeaks == -1)
		return -1;

	if (amountOfPeaks == 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	//if (amountOfPeaks >= 3600)
	//	return 0;

	int cluster = 0;

	for (size_t i = hBase; i < hBase + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	//for (int i = 0; i < amountOfPeaks; i++) {
	//	helpfulArray[startDataIndex + i] = (int)(100*sqrt(data[startDataIndex + i] * data[startDataIndex + i] + intervals[startDataIndex + i] * intervals[startDataIndex + i]));
	//}

	// Масштабирование осей признаков перед кластеризацией. Раньше здесь стояли
	// constexpr mult_peak / mult_interval из configCUDA.h — теперь множители
	// приходят аргументами, чтобы каждая 2D-диаграмма могла настроить их сама
	// (дефолты в объявлении dbscanCUDA — те же константы, поэтому legacy-путь
	// hostLibrary.cu ведёт себя как прежде).
	for (size_t i = 0; i < amountOfPeaks; i++) {
		data[startDataIndex + i] = data[startDataIndex + i] * multPeak;
		intervals[startDataIndex + i] = intervals[startDataIndex + i] * multInterval;
	}

	// Ёмкость стека обхода. Пока инвариант про две половины helpfulArray
	// выполняется, sp < stackCap истинно всегда и проверка ничего не меняет;
	// если инвариант когда-нибудь сломается, мы потеряем соседа вместо записи
	// в ячейку соседней системы.
	const int stackCap = sizeOfHelpfulArray - amountOfPeaks;

	// Обход: каждая непосещённая точка заводит кластер и разворачивается до
	// опустошения стека.
	//
	// Раньше здесь был ОДИН цикл на amountOfPeaks итераций, а стек писался по
	// индексу точки (helpfulArray[... + amountOfPeaks + k] = k), но читался по
	// глубине (... + amountOfPeaks + NumNeibor - 1): клали в одну ячейку, а
	// доставали из другой. Совпадало это лишь случайно, обход обычно вычитывал
	// ноль и топтался по точке 0, часть соседей не разворачивалась никогда, а
	// точку k = 0 нельзя было поставить в очередь в принципе — её "0" в стеке
	// неотличим от пустой ячейки. Пока пиков мало, расхождение маскировалось;
	// на режимах с большим их числом количество кластеров выходило произвольным.
	for (int i = 0; i < amountOfPeaks; i++) {
		if (helpfulArray[hBase + i] != 0) continue;   // уже в кластере

		++cluster;
		helpfulArray[hBase + i] = cluster;

		int sp  = 0;    // глубина стека
		int cur = i;    // точка, соседей которой разворачиваем
		for (;;) {
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (cur == k || helpfulArray[hBase + k] != 0) continue;
				if (distance(data[startDataIndex + cur], intervals[startDataIndex + cur],
					data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
					helpfulArray[hBase + k] = cluster;
					if (sp < stackCap)
						helpfulArray[hBase + amountOfPeaks + sp++] = (numb)k;
				}
			}
			if (sp == 0) break;
			cur = (int)helpfulArray[hBase + amountOfPeaks + --sp];
		}
	}

	// Последний пик из внутренних циклов исключён (k < amountOfPeaks - 1), поэтому
	// соседом он не становится никогда и на последней итерации всегда заводит
	// собственный кластер — его и вычитает "- 1". Оставлено как было: это вопрос
	// к самому алгоритму, а не к очереди. Зато теперь компенсация ТОЧНА: раньше
	// она была верна лишь тогда, когда обход к этому моменту успевал опустеть.
	return cluster - 1;
}

// Глобальная функция DBSCAN

__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray,
	const numb eps, int* outData,
	const numb multPeak, const numb multInterval,
	const size_t peakStride, const size_t helpfulStride)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Коды режима (REGIME_*) проносим в outData как есть: на 2D-карте -1
	// означает fixed point, 0 — unbound, N > 0 — период (число кластеров).
	if (amountOfPeaks[idx] == REGIME_FIXED_POINT)
	{
		outData[idx] = REGIME_FIXED_POINT;
		return;
	}

	if (amountOfPeaks[idx] == REGIME_UNBOUND)
	{
		outData[idx] = REGIME_UNBOUND;
		return;
	}

	// --- Применяем алгоритм dbscan к каждой системе
	// Strides of 0 mean "peaks live in the trajectory buffers" (legacy).
	const size_t pStride = (peakStride    != 0) ? peakStride    : sizeOfBlock;
	const size_t hStride = (helpfulStride != 0) ? helpfulStride : sizeOfBlock;

	outData[idx] = dbscan(data, intervals, helpfulArray, idx * pStride, amountOfPeaks[idx], (int)hStride, idx, eps, outData, multPeak, multInterval, idx * hStride);
}

// ПРИМЕЧАНИЕ: здесь были dbscan_optimized (Spatial Hashing + Stack DFS) и
// ядро dbscanCUDA_optimized. Удалены как мёртвый код: единственная ссылка на
// них — закомментированный вызов в hostLibrary.cu, рабочий 2D-путь идёт через
// dbscanCUDA -> dbscan. Именно они держали per-thread массивы
// int next[max_amount_of_peaks] / labels[max_amount_of_peaks], поэтому после
// удаления max_amount_of_peaks больше НЕ бьёт по local memory и occupancy —
// он ограничивает только число пиков в peakFinder и размеры device-буферов.

// Ядро для LLE
// LLEKernelCUDA (Wolf/Benettin) под NVRTC ВКЛЮЧЁН — curand_kernel.h теперь
// доступен из cudaLibrary.cuh. Всё что после этого kernel'а (LLE_IC, LS,
// dbscan, fastSynchro и пр.) остаётся под гардом, потому что не требуется
// для текущих режимов engine'а и может нести host-only зависимости.
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const numb	NT,
	const numb	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb*			ranges,
	const numb	h,
	const numb	eps,
	int*			indicesOfMutVars,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			resultArray,
	const int		hSweepAxis,
	const numb	transientTime,
	const int		logAxisMask,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride)
{
	extern __shared__ numb s[];
	numb* x = s + threadIdx.x * amountOfInitialConditions;
	numb* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nPtsLimiter)
		return;

	// Per-thread h override for a dt-sweep axis; falls back to the uniform
	// launch-wide h otherwise (identical to the pre-dt-sweep behaviour).
	numb h_local = h;
	if (hSweepAxis == 0)
		h_local = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
	else if (hSweepAxis == 1)
		h_local = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);

	// Degenerate/zero step: ceil(NT/h_local) below would divide by zero.
	// Flag the point as unusable instead of crashing/hanging the thread.
	// nan("") (CUDA math API function), not the NAN macro -- NVRTC doesn't
	// pull in <math.h> here, so the macro is undefined even though builtin
	// device functions like sqrt/pow/log (used elsewhere in this file) are.
	if (h_local <= 0) {
		resultArray[idx] = nan("");
		return;
	}

	// tMax/NT (# of NT-blocks) stays host-computed & uniform -- it's h-independent.
	// Only steps-per-block (NT/h) and skip-steps (transient/h) depend on h, so only
	// those are recomputed per-thread when swept, rounded up so we never fall short
	// of transientTime/NT.
	size_t amountOfNTPoints = (hSweepAxis != -1) ? (size_t)ceil(NT / h_local) : (size_t)(NT / h);
	size_t amountOfPointsForSkip_local = (hSweepAxis != -1) ? (size_t)ceil(transientTime / h_local) : (size_t)amountOfPointsForSkip;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

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
			x[indicesOfMutVars[i]] = isLog ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i)
			                                : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 2) {
		// Host never combines mixed param/IC (par_or_var==2) with an h-axis in the
		// same request (see run_lle_2d/run_ls_2d) -- this branch is unaffected.
		x[indicesOfMutVars[0]]           = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
		                                                             : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
		localValues[indicesOfMutVars[1]] = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
		                                                             : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}

	// Тики считаем только по ведущей траектории x: возмущённая y идёт
	// тем же числом шагов, и считать её значило бы удвоить ожидаемый итог.
	// Отмена же передаётся в оба цикла.
	int ticks = 0;
	const size_t totalSteps = amountOfPointsForSkip_local + (size_t)sizeOfBlock * (size_t)amountOfNTPoints;

	int flag = loopCalculateDiscreteModel_int(x, localValues, h_local, amountOfPointsForSkip_local, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
		1, cancelFlag, progressCounter, progressStride, &ticks);

	if (flag == 0) {
		resultArray[idx] = 999;
		ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
		return;
	}

	//if (flag == -1) {
	//	resultArray[idx] = -999;
	//	return;
	//}

	// Общий seed для всех потоков + номер подпоследовательности = idx.
	// idx как seed давал коррелирующие потоки у соседних точек сетки —
	// curand гарантирует независимость подпоследовательностей при общем seed,
	// а не при разных seed'ах (см. NVIDIA curand guide, "Choosing Seed values").
	const unsigned long long ppSeed = 1234567891ULL;
	curandState_t state;

	curand_init(ppSeed, (unsigned long long)idx, 0, &state);

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0;
	}

	numb zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = curand_uniform(&state) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		//bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		//if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		//flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		//if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }
#ifdef UC_ADAPTIVE_ENSEMBLE
        if ((cancelFlag && *cancelFlag) || !uc_adaptive_ensemble(x,y,1,localValues,h_local,amountOfNTPoints,maxValue)) {
            resultArray[idx] = 999;
            ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
            return;

        }
#else
		flag = loopCalculateDiscreteModel_int(x, localValues, h_local, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
			1, cancelFlag, progressCounter, progressStride, &ticks);

		if (flag == 0) {
			resultArray[idx] = 999;
			ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
			return;
		}

		flag = loopCalculateDiscreteModel_int(y, localValues, h_local, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
			1, cancelFlag, nullptr, 0, nullptr);

		if (flag == 0) {
			resultArray[idx] = 999;
			ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
			return;
		}

#endif

		numb tempData = 0;
		numb tempData2 = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l) {
			tempData2 = (numb)1.0 / eps;
			tempData2 = tempData2 * (x[l] - y[l]);
			tempData += tempData2 * tempData2;

			//tempData2 =  (x[l] - y[l])* ((numb)1.0 / eps);
			//tempData += tempData2 * tempData2;
		}

		tempData = sqrt(tempData);

		if (tempData <= 1e-14) {
			//resultArray[idx] = -99999999;
			//return;
			tempData = 1e-14;
		}

		result += log(tempData);

		//if (tempData != 0)
		//	tempData = (1 / tempData);
		//else
		//	tempData = eps;
		tempData = (1 / tempData);
		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (numb)(x[j] - ((x[j] - y[j]  + 1e-14) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
	ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
}

// Всё что ниже остаётся отрезано от NVRTC — engine эти kernel'ы не зовёт,
// а часть из них содержит host-only зависимости или ещё не верифицирована
// под NVRTC. По мере необходимости (LS, dbscan и др.) — снимать гард точечно.
#ifndef __CUDACC_RTC__

// Ядро для LLE (IC)
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const numb	NT,
	const numb	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb*			ranges,
	const numb	h,
	const numb	eps,
	int*			indicesOfMutVars,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			resultArray)
{
	extern __shared__ numb s[];
	numb* x = s + threadIdx.x * amountOfInitialConditions;
	numb* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	numb zPower = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0;
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		curand_init(seed + i, 0, 0, &state);
		z[i] = curand_uniform(&state) - 0.5;
		//z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}

	loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel_int(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		numb tempData = 0;
		numb tempData2 = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l) {
			tempData2 = 1 / eps;
			tempData2 = tempData2 * (x[l] - y[l]);
			tempData += tempData2 * tempData2;
		}

		tempData = sqrt(tempData);

		//for (int l = 0; l < amountOfInitialConditions; ++l)
		//	tempData += (x[l] - y[l]) * (x[l] - y[l]);
		//tempData = sqrt(tempData) / eps;

		result += log(tempData);

		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (numb)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}

// projectionOperator + gramSchmidtProcess + LSKernelCUDA — выведены из
// NVRTC-гарда (нужны для LS). curand_kernel.h теперь доступен под NVRTC.
#endif // __CUDACC_RTC__  (закрытие гарда, открытого выше для LLE_IC)

//find projection operation (ab)
__device__ __host__ void projectionOperator(numb* a, numb* b, numb* minuend, int amountOfValues)
{
	numb numerator = 0;
	numb denominator = 0;
	for (int i = 0; i < amountOfValues; ++i)
	{
		numerator += a[i] * b[i];
		denominator += b[i] * b[i];
	}

	numb fraction = denominator == 0 ? 0 : numerator / denominator;

	for (int i = 0; i < amountOfValues; ++i)
		minuend[i] -= fraction * b[i];
}

__device__ __host__ void gramSchmidtProcess(numb* a, numb* b, int amountOfVectorsAndValuesInVector, numb* denominators=nullptr/*They are is equale for our task*/)
{
	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[j + i * amountOfVectorsAndValuesInVector] = a[j + i * amountOfVectorsAndValuesInVector];

		for (int j = 0; j < i; ++j)
			projectionOperator(a + i * amountOfVectorsAndValuesInVector,
				b + j * amountOfVectorsAndValuesInVector,
				b + i * amountOfVectorsAndValuesInVector,
				amountOfVectorsAndValuesInVector);
	}

	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		numb denominator = 0;
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			denominator += b[i * amountOfVectorsAndValuesInVector + j] * b[i * amountOfVectorsAndValuesInVector + j];
		denominator = sqrt(denominator);
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[i * amountOfVectorsAndValuesInVector + j] = denominator == 0 ? 0 : b[i * amountOfVectorsAndValuesInVector + j] / denominator;

		if (denominators != nullptr)
			denominators[i] = denominator;
	}
}

// error_estim 7 — показатель сжатия ошибки за один цикл вперёд-назад,
// посчитанный схемой Беннеттина поверх ТОГО ЖЕ forward-backward оператора,
// что и оценщики 4..6. Возвращает log10 ρ за цикл, где ρ — спектральный
// радиус оператора цикла; это ровно та величина, которую даёт линеаризованный
// расчёт через матрицу монодромии (Phi_b*Phi_f) и max|eig|.
//
// Зачем отдельная схема, если 6 тоже делит на число циклов: там мерится
// усиление ОДНОГО вектора, которое зависит от его направления (при малом числе
// циклов карта идёт крапом) и упирается либо в нелинейное насыщение при ρ>1,
// либо в машинный ноль при ρ<1. Здесь возмущение ренормируется на eps после
// КАЖДОГО цикла, так что расчёт всё время идёт в линейном режиме, а базис
// ортонормируется — направление сходится к доминирующему за пару циклов.
//
// Опорная орбита — мастер (вдоль неё и линеаризуют), клоны — AMOUNTOFX слейвов,
// стартующих из Xm(t0) + eps*e_k и подтягиваемых к мастеру теми же K, что и
// обычный слейв. Якобиан не нужен: это та же клон-схема, что в LSKernelCUDA.
//
// denominators[] после GS несут весь спектр |λ_k| оператора цикла; наружу
// отдаём только ведущий (denominators[0]) — если понадобится остальной спектр,
// он тут уже посчитан.
//
// ТОЛЬКО unidirectional: при bidir мастер сам подтягивается к слейву, оператор
// действует на совместном состоянии (2N измерений), и ρ означала бы уже не то,
// что считает линеаризация вдоль свободной мастер-орбиты. Вызывающий код в этом
// случае сюда не заходит.
__device__ numb fsBenettinCycleExponent(
	const numb* masterWindow,
	const numb* values,
	const numb h,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	const int iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	const numb eps,
	const int gsWarmup)
{
	if (!(eps > 0) || iterOfSynchr <= 0)
		return fsDivergedError();

	// Прогревочные циклы: их логарифмы в среднее не идут. Стартовый базис с
	// доминирующим направлением не совпадает, и первые циклы дают
	// нерепрезентативные коэффициенты; в среднем по ВСЕМ циклам это смещение
	// вымывается лишь как O(1/m), из-за чего сходимость требовала сотни циклов.
	// Отбросив разогрев, получаем сходимость по |λ2/λ1|^m — ориентация базиса
	// сходится геометрически, и вымывать из среднего уже нечего.
	// Клампим, а не отбраковываем: хотя бы один цикл обязан попасть в среднее.
	const int warmup = (gsWarmup <= 0) ? 0
		: ((gsWarmup >= iterOfSynchr) ? iterOfSynchr - 1 : gsWarmup);

	numb y[AMOUNTOFX * AMOUNTOFX];   // клоны в абсолютных координатах
	numb z[AMOUNTOFX * AMOUNTOFX];   // ортонормированный базис после GS
	numb denominators[AMOUNTOFX];
	numb Xm[AMOUNTOFX];
	numb K_local[AMOUNTOFX];

	// Стартовый базис — единичный, а не случайный (как в LS): направления тогда
	// детерминированы, ячейка воспроизводится бит-в-бит и seed не нужен. За
	// первые же циклы GS всё равно развернёт базис под оператор.
	for (int k = 0; k < amountOfX; ++k)
		for (int j = 0; j < amountOfX; ++j)
			y[k * amountOfX + j] = masterWindow[j] + ((k == j) ? eps : (numb)0);

	numb sumLog10 = 0;
	int  counted = 0;

	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; ++j) K_local[j] = kForward[j];
		for (int i = 0; i < amountOfIterations - 1; ++i) {
			for (int j = 0; j < amountOfX; ++j)
				Xm[j] = masterWindow[i * amountOfX + j];
			for (int k = 0; k < amountOfX; ++k)
				calculateDiscreteModelforFastSynchro(y + k * amountOfX, Xm, K_local, values, h, 1);
		}

		for (int j = 0; j < amountOfX; ++j) K_local[j] = kBackward[j];
		for (int i = amountOfIterations - 1; i > 0; --i) {
			for (int j = 0; j < amountOfX; ++j)
				Xm[j] = masterWindow[i * amountOfX + j];
			for (int k = 0; k < amountOfX; ++k)
				calculateDiscreteModelforFastSynchro(y + k * amountOfX, Xm, K_local, values, h, 0);
		}

		// Разлёт проверяем раз в цикл, а не на каждом шаге: ренормировка держит
		// клоны в eps-окрестности мастера, так что вылет за maxValue возможен
		// только при уже испорченном окне.
		for (int k = 0; k < amountOfX; ++k)
			if (fsStateDiverged(y + k * amountOfX, amountOfX, maxValue))
				return fsDivergedError();

		// Отклонения от мастера в стартовой точке окна — цикл вернул клоны
		// именно туда (последний проход обратный).
		for (int k = 0; k < amountOfX; ++k)
			for (int j = 0; j < amountOfX; ++j)
				y[k * amountOfX + j] -= masterWindow[j];

		gramSchmidtProcess(y, z, amountOfX, denominators);

		if (!(denominators[0] > 0))
			return fsDivergedError();
		if (m >= warmup) {
			sumLog10 += log10(denominators[0] / eps);
			++counted;
		}

		for (int k = 0; k < amountOfX; ++k)
			for (int j = 0; j < amountOfX; ++j)
				y[k * amountOfX + j] = masterWindow[j] + z[k * amountOfX + j] * eps;
	}

	return sumLog10 / (numb)counted;
}

__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const numb NT,
	const numb tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const size_t amountOfPointsForSkip,
	const int dimension,
	numb* ranges,
	const numb h,
	const numb eps,
	int* indicesOfMutVars,
	numb* initialConditions,
	const int amountOfInitialConditions,
	const numb* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller,
	const int writableVar,
	const numb maxValue,
	numb* resultArray,
	const int hSweepAxis,
	const numb transientTime,
	const int logAxisMask,
	const volatile int* cancelFlag,
	int* progressCounter,
	const int progressStride)
{
	extern __shared__ numb s[];

	unsigned long long buferForMem = 0;
	numb* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	numb* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= nPtsLimiter)
		return;

	// Per-thread h override for a dt-sweep axis; falls back to the uniform
	// launch-wide h otherwise (identical to the pre-dt-sweep behaviour).
	numb h_local = h;
	if (hSweepAxis == 0)
		h_local = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
	else if (hSweepAxis == 1)
		h_local = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
		                                    : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);

	// Degenerate/zero step -- see LLEKernelCUDA above. LS writes N doubles/thread.
	if (h_local <= 0) {
		for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = nan("");
		return;
	}

	// See LLEKernelCUDA above for the tMax/NT vs NT/h reasoning.
	size_t amountOfNTPoints = (hSweepAxis != -1) ? (size_t)ceil(NT / h_local) : (size_t)(NT / h);
	size_t amountOfPointsForSkip_local = (hSweepAxis != -1) ? (size_t)ceil(transientTime / h_local) : (size_t)amountOfPointsForSkip;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

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
			x[indicesOfMutVars[i]] = isLog ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i)
			                                : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 2) {
			// Host never combines mixed param/IC (par_or_var==2) with an h-axis in
			// the same request (see run_ls_2d) -- this branch is unaffected.
			x[indicesOfMutVars[0]]			  = ((logAxisMask >> 0) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0)
			                                                              : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
			localValues[indicesOfMutVars[1]] = ((logAxisMask >> 1) & 1) ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1)
			                                                              : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}

	// Общий seed для всех потоков + номер подпоследовательности = idx (см.
	// LLEKernelCUDA выше — тот же аргумент против idx-как-seed).
	const unsigned long long ppSeed = 1234567891ULL;
	curandState_t state;

	curand_init(ppSeed, (unsigned long long)idx, 0, &state);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		numb zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			z[j * amountOfInitialConditions + i] = curand_uniform(&state) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}

	// Тики — только по ведущей траектории (как в LLEKernelCUDA): N копий
	// делают столько же шагов каждая, и счёт по ним раздул бы ожидаемый итог.
	int ticks = 0;
	const size_t totalSteps = amountOfPointsForSkip_local + (size_t)sizeOfBlock * amountOfNTPoints;

	int flag = loopCalculateDiscreteModel_int(x, localValues, h_local, amountOfPointsForSkip_local, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
		1, cancelFlag, progressCounter, progressStride, &ticks);

	if (flag == 0) {
		for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
		ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
		return;
	}

	//Calculating

	gramSchmidtProcess(z, y, amountOfInitialConditions);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
#ifdef UC_ADAPTIVE_ENSEMBLE
        if ((cancelFlag && *cancelFlag) || !uc_adaptive_ensemble(x,y,AMOUNTOFX,localValues,h_local,amountOfNTPoints,maxValue)) {
            for(int m=0;m<amountOfInitialConditions;++m) resultArray[idx*amountOfInitialConditions+m]=999;
            ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
            return;
        }
#else
		flag = loopCalculateDiscreteModel_int(x, localValues, h_local, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
			1, cancelFlag, progressCounter, progressStride, &ticks);
		if (flag == 0) {
			for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
			ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
			return;
		}
		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel_int(y + j * amountOfInitialConditions, localValues, h_local, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock,
				1, cancelFlag, nullptr, 0, nullptr);
			if (flag == 0) {
				for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
				ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
				return;
			}
		}

#endif

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (numb)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
	ucudaProgressTopUp(progressCounter, progressStride, totalSteps, ticks);
}

// Всё что ниже снова под гардом — engine не зовёт, потенциально хост-only.
#ifndef __CUDACC_RTC__

__global__ void LSKernelICCUDA(
	const int nPts,
	const int nPtsLimiter,
	const numb NT,
	const numb tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const size_t amountOfPointsForSkip,
	const int dimension,
	numb* ranges,
	const numb h,
	const numb eps,
	int* indicesOfMutVars,
	numb* initialConditions,
	const int amountOfInitialConditions,
	const numb* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller,
	const int writableVar,
	const numb maxValue,
	numb* resultArray)
{
	extern __shared__ numb s[];

	unsigned long long buferForMem = 0;
	numb* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	numb* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		numb zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			curand_init(seed + j * amountOfInitialConditions + i, 0, 0, &state);
			z[j * amountOfInitialConditions + i] = curand_uniform(&state) - 0.5;
			//z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}

	loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	gramSchmidtProcess(z, y, amountOfInitialConditions);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error;*/ }

		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel_int(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints,
				amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error; */ }
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (numb)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}

// Нахождение среднего значения пиков и межпиковых интервалов

__global__ void avgPeakFinderCUDA_logMaximas(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// systemCheker — REGIME_* из configCUDA.h (1 = Osc, -1 = FP, 0 = Unbound).
	if (systemCheker[idx] == REGIME_UNBOUND)
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}

	if (systemCheker[idx] == REGIME_FIXED_POINT)
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock-1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0) //No peaks
	{
		outAvgPeaks[idx] = -999;
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += (outPeaks[idx * sizeOfBlock + i]);
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}

	AvgTimeOfPeaks[idx] = AvgTimeOfPeaks[idx] * mult_interval;
	outAvgPeaks[idx] = outAvgPeaks[idx] * mult_peak;

	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	if (outAvgPeaks[idx] > 0)
		outAvgPeaks[idx] = log10(outAvgPeaks[idx] / amountOfPeaks);
	else if (outAvgPeaks[idx] < 0)
		outAvgPeaks[idx] = -log10(abs(outAvgPeaks[idx]) / amountOfPeaks);
	else
		outAvgPeaks[idx] = 0;

	return;
}

// Открываем доступ под NVRTC для basins-pipeline (U-CUDA)
// Группа kernel'ов до CUDA_dbscan_search_unbound_points_kernel включительно
// нужна U-CUDA::run_basins. До правки весь блок был внутри
// `#ifndef __CUDACC_RTC__` (LLE/LS-зависимости), и NVRTC не находил символы
// в PTX (cuModuleGetFunction → "named symbol not found"). Здесь временно
// закрываем guard, оставляя последующие FastSynchro-kernels по-прежнему
// hidden под NVRTC (они тянут зависимости, NonLinAnal оставил их host-only).
#endif // __CUDACC_RTC__ — re-opened below after basins kernels

// Возвращает 4 флага: нужны ли (sum_p, sum_p², sum_i, sum_i²) для feat. Логика
// зеркалит compute_basin_feature: AVG→sum, RMS→sum², STDEV→оба. feature1/2 в
// kernel'е — uniform по warp'у, поэтому ветви компилируются в константы и
// dead code выпиливается.
__device__ __host__ __forceinline__
void feature_needs(int feat, bool& sp, bool& sp2, bool& si, bool& si2)
{
	sp = sp2 = si = si2 = false;
	switch (feat) {
		case BF_AVG_PEAKS:           case BF_LOG_AVG_PEAKS:        sp  = true;             break;
		case BF_AVG_INTERVALS:       case BF_LOG_AVG_INTERVALS:    si  = true;             break;
		case BF_RMS_PEAKS:           case BF_LOG_RMS_PEAKS:        sp2 = true;             break;
		case BF_RMS_INTERVALS:       case BF_LOG_RMS_INTERVALS:    si2 = true;             break;
		case BF_STDEV_PEAKS:         case BF_LOG_STDEV_PEAKS:      sp  = true; sp2 = true; break;
		case BF_STDEV_INTERVALS:     case BF_LOG_STDEV_INTERVALS:  si  = true; si2 = true; break;
		default: break;
	}
}

// Финальный feature dispatch. mean_*/rms_*/std_* должны быть посчитаны, если
// feature_needs пометил соответствующий sum (иначе по контракту приходит 0,
// что для не-выбранной фичи всё равно никогда не используется).
__device__ __host__ __forceinline__
numb compute_basin_feature(int feat,
	numb mean_p, numb mean_i, numb rms_p, numb rms_i, numb std_p, numb std_i)
{
	switch (feat) {
		case BF_AVG_PEAKS:            return mean_p;
		case BF_AVG_INTERVALS:        return mean_i;
		case BF_RMS_PEAKS:            return rms_p;
		case BF_RMS_INTERVALS:        return rms_i;
		case BF_STDEV_PEAKS:          return std_p;
		case BF_STDEV_INTERVALS:      return std_i;
		case BF_LOG_AVG_PEAKS:        return mean_p == (numb)0 ? (numb)0 : (mean_p > 0 ?  log10(mean_p) : -log10(-mean_p));
		case BF_LOG_AVG_INTERVALS:    return mean_i == (numb)0 ? (numb)0 : (mean_i > 0 ?  log10(mean_i) : -log10(-mean_i));
		case BF_LOG_RMS_PEAKS:        return rms_p > 0 ? log10(rms_p) : (numb)-999;
		case BF_LOG_RMS_INTERVALS:    return rms_i > 0 ? log10(rms_i) : (numb)-999;
		case BF_LOG_STDEV_PEAKS:      return std_p > 0 ? log10(std_p) : (numb)-999;
		case BF_LOG_STDEV_INTERVALS:  return std_i > 0 ? log10(std_i) : (numb)-999;
		default:                      return (numb)-999;
	}
}

// calculateDiscreteModelAvgPeaksCUDA -- calculateDiscreteModelCUDA fused with
// avgPeakFinderCUDA for basins of attraction. Same sweep point (shared
// ucudaSetupSweepPoint), same integrator, same REGIME_* semantics and the same
// sentinel values on the output.
//
// Basins asked for two buffers the length of the trajectory per cell (data and
// intervals) and returned two numbers from them. Every feature is a function of
// the running sums PeakStream already keeps, so neither buffer is needed: the
// cell integrates, accumulates, and writes its two features.
//
// hSweepAxis / logAxisMask are fixed at -1 / 0: BasinsRequest has no such
// fields, and run_basins passed exactly these constants to the split path.
__global__ void calculateDiscreteModelAvgPeaksCUDA(
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
	numb*			outAvgPeaks,
	numb*			AvgTimeOfPeaks,
	int*			systemCheker,
	const bool		Par_or_Var,
	const numb		transientTime,
	const numb		tMax,
	const int		feature1,
	const int		feature2,
	const numb		mult1,
	const numb		mult2,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + ( threadIdx.x * sharedStride );
	numb* localValues = localX + amountOfInitialConditions;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	int    ticks = 0;
	numb   h_local;
	size_t skip_local;
	size_t iters_local;
	if (!ucudaSetupSweepPoint(nPts, amountOfCalculatedPoints, idx, dimension, ranges, h,
			indicesOfMutVars, initialConditions, amountOfInitialConditions,
			values, amountOfValues, -1, 0,
			transientTime, tMax, preScaller,
			amountOfPointsForSkip, amountOfIterations,
			localX, localValues, h_local, skip_local, iters_local)) {
		if (systemCheker != nullptr) systemCheker[idx] = REGIME_UNBOUND;
		outAvgPeaks[idx]    = 999;
		AvgTimeOfPeaks[idx] = 999;
		if (progressCounter != nullptr && progressStride > 0)
			atomicAdd(progressCounter, (int)((amountOfPointsForSkip + amountOfIterations) / (size_t)progressStride));
		return;
	}

	int flag = loopCalculateDiscreteModel_int(localX, localValues, h_local, skip_local,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, 0, 1,
		cancelFlag, progressCounter, progressStride, &ticks);

	PeakStream peaks;
	peaks.init(nullptr, nullptr, 0, h * (numb)preScaller, iters_local, 0);

	if (flag == REGIME_OSCILLATION || flag == REGIME_FIXED_POINT)
		flag = loopCalculateDiscreteModelPeaks_int(localX, localValues, h_local, iters_local,
			amountOfInitialConditions, preScaller, writableVar, maxValue, peaks,
			cancelFlag, progressCounter, progressStride, &ticks);

	if (systemCheker != nullptr) systemCheker[idx] = flag;

	if (flag == REGIME_UNBOUND) {
		outAvgPeaks[idx]    = 999;
		AvgTimeOfPeaks[idx] = 999;
	}
	else if (flag == REGIME_FIXED_POINT) {
		// Батч-версия брала последний сэмпл траектории; стрим держит его в last.
		outAvgPeaks[idx]    = peaks.last;
		AvgTimeOfPeaks[idx] = -1.0;
	}
	else {
		const int amountOfPeaks = peaks.count();
		if (amountOfPeaks <= 0) {
			outAvgPeaks[idx]    = -999;
			AvgTimeOfPeaks[idx] = -1.0;
		}
		else {
			// Суммы копились в том же порядке, что и в батч-цикле, поэтому
			// результат совпадает побитово. feature_needs здесь не нужен:
			// считать все шесть статистик один раз на ячейку дешевле, чем
			// ветвиться, а неиспользуемые compute_basin_feature игнорирует.
			const numb inv_n  = (numb)1.0 / (numb)amountOfPeaks;
			const numb mean_p = peaks.sumP * inv_n;
			const numb mean_i = peaks.sumI * inv_n;
			const numb rms_p  = sqrt(peaks.sumP2 * inv_n);
			const numb rms_i  = sqrt(peaks.sumI2 * inv_n);
			const numb var_p  = peaks.sumP2 * inv_n - mean_p * mean_p;
			const numb var_i  = peaks.sumI2 * inv_n - mean_i * mean_i;
			const numb std_p  = sqrt(var_p < 0 ? (numb)0 : var_p);
			const numb std_i  = sqrt(var_i < 0 ? (numb)0 : var_i);

			outAvgPeaks[idx]    = mult1 * compute_basin_feature(feature1, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
			AvgTimeOfPeaks[idx] = mult2 * compute_basin_feature(feature2, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
		}
	}

	if (progressCounter != nullptr && progressStride > 0) {
		const size_t totalSteps = skip_local + iters_local;
		const int expected = (int)(totalSteps / (size_t)progressStride);
		if (expected > ticks) atomicAdd(progressCounter, expected - ticks);
	}
}

__global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h,
	int feature1, int feature2, numb mult1, numb mult2)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks) return;

	// systemCheker — REGIME_* из configCUDA.h (1=Osc, -1=FP, 0=Unbound).
	// FP/Unbound — sentinel values, чтобы downstream DBSCAN их распознал;
	// фичи для них не считаем.
	if (systemCheker[idx] == REGIME_UNBOUND) {
		outAvgPeaks[idx]    = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}
	if (systemCheker[idx] == REGIME_FIXED_POINT) {
		outAvgPeaks[idx]    = data[idx * sizeOfBlock + sizeOfBlock - 1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);
	if (amountOfPeaks <= 0) {
		outAvgPeaks[idx]    = -999;
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	// Какие первичные суммы реально нужны для (feature1, feature2). feature1/2
	// одинаковы для всех threads в launch'е → branches uniform, компилятор
	// выкидывает мёртвые ветки. RAW значения (без mult) — масштаб применяем
	// только к финальной фиче в конце.
	bool a_sp, a_sp2, a_si, a_si2, b_sp, b_sp2, b_si, b_si2;
	feature_needs(feature1, a_sp, a_sp2, a_si, a_si2);
	feature_needs(feature2, b_sp, b_sp2, b_si, b_si2);
	const bool need_sp  = a_sp  || b_sp;
	const bool need_sp2 = a_sp2 || b_sp2;
	const bool need_si  = a_si  || b_si;
	const bool need_si2 = a_si2 || b_si2;

	numb sum_p = 0, sum_p2 = 0, sum_i = 0, sum_i2 = 0;
	for (int i = 0; i < amountOfPeaks; ++i) {
		if (need_sp || need_sp2) {
			numb p = outPeaks[idx * sizeOfBlock + i];
			if (need_sp)  sum_p  += p;
			if (need_sp2) sum_p2 += p * p;
		}
		if (need_si || need_si2) {
			numb it = timeOfPeaks[idx * sizeOfBlock + i];
			if (need_si)  sum_i  += it;
			if (need_si2) sum_i2 += it * it;
		}
	}

	numb inv_n = (numb)1.0 / (numb)amountOfPeaks;
	numb mean_p = need_sp  ? sum_p  * inv_n         : (numb)0;
	numb mean_i = need_si  ? sum_i  * inv_n         : (numb)0;
	numb rms_p  = need_sp2 ? sqrt(sum_p2 * inv_n)   : (numb)0;
	numb rms_i  = need_si2 ? sqrt(sum_i2 * inv_n)   : (numb)0;
	// var = E[x²] - E[x]²; floor at 0 чтобы sqrt не упал от округлений.
	numb std_p = 0;
	if (need_sp && need_sp2) {
		numb v = sum_p2 * inv_n - mean_p * mean_p;
		std_p = sqrt(v < 0 ? (numb)0 : v);
	}
	numb std_i = 0;
	if (need_si && need_si2) {
		numb v = sum_i2 * inv_n - mean_i * mean_i;
		std_i = sqrt(v < 0 ? (numb)0 : v);
	}

	outAvgPeaks[idx]    = mult1 * compute_basin_feature(feature1, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
	AvgTimeOfPeaks[idx] = mult2 * compute_basin_feature(feature2, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
}

__global__ void avgPeakFinderCUDA_for2Dbif(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* PeaksAmount, int* systemCheker, numb h)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// systemCheker — REGIME_* из configCUDA.h (1 = Osc, -1 = FP, 0 = Unbound).
	if (systemCheker[idx] == REGIME_UNBOUND)
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		PeaksAmount[idx] = 0;
		return;
	}

	if (systemCheker[idx] == REGIME_FIXED_POINT)
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock - 1];
		AvgTimeOfPeaks[idx] = -1.0;
		PeaksAmount[idx] = 0;
		return;
	}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0)
	{
		outAvgPeaks[idx] = 1000;
		AvgTimeOfPeaks[idx] = 1000;
		PeaksAmount[idx] = 0;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}
	PeaksAmount[idx] = amountOfPeaks;
	outAvgPeaks[idx] /= amountOfPeaks;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	return;
}

// Seed-поиск (search_fixed / search_clear / search_unbound ниже) находит
// ЛЮБУЮ ещё не размеченную точку нужного типа, чтобы начать от неё новый
// кластер. Раньше все подходящие потоки писали `*res = idx` голой записью
// без атомарности — какой именно idx "победит", зависело от порядка
// завершения warp'ов/блоков и мог отличаться между запусками на одних и тех
// же данных. Из-за этого DBSCAN каждый прогон стартовал кластеры в разном
// порядке → одна и та же физическая область бассейна получала другой
// числовой label → heatmap красил её другим цветом, хотя разбиение на
// кластеры (границы бассейнов) оставалось верным и не менялось.
// Фикс: atomicMin вместо голой записи — детерминированно выбираем
// наименьший idx среди подходящих. Sentinel -1 (бит-паттерн 0xFFFFFFFF)
// численно равен UINT_MAX, поэтому конвенция "res == -1 => не найдено"
// переживает unsigned-перетолкование без изменений на host-стороне.
__global__ void CUDA_dbscan_kernel(numb* data, numb* intervals, int* labels,
	const int amountOfData, const numb eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	//if (idx == 0 && labels[idxCurPoint] == 0)					// Если у idxCurPoint точки нет еще кластера - выдаём его (обычно для первой точки в кластере)
	//	labels[idxCurPoint] = atomicAdd(amountOfClusters, 1);

	labels[idxCurPoint] = amountOfClusters;

	if (labels[idx] != 0)									// У точки уже есть кластер - выпиливаемся из потока
		return;

	if (idx == idxCurPoint)									// Если рассматриваем текущую точку - выпиливаемся из потока
		return;

	// Если расстояние между рассматрвиаемой точкой idx и текущей точкой idxCurPoint <= eps - выдаём точке кластер

	// helpfulArray — REGIME_* (configCUDA.h). Unbound-точки кластера не образуют.
	if (helpfulArray[idxCurPoint] == REGIME_UNBOUND) {
		labels[idxCurPoint] = 0;
		return;
	}

	if (sqrt((data[idxCurPoint] - data[idx]) * (data[idxCurPoint] - data[idx]) + (intervals[idxCurPoint] - intervals[idx]) * (intervals[idxCurPoint] - intervals[idx])) <= eps)
	{
		labels[idx] = labels[idxCurPoint];						// Даем точке кластер. Предполагаем, что у idxCurPoint не может не быть кластера
		neighbors[atomicAdd(amountOfNeighbors, 1)] = idx;		// Фиксируем индекс найденного соседа - его ждет та же участь
	}
}

__global__ void CUDA_dbscan_search_clear_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (labels[idx] == 0 && helpfulArray[idx] == REGIME_OSCILLATION)
	{
		atomicMin((unsigned int*)res, (unsigned int)idx);	// детерминированный выбор seed'а, см. комментарий над CUDA_dbscan_kernel
		return;
	}
}

__global__ void CUDA_dbscan_search_fixed_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (helpfulArray[idx] == REGIME_FIXED_POINT && labels[idx] == 0)
	{
		atomicMin((unsigned int*)res, (unsigned int)idx);	// детерминированный выбор seed'а, см. комментарий над CUDA_dbscan_kernel
		return;
	}
}

__global__ void CUDA_dbscan_search_unbound_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (helpfulArray[idx] == REGIME_UNBOUND && labels[idx] == 0)
	{
		atomicMin((unsigned int*)res, (unsigned int)idx);	// детерминированный выбор seed'а, см. комментарий над CUDA_dbscan_kernel
		return;
	}
}

// FastSynchro kernels — теперь ВНУТРИ NVRTC-окна
// Раньше эта группа была спрятана под `#ifndef __CUDACC_RTC__`, но U-CUDA
// run_fastsync (Release-режим через NVRTC) их подгружает по lowered-имени
// (cuModuleGetFunction → "named symbol not found" иначе). NonLinAnal
// (Debug-build) тоже компилирует их через обычный nvcc-путь — пути совместимы.

__global__ void calculateDiscreteModelforFastSynchroCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const numb		h,
	const numb*		initialConditionsSlave,
	const int		amountOfInitialConditions,
	const numb*		values,
	const numb*		k_forward,
	const numb*		k_backward,
	const int		iterOfSynchr,
	const int		amountOfValues,
	const int		amountOfNTPoints,
	const numb		maxValue,
	numb*			timedomain,
	numb*			output,
	const int		preScaller,
	const int		icRandomOffset,
	const numb		icEps,
	const unsigned long long icSeed,
	const int		gsWarmup,
	const volatile int* cancelFlag,
	int* progressCounter)
{

	// Вычисляем индекс потока, в котором находимся в даный момент
	// Тиков здесь нет: на поток приходится один вызов, делить его нечем.
	// Отмена проверяется один раз на входе — для случая, когда флаг
	// подняли пока ядро стояло в очереди за предыдущим.
	if (cancelFlag != nullptr && *cancelFlag != 0) return;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime )
		//numb Xs[3] = { initialConditions[0] , initialConditions[1], initialConditions[2] };

	output[idx] = loopCalculateDiscreteModelForFastSynchro(
		initialConditionsSlave,//numb* Xs,
		values,//const numb* values,
		h, //const numb h,
		k_forward,//const numb* K_Forward,
		k_backward,//const numb* K_Backward,
		iterOfSynchr,//const numb iterOfSynchr,
		amountOfNTPoints,//const int amountOfIterations,
		amountOfInitialConditions,//const int amountOfX,
		maxValue,//const numb maxValue,
		timedomain,//numb* timedomain,
		amountOfInitialConditions * idx * preScaller,//const int startDataIndex
		icRandomOffset,								 //const int icRandomOffset
		icEps,										 //const numb icEps
		icSeed,										 //const unsigned long long icSeed
		(unsigned long long)idx,					 //const unsigned long long icCell
		gsWarmup									 //const int gsWarmup
	,
		cancelFlag, progressCounter);
	return;
}

__device__ numb loopCalculateDiscreteModelForFastSynchro(
	const numb* initConditionsSlave,
	const numb* values,
	const numb h,
	const numb* K_Forward,
	const numb* K_Backward,
	const numb iterOfSynchr,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	numb* timedomain,
	const int startDataIndex,
	const int icRandomOffset,
	const numb icEps,
	const unsigned long long icSeed,
	const unsigned long long icCell,
	const int gsWarmup,
	const volatile int* cancelFlag,
	int* progressCounter)
{
	// error_estim 7 — Беннеттин поверх того же цикла вперёд-назад; окно мастера
	// залито заранее (fillFSMasterTrajectory). Только unidirectional — см.
	// комментарий у fsBenettinCycleExponent.
	if (error_estim == 7) {
		if (type_of_synch != 0)
			return fsDivergedError();
		return fsBenettinCycleExponent(timedomain + startDataIndex, values, h, amountOfIterations,
			amountOfX, maxValue, (int)iterOfSynchr, K_Forward, K_Backward, icEps, gsWarmup);
	}

	//numb* norm_error = new numb[(amountOfIterations - 0)];
	//numb* Xm = new numb[amountOfX];
	//numb* X_prev = new numb[amountOfX];
	//numb* Xs = new numb[amountOfX];
	//numb* K_local = new numb[amountOfX];
	numb Xm[AMOUNTOFX];
	numb X_prev[AMOUNTOFX];
	numb Xs[AMOUNTOFX];
	numb K_local[AMOUNTOFX];

	numb rms_error = 0;
	//numb err0 = 0;
	//numb err1 = 0;

	// icRandomOffset == 0 — legacy: слейв стартует из фиксированных НУ, одних
	// и тех же для всех потоков.
	// icRandomOffset == 1 — слейв стартует от точки мастера в НАЧАЛЕ окна этого
	// потока (timedomain[startDataIndex] — она уже на аттракторе, транзиент
	// отработал fillFSMasterTrajectory) со случайным отступом в кубе
	// [-eps, +eps] по каждой координате. Счётчик ГПСЧ — idx потока, так что
	// отступ свой в каждой точке траектории и воспроизводим по seed'у.
	for (int j = 0; j < amountOfX; ++j) {
		if (type_of_synch == 1) // bidirectional sycnhro
			Xm[j] = timedomain[startDataIndex + j];

		Xs[j] = icRandomOffset
			? timedomain[startDataIndex + j] + icEps * fsRandSymmetric(icSeed, icCell, j)
			: initConditionsSlave[j];
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
	// Мастер в t0 — timedomain[startDataIndex] при любом type_of_synch: при bidir
	// Xm ровно оттуда и проинициализирован парой строк выше.
	numb err_start = 0;
	if (error_estim == 4 || error_estim == 5 || error_estim == 6) {
		for (int j = 0; j < amountOfX; ++j) {
			numb e = timedomain[startDataIndex + j] - Xs[j];
			err_start += e * e;
		}
		err_start = sqrt(err_start);
	}

	for (int m = 0; m < iterOfSynchr; ++m) {
		// Тик — цикл синхронизации, а не шаг интегрирования: время живёт
		// именно в проходах вперёд-назад, а не в заполнении окна.
		if (cancelFlag != nullptr && *cancelFlag != 0) return fsDivergedError();
		if (progressCounter != nullptr) atomicAdd(progressCounter, 1);

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = K_Forward[j];

		// Глобальный цикл, который производит вычисления заданные amountOfIterations раз
		for (int i = 0; i < amountOfIterations - 1; ++i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
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
					return (numb)(i + 1)*h;
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

		}

		if (error_estim == 1) {
			rms_error = (sqrt(rms_error));
			if ( rms_error <= FS_error_trs ) {
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
			K_local[j] = K_Backward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
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

		}
	}

	//for (int j = 0; j < amountOfIterations - 1; ++j) {
	//	rms_error = rms_error + norm_error[j];
	//}

	// err_stop — та же стартовая точка окна, но после всех проходов.
	// Опорная точка мастера: при unidir мастер задан заранее и не менялся
	// (timedomain[startDataIndex]); при bidir он тоже интегрировался и последним
	// обратным проходом вернулся в начало окна — берём текущий Xm.
	//
	// Деление на ноль НЕ глушим: err_start == 0 (slave стартовал ровно из точки
	// мастера) даёт inf либо NaN, и это честное «не измерено», а не нулевая
	// ошибка. Хост выбрасывает non-finite из min/max, такие точки не попадают
	// в автошкалу.
	if (error_estim == 4 || error_estim == 5 || error_estim == 6) {
		numb err_stop = 0;
		for (int j = 0; j < amountOfX; ++j) {
			numb e = (type_of_synch == 1 ? Xm[j] : timedomain[startDataIndex + j]) - Xs[j];
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
		return (numb)amountOfIterations*h;
	return (numb)nan("");   // недостижимо: все error_estim разобраны выше
}
// (Закрытие NVRTC-guard'а, который ранее окружал FS-kernels, удалено —
// теперь они видимы как обычные символы и для NVRTC, и для nvcc.)
