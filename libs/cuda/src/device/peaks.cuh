__device__ __host__ int peakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h,
	const size_t peakStartIndex, const int peakCapacity)
{
	// Peaks may live in buffers whose per-thread stride is NOT sizeOfBlock (the
	// 2D path sizes them by max_amount_of_peaks instead of by trajectory length).
	// (size_t)-1 = legacy, peaks share the data layout.
	const size_t peakBase = (peakStartIndex == (size_t)-1) ? startDataIndex : peakStartIndex;
	// 0 = unbounded scan (legacy). Otherwise the raw scan stops at peakCapacity,
	// which is what keeps the writes inside a short peak buffer.
	const int peakCap = peakCapacity;

	if (doCalculatePeaks) {
		// Переменная для хранения найденных пиков
		int amountOfPeaks = 0;

		// Начинаем просматривать заданных интервал на наличие пиков
		for (size_t i = startDataIndex + 2; i < startDataIndex + amountOfPoints - 2; ++i)
		{
			if (peakCap > 0 && amountOfPeaks >= peakCap) break;
			// Если текущая точка больше предыдущей и больше ИЛИ РАВНА следующей, то... ( не факт, что это пик ( например: 2 3 3 4 ) )
			if (data[i] - data[i - 1] > eps_peak_delta && data[i] > peak_threshold && data[i] >= data[i + 1]) //
			{
				// От найденной точки начинаем идти вперед, пока не наткнемся на точку строго больше или меньше
				for (size_t j = i; j < startDataIndex + amountOfPoints - 2; ++j)
				{
					// Если наткнулись на точку строго больше, значит это был не пик
					if (data[j] < data[j + 1])
					{
						i = j + 1;	// --- Обновляем внешний счетчик, чтобы дважды не проходить один и тот же интервал
						break;		// --- Возвращаемся к внешнему циклу
					}
					// Если о чудо, мы нашли точку меньше, чем текущая, значит мы нашли пик
					if (data[j] - data[j + 1] > eps_peak_delta)
					{

						if (doInterpolatePeaks) {
							numb denom = data[j - 1] - (numb)2.0 * data[j] + data[j + 1];
							numb delta = 0.0;
							//denom == (numb)0.0 ? delta = 0.0 : delta = 0.5 * (data[j - 1] - data[j + 1]) / denom;
							if (fabs(denom) > 1e-12) {
								delta = 0.5 * (data[j - 1] - data[j + 1]) / denom;
							}
							outPeaks[peakBase + amountOfPeaks] = data[j] - 0.25 * (data[j - 1] - data[j + 1]) * delta;
							timeOfPeaks[peakBase + amountOfPeaks] = (numb)(j - startDataIndex - 1) + delta; // в оригинале delta*h но у нас тут индексы, умнодение на h потом
						}
						else {
							// Если массик outPeaks не пуст, то делаем запись
							if (outPeaks != nullptr)
								outPeaks[peakBase + amountOfPeaks] = data[j]; //data[j];
							// Если массик timeOfPeaks не пуст, то делаем запись
							if (timeOfPeaks != nullptr)
								timeOfPeaks[peakBase + amountOfPeaks] = (numb)(j - startDataIndex - 1);	// (numb)(j - startDataIndex - 1);
							//timeOfPeaks[startDataIndex + amountOfPeaks] = (numb)(i - startDataIndex - 1);	// (numb)(j - startDataIndex - 1);
							//timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (numb)j + (numb)i ) / (numb)2 );	// Выбираем индекс посередине между j и i

						}
						++amountOfPeaks;
						i = j + 1; // Потому что следующая точка точно не может быть пиком ( два пика не могут идти подряд )
						break;
					}
				}
			}
		}

		// Межпиковые интервалы + фильтр по eps_interPeak_delta
		// Опорный (anchor) пик, от него меряем интервал до следующего; всё, что
		// ближе eps_interPeak_delta, отбрасывается, а опора не сдвигается.
		// При eps_interPeak_delta == 0 (дефолт) условие delta >= 0 выполняется
		// всегда и цикл вырождается ровно в прежнее поведение: сдвиг пиков на
		// один индекс влево + Δt между соседями, amountOfPeaks -= 1.
		if (amountOfPeaks > 1) {
			int  writeIdx   = 0;                             // индекс записи результата
			numb anchorTime = timeOfPeaks[peakBase];         // время (в индексах) опорного пика

			for (size_t i = 1; i < amountOfPeaks; ++i) {
				// Абсолютное время текущего пика читаем ДО любой записи —
				// writeIdx всегда <= i, поэтому запись затирает уже прочитанное.
				const numb currentTime = timeOfPeaks[peakBase + i];
				const numb delta = (currentTime - anchorTime) * h;

				if (delta >= eps_interPeak_delta) {
					// Записываем ВТОРОЙ пик пары (текущий) и интервал до него.
					if (outPeaks != nullptr)
						outPeaks[peakBase + writeIdx] = outPeaks[peakBase + i];
					if (timeOfPeaks != nullptr)
						timeOfPeaks[peakBase + writeIdx] = delta;

					++writeIdx;
					anchorTime = currentTime;   // текущий пик — новая опора
				}
			}
			amountOfPeaks = writeIdx;
		}
		else {
			amountOfPeaks = 0;
		}

		if (amountOfPeaks >= max_amount_of_peaks)
			amountOfPeaks = max_amount_of_peaks;

		return amountOfPeaks;
	}
	else {
		int amountOfPeaks = amountOfPoints;

		if (amountOfPeaks >= max_amount_of_peaks)
			amountOfPeaks = max_amount_of_peaks;
		if (peakCap > 0 && amountOfPeaks > peakCap)
			amountOfPeaks = peakCap;

		for (size_t i = 0; i < amountOfPeaks; ++i)
		{
			if (outPeaks != nullptr)
				outPeaks[peakBase + i] = data[startDataIndex + i];

			if (timeOfPeaks != nullptr)
				timeOfPeaks[peakBase + i] = 0;

		}
		return amountOfPeaks - 1;
	}

}

// Нахождение пиков в "data" массиве в многопоточном режиме

__global__ void peakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h, const int* actualIterations,
	const size_t peakStride, const int peakCapacity)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// На входе amountOfPeaks[] содержит REGIME_* от calculateDiscreteModelCUDA.
	// FP и Unbound проносим сквозь этап без изменений: код режима остаётся в
	// массиве и дальше читается DBSCAN'ом (см. dbscanCUDA / avgPeakFinderCUDA).
	// Для REGIME_OSCILLATION значение перезаписывается числом найденных пиков.
	if ( amountOfPeaks[idx] == REGIME_FIXED_POINT )
	{
		amountOfPeaks[idx] = REGIME_FIXED_POINT;
		return;
	}

	if (amountOfPeaks[idx] == REGIME_UNBOUND)
	{
		amountOfPeaks[idx] = REGIME_UNBOUND;
		return;
	}

	// dt-sweep: only the first actualIterations[idx] samples of this thread's
	// (worst-case-sized) block are real data -- the rest is stale/uninitialised
	// from a previous chunk. nullptr (non-swept path) keeps scanning the full
	// sizeOfBlock, unchanged from before.
	size_t scanLen = sizeOfBlock;
	if (actualIterations != nullptr) {
		size_t a = (size_t)actualIterations[idx];
		if (a < scanLen) scanLen = a;
	}

	// peakStride == 0 -- peaks share the trajectory layout (legacy 1D path).
	// Otherwise outPeaks/timeOfPeaks are short per-thread rows of peakStride
	// entries and the scan stops at peakCapacity so it cannot run past the row.
	const size_t peakBase = (peakStride != 0) ? (size_t)idx * peakStride
	                                          : (size_t)idx * sizeOfBlock;

	amountOfPeaks[idx] = peakFinder( data, (size_t)idx * sizeOfBlock, scanLen, outPeaks, timeOfPeaks, h,
		peakBase, peakCapacity );
	return;
}

__global__ void globalPeakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* globalPeakValue)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее
	if (amountOfPeaks[idx] == 0)
	{
		amountOfPeaks[idx] = 0;
		globalPeakValue[idx] = 1e25;
		return;
	}

	globalPeakValue[idx] = globalPeakFinder(data, (size_t)idx * sizeOfBlock, sizeOfBlock);
	return;
}

__global__ void DFT_custom(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* checkerArray, numb* AkCOS, numb* BkSIN, numb* rangesFreq, numb* window, int nFreq, numb h,
	const int logFreqAxis)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	size_t startDataIndex = idx * sizeOfBlock;
	size_t startIndexFreq = idx * nFreq;

	// Читаем один раз через read-only cache вместо двух отдельных обращений
	// к глобальной памяти (было: checkerArray[idx] в каждом if отдельно).
	const int checker = __ldg(&checkerArray[idx]);

	if (checker == -1)
	{
		for (size_t i = 0; i < nFreq; i++) {
			AkCOS[startIndexFreq + i] = -1.0;
			BkSIN[startIndexFreq + i] = -1.0;
		}
		return;
	}

	if (checker == 0)
	{
		for (size_t i = 0; i < nFreq; i++) {
			AkCOS[startIndexFreq + i] = 0.0;
			BkSIN[startIndexFreq + i] = 0.0;
		}
		return;
	}

	const int RESET_INTERVAL = 1000;

	const numb freq_lo = __ldg(&rangesFreq[0]);
	const numb freq_hi = __ldg(&rangesFreq[1]);
	const numb f_step = (freq_hi - freq_lo) / (numb)(nFreq - 1);
	const numb f_start = freq_lo;
	const numb gamma = 2 * pi / (sizeOfBlock - 1.0);
	const numb psi = 2 * pi * h;
	numb f_k = 0;
	numb cos_theta = 0;
	numb sin_theta = 0;
	numb cos_n = 0;
	numb sin_n = 0;
	//numb theta = 2 * pi * h;
	//numb f_k = 0;

	for (size_t k = 0; k < nFreq; k++) {
		AkCOS[startIndexFreq + k] = 0;
		BkSIN[startIndexFreq + k] = 0;
		f_k = logFreqAxis ? getValueByIdx_log((int)k, nFreq, freq_lo, freq_hi, 0)
		                   : (f_start + (numb)k * f_step);
		cos_theta = cos(2.0 * pi * h * f_k);
		sin_theta = sin(2.0 * pi * h * f_k);
		cos_n = 1.0;
		sin_n = 0.0;
		int reset_counter = RESET_INTERVAL;

		for (size_t n = 0; n < sizeOfBlock; n++) {

			// window[n]*data[n] раньше считалось дважды подряд (для AkCOS и
			// для BkSIN отдельно) — считаем один раз и переиспользуем. Оба
			// массива этот kernel только читает — __ldg() (read-only cache).
			const numb wd = __ldg(&window[n]) * __ldg(&data[startDataIndex + n]);
			AkCOS[startIndexFreq + k] += wd * cos_n;
			BkSIN[startIndexFreq + k] += wd * sin_n;

			numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
			numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
			cos_n = new_cos;
			sin_n = new_sin;

			// Та же семантика, что и (n+1)%RESET_INTERVAL==0 && (n+1)<sizeOfBlock,
			// но декрементирующий счётчик вместо деления по модулю на каждой
			// итерации (modulo на GPU заметно дороже сравнения/декремента).
			if (--reset_counter == 0) {
				reset_counter = RESET_INTERVAL;
				if (n + 1 < sizeOfBlock) {
					numb exact_angle = psi * f_k * (numb)(n + 1.0);
					cos_n = cosf(exact_angle);
					sin_n = sinf(exact_angle);
				}
			}
		}
		AkCOS[startIndexFreq + k] /= (numb)sizeOfBlock;
		BkSIN[startIndexFreq + k] /= (numb)sizeOfBlock;
	}

	//for (size_t i = 0; i < nFreq; i++) {
	//	AkCOS[idx * nFreq + i] = rangesFreq[0] + (numb)i * f_step;
	//	BkSIN[idx * nFreq + i] = 2 * i + 1;
	//}

	return;
}

__global__ void MeanAndMedianFreqCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq,  numb* medianFreq)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее
	if (amountOfPeaks[idx] == -1)
	{
		meanFreq[idx] = 0;
		medianFreq[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		meanFreq[idx] = 0;
		medianFreq[idx] = 0;
		return;
	}

	meanFreq[idx] = 0;
	medianFreq[idx] = 1;
	int AoP = amountOfPeaks[idx];
	MeanAndMedianFreq(idx, idx * sizeOfBlock, AoP, outPeaks, timeOfPeaks, meanFreq, medianFreq);
	//return;
}

__global__ void MeanAndVarianceCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее
	if (amountOfPeaks[idx] == -1)
	{
		meanPeak[idx] = 0;
		variancePeak[idx] = 0;
		meanInterval[idx] = 0;
		varianceInterval[idx] = 0;
		maxPeak[idx] = 0;
		maxInterval[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		meanPeak[idx] = 0;
		variancePeak[idx] = 0;
		meanInterval[idx] = 0;
		varianceInterval[idx] = 0;
		maxPeak[idx] = 0;
		maxInterval[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 1)
	{
		meanPeak[idx] = outPeaks[idx];
		variancePeak[idx] = 0;
		meanInterval[idx] = timeOfPeaks[idx];
		varianceInterval[idx] = 0;
		maxPeak[idx] = outPeaks[idx];
		maxInterval[idx] = timeOfPeaks[idx];
		return;
	}

	meanPeak[idx] =			999;
	variancePeak[idx] =		999;
	meanInterval[idx] =		999;
	varianceInterval[idx] = 999;
	maxPeak[idx] =			999;
	maxInterval[idx] =		999;

	int AoP = amountOfPeaks[idx];
	MeanAndVariance(idx, idx * sizeOfBlock, AoP, outPeaks, timeOfPeaks, meanPeak, variancePeak, meanInterval, varianceInterval, maxPeak, maxInterval);
	//return;
}

__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
{
	// Вычисляем индекс потока, в котором находимся в даный момент
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, amountOfPeaks[idx], outPeaks, timeOfPeaks, h);
	return;
}

__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, bool* flags, numb ostrovskyThreshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		flags[idx * 5 + 3] = true;
		return;
	}

	numb lastPoint = data[idx * sizeOfBlock + sizeOfBlock - 1];

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks);

	//FIRST CONDITION
	flags[idx * 5 + 0] = true;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] - outPeaks[i - 1] > 0)
		{
			flags[idx * 5 + 0] = false;
			break;
		}
	}

	//SECOND & THIRD CONDITION
	bool flagOne = false;
	bool flagZero = false;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] > ostrovskyThreshold)
			flagOne = true;
		else
			flagZero = true;
		if (flagOne && flagZero)
			break;
	}

	if (flagOne && flagZero)
		flags[idx * 5 + 1] = true;
	else
		flags[idx * 5 + 1] = false;

	if (flagOne && !flagZero)
		flags[idx * 5 + 2] = false;
	else
		flags[idx * 5 + 2] = true;

	//FOUR CONDITION
	if (amountOfPeaks[idx] == 0 || amountOfPeaks[idx] == 1)
		flags[idx * 5 + 3] = true;
	else
		flags[idx * 5 + 3] = false;

	//FIVE CONDITION
	if (lastPoint > ostrovskyThreshold)
		flags[idx * 5 + 4] = true;
	else
		flags[idx * 5 + 4] = false;
	return;
}

__device__ __host__ int kde(numb* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, numb kdeSamplesInterval1,
	numb kdeSamplesInterval2, numb kdeSmoothH)
{
	if (amountOfPoints == 0)
		return 0;
	if (amountOfPoints == 1 || amountOfPoints == 2)
		return 1;
	if (amountOfPoints > maxAmountOfPeaks)
		return maxAmountOfPeaks;

	numb k1 = kdeSampling * amountOfPoints;
	numb k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	numb delt = 0;
	numb prevPrevData2 = 0;
	numb prevData2 = 0;
	numb data2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < amountOfPoints; ++m)
		{
			numb tempData = (data[startDataIndex + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}

		if (w < 2)
			continue;
		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	return resultKde;
}

__global__ void kdeCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, numb kdeSamplesInterval1,
	numb kdeSamplesInterval2, numb kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		kdeResult[idx] = 0;
		return;
	}
	kdeResult[idx] = kde(data, idx * sizeOfBlock, amountOfPeaks[idx], maxAmountOfPeaks,
		kdeSampling, kdeSamplesInterval1, kdeSamplesInterval2, kdeSmoothH);
}

// Вычисляет расстояние между двумя точками

__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;

	numb dx = abs(x2 - x1);
	numb dy = abs(y2 - y1);

	//return hypot(dx, dy);
	return sqrt(dx*dx + dy*dy);
}

// Функция DBSCAN
