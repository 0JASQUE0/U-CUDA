#include "host_internal.cuh"
__host__ void bifurcation_DFT_1D(
	const numb	tMax,
	const int	nPts,
	const int	nFreq,								// Разрешение диаграммы
	const numb	h,
	const int		amountOfInitialConditions,
	const numb* initialConditions,
	const numb* ranges,
	const numb* rangesFreq,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* values,
	const int		amountOfValues,
	const int		preScaller,
	const numb	eps,
	std::string		OUT_FILE_PATH)
{
	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = tMax / h / preScaller;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 1.0 + sizeof(numb) * nPts * nFreq * 0.0);
	//size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3.0);
	nPtsLimiter = nPtsLimiter > (nPts) ? (nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	int* h_dbscanResult = new int[nPtsLimiter];
	numb* h_AkCOS = new numb[nPtsLimiter * nFreq];
	numb* h_BkSIN = new numb[nPtsLimiter * nFreq];
	numb* h_window = new numb[amountOfPointsInBlock];
	numb* h_data = new numb[amountOfPointsInBlock * nPtsLimiter];
	int* h_amountOfPeaks = new int[nPtsLimiter];
	numb* h_localX = new numb[amountOfInitialConditions];
	numb* h_localValues = new numb[amountOfValues];

	for (int i = 0; i < amountOfInitialConditions; i++)
		h_localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; i++)
		h_localValues[i] = values[i];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	numb* d_rangesFreq;
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_AkCOS;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	numb* d_BkSIN;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	numb* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив
	numb* d_window;

	// Выделяем память в GPU

	const numb gamma = (numb)2.0 * pi / (numb)(amountOfPointsInBlock - 1);
	numb windowSum = (numb)0;
	for (int n = 0; n < amountOfPointsInBlock; ++n) {
		//h_window[n] = (numb)0.53836 - (numb)0.46164 * cos(gamma * n ); // Hamming
		h_window[n] = (numb)0.5 * ((numb)1.0 - cos(gamma * n)); // Hanning
		//h_window[n] = (numb)0.42 - (numb)0.5 * cos(gamma * n) + (numb)0.08 * cos(2 * gamma * n); // Blackman
		//h_window[n] = (numb)0.35875 - (numb)0.48829 * cos(gamma * n) + (numb)0.14128 * cos(2 * gamma * n) - (numb)0.01168 * cos(3 * gamma * n); // Blackman-Harris
		//h_window[n] = (numb)1.0;
		windowSum += h_window[n];
	}
	// Нормировка на единичное среднее: DFT_custom делит сумму на длину блока, а
	// не на sum(w), поэтому без этого абсолютная амплитуда спектра зависит от
	// того, какая из строк выше раскомментирована (для Hanning занижена вдвое).
	// UI-путь (parametric_engine.cpp::cpu_build_window) нормирует так же —
	// иначе Debug и Release давали бы разный масштаб для одной системы.
	if (windowSum > (numb)0) {
		const numb windowMean = windowSum / (numb)amountOfPointsInBlock;
		for (int n = 0; n < amountOfPointsInBlock; ++n)
			h_window[n] /= windowMean;
	}

	gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_rangesFreq, 2 * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_AkCOS, nPtsLimiter * nFreq * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_BkSIN, nPtsLimiter * nFreq * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_window, amountOfPointsInBlock * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU
	gpuErrorCheck(cudaMemcpy(d_window, h_window, amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_rangesFreq, rangesFreq, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuGlobalErrorCheck();
	gpuErrorCheck(cudaDeviceSynchronize());

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл
	int stringCounter_1 = 0;
	// Выводим в самое начало файла исследуемые диапазон

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	data_export::legacy::write_dft1d_config(
		outFileStream, set_precision, continuation_bif1D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, transientTime, h, preScaller, writableVar, indicesOfMutVars[0],
		ranges[0], ranges[1]);
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + "AkCOS.csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << ", " << ranges[1] << "\n";
		outFileStream << rangesFreq[0] << ", " << rangesFreq[1] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + "BkSIN.csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << ", " << ranges[1] << "\n";
		outFileStream << rangesFreq[0] << ", " << rangesFreq[1] << "\n";
	}
	outFileStream.close();

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// Считаем, что один блок не может использовать больше чем 48КБ памяти
		// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
		// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
		// Учитваем, что в блоке не может быть больше 1024 потоков

		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		//blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(numb));

		if (continuation_bif1D == 0) {

			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

			// CUDA функция для расчета траектории систем

			calculateDiscreteModelCUDA << <gridSize, blockSize, ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
					nPtsLimiter,
					amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
					i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
					amountOfPointsForSkip,
					1,							// Размерность ( диаграмма одномерная )
					d_ranges,					// Массив с диапазонами
					h,
					d_indicesOfMutVars,			// Индексы изменяемых параметров
					d_initialConditions,		// Начальные условия
					amountOfInitialConditions,
					d_values,					// Параметры
					amountOfValues,
					amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
					preScaller,
					writableVar,
					maxValue,
					d_data,						// Массив, где будет хранится траектория систем
					d_amountOfPeaks,
					par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		}
		else {
			for (int j = 0; j < nPtsLimiter; j++) {
				numb xPrev[AMOUNTOFX];
				numb checker;
				h_localValues[indicesOfMutVars[0]] = ranges[0] + (numb)(i * originalNPtsLimiter + j) * (ranges[1] - ranges[0]) / ((numb)nPts - (numb)1.0);

				for (int k = 0; k < amountOfPointsForSkip; k++) {
					calculateDiscreteModel(h_localX, h_localValues, h);
				}

				for (int k = 0; k < amountOfPointsInBlock; k++) {

					for (int m = 0; m < amountOfInitialConditions; ++m)
						xPrev[m] = h_localX[m];

					h_data[j * amountOfPointsInBlock + k] = (h_localX[writableVar]);

					for (int m = 0; m < preScaller; m++)
						calculateDiscreteModel(h_localX, h_localValues, h);
				}

				h_amountOfPeaks[j] = 1;

				checker = 0;
				for (int m = 0; m < amountOfInitialConditions; ++m)
					checker = checker + fabsf(h_localX[m]);

				if (isnan(checker) || isinf(checker) || fabsf(checker) > maxValue)
					h_amountOfPeaks[j] = 0;

				numb tempResult = 0;

				for (int m = 0; m < amountOfInitialConditions; ++m)
					tempResult += abs(h_localX[m] - xPrev[m]);

				if (tempResult == 0 || abs(tempResult) < eps_fixed_point)
					h_amountOfPeaks[j] = -1;
			}
			gpuErrorCheck(cudaMemcpy(d_data, h_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrorCheck(cudaMemcpy(d_amountOfPeaks, h_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		blockSize = 64;
		//printf(", %zu", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		printf("Trajectories done\n");
		// CUDA функция для нахождения пиков

		DFT_custom << <gridSize, blockSize >> >
			(
				d_data, 
				amountOfPointsInBlock, 
				nPtsLimiter,
				d_amountOfPeaks, 
				d_AkCOS, 
				d_BkSIN, 
				d_rangesFreq,
				d_window,
				nFreq, 
				h * preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		////
		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_AkCOS, d_AkCOS, nPtsLimiter * nFreq * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_BkSIN, d_BkSIN, nPtsLimiter * nFreq * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		outFileStream.open(OUT_FILE_PATH + "_" + "AkCOS.csv", std::ios::app);
		outFileStream << std::setprecision(set_precision);
		// Сохранение данных в файл
		for (size_t i = 0; i < nPtsLimiter; ++i)
			for (size_t j = 0; j < nFreq; ++j)
				if (outFileStream.is_open())
				{
					if (j != nFreq - 1)
						outFileStream << h_AkCOS[ i * nFreq + j] << ", ";
					else
						outFileStream << h_AkCOS[i * nFreq + j] << "\n";
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_" + "BkSIN.csv", std::ios::app);
		outFileStream << std::setprecision(set_precision);
		// Сохранение данных в файл
		for (size_t i = 0; i < nPtsLimiter; ++i)
			for (size_t j = 0; j < nFreq; ++j)
				if (outFileStream.is_open())
				{
					if (j != nFreq - 1)
						outFileStream << h_BkSIN[i * nFreq + j] << ", ";
					else
						outFileStream << h_BkSIN[i * nFreq + j] << "\n";
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
		outFileStream.close();

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	// Освобождение памяти

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_rangesFreq));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_AkCOS));
	gpuErrorCheck(cudaFree(d_BkSIN));
	gpuErrorCheck(cudaFree(d_window));
	delete[] h_localX;
	delete[] h_localValues;
	delete[] h_amountOfPeaks;
	delete[] h_data;
	delete[] h_window;
	delete[] h_dbscanResult;
	delete[] h_AkCOS;
	delete[] h_BkSIN;
}

// =====================================================================================
//                    RQA — Recurrence Quantification Analysis
// =====================================================================================
// Публичный API и смысл параметров — в rqa.h. Здесь только реализация: одна матрица
// расстояний на GPU и три прохода по ней (гистограмма для подбора eps, диагональные
// линии, вертикальные линии), плюс необязательный подсчёт треугольников для сетевых мер.
//
// Почему НЕ gpuErrorCheck: gpuAssert выходит из процесса (abort = true). Для оффлайн-скриптов
// это нормально, но RQA зовётся из UI-воркера, и провалившийся cudaMalloc на большой матрице
// обязан стать сообщением в окне проекции, а не падением приложения. Проверяется при этом
// КАЖДЫЙ вызов — макросом RQA_CHECK ниже, просто с возвратом ошибки вместо abort.
