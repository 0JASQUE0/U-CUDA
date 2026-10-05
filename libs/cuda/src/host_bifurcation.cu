#include "host_internal.cuh"
__host__ void distributedSystemSimulation(
	const numb	tMax,
	const numb	h,
	const numb	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,
	const numb* initialConditions,
	const int		writableVar,
	const numb	transientTime,
	const numb* values,
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)
{
	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.8;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)	

	// Выделяем память для хранения конечного результата (пики и их количество для каждой системы)

	numb* h_data = new numb[amountOfPointsInBlock * sizeof(numb)];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("Distributed System Simulation\n");
#endif

	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// Считаем, что один блок не может использовать больше чем 48КБ памяти
	// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
	// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
	// Учитваем, что в блоке не может быть больше 1024 потоков
	blockSize = ceil((1024.0f * 8.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
	if (blockSize < 1)
	{
#ifdef DEBUG
		printf("Error : BlockSize < 1; %d line\n", __LINE__);
		exit(1);
#endif
	}

	blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

	gridSize = (amountOfThreads + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

	distributedCalculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
		(
			amountOfPointsForSkip,
			amountOfThreads,
			h,
			hSpecial,
			d_initialConditions,
			amountOfInitialConditions,
			d_values,
			amountOfValues,
			tMax / hSpecial,
			writableVar,
			d_data
			);

	// Проверка на CUDA ошибки
	gpuGlobalErrorCheck();

	// Ждем пока все потоки завершат свою работу
	gpuErrorCheck(cudaDeviceSynchronize());

	// Копирование значений пиков и их количества из памяти GPU в оперативную память

	gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// Точность чисел с плавающей запятой
	outFileStream << std::setprecision(set_precision);

	for (size_t j = 0; j < amountOfPointsInBlock; ++j)
		if (outFileStream.is_open())
		{
			outFileStream << h * j << ", " << h_data[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	delete[] h_data;
}

// Определение функции, для расчета одномерной бифуркационной диаграммы

__host__ void bifurcation1D(
	const numb	tMax,
	const int	nPts,
	const numb	h,
	const int		amountOfInitialConditions,
	const numb*	initialConditions,
	const numb*	ranges,
	const int*		indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb*	values,
	const int		amountOfValues,
	const int		preScaller,
	std::string		OUT_FILE_PATH)
{

	// Максимальный blockSize, ограниченный shared памятью
	// blockSize * requiredSharedMemPerThread <= sharedMemPerBlockLimit
	// blockSize <= sharedMemPerBlockLimit / requiredSharedMemPerThread
	//int maxBlockSizeSharedMem = (int)(sharedMemPerBlockLimit / requiredSharedMemPerThread);

	//if (maxBlockSizeSharedMem < 32) { // Проверка на минимально разумный размер блока (warp size)
	//	fprintf(stderr, "Error: Shared memory requirements are too high for any reasonable block size (min 32 threads).\n");
	//	// Обработка ошибки
	//	return; // или другая логика
	//}

	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = tMax / h / preScaller;
	//int nPts = resolution[0];
	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.92;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Точный расчет памяти для bifurcation1D
	// 1. Память на ОДНУ систему (масштабируется с nPtsLimiter)
	size_t memPerSystem =
		3 * amountOfPointsInBlock * sizeof(numb) +  // d_data, d_outPeaks, d_timeOfPeaks
		2 * sizeof(numb) +                          // d_meanFreq, d_medianFreq
		sizeof(int);                                // d_amountOfPeaks

	// 2. Константная память (выделяется ОДИН раз на весь запуск)
	size_t memConstants =
		2 * sizeof(numb) +                          // d_ranges
		sizeof(int) +                               // d_indicesOfMutVars
		amountOfInitialConditions * sizeof(numb) +  // d_initialConditions
		amountOfValues * sizeof(numb);              // d_values

	// 3. Резерв 10% на overhead драйвера, shared memory и выравнивание
	constexpr float SAFETY_FACTOR = 0.9;
	size_t availableMemory = static_cast<size_t>(freeMemory * SAFETY_FACTOR) - memConstants;

	// 4. Расчет nPtsLimiter
	size_t nPtsLimiter = availableMemory / memPerSystem;

	// 5. Ограничители: минимум 1 блок, максимум nPts
	if (nPtsLimiter < blockSize_setup) nPtsLimiter = blockSize_setup;
	if (nPtsLimiter > (size_t)nPts)    nPtsLimiter = (size_t)nPts;

	// 6. Выравнивание по размеру блока для стабильного occupancy
	nPtsLimiter = (nPtsLimiter / blockSize_setup) * blockSize_setup;

	// 7. Защита от переполнения
	if (nPtsLimiter == 0) {
		fprintf(stderr, "ERROR: Not enough GPU memory for bifurcation1D. Required per system: %zu bytes, Available: %zu bytes\n",
			memPerSystem, availableMemory);
		return;
	}

	size_t originalNPtsLimiter = nPtsLimiter;

	// Старое выделние памяти
	//size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);
	//nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)
	//size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата (пики и их количество для каждой системы)

	numb* h_timeOfPeaks = new numb[nPtsLimiter * amountOfPointsInBlock];
	numb* h_outPeaks = new numb[nPtsLimiter * amountOfPointsInBlock];
	numb* h_meanFreq = new numb[nPtsLimiter];
	numb* h_medianFreq = new numb[nPtsLimiter];
	numb* h_data = new numb[nPtsLimiter * amountOfPointsInBlock];
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
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	numb* d_timeOfPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	numb* d_meanFreq;
	numb* d_medianFreq;

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_timeOfPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_meanFreq, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_medianFreq, nPtsLimiter * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuGlobalErrorCheck();
	gpuErrorCheck(cudaDeviceSynchronize());

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	data_export::legacy::write_bif1d_config(
		outFileStream, set_precision, continuation_bif1D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, transientTime, h, preScaller, writableVar, indicesOfMutVars[0],
		ranges[0], ranges[1]);
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);
	outFileStream.close();
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	outFileStream.close();

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	size_t dataSize = nPtsLimiter * amountOfPointsInBlock * sizeof(numb);
	printf("[continuation_bif1D=%d] Data size: %zu bytes (%.2f GB)\n",
		continuation_bif1D, dataSize, dataSize / 1e9f);

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// Считаем, что один блок не может использовать больше чем 48КБ памяти
		// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
		// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
		// Учитваем, что в блоке не может быть больше 1024 потоков
		//blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		if (continuation_bif1D == 0) {

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

				if (h_localValues[writableVar] == 0) {
					h_localX[0] = -1e-6; h_localX[1] = -1e-6;
					printf("Param: %f value: %f\n", h_localValues[indicesOfMutVars[0]], h_localX[writableVar]);
				}

				for (int k = 0; k < amountOfPointsForSkip; k++) {
					calculateDiscreteModel(h_localX, h_localValues, h);
				}

				//printf("Param: %f value: %f\n", h_localValues[indicesOfMutVars[0]], h_localX[writableVar]);
				//if (abs(h_localX[writableVar]) - eps_fixed_point <= 0) {
				//	h_amountOfPeaks[j] = -1;
				//	break;
				//}

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
			//gpuErrorCheck(cudaMemset(d_data, 0, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
			//gpuGlobalErrorCheck();
			//gpuErrorCheck(cudaDeviceSynchronize());
			printf("Trajectory Calculation done\n");
			gpuErrorCheck(cudaMemcpy(d_data, h_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrorCheck(cudaMemcpy(d_amountOfPeaks, h_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));

		}
		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());
		// CUDA функция для нахождения пиков

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				d_timeOfPeaks,				// Межпиковый интервал здесь нужен
				h * (numb)preScaller);			// Шаг интегрирования нужен

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_timeOfPeaks, d_timeOfPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());
		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		if (calculate_mean_med_freq) {
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MeanAndMedianFreqCUDA, 0, blockSize_setup);
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			MeanAndMedianFreqCUDA << <gridSize, blockSize >> >
				(
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,
					d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
					d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
					d_timeOfPeaks,				// Межпиковый интервал здесь нужен
					d_meanFreq,
					d_medianFreq);			// Шаг интегрирования нужен

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			gpuErrorCheck(cudaMemcpy(h_meanFreq, d_meanFreq, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_medianFreq, d_medianFreq, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());
		}

		// Точность чисел с плавающей запятой
		outFileStream << std::setprecision(set_precision);

		// Сохранение данных в файл
		outFileStream.open(OUT_FILE_PATH, std::ios::app);
		for (size_t k = 0; k < nPtsLimiter; ++k) {
			if (h_amountOfPeaks[k] == 0) {
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " << 0 << ", " << 0 << '\n';
				}
			}
			else if (h_amountOfPeaks[k] == -1) {
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " << 0 << ", " << -1 << '\n';
				}
			}
			else {
				for (size_t j = 0; j < h_amountOfPeaks[k]; ++j) {
					if (outFileStream.is_open())
					{
						outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts, ranges[0], ranges[1], 0) << ", " <<
							h_outPeaks[k * amountOfPointsInBlock + j] << ", " << h_timeOfPeaks[k * amountOfPointsInBlock + j] << '\n';
					}
					else
					{
#ifdef DEBUG
						printf("\nOutput file open error\n");
#endif
						exit(1);
					}
				}
			}
		}
		outFileStream.close();
		if (calculate_mean_med_freq) {
			outFileStream.open(OUT_FILE_PATH + "_mean_med_freq.csv", std::ios::app);
			for (size_t k = 0; k < nPtsLimiter; ++k)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_meanFreq[k] << ", " << h_medianFreq[k] << '\n';
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	// Освобождение памяти
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_timeOfPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_meanFreq));
	gpuErrorCheck(cudaFree(d_medianFreq));
	delete[] h_localX;
	delete[] h_localValues;
	delete[] h_data;
	delete[] h_meanFreq;
	delete[] h_medianFreq;
	delete[] h_timeOfPeaks;
	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

}

/**
 * Функция, для расчета одномерной бифуркационной диаграммы по шагу.
 */
__host__ void bifurcation1DForH(
	const numb	tMax,
	const int		nPts,
	const int		amountOfInitialConditions,
	const numb* initialConditions,
	const numb* ranges,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* values,
	const int		amountOfValues,
	const int		preScaller,
	std::string		OUT_FILE_PATH)
{
	// Количество точек в одном блоке
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата (пики и их количество для каждой системы)

	numb* h_outPeaks = new numb[nPtsLimiter * amountOfPointsInBlock * sizeof(numb)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	numb* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// Считаем, что один блок не может использовать больше чем 48КБ памяти
		// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
		// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
		// Учитваем, что в блоке не может быть больше 1024 потоков
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для расчета траектории систем

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				transientTime,
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,
				d_values,					// Параметры
				amountOfValues,
				tMax,
				preScaller,
				writableVar,
				maxValue,
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// CUDA функция для нахождения пиков

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// Точность чисел с плавающей запятой
		outFileStream << std::setprecision(set_precision);

		// Сохранение данных в файл
		for (size_t k = 0; k < nPtsLimiter; ++k)
			for (size_t j = 0; j < h_amountOfPeaks[k]; ++j)
				if (outFileStream.is_open())
				{
					outFileStream << getValueByIdxLog(originalNPtsLimiter * i + k, nPts,
						ranges[0], ranges[1], 0) << ", " << h_outPeaks[k * amountOfPointsInBlock + j] << '\n';
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	// Освобождение памяти
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

}

// Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN)

__host__ void bifurcation2D(
	const numb	tMax,
	const int	nPts,
	const numb	h,
	const int		amountOfInitialConditions,
	const numb* __restrict__ initialConditions,
	const numb* __restrict__ ranges,
	const int* __restrict__ indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* __restrict__ values,
	const int		amountOfValues,
	const int		preScaller,
	const numb	eps,
	std::string		OUT_FILE_PATH)
{
	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	size_t amountOfPointsInBlock = tMax / h / preScaller;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	size_t amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.92;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Базовая оценка: 3 массива по amountOfPointsInBlock + 2 int + резерв 20%
	size_t baseMemPerSystem = amountOfPointsInBlock * 3 * sizeof(numb) + 2 * sizeof(int);

	// Добавляем статистику, если включена
	if (calculate_mean_med_freq || calculate_mean_and_variance)
		baseMemPerSystem += 8 * sizeof(numb);  // грубая оценка

	// Константы + 20% запас на overhead
	size_t memConstants = (4 + amountOfInitialConditions + amountOfValues) * sizeof(numb) + 2 * sizeof(int);
	size_t nPtsLimiter = static_cast<size_t>((freeMemory - memConstants) / baseMemPerSystem);

	// Ограничители
	nPtsLimiter = std::max(static_cast<size_t>(blockSize_setup),
		std::min(nPtsLimiter, static_cast<size_t>(nPts * nPts)));
	nPtsLimiter = (nPtsLimiter / blockSize_setup) * blockSize_setup;

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	//size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	int*  h_dbscanResult =		new int[nPtsLimiter];
	numb* h_meanFreq =			new numb[nPtsLimiter];
	numb* h_medianFreq =		new numb[nPtsLimiter];
	numb* h_meanPeak =			new numb[nPtsLimiter];
	numb* h_variancePeak =		new numb[nPtsLimiter];
	numb* h_meanInterval =		new numb[nPtsLimiter];
	numb* h_varianceInterval =	new numb[nPtsLimiter];
	numb* h_maxInterval =		new numb[nPtsLimiter];
	numb* h_maxPeak =			new numb[nPtsLimiter];
	numb* h_globalPeak =		new numb[nPtsLimiter];
	int*  h_amountOfPeaks =		new int[nPtsLimiter];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	numb* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив
	numb* d_meanFreq;
	numb* d_medianFreq;
	numb* d_meanPeak		;
	numb* d_variancePeak	;
	numb* d_meanInterval	;
	numb* d_varianceInterval;
	numb* d_maxInterval		;
	numb* d_maxPeak			;
	numb* d_globalPeak		;

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_meanFreq, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_medianFreq, nPtsLimiter * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_meanPeak		, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_variancePeak	, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_meanInterval	, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_varianceInterval, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_maxInterval		, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_maxPeak			, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_globalPeak		, nPtsLimiter * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл
	int stringCounter_1 = 0;
	int stringCounter_2 = 0;
	int stringCounter_3 = 0;
	int stringCounter_4 = 0;
	int stringCounter_5 = 0;
	int stringCounter_6 = 0;
	int stringCounter_7 = 0;
	int stringCounter_8 = 0;
	int stringCounter_9 = 0;
	int stringCounter_10 = 0;
	// Выводим в самое начало файла исследуемые диапазон

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	data_export::legacy::write_bif2d_config(
		outFileStream, set_precision, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, transientTime, h, preScaller, eps, mult_peak, mult_interval,
		writableVar, indicesOfMutVars[0], indicesOfMutVars[1], to_dbl(ranges, 4).data());
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);
	outFileStream << std::setprecision(12);
	if (outFileStream.is_open())
	{	
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	if (calculate_global_peak) {
		outFileStream.open(OUT_FILE_PATH + "_globalPeak.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	if (calculate_mean_med_freq) {
		outFileStream.open(OUT_FILE_PATH + "_meanFreq.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_medFreq.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	if (calculate_mean_and_variance) {
		outFileStream.open(OUT_FILE_PATH + "_meanPeak.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_variancePeak.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_meanInterval.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_varianceInterval.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_maxPeak.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_maxInterval.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_amountOfPeaks.csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	size_t startTime = std::clock();
	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// blockSize фиксирован (blockSize_setup); прежний расчёт из объёма
		// shared memory на поток остался в истории git.
		blockSize = blockSize_setup;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для расчета траектории систем
		size_t sharedMemNeeded = ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize;
		if (sharedMemNeeded > 48 * 1024) {
			fprintf(stderr, "WARNING: Shared memory per block (%zu) exceeds limit (48KB). Reduce blockSize.\n", sharedMemNeeded);
		}

		calculateDiscreteModelCUDA << <gridSize, blockSize, sharedMemNeeded >> >
				(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,
				2,							// Размерность ( диаграмма одномерная )
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

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		if (calculate_global_peak) {
			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			globalPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,
					d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
					d_globalPeak);							// Шаг интегрирования

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			gpuErrorCheck(cudaMemcpy(h_globalPeak, d_globalPeak, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			outFileStream.open(OUT_FILE_PATH + "_globalPeak.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_10 != 0)
						outFileStream << ", ";
					if (stringCounter_10 == nPts)
					{
						outFileStream << "\n";
						stringCounter_10 = 0;
					}
					outFileStream << h_globalPeak[i];
					++stringCounter_10;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			////////////////////////////////////////////////////////////
		}

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		blockSize = blockSize_setup;
		//printf(", %zu", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// CUDA функция для нахождения пиков

		peakFinderCUDA << <gridSize, blockSize >> >
			(   d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();
		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		if (calculate_mean_med_freq) {
			//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MeanAndMedianFreqCUDA, 0, blockSize_setup);
			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			MeanAndMedianFreqCUDA << <gridSize, blockSize >> >
				(
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,
					d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
					d_data,					// Выходной массив, куда будут записаны значения пиков
					d_intervals,				// Межпиковый интервал здесь нужен
					d_meanFreq,
					d_medianFreq);			// Шаг интегрирования нужен

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			gpuErrorCheck(cudaMemcpy(h_meanFreq, d_meanFreq, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_medianFreq, d_medianFreq, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			outFileStream.open(OUT_FILE_PATH + "_meanFreq.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_1 != 0)
						outFileStream << ", ";
					if (stringCounter_1 == nPts)
					{
						outFileStream << "\n";
						stringCounter_1 = 0;
					}
					outFileStream << h_meanFreq[i];
					++stringCounter_1;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			////////////////////////////////////////////////////////////

			outFileStream.open(OUT_FILE_PATH + "_medFreq.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_2 != 0)
						outFileStream << ", ";
					if (stringCounter_2 == nPts)
					{
						outFileStream << "\n";
						stringCounter_2 = 0;
					}
					outFileStream << h_medianFreq[i];
					++stringCounter_2;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();
		}

		if (calculate_mean_and_variance) {
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MeanAndMedianFreqCUDA, 0, blockSize_setup);
			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			MeanAndVarianceCUDA << <gridSize, blockSize >> >
				(
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,
					d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
					d_data,					// Выходной массив, куда будут записаны значения пиков
					d_intervals,				// Межпиковый интервал здесь нужен
					d_meanPeak,
					d_variancePeak,
					d_meanInterval,
					d_varianceInterval,
					d_maxPeak,
					d_maxInterval);			// Шаг интегрирования нужен

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			gpuErrorCheck(cudaMemcpy(h_meanPeak, d_meanPeak, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_variancePeak, d_variancePeak, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_meanInterval, d_meanInterval, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_varianceInterval, d_varianceInterval, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_maxPeak, d_maxPeak, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_maxInterval, d_maxInterval, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			outFileStream.open(OUT_FILE_PATH + "_meanPeak.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_3 != 0)
						outFileStream << ", ";
					if (stringCounter_3 == nPts)
					{
						outFileStream << "\n";
						stringCounter_3 = 0;
					}
					outFileStream << h_meanPeak[i];
					++stringCounter_3;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_variancePeak.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_4 != 0)
						outFileStream << ", ";
					if (stringCounter_4 == nPts)
					{
						outFileStream << "\n";
						stringCounter_4 = 0;
					}
					outFileStream << h_variancePeak[i];
					++stringCounter_4;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_meanInterval.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_5 != 0)
						outFileStream << ", ";
					if (stringCounter_5 == nPts)
					{
						outFileStream << "\n";
						stringCounter_5 = 0;
					}
					outFileStream << h_meanInterval[i];
					++stringCounter_5;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_varianceInterval.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_6 != 0)
						outFileStream << ", ";
					if (stringCounter_6 == nPts)
					{
						outFileStream << "\n";
						stringCounter_6 = 0;
					}
					outFileStream << h_varianceInterval[i];
					++stringCounter_6;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_maxPeak.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_7 != 0)
						outFileStream << ", ";
					if (stringCounter_7 == nPts)
					{
						outFileStream << "\n";
						stringCounter_7 = 0;
					}
					outFileStream << h_maxPeak[i];
					++stringCounter_7;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_maxInterval.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_8 != 0)
						outFileStream << ", ";
					if (stringCounter_8 == nPts)
					{
						outFileStream << "\n";
						stringCounter_8 = 0;
					}
					outFileStream << h_maxInterval[i];
					++stringCounter_8;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();

			outFileStream.open(OUT_FILE_PATH + "_amountOfPeaks.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// Сохранение данных в файл
			for (size_t i = 0; i < nPtsLimiter; ++i)
				if (outFileStream.is_open())
				{
					if (stringCounter_9 != 0)
						outFileStream << ", ";
					if (stringCounter_9 == nPts)
					{
						outFileStream << "\n";
						stringCounter_9 = 0;
					}
					outFileStream << h_amountOfPeaks[i];
					++stringCounter_9;
				}
				else
				{
#ifdef DEBUG
					printf("\nOutput file open error\n");
#endif
					exit(1);
				}
			outFileStream.close();
		}

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();
		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);

		blockSize = blockSize_setup;
		//printf(", %zu\n", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// CUDA функция для алгоритма DBSCAN

		dbscanCUDA << <gridSize, blockSize >> > (d_data, amountOfPointsInBlock, nPtsLimiter, d_amountOfPeaks, d_intervals, d_helpfulArray, eps, d_dbscanResult);

		//dbscanCUDA_optimized << <gridSize, blockSize >> > (d_data, amountOfPointsInBlock, nPtsLimiter, d_amountOfPeaks, d_intervals, eps, d_dbscanResult);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		outFileStream.open(OUT_FILE_PATH, std::ios::app);
		outFileStream << std::setprecision(set_precision);
		// Сохранение данных в файл
		for (size_t i = 0; i < nPtsLimiter; ++i)
			if (outFileStream.is_open())
			{
				if (stringCounter != 0)
					outFileStream << ", ";
				if (stringCounter == nPts)
				{
					outFileStream << "\n";
					stringCounter = 0;
				}

				outFileStream << h_dbscanResult[i];
				++stringCounter;
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
	printf("Time of runnig: %zu ms\n", std::clock() - startTime);
	// Освобождение памяти

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_meanFreq));
	gpuErrorCheck(cudaFree(d_medianFreq));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_globalPeak));
	gpuErrorCheck(cudaFree(d_meanPeak			));
	gpuErrorCheck(cudaFree(d_variancePeak		));
	gpuErrorCheck(cudaFree(d_meanInterval		));
	gpuErrorCheck(cudaFree(d_varianceInterval	));
	gpuErrorCheck(cudaFree(d_maxPeak));
	gpuErrorCheck(cudaFree(d_maxInterval));

	delete[] h_dbscanResult;
	delete[] h_meanFreq;
	delete[] h_medianFreq;
	delete[] h_meanPeak;
	delete[] h_variancePeak;
	delete[] h_meanInterval;
	delete[] h_varianceInterval;
	delete[] h_maxPeak;
	delete[] h_maxInterval;
	delete[] h_amountOfPeaks;
	delete[] h_globalPeak;
}

__host__ void neuronClasterization2D(
	const numb	tMax,
	const int		nPts,
	const numb	h,
	const int		amountOfInitialConditions,
	const numb* initialConditions,
	const numb* ranges,
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
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(numb)];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU

	int* d_sysCheker;			// Указатель на массив в GPU на вспомогательный массив
	numb* d_avgPeaks;
	numb* d_avgIntervals;
	numb* d_helpfulArray;
		//int* d_dbscanResult;

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
//	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
//	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_sysCheker, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

		// Выводим в самое начало файла исследуемые диапазон
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	for (int i = 1; i < 5; i++) {
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
		// Выводим в самое начало файла исследуемые диапазон
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// Считаем, что один блок не может использовать больше чем 48КБ памяти
		// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
		// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
		// Учитваем, что в блоке не может быть больше 1024 потоков

		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(numb));

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для расчета траектории систем

		calculateDiscreteModelCUDA << <gridSize, blockSize, ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,
				2,							// Размерность ( диаграмма одномерная )
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
				d_sysCheker + (i* originalNPtsLimiter),
				par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настроек блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// CUDA функция для нахождения пиков

		//peakFinderCUDA << <gridSize, blockSize >> >
		//	(d_data,						// Данные с траекториями систем
		//		amountOfPointsInBlock,		// Количество точек в одной траектории
		//		nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
		//		d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
		//		d_data,						// Выходной массив, куда будут записаны значения пиков
		//		d_intervals,				// Межпиковый интервал
		//		h * preScaller);							// Шаг интегрирования

		avgPeakFinderCUDA_for2Dbif << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_amountOfPeaks + (i* originalNPtsLimiter),
				d_sysCheker + (i * originalNPtsLimiter),
				h* preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// CUDA функция для алгоритма DBSCAN

		dbscanCUDA << <gridSize, blockSize >> >
			(	d_data,
				amountOfPointsInBlock,
				nPtsLimiter,
				d_amountOfPeaks + (i* originalNPtsLimiter),
				d_intervals,
				d_helpfulArray,
				eps,
				d_dbscanResult + (i* originalNPtsLimiter)
			);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		// CUDA функция для алгоритма DBSCAN

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	numb* h_avgPeaks = new numb[nPts * nPts];
	numb* h_avgIntervals = new numb[nPts * nPts];
	int* h_sysCheker = new int[nPts * nPts];
	int* h_dbscanResult = new int[nPts * nPts];
	int* h_amountOfPeaks = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts* nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts* nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_sysCheker, d_sysCheker, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPts* nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// Освобождение памяти

		// Сохранение найденных бассейнов притяжений в файл

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH, std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_dbscanResult[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_avgIntervals[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_sysCheker[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	stringCounter = 0;
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(4) + ".csv", std::ios::app);
	for (size_t i = 0; i < nPts * nPts; ++i)
		if (outFileStream.is_open())
		{
			if (stringCounter != 0)
				outFileStream << ", ";
			if (stringCounter == nPts)
			{
				outFileStream << "\n";
				stringCounter = 0;
			}
			outFileStream << h_amountOfPeaks[i];
			++stringCounter;
		}
		else
		{
#ifdef DEBUG
			printf("\nOutput file open error\n");
#endif
			exit(1);
		}
	outFileStream.close();

	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_sysCheker));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_sysCheker; 
	delete[] h_amountOfPeaks;
}

// Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) по IC

