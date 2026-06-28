// --- Заголовочный файл ---
#include "hostLibrary.cuh"

// --- Путь для сохранения результирующих файлов ---
//#define OUT_FILE_PATH "C:\\Users\\KiShiVi\\Desktop\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// --- Директива, объявление которой выводит в консоль отладочные сообщения ---
#define DEBUG


__host__ void distributedSystemSimulation(
	const numb	tMax,							// Время моделирования системы
	const numb	h,								// Шаг интегрирования
	const numb	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,				// Массив с начальными условиями
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,							// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)					// Количество параметров	
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h;

	int amountOfThreads = hSpecial / h;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.8;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)	

	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	numb* h_data = new numb[amountOfPointsInBlock * sizeof(numb)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Distributed System Simulation\n");
#endif

	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
	// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
	// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
	// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
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

	// --- Проверка на CUDA ошибки ---
	gpuGlobalErrorCheck();

	// --- Ждем пока все потоки завершат свою работу ---
	gpuErrorCheck(cudaDeviceSynchronize());

	// -------------------------------------------------------------------------------------
	// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
	// -------------------------------------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(h_data, d_data, amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// -------------------------------------------------------------------------------------

	// --- Точность чисел с плавающей запятой ---
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


// ----------------------------------------------------------------------------
// --- Определение функции, для расчета одномерной бифуркационной диаграммы ---
// ----------------------------------------------------------------------------

__host__ void bifurcation1D(
	const numb	tMax,							// Время моделирования системы
	const int	nPts,						// Разрешение диаграммы
	const numb	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb*	initialConditions,				// Массив с начальными условиями
	const numb*	ranges,							// Диаппазон изменения переменной
	const int*		indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
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

	
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;
	//int nPts = resolution[0];
	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.92;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Точный расчет памяти для bifurcation1D ---
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



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

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


	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

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



	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

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

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuGlobalErrorCheck();
	gpuErrorCheck(cudaDeviceSynchronize());


	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		if (continuation_bif1D == 1)
			outFileStream << "1D continuation bifurcation \n";
		if (continuation_bif1D == 0)
			outFileStream << "1D classical bifurcation \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";;
		}
		outFileStream << "CT = " << tMax << "\n";
		outFileStream << "TT = " << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
		outFileStream << "indexVar for peakfinder = " << writableVar << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << "\n";
		outFileStream << "start vlaue = " << ranges[0] << ", stop vlaue = " << ranges[1] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);
	outFileStream.close();
	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	outFileStream.close();
	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	size_t dataSize = nPtsLimiter * amountOfPointsInBlock * sizeof(numb);
	printf("[continuation_bif1D=%d] Data size: %zu bytes (%.2f GB)\n",
		continuation_bif1D, dataSize, dataSize / 1e9f);

	//size_t allocatedSize;
	//cudaDeviceGetAttribute(&allocatedSize, cudaDevAttrMaxAllocationSize, 0);
	//if (dataSize > allocatedSize) {
	//	printf("ERROR: Not enough GPU memory! Requested: %zu, Max allowed: %zu\n",
	//		dataSize, allocatedSize);
	//}

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		//blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelCUDA, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		if (continuation_bif1D == 0) {


			// --------------------------------------------------
			// --- CUDA функция для расчета траектории систем ---
			// --------------------------------------------------

			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
					nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
					amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
					i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
					amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
					1,							// Размерность ( диаграмма одномерная )
					d_ranges,					// Массив с диапазонами
					h,							// Шаг интегрирования
					d_indicesOfMutVars,			// Индексы изменяемых параметров
					d_initialConditions,		// Начальные условия
					amountOfInitialConditions,	// Количество начальных условий
					d_values,					// Параметры
					amountOfValues,				// Количество параметров
					amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
					preScaller,					// Множитель, который уменьшает время и объем расчетов
					writableVar,				// Индекс уравнения, по которому будем строить диаграмму
					maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
					d_data,						// Массив, где будет хранится траектория систем
					d_amountOfPeaks,
					par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

			// --------------------------------------------------
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
		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---


		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());
		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(	d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				d_timeOfPeaks,				// Межпиковый интервал здесь нужен
				h * (numb)preScaller);			// Шаг интегрирования нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_timeOfPeaks, d_timeOfPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		gpuGlobalErrorCheck();
		gpuErrorCheck(cudaDeviceSynchronize());
		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		if (calculate_mean_med_freq) {
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MeanAndMedianFreqCUDA, 0, blockSize_setup);
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			MeanAndMedianFreqCUDA << <gridSize, blockSize >> >
				(
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
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
		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(set_precision);

		// --- Сохранение данных в файл ---
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

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------
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

	// ---------------------------
}


/**
 * Функция, для расчета одномерной бифуркационной диаграммы по шагу.
 */
__host__ void bifurcation1DForH(
	const numb	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,				// Массив с начальными условиями
	const numb* ranges,							// Диапазон изменения шага
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек в одном блоке ---
	int amountOfPointsInBlock = tMax / (ranges[0] < ranges[1] ? ranges[0] : ranges[1]) / preScaller;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	numb* h_outPeaks = new numb[nPtsLimiter * amountOfPointsInBlock * sizeof(numb)];
	int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	numb* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
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

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA_H << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
			(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				transientTime,				// Время пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				tMax,						// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA_H << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_outPeaks,					// Выходной массив, куда будут записаны значения пиков
				nullptr,					// Межпиковый интервал здесь не нужен
				0);							// Шаг интегрирования не нужен

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(set_precision);

		// --- Сохранение данных в файл ---
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

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_outPeaks));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_outPeaks;
	delete[] h_amountOfPeaks;

	// ---------------------------
}


// ------------------------------------------------------------------------
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) ---
// ------------------------------------------------------------------------

__host__ void bifurcation2D(
	const numb	tMax,								// Время моделирования системы
	const int	nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* __restrict__ initialConditions,					// Массив с начальными условиями
	const numb* __restrict__ ranges,								// Диапазоны изменения параметров
	const int* __restrict__ indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* __restrict__ values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	size_t amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
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

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	//size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

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

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

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
	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	// -----------------------------
	gpuErrorCheck(cudaMalloc((void**)&d_meanFreq, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_medianFreq, nPtsLimiter * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_meanPeak		, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_variancePeak	, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_meanInterval	, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_varianceInterval, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_maxInterval		, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_maxPeak			, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_globalPeak		, nPtsLimiter * sizeof(numb)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	// ------------------------------------------------------

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
	// --- Выводим в самое начало файла исследуемые диапазон ---

	
	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "2D bifurcation \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =  " << tMax << "\n";
		outFileStream << "TT =" << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
		outFileStream << "eps_DBSCAN = " << eps << "\n";
		outFileStream << "mult_peak_DBSCAN  = " << mult_peak << "\n";
		outFileStream << "mult_interval_DBSCAN = " << mult_interval << "\n";
		outFileStream << "indexVar for peakfinder = " << writableVar << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
		outFileStream << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
	}
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
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	if (calculate_mean_med_freq) {
		outFileStream.open(OUT_FILE_PATH + "_meanFreq.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_medFreq.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	if (calculate_mean_and_variance) {
		outFileStream.open(OUT_FILE_PATH + "_meanPeak.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_variancePeak.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_meanInterval.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_varianceInterval.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_maxPeak.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_maxInterval.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();

		outFileStream.open(OUT_FILE_PATH + "_amountOfPeaks.csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}
	
	size_t startTime = std::clock();
	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		
		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		
		//printf("blockSize: %zu", blockSize);
//		if (blockSize < 1)
//		{
//#ifdef DEBUG
//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
//			exit(1);
//#endif
//		}
//
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = blockSize_setup;
		//blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(numb));
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------
		size_t sharedMemNeeded = (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize;
		if (sharedMemNeeded > 48 * 1024) {
			fprintf(stderr, "WARNING: Shared memory per block (%zu) exceeds limit (48KB). Reduce blockSize.\n", sharedMemNeeded);
		}

		calculateDiscreteModelCUDA << <gridSize, blockSize, sharedMemNeeded >> >
				(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks,
				par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		if (calculate_global_peak) {
			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			globalPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
					d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
					d_globalPeak);							// Шаг интегрирования

			gpuGlobalErrorCheck();
			gpuErrorCheck(cudaDeviceSynchronize());

			gpuErrorCheck(cudaMemcpy(h_globalPeak, d_globalPeak, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			outFileStream.open(OUT_FILE_PATH + "_globalPeak.csv", std::ios::app);
			outFileStream << std::setprecision(set_precision);
			// --- Сохранение данных в файл ---
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

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		blockSize = blockSize_setup;
		//printf(", %zu", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		peakFinderCUDA << <gridSize, blockSize >> >
			(   d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_amountOfPeaks,			// Выходной массив, куда будут записаны количества пиков для каждой системы
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				h * preScaller);							// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();
		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		if (calculate_mean_med_freq) {
			//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MeanAndMedianFreqCUDA, 0, blockSize_setup);
			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

			MeanAndMedianFreqCUDA << <gridSize, blockSize >> >
				(
					amountOfPointsInBlock,		// Количество точек в одной траектории
					nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
					nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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
			// --- Сохранение данных в файл ---
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

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();
		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);

		blockSize = blockSize_setup;
		//printf(", %zu\n", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		dbscanCUDA << <gridSize, blockSize >> > (d_data, amountOfPointsInBlock, nPtsLimiter, d_amountOfPeaks, d_intervals, d_helpfulArray, eps, d_dbscanResult);

		//dbscanCUDA_optimized << <gridSize, blockSize >> > (d_data, amountOfPointsInBlock, nPtsLimiter, d_amountOfPeaks, d_intervals, eps, d_dbscanResult);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


		outFileStream.open(OUT_FILE_PATH, std::ios::app);
		outFileStream << std::setprecision(set_precision);
		// --- Сохранение данных в файл ---
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
	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

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
	// ---------------------------
}


__host__ void neuronClasterization2D(
	const numb	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,					// Массив с начальными условиями
	const numb* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(numb)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

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
	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

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
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

		// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	for (int i = 1; i < 5; i++) {
		outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(i) + ".csv");
		// --- Выводим в самое начало файла исследуемые диапазон ---
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}
	

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл



	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---

		//blockSize = ceil((1*1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(numb));

		//		if (blockSize < 1)
		//		{
		//#ifdef DEBUG
		//			printf("Error : BlockSize < 1; %d line\n", __LINE__);
		//			exit(1);
		//#endif
		//		}
		//
		//		blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		//
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------


		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_sysCheker + (i* originalNPtsLimiter),
				par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настроек блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

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
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_amountOfPeaks + (i* originalNPtsLimiter),
				d_sysCheker + (i * originalNPtsLimiter),
				h* preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;


		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dbscanCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

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

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------



		// -----------------------------------------
		// --- CUDA функция для алгоритма DBSCAN ---
		// -----------------------------------------

		//dbscanCUDA << <gridSize, blockSize >> >
		//	(d_data,
		//		amountOfPointsInBlock,
		//		nPtsLimiter,
		//		d_amountOfPeaks,
		//		d_intervals,
		//		d_helpfulArray,
		//		eps,
		//		d_dbscanResult);

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		//outFileStream << std::setprecision(set_precision);

//		// --- Сохранение данных в файл ---
//		for (size_t i = 0; i < nPtsLimiter; ++i)
//			if (outFileStream.is_open())
//			{
//				if (stringCounter != 0)
//					outFileStream << ", ";
//				if (stringCounter == nPts)
//				{
//					outFileStream << "\n";
//					stringCounter = 0;
//				}
//				outFileStream << h_dbscanResult[i];
//				++stringCounter;
//			}
//			else
//			{
//#ifdef DEBUG
//				printf("\nOutput file open error\n");
//#endif
//				exit(1);
//			}

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
	


	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

		// --- Сохранение найденных бассейнов притяжений в файл ---

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
	// ---------------------------
}

// ------------------------------------------------------------------------------
// --- Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) по IC ---
// ------------------------------------------------------------------------------



__host__ void LLE1D(
	const numb	tMax,								// Время моделирования системы
	const numb	NT,									// Время нормализации
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const numb	eps,								// Эпсилон для LLE
	const numb* initialConditions,					// Массив с начальными условиями
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,								// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)						// Количество параметров
{
	// --- Количество точек, которое будет смоделировано одной системой во время нормализации NT ---
	size_t amountOfNT_points = NT / h;

	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / NT;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	numb* h_lleResult = new numb[nPtsLimiter];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	   // Указатель на массив с начальными условиями
	numb* d_values;				   // Указатель на массив с параметрами

	numb* d_lleResult;			   // Память для хранения конечного результата

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(numb)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceilf((numb)nPts / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "1D LLE \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =" << " " << tMax << "\n";
		outFileStream << "NT =" << " " << NT << "\n";
		outFileStream << "TT =" << " " << transientTime << "\n";
		outFileStream << "h =" << " " << h << "\n";
		outFileStream << "eps=" << " " << eps << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar =" << " " << indicesOfMutVars[0] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar =" << " " << indicesOfMutVars[0] << "\n";
		outFileStream << "start vlaue = " << ranges[0] << ", stop vlaue = " << ranges[1] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("LLE 1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		//int blockSizeMin;
		//int blockSizeMax;
		int blockSize;		// Переменная для хранения размера блока
		int minGridSize;	// Переменная для хранения минимального размера сетки
		int gridSize;		// Переменная для хранения сетки

		//blockSizeMax = 48000 / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSizeMin = (3 + amountOfValues) * sizeof(numb);
		//blockSize = (blockSizeMax + blockSizeMin) / 2;
		blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions + amountOfValues) * sizeof(numb)));

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		// ------------------------------------
		// --- CUDA функция для расчета LLE ---
		// ------------------------------------

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
			(nPts,								// Общее разрешение
				nPtsLimiter, 						// Разрешение в текущем расчете
				NT, 								// Время нормализации
				tMax, 								// Время моделирования
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,				// Количество точек, которое будет промоделированно до основного расчета (transientTime)
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h, 									// Шаг интегрирования
				eps, 								// Эпсилон
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions, 			// Количество начальных условий
				d_values, 							// Параметры
				amountOfValues, 					// Количество параметров
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,						// Индекс переменной в x[] по которому строим диаграмму
				maxValue, 							// Макксимальное значение переменной при моделировании
				d_lleResult);						// Результирующий массив

		// ------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(set_precision);

		// --- Сохранение данных в файл ---

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0) << ", " << h_lleResult[k] << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}


__host__ void LLE2D(
	const numb tMax,
	const numb NT,
	const int nPts,
	const numb h,
	const numb eps,
	const numb* initialConditions,
	const int amountOfInitialConditions,
	const numb* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const numb maxValue,
	const numb transientTime,
	const numb* values,
	const int amountOfValues,
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory *= 0.8;
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock);

	//nPtsLimiter = 10000; // Pizdec kostil' ot Boga
	
	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;
	//nPtsLimiter = nPtsLimiter > amount_GPU ? amount_GPU : nPtsLimiter;
	size_t originalNPtsLimiter = nPtsLimiter;

	numb* h_lleResult = new numb[nPtsLimiter];

	numb* d_ranges;
	int* d_indicesOfMutVars;
	numb* d_initialConditions;
	numb* d_values;

	numb* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(numb)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((numb)nPts * (numb)nPts) / (numb)nPtsLimiter);

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "2D LLE \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =" << " " << tMax << "\n";
		outFileStream << "NT =" << " " << NT << "\n";
		outFileStream << "TT =" << " " << transientTime << "\n";
		outFileStream << "h =" << " " << h << "\n";
		outFileStream << "eps=" << " " << eps << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
		outFileStream << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LLE2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif
	int stringCounter = 0;

	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LLEKernelCUDA, 0, blockSize_setup);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		//blockSize = ceil((1024.0f * 32.0f) / ((3 * amountOfInitialConditions) * sizeof(numb)));

		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, LLEKernelCUDA, 0, blockSize_setup);
		//blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )


		//LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> > (
		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

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
				outFileStream << h_lleResult[i];
				++stringCounter;
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}

__host__ void LS1D(
	const numb tMax,
	const numb NT,
	const int nPts,
	const numb h,
	const numb eps,
	const numb* initialConditions,
	const int amountOfInitialConditions,
	const numb* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const numb maxValue,
	const numb transientTime,
	const numb* values,
	const int amountOfValues,
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 16;
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	numb* h_lleResult = new numb[nPtsLimiter * amountOfInitialConditions];

	numb* d_ranges;
	int* d_indicesOfMutVars;
	numb* d_initialConditions;
	numb* d_values;

	numb* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(numb)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf((numb)nPts / (numb)nPtsLimiter);

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "1D LS \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =" << " " << tMax << "\n";
		outFileStream << "NT =" << " " << NT << "\n";
		outFileStream << "TT =" << " " << transientTime << "\n";
		outFileStream << "h =" << " " << h << "\n";
		outFileStream << "eps=" << " " << eps << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << "\n";
		outFileStream << "start vlaue = " << ranges[0] << ", stop vlaue = " << ranges[1] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LS1D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;
		
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LSKernelCUDA, 0, blockSize_setup);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		//blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSizeMin = (3 + amountOfValues) * sizeof(numb);
		//blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;


		blockSizeMax = 32000 / ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSizeMin = (3 + amountOfValues) * sizeof(numb);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(numb))* blockSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			1, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < nPtsLimiter; ++k)
			if (outFileStream.is_open())
			{
				outFileStream << getValueByIdx(originalNPtsLimiter * i + k, nPts,
					ranges[0], ranges[1], 0);
				for (int j = 0; j < amountOfInitialConditions; ++j)
					outFileStream << ", " << h_lleResult[k * amountOfInitialConditions + j];
				outFileStream << '\n';
			}
			else
			{
				printf("\nOutput file open error\n");
				exit(1);
			}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] h_lleResult;
}

__host__ void LS2D(
	const numb tMax,
	const numb NT,
	const int nPts,
	const numb h,
	const numb eps,
	const numb* initialConditions,
	const int amountOfInitialConditions,
	const numb* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const numb maxValue,
	const numb transientTime,
	const numb* values,
	const int amountOfValues,
	std::string		OUT_FILE_PATH)
{
	size_t amountOfNT_points = NT / h;
	int amountOfPointsInBlock = tMax / NT;
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;
	size_t totalMemory;

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));

	freeMemory /= 1.0;
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * amountOfInitialConditions * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > nPts * nPts ? nPts * nPts : nPtsLimiter;

	size_t originalNPtsLimiter = nPtsLimiter;

	numb* h_lleResult = new numb[nPtsLimiter * amountOfInitialConditions];

	numb* d_ranges;
	int* d_indicesOfMutVars;
	numb* d_initialConditions;
	numb* d_values;

	numb* d_lleResult;

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(numb)));

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));


	size_t amountOfIteration = (size_t)ceilf(((numb)nPts * (numb)nPts) / (numb)nPtsLimiter);

	std::ofstream outFileStream;
	
	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "2D LS \n";
		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =" << " " << tMax << "\n";
		outFileStream << "NT =" << " " << NT << "\n";
		outFileStream << "TT =" << " " << transientTime << "\n";
		outFileStream << "h =" << " " << h << "\n";
		outFileStream << "eps=" << " " << eps << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
		outFileStream << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
	}
	outFileStream.close();


#ifdef DEBUG
	printf("LS2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int* stringCounter = new int[amountOfInitialConditions];

	for (int i = 0; i < amountOfInitialConditions; ++i)
		stringCounter[i] = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		outFileStream.open(OUT_FILE_PATH + std::to_string(i + 1) + ".csv");
		if (outFileStream.is_open())
		{
			outFileStream << ranges[0] << " " << ranges[1] << "\n";
			outFileStream << ranges[2] << " " << ranges[3] << "\n";
		}
		outFileStream.close();
	}

	for (int i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSizeMin;
		int blockSizeMax;
		int blockSize;
		int minGridSize;
		int gridSize;

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, LSKernelCUDA, 24576, 128);
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		
		blockSizeMax = 32000 / ((amountOfInitialConditions * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSizeMin = (3 + amountOfValues) * sizeof(numb);
		blockSize = blockSizeMax;// (blockSizeMax + blockSizeMin) / 2;
		blockSize = 32;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;

		size_t sharedMemSize = ((3 * amountOfInitialConditions +
			2 * amountOfInitialConditions * amountOfInitialConditions +
			amountOfValues) * sizeof(numb)) * blockSize;

		//LSKernelCUDA << < gridSize, blockSize, ((3 * amountOfInitialConditions + 2 * amountOfInitialConditions * amountOfInitialConditions + amountOfValues) * sizeof(numb))* blockSize >> > (
		LSKernelCUDA << < gridSize, blockSize, sharedMemSize >> > (
			nPts, nPtsLimiter, NT, tMax, amountOfPointsInBlock,
			i * originalNPtsLimiter, amountOfPointsForSkip,
			2, d_ranges, h, eps, d_indicesOfMutVars, d_initialConditions,
			amountOfInitialConditions, d_values, amountOfValues,
			tMax / NT, 1, writableVar,
			maxValue, d_lleResult);

		gpuGlobalErrorCheck();

		gpuErrorCheck(cudaDeviceSynchronize());

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (size_t k = 0; k < amountOfInitialConditions; ++k)
		{
			outFileStream.open(OUT_FILE_PATH + std::to_string(k + 1) + ".csv", std::ios::app);
			for (size_t m = 0 + k; m < nPtsLimiter * amountOfInitialConditions; m = m + amountOfInitialConditions)
			{
				if (outFileStream.is_open())
				{
					if (stringCounter[k] != 0)
						outFileStream << ", ";
					if (stringCounter[k] == nPts)
					{
						outFileStream << "\n";
						stringCounter[k] = 0;
					}
					outFileStream << h_lleResult[m];
					stringCounter[k] = stringCounter[k] + 1;
				}
			}
			outFileStream.close();
		}

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_lleResult));

	delete[] stringCounter;
	delete[] h_lleResult;
}

void CUDA_dbscan(numb* data, numb* intervals, int* labels, int* helpfulArray, const int amountOfData, const numb eps, const int blockSize_fixed)
{
	int resultClusters = 0;
	int amountOfClusters = 0;				// Количество кластеров
	int amountOfNegativeClusters = 0;
	int* amountOfNeighbors = new int[1];			// Вспомогательная переменная - сколько было найдено соседей у точки
	*amountOfNeighbors = 0;
	int* neighbors = new int[amountOfData];			// Вспомогательная переменная - индексы найденных соседей

	int* d_amountOfNeighbors;						// Вспомогательная переменная - сколько было найдено соседей у точки
	int* d_neighbors;								// Вспомогательная переменная - индексы найденных соседей

	cudaMalloc((void**)& d_amountOfNeighbors, sizeof(int));
	cudaMalloc((void**)& d_neighbors, sizeof(int) * amountOfData);

	cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

	int amountOfVisitedPoints = 0;

	int blockSize1;			// Переменная для хранения размера блока
	int minGridSize1;		// Переменная для хранения минимального размера сетки
	int gridSize1;			// Переменная для хранения сетки


	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, blockSize_setup);

	blockSize1 = blockSize1 > blockSize_setup ? blockSize_setup : blockSize1;			// Не превышаем ограничение в 512 потока в блоке
	blockSize1 = blockSize_fixed;
	gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

	int blockSize2;			// Переменная для хранения размера блока
	int minGridSize2;		// Переменная для хранения минимального размера сетки
	int gridSize2;			// Переменная для хранения сетки

	cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, blockSize_setup);

	blockSize2 = blockSize2 > blockSize_setup ? blockSize_setup : blockSize2;			// Не превышаем ограничение в 512 потока в блоке
	blockSize2 = blockSize_fixed;
	gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;

	// Цикл по всем точкам даты
	//while (true)

	int* clearIdx = new int[1];

	int* d_clearIdx;
	cudaMalloc((void**)& d_clearIdx, sizeof(int));

	for (int i = 0; i < amountOfData; i++)
	{
		//int* clearIdx = new int[1];
		*clearIdx = -1;

		//int* d_clearIdx;
		//cudaMalloc((void**)& d_clearIdx, sizeof(int));

		cudaMemcpy(d_clearIdx, clearIdx, sizeof(int), cudaMemcpyHostToDevice);

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_fixed_points_kernel, 0, blockSize_setup);
		//blockSize2 = blockSize2 > blockSize_setup ? blockSize_setup : blockSize2;			// Не превышаем ограничение в 512 потока в блоке
		//gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;

		CUDA_dbscan_search_fixed_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
			amountOfData, d_clearIdx);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		//gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

		if (*clearIdx == -1)
		{

			//cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, CUDA_dbscan_search_clear_points_kernel, 0, blockSize_setup);
			//blockSize2 = blockSize2 > blockSize_setup ? blockSize_setup : blockSize2;			// Не превышаем ограничение в 512 потока в блоке
			//gridSize2 = (amountOfData + blockSize2 - 1) / blockSize2;
			CUDA_dbscan_search_clear_points_kernel << <gridSize2, blockSize2 >> > (data, intervals, helpfulArray, labels,
				amountOfData, d_clearIdx);

			++amountOfClusters;
			resultClusters = amountOfClusters;
			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			//gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(clearIdx, d_clearIdx, sizeof(int), cudaMemcpyDeviceToHost);

			if (*clearIdx == -1)
				break;
		}
		else
		{
			--amountOfNegativeClusters;
			resultClusters = amountOfNegativeClusters;
		}

		*amountOfNeighbors = 0;
		for (size_t i = 0; i < amountOfData; ++i)
			neighbors[i] = 0;

		cudaMemcpy(d_amountOfNeighbors, amountOfNeighbors, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_neighbors, neighbors, sizeof(int) * amountOfData, cudaMemcpyHostToDevice);

		//cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, blockSize_setup);
		//blockSize1 = blockSize1 > blockSize_setup ? blockSize_setup : blockSize1;			// Не превышаем ограничение в 512 потока в блоке
		//gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

		CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
			resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, *clearIdx, helpfulArray);

		if (cudaGetLastError() != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
		}

		gpuGlobalErrorCheck();
		cudaDeviceSynchronize();

		cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);


		for (size_t i = 0; i < *amountOfNeighbors; ++i)
		{

			//cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, CUDA_dbscan_kernel, 0, blockSize_setup);
			//blockSize1 = blockSize1 > blockSize_setup ? blockSize_setup : blockSize1;			// Не превышаем ограничение в 512 потока в блоке
			//gridSize1 = (amountOfData + blockSize1 - 1) / blockSize1;

			CUDA_dbscan_kernel << <gridSize1, blockSize1 >> > (data, intervals, labels, amountOfData, eps,
				resultClusters/*d_amountOfClusters*/, d_amountOfNeighbors, d_neighbors, neighbors[i], helpfulArray);

			if (cudaGetLastError() != cudaSuccess)
			{
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);
			}

			gpuGlobalErrorCheck();
			cudaDeviceSynchronize();

			cudaMemcpy(amountOfNeighbors, d_amountOfNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(neighbors, d_neighbors, sizeof(int) * (*amountOfNeighbors), cudaMemcpyDeviceToHost);

			++amountOfVisitedPoints;
		}

		//delete[] clearIdx;
		
	}

	delete[] clearIdx;
	delete[] amountOfNeighbors;
	delete[] neighbors;

	gpuErrorCheck(cudaFree(d_amountOfNeighbors));
	gpuErrorCheck(cudaFree(d_neighbors));

}

__host__ void basinsOfAttraction(
	const numb	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,					// Массив с начальными условиями
	const numb* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH,
	const int blockSize_fixed)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		
	printf("freeMemory: %zu\n", freeMemory);
	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * (amountOfPointsInBlock * (2)));

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	//nPtsLimiter = nPtsLimiter - (nPtsLimiter % blockSize_fixed);
	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	int* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	numb* d_avgPeaks;
	numb* d_avgIntervals;

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_dbscanResult, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_helpfulArray, nPts * nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)& d_avgPeaks, nPts * nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_avgIntervals, nPts * nPts * sizeof(numb)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "basins of attraction \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =  " << tMax << "\n";
		outFileStream << "TT =" << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
		outFileStream << "eps_DBSCAN = " << eps << "\n";
		outFileStream << "mult_MeanPeak_DBSCAN  = " << mult_peak << "\n";
		outFileStream << "mult_MeanInterval_DBSCAN = " << mult_interval << "\n";
		outFileStream << "indexVar for peakfinder = " << writableVar << "\n";
		outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
		outFileStream << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
	}
	outFileStream.close();

	// ------------------------------------------------------

#ifdef DEBUG
	//printf("Basins of attraction\n");
	//printf("nPtsLimiter : %zu\n", nPtsLimiter);
	//printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл
	outFileStream.open(OUT_FILE_PATH);
	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(set_precision);

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	
	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	printf("Basins, CT: %zu", (int)tMax);
	printf(", res: %zu x %zu", nPts, nPts);
	printf(", amountOfIteration: %zu", amountOfIteration);
	printf(", nPtsLimiter: %zu", nPtsLimiter);
	printf(", blockSize: %zu\n", blockSize_fixed);
	size_t startTime = std::clock();

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 48.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		//cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, calculateDiscreteModelCUDA, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize, 0);
		//printf("Recommended block size: %d\n", blockSize);
		//printf("Recommended grid size: %d\n", gridSize);

		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		//blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = blockSize_fixed;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				d_helpfulArray + (i * originalNPtsLimiter), // Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
				0);			

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, blockSize_setup);
		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = blockSize_fixed;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}
	printf("features done in: %zu ms\n", std::clock() - startTime);

	int* h_dbscanResult = new int[nPts * nPts];
	for (int i = 0; i < nPts * nPts; i++)
		h_dbscanResult[i] = 0;

	
	

	gpuErrorCheck(cudaMemcpy(d_dbscanResult, h_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));

	startTime = clock();
	CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps, blockSize_fixed);
	printf("DBSCAN done in: %zu ms\n\n", std::clock() - startTime);

	numb* h_avgPeaks = new numb[nPts * nPts];
	numb* h_avgIntervals = new numb[nPts * nPts];
	int* h_helpfulArray = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgPeaks,	 d_avgPeaks,	 nPts * nPts * sizeof(numb),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(numb),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));
	
	

	// --- Сохранение найденных бассейнов притяжений в файл ---

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

	// --- Сохранение средних значений пиков в файл ---

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
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение средних значений межпиков в файл ---

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
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение характеристик точек сетки начальных условий в файл ---

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
			//outFileStream << h_avgIntervals[i];
			if (h_helpfulArray[i] != NAN)
				outFileStream << h_helpfulArray[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();


	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_helpfulArray;

	// ---------------------------
}

__host__ void basinsOfAttraction_logAxes(
	const numb	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,					// Массив с начальными условиями
	const numb* ranges,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.95;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * amountOfInitialConditions * 0.5);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(numb)];
	//numb* h_helpfulArray = new numb[nPts * nPts];			// Указатель на массив в GPU на вспомогательный массив

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	int* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	int* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	numb* d_avgPeaks;
	numb* d_avgIntervals;

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)&d_data, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_dbscanResult, nPts * nPts * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_helpfulArray, nPts * nPts * sizeof(int)));


	gpuErrorCheck(cudaMalloc((void**)&d_avgPeaks, nPts * nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_avgIntervals, nPts * nPts * sizeof(numb)));
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "basins of attraction log axes\n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =  " << tMax << "\n";
		outFileStream << "TT =" << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
		outFileStream << "eps_DBSCAN = " << eps << "\n";
		outFileStream << "mult_MeanPeak_DBSCAN  = " << mult_peak << "\n";
		outFileStream << "mult_MeanInterval_DBSCAN = " << mult_interval << "\n";
		outFileStream << "indexVar for peakfinder = " << writableVar << "\n";
		outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
		outFileStream << "start vlaue_2 = " << ranges[2] << ", stop vlaue_2 = " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Точность чисел с плавающей запятой ---
	outFileStream << std::setprecision(set_precision);

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(1) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(2) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH + "_" + std::to_string(3) + ".csv");
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}
	outFileStream.close();
	//stringCounter = 0;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



	int blockSize;			// Переменная для хранения размера блока
	int minGridSize;		// Переменная для хранения минимального размера сетки
	int gridSize;			// Переменная для хранения сетки

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCUDA_logAxes << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			(	nPts,										// Общее разрешение диаграммы - nPts
				nPtsLimiter,								// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,						// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,					// Количество уже посчитанных точек систем
				amountOfPointsForSkip,						// Количество точек для пропуска ( transientTime )
				2,											// Размерность ( диаграмма одномерная )
				d_ranges,									// Массив с диапазонами
				h,											// Шаг интегрирования
				d_indicesOfMutVars,							// Индексы изменяемых параметров
				d_initialConditions,						// Начальные условия
				amountOfInitialConditions,					// Количество начальных условий
				d_values,									// Параметры
				amountOfValues,								// Количество параметров
				amountOfPointsInBlock,						// Количество итераций ( равно количеству точек для одной системы )
				preScaller,									// Множитель, который уменьшает время и объем расчетов
				writableVar,								// Индекс уравнения, по которому будем строить диаграмму
				maxValue,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,										// Массив, где будет хранится траектория систем
				d_helpfulArray + (i * originalNPtsLimiter)	// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
			);												

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_logMaximas, 0, blockSize_setup);

		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

		avgPeakFinderCUDA_logMaximas << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,				// Количетсво систем, высчитываемой в текущей итерации
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);			// Шаг интегрирования

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	int* h_dbscanResult = new int[nPts * nPts];
	for (int i = 0; i < nPts * nPts; i++)
		h_dbscanResult[i] = 0;

	gpuErrorCheck(cudaMemcpy(d_dbscanResult, h_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));


	CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps);


	numb* h_avgPeaks = new numb[nPts * nPts];
	numb* h_avgIntervals = new numb[nPts * nPts];
	int* h_helpfulArray = new int[nPts * nPts];



	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgPeaks, d_avgPeaks, nPts * nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));


	// --- Сохранение найденных бассейнов притяжений в файл ---

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

	// --- Сохранение средних значений пиков в файл ---

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
			if (h_avgPeaks[i] != NAN)
				outFileStream << h_avgPeaks[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение средних значений межпиков в файл ---

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
			//outFileStream << h_avgIntervals[i];
			if (h_avgIntervals[i] != NAN)
				outFileStream << h_avgIntervals[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();

	// --- Сохранение характеристик точек сетки начальных условий в файл ---

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
			//outFileStream << h_avgIntervals[i];
			if (h_helpfulArray[i] != NAN)
				outFileStream << h_helpfulArray[i];
			else
				outFileStream << 999;
			++stringCounter;
		}
	outFileStream.close();


	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));
	gpuErrorCheck(cudaFree(d_avgPeaks));
	gpuErrorCheck(cudaFree(d_avgIntervals));

	delete[] h_dbscanResult;
	delete[] h_avgPeaks;
	delete[] h_avgIntervals;
	delete[] h_helpfulArray;

	// ---------------------------
}


__host__ void TimeDomainCalculation(
	const numb	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const numb	h,								// Шаг интегрирования
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,				// Массив с начальными условиями
	const numb* ranges,							// Диаппазон изменения переменной
	const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH)						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ---------------------------------------------------------------------------------------------------
	// --- Выделяем память для хранения конечного результата (пики и их количество для каждой системы) ---
	// ---------------------------------------------------------------------------------------------------

	//numb* h_outPeaks = new numb[nPtsLimiter * amountOfPointsInBlock * sizeof(numb)];
	//int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	//numb* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	//int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPts * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	//gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	//gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	//std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	//static curandState *states = NULL;

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
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

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
		//calculateDiscreteModelCUDA_rand << <gridSize, blockSize >> >
			(	
				nPts,						// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
				1,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				writableVar,				// Индекс уравнения, по которому будем строить диаграмму
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				d_data,						// Массив, где будет хранится траектория систем
				nullptr,
				par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		//gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------



#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	numb* h_data = new numb[amountOfPointsInBlock * nPts];

	gpuErrorCheck(cudaMemcpy(h_data, d_data, nPts* amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- Точность чисел с плавающей запятой ---

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	outFileStream << std::setprecision(set_precision);

	size_t stringCounter = 0;
	for (size_t k = 0; k < nPts; ++k) {
		for (size_t i = 0; i < amountOfPointsInBlock; ++i) {
			if (outFileStream.is_open())
			{
				if (h_data[i] != NAN)
					outFileStream << h_data[i + k * (size_t)amountOfPointsInBlock];
				else
					outFileStream << 999;

				//if (stringCounter != 0)
				//	outFileStream << ", ";
				//if (stringCounter == amountOfPointsInBlock-1)
				//{
				//	outFileStream << "\n";
				//	stringCounter = 0;
				//}
				//else
				outFileStream << ", ";
				//outFileStream << h_avgIntervals[i];

	//			++stringCounter;
			}
		} 
		outFileStream << "\n";
	}
	outFileStream.close();

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	//gpuErrorCheck(cudaFree(d_outPeaks));
	//gpuErrorCheck(cudaFree(d_amountOfPeaks));

	delete[] h_data;

	//delete[] h_outPeaks;
	//delete[] h_amountOfPeaks;

	// ---------------------------
}

__host__ void FastSynchro(
	const numb		tMax,								// Время моделирования системы
	const numb		transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb		NTime,								// Длина отрезка по которому будет проводиться синхронизация
	const numb*		values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const numb		h,									// Шаг интегрирования
	const numb*		kForward,							// Массив коэффициентов синхронизации вперед
	const numb*		kBackward,							// Массив коэффициентов синхронизации назад
	const numb*		initialConditionsMaster,			// Массив с начальными условиями мастера
	const numb*		initialConditionsSlave,				// Массив с начальными условиями слейва
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb		maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	const int		iterOfSynchr,						// Число итераций синхронизации
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	std::string		OUT_FILE_PATH)						// Эпсилон для алгоритма DBSCAN 
{


	// --- Количество точек, которое используется в окне синхронизации ---
	int amountOfNTPoints = NTime / h;

	// --- Общее количесвто точек в исходной траектории ---
	int amountOfCTPoints = tMax / h;

	// --- Количество точек переходного процесса ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU
	size_t nPts = (amountOfCTPoints / preScaller);

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 1.0;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) *  (2.0 * nPts));

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter; // Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )


	//numb* timeDomain = new numb[(amountOfCTPoints + amountOfNTPoints) * sizeof(numb) * amountOfInitialConditions];
	//numb* arrayZeros = new numb[sizeof(numb) * amountOfInitialConditions];
	//numb* Xm = new numb[sizeof(numb) * amountOfInitialConditions];
	//numb* Xs = new numb[sizeof(numb) * amountOfInitialConditions];

	numb* timeDomain = new numb[(amountOfCTPoints + amountOfNTPoints) * amountOfInitialConditions];
	numb* arrayZeros = new numb[amountOfInitialConditions];
	numb* Xm = new numb[amountOfInitialConditions];
	numb* Xs = new numb[amountOfInitialConditions];

	// --- Инициализация начальных условий ---
	for (int i = 0; i < amountOfInitialConditions; i++) {
		arrayZeros[i] = 0;
		Xm[i] = initialConditionsMaster[i];
		Xs[i] = initialConditionsSlave[i];
	}

	// --- Расчет переходного процесса ---
	for (size_t i = 0; i < amountOfPointsForSkip; i++) {
		calculateDiscreteModelforFastSynchro(Xm, Xm, arrayZeros, values, h, 1);
		//calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h, 1);
	}

	// --- Расчет исходной траектории ---
	for (size_t i = 0; i < amountOfCTPoints + amountOfNTPoints; i++) {

		for (int j = 0; j < amountOfInitialConditions; j++)
			timeDomain[i * amountOfInitialConditions + j] = Xm[j];

		calculateDiscreteModelforFastSynchro(Xm, Xm, arrayZeros, values, h, 1);
		//calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h, 1);

	}

	printf(" --- Calculation of trajectory done\n");

	// --- Выделяем память для хранения конечного результата 
	//numb* h_output = new numb[nPts * sizeof(numb)];
	numb* h_output = new numb[nPts];
	// --- Указатели на области памяти в GPU ---

	numb* d_timeDomain;
	numb* d_output;
	numb* d_Xs;
	numb* d_values;
	numb* d_kForward;
	numb* d_kBackward;

	// --- Выделяем память в GPU ---

	gpuErrorCheck(cudaMalloc((void**)&d_timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_output, nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_Xs, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kForward, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kBackward, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));

	// --- Копируем начальные входные параметры в память GPU ---

	gpuErrorCheck(cudaMemcpy(d_timeDomain, timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_Xs, Xs, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		outFileStream << "Symmetric synch, attractor \n";
		if (type_of_synch == 0)
			outFileStream << "Unidirectional synch \n";
		if (type_of_synch == 1)
			outFileStream << "Bidirectional synch \n";
		if (error_estim == 0)
			outFileStream << "RMS(error) on the last iteration \n";
		if (error_estim == 1)
			outFileStream << "number of iteration to achieve RMS(error) <= FS_error_trs \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0_master[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditionsMaster[kk] << ", ";
			else
				outFileStream << initialConditionsMaster[kk] << " }\n";;
		}
		outFileStream << "X0_slave[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditionsSlave[kk] << ", ";
			else
				outFileStream << initialConditionsSlave[kk] << " }\n";;
		}
		outFileStream << "K_forward[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << kForward[kk] << ", ";
			else
				outFileStream << kForward[kk] << " }\n";
		}
		outFileStream << "K_backward[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << kBackward[kk] << ", ";
			else
				outFileStream << kBackward[kk] << " }\n";
		}
		outFileStream << "iter of synch =  " << iterOfSynchr << "\n";
		outFileStream << "CT = " << tMax << "\n";
		outFileStream << "WT = " << NTime << "\n";
		outFileStream << "TT = " << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
	}
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = nPts - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---

		//blockSize = ceil((1*1024.0f * 4.0f) / (amountOfNTPoints * sizeof(numb)));
		//blockSize = ceil((1 * 1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		//blockSize = 10000 / ((5*amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSize = 5*amountOfNTPoints;
		//
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		blockSize = blockSize_setup;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelforFastSynchroCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------


			//calculateDiscreteModelforFastSynchroCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			//calculateDiscreteModelforFastSynchroCUDA << <4*1024, 16>> > //, 1*(amountOfInitialConditions + amountOfValues + amountOfNTPoints) * sizeof(numb)* 1 >> >

		calculateDiscreteModelforFastSynchroCUDA << < gridSize, blockSize >> >
			(
				nPts,						//const int		nPts,
				nPtsLimiter,				//const int		nPtsLimiter,
				amountOfNTPoints,		//const int		sizeOfBlock,
				h, 							//const numb	h,
				d_Xs,						//numb* initialConditions,
				amountOfInitialConditions,	//const int		amountOfInitialConditions,
				d_values,						//const numb* values,
				d_kForward,					//const numb* k_forward,
				d_kBackward,					//const numb* k_backward,
				iterOfSynchr,							//const int		iterOfSynchr,
				amountOfValues,								//const int		
				amountOfNTPoints,							//const int		amountOfIterations,
				maxValue,									//const numb	maxValue,
				d_timeDomain + (i * originalNPtsLimiter) * amountOfInitialConditions * preScaller,								//numb* timedomain,
				d_output + (i * originalNPtsLimiter),			//numb* output
				preScaller);

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


#ifdef DEBUG
		printf(" --- Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}
	// --- Перенос расчитанного реезультата с gpu 
	gpuErrorCheck(cudaMemcpy(h_output, d_output, nPts * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// --- Освобождение памяти в gpu 
	gpuErrorCheck(cudaFree(d_timeDomain));
	gpuErrorCheck(cudaFree(d_output));
	gpuErrorCheck(cudaFree(d_kForward));
	gpuErrorCheck(cudaFree(d_kBackward));
	gpuErrorCheck(cudaFree(d_values));
	gpuErrorCheck(cudaFree(d_Xs));

	// --- ЗАпись реузльтата в файл 
	outFileStream << std::setprecision(set_precision);

	for (size_t j = 0; j < nPts; ++j)
		if (outFileStream.is_open())
		{
			for (int k = 0; k < amountOfInitialConditions; k++)
				outFileStream << timeDomain[amountOfInitialConditions * j * preScaller + k] << ", ";

			outFileStream << h_output[j] << '\n';
			//outFileStream << timeDomain[amountOfInitialConditions * j * preScaller]		<< ", " 
			//			  << timeDomain[amountOfInitialConditions * j * preScaller + 1] << ", " 
			//			  << timeDomain[amountOfInitialConditions * j * preScaller + 2] << ", " 
			//	          << h_output[j] << '\n';
		}
		else
		{
			printf("\nOutput file open error\n");
			exit(1);
		}
	outFileStream.close();

	printf(" --- Writing in file done\n");

	delete[] arrayZeros;
	delete[] timeDomain;
	delete[] Xm;
	delete[] Xs;
	delete[] h_output;
}

__host__ void FastSynchro_2(
	const numb		NTime,								// Длина отрезка по которому будет проводиться синхронизация
	const int		nPts,								// Разрешение диаграммы
	const numb*		values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const numb		h,									// Шаг интегрирования
	const numb*		ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const numb*		kForward,							// Массив коэффициентов синхронизации вперед
	const numb*		kBackward,							// Массив коэффициентов синхронизации назад
	const numb*		initialConditions,					// Массив с начальными условиями мастера
	const numb*		initConditionsSlave,
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb		maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	const int		iterOfSynchr,						// Число итераций синхронизации
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	std::string		OUT_FILE_PATH)
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = NTime / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = 0;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfInitialConditions * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )



	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

	numb* h_dbscanResult = new numb[nPtsLimiter * sizeof(numb)];

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_initialConditionsSlave;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.
	numb* d_intervals;			// Указатель на массив в GPU с межпиковыми интервалами пиков
	numb* d_dbscanResult;			// Указатель на массив в GPU результирующей матрицы (диаграммы) в GPU
	numb* d_helpfulArray;			// Указатель на массив в GPU на вспомогательный массив

	numb* d_kForward;
	numb* d_kBackward;

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	gpuErrorCheck(cudaMalloc((void**)&d_data, amountOfInitialConditions * nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_ranges, 4 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_indicesOfMutVars, 2 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_initialConditionsSlave, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kForward, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kBackward, amountOfInitialConditions * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)&d_amountOfPeaks, nPtsLimiter * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)&d_intervals, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_dbscanResult, nPtsLimiter * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_helpfulArray, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions,	   initialConditions,	 amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditionsSlave, initConditionsSlave, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// --- Выводим в самое начало файла исследуемые диапазон ---
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---
		blockSize = ceil((1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		if (blockSize < 1)
		{
#ifdef DEBUG
			printf("Error : BlockSize < 1; %d line\n", __LINE__);
			exit(1);
#endif
		}

		blockSize = blockSize > 512 ? 512 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке

		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// --------------------------------------------------
		// --- CUDA функция для расчета траектории систем ---
		// --------------------------------------------------

		calculateDiscreteModelICCforFastSynchro << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			(nPts,							// Общее разрешение диаграммы - nPts
				nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
				amountOfInitialConditions * amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,							// Шаг интегрирования
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				d_initialConditionsSlave,
				amountOfInitialConditions,	// Количество начальных условий
				d_values,					// Параметры
				amountOfValues,				// Количество параметров
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,					// Множитель, который уменьшает время и объем расчетов
				maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
				iterOfSynchr,
				d_kForward,
				d_kBackward,
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks,
				d_dbscanResult);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему


		//const int		nPts,
		//	const int		nPtsLimiter,
		//	const int		sizeOfBlock,
		//	const int		amountOfCalculatedPoints,
		//	const int		dimension,
		//	numb* ranges,
		//	const numb	h,
		//	int* indicesOfMutVars,
		//	numb* initialConditions,
		//	const int		amountOfInitialConditions,
		//	const numb* values,
		//	const int		amountOfValues,
		//	const int		amountOfIterations,
		//	const int		preScaller,
		//	const numb	maxValue,
		//	const int		iterOfSynchr,
		//	const numb* kForward,
		//	const numb* kBackward,
		//	numb* data,
		//	int* maxValueCheckerArray,
		//	numb* FastSynchroError)

		// --------------------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());


		// -------------------------------------------------------------------------------------
		// --- Копирование значений пиков и их количества из памяти GPU в оперативную память ---
		// -------------------------------------------------------------------------------------

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// -------------------------------------------------------------------------------------

		// --- Точность чисел с плавающей запятой ---
		outFileStream << std::setprecision(set_precision);

		// --- Сохранение данных в файл ---
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

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_ranges));
	gpuErrorCheck(cudaFree(d_indicesOfMutVars));
	gpuErrorCheck(cudaFree(d_initialConditions));
	gpuErrorCheck(cudaFree(d_values));

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;


}

__host__ void bifurcation_DFT_1D(
	const numb	tMax,								// Время моделирования системы
	const int	nPts,								// Разрешение диаграммы
	const int	nFreq,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* initialConditions,					// Массив с начальными условиями
	const numb* ranges,								// Диапазоны изменения параметров
	const numb* rangesFreq,								// Диапазоны изменения параметров
	const int* indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH)								// Эпсилон для алгоритма DBSCAN 
{
	// --- Количество точек, которое будет смоделировано одной системой с одним набором параметров ---
	int amountOfPointsInBlock = tMax / h / preScaller;

	// --- Количество точек, которое будет пропущено при моделировании системы ---
	// --- (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах) ---
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// --- Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени ---
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 1.0 + sizeof(numb) * nPts * nFreq * 0.0);
	//size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 3.0);
	nPtsLimiter = nPtsLimiter > (nPts) ? (nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// ----------------------------------------------------------
	// --- Выделяем память для хранения конечного результата  ---
	// ----------------------------------------------------------

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

	// -----------------------------------------
	// --- Указатели на области памяти в GPU ---
	// -----------------------------------------

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

	// -----------------------------------------

	// -----------------------------
	// --- Выделяем память в GPU ---
	// -----------------------------

	const numb gamma = (numb)2.0 * pi / (numb)(amountOfPointsInBlock - 1);
	for (int n = 0; n < amountOfPointsInBlock; ++n) {
		//h_window[n] = (numb)0.53836 - (numb)0.46164 * cos(gamma * n ); // Hamming
		h_window[n] = (numb)0.5 * ((numb)1.0 - cos(gamma * n)); // Hanning
		//h_window[n] = (numb)1.0;
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
	// -----------------------------

	// ---------------------------------------------------------
	// --- Копируем начальные входные параметры в память GPU ---
	// ---------------------------------------------------------
	gpuErrorCheck(cudaMemcpy(d_window, h_window, amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_rangesFreq, rangesFreq, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuGlobalErrorCheck();
	gpuErrorCheck(cudaDeviceSynchronize());
	// ---------------------------------------------------------

	// --- Расчет количества итераций для генерации бифуркационной диаграммы ---
	size_t amountOfIteration = (size_t)ceil((numb)(nPts) / (numb)nPtsLimiter);

	// ------------------------------------------------------
	// --- Открытие выходного текстового файла для записи ---
	// ------------------------------------------------------

	std::ofstream outFileStream;

	// ------------------------------------------------------

#ifdef DEBUG
	printf("Bifurcation 2D\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл
	int stringCounter_1 = 0;
	// --- Выводим в самое начало файла исследуемые диапазон ---


	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	if (outFileStream.is_open())
	{
		outFileStream << std::setprecision(set_precision);
		if (continuation_bif1D == 1)
			outFileStream << "1D continuation bifurcation DFT \n";
		if (continuation_bif1D == 0)
			outFileStream << "1D classical bifurcation DFT \n";

		if (par_or_var == 1)
			outFileStream << "Parameter esimation \n";
		if (par_or_var == 0)
			outFileStream << "Initial conditions esimation \n";
		outFileStream << "a[" << amountOfValues << "] = { ";
		for (int kk = 0; kk < amountOfValues; kk++) {
			if (kk != amountOfValues - 1)
				outFileStream << values[kk] << ", ";
			else
				outFileStream << values[kk] << " }\n";;
		}
		outFileStream << "X0[" << amountOfInitialConditions << "] = { ";
		for (int kk = 0; kk < amountOfInitialConditions; kk++) {
			if (kk != amountOfInitialConditions - 1)
				outFileStream << initialConditions[kk] << ", ";
			else
				outFileStream << initialConditions[kk] << " }\n";
		}
		outFileStream << "CT =  " << tMax << "\n";
		outFileStream << "TT =" << transientTime << "\n";
		outFileStream << "h = " << h << "\n";
		outFileStream << "decimator = " << preScaller << "\n";
		outFileStream << "indexVar for peakfinder = " << writableVar << "\n";
		if (par_or_var == 1)
			outFileStream << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
		if (par_or_var == 0)
			outFileStream << "indexVar for estimation = " << indicesOfMutVars[0] << "\n";
		outFileStream << "start vlaue_1 = " << ranges[0] << ", stop vlaue_1 = " << ranges[1] << "\n";
	}
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


	// --- Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем ---
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// --- Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным ---
		// --- оставшемуся нерасчитанному куску ---
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts) - (nPtsLimiter * i);

		int blockSize;			// Переменная для хранения размера блока
		int minGridSize;		// Переменная для хранения минимального размера сетки
		int gridSize;			// Переменная для хранения сетки

		// --- Считаем, что один блок не может использовать больше чем 48КБ памяти ---
		// --- Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт ---
		// --- Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить ---
		// --- Учитваем, что в блоке не может быть больше 1024 потоков ---

		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		//blockSize = 10000 / ((amountOfInitialConditions + amountOfValues) * sizeof(numb));

		if (continuation_bif1D == 0) {

			blockSize = 32;
			gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

			// --------------------------------------------------
			// --- CUDA функция для расчета траектории систем ---
			// --------------------------------------------------


			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
				(nPts,						// Общее разрешение диаграммы - nPts
					nPtsLimiter,				// Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
					amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
					i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
					amountOfPointsForSkip,		// Количество точек для пропуска ( transientTime )
					1,							// Размерность ( диаграмма одномерная )
					d_ranges,					// Массив с диапазонами
					h,							// Шаг интегрирования
					d_indicesOfMutVars,			// Индексы изменяемых параметров
					d_initialConditions,		// Начальные условия
					amountOfInitialConditions,	// Количество начальных условий
					d_values,					// Параметры
					amountOfValues,				// Количество параметров
					amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
					preScaller,					// Множитель, который уменьшает время и объем расчетов
					writableVar,				// Индекс уравнения, по которому будем строить диаграмму
					maxValue,					// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
					d_data,						// Массив, где будет хранится траектория систем
					d_amountOfPeaks,
					par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

			// --------------------------------------------------
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


		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		// --- Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки ---
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, peakFinderCUDA, 0, blockSize_setup);
		//blockSize = blockSize > blockSize_setup ? blockSize_setup : blockSize;			// Не превышаем ограничение в 512 потока в блоке
		blockSize = 64;
		//printf(", %zu", blockSize);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		printf("Trajectories done\n");
		// -----------------------------------------
		// --- CUDA функция для нахождения пиков ---
		// -----------------------------------------

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

		// -----------------------------------------

		// --- Проверка на CUDA ошибки ---
		gpuGlobalErrorCheck();

		// --- Ждем пока все потоки завершат свою работу ---
		gpuErrorCheck(cudaDeviceSynchronize());

		////
		//gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_AkCOS, d_AkCOS, nPtsLimiter * nFreq * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		gpuErrorCheck(cudaMemcpy(h_BkSIN, d_BkSIN, nPtsLimiter * nFreq * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		outFileStream.open(OUT_FILE_PATH + "_" + "AkCOS.csv", std::ios::app);
		outFileStream << std::setprecision(set_precision);
		// --- Сохранение данных в файл ---
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
		// --- Сохранение данных в файл ---
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

	// ---------------------------
	// --- Освобождение памяти ---
	// ---------------------------

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
	// ---------------------------
}

