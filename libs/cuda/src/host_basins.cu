#include "host_internal.cuh"
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
	std::string		OUT_FILE_PATH,
	const int blockSize_fixed)								// Размер блока CUDA (0 = дефолт blockSize_setup)
{
	// blockSize_fixed == 0 трактуем как «взять дефолт» — этот контракт объявлен
	// в hostLibrary.cuh. Без нормализации ноль доезжал до
	// gridSize = (nPtsLimiter + blockSize - 1) / blockSize и ронял расчёт делением на ноль.
	const int blockSizeBasins = blockSize_fixed > 0 ? blockSize_fixed : blockSize_setup;

	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = tMax / h / preScaller;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.9;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		
	printf("freeMemory: %zu\n", freeMemory);
	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * (amountOfPointsInBlock * (2)));

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	//nPtsLimiter = nPtsLimiter - (nPtsLimiter % blockSize_fixed);
	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Указатели на области памяти в GPU

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

	// Выделяем память в GPU

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

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	data_export::legacy::write_basins_config(
		outFileStream, set_precision, /*log_axes=*/false,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, transientTime, h, preScaller, eps, mult_peak, mult_interval,
		writableVar, indicesOfMutVars[0], indicesOfMutVars[1], to_dbl(ranges, 4).data());
	outFileStream.close();

#ifdef DEBUG
	//printf("Basins of attraction\n");
	//printf("nPtsLimiter : %zu\n", nPtsLimiter);
	//printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл
	outFileStream.open(OUT_FILE_PATH);
	// Точность чисел с плавающей запятой
	outFileStream << std::setprecision(set_precision);

	// Выводим в самое начало файла исследуемые диапазон
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

	printf("Basins, CT: %d", (int)tMax);
	printf(", res: %d x %d", nPts, nPts);
	printf(", amountOfIteration: %zu", amountOfIteration);
	printf(", nPtsLimiter: %zu", nPtsLimiter);
	printf(", blockSize: %d\n", blockSizeBasins);
	size_t startTime = std::clock();

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

		// Считаем, что один блок не может использовать больше чем 48КБ памяти
		// Одному потоку в блоке требуется (amountOfInitialConditions + amountOfValues) * sizeof(numb) байт
		// Производим расчет, какое максимальное количество потоков в блоке мы можем обечпечить
		// Учитваем, что в блоке не может быть больше 1024 потоков
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
		blockSize = blockSizeBasins;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для расчета траектории систем

			calculateDiscreteModelCUDA << <gridSize, blockSize, ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize >> >
			(	nPts,						// Общее разрешение диаграммы - nPts
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
				d_helpfulArray + (i * originalNPtsLimiter), // Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
				0);			

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA, 0, blockSize_setup);
		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		blockSize = blockSizeBasins;
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для нахождения пиков

		avgPeakFinderCUDA << <gridSize, blockSize >> >
				(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
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
	CUDA_dbscan(d_avgPeaks, d_avgIntervals, d_dbscanResult, d_helpfulArray, nPts * nPts, eps, blockSizeBasins);
	printf("DBSCAN done in: %zu ms\n\n", std::clock() - startTime);

	numb* h_avgPeaks = new numb[nPts * nPts];
	numb* h_avgIntervals = new numb[nPts * nPts];
	int* h_helpfulArray = new int[nPts * nPts];

	gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgPeaks,	 d_avgPeaks,	 nPts * nPts * sizeof(numb),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_avgIntervals, d_avgIntervals, nPts * nPts * sizeof(numb),  cudaMemcpyKind::cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_helpfulArray, d_helpfulArray, nPts * nPts * sizeof(int),		cudaMemcpyKind::cudaMemcpyDeviceToHost));

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

	// Сохранение средних значений пиков в файл

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

	// Сохранение средних значений межпиков в файл

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

	// Сохранение характеристик точек сетки начальных условий в файл

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

	// Освобождение памяти

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

}

__host__ void basinsOfAttraction_logAxes(
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

	freeMemory *= 0.95;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * amountOfInitialConditions * 0.5);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	//int* h_dbscanResult = new int[nPtsLimiter * sizeof(numb)];
	//numb* h_helpfulArray = new numb[nPts * nPts];			// Указатель на массив в GPU на вспомогательный массив

	// Указатели на области памяти в GPU

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

	// Выделяем память в GPU

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

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	data_export::legacy::write_basins_config(
		outFileStream, set_precision, /*log_axes=*/true,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, transientTime, h, preScaller, eps, mult_peak, mult_interval,
		writableVar, indicesOfMutVars[0], indicesOfMutVars[1], to_dbl(ranges, 4).data());
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("Basins of attraction\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// Точность чисел с плавающей запятой
	outFileStream << std::setprecision(set_precision);

	// Выводим в самое начало файла исследуемые диапазон
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

	// Основной цикл, который выполняет amountOfIteration расчетов для наборов размером nPtsLimiter систем
	for (int i = 0; i < amountOfIteration; ++i)
	{
		// Если мы на последней итерации, требуется подкорректировать nPtsLimiter и сделать его равным
		// оставшемуся нерасчитанному куску
		if (i == amountOfIteration - 1)
			nPtsLimiter = (nPts * nPts) - (nPtsLimiter * i);

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

		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для расчета траектории систем

		calculateDiscreteModelICCUDA_logAxes << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			(	nPts,										// Общее разрешение диаграммы - nPts
				nPtsLimiter,
				amountOfPointsInBlock,						// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,					// Количество уже посчитанных точек систем
				amountOfPointsForSkip,
				2,											// Размерность ( диаграмма одномерная )
				d_ranges,									// Массив с диапазонами
				h,
				d_indicesOfMutVars,							// Индексы изменяемых параметров
				d_initialConditions,						// Начальные условия
				amountOfInitialConditions,
				d_values,									// Параметры
				amountOfValues,
				amountOfPointsInBlock,						// Количество итераций ( равно количеству точек для одной системы )
				preScaller,
				writableVar,
				maxValue,
				d_data,										// Массив, где будет хранится траектория систем
				d_helpfulArray + (i * originalNPtsLimiter)	// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
			);												

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Используем встроенную функцию CUDA, для нахождения оптимальных настреок блока и сетки
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, avgPeakFinderCUDA_logMaximas, 0, blockSize_setup);

		blockSize = blockSize > 256 ? 256 : blockSize;		// Не превышаем ограничение в 1024 потока в блоке
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		// CUDA функция для нахождения пиков

		avgPeakFinderCUDA_logMaximas << <gridSize, blockSize >> >
			(d_data,						// Данные с траекториями систем
				amountOfPointsInBlock,		// Количество точек в одной траектории
				nPtsLimiter,
				d_avgPeaks + (i * originalNPtsLimiter),
				d_avgIntervals + (i * originalNPtsLimiter),
				d_data,						// Выходной массив, куда будут записаны значения пиков
				d_intervals,				// Межпиковый интервал
				d_helpfulArray + (i * originalNPtsLimiter),
				h * preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
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

	// Сохранение средних значений пиков в файл

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

	// Сохранение средних значений межпиков в файл

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

	// Сохранение характеристик точек сетки начальных условий в файл

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

	// Освобождение памяти

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

}

