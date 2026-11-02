#include "host_internal.cuh"
__host__ void LLE1D(
	const numb	tMax,
	const numb	NT,
	const int		nPts,
	const numb	h,
	const numb	eps,
	const numb* initialConditions,
	const int		amountOfInitialConditions,
	const numb* ranges,
	const int* indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* values,
	const int		amountOfValues,
	std::string		OUT_FILE_PATH)
{
	// Количество точек, которое будет смоделировано одной системой во время нормализации NT
	size_t amountOfNT_points = NT / h;

	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = tMax / NT;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;																// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;																// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));						// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;																// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	numb* h_lleResult = new numb[nPtsLimiter];

	// Указатели на области памяти в GPU

	numb* d_ranges;				   // Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		   // Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	   // Указатель на массив с начальными условиями
	numb* d_values;				   // Указатель на массив с параметрами

	numb* d_lleResult;			   // Память для хранения конечного результата

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	gpuErrorCheck(cudaMalloc((void**)& d_lleResult, nPtsLimiter * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceilf((numb)nPts / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");

	data_export::legacy::write_lyap_config(
		outFileStream, set_precision, data_export::legacy::LyapKind::LLE1D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, NT, transientTime, h, eps, indicesOfMutVars, to_dbl(ranges, 2).data());
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("LLE 1D\n");
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

		// CUDA функция для расчета LLE

		LLEKernelCUDA << < gridSize, blockSize, (3 * amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
			(nPts,								// Общее разрешение
				nPtsLimiter,
				NT,
				tMax,
				amountOfPointsInBlock,				// Количество точек, занимаемое одной системой в "data"
				i * originalNPtsLimiter, 			// Количество уже посчитанных точек
				amountOfPointsForSkip,
				1, 									// Размерность
				d_ranges, 							// Массив, содержащий диапазоны перебираемого параметра
				h,
				eps,
				d_indicesOfMutVars, 				// Индексы изменяемых параметров
				d_initialConditions,				// Начальные условия
				amountOfInitialConditions,
				d_values, 							// Параметры
				amountOfValues,
				tMax / NT, 							// Количество итерация (вычисляется от tMax)
				1, 									// Множитель для ускорения расчетов
				writableVar,
				maxValue,
				d_lleResult);						// Результирующий массив

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		gpuErrorCheck(cudaMemcpy(h_lleResult, d_lleResult, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// Точность чисел с плавающей запятой
		outFileStream << std::setprecision(set_precision);

		// Сохранение данных в файл

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

	// Освобождение памяти

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

	data_export::legacy::write_lyap_config(
		outFileStream, set_precision, data_export::legacy::LyapKind::LLE2D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, NT, transientTime, h, eps, indicesOfMutVars, to_dbl(ranges, 4).data());
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

	data_export::legacy::write_lyap_config(
		outFileStream, set_precision, data_export::legacy::LyapKind::LS1D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, NT, transientTime, h, eps, indicesOfMutVars, to_dbl(ranges, 2).data());
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

	data_export::legacy::write_lyap_config(
		outFileStream, set_precision, data_export::legacy::LyapKind::LS2D, par_or_var,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditions, amountOfInitialConditions).data(), amountOfInitialConditions,
		tMax, NT, transientTime, h, eps, indicesOfMutVars, to_dbl(ranges, 4).data());
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

