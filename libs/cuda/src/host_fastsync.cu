#include "host_internal.cuh"
__host__ void FastSynchro(
	const numb		tMax,
	const numb		transientTime,
	const numb		NTime,
	const numb*		values,
	const int		amountOfValues,
	const numb		h,
	const numb*		kForward,
	const numb*		kBackward,
	const numb*		initialConditionsMaster,			// Массив с начальными условиями мастера
	const numb*		initialConditionsSlave,				// Массив с начальными условиями слейва
	const int		amountOfInitialConditions,
	const numb		maxValue,
	const int		iterOfSynchr,
	const int		preScaller,
	std::string		OUT_FILE_PATH)
{

	// Количество точек, которое используется в окне синхронизации
	int amountOfNTPoints = NTime / h;

	// Общее количесвто точек в исходной траектории
	int amountOfCTPoints = tMax / h;

	// Количество точек переходного процесса
	int amountOfPointsForSkip = transientTime / h;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU
	size_t nPts = (amountOfCTPoints / preScaller);

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 1.0;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
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

	// Инициализация начальных условий
	for (int i = 0; i < amountOfInitialConditions; i++) {
		arrayZeros[i] = 0;
		Xm[i] = initialConditionsMaster[i];
		Xs[i] = initialConditionsSlave[i];
	}

	// Расчет переходного процесса
	for (size_t i = 0; i < amountOfPointsForSkip; i++) {
		calculateDiscreteModelforFastSynchro(Xm, Xm, arrayZeros, values, h, 1);
		//calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h, 1);
	}

	// Расчет исходной траектории
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
	// Указатели на области памяти в GPU

	numb* d_timeDomain;
	numb* d_output;
	numb* d_Xs;
	numb* d_values;
	numb* d_kForward;
	numb* d_kBackward;

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)&d_timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_output, nPts * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_Xs, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kForward, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_kBackward, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)&d_values, amountOfValues * sizeof(numb)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_timeDomain, timeDomain, amountOfInitialConditions * (amountOfCTPoints + amountOfNTPoints) * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_Xs, Xs, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;

	outFileStream.open(OUT_FILE_PATH + "_" + "config.csv");
	data_export::legacy::write_fastsync_config(
		outFileStream, set_precision, type_of_synch, error_estim,
		to_dbl(values, amountOfValues).data(), amountOfValues,
		to_dbl(initialConditionsMaster, amountOfInitialConditions).data(),
		to_dbl(initialConditionsSlave,  amountOfInitialConditions).data(),
		to_dbl(kForward,  amountOfInitialConditions).data(),
		to_dbl(kBackward, amountOfInitialConditions).data(),
		amountOfInitialConditions,
		iterOfSynchr, tMax, NTime, transientTime, h, preScaller);
	outFileStream.close();

	outFileStream.open(OUT_FILE_PATH);

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

		//blockSize = ceil((1*1024.0f * 4.0f) / (amountOfNTPoints * sizeof(numb)));
		//blockSize = ceil((1 * 1024.0f * 32.0f) / ((amountOfInitialConditions + amountOfValues) * sizeof(numb)));
		//blockSize = 10000 / ((5*amountOfInitialConditions + amountOfValues) * sizeof(numb));
		//blockSize = 5*amountOfNTPoints;
		//
		//gridSize = (nPtsLimiter + blockSize - 1) / blockSize;	// Расчет размера сетки ( формула является аналогом ceil() )

		blockSize = blockSize_setup;
		//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDiscreteModelforFastSynchroCUDA, 0, blockSize_setup);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;
		// CUDA функция для расчета траектории систем

			//calculateDiscreteModelforFastSynchroCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			//calculateDiscreteModelforFastSynchroCUDA << <4*1024, 16>> > //, 1*(amountOfInitialConditions + amountOfValues + amountOfNTPoints) * sizeof(numb)* 1 >> >

		calculateDiscreteModelforFastSynchroCUDA << < gridSize, blockSize >> >
			(
				nPts,
				nPtsLimiter,
				amountOfNTPoints,		//const int		sizeOfBlock,
				h,
				d_Xs,						//numb* initialConditions,
				amountOfInitialConditions,
				d_values,						//const numb* values,
				d_kForward,					//const numb* k_forward,
				d_kBackward,					//const numb* k_backward,
				iterOfSynchr,
				amountOfValues,
				amountOfNTPoints,							//const int		amountOfIterations,
				maxValue,
				d_timeDomain + (i * originalNPtsLimiter) * amountOfInitialConditions * preScaller,								//numb* timedomain,
				d_output + (i * originalNPtsLimiter),			//numb* output
				preScaller);

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
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
	const numb		NTime,
	const int		nPts,
	const numb*		values,
	const int		amountOfValues,
	const numb		h,
	const numb*		ranges,
	const int*		indicesOfMutVars,
	const numb*		kForward,
	const numb*		kBackward,
	const numb*		initialConditions,
	const numb*		initConditionsSlave,
	const int		amountOfInitialConditions,
	const numb		maxValue,
	const int		iterOfSynchr,
	const int		preScaller,
	std::string		OUT_FILE_PATH)
{
	// Количество точек, которое будет смоделировано одной системой с одним набором параметров
	int amountOfPointsInBlock = NTime / h / preScaller;

	// Количество точек, которое будет пропущено при моделировании системы
	// (amountOfPointsForSkip первых смоделированных точек не будет учитываться в расчетах)
	int amountOfPointsForSkip = 0;

	size_t freeMemory;											// Переменная для хранения свободного объема памяти в GPU
	size_t totalMemory;											// Переменная для хранения общего объема памяти в GPU

	gpuErrorCheck(cudaMemGetInfo(&freeMemory, &totalMemory));	// Получаем свободный и общий объемы памяти GPU

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfInitialConditions * amountOfPointsInBlock * amountOfInitialConditions);

	nPtsLimiter = nPtsLimiter > (nPts * nPts) ? (nPts * nPts) : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата

	numb* h_dbscanResult = new numb[nPtsLimiter * sizeof(numb)];

	// Указатели на области памяти в GPU

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

	// Выделяем память в GPU

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

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 4 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 2 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions,	   initialConditions,	 amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditionsSlave, initConditionsSlave, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kForward, kForward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_kBackward, kBackward, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)(nPts * nPts) / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	std::ofstream outFileStream;
	outFileStream.open(OUT_FILE_PATH);

#ifdef DEBUG
	printf("Bifurcation 2DIC\n");
	printf("nPtsLimiter : %zu\n", nPtsLimiter);
	printf("Amount of iterations %zu: \n", amountOfIteration);
#endif

	int stringCounter = 0; // Вспомогательная переменная для корректной записи матрицы в файл

	// Выводим в самое начало файла исследуемые диапазон
	if (outFileStream.is_open())
	{
		outFileStream << ranges[0] << " " << ranges[1] << "\n";
		outFileStream << ranges[2] << " " << ranges[3] << "\n";
	}

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

		// CUDA функция для расчета траектории систем

		calculateDiscreteModelICCforFastSynchro << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
			(nPts,							// Общее разрешение диаграммы - nPts
				nPtsLimiter,
				amountOfInitialConditions * amountOfPointsInBlock,		// Количество точек в одной системе ( tMax / h / preScaller ) 
				i * originalNPtsLimiter,	// Количество уже посчитанных точек систем
				2,							// Размерность ( диаграмма одномерная )
				d_ranges,					// Массив с диапазонами
				h,
				d_indicesOfMutVars,			// Индексы изменяемых параметров
				d_initialConditions,		// Начальные условия
				d_initialConditionsSlave,
				amountOfInitialConditions,
				d_values,					// Параметры
				amountOfValues,
				amountOfPointsInBlock,		// Количество итераций ( равно количеству точек для одной системы )
				preScaller,
				maxValue,
				iterOfSynchr,
				d_kForward,
				d_kBackward,
				d_data,						// Массив, где будет хранится траектория систем
				d_amountOfPeaks,
				d_dbscanResult);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		gpuErrorCheck(cudaMemcpy(h_dbscanResult, d_dbscanResult, nPtsLimiter * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		// Точность чисел с плавающей запятой
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

	gpuErrorCheck(cudaFree(d_amountOfPeaks));
	gpuErrorCheck(cudaFree(d_intervals));
	gpuErrorCheck(cudaFree(d_dbscanResult));
	gpuErrorCheck(cudaFree(d_helpfulArray));

	delete[] h_dbscanResult;

}

