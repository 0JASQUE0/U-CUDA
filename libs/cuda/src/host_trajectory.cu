#include "host_internal.cuh"
__host__ void TimeDomainCalculation(
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

	freeMemory *= 0.5;											// Ограничитель памяти (будем занимать лишь часть доступной GPU памяти)		

	// Расчет количества систем, которые мы сможем промоделировать параллельно в один момент времени
	// TODO Сделать расчет требуемой памяти
	size_t nPtsLimiter = freeMemory / (sizeof(numb) * amountOfPointsInBlock * 2);

	nPtsLimiter = nPtsLimiter > nPts ? nPts : nPtsLimiter;	// Если мы можем расчитать больше систем, чем требуется, то ставим ограничитель на максимум (nPts)

	size_t originalNPtsLimiter = nPtsLimiter;				// Запоминаем исходное значение nPts для дальнейших расчетов ( getValueByIdx )

	// Выделяем память для хранения конечного результата (пики и их количество для каждой системы)

	//numb* h_outPeaks = new numb[nPtsLimiter * amountOfPointsInBlock * sizeof(numb)];
	//int* h_amountOfPeaks = new int[nPtsLimiter * sizeof(int)];

	// Указатели на области памяти в GPU

	numb* d_data;					// Указатель на массив в памяти GPU для хранения траектории системы
	numb* d_ranges;				// Указатель на массив с диапазоном изменения переменной
	int* d_indicesOfMutVars;		// Указатель на массив с индексом изменяемой переменной в массиве values
	numb* d_initialConditions;	// Указатель на массив с начальными условиями
	numb* d_values;				// Указатель на массив с параметрами

	//numb* d_outPeaks;				// Указатель на массив в GPU с результирующими пиками биф. диаграммы
	//int* d_amountOfPeaks;		// Указатель на массив в GPU с кол-вом пиков в каждой системе.

	// Выделяем память в GPU

	gpuErrorCheck(cudaMalloc((void**)& d_data, nPts * amountOfPointsInBlock * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_ranges, 2 * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_indicesOfMutVars, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc((void**)& d_initialConditions, amountOfInitialConditions * sizeof(numb)));
	gpuErrorCheck(cudaMalloc((void**)& d_values, amountOfValues * sizeof(numb)));

	//gpuErrorCheck(cudaMalloc((void**)& d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb)));
	//gpuErrorCheck(cudaMalloc((void**)& d_amountOfPeaks, nPtsLimiter * sizeof(int)));

	// Копируем начальные входные параметры в память GPU

	gpuErrorCheck(cudaMemcpy(d_ranges, ranges, 2 * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_indicesOfMutVars, indicesOfMutVars, 1 * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialConditions * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_values, values, amountOfValues * sizeof(numb), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Расчет количества итераций для генерации бифуркационной диаграммы
	size_t amountOfIteration = (size_t)ceil((numb)nPts / (numb)nPtsLimiter);

	// Открытие выходного текстового файла для записи

	//std::ofstream outFileStream;
	//outFileStream.open(OUT_FILE_PATH);

	//static curandState *states = NULL;

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

		calculateDiscreteModelCUDA << <gridSize, blockSize, ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize >> >
		//calculateDiscreteModelCUDA_rand << <gridSize, blockSize >> >
			(	
				nPts,
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
				nullptr,
				par_or_var);			// Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему

		// Проверка на CUDA ошибки
		gpuGlobalErrorCheck();

		// Ждем пока все потоки завершат свою работу
		gpuErrorCheck(cudaDeviceSynchronize());

		// Копирование значений пиков и их количества из памяти GPU в оперативную память

		//gpuErrorCheck(cudaMemcpy(h_outPeaks, d_outPeaks, nPtsLimiter * amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));
		//gpuErrorCheck(cudaMemcpy(h_amountOfPeaks, d_amountOfPeaks, nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));

#ifdef DEBUG
		printf("Progress: %f\%\n", (100.0f / (numb)amountOfIteration) * (i + 1));
#endif
	}

	numb* h_data = new numb[amountOfPointsInBlock * nPts];

	gpuErrorCheck(cudaMemcpy(h_data, d_data, nPts* amountOfPointsInBlock * sizeof(numb), cudaMemcpyKind::cudaMemcpyDeviceToHost));

	// Точность чисел с плавающей запятой

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

				outFileStream << ", ";

			}
		} 
		outFileStream << "\n";
	}
	outFileStream.close();

	// Освобождение памяти
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

}

