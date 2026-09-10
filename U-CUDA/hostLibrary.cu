// Назначение параметров, общих для функций ниже (tMax, nPts, h, ranges,
// writableVar, maxValue, preScaller, eps и т.д.), описано одним словарём
// в hostLibrary.cuh — здесь не дублируется.
#include "hostLibrary.cuh"

// Writer'ы _config.csv: формат легаси-файлов живёт в data_export::legacy, а не
// здесь. Текст файлов от этого не изменился ни на байт — см. комментарий в
// data_export.h о том, почему опечатки и пробелы в нём сохранены.
#include "data_export.h"
#include <vector>

// numb[] -> double[] для legacy CSV-writer'ов.
//
// Они НАМЕРЕННО объявлены с const double* (см. комментарий над namespace legacy
// в data_export.h): при смене typedef numb на float вызов обязан не собраться,
// чтобы расширение типа было ВИДНО в коде, а не подменяло молча точность в
// файле. Растяжка сработала — вот это самое видимое расширение. Сами writer'ы
// и формат файлов не тронуты.
//
// Возвращает vector по значению; временный объект живёт до конца полного
// выражения, т.е. в течение всего вызова writer'а — .data() валиден.
// При numb == double это копия один-в-один (путь холодный: одна запись
// _config.csv на прогон, единицы элементов).
//
// n — сколько элементов ЧИТАЕТ writer, а не размер массива у вызывающего:
// у ranges это 4 для двухосевых диаграмм и 2 для однооосевых
// (data_export.cpp: two_axes в write_lyap_config).
static std::vector<double> to_dbl(const numb* p, int n)
{
	if (p == nullptr || n <= 0) return std::vector<double>();
	return std::vector<double>(p, p + n);
}

// Путь для сохранения результирующих файлов
//#define OUT_FILE_PATH "C:\\Users\\KiShiVi\\Desktop\\mat.csv"
//#define OUT_FILE_PATH "C:\\CUDA\\mat.csv"

// Директива, объявление которой выводит в консоль отладочные сообщения
#define DEBUG

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

			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
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
		size_t sharedMemNeeded = (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize;
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

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
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

			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
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

		calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb) * blockSize >> >
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

			calculateDiscreteModelCUDA << <gridSize, blockSize, (amountOfInitialConditions + amountOfValues) * sizeof(numb)* blockSize >> >
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
#include "rqa.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Потолок стороны матрицы. n = 4096 -> 4096^2 * 8 байт = 134 МБ на устройстве и столько же
// в out.dist. Выше упирается и видеопамять, и осмысленность. Превышение — ошибка, а НЕ
// молчаливое прореживание: числа метрик обязаны относиться к той матрице, которую видно.
constexpr int kRqaMaxPoints   = 4096;
constexpr int kRqaHistBins    = 4096;   // бинов на проход подбора eps (проходов два)
constexpr int kRqaRedThreads  = 256;    // фиксировано: под него рассчитан shared-массив редукции

__device__ __forceinline__ bool rqa_masked(int i, int j, int theiler)
{
	int d = i - j;
	if (d < 0) d = -d;
	return d <= theiler;
}

// D[i][j] = ||p_i - p_j|| в выбранной норме. pts — SoA: pts[k*n + i] = k-я координата i-й точки.
__global__ void rqaDistKernel(const numb* __restrict__ pts, int n, int d, int norm,
	numb* __restrict__ D)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n) return;

	numb acc = 0;
	if (norm == 0) {			// Euclidean
		for (int k = 0; k < d; ++k) {
			const numb t = __ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]);
			acc += t * t;
		}
		acc = sqrt(acc);
	}
	else if (norm == 1) {		// Maximum
		for (int k = 0; k < d; ++k) {
			const numb t = fabs(__ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]));
			if (t > acc) acc = t;
		}
	}
	else {						// Manhattan
		for (int k = 0; k < d; ++k)
			acc += fabs(__ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]));
	}
	D[(size_t)i * n + j] = acc;
}

// max(D) двухступенчатой редукцией: блок -> out[blockIdx.x], хвост досуммирует хост.
__global__ void rqaMaxKernel(const numb* __restrict__ D, size_t total, numb* __restrict__ out)
{
	__shared__ numb sm[kRqaRedThreads];
	const size_t stride = (size_t)gridDim.x * blockDim.x;
	numb m = 0;
	for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
		const numb v = D[t];
		if (v > m) m = v;
	}
	sm[threadIdx.x] = m;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) out[blockIdx.x] = sm[0];
}

// Гистограмма расстояний по [lo, hi) для подбора eps под целевой RR. Считаются ТОЛЬКО пары вне
// окна Тейлера — те же, по которым потом считается сам RR, иначе подобранный порог давал бы
// другой RR, чем показанный в таблице. hist[nbins] — сколько попало ниже lo, hist[nbins+1] — выше.
__global__ void rqaHistKernel(const numb* __restrict__ D, int n, int theiler,
	numb lo, numb hi, int nbins, unsigned long long* __restrict__ hist)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n) return;
	if (rqa_masked(i, j, theiler)) return;

	const numb v = D[(size_t)i * n + j];
	if (v < lo) { atomicAdd(&hist[nbins], 1ULL); return; }
	const int b = (int)((v - lo) / (hi - lo) * nbins);
	if (b >= nbins) { atomicAdd(&hist[nbins + 1], 1ULL); return; }
	atomicAdd(&hist[b < 0 ? 0 : b], 1ULL);
}

// Диагональные линии: поток на диагональ k = t - (n-1). Диагонали внутри окна Тейлера (включая
// LOI) не обрабатываются вовсе — их вклад не должен попасть ни в одну метрику.
// diagCount[t] — число рекуррентных точек на диагонали (нужно для RR и TREND).
__global__ void rqaDiagKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ histDiag, unsigned int* __restrict__ diagCount)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= 2 * n - 1) return;
	const int k = t - (n - 1);
	const int ak = k < 0 ? -k : k;
	if (ak <= theiler) return;

	const int i0 = (k >= 0) ? 0 : -k;
	const int j0 = (k >= 0) ? k : 0;
	const int len = n - ak;

	int run = 0;
	unsigned int cnt = 0;
	for (int s = 0; s < len; ++s) {
		const bool rec = (D[(size_t)(i0 + s) * n + (j0 + s)] <= eps);
		if (rec) { ++run; ++cnt; }
		else if (run > 0) { atomicAdd(&histDiag[run], 1u); run = 0; }
	}
	if (run > 0) atomicAdd(&histDiag[run], 1u);
	diagCount[t] = cnt;
}

// Вертикальные линии: поток на столбец. Состояния клетки: 1 = рекуррентна, 0 = белая,
// 2 = исключена окном Тейлера (или конец столбца). Исключённая клетка РВЁТ и чёрную серию, и
// белый промежуток, и цепочку времён возврата — иначе полоса маски читалась бы как настоящий
// белый интервал и завышала бы T1/T2/W.
__global__ void rqaVertKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ histVert, unsigned int* __restrict__ histWhite,
	unsigned long long* __restrict__ acc)   // [0]=sum T1, [1]=cnt T1, [2]=sum T2, [3]=cnt T2
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n) return;

	int run = 0;			// длина текущей чёрной серии
	int gap = -1;			// длина белого промежутка; -1 = слева ещё не было опорной чёрной точки
	int prevRec = -1;		// индекс предыдущей рекуррентной точки
	int prevStart = -1;		// индекс начала предыдущего чёрного блока
	unsigned long long s1 = 0, c1 = 0, s2 = 0, c2 = 0;

	for (int i = 0; i <= n; ++i) {
		int st;
		if (i == n)								st = 2;
		else if (rqa_masked(i, j, theiler))		st = 2;
		else									st = (D[(size_t)i * n + j] <= eps) ? 1 : 0;

		if (st == 1) {
			if (run == 0) {
				if (gap > 0) atomicAdd(&histWhite[gap], 1u);
				if (prevStart >= 0) { s2 += (unsigned long long)(i - prevStart); ++c2; }
				prevStart = i;
			}
			if (prevRec >= 0) { s1 += (unsigned long long)(i - prevRec); ++c1; }
			prevRec = i;
			++run;
			gap = 0;
		}
		else if (st == 0) {
			if (run > 0) { atomicAdd(&histVert[run], 1u); run = 0; }
			if (gap >= 0) ++gap;
		}
		else {
			if (run > 0) { atomicAdd(&histVert[run], 1u); run = 0; }
			gap = -1; prevRec = -1; prevStart = -1;
		}
	}
	atomicAdd(&acc[0], s1); atomicAdd(&acc[1], c1);
	atomicAdd(&acc[2], s2); atomicAdd(&acc[3], c2);
}

// Степень вершины сети рекуррентности (строка матрицы за вычетом окна Тейлера).
__global__ void rqaDegKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ deg)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	unsigned int k = 0;
	for (int j = 0; j < n; ++j)
		if (!rqa_masked(i, j, theiler) && D[(size_t)i * n + j] <= eps) ++k;
	deg[i] = k;
}

// Треугольники сети: поток на пару i<j. Для существующего ребра (i,j) считаем общих соседей —
// это число треугольников на ребре. triNode[i] в итоге = 2 * (число треугольников при вершине i).
__global__ void rqaTriKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned long long* __restrict__ triNode)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n || j <= i) return;
	if (rqa_masked(i, j, theiler) || D[(size_t)i * n + j] > eps) return;

	unsigned long long t = 0;
	for (int k = 0; k < n; ++k) {
		if (k == i || k == j) continue;
		if (rqa_masked(i, k, theiler) || D[(size_t)i * n + k] > eps) continue;
		if (rqa_masked(j, k, theiler) || D[(size_t)j * n + k] > eps) continue;
		++t;
	}
	if (t) { atomicAdd(&triNode[i], t); atomicAdd(&triNode[j], t); }
}

// Энтропия Шеннона распределения длин линий по гистограмме [from..n].
// norm == true -> делится на ln(число непустых бинов), т.е. приводится к [0,1] (так принято
// определять RTE). Возвращает NaN, если линий нужной длины нет вовсе.
double rqa_entropy(const std::vector<unsigned int>& hist, int from, bool norm)
{
	double total = 0.0;
	int used = 0;
	for (size_t l = (size_t)from; l < hist.size(); ++l)
		if (hist[l]) { total += hist[l]; ++used; }
	if (total <= 0.0) return std::numeric_limits<double>::quiet_NaN();

	double e = 0.0;
	for (size_t l = (size_t)from; l < hist.size(); ++l) {
		if (!hist[l]) continue;
		const double p = hist[l] / total;
		e -= p * std::log(p);
	}
	if (!norm) return e;
	return (used > 1) ? e / std::log((double)used) : 0.0;
}

}  // namespace

bool rqa::compute(const std::vector<std::vector<double>>& traj, double dt_traj,
	const rqa::Config& cfg, rqa::Result& out)
{
	using rqa::Source;
	using rqa::Norm;
	using rqa::EpsMode;

	out = rqa::Result();
	const double kNaN = std::numeric_limits<double>::quiet_NaN();

	// ---------------- валидация входа ----------------
	if (cfg.source == Source::None) {
		out.error = "RQA: choose the state-vector source (full state or delay embedding)";
		return false;
	}
	if ((int)traj.size() < 8 || traj[0].empty()) { out.error = "RQA: trajectory too short"; return false; }
	if (!(dt_traj > 0.0)) { out.error = "RQA: non-positive sample step"; return false; }
	if (cfg.points < 16 || cfg.points > kRqaMaxPoints) {
		out.error = "RQA: RP points must be within 16.." + std::to_string(kRqaMaxPoints);
		return false;
	}
	if (cfg.theiler < 0) { out.error = "RQA: Theiler window must be >= 0"; return false; }
	if (cfg.l_min < 1 || cfg.v_min < 1) { out.error = "RQA: l_min and v_min must be >= 1"; return false; }

	const int nsteps = (int)traj.size();
	const int dim = (int)traj[0].size();

	int d = 0, avail = 0;
	if (cfg.source == Source::StateVector) {
		d = dim;
		avail = nsteps;
	}
	else {
		if (cfg.m < 1) { out.error = "RQA: embedding dimension must be >= 1"; return false; }
		if (cfg.tau < 1) { out.error = "RQA: embedding delay must be >= 1"; return false; }
		if (cfg.var < 0 || cfg.var >= dim) { out.error = "RQA: embedding variable out of range"; return false; }
		d = cfg.m;
		avail = nsteps - (cfg.m - 1) * cfg.tau;
		if (avail < 16) { out.error = "RQA: embedding window (m-1)*tau is longer than the trajectory"; return false; }
	}

	const int n = std::min(cfg.points, avail);
	if (n < 16) { out.error = "RQA: not enough samples"; return false; }
	if (cfg.theiler >= n) { out.error = "RQA: Theiler window >= matrix size"; return false; }

	// ---------------- выборка точек (SoA) ----------------
	// Прореживаем равномерно до n. Шаг между отсчётами постоянен, поэтому dt честный; он выводится
	// в UI, чтобы длины линий (они В ОТСЧЁТАХ) можно было перевести в секунды.
	const double span = (double)(avail - 1) / (double)(n - 1);
	std::vector<numb> pts((size_t)d * n);
	for (int i = 0; i < n; ++i) {
		const int base = (int)std::llround((double)i * span);
		if (cfg.source == Source::StateVector) {
			const std::vector<double>& p = traj[(size_t)base];
			for (int k = 0; k < d; ++k)
				pts[(size_t)k * n + i] = (numb)(k < (int)p.size() ? p[k] : 0.0);
		}
		else {
			for (int k = 0; k < d; ++k) {
				const std::vector<double>& p = traj[(size_t)base + (size_t)k * cfg.tau];
				pts[(size_t)k * n + i] = (numb)(cfg.var < (int)p.size() ? p[cfg.var] : 0.0);
			}
		}
	}
	out.n = n;
	out.dt = span * dt_traj;
	out.t0 = 0.0;
	out.t1 = (double)(avail - 1) * dt_traj;

	for (size_t t = 0; t < pts.size(); ++t)
		if (!std::isfinite((double)pts[t])) { out.error = "RQA: trajectory contains nan/inf"; return false; }

	// ---------------- устройство ----------------
	numb* d_pts = nullptr; numb* d_D = nullptr; numb* d_blockMax = nullptr;
	unsigned long long* d_hist = nullptr; unsigned long long* d_acc = nullptr;
	unsigned long long* d_triNode = nullptr;
	unsigned int* d_histDiag = nullptr; unsigned int* d_histVert = nullptr;
	unsigned int* d_histWhite = nullptr; unsigned int* d_diagCount = nullptr;
	unsigned int* d_deg = nullptr;

	auto free_all = [&]() {
		cudaFree(d_pts); cudaFree(d_D); cudaFree(d_blockMax);
		cudaFree(d_hist); cudaFree(d_acc); cudaFree(d_triNode);
		cudaFree(d_histDiag); cudaFree(d_histVert); cudaFree(d_histWhite);
		cudaFree(d_diagCount); cudaFree(d_deg);
	};

#define RQA_CHECK(call, what)                                                            \
	do {                                                                                 \
		const cudaError_t _rc = (call);                                                  \
		if (_rc != cudaSuccess) {                                                        \
			out.error = std::string("RQA: ") + (what) + ": " + cudaGetErrorString(_rc);  \
			free_all(); return false;                                                    \
		}                                                                                \
	} while (0)
#define RQA_LAUNCH(what)                                                                 \
	do {                                                                                 \
		RQA_CHECK(cudaGetLastError(), what);                                             \
		RQA_CHECK(cudaDeviceSynchronize(), what);                                        \
	} while (0)

	const size_t cells = (size_t)n * (size_t)n;
	const dim3 blk2(16, 16);
	const dim3 grd2((n + blk2.x - 1) / blk2.x, (n + blk2.y - 1) / blk2.y);

	RQA_CHECK(cudaMalloc(&d_pts, pts.size() * sizeof(numb)), "alloc points");
	RQA_CHECK(cudaMalloc(&d_D, cells * sizeof(numb)), "alloc distance matrix");
	RQA_CHECK(cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(numb), cudaMemcpyHostToDevice), "upload points");

	rqaDistKernel<<<grd2, blk2>>>(d_pts, n, d, (int)cfg.norm, d_D);
	RQA_LAUNCH("distance matrix");

	// ---------------- max(D) ----------------
	int redBlocks = (int)((cells + kRqaRedThreads - 1) / kRqaRedThreads);
	if (redBlocks > 1024) redBlocks = 1024;
	RQA_CHECK(cudaMalloc(&d_blockMax, (size_t)redBlocks * sizeof(numb)), "alloc reduction");
	rqaMaxKernel<<<redBlocks, kRqaRedThreads>>>(d_D, cells, d_blockMax);
	RQA_LAUNCH("max distance");
	{
		std::vector<numb> bm((size_t)redBlocks);
		RQA_CHECK(cudaMemcpy(bm.data(), d_blockMax, bm.size() * sizeof(numb), cudaMemcpyDeviceToHost), "read reduction");
		numb mx = 0;
		for (size_t t = 0; t < bm.size(); ++t) if (bm[t] > mx) mx = bm[t];
		out.dist_max = (double)mx;
	}
	if (!(out.dist_max > 0.0)) {
		out.error = "RQA: all sampled points coincide (max distance is 0)";
		free_all(); return false;
	}

	// ---------------- порог eps ----------------
	const long long w = cfg.theiler;
	const long long npairs = (long long)n * n - (2 * w + 1) * (long long)n + w * (w + 1);
	if (npairs <= 0) { out.error = "RQA: Theiler window leaves no pairs"; free_all(); return false; }

	numb eps = 0;
	if (cfg.eps_mode == EpsMode::Absolute) {
		eps = cfg.eps;
	}
	else if (cfg.eps_mode == EpsMode::FracMaxDist) {
		eps = (numb)(cfg.eps_frac * out.dist_max);
	}
	else if (cfg.eps_mode == EpsMode::FracStd) {
		// СКО облака точек от центра масс — тот масштаб, в котором в статьях пишут "eps = 0.1 sigma".
		std::vector<double> mean((size_t)d, 0.0);
		for (int k = 0; k < d; ++k) {
			double s = 0.0;
			for (int i = 0; i < n; ++i) s += (double)pts[(size_t)k * n + i];
			mean[(size_t)k] = s / n;
		}
		double var = 0.0;
		for (int k = 0; k < d; ++k)
			for (int i = 0; i < n; ++i) {
				const double t = (double)pts[(size_t)k * n + i] - mean[(size_t)k];
				var += t * t;
			}
		eps = (numb)(cfg.eps_frac * std::sqrt(var / n));
	}
	else {
		// TargetRR: квантиль распределения расстояний. Два прохода гистограммой вместо сортировки
		// n^2 значений; итоговое разрешение по eps ~ dist_max / 4096^2.
		if (!(cfg.target_rr > 0.0) || !(cfg.target_rr < 1.0)) {
			out.error = "RQA: target RR must be within (0, 1)"; free_all(); return false;
		}
		RQA_CHECK(cudaMalloc(&d_hist, (size_t)(kRqaHistBins + 2) * sizeof(unsigned long long)), "alloc histogram");
		const long long want = (long long)std::llround(cfg.target_rr * (double)npairs);
		double lo = 0.0, hi = out.dist_max;
		long long below = 0;
		for (int pass = 0; pass < 2; ++pass) {
			RQA_CHECK(cudaMemset(d_hist, 0, (size_t)(kRqaHistBins + 2) * sizeof(unsigned long long)), "clear histogram");
			rqaHistKernel<<<grd2, blk2>>>(d_D, n, cfg.theiler, (numb)lo, (numb)hi, kRqaHistBins, d_hist);
			RQA_LAUNCH("distance histogram");
			std::vector<unsigned long long> h((size_t)kRqaHistBins + 2);
			RQA_CHECK(cudaMemcpy(h.data(), d_hist, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read histogram");

			long long cum = below;
			int b = kRqaHistBins - 1;
			for (int t = 0; t < kRqaHistBins; ++t) {
				cum += (long long)h[(size_t)t];
				if (cum >= want) { b = t; break; }
			}
			// below — пары строго левее нового окна: второй проход обязан продолжать ту же
			// накопленную сумму, а не начинать считать RR заново.
			for (int t = 0; t < b; ++t) below += (long long)h[(size_t)t];
			const double step = (hi - lo) / kRqaHistBins;
			lo = lo + step * b;
			hi = lo + step;
			eps = (numb)hi;
		}
	}
	if (!(eps > 0.0)) { out.error = "RQA: eps resolved to a non-positive value"; free_all(); return false; }
	out.eps_used = (double)eps;

	// ---------------- линии ----------------
	RQA_CHECK(cudaMalloc(&d_histDiag, (size_t)(n + 1) * sizeof(unsigned int)), "alloc diagonal histogram");
	RQA_CHECK(cudaMalloc(&d_histVert, (size_t)(n + 1) * sizeof(unsigned int)), "alloc vertical histogram");
	RQA_CHECK(cudaMalloc(&d_histWhite, (size_t)(n + 1) * sizeof(unsigned int)), "alloc white-line histogram");
	RQA_CHECK(cudaMalloc(&d_diagCount, (size_t)(2 * n - 1) * sizeof(unsigned int)), "alloc diagonal counts");
	RQA_CHECK(cudaMalloc(&d_acc, 4 * sizeof(unsigned long long)), "alloc recurrence-time accumulators");
	RQA_CHECK(cudaMemset(d_histDiag, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear diagonal histogram");
	RQA_CHECK(cudaMemset(d_histVert, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear vertical histogram");
	RQA_CHECK(cudaMemset(d_histWhite, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear white-line histogram");
	RQA_CHECK(cudaMemset(d_diagCount, 0, (size_t)(2 * n - 1) * sizeof(unsigned int)), "clear diagonal counts");
	RQA_CHECK(cudaMemset(d_acc, 0, 4 * sizeof(unsigned long long)), "clear accumulators");

	rqaDiagKernel<<<(2 * n - 1 + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_histDiag, d_diagCount);
	RQA_LAUNCH("diagonal lines");
	rqaVertKernel<<<(n + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_histVert, d_histWhite, d_acc);
	RQA_LAUNCH("vertical lines");

	std::vector<unsigned int> hDiag((size_t)n + 1), hVert((size_t)n + 1), hWhite((size_t)n + 1);
	std::vector<unsigned int> dCount((size_t)(2 * n - 1));
	unsigned long long acc[4] = { 0, 0, 0, 0 };
	RQA_CHECK(cudaMemcpy(hDiag.data(), d_histDiag, hDiag.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read diagonal histogram");
	RQA_CHECK(cudaMemcpy(hVert.data(), d_histVert, hVert.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read vertical histogram");
	RQA_CHECK(cudaMemcpy(hWhite.data(), d_histWhite, hWhite.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read white-line histogram");
	RQA_CHECK(cudaMemcpy(dCount.data(), d_diagCount, dCount.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read diagonal counts");
	RQA_CHECK(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost), "read accumulators");

	// ---------------- сетевые меры (опционально) ----------------
	double clustering = kNaN, transitivity = kNaN;
	bool network_valid = false;
	if (cfg.network_measures) {
		RQA_CHECK(cudaMalloc(&d_deg, (size_t)n * sizeof(unsigned int)), "alloc degrees");
		RQA_CHECK(cudaMalloc(&d_triNode, (size_t)n * sizeof(unsigned long long)), "alloc triangle counts");
		RQA_CHECK(cudaMemset(d_triNode, 0, (size_t)n * sizeof(unsigned long long)), "clear triangle counts");
		rqaDegKernel<<<(n + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_deg);
		RQA_LAUNCH("network degrees");
		rqaTriKernel<<<grd2, blk2>>>(d_D, n, eps, cfg.theiler, d_triNode);
		RQA_LAUNCH("network triangles");

		std::vector<unsigned int> deg((size_t)n);
		std::vector<unsigned long long> tri((size_t)n);
		RQA_CHECK(cudaMemcpy(deg.data(), d_deg, deg.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read degrees");
		RQA_CHECK(cudaMemcpy(tri.data(), d_triNode, tri.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read triangle counts");

		// triNode[i] = 2 * (треугольников при вершине i) -> C_i = triNode[i] / (k_i*(k_i-1)).
		double csum = 0.0; int cnodes = 0;
		double triSum = 0.0, tripleSum = 0.0;
		for (int i = 0; i < n; ++i) {
			const double k = (double)deg[(size_t)i];
			if (k < 2.0) continue;
			csum += (double)tri[(size_t)i] / (k * (k - 1.0));
			++cnodes;
			triSum += (double)tri[(size_t)i] * 0.5;
			tripleSum += k * (k - 1.0) * 0.5;
		}
		// transitivity = 3 * (число треугольников) / (число связных троек). triSum здесь — уже
		// СУММА ПО ВЕРШИНАМ T_i, а она равна 3 * (число треугольников) сама по себе (каждый
		// треугольник даёт вклад в три свои вершины), поэтому множителя 3 тут быть не должно.
		// Проверка: у полного графа k_i = n-1, T_i = (n-1)(n-2)/2, и отношение даёт ровно 1.
		clustering = (cnodes > 0) ? csum / cnodes : kNaN;
		transitivity = (tripleSum > 0.0) ? triSum / tripleSum : kNaN;
		network_valid = true;
	}

	// ---------------- матрица наружу ----------------
	// out.dist — double, потому что его ждёт HeatmapView::render. Ветка else существует ради того,
	// чтобы при смене typedef numb на float расширение точности было ВИДНО здесь, а не молча
	// происходило внутри cudaMemcpy с несовпадающим размером элемента.
	out.dist.resize(cells);
	if (sizeof(numb) == sizeof(double)) {
		RQA_CHECK(cudaMemcpy(out.dist.data(), d_D, cells * sizeof(numb), cudaMemcpyDeviceToHost), "read distance matrix");
	}
	else {
		std::vector<numb> tmp(cells);
		RQA_CHECK(cudaMemcpy(tmp.data(), d_D, cells * sizeof(numb), cudaMemcpyDeviceToHost), "read distance matrix");
		for (size_t t = 0; t < cells; ++t) out.dist[t] = (double)tmp[t];
	}
	free_all();
#undef RQA_LAUNCH
#undef RQA_CHECK

	// ---------------- метрики ----------------
	rqa::Metrics& M = out.metrics;

	unsigned long long nrec = 0;
	for (size_t t = 0; t < dCount.size(); ++t) nrec += dCount[t];
	M.RR = (double)nrec / (double)npairs;

	// Диагонали. Знаменатель DET — все рекуррентные точки вне окна Тейлера, т.е. ровно nrec:
	// каждая такая точка принадлежит ровно одной диагональной линии.
	{
		double sumAll = 0.0, sumMin = 0.0, cntMin = 0.0;
		int lmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hDiag[(size_t)l];
			if (p <= 0.0) continue;
			sumAll += (double)l * p;
			if (l >= cfg.l_min) { sumMin += (double)l * p; cntMin += p; lmax = l; }
		}
		M.DET   = (sumAll > 0.0) ? sumMin / sumAll : kNaN;
		M.L     = (cntMin > 0.0) ? sumMin / cntMin : kNaN;
		M.L_max = (lmax > 0) ? (double)lmax : kNaN;
		M.DIV   = (lmax > 0) ? 1.0 / (double)lmax : kNaN;
		M.ENTR  = rqa_entropy(hDiag, cfg.l_min, false);
		M.RATIO = (M.RR > 0.0) ? M.DET / M.RR : kNaN;
	}

	// Вертикали.
	{
		double sumAll = 0.0, sumMin = 0.0, cntMin = 0.0;
		int vmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hVert[(size_t)l];
			if (p <= 0.0) continue;
			sumAll += (double)l * p;
			if (l >= cfg.v_min) { sumMin += (double)l * p; cntMin += p; vmax = l; }
		}
		M.LAM    = (sumAll > 0.0) ? sumMin / sumAll : kNaN;
		M.TT     = (cntMin > 0.0) ? sumMin / cntMin : kNaN;
		M.V_max  = (vmax > 0) ? (double)vmax : kNaN;
		M.V_ENTR = rqa_entropy(hVert, cfg.v_min, false);
	}

	// Белые вертикали и времена возврата. Всё В ОТСЧЁТАХ выборки (умножить на out.dt для секунд).
	{
		double sum = 0.0, cnt = 0.0;
		int wmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hWhite[(size_t)l];
			if (p <= 0.0) continue;
			sum += (double)l * p; cnt += p; wmax = l;
		}
		M.W     = (cnt > 0.0) ? sum / cnt : kNaN;
		M.W_max = (wmax > 0) ? (double)wmax : kNaN;
		M.RTE   = rqa_entropy(hWhite, 1, true);
	}
	M.T1 = (acc[1] > 0) ? (double)acc[0] / (double)acc[1] : kNaN;
	M.T2 = (acc[3] > 0) ? (double)acc[2] / (double)acc[3] : kNaN;

	// TREND: наклон линейной регрессии RR по номеру диагонали (верхний треугольник). Последние
	// 10% диагоналей отброшены — там на линию приходятся единицы точек и RR_k шумит.
	{
		const int kFrom = cfg.theiler + 1;
		const int kTo = (int)std::floor(0.9 * (double)(n - 1));
		double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0, cnt = 0.0;
		for (int k = kFrom; k <= kTo; ++k) {
			const double rr = (double)dCount[(size_t)(k + n - 1)] / (double)(n - k);
			sx += k; sy += rr; sxy += (double)k * rr; sxx += (double)k * (double)k; cnt += 1.0;
		}
		if (cnt >= 2.0) {
			const double den = sxx - sx * sx / cnt;
			M.TREND = (den != 0.0) ? (sxy - sx * sy / cnt) / den : kNaN;
		}
		else M.TREND = kNaN;
	}

	M.clustering    = clustering;
	M.transitivity  = transitivity;
	M.network_valid = network_valid;

	out.ok = true;
	return true;
}
