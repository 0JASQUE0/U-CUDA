#pragma once

#include "cudaMacros.cuh"
#include "configCUDA.h"

// Host-only и runtime-API заголовки недоступны при NVRTC-компиляции.
// <math.h> NVRTC обрабатывает иначе чем nvcc — лучше держать в #ifndef и
// положиться на встроенный <cmath> (NVRTC сам выдаёт нужные device-функции).
#ifndef __CUDACC_RTC__
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#endif

// curand_kernel.h доступен и под NVRTC начиная с CUDA 11.0 (через
// -I CUDA_PATH/include, который ParametricEngine уже подаёт NVRTC).
// Нужен для LLE/LS-кернелов в cudaLibrary.cu (Wolf/Benettin perturbation init).
#include <curand_kernel.h>

__device__ __host__ void calculateDiscreteModelforFastSynchro(numb* X, numb* S1, numb* K, const numb* a, const numb h, const bool directionOfintegration = 1);

// dbscanCUDA_optimized / dbscan_optimized удалены как мёртвый код — см.
// примечание на их прежнем месте в cudaLibrary.cu.

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
	const int		amountOfIterations,
	const numb		maxValue,
	numb*			timedomain,
	numb*			output,
	const int		preScaller);

// swapRole: 0 = grid varies master IC (legacy default — initialConditions
// overridden per cell, initialConditionsSlave fixed); 1 = grid varies slave IC
// (initialConditionsSlave overridden per cell, initialConditions fixed).
//
// amountOfPointsForSkipMaster / ...Slave: транзиент (TT) в шагах h, свой для
// каждой системы. Досадка uncoupled (K=0) идёт per-cell и уже ПОСЛЕ подстановки
// сеточных координат: свипуемая сетка сторона садится на аттрактор из затравки
// своей ячейки, фиксированная — из одной точки (её путь от ячейки не зависит).
// 0 = без транзиента (legacy). size_t, а не int: transientTime/h легко
// переваливает за 2^31 (например TT=1e5 при h=1e-5), и на int это было бы UB.
__global__ void calculateDiscreteModelICCforFastSynchro(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		dimension,
	numb* ranges,
	const numb	h,
	int* indicesOfMutVars,
	numb* initialConditions,
	numb* initialConditionsSlave,
	const int		amountOfInitialConditions,
	const numb* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const numb	maxValue,
	const int		iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	numb* data,
	int* maxValueCheckerArray,
	numb* FastSynchroError,
	int		swapRole = 0,
	size_t	amountOfPointsForSkipMaster = 0,
	size_t	amountOfPointsForSkipSlave = 0);

__device__ numb loopCalculateDiscreteModelForFastSynchro_2(
	numb* x,
	const numb* initialConditionsSlave,
	const numb* values,
	const numb h,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	const int	iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	numb* data,
	const int startDataIndex,
	const int writeStep = 1);

__device__ numb loopCalculateDiscreteModelForFastSynchro(
	const numb* Xs,
	const numb* values,
	const numb h,
	const numb* K_Forward,
	const numb* K_Backward,
	const numb iterOfSynchr,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	numb* timedomain,
	const int startDataIndex);

/**  
 * Считает один шаг дискретной модели системы
 * и записывает новое состояние обратно в x
 * 
 * \param x			- состояние системы; на выходе — состояние после шага
 * \param values	- параметры системы
 * \param h			- шаг интегрирования
 */
__device__ __host__ __forceinline__  void calculateDiscreteModel(numb* x, const numb* values, const numb h);

__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h);
/**
 * Прогоняет несколько шагов модели, попутно складывая траекторию в "data" (если data != nullptr)
 * 
 * \param x						- состояние системы; на выходе — состояние после последнего шага
 * \param values				- параметры системы
 * \param h						- шаг интегрирования
 * \param amountOfIterations	- сколько шагов сделать
 * \param preScaller			- прореживание записи. В data попадает каждая 'preScaller'-я точка
 * \param writableVar			- индекс переменной в x[], которую пишем в data
 * \param maxValue				- порог расходимости. Если |x[writableVar]| > maxValue, функция вернёт false
 * \param data					- буфер для записи траектории
 * \param startDataIndex		- с какого индекса в data начинать запись
 * \param writeStep				- шаг записи в буфер (например, при 2 пишутся индексы 0, 2, 4, ...)
 * \return						- true, если решение не разошлось
 */

__device__  bool loopCalculateDiscreteModel(numb* x, const numb* values, 
	const numb h, const size_t amountOfIterations, const int amountOfX, const int preScaller=0,
	const int writableVar = 0, const numb maxValue = 0,
	numb* data = nullptr, const int startDataIndex = 0, 
	const int writeStep = 1);

//__device__ __host__ int loopCalculateDiscreteModel_int(
__device__ int loopCalculateDiscreteModel_int(
	numb* x, const numb* values,
	const numb h, const size_t amountOfIterations, const int amountOfX =3, const int preScaller = 0,
	const int writableVar = 0, const numb maxValue = 0,
	numb* data = nullptr, const size_t startDataIndex = 0,
	const int writeStep = 1);


/**
 * Ядро: моделирует ансамбль систем, разнесённых по шагу интегрирования
 *
 * \param amountOfThreads			- 
 * \param h							- шаг интегрирования
 * \param hSpecial					- 
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (сколько точек посчитает каждый поток)
 * \param writableVar				- индекс переменной, по которой пишем результат
 * \param data						- буфер, куда сложить посчитанные траектории
 * \return -
 */

__global__ void distributedCalculateDiscreteModelCUDA(
	const size_t		amountOfPointsForSkip,
	const int		amountOfThreads,
	const numb	h,
	const numb	hSpecial,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar = 0,
	numb*			data = nullptr);



/**
 * Ядро: считает траектории ансамбля систем
 * 
 * \param nPts						- всего точек свипа - nPts
 * \param nPtsLimiter				- сколько точек считаем за один запуск (размер чанка) - nPtsLimiter
 * \param sizeOfBlock				- точек в одной траектории ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- сколько точек свипа уже посчитано (смещение чанка)
 * \param amountOfPointsForSkip		- сколько точек пропустить как транзиент ( transientTime )
 * \param dimension					- размерность свипа ( сколько параметров меняем )
 * \param ranges					- границы свипа, по паре на измерение
 * \param h							- шаг интегрирования
 * \param indicesOfMutVars			- индексы варьируемых величин
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (сколько точек посчитает каждый поток)
 * \param preScaller				- прореживание: сколько шагов приходится на одну записанную точку
 * \param writableVar				- индекс переменной, по которой пишем результат
 * \param maxValue					- порог расходимости (по модулю); превысив его, решение считается разошедшимся
 * \param data						- буфер, куда сложить посчитанные траектории
 * \param maxValueCheckerArray		- вспомогательный массив: разошедшимся системам пишется '-1'
 * \param Par_or_Var				- 0 - варьируем начальные условия, 1 - варьируем параметры
 * \return -
 */
__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter,
	const size_t		sizeOfBlock,
	const size_t		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb* __restrict__			ranges,
	const numb	h,
	int* __restrict__			indicesOfMutVars,
	numb* __restrict__			initialConditions,
	const int		amountOfInitialConditions,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const size_t		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb*			data = nullptr,
	int*			maxValueCheckerArray = nullptr,
	const bool		Par_or_Var = 1,
	const int		hSweepAxis = -1,          // -1 = off, 0 = X axis sweeps h, 1 = Y axis sweeps h
	const numb	transientTime = 0,        // raw transient time; only read when hSweepAxis != -1
	const numb	tMax = 0,                 // raw computing time; only read when hSweepAxis != -1
	int*			actualIterations = nullptr, // per-thread actual sample count written to `data` (worst-case-sized buffer); read by peakFinderCUDA
	const int		logAxisMask = 0);         // bit i = axis slot i (X=0,Y=1) is log-distributed; supersedes LINEAR_OR_LOG_DISTRIB


// --------------------------------------------------------------------------



/**
 * Ядро: считает траектории ансамбля систем при свипе по шагу интегрирования
 *
 * \param nPts						- всего точек свипа - nPts
 * \param nPtsLimiter				- сколько точек считаем за один запуск (размер чанка) - nPtsLimiter
 * \param sizeOfBlock				- точек в одной траектории ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- сколько точек свипа уже посчитано (смещение чанка)
 * \param transientTime				- время транзиента ( transientTime )
 * \param dimension					- размерность свипа ( сколько величин меняем )
 * \param ranges					- границы свипа, по паре на измерение
 * \param h							- шаг интегрирования
 * \param indicesOfMutVars			- индексы варьируемых величин
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (сколько точек посчитает каждый поток)
 * \param preScaller				- прореживание: сколько шагов приходится на одну записанную точку
 * \param writableVar				- индекс переменной, по которой пишем результат
 * \param maxValue					- порог расходимости (по модулю); превысив его, решение считается разошедшимся
 * \param data						- буфер, куда сложить посчитанные траектории
 * \param maxValueCheckerArray		- вспомогательный массив: разошедшимся системам пишется '-1'
 * \return -
 */
__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const numb	transientTime,
	const int		dimension,
	numb*			ranges,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const numb	amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb* data = nullptr,
	int* maxValueCheckerArray = nullptr);

// --------------------------------------------------------------------------



/**
 * Ядро: считает траектории ансамбля систем при свипе по начальным условиям
 *
 * \param nPts						- всего точек свипа - nPts
 * \param nPtsLimiter				- сколько точек считаем за один запуск (размер чанка) - nPtsLimiter
 * \param sizeOfBlock				- точек в одной траектории ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- сколько точек свипа уже посчитано (смещение чанка)
 * \param amountOfPointsForSkip		- сколько точек пропустить как транзиент ( transientTime )
 * \param dimension					- размерность свипа ( сколько величин меняем )
 * \param ranges					- границы свипа, по паре на измерение
 * \param h							- шаг интегрирования
 * \param indicesOfMutVars			- индексы варьируемых величин
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (сколько точек посчитает каждый поток)
 * \param preScaller				- прореживание: сколько шагов приходится на одну записанную точку
 * \param writableVar				- индекс переменной, по которой пишем результат
 * \param maxValue					- порог расходимости (по модулю); превысив его, решение считается разошедшимся
 * \param data						- буфер, куда сложить посчитанные траектории
 * \param maxValueCheckerArray		- вспомогательный массив: разошедшимся системам пишется '-1'
 * \return -
 */
__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb*			ranges,
	const numb	h,
	int*			indicesOfMutVars,
	numb*			initialConditions,
	const int		amountOfInitialConditions,
	const numb*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb*			data = nullptr,
	int*			maxValueCheckerArray = nullptr);


__global__ void calculateDiscreteModelICCUDA_logAxes(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const size_t		amountOfPointsForSkip,
	const int		dimension,
	numb* ranges,
	const numb	h,
	int* indicesOfMutVars,
	numb* initialConditions,
	const int		amountOfInitialConditions,
	const numb* values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb* data = nullptr,
	int* maxValueCheckerArray = nullptr);
/**
 * Возвращает значение свипа по плоскому индексу idx
 * Пример:
 * Последовательность:
 * 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
 * 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 * 
 * getValueByIdx(7, 5, 1, 5, 0) = 3
 * getValueByIdx(7, 5, 1, 5, 1) = 2
 * getValueByIdx(7, 5, 1, 5, 2) = 1
 * 
 * \param idx			- плоский индекс точки в сетке свипа
 * \param nPts			- точек на одно измерение сетки
 * \param startRange	- левая граница диапазона
 * \param finishRange	- правая граница диапазона
 * \param valueNumber	- номер варьируемой величины (какое измерение сетки)
 * \return Value		- значение
 */
__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts, 
	const numb startRange, const numb finishRange, const int valueNumber);

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

/**
 * То же, что getValueByIdx, но узлы распределены логарифмически
 *
 * \param idx			- плоский индекс точки в сетке свипа
 * \param nPts			- точек на одно измерение сетки
 * \param startRange	- левая граница диапазона
 * \param finishRange	- правая граница диапазона
 * \param valueNumber	- номер варьируемой величины (какое измерение сетки)
 * \return Value		- значение
 */
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

// getValueByIdx для log-равномерной сетки (drop-in замена getValueByIdx) --
// требует startRange, finishRange > 0. Определение см. cudaLibrary.cu, рядом
// с getValueByIdxLog (эту декларацию сюда обязательно, иначе calculateDiscreteModelCUDA
// вызывает её раньше точки определения в .cu -- NVRTC компилирует файл линейно).
__device__ __host__ __forceinline__ numb getValueByIdx_log(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);



/**
 * Ищет пики в диапазоне [startDataIndex; startDataIndex + amountOfPoints] массива "data"
 * Результат складывается в outPeaks и timeOfPeaks ( если outPeaks != nullptr и timeOfPeaks != nullptr )
 * 
 * \param data				- массив с траекторией
 * \param startDataIndex	- с какого индекса начинать поиск
 * \param amountOfPoints	- сколько точек просмотреть
 * \param outPeaks			- куда сложить значения пиков
 * \param timeOfPeaks		- куда сложить моменты времени пиков
 * \param h					- шаг интегрирования
 * \return - Amount of found peaks
 */
__device__ __host__ numb globalPeakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints);

__device__ __host__ int peakFinder(numb* data, const size_t startDataIndex, const size_t amountOfPoints,
	numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h=0.0025);

__device__ __host__ void MeanAndMedianFreq(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq);
   					     
__device__ __host__ void MeanAndVariance(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval);

__global__ void DFT_custom(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* checkerArray, numb* AkCOS, numb* BkSIN, numb* rangesFreq, numb* window, int nFreq = 0, numb h = 0,
	const int logFreqAxis = 0);

/**
 * Ищет пики в массиве "data" параллельно, блок на траекторию
 * Результат складывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если они != nullptr )
 * 
 * \param data				- массив с траекториями
 * \param sizeOfBlock		- точек в одной траектории
 * \param amountOfBlocks	- сколько траекторий обрабатываем
 * \param amountOfPeaks		- куда сложить число найденных пиков
 * \param outPeaks			- куда сложить значения пиков
 * \param timeOfPeaks		- куда сложить моменты времени пиков
 * \param h					- шаг интегрирования
 */

__global__ void globalPeakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks);

__global__ void peakFinderCUDA( numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0.0025,
	const int* actualIterations = nullptr ); // per-thread valid prefix of `data`; nullptr = always scan full sizeOfBlock

__global__ void MeanAndMedianFreqCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq);

__global__ void MeanAndVarianceCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval);


__device__ __host__ numb sign(numb x);
__device__ __host__ numb psi_m(numb x, numb m, numb d, numb k, numb dmargin);
__device__ __host__ numb chua_multistep_extended(numb x, numb m, numb d, numb kslope, numb dmargin);
__device__ __host__ numb chua_inner(numb x, numb m, numb d, numb kslope, numb mu_val, numb M_val);
__device__ __host__ numb psi_two_scale(numb x, numb m, numb M, numb d, numb kslope);

/**
 * Ищет пики в массиве "data" параллельно, блок на траекторию (свип по шагу h)
 * Результат складывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если они != nullptr )
 *
 * \param data				- массив с траекториями
 * \param sizeOfBlock		- точек в одной траектории
 * \param amountOfBlocks	- сколько траекторий обрабатываем
 * \param amountOfPeaks		- куда сложить число найденных пиков
 * \param outPeaks			- куда сложить значения пиков
 * \param timeOfPeaks		- куда сложить моменты времени пиков
 * \param h					- шаг интегрирования
 */
__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0);



/**
 * Евклидово расстояние между двумя точками
 * 
 * \param x1 - x первой точки
 * \param y1 - y первой точки
 * \param x2 - x второй точки
 * \param y2 - y второй точки
 * \return - расстояние
 */
__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2);



/**
 * Кластеризация DBSCAN
 * 
 * \param data					- значения пиков одной траектории
 * \param intervals				- интервалы между пиками
 * \param helpfulArray			- вспомогательный буфер под промежуточные данные
 * \param startDataIndex		- с какого индекса начинать
 * \param amountOfPeaks			- сколько пиков на входе
 * \param sizeOfHelpfulArray	- размер вспомогательного буфера
 * \param idx					- индекс обрабатываемой системы
 * \param eps					- радиус окрестности DBSCAN
 * \param outData				- куда сложить число найденных кластеров
 */
// multPeak / multInterval — множители осей признаков перед кластеризацией
// (пик и межпиковый интервал). Задаются per-diagram из GUI; дефолты в
// объявлении dbscanCUDA ниже совпадают с константами configCUDA.h, поэтому
// вызовы без этих аргументов (hostLibrary.cu) считают как раньше.
__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData,
	const numb multPeak = mult_peak, const numb multInterval = mult_interval);



/**
 * Ядро DBSCAN
 * 
 * \param data				- массив с траекториями
 * \param sizeOfBlock		- точек в одной траектории
 * \param amountOfBlocks	- сколько траекторий обрабатываем
 * \param amountOfPeaks		- куда сложить число найденных пиков
 * \param intervals			- интервалы между пиками
 * \param helpfulArray		- вспомогательный буфер под промежуточные данные
 * \param eps				- радиус окрестности DBSCAN
 * \param outData			- куда сложить число найденных кластеров
 */
// Дефолты множителей = константы configCUDA.h: <<<>>>-вызовы без этих
// аргументов (hostLibrary.cu) сохраняют прежнее поведение. ВНИМАНИЕ: при
// запуске через driver API (cuLaunchKernel в parametric_engine.cpp) значения
// по умолчанию не подставляются — там массив аргументов обязан содержать все
// 10 параметров.
__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray, const numb eps, int* outData,
	const numb multPeak = mult_peak, const numb multInterval = mult_interval);



/**
 * Ядро LLE
 * 
 * \param nPts						- всего точек свипа
 * \param nPtsLimiter				- сколько точек считаем за один запуск
 * \param NT						- интервал нормировки
 * \param tMax						- время моделирования
 * \param sizeOfBlock				- точек в одной траектории
 * \param amountOfCalculatedPoints	- сколько точек свипа уже посчитано
 * \param amountOfPointsForSkip		- сколько точек пропустить как транзиент (transientTime)
 * \param dimension					- размерность свипа
 * \param ranges					- границы свипа, по паре на измерение
 * \param h							- шаг интегрирования
 * \param eps						- величина начального возмущения
 * \param indicesOfMutVars			- индексы варьируемых величин
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (считается из tMax)
 * \param preScaller				- прореживание записи
 * \param writableVar				- индекс переменной в x[], по которой считаем
 * \param maxValue					- порог расходимости
 * \param resultArray				- куда сложить показатель Ляпунова
 * \return -
 */
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
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb*			resultArray = nullptr,
	const int		hSweepAxis = -1,     // -1 = off, 0 = X axis sweeps h, 1 = Y axis sweeps h
	const numb	transientTime = 0,   // raw transient time; only read when hSweepAxis != -1
	const int		logAxisMask = 0);    // bit i = axis slot i is log-distributed; supersedes LINEAR_OR_LOG_DISTRIB



/**
 * Ядро LLE (свип по начальным условиям)
 *
 * \param nPts						- всего точек свипа
 * \param nPtsLimiter				- сколько точек считаем за один запуск
 * \param NT						- интервал нормировки
 * \param tMax						- время моделирования
 * \param sizeOfBlock				- точек в одной траектории
 * \param amountOfCalculatedPoints	- сколько точек свипа уже посчитано
 * \param amountOfPointsForSkip		- сколько точек пропустить как транзиент (transientTime)
 * \param dimension					- размерность свипа
 * \param ranges					- границы свипа, по паре на измерение
 * \param h							- шаг интегрирования
 * \param eps						- величина начального возмущения
 * \param indicesOfMutVars			- индексы варьируемых величин
 * \param initialConditions			- начальные условия
 * \param amountOfInitialConditions - количество начальных условий
 * \param values					- параметры системы
 * \param amountOfValues			- количество параметров
 * \param amountOfIterations		- количество итераций (считается из tMax)
 * \param preScaller				- прореживание записи
 * \param writableVar				- индекс переменной в x[], по которой считаем
 * \param maxValue					- порог расходимости
 * \param resultArray				- куда сложить показатель Ляпунова
 * \return -
 */
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
	const int		preScaller = 0,
	const int		writableVar = 0,
	const numb	maxValue = 0,
	numb*			resultArray = nullptr);



/**
 * Kernel for metric LS
 *
 * \param nPts - Amount of points
 * \param nPtsLimiter - Amount of points in one calculating
 * \param NT - Normalization time
 * \param tMax - Simulation time
 * \param sizeOfBlock - Size of one memory block in "data" array
 * \param amountOfCalculatedPoints - Amount of calculated points
 * \param amountOfPointsForSkip	- Amount of points for skip (depends on transit time)
 * \param dimension - Calculating dimension
 * \param ranges - Array with variable parameter ranges
 * \param h - Integration step
 * \param eps - Eps
 * \param indicesOfMutVars - Index of unknown variable
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \param amountOfIterations - Amount of iterations (nearly tMax)
 * \param preScaller - Amount of skip points in system. Each 'preScaller' point will be written
 * \param writableVar - Which variable from x[] will be written to the date
 * \param maxValue - Threshold signal level
 * \param resultArray - Result array
 * \return -
 */
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
	const int preScaller = 0,
	const int writableVar = 0,
	const numb maxValue = 0,
	numb* resultArray = nullptr,
	const int hSweepAxis = -1,     // -1 = off, 0 = X axis sweeps h, 1 = Y axis sweeps h
	const numb transientTime = 0,  // raw transient time; only read when hSweepAxis != -1
	const int logAxisMask = 0);    // bit i = axis slot i is log-distributed; supersedes LINEAR_OR_LOG_DISTRIB

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
	const int preScaller = 0,
	const int writableVar = 0,
	const numb maxValue = 0,
	numb* resultArray = nullptr);

/**
 * Считает средние пики и средние интервалы между ними в массиве "data", блок на траекторию
 * Результат складывается в "outAvgPeaks" и "AvgTimeOfPeaks" ( если они != nullptr )
 *
 * \param data				- массив с траекториями
 * \param sizeOfBlock		- точек в одной траектории
 * \param amountOfBlocks	- сколько траекторий обрабатываем
 * \param outAvgPeaks		- куда сложить средние значения пиков
 * \param AvgTimeOfPeaks	- куда сложить средние интервалы между пиками
 * \param h					- шаг интегрирования
 */



__global__ void avgPeakFinderCUDA_logMaximas(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h = 0);

// feature1/feature2 — BF_* коды из configCUDA.h: какую статистику записать
// в outAvgPeaks (feature1) и в AvgTimeOfPeaks (feature2). Дефолты дают
// pre-feature-selection поведение (mean peaks, mean intervals).
// mult1/mult2 — множители, применяемые ПОСЛЕ вычисления соответствующей
// фичи (для подстройки масштаба кластеризации в DBSCAN).
__global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker,
	numb h = 0,
	int feature1 = BF_FEATURE1_DEFAULT, int feature2 = BF_FEATURE2_DEFAULT,
	numb mult1   = mult_avg_peak,       numb mult2   = mult_avg_interval);

__global__ void avgPeakFinderCUDA_for2Dbif(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* PeaksAmount, int* systemCheker, numb h = 0);

__global__ void CUDA_dbscan_kernel(numb* data, numb* intervals, int* labels,
	const int amountOfData, const numb eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray);



__global__ void CUDA_dbscan_search_clear_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);



__global__ void CUDA_dbscan_search_fixed_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);

__global__ void CUDA_dbscan_search_unbound_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res);


