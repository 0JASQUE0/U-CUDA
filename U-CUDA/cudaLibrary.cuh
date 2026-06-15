#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaMacros.cuh"
#include "configCUDA.h"
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

__device__ __host__ void calculateDiscreteModelforFastSynchro(numb* X, numb* S1, numb* K, const numb* a, const numb h, const bool directionOfintegration = 1);

__global__ void dbscanCUDA_optimized(
	const numb* __restrict__ data,
	const size_t sizeOfBlock,
	const int amountOfBlocks,
	const int* __restrict__ amountOfPeaks,
	const numb* __restrict__ intervals,
	const numb eps,
	int* __restrict__ outData);

__device__ int dbscan_optimized(
	const numb* __restrict__ data,       // Амплитуды пиков (только чтение)
	const numb* __restrict__ intervals,  // Межпиковые интервалы (только чтение)
	const int amountOfPeaks,
	const numb eps,
	int* __restrict__ labels);

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
	numb* FastSynchroError);

__device__ numb loopCalculateDiscreteModelForFastSynchro_2(
	numb* x,
	const numb* initialConditionsSlave,
	const numb* values,
	const numb h,
	const int amountOfIterations,
	const int amountOfX,
	const int preScaller,
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
 * Вычисляет следующее значение дискретной модели
 * и записывает результат в x
 * 
 * \param x			- Начальные условия или значения переменных предыдущей точки
 * \param values	- Параметры
 * \param h			- Шаг интегрирования
 */
__device__ __host__ __forceinline__  void calculateDiscreteModel(numb* x, const numb* values, const numb h);

__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h);
/**
 * Вычисляет траекторию для одной системы и записывает результат в "data" (если data != nullptr)
 * 
 * \param x						- Начальные условия для моделирования
 * \param values				- Параметры системы
 * \param h						- Шаг интегрирования
 * \param amountOfIterations	- Количество итераций
 * \param preScaller			- Множитель пропусков. Каждая 'preScaller' точка будет записана в результат
 * \param writableVar			- Какая из переменных в x[] будет записана в результат
 * \param maxValue				- Максимальное значение. Если будет x[writableVar] > maxValue, тогда функция вернет false
 * \param data					- Массив для записи данных
 * \param startDataIndex		- Индекс, с которого следует начинать запись в data
 * \param writeStep				- Шаг записи в массиве с данными (например, если шаг = 2, то запись будет в индексы: 0, 2, 4, ...)
 * \return						- Возаращет true, если ошибок не произошло
 */

__device__  bool loopCalculateDiscreteModel(numb* x, const numb* values, 
	const numb h, const int amountOfIterations, const int amountOfX, const int preScaller=0,
	const int writableVar = 0, const numb maxValue = 0,
	numb* data = nullptr, const int startDataIndex = 0, 
	const int writeStep = 1);

//__device__ __host__ int loopCalculateDiscreteModel_int(
__device__ int loopCalculateDiscreteModel_int(
	numb* x, const numb* values,
	const numb h, const int amountOfIterations, const int amountOfX =3, const int preScaller = 0,
	const int writableVar = 0, const numb maxValue = 0,
	numb* data = nullptr, const size_t startDataIndex = 0,
	const int writeStep = 1);


/**
 * Глобальная функция, которая распределенно вычисляет траекторию одной систем
 *
 * \param amountOfThreads			- 
 * \param h							- Шаг интегрирования
 * \param hSpecial					- 
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param data						- Массив, где будет хранится траектория систем
 * \return -
 */

__global__ void distributedCalculateDiscreteModelCUDA(
	const int		amountOfPointsForSkip,
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
 * Глобальная функция, которая вычисляет траекторию нескольких систем
 * 
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller ) 
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param amountOfPointsForSkip		- Количество точек для пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
 * \param Par_or_Var				- 0 - перебираем начальные условия, 1 - перебираем параметры
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
	const bool		Par_or_Var = 1);


// --------------------------------------------------------------------------



/**
 * Глобальная функция, которая вычисляет траекторию нескольких систем по шагу
 *
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param transientTime				- Время пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
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
 * Глобальная функция, которая вычисляет траекторию нескольких систем (по начальным условиям)
 *
 * \param nPts						- Общее разрешение диаграммы - nPts
 * \param nPtsLimiter				- Разрешение диаграммы, которое рассчитывается на данной итерации - nPtsLimiter
 * \param sizeOfBlock				- Количество точек в одной системе ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек систем
 * \param amountOfPointsForSkip		- Количество точек для пропуска ( transientTime )
 * \param dimension					- Размерность ( диаграмма одномерная )
 * \param ranges					- Массив с диапазонами
 * \param h							- Шаг интегрирования
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итераций ( равно количеству точек для одной системы )
 * \param preScaller				- Множитель, который уменьшает время и объем расчетов
 * \param writableVar				- Индекс уравнения, по которому будем строить диаграмму
 * \param maxValue					- Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
 * \param data						- Массив, где будет хранится траектория систем
 * \param maxValueCheckerArray		- Вспомогательный массив, куда при возникновении ошибки будет записано '-1' в соостветсвующую систему
 * \return -
 */
__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
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
	const int		amountOfPointsForSkip,
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
 * Функция, которая находит индекс в последовательности значений
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
 * \param idx			- Текущий idx в потоке
 * \param nPts			- Количество точек для разбиения диапазона (разрешение)
 * \param startRange	- Левая граница диапазона
 * \param finishRange	- Правая граница диапазона
 * \param valueNumber	- Номер требуемой переменной
 * \return Value		- Результат
 */
__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts, 
	const numb startRange, const numb finishRange, const int valueNumber);

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

/**
 * Функция, которая находит индекс в последовательности значений по логарифмической шкале
 *
 * \param idx			- Текущий idx в потоке
 * \param nPts			- Количество точек для разбиения диапазона (разрешение)
 * \param startRange	- Левая граница диапазона
 * \param finishRange	- Правая граница диапазона
 * \param valueNumber	- Номер требуемой переменной
 * \return Value		- Результат
 */
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);



/**
 * Находит пики в интервале [startDataIndex; startDataIndex + amountOfPoints] в "data" массиве
 * Результат записывается в outPeaks и timeOfPeaks ( если outPeaks != nullptr и timeOfPeaks != nullptr )
 * 
 * \param data				- Данные с точками
 * \param startDataIndex	- Индекс, откуда начинаем искать пики
 * \param amountOfPoints	- Количество точек для поиска пиков
 * \param outPeaks			- Выходной массив для записи в него найденных пиков
 * \param timeOfPeaks		- Выходной массив для записи в него индексов найденных пиков
 * \param h					- Шаг интегрирования (для вычисления межпикового интервала)
 * \return - Amount of found peaks
 */
__device__ __host__ numb globalPeakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints);

__device__ __host__ int peakFinder(numb* data, const size_t startDataIndex, const size_t amountOfPoints,
	numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h=0.0025);

__device__ __host__ void MeanAndMedianFreq(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq);
   					     
__device__ __host__ void MeanAndVariance(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval);

__global__ void DFT_custom(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* checkerArray, numb* AkCOS, numb* BkSIN, numb* rangesFreq, numb* window, int nFreq = 0, numb h = 0);

/**
 * Нахождение пиков в "data" массиве в многопоточном режиме 
 * Результат записывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если outPeaks != nullptr и timeOfPeaks != nullptr и amountOfPeaks != nullptr )
 * 
 * \param data				- Данные
 * \param sizeOfBlock		- Количество точек для одной системы ( tmax / h / preScaller )
 * \param amountOfBlocks	- Количество блоков ( систем ) в массиве "data"
 * \param amountOfPeaks		- Выходной массив, содержащий количество пиков для каждого блока ( системы )
 * \param outPeaks			- Выходной массив, содержащий fамплитуды найденных пиков
 * \param timeOfPeaks		- Выходной массив, содержащий межпиковые интервалы найденных пиков
 * \param h					- Шаг интегрирования ( для вычисления межпикового интервала )
 */

__global__ void globalPeakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks);

__global__ void peakFinderCUDA( numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0.0025 );

__global__ void MeanAndMedianFreqCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq);

__global__ void MeanAndVarianceCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval);


__device__ __host__ numb sign(numb x);
__device__ __host__ numb psi_m(numb x, numb m, numb d, numb k, numb dmargin);
__device__ __host__ numb chua_multistep_extended(numb x, numb m, numb d, numb kslope, numb dmargin);

/**
 * Нахождение пиков в "data" массиве в многопоточном режиме
 * Результат записывается в "outPeaks", "timeOfPeaks" и "amountOfPeaks" ( если outPeaks != nullptr и timeOfPeaks != nullptr и amountOfPeaks != nullptr )
 *
 * \param data				- Данные
 * \param sizeOfBlock		- Количество точек для одной системы ( tmax / h / preScaller )
 * \param amountOfBlocks	- Количество блоков ( систем ) в массиве "data"
 * \param amountOfPeaks		- Выходной массив, содержащий количество пиков для каждого блока ( системы )
 * \param outPeaks			- Выходной массив, содержащий fамплитуды найденных пиков
 * \param timeOfPeaks		- Выходной массив, содержащий межпиковые интервалы найденных пиков
 * \param h					- Шаг интегрирования ( для вычисления межпикового интервала )
 */
__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0);



/**
 * Вычисляет расстояние между двумя точками
 * 
 * \param x1 - x первой точки
 * \param y1 - y первой точки
 * \param x2 - x второй точки
 * \param y2 - y второй точки
 * \return - Дистанция
 */
__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2);



/**
 * Функция DBSCAN
 * 
 * \param data					- Данные (пики)
 * \param intervals				- Межпиковые интервалы
 * \param helpfulArray			- Вспомогательный массив
 * \param startDataIndex		- Индекс, с которого будет вестись запись в outData
 * \param amountOfPeaks			- Массив с количеством пиков в каждой системе
 * \param sizeOfHelpfulArray	- Размер вспомогательного массива
 * \param idx					- Текущий idx в потоке
 * \param eps					- Эпселон
 * \param outData				- Результирующий массив
 */
__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData);



/**
 * Глобальная функция DBSCAN
 * 
 * \param data				- Данные (пики)
 * \param sizeOfBlock		- Количество точек в одной системе
 * \param amountOfBlocks	- Количество блоков (систем) в data
 * \param amountOfPeaks		- Массив, содержащий количество пиков для каждого блока в data
 * \param intervals			- Межпиковые интервалы
 * \param helpfulArray		- Вспомогательный массив
 * \param eps				- Эпселон
 * \param outData			- Результирующий массив
 */
__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray, const numb eps, int* outData);



/**
 * Ядро для LLE
 * 
 * \param nPts						- Общее разрешение
 * \param nPtsLimiter				- Разрешение в текущем расчете
 * \param NT						- Время нормализации
 * \param tMax						- Время моделирования
 * \param sizeOfBlock				- Количество точек, занимаемое одной системой в "data"
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек
 * \param amountOfPointsForSkip		- Количество точек, которое будет промоделированно до основного расчета (transientTime)
 * \param dimension					- Размерность
 * \param ranges					- Массив, содержащий диапазоны перебираемого параметра
 * \param h							- Шаг интегрирования
 * \param eps						- Эпсилон
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итерация (вычисляется от tMax)
 * \param preScaller				- Множитель для ускорения расчетов
 * \param writableVar				- Индекс переменной в x[] по которому строим диаграмму
 * \param maxValue					- Макксимальное значение переменной при моделировании
 * \param resultArray				- Результирующий массив
 * \return -
 */
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const numb	NT,
	const numb	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
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
 * Ядро для LLE (IC)
 *
 * \param nPts						- Общее разрешение
 * \param nPtsLimiter				- Разрешение в текущем расчете
 * \param NT						- Время нормализации
 * \param tMax						- Время моделирования
 * \param sizeOfBlock				- Количество точек, занимаемое одной системой в "data"
 * \param amountOfCalculatedPoints	- Количество уже посчитанных точек
 * \param amountOfPointsForSkip		- Количество точек, которое будет промоделированно до основного расчета (transientTime)
 * \param dimension					- Размерность
 * \param ranges					- Массив, содержащий диапазоны перебираемого параметра
 * \param h							- Шаг интегрирования
 * \param eps						- Эпсилон
 * \param indicesOfMutVars			- Индексы изменяемых параметров
 * \param initialConditions			- Начальные условия
 * \param amountOfInitialConditions - Количество начальных условий
 * \param values					- Параметры
 * \param amountOfValues			- Количество параметров
 * \param amountOfIterations		- Количество итерация (вычисляется от tMax)
 * \param preScaller				- Множитель для ускорения расчетов
 * \param writableVar				- Индекс переменной в x[] по которому строим диаграмму
 * \param maxValue					- Макксимальное значение переменной при моделировании
 * \param resultArray				- Результирующий массив
 * \return -
 */
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const numb	NT,
	const numb	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
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
	const int amountOfPointsForSkip,
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

__global__ void LSKernelICCUDA(
	const int nPts,
	const int nPtsLimiter,
	const numb NT,
	const numb tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
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
 * Нахождение среднего значения пиков и межпиковых интервалов в "data" массиве в многопоточном режиме
 * Результат записывается в "outAvgPeaks", "AvgTimeOfPeaks" ( если outPeaks != nullptr и timeOfPeaks != nullptr )
 *
 * \param data				- Данные
 * \param sizeOfBlock		- Количество точек для одной системы ( tmax / h / preScaller )
 * \param amountOfBlocks	- Количество блоков ( систем ) в массиве "data"
 * \param outAvgPeaks		- Выходной массив, содержащий fамплитуды найденных пиков
 * \param AvgTimeOfPeaks	- Выходной массив, содержащий межпиковые интервалы найденных пиков
 * \param h					- Шаг интегрирования ( для вычисления межпикового интервала )
 */



__global__ void avgPeakFinderCUDA_logMaximas(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h = 0);

__global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h = 0);

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


