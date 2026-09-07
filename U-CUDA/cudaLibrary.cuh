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
// Словарь параметров, общих для ядер ниже (в их описаниях не повторяется):
//   nPts / nPtsLimiter       - всего точек свипа / сколько считаем за один запуск (чанк)
//   sizeOfBlock              - точек в одной траектории (tMax / h / preScaller)
//   amountOfCalculatedPoints - смещение чанка: сколько точек свипа уже посчитано
//   amountOfPointsForSkip    - транзиент в шагах h (transientTime / h)
//   dimension / ranges       - размерность свипа / его границы, по паре на измерение
//   indicesOfMutVars         - индексы варьируемых величин
//   initialConditions, amountOfInitialConditions - начальные условия и их количество
//   values, amountOfValues   - параметры системы и их количество
//   h / amountOfIterations   - шаг интегрирования / число шагов (считается из tMax)
//   preScaller               - прореживание записи: шагов на одну записанную точку
//   writableVar              - индекс переменной в x[], которую пишем
//   maxValue                 - порог расходимости по модулю; превысив его, решение
//                              считается разошедшимся
//   data                     - буфер под посчитанные траектории
//   maxValueCheckerArray     - разошедшимся системам пишется -1
//   amountOfBlocks           - сколько траекторий обрабатываем
//   NT / eps (LLE, LS)       - интервал нормировки / величина начального возмущения
//   resultArray              - куда сложить показатель(и) Ляпунова

__device__ __host__ void calculateDiscreteModelforFastSynchro(numb* X, numb* S1, numb* K, const numb* a, const numb h, const bool directionOfintegration = 1);

// dbscanCUDA_optimized / dbscan_optimized удалены как мёртвый код — см.
// примечание на их прежнем месте в cudaLibrary.cu.

// icRandomOffset / icEps / icSeed — НУ слейва:
//   0 (legacy) — initialConditionsSlave, одни и те же для всех потоков;
//   1          — слейв стартует от точки мастера в начале окна СВОЕГО потока
//                (timedomain[startDataIndex], уже на аттракторе) со случайным
//                отступом в кубе [-icEps, +icEps] по каждой координате.
//                icSeed делает отступ воспроизводимым: тот же seed — та же карта.
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
	const int		preScaller,
	const int		icRandomOffset = 0,
	const numb		icEps = 0,
	const unsigned long long icSeed = 0,
	const int		gsWarmup = 0);

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
//
// icRandomOffset / icEps / icSeed — НУ ФИКСИРОВАННОЙ стороны:
//   0 (legacy) — берутся из своей затравки (initialConditions* как раньше);
//   1          — берутся от свипуемой стороны ПОСЛЕ её транзиента, со
//                случайным отступом в кубе [-icEps, +icEps] по каждой
//                координате. Кто свипуемый, решает swapRole; транзиент
//                фиксированной стороны в этом режиме не считается.
//                Счётчик ГПСЧ — глобальный индекс ячейки, поэтому карта не
//                зависит от разбиения расчёта на чанки.
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
	size_t	amountOfPointsForSkipSlave = 0,
	int		icRandomOffset = 0,
	numb	icEps = 0,
	unsigned long long icSeed = 0,
	int		gsWarmup = 0);

// Показатель сжатия ошибки за цикл вперёд-назад по схеме Беннеттина
// (error_estim 7). Возвращает log10 ρ за цикл; определение — в cudaLibrary.cu
// после gramSchmidtProcess, объявление здесь, потому что зовут его из
// loop...FastSynchro/_2, которые лежат в файле выше.
// masterWindow — окно мастер-траектории (amountOfIterations точек по amountOfX,
// row-major), eps — величина возмущения клонов. Только unidirectional.
__device__ numb fsBenettinCycleExponent(
	const numb* masterWindow,
	const numb* values,
	const numb h,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	const int iterOfSynchr,
	const numb* kForward,
	const numb* kBackward,
	const numb eps,
	const int gsWarmup);

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
	const int writeStep = 1,
	const numb icEps = 0,
	const int gsWarmup = 0);

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
	const int startDataIndex,
	const int icRandomOffset = 0,
	const numb icEps = 0,
	const unsigned long long icSeed = 0,
	const unsigned long long icCell = 0,
	const int gsWarmup = 0);

// Один шаг дискретной модели: новое состояние пишется обратно в x.
__device__ __host__ __forceinline__  void calculateDiscreteModel(numb* x, const numb* values, const numb h);

__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h);
// Несколько шагов модели; траектория складывается в data (если data != nullptr).
// Возвращает false, если |x[writableVar]| превысил maxValue.
// writeStep - шаг записи в буфер (при 2 заполняются индексы 0, 2, 4, ...).
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

// Ядро: ансамбль систем, разнесённых по шагу интегрирования (hSpecial).
// amountOfThreads - размер ансамбля.
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

// Ядро: траектории ансамбля систем.
// Par_or_Var: 0 - варьируем начальные условия, 1 - параметры.
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
	const int		logAxisMask = 0);         // bit i = axis slot i (X=0,Y=1) is log-distributed

// Ядро: траектории ансамбля при свипе по шагу интегрирования.
// transientTime здесь в единицах времени, а не в шагах.
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

// Ядро: траектории ансамбля при свипе по начальным условиям.
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
// Значение свипа по плоскому индексу idx сетки nPts^dim в [startRange; finishRange];
// valueNumber - номер измерения сетки. Для nPts=5 первые три измерения дают:
//   dim 0: 1 2 3 4 5 1 2 3 4 5 ...   getValueByIdx(7, 5, 1, 5, 0) = 3
//   dim 1: 1 1 1 1 1 2 2 2 2 2 ...   getValueByIdx(7, 5, 1, 5, 1) = 2
//   dim 2: 1 1 1 1 1 1 1 1 1 1 ...   getValueByIdx(7, 5, 1, 5, 2) = 1
__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts, 
	const numb startRange, const numb finishRange, const int valueNumber);

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

// То же, что getValueByIdx, но узлы распределены логарифмически.
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

// getValueByIdx для log-равномерной сетки (drop-in замена getValueByIdx) --
// требует startRange, finishRange > 0. Определение см. cudaLibrary.cu, рядом
// с getValueByIdxLog (эту декларацию сюда обязательно, иначе calculateDiscreteModelCUDA
// вызывает её раньше точки определения в .cu -- NVRTC компилирует файл линейно).
__device__ __host__ __forceinline__ numb getValueByIdx_log(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

// Поиск пиков в data[startDataIndex; startDataIndex + amountOfPoints).
// peakFinder дополнительно кладёт значения пиков в outPeaks, моменты времени
// в timeOfPeaks (те, что != nullptr) и возвращает число найденных пиков.
__device__ __host__ numb globalPeakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints);

__device__ __host__ int peakFinder(numb* data, const size_t startDataIndex, const size_t amountOfPoints,
	numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h=0.0025);

__device__ __host__ void MeanAndMedianFreq(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq);

__device__ __host__ void MeanAndVariance(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval);

__global__ void DFT_custom(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* checkerArray, numb* AkCOS, numb* BkSIN, numb* rangesFreq, numb* window, int nFreq = 0, numb h = 0,
	const int logFreqAxis = 0);

// Параллельный поиск пиков, блок на траекторию: число пиков -> amountOfPeaks,
// значения -> outPeaks, моменты времени -> timeOfPeaks (те, что != nullptr).
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

// То же, что peakFinderCUDA, для свипа по шагу интегрирования.
__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0);

// Евклидово расстояние между точками (x1, y1) и (x2, y2).
__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2);

// Кластеризация DBSCAN по признакам (значение пика; межпиковый интервал) одной
// траектории: data / intervals - признаки, helpfulArray - буфер на
// sizeOfHelpfulArray элементов, idx - индекс системы, eps - радиус окрестности,
// outData - куда сложить число найденных кластеров.
// multPeak / multInterval — множители осей признаков перед кластеризацией
// (пик и межпиковый интервал). Задаются per-diagram из GUI; дефолты в
// объявлении dbscanCUDA ниже совпадают с константами configCUDA.h, поэтому
// вызовы без этих аргументов (hostLibrary.cu) считают как раньше.
__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData,
	const numb multPeak = mult_peak, const numb multInterval = mult_interval);

// Ядро DBSCAN, блок на траекторию; смысл аргументов см. у dbscan выше.
// Дефолты множителей = константы configCUDA.h: <<<>>>-вызовы без этих
// аргументов (hostLibrary.cu) сохраняют прежнее поведение. ВНИМАНИЕ: при
// запуске через driver API (cuLaunchKernel в parametric_engine.cpp) значения
// по умолчанию не подставляются — там массив аргументов обязан содержать все
// 10 параметров.
__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray, const numb eps, int* outData,
	const numb multPeak = mult_peak, const numb multInterval = mult_interval);

// Ядро LLE (старший показатель Ляпунова).
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
	const int		logAxisMask = 0);    // bit i = axis slot i is log-distributed

// Ядро LLE, свип по начальным условиям.
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

// Ядро спектра показателей Ляпунова (LS).
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
	const int logAxisMask = 0);    // bit i = axis slot i is log-distributed

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

// Средние значения пиков и средние межпиковые интервалы, блок на траекторию:
// -> outAvgPeaks и AvgTimeOfPeaks (те, что != nullptr).
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
