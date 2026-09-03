#pragma once

// Библиотеки CUDA
#include "configCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// KiShiVi библиотеки для работы с CUDA

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"

// Встроенные библиотеки

#include <iomanip>
#include <string>

// Общие параметры функций ниже (в их объявлениях не повторяются):
//   tMax          - время моделирования системы
//   transientTime - время, моделируемое до начала расчёта диаграммы (транзиент)
//   NT / NTime    - время нормализации (LLE, LS) / длина отрезка синхронизации
//   nPts          - разрешение диаграммы
//   h             - шаг интегрирования
//   initialConditions, amountOfInitialConditions - начальные условия и их
//                   количество (= число уравнений в системе)
//   values, amountOfValues - параметры системы и их количество
//   ranges        - диапазоны изменения свипуемых величин, по паре на измерение
//   indicesOfMutVars - индексы изменяемых величин в values
//   writableVar   - индекс уравнения, по которому строится диаграмма
//   maxValue      - порог по модулю, выше которого система считается разошедшейся
//   preScaller    - прореживание: рассчитывается только каждая preScaller-я точка
//   eps           - радиус окрестности DBSCAN (bifurcation2D, basins*,
//                   neuronClasterization2D) либо величина возмущения (LLE*, LS*)
//   kForward / kBackward - массивы коэффициентов синхронизации вперёд и назад
//   iterOfSynchr  - число итераций синхронизации
//   OUT_FILE_PATH - путь для сохранения результата в CSV

// Быстрая синхронизация пары мастер/слейв: временная реализация ошибки.
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
	std::string		OUT_FILE_PATH);

// Быстрая синхронизация: карта ошибки по сетке nPts x nPts.
__host__ void FastSynchro_2(
	const numb	NTime,
	const int		nPts,
	const numb*	values,
	const int		amountOfValues,
	const numb	h,
	const numb*	ranges,
	const int*		indicesOfMutVars,
	const numb*	kForward,
	const numb*	kBackward,
	const numb*	initialConditions,
	const numb* initConditionsSlave,
	const int		amountOfInitialConditions,
	const numb	maxValue,
	const int		iterOfSynchr,
	const int		preScaller,
	std::string		OUT_FILE_PATH);

// Ансамбль систем, разнесённых по шагу интегрирования (hSpecial — смещение между потоками).
__host__ void distributedSystemSimulation(
	const numb	tMax,
	const numb	h,
	const numb	hSpecial,
	const int		amountOfInitialConditions,
	const numb*	initialConditions,
	const int		writableVar,
	const numb	transientTime,
	const numb*	values,
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);

// Одномерная бифуркационная диаграмма.
__host__ void bifurcation1D(
	const numb	tMax,
	const int	nPts,
	const numb	h,
	const int	amountOfInitialConditions,
	const numb*	initialConditions,
	const numb*	ranges,
	const int*	indicesOfMutVars,
	const int	writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb*	values,
	const int	amountOfValues,
	const int	preScaller,
	std::string	OUT_FILE_PATH);

// Временные реализации для набора точек свипа.
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
	std::string		OUT_FILE_PATH);

// Одномерная бифуркационная диаграмма по шагу интегрирования (ranges — диапазон h).
__host__ void bifurcation1DForH(
	const numb	tMax,
	const int		nPts,
	const int		amountOfInitialConditions,
	const numb*	initialConditions,
	const numb*	ranges,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb*	values,
	const int		amountOfValues,
	const int		preScaller,
	std::string		OUT_FILE_PATH);

// Двумерная бифуркационная диаграмма (кластеризация DBSCAN).
__host__ void bifurcation2D(
	const numb	tMax,
	const int		nPts,
	const numb	h,
	const int		amountOfInitialConditions,
	const numb* __restrict__	initialConditions,
	const numb* __restrict__	ranges,
	const int* __restrict__		indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const int		preScaller,
	const numb	eps,
	std::string		OUT_FILE_PATH);

// Одномерная бифуркационная диаграмма в частотной области (спектр по DFT).
__host__ void bifurcation_DFT_1D(
	const numb	tMax,
	const int	nPts,
	const int	nFreq,								// Разрешение по частоте
	const numb	h,
	const int		amountOfInitialConditions,
	const numb* initialConditions,
	const numb* ranges,
	const numb* rangesFreq,								// Диапазон частот
	const int* indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb* values,
	const int		amountOfValues,
	const int		preScaller,
	const numb	eps,
	std::string		OUT_FILE_PATH);

// Двумерная карта режимов нейронной модели (кластеризация DBSCAN).
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
	std::string		OUT_FILE_PATH);

// Одномерная диаграмма старшего показателя Ляпунова.
__host__ void LLE1D(
	const numb	tMax,
	const numb	NT,
	const int		nPts,
	const numb	h,
	const numb	eps,
	const numb*	initialConditions,
	const int		amountOfInitialConditions,
	const numb*	ranges,
	const int*		indicesOfMutVars,
	const int		writableVar,
	const numb	maxValue,
	const numb	transientTime,
	const numb*	values,
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);

// Двумерная диаграмма старшего показателя Ляпунова.
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
	std::string		OUT_FILE_PATH);

// Одномерная диаграмма спектра показателей Ляпунова.
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
	std::string		OUT_FILE_PATH);

// Двумерная диаграмма спектра показателей Ляпунова.
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
	std::string		OUT_FILE_PATH);

void CUDA_dbscan(numb* data, numb* intervals, int* labels, int* helpfulArray, const int amountOfData, const numb eps, const int blockSize_fixed = blockSize_setup);

// Бассейны притяжения: свип по начальным условиям + DBSCAN
// (blockSize_fixed — размер блока CUDA, 0 = дефолт blockSize_setup).
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
	const int blockSize_fixed);

// То же, что basinsOfAttraction, но узлы сетки распределены логарифмически.
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
	std::string		OUT_FILE_PATH);
