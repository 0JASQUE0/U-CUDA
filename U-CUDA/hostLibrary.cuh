#pragma once

// -----------------------
// --- Библиотеки CUDA ---
// -----------------------
#include "configCUDA.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -----------------------

// --------------------------------------------
// --- KiShiVi библиотеки для работы с CUDA ---
// --------------------------------------------

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"

// --------------------------------------------

// -----------------------------
// --- Встроенные библиотеки ---
// -----------------------------

#include <iomanip>
#include <string>

// -----------------------------


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
	std::string		OUT_FILE_PATH);						// Эпсилон для алгоритма DBSCAN 

__host__ void FastSynchro_2(
	const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
	const int		nPts,							// Разрешение диаграммы
	const numb*	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const numb	h,									// Шаг интегрирования
	const numb*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const numb*	kForward,							// Массив коэффициентов синхронизации вперед
	const numb*	kBackward,							// Массив коэффициентов синхронизации назад
	const numb*	initialConditions,			// Массив с начальными условиями мастера
	const numb* initConditionsSlave,
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
	const int		iterOfSynchr,						// Число итераций синхронизации
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	std::string		OUT_FILE_PATH);

/**
 * Функция, для расчета одномерной бифуркационной диаграммы.
 */
__host__ void distributedSystemSimulation(
	const numb	tMax,							// Время моделирования системы
	const numb	h,								// Шаг интегрирования
	const numb	hSpecial,						// Шаг смещения между потоками
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb*	initialConditions,				// Массив с начальными условиями
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb*	values,							// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);				// Количество параметров				

/**
 * Функция, для расчета одномерной бифуркационной диаграммы.
 */

__host__ void bifurcation1D(
	const numb	tMax,								// Время моделирования системы
	const int	nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int	amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb*	initialConditions,					// Массив с начальными условиями
	const numb*	ranges,								// Диапазон изменения переменной
	const int*	indicesOfMutVars,					// Индекс изменяемой переменной в массиве values
	const int	writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb*	values,								// Параметры
	const int	amountOfValues,						// Количество параметров
	const int	preScaller,
	std::string	OUT_FILE_PATH);						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)


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
	std::string		OUT_FILE_PATH);						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)

/**
 * Функция, для расчета одномерной бифуркационной диаграммы по шагу.
 */
__host__ void bifurcation1DForH(
	const numb	tMax,							// Время моделирования системы
	const int		nPts,							// Разрешение диаграммы
	const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
	const numb*	initialConditions,				// Массив с начальными условиями
	const numb*	ranges,							// Диапазон изменения шага
	const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
	const numb*	values,							// Параметры
	const int		amountOfValues,					// Количество параметров
	const int		preScaller,
	std::string		OUT_FILE_PATH);					// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)






/**
 * Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN)
 */
__host__ void bifurcation2D(
	const numb	tMax,								// Время моделирования системы
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb* __restrict__	initialConditions,					// Массив с начальными условиями
	const numb* __restrict__	ranges,								// Диапазоны изменения параметров
	const int* __restrict__		indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb* __restrict__	values,								// Параметры
	const int		amountOfValues,						// Количество параметров
	const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
	const numb	eps,
	std::string		OUT_FILE_PATH);								// Эпсилон для алгоритма DBSCAN 

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
	std::string		OUT_FILE_PATH);								// Эпсилон для алгоритма DBSCAN 

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
	std::string		OUT_FILE_PATH);								// Эпсилон для алгоритма DBSCAN 


/**
 * Построение 1D LLE диаграммы
 */
__host__ void LLE1D(
	const numb	tMax,								// Время моделирования системы
	const numb	NT,									// Время нормализации
	const int		nPts,								// Разрешение диаграммы
	const numb	h,									// Шаг интегрирования
	const numb	eps,								// Эпсилон для LLE
	const numb*	initialConditions,					// Массив с начальными условиями
	const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
	const numb*	ranges,								// Диапазоны изменения параметров
	const int*		indicesOfMutVars,					// Индексы изменяемых параметров
	const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
	const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
	const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
	const numb*	values,								// Параметры
	const int		amountOfValues,
	std::string		OUT_FILE_PATH);					// Количество параметров



/**
 * Construction of a 2D LLE diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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



/**
 * Construction of a 2D LLE diagram (for initial conditions)
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */




/**
 * Construction of a 1D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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


/**
 * Construction of a 2D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
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


/**
 * Функция, для расчета двумерной бифуркационной диаграммы (DBSCAN) (for initial conditions)
 */
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
	const int blockSize_fixed);								// Эпсилон для алгоритма DBSCAN 




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
	std::string		OUT_FILE_PATH);								// Эпсилон для алгоритма DBSCAN 

