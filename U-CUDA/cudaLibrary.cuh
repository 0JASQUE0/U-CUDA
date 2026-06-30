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

__global__ void dbscanCUDA_optimized(
	const numb* __restrict__ data,
	const size_t sizeOfBlock,
	const int amountOfBlocks,
	const int* __restrict__ amountOfPeaks,
	const numb* __restrict__ intervals,
	const numb eps,
	int* __restrict__ outData);

__device__ int dbscan_optimized(
	const numb* __restrict__ data,       // ��������� ����� (������ ������)
	const numb* __restrict__ intervals,  // ���������� ��������� (������ ������)
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

// swapRole: 0 = grid varies master IC (legacy default — initialConditions
// overridden per cell, initialConditionsSlave fixed); 1 = grid varies slave IC
// (initialConditionsSlave overridden per cell, initialConditions fixed).
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
	int		swapRole = 0);

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
 * ��������� ��������� �������� ���������� ������
 * � ���������� ��������� � x
 * 
 * \param x			- ��������� ������� ��� �������� ���������� ���������� �����
 * \param values	- ���������
 * \param h			- ��� ��������������
 */
__device__ __host__ __forceinline__  void calculateDiscreteModel(numb* x, const numb* values, const numb h);

__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h);
/**
 * ��������� ���������� ��� ����� ������� � ���������� ��������� � "data" (���� data != nullptr)
 * 
 * \param x						- ��������� ������� ��� �������������
 * \param values				- ��������� �������
 * \param h						- ��� ��������������
 * \param amountOfIterations	- ���������� ��������
 * \param preScaller			- ��������� ���������. ������ 'preScaller' ����� ����� �������� � ���������
 * \param writableVar			- ����� �� ���������� � x[] ����� �������� � ���������
 * \param maxValue				- ������������ ��������. ���� ����� x[writableVar] > maxValue, ����� ������� ������ false
 * \param data					- ������ ��� ������ ������
 * \param startDataIndex		- ������, � �������� ������� �������� ������ � data
 * \param writeStep				- ��� ������ � ������� � ������� (��������, ���� ��� = 2, �� ������ ����� � �������: 0, 2, 4, ...)
 * \return						- ��������� true, ���� ������ �� ���������
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
 * ���������� �������, ������� ������������� ��������� ���������� ����� ������
 *
 * \param amountOfThreads			- 
 * \param h							- ��� ��������������
 * \param hSpecial					- 
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param data						- ������, ��� ����� �������� ���������� ������
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
 * ���������� �������, ������� ��������� ���������� ���������� ������
 * 
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller ) 
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param amountOfPointsForSkip		- ���������� ����� ��� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
 * \param Par_or_Var				- 0 - ���������� ��������� �������, 1 - ���������� ���������
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
 * ���������� �������, ������� ��������� ���������� ���������� ������ �� ����
 *
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param transientTime				- ����� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
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
 * ���������� �������, ������� ��������� ���������� ���������� ������ (�� ��������� ��������)
 *
 * \param nPts						- ����� ���������� ��������� - nPts
 * \param nPtsLimiter				- ���������� ���������, ������� �������������� �� ������ �������� - nPtsLimiter
 * \param sizeOfBlock				- ���������� ����� � ����� ������� ( tMax / h / preScaller )
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� ����� ������
 * \param amountOfPointsForSkip		- ���������� ����� ��� �������� ( transientTime )
 * \param dimension					- ����������� ( ��������� ���������� )
 * \param ranges					- ������ � �����������
 * \param h							- ��� ��������������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� ( ����� ���������� ����� ��� ����� ������� )
 * \param preScaller				- ���������, ������� ��������� ����� � ����� ��������
 * \param writableVar				- ������ ���������, �� �������� ����� ������� ���������
 * \param maxValue					- ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
 * \param data						- ������, ��� ����� �������� ���������� ������
 * \param maxValueCheckerArray		- ��������������� ������, ���� ��� ������������� ������ ����� �������� '-1' � ��������������� �������
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
 * �������, ������� ������� ������ � ������������������ ��������
 * ������:
 * ������������������:
 * 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
 * 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 * 
 * getValueByIdx(7, 5, 1, 5, 0) = 3
 * getValueByIdx(7, 5, 1, 5, 1) = 2
 * getValueByIdx(7, 5, 1, 5, 2) = 1
 * 
 * \param idx			- ������� idx � ������
 * \param nPts			- ���������� ����� ��� ��������� ��������� (����������)
 * \param startRange	- ����� ������� ���������
 * \param finishRange	- ������ ������� ���������
 * \param valueNumber	- ����� ��������� ����������
 * \return Value		- ���������
 */
__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts, 
	const numb startRange, const numb finishRange, const int valueNumber);

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);

/**
 * �������, ������� ������� ������ � ������������������ �������� �� ��������������� �����
 *
 * \param idx			- ������� idx � ������
 * \param nPts			- ���������� ����� ��� ��������� ��������� (����������)
 * \param startRange	- ����� ������� ���������
 * \param finishRange	- ������ ������� ���������
 * \param valueNumber	- ����� ��������� ����������
 * \return Value		- ���������
 */
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber);



/**
 * ������� ���� � ��������� [startDataIndex; startDataIndex + amountOfPoints] � "data" �������
 * ��������� ������������ � outPeaks � timeOfPeaks ( ���� outPeaks != nullptr � timeOfPeaks != nullptr )
 * 
 * \param data				- ������ � �������
 * \param startDataIndex	- ������, ������ �������� ������ ����
 * \param amountOfPoints	- ���������� ����� ��� ������ �����
 * \param outPeaks			- �������� ������ ��� ������ � ���� ��������� �����
 * \param timeOfPeaks		- �������� ������ ��� ������ � ���� �������� ��������� �����
 * \param h					- ��� �������������� (��� ���������� ����������� ���������)
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
 * ���������� ����� � "data" ������� � ������������� ������ 
 * ��������� ������������ � "outPeaks", "timeOfPeaks" � "amountOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr � amountOfPeaks != nullptr )
 * 
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param amountOfPeaks		- �������� ������, ���������� ���������� ����� ��� ������� ����� ( ������� )
 * \param outPeaks			- �������� ������, ���������� f��������� ��������� �����
 * \param timeOfPeaks		- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
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
__device__ __host__ numb chua_inner(numb x, numb m, numb d, numb kslope, numb mu_val, numb M_val);
__device__ __host__ numb psi_two_scale(numb x, numb m, numb M, numb d, numb kslope);

/**
 * ���������� ����� � "data" ������� � ������������� ������
 * ��������� ������������ � "outPeaks", "timeOfPeaks" � "amountOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr � amountOfPeaks != nullptr )
 *
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param amountOfPeaks		- �������� ������, ���������� ���������� ����� ��� ������� ����� ( ������� )
 * \param outPeaks			- �������� ������, ���������� f��������� ��������� �����
 * \param timeOfPeaks		- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
 */
__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks = nullptr, numb* outPeaks = nullptr, numb* timeOfPeaks = nullptr, numb h = 0);



/**
 * ��������� ���������� ����� ����� �������
 * 
 * \param x1 - x ������ �����
 * \param y1 - y ������ �����
 * \param x2 - x ������ �����
 * \param y2 - y ������ �����
 * \return - ���������
 */
__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2);



/**
 * ������� DBSCAN
 * 
 * \param data					- ������ (����)
 * \param intervals				- ���������� ���������
 * \param helpfulArray			- ��������������� ������
 * \param startDataIndex		- ������, � �������� ����� ������� ������ � outData
 * \param amountOfPeaks			- ������ � ����������� ����� � ������ �������
 * \param sizeOfHelpfulArray	- ������ ���������������� �������
 * \param idx					- ������� idx � ������
 * \param eps					- �������
 * \param outData				- �������������� ������
 */
__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData);



/**
 * ���������� ������� DBSCAN
 * 
 * \param data				- ������ (����)
 * \param sizeOfBlock		- ���������� ����� � ����� �������
 * \param amountOfBlocks	- ���������� ������ (������) � data
 * \param amountOfPeaks		- ������, ���������� ���������� ����� ��� ������� ����� � data
 * \param intervals			- ���������� ���������
 * \param helpfulArray		- ��������������� ������
 * \param eps				- �������
 * \param outData			- �������������� ������
 */
__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray, const numb eps, int* outData);



/**
 * ���� ��� LLE
 * 
 * \param nPts						- ����� ����������
 * \param nPtsLimiter				- ���������� � ������� �������
 * \param NT						- ����� ������������
 * \param tMax						- ����� �������������
 * \param sizeOfBlock				- ���������� �����, ���������� ����� �������� � "data"
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� �����
 * \param amountOfPointsForSkip		- ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
 * \param dimension					- �����������
 * \param ranges					- ������, ���������� ��������� ������������� ���������
 * \param h							- ��� ��������������
 * \param eps						- �������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� (����������� �� tMax)
 * \param preScaller				- ��������� ��� ��������� ��������
 * \param writableVar				- ������ ���������� � x[] �� �������� ������ ���������
 * \param maxValue					- ������������� �������� ���������� ��� �������������
 * \param resultArray				- �������������� ������
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
 * ���� ��� LLE (IC)
 *
 * \param nPts						- ����� ����������
 * \param nPtsLimiter				- ���������� � ������� �������
 * \param NT						- ����� ������������
 * \param tMax						- ����� �������������
 * \param sizeOfBlock				- ���������� �����, ���������� ����� �������� � "data"
 * \param amountOfCalculatedPoints	- ���������� ��� ����������� �����
 * \param amountOfPointsForSkip		- ���������� �����, ������� ����� ���������������� �� ��������� ������� (transientTime)
 * \param dimension					- �����������
 * \param ranges					- ������, ���������� ��������� ������������� ���������
 * \param h							- ��� ��������������
 * \param eps						- �������
 * \param indicesOfMutVars			- ������� ���������� ����������
 * \param initialConditions			- ��������� �������
 * \param amountOfInitialConditions - ���������� ��������� �������
 * \param values					- ���������
 * \param amountOfValues			- ���������� ����������
 * \param amountOfIterations		- ���������� �������� (����������� �� tMax)
 * \param preScaller				- ��������� ��� ��������� ��������
 * \param writableVar				- ������ ���������� � x[] �� �������� ������ ���������
 * \param maxValue					- ������������� �������� ���������� ��� �������������
 * \param resultArray				- �������������� ������
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
 * ���������� �������� �������� ����� � ���������� ���������� � "data" ������� � ������������� ������
 * ��������� ������������ � "outAvgPeaks", "AvgTimeOfPeaks" ( ���� outPeaks != nullptr � timeOfPeaks != nullptr )
 *
 * \param data				- ������
 * \param sizeOfBlock		- ���������� ����� ��� ����� ������� ( tmax / h / preScaller )
 * \param amountOfBlocks	- ���������� ������ ( ������ ) � ������� "data"
 * \param outAvgPeaks		- �������� ������, ���������� f��������� ��������� �����
 * \param AvgTimeOfPeaks	- �������� ������, ���������� ���������� ��������� ��������� �����
 * \param h					- ��� �������������� ( ��� ���������� ����������� ��������� )
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


