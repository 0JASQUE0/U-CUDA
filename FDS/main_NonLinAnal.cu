#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <conio.h>
#include "hostLibrary.cuh"
#include "windows.h"
#include <cstring>
//
//
//
int main()
{
	size_t startTime = std::clock();
//
//	// Kopets Mischenko
//	////numb init[6]{ 0, 0, 0, 0, 0, 0 };
//
//	//////numb params[10]{
//	//////0.03,	//gamma1,
//	//////0.088,	//gamma2,
//	//////7.5	,	//eps11,
//	//////10	,	//eps12,
//	//////24	,	//eps21,
//	//////11	,	//eps22,
//	//////0.1	,	//d,
//	//////0.3	,	//y_syn,
//	//////0.3	,	//theta_syn,
//	//////0.05	//k_syn;
//	//////};
//	////
//	////////Figure 6
//	////numb params[10]{
//	////0.05,	//0	gamma1,
//	////0.05,	//1	gamma2,
//	////5.0	,	//2	eps11,
//	////10.0,	//3	eps12,
//	////5.0	,	//4	eps21,
//	////11.0,	//5	eps22,
//	////0.1	,	//6	d,
//	////0.3	,	//7	y_syn,
//	////0.3	,	//8	theta_syn,
//	////0.01	//9	k_syn;
//	////};
//	////
//	////numb CT, TT;
//
//	//////Figure 7a
//	/////*/numb params[10]{
//	////0.03,	//0	gamma1,
//	////0.0,	//1	gamma2,
//	////7.5	,	//2	eps11,
//	////10.0,	//3	eps12,
//	////24.0	,	//4	eps21,
//	////11.0,	//5	eps22,
//	////0.0	,	//6	d,
//	////0.3	,	//7	y_syn,
//	////0.3	,	//8	theta_syn,
//	////0.01	//9	k_syn;
//	////};*/
//
//	//Figure 7b
//	/*numb params[10]{
//	0.05,	//0	gamma1,
//	0.0,	//1	gamma2,
//	19.5	,	//2	eps11,
//	10.0,	//3	eps12,
//	24.0	,	//4	eps21,
//	11.0,	//5	eps22,
//	0.0	,	//6	d,
//	0.3	,	//7	y_syn,
//	0.3	,	//8	theta_syn,
//	0.01	//9	k_syn;
//	};*/
//
	params[6] = -0.5;
	//params[1] = 0.04475;
	
	bifurcation1D(
		5000,		//const numb	tMax,							// Время моделирования системы
		1001,		//const int		nPts,						// Разрешение диаграммы
		0.05,		//const numb	h,								// Шаг интегрирования
		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
		init,//const numb * initialConditions,				// Массив с начальными условиями
		new numb[2]{ 0.04, 0.06 },//const numb * ranges,							// Диаппазон изменения переменной
		new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
		4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
		10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
		20000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
		params,//const numb * values,							// Параметры
		sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
		1,//const int		preScaller,
		"D:\\CUDAresults\\bif1D_KopetsMischenko_02.csv"//std::string		OUT_FILE_PATH
	);
//
//	//params[6] = 0.27;
//	//params[1] = 0.04475;
//	//bifurcation1D(
//	//	5000,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.05,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.04, 0.06 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	20000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_KopetsMischenko_03.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	//TimeDomainCalculation(
//	//	1000,	//const numb	tMax,							// Время моделирования системы
//	//	3,		//const int		nPts,							// Разрешение диаграммы
//	//	0.01,	//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.63, 0.63 },//const numb* ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 6 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	1000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	2,//const int		preScaller,
//	//	"D:\\CUDAresults\\TD_KopetsMischenko_h=0.01.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	///*CT = 125.65 * (100.0 + 1.0);
//	//TT = */125.65 * (100.0);
//	
//	//LLE2D(
//	//	CT,	//const numb tMax,
//	//	5.0,	//const numb NT,
//	//	5,	//const int nPts,
//	//	0.05,	//const numb h,
//	//	1e-5,	//const numb eps,
//	//	init,	//const numb * initialConditions,
//	//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//	new numb[4]{ -0.75, 0.75, 0.034, 0.068 },//const numb * ranges,
//	//	new int[2] { 6, 1},//const int* indicesOfMutVars,
//	//	1,		//const int writableVar,
//	//	100000000,	//const numb maxValue,
//	//	TT,	//const numb transientTime,
//	//	params,	//const numb * values,
//	//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//	"D:\\CUDAresults\\LS2D_KopetsMischenko_41_1.csv"//std::string		OUT_FILE_PATH
//	//);
//
	LS2D(
		CT,	//const numb tMax,
		5.0,	//const numb NT,
		501,	//const int nPts,
		0.05,	//const numb h,
		1e-5,	//const numb eps,
		init,	//const numb * initialConditions,
		sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
		new numb[4]{ -0.75, 0.75, 0.034, 0.068 },//const numb * ranges,
		new int[2] { 6, 1},//const int* indicesOfMutVars,
		1,		//const int writableVar,
		100000000,	//const numb maxValue,
		TT,	//const numb transientTime,
		params,	//const numb * values,
		sizeof(params) / sizeof(numb),//const int amountOfValues,
		"D:\\CUDAresults\\LS2D_KopetsMischenko_41_1.csv"//std::string		OUT_FILE_PATH
	);
//	
//	////////bifurcation2D(
//	////////	CT, //const numb	tMax,								// Время моделирования системы
//	////////	11, //const int		nPts,								// Разрешение диаграммы
//	////////	0.05, //const numb	h,									// Шаг интегрирования
//	////////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	////////	init,//const numb* initialConditions,					// Массив с начальными условиями
//	////////	new numb[4]{ -0.75, 0.75, 0.034, 0.068 },//const numb* ranges,								// Диапазоны изменения параметров
//	////////	new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	////////	//new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	////////	//new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	////////	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	////////	1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	////////	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	////////	params,//const numb* values,								// Параметры
//	////////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	////////	10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	////////	1.0,//const numb	eps,
//	////////	"D:\\CUDAresults\\bif2D_KopetsMischenko_41_1.csv" //std::string		OUT_FILE_PATH
//	////////);
//	////////
//	////////bifurcation2D(
//	////////	CT, //const numb	tMax,								// Время моделирования системы
//	////////	801, //const int		nPts,								// Разрешение диаграммы
//	////////	0.05, //const numb	h,									// Шаг интегрирования
//	////////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	////////	init,//const numb* initialConditions,					// Массив с начальными условиями
//	////////	new numb[4]{ -0.75, 0.75, 0.034, 0.068 },//const numb* ranges,								// Диапазоны изменения параметров
//	////////	new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	////////	//new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	////////	//new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	////////	4, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	////////	1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	////////	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	////////	params,//const numb* values,								// Параметры
//	////////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	////////	10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	////////	1.0,//const numb	eps,
//	////////	"D:\\CUDAresults\\bif2D_KopetsMischenko_41_2.csv" //std::string		OUT_FILE_PATH
//	////////);
//	//////LLE2D(
//	//////	CT/2.0,	//const numb tMax,
//	//////	5.0,	//const numb NT,
//	//////	801,	//const int nPts,
//	//////	0.05,	//const numb h,
//	//////	1e-6,	//const numb eps,
//	//////	init,	//const numb * initialConditions,
//	//////	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//////	new numb[4]{ -0.75, 0.75, 0.034, 0.068 },//const numb * ranges,
//	//////	new int[2] { 6, 1},//const int* indicesOfMutVars,
//	//////	0 ,		//const int writableVar,
//	//////	100000,	//const numb maxValue,
//	//////	TT,	//const numb transientTime,
//	//////	params,	//const numb * values,
//	//////	sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//////	"D:\\CUDAresults\\LLE2D_KopetsMischenko_41_1.csv"//std::string		OUT_FILE_PATH
//	//////);
//	//////
//	////// //Figure 7a
//	////// params[0] = 0.03;	//0	gamma1,
//	////// params[1] = 0.0;	//1	gamma2,
//	////// params[2] = 7.5;	//2	eps11,
//	////// params[3] = 10.0;	//3	eps12,
//	////// params[4] = 24.0;	//4	eps21,
//	////// params[5] = 11.0;	//5	eps22,
//	////// params[6] = 0.0;	//6	d,
//	////// params[7] = 0.3;	//7	y_syn,
//	////// params[8] = 0.3;	//8	theta_syn,
//	////// params[9] = 0.01;	//9	k_syn;
//	////// CT = 209.45 * (100.0 + 1.0);
//	////// TT = 209.45 * (500.0);
//	////// //bifurcation2D(
//	//////	// CT, //const numb	tMax,								// Время моделирования системы
//	//////	// 11, //const int		nPts,								// Разрешение диаграммы
//	//////	// 0.05, //const numb	h,									// Шаг интегрирования
//	//////	// sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//////	// init,//const numb* initialConditions,					// Массив с начальными условиями
//	//////	// new numb[4]{ -0.75, 0.75, 0.07, 0.11 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// //new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// //new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// 1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//////	// 1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//////	// TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//////	// params,//const numb* values,								// Параметры
//	//////	// sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//////	// 10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//////	// 1.0,//const numb	eps,
//	//////	// "D:\\CUDAresults\\bif2D_KopetsMischenko_42_1.csv" //std::string		OUT_FILE_PATH
//	////// //);
//	////// //bifurcation2D(
//	//////	// CT, //const numb	tMax,								// Время моделирования системы
//	//////	// 801, //const int		nPts,								// Разрешение диаграммы
//	//////	// 0.05, //const numb	h,									// Шаг интегрирования
//	//////	// sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//////	// init,//const numb* initialConditions,					// Массив с начальными условиями
//	//////	// new numb[4]{ -0.75, 0.75, 0.07, 0.11 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// //new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// //new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// 4, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//////	// 1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//////	// TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//////	// params,//const numb* values,								// Параметры
//	//////	// sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//////	// 10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//////	// 1.0,//const numb	eps,
//	//////	// "D:\\CUDAresults\\bif2D_KopetsMischenko_42_2.csv" //std::string		OUT_FILE_PATH
//	////// //);
//	////// LLE2D(
//	//////	 CT / 2.0,	//const numb tMax,
//	//////	 5.0,	//const numb NT,
//	//////	 801,	//const int nPts,
//	//////	 0.05,	//const numb h,
//	//////	 1e-6,	//const numb eps,
//	//////	 init,	//const numb * initialConditions,
//	//////	 sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//////	 new numb[4]{ -0.75, 0.75, 0.07, 0.11 },//const numb * ranges,
//	//////	 new int[2] { 6, 1},//const int* indicesOfMutVars,
//	//////	 0,		//const int writableVar,
//	//////	 100000,	//const numb maxValue,
//	//////	 TT,	//const numb transientTime,
//	//////	 params,	//const numb * values,
//	//////	 sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//////	 "D:\\CUDAresults\\LLE2D_KopetsMischenko_42_1.csv"//std::string		OUT_FILE_PATH
//	////// );
//	 //Figure 7b
//	 //params[0] = 0.05;	//0	gamma1,
//	 //params[1] = 0.0;	//1	gamma2,
//	 //params[2] = 19.5;	//2	eps11,
//	 //params[3] = 10.0;	//3	eps12,
//	 //params[4] = 24.0;	//4	eps21,
//	 //params[5] = 11.0;	//5	eps22,
//	 //params[6] = 0.0;	//6	d,
//	 //params[7] = 0.3;	//7	y_syn,
//	 //params[8] = 0.3;	//8	theta_syn,
//	 //params[9] = 0.01;	//9	k_syn;
//	 //CT = 251.34 * (100.0 + 1.0);
//	 //TT = 251.34 * (500.0);
//
//	 //LS2D(
//		// CT,	//const numb tMax,
//		// 5.0,	//const numb NT,
//		// 501,	//const int nPts,
//		// 0.05,	//const numb h,
//		// 1e-5,	//const numb eps,
//		// init,	//const numb * initialConditions,
//		// sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		// new numb[4]{ -0.75, 0.75, 0.05, 0.095 },//const numb * ranges,
//		// new int[2] { 6, 1},//const int* indicesOfMutVars,
//		// 1,		//const int writableVar,
//		// 100000000,	//const numb maxValue,
//		// TT,	//const numb transientTime,
//		// params,	//const numb * values,
//		// sizeof(params) / sizeof(numb),//const int amountOfValues,
//		// "D:\\CUDAresults\\LS2D_KopetsMischenko_43_1.csv"//std::string		OUT_FILE_PATH
//	 //);
//
//	////// //bifurcation2D(
//	//////	// CT, //const numb	tMax,								// Время моделирования системы
//	//////	// 11, //const int		nPts,								// Разрешение диаграммы
//	//////	// 0.05, //const numb	h,									// Шаг интегрирования
//	//////	// sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//////	// init,//const numb* initialConditions,					// Массив с начальными условиями
//	//////	// new numb[4]{ -0.75, 0.75, 0.05, 0.095 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// //new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// //new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// 1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//////	// 1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//////	// TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//////	// params,//const numb* values,								// Параметры
//	//////	// sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//////	// 2, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//////	// 1.0,//const numb	eps,
//	//////	// "D:\\CUDAresults\\bif2D_KopetsMischenko_43_1.csv" //std::string		OUT_FILE_PATH
//	////// //);
//	////// //bifurcation2D(
//	//////	// CT, //const numb	tMax,								// Время моделирования системы
//	//////	// 801, //const int		nPts,								// Разрешение диаграммы
//	//////	// 0.05, //const numb	h,									// Шаг интегрирования
//	//////	// sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//////	// init,//const numb* initialConditions,					// Массив с начальными условиями
//	//////	// new numb[4]{ -0.75, 0.75, 0.05, 0.095 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// //new numb[4]{ -1, 1, 0.05, 0.15 },//const numb* ranges,								// Диапазоны изменения параметров
//	//////	// //new int[2] { 6, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//////	// 4, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//////	// 1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//////	// TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//////	// params,//const numb* values,								// Параметры
//	//////	// sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//////	// 10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//////	// 1.0,//const numb	eps,
//	//////	// "D:\\CUDAresults\\bif2D_KopetsMischenko_43_2.csv" //std::string		OUT_FILE_PATH
//	////// //);
//	////// LLE2D(
//	//////	 CT / 2.0,	//const numb tMax,
//	//////	 5.0,	//const numb NT,
//	//////	 801,	//const int nPts,
//	//////	 0.05,	//const numb h,
//	//////	 1e-6,	//const numb eps,
//	//////	 init,	//const numb * initialConditions,
//	//////	 sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//////	 new numb[4]{ -0.75, 0.75, 0.05, 0.095 },//const numb * ranges,
//	//////	 new int[2] { 6, 1},//const int* indicesOfMutVars,
//	//////	 0,		//const int writableVar,
//	//////	 100000,	//const numb maxValue,
//	//////	 TT,	//const numb transientTime,
//	//////	 params,	//const numb * values,
//	//////	 sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//////	 "D:\\CUDAresults\\LLE2D_KopetsMischenko_43_1.csv"//std::string		OUT_FILE_PATH
//	////// );
//	// 
//	//TimeDomainCalculation(
//	//	3000,	//const numb	tMax,							// Время моделирования системы
//	//	2,		//const int		nPts,							// Разрешение диаграммы
//	//	0.05,	//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.255, 0.255 },//const numb* ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 6 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	1000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	4000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\TD_KopetsMischenko_h=0.01.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	
//	//params[1] = 0.04965;
//	//std::string path;
//	//int decimator[11]{1,2,4,5,8,10,15,20,25,40,50};
//	//for (int j = 0; j < 11; j++) {
//	//	path = "D:\\CUDAresults\\bif1D_KopetsMischenko_par_d2_decimator=" + std::to_string(decimator[j]) + ".csv";
//
		bifurcation1D(
			3000,		//const numb	tMax,							// Время моделирования системы
			1001,		//const int		nPts,						// Разрешение диаграммы
			0.05,		//const numb	h,								// Шаг интегрирования
			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
			init,//const numb * initialConditions,				// Массив с начальными условиями
			new numb[2]{ -0.75, 0.75 },//const numb * ranges,							// Диаппазон изменения переменной
			new int[1] { 6 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
			4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
			10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
			4000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
			params,//const numb * values,							// Параметры
			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
			1,//const int		preScaller,
			path//std::string		OUT_FILE_PATH
		);
	}
//
//	//TimeDomainCalculation(
//	//	3000,	//const numb	tMax,							// Время моделирования системы
//	//	2,		//const int		nPts,							// Разрешение диаграммы
//	//	0.05,	//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.255, 0.255 },//const numb* ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 6 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	4,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	1000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	4000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\TD_KopetsMischenko_h=0.01.csv"//std::string		OUT_FILE_PATH
//	//);
//	
//	// Burkin Matreshka 2.0 --------------------------------------------------------------------------------------------------------------------
//
//	//numb init[5]{ 0.1, 0, 0, 0, 0 };
//	//numb params[5]{ 0.0, 10, 0.5, 18, 1.7 };
//	//numb CT = 300;
//	//numb TT = 2500;
//	//numb h = 0.01;
//	//int res = 101;
//
//	////basinsOfAttraction(
//	////	CT,									// Время моделирования системы
//	////	res,									// Разрешение диаграммы
//	////	h,									// Шаг интегрирования
//	////	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	////	init,									// Массив с начальными условиями
//	////	new numb[4]{ -10, 10, -10, 10 },
//	////	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	////	4,										// Индекс уравнения, по которому будем строить диаграмму
//	////	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	////	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	////	params,									// Параметры
//	////	sizeof(params) / sizeof(numb),		// Количество параметров
//	////	2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	////	0.15,									// Эпсилон для алгоритма DBSCAN
//	////	"D:\\CUDAresults\\Basins_BurkinMatreshka_2.csv",
//	////	32
//	////);
//	//
//	//bifurcation2D(
//	//	CT, //const numb	tMax,								// Время моделирования системы
//	//	401, //const int		nPts,								// Разрешение диаграммы
//	//	h, //const numb	h,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,					// Массив с начальными условиями
//	//	//new numb[4]{ -7, 4, -10, 15 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new numb[4]{ 1e2, 1e-6, 1e2, 1e-6 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new numb[4]{ 1, 1.25, -1, -0.75 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	new numb[4]{ -4, 5, 1.6, 2.3 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new int[2] { 0, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	new int[2] { 2, 4 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	1e15, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.001,//const numb	eps,
//	//	"D:\\CUDAresults\\Basins2_BurkinMatreshka_2_3_u.csv" //std::string		OUT_FILE_PATH
//	//);
//	//
//	//Basins Estimation
//	/*cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
//	printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
//	std::cout << "Max blocks on SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
//	std::cout << "Max treads on SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
//	std::cout << "Shared memory on SM: " << prop.sharedMemPerMultiprocessor << std::endl;
//	std::cout << "Amount SM: " << prop.multiProcessorCount << std::endl;
//	
//	numb params[2]{ 0.5, 0.1665 };
//	numb init[3]{ 0.0, 0.0, 0.0 };
//	//numb BS_array[7]{ 8, 16, 32, 64, 128, 256, 512};
//	numb BS_array[7]{ 8, 16, 32, 64, 128, 256, 512};
//	numb CT_array[6]{ 500, 1000, 1500, 2000, 2500, 3000 };
//	numb res_array[6]{ 100, 200, 400 ,600, 800, 1000};
//	//numb CT_array[6]{ 500, 1000, 1500, 2000, 2500, 3000 };
//	//numb res_array[6]{ 100, 200, 400, 600, 800, 1000 };
//	numb CT = 500;
//	numb TT = 500;
//	int res = 100;
//	int BS = 32;
//	numb h = 0.01;
//	int blockSize_fixed = 64;
//	std::string path0 = "D:\\CUDAresults\\Basins_THOMAS_CT=";
//	std::string path = "";
//	basinsOfAttraction(
//		CT,									// Время моделирования системы
//		res,									// Разрешение диаграммы
//		h,									// Шаг интегрирования
//		AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//		init,									// Массив с начальными условиями
//		new numb[4]{ -6, 6, -6, 6 },
//		new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		0,										// Индекс уравнения, по которому будем строить диаграмму
//		100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//		params,									// Параметры
//		sizeof(params) / sizeof(numb),		// Количество параметров
//		1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		0.15,									// Эпсилон для алгоритма DBSCAN
//		"D:\\CUDAresults\\Basins_THOMAS_CT.csv",
//		blockSize_fixed
//	);
//	printf("--- Let's go! ---\n");
//	zfor (int i = 0; i < sizeof(res_array) / sizeof(numb); i++) {
//		res = res_array[i];
//
//		for (int j = 0; j < sizeof(CT_array) / sizeof(numb); j++) {
//			CT = CT_array[j];
//
//			for (int k = 0; k < sizeof(BS_array) / sizeof(numb); k++) {
//				BS = BS_array[k];
//				path = path0 + std::to_string((int)CT) + "_res=" + std::to_string((int)res) + ".csv";
//
//				basinsOfAttraction(
//					CT,									// Время моделирования системы
//					res,									// Разрешение диаграммы
//					h,									// Шаг интегрирования
//					sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//					init,									// Массив с начальными условиями
//					new numb[4]{ -6, 6, -6, 6 },
//					new int[2] { 0, 1 },						// Индексы изменяемых параметров
//					0,										// Индекс уравнения, по которому будем строить диаграмму
//					100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//					TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//					params,									// Параметры
//					sizeof(params) / sizeof(numb),		// Количество параметров
//					1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//					0.15,									// Эпсилон для алгоритма DBSCAN
//					path,
//					BS
//				);
//			}
//		}
//	}
//	printf("--- Let's go! ---\n");
//	CT = 500;
//	for (int i = 0; i < sizeof(res_array) / sizeof(numb); i++) {
//		res = res_array[i];
//		for (int k = 0; k < sizeof(BS_array) / sizeof(numb); k++) {
//			BS = BS_array[k];
//		path = path0 + std::to_string((int)CT) + "_res=" + std::to_string((int)res) + ".csv";
//			basinsOfAttraction(
//				CT,									// Время моделирования системы
//				res,									// Разрешение диаграммы
//				h,									// Шаг интегрирования
//				sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//				init,									// Массив с начальными условиями
//				new numb[4]{ -6, 6, -6, 6 },
//				new int[2] { 0, 1 },						// Индексы изменяемых параметров
//				0,										// Индекс уравнения, по которому будем строить диаграмму
//				100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//				TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//				params,									// Параметры
//				sizeof(params) / sizeof(numb),		// Количество параметров
//				1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//				0.05,									// Эпсилон для алгоритма DBSCAN
//				path,
//				BS
//			);
//		}
//	} */
//
//	// Rossler Fast Synhro
//	
//	//numb params[4]{0.5, 0.2, 0.2, 5.7};
//	//numb init[3]{ 0, 0.1, 0};
//	//bifurcation1D(
//	//	500,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.01,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -0.1, 0.15 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Rossler_Composite_3_h=0.01.csv"//std::string		OUT_FILE_PATH
//	//);
//	//bifurcation1D(
//	//	500,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.005,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -0.1, 0.15 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	2,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Rossler_Composite_3_h=0.005.csv"//std::string		OUT_FILE_PATH
//	//);
//	//bifurcation1D(
//	//	500,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.0025,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -0.1, 0.15 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	4,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Rossler_Composite_3_h=0.0025.csv"//std::string		OUT_FILE_PATH
//	//);
//	//params[2] = 0.9;
//	//bifurcation1D(
//	//	500,		//const numb	tMax,							// Время моделирования системы
//	//	3001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.01,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.33, 0.36 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	4,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Rossler_testInterp_fiveParabolicInterp.csv"//std::string		OUT_FILE_PATH
//	//);
//	//bifurcation2D(
//	//	200, //const numb	tMax,								// Время моделирования системы
//	//	500, //const int		nPts,								// Разрешение диаграммы
//	//	0.01, //const numb	h,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,					// Массив с начальными условиями
//	//	new numb[4]{ 0.05, 0.35, 0.05, 0.35 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	new int[2] { 1, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	1800, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.5,//const numb	eps,
//	//	"D:\\CUDAresults\\bif2D_Rossler.csv" //std::string		OUT_FILE_PATH
//	//);
//	//LS2D(
//	//	1500,	//const numb tMax,
//	//	1.0,	//const numb NT,
//	//	301,	//const int nPts,
//	//	0.01,	//const numb h,
//	//	1e-5,	//const numb eps,
//	//	init,	//const numb * initialConditions,
//	//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//	new numb[4]{ 0.0, 0.4, 0.0, 0.4 },//const numb * ranges,
//	//	new int[2] { 1, 2},//const int* indicesOfMutVars,
//	//	0,		//const int writableVar,
//	//	100000000,	//const numb maxValue,
//	//	2500,	//const numb transientTime,
//	//	params,	//const numb * values,
//	//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//	"D:\\CUDAresults\\LS2D_Rossler.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	/*FastSynchro(
//		1000,										//const numb	tMax,								// Время моделирования системы
//		300,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		1.0,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		params,										//const numb* values,								// Параметры
//		sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		0.01,										//const numb	h,									// Шаг интегрирования
//		new numb[3]{ 0, 3, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		new numb[3]{ 0, 20, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		new numb[3]{ 0, 0, 0},				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		200,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		1,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		"D:\\CUDAresults\\FS_Rossler.csv"//std::string		OUT_FILE_PATH
//	);
//	FastSynchro_2(
//		1.0,							//const numb		NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		300,							//const int		nPts,								// Разрешение диаграммы
//		params,							//const numb* values,								// Параметры
//		sizeof(params) / sizeof(numb),	//const int		amountOfValues,						// Количество параметров
//		0.01,							//const numb		h,									// Шаг интегрирования
//		new numb[4]{ -10, 10, -10, 10 },//const numb* ranges,								// Диапазоны изменения параметров
//		new int[2]{ 0, 1 },				//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		new numb[3]{ 0, 3, 0 },			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		new numb[3]{ 0, 20, 0 },		//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		init,							//const numb* initialConditions,					// Массив с начальными условиями мастера
//		new numb[3]{ 0, 0, 0 },			//const numb* initConditionsSlave,
//		sizeof(init) / sizeof(numb),	//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		1000000,						//const numb		maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		200,							//const int		iterOfSynchr,						// Число итераций синхронизации
//		1,								//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		"D:\\CUDAresults\\FS2_Rossler.csv"//std::string		OUT_FILE_PATH
//	);*/
//
//	// Faiza system 23022026 
//	/*
//	numb params[5]{0.5, 1.7, 0.8, 10.0, -0.92};
//	numb init[3]{ 0.2, 0.0, 0.0 };
//	numb CT = 500;
//	numb TT = 1000;
//	numb h = 0.005;
//	int res = 101;
//	//
//	//params[1] = 1.9408;
//	//init[0] = 0.2;
//	//
//	//bifurcation2D(
//	//	CT, //const numb	tMax,								// Время моделирования системы
//	//	801, //const int		nPts,								// Разрешение диаграммы
//	//	h, //const numb	h,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,					// Массив с начальными условиями
//	//	new numb[4]{ 0, 15, -1, 1 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	new int[2] { 1, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
//	//	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//	2, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.2,//const numb	eps,
//	//	"D:\\CUDAresults\\bif2D_Faiza_23022026_par_a_vs_b_2.csv" //std::string		OUT_FILE_PATH
//	//);
//	//
//	//LS2D(
//	//	5*CT,	//const numb tMax,
//	//	1.0,	//const numb NT,
//	//	801,	//const int nPts,
//	//	h,	//const numb h,
//	//	1e-5,	//const numb eps,
//	//	init,	//const numb * initialConditions,
//	//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//	new numb[4]{ 0, 15, -1, 1 },//const numb * ranges,
//	//	new int[2] { 1, 2},//const int* indicesOfMutVars,
//	//	1,		//const int writableVar,
//	//	100000000,	//const numb maxValue,
//	//	TT,	//const numb transientTime,
//	//	params,	//const numb * values,
//	//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//	"D:\\CUDAresults\\LS2D_Faiza_23022026_par_a_vs_b_02.csv"//std::string		OUT_FILE_PATH
//	//);
//	
//	numb a_array[4]{1.94, 2.51, 2.645 };
//
//	for (int i = 0; i < 3; i++) {
//		params[1] = a_array[i];
//		std::string path = "D:\\CUDAresults\\Basins_Faiza_2302202_a=" + std::to_string(params[1]) + "_HR.csv";
//		basinsOfAttraction(
//			CT,									// Время моделирования системы
//			801,									// Разрешение диаграммы
//			h,									// Шаг интегрирования
//			AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//			init,									// Массив с начальными условиями
//			new numb[4]{ -10, 10, -10, 10 },
//			new int[2] { 0, 2 },						// Индексы изменяемых параметров
//			0,										// Индекс уравнения, по которому будем строить диаграмму
//			100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//			params,									// Параметры
//			sizeof(params) / sizeof(numb),		// Количество параметров
//			4,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.1,									// Эпсилон для алгоритма DBSCAN
//			path,
//			blockSize_setup
//		);
//	}*/
//	
//	//init[1] = sqrt(10.0);
//	//params[1] = 2.51;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	101,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 2 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.1,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_Faiza_2302202_a=.csv",
//	//	blockSize_setup
//	//);
//
//	//bifurcation1DForH(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,	//const int		nPts,							// Разрешение диаграммы
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,		//const numb* initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.001, 0.1 },//const numb* ranges,							// Диапазон изменения шага
//	//	1,			//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,		//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,			//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,		//const numb* values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,			//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_h_Faiza_23022026.csv"//std::string		OUT_FILE_PATH
//	//);		
//				
//	//bifurcation1D(
//	//	250,		//const numb	tMax,							// Время моделирования системы
//	//	10001,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 1.5, 3.5 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_a_01.csv"//std::string		OUT_FILE_PATH
//	//);
//	//init[0] = -0.2;
//	//bifurcation1D(
//	//	250,		//const numb	tMax,							// Время моделирования системы
//	//	10001,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 1.5, 3.5 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_a_02.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	//
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -0.6, 1.3 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_b_01.csv"//std::string		OUT_FILE_PATH
//	//);
//	//
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 2.0, 18.0 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 3 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_c_01.csv"//std::string		OUT_FILE_PATH
//	//);
//	//LS1D(
//	//	5.0*CT, //const numb	tMax,								// Время моделирования системы
//	//	1.0,	//const numb	NT,									// Время нормализации
//	//	res,	//const int		nPts,								// Разрешение диаграммы
//	//	h,	//const numb	h,									// Шаг интегрирования
//	//	1e-5,	//const numb	eps,								// Эпсилон для LLE
//	//	init,//const numb * initialConditions,					// Массив с начальными условиями
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	new numb[2]{ 0.4, 2.2 },//const numb * ranges,								// Диапазоны изменения параметров
//	//	new int[1] { 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	1,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,
//	//	"D:\\CUDAresults\\LS1D_Faiza_23022026_par_a_01.csv"//std::string		OUT_FILE_PATH
//	//);
//	//init[0] = -0.2; 
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 0.4, 2.2 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_a_02.csv"//std::string		OUT_FILE_PATH
//	//);
//	//
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -0.6, 1.3 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_b_02.csv"//std::string		OUT_FILE_PATH
//	//);
//	//
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	res,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 2.0, 18.0 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 3 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_Faiza_23022026_par_c_02.csv"//std::string		OUT_FILE_PATH
//	//);
//	//LS1D(
//	//	5.0*CT, //const numb	tMax,								// Время моделирования системы
//	//	1.0,	//const numb	NT,									// Время нормализации
//	//	res,	//const int		nPts,								// Разрешение диаграммы
//	//	h,	//const numb	h,									// Шаг интегрирования
//	//	1e-5,	//const numb	eps,								// Эпсилон для LLE
//	//	init,//const numb * initialConditions,					// Массив с начальными условиями
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	new numb[2]{ 0.4, 2.2 },//const numb * ranges,								// Диапазоны изменения параметров
//	//	new int[1] { 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	1,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,
//	//	"D:\\CUDAresults\\LS1D_Faiza_23022026_par_a_02.csv"//std::string		OUT_FILE_PATH
//	//);
//
//
//	// Chua Shirnin FastSynchro 
//
//	/*numb params[1]{0.5};
//numb init[3]{-4.50765, -0.270632,   0.110443};
//numb f = 100251;// Hz
//numb h = 1.0 / f;
//numb CT = (numb)100000 * h;
//numb TT = (numb)10000 * h;
//numb WT = (numb)500 * h;
//numb start = -200;
//numb stop = 200;
//
//std::string path;
//
//WT = (numb)250 * h;
//FastSynchro(
//	CT,											//const numb	tMax,								// Время моделирования системы
//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//	params,										//const numb* values,								// Параметры
//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//	h,										//const numb	h,									// Шаг интегрирования
//	new numb[3]{ 50000, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//	new numb[3]{ 5000, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//	new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//	1000,										//const int		iterOfSynchr,						// Число итераций синхронизации
//	4,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	"D:\\CUDAresults\\FS_ChuaShirnin_x_K=5000_WT=250.csv"//std::string		OUT_FILE_PATH
//);
//
//WT = (numb)200 * h;
//FastSynchro(
//	CT,											//const numb	tMax,								// Время моделирования системы
//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//	params,										//const numb* values,								// Параметры
//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//	h,										//const numb	h,									// Шаг интегрирования
//	new numb[3]{  50000, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//	new numb[3]{ 5000, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//	new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//	1000,										//const int		iterOfSynchr,						// Число итераций синхронизации
//	4,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	"D:\\CUDAresults\\FS_ChuaShirnin_x_K=5000_WT=200.csv"//std::string		OUT_FILE_PATH
//);
//
//WT = (numb)150 * h;
//FastSynchro(
//	CT,											//const numb	tMax,								// Время моделирования системы
//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//	params,										//const numb* values,								// Параметры
//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//	h,										//const numb	h,									// Шаг интегрирования
//	new numb[3]{  50000, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//	new numb[3]{ 5000, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//	new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//	1000,										//const int		iterOfSynchr,						// Число итераций синхронизации
//	4,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	"D:\\CUDAresults\\FS_ChuaShirnin_x_K=5000_WT=150.csv"//std::string		OUT_FILE_PATH
//);
//
//WT = (numb)100 * h;
//FastSynchro(
//	CT,											//const numb	tMax,								// Время моделирования системы
//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//	params,										//const numb* values,								// Параметры
//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//	h,										//const numb	h,									// Шаг интегрирования
//	new numb[3]{  50000, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//	new numb[3]{  5000, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//	new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//	1000,										//const int		iterOfSynchr,						// Число итераций синхронизации
//	4,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	"D:\\CUDAresults\\FS_ChuaShirnin_x_K=5000_WT=100.csv"//std::string		OUT_FILE_PATH
//);
//
//WT = (numb)60 * h;
//FastSynchro(
//	CT,											//const numb	tMax,								// Время моделирования системы
//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//	params,										//const numb* values,								// Параметры
//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//	h,										//const numb	h,									// Шаг интегрирования
//	new numb[3]{  50000, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//	new numb[3]{ 5000, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//	new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//	1000,										//const int		iterOfSynchr,						// Число итераций синхронизации
//	4,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	"D:\\CUDAresults\\FS_ChuaShirnin_x_K=5000_WT=60.csv"//std::string		OUT_FILE_PATH
//);
//*/
//
//	//for (int i = 50; i >= 0; i--) {
////	for (int j = 50; j >= i; j--) {
////		numb Kf = start + i*(stop - start) / 50.0;
////		numb Kb = start + j*(stop - start) / 50.0;
////		path = "D:\\CUDAresults\\FS_ChuaShirnin_Kf=" + std::to_string(Kf) + "_Kb=" + std::to_string(Kb) + ".csv";
////
////		FastSynchro(
////			CT,											//const numb	tMax,								// Время моделирования системы
////			TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////			WT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
////			params,										//const numb* values,								// Параметры
////			sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
////			h,										//const numb	h,									// Шаг интегрирования
////			new numb[3]{ Kf, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
////			new numb[3]{ Kb, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
////			init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
////			new numb[3]{ 1e-4, 1e-4, 1e-4 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
////			sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////			1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
////			10000,										//const int		iterOfSynchr,						// Число итераций синхронизации
////			10,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////			path//std::string		OUT_FILE_PATH
////		);
////	}
////}
//
//
//	// Lorenz-like 5D system
//	/*numb params[7]{0.5, 10, 8.0 / 3.0, 28, -2, 0.4, 8};
//	numb init[5]{ 1, 1, 1, 1, 1 };
//	int iter;
//	FastSynchro(
//		250,										//const numb	tMax,								// Время моделирования системы
//		100,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		0.5,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		params,										//const numb* values,								// Параметры
//		sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		0.002,										//const numb	h,									// Шаг интегрирования
//		new numb[5]{ 0, 0, 10, 0, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		new numb[5]{ 0, 0, 10, 0, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		new numb[5]{ 0.0, 0.0, 0.0, 0.0, 0.0 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		350,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		1,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		"D:\\CUDAresults\\FS_Lorenz5D.csv"//std::string		OUT_FILE_PATH
//	);*/
//
//	// Hindamarsh Rose 
//
//	//numb params[13]{0.5, 1, 3, 1, 5, 0.0125, 4, -1.6, 2.26, 0.01, 2, 0 ,0.5};
//	//numb init[4]{ -1.6, -12, 0.0, 0.0 };
//	//size_t startTime = std::clock();
//	//
//	//basinsOfAttraction(
//	//	1000,									// Время моделирования системы
//	//	301,									// Разрешение диаграммы
//	//	0.01,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -20, 20, -20, 20 },
//	//	new int[2] { 1, 2 },						// Индексы изменяемых параметров
//	//	0,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	2000,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	10,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.2,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_HR_001.csv",
//	//	32
//	//);
//	//bifurcation2D(
//	//	10000, //const numb	tMax,								// Время моделирования системы
//	//	601, //const int		nPts,								// Разрешение диаграммы
//	//	0.01, //const numb	h,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb* initialConditions,					// Массив с начальными условиями
//	//	new numb[4]{ 0.001, 0.01, 1.5, 2.5 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	new int[2] { 5, 8 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
//	//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	10000, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb* values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//	2, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.3,//const numb	eps,
//	//	"D:\\CUDAresults\\bif2D_HR_4.csv" //std::string		OUT_FILE_PATH
//	//);
//	//params[8] = 2.0;
//	//bifurcation1D(
//	//	5000,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.01,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ 2.6, 3.6 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 8 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	10000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_HR_neron.csv"//std::string		OUT_FILE_PATH
//	//);
//	//
//	//LLE1D(
//	//	15000, //const numb	tMax,								// Время моделирования системы
//	//	0.1,	//const numb	NT,									// Время нормализации
//	//	1001,	//const int		nPts,								// Разрешение диаграммы
//	//	0.01,	//const numb	h,									// Шаг интегрирования
//	//	1e-4,	//const numb	eps,								// Эпсилон для LLE
//	//	init,//const numb * initialConditions,					// Массив с начальными условиями
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	new numb[2]{ 2.6, 3.6 },//const numb * ranges,								// Диапазоны изменения параметров
//	//	new int[1] { 8 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	10000,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,
//	//	"D:\\CUDAresults\\LLE1D_HR_neron.csv"//std::string		OUT_FILE_PATH
//	//);
//	//
//	//bifurcation_DFT_1D(
//	//	25000,	//const numb	tMax,								// Время моделирования системы
//	//	1001,	//const int	nPts,								// Разрешение диаграммы
//	//	1000,		//const int	nFreq,								// Разрешение диаграммы
//	//	0.01,	//const numb	h,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//	//	init,	//const numb* initialConditions,					// Массив с начальными условиями
//	//	new numb[2]{ -4, -1.5 },//const numb* ranges,								// Диапазоны изменения параметров
//	//	new numb[2]{ 0.001, 0.025 },//const numb* rangesFreq,								// Диапазоны изменения параметров
//	//	new int[1] { 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//	//	0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	20000,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,	//const numb* values,								// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//	//	20,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,//const numb	eps,
//	//	"D:\\CUDAresults\\bifDFT1D_HR_neron.csv"//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
//	//);
//	//printf("Calculations done in: %zu ms\n", std::clock() - startTime);
//
//	//Lorenz
//	/*numb params[4]{0.5, 10, 28, 2.6667};
//	numb init[3]{ 0.1, 0.1, 15.0 };
//	size_t startTime = std::clock();
//	int indexParameter = 1;
//	numb* ranges = new numb[2]{ 5, 20 };
//
//	bifurcation1D(
//		1000,		//const numb	tMax,							// Время моделирования системы
//		1000,		//const int		nPts,						// Разрешение диаграммы
//		0.0025,		//const numb	h,								// Шаг интегрирования
//		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		init,//const numb * initialConditions,				// Массив с начальными условиями
//		ranges,//const numb * ranges,							// Диаппазон изменения переменной
//		new int[1] { indexParameter },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		2,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		1500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		params,//const numb * values,							// Параметры
//		sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		5,//const int		preScaller,
//		"D:\\CUDAresults\\bif1D_LORENZ.csv"//std::string		OUT_FILE_PATH
//	);
//
//	bifurcation_DFT_1D(
//		1000,	//const numb	tMax,								// Время моделирования системы
//		1000,	//const int	nPts,								// Разрешение диаграммы
//		500,		//const int	nFreq,								// Разрешение диаграммы
//		0.0025,	//const numb	h,									// Шаг интегрирования
//		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		init,	//const numb* initialConditions,					// Массив с начальными условиями
//		ranges,//const numb* ranges,								// Диапазоны изменения параметров
//		new numb[2]{ 1, 1.5 },//const numb* rangesFreq,								// Диапазоны изменения параметров
//		new int[1] { indexParameter },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		2,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		10000000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		1500,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		params,	//const numb* values,								// Параметры
//		sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		5,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		0.15,//const numb	eps,
//		"D:\\CUDAresults\\bifDFT1D_LORENZ.csv"//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN
//	);
//	printf("Calculations done in: %zu ms\n", std::clock() - startTime);*/
//
//	// Rossler _______________________________________________________________________________________________________________________________________________
//	/*
//		numb params[4]{ 0.5, 0.2, 0.2, 5.7 };
//		numb init[3]{ 0.0, 0.1, 0.0 };
//		size_t startTime = std::clock();
//		bifurcation2D(
//			1000, //const numb	tMax,								// Время моделирования системы
//			400, //const int		nPts,								// Разрешение диаграммы
//			0.01, //const numb	h,									// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,					// Массив с начальными условиями
//			new numb[4]{ 0.0, 0.4, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
//			new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//			0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//			100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			1500, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,								// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//			1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.2,//const numb	eps,
//			"D:\\CUDAresults\\bif2D_ROSSLER.csv" //std::string		OUT_FILE_PATH
//		);
//		bifurcation1D(
//			5000,		//const numb	tMax,							// Время моделирования системы
//			500,		//const int		nPts,						// Разрешение диаграммы
//			1,		//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb * initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 3.4, 4 },//const numb * ranges,							// Диаппазон изменения переменной
//			new int[1] { 1},//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			100000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			1500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb * values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			1,//const int		preScaller,
//			"D:\\CUDAresults\\bif1D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);
//		bifurcation_DFT_1D(
//			2000,	//const numb	tMax,								// Время моделирования системы
//			500,	//const int	nPts,								// Разрешение диаграммы
//			500,		//const int	nFreq,								// Разрешение диаграммы
//			0.01,	//const numb	h,									// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//			init,	//const numb* initialConditions,					// Массив с начальными условиями
//			new numb[2]{ 3.4, 4},//const numb* ranges,								// Диапазоны изменения параметров
//			new numb[2]{ 0.00, 10 },//const numb* rangesFreq,								// Диапазоны изменения параметров
//			new int[1] { 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//			0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//			100000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			1500,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//			params,	//const numb* values,								// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//			10,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.15,//const numb	eps,
//			"D:\\CUDAresults\\bifDFT1D_ROSSLER.csv"//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN
//			);
//		printf("features done in: %zu ms\n", std::clock() - startTime);
//		bifurcation1D(
//			1500,		//const numb	tMax,							// Время моделирования системы
//			800,		//const int		nPts,						// Разрешение диаграммы
//			0.01,		//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb * initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 0.5 },//const numb * ranges,							// Диаппазон изменения переменной
//			new int[1] { 2},//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			2,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			100000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			5000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb * values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			1,//const int		preScaller,
//			"D:\\CUDAresults\\bif1D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);
//		LLE1D(
//			500, //const numb	tMax,								// Время моделирования системы
//			1.0,	//const numb	NT,									// Время нормализации
//			1000,	//const int		nPts,								// Разрешение диаграммы
//			0.01,	//const numb	h,									// Шаг интегрирования
//			1e-2,	//const numb	eps,								// Эпсилон для LLE
//			init,//const numb * initialConditions,					// Массив с начальными условиями
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//			new numb[2]{ 0.0, 0.5 },//const numb * ranges,								// Диапазоны изменения параметров
//			new int[1] { 2},//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//			0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//			100000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			1500,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb * values,								// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,
//			"D:\\CUDAresults\\LLE1D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);					// Количество параметров
//
//		LLE2D(
//			5000,	//const numb tMax,
//			1.0,	//const numb NT,
//			2001,	//const int nPts,
//			0.01,	//const numb h,
//			1e-3,	//const numb eps,
//			init,	//const numb * initialConditions,
//			sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//			new numb[4]{ 0.0, 0.5, 0.0, 0.5 },//const numb * ranges,
//			new int[2] { 2, 1},//const int* indicesOfMutVars,
//			0,		//const int writableVar,
//			100000,	//const numb maxValue,
//			1500,	//const numb transientTime,
//			params,	//const numb * values,
//			sizeof(params) / sizeof(numb),//const int amountOfValues,
//			"D:\\CUDAresults\\LLE2D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);
//		LS1D(
//			10000, //const numb	tMax,								// Время моделирования системы
//			0.1,	//const numb	NT,									// Время нормализации
//			800,	//const int		nPts,								// Разрешение диаграммы
//			0.01,	//const numb	h,									// Шаг интегрирования
//			1e-7,	//const numb	eps,								// Эпсилон для LLE
//			init,//const numb * initialConditions,					// Массив с начальными условиями
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//			new numb[2]{ 0.0, 0.5 },//const numb * ranges,								// Диапазоны изменения параметров
//			new int[1] { 2},//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//			0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//			100000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			5000,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb * values,								// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,
//			"D:\\CUDAresults\\LS1D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);
//		LS2D(
//			1500,	//const numb tMax,
//			1.0,	//const numb NT,
//			1001,	//const int nPts,
//			0.02,	//const numb h,
//			1e-3,	//const numb eps,
//			init,	//const numb * initialConditions,
//			sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//			new numb[4]{ 0.0, 0.5, 0.0, 0.5 },//const numb * ranges,
//			new int[2] { 2, 1},//const int* indicesOfMutVars,
//			0,		//const int writableVar,
//			100000,	//const numb maxValue,
//			1500,	//const numb transientTime,
//			params,	//const numb * values,
//			sizeof(params) / sizeof(numb),//const int amountOfValues,
//			"D:\\CUDAresults\\LS2D_ROSSLER.csv"//std::string		OUT_FILE_PATH
//		);
//		*/
//
//
//	// Faiza 6D _______________________________________________________________________________________________________________________________________________
//
//	//numb params[8]{ 0.5, 5.5, 3.7, 2.5, 0.5, 0.5, 0.3, 0.02 };
//		//numb init[6]{ 0.1, 0.1, 0.0, 0.0, 0.0, 0.0 };
//		//numb CT = 2000;
//		//numb TT = 100000;
//		//
//		//LLE2D(
//		//	CT,	//const numb tMax,
//		//	1.0,	//const numb NT,
//		//	101,	//const int nPts,
//		//	0.01,	//const numb h,
//		//	1e-5,	//const numb eps,
//		//	init,	//const numb * initialConditions,
//		//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		//	new numb[4]{ 2.0, 3.0, 0.4, 0.7 },//const numb * ranges,
//		//	new int[2] { 3, 4 },//const int* indicesOfMutVars,
//		//	0,		//const int writableVar,
//		//	100000,	//const numb maxValue,
//		//	TT,	//const numb transientTime,
//		//	params,	//const numb * values,
//		//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_Faiza6D_003.csv"//std::string		OUT_FILE_PATH
//		//);
//		//bifurcation2D(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	101, //const int		nPts,								// Разрешение диаграммы
//		//	0.01, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4]{ 2.0, 3.0, 0.4, 0.7 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 3, 4 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	5, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.02,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2D_Faiza6D_003.csv" //std::string		OUT_FILE_PATH
//		//);
//		//bifurcation1D(
//		//	CT,		//const numb	tMax,							// Время моделирования системы
//		//	2001,		//const int		nPts,						// Разрешение диаграммы
//		//	0.01,		//const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb * initialConditions,				// Массив с начальными условиями
//		//	new numb[2]{ 1.5, 2.5 },//const numb * ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 3 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb * values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	2,//const int		preScaller,
//		//	"D:\\CUDAresults\\bif1D_Faiza6D_003.csv"//std::string		OUT_FILE_PATH
//		//);
//		//basinsOfAttraction(
//		//	CT,									// Время моделирования системы
//		//	101,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -1, 1, -1, 1 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	2,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.1,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_Faiza6D_001.csv",
//		//	32
//		//);
//		//LS1D(
//		//	1*CT, //const numb	tMax,								// Время моделирования системы
//		//	1.0,	//const numb	NT,									// Время нормализации
//		//	251,	//const int		nPts,								// Разрешение диаграммы
//		//	0.01,	//const numb	h,									// Шаг интегрирования
//		//	1e-6,	//const numb	eps,								// Эпсилон для LLE
//		//	init,//const numb * initialConditions,					// Массив с начальными условиями
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	new numb[2]{ 0.4, 0.65 },//const numb * ranges,								// Диапазоны изменения параметров
//		//	new int[1] { 4 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	0,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb * values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,
//		//	"D:\\CUDAresults\\LS1D_Faiza6D_001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//
//	// Muni Aditional figures
//
//	//numb init[2] = { 0.5, 1.2 }; // начальные условия
////numb params[11] = {
////		0.5,   
////		0.08,  //alpha
////		0.3,   //beta
////		0.8,   //gamma
////		0.67,  //c
////		0.5,   //d
////		2,	   //p
////		1.8,   //rho
////		2,	   //omega
////		0.5,   //delta
////		1.3	   //h
////};
////numb params[11] = {
////		0.5,
////		0.6,  //alpha
////		0.3,   //beta
////		0.8,   //gamma
////		0.67,  //c
////		0.4,   //d
////		3.4,	   //p
////		1.8,   //rho
////		2.2,	   //omega
////		0.4,   //delta
////		1.3	   //h
////};
////numb params[11] = {
////		0.5,   
////		0.08,	//alpha
////		0.15,   //beta
////		0.8,	//gamma
////		0.67,	//c
////		0.1,	//d
////		3.5,	//p
////		3.5,	//rho
////		0.2,	//omega
////		0.15,   //delta
////		1.3		//h
////};
////bifurcation2D(
////	400, //const numb	tMax,								// Время моделирования системы
////	801, //const int		nPts,								// Разрешение диаграммы
////	1.0, //const numb	h,									// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	init,//const numb* initialConditions,					// Массив с начальными условиями
////	//new numb[4]{ 0, 0.2, 0.1, 0.25 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 8, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	new numb[4]{ 0, 30, 0, 1 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[2] { 7, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 100, 1000, 4500, 7500 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 4, 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
////	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	1e13, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	20000, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	0.1,//const numb	eps,
////	"D:\\CUDAresults\\Fig9_bif2D_Muni2.csv" //std::string		OUT_FILE_PATH
////);
////LLE2D(
////	4000,	//const numb tMax,
////	1,	//const numb NT,
////	801,	//const int nPts,
////	1,	//const numb h,
////	1e-6,	//const numb eps,
////	init,	//const numb * initialConditions,
////	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
////	//new numb[4]{ 0, 0.2, 0.1, 0.25 },//const numb * ranges,
////	//new int[2]{ 8, 2 },//const int* indicesOfMutVars,
////	new numb[4]{ 0, 30, 0, 1 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[2] { 7, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	0,		//const int writableVar,
////	1e13,	//const numb maxValue,
////	20000,	//const numb transientTime,
////	params,	//const numb * values,
////	sizeof(params) / sizeof(numb),//const int amountOfValues,
////	"D:\\CUDAresults\\LLE2D_Fig9_Muni.csv"//std::string		OUT_FILE_PATH
////);
//
//	//bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 0.2 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 8 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\fig10_bif1D_Muni_x.csv"//std::string		OUT_FILE_PATH
////);
////
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 0.2 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 8 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\fig10_bif1D_Muni_y.csv"//std::string		OUT_FILE_PATH
////);
//
//	//bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 1 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\fig6_bif1D_Muni_x.csv"//std::string		OUT_FILE_PATH
////);
////
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 1 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\fig6_bif1D_Muni_y.csv"//std::string		OUT_FILE_PATH
////);
////params[2] = 0.3;
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 7 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 7 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_Muni_beta=0.3.csv"//std::string		OUT_FILE_PATH
////);
////LLE1D(
////	2000,//const numb	tMax,								// Время моделирования системы
////	1.0,//const numb	NT,									// Время нормализации
////	2001,//const int		nPts,								// Разрешение диаграммы
////	1,//const numb	h,									// Шаг интегрирования
////	1e-4,//const numb	eps,								// Эпсилон для LLE
////	init, //const numb* initialConditions,					// Массив с начальными условиями
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	new numb[2]{ 0, 7 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[1] { 7 },//const int* indicesOfMutVars,						// Индексы изменяемых параметров
////	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	"D:\\CUDAresults\\LLE1D_Muni_beta=0.3.csv"//std::string		OUT_FILE_PATH
////);
////params[2] = 0.5;
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 7 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 7 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_Muni_beta=0.5.csv"//std::string		OUT_FILE_PATH
////);
////
////params[2] = 0.7;
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 12 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 7 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_Muni_beta=0.7.csv"//std::string		OUT_FILE_PATH
////);
////
////params[2] = 0.8;
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 18 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 7 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_Muni_beta=0.8.csv"//std::string		OUT_FILE_PATH
////);
////
////params[2] = 0.9;
////bifurcation1D(
////	1000,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 35 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 7 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	2000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_Muni_beta=0.9.csv"//std::string		OUT_FILE_PATH
////);
////LLE2D(
////	4000,	//const numb tMax,
////	1,	//const numb NT,
////	101,	//const int nPts,
////	1,	//const numb h,
////	1e-2,	//const numb eps,
////	init,	//const numb * initialConditions,
////	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
////	new numb[4]{ 0, 30, 0, 1.0 },//const numb * ranges,
////	new int[2]{ 7, 2 },//const int* indicesOfMutVars,
////	0,		//const int writableVar,
////	1e25,	//const numb maxValue,
////	2000,	//const numb transientTime,
////	params,	//const numb * values,
////	sizeof(params) / sizeof(numb),//const int amountOfValues,
////	"D:\\CUDAresults\\LLE2D_Muni.csv"//std::string		OUT_FILE_PATH
////);
////bifurcation2D(
////	400, //const numb	tMax,								// Время моделирования системы
////	801, //const int		nPts,								// Разрешение диаграммы
////	1.0, //const numb	h,									// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	init,//const numb* initialConditions,					// Массив с начальными условиями
////	new numb[4]{ 0, 30, 0, 1.0 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[2] { 7, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 100, 1000, 4500, 7500 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 4, 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
////	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
////	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	1e25, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	20000, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	0.1,//const numb	eps,
////	"D:\\CUDAresults\\bif2D_Muni.csv" //std::string		OUT_FILE_PATH
////);
//
//
//// Matreshka 3.0;
//
//	// a - a[1]
//	// b - a[2]
//	// m - a[3]
//	// d - a[4]
//	// k - a[5]
//	// dmargin - a[6]
//
////numb params[7]{ 0.5, 15.375, 27, 7, 1.5, 20.1, 3.1 };
//////numb params[3]{ 0.5, 15.375, 27 };
////numb init[3]{ 0, 0, 0 };
////
////basinsOfAttraction(
////	500,         // Время моделирования системы
////	201,         // Разрешение диаграммы
////	0.001,         // Шаг интегрирования
////	sizeof(init) / sizeof(numb),   // Количество начальных условий ( уравнений в системе )
////	init,         // Массив с начальными условиями
////	//new numb[4]{ -100, 50, -50, 100 },
////	new numb[4]{ -35, 35, -35, 35 },
////	//new numb[4]{ -25, -20, 20, 25 },
////	//new numb[4]{ -22.8, -22, 22, 22.8 },
////	//new numb[4]{ -22.34, -22.26, 22.24, 22.34 },
////	//new numb[4]{ -22.308, -22.294, 22.294, 22.308 },
////	//new numb[4]{ -22.303, -22.301, 22.301, 22.303 },
////	new int[2] { 0, 2 },      // Индексы изменяемых параметров
////	0,          // Индекс уравнения, по которому будем строить диаграмму
////	100000000,        // Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	500,         // Время, которое будет промоделировано перед расчетом диаграммы
////	params,         // Параметры
////	sizeof(params) / sizeof(numb),  // Количество параметров
////	1,          // Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	0.25,         // Эпсилон для алгоритма DBSCAN
////	"D:\\CUDAresults\\Chua_Matryoshkaa2.csv",
////	32
////);
//
//
//// TIKARIMOV NEURON -------
//
////numb init[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // начальные условия
////numb params[13] = {
////		1.515,   // 0 Vin
////		10000,		// 1 R
////		1.0e-3,		// 2 C1 uF
////		1.395e-3,		// 3 C2 uF
////		360,	// 4 L  uHn
////		2090,// 5 R_G
////		806,		// 6 Ron
////		1.0e6,		// 7 Roff
////		0.99,		// 8 TAU uTAU
////		20,			// 9 TAU2 uTAU
////		0.267,		// 10 Von1
////		0.1,		// 11 Voff1
////		0.0			// 12 d
////	};
//
////numb params[13] = {
////		1.6395,   // 0 Vin
////		10000,		// 1 R
////		1.0e-3,		// 2 C1 uF
////		1.395e-3,		// 3 C2 uF
////		340,	// 4 L  uHn
////		2850,// 5 R_G
////		806,		// 6 Ron
////		1.0e6,		// 7 Roff
////		0.99,		// 8 TAU uTAU
////		20,			// 9 TAU2 uTAU
////		0.267,		// 10 Von1
////		0.1,		// 11 Voff1
////		0.0			// 12 d
////};
//
//
////LLE2D(
////	600,	//const numb tMax,
////	2.0,	//const numb NT,
////	601,	//const int nPts,
////	0.001,	//const numb h,
////	1e-5,	//const numb eps,
////	init,	//const numb * initialConditions,
////	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
////	new numb[4]{ 0, 0.02, 1.6, 1.7 },//const numb * ranges,
////	new int[2] { 12, 0},//const int* indicesOfMutVars,
////	5,		//const int writableVar,
////	100000000,	//const numb maxValue,
////	200,	//const numb transientTime,
////	params,	//const numb * values,
////	sizeof(params) / sizeof(numb),//const int amountOfValues,
////	"D:\\CUDAresults\\LLE2D_tikarimovneuron_rev2.csv"//std::string		OUT_FILE_PATH
////);
//
////bifurcation2D(
////	200, //const numb	tMax,								// Время моделирования системы
////	101, //const int		nPts,								// Разрешение диаграммы
////	0.002, //const numb	h,									// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	init,//const numb* initialConditions,					// Массив с начальными условиями
////	new numb[4]{ 0, 0.02, 1.6, 1.7 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[2] { 12, 0 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров					// Индексы изменяемых параметро
////	5, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	1e25, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	200, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	1.0,//const numb	eps,
////	"D:\\CUDAresults\\bif2D_tikarimovneuron_rev1.csv" //std::string		OUT_FILE_PATH
////);
//
////bifurcation1D(
////	500,		//const numb	tMax,							// Время моделирования системы
////	1001,		//const int		nPts,						// Разрешение диаграммы
////	0.002,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 0.02 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 12 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	5,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_TIKARIMOV_V2_d_forward_noInterp.csv"//std::string		OUT_FILE_PATH
////);
//
//
////	bifurcation1D(
////		2500,		//const numb	tMax,							// Время моделирования системы
////		2001,		//const int		nPts,						// Разрешение диаграммы
////		0.002,		//const numb	h,								// Шаг интегрирования
////		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////		init,//const numb * initialConditions,				// Массив с начальными условиями
////		new numb[2]{ 0.02, 0.0 },//const numb * ranges,							// Диаппазон изменения переменной
////		new int[1] { 12 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////		5,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////		10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////		500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////		params,//const numb * values,							// Параметры
////		sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////		1,//const int		preScaller,
////		"D:\\CUDAresults\\bif1D_TIKARIMOV_V2_d_backward_forRNF.csv"//std::string		OUT_FILE_PATH
////	);
////
////	bifurcation_DFT_1D(
////		2500,	//const numb	tMax,								// Время моделирования системы
////		2001,	//const int	nPts,								// Разрешение диаграммы
////		2001,		//const int	nFreq,								// Разрешение диаграммы
////		0.002,	//const numb	h,									// Шаг интегрирования
////		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////		init,	//const numb* initialConditions,					// Массив с начальными условиями
////		new numb[2]{ 0.02, 0.0 },//const numb* ranges,								// Диапазоны изменения параметров
////		new numb[2]{ 0.01, 1 },//const numb* rangesFreq,								// Диапазоны изменения параметров
////		new int[1] { 12 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////		5,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////		10000000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////		500,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////		params,	//const numb* values,								// Параметры
////		sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////		20,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////		0.15,//const numb	eps,
////		"D:\\CUDAresults\\bif1D_DFT_TIKARIMOV_V2_d_backward_forRNF.csv"//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
////);
//
//////params[5] = 6000;
//////bifurcation2D(
//////	300, //const numb	tMax,								// Время моделирования системы
//////	51, //const int		nPts,								// Разрешение диаграммы
//////	0.0002, //const numb	h,									// Шаг интегрирования
//////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//////	init,//const numb* initialConditions,					// Массив с начальными условиями
//////	//new numb[4]{ 0, 1, 0, 10000 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 0, 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	new numb[4]{ 360, 365, 12000, 27000 },//const numb* ranges,								// Диапазоны изменения параметров
//////	new int[2] { 4, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 0, 60000, 0, 10000 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 1, 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
//////	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//////	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//////	500, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//////	params,//const numb* values,								// Параметры
//////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//////	10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//////	0.02,//const numb	eps,
//////	"D:\\CUDAresults\\bif2D_TIKARIMOV_1_5_002.csv" //std::string		OUT_FILE_PATH
//////);
////
//////bifurcation2D(
//////	300, //const numb	tMax,								// Время моделирования системы
//////	51, //const int		nPts,								// Разрешение диаграммы
//////	0.0002, //const numb	h,									// Шаг интегрирования
//////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//////	init,//const numb* initialConditions,					// Массив с начальными условиями
//////	new numb[4]{ 1e-3, 50e-3, 1e-3, 50e-3 },//const numb* ranges,								// Диапазоны изменения параметров
//////	new int[2] { 1, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 100, 1000, 4500, 7500 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 4, 5 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 0.001, 0.01, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 9, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//////	//new numb[4]{ 3, 10, 0, 0.5 },//const numb* ranges,								// Диапазоны изменения параметров
//////	//new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметро
//////	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//////	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//////	600, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//////	params,//const numb* values,								// Параметры
//////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//////	10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//////	0.02,//const numb	eps,
//////	"D:\\CUDAresults\\bif2D_TIKARIMOV_1_2.csv" //std::string		OUT_FILE_PATH
//////);
////
////params[1] = 21900;
////params[5] = 6000;
////bifurcation1D(
////	300,		//const numb	tMax,							// Время моделирования системы
////	601,		//const int		nPts,						// Разрешение диаграммы
////	0.0002,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 20, 00 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 12 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	1,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	300,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	10,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_TIKARIMOV.csv"//std::string		OUT_FILE_PATH
////);
////
////LLE1D(
////	500,//const numb	tMax,								// Время моделирования системы
////	2.0,//const numb	NT,									// Время нормализации
////	1001,//const int		nPts,								// Разрешение диаграммы
////	0.0002,//const numb	h,									// Шаг интегрирования
////	1e-4,//const numb	eps,								// Эпсилон для LLE
////	init, //const numb* initialConditions,					// Массив с начальными условиями
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	new numb[2]{ 100, 1000 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[1] { 4 },//const int* indicesOfMutVars,						// Индексы изменяемых параметров
////	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	1000,//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	"D:\\CUDAresults\\LLE1D_TIKARIMOV.csv"//std::string		OUT_FILE_PATH
////);
//
////LLE2D(
////	5e-3,	//const numb tMax,
////	2e-6,	//const numb NT,
////	101,	//const int nPts,
////	2e-9,	//const numb h,
////	1e-6,	//const numb eps,
////	init,	//const numb * initialConditions,
////	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
////	new numb[4]{ 100e-6, 1000e-6, 4500, 7500 },//const numb * ranges,
////	new int[2]{ 4, 5 },//const int* indicesOfMutVars,
////	1,		//const int writableVar,
////	10000000,	//const numb maxValue,
////	5e-3,	//const numb transientTime,
////	params,	//const numb * values,
////	sizeof(params) / sizeof(numb),//const int amountOfValues,
////	"D:\\CUDAresults\\LLE2D_TIKARIMOV.csv"//std::string		OUT_FILE_PATH
////);
//
////bifurcation1D(
////	1e-3,		//const numb	tMax,							// Время моделирования системы
////	2001,		//const int		nPts,						// Разрешение диаграммы
////	1e-8,		//const numb	h,								// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
////	init,//const numb * initialConditions,				// Массив с начальными условиями
////	new numb[2]{ 0, 1 },//const numb * ranges,							// Диаппазон изменения переменной
////	new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
////	5,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
////	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	5e-3,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb * values,							// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
////	1,//const int		preScaller,
////	"D:\\CUDAresults\\bif1D_TIKARIMOV.csv"//std::string		OUT_FILE_PATH
////);
//
//
////Pala-Machaczek from Cudaynamics
///*numb params[6]{10, 1, 3.14, -2.0 / 3.0};
//numb init[3]{ 0.5, 4.0, 1.0 };
//	
////bifurcation2D(
////	300, //const numb	tMax,								// Время моделирования системы
////	100, //const int		nPts,								// Разрешение диаграммы
////	0.005, //const numb	h,									// Шаг интегрирования
////	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////	init,//const numb* initialConditions,					// Массив с начальными условиями
////	new numb[4]{ 5, 25, 0.45, 1.45 },//const numb* ranges,								// Диапазоны изменения параметров
////	new int[2] { 0, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////	100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	1700, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////	params,//const numb* values,								// Параметры
////	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	0.1,//const numb	eps,
////	"D:\\CUDAresults\\bif2D_palaMachaczek_Cudaynamics.csv" //std::string		OUT_FILE_PATH
////);
//
//	bifurcation2D(
//		300, //const numb	tMax,								// Время моделирования системы
//		100, //const int		nPts,								// Разрешение диаграммы
//		0.005, //const numb	h,									// Шаг интегрирования
//		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		init,//const numb* initialConditions,					// Массив с начальными условиями
//		new numb[4]{ 5, 25, 0.45, 1.45 },//const numb* ranges,								// Диапазоны изменения параметров
//		new int[2] { 0, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		1700, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		params,//const numb* values,								// Параметры
//		sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		1.0,//const numb	eps,
//		"D:\\CUDAresults\\bif2D_palaMachaczek_Cudaynamics.csv" //std::string		OUT_FILE_PATH
//);
//	*/
//
//// SPrott 14
//
//	//numb params[4]{ 0.5, 2.0, 1.0, 2.0 };
//	//numb init[3]{ 0.0, 0, 0 };
//
//	//bifurcation1D(
//	//	1500,		//const numb	tMax,							// Время моделирования системы
//	//	10001,		//const int		nPts,						// Разрешение диаграммы
//	//	0.01,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ -1, 3 },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	5000,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\Bif1D_Sprott14_varX_2.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	//numb par_1_start = 1.4;
//	//numb par_2_start = 1.1;
//	//numb par_1_stop =  1.8;
//	//numb par_2_stop =  1.5;
//	//params[2] = 1.0;
//	//int par_1_res = 5;
//	//int par_2_res = 5;
//
//	//LS2D(
//	//		5000,	//const numb tMax,
//	//		2.0,	//const numb NT,
//	//		201,	//const int nPts,
//	//		0.05,	//const numb h,
//	//		1e-5,	//const numb eps,
//	//		init,	//const numb * initialConditions,
//	//		sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//		new numb[4]{ -1, 3, 1.4, 1.8 },//const numb * ranges,
//	//		new int[2] { 0, 1},//const int* indicesOfMutVars,
//	//		1,		//const int writableVar,
//	//		100000000,	//const numb maxValue,
//	//		50000,	//const numb transientTime,
//	//		params,	//const numb * values,
//	//		sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//		"D:\\CUDAresults\\LS2D_Sprott14_var_vs_par_1.csv"//std::string		OUT_FILE_PATH
//	//	);
//
//	//for (int i = 0; i < par_1_res; i++) {
//	//	for (int j = 0; j < par_2_res; j++) {
//	//		std::string path = "D:\\CUDAresults\\LS2D_Sprott14_8_i=" + std::to_string((int)i) + "_j=" + std::to_string((int)j) + ".csv";
//	//		params[1] = par_1_start + (numb)i*(par_1_stop - par_1_start) / ((numb)par_1_res - 1.0);
//	//		params[3] = par_2_start + (numb)j*(par_2_stop - par_2_start) / ((numb)par_2_res - 1.0);
//	//		LS2D(
//	//			5000,	//const numb tMax,
//	//			2.0,	//const numb NT,
//	//			51,	//const int nPts,
//	//			0.05,	//const numb h,
//	//			1e-5,	//const numb eps,
//	//			init,	//const numb * initialConditions,
//	//			sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//	//			new numb[4]{ -1, 3, -2, 2 },//const numb * ranges,
//	//			new int[2] { 0, 1},//const int* indicesOfMutVars,
//	//			1,		//const int writableVar,
//	//			100000000,	//const numb maxValue,
//	//			1500,	//const numb transientTime,
//	//			params,	//const numb * values,
//	//			sizeof(params) / sizeof(numb),//const int amountOfValues,
//	//			path//std::string		OUT_FILE_PATH
//	//		);
//	//	}
//	//}
//
////basinsOfAttraction(
////	1000,									// Время моделирования системы
////	301,									// Разрешение диаграммы
////	0.01,									// Шаг интегрирования
////	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
////	init,									// Массив с начальными условиями
////	new numb[4]{ -1, 3, -2, 2 },
////	new int[2] { 0, 1 },					// Индексы изменяемых параметров
////	2,										// Индекс уравнения, по которому будем строить диаграмму
////	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////	1000,										// Время, которое будет промоделировано перед расчетом диаграммы
////	params,									// Параметры
////	sizeof(params) / sizeof(numb),			// Количество параметров
////	10,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////	0.2,									// Эпсилон для алгоритма DBSCAN
////	"D:\\CUDAresults\\Basins_Sprott14.csv",
////	32
////);
//
//// Bo Sang 14.05.2026 ------------------------------------------------------------------------------------------------------------------------------------------------
////k[0][j] = -X1[1] + a[2] * sin(X1[2]);
////k[1][j] = X1[0];
////k[2][j] = a[1] * X1[0] - X1[2];
//
//		//numb params[3]{ 0.5, 1.3, 10 };
//		//numb init[3]{ 0.0, 0.0, 0.0 };
//		//numb init[3]{ 11.0, 0.0, 1.3 };
//		//numb init[3]{ 0.0, 0.0, 1.0};
//
//		//bifurcation2D(
//		//	500, //const numb	tMax,								// Время моделирования системы
//		//	301, //const int		nPts,								// Разрешение диаграммы
//		//	0.01, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4]{ -40, 40, -3, 3 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 2, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	5000, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	10, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2D_BoSang_1405_3.csv" //std::string		OUT_FILE_PATH
//		//);
//
//		//	bifurcation1D(
//		//	500,		//const numb	tMax,							// Время моделирования системы
//		//	1001,		//const int		nPts,						// Разрешение диаграммы
//		//	0.01,		//const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb * initialConditions,				// Массив с начальными условиями
//		//	new numb[2]{ 0, -10 },//const numb * ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb * values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,
//		//	"D:\\CUDAresults\\bif1D_BoSang_1405_1.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//basinsOfAttraction(
//		//	1000,									// Время моделирования системы
//		//	301,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -50, 50, -50, 50 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang_1405_a=1.3_F=10_3.csv",
//		//	32
//		//);
//		//params[2] = -10;
//		//basinsOfAttraction(
//		//	1000,									// Время моделирования системы
//		//	301,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -50, 50, -50, 50 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang_1405_a=1.3_F=-10_3.csv",
//		//	32
//		//);
//
//// Rybin BoSang 18.05.2026
//
////k[0][j] = -X1[1] + a[1] * sin(X1[2]);
////k[1][j] = X1[0];
////k[2][j] = -a[2] * X1[2] + a[3] * X1[2] * X1[2] * X1[2] + a[4] * X1[0];
//
//		numb params[5]{ 0.5, -15, -1.0, -0.0001, 1.0 };
//		numb init[3]{ 0.0, 0.0, 0.0 };
//		basinsOfAttraction(
//			1000,									// Время моделирования системы
//			101,									// Разрешение диаграммы
//			0.01,									// Шаг интегрирования
//			sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//			init,									// Массив с начальными условиями
//			new numb[4]{ -100, 100, -100, 100 },
//			new int[2] { 0, 1 },					// Индексы изменяемых параметров
//			1,										// Индекс уравнения, по которому будем строить диаграмму
//			100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			5000,										// Время, которое будет промоделировано перед расчетом диаграммы
//			params,									// Параметры
//			sizeof(params) / sizeof(numb),			// Количество параметров
//			2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.2,									// Эпсилон для алгоритма DBSCAN
//			"D:\\CUDAresults\\Basins_BoSang_1805_F=-15_1.csv",
//			32
//		);
//		//basinsOfAttraction(
//		//	1750,									// Время моделирования системы
//		//	401,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -120, 120, -120, 120 },
//		//	new int[2] { 0, 2 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	5000,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang_1805_2_2.csv",
//		//	32
//		//);
//		basinsOfAttraction(
//			1000,									// Время моделирования системы
//			101,									// Разрешение диаграммы
//			0.01,									// Шаг интегрирования
//			sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//			init,									// Массив с начальными условиями
//			new numb[4]{ -100, 100, -100, 100 },
//			new int[2] { 1, 2 },					// Индексы изменяемых параметров
//			1,										// Индекс уравнения, по которому будем строить диаграмму
//			100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			5000,										// Время, которое будет промоделировано перед расчетом диаграммы
//			params,									// Параметры
//			sizeof(params) / sizeof(numb),			// Количество параметров
//			2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.2,									// Эпсилон для алгоритма DBSCAN
//			"D:\\CUDAresults\\Basins_BoSang_1805_F=-15_3.csv",
//			32
//		);
//
//		//params[1] = -params[1];
//		//basinsOfAttraction(
//		//	1000,									// Время моделирования системы
//		//	201,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -10, 10, -10, 10 },
//		//	new int[2] { 0, 2 },					// Индексы изменяемых параметров
//		//	2,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_Rybin-BoSang_1805_2.csv",
//		//	32
//		//);
//
////	BoSang Cubic 18.05.2026
///*//k[0][j] = -X1[1] + a[1] * sin(X1[2]);
////k[1][j] = X1[0];
////k[2][j] = -a[2] * X1[2] + a[3] * X1[2] * X1[2] * X1[2] + a[4] * X1[0];
//
//		numb params[5]{ 0.5, 10, 1.0, 0.0, 1.5};
//		numb init[3]{ 0.0, 0.0, 0.0 };
//
//		//basinsOfAttraction(
//		//	500,									// Время моделирования системы
//		//	201,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -20, 20, -20, 20 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSangCubic_1805_1.csv",
//		//	32
//		//);
//
//		numb par_1_start = -20;
//		numb par_1_stop  =  20;
//		numb par_2_start = -0.001;
//		numb par_2_stop  =  0.0;
//		int par_1_res    = 9;
//		int par_2_res    = 9;
//
//
//
//for (int i = 0; i < par_1_res; i++) {
//	for (int j = 0; j < par_2_res; j++) {
//		std::string path = "D:\\CUDAresults\\Basins_BoSangCubic_6_i=" + std::to_string((int)i) + "_j=" + std::to_string((int)j) + ".csv";
//		params[1] = par_1_start + (numb)i*(par_1_stop - par_1_start) / ((numb)par_1_res - 1.0);
//		params[3] = par_2_start + (numb)j*(par_2_stop - par_2_start) / ((numb)par_2_res - 1.0);
//
//		basinsOfAttraction(
//			1000,									// Время моделирования системы
//			61,									// Разрешение диаграммы
//			0.01,									// Шаг интегрирования
//			sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//			init,									// Массив с начальными условиями
//			new numb[4]{ -50, 50, -50, 50 },
//			new int[2] { 0, 1 },					// Индексы изменяемых параметров
//			1,										// Индекс уравнения, по которому будем строить диаграмму
//			100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			5000,										// Время, которое будет промоделировано перед расчетом диаграммы
//			params,									// Параметры
//			sizeof(params) / sizeof(numb),			// Количество параметров
//			2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//			0.2,									// Эпсилон для алгоритма DBSCAN
//			path,
//			32
//		);
//		path = "D:\\CUDAresults\\LLE_BoSangCubic_6_i=" + std::to_string((int)i) + "_j=" + std::to_string((int)j) + ".csv";
//		LLE2D(
//			2500,	//const numb tMax,
//			1.0,	//const numb NT,
//			61,	//const int nPts,
//			0.01,	//const numb h,
//			1e-5,	//const numb eps,
//			init,	//const numb * initialConditions,
//			sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//			new numb[4]{ -50, 50, -50, 50 },//const numb * ranges,
//			new int[2]{ 0, 1 },//const int* indicesOfMutVars,
//			1,		//const int writableVar,
//			10000000,	//const numb maxValue,
//			5000,	//const numb transientTime,
//			params,	//const numb * values,
//			sizeof(params) / sizeof(numb),//const int amountOfValues,
//			path//std::string		OUT_FILE_PATH
//		);
//	}
//}*/
//
//// Bo Sang 23.11.2025 ------------------------------------------------------------------------------------------------------------------------------------------------
//	
//	////k[0][j] = -X1[1] + X1[2];
//	////k[1][j] = X1[0];
//	////k[2][j] = -a[1]*X1[2] + a[2] * sin(X1[0]);
//		//numb params[3]{ 0.5, 1, 20 };
//		//numb init[3]{ 4.0, 0.0, 0.0 };
//		//numb init_1[3]{ 4.0, 0, 0 };
//		//numb init_2[3]{ -4.0, 0, 0 };
//		//numb ranges_1[2]{ 30, -30};
//		//numb ranges_2[2]{ 0, 30};
//		//numb CT = 1000;
//		//numb TT = 1000;
//		//numb ranges[4]{ -5, 5, -5, 5 };
//		//numb ranges_1D[2]{ -5, 5};
//		//int indeces[2]{ 2, 1 };
//		//int indexParam = 2;
//		//int WriteIndexVariable = 0;
//		//int res = 800;
//
//		//basinsOfAttraction(
//		//	1.0*CT,									// Время моделирования системы
//		//	res,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -100, 100, -100, 100 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1*TT,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	10,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang_2_mu=1_F=20_absXY=50_r1.csv",
//		//	32
//		//);
//		//params[2] = -20;
//		//basinsOfAttraction(
//		//	1.0 * CT,									// Время моделирования системы
//		//	res,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4]{ -100, 100, -100, 100 },
//		//	new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1 * TT,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),			// Количество параметров
//		//	10,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang_2_mu=1_F=-20_absXY=50_r1.csv",
//		//	32
//		//);
//
//	//	//std::string path0 = "D:\\CUDAresults\\bif1Dcont_BoSang_2_";
//	//	//std::string path1 = "D:\\CUDAresults\\bifDFT1D_BoSang_2_";
//	//	//std::string path;
//
//		//for (int i = 0; i < 3; i++) {
//		//	init[0] = 5; init[1] = 5; init[2] = 0.0;
//		//	WriteIndexVariable = i;
//		//	std::string path = "D:\\CUDAresults\\Bif1D_BoSangSin_05_CT=" + std::to_string((int)CT) + "_TT=" + std::to_string((int)TT) + "_varInd=" + std::to_string((int)indexParam)
//		//		+ "_WriteVarInd=" + std::to_string((int)WriteIndexVariable) + ".csv";
//		//	bifurcation1D(
//		//		CT,		//const numb	tMax,							// Время моделирования системы
//		//		2001,		//const int		nPts,						// Разрешение диаграммы
//		//		0.01,		//const numb	h,								// Шаг интегрирования
//		//		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//		init,//const numb * initialConditions,				// Массив с начальными условиями
//		//		new numb[2]{ -20, 20 },//const numb * ranges,							// Диаппазон изменения переменной
//		//		new int[1] { indexParam },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//		WriteIndexVariable,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//		10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//		TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//		params,//const numb * values,							// Параметры
//		//		sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//		1,//const int		preScaller,
//		//		path//std::string		OUT_FILE_PATH
//		//	);
//
//		//	init[0] = -5; init[1] = -5; init[2] = 0;
//		//	path = "D:\\CUDAresults\\Bif1D_BoSangSin_06_CT=" + std::to_string((int)CT) + "_TT=" + std::to_string((int)TT) + "_varInd=" + std::to_string((int)indexParam)
//		//		+ "_WriteVarInd=" + std::to_string((int)WriteIndexVariable) + ".csv";
//		//	bifurcation1D(
//		//		CT,		//const numb	tMax,							// Время моделирования системы
//		//		2001,		//const int		nPts,						// Разрешение диаграммы
//		//		0.01,		//const numb	h,								// Шаг интегрирования
//		//		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//		init,//const numb * initialConditions,				// Массив с начальными условиями
//		//		new numb[2]{ -20, 20 },//const numb * ranges,							// Диаппазон изменения переменной
//		//		new int[1] { indexParam },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//		WriteIndexVariable,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//		10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//		TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//		params,//const numb * values,							// Параметры
//		//		sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//		1,//const int		preScaller,
//		//		path//std::string		OUT_FILE_PATH
//		//	);
//		//}
//		//size_t startTime = std::clock();
//
//		//params[1] = 0.05; params[2] = 9.6;
//		//init[0] = 4.0; init[1] = 0.0; init[2] = 0.0;
//
//		//LLE2D(
//		//	5.0*CT,	//const numb tMax,
//		//	1.0,	//const numb NT,
//		//	201,	//const int nPts,
//		//	0.01,	//const numb h,
//		//	1e-6,	//const numb eps,
//		//	init,	//const numb * initialConditions,
//		//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		//	new numb[4]{ -30, 30, 0, 0.15 },//const numb * ranges,
//		//	new int[2]{ 2, 1 },//const int* indicesOfMutVars,
//		//	1,		//const int writableVar,
//		//	10000000,	//const numb maxValue,
//		//	20.0*CT,	//const numb transientTime,
//		//	params,	//const numb * values,
//		//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_BoSang_2_033.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//for (int i = 0; i < 2; i++) {
//		//	if (i == 0) {
//		//		init[0] = init_1[0];
//		//		init[1] = init_1[1];
//		//		init[2] = init_1[2];
//		//	}
//		//	else if (i == 1) {
//		//		init[0] = init_2[0];
//		//		init[1] = init_2[1];
//		//		init[2] = init_2[2];
//		//	}
//		//	for (int j = 0; j < 1; j++) {
//		//		if (j == 0) {
//		//			ranges_1D[0] = ranges_1[0];
//		//			ranges_1D[1] = ranges_1[1];
//		//		}
//		//		else if (j == 1) {
//		//			ranges_1D[0] = ranges_2[0];
//		//			ranges_1D[1] = ranges_2[1];
//		//		}
//		//		path = path0 + "X0=" + std::to_string((int)init[0]) + "_par_stop=" + std::to_string((int)ranges_1D[1]) + "_par_ind=" + std::to_string((int)indexParam) + ".csv";
//		//		bifurcation1D(
//		//			CT,		//const numb	tMax,							// Время моделирования системы
//		//			2001,		//const int		nPts,						// Разрешение диаграммы
//		//			0.01,		//const numb	h,								// Шаг интегрирования
//		//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//			init,//const numb * initialConditions,				// Массив с начальными условиями
//		//			ranges_1D,//const numb * ranges,							// Диаппазон изменения переменной
//		//			new int[1] { indexParam },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//			10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//			10 * CT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//			params,//const numb * values,							// Параметры
//		//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//			1,//const int		preScaller,
//		//			path//std::string		OUT_FILE_PATH
//		//		);
//		//		//path = path1 + "X0=" + std::to_string((int)init[0]) + "par_stop=" + std::to_string((int)ranges_1D[1]) + ".csv";
//		//		//bifurcation_DFT_1D(
//		//		//	CT * 10,	//const numb	tMax,								// Время моделирования системы
//		//		//	2001,	//const int	nPts,								// Разрешение диаграммы
//		//		//	2001,		//const int	nFreq,								// Разрешение диаграммы
//		//		//	0.01,	//const numb	h,									// Шаг интегрирования
//		//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//		//	init,	//const numb* initialConditions,					// Массив с начальными условиями
//		//		//	ranges_1D,//const numb* ranges,								// Диапазоны изменения параметров
//		//		//	new numb[2]{ 0.000, 1 },//const numb* rangesFreq,								// Диапазоны изменения параметров
//		//		//	new int[1] { 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//		//	1,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//		//	10000000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//		//	CT,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//		//	params,	//const numb* values,								// Параметры
//		//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//		//	1,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//		//	0.15,//const numb	eps,
//		//		//	path//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
//		//		//);
//		//	}
//		//}
//
//		//bifurcation2D(
//		//	2*CT, //const numb	tMax,								// Время моделирования системы
//		//	res, //const int		nPts,								// Разрешение диаграммы
//		//	0.01, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4]{ -30, 30, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	indeces,//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.25,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2D_BoSang_2_003.csv" //std::string		OUT_FILE_PATH
//		//);	
//		//
//
//		//bifurcation2D(
//		//	2 * CT, //const numb	tMax,								// Время моделирования системы
//		//	res, //const int		nPts,								// Разрешение диаграммы
//		//	0.01, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4]{ -30, 30, 0, 15 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	indeces,//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	4, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.25,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2D_BoSang_2_007_01_re1001.csv" //std::string		OUT_FILE_PATH
//		//);
//		//
//		//LLE2D(
//		//	10.0 * CT,	//const numb tMax,
//		//	0.5,	//const numb NT,
//		//	res,	//const int nPts,
//		//	0.01,	//const numb h,
//		//	1e-6,	//const numb eps,
//		//	init,	//const numb * initialConditions,
//		//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		//	new numb[4]{ -30, 30, 0, 15 },//const numb * ranges,
//		//	indeces,//const int* indicesOfMutVars,
//		//	1,		//const int writableVar,
//		//	10000000,	//const numb maxValue,
//		//	TT,	//const numb transientTime,
//		//	params,	//const numb * values,
//		//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_BoSang_2_007_01_re1001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//bifurcation2D(
//		//	2 * CT, //const numb	tMax,								// Время моделирования системы
//		//	res, //const int		nPts,								// Разрешение диаграммы
//		//	0.01, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4]{ -30, 30, 0, 2 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	indeces,//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	1, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	4, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.25,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2D_BoSang_2_007_02_re1001.csv" //std::string		OUT_FILE_PATH
//		//);
//
//		//LLE2D(
//		//	10.0 * CT,	//const numb tMax,
//		//	0.5,	//const numb NT,
//		//	res,	//const int nPts,
//		//	0.01,	//const numb h,
//		//	1e-6,	//const numb eps,
//		//	init,	//const numb * initialConditions,
//		//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		//	new numb[4]{ -30, 30, 0, 2 },//const numb * ranges,
//		//	indeces,//const int* indicesOfMutVars,
//		//	1,		//const int writableVar,
//		//	10000000,	//const numb maxValue,
//		//	TT,	//const numb transientTime,
//		//	params,	//const numb * values,
//		//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_BoSang_2_007_02_re1001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//params[1] = 1.00; params[2] = 20.0;
//		//init[0] = 4.0; init[1] = 0.0; init[2] = 0.0;
//		////numb array_a[5]{16,-16,-16,25,-25};
//		//numb array_a[2]{ 20,-20 };
//		////numb array_XY[5]{ 8,3,8,13,13 };
//		////numb array_XY[5]{ 2, 0.75, 2, 2.6, 2.6 };
//		//numb array_XY[2]{ 23,23 };
//		////params[1] = 4; params[3] = 4;
//		//int res = 101;
//		//std::string path0 = "D:\\CUDAresults\\Basins_BoSang_2_";
//		//std::string path1 = "D:\\CUDAresults\\LLE2DIC_BoSang_2_";
//		//std::string path;
//		//for (int i = 0; i < 2; i++) {
//		//	params[2] = array_a[i]; 
//		//	path = path0 + "mu=" + std::to_string((int)params[1]) + "_F=" + std::to_string((int)params[2]) + 
//		//		"_absXY=" + std::to_string((int)array_XY[i]) + ".csv";
//		//		basinsOfAttraction(
//		//		2.0*CT,									// Время моделирования системы
//		//		res,									// Разрешение диаграммы
//		//		0.01,									// Шаг интегрирования
//		//		sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//		init,									// Массив с начальными условиями
//		//		new numb[4]{ -array_XY[i], array_XY[i], -array_XY[i], array_XY[i] },
//		//		new int[2] { 0, 1 },					// Индексы изменяемых параметров
//		//		1,										// Индекс уравнения, по которому будем строить диаграмму
//		//		100000,									// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//		1*TT,										// Время, которое будет промоделировано перед расчетом диаграммы
//		//		params,									// Параметры
//		//		sizeof(params) / sizeof(numb),			// Количество параметров
//		//		2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//		0.05,									// Эпсилон для алгоритма DBSCAN
//		//		path,
//		//		32
//		//	);
//		//	//path = path1 + "mu=" + std::to_string((int)params[1]) + "_F=" + std::to_string((int)params[2]) +
//		//	//		"_absXY=" + std::to_string((int)array_XY[i]) + ".csv";
//		//	//LLE2D(
//		//	//	CT*20.0,	//const numb tMax,
//		//	//	0.5,	//const numb NT,
//		//	//	res,	//const int nPts,
//		//	//	0.01,	//const numb h,
//		//	//	1e-3,	//const numb eps,
//		//	//	init,	//const numb * initialConditions,
//		//	//	sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		//	//	new numb[4]{ -array_XY[i], array_XY[i], -array_XY[i], array_XY[i] },//const numb * ranges,
//		//	//	new int[2] { 0, 1 },//const int* indicesOfMutVars,
//		//	//	1,		//const int writableVar,
//		//	//	1000000,	//const numb maxValue,
//		//	//	TT,	//const numb transientTime,
//		//	//	params,	//const numb * values,
//		//	//	sizeof(params) / sizeof(numb),//const int amountOfValues,
//		//	//	path//std::string		OUT_FILE_PATH
//		//	//);
//		//} 
//
//		//bifurcation1D(
//		//	1000,		//const numb	tMax,							// Время моделирования системы
//		//	2001,		//const int		nPts,						// Разрешение диаграммы
//		//	0.01,		//const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb * initialConditions,				// Массив с начальными условиями
//		//	new numb[2]{ 1, 5 },//const numb * ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	2,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb * values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,
//		//	"D:\\CUDAresults\\bif1D_BoSang_002.csv"//std::string		OUT_FILE_PATH
//		//);
//		//init[0] = -1; init[1] = -1; init[2] = -1;
//		//bifurcation1D(
//		//	1000,		//const numb	tMax,							// Время моделирования системы
//		//	2001,		//const int		nPts,						// Разрешение диаграммы
//		//	0.01,		//const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb * initialConditions,				// Массив с начальными условиями
//		//	new numb[2]{ 1, 5 },//const numb * ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	2,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	500,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb * values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,
//		//	"D:\\CUDAresults\\bif1D_BoSang_003.csv"//std::string		OUT_FILE_PATH
//		//);
//		//bifurcation_DFT_1D(
//		//	1000,	//const numb	tMax,								// Время моделирования системы
//		//	601,	//const int	nPts,								// Разрешение диаграммы
//		//	1000,		//const int	nFreq,								// Разрешение диаграммы
//		//	0.01,	//const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,	//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[2]{ 1, 5 },//const numb* ranges,								// Диапазоны изменения параметров
//		//	new numb[2]{ 0.001, 10 },//const numb* rangesFreq,								// Диапазоны изменения параметров
//		//	new int[1] { 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	2,		//const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000000,	//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	500,	//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,	//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	5,	//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.15,//const numb	eps,
//		//	"D:\\CUDAresults\\bifDFT1D_BoSang_002.csv"//std::string		OUT_FILE_PATH								// Эпсилон для алгоритма DBSCAN 
//		//);
//
//// Bo Sang 03.03.2026 ---------------------------------------------------------------------------------------------------------------------------------------
////numb params[6]{ 0.5, 1.3, 3, 1.15, 3, -11 };
////	numb params[6]{ 0.5, 1.0, 1, 1.0, 1, -20 };
//	//numb params[6]{ 0.5, 1.3, -1, 1, 0, 11 };
///*
//numb params[6]{ 0.5, 1.0, 0.0, 0.0, 0.0, 14 };
//	numb init[3]{ 5, 0, 0 };
//	numb CT = 600;
//	numb TT = 1000;
//	numb h = 0.005;
//	numb Pstart = -1;
//	numb Pstop = 1;
//
//	//LLE2D(
//	//	CT,												//const numb tMax,
//	//	1.0,											//const numb NT,
//	//	601,											//const int nPts,
//	//	h,												//const numb h,
//	//	1e-6,											//const numb eps,
//	//	init,											//const numb* initialConditions,
//	//	sizeof(init) / sizeof(numb),					//const int amountOfInitialConditions,
//	//	new numb[4] { -2, 2, -2, 2 },				//const numb* ranges,
//	//	new int[2] { 2, 4 },							//const int* indicesOfMutVars,
//	//	0,												//const int writableVar,
//	//	100000,											//const numb maxValue,
//	//	TT,												//const numb transientTime,
//	//	params,											//const numb* values,
//	//	sizeof(params) / sizeof(numb),				//const int amountOfValues,
//	//	"D:\\CUDAresults\\LLE2D_BoSang_F=14_a_vs_b_02.csv"	//std::string		OUT_FILE_PATH
//	//);
//
//	LS2D(
//		CT,	//const numb tMax,
//		1.0,	//const numb NT,
//		201,	//const int nPts,
//		h,	//const numb h,
//		1e-6,	//const numb eps,
//		init,	//const numb * initialConditions,
//		sizeof(init) / sizeof(numb),//const int amountOfInitialConditions,
//		new numb[4]{ -2, 2, -2, 2 },//const numb * ranges,
//		new int[2] { 2, 4},//const int* indicesOfMutVars,
//		0,		//const int writableVar,
//		100000,	//const numb maxValue,
//		TT,	//const numb transientTime,
//		params,	//const numb * values,
//		sizeof(params) / sizeof(numb),//const int amountOfValues,
//		"D:\\CUDAresults\\LS2D_BoSang_F=14_a_vs_b_02.csv"//std::string		OUT_FILE_PATH
//	);
//	*/
//
//	//	bifurcation2D(
////		CT, //const numb	tMax,								// Время моделирования системы
////		601, //const int		nPts,								// Разрешение диаграммы
////		h, //const numb	h,									// Шаг интегрирования
////		sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
////		init,//const numb* initialConditions,					// Массив с начальными условиями
////		new numb[4]{ -2, 2, -2, 2 },//const numb* ranges,								// Диапазоны изменения параметров
////		new int[2] { 2, 4 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
////		0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
////		100000000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
////		TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
////		params,//const numb* values,								// Параметры
////		sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
////		2, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
////		0.1,//const numb	eps,
////		"D:\\CUDAresults\\bif2D_BoSang_F=14_a_vs_b_02.csv" //std::string		OUT_FILE_PATH
////);
//
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ Pstart, Pstop },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_BoSang_03032026_par_F_03.csv"//std::string		OUT_FILE_PATH
//	//);
//	//bifurcation1D(
//	//	CT,		//const numb	tMax,							// Время моделирования системы
//	//	1001,		//const int		nPts,						// Разрешение диаграммы
//	//	h,		//const numb	h,								// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//	//	init,//const numb * initialConditions,				// Массив с начальными условиями
//	//	new numb[2]{ Pstop, Pstart },//const numb * ranges,							// Диаппазон изменения переменной
//	//	new int[1] { 2 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//	//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//	//	10000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,//const numb * values,							// Параметры
//	//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//	//	1,//const int		preScaller,
//	//	"D:\\CUDAresults\\bif1D_BoSang_03032026_par_F_03_back.csv"//std::string		OUT_FILE_PATH
//	//);
//
//	//params[5] = 20; 
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	201,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -50, 50, -50, 50 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	0,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_Nshape_XY_F=20_res200_01.csv",
//	//	blockSize_setup
//	//);
//
//	//params[5] = -20;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	201,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -50, 50, -50, 50 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	0,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_Nshape_XY_F=-20_res200_01.csv",
//	//	blockSize_setup
//	//);
//
//	//params[5] = -11; params[2] = -1; params[3] = 10;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	101,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_03032026_XY_F=-11_res500_04.csv",
//	//	blockSize_setup
//	//);
//	//params[5] = -11; params[2] = -3; params[3] = 0.1;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	501,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_03032026_XY_F=-11_res500_5.csv",
//	//	blockSize_setup
//	//);
//
//	//params[5] = -11; params[2] = -0.1;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	101,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_03032026_XY_F=-11_res500_4.csv",
//	//	blockSize_setup
//	//);
//
//	//params[5] = -11; params[2] = 0;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	101,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_03032026_XY_F=-11_res500_5.csv",
//	//	blockSize_setup
//	//);
//
//	//params[5] = 11;
//	//basinsOfAttraction(
//	//	CT,									// Время моделирования системы
//	//	101,									// Разрешение диаграммы
//	//	h,									// Шаг интегрирования
//	//	AMOUNTOFX,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -10, 10, -10, 10 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSang_03032026_XY_F=11_res100_1.csv",
//	//	blockSize_setup
//	//);
//
//	// Bo sang 18.03.2026
//	 
//	//k[0][j] = (-X1[1] + X1[2]);
//	//k[1][j] = (X1[0]);
//	//k[2][j] = (-X1[2] + a[1] * sin(X1[0]) + a[2] * X1[2] * X1[2] * X1[2]);
//	//numb params[3]{ 0.5, -13, 0.01 };
//	//numb init[3]{ 0, 0, 0 };
//	//params[1] = 13;
//	//basinsOfAttraction(
//	//	700,									// Время моделирования системы
//	//	501,									// Разрешение диаграммы
//	//	0.01,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -20, 20, -12, 12 },
//	//	new int[2] { 1, 2 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	500,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	5,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.2,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSangCubic_yz_001.csv",
//	//	32
//	//);
//	//params[1] = -13;
//	//basinsOfAttraction(
//	//	700,									// Время моделирования системы
//	//	501,									// Разрешение диаграммы
//	//	0.01,									// Шаг интегрирования
//	//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4] { -20, 20, -12, 12},
//	//	new int[2] { 1, 2 },						// Индексы изменяемых параметров
//	//	1,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	500,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	5,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.2,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_BoSangCubic_yz_002.csv",
//	//	32
//	//);
//
//		// Lorenz  fast synchro ------------------------------------------------------------------------------------------------------------------------------------------------
//		
//		//numb params[4]{ 0.5, 10, 28, 8.0/3.0 };
//		//numb init[3]{ 3, -3, 0 };
//		//int iter;
//		//std::string path0 = "D:\\CUDAresults\\FS_Lorenz_2_iter=";
//		//std::string path;
//		////for (int i = 0; i < 1000; i++) {
//		////	iter = 1 + i * 1;
//		////	path = path0 + std::to_string(iter) + ".csv";
//		////	FastSynchro(
//		////		100,										//const numb	tMax,								// Время моделирования системы
//		////		1000,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		////		0.1,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		////		params,										//const numb* values,								// Параметры
//		////		sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		////		0.001,										//const numb	h,									// Шаг интегрирования
//		////		new numb[3]{ 0, 40, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		////		new numb[3]{ 0, 40, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		////		init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		////		new numb[3]{ 0.0, 0.0, 0.0 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		////		sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		////		1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		////		iter,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		////		10,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		////		path										//std::string		OUT_FILE_PATH
//		////	);
//		////}
//		//FastSynchro(
//		//	500,										//const numb	tMax,								// Время моделирования системы
//		//	1000,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	0.15,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,										//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		//	0.005,										//const numb	h,									// Шаг интегрирования
//		//	new numb[3]{ 0, 40, 0 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[3]{ 0, 40, 0 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	init,				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[3]{ 1e-6, 1e-6, 0 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	200,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_Lorenz_Y.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		// Chen adaptive fast synchro ------------------------------------------------------------------------------------------------------------------------------------------------
//		
//		//numb params[4]{ 0.5, 35, 3, 28 };
//		//numb init[6]{ 0.2, 0.2, 18, 35, 3 ,28 };
//		//FastSynchro(
//		//	1500,										//const numb	tMax,								// Время моделирования системы
//		//	100,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	0.15,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,										//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		//	0.0025,										//const numb	h,									// Шаг интегрирования
//		//	new numb[6] { 42, 42, 42, 0, 0, 0},						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[6] { 42, 42, 42, 0, 0, 0},						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[6] { 0.2, 0.2, 18, 35, 3 ,28 },				//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[6] { 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	10,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_adaptiveChen.csv"//std::string		OUT_FILE_PATH
//		//);
//		//numb params[1]{ 0.5 };
//		//numb init[3]{ 1.5, 0, 0 };
//		//FastSynchro(
//		//	1000,										//const numb	tMax,								// Время моделирования системы
//		//	8000,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	2.0,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,										//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),				//const int		amountOfValues,						// Количество параметров
//		//	0.01,										//const numb	h,									// Шаг интегрирования
//		//	new numb[3]{ 0, 0, 2 },						//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[3]{ 0, 0, 2 },						//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	init,										//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[3]{ 1e-12, 0, 0 },				//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),				//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,									//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	20,										//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	5,											//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS2_Sprott14.csv"//std::string		OUT_FILE_PATH
//		//);
//		//params[1] = 0.248;
//		//params[2] = 0.022;
//		//basinsOfAttraction(
//		//	250,									// Время моделирования системы
//		//	501,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { -23, 32, -27, 20},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	1,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	1500,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.05,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_ROSSLER1.csv"
//		//);
//
//		//Log basins for artur/burkin Matreshka
//
//		//numb params[8]{ 0.0, -19, -3.5, -3.2, 3.5, 1, 0.3, 1 };
//		//numb init[3]{ 0.0, 0.0, 0.0 };
//
//
//		//basinsOfAttraction_logAxes(
//		//	600,									// Время моделирования системы
//		//	200,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { 100, 0.1, 100, 0.1},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	600,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\BasinsLog_MatrArturBurk_lam_3.5_0003.csv"
//		//);
//
//		//basinsOfAttraction_2(
//		//	600,									// Время моделирования системы
//		//	500,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { -100, 100, -100, 100},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	600,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_MatrArturBurk_lam_3.5_0002.csv"
//		//);
//	//
//	//numb params[5]{ 0.5, 15, 5, 25, 0.8 };
//	//numb init[4]{ 1,1,1,1 };
//	//
//	//basinsOfAttraction(
//	//	200,									// Время моделирования системы
//	//	301,									// Разрешение диаграммы
//	//	0.005,									// Шаг интегрирования
//	//	4,			// Количество начальных условий ( уравнений в системе )
//	//	init,									// Массив с начальными условиями
//	//	new numb[4]{ -5, 5, -5, 5 },
//	//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//	//	0,										// Индекс уравнения, по которому будем строить диаграмму
//	//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//	//	300,									// Время, которое будет промоделировано перед расчетом диаграммы
//	//	params,									// Параметры
//	//	sizeof(params) / sizeof(numb),		// Количество параметров
//	//	2,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//	//	0.15,									// Эпсилон для алгоритма DBSCAN
//	//	"D:\\CUDAresults\\Basins_NovelSciRep.csv",
//	//	32
//	//);
//
//
//		//Log basins for artur Matreshka
//
//		//numb params[10]{ 0.5, 8.03911207626026, -19, -3.5, -3.2, 5, 1.4, 50, 15,1 };
//		//numb lambda = 2;
//		// 
//		//numb params[7]{ 0.0, 9, 12, 0, 5, 1, 0.21};
//		//numb params[8]{ 0.0, 8.7, 11.1, 0, 4, 1.4, 0.15, 1.3};
//		//numb init[3]{ 0.0, 0.0, 0.0 };
//
//		//basinsOfAttraction_logAxes(
//		//	300,									// Время моделирования системы
//		//	100,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { 100, 0.1, 100, 0.1},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	10000,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.05,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\BasinsLog_MatrArtur_lam_5.0_0001.csv"
//		//);
//
//		//basinsOfAttraction_2(
//		//	300,									// Время моделирования системы
//		//	100,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { -100, 100, -100, 100},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	10000,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.1,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_MatrArtur_lam_5.0_0001.csv"
//		//);
//
//		// Neuron Izhikevich (4-parameter)
//		//						  1	   2	3	 4	   5	6	   7	 8		9	 10	  11	
//		//numb params[12]{ 0.5,	0.1, 0.2, -65, 2.0, 0.04, 5.0, 140.0, 30.0, 0.002, 10.0, 0.0  };
//		//numb init[3]{ 0.0, 0.0, 0.0 };
//		//numb CT = 1500;
//		//numb TT = 500;
//		//numb h = 0.001;
//
//		//bifurcation1D_MeanAndMedianFreq(
//		//	CT,	//const numb	tMax,							// Время моделирования системы
//		//	1000,	//const int		nPts,						// Разрешение диаграммы
//		//	h, //const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init, //const numb* initialConditions,				// Массив с начальными условиями
//		//	new numb[2] { 0.0, 20.0},//const numb* ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 10 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	1000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\bif1Dmedmeanfreq_NeuronIzhi_001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		// Neuron Wilson
//		//						1		2	3		4		5		6	7	8		9	10		11	12		13	
//		//numb params[14]{ 0.5,	0.8, 1.9, 17.81, 47.71, 32.63, -0.55, -26.0, 0.92, 1.35, 1.03, 0.264, 0.007, 0.075 };
//		//						C	 tau	0	  1		2		3	 5     4		6	7	   Ifreq Iamp  Idc
//		/*numb params[15]{0.5,	0.8, 1.9, 0.400, 2.17, 32.63, 1.25, 26.0, -0.22, 1.35, 0.0874, 0.5, 0.045, 0.00, 0.5};
//
//		numb init[3]{ 0.0, 0.0, 0.0 };
//		numb CT = 5;
//		numb TT = 0;
//		numb h = 0.01;
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.01,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			1,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.01.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.005,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			2,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.005.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.0025,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			4,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.0025.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.00125,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			8,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.00125.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.001,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			10,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.001.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.0005,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			20,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.0005.csv"//std::string		OUT_FILE_PATH
//		);
//
//		TimeDomainCalculation(
//			CT,	//const numb	tMax,							// Время моделирования системы
//			1,		//const int		nPts,							// Разрешение диаграммы
//			0.0001,	//const numb	h,								// Шаг интегрирования
//			sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//			init,//const numb* initialConditions,				// Массив с начальными условиями
//			new numb[2]{ 0.0, 00.0 },//const numb* ranges,							// Диаппазон изменения переменной
//			new int[1] { 0 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//			0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//			10000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//			0,//const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//			params,//const numb* values,							// Параметры
//			sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//			100,//const int		preScaller,
//			"D:\\CUDAresults\\TD_wilson_h=0.0001.csv"//std::string		OUT_FILE_PATH
//		);
//		*/
//
//
//		//	bifurcation1D_MeanAndMedianFreq(
//		//	CT,	//const numb	tMax,							// Время моделирования системы
//		//	5000,	//const int		nPts,						// Разрешение диаграммы
//		//	h, //const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init, //const numb* initialConditions,				// Массив с начальными условиями
//		//	new numb[2] { 0.0, 1.0},//const numb* ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 11 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	1000000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\bif1Dmedmeanfreq_NeuronWilson_001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//numb params[4]{ 10, 0.096, 1, 0 };
//		//numb init[3]{ 0.2, 0.2, 0.0 };
//		//numb CT = 5000;
//		//numb TT = 0;
//		//numb h = 0.1;
//
//		//	bifurcation1D_MeanAndMedianFreq(
//		//	CT,	//const numb	tMax,							// Время моделирования системы
//		//	1000,	//const int		nPts,						// Разрешение диаграммы
//		//	h, //const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init, //const numb* initialConditions,				// Массив с начальными условиями
//		//	new numb[2] { 0.08, 0.11},//const numb* ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	0,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\bif1Dmedmeanfreq_FHN_001.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//numb params[4]{ 0.5, 9, 2, -3.5 };
//		//numb init[3]{ 1.1, 1.5, 1.8 };
//		//numb CT = 50;
//		//numb TT = 250;
//		//numb h = 0.001;
//
//		//bifurcation2D(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	200, //const int		nPts,								// Разрешение диаграммы
//		//	h, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4] { 2, 6, -3.5, -1},//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 2, 3 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	2, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.05,//const numb	eps,
//		//	"D:\\CUDAresults\\2DBif_Jiri_2_3_001.csv" //std::string		OUT_FILE_PATH
//		//);
//
//		//LLE2D(
//		//	CT,												//const numb tMax,
//		//	0.5,											//const numb NT,
//		//	200,											//const int nPts,
//		//	h,												//const numb h,
//		//	1e-6,											//const numb eps,
//		//	init,											//const numb* initialConditions,
//		//	sizeof(init) / sizeof(numb),					//const int amountOfInitialConditions,
//		//	new numb[4] { 2, 6, -3.5, -1},				//const numb* ranges,
//		//	new int[2] { 2, 3 },							//const int* indicesOfMutVars,
//		//	0,												//const int writableVar,
//		//	100000,											//const numb maxValue,
//		//	TT,												//const numb transientTime,
//		//	params,											//const numb* values,
//		//	sizeof(params) / sizeof(numb),				//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_Jiri_2_3_001.csv"	//std::string		OUT_FILE_PATH
//		//);
//
//
//		//numb params[4]{ 0.5, 0.2, 0.2, 5.7 };
//		//numb init[3]{ 0.1, 0.1, 0.1 };
//		//numb CT = 1000;
//		//numb TT = 1000;
//		//numb h = 0.01;
//
//		//bifurcation1D_MeanAndMedianFreq(
//		//	50*CT,	//const numb	tMax,							// Время моделирования системы
//		//	1200,	//const int		nPts,						// Разрешение диаграммы
//		//	h, //const numb	h,								// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,		// Количество начальных условий ( уравнений в системе )
//		//	init, //const numb* initialConditions,				// Массив с начальными условиями
//		//	new numb[2] { 0.15, 0.35},//const numb* ranges,							// Диаппазон изменения переменной
//		//	new int[1] { 1 },//const int* indicesOfMutVars,				// Индекс изменяемой переменной в массиве values
//		//	2,//const int		writableVar,					// Индекс уравнения, по которому будем строить диаграмму
//		//	100000,//const numb	maxValue,						// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,					// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,							// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,					// Количество параметров
//		//	1,//const int		preScaller,						// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\bif1Dmedmeanfreq_Rossler_002.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//bifurcation2D_MeanAndMedianFreq(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	600, //const int		nPts,								// Разрешение диаграммы
//		//	h, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4] { 0.15, 0.35, 0.05, 0.35},//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 1, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	2, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.2,//const numb	eps,
//		//	"D:\\CUDAresults\\bif2Dmedmeanfreq_Rossler_002.csv" //std::string		OUT_FILE_PATH
//		//);
//
//		 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//		//numb params[7]{ 0.5, -0.1, -1, 1.4, 1, 1, 1 };
//		//numb params[7]{ 0.5, -0.1, -1, 0.95, 1, 1, 1 };
//		//numb params[7]{ 0.5, 0.0, -1, 0.95, 1, 1, 1 };
//		//numb init[3]{ 0, 0, 0 };
//		//numb CT = 700;
//		//numb TT = 300;
//		//numb h = 0.05;
//
//		//LLE2D(
//		//	CT,												//const numb tMax,
//		//	0.5,											//const numb NT,
//		//	200,											//const int nPts,
//		//	h,												//const numb h,
//		//	1e-6,											//const numb eps,
//		//	init,											//const numb* initialConditions,
//		//	sizeof(init) / sizeof(numb),					//const int amountOfInitialConditions,
//		//	new numb[4] {  -0.3, 0, 0.5, 1},				//const numb* ranges,
//		//	new int[2] { 1, 3 },							//const int* indicesOfMutVars,
//		//	0,												//const int writableVar,
//		//	100000,											//const numb maxValue,
//		//	TT,												//const numb transientTime,
//		//	params,											//const numb* values,
//		//	sizeof(params) / sizeof(numb),				//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2D_BoSang23092025_001.csv"	//std::string		OUT_FILE_PATH
//		//);
//
//		//LLE2DIC(
//		//	1*CT,												//const numb tMax,
//		//	1.0,											//const numb NT,
//		//	500,											//const int nPts,
//		//	h,												//const numb h,
//		//	1e-6,											//const numb eps,
//		//	init,											//const numb* initialConditions,
//		//	sizeof(init) / sizeof(numb),					//const int amountOfInitialConditions,
//		//	//new numb[4] { -50, 90, -50, 120},				//const numb* ranges,
//		//	new numb[4] { -10, 10, -10, 10},				//const numb* ranges,
//		//	new int[2] { 1, 2 },							//const int* indicesOfMutVars,
//		//	0,												//const int writableVar,
//		//	100000,											//const numb maxValue,
//		//	TT,												//const numb transientTime,
//		//	params,											//const numb* values,
//		//	sizeof(params) / sizeof(numb),				//const int amountOfValues,
//		//	"D:\\CUDAresults\\LLE2DIC_BoSang23092025_008.csv"	//std::string		OUT_FILE_PATH
//		//);
//
//		//params[1] = 1;
//		//LS2DIC(
//		//	1 * CT,												//const numb tMax,
//		//	1.0,											//const numb NT,
//		//	500,											//const int nPts,
//		//	h,												//const numb h,
//		//	1e-6,											//const numb eps,
//		//	init,											//const numb* initialConditions,
//		//	sizeof(init) / sizeof(numb),					//const int amountOfInitialConditions,
//		//	//new numb[4] { -50, 90, -50, 120},				//const numb* ranges,
//		//	new numb[4] { -20, 20, -20, 20},				//const numb* ranges,
//		//	new int[2] { 1, 2 },							//const int* indicesOfMutVars,
//		//	0,												//const int writableVar,
//		//	100000,											//const numb maxValue,
//		//	TT,												//const numb transientTime,
//		//	params,											//const numb* values,
//		//	sizeof(params) / sizeof(numb),				//const int amountOfValues,
//		//	"D:\\CUDAresults\\LS2DIC_BoSang23092025_009.csv"	//std::string		OUT_FILE_PATH
//		//);
//		//	
//		//basinsOfAttraction_2(
//		//	 CT,									// Время моделирования системы
//		//	 100,									// Разрешение диаграммы
//		//	 h,									// Шаг интегрирования
//		//	 sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	 init,									// Массив с начальными условиями
//		//	 new numb[4] { -20, 20, -20, 20},
//		//	 new int[2] { 1, 2 },						// Индексы изменяемых параметров
//		//	 0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	 100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	 TT,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	 params,									// Параметры
//		//	 sizeof(params) / sizeof(numb),		// Количество параметров
//		//	 1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	 0.01,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_BoSang23092025_0091.csv"
//		//);	
//
//
//
//		//bifurcation2DIC(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	400, //const int		nPts,								// Разрешение диаграммы
//		//	h, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4] { -50, 90, -50, 120},//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 1, 2 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.01,//const numb	eps,
//		//	"D:\\CUDAresults\\2DBifIC_BoSang23092025_1_2_001.csv" //std::string		OUT_FILE_PATH
//		//);
//
//
//
//
//		//bifurcation2D(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	100, //const int		nPts,								// Разрешение диаграммы
//		//	h, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4] { -0.3, 0, 0.5, 1},//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 1, 3 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	2, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	100000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.5,//const numb	eps,
//		//	"D:\\CUDAresults\\2DBif_BoSang23092025_1_3_001.csv" //std::string		OUT_FILE_PATH
//		//);
//
//
//		//numb params[4]{ 0.2, 0.2, 5.7, 0.5 };
//		//numb init[3]{ 0.1, 0.1, 0.1 };
//		//numb CT = 500;
//		//numb TT = 1000;
//		//numb h = 0.01;
//
//		//bifurcation2D(
//		//	CT, //const numb	tMax,								// Время моделирования системы
//		//	500, //const int		nPts,								// Разрешение диаграммы
//		//	h, //const numb	h,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	init,//const numb* initialConditions,					// Массив с начальными условиями
//		//	new numb[4] { 0.05, 0.35, 0.05, 1},//const numb* ranges,								// Диапазоны изменения параметров
//		//	new int[2] { 0, 1 },//const int* indicesOfMutVars,					// Индексы изменяемых параметров
//		//	0, //const int		writableVar,						// Индекс уравнения, по которому будем строить диаграмму
//		//	10000, //const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	TT, //const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),//const int		amountOfValues,						// Количество параметров
//		//	1, //const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.05,//const numb	eps,
//		//	"D:\\CUDAresults\\2DBif_RosslerMNDS_02.csv" //std::string		OUT_FILE_PATH
//		//);
//
//		// //Chen with variable symmetry
//		// numb params[4]{ 0.77, 40, 3, 28 };
//		// numb init[3]{ 0, 0, 18 };
//
//		//basinsOfAttraction_2(
//		//	 100,									// Время моделирования системы
//		//	 800,									// Разрешение диаграммы
//		//	 0.01,									// Шаг интегрирования
//		//	 sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	 init,									// Массив с начальными условиями
//		//	 new numb[4] { -16, 16, -16, 16},
//		//	 new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	 0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	 100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	 200,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	 params,									// Параметры
//		//	 sizeof(params) / sizeof(numb),		// Количество параметров
//		//	 1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	 0.1,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_Chen_0001.csv"
//		//);
//
//		//Sine Chaotic map
//		//numb params[3]{ 0.5, 3, 3 };
//		//numb init[3]{ 0, 0, 0 };
//
//		//basinsOfAttraction_2(
//		//	300,									// Время моделирования системы
//		//	800,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { -5, 5, -6, 6},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	750,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.1,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_3DsineOscil_0002.csv"
//		//);
//
//		//
//		//numb params[8]{ 0.5, 7, 6, 1, 1.4, 0.6, -0.2, -2 };
//		//numb init[4]{ 0, 0, 0, 0 };
//
//		//basinsOfAttraction_2(
//		//	500,									// Время моделирования системы
//		//	500,									// Разрешение диаграммы
//		//	0.01,									// Шаг интегрирования
//		//	sizeof(init) / sizeof(numb),			// Количество начальных условий ( уравнений в системе )
//		//	init,									// Массив с начальными условиями
//		//	new numb[4] { -5, 5, -5, 5},
//		//	new int[2] { 0, 1 },						// Индексы изменяемых параметров
//		//	0,										// Индекс уравнения, по которому будем строить диаграмму
//		//	100000000,								// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся"
//		//	750,									// Время, которое будет промоделировано перед расчетом диаграммы
//		//	params,									// Параметры
//		//	sizeof(params) / sizeof(numb),		// Количество параметров
//		//	1,										// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	0.1,									// Эпсилон для алгоритма DBSCAN
//		//	"D:\\CUDAresults\\Basins_OrlovskiiMemMultistable_0002.csv"
//		//);
//
//		// ---------------- CANG 4D -----------------
//
//		//numb CT = 50;
//		//numb TT = 110;
//		//numb NT = 0.05;
//		//numb h = 0.001;
//		//numb params[8] = { 0.5, 50, -16, 10, 0.2,	10,	16,	0.5 };
//		//numb init[4] = { 0.1,	0.1, 0.1, 0.1 };
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 0, 0, 100},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 0, 0, 100},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_W_100_100_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 10, 0, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 10, 0, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_X_10_10_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 10, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 10, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_Y_10_10_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 100, 0, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 0, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_X_100_0_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 100, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 0, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_Y_100_0_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 0, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 100, 0, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_X_0_100_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 0, 0, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 100, 0, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_Y_0_100_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//		//FastSynchro(
//		//	CT,										//const numb	tMax,								// Время моделирования системы
//		//	TT,										//const numb	transientTime,						// Время, которое будет промоделировано перед расчетом диаграммы
//		//	NT,										//const numb	NTime,								// Длина отрезка по которому будет проводиться синхронизация
//		//	params,									//const numb* values,								// Параметры
//		//	sizeof(params) / sizeof(numb),		//const int		amountOfValues,						// Количество параметров
//		//	h,										//const numb	h,									// Шаг интегрирования
//		//	new numb[4] { 0, 0, 100, 0},			//const numb* kForward,							// Массив коэффициентов синхронизации вперед
//		//	new numb[4] { 0, 0, 100, 0},			//const numb* kBackward,							// Массив коэффициентов синхронизации назад
//		//	new numb[4] { 0.1, 0.1, 0.1, 0.1},	//const numb* initialConditionsMaster,			// Массив с начальными условиями мастера
//		//	new numb[4] { 0.2, 0.2, 0.2, 0.2},	//const numb* initialConditionsSlave,				// Массив с начальными условиями слейва
//		//	sizeof(init) / sizeof(numb),			//const int		amountOfInitialConditions,			// Количество начальных условий ( уравнений в системе )
//		//	1000000,								//const numb	maxValue,							// Максимальное значение (по модулю), выше которого система считаемся "расшедшейся
//		//	50000,									//const int		iterOfSynchr,						// Число итераций синхронизации
//		//	1,										//const int		preScaller,							// Множитель, который уменьшает время и объем расчетов (будет рассчитываться только каждая 'preScaller' точка)
//		//	"D:\\CUDAresults\\FS_CANG4D_Z_100_100_iter50000_3.csv"//std::string		OUT_FILE_PATH
//		//);
//
//
	printf("Time of runnig: %zu ms\n", std::clock() - startTime);
	getch();
	return 0;
}
