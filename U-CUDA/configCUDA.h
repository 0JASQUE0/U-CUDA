#pragma once
#include <math_constants.h>

typedef double numb;

// AMOUNTOFX оборачивается в #ifndef, чтобы NVRTC-вызывающие проекты могли
// переопределить размерность системы через #define AMOUNTOFX N перед include.
// Обычная nvcc-сборка ничего не определяет — поведение прежнее.
#ifndef AMOUNTOFX
constexpr int AMOUNTOFX = 3;
#endif
constexpr int CHECK_INTERVAL = 100;

// --- Calculate peaks ---
// 1 -- Yes; 
// 0 -- No
constexpr bool doCalculatePeaks = 1;

// --- data[startDataIndex + i] = xDataMultiplier * (x[writableVar]); ---
constexpr numb xDataMultiplier = 1.0;

// --- Parabolic interpolation of Peaks and InterPeaks
// 1 -- Yes
// 0 -- No
constexpr bool doInterpolatePeaks = 1;

// --- CHOOSE CLASSICAL OR CONTINUATION BIFURCATION DIAGRAM ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation and DFT diagram)
constexpr bool continuation_bif1D = 0;

// --- CHOOSE PARAMETER OR VARIABLE ANALYSIS ---
// 2 -- variable and parameter analysis (first variable, second parameter)
// 1 -- parameter analysis;
// 0 -- variable analysis
// Обёрнуто в #ifndef, чтобы NVRTC-template мог переопределить через #define.
#ifndef par_or_var
constexpr int par_or_var = 1;
#endif

// --- CHOOSE LIN OR LOG variable/parameter distribution ---
// 1 -- lin; [Xmin,Xmax] or [Xmin,Xmax,Ymin,Ymax]
// 0 -- log; [log10(Xmin)
constexpr bool LINEAR_OR_LOG_DISTRIB = 1;


// --- CHOOSE AVG OR LOG10(AVG) ANALYSIS ---
// 1 -- avg peak analysis; 
// 0 -- log10(avg peak) analysis
constexpr bool lin_or_log = 1; 

// --- CALCULATE ADDITIONALLY MEAN AND MEDIAN FREQUENCY? ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation diagram)
constexpr bool calculate_mean_med_freq = 0; 

// --- CALCULATE ADDITIONALLY MEAN AND VARIANCE OF CHOOSEN VARIABLE? ---
// 1 -- yes; 
// 0 -- no (only for 1D bifurcation diagram)
constexpr bool calculate_mean_and_variance = 0; 

// --- CALCULATE GLOBAL PEAK VALUE? ---
// 1 -- yes; 
// 0 -- no (only for 2D bifurcation diagram)
constexpr bool calculate_global_peak = 0;

// --- THRESHOLDs FOR ESTIMATION ---
constexpr numb eps_fixed_point = 1e-8;		// Minimal diffrence between two last points to mark it fixed point regime
constexpr numb eps_peak_delta = 1e-14;		// minimal differnece between two neibor points to mark it peak
constexpr numb eps_interPeak_delta = 0.0;	// minimal interspike interval between two peak points
constexpr numb peak_threshold = -1e25;	// threshold for peakfinder 
constexpr int max_amount_of_peaks = 2500;	// max amount of peaks for estimation for peakfinder 

// --- MULTIPLIERS FOR DBSCAN (2D bifurcation diagrams) ---
constexpr numb mult_peak  = 1.0; // multiplier for peak values for DBSCAN //1000
constexpr numb mult_interval = 0.0; // multiplier for interval values for DBSCAN //4

// --- MULTIPLIERS FOR DBSCAN (Basins of attraction) ---
constexpr numb mult_avg_peak = 1.0; // multiplier for average peak values for DBSCAN (basins of attraction)
constexpr numb mult_avg_interval = 1.0; // multiplier for average interval values for DBSCAN (basins of attraction)

constexpr int blockSize_setup = 32; // default blockSize value
constexpr int set_precision  = 15; // precision of numbers in writng final csv files

// ---  ---  ---  ---  --- Fast Synchro ---  ---  ---  ---  --- 

constexpr numb FS_error_trs = 1e-12;

// 0 - unidirictional sycnhro; 
// 1 - bidirectional synchro
constexpr int type_of_synch = 0; 

// 0 - RMS(error) on the last iteration; 
// 1 - number of iteration to achieve RMS(error) <= FS_error_trs; 
// 2 - RMS(error) on the last point on the last iteration;
constexpr int error_estim = 2; 

constexpr int amount_GPU = 1920; // precision of numbers in writng final csv files
constexpr numb pi	 = 3.1415926535897932384626433832795;
constexpr numb euler = 2.7182818284590452353602874713527;
//constexpr int blockSize_fixed = 1024;