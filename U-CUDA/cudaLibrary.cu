#include "cudaLibrary.cuh"
#include "configCUDA.h"
#include <math.h>
#include <cmath>

// Host-only и runtime-API заголовки недоступны при NVRTC-компиляции.
#ifndef __CUDACC_RTC__
#include <curand_kernel.h>
#include <iostream>
#include <cuda_runtime.h>
#endif



// ---------------------------------------------------------------------------------
// --- Вычисляет следующее значение дискретной модели и записывает результат в x ---
// ---------------------------------------------------------------------------------

// calculateDiscreteModel_rand зависит от curand_kernel.h, которого нет в NVRTC.
// Под NVRTC эта функция не нужна (kernel-ы для bif1d/LLE/basins её не зовут).
#ifndef __CUDACC_RTC__
__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h)
{
	curandState_t state;
	//curand_init(seed, 0, 0, &state);

	//numb h1 = h / 2;

	//X[0] = X[0] - h1 * (X[1] + X[2]);
	//X[1] = X[1] + h1 * (X[0] + X[1] * a[0]);
	//X[2] = X[2] + h1 * (a[1] + X[2] * (X[0] - a[2]));

	//X[2] = (X[2] + h1 * a[1]) / (1 - h1 * (X[0] - a[2]));
	//X[1] = (X[1] + h1 * X[0]) / (1 - h1 * a[0]);
	//X[0] = X[0] - h1 * (X[1] + X[2]);

	//X[0] = X[0] + 0.5*(curand_uniform(&state) - 0.5);


	///////////////////////////////////////////////////////////////////////////////////////////////////////
	numb X1[3], k[4][4], Im, Id, Iin;
	numb pi = 3.14159265359;
	numb u1, u2;
	int N = 3;
	int i, j;


	for (i = 0; i < N; i++) {
		X1[i] = X[i];
	}

	for (j = 0; j < 4; j++) {


		//Id = a[4] * (exp((X[0] + a[6]) / a[9]) - exp(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * exp(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atan(a[17] * (X[0] + a[6] - a[18])) + atan(a[17] * (X[0] + a[6] + a[18])));

		//if ((X[0] + a[7]) > 0)
		//	Im = (X[0] + a[7]) * X[1] / a[19] + a[5];
		//else
		//	Im = (X[0] + a[7]) * X[1] / a[20] - a[5];

		////Iin   = a[25];
		//Iin = a[25] * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
		////Iin = a[25] *  (fmod(X[2] + a[28], a[26])  )/a[26];		

		//k[0][j] = (Iin - Im - Id) / a[8];
		//k[1][j] = (1 / a[21]) * (1 / (1 + exp(-1 / (a[15] * a[15]) * ((X[0] + a[7]) - X[3]) * ((X[0] + a[7]) - a[13])))) * ((1 - 1 / (exp((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + exp(-1 / (a[16] * a[16]) * ((X[0] + a[7]) - a[14]) * ((X[0] + a[7]) - X[4])))) * ((1 - 1 / (exp((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (exp(a[1] * (1 - X[1]) + a[24]))));
		
		////////////////////////////////////////////////////////////////////////////////////

		Id = a[4] * (expf((X[0] + a[6]) / a[9]) - expf(-(X[0] + a[6]) / a[9])) + (a[2] / a[10]) * (X[0] + a[6]) * expf(-(X[0] + a[6] - a[10]) / a[10]) + a[3] * (atanf(a[17] * (X[0] + a[6] - a[18])) + atanf(a[17] * (X[0] + a[6] + a[18])));

		if ((X[0] + a[7]) > 0)
			Im = (X[0] + a[7]) * X[1] / a[19] + a[5];
		else
			Im = (X[0] + a[7]) * X[1] / a[20] - a[5];

		//Iin   = a[25];
		Iin = a[25] * ((fmod(X[2] + a[28], a[26]) < a[27]) ? 1 : 0);
		//Iin = a[25] *  (fmod(X[2] + a[28], a[26])  )/a[26];		

		k[0][j] = (Iin - Im - Id) / a[8];
		k[1][j] = (1 / a[21]) * (1 / (1 + expf(-1 / (a[15] * a[15]) * ((X[0] + a[7]) - X[3]) * ((X[0] + a[7]) - a[13])))) * ((1 - 1 / (expf((a[1] * X[1] + a[23])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]))))) - (1 / a[22]) * (1 - 1 / (1 + expf(-1 / (a[16] * a[16]) * ((X[0] + a[7]) - a[14]) * ((X[0] + a[7]) - X[4])))) * ((1 - 1 / (expf((a[1] * X[1])))) * (1 - X[1]) + X[1] * (1 - 1 / (expf(a[1] * (1 - X[1]) + a[24]))));



		if ((k[1][j] < 0) && (X[5] == 0)) {
			curand_init(seed, 0, 0, &state);
			u1 = curand_uniform(&state);
			curand_init(seed+50, 0, 0, &state);
			u2 = curand_uniform(&state);
			X[3] = sqrt(-2 * log(u1)) * cos(2 * pi * u2) * sqrt(0.0005) + a[11];
			X[5] = 1;
			X[6] = 0;
		}
		else if ((k[1][j] > 0) && (X[6] == 0)) {
			curand_init(seed, 0, 0, &state);
			u1 = curand_uniform(&state);
			curand_init(seed+25, 0, 0, &state);
			u2 = curand_uniform(&state);
			X[4] = sqrt(-2 * log(u1)) * sin(2 * pi * u2) * sqrt(0.0004) + a[12];
			X[5] = 0;
			X[6] = 1;
		}
		k[2][j] = 1;


		if (j == 3) {
			for (i = 0; i < N; i++) {
				X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;
			}
		}
		else if (j == 2) {
			for (i = 0; i < N; i++) {
				X1[i] = X[i] + h * k[i][j];
			}
		}
		else {
			for (i = 0; i < N; i++) {
				X1[i] = X[i] + 0.5 * h * k[i][j];
			}
		}
	}
	X[7] = Iin;
}

__device__ __host__ __forceinline__ void calculateDiscreteModel(numb* X, const numb* a, const numb h)
{
	double X1[3];
	X1[0] = X[0] + 0.5 * h * (a[1] * (X[1] - X[0]));
	X1[1] = X[1] + 0.5 * h * (X[0] * (a[2] - X[2]) - X[1]);
	X1[2] = X[2] + 0.5 * h * (X[0] * X[1] - a[3] * X[2]);
	X[0] = X[0] + h * (a[1] * (X1[1] - X1[0]));
	X[1] = X[1] + h * (X1[0] * (a[2] - X1[2]) - X1[1]);
	X[2] = X[2] + h * (X1[0] * X1[1] - a[3] * X1[2]);
}
#endif // __CUDACC_RTC__

__device__ __host__ numb sign(numb x) {
	if (x > 0)
		return 1.0;
	else if (x < 0)
		return -1.0;
	else
		return 0;
}

//__device__ __host__ numb chua_multistep_extended(numb x, numb m, numb d, numb kslope, numb dmargin) {
//	
//	
//
//	if (d < 1e-14) {
//		return -x;;
//	}
//	numb y = 0;
//	numb sg = 1;
//	if (x < 0) {
//		x = -x;
//		sg = -1;
//	}
//
//	if (x < (m - 2) / (m - 1) / kslope * d) {
//		y = -kslope * x;
//	}
//	else {
//		if (x < (m - 2) / (m - 1) * d) {
//			y = -(m - 2) / (m - 1) * d;
//		}
//		else {
//			if (x > (m / (m - 1) * d)) {
//				if (x < dmargin * d) {
//					y = -m / (m - 1) * d;
//				}
//				else {
//					y = -kslope * (x - dmargin * d) - m / (m - 1) * d;
//				}
//			}
//			else {
//				numb xnew = (x - d);
//				y = chua_multistep_extended(xnew, m, d / m, kslope, dmargin) - d;
//			}
//		}
//
//	}
//	return sg * y;
//}

__device__ __host__ numb chua_multistep_extended(numb x, numb m, numb d, numb kslope, numb dmargin) {
	// Защита от некорректных параметров
	if (m <= 1.0 || kslope <= 0.0) return -x;

	numb sg_prod = 1.0;    // mulst sg_i — накопленное произведение знаков
	numb offset = 0.0;     // sum (sg_prod_k * d_k) — накопленное смещение



	const int MAX_ITER = 10000;
	int iter = 0;

	while (iter < MAX_ITER) {
		// Базовый случай рекурсии
		if (abs(d) < 1e-14) {
			return sg_prod * (-x) - offset;
		}

		// 1. Определяем знак ТЕКУЩЕГО аргумента и работаем с |x|
		numb sg_current = (x < 0) ? -1.0 : 1.0;
		numb x_abs = (x < 0) ? -x : x;
		sg_prod *= sg_current;  // накапливаем произведение знаков

		// 2. Вычисляем пороги для текущего масштаба d
		numb thr1 = (m - 2) / (m - 1) / kslope * d;
		numb thr2 = (m - 2) / (m - 1) * d;
		numb thr3 = m / (m - 1) * d;



		// 3. Обработка нерекурсивных зон
		if (x_abs < thr1) {
			return sg_prod * (-kslope * x_abs) - offset;
		}
		else {
			if (x_abs < thr2) {
				return sg_prod * (-(m - 2) / (m - 1) * d) - offset;
			}
			else {
				if (x_abs > thr3) {
					numb y = (x_abs < dmargin * d)
						? -m / (m - 1) * d
						: -kslope * (x_abs - dmargin * d) - m / (m - 1) * d;
					return sg_prod * y - offset;
				}
				else {
					// 4. Рекурсивный случай: эквивалент f(x,d) = f(|x|-d, d/m) - d
					// Смещение -d умножается на ТЕКУЩЕЕ произведение знаков (уже включая sg_current)
					offset += sg_prod * d;
					x = x_abs - d;   // новый аргумент для следующей итерации (может быть < 0)
					d = d / m;       // уменьшаем масштаб
					++iter;
				}
			}	
	
		}
			
			

	}

	// Защита от зацикливания (теоретически недостижимо при корректных параметрах)
	return sg_prod * (-x) - offset;
}

__device__ __host__ numb psi_m(numb x, numb M, numb d, numb k, numb dmargin) {

	numb m_inner = M;
	numb m = 15;
	numb A = m - 1;
	numb C = 0.99;
	numb val = 0;
	numb sig_a = (dmargin * k - m_inner / (m_inner - 1)) * d / (k - 1);
	//phi_0 = @(x)(0.19 * ( chua_multistep_extended(x, m_inner, d, k, dmargin) + x) );
	numb x0 = sig_a;

	if (x == 0)
		x = 1e-15;
	
	bool flag = 1;

	while (flag) {
		numb x1 = m * x0;
		if (abs(x) > x1) { //if x is beyond local interval, increase
			x0 = m * x0;
			A = A * m;
		}
		else {
			if (abs(x) > x0 && abs(x) <= x1) { // if x is in local interval, compute
				flag = 0;
				//val = C * sign(x) * a * phi_0((abs(x) - x0) / a);
				numb x_arg = (abs(x) - x0) / A;
				val = C * sign(x) * A * ( 0.19 * ( chua_multistep_extended(x_arg, m_inner, d, k, dmargin ) + x_arg ) );
										//( 0.19 * (chua_multistep_extended( x,				  m_inner, d, k, dmargin ) + x)					);
			}
			else {
				x0 = x0 / m;
				A = A / m;
			}
		}

		}
	return val;
}

__device__ __host__ void calculateDiscreteModelforFastSynchro(numb* X, numb* S1, numb* K, const numb* a, const numb h, const bool directionOfintegration)
{

	numb N[AMOUNTOFX], h1, h2;

	if (directionOfintegration == 1) {
		h1 = h * a[0];
		h2 = h * (1 - a[0]);
		for (int i = 0 ; i < AMOUNTOFX; i ++)
			N[i] = K[i] * (S1[i] - X[i]);
	}
	if (directionOfintegration == 0) {
		h1 = -h * a[0];
		h2 = -h * (1 - a[0]);
		for (int i = 0; i < AMOUNTOFX; i++)
			N[i] = -K[i] * (S1[i] - X[i]);
	}


	//// --- CANG 4D CD ---
	//X[0] =  X[0] + h1 * ( - a[1] * X[0] - a[5] * X[3] + X[1] * X[2]);
	//X[1] =  X[1] + h1 * (  a[2] * X[1] + X[0] * X[2]);
	//X[2] =  X[2] + h1 * (  a[3] * X[2] + a[6] * X[3] - X[0] * X[1]);
	//X[3] =  X[3] + h1 * (  a[4] * X[3] - a[7] * X[2]);
	//X[3] = (X[3] + h2 * ( - a[7] * X[2])) / (1 - h2 * a[4]);
	//X[2] = (X[2] + h2 * (  a[6] * X[3] - X[0] * X[1])) / (1 - h2 * a[3]);
	//X[1] = (X[1] + h2 * (  X[0] * X[2])) / (1 - h2 * a[2]);
	//X[0] = (X[0] + h2 * ( - a[5] * X[3] + X[1] * X[2])) / (1 + h2 * a[1]);

	////// --- Lorenz CD ---
	X[0] = X[0] + h1 * (a[1] * (X[1] - X[0]) );
	X[1] = X[1] + h1 * (X[0] * (a[2] - X[2]) - X[1] );
	X[2] = X[2] + h1 * (X[0] * X[1] - a[3] * X[2] );
	X[2] = (X[2] + h2 * (X[0] * X[1] )) / (1 + h2 * a[3]);
	X[1] = (X[1] + h2 * (X[0] * (a[2] - X[2]) )) / (1 + h2);
	X[0] = (X[0] + h2 * (a[1] * (X[1]) )) / (1 + a[1] * h2);

	//// --- Nose-Hoover CD ---
	//X[0] = X[0] + h1 * (a[1] * X[1] );
	//X[1] = (X[1] - h1 * (X[0] )) / (1 - h1 * a[2] * X[2]);
	//X[2] = X[2] + h1 * (1 - a[3] * X[1] * X[1] );
	//X[2] = X[2] + h2 * (1 - a[3] * X[1] * X[1] );
	//X[1] = X[1] + h2 * (a[2] * X[1] * X[2] - X[0] );
	//X[0] = X[0] + h2 * (a[1] * X[1] );

	// --- Rossler CD ---
	//X[0] = X[0] + h1 * (-X[1] - X[2] );
	//X[1] = (X[1] + h1 * (X[0] )) / (1.0 - a[1] * h1);
	//X[2] = (X[2] + h1 * (a[2] )) / (1.0 - h1 * (X[0] - a[3]));
	//X[2] = X[2] + h2 * (a[2] + X[2] * (X[0] - a[3]));
	//X[1] = X[1] + h2 * (X[0] + a[1] * X[1] );
	//X[0] = X[0] + h2 * (-X[1] - X[2] );

	// --- Sprott14 CD ---
	//X[0] = X[0] + h1 * (X[1] + 2.0 * X[0] * X[1] + X[0] * X[2]);
	//X[1] = X[1] + h1 * (1.0 - 2.0 * X[0] * X[0] + X[1] * X[2]);
	//X[2] = X[2] + h1 * (X[0] - X[0] * X[0] - X[1] * X[1]);
	//X[2] = X[2] + h2 * (X[0] - X[0] * X[0] - X[1] * X[1]);
	//X[1] = (X[1] + h2 * (1.0 - 2.0 * X[0] * X[0])) / (1.0 - h2 * X[2]);
	//X[0] = (X[0] + h2 * (X[1])) / (1.0 - h2 * (2.0 * X[1] + X[2]));

	// --- Lorenz-like 5D CD ---
	//X[0] = X[0] + h1 * (a[1] * (X[1] - X[0]) + X[3]);
	//X[1] = X[1] + h1 * (a[3] * X[0] - X[0] * X[2] + X[4]);
	//X[2] = X[2] + h1 * (-a[2] * X[2] + X[0] * X[1]);
	//X[3] = X[3] + h1 * (-a[4] * X[3] - X[0] * X[2]);
	//X[4] = X[4] + h1 * (-a[5] * X[0] - a[6] * X[1]);
	//X[4] = X[4] + h2 * (-a[5] * X[0] - a[6] * X[1]);
	//X[3] = (X[3] + h2 * (-X[0] * X[2])) / (1.0 + h2 * a[4]);
	//X[2] = (X[2] + h2 * X[0] * X[1]) / (1.0 + h2 * a[2]);
	//X[1] = X[1] + h2 * (a[3] * X[0] - X[0] * X[2] + X[4]);
	//X[0] = (X[0] + h2 * (a[1] * X[1] + X[3])) / (1.0 + h2 * a[1]);

	//numb U[3];
	//N[0] = (S1[0] - X[0]);
	//N[1] = (S1[1] - X[1]);
	//N[2] = (S1[2] - X[2]);
	//if (directionOfintegration == 1) {
	//	U[0] = K[0] * N[0];
	//	U[1] = -(S1[0] * S1[2] - X[0] * X[2]) + K[1] * N[1];
	//	U[2] = (S1[0] * S1[1] - X[0] * X[1]) + K[2] * N[2];
	//	//U[0] = K[0] * N[0];
	//	//U[1] = K[1] * N[1];
	//	//U[2] = K[2] * N[2];
	//	X[3] = X[3] + h * (N[0] * (X[1] - X[0]) - N[1] * X[0]); //a[1]
	//	X[4] = X[4] + h * (-N[2] * X[2]);						//a[2]
	//	X[5] = X[5] + h * (N[1] * (X[0] + X[1]));				//a[3]
	//}
	//if (directionOfintegration == 0) {
	//	U[0] = -K[0] * N[0];
	//	U[1] = -(S1[0] * S1[2] - X[0] * X[2]) - K[1] * N[1];
	//	U[2] = (S1[0] * S1[1] - X[0] * X[1]) - K[2] * N[2];
	//	//U[0] = -K[0] * N[0];
	//	//U[1] = -K[1] * N[1];
	//	//U[2] = -K[2] * N[2];
	//	X[3] = X[3] - h * (N[0] * (X[1] - X[0]) - N[1] * X[0]); //a[1]
	//	X[4] = X[4] - h * (-N[2] * X[2]);						//a[2]
	//	X[5] = X[5] - h * (N[1] * (X[0] + X[1]));				//a[3]
	//}
	//X[0] = X[0] + h1 * ( U[0] + X[3] * (X[1] - X[0]));
	//X[1] = X[1] + h1 * ( U[1] + (X[5] - X[3]) * X[0] - X[0] * X[2] + X[5] * X[1]);
	//X[2] = X[2] + h1 * ( U[2] + X[0] * X[1] - X[4] * X[2]);
	//X[2] = (X[2] + h2 * (U[2] + X[0] * X[1])) / (1 + X[4] * h2);
	//X[1] = (X[1] + h2 * (U[1] + (X[5] - X[3]) * X[0] - X[0] * X[2])) / (1 - X[5] * h2);
	//X[0] = (X[0] + h2 * (U[0] + X[3] * (X[1]))) / (1 + h2 * X[3]);

	//Chua Shirnin CD
	/*numb Ep = 1.0; //positive E
	numb En = -0.9; //negative E
	numb an = -2956.4381;
	numb bn = -1133.7795;
	numb cn = 4892.6766;
	numb ap = 3140.4641;
	numb bp = -1149.9195;
	numb cp = 4894.9515;
	numb ac = -38.7605;
	numb bc = 2065.6838;
	numb cc = 5031.1177;
	numb dc = -122.2301;
	numb py = 452.7923;
	numb qy = -8916.0849;
	numb ry = 8480.8662;
	numb pz = 452.0403;
	numb qz = -9447.7078;
	numb rz = 8465.1742;
	//numb h2 = h / 2.0;
	numb x = X[0]; 
	numb y = X[1]; 
	numb z = X[2];

	// half step forward (implicit)
	z = (z + h2 * (pz * x + qz * y)) / (1 - h2 * rz);
	y = (y + h2 * (py * x + ry * z)) / (1 - h2 * qy);
	if (x < En) {
		x = (x + h2 * (an + cn * y)) / (1 - h2 * bn);
	}
	else if (x > Ep) {
		x = (x + h2 * (ap + cp * y)) / (1 - h2 * bp);
	}
	else {
		x = (x + h2 * (ac + cc * y + dc * z)) / (1 - h2 * bc);
	}

	// half step forward(explicit)
	if (x < En) {
		x = x + h2 * (an + bn * x + cn * y);
	}
	else if (x > Ep) {
		x = x + h2 * (ap + bp * x + cp * y);
	}
	else {
		x = x + h2 * (ac + bc * x + cc * y + dc * z);
	}
	y = y + h2 * (py * x + qy * y + ry * z);
	z = z + h2 * (pz * x + qz * y + rz * z);

	X[0] = x; 
	X[1] = y; 
	X[2] = z;
	*/

	// Need for conventional FastSynchro
	if (directionOfintegration == 1) {
		for (int i = 0; i < AMOUNTOFX; i++)
			X[i] = X[i] + N[i] * h;
	}
	if (directionOfintegration == 0) {
		for (int i = 0; i < AMOUNTOFX; i++)
			X[i] = X[i] - N[i] * h;
	}


}

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
	numb* FastSynchroError)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ numb s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	// 1 - stability, 0 - fixed point, -1 - unbound solution
	FastSynchroError[idx] = loopCalculateDiscreteModelForFastSynchro_2(localX, initialConditionsSlave, localValues, h, amountOfIterations,
		amountOfInitialConditions, preScaller, maxValue, iterOfSynchr, kForward, kBackward, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}

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
	const int writeStep)
{
	//numb* Xm = new numb[amountOfX];
	//numb* Xs = new numb[amountOfX];
	//numb* arrayZeros = new numb[amountOfX];
	//numb* K_local = new numb[amountOfX];
	
	numb Xm[AMOUNTOFX];
	numb Xs[AMOUNTOFX];
	numb X_prev[AMOUNTOFX];
	numb arrayZeros[AMOUNTOFX];
	numb K_local[AMOUNTOFX];

	numb rms_error = 0;

	if (data != nullptr) {
		for (int w = 0; w < amountOfX; w++) {
			data[startDataIndex + w] = x[w];
			//Xs[w] = x[w] - 0.005;
			Xs[w] = initialConditionsSlave[w];
			Xm[w] = x[w];
			arrayZeros[w] = 0;
		}
	}
	if (type_of_synch == 0) {
		for (int i = 1; i < amountOfIterations; i++) {
			calculateDiscreteModelforFastSynchro(Xm, arrayZeros, arrayZeros, values, h, 1);

			for (int w = 0; w < amountOfX; w++)
				data[startDataIndex + w + i * amountOfX] = Xm[w];
		}
	}

	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = kForward[j];

		// --- Глобальный цикл, который производит вычисления заданные amountOfIterations раз ---
		for (int i = 0; i < amountOfIterations - 1; ++i) {

			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j) {
					Xm[j] = data[startDataIndex + i * amountOfX + j];
					//Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
				}
			}

			if (error_estim == 0) {
				if (m == iterOfSynchr - 1) {
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 1) {
				if (i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 2) {
				if (m == iterOfSynchr - 1 && i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 3) {
				rms_error = 0;
				for (int j = 0; j < amountOfX; ++j)
					rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				rms_error = (sqrt(rms_error));
				if (rms_error <= FS_error_trs) {
					return (numb)(i + 1) * h;
				}

			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 1);
			
			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 1);
			}
		
		}

		if (error_estim == 1) {
			rms_error = (sqrt(rms_error));
			if (rms_error <= FS_error_trs) {
				//delete[] Xm;
				//delete[] Xs;
				//delete[] X_prev;
				//delete[] K_local;
				//delete[] norm_error;

				return (numb)(m + 1);
			}
			else
				rms_error = 0;
		}

		for (int j = 0; j < amountOfX; ++j)
			K_local[j] = kBackward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = data[startDataIndex + i * amountOfX + j];
			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 0);

			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 0);
			}
		}
	}

	if (error_estim == 0)
		rms_error = sqrt(rms_error / (numb)(amountOfIterations - 1));

	if (error_estim == 2)
		rms_error = sqrt(rms_error);

	if (rms_error <= FS_error_trs)
		rms_error = log10(FS_error_trs);
	else
		rms_error = log10(rms_error);

	if (isinf(rms_error) || isnan(rms_error) || rms_error >= 3)
		rms_error = 3;

	//delete[] Xm;
	//delete[] Xs;
	//delete[] X_prev;
	//delete[] K_local;
	//delete[] norm_error;

	if (error_estim == 0)
		return rms_error;
	if (error_estim == 1)
		return (numb)iterOfSynchr;
	if (error_estim == 2)
		return rms_error;
	if (error_estim == 3)
		return (numb)amountOfIterations * h;
}

__device__  bool loopCalculateDiscreteModel(numb* x, const numb* values, 
	const numb h, const int amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, numb* data, 
	const int startDataIndex, const int writeStep)
{

	numb xPrev[AMOUNTOFX];
	// --- Глобальный цикл, который производит вычисления заданные amountOfIterations раз ---
	for ( int i = 0; i < amountOfIterations; ++i )
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// --- Если все-таки передали массив для записи - записываем значение переменной ---
		if ( data != nullptr )
			data[startDataIndex + i * writeStep] = x[writableVar];

		// --- Моделируем систему preScaller раз ( то есть если preScaller > 1, то мы пропустим ( preScaller - 1 ) в смоделированной траектории ) ---
		for ( int j = 0; j < preScaller; ++j )
			calculateDiscreteModel(x, values, h);

		// --- Если isnan или isinf - возвращаем false, ибо это нежелательное поведение системы ---
		if ( isnan( x[writableVar] ) || isinf( x[writableVar] ) )
		{
			return false;
		}

		// --- Если maxValue == 0, это значит пользователь не выставил ограничение, иначе требуется его проверить ---
		if ( maxValue != 0 )
			if ( fabsf( x[writableVar] ) > maxValue )
			{
				return false;
			}
	}

	// --- Проверка на сваливание в точку ---
	//numb tempResult = 0;
	//for (int j = 0; j < amountOfX; ++j)
	//{
	//	tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	//}

	//if (tempResult == 0)
	//{
	//	delete[] xPrev;
	//	return false;
	//}

	//if (sqrt(tempResult) < 1e-12)
	//{
	//	delete[] xPrev;
	//	return false;
	//}

	return true;
}


__device__  __host__ int loopCalculateDiscreteModel_int(
	numb* x, const numb* values,
	const numb h, const int amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const numb maxValue, numb* data,
	const size_t startDataIndex, const int writeStep)
{
	//numb* xPrev = new numb[amountOfX];
	numb xPrev[AMOUNTOFX];
	numb checker;


	// --- Глобальный цикл, который производит вычисления заданные amountOfIterations раз ---
	for (size_t i = 0; i < amountOfIterations; ++i)
	{	
		//#pragma unroll
		//for (int j = 0; j < AMOUNTOFX; ++j)
		//{
		//	xPrev[j] = x[j];
		//}
		// --- Если все-таки передали массив для записи - записываем значение переменной ---


		if (data != nullptr) {
			//data[startDataIndex + i] = x[0] + pi* x[1] + euler*x[2];
			data[startDataIndex + i] = (x[writableVar]);

			//#pragma unroll
			for (int j = 0; j < preScaller; ++j) {
				//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
				calculateDiscreteModel(x, values, h);
			}
		}
		else
			calculateDiscreteModel(x, values, h);
	
		if (i % CHECK_INTERVAL == 0) {
			checker = 0;
			//#pragma unroll
			for (int j = 0; j < AMOUNTOFX; ++j) {
				checker = checker + abs(x[j]);
			}

			// --- Если isnan или isinf - возвращаем false, ибо это нежелательное поведение системы ---
			if (isnan(checker) || isinf(checker))
			{
				//delete[] xPrev;
				return 0;
			}

			// --- Если maxValue == 0, это значит пользователь не выставил ограничение, иначе требуется его проверить ---
			if (maxValue != 0)
				if (abs(checker) > maxValue)
				{
					//delete[] xPrev;
					return 0;
				}
		}
	}

	//#pragma unroll
	for (int j = 0; j < AMOUNTOFX; ++j)
	{
		xPrev[j] = x[j];
	}
	// --- Если все-таки передали массив для записи - записываем значение переменной ---



	//#pragma unroll
	for (int j = 0; j < preScaller; ++j) {
		//calculateDiscreteModel_rand(startDataIndex + i * writeStep, x, values, h);
		calculateDiscreteModel(x, values, h);
	}



	//// 1 - stability, -1 - fixed point, 0 - unbound solution
	//// --- Проверка на сваливание в точку ---
	numb tempResult = 0;

	//for (int j = 0; j < 1; ++j)
	//#pragma unroll
	for (int j = 0; j < AMOUNTOFX; ++j)
	{
		tempResult += abs(x[j] - xPrev[j]);
	}

	//if (tempResult == 0)
	//{
	//	//delete[] xPrev;
	//	return -1;
	//}

	if (abs(tempResult) < eps_fixed_point)
	{
		//delete[] xPrev;
		return -1;
	}

	//delete[] xPrev;
	return 1;
}


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
	const int		writableVar,
	numb*			data)
{
	extern __shared__ numb s[];
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfThreads)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0);

	loopCalculateDiscreteModel(localX, localValues, h, idx,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0, 0);

	loopCalculateDiscreteModel(localX, localValues, hSpecial, amountOfIterations,
		amountOfInitialConditions, 1, writableVar, 0, data, idx, amountOfThreads);

	return;
}



// --------------------------------------------------------------------------
// --- Глобальная функция, которая вычисляет траекторию нескольких систем ---
// --------------------------------------------------------------------------

__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const size_t	sizeOfBlock,
	const size_t	amountOfCalculatedPoints,
	const size_t	amountOfPointsForSkip,
	const int		dimension, 
	numb* __restrict__			ranges,
	const numb		h,
	int* __restrict__			indicesOfMutVars,
	numb* __restrict__			initialConditions,
	const int		amountOfInitialConditions, 
	const numb* __restrict__	values,
	const int		amountOfValues,
	const size_t		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const numb	maxValue, 
	numb*			data, 
	int*			maxValueCheckerArray, 
	const bool		Par_or_Var)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	
	extern __shared__ numb s[];
	//////// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	numb* localX = s + ( threadIdx.x * amountOfInitialConditions );
	numb* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );


	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	#pragma unroll
	for ( int i = 0; i < AMOUNTOFX; ++i )
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---

	//if (par_or_var){
	//	for (int i = 0; i < dimension; ++i)
	//		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	//}
	//else {
	//	for (int i = 0; i < dimension; ++i)
	//		localX[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	//}

	if (par_or_var == 1) {
		for (int i = 0; i < dimension; ++i) {
			if (LINEAR_OR_LOG_DISTRIB == 1) localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
			if (LINEAR_OR_LOG_DISTRIB == 0) localValues[indicesOfMutVars[i]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 0) {
		for (int i = 0; i < dimension; ++i) {
			if (LINEAR_OR_LOG_DISTRIB == 1) localX[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
			if (LINEAR_OR_LOG_DISTRIB == 0) localX[indicesOfMutVars[i]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		}
	}
	else if (par_or_var == 2) {
		if (LINEAR_OR_LOG_DISTRIB == 1) localX[indicesOfMutVars[0]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
		if (LINEAR_OR_LOG_DISTRIB == 1) localValues[indicesOfMutVars[1]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
		if (LINEAR_OR_LOG_DISTRIB == 0) localX[indicesOfMutVars[0]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
		if (LINEAR_OR_LOG_DISTRIB == 0) localValues[indicesOfMutVars[1]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}

	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, preScaller, writableVar, maxValue, nullptr, (size_t)idx * sizeOfBlock, 1);

	// --- Теперь уже по-взрослому моделируем систему --- 
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,
			amountOfInitialConditions, preScaller, writableVar, maxValue, data, (size_t)idx * sizeOfBlock, 1);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}
	//delete[] localX;
	//delete[] localValues;
	return;
}



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
	const numb	tMax,
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			data,
	int*			maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ numb s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	//for (int i = 0; i < dimension; ++i)
	//	localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
	//		nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	numb h = (numb)pow(10.0, getValueByIdxLog(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0));

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, transientTime / h,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, tMax / h / preScaller,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;
	else
		maxValueCheckerArray[idx] = tMax / h / preScaller;

	return;
}



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
	const int		preScaller,
	const int		writableVar, 
	const numb	maxValue, 
	numb*			data, 
	int*			maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ numb s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	numb* localX = s + ( threadIdx.x * amountOfInitialConditions );
	numb* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	//numb* localX = new numb[amountOfInitialConditions];
	//numb* localValues = new numb[amountOfValues];

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	//__device__ __host__ numb getValueByIdx(const int idx, const int nPts,
	//	const numb startRange, const numb finishRange, const int valueNumber)
	//{
	//	return startRange + (((int)((int)idx / powf((numb)nPts, (numb)valueNumber)) % nPts)
	//		* ((numb)(finishRange - startRange) / (numb)(nPts - 1)));
	//}

	//// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	//bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
	//	amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	//// --- Теперь уже по-взрослому моделируем систему --- 
	//if (flag)
	//	flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
	//	amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	//// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	//if (!flag && maxValueCheckerArray != nullptr)
	//	maxValueCheckerArray[idx] = -1;
	//else
	//	maxValueCheckerArray[idx] = 1;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
	
	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations,amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}

	//delete[] localX;
	//delete[] localValues;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return;
}

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
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb* data,
	int* maxValueCheckerArray)
{
	// --- Общая память в рамках одного блока ---
	// --- Строение памяти: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., следуюший поток...} ---
	extern __shared__ numb s[];

	// --- В каждом потоке создаем указатель на параметры и переменные, чтобы работать с ними как с массивами ---
	numb* localX = s + (threadIdx.x * amountOfInitialConditions);
	numb* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);


	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Определяем localX[] начальными условиями ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- Определяем localValues[] начальными параметрами ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Меняем значение изменяемых параметров на результат функции getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx_forLogBains(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 

	// 1 - stability, 0 - fixed point, -1 - unbound solution
	int flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Теперь уже по-взрослому моделируем систему --- 
	if (flag == 1 || flag == -1)
		flag = loopCalculateDiscreteModel_int(localX, localValues, h, amountOfIterations, amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Если функция моделирования выдала false - значит мы даже не будем смотреть на эту систему в дальнейшем анализе ---

	if (maxValueCheckerArray != nullptr) {
		maxValueCheckerArray[idx] = flag;
	}
	return;
}

// --- Функция, которая находит индекс в последовательности значений ---

//__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
//	const numb startRange, const numb finishRange, const int valueNumber)
//{
//	return startRange + (numb)( ( (int64_t)( (int64_t)idx / pow( (numb)nPts, (numb)valueNumber) ) % (int64_t)nPts ) * ( (numb)( finishRange - startRange ) / (numb)( nPts - 1.0) ) );
//}

//__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
//	const numb startRange, const numb finishRange,
//	const int valueNumber)
//{
//	if (nPts <= 1) return startRange; // Или (startRange + finishRange) / 2, в зависимости от логики
//
//	// 1. Целочисленное вычисление индекса точки по оси (без pow и float-деления)
//	// Если valueNumber == 0 (ось X), делитель равен 1.
//	// Если valueNumber == 1 (ось Y), делитель равен nPts.
//	const int64_t divisor = (valueNumber == 0) ? 1 : nPts;
//	const int64_t pointIdx = (idx / divisor) % nPts;
//
//
//	// 2. Вычисление шага сетки. 
//	// Приводим (nPts - 1) к типу numb, чтобы избежать неявного повышения до double
//	const numb step = (finishRange - startRange) / static_cast<numb>(nPts - 1);
//
//	// 3. Итоговое значение
//	return startRange + static_cast<numb>(pointIdx) * step;
//}

__device__ __host__ numb getValueByIdx(const size_t idx, const int nPts,
	const numb startRange, const numb finishRange,
	const int valueNumber)
{
	// 1. Обработка вырожденных случаев
	if (nPts <= 0) return startRange;
	if (nPts == 1) return finishRange;

	// 2. Целочисленное вычисление индекса точки по оси
	const int64_t divisor = (valueNumber == 0) ? 1 : nPts;
	const int64_t pointIdx = (idx / divisor) % nPts;

	// 3. Вычисление параметра t (от 0.0 до 1.0)
	const numb t = static_cast<numb>(pointIdx) / static_cast<numb>(nPts - 1);

	// 4. Формула линейной интерполяции (обеспечивает строгую симметрию)
	return (static_cast<numb>(1.0) - t) * startRange + t * finishRange;
}

__device__ __host__ numb getValueByIdx_forLogBains(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	numb log10_xstart = log10(startRange);
	numb log10_xstop = log10(finishRange);
	int n = ((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);
	//0 1 2 3 4 5	n = 6
	if (n < nPts / 2) {
		return -pow(10, log10_xstart + (numb)n * (log10_xstop - log10_xstart) / (numb)(nPts / 2 - 1));
	}
	else {
		return pow(10, log10_xstop + (numb)(n - nPts / 2) * (log10_xstart - log10_xstop) / (numb)(nPts / 2 - 1));
	}

}


// --- Функция, которая находит индекс в последовательности значений ---
__device__ __host__ numb getValueByIdxLog(const int idx, const int nPts,
	const numb startRange, const numb finishRange, const int valueNumber)
{
	return log10(startRange) + (((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts)
		* ((numb)(log10(finishRange) - log10(startRange)) / (numb)(nPts - 1)));
}



// ---------------------------------------------------------------------------------------------------
// --- Находит пики в интервале [startDataIndex; startDataIndex + amountOfPoints] в "data" массиве ---
// ---------------------------------------------------------------------------------------------------
//peakFinder(data, idx* sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);


__device__ __host__ void MeanAndMedianFreq(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq, numb* medianFreq)
{
	numb MeanFreq = 0;

	for (size_t i = 0; i < amountOfPeaks; i++) {
		MeanFreq = MeanFreq + ((numb)1.0 / timeOfPeaks[startDataIndex + i]);
	}
	meanFreq[idx] = MeanFreq / (numb)amountOfPeaks;

	medianFreq[idx] = 1.0 / timeOfPeaks[startDataIndex];

	//for (size_t i = 0; i < amountOfPeaks - 1; ++i) {
	//	size_t min_idx = i;
	//	for (size_t j = i + 1; j < amountOfPeaks; ++j) {
	//		if (timeOfPeaks[startDataIndex + j] < timeOfPeaks[startDataIndex + min_idx]) {
	//			min_idx = j;
	//		}
	//	}
	//	numb localII = timeOfPeaks[startDataIndex + min_idx];
	//	timeOfPeaks[startDataIndex + min_idx] = timeOfPeaks[startDataIndex + i];
	//	timeOfPeaks[startDataIndex + i] = localII;
	//}

	//if (amountOfPeaks % 2 == 0) {
	//	//medianFreq[idx] = 0.5 *((1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks)) - 1]) + (1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks)) + 0]));
	//	medianFreq[idx] = 0.5 * ((1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks)) ]) + (1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks)) + 1]));

	//}
	//else {
	//	//medianFreq[idx] = 1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks - 1))];
	//	medianFreq[idx] = 1.0 / timeOfPeaks[startDataIndex + (int)(0.5 * (amountOfPeaks + 1))];
	//}
}

__device__ __host__ void MeanAndVariance(const int idx, const int startDataIndex, int amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval)
{
	numb localValue = 0;

	for (size_t i = 0; i < amountOfPeaks; i++) {
		localValue = localValue + outPeaks[startDataIndex + i];
	}
	meanPeak[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; i++) {
		localValue = localValue + timeOfPeaks[startDataIndex + i];
	}
	meanInterval[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; ++i)
		localValue = localValue + (outPeaks[startDataIndex + i] - meanPeak[idx]) * (outPeaks[startDataIndex + i] - meanPeak[idx]);

	variancePeak[idx] = localValue / (numb)amountOfPeaks;

	localValue = 0;
	for (size_t i = 0; i < amountOfPeaks; ++i)
		localValue = localValue + (timeOfPeaks[startDataIndex + i] - meanInterval[idx]) * (timeOfPeaks[startDataIndex + i] - meanInterval[idx]);

	varianceInterval[idx] = localValue / (numb)amountOfPeaks;

	localValue = -9999999999;

	for (size_t i = 0; i < amountOfPeaks; ++i)
		if (outPeaks[startDataIndex + i] > localValue)
			localValue = outPeaks[startDataIndex + i];
	
	maxPeak[idx] = localValue;

	localValue = -9999999999;

	for (size_t i = 0; i < amountOfPeaks; ++i)
		if (timeOfPeaks[startDataIndex + i] > localValue)
			localValue = timeOfPeaks[startDataIndex + i];

	maxInterval[idx] = localValue;
}

__device__ __host__ numb globalPeakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints)
{
		// --- Переменная для хранения найденных пиков ---
		numb maxValue = -1e25;
		// --- Начинаем просматривать заданных интервал на наличие пиков ---
		for (size_t i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i)
		{
			// --- Если текущая точка больше предыдущей и больше ИЛИ РАВНА следующей, то... ( не факт, что это пик ( например: 2 3 3 4 ) ) ---
			if (data[i] >= maxValue) //
			{
				maxValue = data[i];
			}
				
		}

		return maxValue;
}

__device__ __host__ int peakFinder(numb* data, const size_t startDataIndex,
	const size_t amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h)
{

	if (doCalculatePeaks) {
		// --- Переменная для хранения найденных пиков ---
		int amountOfPeaks = 0;

		// --- Начинаем просматривать заданных интервал на наличие пиков ---
		for (size_t i = startDataIndex + 2; i < startDataIndex + amountOfPoints - 2; ++i)
		{
			// --- Если текущая точка больше предыдущей и больше ИЛИ РАВНА следующей, то... ( не факт, что это пик ( например: 2 3 3 4 ) ) ---
			if (data[i] - data[i - 1] > eps_peak_delta && data[i] > peak_threshold && data[i] >= data[i + 1]) //
			{
				// --- От найденной точки начинаем идти вперед, пока не наткнемся на точку строго больше или меньше ---
				for (size_t j = i; j < startDataIndex + amountOfPoints - 2; ++j)
				{
					// --- Если наткнулись на точку строго больше, значит это был не пик ---
					if (data[j] < data[j + 1])
					{
						i = j + 1;	// --- Обновляем внешний счетчик, чтобы дважды не проходить один и тот же интервал
						break;		// --- Возвращаемся к внешнему циклу
					}
					// --- Если о чудо, мы нашли точку меньше, чем текущая, значит мы нашли пик ---
					if (data[j] - data[j + 1] > eps_peak_delta)
					{
						
						if (doInterpolatePeaks) {
							numb denom = data[j - 1] - (numb)2.0 * data[j] + data[j + 1];
							numb delta = 0.0;
							//denom == (numb)0.0 ? delta = 0.0 : delta = 0.5 * (data[j - 1] - data[j + 1]) / denom;
							if (fabs(denom) > 1e-12) {
								delta = 0.5 * (data[j - 1] - data[j + 1]) / denom;
							}
							outPeaks[startDataIndex + amountOfPeaks] = data[j] - 0.25 * (data[j - 1] - data[j + 1]) * delta;
							timeOfPeaks[startDataIndex + amountOfPeaks] = (numb)(j - startDataIndex - 1) + delta; // в оригинале delta*h но у нас тут индексы, умнодение на h потом
						}
						else {
							// --- Если массик outPeaks не пуст, то делаем запись ---
							if (outPeaks != nullptr)
								outPeaks[startDataIndex + amountOfPeaks] = data[j]; //data[j];
							// --- Если массик timeOfPeaks не пуст, то делаем запись ---
							if (timeOfPeaks != nullptr)
								timeOfPeaks[startDataIndex + amountOfPeaks] = (numb)(j - startDataIndex - 1);	// (numb)(j - startDataIndex - 1);
							//timeOfPeaks[startDataIndex + amountOfPeaks] = (numb)(i - startDataIndex - 1);	// (numb)(j - startDataIndex - 1);
							//timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (numb)j + (numb)i ) / (numb)2 );	// Выбираем индекс посередине между j и i
						
						}
						++amountOfPeaks;
						i = j + 1; // Потому что следующая точка точно не может быть пиком ( два пика не могут идти подряд )
						break;
					}
				}
			}
		}

		// --- Вычисляем межпиковые интервалы ---
		if (amountOfPeaks > 1) {
			////////////////// --- Пробегаемся по всем найденным пикам и их индексам ---

			for (size_t i = 0; i < amountOfPeaks - 1; i++)
			{
				// --- Смещаем все пики на один индекс влево, а первый пик удаляем ---
				if (outPeaks != nullptr)
					outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
				// --- Вычисляем межпиковый интервал. Это разница индекса следующего пика и предыдущего, умноженная на шаг ---
				if (timeOfPeaks != nullptr)
					timeOfPeaks[startDataIndex + i] = (numb)( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h;

			}
			// --- Так как один пик удалили - вычитаем единицу из результата ---
			amountOfPeaks = amountOfPeaks - 1;

			//int writeIdx = 0;                          // индекс записи результата
			//int anchorIdx = 0;                         // индекс опорного пика (в исходных данных)
			//numb anchorTime = timeOfPeaks[startDataIndex];  // абсолютное время опорного пика

			//// Перебираем пики, начиная со второго (индекс 1)
			//for (size_t i = 1; i < amountOfPeaks; i++) {
			//	// Читаем абсолютное время текущего пика ДО любой записи
			//	numb currentTime = timeOfPeaks[startDataIndex + i];

			//	// Считаем интервал от опорного до текущего
			//	numb delta = (currentTime - anchorTime) * h;

			//	if (delta >= eps_interPeak_delta) {  // порог 
			//		// Записываем ВТОРОЙ пик (текущий) и интервал до него
			//		if (outPeaks != nullptr)
			//			outPeaks[startDataIndex + writeIdx] = outPeaks[startDataIndex + i];
			//		if (timeOfPeaks != nullptr)
			//			timeOfPeaks[startDataIndex + writeIdx] = delta;  // интервал от опорного до текущего

			//		writeIdx++;

			//		// Текущий пик становится новым опорным для следующего поиска
			//		anchorIdx = i;
			//		anchorTime = currentTime;
			//	}
			//	// Если delta < 75.0 — просто идём дальше, не записываем
			//}
			//amountOfPeaks = writeIdx;  // обновляем количество пиков в результате

		}
		else {
			amountOfPeaks = 0;
		}

		if (amountOfPeaks >= max_amount_of_peaks)
			amountOfPeaks = max_amount_of_peaks;

		return amountOfPeaks;
	}
	else {
		int amountOfPeaks = amountOfPoints;

		if (amountOfPeaks >= max_amount_of_peaks)
			amountOfPeaks = max_amount_of_peaks;

		for (size_t i = 0; i < amountOfPeaks; ++i)
		{
			if (outPeaks != nullptr)
				outPeaks[startDataIndex + i] = data[startDataIndex + i];

			if (timeOfPeaks != nullptr)
				timeOfPeaks[startDataIndex + i] = 0;

		}
		return amountOfPeaks - 1;
	}

}



// ----------------------------------------------------------------
// --- Нахождение пиков в "data" массиве в многопоточном режиме ---
// ----------------------------------------------------------------

__global__ void peakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if ( amountOfPeaks[idx] == -1 )
	{
		amountOfPeaks[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	
	amountOfPeaks[idx] = peakFinder( data, (size_t)idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h );
	return;
}

__global__ void globalPeakFinderCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* globalPeakValue)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == 0)
	{
		amountOfPeaks[idx] = 0;
		globalPeakValue[idx] = 1e25;
		return;
	}


	globalPeakValue[idx] = globalPeakFinder(data, (size_t)idx * sizeOfBlock, sizeOfBlock);
	return;
}


__global__ void DFT_custom(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* checkerArray, numb* AkCOS, numb* BkSIN, numb* rangesFreq, numb* window, int nFreq, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	size_t startDataIndex = idx * sizeOfBlock;
	size_t startIndexFreq = idx * nFreq;

	if (checkerArray[idx] == -1)
	{
		for (size_t i = 0; i < nFreq; i++) {
			AkCOS[startIndexFreq + i] = -1.0;
			BkSIN[startIndexFreq + i] = -1.0;
		}
		return;
	}

	if (checkerArray[idx] == 0)
	{
		for (size_t i = 0; i < nFreq; i++) {
			AkCOS[startIndexFreq + i] = 0.0;
			BkSIN[startIndexFreq + i] = 0.0;
		}
		return;
	}

	const int RESET_INTERVAL = 1000;

	const numb f_step = (rangesFreq[1] - rangesFreq[0]) / (numb)(nFreq - 1);
	const numb f_start = rangesFreq[0];
	const numb gamma = 2 * pi / (sizeOfBlock - 1.0);
	const numb psi = 2 * pi * h;
	numb f_k = 0;
	numb cos_theta = 0;
	numb sin_theta = 0;
	numb cos_n = 0;
	numb sin_n = 0;
	//numb theta = 2 * pi * h;
	//numb f_k = 0;

	for (size_t k = 0; k < nFreq; k++) {
		AkCOS[startIndexFreq + k] = 0;
		BkSIN[startIndexFreq + k] = 0;
		f_k = f_start + (numb)k * f_step;
		cos_theta = cos(2.0 * pi * h * f_k);
		sin_theta = sin(2.0 * pi * h * f_k);
		cos_n = 1.0;
		sin_n = 0.0;

		for (size_t n = 0; n < sizeOfBlock; n++) {

			//numb h_window = 0.53836 - 0.46164 * cos(gamma * n ); // Hamming
			//numb h_window = 0.5*(1.0 - cos(gamma * n)); // Hanning
			//numb h_window = 1.0;

			//AkCOS[startIndexFreq + k] += data[startDataIndex + n] * cos(theta * f_k * (numb)n);
			//BkSIN[startIndexFreq + k] += data[startDataIndex + n] * sin(theta * f_k * (numb)n);

			AkCOS[startIndexFreq + k] += window[n] * data[startDataIndex + n] * cos_n;
			BkSIN[startIndexFreq + k] += window[n] * data[startDataIndex + n] * sin_n;

			numb new_cos = cos_n * cos_theta - sin_n * sin_theta;
			numb new_sin = sin_n * cos_theta + cos_n * sin_theta;
			cos_n = new_cos;
			sin_n = new_sin;

			if ((n + 1) % RESET_INTERVAL == 0 && (n + 1) < sizeOfBlock) {
				numb exact_angle = psi * f_k * (numb)(n + 1.0);
				cos_n = cosf(exact_angle);
				sin_n = sinf(exact_angle);
			}
		}
		AkCOS[startIndexFreq + k] /= (numb)sizeOfBlock;
		BkSIN[startIndexFreq + k] /= (numb)sizeOfBlock;
	}

	//for (size_t i = 0; i < nFreq; i++) {
	//	AkCOS[idx * nFreq + i] = rangesFreq[0] + (numb)i * f_step;
	//	BkSIN[idx * nFreq + i] = 2 * i + 1;
	//}

	return;
}


__global__ void MeanAndMedianFreqCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanFreq,  numb* medianFreq)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1)
	{
		meanFreq[idx] = 0;
		medianFreq[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		meanFreq[idx] = 0;
		medianFreq[idx] = 0;
		return;
	}

	meanFreq[idx] = 0;
	medianFreq[idx] = 1;
	int AoP = amountOfPeaks[idx];
	MeanAndMedianFreq(idx, idx * sizeOfBlock, AoP, outPeaks, timeOfPeaks, meanFreq, medianFreq);
	//return;
}

__global__ void MeanAndVarianceCUDA(const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb* meanPeak, numb* variancePeak, numb* meanInterval, numb* varianceInterval, numb* maxPeak, numb* maxInterval)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1)
	{
		meanPeak[idx] = 0;
		variancePeak[idx] = 0;
		meanInterval[idx] = 0;
		varianceInterval[idx] = 0;
		maxPeak[idx] = 0;
		maxInterval[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		meanPeak[idx] = 0;
		variancePeak[idx] = 0;
		meanInterval[idx] = 0;
		varianceInterval[idx] = 0;
		maxPeak[idx] = 0;
		maxInterval[idx] = 0;
		return;
	}

	if (amountOfPeaks[idx] == 1)
	{
		meanPeak[idx] = outPeaks[idx];
		variancePeak[idx] = 0;
		meanInterval[idx] = timeOfPeaks[idx];
		varianceInterval[idx] = 0;
		maxPeak[idx] = outPeaks[idx];
		maxInterval[idx] = timeOfPeaks[idx];
		return;
	}

	meanPeak[idx] =			999;
	variancePeak[idx] =		999;
	meanInterval[idx] =		999;
	varianceInterval[idx] = 999;
	maxPeak[idx] =			999;
	maxInterval[idx] =		999;

	int AoP = amountOfPeaks[idx];
	MeanAndVariance(idx, idx * sizeOfBlock, AoP, outPeaks, timeOfPeaks, meanPeak, variancePeak, meanInterval, varianceInterval, maxPeak, maxInterval);
	//return;
}


__global__ void peakFinderCUDA_H(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, amountOfPeaks[idx], outPeaks, timeOfPeaks, h);
	return;
}



__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, bool* flags, numb ostrovskyThreshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		flags[idx * 5 + 3] = true;
		return;
	}

	numb lastPoint = data[idx * sizeOfBlock + sizeOfBlock - 1];

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks);

	//FIRST CONDITION
	flags[idx * 5 + 0] = true;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] - outPeaks[i - 1] > 0)
		{
			flags[idx * 5 + 0] = false;
			break;
		}
	}

	//SECOND & THIRD CONDITION
	bool flagOne = false;
	bool flagZero = false;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] > ostrovskyThreshold)
			flagOne = true;
		else
			flagZero = true;
		if (flagOne && flagZero)
			break;
	}

	if (flagOne && flagZero)
		flags[idx * 5 + 1] = true;
	else
		flags[idx * 5 + 1] = false;

	if (flagOne && !flagZero)
		flags[idx * 5 + 2] = false;
	else
		flags[idx * 5 + 2] = true;

	//FOUR CONDITION
	if (amountOfPeaks[idx] == 0 || amountOfPeaks[idx] == 1)
		flags[idx * 5 + 3] = true;
	else
		flags[idx * 5 + 3] = false;

	//FIVE CONDITION
	if (lastPoint > ostrovskyThreshold)
		flags[idx * 5 + 4] = true;
	else
		flags[idx * 5 + 4] = false;
	return;
}



__device__ __host__ int kde(numb* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, numb kdeSamplesInterval1,
	numb kdeSamplesInterval2, numb kdeSmoothH)
{
	if (amountOfPoints == 0)
		return 0;
	if (amountOfPoints == 1 || amountOfPoints == 2)
		return 1;
	if (amountOfPoints > maxAmountOfPeaks)
		return maxAmountOfPeaks;

	numb k1 = kdeSampling * amountOfPoints;
	numb k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	numb delt = 0;
	numb prevPrevData2 = 0;
	numb prevData2 = 0;
	numb data2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < amountOfPoints; ++m)
		{
			numb tempData = (data[startDataIndex + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}

		if (w < 2)
			continue;
		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	return resultKde;
}



__global__ void kdeCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, numb kdeSamplesInterval1,
	numb kdeSamplesInterval2, numb kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		kdeResult[idx] = 0;
		return;
	}
	kdeResult[idx] = kde(data, idx * sizeOfBlock, amountOfPeaks[idx], maxAmountOfPeaks,
		kdeSampling, kdeSamplesInterval1, kdeSamplesInterval2, kdeSmoothH);
}


// ------------------------------------------------
// --- Вычисляет расстояние между двумя точками ---
// ------------------------------------------------

__device__ __host__ numb distance(numb x1, numb y1, numb x2, numb y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;

	numb dx = abs(x2 - x1);
	numb dy = abs(y2 - y1);

	//return hypot(dx, dy);
	return sqrt(dx*dx + dy*dy);
}



// ----------------------
// --- Функция DBSCAN ---
// ----------------------
__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const numb eps, int* outData)
{
	// ------------------------------------------------------------
	// --- Если пиков 0 или 1 - даже не обрабатываем эти случаи ---
	// ------------------------------------------------------------

	if (amountOfPeaks == -1)
		return -1;

	if (amountOfPeaks == 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	//if (amountOfPeaks >= 3600)
	//	return 0;


	// ------------------------------------------------------------


	int cluster = 0;
	int NumNeibor = 0;

	for (size_t i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	// ------------------------------------------------------------
	//for (int i = 0; i < amountOfPeaks; i++) {
	//	helpfulArray[startDataIndex + i] = (int)(100*sqrt(data[startDataIndex + i] * data[startDataIndex + i] + intervals[startDataIndex + i] * intervals[startDataIndex + i]));
	//}
	// ------------------------------------------------------------


	for (size_t i = 0; i < amountOfPeaks; i++) {
		data[startDataIndex + i] = data[startDataIndex + i] * mult_peak;
		intervals[startDataIndex + i] = intervals[startDataIndex + i] * mult_interval;
	}


	for (size_t i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1];
			helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}
		else if (helpfulArray[startDataIndex + i] == 0) {
			NumNeibor = 0;
			++cluster;
			helpfulArray[startDataIndex + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
				
			}
		}

	return cluster - 1;
}


//__device__ __host__ int dbscan(numb* data, numb* intervals, numb* helpfulArray,
//	const size_t startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
//	const int idx, const numb eps, int* outData)
//{
//	// ------------------------------------------------------------
//	// --- Валидация входных параметров ---
//	// ------------------------------------------------------------
//
//	// Некорректное количество пиков
//	if (amountOfPeaks <= 0)
//		return 0;
//
//	// Проверка: достаточно ли памяти в helpfulArray?
//	// Нам нужно: amountOfPeaks (метки) + amountOfPeaks (стек)
//	// Если памяти мало — работаем в безопасном режиме (только метки, но медленно)
//	const bool hasStackBuffer = (sizeOfHelpfulArray >= 2 * amountOfPeaks);
//
//	// ------------------------------------------------------------
//	// --- Предобработка данных (масштабирование) ---
//	// ------------------------------------------------------------
//
//	// Предполагается, что mult_peak и mult_interval определены глобально 
//	// или переданы через параметры/константы
//	for (int i = 0; i < amountOfPeaks; i++) {
//		data[startDataIndex + i] = data[startDataIndex + i] * mult_peak;
//		intervals[startDataIndex + i] = intervals[startDataIndex + i] * mult_interval;
//	}
//
//	// ------------------------------------------------------------
//	// --- Инициализация массива меток ---
//	// ------------------------------------------------------------
//
//	// helpfulArray[startDataIndex + i] будет хранить ID кластера для точки i
//	// 0 означает "не посещен" (кластеры нумеруются с 1)
//
//	// Оптимизация: очищаем только нужный диапазон
//	// Если helpfulArray используется где-то еще, возможно, потребуется очистка всего sizeOfHelpfulArray
//	for (int i = 0; i < amountOfPeaks; ++i) {
//		helpfulArray[startDataIndex + i] = 0;
//	}
//
//	// Указатель на стек в helpfulArray (если есть место)
//	// Стек хранит индексы точек для обхода
//	// Расположен сразу после меток: [метки][стек]
//	// Индекс в helpfulArray для стека: startDataIndex + amountOfPeaks + offset
//	int* stackBuffer = hasStackBuffer ?
//		reinterpret_cast<int*>(&helpfulArray[startDataIndex + amountOfPeaks]) : nullptr;
//
//	int cluster_count = 0;
//
//	// ------------------------------------------------------------
//	// --- Основной цикл кластеризации (Поиск связных компонент) ---
//	// ------------------------------------------------------------
//
//	// При minPts = 1 DBSCAN вырождается в поиск компонент связности графа,
//	// где ребра проведены между точками с расстоянием < eps
//
//	for (int i = 0; i < amountOfPeaks; i++) {
//
//		// Если точка уже имеет метку (посещена), пропускаем её
//		if (helpfulArray[startDataIndex + i] != 0)
//			continue;
//
//		// Начинаем новый кластер
//		++cluster_count;
//		int current_cluster_id = cluster_count;
//
//		// --- Инициализация стека для обхода в глубину (DFS) ---
//		int stack_size = 0;
//
//		// Push: добавляем стартовую точку i в стек
//		// Если есть выделенный буфер — пишем туда, иначе используем локальный массив (для малых N)
//		// В данном примере предполагаем, что stackBuffer доступен
//		if (hasStackBuffer) {
//			stackBuffer[stack_size++] = i;
//		}
//		else {
//			// Fallback: если буфера нет, кладем индекс прямо в метку (хак) 
//			// и потом восстанавливаем. Но лучше просто выделить память.
//			// Для простоты здесь просто пропускаем стек и делаем рекурсивный вызов 
//			// (не рекомендуется для GPU) или просто помечаем точку.
//			// В исправленном коде мы требуем наличия буфера.
//			// Здесь для надежности просто помечаем текущую и переходим к следующей, 
//			// если нет места под стек (упрощение для примера).
//			// Но правильнее:
//			// Вернуть ошибку или требовать достаточный sizeOfHelpfulArray.
//		}
//
//		helpfulArray[startDataIndex + i] = current_cluster_id;
//
//		// --- Обход графа (DFS) ---
//		while (stack_size > 0) {
//			// Pop: извлекаем индекс из стека
//			int curr_idx = hasStackBuffer ? stackBuffer[--stack_size] : -1;
//
//			// Защитная проверка
//			if (curr_idx < 0 || curr_idx >= amountOfPeaks) continue;
//
//			// Перебор всех возможных соседей
//			// ИСПРАВЛЕНО: цикл до amountOfPeaks (ранее было amountOfPeaks - 1)
//			for (int k = 0; k < amountOfPeaks; k++) {
//
//				// Не сравниваем точку саму с собой
//				if (k == curr_idx) continue;
//
//				// Если точка уже посещена (имеет метку), пропускаем
//				// Это критически важно, чтобы не зациклиться
//				if (helpfulArray[startDataIndex + k] != 0)
//					continue;
//
//				// Вычисление расстояния
//				// Функция distance должна быть определена как __device__ __host__
//				numb dist = distance(
//					data[startDataIndex + curr_idx],
//					intervals[startDataIndex + curr_idx],
//					data[startDataIndex + k],
//					intervals[startDataIndex + k]
//				);
//
//				// Проверка условия соседства
//				if (dist < eps) {
//					// Помечаем соседа как принадлежащий к текущему кластеру
//					helpfulArray[startDataIndex + k] = current_cluster_id;
//
//					// Добавляем соседа в стек для дальнейшего обхода его соседей
//					// ИСПРАВЛЕНО: пишем по индексу stack_size, а не по индексу k!
//					if (hasStackBuffer && stack_size < amountOfPeaks) {
//						stackBuffer[stack_size++] = k;
//					}
//				}
//			}
//		}
//	}
//
//	// ------------------------------------------------------------
//	// --- Завершение ---
//	// ------------------------------------------------------------
//
//	// Возвращаем количество найденных кластеров
//	// Если кластеров нет, вернется 0.
//
//	// Если outData передан, можно записать туда подробную статистику
//	// if (outData != nullptr) *outData = cluster_count;
//
//	return cluster_count;
//}


// ---------------------------------
// --- Глобальная функция DBSCAN ---
// ---------------------------------

__global__ void dbscanCUDA(numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, numb* intervals, numb* helpfulArray,
	const numb eps, int* outData)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	if (amountOfPeaks[idx] == -1 )
	{
		outData[idx] = -1;
		return;
	}

	if (amountOfPeaks[idx] == 0)
	{
		outData[idx] = 0;
		return;
	}

	// --- Применяем алгоритм dbscan к каждой системе
	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
}

// -------------------------------------------------------------------------------- -
// --- Оптимизированная версия DBSCAN (Spatial Hashing + Stack DFS)
// --- Размеры массивов жестко привязаны к max_amount_of_peaks из configCUDA.h
// ---------------------------------------------------------------------------------
__device__ int dbscan_optimized(
	const numb * __restrict__ data,       // Амплитуды пиков (только чтение)
	const numb * __restrict__ intervals,  // Межпиковые интервалы (только чтение)
	const int amountOfPeaks,
	const numb eps,
	int* __restrict__ labels)            // Выход: 0=не посещен, >0=ID кластера
{
	if (amountOfPeaks <= 0) return 0;
	if (amountOfPeaks == 1) { labels[0] = 1; return 1; }

	// 1. Защита и инициализация
	const int N = min(amountOfPeaks, max_amount_of_peaks);
	const numb invEps = 1.0 / eps;
	const numb eps2 = eps * eps;
	for (int i = 0; i < N; ++i) labels[i] = 0;

	// 2. Bounding Box (масштабирование на лету)
	numb minX = data[0] * mult_peak, maxX = minX;
	numb minY = intervals[0] * mult_interval, maxY = minY;
	for (int i = 1; i < N; ++i) {
		numb x = data[i] * mult_peak;
		numb y = intervals[i] * mult_interval;
		if (x < minX) minX = x; else if (x > maxX) maxX = x;
		if (y < minY) minY = y; else if (y > maxY) maxY = y;
	}

	// 3. Сетка (Grid) 16x16 = 256 ячеек (1 КБ)
	// Это оптимально для sm_75: не перегружает Local Memory
	constexpr int G_DIM = 16;
	int gCols = max(1, min(G_DIM, (int)((maxX - minX) * invEps) + 1));
	int gRows = max(1, min(G_DIM, (int)((maxY - minY) * invEps) + 1));
	int gCells = gCols * gRows;

	int head[256];
	for (int i = 0; i < gCells; ++i) head[i] = -1;

	// 4. Связный список (4 КБ)
	int next[max_amount_of_peaks];
	for (int i = 0; i < N; ++i) {
		numb x = data[i] * mult_peak;
		numb y = intervals[i] * mult_interval;
		int cx = min(gCols - 1, (int)((x - minX) * invEps));
		int cy = min(gRows - 1, (int)((y - minY) * invEps));
		int cell = cy * gCols + cx;
		next[i] = head[cell];
		head[cell] = i;
	}

	// 5. DFS Кластеризация
	// Стек ограничен 128 элементами (0.5 КБ). 
	// Это предотвращает переполнение Local Memory на старых архитектурах.
	int stack[128];
	int stackTop = 0;
	int clusterCount = 0;

	for (int i = 0; i < N; ++i) {
		if (labels[i] != 0) continue;

		++clusterCount;
		labels[i] = clusterCount;
		stackTop = 0;
		stack[stackTop++] = i;

		while (stackTop > 0) {
			int curr = stack[--stackTop];
			numb xc = data[curr] * mult_peak;
			numb yc = intervals[curr] * mult_interval;

			int cx = min(gCols - 1, (int)((xc - minX) * invEps));
			int cy = min(gRows - 1, (int)((yc - minY) * invEps));

			int sCx = max(0, cx - 1), eCx = min(gCols - 1, cx + 1);
			int sCy = max(0, cy - 1), eCy = min(gRows - 1, cy + 1);

			// Поиск в 9 соседних ячейках
			for (int cy2 = sCy; cy2 <= eCy; ++cy2) {
				for (int cx2 = sCx; cx2 <= eCx; ++cx2) {
					int nb_idx = head[cy2 * gCols + cx2];
					while (nb_idx != -1) {
						if (labels[nb_idx] == 0) {
							numb dx = data[nb_idx] * mult_peak - xc;
							numb dy = intervals[nb_idx] * mult_interval - yc;
							// Квадрат евклидова расстояния
							if (dx * dx + dy * dy <= eps2) {
								labels[nb_idx] = clusterCount;
								// Добавляем в стек, только если есть место
								if (stackTop < 128) stack[stackTop++] = nb_idx;
							}
						}
						nb_idx = next[nb_idx];
					}
				}
			}
		}
	}
	return clusterCount;
}

// ---------------------------------------------------------------------------------
// --- Оптимизированное ядро DBSCAN
// ---------------------------------------------------------------------------------
__global__ void dbscanCUDA_optimized(
	const numb* __restrict__ data,
	const size_t sizeOfBlock,
	const int amountOfBlocks,
	const int* __restrict__ amountOfPeaks,
	const numb* __restrict__ intervals,
	const numb eps,
	int* __restrict__ outData)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks) return;

	if (amountOfPeaks[idx] == -1) { outData[idx] = -1; return; }
	if (amountOfPeaks[idx] == 0) { outData[idx] = 0;  return; }

	// Локальный массив меток (выделяется на поток, размер берется из configCUDA.h)
	int labels[max_amount_of_peaks];

	outData[idx] = dbscan_optimized(
		data + idx * sizeOfBlock,
		intervals + idx * sizeOfBlock,
		amountOfPeaks[idx],
		eps,
		labels
	);
}

// --------------------
// --- Ядро для LLE ---
// --------------------
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
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			resultArray)
{
	extern __shared__ numb s[];
	numb* x = s + threadIdx.x * amountOfInitialConditions;
	numb* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//if (par_or_var) {
	//	for (int i = 0; i < dimension; ++i)
	//		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	//}
	//else {
	//	for (int i = 0; i < dimension; ++i)
	//		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	//}

	if (par_or_var == 1) {
		for (int i = 0; i < dimension; ++i)
			localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	}
	else if (par_or_var == 0) {
		for (int i = 0; i < dimension; ++i)
			x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	}
	else if (par_or_var == 2) {
		x[indicesOfMutVars[0]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
		localValues[indicesOfMutVars[1]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}

	int flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	if (flag == 0) {
		resultArray[idx] = 999;
		return;
	}

	if (flag == -1) {
		resultArray[idx] = -999;
		return;
	}

	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0;
	}

	numb zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		curand_init(seed + i, 0, 0, &state);
		z[i] = curand_uniform(&state) - 0.5;
		//z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		//bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		//if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		//flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		//if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }
		flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

		if (flag == 0) {
			resultArray[idx] = 999;
			return;
		}

		flag = loopCalculateDiscreteModel_int(y, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

		if (flag == 0) {
			resultArray[idx] = 999;
			return;
		}

		numb tempData = 0;
		numb tempData2 = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l) {
			tempData2 = (numb)1.0 / eps;
			tempData2 = tempData2 * (x[l] - y[l]);
			tempData += tempData2 * tempData2;

			//tempData2 =  (x[l] - y[l])* ((numb)1.0 / eps);
			//tempData += tempData2 * tempData2;
		}

		tempData = sqrt(tempData);

		if (tempData <= 1e-14) {
			//resultArray[idx] = -99999999;
			//return;
			tempData = 1e-14;
		}

		result += log(tempData);
		
		//if (tempData != 0)
		//	tempData = (1 / tempData);
		//else
		//	tempData = eps;
		tempData = (1 / tempData);
		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (numb)(x[j] - ((x[j] - y[j]  + 1e-14) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



// -------------------------
// --- Ядро для LLE (IC) ---
// -------------------------
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
	const int		preScaller,
	const int		writableVar,
	const numb	maxValue,
	numb*			resultArray)
{
	extern __shared__ numb s[];
	numb* x = s + threadIdx.x * amountOfInitialConditions;
	numb* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	numb* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	numb zPower = 0;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0;
	}


	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		curand_init(seed + i, 0, 0, &state);
		z[i] = curand_uniform(&state) - 0.5;
		//z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel_int(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		numb tempData = 0;
		numb tempData2 = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l) {
			tempData2 = 1 / eps;
			tempData2 = tempData2 * (x[l] - y[l]);
			tempData += tempData2 * tempData2;
		}

		tempData = sqrt(tempData);

		//for (int l = 0; l < amountOfInitialConditions; ++l)
		//	tempData += (x[l] - y[l]) * (x[l] - y[l]);
		//tempData = sqrt(tempData) / eps;

		result += log(tempData);

		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (numb)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



//find projection operation (ab)
__device__ __host__ void projectionOperator(numb* a, numb* b, numb* minuend, int amountOfValues)
{
	numb numerator = 0;
	numb denominator = 0;
	for (int i = 0; i < amountOfValues; ++i)
	{
		numerator += a[i] * b[i];
		denominator += b[i] * b[i];
	}

	numb fraction = denominator == 0 ? 0 : numerator / denominator;

	for (int i = 0; i < amountOfValues; ++i)
		minuend[i] -= fraction * b[i];
}



__device__ __host__ void gramSchmidtProcess(numb* a, numb* b, int amountOfVectorsAndValuesInVector, numb* denominators=nullptr/*They are is equale for our task*/)
{
	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[j + i * amountOfVectorsAndValuesInVector] = a[j + i * amountOfVectorsAndValuesInVector];

		for (int j = 0; j < i; ++j)
			projectionOperator(a + i * amountOfVectorsAndValuesInVector,
				b + j * amountOfVectorsAndValuesInVector,
				b + i * amountOfVectorsAndValuesInVector,
				amountOfVectorsAndValuesInVector);
	}

	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		numb denominator = 0;
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			denominator += b[i * amountOfVectorsAndValuesInVector + j] * b[i * amountOfVectorsAndValuesInVector + j];
		denominator = sqrt(denominator);
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[i * amountOfVectorsAndValuesInVector + j] = denominator == 0 ? 0 : b[i * amountOfVectorsAndValuesInVector + j] / denominator;

		if (denominators != nullptr)
			denominators[i] = denominator;
	}
}



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
	const int preScaller,
	const int writableVar,
	const numb maxValue,
	numb* resultArray)
{
	extern __shared__ numb s[];

	unsigned long long buferForMem = 0;
	numb* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	numb* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	if (par_or_var == 1) {
		for (int i = 0; i < dimension; ++i)
			localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	}
	else if (par_or_var == 0) {
		for (int i = 0; i < dimension; ++i)
			x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
	}
	else if (par_or_var == 2) {
			x[indicesOfMutVars[0]]			 = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0);
			localValues[indicesOfMutVars[1]] = getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[2], ranges[3], 1);
	}


	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		numb zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			curand_init(seed + j * amountOfInitialConditions + i, 0, 0, &state);
			z[j * amountOfInitialConditions + i] = curand_uniform(&state) - 0.5;
			//z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}

	int flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	if (flag == 0) {
		for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
		return;
	}

	//Calculating


	gramSchmidtProcess(z, y, amountOfInitialConditions);


	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (flag == 0) {
			for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
			return;
		}
		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel_int(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints, amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (flag == 0) {
				for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 999;
				return;
			}
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		// 
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (numb)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}

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
	const int preScaller,
	const int writableVar,
	const numb maxValue,
	numb* resultArray)
{
	extern __shared__ numb s[];

	unsigned long long buferForMem = 0;
	numb* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	numb* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	numb* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	numb* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);



	size_t seed = idx;
	curandState_t state;

	curand_init(seed, 0, 0, &state);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		numb zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			curand_init(seed + j * amountOfInitialConditions + i, 0, 0, &state);
			z[j * amountOfInitialConditions + i] = curand_uniform(&state) - 0.5;
			//z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}


	loopCalculateDiscreteModel_int(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating


	gramSchmidtProcess(z, y, amountOfInitialConditions);


	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//numb result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel_int(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error;*/ }

		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel_int(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints,
				amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error; */ }
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		// 
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (numb)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}

// ------------------------------------------------------------------
// --- Нахождение среднего значения пиков и межпиковых интервалов ---
// ------------------------------------------------------------------

__global__ void avgPeakFinderCUDA_logMaximas(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// 1 - stability, -1 - fixed point, 0 - unbound solution
	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock-1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}



	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	//if (outAvgPeaks[idx] == -1)
	//{
	//	outAvgPeaks[idx] = NAN;
	//	AvgTimeOfPeaks[idx] = NAN;
	//	return;
	//}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	//__device__ __host__ int peakFinder(numb* data, const int startDataIndex,
	//	const int amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h)

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0) //No peaks
	{
		outAvgPeaks[idx] = -999;
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += (outPeaks[idx * sizeOfBlock + i]);
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}

	AvgTimeOfPeaks[idx] = AvgTimeOfPeaks[idx] * mult_interval;
	outAvgPeaks[idx] = outAvgPeaks[idx] * mult_peak;

	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	if (outAvgPeaks[idx] > 0)
		outAvgPeaks[idx] = log10(outAvgPeaks[idx] / amountOfPeaks);
	else if (outAvgPeaks[idx] < 0)
		outAvgPeaks[idx] = -log10(abs(outAvgPeaks[idx]) / amountOfPeaks);
	else
		outAvgPeaks[idx] = 0;

	return;
}

__global__ void avgPeakFinderCUDA(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* systemCheker, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// 1 - stability, -1 - fixed point, 0 - unbound solution
	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock - 1];
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	//if (outAvgPeaks[idx] == -1)
	//{
	//	outAvgPeaks[idx] = NAN;
	//	AvgTimeOfPeaks[idx] = NAN;
	//	return;
	//}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	//__device__ __host__ int peakFinder(numb* data, const int startDataIndex,
	//	const int amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h)

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);


	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}

	if (amountOfPeaks <= 0) //No peaks
	{
		outAvgPeaks[idx] = -999;
		AvgTimeOfPeaks[idx] = -1.0;
		return;
	}

	AvgTimeOfPeaks[idx] = AvgTimeOfPeaks[idx] * mult_avg_interval;
	outAvgPeaks[idx] = outAvgPeaks[idx] * mult_avg_peak;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	if (lin_or_log == 1)
		outAvgPeaks[idx] /= amountOfPeaks;
	else {
		if (outAvgPeaks[idx] > 0)
			outAvgPeaks[idx] = log10(outAvgPeaks[idx] / amountOfPeaks);
		else if (outAvgPeaks[idx] < 0)
			outAvgPeaks[idx] = -log10(abs(outAvgPeaks[idx]) / amountOfPeaks);
		else
			outAvgPeaks[idx] = 0;
	}

	return;
}

__global__ void avgPeakFinderCUDA_for2Dbif(numb* data, const int sizeOfBlock, const int amountOfBlocks,
	numb* outAvgPeaks, numb* AvgTimeOfPeaks, numb* outPeaks, numb* timeOfPeaks, int* PeaksAmount, int* systemCheker, numb h)
{
	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// 1 - stability, -1 - fixed point, 0 - unbound solution
	if (systemCheker[idx] == 0) // unbound solution
	{
		outAvgPeaks[idx] = 999;
		AvgTimeOfPeaks[idx] = 999;
		PeaksAmount[idx] = 0;
		return;
	}

	if (systemCheker[idx] == -1) //fixed point
	{
		outAvgPeaks[idx] = data[idx * sizeOfBlock + sizeOfBlock - 1];
		AvgTimeOfPeaks[idx] = -1.0;
		PeaksAmount[idx] = 0;
		return;
	}

	// --- Если на предыдущих этапах систему уже отметили как "непригодную", то пропускаем ее ---
	//if (outAvgPeaks[idx] == -1)
	//{
	//	outAvgPeaks[idx] = NAN;
	//	AvgTimeOfPeaks[idx] = NAN;
	//	return;
	//}

	outAvgPeaks[idx] = 0;
	AvgTimeOfPeaks[idx] = 0;

	//__device__ __host__ int peakFinder(numb* data, const int startDataIndex,
	//	const int amountOfPoints, numb* outPeaks, numb* timeOfPeaks, numb h)

	int amountOfPeaks = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h);

	if (amountOfPeaks <= 0)
	{
		outAvgPeaks[idx] = 1000;
		AvgTimeOfPeaks[idx] = 1000;
		PeaksAmount[idx] = 0;
		return;
	}

	for (int i = 0; i < amountOfPeaks; ++i)
	{
		outAvgPeaks[idx] += outPeaks[idx * sizeOfBlock + i];
		AvgTimeOfPeaks[idx] += timeOfPeaks[idx * sizeOfBlock + i];
	}
	PeaksAmount[idx] = amountOfPeaks;
	outAvgPeaks[idx] /= amountOfPeaks;
	AvgTimeOfPeaks[idx] /= amountOfPeaks;

	return;
}

__global__ void CUDA_dbscan_kernel(numb* data, numb* intervals, int* labels,
	const int amountOfData, const numb eps, int amountOfClusters,
	int* amountOfNeighbors, int* neighbors, int idxCurPoint, int* helpfulArray)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	//if (idx == 0 && labels[idxCurPoint] == 0)					// Если у idxCurPoint точки нет еще кластера - выдаём его (обычно для первой точки в кластере)
	//	labels[idxCurPoint] = atomicAdd(amountOfClusters, 1);

	labels[idxCurPoint] = amountOfClusters;

	if (labels[idx] != 0)									// У точки уже есть кластер - выпиливаемся из потока
		return;

	if (idx == idxCurPoint)									// Если рассматриваем текущую точку - выпиливаемся из потока
		return;

	// Если расстояние между рассматрвиаемой точкой idx и текущей точкой idxCurPoint <= eps - выдаём точке кластер

	if (helpfulArray[idxCurPoint] == 0) {
		labels[idxCurPoint] = 0;
		return;
	}

	if (sqrt((data[idxCurPoint] - data[idx]) * (data[idxCurPoint] - data[idx]) + (intervals[idxCurPoint] - intervals[idx]) * (intervals[idxCurPoint] - intervals[idx])) <= eps)
	{
		labels[idx] = labels[idxCurPoint];						// Даем точке кластер. Предполагаем, что у idxCurPoint не может не быть кластера
		neighbors[atomicAdd(amountOfNeighbors, 1)] = idx;		// Фиксируем индекс найденного соседа - его ждет та же участь
	}
}



__global__ void CUDA_dbscan_search_clear_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (labels[idx] == 0 && helpfulArray[idx] == 1)
	{
		*res = idx;
		return;
	}
}



__global__ void CUDA_dbscan_search_fixed_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (helpfulArray[idx] == -1 && labels[idx] == 0)
	{
		*res = idx;
		return;
	}
}

__global__ void CUDA_dbscan_search_unbound_points_kernel(numb* data, numb* intervals, int* helpfulArray, int* labels,
	const int amountOfData, int* res)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;		// Вычисляем текущий индекс потока
	if (idx >= amountOfData)								// Если индекс больше - выпиливаемся из потока
		return;

	if (helpfulArray[idx] == 0 && labels[idx] == 0)
	{
		*res = idx;
		return;
	}
}


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
	const int		amountOfNTPoints,
	const numb		maxValue,
	numb*			timedomain,
	numb*			output,
	const int		preScaller)
{

	// --- Вычисляем индекс потока, в котором находимся в даный момент ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Если существует поток с большим индексом, чем требуется - сразу завершаем его
		return;

	// --- Прогоняем систему amountOfPointsForSkip раз ( для отработки transientTime ) --- 
		//numb Xs[3] = { initialConditions[0] , initialConditions[1], initialConditions[2] };


	output[idx] = loopCalculateDiscreteModelForFastSynchro(
		initialConditionsSlave,//numb* Xs,
		values,//const numb* values,
		h, //const numb h,
		k_forward,//const numb* K_Forward,
		k_backward,//const numb* K_Backward,
		iterOfSynchr,//const numb iterOfSynchr,
		amountOfNTPoints,//const int amountOfIterations,
		amountOfInitialConditions,//const int amountOfX,
		maxValue,//const numb maxValue,
		timedomain,//numb* timedomain,
		amountOfInitialConditions * idx * preScaller//const int startDataIndex
	);
	return;
}

__device__ numb loopCalculateDiscreteModelForFastSynchro(
	const numb* initConditionsSlave,
	const numb* values,
	const numb h,
	const numb* K_Forward,
	const numb* K_Backward,
	const numb iterOfSynchr,
	const int amountOfIterations,
	const int amountOfX,
	const numb maxValue,
	numb* timedomain,
	const int startDataIndex)
{
	//numb* norm_error = new numb[(amountOfIterations - 0)];
	//numb* Xm = new numb[amountOfX];
	//numb* X_prev = new numb[amountOfX];
	//numb* Xs = new numb[amountOfX];
	//numb* K_local = new numb[amountOfX];
	numb Xm[AMOUNTOFX];
	numb X_prev[AMOUNTOFX];
	numb Xs[AMOUNTOFX];
	numb K_local[AMOUNTOFX];

	numb rms_error = 0;
	//numb err0 = 0;
	//numb err1 = 0;


	for (int j = 0; j < amountOfX; ++j) {
		if (type_of_synch == 1) // bidirectional sycnhro
			Xm[j] = timedomain[startDataIndex + j];

		Xs[j] = initConditionsSlave[j];
		//Xs[j] = timedomain[startDataIndex + j] + 0.01;
	}



	for (int m = 0; m < iterOfSynchr; ++m) {

		for (int j = 0; j < amountOfX; j++)
			K_local[j] = K_Forward[j];

		// --- Глобальный цикл, который производит вычисления заданные amountOfIterations раз ---
		for (int i = 0; i < amountOfIterations - 1; ++i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
			}

			if (error_estim == 0) {
				if (m == iterOfSynchr - 1) {
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 1) {			
				if (i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 2) {
				if (m == iterOfSynchr - 1 && i == amountOfIterations - 2) {
					rms_error = 0;
					for (int j = 0; j < amountOfX; ++j)
						rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				}
			}
			else if (error_estim == 3) {
				rms_error = 0;
				for (int j = 0; j < amountOfX; ++j)
					rms_error = rms_error + (Xm[j] - Xs[j]) * (Xm[j] - Xs[j]);
				rms_error = (sqrt(rms_error));
				if (rms_error <= FS_error_trs) {
					return (numb)(i + 1)*h;
				}

			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 1);

			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 1);
			}

		}

		if (error_estim == 1) {
			rms_error = (sqrt(rms_error));
			if ( rms_error <= FS_error_trs ) {
				//delete[] Xm;
				//delete[] Xs;
				//delete[] X_prev;
				//delete[] K_local;
				//delete[] norm_error;

				return (numb)(m + 1);
			}
			else
				rms_error = 0;
		}


		for (int j = 0; j < amountOfX; ++j)
			K_local[j] = K_Backward[j];

		for (int i = amountOfIterations - 1; i > 0; --i)
		{
			if (type_of_synch == 0) {
				for (int j = 0; j < amountOfX; ++j)
					Xm[j] = timedomain[startDataIndex + i * amountOfX + j];
			}

			if (type_of_synch == 1) { // bidirectional sycnhro
				for (int j = 0; j < amountOfX; ++j) {
					X_prev[j] = Xs[j];
				}
			}

			calculateDiscreteModelforFastSynchro(Xs, Xm, K_local, values, h, 0);

			if (type_of_synch == 1) { // bidirectional sycnhro
				calculateDiscreteModelforFastSynchro(Xm, X_prev, K_local, values, h, 0);
			}

		}
	}


	//for (int j = 0; j < amountOfIterations - 1; ++j) {
	//	rms_error = rms_error + norm_error[j];
	//}

	if (error_estim == 0)
		rms_error = sqrt(rms_error / (numb)(amountOfIterations - 1));

	if (error_estim == 2)
		rms_error = sqrt(rms_error);
	
	if (rms_error <= FS_error_trs)
		rms_error = log10(FS_error_trs);
	else
		rms_error = log10(rms_error);

	if (isinf(rms_error) || isnan(rms_error) || rms_error >= 3)
		rms_error = 3;

	//delete[] Xm;
	//delete[] Xs;
	//delete[] X_prev;
	//delete[] K_local;
	//delete[] norm_error;

	if (error_estim == 0)
		return rms_error;
	if (error_estim == 1)
		return (numb)iterOfSynchr;
	if (error_estim == 2)
		return rms_error;
	if (error_estim == 3)
		return (numb)amountOfIterations*h;
}