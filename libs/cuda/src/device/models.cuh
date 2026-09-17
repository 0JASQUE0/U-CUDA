#include "ucuda/cuda/cudaLibrary.cuh"
#include "ucuda/domain/configCUDA.h"

// Host-only и runtime-API заголовки недоступны при NVRTC-компиляции.
// <math.h> / <cmath> тоже прячем — device-код в NVRTC получает sin/cos/sqrt/isnan/...
// как встроенные __device__ функции CUDA, заголовок не нужен.
#ifndef __CUDACC_RTC__
#include <math.h>
#include <cmath>
#include <curand_kernel.h>
#include <iostream>
#include <cuda_runtime.h>
#endif

// Вычисляет следующее значение дискретной модели и записывает результат в x

// calculateDiscreteModel_rand зависит от curand_kernel.h, которого нет в NVRTC.
// Под NVRTC эта функция не нужна (kernel-ы для bif1d/LLE/basins её не зовут).
#ifndef __CUDACC_RTC__
__device__ void calculateDiscreteModel_rand(size_t seed, numb* X, const numb* a, const numb h)
{
	curandState_t state;
	numb X1[3], k[4][4], Im, Id, Iin;
	numb pi = 3.14159265359;
	numb u1, u2;
	int N = 3;
	int i, j;

	for (i = 0; i < N; i++) {
		X1[i] = X[i];
	}

	for (j = 0; j < 4; j++) {

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
	numb X1[3];
	numb k[3][4];
	int N = 3;
	int i, j;
	for (i = 0; i < N; i++) {
		X1[i] = X[i];
	}
	for (j = 0; j < 4; j++) {
		k[0][j] = (-X1[1] - X1[2]);
		k[1][j] = (X1[0] + a[1] * X1[1]);
		k[2][j] = (a[2] + X1[2] * (X1[0] - a[3]));

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

// Внутренняя функция: обрабатывает рекурсию по m (бинарное дерево)
// Аналог chua_multistep_extended из твоего примера
__device__ __host__ numb chua_inner(numb x, numb m, numb d, numb kslope, numb mu_val, numb M_val) {
	numb sg_prod = 1.0;    // Накопленное произведение знаков
	numb offset = 0.0;     // Накопленное смещение

	const int MAX_ITER = 100; // Ограничивает уровни внутренней матрешки
	int iter = 0;

	while (iter < MAX_ITER) {
		// Базовый случай: масштаб стал слишком маленьким
		if (d < 1e-14) {
			return sg_prod * (-x) - offset;
		}

		numb sg_current = (x < 0) ? -1.0 : 1.0;
		numb x_abs = (x < 0) ? -x : x;
		sg_prod *= sg_current;

		// Пороги из MATLAB-кода
		numb b_val = 2.0 * d * (kslope * m - m - kslope + 1.0) / M_val / (m - 1.0);
		numb thr_lin = (d * (m - 2.0) / (m - 1.0) + b_val) / kslope;
		numb thr_hor_left = (m - 2.0) / (m - 1.0) * d;
		numb thr_hor_right = m / (m - 1.0) * d;

		// Проверка зон
		if (x_abs < thr_lin) {
			// Линейный участок
			numb y = -kslope * x_abs + b_val;
			return sg_prod * y - offset;
		}
		else if (x_abs < thr_hor_left) {
			// Горизонтальный участок слева
			numb y = -(m - 2.0) / (m - 1.0) * d;
			return sg_prod * y - offset;
		}
		else if (x_abs > thr_hor_right) {
			// Горизонтальный участок справа 
			// (x_abs <= mu*d гарантированно, так как мы внутри базового интервала)
			numb y = -m / (m - 1.0) * d;
			return sg_prod * y - offset;
		}
		else {
			// Фрактальная часть с масштабированием по m: f(x) = f(x-d, d/m) - d
			offset += sg_prod * d;
			x = x_abs - d;
			d = d / m;
			++iter;
		}
	}
	// Защита от зацикливания
	return sg_prod * (-x) - offset;
}

// Внешняя функция: обрабатывает рекурсию по M и собирает итоговое выражение
// Аналог psi_m из твоего примера
__device__ __host__ numb psi_two_scale(numb x, numb m, numb M, numb d, numb kslope) {
	// Вычисляем mu и границу базового интервала [x0, x0 * M]
	numb mu_val = (m * (2.0 - 1.0 / kslope) - 2.0 * (1.0 - 1.0 / kslope)) / (m - 1.0);
	numb x0 = mu_val * d / M;

	if (x == 0) x = 1e-15;

	numb sg = (x < 0) ? -1.0 : 1.0;
	numb x_abs = (x < 0) ? -x : x;

	numb current_x0 = x0;
	numb current_A = 1.0; // Множитель масштаба
	bool flag = true;

	const int MAX_OUTER_ITER = 100; // Ограничивает уровни внешней матрешки
	int outer_iter = 0;

	// Находим нужную октаву (интервал, масштабированный по M)
	while (flag && outer_iter < MAX_OUTER_ITER) {
		numb x1 = M * current_x0;
		if (x_abs > x1) {
			// x слишком большой, увеличиваем интервал
			current_x0 = M * current_x0;
			current_A = current_A * M;
		}
		else if (x_abs > current_x0 && x_abs <= x1) {
			// Попали в нужный интервал
			flag = false;
		}
		else {
			// x слишком маленький, уменьшаем интервал
			current_x0 = current_x0 / M;
			current_A = current_A / M;
		}
		outer_iter++;
	}

	// Отображаем x_abs в базовый интервал [x0, x0 * M]
	numb x_arg = (x_abs - current_x0) / current_A + x0;

	// Вычисляем значение внутренней функции для базового интервала
	numb y_inner = chua_inner(x_arg, m, d, kslope, mu_val, M);

	// Масштабируем результат обратно (учитываем внешнее масштабирование по M)
	numb y_scaled = current_A * y_inner;

	// Возвращаем готовый член уравнения: 0.19 * (chua_twoscale(x) + x)
	numb res = 0.19 * (sg * y_scaled + x);

	return res;
}

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

