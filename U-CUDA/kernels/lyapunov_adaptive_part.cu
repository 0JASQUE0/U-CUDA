// lyapunov_adaptive_part.cu — старший показатель (LLE) и спектр Ляпунова (LS) с
// адаптивным шагом.
//
// Не самостоятельный шаблон: движок приклеивает его после adaptive_part.cu к
// bifurcation2d.template.cu с UCUDA_AD_NO_SWEEP_KERNELS (ядра пиков не нужны) и
// UCUDA_AD_NO_DENSE (плотный выход не нужен: перенормировка только в узлах шага).
//
// Алгоритм — тот же, что у LLEKernelCUDA / LSKernelCUDA (Wolf / Benettin: клоны в
// eps-окрестности базовой траектории, для LS — с ортогонализацией Грама-Шмидта), и
// раскладка результата та же (NC чисел на точку, 999 — разлёт). Клоны делают ТОТ ЖЕ шаг,
// что базовая траектория x, тем же вложенным методом (с FSAL), и регулятор судит не
// только по ошибке x, но и по ошибке возмущений delta = y - x: её оценка — разность
// оценок E клона и x, масштаб — rtol * |delta| (RMS). Без этого у устойчивого равновесия
// ошибка x почти нулевая, шаг дорастает до границы устойчивости явного метода, и клоны
// видят численное отображение вместо потока: LLE Лоренца при r < 24.7 выходил ~0 вместо
// -0.5..-0.1. Направления начального возмущения — тем же генератором и seed, что у ядер
// постоянного шага.
//
// Перенормировка (renorm):
//   0 — ровно в T0 + k*NT: шаг, переходящий границу, обрезается по ней; после границы
//       шаг берётся из hfree — предложение после обрезанного шага занижено;
//   1 — в первом узле после T0 + k*NT: шаги не обрезаются, блоки чуть длиннее NT
//       (последний — ровно до T0 + nBlocks*NT).
// Итог, как у постоянного шага, — сумма логарифмов, делённая на проинтегрированное время
// nBlocks*NT (не на tMax: при tMax, не кратном NT, блоков меньше).
//
// Время сборки: транзиент, блоки и перенормировка — в одном цикле, чтобы в ядре была
// одна копия шага (см. шапку adaptive_part.cu).

// Начальные возмущения: LLE — случайное направление, LS — NC случайных векторов,
// ортонормированных Грамом-Шмидтом; y = x + eps * направление.
// Генератор начальных направлений — побитовая копия заглушки curand из шаблонов LLE/LS
// (lle1d/ls1d/lle2d/ls2d.template.cu: splitmix по seed и номеру точки, затем LCG), под
// которыми NVRTC собирает LLEKernelCUDA / LSKernelCUDA. Свой, потому что этот модуль собран
// на bifurcation2d.template.cu, а его заглушка номер подпоследовательности игнорирует: все
// точки свипа получали ОДНУ и ту же начальную рамку. Результат при конечном T зависит от
// рамки (~ln(1/c)/T), и общая рамка давала гладкую, но сдвинутую одинаково для всех точек
// кривую, которая не совпадала с постоянным шагом даже при h0 = h_min = h_max.
struct UcudaLyapRng { unsigned long long s; };
__device__ __forceinline__ void ucudaLyapRngInit(unsigned long long seed, unsigned long long sequence, UcudaLyapRng& r)
{
	unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	r.s = z ^ (z >> 31);
}
__device__ __forceinline__ float ucudaLyapRngUniform(UcudaLyapRng& r)
{
	r.s = r.s * 6364136223846793005ULL + 1442695040888963407ULL;
	return (float)(((r.s >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}

template <int NC>
__device__ __forceinline__ void ucudaLyapInitClones(const numb* x, const numb eps, const int idx, numb* y, numb* z)
{
	constexpr int n = AMOUNTOFX;
	const unsigned long long ppSeed = 1234567891ULL;   // как у LLEKernelCUDA / LSKernelCUDA
	UcudaLyapRng state;
	ucudaLyapRngInit(ppSeed, (unsigned long long)idx, state);
	for (int j = 0; j < NC; ++j) {
		numb zPower = 0;
		for (int i = 0; i < n; ++i) {
			z[j * n + i] = ucudaLyapRngUniform(state) - 0.5;
			zPower += z[j * n + i] * z[j * n + i];
		}
		zPower = sqrt(zPower);
		for (int i = 0; i < n; ++i) z[j * n + i] /= zPower;
	}
	if constexpr (NC == 1) {
		for (int i = 0; i < n; ++i) y[i] = z[i] * eps + x[i];
	} else {
		gramSchmidtProcess(z, y, n);
		for (int j = 0; j < NC; ++j)
			for (int i = 0; i < n; ++i) y[j * n + i] = y[j * n + i] * eps + x[i];
	}
}

// Перенормировка: накопить логарифмы растяжения и вернуть клоны на расстояние eps.
// Формулы — ровно как в LLEKernelCUDA / LSKernelCUDA.
template <int NC>
__device__ __forceinline__ void ucudaLyapRenorm(const numb* x, const numb eps, numb* y, numb* z, numb* acc)
{
	constexpr int n = AMOUNTOFX;
	if constexpr (NC == 1) {
		numb d = 0;
		for (int l = 0; l < n; ++l) {
			const numb t = ((numb)1.0 / eps) * (x[l] - y[l]);
			d += t * t;
		}
		d = sqrt(d);
		if (d <= 1e-14) d = 1e-14;
		acc[0] += log(d);
		const numb inv = 1 / d;
		for (int j = 0; j < n; ++j) y[j] = (numb)(x[j] - ((x[j] - y[j] + 1e-14) * inv));
	} else {
		numb den[NC];
		for (int k = 0; k < NC; ++k)
			for (int l = 0; l < n; ++l) y[k * n + l] -= x[l];
		gramSchmidtProcess(y, z, n, den);
		for (int k = 0; k < NC; ++k) {
			acc[k] += log(den[k] / eps);
			for (int j = 0; j < n; ++j) y[k * n + j] = (numb)(x[j] + z[k * n + j] * eps);
		}
	}
}

// Клоны как добавка к шагу (см. UcudaAdNoExtra в ucuda_adaptive.cuh): на попытке — вложенный
// шаг каждого клона и норма ошибки возмущения, на принятии — клоны становятся новыми.
template <int NC>
struct UcudaLyapClones {
	numb y[NC * AMOUNTOFX], F[NC * AMOUNTOFX];     // клоны и f(клонов)
	numb Yc[NC * AMOUNTOFX], Fc[NC * AMOUNTOFX];   // попытка
	numb Ec[UCUDA_AD_MAXLOW * AMOUNTOFX];
	numb rtol;
	int  nlo;
	bool active;                                   // false — транзиент, клонов ещё нет

	template <class K>
	UCUDA_HD numb attempt(const K& k, const numb* a, const numb* X, const numb* Y, const numb* E, numb h)
	{
		if (!active) return 0;
		constexpr int n = AMOUNTOFX;
		numb W[UCUDA_AD_MAXSTAGES * AMOUNTOFX];
		numb worst = 0;
		for (int c = 0; c < NC; ++c) {
			k.emb(y + c * n, F + c * n, a, h, Yc + c * n, Ec, Fc + c * n, W);
			numb d0 = 0, d1 = 0, s5 = 0, s3 = 0;
			for (int i = 0; i < n; ++i) {
				const numb p0 = y[c * n + i] - X[i], p1 = Yc[c * n + i] - Y[i];
				d0 += p0 * p0; d1 += p1 * p1;
				const numb e5 = Ec[i] - E[i];
				s5 += e5 * e5;
				if (nlo >= 2) { const numb e3 = Ec[n + i] - E[n + i]; s3 += e3 * e3; }
			}
			// sc^2 = (rtol * RMS(delta))^2, delta — большее из начала и конца попытки.
			const numb sc2 = rtol * rtol * (d0 > d1 ? d0 : d1) / (numb)n;
			if (!(sc2 > 0)) continue;
			s5 /= sc2; s3 /= sc2;
			numb err;
			if (nlo >= 2) err = (s5 == 0 && s3 == 0) ? (numb)0 : s5 / sqrt((s5 + (numb)0.01 * s3) * (numb)n);
			else          err = sqrt(s5 / (numb)n);
			if (err != err) return err;
			if (err > worst) worst = err;
		}
		return worst;
	}
	UCUDA_HD void commit()
	{
		if (!active) return;
		for (int i = 0; i < NC * AMOUNTOFX; ++i) { y[i] = Yc[i]; F[i] = Fc[i]; }
	}
};

// Одна точка свипа: транзиент по x, затем nBlocks блоков NT с клонами. res[NC] — показатели.
// Возвращает 1, 0 — разлёт (x или клон ушёл за maxValue / в NaN) или отмена.
template <int NC>
__device__ __forceinline__ int ucudaLyapAdPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb tTr, const numb tMax, const numb NT, const int nBlocks, const int nWarm,
	const numb eps, const int renorm, const numb maxValue, const int idx, numb* res,
	const volatile int* cancelFlag, UcudaAdProgress& prog)
{
	constexpr int n = AMOUNTOFX;
	UcudaLyapClones<NC> cl;
	cl.rtol   = P.rtol > 0 ? P.rtol : P.atol[0];   // чисто абсолютный допуск — как относительный для delta
	cl.nlo    = P.nlo;
	cl.active = false;
	numb z[NC * n], acc[NC], accWarm[NC];
	for (int c = 0; c < NC; ++c) { acc[c] = 0; accWarm[c] = 0; }
	// Транзиент касательных векторов: блоки 1..nWarm перенормируются в accWarm (не в сумму),
	// счёт времени — с конца последнего из них (tAcc0).
	const int nAll = nBlocks + nWarm;
	numb tAcc0 = tTr;
	const numb tEnd = tTr + (numb)nAll * NT;
	numb tb = tTr;            // ближайшая граница: конец транзиента, затем T0 + k*NT
	int  k = 0;               // 0 — транзиент, дальше — номер блока
	int  cnt = 0;
	for (;;) {
		if (!(S.t < tb)) {
			const bool clipped = (k == 0) || (renorm == 0);   // шаг к границе обрезался
			if (k == 0) ucudaLyapInitClones<NC>(S.X, eps, idx, cl.y, z);
			else        ucudaLyapRenorm<NC>(S.X, eps, cl.y, z, k > nWarm ? acc : accWarm);
			if (k == nWarm) tAcc0 = S.t;
			for (int c = 0; c < NC; ++c) K.rhs(cl.y + c * n, a, cl.F + c * n);   // клоны сдвинуты — f заново
			cl.active = true;
			if (clipped) S.h = S.hfree;
			++k;
			if (renorm != 0)   // шаг длиннее NT мог пройти несколько границ
				while (k <= nAll && !(S.t < tTr + (numb)k * NT)) ++k;
			if (k > nAll) break;
			tb = tTr + (numb)k * NT;
			continue;
		}
		ucuda_ad_step_x(S, K, a, P, (k == 0 || renorm == 0) ? tb : tEnd, cl);
		if (S.diverged || ucudaAdOut(S.X, maxValue)) return 0;
		if (k > 0)
			for (int c = 0; c < NC; ++c)
				if (ucudaAdOut(cl.y + c * n, maxValue)) return 0;
		prog.update(S.t);
		if ((++cnt & 63) == 0 && cancelFlag != nullptr && *cancelFlag != 0) return 0;
	}
	const numb tIntegrated = S.t - tAcc0;   // последний блок кончается ровно в tEnd при любом renorm
	for (int c = 0; c < NC; ++c) res[c] = acc[c] / tIntegrated;
	return 1;
}

// Общее тело ядер: точка свипа (система или настройка шага по осям), расчёт, запись
// NC чисел в resultArray[idx*NC ..] (999 — разлёт) и статистики шага.
template <int NC>
__device__ __forceinline__ void ucudaLyapAdKernel(
	const int nPts, const int nPtsLimiter, const size_t amountOfCalculatedPoints, const int dimension,
	const numb* ranges, const int* indicesOfMutVars, const numb* initialConditions,
	const int amountOfInitialConditions, const numb* values, const int amountOfValues,
	const numb maxValue, numb* resultArray, const int logAxisMask,
	const UcudaAdaptParams* Pbase, const int* axisKind, const numb tolRatio,
	const numb transientTime, const numb tMax, const numb NT, const int nBlocks, const int nWarm, const numb eps,
	const int renorm, const volatile int* cancelFlag, int* progressCounter, const int progressStride,
	const numb progressDt, const size_t progressUnits, numb* adStats)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + (threadIdx.x * sharedStride);
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	UcudaAdaptParams P = *Pbase;
	ucudaSetupSweepPointAd(nPts, amountOfCalculatedPoints, idx, dimension, ranges, indicesOfMutVars,
		initialConditions, values, amountOfValues, logAxisMask, axisKind, tolRatio,
		localX, localValues, P);

	const UcudaKrsFns K{};
	UcudaAdaptState S;
	ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);
	UcudaAdProgress prog;
	prog.init(progressCounter, progressStride, progressDt);

	numb res[NC];
	const int ok = ucudaLyapAdPoint<NC>(S, K, localValues, P, transientTime, tMax, NT, nBlocks, nWarm, eps, renorm,
		maxValue, idx, res, cancelFlag, prog);
	for (int m = 0; m < NC; ++m) resultArray[(size_t)idx * NC + m] = ok ? res[m] : (numb)999;
	ucudaAdWriteStats(adStats, idx, S);
	prog.top_up((int)(progressUnits / (size_t)(progressStride > 0 ? progressStride : 1)));
}

#define UCUDA_LYAP_AD_PARAMS \
	const int nPts, const int nPtsLimiter, const size_t amountOfCalculatedPoints, const int dimension, \
	const numb* __restrict__ ranges, const int* __restrict__ indicesOfMutVars, \
	const numb* __restrict__ initialConditions, const int amountOfInitialConditions, \
	const numb* __restrict__ values, const int amountOfValues, const numb maxValue, numb* resultArray, \
	const int logAxisMask, const UcudaAdaptParams* __restrict__ Pbase, const int* __restrict__ axisKind, \
	const numb tolRatio, const numb transientTime, const numb tMax, const numb NT, const int nBlocks, const int nWarm, \
	const numb eps, const int renorm, const volatile int* cancelFlag, int* progressCounter, \
	const int progressStride, const numb progressDt, const size_t progressUnits, numb* adStats
#define UCUDA_LYAP_AD_ARGS \
	nPts, nPtsLimiter, amountOfCalculatedPoints, dimension, ranges, indicesOfMutVars, initialConditions, \
	amountOfInitialConditions, values, amountOfValues, maxValue, resultArray, logAxisMask, Pbase, axisKind, \
	tolRatio, transientTime, tMax, NT, nBlocks, nWarm, eps, renorm, cancelFlag, progressCounter, progressStride, \
	progressDt, progressUnits, adStats

// LLE: один клон, одно число на точку (как LLEKernelCUDA).
__global__ void lyapunovLLEAdCUDA(UCUDA_LYAP_AD_PARAMS)
{
	ucudaLyapAdKernel<1>(UCUDA_LYAP_AD_ARGS);
}

// LS: AMOUNTOFX клонов, спектр из AMOUNTOFX чисел на точку (как LSKernelCUDA).
__global__ void lyapunovLSAdCUDA(UCUDA_LYAP_AD_PARAMS)
{
	ucudaLyapAdKernel<AMOUNTOFX>(UCUDA_LYAP_AD_ARGS);
}
