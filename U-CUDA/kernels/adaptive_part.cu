// adaptive_part.cu — адаптивный шаг в параметрических свипах (БД 1D/2D, бассейны).
//
// Не самостоятельный шаблон: движок (parametric_engine.cpp) приклеивает этот текст
// к концу базового шаблона (bifurcation2d.template.cu — после его
// #include "cudaLibrary.cu") и подставляет тела четырёх функций адаптивного шага
// (codegen_adaptive). Сами шаблоны и cudaLibrary.cu не меняются, поэтому путь с
// постоянным шагом компилируется ровно так же, как до появления адаптивного.
// (Без двойных фигурных скобок в комментариях: replace_all заменит и их.)
//
// Режимы вывода:
//   равномерная сетка — отсчёты в T0 + i*dt через плотный выход, дальше тот же
//     PeakStream, что у постоянного шага: пики, интервалы и тест неподвижной
//     точки не отличаются ничем, кроме способа получить отсчёт;
//   узлы шага — каждый dec-й принятый узел идёт в PeakStreamNU (ниже): пик —
//     смена знака производной сигнала, время — абсолютное, неподвижная точка —
//     sum|f(x)| < eps_fixed_point в конце записи.
// Модуль метрик (metrics_adaptive_part.cu) берёт отсюда драйвер и помощники, а
// ядра свипов выключает макросом UCUDA_AD_NO_SWEEP_KERNELS; ядро continuation
// собирается отдельным модулем (UCUDA_AD_CONT_KERNEL) — только когда оно нужно.
//
// Время сборки: каждый вызов ucuda_ad_step разворачивается в полную копию шага
// (стадии, норма, все регуляторы), и ptxas тратит время пропорционально числу
// копий. Поэтому транзиент не отдельный цикл, а первая цель записи (ucudaAdUniformPoint,
// ucudaAdNodesPoint): две копии шага на ядро свипа вместо трёх, одна у метрик. Цикл
// вокруг шага при этом остаётся тесным, как в ucuda_ad_advance_to: один общий цикл
// на все режимы собирался быстрее, но считал сетку на 15% медленнее, а шаг не инлайн —
// ещё медленнее. Помощники — __forceinline__: в RDC обычная __device__-функция
// компилируется ещё и отдельной внешней копией.

#define UCUDA_AD_KRS_FUNCS
__device__ __host__ __forceinline__ void ucuda_krs_rhs(const numb* X, const numb* a, numb* F) {
{{KRS_RHS_BODY}}
}
__device__ __host__ __forceinline__ void ucuda_krs_emb(const numb* X, const numb* F0, const numb* a,
    const numb h, numb* Y, numb* E, numb* F1, numb* W) {
{{KRS_EMB_BODY}}
}
__device__ __host__ __forceinline__ void ucuda_krs_dprep(const numb* X, const numb* Y, const numb* F0,
    const numb* F1, const numb* a, const numb h, numb* W, numb* D) {
{{KRS_DPREP_BODY}}
}
__device__ __host__ __forceinline__ void ucuda_krs_deval(const numb* D, const numb th, numb* Yo) {
{{KRS_DEVAL_BODY}}
}

// Пользовательский регулятор (C body из библиотеки) — между раскладкой драйвера и им самим;
// у встроенных плейсхолдер пустой.
#define UCUDA_ADAPT_LAYOUT_ONLY
#include "ucuda_adaptive.cuh"
#undef UCUDA_ADAPT_LAYOUT_ONLY
{{CTRL_CUSTOM}}
#include "ucuda_adaptive.cuh"

// Оси свипа (axisKind[i]): система или настройка шага.
#define UCUDA_AXIS_SYSTEM 0     // параметр / НУ — как у постоянного шага (par_or_var)
#define UCUDA_AXIS_RTOL   2
#define UCUDA_AXIS_ATOL   3     // atol одинаковый у всех переменных
#define UCUDA_AXIS_TOL    4     // rtol = v, atol = v * tolRatio
#define UCUDA_AXIS_CTRL   10    // + k: k-й параметр регулятора

// Наблюдаемый сигнал: та же формула, что у loopCalculateDiscreteModel_int
// (writableVar < 0 — комбинация первых переменных). Линейна, поэтому та же
// функция от f(x) даёт производную сигнала.
__device__ __forceinline__ numb ucudaAdObs(const numb* v, const int writableVar)
{
	if (writableVar < 0) {
		if constexpr (AMOUNTOFX >= 3)
			return v[0] + pi * v[1] + euler * v[2];
		else if constexpr (AMOUNTOFX == 2)
			return v[0] + pi * v[1];
		else
			return v[0];
	}
	return v[writableVar];
}

#ifdef UCUDA_AD_NODES_KERNEL
// То же значение весами: r = sum w_j v_j, w_j = 1 / 0 (одна переменная) или 1, pi, euler
// (комбинация) — та же цепочка FMA, что выше. Слагаемое с нулевым весом пропускается, а не
// умножается: 0 * inf = NaN испортил бы сигнал из-за ненаблюдаемой компоненты. Индексы — константы.
// v[writableVar] (индекс известен лишь при запуске; выбор перебором компилятор сворачивает
// обратно в него) держал бы в локальной памяти массив, на который смотрит v, — X и f(X)
// состояния шага, а с ними и всё состояние: каждый принятый шаг дописывал его туда
// (lg_throttle на реальных частотах).
__device__ __forceinline__ numb ucudaAdObsW(const numb* v, const int writableVar)
{
	numb r = 0;
	for (int j = 0; j < AMOUNTOFX; ++j) {
		const numb w = writableVar < 0
			? (j == 0 ? (numb)1 : (j == 1 ? (numb)pi : (j == 2 ? (numb)euler : (numb)0)))
			: (writableVar == j ? (numb)1 : (numb)0);
		r = (w != (numb)0) ? r + w * v[j] : r;
	}
	return r;
}
#define UCUDA_AD_OBS ucudaAdObsW
#else
#define UCUDA_AD_OBS ucudaAdObs
#endif

// Разлёт: как у циклов постоянного шага — nan/inf или sum|x| > maxValue (0 — без порога).
__device__ __forceinline__ bool ucudaAdOut(const numb* x, const numb maxValue)
{
	numb checker = 0;
	for (int j = 0; j < AMOUNTOFX; ++j) checker += fabs(x[j]);
	if (isnan(checker) || isinf(checker)) return true;
	return maxValue != 0 && checker > maxValue;
}

// Ось настройки шага (kind != UCUDA_AXIS_SYSTEM): значение v — в копию параметров P.
__device__ __forceinline__ void ucudaAdApplyStepAxis(const int kind, const numb v, const numb tolRatio,
	UcudaAdaptParams& P)
{
	if (kind == UCUDA_AXIS_RTOL) P.rtol = v;
	else if (kind == UCUDA_AXIS_ATOL) { for (int j = 0; j < AMOUNTOFX; ++j) P.atol[j] = v; }
	else if (kind == UCUDA_AXIS_TOL)  { P.rtol = v; for (int j = 0; j < AMOUNTOFX; ++j) P.atol[j] = v * tolRatio; }
	else if (kind >= UCUDA_AXIS_CTRL && kind < UCUDA_AXIS_CTRL + UCUDA_CTL_NPAR) P.c[kind - UCUDA_AXIS_CTRL] = v;
}

// Настройки шага, общие для всех нитей (ядро узлов, UCUDA_AD_NODES_KERNEL), оси свипа не
// правят: такой свип движок туда не отправляет (ad_nodes_module).
__device__ __forceinline__ void ucudaAdApplyStepAxis(const int, const numb, const numb, const UcudaAdaptParams&) {}

// Точка свипа: система — как ucudaSetupSweepPoint, оси настроек шага правят
// локальную копию параметров шага P (PT = UcudaAdaptParams); у ядра узлов P — общая
// константа (PT = const UcudaAdaptParams), и оси бывают только системные.
template <class PT>
__device__ __forceinline__ void ucudaSetupSweepPointAd(
	const int nPts, const size_t amountOfCalculatedPoints, const int idx, const int dimension,
	const numb* ranges, const int* indicesOfMutVars,
	const numb* initialConditions, const numb* values, const int amountOfValues,
	const int logAxisMask, const int* axisKind, const numb tolRatio,
	numb* localX, numb* localValues, PT& P)
{
	for (int i = 0; i < AMOUNTOFX; ++i) localX[i] = initialConditions[i];
	for (int i = 0; i < amountOfValues; ++i) localValues[i] = values[i];
	for (int i = 0; i < dimension; ++i) {
		const bool isLog = (logAxisMask >> i) & 1;
		const numb v = isLog ? getValueByIdx_log(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i)
		                     : getValueByIdx(amountOfCalculatedPoints + idx, nPts, ranges[i * 2], ranges[i * 2 + 1], i);
		const int kind = axisKind[i];
		if (kind == UCUDA_AXIS_SYSTEM) {
			// par_or_var: 1 — обе оси по параметрам, 0 — по НУ, 2 — ось 0 по НУ, ось 1 по параметру
			const bool toX = (par_or_var == 0) || (par_or_var == 2 && i == 0);
			if (toX) localX[indicesOfMutVars[i]] = v;
			else     localValues[indicesOfMutVars[i]] = v;
		}
		else ucudaAdApplyStepAxis(kind, v, tolRatio, P);
	}
}

// Прогресс адаптивных ядер: точка отчитывает свои progressUnits / stride тиков целиком, в
// конце (ucudaProgressTopUp из cudaLibrary.cu). Тиков по модельному времени внутри точки нет:
// каждый был атомиком в память хоста через PCIe, и вокруг него нити варпа ждали друг друга
// (BSYNC). Rossler DOP853: LLE 2D 128x128 0.48 -> 0.9 s, БД и метрики 2D на сетке вывода +20%.

// Поиск пиков на узлах адаптивного шага. Пик — смена знака производной сигнала
// с + на - между соседними узлами; принимается, если вершина поднимается над
// минимумом сигнала после предыдущего пика не меньше чем на eps_peak_delta и
// выше peak_threshold. Вершина — по способу interp:
//   0 — больший из двух узлов;
//   1 — парабола через три последних узла (формула постоянного шага, обобщённая
//       на неравные шаги: при h1 == h2 совпадает с ней);
//   2 — кубический Эрмит на шаге со сменой знака по значениям и точным
//       производным в узлах, вершина — корень его производной.
// Время пика абсолютное; интервалы, их фильтр eps_interPeak_delta, потолки и
// суммы — как у PeakStream.

// Вершина (максимум) на отрезке [t1, t] между узлами, где d1 > 0 >= d, способом interp
// (см. выше); (t0, x0) — узел перед ним, нужен параболе при n >= 2 (n — узлов до t).
// Общая для пиков (PeakStreamNU) и экстремумов метрик на узлах (MetricsAccumNU); минимум —
// та же функция от -x, -d.
__device__ __host__ __forceinline__ void ucudaAdNodeVertex(const int interp, const int n,
	const numb t0, const numb x0, const numb t1, const numb x1, const numb d1,
	const numb t, const numb x, const numb d, numb& pv, numb& pt)
{
	pv = x1; pt = t1;
	if (x > x1) { pv = x; pt = t; }
	if (interp == 1 && n >= 2) {
		const numb h1 = t1 - t0, h2 = t - t1;
		if (h1 > 0 && h2 > 0) {
			const numb e1 = (x1 - x0) / h1, e2 = (x - x1) / h2;
			const numb c = (e2 - e1) / (h1 + h2);
			const numb b = (e1 * h2 + e2 * h1) / (h1 + h2);
			if (c < 0) {
				numb tau = -b / (2 * c);
				if (tau < -h1) tau = -h1;
				if (tau > h2)  tau = h2;
				pv = x1 + tau * (b + c * tau);
				pt = t1 + tau;
			}
		}
	}
	else if (interp == 2) {
		const numb h = t - t1;
		if (h > 0) {
			const numb D  = x - x1;
			const numb c1 = h * d1;
			const numb c2 = 3 * D - h * (2 * d1 + d);
			const numb c3 = h * (d1 + d) - 2 * D;
			// p'(th) = c1 + 2 c2 th + 3 c3 th^2; ищем корень на [0, 1] с p'' < 0.
			const numb A = 3 * c3, B = 2 * c2, C = c1;
			numb th = -1;
			if (fabs(A) <= (numb)1e-14 * (fabs(B) + fabs(C))) {
				if (B != 0) th = -C / B;
			} else {
				const numb disc = B * B - 4 * A * C;
				if (disc >= 0) {
					const numb sq = sqrt(disc);
					const numb q  = -(numb)0.5 * (B + (B >= 0 ? sq : -sq));
					const numb r1 = q / A;
					const numb r2 = (q != 0) ? C / q : r1;
					if (r1 >= 0 && r1 <= 1 && B + 2 * A * r1 < 0)      th = r1;
					else if (r2 >= 0 && r2 <= 1 && B + 2 * A * r2 < 0) th = r2;
				}
			}
			if (th >= 0 && th <= 1) {
				pv = x1 + th * (c1 + th * (c2 + th * c3));
				pt = t1 + th * h;
			}
		}
	}
}

struct PeakStreamNU
{
	numb*  outPeaks;
	numb*  timeOfPeaks;
	size_t peakBase;
	int    capacity;
	int    interp;
	numb   t0, x0;          // узел n-2
	numb   t1, x1, d1;      // узел n-1
	int    n;
	numb   runMin;
	bool   done;
	int    raw;
	int    emitted;
	numb   anchorTime;
	bool   haveAnchor;
	numb   sumP, sumP2, sumI, sumI2;
	numb   last;

	__device__ void init(numb* peaks, numb* times, size_t base, int cap, int interpMode)
	{
		outPeaks = peaks; timeOfPeaks = times; peakBase = base; capacity = cap; interp = interpMode;
		t0 = x0 = t1 = x1 = d1 = (numb)0;
		n = 0; runMin = (numb)0; done = false; raw = 0; emitted = 0;
		anchorTime = (numb)0; haveAnchor = false;
		sumP = sumP2 = sumI = sumI2 = (numb)0;
		last = (numb)0;
	}

	__device__ void emitInterval(numb value, numb time)
	{
		if (!haveAnchor) { anchorTime = time; haveAnchor = true; return; }
		const numb delta = time - anchorTime;
		if (delta < eps_interPeak_delta) return;
		if (emitted < max_amount_of_peaks) {
			if (outPeaks != nullptr)    outPeaks[peakBase + emitted]    = value;
			if (timeOfPeaks != nullptr) timeOfPeaks[peakBase + emitted] = delta;
			sumP += value;   sumP2 += value * value;
			sumI += delta;   sumI2 += delta * delta;
		}
		++emitted;
		anchorTime = time;
	}

	// Вершина на отрезке [t1, t], где d1 > 0 >= d.
	__device__ void vertex(numb t, numb x, numb d, numb& pv, numb& pt) const
	{
		ucudaAdNodeVertex(interp, n, t0, x0, t1, x1, d1, t, x, d, pv, pt);
	}

	__device__ void pushNode(numb t, numb x, numb d)
	{
		last = x;
		if (n == 0) runMin = x;
		else if (!done) {
			if (d1 > 0 && d <= 0) {
				numb pv, pt;
				vertex(t, x, d, pv, pt);
				if (pv - runMin >= eps_peak_delta && pv > peak_threshold) {
					emitInterval(pv, pt);
					++raw;
					if (capacity > 0 && raw >= capacity) done = true;
					runMin = x;
				}
			}
			if (x < runMin) runMin = x;
		}
		t0 = t1; x0 = x1;
		t1 = t;  x1 = x;  d1 = d;
		++n;
	}

	__device__ int count() const
	{
		return (emitted >= max_amount_of_peaks) ? max_amount_of_peaks : emitted;
	}
};

// Транзиент и запись на равномерной сетке с одним вызовом шага. Цели по очереди:
// сначала tTr с пределом tTr (транзиент: последний шаг обрезается), затем отсчёты
// T0 + i*dtS (i < iters) с пределом tEnd = T0 + (iters + 1)*dtS — шагать, пока S.t < цели,
// и взять отсчёт плотным выходом, как ucuda_ad_advance_to. Разлёт — по S.X раз в
// CHECK_INTERVAL принятых шагов и по последнему отсчёту. Неподвижная точка — разность
// отсчётов в T0 + iters*dtS и в tEnd, как у loopCalculateDiscreteModelPeaks_int. push(y) —
// приёмник отсчёта.
#ifndef UCUDA_AD_NO_DENSE   // сетка вывода — плотным выходом; модулю LLE/LS не нужна
template <class Push>
__device__ __forceinline__ int ucudaAdUniformPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb tTr, const numb dtS, const size_t iters, const numb maxValue,
	Push& push, const volatile int* cancelFlag)
{
	const numb tEnd = tTr + (numb)(iters + 1) * dtS;
	numb y[AMOUNTOFX], y1[AMOUNTOFX];
	// Порог разлёта и флаг отмены (память хоста, чтение через PCIe) — раз в CHECK_INTERVAL
	// принятых шагов, не отсчётов: отсчётов в несколько раз больше, и счёт по ним стоил
	// БД / метрикам 2D на сетке ~10% времени.
	int cnt = 0;
	// k == 0 — транзиент, k == i + 1 — отсчёт i (последний, i == iters + 1, — сам tEnd).
	for (size_t k = 0; k <= iters + 2; ++k) {
		const bool   tr  = (k == 0);
		const size_t i   = k - 1;
		const numb   tt  = tr ? tTr : (i == iters + 1 ? tEnd : tTr + (numb)i * dtS);
		const numb   lim = tr ? tTr : tEnd;
		while (S.t < tt && !S.diverged) {
			ucuda_ad_step(S, K, a, P, lim);
			if (++cnt == CHECK_INTERVAL) {
				cnt = 0;
				if (ucudaAdOut(S.X, maxValue)) return REGIME_UNBOUND;
				if (cancelFlag != nullptr && *cancelFlag != 0) return REGIME_UNBOUND;
			}
		}
		if (S.diverged) return REGIME_UNBOUND;
		if (tr) continue;
		if (i < iters) {
			ucuda_ad_eval(S, K, a, P, tt, y);
			push(y);
		} else if (i == iters) {
			ucuda_ad_eval(S, K, a, P, tt, y1);
		} else {
			ucuda_ad_eval(S, K, a, P, tt, y);
			if (S.diverged || ucudaAdOut(y, maxValue)) return REGIME_UNBOUND;   // хвост короче CHECK_INTERVAL
			numb d = 0;
			for (int j = 0; j < AMOUNTOFX; ++j) d += fabs(y[j] - y1[j]);
			return (d < eps_fixed_point) ? REGIME_FIXED_POINT : REGIME_OSCILLATION;
		}
	}
	return REGIME_OSCILLATION;   // не достигается: k == iters + 2 возвращает выше
}

#endif // UCUDA_AD_NO_DENSE

// Транзиент и запись по узлам шага с одним вызовом шага: до tTr — транзиент, затем
// начальный узел, каждый dec-й принятый и последний — в push(t, X, F) (F = f(X)), до tEnd.
// Разлёт — по S.X раз в CHECK_INTERVAL принятых шагов и в конце; неподвижная точка —
// sum|f(x)| < eps_fixed_point в конце.
template <class Push>
__device__ __forceinline__ int ucudaAdNodesPointT(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb tTr, const numb tEnd, const int dec, const numb maxValue,
	Push& push, const volatile int* cancelFlag)
{
	bool rec = false;
	int  cnt = 0, nst = 0;
	for (;;) {
		if (!rec && !(S.t < tTr)) {
			rec = true;
			push(S.t, S.X, S.F0);
		}
		if (rec && !(S.t < tEnd)) break;
		ucuda_ad_step(S, K, a, P, rec ? tEnd : tTr);
		if (S.diverged) return REGIME_UNBOUND;   // NaN/inf: шаг не принимается с бесконечной ошибкой
		if (rec) {
			++cnt;
			if ((dec <= 1) || (cnt % dec) == 0 || !(S.t < tEnd))
				push(S.t, S.X, S.F0);
		}
		// Порог разлёта и флаг отмены (память хоста, чтение через PCIe) — раз в CHECK_INTERVAL
		// принятых шагов, как у циклов постоянного шага.
		if (++nst == CHECK_INTERVAL) {
			nst = 0;
			if (ucudaAdOut(S.X, maxValue)) return REGIME_UNBOUND;
			if (cancelFlag != nullptr && *cancelFlag != 0) return REGIME_UNBOUND;
		}
	}
	if (ucudaAdOut(S.X, maxValue)) return REGIME_UNBOUND;   // хвост короче CHECK_INTERVAL
	numb sf = 0;
	for (int j = 0; j < AMOUNTOFX; ++j) sf += fabs(S.F0[j]);
	return (sf < eps_fixed_point) ? REGIME_FIXED_POINT : REGIME_OSCILLATION;
}

// Узел -> пики (БД, бассейны).
struct UcudaAdPushNodePeaks {
	PeakStreamNU& peaks;
	int           writableVar;
	__device__ __forceinline__ void operator()(const numb t, const numb* X, const numb* F) {
		peaks.pushNode(t, UCUDA_AD_OBS(X, writableVar), UCUDA_AD_OBS(F, writableVar));
	}
};

__device__ __forceinline__ int ucudaAdNodesPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb tTr, const numb tEnd, const int dec,
	const int writableVar, const numb maxValue, PeakStreamNU& peaks,
	const volatile int* cancelFlag)
{
	UcudaAdPushNodePeaks push{ peaks, writableVar };
	return ucudaAdNodesPointT(S, K, a, P, tTr, tEnd, dec, maxValue, push, cancelFlag);
}

#ifndef UCUDA_AD_NO_DENSE
// Отсчёт сетки -> PeakStream пиков.
struct UcudaAdPushPeaks {
	PeakStream& peaks;
	int         writableVar;
	__device__ void operator()(const numb* y) { peaks.push(ucudaAdObs(y, writableVar)); }
};

// Транзиент и запись одной точки свипа в режиме raw: пики — в PeakStream (сетка)
// или PeakStreamNU (узлы). Возвращает режим.
__device__ __forceinline__ int ucudaAdPeaksPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb transientTime, const numb tRec, const numb dtS,
	const size_t iters, const int raw, const int preScaller, const int writableVar, const numb maxValue,
	PeakStream& pu, PeakStreamNU& pn, const volatile int* cancelFlag)
{
	if (raw)
		return ucudaAdNodesPoint(S, K, a, P, transientTime, transientTime + tRec, preScaller, writableVar,
			maxValue, pn, cancelFlag);
	UcudaAdPushPeaks push{ pu, writableVar };
	return ucudaAdUniformPoint(S, K, a, P, transientTime, dtS, iters, maxValue, push, cancelFlag);
}

#else
// Модуль без плотного выхода — только узлы шага (движок собирает его для raw_nodes): в
// состоянии нити нет D, Xp, Fp, а стадии W живут лишь внутри попытки. Сетку такой модуль
// не считает — движок с ней его не запускает.
__device__ __forceinline__ int ucudaAdPeaksPoint(UcudaAdaptState& S, const UcudaKrsFns& K, const numb* a,
	const UcudaAdaptParams& P, const numb transientTime, const numb tRec, const numb dtS,
	const size_t iters, const int raw, const int preScaller, const int writableVar, const numb maxValue,
	PeakStream& pu, PeakStreamNU& pn, const volatile int* cancelFlag)
{
	(void)dtS; (void)iters; (void)raw; (void)pu;
	return ucudaAdNodesPoint(S, K, a, P, transientTime, transientTime + tRec, preScaller, writableVar,
		maxValue, pn, cancelFlag);
}

#endif // UCUDA_AD_NO_DENSE

__device__ __forceinline__ void ucudaAdWriteStats(numb* adStats, const int idx, const UcudaAdaptState& S)
{
	if (adStats == nullptr) return;
	numb* st = adStats + (size_t)idx * 4;
	st[0] = (numb)S.st.nacc;
	st[1] = (numb)S.st.nrej;
	st[2] = (numb)S.st.nforced;
	st[3] = S.st.nacc > 0 ? S.st.hsum / (numb)S.st.nacc : (numb)0;
}

// Настройки шага в ядре свипа. У ядра узлов (UCUDA_AD_NODES_KERNEL) — аргумент по значению
// (__grid_constant__): лежит в константном банке аргументов, нить его не копирует. У общего
// ядра — копия на нить: ось свипа может править rtol / atol / параметр регулятора.
#ifdef UCUDA_AD_NODES_KERNEL
#define UCUDA_AD_P_ARG   const __grid_constant__ UcudaAdaptParams Pbase
#define UCUDA_AD_P_LOCAL const UcudaAdaptParams& P = Pbase;
#else
#define UCUDA_AD_P_ARG   const UcudaAdaptParams* __restrict__ Pbase
#define UCUDA_AD_P_LOCAL UcudaAdaptParams P = *Pbase;
#endif

#ifndef UCUDA_AD_NO_SWEEP_KERNELS

// БД 1D/2D: то же, что calculateDiscreteModelPeaksCUDA (раскладка пиков,
// интервалов и кодов режима одна и та же — dbscanCUDA читает её без изменений),
// но шагом управляет регулятор. adStats[idx*4] — принятые, отвергнутые,
// вынужденные шаги и средний h (nullptr — не писать).
__global__ void calculateDiscreteModelPeaksAdCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const size_t	amountOfCalculatedPoints,
	const int		dimension,
	const numb* __restrict__	ranges,
	const int* __restrict__		indicesOfMutVars,
	const numb* __restrict__	initialConditions,
	const int		amountOfInitialConditions,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const int		writableVar,
	const numb		maxValue,
	numb*			outPeaks,
	numb*			timeOfPeaks,
	int*			maxValueCheckerArray,
	const int		logAxisMask,
	const size_t	peakStride,
	const int		peakCapacity,
	UCUDA_AD_P_ARG,
	const int* __restrict__		axisKind,
	const numb		tolRatio,
	const numb		transientTime,
	const numb		tRec,
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
	const int		raw,
	const int		peakInterp,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride,
	const size_t	progressUnits,
	numb*			adStats)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + (threadIdx.x * sharedStride);
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	UCUDA_AD_P_LOCAL
	ucudaSetupSweepPointAd(nPts, amountOfCalculatedPoints, idx, dimension, ranges, indicesOfMutVars,
		initialConditions, values, amountOfValues, logAxisMask, axisKind, tolRatio,
		localX, localValues, P);

	const UcudaKrsFns K{};
	UcudaAdaptState S;
	ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);

	PeakStream   pu;
	PeakStreamNU pn;
	if (!raw) pu.init(outPeaks, timeOfPeaks, (size_t)idx * peakStride, dtOut * (numb)preScaller, iters, peakCapacity);
	else      pn.init(outPeaks, timeOfPeaks, (size_t)idx * peakStride, peakCapacity, peakInterp);
	const int flag  = ucudaAdPeaksPoint(S, K, localValues, P, transientTime, tRec, dtOut * (numb)preScaller,
		iters, raw, preScaller, writableVar, maxValue, pu, pn, cancelFlag);
	const int count = raw ? pn.count() : pu.count();
	// Как у слитого ядра: OSCILLATION заменяется числом пиков, FP и UNBOUND остаются кодами.
	if (maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = (flag == REGIME_OSCILLATION) ? count : flag;
	ucudaAdWriteStats(adStats, idx, S);
	ucudaProgressTopUp(progressCounter, progressStride, progressUnits, 0);
}

// Признаки бассейна по суммам пиков — та же арифметика, что в
// calculateDiscreteModelAvgPeaksCUDA.
__device__ __forceinline__ void ucudaAdBasinFeatures(const int flag, const int amountOfPeaks,
	const numb sumP, const numb sumP2, const numb sumI, const numb sumI2, const numb last,
	const int feature1, const int feature2, const numb mult1, const numb mult2,
	numb& out1, numb& out2)
{
	if (flag == REGIME_UNBOUND) { out1 = 999; out2 = 999; return; }
	if (flag == REGIME_FIXED_POINT) { out1 = last; out2 = -1.0; return; }
	if (amountOfPeaks <= 0) { out1 = -999; out2 = -1.0; return; }
	const numb inv_n  = (numb)1.0 / (numb)amountOfPeaks;
	const numb mean_p = sumP * inv_n;
	const numb mean_i = sumI * inv_n;
	const numb rms_p  = sqrt(sumP2 * inv_n);
	const numb rms_i  = sqrt(sumI2 * inv_n);
	const numb var_p  = sumP2 * inv_n - mean_p * mean_p;
	const numb var_i  = sumI2 * inv_n - mean_i * mean_i;
	const numb std_p  = sqrt(var_p < 0 ? (numb)0 : var_p);
	const numb std_i  = sqrt(var_i < 0 ? (numb)0 : var_i);
	out1 = mult1 * compute_basin_feature(feature1, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
	out2 = mult2 * compute_basin_feature(feature2, mean_p, mean_i, rms_p, rms_i, std_p, std_i);
}

// Бассейны: то же, что calculateDiscreteModelAvgPeaksCUDA, с адаптивным шагом.
__global__ void calculateDiscreteModelAvgPeaksAdCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const size_t	amountOfCalculatedPoints,
	const int		dimension,
	const numb* __restrict__	ranges,
	const int* __restrict__		indicesOfMutVars,
	const numb* __restrict__	initialConditions,
	const int		amountOfInitialConditions,
	const numb* __restrict__	values,
	const int		amountOfValues,
	const int		writableVar,
	const numb		maxValue,
	numb*			outAvgPeaks,
	numb*			AvgTimeOfPeaks,
	int*			systemCheker,
	const int		feature1,
	const int		feature2,
	const numb		mult1,
	const numb		mult2,
	UCUDA_AD_P_ARG,
	const int* __restrict__		axisKind,
	const numb		tolRatio,
	const numb		transientTime,
	const numb		tRec,
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
	const int		raw,
	const int		peakInterp,
	const volatile int* cancelFlag,
	int*			progressCounter,
	const int		progressStride,
	const size_t	progressUnits,
	numb*			adStats)
{
	extern __shared__ numb s[];
	const int sharedStride = ucuda_shared_stride(amountOfInitialConditions, amountOfValues);
	numb* localX = s + (threadIdx.x * sharedStride);
	numb* localValues = localX + amountOfInitialConditions;

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)
		return;

	UCUDA_AD_P_LOCAL
	ucudaSetupSweepPointAd(nPts, amountOfCalculatedPoints, idx, dimension, ranges, indicesOfMutVars,
		initialConditions, values, amountOfValues, 0, axisKind, tolRatio, localX, localValues, P);

	const UcudaKrsFns K{};
	UcudaAdaptState S;
	ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);

	PeakStream   pu;
	PeakStreamNU pn;
	if (!raw) pu.init(nullptr, nullptr, 0, dtOut * (numb)preScaller, iters, 0);
	else      pn.init(nullptr, nullptr, 0, 0, peakInterp);
	const int flag = ucudaAdPeaksPoint(S, K, localValues, P, transientTime, tRec, dtOut * (numb)preScaller,
		iters, raw, preScaller, writableVar, maxValue, pu, pn, cancelFlag);
	const int  count = raw ? pn.count() : pu.count();
	const numb sumP  = raw ? pn.sumP  : pu.sumP;
	const numb sumP2 = raw ? pn.sumP2 : pu.sumP2;
	const numb sumI  = raw ? pn.sumI  : pu.sumI;
	const numb sumI2 = raw ? pn.sumI2 : pu.sumI2;
	const numb last  = raw ? pn.last  : pu.last;
	if (systemCheker != nullptr) systemCheker[idx] = flag;
	numb f1, f2;
	ucudaAdBasinFeatures(flag, count, sumP, sumP2, sumI, sumI2, last, feature1, feature2, mult1, mult2, f1, f2);
	outAvgPeaks[idx] = f1;
	AvgTimeOfPeaks[idx] = f2;
	ucudaAdWriteStats(adStats, idx, S);
	ucudaProgressTopUp(progressCounter, progressStride, progressUnits, 0);
}

#endif // UCUDA_AD_NO_SWEEP_KERNELS

#ifdef UCUDA_AD_CONT_KERNEL

// БД 1D, continuation: одна нить идёт по точкам цепочки подряд, и каждая стартует
// с конечного состояния предыдущей — как bifurcation1dContinuationKernel постоянного
// шага, но вместе с X переносятся предложенный шаг и память регулятора
// (ucuda_ad_restart). Раскладка пиков и кодов — как у calculateDiscreteModelPeaksAdCUDA,
// строка j — j-я точка цепочки (reverse разворачивает саму цепочку). Ось
// (axisKind[0]) — параметр a[mutParamIdx] или настройка шага. Тик прогресса — точка.
__global__ void calculateDiscreteModelPeaksAdContCUDA(
	const int		nPts,
	const numb		lo,
	const numb		hi,
	const int		reverse,
	const int		logScale,
	const int		mutParamIdx,
	const numb* __restrict__	baseValues,
	const int		amountOfValues,
	const numb* __restrict__	baseX,
	const int		writableVar,
	const numb		maxValue,
	numb*			outPeaks,
	numb*			timeOfPeaks,
	int*			maxValueCheckerArray,
	const size_t	peakStride,
	const int		peakCapacity,
	const UcudaAdaptParams* __restrict__ Pbase,
	const int* __restrict__		axisKind,
	const numb		tolRatio,
	const numb		transientTime,
	const numb		tRec,
	const numb		dtOut,
	const int		preScaller,
	const size_t	iters,
	const int		raw,
	const int		peakInterp,
	const volatile int* cancelFlag,
	int*			progressCounter,
	numb*			adStats)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	numb x[AMOUNTOFX];
	numb a[64];   // kMaxAmountOfValues в движке
	for (int i = 0; i < AMOUNTOFX; ++i) x[i] = baseX[i];
	for (int i = 0; i < amountOfValues && i < 64; ++i) a[i] = baseValues[i];

	UcudaAdaptParams P = *Pbase;
	const int kind = axisKind[0];
	const UcudaKrsFns K{};
	UcudaAdaptState S;

	for (int j = 0; j < nPts; ++j) {
		if (cancelFlag != nullptr && *cancelFlag != 0) return;
		if (progressCounter != nullptr) atomicAdd(progressCounter, 1);
		const numb v = ucuda_node_value_cont(j, nPts, lo, hi, logScale != 0, reverse != 0);
		if (kind == UCUDA_AXIS_SYSTEM) a[mutParamIdx] = v;
		else ucudaAdApplyStepAxis(kind, v, tolRatio, P);

		if (j == 0) ucuda_ad_init(S, K, AMOUNTOFX, x, (numb)0, a, P);
		else        ucuda_ad_restart(S, K, (numb)0, a, P);

		PeakStream   pu;
		PeakStreamNU pn;
		if (!raw) pu.init(outPeaks, timeOfPeaks, (size_t)j * peakStride, dtOut * (numb)preScaller, iters, peakCapacity);
		else      pn.init(outPeaks, timeOfPeaks, (size_t)j * peakStride, peakCapacity, peakInterp);
		// Разошедшееся состояние предыдущей точки переносится, как у постоянного
		// шага, — и сразу даёт UNBOUND, без спуска шага до h_min.
		const int flag  = ucudaAdOut(S.X, maxValue) ? REGIME_UNBOUND
		                : ucudaAdPeaksPoint(S, K, a, P, transientTime, tRec, dtOut * (numb)preScaller,
		                                    iters, raw, preScaller, writableVar, maxValue, pu, pn, cancelFlag);
		const int count = raw ? pn.count() : pu.count();
		if (maxValueCheckerArray != nullptr)
			maxValueCheckerArray[j] = (flag == REGIME_OSCILLATION) ? count : flag;
		ucudaAdWriteStats(adStats, j, S);
	}
}

#endif // UCUDA_AD_CONT_KERNEL
