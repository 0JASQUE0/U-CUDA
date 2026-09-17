#include "host_internal.cuh"
#include "ucuda/domain/rqa.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

// Потолок стороны матрицы. n = 4096 -> 4096^2 * 8 байт = 134 МБ на устройстве и столько же
// в out.dist. Выше упирается и видеопамять, и осмысленность. Превышение — ошибка, а НЕ
// молчаливое прореживание: числа метрик обязаны относиться к той матрице, которую видно.
constexpr int kRqaMaxPoints   = 4096;
constexpr int kRqaHistBins    = 4096;   // бинов на проход подбора eps (проходов два)
constexpr int kRqaRedThreads  = 256;    // фиксировано: под него рассчитан shared-массив редукции

__device__ __forceinline__ bool rqa_masked(int i, int j, int theiler)
{
	int d = i - j;
	if (d < 0) d = -d;
	return d <= theiler;
}

// D[i][j] = ||p_i - p_j|| в выбранной норме. pts — SoA: pts[k*n + i] = k-я координата i-й точки.
__global__ void rqaDistKernel(const numb* __restrict__ pts, int n, int d, int norm,
	numb* __restrict__ D)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n) return;

	numb acc = 0;
	if (norm == 0) {			// Euclidean
		for (int k = 0; k < d; ++k) {
			const numb t = __ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]);
			acc += t * t;
		}
		acc = sqrt(acc);
	}
	else if (norm == 1) {		// Maximum
		for (int k = 0; k < d; ++k) {
			const numb t = fabs(__ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]));
			if (t > acc) acc = t;
		}
	}
	else {						// Manhattan
		for (int k = 0; k < d; ++k)
			acc += fabs(__ldg(&pts[(size_t)k * n + i]) - __ldg(&pts[(size_t)k * n + j]));
	}
	D[(size_t)i * n + j] = acc;
}

// max(D) двухступенчатой редукцией: блок -> out[blockIdx.x], хвост досуммирует хост.
__global__ void rqaMaxKernel(const numb* __restrict__ D, size_t total, numb* __restrict__ out)
{
	__shared__ numb sm[kRqaRedThreads];
	const size_t stride = (size_t)gridDim.x * blockDim.x;
	numb m = 0;
	for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
		const numb v = D[t];
		if (v > m) m = v;
	}
	sm[threadIdx.x] = m;
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x]) sm[threadIdx.x] = sm[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) out[blockIdx.x] = sm[0];
}

// Гистограмма расстояний по [lo, hi) для подбора eps под целевой RR. Считаются ТОЛЬКО пары вне
// окна Тейлера — те же, по которым потом считается сам RR, иначе подобранный порог давал бы
// другой RR, чем показанный в таблице. hist[nbins] — сколько попало ниже lo, hist[nbins+1] — выше.
__global__ void rqaHistKernel(const numb* __restrict__ D, int n, int theiler,
	numb lo, numb hi, int nbins, unsigned long long* __restrict__ hist)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n) return;
	if (rqa_masked(i, j, theiler)) return;

	const numb v = D[(size_t)i * n + j];
	if (v < lo) { atomicAdd(&hist[nbins], 1ULL); return; }
	const int b = (int)((v - lo) / (hi - lo) * nbins);
	if (b >= nbins) { atomicAdd(&hist[nbins + 1], 1ULL); return; }
	atomicAdd(&hist[b < 0 ? 0 : b], 1ULL);
}

// Диагональные линии: поток на диагональ k = t - (n-1). Диагонали внутри окна Тейлера (включая
// LOI) не обрабатываются вовсе — их вклад не должен попасть ни в одну метрику.
// diagCount[t] — число рекуррентных точек на диагонали (нужно для RR и TREND).
__global__ void rqaDiagKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ histDiag, unsigned int* __restrict__ diagCount)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= 2 * n - 1) return;
	const int k = t - (n - 1);
	const int ak = k < 0 ? -k : k;
	if (ak <= theiler) return;

	const int i0 = (k >= 0) ? 0 : -k;
	const int j0 = (k >= 0) ? k : 0;
	const int len = n - ak;

	int run = 0;
	unsigned int cnt = 0;
	for (int s = 0; s < len; ++s) {
		const bool rec = (D[(size_t)(i0 + s) * n + (j0 + s)] <= eps);
		if (rec) { ++run; ++cnt; }
		else if (run > 0) { atomicAdd(&histDiag[run], 1u); run = 0; }
	}
	if (run > 0) atomicAdd(&histDiag[run], 1u);
	diagCount[t] = cnt;
}

// Вертикальные линии: поток на столбец. Состояния клетки: 1 = рекуррентна, 0 = белая,
// 2 = исключена окном Тейлера (или конец столбца). Исключённая клетка РВЁТ и чёрную серию, и
// белый промежуток, и цепочку времён возврата — иначе полоса маски читалась бы как настоящий
// белый интервал и завышала бы T1/T2/W.
__global__ void rqaVertKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ histVert, unsigned int* __restrict__ histWhite,
	unsigned long long* __restrict__ acc)   // [0]=sum T1, [1]=cnt T1, [2]=sum T2, [3]=cnt T2
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= n) return;

	int run = 0;			// длина текущей чёрной серии
	int gap = -1;			// длина белого промежутка; -1 = слева ещё не было опорной чёрной точки
	int prevRec = -1;		// индекс предыдущей рекуррентной точки
	int prevStart = -1;		// индекс начала предыдущего чёрного блока
	unsigned long long s1 = 0, c1 = 0, s2 = 0, c2 = 0;

	for (int i = 0; i <= n; ++i) {
		int st;
		if (i == n)								st = 2;
		else if (rqa_masked(i, j, theiler))		st = 2;
		else									st = (D[(size_t)i * n + j] <= eps) ? 1 : 0;

		if (st == 1) {
			if (run == 0) {
				if (gap > 0) atomicAdd(&histWhite[gap], 1u);
				if (prevStart >= 0) { s2 += (unsigned long long)(i - prevStart); ++c2; }
				prevStart = i;
			}
			if (prevRec >= 0) { s1 += (unsigned long long)(i - prevRec); ++c1; }
			prevRec = i;
			++run;
			gap = 0;
		}
		else if (st == 0) {
			if (run > 0) { atomicAdd(&histVert[run], 1u); run = 0; }
			if (gap >= 0) ++gap;
		}
		else {
			if (run > 0) { atomicAdd(&histVert[run], 1u); run = 0; }
			gap = -1; prevRec = -1; prevStart = -1;
		}
	}
	atomicAdd(&acc[0], s1); atomicAdd(&acc[1], c1);
	atomicAdd(&acc[2], s2); atomicAdd(&acc[3], c2);
}

// Степень вершины сети рекуррентности (строка матрицы за вычетом окна Тейлера).
__global__ void rqaDegKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned int* __restrict__ deg)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	unsigned int k = 0;
	for (int j = 0; j < n; ++j)
		if (!rqa_masked(i, j, theiler) && D[(size_t)i * n + j] <= eps) ++k;
	deg[i] = k;
}

// Треугольники сети: поток на пару i<j. Для существующего ребра (i,j) считаем общих соседей —
// это число треугольников на ребре. triNode[i] в итоге = 2 * (число треугольников при вершине i).
__global__ void rqaTriKernel(const numb* __restrict__ D, int n, numb eps, int theiler,
	unsigned long long* __restrict__ triNode)
{
	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= n || j >= n || j <= i) return;
	if (rqa_masked(i, j, theiler) || D[(size_t)i * n + j] > eps) return;

	unsigned long long t = 0;
	for (int k = 0; k < n; ++k) {
		if (k == i || k == j) continue;
		if (rqa_masked(i, k, theiler) || D[(size_t)i * n + k] > eps) continue;
		if (rqa_masked(j, k, theiler) || D[(size_t)j * n + k] > eps) continue;
		++t;
	}
	if (t) { atomicAdd(&triNode[i], t); atomicAdd(&triNode[j], t); }
}

// Энтропия Шеннона распределения длин линий по гистограмме [from..n].
// norm == true -> делится на ln(число непустых бинов), т.е. приводится к [0,1] (так принято
// определять RTE). Возвращает NaN, если линий нужной длины нет вовсе.
double rqa_entropy(const std::vector<unsigned int>& hist, int from, bool norm)
{
	double total = 0.0;
	int used = 0;
	for (size_t l = (size_t)from; l < hist.size(); ++l)
		if (hist[l]) { total += hist[l]; ++used; }
	if (total <= 0.0) return std::numeric_limits<double>::quiet_NaN();

	double e = 0.0;
	for (size_t l = (size_t)from; l < hist.size(); ++l) {
		if (!hist[l]) continue;
		const double p = hist[l] / total;
		e -= p * std::log(p);
	}
	if (!norm) return e;
	return (used > 1) ? e / std::log((double)used) : 0.0;
}

}  // namespace

bool rqa::compute(const std::vector<std::vector<double>>& traj, double dt_traj,
	const rqa::Config& cfg, rqa::Result& out)
{
	using rqa::Source;
	using rqa::Norm;
	using rqa::EpsMode;

	out = rqa::Result();
	const double kNaN = std::numeric_limits<double>::quiet_NaN();

	// ---------------- валидация входа ----------------
	if (cfg.source == Source::None) {
		out.error = "RQA: choose the state-vector source (full state or delay embedding)";
		return false;
	}
	if ((int)traj.size() < 8 || traj[0].empty()) { out.error = "RQA: trajectory too short"; return false; }
	if (!(dt_traj > 0.0)) { out.error = "RQA: non-positive sample step"; return false; }
	if (cfg.points < 16 || cfg.points > kRqaMaxPoints) {
		out.error = "RQA: RP points must be within 16.." + std::to_string(kRqaMaxPoints);
		return false;
	}
	if (cfg.theiler < 0) { out.error = "RQA: Theiler window must be >= 0"; return false; }
	if (cfg.l_min < 1 || cfg.v_min < 1) { out.error = "RQA: l_min and v_min must be >= 1"; return false; }

	const int nsteps = (int)traj.size();
	const int dim = (int)traj[0].size();

	int d = 0, avail = 0;
	if (cfg.source == Source::StateVector) {
		d = dim;
		avail = nsteps;
	}
	else {
		if (cfg.m < 1) { out.error = "RQA: embedding dimension must be >= 1"; return false; }
		if (cfg.tau < 1) { out.error = "RQA: embedding delay must be >= 1"; return false; }
		if (cfg.var < 0 || cfg.var >= dim) { out.error = "RQA: embedding variable out of range"; return false; }
		d = cfg.m;
		avail = nsteps - (cfg.m - 1) * cfg.tau;
		if (avail < 16) { out.error = "RQA: embedding window (m-1)*tau is longer than the trajectory"; return false; }
	}

	const int n = std::min(cfg.points, avail);
	if (n < 16) { out.error = "RQA: not enough samples"; return false; }
	if (cfg.theiler >= n) { out.error = "RQA: Theiler window >= matrix size"; return false; }

	// ---------------- выборка точек (SoA) ----------------
	// Прореживаем равномерно до n. Шаг между отсчётами постоянен, поэтому dt честный; он выводится
	// в UI, чтобы длины линий (они В ОТСЧЁТАХ) можно было перевести в секунды.
	const double span = (double)(avail - 1) / (double)(n - 1);
	std::vector<numb> pts((size_t)d * n);
	for (int i = 0; i < n; ++i) {
		const int base = (int)std::llround((double)i * span);
		if (cfg.source == Source::StateVector) {
			const std::vector<double>& p = traj[(size_t)base];
			for (int k = 0; k < d; ++k)
				pts[(size_t)k * n + i] = (numb)(k < (int)p.size() ? p[k] : 0.0);
		}
		else {
			for (int k = 0; k < d; ++k) {
				const std::vector<double>& p = traj[(size_t)base + (size_t)k * cfg.tau];
				pts[(size_t)k * n + i] = (numb)(cfg.var < (int)p.size() ? p[cfg.var] : 0.0);
			}
		}
	}
	out.n = n;
	out.dt = span * dt_traj;
	out.t0 = 0.0;
	out.t1 = (double)(avail - 1) * dt_traj;

	for (size_t t = 0; t < pts.size(); ++t)
		if (!std::isfinite((double)pts[t])) { out.error = "RQA: trajectory contains nan/inf"; return false; }

	// ---------------- устройство ----------------
	numb* d_pts = nullptr; numb* d_D = nullptr; numb* d_blockMax = nullptr;
	unsigned long long* d_hist = nullptr; unsigned long long* d_acc = nullptr;
	unsigned long long* d_triNode = nullptr;
	unsigned int* d_histDiag = nullptr; unsigned int* d_histVert = nullptr;
	unsigned int* d_histWhite = nullptr; unsigned int* d_diagCount = nullptr;
	unsigned int* d_deg = nullptr;

	auto free_all = [&]() {
		cudaFree(d_pts); cudaFree(d_D); cudaFree(d_blockMax);
		cudaFree(d_hist); cudaFree(d_acc); cudaFree(d_triNode);
		cudaFree(d_histDiag); cudaFree(d_histVert); cudaFree(d_histWhite);
		cudaFree(d_diagCount); cudaFree(d_deg);
	};

#define RQA_CHECK(call, what)                                                            \
	do {                                                                                 \
		const cudaError_t _rc = (call);                                                  \
		if (_rc != cudaSuccess) {                                                        \
			out.error = std::string("RQA: ") + (what) + ": " + cudaGetErrorString(_rc);  \
			free_all(); return false;                                                    \
		}                                                                                \
	} while (0)
#define RQA_LAUNCH(what)                                                                 \
	do {                                                                                 \
		RQA_CHECK(cudaGetLastError(), what);                                             \
		RQA_CHECK(cudaDeviceSynchronize(), what);                                        \
	} while (0)

	const size_t cells = (size_t)n * (size_t)n;
	const dim3 blk2(16, 16);
	const dim3 grd2((n + blk2.x - 1) / blk2.x, (n + blk2.y - 1) / blk2.y);

	RQA_CHECK(cudaMalloc(&d_pts, pts.size() * sizeof(numb)), "alloc points");
	RQA_CHECK(cudaMalloc(&d_D, cells * sizeof(numb)), "alloc distance matrix");
	RQA_CHECK(cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(numb), cudaMemcpyHostToDevice), "upload points");

	rqaDistKernel<<<grd2, blk2>>>(d_pts, n, d, (int)cfg.norm, d_D);
	RQA_LAUNCH("distance matrix");

	// ---------------- max(D) ----------------
	int redBlocks = (int)((cells + kRqaRedThreads - 1) / kRqaRedThreads);
	if (redBlocks > 1024) redBlocks = 1024;
	RQA_CHECK(cudaMalloc(&d_blockMax, (size_t)redBlocks * sizeof(numb)), "alloc reduction");
	rqaMaxKernel<<<redBlocks, kRqaRedThreads>>>(d_D, cells, d_blockMax);
	RQA_LAUNCH("max distance");
	{
		std::vector<numb> bm((size_t)redBlocks);
		RQA_CHECK(cudaMemcpy(bm.data(), d_blockMax, bm.size() * sizeof(numb), cudaMemcpyDeviceToHost), "read reduction");
		numb mx = 0;
		for (size_t t = 0; t < bm.size(); ++t) if (bm[t] > mx) mx = bm[t];
		out.dist_max = (double)mx;
	}
	if (!(out.dist_max > 0.0)) {
		out.error = "RQA: all sampled points coincide (max distance is 0)";
		free_all(); return false;
	}

	// ---------------- порог eps ----------------
	const long long w = cfg.theiler;
	const long long npairs = (long long)n * n - (2 * w + 1) * (long long)n + w * (w + 1);
	if (npairs <= 0) { out.error = "RQA: Theiler window leaves no pairs"; free_all(); return false; }

	numb eps = 0;
	if (cfg.eps_mode == EpsMode::Absolute) {
		eps = cfg.eps;
	}
	else if (cfg.eps_mode == EpsMode::FracMaxDist) {
		eps = (numb)(cfg.eps_frac * out.dist_max);
	}
	else if (cfg.eps_mode == EpsMode::FracStd) {
		// СКО облака точек от центра масс — тот масштаб, в котором в статьях пишут "eps = 0.1 sigma".
		std::vector<double> mean((size_t)d, 0.0);
		for (int k = 0; k < d; ++k) {
			double s = 0.0;
			for (int i = 0; i < n; ++i) s += (double)pts[(size_t)k * n + i];
			mean[(size_t)k] = s / n;
		}
		double var = 0.0;
		for (int k = 0; k < d; ++k)
			for (int i = 0; i < n; ++i) {
				const double t = (double)pts[(size_t)k * n + i] - mean[(size_t)k];
				var += t * t;
			}
		eps = (numb)(cfg.eps_frac * std::sqrt(var / n));
	}
	else {
		// TargetRR: квантиль распределения расстояний. Два прохода гистограммой вместо сортировки
		// n^2 значений; итоговое разрешение по eps ~ dist_max / 4096^2.
		if (!(cfg.target_rr > 0.0) || !(cfg.target_rr < 1.0)) {
			out.error = "RQA: target RR must be within (0, 1)"; free_all(); return false;
		}
		RQA_CHECK(cudaMalloc(&d_hist, (size_t)(kRqaHistBins + 2) * sizeof(unsigned long long)), "alloc histogram");
		const long long want = (long long)std::llround(cfg.target_rr * (double)npairs);
		double lo = 0.0, hi = out.dist_max;
		long long below = 0;
		for (int pass = 0; pass < 2; ++pass) {
			RQA_CHECK(cudaMemset(d_hist, 0, (size_t)(kRqaHistBins + 2) * sizeof(unsigned long long)), "clear histogram");
			rqaHistKernel<<<grd2, blk2>>>(d_D, n, cfg.theiler, (numb)lo, (numb)hi, kRqaHistBins, d_hist);
			RQA_LAUNCH("distance histogram");
			std::vector<unsigned long long> h((size_t)kRqaHistBins + 2);
			RQA_CHECK(cudaMemcpy(h.data(), d_hist, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read histogram");

			long long cum = below;
			int b = kRqaHistBins - 1;
			for (int t = 0; t < kRqaHistBins; ++t) {
				cum += (long long)h[(size_t)t];
				if (cum >= want) { b = t; break; }
			}
			// below — пары строго левее нового окна: второй проход обязан продолжать ту же
			// накопленную сумму, а не начинать считать RR заново.
			for (int t = 0; t < b; ++t) below += (long long)h[(size_t)t];
			const double step = (hi - lo) / kRqaHistBins;
			lo = lo + step * b;
			hi = lo + step;
			eps = (numb)hi;
		}
	}
	if (!(eps > 0.0)) { out.error = "RQA: eps resolved to a non-positive value"; free_all(); return false; }
	out.eps_used = (double)eps;

	// ---------------- линии ----------------
	RQA_CHECK(cudaMalloc(&d_histDiag, (size_t)(n + 1) * sizeof(unsigned int)), "alloc diagonal histogram");
	RQA_CHECK(cudaMalloc(&d_histVert, (size_t)(n + 1) * sizeof(unsigned int)), "alloc vertical histogram");
	RQA_CHECK(cudaMalloc(&d_histWhite, (size_t)(n + 1) * sizeof(unsigned int)), "alloc white-line histogram");
	RQA_CHECK(cudaMalloc(&d_diagCount, (size_t)(2 * n - 1) * sizeof(unsigned int)), "alloc diagonal counts");
	RQA_CHECK(cudaMalloc(&d_acc, 4 * sizeof(unsigned long long)), "alloc recurrence-time accumulators");
	RQA_CHECK(cudaMemset(d_histDiag, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear diagonal histogram");
	RQA_CHECK(cudaMemset(d_histVert, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear vertical histogram");
	RQA_CHECK(cudaMemset(d_histWhite, 0, (size_t)(n + 1) * sizeof(unsigned int)), "clear white-line histogram");
	RQA_CHECK(cudaMemset(d_diagCount, 0, (size_t)(2 * n - 1) * sizeof(unsigned int)), "clear diagonal counts");
	RQA_CHECK(cudaMemset(d_acc, 0, 4 * sizeof(unsigned long long)), "clear accumulators");

	rqaDiagKernel<<<(2 * n - 1 + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_histDiag, d_diagCount);
	RQA_LAUNCH("diagonal lines");
	rqaVertKernel<<<(n + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_histVert, d_histWhite, d_acc);
	RQA_LAUNCH("vertical lines");

	std::vector<unsigned int> hDiag((size_t)n + 1), hVert((size_t)n + 1), hWhite((size_t)n + 1);
	std::vector<unsigned int> dCount((size_t)(2 * n - 1));
	unsigned long long acc[4] = { 0, 0, 0, 0 };
	RQA_CHECK(cudaMemcpy(hDiag.data(), d_histDiag, hDiag.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read diagonal histogram");
	RQA_CHECK(cudaMemcpy(hVert.data(), d_histVert, hVert.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read vertical histogram");
	RQA_CHECK(cudaMemcpy(hWhite.data(), d_histWhite, hWhite.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read white-line histogram");
	RQA_CHECK(cudaMemcpy(dCount.data(), d_diagCount, dCount.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read diagonal counts");
	RQA_CHECK(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost), "read accumulators");

	// ---------------- сетевые меры (опционально) ----------------
	double clustering = kNaN, transitivity = kNaN;
	bool network_valid = false;
	if (cfg.network_measures) {
		RQA_CHECK(cudaMalloc(&d_deg, (size_t)n * sizeof(unsigned int)), "alloc degrees");
		RQA_CHECK(cudaMalloc(&d_triNode, (size_t)n * sizeof(unsigned long long)), "alloc triangle counts");
		RQA_CHECK(cudaMemset(d_triNode, 0, (size_t)n * sizeof(unsigned long long)), "clear triangle counts");
		rqaDegKernel<<<(n + 255) / 256, 256>>>(d_D, n, eps, cfg.theiler, d_deg);
		RQA_LAUNCH("network degrees");
		rqaTriKernel<<<grd2, blk2>>>(d_D, n, eps, cfg.theiler, d_triNode);
		RQA_LAUNCH("network triangles");

		std::vector<unsigned int> deg((size_t)n);
		std::vector<unsigned long long> tri((size_t)n);
		RQA_CHECK(cudaMemcpy(deg.data(), d_deg, deg.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read degrees");
		RQA_CHECK(cudaMemcpy(tri.data(), d_triNode, tri.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read triangle counts");

		// triNode[i] = 2 * (треугольников при вершине i) -> C_i = triNode[i] / (k_i*(k_i-1)).
		double csum = 0.0; int cnodes = 0;
		double triSum = 0.0, tripleSum = 0.0;
		for (int i = 0; i < n; ++i) {
			const double k = (double)deg[(size_t)i];
			if (k < 2.0) continue;
			csum += (double)tri[(size_t)i] / (k * (k - 1.0));
			++cnodes;
			triSum += (double)tri[(size_t)i] * 0.5;
			tripleSum += k * (k - 1.0) * 0.5;
		}
		// transitivity = 3 * (число треугольников) / (число связных троек). triSum здесь — уже
		// СУММА ПО ВЕРШИНАМ T_i, а она равна 3 * (число треугольников) сама по себе (каждый
		// треугольник даёт вклад в три свои вершины), поэтому множителя 3 тут быть не должно.
		// Проверка: у полного графа k_i = n-1, T_i = (n-1)(n-2)/2, и отношение даёт ровно 1.
		clustering = (cnodes > 0) ? csum / cnodes : kNaN;
		transitivity = (tripleSum > 0.0) ? triSum / tripleSum : kNaN;
		network_valid = true;
	}

	// ---------------- матрица наружу ----------------
	// out.dist — double, потому что его ждёт HeatmapView::render. Ветка else существует ради того,
	// чтобы при смене typedef numb на float расширение точности было ВИДНО здесь, а не молча
	// происходило внутри cudaMemcpy с несовпадающим размером элемента.
	out.dist.resize(cells);
	if (sizeof(numb) == sizeof(double)) {
		RQA_CHECK(cudaMemcpy(out.dist.data(), d_D, cells * sizeof(numb), cudaMemcpyDeviceToHost), "read distance matrix");
	}
	else {
		std::vector<numb> tmp(cells);
		RQA_CHECK(cudaMemcpy(tmp.data(), d_D, cells * sizeof(numb), cudaMemcpyDeviceToHost), "read distance matrix");
		for (size_t t = 0; t < cells; ++t) out.dist[t] = (double)tmp[t];
	}
	free_all();
#undef RQA_LAUNCH
#undef RQA_CHECK

	// ---------------- метрики ----------------
	rqa::Metrics& M = out.metrics;

	unsigned long long nrec = 0;
	for (size_t t = 0; t < dCount.size(); ++t) nrec += dCount[t];
	M.RR = (double)nrec / (double)npairs;

	// Диагонали. Знаменатель DET — все рекуррентные точки вне окна Тейлера, т.е. ровно nrec:
	// каждая такая точка принадлежит ровно одной диагональной линии.
	{
		double sumAll = 0.0, sumMin = 0.0, cntMin = 0.0;
		int lmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hDiag[(size_t)l];
			if (p <= 0.0) continue;
			sumAll += (double)l * p;
			if (l >= cfg.l_min) { sumMin += (double)l * p; cntMin += p; lmax = l; }
		}
		M.DET   = (sumAll > 0.0) ? sumMin / sumAll : kNaN;
		M.L     = (cntMin > 0.0) ? sumMin / cntMin : kNaN;
		M.L_max = (lmax > 0) ? (double)lmax : kNaN;
		M.DIV   = (lmax > 0) ? 1.0 / (double)lmax : kNaN;
		M.ENTR  = rqa_entropy(hDiag, cfg.l_min, false);
		M.RATIO = (M.RR > 0.0) ? M.DET / M.RR : kNaN;
	}

	// Вертикали.
	{
		double sumAll = 0.0, sumMin = 0.0, cntMin = 0.0;
		int vmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hVert[(size_t)l];
			if (p <= 0.0) continue;
			sumAll += (double)l * p;
			if (l >= cfg.v_min) { sumMin += (double)l * p; cntMin += p; vmax = l; }
		}
		M.LAM    = (sumAll > 0.0) ? sumMin / sumAll : kNaN;
		M.TT     = (cntMin > 0.0) ? sumMin / cntMin : kNaN;
		M.V_max  = (vmax > 0) ? (double)vmax : kNaN;
		M.V_ENTR = rqa_entropy(hVert, cfg.v_min, false);
	}

	// Белые вертикали и времена возврата. Всё В ОТСЧЁТАХ выборки (умножить на out.dt для секунд).
	{
		double sum = 0.0, cnt = 0.0;
		int wmax = 0;
		for (int l = 1; l <= n; ++l) {
			const double p = (double)hWhite[(size_t)l];
			if (p <= 0.0) continue;
			sum += (double)l * p; cnt += p; wmax = l;
		}
		M.W     = (cnt > 0.0) ? sum / cnt : kNaN;
		M.W_max = (wmax > 0) ? (double)wmax : kNaN;
		M.RTE   = rqa_entropy(hWhite, 1, true);
	}
	M.T1 = (acc[1] > 0) ? (double)acc[0] / (double)acc[1] : kNaN;
	M.T2 = (acc[3] > 0) ? (double)acc[2] / (double)acc[3] : kNaN;

	// TREND: наклон линейной регрессии RR по номеру диагонали (верхний треугольник). Последние
	// 10% диагоналей отброшены — там на линию приходятся единицы точек и RR_k шумит.
	{
		const int kFrom = cfg.theiler + 1;
		const int kTo = (int)std::floor(0.9 * (double)(n - 1));
		double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0, cnt = 0.0;
		for (int k = kFrom; k <= kTo; ++k) {
			const double rr = (double)dCount[(size_t)(k + n - 1)] / (double)(n - k);
			sx += k; sy += rr; sxy += (double)k * rr; sxx += (double)k * (double)k; cnt += 1.0;
		}
		if (cnt >= 2.0) {
			const double den = sxx - sx * sx / cnt;
			M.TREND = (den != 0.0) ? (sxy - sx * sy / cnt) / den : kNaN;
		}
		else M.TREND = kNaN;
	}

	M.clustering    = clustering;
	M.transitivity  = transitivity;
	M.network_valid = network_valid;

	out.ok = true;
	return true;
}
