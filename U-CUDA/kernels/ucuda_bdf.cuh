// BDF 1..5 переменного порядка и шага в форме DASSL (DDASTP, Brenan-Campbell-Petzold) — один текст для
// CPU (multistep_bdf.h, провайдер на SystemEvaluator) и GPU (NVRTC: Analysis — bdf_phase_kernel в
// NvrtcEngine, свипы Parametric — драйвер ниже под UCUDA_BDF_DRIVER вместо адаптивного).
//
// Ядро алгоритма — UcudaBdfCore и ucuda_bdf_start / ucuda_bdf_step / ucuda_bdf_interp, шаблоны по
// провайдеру Fn:
//   fn.res(y, t, F)   — F = [f(y, t); g(y, t)] (у ОДУ g-строк нет, nx = N);
//   fn.jac(y, t, J)   — J = d[f; g]/dy, построчно N x N.
// История — модифицированные разделённые разности phi[1..k+1] по реальным узлам; коэффициенты alpha,
// beta, gamma, sigma, psi пересчитываются после смены шага / порядка, история не масштабируется.
// Прогноз y = sum phi*_j, y' = sum gamma_j phi*_j; коррекция y' = y'_pred + cj (y - y_pred),
// x-строки y'_x - f = 0, z-строки g = 0, Ньютон с полным якобианом (cj I - f_y | g_y); err = ck ||e||;
// порядок k-1 / k / k+1 по оценкам DASSL, начальная фаза — порядок +1 и шаг x2; WRMS-норма с
// множителем h^(k-1) у переменной индекса k >= 2; вывод — интерполяция DDATRP (с производной).
// Рабочая память — ws размера ucuda_bdf_ws_size(N).
//
// Файл включается дважды-трижды (сначала ядро, затем — разделы драйвера и LLE/LS под макросами),
// поэтому вместо #pragma once — свои сторожа разделов.

#ifndef UCUDA_BDF_CORE_DONE
#define UCUDA_BDF_CORE_DONE

#ifdef __CUDACC__
#define UCUDA_BDF_HD __host__ __device__
#else
#define UCUDA_BDF_HD
#include <cmath>
#endif

#define UCUDA_BDF_MAXK 5
#define UCUDA_BDF_WS(N) ((UCUDA_BDF_MAXK + 3) * (N) + 9 * (N) + 2 * (N) * (N))

struct UcudaBdfParams {
    numb rtol;
    numb h0;      // 0 — автоматически
    numb hmax;    // <= 0 — без ограничения
    int  maxk;    // 1..5
    // h_min пользователя (> 0): шаг не меньше; попытка на нём, не прошедшая по ошибке, принимается
    // вынужденно (код 2 в журнале), как у адаптивного драйвера; Ньютон, не сошедшийся на h_min, — отказ
    // (hmin_fail). 0 — только защитный порог 10 ulp(t) (свипы).
    numb hmin;
};

struct UcudaBdfStats {
    long long nacc, nrej, nforced, nrhs, njac, nnewton_fail;
    numb hmin, hmax, hsum;
    int max_q, diverged, truncated, hmin_fail;
};

// Состояние DASSL (без рабочей памяти — она отдельным массивом ws).
struct UcudaBdfCore {
    numb alpha[UCUDA_BDF_MAXK + 3], beta[UCUDA_BDF_MAXK + 3], gamma[UCUDA_BDF_MAXK + 3],
         sigma[UCUDA_BDF_MAXK + 3], psi[UCUDA_BDF_MAXK + 3];
    numb t, h, hold, hs;
    int  N, nx, maxk, k, kold, ns, iphase, nef;
    // Журнал попыток (только Analysis, ucuda_bdf_run): по 4 числа — t начала шага, h, err, код (1 принят,
    // 0 отвергнут), как у адаптивного драйвера. nullptr — не вести (свипы, клоны).
    numb* log;
    int   log_cap, log_n;
};

UCUDA_BDF_HD inline int ucuda_bdf_ws_size(int N) { return UCUDA_BDF_WS(N); }

// LU с выбором главного по столбцу и решение на месте; false — вырождена.
UCUDA_BDF_HD inline bool ucuda_bdf_lu_solve(numb* A, numb* b, int n) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i) if (fabs(A[i * n + k]) > fabs(A[p * n + k])) p = i;
        if (!(fabs(A[p * n + k]) > (numb)1e-300)) return false;
        if (p != k) {
            for (int j = 0; j < n; ++j) { const numb s = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = s; }
            const numb s = b[k]; b[k] = b[p]; b[p] = s;
        }
        for (int i = k + 1; i < n; ++i) {
            const numb m = A[i * n + k] / A[k * n + k];
            if (m == (numb)0) continue;
            for (int j = k; j < n; ++j) A[i * n + j] -= m * A[k * n + j];
            b[i] -= m * b[k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        numb r = b[k];
        for (int j = k + 1; j < n; ++j) r -= A[k * n + j] * b[j];
        b[k] = r / A[k * n + k];
    }
    return true;
}

// Раскладка ws: phi[(j)*N] j = 0..MAXK+2, F, R, y, ypr, yprp, ev, wt, dl, yo, J (N*N), A (N*N).
#define UCUDA_BDF_PHI(ws, N, j) ((ws) + (j) * (N))
#define UCUDA_BDF_V(ws, N, k)   ((ws) + (UCUDA_BDF_MAXK + 3) * (N) + (k) * (N))   // k: 0 F,1 R,2 y,3 ypr,4 yprp,5 ev,6 wt,7 dl,8 yo
#define UCUDA_BDF_J(ws, N)      ((ws) + (UCUDA_BDF_MAXK + 3) * (N) + 9 * (N))
#define UCUDA_BDF_A(ws, N)      (UCUDA_BDF_J(ws, N) + (N) * (N))

UCUDA_BDF_HD inline numb ucuda_bdf_wrms(const UcudaBdfCore& C, const numb* ws, const int* kidx, const numb* v) {
    const int N = C.N;
    const numb* wt = UCUDA_BDF_V(ws, N, 6);
    numb s2 = 0;
    for (int i = 0; i < N; ++i) {
        numb q = v[i] * wt[i];
        for (int e = 1; e < kidx[i]; ++e) q *= C.hs;
        s2 += q * q;
    }
    return sqrt(s2 / (numb)N);
}

UCUDA_BDF_HD inline bool ucuda_bdf_finite(const numb* v, int N) {
    for (int i = 0; i < N; ++i) if (!(fabs(v[i]) <= (numb)1e300)) return false;
    return true;
}

UCUDA_BDF_HD inline void ucuda_bdf_weights(const UcudaBdfCore& C, numb* ws, const numb* atol, const UcudaBdfParams& P,
                                           const numb* v) {
    numb* wt = UCUDA_BDF_V(ws, C.N, 6);
    for (int i = 0; i < C.N; ++i) wt[i] = (numb)1 / (P.rtol * fabs(v[i]) + atol[i]);
}

// Старт с (t0, y): project = 1 — сначала z из g(x, z) = 0 Ньютоном при фиксированном x (точка свипа,
// клон LLE: z прежний для других параметров / возмущён); у индекса >= 2 dg/dz вырожден — z остаётся.
// y — вход и выход (после проекции). span — длина интервала для начального шага.
template <class Fn>
UCUDA_BDF_HD inline void ucuda_bdf_start(UcudaBdfCore& C, Fn& fn, int N, int nx, const int* kidx, numb* y0,
                                         const numb* atol, const UcudaBdfParams& P, numb t0, numb span, int project,
                                         numb* ws, UcudaBdfStats& st) {
    C.N = N; C.nx = nx;
    C.log = nullptr; C.log_cap = 0; C.log_n = 0;
    C.maxk = P.maxk < 1 ? 1 : (P.maxk > UCUDA_BDF_MAXK ? UCUDA_BDF_MAXK : P.maxk);
    for (int j = 0; j < UCUDA_BDF_MAXK + 3; ++j) C.alpha[j] = C.beta[j] = C.gamma[j] = C.sigma[j] = C.psi[j] = 0;
    numb* F = UCUDA_BDF_V(ws, N, 0);
    numb* R = UCUDA_BDF_V(ws, N, 1);
    numb* yp = UCUDA_BDF_V(ws, N, 4);
    numb* J = UCUDA_BDF_J(ws, N);
    numb* A = UCUDA_BDF_A(ws, N);
    const int nz = N - nx;
    if (project && nz > 0) {
        for (int it = 0; it < 20; ++it) {
            fn.res(y0, t0, F);
            ++st.nrhs;
            numb gn = 0, zn = 0;
            for (int q = 0; q < nz; ++q) { R[q] = -F[nx + q]; gn = fabs(F[nx + q]) > gn ? fabs(F[nx + q]) : gn; }
            if (!(gn <= (numb)1e300) || gn == 0) break;
            fn.jac(y0, t0, J);
            ++st.njac;
            for (int p = 0; p < nz; ++p) for (int q = 0; q < nz; ++q) A[p * nz + q] = J[(nx + p) * N + nx + q];
            if (!ucuda_bdf_lu_solve(A, R, nz)) break;
            numb dn = 0;
            for (int q = 0; q < nz; ++q) {
                y0[nx + q] += R[q];
                dn = fabs(R[q]) > dn ? fabs(R[q]) : dn;
                zn = fabs(y0[nx + q]) > zn ? fabs(y0[nx + q]) : zn;
            }
            if (dn <= (numb)1e-14 * ((numb)1 + zn)) break;
        }
    }
    fn.res(y0, t0, F);
    ++st.nrhs;
    for (int i = 0; i < N; ++i) yp[i] = i < nx ? F[i] : (numb)0;
    ucuda_bdf_weights(C, ws, atol, P, y0);
    C.hs = 1;
    numb h = P.h0;
    if (!(h > 0)) {   // DASSL: h0 = min(1e-3 |tout - t|, 0.5 / ||y'||)
        const numb ypn = ucuda_bdf_wrms(C, ws, kidx, yp);
        h = (numb)1e-3 * (span > (numb)1e-12 ? span : (numb)1e-12);
        if (ypn > (numb)0.5 / h) h = (numb)0.5 / ypn;
    }
    if (P.hmax > 0 && h > P.hmax) h = P.hmax;
    for (int i = 0; i < N; ++i) { UCUDA_BDF_PHI(ws, N, 1)[i] = y0[i]; UCUDA_BDF_PHI(ws, N, 2)[i] = h * yp[i]; }
    C.psi[1] = h;
    C.t = t0; C.h = h; C.hold = 0;
    C.k = 1; C.kold = 0; C.ns = 0; C.iphase = 0; C.nef = 0;
}

// Один принятый шаг (попытки повторяются после отказов). Шаг не заходит за tstop. 1 — принят;
// 0 — h ниже h_min (st.hmin_fail) или решение не конечно (st.diverged). y = phi1 после шага.
template <class Fn>
UCUDA_BDF_HD inline int ucuda_bdf_step(UcudaBdfCore& C, Fn& fn, const int* kidx, const numb* atol,
                                       const UcudaBdfParams& P, numb tstop, numb* ws, UcudaBdfStats& st) {
    const int N = C.N, nx = C.nx, MAXK = C.maxk;
    numb* F = UCUDA_BDF_V(ws, N, 0);
    numb* R = UCUDA_BDF_V(ws, N, 1);
    numb* y = UCUDA_BDF_V(ws, N, 2);
    numb* ypr = UCUDA_BDF_V(ws, N, 3);
    numb* yprp = UCUDA_BDF_V(ws, N, 4);
    numb* ev = UCUDA_BDF_V(ws, N, 5);
    numb* dl = UCUDA_BDF_V(ws, N, 7);
    numb* J = UCUDA_BDF_J(ws, N);
    numb* A = UCUDA_BDF_A(ws, N);
    numb* alpha = C.alpha; numb* beta = C.beta; numb* gamma = C.gamma; numb* sigma = C.sigma; numb* psi = C.psi;
    const numb eps = (numb)2.220446049250313e-16;
    for (;;) {
        const numb hmin = 10 * eps * (fabs(C.t) > 1 ? fabs(C.t) : (numb)1);
        const numb hfloor = P.hmin > hmin ? P.hmin : (numb)0;   // h_min пользователя
        if (C.h < hfloor) C.h = hfloor;
        if (C.t + C.h > tstop) C.h = tstop - C.t;   // ровно до tstop (DASSL: TSTOP)
        if (!(C.h >= hmin)) {
            if (C.t + hmin > tstop && tstop > C.t) C.h = tstop - C.t;   // хвост короче h_min — им и шагнуть
            else { st.hmin_fail = 1; st.diverged = 1; return 0; }
        }
        const numb h = C.h;
        const int k = C.k, kp1 = k + 1, kp2 = k + 2, km1 = k - 1;
        // --- коэффициенты (DDASTP, блок 200) ---
        if (h != C.hold || k != C.kold) C.ns = 0;
        C.ns = C.ns + 1 < C.kold + 2 ? C.ns + 1 : C.kold + 2;
        const int ns = C.ns, nsp1 = ns + 1;
        if (kp1 >= ns) {
            beta[1] = 1; alpha[1] = 1; gamma[1] = 0; sigma[1] = 1;
            numb temp1 = h;
            for (int i = 2; i <= kp1; ++i) {
                const numb temp2 = psi[i - 1];
                psi[i - 1] = temp1;
                beta[i] = beta[i - 1] * psi[i - 1] / temp2;
                temp1 = temp2 + h;
                alpha[i] = h / temp1;
                sigma[i] = (numb)(i - 1) * sigma[i - 1] * alpha[i];
                gamma[i] = gamma[i - 1] + alpha[i - 1] / h;
            }
            psi[kp1] = temp1;
        }
        numb alphas = 0, alpha0 = 0;
        for (int i = 1; i <= k; ++i) { alphas -= (numb)1 / (numb)i; alpha0 -= alpha[i]; }
        const numb cj = -alphas / h;
        numb ck = fabs(alpha[kp1] + alphas - alpha0);
        if (alpha[kp1] > ck) ck = alpha[kp1];
        if (kp1 >= nsp1)
            for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) UCUDA_BDF_PHI(ws, N, j)[i] *= beta[j];
        // --- прогноз ---
        const numb told = C.t;
        const numb t = told + h;
        for (int i = 0; i < N; ++i) {
            numb a0 = 0, a1 = 0;
            for (int j = 1; j <= kp1; ++j) { a0 += UCUDA_BDF_PHI(ws, N, j)[i]; a1 += gamma[j] * UCUDA_BDF_PHI(ws, N, j)[i]; }
            ypr[i] = a0; yprp[i] = a1; y[i] = a0;
        }
        C.hs = h;
        // --- коррекция: Ньютон ---
        bool conv = false;
        for (int it = 0; it < 6; ++it) {
            fn.res(y, t, F);
            ++st.nrhs;
            if (!ucuda_bdf_finite(F, N)) break;
            for (int i = 0; i < N; ++i)
                R[i] = i < nx ? -(yprp[i] + cj * (y[i] - ypr[i]) - F[i]) : -F[i];
            fn.jac(y, t, J);
            ++st.njac;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    A[i * N + j] = i < nx ? (i == j ? cj : (numb)0) - J[i * N + j] : J[i * N + j];
            if (!ucuda_bdf_lu_solve(A, R, N)) break;
            for (int i = 0; i < N; ++i) y[i] += R[i];
            if (!ucuda_bdf_finite(y, N)) break;
            if (ucuda_bdf_wrms(C, ws, kidx, R) < (numb)1e-3) { conv = true; break; }
        }
        numb enorm = (numb)1e300, erkm1 = 0, terkm1 = 0, est = 0, terk = 0;
        int knew = k;
        if (conv) {
            for (int i = 0; i < N; ++i) ev[i] = y[i] - ypr[i];
            enorm = ucuda_bdf_wrms(C, ws, kidx, ev);
            const numb erk = sigma[k + 1] * enorm;
            terk = (numb)(k + 1) * erk;
            est = erk;
            if (k > 1) {   // оценка для порядка k-1 (DDASTP, блок 400)
                for (int i = 0; i < N; ++i) dl[i] = UCUDA_BDF_PHI(ws, N, kp1)[i] + ev[i];
                erkm1 = sigma[k] * ucuda_bdf_wrms(C, ws, kidx, dl);
                terkm1 = (numb)k * erkm1;
                bool lower;
                if (k > 2) {
                    for (int i = 0; i < N; ++i) dl[i] += UCUDA_BDF_PHI(ws, N, k)[i];
                    const numb terkm2 = (numb)(k - 1) * sigma[k - 1] * ucuda_bdf_wrms(C, ws, kidx, dl);
                    lower = (terkm1 > terkm2 ? terkm1 : terkm2) <= terk;
                } else lower = terkm1 <= (numb)0.5 * terk;
                if (lower) { knew = km1; est = erkm1; }
            }
        } else ++st.nnewton_fail;
        const numb errv = conv ? ck * enorm : (numb)1e300;
        const bool at_floor = hfloor > 0 && h <= hfloor;
        const bool forced = at_floor && conv && !(errv <= (numb)1);   // на h_min ошибку не исправить — принять
        if (C.log != nullptr && C.log_n < C.log_cap) {
            numb* r = C.log + 4 * C.log_n++;
            r[0] = C.t; r[1] = h; r[2] = errv; r[3] = errv <= (numb)1 ? (numb)1 : (forced ? (numb)2 : (numb)0);
        }
        if (at_floor && !conv) { st.hmin_fail = 1; st.diverged = 1; return 0; }   // Ньютон не сошёлся и на h_min
        if (forced) ++st.nforced;
        if (!(errv <= (numb)1) && !forced) {
            // --- отказ: история назад (DDASTP, блок 690) ---
            if (kp1 >= nsp1)
                for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) UCUDA_BDF_PHI(ws, N, j)[i] /= beta[j];
            for (int i = 2; i <= kp1; ++i) psi[i - 1] = psi[i] - h;
            ++st.nrej;
            if (!conv) C.h *= (numb)0.25;
            else {
                ++C.nef;
                if (C.nef == 1) {
                    C.k = knew;
                    numb r = (numb)0.9 * pow((numb)2 * est + (numb)1e-4, (numb)-1 / (numb)(C.k + 1));
                    r = r < (numb)0.25 ? (numb)0.25 : (r > (numb)0.9 ? (numb)0.9 : r);
                    C.h *= r;
                } else if (C.nef == 2) { C.k = knew; C.h *= (numb)0.25; }
                else { C.k = 1; C.h *= (numb)0.25; }
            }
            C.iphase = 1;
            C.hold = 0;
            continue;
        }
        // --- принят (DDASTP, блок 500) ---
        C.nef = 0;
        C.t = t;
        C.kold = k;
        C.hold = h;
        ++st.nacc;
        st.hmin = st.nacc == 1 ? h : (h < st.hmin ? h : st.hmin);
        if (h > st.hmax) st.hmax = h;
        st.hsum += h;
        if (k > st.max_q) st.max_q = k;
        if (knew == km1 || k == MAXK) C.iphase = 1;
        numb hnew = h;
        if (C.iphase == 0) {
            C.k = kp1;
            hnew = 2 * h;
        } else {
            if (knew == km1) { C.k = km1; est = erkm1; }
            else if (kp1 <= ns && k < MAXK) {
                for (int i = 0; i < N; ++i) dl[i] = ev[i] - UCUDA_BDF_PHI(ws, N, kp2)[i];
                const numb terkp1 = ucuda_bdf_wrms(C, ws, kidx, dl), erkp1 = terkp1 / (numb)(k + 2);
                if (k > 1) {
                    if (terkm1 <= (terk < terkp1 ? terk : terkp1)) { C.k = km1; est = erkm1; }
                    else if (!(terkp1 >= terk)) { C.k = kp1; est = erkp1; }
                } else if (!(terkp1 >= (numb)0.5 * terk)) { C.k = kp1; est = erkp1; }
            }
            const numb r = pow((numb)2 * est + (numb)1e-4, (numb)-1 / (numb)(C.k + 1));
            if (r >= (numb)2) hnew = 2 * h;
            else if (r <= (numb)1) hnew = h * (r < (numb)0.5 ? (numb)0.5 : (r > (numb)0.9 ? (numb)0.9 : r));
        }
        if (P.hmax > 0 && hnew > P.hmax) hnew = P.hmax;
        // разности для следующего шага (kp1, kp2 — порядка этого шага)
        if (k < MAXK) for (int i = 0; i < N; ++i) UCUDA_BDF_PHI(ws, N, kp2)[i] = ev[i];
        for (int i = 0; i < N; ++i) UCUDA_BDF_PHI(ws, N, kp1)[i] += ev[i];
        for (int j = kp1 - 1; j >= 1; --j)
            for (int i = 0; i < N; ++i) UCUDA_BDF_PHI(ws, N, j)[i] += UCUDA_BDF_PHI(ws, N, j + 1)[i];
        if (!ucuda_bdf_finite(y, N)) { st.diverged = 1; return 0; }
        ucuda_bdf_weights(C, ws, atol, P, y);
        C.h = hnew;
        return 1;
    }
}

// y(tout) и y'(tout) (ypo может быть nullptr) по истории последнего шага (DDATRP); tout в его пределах.
UCUDA_BDF_HD inline void ucuda_bdf_interp(const UcudaBdfCore& C, const numb* ws, numb tout, numb* yo, numb* ypo) {
    const int N = C.N;
    const numb temp1 = tout - C.t;
    for (int i = 0; i < N; ++i) { yo[i] = UCUDA_BDF_PHI(ws, N, 1)[i]; if (ypo) ypo[i] = 0; }
    numb c = 1, d = 0, gam = temp1 / C.psi[1];
    for (int j = 2; j <= C.kold + 1; ++j) {
        d = d * gam + c / C.psi[j - 1];
        c *= gam;
        gam = (temp1 + C.psi[j - 1]) / C.psi[j];
        for (int i = 0; i < N; ++i) {
            yo[i] += c * UCUDA_BDF_PHI(ws, N, j)[i];
            if (ypo) ypo[i] += d * UCUDA_BDF_PHI(ws, N, j)[i];
        }
    }
}

// Траектория от y0 (t = 0) до t_skip + t_rec (Analysis): raw = 0 — сетка total точек через dt от
// t_skip, raw = 1 — узлы шага после t_skip (не больше max_pts). Провайдер ещё и выводит:
// fn.emit(i, t, y), fn.cancelled() (раз в 1024 шага). Возвращает число выведенных точек.
template <class Fn>
UCUDA_BDF_HD int ucuda_bdf_run(Fn& fn, int N, int nx, const int* kidx, const numb* y0, const numb* atol,
                               const UcudaBdfParams& P, numb t_skip, numb t_rec, numb dt, int total, int raw,
                               int max_pts, numb* ws, UcudaBdfStats& st,
                               numb* log = nullptr, int log_cap = 0, int* log_n = nullptr) {
    st.nacc = st.nrej = st.nforced = st.nrhs = st.njac = st.nnewton_fail = 0;
    st.hmin = st.hmax = st.hsum = 0;
    st.max_q = st.diverged = st.truncated = st.hmin_fail = 0;
    const numb tend = t_skip + t_rec;
    UcudaBdfCore C;
    numb* y = UCUDA_BDF_V(ws, N, 2);
    numb* yo = UCUDA_BDF_V(ws, N, 8);
    for (int i = 0; i < N; ++i) yo[i] = y0[i];
    ucuda_bdf_start(C, fn, N, nx, kidx, yo, atol, P, (numb)0, tend, 0, ws, st);
    C.log_cap = log_cap;   // журнал попыток (start его сбрасывает)
    int nout = 0;
    if (t_skip <= 0) fn.emit(nout++, (numb)0, yo);
    long long iter = 0;
    while (C.t < tend && (raw ? nout < max_pts : nout < total)) {
        if ((++iter & 1023) == 0 && fn.cancelled()) break;
        // Журнал — с шага, который заходит за t_skip: переходный процесс на графике h(t) не нужен
        // (остальные проекции его тоже не показывают) и не съедает log_cap.
        C.log = C.t + C.h > t_skip ? log : nullptr;
        if (!ucuda_bdf_step(C, fn, kidx, atol, P, (numb)1e300, ws, st)) break;
        if (raw) {
            if (C.t >= t_skip) fn.emit(nout++, C.t, y);
        } else {
            while (nout < total) {
                const numb ts = t_skip + (numb)nout * dt;
                if (ts > C.t) break;
                ucuda_bdf_interp(C, ws, ts, yo, nullptr);
                fn.emit(nout++, ts, yo);
            }
        }
    }
    if (raw && nout >= max_pts && C.t < tend) st.truncated = 1;
    if (log_n != nullptr) *log_n = C.log_n;
    return nout;
}

#endif // UCUDA_BDF_CORE_DONE

// ---- Драйвер с интерфейсом адаптивного шага (свипы Parametric на BDF) ---------------------------
// Включает ucuda_adaptive.cuh вместо своего драйвера при UCUDA_BDF_DRIVER: ядра свипов (БД, бассейны,
// Metrics, LLE/LS) зовут те же ucuda_ad_init / restart / step / eval / advance_to и читают S.X, S.t,
// S.F0 (производная в узле — DDATRP), S.st, S.diverged. Провайдер — UcudaKrsFns модуля: rhs = [f; g],
// dprep с D = якобиан (так движок кладёт тела BDF в слоты адаптивного модуля). UCUDA_BDF_NX — число
// дифференциальных переменных, UCUDA_BDF_KIDX — индексы переменных списком инициализации.
// Точка свипа стартует с проекцией z на g = 0 (связь зависит от параметров точки).
#if defined(UCUDA_BDF_DRIVER) && defined(UCUDA_BDF_DRIVER_SECTION) && !defined(UCUDA_BDF_DRIVER_DONE)
#define UCUDA_BDF_DRIVER_DONE
#ifndef UCUDA_BDF_NX
#define UCUDA_BDF_NX AMOUNTOFX
#endif

template <class K>
struct UcudaBdfKrs {
    const K& k; const numb* a;
    UCUDA_HD void res(const numb* y, numb t, numb* F) { (void)t; k.rhs(y, UCUDA_TA(t) a, F); }
    UCUDA_HD void jac(const numb* y, numb t, numb* J) { (void)t; k.dprep(y, y, y, y, UCUDA_TA(t) a, (numb)0, (numb*)nullptr, J); }
};

struct UcudaAdaptState {
    numb t, h, hfree;
    numb X[AMOUNTOFX];
    numb F0[AMOUNTOFX];                            // y'(t) — производная интерполянта в узле
    int  n, diverged, log_n;
    UcudaAdaptStats st;
    UcudaBdfStats bst;
    UcudaBdfCore C;
    UcudaBdfParams BP;
    numb atol[AMOUNTOFX];
    int  kidx[AMOUNTOFX];
    numb ws[UCUDA_BDF_WS(AMOUNTOFX)];
};

UCUDA_HD inline void ucuda_bdf_sync(UcudaAdaptState& S) {
    S.t = S.C.t;
    S.h = S.C.h; S.hfree = S.C.h;
    S.st.nacc = (unsigned long long)S.bst.nacc; S.st.nrej = (unsigned long long)S.bst.nrej;
    S.st.nforced = 0; S.st.nrhs = (unsigned long long)S.bst.nrhs;
    S.st.hmin = S.bst.hmin; S.st.hmax = S.bst.hmax; S.st.hsum = S.bst.hsum;
}

template <class K>
UCUDA_HD inline void ucuda_bdf_begin(UcudaAdaptState& S, const K& k, numb t0, const numb* a,
                                     const UcudaAdaptParams& P, numb h0) {
    S.BP.rtol = P.rtol;
    S.BP.h0 = h0;
    S.BP.hmax = (P.hmax > 0 && P.hmax <= UCUDA_AD_BIG) ? P.hmax : (numb)0;
    S.BP.maxk = UCUDA_BDF_MAXK;
    S.BP.hmin = 0;   // свипы: только защитный порог 10 ulp(t), как было
    for (int i = 0; i < AMOUNTOFX; ++i) S.atol[i] = P.atol[i];
#ifdef UCUDA_BDF_KIDX
    const int kk[AMOUNTOFX] = UCUDA_BDF_KIDX;
    for (int i = 0; i < AMOUNTOFX; ++i) S.kidx[i] = kk[i] > 0 ? kk[i] : 1;
#else
    for (int i = 0; i < AMOUNTOFX; ++i) S.kidx[i] = 1;
#endif
    S.bst.nacc = S.bst.nrej = S.bst.nforced = S.bst.nrhs = S.bst.njac = S.bst.nnewton_fail = 0;
    S.bst.hmin = S.bst.hmax = S.bst.hsum = 0;
    S.bst.max_q = S.bst.diverged = S.bst.truncated = S.bst.hmin_fail = 0;
    UcudaBdfKrs<K> fn{ k, a };
    ucuda_bdf_start(S.C, fn, AMOUNTOFX, UCUDA_BDF_NX, S.kidx, S.X, S.atol, S.BP, t0,
                    P.span > 0 ? P.span : (numb)1, 1, S.ws, S.bst);
    // производная в начале: y'0 из прогноза — f по x, 0 по z
    const numb* F = UCUDA_BDF_V(S.ws, AMOUNTOFX, 0);
    for (int i = 0; i < AMOUNTOFX; ++i) S.F0[i] = i < UCUDA_BDF_NX ? F[i] : (numb)0;
    S.diverged = 0;
    S.log_n = 0;
    ucuda_bdf_sync(S);
}

template <class K>
UCUDA_HD inline void ucuda_ad_init(UcudaAdaptState& S, const K& k, int n, const numb* X0, numb t0,
                                   const numb* a, const UcudaAdaptParams& P,
                                   numb* log = nullptr, int log_cap = 0) {
    (void)log; (void)log_cap;
    S.n = n;
    for (int i = 0; i < AMOUNTOFX; ++i) S.X[i] = X0[i];
    ucuda_bdf_begin(S, k, t0, a, P, P.h0 > 0 ? P.h0 : (numb)0);
}

// Продолжение (continuation): старт с того же X при новых параметрах, история — заново (порядок 1),
// начальный шаг — прежний предложенный.
template <class K>
UCUDA_HD inline void ucuda_ad_restart(UcudaAdaptState& S, const K& k, numb t0,
                                      const numb* a, const UcudaAdaptParams& P) {
    const numb hc = S.hfree;
    ucuda_bdf_begin(S, k, t0, a, P, (hc > 0 && hc <= UCUDA_AD_BIG) ? hc : (P.h0 > 0 ? P.h0 : (numb)0));
}

template <class K>
UCUDA_HD inline void ucuda_ad_step(UcudaAdaptState& S, const K& k, const numb* a,
                                   const UcudaAdaptParams& P, numb tEnd) {
    (void)P;
    UcudaBdfKrs<K> fn{ k, a };
    if (!ucuda_bdf_step(S.C, fn, S.kidx, S.atol, S.BP, tEnd, S.ws, S.bst)) { S.diverged = 1; ucuda_bdf_sync(S); return; }
    ucuda_bdf_interp(S.C, S.ws, S.C.t, S.X, S.F0);
    ucuda_bdf_sync(S);
}

// y(tt) для tt в пределах последнего шага — интерполяция DDATRP.
template <class K>
UCUDA_HD inline void ucuda_ad_eval(UcudaAdaptState& S, const K& k, const numb* a,
                                   const UcudaAdaptParams& P, numb tt, numb* Yo) {
    (void)k; (void)a; (void)P;
    if (tt >= S.t) { for (int i = 0; i < AMOUNTOFX; ++i) Yo[i] = S.X[i]; return; }
    ucuda_bdf_interp(S.C, S.ws, tt, Yo, nullptr);
}

template <class K>
UCUDA_HD inline void ucuda_ad_advance_to(UcudaAdaptState& S, const K& k, const numb* a,
                                         const UcudaAdaptParams& P, numb tt, numb tEnd, numb* Yo) {
    while (S.t < tt && !S.diverged) ucuda_ad_step(S, k, a, P, tEnd);
    ucuda_ad_eval(S, k, a, P, tt, Yo);
}
#endif // UCUDA_BDF_DRIVER_SECTION

// ---- LLE / LS на BDF: клоны — свои траектории BDF -------------------------------------------------
// Включается из раздела LLE/LS ucuda_adaptive.cuh вместо UcudaLyapClones / ucuda_lyap_point, когда
// модуль собран с UCUDA_BDF_DRIVER. У многошагового метода нельзя вести клоны "тем же шагом" (их
// история своя), поэтому каждый блок NT: опорная траектория шагает до границы блока ровно (tstop),
// каждый клон стартует заново (порядок 1, проекция z на g = 0) с перенормированного состояния и
// шагает до той же границы; затем — та же перенормировка, что у адаптивного шага.
#if defined(UCUDA_BDF_DRIVER) && defined(UCUDA_BDF_LYAP_SECTION) && !defined(UCUDA_BDF_LYAP_DONE)
#define UCUDA_BDF_LYAP_DONE

template <int NC>
struct UcudaLyapClones {
    numb y[NC * AMOUNTOFX];
    bool active;
    UcudaBdfCore C;                                // клоны идут по очереди — ядро и память общие
    numb ws[UCUDA_BDF_WS(AMOUNTOFX)];
};

template <int NC, class K>
UCUDA_HD inline int ucuda_lyap_point(UcudaAdaptState& S, const K& Kf, const numb* a, const UcudaAdaptParams& P,
                                     UcudaLyapClones<NC>& cl, const bool carried, const numb tTr, const numb NT,
                                     const int nBlocks, const int nWarm, const numb eps, const int renorm,
                                     const numb maxValue, const int seq, numb* res,
                                     const volatile int* cancelFlag) {
    (void)renorm;
    constexpr int n = AMOUNTOFX;
    numb z[NC * n], acc[NC], accWarm[NC];
    for (int c = 0; c < NC; ++c) { acc[c] = 0; accWarm[c] = 0; }
    if (!carried) {
        cl.active = false;
        while (S.t < tTr && !S.diverged) ucuda_ad_step(S, Kf, a, P, tTr);
        if (S.diverged || ucuda_lyap_out(S.X, maxValue)) return 0;
        ucuda_lyap_init_clones<NC>(S.X, eps, seq, cl.y, z);
    }
    cl.active = true;
    int nW = nWarm;
    if (carried) { const int ns = (int)(tTr / NT); if (ns > nW) nW = ns; }
    const int nAll = nBlocks + nW;
    const numb t0 = S.t;
    numb tAcc0 = t0;
    UcudaBdfKrs<K> fn{ Kf, a };
    UcudaBdfStats cst;
    for (int kb = 1; kb <= nAll; ++kb) {
        const numb tStart = S.t, tb = t0 + (numb)kb * NT;
        while (S.t < tb && !S.diverged) ucuda_ad_step(S, Kf, a, P, tb);
        if (S.diverged || ucuda_lyap_out(S.X, maxValue)) return 0;
        for (int c = 0; c < NC; ++c) {
            numb* yc = cl.y + c * n;
            cst.nacc = cst.nrej = cst.nforced = cst.nrhs = cst.njac = cst.nnewton_fail = 0;
            cst.hmin = cst.hmax = cst.hsum = 0;
            cst.max_q = cst.diverged = cst.truncated = cst.hmin_fail = 0;
            ucuda_bdf_start(cl.C, fn, n, UCUDA_BDF_NX, S.kidx, yc, S.atol, S.BP, tStart, NT, 1, cl.ws, cst);
            while (cl.C.t < tb)
                if (!ucuda_bdf_step(cl.C, fn, S.kidx, S.atol, S.BP, tb, cl.ws, cst)) return 0;
            for (int i = 0; i < n; ++i) yc[i] = UCUDA_BDF_V(cl.ws, n, 2)[i];
            if (ucuda_lyap_out(yc, maxValue)) return 0;
        }
        ucuda_lyap_renorm<NC>(S.X, eps, cl.y, z, kb > nW ? acc : accWarm);
        if (kb == nW) tAcc0 = S.t;
        if (UCUDA_AD_CANCELLED(cancelFlag)) return -1;
    }
    const numb tIntegrated = S.t - tAcc0;
    for (int c = 0; c < NC; ++c) res[c] = acc[c] / tIntegrated;
    return 1;
}
#endif // UCUDA_BDF_LYAP_SECTION
