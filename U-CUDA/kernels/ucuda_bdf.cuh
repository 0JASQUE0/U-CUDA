#pragma once
// BDF 1..5 переменного порядка и шага в форме DASSL (DDASTP, Brenan-Campbell-Petzold) — один текст для
// CPU (multistep_bdf.h, провайдер на SystemEvaluator) и GPU (NVRTC, провайдер на сгенерированных телах).
// Шаблон по провайдеру Fn:
//   fn.res(y, t, F)   — F = [f(y, t); g(y, t)] (у ОДУ g-строк нет, nx = N);
//   fn.jac(y, t, J)   — J = d[f; g]/dy, построчно N x N;
//   fn.emit(i, t, y)  — точка вывода номер i;
//   fn.cancelled()    — запрос отмены (смотрится раз в 1024 шага).
// История — модифицированные разделённые разности phi[1..k+1] по реальным узлам; коэффициенты alpha,
// beta, gamma, sigma, psi пересчитываются после смены шага / порядка, история не масштабируется.
// Прогноз y = sum phi*_j, y' = sum gamma_j phi*_j; коррекция y' = y'_pred + cj (y - y_pred),
// x-строки y'_x - f = 0, z-строки g = 0, Ньютон с полным якобианом (cj I - f_y | g_y); err = ck ||e||;
// порядок k-1 / k / k+1 по оценкам DASSL, начальная фаза — порядок +1 и шаг x2; WRMS-норма с
// множителем h^(k-1) у переменной индекса k >= 2; вывод — интерполяция DDATRP.
// Рабочая память — ws размера ucuda_bdf_ws_size(N): ядро держит её в локальной памяти нити.

#ifdef __CUDACC__
#define UCUDA_BDF_HD __host__ __device__
#else
#define UCUDA_BDF_HD
#include <cmath>
#endif

#define UCUDA_BDF_MAXK 5

struct UcudaBdfParams {
    numb rtol;
    numb h0;      // 0 — автоматически
    numb hmax;    // <= 0 — без ограничения
    int  maxk;    // 1..5
};

struct UcudaBdfStats {
    long long nacc, nrej, nrhs, njac, nnewton_fail;
    numb hmin, hmax, hsum;
    int max_q, diverged, truncated, hmin_fail;
};

UCUDA_BDF_HD inline int ucuda_bdf_ws_size(int N) { return (UCUDA_BDF_MAXK + 3) * N + 9 * N + 2 * N * N; }

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

// Траектория от y0 (t = 0) до t_skip + t_rec. raw = 0 — сетка total точек через dt от t_skip,
// raw = 1 — узлы шага после t_skip (не больше max_pts). atol[N]. Возвращает число выведенных точек.
template <class Fn>
UCUDA_BDF_HD int ucuda_bdf_run(Fn& fn, int N, int nx, const int* kidx, const numb* y0, const numb* atol,
                               const UcudaBdfParams& P, numb t_skip, numb t_rec, numb dt, int total, int raw,
                               int max_pts, numb* ws, UcudaBdfStats& st) {
    st.nacc = st.nrej = st.nrhs = st.njac = st.nnewton_fail = 0;
    st.hmin = st.hmax = st.hsum = 0;
    st.max_q = st.diverged = st.truncated = st.hmin_fail = 0;
    const numb tend = t_skip + t_rec;
    const int MAXK = P.maxk < 1 ? 1 : (P.maxk > UCUDA_BDF_MAXK ? UCUDA_BDF_MAXK : P.maxk);
    numb* phi = ws;                                   // phi[j*N + i], j = 1..MAXK+2
    numb* F   = phi + (UCUDA_BDF_MAXK + 3) * N;
    numb* R   = F + N;
    numb* y   = R + N;
    numb* ypr = y + N;
    numb* yprp = ypr + N;
    numb* ev  = yprp + N;
    numb* wt  = ev + N;
    numb* dl  = wt + N;
    numb* yo  = dl + N;
    numb* J   = yo + N;
    numb* A   = J + N * N;
#define UCUDA_BDF_P(j) (phi + (j) * N)
    numb alpha[UCUDA_BDF_MAXK + 3], beta[UCUDA_BDF_MAXK + 3], gamma[UCUDA_BDF_MAXK + 3],
         sigma[UCUDA_BDF_MAXK + 3], psi[UCUDA_BDF_MAXK + 3];
    for (int j = 0; j < UCUDA_BDF_MAXK + 3; ++j) alpha[j] = beta[j] = gamma[j] = sigma[j] = psi[j] = 0;
    numb hs = 1;   // h для множителя h^(k-1) в норме
    auto wrms = [&](const numb* v) {
        numb s2 = 0;
        for (int i = 0; i < N; ++i) {
            numb q = v[i] * wt[i];
            for (int e = 1; e < kidx[i]; ++e) q *= hs;
            s2 += q * q;
        }
        return sqrt(s2 / (numb)N);
    };
    auto finite = [&](const numb* v) {
        for (int i = 0; i < N; ++i) if (!(fabs(v[i]) <= (numb)1e300)) return false;
        return true;
    };
    auto weights = [&](const numb* v) {
        for (int i = 0; i < N; ++i) wt[i] = (numb)1 / (P.rtol * fabs(v[i]) + atol[i]);
    };

    // старт: y'0 — f по x, 0 по z (первый шаг порядка 1 это переносит, как DASSL)
    for (int i = 0; i < N; ++i) y[i] = y0[i];
    fn.res(y, (numb)0, F);
    ++st.nrhs;
    for (int i = 0; i < N; ++i) yprp[i] = i < nx ? F[i] : (numb)0;
    weights(y);
    numb h = P.h0;
    if (!(h > 0)) {   // DASSL: h0 = min(1e-3 |tout - t|, 0.5 / ||y'||)
        const numb ypn = wrms(yprp);
        h = (numb)1e-3 * (tend > (numb)1e-12 ? tend : (numb)1e-12);
        if (ypn > (numb)0.5 / h) h = (numb)0.5 / ypn;
    }
    if (P.hmax > 0 && h > P.hmax) h = P.hmax;
    for (int i = 0; i < N; ++i) { UCUDA_BDF_P(1)[i] = y[i]; UCUDA_BDF_P(2)[i] = h * yprp[i]; }
    psi[1] = h;

    int k = 1, kold = 0, ns = 0, iphase = 0, nef = 0, nout = 0;
    numb t = 0, hold = 0;
    const numb eps = (numb)2.220446049250313e-16;
    if (t_skip <= 0) { fn.emit(nout++, (numb)0, y); }
    long long iter = 0;
    while (t < tend && (raw ? nout < max_pts : nout < total)) {
        if ((++iter & 1023) == 0 && fn.cancelled()) break;
        const numb hmin = 10 * eps * (fabs(t) > 1 ? fabs(t) : (numb)1);
        if (h < hmin) { st.hmin_fail = 1; st.diverged = 1; break; }
        const int kp1 = k + 1, kp2 = k + 2, km1 = k - 1;
        // --- коэффициенты (DDASTP, блок 200) ---
        if (h != hold || k != kold) ns = 0;
        ns = ns + 1 < kold + 2 ? ns + 1 : kold + 2;
        const int nsp1 = ns + 1;
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
            for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) UCUDA_BDF_P(j)[i] *= beta[j];
        // --- прогноз ---
        const numb told = t;
        t = told + h;
        for (int i = 0; i < N; ++i) {
            numb a0 = 0, a1 = 0;
            for (int j = 1; j <= kp1; ++j) { a0 += UCUDA_BDF_P(j)[i]; a1 += gamma[j] * UCUDA_BDF_P(j)[i]; }
            ypr[i] = a0; yprp[i] = a1; y[i] = a0;
        }
        hs = h;
        // --- коррекция: Ньютон ---
        bool conv = false;
        for (int it = 0; it < 6; ++it) {
            fn.res(y, t, F);
            ++st.nrhs;
            if (!finite(F)) break;
            for (int i = 0; i < N; ++i)
                R[i] = i < nx ? -(yprp[i] + cj * (y[i] - ypr[i]) - F[i]) : -F[i];
            fn.jac(y, t, J);
            ++st.njac;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    A[i * N + j] = i < nx ? (i == j ? cj : (numb)0) - J[i * N + j] : J[i * N + j];
            if (!ucuda_bdf_lu_solve(A, R, N)) break;
            for (int i = 0; i < N; ++i) y[i] += R[i];
            if (!finite(y)) break;
            if (wrms(R) < (numb)1e-3) { conv = true; break; }
        }
        numb enorm = (numb)1e300, erkm1 = 0, terkm1 = 0, est = 0, terk = 0;
        int knew = k;
        if (conv) {
            for (int i = 0; i < N; ++i) ev[i] = y[i] - ypr[i];
            enorm = wrms(ev);
            const numb erk = sigma[k + 1] * enorm;
            terk = (numb)(k + 1) * erk;
            est = erk;
            if (k > 1) {   // оценка для порядка k-1 (DDASTP, блок 400)
                for (int i = 0; i < N; ++i) dl[i] = UCUDA_BDF_P(kp1)[i] + ev[i];
                erkm1 = sigma[k] * wrms(dl);
                terkm1 = (numb)k * erkm1;
                bool lower;
                if (k > 2) {
                    for (int i = 0; i < N; ++i) dl[i] += UCUDA_BDF_P(k)[i];
                    const numb terkm2 = (numb)(k - 1) * sigma[k - 1] * wrms(dl);
                    lower = (terkm1 > terkm2 ? terkm1 : terkm2) <= terk;
                } else lower = terkm1 <= (numb)0.5 * terk;
                if (lower) { knew = km1; est = erkm1; }
            }
        } else ++st.nnewton_fail;
        const numb errv = conv ? ck * enorm : (numb)1e300;
        if (!(errv <= (numb)1)) {
            // --- отказ: история назад (DDASTP, блок 690) ---
            t = told;
            if (kp1 >= nsp1)
                for (int j = nsp1; j <= kp1; ++j) for (int i = 0; i < N; ++i) UCUDA_BDF_P(j)[i] /= beta[j];
            for (int i = 2; i <= kp1; ++i) psi[i - 1] = psi[i] - h;
            ++st.nrej;
            if (!conv) h *= (numb)0.25;
            else {
                ++nef;
                if (nef == 1) {
                    k = knew;
                    numb r = (numb)0.9 * pow((numb)2 * est + (numb)1e-4, (numb)-1 / (numb)(k + 1));
                    r = r < (numb)0.25 ? (numb)0.25 : (r > (numb)0.9 ? (numb)0.9 : r);
                    h *= r;
                } else if (nef == 2) { k = knew; h *= (numb)0.25; }
                else { k = 1; h *= (numb)0.25; }
            }
            iphase = 1;
            hold = 0;
            continue;
        }
        // --- принят (DDASTP, блок 500) ---
        nef = 0;
        const int kstep = k;
        kold = k;
        hold = h;
        ++st.nacc;
        st.hmin = st.nacc == 1 ? h : (h < st.hmin ? h : st.hmin);
        if (h > st.hmax) st.hmax = h;
        st.hsum += h;
        if (k > st.max_q) st.max_q = k;
        if (knew == km1 || k == MAXK) iphase = 1;
        numb hnew = h;
        if (iphase == 0) {
            k = kp1;
            hnew = 2 * h;
        } else {
            if (knew == km1) { k = km1; est = erkm1; }
            else if (kp1 <= ns && k < MAXK) {
                for (int i = 0; i < N; ++i) dl[i] = ev[i] - UCUDA_BDF_P(kp2)[i];
                const numb terkp1 = wrms(dl), erkp1 = terkp1 / (numb)(k + 2);
                if (k > 1) {
                    if (terkm1 <= (terk < terkp1 ? terk : terkp1)) { k = km1; est = erkm1; }
                    else if (!(terkp1 >= terk)) { k = kp1; est = erkp1; }
                } else if (!(terkp1 >= (numb)0.5 * terk)) { k = kp1; est = erkp1; }
            }
            const numb r = pow((numb)2 * est + (numb)1e-4, (numb)-1 / (numb)(k + 1));
            if (r >= (numb)2) hnew = 2 * h;
            else if (r <= (numb)1) hnew = h * (r < (numb)0.5 ? (numb)0.5 : (r > (numb)0.9 ? (numb)0.9 : r));
        }
        if (P.hmax > 0 && hnew > P.hmax) hnew = P.hmax;
        // разности для следующего шага (kp1, kp2 — порядка этого шага)
        if (kstep < MAXK) for (int i = 0; i < N; ++i) UCUDA_BDF_P(kp2)[i] = ev[i];
        for (int i = 0; i < N; ++i) UCUDA_BDF_P(kp1)[i] += ev[i];
        for (int j = kp1 - 1; j >= 1; --j) for (int i = 0; i < N; ++i) UCUDA_BDF_P(j)[i] += UCUDA_BDF_P(j + 1)[i];
        if (!finite(y)) { st.diverged = 1; break; }
        weights(y);
        // --- вывод: DDATRP по phi и psi этого шага ---
        if (raw) {
            if (t >= t_skip) fn.emit(nout++, t, y);
        } else {
            while (nout < total) {
                const numb ts = t_skip + (numb)nout * dt;
                if (ts > t) break;
                const numb temp1 = ts - t;
                for (int i = 0; i < N; ++i) yo[i] = UCUDA_BDF_P(1)[i];
                numb c = 1, gam = temp1 / psi[1];
                for (int j = 2; j <= kstep + 1; ++j) {
                    c *= gam;
                    gam = (temp1 + psi[j - 1]) / psi[j];
                    for (int i = 0; i < N; ++i) yo[i] += c * UCUDA_BDF_P(j)[i];
                }
                fn.emit(nout++, ts, yo);
            }
        }
        h = hnew;
    }
#undef UCUDA_BDF_P
    if (raw && nout >= max_pts && t < tend) st.truncated = 1;
    return nout;
}
