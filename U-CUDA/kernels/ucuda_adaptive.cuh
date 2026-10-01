// ucuda_adaptive.cuh — адаптивный шаг: регуляторы и драйвер интегрирования.
//
// Один исходник для GPU (NVRTC, шаблоны kernels/*.template.cu) и CPU (MSVC),
// чтобы последовательность шагов не зависела от устройства.
//
// Две части:
//   * раскладка (UcudaAdaptParams и структуры регулятора) — не зависит от
//     AMOUNTOFX, её включает и хост (parametric_engine), определив
//     UCUDA_ADAPT_LAYOUT_ONLY: параметры уходят в ядро байт в байт;
//   * драйвер — функции-шаблоны по провайдеру функций схемы K (rhs, emb,
//     dprep, deval — тела печатает codegen_adaptive, см. AdaptiveCode в
//     codegen.hpp). На GPU провайдер — UcudaKrsFns: с UCUDA_AD_KRS_FUNCS он
//     вызывает ucuda_krs_rhs / _emb / _dprep / _deval, которые шаблон
//     определяет до #include. Размер массивов состояния — UCUDA_AD_NMAX
//     (по умолчанию AMOUNTOFX), сама размерность — во время выполнения.
//
// Шаг выбирается так (см. ucuda_ad_step): предложенный h зажимается в
// [h_min, h_max]; при h <= h_min шаг делается принудительно и принимается
// даже при отказе регулятора — такой шаг считается «вынужденным». После maxrej
// отказов подряд следующая попытка тоже делается с h_min и принимается без
// условий (вынужденной она считается, только если регулятор отказал и ей).
// maxrej = UCUDA_AD_NO_RETRY — без повторов: каждая попытка с конечной ошибкой
// принимается, отказ регулятора лишь уменьшает следующий шаг (такие шаги — тоже
// «вынужденные»); допуск тогда — цель, а не гарантия. Шаг,
// переходящий tEnd, обрезается до tEnd (как в scipy; clip = 1 — как у Хайрера,
// с запасом 1.01 h). Регулятор сам решает, принять ли шаг, и предлагает
// следующий h; драйвер лишь следит, чтобы отказ уменьшал шаг хотя бы до 0.9 h
// (иначе неудачный пользовательский регулятор зациклил бы поток).

// Без #pragma once: хост включает файл дважды — сначала только раскладку
// (adaptive_settings.h), потом, в CPU-интеграторе, целиком. Каждая часть
// защищена своим макросом.
#ifndef UCUDA_ADAPTIVE_LAYOUT_DONE
#define UCUDA_ADAPTIVE_LAYOUT_DONE

#if !defined(__CUDACC_RTC__)
#include <cstring>   // memcpy: математика регулятора на CPU (ucuda_ctl_log2 и др.)
#endif
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define UCUDA_HD __host__ __device__
#else
#define UCUDA_HD
#endif

#define UCUDA_AD_MAXN      32   // потолок размерности (kMaxAmountOfX движка)
#define UCUDA_AD_MAXSTAGES 16   // kAdaptMaxStages в codegen.hpp
#define UCUDA_AD_MAXDENSE  8    // kAdaptMaxDense
#define UCUDA_AD_MAXLOW    2    // kAdaptMaxLow
#define UCUDA_CTL_NPAR     12   // параметров регулятора
#define UCUDA_CTL_HIST     3    // принятых шагов в истории регулятора
#define UCUDA_CTL_USER     8    // свободная память регулятора
#define UCUDA_AD_NO_RETRY  (-1) // UcudaAdaptParams::maxrej: шаг не повторяется (поле UI "max rejects" = 0)

// Встроенные регуляторы (UcudaAdaptParams::ctrl), параметры — c[]:
//   HAIRER — dopri5 / dop853 Хайрера: c = {safe, fac1, fac2, beta, kbeta}.
//            Показатель fac11 = err^(1/(q+1) - kbeta*beta), PI-поправка
//            facold^beta. dopri5: {0.9, 0.2, 10, 0.04, 0.75}, dop853:
//            {0.9, 0.333, 6, 0, 0.2}. Шаг принимается при err <= 1.
//   SCIPY  — RungeKutta._step_impl scipy: c = {safety, min_factor, max_factor},
//            {0.9, 0.2, 10}; принимается при err < 1, после отказа шаг не растёт.
//   I      — элементарный: c = {safety, facmin, facmax}.
//   PI     — Густафссон: c = {safety, facmin, facmax, kI, kP},
//            fac = (th/e_n)^(kI/k) (e_{n-1}/e_n)^(kP/k), k = q+1 (обычно kI = 0.3, kP = 0.4).
//   FILTER — цифровой фильтр Сёдерлинда: c = {safety, facmin, facmax, b1, b2,
//            b3, a2, a3, limiter}, rho = (th/e_n)^(b1/k) (th/e_{n-1})^(b2/k)
//            (th/e_{n-2})^(b3/k) rho_{n-1}^(-a2) rho_{n-2}^(-a3);
//            limiter = 1 — сглаживание 1 + atan(rho - 1), иначе только зажим.
//            У PI и FILTER safety задаёт ЦЕЛЕВУЮ ошибку th = safety^k (как у Сёдерлинда
//            eps = theta*tol), а не множит rho: на гладком участке оба сходятся к err = th,
//            как и элементарный закон safety * err^(-1/k). Множитель safety * rho давал
//            цель safety^(k / sum kbeta): PI (kI = 0.3) на DOP853 — err ~ 0.06, H321 —
//            ~ 5e-4 и вдвое больше шагов.
//   CUSTOM — пользовательский: C body из библиотеки (AdaptiveUserCtrl), печатается
//            в ucuda_ctrl_custom (adaptive_ctrl_source).
#define UCUDA_CTRL_HAIRER 0
#define UCUDA_CTRL_SCIPY  1
#define UCUDA_CTRL_I      2
#define UCUDA_CTRL_PI     3
#define UCUDA_CTRL_FILTER 4
#define UCUDA_CTRL_CUSTOM 5

// Пока работаем только с регулятором Хайрера: остальные законы (SciPy, I, PI, Filter,
// пользовательский C body) в шаг не компилируются, а UI и сессии их не предлагают
// (adaptive_settings.h, gui.cpp, session_io.cpp). Вернуть все — #define UCUDA_AD_ALL_CTRL.
#ifndef UCUDA_AD_ALL_CTRL
#define UCUDA_AD_HAIRER_ONLY 1
#endif

// Параметры адаптивного шага — один экземпляр на запуск. Только numb и int
// фиксированного размера: хост заполняет ту же структуру и передаёт её в ядро.
struct UcudaAdaptParams {
    numb rtol;
    numb atol[UCUDA_AD_MAXN];   // по переменным (скаляр хост размножает)
    numb h0;                    // <= 0 — автоматически (select_initial_step)
    numb hmin;                  // <= 0 — max(10 ulp(t), 1e-12 * span), см. ucuda_ad_hmin_auto
    numb hmax;                  // <= 0 — без ограничения
    numb span;                  // длина всего интервала (нужна выбору h0)
    numb c[UCUDA_CTL_NPAR];     // параметры регулятора
    int  q;                     // порядок оценщика: показатель 1/(q+1)
    int  nlo;                   // число оценочных решений
    int  ctrl;                  // UCUDA_CTRL_*
    int  clip;                  // 0 — обрезка по tEnd как в scipy, 1 — как у Хайрера
    int  emb_rhs;               // вычислений f на попытку шага (статистика)
    int  dprep_rhs;             // вычислений f на плотный выход (статистика)
    int  h0mode;                // выбор h0: 0 — scipy, 1 — Хайрер (hinit dopri5/dop853)
    int  maxrej;                // > 0 — после стольких отказов подряд попытка делается с h_min
                                //   и принимается без условий; 0 — без предела;
                                //   UCUDA_AD_NO_RETRY — без повторов (каждая попытка принимается)
};

// Вход регулятора. Норму ошибки он считает сам (по yerr и масштабу от y0, y1), см.
// ucuda_ctl_err. Оценочное решение, если нужно, — y1 - yerr.
struct UcudaCtlIn {
    const numb* y0;     // y_n — начало шага
    const numb* y1;     // старшее решение y_{n+1}
    const numb* yerr;   // оценки ошибки y1 - yhat_m, nlo подряд по n (DOP853: по 5-му, 3-му
                        //   порядку); схема считает их по разности весов, без вычитания решений
    const numb* atol;   // n значений
    const numb* c;      // параметры регулятора
    numb rtol;
    numb h;             // размер этой попытки
    numb hmin, hmax;    // hmax <= 0 — без ограничения
    numb t;             // t_n
    int  n, nlo, q;
    int  nrej;          // сколько попыток этого шага уже отвергнуто
    const struct UcudaCtlConst* cc;   // производные константы законов (ucuda_ctl_prepare)
    numb err_extra;     // добавочная норма ошибки (LLE/LS — ошибка возмущений), 0 — нет;
                        //   ucuda_ctl_err берёт большую из неё и своей
};

// Память регулятора между шагами. h[0], err[0] — последний принятый шаг,
// h[1], err[1] — предыдущий и т.д.; заполнено min(nacc, UCUDA_CTL_HIST).
struct UcudaCtlMem {
    numb h[UCUDA_CTL_HIST];
    numb err[UCUDA_CTL_HIST];
    numb user[UCUDA_CTL_USER];
    int  nacc;
    int  pad;
};

struct UcudaCtlOut {
    numb h;         // предлагаемый следующий шаг (после отказа — для повтора)
    numb err;       // оценка ошибки этой попытки (норма, 1 = на границе допуска)
    int  accept;
    int  pad;
};

// Производные константы встроенных законов: зависят только от P (c[], q), поэтому
// считаются один раз на траекторию (ucuda_ad_init / ucuda_ad_restart), а не на каждой
// попытке — там это деления в double. Не на хосте: ось свипа по параметру регулятора
// правит P.c уже на устройстве.
struct UcudaCtlConst {
    numb facc1, facc2;  // Hairer: 1/fac1, 1/fac2
    numb expo1;         // Hairer: 1/(q+1) - kbeta*beta
    numb expo;          // SciPy, I и отказы PI/Filter: -1/(q+1)
    numb isafe;         // Hairer: 1/safe
    numb b1, b2, b3;    // PI: kI/k, kP/k; Filter: b1/k, b2/k, b3/k (c[3..5]/k, k = q+1)
    numb lth_pi;        // PI: log2 (th^(kI/k)) = kI log2 safety, th = safety^k — целевая ошибка
    numb lth_f;         // Filter: (b1 + b2 + b3) log2 safety
};

UCUDA_HD inline void ucuda_ctl_prepare(const UcudaAdaptParams& P, UcudaCtlConst& cc) {
    cc.facc1 = 1 / P.c[1];
    cc.facc2 = 1 / P.c[2];
    cc.expo1 = (numb)1 / (P.q + 1) - P.c[4] * P.c[3];
    cc.expo  = (numb)-1 / (P.q + 1);
    cc.isafe = 1 / P.c[0];
    const numb k = (numb)(P.q + 1);
    cc.b1 = P.c[3] / k; cc.b2 = P.c[4] / k; cc.b3 = P.c[5] / k;
    const numb ls = log2(P.c[0]);
    cc.lth_pi = P.c[3] * ls;
    cc.lth_f  = (P.c[3] + P.c[4] + P.c[5]) * ls;
}

// Статистика одной траектории.
struct UcudaAdaptStats {
    numb hmin, hmax, hsum;            // по принятым шагам
    unsigned long long nacc, nrej, nforced, nrhs;
};

// ---- Регуляторы: нормы и встроенные законы ------------------------------------
//
// Норма ошибки и степени в законах считаются в float: множителю шага хватает шести
// знаков (решение о приёме сдвигается лишь у самой границы, на 1e-7 от err = 1), а
// деление, корень и pow в double на GeForce (FP64 = 1/32 от FP32) стоят как несколько
// вычислений f. На DOP853 и Лоренце это −30..−50% времени шага в зависимости от
// регулятора. UCUDA_AD_EXACT_CTL возвращает double — для побитовой сверки
// последовательности шагов с dop853.c / scipy.

// ---- Математика регулятора на CPU -------------------------------------------------
// Попытка шага зовёт frexp, log2f, exp2f, nextafter (h_min) и с десяток fmin / fmax. На GPU
// это инструкции, а на CPU — вызовы CRT: в exe (/MD) exp2f ~36 нс, log2f 23, frexp 16,
// nextafter 10, fmin / fmax по ~4. На хосте их заменяют встроенные версии ниже (<= 0.5 ulp
// float, особые значения — через CRT; h_min — тот же результат); код GPU не меняется.
// Попытка RK45 на Рёсслере (сама схема ~50 нс): exe 230 -> 130 нс, DLL cl.exe (статическая CRT,
// Order -> Performance) 137 -> 120 нс. Остаток — задержка цепочки норма -> log2 -> exp2 -> h,
// которую ждёт следующая попытка: на дешёвой правой части попытка всё равно в ~2.5 раза
// дороже шага той же схемы с постоянным h.
UCUDA_HD inline numb ucuda_fmax(numb a, numb b) {   // как fmax: NaN отбрасывается
#ifdef __CUDA_ARCH__
    return fmax(a, b);
#else
    return (a > b || b != b) ? a : b;
#endif
}
UCUDA_HD inline numb ucuda_fmin(numb a, numb b) {
#ifdef __CUDA_ARCH__
    return fmin(a, b);
#else
    return (a < b || b != b) ? a : b;
#endif
}
UCUDA_HD inline float ucuda_fmaxf(float a, float b) {
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return (a > b || b != b) ? a : b;
#endif
}
UCUDA_HD inline float ucuda_fminf(float a, float b) {
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return (a < b || b != b) ? a : b;
#endif
}

// 2^y во float. CPU: |y| < 126 — 2^k 2^f, k — ближайшее целое, |f| <= 1/2, 2^f — ряд Тейлора
// e^(f ln 2) до 8-й степени (погрешность < 3e-10); остальное (переполнение, денормалы, inf, NaN) — CRT.
// Многочлен — по схеме Эстрина: регулятор стоит на пути от ошибки попытки к следующему шагу,
// и важна задержка цепочки, а не число операций (Горнер — 9 зависимых умножений и сложений).
UCUDA_HD inline float ucuda_ctl_exp2(float y) {
#ifdef __CUDA_ARCH__
    return exp2f(y);
#else
    if (!(y > -126.0f && y < 126.0f)) return exp2f(y);
    const int k = (int)(y >= 0 ? y + 0.5f : y - 0.5f);
    const double f = (double)y - k, f2 = f * f, f4 = f2 * f2;   // c_j = ln2^j / j!
    const double q0 = (1 + 0.69314718055994531 * f) + (0.24022650695910071 + 0.055504108664821579 * f) * f2;
    const double q1 = (9.6181291076284772e-3 + 1.3333558146428443e-3 * f)
                    + (1.5403530393381609e-4 + 1.5252733804059840e-5 * f) * f2;
    const double p = q0 + (q1 + 1.3215486790144307e-6 * f4) * f4;
    const unsigned long long b = (unsigned long long)(k + 1023) << 52;
    double sc; memcpy(&sc, &b, sizeof sc);
    return (float)(p * sc);
#endif
}

// sc_i = atol_i + rtol*max(|y0_i|, |y1_i|) — как у Хайрера и scipy.
UCUDA_HD inline numb ucuda_ctl_scale(const UcudaCtlIn& in, int i) {
    return in.atol[i] + in.rtol * ucuda_fmax(fabs(in.y0[i]), fabs(in.y1[i]));
}

// Сумма квадратов масштабированной m-й оценки ошибки.
UCUDA_HD inline numb ucuda_ctl_sum2(const UcudaCtlIn& in, int m) {
    numb s = 0;
    for (int i = 0; i < in.n; ++i) {
        numb e = in.yerr[m * in.n + i] / ucuda_ctl_scale(in, i);
        s += e * e;
    }
    return s;
}

// Норма ошибки: RMS по первому оценщику; при двух оценщиках — комбинация
// DOP853 (Хайрер, scipy): s5 / sqrt(n (s5 + 0.01 s3)).
UCUDA_HD inline numb ucuda_ctl_err_exact(const UcudaCtlIn& in) {
    if (in.nlo >= 2) {
        numb s5 = ucuda_ctl_sum2(in, 0), s3 = ucuda_ctl_sum2(in, 1);
        if (s5 == 0 && s3 == 0) return 0;
        return s5 / sqrt((s5 + (numb)0.01 * s3) * (numb)in.n);
    }
    return sqrt(ucuda_ctl_sum2(in, 0)) / sqrt((numb)in.n);
}

// То же, что ucuda_ctl_err_exact, в float: отношение оценки к масштабу и суммы; масштаб общий для обоих
// оценщиков и берётся один раз, обратной величиной. Суммы вне [0, 1e30) — огромная
// ошибка, масштаб у нуля (atol = 0; 1/sc = inf), inf или NaN — пересчитываются в double:
// признак разлёта на h_min смотрит, конечна ли ошибка, и переполнение float не должно
// выдать себя за разлёт (а переполнение знаменателя — за нулевую ошибку).
UCUDA_HD inline numb ucuda_ctl_err_own(const UcudaCtlIn& in) {
#ifndef UCUDA_AD_EXACT_CTL
    const int n = in.n;
    float s0 = 0, s1 = 0;
    if (in.nlo >= 2) {
        for (int i = 0; i < n; ++i) {
            const float r  = 1.0f / (float)ucuda_ctl_scale(in, i);
            const float e0 = (float)in.yerr[i] * r;
            const float e1 = (float)in.yerr[n + i] * r;
            s0 += e0 * e0;
            s1 += e1 * e1;
        }
    } else {
        for (int i = 0; i < n; ++i) {   // один оценщик: обратная величина ничего не экономит
            const float e0 = (float)in.yerr[i] / (float)ucuda_ctl_scale(in, i);
            s0 += e0 * e0;
        }
    }
    if (s0 < 1e30f && s1 < 1e30f) {   // !(<) ловит и NaN
        if (in.nlo < 2) return (numb)sqrtf(s0 / (float)in.n);
        if (s0 == 0 && s1 == 0) return 0;
        return (numb)(s0 / sqrtf((s0 + 0.01f * s1) * (float)in.n));
    }
#endif
    return ucuda_ctl_err_exact(in);
}

// Норма ошибки попытки для законов: своя и добавочная (err_extra), большая из двух;
// NaN любой из них проходит насквозь.
UCUDA_HD inline numb ucuda_ctl_err(const UcudaCtlIn& in) {
    const numb e = ucuda_ctl_err_own(in), x = in.err_extra;
    if (e != e || x != x) return e != e ? e : x;
    return x > e ? x : e;
}

// x^y для законов регулятора. Быстрая ветка — через log2: x = m 2^e (frexp),
// log2 x = e + log2f(m), поэтому вход не переполняет float во всём диапазоне double;
// результат вне float (0 или inf) законы и так зажимают в [facmin, facmax].
UCUDA_HD inline float ucuda_ctl_log2(numb x) {
#ifdef __CUDA_ARCH__
    int e;
    const numb m = frexp(x, &e);
    return (float)e + log2f((float)m);
#else
    // CPU: нормальное x > 0 — показатель и мантисса из битов, log2 мантиссы в [sqrt(1/2), sqrt(2))
    // рядом 2 atanh(s) / ln 2, s = (m - 1)/(m + 1), |s| <= 0.172 (погрешность < 3e-11), по Эстрину.
    // 0, < 0, денормалы, inf, NaN — через CRT.
    const double d = (double)x;
    unsigned long long b; memcpy(&b, &d, sizeof b);
    const int be = (int)((b >> 52) & 0x7ff);
    if ((b >> 63) != 0 || be == 0 || be == 0x7ff) return (float)log2(d);
    b = (b & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m; memcpy(&m, &b, sizeof m);
    int e = be - 1023;
    if (m > 1.4142135623730951) { m *= 0.5; ++e; }
    const double s = (m - 1) / (m + 1), s2 = s * s, s4 = s2 * s2, s8 = s4 * s4;
    const double p = s * ((2.8853900817779268 + 0.96179669392597560 * s2)
                        + (0.57707801635558541 + 0.41219858311113243 * s2) * s4
                        + (0.32059889797532522 + 0.26230818925253882 * s2) * s8);
    return (float)((double)e + p);
#endif
}

UCUDA_HD inline numb ucuda_ctl_pow(numb x, numb y) {
#ifdef UCUDA_AD_EXACT_CTL
    return pow(x, y);
#else
    return (numb)ucuda_ctl_exp2((float)y * ucuda_ctl_log2(x));
#endif
}

UCUDA_HD inline numb ucuda_clamp(numb x, numb lo, numb hi) { return ucuda_fmin(hi, ucuda_fmax(lo, x)); }

UCUDA_HD inline void ucuda_ctrl_hairer(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    const numb safe = in.c[0], facc1 = in.cc->facc1, facc2 = in.cc->facc2;
    const numb beta = in.c[3], expo1 = in.cc->expo1;
    const numb err = ucuda_ctl_err(in);
    const numb facold = m.nacc > 0 ? m.user[0] : (numb)1e-4;
#ifdef UCUDA_AD_EXACT_CTL
    const numb fac11 = ucuda_ctl_pow(err, expo1);
    numb fac = beta == 0 ? fac11 : fac11 / ucuda_ctl_pow(facold, beta);   // dop853: beta = 0
    fac = fmax(facc2, fmin(facc1, fac / safe));
    numb hnew = in.h / fac;
#else
    // Во float и в логарифмах: fac11 / facold^beta = 2^(expo1 log2 err - beta log2 facold),
    // деление на safe — умножение на 1/safe, h / fac — умножение на 1/fac. NaN в err даёт,
    // как и в double-ветке, fac = facc1 (fminf/fmaxf отбрасывают NaN).
    (void)safe;
    const float l11 = (float)expo1 * ucuda_ctl_log2(err);
    const float lfac = beta == 0 ? l11 : l11 - (float)beta * ucuda_ctl_log2(facold);
    const float isafe = (float)in.cc->isafe;
    const float fac = ucuda_fmaxf((float)facc2, ucuda_fminf((float)facc1, ucuda_ctl_exp2(lfac) * isafe));
    numb hnew = in.h * (numb)(1.0f / fac);
#endif
    o.err = err;
    if (err <= 1) {
        o.accept = 1;
        m.user[0] = ucuda_fmax(err, (numb)1e-4);
        if (in.hmax > 0 && hnew > in.hmax) hnew = in.hmax;
        if (in.nrej > 0) hnew = ucuda_fmin(hnew, in.h);
    } else {
        o.accept = 0;
#ifdef UCUDA_AD_EXACT_CTL
        hnew = in.h / fmin(facc1, fac11 / safe);
#else
        hnew = in.h * (numb)(1.0f / ucuda_fminf((float)facc1, ucuda_ctl_exp2(l11) * isafe));
#endif
    }
    o.h = hnew;
}

UCUDA_HD inline void ucuda_ctrl_scipy(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    (void)m;
    const numb safety = in.c[0], minf = in.c[1], maxf = in.c[2];
    const numb expo = in.cc->expo;
    const numb err = ucuda_ctl_err(in);
    o.err = err;
    if (err < 1) {
        numb factor = (err == 0) ? maxf : ucuda_fmin(maxf, safety * ucuda_ctl_pow(err, expo));
        if (in.nrej > 0) factor = ucuda_fmin((numb)1, factor);
        o.h = in.h * factor;
        o.accept = 1;
    } else {
        o.h = in.h * ucuda_fmax(minf, safety * ucuda_ctl_pow(err, expo));
        o.accept = 0;
    }
}

// Элементарный закон: общий для I, отказов у PI/FILTER и первых шагов без истории.
UCUDA_HD inline void ucuda_ctrl_elem(const UcudaCtlIn& in, numb err, UcudaCtlOut& o) {
    const numb safety = in.c[0], facmin = in.c[1], facmax = in.c[2];
    const numb expo = in.cc->expo;
    o.err = err;
    o.accept = err <= 1;
    const numb hi = (o.accept && in.nrej == 0) ? facmax : (numb)1;
    const numb fac = (err == 0) ? hi : ucuda_clamp(safety * ucuda_ctl_pow(err, expo), facmin, hi);
    o.h = in.h * fac;
}

UCUDA_HD inline void ucuda_ctrl_i(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    (void)m;
    ucuda_ctrl_elem(in, ucuda_ctl_err(in), o);
}

UCUDA_HD inline void ucuda_ctrl_pi(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    const numb err = ucuda_ctl_err(in);
    // !(err <= 1): NaN тоже к элементарному закону — он его отвергает; формула PI с NaN
    // зажималась в facmin и принимала шаг.
    if (!(err <= 1) || m.nacc < 1 || err == 0 || !(m.err[0] > 0)) { ucuda_ctrl_elem(in, err, o); return; }
    const numb facmin = in.c[1], facmax = in.c[2];
    const numb kI = in.cc->b1, kP = in.cc->b2;
#ifdef UCUDA_AD_EXACT_CTL
    numb fac = exp2(in.cc->lth_pi) * pow(err, -kI) * pow(m.err[0] / err, kP);
#else
    // (th/err)^kI (err_prev/err)^kP = 2^(lth - kI L - kP L + kP L_prev): одна exp2, без деления.
    const float le = ucuda_ctl_log2(err), l0 = ucuda_ctl_log2(m.err[0]);
    numb fac = (numb)ucuda_ctl_exp2((float)in.cc->lth_pi - (float)kI * le + (float)kP * (l0 - le));
#endif
    fac = ucuda_clamp(fac, facmin, in.nrej > 0 ? (numb)1 : facmax);
    o.err = err; o.accept = 1; o.h = in.h * fac;
}

UCUDA_HD inline void ucuda_ctrl_filter(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    const numb err = ucuda_ctl_err(in);
    if (!(err <= 1) || m.nacc < 2 || err == 0 || !(m.err[0] > 0) || !(m.err[1] > 0)) {   // NaN — см. PI
        ucuda_ctrl_elem(in, err, o); return;
    }
    const numb facmin = in.c[1], facmax = in.c[2];
    const numb b1 = in.cc->b1, b2 = in.cc->b2, b3 = in.cc->b3, a2 = in.c[6], a3 = in.c[7];
    // При nacc >= 2 в памяти уже два принятых шага: err[1] = e_{n-2}, h[1] = h_{n-2}.
    const numb e2 = m.err[1];
#ifdef UCUDA_AD_EXACT_CTL
    const numb rho1 = in.h / m.h[0];                         // h_n / h_{n-1}
    const numb rho2 = m.h[0] / m.h[1];                       // h_{n-1} / h_{n-2}
    numb rho = exp2(in.cc->lth_f) * pow(err, -b1) * pow(m.err[0], -b2) * pow(e2, -b3)
             * pow(rho1, -a2) * pow(rho2, -a3);
#else
    // Произведение степеней — сумма логарифмов и одна exp2; отношения шагов — разности
    // log2 h, без деления.
    const float lh0 = ucuda_ctl_log2(m.h[0]);
    const float lr1 = ucuda_ctl_log2(in.h) - lh0;                            // log2 (h_n / h_{n-1})
    const float lr2 = lh0 - ucuda_ctl_log2(m.h[1]);                          // log2 (h_{n-1} / h_{n-2})
    numb rho = (numb)ucuda_ctl_exp2((float)in.cc->lth_f
                           - (float)b1 * ucuda_ctl_log2(err) - (float)b2 * ucuda_ctl_log2(m.err[0])
                           - (float)b3 * ucuda_ctl_log2(e2) - (float)a2 * lr1 - (float)a3 * lr2);
#endif
#ifdef UCUDA_AD_EXACT_CTL
    if (in.c[8] > 0) rho = 1 + atan(rho - 1);
#else
    if (in.c[8] > 0) rho = 1 + (numb)atanf((float)(rho - 1));   // сглаживанию хватает float
#endif
    rho = ucuda_clamp(rho, facmin, in.nrej > 0 ? (numb)1 : facmax);
    o.err = err; o.accept = 1; o.h = in.h * rho;
}

#endif // UCUDA_ADAPTIVE_LAYOUT_DONE

#if !defined(UCUDA_ADAPT_LAYOUT_ONLY) && !defined(UCUDA_ADAPTIVE_DRIVER_DONE)
#define UCUDA_ADAPTIVE_DRIVER_DONE

// Размер массивов состояния: на GPU — AMOUNTOFX шаблона, на CPU — потолок
// UCUDA_AD_MAXN (размерность там известна только во время выполнения).
#ifndef UCUDA_AD_NMAX
#define UCUDA_AD_NMAX AMOUNTOFX
#endif

// Размерность в циклах драйвера. С UCUDA_AD_STATIC_N на GPU — AMOUNTOFX, константа
// компиляции: циклы разворачиваются, и состояние лежит в регистрах, а не в локальной
// памяти (ptxas, DOP853 на Лоренце: стековый кадр 1056 Б -> 80 Б, регистров 166 -> 255).
// Выгодно, когда потоков мало (ядро Analysis: −8..15%); в свипах на весь GPU падение
// занятости перевешивает (bif2d 256x256, DOP853 с плотным выходом: 7.4 -> 13.5 с),
// поэтому по умолчанию — S.n, как на CPU, где размерность известна только при запуске.
#if defined(__CUDA_ARCH__) && defined(UCUDA_AD_STATIC_N)
#define UCUDA_AD_N(S) AMOUNTOFX
#else
#define UCUDA_AD_N(S) ((S).n)
#endif

// ---- Пользовательский регулятор -------------------------------------------------
// Модуль включает этот заголовок дважды: сперва с UCUDA_ADAPT_LAYOUT_ONLY (структуры,
// нормы, встроенные законы), затем — определив UCUDA_HAS_CUSTOM_CTRL и функцию
//   void ucuda_ctrl_custom(const UcudaCtlIn&, UcudaCtlMem&, UcudaCtlOut&)
// (adaptive_ctrl_source в adaptive_settings.h) — целиком. Так тело пользователя видит
// все помощники, а встраивается в шаг, как встроенные законы.

UCUDA_HD inline void ucuda_step_ctrl(int ctrl, const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    o.accept = 0; o.err = 0; o.h = in.h; o.pad = 0;
#ifdef UCUDA_AD_HAIRER_ONLY
    (void)ctrl;
    ucuda_ctrl_hairer(in, m, o);
#else
    switch (ctrl) {
    case UCUDA_CTRL_SCIPY:  ucuda_ctrl_scipy(in, m, o);  break;
    case UCUDA_CTRL_I:      ucuda_ctrl_i(in, m, o);      break;
    case UCUDA_CTRL_PI:     ucuda_ctrl_pi(in, m, o);     break;
    case UCUDA_CTRL_FILTER: ucuda_ctrl_filter(in, m, o); break;
#ifdef UCUDA_HAS_CUSTOM_CTRL
    case UCUDA_CTRL_CUSTOM: ucuda_ctrl_custom(in, m, o); break;
#endif
    default:                ucuda_ctrl_hairer(in, m, o); break;
    }
#endif
}

// ---- Драйвер ---------------------------------------------------------------------
//
// Функции схемы подаются объектом K с методами (сигнатуры — AdaptiveCode):
//   rhs(X, a, F), emb(X, F0, a, h, Y, E, F1, W), dprep(X, Y, F0, F1, a, h, W, D),
//   deval(D, th, Yo).
// На GPU это UcudaKrsFns (ниже) — вызовы ucuda_krs_*, которые определяет шаблон;
// на CPU — вычислитель правых частей с таблицами Бутчера (integrator.cpp).

#ifdef UCUDA_AD_KRS_FUNCS
struct UcudaKrsFns {
    UCUDA_HD void rhs(const numb* X, const numb* a, numb* F) const { ucuda_krs_rhs(X, a, F); }
    UCUDA_HD void emb(const numb* X, const numb* F0, const numb* a, const numb h,
                      numb* Y, numb* E, numb* F1, numb* W) const {
        ucuda_krs_emb(X, F0, a, h, Y, E, F1, W);
    }
    UCUDA_HD void dprep(const numb* X, const numb* Y, const numb* F0, const numb* F1,
                        const numb* a, const numb h, numb* W, numb* D) const {
        ucuda_krs_dprep(X, Y, F0, F1, a, h, W, D);
    }
    UCUDA_HD void deval(const numb* D, const numb th, numb* Yo) const { ucuda_krs_deval(D, th, Yo); }
};
#endif

struct UcudaAdaptState {
    numb t;                                        // t_n
    numb h;                                        // предложенный следующий шаг
    numb hfree;                                    // то же после последнего шага, не обрезанного
                                                   //   концом интервала (для ucuda_ad_restart)
    numb X[UCUDA_AD_NMAX];                         // y_n
    numb F0[UCUDA_AD_NMAX];                        // f(y_n)
    numb tp, hp;                                   // последний принятый шаг [tp, tp + hp]
#ifndef UCUDA_AD_NO_DENSE
    // Плотный выход. С UCUDA_AD_NO_DENSE (LLE/LS: перенормировка только в узлах шага)
    // этого нет вовсе, а стадии живут лишь внутри ucuda_ad_step: состояние потока
    // короче на (MAXDENSE + 2) * n чисел плюс сами стадии.
    numb Xp[UCUDA_AD_NMAX];                        // его начало y_{n-1}
    numb Fp[UCUDA_AD_NMAX];                        // f(y_{n-1})
    numb W[UCUDA_AD_MAXSTAGES * UCUDA_AD_NMAX];    // стадии последней попытки
    numb D[UCUDA_AD_MAXDENSE * UCUDA_AD_NMAX];     // плотный выход последнего шага
    int  dense_ready;                              // D соответствует [tp, t]
#endif
    int  n;                                        // размерность
    int  last_forced;                              // последний шаг сделан принудительно
    int  nrej_run;                                 // отказов подряд у текущего шага (ucuda_ad_try_x)
    int  diverged;                                 // решение ушло в NaN/inf даже на h_min
    UcudaCtlMem mem;
    UcudaCtlConst cc;                              // производные константы законов
    UcudaAdaptStats st;
    numb* log;                                     // лог попыток: {t, h, err, код} на запись
    int  log_cap;                                  //   код: 1 — принят, 0 — отвергнут,
    int  log_n;                                    //        2 — вынужденный
};

// Нижняя граница шага по умолчанию: большее из 10 ulp(t) (scipy: min_step) и 1e-12 * span.
// Одних 10 ulp мало: регулятор, который упёрся в недостижимый допуск (пол округления ошибки
// клонов LLE/LS, слишком строгий rtol), держит шаг на границе, и при t ~ 50 это 7e-14 —
// ~1e15 шагов до конца интервала, расчёт не кончается. С 1e-12 * span — не больше 1e12
// вынужденных шагов на весь интервал, а на нормальных траекториях граница не достигается.
// span <= 0 (не задан) — только 10 ulp.
UCUDA_HD inline numb ucuda_ad_hmin_auto(numb t, numb span) {
#ifdef __CUDA_ARCH__
    const numb u = 10 * fabs(nextafter(t, (numb)1e308) - t);
#else
    // CPU: nextafter в CRT — вызов на каждой попытке; у 0 < t < 1e308 следующее число вверх —
    // соседний битовый код (результат тот же). Остальное — через CRT.
    numb u;
    if (t > 0 && t < (numb)1e308) {
        const double d = (double)t;
        unsigned long long b; memcpy(&b, &d, sizeof b); ++b;
        double up; memcpy(&up, &b, sizeof up);
        u = 10 * (numb)(up - d);
    } else u = 10 * fabs(nextafter(t, (numb)1e308) - t);
#endif
    const numb s = span > 0 ? (numb)1e-12 * span : (numb)0;
    return u > s ? u : s;
}

// Начальный шаг: select_initial_step scipy (Hairer-Nørsett-Wanner I, II.4).
template <class K>
UCUDA_HD inline numb ucuda_ad_h0(const K& k, int n, const numb* X, const numb* F0, const numb* a,
                                 const UcudaAdaptParams& P, UcudaAdaptStats& st) {
    if (!(P.span > 0)) return 0;
    numb s0 = 0, s1 = 0;
    for (int i = 0; i < n; ++i) {
        const numb sc = P.atol[i] + fabs(X[i]) * P.rtol;
        s0 += (X[i] / sc) * (X[i] / sc);
        s1 += (F0[i] / sc) * (F0[i] / sc);
    }
    const numb d0 = sqrt(s0) / sqrt((numb)n), d1 = sqrt(s1) / sqrt((numb)n);
    numb h0 = (d0 < (numb)1e-5 || d1 < (numb)1e-5) ? (numb)1e-6 : (numb)0.01 * d0 / d1;
    h0 = fmin(h0, P.span);
    numb X1[UCUDA_AD_NMAX], F1[UCUDA_AD_NMAX];
    for (int i = 0; i < n; ++i) X1[i] = X[i] + h0 * F0[i];
    k.rhs(X1, a, F1);
    st.nrhs += 1;
    numb s2 = 0;
    for (int i = 0; i < n; ++i) {
        const numb sc = P.atol[i] + fabs(X[i]) * P.rtol;
        const numb d = (F1[i] - F0[i]) / sc;
        s2 += d * d;
    }
    const numb d2 = sqrt(s2) / sqrt((numb)n) / h0;
    const numb h1 = (d1 <= (numb)1e-15 && d2 <= (numb)1e-15)
                  ? fmax((numb)1e-6, h0 * (numb)1e-3)
                  : pow((numb)0.01 / fmax(d1, d2), (numb)1 / (P.q + 1));
    numb h = fmin(fmin((numb)100 * h0, h1), P.span);
    if (P.hmax > 0) h = fmin(h, P.hmax);
    return h;
}

// Начальный шаг: hinit Хайрера (dopri5 / dop853). От scipy отличается нормами:
// суммы квадратов без деления на n, порог 1e-10 по сумме, hmax по умолчанию —
// весь интервал.
template <class K>
UCUDA_HD inline numb ucuda_ad_h0_hairer(const K& k, int n, const numb* X, const numb* F0, const numb* a,
                                        const UcudaAdaptParams& P, UcudaAdaptStats& st) {
    const numb hmax = P.hmax > 0 ? P.hmax : P.span;
    if (!(hmax > 0)) return 0;
    numb dnf = 0, dny = 0;
    for (int i = 0; i < n; ++i) {
        const numb sk = P.atol[i] + P.rtol * fabs(X[i]);
        dnf += (F0[i] / sk) * (F0[i] / sk);
        dny += (X[i] / sk) * (X[i] / sk);
    }
    numb h = (dnf <= (numb)1e-10 || dny <= (numb)1e-10) ? (numb)1e-6 : sqrt(dny / dnf) * (numb)0.01;
    h = fmin(h, hmax);
    numb X1[UCUDA_AD_NMAX], F1[UCUDA_AD_NMAX];
    for (int i = 0; i < n; ++i) X1[i] = X[i] + h * F0[i];
    k.rhs(X1, a, F1);
    st.nrhs += 1;
    numb der2 = 0;
    for (int i = 0; i < n; ++i) {
        const numb sk = P.atol[i] + P.rtol * fabs(X[i]);
        const numb d = (F1[i] - F0[i]) / sk;
        der2 += d * d;
    }
    der2 = sqrt(der2) / h;
    const numb der12 = fmax(fabs(der2), sqrt(dnf));
    const numb h1 = der12 <= (numb)1e-15 ? fmax((numb)1e-6, fabs(h) * (numb)1e-3)
                                         : pow((numb)0.01 / der12, (numb)1 / (P.q + 1));
    return fmin((numb)100 * fabs(h), fmin(h1, hmax));
}

// Старт с (t0, X0). log — буфер на log_cap записей по 4 numb или nullptr.
template <class K>
UCUDA_HD inline void ucuda_ad_init(UcudaAdaptState& S, const K& k, int n, const numb* X0, numb t0,
                                   const numb* a, const UcudaAdaptParams& P,
                                   numb* log = nullptr, int log_cap = 0) {
    S.n = n;
    S.st.hmin = 0; S.st.hmax = 0; S.st.hsum = 0;
    S.st.nacc = 0; S.st.nrej = 0; S.st.nforced = 0; S.st.nrhs = 1;
    for (int i = 0; i < n; ++i) S.X[i] = X0[i];
    S.t = t0;
    k.rhs(S.X, a, S.F0);
#ifndef UCUDA_AD_NO_DENSE
    for (int i = 0; i < n; ++i) { S.Xp[i] = S.X[i]; S.Fp[i] = S.F0[i]; }
    S.dense_ready = 0;
#endif
    S.tp = t0; S.hp = 0;
    S.last_forced = 0; S.diverged = 0; S.nrej_run = 0;
    for (int i = 0; i < UCUDA_CTL_HIST; ++i) { S.mem.h[i] = 0; S.mem.err[i] = 0; }
    for (int i = 0; i < UCUDA_CTL_USER; ++i) S.mem.user[i] = 0;
    S.mem.nacc = 0; S.mem.pad = 0;
    S.log = log; S.log_cap = log_cap; S.log_n = 0;
    ucuda_ctl_prepare(P, S.cc);
    S.h = P.h0 > 0 ? P.h0
        : (P.h0mode == 1 ? ucuda_ad_h0_hairer(k, n, S.X, S.F0, a, P, S.st)
                         : ucuda_ad_h0(k, n, S.X, S.F0, a, P, S.st));
    S.hfree = S.h;
}

// Продолжение с того же состояния при новых параметрах a / P (continuation): X,
// предложенный шаг и память регулятора переносятся, время, статистика и лог — с
// начала, f(X) пересчитывается. Шаг берётся из hfree: последний шаг интервала
// обрезан по его концу, и предложение после него занижено. Нет разумного шага
// (первый запуск, разлёт) — начальный, как у ucuda_ad_init.
template <class K>
UCUDA_HD inline void ucuda_ad_restart(UcudaAdaptState& S, const K& k, numb t0,
                                      const numb* a, const UcudaAdaptParams& P) {
    const int n = UCUDA_AD_N(S);
    const numb hc = S.hfree;
    S.st.hmin = 0; S.st.hmax = 0; S.st.hsum = 0;
    S.st.nacc = 0; S.st.nrej = 0; S.st.nforced = 0; S.st.nrhs = 1;
    S.t = t0;
    k.rhs(S.X, a, S.F0);
#ifndef UCUDA_AD_NO_DENSE
    for (int i = 0; i < n; ++i) { S.Xp[i] = S.X[i]; S.Fp[i] = S.F0[i]; }
    S.dense_ready = 0;
#endif
    S.tp = t0; S.hp = 0;
    S.last_forced = 0; S.diverged = 0; S.nrej_run = 0;
    S.log_n = 0;
    ucuda_ctl_prepare(P, S.cc);
    if (hc > 0 && hc <= (numb)1e300) S.h = hc;   // !(<=) отсекает и inf, и NaN
    else S.h = P.h0 > 0 ? P.h0
             : (P.h0mode == 1 ? ucuda_ad_h0_hairer(k, n, S.X, S.F0, a, P, S.st)
                              : ucuda_ad_h0(k, n, S.X, S.F0, a, P, S.st));
    S.hfree = S.h;
}

// Добавка к шагу (ucuda_ad_step_x): attempt() зовётся на каждой попытке после старшего
// решения Y и оценок ошибки E базовой траектории и возвращает свою норму ошибки попытки
// (1 — на границе допуска; регулятор судит по большей из двух), commit() — попытка
// принята. Так LLE/LS ведут клоны тем же шагом и заодно контролируют их ошибку.
struct UcudaAdNoExtra {
    template <class K>
    UCUDA_HD numb attempt(const K&, const numb*, const numb*, const numb*, const numb*, numb) { return 0; }
    UCUDA_HD void commit() {}
};

// Одна попытка шага из (S.t, S.X), не переходя tEnd. Вызывать при S.t < tEnd.
// Возвращает 1 — шаг принят (S продвинут), 0 — отказ (S.h уменьшен, S.nrej_run + 1)
// или разлёт (S.diverged = 1). После P.maxrej отказов подряд (если > 0) попытка делается
// с h_min и принимается без условий — предел числа проверок на шаг. Если ошибка не
// число даже на h_min (решение ушло в NaN/inf), шаг не принимается, а S.diverged = 1:
// дальше интегрировать бессмысленно, а вынужденные шаги по 10 ulp(t) не закончились
// бы никогда. Вызывающий обязан проверять diverged в своих циклах.
//
// LLE/LS (ucuda_lyap_point) делают одну попытку за итерацию, а не целый шаг (ucuda_ad_step_x):
// отказы у нитей варпа случаются в разное время, и с повторами внутри шага варп на
// каждом шаге ждал бы самую невезучую нить; LS 2D так в 1.5 раза быстрее. Свипы БД, бассейнов
// и метрик зовут целый шаг: у них та же перестройка оказалась в 1.3-1.6 раза медленнее.
// Последовательность попыток каждой нити та же.
template <class K, class Ext>
UCUDA_HD inline int ucuda_ad_try_x(UcudaAdaptState& S, const K& k, const numb* a,
                                   const UcudaAdaptParams& P, numb tEnd, Ext& ext) {
    const int n = UCUDA_AD_N(S);
    numb Y[UCUDA_AD_NMAX], E[UCUDA_AD_MAXLOW * UCUDA_AD_NMAX], F1[UCUDA_AD_NMAX];
#ifdef UCUDA_AD_NO_DENSE
    numb W[UCUDA_AD_MAXSTAGES * UCUDA_AD_NMAX];    // стадии попытки: после шага не нужны
#else
    numb* W = S.W;
#endif
    const int  nrej = S.nrej_run;
    const numb hmin = P.hmin > 0 ? P.hmin : ucuda_ad_hmin_auto(S.t, P.span);   // t в повторах не меняется
    {
        numb h = S.h;
        if (P.hmax > 0 && h > P.hmax) h = P.hmax;
        int forced = 0;
        if (!(h > hmin) || (P.maxrej > 0 && nrej >= P.maxrej)) { h = hmin; forced = 1; }   // !(>) ловит и NaN
        numb tn = S.t + h;
        int last = 0;
        if (P.clip ? (S.t + (numb)1.01 * h > tEnd) : (tn > tEnd)) {
            tn = tEnd; h = tEnd - S.t; last = 1;
        }

        k.emb(S.X, S.F0, a, h, Y, E, F1, W);
#ifndef UCUDA_AD_NODES_KERNEL   // ядро узлов (свипы) выводит только nacc, nrej, nforced, hsum
        S.st.nrhs += P.emb_rhs;
#endif
        const numb err_extra = ext.attempt(k, a, S.X, Y, E, h);

        UcudaCtlIn in;
        in.y0 = S.X; in.y1 = Y; in.yerr = E; in.atol = P.atol; in.c = P.c;
        in.rtol = P.rtol; in.h = h; in.hmin = hmin; in.hmax = P.hmax; in.t = S.t;
        in.n = n; in.nlo = P.nlo; in.q = P.q; in.nrej = nrej; in.cc = &S.cc; in.err_extra = err_extra;
        UcudaCtlOut o;
        ucuda_step_ctrl(P.ctrl, in, S.mem, o);

        const int finite = o.err <= (numb)1e300;   // !(<=) ловит и NaN
        if (forced && !finite) { S.diverged = 1; return 0; }
        // Без повторов принимается любая попытка с конечной ошибкой; NaN/inf — как обычный
        // отказ: шаг уменьшается, а на h_min разлёт ловится выше. Так же — приём регулятором
        // попытки с NaN/inf (пользовательский закон, который про NaN не подумал).
        const int acc  = o.accept && finite;
        const int take = acc || forced || (P.maxrej < 0 && finite);
#ifndef UCUDA_AD_NODES_KERNEL   // лог попыток — только у Analysis
        if (S.log != nullptr && S.log_n < S.log_cap) {
            numb* r = S.log + 4 * S.log_n++;
            r[0] = S.t; r[1] = h; r[2] = o.err;
            r[3] = acc ? (numb)1 : (take ? (numb)2 : (numb)0);
        }
#endif
        if (take) {
            ext.commit();
            if (!acc) S.st.nforced++;
#ifndef UCUDA_AD_NODES_KERNEL   // tp, hp — плотному выходу, которого у ядра узлов нет
            S.tp = S.t; S.hp = h;
#endif
            for (int i = 0; i < n; ++i) {
#ifndef UCUDA_AD_NO_DENSE
                S.Xp[i] = S.X[i]; S.Fp[i] = S.F0[i];
#endif
                S.X[i] = Y[i];    S.F0[i] = F1[i];
            }
            S.t = last ? tEnd : tn;
#if !(defined(UCUDA_AD_NODES_KERNEL) && defined(UCUDA_AD_HAIRER_ONLY))
            // История h и err — PI и фильтрам; Хайрер берёт только m.user[0] и счётчик.
            for (int j = UCUDA_CTL_HIST - 1; j > 0; --j) { S.mem.h[j] = S.mem.h[j - 1]; S.mem.err[j] = S.mem.err[j - 1]; }
            S.mem.h[0] = h; S.mem.err[0] = o.err;
#endif
            S.mem.nacc++;
#ifndef UCUDA_AD_NODES_KERNEL
            if (S.st.nacc == 0 || h < S.st.hmin) S.st.hmin = h;
            if (S.st.nacc == 0 || h > S.st.hmax) S.st.hmax = h;
#endif
            S.st.hsum += h; S.st.nacc++;
            S.h = (o.h > 0) ? o.h : h;   // регулятор вернул мусор — оставить шаг
            if (!last) S.hfree = S.h;
#ifndef UCUDA_AD_NO_DENSE
            S.dense_ready = 0;
#endif
#ifndef UCUDA_AD_NODES_KERNEL
            S.last_forced = forced;
#endif
            S.nrej_run = 0;
            return 1;
        }
        S.st.nrej++; S.nrej_run = nrej + 1;
        numb hn = o.h;
        if (!(hn <= (numb)0.9 * h)) hn = (numb)0.9 * h;
        S.h = hn;
    }
    return 0;
}

template <class K>
UCUDA_HD inline int ucuda_ad_try(UcudaAdaptState& S, const K& k, const numb* a,
                                 const UcudaAdaptParams& P, numb tEnd) {
    UcudaAdNoExtra none;
    return ucuda_ad_try_x(S, k, a, P, tEnd, none);
}

// Один принятый шаг (попытки до принятия или разлёта) — для CPU и одиночных траекторий.
template <class K, class Ext>
UCUDA_HD inline void ucuda_ad_step_x(UcudaAdaptState& S, const K& k, const numb* a,
                                     const UcudaAdaptParams& P, numb tEnd, Ext& ext) {
    while (!ucuda_ad_try_x(S, k, a, P, tEnd, ext) && !S.diverged) {}
}

template <class K>
UCUDA_HD inline void ucuda_ad_step(UcudaAdaptState& S, const K& k, const numb* a,
                                   const UcudaAdaptParams& P, numb tEnd) {
    UcudaAdNoExtra none;
    ucuda_ad_step_x(S, k, a, P, tEnd, none);
}

#ifndef UCUDA_AD_NO_DENSE
// y(tt) для tt в [S.tp, S.t] — плотным выходом последнего принятого шага.
template <class K>
UCUDA_HD inline void ucuda_ad_eval(UcudaAdaptState& S, const K& k, const numb* a,
                                   const UcudaAdaptParams& P, numb tt, numb* Yo) {
    if (tt >= S.t || !(S.hp > 0)) {
        for (int i = 0; i < UCUDA_AD_N(S); ++i) Yo[i] = S.X[i];
        return;
    }
    if (!S.dense_ready) {
        k.dprep(S.Xp, S.X, S.Fp, S.F0, a, S.hp, S.W, S.D);
        S.st.nrhs += P.dprep_rhs;
        S.dense_ready = 1;
    }
    k.deval(S.D, (tt - S.tp) / S.hp, Yo);
}

// Равномерная сетка вывода: шагать, пока узел не пройдёт tt (шаг обрезается
// только по tEnd, не по tt), и вернуть y(tt) плотным выходом.
template <class K>
UCUDA_HD inline void ucuda_ad_advance_to(UcudaAdaptState& S, const K& k, const numb* a,
                                         const UcudaAdaptParams& P, numb tt, numb tEnd, numb* Yo) {
    while (S.t < tt && !S.diverged) ucuda_ad_step(S, k, a, P, tEnd);
    ucuda_ad_eval(S, k, a, P, tt, Yo);
}
#endif // UCUDA_AD_NO_DENSE

#endif // UCUDA_ADAPTIVE_DRIVER_DONE

// ---- LLE / LS с адаптивным шагом: общий для GPU и CPU код --------------------------
// Включается макросом UCUDA_AD_LYAPUNOV (модуль lyapunov_adaptive_part.cu на GPU, DLL
// CPU-ветки из krs_cpu.cpp). Нужен AMOUNTOFX константой компиляции: клоны — массивы.
//
// Алгоритм — тот же, что у LLEKernelCUDA / LSKernelCUDA (Wolf / Benettin: клоны в
// eps-окрестности базовой траектории, для LS — с ортогонализацией Грама-Шмидта), и
// раскладка результата та же. Клоны делают ТОТ ЖЕ шаг, что базовая траектория x, тем же
// вложенным методом (с FSAL), и регулятор судит не только по ошибке x, но и по ошибке
// возмущений delta = y - x: её оценка — разность оценок E клона и x, масштаб — rtol * |delta|
// (RMS). Без этого у устойчивого равновесия ошибка x почти нулевая, шаг дорастает до
// границы устойчивости явного метода, и клоны видят численное отображение вместо потока:
// LLE Лоренца при r < 24.7 выходил ~0 вместо -0.5..-0.1.
//
// Перенормировка (renorm):
//   0 — ровно в T0 + k*NT: шаг, переходящий границу, обрезается по ней; после границы
//       шаг берётся из hfree — предложение после обрезанного шага занижено;
//   1 — в первом узле после T0 + k*NT: шаги не обрезаются, блоки чуть длиннее NT
//       (последний — ровно до T0 + nBlocks*NT).
// Итог — сумма логарифмов учётных блоков, делённая на их время.
#if defined(UCUDA_AD_LYAPUNOV) && !defined(UCUDA_ADAPT_LAYOUT_ONLY) && !defined(UCUDA_ADAPTIVE_LYAP_DONE)
#define UCUDA_ADAPTIVE_LYAP_DONE

// Генератор начальных направлений — побитовая копия заглушки curand из шаблонов LLE/LS
// (lle1d/ls1d/lle2d/ls2d.template.cu: splitmix по seed и номеру точки, затем LCG), под
// которыми NVRTC собирает LLEKernelCUDA / LSKernelCUDA, и CPU-портов (cpu_curand_*). Свой,
// потому что модуль собран на bifurcation2d.template.cu, а его заглушка номер
// подпоследовательности игнорирует: все точки свипа получали ОДНУ начальную рамку.
struct UcudaLyapRng { unsigned long long s; };
UCUDA_HD inline void ucuda_lyap_rng_init(unsigned long long seed, unsigned long long sequence, UcudaLyapRng& r) {
    unsigned long long z = seed + sequence * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    r.s = z ^ (z >> 31);
}
UCUDA_HD inline float ucuda_lyap_rng_uniform(UcudaLyapRng& r) {
    r.s = r.s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(((r.s >> 40) & 0xFFFFFFULL) + 1ULL) / 16777216.0f;
}

// Грам-Шмидт — копия projectionOperator / gramSchmidtProcess из cudaLibrary.cu с тем же
// порядком операций (DLL CPU-ветки cudaLibrary.cu не видит).
UCUDA_HD inline void ucuda_lyap_proj(const numb* a, const numb* b, numb* minuend, int n) {
    numb numerator = 0, denominator = 0;
    for (int i = 0; i < n; ++i) { numerator += a[i] * b[i]; denominator += b[i] * b[i]; }
    const numb fraction = denominator == 0 ? 0 : numerator / denominator;
    for (int i = 0; i < n; ++i) minuend[i] -= fraction * b[i];
}
UCUDA_HD inline void ucuda_lyap_gs(const numb* a, numb* b, int n, numb* den) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) b[j + i * n] = a[j + i * n];
        for (int j = 0; j < i; ++j) ucuda_lyap_proj(a + i * n, b + j * n, b + i * n, n);
    }
    for (int i = 0; i < n; ++i) {
        numb d = 0;
        for (int j = 0; j < n; ++j) d += b[i * n + j] * b[i * n + j];
        d = sqrt(d);
        for (int j = 0; j < n; ++j) b[i * n + j] = d == 0 ? 0 : b[i * n + j] / d;
        if (den != nullptr) den[i] = d;
    }
}

// Разлёт: nan/inf или sum|x| > maxValue (0 — без порога), как у циклов постоянного шага.
UCUDA_HD inline bool ucuda_lyap_out(const numb* x, const numb maxValue) {
    numb checker = 0;
    for (int j = 0; j < AMOUNTOFX; ++j) checker += fabs(x[j]);
    if (!(checker == checker) || !(checker - checker == 0)) return true;   // NaN или inf
    return maxValue != 0 && checker > maxValue;
}

// Начальные возмущения: LLE — случайное направление, LS — NC случайных векторов,
// ортонормированных Грамом-Шмидтом; y = x + eps * направление.
template <int NC>
UCUDA_HD inline void ucuda_lyap_init_clones(const numb* x, const numb eps, const int seq, numb* y, numb* z) {
    constexpr int n = AMOUNTOFX;
    UcudaLyapRng state;
    ucuda_lyap_rng_init(1234567891ULL, (unsigned long long)seq, state);   // seed — как у LLE/LSKernelCUDA
    for (int j = 0; j < NC; ++j) {
        numb zPower = 0;
        for (int i = 0; i < n; ++i) {
            z[j * n + i] = ucuda_lyap_rng_uniform(state) - 0.5;
            zPower += z[j * n + i] * z[j * n + i];
        }
        zPower = sqrt(zPower);
        for (int i = 0; i < n; ++i) z[j * n + i] /= zPower;
    }
    if constexpr (NC == 1) {
        for (int i = 0; i < n; ++i) y[i] = z[i] * eps + x[i];
    } else {
        ucuda_lyap_gs(z, y, n, nullptr);
        for (int j = 0; j < NC; ++j)
            for (int i = 0; i < n; ++i) y[j * n + i] = y[j * n + i] * eps + x[i];
    }
}

// Перенормировка: накопить логарифмы растяжения и вернуть клоны на расстояние eps.
// Формулы — ровно как в LLEKernelCUDA / LSKernelCUDA.
template <int NC>
UCUDA_HD inline void ucuda_lyap_renorm(const numb* x, const numb eps, numb* y, numb* z, numb* acc) {
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
        ucuda_lyap_gs(y, z, n, den);
        for (int k = 0; k < NC; ++k) {
            acc[k] += log(den[k] / eps);
            for (int j = 0; j < n; ++j) y[k * n + j] = (numb)(x[j] + z[k * n + j] * eps);
        }
    }
}

// Пол масштаба ошибки клонов: K машинных эпсилон от величины переменной (см. attempt ниже).
#ifndef UCUDA_LYAP_NOISE_K
#define UCUDA_LYAP_NOISE_K 100
#endif
#define UCUDA_NUMB_EPS (sizeof(numb) == 4 ? 1.1920928955078125e-7 : 2.220446049250313e-16)

// Клоны как добавка к шагу (см. UcudaAdNoExtra): на попытке — вложенный шаг каждого клона
// и норма ошибки возмущения, на принятии — клоны становятся новыми.
template <int NC>
struct UcudaLyapClones {
    numb y[NC * AMOUNTOFX], F[NC * AMOUNTOFX];     // клоны и f(клонов)
    numb Yc[NC * AMOUNTOFX], Fc[NC * AMOUNTOFX];   // попытка
    numb Ec[UCUDA_AD_MAXLOW * AMOUNTOFX];
    numb rtol;
    int  nlo;
    bool active;                                   // false — транзиент, клонов ещё нет

    template <class K>
    UCUDA_HD numb attempt(const K& k, const numb* a, const numb* X, const numb* Y, const numb* E, numb h) {
        if (!active) return 0;
        constexpr int n = AMOUNTOFX;
        numb W[UCUDA_AD_MAXSTAGES * AMOUNTOFX];
        numb worst = 0;
        for (int c = 0; c < NC; ++c) {
            k.emb(y + c * n, F + c * n, a, h, Yc + c * n, Ec, Fc + c * n, W);
            numb d0 = 0, d1 = 0;
            for (int i = 0; i < n; ++i) {
                const numb p0 = y[c * n + i] - X[i], p1 = Yc[c * n + i] - Y[i];
                d0 += p0 * p0; d1 += p1 * p1;
            }
            // sc^2 = (rtol * RMS(delta))^2, delta — большее из начала и конца попытки.
            constexpr numb inv_n = (numb)1 / (numb)n;
            const numb sc2 = rtol * rtol * (d0 > d1 ? d0 : d1) * inv_n;
            if (!(sc2 > 0)) continue;
            // Пол масштаба покомпонентно: sc_i = max(rtol * RMS(delta), K * eps * max|x_i, y_i|) —
            // шум округления разности клона и x (и их оценок E) в этой компоненте. Без пола при
            // rtol * |delta| ниже разрешения double (rtol 1e-9, eps 1e-8 у Лоренца) ошибка упиралась
            // в шум, и регулятор дробил шаг: ~170x шагов при тех же показателях.
            // Веса 1/sc_i^2: одно деление на клон, ещё по одному — на компоненту, где сработал пол.
            const numb isc2 = 1 / sc2;
            numb s5 = 0, s3 = 0;
            for (int i = 0; i < n; ++i) {
                numb ax = fabs(X[i]);
                const numb a1 = fabs(Y[i]), a2 = fabs(y[c * n + i]), a3 = fabs(Yc[c * n + i]);
                if (a1 > ax) ax = a1;
                if (a2 > ax) ax = a2;
                if (a3 > ax) ax = a3;
                const numb fl = (numb)UCUDA_LYAP_NOISE_K * (numb)UCUDA_NUMB_EPS * ax;
                const numb fl2 = fl * fl;
                numb w = isc2;
                if (fl2 > sc2) w = 1 / fl2;
                const numb e5 = Ec[i] - E[i];
                s5 += e5 * e5 * w;
                if (nlo >= 2) { const numb e3 = Ec[n + i] - E[n + i]; s3 += e3 * e3 * w; }
            }
            numb err;
            if (nlo >= 2) err = (s5 == 0 && s3 == 0) ? (numb)0 : s5 / sqrt((s5 + (numb)0.01 * s3) * (numb)n);
            else          err = sqrt(s5 * inv_n);
            if (err != err) return err;
            if (err > worst) worst = err;
        }
        return worst;
    }
    UCUDA_HD void commit() {
        if (!active) return;
        for (int i = 0; i < NC * AMOUNTOFX; ++i) { y[i] = Yc[i]; F[i] = Fc[i]; }
    }
};

// Одна точка. Свежая (carried = false): транзиент tTr по одной x, затем клоны (направления —
// подпоследовательность seq), nWarm неучётных блоков, nBlocks учётных. Перенесённая с
// предыдущей точки цепочки continuation (carried = true, клоны в cl уже прикреплены, S после
// ucuda_ad_restart): транзиента по одной x нет — он идёт блоками с клонами без учёта
// (max(tTr/NT, nWarm) блоков, как settleBlocks у постоянного шага), чтобы щуп не терял
// ориентацию. res[NC] — показатели. Возвращает 1, 0 — разлёт или отмена.
template <int NC, class K>
UCUDA_HD inline int ucuda_lyap_point(UcudaAdaptState& S, const K& Kf, const numb* a, const UcudaAdaptParams& P,
                                     UcudaLyapClones<NC>& cl, const bool carried, const numb tTr, const numb NT,
                                     const int nBlocks, const int nWarm, const numb eps, const int renorm,
                                     const numb maxValue, const int seq, numb* res,
                                     const volatile int* cancelFlag) {
    constexpr int n = AMOUNTOFX;
    cl.rtol = P.rtol > 0 ? P.rtol : P.atol[0];   // чисто абсолютный допуск — как относительный для delta
    cl.nlo  = P.nlo;
    if (!carried) cl.active = false;
    numb z[NC * n], acc[NC], accWarm[NC];
    for (int c = 0; c < NC; ++c) { acc[c] = 0; accWarm[c] = 0; }
    const numb t0  = carried ? S.t : tTr;          // начало блоков с клонами
    int nW = nWarm;
    if (carried) { const int ns = (int)(tTr / NT); if (ns > nW) nW = ns; }
    const int  nAll = nBlocks + nW;
    numb tAcc0 = t0;                               // начало учётных блоков
    const numb tEnd = t0 + (numb)nAll * NT;
    numb tb = t0;                                  // ближайшая граница
    int  k = 0;                                    // 0 — до прикрепления клонов, дальше — номер блока
    int  cnt = 0;
    for (;;) {
        if (!(S.t < tb)) {
            const bool clipped = (k == 0) || (renorm == 0);   // шаг к границе обрезался
            if (k == 0) { if (!carried) ucuda_lyap_init_clones<NC>(S.X, eps, seq, cl.y, z); }
            else        ucuda_lyap_renorm<NC>(S.X, eps, cl.y, z, k > nW ? acc : accWarm);
            if (k == nW) tAcc0 = S.t;
            for (int c = 0; c < NC; ++c) Kf.rhs(cl.y + c * n, a, cl.F + c * n);   // клоны сдвинуты (или новые a) — f заново
            cl.active = true;
            if (clipped) S.h = S.hfree;
            ++k;
            if (renorm != 0)   // шаг длиннее NT мог пройти несколько границ
                while (k <= nAll && !(S.t < t0 + (numb)k * NT)) ++k;
            if (k > nAll) break;
            tb = t0 + (numb)k * NT;   // теперь S.t < tb — попытка в той же итерации
        }
        // Одна попытка за итерацию (см. ucuda_ad_try_x), граница блока — в той же итерации,
        // что и шаг: варп шагает в ногу, а перенормировка — короткая ветка у части нитей.
        if (!ucuda_ad_try_x(S, Kf, a, P, (k == 0 || renorm == 0) ? tb : tEnd, cl)) {
            if (S.diverged) return 0;
            continue;
        }
        // Порог разлёта (опорная и клоны) и флаг отмены — раз в CHECK_INTERVAL принятых шагов,
        // как у постоянного шага; NaN/inf не принимается вовсе (ошибка попытки бесконечна).
        if (++cnt == CHECK_INTERVAL) {
            cnt = 0;
            if (ucuda_lyap_out(S.X, maxValue)) return 0;
            if (k > 0)
                for (int c = 0; c < NC; ++c)
                    if (ucuda_lyap_out(cl.y + c * n, maxValue)) return 0;
            if (cancelFlag != nullptr && *cancelFlag != 0) return 0;
        }
    }
    if (ucuda_lyap_out(S.X, maxValue)) return 0;   // хвост короче CHECK_INTERVAL
    for (int c = 0; c < NC; ++c)
        if (ucuda_lyap_out(cl.y + c * n, maxValue)) return 0;
    const numb tIntegrated = S.t - tAcc0;   // последний блок кончается ровно в tEnd при любом renorm
    for (int c = 0; c < NC; ++c) res[c] = acc[c] / tIntegrated;
    return 1;
}

// Ось настройки шага (kind != 0): значение v — в параметры P (коды — UCUDA_AXIS_* в adaptive_part.cu).
UCUDA_HD inline void ucuda_lyap_apply_axis(const int kind, const numb v, const numb tolRatio, UcudaAdaptParams& P) {
    if (kind == 2) P.rtol = v;
    else if (kind == 3) { for (int j = 0; j < AMOUNTOFX; ++j) P.atol[j] = v; }
    else if (kind == 4) { P.rtol = v; for (int j = 0; j < AMOUNTOFX; ++j) P.atol[j] = v * tolRatio; }
    else if (kind >= 10 && kind < 10 + UCUDA_CTL_NPAR) P.c[kind - 10] = v;
}

// Свип 1D по цепочке точек — continuation (каждая точка стартует с конечного состояния
// предыдущей; переносятся x, шаг и память регулятора — ucuda_ad_restart — и прикреплённые
// клоны) или, при continuation = 0, классический (каждая точка — заново от baseX, направления
// клонов — подпоследовательность j, как idx у ядра). Значения оси — ucuda_node_value_cont
// (reverse разворачивает цепочку). Разлёт рвёт цепочку: следующая точка — заново от baseX с
// новым направлением (подпоследовательность j + 1), как у постоянного шага. result[j*NC ..] —
// показатели (NaN — разлёт), adStats[j*4 ..] — статистика шага; тик прогресса — точка.
template <int NC, class K>
UCUDA_HD inline void ucuda_lyap_chain(const K& Kf, const int continuation, const int nPts, const numb lo,
                                      const numb hi, const int reverse, const int logScale, const int mutParamIdx,
                                      const numb* baseValues, const int amountOfValues, const numb* baseX,
                                      const UcudaAdaptParams& Pbase, const int axisKind, const numb tolRatio,
                                      const numb tTr, const numb NT, const int nBlocks, const int nWarm,
                                      const numb eps, const int renorm, const numb maxValue, numb* result,
                                      numb* adStats, const volatile int* cancelFlag, int* progressCounter) {
    numb x[AMOUNTOFX];
    numb a[64];   // kMaxAmountOfValues в движке
    for (int i = 0; i < AMOUNTOFX; ++i) x[i] = baseX[i];
    for (int i = 0; i < amountOfValues && i < 64; ++i) a[i] = baseValues[i];
    UcudaAdaptParams P = Pbase;
    UcudaAdaptState S;
    UcudaLyapClones<NC> cl;
    cl.active = false;
    bool attached = false;
    int  seq = 0;
    const numb qnan = sqrt((numb)-1);
    for (int j = 0; j < nPts; ++j) {
        if (cancelFlag != nullptr && *cancelFlag != 0) return;
        if (progressCounter != nullptr) {
#ifdef __CUDA_ARCH__
            atomicAdd(progressCounter, 1);
#else
            ++*progressCounter;
#endif
        }
        const numb v = ucuda_node_value_cont(j, nPts, lo, hi, logScale != 0, continuation != 0 && reverse != 0);
        if (axisKind == 0) a[mutParamIdx] = v;
        else ucuda_lyap_apply_axis(axisKind, v, tolRatio, P);
        if (!continuation) { attached = false; seq = j; }
        if (!attached) {
            for (int i = 0; i < AMOUNTOFX; ++i) x[i] = baseX[i];
            ucuda_ad_init(S, Kf, AMOUNTOFX, x, (numb)0, a, P);
        } else {
            ucuda_ad_restart(S, Kf, (numb)0, a, P);
        }
        numb res[NC];
        const int ok = ucuda_lyap_out(S.X, maxValue) ? 0
                     : ucuda_lyap_point<NC>(S, Kf, a, P, cl, attached, tTr, NT, nBlocks, nWarm, eps, renorm,
                                            maxValue, seq, res, cancelFlag);
        for (int m = 0; m < NC; ++m) result[(size_t)j * NC + m] = ok ? res[m] : qnan;
        if (adStats != nullptr) {
            numb* st = adStats + (size_t)j * 4;
            st[0] = (numb)S.st.nacc; st[1] = (numb)S.st.nrej; st[2] = (numb)S.st.nforced;
            st[3] = S.st.nacc > 0 ? S.st.hsum / (numb)S.st.nacc : (numb)0;
        }
        if (ok) attached = continuation != 0;
        else    { attached = false; seq = j + 1; }
    }
}

// Классический свип (каждая точка заново от baseX, направления — подпоследовательность j)
// по куску точек [j0, j1) сетки из nPts: ровно то, что ucuda_lyap_chain делает при
// continuation = 0, но кусками — CPU раздаёт их потокам. Выходы — с j0: result[(j-j0)*NC ..],
// adStats[(j-j0)*4 ..]. Тик прогресса — точка.
template <int NC, class K>
UCUDA_HD inline void ucuda_lyap_classic_range(const K& Kf, const int j0, const int j1, const int nPts,
                                              const numb lo, const numb hi, const int logScale,
                                              const int mutParamIdx, const numb* baseValues,
                                              const int amountOfValues, const numb* baseX,
                                              const UcudaAdaptParams& Pbase, const int axisKind,
                                              const numb tolRatio, const numb tTr, const numb NT,
                                              const int nBlocks, const int nWarm, const numb eps,
                                              const int renorm, const numb maxValue, numb* result,
                                              numb* adStats, const volatile int* cancelFlag,
                                              int* progressCounter) {
    numb a[64];   // kMaxAmountOfValues в движке
    for (int i = 0; i < amountOfValues && i < 64; ++i) a[i] = baseValues[i];
    UcudaAdaptParams P = Pbase;
    UcudaAdaptState S;
    UcudaLyapClones<NC> cl;
    cl.active = false;
    const numb qnan = sqrt((numb)-1);
    for (int j = j0; j < j1; ++j) {
        if (cancelFlag != nullptr && *cancelFlag != 0) return;
        if (progressCounter != nullptr) {
#ifdef __CUDA_ARCH__
            atomicAdd(progressCounter, 1);
#else
            ++*progressCounter;
#endif
        }
        const numb v = ucuda_node_value_cont(j, nPts, lo, hi, logScale != 0, false);
        if (axisKind == 0) a[mutParamIdx] = v;
        else ucuda_lyap_apply_axis(axisKind, v, tolRatio, P);
        ucuda_ad_init(S, Kf, AMOUNTOFX, baseX, (numb)0, a, P);
        numb res[NC];
        const int ok = ucuda_lyap_out(S.X, maxValue) ? 0
                     : ucuda_lyap_point<NC>(S, Kf, a, P, cl, false, tTr, NT, nBlocks, nWarm, eps, renorm,
                                            maxValue, j, res, cancelFlag);
        const size_t r = (size_t)(j - j0);
        for (int m = 0; m < NC; ++m) result[r * NC + m] = ok ? res[m] : qnan;
        if (adStats != nullptr) {
            numb* st = adStats + r * 4;
            st[0] = (numb)S.st.nacc; st[1] = (numb)S.st.nrej; st[2] = (numb)S.st.nforced;
            st[3] = S.st.nacc > 0 ? S.st.hsum / (numb)S.st.nacc : (numb)0;
        }
    }
}

#endif // UCUDA_AD_LYAPUNOV
