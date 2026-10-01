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

// Параметры адаптивного шага — один экземпляр на запуск. Только numb и int
// фиксированного размера: хост заполняет ту же структуру и передаёт её в ядро.
struct UcudaAdaptParams {
    numb rtol;
    numb atol[UCUDA_AD_MAXN];   // по переменным (скаляр хост размножает)
    numb h0;                    // <= 0 — автоматически (select_initial_step)
    numb hmin;                  // <= 0 — 10 ulp(t), как в scipy
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

// sc_i = atol_i + rtol*max(|y0_i|, |y1_i|) — как у Хайрера и scipy.
UCUDA_HD inline numb ucuda_ctl_scale(const UcudaCtlIn& in, int i) {
    return in.atol[i] + in.rtol * fmax(fabs(in.y0[i]), fabs(in.y1[i]));
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
    int e;
    const numb m = frexp(x, &e);
    return (float)e + log2f((float)m);
}

UCUDA_HD inline numb ucuda_ctl_pow(numb x, numb y) {
#ifdef UCUDA_AD_EXACT_CTL
    return pow(x, y);
#else
    return (numb)exp2f((float)y * ucuda_ctl_log2(x));
#endif
}

UCUDA_HD inline numb ucuda_clamp(numb x, numb lo, numb hi) { return fmin(hi, fmax(lo, x)); }

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
    const float fac = fmaxf((float)facc2, fminf((float)facc1, exp2f(lfac) * isafe));
    numb hnew = in.h * (numb)(1.0f / fac);
#endif
    o.err = err;
    if (err <= 1) {
        o.accept = 1;
        m.user[0] = fmax(err, (numb)1e-4);
        if (in.hmax > 0 && hnew > in.hmax) hnew = in.hmax;
        if (in.nrej > 0) hnew = fmin(hnew, in.h);
    } else {
        o.accept = 0;
#ifdef UCUDA_AD_EXACT_CTL
        hnew = in.h / fmin(facc1, fac11 / safe);
#else
        hnew = in.h * (numb)(1.0f / fminf((float)facc1, exp2f(l11) * isafe));
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
        numb factor = (err == 0) ? maxf : fmin(maxf, safety * ucuda_ctl_pow(err, expo));
        if (in.nrej > 0) factor = fmin((numb)1, factor);
        o.h = in.h * factor;
        o.accept = 1;
    } else {
        o.h = in.h * fmax(minf, safety * ucuda_ctl_pow(err, expo));
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
    if (err > 1 || m.nacc < 1 || err == 0 || !(m.err[0] > 0)) { ucuda_ctrl_elem(in, err, o); return; }
    const numb facmin = in.c[1], facmax = in.c[2];
    const numb kI = in.cc->b1, kP = in.cc->b2;
#ifdef UCUDA_AD_EXACT_CTL
    numb fac = exp2(in.cc->lth_pi) * pow(err, -kI) * pow(m.err[0] / err, kP);
#else
    // (th/err)^kI (err_prev/err)^kP = 2^(lth - kI L - kP L + kP L_prev): одна exp2, без деления.
    const float le = ucuda_ctl_log2(err), l0 = ucuda_ctl_log2(m.err[0]);
    numb fac = (numb)exp2f((float)in.cc->lth_pi - (float)kI * le + (float)kP * (l0 - le));
#endif
    fac = ucuda_clamp(fac, facmin, in.nrej > 0 ? (numb)1 : facmax);
    o.err = err; o.accept = 1; o.h = in.h * fac;
}

UCUDA_HD inline void ucuda_ctrl_filter(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) {
    const numb err = ucuda_ctl_err(in);
    if (err > 1 || m.nacc < 2 || err == 0 || !(m.err[0] > 0) || !(m.err[1] > 0)) {
        ucuda_ctrl_elem(in, err, o); return;
    }
    const numb facmin = in.c[1], facmax = in.c[2];
    const numb b1 = in.cc->b1, b2 = in.cc->b2, b3 = in.cc->b3, a2 = in.c[6], a3 = in.c[7];
    const numb e2 = m.nacc >= 3 ? m.err[1] : m.err[0];
#ifdef UCUDA_AD_EXACT_CTL
    const numb rho1 = in.h / m.h[0];                         // h_n / h_{n-1}
    const numb rho2 = m.nacc >= 3 ? m.h[0] / m.h[1] : (numb)1;
    numb rho = exp2(in.cc->lth_f) * pow(err, -b1) * pow(m.err[0], -b2) * pow(e2, -b3)
             * pow(rho1, -a2) * pow(rho2, -a3);
#else
    // Произведение степеней — сумма логарифмов и одна exp2; отношения шагов — разности
    // log2 h, без деления.
    const float lh0 = ucuda_ctl_log2(m.h[0]);
    const float lr1 = ucuda_ctl_log2(in.h) - lh0;                            // log2 (h_n / h_{n-1})
    const float lr2 = m.nacc >= 3 ? lh0 - ucuda_ctl_log2(m.h[1]) : 0.0f;    // log2 (h_{n-1} / h_{n-2})
    numb rho = (numb)exp2f((float)in.cc->lth_f
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
    int  diverged;                                 // решение ушло в NaN/inf даже на h_min
    UcudaCtlMem mem;
    UcudaCtlConst cc;                              // производные константы законов
    UcudaAdaptStats st;
    numb* log;                                     // лог попыток: {t, h, err, код} на запись
    int  log_cap;                                  //   код: 1 — принят, 0 — отвергнут,
    int  log_n;                                    //        2 — вынужденный
};

// 10 ulp(t) — нижняя граница шага по умолчанию (scipy: min_step).
UCUDA_HD inline numb ucuda_ad_hmin_auto(numb t) {
    return 10 * fabs(nextafter(t, (numb)1e308) - t);
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
    S.last_forced = 0; S.diverged = 0;
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
    S.last_forced = 0; S.diverged = 0;
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

// Один принятый шаг из (S.t, S.X), не переходя tEnd. Вызывать при S.t < tEnd.
// После P.maxrej отказов подряд (если > 0) попытка делается с h_min и принимается без
// условий — предел числа проверок на шаг. Если ошибка не число даже на h_min (решение ушло в NaN/inf), шаг не
// принимается, а S.diverged = 1: дальше интегрировать бессмысленно, а
// вынужденные шаги по 10 ulp(t) не закончились бы никогда. Вызывающий обязан
// проверять diverged в своих циклах.
template <class K, class Ext>
UCUDA_HD inline void ucuda_ad_step_x(UcudaAdaptState& S, const K& k, const numb* a,
                                     const UcudaAdaptParams& P, numb tEnd, Ext& ext) {
    const int n = UCUDA_AD_N(S);
    numb Y[UCUDA_AD_NMAX], E[UCUDA_AD_MAXLOW * UCUDA_AD_NMAX], F1[UCUDA_AD_NMAX];
#ifdef UCUDA_AD_NO_DENSE
    numb W[UCUDA_AD_MAXSTAGES * UCUDA_AD_NMAX];    // стадии попытки: после шага не нужны
#else
    numb* W = S.W;
#endif
    int nrej = 0;
    const numb hmin = P.hmin > 0 ? P.hmin : ucuda_ad_hmin_auto(S.t);   // t в повторах не меняется
    for (;;) {
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
        S.st.nrhs += P.emb_rhs;
        const numb err_extra = ext.attempt(k, a, S.X, Y, E, h);

        UcudaCtlIn in;
        in.y0 = S.X; in.y1 = Y; in.yerr = E; in.atol = P.atol; in.c = P.c;
        in.rtol = P.rtol; in.h = h; in.hmin = hmin; in.hmax = P.hmax; in.t = S.t;
        in.n = n; in.nlo = P.nlo; in.q = P.q; in.nrej = nrej; in.cc = &S.cc; in.err_extra = err_extra;
        UcudaCtlOut o;
        ucuda_step_ctrl(P.ctrl, in, S.mem, o);

        if (forced && !(o.err <= (numb)1e300)) { S.diverged = 1; return; }   // !(<=) ловит и NaN
        // Без повторов принимается любая попытка с конечной ошибкой; NaN/inf — как обычный
        // отказ: шаг уменьшается, а на h_min разлёт ловится выше.
        const int take = o.accept || forced || (P.maxrej < 0 && o.err <= (numb)1e300);
        if (S.log != nullptr && S.log_n < S.log_cap) {
            numb* r = S.log + 4 * S.log_n++;
            r[0] = S.t; r[1] = h; r[2] = o.err;
            r[3] = o.accept ? (numb)1 : (take ? (numb)2 : (numb)0);
        }
        if (take) {
            ext.commit();
            if (!o.accept) S.st.nforced++;
            S.tp = S.t; S.hp = h;
            for (int i = 0; i < n; ++i) {
#ifndef UCUDA_AD_NO_DENSE
                S.Xp[i] = S.X[i]; S.Fp[i] = S.F0[i];
#endif
                S.X[i] = Y[i];    S.F0[i] = F1[i];
            }
            S.t = last ? tEnd : tn;
            for (int j = UCUDA_CTL_HIST - 1; j > 0; --j) { S.mem.h[j] = S.mem.h[j - 1]; S.mem.err[j] = S.mem.err[j - 1]; }
            S.mem.h[0] = h; S.mem.err[0] = o.err; S.mem.nacc++;
            if (S.st.nacc == 0 || h < S.st.hmin) S.st.hmin = h;
            if (S.st.nacc == 0 || h > S.st.hmax) S.st.hmax = h;
            S.st.hsum += h; S.st.nacc++;
            S.h = (o.h > 0) ? o.h : h;   // регулятор вернул мусор — оставить шаг
            if (!last) S.hfree = S.h;
#ifndef UCUDA_AD_NO_DENSE
            S.dense_ready = 0;
#endif
            S.last_forced = forced;
            return;
        }
        S.st.nrej++; ++nrej;
        numb hn = o.h;
        if (!(hn <= (numb)0.9 * h)) hn = (numb)0.9 * h;
        S.h = hn;
    }
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
