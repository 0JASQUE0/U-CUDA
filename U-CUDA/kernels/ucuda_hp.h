#pragma once
// ucuda_hp.h — double-double арифметика (~32 значащих цифры) для расчётов
// повышенной точности.
//
// Зачем: оценщик порядка на вкладке Order считает p = log2(E1/E2), где
// E1 = max|y_h - y_h/2|. Это вычитание близких чисел, и в double разность
// садится на полку округления 2*eps*scale*sqrt(4N) ~ 3e-13 (см. run_order_cpu
// в order_session.cpp). Схема порядка p даёт E1 ~ h^p, поэтому при p >= 8
// пригодного окна по h практически не остаётся: полка приходит раньше
// асимптотики. Мантисса 106 бит (eps = 4.9e-32) опускает полку на 16 десятичных
// порядков и возвращает окно даже для p = 16.
//
// Тип представляет число НЕПЕРЕКРЫВАЮЩЕЙСЯ парой double: hi несёт старшие
// разряды, lo — остаток, причём |lo| <= 0.5 ulp(hi). Все операции построены на
// безошибочных преобразованиях (two_sum / two_prod Деккера-Кнута): они дают
// точную сумму/произведение в виде такой же пары, поэтому разряды не теряются
// там, где обычный double уже округлил.
//
// ВАЖНО про флаги компилятора. Вся конструкция держится на том, что выражения
// вида (s - a) вычисляются буквально. При /fp:fast компилятор вправе свернуть
// (a + b) - a в b, и тогда все err-члены станут нулями — точность молча
// упадёт до обычного double. Собирать только с /fp:precise (умолчание MSVC)
// или /fp:strict. По той же причине здесь нет std::fma: на MSVC без /arch:AVX2
// это вызов библиотечной функции, который стоит дороже, чем сплиттинг Деккера.
//
// Элементарные функции дают ~1e-31 относительной точности (не корректное
// округление последнего разряда — для оценки порядка схемы этого с запасом).
// Аргумент sin/cos приводится умножением на dd-константу pi/2, поэтому ошибка
// приведения растёт как |x|*1e-32: при |x| ~ 1e4 это 1e-28.
//
// Источник алгоритмов — стандартный набор QD (Bailey, Hida, Li), переписанный
// здесь без внешних зависимостей: заголовок читает и cl.exe при сборке DLL
// пользовательской КРС (krs_cpu.cpp), и сам проект.

#include <cmath>
#include <limits>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define UCUDA_HP_HD __host__ __device__
#else
#define UCUDA_HP_HD
#endif

namespace ucuda {

// 2^27 + 1 — множитель сплиттинга Деккера: разрезает мантиссу 53 бита на две
// половины по 26-27 бит, произведение которых представимо точно.
constexpr double kSplitter     = 134217729.0;
constexpr double kSplitThresh  = 6.69692879491417e+299;

struct dd {
    double hi, lo;

    constexpr dd()                     : hi(0.0), lo(0.0) {}
    constexpr dd(double x)             : hi(x),   lo(0.0) {}
    constexpr dd(double h, double l)   : hi(h),   lo(l)   {}

    // explicit: иначе dd стал бы кандидатом в каждом вызове double-перегрузки
    // (std::pow, std::sqrt, ...) и выиграл бы там, где нужна dd-версия.
    UCUDA_HP_HD explicit operator double() const { return hi; }
    // Отдельно к int: MSVC не строит цепочку "operator double + сужение" при
    // явном приведении, а (int)p в configCUDA.h (ucmplx pow) на этом падает.
    UCUDA_HP_HD explicit operator int() const { return (int)hi; }
};

// ---- Безошибочные преобразования ----

// Требует |a| >= |b|. Возвращает s = a+b округлённое и точный остаток err.
UCUDA_HP_HD inline double quick_two_sum(double a, double b, double& err) {
    const double s = a + b;
    err = b - (s - a);
    return s;
}

UCUDA_HP_HD inline double two_sum(double a, double b, double& err) {
    const double s  = a + b;
    const double bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

UCUDA_HP_HD inline void split_d(double a, double& hi, double& lo) {
    // Ветка для гигантских |a| нужна, чтобы kSplitter * a не переполнилось.
    if (a > kSplitThresh || a < -kSplitThresh) {
        a *= 3.7252902984619140625e-09;           // 2^-28
        const double t = kSplitter * a;
        hi = t - (t - a);
        lo = a - hi;
        hi *= 268435456.0;                        // 2^28
        lo *= 268435456.0;
        return;
    }
    const double t = kSplitter * a;
    hi = t - (t - a);
    lo = a - hi;
}

UCUDA_HP_HD inline double two_prod(double a, double b, double& err) {
    double ah, al, bh, bl;
    const double p = a * b;
    split_d(a, ah, al);
    split_d(b, bh, bl);
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return p;
}

UCUDA_HP_HD inline double two_sqr(double a, double& err) {
    double h, l;
    const double q = a * a;
    split_d(a, h, l);
    err = ((h * h - q) + 2.0 * h * l) + l * l;
    return q;
}

// ---- Арифметика ----

UCUDA_HP_HD inline dd operator-(const dd& a) { return dd(-a.hi, -a.lo); }

UCUDA_HP_HD inline dd operator+(const dd& a, const dd& b) {
    double s1, s2, t1, t2;
    s1 = two_sum(a.hi, b.hi, s2);
    t1 = two_sum(a.lo, b.lo, t2);
    s2 += t1;
    s1 = quick_two_sum(s1, s2, s2);
    s2 += t2;
    s1 = quick_two_sum(s1, s2, s2);
    return dd(s1, s2);
}

UCUDA_HP_HD inline dd operator+(const dd& a, double b) {
    double s1, s2;
    s1 = two_sum(a.hi, b, s2);
    s2 += a.lo;
    s1 = quick_two_sum(s1, s2, s2);
    return dd(s1, s2);
}

UCUDA_HP_HD inline dd operator+(double a, const dd& b) { return b + a; }

UCUDA_HP_HD inline dd operator-(const dd& a, const dd& b) { return a + (-b); }
UCUDA_HP_HD inline dd operator-(const dd& a, double b)    { return a + (-b); }
UCUDA_HP_HD inline dd operator-(double a, const dd& b)    { return (-b) + a; }

UCUDA_HP_HD inline dd operator*(const dd& a, const dd& b) {
    double p1, p2;
    p1 = two_prod(a.hi, b.hi, p2);
    p2 += (a.hi * b.lo + a.lo * b.hi);
    p1 = quick_two_sum(p1, p2, p2);
    return dd(p1, p2);
}

UCUDA_HP_HD inline dd operator*(const dd& a, double b) {
    double p1, p2;
    p1 = two_prod(a.hi, b, p2);
    p2 += a.lo * b;
    p1 = quick_two_sum(p1, p2, p2);
    return dd(p1, p2);
}

UCUDA_HP_HD inline dd operator*(double a, const dd& b) { return b * a; }

// Умножение на степень двойки — точное и без единого преобразования.
UCUDA_HP_HD inline dd mul_pwr2(const dd& a, double b) { return dd(a.hi * b, a.lo * b); }

UCUDA_HP_HD inline dd sqr(const dd& a) {
    double p1, p2;
    p1 = two_sqr(a.hi, p2);
    p2 += 2.0 * a.hi * a.lo;
    p2 += a.lo * a.lo;
    p1 = quick_two_sum(p1, p2, p2);
    return dd(p1, p2);
}

// Три шага уточнения частного: каждый снимает ~53 бита невязки.
UCUDA_HP_HD inline dd operator/(const dd& a, const dd& b) {
    double q1 = a.hi / b.hi;
    dd r = a - b * q1;
    const double q2 = r.hi / b.hi;
    r = r - b * q2;
    const double q3 = r.hi / b.hi;
    double e;
    q1 = quick_two_sum(q1, q2, e);
    return dd(q1, e) + q3;
}

UCUDA_HP_HD inline dd operator/(const dd& a, double b)    { return a / dd(b); }
UCUDA_HP_HD inline dd operator/(double a, const dd& b)    { return dd(a) / b; }

UCUDA_HP_HD inline dd& operator+=(dd& a, const dd& b) { a = a + b; return a; }
UCUDA_HP_HD inline dd& operator+=(dd& a, double b)    { a = a + b; return a; }
UCUDA_HP_HD inline dd& operator-=(dd& a, const dd& b) { a = a - b; return a; }
UCUDA_HP_HD inline dd& operator-=(dd& a, double b)    { a = a - b; return a; }
UCUDA_HP_HD inline dd& operator*=(dd& a, const dd& b) { a = a * b; return a; }
UCUDA_HP_HD inline dd& operator*=(dd& a, double b)    { a = a * b; return a; }
UCUDA_HP_HD inline dd& operator/=(dd& a, const dd& b) { a = a / b; return a; }
UCUDA_HP_HD inline dd& operator/=(dd& a, double b)    { a = a / b; return a; }

// ---- Сравнения ----

UCUDA_HP_HD inline bool operator==(const dd& a, const dd& b) { return a.hi == b.hi && a.lo == b.lo; }
UCUDA_HP_HD inline bool operator==(const dd& a, double b)    { return a.hi == b && a.lo == 0.0; }
UCUDA_HP_HD inline bool operator==(double a, const dd& b)    { return b == a; }
UCUDA_HP_HD inline bool operator!=(const dd& a, const dd& b) { return !(a == b); }
UCUDA_HP_HD inline bool operator!=(const dd& a, double b)    { return !(a == b); }
UCUDA_HP_HD inline bool operator!=(double a, const dd& b)    { return !(b == a); }

UCUDA_HP_HD inline bool operator<(const dd& a, const dd& b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
UCUDA_HP_HD inline bool operator<(const dd& a, double b) {
    return a.hi < b || (a.hi == b && a.lo < 0.0);
}
UCUDA_HP_HD inline bool operator<(double a, const dd& b) {
    return a < b.hi || (a == b.hi && b.lo > 0.0);
}
UCUDA_HP_HD inline bool operator>(const dd& a, const dd& b) { return b < a; }
UCUDA_HP_HD inline bool operator>(const dd& a, double b)    { return b < a; }
UCUDA_HP_HD inline bool operator>(double a, const dd& b)    { return b < a; }
UCUDA_HP_HD inline bool operator<=(const dd& a, const dd& b) { return !(b < a); }
UCUDA_HP_HD inline bool operator<=(const dd& a, double b)    { return !(b < a); }
UCUDA_HP_HD inline bool operator<=(double a, const dd& b)    { return !(b < a); }
UCUDA_HP_HD inline bool operator>=(const dd& a, const dd& b) { return !(a < b); }
UCUDA_HP_HD inline bool operator>=(const dd& a, double b)    { return !(a < b); }
UCUDA_HP_HD inline bool operator>=(double a, const dd& b)    { return !(a < b); }

// ---- Константы ----
// hi + lo несут ~32 цифры; double-литерал дал бы только 17, и весь выигрыш
// точности терялся бы на первом же вхождении pi в правую часть.

constexpr dd c_pi  (3.141592653589793,  1.2246467991473532e-16);
constexpr dd c_2pi (6.283185307179586,  2.4492935982947064e-16);
constexpr dd c_pi2 (1.5707963267948966, 6.123233995736766e-17);
constexpr dd c_pi4 (0.7853981633974483, 3.061616997868383e-17);
constexpr dd c_e   (2.718281828459045,  1.4456468917292502e-16);
constexpr dd c_ln2 (0.6931471805599453, 2.3190468138462996e-17);
constexpr dd c_ln10(2.302585092994046, -2.1707562233822494e-16);

// 1/n!, n = 0..32 — хватает и ряду exp (после приведения аргумента), и рядам
// sin/cos на |r| <= pi/4.
constexpr dd c_inv_fact[33] = {
    dd(1.0, 0.0),
    dd(1.0, 0.0),
    dd(0.5, 0.0),
    dd(0.16666666666666666, 9.25185853854297e-18),
    dd(0.041666666666666664, 2.3129646346357427e-18),
    dd(0.008333333333333333, 1.1564823173178714e-19),
    dd(0.001388888888888889, -5.300543954373577e-20),
    dd(0.0001984126984126984, 1.7209558293420705e-22),
    dd(2.48015873015873e-05, 2.1511947866775882e-23),
    dd(2.7557319223985893e-06, -1.858393274046472e-22),
    dd(2.755731922398589e-07, 2.3767714622250297e-23),
    dd(2.505210838544172e-08, -1.448814070935912e-24),
    dd(2.08767569878681e-09, -1.20734505911326e-25),
    dd(1.6059043836821613e-10, 1.2585294588752098e-26),
    dd(1.1470745597729725e-11, 2.0655512752830745e-28),
    dd(7.647163731819816e-13, 7.03872877733453e-30),
    dd(4.779477332387385e-14, 4.399205485834081e-31),
    dd(2.8114572543455206e-15, 1.6508842730861433e-31),
    dd(1.5619206968586225e-16, 1.1910679660273754e-32),
    dd(8.22063524662433e-18, 2.2141894119604265e-34),
    dd(4.110317623312165e-19, 1.4412973378659527e-36),
    dd(1.9572941063391263e-20, -1.3643503830087908e-36),
    dd(8.896791392450574e-22, -7.911402614872376e-38),
    dd(3.868170170630684e-23, -8.843177655482344e-40),
    dd(1.6117375710961184e-24, -3.6846573564509766e-41),
    dd(6.446950284384474e-26, -1.9330404233703465e-42),
    dd(2.4795962632247976e-27, -1.2953730964765229e-43),
    dd(9.183689863795546e-29, 1.4303150396787322e-45),
    dd(3.279889237069838e-30, 1.5117542744029879e-46),
    dd(1.1309962886447716e-31, 1.0498015412959506e-47),
    dd(3.7699876288159054e-33, 2.5870347832750324e-49),
    dd(1.216125041553518e-34, 5.586290567888806e-51),
    dd(3.8003907548547434e-36, 1.7457158024652518e-52),
};

// ---- Классификация и округление ----

UCUDA_HP_HD inline bool isnan   (const dd& a) { return std::isnan(a.hi) || std::isnan(a.lo); }
UCUDA_HP_HD inline bool isinf   (const dd& a) { return std::isinf(a.hi); }
UCUDA_HP_HD inline bool isfinite(const dd& a) { return std::isfinite(a.hi) && std::isfinite(a.lo); }
UCUDA_HP_HD inline bool signbit (const dd& a) { return std::signbit(a.hi); }
UCUDA_HP_HD inline double to_double(const dd& a) { return a.hi; }

UCUDA_HP_HD inline dd fabs(const dd& a) { return a.hi < 0.0 ? dd(-a.hi, -a.lo) : a; }
UCUDA_HP_HD inline dd abs (const dd& a) { return fabs(a); }

UCUDA_HP_HD inline dd floor(const dd& a) {
    double h = std::floor(a.hi), l = 0.0;
    // hi уже целое -> дробная часть сидит в lo, и округлять надо её.
    if (h == a.hi) {
        l = std::floor(a.lo);
        h = quick_two_sum(h, l, l);
    }
    return dd(h, l);
}

UCUDA_HP_HD inline dd ceil(const dd& a) {
    double h = std::ceil(a.hi), l = 0.0;
    if (h == a.hi) {
        l = std::ceil(a.lo);
        h = quick_two_sum(h, l, l);
    }
    return dd(h, l);
}

// Отбрасывание дробной части (к нулю).
UCUDA_HP_HD inline dd aint(const dd& a) { return a.hi >= 0.0 ? floor(a) : ceil(a); }

UCUDA_HP_HD inline dd ldexp(const dd& a, int n) {
    return dd(std::ldexp(a.hi, n), std::ldexp(a.lo, n));
}

UCUDA_HP_HD inline dd copysign(const dd& a, const dd& b) {
    const dd m = fabs(a);
    return std::signbit(b.hi) ? -m : m;
}
UCUDA_HP_HD inline dd copysign(const dd& a, double b) { return copysign(a, dd(b)); }

UCUDA_HP_HD inline dd fmod(const dd& a, const dd& b) { return a - b * aint(a / b); }
UCUDA_HP_HD inline dd fmin(const dd& a, const dd& b) { return a < b ? a : b; }
UCUDA_HP_HD inline dd fmax(const dd& a, const dd& b) { return a > b ? a : b; }

// ---- Элементарные функции ----

UCUDA_HP_HD inline dd sqrt(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(0.0);
    if (a.hi < 0.0) return dd(std::numeric_limits<double>::quiet_NaN());
    // Одна итерация Ньютона по обратному корню поверх double-приближения.
    const double x  = 1.0 / std::sqrt(a.hi);
    const double ax = a.hi * x;
    const dd     d  = a - sqr(dd(ax));
    return dd(ax) + d.hi * (x * 0.5);
}

UCUDA_HP_HD inline dd hypot(const dd& x, const dd& y) {
    const dd ax = fabs(x), ay = fabs(y);
    if (ax.hi == 0.0) return ay;
    if (ay.hi == 0.0) return ax;
    // Деление на большую компоненту — чтобы квадрат не переполнился.
    if (ax < ay) return ay * sqrt(dd(1.0) + sqr(ax / ay));
    return ax * sqrt(dd(1.0) + sqr(ay / ax));
}

UCUDA_HP_HD inline dd exp(const dd& a) {
    if (a.hi <= -709.0) return dd(0.0);
    if (a.hi >=  709.0) return dd(std::numeric_limits<double>::infinity());
    if (a.hi == 0.0 && a.lo == 0.0) return dd(1.0);

    // Приведение: a = m*ln2 + r, затем r делится ещё на 512, чтобы ряд сходился
    // за десяток членов; масштаб возвращается девятью удвоениями.
    const double inv_k = 1.0 / 512.0;
    const double m = std::floor(a.hi / c_ln2.hi + 0.5);
    const dd r = mul_pwr2(a - c_ln2 * m, inv_k);

    dd p = sqr(r);
    dd s = r + mul_pwr2(p, 0.5);
    p = p * r;
    dd t = p * c_inv_fact[3];
    int i = 3;
    do {
        s = s + t;
        p = p * r;
        ++i;
        t = p * c_inv_fact[i];
    } while (std::fabs(t.hi) > inv_k * 1.0e-34 && i < 16);
    s = s + t;

    // exp(2x) = 2*exp(x) + exp(x)^2 при s = exp(x) - 1.
    for (int j = 0; j < 9; ++j) s = mul_pwr2(s, 2.0) + sqr(s);
    s = s + 1.0;
    return ldexp(s, (int)m);
}

UCUDA_HP_HD inline dd log(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0)
        return dd(-std::numeric_limits<double>::infinity());
    if (a.hi < 0.0) return dd(std::numeric_limits<double>::quiet_NaN());
    if (a.hi == 1.0 && a.lo == 0.0) return dd(0.0);
    // Ньютон по exp: один шаг удваивает верные разряды (1e-16 -> 1e-32).
    dd x(std::log(a.hi));
    x = x + a * exp(-x) - 1.0;
    return x;
}

UCUDA_HP_HD inline dd log10(const dd& a) { return log(a) / c_ln10; }
UCUDA_HP_HD inline dd log2 (const dd& a) { return log(a) / c_ln2;  }

// Ряды для |a| <= pi/4 — там они сходятся за 15 членов до 1e-33.
UCUDA_HP_HD inline dd sin_taylor(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(0.0);
    const dd x = -sqr(a);
    dd r = a, s = a, t;
    int i = 1;
    do {
        r = r * x;
        i += 2;
        t = r * c_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 32 && std::fabs(t.hi) > 1.0e-35);
    return s;
}

UCUDA_HP_HD inline dd cos_taylor(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(1.0);
    const dd x = -sqr(a);
    dd r(1.0), s(1.0), t;
    int i = 0;
    do {
        r = r * x;
        i += 2;
        t = r * c_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 32 && std::fabs(t.hi) > 1.0e-35);
    return s;
}

// Приведение к [-pi/4, pi/4] и разворот по четверти. Обе функции считаются
// заодно: atan2 и tan нужны обе, а стоят они вместе не дороже одной.
UCUDA_HP_HD inline void sincos(const dd& a, dd& sn, dd& cs) {
    const double nd = std::floor(a.hi / c_pi2.hi + 0.5);
    const dd r = a - c_pi2 * nd;
    const dd s = sin_taylor(r), c = cos_taylor(r);
    long long q = (long long)std::fmod(nd, 4.0);
    if (q < 0) q += 4;
    switch (q) {
        case 0:  sn =  s; cs =  c; break;
        case 1:  sn =  c; cs = -s; break;
        case 2:  sn = -s; cs = -c; break;
        default: sn = -c; cs =  s; break;
    }
}

UCUDA_HP_HD inline dd sin(const dd& a) { dd s, c; sincos(a, s, c); return s; }
UCUDA_HP_HD inline dd cos(const dd& a) { dd s, c; sincos(a, s, c); return c; }
UCUDA_HP_HD inline dd tan(const dd& a) { dd s, c; sincos(a, s, c); return s / c; }

UCUDA_HP_HD inline dd atan2(const dd& y, const dd& x) {
    if (x.hi == 0.0 && x.lo == 0.0) {
        if (y.hi == 0.0 && y.lo == 0.0)
            return dd(std::numeric_limits<double>::quiet_NaN());
        return y.hi > 0.0 ? c_pi2 : -c_pi2;
    }
    if (y.hi == 0.0 && y.lo == 0.0) return x.hi > 0.0 ? dd(0.0) : c_pi;

    const dd r  = sqrt(sqr(x) + sqr(y));
    const dd xx = x / r, yy = y / r;
    dd z(std::atan2(y.hi, x.hi));
    dd sz, cz;
    sincos(z, sz, cz);
    // Ньютон по той из функций, которая в этой точке круче: у второй
    // производная близка к нулю и шаг разносило бы.
    if (std::fabs(xx.hi) > std::fabs(yy.hi)) z = z + (yy - sz) / cz;
    else                                     z = z - (xx - cz) / sz;
    return z;
}

UCUDA_HP_HD inline dd atan(const dd& a) { return atan2(a, dd(1.0)); }

UCUDA_HP_HD inline dd asin(const dd& a) {
    const dd m = fabs(a);
    if (m > 1.0) return dd(std::numeric_limits<double>::quiet_NaN());
    if (m == 1.0) return a.hi > 0.0 ? c_pi2 : -c_pi2;
    return atan2(a, sqrt(dd(1.0) - sqr(a)));
}

UCUDA_HP_HD inline dd acos(const dd& a) {
    const dd m = fabs(a);
    if (m > 1.0) return dd(std::numeric_limits<double>::quiet_NaN());
    if (m == 1.0) return a.hi > 0.0 ? dd(0.0) : c_pi;
    return atan2(sqrt(dd(1.0) - sqr(a)), a);
}

UCUDA_HP_HD inline dd sinh(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(0.0);
    if (std::fabs(a.hi) > 0.05) {
        const dd e = exp(a);
        return mul_pwr2(e - dd(1.0) / e, 0.5);
    }
    // При малом аргументе exp(a) - exp(-a) — катастрофическое вычитание,
    // ряд же даёт полную точность.
    const dd x = sqr(a);
    dd r = a, s = a, t;
    int i = 1;
    do {
        r = r * x;
        i += 2;
        t = r * c_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 32 && std::fabs(t.hi) > std::fabs(a.hi) * 1.0e-34);
    return s;
}

UCUDA_HP_HD inline dd cosh(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(1.0);
    const dd e = exp(a);
    return mul_pwr2(e + dd(1.0) / e, 0.5);
}

UCUDA_HP_HD inline dd tanh(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(0.0);
    if (std::fabs(a.hi) > 0.05) {
        const dd e = exp(a), ie = dd(1.0) / e;
        return (e - ie) / (e + ie);
    }
    const dd s = sinh(a);
    return s / sqrt(dd(1.0) + sqr(s));
}

// Целая степень — бинарным возведением: точнее exp(n*log a) и работает при
// отрицательном основании.
UCUDA_HP_HD inline dd npwr(const dd& a, int n) {
    if (n == 0) return dd(1.0);
    dd r = a, s(1.0);
    int k = n < 0 ? -n : n;
    while (k > 0) {
        if (k & 1) s = s * r;
        k >>= 1;
        if (k > 0) r = sqr(r);
    }
    return n < 0 ? dd(1.0) / s : s;
}

UCUDA_HP_HD inline dd pow(const dd& a, const dd& b) {
    const double bi = std::floor(b.hi);
    if (b.lo == 0.0 && bi == b.hi && std::fabs(bi) <= 1024.0)
        return npwr(a, (int)bi);
    return exp(b * log(a));
}
UCUDA_HP_HD inline dd pow(const dd& a, double b)    { return pow(a, dd(b)); }
UCUDA_HP_HD inline dd pow(double a, const dd& b)    { return pow(dd(a), b); }
UCUDA_HP_HD inline dd pow(const dd& a, int n)       { return npwr(a, n); }

UCUDA_HP_HD inline dd cbrt(const dd& a) {
    if (a.hi == 0.0 && a.lo == 0.0) return dd(0.0);
    dd y(std::cbrt(a.hi));
    // Ньютон для y^3 = a: y <- y + (a/y^2 - y)/3.
    y = y + (a / sqr(y) - y) / 3.0;
    return y;
}

} // namespace ucuda

// windows.h определяет min/max макросами, и без NOMINMAX они съедают
// объявления numeric_limits::min() / ::max() ниже — ошибка при этом вылезает
// не здесь, а каскадом в <chrono> у того, кто нас подключил.
#ifdef min
#pragma push_macro("min")
#undef min
#define UCUDA_HP_POPPED_MIN
#endif
#ifdef max
#pragma push_macro("max")
#undef max
#define UCUDA_HP_POPPED_MAX
#endif

namespace std {

// Специализация нужна шаблонному коду (order_session.cpp), который берёт
// epsilon() для полки округления одинаково для double и для dd.
template <>
class numeric_limits<ucuda::dd> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed      = true;
    static constexpr bool is_integer     = false;
    static constexpr bool is_exact       = false;
    static constexpr bool is_iec559      = false;
    static constexpr bool has_infinity   = true;
    static constexpr bool has_quiet_NaN  = true;
    static constexpr int  radix          = 2;
    static constexpr int  digits         = 104;   // 2 x 53 минус перекрытие
    static constexpr int  digits10       = 31;
    static constexpr int  max_digits10   = 33;
    static constexpr int  min_exponent   = -968;
    static constexpr int  max_exponent   = 1024;

    // 2^-104
    static constexpr ucuda::dd epsilon() noexcept {
        return ucuda::dd(4.93038065763132e-32, 0.0);
    }
    // Наименьшее нормализованное, при котором ЕЩЁ работает вся мантисса пары.
    static constexpr ucuda::dd min() noexcept {
        return ucuda::dd(2.0041683600089728e-292, 0.0);
    }
    static constexpr ucuda::dd max() noexcept {
        return ucuda::dd(1.7976931348623157e308, 9.979201547673598e291);
    }
    static constexpr ucuda::dd lowest() noexcept {
        return ucuda::dd(-1.7976931348623157e308, -9.979201547673598e291);
    }
    static constexpr ucuda::dd infinity() noexcept {
        return ucuda::dd(numeric_limits<double>::infinity(), 0.0);
    }
    static constexpr ucuda::dd quiet_NaN() noexcept {
        return ucuda::dd(numeric_limits<double>::quiet_NaN(), 0.0);
    }
};

} // namespace std

// Возвращаем макросы вызывающему: он мог рассчитывать на них дальше по файлу.
#ifdef UCUDA_HP_POPPED_MIN
#pragma pop_macro("min")
#undef UCUDA_HP_POPPED_MIN
#endif
#ifdef UCUDA_HP_POPPED_MAX
#pragma pop_macro("max")
#undef UCUDA_HP_POPPED_MAX
#endif
