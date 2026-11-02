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


// ===========================================================================
// quad-double: четвёрка double, ~212 бит мантиссы (~62 десятичные цифры)
// ===========================================================================
//
// Та же идея, что у dd, но число разложено на ЧЕТЫРЕ неперекрывающиеся части:
// x = x[0] + x[1] + x[2] + x[3], каждая следующая ниже половины ulp
// предыдущей. eps = 2^-209 = 1.2e-63 против 4.9e-32 у dd.
//
// Сложение и умножение построены на grow-expansion Шевчука, а не на
// оптимизированных three_sum/renorm из QD. Причина прагматическая: у
// grow-expansion есть доказанное свойство — добавление одного double к
// неперекрывающемуся разложению снова даёт неперекрывающееся, — поэтому код
// проверяется по одной теореме, а не по цепочке условий на порядок величин.
// Неверная перенормировка не падает и не даёт мусора: она тихо теряет
// разряды, то есть выглядит как рабочий qd, который почему-то не лучше dd.
// Цена за ясность: сложение выходит вровень с QD (~96 флопов), умножение
// примерно в полтора раза дороже оптимального.
//
// Внутри используется two_sum, а не quick_two_sum: у второго предусловие
// |a| >= |b|, которое здесь пришлось бы доказывать на каждом шаге.
//
// Элементарные функции почти все строятся одним шагом Ньютона поверх dd:
// Ньютон удваивает число верных разрядов, а 32 цифры уже есть. Своей
// реализации требуют только exp (Ньютон для него замкнулся бы на log) и
// sin/cos.

struct qd {
    double x[4];

    constexpr qd() : x{ 0.0, 0.0, 0.0, 0.0 } {}
    constexpr qd(double a) : x{ a, 0.0, 0.0, 0.0 } {}
    constexpr qd(double a, double b, double c, double d) : x{ a, b, c, d } {}

    UCUDA_HP_HD explicit operator double() const { return x[0]; }
    UCUDA_HP_HD explicit operator int()    const { return (int)x[0]; }
};

UCUDA_HP_HD inline dd     to_dd(const qd& a)     { return dd(a.x[0], a.x[1]); }
UCUDA_HP_HD inline qd     to_qd(const dd& a)     { return qd(a.hi, a.lo, 0.0, 0.0); }
UCUDA_HP_HD inline double to_double(const qd& a) { return a.x[0]; }

// Наибольшее число слагаемых, которое сводит qd_from_terms (умножение даёт 16).
constexpr int kQdMaxTerms = 20;

// Точная сумма набора double, сведённая к плотной четвёрке.
//
// Два шага, и оба обязательны.
//
// 1. expansion-sum: слагаемые добавляются по одному, каждое проходит снизу
//    вверх по уже накопленному разложению. Буфер держит ВСЕ компоненты,
//    поэтому на этом шаге не происходит ни одного округления — сумма точна.
//
// 2. compress: expansion-sum даёт неперекрывающееся разложение, но не
//    плотное — компоненты могут нести по несколько бит каждая, и четырёх
//    старших тогда не хватает на 212 бит. Сжатие (два прохода Шевчука: вниз
//    от старшего, потом вверх) делает разложение НЕСМЕЖНЫМ, то есть каждый
//    следующий компонент ниже половины ulp предыдущего. Только после этого
//    отбрасывание пятого и далее — законное округление на 2^-212.
//
// Без шага 2 всё работает и даёт правильные значения, просто с точностью
// где-то между dd и qd — тот самый тихий отказ, который видно лишь проверкой
// инварианта на случайных данных.
UCUDA_HP_HD inline qd qd_from_terms(const double* t, int n) {
    double e[kQdMaxTerms + 1];
    int m = 0;

    // Шаг 1. e хранится по ВОЗРАСТАНИЮ величины — так grow не требует сдвигов.
    for (int k = 0; k < n; ++k) {
        double Q = t[k], lo;
        for (int i = 0; i < m; ++i) {
            Q = two_sum(e[i], Q, lo);
            e[i] = lo;
        }
        e[m++] = Q;
    }
    if (m == 0) return qd(0.0);

    // Шаг 2, проход вниз: старшие суммы откладываются в g, остаток идёт дальше.
    double g[kQdMaxTerms + 1];
    double Q = e[m - 1];
    int bottom = m - 1;
    for (int i = m - 2; i >= 0; --i) {
        double q;
        // two_sum, а не quick_two_sum: у второго предусловие |a| >= |b|,
        // которое здесь пришлось бы доказывать на каждом шаге.
        Q = two_sum(Q, e[i], q);
        if (q != 0.0) { g[bottom--] = Q; Q = q; }
    }
    g[bottom] = Q;

    // Проход вверх: собирает окончательное несмежное разложение.
    double h[kQdMaxTerms + 1];
    int top = 0;
    for (int i = bottom + 1; i < m; ++i) {
        double q;
        Q = two_sum(g[i], Q, q);
        if (q != 0.0) h[top++] = q;
    }
    h[top++] = Q;

    qd r;
    for (int i = 0; i < 4; ++i) r.x[i] = (top - 1 - i >= 0) ? h[top - 1 - i] : 0.0;
    return r;
}

// ---- Арифметика ----

UCUDA_HP_HD inline qd operator-(const qd& a) {
    return qd(-a.x[0], -a.x[1], -a.x[2], -a.x[3]);
}

UCUDA_HP_HD inline qd operator+(const qd& a, const qd& b) {
    const double t[8] = { a.x[0], a.x[1], a.x[2], a.x[3],
                          b.x[0], b.x[1], b.x[2], b.x[3] };
    return qd_from_terms(t, 8);
}

UCUDA_HP_HD inline qd operator+(const qd& a, double b) {
    const double t[5] = { a.x[0], a.x[1], a.x[2], a.x[3], b };
    return qd_from_terms(t, 5);
}

UCUDA_HP_HD inline qd operator+(double a, const qd& b) { return b + a; }
UCUDA_HP_HD inline qd operator-(const qd& a, const qd& b) { return a + (-b); }
UCUDA_HP_HD inline qd operator-(const qd& a, double b)    { return a + (-b); }
UCUDA_HP_HD inline qd operator-(double a, const qd& b)    { return (-b) + a; }

UCUDA_HP_HD inline qd operator*(const qd& a, double b) {
    double t[7], p, q;
    int n = 0;
    p = two_prod(a.x[0], b, q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[1], b, q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[2], b, q); t[n++] = p; t[n++] = q;
    t[n++] = a.x[3] * b;
    return qd_from_terms(t, n);
}

UCUDA_HP_HD inline qd operator*(double a, const qd& b) { return b * a; }

// Произведения с i+j <= 2 берутся точно (с остатком), с i+j == 3 — простым
// умножением: сам член имеет порядок 2^-159, и его ошибка округления 2^-212
// уже ниже разрешения четвёрки. Члены с i+j >= 4 отброшены по той же причине.
UCUDA_HP_HD inline qd operator*(const qd& a, const qd& b) {
    double t[16], p, q;
    int n = 0;
    p = two_prod(a.x[0], b.x[0], q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[0], b.x[1], q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[1], b.x[0], q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[0], b.x[2], q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[1], b.x[1], q); t[n++] = p; t[n++] = q;
    p = two_prod(a.x[2], b.x[0], q); t[n++] = p; t[n++] = q;
    t[n++] = a.x[0] * b.x[3];
    t[n++] = a.x[1] * b.x[2];
    t[n++] = a.x[2] * b.x[1];
    t[n++] = a.x[3] * b.x[0];
    return qd_from_terms(t, n);
}

UCUDA_HP_HD inline qd sqr(const qd& a) { return a * a; }

UCUDA_HP_HD inline qd mul_pwr2(const qd& a, double b) {
    return qd(a.x[0] * b, a.x[1] * b, a.x[2] * b, a.x[3] * b);
}

// Деление в столбик с «цифрой» в 53 бита: пять уточнений покрывают 212 бит.
UCUDA_HP_HD inline qd operator/(const qd& a, const qd& b) {
    double q[5];
    q[0] = a.x[0] / b.x[0];
    qd r = a - b * q[0];
    q[1] = r.x[0] / b.x[0];
    r = r - b * q[1];
    q[2] = r.x[0] / b.x[0];
    r = r - b * q[2];
    q[3] = r.x[0] / b.x[0];
    r = r - b * q[3];
    q[4] = r.x[0] / b.x[0];
    return qd_from_terms(q, 5);
}

UCUDA_HP_HD inline qd operator/(const qd& a, double b) { return a / qd(b); }
UCUDA_HP_HD inline qd operator/(double a, const qd& b) { return qd(a) / b; }

UCUDA_HP_HD inline qd& operator+=(qd& a, const qd& b) { a = a + b; return a; }
UCUDA_HP_HD inline qd& operator+=(qd& a, double b)    { a = a + b; return a; }
UCUDA_HP_HD inline qd& operator-=(qd& a, const qd& b) { a = a - b; return a; }
UCUDA_HP_HD inline qd& operator-=(qd& a, double b)    { a = a - b; return a; }
UCUDA_HP_HD inline qd& operator*=(qd& a, const qd& b) { a = a * b; return a; }
UCUDA_HP_HD inline qd& operator*=(qd& a, double b)    { a = a * b; return a; }
UCUDA_HP_HD inline qd& operator/=(qd& a, const qd& b) { a = a / b; return a; }
UCUDA_HP_HD inline qd& operator/=(qd& a, double b)    { a = a / b; return a; }

// ---- Сравнения ----

UCUDA_HP_HD inline bool operator==(const qd& a, const qd& b) {
    return a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] && a.x[3] == b.x[3];
}
UCUDA_HP_HD inline bool operator==(const qd& a, double b) {
    return a.x[0] == b && a.x[1] == 0.0 && a.x[2] == 0.0 && a.x[3] == 0.0;
}
UCUDA_HP_HD inline bool operator==(double a, const qd& b) { return b == a; }
UCUDA_HP_HD inline bool operator!=(const qd& a, const qd& b) { return !(a == b); }
UCUDA_HP_HD inline bool operator!=(const qd& a, double b)    { return !(a == b); }
UCUDA_HP_HD inline bool operator!=(double a, const qd& b)    { return !(b == a); }

UCUDA_HP_HD inline bool operator<(const qd& a, const qd& b) {
    if (a.x[0] != b.x[0]) return a.x[0] < b.x[0];
    if (a.x[1] != b.x[1]) return a.x[1] < b.x[1];
    if (a.x[2] != b.x[2]) return a.x[2] < b.x[2];
    return a.x[3] < b.x[3];
}
UCUDA_HP_HD inline bool operator<(const qd& a, double b) { return a < qd(b); }
UCUDA_HP_HD inline bool operator<(double a, const qd& b) { return qd(a) < b; }
UCUDA_HP_HD inline bool operator>(const qd& a, const qd& b) { return b < a; }
UCUDA_HP_HD inline bool operator>(const qd& a, double b)    { return qd(b) < a; }
UCUDA_HP_HD inline bool operator>(double a, const qd& b)    { return b < qd(a); }
UCUDA_HP_HD inline bool operator<=(const qd& a, const qd& b) { return !(b < a); }
UCUDA_HP_HD inline bool operator<=(const qd& a, double b)    { return !(qd(b) < a); }
UCUDA_HP_HD inline bool operator<=(double a, const qd& b)    { return !(b < qd(a)); }
UCUDA_HP_HD inline bool operator>=(const qd& a, const qd& b) { return !(a < b); }
UCUDA_HP_HD inline bool operator>=(const qd& a, double b)    { return !(a < qd(b)); }
UCUDA_HP_HD inline bool operator>=(double a, const qd& b)    { return !(qd(a) < b); }

// ---- Константы ----
// Четыре части вместо двух: double-литерал нёс бы только 17 цифр из 62.

constexpr qd c_qd_pi(3.141592653589793, 1.2246467991473532e-16, -2.9947698097183397e-33, 1.1124542208633653e-49);
constexpr qd c_qd_2pi(6.283185307179586, 2.4492935982947064e-16, -5.989539619436679e-33, 2.2249084417267306e-49);
constexpr qd c_qd_pi2(1.5707963267948966, 6.123233995736766e-17, -1.4973849048591698e-33, 5.562271104316826e-50);
constexpr qd c_qd_pi4(0.7853981633974483, 3.061616997868383e-17, -7.486924524295849e-34, 2.781135552158413e-50);
constexpr qd c_qd_e(2.718281828459045, 1.4456468917292502e-16, -2.1277171080381768e-33, 1.5156301598412191e-49);
constexpr qd c_qd_ln2(0.6931471805599453, 2.3190468138462996e-17, 5.707708438416212e-34, -3.5824322106018114e-50);
constexpr qd c_qd_ln10(2.302585092994046, -2.1707562233822494e-16, -9.984262454465777e-33, -4.023357454450206e-49);

constexpr qd c_qd_inv_fact[41] = {
    qd(1.0, 0.0, 0.0, 0.0),  // 1/0!
    qd(1.0, 0.0, 0.0, 0.0),  // 1/1!
    qd(0.5, 0.0, 0.0, 0.0),  // 1/2!
    qd(0.16666666666666666, 9.25185853854297e-18, 5.135813185032629e-34, 2.850949024098342e-50),  // 1/3!
    qd(0.041666666666666664, 2.3129646346357427e-18, 1.2839532962581572e-34, 7.127372560245855e-51),  // 1/4!
    qd(0.008333333333333333, 1.1564823173178714e-19, 1.6049416203226965e-36, 2.2273039250768297e-53),  // 1/5!
    qd(0.001388888888888889, -5.300543954373577e-20, -1.7386867553495878e-36, -1.6333562117230084e-52),  // 1/6!
    qd(0.0001984126984126984, 1.7209558293420705e-22, 1.4926912391394127e-40, 1.2947032674600247e-58),  // 1/7!
    qd(2.48015873015873e-05, 2.1511947866775882e-23, 1.865864048924266e-41, 1.6183790843250309e-59),  // 1/8!
    qd(2.7557319223985893e-06, -1.858393274046472e-22, 8.491754604881993e-39, -5.726616407894296e-55),  // 1/9!
    qd(2.755731922398589e-07, 2.3767714622250297e-23, -3.263188903340883e-40, 1.6143511186040442e-56),  // 1/10!
    qd(2.505210838544172e-08, -1.448814070935912e-24, 2.0426735146714455e-41, -8.496326720071632e-58),  // 1/11!
    qd(2.08767569878681e-09, -1.20734505911326e-25, 1.702227928892871e-42, 1.416095321503967e-58),  // 1/12!
    qd(1.6059043836821613e-10, 1.2585294588752098e-26, -5.31334602762985e-43, 3.5402147259760553e-59),  // 1/13!
    qd(1.1470745597729725e-11, 2.0655512752830745e-28, 6.889079232466646e-45, 5.729200026551091e-61),  // 1/14!
    qd(7.647163731819816e-13, 7.03872877733453e-30, -7.827539277162583e-48, 1.9213864944379024e-64),  // 1/15!
    qd(4.779477332387385e-14, 4.399205485834081e-31, -4.892212048226615e-49, 1.200866559023689e-65),  // 1/16!
    qd(2.8114572543455206e-15, 1.6508842730861433e-31, -2.877771793074479e-50, 4.2711068925629355e-67),  // 1/17!
    qd(1.5619206968586225e-16, 1.1910679660273754e-32, -4.577506059629983e-49, 2.874941423408996e-67),  // 1/18!
    qd(8.22063524662433e-18, 2.2141894119604265e-34, -1.508914023774199e-50, 1.4007295151478155e-67),  // 1/19!
    qd(4.110317623312165e-19, 1.4412973378659527e-36, -5.285627548789812e-53, -4.147647256357657e-70),  // 1/20!
    qd(1.9572941063391263e-20, -1.3643503830087908e-36, 1.3392348251125064e-53, -6.821089424149331e-70),  // 1/21!
    qd(8.896791392450574e-22, -7.911402614872376e-38, -3.1877976790570933e-54, 1.2705781017520566e-70),  // 1/22!
    qd(3.868170170630684e-23, -8.843177655482344e-40, 3.8718157106173247e-56, -1.9565257531522557e-72),  // 1/23!
    qd(1.6117375710961184e-24, -3.6846573564509766e-41, 1.613256546090552e-57, -8.1521906381344e-74),  // 1/24!
    qd(6.446950284384474e-26, -1.9330404233703465e-42, -1.5213023807039144e-58, 6.643772737212958e-75),  // 1/25!
    qd(2.4795962632247976e-27, -1.2953730964765229e-43, 6.403390159849962e-60, -8.460245627706746e-77),  // 1/26!
    qd(9.183689863795546e-29, 1.4303150396787322e-45, -8.551226774650505e-62, 8.381467100234538e-78),  // 1/27!
    qd(3.279889237069838e-30, 1.5117542744029879e-46, 8.058517719519716e-63, -9.096480530710929e-81),  // 1/28!
    qd(1.1309962886447716e-31, 1.0498015412959506e-47, -4.346150929397795e-64, -4.966779800140056e-81),  // 1/29!
    qd(3.7699876288159054e-33, 2.5870347832750324e-49, 3.23789002742564e-66, 2.5612859105788573e-82),  // 1/30!
    qd(1.216125041553518e-34, 5.586290567888806e-51, 6.615948578082792e-68, -3.162044228952086e-84),  // 1/31!
    qd(3.8003907548547434e-36, 1.7457158024652518e-52, 2.0674839306508725e-69, -9.881388215475268e-86),  // 1/32!
    qd(1.151633562077195e-37, -6.09957445788454e-54, -5.3447496196594105e-70, 2.625312623850008e-86),  // 1/33!
    qd(3.387157535521162e-39, 5.09056148151085e-56, 3.989567349036344e-72, -1.1495129447909262e-88),  // 1/34!
    qd(9.67759295863189e-41, 3.202295548645562e-57, 6.547507205018101e-74, -5.913342841536076e-91),  // 1/35!
    qd(2.6882202662866363e-42, 5.355061165943334e-59, -1.1290601987449868e-75, -7.097143528535273e-92),  // 1/36!
    qd(7.265460179153071e-44, -4.364097149354446e-61, 2.5503250121018375e-77, 3.6225969322843096e-94),  // 1/37!
    qd(1.911963205040282e-45, -2.7860822176883126e-62, 2.0347437224101328e-78, -9.139393622461627e-95),  // 1/38!
    qd(4.902469756513544e-47, -1.213019100517928e-63, -4.4707180011376586e-80, 5.37597340717859e-97),  // 1/39!
    qd(1.2256174391283858e-48, 6.033927348315605e-68, 4.0762496124582373e-84, -1.5982035330798997e-102),  // 1/40!
};

// ---- Классификация и округление ----

UCUDA_HP_HD inline bool isnan(const qd& a) {
    return std::isnan(a.x[0]) || std::isnan(a.x[1]) || std::isnan(a.x[2]) || std::isnan(a.x[3]);
}
UCUDA_HP_HD inline bool isinf(const qd& a) { return std::isinf(a.x[0]); }
UCUDA_HP_HD inline bool isfinite(const qd& a) {
    return std::isfinite(a.x[0]) && std::isfinite(a.x[1])
        && std::isfinite(a.x[2]) && std::isfinite(a.x[3]);
}
UCUDA_HP_HD inline bool signbit(const qd& a) { return std::signbit(a.x[0]); }

UCUDA_HP_HD inline qd fabs(const qd& a) { return a.x[0] < 0.0 ? -a : a; }
UCUDA_HP_HD inline qd abs (const qd& a) { return fabs(a); }

UCUDA_HP_HD inline qd ldexp(const qd& a, int n) {
    return qd(std::ldexp(a.x[0], n), std::ldexp(a.x[1], n),
              std::ldexp(a.x[2], n), std::ldexp(a.x[3], n));
}

// Округление идёт по компонентам: пока очередной уже целый, дробная часть
// сидит в следующем.
UCUDA_HP_HD inline qd floor(const qd& a) {
    double t[4] = { std::floor(a.x[0]), 0.0, 0.0, 0.0 };
    if (t[0] == a.x[0]) {
        t[1] = std::floor(a.x[1]);
        if (t[1] == a.x[1]) {
            t[2] = std::floor(a.x[2]);
            if (t[2] == a.x[2]) t[3] = std::floor(a.x[3]);
        }
    }
    return qd_from_terms(t, 4);
}

UCUDA_HP_HD inline qd ceil(const qd& a) {
    double t[4] = { std::ceil(a.x[0]), 0.0, 0.0, 0.0 };
    if (t[0] == a.x[0]) {
        t[1] = std::ceil(a.x[1]);
        if (t[1] == a.x[1]) {
            t[2] = std::ceil(a.x[2]);
            if (t[2] == a.x[2]) t[3] = std::ceil(a.x[3]);
        }
    }
    return qd_from_terms(t, 4);
}

UCUDA_HP_HD inline qd aint(const qd& a) { return a.x[0] >= 0.0 ? floor(a) : ceil(a); }

UCUDA_HP_HD inline qd copysign(const qd& a, const qd& b) {
    const qd m = fabs(a);
    return std::signbit(b.x[0]) ? -m : m;
}
UCUDA_HP_HD inline qd copysign(const qd& a, double b) { return copysign(a, qd(b)); }

UCUDA_HP_HD inline qd fmod(const qd& a, const qd& b) { return a - b * aint(a / b); }
UCUDA_HP_HD inline qd fmin(const qd& a, const qd& b) { return a < b ? a : b; }
UCUDA_HP_HD inline qd fmax(const qd& a, const qd& b) { return a > b ? a : b; }

// ---- Элементарные функции ----

UCUDA_HP_HD inline qd sqrt(const qd& a) {
    if (a.x[0] == 0.0) return qd(0.0);
    if (a.x[0] < 0.0)  return qd(std::numeric_limits<double>::quiet_NaN());
    // Затравка 1/sqrt(a) в dd — 32 цифры.
    qd x = to_qd(dd(1.0) / sqrt(to_dd(a)));
    // Ньютон по обратному корню (без деления): x <- x + x(1 - a x^2)/2.
    x = x + x * (qd(1.0) - a * x * x) * 0.5;
    // Перевод к самому корню и ещё один шаг — он же убирает ошибку перевода.
    qd y = a * x;
    y = y + (a - y * y) * x * 0.5;
    return y;
}

UCUDA_HP_HD inline qd hypot(const qd& x, const qd& y) {
    const qd ax = fabs(x), ay = fabs(y);
    if (ax.x[0] == 0.0) return ay;
    if (ay.x[0] == 0.0) return ax;
    if (ax < ay) return ay * sqrt(qd(1.0) + sqr(ax / ay));
    return ax * sqrt(qd(1.0) + sqr(ay / ax));
}

UCUDA_HP_HD inline qd exp(const qd& a) {
    if (a.x[0] <= -709.0) return qd(0.0);
    if (a.x[0] >=  709.0) return qd(std::numeric_limits<double>::infinity());
    if (a.x[0] == 0.0)    return qd(1.0);

    // a = m*ln2 + r, затем r уменьшается ещё в 256 раз: ряд тогда сходится за
    // два десятка членов, а масштаб возвращается восемью удвоениями.
    const double m = std::floor(a.x[0] / c_qd_ln2.x[0] + 0.5);
    qd r = (a - c_qd_ln2 * m) * (1.0 / 256.0);

    qd p = sqr(r);
    qd s = r + mul_pwr2(p, 0.5);
    p = p * r;
    qd t = p * c_qd_inv_fact[3];
    int i = 3;
    do {
        s = s + t;
        p = p * r;
        ++i;
        t = p * c_qd_inv_fact[i];
    } while (std::fabs(t.x[0]) > 1.0e-72 && i < 30);
    s = s + t;

    // exp(2x) - 1 = 2(exp(x) - 1) + (exp(x) - 1)^2
    for (int j = 0; j < 8; ++j) s = mul_pwr2(s, 2.0) + sqr(s);
    s = s + 1.0;
    return ldexp(s, (int)m);
}

UCUDA_HP_HD inline qd log(const qd& a) {
    if (a.x[0] == 0.0) return qd(-std::numeric_limits<double>::infinity());
    if (a.x[0] < 0.0)  return qd(std::numeric_limits<double>::quiet_NaN());
    if (a == 1.0)      return qd(0.0);
    // Ньютон по exp поверх dd-затравки: 32 цифры -> 64. Второй шаг снимает
    // остаток на уровне последних ulp.
    qd x = to_qd(log(to_dd(a)));
    x = x + a * exp(-x) - 1.0;
    x = x + a * exp(-x) - 1.0;
    return x;
}

UCUDA_HP_HD inline qd log10(const qd& a) { return log(a) / c_qd_ln10; }
UCUDA_HP_HD inline qd log2 (const qd& a) { return log(a) / c_qd_ln2;  }

// Ряды для |a| <= pi/256: сходятся за пару десятков членов.
UCUDA_HP_HD inline qd sin_taylor(const qd& a) {
    if (a.x[0] == 0.0) return qd(0.0);
    const qd x = -sqr(a);
    qd r = a, s = a, t;
    int i = 1;
    do {
        r = r * x;
        i += 2;
        t = r * c_qd_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 40 && std::fabs(t.x[0]) > 1.0e-75);
    return s;
}

UCUDA_HP_HD inline qd cos_taylor(const qd& a) {
    if (a.x[0] == 0.0) return qd(1.0);
    const qd x = -sqr(a);
    qd r(1.0), s(1.0), t;
    int i = 0;
    do {
        r = r * x;
        i += 2;
        t = r * c_qd_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 40 && std::fabs(t.x[0]) > 1.0e-75);
    return s;
}

// Приведение к [-pi/4, pi/4], затем ещё шесть делений пополам — и столько же
// удвоений обратно. Удвоение (s,c) -> (2sc, 1-2s^2) не теряет разрядов, а
// короткий аргумент избавляет от таблиц sin/cos в узлах, которые иначе
// понадобились бы для сходимости ряда.
UCUDA_HP_HD inline void sincos(const qd& a, qd& sn, qd& cs) {
    const double nd = std::floor(a.x[0] / c_qd_pi2.x[0] + 0.5);
    const qd r = (a - c_qd_pi2 * nd) * (1.0 / 64.0);

    qd s = sin_taylor(r), c = cos_taylor(r);
    for (int j = 0; j < 6; ++j) {
        const qd s2 = mul_pwr2(s * c, 2.0);
        const qd c2 = qd(1.0) - mul_pwr2(sqr(s), 2.0);
        s = s2; c = c2;
    }

    long long q = (long long)std::fmod(nd, 4.0);
    if (q < 0) q += 4;
    switch (q) {
        case 0:  sn =  s; cs =  c; break;
        case 1:  sn =  c; cs = -s; break;
        case 2:  sn = -s; cs = -c; break;
        default: sn = -c; cs =  s; break;
    }
}

UCUDA_HP_HD inline qd sin(const qd& a) { qd s, c; sincos(a, s, c); return s; }
UCUDA_HP_HD inline qd cos(const qd& a) { qd s, c; sincos(a, s, c); return c; }
UCUDA_HP_HD inline qd tan(const qd& a) { qd s, c; sincos(a, s, c); return s / c; }

UCUDA_HP_HD inline qd atan2(const qd& y, const qd& x) {
    if (x.x[0] == 0.0) {
        if (y.x[0] == 0.0) return qd(std::numeric_limits<double>::quiet_NaN());
        return y.x[0] > 0.0 ? c_qd_pi2 : -c_qd_pi2;
    }
    if (y.x[0] == 0.0) return x.x[0] > 0.0 ? qd(0.0) : c_qd_pi;

    const qd r  = sqrt(sqr(x) + sqr(y));
    const qd xx = x / r, yy = y / r;
    qd z = to_qd(atan2(to_dd(y), to_dd(x)));
    qd sz, cz;
    // Два шага Ньютона: затравка ограничена точностью самого dd (4.9e-32),
    // и после одного шага остаток лёг бы ровно на уровень eps(qd).
    for (int it = 0; it < 2; ++it) {
        sincos(z, sz, cz);
        if (std::fabs(xx.x[0]) > std::fabs(yy.x[0])) z = z + (yy - sz) / cz;
        else                                         z = z - (xx - cz) / sz;
    }
    return z;
}

UCUDA_HP_HD inline qd atan(const qd& a) { return atan2(a, qd(1.0)); }

UCUDA_HP_HD inline qd asin(const qd& a) {
    const qd m = fabs(a);
    if (m > 1.0)  return qd(std::numeric_limits<double>::quiet_NaN());
    if (m == 1.0) return a.x[0] > 0.0 ? c_qd_pi2 : -c_qd_pi2;
    return atan2(a, sqrt(qd(1.0) - sqr(a)));
}

UCUDA_HP_HD inline qd acos(const qd& a) {
    const qd m = fabs(a);
    if (m > 1.0)  return qd(std::numeric_limits<double>::quiet_NaN());
    if (m == 1.0) return a.x[0] > 0.0 ? qd(0.0) : c_qd_pi;
    return atan2(sqrt(qd(1.0) - sqr(a)), a);
}

UCUDA_HP_HD inline qd sinh(const qd& a) {
    if (a.x[0] == 0.0) return qd(0.0);
    if (std::fabs(a.x[0]) > 0.05) {
        const qd e = exp(a);
        return mul_pwr2(e - qd(1.0) / e, 0.5);
    }
    // При малом аргументе exp(a) - exp(-a) — катастрофическое вычитание.
    const qd x = sqr(a);
    qd r = a, s = a, t;
    int i = 1;
    do {
        r = r * x;
        i += 2;
        t = r * c_qd_inv_fact[i];
        s = s + t;
    } while (i + 2 <= 40 && std::fabs(t.x[0]) > std::fabs(a.x[0]) * 1.0e-66);
    return s;
}

UCUDA_HP_HD inline qd cosh(const qd& a) {
    if (a.x[0] == 0.0) return qd(1.0);
    const qd e = exp(a);
    return mul_pwr2(e + qd(1.0) / e, 0.5);
}

UCUDA_HP_HD inline qd tanh(const qd& a) {
    if (a.x[0] == 0.0) return qd(0.0);
    if (std::fabs(a.x[0]) > 0.05) {
        const qd e = exp(a), ie = qd(1.0) / e;
        return (e - ie) / (e + ie);
    }
    const qd s = sinh(a);
    return s / sqrt(qd(1.0) + sqr(s));
}

UCUDA_HP_HD inline qd npwr(const qd& a, int n) {
    if (n == 0) return qd(1.0);
    qd r = a, s(1.0);
    int k = n < 0 ? -n : n;
    while (k > 0) {
        if (k & 1) s = s * r;
        k >>= 1;
        if (k > 0) r = sqr(r);
    }
    return n < 0 ? qd(1.0) / s : s;
}

UCUDA_HP_HD inline qd pow(const qd& a, const qd& b) {
    const double bi = std::floor(b.x[0]);
    if (b.x[1] == 0.0 && bi == b.x[0] && std::fabs(bi) <= 1024.0)
        return npwr(a, (int)bi);
    return exp(b * log(a));
}
UCUDA_HP_HD inline qd pow(const qd& a, double b) { return pow(a, qd(b)); }
UCUDA_HP_HD inline qd pow(double a, const qd& b) { return pow(qd(a), b); }
UCUDA_HP_HD inline qd pow(const qd& a, int n)    { return npwr(a, n); }

UCUDA_HP_HD inline qd cbrt(const qd& a) {
    if (a.x[0] == 0.0) return qd(0.0);
    // Затравка в dd, затем один шаг Ньютона: y <- y + (a/y^2 - y)/3.
    qd y = to_qd(cbrt(to_dd(a)));
    y = y + (a / sqr(y) - y) / 3.0;
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

// Та же роль, что у специализации для dd: шаблонный код берёт epsilon()
// одинаково для всех трёх точностей.
template <>
class numeric_limits<ucuda::qd> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed      = true;
    static constexpr bool is_integer     = false;
    static constexpr bool is_exact       = false;
    static constexpr bool is_iec559      = false;
    static constexpr bool has_infinity   = true;
    static constexpr bool has_quiet_NaN  = true;
    static constexpr int  radix          = 2;
    static constexpr int  digits         = 209;   // 4 x 53 минус перекрытие
    static constexpr int  digits10       = 62;
    static constexpr int  max_digits10   = 64;
    static constexpr int  min_exponent   = -863;
    static constexpr int  max_exponent   = 1024;

    // 2^-209
    static constexpr ucuda::qd epsilon() noexcept {
        return ucuda::qd(1.2154326714572501e-63, 0.0, 0.0, 0.0);
    }
    // Ниже этого порога младшие компоненты уходят в денормали и четвёрка
    // перестаёт нести все 212 бит.
    static constexpr ucuda::qd min() noexcept {
        return ucuda::qd(1.6259745436952323e-260, 0.0, 0.0, 0.0);
    }
    static constexpr ucuda::qd max() noexcept {
        return ucuda::qd(1.7976931348623157e308, 9.979201547673598e291,
                         5.5395696628011126e275, 3.0750788930784049e259);
    }
    static constexpr ucuda::qd lowest() noexcept {
        return ucuda::qd(-1.7976931348623157e308, -9.979201547673598e291,
                         -5.5395696628011126e275, -3.0750788930784049e259);
    }
    static constexpr ucuda::qd infinity() noexcept {
        return ucuda::qd(numeric_limits<double>::infinity(), 0.0, 0.0, 0.0);
    }
    static constexpr ucuda::qd quiet_NaN() noexcept {
        return ucuda::qd(numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0);
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
