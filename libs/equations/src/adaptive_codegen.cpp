#include "expression_internal.h"
using namespace ucuda::equations_detail;
void adaptive_err_weights(const MultiDoubleCoef* b, int S, const MultiDoubleCoef* bh, int lc,
                          MultiDoubleCoef* w) {
    auto two_sum = [](double a, double c, double& err) {
        const double s = a + c, bb = s - a;
        err = (a - (s - bb)) + (c - bb);
        return s;
    };
    for (int j = 0; j < lc; ++j) {
        const MultiDoubleCoef z = {};
        const MultiDoubleCoef& x = j < S ? b[j] : z;
        double e0 = 0, e1 = 0;
        const double s = two_sum(x.c[0], -bh[j].c[0], e0);
        double lo = e0;
        for (int k = 1; k < 4; ++k) lo += x.c[k] - bh[j].c[k];
        const double hi = two_sum(s, lo, e1);
        w[j] = MultiDoubleCoef{ { hi, e1, 0.0, 0.0 } };
    }
}

bool scheme_supports_adaptive(Scheme sch) {
    return sch == Scheme::RK45 || sch == Scheme::DOP853
        || sch == Scheme::DOPRI78 || sch == Scheme::DOPRI78Legacy;
}

AdaptiveCode codegen_adaptive(const System& s, Scheme sch) {
    if (s.vars.size() != s.rhs.size())
        throw std::runtime_error("vars/rhs size mismatch");
    AdaptiveCode c;
    c.rhs = adaptive_rhs_body(s);
    EmbSpec e{};
    MultiDoubleCoef w[14];   // веса оценки ошибки b - b^ (RK45, DOPRI78); живут до adaptive_emb_body
    switch (sch) {
    case Scheme::RK45:
        adaptive_err_weights(&kRK45B[0][0], 6, &kRK45B[1][0], 7, w);
        e = { &kRK45A[0][0], 6, &kRK45B[0][0], 6, { w, nullptr }, 1, 7 };
        c.p = 5; c.q = 4; c.nlo = 1; c.wstages = 7; c.dsize = 5;
        c.emb_rhs = 6; c.dprep_rhs = 0; c.dense_order = 4;
        c.dprep = rk45_dprep_body(s); c.deval = rk45_deval_body(s);
        break;
    case Scheme::DOP853:
        e = { &kDop853A[0][0], 11, kDop853B, 12, { kDop853E5, kDop853E3 }, 2, 13 };   // E5, E3 — уже b - b^
        c.p = 8; c.q = 7; c.nlo = 2; c.wstages = 16; c.dsize = 8;
        c.emb_rhs = 12; c.dprep_rhs = 3; c.dense_order = 7;
        c.dprep = dop853_dprep_body(s); c.deval = dop853_deval_body(s);
        break;
    case Scheme::DOPRI78:
    case Scheme::DOPRI78Legacy: {
        const bool legacy = (sch == Scheme::DOPRI78Legacy);
        const MultiDoubleCoef (*A)[12] = legacy ? kDopri78LegacyA : kDopri78A;
        const MultiDoubleCoef (*B)[13] = legacy ? kDopri78LegacyB : kDopri78B;
        // f(Y) у DOPRI78 не стадия метода: оценщик z её не весит (lc = 13), а
        // считается она ради плотного выхода и как F0 следующего шага.
        adaptive_err_weights(&B[0][0], 13, &B[1][0], 13, w);
        e = { &A[0][0], 12, &B[0][0], 13, { w, nullptr }, 1, 13 };
        c.p = 8; c.q = 7; c.nlo = 1; c.wstages = 14; c.dsize = 4;
        c.emb_rhs = 13; c.dprep_rhs = 0; c.dense_order = 3;
        c.dprep = hermite_dprep_body(s); c.deval = hermite_deval_body(s);
        break;
    }
    default:
        throw std::runtime_error("scheme has no embedded error estimate (adaptive step needs RK45, DOPRI78 or DOP853)");
    }
    c.emb = adaptive_emb_body(s, e);
    return c;
}

std::string codegen_coupling(const System& s, const std::vector<std::string>& exprs,
                             const std::string& self_state, const std::string& nbr_state,
                             const std::string& self_par, const std::string& nbr_par,
                             const std::string& weight, const std::string& dst,
                             const std::string& indent) {
    if (exprs.size() != s.vars.size())
        throw std::runtime_error("coupling: expression count != number of variables");
    NameMap nm;
    // Вес кладём ПЕРВЫМ: имя из системы должно его перекрывать, а не наоборот.
    nm.m["K"] = weight;
    nm.m["w"] = weight;
    for (const auto& c : math_constants()) nm.m[c] = c;
    for (size_t i = 0; i < s.vars.size(); ++i) {
        const std::string idx = "[" + std::to_string(i) + "]";
        nm.m[s.vars[i]]        = self_state + idx;
        nm.m[s.vars[i] + "_i"] = self_state + idx;
        nm.m[s.vars[i] + "_j"] = nbr_state + idx;
    }
    for (size_t j = 0; j < s.params.size(); ++j) {
        const std::string idx = "[" + std::to_string(1 + (int)j) + "]";
        nm.m[s.params[j]]        = self_par + idx;
        nm.m[s.params[j] + "_i"] = self_par + idx;
        nm.m[s.params[j] + "_j"] = nbr_par + idx;
    }
    std::ostringstream o;
    for (size_t i = 0; i < exprs.size(); ++i) {
        if (exprs[i].empty()) continue;
        Parser p(exprs[i], false);
        PN ast = p.parse();
        o << indent << dst << "[" << i << "] += (";
        emit(ast, nm, o);
        o << ");\n";
    }
    return o.str();
}

Scheme scheme_from_name(const std::string& name) {
    if (name == "Euler-Cromer")      return Scheme::EulerCromer;
    if (name == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (name == "RK4")               return Scheme::RK4;
    if (name == "DOPRI78")           return Scheme::DOPRI78;
    if (name == "DOPRI78 (legacy)")  return Scheme::DOPRI78Legacy;
    if (name == "RK45")              return Scheme::RK45;
    if (name == "DOP853")            return Scheme::DOP853;
    if (name == "CD")                return Scheme::CD;
    if (name == "Complex CD")        return Scheme::ComplexCD;
    if (name == "Complex CD4")       return Scheme::ComplexCD4;
    if (name == "CCD4 (o4s3)")       return Scheme::ComplexCD4S3;
    if (name == "CCD4 (o4s4)")       return Scheme::ComplexCD4S4;
    if (name == "CCD4 (s-split)_{0-1}") return Scheme::ComplexCD4SS01;
    if (name == "CCD4 (s-split)_{1-0}") return Scheme::ComplexCD4SS10;
    if (name == "CD_{1-0}")          return Scheme::CD10;
    if (name == "Complex CD_{1-0}")  return Scheme::ComplexCD10;
    if (name == "Complex CD4_{1-0}") return Scheme::ComplexCD4_10;
    if (name == "CCD4 (o4s3)_{1-0}") return Scheme::ComplexCD4S3_10;
    if (name == "CCD4 (o4s4)_{1-0}") return Scheme::ComplexCD4S4_10;
    if (name == "Implicit Euler")    return Scheme::ImplicitEuler;
    if (name == "Implicit Midpoint") return Scheme::ImplicitMidpoint;
    if (name == "SEMP")              return Scheme::SEMP;
    if (name == "SIMP")              return Scheme::SIMP;
    if (name == "D")                 return Scheme::D;
    if (name == "Complex Implicit Euler") return Scheme::ComplexIEuler;
    if (name == "GBS (n=2)")         return Scheme::GBS;
    if (name == "GBS 2-4")           return Scheme::GBS24;
    if (name == "GBS 2-4-6")         return Scheme::GBS246;
    if (name == "GBS 2-4-6-8")       return Scheme::GBS2468;
    if (name == "GBS 2-4-6-8-10")    return Scheme::GBS246810;
    if (name == "GBS 2-4-6-8-10-12") return Scheme::GBS24681012;
    if (name == "Map")               return Scheme::Map;
    return Scheme::Euler;
}

// --- Паспорт встроенных схем -----------------------------------------------
// Порядки — те, что замерены и записаны в подсказках чекбоксов (см. gui.cpp).
// Напоминание про a[0] = 1/2: у CD, Complex CD, Complex CD4, CCD4 (o4s3),
// CCD4 (o4s4) (и их вариантов _{1-0}), SEMP и SIMP
// заявленный порядок достигается только при этом значении, иначе все они
// падают до первого. Здесь стоит паспортный (то есть при a[0] = 1/2) —
// предупредить пользователя обязан UI.
namespace {
// Порядок строк = порядок в таблице схем UI (kBuiltinSchemes в gui.cpp): по
// нему sort_wrapper_schemes ставит опорники обёрток, и список обёрток
// читается теми же блоками, что и список встроенных.
struct SchemeRow { const char* name; int order; bool symmetric; };
const SchemeRow kSchemeRows[] = {
    { "Euler",                  1, false },
    { "Euler-Cromer",           1, false },
    { "D",                      1, false },
    { "Implicit Euler",         1, false },
    { "Explicit Midpoint",      2, false },
    { "Implicit Midpoint",      2, true  },  // Phi* ∘ Phi, самосопряжённая
    { "CD",                     2, true  },  // самосопряжённая при a[0] = 1/2
    // {1-0}: неявная половина первой; порядок и симметрия как у исходных.
    { "CD_{1-0}",               2, true  },
    { "Complex CD",             2, false },
    { "Complex CD_{1-0}",       2, false },
    { "Complex Implicit Euler", 2, false },
    { "SEMP",                   2, false },  // предиктор-корректор, не Phi* ∘ Phi
    { "SIMP",                   2, false },
    { "RK4",                    4, false },
    { "Complex CD4",            4, false },
    { "Complex CD4_{1-0}",      4, false },
    { "CCD4 (o4s3)",            4, true  },  // палиндром (a, 1-2a, a)
    { "CCD4 (o4s3)_{1-0}",      4, true  },
    { "CCD4 (o4s4)",            4, true  },  // палиндром (g/2, gc/2, gc/2, g/2)
    { "CCD4 (o4s4)_{1-0}",      4, true  },
    { "CCD4 (s-split)_{0-1}",   4, false },  // самосопряжена до Re, после Re дефект O(h^6)
    { "CCD4 (s-split)_{1-0}",   4, false },  // то же над сопряжённым базовым методом
    { "RK45",                   5, false },  // Dormand-Prince 5(4), шаг решением 5-го порядка
    { "DOPRI78",                8, false },
    { "DOPRI78 (legacy)",       8, false },  // дроби статьи без уточнения
    { "DOP853",                 8, false },
    // ГБШ: опорная GBS (n=2) не самосопряжена как одношаговый метод
    // (дефект O(h^6)), поэтому Extr над ней на трёх стадиях даёт 5, а не 6.
    // Экстраполяторы — не база для дальнейших обёрток.
    { "GBS (n=2)",              2, false },
    { "GBS 2-4",                4, false },
    { "GBS 2-4-6",              6, false },
    { "GBS 2-4-6-8",            8, false },
    { "GBS 2-4-6-8-10",        10, false },
    { "GBS 2-4-6-8-10-12",     12, false },
};

// Номер строки паспорта, -1 — не встроенная схема.
int builtin_scheme_rank(const std::string& name) {
    const int n = (int)(sizeof(kSchemeRows) / sizeof(kSchemeRows[0]));
    for (int i = 0; i < n; ++i)
        if (name == kSchemeRows[i].name) return i;
    return -1;
}
} // namespace

bool builtin_scheme_traits(const std::string& name, int* order, bool* symmetric) {
    const int i = builtin_scheme_rank(name);
    if (i < 0) return false;
    if (order)     *order     = kSchemeRows[i].order;
    if (symmetric) *symmetric = kSchemeRows[i].symmetric;
    return true;
}

// --- Имя обёртки: необязательная метка --------------------------------------

namespace {

    std::string wrap_trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    // Голова имени — то, что стоит до ПОСЛЕДНЕЙ '|', то есть до коэффициентов.
    // Метка, если она есть, идёт первой: "s17o8|Short CD". Режем по ПЕРВОЙ
    // черте, тогда как база берётся по последней, — старое имя без метки
    // ("Short CD") головы с чертой не имеет и разбирается ровно как раньше.
    bool wrap_split_head(const std::string& head, std::string* label,
                         std::string* base, std::string* err) {
        const size_t bar = head.find('|');
        if (bar == std::string::npos) { label->clear(); *base = wrap_trim(head); return true; }
        *label = wrap_trim(head.substr(0, bar));
        *base  = wrap_trim(head.substr(bar + 1));
        if (label->empty()) { if (err) *err = "empty scheme label"; return false; }
        if (!wrapper_label_ok(*label)) {
            if (err) *err = "scheme label must not contain '|', ',', '(' or ')'";
            return false;
        }
        return true;
    }

} // namespace

bool wrapper_label_ok(const std::string& label) {
    if (label.empty()) return false;
    if (wrap_trim(label) != label) return false;
    for (char c : label)
        if (c == '|' || c == ',' || c == '(' || c == ')') return false;
    return true;
}

std::string wrapper_sanitize_label(const std::string& label) {
    std::string out;
    for (char c : label)
        if (c != '|' && c != ',' && c != '(' && c != ')') out += c;
    return wrap_trim(out);
}

// --- Экстраполяция Ричардсона ----------------------------------------------

static const char* const kExtrPrefix  = "Extr(";
static const char* const kExtrZPrefix = "ExtrZ(";   // Re только на выходе

static bool starts_with(const std::string& s, const char* pref) {
    const size_t n = std::strlen(pref);
    return s.size() >= n && s.compare(0, n, pref) == 0;
}

std::string make_extrapolation_name(const std::string& base, const std::vector<int>& n,
                                    const std::string& label, bool re_at_output) {
    std::string s = re_at_output ? kExtrZPrefix : kExtrPrefix;
    if (!label.empty()) s += label + "|";
    s += base + "|";
    for (size_t k = 0; k < n.size(); ++k) {
        if (k) s += ",";
        s += std::to_string(n[k]);
    }
    return s + ")";
}

// "12" -> 12. Отказывает на пустой строке, нецифрах и выходе за диапазон,
// чтобы кривое имя из старого JSON не превратилось молча в рабочую схему.
static bool extr_parse_substeps(const std::string& s, int* out) {
    if (s.empty() || s.size() > 9) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    const long v = std::strtol(s.c_str(), nullptr, 10);
    if (v < 1 || v > kExtrMaxSubsteps) return false;
    *out = (int)v;
    return true;
}

bool parse_extrapolation_name(const std::string& name, ExtrapolationSpec* out,
                              std::string* err) {
    auto fail = [&](const char* msg) { if (err) *err = msg; return false; };
    auto fail_s = [&](const std::string& msg) { if (err) *err = msg; return false; };

    const bool re_out = starts_with(name, kExtrZPrefix);
    const std::string pref = re_out ? kExtrZPrefix : kExtrPrefix;
    if (name.size() <= pref.size() || name.compare(0, pref.size(), pref) != 0)
        return fail("not an extrapolated scheme name");
    if (name.back() != ')')
        return fail("missing closing ')'");

    const std::string inner = name.substr(pref.size(), name.size() - pref.size() - 1);
    // rfind, а не find: имя базы по правилам валидации '|' не содержит, но если
    // в старом JSON такое всё же лежит — резать надо по последней черте, иначе
    // список n разберётся как часть имени и схема тихо станет другой.
    const size_t bar = inner.rfind('|');
    if (bar == std::string::npos) return fail("missing '|' between base and substep list");

    ExtrapolationSpec spec;
    spec.re_at_output = re_out;
    // Голова — "<база>" либо "<метка>|<база>": метку добавили позже, и имена
    // без неё обязаны разбираться как прежде.
    if (!wrap_split_head(inner.substr(0, bar), &spec.label, &spec.base, err)) return false;
    if (spec.base.empty()) return fail("empty base scheme name");
    // Вложенность запрещена: порядок обёртки над обёрткой считается не как
    // p + K - 1, и заодно это отрезает бесконечную рекурсию в резолвере.
    if (starts_with(spec.base, kExtrPrefix) || starts_with(spec.base, kExtrZPrefix))
        return fail("base scheme cannot itself be extrapolated");

    const std::string tail = inner.substr(bar + 1);
    size_t pos = 0;
    while (true) {
        const size_t comma = tail.find(',', pos);
        const std::string tok = tail.substr(pos, comma == std::string::npos
                                                ? std::string::npos : comma - pos);
        int v = 0;
        if (!extr_parse_substeps(tok, &v))
            return fail_s("substep counts must be integers in 1.."
                          + std::to_string(kExtrMaxSubsteps));
        spec.n.push_back(v);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    const int K = (int)spec.n.size();
    // Сообщения печатаются из констант, а не из букв: потолок стадий уже
    // поднимали, и расхождение текста с проверкой врёт пользователю.
    if (K < kExtrMinStages)
        return fail_s("at least " + std::to_string(kExtrMinStages) + " stages are needed");
    if (K > kExtrMaxStages)
        return fail_s("at most " + std::to_string(kExtrMaxStages) + " stages are supported");
    for (int k = 1; k < K; ++k)
        if (spec.n[k] <= spec.n[k - 1]) return fail("substep counts must strictly increase");

    if (out) *out = spec;
    return true;
}

int extrapolation_order(int stages, int p, bool symmetric) {
    if (stages < 1) return p;
    return symmetric ? p + 2 * (stages - 1) : p + stages - 1;
}

// Точная рациональная арифметика для весов экстраполяции.
//
// Веса — отношения небольших целых, но напечатанные десятичным литералом они
// несут ошибку ~1e-16 ОТ СЕБЯ: литерал читается как double, в какой бы
// точности ни считала схема. Экстраполяция складывает почти равные величины с
// коэффициентами порядка единиц (у Extr(CD|1,2,3,4) это 16/45, -729/280,
// 1024/315), так что ошибка коэффициентов не сокращается, а садится прямо в
// результат макрошага. На dd-ветке вкладки Order это держало полку на уровне
// double и сводило расширенную точность к нулю: p валился с 8 до -1 там, где
// с точными дробями он остаётся 8.01 при E1 = 2e-18.
//
// Поэтому печатаем дробь, а не число: деление выполняется уже в numb.
namespace {

struct ExRat { long long n, d; };

long long ex_gcd(long long a, long long b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { const long long t = a % b; a = b; b = t; }
    return a ? a : 1;
}

// false — переполнение; вызывающий откатывается на печать литералом.
bool ex_mul(long long a, long long b, long long& r) {
    if (a == 0 || b == 0) { r = 0; return true; }
    const long long q = a * b;
    if (q / b != a) return false;
    r = q;
    return true;
}

bool ex_norm(ExRat& x) {
    if (x.d == 0) return false;
    if (x.d < 0) { x.d = -x.d; x.n = -x.n; }
    const long long g = ex_gcd(x.n, x.d);
    x.n /= g;
    x.d /= g;
    return true;
}

bool ex_mul_rat(ExRat a, ExRat b, ExRat& out) {
    // Сокращаем крест-накрест ДО умножения: так переполнение наступает много
    // позже, а знаменатели тут растут как произведение разностей n^q.
    const long long g1 = ex_gcd(a.n, b.d), g2 = ex_gcd(b.n, a.d);
    a.n /= g1; b.d /= g1;
    b.n /= g2; a.d /= g2;
    if (!ex_mul(a.n, b.n, out.n)) return false;
    if (!ex_mul(a.d, b.d, out.d)) return false;
    return ex_norm(out);
}

bool ex_add_rat(ExRat a, ExRat b, ExRat& out) {
    const long long g = ex_gcd(a.d, b.d);
    long long den, t1, t2;
    if (!ex_mul(a.d, b.d / g, den)) return false;
    if (!ex_mul(a.n, b.d / g, t1)) return false;
    if (!ex_mul(b.n, a.d / g, t2)) return false;
    if ((t2 > 0 && t1 > LLONG_MAX - t2) || (t2 < 0 && t1 < LLONG_MIN - t2)) return false;
    out.n = t1 + t2;
    out.d = den;
    return ex_norm(out);
}

bool ex_pow(long long base, int e, long long& out) {
    out = 1;
    for (int i = 0; i < e; ++i)
        if (!ex_mul(out, base, out)) return false;
    return true;
}

// Те же alpha, что в extrapolation_weights, но без единого округления.
// Подстановка u_k = 1/n_k^q в alpha_k ~ n_k^p * prod 1/(u_k - u_m) даёт
// gamma_k = n_k^(p + q(K-2)) / prod_{m!=k} (n_m^q - n_k^q); общий множитель
// сокращается нормировкой суммы в единицу.
bool extrapolation_weights_rational(const std::vector<int>& n, int p, bool symmetric,
                                    std::vector<ExRat>& out) {
    const int K = (int)n.size();
    if (K < 2) return false;
    const int q = symmetric ? 2 : 1;
    const int e = p + q * (K - 2);
    if (e < 0) return false;

    std::vector<ExRat> g((size_t)K);
    ExRat sum{ 0, 1 };
    for (int k = 0; k < K; ++k) {
        long long num = 0, den = 1, pk = 1;
        if (!ex_pow((long long)n[k], e, num)) return false;
        if (!ex_pow((long long)n[k], q, pk))  return false;
        for (int m = 0; m < K; ++m) {
            if (m == k) continue;
            long long pm = 1;
            if (!ex_pow((long long)n[m], q, pm)) return false;
            if (!ex_mul(den, pm - pk, den))      return false;
        }
        g[(size_t)k] = ExRat{ num, den };
        if (!ex_norm(g[(size_t)k]))               return false;
        if (!ex_add_rat(sum, g[(size_t)k], sum))  return false;
    }
    if (sum.n == 0) return false;

    out.assign((size_t)K, ExRat{ 0, 1 });
    for (int k = 0; k < K; ++k) {
        const ExRat inv{ sum.d, sum.n };
        if (!ex_mul_rat(g[(size_t)k], inv, out[(size_t)k])) return false;
        // Числитель и знаменатель обязаны быть точны в double, иначе печать
        // дробью ничего не меняет: литерал снова округлится.
        constexpr long long kExact = 9007199254740992LL;   // 2^53
        if (out[(size_t)k].n >  kExact || out[(size_t)k].n < -kExact ||
            out[(size_t)k].d >  kExact) return false;
    }
    return true;
}

std::string ex_rat_expr(const ExRat& r) {
    std::string s = "(numb)(" + std::to_string(r.n) + ".0)";
    if (r.d != 1) s = "(" + s + "/(numb)(" + std::to_string(r.d) + ".0))";
    return s;
}

} // namespace

std::vector<double> extrapolation_weights(const std::vector<int>& n, int p, bool symmetric) {
    const int K = (int)n.size();
    std::vector<double> u((size_t)K), alpha((size_t)K);
    for (int k = 0; k < K; ++k) {
        const double inv = 1.0 / (double)n[k];
        u[k] = symmetric ? inv * inv : inv;
    }
    // alpha_k ~ n_k^p * prod_{m != k} 1/(u_k - u_m). Вектор обратных произведений
    // разностей — это коэффициенты (K-1)-й разделённой разности, то есть ровно
    // тот единственный (с точностью до масштаба) вектор, который ортогонален
    // 1, u, ..., u^(K-2). Нормировка суммы в 1 закрывает условие состоятельности.
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        double prod = 1.0;
        for (int m = 0; m < K; ++m)
            if (m != k) prod *= (u[k] - u[m]);
        alpha[k] = std::pow((double)n[k], (double)p) / prod;
        sum += alpha[k];
    }
    for (int k = 0; k < K; ++k) alpha[k] /= sum;
    return alpha;
}

bool extrapolation_weights_exact(const std::vector<int>& n, int p, bool symmetric,
                                 std::vector<std::string>* out) {
    std::vector<ExRat> r;
    if (!extrapolation_weights_rational(n, p, symmetric, r)) return false;
    if (!out) return true;
    out->clear();
    out->reserve(r.size());
    for (const ExRat& v : r)
        out->push_back(v.d == 1 ? std::to_string(v.n)
                                : std::to_string(v.n) + "/" + std::to_string(v.d));
    return true;
}

// ГБШ (см. Scheme::GBS в codegen.hpp). Не обёртка над готовым телом, как
// wrap_extrapolation, а свой эмиттер: стадия держит ДВА состояния (z_{m-1},
// z_m) через все подшаги, и повтором одношаговой базы её не выразить.
// Правые части печатаются по одному разу на каждое из трёх мест (старт,
// leapfrog, сглаживание), а номер стадии идёт рантайм-циклом, поэтому тело не
// растёт с K. Веса — те же, что у Extr с симметричной базой порядка 2, точными
// дробями, когда сходятся.
namespace ucuda::equations_detail {
std::string scheme_gbs(const System& s, int K) {
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size();
    if (N < 1) throw std::runtime_error("GBS needs a non-empty system");
    const std::vector<int> n = gbs_substeps(K);
    const auto f0 = rhs_over(s, "X0_gbs");
    const auto f1 = rhs_over(s, "Z1_gbs");

    std::vector<double> alpha(1, 1.0);
    std::vector<ExRat>  arat(1, ExRat{ 1, 1 });
    bool alpha_exact = true;
    if (K > 1) {
        alpha       = extrapolation_weights(n, 2, true);
        alpha_exact = extrapolation_weights_rational(n, 2, true, arat);
    }
    int cost = 0;
    for (int v : n) cost += v + 1;   // n подшагов + сглаживание

    const std::string Ns = std::to_string(N), Ks = std::to_string(K);
    std::ostringstream o;
    o << "    // --- GBS";
    if (K == 1) o << " (n=2)";
    else for (int k = 0; k < K; ++k) o << (k ? "-" : " ") << n[k];
    o << ": Gragg modified midpoint, " << K << " stage" << (K == 1 ? "" : "s")
      << ", order " << 2 * K << ", " << cost << " RHS evaluations per step ---\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    o << "    numb X0_gbs[" << Ns << "], AC_gbs[" << Ns << "], Z0_gbs[" << Ns
      << "], Z1_gbs[" << Ns << "], ZT_gbs[" << Ns << "];\n";
    o << "    const numb W_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) {
        if (k) o << ", ";
        o << (alpha_exact ? ex_rat_expr(arat[(size_t)k]) : ("(numb)(" + fmtnum(alpha[k]) + ")"));
    }
    o << " };\n";
    o << "    int i_gbs, m_gbs, k_gbs;\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { X0_gbs[i_gbs] = X[i_gbs]; AC_gbs[i_gbs] = (numb)0; }\n";
    o << "    for (k_gbs = 0; k_gbs < " << Ks << "; ++k_gbs) {\n";
    o << "        const int  n_gbs  = 2 * (k_gbs + 1);\n";
    o << "        const numb hs_gbs = h / (numb)n_gbs;\n";
    o << "        // старт: явный Эйлер на первый подшаг\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) Z0_gbs[i_gbs] = X0_gbs[i_gbs];\n";
    for (int i = 0; i < N; ++i)
        o << "        Z1_gbs[" << i << "] = X0_gbs[" << i << "] + hs_gbs * (" << f0[i] << ");\n";
    o << "        // leapfrog: z_{m+1} = z_{m-1} + 2*hs*f(z_m), всего n подшагов\n";
    o << "        for (m_gbs = 1; m_gbs < n_gbs; ++m_gbs) {\n";
    for (int i = 0; i < N; ++i)
        o << "            ZT_gbs[" << i << "] = Z0_gbs[" << i << "] + (numb)2 * hs_gbs * (" << f1[i] << ");\n";
    o << "            for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { Z0_gbs[i_gbs] = Z1_gbs[i_gbs]; Z1_gbs[i_gbs] = ZT_gbs[i_gbs]; }\n";
    o << "        }\n";
    o << "        // сглаживание Грэгга: (z_{n-1} + z_n + hs*f(z_n)) / 2\n";
    for (int i = 0; i < N; ++i)
        o << "        AC_gbs[" << i << "] += W_gbs[k_gbs] * (numb)0.5 * (Z0_gbs[" << i << "] + Z1_gbs["
          << i << "] + hs_gbs * (" << f1[i] << "));\n";
    o << "    }\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) X[i_gbs] = AC_gbs[i_gbs];\n";
    return o.str();
}

// --- Подстановка тела базы в обёртки ------------------------------------------
//
// Обёртки (Extr, ExtrZ, Comp) печатают тело базы прямо в каждую стадию, блоком
// с собственным "const numb h = <подшаг>": внутренний h затеняет макрошаг, и
// тело по-прежнему не правится ни в одном символе. Раньше тело уходило один раз
// в локальную лямбду step_ex(h), и это решал компилятор: NVRTC её встраивал
// всегда, а MSVC на CPU-пути — не всегда. У ExtrZ(Complex CD4|1,2,3) он оставил
// шесть настоящих вызовов на шаг, и шаг выходил на треть медленнее (354 против
// 268 нс, побитово тот же результат). Текст растёт в K раз, но ровно столько
// копий NVRTC делал и сам.
//
// Лямбда остаётся запасным путём для тела с return или goto: return внутри
// подставленного тела вышел бы из всего calculateDiscreteModel, а метка,
// размноженная по стадиям, не скомпилировалась бы (метки видны во всей функции).
bool wrapper_can_inline(const std::string& body) {
    for (const char* kw : { "return", "goto" }) {
        const size_t n = std::strlen(kw);
        for (size_t p = body.find(kw); p != std::string::npos; p = body.find(kw, p + 1)) {
            const bool left  = p == 0 || !(std::isalnum((unsigned char)body[p - 1]) || body[p - 1] == '_');
            const bool right = p + n >= body.size()
                               || !(std::isalnum((unsigned char)body[p + n]) || body[p + n] == '_');
            if (left && right) return false;
        }
    }
    return true;
}

// Тело построчно с дополнительным отступом: панель KRS показывает текст как есть.
void emit_indented(std::ostringstream& o, const std::string& body, const char* indent) {
    size_t pos = 0;
    while (pos < body.size()) {
        size_t eol = body.find('\n', pos);
        if (eol == std::string::npos) eol = body.size();
        const std::string line = body.substr(pos, eol - pos);
        if (!line.empty()) o << indent << line;
        o << "\n";
        pos = eol + 1;
    }
}
} // namespace

std::string wrap_extrapolation(const std::string& base_body, int N,
                               const std::vector<int>& n, int p, bool symmetric,
                               const std::string& base_name) {
    const int K = (int)n.size();
    if (K < 1) throw std::runtime_error("extrapolation needs at least one stage");
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");

    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    // Точные дроби — если сошлись (см. extrapolation_weights_rational).
    std::vector<ExRat> arat;
    const bool alpha_exact = extrapolation_weights_rational(n, p, symmetric, arat);
    long long cost = 0;
    for (int k = 0; k < K; ++k) cost += n[k];

    const std::string Ns = std::to_string(N);
    std::ostringstream o;

    o << "    // --- " << make_extrapolation_name(base_name, n) << " ---\n";
    o << "    // base: order " << p << ", " << (symmetric ? "symmetric" : "non-symmetric")
      << "  ->  extrapolated order " << extrapolation_order(K, p, symmetric) << "\n";
    o << "    // " << K << " stages, " << cost << " base steps per macro-step\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n[" << k << "] = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    o << "    numb X0_ex[" << Ns << "], AC_ex[" << Ns << "];\n";
    o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { X0_ex[i_ex] = X[i_ex]; AC_ex[i_ex] = (numb)0; }\n\n";

    // Тело базы — в каждую стадию, блоком со своим h (см. wrapper_can_inline).
    // Внутренний h затеняет макрошаг, поэтому любое "h" в базе становится
    // подшагом без единой правки текста — это же позволяет обернуть кастомную
    // КРС, содержимое которой нам непрозрачно.
    const bool inl = wrapper_can_inline(base_body);
    if (inl) {
        o << "    const numb h_ex = h;\n\n";
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_ex = [&](const numb h) {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
    }

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << "\n";
        if (inl) {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) {\n";
            o << "        const numb h = h_ex / (numb)" << fmtnum((double)n[k]) << ";\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        } else {
            o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) step_ex(h / (numb)"
              << fmtnum((double)n[k]) << ");\n";
        }
        o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { AC_ex[i_ex] += "
          << (alpha_exact ? ex_rat_expr(arat[(size_t)k])
                          : ("(numb)(" + fmtnum(alpha[k]) + ")"))
          << " * X[i_ex];";
        // После последней стадии откатывать нечего — X сразу перезаписывается.
        if (k + 1 < K) o << " X[i_ex] = X0_ex[i_ex];";
        o << " }\n";
    }

    o << "\n    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) X[i_ex] = AC_ex[i_ex];\n";
    return o.str();
}

// ExtrZ. Отличие от wrap_extrapolation ровно одно: состояние стадий,
// начальная точка и накопитель — ucmplx, ядро базы работает прямо на Z[] без
// Re, а Re берётся один раз после суммы. Тогда стадия из n подшагов —
// это exp(n * log Psi_{h/n}) целиком, её разложение по 1/n идёт с
// вещественными членами только при чётных степенях, и симметричные веса
// гасят их через одну (см. ExtrapolationSpec в codegen.hpp).
std::string wrap_extrapolation_complex(const std::string& core_body, int N,
                                       const std::vector<int>& n, int p,
                                       const std::string& base_name) {
    const int K = (int)n.size();
    if (K < 1) throw std::runtime_error("extrapolation needs at least one stage");
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");

    const bool symmetric = extrapolation_symmetric(true, false);
    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    std::vector<ExRat> arat;
    const bool alpha_exact = extrapolation_weights_rational(n, p, symmetric, arat);
    long long cost = 0;
    for (int k = 0; k < K; ++k) cost += n[k];

    const std::string Ns = std::to_string(N);
    std::ostringstream o;

    o << "    // --- " << make_extrapolation_name(base_name, n, {}, true) << " ---\n";
    o << "    // base: order " << p << ", complex state across substeps, Re only on output\n";
    o << "    //   ->  extrapolated order " << extrapolation_order(K, p, symmetric) << "\n";
    o << "    // " << K << " stages, " << cost << " base steps per macro-step\n";
    for (int k = 0; k < K; ++k) {
        o << "    //   n[" << k << "] = " << n[k] << "   alpha = " << fmtnum(alpha[k]);
        if (alpha_exact) o << " = " << arat[(size_t)k].n << "/" << arat[(size_t)k].d;
        o << "\n";
    }

    // Z — то самое имя, над которым работает ядро базы.
    o << "    ucmplx Z[" << Ns << "], Z0_ex[" << Ns << "], AC_ex[" << Ns << "];\n";
    o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) {\n"
      << "        Z0_ex[i_ex] = ucmplx(X[i_ex], 0.0); Z[i_ex] = Z0_ex[i_ex];"
      << " AC_ex[i_ex] = ucmplx(0.0, 0.0);\n    }\n\n";

    // Ядро — встроенная CD-схема, return/goto в нём нет: подставляется всегда.
    o << "    const numb h_ex = h;\n\n";

    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s")
          << " of h/" << n[k] << ", no Re in between\n";
        o << "    for (int s_ex = 0; s_ex < " << n[k] << "; ++s_ex) {\n";
        o << "        const numb h = h_ex / (numb)" << fmtnum((double)n[k]) << ";\n";
        emit_indented(o, core_body, "    ");
        o << "    }\n";
        o << "    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) { AC_ex[i_ex] = AC_ex[i_ex] + "
          << (alpha_exact ? ex_rat_expr(arat[(size_t)k])
                          : ("(numb)(" + fmtnum(alpha[k]) + ")"))
          << " * Z[i_ex];";
        if (k + 1 < K) o << " Z[i_ex] = Z0_ex[i_ex];";
        o << " }\n";
    }

    o << "\n    for (int i_ex = 0; i_ex < " << Ns << "; ++i_ex) X[i_ex] = AC_ex[i_ex].re;\n";
    return o.str();
}

// --- Адаптивный шаг экстраполяторов ----------------------------------------
//
// Вложенная пара из одних и тех же стадий T_1..T_K: старшее решение Y = sum alpha_k T_k
// (веса и порядок — как у постоянного шага), младшее — по первым K-1 стадиям с весами
// beta (порядок extrapolation_order(K-1)). Оценка ошибки E = Y - Yhat = sum (alpha_k -
// beta_k) T_k, beta_K = 0 — по разности весов, как у RK-пар. Это обычная локальная
// экстраполяция: регулятор ведёт по ошибке Yhat (показатель 1/(q+1)), а шаг продолжает Y.
// Y печатается тем же выражением, что в wrap_extrapolation / scheme_gbs, поэтому
// принятый шаг с тем же h совпадает с шагом постоянного Extr / GBS (GBS берёт f(X) из F0
// — то же значение, на вычисление f меньше на стадию). F1 = f(Y): FSAL драйвера и
// эрмитов плотный выход 3-го порядка (как у DOPRI78).
namespace {

// Веса alpha (по всем K стадиям) и разности alpha - beta литералами: alpha — точными
// дробями, когда сходятся (как у постоянного шага), разности — double: это только оценка.
struct ExtrAdWeights {
    std::vector<std::string> alpha, diff;
};

ExtrAdWeights extr_ad_weights(const std::vector<int>& n, int p, bool symmetric) {
    const int K = (int)n.size();
    const std::vector<double> alpha = extrapolation_weights(n, p, symmetric);
    const std::vector<int> nlo(n.begin(), n.end() - 1);
    const std::vector<double> beta = extrapolation_weights(nlo, p, symmetric);
    std::vector<ExRat> arat;
    const bool exact = extrapolation_weights_rational(n, p, symmetric, arat);
    ExtrAdWeights w;
    for (int k = 0; k < K; ++k) {
        w.alpha.push_back(exact ? ex_rat_expr(arat[(size_t)k]) : ("(numb)(" + fmtnum(alpha[k]) + ")"));
        const double b = k < K - 1 ? beta[(size_t)k] : 0.0;
        w.diff.push_back("(numb)(" + fmtnum(alpha[k] - b) + ")");
    }
    return w;
}

// Общая часть AdaptiveCode экстраполятора: правая часть, плотный выход, порядки.
AdaptiveCode extr_ad_code(const System& s, int p, int q, int emb_rhs) {
    AdaptiveCode c;
    c.rhs = adaptive_rhs_body(s);
    c.p = p; c.q = q; c.nlo = 1; c.wstages = 0; c.dsize = 4;
    c.emb_rhs = emb_rhs; c.dprep_rhs = 0; c.dense_order = 3;
    c.dprep = hermite_dprep_body(s); c.deval = hermite_deval_body(s);
    return c;
}

// Хвост emb: Y, E и F1 = f(Y).
void extr_ad_finish(std::ostringstream& o, const System& s, const char* ac, const char* ec) {
    const int N = (int)s.vars.size();
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { Y[i_ea] = " << ac << "[i_ea]; E[i_ea] = "
      << ec << "[i_ea]; }\n";
    const auto fy = rhs_over(s, "Y");
    for (int v = 0; v < N; ++v) o << "    F1[" << v << "] = (" << fy[(size_t)v] << ");\n";
}

} // namespace

AdaptiveCode codegen_adaptive_extrapolation(const System& s, const std::string& base_body,
                                            const std::vector<int>& n, int p, bool symmetric,
                                            const std::string& base_name, double base_rhs) {
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size(), K = (int)n.size();
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs an extrapolation with at least two stages "
                                        "(the lower-order solution comes from the first K-1)");
    if (base_body.empty()) throw std::runtime_error("extrapolation base '" + base_name + "' has no step body");
    const ExtrAdWeights w = extr_ad_weights(n, p, symmetric);
    long long cost = 0;
    for (int v : n) cost += v;
    const int P = extrapolation_order(K, p, symmetric), Q = extrapolation_order(K - 1, p, symmetric);

    std::ostringstream o;
    o << "    // --- adaptive " << make_extrapolation_name(base_name, n) << ": Y over " << K
      << " stages (order " << P << "), estimate over the first " << K - 1 << " (order " << Q << ") ---\n";
    o << "    numb X0_ea[" << N << "], AC_ea[" << N << "], EC_ea[" << N << "];\n";
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { X0_ea[i_ea] = X[i_ea]; AC_ea[i_ea] = (numb)0; "
         "EC_ea[i_ea] = (numb)0; }\n";
    o << "    (void)F0; (void)W;\n";
    o << "    {\n";
    // База шагает по X на месте: локальная копия затеняет константный вход emb.
    o << "    numb XS_ea[" << N << "];\n";
    o << "    numb* const X = XS_ea;\n";
    const bool inl = wrapper_can_inline(base_body);
    if (inl) {
        o << "    const numb h_ea = h;\n";
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_ea = [&](const numb h) {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n";
    }
    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s") << " of h/" << n[k] << "\n";
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) X[i_ea] = X0_ea[i_ea];\n";
        if (inl) {
            o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) {\n";
            o << "        const numb h = h_ea / (numb)" << fmtnum((double)n[k]) << ";\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        } else {
            o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) step_ea(h / (numb)" << fmtnum((double)n[k]) << ");\n";
        }
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { AC_ea[i_ea] += " << w.alpha[(size_t)k]
          << " * X[i_ea]; EC_ea[i_ea] += " << w.diff[(size_t)k] << " * X[i_ea]; }\n";
    }
    o << "    }\n";
    extr_ad_finish(o, s, "AC_ea", "EC_ea");

    // Вызовов f на попытку: все подшаги всех стадий + F1 = f(Y). Дробная f базы (неявный
    // полушаг CD по компонентам) округляется уже в произведении.
    AdaptiveCode c = extr_ad_code(s, P, Q, base_rhs > 0 ? (int)std::lround((double)cost * base_rhs) + 1 : 1);
    c.emb = o.str();
    return c;
}

// ExtrZ: стадии и обе суммы — ucmplx, ядро базы шагает по Z[] без Re между подшагами, Re
// берётся на выходе (как wrap_extrapolation_complex). Веса — симметричные: разложение Re стадии
// идёт по чётным степеням, поэтому Re младшей комбинации — решение порядка
// extrapolation_order(K-1, p, true), и E = Re sum (alpha - beta) Z — его оценка.
AdaptiveCode codegen_adaptive_extrapolation_complex(const System& s, const std::string& core_body,
                                                    const std::vector<int>& n, int p,
                                                    const std::string& base_name, double base_rhs) {
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size(), K = (int)n.size();
    if (N < 1) throw std::runtime_error("extrapolation needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs an extrapolation with at least two stages "
                                        "(the lower-order solution comes from the first K-1)");
    if (core_body.empty()) throw std::runtime_error("ExtrZ base '" + base_name + "' has no complex core");
    const bool symmetric = extrapolation_symmetric(true, false);
    const ExtrAdWeights w = extr_ad_weights(n, p, symmetric);
    const int P = extrapolation_order(K, p, symmetric), Q = extrapolation_order(K - 1, p, symmetric);

    std::ostringstream o;
    o << "    // --- adaptive " << make_extrapolation_name(base_name, n, {}, true) << ": Y = Re over " << K
      << " stages (order " << P << "), estimate over the first " << K - 1 << " (order " << Q << ") ---\n";
    // Z — имя, над которым работает ядро базы.
    o << "    ucmplx Z[" << N << "], Z0_ea[" << N << "], AC_ea[" << N << "], EC_ea[" << N << "];\n";
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) {\n"
      << "        Z0_ea[i_ea] = ucmplx(X[i_ea], 0.0); AC_ea[i_ea] = ucmplx(0.0, 0.0); EC_ea[i_ea] = ucmplx(0.0, 0.0);\n"
      << "    }\n";
    o << "    (void)F0; (void)W;\n";
    // Ядро — встроенная CD-схема, return/goto в нём нет: подставляется всегда.
    o << "    const numb h_ea = h;\n";
    for (int k = 0; k < K; ++k) {
        o << "    // stage " << k << ": " << n[k] << " substep" << (n[k] == 1 ? "" : "s") << " of h/" << n[k]
          << ", no Re in between\n";
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) Z[i_ea] = Z0_ea[i_ea];\n";
        o << "    for (int s_ea = 0; s_ea < " << n[k] << "; ++s_ea) {\n";
        o << "        const numb h = h_ea / (numb)" << fmtnum((double)n[k]) << ";\n";
        emit_indented(o, core_body, "    ");
        o << "    }\n";
        // Сумма — тем же выражением, что у постоянного ExtrZ (AC = AC + alpha * Z).
        o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { AC_ea[i_ea] = AC_ea[i_ea] + " << w.alpha[(size_t)k]
          << " * Z[i_ea]; EC_ea[i_ea] = EC_ea[i_ea] + " << w.diff[(size_t)k] << " * Z[i_ea]; }\n";
    }
    o << "    for (int i_ea = 0; i_ea < " << N << "; ++i_ea) { Y[i_ea] = AC_ea[i_ea].re; E[i_ea] = EC_ea[i_ea].re; }\n";
    const auto fy = rhs_over(s, "Y");
    for (int v = 0; v < N; ++v) o << "    F1[" << v << "] = (" << fy[(size_t)v] << ");\n";

    // Как у Extr: подшаги стадий x f базы (комплексное ядро делает те же вызовы) + F1.
    long long cost = 0;
    for (int v : n) cost += v;
    AdaptiveCode c = extr_ad_code(s, P, Q, base_rhs > 0 ? (int)std::lround((double)cost * base_rhs) + 1 : 1);
    c.emb = o.str();
    return c;
}

AdaptiveCode codegen_adaptive_gbs(const System& s, int K) {
    if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
    const int N = (int)s.vars.size();
    if (N < 1) throw std::runtime_error("GBS needs a non-empty system");
    if (K < 2) throw std::runtime_error("adaptive step needs GBS 2-4 or longer (the lower-order solution comes "
                                        "from the first K-1 stages)");
    const std::vector<int> n = gbs_substeps(K);
    const ExtrAdWeights w = extr_ad_weights(n, 2, true);
    const auto f1 = rhs_over(s, "Z1_gbs");
    int cost = 0;
    for (int v : n) cost += v;   // n - 1 leapfrog + сглаживание; f(X) — из F0

    const std::string Ns = std::to_string(N), Ks = std::to_string(K);
    std::ostringstream o;
    o << "    // --- adaptive GBS";
    for (int k = 0; k < K; ++k) o << (k ? "-" : " ") << n[k];
    o << ": Y order " << 2 * K << ", estimate over the first " << K - 1 << " stages (order " << 2 * (K - 1) << ") ---\n";
    o << "    numb X0_gbs[" << Ns << "], AC_gbs[" << Ns << "], EC_gbs[" << Ns << "], Z0_gbs[" << Ns
      << "], Z1_gbs[" << Ns << "], ZT_gbs[" << Ns << "], T_gbs[" << Ns << "];\n";
    o << "    const numb W_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) o << (k ? ", " : "") << w.alpha[(size_t)k];
    o << " };\n";
    o << "    const numb V_gbs[" << Ks << "] = { ";
    for (int k = 0; k < K; ++k) o << (k ? ", " : "") << w.diff[(size_t)k];
    o << " };\n";
    o << "    (void)W;\n";
    o << "    int i_gbs, m_gbs, k_gbs;\n";
    o << "    for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { X0_gbs[i_gbs] = X[i_gbs]; AC_gbs[i_gbs] = (numb)0; "
         "EC_gbs[i_gbs] = (numb)0; }\n";
    o << "    for (k_gbs = 0; k_gbs < " << Ks << "; ++k_gbs) {\n";
    o << "        const int  n_gbs  = 2 * (k_gbs + 1);\n";
    o << "        const numb hs_gbs = h / (numb)n_gbs;\n";
    o << "        // старт: явный Эйлер на первый подшаг, f(X) — F0 драйвера\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) {\n";
    o << "            Z0_gbs[i_gbs] = X0_gbs[i_gbs];\n";
    o << "            Z1_gbs[i_gbs] = X0_gbs[i_gbs] + hs_gbs * F0[i_gbs];\n";
    o << "        }\n";
    o << "        // leapfrog: z_{m+1} = z_{m-1} + 2*hs*f(z_m), всего n подшагов\n";
    o << "        for (m_gbs = 1; m_gbs < n_gbs; ++m_gbs) {\n";
    for (int i = 0; i < N; ++i)
        o << "            ZT_gbs[" << i << "] = Z0_gbs[" << i << "] + (numb)2 * hs_gbs * (" << f1[(size_t)i] << ");\n";
    o << "            for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { Z0_gbs[i_gbs] = Z1_gbs[i_gbs]; Z1_gbs[i_gbs] = ZT_gbs[i_gbs]; }\n";
    o << "        }\n";
    // (numb)0.5 * (...) и W * (0.5 * x) == (W * 0.5) * x точно: множитель — степень двойки.
    o << "        // сглаживание Грэгга: (z_{n-1} + z_n + hs*f(z_n)) / 2\n";
    for (int i = 0; i < N; ++i)
        o << "        T_gbs[" << i << "] = (numb)0.5 * (Z0_gbs[" << i << "] + Z1_gbs[" << i << "] + hs_gbs * ("
          << f1[(size_t)i] << "));\n";
    o << "        for (i_gbs = 0; i_gbs < " << Ns << "; ++i_gbs) { AC_gbs[i_gbs] += W_gbs[k_gbs] * T_gbs[i_gbs]; "
         "EC_gbs[i_gbs] += V_gbs[k_gbs] * T_gbs[i_gbs]; }\n";
    o << "    }\n";
    extr_ad_finish(o, s, "AC_gbs", "EC_gbs");

    AdaptiveCode c = extr_ad_code(s, 2 * K, 2 * (K - 1), cost + 1);
    c.emb = o.str();
    return c;
}

int builtin_scheme_rhs_per_step(const std::string& name) {
    static const struct { const char* name; int k; } kTab[] = {
        { "Euler", 1 }, { "Explicit Midpoint", 2 }, { "RK4", 4 }, { "RK45", 6 },
        { "DOP853", 12 }, { "DOPRI78", 13 }, { "DOPRI78 (legacy)", 13 },
    };
    for (const auto& t : kTab) if (name == t.name) return t.k;
    return 0;
}

// Неявный полушаг CD (и стадия SIMP, и шаг D): по компоненте — одно вычисление f_i, если она
// линейна по своей переменной (аналитика), иначе CD_ITERS простых итераций; в единицах полной f.
static double cd_implicit_sweep_rhs(const System& s) {
    const int N = (int)s.vars.size();
    if (N < 1 || (int)s.rhs.size() != N) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        Parser p(s.rhs[(size_t)i], s.latex);
        PN coef, rem;
        sum += cd_try_extract_linear(p.parse(), s.vars[(size_t)i], coef, rem) ? 1.0 : (double)CD_ITERS;
    }
    return sum / (double)N;
}

double scheme_rhs_per_step(const System& s, const std::string& name) {
    if (const int k = builtin_scheme_rhs_per_step(name); k > 0) return (double)k;
    ExtrapolationSpec es;
    if (parse_extrapolation_name(name, &es)) {
        long long cost = 0;
        for (int v : es.n) cost += v;
        return (double)cost * scheme_rhs_per_step(s, es.base);   // ExtrZ: ядро базы — те же f
    }
    CompositionSpec cs;
    if (parse_composition_name(name, &cs))
        return (double)cs.gammas.size() * scheme_rhs_per_step(s, cs.base);
    const Scheme sch = scheme_from_name(name);
    if (const int K = gbs_stage_count(sch); K >= 1) {
        int cost = 1;                                 // f(X) на всех стадиях одна
        for (int k = 1; k <= K; ++k) cost += 2 * k;   // n - 1 leapfrog + сглаживание
        return (double)cost;
    }
    try {
        switch (sch) {
        case Scheme::EulerCromer:  return 1.0;
        case Scheme::SEMP:         return 2.0;
        case Scheme::SIMP:         return cd_implicit_sweep_rhs(s) + 1.0;
        case Scheme::D:            return cd_implicit_sweep_rhs(s);
        case Scheme::CD: case Scheme::CD10:
        case Scheme::ComplexCD: case Scheme::ComplexCD10:
            return 1.0 + cd_implicit_sweep_rhs(s);
        case Scheme::ComplexCD4: case Scheme::ComplexCD4_10:
        case Scheme::ComplexCD4SS01: case Scheme::ComplexCD4SS10:
            return 2.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::ComplexCD4S3: case Scheme::ComplexCD4S3_10:
            return 3.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::ComplexCD4S4: case Scheme::ComplexCD4S4_10:
            return 4.0 * (1.0 + cd_implicit_sweep_rhs(s));
        case Scheme::Map:          return 1.0;
        default:                   return 0.0;   // Ньютон (Implicit Euler/Midpoint, Complex IE), кастомная КРС
        }
    } catch (const std::exception&) {
        return 0.0;   // правая часть не разбирается — счёт неизвестен, а не ошибка
    }
}

bool adaptive_scheme_name_supported(const std::string& name) {
    if (scheme_supports_adaptive(scheme_from_name(name)) && name != "Euler") return true;
    if (gbs_stage_count(scheme_from_name(name)) >= 2) return true;
    ExtrapolationSpec spec;
    return parse_extrapolation_name(name, &spec) && spec.n.size() >= 2;   // Extr и ExtrZ
}

int adaptive_scheme_q_guess(const std::string& name) {
    if (name == "RK45") return 4;
    const int K = gbs_stage_count(scheme_from_name(name));
    if (K >= 2) return 2 * (K - 1);
    ExtrapolationSpec spec;
    int p = 0; bool sym = false;
    if (parse_extrapolation_name(name, &spec) && spec.n.size() >= 2 && builtin_scheme_traits(spec.base, &p, &sym))
        return extrapolation_order((int)spec.n.size() - 1, p, extrapolation_symmetric(spec.re_at_output, sym));
    return 7;
}

// --- Композиция -------------------------------------------------------------

static const char* const kCompPrefix = "Comp(";

std::string make_composition_name(const std::string& base,
                                  const std::vector<std::string>& gammas,
                                  const std::string& label) {
    std::string s = kCompPrefix;
    if (!label.empty()) s += label + "|";
    s += base + "|";
    for (size_t k = 0; k < gammas.size(); ++k) {
        if (k) s += ",";
        s += gammas[k];
    }
    return s + ")";
}

namespace {

    std::string comp_trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
        return s.substr(a, b - a);
    }

    // Режем только по запятым ВЕРХНЕГО уровня: коэффициент — выражение, и
    // запятая внутри pow(g,2) не должна заканчивать токен.
    bool comp_split(const std::string& tail, std::vector<std::string>& out) {
        int depth = 0;
        std::string cur;
        for (char c : tail) {
            if (c == '(') ++depth;
            else if (c == ')' && --depth < 0) return false;
            if (c == ',' && depth == 0) { out.push_back(cur); cur.clear(); continue; }
            cur += c;
        }
        if (depth != 0) return false;
        out.push_back(cur);
        return true;
    }

    // Сворачивает коэффициент в число. false на любом символе — тогда значение
    // известно только в рантайме, и звать нас незачем.
    bool comp_fold(const PN& n, double* v) {
        if (!n) return false;
        double a = 0, b = 0;
        switch (n->kind) {
        case Node::Num: *v = n->num; return true;
        case Node::Neg: if (!comp_fold(n->a, &a)) return false; *v = -a; return true;
        case Node::Add: case Node::Sub: case Node::Mul: case Node::Div: case Node::Pow:
            if (!comp_fold(n->a, &a) || !comp_fold(n->b, &b)) return false;
            switch (n->kind) {
            case Node::Add: *v = a + b; break;
            case Node::Sub: *v = a - b; break;
            case Node::Mul: *v = a * b; break;
            case Node::Div: if (b == 0.0) return false; *v = a / b; break;
            default:        *v = std::pow(a, b); break;
            }
            return true;
        default: return false;   // Sym / Call
        }
    }

} // namespace

bool parse_composition_name(const std::string& name, CompositionSpec* out,
                            std::string* err) {
    auto fail = [&](const char* msg) { if (err) *err = msg; return false; };
    auto fail_s = [&](const std::string& msg) { if (err) *err = msg; return false; };

    const std::string pref = kCompPrefix;
    if (name.size() <= pref.size() || name.compare(0, pref.size(), pref) != 0)
        return fail("not a composed scheme name");
    if (name.back() != ')')
        return fail("missing closing ')'");

    const std::string inner = name.substr(pref.size(), name.size() - pref.size() - 1);
    // rfind, как у Extr: имя базы по правилам валидации '|' не содержит, но из
    // старого JSON могло приехать что угодно — резать надо по последней черте.
    const size_t bar = inner.rfind('|');
    if (bar == std::string::npos) return fail("missing '|' between base and coefficients");

    CompositionSpec spec;
    // Голова — "<база>" либо "<метка>|<база>", как у Extr.
    if (!wrap_split_head(inner.substr(0, bar), &spec.label, &spec.base, err)) return false;
    if (spec.base.empty()) return fail("empty base scheme name");
    // Обёртка над обёрткой отрезает рекурсию в резолвере, как и у Extr.
    if (spec.base.compare(0, pref.size(), pref) == 0 ||
        spec.base.compare(0, 5, "Extr(") == 0 || spec.base.compare(0, 6, "ExtrZ(") == 0)
        return fail("base scheme cannot itself be a wrapper");

    std::vector<std::string> toks;
    if (!comp_split(inner.substr(bar + 1), toks)) return fail("unbalanced parentheses");

    for (const std::string& t : toks) {
        const std::string g = comp_trim(t);
        if (g.empty()) return fail("empty coefficient");
        // Синтаксис проверяем здесь, чтобы кривое имя не доехало до Run: имена
        // резолвятся позже, в wrap_composition, где уже известна система.
        try { Parser(g, false).parse(); }
        catch (const std::exception& e) {
            if (err) *err = "coefficient \"" + g + "\": " + e.what();
            return false;
        }
        spec.gammas.push_back(g);
    }

    const int K = (int)spec.gammas.size();
    if (K < kCompMinStages)
        return fail_s("at least " + std::to_string(kCompMinStages) + " stages are needed");
    if (K > kCompMaxStages)
        return fail_s("at most " + std::to_string(kCompMaxStages) + " stages are supported");

    if (out) *out = spec;
    return true;
}

namespace {

    // Чистый числовой литерал: знак, цифры, точка, экспонента — и ничего
    // больше. Нужен ровно одному месту: решить, был ли первый столбец строки
    // нумерующим. Поэтому и такая строгость — "- 2*g1" числом не считается,
    // и "1 - 2*g1" не теряет свою единицу.
    bool comp_is_plain_number(const std::string& s) {
        if (s.empty()) return false;
        size_t i = 0;
        if (s[i] == '+' || s[i] == '-') ++i;
        bool digits = false, dot = false;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.')) {
            if (s[i] == '.') { if (dot) return false; dot = true; }
            else digits = true;
            ++i;
        }
        if (!digits) return false;
        if (i == s.size()) return true;
        if (s[i] != 'e' && s[i] != 'E') return false;
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        if (i == s.size()) return false;
        while (i < s.size()) { if (!std::isdigit((unsigned char)s[i])) return false; ++i; }
        return true;
    }

    // Срезает нумерующий первый столбец: "3<tab>-0.3825" -> "-0.3825".
    // Разделителем считаются пробелы/табы и ')' ';' ':' — но НЕ точка (иначе
    // "1.35" распалось бы на индекс 1 и коэффициент 35) и НЕ запятая (её
    // разбирает разделение строки на коэффициенты).
    std::string comp_strip_index(const std::string& line) {
        size_t i = 0;
        while (i < line.size() && std::isdigit((unsigned char)line[i])) ++i;
        if (i == 0 || i == line.size()) return line;
        size_t j = i;
        while (j < line.size() && (std::isspace((unsigned char)line[j])
                                   || line[j] == ')' || line[j] == ';' || line[j] == ':')) ++j;
        if (j == i || j == line.size()) return line;
        const std::string rest = comp_trim(line.substr(j));
        return comp_is_plain_number(rest) ? rest : line;
    }

} // namespace

bool parse_composition_coeff_file(const std::string& text,
                                  std::vector<std::string>* out, std::string* err) {
    auto fail = [&](const std::string& msg) { if (err) *err = msg; return false; };

    std::vector<std::string> coeffs;
    size_t pos = 0;
    int line_no = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const std::string raw = text.substr(pos, eol == std::string::npos
                                                     ? std::string::npos : eol - pos);
        pos = (eol == std::string::npos) ? text.size() + 1 : eol + 1;
        ++line_no;

        // comp_trim снимает и '\r' от CRLF, и табуляцию в конце строки —
        // именно в таком виде таблицы коэффициентов обычно и лежат.
        std::string line = comp_trim(raw);
        if (line.empty()) continue;
        if (line[0] == '#' || line[0] == '%' || line.compare(0, 2, "//") == 0) continue;

        line = comp_strip_index(line);

        // Запятые верхнего уровня — тем же расщепителем, что у имени схемы:
        // так "g1, 1-2*g1, g1" в одну строку работает, а pow(g,2) не рвётся.
        std::vector<std::string> toks;
        if (!comp_split(line, toks))
            return fail("line " + std::to_string(line_no) + ": unbalanced parentheses");
        for (const std::string& t : toks) {
            const std::string g = comp_trim(t);
            if (g.empty()) continue;   // хвостовая запятая — не ошибка
            try { Parser(g, false).parse(); }
            catch (const std::exception& e) {
                return fail("line " + std::to_string(line_no) + ": \"" + g + "\": " + e.what());
            }
            coeffs.push_back(g);
        }
    }

    if ((int)coeffs.size() < kCompMinStages)
        return fail("found " + std::to_string(coeffs.size()) + " coefficient(s), at least "
                    + std::to_string(kCompMinStages) + " are needed");
    if ((int)coeffs.size() > kCompMaxStages)
        return fail("found " + std::to_string(coeffs.size()) + " coefficients, at most "
                    + std::to_string(kCompMaxStages) + " stages are supported");

    if (out) *out = coeffs;
    return true;
}

bool composition_sums(const CompositionSpec& spec, double* sum, double* cube_sum) {
    double s = 0.0, c = 0.0;
    for (const std::string& g : spec.gammas) {
        double v = 0.0;
        PN ast;
        try { ast = Parser(g, false).parse(); } catch (...) { return false; }
        if (!comp_fold(ast, &v)) return false;
        s += v;
        c += v * v * v;
    }
    if (sum)      *sum = s;
    if (cube_sum) *cube_sum = c;
    return true;
}

namespace {

    // comp_fold с окружением: Sym берётся из env, Call считается для функций,
    // которые в коэффициенте шага вообще осмысленны (sqrt(2), cbrt(2) в
    // классических композициях). Неизвестное имя — честный false, а не ноль:
    // подставить ноль значило бы показать пользователю число, которого схема
    // считать не будет.
    bool comp_fold_env(const PN& n, const std::map<std::string, double>& env, double* v) {
        if (!n) return false;
        double a = 0, b = 0;
        switch (n->kind) {
        case Node::Num: *v = n->num; return true;
        case Node::Sym: {
            auto it = env.find(n->name);
            if (it == env.end()) return false;
            *v = it->second; return true;
        }
        case Node::Neg:
            if (!comp_fold_env(n->a, env, &a)) return false;
            *v = -a; return true;
        case Node::Add: case Node::Sub: case Node::Mul: case Node::Div: case Node::Pow:
            if (!comp_fold_env(n->a, env, &a) || !comp_fold_env(n->b, env, &b)) return false;
            switch (n->kind) {
            case Node::Add: *v = a + b; break;
            case Node::Sub: *v = a - b; break;
            case Node::Mul: *v = a * b; break;
            case Node::Div: if (b == 0.0) return false; *v = a / b; break;
            default:        *v = std::pow(a, b); break;
            }
            return true;
        case Node::Call: {
            if (n->args.size() == 1) {
                if (!comp_fold_env(n->args[0], env, &a)) return false;
                const std::string& f = n->name;
                if (f == "sqrt") { if (a < 0) return false; *v = std::sqrt(a); return true; }
                if (f == "cbrt")               { *v = std::cbrt(a); return true; }
                if (f == "exp")                { *v = std::exp(a);  return true; }
                if (f == "log")  { if (a <= 0) return false; *v = std::log(a); return true; }
                if (f == "sin")                { *v = std::sin(a);  return true; }
                if (f == "cos")                { *v = std::cos(a);  return true; }
                if (f == "tan")                { *v = std::tan(a);  return true; }
                if (f == "fabs" || f == "abs") { *v = std::fabs(a); return true; }
                return false;
            }
            if (n->args.size() == 2 && n->name == "pow") {
                if (!comp_fold_env(n->args[0], env, &a)) return false;
                if (!comp_fold_env(n->args[1], env, &b)) return false;
                *v = std::pow(a, b); return true;
            }
            return false;
        }
        default: return false;
        }
    }

    // Округляет КАЖДЫЙ числовой литерал в строке, не трогая имена и операторы.
    // Работает и на цельном числе, и на выражении: "-1.2599210498948732*g1"
    // укорачивается так же, как "1.3512071919596578".
    std::string comp_round_literals(const std::string& s, int digits) {
        std::string out;
        for (size_t i = 0; i < s.size(); ) {
            const char c = s[i];
            // Идентификатор забирает свои цифры целиком, иначе g1 стало бы g1.
            if (std::isalpha((unsigned char)c) || c == '_') {
                while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '_'))
                    out += s[i++];
                continue;
            }
            const bool starts_num = std::isdigit((unsigned char)c)
                || (c == '.' && i + 1 < s.size() && std::isdigit((unsigned char)s[i + 1]));
            if (!starts_num) { out += c; ++i; continue; }

            const char* start = s.c_str() + i;
            char* end = nullptr;
            const double v = std::strtod(start, &end);
            if (end == start) { out += c; ++i; continue; }
            const size_t len = (size_t)(end - start);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
            // Короче из двух: у "2" и "1e-05" %g даёт ровно исходное.
            if (std::strlen(buf) < len) out += buf;
            else                        out.append(start, len);
            i += len;
        }
        return out;
    }

} // namespace

bool composition_gamma_values(const CompositionSpec& spec,
                              const std::map<std::string, std::string>& param_values,
                              std::vector<double>* out, std::string* err) {
    // Окружение: сначала константы, затем параметры. Значение параметра само
    // разбирается выражением (в библиотеке лежат "8/3" и "1e-5"), но уже БЕЗ
    // окружения: параметр, сосланный на другой параметр, — это цепочка, и
    // раскрывать её молча значило бы угадывать порядок вычисления.
    std::map<std::string, double> env;
    env["pi"] = 3.14159265358979323846;
    for (const auto& kv : param_values) {
        if (kv.second.empty()) continue;
        double v = 0.0;
        PN ast;
        try { ast = Parser(kv.second, false).parse(); } catch (...) { continue; }
        if (comp_fold(ast, &v)) env[kv.first] = v;
    }

    const int K = (int)spec.gammas.size();
    if (out) out->assign((size_t)K, std::numeric_limits<double>::quiet_NaN());
    bool all_ok = true;
    for (int k = 0; k < K; ++k) {
        const std::string& g = spec.gammas[(size_t)k];
        double v = 0.0;
        PN ast;
        try { ast = Parser(g, false).parse(); }
        catch (...) {
            if (all_ok && err) *err = "coefficient \"" + g + "\" does not parse";
            all_ok = false; continue;
        }
        if (!comp_fold_env(ast, env, &v)) {
            if (all_ok && err)
                *err = "coefficient \"" + g + "\" needs a parameter value that is not set";
            all_ok = false; continue;
        }
        if (out) (*out)[(size_t)k] = v;
    }
    return all_ok;
}

std::string wrapper_display_name(const std::string& name, int digits) {
    // Метка пользователя ЗАМЕНЯЕТ описание целиком: её задавали ровно затем,
    // чтобы в списке и в комбо стояло "S17o8", а не семнадцать коэффициентов.
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return esp.label.empty() ? name : esp.label;

    CompositionSpec spec;
    if (!parse_composition_name(name, &spec)) return name;
    if (!spec.label.empty()) return spec.label;
    if (digits < 1)  digits = 1;
    if (digits > 17) digits = 17;

    std::vector<std::string> shown;
    shown.reserve(spec.gammas.size());
    for (const std::string& g : spec.gammas)
        shown.push_back(comp_round_literals(g, digits));
    return make_composition_name(spec.base, shown);
}

// --- Порядок списка обёрток --------------------------------------------------

namespace {

// Ключ сортировки обёртки. Метку не содержит: переименование строку не двигает.
struct WrapperKey {
    int              kind = 3;      // 0 Extr, 1 ExtrZ, 2 Comp, 3 не разобралось
    int              base_rank = 0; // строка паспорта; кастомные — после всех встроенных
    std::string      base;          // для кастомных опорников — алфавит
    int              stages = 0;
    std::vector<int> n;             // подшаги Extr
    std::string      canon;         // имя без метки — последний разделитель
};

WrapperKey wrapper_key(const std::string& name) {
    WrapperKey k;
    ExtrapolationSpec esp;
    CompositionSpec   csp;
    if (parse_extrapolation_name(name, &esp)) {
        k.kind   = esp.re_at_output ? 1 : 0;
        k.base   = esp.base;
        k.stages = (int)esp.n.size();
        k.n      = esp.n;
    } else if (parse_composition_name(name, &csp)) {
        k.kind   = 2;
        k.base   = csp.base;
        k.stages = (int)csp.gammas.size();
    }
    const int r = builtin_scheme_rank(k.base);
    k.base_rank = r >= 0 ? r : INT_MAX;
    k.canon     = wrapper_canonical_name(name);
    return k;
}

bool wrapper_key_less(const WrapperKey& x, const WrapperKey& y) {
    if (x.kind != y.kind)           return x.kind < y.kind;
    if (x.base_rank != y.base_rank) return x.base_rank < y.base_rank;
    if (x.base != y.base)           return x.base < y.base;
    if (x.stages != y.stages)       return x.stages < y.stages;
    if (x.n != y.n)                 return x.n < y.n;
    return x.canon < y.canon;
}

} // namespace

void sort_wrapper_schemes(std::vector<std::string>& names) {
    std::vector<std::pair<WrapperKey, std::string>> v;
    v.reserve(names.size());
    for (auto& nm : names) v.emplace_back(wrapper_key(nm), std::move(nm));
    std::stable_sort(v.begin(), v.end(), [](const auto& x, const auto& y) {
        return wrapper_key_less(x.first, y.first);
    });
    for (size_t i = 0; i < v.size(); ++i) names[i] = std::move(v[i].second);
}

std::string wrapper_group_title(const std::string& name) {
    static const char* const kKind[] = { "Extr", "ExtrZ", "Comp" };
    const WrapperKey k = wrapper_key(name);
    if (k.kind > 2) return {};
    return std::string(kKind[k.kind]) + " over " + k.base;
}

std::string wrapper_canonical_name(const std::string& name) {
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return make_extrapolation_name(esp.base, esp.n, {}, esp.re_at_output);
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return make_composition_name(csp.base, csp.gammas);
    return name;
}

std::string wrapper_relabel(const std::string& name, const std::string& label) {
    const std::string lb = wrapper_sanitize_label(label);
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp))
        return make_extrapolation_name(esp.base, esp.n, lb, esp.re_at_output);
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return make_composition_name(csp.base, csp.gammas, lb);
    return name;
}

std::string wrapper_label(const std::string& name) {
    ExtrapolationSpec esp;
    if (parse_extrapolation_name(name, &esp)) return esp.label;
    CompositionSpec csp;
    if (parse_composition_name(name, &csp)) return csp.label;
    return {};
}

std::string wrap_composition(const std::string& base_body, const System& sys,
                             const std::vector<std::string>& gammas,
                             int p, bool symmetric, const std::string& base_name) {
    const int K = (int)gammas.size();
    if (K < 1) throw std::runtime_error("composition needs at least one stage");
    if (base_body.empty()) throw std::runtime_error("composition needs a base body");

    // Коэффициенты видят ТОЛЬКО параметры и константы: переменная состояния
    // здесь означала бы шаг, зависящий от X посреди шага.
    NameMap nm;
    for (size_t j = 0; j < sys.params.size(); ++j)
        nm.m[sys.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
    for (const auto& c : math_constants()) nm.m[c] = c;

    std::vector<std::string> gc((size_t)K);
    for (int k = 0; k < K; ++k) {
        try { gc[(size_t)k] = emit_to_str(Parser(gammas[(size_t)k], false).parse(), nm); }
        catch (const std::exception& e) {
            throw std::runtime_error("composition coefficient \"" + gammas[(size_t)k]
                                     + "\": " + e.what()
                                     + " (only system parameters and pi are allowed)");
        }
    }

    std::ostringstream o;
    o << "    // --- " << make_composition_name(base_name, gammas) << " ---\n";
    o << "    // base: order " << p << ", " << (symmetric ? "symmetric" : "non-symmetric")
      << "; " << K << " stages, " << K << " base steps per macro-step\n";

    CompositionSpec probe;
    probe.gammas = gammas;
    double gsum = 0.0, gcube = 0.0;
    if (!composition_sums(probe, &gsum, &gcube)) {
        o << "    // coefficients are symbolic -- the order follows their run-time\n"
             "    // values; sweep them in the Order tab to see it\n";
    } else {
        o << "    // sum(gamma) = " << fmtnum(gsum)
          << (std::fabs(gsum - 1.0) < 1e-12
                  ? "  (consistent)\n"
                  : "  -- NOT 1: this integrates the field scaled by that factor\n");
        o << "    // sum(gamma^3) = " << fmtnum(gcube);
        if (symmetric && std::fabs(gcube) < 1e-12) o << "  ->  expected order " << (p + 2) << "\n";
        else if (symmetric)                        o << "  -- not 0, order stays " << p << "\n";
        else                                       o << "  (base is not symmetric: order not implied)\n";
    }
    for (int k = 0; k < K; ++k)
        o << "    //   gamma[" << k << "] = " << gammas[(size_t)k]
          << (gc[(size_t)k] == gammas[(size_t)k] ? "" : "  ->  " + gc[(size_t)k]) << "\n";

    // Та же подстановка, что у wrap_extrapolation, и по той же причине:
    // внутренний h затеняет макрошаг, поэтому база не правится ни в одном символе.
    if (wrapper_can_inline(base_body)) {
        o << "    const numb h_co = h;\n";
        for (int k = 0; k < K; ++k) {
            o << "    {   // stage " << k << "\n";
            o << "        const numb h = h_co * (" << gc[(size_t)k] << ");\n";
            emit_indented(o, base_body, "    ");
            o << "    }\n";
        }
    } else {
        o << "    // base contains return/goto: kept in a lambda, see wrapper_can_inline\n";
        o << "    auto step_co = [&](const numb h) {\n";
        emit_indented(o, base_body, "    ");
        o << "    };\n\n";
        for (int k = 0; k < K; ++k)
            o << "    step_co(h * (" << gc[(size_t)k] << "));\n";
    }

    return o.str();
}

std::string codegen_scheme_cpu_equivalent(const System& s, Scheme sch) {
    // Одна форма на оба пути. CPU-интегратор считает тот же AST через
    // байткод-интерпретатор, а диагонально-неявные схемы (CD, Complex CD, SIMP,
    // D) решают каждое уравнение той же аналитической формулой — коэффициенты
    // даёт SystemEvaluator::solve_diag_implicit, построенный тем же
    // cd_try_extract_linear, что и кодоген. Итерации остаются только там, где
    // их эмитит и GPU: уравнение нелинейно по своей переменной.
    // Расхождение возможно лишь в группировке FMA у компилятора.
    return codegen_scheme(s, sch);
}

std::vector<std::string> codegen_rhs(const System& s) {
    return rhs_over(s, "X");
}

// Нормализация значения параметра
std::string normalize_value(const std::string& value) {
    // trim
    size_t a = value.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return ""; // пустое = не задано
    size_t b = value.find_last_not_of(" \t\n\r");
    std::string v = value.substr(a, b - a + 1);

    // парсим как обычное выражение (не LaTeX) и печатаем в C-синтаксис.
    // числа получат .0, "8/3" -> "8.0 / 3.0".
    Parser p(v, false);
    PN ast = p.parse();
    NameMap empty; // значения не содержат имён переменных/параметров
    std::ostringstream o;
    emit(ast, empty, o);
    return o.str();
}
