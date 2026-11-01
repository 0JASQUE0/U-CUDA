#include "expression_internal.h"
using namespace ucuda::equations_detail;

namespace ucuda::equations_detail {
PN mk(Node::Kind k) { auto n = std::make_shared<Node>(); n->kind = k; return n; }
}

namespace ucuda::equations_detail {
const std::set<std::string>& known_funcs() {
        static const std::set<std::string> f = {
            "sin","cos","tan","asin","acos","atan","sinh","cosh","tanh",
            "exp","log","log2","log10","sqrt","cbrt","fabs","abs",
            "pow","atan2","fmod","floor","ceil",
            "copysign",  // производная fabs (pn_diff): её печатает pn_print, и текст читается обратно
            "fmin", "fmax", "min", "max"   // min/max разбираются в fmin/fmax (так их печатает emit)
        };
        return f;
    }
}

namespace ucuda::equations_detail {
const std::map<std::string, std::string>& latex_funcs() {
        static const std::map<std::string, std::string> m = {
            {"sin","sin"},{"cos","cos"},{"tan","tan"},{"arcsin","asin"},
            {"arccos","acos"},{"arctan","atan"},{"sinh","sinh"},{"cosh","cosh"},
            {"tanh","tanh"},{"exp","exp"},{"ln","log"},{"log","log"},{"sqrt","sqrt"}
        };
        return m;
    }
}

namespace ucuda::equations_detail {
const std::map<std::string, std::string>& latex_symbols() {
        static const std::map<std::string, std::string> m = {
            {"alpha","alpha"},{"beta","beta"},{"gamma","gamma"},{"delta","delta"},
            {"epsilon","epsilon"},{"varepsilon","epsilon"},{"zeta","zeta"},{"eta","eta"},
            {"theta","theta"},{"vartheta","theta"},{"iota","iota"},{"kappa","kappa"},
            {"lambda","lambda"},{"mu","mu"},{"nu","nu"},{"xi","xi"},{"pi","pi"},
            {"rho","rho"},{"sigma","sigma"},{"tau","tau"},{"phi","phi"},{"varphi","phi"},
            {"chi","chi"},{"psi","psi"},{"omega","omega"},{"Gamma","Gamma"},
            {"Delta","Delta"},{"Theta","Theta"},{"Lambda","Lambda"},{"Sigma","Sigma"},
            {"Phi","Phi"},{"Psi","Psi"},{"Omega","Omega"}
        };
        return m;
    }
}

namespace ucuda::equations_detail {
int prec(Node::Kind k) {
        switch (k) {
        case Node::Add: case Node::Sub: return 1;
        case Node::Mul: case Node::Div: return 2;
        case Node::Neg: return 3; case Node::Pow: return 4; default: return 5;
        }
    }
}

namespace ucuda::equations_detail {
std::string fmtnum(double v) {
        // целые числа в разумном диапазоне печатаем как "N.0", без научной формы
        // (иначе 10 -> "1e+01"). Большие/дробные -> кратчайший round-trip.
        if (v == (double)(long long)v) {
            long long iv = (long long)v;
            if (iv > -1000000000000LL && iv < 1000000000000LL) {
                return std::to_string(iv) + ".0";
            }
        }
        char buf[64];
        for (int p = 1; p <= 17; ++p) {
            std::snprintf(buf, sizeof(buf), "%.*g", p, v);
            if (std::strtod(buf, nullptr) == v) break;
        }
        std::string s(buf);
        if (s.find('.') == std::string::npos && s.find('e') == std::string::npos
            && s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) s += ".0";
        return s;
    }
}

namespace ucuda::equations_detail {
bool div_by_reciprocal(const PN& n) {
        if (n->kind != Node::Div || n->b->kind != Node::Num || n->a->kind == Node::Num) return false;
        const double c = std::fabs(n->b->num);
        return std::isfinite(c) && c >= 1e-30 && c <= 1e30;
    }
}

namespace ucuda::equations_detail {
void emit(const PN& n, const NameMap& nm, std::ostream& o) {
        switch (n->kind) {
        // Литерал всегда через (numb): голый "0.5" — double, и при numb = float всё
        // выражение уходило в FP64 (F2F + DFMA). Отрицательный — в скобках, чтобы
        // остаться атомом. Текст "(numb)0.0" сравнивается в jac_emit (Am = I - hc*J).
        case Node::Num:
            if (std::signbit(n->num)) o << "((numb)" << fmtnum(n->num) << ")";
            else                      o << "(numb)" << fmtnum(n->num);
            break;
        case Node::Sym:o << nm.resolve(n->name); break;
        case Node::Neg:
            o << "-";
            // "- -x" и "-(-x)^2" дают Neg(Neg(..)); без скобок вышло бы "--X[0]" —
            // в C это декремент состояния. То же для отрицательного литерала.
            if (n->a->kind == Node::Neg || (n->a->kind == Node::Num && std::signbit(n->a->num))) {
                o << "("; emit(n->a, nm, o); o << ")";
            }
            else emit_child(n->a, nm, o, prec(Node::Neg));
            break;
        case Node::Add:emit_child(n->a, nm, o, 1); o << " + "; emit_child(n->b, nm, o, 1); break;
        case Node::Sub:emit_child(n->a, nm, o, 1); o << " - "; emit_child(n->b, nm, o, 2); break;
        case Node::Mul:emit_child(n->a, nm, o, 2); o << " * "; emit_child(n->b, nm, o, 2); break;
        case Node::Div:
            emit_child(n->a, nm, o, 2);
            if (div_by_reciprocal(n)) { o << " * ((numb)1 / "; emit_child(n->b, nm, o, 3); o << ")"; }
            else                      { o << " / ";            emit_child(n->b, nm, o, 3); }
            break;
        case Node::Pow: {
            if (n->b->kind == Node::Num) {
                double e = n->b->num; int ie = (int)e;
                if (e == (double)ie && ie >= 2 && ie <= 4) { o << "("; for (int k = 0; k < ie; ++k) { if (k)o << " * "; emit_child(n->a, nm, o, 2); }o << ")"; break; }
            }
            o << "pow("; emit(n->a, nm, o); o << ", "; emit(n->b, nm, o); o << ")"; break;
        }
        case Node::Call:o << n->name << "("; for (size_t k = 0; k < n->args.size(); ++k) { if (k)o << ", "; emit(n->args[k], nm, o); }o << ")"; break;
        }
    }
}

namespace ucuda::equations_detail {
const std::set<std::string>& math_constants() {
        static const std::set<std::string> c = { "pi" };
        return c;
    }
}

namespace ucuda::equations_detail {
NameMap build_namemap(const System& s, const std::string& st, const std::string& tname ) {
        NameMap nm;
        for (size_t i = 0; i < s.vars.size(); ++i) nm.m[s.vars[i]] = st + "[" + std::to_string(i) + "]";
        for (size_t j = 0; j < s.params.size(); ++j) nm.m[s.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
        // константы маппятся сами на себя
        for (const auto& c : math_constants()) nm.m[c] = c;
        if (!tname.empty() && !nm.m.count("t")) nm.m["t"] = tname;
        // Уровень B: z — решение Ньютона текущего вычисления f (ucuda_zb_v), не элемент массива стадии.
        for (size_t k = 0; k < s.zb_names.size(); ++k) nm.m[s.zb_names[k]] = "ucuda_zb_v[" + std::to_string(k) + "]";
        return nm;
    }
}

namespace ucuda::equations_detail {
std::vector<std::string> rhs_over(const System& s, const std::string& st, const std::string& tname ) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        NameMap nm = build_namemap(s, st, tname); std::vector<std::string> out;
        for (size_t i = 0; i < s.rhs.size(); ++i) {
            Parser p(s.rhs[i], s.latex); PN ast = p.parse();
            std::ostringstream o;
            // Уровень B: f над массивом стадии st — сначала z = phi(st) (ucuda_zb, запятая задаёт порядок;
            // повтор с тем же x и t берёт кэш), потом само выражение.
            if (!s.zb_g.empty()) o << "(ucuda_zb(" << st << ", " << (tname.empty() ? "(numb)0" : tname) << "), (";
            emit(ast, nm, o);
            if (!s.zb_g.empty()) o << "))";
            out.push_back(o.str());
        }
        return out;
    }
}

namespace ucuda::equations_detail {
int cd_fp_iters_of(const System& s) {
        return s.cd_fp_iters < 1 ? 1 : s.cd_fp_iters;
    }
}

namespace ucuda::equations_detail {
bool pn_contains_var(const PN& n, const std::string& v) {
        if (!n) return false;
        switch (n->kind) {
        case Node::Sym:  return n->name == v;
        case Node::Num:  return false;
        case Node::Neg:  return pn_contains_var(n->a, v);
        case Node::Add: case Node::Sub:
        case Node::Mul: case Node::Div:
        case Node::Pow:
            return pn_contains_var(n->a, v) || pn_contains_var(n->b, v);
        case Node::Call:
            for (auto& c : n->args) if (pn_contains_var(c, v)) return true;
            return false;
        }
        return false;
    }
}

namespace ucuda::equations_detail {
PN pn_num(double v) { auto n = mk(Node::Num); n->num = v; return n; }
}

namespace ucuda::equations_detail {
bool pn_is_zero(const PN& n) { return n && n->kind == Node::Num && n->num == 0.0; }
}

namespace ucuda::equations_detail {
bool pn_is_one (const PN& n) { return n && n->kind == Node::Num && n->num == 1.0; }
}

namespace ucuda::equations_detail {
PN pn_neg(PN a) {
        if (a && a->kind == Node::Neg) return a->a;                       // -(-x) -> x
        if (a && a->kind == Node::Num) return pn_num(-a->num);            // -(c)  -> (-c)
        auto n = mk(Node::Neg); n->a = std::move(a); return n;
    }
}

namespace ucuda::equations_detail {
PN pn_add(PN a, PN b) { auto n = mk(Node::Add); n->a = std::move(a); n->b = std::move(b); return n; }
}

namespace ucuda::equations_detail {
PN pn_sub(PN a, PN b) { auto n = mk(Node::Sub); n->a = std::move(a); n->b = std::move(b); return n; }
}

namespace ucuda::equations_detail {
bool pn_is_neg_one(const PN& n) {
        if (!n) return false;
        if (n->kind == Node::Num && n->num == -1.0) return true;
        if (n->kind == Node::Neg && pn_is_one(n->a)) return true;
        return false;
    }
}

namespace ucuda::equations_detail {
PN pn_mul(PN a, PN b) {
        if (pn_is_zero(a) || pn_is_zero(b)) return pn_num(0);
        if (pn_is_one(a)) return b;
        if (pn_is_one(b)) return a;
        if (pn_is_neg_one(a)) return pn_neg(std::move(b));                // (-1)*x -> -x
        if (pn_is_neg_one(b)) return pn_neg(std::move(a));                // x*(-1) -> -x
        auto n = mk(Node::Mul); n->a = std::move(a); n->b = std::move(b); return n;
    }
}

namespace ucuda::equations_detail {
PN pn_div(PN a, PN b) { auto n = mk(Node::Div); n->a = std::move(a); n->b = std::move(b); return n; }
}

namespace ucuda::equations_detail {
bool cd_try_extract_linear(const PN& n, const std::string& v, PN& coef, PN& rem) {
        if (!n) { coef = pn_num(0); rem = pn_num(0); return true; }
        switch (n->kind) {
        case Node::Num:
            coef = pn_num(0); rem = n; return true;
        case Node::Sym:
            if (n->name == v) { coef = pn_num(1); rem = pn_num(0); }
            else              { coef = pn_num(0); rem = n; }
            return true;
        case Node::Neg: {
            PN ca, ra;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            coef = pn_is_zero(ca) ? pn_num(0) : pn_neg(ca);
            rem  = pn_is_zero(ra) ? pn_num(0) : pn_neg(ra);
            return true;
        }
        case Node::Add: {
            PN ca, ra, cb, rb;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            if (!cd_try_extract_linear(n->b, v, cb, rb)) return false;
            coef = pn_is_zero(ca) ? cb : (pn_is_zero(cb) ? ca : pn_add(ca, cb));
            rem  = pn_is_zero(ra) ? rb : (pn_is_zero(rb) ? ra : pn_add(ra, rb));
            return true;
        }
        case Node::Sub: {
            PN ca, ra, cb, rb;
            if (!cd_try_extract_linear(n->a, v, ca, ra)) return false;
            if (!cd_try_extract_linear(n->b, v, cb, rb)) return false;
            coef = pn_is_zero(cb) ? ca : (pn_is_zero(ca) ? pn_neg(cb) : pn_sub(ca, cb));
            rem  = pn_is_zero(rb) ? ra : (pn_is_zero(ra) ? pn_neg(rb) : pn_sub(ra, rb));
            return true;
        }
        case Node::Mul: {
            bool la = pn_contains_var(n->a, v);
            bool lb = pn_contains_var(n->b, v);
            if (la && lb) return false;                    // v*v-like coupling
            if (!la && !lb) { coef = pn_num(0); rem = n; return true; }
            PN linear_side = la ? n->a : n->b;
            PN const_side  = la ? n->b : n->a;
            PN ci, ri;
            if (!cd_try_extract_linear(linear_side, v, ci, ri)) return false;
            coef = pn_is_zero(ci) ? pn_num(0) : pn_mul(const_side, ci);
            rem  = pn_is_zero(ri) ? pn_num(0) : pn_mul(const_side, ri);
            return true;
        }
        case Node::Div: {
            if (pn_contains_var(n->b, v)) return false;    // v in denominator
            PN ci, ri;
            if (!cd_try_extract_linear(n->a, v, ci, ri)) return false;
            coef = pn_is_zero(ci) ? pn_num(0) : pn_div(ci, n->b);
            rem  = pn_is_zero(ri) ? pn_num(0) : pn_div(ri, n->b);
            return true;
        }
        case Node::Pow:
            if (pn_contains_var(n->a, v) || pn_contains_var(n->b, v)) return false;
            coef = pn_num(0); rem = n; return true;
        case Node::Call:
            for (auto& c : n->args) if (pn_contains_var(c, v)) return false;
            coef = pn_num(0); rem = n; return true;
        }
        return false;
    }
}

namespace ucuda::equations_detail {
const std::string* cd_find_complex_unsafe(const PN& n) {
        if (!n) return nullptr;
        if (n->kind == Node::Call && (n->name == "fmod" || n->name == "atan2" || n->name == "fmin" || n->name == "fmax"))
            return &n->name;
        if (const std::string* f = cd_find_complex_unsafe(n->a)) return f;
        if (const std::string* f = cd_find_complex_unsafe(n->b)) return f;
        for (const auto& c : n->args)
            if (const std::string* f = cd_find_complex_unsafe(c)) return f;
        return nullptr;
    }
}

namespace ucuda::equations_detail {
void cd_check_complex_safe(const PN& n) {
        if (const std::string* f = cd_find_complex_unsafe(n))
            throw std::runtime_error("Complex CD: function " + *f +
                " is not defined in complex arithmetic - pick another scheme");
    }
}

namespace ucuda::equations_detail {
PN pn_call1(const char* f, PN u) {
        auto n = mk(Node::Call); n->name = f; n->args.push_back(std::move(u)); return n;
    }
}

namespace ucuda::equations_detail {
PN pn_call2(const char* f, PN u, PN w) {
        auto n = mk(Node::Call); n->name = f;
        n->args.push_back(std::move(u)); n->args.push_back(std::move(w)); return n;
    }
}

namespace ucuda::equations_detail {
PN pn_pow(PN a, PN b) {
        // a^0 -> 1 and a^1 -> a, so the power rule never leaves pow(x, 1.0) behind
        if (b && b->kind == Node::Num) {
            if (b->num == 0.0) return pn_num(1);
            if (b->num == 1.0) return a;
        }
        auto n = mk(Node::Pow); n->a = std::move(a); n->b = std::move(b); return n;
    }
}

namespace ucuda::equations_detail {
PN jd_add(PN a, PN b) {
        if (pn_is_zero(a)) return b;
        if (pn_is_zero(b)) return a;
        return pn_add(std::move(a), std::move(b));
    }
}

namespace ucuda::equations_detail {
PN jd_sub(PN a, PN b) {
        if (pn_is_zero(b)) return a;
        if (pn_is_zero(a)) return pn_neg(std::move(b));
        return pn_sub(std::move(a), std::move(b));
    }
}

namespace ucuda::equations_detail {
PN pn_diff_call(const PN& n, const std::string& v) {
        const std::string& f = n->name;
        // pow / atan2 / fmod are the only binary calls the parser produces
        if (f == "pow") {
            if (n->args.size() != 2) throw std::runtime_error("jacobian: pow needs 2 args");
            return pn_diff(pn_pow(n->args[0], n->args[1]), v);
        }
        if (f == "fmin" || f == "fmax") {
            // ветвь выбирает знак u - w: s = copysign(1, u - w); fmin = u при s < 0
            if (n->args.size() != 2) throw std::runtime_error("jacobian: " + f + " needs 2 args");
            const PN& u = n->args[0]; const PN& w = n->args[1];
            const PN du = pn_diff(u, v), dw = pn_diff(w, v);
            if (pn_is_zero(du) && pn_is_zero(dw)) return pn_num(0);
            const PN s = pn_call2("copysign", pn_num(1), pn_sub(u, w));
            const PN pu = pn_mul(pn_num(0.5), f == "fmin" ? pn_sub(pn_num(1), s) : pn_add(pn_num(1), s));
            const PN pw = pn_mul(pn_num(0.5), f == "fmin" ? pn_add(pn_num(1), s) : pn_sub(pn_num(1), s));
            if (pn_is_zero(dw)) return pn_mul(pu, du);
            if (pn_is_zero(du)) return pn_mul(pw, dw);
            return pn_add(pn_mul(pu, du), pn_mul(pw, dw));
        }
        if (f == "copysign") {
            // |a| sgn(b): по a — sgn(a) sgn(b), по b — 0 почти всюду (ступеньки условий импорта схем)
            if (n->args.size() != 2) throw std::runtime_error("jacobian: copysign needs 2 args");
            const PN da = pn_diff(n->args[0], v);
            if (pn_is_zero(da)) return pn_num(0);
            return pn_mul(pn_mul(pn_call2("copysign", pn_num(1), n->args[0]), pn_call2("copysign", pn_num(1), n->args[1])), da);
        }
        if (f == "atan2") {
            if (n->args.size() != 2) throw std::runtime_error("jacobian: atan2 needs 2 args");
            const PN& u = n->args[0]; const PN& w = n->args[1];
            PN num = jd_sub(pn_mul(w, pn_diff(u, v)), pn_mul(u, pn_diff(w, v)));
            if (pn_is_zero(num)) return pn_num(0);
            return pn_div(num, pn_add(pn_mul(u, u), pn_mul(w, w)));
        }
        if (n->args.size() != 1) throw std::runtime_error("jacobian: " + f + " needs 1 arg");
        const PN& u = n->args[0];
        PN du = pn_diff(u, v);
        if (pn_is_zero(du)) return pn_num(0);

        PN g;  // outer derivative, d f / d u
        if      (f == "sin")   g = pn_call1("cos", u);
        else if (f == "cos")   g = pn_neg(pn_call1("sin", u));
        else if (f == "tan")   g = pn_div(pn_num(1), pn_mul(pn_call1("cos", u), pn_call1("cos", u)));
        else if (f == "asin")  g = pn_div(pn_num(1), pn_call1("sqrt", pn_sub(pn_num(1), pn_mul(u, u))));
        else if (f == "acos")  g = pn_neg(pn_div(pn_num(1), pn_call1("sqrt", pn_sub(pn_num(1), pn_mul(u, u)))));
        else if (f == "atan")  g = pn_div(pn_num(1), pn_add(pn_num(1), pn_mul(u, u)));
        else if (f == "sinh")  g = pn_call1("cosh", u);
        else if (f == "cosh")  g = pn_call1("sinh", u);
        else if (f == "tanh")  g = pn_sub(pn_num(1), pn_mul(pn_call1("tanh", u), pn_call1("tanh", u)));
        else if (f == "exp")   g = pn_call1("exp", u);
        else if (f == "log")   g = pn_div(pn_num(1), u);
        else if (f == "log2")  g = pn_div(pn_num(1), pn_mul(u, pn_num(JD_LN2)));
        else if (f == "log10") g = pn_div(pn_num(1), pn_mul(u, pn_num(JD_LN10)));
        else if (f == "sqrt")  g = pn_div(pn_num(1), pn_mul(pn_num(2), pn_call1("sqrt", u)));
        else if (f == "cbrt")  g = pn_div(pn_num(1), pn_mul(pn_num(3), pn_mul(pn_call1("cbrt", u), pn_call1("cbrt", u))));
        // sysparse rewrites every |...| as fabs(), so the Chua-family piecewise systems
        // in this project hang on this line. copysign rather than the device-side sign():
        // it is a builtin for nvcc, NVRTC and MSVC alike, so the same emitted text
        // compiles on every codegen consumer with no extra header. At u == 0 fabs has no
        // derivative anyway -- copysign(1,-0.0) is -1 there, sign(0) would be 0.
        else if (f == "fabs" || f == "abs") g = pn_call2("copysign", pn_num(1), u);
        else throw std::runtime_error("jacobian: no derivative for function " + f);

        return pn_mul(g, du);
    }
}

namespace ucuda::equations_detail {
PN pn_diff(const PN& n, const std::string& v) {
        if (!n) return pn_num(0);
        switch (n->kind) {
        case Node::Num: return pn_num(0);
        // params are Sym as well, and correctly differentiate to 0
        case Node::Sym: return pn_num(n->name == v ? 1.0 : 0.0);
        case Node::Neg: {
            PN d = pn_diff(n->a, v);
            return pn_is_zero(d) ? pn_num(0) : pn_neg(d);
        }
        case Node::Add: return jd_add(pn_diff(n->a, v), pn_diff(n->b, v));
        case Node::Sub: return jd_sub(pn_diff(n->a, v), pn_diff(n->b, v));
        case Node::Mul: return jd_add(pn_mul(pn_diff(n->a, v), n->b),
                                      pn_mul(n->a, pn_diff(n->b, v)));
        case Node::Div: {
            PN da = pn_diff(n->a, v), db = pn_diff(n->b, v);
            // v-free denominator is the common case; skip the quotient rule there
            if (pn_is_zero(db)) return pn_is_zero(da) ? pn_num(0) : pn_div(da, n->b);
            PN num = jd_sub(pn_mul(da, n->b), pn_mul(n->a, db));
            if (pn_is_zero(num)) return pn_num(0);
            return pn_div(num, pn_mul(n->b, n->b));
        }
        case Node::Pow: {
            PN da = pn_diff(n->a, v), db = pn_diff(n->b, v);
            if (pn_is_zero(da) && pn_is_zero(db)) return pn_num(0);
            if (pn_is_zero(db)) {
                // Power rule whenever the exponent is v-free (literal OR parameter).
                // The general form below needs log(a), undefined for a < 0, and
                // negative bases are ordinary here.
                PN em1 = (n->b->kind == Node::Num) ? pn_num(n->b->num - 1.0)
                                                   : pn_sub(n->b, pn_num(1));
                return pn_mul(pn_mul(n->b, pn_pow(n->a, em1)), da);
            }
            // general: a^b * (db*log(a) + b*da/a)
            PN t1 = pn_mul(db, pn_call1("log", n->a));
            PN t2 = pn_is_zero(da) ? pn_num(0) : pn_div(pn_mul(n->b, da), n->a);
            return pn_mul(pn_pow(n->a, n->b), jd_add(t1, t2));
        }
        case Node::Call: return pn_diff_call(n, v);
        }
        return pn_num(0);
    }
}

namespace ucuda::equations_detail {
void jac_check_differentiable(const PN& n) {
        if (!n) return;
        if (n->kind == Node::Call &&
            (n->name == "fmod" || n->name == "floor" || n->name == "ceil"))
            throw std::runtime_error("Implicit scheme: function " + n->name +
                " is not differentiable, no Jacobian can be built -- pick an explicit scheme");
        jac_check_differentiable(n->a);
        jac_check_differentiable(n->b);
        for (const auto& c : n->args) jac_check_differentiable(c);
    }
}

namespace ucuda::equations_detail {
std::vector<std::string> jac_over(const System& s, const std::string& st, const std::string& tname ) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        NameMap nm = build_namemap(s, st, tname);
        std::vector<std::string> out((size_t)N * N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            jac_check_differentiable(ast);
            for (int j = 0; j < N; ++j)
                out[(size_t)i * N + j] = emit_to_str(pn_diff(ast, s.vars[j]), nm);
        }
        return out;
    }
}

namespace ucuda::equations_detail {
std::string md_literal(const MultiDoubleCoef& r) {
        if (r.c[0] == 0.0) return "0";
        std::string t;
        for (int m = 0; m < 4 && r.c[m] != 0.0; ++m) {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", r.c[m]);
            t += (m ? " + (numb)(" : "(numb)(") + std::string(buf) + ")";
        }
        return t;
    }
}

namespace ucuda::equations_detail {
const char* md_table_qual() {
        return sizeof(numb) < sizeof(double) ? "    const numb " : "    static const numb ";
    }
}

namespace ucuda::equations_detail {
void md_table(std::ostringstream& o, const char* decl, const MultiDoubleCoef* v,
                  int rows, int cols, int ld) {
        o << md_table_qual() << decl << " = {\n";
        for (int i = 0; i < rows; ++i) {
            o << "        {";
            for (int j = 0; j < cols; ++j) o << (j ? ", " : "") << md_literal(v[i * ld + j]);
            o << "}" << (i + 1 < rows ? "," : "") << "\n";
        }
        o << "    };\n";
    }
}

namespace ucuda::equations_detail {
void stage_time_table(std::ostringstream& o, const MultiDoubleCoef* A, int S, int ld) {
        o << md_table_qual() << "Cs[" << S << "] = { ";
        for (int i = 0; i < S; ++i) {
            long double c = 0;
            for (int j = 0; j < i && j < ld; ++j)
                for (int m = 0; m < 4; ++m) c += (long double)A[i * ld + j].c[m];
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", (double)c);
            o << (i ? ", " : "") << "(numb)(" << buf << ")";
        }
        o << " };\n";
    }
}

namespace ucuda::equations_detail {
std::string adaptive_emb_body(const System& s, const EmbSpec& e) {
        const int N = (int)s.vars.size(), S = e.S;
        // Явное время: стадия i — в момент ucuda_t + Cs[i]*h, f(Y) — в ucuda_t + h.
        const std::string tn = time_name(s, "ucuda_ts");
        const auto kx = rhs_over(s, "X1", tn), ky = rhs_over(s, "Y", time_name(s, "(ucuda_t + h)"));
        std::ostringstream o;
        md_table(o, ("M[" + std::to_string(S) + "][" + std::to_string(S - 1) + "]").c_str(), e.A, S, S - 1, e.ld);
        md_table(o, ("B[1][" + std::to_string(S) + "]").c_str(), e.b, 1, S, S);
        if (!tn.empty()) stage_time_table(o, e.A, S, e.ld);
        o << md_table_qual() << "L[" << e.nlo << "][" << e.lc << "] = {\n";
        for (int m = 0; m < e.nlo; ++m) {
            o << "        {";
            for (int i = 0; i < e.lc; ++i) o << (i ? ", " : "") << md_literal(e.lo[m][i]);
            o << "}" << (m + 1 < e.nlo ? "," : "") << "\n";
        }
        o << "    };\n";
        auto nz = [](const MultiDoubleCoef& c) { return c.c[0] != 0.0; };
        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int l;\n";
        if (!tn.empty()) o << "    numb ucuda_ts;\n";
        o << "    for (l = 0; l < N; ++l) W[l] = F0[l];\n";
        for (int i = 1; i < S; ++i) {
            o << "    for (l = 0; l < N; ++l) {\n";
            o << "        X2[l] = 0;\n";
            for (int j = 0; j < i; ++j)
                if (nz(e.A[i * e.ld + j]))
                    o << "        X2[l] += M[" << i << "][" << j << "] * W[" << j << " * N + l];\n";
            o << "        X1[l] = X[l] + h * X2[l];\n";
            o << "    }\n";
            if (!tn.empty()) o << "    ucuda_ts = ucuda_t + Cs[" << i << "] * h;\n";
            for (int v = 0; v < N; ++v)
                o << "    W[" << i << " * N + " << v << "] = (" << kx[v] << ");\n";
        }
        o << "    for (l = 0; l < N; ++l) {\n";
        o << "        X2[l] = 0;\n";
        for (int i = 0; i < S; ++i)
            if (nz(e.b[i])) o << "        X2[l] += B[0][" << i << "] * W[" << i << " * N + l];\n";
        o << "        Y[l] = X[l] + h * X2[l];\n";
        o << "    }\n";
        for (int v = 0; v < N; ++v)
            o << "    F1[" << v << "] = (" << ky[v] << ");\n";
        o << "    for (l = 0; l < N; ++l) W[" << S << " * N + l] = F1[l];\n";
        for (int m = 0; m < e.nlo; ++m) {
            o << "    for (l = 0; l < N; ++l) {\n";
            o << "        X2[l] = 0;\n";
            for (int i = 0; i < e.lc; ++i)
                if (nz(e.lo[m][i])) o << "        X2[l] += L[" << m << "][" << i << "] * W[" << i << " * N + l];\n";
            o << "        E[" << m << " * N + l] = h * X2[l];\n";
            o << "    }\n";
        }
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string adaptive_rhs_body(const System& s) {
        const auto k = rhs_over(s, "X", time_name(s));   // f в момент аргумента ucuda_t
        std::ostringstream o;
        for (size_t v = 0; v < k.size(); ++v) o << "    F[" << v << "] = (" << k[v] << ");\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string rk45_dprep_body(const System& s) {
        const int N = (int)s.vars.size();
        std::ostringstream o;
        md_table(o, "P[7][4]", &kRK45P[0][0], 7, 4, 4);
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";
        o << "    (void)Y; (void)F0; (void)F1; (void)a;\n";
        o << "    for (l = 0; l < N; ++l) D[l] = X[l];\n";
        o << "    for (j = 0; j < 4; ++j)\n";
        o << "        for (l = 0; l < N; ++l) {\n";
        o << "            numb acc = 0;\n";
        o << "            for (i = 0; i < 7; ++i) acc += W[i * N + l] * P[i][j];\n";
        o << "            D[(1 + j) * N + l] = h * acc;\n";
        o << "        }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string rk45_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l)\n";
        o << "        Yo[l] = D[l] + th * (D[N + l] + th * (D[2 * N + l] + th * (D[3 * N + l] + th * D[4 * N + l])));\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dop853_dprep_body(const System& s) {
        const int N = (int)s.vars.size();
        const std::string tn = time_name(s, "ucuda_ts");
        const auto kx = rhs_over(s, "X1", tn);
        std::ostringstream o;
        md_table(o, "AE[3][15]", &kDop853AExt[0][0], 3, 15, 15);
        md_table(o, "DD[4][16]", &kDop853D[0][0], 4, 16, 16);
        if (!tn.empty()) {
            // Узлы доп. стадий 13..15 — суммы строк AE (у стадии 13 + i — первые 13 + i).
            o << md_table_qual() << "CE[3] = { ";
            for (int i = 0; i < 3; ++i) {
                long double c = 0;
                for (int j = 0; j < 13 + i; ++j)
                    for (int m = 0; m < 4; ++m) c += (long double)kDop853AExt[i][j].c[m];
                char buf[40];
                std::snprintf(buf, sizeof(buf), "%.17g", (double)c);
                o << (i ? ", " : "") << "(numb)(" << buf << ")";
            }
            o << " };\n";
        }
        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l, r;\n";
        o << "    for (i = 0; i < 3; ++i) {\n";
        o << "        for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "        for (j = 0; j < 13 + i; ++j)\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X2[l] += AE[i][j] * W[j * N + l];\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X1[l] = X[l] + h * X2[l];\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + CE[i] * h;\n";
        for (int v = 0; v < N; ++v)
            o << "        W[(13 + i) * N + " << v << "] = (" << kx[v] << ");\n";
        o << "    }\n";
        o << "    for (l = 0; l < N; ++l) {\n";
        o << "        numb dy = Y[l] - X[l];\n";
        o << "        D[l] = X[l];\n";
        o << "        D[N + l] = dy;\n";
        o << "        D[2 * N + l] = h * F0[l] - dy;\n";
        o << "        D[3 * N + l] = 2 * dy - h * (F1[l] + F0[l]);\n";
        o << "    }\n";
        o << "    for (r = 0; r < 4; ++r)\n";
        o << "        for (l = 0; l < N; ++l) {\n";
        o << "            numb acc = 0;\n";
        o << "            for (i = 0; i < 16; ++i) acc += DD[r][i] * W[i * N + l];\n";
        o << "            D[(4 + r) * N + l] = h * acc;\n";
        o << "        }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dop853_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    numb u = 1 - th;\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        numb y = D[7 * N + l] * th;\n";
        o << "        y = (y + D[6 * N + l]) * u;\n";
        o << "        y = (y + D[5 * N + l]) * th;\n";
        o << "        y = (y + D[4 * N + l]) * u;\n";
        o << "        y = (y + D[3 * N + l]) * th;\n";
        o << "        y = (y + D[2 * N + l]) * u;\n";
        o << "        y = (y + D[N + l]) * th;\n";
        o << "        Yo[l] = y + D[l];\n";
        o << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string hermite_dprep_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    (void)a; (void)W;\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        D[l] = X[l];\n";
        o << "        D[N + l] = Y[l];\n";
        o << "        D[2 * N + l] = h * F0[l];\n";
        o << "        D[3 * N + l] = h * F1[l];\n";
        o << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string hermite_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        numb d = D[N + l] - D[l], f0 = D[2 * N + l], f1 = D[3 * N + l];\n";
        o << "        Yo[l] = D[l] + th * (f0 + th * ((3 * d - 2 * f0 - f1) + th * (f0 + f1 - 2 * d)));\n";
        o << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
bool sparse_lu_build(const std::vector<MatEntry>& pat, int n, const std::vector<char>& df, SparseLu& S) {
        // UCUDA_DAE_DENSE_LU=1 — плотная LU, как до этапа 6 (сравнение скорости и результатов).
        static const bool dense_forced = [] { const char* e = std::getenv("UCUDA_DAE_DENSE_LU"); return e && *e == '1'; }();
        if (dense_forced) return false;
        S = SparseLu{};
        for (const MatEntry& e : pat) S.orig.insert({ e.row, e.col });
        for (int i = 0; i < n; ++i) {
            if (df[(size_t)i]) ++S.n1;
            else { S.Rb.push_back(i); S.Cb.push_back(i); }
        }
        S.nb = (int)S.Rb.size();
        std::vector<MatEntry> full = pat;
        for (int r : S.Rb) for (int c : S.Cb) if (!S.orig.count({ r, c })) full.push_back(MatEntry{ r, c });
        std::string err;
        if (!plan_elimination(full, n, S.plan, &err, &df)) return false;
        for (int k = 0; k < S.n1; ++k)   // первая фаза — ровно диагонали строк df
            if (S.plan.steps[(size_t)k].pivot_row != S.plan.steps[(size_t)k].pivot_col || !df[(size_t)S.plan.steps[(size_t)k].pivot_row])
                return false;
        // Модель стоимости: умножения-сложения первой фазы (по плану) + nb^3/3 блока против n^3/3 плотной;
        // разреженная — только при выигрыше хотя бы вдвое. И первая фаза не длиннее 4000 операций: это
        // прямолинейный код, и его объём стоит дороже арифметики. Andrews, Radau + GGL: 7884 операции первой
        // фазы + блок 57 x 57 — по модели в 4.6 раза дешевле плотной 99 x 99, на деле в 1.4 раза медленнее
        // (CPU, cl /O2); Radau без GGL (909 операций, блок 39) — в 3.9 раза быстрее.
        {
            for (int k = 0; k < S.n1; ++k)
                S.ops1 += (double)S.plan.steps[(size_t)k].rows.size() * (double)S.plan.steps[(size_t)k].cols.size();
            if (S.ops1 > 4000.0) return false;
            if (2.0 * (S.ops1 + (double)S.nb * S.nb * S.nb / 3.0) > (double)n * n * n / 3.0) return false;
        }
        for (int a = 0; a < S.nb; ++a)
            for (int b = 0; b < S.nb; ++b) S.id[{ S.Rb[(size_t)a], S.Cb[(size_t)b] }] = a * S.nb + b;
        for (const MatEntry& e : S.plan.filled)
            if (!S.id.count({ e.row, e.col })) S.id[{ e.row, e.col }] = (int)S.id.size();
        S.nnz = (int)S.id.size();
        return true;
    }
}

namespace ucuda::equations_detail {
void sparse_lu_factor(std::ostream& o, const SparseLu& S, const char* ind, const char* fallback) {
        for (int k = 0; k < S.n1; ++k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            o << ind << "if (!" << fallback << ") { dgn = " << S.at(st.pivot_row, st.pivot_col) << "; mx = fabs(dgn) * (numb)1e3;";
            for (int r : st.rows) o << " av = fabs(" << S.at(r, st.pivot_col) << "); if (av > mx) mx = av;";
            o << " if (fabs(dgn) < (numb)1e-30 || fabs(dgn) * (numb)1e3 < mx) " << fallback << " = 1;";
            if (!st.rows.empty()) {
                o << " else {\n";
                for (int r : st.rows) {
                    o << ind << "    mlt = " << S.at(r, st.pivot_col) << " / dgn; " << S.at(r, st.pivot_col) << " = mlt;";
                    for (int c : st.cols) o << " " << S.at(r, c) << " -= mlt * " << S.at(st.pivot_row, c) << ";";
                    o << "\n";
                }
                o << ind << "} }\n";
            } else o << " }\n";
        }
        if (S.nb == 0) return;
        const std::string nb = std::to_string(S.nb);
        o << ind << "if (!" << fallback << ") {   /* dense block " << nb << " x " << nb << ", partial pivoting */\n";
        o << ind << "    for (ik = 0; ik < " << nb << "; ik++) {\n";
        o << ind << "        ip = ik; mx = fabs(Asp[ik * " << nb << " + ik]);\n";
        o << ind << "        for (ir = ik + 1; ir < " << nb << "; ir++) { av = fabs(Asp[ir * " << nb << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n";
        o << ind << "        ucuda_pv[ik] = ip;\n";
        o << ind << "        if (ip != ik) for (ic = ik; ic < " << nb << "; ic++) { mlt = Asp[ik * " << nb << " + ic]; Asp[ik * " << nb
          << " + ic] = Asp[ip * " << nb << " + ic]; Asp[ip * " << nb << " + ic] = mlt; }\n";
        o << ind << "        dgn = Asp[ik * " << nb << " + ik];\n";
        o << ind << "        if (fabs(dgn) < (numb)1e-30) { sing = 1; break; }\n";
        o << ind << "        for (ir = ik + 1; ir < " << nb << "; ir++) {\n";
        o << ind << "            mlt = Asp[ir * " << nb << " + ik] / dgn; Asp[ir * " << nb << " + ik] = mlt;\n";
        o << ind << "            for (ic = ik + 1; ic < " << nb << "; ic++) Asp[ir * " << nb << " + ic] -= mlt * Asp[ik * " << nb << " + ic];\n";
        o << ind << "        }\n";
        o << ind << "    }\n";
        o << ind << "}\n";
    }
}

namespace ucuda::equations_detail {
void sparse_lu_decl(std::ostream& o, const SparseLu& S, int nsol) {
        o << "    numb Asp[" << S.nnz << "], Dv[" << nsol << "];   /* sparse LU: " << S.n1 << " pivots by plan ("
          << (long long)S.ops1 << " multiply-adds), dense block " << S.nb << " x " << S.nb << "; dense fallback */\n";
        o << "    int  ucuda_dn = 0, ucuda_pv[" << (S.nb > 0 ? S.nb : 1) << "];\n";
        o << "    const int ucuda_rb[" << (S.nb > 0 ? S.nb : 1) << "] = { ";
        for (int a = 0; a < S.nb; ++a) o << (a ? ", " : "") << S.Rb[(size_t)a];
        if (S.nb == 0) o << "0";
        o << " };\n";
    }
}

namespace ucuda::equations_detail {
void sparse_lu_solve(std::ostream& o, const SparseLu& S, const std::string& rhs, const std::string& sol, const char* ind) {
        for (int k = 0; k < S.n1; ++k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            for (int r : st.rows)
                o << ind << rhs << "[" << r << "] -= " << S.at(r, st.pivot_col) << " * " << rhs << "[" << st.pivot_row << "];\n";
        }
        if (S.nb > 0) {
            const std::string nb = std::to_string(S.nb);
            o << ind << "for (ik = 0; ik < " << nb << "; ik++) {\n";
            o << ind << "    ip = ucuda_pv[ik]; if (ip != ik) { mlt = " << rhs << "[ucuda_rb[ik]]; " << rhs << "[ucuda_rb[ik]] = " << rhs
              << "[ucuda_rb[ip]]; " << rhs << "[ucuda_rb[ip]] = mlt; }\n";
            o << ind << "    for (ir = ik + 1; ir < " << nb << "; ir++) " << rhs << "[ucuda_rb[ir]] -= Asp[ir * " << nb << " + ik] * "
              << rhs << "[ucuda_rb[ik]];\n";
            o << ind << "}\n";
            o << ind << "for (ik = " << S.nb - 1 << "; ik >= 0; ik--) {\n";
            o << ind << "    mlt = " << rhs << "[ucuda_rb[ik]];\n";
            o << ind << "    for (ic = ik + 1; ic < " << nb << "; ic++) mlt -= Asp[ik * " << nb << " + ic] * " << sol << "[ucuda_rb[ic]];\n";
            o << ind << "    " << sol << "[ucuda_rb[ik]] = mlt / Asp[ik * " << nb << " + ik];\n";
            o << ind << "}\n";
        }
        for (int k = S.n1 - 1; k >= 0; --k) {
            const EliminationStep& st = S.plan.steps[(size_t)k];
            o << ind << sol << "[" << st.pivot_col << "] = (" << rhs << "[" << st.pivot_row << "]";
            for (int c : st.cols) o << " - " << S.at(st.pivot_row, c) << " * " << sol << "[" << c << "]";
            o << ") / " << S.at(st.pivot_row, st.pivot_col) << ";\n";
        }
    }
}

namespace ucuda::equations_detail {
std::string dae_norm_w(const System& s, int i) {
        const int k = i < (int)s.dae_index.size() ? s.dae_index[(size_t)i] : 1;
        std::string w;
        for (int p = 1; p < k; ++p) w += "h * ";
        return w;
    }
}

namespace ucuda::equations_detail {
bool needs_paren_after_mul(const PN& n) {
        return n && (n->kind == Node::Add || n->kind == Node::Sub);
    }
}

namespace ucuda::equations_detail {
std::string wrap_paren(const std::string& s, bool w) {
        return w ? "(" + s + ")" : s;
    }
}

namespace ucuda::equations_detail {
char peel_sign(PN& n, char pos, char neg) {
        if (!n) return pos;
        if (n->kind == Node::Neg) { n = n->a; return neg; }
        if (n->kind == Node::Num && n->num < 0.0) { n = pn_num(-n->num); return neg; }
        return pos;
    }
}

namespace ucuda::equations_detail {
std::string mul_step(const PN& factor, const std::string& factor_c, const char* hs) {
        return pn_is_one(factor) ? std::string(hs) : hs + (" * " + factor_c);
    }
}

namespace ucuda::equations_detail {
bool pn_free_of_vars(const PN& n, const std::vector<std::string>& vars) {
        for (const std::string& v : vars) if (pn_contains_var(n, v)) return false;
        return true;
    }
}

namespace ucuda::equations_detail {
PN pn_mul_step(const PN& e) {
        if (e && (e->kind == Node::Mul || e->kind == Node::Div)) {
            PN n = mk(e->kind);
            n->a = pn_mul_step(e->a);
            n->b = e->b;
            return n;
        }
        PN step = mk(Node::Sym); step->name = kStepSym;
        return pn_mul(std::move(step), e);
    }
}

namespace ucuda::equations_detail {
int func_id(const std::string& nm) {
        if (nm == "sin")return F_SIN; if (nm == "cos")return F_COS; if (nm == "tan")return F_TAN;
        if (nm == "asin")return F_ASIN; if (nm == "acos")return F_ACOS; if (nm == "atan")return F_ATAN;
        if (nm == "sinh")return F_SINH; if (nm == "cosh")return F_COSH; if (nm == "tanh")return F_TANH;
        if (nm == "exp")return F_EXP; if (nm == "log")return F_LOG; if (nm == "log2")return F_LOG2;
        if (nm == "log10")return F_LOG10; if (nm == "sqrt")return F_SQRT; if (nm == "cbrt")return F_CBRT;
        if (nm == "fabs" || nm == "abs")return F_FABS;
        if (nm == "floor")return F_FLOOR; if (nm == "ceil")return F_CEIL;
        if (nm == "pow")return F_POW; if (nm == "atan2")return F_ATAN2; if (nm == "fmod")return F_FMOD;
        if (nm == "copysign")return F_COPYSIGN; if (nm == "fmin")return F_FMIN; if (nm == "fmax")return F_FMAX;
        throw std::runtime_error("eval: unknown function " + nm);
    }
}

namespace ucuda::equations_detail {
double apply_func1(int fid, double x) {
        switch (fid) {
        case F_SIN:return std::sin(x); case F_COS:return std::cos(x); case F_TAN:return std::tan(x);
        case F_ASIN:return std::asin(x); case F_ACOS:return std::acos(x); case F_ATAN:return std::atan(x);
        case F_SINH:return std::sinh(x); case F_COSH:return std::cosh(x); case F_TANH:return std::tanh(x);
        case F_EXP:return std::exp(x); case F_LOG:return std::log(x); case F_LOG2:return std::log2(x);
        case F_LOG10:return std::log10(x); case F_SQRT:return std::sqrt(x); case F_CBRT:return std::cbrt(x);
        case F_FABS:return std::fabs(x);
        case F_FLOOR:return std::floor(x); case F_CEIL:return std::ceil(x);
        } return 0;
    }
}

namespace ucuda::equations_detail {
double apply_func2(int fid, double a, double b) {
        switch (fid) {
        case F_POW:return std::pow(a, b); case F_ATAN2:return std::atan2(a, b); case F_FMOD:return std::fmod(a, b);
        case F_COPYSIGN:return std::copysign(a, b);
        case F_FMIN:    return std::fmin(a, b);
        case F_FMAX:    return std::fmax(a, b);
        } return 0;
    }
}

namespace ucuda::equations_detail {
ucmplx apply_func1(int fid, ucmplx x) {
        switch (fid) {
        case F_SIN:return sin(x); case F_COS:return cos(x); case F_TAN:return tan(x);
        case F_ASIN:return asin(x); case F_ACOS:return acos(x); case F_ATAN:return atan(x);
        case F_SINH:return sinh(x); case F_COSH:return cosh(x); case F_TANH:return tanh(x);
        case F_EXP:return exp(x); case F_LOG:return log(x); case F_LOG2:return log2(x);
        case F_LOG10:return log10(x); case F_SQRT:return sqrt(x); case F_CBRT:return cbrt(x);
        case F_FABS:return fabs(x);
        case F_FLOOR:return floor(x); case F_CEIL:return ceil(x);
        } return ucmplx(0, 0);
    }
}

namespace ucuda::equations_detail {
ucmplx apply_func2(int fid, ucmplx a, ucmplx b) {
        switch (fid) {
        case F_POW:return pow(a, b);
        case F_COPYSIGN:return copysign(a.re, b);
        } return ucmplx(0, 0);
    }
}

namespace ucuda::equations_detail {
double  op_pow(double a, double b) { return std::pow(a, b); }
}

namespace ucuda::equations_detail {
ucmplx  op_pow(ucmplx a, ucmplx b) { return pow(a, b); }
}
