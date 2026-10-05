#include "expression_internal.h"

namespace ucuda::equations_detail {

    PN mk(Node::Kind k) { auto n = std::make_shared<Node>(); n->kind = k; return n; }


    const std::set<std::string>& known_funcs() {
        static const std::set<std::string> f = {
            "sin","cos","tan","asin","acos","atan","sinh","cosh","tanh",
            "exp","log","log2","log10","sqrt","cbrt","fabs","abs",
            "pow","atan2","fmod","floor","ceil"
        };
        return f;
    }

    const std::map<std::string, std::string>& latex_funcs() {
        static const std::map<std::string, std::string> m = {
            {"sin","sin"},{"cos","cos"},{"tan","tan"},{"arcsin","asin"},
            {"arccos","acos"},{"arctan","atan"},{"sinh","sinh"},{"cosh","cosh"},
            {"tanh","tanh"},{"exp","exp"},{"ln","log"},{"log","log"},{"sqrt","sqrt"}
        };
        return m;
    }

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


    int prec(Node::Kind k) {
        switch (k) {
        case Node::Add: case Node::Sub: return 1;
        case Node::Mul: case Node::Div: return 2;
        case Node::Neg: return 3; case Node::Pow: return 4; default: return 5;
        }
    }

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

    void emit(const PN& n, const NameMap& nm, std::ostream& o) {
        switch (n->kind) {
        case Node::Num:o << fmtnum(n->num); break;
        case Node::Sym:o << nm.resolve(n->name); break;
        case Node::Neg:o << "-"; emit_child(n->a, nm, o, prec(Node::Neg)); break;
        case Node::Add:emit_child(n->a, nm, o, 1); o << " + "; emit_child(n->b, nm, o, 1); break;
        case Node::Sub:emit_child(n->a, nm, o, 1); o << " - "; emit_child(n->b, nm, o, 2); break;
        case Node::Mul:emit_child(n->a, nm, o, 2); o << " * "; emit_child(n->b, nm, o, 2); break;
        case Node::Div:emit_child(n->a, nm, o, 2); o << " / "; emit_child(n->b, nm, o, 3); break;
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


    // Зарезервированные математические константы: остаются именем в коде,
    // не идут в a[]. Определяются в шапке ядра отдельно (напр. const double pi = M_PI;).
    const std::set<std::string>& math_constants() {
        static const std::set<std::string> c = { "pi" };
        return c;
    }

    NameMap build_namemap(const System& s, const std::string& st) {
        NameMap nm;
        for (size_t i = 0; i < s.vars.size(); ++i) nm.m[s.vars[i]] = st + "[" + std::to_string(i) + "]";
        for (size_t j = 0; j < s.params.size(); ++j) nm.m[s.params[j]] = "a[" + std::to_string(1 + (int)j) + "]";
        // константы маппятся сами на себя
        for (const auto& c : math_constants()) nm.m[c] = c;
        return nm;
    }

    std::vector<std::string> rhs_over(const System& s, const std::string& st) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        NameMap nm = build_namemap(s, st); std::vector<std::string> out;
        for (size_t i = 0; i < s.rhs.size(); ++i) {
            Parser p(s.rhs[i], s.latex); PN ast = p.parse();
            std::ostringstream o; emit(ast, nm, o); out.push_back(o.str());
        }
        return out;
    }


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


    PN pn_num(double v) { auto n = mk(Node::Num); n->num = v; return n; }

    bool pn_is_zero(const PN& n) { return n && n->kind == Node::Num && n->num == 0.0; }

    bool pn_is_one (const PN& n) { return n && n->kind == Node::Num && n->num == 1.0; }


    // Peephole constructors: fold trivial identities so the generated C code
    // stays readable. Only algebraic no-ops (x*1, 1*x, x*0, 0*x, --x, -(Num)),
    // never re-associates. Correctness is unchanged.
    PN pn_neg(PN a) {
        if (a && a->kind == Node::Neg) return a->a;                       // -(-x) -> x
        if (a && a->kind == Node::Num) return pn_num(-a->num);            // -(c)  -> (-c)
        auto n = mk(Node::Neg); n->a = std::move(a); return n;
    }

    PN pn_add(PN a, PN b) { auto n = mk(Node::Add); n->a = std::move(a); n->b = std::move(b); return n; }

    PN pn_sub(PN a, PN b) { auto n = mk(Node::Sub); n->a = std::move(a); n->b = std::move(b); return n; }

    // Parser emits literal -1 as Neg(Num(1)) rather than Num(-1), so catch
    // both shapes here (my own pn_neg fold produces Num(-1), the parser does
    // not).
    bool pn_is_neg_one(const PN& n) {
        if (!n) return false;
        if (n->kind == Node::Num && n->num == -1.0) return true;
        if (n->kind == Node::Neg && pn_is_one(n->a)) return true;
        return false;
    }

    PN pn_mul(PN a, PN b) {
        if (pn_is_zero(a) || pn_is_zero(b)) return pn_num(0);
        if (pn_is_one(a)) return b;
        if (pn_is_one(b)) return a;
        if (pn_is_neg_one(a)) return pn_neg(std::move(b));                // (-1)*x -> -x
        if (pn_is_neg_one(b)) return pn_neg(std::move(a));                // x*(-1) -> -x
        auto n = mk(Node::Mul); n->a = std::move(a); n->b = std::move(b); return n;
    }

    PN pn_div(PN a, PN b) { auto n = mk(Node::Div); n->a = std::move(a); n->b = std::move(b); return n; }


    // Try to split n = coef*v + rem with coef, rem both free of v.
    // Returns false when v enters non-linearly and the analytic solve is
    // unsafe: product v*v, division by an expression that contains v, or v
    // appearing inside a pow-exponent or function call.
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


    // Complex CD only: fmod and atan2 have no complex counterpart (both are
    // defined through the sign/magnitude of REAL arguments), so configCUDA.h
    // deliberately provides no ucmplx overload for them. Catch them here, at
    // codegen time, so the user gets the reason instead of an NVRTC template
    // error from the middle of a generated kernel.
    void cd_check_complex_safe(const PN& n) {
        if (!n) return;
        if (n->kind == Node::Call && (n->name == "fmod" || n->name == "atan2"))
            throw std::runtime_error("Complex CD: function " + n->name +
                " is not defined in complex arithmetic - pick another scheme");
        cd_check_complex_safe(n->a);
        cd_check_complex_safe(n->b);
        for (const auto& c : n->args) cd_check_complex_safe(c);
    }


    PN pn_call1(const char* f, PN u) {
        auto n = mk(Node::Call); n->name = f; n->args.push_back(std::move(u)); return n;
    }

    PN pn_call2(const char* f, PN u, PN w) {
        auto n = mk(Node::Call); n->name = f;
        n->args.push_back(std::move(u)); n->args.push_back(std::move(w)); return n;
    }

    PN pn_pow(PN a, PN b) {
        // a^0 -> 1 and a^1 -> a, so the power rule never leaves pow(x, 1.0) behind
        if (b && b->kind == Node::Num) {
            if (b->num == 0.0) return pn_num(1);
            if (b->num == 1.0) return a;
        }
        auto n = mk(Node::Pow); n->a = std::move(a); n->b = std::move(b); return n;
    }

    PN jd_add(PN a, PN b) {
        if (pn_is_zero(a)) return b;
        if (pn_is_zero(b)) return a;
        return pn_add(std::move(a), std::move(b));
    }

    PN jd_sub(PN a, PN b) {
        if (pn_is_zero(b)) return a;
        if (pn_is_zero(a)) return pn_neg(std::move(b));
        return pn_sub(std::move(a), std::move(b));
    }


    PN pn_diff_call(const PN& n, const std::string& v) {
        const std::string& f = n->name;
        // pow / atan2 / fmod are the only binary calls the parser produces
        if (f == "pow") {
            if (n->args.size() != 2) throw std::runtime_error("jacobian: pow needs 2 args");
            return pn_diff(pn_pow(n->args[0], n->args[1]), v);
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


    // Same role as cd_check_complex_safe: refuse at codegen time, with a readable
    // reason, instead of letting NVRTC fail inside a generated kernel. These three
    // parse fine but have no usable derivative (0 almost everywhere / undefined).
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


    // Jacobian entries as C strings, row-major: out[i*N + j] = d f_i / d vars[j],
    // emitted over state array st. Counterpart of rhs_over above.
    std::vector<std::string> jac_over(const System& s, const std::string& st) {
        if (s.vars.size() != s.rhs.size()) throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        NameMap nm = build_namemap(s, st);
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

    // Dormand-Prince 8(7), 13-стадийный. Матрицы M[13][12] и B[2][13] — // const, компилятор положит в constant memory. Считаем y (8-й, b[0]) и z
    // (7-й, b[1]) — z пока не используется, но оставлен под будущий адаптивный
    // шаг по |y - z|. На каждый step X := y.
    //
    // Коэффициент печатается суммой "(numb)(c0) + (numb)(c1) + ...": в double
    // она сворачивается при компиляции в c0, в dd/qd CPU-ветки вкладки Order
    // складывается точно (two_sum) и несёт 32/62 знака коэффициента.
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

    // "const numb <decl> = {...};" — rows x cols из таблицы с шагом
    // строки ld (ld > cols — берётся левый верхний угол).
    void md_table(std::ostringstream& o, const char* decl, const MultiDoubleCoef* v,
                  int rows, int cols, int ld) {
        o << "    static const numb " << decl << " = {\n";
        for (int i = 0; i < rows; ++i) {
            o << "        {";
            for (int j = 0; j < cols; ++j) o << (j ? ", " : "") << md_literal(v[i * ld + j]);
            o << "}" << (i + 1 < rows ? "," : "") << "\n";
        }
        o << "    };\n";
    }


    std::string adaptive_emb_body(const System& s, const EmbSpec& e) {
        const int N = (int)s.vars.size(), S = e.S;
        const auto kx = rhs_over(s, "X1"), ky = rhs_over(s, "Y");
        std::ostringstream o;
        md_table(o, ("M[" + std::to_string(S) + "][" + std::to_string(S - 1) + "]").c_str(), e.A, S, S - 1, e.ld);
        md_table(o, ("B[1][" + std::to_string(S) + "]").c_str(), e.b, 1, S, S);
        o << "    static const numb L[" << e.nlo << "][" << e.lc << "] = {\n";
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
        o << "    for (l = 0; l < N; ++l) W[l] = F0[l];\n";
        for (int i = 1; i < S; ++i) {
            o << "    for (l = 0; l < N; ++l) {\n";
            o << "        X2[l] = 0;\n";
            for (int j = 0; j < i; ++j)
                if (nz(e.A[i * e.ld + j]))
                    o << "        X2[l] += M[" << i << "][" << j << "] * W[" << j << " * N + l];\n";
            o << "        X1[l] = X[l] + h * X2[l];\n";
            o << "    }\n";
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


    std::string adaptive_rhs_body(const System& s) {
        const auto k = rhs_over(s, "X");
        std::ostringstream o;
        for (size_t v = 0; v < k.size(); ++v) o << "    F[" << v << "] = (" << k[v] << ");\n";
        return o.str();
    }


    // Плотный выход RK45 — полином Шампайна 4-й степени, форма scipy:
    // y(th) = X + h * sum_j Q_j th^(j+1), Q = K^T P. D = [X, hQ_0..hQ_3].
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

    std::string rk45_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l)\n";
        o << "        Yo[l] = D[l] + th * (D[N + l] + th * (D[2 * N + l] + th * (D[3 * N + l] + th * D[4 * N + l])));\n";
        return o.str();
    }


    // Плотный выход DOP853 — полином 7-й степени Хайрера, форма scipy
    // (Dop853DenseOutput): три доп. стадии 13..15, затем D = [X, F0..F6].
    std::string dop853_dprep_body(const System& s) {
        const int N = (int)s.vars.size();
        const auto kx = rhs_over(s, "X1");
        std::ostringstream o;
        md_table(o, "AE[3][15]", &kDop853AExt[0][0], 3, 15, 15);
        md_table(o, "DD[4][16]", &kDop853D[0][0], 4, 16, 16);
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


    // Кубический Эрмит по концам шага (DOPRI78): D = [X, Y, hF0, hF1],
    // y(th) = X + th*hF0 + th^2*(3d - 2hF0 - hF1) + th^3*(hF0 + hF1 - 2d), d = Y - X.
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

    std::string hermite_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        numb d = D[N + l] - D[l], f0 = D[2 * N + l], f1 = D[3 * N + l];\n";
        o << "        Yo[l] = D[l] + th * (f0 + th * ((3 * d - 2 * f0 - f1) + th * (f0 + f1 - 2 * d)));\n";
        o << "    }\n";
        return o.str();
    }


    // Общий кирпич диагонально-неявного полушага: одно уравнение
    // X[i] = X_saved + hs * f_i(X), решённое относительно своей переменной.
    // Им пользуются Phi* в CD (hs = h2, обратный порядок) и стадия SIMP
    // (hs = h1, прямой порядок) — сам способ решения у них общий.
    //
    // Wrap a subterm in parentheses only when its top-level operator binds
    // less tightly than '*'/'/' (i.e. Add or Sub); otherwise emit it raw so
    // that `h2 * X[0] * X[1]` stays a left-associative multiplication chain
    // instead of turning into `h2 * (X[0] * X[1])`. Under FMA both forms
    // are algebraically equal but not bit-identical, and a chaotic system
    // (Lorenz, Rossler, ...) amplifies that ULP-level split over ~10^3-10^4
    // steps into visibly different trajectories.
    bool needs_paren_after_mul(const PN& n) {
        return n && (n->kind == Node::Add || n->kind == Node::Sub);
    }

    std::string wrap_paren(const std::string& s, bool w) {
        return w ? "(" + s + ")" : s;
    }

    // Absorb a leading minus of a rem/coef into the operator sign: peel one
    // Neg or a negative Num off the node, so that `- h2 * -a[1]` and
    // `+ h2 * -a[1]` come out as `+ h2 * a[1]` and `- h2 * a[1]` -- the
    // shape a person would write by hand, and one less negation for the
    // compiler to chase. Returns the sign character to emit before "h2 * ".
    char peel_sign(PN& n, char pos, char neg) {
        if (!n) return pos;
        if (n->kind == Node::Neg) { n = n->a; return neg; }
        if (n->kind == Node::Num && n->num < 0.0) { n = pn_num(-n->num); return neg; }
        return pos;
    }

    // "h2 * factor" -> "h2" when factor == 1 (post-peel-sign), same reason
    // as the pn_mul(x, Num(1)) fold: the multiplication reads like noise
    // both to a human and to the peephole compiler expects.
    std::string mul_step(const PN& factor, const std::string& factor_c, const char* hs) {
        return pn_is_one(factor) ? std::string(hs) : hs + (" * " + factor_c);
    }


    // Собирает выражение "hs * expr" в ТОЙ ЖЕ группировке, в какой его получит
    // компилятор из текста, который печатает emit_diag_implicit_eq. mul_step
    // выводит "hs * " + текст множителя, а скобки вокруг множителя ставятся
    // только при Add/Sub сверху (needs_paren_after_mul), поэтому "hs * a * b"
    // разбирается как ((hs*a)*b): множитель шага заходит в САМЫЙ ЛЕВЫЙ конец
    // цепочки умножений и делений. Интерпретатор, посчитав hs*(a*b), разошёлся
    // бы с GPU на последний бит — в хаотической системе это видно уже через
    // тысячу шагов, ровно как с --fmad.
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


    int func_id(const std::string& nm) {
        if (nm == "sin")return F_SIN; if (nm == "cos")return F_COS; if (nm == "tan")return F_TAN;
        if (nm == "asin")return F_ASIN; if (nm == "acos")return F_ACOS; if (nm == "atan")return F_ATAN;
        if (nm == "sinh")return F_SINH; if (nm == "cosh")return F_COSH; if (nm == "tanh")return F_TANH;
        if (nm == "exp")return F_EXP; if (nm == "log")return F_LOG; if (nm == "log2")return F_LOG2;
        if (nm == "log10")return F_LOG10; if (nm == "sqrt")return F_SQRT; if (nm == "cbrt")return F_CBRT;
        if (nm == "fabs" || nm == "abs")return F_FABS;
        if (nm == "pow")return F_POW; if (nm == "atan2")return F_ATAN2; if (nm == "fmod")return F_FMOD;
        if (nm == "copysign")return F_COPYSIGN;
        throw std::runtime_error("eval: unknown function " + nm);
    }


    double apply_func1(int fid, double x) {
        switch (fid) {
        case F_SIN:return std::sin(x); case F_COS:return std::cos(x); case F_TAN:return std::tan(x);
        case F_ASIN:return std::asin(x); case F_ACOS:return std::acos(x); case F_ATAN:return std::atan(x);
        case F_SINH:return std::sinh(x); case F_COSH:return std::cosh(x); case F_TANH:return std::tanh(x);
        case F_EXP:return std::exp(x); case F_LOG:return std::log(x); case F_LOG2:return std::log2(x);
        case F_LOG10:return std::log10(x); case F_SQRT:return std::sqrt(x); case F_CBRT:return std::cbrt(x);
        case F_FABS:return std::fabs(x);
        } return 0;
    }

    double apply_func2(int fid, double a, double b) {
        switch (fid) {
        case F_POW:return std::pow(a, b); case F_ATAN2:return std::atan2(a, b); case F_FMOD:return std::fmod(a, b);
        case F_COPYSIGN:return std::copysign(a, b);
        } return 0;
    }


    // Комплексные версии — те же id, те же формулы, что и в device-коде: обе
    // ветки зовут функции из configCUDA.h, поэтому CPU-траектория Complex CD
    // повторяет GPU-шную операция в операцию (в пределах разной группировки FMA).
    // atan2/fmod комплексного смысла не имеют и до сюда не доходят: система с
    // ними отсекается в cd_check_complex_safe ещё на кодгене.
    ucmplx apply_func1(int fid, ucmplx x) {
        switch (fid) {
        case F_SIN:return sin(x); case F_COS:return cos(x); case F_TAN:return tan(x);
        case F_ASIN:return asin(x); case F_ACOS:return acos(x); case F_ATAN:return atan(x);
        case F_SINH:return sinh(x); case F_COSH:return cosh(x); case F_TANH:return tanh(x);
        case F_EXP:return exp(x); case F_LOG:return log(x); case F_LOG2:return log2(x);
        case F_LOG10:return log10(x); case F_SQRT:return sqrt(x); case F_CBRT:return cbrt(x);
        case F_FABS:return fabs(x);
        } return ucmplx(0, 0);
    }

    ucmplx apply_func2(int fid, ucmplx a, ucmplx b) {
        switch (fid) {
        case F_POW:return pow(a, b);
        } return ucmplx(0, 0);
    }


    // Степень отдельной функцией: у double это std::pow, у ucmplx — перегрузка
    // из configCUDA.h (со спрямлением целых показателей). Нужна, чтобы
    // run_program остался одним шаблоном на оба типа состояния.
    double  op_pow(double a, double b) { return std::pow(a, b); }

    ucmplx  op_pow(ucmplx a, ucmplx b) { return pow(a, b); }
}
