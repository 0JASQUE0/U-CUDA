#include "expression_internal.h"

namespace ucuda::equations_detail {

    void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) {
        bool paren = prec(n->kind) < pp;
        if (paren) { o << "("; emit(n, nm, o); o << ")"; }
        else { emit(n, nm, o); }
    }


    std::string emit_to_str(const PN& n, const NameMap& nm) {
        std::ostringstream o; emit(n, nm, o); return o.str();
    }


    // Схемы
    // Discrete map x_{n+1} = f(x_n): the right-hand side is the step, h unused.
    // The temp buffer is required - updating in place would be Gauss-Seidel.
    std::string scheme_map(const System& s) {
        int N = (int)s.vars.size(); auto f = rhs_over(s, "X"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i) o << "    X1[" << i << "] = (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }

    std::string scheme_euler(const System& s) {
        int N = s.vars.size(); auto f = rhs_over(s, "X"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }

    std::string scheme_euler_cromer(const System& s) {
        auto f = rhs_over(s, "X"); std::ostringstream o;
        for (size_t i = 0; i < f.size(); ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n"; return o.str();
    }

    std::string scheme_midpoint(const System& s) {
        int N = s.vars.size();
        auto f0 = rhs_over(s, "X"); auto f1 = rhs_over(s, "X1"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + 0.5 * h * (" << f0[i] << ");\n";
        for (int i = 0; i < N; ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f1[i] << ");\n"; return o.str();
    }

    std::string scheme_rk4(const System& s) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    numb k[" << N << "][4];\n    int N = " << N << ";\n    int i, j;\n";
        o << "    for (i = 0; i < N; i++) {\n        X1[i] = X[i];\n    }\n";
        o << "    for (j = 0; j < 4; j++) {\n";
        for (int i = 0; i < N; ++i)o << "        k[" << i << "][j] = (" << k[i] << ");\n";
        o << "        if (j == 3) {\n            for (i = 0; i < N; i++) {\n            X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) / 6;\n            }\n        }\n";
        o << "        else if (j == 2) {\n            for (i = 0; i < N; i++) {\n            X1[i] = X[i] + h * k[i][j];\n            }\n        }\n";
        o << "        else {\n            for (i = 0; i < N; i++) {\n                X1[i] = X[i] + 0.5 * h * k[i][j];\n            }\n        }\n    }\n"; return o.str();
    }


    std::string scheme_dopri78(const System& s, bool legacy ) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;
        md_table(o, "M[13][12]", legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12, 12);
        md_table(o, "B[2][13]",  legacy ? &kDopri78LegacyB[0][0] : &kDopri78B[0][0], 2, 13, 13);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb y[" << N << "];\n";
        o << "    numb z[" << N << "];\n";
        o << "    numb k[" << N << "][13];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < 13; ++i) {\n";
        for (int v = 0; v < N; ++v)
            o << "        k[" << v << "][i] = (" << k[v] << ");\n";
        o << "        if (i != 12) {\n";
        o << "            for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "            for (j = 0; j < i + 1; ++j)\n";
        o << "                for (l = 0; l < N; ++l)\n";
        o << "                    X2[l] += M[i + 1][j] * k[l][j];\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X1[l] = X[l] + h * X2[l];\n";
        o << "        }\n";
        o << "    }\n";

        // 8-й порядок: y
        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < 13; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[0][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        y[l] = X[l] + h * X2[l];\n";

        // 7-й порядок: z (future-proof для адаптивного шага)
        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < 13; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[1][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        z[l] = X[l] + h * X2[l];\n";
        o << "    (void)z[0];\n";

        o << "    for (l = 0; l < N; ++l) X[l] = y[l];\n";
        return o.str();
    }


    // Явный РК с постоянным шагом по таблице Бутчера: RK45 (S = 6) и DOP853
    // (S = 12). A — строки стадий 0..S-1 с шагом ld (строка 0 нулевая,
    // печатаются столбцы 0..S-2), b — S весов старшего решения. Форма кода
    // та же, что у DOPRI78: стадии в цикле, таблицы const. Оценщики и
    // стадия FSAL здесь не нужны — их считает адаптивный вариант.
    std::string scheme_erk(const System& s, const MultiDoubleCoef* A, int ld,
                           const MultiDoubleCoef* b, int S) {
        int N = s.vars.size(); auto k = rhs_over(s, "X1"); std::ostringstream o;
        const std::string sm1 = std::to_string(S - 1), ss = std::to_string(S);
        md_table(o, ("M[" + ss + "][" + sm1 + "]").c_str(), A, S, S - 1, ld);
        md_table(o, ("B[1][" + ss + "]").c_str(), b, 1, S, S);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb k[" << N << "][" << S << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < " << S << "; ++i) {\n";
        for (int v = 0; v < N; ++v)
            o << "        k[" << v << "][i] = (" << k[v] << ");\n";
        o << "        if (i != " << S - 1 << ") {\n";
        o << "            for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "            for (j = 0; j < i + 1; ++j)\n";
        o << "                for (l = 0; l < N; ++l)\n";
        o << "                    X2[l] += M[i + 1][j] * k[l][j];\n";
        o << "            for (l = 0; l < N; ++l)\n";
        o << "                X1[l] = X[l] + h * X2[l];\n";
        o << "        }\n";
        o << "    }\n";

        o << "    for (l = 0; l < N; ++l) X2[l] = 0;\n";
        o << "    for (i = 0; i < " << S << "; ++i)\n";
        o << "        for (l = 0; l < N; ++l)\n";
        o << "            X2[l] += B[0][i] * k[l][i];\n";
        o << "    for (l = 0; l < N; ++l)\n";
        o << "        X[l] = X[l] + h * X2[l];\n";
        return o.str();
    }


    std::string scheme_implicit_common(const System& s, ImplicitKind kind) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();

        const bool  cx  = (kind == ImplicitKind::ComplexEuler);
        // Массивы состояния, из которых читают сгенерированные выражения:
        // X/Xn у вещественных схем (как было), Z/Zn у комплексной.
        const char* stv = cx ? "Z"  : "X";
        const char* stn = cx ? "Zn" : "Xn";
        const char* sty = cx ? "ucmplx" : "numb";

        auto f0 = rhs_over(s, stv);    // predictor, evaluated at the old state
        auto fn = rhs_over(s, stn);    // residual, evaluated at the Newton iterate
        auto Jn = jac_over(s, stn);    // Jacobian, same point as the residual

        if (cx) {
            // Те же запреты, что у Complex CD: у fabs/floor/fmod нет
            // аналитического продолжения в комплексную плоскость, и шаг с
            // мнимой частью по ним считать нечем.
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                cd_check_complex_safe(p.parse());
            }
        }

        const bool   full  = s.newton_full;
        const double tol   = s.newton_tol > 0.0 ? s.newton_tol : 1e-10;
        const int    maxit = s.newton_max_iters > 0 ? s.newton_max_iters : 1;

        // Компаунд-операторов у ucmplx нет (см. configCUDA.h), поэтому в
        // комплексном режиме печатаем развёрнутую форму. У вещественных схем
        // текст обязан остаться ПОБАЙТОВО прежним: он уходит в ключ кэша PTX
        // и в отладочную панель как «что считает CPU».
        auto sub_eq = [cx](const std::string& lhs, const std::string& rhs) {
            return cx ? (lhs + " = " + lhs + " - " + rhs + ";")
                      : (lhs + " -= " + rhs + ";");
        };
        auto div_eq = [cx](const std::string& lhs, const std::string& rhs) {
            return cx ? (lhs + " = " + lhs + " / " + rhs + ";")
                      : (lhs + " /= " + rhs + ";");
        };
        // Вещественный литерал в матрицу: у ucmplx конструктор из numb ЯВНЫЙ,
        // так что Am[k] = 1.0 в комплексном режиме просто не скомпилируется.
        auto lit = [cx](const char* v) {
            return cx ? ("ucmplx(" + std::string(v) + ", 0.0)") : std::string(v);
        };
        const std::string absf = cx ? "ucmplx_abs" : "fabs";

        std::ostringstream o;
        if (cx) {
            o << "    ucmplx Z[" << N << "];\n";
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = ucmplx(X[" << i << "], 0.0);\n";
        }
        o << "    " << sty << " " << stn << "[" << N << "];\n";
        o << "    " << sty << " Fv[" << N << "];\n";
        o << "    " << sty << " Am[" << N * N << "];\n";
        o << "    int  piv[" << N << "];\n";
        o << "    const int  ndim = " << N << ";\n";
        if (cx)
            // Переприсваивается между проходами, поэтому не const.
            o << "    ucmplx hc = ucmplx(a[0] * h, h * " << fmtnum(CIE_IMAG) << ");\n";
        else
            o << "    const numb hc = "
              << (kind == ImplicitKind::Euler ? "h" : "(0.5 * h)") << ";\n";
        o << "    const numb ntol = " << fmtnum(tol) << ";\n";
        o << "    int it, ik, ir, ic, ip, sing, refresh;\n";
        if (cx) {
            o << "    numb mx, av, nrm, prevn;\n";
            o << "    ucmplx dgn, mlt;\n";
        } else {
            o << "    numb mx, av, dgn, mlt, nrm, prevn;\n";
        }

        // Один ньютоновский проход целиком. У вещественных схем печатается
        // один раз, у комплексной — дважды, с разными tau (см. ниже).
        auto emit_pass = [&]() {
        // Predictor: explicit Euler over the same hc, so the Newton start is already
        // second-order accurate for the midpoint variant.
        o << "\n    /* predictor: explicit Euler */\n";
        for (int i = 0; i < N; ++i)
            o << "    " << stn << "[" << i << "] = " << stv << "[" << i
              << "] + hc * (" << f0[i] << ");\n";

        o << "\n    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < " << maxit << "; it++) {\n";

        // One emission of the Jacobian + LU, entered on iteration 0 and again
        // whenever `refresh` is set below. Full Newton sets it every iteration;
        // modified Newton only when the frozen matrix stops contracting.
        o << "        if (refresh) {\n";
        o << "            /* Am = I - hc * J(" << stn << ") */\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Am[" << (i * N + j) << "] = ";
                if (d == "0.0") o << lit(i == j ? "1.0" : "0.0") << ";\n";
                else if (i == j) o << "1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
        }
        o << "            /* LU with partial pivoting */\n";
        o << "            for (ik = 0; ik < ndim; ik++) {\n";
        o << "                ip = ik; mx = " << absf << "(Am[ik * ndim + ik]);\n";
        o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
        o << "                    av = " << absf << "(Am[ir * ndim + ik]);\n";
        o << "                    if (av > mx) { mx = av; ip = ir; }\n";
        o << "                }\n";
        o << "                piv[ik] = ip;\n";
        o << "                if (ip != ik)\n";
        o << "                    for (ic = 0; ic < ndim; ic++) {\n";
        o << "                        mlt = Am[ik * ndim + ic];\n";
        o << "                        Am[ik * ndim + ic] = Am[ip * ndim + ic];\n";
        o << "                        Am[ip * ndim + ic] = mlt;\n";
        o << "                    }\n";
        o << "                dgn = Am[ik * ndim + ik];\n";
        // Singular pivot: flag and bail instead of dividing. Nudging the pivot
        // would only turn the correction into inf and then nan, killing a
        // trajectory that is otherwise alive; leaving the step on the predictor
        // costs one order locally and nothing globally.
        o << "                if (" << absf << "(dgn) < 1e-30) { sing = 1; break; }\n";
        o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
        o << "                    mlt = Am[ir * ndim + ik] / dgn;\n";
        o << "                    Am[ir * ndim + ik] = mlt;\n";
        o << "                    for (ic = ik + 1; ic < ndim; ic++)\n";
        o << "                        " << sub_eq("Am[ir * ndim + ic]", "mlt * Am[ik * ndim + ic]") << "\n";
        o << "                }\n";
        o << "            }\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        o << "        if (sing) break;\n";

        o << "        /* residual F(" << stn << ") = " << stn << " - " << stv
          << " - hc * f(" << stn << ") */\n";
        for (int i = 0; i < N; ++i)
            o << "        Fv[" << i << "] = " << stn << "[" << i << "] - " << stv << "[" << i
              << "] - hc * (" << fn[i] << ");\n";
        o << "        /* solve Am * dX = Fv in place */\n";
        o << "        for (ik = 0; ik < ndim; ik++) {\n";
        o << "            ip = piv[ik];\n";
        o << "            if (ip != ik) { mlt = Fv[ik]; Fv[ik] = Fv[ip]; Fv[ip] = mlt; }\n";
        o << "            for (ir = ik + 1; ir < ndim; ir++)\n";
        o << "                " << sub_eq("Fv[ir]", "Am[ir * ndim + ik] * Fv[ik]") << "\n";
        o << "        }\n";
        o << "        for (ik = ndim - 1; ik >= 0; ik--) {\n";
        o << "            for (ic = ik + 1; ic < ndim; ic++)\n";
        o << "                " << sub_eq("Fv[ik]", "Am[ik * ndim + ic] * Fv[ic]") << "\n";
        o << "            " << div_eq("Fv[ik]", "Am[ik * ndim + ik]") << "\n";
        o << "        }\n";
        o << "        nrm = 0.0;\n";
        for (int i = 0; i < N; ++i) {
            if (cx)
                // |dX|^2 в комплексном случае — по модулю, а не по квадрату
                // самого числа: Fv[i]*Fv[i] у комплексного не вещественно.
                o << "        Zn[" << i << "] = Zn[" << i << "] - Fv[" << i << "];"
                  << " nrm += Fv[" << i << "].re * Fv[" << i << "].re + Fv[" << i
                  << "].im * Fv[" << i << "].im;\n";
            else
                o << "        Xn[" << i << "] -= Fv[" << i << "]; nrm += Fv[" << i
                  << "] * Fv[" << i << "];\n";
        }
        o << "        if (nrm < ntol * ntol) break;\n";
        if (full)
            o << "        refresh = 1;\n";
        else
            // nrm and prevn are SQUARED norms, so 0.25 is the 0.5 contraction
            // factor. Without this a frozen Jacobian diverges outright on a stiff
            // nonlinear system (measured on Van der Pol, mu = 100, h >= 0.01);
            // refreshing recovers the full-Newton answer at half the Jacobians.
            o << "        if (nrm > 0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n\n";

        if (kind == ImplicitKind::Euler)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Xn[" << i << "];\n";
        else if (kind == ImplicitKind::Midpoint)
            // Xn holds the stage value Y; recover the endpoint from it.
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = 2.0 * Xn[" << i << "] - X[" << i << "];\n";
        else
            // Комплексный проход кладёт результат обратно в Z: второй проход
            // стартует с него, а после второго Re Z уходит в X.
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = Zn[" << i << "];\n";
        };  // emit_pass

        emit_pass();
        if (cx) {
            // Второй полушаг — сопряжённый: tau2 = h*(1 - a[0]) - i*h*CIE_IMAG.
            // tau1 + tau2 = h при любом a[0], но h^2-член гасится только при
            // a[0] = 1/2, и только там метод второго порядка.
            o << "\n    hc = ucmplx((1 - a[0]) * h, -h * " << fmtnum(CIE_IMAG) << ");\n";
            emit_pass();
            // Наружу — только Re, как у Complex CD: всё вокруг
            // calculateDiscreteModel остаётся вещественным.
            o << "\n";
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";
        }

        return o.str();
    }

    std::string scheme_implicit_euler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::Euler); }

    std::string scheme_implicit_midpoint(const System& s) { return scheme_implicit_common(s, ImplicitKind::Midpoint); }

    std::string scheme_complex_ieuler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::ComplexEuler); }

    // saved — имя временной под X_saved в итерационной ветке: у Complex CD4 два
    // прохода живут в одной области видимости, поэтому имя приходит снаружи.
    void emit_diag_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                               const std::string& v, const std::string& x,
                               const NameMap& nm, const char* hs, const char* sty,
                               const std::string& saved) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);

        if (linear && pn_is_zero(coef)) {
            // v absent from f_i -> explicit one-shot update.
            char sgn = peel_sign(rem, '+', '-');
            std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
            o << "    " << x << " = " << x
              << " " << sgn << " " << mul_step(rem, rem_c, hs) << ";\n";
        }
        else if (linear) {
            // Denominator: 1 (- | +) hs * |coef|.
            char dsgn = peel_sign(coef, '-', '+');
            std::string coef_c = wrap_paren(emit_to_str(coef, nm), needs_paren_after_mul(coef));
            if (pn_is_zero(rem))
                o << "    " << x << " = " << x
                  << " / (1 " << dsgn << " " << mul_step(coef, coef_c, hs) << ");\n";
            else {
                // Numerator: X_saved (+ | -) hs * |rem|.
                char nsgn = peel_sign(rem, '+', '-');
                std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
                o << "    " << x << " = (" << x
                  << " " << nsgn << " " << mul_step(rem, rem_c, hs)
                  << ") / (1 " << dsgn << " " << mul_step(coef, coef_c, hs) << ");\n";
            }
        }
        else {
            // v enters non-linearly -> fixed-point iterations from X_saved.
            std::string rhs_c = emit_to_str(rhs_ast, nm);
            bool w = needs_paren_after_mul(rhs_ast);
            o << "    " << sty << " " << saved << " = " << x << ";\n";
            for (int k = 0; k < CD_ITERS; ++k)
                o << "    " << x << " = " << saved
                  << " + " << hs << " * " << wrap_paren(rhs_c, w) << ";\n";
        }
    }


    // core_only -- ядро для ExtrZ: только проходы над Z[], объявленным снаружи,
    // без перехода X -> Z на входе и без Re на выходе (см. codegen.hpp).
    // flip_halves -- вариант {1-0} любой CD-схемы: в каждом проходе сначала
    // неявная половина (h1, обратный порядок), потом явная (h2, прямой). Это
    // та же композиция над сопряжённым базовым методом: условия порядка
    // переходят в себя заменой N_k -> (-1)^(k+1) N_k, порядок и симметрия не
    // меняются. Замерено (Лоренц, Рёсслер): CD и Complex CD -- 2, Complex CD4,
    // o4s3, o4s4 -- 4; у CD4-семейства {1-0} точнее {0-1} в 1.5-4 раза.
    std::string scheme_cd_common(const System& s, CdKind kind, bool core_only ,
                                 bool flip_halves ) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();
        if (N < 2) throw std::runtime_error("CD method requires N >= 2");

        const bool cx = (kind != CdKind::Real);
        // State array the generated expressions read from: X[] for the real
        // scheme (in-place, as before), Z[] for the complex ones.
        const char* stv = cx ? "Z" : "X";
        const char* sty = cx ? "ucmplx" : "numb";

        NameMap nm = build_namemap(s, stv);
        std::vector<PN> rhs_ast(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            rhs_ast[i] = p.parse();
        }
        if (cx)
            for (int i = 0; i < N; ++i) cd_check_complex_safe(rhs_ast[i]);

        std::ostringstream o;
        if (cx && !core_only) {
            o << "    ucmplx Z[" << N << "];\n";
            for (int i = 0; i < N; ++i)
                o << "    Z[" << i << "] = ucmplx(X[" << i << "], 0.0);\n";
        }
        // Полушаги h1/h2 каждого прохода композиции — выражениями над
        // объявленными выше переменными. Пусто у CD и Complex CD: у них ровно
        // один проход на весь h.
        std::vector<std::string> pass_h1, pass_h2;
        if (kind == CdKind::Real) {
            o << "    numb h1 = h * a[0];\n";
            o << "    numb h2 = h * (1 - a[0]);\n";
        }
        else if (kind == CdKind::Cx) {
            o << kCcdImagDecl;
            o << "    ucmplx h1 = ucmplx(a[0] * h,  h * ccd_im);\n";
            o << "    ucmplx h2 = ucmplx((1 - a[0]) * h, -h * ccd_im);\n";
        }
        else if (kind == CdKind::Cx4ss01 || kind == CdKind::Cx4ss10) {
            // Разбиение комплексное и фиксированное, a[0] не участвует:
            // проход 1 -- (p, q), проход 2 -- (q, p). У {1-0} h1 -- это
            // неявная половина, она идёт первой (см. emit_pass).
            o << "    ucmplx cp = ucmplx(h / (numb)6, -h / (numb)6);   // (1-i)h/6\n";
            o << "    ucmplx cq = ucmplx(h / (numb)3,  h / (numb)6);   // (2+i)h/6\n";
            pass_h1 = { "cp", "cq" };
            pass_h2 = { "cq", "cp" };
            o << "    ucmplx h1 = " << pass_h1[0] << ";\n";
            o << "    ucmplx h2 = " << pass_h2[0] << ";\n";
        }
        else {
            // Комплексная композиция: s делит уже СВОЙ комплексный подшаг
            // внутри прохода, поэтому здесь объявляются только сами подшаги,
            // а h1/h2 перевыставляются перед каждым проходом.
            // ccd_im нужен всем, кроме o4s3: у той свой коэффициент, и лишняя
            // константа в теле была бы неиспользованной.
            std::vector<std::string> cgs;
            if (kind != CdKind::Cx4s3) o << kCcdImagDecl;
            if (kind == CdKind::Cx4) {
                // gamma*h and conj(gamma)*h.
                o << "    ucmplx g  = ucmplx(0.5 * h,  h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx(0.5 * h, -h * ccd_im);\n";
                cgs = { "g", "gc" };
            }
            else if (kind == CdKind::Cx4s4) {
                // (gamma/2, conj/2, conj/2, gamma/2) — палиндром.
                o << "    ucmplx g  = ucmplx(0.25 * h,  0.5 * h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx(0.25 * h, -0.5 * h * ccd_im);\n";
                cgs = { "g", "gc", "gc", "g" };
            }
            else {
                // (alpha, 1 - 2*alpha, alpha) — палиндром. Средний коэффициент
                // выражен через alpha, а не литералом: тогда сумма подшагов
                // равна h ТОЧНО и в плавающей арифметике тоже.
                o << kCcd3Decl;
                o << "    ucmplx ga = ucmplx(ccd3_re * h, ccd3_im * h);\n";
                o << "    ucmplx gb = ucmplx(((numb)1 - (numb)2 * ccd3_re) * h,"
                     " -(numb)2 * ccd3_im * h);\n";
                cgs = { "ga", "gb", "ga" };
            }
            for (const std::string& g : cgs) {
                pass_h1.push_back(g + " * a[0]");
                pass_h2.push_back(g + " * (1 - a[0])");
            }
            o << "    ucmplx h1 = " << pass_h1[0] << ";\n";
            o << "    ucmplx h2 = " << pass_h2[0] << ";\n";
        }

        // Phi*_{h2}: diagonally-implicit half-step, reverse order. Сам разбор
        // уравнения — в emit_diag_implicit_eq выше: стадия SIMP гоняет тот же
        // код с h1 и в прямом порядке.

        // Один проход CD целиком: явный полушаг h1 вперёд, неявный h2 назад.
        // Complex CD4 зовёт его дважды с разными h1/h2, поэтому имена временных
        // переменных неявной ветки получают суффикс — иначе второй проход
        // переобъявил бы x0_cd в той же области видимости.
        // У вариантов {1-0} (CCD4 (s-split)_{1-0} и flip_halves) половины
        // меняются местами: неявная (h1, обратный порядок) первой, явная (h2,
        // прямой порядок) второй.
        const bool implicit_first = flip_halves || (kind == CdKind::Cx4ss10);
        auto emit_pass = [&](const char* sfx) {
        // Phi_{h1}: explicit half-step, forward order. Each X[i] update sees
        // the just-written values of X[0..i-1] (Euler-Cromer coupling).
        auto emit_explicit = [&](const char* step) {
            for (int i = 0; i < N; ++i)
                o << "    " << stv << "[" << i << "] = " << stv << "[" << i << "] + "
                  << step << " * (" << emit_to_str(rhs_ast[i], nm) << ");\n";
        };
        auto emit_implicit = [&](const char* step) {
            for (int i = N - 1; i >= 0; --i)
                emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i],
                                      stv + ("[" + std::to_string(i) + "]"), nm,
                                      step, sty, "x" + std::to_string(i) + "_cd" + sfx);
        };
        if (implicit_first) { emit_implicit("h1"); emit_explicit("h2"); }
        else                { emit_explicit("h1"); emit_implicit("h2"); }
        };  // emit_pass

        if (pass_h1.empty()) {
            emit_pass("");
        }
        else {
            for (size_t k = 0; k < pass_h1.size(); ++k) {
                if (k) {
                    o << "    h1 = " << pass_h1[k] << ";\n";
                    o << "    h2 = " << pass_h2[k] << ";\n";
                }
                // Суффикс временных неявной ветки: без него второй проход
                // переобъявил бы x0_cd в той же области видимости.
                const std::string sfx = k ? ("_" + std::to_string(k)) : std::string();
                emit_pass(sfx.c_str());
            }
        }

        // Наружу — только действительная часть: X[] вещественный и на входе, и
        // на выходе, поэтому вся обвязка (ядра, LLE/LS, бассейны, рендер) о
        // комплексности не знает.
        if (cx && !core_only)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";

        return o.str();
    }


    std::string scheme_cd(const System& s)          { return scheme_cd_common(s, CdKind::Real); }

    std::string scheme_complex_cd(const System& s)  { return scheme_cd_common(s, CdKind::Cx); }

    std::string scheme_complex_cd4(const System& s) { return scheme_cd_common(s, CdKind::Cx4); }

    std::string scheme_complex_cd4_s3(const System& s) { return scheme_cd_common(s, CdKind::Cx4s3); }

    std::string scheme_complex_cd4_s4(const System& s) { return scheme_cd_common(s, CdKind::Cx4s4); }

    std::string scheme_complex_cd4_ss01(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss01); }

    std::string scheme_complex_cd4_ss10(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss10); }


    // SEMP / SIMP — методы средней точки, у которых СТАДИЯ считается
    // последовательно по компонентам (Гаусс-Зейдель) вместо полной неявной
    // системы Implicit Midpoint:
    //
    //   X1 = X                                  — база для корректора
    //   стадия, шаг h1 = s*h, прямой порядок, in-place по X:
    //     SEMP: X[i] = X[i] + h1 * f_i(X)       — явно, i-е уравнение уже видит
    //           новые X[0..i-1] (связка как у Euler-Cromer);
    //     SIMP: X[i] = X_saved + h1 * f_i(X)    — решено относительно своей
    //           переменной, той же аналитикой/итерациями, что Phi* в CD.
    //   корректор — полный шаг от X1, все уравнения читают стадию:
    //     X1[i] = X1[i] + h * f_i(X);  затем X = X1.
    //
    // Порядок 2 достигается ТОЛЬКО при s = 1/2: разложение даёт
    // X_next = X + h*F + s*h^2*F'F против h^2/2*F'F у точного решения, так что
    // при s != 1/2 обе схемы падают до первого — ровно как CD. Перестановка
    // аргументов и диагональная неявность порядок не портят: стадия
    // возмущается на O(h^2), а входит в корректор с множителем h, то есть
    // локально это O(h^3).
    // Замерено на эмитируемом коде (эталон RK4 h = 1e-6, s = 0.5): Rossler
    // T = 10 — 2.00 у обеих схем, ошибка примерно вчетверо ниже, чем у явной
    // средней точки при равном h; Lorenz T = 2 — 2.00 у обеих, причём SEMP на
    // этой системе точнее SIMP примерно на порядок. При s = 0.3 обе дают 1.00.
    std::string scheme_semi_midpoint(const System& s, bool implicit_stage) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        NameMap nm = build_namemap(s, "X");
        std::vector<PN> rhs_ast(N);
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            rhs_ast[i] = p.parse();
        }

        std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        o << "    numb h1 = h * a[0];\n";
        for (int i = 0; i < N; ++i)
            o << "    X1[" << i << "] = X[" << i << "];\n";

        for (int i = 0; i < N; ++i) {
            const std::string x = "X[" + std::to_string(i) + "]";
            const std::string saved = "x" + std::to_string(i) + "_si";
            if (!implicit_stage)
                o << "    " << x << " = " << x << " + h1 * ("
                  << emit_to_str(rhs_ast[i], nm) << ");\n";
            else
                emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i], x, nm,
                                      "h1", "numb", saved);
        }

        for (int i = 0; i < N; ++i)
            o << "    X1[" << i << "] = X1[" << i << "] + h * ("
              << emit_to_str(rhs_ast[i], nm) << ");\n";
        for (int i = 0; i < N; ++i)
            o << "    X[" << i << "] = X1[" << i << "];\n";
        return o.str();
    }

    std::string scheme_semp(const System& s) { return scheme_semi_midpoint(s, false); }

    std::string scheme_simp(const System& s) { return scheme_semi_midpoint(s, true); }


    // D — диагонально-неявный метод первого порядка, он же Phi* из CD и он же
    // стадия SIMP, взятая как самостоятельный шаг:
    //   для i = 0..N-1:  X[i] = X_saved + h * f_i(X),
    // решённое относительно своей переменной (аналитически, если f_i линейна по
    // ней, иначе CD_ITERS простых итераций), в прямом порядке — i-е уравнение
    // уже видит новые X[0..i-1]. Диагонально-неявный близнец Euler-Cromer:
    // та же связка по компонентам, но каждое уравнение решается, а не считается.
    // Перекрёстные члены (как и у CD) в решение не входят — это НЕ полный
    // неявный Эйлер, здесь нет ни якобиана, ни Ньютона.
    // Коэффициент симметрии a[0] не используется: шаг не делится пополам.
    // Замерено (эталон RK4 h = 1e-6): Rossler T = 10 — 1.00, Lorenz T = 2 — 1.00.
    std::string scheme_d(const System& s) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        NameMap nm = build_namemap(s, "X");
        std::ostringstream o;
        for (int i = 0; i < N; ++i) {
            Parser p(s.rhs[i], s.latex);
            PN ast = p.parse();
            const std::string x = "X[" + std::to_string(i) + "]";
            const std::string saved = "x" + std::to_string(i) + "_d";
            emit_diag_implicit_eq(o, ast, s.vars[i], x, nm, "h", "numb", saved);
        }
        return o.str();
    }
}
