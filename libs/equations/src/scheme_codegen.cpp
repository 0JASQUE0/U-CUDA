#include "expression_internal.h"
using namespace ucuda::equations_detail;

namespace ucuda::equations_detail {
void emit_child(const PN& n, const NameMap& nm, std::ostream& o, int pp) {
        bool paren = prec(n->kind) < pp;
        if (paren) { o << "("; emit(n, nm, o); o << ")"; }
        else { emit(n, nm, o); }
    }
}

namespace ucuda::equations_detail {
std::string emit_to_str(const PN& n, const NameMap& nm) {
        std::ostringstream o; emit(n, nm, o); return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_map(const System& s) {
        int N = (int)s.vars.size(); auto f = rhs_over(s, "X"); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i) o << "    X1[" << i << "] = (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }
}

namespace ucuda::equations_detail {
std::string time_name(const System& s, const char* stage_expr ) {
        if (!system_uses_time(s)) return {};
        return stage_expr ? std::string(stage_expr) : std::string(kStepTimeArg);
    }
}

namespace ucuda::equations_detail {
std::string scheme_euler(const System& s) {
        int N = s.vars.size(); auto f = rhs_over(s, "X", time_name(s)); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    int i;\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n";
        o << "    for (i = 0; i < " << N << "; i++)\n        X[i] = X1[i];\n"; return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_euler_cromer(const System& s) {
        auto f = rhs_over(s, "X", time_name(s)); std::ostringstream o;
        for (size_t i = 0; i < f.size(); ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f[i] << ");\n"; return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_midpoint(const System& s) {
        int N = s.vars.size();
        auto f0 = rhs_over(s, "X", time_name(s));
        auto f1 = rhs_over(s, "X1", time_name(s, "(ucuda_t + (numb)0.5 * h)")); std::ostringstream o;
        o << "    numb X1[" << N << "];\n";
        for (int i = 0; i < N; ++i)o << "    X1[" << i << "] = X[" << i << "] + (numb)0.5 * h * (" << f0[i] << ");\n";
        for (int i = 0; i < N; ++i)o << "    X[" << i << "] = X[" << i << "] + h * (" << f1[i] << ");\n"; return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_rk4(const System& s) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        o << "    numb X1[" << N << "];\n    numb k[" << N << "][4];\n    int N = " << N << ";\n    int i, j;\n";
        o << "    for (i = 0; i < N; i++) {\n        X1[i] = X[i];\n    }\n";
        o << "    for (j = 0; j < 4; j++) {\n";
        // Стадии в моменты t, t + h/2, t + h/2, t + h.
        if (!tn.empty())
            o << "        const numb ucuda_ts = ucuda_t + (j == 0 ? (numb)0 : (j == 3 ? h : (numb)0.5 * h));\n";
        for (int i = 0; i < N; ++i)o << "        k[" << i << "][j] = (" << k[i] << ");\n";
        o << "        if (j == 3) {\n            for (i = 0; i < N; i++) {\n            X[i] = X[i] + h * (k[i][0] + 2 * k[i][1] + 2 * k[i][2] + k[i][3]) * ((numb)1 / (numb)6);\n            }\n        }\n";
        o << "        else if (j == 2) {\n            for (i = 0; i < N; i++) {\n            X1[i] = X[i] + h * k[i][j];\n            }\n        }\n";
        o << "        else {\n            for (i = 0; i < N; i++) {\n                X1[i] = X[i] + (numb)0.5 * h * k[i][j];\n            }\n        }\n    }\n"; return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_dopri78(const System& s, bool legacy ) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        md_table(o, "M[13][12]", legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12, 12);
        md_table(o, "B[2][13]",  legacy ? &kDopri78LegacyB[0][0] : &kDopri78B[0][0], 2, 13, 13);
        if (!tn.empty()) stage_time_table(o, legacy ? &kDopri78LegacyA[0][0] : &kDopri78A[0][0], 13, 12);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb y[" << N << "];\n";
        o << "    numb z[" << N << "];\n";
        o << "    numb k[" << N << "][13];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < 13; ++i) {\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + Cs[i] * h;\n";
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
}

namespace ucuda::equations_detail {
std::string scheme_erk(const System& s, const MultiDoubleCoef* A, int ld,
                           const MultiDoubleCoef* b, int S) {
        const std::string tn = time_name(s, "ucuda_ts");
        int N = s.vars.size(); auto k = rhs_over(s, "X1", tn); std::ostringstream o;
        const std::string sm1 = std::to_string(S - 1), ss = std::to_string(S);
        md_table(o, ("M[" + ss + "][" + sm1 + "]").c_str(), A, S, S - 1, ld);
        md_table(o, ("B[1][" + ss + "]").c_str(), b, 1, S, S);
        if (!tn.empty()) stage_time_table(o, A, S, ld);

        o << "    numb X1[" << N << "];\n";
        o << "    numb X2[" << N << "];\n";
        o << "    numb k[" << N << "][" << S << "];\n";
        o << "    int N = " << N << ";\n";
        o << "    int i, j, l;\n";

        o << "    for (i = 0; i < N; ++i) X1[i] = X[i];\n";
        o << "    for (i = 0; i < " << S << "; ++i) {\n";
        if (!tn.empty()) o << "        const numb ucuda_ts = ucuda_t + Cs[i] * h;\n";
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
}

namespace ucuda::equations_detail {
std::string scheme_implicit_common(const System& s, ImplicitKind kind, int dae_nx ,
                                       bool dae_trap , const char* dae_hc ) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        const int N = (int)s.vars.size();
        const bool dae = dae_nx >= 0;

        const bool  cx  = (kind == ImplicitKind::ComplexEuler);
        // Массивы состояния, из которых читают сгенерированные выражения:
        // X/Xn у вещественных схем (как было), Z/Zn у комплексной.
        const char* stv = cx ? "Z"  : "X";
        const char* stn = cx ? "Zn" : "Xn";
        const char* sty = cx ? "ucmplx" : "numb";

        // Время: предиктор в t_n, невязка и якобиан — в момент неизвестной (Euler, ДАУ: t + h;
        // Midpoint: стадия Y в t + h/2). Комплексный Эйлер времени не поддерживает (codegen_scheme
        // откажет раньше). Без t в системе time_name пуст и текст прежний.
        const std::string t_new = cx ? std::string()
            : time_name(s, kind == ImplicitKind::Midpoint ? "(ucuda_t + (numb)0.5 * h)" : "(ucuda_t + h)");
        auto f0 = rhs_over(s, stv, cx ? std::string() : time_name(s));   // predictor, evaluated at the old state
        auto fn = rhs_over(s, stn, t_new);    // residual, evaluated at the Newton iterate
        auto Jn = jac_over(s, stn, t_new);    // Jacobian, same point as the residual

        if (cx) {
            // Те же запреты, что у Complex CD: у atan2/fmod нет комплексного
            // смысла, и шаг с мнимой частью по ним считать нечем (floor/ceil/fmod
            // уже отсёк jac_over). fabs разрешён: |z| и copysign(1, Re u) в
            // якобиане — см. configCUDA.h.
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                cd_check_complex_safe(p.parse());
            }
        }

        // Допуск, предел итераций и режим якобиана — из a[] (см. NewtonSettings в codegen.hpp).
        const int nslot = newton_slot(s);

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
        // Этап 6: ДАУ больше 8 переменных — разреженная LU по структуре (плотная в форме цикла уходит в
        // локальную память и стоит N^3); до 8 — плотная раскатанная, текст прежний.
        SparseLu SL;
        bool sparse = false;
        if (dae && N > 8) {
            std::vector<MatEntry> pat;
            std::vector<char> df((size_t)N, 0);
            for (int i = 0; i < N; ++i) {
                if (i < dae_nx) { df[(size_t)i] = 1; pat.push_back(MatEntry{ i, i }); }
                for (int j = 0; j < N; ++j)
                    if (Jn[(size_t)i * N + j] != "(numb)0.0" && !(i < dae_nx && i == j)) pat.push_back(MatEntry{ i, j });
            }
            sparse = sparse_lu_build(pat, N, df, SL);
        }
        if (sparse) sparse_lu_decl(o, SL, N);
        o << "    " << sty << " Am[" << N * N << "];\n";
        o << "    int  piv[" << N << "];\n";
        if (cx)
            // Переприсваивается между проходами, поэтому не const.
            o << "    ucmplx hc = ucmplx(a[0] * h, h * (numb)" << fmtnum(CIE_IMAG) << ");\n";
        else if (dae_hc)
            o << "    const numb hc = " << dae_hc << ";\n";
        else
            o << "    const numb hc = "
              << (kind == ImplicitKind::Euler && !dae_trap ? "h" : "((numb)0.5 * h)") << ";\n";
        if (dae_trap) {
            // f(X_n, t_n) — постоянное слагаемое невязки трапеций.
            o << "    numb Fd[" << (dae_nx > 0 ? dae_nx : 1) << "];\n";
            for (int i = 0; i < dae_nx; ++i) o << "    Fd[" << i << "] = (" << f0[i] << ");\n";
        }
        o << "    const numb ntol  = a[" << nslot << "];\n";
        o << "    const int  nmax  = (int)a[" << nslot + 1 << "];\n";
        o << "    const int  nfull = (int)a[" << nslot + 2 << "];\n";
        // LU/solve are printed unrolled up to 8 variables; beyond that the unrolled text
        // grows as N^3 and Am (N^2 values) would not fit in registers anyway, so the
        // loop form stays.
        const bool unroll = N <= 8;
        o << "    int it, ip, sing, refresh;\n";
        if (!unroll) {
            o << "    const int  ndim = " << N << ";\n";
            o << "    int ik, ir, ic;\n";
        }
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
        o << (dae ? "\n    /* predictor: X_n */\n" : "\n    /* predictor: explicit Euler */\n");
        for (int i = 0; i < N; ++i) {
            if (dae)
                o << "    " << stn << "[" << i << "] = " << stv << "[" << i << "];\n";
            else
                o << "    " << stn << "[" << i << "] = " << stv << "[" << i
                  << "] + hc * (" << f0[i] << ");\n";
        }

        o << "\n    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < nmax; it++) {\n";

        // One emission of the Jacobian + LU, entered on iteration 0 and again
        // whenever `refresh` is set below. Full Newton sets it every iteration;
        // modified Newton only when the frozen matrix stops contracting.
        o << "        if (refresh) {\n";
        o << "            /* Am = I - hc * J(" << stn << ") */\n";
        if (sparse) {
            for (const auto& kv : SL.id) {
                const int i = kv.first.first, j = kv.first.second;
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Asp[" << kv.second << "] = ";
                if (!SL.orig.count(kv.first)) o << "(numb)0.0;\n";   // fill-in
                else if (i >= dae_nx) o << "(" << d << ");\n";
                else if (d == "(numb)0.0") o << "(numb)1.0;\n";
                else if (i == j) o << "(numb)1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
            o << "            /* sparse LU, pivots chosen by structure (sparse_lu.h); a failed threshold -> dense */\n";
            o << "            ucuda_dn = 0;\n";
            sparse_lu_factor(o, SL, "            ", "ucuda_dn");
            o << "            if (ucuda_dn) {\n";
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const std::string& d = Jn[(size_t)i * N + j];
                o << "            Am[" << (i * N + j) << "] = ";
                if (dae && i >= dae_nx) o << (d == "(numb)0.0" ? std::string("0.0") : "(" + d + ")") << ";\n";
                else if (d == "(numb)0.0") o << lit(i == j ? "1.0" : "0.0") << ";\n";
                else if (i == j) o << "(numb)1.0 - hc * (" << d << ");\n";
                else             o << "-hc * (" << d << ");\n";
            }
        }
        // LU with partial pivoting, printed unrolled: every index is a compile-time
        // constant and the row swap is a chain of `if (ip == r)` swaps, so Am stays in
        // registers. The loop form indexed Am[ip * ndim + ic] with a runtime pivot and put
        // the whole matrix in local memory (BD 2D Lorenz 256^2, float: 1002 -> 410 ms;
        // double: +2 %, it is FP64-bound). Same operations in the same order -- results
        // are bit-identical to the loop form.
        auto A = [N](int r, int c) { return "Am[" + std::to_string(r * N + c) + "]"; };
        if (!unroll) {
            o << "            /* LU with partial pivoting */\n";
            o << "            for (ik = 0; ik < ndim; ik++) {\n";
            o << "                ip = ik; mx = " << absf << "(Am[ik * ndim + ik]);\n";
            o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
            o << "                    av = " << absf << "(Am[ir * ndim + ik]);\n";
            o << "                    if (av > mx) { mx = av; ip = ir; }\n";
            o << "                }\n";
            o << "                piv[ik] = ip;\n";
            // Переставляются только столбцы от ik (LINPACK dgefa): множители L прежних шагов остаются на
            // своих строках, иначе прямой ход ниже, повторяющий перестановки по одной, берёт чужие
            // множители. Полная перестановка строк давала неверную поправку Ньютона, как только выбор
            // главного срабатывал после первого столбца (строки связей ДАУ) — см. CLAUDE.md, DAE.
            o << "                if (ip != ik)\n";
            o << "                    for (ic = ik; ic < ndim; ic++) {\n";
            o << "                        mlt = Am[ik * ndim + ic];\n";
            o << "                        Am[ik * ndim + ic] = Am[ip * ndim + ic];\n";
            o << "                        Am[ip * ndim + ic] = mlt;\n";
            o << "                    }\n";
            o << "                dgn = Am[ik * ndim + ik];\n";
            o << "                if (" << absf << "(dgn) < (numb)1e-30) { sing = 1; break; }\n";
            o << "                for (ir = ik + 1; ir < ndim; ir++) {\n";
            o << "                    mlt = Am[ir * ndim + ik] / dgn;\n";
            o << "                    Am[ir * ndim + ik] = mlt;\n";
            o << "                    for (ic = ik + 1; ic < ndim; ic++)\n";
            o << "                        " << sub_eq("Am[ir * ndim + ic]", "mlt * Am[ik * ndim + ic]") << "\n";
            o << "                }\n";
            o << "            }\n";
        }
        if (unroll) o << "            /* LU with partial pivoting, unrolled */\n";
        for (int k = 0; unroll && k < N; ++k) {
            o << "            if (!sing) {\n";
            o << "                ip = " << k << "; mx = " << absf << "(" << A(k, k) << ");\n";
            for (int r = k + 1; r < N; ++r)
                o << "                av = " << absf << "(" << A(r, k) << "); if (av > mx) { mx = av; ip = " << r << "; }\n";
            o << "                piv[" << k << "] = ip;\n";
            for (int r = k + 1; r < N; ++r) {
                o << "                if (ip == " << r << ") {";
                for (int c = k; c < N; ++c)   // от k, см. форму цикла выше
                    o << " mlt = " << A(k, c) << "; " << A(k, c) << " = " << A(r, c) << "; " << A(r, c) << " = mlt;";
                o << " }\n";
            }
            o << "                dgn = " << A(k, k) << ";\n";
            // Singular pivot: flag and bail instead of dividing. Nudging the pivot
            // would only turn the correction into inf and then nan, killing a
            // trajectory that is otherwise alive; leaving the step on the predictor
            // costs one order locally and nothing globally.
            o << "                if (" << absf << "(dgn) < (numb)1e-30) sing = 1;\n";
            if (k + 1 < N) {
                o << "                else {\n";
                for (int r = k + 1; r < N; ++r) {
                    o << "                    mlt = " << A(r, k) << " / dgn;\n";
                    o << "                    " << A(r, k) << " = mlt;\n";
                    for (int c = k + 1; c < N; ++c)
                        o << "                    " << sub_eq(A(r, c), "mlt * " + A(k, c)) << "\n";
                }
                o << "                }\n";
            }
            o << "            }\n";
        }
        if (sparse) o << "            }   /* ucuda_dn */\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        o << "        if (sing) break;\n";

        o << "        /* residual F(" << stn << ") = " << stn << " - " << stv
          << " - hc * f(" << stn << ") */\n";
        for (int i = 0; i < N; ++i) {
            if (dae && i >= dae_nx) { o << "        Fv[" << i << "] = (" << fn[i] << ");\n"; continue; }
            o << "        Fv[" << i << "] = " << stn << "[" << i << "] - " << stv << "[" << i
              << "] - hc * (" << fn[i] << ")";
            if (dae_trap) o << " - hc * Fd[" << i << "]";
            o << ";\n";
        }
        if (sparse) {
            o << "        if (!ucuda_dn) {\n";
            o << "        /* sparse solve: Dv = A^-1 Fv */\n";
            sparse_lu_solve(o, SL, "Fv", "Dv", "        ");
            for (int i = 0; i < N; ++i) o << "        Fv[" << i << "] = Dv[" << i << "];\n";
            o << "        } else {\n";
        }
        if (!unroll) {
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
        }
        if (sparse) o << "        }   /* ucuda_dn */\n";
        if (unroll) o << "        /* solve Am * dX = Fv in place, unrolled */\n";
        auto F = [](int i) { return "Fv[" + std::to_string(i) + "]"; };
        for (int k = 0; unroll && k < N; ++k) {
            o << "        ip = piv[" << k << "];\n";
            for (int r = k + 1; r < N; ++r)
                o << "        if (ip == " << r << ") { mlt = " << F(k) << "; " << F(k) << " = " << F(r)
                  << "; " << F(r) << " = mlt; }\n";
            for (int r = k + 1; r < N; ++r)
                o << "        " << sub_eq(F(r), A(r, k) + " * " + F(k)) << "\n";
        }
        for (int k = N - 1; unroll && k >= 0; --k) {
            for (int c = k + 1; c < N; ++c)
                o << "        " << sub_eq(F(k), A(k, c) + " * " + F(c)) << "\n";
            o << "        " << div_eq(F(k), A(k, k)) << "\n";
        }
        o << "        nrm = 0.0;\n";
        for (int i = 0; i < N; ++i) {
            if (cx)
                // |dX|^2 в комплексном случае — по модулю, а не по квадрату
                // самого числа: Fv[i]*Fv[i] у комплексного не вещественно.
                o << "        Zn[" << i << "] = Zn[" << i << "] - Fv[" << i << "];"
                  << " nrm += Fv[" << i << "].re * Fv[" << i << "].re + Fv[" << i
                  << "].im * Fv[" << i << "].im;\n";
            else if (dae && !dae_norm_w(s, i).empty())
                // индекс k >= 2: поправка в норме с множителем h^(k-1) (как RADAU5), см. System::dae_index
                o << "        Xn[" << i << "] -= Fv[" << i << "]; mlt = " << dae_norm_w(s, i) << "Fv[" << i
                  << "]; nrm += mlt * mlt;\n";
            else
                o << "        Xn[" << i << "] -= Fv[" << i << "]; nrm += Fv[" << i
                  << "] * Fv[" << i << "];\n";
        }
        o << "        if (nrm < ntol * ntol) break;\n";
        if (dae) {
            // ДАУ: ещё и относительный критерий. На индексе 2-3 обусловленность ~1/h^k, а
            // множитель Лагранжа порядка десятков: шум поправки ~eps*cond*|X| не опускается
            // ниже абсолютного tol, и каждый шаг впустую шёл до max_iters (двойной маятник,
            // h = 1e-3: ||dX||^2 ~ 5e-20 при tol^2 = 1e-20, связи уже до 1e-16).
            o << "        if (nrm < ntol * ntol * (";
            for (int i = 0; i < N; ++i) {
                const std::string w = dae_norm_w(s, i);
                if (w.empty()) o << (i ? " + " : "") << stn << "[" << i << "] * " << stn << "[" << i << "]";
                else o << (i ? " + " : "") << "(" << w << stn << "[" << i << "]) * (" << w << stn << "[" << i << "])";
            }
            o << ")) break;\n";
        }
        // Precision floor, float only: there ||dX|| never drops below ~eps*|X| ~ 1e-6, so
        // at the usual tol = 1e-10 every step ran all max_iters passes and, the correction
        // no longer shrinking, rebuilt the Jacobian and LU on almost every one. Stop at
        // ||dX||^2 < (16 eps)^2 |Xn|^2 instead (BD 2D Lorenz 256^2: 410 -> 68 ms with the
        // unrolled LU). It changes float results at round-off level; in double (and dd/qd
        // on the CPU) the constant condition drops the branch and nothing changes. A plain
        // `if`, not `if constexpr`: cl.exe builds the CPU DLL from this text in C++14.
        o << "        if (sizeof(numb) == 4) {\n";
        o << "            if (nrm < (numb)3.6e-12f * (";
        for (int i = 0; i < N; ++i) {
            if (i) o << " + ";
            if (cx) o << stn << "[" << i << "].re * " << stn << "[" << i << "].re + "
                      << stn << "[" << i << "].im * " << stn << "[" << i << "].im";
            else    o << stn << "[" << i << "] * " << stn << "[" << i << "]";
        }
        o << ")) break;\n";
        o << "        }\n";
        // full Newton refreshes every iteration. Otherwise nrm and prevn are SQUARED
        // norms, so 0.25 is the 0.5 contraction factor. Without this a frozen Jacobian
        // diverges outright on a stiff nonlinear system (measured on Van der Pol,
        // mu = 100, h >= 0.01); refreshing recovers the full-Newton answer at half the
        // Jacobians.
        o << "        if (nfull || nrm > (numb)0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n\n";

        if (kind == ImplicitKind::Euler)
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Xn[" << i << "];\n";
        else if (kind == ImplicitKind::Midpoint)
            // Xn holds the stage value Y; recover the endpoint from it.
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = (numb)2.0 * Xn[" << i << "] - X[" << i << "];\n";
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
            o << "\n    hc = ucmplx((1 - a[0]) * h, -h * (numb)" << fmtnum(CIE_IMAG) << ");\n";
            emit_pass();
            // Наружу — только Re, как у Complex CD: всё вокруг
            // calculateDiscreteModel остаётся вещественным.
            o << "\n";
            for (int i = 0; i < N; ++i)
                o << "    X[" << i << "] = Z[" << i << "].re;\n";
        }

        return o.str();
    }
}

namespace ucuda::equations_detail {
void dae_require_index1(const std::vector<std::string>& J, int N, int nx, const char* what) {
        for (int i = nx; i < N; ++i) {
            bool any = false;
            for (int j = nx; j < N; ++j) any = any || J[(size_t)i * N + j] != "(numb)0.0";
            if (!any)
                throw std::runtime_error(std::string(what) + " need index 1, but constraint " + std::to_string(i - nx + 1)
                    + " contains no algebraic variable (index >= 2): use DAE Radau IIA, BDF1, SDIRK2, TR-BDF2 or a Rosenbrock method");
        }
    }
}

namespace ucuda::equations_detail {
std::string dae_tr_bdf2(const System& s) {
        const int nx = s.dae_nx;
        const double g = 2.0 - std::sqrt(2.0);
        const double c1 = 1.0 / (g * (2.0 - g)), c2 = (1.0 - g) * (1.0 - g) / (g * (2.0 - g)),
                     c3 = (1.0 - g) / (2.0 - g);
        char hc3[64];
        std::snprintf(hc3, sizeof(hc3), "((numb)%.17g * h)", c3);
        std::ostringstream o;
        o << "    /* TR-BDF2: trapezoid to t + gamma*h, then BDF2 stage */\n";
        o << "    const numb ucuda_h0 = h;\n";
        o << "    numb ucuda_xn[" << (nx > 0 ? nx : 1) << "];\n";
        for (int l = 0; l < nx; ++l) o << "    ucuda_xn[" << l << "] = X[" << l << "];\n";
        o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(g) << ";\n"
          << scheme_implicit_common(s, ImplicitKind::Euler, nx, true) << "    }\n";
        for (int l = 0; l < nx; ++l)
            o << "    X[" << l << "] = (numb)" << fmtnum(c1) << " * X[" << l << "] - (numb)" << fmtnum(c2)
              << " * ucuda_xn[" << l << "];\n";
        o << "    {\n" << scheme_implicit_common(s, ImplicitKind::Euler, nx, false, hc3) << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_sdirk2(const System& s) {
        const int nx = s.dae_nx;
        const double g = 1.0 - std::sqrt(0.5), c = (1.0 - g) / g;
        char hcg[64];
        std::snprintf(hcg, sizeof(hcg), "((numb)%.17g * h)", g);
        std::ostringstream o;
        o << "    /* SDIRK2 (Alexander): two implicit Euler stages with gamma*h */\n";
        o << "    const numb ucuda_h0 = h;\n";
        o << "    numb ucuda_xn[" << (nx > 0 ? nx : 1) << "];\n";
        for (int l = 0; l < nx; ++l) o << "    ucuda_xn[" << l << "] = X[" << l << "];\n";
        o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(g) << ";\n"
          << scheme_implicit_common(s, ImplicitKind::Euler, nx, false) << "    }\n";
        for (int l = 0; l < nx; ++l)
            o << "    X[" << l << "] = ucuda_xn[" << l << "] + (numb)" << fmtnum(c) << " * (X[" << l
              << "] - ucuda_xn[" << l << "]);\n";
        o << "    {\n" << scheme_implicit_common(s, ImplicitKind::Euler, nx, false, hcg) << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_radau5(const System& s) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, M = 3 * N;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const double r6 = std::sqrt(6.0);
        const double A[9] = { (88 - 7 * r6) / 360, (296 - 169 * r6) / 1800, (-2 + 3 * r6) / 225,
                              (296 + 169 * r6) / 1800, (88 + 7 * r6) / 360, (-2 - 3 * r6) / 225,
                              (16 - r6) / 36, (16 + r6) / 36, 1.0 / 9 };
        const double C[3] = { (4 - r6) / 10, (4 + r6) / 10, 1.0 };
        std::ostringstream o;
        o << "    /* Radau IIA (3 stages, order 5), full Newton on " << M << " unknowns */\n";
        o << "    const numb RA[9] = { ";
        for (int k = 0; k < 9; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(A[k]);
        o << " };\n    const numb RC[3] = { ";
        for (int k = 0; k < 3; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(C[k]);
        o << " };\n";
        // Этап 6: матрица 3N x 3N блочная (стадии связаны только через x-строки с ненулевым J) — разреженная
        // LU по структуре; плотная (форма цикла) — только если план не построился.
        SparseLu SL;
        bool sparse = false;
        {
            std::vector<MatEntry> pat;
            std::vector<char> df((size_t)M, 0);
            for (int bi = 0; bi < 3; ++bi)
                for (int l = 0; l < N; ++l) {
                    const int r = bi * N + l;
                    if (l < nx) df[(size_t)r] = 1;
                    for (int bj = 0; bj < 3; ++bj)
                        for (int m = 0; m < N; ++m) {
                            const bool jz = JS[(size_t)l * N + m] == "(numb)0.0";
                            const bool live = l < nx ? ((bi == bj && l == m) || !jz) : (bi == bj && !jz);
                            if (live) pat.push_back(MatEntry{ r, bj * N + m });
                        }
                }
            sparse = sparse_lu_build(pat, M, df, SL);
        }
        o << "    numb Sv[" << N << "], W[" << M << "], Fs[" << M << "], Js[" << 3 * N * N << "], Rv[" << M << "], Wt[" << N << "];\n";
        if (sparse) sparse_lu_decl(o, SL, M);
        o << "    numb Am[" << M * M << "];\n    int  piv[" << M << "];\n";
        o << "    const numb ntol  = a[" << nslot << "];\n";
        o << "    const int  nmax  = (int)a[" << nslot + 1 << "];\n";
        o << "    const int  nfull = (int)a[" << nslot + 2 << "];\n";
        o << "    int it, bi, bj, l, m, ik, ir, ic, ip, sing, refresh;\n";
        o << "    numb mx, av, dgn, mlt, nrm, xx, prevn;\n";
        if (ht) o << "    numb ucuda_ts = ucuda_t;\n";
        // множители нормы h^(k-1)
        for (int i = 0; i < N; ++i) {
            const std::string w = dae_norm_w(s, i);
            o << "    Wt[" << i << "] = " << (w.empty() ? std::string("(numb)1.0") : w.substr(0, w.size() - 3)) << ";\n";
        }
        // предиктор
        for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
        for (int l = 0; l < nx; ++l) o << "    Rv[" << l << "] = (" << fS[l] << ");\n";
        o << "    for (bi = 0; bi < 3; bi++) {\n";
        o << "        for (l = 0; l < " << N << "; l++) W[bi * " << N << " + l] = X[l];\n";
        o << "        for (l = 0; l < " << nx << "; l++) W[bi * " << N << " + l] += RC[bi] * h * Rv[l];\n";
        o << "    }\n";
        // Упрощённый Ньютон RADAU5 (Hairer-Wanner IV.8) при newton_full = 0: J в X один раз на шаг,
        // преобразование T^-1 A^-1 T = diag(gamma, [[alpha, -beta], [beta, alpha]]) делит систему 3N на
        // вещественную N x N (gamma/h M - J) и комплексную ((alpha + i beta)/h M - J), здесь — вещественную
        // 2N x 2N. Не сошёлся (или матрица вырождена) — тот же шаг полным Ньютоном ниже, с предиктора.
        // newton_full = 1 (умолчание ДАУ) — прежний текст и прежние числа.
        {
            // T и Lambda — из таблицы: A^-1, её собственные числа (кубика) и векторы, T = [v1, Re v, Im v].
            double Ai[9];
            {
                const double* a = A;
                const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
                                 + a[2] * (a[3] * a[7] - a[4] * a[6]);
                Ai[0] = (a[4] * a[8] - a[5] * a[7]) / det; Ai[1] = (a[2] * a[7] - a[1] * a[8]) / det; Ai[2] = (a[1] * a[5] - a[2] * a[4]) / det;
                Ai[3] = (a[5] * a[6] - a[3] * a[8]) / det; Ai[4] = (a[0] * a[8] - a[2] * a[6]) / det; Ai[5] = (a[2] * a[3] - a[0] * a[5]) / det;
                Ai[6] = (a[3] * a[7] - a[4] * a[6]) / det; Ai[7] = (a[1] * a[6] - a[0] * a[7]) / det; Ai[8] = (a[0] * a[4] - a[1] * a[3]) / det;
            }
            using cd = std::complex<double>;
            // характеристический многочлен l^3 - c2 l^2 + c1 l - c0
            const double c2 = Ai[0] + Ai[4] + Ai[8];
            const double c1 = Ai[0] * Ai[4] - Ai[1] * Ai[3] + Ai[0] * Ai[8] - Ai[2] * Ai[6] + Ai[4] * Ai[8] - Ai[5] * Ai[7];
            const double c0 = Ai[0] * (Ai[4] * Ai[8] - Ai[5] * Ai[7]) - Ai[1] * (Ai[3] * Ai[8] - Ai[5] * Ai[6])
                            + Ai[2] * (Ai[3] * Ai[7] - Ai[4] * Ai[6]);
            double g = 3.6;
            for (int k = 0; k < 60; ++k) g -= (((g - c2) * g + c1) * g - c0) / ((3 * g - 2 * c2) * g + c1);
            // l^2 - (c2 - g) l + c0/g
            const double pb = c2 - g, pc = c0 / g;
            const double al = pb / 2, be = std::sqrt(pc - al * al);
            auto nullv = [&](cd lam) {   // ядро A^-1 - lam I: векторное произведение двух строк
                cd r0[3] = { Ai[0] - lam, Ai[1], Ai[2] }, r1[3] = { Ai[3], Ai[4] - lam, Ai[5] };
                std::array<cd, 3> v = { r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0] };
                return v;
            };
            const auto v1 = nullv(cd(g, 0)), vc = nullv(cd(al, -be));   // lam = al - i be: Lambda = [[al, -be], [be, al]]
            double T[9], Ti[9];
            for (int r = 0; r < 3; ++r) { T[r * 3] = v1[(size_t)r].real(); T[r * 3 + 1] = vc[(size_t)r].real(); T[r * 3 + 2] = vc[(size_t)r].imag(); }
            {
                const double* a = T;
                const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
                                 + a[2] * (a[3] * a[7] - a[4] * a[6]);
                Ti[0] = (a[4] * a[8] - a[5] * a[7]) / det; Ti[1] = (a[2] * a[7] - a[1] * a[8]) / det; Ti[2] = (a[1] * a[5] - a[2] * a[4]) / det;
                Ti[3] = (a[5] * a[6] - a[3] * a[8]) / det; Ti[4] = (a[0] * a[8] - a[2] * a[6]) / det; Ti[5] = (a[2] * a[3] - a[0] * a[5]) / det;
                Ti[6] = (a[3] * a[7] - a[4] * a[6]) / det; Ti[7] = (a[1] * a[6] - a[0] * a[7]) / det; Ti[8] = (a[0] * a[4] - a[1] * a[3]) / det;
            }
            // Lambda = Ti Ai T: проверка блочной формы и знаков (alpha, -beta; beta, alpha)
            double L[9];
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    double sacc = 0;
                    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3; ++q) sacc += Ti[r * 3 + p] * Ai[p * 3 + q] * T[q * 3 + c];
                    L[r * 3 + c] = sacc;
                }
            const double lg = L[0], la = L[4], lb = L[7];
            if (std::fabs(L[1]) + std::fabs(L[2]) + std::fabs(L[3]) + std::fabs(L[6]) > 1e-10
                || std::fabs(L[8] - la) > 1e-10 || std::fabs(L[5] + lb) > 1e-10)
                throw std::runtime_error("Radau IIA: eigen-decomposition of A^-1 failed");
            const int N2 = 2 * N;
            o << "    /* RADAU5 simplified Newton (newton_full = 0): gamma " << fmtnum(lg) << ", alpha " << fmtnum(la)
              << ", beta " << fmtnum(lb) << " */\n";
            o << "    int rd_done = 0;\n";
            o << "    if (!nfull) {\n";
            o << "        const numb RT[9] = { ";
            for (int k = 0; k < 9; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(T[k]);
            o << " };\n        const numb RTI[9] = { ";
            for (int k = 0; k < 9; ++k) o << (k ? ", " : "") << "(numb)" << fmtnum(Ti[k]);
            o << " };\n";
            o << "        const numb rg = (numb)" << fmtnum(lg) << " / h, ra = (numb)" << fmtnum(la) << " / h, rb = (numb)"
              << fmtnum(lb) << " / h;\n";
            o << "        numb Zr[" << M << "], Gw[" << M << "], E1[" << N * N << "], E2[" << N2 * N2 << "];\n";
            o << "        int p1[" << N << "], p2[" << N2 << "], ndim, okn = 0;\n";
            // J в X
            o << "        for (l = 0; l < " << N << "; l++) Sv[l] = X[l];\n";
            if (ht) o << "        ucuda_ts = ucuda_t;\n";
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const std::string& d = JS[(size_t)i * N + j];
                    o << "        Js[" << i * N + j << "] = " << (d == "(numb)0.0" ? std::string("(numb)0.0") : "(" + d + ")") << ";\n";
                }
            o << "        for (l = 0; l < " << N << "; l++) for (m = 0; m < " << N << "; m++) {\n";
            o << "            mlt = (l == m && l < " << nx << ") ? (numb)1.0 : (numb)0.0;\n";
            o << "            E1[l * " << N << " + m] = rg * mlt - Js[l * " << N << " + m];\n";
            o << "            E2[l * " << N2 << " + m] = ra * mlt - Js[l * " << N << " + m];\n";
            o << "            E2[l * " << N2 << " + " << N << " + m] = -rb * mlt;\n";
            o << "            E2[(" << N << " + l) * " << N2 << " + m] = rb * mlt;\n";
            o << "            E2[(" << N << " + l) * " << N2 << " + " << N << " + m] = ra * mlt - Js[l * " << N << " + m];\n";
            o << "        }\n";
            auto lu = [&](const char* E, const char* P, const char* n) {
                o << "        ndim = " << n << ";\n";
                o << "        for (ik = 0; ik < ndim && !sing; ik++) {\n";
                o << "            ip = ik; mx = fabs(" << E << "[ik * ndim + ik]);\n";
                o << "            for (ir = ik + 1; ir < ndim; ir++) { av = fabs(" << E << "[ir * ndim + ik]); if (av > mx) { mx = av; ip = ir; } }\n";
                o << "            " << P << "[ik] = ip;\n";
                o << "            if (ip != ik) for (ic = ik; ic < ndim; ic++) { mlt = " << E << "[ik * ndim + ic]; " << E
                  << "[ik * ndim + ic] = " << E << "[ip * ndim + ic]; " << E << "[ip * ndim + ic] = mlt; }\n";
                o << "            dgn = " << E << "[ik * ndim + ik];\n";
                o << "            if (fabs(dgn) < (numb)1e-30) { sing = 1; break; }\n";
                o << "            for (ir = ik + 1; ir < ndim; ir++) {\n";
                o << "                mlt = " << E << "[ir * ndim + ik] / dgn; " << E << "[ir * ndim + ik] = mlt;\n";
                o << "                for (ic = ik + 1; ic < ndim; ic++) " << E << "[ir * ndim + ic] -= mlt * " << E << "[ik * ndim + ic];\n";
                o << "            }\n";
                o << "        }\n";
            };
            auto solve = [&](const char* E, const char* P, const char* n, const char* b) {
                o << "            ndim = " << n << ";\n";
                o << "            for (ik = 0; ik < ndim; ik++) {\n";
                o << "                ip = " << P << "[ik]; if (ip != ik) { mlt = " << b << "[ik]; " << b << "[ik] = " << b << "[ip]; " << b << "[ip] = mlt; }\n";
                o << "                for (ir = ik + 1; ir < ndim; ir++) " << b << "[ir] -= " << E << "[ir * ndim + ik] * " << b << "[ik];\n";
                o << "            }\n";
                o << "            for (ik = ndim - 1; ik >= 0; ik--) {\n";
                o << "                for (ic = ik + 1; ic < ndim; ic++) " << b << "[ik] -= " << E << "[ik * ndim + ic] * " << b << "[ic];\n";
                o << "                " << b << "[ik] /= " << E << "[ik * ndim + ik];\n";
                o << "            }\n";
            };
            o << "        sing = 0;\n";
            lu("E1", "p1", std::to_string(N).c_str());
            lu("E2", "p2", std::to_string(N2).c_str());
            o << "        for (ik = 0; ik < " << M << "; ik++) Zr[ik] = W[ik] - X[ik % " << N << "];\n";
            o << "        prevn = 1e300;\n";
            o << "        for (it = 0; it < nmax && !sing; it++) {\n";
            o << "            for (bj = 0; bj < 3; bj++) {\n";
            o << "                for (l = 0; l < " << N << "; l++) Sv[l] = X[l] + Zr[bj * " << N << " + l];\n";
            if (ht) o << "                ucuda_ts = ucuda_t + RC[bj] * h;\n";
            for (int i = 0; i < N; ++i) o << "                Fs[bj * " << N << " + " << i << "] = (" << fS[i] << ");\n";
            o << "            }\n";
            // Gw = (T^-1 x I) F - (Lambda x M) (T^-1 x I) Z / h  (Rv — T^-1 Z)
            o << "            for (l = 0; l < " << N << "; l++) {\n";
            o << "                for (bi = 0; bi < 3; bi++) {\n";
            o << "                    Gw[bi * " << N << " + l] = RTI[bi * 3] * Fs[l] + RTI[bi * 3 + 1] * Fs[" << N << " + l] + RTI[bi * 3 + 2] * Fs[" << 2 * N << " + l];\n";
            o << "                    Rv[bi * " << N << " + l] = RTI[bi * 3] * Zr[l] + RTI[bi * 3 + 1] * Zr[" << N << " + l] + RTI[bi * 3 + 2] * Zr[" << 2 * N << " + l];\n";
            o << "                }\n";
            o << "                if (l < " << nx << ") {\n";
            o << "                    Gw[l] -= rg * Rv[l];\n";
            o << "                    Gw[" << N << " + l] -= ra * Rv[" << N << " + l] - rb * Rv[" << 2 * N << " + l];\n";
            o << "                    Gw[" << 2 * N << " + l] -= rb * Rv[" << N << " + l] + ra * Rv[" << 2 * N << " + l];\n";
            o << "                }\n";
            o << "            }\n";
            solve("E1", "p1", std::to_string(N).c_str(), "Gw");
            o << "            {\n";
            o << "            numb* const G2 = Gw + " << N << ";\n";
            solve("E2", "p2", std::to_string(N2).c_str(), "G2");
            o << "            }\n";
            // dZ = (T x I) dW
            o << "            nrm = 0; xx = 0;\n";
            o << "            for (l = 0; l < " << N << "; l++) for (bi = 0; bi < 3; bi++) {\n";
            o << "                ik = bi * " << N << " + l;\n";
            o << "                mlt = RT[bi * 3] * Gw[l] + RT[bi * 3 + 1] * Gw[" << N << " + l] + RT[bi * 3 + 2] * Gw[" << 2 * N << " + l];\n";
            o << "                Rv[ik] = mlt;\n";
            o << "            }\n";
            o << "            for (ik = 0; ik < " << M << "; ik++) {\n";
            o << "                Zr[ik] += Rv[ik];\n";
            o << "                mlt = Wt[ik % " << N << "] * Rv[ik]; nrm += mlt * mlt;\n";
            o << "                mlt = Wt[ik % " << N << "] * (X[ik % " << N << "] + Zr[ik]); xx += mlt * mlt;\n";
            o << "            }\n";
            o << "            if (!(nrm == nrm)) break;\n";
            o << "            if (nrm < ntol * ntol || nrm < ntol * ntol * xx) { okn = 1; break; }\n";
            o << "            if (sizeof(numb) == 4 && nrm < (numb)3.6e-12f * xx) { okn = 1; break; }\n";
            o << "            if (it > 0 && nrm > prevn) break;   /* расходится: полный Ньютон */\n";
            o << "            prevn = nrm;\n";
            o << "        }\n";
            o << "        if (okn) { for (ik = 0; ik < " << M << "; ik++) W[ik] = X[ik % " << N << "] + Zr[ik]; rd_done = 1; }\n";
            o << "    }\n";
        }
        o << "    if (!rd_done) {\n";
        o << "    sing = 0; refresh = 1; prevn = 1e300;\n";
        o << "    for (it = 0; it < nmax; it++) {\n";
        // f, g и (при обновлении) якобианы в каждой стадии
        o << "        for (bj = 0; bj < 3; bj++) {\n";
        o << "            for (l = 0; l < " << N << "; l++) Sv[l] = W[bj * " << N << " + l];\n";
        if (ht) o << "            ucuda_ts = ucuda_t + RC[bj] * h;\n";
        for (int i = 0; i < N; ++i) o << "            Fs[bj * " << N << " + " << i << "] = (" << fS[i] << ");\n";
        o << "            if (refresh) {\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = JS[(size_t)i * N + j];
                o << "                Js[bj * " << N * N << " + " << i * N + j << "] = "
                  << (d == "(numb)0.0" ? std::string("(numb)0.0") : "(" + d + ")") << ";\n";
            }
        o << "            }\n";
        o << "        }\n";
        o << "        if (refresh) {\n";
        if (sparse) {
            for (const auto& kv : SL.id) {
                const int r = kv.first.first, c = kv.first.second;
                const int bi = r / N, l = r % N, bj = c / N, m = c % N;
                o << "            Asp[" << kv.second << "] = ";
                if (!SL.orig.count(kv.first)) { o << "(numb)0.0;\n"; continue; }
                const std::string js = "Js[" + std::to_string(bj * N * N + l * N + m) + "]";
                const bool jz = JS[(size_t)l * N + m] == "(numb)0.0";
                if (l < nx) {
                    if (bi == bj && l == m) o << "(numb)1.0" << (jz ? "" : " - h * RA[" + std::to_string(bi * 3 + bj) + "] * " + js);
                    else o << "-h * RA[" << bi * 3 + bj << "] * " << js;
                } else o << js;
                o << ";\n";
            }
            o << "            ucuda_dn = 0;\n";
            sparse_lu_factor(o, SL, "            ", "ucuda_dn");
            o << "            if (ucuda_dn) {\n";
        }
        {
        o << "            for (bi = 0; bi < 3; bi++) for (l = 0; l < " << N << "; l++)\n";
        o << "            for (bj = 0; bj < 3; bj++) for (m = 0; m < " << N << "; m++) {\n";
        o << "                ir = bi * " << N << " + l; ic = bj * " << N << " + m;\n";
        o << "                if (l < " << nx << ") Am[ir * " << M << " + ic] = (bi == bj && l == m ? (numb)1.0 : (numb)0.0)"
             " - h * RA[bi * 3 + bj] * Js[bj * " << N * N << " + l * " << N << " + m];\n";
        o << "                else Am[ir * " << M << " + ic] = bi == bj ? Js[bi * " << N * N << " + l * " << N << " + m] : (numb)0.0;\n";
        o << "            }\n";
        o << "            for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "                ip = ik; mx = fabs(Am[ik * " << M << " + ik]);\n";
        o << "                for (ir = ik + 1; ir < " << M << "; ir++) { av = fabs(Am[ir * " << M << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n";
        o << "                piv[ik] = ip;\n";
        o << "                if (ip != ik) for (ic = ik; ic < " << M << "; ic++) { mlt = Am[ik * " << M << " + ic]; Am[ik * " << M
          << " + ic] = Am[ip * " << M << " + ic]; Am[ip * " << M << " + ic] = mlt; }\n";
        o << "                dgn = Am[ik * " << M << " + ik];\n";
        o << "                if (fabs(dgn) < (numb)1e-30) { sing = 1; break; }\n";   // 1e-300 в float — ноль
        o << "                for (ir = ik + 1; ir < " << M << "; ir++) {\n";
        o << "                    mlt = Am[ir * " << M << " + ik] / dgn; Am[ir * " << M << " + ik] = mlt;\n";
        o << "                    for (ic = ik + 1; ic < " << M << "; ic++) Am[ir * " << M << " + ic] -= mlt * Am[ik * " << M << " + ic];\n";
        o << "                }\n";
        o << "            }\n";
        if (sparse) o << "            }   /* ucuda_dn */\n";
        o << "            refresh = 0;\n";
        o << "        }\n";
        }
        o << "        if (sing) break;\n";
        // невязка
        o << "        for (bi = 0; bi < 3; bi++) for (l = 0; l < " << N << "; l++) {\n";
        o << "            ir = bi * " << N << " + l;\n";
        o << "            if (l < " << nx << ") Rv[ir] = W[ir] - X[l] - h * (RA[bi * 3] * Fs[l] + RA[bi * 3 + 1] * Fs["
          << N << " + l] + RA[bi * 3 + 2] * Fs[" << 2 * N << " + l]);\n";
        o << "            else Rv[ir] = Fs[ir];\n";
        o << "        }\n";
        // решение
        if (sparse) {
            o << "        if (!ucuda_dn) {\n";
            sparse_lu_solve(o, SL, "Rv", "Dv", "        ");
            o << "        for (ik = 0; ik < " << M << "; ik++) Rv[ik] = Dv[ik];\n";
            o << "        } else {\n";
        }
        {
        o << "        for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "            ip = piv[ik]; if (ip != ik) { mlt = Rv[ik]; Rv[ik] = Rv[ip]; Rv[ip] = mlt; }\n";
        o << "            for (ir = ik + 1; ir < " << M << "; ir++) Rv[ir] -= Am[ir * " << M << " + ik] * Rv[ik];\n";
        o << "        }\n";
        o << "        for (ik = " << M - 1 << "; ik >= 0; ik--) {\n";
        o << "            for (ic = ik + 1; ic < " << M << "; ic++) Rv[ik] -= Am[ik * " << M << " + ic] * Rv[ic];\n";
        o << "            Rv[ik] /= Am[ik * " << M << " + ik];\n";
        o << "        }\n";
        if (sparse) o << "        }   /* ucuda_dn */\n";
        }
        o << "        nrm = 0; xx = 0;\n";
        o << "        for (ik = 0; ik < " << M << "; ik++) {\n";
        o << "            W[ik] -= Rv[ik];\n";
        o << "            mlt = Wt[ik % " << N << "] * Rv[ik]; nrm += mlt * mlt;\n";
        o << "            mlt = Wt[ik % " << N << "] * W[ik];  xx += mlt * mlt;\n";
        o << "        }\n";
        o << "        if (nrm < ntol * ntol || nrm < ntol * ntol * xx) break;\n";
        o << "        if (sizeof(numb) == 4 && nrm < (numb)3.6e-12f * xx) break;\n";
        o << "        if (nfull || nrm > (numb)0.25 * prevn) refresh = 1;\n";
        o << "        prevn = nrm;\n";
        o << "    }\n";
        o << "    }   /* !rd_done */\n";
        o << "    for (l = 0; l < " << N << "; l++) X[l] = W[" << 2 * N << " + l];\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_explicit_rk(const System& s, int S, const double* A, const double* b, const double* c) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const std::string tn = system_uses_time(s) ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        dae_require_index1(JS, N, nx, "explicit DAE methods");
        const int nslot = newton_slot(s);
        std::ostringstream o;
        o << "    numb Sv[" << N << "];\n";
        o << "    numb Kx[" << (nx * S > 0 ? nx * S : 1) << "];\n";
        o << "    numb Gz[" << (nz > 0 ? nz : 1) << "], Jz[" << (nz * nz > 0 ? nz * nz : 1) << "];\n";
        o << "    const numb ntol = a[" << nslot << "];\n";
        o << "    const int  nmax = (int)a[" << nslot + 1 << "];\n";
        o << "    int it, ik, ir, ic, ip;\n";
        o << "    numb mx, av, mlt, nrm, zz;\n";
        if (!tn.empty()) o << "    numb ucuda_ts;\n";
        for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
        // Ньютон по z над Sv (x фиксированы); Gauss с выбором главного в форме цикла.
        auto z_solve = [&](const std::string& tstage) {
            if (!tn.empty()) o << "    ucuda_ts = " << tstage << ";\n";
            o << "    for (it = 0; it < nmax; it++) {\n";
            for (int k = 0; k < nz; ++k) o << "        Gz[" << k << "] = (" << fS[nx + k] << ");\n";
            for (int i = 0; i < nz; ++i)
                for (int j = 0; j < nz; ++j)
                    o << "        Jz[" << i * nz + j << "] = (" << JS[(size_t)(nx + i) * N + nx + j] << ");\n";
            o << "        for (ik = 0; ik < " << nz << "; ik++) {\n"
                 "            ip = ik; mx = fabs(Jz[ik * " << nz << " + ik]);\n"
                 "            for (ir = ik + 1; ir < " << nz << "; ir++) { av = fabs(Jz[ir * " << nz << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
                 "            if (ip != ik) { for (ic = 0; ic < " << nz << "; ic++) { mlt = Jz[ik * " << nz << " + ic]; Jz[ik * " << nz << " + ic] = Jz[ip * " << nz << " + ic]; Jz[ip * " << nz << " + ic] = mlt; }"
                 " mlt = Gz[ik]; Gz[ik] = Gz[ip]; Gz[ip] = mlt; }\n"
                 "            if (fabs(Jz[ik * " << nz << " + ik]) < (numb)1e-300) break;\n"
                 "            for (ir = ik + 1; ir < " << nz << "; ir++) {\n"
                 "                mlt = Jz[ir * " << nz << " + ik] / Jz[ik * " << nz << " + ik];\n"
                 "                for (ic = ik; ic < " << nz << "; ic++) Jz[ir * " << nz << " + ic] -= mlt * Jz[ik * " << nz << " + ic];\n"
                 "                Gz[ir] -= mlt * Gz[ik];\n"
                 "            }\n"
                 "        }\n"
                 "        nrm = 0; zz = 0;\n"
                 "        for (ik = " << nz - 1 << "; ik >= 0; ik--) {\n"
                 "            for (ic = ik + 1; ic < " << nz << "; ic++) Gz[ik] -= Jz[ik * " << nz << " + ic] * Gz[ic];\n"
                 "            Gz[ik] /= Jz[ik * " << nz << " + ik];\n"
                 "            Sv[" << nx << " + ik] -= Gz[ik]; nrm += Gz[ik] * Gz[ik]; zz += Sv[" << nx << " + ik] * Sv[" << nx << " + ik];\n"
                 "        }\n"
                 "        if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n"
                 "    }\n";
        };
        char buf[64];
        auto cnum = [&](double v) { std::snprintf(buf, sizeof(buf), "(numb)%.17g", v); return std::string(buf); };
        for (int st = 0; st < S; ++st) {
            o << "    /* stage " << st << " */\n";
            if (st > 0)
                for (int l = 0; l < nx; ++l) {
                    o << "    Sv[" << l << "] = X[" << l << "]";
                    for (int j = 0; j < st; ++j)
                        if (A[st * S + j] != 0.0) o << " + h * " << cnum(A[st * S + j]) << " * Kx[" << j * nx + l << "]";
                    o << ";\n";
                }
            z_solve(c[st] == 0.0 ? std::string("ucuda_t") : "ucuda_t + " + cnum(c[st]) + " * h");
            for (int l = 0; l < nx; ++l) o << "    Kx[" << st * nx + l << "] = (" << fS[l] << ");\n";
        }
        o << "    /* x_{n+1}, then z_{n+1} from the constraint */\n";
        for (int l = 0; l < nx; ++l) {
            o << "    X[" << l << "] = X[" << l << "]";
            for (int j = 0; j < S; ++j) if (b[j] != 0.0) o << " + h * " << cnum(b[j]) << " * Kx[" << j * nx + l << "]";
            o << ";\n    Sv[" << l << "] = X[" << l << "];\n";
        }
        z_solve("ucuda_t + h");
        for (int k = 0; k < nz; ++k) o << "    X[" << nx + k << "] = Sv[" << nx + k << "];\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_ros2(const System& s) {
        const bool ht = system_uses_time(s);
        const int N = (int)s.vars.size(), nx = s.dae_nx;
        const auto F0 = rhs_over(s, "X", time_name(s)), FY = rhs_over(s, "Yv", time_name(s, "(ucuda_t + h)"));
        const auto J = jac_over(s, "X", time_name(s));
        std::vector<std::string> Ft;
        if (ht) {
            NameMap nm = build_namemap(s, "X", kStepTimeArg);
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                Ft.push_back(emit_to_str(pn_diff(p.parse(), "t"), nm));
            }
        }
        const double gam = 1.0 + 1.0 / std::sqrt(2.0);
        std::ostringstream o;
        o << "    /* Rosenbrock ROS2, gamma = 1 + 1/sqrt(2) */\n";
        o << "    const numb gh = (numb)" << fmtnum(gam) << " * h;\n";
        o << "    numb Am[" << N * N << "], K1[" << N << "], K2[" << N << "], Yv[" << N << "];\n";
        o << "    int piv[" << N << "];\n    int ik, ir, ic, ip;\n    numb mx, av, mlt;\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = J[(size_t)i * N + j];
                const bool diag_m = (i == j && i < nx);
                o << "    Am[" << i * N + j << "] = " << (diag_m ? "(numb)1.0" : "(numb)0.0");
                if (d != "(numb)0.0") o << " - gh * (" << d << ")";
                o << ";\n";
            }
        // LU с выбором главного (форма цикла: N мало, матрица одна на шаг)
        o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
             "        ip = ik; mx = fabs(Am[ik * " << N << " + ik]);\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) { av = fabs(Am[ir * " << N << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
             "        piv[ik] = ip;\n"
             "        if (ip != ik) for (ic = ik; ic < " << N << "; ic++) { mlt = Am[ik * " << N << " + ic]; Am[ik * " << N << " + ic] = Am[ip * " << N << " + ic]; Am[ip * " << N << " + ic] = mlt; }\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) {\n"
             "            mlt = Am[ir * " << N << " + ik] / Am[ik * " << N << " + ik];\n"
             "            Am[ir * " << N << " + ik] = mlt;\n"
             "            for (ic = ik + 1; ic < " << N << "; ic++) Am[ir * " << N << " + ic] -= mlt * Am[ik * " << N << " + ic];\n"
             "        }\n"
             "    }\n";
        auto solve = [&](const char* v) {
            o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
                 "        ip = piv[ik]; if (ip != ik) { mlt = " << v << "[ik]; " << v << "[ik] = " << v << "[ip]; " << v << "[ip] = mlt; }\n"
                 "        for (ir = ik + 1; ir < " << N << "; ir++) " << v << "[ir] -= Am[ir * " << N << " + ik] * " << v << "[ik];\n"
                 "    }\n"
                 "    for (ik = " << N - 1 << "; ik >= 0; ik--) {\n"
                 "        for (ic = ik + 1; ic < " << N << "; ic++) " << v << "[ik] -= Am[ik * " << N << " + ic] * " << v << "[ic];\n"
                 "        " << v << "[ik] /= Am[ik * " << N << " + ik];\n"
                 "    }\n";
        };
        if (ht) o << "    numb Ftv[" << N << "];\n";
        for (int i = 0; i < N && ht; ++i) o << "    Ftv[" << i << "] = gh * h * (" << Ft[i] << ");\n";
        for (int i = 0; i < N; ++i)
            o << "    K1[" << i << "] = h * (" << F0[i] << ")" << (ht ? " + Ftv[" + std::to_string(i) + "]" : "") << ";\n";
        solve("K1");
        for (int i = 0; i < N; ++i) o << "    Yv[" << i << "] = X[" << i << "] + K1[" << i << "];\n";
        for (int i = 0; i < N; ++i) {
            o << "    K2[" << i << "] = h * (" << FY[i] << ")";
            if (i < nx) o << " - (numb)2.0 * K1[" << i << "]";
            if (ht) o << " - Ftv[" << i << "]";
            o << ";\n";
        }
        solve("K2");
        for (int i = 0; i < N; ++i)
            o << "    X[" << i << "] = X[" << i << "] + (numb)1.5 * K1[" << i << "] + (numb)0.5 * K2[" << i << "];\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
const RosTab& ros3p_tab() {
        static const RosTab t = { "ROS3P (Lang-Verwer), order 3", 3, 7.886751345948129e-01,
            { 0.0, 1.0, 1.0 },
            { 7.886751345948129e-01, -2.113248654051871e-01, -1.077350269189626e+00 },
            { 2.0, 5.773502691896258e-01, 4.226497308103742e-01 },
            { {}, { 1.267949192431123e+00 }, { 1.267949192431123e+00, 0.0 } },
            { {}, { -1.607695154586736e+00 }, { -3.464101615137755e+00, -1.732050807568877e+00 } } };
        return t;
    }
}

namespace ucuda::equations_detail {
const RosTab& rodas4_tab() {
        static const double A51 = 0.1221224509226641e+01, A52 = 0.6019134481288629e+01,
                            A53 = 0.1253708332932087e+02, A54 = -0.6878860361058950e+00;
        static const RosTab t = { "Rodas4 (Hairer-Wanner), order 4", 6, 0.25,
            { 0.0, 0.386, 0.21, 0.63, 1.0, 1.0 },
            { 0.25, -0.1043, 0.1035, -0.362e-01, 0.0, 0.0 },
            { A51, A52, A53, A54, 1.0, 1.0 },
            { {},
              { 0.1544000000000000e+01 },
              { 0.9466785280815826e+00, 0.2557011698983284e+00 },
              { 0.3314825187068521e+01, 0.2896124015972201e+01, 0.9986419139977817e+00 },
              { A51, A52, A53, A54 },
              { A51, A52, A53, A54, 1.0 } },
            { {},
              { -0.5668800000000000e+01 },
              { -0.2430093356833875e+01, -0.2063599157091915e+00 },
              { -0.1073529058151375e+00, -0.9594562251023355e+01, -0.2047028614809616e+02 },
              { 0.7496443313967647e+01, -0.1024680431464352e+02, -0.3399990352819905e+02, 0.1170890893206160e+02 },
              { 0.8083246795921522e+01, -0.7981132988064893e+01, -0.3152159432874371e+02, 0.1631930543123136e+02,
                -0.6058818238834054e+01 } },
            // вложенное решение порядка 3 — вход 6-й стадии, E = u_6 (rodas.f: ERR = AK6)
            { A51, A52, A53, A54, 1.0, 0.0 },
            { 0.1012623508344586e+02, -0.7487995877610167e+01, -0.3480091861555747e+02,
              -0.7992771707568823e+01, 0.1025137723295662e+01, 0.0 },
            { -0.6762803392801253e+00, 0.6087714651680015e+01, 0.1643084320892478e+02,
              0.2476722511418386e+02, -0.6594389125716872e+01, 0.0 },
            3 };
        return t;
    }
}

namespace ucuda::equations_detail {
std::string dae_rosenbrock(const System& s, const RosTab& T, bool emb ) {
        const bool ht = system_uses_time(s);
        const int N = (int)s.vars.size(), nx = s.dae_nx, S = T.S;
        const auto J = jac_over(s, "X", time_name(s));
        std::vector<std::string> Ft;
        if (ht) {
            NameMap nm = build_namemap(s, "X", kStepTimeArg);
            for (int i = 0; i < N; ++i) {
                Parser p(s.rhs[i], s.latex);
                Ft.push_back(emit_to_str(pn_diff(p.parse(), "t"), nm));
            }
        }
        std::ostringstream o;
        o << "    /* Rosenbrock " << T.title << " */\n";
        o << "    const numb ihg = (numb)1 / ((numb)" << fmtnum(T.gamma) << " * h), ih = (numb)1 / h;\n";
        o << "    numb Am[" << N * N << "], U[" << S * N << "], Yv[" << N << "];\n";
        o << "    int piv[" << N << "];\n    int ik, ir, ic, ip;\n    numb mx, av, mlt;\n";
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                const std::string& d = J[(size_t)i * N + j];
                o << "    Am[" << i * N + j << "] = " << ((i == j && i < nx) ? "ihg" : "(numb)0.0");
                if (d != "(numb)0.0") o << " - (" << d << ")";
                o << ";\n";
            }
        o << "    for (ik = 0; ik < " << N << "; ik++) {\n"
             "        ip = ik; mx = fabs(Am[ik * " << N << " + ik]);\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) { av = fabs(Am[ir * " << N << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
             "        piv[ik] = ip;\n"
             "        if (ip != ik) for (ic = ik; ic < " << N << "; ic++) { mlt = Am[ik * " << N << " + ic]; Am[ik * " << N << " + ic] = Am[ip * " << N << " + ic]; Am[ip * " << N << " + ic] = mlt; }\n"
             "        for (ir = ik + 1; ir < " << N << "; ir++) {\n"
             "            mlt = Am[ir * " << N << " + ik] / Am[ik * " << N << " + ik];\n"
             "            Am[ir * " << N << " + ik] = mlt;\n"
             "            for (ic = ik + 1; ic < " << N << "; ic++) Am[ir * " << N << " + ic] -= mlt * Am[ik * " << N << " + ic];\n"
             "        }\n"
             "    }\n";
        if (ht) {
            o << "    numb Ftv[" << N << "];\n";
            for (int i = 0; i < N; ++i) o << "    Ftv[" << i << "] = h * (" << Ft[i] << ");\n";
        }
        for (int st = 0; st < S; ++st) {
            const std::string base = "U + " + std::to_string(st * N);
            o << "    /* stage " << st + 1 << " */\n";
            for (int i = 0; i < N; ++i) {
                o << "    Yv[" << i << "] = X[" << i << "]";
                for (int j = 0; j < st; ++j)
                    if (T.a[st][j] != 0.0) o << " + (numb)(" << fmtnum(T.a[st][j]) << ") * U[" << j * N + i << "]";
                o << ";\n";
            }
            const std::string tst = T.alpha[st] == 0.0 ? std::string() : "(ucuda_t + (numb)(" + fmtnum(T.alpha[st]) + ") * h)";
            const auto F = rhs_over(s, "Yv", tst.empty() ? time_name(s) : time_name(s, tst.c_str()));
            for (int i = 0; i < N; ++i) {
                o << "    U[" << st * N + i << "] = (" << F[i] << ")";
                if (i < nx)
                    for (int j = 0; j < st; ++j)
                        if (T.c[st][j] != 0.0) o << " + (numb)(" << fmtnum(T.c[st][j]) << ") * ih * U[" << j * N + i << "]";
                if (ht && T.gt[st] != 0.0) o << " + (numb)(" << fmtnum(T.gt[st]) << ") * Ftv[" << i << "]";
                o << ";\n";
            }
            o << "    {\n        numb* v = " << base << ";\n";
            o << "        for (ik = 0; ik < " << N << "; ik++) {\n"
                 "            ip = piv[ik]; if (ip != ik) { mlt = v[ik]; v[ik] = v[ip]; v[ip] = mlt; }\n"
                 "            for (ir = ik + 1; ir < " << N << "; ir++) v[ir] -= Am[ir * " << N << " + ik] * v[ik];\n"
                 "        }\n"
                 "        for (ik = " << N - 1 << "; ik >= 0; ik--) {\n"
                 "            for (ic = ik + 1; ic < " << N << "; ic++) v[ik] -= Am[ik * " << N << " + ic] * v[ic];\n"
                 "            v[ik] /= Am[ik * " << N << " + ik];\n"
                 "        }\n    }\n";
        }
        for (int i = 0; i < N; ++i) {
            o << "    " << (emb ? "Y[" : "X[") << i << "] = X[" << i << "]";
            for (int j = 0; j < S; ++j)
                if (T.m[j] != 0.0) o << " + (numb)(" << fmtnum(T.m[j]) << ") * U[" << j * N + i << "]";
            o << ";\n";
        }
        if (emb) {
            o << "    (void)F0;\n";
            for (int i = 0; i < N; ++i) {
                o << "    E[" << i << "] = (numb)0";
                for (int j = 0; j < S; ++j)
                    if (T.m[j] != T.mhat[j]) o << " + (numb)(" << fmtnum(T.m[j] - T.mhat[j]) << ") * U[" << j * N + i << "]";
                o << ";\n";
            }
            o << "    for (ik = 0; ik < " << S * N << "; ik++) W[ik] = U[ik];\n";
            const auto fy = rhs_over(s, "Y", time_name(s, "(ucuda_t + h)"));
            for (int i = 0; i < N; ++i) o << "    F1[" << i << "] = (" << fy[(size_t)i] << ");\n";
        }
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string ros_dprep_body(const System& s, const RosTab& T) {
        const int N = (int)s.vars.size();
        std::ostringstream o;
        o << "    (void)F0; (void)F1; (void)a; (void)h;\n";
        for (int i = 0; i < N; ++i) {
            o << "    D[" << i << "] = X[" << i << "]; D[" << N + i << "] = Y[" << i << "];\n";
            o << "    D[" << 2 * N + i << "] = (numb)0";
            for (int j = 0; j < T.S; ++j)
                if (T.d2[j] != 0.0) o << " + (numb)(" << fmtnum(T.d2[j]) << ") * W[" << j * N + i << "]";
            o << ";\n    D[" << 3 * N + i << "] = (numb)0";
            for (int j = 0; j < T.S; ++j)
                if (T.d3[j] != 0.0) o << " + (numb)(" << fmtnum(T.d3[j]) << ") * W[" << j * N + i << "]";
            o << ";\n";
        }
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string ros_deval_body(const System& s) {
        std::ostringstream o;
        o << "    int N = " << s.vars.size() << ";\n";
        o << "    for (int l = 0; l < N; ++l) {\n";
        o << "        const numb s1 = (numb)1 - th;\n";
        o << "        Yo[l] = D[l] * s1 + th * (D[N + l] + s1 * (D[2 * N + l] + th * D[3 * N + l]));\n";
        o << "    }\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_cd(const System& s, bool yoshida) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const int MM = 1 + nz;
        // Ньютон по неизвестным U (индексы в Sv): невязки R, якобиан Jm (|U| x |U|), форма цикла.
        auto newton = [&](std::ostringstream& o, const std::vector<int>& U, const std::vector<std::string>& R,
                          const std::vector<std::string>& Jm) {
            const int M = (int)U.size();
            o << "        for (it = 0; it < nmax; it++) {\n";
            for (int k = 0; k < M; ++k) o << "            Gm[" << k << "] = (" << R[k] << ");\n";
            for (int k = 0; k < M * M; ++k) o << "            Jmm[" << k << "] = (" << Jm[k] << ");\n";
            o << "            for (ik = 0; ik < " << M << "; ik++) {\n"
                 "                ip = ik; mx = fabs(Jmm[ik * " << M << " + ik]);\n"
                 "                for (ir = ik + 1; ir < " << M << "; ir++) { av = fabs(Jmm[ir * " << M << " + ik]); if (av > mx) { mx = av; ip = ir; } }\n"
                 "                if (ip != ik) { for (ic = 0; ic < " << M << "; ic++) { mlt = Jmm[ik * " << M << " + ic]; Jmm[ik * " << M << " + ic] = Jmm[ip * " << M << " + ic]; Jmm[ip * " << M << " + ic] = mlt; }"
                 " mlt = Gm[ik]; Gm[ik] = Gm[ip]; Gm[ip] = mlt; }\n"
                 "                if (fabs(Jmm[ik * " << M << " + ik]) < (numb)1e-300) break;\n"
                 "                for (ir = ik + 1; ir < " << M << "; ir++) {\n"
                 "                    mlt = Jmm[ir * " << M << " + ik] / Jmm[ik * " << M << " + ik];\n"
                 "                    for (ic = ik; ic < " << M << "; ic++) Jmm[ir * " << M << " + ic] -= mlt * Jmm[ik * " << M << " + ic];\n"
                 "                    Gm[ir] -= mlt * Gm[ik];\n"
                 "                }\n"
                 "            }\n"
                 "            for (ik = " << M - 1 << "; ik >= 0; ik--) {\n"
                 "                for (ic = ik + 1; ic < " << M << "; ic++) Gm[ik] -= Jmm[ik * " << M << " + ic] * Gm[ic];\n"
                 "                Gm[ik] /= Jmm[ik * " << M << " + ik];\n"
                 "            }\n"
                 "            nrm = 0; zz = 0;\n";
            for (int k = 0; k < M; ++k)
                o << "            Sv[" << U[k] << "] -= Gm[" << k << "]; nrm += Gm[" << k << "] * Gm[" << k << "]; zz += Sv["
                  << U[k] << "] * Sv[" << U[k] << "];\n";
            o << "            if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n"
                 "        }\n";
        };
        dae_require_index1(JS, N, nx, "DAE CD");
        std::vector<int> Uz;
        std::vector<std::string> Rz, Jz;
        for (int k = 0; k < nz; ++k) { Uz.push_back(nx + k); Rz.push_back(fS[nx + k]); }
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nz; ++j) Jz.push_back(JS[(size_t)(nx + i) * N + nx + j]);

        auto cd_step = [&](std::ostringstream& o) {
            o << "    numb Sv[" << N << "], Gm[" << MM << "], Jmm[" << MM * MM << "];\n";
            o << "    numb ucuda_xo, mx, av, mlt, nrm, zz;\n    int it, ik, ir, ic, ip;\n";
            if (ht) o << "    numb ucuda_ts;\n";
            o << "    const numb h1 = a[0] * h, h2 = h - h1;\n";
            o << "    const numb ntol = a[" << nslot << "];\n    const int  nmax = (int)a[" << nslot + 1 << "];\n";
            for (int i = 0; i < N; ++i) o << "    Sv[" << i << "] = X[" << i << "];\n";
            o << "    /* D: explicit, forward order; z re-solved after every update */\n";
            for (int i = 0; i < nx; ++i) {
                if (ht) o << "    ucuda_ts = ucuda_t;\n";
                o << "    Sv[" << i << "] = Sv[" << i << "] + h1 * (" << fS[i] << ");\n";
                newton(o, Uz, Rz, Jz);
            }
            o << "    /* D*: implicit, reverse order; (x_i, z) by one Newton */\n";
            for (int i = nx - 1; i >= 0; --i) {
                if (ht) o << "    ucuda_ts = ucuda_t + h;\n";
                o << "    ucuda_xo = Sv[" << i << "];\n";
                std::vector<int> U = { i };
                std::vector<std::string> R = { "Sv[" + std::to_string(i) + "] - ucuda_xo - h2 * (" + fS[i] + ")" };
                for (int k = 0; k < nz; ++k) { U.push_back(nx + k); R.push_back(fS[nx + k]); }
                std::vector<std::string> Jm;
                for (int r = 0; r < MM; ++r)
                    for (int c = 0; c < MM; ++c) {
                        const int row = r == 0 ? i : nx + r - 1, col = U[c];
                        const std::string& d = JS[(size_t)row * N + col];
                        if (r == 0) Jm.push_back(std::string(c == 0 ? "(numb)1.0" : "(numb)0.0") + " - h2 * (" + d + ")");
                        else        Jm.push_back(d);
                    }
                newton(o, U, R, Jm);
            }
            for (int i = 0; i < N; ++i) o << "    X[" << i << "] = Sv[" << i << "];\n";
        };
        std::ostringstream o;
        if (!yoshida) { cd_step(o); return o.str(); }
        const double cb = std::cbrt(2.0), g1 = 1.0 / (2.0 - cb), g2 = -cb / (2.0 - cb);
        o << "    /* CD + Yoshida-4: CD(g1 h), CD(g2 h), CD(g1 h); 4th order at a[0] = 0.5 */\n";
        o << "    const numb ucuda_h0 = h;\n";
        if (ht) o << "    const numb ucuda_t0 = ucuda_t;\n";
        const double gs[3] = { g1, g2, g1 }, off[3] = { 0.0, g1, g1 + g2 };
        for (int k = 0; k < 3; ++k) {
            o << "    {\n    const numb h = ucuda_h0 * (numb)" << fmtnum(gs[k]) << ";\n";
            if (ht) o << "    const numb ucuda_t = ucuda_t0 + ucuda_h0 * (numb)" << fmtnum(off[k]) << ";\n";
            std::ostringstream b;
            cd_step(b);
            o << b.str() << "    }\n";
        }
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string dae_cd2(const System& s) {
        const int N = (int)s.vars.size(), nx = s.dae_nx, nz = N - nx;
        const bool ht = system_uses_time(s);
        const std::string tn = ht ? "ucuda_ts" : "";
        const auto fS = rhs_over(s, "Sv", tn);
        const auto JS = jac_over(s, "Sv", tn);
        const int nslot = newton_slot(s);
        const int NZ = nz > 0 ? nz : 1;
        std::ostringstream o;
        o << "    /* DAE CD, index 2: z constant per half-step, chosen so that g holds at its end */\n";
        o << "    numb Sv[" << N << "], Xb[" << (nx > 0 ? nx : 1) << "], zA[" << NZ << "], zB[" << NZ << "], Rv[" << NZ
          << "], R0[" << NZ << "], Jf[" << NZ * NZ << "];\n";
        o << "    numb ucuda_xo, ucuda_r, ucuda_d, mx, mlt, zt, dk, nrm, zz;\n    int it, jt, ik, ir, ic, ip;\n";
        if (ht) o << "    numb ucuda_ts;\n";
        o << "    const numb h1 = a[0] * h, h2 = h - h1;\n";
        o << "    const numb ntol = a[" << nslot << "];\n    const int  nmax = (int)a[" << nslot + 1 << "];\n";
        o << "    const numb ucuda_fd = sizeof(numb) == 4 ? (numb)3e-4 : (numb)1.5e-8;\n";
        // проход полушага при данной z: x из Xb, результат в Sv, невязка связи в Rv
        o << "    auto ucuda_sweep = [&](int star, const numb* zv) {\n";
        o << "        for (ik = 0; ik < " << nx << "; ik++) Sv[ik] = Xb[ik];\n";
        o << "        for (ik = 0; ik < " << nz << "; ik++) Sv[" << nx << " + ik] = zv[ik];\n";
        o << "        if (!star) {\n";
        if (ht) o << "            ucuda_ts = ucuda_t;\n";
        for (int i = 0; i < nx; ++i) o << "            Sv[" << i << "] = Sv[" << i << "] + h1 * (" << fS[i] << ");\n";
        if (ht) o << "            ucuda_ts = ucuda_t + h1;\n";
        o << "        } else {\n";
        if (ht) o << "            ucuda_ts = ucuda_t + h;\n";
        for (int i = nx - 1; i >= 0; --i) {
            const std::string& d = JS[(size_t)i * N + i];
            o << "            ucuda_xo = Sv[" << i << "];\n";
            o << "            for (jt = 0; jt < nmax; jt++) {\n";
            o << "                ucuda_r = Sv[" << i << "] - ucuda_xo - h2 * (" << fS[i] << ");\n";
            o << "                ucuda_d = (numb)1.0" << (d == "(numb)0.0" ? std::string() : " - h2 * (" + d + ")") << ";\n";
            o << "                ucuda_r /= ucuda_d; Sv[" << i << "] -= ucuda_r;\n";
            o << "                if (fabs(ucuda_r) <= ntol * ((numb)1.0 + fabs(Sv[" << i << "]))) break;\n";
            o << "            }\n";
        }
        o << "        }\n";
        for (int k = 0; k < nz; ++k) o << "        Rv[" << k << "] = (" << fS[(size_t)(nx + k)] << ");\n";
        o << "    };\n";
        // Ньютон по z полушага с разностным якобианом
        o << "    auto ucuda_zsolve = [&](int star, numb* zv) {\n";
        o << "        for (it = 0; it < nmax; it++) {\n";
        o << "            ucuda_sweep(star, zv);\n";
        o << "            for (ik = 0; ik < " << nz << "; ik++) R0[ik] = Rv[ik];\n";
        o << "            for (ic = 0; ic < " << nz << "; ic++) {\n";
        o << "                zt = zv[ic]; dk = ucuda_fd * ((numb)1.0 + fabs(zt)); zv[ic] = zt + dk;\n";
        o << "                ucuda_sweep(star, zv);\n";
        o << "                for (ir = 0; ir < " << nz << "; ir++) Jf[ir * " << nz << " + ic] = (Rv[ir] - R0[ir]) / dk;\n";
        o << "                zv[ic] = zt;\n";
        o << "            }\n";
        o << "            for (ik = 0; ik < " << nz << "; ik++) {\n"
             "                ip = ik; mx = fabs(Jf[ik * " << nz << " + ik]);\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) if (fabs(Jf[ir * " << nz << " + ik]) > mx) { mx = fabs(Jf[ir * " << nz << " + ik]); ip = ir; }\n"
             "                if (ip != ik) { for (ic = ik; ic < " << nz << "; ic++) { mlt = Jf[ik * " << nz << " + ic]; Jf[ik * " << nz << " + ic] = Jf[ip * " << nz << " + ic]; Jf[ip * " << nz << " + ic] = mlt; } mlt = R0[ik]; R0[ik] = R0[ip]; R0[ip] = mlt; }\n"
             "                if (!(mx > (numb)0)) break;\n"
             "                for (ir = ik + 1; ir < " << nz << "; ir++) {\n"
             "                    mlt = Jf[ir * " << nz << " + ik] / Jf[ik * " << nz << " + ik];\n"
             "                    for (ic = ik; ic < " << nz << "; ic++) Jf[ir * " << nz << " + ic] -= mlt * Jf[ik * " << nz << " + ic];\n"
             "                    R0[ir] -= mlt * R0[ik];\n"
             "                }\n"
             "            }\n";
        o << "            nrm = 0; zz = 0;\n";
        o << "            for (ik = " << nz - 1 << "; ik >= 0; ik--) {\n"
             "                for (ic = ik + 1; ic < " << nz << "; ic++) R0[ik] -= Jf[ik * " << nz << " + ic] * R0[ic];\n"
             "                R0[ik] /= Jf[ik * " << nz << " + ik];\n"
             "                zv[ik] -= R0[ik]; nrm += R0[ik] * R0[ik]; zz += zv[ik] * zv[ik];\n"
             "            }\n";
        o << "            if (!(nrm == nrm)) break;\n";
        o << "            if (nrm < ntol * ntol || nrm < ntol * ntol * zz) break;\n";
        o << "        }\n";
        o << "        ucuda_sweep(star, zv);\n";
        o << "    };\n";
        for (int l = 0; l < nx; ++l) o << "    Xb[" << l << "] = X[" << l << "];\n";
        for (int k = 0; k < nz; ++k) o << "    zA[" << k << "] = X[" << nx + k << "];\n";
        o << "    ucuda_zsolve(0, zA);\n";
        for (int l = 0; l < nx; ++l) o << "    Xb[" << l << "] = Sv[" << l << "];\n";
        for (int k = 0; k < nz; ++k) o << "    zB[" << k << "] = zA[" << k << "];\n";
        o << "    ucuda_zsolve(1, zB);\n";
        for (int l = 0; l < nx; ++l) o << "    X[" << l << "] = Sv[" << l << "];\n";
        for (int k = 0; k < nz; ++k)
            o << "    X[" << nx + k << "] = (numb)0.5 * (zA[" << k << "] + zB[" << k << "]);\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_implicit_euler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::Euler); }
}

namespace ucuda::equations_detail {
std::string scheme_implicit_midpoint(const System& s) { return scheme_implicit_common(s, ImplicitKind::Midpoint); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_ieuler(const System& s)    { return scheme_implicit_common(s, ImplicitKind::ComplexEuler); }
}

namespace ucuda::equations_detail {
void emit_diag_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                               const std::string& v, const std::string& x,
                               const NameMap& nm, const char* hs, const char* sty,
                               const std::string& saved,
                               PrepList* prep ,
                               const std::vector<std::string>* state_vars ,
                               const std::string& prep_name ) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);

        if (linear && prep && !pn_is_zero(coef) && pn_free_of_vars(coef, *state_vars)) {
            char dsgn = peel_sign(coef, '-', '+');
            std::string coef_c = wrap_paren(emit_to_str(coef, nm), needs_paren_after_mul(coef));
            prep->emplace_back(prep_name, "(numb)1 / (1 " + std::string(1, dsgn) + " "
                                          + mul_step(coef, coef_c, hs) + ")");
            if (pn_is_zero(rem))
                o << "    " << x << " = " << x << " * " << prep_name << ";\n";
            else {
                char nsgn = peel_sign(rem, '+', '-');
                std::string rem_c = wrap_paren(emit_to_str(rem, nm), needs_paren_after_mul(rem));
                o << "    " << x << " = (" << x
                  << " " << nsgn << " " << mul_step(rem, rem_c, hs)
                  << ") * " << prep_name << ";\n";
            }
            return;
        }

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
}

namespace ucuda::equations_detail {
void fp_collect_terms(const PN& n, bool neg, std::vector<std::pair<bool, PN>>& out) {
        if (n && n->kind == Node::Add) { fp_collect_terms(n->a, neg, out); fp_collect_terms(n->b, neg, out); return; }
        if (n && n->kind == Node::Sub) { fp_collect_terms(n->a, neg, out); fp_collect_terms(n->b, !neg, out); return; }
        if (n && n->kind == Node::Neg) { fp_collect_terms(n->a, !neg, out); return; }
        out.emplace_back(neg, n);
    }
}

namespace ucuda::equations_detail {
PN fp_sum_terms(const std::vector<std::pair<bool, PN>>& t) {
        PN acc;
        for (const auto& e : t) {
            if (!acc) acc = e.first ? pn_neg(e.second) : e.second;
            else      acc = e.first ? pn_sub(acc, e.second) : pn_add(acc, e.second);
        }
        return acc;   // nullptr -- слагаемых нет
    }
}

namespace ucuda::equations_detail {
void fp_split_terms(const PN& f, const std::string& v, PN& r, PN& g) {
        std::vector<std::pair<bool, PN>> terms, tr, tg;
        fp_collect_terms(f, false, terms);
        for (const auto& e : terms) (pn_contains_var(e.second, v) ? tg : tr).push_back(e);
        r = fp_sum_terms(tr);
        g = fp_sum_terms(tg);
    }
}

namespace ucuda::equations_detail {
std::string fp_factor(const PN& e, const NameMap& nm) {
        const bool atom = e && (e->kind == Node::Sym || e->kind == Node::Call
                                || (e->kind == Node::Num && e->num >= 0.0));
        return wrap_paren(emit_to_str(e, nm), !atom);
    }
}

namespace ucuda::equations_detail {
void emit_fp_implicit_eq(std::ostringstream& o, const PN& rhs_ast,
                             const std::string& v, const std::string& x,
                             const NameMap& nm, const char* hs,
                             const std::string& sfx, bool fast, PrepList* prep,
                             const std::vector<std::string>* state_vars,
                             const std::string& prep_name) {
        PN coef, rem;
        const bool linear = cd_try_extract_linear(rhs_ast, v, coef, rem);
        if (linear && (fast || pn_is_zero(coef))) {
            emit_diag_implicit_eq(o, rhs_ast, v, x, nm, hs, "numb", "x" + sfx + "_cd",
                                  fast ? prep : nullptr, state_vars, prep_name);
            return;
        }
        PN r, g;
        if (linear) { r = pn_is_zero(rem) ? nullptr : rem; g = coef; }
        else        fp_split_terms(rhs_ast, v, r, g);
        const std::string xs = "x" + sfx + "_fp";
        o << "    numb " << xs << " = " << x;
        if (r) o << " + " << hs << " * " << fp_factor(r, nm);
        o << ";\n";
        if (linear) {
            const std::string cs = "c" + sfx + "_fp";
            o << "    numb " << cs << " = " << hs << " * " << fp_factor(g, nm) << ";\n";
            o << "    for (int cdfp_k = 0; cdfp_k < cdfp_n; ++cdfp_k)\n";
            o << "        " << x << " = " << xs << " + " << cs << " * " << x << ";\n";
        }
        else {
            o << "    for (int cdfp_k = 0; cdfp_k < cdfp_n; ++cdfp_k)\n";
            o << "        " << x << " = " << xs << " + " << hs << " * " << fp_factor(g, nm) << ";\n";
        }
    }
}

namespace ucuda::equations_detail {
std::string scheme_cd_common(const System& s, CdKind kind, bool core_only ,
                                 bool flip_halves , bool fast ,
                                 bool fp ) {
        // fast -- "CD fast": только вещественный CD. h1, h2 и обратные
        // знаменатели, не зависящие от состояния, идут в блок предрасчёта
        // (см. krs_split_prep в codegen.hpp).
        // fp -- "CD FP" / "CD fast FP": неявная половина простыми итерациями
        // (emit_fp_implicit_eq), их число -- a[M+1] (cd_fp_slot).
        if (fast && kind != CdKind::Real)
            throw std::runtime_error("CD fast: only the real CD is supported");
        if (fp && (kind != CdKind::Real || flip_halves))
            throw std::runtime_error("CD FP: only the real CD {0-1} is supported");
        PrepList prep;
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
        // Явное время (вещественный CD, CD {1-0}, CD FP): явный полушаг — в начале своего
        // отрезка, неявный — в конце. {0-1}: t_n и t_n + h; {1-0}: оба в t_n + h1. Без t
        // в системе обе таблицы имён совпадают с nm, и текст прежний.
        const bool time_cd = !cx && !fast && system_uses_time(s);
        const bool flip_cd = flip_halves || (kind == CdKind::Cx4ss10);
        const NameMap nm_exp = time_cd ? build_namemap(s, stv, flip_cd ? "(ucuda_t + h1)" : "ucuda_t") : nm;
        const NameMap nm_imp = time_cd ? build_namemap(s, stv, flip_cd ? "(ucuda_t + h1)" : "(ucuda_t + h)") : nm;
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
        if (kind == CdKind::Real && fast) {
            prep.emplace_back("h1", "h * a[0]");
            prep.emplace_back("h2", "h * (1 - a[0])");
        }
        else if (kind == CdKind::Real) {
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
            o << "    ucmplx cp = ucmplx(h * ((numb)1 / (numb)6), -h * ((numb)1 / (numb)6));   // (1-i)h/6\n";
            o << "    ucmplx cq = ucmplx(h * ((numb)1 / (numb)3),  h * ((numb)1 / (numb)6));   // (2+i)h/6\n";
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
                o << "    ucmplx g  = ucmplx((numb)0.5 * h,  h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx((numb)0.5 * h, -h * ccd_im);\n";
                cgs = { "g", "gc" };
            }
            else if (kind == CdKind::Cx4s4) {
                // (gamma/2, conj/2, conj/2, gamma/2) — палиндром.
                o << "    ucmplx g  = ucmplx((numb)0.25 * h,  (numb)0.5 * h * ccd_im);\n";
                o << "    ucmplx gc = ucmplx((numb)0.25 * h, -(numb)0.5 * h * ccd_im);\n";
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
        // "CD FP": число итераций из a[M+1]; объявляется, только если хоть одно
        // уравнение действительно уходит на итерации (у CD fast FP на системе,
        // линейной по своим переменным, тело совпадает с CD fast).
        if (fp) {
            bool need = false;
            for (int i = 0; i < N && !need; ++i) {
                PN coef, rem;
                const bool lin = cd_try_extract_linear(rhs_ast[i], s.vars[i], coef, rem);
                need = !(lin && (fast || pn_is_zero(coef)));
            }
            if (need)
                o << "    const int cdfp_n = (int)a[" << cd_fp_slot(s) << "];\n";
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
                  << step << " * (" << emit_to_str(rhs_ast[i], nm_exp) << ");\n";
        };
        auto emit_implicit = [&](const char* step) {
            for (int i = N - 1; i >= 0; --i)
                if (fp)
                    emit_fp_implicit_eq(o, rhs_ast[i], s.vars[i],
                                        stv + ("[" + std::to_string(i) + "]"), nm_imp,
                                        step, std::to_string(i) + sfx,
                                        fast, &prep, &s.vars,
                                        "cdf_r" + std::to_string(i) + sfx);
                else
                    emit_diag_implicit_eq(o, rhs_ast[i], s.vars[i],
                                          stv + ("[" + std::to_string(i) + "]"), nm_imp,
                                          step, sty, "x" + std::to_string(i) + "_cd" + sfx,
                                          fast ? &prep : nullptr, &s.vars,
                                          "cdf_r" + std::to_string(i) + sfx);
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

        if (fast) {
            std::ostringstream full;
            full << kKrsPrepBegin << "\n";
            for (const auto& c : prep)
                full << "    const numb " << c.first << " = " << c.second << ";\n";
            full << kKrsPrepEnd << "\n" << o.str();
            return full.str();
        }
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_cd(const System& s)          { return scheme_cd_common(s, CdKind::Real); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd(const System& s)  { return scheme_cd_common(s, CdKind::Cx); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd4(const System& s) { return scheme_cd_common(s, CdKind::Cx4); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd4_s3(const System& s) { return scheme_cd_common(s, CdKind::Cx4s3); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd4_s4(const System& s) { return scheme_cd_common(s, CdKind::Cx4s4); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd4_ss01(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss01); }
}

namespace ucuda::equations_detail {
std::string scheme_complex_cd4_ss10(const System& s) { return scheme_cd_common(s, CdKind::Cx4ss10); }
}

namespace ucuda::equations_detail {
std::string scheme_semi_midpoint(const System& s, bool implicit_stage) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        // Явное время: явная стадия (Эйлер-Кромер на h1) — в t, диагонально-неявная — в t + h1
        // (неявный Эйлер на h1); корректор берёт f по значениям стадии — тоже в t + h1.
        NameMap nm = build_namemap(s, "X", time_name(s, implicit_stage ? "(ucuda_t + h1)" : nullptr));
        NameMap nm_corr = build_namemap(s, "X", time_name(s, "(ucuda_t + h1)"));
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
              << emit_to_str(rhs_ast[i], nm_corr) << ");\n";
        for (int i = 0; i < N; ++i)
            o << "    X[" << i << "] = X1[" << i << "];\n";
        return o.str();
    }
}

namespace ucuda::equations_detail {
std::string scheme_semp(const System& s) { return scheme_semi_midpoint(s, false); }
}

namespace ucuda::equations_detail {
std::string scheme_simp(const System& s) { return scheme_semi_midpoint(s, true); }
}

namespace ucuda::equations_detail {
std::string scheme_d(const System& s) {
        if (s.vars.size() != s.rhs.size())
            throw std::runtime_error("vars/rhs size mismatch");
        int N = (int)s.vars.size();

        NameMap nm = build_namemap(s, "X", time_name(s, "(ucuda_t + h)"));   // неявный: f в t + h
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
