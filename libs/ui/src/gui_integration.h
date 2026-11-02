#pragma once
// Схемы и блок Integration: таблица встроенных схем, выбор схемы, цели свипа, адаптивный шаг,
// библиотека регуляторов, Newton, рассылка настроек.
#include "gui_common.h"

namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_needs_compiled_body(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes);


[[nodiscard]] bool scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes);


[[nodiscard]] bool scheme_uses_cd_fp_iters(const std::string& scheme_name);


[[nodiscard]] bool scheme_uses_newton(const std::string& scheme_name);


static const char* const kCdFpItersTip =
    "Simple iterations n of the CD FP implicit half-step (integer >= 1).\n"
    "Passed to the kernel at run time as a[M+1], like s in a[0]:\n"
    "changing it does not recompile. Iteration residual O(h^(n+1)):\n"
    "order 2 needs n >= 2.";

void set_scheme_masks(const AppModel* m);

struct BuiltinScheme {
    const char* name;
    int         order;
    // Флаг "генерировать эту схему" в AppModel. Через него System tab рисует
    // чекбоксы циклом — раньше список схем был выписан там ещё раз руками, и
    // Complex Implicit Euler, добавленный только в одну из копий, не попадал
    // ни под "Select all", ни под "Clear all".
    bool SystemEditor::*flag;
    // Подсказка чекбокса; nullptr — схема говорит сама за себя.
    const char* tooltip;
    // Своя группа вместо "Order N" (nullptr — группа по порядку). Схемы одной
    // группы стоят в таблице подряд: шапка печатается на смене заголовка.
    const char* group = nullptr;
};


[[nodiscard]] std::string scheme_group_title(const BuiltinScheme& b);


static const char* const kMapSchemeName = "Map";


[[nodiscard]] bool scheme_is_map(const std::string& name);


static const BuiltinScheme kBuiltinSchemes[] = {
    { "Euler",                    1, &AppModel::scheme_euler, nullptr },
    { "Euler-Cromer",             1, &AppModel::scheme_cromer, nullptr },
    { "D",                        1, &AppModel::scheme_dmethod,
      "Diagonally-implicit first-order method: the stage of SIMP\n"
      "(and the Phi* half-step of CD) taken as a step of its own,\n"
      "on the full h. X[i] = X + h*f_i(X) is solved for its own\n"
      "variable (analytically when linear in it, 4 fixed-point\n"
      "iterations otherwise), forward order, so each equation\n"
      "already sees the new X[0..i-1].\n"
      "The diagonally-implicit twin of Euler-Cromer: no Jacobian,\n"
      "no Newton, and cross terms stay outside the solve." },
    { "Implicit Euler",           1, &AppModel::scheme_ieuler,
      "X_next = X + h*f(X_next), solved by Newton on a\n"
      "symbolically differentiated Jacobian.\n"
      "Order 1, L-stable: unlike the explicit schemes it stays\n"
      "bounded on stiff systems at steps where RK4 diverges.\n"
      "Needs a differentiable RHS (no floor/ceil/fmod)." },
    { "Explicit Midpoint",        2, &AppModel::scheme_midpoint, nullptr },
    { "Implicit Midpoint",        2, &AppModel::scheme_imidpoint,
      "Solved for the stage Y = (X + X_next)/2:\n"
      "Y = X + (h/2)*f(Y), then X_next = 2*Y - X.\n"
      "Order 2 (measured 2.00 on Lorenz), A-stable, symmetric\n"
      "and symplectic. Same Newton solver as Implicit Euler." },
    { "CD",                       2, &AppModel::scheme_cd, nullptr },
    { "CD_{1-0}",                   2, &AppModel::scheme_cd10,
      "Variant {1-0} of CD: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 2 and symmetry as in CD.\n"
      "Measured: Lorenz 1.03x, Rossler 1.6x larger error than CD.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CD fast",                  2, &AppModel::scheme_cdfast,
      "CD with a precompute stage: h1, h2 and every reciprocal\n"
      "denominator 1/(1 - h2*coef) whose coef does not depend on\n"
      "the state are computed once per trajectory, before the\n"
      "first step; the implicit half-step then multiplies instead\n"
      "of dividing. Same method and order as CD; the result\n"
      "differs from CD only at round-off level." },
    { "CD FP",                    2, &AppModel::scheme_cdfp,
      "CD whose implicit half-step is solved by n simple\n"
      "iterations for EVERY equation (no analytic division);\n"
      "n is the \"FP iterations n\" field of each tab, next to s\n"
      "(a run-time value in a[M+1], like s in a[0]).\n"
      "Terms of f_i free of x_i are added once before the\n"
      "iterations; a linear f_i = c*x_i + r iterates\n"
      "x = x0 + (h2*c)*x. Iteration residual O(h^(n+1)):\n"
      "order 2 needs n >= 2." },
    { "CD fast FP",               2, &AppModel::scheme_cdfastfp,
      "CD fast (precomputed reciprocal denominators for\n"
      "linear equations) + n simple iterations, with the\n"
      "x_i-free terms hoisted, for equations nonlinear in\n"
      "their own variable. Order 2 needs n >= 2." },
    { "Complex CD",               2, &AppModel::scheme_ccd,
      "CD with complex half-steps:\n"
      "h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6,\n"
      "s = a[0] is the same symmetry coefficient as in CD.\n"
      "The step is evaluated in complex arithmetic; only Re is kept.\n"
      "At s = 0.5 (default) the half-steps are conjugate and order is 2." },
    { "Complex CD_{1-0}",           2, &AppModel::scheme_ccd10,
      "Variant {1-0} of Complex CD: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 2 and symmetry as in Complex CD.\n"
      "Error about the same as Complex CD.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "Complex Implicit Euler",   2, &AppModel::scheme_cieuler,
      "Composition of TWO implicit Euler steps with conjugate\n"
      "complex steps: tau1 = h*(s + i/2), tau2 = h*(1 - s - i/2),\n"
      "s = a[0], the same symmetry slot as CD. Each half-step is a\n"
      "full Newton solve on the symbolic Jacobian, in complex\n"
      "arithmetic; Re is taken once at the end.\n"
      "The coefficients are forced, not tuned: composing a first-\n"
      "order method to second order needs tau1 + tau2 = h AND\n"
      "tau1^2 + tau2^2 = 0, whose only solution is (1 +- i)/2.\n"
      "Order 2 at s = 0.5 only; elsewhere the h^2 term keeps a real\n"
      "part that Re does not remove and the order drops to 1." },
    { "SEMP",                     2, &AppModel::scheme_semp,
      "Semi-explicit midpoint. Stage: X~[i] = X[i] + h1*f_i(X),\n"
      "component by component in forward order (each equation\n"
      "already sees the new X[0..i-1]), h1 = s*h. Corrector:\n"
      "X_next = X + h*f(X~). Two RHS passes, nothing to solve.\n"
      "Order 2 at s = a[0] = 0.5 only: the expansion carries\n"
      "s*h^2*F'F against h^2/2*F'F, so other s give order 1." },
    { "SIMP",                     2, &AppModel::scheme_simp,
      "Semi-implicit midpoint. Same corrector as SEMP, but the\n"
      "stage is diagonally implicit: each equation is solved for\n"
      "its own variable (analytically when linear in it, 4 fixed-\n"
      "point iterations otherwise), exactly as Phi* does in CD.\n"
      "No Jacobian, no Newton. Order 2 at s = a[0] = 0.5 only." },
    { "RK4",                      4, &AppModel::scheme_rk4, nullptr },
    { "Complex CD4",              4, &AppModel::scheme_ccd4,
      "Composition of TWO symmetric CDs with complex steps:\n"
      "pass 1 uses step gamma*h, pass 2 uses conj(gamma)*h,\n"
      "gamma = 1/2 + i*sqrt(3)/6. Re is taken once at the end.\n"
      "Order 4 (measured 4.00 on Lorenz and Rossler).\n"
      "Requires s = a[0] = 0.5: for other s the inner CD is\n"
      "asymmetric and the order drops to first." },
    { "Complex CD4_{1-0}",          4, &AppModel::scheme_ccd4_10,
      "Variant {1-0} of Complex CD4: each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in Complex CD4.\n"
      "Measured: 4.0x smaller error on Lorenz, 1.5x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (o4s3)",              4, &AppModel::scheme_ccd4s3,
      "SYMMETRIC order-4 composition of THREE symmetric CDs:\n"
      "steps (alpha, 1-2*alpha, alpha)*h, palindromic, hence\n"
      "self-adjoint as a complex composition.\n"
      "alpha = 1/(2 - 2^(1/3)*exp(2*pi*i/3))\n"
      "      = 0.3243964040201712 + 0.1345862724908067i.\n"
      "Complex CD4 itself cannot be made symmetric: a palindrome\n"
      "of two stages forces g1 = g2, and then sum g = 1 and\n"
      "sum g^3 = 0 are incompatible -- three stages is the minimum.\n"
      "Same cubic as Yoshida's real triple jump, but the complex\n"
      "branch: no backward substep (min Re = 0.324) and an error\n"
      "constant 215x smaller (0.0247 against 5.29).\n"
      "Re is taken once at the end of the step; the symmetry defect\n"
      "that leaves is O(h^10), against O(h^8) for Complex CD4.\n"
      "Error expands in EVEN powers of h (measured: Richardson on\n"
      "h^2 gives order 6), so it is a legitimate extrapolation base.\n"
      "Requires s = a[0] = 0.5, like every CD-family scheme.\n"
      "Costs 3 CD passes against 2; at equal total work the error\n"
      "matches Complex CD4 to within 4%." },
    { "CCD4 (o4s3)_{1-0}",          4, &AppModel::scheme_ccd4s3_10,
      "Variant {1-0} of CCD4 (o4s3): each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in CCD4 (o4s3).\n"
      "Measured: 3.8x smaller error on Lorenz, 2.4x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (o4s4)",              4, &AppModel::scheme_ccd4s4,
      "SYMMETRIC order-4 composition of FOUR symmetric CDs:\n"
      "steps (gamma/2, conj/2, conj/2, gamma/2)*h, palindromic.\n"
      "This is Complex CD4 symmetrized by composing it with its own\n"
      "adjoint on the half step; the palindromic four-stage order\n"
      "conditions have exactly this solution, up to conjugation.\n"
      "Smallest error constant of the three (0.00759), but 4 CD\n"
      "passes per step -- at equal total work it is identical to\n"
      "Complex CD4 and 4% better than CCD4 (o4s3).\n"
      "Same properties otherwise: symmetry defect O(h^10) after Re,\n"
      "even-power error expansion, requires s = a[0] = 0.5." },
    { "CCD4 (o4s4)_{1-0}",          4, &AppModel::scheme_ccd4s4_10,
      "Variant {1-0} of CCD4 (o4s4): each CD pass runs the IMPLICIT\n"
      "half first (h1 = s*h, reverse order), then the explicit one\n"
      "(h2, forward order). The same composition over the adjoint\n"
      "basic method: order 4 and symmetry as in CCD4 (o4s4).\n"
      "Measured: 4.0x smaller error on Lorenz, 1.5x on Rossler.\n"
      "The plain name is the {0-1} variant (explicit half first)." },
    { "CCD4 (s-split)_{0-1}",     4, &AppModel::scheme_ccd4ss01,
      "Dual of Complex CD4: TWO CD passes with REAL steps h/2,\n"
      "the complex part moved into the split inside each CD:\n"
      "pass 1: h1 = (1-i)h/6, h2 = (2+i)h/6  (s1 = (1-i)/3),\n"
      "pass 2: h1 = (2+i)h/6, h2 = (1-i)h/6  (s2 = 1 - s1).\n"
      "Both h^3 coefficients are purely imaginary and Re removes\n"
      "them: order 4 (measured 4.0 on Lorenz and Rossler).\n"
      "Same cost as Complex CD4; the error constant depends on\n"
      "the problem (2.9x smaller on Lorenz, 3.1x larger on\n"
      "Rossler). Slightly larger stability regions.\n"
      "NOT symmetric after Re (defect O(h^6)): Richardson on h^2\n"
      "gives order 5, not 6 (ExtrZ restores 6 and 8).\n"
      "{0-1}: explicit half first in each pass, as in every CD here.\n"
      "The split is part of the method: a[0] is NOT read." },
    { "CCD4 (s-split)_{1-0}",     4, &AppModel::scheme_ccd4ss10,
      "Same half-steps as CCD4 (s-split)_{0-1}, but each pass runs\n"
      "the IMPLICIT half first (reverse order), then the explicit one:\n"
      "E*_a, E_b | E*_b, E_a with a = (1-i)/6, b = (2+i)/6.\n"
      "This is the same composition over the adjoint basic method:\n"
      "the [N1,N2] coefficient flips sign but stays imaginary, so the\n"
      "order is still 4 (measured 4.0 on Lorenz and Rossler).\n"
      "Error constant differs from {0-1} and depends on the problem:\n"
      "10x larger on Lorenz, 2.8x smaller on Rossler.\n"
      "a[0] is NOT read." },
    { "RK45",                     5, &AppModel::scheme_rk45,
      "Dormand-Prince 5(4)7M: the ode45 / scipy RK45 pair.\n"
      "With a fixed step: 6 stages, order 5 (the 5th-order\n"
      "solution advances the step; the 4th-order one and the\n"
      "FSAL 7th stage are only for error control).\n"
      "Coefficients are the exact published fractions." },
    { "DOPRI78",                  8, &AppModel::scheme_dopri78,
      "Dormand-Prince RK8(7)13M, 13 stages, order 8 (the 7th-order\n"
      "embedded solution is computed but not used).\n"
      "Coefficients refined from the published fractions: the values\n"
      "the authors chose (c2..c8, c10, c12 = c13 = 1, rows 2-6,\n"
      "a75, a76, b13 = 1/4, b^12 = 2/45) are kept exact, the rest\n"
      "solve all 200 + 85 order conditions to ~1e-65 (corrections\n"
      "<= 1.4e-17) and are stored to ~62 digits: double uses the\n"
      "nearest double, dd/qd in the Order tab get the full precision.\n"
      "No coefficient error floor in dd/qd (was ~1e-17).\n"
      "Note: the leading error term is minimized (the M in 13M), so\n"
      "the observed order at practical steps is ~9-10, reaching 8\n"
      "only at small h." },
    { "DOPRI78 (legacy)",         8, &AppModel::scheme_dopri78_legacy,
      "RK8(7)13M with the published rational coefficients of\n"
      "Prince & Dormand (1981), taken as is (no refinement).\n"
      "They satisfy the order conditions only to ~1e-18: in double\n"
      "this is indistinguishable from DOPRI78, in dd the error\n"
      "floors at ~1e-17. Kept for comparison." },
    { "DOP853",                   8, &AppModel::scheme_dop853,
      "Hairer's DOP853: explicit RK of order 8 with 12 stages,\n"
      "embedded 5th- and 3rd-order error estimators and a\n"
      "7th-order dense output (the scipy DOP853 method).\n"
      "With a fixed step only the 8th-order solution is used.\n"
      "Coefficients refined from the published 30-digit values\n"
      "by solving all 200 order conditions to 3e-78: dd/qd in\n"
      "the Order tab get the full precision." },
    { "GBS (n=2)",               2, &AppModel::scheme_gbs,
      "Gragg's modified midpoint with n = 2 substeps of h/2, the\n"
      "base of the Gragg-Bulirsch-Stoer family:\n"
      "z1 = X + (h/2) f(X),  z2 = X + h f(z1),\n"
      "X_next = (z1 + z2 + (h/2) f(z2)) / 2   (Gragg smoothing).\n"
      "Order 2, 3 RHS evaluations per step. NOT self-adjoint as a\n"
      "one-step method (defect O(h^6)), so Extr(GBS (n=2)|1,2,3)\n"
      "reaches only 5; the GBS 2-4-6 ... entries below keep ONE\n"
      "leapfrog chain per stage and reach the full 2K.", "GBS" },
    { "GBS 2-4",                 4, &AppModel::scheme_gbs24,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 4 (measured 4.1 on Rossler), 8 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6",               6, &AppModel::scheme_gbs246,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 6 (measured 6.1 on Rossler), 15 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8",             8, &AppModel::scheme_gbs2468,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 8, 24 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8-10",          10, &AppModel::scheme_gbs246810,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8, 10: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 10, 35 RHS evaluations per step.", "GBS" },
    { "GBS 2-4-6-8-10-12",       12, &AppModel::scheme_gbs24681012,
      "Gragg-Bulirsch-Stoer, stages n = 2, 4, 6, 8, 10, 12: stage k is ONE\n"
      "leapfrog chain of n substeps h/n started by Euler and closed\n"
      "by Gragg smoothing; stages run from the same X and are\n"
      "combined with Richardson weights in 1/n^2.\n"
      "Order 12, 48 RHS evaluations per step.", "GBS" },
};


static constexpr int kBuiltinSchemeCount =
    (int)(sizeof(kBuiltinSchemes) / sizeof(kBuiltinSchemes[0]));

[[nodiscard]] std::vector<const char*> c_str_list(const std::vector<std::string>& v);


bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick = {},
                              const std::vector<std::string>* enabled_builtins = nullptr,
                              AppModel* bc = nullptr,
                              bool is_map = false,
                              const std::vector<std::string>* wrapper_schemes = nullptr,
                              bool broadcast = true);


void draw_sweep_target_combo(const char* label,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars,
                                    int& par_index, bool& over_var, int& var_index,
                                    bool& over_h,
                                    bool* other_over_h = nullptr,
                                    bool note_when_empty = false,
                                    float width = kComboW,
                                    bool allow_s = false,
                                    AppModel* bc = nullptr,
                                    BroadcastField which = BroadcastField::SweepTarget,
                                    bool allow_h = true,
                                    int* ad_axis = nullptr,
                                    const AdaptiveSettings* ad = nullptr);


enum AdaptiveUiOpt {
    kAdUiRaw    = 1,    // выбор сетки вывода: равномерная / узлы шага
    kAdUiInterp = 2,    // интерполяция пиков на узлах шага
    kAdUiMaxPts = 4,    // потолок числа узлов (Analysis)
    kAdUiRenorm = 8,    // перенормировка LLE/LS
    kAdUiMinMax = 16,   // Metrics: min/max по интерполированным экстремумам и в Fixed
};


void draw_ctrl_library_window(AppModel& m);


bool draw_adaptive_block(const char* id, AdaptiveSettings& a, const std::string& scheme,
                                int opts, bool is_map, AppModel* bc = nullptr);


bool draw_newton_fields(NewtonSettings& n, float w, const char* id = "");


struct IntegrationFields {
    std::string* h           = nullptr;
    std::string* symmetry_s  = nullptr;
    std::string* cd_fp_iters = nullptr;   // CD FP: число простых итераций (a[M+1])
    NewtonSettings* newton   = nullptr;   // неявные схемы: Ньютон (a[M+2..M+4])
    std::string* t_max       = nullptr;
    std::string* transient   = nullptr;
    std::string* pre_scaller = nullptr;
    std::string* max_value   = nullptr;
    // Адаптивный шаг: nullptr — у вкладки только постоянный шаг (DFT, FastSync,
    // Network, Stability). adaptive_opts — AdaptiveUiOpt.
    AdaptiveSettings* adaptive = nullptr;
    int adaptive_opts = 0;
};


bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f,
                                   AppModel* bc = nullptr,
                                   bool is_map = false);

[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) ;


[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) ;

std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) ;


void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) ;


std::vector<std::pair<std::string, std::string>> user_ctrl_list() ;


std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values);
}
