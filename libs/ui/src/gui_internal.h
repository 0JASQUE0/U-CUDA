#pragma once
#include "ui_resources.h"
#include "ucuda/platform/platform_paths.h"

// Общее для файлов интерфейса (gui*.cpp): включения, типы, константы и функции, которыми
// пользуются несколько вкладок. Получено механическим разносом gui.cpp; определения функций — в
// gui_*.cpp по вкладкам (gui_widgets.cpp — общие виджеты).
#include "ucuda/ui/gui.h"
#include "ucuda/equations/dae_ir.h"
#include "ucuda/application/compute_services.h" // проверки, файлы схем, нетлист: без cuda / numerics
#include <fstream>
#include "imgui.h"
#include "imgui_internal.h"   // DockBuilder* API — needed for Custom Workspace per-tab dockspaces.
#include "implot.h"
#include "implot3d.h"
#include "ucuda/plotting/plot_renderer.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/plotting/plot_view_2d.h"
#include "ucuda/plotting/plot_view_3d.h"
#include "ucuda/plotting/heatmap_view.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/domain/num_parse.h"   // parse_num — единый разбор числовых полей
#include "ucuda/plotting/digit_input.h" // DigitInput::ComputeStep — ↑/↓ по значению в списке (atol)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


// Именованные константы вместо магических чисел
// Значение, которым все хитмапы помечают «данных нет» (расходящаяся ячейка).
// HeatmapView красит его в серый; одно и то же число в colored-1D, DFT и Basins.

namespace ucuda::ui_detail {
static constexpr double kSentinelNoData = 999.0;


constexpr int    kColormapTurbo  = slancm_id(168);


static constexpr int    kErrorBoxLines  = 12;


static constexpr int    kRendererEvictFrames = 600;


static constexpr double kBasinsPhaseDebounceSec = 0.3;


static constexpr double kCustomSliderDebounceSec = 0.5;


static constexpr double kLibraryCacheTtlSec = 10.0;


static constexpr float  kFieldW = 120.0f;


static constexpr float  kComboW = 160.0f;


std::string get_exe_dir_with_sep();


ImVec4 ic_base_color(int ic_index);


ImVec4 shade_of(ImVec4 base, int vi, int nv);


ImVec4 ic_var_shade(int ic_index, int vi, int nv);


[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars);


[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars);


std::vector<char>& input_scratch(const std::string& str, size_t extra);


bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size);


void draw_resize_handle(const char* id, float& height,
                               float min_h = 40.0f, float max_h = 1200.0f);


void draw_latex_preview(const char* id, const std::string& src, ImVec2 size);


void draw_rhs_preview(const AppModel& model);


bool InputTextStr(const char* label, std::string& str, float width = 0.0f);


bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width = 0.0f);


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


[[nodiscard]] std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h = false);


[[nodiscard]] std::string label_identity(const std::string& s);


void refresh_auto_labels(AppModel& model);


[[nodiscard]] inline double parse_ratio_or(const std::string& v, double def) {
    // Round-trip через numb: интерфейс обязан показывать и отдавать в расчёт
    // РОВНО то число, которое увидит ядро. Сегодня numb == double и это no-op,
    // но если numb станет float — подпись оси, кламп fix-значения и привязка
    // крестика к сетке не начнут врать относительно посчитанного.
    return (double)(numb)parse_num(v, def);
}


[[nodiscard]] inline int parse_int_or(const std::string& v, int def) {
    return parse_num_int(v, def);
}


bool InputNumStr(const char* label, std::string& str, float width = 0.0f,
                        const std::function<void()>& ctx_menu = {},
                        bool allow_empty = false);


bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width = 0.0f);


void do_undo(AppModel& m);


void do_redo(AppModel& m);


void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value,
                                 const std::function<void()>& extra = {});


void set_field_library(SystemLibrary* lib);


void set_scheme_masks(const AppModel* m);


bool field_restore_menu_items(const AppModel* model, BroadcastField f, const std::string& name,
                              std::map<std::string, std::string>& values, const std::vector<std::string>& names);


bool field_restore_popup(const char* popup_id, const AppModel* model, BroadcastField f, const std::string& name,
                         std::map<std::string, std::string>& values, const std::vector<std::string>& names);


struct BuiltinScheme {
    const char* name;
    int         order;
    // Флаг "генерировать эту схему" в AppModel. Через него System tab рисует
    // чекбоксы циклом — раньше список схем был выписан там ещё раз руками, и
    // Complex Implicit Euler, добавленный только в одну из копий, не попадал
    // ни под "Select all", ни под "Clear all".
    bool AppModel::*flag;
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


struct ModeTab {
    const char*       id;
    const char*       label;
    AppModel::AppMode mode;
    bool              always;
};


static const ModeTab kModeTabs[] = {
    { "Library",    "Library",        AppModel::AppMode::Library,    true  },
    { "Analysis",   "Phase analysis", AppModel::AppMode::Analysis,   false },
    { "Parametric", "Parametric",     AppModel::AppMode::Parametric, false },
    { "Dft1D",      "1D DFT",         AppModel::AppMode::Dft1D,      false },
    { "Basins",     "Basins",         AppModel::AppMode::Basins,     false },
    { "FastSync",   "Fast Synchro",   AppModel::AppMode::FastSync,   false },
    { "Custom",     "Custom",         AppModel::AppMode::Custom,     false },
    { "Order",      "Order",          AppModel::AppMode::Order,      false },
    { "Network",    "Network",        AppModel::AppMode::Network,    false },
    { "Neural",     "Neural",         AppModel::AppMode::Neural,     false },
    { "Settings",   "Settings",       AppModel::AppMode::Settings,   true  },
};


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


void draw_param_tree(ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table = false,
                     const std::function<void(const std::vector<std::string>&)>& group_menu = {});


void draw_param_groups_window(AppModel& m, const std::vector<std::string>& params,
                              const std::vector<std::string>& vars, const char* id);


[[nodiscard]] std::string field_label_auto(const std::string& name);


[[nodiscard]] std::string field_label(const FieldLabels* labels, const std::string& name);


void draw_field_name(const std::string& name, const FieldLabels* labels);


void set_field_latex(bool on);


void set_field_labels(const FieldLabels* labels);


bool var_combo_begin(const char* id, const std::string& cur, bool cur_is_name = true);


void var_combo_preview();


bool var_selectable(const std::string& name, bool selected);


[[nodiscard]] std::string plot_var_name(const std::string& name);


void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note = nullptr,
                                  AppModel* bc = nullptr,
                                  BroadcastField field = BroadcastField::Param,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten = {});


void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint);


void draw_error_box(const char* id, const std::string& err,
                           int lines = kErrorBoxLines);


void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write);


struct ViewRangeSlots {
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    bool*        y_manual = nullptr;
};


struct ViewRangeTarget {
    std::string  name;                  // подпись в меню, когда целей несколько
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    // Необязательный флаг "Y задан вручную" (colored-1D у Bif): без него
    // записанные ymin/ymax остались бы лежать мёртвым грузом при авто-Y.
    bool*        y_manual = nullptr;
    // Тип диаграммы — по нему работает Ctrl+Shift+T. "Parametric 2D" общий у
    // Bif/LLE/LS: у них совпадает и смысл осей, и кодировка цели свипа, поэтому
    // приблизив 2D-бифуркацию осмысленно подтянуть 2D LLE и 2D LS. У Basins и
    // FastSync оси — сетка НУ, у DFT по Y частоты; они в свои типы и попадают.
    // Пусто у целей, собранных на месте: тип заполняет только общий перечислитель.
    std::string kind;
    std::string group;   // подпись для списка галочек: "Bifurcation 2D", ...
    // Как найти ЭТУ ЖЕ цель потом. Нужно отмене: между Ctrl+T и Ctrl+Z вкладку
    // диаграммы могут закрыть, вектор конфигов переедет, и указатели выше
    // повиснут. Лямбда ищет конфиг по индексу и сверяет label; если его больше
    // нет или на его месте другой — возвращает нули, и откат этого поля просто
    // не делается. Пусто = цель незаписываемая (см. view_range_bound).
    std::function<ViewRangeSlots()> relocate;
};


template <class T>
[[nodiscard]] static ViewRangeTarget vr_target(std::vector<T>& vec, int idx,
        std::string T::*x_lo, std::string T::*x_hi,
        std::string T::*y_lo = nullptr, std::string T::*y_hi = nullptr,
        bool T::*y_manual = nullptr) {
    ViewRangeTarget t;
    if (idx < 0 || idx >= (int)vec.size()) return t;   // цели нет — пустая
    t.relocate = [&vec, idx, label_id = label_identity(vec[(size_t)idx].label),
                  x_lo, x_hi, y_lo, y_hi, y_manual]() {
        ViewRangeSlots s;
        if (idx < 0 || idx >= (int)vec.size()) return s;
        T& c = vec[(size_t)idx];
        // Сверяем ИМЯ ОСИ, а не подпись целиком: диапазон в автоподписи 1D-диаграммы
        // меняет сама эта команда (см. label_identity).
        if (label_identity(c.label) != label_id) return s;   // на этом индексе другой конфиг
        s.x_lo = &(c.*x_lo); s.x_hi = &(c.*x_hi);
        if (y_lo)     s.y_lo = &(c.*y_lo);
        if (y_hi)     s.y_hi = &(c.*y_hi);
        if (y_manual) s.y_manual = &(c.*y_manual);
        return s;
    };
    const ViewRangeSlots live = t.relocate();
    t.name     = vec[(size_t)idx].label;
    t.x_lo     = live.x_lo;
    t.x_hi     = live.x_hi;
    t.y_lo     = live.y_lo;
    t.y_hi     = live.y_hi;
    t.y_manual = live.y_manual;
    return t;
}


[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo = nullptr, std::string* y_hi = nullptr);


void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model);


void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model);


[[nodiscard]] bool plot_window_active();


void draw_label_rename(const char* label_with_id, std::string& label,
                              float width = 200.0f);


struct TabBarResult {
    int active    = -1;
    int to_remove = -1;
};


void track_dock_selection(AppModel& model);


TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select = -1);


struct RunAllGroup {
    const char* title   = nullptr;   // заголовок секции; nullptr — без него
    const char* pick_id = "";        // неизменная часть ID чекбокса ("pbd_", …)
    int         n       = 0;
    std::function<std::string(int)> label;
    std::function<void(int)>        enqueue;   // положить элемент i в очередь
};


void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued);


enum class ParamPlotKind { Bifurcation, LLE, LS, Metrics };


void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind);


bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body);


void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip);


bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix);


void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd);


template <class Cfg>
static void draw_continuation_device_block(Cfg& c, const char* id, bool blocked,
                                           bool device_locked) {
    const std::string sfx = id;
    if (blocked) {
        c.continuation = false;
        c.continuation_reverse = false;
    }

    ImGui::BeginDisabled(blocked);
    bool cont = c.continuation;
    if (ImGui::Checkbox(("Continuation##" + sfx).c_str(), &cont)) {
        c.continuation = cont;
        // При включении continuation по умолчанию уходим на CPU (GPU-ветка там
        // однопоточная), при выключении возвращаемся на GPU. В обе стороны это
        // только дефолт: радио ниже активно, и выбор пользователя держится.
        c.use_gpu = !cont;
    }
    if (c.continuation) {
        ImGui::SameLine();
        int dir = c.continuation_reverse ? 1 : 0;
        ImGui::RadioButton(("forward##"  + sfx).c_str(), &dir, 0); ImGui::SameLine();
        ImGui::RadioButton(("backward##" + sfx).c_str(), &dir, 1);
        c.continuation_reverse = (dir == 1);
    }
    ImGui::EndDisabled();
    if (blocked)
        ImGui::TextDisabled("(continuation: param- or h-sweep only, not IC, not 2D)");

    int dev = c.use_gpu ? 0 : 1;
    ImGui::BeginDisabled(device_locked);
    ImGui::RadioButton(("GPU##dev" + sfx).c_str(), &dev, 0); ImGui::SameLine();
    ImGui::RadioButton(("CPU##dev" + sfx).c_str(), &dev, 1);
    ImGui::EndDisabled();
    if (!device_locked) c.use_gpu = (dev == 0);
    if (c.continuation && c.use_gpu) {
        ImGui::SameLine();
        ImGui::TextDisabled("(continuation on the GPU is single-threaded, CPU is faster)");
    }
    if (!c.use_gpu) {
        std::string why;
        if (!cpu_compiler_available(&why)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(no compiler: %s)", why.c_str());
        }
    }
}


void apply_snap_x(Plot2DView& view, double lo, double hi, int n);


[[nodiscard]] double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation);


struct SweepAxisMember {
    int    kind      = 0;      // 0 = параметр, 1 = переменная (IC), 2 = шаг h
    int    index     = -1;     // индекс в params/vars (для kind==2 не нужен)
    bool   log_scale = false;
};


void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars);


template <class Cfg>
static void configure_sweep_x_axis_from(Plot2DView& view,
                                        const std::vector<int>& members,
                                        const std::vector<Cfg>& cfgs,
                                        const std::vector<std::string>& params,
                                        const std::vector<std::string>& vars) {
    std::vector<SweepAxisMember> out;
    out.reserve(members.size());
    for (int idx : members) {
        if (idx < 0 || idx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)idx];
        if (!c.last_run_ok) continue;
        SweepAxisMember m;
        m.kind      = c.sweep_over_h ? 2 : (c.sweep_over_var ? 1 : 0);
        m.index     = c.sweep_over_var ? c.var_sweep_index : c.param_index;
        m.log_scale = c.log_scale;
        out.push_back(m);
    }
    configure_sweep_x_axis(view, out, params, vars);
}


void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text);


template <class Cfg>
static void apply_snap_x_from_first_member(Plot2DView& view,
                                           const std::vector<int>& members,
                                           const std::vector<Cfg>& cfgs) {
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (members.empty()) return;
    // Первый член С РЕЗУЛЬТАТОМ, а не просто первый: сетку задаёт то, что
    // реально нарисовано, а members[0] может быть ещё не посчитан — тогда
    // снап уходил на поля конфига и дальше на их дефолты.
    const Cfg* pick = nullptr;
    for (int aidx : members) {
        if (aidx < 0 || aidx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)aidx];
        if (!pick) pick = &c;
        if (c.last_run_ok && c.result.n_pts > 1) { pick = &c; break; }
    }
    if (!pick) return;
    apply_snap_x_from_config(view, pick->last_run_ok,
                             pick->result.param_lo, pick->result.param_hi, pick->result.n_pts,
                             pick->param_lo_text, pick->param_hi_text, pick->n_pts_text);
}


void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb);


static constexpr int kNoExponent = -999;


HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx = kNoExponent);


void invalidate_view(HeatmapView& hv);


void invalidate_view(Plot2DView& v);


void invalidate_view(Plot3DView& v);


template <class K, class V>
static void invalidate_views(std::map<K, std::unique_ptr<V>>& m) {
    for (auto& kv : m) if (kv.second) invalidate_view(*kv.second);
}


bool session_reloaded(const AppModel& model, AppModel::AppMode mode, int& seen);


std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series);


void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation);


void colormap_strip_item(int id, float w, float h);


bool colormap_combo(const char* label, int* id, float width);


struct HeatmapToolbarOpts {
    // Куда сохранить выбор colormap'а (per-config + save_session). Пусто —
    // выбор живёт только в самом view (Custom Basins).
    std::function<void(int)> persist_colormap;
    // Доп. элементы сразу после combo Colormap: LS exponent picker,
    // Basins "Renumber (spiral)".
    std::function<void()>    extras;
    // Доп. элементы в самом конце строки (FastSync Line width / Alpha).
    std::function<void()>    extras_tail;
    // Swap axes осмыслен только там, где оси взаимозаменяемы (2D-свип).
    bool show_swap = true;
};


bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o = {});


void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick);


void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen);


std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values);


enum class LibTab : char { Ode = 0, Dae = 1, Map = 2 };


static constexpr int kLibTabCount = 3;


static const char* const kLibTabName[kLibTabCount] = { "ODE", "DAE", "MAP" };


LibTab lib_tab_of(bool is_dae, bool is_map);


struct LibraryListCache {
    std::vector<std::string> names;
    std::vector<int>         custom_counts;   // параллелен names
    std::vector<char>        tab;             // параллелен names: LibTab записи
    std::string              sel_name;      // для какой системы прочитано превью ниже
    bool                     sel_ok = false;  // запись прочиталась
    SystemRecord             sel_rec;
    std::string              sel_desc;        // describe_record(sel_rec), готов для буфера обмена
    double                   filled_at = -1.0;
    bool                     valid     = false;
};


const LibraryListCache& library_list_cached(SystemLibrary& lib);


void draw_library_list(AppModel& model, SystemLibrary& lib, LibTab tab);


void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb);


std::string phase_axis_name(const PhaseAnalysisSession& s, int ax);


void phase_axis_combo(const char* id, PhaseAnalysisSession& s, int& axis);


void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc = nullptr,
                                const GuiCallbacks* cbp = nullptr);


std::function<const char*(int, int)> regime_cell_label(const std::vector<int>& flags,
                                                              int row_stride,
                                                              bool with_oscillation = false);


std::function<bool(int, int)> cancelled_cell_mask(const std::vector<int>& flags, int row_stride);


void draw_adaptive_stats(const std::vector<double>& st);


void draw_regime_summary(const std::vector<int>& flags, RegimeCounts* cache = nullptr, int gen = 0);


using ProjHookFn = std::function<void(int proj_index, const std::string& title)>;


struct PhaseSeriesStyle {
    ImVec4 color      = ImVec4(1, 1, 1, 1);
    bool   as_point   = false;   // только Phase2D; в time domain всегда линия
    float  point_size = 6.0f;
    int    marker     = (int)PointMarker::Circle;
};


using PhaseStyleFn = std::function<bool(int ic_index, PhaseSeriesStyle& out)>;


struct FeatureClusterParams {
    bool   valid         = false;
    double mult_peak     = 1.0;
    double mult_interval = 1.0;
    double eps           = 0.0;
};


void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0,
                                    const PhaseStyleFn& style_fn = {},
                                    const FeatureClusterParams& clust = {});


void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id);


void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_dft1d_controls(AppModel& model, SystemLibrary& lib);


void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_basins_controls(AppModel& model, SystemLibrary& lib);


void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb);


void basins_phase_tick(AppModel& model);


void draw_fastsync_controls(AppModel& model, SystemLibrary& lib);


void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb);


int builtin_scheme_order(const std::string& name);


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/);


void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb);


static const char* const kNetTopologyNames[] = {
    "Custom", "Chain", "Ring", "Star", "Grid", "King graph", "All-to-all", "Small world"
};


double net_value_at(const NetworkResult& r, int p, int node, int var);


int net_degree(const NetworkConfig& c, int node);


void net_mark_custom(NetworkConfig& c);


std::string net_validate_laws(const NetworkSession& s, const NetworkConfig& c);


void draw_network_controls(AppModel& model, SystemLibrary& /*lib*/);
// Neural — большие сети: окно настроек и кривые (means: средние по популяциям, иначе выбранные узлы).
void draw_neural_controls(AppModel& model, SystemLibrary& lib);
enum { kNeuralMeans = 0, kNeuralNodes, kNeuralSync, kNeuralSweep, kNeuralInput, kNeuralLyap, kNeuralRate, kNeuralRaster, kNeuralWeights, kNeuralReadout, kNeuralWindowCount };
void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, int what);
// Карта свипа по двум осям; клик по ячейке выбирает сеть в окнах от времени.
void draw_neural_map_window(AppModel& model, PlotRenderer& renderer, std::map<int, std::unique_ptr<HeatmapView>>& maps);
void draw_network_graph_window(AppModel& model);


void draw_network_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view);


void draw_network_raster_window(AppModel& model, PlotRenderer& renderer, HeatmapView& view);


void draw_parametric_controls(AppModel& model, SystemLibrary& lib);


// Target — то, что принимает parse: сама сессия или её сохраняемое состояние (*State).
template <class Session, class Target>
static bool apply_session_json(AppModel& model,
                               const std::string& json,
                               Session& dst,
                               bool (*parse)(const std::string&, Target&),
                               const char* what)
{
    if (json.empty()) return true;
    Session probe;
    if (!parse(json, probe)) {
        model.session_load_warning =
            std::string("saved session '") + what + "' is corrupt - ignored";
        std::printf("[session] %s: parse failed, keeping defaults from record\n", what);
        std::fflush(stdout);
        return false;
    }
    if (!parse(json, dst)) {                 // не должно случаться, см. выше
        model.session_load_warning =
            std::string("saved session '") + what + "' failed to apply";
        return false;
    }
    model.session_load_warning.clear();
    return true;
}


void order_autosave_seed(AppModel& model);



void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size);

// Индикатор компьюта в верхней строке. Раньше состояние снималось лестницей из двенадцати
// почти одинаковых веток: каждая доставала из своей сессии одну и ту же четвёрку — подпись
// запущенного конфига, время старта, cancel_token, progress_token. Ветки успели разойтись:
// три подставляли запасное имя только при выходе индекса за границы, три — ещё и при пустом label.
enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom, Order, Metrics, Neural };


struct BusyInfo {
    BusyKind    kind = BusyKind::None;
    std::string what;
    std::chrono::steady_clock::time_point start{};
    bool  cancelling = false;   // пользователь нажал Stop, ждём движок
    float progress   = 0.0f;    // 0..1 из session.progress_token
};


template <class Vec>
[[nodiscard]] static std::string running_label(const Vec& items, int idx, const char* fallback) {
    if (idx >= 0 && idx < (int)items.size() && !items[idx].label.empty())
        return items[idx].label;
    return fallback;
}


template <class Session>
[[nodiscard]] static BusyInfo busy_from(const Session& s, BusyKind kind, std::string what) {
    BusyInfo b;
    b.kind  = kind;
    b.what  = std::move(what);
    b.start = s.compute_start_time;
    b.cancelling = s.cancel_token && s.cancel_token->load(std::memory_order_relaxed);
    if (s.progress_token)
        b.progress = s.progress_token->load(std::memory_order_relaxed);
    return b;
}


[[nodiscard]] TrajWork running_traj_work(const AppModel& m);


[[nodiscard]] TrajWork current_tab_traj_work(const AppModel& m);


void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) ;


static auto& basins_heatmaps() { return ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1000>(); }





void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc, int* stopped = nullptr) ;


[[nodiscard]] std::string ic_legend_text(const AnalysisResult& res, size_t k) ;


void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) ;


void draw_level2d_detail(AppModel& model, CustomSession& cs) ;


void draw_level1d_detail(CustomSession& cs) ;


void draw_level3_detail(AppModel& model, CustomSession& cs) ;


void draw_custom_controls(AppModel& model, SystemLibrary& lib) ;


void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) ;


[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) ;


[[nodiscard]] std::string custom_win_suffix(const std::string& sys) ;


void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) ;


struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};


void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) ;


void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) ;


void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) ;


[[nodiscard]] unsigned basins_base_oid(int config_index) ;


ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id,
                              double vmin, double vmax) ;


void draw_basins_phase_controls(AppModel& model, int cfg_idx) ;


void draw_network_topology_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_coupling_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_nodes_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_edges_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) ;


void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label = "points") ;


void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) ;


void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) ;


OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) ;


void order_rehome_config(AppModel& model, int idx) ;


void order_float_build_warning() ;


const char* order_window_kind_name(OrderPlotWindow::Kind k) ;


bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) ;


ImVec4 order_shade(ImVec4 c, float k) ;


int order_marker_for(int within) ;


ImVec4 order_base_color(int idx, int palette) ;


void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) ;


data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) ;


void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) ;


void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) ;


void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) ;


void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) ;


void draw_diagram_controls(AppModel& model, BifurcationAnalysisSession& s, int idx) ;


void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) ;


ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) ;


void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) ;


void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_lle_curve_controls(AppModel& model, LLEAnalysisSession& s, int idx) ;


void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) ;


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


const char* signal_metric_tooltip(int m) ;


void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) ;


void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) ;


void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                              ParametricPlotWindow& win,
                              PlotRenderer& renderer, Plot2DView& view,
                              std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


struct ParamPlotMatchItem { int index; std::string label; };


[[nodiscard]] std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) ;


int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) ;


void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) ;


bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) ;


void draw_generated_code_block(AppModel& model, const GuiCallbacks& cb) ;


void draw_wrapper_coefficients(const std::string& name,
                                      const std::vector<CustomScheme>& custom_schemes,
                                      const std::map<std::string, std::string>& param_values,
                                      const char* id) ;


void begin_coeff_column(const char* id, int rows, int max_rows) ;


std::string wrapper_name_field(const char* id, std::string& text,
                                      const std::string& default_name) ;


bool extr_base_needs_half_s(const std::string& nm) ;


void draw_extrapolation_builder(AppModel& model) ;


void comp_load_coeffs_from_file(AppModel& model, const std::string& path) ;


void draw_composition_builder(AppModel& model, const GuiCallbacks& cb) ;


void draw_wrapper_list(AppModel& model) ;


void draw_builtin_scheme_checkboxes(AppModel& model) ;


void draw_dae_method_checkboxes(AppModel& model) ;


void draw_system_tab(AppModel& model, const GuiCallbacks& cb) ;


void draw_parameters_tab(AppModel& model) ;


[[nodiscard]] std::string trim_copy(const std::string& s) ;


void invalidate_library_cache() ;


void persist_hidden_systems(const AppModel& model) ;


std::string describe_record(const SystemRecord& r) ;


void library_editor_save(AppModel& model, SystemLibrary& lib) ;


std::string dot_latex(const std::string& name, const FieldLabels* labels = nullptr) ;


int input_str_resize(ImGuiInputTextCallbackData* d) ;


int filter_comma_to_dot(ImGuiInputTextCallbackData* data) ;


[[nodiscard]] bool is_numeric_string(const std::string& s) ;


[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) ;


[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) ;


[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) ;


template <class Cfg>
[[nodiscard]] static std::string auto_label_sweep(const Cfg& c,
                                                  const std::vector<std::string>& params,
                                                  const std::vector<std::string>& vars) {
    const std::string x = auto_axis_name(params, vars, c.param_index,
                                         c.sweep_over_var, c.var_sweep_index, c.sweep_over_h);
    if (c.mode_2d) {
        const std::string y = auto_axis_name(params, vars, c.param_index_2,
                                             c.sweep_over_var_2, c.var_sweep_index_2,
                                             c.sweep_over_h_2);
        return x + " x " + y;
    }
    return sweep_range_label(x, c.param_lo_text, c.param_hi_text);
}


[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) ;


[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) ;


[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) ;


int list_digit_step_callback(ImGuiInputTextCallbackData* data) ;


bool InputNumListStr(const char* label, std::string& str, float width = 0.0f) ;


void draw_undo_menu_item(AppModel* m) ;


std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) ;


void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) ;


std::vector<std::pair<std::string, std::string>> user_ctrl_list() ;


const SystemRecord* field_lib_record(const AppModel* m, bool reload) ;


MathExtent field_tex_extent(const std::string& tex, float fs) ;


float field_tex_size() ;


void field_tex_draw(ImDrawList* dl, ImVec2 p, const std::string& tex, float box_h, bool fit = false) ;


struct PgNode {
    std::string seg, path;
    const ParamGroup* g = nullptr;
    std::vector<PgNode> kids;
    int count = 0;   // параметров из names в узле и под ним
};


int pg_count(PgNode& n, const std::set<std::string>& present) ;


struct PgDrop { std::string from, to; bool after = false; };


std::string pg_payload(const ParamGroups* list, const std::string& path) ;


void pg_drag_node(ParamGroups* list, const std::string& path, const std::string& shown, PgDrop& drop) ;


using PgMenu = std::function<void(const std::vector<std::string>&)>;


void pg_collect(const PgNode& n, const std::set<std::string>& present, std::vector<std::string>& out) ;


bool pg_header(const std::string& label, const std::string& id, bool in_table, ParamGroups* list,
               const std::string& seg, PgDrop& drop, const PgMenu& menu,
               const std::function<std::vector<std::string>()>& names_of) ;


void pg_draw(const PgNode& n, const std::set<std::string>& present,
             const std::function<void(const std::string&)>& item, bool in_table, ParamGroups* list, PgDrop& drop,
             const PgMenu& menu) ;


[[nodiscard]] std::string fmt_view_bound(double v, double span) ;


[[nodiscard]] std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) ;


[[nodiscard]] std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) ;


struct ViewRangeValues {
    bool has_x = false; std::string x_lo, x_hi;
    bool has_y = false; std::string y_lo, y_hi;
    const char* source = "view";   // для подписи в стеке отмены
};


[[nodiscard]] ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) ;


[[nodiscard]] ViewRangeValues values_from_target(const ViewRangeTarget& t) ;


void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) ;


void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) ;


[[nodiscard]] bool view_range_bound(const std::vector<ViewRangeTarget>& targets) ;


[[nodiscard]] bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) ;


std::vector<double>& window_node_buf(int window_id) ;


inline LibraryListCache g_library_cache;


inline WsRenameState g_ws_rename;


inline const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";


inline const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D",
    "Metrics 1D", "Metrics 2D"
};
}
