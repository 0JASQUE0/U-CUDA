#include "expression_internal.h"
using namespace ucuda::equations_detail;
 // anonymous namespace

// Реализация SystemEvaluator
struct SystemEvaluator::Impl {
    int dim = 0;
    std::vector<std::vector<Instr>> programs; // по одной на уравнение
    // Jacobian, row-major dim*dim, from the same symbolic derivatives the GPU
    // schemes emit. Empty when the system has no derivative (floor/ceil/fmod) --
    // that must not break explicit schemes, which never ask for it.
    std::vector<std::vector<Instr>> jac_programs;
    // Диагональное разложение f_i = coef_i * x_i + rem_i для схем, решающих
    // каждое уравнение относительно своей переменной (CD, SIMP, D). Строится
    // тем же cd_try_extract_linear, что и GPU-ветка, поэтому решение "линейно
    // / нелинейно" у CPU и GPU совпадает по построению. diag_linear[i] == 0 —
    // разложения нет, программы пустые, шаг уходит на итерации.
    std::vector<char> diag_linear;
    // Знак, вынесенный из coef/rem наружу (peel_sign в кодогене): программы
    // считают hs * |rem| и hs * |coef|, а знак попадает в оператор формулы.
    // Побитово это нейтрально — смена знака в IEEE точна.
    std::vector<char> diag_rem_neg, diag_coef_neg;
    // coef не зависит от переменных состояния (и не ноль): "CD fast" умножает
    // на 1/(1 -+ hs*|coef|) вместо деления, как его GPU-предрасчёт.
    std::vector<char> diag_coef_free;
    std::vector<std::vector<Instr>> diag_coef, diag_rem;
    // "CD FP": программы итераций emit_fp_implicit_eq. Линейная f_i: fp_r = rem,
    // fp_g = coef; нелинейная: слагаемые без x_i и с x_i. fp_has_r == 0 —
    // вынесенной части нет. diag_coef_zero — coef == 0 (явная формула).
    std::vector<char> fp_has_r, diag_coef_zero;
    std::vector<std::vector<Instr>> fp_r, fp_g;
    int fp_slot = 1;   // a[fp_slot] -- число итераций CD FP (cd_fp_slot)
    // Первая функция правых частей без комплексного смысла (cd_find_complex_unsafe),
    // пусто — таких нет.
    std::string complex_unsafe;
    int newton_slot = 2;   // a[newton_slot..+2] -- tol, max_iters, full (newton_slot)
    int max_stack = 16;                        // глубина стека (с запасом)
    mutable std::vector<double> stack;
    // Отдельный стек под комплексный проход: eval и eval_complex не зовутся
    // одновременно (обе — из шага одного интегратора), но держать один буфер
    // на два типа нельзя. Аллоцируется по требованию — обычные схемы за него
    // не платят.
    mutable std::vector<ucmplx> stack_c;
};


SystemEvaluator::SystemEvaluator(const System& sys) : impl_(new Impl) {
    if (!sys.out_exprs.empty() || !sys.zb_g.empty())   // z-выходы считает шаг КРС (krs_cpu), не вычислитель
        throw std::runtime_error("SystemEvaluator: z outputs need the compiled step (krs_cpu)");
    if (sys.vars.size() != sys.rhs.size())
        throw std::runtime_error("SystemEvaluator: vars/rhs size mismatch");
    // ДАУ: строки связей 0 = g тут посчитались бы как производные — тихо неверный ответ.
    // Все CPU-пути через вычислитель правых частей для ДАУ закрыты, пока не проведены.
    if (sys.dae_nx >= 0)
        throw std::runtime_error("DAE systems run on the GPU only so far (turn on GPU)");
    impl_->dim = (int)sys.vars.size();
    // индексы имён
    std::map<std::string, int> var_index, param_index;
    for (size_t i = 0; i < sys.vars.size(); ++i) var_index[sys.vars[i]] = (int)i;
    for (size_t j = 0; j < sys.params.size(); ++j) param_index[sys.params[j]] = (int)j;
    // парсим и компилируем каждое уравнение
    impl_->programs.resize(impl_->dim);
    impl_->diag_linear.assign((size_t)impl_->dim, 0);
    impl_->diag_rem_neg.assign((size_t)impl_->dim, 0);
    impl_->diag_coef_neg.assign((size_t)impl_->dim, 0);
    impl_->diag_coef_free.assign((size_t)impl_->dim, 0);
    impl_->diag_coef.resize(impl_->dim);
    impl_->diag_rem.resize(impl_->dim);
    impl_->fp_has_r.assign((size_t)impl_->dim, 0);
    impl_->diag_coef_zero.assign((size_t)impl_->dim, 0);
    impl_->fp_r.resize(impl_->dim);
    impl_->fp_g.resize(impl_->dim);
    int maxdepth = 8;
    for (int i = 0; i < impl_->dim; ++i) {
        Parser p(sys.rhs[i], sys.latex);
        PN ast = p.parse();
        ByteCompiler bc{ impl_->programs[i], var_index, param_index };
        bc.compile(ast);
        // оценка глубины стека: число push не превышает длину программы
        int depth = (int)impl_->programs[i].size() + 4;
        if (depth > maxdepth) maxdepth = depth;
        if (impl_->complex_unsafe.empty())
            if (const std::string* f = cd_find_complex_unsafe(ast)) impl_->complex_unsafe = *f;

        // Разложение по своей переменной — для диагонально-неявных схем.
        PN coef, rem;
        const bool lin = cd_try_extract_linear(ast, sys.vars[i], coef, rem);
        {
            // До peel_sign ниже: он меняет coef/rem на месте.
            PN r, g;
            if (lin) { r = pn_is_zero(rem) ? nullptr : rem; g = coef; }
            else     fp_split_terms(ast, sys.vars[i], r, g);
            impl_->diag_coef_zero[(size_t)i] = lin && pn_is_zero(coef);
            impl_->fp_has_r[(size_t)i] = r ? 1 : 0;
            if (r) { ByteCompiler b{ impl_->fp_r[i], var_index, param_index }; b.compile(r); }
            if (g) { ByteCompiler b{ impl_->fp_g[i], var_index, param_index }; b.compile(g); }
            for (const auto* prog : { &impl_->fp_r[i], &impl_->fp_g[i] }) {
                int d = (int)prog->size() + 4;
                if (d > maxdepth) maxdepth = d;
            }
        }
        if (lin) {
            // Зеркалим emit_diag_implicit_eq вплоть до формы выражения: сперва
            // тот же вынос знака, затем тот же порядок умножений.
            // "CD fast" берёт обратный знаменатель из предрасчёта только там,
            // где coef не зависит от состояния, — то же условие, что в кодогене.
            impl_->diag_coef_free[(size_t)i] = !pn_is_zero(coef) && pn_free_of_vars(coef, sys.vars);
            impl_->diag_rem_neg[(size_t)i]  = (peel_sign(rem,  '+', '-') == '-');
            impl_->diag_coef_neg[(size_t)i] = (peel_sign(coef, '-', '+') == '+');
            ByteCompiler bcc{ impl_->diag_coef[i], var_index, param_index };
            bcc.compile(pn_mul_step(coef));
            ByteCompiler bcr{ impl_->diag_rem[i], var_index, param_index };
            bcr.compile(pn_mul_step(rem));
            impl_->diag_linear[(size_t)i] = 1;
            for (const auto* prog : { &impl_->diag_coef[i], &impl_->diag_rem[i] }) {
                int d = (int)prog->size() + 4;
                if (d > maxdepth) maxdepth = d;
            }
        }
    }
    // Jacobian programs. Built eagerly (N*N tiny programs, negligible at the usual
    // N = 3) but tolerantly: a non-differentiable system must still work with the
    // explicit schemes, so failure leaves jac_programs empty and is reported by
    // has_jacobian() rather than thrown here.
    try {
        std::vector<std::vector<Instr>> jp((size_t)impl_->dim * impl_->dim);
        for (int i = 0; i < impl_->dim; ++i) {
            Parser p(sys.rhs[i], sys.latex);
            PN ast = p.parse();
            jac_check_differentiable(ast);
            for (int j = 0; j < impl_->dim; ++j) {
                ByteCompiler bc{ jp[(size_t)i * impl_->dim + j], var_index, param_index };
                bc.compile(pn_diff(ast, sys.vars[j]));
                int depth = (int)jp[(size_t)i * impl_->dim + j].size() + 4;
                if (depth > maxdepth) maxdepth = depth;
            }
        }
        impl_->jac_programs = std::move(jp);
    }
    catch (const std::exception&) {
        impl_->jac_programs.clear();
    }

    impl_->newton_slot      = newton_slot(sys);
    impl_->fp_slot          = cd_fp_slot(sys);

    impl_->max_stack = maxdepth;
    impl_->stack.resize(maxdepth);
}


SystemEvaluator::~SystemEvaluator() = default;

SystemEvaluator::SystemEvaluator(SystemEvaluator&&) noexcept = default;

SystemEvaluator& SystemEvaluator::operator=(SystemEvaluator&&) noexcept = default;


int SystemEvaluator::dim() const { return impl_->dim; }


void SystemEvaluator::eval(const double* X, const double* a, double* deriv) const {
    double* st = impl_->stack.data();
    for (int i = 0; i < impl_->dim; ++i)
        deriv[i] = run_program(impl_->programs[i], X, a, st);
}


bool SystemEvaluator::has_jacobian() const { return !impl_->jac_programs.empty(); }


const std::string& SystemEvaluator::complex_unsafe_function() const { return impl_->complex_unsafe; }


void SystemEvaluator::eval_jacobian(const double* X, const double* a, double* J) const {
    const int n = impl_->dim;
    if (impl_->jac_programs.empty()) {
        for (int k = 0; k < n * n; ++k) J[k] = 0.0;
        return;
    }
    double* st = impl_->stack.data();
    for (int k = 0; k < n * n; ++k)
        J[k] = run_program(impl_->jac_programs[(size_t)k], X, a, st);
}


void SystemEvaluator::eval_jacobian_complex(const ucmplx* Z, const double* a, ucmplx* J) const {
    const int n = impl_->dim;
    if (impl_->jac_programs.empty()) {
        for (int k = 0; k < n * n; ++k) J[k] = ucmplx(0.0, 0.0);
        return;
    }
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    for (int k = 0; k < n * n; ++k)
        J[k] = run_program(impl_->jac_programs[(size_t)k], Z, a, st);
}


bool   SystemEvaluator::newton_full(const double* a) const      { return (int)a[impl_->newton_slot + 2] != 0; }

double SystemEvaluator::newton_tol(const double* a) const       { return a[impl_->newton_slot]; }

int    SystemEvaluator::newton_max_iters(const double* a) const { return (int)a[impl_->newton_slot + 1]; }


// Сборка результата повторяет emit_diag_implicit_eq оператор в оператор:
//   X[i] = (X[i] <+|-> hs*|rem|) / (1 <-|+> hs*|coef|).
// Случаи coef == 0 и rem == 0 отдельно не разбираются: на GPU они лишь убирают
// из текста деление на 1 и прибавление нуля, а это побитово тождественные
// операции (деление на 1.0 и сложение с 0.0 в IEEE точны).
bool SystemEvaluator::solve_diag_implicit(int i, double* X, const double* a, double hs,
                                          bool recip) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    double* st = impl_->stack.data();
    const double num = run_program(impl_->diag_rem[(size_t)i],  X, a, st, hs);
    const double den = run_program(impl_->diag_coef[(size_t)i], X, a, st, hs);
    const double xn  = impl_->diag_rem_neg[(size_t)i]  ? X[i] - num : X[i] + num;
    const double d   = impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den;
    // "CD fast": X * (1/d), как предрасчитанный cdf_r<i> на GPU.
    X[i] = (recip && impl_->diag_coef_free[(size_t)i]) ? xn * (1.0 / d) : xn / d;
    return true;
}


// Зеркало emit_fp_implicit_eq оператор в оператор:
//   x_fp = X[i] (+ hs * r);  X[i] = x_fp + c * X[i] (c = hs * coef) | x_fp + hs * g.
void SystemEvaluator::solve_diag_fp(int i, double* X, const double* a, double hs,
                                    bool fast) const {
    if (i < 0 || i >= impl_->dim) return;
    const int iters = (int)a[impl_->fp_slot];
    const size_t k = (size_t)i;
    const bool lin = impl_->diag_linear[k] != 0;
    if (lin && (fast || impl_->diag_coef_zero[k])) {
        solve_diag_implicit(i, X, a, hs, fast);
        return;
    }
    double* st = impl_->stack.data();
    const double xs = impl_->fp_has_r[k] ? X[i] + hs * run_program(impl_->fp_r[k], X, a, st)
                                         : X[i];
    if (lin) {
        const double c = hs * run_program(impl_->fp_g[k], X, a, st);
        for (int it = 0; it < iters; ++it) X[i] = xs + c * X[i];
    }
    else {
        for (int it = 0; it < iters; ++it) X[i] = xs + hs * run_program(impl_->fp_g[k], X, a, st);
    }
}


bool SystemEvaluator::solve_diag_implicit_complex(int i, ucmplx* Z, const double* a,
                                                  ucmplx hs) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    const ucmplx num = run_program(impl_->diag_rem[(size_t)i],  Z, a, st, hs);
    const ucmplx den = run_program(impl_->diag_coef[(size_t)i], Z, a, st, hs);
    Z[i] = (impl_->diag_rem_neg[(size_t)i]  ? Z[i] - num : Z[i] + num)
         / (impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den);
    return true;
}


void SystemEvaluator::eval_complex(const ucmplx* X, const double* a, ucmplx* deriv) const {
    if ((int)impl_->stack_c.size() < impl_->max_stack)
        impl_->stack_c.resize(impl_->max_stack);
    ucmplx* st = impl_->stack_c.data();
    for (int i = 0; i < impl_->dim; ++i)
        deriv[i] = run_program(impl_->programs[i], X, a, st);
}


bool scheme_has_complex_core(const std::string& name) {
    return name == "Complex CD" || name == "Complex CD4" || name == "CCD4 (o4s3)"
        || name == "CCD4 (o4s4)" || name == "CCD4 (s-split)_{0-1}"
        || name == "CCD4 (s-split)_{1-0}" || name == "Complex CD_{1-0}"
        || name == "Complex CD4_{1-0}" || name == "CCD4 (o4s3)_{1-0}"
        || name == "CCD4 (o4s4)_{1-0}";
}


std::string codegen_scheme_complex_core(const System& s, Scheme sch) {
    if (!s.out_exprs.empty() || !s.zb_g.empty())
        throw std::runtime_error("z outputs (Solve as ODE): ExtrZ (complex extrapolation core) is not available - pick another scheme");
    if (system_uses_time(s)) throw std::runtime_error(kTimeUnsupportedMsg);
    switch (sch) {
    case Scheme::ComplexCD:    return scheme_cd_common(s, CdKind::Cx,    true);
    case Scheme::ComplexCD4:   return scheme_cd_common(s, CdKind::Cx4,   true);
    case Scheme::ComplexCD4S3: return scheme_cd_common(s, CdKind::Cx4s3, true);
    case Scheme::ComplexCD4S4: return scheme_cd_common(s, CdKind::Cx4s4, true);
    case Scheme::ComplexCD4SS01: return scheme_cd_common(s, CdKind::Cx4ss01, true);
    case Scheme::ComplexCD4SS10: return scheme_cd_common(s, CdKind::Cx4ss10, true);
    case Scheme::ComplexCD10:     return scheme_cd_common(s, CdKind::Cx,    true, true);
    case Scheme::ComplexCD4_10:   return scheme_cd_common(s, CdKind::Cx4,   true, true);
    case Scheme::ComplexCD4S3_10: return scheme_cd_common(s, CdKind::Cx4s3, true, true);
    case Scheme::ComplexCD4S4_10: return scheme_cd_common(s, CdKind::Cx4s4, true, true);
    default: break;
    }
    throw std::runtime_error("scheme has no complex core (ExtrZ needs a built-in complex CD scheme)");
}


int gbs_stage_count(Scheme sch) {
    switch (sch) {
    case Scheme::GBS:         return 1;
    case Scheme::GBS24:       return 2;
    case Scheme::GBS246:      return 3;
    case Scheme::GBS2468:     return 4;
    case Scheme::GBS246810:   return 5;
    case Scheme::GBS24681012: return 6;
    default:                  return 0;
    }
}


std::vector<int> gbs_substeps(int K) {
    std::vector<int> n;
    for (int k = 1; k <= K; ++k) n.push_back(2 * k);
    return n;
}
