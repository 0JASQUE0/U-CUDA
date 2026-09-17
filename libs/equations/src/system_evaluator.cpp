#include "expression_internal.h"
using namespace ucuda::equations_detail;
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
    std::vector<std::vector<Instr>> diag_coef, diag_rem;
    bool   newton_full = false;
    double newton_tol = 1e-10;
    int    newton_max_iters = 8;
    int max_stack = 16;                        // глубина стека (с запасом)
    mutable std::vector<double> stack;
    // Отдельный стек под комплексный проход: eval и eval_complex не зовутся
    // одновременно (обе — из шага одного интегратора), но держать один буфер
    // на два типа нельзя. Аллоцируется по требованию — обычные схемы за него
    // не платят.
    mutable std::vector<ucmplx> stack_c;
};

SystemEvaluator::SystemEvaluator(const System& sys) : impl_(new Impl) {
    if (sys.vars.size() != sys.rhs.size())
        throw std::runtime_error("SystemEvaluator: vars/rhs size mismatch");
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
    impl_->diag_coef.resize(impl_->dim);
    impl_->diag_rem.resize(impl_->dim);
    int maxdepth = 8;
    for (int i = 0; i < impl_->dim; ++i) {
        Parser p(sys.rhs[i], sys.latex);
        PN ast = p.parse();
        ByteCompiler bc{ impl_->programs[i], var_index, param_index };
        bc.compile(ast);
        // оценка глубины стека: число push не превышает длину программы
        int depth = (int)impl_->programs[i].size() + 4;
        if (depth > maxdepth) maxdepth = depth;

        // Разложение по своей переменной — для диагонально-неявных схем.
        PN coef, rem;
        if (cd_try_extract_linear(ast, sys.vars[i], coef, rem)) {
            // Зеркалим emit_diag_implicit_eq вплоть до формы выражения: сперва
            // тот же вынос знака, затем тот же порядок умножений.
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

    impl_->newton_full      = sys.newton_full;
    impl_->newton_tol       = sys.newton_tol;
    impl_->newton_max_iters = sys.newton_max_iters;

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

bool   SystemEvaluator::newton_full() const      { return impl_->newton_full; }
double SystemEvaluator::newton_tol() const       { return impl_->newton_tol; }
int    SystemEvaluator::newton_max_iters() const { return impl_->newton_max_iters; }

// Сборка результата повторяет emit_diag_implicit_eq оператор в оператор:
//   X[i] = (X[i] <+|-> hs*|rem|) / (1 <-|+> hs*|coef|).
// Случаи coef == 0 и rem == 0 отдельно не разбираются: на GPU они лишь убирают
// из текста деление на 1 и прибавление нуля, а это побитово тождественные
// операции (деление на 1.0 и сложение с 0.0 в IEEE точны).
bool SystemEvaluator::solve_diag_implicit(int i, double* X, const double* a, double hs) const {
    if (i < 0 || i >= impl_->dim || !impl_->diag_linear[(size_t)i]) return false;
    double* st = impl_->stack.data();
    const double num = run_program(impl_->diag_rem[(size_t)i],  X, a, st, hs);
    const double den = run_program(impl_->diag_coef[(size_t)i], X, a, st, hs);
    X[i] = (impl_->diag_rem_neg[(size_t)i]  ? X[i] - num : X[i] + num)
         / (impl_->diag_coef_neg[(size_t)i] ? 1 + den    : 1 - den);
    return true;
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

// Публичная функция
