// ДАУ (вкладка DAE, этапы 1-7). Системы заданы здесь же, без library/ (она не в репозитории).
//  CPU (cl.exe-шаг krs_cpu):
//   - разбор и структурный индекс трёх форм маятника;
//   - порядки самосходимостью: Radau IIA (индекс 2: x 5, z 3), Radau IIA + GGL (индекс 3: x 5),
//     SDIRK2 / ROS3P / Rodas4 на индексе 1; связь держится на длинном прогоне (индекс 3);
//   - согласование НУ со скрытыми связями (индекс 3 из сдвинутого старта);
//   - «Solve as ODE»: уровень A = DAE RK4, уровень C держит связь;
//   - отказы: трапеции на индексе 3, CD на индексе 2, CD (index 2) на индексе 1;
//   - разреженная LU (двойной маятник, N = 10): тело её использует, Ньютон сходится;
//  GPU (если есть CUDA):
//   - z0 согласуется в каждой точке свипа (Metrics: свип с z0 = 0 = отдельные согласованные старты);
//   - LLE / LS на ДАУ дают показатели приведённой ОДУ, GPU = CPU, продолжение;
//   - Metrics двойного маятника на Radau + GGL: GPU = CPU; ось по z отклоняется.
//  Этап 9b: адаптивный Rodas4 (индекс 1) — RC против точного решения (конец и плотный выход, CPU и GPU),
//   связь Кеплера против мелкого постоянного шага, отказы (индекс 2, метод без оценки).
//  Этап 9c: BDF 1-5 (свой многошаговый драйвер): ОДУ с точным решением (порядок до 5, точность по допуску),
//   Робертсон (ДАУ индекса 1) против Test Set, маятник индекса 2 при tol 1e-6..1e-10 против Radau, индекс 3 через GGL против Radau IIA + GGL.
//   GPU: BDF 1-5 в модуле NVRTC (kernels/ucuda_bdf.cuh) = CPU на маятниках индекса 2 и 3 и Робертсоне.
//   Свипы на BDF (драйвер UCUDA_BDF_DRIVER): Metrics на ДАУ с z0 по точкам, БД 1D Ван дер Поля, LLE / LS на ДАУ.
#include "test_util.h"
#include "dae_ir.h"
#include "adaptive_settings.h"
#include "krs_cpu.h"
#include "multistep_bdf.h"
#include "nvrtc_engine.h"
#include "parametric_engine.h"
#include <cuda.h>
#include <map>

namespace {

const char* kPend2 = "x' = u\ny' = v\nu' = -lam*x\nv' = -lam*y - g\n0 = x*u + y*v";
const char* kPend3 = "x' = u\ny' = v\nu' = -lam*x\nv' = -lam*y - g\n0 = x^2 + y^2 - L^2";
const char* kDouble =
    "x1' = u1\ny1' = v1\nx2' = u2\ny2' = v2\n"
    "u1' = (-2*l1*x1 + 2*l2*(x2 - x1))/m1\nv1' = (-2*l1*y1 + 2*l2*(y2 - y1))/m1 - g\n"
    "u2' = -2*l2*(x2 - x1)/m2\nv2' = -2*l2*(y2 - y1)/m2 - g\n"
    "0 = x1^2 + y1^2 - L1^2\n0 = (x2 - x1)^2 + (y2 - y1)^2 - L2^2";

System pend(const char* text, const char* index) {
    return parse_dae(text, "x, y, u, v", "lam", "L, g", index).as_codegen_system();
}
const std::vector<double> kPendX0 = { std::sin(1.0), -std::cos(1.0), 0.0, 0.0, 9.81 * std::cos(1.0) };
std::vector<double> tail(std::vector<double> a) { a.insert(a.end(), { 4, 1e-12, 25, 1 }); return a; }
const std::vector<double> kPendA = tail({ 0.5, 1.0, 9.81 });

// Тело шага на CPU (DLL через cl.exe); кэш по тексту тела.
struct Cpu {
    std::map<std::string, std::unique_ptr<KrsCpuStep>> cache;
    KrsCpuStep& get(const std::string& body, int n, int na) {
        auto& p = cache[body];
        if (!p) {
            p.reset(new KrsCpuStep);
            std::vector<KrsCpuDiag> d;
            if (!p->compile(body, n, na, d)) {
                std::string m = "CPU compile of a DAE body failed:";
                for (auto& x : d) m += "\n  line " + std::to_string(x.line) + ": " + x.message;
                throw std::runtime_error(m);
            }
        }
        return *p;
    }
    std::vector<double> run(const std::string& body, std::vector<double> X, const std::vector<double>& a, double h, double T) {
        KrsCpuStep& s = get(body, (int)X.size(), (int)a.size());
        std::vector<numb> x(X.begin(), X.end()), av(a.begin(), a.end());
        s.step_index(0);
        const long n = std::lround(T / h);
        for (long i = 0; i < n; ++i) s.fn()(x.data(), av.data(), (numb)h);
        return std::vector<double>(x.begin(), x.end());
    }
} cpu;

double dist(const std::vector<double>& a, const std::vector<double>& b, int lo, int hi) {
    double d = 0;
    for (int i = lo; i < hi; ++i) d = std::fmax(d, std::fabs(a[(size_t)i] - b[(size_t)i]));
    return d;
}
// Порядок по x и z самосходимостью: X(T) при h, h/2, h/4.
void orders(const System& s, DaeScheme m, const std::vector<double>& X0, const std::vector<double>& a, double T,
            double h, double* px, double* pz) {
    const std::string b = codegen_dae_scheme(s, m);
    const auto y0 = cpu.run(b, X0, a, h, T), y1 = cpu.run(b, X0, a, h / 2, T), y2 = cpu.run(b, X0, a, h / 4, T);
    const int nx = s.dae_nx, n = (int)s.vars.size();
    *px = std::log2(dist(y0, y1, 0, nx) / dist(y1, y2, 0, nx));
    *pz = std::log2(dist(y0, y1, nx, n) / dist(y1, y2, nx, n));
}
void order_at_least(const System& s, DaeScheme m, const std::vector<double>& X0, const std::vector<double>& a,
                    double T, double h, double want_x, double want_z, const char* what) {
    double px, pz;
    orders(s, m, X0, a, T, h, &px, &pz);
    char b[200];
    std::snprintf(b, sizeof b, "%s: order x %.2f (want >= %.1f), z %.2f (want >= %.1f)", what, px, want_x, pz, want_z);
    ut::require(px >= want_x && pz >= want_z, b);
    std::printf("  %s\n", b);
}

void cpu_tests() {
    // 1. индексы
    ut::require(dae_structural_index(parse_dae("x' = -x + z\n0 = z - x^2", "x", "z", "", "").as_codegen_system()) == 1,
                "index 1 not detected");
    ut::require(dae_structural_index(pend(kPend2, "lam=2")) == 2, "index 2 pendulum");
    ut::require(dae_structural_index(pend(kPend3, "u=2, v=2, lam=3")) == 3, "index 3 pendulum");

    // 2. порядки
    const System p2 = pend(kPend2, "lam=2"), p3 = pend(kPend3, "u=2, v=2, lam=3");
    order_at_least(p2, DaeScheme::Radau5, kPendX0, kPendA, 2.0, 0.04, 4.5, 2.7, "Radau IIA, index 2");
    order_at_least(p3, DaeScheme::Radau5GGL, kPendX0, kPendA, 2.0, 0.04, 4.5, 2.7, "Radau IIA + GGL, index 3");
    const System k1 = parse_dae("x' = -y + z*x\ny' = x\n0 = z^3 + z - 0.2*x*y - 0.1", "x, y", "z", "", "").as_codegen_system();
    std::vector<double> k1x = { 1.0, 0.0, 0.0 };
    std::string why;
    ut::require(dae_consistent_z(k1, tail({ 0.5 }).data(), 5, 0.0, k1x.data(), &why), "index-1 z0: " + why);
    order_at_least(k1, DaeScheme::SDIRK2, k1x, tail({ 0.5 }), 1.0, 0.05, 1.8, 0.8, "SDIRK2, index 1");
    order_at_least(k1, DaeScheme::ROS3P, k1x, tail({ 0.5 }), 1.0, 0.05, 2.7, 2.7, "ROS3P, index 1");
    order_at_least(k1, DaeScheme::Rodas4, k1x, tail({ 0.5 }), 1.0, 0.05, 3.7, 3.7, "Rodas4, index 1");

    // 3. связь индекса 3 на длинном прогоне (BDF1, Radau)
    for (DaeScheme m : { DaeScheme::BDF1, DaeScheme::Radau5 }) {
        const auto y = cpu.run(codegen_dae_scheme(p3, m), kPendX0, kPendA, 0.01, 20.0);
        const double g = std::fabs(y[0] * y[0] + y[1] * y[1] - 1.0);
        ut::require(g < 1e-12, ut::num("index 3 constraint after t = 20", g, 0, 1e-12));
    }

    // 4. НУ со скрытыми связями: старт сдвинут на 0.01, lambda = 0
    std::vector<double> X = { kPendX0[0] + 0.01, kPendX0[1] + 0.01, 0.01, 0.01, 0.0 };
    ut::require(dae_consistent_z(p3, kPendA.data(), (int)kPendA.size(), 0.0, X.data(), &why), "hidden constraints: " + why);
    ut::approx(X[0] * X[0] + X[1] * X[1], 1.0, 1e-13, "position on the circle");
    ut::approx(X[0] * X[2] + X[1] * X[3], 0.0, 1e-13, "velocity tangent");
    // lambda из второй производной: (u^2 + v^2) - lam (x^2 + y^2) - g y = 0
    ut::approx(X[4], (X[2] * X[2] + X[3] * X[3] - 9.81 * X[1]), 1e-11, "lambda from the hidden constraint");

    // 5. «Solve as ODE»: уровень A = тот же метод ДАУ (RK4 со связью на стадиях); уровень C держит связь
    {
        const System lin = parse_dae("x' = -x + z\ny' = -2*y + z\n0 = 2*z - x - y", "x, y", "z", "", "").as_codegen_system();
        System ode;
        std::vector<std::string> zex;
        ut::require(dae_reduce_to_ode(lin, ode, &zex, &why), "level A: " + why);
        ode.vars.push_back("z");
        ode.out_exprs = zex;
        const auto a = cpu.run(codegen_scheme(ode, Scheme::RK4), { 1, 2, 1.5 }, tail({ 0.5 }), 0.01, 3.0);
        const auto b = cpu.run(codegen_dae_scheme(lin, DaeScheme::RK4), { 1, 2, 1.5 }, tail({ 0.5 }), 0.01, 3.0);
        ut::require(dist(a, b, 0, 3) < 1e-13, ut::num("level A vs DAE RK4", dist(a, b, 0, 3), 0, 1e-13));

        System s1;
        ut::require(dae_stabilized_index1(p3, 10.0, s1, &why), "level C: " + why);
        ut::require(dae_reduce_to_ode(s1, ode, &zex, &why), "level C -> level A: " + why);
        ode.vars.push_back("lam");
        ode.out_exprs = zex;
        const auto y = cpu.run(codegen_scheme(ode, Scheme::RK4), kPendX0, kPendA, 0.01, 20.0);
        const double g = std::fabs(y[0] * y[0] + y[1] * y[1] - 1.0);
        ut::require(g < 1e-6, ut::num("level C constraint drift after t = 20", g, 0, 1e-6));
    }

    // 6. отказы
    auto throws = [](auto&& f) { try { f(); } catch (const std::exception&) { return true; } return false; };
    ut::require(throws([&] { codegen_dae_scheme(p3, DaeScheme::Trapezoidal); }), "trapezoid accepted index 3");
    ut::require(throws([&] { codegen_dae_scheme(p2, DaeScheme::CD); }), "DAE CD accepted index 2");
    ut::require(throws([&] { codegen_dae_scheme(k1, DaeScheme::CDIndex2); }), "CD (index 2) accepted index 1");
    ut::require(!throws([&] { codegen_dae_scheme(p2, DaeScheme::CDIndex2); }), "CD (index 2) refused index 2");

    // 7. разреженная LU: двойной маятник (N = 10) — тело её использует, связь держится
    const System dp = parse_dae(kDouble, "x1, y1, x2, y2, u1, v1, u2, v2", "l1, l2", "m1, m2, L1, L2, g",
                                "u1=2, v1=2, u2=2, v2=2, l1=3, l2=3").as_codegen_system();
    const std::vector<double> dpx = { 0.9092974268256817, 0.4161468365471424, 1.5077695709296381, 1.2172904520940762,
                                      0, 0, 0, 0, -3.31943268985202, -1.4565381219914928 };
    const std::vector<double> dpa = tail({ 0.5, 1, 1, 1, 1, 9.81 });
    for (DaeScheme m : { DaeScheme::BDF1, DaeScheme::Radau5GGL }) {
        const std::string b = codegen_dae_scheme(dp, m);
        ut::require(b.find("sparse LU:") != std::string::npos, "double pendulum body without the sparse LU");
        const auto y = cpu.run(b, dpx, dpa, 1e-3, 5.0);
        const double g = std::fmax(std::fabs(y[0] * y[0] + y[1] * y[1] - 1),
                                   std::fabs((y[2] - y[0]) * (y[2] - y[0]) + (y[3] - y[1]) * (y[3] - y[1]) - 1));
        ut::require(g < 1e-12, ut::num("double pendulum constraints after t = 5", g, 0, 1e-12));
    }
}

void gpu_tests() {
    ParametricEngine eng;
    // 8. z0 по точкам свипа: x' = -x + z, 0 = z - p (cos x + 1), трапеции, z0 = 0 несогласована
    {
        const System s = parse_dae("x' = -x + z\n0 = z - p*(cos(x) + 1)", "x", "z", "p", "").as_codegen_system();
        const std::string body = codegen_dae_scheme(s, DaeScheme::Trapezoidal);
        for (int cpu_side = 0; cpu_side < 2; ++cpu_side) {
            SignalMetricsRequest q;
            q.krs_body = body; q.amountOfX = 2; q.initial_conditions = { 0.3, 0.0 }; q.base_values = tail({ 0.5, 1.0 });
            q.param_index = 1; q.param_lo = 0.5; q.param_hi = 2.0; q.n_pts = 4; q.use_cpu = cpu_side;
            q.h = 0.05; q.t_max = 2;
            const auto A = eng.run_signal_metrics(q);
            ut::require(A.ok, "metrics sweep: " + A.error);
            for (int j = 0; j < 4; ++j) {
                const double p = 0.5 + 1.5 * j / 3;
                SignalMetricsRequest b = q;
                b.param_lo = b.param_hi = p; b.n_pts = 1; b.initial_conditions = { 0.3, p * (std::cos(0.3) + 1) };
                const auto B = eng.run_signal_metrics(b);
                ut::require(B.ok, "metrics single point: " + B.error);
                for (int m = 0; m < SIGM_COUNT; ++m) {
                    if (A.values[m].size() <= (size_t)j || B.values[m].empty() || std::isnan(B.values[m][0])) continue;
                    ut::approx(A.values[m][(size_t)j], B.values[m][0], 1e-12, cpu_side ? "z0 per point (CPU)" : "z0 per point (GPU)");
                }
            }
        }
        SignalMetricsRequest z;
        z.krs_body = body; z.amountOfX = 2; z.initial_conditions = { 0.3, 0.0 }; z.base_values = tail({ 0.5, 1.0 });
        z.sweep_over_var = true; z.var_sweep_index = 1; z.param_lo = 0; z.param_hi = 1; z.n_pts = 4; z.h = 0.05; z.t_max = 1;
        ut::require(!eng.run_signal_metrics(z).ok, "IC sweep over an algebraic variable accepted");
    }
    // 9. LLE / LS: x1' = -x1 + z, x2' = -2 x2, 0 = z - 0.5 x1  =>  спектр {-0.5, -2}
    {
        const System s = parse_dae("x1' = -x1 + z\nx2' = -2*x2\n0 = z - 0.5*x1", "x1, x2", "z", "p", "").as_codegen_system();
        for (DaeScheme m : { DaeScheme::Trapezoidal, DaeScheme::Radau5 }) {
            const std::string body = codegen_dae_scheme(s, m);
            LLE1DRequest l;
            l.krs_body = body; l.amountOfX = 3; l.initial_conditions = { 1, 1, 0.5 }; l.base_values = tail({ 0.5, 0.0 });
            l.param_index = 1; l.param_lo = 0; l.param_hi = 1; l.n_pts = 2; l.h = 0.01; l.t_max = 40; l.NT = 0.5; l.eps = 1e-6; l.vector_transient = 10;
            const auto g = eng.run_lle_1d(l);
            ut::require(g.ok, "DAE LLE GPU: " + g.error);
            ut::approx(g.lyapunov[0], -0.5, 2e-3, "DAE LLE");
            l.use_cpu = true;
            const auto c = eng.run_lle_1d(l);
            ut::require(c.ok, "DAE LLE CPU: " + c.error);
            ut::approx(c.lyapunov[0], g.lyapunov[0], 1e-9, "DAE LLE CPU vs GPU");

            LS1DRequest q;
            q.krs_body = body; q.amountOfX = 3; q.initial_conditions = { 1, 1, 0.5 }; q.base_values = tail({ 0.5, 0.0 });
            q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2; q.h = 0.01; q.t_max = 40; q.NT = 0.5; q.eps = 1e-6; q.vector_transient = 10;
            for (int cont = 0; cont < 2; ++cont) {
                q.continuation = cont;
                const auto r = eng.run_ls_1d(q);
                ut::require(r.ok && r.spectrum.size() == 2 && r.spectrum[0].size() == 2, "DAE LS: " + r.error);
                ut::approx(r.spectrum[0][0], -0.5, 2e-3, cont ? "DAE LS lambda1 (continuation)" : "DAE LS lambda1");
                ut::approx(r.spectrum[0][1], -2.0, 2e-3, cont ? "DAE LS lambda2 (continuation)" : "DAE LS lambda2");
            }
        }
    }
    // 10. Metrics двойного маятника на Radau + GGL (разреженная LU): GPU = CPU
    {
        const System dp = parse_dae(kDouble, "x1, y1, x2, y2, u1, v1, u2, v2", "l1, l2", "m1, m2, L1, L2, g",
                                    "u1=2, v1=2, u2=2, v2=2, l1=3, l2=3").as_codegen_system();
        SignalMetricsResult R[2];
        for (int c = 0; c < 2; ++c) {
            SignalMetricsRequest q;
            q.krs_body = codegen_dae_scheme(dp, DaeScheme::Radau5GGL); q.amountOfX = 10;
            q.initial_conditions = { 0.9092974268256817, 0.4161468365471424, 1.5077695709296381, 1.2172904520940762,
                                     0, 0, 0, 0, -3.31943268985202, -1.4565381219914928 };
            q.base_values = tail({ 0.5, 1, 1, 1, 1, 9.81 }); q.param_index = 5; q.param_lo = 9; q.param_hi = 10;
            q.n_pts = 3; q.use_cpu = c; q.h = 1e-3; q.t_max = 1;
            R[c] = eng.run_signal_metrics(q);
            ut::require(R[c].ok, std::string(c ? "CPU" : "GPU") + " Radau + GGL metrics: " + R[c].error);
        }
        for (int m = 0; m < SIGM_COUNT; ++m)
            for (size_t i = 0; i < R[0].values[m].size(); ++i) {
                const double a = R[0].values[m][i], b = R[1].values[m][i];
                if (std::isnan(a) && std::isnan(b)) continue;
                ut::require(std::fabs(a - b) <= 1e-8 * (1 + std::fabs(b)), ut::num("Radau + GGL GPU vs CPU", a, b, 1e-8));
            }
    }
}

// --- этап 9b: адаптивный Rodas4 на ДАУ индекса 1 ------------------------------------------------
// RC: u' = i/C, 0 = E - R i - u; u = E (1 - e^(-t/RC)), i = (E - u)/R.
const char* kRc = "u' = i/C\n0 = E - R*i - u";
System rc_sys() { return parse_dae(kRc, "u", "i", "R, C, E").as_codegen_system(); }
const std::vector<double> kRcA = tail({ 0.5, 2.0, 0.5, 3.0 });   // R C E, RC = 1
double rc_u(double t) { return 3.0 * (1 - std::exp(-t)); }

UcudaAdaptParams ad_params(const AdaptiveCode& code, int n, double span, const char* rtol, const char* atol) {
    AdaptiveSettings st;
    st.enabled = true;
    st.rtol = rtol;
    st.atol = atol;
    UcudaAdaptParams P{};
    std::string err;
    const bool ok = adaptive_build_params(st, code, n, span, P, err);
    ut::require(ok, "adaptive_build_params: " + err);
    return P;
}

struct AdCpu {
    AdaptiveCpuModule mod;
    explicit AdCpu(const AdaptiveCode& c, int n) {
        std::vector<KrsCpuDiag> d;
        const bool ok = mod.compile(c.rhs, c.emb, c.dprep, c.deval, "", n, "", d);
        std::string m;
        for (auto& x : d) m += "\n  line " + std::to_string(x.line) + ": " + x.message;
        ut::require(ok && mod.endpoint() && mod.phase(), "adaptive DAE module (CPU) builds:" + m);
    }
};

void adaptive_cpu() {
    const System rc = rc_sys();
    const AdaptiveCode code = codegen_adaptive_dae(rc, DaeScheme::Rodas4);
    AdCpu m(code, 2);
    std::vector<double> x0 = { 0.0, 1.5 };   // согласовано: i = (E - u)/R
    {   // конечная точка: точность по допуску, число шагов растёт при ужесточении
        double y[2], st[8];
        long steps[2];
        const char* tol[2] = { "1e-5", "1e-9" };
        for (int k = 0; k < 2; ++k) {
            const UcudaAdaptParams P = ad_params(code, 2, 5.0, tol[k], tol[k]);
            m.mod.endpoint()(x0.data(), kRcA.data(), &P, 5.0, y, st, nullptr);
            steps[k] = (long)st[0];
            const double eu = std::fabs(y[0] - rc_u(5.0)), ei = std::fabs(y[1] - (3.0 - y[0]) / 2.0);
            std::printf("  Rodas4 adaptive, tol %s: %ld steps, %ld rejected, |u - u*| = %.2e, |g| = %.2e\n",
                        tol[k], steps[k], (long)st[1], eu, ei);
            ut::require(eu < (k ? 1e-7 : 1e-3), ut::num("RC, adaptive Rodas4: u(5)", y[0], rc_u(5.0), k ? 1e-7 : 1e-3));
            ut::require(ei < 1e-12, "RC, adaptive Rodas4: constraint holds at the end");
        }
        ut::require(steps[1] > 2 * steps[0], "tighter tolerance takes more steps");
    }
    {   // плотный выход: равномерная сетка 0.05 при длинных шагах
        const UcudaAdaptParams P = ad_params(code, 2, 5.0, "1e-8", "1e-8");
        const int total = 100;
        std::vector<double> data((size_t)2 * total), times(total), logs(4), stats(9);
        int log_count = 0;
        double fh = 0;
        const int n = m.mod.phase()(x0.data(), kRcA.data(), &P, 0.0, 5.0, 0.05, total, 0, total, 0, data.data(),
                                    times.data(), logs.data(), &log_count, stats.data(), &fh, nullptr);
        ut::require(n == total, "dense output gives every grid point");
        double eu = 0, ei = 0;
        for (int k = 0; k < total; ++k) {   // data[k * dim + c]
            const double t = k * 0.05, u = data[(size_t)2 * k], i = data[(size_t)2 * k + 1];
            eu = std::fmax(eu, std::fabs(u - rc_u(t)));
            ei = std::fmax(ei, std::fabs(i - (3.0 - rc_u(t)) / 2.0));
        }
        std::printf("  Rodas4 dense output on 0.05 grid (%.0f steps): max |u - u*| = %.2e, max |i - i*| = %.2e\n",
                    stats[0], eu, ei);
        ut::require(eu < 1e-6 && ei < 1e-6, ut::num("dense output, x and z", std::fmax(eu, ei), 0, 1e-6));
    }
    {   // нелинейная связь Кеплера: 0 = z + b sin z - x, x' = -x + z (индекс 1) против мелкого шага
        const System k1 = parse_dae("x' = -x + z\n0 = z + b*sin(z) - x", "x", "z", "b").as_codegen_system();
        const AdaptiveCode ck = codegen_adaptive_dae(k1, DaeScheme::Rodas4);
        AdCpu mk(ck, 2);
        std::vector<double> X = { 2.0, 0.0 };
        std::string why;
        const bool zok = dae_consistent_z(k1, tail({ 0.5, 0.5 }).data(), 6, 0.0, X.data(), &why);
        ut::require(zok, "Kepler z0: " + why);
        const auto ref = cpu.run(codegen_dae_scheme(k1, DaeScheme::Rodas4), X, tail({ 0.5, 0.5 }), 1e-3, 3.0);
        const UcudaAdaptParams P = ad_params(ck, 2, 3.0, "1e-9", "1e-9");
        double y[2], st[8];
        const std::vector<double> a = tail({ 0.5, 0.5 });
        mk.mod.endpoint()(X.data(), a.data(), &P, 3.0, y, st, nullptr);
        ut::approx(y[0], ref[0], 1e-7, "Kepler, adaptive Rodas4: x(3)");
        ut::approx(y[1], ref[1], 1e-7, "Kepler, adaptive Rodas4: z(3)");
    }
    // отказы
    bool threw = false;
    try { codegen_adaptive_dae(pend(kPend2, ""), DaeScheme::Rodas4); } catch (const std::exception&) { threw = true; }
    ut::require(threw, "adaptive Rodas4 refuses index 2");
    threw = false;
    try { codegen_adaptive_dae(rc, DaeScheme::BDF1); } catch (const std::exception&) { threw = true; }
    ut::require(threw, "adaptive step refuses a DAE method without an embedded estimate");
    ut::require(adaptive_scheme_name_ok("DAE Rodas4"), "DAE Rodas4 is offered for the adaptive step");
}

// --- этап 9c: многошаговый BDF 1-5 (свой драйвер, CPU) --------------------------------------
void bdf_tests() {
    std::vector<std::vector<double>> tr;
    std::vector<double> tm;
    std::string err;
    {   // ОДУ x' = -2x + cos t, x(0) = 1: x = (2 cos t + sin t)/5 + (3/5) e^(-2t)
        System s;
        s.vars = { "x" };
        s.rhs = { "-2*x + cos(t)" };
        auto ex = [](double t) { return (2 * std::cos(t) + std::sin(t)) / 5 + 0.6 * std::exp(-2 * t); };
        const double x0 = 1.0, a0[1] = { 0.5 };
        double e_prev = 1;
        for (double tol : { 1e-5, 1e-9 }) {
            BdfSettings bs;
            bs.rtol = tol; bs.atol = { tol };
            BdfStats st;
            const bool ok = bdf_integrate(s, a0, 1, &x0, bs, 0.0, 10.0, 0.1, 101, false, 0, tr, tm, st, &err);
            ut::require(ok && tr.size() == 101, "BDF ODE run: " + err);
            double e = 0;
            for (int k = 0; k < 101; ++k) e = std::fmax(e, std::fabs(tr[(size_t)k][0] - ex(k * 0.1)));
            std::printf("  BDF 1-5, ODE, tol %.0e: %lld steps, %lld rejected, max order %d, max error %.2e\n",
                        tol, st.nacc, st.nrej, st.max_q, e);
            ut::require(e < 200 * tol, ut::num("BDF ODE error vs tolerance", e, 0, 200 * tol));
            ut::require(st.max_q == 5, "BDF raises the order to 5 on a smooth problem");
            ut::require(e < e_prev, "tighter tolerance, smaller error");
            e_prev = e;
        }
    }
    {   // Робертсон в форме ДАУ индекса 1, эталон Test Set (Бари) при t = 40
        const System s = parse_dae("y1' = -0.04*y1 + 10000*y2*y3\ny2' = 0.04*y1 - 10000*y2*y3 - 30000000*y2^2\n0 = y1 + y2 + y3 - 1",
                                   "y1, y2", "y3", "").as_codegen_system();
        const double y0[3] = { 1, 0, 0 }, a0[1] = { 0.5 };
        BdfSettings bs;
        bs.rtol = 1e-8; bs.atol = { 1e-12 };
        BdfStats st;
        const bool ok = bdf_integrate(s, a0, 1, y0, bs, 0.0, 40.0, 40.0, 2, false, 0, tr, tm, st, &err);
        ut::require(ok && tr.size() == 2, "BDF Robertson: " + err);
        const auto& y = tr[1];
        std::printf("  DAE BDF 1-5, Robertson to t = 40: %lld steps, %lld rejected, max order %d; y = %.10f %.6e %.10f\n",
                    st.nacc, st.nrej, st.max_q, y[0], y[1], y[2]);
        ut::approx(y[0], 0.7158270687193353, 1e-6, "Robertson y1(40)");
        ut::approx(y[1] / 9.185534764557e-06, 1.0, 1e-4, "Robertson y2(40) relative");
        ut::approx(y[0] + y[1] + y[2], 1.0, 1e-13, "Robertson constraint");
        // жёсткость не душит шаг: 326 шагов (явный метод взял бы ~1e7)
        ut::require(st.nacc < 1000, "stiff problem in a few hundred steps");
    }
    {   // маятник индекса 2 против Radau IIA мелким шагом; допуски 1e-7 / 1e-9 ломали форму Нордсика
        const System s = pend(kPend2, "lam=2");
        const auto ref = cpu.run(codegen_dae_scheme(s, DaeScheme::Radau5), kPendX0, kPendA, 1e-3, 1.0);
        for (double tol : { 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 }) {
            BdfSettings bs;
            bs.rtol = tol; bs.atol = { tol };
            BdfStats st;
            const bool ok = bdf_integrate(s, kPendA.data(), (int)kPendA.size(), kPendX0.data(), bs, 0.0, 1.0, 1.0, 2,
                                          false, 0, tr, tm, st, &err);
            ut::require(ok, "BDF pendulum index 2: " + err);
            const double ex = dist(tr[1], ref, 0, 4), el = std::fabs(tr[1][4] - ref[4]);
            std::printf("  DAE BDF 1-5, pendulum index 2, tol %.0e: %lld steps, %lld rejected, max order %d, "
                        "|x - ref| = %.2e, |lam - ref| = %.2e\n", tol, st.nacc, st.nrej, st.max_q, ex, el);
            ut::require(ex < 100 * tol && el < 1000 * tol, ut::num("BDF index 2 follows the tolerance", ex, 0, 100 * tol));
        }
        // индекс 3 — через GGL (сведение к индексу 2, mu в выход не идут) против Radau IIA + GGL
        const System s3 = pend(kPend3, "");
        const auto ref3 = cpu.run(codegen_dae_scheme(s3, DaeScheme::Radau5GGL), kPendX0, kPendA, 1e-3, 1.0);
        for (double tol : { 1e-7, 1e-9 }) {
            BdfSettings bs;
            bs.rtol = tol; bs.atol = { tol };
            BdfStats st;
            const bool ok = bdf_integrate(s3, kPendA.data(), (int)kPendA.size(), kPendX0.data(), bs, 0.0, 1.0, 1.0, 2,
                                          false, 0, tr, tm, st, &err);
            ut::require(ok && tr[1].size() == 5, "BDF pendulum index 3 (GGL): " + err);
            const double ex = dist(tr[1], ref3, 0, 4), el = std::fabs(tr[1][4] - ref3[4]),
                         g = std::fabs(tr[1][0] * tr[1][0] + tr[1][1] * tr[1][1] - 1);
            std::printf("  DAE BDF 1-5, pendulum index 3 via GGL, tol %.0e: %lld steps, |x - ref| = %.2e, "
                        "|lam - ref| = %.2e, |g| = %.1e\n", tol, st.nacc, ex, el, g);
            ut::require(ex < 100 * tol && el < 1000 * tol && g < 10 * tol, ut::num("BDF index 3 via GGL", ex, 0, 100 * tol));
        }
    }
    // имена в списке схем; постоянный шаг на них отказывает (krs_for_scheme_impl)
    ut::require(scheme_is_multistep("DAE BDF 1-5") && adaptive_scheme_name_ok("BDF 1-5"), "BDF names registered");
}

// BDF 1-5 на GPU (kernels/ucuda_bdf.cuh в модуле NVRTC) против той же формы DASSL на CPU.
void bdf_gpu() {
    struct Case { const char* name; System s; std::vector<double> y0, a; double T, tol; };
    std::vector<Case> cases;
    cases.push_back({ "pendulum index 2", pend(kPend2, "lam=2"), kPendX0, kPendA, 1.0, 1e-8 });
    cases.push_back({ "pendulum index 3 (GGL)", pend(kPend3, ""), kPendX0, kPendA, 1.0, 1e-8 });
    cases.push_back({ "Robertson", parse_dae("y1' = -0.04*y1 + 10000*y2*y3\ny2' = 0.04*y1 - 10000*y2*y3 - 30000000*y2^2\n"
                                             "0 = y1 + y2 + y3 - 1", "y1, y2", "y3", "").as_codegen_system(),
                      { 1, 0, 0 }, { 0.5 }, 40.0, 1e-8 });
    NvrtcEngine eng;
    for (const Case& c : cases) {
        const int dim = (int)c.y0.size(), total = 41;
        const double dt = c.T / (total - 1);
        BdfSettings bs;
        bs.rtol = c.tol; bs.atol = { c.tol * (c.name[0] == 'R' ? 1e-4 : 1.0) };
        std::vector<std::vector<double>> tr;
        std::vector<double> tm;
        BdfStats st;
        std::string err;
        { const bool ok_ = bdf_integrate(c.s, c.a.data(), (int)c.a.size(), c.y0.data(), bs, 0.0, c.T, dt, total, false, 0,
                                         tr, tm, st, &err); ut::require(ok_, std::string("CPU BDF ") + c.name + ": " + err); }
        System sp;
        { const bool ok_ = bdf_prepare_system(c.s, sp, &err); ut::require(ok_, err); }
        BdfGpuRequest rq;
        codegen_bdf_bodies(sp, rq.res, rq.jac);
        const int ng = (int)sp.vars.size();
        rq.amountOfX = ng;
        rq.nx = sp.dae_nx < 0 ? ng : sp.dae_nx;
        rq.kidx.assign((size_t)ng, 1);
        for (int i = 0; i < ng && i < (int)sp.dae_index.size(); ++i) rq.kidx[(size_t)i] = std::max(1, sp.dae_index[(size_t)i]);
        rq.atol.assign((size_t)ng, bs.atol[0]);
        rq.rtol = bs.rtol;
        rq.N = 2;   // две одинаковые траектории — нити не мешают друг другу
        rq.ic_flat.assign((size_t)2 * ng, 0.0);
        for (int k = 0; k < 2; ++k) for (int i = 0; i < dim; ++i) rq.ic_flat[(size_t)k * ng + i] = c.y0[(size_t)i];
        rq.values = c.a;
        rq.t_rec = c.T; rq.dt = dt; rq.total = total;
        PhaseAdaptiveResult out;
        const bool ok = eng.run_phase_portraits_bdf(rq, out);
        ut::require(ok, std::string("GPU BDF ") + c.name + ": " + eng.error());
        ut::require((int)out.traj[0].size() == total && (int)tr.size() == total, "GPU BDF: every grid point");
        double d = 0, d01 = 0;
        for (int k = 0; k < total; ++k)
            for (int i = 0; i < dim; ++i) {
                d = std::fmax(d, std::fabs(out.traj[0][(size_t)k][(size_t)i] - tr[(size_t)k][(size_t)i]));
                d01 = std::fmax(d01, std::fabs(out.traj[0][(size_t)k][(size_t)i] - out.traj[1][(size_t)k][(size_t)i]));
            }
        std::printf("  BDF 1-5 GPU vs CPU, %s: steps %.0f / %lld, max |GPU - CPU| = %.2e\n", c.name, out.stats[0].nacc,
                    st.nacc, d);
        ut::require(d < 100 * c.tol && d01 == 0, ut::num((std::string("BDF GPU = CPU, ") + c.name).c_str(), d, 0, 100 * c.tol));
    }
}

// Свипы Parametric на BDF 1-5: запрос — как fill_adaptive_request (analysis_session.cpp).
AdaptiveRequest bdf_sweep_request(const System& s, double tol, double span) {
    AdaptiveRequest a;
    a.enabled = true;
    AdaptiveCode code;
    codegen_bdf_bodies(s, code.rhs, code.dprep, "D");
    code.p = 5; code.q = 4; code.nlo = 1; code.emb_rhs = 1; code.dense_order = 3;
    AdaptiveSettings st;
    st.enabled = true;
    char tb[32];
    std::snprintf(tb, sizeof(tb), "%g", tol);
    st.rtol = tb;
    st.atol = tb;
    std::string err;
    const int n = (int)s.vars.size();
    { const bool ok_ = adaptive_build_params(st, code, n, span, a.params, err); ut::require(ok_, "BDF params: " + err); }
    a.rhs = code.rhs; a.dprep = code.dprep;
    a.deval = "    (void)D; (void)th; (void)Yo;\n";
    std::string kidx;
    for (int i = 0; i < n; ++i)
        kidx += (i ? "," : "") + std::to_string(i < (int)s.dae_index.size() && s.dae_index[(size_t)i] > 1 ? s.dae_index[(size_t)i] : 1);
    a.prefix = "#define UCUDA_BDF_DRIVER 1\n#define UCUDA_BDF_NX " + std::to_string(s.dae_nx >= 0 ? s.dae_nx : n)
             + "\n#define UCUDA_BDF_KIDX {" + kidx + "}\n";
    return a;
}

void bdf_sweeps() {
    ParametricEngine eng;
    // Metrics: связь зависит от параметра, z0 = 0 несогласован — проекция в каждой точке
    {
        const System s = parse_dae("x' = -x + z\n0 = z - p*(cos(x) + 1)", "x", "z", "p", "").as_codegen_system();
        for (int cpu_side = 0; cpu_side < 2; ++cpu_side) {
            SignalMetricsRequest q;
            q.krs_body = codegen_dae_scheme(s, DaeScheme::BDF1);
            q.amountOfX = 2; q.initial_conditions = { 0.3, 0.0 }; q.base_values = tail({ 0.5, 1.0 });
            q.param_index = 1; q.param_lo = 0.5; q.param_hi = 2.0; q.n_pts = 4; q.use_cpu = cpu_side;
            q.h = 0.05; q.t_max = 2;
            q.adaptive = bdf_sweep_request(s, 1e-9, 2.0);
            const auto A = eng.run_signal_metrics(q);
            ut::require(A.ok, std::string("BDF metrics sweep (") + (cpu_side ? "CPU" : "GPU") + "): " + A.error);
            for (int j = 0; j < 4; ++j) {
                const double p = 0.5 + 1.5 * j / 3;
                SignalMetricsRequest b = q;
                b.param_lo = b.param_hi = p; b.n_pts = 1; b.initial_conditions = { 0.3, p * (std::cos(0.3) + 1) };
                const auto B = eng.run_signal_metrics(b);
                ut::require(B.ok, "BDF metrics single point: " + B.error);
                for (int m = 0; m < SIGM_COUNT; ++m) {
                    if (A.values[m].size() <= (size_t)j || B.values[m].empty() || std::isnan(B.values[m][0])) continue;
                    ut::approx(A.values[m][(size_t)j], B.values[m][0], 1e-8,
                               cpu_side ? "BDF sweep: z0 per point (CPU)" : "BDF sweep: z0 per point (GPU)");
                }
            }
        }
        std::printf("  BDF 1-5 Metrics sweep on a DAE: z0 made consistent per point (GPU and CPU)\n");
    }
    // БД 1D (ОДУ): Ван дер Поль, mu = 1 — пики x предельного цикла 2.00862 (классика, continuation, CPU)
    {
        System s;
        s.vars = { "x", "y" };
        s.params = { "mu" };
        s.rhs = { "y", "mu*(1 - x^2)*y - x" };
        for (int mode = 0; mode < 3; ++mode) {
            Bifurcation1DRequest b;
            b.krs_body = codegen_scheme(s, Scheme::Euler);
            b.amountOfX = 2; b.initial_conditions = { 0.5, 0.0 }; b.base_values = tail({ 0.5, 1.0 });
            b.param_index = 1; b.param_lo = 1.0; b.param_hi = 1.0; b.n_pts = 2; b.writable_var = 0;
            b.h = 0.01; b.transient_time = 60; b.t_max = 30;
            b.continuation = mode >= 1; b.use_cpu = mode == 2;
            b.adaptive = bdf_sweep_request(s, 1e-9, 90.0);
            const auto r = eng.run_bifurcation_1d(b);
            ut::require(r.ok && !r.bifurcation_points.empty(), "BDF BD 1D: " + r.error);
            double pk = 0;
            int cnt = 0;
            for (double v : r.bifurcation_points[0]) if (std::isfinite(v)) { pk = std::fmax(pk, v); ++cnt; }
            const char* name = mode == 0 ? "classic GPU" : mode == 1 ? "continuation GPU" : "continuation CPU";
            std::printf("  BDF 1-5 BD 1D (van der Pol), %s: %d peaks, max %.6f (limit cycle 2.008620)\n", name, cnt, pk);
            ut::require(cnt >= 3, "BDF BD 1D: peaks found");
            ut::approx(pk, 2.00862, 1e-4, "BDF BD 1D: van der Pol peak");
        }
    }
    // LLE / LS: x1' = -x1 + z, x2' = -2 x2, 0 = z - 0.5 x1  =>  спектр {-0.5, -2}
    {
        const System s = parse_dae("x1' = -x1 + z\nx2' = -2*x2\n0 = z - 0.5*x1", "x1, x2", "z", "p", "").as_codegen_system();
        LLE1DRequest l;
        l.krs_body = codegen_dae_scheme(s, DaeScheme::BDF1);
        l.amountOfX = 3; l.initial_conditions = { 1, 1, 0.5 }; l.base_values = tail({ 0.5, 0.0 });
        l.param_index = 1; l.param_lo = 0; l.param_hi = 1; l.n_pts = 2; l.h = 0.01; l.t_max = 40; l.NT = 0.5; l.eps = 1e-6;
        l.vector_transient = 10;
        l.adaptive = bdf_sweep_request(s, 1e-9, 45.0);
        const auto g = eng.run_lle_1d(l);
        ut::require(g.ok, "BDF LLE GPU: " + g.error);
        l.use_cpu = true;
        const auto c = eng.run_lle_1d(l);
        ut::require(c.ok, "BDF LLE CPU: " + c.error);
        std::printf("  BDF 1-5 LLE on a DAE: GPU %.6f, CPU %.6f (exact -0.5)\n", g.lyapunov[0], c.lyapunov[0]);
        ut::approx(g.lyapunov[0], -0.5, 2e-3, "BDF LLE");
        // клоны BDF интегрируются независимо: их точность относительно возмущения ~ tol / eps = 1e-3,
        // и разное округление CPU / GPU сдвигает показатель на ~1e-5 (у адаптивного шага ошибку клонов
        // держит общий регулятор — там 1e-9)
        ut::approx(c.lyapunov[0], g.lyapunov[0], 1e-4, "BDF LLE CPU vs GPU");
        LS1DRequest q;
        q.krs_body = l.krs_body;
        q.amountOfX = 3; q.initial_conditions = { 1, 1, 0.5 }; q.base_values = tail({ 0.5, 0.0 });
        q.param_index = 1; q.param_lo = 0; q.param_hi = 1; q.n_pts = 2; q.h = 0.01; q.t_max = 40; q.NT = 0.5; q.eps = 1e-6;
        q.vector_transient = 10;
        q.adaptive = l.adaptive;
        for (int cont = 0; cont < 2; ++cont) {
            q.continuation = cont;
            const auto r = eng.run_ls_1d(q);
            ut::require(r.ok && r.spectrum.size() == 2 && r.spectrum[0].size() == 2, "BDF LS: " + r.error);
            std::printf("  BDF 1-5 LS on a DAE%s: %.6f %.6f (exact -0.5 -2)\n", cont ? " (continuation)" : "",
                        r.spectrum[0][0], r.spectrum[0][1]);
            ut::approx(r.spectrum[0][0], -0.5, 2e-3, "BDF LS lambda1");
            ut::approx(r.spectrum[0][1], -2.0, 2e-3, "BDF LS lambda2");
        }
    }
}

void adaptive_gpu() {
    const System rc = rc_sys();
    const AdaptiveCode code = codegen_adaptive_dae(rc, DaeScheme::Rodas4);
    NvrtcEngine eng;
    PhaseAdaptiveRequest rq;
    rq.rhs = code.rhs; rq.emb = code.emb; rq.dprep = code.dprep; rq.deval = code.deval;
    rq.amountOfX = 2; rq.ic_flat = { 0.0, 1.5 }; rq.N = 1; rq.values = kRcA;
    rq.params = ad_params(code, 2, 5.0, "1e-8", "1e-8");
    rq.t_rec = 5.0; rq.dt = 0.05; rq.total = 100;
    PhaseAdaptiveResult out;
    const bool ok = eng.run_phase_portraits_adaptive(rq, out);
    ut::require(ok, "GPU adaptive DAE Rodas4: " + eng.error());
    double eu = 0;
    for (int k = 0; k < 100 && k < (int)out.traj[0].size(); ++k)
        eu = std::fmax(eu, std::fabs(out.traj[0][(size_t)k][0] - rc_u(k * 0.05)));
    ut::require(out.traj[0].size() == 100 && eu < 1e-6, ut::num("GPU adaptive Rodas4, dense u", eu, 0, 1e-6));
}

}  // namespace

int main() {
    return ut::run_main("dae", []() -> int {
        set_nvrtc_rdc(true);   // как в приложении с _app_config.json: модули ДАУ без rdc собираются минутами
        cpu_tests();
        adaptive_cpu();
        bdf_tests();
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: GPU part skipped\n"); return ut::kSkip; }
        gpu_tests();
        adaptive_gpu();
        bdf_gpu();
        bdf_sweeps();
        return 0;
    });
}
