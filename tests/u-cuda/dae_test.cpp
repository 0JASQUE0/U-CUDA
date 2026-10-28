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
#include "test_util.h"
#include "dae_ir.h"
#include "krs_cpu.h"
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

}  // namespace

int main() {
    return ut::run_main("dae", []() -> int {
        set_nvrtc_rdc(true);   // как в приложении с _app_config.json: модули ДАУ без rdc собираются минутами
        cpu_tests();
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: GPU part skipped\n"); return ut::kSkip; }
        gpu_tests();
        return 0;
    });
}
