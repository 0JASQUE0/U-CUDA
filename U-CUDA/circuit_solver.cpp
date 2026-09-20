#include "circuit_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
const double kPi = 3.14159265358979323846;
}

bool CircuitSolver::build(const CircuitGraph& g, const CircuitSolverConfig& cfg, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    if (g.nodes.empty())   return fail("circuit solver: empty graph");
    if (cfg.opamp.slew != 0.0)
        return fail("circuit solver: slew rate is not modelled yet (phase 2.5 decides whether it is needed)");

    g_   = g;
    cfg_ = cfg;
    stats_ = CircuitSolverStats();

    n_nodes_ = (int)g_.nodes.size();
    branches_.clear();
    caps_.clear();

    for (const CircuitCapacitor& c : g_.capacitors)
        caps_.push_back(Cap{ c.a, c.b, c.farad, 0.0, 0.0 });

    for (size_t i = 0; i < g_.vsources.size(); ++i)
        branches_.push_back(Branch{ BranchKind::VSource, (int)i, g_.vsources[i].node, -1 });

    for (size_t i = 0; i < g_.opamps.size(); ++i) {
        if (cfg_.opamp.ideal) {
            branches_.push_back(Branch{ BranchKind::OpAmpIdeal, (int)i, g_.opamps[i].out, -1 });
        } else {
            // Однополюсная модель разворачивается в gm-каскад на внутренний узел,
            // R_p и C_p на землю и ограниченный буфер — всё через обычные штампы.
            const int pole = n_nodes_++;
            g_.nodes.push_back(CircuitNode{ "p_" + g_.nodes[(size_t)g_.opamps[i].out].name });
            caps_.push_back(Cap{ pole, 0, cfg_.opamp_cpole, 0.0, 0.0 });
            branches_.push_back(Branch{ BranchKind::OpAmpReal, (int)i, g_.opamps[i].out, pole });
        }
    }

    for (size_t i = 0; i < g_.multipliers.size(); ++i)
        branches_.push_back(Branch{ BranchKind::Multiplier, (int)i, g_.multipliers[i].out, -1 });

    if (!cfg_.opamp.ideal) {
        if (cfg_.opamp.gbw_hz <= 0.0 || cfg_.opamp.a0 <= 0.0 || cfg_.opamp_cpole <= 0.0)
            return fail("circuit solver: op-amp A0, GBW and pole capacitance must be positive");
        gm_ = 2.0 * kPi * cfg_.opamp.gbw_hz * cfg_.opamp_cpole;
        rp_ = cfg_.opamp.a0 / gm_;
    }

    n_v_  = n_nodes_ - 1;
    size_ = n_v_ + (int)branches_.size();
    if (size_ <= 0) return fail("circuit solver: nothing to solve");

    x_.assign((size_t)size_, 0.0);
    jac_.assign((size_t)size_ * size_, 0.0);
    res_.assign((size_t)size_, 0.0);
    delta_.assign((size_t)size_, 0.0);
    pivot_.assign((size_t)size_, 0);
    started_ = false;
    return true;
}

void CircuitSolver::set_initial_variables(const std::vector<double>& x0) {
    // Состояние живёт в заряде конденсатора обратной связи, а не в узле:
    // q = C*(V(s) - V(out)) = C*(0 - x0) при виртуальной земле на суммирующем.
    const int n = (int)std::min(x0.size(), g_.var_node.size());
    for (int i = 0; i < n; ++i) {
        for (size_t c = 0; c < caps_.size(); ++c) {
            if (caps_[c].b != g_.var_node[(size_t)i]) continue;
            caps_[c].q_prev  = -caps_[c].farad * x0[(size_t)i];
            caps_[c].q_prev2 = caps_[c].q_prev;
        }
    }
    started_ = false;
}

void CircuitSolver::assemble(double alpha, bool bdf2, double h) {
    std::fill(jac_.begin(), jac_.end(), 0.0);
    std::fill(res_.begin(), res_.end(), 0.0);

    auto V = [&](int node) { return node > 0 ? x_[(size_t)(node - 1)] : 0.0; };

    for (const CircuitResistor& r : g_.resistors) {
        const double g = r.siemens;
        if (g == 0.0) continue;                 // моном с нулевым коэффициентом = разрыв
        const double i = g * (V(r.a) - V(r.b));
        add_f(idx(r.a), +i);
        add_f(idx(r.b), -i);
        add_j(idx(r.a), idx(r.a), +g); add_j(idx(r.a), idx(r.b), -g);
        add_j(idx(r.b), idx(r.a), -g); add_j(idx(r.b), idx(r.b), +g);
    }

    for (const Cap& c : caps_) {
        const double v    = V(c.a) - V(c.b);
        const double q    = c.farad * v;
        const double hist = bdf2 ? (4.0 * c.q_prev - c.q_prev2) / (2.0 * h)
                                 : c.q_prev / h;
        const double i    = alpha * q - hist;
        const double g    = alpha * c.farad;
        add_f(idx(c.a), +i);
        add_f(idx(c.b), -i);
        add_j(idx(c.a), idx(c.a), +g); add_j(idx(c.a), idx(c.b), -g);
        add_j(idx(c.b), idx(c.a), -g); add_j(idx(c.b), idx(c.b), +g);
    }

    for (size_t k = 0; k < branches_.size(); ++k) {
        const Branch& br = branches_[k];
        const int     row = n_v_ + (int)k;

        // Ток ветви втекает в узел out — вклад в его КЗТ и столбец якобиана.
        add_f(idx(br.out), +x_[(size_t)row]);
        add_j(idx(br.out), row, +1.0);

        switch (br.kind) {
        case BranchKind::VSource: {
            const CircuitVSource& s = g_.vsources[(size_t)br.elem];
            res_[(size_t)row] += V(s.node) - s.volt;
            add_j(row, idx(s.node), 1.0);
            break;
        }
        case BranchKind::OpAmpIdeal: {
            // Нуллор: входы уравнены, выход отдаёт любой ток.
            const CircuitOpAmp& o = g_.opamps[(size_t)br.elem];
            res_[(size_t)row] += V(o.in_plus) - V(o.in_minus);
            add_j(row, idx(o.in_plus),  +1.0);
            add_j(row, idx(o.in_minus), -1.0);
            break;
        }
        case BranchKind::OpAmpReal: {
            const CircuitOpAmp& o = g_.opamps[(size_t)br.elem];
            // gm-каскад: ток gm*(V+ - V-) втекает во внутренний узел полюса.
            const double gin = gm_ * (V(o.in_plus) - V(o.in_minus));
            add_f(idx(br.pole), -gin);
            add_j(idx(br.pole), idx(o.in_plus),  -gm_);
            add_j(idx(br.pole), idx(o.in_minus), +gm_);
            // R_p на землю задаёт усиление на постоянном токе вместе с gm.
            const double gp = 1.0 / rp_;
            add_f(idx(br.pole), +gp * V(br.pole));
            add_j(idx(br.pole), idx(br.pole), +gp);
            // Мягкое ограничение выхода: жёсткий клип обнуляет производную.
            const double s = cfg_.opamp.vsat;
            const double th = std::tanh(V(br.pole) / s);
            res_[(size_t)row] += V(br.out) - s * th;
            add_j(row, idx(br.out),  +1.0);
            add_j(row, idx(br.pole), -(1.0 - th * th));
            break;
        }
        case BranchKind::Multiplier: {
            const CircuitMultiplier& m = g_.multipliers[(size_t)br.elem];
            const double dx = V(m.x1) - V(m.x2);
            const double dy = V(m.y1) - V(m.y2);
            res_[(size_t)row] += V(m.out) - (dx * dy / 10.0 + V(m.z));
            add_j(row, idx(m.out), +1.0);
            add_j(row, idx(m.x1), -dy / 10.0);
            add_j(row, idx(m.x2), +dy / 10.0);
            add_j(row, idx(m.y1), -dx / 10.0);
            add_j(row, idx(m.y2), +dx / 10.0);
            add_j(row, idx(m.z),  -1.0);
            break;
        }
        }
    }
}

bool CircuitSolver::solve_linear(std::string* err) {
    const int n = size_;
    for (int i = 0; i < n; ++i) pivot_[(size_t)i] = i;

    for (int k = 0; k < n; ++k) {
        int    p = k;
        double best = std::fabs(jac_[(size_t)k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(jac_[(size_t)i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (best < cfg_.pivot_min) {
            stats_.pivot_failed = true;
            if (err) {
                char buf[192];
                std::snprintf(buf, sizeof(buf),
                    "circuit solver: matrix is singular at column %d (pivot %.3g) - "
                    "usually a topology error, not a numerical one", k, best);
                *err = buf;
            }
            return false;
        }
        if (p != k) {
            for (int j = 0; j < n; ++j)
                std::swap(jac_[(size_t)k * n + j], jac_[(size_t)p * n + j]);
            std::swap(res_[(size_t)k], res_[(size_t)p]);
        }
        const double d = jac_[(size_t)k * n + k];
        for (int i = k + 1; i < n; ++i) {
            const double f = jac_[(size_t)i * n + k] / d;
            if (f == 0.0) continue;
            jac_[(size_t)i * n + k] = 0.0;
            for (int j = k + 1; j < n; ++j)
                jac_[(size_t)i * n + j] -= f * jac_[(size_t)k * n + j];
            res_[(size_t)i] -= f * res_[(size_t)k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = res_[(size_t)i];
        for (int j = i + 1; j < n; ++j) s -= jac_[(size_t)i * n + j] * delta_[(size_t)j];
        delta_[(size_t)i] = s / jac_[(size_t)i * n + i];
    }
    return true;
}

bool CircuitSolver::step(double h, std::string* err) {
    if (h <= 0.0) { if (err) *err = "circuit solver: step must be positive"; return false; }

    const bool   bdf2  = started_;
    const double alpha = bdf2 ? 1.5 / h : 1.0 / h;

    int    iter = 0;
    double last_norm = 0.0;
    for (; iter < cfg_.newton_max_iters; ++iter) {
        assemble(alpha, bdf2, h);
        // res_ несёт F(x); решаем J*delta = -F.
        for (double& v : res_) v = -v;
        if (!solve_linear(err)) return false;

        double norm = 0.0;
        for (int i = 0; i < size_; ++i) {
            double d = delta_[(size_t)i];
            if (d >  cfg_.newton_max_step) d =  cfg_.newton_max_step;
            if (d < -cfg_.newton_max_step) d = -cfg_.newton_max_step;
            x_[(size_t)i] += d;
            norm = std::max(norm, std::fabs(d));
        }
        last_norm = norm;
        ++stats_.newton_iters;
        if (norm < cfg_.newton_tol) { ++iter; break; }
    }

    stats_.max_iters_in_step = std::max(stats_.max_iters_in_step, iter);
    stats_.max_residual      = std::max(stats_.max_residual, last_norm);
    if (last_norm >= cfg_.newton_tol) stats_.converged_always = false;

    auto V = [&](int node) { return node > 0 ? x_[(size_t)(node - 1)] : 0.0; };
    for (Cap& c : caps_) {
        c.q_prev2 = c.q_prev;
        c.q_prev  = c.farad * (V(c.a) - V(c.b));
    }
    started_ = true;
    ++stats_.steps;
    return true;
}

double CircuitSolver::node_voltage(int node) const {
    return node > 0 && node - 1 < size_ ? x_[(size_t)(node - 1)] : 0.0;
}

double CircuitSolver::variable(int i) const {
    if (i < 0 || i >= (int)g_.var_node.size()) return 0.0;
    return node_voltage(g_.var_node[(size_t)i]);
}

// --- прогон ------------------------------------------------------------------

CircuitRunResult circuit_simulate(const CircuitGraph& g,
                                  const CircuitSolverConfig& cfg,
                                  const std::vector<double>& x0,
                                  double h_ode, double t_end_ode,
                                  double sample_every_ode) {
    CircuitRunResult out;
    if (g.time_scale <= 0.0) { out.error = "circuit solver: graph has no time scale"; return out; }
    if (h_ode <= 0.0 || t_end_ode <= 0.0 || sample_every_ode <= 0.0) {
        out.error = "circuit solver: step, horizon and sampling interval must be positive";
        return out;
    }

    CircuitSolver s;
    if (!s.build(g, cfg, &out.error)) return out;
    s.set_initial_variables(x0);

    const double h_sec   = h_ode / g.time_scale;
    const int    nvars   = (int)g.var_node.size();
    const int    nsteps  = (int)std::llround(t_end_ode / h_ode);
    const int    every   = std::max(1, (int)std::llround(sample_every_ode / h_ode));

    out.x.assign((size_t)nvars, std::vector<double>());
    for (int i = 0; i < nsteps; ++i) {
        if (!s.step(h_sec, &out.error)) { out.stats = s.stats(); return out; }
        if ((i + 1) % every == 0) {
            out.t.push_back((i + 1) * h_ode);
            for (int v = 0; v < nvars; ++v) out.x[(size_t)v].push_back(s.variable(v));
        }
    }
    out.stats = s.stats();
    out.ok    = true;
    return out;
}

// --- самопроверка против эталона фазы 0 --------------------------------------

namespace {

struct RefRow { double t, u, v, w; };

bool load_reference(const char* path, std::vector<RefRow>& rows) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line)) return false;        // заголовок
    while (std::getline(f, line)) {
        for (char& ch : line) if (ch == ',') ch = ' ';
        std::istringstream ls(line);
        RefRow r{};
        double tc = 0.0;
        if (ls >> r.t >> tc >> r.u >> r.v >> r.w) rows.push_back(r);
    }
    return !rows.empty();
}

}  // namespace

std::string circuit_solver_selftest(const char* fixture_csv, bool* ok) {
    std::ostringstream o;
    bool good = true;
    auto check = [&](bool cond, const std::string& what) {
        o << (cond ? "  ok   " : "  FAIL ") << what << "\n";
        if (!cond) good = false;
    };

    System sys;
    sys.vars   = { "u", "v", "w" };
    sys.params = { "sigma", "r", "b" };
    sys.rhs    = { "sigma*(v - u)", "r*u - v - 20*u*w", "5*u*v - b*w" };
    const std::vector<double> vals = { 0.0, 16.0, 45.6, 4.0 };

    PolyExtractConfig pc;
    std::vector<PolyRhs> poly;
    std::string err;
    if (!extract_quadratic(sys, vals, pc, poly, &err)) {
        if (ok) *ok = false;
        return "  FAIL extraction: " + err + "\n";
    }
    SynthesisConfig sc;
    CircuitGraph g;
    if (!synthesize_circuit(sys, poly, sc, g, &err)) {
        if (ok) *ok = false;
        return "  FAIL synthesis: " + err + "\n";
    }

    CircuitSolverConfig cfg;          // идеальные ОУ: сверяемся с оракулом
    const std::vector<double> x0 = { 0.1, 0.1, 0.1 };

    // 1. Короткий горизонт — поточечно против эталона.
    if (fixture_csv) {
        std::vector<RefRow> ref;
        if (!load_reference(fixture_csv, ref)) {
            check(false, std::string("reference CSV not readable: ") + fixture_csv);
        } else {
            CircuitRunResult run = circuit_simulate(g, cfg, x0, 1.0e-4, 2.0, 0.01);
            check(run.ok, std::string("short run completed") + (run.ok ? "" : ": " + run.error));
            if (run.ok) {
                double worst = 0.0;
                size_t n = std::min(run.t.size(), (size_t)200);
                for (size_t i = 0; i < n; ++i) {
                    const RefRow& r = ref[i + 1];   // строка 0 эталона — t = 0
                    worst = std::max(worst, std::fabs(run.x[0][i] - r.u));
                    worst = std::max(worst, std::fabs(run.x[1][i] - r.v));
                    worst = std::max(worst, std::fabs(run.x[2][i] - r.w));
                }
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                    "pointwise vs reference over t <= 2: max |dV| = %.3e", worst);
                check(worst < 5.0e-3, buf);
                std::snprintf(buf, sizeof(buf), "Newton needed at most %d iterations per step",
                              run.stats.max_iters_in_step);
                check(run.stats.max_iters_in_step <= 6, buf);
                check(run.stats.converged_always, "Newton converged on every step");
            }
        }
    }

    // 2. Длинный горизонт — по инвариантам: поточечно на хаосе нечего сравнивать.
    CircuitRunResult run = circuit_simulate(g, cfg, x0, 1.0e-4, 50.0, 0.01);
    check(run.ok, std::string("long run completed") + (run.ok ? "" : ": " + run.error));
    if (run.ok) {
        // Сравниваем по std, а не по min/max: возмущение НУ на 1e-9 двигает границы
        // аттрактора на 0.19 В, а std — на 0.004 В (замер в tests/fixtures/README.md).
        const char*  names[3]   = { "u", "v", "w" };
        const double ref_std[3] = { 1.26194907, 1.43477968, 0.66448935 };
        const double ref_lo[3]  = { -2.78001954, -3.69699901, 0.51844278 };
        const double ref_hi[3]  = {  2.90378617,  3.92441228, 3.70392581 };
        const size_t skip       = run.t.size() / 5;   // тот же отброс переходного, что в эталоне
        for (int v = 0; v < 3; ++v) {
            double mean = 0.0, mn = 1e300, mx = -1e300;
            size_t cnt = 0;
            for (size_t i = skip; i < run.t.size(); ++i) {
                const double s = run.x[(size_t)v][i];
                mean += s; ++cnt;
                mn = std::min(mn, s);
                mx = std::max(mx, s);
            }
            mean /= (double)cnt;
            double var = 0.0;
            for (size_t i = skip; i < run.t.size(); ++i) {
                const double d = run.x[(size_t)v][i] - mean;
                var += d * d;
            }
            const double sd = std::sqrt(var / (double)cnt);
            char buf[192];
            std::snprintf(buf, sizeof(buf), "std of %s: %.4f vs reference %.4f",
                          names[v], sd, ref_std[v]);
            check(std::fabs(sd - ref_std[v]) < 0.02, buf);
            // Границы — только как грубая проверка, что аттрактор тот же по масштабу.
            std::snprintf(buf, sizeof(buf), "%s stays on the same attractor: [%.3f, %.3f]",
                          names[v], mn, mx);
            check(mn > ref_lo[v] - 0.5 && mx < ref_hi[v] + 0.5
                  && mn < ref_lo[v] + 0.5 && mx > ref_hi[v] - 0.5, buf);
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%lld steps, %lld Newton iterations (%.2f per step)",
                      run.stats.steps, run.stats.newton_iters,
                      (double)run.stats.newton_iters / (double)std::max(1LL, run.stats.steps));
        o << "  info " << buf << "\n";
        check(!run.stats.pivot_failed, "no singular matrix");
    }

    // 3. Однополюсный ОУ должен собираться и считать — численную достоверность
    //    этой модели доказывает фаза 2.5, здесь только работоспособность.
    CircuitSolverConfig real = cfg;
    real.opamp.ideal = false;
    CircuitRunResult rr = circuit_simulate(g, real, x0, 1.0e-5, 5.0, 0.01);
    check(rr.ok, std::string("one-pole op-amp model runs") + (rr.ok ? "" : ": " + rr.error));
    if (rr.ok) {
        double mx = 0.0;
        for (size_t i = 0; i < rr.t.size(); ++i) mx = std::max(mx, std::fabs(rr.x[0][i]));
        char buf[160];
        std::snprintf(buf, sizeof(buf), "one-pole run stays bounded (max |u| = %.3f V)", mx);
        check(mx < 20.0, buf);
    }

    if (ok) *ok = good;
    o << (good ? "SOLVER SELFTEST PASSED\n" : "SOLVER SELFTEST FAILED\n");
    return o.str();
}
