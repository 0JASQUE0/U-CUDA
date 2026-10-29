// Импорт SPICE-netlist'а в ДАУ (этап 8). Всё на CPU (cl.exe-шаг krs_cpu):
//   - RC с источниками SIN / PULSE / PWL против скалярной ОДУ (RK4 мелким шагом);
//   - интегратор на TL072 (таблица замен -> нуллор) внутри .subckt с параметрами и .param;
//   - F / H (токи через источник напряжения), отказы (диод, неизвестная подсхема, I() без источника);
//   - круговой тест: Лоренц -> синтез -> netlist (B и POLY) -> обратно в граф (та же топология)
//     -> импорт в ДАУ -> траектория = CircuitSolver (идеальные ОУ).
#include "test_util.h"
#include "spice_parse.h"
#include "circuit_netlist.h"
#include "circuit_solver.h"
#include "circuit_synth.h"
#include "dae_ir.h"
#include "krs_cpu.h"
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

namespace {

std::vector<double> tail(std::vector<double> a) { a.insert(a.end(), { 4, 1e-12, 25, 1 }); return a; }

struct Imported {
    SpiceImportResult r;
    System sys;
    std::vector<double> a, X;
    int idx(const std::string& v) const {
        for (size_t i = 0; i < sys.vars.size(); ++i) if (sys.vars[i] == v) return (int)i;
        throw std::runtime_error("no variable " + v);
    }
};

Imported import(const std::string& net) {
    Imported m;
    std::string err;
    ut::require(spice_import(net, SpiceImportOptions{}, m.r, &err), "import: " + err);
    const DaeSystem d = parse_dae(m.r.plain_text, m.r.vars_text, m.r.alg_text, m.r.params_text);
    m.sys = d.as_codegen_system();
    std::vector<double> a = { 0.5 };
    for (const auto& p : d.params) a.push_back(std::stod(m.r.param_values.at(p)));
    m.a = tail(a);
    for (const auto& v : m.sys.vars) {
        auto it = m.r.init_conditions.find(v);
        m.X.push_back(it == m.r.init_conditions.end() ? 0.0 : std::stod(it->second));
    }
    ut::require(dae_consistent_z(m.sys, m.a.data(), (int)m.a.size(), 0.0, m.X.data(), &err), "consistent z0: " + err);
    return m;
}

std::map<std::string, std::unique_ptr<KrsCpuStep>> g_cache;
std::vector<double> run(const Imported& m, DaeScheme sch, double h, double T) {
    const std::string body = codegen_dae_scheme(m.sys, sch);
    auto& p = g_cache[body];
    if (!p) {
        p.reset(new KrsCpuStep);
        std::vector<KrsCpuDiag> d;
        if (!p->compile(body, (int)m.X.size(), (int)m.a.size(), d)) {
            std::string e = "CPU compile failed:";
            for (auto& x : d) e += "\n  line " + std::to_string(x.line) + ": " + x.message;
            throw std::runtime_error(e);
        }
    }
    std::vector<numb> x(m.X.begin(), m.X.end()), av(m.a.begin(), m.a.end());
    p->step_index(0);
    const long n = std::lround(T / h);
    for (long i = 0; i < n; ++i) p->fn()(x.data(), av.data(), (numb)h);
    return std::vector<double>(x.begin(), x.end());
}

// y' = f(t, y), RK4 мелким шагом — эталон для RC.
double ode_ref(std::function<double(double, double)> f, double y, double T, int n) {
    const double h = T / n;
    double t = 0;
    for (int i = 0; i < n; ++i) {
        const double k1 = f(t, y), k2 = f(t + h / 2, y + h / 2 * k1), k3 = f(t + h / 2, y + h / 2 * k2),
                     k4 = f(t + h, y + h * k3);
        y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        t += h;
    }
    return y;
}
double clamp01(double u) { return std::fmin(std::fmax(u, 0.0), 1.0); }

void sources() {
    const double kPi = 3.14159265358979323846;
    {   // SIN: VO VA FREQ TD THETA PHASE
        Imported m = import("* rc sin\nV1 in 0 SIN(0.5 1 50 0 0 30)\nR1 in out 1k\nC1 out 0 10u IC=0.2\n.tran 1e-5 0.05\n");
        ut::require(m.r.step_h == "1e-05" && m.r.t_stop == "0.05", ".tran read");
        ut::require(m.r.param_values.count("v1_f") && m.r.param_values.at("v1_f") == "50", "SIN frequency is a parameter");
        const double ref = ode_ref([&](double t, double y) {
            return (0.5 + std::sin(2 * kPi * 50 * t + kPi / 6) - y) / 1e-2; }, 0.2, 0.05, 200000);
        const auto y = run(m, DaeScheme::Radau5, 1e-4, 0.05);
        ut::approx(y[(size_t)m.idx("vc_c1")], ref, 1e-8, "SIN source, RC, Radau IIA");
    }
    {   // PULSE с периодом: раскатан в импульсы до TSTOP
        Imported m = import("* rc pulse\nV1 in 0 PULSE(0 1 0.2m 0.1m 0.1m 0.5m 1.5m)\nR1 in out 1k\nC1 out 0 1u\n"
                            ".tran 1e-5 4m\n");
        auto vs = [](double t) {
            double v = 0;
            for (int k = 0; k < 3; ++k) {
                const double s = 0.2e-3 + k * 1.5e-3;
                v += clamp01((t - s) / 1e-4) - clamp01((t - s - 1e-4 - 0.5e-3) / 1e-4);
            }
            return v;
        };
        const double ref = ode_ref([&](double t, double y) { return (vs(t) - y) / 1e-3; }, 0.0, 4e-3, 400000);
        const auto y = run(m, DaeScheme::Radau5, 1e-5, 4e-3);
        ut::approx(y[(size_t)m.idx("vc_c1")], ref, 1e-8, "periodic PULSE, RC");
        bool note = false;
        for (const auto& s : m.r.notes) note |= s.find("unrolled into 3 pulses") != std::string::npos;
        ut::require(note, "PULSE unrolling is reported");
    }
    {   // PWL тока в R || C; I течёт из 0 через источник в out
        Imported m = import("* rc pwl\nI1 0 out PWL(0 0 1m 2m 3m -1m 5m 0)\nR1 out 0 1k\nC1 out 0 1u\n");
        auto is = [](double t) {
            return 2e-3 * clamp01(t / 1e-3) - 3e-3 * clamp01((t - 1e-3) / 2e-3) + 1e-3 * clamp01((t - 3e-3) / 2e-3);
        };
        const double ref = ode_ref([&](double t, double y) { return (is(t) - y / 1e3) / 1e-6; }, 0.0, 6e-3, 600000);
        const auto y = run(m, DaeScheme::Radau5, 1e-5, 6e-3);
        ut::approx(y[(size_t)m.idx("vc_c1")], ref, 1e-8, "PWL current source, R || C");
    }
}

void subcircuits() {
    // Интегратор: выход = -(1/(R C)) * t * Vin = -500 t при R = 1k, C = 2u.
    Imported m = import("* integrator\n.param rin=1k\n.subckt integ in out params: c=1u\nR1 in s {rin}\nC1 s out {c}\n"
                        "X1 0 s out TL072\n.ends\nV1 a 0 DC 1\nX2 a b integ c=2u\nRl b 0 10k\n.tran 1e-5 1m\n");
    ut::require(m.r.param_values.at("x2_r1") == "1000" && m.r.param_values.at("x2_c1") == "2e-06",
                "subcircuit values: .param and instance override");
    bool note = false;
    for (const auto& s : m.r.notes) note |= s.find("TL072 replaced by an ideal op-amp") != std::string::npos;
    ut::require(note, "TL072 replacement is reported");
    const auto y = run(m, DaeScheme::BDF1, 1e-5, 1e-3);
    ut::approx(y[(size_t)m.idx("v_b")], -0.5, 1e-9, "integrator output at 1 ms");

    // F / H: ток через V1 = -2 мА (V1 отдаёт ток в R1). F1 даёт 3 * i(V1) из 0 в out, H1 — 500 * i(V1).
    Imported f = import("* f h\nV1 a 0 DC 2\nR1 a 0 1k\nF1 0 out V1 3\nR2 out 0 1k\nC2 out 0 1u\n"
                        "H1 c 0 V1 500\nR3 c 0 1k\n");
    const auto z = run(f, DaeScheme::Radau5, 1e-5, 1e-2);
    ut::approx(z[(size_t)f.idx("vc_c2")], -6.0 * (1.0 - std::exp(-10.0)), 1e-8, "CCCS (F)");
    ut::approx(z[(size_t)f.idx("v_c")], -1.0, 1e-10, "CCVS (H)");

    // Отказы.
    SpiceImportResult r;
    std::string err;
    ut::require(!spice_import("* d\nV1 a 0 1\nD1 a b dmod\nC1 b 0 1u\n", {}, r, &err) &&
                err.find("stage 9") != std::string::npos, "diode refused: " + err);
    ut::require(!spice_import("* x\nX1 a b nosuch\nC1 a 0 1u\nR1 b 0 1k\n", {}, r, &err) &&
                err.find("unknown subcircuit") != std::string::npos, "unknown subcircuit refused: " + err);
    ut::require(!spice_import("* i\nB1 a 0 I = I(Vx)\nC1 a 0 1u\n", {}, r, &err) &&
                err.find("I(vx)") != std::string::npos, "I() of a missing source refused: " + err);
}

// Сигнатуры топологии с допуском на числа: эмиттер пишет номиналы с 10 знаками.
bool same_signature(const std::string& a, const std::string& b, std::string* why) {
    std::istringstream ia(a), ib(b);
    std::string la, lb;
    while (true) {
        const bool ga = (bool)std::getline(ia, la), gb = (bool)std::getline(ib, lb);
        if (ga != gb) { *why = "different element count"; return false; }
        if (!ga) return true;
        std::istringstream ta(la), tb(lb);
        std::string x, y;
        while (ta >> x) {
            if (!(tb >> y)) { *why = la + " | " + lb; return false; }
            if (x == y) continue;
            char* e1; char* e2;
            const double u = std::strtod(x.c_str(), &e1), v = std::strtod(y.c_str(), &e2);
            if (*e1 || *e2 || std::fabs(u - v) > 1e-9 * std::fabs(v)) { *why = la + " | " + lb; return false; }
        }
    }
}

void round_trip() {
    System sys;
    sys.vars = { "u", "v", "w" };
    sys.params = { "sigma", "r", "b" };
    sys.rhs = { "sigma*(v - u)", "r*u - v - 20*u*w", "5*u*v - b*w" };
    std::vector<PolyRhs> poly;
    std::string err;
    ut::require(extract_quadratic(sys, { 0.0, 16.0, 45.6, 4.0 }, PolyExtractConfig{}, poly, &err), "extract: " + err);
    CircuitGraph g;
    ut::require(synthesize_circuit(sys, poly, SynthesisConfig{}, g, &err), "synthesize: " + err);
    const std::vector<double> x0 = { 0.1, 0.1, 0.1 };

    for (NetlistDialect dia : { NetlistDialect::Behavioral, NetlistDialect::Poly }) {
        const char* dn = dia == NetlistDialect::Behavioral ? "B sources" : "POLY";
        NetlistOptions o;
        o.var_names = sys.vars;
        o.x0_volts = x0;
        o.dialect = dia;
        const std::string net = emit_spice_netlist(g, o);

        CircuitGraph back;
        ut::require(spice_to_circuit_graph(net, back, &err), std::string("netlist -> graph (") + dn + "): " + err);
        std::string why;
        ut::require(same_signature(circuit_topology_signature(back), circuit_topology_signature(g), &why),
                    std::string("topology signature after the round trip (") + dn + "): " + why);

        Imported m = import(net);
        std::printf("  %s: %s\n", dn, m.r.summary.c_str());
        const double T = 1.0 / g.time_scale;   // одна единица времени ОДУ
        const auto y = run(m, DaeScheme::Radau5, T / 200, T);
        const CircuitRunResult ref = circuit_simulate(g, CircuitSolverConfig{}, x0, 1e-4, 1.0, 1.0);
        ut::require(ref.ok, "circuit_simulate: " + ref.error);
        for (size_t i = 0; i < sys.vars.size(); ++i) {
            std::string var;
            for (const auto& kv : m.r.node_vars) if (kv.first == g.nodes[(size_t)g.var_node[i]].name) var = kv.second;
            ut::require(!var.empty(), "state node of " + sys.vars[i] + " found");
            ut::approx(y[(size_t)m.idx(var)], ref.x[i].back(), 1e-5,
                       (std::string(dn) + ": V(n_" + sys.vars[i] + ") at t = 1 vs CircuitSolver").c_str());
        }
    }

    // Эталон фазы 0, написанный руками: ОУ — E с усилением 1E6 (-> нуллоры), умножители — B.
    std::ifstream f("../../fixtures/lorenz_reference.cir", std::ios::binary);
    ut::require((bool)f, "tests/fixtures/lorenz_reference.cir found");
    std::stringstream ss;
    ss << f.rdbuf();
    Imported m = import(ss.str());
    std::printf("  fixture: %s\n", m.r.summary.c_str());
    ut::require(m.sys.dae_nx == 3 && m.r.summary.find("op-amps as nullors 5") != std::string::npos,
                "fixture: 3 capacitors, 5 op-amps as nullors");
    const auto y = run(m, DaeScheme::Radau5, 1e-5, 1e-3);
    ut::require(std::isfinite(y[(size_t)m.idx("v_n_u")]) && std::fabs(y[(size_t)m.idx("v_n_u")]) < 5.0,
                "fixture runs for 1 ms");
}

}  // namespace

int main() {
    return ut::run_main("spice", []() -> int {
        sources();
        subcircuits();
        round_trip();
        return 0;
    });
}
