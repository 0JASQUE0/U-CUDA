// Импорт SPICE-netlist'а в ДАУ (этапы 8, 9a и 9d). Всё на CPU (cl.exe-шаг krs_cpu):
//   - этап 9d: JFET (насыщение, линейная область, P = -N), F POLY, ОУ с JFET-входом "как есть" против идеального;
//   - RC с источниками SIN / PULSE / PWL против скалярной ОДУ (RK4 мелким шагом);
//   - интегратор на TL072 (таблица замен -> нуллор) внутри .subckt с параметрами и .param;
//   - F / H (токи через источник напряжения), отказы (MOSFET, диод без .model, неизвестная подсхема, I() без источника);
//   - диоды и BJT по .model (выпрямитель, каскад ОЭ NPN/PNP в рабочей точке, RS), топология,
//     цепь Чуа на встречных диодах = та же система, выписанная как ОДУ;
//   - круговой тест: Лоренц -> синтез -> netlist (B и POLY) -> обратно в граф (та же топология)
//     -> импорт в ДАУ -> траектория = CircuitSolver (идеальные ОУ).
#include "num_parse.h"
#include "test_util.h"
#include "spice_parse.h"
#include "circuit_netlist.h"
#include "circuit_solver.h"
#include "circuit_synth.h"
#include "dae_devices.h"
#include "dae_ir.h"
#include "krs_cpu.h"
#include "system_library.h"
#include <algorithm>
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
    { const bool ok_ = spice_import(net, SpiceImportOptions{}, m.r, &err); ut::require(ok_, "import: " + err); }
    const DaeSystem d = parse_dae(m.r.plain_text, m.r.vars_text, m.r.alg_text, m.r.params_text);
    m.sys = d.as_codegen_system();
    std::vector<double> a = { 0.5 };
    for (const auto& p : d.params) a.push_back(parse_num(m.r.param_values.at(p), 0.0));
    m.a = tail(a);
    for (const auto& v : m.sys.vars) {
        auto it = m.r.init_conditions.find(v);
        m.X.push_back(it == m.r.init_conditions.end() ? 0.0 : parse_num(it->second, 0.0));
    }
    const bool zok = dae_consistent_z(m.sys, m.a.data(), (int)m.a.size(), 0.0, m.X.data(), &err);
    ut::require(zok, "consistent z0: " + err + "\n" + m.r.plain_text);
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
    ut::require(m.r.param_values.at("x2_r1") == "1k" && m.r.param_values.at("x2_c1") == "2u",
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
    ut::require(!spice_import("* z\nV1 a 0 1\nZ1 a a 0 zmod\nC1 a 0 1u\n", {}, r, &err) &&
                err.find("MESFET") != std::string::npos, "MESFET refused: " + err);
    ut::require(!spice_import("* d\nV1 a 0 1\nD1 a b nomodel\nC1 b 0 1u\n", {}, r, &err) &&
                err.find("unknown model") != std::string::npos, "diode without .model refused: " + err);
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
    { const bool ok_ = extract_quadratic(sys, { 0.0, 16.0, 45.6, 4.0 }, PolyExtractConfig{}, poly, &err); ut::require(ok_, "extract: " + err); }
    CircuitGraph g;
    { const bool ok_ = synthesize_circuit(sys, poly, SynthesisConfig{}, g, &err); ut::require(ok_, "synthesize: " + err); }
    const std::vector<double> x0 = { 0.1, 0.1, 0.1 };

    for (NetlistDialect dia : { NetlistDialect::Behavioral, NetlistDialect::Poly }) {
        const char* dn = dia == NetlistDialect::Behavioral ? "B sources" : "POLY";
        NetlistOptions o;
        o.var_names = sys.vars;
        o.x0_volts = x0;
        o.dialect = dia;
        const std::string net = emit_spice_netlist(g, o);

        CircuitGraph back;
        { const bool ok_ = spice_to_circuit_graph(net, back, &err); ut::require(ok_, std::string("netlist -> graph (") + dn + "): " + err); }
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

// --- этап 9a: диоды, BJT, рабочая точка, топология ----------------------------------------

const double kVt = 0.025864186384551;
double id_shockley(double v, double is, double n) { return is * (std::exp(v / (n * kVt)) - 1) + 1e-12 * v; }

void devices() {
    {   // Однополупериодный выпрямитель на RC: диод против той же формулы в скалярной ОДУ.
        Imported m = import("* rectifier\nV1 in 0 SIN(0 5 50)\nD1 in out dmod\nR1 out 0 1k\nC1 out 0 10u IC=0\n"
                            ".model dmod D(IS=1e-14 N=1.5)\n.tran 1e-5 0.04 uic\n");
        ut::require(m.r.uic && m.r.param_values.at("dmod_is") == "1e-14", "UIC and the model parameters read");
        const double kPi = 3.14159265358979323846;
        const double ref = ode_ref([&](double t, double y) {
            return (id_shockley(5 * std::sin(2 * kPi * 50 * t) - y, 1e-14, 1.5) - y / 1e3) / 1e-5; }, 0.0, 0.04, 2000000);
        const auto y = run(m, DaeScheme::Radau5, 1e-5, 0.04);
        ut::approx(y[(size_t)m.idx("vc_c1")], ref, 1e-6, "half-wave rectifier, diode");
    }
    // Каскад с общим эмиттером: рабочая точка против независимого 2D-Ньютона по тем же формулам.
    auto ce = [&](bool pnp) {
        const char* s = pnp ? "-" : "";
        std::string net = std::string("* ce\nVcc vcc 0 DC ") + s + "10\nRc vcc c 1k\nRb vcc b 470k\nQ1 c b 0 qm\n"
                        "C1 c 0 1n\n.model qm " + (pnp ? "PNP" : "NPN") + "(IS=1e-15 BF=100 BR=2 VAF=50)\n";
        SpiceImportResult r;
        std::string err;
        { const bool ok_ = spice_import(net, {}, r, &err); ut::require(ok_, "CE import: " + err); }
        ut::require(!r.uic && !r.has_ic, "no UIC, no IC: the DC operating point applies");
        Imported m = import(net);
        std::vector<double> X(m.X.size(), 0.0);
        int it = 0;
        { const bool ok_ = dae_dc_operating_point(m.sys, m.a.data(), (int)m.a.size(), 0.0, X.data(), &it, &err); ut::require(ok_, "CE OP: " + err); }
        return std::make_pair(X[(size_t)m.idx("v_c")], X[(size_t)m.idx("v_b")]);
    };
    {
        // Опорное решение: неизвестные vb, vc; KCL в b и c.
        const double IS = 1e-15, BF = 100, BR = 2, VAF = 50;
        double vb = 0.6, vc = 5;
        for (int k = 0; k < 100; ++k) {
            auto F = [&](double b, double c, double* f) {
                const double ebe = std::exp(b / kVt), ebc = std::exp((b - c) / kVt);
                const double ic = IS * (ebe - ebc) * (1 - (b - c) / VAF) - IS / BR * (ebc - 1) - 1e-12 * (b - c);
                const double ib = IS / BF * (ebe - 1) + IS / BR * (ebc - 1) + 1e-12 * (b + (b - c));
                f[0] = (10 - b) / 470e3 - ib;
                f[1] = (10 - c) / 1e3 - ic;
            };
            double f[2], fb[2], fc[2];
            F(vb, vc, f);
            F(vb + 1e-7, vc, fb);
            F(vb, vc + 1e-7, fc);
            const double j00 = (fb[0] - f[0]) / 1e-7, j10 = (fb[1] - f[1]) / 1e-7,
                         j01 = (fc[0] - f[0]) / 1e-7, j11 = (fc[1] - f[1]) / 1e-7, det = j00 * j11 - j01 * j10;
            double db = -(f[0] * j11 - j01 * f[1]) / det, dc = -(j00 * f[1] - f[0] * j10) / det;
            db = std::fmax(-0.1, std::fmin(0.1, db));
            vb += db;
            vc += dc;
        }
        const auto npn = ce(false), pnp = ce(true);
        ut::approx(npn.first, vc, 1e-6, "NPN common emitter: V(c) at the operating point");
        ut::approx(npn.second, vb, 1e-6, "NPN common emitter: V(b)");
        ut::approx(pnp.first, -npn.first, 1e-9, "PNP mirror: V(c) = -V(c) of NPN");
        ut::approx(pnp.second, -npn.second, 1e-9, "PNP mirror: V(b)");
        std::printf("  CE stage: Vb = %.6f V, Vc = %.6f V\n", npn.second, npn.first);
    }
    {   // Диод с RS: внутренний узел; ток через R1 = ток диода по напряжению на переходе.
        Imported m = import("* rs\nV1 in 0 DC 5\nR1 in a 1k\nD1 a 0 dr\nC1 a 0 1n\n.model dr D(IS=1e-14 RS=10)\n");
        std::vector<double> X(m.X.size(), 0.0);
        int it = 0;
        std::string err;
        { const bool ok_ = dae_dc_operating_point(m.sys, m.a.data(), (int)m.a.size(), 0.0, X.data(), &it, &err); ut::require(ok_, "RS OP: " + err); }
        const double va = X[(size_t)m.idx("v_d1_a")], vin = X[(size_t)m.idx("v_a")];
        ut::approx((5 - vin) / 1e3, id_shockley(va, 1e-14, 1.0), 1e-12, "diode RS: KCL through the internal node");
        ut::approx((vin - va) / 10, id_shockley(va, 1e-14, 1.0), 1e-12, "diode RS: drop on RS");
    }
    // Топология.
    SpiceImportResult r;
    std::string err;
    ut::require(!spice_import("* vloop\nV1 a 0 1\nV2 a 0 2\nC1 a 0 1u\n", {}, r, &err) &&
                err.find("loop of voltage sources") != std::string::npos, "V loop refused: " + err);
    ut::require(!spice_import("* icut\nI1 0 a 1m\nI2 a b 1m\nR1 b 0 1k\nC1 b 0 1u\n", {}, r, &err) &&
                err.find("only to current sources") != std::string::npos, "current-source cutset refused: " + err);
    { const bool ok_ = spice_import("* float\nV1 a 0 1\nR1 a b 1k\nC1 b c 1u\nC2 c 0 1u\n", {}, r, &err); ut::require(ok_, "floating: " + err); }
    bool fl = false;
    for (const auto& s : r.notes) fl |= s.find("no DC path to ground: c") != std::string::npos;
    ut::require(fl, "node without a DC path is reported");
}

// --- этап 9d: JFET, F/H POLY, ОУ "как есть" и копия на идеальных -------------------------------
std::vector<double> op_point(const std::string& net, Imported& m, const SpiceImportOptions& o = {}) {
    std::string err;
    { const bool ok_ = spice_import(net, o, m.r, &err); ut::require(ok_, "import: " + err); }
    const DaeSystem d = parse_dae(m.r.plain_text, m.r.vars_text, m.r.alg_text, m.r.params_text);
    m.sys = d.as_codegen_system();
    std::vector<double> a = { 0.5 };
    for (const auto& p : d.params) a.push_back(parse_num(m.r.param_values.at(p), 0.0));
    m.a = tail(a);
    std::vector<double> X(m.sys.vars.size(), 0.0);
    int it = 0;
    { const bool ok_ = dae_dc_operating_point(m.sys, m.a.data(), (int)m.a.size(), 0.0, X.data(), &it, &err);
      ut::require(ok_, "DC OP: " + err); }
    return X;
}

void devices_9d() {
    // JFET: ток стока по Шихману–Ходжесу (насыщение и линейная область), N и P зеркально.
    auto jfet = [&](double vds, double vgs, bool p) {
        const char* s = p ? "-" : "";
        Imported m;
        const auto X = op_point(std::string("* jfet\nVd d 0 DC ") + s + std::to_string(vds) + "\nVg g 0 DC " + s +
                                std::to_string(vgs) + "\nJ1 d g 0 jm\nC1 d 0 1n\n.model jm " + (p ? "PJF" : "NJF") +
                                "(VTO=-2 BETA=1e-3 LAMBDA=1e-3 IS=1e-14)\n", m);   // VTO < 0 у обоих типов, знак — у типа
        return -X[(size_t)m.idx("i_vd")];   // ток, который Vd отдаёт в сток
    };
    ut::approx(jfet(5, -1, false), 1e-3 * 1 * (1 + 5e-3), 1e-9, "N-JFET saturation: BETA (vgs - VTO)^2 (1 + LAMBDA vds)");
    ut::approx(jfet(0.5, 0, false), 1e-3 * 0.5 * (4 - 0.5) * (1 + 5e-4), 1e-9, "N-JFET linear region");
    ut::approx(jfet(5, -1, true), -1e-3 * 1 * (1 + 5e-3), 1e-9, "P-JFET = -N-JFET");
    // F с POLY(2): ток = 1*i(V1) + 2*i(V2); i(V1) = -1 мА, i(V2) = -2 мА -> -5 мА из 0 в out через 1k.
    {
        Imported m;
        const auto X = op_point("* fpoly\nV1 a 0 1\nR1 a 0 1k\nV2 b 0 2\nR2 b 0 1k\nF1 0 out POLY(2) V1 V2 0 1 2\n"
                                "R3 out 0 1k\nC1 out 0 1n\n", m);
        ut::approx(X[(size_t)m.idx("v_out")], -5.0, 1e-9, "F POLY(2) of two source currents");
    }
    // ОУ с JFET-входом (своя макромодель под именем TL072): инвертирующий усилитель x(-10).
    const std::string amp =
        "* inverting amplifier\n"
        ".subckt TL072 inp inn vcc vee out\n"
        "J1 d1 inp s jn\nJ2 d2 inn s jn\nIss s vee 400u\nR1 vcc d1 10k\nR2 vcc d2 10k\n"
        "G1 0 o2 d2 d1 1m\nR3 o2 0 100meg\nC2 o2 0 10p\nE1 out 0 o2 0 1\n"
        ".model jn NJF(VTO=-2 BETA=1e-3 LAMBDA=1e-3 IS=1e-14)\n.ends\n"
        "Vin in 0 DC 0.1\nRin in m 1k\nRf m out 10k\nX1 0 m vcc vee out TL072\nVcc vcc 0 15\nVee vee 0 -15\n"
        "Cl out 0 1n\n";
    Imported real, ideal;
    SpiceImportOptions io;
    io.ideal_opamps = true;
    const auto xr = op_point(amp, real), xi = op_point(amp, ideal, io);
    bool as_is = false, repl = false;
    for (const auto& s : real.r.notes) as_is |= s.find("TL072 imported as is") != std::string::npos;
    for (const auto& s : ideal.r.notes) repl |= s.find("TL072 replaced by an ideal op-amp") != std::string::npos;
    ut::require(as_is && repl, "op-amp mode is reported (as is / ideal)");
    ut::require(real.r.summary.find("semiconductors 2") != std::string::npos, "as is: the JFET pair is in the circuit");
    const double vr = xr[(size_t)real.idx("v_out")], vi = xi[(size_t)ideal.idx("v_out")];
    std::printf("  inverting amp x(-10): real model V(out) = %.6f, ideal = %.6f\n", vr, vi);
    ut::approx(vi, -1.0, 1e-12, "ideal op-amp: exactly -1 V");
    ut::approx(vr, -1.0, 2e-3, "JFET op-amp model as is: -1 V within the finite open-loop gain");
    ut::require(std::fabs(vr - vi) > 1e-9, "the real model is not silently the ideal one");
}

// Цепь Чуа с гладкой нелинейностью: отрицательная проводимость (G) и встречные диоды. Импорт против
// той же системы, выписанной руками как ОДУ.
void chua() {
    const double C1 = 10e-9, C2 = 100e-9, L = 18e-3, R = 1.8e3, gm = -1.0 / 1.2e3, IS = 1e-14;
    Imported m = import("* chua\nR1 n1 n2 1.8k\nC1 n1 0 10n IC=0.1\nC2 n2 0 100n IC=0\nL1 n2 0 18m IC=0\n"
                        "G1 n1 0 n1 0 {-1/1.2k}\nD1 n1 0 dm\nD2 0 n1 dm\n.model dm D(IS=1e-14)\n.tran 1e-7 1m uic\n");
    auto f = [&](const double* y, double* d) {
        const double v1 = y[0], v2 = y[1], il = y[2];
        d[0] = ((v2 - v1) / R - gm * v1 - id_shockley(v1, IS, 1) + id_shockley(-v1, IS, 1)) / C1;
        d[1] = ((v1 - v2) / R - il) / C2;
        d[2] = v2 / L;
    };
    double y[3] = { 0.1, 0, 0 };
    const double T = 1e-3;
    const int n = 2000000;
    const double h = T / n;
    for (int i = 0; i < n; ++i) {
        double k1[3], k2[3], k3[3], k4[3], t[3];
        f(y, k1);
        for (int j = 0; j < 3; ++j) t[j] = y[j] + h / 2 * k1[j];
        f(t, k2);
        for (int j = 0; j < 3; ++j) t[j] = y[j] + h / 2 * k2[j];
        f(t, k3);
        for (int j = 0; j < 3; ++j) t[j] = y[j] + h * k3[j];
        f(t, k4);
        for (int j = 0; j < 3; ++j) y[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    }
    const auto x = run(m, DaeScheme::Radau5, 2e-7, T);
    std::printf("  Chua at 1 ms: V(n1) = %.6f, V(n2) = %.6f, iL = %.6e\n", y[0], y[1], y[2]);
    ut::approx(x[(size_t)m.idx("vc_c1")], y[0], 1e-6, "Chua (diodes, G): V(C1) = hand-written ODE");
    ut::approx(x[(size_t)m.idx("vc_c2")], y[1], 1e-6, "Chua: V(C2)");
    ut::approx(x[(size_t)m.idx("il_l1")], y[2], 1e-9, "Chua: iL");
}

}  // namespace

// --- хвосты этапа 9: MOSFET level 1, ёмкости переходов C(v), взаимная индуктивность K ----------------
void devices_tails() {
    // MOSFET: BETA = KP W / (2 L) = 1e-3; насыщение, линейная область, PMOS зеркально, эффект подложки.
    auto mos = [&](double vds, double vgs, double vb, bool p, const char* extra) {
        const char* s = p ? "-" : "";
        Imported m;
        char net[512];
        std::snprintf(net, sizeof(net), "* mos\nVd d 0 DC %s%g\nVg g 0 DC %s%g\nVb b 0 DC %s%g\nM1 d g 0 b mm W=10u L=10u\n"
                      "C1 d 0 1n\n.model mm %s(LEVEL=1 VTO=%s1 KP=2e-3 LAMBDA=0.01%s)\n",
                      s, vds, s, vgs, s, vb, p ? "PMOS" : "NMOS", s, extra);
        const auto X = op_point(net, m);
        return -X[(size_t)m.idx("i_vd")];
    };
    ut::approx(mos(5, 2, 0, false, ""), 1e-3 * 1 * (1 + 0.05), 1e-10, "NMOS saturation: BETA (vgs - VT)^2 (1 + LAMBDA vds)");
    ut::approx(mos(0.5, 3, 0, false, ""), 1e-3 * 0.5 * (4 - 0.5) * (1 + 0.005), 1e-10, "NMOS linear region");
    ut::approx(mos(5, 2, 0, true, ""), -1e-3 * 1 * (1 + 0.05), 1e-10, "PMOS = -NMOS");
    const double vt = 1 + 0.5 * (std::sqrt(0.6 + 1) - std::sqrt(0.6));
    ut::approx(mos(5, 3, -1, false, " GAMMA=0.5 PHI=0.6"), 1e-3 * (3 - vt) * (3 - vt) * (1 + 0.05), 1e-10,
               "NMOS body effect: VT = VTO + GAMMA (sqrt(PHI - vbs) - sqrt(PHI))");
    {   // LEVEL 3 — отказ, а не тихая подмена level 1
        SpiceImportResult r;
        std::string err;
        const bool ok = spice_import("* l3\nVd d 0 DC 1\nM1 d d 0 0 mm\nC1 d 0 1n\n.model mm NMOS(LEVEL=3)\n", {}, r, &err);
        ut::require(!ok && err.find("LEVEL") != std::string::npos, "MOSFET level 3 is refused: " + err);
    }
    std::printf("  MOSFET level 1: saturation, linear region, PMOS = -NMOS, body effect - exact\n");

    // Ёмкость перехода диода C(v): заряд через R до -5 В (обратное смещение) и до +1 В (выше FC VJ —
    // линейное продолжение), против скалярной ОДУ C(v) v' = (Vs - v)/R - Id(v).
    for (double vs : { -5.0, 1.0 }) {
        char net[256];
        std::snprintf(net, sizeof(net), "* cj\nV1 in 0 DC %g\nR1 in a 1k\nD1 a 0 dm\n.model dm D(IS=1e-14 CJO=10p VJ=0.7 M=0.5)\n"
                      ".tran 1n 50n uic\n", vs);
        const Imported m = import(net);
        const auto y = run(m, DaeScheme::Radau5, 1e-11, 50e-9);
        const double Vt = 0.025864186384551, cj0 = 10e-12, vj = 0.7, mj = 0.5, fc = 0.5;
        auto C = [&](double v) {
            const double vm = std::fmin(v, fc * vj);
            return cj0 * std::pow(1 - vm / vj, -mj) + cj0 * mj / (vj * std::pow(1 - fc, 1 + mj)) * std::fmax(v - fc * vj, 0.0);
        };
        auto Id = [&](double v) { const double u = v / Vt; return 1e-14 * ((u <= 40 ? std::exp(u) : std::exp(40.0) * (1 + u - 40)) - 1) + 1e-12 * v; };
        const double ref = ode_ref([&](double, double v) { return ((vs - v) / 1e3 - Id(v)) / C(v); }, 0.0, 50e-9, 50000);
        const double got = y[(size_t)m.idx("vc_d1_cj")];
        std::printf("  diode junction C(v), charge to %+g V: v(50 ns) = %.9f (ODE %.9f)\n", vs, got, ref);
        ut::approx(got, ref, 1e-6, vs < 0 ? "junction capacitance, reverse bias" : "junction capacitance above FC*VJ");
    }

    // K: L1 (1 мГн) от 1 В через 10 Ом, L2 (4 мГн) на 50 Ом, k = 0.9 — против ОДУ L i' = V(i).
    {
        const Imported m = import("* k\nV1 in 0 DC 1\nR1 in a 10\nL1 a 0 1m\nL2 b 0 4m\nR2 b 0 50\nK1 L1 L2 0.9\n.tran 1u 100u uic\n");
        ut::require(m.r.alg_text.find("dl_l1") != std::string::npos, "K: inductor derivatives become z");
        const auto y = run(m, DaeScheme::Radau5, 1e-8, 100e-6);
        const double L1 = 1e-3, L2 = 4e-3, M = 0.9 * std::sqrt(L1 * L2), det = L1 * L2 - M * M;
        auto f = [&](const double* i, double* d) {
            const double v1 = 1 - 10 * i[0], v2 = -50 * i[1];
            d[0] = (L2 * v1 - M * v2) / det;
            d[1] = (L1 * v2 - M * v1) / det;
        };
        double i[2] = { 0, 0 };
        const int n = 100000;
        const double h = 100e-6 / n;
        for (int s = 0; s < n; ++s) {
            double k1[2], k2[2], k3[2], k4[2], t[2];
            f(i, k1);
            for (int c = 0; c < 2; ++c) t[c] = i[c] + h / 2 * k1[c];
            f(t, k2);
            for (int c = 0; c < 2; ++c) t[c] = i[c] + h / 2 * k2[c];
            f(t, k3);
            for (int c = 0; c < 2; ++c) t[c] = i[c] + h * k3[c];
            f(t, k4);
            for (int c = 0; c < 2; ++c) i[c] += h / 6 * (k1[c] + 2 * k2[c] + 2 * k3[c] + k4[c]);
        }
        const double i1 = y[(size_t)m.idx("il_l1")], i2 = y[(size_t)m.idx("il_l2")];
        std::printf("  mutual inductance K: i1 = %.9f, i2 = %.9f (ODE %.9f %.9f)\n", i1, i2, i[0], i[1]);
        ut::approx(i1, i[0], 1e-7, "K: primary current");
        ut::approx(i2, i[1], 1e-7, "K: secondary current");
        SpiceImportResult r;
        std::string err;
        const bool ok = spice_import("* k1\nV1 a 0 DC 1\nL1 a 0 1m\nL2 b 0 1m\nR2 b 0 1\nK1 L1 L2 1.5\n", {}, r, &err);
        ut::require(!ok, "|K| > 1 is refused");
        SpiceImportResult r1;
        const bool ok1 = spice_import("* k1\nV1 a 0 DC 1\nL1 a 0 1m\nL2 b 0 1m\nR2 b 0 1\nK1 L1 L2 1\n", {}, r1, &err);
        std::string nt;
        for (const auto& s : r1.notes) nt += s;
        ut::require(ok1 && nt.find("index 2") != std::string::npos, "K = 1: imported with an index-2 note: " + err);
    }
}

// Условия в B: компаратор с гистерезисом нет — просто порог; ступенька copysign, без ветвлений.
void b_conditionals() {
    const double kPi = 3.14159265358979323846;
    const Imported m = import("* cond\nV1 in 0 SIN(0 1 1k)\nB1 a 0 V = V(in) > 0.5 ? 2 : -1\nR1 a c 1k\nC1 c 0 100n\n"
                              "B2 b 0 V = if(V(in) > 0 && !(V(in) >= 0.8), limit(5*V(in), 0, 3), uramp(-V(in)) + u(V(in) - 2))\n"
                              "R2 b 0 1k\n.tran 1u 0.7m uic\n");
    const double T = 0.7e-3;
    const auto y = run(m, DaeScheme::Radau5, 1e-8, T);
    auto s = [&](double t) { return std::sin(2 * kPi * 1e3 * t) > 0.5 ? 2.0 : -1.0; };
    const double ref = ode_ref([&](double t, double v) { return (s(t) - v) / 1e-4; }, 0.0, T, 700000);
    const double got = y[(size_t)m.idx("vc_c1")];
    const double vin = std::sin(2 * kPi * 1e3 * T);
    const double bref = (vin > 0 && !(vin >= 0.8)) ? std::fmin(std::fmax(5 * vin, 0.0), 3.0) : std::fmax(-vin, 0.0);
    const double bgot = y[(size_t)m.idx("v_b")];
    std::printf("  B conditionals: comparator -> RC v(0.7 ms) = %.7f (ODE %.7f); if/&&/!/limit/uramp node %.9f (%.9f)\n",
                got, ref, bgot, bref);
    ut::approx(got, ref, 2e-4, "B comparator ?: drives RC as the step function");   // разрыв: O(h) у обоих
    ut::approx(bgot, bref, 1e-9, "B if(), &&, !, limit(), uramp(), u()");
    SpiceImportResult r;
    std::string err;
    const bool ok = spice_import("* bad\nV1 in 0 1\nB1 a 0 V = V(in) = 1\nR1 a 0 1\nC1 a 0 1n\n", {}, r, &err);
    ut::require(!ok && err.find("==") != std::string::npos, "single '=' in B is refused: " + err);
}

// Сечения L-I: две катушки последовательно через узел без других ветвей; катушка с источником тока.
void li_cutsets() {
    auto notes = [](const std::string& net) {
        SpiceImportResult r;
        std::string err;
        { const bool ok_ = spice_import(net, {}, r, &err); ut::require(ok_, "import: " + err); }
        std::string all;
        for (const auto& n : r.notes) all += n + "\n";
        return all;
    };
    const std::string a = notes("* ll\nV1 in 0 DC 1\nR1 in a 10\nL1 a b 1m\nL2 b 0 2m\n");
    ut::require(a.find("cutset") != std::string::npos && a.find("L1") != std::string::npos, "series L-L node: cutset\n" + a);
    const std::string b = notes("* li\nI1 0 a DC 1m\nL1 a 0 1m\nR1 a c 1k\nC1 c 0 1n\nL2 c 0 1m\n");
    ut::require(b.find("cutset") == std::string::npos, "R joins the node: no cutset\n" + b);
    const std::string c = notes("* rl\nV1 in 0 DC 1\nR1 in a 10\nL1 a 0 1m\nR2 a 0 5\n");
    ut::require(c.find("cutset") == std::string::npos, "RL: no cutset\n" + c);
    std::printf("  L-I cutsets: series inductors through a bare node reported (index 2), RL circuits not\n");
}

// Группы параметров (только интерфейс): раскладка при импорте, слияние с правками пользователя при
// повторном импорте, правка групп, хранение в system.json.
void param_groups() {
    const std::string net =
        "* groups\nV1 in 0 SIN(0 1 1k)\nR1 in a 1k\nR2 a b 10k\nC1 b 0 10n\nL1 b c 1m\nD1 c 0 dmod\n"
        ".model dmod D(IS=1e-14 RS=1)\n"
        ".subckt amp1 p n o\nR1 p n 1meg\nE1 o 0 p n 1000\n.ends\n"
        "XU1 a b o1 amp1\nXU2 b 0 o2 amp1\nR3 o1 0 1k\nR4 o2 0 1k\n.tran 1u 1m\n";
    SpiceImportResult r;
    std::string err;
    { const bool ok_ = spice_import(net, {}, r, &err); ut::require(ok_, "import: " + err); }
    std::map<std::string, std::string> g(r.param_groups.begin(), r.param_groups.end());
    std::vector<std::string> names;
    for (const auto& pg : r.param_groups) names.push_back(pg.first);
    ut::require(names.size() == r.param_values.size(), "every parameter has a group entry");
    auto in = [&](const std::string& p, const std::string& path) {
        ut::require(g.count(p) && g[p] == path, p + " -> " + (g.count(p) ? g[p] : "?") + ", expected " + path);
    };
    in("r1", "Resistors"); in("r3", "Resistors"); in("c1", "Capacitors"); in("l1", "Inductors");
    in("v1_va", "Sources"); in("dmod_rs", "Models/dmod");
    int u1 = 0, u2 = 0;
    for (const auto& kv : g) { u1 += kv.second == "amp1/XU1"; u2 += kv.second == "amp1/XU2"; }
    ut::require(u1 == 2 && u2 == 2, "each amp1 instance has its own group of 2 (R1, E1)");

    // Пользователь переносит r1 и переименовывает группу; повторный импорт не трогает его правки.
    ParamGroups G;
    pg_merge_auto(G, names, r.param_groups);
    pg_move(G, "r1", "Input");
    pg_rename(G, "amp1", "Amplifiers");
    ut::require(pg_find(G, "Amplifiers/XU1") && !pg_find(G, "amp1/XU1"), "rename moves the subtree");
    std::vector<std::string> names2 = names;
    names2.erase(std::remove(names2.begin(), names2.end(), "r4"), names2.end());
    pg_merge_auto(G, names2, r.param_groups);
    ut::require(pg_group_of(G, "r1") == "Input", "re-import keeps the user's group");
    ut::require(pg_group_of(G, "r4").empty(), "re-import drops a removed parameter");
    ut::require(pg_group_of(G, "amp1.xu1_r1").empty() || pg_group_of(G, "amp1.xu1_r1") == "Amplifiers/XU1",
                "renamed instance group keeps its parameters");
    pg_remove(G, "Amplifiers");
    ut::require(!pg_find(G, "Amplifiers/XU2"), "remove takes the subtree");

    // system.json: строка "param_groups", туда и обратно без потерь.
    SystemRecord rec;
    rec.name = "groups";
    rec.param_groups = G;
    const SystemRecord back = record_from_json(record_to_json(rec));
    ut::require(pg_to_text(back.param_groups) == pg_to_text(G), "param_groups survive system.json");
    SystemRecord none;
    ut::require(record_to_json(none).find("param_groups") == std::string::npos, "no groups - no key (old files unchanged)");
    std::printf("  parameter groups: R / C / L / sources / Models/<model> / <subckt>/<instance> at import, user edits"
                " kept on re-import, system.json round trip (%zu groups)\n", G.size());

    // Порядок групп импорта — по виду элемента (не по алфавиту и не по netlist'у).
    std::vector<std::string> tops;
    for (const auto& pg : r.param_groups) {
        const std::string t = pg.second.substr(0, pg.second.find('/'));
        if (std::find(tops.begin(), tops.end(), t) == tops.end()) tops.push_back(t);
    }
    const std::vector<std::string> want_tops = { "Resistors", "Capacitors", "Inductors", "Sources", "amp1", "Models" };
    ut::require(tops == want_tops, "import group order: R, C, L, sources, subcircuits, models");

    // Группы начальных условий: x по элементу (переходы прибора — Devices/<прибор>), все z — Algebraic.
    {
        SpiceImportResult rq;
        const bool ok_ = spice_import("* ic groups\nV1 vcc 0 DC 5\nR1 vcc c 1k\nRB vcc b 100k\nQ1 c b 0 qm\nC1 c out 1u\n"
                                      "R2 out 0 1k\nL1 out 0 1m\n.model qm NPN(IS=1e-14 BF=100 CJE=1p CJC=1p)\n.tran 1u 1m\n",
                                      {}, rq, &err);
        ut::require(ok_, "import: " + err);
        std::map<std::string, std::string> ig(rq.ic_groups.begin(), rq.ic_groups.end());
        auto ic_in = [&](const std::string& v, const std::string& path) {
            ut::require(ig.count(v) && ig[v] == path, v + " -> " + (ig.count(v) ? ig[v] : "?") + ", expected " + path);
        };
        ic_in("vc_c1", "Capacitors"); ic_in("il_l1", "Inductors"); ic_in("vc_q1_cje", "Devices/Q1");
        ic_in("v_out", "Algebraic (computed)/Node potentials"); ic_in("ic_c1", "Algebraic (computed)/Branch currents");
        ut::require(ig.size() == rq.init_conditions.size(), "every variable has an IC group entry");
        auto lab = [&](const std::string& n, const std::string& want) {
            auto it = rq.field_labels.find(n);
            ut::require(it != rq.field_labels.end() && it->second == want,
                        "label " + n + " = " + (it == rq.field_labels.end() ? "?" : it->second) + ", expected " + want);
        };
        lab("vc_c1", "vc_{C1}"); lab("ic_c1", "ic_{C1}"); lab("il_l1", "il_{L1}"); lab("i_v1", "i_{V1}");
        lab("vc_q1_cje", "vc_{Q1,cje}"); lab("r1", "R_{1}"); lab("rb", "R_{B}"); lab("c1", "C_{1}"); lab("l1", "L_{1}");
        ut::require(rq.ic_groups.front().second == "Capacitors" && rq.ic_groups.back().second.rfind("Algebraic", 0) == 0,
                    "IC groups: states first, algebraic last");
    }
    // KiCad: неназванная цепь Net-(LED1-K) приходит как net-_led1-k_ — имя без префикса.
    {
        SpiceImportResult rn;
        const bool ok_ = spice_import("* net\nV1 in 0 DC 1\nR1 in net-_led1-k_ 1k\nC1 net-_led1-k_ 0 1u\n", {}, rn, &err);
        ut::require(ok_, "import: " + err);
        ut::require(rn.alg_text.find("v_led1_k") != std::string::npos, "net-_led1-k_ -> v_led1_k: " + rn.alg_text);
    }

    // Перетаскивание: порядок записей = порядок на экране, переставляются только соседи.
    ParamGroups D = pg_from_text("B: b1\nA: a1\nA/y: y1\nA/x: x1\nC: c1");
    ut::require(pg_rows(D) == std::vector<std::string>({ "B", "A", "A/y", "A/x", "C" }), "rows keep the file order");
    pg_reorder(D, "C", "B", false);
    pg_reorder(D, "A/x", "A/y", false);
    pg_reorder(D, "A/x", "B", false);   // не соседи — ничего
    ut::require(pg_rows(D) == std::vector<std::string>({ "C", "B", "A", "A/x", "A/y" }), "reorder among siblings");
    pg_reorder(D, "A", "C", true);
    ut::require(pg_rows(D) == std::vector<std::string>({ "C", "A", "A/x", "A/y", "B" }), "a group moves with its subtree");
    pg_get(D, "A/z");
    ut::require(pg_rows(D)[4] == "A/z", "a new subgroup goes after its siblings");
    ut::require(pg_to_text(pg_from_text(pg_to_text(D))) == pg_to_text(D), "order survives the text form");

    // Подписи полей и группы НУ в system.json.
    SystemRecord lr;
    lr.name = "labels";
    lr.ic_groups = pg_from_text("Capacitors: vc_c1\nAlgebraic (computed): v_out");
    lr.field_labels = { { "vc_c1", "V_{C1}" }, { "sigma", "\\sigma" } };
    const SystemRecord lb = record_from_json(record_to_json(lr));
    ut::require(pg_to_text(lb.ic_groups) == pg_to_text(lr.ic_groups) && lb.field_labels == lr.field_labels,
                "ic_groups and field_labels survive system.json");
    std::printf("  IC groups (Capacitors / Inductors / Devices/<dev> / Algebraic), KiCad net names, drag reorder,"
                " labels round trip\n");
}

// .ic без UIC (SPICE): рабочая точка с удержанным узлом — op_plain_text, KCL узла заменён на v - V = 0.
void ic_without_uic() {
    SpiceImportResult r;
    std::string err;
    { const bool ok_ = spice_import("* ic\nV1 in 0 DC 5\nR1 in a 1k\nC1 a 0 1u\nR2 a 0 1k\n.ic v(a)=1\n.tran 1u 1m\n", {}, r, &err);
      ut::require(ok_, "import: " + err); }
    ut::require(!r.uic && r.has_ic && !r.op_plain_text.empty(), ".ic without UIC gives the held-node DC OP equations");
    const System s = parse_dae(r.op_plain_text, r.vars_text, r.alg_text, r.params_text, "").as_codegen_system();
    std::vector<double> a = { 0.5 };
    const DaeSystem d = parse_dae(r.op_plain_text, r.vars_text, r.alg_text, r.params_text, "");
    for (const auto& p : d.params) a.push_back(parse_num(r.param_values.at(p), 0.0));
    a.insert(a.end(), { 4, 1e-10, 25, 1 });
    std::vector<double> X(s.vars.size(), 0.0);
    int it = 0;
    std::string why;
    { const bool ok_ = dae_dc_operating_point(s, a.data(), (int)a.size(), 0.0, X.data(), &it, &why);
      ut::require(ok_, "held DC OP: " + why); }
    double va = 0, vc = 0;
    for (size_t i = 0; i < s.vars.size(); ++i) {
        if (s.vars[i] == "v_a") va = X[i];
        if (s.vars[i] == "vc_c1") vc = X[i];
    }
    ut::approx(va, 1.0, 1e-12, ".ic node held at 1 V in the DC OP");
    ut::approx(vc, 1.0, 1e-12, "capacitor starts from the held node voltage (free OP would give 2.5 V)");
    SpiceImportResult u;
    { const bool ok_ = spice_import("* ic uic\nV1 in 0 DC 5\nR1 in a 1k\nC1 a 0 1u\n.ic v(a)=1\n.tran 1u 1m uic\n", {}, u, &err);
      ut::require(ok_, "import UIC: " + err); }
    ut::require(u.uic && u.op_plain_text.empty(), "UIC: no DC OP, .ic as given");
    std::printf("  .ic without UIC: DC OP with the node held (v(a) = %.3f V, vc = %.3f V); with UIC no DC OP\n", va, vc);
}

// Упрощения приборов (SpiceImportOptions): без ёмкостей переходов — меньше переменных состояния; идеальный
// диод — касательная при 10 мА (точен около этого тока); идеальный BJT — рабочая точка каскада близка к полной.
// Приставки СИ в полях (num_parse.h) и номиналы импорта с ними (fmt_num_si).
void si_prefixes() {
    auto eq = [](const std::string& s, double want) {
        double v = 0.0;
        ut::require(parse_num_checked(s, v) && v == want, "'" + s + "' -> " + std::to_string(v));
    };
    eq("10k", 1e4); eq("4.7u", 4.7e-6); eq("10n", 1e-8); eq("15p", 15e-12); eq("1m", 1e-3); eq("2M", 2e6);
    eq("1meg", 1e6); eq("1.5Meg", 1.5e6); eq("3G", 3e9); eq("1T", 1e12); eq("100f", 100e-15);
    eq("2.2\xC2\xB5", 2.2e-6); eq("1e-3", 1e-3); eq("2*1k", 2e3); eq("1k/4", 250.0); eq("-3.3k", -3300.0);
    double v = 0.0;
    for (const char* bad : { "2pi", "5asdf", "1kk", "k", "1x", "1mega" })
        ut::require(!parse_num_checked(bad, v), std::string("'") + bad + "' must stay invalid");
    ut::require(parse_num("2*pi", 0.0) == 2.0 * 3.1415926535897932384626433832795, "pi still works");
    for (double x : { 15000.0, 1e-8, 4.7e-6, 1e-3, 2.2e-12, 100.0, 1.5e6, 0.0, 1e-20, 12345.678 }) {
        const std::string s = fmt_num_si(x);
        ut::require(parse_num(s, -1.0) == x, "fmt_num_si round trip: " + s);
    }
    ut::require(fmt_num_si(15000.0) == "15k" && fmt_num_si(1e-8) == "10n" && fmt_num_si(4.7e-6) == "4.7u",
                "fmt_num_si: " + fmt_num_si(15000.0) + " " + fmt_num_si(1e-8) + " " + fmt_num_si(4.7e-6));
    SpiceImportResult r;
    std::string err;
    const bool ok_ = spice_import("* si\nV1 in 0 DC 1\nR1 in a 15k\nC1 a 0 10n\nL1 a 0 4.7m\n", {}, r, &err);
    ut::require(ok_, "import: " + err);
    ut::require(r.param_values.at("r1") == "15k" && r.param_values.at("c1") == "10n" && r.param_values.at("l1") == "4.7m",
                "import values: " + r.param_values.at("r1") + " " + r.param_values.at("c1") + " " + r.param_values.at("l1"));
    std::printf("  SI prefixes: f p n u/\xC2\xB5 m k M/meg G T, exact (10n == 1e-8), import values 15k / 10n / 4.7m\n");
}

void device_simplifications() {
    const std::string rect = "* rect\nV1 in 0 SIN(0 5 50)\nD1 in out dm\nR1 out 0 330\nC1 out 0 10u\n"
                             ".model dm D(IS=1e-14 N=1.5 RS=0.5 CJO=5p)\n.tran 1u 20m\n";
    SpiceImportResult full, nocap, ideal;
    std::string err;
    SpiceImportOptions o;
    { const bool ok_ = spice_import(rect, o, full, &err); ut::require(ok_, "full: " + err); }
    o.no_junction_caps = true;
    { const bool ok_ = spice_import(rect, o, nocap, &err); ut::require(ok_, "no caps: " + err); }
    o.ideal_diodes = true;
    { const bool ok_ = spice_import(rect, o, ideal, &err); ut::require(ok_, "ideal: " + err); }
    auto count = [](const std::string& s) { return s.empty() ? 0 : (int)std::count(s.begin(), s.end(), ',') + 1; };
    ut::require(count(full.vars_text) == 2 && count(nocap.vars_text) == 1, "no junction caps: the diode's C(v) state is gone");
    ut::require(ideal.param_values.count("dm_vf") && ideal.param_values.count("dm_ron") && !ideal.param_values.count("dm_is"),
                "ideal diode: parameters dm_vf / dm_ron instead of IS / N / RS");
    // Ток идеального диода при 10 мА — как у экспоненты с RS (касательная в этой точке)
    const IdealJunction f = ideal_junction_fit(1e-14, 1.5, 0.5);
    const double v10 = 1.5 * kDeviceVt * std::log(1e-2 / 1e-14 + 1) + 0.5 * 1e-2;
    ut::approx((v10 - f.vf) / f.ron, 1e-2, 1e-12, "ideal diode: exact at 10 mA");
    // Идеальный BJT: рабочая точка каскада ОЭ против полной модели
    const std::string ce = "* ce\nVcc vcc 0 DC 12\nRb vcc b 470k\nRc vcc c 2.2k\nQ1 c b 0 qm\n"
                           ".model qm NPN(IS=1e-14 BF=150 BR=3 VAF=100 RB=10 RC=1 RE=0.5 CJE=10p CJC=5p)\n.tran 1u 1m\n";
    auto op_vc = [&](bool ideal_q) {
        SpiceImportOptions q;
        q.ideal_bjts = ideal_q;
        SpiceImportResult r;
        { const bool ok_ = spice_import(ce, q, r, &err); ut::require(ok_, "CE: " + err); }
        const DaeSystem d = parse_dae(r.plain_text, r.vars_text, r.alg_text, r.params_text, "");
        const System s = d.as_codegen_system();
        std::vector<double> a = { 0.5 };
        for (const auto& p : d.params) a.push_back(parse_num(r.param_values.at(p), 0.0));
        a.insert(a.end(), { 4, 1e-10, 25, 1 });
        std::vector<double> X(s.vars.size(), 0.0);
        int it = 0;
        std::string why;
        { const bool ok_ = dae_dc_operating_point(s, a.data(), (int)a.size(), 0.0, X.data(), &it, &why);
          ut::require(ok_, "CE OP: " + why); }
        for (size_t i = 0; i < s.vars.size(); ++i) if (s.vars[i] == "v_c") return X[i];
        return -1.0;
    };
    const double vf = op_vc(false), vi = op_vc(true);
    std::printf("  device simplifications: no junction caps -1 state; ideal diode exact at 10 mA (Vf %.3f V, Ron %.2f Ohm); "
                "ideal BJT CE stage Vc %.3f V (full model %.3f V)\n", f.vf, f.ron, vi, vf);
    ut::approx(vi, vf, 0.1 * vf, "ideal BJT: CE operating point within 10% of the full model");
}

int main() {
    return ut::run_main("spice", []() -> int {
        sources();
        subcircuits();
        round_trip();
        devices();
        chua();
        devices_9d();
        devices_tails();
        b_conditionals();
        li_cutsets();
        param_groups();
        ic_without_uic();
        device_simplifications();
        si_prefixes();
        return 0;
    });
}
