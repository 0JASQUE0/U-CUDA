#include "circuit_synth.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

// Канонический порядок мономов: свободный, линейные по индексу, квадратичные по паре.
long long term_key(int v1, int v2) {
    return (long long)(v1 + 1) * 1024 + (v2 + 1);
}

struct Probe {
    SystemEvaluator ev;
    int             n;
    std::vector<double> a;

    Probe(const System& sys, const std::vector<double>& values)
        : ev(sys), n((int)sys.vars.size()), a(values) {}

    void at(const std::vector<double>& X, std::vector<double>& out) {
        out.assign(n, 0.0);
        ev.eval(X.data(), a.data(), out.data());
    }
};

double poly_value(const PolyRhs& p, const std::vector<double>& X) {
    double s = 0.0;
    for (const PolyTerm& t : p.terms) {
        double v = t.coeff;
        if (t.v1 >= 0) v *= X[t.v1];
        if (t.v2 >= 0) v *= X[t.v2];
        s += v;
    }
    return s;
}

std::string var_or_num(const System& sys, int v) {
    return v >= 0 ? sys.vars[(size_t)v] : std::string("1");
}

std::string term_text(const System& sys, const PolyTerm& t) {
    std::ostringstream o;
    o.setf(std::ios::showpos);
    o << t.coeff;
    o.unsetf(std::ios::showpos);
    if (t.v1 >= 0) o << "*" << var_or_num(sys, t.v1);
    if (t.v2 >= 0) o << "*" << var_or_num(sys, t.v2);
    return o.str();
}

}  // namespace

// --- извлечение мономов -------------------------------------------------------

bool extract_quadratic(const System& sys,
                       const std::vector<double>& values,
                       const PolyExtractConfig& cfg,
                       std::vector<PolyRhs>& out,
                       std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    const int n = (int)sys.vars.size();
    if (n <= 0)                          return fail("circuit synthesis: system has no state variables");
    if (sys.is_map)                      return fail("circuit synthesis: a discrete map has no circuit form");
    if (values.size() < sys.params.size() + 1)
        return fail("circuit synthesis: parameter array is shorter than the system needs");

    std::unique_ptr<Probe> probe;
    try {
        probe.reset(new Probe(sys, values));
    } catch (const std::exception& e) {
        return fail(std::string("circuit synthesis: cannot parse the system - ") + e.what());
    }
    // floor/ceil/fmod: не полиномы, но невязка на них шумит — отказать понятнее здесь.
    if (!probe->ev.has_jacobian())
        return fail("circuit synthesis: the system uses floor/ceil/fmod, which no op-amp circuit realizes");

    const double r = cfg.probe_radius > 0.0 ? cfg.probe_radius : 1.0;

    std::vector<double> X(n, 0.0), f0, fp, fm, fij;
    probe->at(X, f0);

    // b[eq][i] — линейный коэффициент, q[eq][i][j] — квадратичный (j >= i).
    std::vector<std::vector<double>> b(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<std::vector<double>>> q(
        n, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));

    for (int i = 0; i < n; ++i) {
        X.assign(n, 0.0); X[i] = +r; probe->at(X, fp);
        X.assign(n, 0.0); X[i] = -r; probe->at(X, fm);
        for (int e = 0; e < n; ++e) {
            b[e][i]    = (fp[e] - fm[e]) / (2.0 * r);
            q[e][i][i] = ((fp[e] + fm[e]) * 0.5 - f0[e]) / (r * r);
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            X.assign(n, 0.0); X[i] = r; X[j] = r; probe->at(X, fij);
            for (int e = 0; e < n; ++e) {
                q[e][i][j] = (fij[e] - f0[e] - r * b[e][i] - r * b[e][j]
                              - r * r * q[e][i][i] - r * r * q[e][j][j]) / (r * r);
            }
        }
    }

    // Порог отсева — относительный: коэффициенты у Лоренца различаются на два порядка.
    std::vector<double> scale(n, 1.0);
    for (int e = 0; e < n; ++e) {
        double m = std::fabs(f0[e]);
        for (int i = 0; i < n; ++i) {
            m = std::max(m, std::fabs(b[e][i]));
            for (int j = i; j < n; ++j) m = std::max(m, std::fabs(q[e][i][j]));
        }
        scale[e] = m > 0.0 ? m : 1.0;
    }

    out.assign(n, PolyRhs());
    for (int e = 0; e < n; ++e) {
        const double eps = cfg.zero_tol * scale[e];
        if (std::fabs(f0[e]) > eps) out[e].terms.push_back(PolyTerm{ -1, -1, f0[e] });
        for (int i = 0; i < n; ++i)
            if (std::fabs(b[e][i]) > eps) out[e].terms.push_back(PolyTerm{ i, -1, b[e][i] });
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j)
                if (std::fabs(q[e][i][j]) > eps) out[e].terms.push_back(PolyTerm{ i, j, q[e][i][j] });
    }

    // Проверка, что это вообще полином степени <= 2: восстанавливаем и сравниваем.
    std::mt19937 rng(20260920u);
    std::uniform_real_distribution<double> uni(-r, r);
    std::vector<double> fx;
    for (int s = 0; s < cfg.residual_samples; ++s) {
        for (int i = 0; i < n; ++i) X[i] = uni(rng);
        probe->at(X, fx);
        for (int e = 0; e < n; ++e) {
            const double approx = poly_value(out[e], X);
            const double tol    = cfg.residual_tol * (1.0 + std::fabs(fx[e]) + scale[e]);
            if (std::fabs(approx - fx[e]) > tol) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "circuit synthesis: equation %d for '%s' is not a polynomial of degree <= 2 "
                    "(residual %.3g at a probe point); cubic and transcendental terms are not realizable yet",
                    e + 1, sys.vars[(size_t)e].c_str(), std::fabs(approx - fx[e]));
                return fail(buf);
            }
        }
    }
    return true;
}

bool extract_quadratic_union(const System& sys,
                             const std::vector<std::vector<double>>& values_sets,
                             const PolyExtractConfig& cfg,
                             std::vector<PolyRhs>& out,
                             std::string* err) {
    if (values_sets.empty()) { if (err) *err = "circuit synthesis: no parameter set given"; return false; }

    std::vector<std::vector<PolyRhs>> all;
    all.reserve(values_sets.size());
    for (const std::vector<double>& v : values_sets) {
        std::vector<PolyRhs> p;
        if (!extract_quadratic(sys, v, cfg, p, err)) return false;
        all.push_back(std::move(p));
    }

    const int n = (int)sys.vars.size();
    out.assign(n, PolyRhs());
    for (int e = 0; e < n; ++e) {
        // Структура — объединение; число — из первого набора, даже если оно нулевое.
        std::map<long long, PolyTerm> merged;
        for (size_t s = 0; s < all.size(); ++s) {
            for (const PolyTerm& t : all[s][e].terms) {
                const long long k = term_key(t.v1, t.v2);
                auto it = merged.find(k);
                if (it == merged.end()) {
                    PolyTerm nt = t;
                    if (s != 0) nt.coeff = 0.0;   // в базовом наборе монома нет
                    merged[k] = nt;
                }
            }
        }
        // Значения базового набора перекрывают всё, что пришло из остальных.
        for (const PolyTerm& t : all[0][e].terms) merged[term_key(t.v1, t.v2)].coeff = t.coeff;
        for (const auto& kv : merged) out[e].terms.push_back(kv.second);
    }
    return true;
}

// --- синтез графа -------------------------------------------------------------

int CircuitGraph::add_node(const std::string& name) {
    nodes.push_back(CircuitNode{ name });
    return (int)nodes.size() - 1;
}

bool synthesize_circuit(const System& sys,
                        const std::vector<PolyRhs>& rhs,
                        const SynthesisConfig& cfg,
                        CircuitGraph& g,
                        std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };

    const int n = (int)sys.vars.size();
    if ((int)rhs.size() != n) return fail("circuit synthesis: right-hand side count does not match the system");
    if (cfg.r_unit <= 0.0 || cfg.c_int <= 0.0)
        return fail("circuit synthesis: unit resistor and integrator capacitor must be positive");

    g = CircuitGraph();
    g.add_node("gnd");
    g.time_scale = 1.0 / (cfg.r_unit * cfg.c_int);

    // Интеграторы: выход несёт +x_i, суммирующий узел — виртуальная земля.
    std::vector<int> sum_node(n, 0);
    g.var_node.assign(n, 0);
    g.var_node_inv.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        const std::string& v = sys.vars[(size_t)i];
        g.var_node[i] = g.add_node("n_" + v);
        sum_node[i]   = g.add_node("s_" + v);
        g.capacitors.push_back(CircuitCapacitor{ sum_node[i], g.var_node[i], cfg.c_int });
        g.opamps.push_back(CircuitOpAmp{ 0, sum_node[i], g.var_node[i] });
    }

    // Инвертор заводится по требованию: так топология совпадает с ручной эталонной.
    auto inverted = [&](int i) -> int {
        if (g.var_node_inv[i] >= 0) return g.var_node_inv[i];
        const std::string& v = sys.vars[(size_t)i];
        const int s   = g.add_node("s_i" + v);
        const int out = g.add_node("n_" + v + "n");
        g.resistors.push_back(CircuitResistor{ g.var_node[i], s, 1.0 / cfg.r_unit, "inv " + v + " in" });
        g.resistors.push_back(CircuitResistor{ s, out,         1.0 / cfg.r_unit, "inv " + v + " fb" });
        g.opamps.push_back(CircuitOpAmp{ 0, s, out });
        g.var_node_inv[i] = out;
        return out;
    };
    if (cfg.always_invert)
        for (int i = 0; i < n; ++i) inverted(i);

    // Две опоры под свободные члены, каждая заводится по требованию: знак задаётся
    // полярностью опоры, потому что резистор несёт только модуль коэффициента.
    int ref_neg = -1, ref_pos = -1;
    auto ref_node = [&](double sign) {
        int& slot = (sign > 0.0) ? ref_neg : ref_pos;
        if (slot < 0) {
            slot = g.add_node(sign > 0.0 ? "n_refn" : "n_refp");
            g.vsources.push_back(CircuitVSource{ slot, sign > 0.0 ? -1.0 : +1.0 });
        }
        return slot;
    };

    for (int e = 0; e < n; ++e) {
        for (const PolyTerm& t : rhs[(size_t)e].terms) {
            const double c    = t.coeff;
            const double sign = (c < 0.0) ? -1.0 : 1.0;
            const std::string origin = sys.vars[(size_t)e] + "' : " + term_text(sys, t);

            if (t.v1 < 0 && t.v2 < 0) {
                g.resistors.push_back(CircuitResistor{ ref_node(sign), sum_node[e],
                                                       std::fabs(c) / cfg.r_unit, origin });
                continue;
            }

            if (t.v2 < 0) {
                // Линейный член: интегратор инвертирует, поэтому положительный
                // коэффициент питается от -x, отрицательный от +x.
                const int src = (sign > 0.0) ? inverted(t.v1) : g.var_node[t.v1];
                g.resistors.push_back(CircuitResistor{ src, sum_node[e],
                                                       std::fabs(c) / cfg.r_unit, origin });
                continue;
            }

            // Квадратичный член. AD633 делит на 10, поэтому эффективный коэффициент 10*|c|.
            // Знак задаётся полярностью входов умножителя, а не выбором узла-источника.
            int mx = g.var_node[t.v1];
            int my = g.var_node[t.v2];
            if (sign > 0.0) {
                // Нужен -x*y/10: инвертируем один вход. Предпочитаем тот, у кого
                // инвертор уже есть — иначе младший индекс, чтобы выбор был детерминирован.
                if      (g.var_node_inv[t.v1] >= 0) mx = g.var_node_inv[t.v1];
                else if (g.var_node_inv[t.v2] >= 0) my = g.var_node_inv[t.v2];
                else                                mx = inverted(t.v1);
            }
            char mname[64];
            std::snprintf(mname, sizeof(mname), "n_m%d", (int)g.multipliers.size() + 1);
            const int mout = g.add_node(mname);
            g.multipliers.push_back(CircuitMultiplier{ mx, 0, my, 0, 0, mout });
            g.resistors.push_back(CircuitResistor{ mout, sum_node[e],
                                                   10.0 * std::fabs(c) / cfg.r_unit, origin });
        }
    }
    return true;
}

// --- диагностика --------------------------------------------------------------

std::string circuit_topology_signature(const CircuitGraph& g) {
    auto nm = [&](int i) { return g.nodes[(size_t)i].name; };
    std::vector<std::string> lines;
    char buf[256];
    for (const CircuitResistor& r : g.resistors) {
        std::snprintf(buf, sizeof(buf), "R %s %s %.12g", nm(r.a).c_str(), nm(r.b).c_str(), r.siemens);
        lines.push_back(buf);
    }
    for (const CircuitCapacitor& c : g.capacitors) {
        std::snprintf(buf, sizeof(buf), "C %s %s %.12g", nm(c.a).c_str(), nm(c.b).c_str(), c.farad);
        lines.push_back(buf);
    }
    for (const CircuitOpAmp& o : g.opamps) {
        std::snprintf(buf, sizeof(buf), "E %s %s %s", nm(o.out).c_str(), nm(o.in_plus).c_str(), nm(o.in_minus).c_str());
        lines.push_back(buf);
    }
    for (const CircuitMultiplier& m : g.multipliers) {
        std::snprintf(buf, sizeof(buf), "M %s %s %s", nm(m.out).c_str(), nm(m.x1).c_str(), nm(m.y1).c_str());
        lines.push_back(buf);
    }
    for (const CircuitVSource& v : g.vsources) {
        std::snprintf(buf, sizeof(buf), "V %s %.12g", nm(v.node).c_str(), v.volt);
        lines.push_back(buf);
    }
    std::sort(lines.begin(), lines.end());
    std::ostringstream o;
    for (const std::string& s : lines) o << s << "\n";
    return o.str();
}

std::string circuit_dump(const CircuitGraph& g) {
    auto nm = [&](int i) { return g.nodes[(size_t)i].name; };
    std::ostringstream o;
    char buf[320];
    std::snprintf(buf, sizeof(buf), "nodes %d, R %d, C %d, opamp %d, mult %d, vsrc %d, time scale %.6g 1/s\n",
                  (int)g.nodes.size(), (int)g.resistors.size(), (int)g.capacitors.size(),
                  (int)g.opamps.size(), (int)g.multipliers.size(), (int)g.vsources.size(), g.time_scale);
    o << buf;
    for (const CircuitResistor& r : g.resistors) {
        std::snprintf(buf, sizeof(buf), "  R %-8s -> %-8s  %12.6g Ohm   %s\n",
                      nm(r.a).c_str(), nm(r.b).c_str(), r.ohm(), r.origin.c_str());
        o << buf;
    }
    for (const CircuitCapacitor& c : g.capacitors) {
        std::snprintf(buf, sizeof(buf), "  C %-8s -> %-8s  %12.6g F\n", nm(c.a).c_str(), nm(c.b).c_str(), c.farad);
        o << buf;
    }
    for (const CircuitMultiplier& m : g.multipliers) {
        std::snprintf(buf, sizeof(buf), "  M %-8s = %s * %s / 10\n",
                      nm(m.out).c_str(), nm(m.x1).c_str(), nm(m.y1).c_str());
        o << buf;
    }
    return o.str();
}

bool circuit_check(const CircuitGraph& g, std::vector<std::string>& problems) {
    problems.clear();
    const int nn = (int)g.nodes.size();
    std::vector<int> degree(nn, 0), driven(nn, 0);

    for (const CircuitResistor& r : g.resistors) { ++degree[r.a]; ++degree[r.b]; }
    for (const CircuitCapacitor& c : g.capacitors) { ++degree[c.a]; ++degree[c.b]; }
    for (const CircuitOpAmp& o : g.opamps) { ++degree[o.out]; ++driven[o.out]; ++degree[o.in_minus]; }
    for (const CircuitMultiplier& m : g.multipliers) {
        ++degree[m.out]; ++driven[m.out];
        ++degree[m.x1]; ++degree[m.y1];
    }
    for (const CircuitVSource& v : g.vsources) { ++degree[v.node]; ++driven[v.node]; }

    for (int i = 1; i < nn; ++i) {
        if (degree[i] == 0)
            problems.push_back("node '" + g.nodes[(size_t)i].name + "' is not connected to anything");
        else if (degree[i] == 1 && !driven[i])
            problems.push_back("node '" + g.nodes[(size_t)i].name + "' is dangling (one connection, nothing drives it)");
        if (driven[i] > 1)
            problems.push_back("node '" + g.nodes[(size_t)i].name + "' is driven by more than one source");
    }
    return problems.empty();
}

// --- самопроверка против эталонной фикстуры -----------------------------------

std::string circuit_synth_selftest(bool* ok) {
    std::ostringstream o;
    bool good = true;
    auto check = [&](bool cond, const std::string& what) {
        o << (cond ? "  ok   " : "  FAIL ") << what << "\n";
        if (!cond) good = false;
    };

    // Масштабированный Лоренц из tests/fixtures/lorenz_reference.cir.
    System sys;
    sys.vars   = { "u", "v", "w" };
    sys.params = { "sigma", "r", "b" };
    sys.rhs    = { "sigma*(v - u)", "r*u - v - 20*u*w", "5*u*v - b*w" };

    const std::vector<double> vals = { 0.0, 16.0, 45.6, 4.0 };   // a[0] зарезервирован

    PolyExtractConfig pc;
    std::vector<PolyRhs> poly;
    std::string err;
    if (!extract_quadratic(sys, vals, pc, poly, &err)) {
        o << "  FAIL extraction: " << err << "\n";
        if (ok) *ok = false;
        return o.str();
    }

    o << "monomials:\n";
    for (size_t e = 0; e < poly.size(); ++e) {
        o << "  d" << sys.vars[e] << "/dt =";
        for (const PolyTerm& t : poly[e].terms) o << " " << term_text(sys, t);
        o << "\n";
    }

    check(poly[0].terms.size() == 2, "du/dt has 2 monomials");
    check(poly[1].terms.size() == 3, "dv/dt has 3 monomials");
    check(poly[2].terms.size() == 2, "dw/dt has 2 monomials");

    auto coeff_of = [&](int e, int v1, int v2) {
        for (const PolyTerm& t : poly[(size_t)e].terms)
            if (t.v1 == v1 && t.v2 == v2) return t.coeff;
        return 0.0;
    };
    auto near = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * (1.0 + std::fabs(b)); };

    check(near(coeff_of(0, 1, -1),  16.0), "coefficient of v in du/dt is +sigma");
    check(near(coeff_of(0, 0, -1), -16.0), "coefficient of u in du/dt is -sigma");
    check(near(coeff_of(1, 0, -1),  45.6), "coefficient of u in dv/dt is +r");
    check(near(coeff_of(1, 1, -1),  -1.0), "coefficient of v in dv/dt is -1");
    check(near(coeff_of(1, 0,  2), -20.0), "coefficient of u*w in dv/dt is -20");
    check(near(coeff_of(2, 0,  1),   5.0), "coefficient of u*v in dw/dt is +5");
    check(near(coeff_of(2, 2, -1),  -4.0), "coefficient of w in dw/dt is -b");

    SynthesisConfig sc;
    CircuitGraph g;
    if (!synthesize_circuit(sys, poly, sc, g, &err)) {
        o << "  FAIL synthesis: " << err << "\n";
        if (ok) *ok = false;
        return o.str();
    }

    o << circuit_dump(g);

    check(g.opamps.size()      == 5, "5 op-amps (3 integrators + 2 inverters)");
    check(g.multipliers.size() == 2, "2 multipliers");
    check(g.capacitors.size()  == 3, "3 integrator capacitors");
    check(g.resistors.size()   == 11, "11 resistors (7 summing + 2x2 inverter)");
    check(g.vsources.empty(),        "no reference source (no constant terms)");
    check(std::fabs(g.time_scale - 1000.0) < 1e-9, "time scale is 1000 1/s");

    // Номиналы против поузловой таблицы в tests/fixtures/README.md.
    auto has_r = [&](const char* from, const char* to, double ohm) {
        for (const CircuitResistor& r : g.resistors) {
            if (g.nodes[(size_t)r.a].name == from && g.nodes[(size_t)r.b].name == to
                && std::fabs(r.ohm() - ohm) <= 1e-6 * ohm) return true;
        }
        return false;
    };
    check(has_r("n_vn", "s_u", 6250.0),              "R(-v -> integrator u) = 6250 Ohm");
    check(has_r("n_u",  "s_u", 6250.0),              "R(+u -> integrator u) = 6250 Ohm");
    check(has_r("n_un", "s_v", 100000.0 / 45.6),     "R(-u -> integrator v) = 100k/45.6");
    check(has_r("n_v",  "s_v", 100000.0),            "R(+v -> integrator v) = 100k");
    check(has_r("n_m1", "s_v", 500.0),               "R(u*w -> integrator v) = 500 Ohm");
    check(has_r("n_m2", "s_w", 2000.0),              "R(u*v -> integrator w) = 2 kOhm");
    check(has_r("n_w",  "s_w", 25000.0),             "R(+w -> integrator w) = 25 kOhm");

    // Полярности умножителей: -20*u*w берётся неинвертированным, +5*u*v — с инверсией.
    check(g.multipliers.size() == 2
          && g.nodes[(size_t)g.multipliers[0].x1].name == "n_u"
          && g.nodes[(size_t)g.multipliers[0].y1].name == "n_w",
          "multiplier 1 fed with +u and +w");
    check(g.multipliers.size() == 2
          && g.nodes[(size_t)g.multipliers[1].x1].name == "n_un"
          && g.nodes[(size_t)g.multipliers[1].y1].name == "n_v",
          "multiplier 2 fed with -u and +v");

    std::vector<std::string> problems;
    check(circuit_check(g, problems), "topology check reports no problems");
    for (const std::string& p : problems) o << "    " << p << "\n";

    // Отказ на нереализуемом: кубический моном.
    System cubic = sys;
    cubic.rhs[2] = "5*u*v*w - b*w";
    std::vector<PolyRhs> dummy;
    std::string cerr;
    check(!extract_quadratic(cubic, vals, pc, dummy, &cerr), "cubic monomial is refused");
    o << "    refusal text: " << cerr << "\n";

    // Отказ на трансцендентном.
    System trig = sys;
    trig.rhs[0] = "sigma*(v - u) + sin(w)";
    check(!extract_quadratic(trig, vals, pc, dummy, &cerr), "transcendental term is refused");

    // Объединение по наборам параметров: sigma = 0 не должна убирать элементы.
    std::vector<std::vector<double>> sets = { { 0.0, 0.0, 45.6, 4.0 }, { 0.0, 16.0, 45.6, 4.0 } };
    std::vector<PolyRhs> upoly;
    if (extract_quadratic_union(sys, sets, pc, upoly, &err)) {
        check(upoly[0].terms.size() == 2, "union keeps both monomials of du/dt at sigma = 0");
        check(std::fabs(upoly[0].terms[0].coeff) < 1e-12, "their coefficients are zero in the base set");
    } else {
        check(false, std::string("union extraction: ") + err);
    }

    if (ok) *ok = good;
    o << (good ? "SELFTEST PASSED\n" : "SELFTEST FAILED\n");
    return o.str();
}
