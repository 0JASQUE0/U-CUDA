#include "neural_session.h"
#include "num_parse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

void log_neural_run(const char* label, bool ok, double secs) {
    std::printf("[run] %s %s in %.2fs\n", ok ? "Done" : "Cancelled", (label && *label) ? label : "(unnamed)", secs);
    std::fflush(stdout);
}

// Запас от int в CSR и от памяти хоста: полный граф на 100 000 узлов отказывает здесь, а не в malloc.
constexpr long long kMaxArcs = 500000000LL;

double num_or(const std::string& s, double fallback) {
    return s.empty() ? fallback : parse_num(s, fallback);
}
int int_or(const std::string& s, int fallback) {
    const double v = num_or(s, (double)fallback);
    if (!std::isfinite(v) || std::fabs(v) > 2.0e9) throw std::runtime_error("number out of range: " + s);
    return (int)std::llround(v);
}

uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// SplitMix64: один поток на (seed, a, b), порядок обхода на числа не влияет. Начало проходит через mix:
// при линейном по a, b начале соседние узлы получали одну последовательность со сдвигом.
struct Stream {
    uint64_t s;
    Stream(uint64_t seed, uint64_t a, uint64_t b) {
        s = mix64(seed + 0x9E3779B97F4A7C15ULL);
        s = mix64(s ^ (a + 1) * 0xC2B2AE3D27D4EB4FULL);
        s = mix64(s ^ (b + 1) * 0x165667B19E3779F9ULL);
    }
    uint64_t next() { return mix64(s += 0x9E3779B97F4A7C15ULL); }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
    int below(int n) { return (int)(uni() * (double)n); }
};

// Среднее по времени конечных значений; у остановленного экземпляра NaN, а не среднее по началу.
void neural_sweep_summary(NeuralOutcome& out, int order_var) {
    const NeuralResult& r = out.result;
    const int nInst = (int)r.instance_stop.size(), K = out.pops_per_instance;
    if (nInst < 1 || K < 1 || r.n_points < 1) return;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int nPops = nInst * K, var = order_var >= 0 ? order_var : 0;
    out.sweep_order.assign((size_t)nPops, nan);
    out.sweep_spread.assign((size_t)nPops, nan);
    for (int q = 0; q < nPops; ++q) {
        if (r.instance_stop[(size_t)(q / K)] >= 0) continue;
        double so = 0.0, ss = 0.0;
        int no = 0, ns = 0;
        for (int p = 0; p < r.n_points; ++p) {
            if (!r.order.empty()) {
                const double v = r.order[(size_t)p * nPops + q];
                if (std::isfinite(v)) { so += v; ++no; }
            }
            if (!r.spread.empty()) {
                const double v = r.spread[((size_t)p * nPops + q) * r.n_vars + var];
                if (std::isfinite(v)) { ss += v; ++ns; }
            }
        }
        if (no > 0) out.sweep_order[(size_t)q] = so / no;
        if (ns > 0) out.sweep_spread[(size_t)q] = ss / ns;
    }
}

struct Rows {
    std::vector<long long> start;   // [n_to + 1]
    std::vector<int> src;           // локальные номера источника
    double expected = 0.0;          // ожидаемая степень
};

// Округлённое вниз log(1 - u)/log(1 - p) не обходит каждую пару: O(дуг), а не O(N^2).
void row_probability(Stream& g, int n_from, int self, double p, std::vector<int>& row) {
    if (p <= 0.0) return;
    if (p >= 1.0) { for (int s = 0; s < n_from; ++s) if (s != self) row.push_back(s); return; }
    const double lq = std::log1p(-p);
    for (long long s = -1;;) {
        const double step = std::floor(std::log1p(-g.uni()) / lq);
        if (step >= (double)n_from) break;
        s += 1 + (long long)step;
        if (s >= n_from) break;
        if ((int)s != self) row.push_back((int)s);
    }
}
// m различных чисел из [0, avail), self пропускается; большая степень считается через дополнение.
void row_fixed(Stream& g, int n_from, int self, int d, std::vector<int>& row, std::vector<int>& tmp) {
    const int avail = n_from - (self >= 0 ? 1 : 0);
    const bool invert = d > avail / 2;
    const int m = invert ? avail - d : d;
    tmp.clear();
    while ((int)tmp.size() < m) {
        for (int need = m - (int)tmp.size(); need > 0; --need) tmp.push_back(g.below(avail));
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    }
    const auto real = [&](int v) { return (self >= 0 && v >= self) ? v + 1 : v; };
    if (!invert) { for (int v : tmp) row.push_back(real(v)); return; }
    size_t at = 0;
    for (int v = 0; v < avail; ++v) {
        if (at < tmp.size() && tmp[at] == v) { ++at; continue; }
        row.push_back(real(v));
    }
}
void row_regular(const NeuralProjection& p, int n, int w, int h, int k, int i, std::vector<int>& row) {
    if (p.rule == NeuralRule::Chain) {
        if (i > 0) row.push_back(i - 1);
        if (!p.directed && i + 1 < n) row.push_back(i + 1);
    } else if (p.rule == NeuralRule::Ring) {
        for (int j = 1; j <= k; ++j) {
            row.push_back((i + n - j % n) % n);
            if (!p.directed) row.push_back((i + j) % n);
        }
    } else {
        // Как у Network: тор замыкается только при стороне больше 2, иначе дуга уже есть.
        const int x = i % w, y = i / w;
        const auto at = [&](int ax, int ay) {
            if (ax < 0 || ax >= w) { if (!p.periodic || w <= 2) return -1; ax = (ax + w) % w; }
            if (ay < 0 || ay >= h) { if (!p.periodic || h <= 2) return -1; ay = (ay + h) % h; }
            return ay * w + ax;
        };
        const auto add = [&](int dx, int dy) { const int v = at(x + dx, y + dy); if (v >= 0) row.push_back(v); };
        add(-1, 0); add(0, -1);
        if (!p.directed) { add(1, 0); add(0, 1); }
        if (p.rule == NeuralRule::King) {
            add(-1, -1); add(1, -1);
            if (!p.directed) { add(1, 1); add(-1, 1); }
        }
    }
}
// Уоттс-Строгац: кольцо, затем конец ребра переносится с вероятностью p без петель и повторов.
void rows_small_world(const NeuralProjection& p, uint64_t seed, int proj, int n, int k, double rewire, Rows& r) {
    struct E { int a, b; };
    std::vector<E> edges;
    std::unordered_set<long long> seen;
    const auto key = [&](int a, int b) { if (a > b) std::swap(a, b); return (long long)a * n + b; };
    edges.reserve((size_t)n * k);
    seen.reserve((size_t)n * k * 2);
    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j) {
            const int b = (i + j) % n;
            if (b != i && seen.insert(key(i, b)).second) edges.push_back({ i, b });
        }
    Stream g(seed, (uint64_t)proj, 0x535749ULL);
    for (E& e : edges) {
        if (g.uni() >= rewire) continue;
        for (int attempt = 0; attempt < 32; ++attempt) {
            const int cand = g.below(n);
            if (cand == e.a || seen.count(key(e.a, cand))) continue;
            seen.erase(key(e.a, e.b));
            seen.insert(key(e.a, cand));
            e.b = cand;
            break;
        }
    }
    std::vector<long long> count((size_t)n + 1, 0);
    for (const E& e : edges) { ++count[(size_t)e.b + 1]; if (!p.directed) ++count[(size_t)e.a + 1]; }
    for (int i = 0; i < n; ++i) count[(size_t)i + 1] += count[(size_t)i];
    r.start = count;
    r.src.resize((size_t)count[(size_t)n]);
    for (const E& e : edges) {
        r.src[(size_t)count[(size_t)e.b]++] = e.a;
        if (!p.directed) r.src[(size_t)count[(size_t)e.a]++] = e.b;
    }
    for (int i = 0; i < n; ++i)
        std::sort(r.src.begin() + (std::ptrdiff_t)r.start[(size_t)i], r.src.begin() + (std::ptrdiff_t)r.start[(size_t)i + 1]);
    r.expected = (double)r.src.size() / (double)n;
}

Rows build_rows(const NeuralProjection& p, int proj, uint64_t seed, int n_from, int n_to) {
    Rows r;
    const bool same = p.from == p.to;
    const bool regular = p.rule != NeuralRule::Probability && p.rule != NeuralRule::FixedDegree;
    if (regular && !same) throw std::runtime_error("regular topologies need a projection inside one population");

    double prob = 0.0;
    int d = 0, k = 1, w = 1, h = 1;
    double rewire = 0.0;
    if (p.rule == NeuralRule::Probability) {
        prob = num_or(p.p_text, 0.0);
        if (!(prob >= 0.0 && prob <= 1.0)) throw std::runtime_error("probability must be in [0, 1]");
        r.expected = prob * (double)(n_from - (same ? 1 : 0));
    } else if (p.rule == NeuralRule::FixedDegree) {
        d = int_or(p.degree_text, 0);
        if (d < 0 || d > n_from - (same ? 1 : 0)) throw std::runtime_error("the degree does not fit the source population");
        r.expected = d;
    } else if (p.rule == NeuralRule::Ring || p.rule == NeuralRule::SmallWorld) {
        k = int_or(p.k_text, 1);
        if (k < 1 || 2 * k >= n_to) throw std::runtime_error("neighbours each side must be between 1 and (size - 1) / 2");
    } else if (p.rule == NeuralRule::Grid || p.rule == NeuralRule::King) {
        w = int_or(p.width_text, 1);
        if (w < 1 || n_to % w != 0) throw std::runtime_error("the population size must divide by the lattice width");
        h = n_to / w;
    }
    if (p.rule == NeuralRule::SmallWorld) {
        rewire = num_or(p.rewire_text, 0.0);
        if (!(rewire >= 0.0 && rewire <= 1.0)) throw std::runtime_error("rewire probability must be in [0, 1]");
        rows_small_world(p, seed, proj, n_to, k, rewire, r);
        return r;
    }
    if ((double)n_to * r.expected > (double)kMaxArcs)
        throw std::runtime_error("a projection would hold more than " + std::to_string(kMaxArcs) + " arcs");

    r.start.assign((size_t)n_to + 1, 0);
    const size_t guess = (size_t)((double)n_to * (r.expected > 0.0 ? r.expected : 8.0) * 1.05);
    r.src.reserve(guess);
    std::vector<int> row, tmp;
    for (int i = 0; i < n_to; ++i) {
        row.clear();
        Stream g(seed, (uint64_t)proj, (uint64_t)i);
        const int self = same ? i : -1;
        if (p.rule == NeuralRule::Probability) row_probability(g, n_from, self, prob, row);
        else if (p.rule == NeuralRule::FixedDegree) row_fixed(g, n_from, self, d, row, tmp);
        else {
            row_regular(p, n_to, w, h, k, i, row);
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            row.erase(std::remove(row.begin(), row.end(), i), row.end());
        }
        r.src.insert(r.src.end(), row.begin(), row.end());
        r.start[(size_t)i + 1] = (long long)r.src.size();
        if ((long long)r.src.size() > kMaxArcs)
            throw std::runtime_error("a projection would hold more than " + std::to_string(kMaxArcs) + " arcs");
    }
    if (regular) r.expected = (double)r.src.size() / (double)n_to;
    return r;
}

// FNV-1a по словам: хеш графа в канонической нумерации.
struct Fnv {
    uint64_t h = 0xCBF29CE484222325ULL;
    void add(uint64_t v) { h = (h ^ v) * 0x100000001B3ULL; }
};
std::string hex64(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return b;
}
}  // namespace

const char* neural_rule_name(NeuralRule r) {
    switch (r) {
    case NeuralRule::Probability: return "Probability";
    case NeuralRule::FixedDegree: return "Fixed in-degree";
    case NeuralRule::Chain:       return "Chain";
    case NeuralRule::Ring:        return "Ring";
    case NeuralRule::Grid:        return "Grid";
    case NeuralRule::King:        return "King graph";
    case NeuralRule::SmallWorld:  return "Small world";
    }
    return "?";
}
const char* neural_schedule_name(NeuralSchedule s) {
    switch (s) {
    case NeuralSchedule::Stage: return "Stage (RK4, 4 exchanges)";
    case NeuralSchedule::Heun:  return "Heun (2 exchanges)";
    case NeuralSchedule::Post:  return "Post (1 exchange)";
    case NeuralSchedule::Old:   return "Old state (Network)";
    }
    return "?";
}
const char* neural_sweep_name(NeuralSweepKind k) {
    switch (k) {
    case NeuralSweepKind::Coupling:        return "Coupling K";
    case NeuralSweepKind::Parameter:       return "Parameter";
    case NeuralSweepKind::ParameterSpread: return "Parameter spread";
    case NeuralSweepKind::Replica:         return "Replicas (seed)";
    }
    return "?";
}

NeuralGraph neural_build_graph(const NeuralConfig& c) {
    NeuralGraph g;
    const int nPops = (int)c.populations.size();
    if (nPops == 0) throw std::runtime_error("the network has no populations");
    g.pop_start.assign((size_t)nPops + 1, 0);
    long long total = 0;
    for (int p = 0; p < nPops; ++p) {
        const int n = int_or(c.populations[(size_t)p].size_text, 0);
        if (n < 1) throw std::runtime_error("a population must have at least one node");
        total += n;
        if (total > 2000000000LL) throw std::runtime_error("too many nodes");
        g.pop_start[(size_t)p + 1] = (int)total;
    }
    g.N = (int)total;
    const double K = num_or(c.coupling_text, 0.0);
    const uint64_t seed = (uint64_t)int_or(c.seed_text, 1);

    std::vector<Rows> rows(c.projections.size());
    for (size_t q = 0; q < c.projections.size(); ++q) {
        const NeuralProjection& p = c.projections[q];
        if (p.from < 0 || p.from >= nPops || p.to < 0 || p.to >= nPops) throw std::runtime_error("a projection points to a missing population");
        if (p.law < 0 || p.law >= (int)c.laws.size()) throw std::runtime_error("a projection uses a missing coupling law");
        rows[q] = build_rows(p, (int)q, seed, g.pop_start[(size_t)p.from + 1] - g.pop_start[(size_t)p.from],
                             g.pop_start[(size_t)p.to + 1] - g.pop_start[(size_t)p.to]);
    }
    long long arcs = 0;
    for (const Rows& r : rows) arcs += (long long)r.src.size();
    if (arcs > kMaxArcs) throw std::runtime_error("the network would hold more than " + std::to_string(kMaxArcs) + " arcs");

    std::vector<double> expected_in((size_t)nPops, 0.0);
    for (size_t q = 0; q < c.projections.size(); ++q) expected_in[(size_t)c.projections[q].to] += rows[q].expected;

    g.start.assign((size_t)g.N + 1, 0);
    g.src.reserve((size_t)arcs); g.w.reserve((size_t)arcs); g.law.reserve((size_t)arcs);
    struct Arc { int src, law; double w; };
    std::vector<Arc> row;
    g.degree_min = std::numeric_limits<int>::max(); g.degree_max = 0;
    for (int pop = 0; pop < nPops; ++pop) {
        const int n = g.pop_start[(size_t)pop + 1] - g.pop_start[(size_t)pop];
        for (int t = 0; t < n; ++t) {
            row.clear();
            long long degree = 0;
            for (size_t q = 0; q < c.projections.size(); ++q)
                if (c.projections[q].to == pop) degree += rows[q].start[(size_t)t + 1] - rows[q].start[(size_t)t];
            for (size_t q = 0; q < c.projections.size(); ++q) {
                const NeuralProjection& p = c.projections[q];
                if (p.to != pop) continue;
                const Rows& r = rows[q];
                const long long a0 = r.start[(size_t)t], a1 = r.start[(size_t)t + 1];
                if (a0 == a1) continue;
                double div = 1.0;
                if (p.norm == NeuralNorm::Projection) div = p.norm_expected ? r.expected : (double)(a1 - a0);
                else if (p.norm == NeuralNorm::Total) div = p.norm_expected ? expected_in[(size_t)pop] : (double)degree;
                if (!(div > 0.0)) div = 1.0;
                const double base = K * num_or(p.weight_text, 1.0) / div;
                const double spread = num_or(p.weight_spread_text, 0.0);
                Stream gw(seed ^ 0x57454947ULL, (uint64_t)q, (uint64_t)t);
                const int off = g.pop_start[(size_t)p.from];
                for (long long a = a0; a < a1; ++a) {
                    const double f = spread != 0.0 ? 1.0 + spread * (2.0 * gw.uni() - 1.0) : 1.0;
                    row.push_back({ r.src[(size_t)a] + off, p.law, base * f });
                }
            }
            std::stable_sort(row.begin(), row.end(), [](const Arc& a, const Arc& b) { return a.src < b.src; });
            const int node = g.pop_start[(size_t)pop] + t;
            g.start[(size_t)node] = (int)g.src.size();
            for (const Arc& a : row) { g.src.push_back(a.src); g.law.push_back(a.law); g.w.push_back(a.w); }
            g.degree_min = std::min(g.degree_min, (int)row.size());
            g.degree_max = std::max(g.degree_max, (int)row.size());
        }
    }
    g.start[(size_t)g.N] = (int)g.src.size();

    Fnv f;
    f.add((uint64_t)g.N);
    for (int v : g.pop_start) f.add((uint64_t)(uint32_t)v);
    for (int v : g.start) f.add((uint64_t)(uint32_t)v);
    for (size_t e = 0; e < g.src.size(); ++e) {
        uint64_t bits;
        std::memcpy(&bits, &g.w[e], sizeof bits);
        f.add(((uint64_t)(uint32_t)g.src[e] << 8) | (uint64_t)(uint8_t)g.law[e]);
        f.add(bits);
    }
    g.hash = hex64(f.h);
    return g;
}

std::string neural_graph_recipe(const NeuralConfig& c) {
    std::ostringstream o;
    o << "v" << kNeuralGraphVersion << "|K=" << c.coupling_text << "|seed=" << c.seed_text << "|laws=" << c.laws.size();
    for (const NeuralPopulation& p : c.populations) o << "|pop:" << p.size_text;
    for (const NeuralProjection& p : c.projections)
        o << "|proj:" << p.from << ">" << p.to << ":" << (int)p.rule << ":" << p.p_text << ":" << p.degree_text << ":" << p.k_text
          << ":" << p.width_text << ":" << p.rewire_text << ":" << p.periodic << p.directed << ":" << p.weight_text << ":"
          << p.weight_spread_text << ":" << (int)p.norm << p.norm_expected << ":" << p.law;
    if (c.sweep.enabled)
        o << "|sweep:" << (int)c.sweep.kind << ":" << c.sweep.population << ":" << c.sweep.param << ":" << c.sweep.lo_text
          << ":" << c.sweep.hi_text << ":" << c.sweep.points_text << ":" << c.sweep.log;
    return o.str();
}

std::string neural_model_text(const System& sys) {
    std::string out;
    for (const auto& r : sys.rhs) out += r + ";";
    return out;
}

NeuralPopulation neural_default_population(const SystemRecord* r, const std::vector<std::string>& vars,
                                           const std::vector<std::string>& params) {
    NeuralPopulation p;
    for (const auto& name : params) {
        std::string v;
        if (r) { auto it = r->param_values.find(name); if (it != r->param_values.end()) v = it->second; }
        p.param_values[name] = v;
    }
    for (const auto& name : vars) {
        std::string v;
        if (r) { auto it = r->init_conditions.find(name); if (it != r->init_conditions.end()) v = it->second; }
        p.initial_conditions[name] = v;
        p.ic_spread[name] = "0.1";   // из одинаковых НУ диффузионная сеть стартует синхронной
    }
    return p;
}

NeuralRequest neural_build_request(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                                   const std::vector<std::string>& vars, const std::vector<std::string>& params,
                                   const NeuralConfig& c, const NeuralGraph& g) {
    if (sys.dae_nx >= 0) throw std::runtime_error("DAE systems are not supported on this tab yet");
    if (sys.is_map && c.schedule == NeuralSchedule::Stage) throw std::runtime_error("a map has no stages: pick another schedule");
    NeuralRequest r;
    r.schedule = c.schedule;
    const bool stage = c.schedule == NeuralSchedule::Stage;
    r.krs_body = compute_krs_for_scheme(custom_schemes, sys, stage ? std::string("RK4") : c.scheme);
    if (r.krs_body.empty()) throw std::runtime_error(krs_empty_error(custom_schemes, sys, stage ? std::string("RK4") : c.scheme));
    if (stage) r.rhs_body = codegen_rhs_body(sys);
    r.coupling_body = net_coupling_body(sys, c.laws);
    r.n_laws = (int)c.laws.size();
    r.amountOfX = (int)vars.size();
    r.amountOfValues = (int)params.size() + 1 + kSchemeTailSlots;
    r.n_nodes = g.N;
    r.edge_start = g.start; r.edge_src = g.src; r.edge_w = g.w;
    if (r.n_laws > 1) r.edge_law = g.law;

    const uint64_t seed = (uint64_t)int_or(c.seed_text, 1);
    r.values.assign((size_t)g.N * (size_t)r.amountOfValues, 0.0);
    r.initial_conditions.assign((size_t)g.N * (size_t)r.amountOfX, 0.0);
    const double s_default = num_or(c.symmetry_s, 0.5);
    for (size_t pop = 0; pop < c.populations.size(); ++pop) {
        const NeuralPopulation& p = c.populations[pop];
        std::vector<double> value(params.size()), spread(params.size()), x0(vars.size()), dx(vars.size());
        const auto get = [](const std::map<std::string, std::string>& m, const std::string& k) {
            auto it = m.find(k);
            return it == m.end() ? std::string() : it->second;
        };
        for (size_t j = 0; j < params.size(); ++j) { value[j] = num_or(get(p.param_values, params[j]), 0.0); spread[j] = num_or(get(p.param_spread, params[j]), 0.0); }
        for (size_t k = 0; k < vars.size(); ++k) { x0[k] = num_or(get(p.initial_conditions, vars[k]), 0.0); dx[k] = num_or(get(p.ic_spread, vars[k]), 0.0); }
        for (int i = g.pop_start[pop]; i < g.pop_start[pop + 1]; ++i) {
            double* a = r.values.data() + (size_t)i * r.amountOfValues;
            a[0] = s_default;
            scheme_fill_tail(a + params.size() + 1, c.cd_fp_iters, c.newton);
            Stream gp(seed ^ 0x50415241ULL, (uint64_t)pop, (uint64_t)(i - g.pop_start[pop]));
            for (size_t j = 0; j < params.size(); ++j)
                a[j + 1] = value[j] + (spread[j] != 0.0 ? spread[j] * (2.0 * gp.uni() - 1.0) : 0.0);
            Stream gi(seed ^ 0x49435331ULL, (uint64_t)pop, (uint64_t)(i - g.pop_start[pop]));
            for (size_t k = 0; k < vars.size(); ++k)
                r.initial_conditions[(size_t)i * r.amountOfX + k] = x0[k] + (dx[k] != 0.0 ? dx[k] * (2.0 * gi.uni() - 1.0) : 0.0);
        }
    }
    r.h = parse_num(c.h_text, 0.01);
    r.t_max = parse_num(c.t_max_text, 100.0);
    r.transient = parse_num(c.transient_text, 0.0);
    r.pre_scaller = std::max(1, int_or(c.pre_scaller_text, 1));
    r.max_value = parse_num(c.max_value_text, 1.0e6);

    for (size_t k = 0; k < vars.size(); ++k) r.observe.vars.push_back((int)k);
    r.observe.pop_start = g.pop_start;
    r.observe.spread = true;
    r.observe.order_var = c.order_var < (int)vars.size() ? c.order_var : -1;
    r.observe.order_threshold = num_or(c.order_threshold_text, 0.0);
    const int per_pop = std::max(0, int_or(c.observed_per_pop_text, 8));
    for (size_t pop = 0; pop + 1 < g.pop_start.size(); ++pop) {
        const int n = g.pop_start[pop + 1] - g.pop_start[pop], m = std::min(per_pop, n);
        for (int j = 0; j < m; ++j) r.observe.nodes.push_back(g.pop_start[pop] + (int)((long long)j * n / m));
    }
    return r;
}

std::vector<double> neural_sweep_values(const NeuralConfig& c) {
    std::vector<double> out;
    const NeuralSweep& s = c.sweep;
    if (!s.enabled) return out;
    const int n = int_or(s.points_text, 0);
    if (n < 1) throw std::runtime_error("a sweep needs at least one point");
    out.resize((size_t)n);
    if (s.kind == NeuralSweepKind::Replica) {
        const int seed = int_or(c.seed_text, 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = (double)seed + i;
        return out;
    }
    const double lo = num_or(s.lo_text, 0.0), hi = num_or(s.hi_text, 0.0);
    if (!std::isfinite(lo) || !std::isfinite(hi)) throw std::runtime_error("the sweep range is not a number");
    if (s.log && !(lo > 0.0 && hi > 0.0)) throw std::runtime_error("a log sweep needs a positive range");
    for (int i = 0; i < n; ++i) {
        const double f = n > 1 ? (double)i / (double)(n - 1) : 0.0;
        out[(size_t)i] = s.log ? std::exp(std::log(lo) + f * (std::log(hi) - std::log(lo))) : lo + f * (hi - lo);
    }
    if (n > 1) out.back() = hi;
    return out;
}

NeuralConfig neural_sweep_instance(const NeuralConfig& c, int index, double value) {
    NeuralConfig o = c;
    o.sweep.enabled = false;
    const NeuralSweep& s = c.sweep;
    const std::string text = fmt_num_shortest(value);
    if (s.kind == NeuralSweepKind::Coupling) o.coupling_text = text;
    else if (s.kind == NeuralSweepKind::Replica) o.seed_text = std::to_string(int_or(c.seed_text, 1) + index);
    else {
        if (s.param.empty()) throw std::runtime_error("the sweep has no parameter");
        if (s.population >= (int)o.populations.size()) throw std::runtime_error("the sweep points to a missing population");
        for (int p = 0; p < (int)o.populations.size(); ++p) {
            if (s.population >= 0 && s.population != p) continue;
            auto& m = s.kind == NeuralSweepKind::Parameter ? o.populations[(size_t)p].param_values : o.populations[(size_t)p].param_spread;
            m[s.param] = text;
        }
    }
    return o;
}

NeuralPlan neural_plan(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                       const std::vector<std::string>& vars, const std::vector<std::string>& params,
                       const NeuralConfig& c) {
    NeuralPlan plan;
    plan.pops_per_instance = (int)c.populations.size();
    plan.sweep_values = neural_sweep_values(c);
    if (plan.sweep_values.empty()) {
        const NeuralGraph g = neural_build_graph(c);
        plan.request = neural_build_request(sys, custom_schemes, vars, params, c, g);
        plan.hash = g.hash;
        plan.arcs = (long long)g.src.size();
        plan.degree_min = g.degree_min; plan.degree_max = g.degree_max;
        return plan;
    }
    // Экземпляры подряд: номера узлов и популяций сдвигаются, дуга наружу экземпляра появиться не может.
    NeuralRequest& a = plan.request;
    Fnv f;
    long long nodes = 0;
    plan.degree_min = std::numeric_limits<int>::max();
    for (int i = 0; i < (int)plan.sweep_values.size(); ++i) {
        const NeuralConfig one = neural_sweep_instance(c, i, plan.sweep_values[(size_t)i]);
        const NeuralGraph g = neural_build_graph(one);
        NeuralRequest r = neural_build_request(sys, custom_schemes, vars, params, one, g);
        if (nodes + r.n_nodes > 2000000000LL) throw std::runtime_error("the sweep holds too many nodes");
        plan.arcs += (long long)r.edge_src.size();
        if (plan.arcs > kMaxArcs) throw std::runtime_error("the sweep would hold more than " + std::to_string(kMaxArcs) + " arcs");
        const int off = (int)nodes, arc_off = (int)a.edge_src.size();
        for (const char ch : g.hash) f.add((uint64_t)(uint8_t)ch);
        plan.degree_min = (std::min)(plan.degree_min, g.degree_min);
        plan.degree_max = (std::max)(plan.degree_max, g.degree_max);
        if (i == 0) {
            a = std::move(r);
            a.instance_start = { 0, a.n_nodes };
            nodes = a.n_nodes;
            continue;
        }
        a.values.insert(a.values.end(), r.values.begin(), r.values.end());
        a.initial_conditions.insert(a.initial_conditions.end(), r.initial_conditions.begin(), r.initial_conditions.end());
        a.edge_start.pop_back();
        for (int v : r.edge_start) a.edge_start.push_back(v + arc_off);
        for (int v : r.edge_src) a.edge_src.push_back(v + off);
        a.edge_w.insert(a.edge_w.end(), r.edge_w.begin(), r.edge_w.end());
        a.edge_law.insert(a.edge_law.end(), r.edge_law.begin(), r.edge_law.end());
        a.observe.pop_start.pop_back();
        for (int v : r.observe.pop_start) a.observe.pop_start.push_back(v + off);
        for (int v : r.observe.nodes) a.observe.nodes.push_back(v + off);
        nodes += r.n_nodes;
        a.n_nodes = (int)nodes;
        a.instance_start.push_back(a.n_nodes);
    }
    plan.hash = hex64(f.h);
    return plan;
}

void NeuralSession::load_from_record(const SystemRecord& r, const std::vector<std::string>& vars_,
                                     const std::vector<std::string>& params_) {
    vars = vars_; params = params_;
    enabled_builtin_schemes = enabled_builtins_from_record(r);
    NeuralConfig c;
    c.label = "Experiment 1";
    c.h_text = default_h_from_record(r);
    c.scheme = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    c.cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    c.newton = newton_from_record(r);
    c.laws.push_back(net_default_law(vars));
    c.populations.push_back(neural_default_population(&r, vars, params));
    c.populations[0].name = "A";
    NeuralProjection p;
    c.projections.push_back(p);
    configs.clear();
    configs.push_back(std::move(c));
    active_config_index = 0;
    running_config_index = -1;
}

void NeuralSession::add_config() {
    NeuralConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = NeuralResult{};
        c.instance_start.clear(); c.sweep_values.clear(); c.sweep_order.clear(); c.sweep_spread.clear();
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.graph_hash.clear(); c.graph_recipe.clear();
    } else {
        c.laws.push_back(net_default_law(vars));
        c.populations.push_back(neural_default_population(nullptr, vars, params));
    }
    c.label = "Experiment " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}

void NeuralSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size() || configs.size() == 1) return;
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (running_config_index > i) --running_config_index;
    if (active_config_index >= (int)configs.size()) active_config_index = (int)configs.size() - 1;
}

bool NeuralSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight || config_idx < 0 || config_idx >= (int)configs.size()) return false;
    NeuralConfig& c = configs[(size_t)config_idx];
    c.last_error.clear();
    c.hash_mismatch = false;
    last_run_label.clear();

    NeuralConfig snap = c;
    snap.result = NeuralResult{};
    cancel_token = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);
    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();

    // Граф на 10 млн дуг строится секунды: всё это в рабочем потоке, а не в кадре.
    run_future = std::async(std::launch::async,
        [&engine, snap = std::move(snap), sys_copy = sys, cs = custom_schemes, v = vars, q = params,
         cancel = cancel_token, progress = progress_token]() {
        NeuralOutcome out;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            NeuralPlan plan = neural_plan(sys_copy, cs, v, q, snap);
            out.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            out.hash = plan.hash;
            out.recipe = neural_graph_recipe(snap);
            out.arcs = plan.arcs;
            out.degree_min = plan.degree_min; out.degree_max = plan.degree_max;
            if (!snap.graph_hash.empty() && snap.graph_recipe == out.recipe && snap.graph_hash != plan.hash) {
                out.hash_mismatch = true;
                out.result.error = "this recipe no longer gives the saved graph (hash " + snap.graph_hash + ", now " + plan.hash
                                 + "): the generator or its library changed";
                return out;
            }
            plan.request.cancel = cancel; plan.request.progress = progress;
            out.result = engine.run_neural(plan.request);
            out.instance_start = plan.request.instance_start;
            if (out.instance_start.empty()) out.instance_start = { 0, plan.request.n_nodes };
            out.pops_per_instance = plan.pops_per_instance;
            out.sweep_values = std::move(plan.sweep_values);
            neural_sweep_summary(out, plan.request.observe.order_var);
        } catch (const std::exception& e) {
            out.result.error = e.what();
        }
        return out;
    });
    return true;
}

void NeuralSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}

bool NeuralSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false; running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    const int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("network");
    NeuralOutcome out = run_future.get();
    const bool cancelled = out.result.cancelled;
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        NeuralConfig& c = configs[(size_t)idx];
        c.last_run_ok = out.result.ok;
        c.last_error = out.result.ok ? std::string() : out.result.error;
        c.hash_mismatch = out.hash_mismatch;
        if (!out.hash.empty() && !out.hash_mismatch) {
            c.graph_hash = out.hash; c.graph_recipe = out.recipe;
            c.model_text = neural_model_text(sys);
            c.arcs = out.arcs; c.degree_min = out.degree_min; c.degree_max = out.degree_max;
            c.build_seconds = out.build_seconds;
        }
        if (out.result.ok) {
            c.result = std::move(out.result);
            c.instance_start = std::move(out.instance_start);
            c.sweep_values = std::move(out.sweep_values);
            c.sweep_order = std::move(out.sweep_order);
            c.sweep_spread = std::move(out.sweep_spread);
            c.pops_per_instance = out.pops_per_instance;
            ++c.data_generation;
        }
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_neural_run(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
