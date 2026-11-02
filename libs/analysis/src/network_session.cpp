#include "ucuda/analysis/network_session.h"
#include "ucuda/domain/num_parse.h"
#include "ucuda/equations/codegen.hpp"   // codegen_coupling
#include "ucuda/equations/sysparse.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <sstream>

namespace {

double parse_d(const std::string& s, double def) { return parse_num(s, def); }
int    parse_i(const std::string& s, int def)    { return parse_num_int(s, def); }

void log_run_completed(const char* label, bool ok, double secs) {
    std::printf("[run] %s %s in %.2fs\n",
                ok ? "Done" : "Cancelled",
                (label && *label) ? label : "(unnamed)",
                secs);
    std::fflush(stdout);
}

// splitmix64 — тот же генератор, что в шаблонах ядер. Нужен детерминизм по
// (seed, i, j): и топология, и разброс НУ обязаны воспроизводиться от запуска
// к запуску, иначе картинку невозможно ни повторить, ни сравнить.
unsigned long long splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Равномерное [0,1) из тройки (seed, a, b).
double rand01(unsigned long long seed, unsigned long long a, unsigned long long b) {
    const unsigned long long h = splitmix(splitmix(seed ^ (a * 0x100000001B3ULL)) ^ b);
    return (double)(h >> 11) / (double)(1ULL << 53);
}

// Ключ неориентированного ребра: порядок концов не важен, петли и дубликаты
// генераторы не выдают.
long long edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (long long)a * (long long)kMaxNetworkNodes + (long long)b;
}

void place_circle(std::vector<NetNode>& nodes, int first, int count, float r) {
    for (int i = 0; i < count; ++i) {
        const double t = 2.0 * 3.14159265358979323846 * (double)i / (double)std::max(count, 1);
        nodes[(size_t)(first + i)].ui_x = 0.5f + r * (float)std::cos(t);
        nodes[(size_t)(first + i)].ui_y = 0.5f + r * (float)std::sin(t);
    }
}

} // namespace

const char* net_topology_name(NetTopology t) {
    switch (t) {
    case NetTopology::Custom:     return "Custom";
    case NetTopology::Chain:      return "Chain";
    case NetTopology::Ring:       return "Ring";
    case NetTopology::Star:       return "Star";
    case NetTopology::Grid:       return "Grid";
    case NetTopology::King:       return "King graph";
    case NetTopology::AllToAll:   return "All-to-all";
    case NetTopology::SmallWorld: return "Small world";
    }
    return "Custom";
}

NetCouplingLaw net_default_law(const std::vector<std::string>& vars) {
    NetCouplingLaw law;
    law.name = "diffusive";
    law.expr.assign(vars.size(), std::string());
    if (!vars.empty())
        law.expr[0] = "K*(" + vars[0] + "_j - " + vars[0] + "_i)";
    return law;
}

std::string net_coupling_body(const System& sys, const std::vector<NetCouplingLaw>& laws) {
    std::string out;
    for (size_t i = 0; i < laws.size(); ++i) {
        const std::vector<std::string>& e = laws[i].expr;
        // Закон, у которого все выражения пусты, всё равно получает свою
        // ветку: номера законов — это индексы, и пропуск сдвинул бы рёбра на
        // чужой закон.
        std::vector<std::string> padded = e;
        padded.resize(sys.vars.size());
        out += "        case " + std::to_string(i) + ": {\n";
        out += codegen_coupling(sys, padded);
        out += "        } break;\n";
    }
    return out;
}

void net_build_csr(const NetworkConfig& c, double coupling,
                   std::vector<int>& edge_start, std::vector<int>& edge_src,
                   std::vector<double>& edge_w, std::vector<int>& edge_law)
{
    const int n = (int)c.nodes.size();
    edge_start.assign((size_t)n + 1, 0);
    edge_src.clear(); edge_w.clear(); edge_law.clear();
    if (n == 0) return;

    // Дуга — это (приёмник, источник). Двунаправленное ребро даёт две.
    struct Arc { int to, from, law; double w; };
    std::vector<Arc> arcs;
    arcs.reserve(c.edges.size() * 2);
    for (const NetEdge& e : c.edges) {
        if (e.from < 0 || e.from >= n || e.to < 0 || e.to >= n) continue;
        if (e.from == e.to) continue;   // петля: связь узла с самим собой тождественно ноль
        const double w = coupling * parse_d(e.weight_text, 1.0);
        const int law = (e.law >= 0 && e.law < (int)c.laws.size()) ? e.law : 0;
        arcs.push_back({ e.to, e.from, law, w });
        if (e.bidirectional) arcs.push_back({ e.from, e.to, law, w });
    }

    std::vector<int> count((size_t)n, 0);
    for (const Arc& a : arcs) ++count[(size_t)a.to];
    for (int i = 0; i < n; ++i) edge_start[(size_t)i + 1] = edge_start[(size_t)i] + count[(size_t)i];

    // Нормировка на степень ПРИЁМНИКА (см. NetworkConfig::normalize_coupling).
    if (c.normalize_coupling)
        for (Arc& a : arcs)
            if (count[(size_t)a.to] > 0) a.w /= (double)count[(size_t)a.to];

    const size_t total = arcs.size();
    edge_src.assign(total, 0);
    edge_w.assign(total, 0.0);
    edge_law.assign(total, 0);
    std::vector<int> cursor(edge_start.begin(), edge_start.end() - 1);
    for (const Arc& a : arcs) {
        const int pos = cursor[(size_t)a.to]++;
        edge_src[(size_t)pos] = a.from;
        edge_w[(size_t)pos]   = a.w;
        edge_law[(size_t)pos] = a.law;
    }
}

std::string net_generate_topology(NetworkConfig& c, const std::vector<std::string>& vars) {
    (void)vars;
    if (c.topology == NetTopology::Custom) return {};

    const bool is_grid = (c.topology == NetTopology::Grid || c.topology == NetTopology::King);
    int gw = std::max(1, parse_i(c.gen_w_text, 4));
    int gh = std::max(1, parse_i(c.gen_h_text, 4));
    int n  = is_grid ? gw * gh : std::max(1, parse_i(c.gen_n_text, 16));
    if (n > kMaxNetworkNodes)
        return "too many nodes: the cap is " + std::to_string(kMaxNetworkNodes);

    const int k = std::max(1, parse_i(c.gen_k_text, 1));
    const double p = std::min(1.0, std::max(0.0, parse_d(c.gen_p_text, 0.1)));
    const unsigned long long seed = (unsigned long long)std::max(0, parse_i(c.gen_seed_text, 1));

    // Переопределения узлов переживают смену топологии: расстройка ансамбля —
    // это ручная работа, терять её при каждом перегенерировании нельзя.
    std::vector<NetNode> old = std::move(c.nodes);
    c.nodes.assign((size_t)n, NetNode{});
    for (int i = 0; i < n; ++i) {
        if (i < (int)old.size()) {
            c.nodes[(size_t)i].param_values       = std::move(old[(size_t)i].param_values);
            c.nodes[(size_t)i].initial_conditions = std::move(old[(size_t)i].initial_conditions);
        }
        // Подпись = индекс: рёбра адресуют узлы индексами, и нумерация
        // с единицы заставляла бы пересчитывать в уме при каждой правке.
        c.nodes[(size_t)i].label = std::to_string(i);
    }

    // Закон связи переживает перегенерацию, если он был ОДИН на всю сеть:
    // иначе выбранный «закон на все рёбра» сбрасывался бы каждым изменением
    // числа узлов.
    int common_law = c.edges.empty() ? 0 : c.edges.front().law;
    for (const NetEdge& e : c.edges)
        if (e.law != common_law) { common_law = 0; break; }
    if (common_law < 0 || common_law >= (int)c.laws.size()) common_law = 0;

    std::set<long long> seen;
    std::vector<NetEdge> edges;
    auto add = [&](int a, int b) {
        if (a == b || a < 0 || b < 0 || a >= n || b >= n) return;
        if (!c.gen_directed && !seen.insert(edge_key(a, b)).second) return;
        NetEdge e;
        e.from = a; e.to = b;
        e.bidirectional = !c.gen_directed;
        e.weight_text = "1";
        e.law = common_law;
        edges.push_back(e);
    };

    switch (c.topology) {
    case NetTopology::Chain:
        for (int i = 0; i + 1 < n; ++i) add(i, i + 1);
        for (int i = 0; i < n; ++i) {
            c.nodes[(size_t)i].ui_x = (n == 1) ? 0.5f : 0.06f + 0.88f * (float)i / (float)(n - 1);
            c.nodes[(size_t)i].ui_y = 0.5f;
        }
        break;

    case NetTopology::Ring:
        for (int i = 0; i < n; ++i)
            for (int j = 1; j <= k; ++j) add(i, (i + j) % n);
        place_circle(c.nodes, 0, n, 0.42f);
        break;

    case NetTopology::Star:
        for (int i = 1; i < n; ++i) add(0, i);
        place_circle(c.nodes, 1, n - 1, 0.42f);
        c.nodes[0].ui_x = 0.5f; c.nodes[0].ui_y = 0.5f;
        break;

    case NetTopology::AllToAll:
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) add(i, j);
        place_circle(c.nodes, 0, n, 0.42f);
        break;

    case NetTopology::Grid:
    case NetTopology::King: {
        const bool king = (c.topology == NetTopology::King);
        auto idx = [&](int x, int y) { return y * gw + x; };
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                const int a = idx(x, y);
                // Только «вправо» и «вниз»: обратные направления даст сам
                // обход, а с ними пришли бы дубликаты.
                if (x + 1 < gw)          add(a, idx(x + 1, y));
                else if (c.gen_periodic && gw > 2) add(a, idx(0, y));
                if (y + 1 < gh)          add(a, idx(x, y + 1));
                else if (c.gen_periodic && gh > 2) add(a, idx(x, 0));
                if (king) {
                    const int xr = (x + 1 < gw) ? x + 1 : (c.gen_periodic && gw > 2 ? 0 : -1);
                    const int yd = (y + 1 < gh) ? y + 1 : (c.gen_periodic && gh > 2 ? 0 : -1);
                    const int xl = (x > 0) ? x - 1 : (c.gen_periodic && gw > 2 ? gw - 1 : -1);
                    if (xr >= 0 && yd >= 0) add(a, idx(xr, yd));
                    if (xl >= 0 && yd >= 0) add(a, idx(xl, yd));
                }
            }
        }
        for (int y = 0; y < gh; ++y)
            for (int x = 0; x < gw; ++x) {
                NetNode& nd = c.nodes[(size_t)idx(x, y)];
                nd.ui_x = (gw == 1) ? 0.5f : 0.08f + 0.84f * (float)x / (float)(gw - 1);
                nd.ui_y = (gh == 1) ? 0.5f : 0.08f + 0.84f * (float)y / (float)(gh - 1);
            }
        break;
    }

    case NetTopology::SmallWorld: {
        // Уоттс-Строгац: кольцо с k соседями, затем каждое ребро с
        // вероятностью p переподключается случайным концом. Перепривязка
        // НЕ создаёт петель и дубликатов — иначе средняя степень уезжала бы
        // вниз вместе с p, и «маленький мир» мерился бы не тем.
        for (int i = 0; i < n; ++i)
            for (int j = 1; j <= k; ++j) add(i, (i + j) % n);
        for (size_t e = 0; e < edges.size(); ++e) {
            if (rand01(seed, e, 0) >= p) continue;
            for (int attempt = 0; attempt < 32; ++attempt) {
                const int cand = (int)(rand01(seed, e, (unsigned long long)attempt + 1) * (double)n) % n;
                const int a = edges[e].from;
                if (cand == a) continue;
                const long long old_key = edge_key(edges[e].from, edges[e].to);
                const long long new_key = edge_key(a, cand);
                if (seen.count(new_key)) continue;
                seen.erase(old_key);
                seen.insert(new_key);
                edges[e].to = cand;
                break;
            }
        }
        place_circle(c.nodes, 0, n, 0.42f);
        break;
    }

    case NetTopology::Custom:
        break;
    }

    c.edges = std::move(edges);
    return {};
}

namespace {

// Значение параметра/НУ узла: своё, если задано непустой строкой, иначе общее.
std::string node_value(const std::map<std::string, std::string>& own,
                       const std::map<std::string, std::string>& base,
                       const std::string& key)
{
    auto it = own.find(key);
    if (it != own.end() && !it->second.empty()) return it->second;
    auto ib = base.find(key);
    return (ib != base.end()) ? ib->second : std::string();
}

NetworkRequest build_network_request(const NetworkSession& s, const NetworkConfig& c) {
    NetworkRequest req;
    req.krs_body      = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
    req.coupling_body = net_coupling_body(s.sys, c.laws);   // бросает при ошибке разбора
    req.amountOfX     = (int)s.vars.size();
    req.n_nodes       = (int)c.nodes.size();
    req.amountOfValues = (int)s.params.size() + 1 + kSchemeTailSlots;   // [s, p_1..p_M, n, Newton]

    const double spread = parse_d(c.ic_spread_text, 0.0);
    const unsigned long long ic_seed = (unsigned long long)std::max(0, parse_i(c.ic_seed_text, 1));

    req.values.assign((size_t)req.n_nodes * (size_t)req.amountOfValues, 0.0);
    req.initial_conditions.assign((size_t)req.n_nodes * (size_t)req.amountOfX, 0.0);
    for (int i = 0; i < req.n_nodes; ++i) {
        const NetNode& nd = c.nodes[(size_t)i];
        req.values[(size_t)i * req.amountOfValues] =
            parse_d(nd.symmetry_s.empty() ? c.symmetry_s : nd.symmetry_s, 0.5);
        scheme_fill_tail(req.values.data() + (size_t)i * req.amountOfValues + s.params.size() + 1,
                         c.cd_fp_iters, c.newton);   // a[M+1..M+4]: CD FP n, Newton
        for (size_t j = 0; j < s.params.size(); ++j) {
            const std::string v = node_value(nd.param_values, c.param_values, s.params[j]);
            req.values[(size_t)i * req.amountOfValues + j + 1] = parse_d(v, 0.0);
        }
        for (size_t k = 0; k < s.vars.size(); ++k) {
            const std::string v = node_value(nd.initial_conditions, c.initial_conditions, s.vars[k]);
            double x = parse_d(v, 0.0);
            // Разброс не трогает узлы с СОБСТВЕННЫМИ НУ: их пользователь
            // выставил руками, и шуметь поверх — значит игнорировать ввод.
            auto own = nd.initial_conditions.find(s.vars[k]);
            const bool has_own = (own != nd.initial_conditions.end() && !own->second.empty());
            if (!has_own && spread != 0.0)
                x += spread * (2.0 * rand01(ic_seed, (unsigned long long)i, (unsigned long long)k) - 1.0);
            req.initial_conditions[(size_t)i * req.amountOfX + k] = x;
        }
    }

    net_build_csr(c, parse_d(c.coupling_text, 1.0),
                  req.edge_start, req.edge_src, req.edge_w, req.edge_law);

    req.h           = parse_d(c.h_text, 0.01);
    req.t_max       = parse_d(c.t_max_text, 100.0);
    req.transient   = parse_d(c.transient_text, 0.0);
    req.pre_scaller = std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_points  = std::max(2, parse_i(c.max_points_text, 20000));
    req.max_value   = parse_d(c.max_value_text, 1.0e6);
    return req;
}

} // namespace

NetworkMonolithic net_monolithic_snapshot(const NetworkState& s, const NetworkConfig& c) {
    auto require = [](bool ok, const std::string& msg) {
        if (!ok) throw std::runtime_error("Monolithic export: " + msg);
    };
    require(!s.sys.is_map && s.sys.dae_nx < 0 && s.sys.out_exprs.empty() && s.sys.zb_g.empty(),
            "only ordinary ODE systems are supported");
    require(s.vars == s.sys.vars && s.params == s.sys.params, "system symbol order is inconsistent");
    const size_t n = c.nodes.size(), dim = s.vars.size();
    require(n > 0 && dim > 0 && s.sys.rhs.size() == dim, "empty or incomplete system/network");
    require(n <= 32 / dim, "the snapshot exceeds the limit of 32 state variables");
    auto number = [&](const std::string& text, double fallback, const std::string& what) {
        double value = fallback;
        require(text.empty() || parse_num_checked(text, value), "invalid numeric value for " + what);
        return value;
    };
    // Validate rather than silently dropping malformed edges or changing a law.
    for (const auto& e : c.edges) {
        require(e.from >= 0 && (size_t)e.from < n && e.to >= 0 && (size_t)e.to < n, "edge outside node list");
        require(e.law >= 0 && (size_t)e.law < c.laws.size(), "invalid coupling law index");
        number(e.weight_text, 1.0, "edge weight");
    }
    NetworkMonolithic out;
    auto& sys = out.sys;
    auto& r = out.record;
    r.name = (s.loaded_system_name.empty() ? "Network" : s.loaded_system_name)
        + " - " + c.label + " (monolithic)";
    r.mode = "Plain"; r.param_order = "AsInAlphabet";
    r.note = "Independent Network snapshot. Node indices start at 1. Coupling is included in every integration stage.\n"
             "Weights and incoming-degree normalization are fixed; K is the global coupling parameter.\n"
             "Initial spread is realized and stored. Later Network changes do not affect this record.";
    r.step_h = c.h_text; r.symmetry_s = c.symmetry_s; r.cd_fp_iters = c.cd_fp_iters;
    r.newton_tol = c.newton.tol; r.newton_max_iters = c.newton.max_iters; r.newton_full = c.newton.full;
    r.scheme_euler = r.scheme_cromer = r.scheme_midpoint = r.scheme_rk4 = true;
    r.scheme_dopri78 = r.scheme_dopri78_legacy = r.scheme_rk45 = r.scheme_dop853 = true;
    r.scheme_cd = r.scheme_cdfast = r.scheme_cdfp = r.scheme_cdfastfp = true;
    r.scheme_ccd = r.scheme_ccd4 = r.scheme_ccd4s3 = r.scheme_ccd4s4 = true;
    r.scheme_ccd4ss01 = r.scheme_ccd4ss10 = r.scheme_cd10 = r.scheme_ccd10 = true;
    r.scheme_ccd4_10 = r.scheme_ccd4s3_10 = r.scheme_ccd4s4_10 = true;
    r.scheme_ieuler = r.scheme_imidpoint = r.scheme_semp = r.scheme_simp = r.scheme_dmethod = r.scheme_cieuler = true;
    r.scheme_gbs = r.scheme_gbs24 = r.scheme_gbs246 = r.scheme_gbs2468 = r.scheme_gbs246810 = r.scheme_gbs24681012 = true;

    std::set<std::string> used{"K", "pi", "t"};
    // Reserve common names first: e.g. parameter x1 must not become a node variable.
    for (const auto& p : s.params) if (p != "K") used.insert(p);
    auto unique = [&](const std::string& preferred) {
        std::string name = preferred;
        for (int suffix = 2; !used.insert(name).second; ++suffix) name = preferred + "_" + std::to_string(suffix);
        return name;
    };
    std::vector<std::vector<std::string>> vars(n), params(n);
    for (size_t i = 0; i < n; ++i)
        for (const auto& v : s.vars) {
            const auto name = unique(v + std::to_string(i + 1));
            vars[i].push_back(name); sys.vars.push_back(name);
        }
    auto add_parameter = [&](const std::string& name, double value) {
        sys.params.push_back(name); r.param_values[name] = fmt_num_shortest(value);
    };
    for (const auto& p : s.params) {
        std::string shared;
        for (const auto& node : c.nodes) {
            const auto it = node.param_values.find(p);
            if (it == node.param_values.end() || it->second.empty()) {
                shared = p == "K" ? unique("K_model") : p;
                const auto base = c.param_values.find(p);
                add_parameter(shared, number(base == c.param_values.end() ? "" : base->second, 0.0, p));
                break;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const auto own = c.nodes[i].param_values.find(p);
            std::string name = shared;
            if (own != c.nodes[i].param_values.end() && !own->second.empty()) {
                name = unique(p + std::to_string(i + 1));
                add_parameter(name, number(own->second, 0.0, name));
            }
            params[i].push_back(name);
        }
    }
    add_parameter("K", number(c.coupling_text, 1.0, "global coupling"));
    require(sys.params.size() <= 64 - 1 - kSchemeTailSlots, "the snapshot exceeds the limit of 59 parameters");

    const double spread = number(c.ic_spread_text, 0.0, "initial spread");
    const auto seed = (unsigned long long)std::max(0, parse_i(c.ic_seed_text, 1));
    for (size_t i = 0; i < n; ++i) {
        std::map<std::string, std::string> names{{"pi", "pi"}, {"t", "t"}};
        for (size_t k = 0; k < dim; ++k) names[s.vars[k]] = vars[i][k];
        for (size_t j = 0; j < s.params.size(); ++j) names[s.params[j]] = params[i][j];
        names["pi"] = "pi"; // Same constant precedence as build_namemap in codegen.
        for (size_t k = 0; k < dim; ++k) {
            sys.rhs.push_back(codegen_substitute_expression(s.sys.rhs[k], s.sys.latex, names));
            double x = number(node_value(c.nodes[i].initial_conditions, c.initial_conditions, s.vars[k]), 0.0, vars[i][k]);
            const auto own = c.nodes[i].initial_conditions.find(s.vars[k]);
            if ((own == c.nodes[i].initial_conditions.end() || own->second.empty()) && spread != 0.0)
                x += spread * (2.0 * rand01(seed, i, k) - 1.0);
            require(std::isfinite(x), "initial spread overflow");
            r.init_conditions[vars[i][k]] = fmt_num_shortest(x);
        }
    }
    // Reuse Network's arc order and direction, but print division by degree explicitly.
    NetworkConfig raw;
    raw.nodes.resize(n); raw.edges = c.edges; raw.laws = c.laws;
    raw.normalize_coupling = false;
    std::vector<int> start, src, law;
    std::vector<double> weight;
    net_build_csr(raw, 1.0, start, src, weight, law);
    for (size_t i = 0; i < n; ++i) {
        const int degree = start[i + 1] - start[i];
        for (int e = start[i]; e < start[i + 1]; ++e) {
            const size_t j = (size_t)src[e];
            std::string w = "(K*(" + fmt_num_shortest(weight[e]) + "))";
            if (c.normalize_coupling) w = "(" + w + "/" + std::to_string(degree) + ")";
            std::map<std::string, std::string> names{{"K", w}, {"w", w}, {"pi", "pi"}};
            for (size_t k = 0; k < dim; ++k) {
                names[s.vars[k]] = names[s.vars[k] + "_i"] = vars[i][k];
                names[s.vars[k] + "_j"] = vars[j][k];
            }
            // Same shadowing order as codegen_coupling, including a model parameter K/w.
            for (size_t p = 0; p < s.params.size(); ++p) {
                names[s.params[p]] = names[s.params[p] + "_i"] = params[i][p];
                names[s.params[p] + "_j"] = params[j][p];
            }
            const auto& expr = c.laws[(size_t)law[e]].expr;
            require(expr.size() <= dim, "coupling expression count exceeds node dimension");
            for (size_t k = 0; k < expr.size(); ++k) if (!expr[k].empty())
                sys.rhs[i * dim + k] += "+(" + codegen_substitute_expression(expr[k], false, names) + ")";
        }
    }
    auto list = [](const std::vector<std::string>& names) {
        std::string text;
        for (const auto& name : names) { if (!text.empty()) text += ", "; text += name; }
        return text;
    };
    r.vars_text = list(sys.vars); r.params_text = list(sys.params);
    // The common input parser separates equations with two backslashes.
    for (size_t k = 0; k < sys.vars.size(); ++k)
        r.plain_text += sys.vars[k] + "' = " + sys.rhs[k] + " " + std::string(2, '\\') + "\n";
    std::vector<std::string> alphabet = sys.vars;
    alphabet.insert(alphabet.end(), sys.params.begin(), sys.params.end());
    const auto parsed = parse_system_from_latex(r.plain_text, alphabet);
    require(parsed.vars == sys.vars && parsed.params == sys.params, "record parser changed the symbol layout");
    // Validate the complete mathematical system before allowing a library write.
    codegen_rhs(sys);
    return out;
}

void NetworkSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_)
{
    monolithic_saved_name.clear();
    monolithic_error.clear();
    vars   = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    NetworkConfig c;
    c.label      = "Network 1";
    c.h_text     = default_h_from_record(r);
    c.scheme     = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    c.cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    c.newton = newton_from_record(r);

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        c.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }

    c.laws.clear();
    c.laws.push_back(net_default_law(vars));
    net_generate_topology(c, vars);

    configs.clear();
    configs.push_back(std::move(c));
    active_config_index  = 0;
    running_config_index = -1;
}

void NetworkSession::add_config() {
    NetworkConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = NetworkResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
    } else {
        for (const auto& p : params) c.param_values[p] = "";
        for (const auto& v : vars)   c.initial_conditions[v] = "";
        c.laws.push_back(net_default_law(vars));
        net_generate_topology(c, vars);
    }
    c.label = "Network " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}

void NetworkSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    if (configs.size() == 1) return;          // последний конфиг вкладке нужен
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (running_config_index > i) --running_config_index;
    if (active_config_index >= (int)configs.size()) active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}

bool NetworkSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    NetworkConfig& c = configs[(size_t)config_idx];
    c.last_error.clear();
    last_run_label.clear();

    NetworkRequest req;
    try {
        req = build_network_request(*this, c);
    } catch (const std::exception& e) {
        // Единственный источник исключения здесь — разбор выражения связи.
        c.last_error = std::string("coupling: ") + e.what();
        return false;
    }
    if (req.krs_body.empty()) {
        c.last_error = krs_empty_error(custom_schemes, sys, c.scheme);
        return false;
    }
    if (req.n_nodes <= 0) {
        c.last_error = "the network has no nodes";
        return false;
    }

    cancel_token   = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);
    req.cancel   = cancel_token;
    req.progress = progress_token;

    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_network(req);
    });
    return true;
}

void NetworkSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}

bool NetworkSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    const int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("network");

    NetworkResult r = run_future.get();
    const bool cancelled = r.cancelled;
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        NetworkConfig& c = configs[(size_t)idx];
        c.last_run_ok = r.ok;
        c.last_error  = r.ok ? std::string() : r.error;
        c.result      = std::move(r);
        ++c.data_generation;
    }

    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);

    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
