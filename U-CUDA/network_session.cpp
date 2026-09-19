#include "network_session.h"
#include "num_parse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

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
        c.nodes[(size_t)i].label = std::to_string(i + 1);
    }

    std::set<long long> seen;
    std::vector<NetEdge> edges;
    auto add = [&](int a, int b) {
        if (a == b || a < 0 || b < 0 || a >= n || b >= n) return;
        if (!c.gen_directed && !seen.insert(edge_key(a, b)).second) return;
        NetEdge e;
        e.from = a; e.to = b;
        e.bidirectional = !c.gen_directed;
        e.weight_text = "1";
        e.law = 0;
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
    req.amountOfValues = (int)s.params.size() + 1;          // a[0] = symmetry s

    const double spread = parse_d(c.ic_spread_text, 0.0);
    const unsigned long long ic_seed = (unsigned long long)std::max(0, parse_i(c.ic_seed_text, 1));

    req.values.assign((size_t)req.n_nodes * (size_t)req.amountOfValues, 0.0);
    req.initial_conditions.assign((size_t)req.n_nodes * (size_t)req.amountOfX, 0.0);
    for (int i = 0; i < req.n_nodes; ++i) {
        const NetNode& nd = c.nodes[(size_t)i];
        req.values[(size_t)i * req.amountOfValues] = parse_d(c.symmetry_s, 0.5);
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

void NetworkSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_)
{
    vars   = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    NetworkConfig c;
    c.label      = "Network 1";
    c.h_text     = default_h_from_record(r);
    c.scheme     = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

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
        c.last_error = "krs_code is empty (no valid system or scheme)";
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
