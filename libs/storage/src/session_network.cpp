#include "session_json_internal.h"
namespace ucuda::storage_detail {
void jmap_nonempty(std::ostringstream& o, const std::map<std::string, std::string>& m) {
    o << "{"; bool first = true;
    for (const auto& kv : m) {
        if (kv.second.empty()) continue;
        if (!first) o << ",";
        o << '"' << esc(kv.first) << "\":\"" << esc(kv.second) << '"';
        first = false;
    }
    o << "}";
}


void write_network_config(std::ostringstream& o, const NetworkConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
    o << "\"scheme\":";           jstr(o, c.scheme);           o << ",";
    o << "\"symmetry_s\":";       jstr(o, c.symmetry_s);       o << ",";
    o << "\"h_text\":";           jstr(o, c.h_text);           o << ",";
    o << "\"t_max_text\":";       jstr(o, c.t_max_text);       o << ",";
    o << "\"transient_text\":";   jstr(o, c.transient_text);   o << ",";
    o << "\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text); o << ",";
    o << "\"max_value_text\":";   jstr(o, c.max_value_text);   o << ",";
    o << "\"max_points_text\":";  jstr(o, c.max_points_text);  o << ",";
    o << "\"coupling_text\":";    jstr(o, c.coupling_text);    o << ",";
    o << "\"normalize_coupling\":" << (c.normalize_coupling ? "true" : "false") << ",";
    o << "\"topology\":"          << (int)c.topology         << ",";
    o << "\"gen_n_text\":";       jstr(o, c.gen_n_text);       o << ",";
    o << "\"gen_k_text\":";       jstr(o, c.gen_k_text);       o << ",";
    o << "\"gen_w_text\":";       jstr(o, c.gen_w_text);       o << ",";
    o << "\"gen_h_text\":";       jstr(o, c.gen_h_text);       o << ",";
    o << "\"gen_p_text\":";       jstr(o, c.gen_p_text);       o << ",";
    o << "\"gen_seed_text\":";    jstr(o, c.gen_seed_text);    o << ",";
    o << "\"gen_periodic\":"      << (c.gen_periodic ? "true" : "false") << ",";
    o << "\"gen_directed\":"      << (c.gen_directed ? "true" : "false") << ",";
    o << "\"ic_spread_text\":";   jstr(o, c.ic_spread_text);   o << ",";
    o << "\"ic_seed_text\":";     jstr(o, c.ic_seed_text);     o << ",";
    o << "\"param_values\":";     jmap(o, c.param_values);     o << ",";
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";

    o << "\"laws\":[";
    for (size_t i = 0; i < c.laws.size(); ++i) {
        if (i) o << ",";
        o << "{\"name\":"; jstr(o, c.laws[i].name);
        o << ",\"expr\":[";
        for (size_t k = 0; k < c.laws[i].expr.size(); ++k) {
            if (k) o << ",";
            jstr(o, c.laws[i].expr[k]);
        }
        o << "]}";
    }
    o << "],";

    o << "\"nodes\":[";
    for (size_t i = 0; i < c.nodes.size(); ++i) {
        if (i) o << ",";
        const NetNode& nd = c.nodes[i];
        o << "{\"label\":"; jstr(o, nd.label);
        o << ",\"x\":" << nd.ui_x << ",\"y\":" << nd.ui_y;
        if (!nd.symmetry_s.empty()) { o << ",\"s\":"; jstr(o, nd.symmetry_s); }
        o << ",\"params\":"; jmap_nonempty(o, nd.param_values);
        o << ",\"ic\":";     jmap_nonempty(o, nd.initial_conditions);
        o << "}";
    }
    o << "],";

    o << "\"edges\":[";
    for (size_t i = 0; i < c.edges.size(); ++i) {
        if (i) o << ",";
        const NetEdge& e = c.edges[i];
        o << "{\"from\":" << e.from << ",\"to\":" << e.to
          << ",\"bidir\":" << (e.bidirectional ? "true" : "false")
          << ",\"w\":"; jstr(o, e.weight_text);
        o << ",\"law\":" << e.law << "}";
    }
    o << "]";
    o << "}";
}


bool read_network_field(JP& p, NetworkConfig& c, const std::string& key) {
    if      (key == "label")              c.label              = p.str();
    else if (key == "scheme")             c.scheme             = p.str();
    else if (key == "symmetry_s")         c.symmetry_s         = p.str();
    else if (key == "h_text")             c.h_text             = p.str();
    else if (key == "t_max_text")         c.t_max_text         = p.str();
    else if (key == "transient_text")     c.transient_text     = p.str();
    else if (key == "pre_scaller_text")   c.pre_scaller_text   = p.str();
    else if (key == "max_value_text")     c.max_value_text     = p.str();
    else if (key == "max_points_text")    c.max_points_text    = p.str();
    else if (key == "coupling_text")      c.coupling_text      = p.str();
    else if (key == "normalize_coupling") c.normalize_coupling = p.boolean();
    else if (key == "topology")           c.topology           = (NetTopology)std::stoi(p.str_or_num());
    else if (key == "gen_n_text")         c.gen_n_text         = p.str();
    else if (key == "gen_k_text")         c.gen_k_text         = p.str();
    else if (key == "gen_w_text")         c.gen_w_text         = p.str();
    else if (key == "gen_h_text")         c.gen_h_text         = p.str();
    else if (key == "gen_p_text")         c.gen_p_text         = p.str();
    else if (key == "gen_seed_text")      c.gen_seed_text      = p.str();
    else if (key == "gen_periodic")       c.gen_periodic       = p.boolean();
    else if (key == "gen_directed")       c.gen_directed       = p.boolean();
    else if (key == "ic_spread_text")     c.ic_spread_text     = p.str();
    else if (key == "ic_seed_text")       c.ic_seed_text       = p.str();
    else if (key == "param_values")       c.param_values       = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions = p.map_ss();
    else if (key == "laws") {
        c.laws.clear();
        p.expect('[');
        if (!p.opt(']')) {
            while (true) {
                NetCouplingLaw law;
                p.expect('{');
                if (!p.opt('}')) {
                    while (true) {
                        std::string k = p.str(); p.expect(':');
                        if (k == "name") law.name = p.str();
                        else if (k == "expr") {
                            p.expect('[');
                            if (!p.opt(']')) {
                                while (true) {
                                    law.expr.push_back(p.str());
                                    if (p.opt(',')) continue;
                                    p.expect(']'); break;
                                }
                            }
                        }
                        else p.skip_value();
                        if (p.opt(',')) continue;
                        p.expect('}'); break;
                    }
                }
                c.laws.push_back(std::move(law));
                if (p.opt(',')) continue;
                p.expect(']'); break;
            }
        }
    }
    else if (key == "nodes") {
        c.nodes.clear();
        p.expect('[');
        if (!p.opt(']')) {
            while (true) {
                NetNode nd;
                p.expect('{');
                if (!p.opt('}')) {
                    while (true) {
                        std::string k = p.str(); p.expect(':');
                        if      (k == "label")  nd.label = p.str();
                        else if (k == "s")      nd.symmetry_s = p.str();
                        else if (k == "x")      nd.ui_x  = (float)std::stod(p.str_or_num());
                        else if (k == "y")      nd.ui_y  = (float)std::stod(p.str_or_num());
                        else if (k == "params") nd.param_values = p.map_ss();
                        else if (k == "ic")     nd.initial_conditions = p.map_ss();
                        else p.skip_value();
                        if (p.opt(',')) continue;
                        p.expect('}'); break;
                    }
                }
                c.nodes.push_back(std::move(nd));
                if (p.opt(',')) continue;
                p.expect(']'); break;
            }
        }
    }
    else if (key == "edges") {
        c.edges.clear();
        p.expect('[');
        if (!p.opt(']')) {
            while (true) {
                NetEdge e;
                p.expect('{');
                if (!p.opt('}')) {
                    while (true) {
                        std::string k = p.str(); p.expect(':');
                        if      (k == "from")  e.from = std::stoi(p.str_or_num());
                        else if (k == "to")    e.to   = std::stoi(p.str_or_num());
                        else if (k == "bidir") e.bidirectional = p.boolean();
                        else if (k == "w")     e.weight_text = p.str();
                        else if (k == "law")   e.law = std::stoi(p.str_or_num());
                        else p.skip_value();
                        if (p.opt(',')) continue;
                        p.expect('}'); break;
                    }
                }
                c.edges.push_back(e);
                if (p.opt(',')) continue;
                p.expect(']'); break;
            }
        }
    }
    else return false;
    return true;
}


// SignalMetricsAnalysisSession — `_last_metrics.json`. Структура как у LLE:
// массив конфигов, результат не хранится (после загрузки — Run заново).



}
using namespace ucuda::storage_detail;
std::string session_to_json_network(const NetworkSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_network_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_network(const std::string& json, NetworkSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                s.configs.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        NetworkConfig nc;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_network_field(p, nc, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        // Закон связи нужен ВСЕГДА: рёбра ссылаются на него
                        // индексом, и пустой список законов превратил бы файл
                        // в сеть, которую нельзя посчитать.
                        if (nc.laws.empty()) nc.laws.push_back(net_default_law(s.vars));
                        s.configs.push_back(std::move(nc));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") s.active_config_index = std::stoi(p.str_or_num());
            else                                    p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
            s.active_config_index = 0;
        s.running_config_index = -1;
        return true;
    } catch (...) {
        return false;
    }
}


 // namespace
