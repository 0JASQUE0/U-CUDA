#include "session_json_internal.h"
#include <algorithm>

using namespace ucuda::storage_detail;

namespace {

// Тот же набор ключей, что пишет write_shared_config (читатель Custom встроен в
// custom_state_from_json, поэтому здесь своя копия).
void read_shared_cfg(const std::string& raw, CustomTabSharedConfig& c) {
    JP q(raw);
    q.expect('{');
    if (q.opt('}')) return;
    while (true) {
        std::string k = q.str(); q.expect(':');
        if      (k == "scheme")                  c.scheme = q.str();
        else if (k == "symmetry_s")              c.symmetry_s = q.str();
        else if (k == "cd_fp_iters")             c.cd_fp_iters = q.str();
        else if (k == "newton_tol")              c.newton.tol = q.str();
        else if (k == "newton_max_iters")        c.newton.max_iters = q.str();
        else if (k == "newton_full")             c.newton.full = q.boolean();
        else if (k == "h_text")                  c.h_text = q.str();
        else if (k == "t_max_text")              c.t_max_text = q.str();
        else if (k == "transient_text")          c.transient_text = q.str();
        else if (k == "adaptive")                read_adaptive(q, c.adaptive);
        else if (k == "pre_scaller_text")        c.pre_scaller_text = q.str();
        else if (k == "max_value_text")          c.max_value_text = q.str();
        else if (k == "initial_conditions")      c.initial_conditions = q.map_ss();
        else if (k == "param_values")            c.param_values = q.map_ss();
        else if (k == "axis_x_par_index")        c.axis_x_par_index = std::stoi(q.str_or_num());
        else if (k == "axis_x_over_var")         c.axis_x_over_var = q.boolean();
        else if (k == "axis_x_over_h")           c.axis_x_over_h = q.boolean();
        else if (k == "axis_x_var_index")        c.axis_x_var_index = std::stoi(q.str_or_num());
        else if (k == "axis_x_lo_text")          c.axis_x_lo_text = q.str();
        else if (k == "axis_x_hi_text")          c.axis_x_hi_text = q.str();
        else if (k == "axis_x_log")              c.axis_x_log = q.boolean();
        else if (k == "axis_y_par_index")        c.axis_y_par_index = std::stoi(q.str_or_num());
        else if (k == "axis_y_over_var")         c.axis_y_over_var = q.boolean();
        else if (k == "axis_y_over_h")           c.axis_y_over_h = q.boolean();
        else if (k == "axis_y_var_index")        c.axis_y_var_index = std::stoi(q.str_or_num());
        else if (k == "axis_y_lo_text")          c.axis_y_lo_text = q.str();
        else if (k == "axis_y_hi_text")          c.axis_y_hi_text = q.str();
        else if (k == "axis_y_log")              c.axis_y_log = q.boolean();
        else if (k == "bif_writable_var")        c.bif_writable_var = std::stoi(q.str_or_num());
        else if (k == "resolution_text")         c.resolution_text = q.str();
        else if (k == "inherit_sweep_from_2d")   c.inherit_sweep_from_2d = q.boolean();
        else if (k == "sweep_x_par_index")       c.sweep_x_par_index = std::stoi(q.str_or_num());
        else if (k == "sweep_x_over_var")        c.sweep_x_over_var = q.boolean();
        else if (k == "sweep_x_over_h")          c.sweep_x_over_h = q.boolean();
        else if (k == "sweep_x_var_index")       c.sweep_x_var_index = std::stoi(q.str_or_num());
        else if (k == "sweep_x_lo_text")         c.sweep_x_lo_text = q.str();
        else if (k == "sweep_x_hi_text")         c.sweep_x_hi_text = q.str();
        else if (k == "sweep_x_log")             c.sweep_x_log = q.boolean();
        else if (k == "n_x_1d_text")             c.n_x_1d_text = q.str();
        else if (k == "sweep_y_par_index")       c.sweep_y_par_index = std::stoi(q.str_or_num());
        else if (k == "sweep_y_over_var")        c.sweep_y_over_var = q.boolean();
        else if (k == "sweep_y_over_h")          c.sweep_y_over_h = q.boolean();
        else if (k == "sweep_y_var_index")       c.sweep_y_var_index = std::stoi(q.str_or_num());
        else if (k == "sweep_y_lo_text")         c.sweep_y_lo_text = q.str();
        else if (k == "sweep_y_hi_text")         c.sweep_y_hi_text = q.str();
        else if (k == "sweep_y_log")             c.sweep_y_log = q.boolean();
        else if (k == "n_y_1d_text")             c.n_y_1d_text = q.str();
        else if (k == "l1d_h_text")              c.l1d_h_text = q.str();
        else if (k == "l1d_transient_text")      c.l1d_transient_text = q.str();
        else if (k == "l1d_t_max_text")          c.l1d_t_max_text = q.str();
        else if (k == "continuation_1d_enabled") c.continuation_1d_enabled = q.boolean();
        else if (k == "plot_inter_peaks_1d")     c.plot_inter_peaks_1d = q.boolean();
        else if (k == "fix_x_value")             c.fix_x_value = std::stod(q.str_or_num());
        else if (k == "fix_y_value")             c.fix_y_value = std::stod(q.str_or_num());
        else                                     q.skip_value();
        if (q.opt(',')) continue;
        q.expect('}'); break;
    }
}

std::vector<int> read_int_array(JP& q) {
    std::vector<int> v;
    q.expect('[');
    if (q.opt(']')) return v;
    while (true) {
        v.push_back(std::stoi(q.str_or_num()));
        if (q.opt(',')) continue;
        q.expect(']'); break;
    }
    return v;
}

} // namespace

std::string tree_state_to_json(const AnalysisTree& t, const std::map<int, std::string>& leaf_json) {
    std::ostringstream o;
    o << "{\n  \"version\":1,\n";
    o << "  \"next_id\":" << t.next_id << ",\n";
    o << "  \"selected_id\":" << t.selected_id << ",\n";
    o << "  \"nodes\":[";
    for (size_t i = 0; i < t.nodes.size(); ++i) {
        const TreeNode& n = t.nodes[i];
        o << (i ? ",\n    " : "\n    ") << "{";
        o << "\"id\":" << n.id << ",\"parent\":" << n.parent << ",";
        o << "\"kind\":"; jstr(o, tree_kind_key(n.kind)); o << ",";
        o << "\"name\":"; jstr(o, n.name); o << ",";
        o << "\"enabled\":" << (n.enabled ? "true" : "false") << ",";
        o << "\"expanded\":" << (n.expanded ? "true" : "false") << ",";
        o << "\"was_expanded\":" << (n.was_expanded ? "true" : "false") << ",";
        o << "\"follow_cursor\":" << (n.follow_cursor ? "true" : "false") << ",";
        o << "\"live\":" << (n.live ? "true" : "false") << ",";
        o << "\"slice_dir\":" << n.slice_dir << ",";
        o << "\"children\":[";
        for (size_t c = 0; c < n.children.size(); ++c) o << (c ? "," : "") << n.children[c];
        o << "]";
        if (tree_kind_is_context(n.kind)) {
            o << ",\"cfg\":"; write_shared_config(o, n.cfg);
        } else {
            auto it = leaf_json.find(n.id);
            if (it != leaf_json.end() && !it->second.empty()) o << ",\"leaf\":" << it->second;
        }
        o << "}";
    }
    o << "\n  ]\n}\n";
    return o.str();
}

bool tree_state_from_json(const std::string& json, AnalysisTree& out, std::map<int, std::string>& leaf_json) {
    AnalysisTree t;
    std::map<int, std::string> leaves;
    try {
        JP p(json);
        p.expect('{');
        if (!p.opt('}')) {
            while (true) {
                std::string key = p.str(); p.expect(':');
                if      (key == "next_id")     t.next_id = std::stoi(p.str_or_num());
                else if (key == "selected_id") t.selected_id = std::stoi(p.str_or_num());
                else if (key == "nodes") {
                    p.expect('[');
                    if (!p.opt(']')) {
                        while (true) {
                            TreeNode n;
                            bool kind_ok = false;
                            p.expect('{');
                            if (!p.opt('}')) {
                                while (true) {
                                    std::string k = p.str(); p.expect(':');
                                    if      (k == "id")            n.id = std::stoi(p.str_or_num());
                                    else if (k == "parent")        n.parent = std::stoi(p.str_or_num());
                                    else if (k == "kind")          kind_ok = tree_kind_from_key(p.str(), n.kind);
                                    else if (k == "name")          n.name = p.str();
                                    else if (k == "enabled")       n.enabled = p.boolean();
                                    else if (k == "expanded")      n.expanded = p.boolean();
                                    else if (k == "was_expanded")  n.was_expanded = p.boolean();
                                    else if (k == "follow_cursor") n.follow_cursor = p.boolean();
                                    else if (k == "live")          n.live = p.boolean();
                                    else if (k == "slice_dir")     n.slice_dir = std::stoi(p.str_or_num());
                                    else if (k == "children")      n.children = read_int_array(p);
                                    else if (k == "cfg" || k == "leaf") {
                                        p.ws();
                                        const size_t a = p.i;
                                        p.skip_value();
                                        std::string raw = json.substr(a, p.i - a);
                                        if (k == "cfg") read_shared_cfg(raw, n.cfg);
                                        else            leaves[n.id] = std::move(raw);
                                    }
                                    else p.skip_value();
                                    if (p.opt(',')) continue;
                                    p.expect('}'); break;
                                }
                            }
                            // Неизвестный вид узла (файл новее программы) — дерево не грузим целиком.
                            if (!kind_ok) return false;
                            t.nodes.push_back(std::move(n));
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
    }
    catch (...) {
        return false;
    }
    // Проверка связности: корень System первым, у каждого ребёнка верный родитель.
    if (t.nodes.empty() || t.nodes.front().kind != TreeNodeKind::System || t.nodes.front().parent != 0)
        return false;
    int max_id = 0;
    for (const auto& n : t.nodes) {
        max_id = std::max(max_id, n.id);
        for (int c : n.children) {
            const TreeNode* cn = t.find(c);
            if (!cn || cn->parent != n.id) return false;
        }
        if (n.parent != 0 && !t.find(n.parent)) return false;
    }
    if (t.next_id <= max_id) t.next_id = max_id + 1;
    if (!t.find(t.selected_id)) t.selected_id = t.nodes.front().id;
    out = std::move(t);
    leaf_json = std::move(leaves);
    return true;
}
