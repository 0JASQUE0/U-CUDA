#include "session_json_internal.h"
namespace ucuda::storage_detail {
void write_order_config(std::ostringstream& o, const OrderConfig& c) {
    o << "{";
    o << "\"label\":";           jstr(o, c.label);           o << ",";
    o << "\"scheme\":";          jstr(o, c.scheme);          o << ",";
    o << "\"symmetry_s\":";      jstr(o, c.symmetry_s);      o << ",";
    o << "\"calc_kind\":"        << c.calc_kind             << ",";
    o << "\"two_d\":"            << (c.two_d ? "true" : "false") << ",";
    o << "\"axis_x_target\":"    << c.axis_x_target         << ",";
    o << "\"axis_x_lo_text\":";  jstr(o, c.axis_x_lo_text);  o << ",";
    o << "\"axis_x_hi_text\":";  jstr(o, c.axis_x_hi_text);  o << ",";
    o << "\"axis_x_log\":"       << (c.axis_x_log ? "true" : "false") << ",";
    o << "\"axis_x_n_text\":";   jstr(o, c.axis_x_n_text);   o << ",";
    o << "\"axis_y_target\":"    << c.axis_y_target         << ",";
    o << "\"axis_y_lo_text\":";  jstr(o, c.axis_y_lo_text);  o << ",";
    o << "\"axis_y_hi_text\":";  jstr(o, c.axis_y_hi_text);  o << ",";
    o << "\"axis_y_log\":"       << (c.axis_y_log ? "true" : "false") << ",";
    o << "\"h_text\":";          jstr(o, c.h_text);          o << ",";
    o << "\"t_max_text\":";      jstr(o, c.t_max_text);      o << ",";
    o << "\"max_value_text\":";  jstr(o, c.max_value_text);  o << ",";
    o << "\"use_gpu\":"          << (c.use_gpu ? "true" : "false") << ",";
    o << "\"cpu_precision\":"    << c.cpu_precision << ",";
    o << "\"snap_steps\":"       << (c.snap_steps ? "true" : "false") << ",";
    o << "\"endpoint_only\":"    << (c.endpoint_only ? "true" : "false") << ",";
    o << "\"perf_repeats_text\":";  jstr(o, c.perf_repeats_text);  o << ",";
    o << "\"perf_warmup_text\":";   jstr(o, c.perf_warmup_text);   o << ",";
    o << "\"perf_replicas_text\":"; jstr(o, c.perf_replicas_text); o << ",";
    o << "\"perf_ref_scheme\":";   jstr(o, c.perf_ref_scheme);   o << ",";
    o << "\"perf_end_ref\":" << c.perf_end_ref << ",";
    o << "\"perf_tol_lo_text\":"; jstr(o, c.perf_tol_lo_text); o << ",";
    o << "\"perf_tol_hi_text\":"; jstr(o, c.perf_tol_hi_text); o << ",";
    o << "\"perf_tol_n_text\":";  jstr(o, c.perf_tol_n_text);  o << ",";
    write_adaptive(o, c.adaptive, "", ",");
    o << "\"perf_ref_substeps_text\":"; jstr(o, c.perf_ref_substeps_text); o << ",";
    o << "\"stab_k_text\":";      jstr(o, c.stab_k_text);      o << ",";
    o << "\"stab_r_text\":";      jstr(o, c.stab_r_text);      o << ",";
    o << "\"stab_h_text\":";      jstr(o, c.stab_h_text);      o << ",";
    o << "\"stab_sig_lo_text\":"; jstr(o, c.stab_sig_lo_text); o << ",";
    o << "\"stab_sig_hi_text\":"; jstr(o, c.stab_sig_hi_text); o << ",";
    o << "\"stab_om_lo_text\":";  jstr(o, c.stab_om_lo_text);  o << ",";
    o << "\"stab_om_hi_text\":";  jstr(o, c.stab_om_hi_text);  o << ",";
    o << "\"stab_n_text\":";      jstr(o, c.stab_n_text);      o << ",";
    o << "\"stab_idx_a\":" << c.stab_idx_a << ",";
    o << "\"stab_idx_b\":" << c.stab_idx_b << ",";
    o << "\"stab_idx_c\":" << c.stab_idx_c << ",";
    o << "\"stab_idx_d\":" << c.stab_idx_d << ",";
    o << "\"stab_pref_tol_text\":"; jstr(o, c.stab_pref_tol_text); o << ",";
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";
    o << "\"param_values\":";       jmap(o, c.param_values);
    o << "}";
}


bool read_order_field(JP& p, OrderConfig& c, const std::string& key) {
    if      (key == "label")            c.label            = p.str();
    else if (key == "scheme")           c.scheme           = p.str();
    else if (key == "symmetry_s")       c.symmetry_s       = p.str();
    else if (key == "calc_kind") {
        const int k = std::stoi(p.str_or_num());
        c.calc_kind = (k == kOrderCalcPerf || k == kOrderCalcStab) ? k : kOrderCalcOrder;
    }
    else if (key == "stab_k_text")      c.stab_k_text      = p.str();
    else if (key == "stab_r_text")      c.stab_r_text      = p.str();
    else if (key == "stab_h_text")      c.stab_h_text      = p.str();
    else if (key == "stab_sig_lo_text") c.stab_sig_lo_text = p.str();
    else if (key == "stab_sig_hi_text") c.stab_sig_hi_text = p.str();
    else if (key == "stab_om_lo_text")  c.stab_om_lo_text  = p.str();
    else if (key == "stab_om_hi_text")  c.stab_om_hi_text  = p.str();
    else if (key == "stab_n_text")      c.stab_n_text      = p.str();
    else if (key == "stab_idx_a")       c.stab_idx_a       = std::stoi(p.str_or_num());
    else if (key == "stab_idx_b")       c.stab_idx_b       = std::stoi(p.str_or_num());
    else if (key == "stab_idx_c")       c.stab_idx_c       = std::stoi(p.str_or_num());
    else if (key == "stab_idx_d")       c.stab_idx_d       = std::stoi(p.str_or_num());
    else if (key == "stab_pref_tol_text") c.stab_pref_tol_text = p.str();
    else if (key == "two_d")            c.two_d            = p.boolean();
    else if (key == "axis_x_target")    c.axis_x_target    = std::stoi(p.str_or_num());
    else if (key == "axis_x_lo_text")   c.axis_x_lo_text   = p.str();
    else if (key == "axis_x_hi_text")   c.axis_x_hi_text   = p.str();
    else if (key == "axis_x_log")       c.axis_x_log       = p.boolean();
    else if (key == "axis_x_n_text")    c.axis_x_n_text    = p.str();
    else if (key == "axis_y_target")    c.axis_y_target    = std::stoi(p.str_or_num());
    else if (key == "axis_y_lo_text")   c.axis_y_lo_text   = p.str();
    else if (key == "axis_y_hi_text")   c.axis_y_hi_text   = p.str();
    else if (key == "axis_y_log")       c.axis_y_log       = p.boolean();
    else if (key == "h_text")           c.h_text           = p.str();
    else if (key == "t_max_text")       c.t_max_text       = p.str();
    else if (key == "max_value_text")   c.max_value_text   = p.str();
    else if (key == "use_gpu")          c.use_gpu          = p.boolean();
    else if (key == "cpu_precision") {
        const int k = std::stoi(p.str_or_num());
        c.cpu_precision = (k == kOrderPrecDD || k == kOrderPrecQD) ? k : kOrderPrecDouble;
    }
    else if (key == "snap_steps")       c.snap_steps       = p.boolean();
    else if (key == "endpoint_only")    c.endpoint_only    = p.boolean();
    else if (key == "perf_repeats_text")  c.perf_repeats_text  = p.str();
    else if (key == "perf_warmup_text")   c.perf_warmup_text   = p.str();
    else if (key == "perf_replicas_text") c.perf_replicas_text = p.str();
    else if (key == "perf_ref_scheme")       c.perf_ref_scheme       = p.str();
    else if (key == "perf_end_ref")          { try { c.perf_end_ref = std::stoi(p.str_or_num()); } catch (...) {} }
    else if (key == "perf_tol_lo_text")      c.perf_tol_lo_text      = p.str();
    else if (key == "perf_tol_hi_text")      c.perf_tol_hi_text      = p.str();
    else if (key == "perf_tol_n_text")       c.perf_tol_n_text       = p.str();
    else if (key == "adaptive")              read_adaptive(p, c.adaptive);
    else if (key == "perf_ref_substeps_text") c.perf_ref_substeps_text = p.str();
    else if (key == "initial_conditions") c.initial_conditions = p.map_ss();
    else if (key == "param_values")       c.param_values       = p.map_ss();
    else return false;
    return true;
}


// Custom-tab bundle: shared config + five embedded sub-session objects.


}
using namespace ucuda::storage_detail;
std::string session_to_json_order(const OrderAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_order_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_order(const std::string& json, OrderAnalysisSession& s) {
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
                        OrderConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_order_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        s.configs.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") s.active_config_index = std::stoi(p.str_or_num());
            else                                   p.skip_value();
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


std::string session_to_json_order_windows(const std::vector<OrderPlotWindow>& wins) {
    std::ostringstream o;
    o << "{\n  \"windows\":[";
    for (size_t i = 0; i < wins.size(); ++i) {
        if (i) o << ",";
        const auto& w = wins[i];
        o << "{\"id\":" << w.id;
        o << ",\"kind\":" << (int)w.kind;
        o << ",\"label\":"; jstr(o, w.label);
        o << ",\"label_is_manual\":" << (w.label_is_manual ? "true" : "false");
        o << ",\"x_log\":"        << (w.x_log ? "true" : "false");
        o << ",\"y_log\":"        << (w.y_log ? "true" : "false");
        o << ",\"show_e2\":"      << (w.show_e2 ? "true" : "false");
        o << ",\"show_nominal\":" << (w.show_nominal ? "true" : "false");
        o << ",\"map_error\":"    << (w.map_error ? "true" : "false");
        o << ",\"stab_view\":"     << w.stab_view;
        o << ",\"stab_show_pref\":" << (w.stab_show_pref ? "true" : "false");
        o << ",\"stab_pref_fill\":" << w.stab_pref_fill;
        o << ",\"stab_pref_edge\":" << w.stab_pref_edge;
        o << ",\"stab_pref_edge_w\":" << w.stab_pref_edge_w;
        o << ",\"stab_paper_cmap\":" << (w.stab_paper_cmap ? "true" : "false");
        o << ",\"colormap_idx\":" << w.colormap_idx;
        o << ",\"error_source\":" << w.error_source;
        o << ",\"time_unit\":"    << w.time_unit;
        o << ",\"perf_cost\":"    << w.perf_cost;
        o << ",\"show_min\":" << (w.show_min ? "true" : "false");
        o << ",\"show_avg\":" << (w.show_avg ? "true" : "false");
        o << ",\"show_max\":" << (w.show_max ? "true" : "false");
        o << ",\"custom_line_style\":" << (w.custom_line_style ? "true" : "false");
        o << ",\"line_width\":"   << w.line_width;
        o << ",\"line_alpha\":"   << w.line_alpha;
        o << ",\"points_mode\":"  << (w.points_mode ? "true" : "false");
        o << ",\"point_marker\":" << w.point_marker;
        o << ",\"point_size\":"   << w.point_size;
        o << ",\"node_markers\":" << (w.node_markers ? "true" : "false");
        o << ",\"vary_markers\":" << (w.vary_markers ? "true" : "false");
        o << ",\"palette\":"      << w.palette;
        o << ",\"members\":[";
        for (size_t k = 0; k < w.members.size(); ++k) { if (k) o << ","; o << w.members[k]; }
        o << "]}";
    }
    o << "]\n}\n";
    return o.str();
}


bool session_from_json_order_windows(const std::string& json, std::vector<OrderPlotWindow>& wins) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) { wins.clear(); return true; }
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "windows") {
                wins.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        OrderPlotWindow w;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k = p.str(); p.expect(':');
                                if      (k == "id")              w.id              = std::stoi(p.str_or_num());
                                else if (k == "kind") {
                                    const int kk = std::stoi(p.str_or_num());
                                    w.kind = (kk >= 0 && kk <= 3) ? (OrderPlotWindow::Kind)kk
                                                                  : OrderPlotWindow::Kind::P;
                                }
                                else if (k == "label")           w.label           = p.str();
                                else if (k == "label_is_manual") w.label_is_manual = p.boolean();
                                else if (k == "x_log")           w.x_log           = p.boolean();
                                else if (k == "y_log")           w.y_log           = p.boolean();
                                else if (k == "show_e2")         w.show_e2         = p.boolean();
                                else if (k == "show_nominal")    w.show_nominal    = p.boolean();
                                else if (k == "map_error")       w.map_error       = p.boolean();
                                else if (k == "stab_view")       w.stab_view       = std::stoi(p.str_or_num());
                                else if (k == "stab_show_pref")  w.stab_show_pref  = p.boolean();
                                // Цвет IM_COL32 с альфой в старшем байте не
                                // влезает в int — stoul, не stoi.
                                else if (k == "stab_pref_fill")  w.stab_pref_fill  = (unsigned)std::stoul(p.str_or_num());
                                else if (k == "stab_pref_edge")  w.stab_pref_edge  = (unsigned)std::stoul(p.str_or_num());
                                else if (k == "stab_paper_cmap") w.stab_paper_cmap = p.boolean();
                                else if (k == "stab_pref_edge_w") w.stab_pref_edge_w = std::max(0, std::stoi(p.str_or_num()));
                                else if (k == "colormap_idx")    w.colormap_idx    = std::stoi(p.str_or_num());
                                else if (k == "error_source")    w.error_source    = std::stoi(p.str_or_num());
                                else if (k == "time_unit")       w.time_unit       = std::stoi(p.str_or_num());
                                else if (k == "perf_cost")       w.perf_cost       = std::stoi(p.str_or_num());
                                else if (k == "show_min")        w.show_min        = p.boolean();
                                else if (k == "show_avg")        w.show_avg        = p.boolean();
                                else if (k == "show_max")        w.show_max        = p.boolean();
                                else if (k == "custom_line_style") w.custom_line_style = p.boolean();
                                else if (k == "line_width")      w.line_width      = std::stof(p.str_or_num());
                                else if (k == "line_alpha")      w.line_alpha      = std::stof(p.str_or_num());
                                else if (k == "points_mode")     w.points_mode     = p.boolean();
                                else if (k == "point_marker")    w.point_marker    = std::stoi(p.str_or_num());
                                else if (k == "point_size")      w.point_size      = std::stof(p.str_or_num());
                                else if (k == "node_markers")    w.node_markers    = p.boolean();
                                else if (k == "vary_markers")    w.vary_markers    = p.boolean();
                                else if (k == "palette")         w.palette         = std::stoi(p.str_or_num());
                                else if (k == "members") {
                                    w.members.clear();
                                    p.expect('[');
                                    if (!p.opt(']')) {
                                        while (true) {
                                            w.members.push_back(std::stoi(p.str_or_num()));
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
                        wins.push_back(std::move(w));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        return true;
    } catch (...) {
        wins.clear();
        return false;
    }
}
