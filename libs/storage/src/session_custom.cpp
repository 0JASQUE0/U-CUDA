#include "session_json_internal.h"

using namespace ucuda::storage_detail;

 // namespace

std::string session_to_json_custom(const CustomSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"shared\":";  write_shared_config(o, s.shared);            o << ",\n";
    o << "  \"bif\":";     o << session_to_json_parametric(s.bif_session); o << ",\n";
    o << "  \"lle\":";     o << session_to_json_lle(s.lle_session);        o << ",\n";
    o << "  \"ls\":";      o << session_to_json_ls(s.ls_session);          o << ",\n";
    o << "  \"phase\":";   o << session_to_json(s.phase_session);          o << ",\n";
    o << "  \"basins\":";  o << session_to_json_basins(s.basins_session);   o << ",\n";
    o << "  \"workspace\":"; write_workspace(o, s.workspace);
    o << "\n}\n";
    return o.str();
}


bool session_from_json_custom(const std::string& json, CustomSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            p.ws();
            size_t val_start = p.i;
            p.skip_value();
            size_t val_end = p.i;
            std::string raw = json.substr(val_start, val_end - val_start);

            if (key == "shared") {
                JP q(raw); auto& c = s.shared;
                q.expect('{');
                if (!q.opt('}')) {
                    while (true) {
                        std::string k = q.str(); q.expect(':');
                        if      (k == "level_2d_enabled")        c.level_2d_enabled = q.boolean();
                        else if (k == "level_1d_enabled")        c.level_1d_enabled = q.boolean();
                        else if (k == "level_phase_enabled")     c.level_phase_enabled = q.boolean();
                        else if (k == "scheme")                  c.scheme = q.str();
                        else if (k == "symmetry_s")              c.symmetry_s = q.str();
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
                        else if (k == "axis_x_over_h")           c.axis_x_over_h   = q.boolean();
                        else if (k == "axis_x_var_index")        c.axis_x_var_index = std::stoi(q.str_or_num());
                        else if (k == "axis_x_lo_text")          c.axis_x_lo_text = q.str();
                        else if (k == "axis_x_hi_text")          c.axis_x_hi_text = q.str();
                        else if (k == "axis_x_log")              c.axis_x_log     = q.boolean();
                        else if (k == "axis_y_par_index")        c.axis_y_par_index = std::stoi(q.str_or_num());
                        else if (k == "axis_y_over_var")         c.axis_y_over_var = q.boolean();
                        else if (k == "axis_y_over_h")           c.axis_y_over_h   = q.boolean();
                        else if (k == "axis_y_var_index")        c.axis_y_var_index = std::stoi(q.str_or_num());
                        else if (k == "axis_y_lo_text")          c.axis_y_lo_text = q.str();
                        else if (k == "axis_y_hi_text")          c.axis_y_hi_text = q.str();
                        else if (k == "axis_y_log")              c.axis_y_log     = q.boolean();
                        else if (k == "bif_writable_var")        c.bif_writable_var = std::stoi(q.str_or_num());
                        else if (k == "resolution_text")         c.resolution_text = q.str();
                        // Backwards-compat: sessions saved before N_x/N_y were
                        // merged into a single Resolution field. Keep n_x_text's
                        // value (it was the drive-both-axes source anyway);
                        // silently discard n_y_text.
                        else if (k == "n_x_text")                c.resolution_text = q.str();
                        else if (k == "n_y_text")                (void)q.str();
                        else if (k == "bif2d_enabled")           c.bif2d_enabled = q.boolean();
                        else if (k == "lle2d_enabled")           c.lle2d_enabled = q.boolean();
                        else if (k == "ls2d_enabled")            c.ls2d_enabled = q.boolean();
                        else if (k == "inherit_sweep_from_2d")   c.inherit_sweep_from_2d = q.boolean();
                        else if (k == "sweep_x_par_index")       c.sweep_x_par_index = std::stoi(q.str_or_num());
                        else if (k == "sweep_x_over_var")        c.sweep_x_over_var = q.boolean();
                        else if (k == "sweep_x_over_h")          c.sweep_x_over_h  = q.boolean();
                        else if (k == "sweep_x_var_index")       c.sweep_x_var_index = std::stoi(q.str_or_num());
                        else if (k == "sweep_x_lo_text")         c.sweep_x_lo_text = q.str();
                        else if (k == "sweep_x_hi_text")         c.sweep_x_hi_text = q.str();
                        else if (k == "sweep_x_log")             c.sweep_x_log     = q.boolean();
                        else if (k == "n_x_1d_text")             c.n_x_1d_text = q.str();
                        else if (k == "sweep_y_par_index")       c.sweep_y_par_index = std::stoi(q.str_or_num());
                        else if (k == "sweep_y_over_var")        c.sweep_y_over_var = q.boolean();
                        else if (k == "sweep_y_over_h")          c.sweep_y_over_h  = q.boolean();
                        else if (k == "sweep_y_var_index")       c.sweep_y_var_index = std::stoi(q.str_or_num());
                        else if (k == "sweep_y_lo_text")         c.sweep_y_lo_text = q.str();
                        else if (k == "sweep_y_hi_text")         c.sweep_y_hi_text = q.str();
                        else if (k == "sweep_y_log")             c.sweep_y_log     = q.boolean();
                        else if (k == "n_y_1d_text")             c.n_y_1d_text = q.str();
                        else if (k == "l1d_h_text")              c.l1d_h_text = q.str();
                        else if (k == "l1d_transient_text")      c.l1d_transient_text = q.str();
                        else if (k == "l1d_t_max_text")          c.l1d_t_max_text = q.str();
                        else if (k == "bif1d_x_enabled")         c.bif1d_x_enabled = q.boolean();
                        else if (k == "bif1d_y_enabled")         c.bif1d_y_enabled = q.boolean();
                        else if (k == "lle1d_x_enabled")         c.lle1d_x_enabled = q.boolean();
                        else if (k == "lle1d_y_enabled")         c.lle1d_y_enabled = q.boolean();
                        else if (k == "ls1d_x_enabled")          c.ls1d_x_enabled = q.boolean();
                        else if (k == "ls1d_y_enabled")          c.ls1d_y_enabled = q.boolean();
                        else if (k == "continuation_1d_enabled") c.continuation_1d_enabled = q.boolean();
                        else if (k == "plot_inter_peaks_1d")     c.plot_inter_peaks_1d = q.boolean();
                        else if (k == "fix_x_value")             c.fix_x_value = std::stod(q.str_or_num());
                        else if (k == "fix_y_value")             c.fix_y_value = std::stod(q.str_or_num());
                        else if (k == "auto_recompute_1d")       c.auto_recompute_1d = q.boolean();
                        else if (k == "level3_kind")             c.level3_kind = std::stoi(q.str_or_num());
                        else if (k == "autorun_on_drilldown")    c.autorun_on_drilldown = q.boolean();
                        else                                     q.skip_value();
                        if (q.opt(',')) continue;
                        q.expect('}'); break;
                    }
                }
            }
            else if (key == "bif")    session_from_json_parametric(raw, s.bif_session);
            else if (key == "lle")    session_from_json_lle(raw, s.lle_session);
            else if (key == "ls")     session_from_json_ls(raw, s.ls_session);
            else if (key == "phase")  session_from_json(raw, s.phase_session);
            else if (key == "basins") session_from_json_basins(raw, s.basins_session);
            else if (key == "workspace") {
                JP q(raw); auto& ws = s.workspace;
                ws.tabs.clear();
                q.expect('{');
                if (!q.opt('}')) {
                    while (true) {
                        std::string k = q.str(); q.expect(':');
                        if      (k == "active_tab_id") ws.active_tab_id = std::stoi(q.str_or_num());
                        else if (k == "next_tab_id")   ws.next_tab_id   = std::stoi(q.str_or_num());
                        else if (k == "controls_width") ws.controls_width = (float)std::stod(q.str_or_num());
                        else if (k == "tabs") {
                            q.expect('[');
                            if (!q.opt(']')) {
                                while (true) {
                                    q.expect('{');
                                    WorkspaceTab t;
                                    while (true) {
                                        std::string tk = q.str(); q.expect(':');
                                        if      (tk == "id")   t.id   = std::stoi(q.str_or_num());
                                        else if (tk == "name") t.name = q.str();
                                        else q.skip_value();
                                        if (q.opt(',')) continue;
                                        q.expect('}'); break;
                                    }
                                    ws.tabs.push_back(std::move(t));
                                    if (q.opt(',')) continue;
                                    q.expect(']'); break;
                                }
                            }
                        }
                        else q.skip_value();
                        if (q.opt(',')) continue;
                        q.expect('}'); break;
                    }
                }
                for (auto& t : ws.tabs) if (t.id >= ws.next_tab_id) ws.next_tab_id = t.id + 1;
                if (ws.tabs.empty()) {
                    ws.tabs.push_back({1, "Tab 1"});
                    ws.active_tab_id = 1;
                    if (ws.next_tab_id <= 1) ws.next_tab_id = 2;
                }
            }

            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        return true;
    }
    catch (...) {
        return false;
    }
}
