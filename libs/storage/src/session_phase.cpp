#include "session_json_internal.h"

using namespace ucuda::storage_detail;

 // namespace

std::string session_to_json(const PhaseAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"step_h\":"; jstr(o, s.step_h); o << ",\n";
    o << "  \"sim_time\":"; jstr(o, s.sim_time); o << ",\n";
    o << "  \"skip_time\":"; jstr(o, s.skip_time); o << ",\n";
    o << "  \"scheme\":"; jstr(o, s.scheme); o << ",\n";
    o << "  \"symmetry_s\":"; jstr(o, s.symmetry_s); o << ",\n";
    o << "  \"decimation\":"; jstr(o, s.decimation); o << ",\n";
    write_adaptive(o, s.adaptive, "  ", ",\n");
    o << "  \"auto_recompute\":" << (s.auto_recompute ? "true" : "false") << ",\n";
    o << "  \"legend_show_ic\":" << (s.legend_show_ic ? "true" : "false") << ",\n";
    o << "  \"use_gpu\":" << (s.use_gpu ? "true" : "false") << ",\n";
    // Continuation: only persist user-facing toggles; runtime seed/timer are
    // rebuilt from scratch on Start so a load never resurrects a running loop.
    o << "  \"continuation_mode\":" << (s.continuation_mode ? "true" : "false") << ",\n";
    o << "  \"continuation_delay_ms\":"; jstr(o, s.continuation_delay_ms); o << ",\n";
    o << "  \"continuation_peaks_cap\":" << s.continuation_peaks_cap << ",\n";
    o << "  \"param_values\":"; jmap(o, s.param_values); o << ",\n";
    // ic_sets: [ {label, visible, values{...}} ]
    o << "  \"ic_sets\":[";
    for (size_t k = 0; k < s.ic_sets.size(); ++k) {
        if (k)o << ","; const auto& ic = s.ic_sets[k];
        o << "{\"label\":"; jstr(o, ic.label);
        o << ",\"visible\":" << (ic.visible ? "true" : "false");
        o << ",\"values\":"; jmap(o, ic.values); o << "}";
    }
    o << "],\n";
    // projections: [ {label,type,ax,ay,az,show_var[...]} ]
    o << "  \"projections\":[";
    for (size_t k = 0; k < s.projections.size(); ++k) {
        if (k)o << ","; const auto& p = s.projections[k];
        o << "{\"label\":"; jstr(o, p.label);
        o << ",\"type\":" << (int)p.type;
        o << ",\"ax\":" << p.axis_x << ",\"ay\":" << p.axis_y << ",\"az\":" << p.axis_z;
        o << ",\"cls\":" << (p.custom_line_style ? "true" : "false");
        o << ",\"lw\":" << p.line_width << ",\"al\":" << p.alpha;
        o << ",\"pts\":" << (p.draw_points ? "true" : "false") << ",\"ps\":" << p.point_size;
        if (p.y_log) o << ",\"ylog\":true";
        if (p.point_alpha != 1.0f) o << ",\"pa\":" << p.point_alpha;
        write_rqa_fields(o, p);
        o << ",\"show_var\":[";
        for (size_t v = 0; v < p.show_var.size(); ++v) { if (v)o << ","; o << (p.show_var[v] ? "true" : "false"); }
        o << "]}";
    }
    o << "],\n";
    o << "  \"layout\":\"\"\n"; // зарезервировано
    o << "}\n";
    return o.str();
}


bool session_from_json(const std::string& json, PhaseAnalysisSession& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "step_h")     s.step_h = p.str();
            else if (key == "sim_time")   s.sim_time = p.str();
            else if (key == "skip_time")  s.skip_time = p.str();
            else if (key == "scheme")     s.scheme = p.str();
            else if (key == "symmetry_s") s.symmetry_s = p.str();
            else if (key == "decimation") s.decimation = p.str();
            else if (key == "adaptive")   read_adaptive(p, s.adaptive);
            else if (key == "auto_recompute") s.auto_recompute = p.boolean();
            else if (key == "legend_show_ic") s.legend_show_ic = p.boolean();
            else if (key == "use_gpu")        s.use_gpu = p.boolean();
            else if (key == "continuation_mode")     s.continuation_mode = p.boolean();
            else if (key == "continuation_delay_ms") s.continuation_delay_ms = p.str();
            else if (key == "continuation_peaks_cap") {
                try { s.continuation_peaks_cap = std::stoi(p.str_or_num()); } catch (...) {}
            }
            else if (key == "param_values")   s.param_values = p.map_ss();
            else if (key == "ic_sets") {
                s.ic_sets.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        InitialConditionSet ic;
                        while (true) {
                            std::string k = p.str(); p.expect(':');
                            if (k == "label")   ic.label = p.str();
                            else if (k == "visible") ic.visible = p.boolean();
                            else if (k == "values")  ic.values = p.map_ss();
                            else p.skip_value();
                            if (p.opt(',')) continue;
                            p.expect('}'); break;
                        }
                        s.ic_sets.push_back(std::move(ic));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "projections") {
                s.projections.clear();
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        Projection pr;
                        while (true) {
                            std::string k = p.str(); p.expect(':');
                            if (k == "label") pr.label = p.str();
                            else if (k == "type") { pr.type = (ProjType)std::stoi(p.str_or_num()); }
                            else if (k == "ax")    pr.axis_x = std::stoi(p.str_or_num());
                            else if (k == "ay")    pr.axis_y = std::stoi(p.str_or_num());
                            else if (k == "az")    pr.axis_z = std::stoi(p.str_or_num());
                            else if (k == "cls")   pr.custom_line_style = p.boolean();
                            else if (k == "lw")    pr.line_width = (float)std::stod(p.str_or_num());
                            else if (k == "al")    pr.alpha      = (float)std::stod(p.str_or_num());
                        else if (k == "pts")   pr.draw_points = p.boolean();
                        else if (k == "ps")    pr.point_size = (float)std::stod(p.str_or_num());
                            else if (k == "pts")   pr.draw_points = p.boolean();
                            else if (k == "ps")    pr.point_size = (float)std::stod(p.str_or_num());
                            else if (k == "ylog")  pr.y_log = p.boolean();
                            else if (k == "pa")    pr.point_alpha = (float)std::stod(p.str_or_num());
                            else if (k == "show_var") {
                                pr.show_var.clear();
                                p.expect('[');
                                if (!p.opt(']')) { while (true) { pr.show_var.push_back(p.boolean()); if (p.opt(','))continue; p.expect(']'); break; } }
                            }
                            else if (read_rqa_field(k, p, pr)) {}
                            else p.skip_value();
                            if (p.opt(',')) continue;
                            p.expect('}'); break;
                        }
                        s.projections.push_back(std::move(pr));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        // КРС зависит от scheme: после загрузки сессии перегенерируем, иначе
        // GPU-путь считал бы по старому методу (combo показывает новый scheme,
        // а krs_code остался прежним).
        s.krs_code.clear(); // Regenerated by the analysis layer when it snapshots a run.
        return true;
    }
    catch (...) {
        return false;
    }
}
