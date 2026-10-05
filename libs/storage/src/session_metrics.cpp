#include "session_json_internal.h"
namespace ucuda::storage_detail {
void write_metrics_config(std::ostringstream& o, const SignalMetricsConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);
    o << ",\"label_is_manual\":"  << (c.label_is_manual ? "true" : "false");
    o << ",\"scheme\":";          jstr(o, c.scheme);
    o << ",\"symmetry_s\":";      jstr(o, c.symmetry_s);
    o << ",\"param_index\":"      << c.param_index;
    o << ",\"sweep_over_var\":"   << (c.sweep_over_var ? "true" : "false");
    o << ",\"sweep_over_h\":"     << (c.sweep_over_h ? "true" : "false");
    o << ",\"log_scale\":"        << (c.log_scale ? "true" : "false");
    o << ",\"var_sweep_index\":"  << c.var_sweep_index;
    o << ",\"param_lo_text\":";   jstr(o, c.param_lo_text);
    o << ",\"param_hi_text\":";   jstr(o, c.param_hi_text);
    o << ",\"n_pts_text\":";      jstr(o, c.n_pts_text);
    o << ",\"continuation\":"         << (c.continuation ? "true" : "false");
    o << ",\"continuation_reverse\":" << (c.continuation_reverse ? "true" : "false");
    o << ",\"use_gpu\":"              << (c.use_gpu ? "true" : "false");
    o << ",\"writable_var\":"     << c.writable_var;
    o << ",\"h_text\":";          jstr(o, c.h_text);
    o << ",\"t_max_text\":";      jstr(o, c.t_max_text);
    o << ",\"transient_text\":";  jstr(o, c.transient_text);
    o << ",\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text);
    o << ",\"max_value_text\":";  jstr(o, c.max_value_text);
    write_adaptive(o, c.adaptive, ",", "");
    if (c.ad_axis)   o << ",\"ad_axis\":"   << c.ad_axis;
    if (c.ad_axis_2) o << ",\"ad_axis_2\":" << c.ad_axis_2;
    // Пишем ВЫКЛЮЧЕННЫЕ метрики: метрика, добавленная в будущей версии, в
    // старом файле не упомянута и потому загрузится включённой, как в новом конфиге.
    o << ",\"metric_mask_off\":"  << (kSignalMetricAllMask & ~c.metric_mask);
    o << ",\"param_values\":";    jmap(o, c.param_values);
    o << ",\"initial_conditions\":"; jmap(o, c.initial_conditions);
    o << ",\"csv_save_enabled\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, c.csv_output_path);
    o << ",\"mode_2d\":"           << (c.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"     << c.param_index_2;
    o << ",\"sweep_over_var_2\":"  << (c.sweep_over_var_2 ? "true" : "false");
    o << ",\"sweep_over_h_2\":"    << (c.sweep_over_h_2 ? "true" : "false");
    o << ",\"log_scale_2\":"       << (c.log_scale_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":" << c.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";  jstr(o, c.param_lo_2_text);
    o << ",\"param_hi_2_text\":";  jstr(o, c.param_hi_2_text);
    o << ",\"colormap_idx\":"      << c.colormap_idx;
    o << "}";
}


bool read_metrics_config_field(JP& p, SignalMetricsConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "label_is_manual")    c.label_is_manual   = p.boolean();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "param_index")        c.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     c.sweep_over_var    = p.boolean();
    else if (key == "sweep_over_h")       c.sweep_over_h      = p.boolean();
    else if (key == "log_scale")          c.log_scale         = p.boolean();
    else if (key == "var_sweep_index")    c.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "param_lo_text")      c.param_lo_text     = p.str();
    else if (key == "param_hi_text")      c.param_hi_text     = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "writable_var")       c.writable_var      = std::stoi(p.str_or_num());
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "pre_scaller_text")   c.pre_scaller_text  = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "adaptive")           { read_adaptive(p, c.adaptive);
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis")            { try { c.ad_axis   = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis_2")          { try { c.ad_axis_2 = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "continuation")       c.continuation      = p.boolean();
    else if (key == "continuation_reverse") c.continuation_reverse = p.boolean();
    else if (key == "use_gpu")            c.use_gpu           = p.boolean();
    else if (key == "metric_mask_off")    c.metric_mask       = kSignalMetricAllMask & ~std::stoi(p.str_or_num());
    // Первый формат — маска включённых при 9 метриках (биты 0..8). Всё, что
    // появилось позже, он выразить не мог — такие метрики включаем.
    else if (key == "metric_mask")        c.metric_mask       = (std::stoi(p.str_or_num()) & 0x1FF)
                                                              | (kSignalMetricAllMask & ~0x1FF);
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    else if (key == "mode_2d")            c.mode_2d           = p.boolean();
    else if (key == "param_index_2")      c.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   c.sweep_over_var_2  = p.boolean();
    else if (key == "sweep_over_h_2")     c.sweep_over_h_2    = p.boolean();
    else if (key == "log_scale_2")        c.log_scale_2       = p.boolean();
    else if (key == "var_sweep_index_2")  c.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    c.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    c.param_hi_2_text   = p.str();
    else if (key == "colormap_idx")       c.colormap_idx      = std::stoi(p.str_or_num());
    else return false;
    return true;
}


// DFT1D plot windows: same flat-array shape as parametric windows, minus
// kind/mode_2d/colored_1d (DFT1D has only one display kind).
// OrderAnalysisSession — `_last_order.json`. Multi-config layout как у
// DFT1D/Basins/FastSync. Настройки ОТОБРАЖЕНИЯ здесь не пишутся: они живут на
// окнах графика (`_last_order_windows.json`), а не на конфиге. Result, как
// везде, не сохраняется.
}
using namespace ucuda::storage_detail;
std::string session_to_json_metrics(const SignalMetricsAnalysisSession& s) {
    std::ostringstream o;
    o << "{\n";
    o << "  \"active_config_index\":" << s.active_config_index << ",\n";
    o << "  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_metrics_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n";
    o << "}\n";
    return o.str();
}


bool session_from_json_metrics(const std::string& json, SignalMetricsAnalysisSession& s) {
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
                        SignalMetricsConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_metrics_config_field(p, c, k2)) p.skip_value();
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
            else if (key == "active_config_index") {
                s.active_config_index = std::stoi(p.str_or_num());
            }
            else {
                p.skip_value();
            }
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (s.configs.empty()) s.add_config();
        if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
            s.active_config_index = 0;
        s.running_config_index = -1;
        return true;
    }
    catch (...) {
        return false;
    }
}
