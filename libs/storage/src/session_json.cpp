#include "session_json_internal.h"

namespace ucuda::storage_detail {






// сериализация


    std::string esc(const std::string& s) {
        std::string o; o.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
            }
        }
        return o;
    }


    void jstr(std::ostringstream& o, const std::string& v) { o << '"' << esc(v) << '"'; }


    void jmap(std::ostringstream& o, const std::map<std::string, std::string>& m) {
        o << "{"; bool first = true;
        for (auto& p : m) { if (!first)o << ","; o << '"' << esc(p.first) << "\":\"" << esc(p.second) << '"'; first = false; }
        o << "}";
    }



void write_adaptive(std::ostringstream& o, const adaptive::Options& a) {
    o << std::setprecision(17) << "{";
    o << "\"method\":" << a.method;
    o << ",\"order\":" << a.order;
    o << ",\"rtol\":" << a.rtol;
    o << ",\"atol\":" << a.atol;
    o << ",\"h_initial\":" << a.h_initial;
    o << ",\"h_min\":" << a.h_min;
    o << ",\"h_max\":" << a.h_max;
    o << ",\"max_steps\":" << a.max_steps;
    o << ",\"max_rejects\":" << a.max_rejects;
    o << "}";
}

void read_adaptive(JP& p, adaptive::Options& a) {
    a = {}; p.expect('{'); if(p.opt('}')) return;
    while(true) { std::string key=p.str(); p.expect(':');
        if(key=="method") a.method=std::stoi(p.str_or_num());
        else if(key=="order") a.order=std::stoi(p.str_or_num());
        else if(key=="rtol") a.rtol=std::stod(p.str_or_num());
        else if(key=="atol") a.atol=std::stod(p.str_or_num());
        else if(key=="h_initial") a.h_initial=std::stod(p.str_or_num());
        else if(key=="h_min") a.h_min=std::stod(p.str_or_num());
        else if(key=="h_max") a.h_max=std::stod(p.str_or_num());
        else if(key=="max_steps") a.max_steps=std::stoi(p.str_or_num());
        else if(key=="max_rejects") a.max_rejects=std::stoi(p.str_or_num());
        else p.skip_value();
        if(p.opt(',')) continue; p.expect('}'); break;
    }
}


// ---- RQA-поля проекции ----
// Пишутся ТОЛЬКО когда источник реально выбран (rqa_source != None). Иначе файлы всех
// существующих сессий распухли бы на 17 ключей в каждой проекции ради настроек, которых нет.
// Читателю ключи не обязательны: неизвестные глотает skip_value, отсутствующие остаются на
// дефолтах Projection — поэтому старые сессии грузятся без единой правки.
void write_rqa_fields(std::ostringstream& o, const Projection& p) {
    if (p.rqa_source == rqa::Source::None) return;
    o << ",\"rq_src\":" << (int)p.rqa_source
      << ",\"rq_ic\":" << p.rqa_ic
      << ",\"rq_var\":" << p.rqa_var
      << ",\"rq_norm\":" << (int)p.rqa_norm
      << ",\"rq_em\":" << (int)p.rqa_eps_mode
      << ",\"rq_net\":" << (p.rqa_network ? "true" : "false")
      << ",\"rq_cont\":" << (p.rqa_in_continuation ? "true" : "false")
      << ",\"rq_vm\":" << p.rqa_view_mode
      << ",\"rq_cmap\":" << p.rqa_colormap
      << ",\"rq_rev\":" << (p.rqa_reverse_cmap ? "true" : "false")
      << ",\"rq_as\":" << (p.rqa_autoscale ? "true" : "false");
    o << ",\"rq_cbmin\":"; jstr(o, p.rqa_cbar_vmin_text);
    o << ",\"rq_cbmax\":"; jstr(o, p.rqa_cbar_vmax_text);
    o << ",\"rq_m\":"; jstr(o, p.rqa_m_text);
    o << ",\"rq_tau\":"; jstr(o, p.rqa_tau_text);
    o << ",\"rq_eps\":"; jstr(o, p.rqa_eps_text);
    o << ",\"rq_ef\":"; jstr(o, p.rqa_eps_frac_text);
    o << ",\"rq_rr\":"; jstr(o, p.rqa_rr_text);
    o << ",\"rq_th\":"; jstr(o, p.rqa_theiler_text);
    o << ",\"rq_lmin\":"; jstr(o, p.rqa_lmin_text);
    o << ",\"rq_vmin\":"; jstr(o, p.rqa_vmin_text);
    o << ",\"rq_pts\":"; jstr(o, p.rqa_points_text);
}


// true, если ключ был RQA-шный и уже разобран.
bool read_rqa_field(const std::string& k, JP& p, Projection& pr) {
    if (k == "rq_src")       pr.rqa_source   = (rqa::Source)std::stoi(p.str_or_num());
    else if (k == "rq_ic")   pr.rqa_ic       = std::stoi(p.str_or_num());
    else if (k == "rq_var")  pr.rqa_var      = std::stoi(p.str_or_num());
    else if (k == "rq_norm") pr.rqa_norm     = (rqa::Norm)std::stoi(p.str_or_num());
    else if (k == "rq_em")   pr.rqa_eps_mode = (rqa::EpsMode)std::stoi(p.str_or_num());
    else if (k == "rq_net")  pr.rqa_network  = p.boolean();
    else if (k == "rq_cont") pr.rqa_in_continuation = p.boolean();
    else if (k == "rq_vm")   pr.rqa_view_mode = std::stoi(p.str_or_num());
    else if (k == "rq_cmap") pr.rqa_colormap  = std::stoi(p.str_or_num());
    else if (k == "rq_rev")  pr.rqa_reverse_cmap = p.boolean();
    else if (k == "rq_as")   pr.rqa_autoscale = p.boolean();
    else if (k == "rq_cbmin") pr.rqa_cbar_vmin_text = p.str();
    else if (k == "rq_cbmax") pr.rqa_cbar_vmax_text = p.str();
    else if (k == "rq_m")    pr.rqa_m_text    = p.str();
    else if (k == "rq_tau")  pr.rqa_tau_text  = p.str();
    else if (k == "rq_eps")  pr.rqa_eps_text  = p.str();
    else if (k == "rq_ef")   pr.rqa_eps_frac_text = p.str();
    else if (k == "rq_rr")   pr.rqa_rr_text   = p.str();
    else if (k == "rq_th")   pr.rqa_theiler_text = p.str();
    else if (k == "rq_lmin") pr.rqa_lmin_text = p.str();
    else if (k == "rq_vmin") pr.rqa_vmin_text = p.str();
    else if (k == "rq_pts")  pr.rqa_points_text = p.str();
    else return false;
    return true;
}

// BifurcationAnalysisSession

// Сериализует одну БД в JSON-объект (без обёртки фигурными). Используется
// внутри массива "diagrams".
void write_diagram(std::ostringstream& o, const BifurcationDiagramConfig& bd) {
    o << "{";
    o << "\"label\":";            jstr(o, bd.label);
    o << ",\"label_is_manual\":"  << (bd.label_is_manual ? "true" : "false");
    o << ",\"adaptive\":"; write_adaptive(o,bd.adaptive);
    o << ",\"scheme\":";          jstr(o, bd.scheme);
    o << ",\"symmetry_s\":";      jstr(o, bd.symmetry_s);
    o << ",\"param_index\":"      << bd.param_index;
    o << ",\"sweep_over_var\":"   << (bd.sweep_over_var ? "true" : "false");
    o << ",\"sweep_over_h\":"     << (bd.sweep_over_h ? "true" : "false");
    o << ",\"log_scale\":"        << (bd.log_scale ? "true" : "false");
    o << ",\"var_sweep_index\":"  << bd.var_sweep_index;
    o << ",\"continuation\":"     << (bd.continuation ? "true" : "false");
    o << ",\"continuation_reverse\":" << (bd.continuation_reverse ? "true" : "false");
    o << ",\"use_gpu\":"          << (bd.use_gpu ? "true" : "false");
    o << ",\"param_lo_text\":";   jstr(o, bd.param_lo_text);
    o << ",\"param_hi_text\":";   jstr(o, bd.param_hi_text);
    o << ",\"n_pts_text\":";      jstr(o, bd.n_pts_text);
    o << ",\"writable_var\":"     << bd.writable_var;
    o << ",\"h_text\":";          jstr(o, bd.h_text);
    o << ",\"t_max_text\":";      jstr(o, bd.t_max_text);
    o << ",\"transient_text\":";  jstr(o, bd.transient_text);
    o << ",\"pre_scaller_text\":";jstr(o, bd.pre_scaller_text);
    o << ",\"max_value_text\":";  jstr(o, bd.max_value_text);
    o << ",\"param_values\":";    jmap(o, bd.param_values);
    o << ",\"initial_conditions\":"; jmap(o, bd.initial_conditions);
    o << ",\"csv_save_enabled\":" << (bd.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, bd.csv_output_path);
    o << ",\"plot_inter_peaks\":" << (bd.plot_inter_peaks ? "true" : "false");
    // 2D-mode config: result_2d not persisted (too heavy), user runs again.
    o << ",\"mode_2d\":"           << (bd.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"     << bd.param_index_2;
    o << ",\"sweep_over_var_2\":"  << (bd.sweep_over_var_2 ? "true" : "false");
    o << ",\"sweep_over_h_2\":"    << (bd.sweep_over_h_2 ? "true" : "false");
    o << ",\"log_scale_2\":"       << (bd.log_scale_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":" << bd.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";  jstr(o, bd.param_lo_2_text);
    o << ",\"param_hi_2_text\":";  jstr(o, bd.param_hi_2_text);
    o << ",\"eps_dbscan_text\":";  jstr(o, bd.eps_dbscan_text);
    o << ",\"mult_peak_text\":";     jstr(o, bd.mult_peak_text);
    o << ",\"mult_interval_text\":"; jstr(o, bd.mult_interval_text);
    o << ",\"colormap_idx\":"      << bd.colormap_idx;
    o << ",\"colored_1d\":"            << (bd.colored_1d ? "true" : "false");
    o << ",\"colored_1d_b_text\":";    jstr(o, bd.colored_1d_b_text);
    o << ",\"colored_1d_custom_y\":"   << (bd.colored_1d_custom_y ? "true" : "false");
    o << ",\"colored_1d_ymin_text\":"; jstr(o, bd.colored_1d_ymin_text);
    o << ",\"colored_1d_ymax_text\":"; jstr(o, bd.colored_1d_ymax_text);
    o << ",\"colored_1d_log\":"        << (bd.colored_1d_log ? "true" : "false");
    o << ",\"custom_point_style\":"    << (bd.custom_point_style ? "true" : "false");
    o << ",\"point_marker\":"          << bd.point_marker;
    o << ",\"point_size\":"            << bd.point_size;
    o << ",\"point_alpha\":"           << bd.point_alpha;
    o << "}";
}


// Читает одно поле в БД. Возвращает true, если ключ распознан.
// Используется и при разборе нового формата (внутри "diagrams"), и при
// разборе legacy-формата (плоские ключи на верхнем уровне = одна БД).
bool read_diagram_field(JP& p, BifurcationDiagramConfig& bd, const std::string& key) {
    if      (key == "label")              bd.label             = p.str();
    else if (key == "label_is_manual")    bd.label_is_manual   = p.boolean();
    else if (key == "scheme")             bd.scheme            = p.str();
    else if (key == "adaptive") read_adaptive(p,bd.adaptive);
    else if (key == "symmetry_s")         bd.symmetry_s        = p.str();
    else if (key == "param_index")        bd.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     bd.sweep_over_var    = p.boolean();
    else if (key == "sweep_over_h")       bd.sweep_over_h      = p.boolean();
    else if (key == "log_scale")          bd.log_scale         = p.boolean();
    else if (key == "var_sweep_index")    bd.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "continuation")       bd.continuation      = p.boolean();
    else if (key == "continuation_reverse") bd.continuation_reverse = p.boolean();
    else if (key == "use_gpu")            bd.use_gpu           = p.boolean();
    else if (key == "param_lo_text")      bd.param_lo_text     = p.str();
    else if (key == "param_hi_text")      bd.param_hi_text     = p.str();
    else if (key == "n_pts_text")         bd.n_pts_text        = p.str();
    else if (key == "writable_var")       bd.writable_var      = std::stoi(p.str_or_num());
    else if (key == "h_text")             bd.h_text            = p.str();
    else if (key == "t_max_text")         bd.t_max_text        = p.str();
    else if (key == "transient_text")     bd.transient_text    = p.str();
    else if (key == "pre_scaller_text")   bd.pre_scaller_text  = p.str();
    else if (key == "max_value_text")     bd.max_value_text    = p.str();
    else if (key == "param_values")       bd.param_values      = p.map_ss();
    else if (key == "initial_conditions") bd.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   bd.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    bd.csv_output_path   = p.str();
    else if (key == "plot_inter_peaks")   bd.plot_inter_peaks  = p.boolean();
    // 2D-mode: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора.
    else if (key == "mode_2d")            bd.mode_2d           = p.boolean();
    else if (key == "param_index_2")      bd.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   bd.sweep_over_var_2  = p.boolean();
    else if (key == "sweep_over_h_2")     bd.sweep_over_h_2    = p.boolean();
    else if (key == "log_scale_2")        bd.log_scale_2       = p.boolean();
    else if (key == "var_sweep_index_2")  bd.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    bd.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    bd.param_hi_2_text   = p.str();
    else if (key == "eps_dbscan_text")    bd.eps_dbscan_text   = p.str();
    // Отсутствие ключей = дефолты конфига (константы configCUDA.h).
    else if (key == "mult_peak_text")     bd.mult_peak_text     = p.str();
    else if (key == "mult_interval_text") bd.mult_interval_text = p.str();
    else if (key == "colormap_idx")       bd.colormap_idx      = std::stoi(p.str_or_num());
    else if (key == "colored_1d")            bd.colored_1d            = p.boolean();
    else if (key == "colored_1d_b_text")     bd.colored_1d_b_text     = p.str();
    else if (key == "colored_1d_custom_y")   bd.colored_1d_custom_y   = p.boolean();
    else if (key == "colored_1d_ymin_text")  bd.colored_1d_ymin_text  = p.str();
    else if (key == "colored_1d_ymax_text")  bd.colored_1d_ymax_text  = p.str();
    else if (key == "colored_1d_log")        bd.colored_1d_log        = p.boolean();
    else if (key == "custom_point_style")    bd.custom_point_style    = p.boolean();
    else if (key == "point_marker")          bd.point_marker          = std::stoi(p.str_or_num());
    else if (key == "point_size")            bd.point_size            = (float)std::stod(p.str_or_num());
    else if (key == "point_alpha")           bd.point_alpha           = (float)std::stod(p.str_or_num());
    else return false;
    return true;
}


// LLEAnalysisSession — отдельный JSON-файл `_last_lle.json`. Структура та же
// что у parametric (массив объектов), отличаются только поля per-«прогон».



void write_lle_curve(std::ostringstream& o, const LLECurveConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);
    o << ",\"continuation\":"         << (c.continuation ? "true" : "false");
    o << ",\"continuation_reverse\":" << (c.continuation_reverse ? "true" : "false");
    o << ",\"use_gpu\":"              << (c.use_gpu ? "true" : "false");
    o << ",\"label_is_manual\":"  << (c.label_is_manual ? "true" : "false");
    o << ",\"adaptive\":"; write_adaptive(o,c.adaptive);
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
    o << ",\"h_text\":";          jstr(o, c.h_text);
    o << ",\"t_max_text\":";      jstr(o, c.t_max_text);
    o << ",\"transient_text\":";  jstr(o, c.transient_text);
    o << ",\"max_value_text\":";  jstr(o, c.max_value_text);
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    o << ",\"param_values\":";    jmap(o, c.param_values);
    o << ",\"initial_conditions\":"; jmap(o, c.initial_conditions);
    o << ",\"csv_save_enabled\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, c.csv_output_path);
    // 2D-режим (heatmap): persistим только конфиг, result_2d не храним
    // (на 200x200 это ~40k чисел в JSON — слишком тяжело). После load
    // пользователь нажмёт Run заново.
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


bool read_lle_curve_field(JP& p, LLECurveConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "continuation")       c.continuation      = p.boolean();
    else if (key == "continuation_reverse") c.continuation_reverse = p.boolean();
    else if (key == "use_gpu")            c.use_gpu           = p.boolean();
    else if (key == "label_is_manual")    c.label_is_manual   = p.boolean();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "adaptive") read_adaptive(p,c.adaptive);
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "param_index")        c.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     c.sweep_over_var    = p.boolean();
    else if (key == "sweep_over_h")       c.sweep_over_h      = p.boolean();
    else if (key == "log_scale")          c.log_scale         = p.boolean();
    else if (key == "var_sweep_index")    c.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "param_lo_text")      c.param_lo_text     = p.str();
    else if (key == "param_hi_text")      c.param_hi_text     = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    // 2D-режим: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора (mode_2d=false, ...).
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


// LyapunovSpectrumAnalysisSession — `_last_ls.json`. Поля LSCurveConfig
// идентичны LLECurveConfig — копия LLE-сериализатора.



void write_ls_curve(std::ostringstream& o, const LSCurveConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);
    o << ",\"continuation\":"         << (c.continuation ? "true" : "false");
    o << ",\"continuation_reverse\":" << (c.continuation_reverse ? "true" : "false");
    o << ",\"use_gpu\":"              << (c.use_gpu ? "true" : "false");
    o << ",\"label_is_manual\":"  << (c.label_is_manual ? "true" : "false");
    o << ",\"adaptive\":"; write_adaptive(o,c.adaptive);
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
    o << ",\"h_text\":";          jstr(o, c.h_text);
    o << ",\"t_max_text\":";      jstr(o, c.t_max_text);
    o << ",\"transient_text\":";  jstr(o, c.transient_text);
    o << ",\"max_value_text\":";  jstr(o, c.max_value_text);
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    o << ",\"param_values\":";    jmap(o, c.param_values);
    o << ",\"initial_conditions\":"; jmap(o, c.initial_conditions);
    o << ",\"csv_save_enabled\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, c.csv_output_path);
    // 2D-mode config: result_2d not persisted (too heavy), user runs again.
    o << ",\"mode_2d\":"             << (c.mode_2d ? "true" : "false");
    o << ",\"param_index_2\":"       << c.param_index_2;
    o << ",\"sweep_over_var_2\":"    << (c.sweep_over_var_2 ? "true" : "false");
    o << ",\"sweep_over_h_2\":"      << (c.sweep_over_h_2 ? "true" : "false");
    o << ",\"log_scale_2\":"         << (c.log_scale_2 ? "true" : "false");
    o << ",\"var_sweep_index_2\":"   << c.var_sweep_index_2;
    o << ",\"param_lo_2_text\":";    jstr(o, c.param_lo_2_text);
    o << ",\"param_hi_2_text\":";    jstr(o, c.param_hi_2_text);
    o << ",\"display_exponent_idx\":" << c.display_exponent_idx;
    o << ",\"colormap_idx\":"         << c.colormap_idx;
    o << "}";
}


bool read_ls_curve_field(JP& p, LSCurveConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "continuation")       c.continuation      = p.boolean();
    else if (key == "continuation_reverse") c.continuation_reverse = p.boolean();
    else if (key == "use_gpu")            c.use_gpu           = p.boolean();
    else if (key == "label_is_manual")    c.label_is_manual   = p.boolean();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "adaptive") read_adaptive(p,c.adaptive);
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "param_index")        c.param_index       = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")     c.sweep_over_var    = p.boolean();
    else if (key == "sweep_over_h")       c.sweep_over_h      = p.boolean();
    else if (key == "log_scale")          c.log_scale         = p.boolean();
    else if (key == "var_sweep_index")    c.var_sweep_index   = std::stoi(p.str_or_num());
    else if (key == "param_lo_text")      c.param_lo_text     = p.str();
    else if (key == "param_hi_text")      c.param_hi_text     = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    // 2D-mode: backwards-compatible — старые JSON без этих ключей оставят
    // дефолты конструктора.
    else if (key == "mode_2d")            c.mode_2d           = p.boolean();
    else if (key == "param_index_2")      c.param_index_2     = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var_2")   c.sweep_over_var_2  = p.boolean();
    else if (key == "sweep_over_h_2")     c.sweep_over_h_2    = p.boolean();
    else if (key == "log_scale_2")        c.log_scale_2       = p.boolean();
    else if (key == "var_sweep_index_2")  c.var_sweep_index_2 = std::stoi(p.str_or_num());
    else if (key == "param_lo_2_text")    c.param_lo_2_text   = p.str();
    else if (key == "param_hi_2_text")    c.param_hi_2_text   = p.str();
    else if (key == "display_exponent_idx") c.display_exponent_idx = std::stoi(p.str_or_num());
    else if (key == "colormap_idx")         c.colormap_idx          = std::stoi(p.str_or_num());
    else return false;
    return true;
}


// Dft1DAnalysisSession — `_last_dft1d.json`. Multi-config layout как у
// Basins/FastSync (см. read_fastsync_field — тот же "чистый", без legacy
// fallback, шаблон, т.к. это новая фича без старых сохранений). Result не
// сохраняется, как и display_cache* (транзиентный, перестраивается лениво).
void write_dft1d_config(std::ostringstream& o, const Dft1DConfig& c) {
    o << "{";
    o << "\"label\":";              jstr(o, c.label);              o << ",";
    o << "\"label_is_manual\":"     << (c.label_is_manual ? "true" : "false") << ",";
    o << "\"adaptive\":"; write_adaptive(o,c.adaptive); o << ",";
    o << "\"scheme\":";             jstr(o, c.scheme);             o << ",";
    o << "\"symmetry_s\":";         jstr(o, c.symmetry_s);         o << ",";
    o << "\"param_index\":"         << c.param_index              << ",";
    o << "\"sweep_over_var\":"      << (c.sweep_over_var ? "true" : "false") << ",";
    o << "\"var_sweep_index\":"     << c.var_sweep_index          << ",";
    o << "\"continuation\":"        << (c.continuation ? "true" : "false") << ",";
    o << "\"continuation_reverse\":"<< (c.continuation_reverse ? "true" : "false") << ",";
    o << "\"sweep_over_h\":"        << (c.sweep_over_h ? "true" : "false") << ",";
    o << "\"log_scale\":"           << (c.log_scale ? "true" : "false") << ",";
    o << "\"use_gpu\":"             << (c.use_gpu ? "true" : "false") << ",";
    o << "\"param_lo_text\":";      jstr(o, c.param_lo_text);      o << ",";
    o << "\"param_hi_text\":";      jstr(o, c.param_hi_text);      o << ",";
    o << "\"n_pts_text\":";         jstr(o, c.n_pts_text);         o << ",";
    o << "\"writable_var\":"        << c.writable_var             << ",";
    o << "\"n_freq_text\":";        jstr(o, c.n_freq_text);        o << ",";
    o << "\"freq_lo_text\":";       jstr(o, c.freq_lo_text);       o << ",";
    o << "\"freq_hi_text\":";       jstr(o, c.freq_hi_text);       o << ",";
    o << "\"freq_log_scale\":"      << (c.freq_log_scale ? "true" : "false") << ",";
    o << "\"window_type\":"         << c.window_type              << ",";
    o << "\"h_text\":";             jstr(o, c.h_text);             o << ",";
    o << "\"t_max_text\":";         jstr(o, c.t_max_text);         o << ",";
    o << "\"transient_text\":";     jstr(o, c.transient_text);     o << ",";
    o << "\"pre_scaller_text\":";   jstr(o, c.pre_scaller_text);   o << ",";
    o << "\"max_value_text\":";     jstr(o, c.max_value_text);     o << ",";
    o << "\"csv_save_enabled\":"    << (c.csv_save_enabled ? "true" : "false") << ",";
    o << "\"csv_output_path\":";    jstr(o, c.csv_output_path);    o << ",";
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";
    o << "\"param_values\":";       jmap(o, c.param_values);       o << ",";
    o << "\"display_mode\":"       << c.display_mode             << ",";
    o << "\"normalize\":"          << (c.normalize ? "true" : "false") << ",";
    o << "\"db_scale\":"           << (c.db_scale ? "true" : "false") << ",";
    o << "\"db_floor_text\":";      jstr(o, c.db_floor_text);      o << ",";
    o << "\"colormap_idx\":"       << c.colormap_idx;
    o << "}";
}


bool read_dft1d_field(JP& p, Dft1DConfig& c, const std::string& key) {
    if      (key == "label")               c.label               = p.str();
    else if (key == "label_is_manual")     c.label_is_manual     = p.boolean();
    else if (key == "scheme")              c.scheme              = p.str();
    else if (key == "adaptive") read_adaptive(p,c.adaptive);
    else if (key == "symmetry_s")          c.symmetry_s          = p.str();
    else if (key == "param_index")         c.param_index         = std::stoi(p.str_or_num());
    else if (key == "sweep_over_var")      c.sweep_over_var      = p.boolean();
    else if (key == "var_sweep_index")     c.var_sweep_index     = std::stoi(p.str_or_num());
    else if (key == "continuation")        c.continuation        = p.boolean();
    else if (key == "sweep_over_h")        c.sweep_over_h        = p.boolean();
    else if (key == "log_scale")           c.log_scale           = p.boolean();
    else if (key == "use_gpu")             c.use_gpu             = p.boolean();
    else if (key == "continuation_reverse")c.continuation_reverse= p.boolean();
    else if (key == "param_lo_text")       c.param_lo_text       = p.str();
    else if (key == "param_hi_text")       c.param_hi_text       = p.str();
    else if (key == "n_pts_text")          c.n_pts_text          = p.str();
    else if (key == "writable_var")        c.writable_var        = std::stoi(p.str_or_num());
    else if (key == "n_freq_text")         c.n_freq_text         = p.str();
    else if (key == "freq_lo_text")        c.freq_lo_text        = p.str();
    else if (key == "freq_hi_text")        c.freq_hi_text        = p.str();
    else if (key == "freq_log_scale")      c.freq_log_scale      = p.boolean();
    else if (key == "window_type") {
        // Клампим: сессия могла прийти из сборки с другим набором окон, а
        // ImGui::Combo с индексом вне списка рисует пустую строку.
        const int wt = std::stoi(p.str_or_num());
        c.window_type = (wt >= 0 && wt <= 4) ? wt : 1;
    }
    else if (key == "h_text")              c.h_text              = p.str();
    else if (key == "t_max_text")          c.t_max_text          = p.str();
    else if (key == "transient_text")      c.transient_text      = p.str();
    else if (key == "pre_scaller_text")    c.pre_scaller_text    = p.str();
    else if (key == "max_value_text")      c.max_value_text      = p.str();
    else if (key == "csv_save_enabled")    c.csv_save_enabled    = p.boolean();
    else if (key == "csv_output_path")     c.csv_output_path     = p.str();
    else if (key == "initial_conditions")  c.initial_conditions  = p.map_ss();
    else if (key == "param_values")        c.param_values        = p.map_ss();
    else if (key == "display_mode")        c.display_mode        = std::stoi(p.str_or_num());
    else if (key == "normalize")           c.normalize           = p.boolean();
    else if (key == "db_scale")            c.db_scale            = p.boolean();
    else if (key == "db_floor_text")       c.db_floor_text       = p.str();
    else if (key == "colormap_idx")        c.colormap_idx        = std::stoi(p.str_or_num());
    else return false;
    return true;
}


// BasinsAnalysisSession — `_last_basins.json`. Один config на сессию
// (без curves-vector). Result не сохраняется.

// Запись одного BasinsConfig в JSON (без внешних { } — пишет голый объект), используется внутри
// массива "configs". Проекции фазовых портретов по бассейнам: формат тот же, что у "projections" в
// session_to_json(PhaseAnalysisSession); дублируется, а не шарится, чтобы не трогать рабочий
// сериализатор Phase-сессии.
void write_basins_phase_projections(std::ostringstream& o,
                                           const std::vector<Projection>& projs) {
    o << "[";
    for (size_t k = 0; k < projs.size(); ++k) {
        if (k) o << ",";
        const Projection& p = projs[k];
        o << "{\"label\":"; jstr(o, p.label);
        o << ",\"type\":" << (int)p.type;
        o << ",\"ax\":" << p.axis_x << ",\"ay\":" << p.axis_y << ",\"az\":" << p.axis_z;
        o << ",\"cls\":" << (p.custom_line_style ? "true" : "false");
        o << ",\"lw\":" << p.line_width << ",\"al\":" << p.alpha;
        write_rqa_fields(o, p);
        o << ",\"show_var\":[";
        for (size_t v = 0; v < p.show_var.size(); ++v) { if (v) o << ","; o << (p.show_var[v] ? "true" : "false"); }
        o << "]}";
    }
    o << "]";
}


void read_basins_phase_projections(JP& p, std::vector<Projection>& out) {
    out.clear();
    p.expect('[');
    if (p.opt(']')) return;
    while (true) {
        p.expect('{');
        Projection pr;
        while (true) {
            std::string k = p.str(); p.expect(':');
            if      (k == "label") pr.label = p.str();
            else if (k == "type")  pr.type = (ProjType)std::stoi(p.str_or_num());
            else if (k == "ax")    pr.axis_x = std::stoi(p.str_or_num());
            else if (k == "ay")    pr.axis_y = std::stoi(p.str_or_num());
            else if (k == "az")    pr.axis_z = std::stoi(p.str_or_num());
            else if (k == "cls")   pr.custom_line_style = p.boolean();
            else if (k == "lw")    pr.line_width = (float)std::stod(p.str_or_num());
            else if (k == "al")    pr.alpha      = (float)std::stod(p.str_or_num());
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
        out.push_back(std::move(pr));
        if (p.opt(',')) continue;
        p.expect(']'); break;
    }
}


void write_basins_config(std::ostringstream& o, const BasinsConfig& c,
                                const BasinsPhaseSlot* phase_slot) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
    o << "\"adaptive\":"; write_adaptive(o,c.adaptive); o << ",";
    o << "\"scheme\":";           jstr(o, c.scheme);           o << ",";
    o << "\"symmetry_s\":";       jstr(o, c.symmetry_s);       o << ",";
    o << "\"axis_x_var\":"        << c.axis_x_var            << ",";
    o << "\"axis_y_var\":"        << c.axis_y_var            << ",";
    o << "\"axis_x_lo_text\":";   jstr(o, c.axis_x_lo_text);   o << ",";
    o << "\"axis_x_hi_text\":";   jstr(o, c.axis_x_hi_text);   o << ",";
    o << "\"axis_y_lo_text\":";   jstr(o, c.axis_y_lo_text);   o << ",";
    o << "\"axis_y_hi_text\":";   jstr(o, c.axis_y_hi_text);   o << ",";
    o << "\"n_pts_text\":";       jstr(o, c.n_pts_text);       o << ",";
    o << "\"writable_var\":"      << c.writable_var          << ",";
    o << "\"h_text\":";           jstr(o, c.h_text);           o << ",";
    o << "\"t_max_text\":";       jstr(o, c.t_max_text);       o << ",";
    o << "\"transient_text\":";   jstr(o, c.transient_text);   o << ",";
    o << "\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text); o << ",";
    o << "\"max_value_text\":";   jstr(o, c.max_value_text);   o << ",";
    o << "\"eps_dbscan_text\":";  jstr(o, c.eps_dbscan_text);  o << ",";
    o << "\"csv_save_enabled\":"  << (c.csv_save_enabled ? "true" : "false") << ",";
    o << "\"csv_output_path\":";  jstr(o, c.csv_output_path);  o << ",";
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";
    o << "\"param_values\":";     jmap(o, c.param_values);     o << ",";
    o << "\"feature1\":"          << c.feature1            << ",";
    o << "\"feature2\":"          << c.feature2            << ",";
    o << "\"mult_feature1_text\":"; jstr(o, c.mult_feature1_text); o << ",";
    o << "\"mult_feature2_text\":"; jstr(o, c.mult_feature2_text); o << ",";
    o << "\"renumber_spiral\":"   << (c.renumber_spiral ? "true" : "false") << ",";
    // Per-tab colormap — четыре отдельных скаляра, а не массив: парсер JP
    // читает скаляры, и добавлять ради этого чтение массивов излишне.
    o << "\"colormap_basins\":"   << c.colormap_idx[0] << ",";
    o << "\"colormap_feature1\":" << c.colormap_idx[1] << ",";
    o << "\"colormap_feature2\":" << c.colormap_idx[2] << ",";
    o << "\"colormap_states\":"   << c.colormap_idx[3] << ",";
    o << "\"active_plot_tab\":"   << c.active_plot_tab << ",";
    // Фазовые портреты по бассейнам: настройки — в самом config'е, список окон
    // берётся из parallel-слота (сам слот move-only и в config не помещается).
    o << "\"pp_t_max_text\":";          jstr(o, c.pp_t_max_text);          o << ",";
    o << "\"pp_transient_text\":";      jstr(o, c.pp_transient_text);      o << ",";
    o << "\"pp_prescaller_text\":";     jstr(o, c.pp_prescaller_text);     o << ",";
    o << "\"pp_max_attractors_text\":"; jstr(o, c.pp_max_attractors_text); o << ",";
    o << "\"pp_use_gpu\":"     << (c.pp_use_gpu ? "true" : "false") << ",";
    o << "\"pp_autorun\":"     << (c.pp_autorun ? "true" : "false") << ",";
    o << "\"pp_marker_size\":" << c.pp_marker_size << ",";
    o << "\"phase_projections\":";
    if (phase_slot) write_basins_phase_projections(o, phase_slot->phase.projections);
    else            o << "[]";
    o << "}";
}


// Чтение одного поля BasinsConfig. Возвращает true если ключ распознан и
// значение прочитано из p. Иначе false — caller должен сам сделать skip_value.
// Используется и при разборе нового формата (внутри "configs"), и при
// разборе legacy плоского формата.
bool read_basins_field(JP& p, BasinsConfig& c, const std::string& key) {
    if      (key == "label")              c.label             = p.str();
    else if (key == "scheme")             c.scheme            = p.str();
    else if (key == "adaptive") read_adaptive(p,c.adaptive);
    else if (key == "symmetry_s")         c.symmetry_s        = p.str();
    else if (key == "axis_x_var")         c.axis_x_var        = std::stoi(p.str_or_num());
    else if (key == "axis_y_var")         c.axis_y_var        = std::stoi(p.str_or_num());
    else if (key == "axis_x_lo_text")     c.axis_x_lo_text    = p.str();
    else if (key == "axis_x_hi_text")     c.axis_x_hi_text    = p.str();
    else if (key == "axis_y_lo_text")     c.axis_y_lo_text    = p.str();
    else if (key == "axis_y_hi_text")     c.axis_y_hi_text    = p.str();
    else if (key == "n_pts_text")         c.n_pts_text        = p.str();
    else if (key == "writable_var")       c.writable_var      = std::stoi(p.str_or_num());
    else if (key == "h_text")             c.h_text            = p.str();
    else if (key == "t_max_text")         c.t_max_text        = p.str();
    else if (key == "transient_text")     c.transient_text    = p.str();
    else if (key == "pre_scaller_text")   c.pre_scaller_text  = p.str();
    else if (key == "max_value_text")     c.max_value_text    = p.str();
    else if (key == "eps_dbscan_text")    c.eps_dbscan_text   = p.str();
    else if (key == "csv_save_enabled")   c.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    c.csv_output_path   = p.str();
    else if (key == "initial_conditions") c.initial_conditions= p.map_ss();
    else if (key == "param_values")       c.param_values      = p.map_ss();
    else if (key == "feature1")           c.feature1           = std::stoi(p.str_or_num());
    else if (key == "feature2")           c.feature2           = std::stoi(p.str_or_num());
    else if (key == "mult_feature1_text") c.mult_feature1_text = p.str();
    else if (key == "mult_feature2_text") c.mult_feature2_text = p.str();
    else if (key == "renumber_spiral")    c.renumber_spiral    = p.boolean();
    else if (key == "colormap_basins")    c.colormap_idx[0]    = std::stoi(p.str_or_num());
    else if (key == "colormap_feature1")  c.colormap_idx[1]    = std::stoi(p.str_or_num());
    else if (key == "colormap_feature2")  c.colormap_idx[2]    = std::stoi(p.str_or_num());
    else if (key == "colormap_states")    c.colormap_idx[3]    = std::stoi(p.str_or_num());
    else if (key == "active_plot_tab")    c.active_plot_tab   = std::stoi(p.str_or_num());
    else if (key == "pp_t_max_text")          c.pp_t_max_text          = p.str();
    else if (key == "pp_transient_text")      c.pp_transient_text      = p.str();
    else if (key == "pp_prescaller_text")     c.pp_prescaller_text     = p.str();
    else if (key == "pp_max_attractors_text") c.pp_max_attractors_text = p.str();
    else if (key == "pp_use_gpu")             c.pp_use_gpu             = p.boolean();
    else if (key == "pp_autorun")             c.pp_autorun             = p.boolean();
    else if (key == "pp_marker_size")         c.pp_marker_size         = (float)std::stod(p.str_or_num());
    else return false;
    return true;
}


// FastSyncAnalysisSession JSON — multi-config layout как у basins.
void write_fastsync_config(std::ostringstream& o, const FastSyncConfig& c) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
    o << "\"adaptive\":"; write_adaptive(o,c.adaptive); o << ",";
    o << "\"scheme\":";           jstr(o, c.scheme);           o << ",";
    o << "\"symmetry_s\":";       jstr(o, c.symmetry_s);       o << ",";
    o << "\"mode\":"              << c.mode                  << ",";
    o << "\"h_text\":";           jstr(o, c.h_text);           o << ",";
    o << "\"iter_of_synchr_text\":"; jstr(o, c.iter_of_synchr_text); o << ",";
    o << "\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text); o << ",";
    o << "\"max_value_text\":";   jstr(o, c.max_value_text);   o << ",";
    o << "\"t_max_text\":";       jstr(o, c.t_max_text);       o << ",";
    o << "\"transient_text\":";   jstr(o, c.transient_text);   o << ",";
    o << "\"window_text\":";      jstr(o, c.window_text);      o << ",";
    o << "\"axis_x_var\":"        << c.axis_x_var            << ",";
    o << "\"axis_y_var\":"        << c.axis_y_var            << ",";
    o << "\"axis_x_lo_text\":";   jstr(o, c.axis_x_lo_text);   o << ",";
    o << "\"axis_x_hi_text\":";   jstr(o, c.axis_x_hi_text);   o << ",";
    o << "\"axis_y_lo_text\":";   jstr(o, c.axis_y_lo_text);   o << ",";
    o << "\"axis_y_hi_text\":";   jstr(o, c.axis_y_hi_text);   o << ",";
    o << "\"n_pts_text\":";       jstr(o, c.n_pts_text);       o << ",";
    o << "\"transient_slave_text\":"; jstr(o, c.transient_slave_text); o << ",";
    o << "\"grid_swap_master_slave\":" << (c.grid_swap_master_slave ? "true" : "false") << ",";
    o << "\"ic_random_offset\":" << (c.ic_random_offset ? "true" : "false") << ",";
    o << "\"ic_eps_text\":";     jstr(o, c.ic_eps_text);      o << ",";
    o << "\"ic_seed_text\":";    jstr(o, c.ic_seed_text);     o << ",";
    o << "\"gs_warmup_text\":";  jstr(o, c.gs_warmup_text);   o << ",";
    o << "\"type_of_synch\":"     << c.type_of_synch         << ",";
    o << "\"error_estim\":"       << c.error_estim           << ",";
    o << "\"fs_error_trs_text\":"; jstr(o, c.fs_error_trs_text); o << ",";
    o << "\"colormap_idx\":"      << c.colormap_idx          << ",";
    o << "\"autoscale_color\":"   << (c.autoscale_color ? "true" : "false") << ",";
    o << "\"c_min_text\":";       jstr(o, c.c_min_text);       o << ",";
    o << "\"c_max_text\":";       jstr(o, c.c_max_text);       o << ",";
    o << "\"line_width\":"        << c.line_width            << ",";
    o << "\"alpha\":"             << c.alpha                 << ",";
    o << "\"swap_axes\":"         << (c.swap_axes ? "true" : "false") << ",";
    o << "\"invert_depth\":"      << (c.invert_depth ? "true" : "false") << ",";
    o << "\"csv_save_enabled\":"  << (c.csv_save_enabled ? "true" : "false") << ",";
    o << "\"csv_output_path\":";  jstr(o, c.csv_output_path);  o << ",";
    o << "\"ic_master\":";        jmap(o, c.ic_master);        o << ",";
    o << "\"ic_slave\":";         jmap(o, c.ic_slave);         o << ",";
    o << "\"k_forward\":";        jmap(o, c.k_forward);        o << ",";
    o << "\"k_backward\":";       jmap(o, c.k_backward);       o << ",";
    o << "\"param_values\":";     jmap(o, c.param_values);
    o << "}";
}


bool read_fastsync_field(JP& p, FastSyncConfig& c, const std::string& key) {
    if      (key == "label")               c.label               = p.str();
    else if (key == "scheme")              c.scheme              = p.str();
    else if (key == "adaptive") read_adaptive(p,c.adaptive);
    else if (key == "symmetry_s")          c.symmetry_s          = p.str();
    else if (key == "mode")                c.mode                = std::stoi(p.str_or_num());
    else if (key == "h_text")              c.h_text              = p.str();
    else if (key == "iter_of_synchr_text") c.iter_of_synchr_text = p.str();
    else if (key == "pre_scaller_text")    c.pre_scaller_text    = p.str();
    else if (key == "max_value_text")      c.max_value_text      = p.str();
    else if (key == "t_max_text")          c.t_max_text          = p.str();
    else if (key == "transient_text")      c.transient_text      = p.str();
    else if (key == "transient_slave_text") c.transient_slave_text = p.str();
    else if (key == "window_text")         c.window_text         = p.str();
    // Backward-compat: старые сессии могли называть это поле "n_time_text".
    else if (key == "n_time_text")         c.window_text         = p.str();
    else if (key == "axis_x_var")          c.axis_x_var          = std::stoi(p.str_or_num());
    else if (key == "axis_y_var")          c.axis_y_var          = std::stoi(p.str_or_num());
    else if (key == "axis_x_lo_text")      c.axis_x_lo_text      = p.str();
    else if (key == "axis_x_hi_text")      c.axis_x_hi_text      = p.str();
    else if (key == "axis_y_lo_text")      c.axis_y_lo_text      = p.str();
    else if (key == "axis_y_hi_text")      c.axis_y_hi_text      = p.str();
    else if (key == "n_pts_text")          c.n_pts_text          = p.str();
    else if (key == "grid_swap_master_slave") c.grid_swap_master_slave = p.boolean();
    // Сессии старше этого поля читаются как ic_random_offset = false, то есть
    // ровно с прежним поведением (фиксированные НУ слейва).
    else if (key == "ic_random_offset")    c.ic_random_offset    = p.boolean();
    else if (key == "ic_eps_text")         c.ic_eps_text         = p.str();
    else if (key == "ic_seed_text")        c.ic_seed_text        = p.str();
    else if (key == "gs_warmup_text")      c.gs_warmup_text      = p.str();
    else if (key == "type_of_synch")       c.type_of_synch       = std::stoi(p.str_or_num());
    else if (key == "error_estim")         c.error_estim         = std::stoi(p.str_or_num());
    else if (key == "fs_error_trs_text")   c.fs_error_trs_text   = p.str();
    else if (key == "colormap_idx")        c.colormap_idx        = std::stoi(p.str_or_num());
    else if (key == "autoscale_color")     c.autoscale_color     = p.boolean();
    else if (key == "c_min_text")          c.c_min_text          = p.str();
    else if (key == "c_max_text")          c.c_max_text          = p.str();
    else if (key == "line_width")          c.line_width          = (float)std::stod(p.str_or_num());
    else if (key == "alpha")               c.alpha               = (float)std::stod(p.str_or_num());
    else if (key == "swap_axes")           c.swap_axes           = p.boolean();
    else if (key == "invert_depth")        c.invert_depth        = p.boolean();
    else if (key == "csv_save_enabled")    c.csv_save_enabled    = p.boolean();
    else if (key == "csv_output_path")     c.csv_output_path     = p.str();
    else if (key == "decimator_view")      p.skip_value();   // legacy, не используется
    else if (key == "ic_master")           c.ic_master           = p.map_ss();
    else if (key == "ic_slave")            c.ic_slave            = p.map_ss();
    else if (key == "k_forward")           c.k_forward           = p.map_ss();
    else if (key == "k_backward")          c.k_backward          = p.map_ss();
    else if (key == "param_values")        c.param_values        = p.map_ss();
    else return false;
    return true;
}


// Custom-tab bundle: shared config + five embedded sub-session objects.


void write_shared_config(std::ostringstream& o, const CustomTabSharedConfig& c) {
    auto B = [&](const char* name, bool v) {
        o << "\""<<name<<"\":" << (v ? "true" : "false") << ",";
    };
    auto S = [&](const char* name, const std::string& v) {
        o << "\""<<name<<"\":"; jstr(o, v); o << ",";
    };
    auto I = [&](const char* name, int v) { o << "\""<<name<<"\":" << v << ","; };
    auto D = [&](const char* name, double v) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.17g", v);
        o << "\""<<name<<"\":" << buf << ",";
    };
    o << "{";
    B("level_2d_enabled",    c.level_2d_enabled);
    B("level_1d_enabled",    c.level_1d_enabled);
    B("level_phase_enabled", c.level_phase_enabled);
    o << "\"adaptive\":"; write_adaptive(o,c.adaptive); o << ",";
    S("scheme",              c.scheme);
    S("symmetry_s",          c.symmetry_s);
    S("h_text",              c.h_text);
    S("t_max_text",          c.t_max_text);
    S("transient_text",      c.transient_text);
    S("pre_scaller_text",    c.pre_scaller_text);
    S("max_value_text",      c.max_value_text);
    o << "\"initial_conditions\":"; jmap(o, c.initial_conditions); o << ",";
    o << "\"param_values\":";       jmap(o, c.param_values);       o << ",";
    I("axis_x_par_index",    c.axis_x_par_index);
    B("axis_x_over_var",     c.axis_x_over_var);
    B("axis_x_over_h",       c.axis_x_over_h);
    I("axis_x_var_index",    c.axis_x_var_index);
    S("axis_x_lo_text",      c.axis_x_lo_text);
    S("axis_x_hi_text",      c.axis_x_hi_text);
    B("axis_x_log",          c.axis_x_log);
    I("axis_y_par_index",    c.axis_y_par_index);
    B("axis_y_over_var",     c.axis_y_over_var);
    B("axis_y_over_h",       c.axis_y_over_h);
    I("axis_y_var_index",    c.axis_y_var_index);
    S("axis_y_lo_text",      c.axis_y_lo_text);
    S("axis_y_hi_text",      c.axis_y_hi_text);
    B("axis_y_log",          c.axis_y_log);
    S("resolution_text",     c.resolution_text);
    I("bif_writable_var",    c.bif_writable_var);
    B("bif2d_enabled",       c.bif2d_enabled);
    B("lle2d_enabled",       c.lle2d_enabled);
    B("ls2d_enabled",        c.ls2d_enabled);
    B("inherit_sweep_from_2d", c.inherit_sweep_from_2d);
    I("sweep_x_par_index",   c.sweep_x_par_index);
    B("sweep_x_over_var",    c.sweep_x_over_var);
    B("sweep_x_over_h",      c.sweep_x_over_h);
    I("sweep_x_var_index",   c.sweep_x_var_index);
    S("sweep_x_lo_text",     c.sweep_x_lo_text);
    S("sweep_x_hi_text",     c.sweep_x_hi_text);
    B("sweep_x_log",         c.sweep_x_log);
    S("n_x_1d_text",         c.n_x_1d_text);
    I("sweep_y_par_index",   c.sweep_y_par_index);
    B("sweep_y_over_var",    c.sweep_y_over_var);
    B("sweep_y_over_h",      c.sweep_y_over_h);
    I("sweep_y_var_index",   c.sweep_y_var_index);
    S("sweep_y_lo_text",     c.sweep_y_lo_text);
    S("sweep_y_hi_text",     c.sweep_y_hi_text);
    B("sweep_y_log",         c.sweep_y_log);
    S("n_y_1d_text",         c.n_y_1d_text);
    S("l1d_h_text",          c.l1d_h_text);
    S("l1d_transient_text",  c.l1d_transient_text);
    S("l1d_t_max_text",      c.l1d_t_max_text);
    B("bif1d_x_enabled",     c.bif1d_x_enabled);
    B("bif1d_y_enabled",     c.bif1d_y_enabled);
    B("lle1d_x_enabled",     c.lle1d_x_enabled);
    B("lle1d_y_enabled",     c.lle1d_y_enabled);
    B("ls1d_x_enabled",      c.ls1d_x_enabled);
    B("ls1d_y_enabled",      c.ls1d_y_enabled);
    B("continuation_1d_enabled", c.continuation_1d_enabled);
    B("plot_inter_peaks_1d", c.plot_inter_peaks_1d);
    D("fix_x_value",         c.fix_x_value);
    D("fix_y_value",         c.fix_y_value);
    B("auto_recompute_1d",   c.auto_recompute_1d);
    I("level3_kind",         c.level3_kind);
    o << "\"autorun_on_drilldown\":" << (c.autorun_on_drilldown ? "true" : "false");
    o << "}";
}


// Workspace metadata: tabs, active/next id, splitter width. imgui.ini owns
// the per-window dock layout inside each tab (keyed off the tab-derived
// dockspace id in ws_dock_id).
void write_workspace(std::ostringstream& o, const CustomWorkspace& ws) {
    o << "{";
    o << "\"active_tab_id\":" << ws.active_tab_id << ",";
    o << "\"next_tab_id\":"   << ws.next_tab_id   << ",";
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", (double)ws.controls_width);
        o << "\"controls_width\":" << buf << ",";
    }
    o << "\"tabs\":[";
    for (size_t i = 0; i < ws.tabs.size(); ++i) {
        if (i) o << ",";
        o << "{\"id\":" << ws.tabs[i].id << ",\"name\":";
        jstr(o, ws.tabs[i].name);
        o << "}";
    }
    o << "]}";
}
}
