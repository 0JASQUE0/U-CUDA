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


// ---- Библиотека регуляторов шага ----
void write_str_array(std::ostringstream& o, const std::vector<std::string>& v) {
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) o << ","; jstr(o, v[i]); }
    o << "]";
}


void read_str_array(JP& p, std::vector<std::string>& v) {
    v.clear();
    p.expect('[');
    if (p.opt(']')) return;
    while (true) {
        v.push_back(p.str_or_num());
        if (p.opt(',')) continue;
        p.expect(']');
        break;
    }
}


void write_user_ctrl(std::ostringstream& o, const AdaptiveUserCtrl& u) {
    o << "{\"name\":"; jstr(o, u.name);
    o << ",\"kind\":\"" << (u.kind == kUserCtrlFilter ? "filter" : "body") << "\"";
    o << ",\"hairer_rules\":" << (u.hairer_rules ? "true" : "false");
    o << ",\"par_names\":";  write_str_array(o, u.par_names);
    o << ",\"par_values\":"; write_str_array(o, u.par_values);
    o << ",\"tip\":";  jstr(o, u.tip);
    o << ",\"body\":"; jstr(o, u.body);
    if (!u.prep.empty()) { o << ",\"prep\":"; jstr(o, u.prep); }   // раздел подготовки C body
    o << "}";
}


void read_user_ctrl(JP& p, AdaptiveUserCtrl& u) {
    u = AdaptiveUserCtrl();
    p.expect('{');
    if (p.opt('}')) return;
    while (true) {
        const std::string k = p.str();
        p.expect(':');
        if      (k == "name")         u.name = p.str();
        else if (k == "kind")         u.kind = p.str() == "filter" ? kUserCtrlFilter : kUserCtrlBody;
        else if (k == "hairer_rules") u.hairer_rules = p.boolean() ? 1 : 0;
        else if (k == "par_names")    read_str_array(p, u.par_names);
        else if (k == "par_values")   read_str_array(p, u.par_values);
        else if (k == "tip")          u.tip = p.str();
        else if (k == "body")         u.body = p.str();
        else if (k == "prep")         u.prep = p.str();
        else p.skip_value();
        if (p.opt(',')) continue;
        p.expect('}');
        break;
    }
}


// Тот же ли регулятор считается (имя и описание не в счёт).
bool same_ctrl_definition(const AdaptiveUserCtrl& a, const AdaptiveUserCtrl& b) {
    return a.kind == b.kind && a.body == b.body && a.prep == b.prep && a.par_names == b.par_names
        && a.par_values == b.par_values && a.hairer_rules == b.hairer_rules;
}


// Регулятор из сессии (ctrl_def) -> библиотека; возвращает имя, под которым он там есть.
// Нет такого имени — добавляется как есть. Есть с тем же определением — оно и есть. Есть, но
// считает иначе (правили после сохранения сессии, или сессия с другой машины), — сессия
// получает своё: запись с тем же определением под любым именем или новая "<имя> (session)".
// Раньше библиотека молча побеждала, и сессия считалась не тем регулятором, что сохранён.
std::string import_session_ctrl(const AdaptiveUserCtrl& u) {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::string use = u.name;
    bool add = false;
    {
        std::lock_guard<std::mutex> lk(L.mu);
        const AdaptiveUserCtrl* same_name = nullptr;
        for (const AdaptiveUserCtrl& x : L.items) if (x.name == u.name) { same_name = &x; break; }
        if (same_name == nullptr) {
            L.items.push_back(u);
            add = true;
        } else if (!same_ctrl_definition(*same_name, u)) {
            const AdaptiveUserCtrl* same_def = nullptr;
            for (const AdaptiveUserCtrl& x : L.items) if (same_ctrl_definition(x, u)) { same_def = &x; break; }
            if (same_def != nullptr) {
                use = same_def->name;
            } else {
                auto taken = [&](const std::string& s) {
                    for (const AdaptiveUserCtrl& x : L.items) if (x.name == s) return true;
                    return false;
                };
                use = u.name + " (session)";
                for (int k = 2; taken(use); ++k) use = u.name + " (session " + std::to_string(k) + ")";
                AdaptiveUserCtrl c = u;
                c.name = use;
                L.items.push_back(c);
                add = true;
            }
        }
    }
    if (add) ucuda::storage_detail::save_ctrl_library(nullptr);
    return use;
}


// ---- Адаптивный шаг ----
// Вложенный объект "adaptive" пишется, только если настройки отличаются от
// умолчания: сессии без адаптивного шага не меняются ни на байт, а старые
// читаются как Fixed (ключа нет — остаются дефолты AdaptiveSettings).
void write_adaptive(std::ostringstream& o, const AdaptiveSettings& a,
                    const char* prefix, const char* suffix) {
    if (a == AdaptiveSettings()) return;
    o << prefix << "\"adaptive\":{";
    o << "\"enabled\":" << (a.enabled ? "true" : "false");
    o << ",\"raw_nodes\":" << (a.raw_nodes ? "true" : "false");
    o << ",\"rtol\":"; jstr(o, a.rtol);
    o << ",\"atol\":"; jstr(o, a.atol);
    o << ",\"h0\":";   jstr(o, a.h0);
    o << ",\"hmin\":"; jstr(o, a.hmin);
    o << ",\"hmax\":"; jstr(o, a.hmax);
    o << ",\"max_rej\":"; jstr(o, a.max_rej);
    o << ",\"ctrl\":"; jstr(o, a.ctrl);
    // 2 — имена после разделения Хайрера ("Hairer fast", "Hairer numb"). Без метки "Hairer" —
    // прежний (float) закон, см. read_adaptive.
    o << ",\"ctrl_v\":2";
    o << ",\"ctrl_params\":[";
    for (size_t i = 0; i < a.ctrl_params.size(); ++i) { if (i) o << ","; jstr(o, a.ctrl_params[i]); }
    o << "]";
    o << ",\"peak_interp\":" << a.peak_interp;
    o << ",\"max_points\":"; jstr(o, a.max_points);
    o << ",\"lyap_renorm\":" << a.lyap_renorm;
    o << ",\"minmax_interp_fixed\":" << (a.minmax_interp_fixed ? "true" : "false");
    // Регулятор из библиотеки едет с сессией целиком: на машине, где его нет, он
    // добавится в библиотеку при чтении (read_adaptive).
    AdaptiveUserCtrl u;
    if (!adaptive_find_builtin(a.ctrl) && adaptive_find_user_ctrl(a.ctrl, &u)) {
        o << ",\"ctrl_def\":";
        write_user_ctrl(o, u);
    }
    o << "}" << suffix;
}


void read_adaptive(JP& p, AdaptiveSettings& a) {
    a = AdaptiveSettings();
    // The retired local solver used a numeric `method` selector and different
    // tolerance semantics. Its block must not enable the upstream solver.
    JP probe = p;
    probe.expect('{');
    if (!probe.opt('}')) {
        while (true) {
            const std::string key = probe.str();
            probe.expect(':');
            if (key == "method") { p.skip_value(); return; }
            probe.skip_value();
            if (probe.opt(',')) continue;
            probe.expect('}');
            break;
        }
    }
    p.expect('{');
    if (p.opt('}')) return;
    std::string def_name, def_use;   // ctrl_def: имя в сессии -> имя в библиотеке
    int ctrl_v = 1;
    while (true) {
        const std::string k = p.str();
        p.expect(':');
        if      (k == "enabled")     a.enabled   = p.boolean();
        else if (k == "ctrl_v")      { try { ctrl_v = std::stoi(p.str_or_num()); } catch (...) {} }
        else if (k == "raw_nodes")   a.raw_nodes = p.boolean();
        else if (k == "rtol")        a.rtol = p.str();
        else if (k == "atol")        a.atol = p.str();
        else if (k == "h0")          a.h0   = p.str();
        else if (k == "hmin")        a.hmin = p.str();
        else if (k == "hmax")        a.hmax = p.str();
        else if (k == "max_rej")     a.max_rej = p.str();
        else if (k == "ctrl")        a.ctrl = p.str();
        else if (k == "ctrl_params") {
            a.ctrl_params.clear();
            p.expect('[');
            if (!p.opt(']')) {
                while (true) {
                    a.ctrl_params.push_back(p.str());
                    if (p.opt(',')) continue;
                    p.expect(']');
                    break;
                }
            }
        }
        else if (k == "peak_interp") { try { a.peak_interp = std::stoi(p.str_or_num()); } catch (...) {} }
        else if (k == "max_points")  a.max_points = p.str();
        else if (k == "lyap_renorm") { try { a.lyap_renorm = std::stoi(p.str_or_num()); } catch (...) {} }
        else if (k == "minmax_interp_fixed") a.minmax_interp_fixed = p.boolean();
        else if (k == "ctrl_def") {
            // Регулятор едет с сессией: в библиотеку, если его там нет, а если под тем же именем
            // лежит другое определение — под своим именем (import_session_ctrl).
            AdaptiveUserCtrl u;
            read_user_ctrl(p, u);
            if (!u.name.empty() && !adaptive_find_builtin(u.name)) {
                def_name = u.name;
                def_use  = import_session_ctrl(u);
            }
        }
        else p.skip_value();
        if (p.opt(',')) continue;
        p.expect('}');
        break;
    }
    // Порядок ключей в файле любой, поэтому переименование — после разбора.
    if (!def_name.empty() && a.ctrl == def_name && def_use != def_name) a.ctrl = def_use;
    // "Hairer" больше нет: без метки (сессии до разделения Хайрера) это float-закон — "Hairer fast",
    // с меткой 2 — вариант в numb — "Hairer numb". Параметры те же.
    if (a.ctrl == "Hairer") a.ctrl = ctrl_v < 2 ? "Hairer fast" : "Hairer numb";
    // Регулятор, которого в этой сборке нет (UCUDA_AD_HAIRER_ONLY), — на Хайрера с его
    // параметрами по умолчанию: чужие параметры к нему не подходят.
    if (adaptive_ctrl_effective(a.ctrl) != a.ctrl) {
        a.ctrl = adaptive_ctrl_effective(a.ctrl);
        a.ctrl_params.clear();
        a.ctrl_replaced = true;
    }
}


// Регулятор заменён (read_adaptive): ось свипа по параметру прежнего регулятора
// (kAdAxisCtrl + k) указывала бы теперь на другой параметр Хайрера или на c[k], которого
// он не читает, и свип молча считал бы не то. Такая ось возвращается к обычной цели свипа.
// Зовётся и после регулятора, и после осей: порядок ключей в файле любой.
void drop_foreign_ctrl_axes(const AdaptiveSettings& a, int& ax, int& ax2) {
    if (!a.ctrl_replaced) return;
    if (ax  >= kAdAxisCtrl) ax  = kAdAxisSystem;
    if (ax2 >= kAdAxisCtrl) ax2 = kAdAxisSystem;
}

// BifurcationAnalysisSession

// Сериализует одну БД в JSON-объект (без обёртки фигурными). Используется
// внутри массива "diagrams".
void write_diagram(std::ostringstream& o, const BifurcationDiagramConfig& bd) {
    o << "{";
    o << "\"label\":";            jstr(o, bd.label);
    o << ",\"label_is_manual\":"  << (bd.label_is_manual ? "true" : "false");
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
    write_adaptive(o, bd.adaptive, ",", "");
    if (bd.ad_axis)   o << ",\"ad_axis\":"   << bd.ad_axis;
    if (bd.ad_axis_2) o << ",\"ad_axis_2\":" << bd.ad_axis_2;
    o << ",\"param_values\":";    jmap(o, bd.param_values);
    o << ",\"initial_conditions\":"; jmap(o, bd.initial_conditions);
    o << ",\"csv_save_enabled\":" << (bd.csv_save_enabled ? "true" : "false");
    o << ",\"csv_output_path\":"; jstr(o, bd.csv_output_path);
    o << ",\"plot_inter_peaks\":" << (bd.plot_inter_peaks ? "true" : "false");
    o << ",\"plot_all_iterates\":" << (bd.plot_all_iterates ? "true" : "false");
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
    else if (key == "adaptive")           { read_adaptive(p, bd.adaptive);
                                            drop_foreign_ctrl_axes(bd.adaptive, bd.ad_axis, bd.ad_axis_2); }
    else if (key == "ad_axis")            { try { bd.ad_axis   = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(bd.adaptive, bd.ad_axis, bd.ad_axis_2); }
    else if (key == "ad_axis_2")          { try { bd.ad_axis_2 = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(bd.adaptive, bd.ad_axis, bd.ad_axis_2); }
    else if (key == "param_values")       bd.param_values      = p.map_ss();
    else if (key == "initial_conditions") bd.initial_conditions= p.map_ss();
    else if (key == "csv_save_enabled")   bd.csv_save_enabled  = p.boolean();
    else if (key == "csv_output_path")    bd.csv_output_path   = p.str();
    else if (key == "plot_inter_peaks")   bd.plot_inter_peaks  = p.boolean();
    else if (key == "plot_all_iterates")  bd.plot_all_iterates = p.boolean();
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
    write_adaptive(o, c.adaptive, ",", "");
    if (c.ad_axis)   o << ",\"ad_axis\":"   << c.ad_axis;
    if (c.ad_axis_2) o << ",\"ad_axis_2\":" << c.ad_axis_2;
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    if (c.vtr_text != "0") { o << ",\"vtr_text\":"; jstr(o, c.vtr_text); }
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
    else if (key == "adaptive")           { read_adaptive(p, c.adaptive);
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis")            { try { c.ad_axis   = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis_2")          { try { c.ad_axis_2 = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "vtr_text")           c.vtr_text          = p.str();
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
    write_adaptive(o, c.adaptive, ",", "");
    if (c.ad_axis)   o << ",\"ad_axis\":"   << c.ad_axis;
    if (c.ad_axis_2) o << ",\"ad_axis_2\":" << c.ad_axis_2;
    o << ",\"eps_text\":";        jstr(o, c.eps_text);
    o << ",\"nt_text\":";         jstr(o, c.nt_text);
    if (c.vtr_text != "0") { o << ",\"vtr_text\":"; jstr(o, c.vtr_text); }
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
    else if (key == "adaptive")           { read_adaptive(p, c.adaptive);
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis")            { try { c.ad_axis   = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "ad_axis_2")          { try { c.ad_axis_2 = std::stoi(p.str_or_num()); } catch (...) {}
                                            drop_foreign_ctrl_axes(c.adaptive, c.ad_axis, c.ad_axis_2); }
    else if (key == "eps_text")           c.eps_text          = p.str();
    else if (key == "nt_text")            c.nt_text           = p.str();
    else if (key == "vtr_text")           c.vtr_text          = p.str();
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
        o << ",\"pts\":" << (p.draw_points ? "true" : "false") << ",\"ps\":" << p.point_size;
        if (p.y_log) o << ",\"ylog\":true";
        if (p.point_alpha != 1.0f) o << ",\"pa\":" << p.point_alpha;
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
            else if (k == "pts")   pr.draw_points = p.boolean();   // писались всегда, не читались
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
        out.push_back(std::move(pr));
        if (p.opt(',')) continue;
        p.expect(']'); break;
    }
}


void write_basins_config(std::ostringstream& o, const BasinsConfig& c,
                                const BasinsPhaseSlot* phase_slot) {
    o << "{";
    o << "\"label\":";            jstr(o, c.label);            o << ",";
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
    write_adaptive(o, c.adaptive, "", ",");
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
    else if (key == "adaptive")           read_adaptive(p, c.adaptive);
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
    o << "\"plot_3d\":"           << (c.plot_3d ? "true" : "false") << ",";
    o << "\"axis_z_var\":"        << c.axis_z_var            << ",";
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
    else if (key == "plot_3d")             c.plot_3d             = p.boolean();
    else if (key == "axis_z_var")          c.axis_z_var          = std::stoi(p.str_or_num());
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


// ---------------------------------------------------------------------------
// NetworkSession JSON — multi-config layout как у basins/fastsync, плюс две
// вложенные коллекции: узлы (переопределения параметров и НУ, положение в
// редакторе) и рёбра. Результат расчёта не пишется — он на порядки больше
// самой сессии и восстанавливается одним Run.
// Переопределения узла: пустое значение значит «берём общее», и таблица UI
// заводит такой ключ на КАЖДЫЙ параметр каждого узла просто потому, что по нему
// кликнули. В файл они не идут — иначе сессия из 256 узлов распухает на
// тысячи пустых строк, не неся ничего.
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
    S("scheme",              c.scheme);
    S("symmetry_s",          c.symmetry_s);
    S("h_text",              c.h_text);
    S("t_max_text",          c.t_max_text);
    S("transient_text",      c.transient_text);
    S("pre_scaller_text",    c.pre_scaller_text);
    S("max_value_text",      c.max_value_text);
    write_adaptive(o, c.adaptive, "", ",");
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


// ---- Библиотека регуляторов шага (step_controllers.json) ----
bool load_ctrl_library(const std::string& path, std::string* err) {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    {
        std::lock_guard<std::mutex> lk(L.mu);
        L.path = path;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) return true;   // файла ещё нет — пустая библиотека
    std::stringstream ss; ss << f.rdbuf();
    std::string text = ss.str();
    if (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB
        && (unsigned char)text[2] == 0xBF) text.erase(0, 3);
    std::vector<AdaptiveUserCtrl> items;
    try {
        JP p(text);
        p.expect('{');
        if (!p.opt('}')) {
            while (true) {
                const std::string k = p.str();
                p.expect(':');
                if (k == "controllers") {
                    p.expect('[');
                    if (!p.opt(']')) {
                        while (true) {
                            AdaptiveUserCtrl u;
                            read_user_ctrl(p, u);
                            if (!u.name.empty()) items.push_back(u);
                            if (p.opt(',')) continue;
                            p.expect(']');
                            break;
                        }
                    }
                }
                else p.skip_value();
                if (p.opt(',')) continue;
                p.expect('}');
                break;
            }
        }
    }
    catch (const std::exception& e) {
        if (err) *err = path + ": " + e.what();
        return false;
    }
    std::lock_guard<std::mutex> lk(L.mu);
    L.items = std::move(items);
    return true;
}


bool save_ctrl_library(std::string* err) {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::vector<AdaptiveUserCtrl> items;
    std::string path;
    // Писателей двое: окно библиотеки (UI) и чтение сессии с ctrl_def (read_adaptive), и
    // поток чтения бывает не UI. Замок файла — на всю запись, и снимок записей берётся под
    // ним: иначе писатель с более старым снимком мог бы записать его последним.
    static std::mutex file_mu;
    std::lock_guard<std::mutex> file_lk(file_mu);
    {
        std::lock_guard<std::mutex> lk(L.mu);
        items = L.items;
        path = L.path;
    }
    if (path.empty()) { if (err) *err = "the controller library has no file"; return false; }
    std::ostringstream o;
    o << "{\"version\":1,\"controllers\":[";
    for (size_t i = 0; i < items.size(); ++i) {
        o << (i ? ",\n" : "\n");
        write_user_ctrl(o, items[i]);
    }
    o << "\n]}\n";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { if (err) *err = "cannot write " + tmp; return false; }
        const std::string text = o.str();
        f.write(text.data(), (std::streamsize)text.size());
        if (!f) { if (err) *err = "cannot write " + tmp; return false; }
    }
    // Замена одним шагом: filesystem::rename на Windows — MoveFileEx с REPLACE_EXISTING.
    // Прежние remove + rename при сбое между ними оставляли библиотеку без файла.
    std::error_code ec;
    std::filesystem::rename(std::filesystem::path(tmp), std::filesystem::path(path), ec);
    // Без ec.message(): в русской Windows это текст в CP1251, а ImGui рисует UTF-8.
    if (ec) { if (err) *err = "cannot replace " + path + " (error " + std::to_string(ec.value()) + ")"; return false; }
    return true;
}
}
using namespace ucuda::storage_detail;



bool load_ctrl_library(const std::string& path, std::string* err) { return ucuda::storage_detail::load_ctrl_library(path, err); }
bool save_ctrl_library(std::string* err) { return ucuda::storage_detail::save_ctrl_library(err); }
