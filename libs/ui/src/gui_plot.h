#pragma once
// Графики: экспорт, диапазоны видов, вкладки конфигов, Run all, панели стилей, оси свипа,
// тепловые карты и палитры, буферы окон.
#include "gui_fields.h"

namespace ucuda::ui_detail {
void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint);


void draw_error_box(const char* id, const std::string& err,
                           int lines = kErrorBoxLines);


void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write);


struct ViewRangeSlots {
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    bool*        y_manual = nullptr;
};


struct ViewRangeTarget {
    std::string  name;                  // подпись в меню, когда целей несколько
    std::string* x_lo = nullptr;
    std::string* x_hi = nullptr;
    std::string* y_lo = nullptr;
    std::string* y_hi = nullptr;
    // Необязательный флаг "Y задан вручную" (colored-1D у Bif): без него
    // записанные ymin/ymax остались бы лежать мёртвым грузом при авто-Y.
    bool*        y_manual = nullptr;
    // Тип диаграммы — по нему работает Ctrl+Shift+T. "Parametric 2D" общий у
    // Bif/LLE/LS: у них совпадает и смысл осей, и кодировка цели свипа, поэтому
    // приблизив 2D-бифуркацию осмысленно подтянуть 2D LLE и 2D LS. У Basins и
    // FastSync оси — сетка НУ, у DFT по Y частоты; они в свои типы и попадают.
    // Пусто у целей, собранных на месте: тип заполняет только общий перечислитель.
    std::string kind;
    std::string group;   // подпись для списка галочек: "Bifurcation 2D", ...
    // Как найти ЭТУ ЖЕ цель потом. Нужно отмене: между Ctrl+T и Ctrl+Z вкладку
    // диаграммы могут закрыть, вектор конфигов переедет, и указатели выше
    // повиснут. Лямбда ищет конфиг по индексу и сверяет label; если его больше
    // нет или на его месте другой — возвращает нули, и откат этого поля просто
    // не делается. Пусто = цель незаписываемая (см. view_range_bound).
    std::function<ViewRangeSlots()> relocate;
};


template <class T>
[[nodiscard]] static ViewRangeTarget vr_target(std::vector<T>& vec, int idx,
        std::string T::*x_lo, std::string T::*x_hi,
        std::string T::*y_lo = nullptr, std::string T::*y_hi = nullptr,
        bool T::*y_manual = nullptr) {
    ViewRangeTarget t;
    if (idx < 0 || idx >= (int)vec.size()) return t;   // цели нет — пустая
    t.relocate = [&vec, idx, label_id = label_identity(vec[(size_t)idx].label),
                  x_lo, x_hi, y_lo, y_hi, y_manual]() {
        ViewRangeSlots s;
        if (idx < 0 || idx >= (int)vec.size()) return s;
        T& c = vec[(size_t)idx];
        // Сверяем ИМЯ ОСИ, а не подпись целиком: диапазон в автоподписи 1D-диаграммы
        // меняет сама эта команда (см. label_identity).
        if (label_identity(c.label) != label_id) return s;   // на этом индексе другой конфиг
        s.x_lo = &(c.*x_lo); s.x_hi = &(c.*x_hi);
        if (y_lo)     s.y_lo = &(c.*y_lo);
        if (y_hi)     s.y_hi = &(c.*y_hi);
        if (y_manual) s.y_manual = &(c.*y_manual);
        return s;
    };
    const ViewRangeSlots live = t.relocate();
    t.name     = vec[(size_t)idx].label;
    t.x_lo     = live.x_lo;
    t.x_hi     = live.x_hi;
    t.y_lo     = live.y_lo;
    t.y_hi     = live.y_hi;
    t.y_manual = live.y_manual;
    return t;
}


[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo = nullptr, std::string* y_hi = nullptr);


void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model);


void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model);


[[nodiscard]] bool plot_window_active();


void draw_label_rename(const char* label_with_id, std::string& label,
                              float width = 200.0f);


struct TabBarResult {
    int active    = -1;
    int to_remove = -1;
};


void track_dock_selection(AppModel& model);


TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select = -1);


struct RunAllGroup {
    const char* title   = nullptr;   // заголовок секции; nullptr — без него
    const char* pick_id = "";        // неизменная часть ID чекбокса ("pbd_", …)
    int         n       = 0;
    std::function<std::string(int)> label;
    std::function<void(int)>        enqueue;   // положить элемент i в очередь
};


void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued);


enum class ParamPlotKind { Bifurcation, LLE, LS, Metrics };


void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind);


bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body);


void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip);


bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix);


void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd);


template <class Cfg>
static void draw_continuation_device_block(Cfg& c, const char* id, bool blocked,
                                           bool device_locked) {
    const std::string sfx = id;
    if (blocked) {
        c.continuation = false;
        c.continuation_reverse = false;
    }

    ImGui::BeginDisabled(blocked);
    bool cont = c.continuation;
    if (ImGui::Checkbox(("Continuation##" + sfx).c_str(), &cont)) {
        c.continuation = cont;
        // При включении continuation по умолчанию уходим на CPU (GPU-ветка там
        // однопоточная), при выключении возвращаемся на GPU. В обе стороны это
        // только дефолт: радио ниже активно, и выбор пользователя держится.
        c.use_gpu = !cont;
    }
    if (c.continuation) {
        ImGui::SameLine();
        int dir = c.continuation_reverse ? 1 : 0;
        ImGui::RadioButton(("forward##"  + sfx).c_str(), &dir, 0); ImGui::SameLine();
        ImGui::RadioButton(("backward##" + sfx).c_str(), &dir, 1);
        c.continuation_reverse = (dir == 1);
    }
    ImGui::EndDisabled();
    if (blocked)
        ImGui::TextDisabled("(continuation: param- or h-sweep only, not IC, not 2D)");

    int dev = c.use_gpu ? 0 : 1;
    ImGui::BeginDisabled(device_locked);
    ImGui::RadioButton(("GPU##dev" + sfx).c_str(), &dev, 0); ImGui::SameLine();
    ImGui::RadioButton(("CPU##dev" + sfx).c_str(), &dev, 1);
    ImGui::EndDisabled();
    if (!device_locked) c.use_gpu = (dev == 0);
    if (c.continuation && c.use_gpu) {
        ImGui::SameLine();
        ImGui::TextDisabled("(continuation on the GPU is single-threaded, CPU is faster)");
    }
    if (!c.use_gpu) {
        std::string why;
        if (!cpu_compiler_available(&why)) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(no compiler: %s)", why.c_str());
        }
    }
}


void apply_snap_x(Plot2DView& view, double lo, double hi, int n);


[[nodiscard]] double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation);


struct SweepAxisMember {
    int    kind      = 0;      // 0 = параметр, 1 = переменная (IC), 2 = шаг h
    int    index     = -1;     // индекс в params/vars (для kind==2 не нужен)
    bool   log_scale = false;
};


void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars);


template <class Cfg>
static void configure_sweep_x_axis_from(Plot2DView& view,
                                        const std::vector<int>& members,
                                        const std::vector<Cfg>& cfgs,
                                        const std::vector<std::string>& params,
                                        const std::vector<std::string>& vars) {
    std::vector<SweepAxisMember> out;
    out.reserve(members.size());
    for (int idx : members) {
        if (idx < 0 || idx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)idx];
        if (!c.last_run_ok) continue;
        SweepAxisMember m;
        m.kind      = c.sweep_over_h ? 2 : (c.sweep_over_var ? 1 : 0);
        m.index     = c.sweep_over_var ? c.var_sweep_index : c.param_index;
        m.log_scale = c.log_scale;
        out.push_back(m);
    }
    configure_sweep_x_axis(view, out, params, vars);
}


void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text);


template <class Cfg>
static void apply_snap_x_from_first_member(Plot2DView& view,
                                           const std::vector<int>& members,
                                           const std::vector<Cfg>& cfgs) {
    view.snap_x_to_grid = true;
    view.snap_x_n       = 0;
    if (members.empty()) return;
    // Первый член С РЕЗУЛЬТАТОМ, а не просто первый: сетку задаёт то, что
    // реально нарисовано, а members[0] может быть ещё не посчитан — тогда
    // снап уходил на поля конфига и дальше на их дефолты.
    const Cfg* pick = nullptr;
    for (int aidx : members) {
        if (aidx < 0 || aidx >= (int)cfgs.size()) continue;
        const Cfg& c = cfgs[(size_t)aidx];
        if (!pick) pick = &c;
        if (c.last_run_ok && c.result.n_pts > 1) { pick = &c; break; }
    }
    if (!pick) return;
    apply_snap_x_from_config(view, pick->last_run_ok,
                             pick->result.param_lo, pick->result.param_hi, pick->result.n_pts,
                             pick->param_lo_text, pick->param_hi_text, pick->n_pts_text);
}


void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb);


static constexpr int kNoExponent = -999;


HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx = kNoExponent);


void invalidate_view(HeatmapView& hv);


void invalidate_view(Plot2DView& v);


void invalidate_view(Plot3DView& v);


template <class K, class V>
static void invalidate_views(std::map<K, std::unique_ptr<V>>& m) {
    for (auto& kv : m) if (kv.second) invalidate_view(*kv.second);
}


bool session_reloaded(const AppModel& model, AppModel::AppMode mode, int& seen);


std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series);


void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation);


void colormap_strip_item(int id, float w, float h);


bool colormap_combo(const char* label, int* id, float width);


struct HeatmapToolbarOpts {
    // Куда сохранить выбор colormap'а (per-config + save_session). Пусто —
    // выбор живёт только в самом view (Custom Basins).
    std::function<void(int)> persist_colormap;
    // Доп. элементы сразу после combo Colormap: LS exponent picker,
    // Basins "Renumber (spiral)".
    std::function<void()>    extras;
    // Доп. элементы в самом конце строки (FastSync Line width / Alpha).
    std::function<void()>    extras_tail;
    // Swap axes осмыслен только там, где оси взаимозаменяемы (2D-свип).
    bool show_swap = true;
};


bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o = {});


void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick);


void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen);

[[nodiscard]] std::string fmt_view_bound(double v, double span) ;


[[nodiscard]] std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) ;


[[nodiscard]] std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) ;


struct ViewRangeValues {
    bool has_x = false; std::string x_lo, x_hi;
    bool has_y = false; std::string y_lo, y_hi;
    const char* source = "view";   // для подписи в стеке отмены
};


[[nodiscard]] ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) ;


[[nodiscard]] ViewRangeValues values_from_target(const ViewRangeTarget& t) ;


void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) ;


void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) ;


[[nodiscard]] bool view_range_bound(const std::vector<ViewRangeTarget>& targets) ;


[[nodiscard]] bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) ;


std::vector<double>& window_node_buf(int window_id) ;
}
