#include "tab_tree.h"
#include "gui_inputs.h"
#include "gui_plot.h"
#include "tab_phase.h"
#include "tab_basins.h"

namespace ucuda::ui_detail {

namespace {

// Вид окна листа: живёт по id узла, а не в слотах, как у Custom.
struct LeafViews {
    std::unique_ptr<HeatmapView>  hv;
    std::unique_ptr<Plot2DView>   pv;
    std::unique_ptr<PlotRenderer> rnd;
    std::vector<SweepCurveCache> bufs;
    BifurcationPointCache bif_points;
    std::optional<unsigned> basins_owner;
    int cmap_seen = -2;
    int exp_seen  = -999;
    int swap_seen = -1;
    int gen_fit   = -1;   // поколение данных, под которое окно уже подогнано
    bool custom_line_style = false;
    float line_width = 1.5f;
    float line_alpha = 1.0f;
};
// Ключ — id листа * 16 + номер окна (0 — основное, 1..15 — дополнительные).
using ViewMap = std::map<int, LeafViews>;
constexpr int kViewsPerLeaf = 16;

auto& hidden_windows() {
    return ui_resource<std::map<std::string, std::unordered_set<std::string>>, 7703>();
}

std::string leaf_window_title(const TreeNode& node, const std::string& suffix, int view = 0) {
    const std::string id = std::to_string(node.id);
    if (view == 0) return node.name + "###tree_" + id + suffix;
    return node.name + " [" + std::to_string(view + 1) + "]###tree_" + id
         + "_v" + std::to_string(view) + suffix;
}

std::string tree_system_suffix(const std::string& system) {
    return system.empty() ? std::string{} : "_sys_" + system;
}

std::string axis_label(const TreeSession& ts, const EffectiveSweep& e) {
    if (e.over_h) return "h";
    if (e.over_var)
        return (e.var_index >= 0 && e.var_index < (int)ts.vars.size()) ? ts.vars[e.var_index] + " (IC)" : "x (IC)";
    return (e.par_index >= 0 && e.par_index < (int)ts.params.size()) ? ts.params[e.par_index] : "param";
}

// Док-узел, где уже стоят окна листьев (обновляется каждый кадр).
ImGuiID g_leaf_dock = 0;

// Новое окно — к другим окнам листьев текущей системы.
void ensure_docked(const std::string& title) {
    if (!g_tree_dockspace_id || ImGui::GetIO().MouseDown[0]) return;
    ImGuiID persisted = 0;
    if (ImGuiWindow* w = ImGui::FindWindowByName(title.c_str())) persisted = w->DockId;
    else if (ImGuiWindowSettings* s = ImGui::FindWindowSettingsByID(ImHashStr(title.c_str())))
        persisted = s->DockId;
    const auto belongs_to_tree = [](ImGuiDockNode* node) {
        return node && node->IsLeafNode()
            && ImGui::DockNodeGetRootNode(node)->ID == g_tree_dockspace_id;
    };
    if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(persisted)) {
        if (belongs_to_tree(node) || !ImGui::DockNodeGetRootNode(node)->IsDockSpace()) return;
    }
    ImGuiID target = 0;
    if (belongs_to_tree(ImGui::DockBuilderGetNode(g_leaf_dock))) target = g_leaf_dock;
    else if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(g_tree_dockspace_id)) {
        while (node->ChildNodes[0]) node = node->ChildNodes[0];
        target = node->ID;
    }
    if (target) ImGui::DockBuilderDockWindow(title.c_str(), target);
}

// Вывести окно листа вперёд, но не в ущерб панели дерева: если окно делит с ней вкладки
// одного узла, фокус на нём спрятал бы дерево.
void focus_leaf_window(const std::string& title, const std::string& controls) {
    ImGuiWindow* w   = ImGui::FindWindowByName(title.c_str());
    ImGuiWindow* ctl = ImGui::FindWindowByName(controls.c_str());
    if (w && ctl && w->DockNode && w->DockNode == ctl->DockNode) return;
    ImGui::SetNextWindowFocus();
}

// Рамка цветом размерности у окна выбранного листа; клик по окну выбирает лист.
void leaf_window_frame(TreeSession& ts, int id, int slot) {
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows)
        && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        ts.tree.selected_id = id;
    if (ts.tree.selected_id != id) return;
    const ImVec2 p = ImGui::GetWindowPos(), s = ImGui::GetWindowSize();
    ImGui::GetWindowDrawList()->PushClipRectFullScreen();
    ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + s.x, p.y + s.y),
        ImGui::GetColorU32(tree_dim_color(slot, true)), 0.0f, 0, 2.0f);
    ImGui::GetWindowDrawList()->PopClipRect();
}

bool begin_leaf_window(TreeSession& ts, const TreeNode& n, const std::string& title, bool* p_open,
                       const std::string& controls, int view_k = 0) {
    ensure_docked(title);
    if (g_tree_focus_leaf == n.id && g_tree_focus_view == view_k) {
        focus_leaf_window(title, controls);
        g_tree_focus_leaf = 0;
    }
    const bool vis = ImGui::Begin(title.c_str(), p_open);
    if (ImGui::GetWindowDockID() != 0) g_leaf_dock = ImGui::GetWindowDockID();
    leaf_window_frame(ts, n.id, tree_kind_dim_slot(n.kind));
    return vis;
}

// exp_store — показатель LS этого окна (у основного — в конфиге, у дополнительных — в узле).
void draw_2d_leaf(AppModel& model, TreeSession& ts, const TreeNode& n, TreeLeafRuntime& rt,
                  LeafViews& v, const GuiCallbacks& cb, int owner_id, bool primary, int* exp_store) {
    if (!v.hv)  v.hv  = std::make_unique<HeatmapView>();
    if (!v.rnd) v.rnd = std::make_unique<PlotRenderer>();
    HeatmapView& hv = *v.hv;

    // Данные и вид из конфига листа: ось и диапазон описывают один прогон.
    bool has = false, busy = false;
    int n_pts = 0, gen = 0;
    const double* values = nullptr;
    double lx = 0, hx = 1, ly = 0, hy = 1, vmin = 0, vmax = 1;
    bool log_x = false, log_y = false;
    const std::vector<int>* flags = nullptr;
    int* cmap = nullptr;
    bool* fit = nullptr;
    LSCurveConfig* ls_cfg = nullptr;
    if (n.kind == TreeNodeKind::Bif2D) {
        auto& d = rt.bif->diagrams[0];
        has = d.last_run_2d_ok && !d.result_2d.values.empty();
        n_pts = d.result_2d.n_pts; values = d.result_2d.values.data();
        lx = d.result_2d.param_lo; hx = d.result_2d.param_hi;
        ly = d.result_2d.param_lo_2; hy = d.result_2d.param_hi_2;
        vmin = d.result_2d.min_val; vmax = d.result_2d.max_val;
        gen = d.data_generation_2d; flags = &d.result_2d.flags;
        log_x = d.log_scale; log_y = d.log_scale_2;
        cmap = &d.colormap_idx; fit = &d.fit_request_2d; busy = rt.bif->in_flight;
    } else if (n.kind == TreeNodeKind::LLE2D) {
        auto& c = rt.lle->curves[0];
        has = c.last_run_2d_ok && !c.result_2d.values.empty();
        n_pts = c.result_2d.n_pts; values = c.result_2d.values.data();
        lx = c.result_2d.param_lo; hx = c.result_2d.param_hi;
        ly = c.result_2d.param_lo_2; hy = c.result_2d.param_hi_2;
        vmin = c.result_2d.min_val; vmax = c.result_2d.max_val;
        gen = c.data_generation_2d; flags = &c.result_2d.flags;
        log_x = c.log_scale; log_y = c.log_scale_2;
        cmap = &c.colormap_idx; fit = &c.fit_request_2d; busy = rt.lle->in_flight;
    } else {
        auto& c = rt.ls->curves[0];
        ls_cfg = &c;
        has = c.last_run_2d_ok && !c.result_2d.values.empty();
        n_pts = c.result_2d.n_pts;
        values = c.result_2d.values.empty() ? nullptr : c.result_2d.values.data();
        lx = c.result_2d.param_lo; hx = c.result_2d.param_hi;
        ly = c.result_2d.param_lo_2; hy = c.result_2d.param_hi_2;
        vmin = c.result_2d.min_val.empty() ? 0.0 : c.result_2d.min_val[0];
        vmax = c.result_2d.max_val.empty() ? 1.0 : c.result_2d.max_val[0];
        gen = c.data_generation_2d; flags = &c.result_2d.flags;
        log_x = c.log_scale; log_y = c.log_scale_2;
        cmap = &c.colormap_idx; fit = &c.fit_request_2d; busy = rt.ls->in_flight;
        if (!exp_store) exp_store = &c.display_exponent_idx;
        if (v.exp_seen != *exp_store) {
            v.exp_seen = *exp_store;
            hv.display_exponent_idx = *exp_store;
        }
    }
    if (v.cmap_seen != *cmap) {
        v.cmap_seen = *cmap;
        hv.colormap = (HeatmapColormap)colormap_id_or(*cmap, model.heatmap_colormap);
    }

    const double* ls_plane = nullptr;
    double ls_vmin = 0.0, ls_vmax = 1.0;
    int ls_gen = gen;
    {
        HeatmapToolbarOpts topts;
        // Колормап доп. окна живёт в его виде: основной хранится в конфиге.
        if (primary) topts.persist_colormap = [&](int cm) { *cmap = cm; v.cmap_seen = cm; ts.dirty = true; };
        if (ls_cfg) topts.extras = [&]() {
            if (!ls_cfg->last_run_2d_ok || ls_cfg->result_2d.n_exponents <= 0) return;
            draw_ls_exponent_picker(hv, ls_cfg->result_2d.n_exponents,
                                    [&](int j) { *exp_store = j; v.exp_seen = j; ts.dirty = true; });
            ls_resolve_plane(*ls_cfg, hv.display_exponent_idx, ls_plane, ls_vmin, ls_vmax, ls_gen);
        };
        draw_heatmap_toolbar(hv, topts);
    }
    if (!has) {
        ImGui::TextDisabled(busy ? "Computing..." : "No data yet. Press Run.");
        return;
    }

    const CustomTabSharedConfig s = ts.effective_shared(n.id);
    hv.x_axis.name = axis_label(ts, effective_sweep_x(s));
    hv.y_axis.name = axis_label(ts, effective_sweep_y(s));
    hv.x_axis.log_scale = log_x;
    hv.y_axis.log_scale = log_y;

    // Крест — курсор региона; перетаскивание двигает его, пересчёт — после отпускания.
    const int region = ts.region_of(n.id);
    if (const TreeNode* r = ts.tree.find(region)) {
        hv.crosshair_x = r->cfg.fix_x_value;
        hv.crosshair_y = r->cfg.fix_y_value;
    }
    auto move = [&ts, region](double sx, double sy) {
        TreeNode* r = ts.tree.find(region);
        if (!r) return;
        r->cfg.fix_x_value = sx;
        r->cfg.fix_y_value = sy;
        tree_cursor_moved(region);
    };
    hv.on_left_drag  = [move](int, int, double sx, double sy) { move(sx, sy); };
    hv.on_left_click = [move](int, int, double sx, double sy) { move(sx, sy); };

    TreeNode* reg = ts.tree.find(region);
    std::vector<ViewRangeTarget> vrt;
    if (reg) vrt.push_back(vr_target_fixed(reg->name, &reg->cfg.axis_x_lo_text, &reg->cfg.axis_x_hi_text,
                                           &reg->cfg.axis_y_lo_text, &reg->cfg.axis_y_hi_text));
    const TreeNodeKind kind = n.kind;
    TreeLeafRuntime* rtp = &rt;
    hv.popup_extras = [kind, rtp, busy, &cb, &vrt, &model, &hv]() {
        draw_export_menu_item(busy, cb, [kind, rtp](const std::string& path) {
            if      (kind == TreeNodeKind::Bif2D) data_export::export_bif2d(rtp->bif->diagrams[0].result_2d, path);
            else if (kind == TreeNodeKind::LLE2D) data_export::export_lle2d(rtp->lle->curves[0].result_2d, path);
            else                                  data_export::export_ls2d(rtp->ls->curves[0].result_2d, path);
        });
        if (!vrt.empty()) draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
    };
    if (flags) {
        hv.cell_regime = regime_cell_label(*flags, n_pts);
        hv.cell_not_computed = cancelled_cell_mask(*flags, n_pts);
    }
    if (kind == TreeNodeKind::Bif2D) {
        hv.value_name = "period";
        hv.value_is_regime_code = true;
    }
    const double* vals = ls_plane ? ls_plane : values;
    // Основное окно гасит запрос конфига; любое окно подгоняется под новое поколение данных.
    const bool take_fit = (primary && *fit) || v.gen_fit != gen;
    if (primary) *fit = false;
    v.gen_fit = gen;
    const ImVec2 avail = ImGui::GetContentRegionAvail(), origin = ImGui::GetCursorScreenPos();
    hv.render(*v.rnd, origin, avail, owner_id, ls_plane ? ls_gen : gen, n_pts, n_pts, vals,
              lx, hx, ly, hy, ls_plane ? ls_vmin : vmin, ls_plane ? ls_vmax : vmax, take_fit);
    if (!vrt.empty()) handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(), hv.swap_axes, &model);
}

void draw_1d_leaf(AppModel& model, TreeSession& ts, const TreeNode& n, TreeLeafRuntime& rt,
                  LeafViews& v, const GuiCallbacks& cb, int owner_id, bool primary) {
    if (!v.pv)  v.pv  = std::make_unique<Plot2DView>();
    Plot2DView& view = *v.pv;
    const ParamPlotKind pkind = n.kind == TreeNodeKind::Bif1D ? ParamPlotKind::Bifurcation
                              : n.kind == TreeNodeKind::LLE1D ? ParamPlotKind::LLE : ParamPlotKind::LS;
    configure_param_plot_view(view, pkind);

    const TreeNode* sl = ts.tree.find(ts.tree.nearest(n.id, TreeNodeKind::Slice1D));
    const int dir = sl ? sl->slice_dir : 0;
    const CustomTabSharedConfig s = ts.effective_shared(n.id);
    const EffectiveSweep own = dir == 0 ? effective_sweep_x(s) : effective_sweep_y(s);
    view.x_axis.name = axis_label(ts, own);

    bool ok = false, busy = false, cont = false, log = false;
    int data_gen = 0;
    double lo = 0, hi = 1;
    std::string n_text;
    bool* fit = nullptr;
    if (n.kind == TreeNodeKind::Bif1D) {
        auto& d = rt.bif->diagrams[0];
        ok = d.last_run_ok; data_gen = d.data_generation; busy = rt.bif->in_flight;
        lo = parse_ratio_or(d.param_lo_text, 0.0); hi = parse_ratio_or(d.param_hi_text, 1.0);
        log = d.log_scale; cont = d.continuation; n_text = d.n_pts_text; fit = &d.fit_request;
        view.y_axis.name = (d.writable_var >= 0 && d.writable_var < (int)ts.vars.size())
                         ? ts.vars[d.writable_var] : "X";
        if (d.plot_inter_peaks) view.y_axis.name += " interval";
        if (draw_point_style_toolbar(d, ("tree" + std::to_string(n.id)).c_str())) ts.dirty = true;
        apply_point_style(view, d);
    } else if (n.kind == TreeNodeKind::LLE1D) {
        auto& c = rt.lle->curves[0];
        ok = c.last_run_ok; data_gen = c.data_generation; busy = rt.lle->in_flight;
        lo = parse_ratio_or(c.param_lo_text, 0.0); hi = parse_ratio_or(c.param_hi_text, 1.0);
        log = c.log_scale; cont = c.continuation; n_text = c.n_pts_text; fit = &c.fit_request;
    } else {
        auto& c = rt.ls->curves[0];
        ok = c.last_run_ok; data_gen = c.data_generation; busy = rt.ls->in_flight;
        lo = parse_ratio_or(c.param_lo_text, 0.0); hi = parse_ratio_or(c.param_hi_text, 1.0);
        log = c.log_scale; cont = c.continuation; n_text = c.n_pts_text; fit = &c.fit_request;
    }
    if (pkind != ParamPlotKind::Bifurcation) {
        draw_style_toolbar("Custom line style", "tree_lyapunov", v.custom_line_style, [&v]() {
            ImGui::SetNextItemWidth(150.0f);
            bool changed = slider_float_typed("Line width##tree_lyapunov", &v.line_width, 0.5f, 12.0f, "%.1f");
            ImGui::SetNextItemWidth(150.0f);
            changed |= slider_float_typed("Alpha##tree_lyapunov", &v.line_alpha, 0.05f, 1.0f, "%.2f");
            return changed;
        });
        if (v.custom_line_style) view.line_thickness_px = v.line_width;
    }
    if (!ok) {
        ImGui::TextDisabled(busy ? "Computing..." : "No data yet. Press Run.");
        return;
    }

    // Оси переставлены (ПКМ «Swap axes»): параметр свипа по вертикали. Флаг — вид окна,
    // он сохраняется в imgui.ini и раскладках; при смене перефит.
    PlotWindowPrefs& wp = plot_window_prefs_current();
    const bool swap = wp.swap_axes;
    if (v.swap_seen != (int)swap) {
        if (v.swap_seen != -1) {
            view.x_axis.user_range = view.y_axis.user_range = false;
            view.view_valid = false;
        }
        v.swap_seen = (int)swap;
    }

    // Ось — весь свип среза, а не bbox данных: крест среза совпадает с осью карты.
    view.x_fit_use_explicit = !swap && hi != lo;
    view.y_fit_use_explicit =  swap && hi != lo;
    view.x_fit_min = view.y_fit_min = std::min(lo, hi);
    view.x_fit_max = view.y_fit_max = std::max(lo, hi);
    view.pad_x = swap;
    view.pad_y = !swap;
    const bool sweep_log = log && std::min(lo, hi) > 0.0;
    view.x_axis.log_scale = !swap && sweep_log;
    view.y_axis.log_scale =  swap && sweep_log;

    // Крест — координата курсора вдоль оси среза.
    const double cross = dir == 0 ? s.fix_x_value : s.fix_y_value;
    view.crosshair_x = swap ? std::numeric_limits<double>::quiet_NaN() : cross;
    view.crosshair_y = swap ? cross : std::numeric_limits<double>::quiet_NaN();
    view.crosshair_along_y = swap;
    view.crosshair_x_color = dir == 1 ? 0xFFFF9028u : 0xFF50A0FFu;
    view.crosshair_y_color = view.crosshair_x_color;

    int snap_n = parse_int_or(n_text, 2);
    if (snap_n < 2) snap_n = 2;
    const double slo = std::min(lo, hi), shi = std::max(lo, hi);
    const double step = (shi - slo) / (double)(snap_n - 1);
    auto snap = [slo, shi, step, snap_n, sweep_log, cont](double w) {
        if (step <= 0.0) return w;
        int i;
        if (sweep_log) {
            if (!(w > 0.0)) return slo;
            const double l0 = std::log10(slo), l1 = std::log10(shi);
            i = (int)std::round((std::log10(w) - l0) / ((l1 - l0) / (double)(snap_n - 1)));
        } else {
            i = (int)std::round((w - slo) / step);
        }
        i = std::clamp(i, 0, snap_n - 1);
        return std::clamp(sweep_value_at(i, snap_n, slo, shi, sweep_log, false, cont), slo, shi);
    };
    apply_snap_x(view, slo, shi, snap_n);
    if (swap) view.snap_x_to_grid = false;   // по X теперь данные; узел свипа ловит snap ниже

    // Перетаскивание креста по срезу двигает курсор владельца вдоль своей оси.
    const int owner = ts.cursor_owner(n.id);
    const bool follows = sl && (sl->follow_cursor || ts.region_of(n.id) == 0);
    if (follows && owner) {
        const bool owner_is_region = ts.region_of(n.id) != 0;
        auto move = [&ts, owner, owner_is_region, dir, snap](double w) {
            TreeNode* o = ts.tree.find(owner);
            if (!o) return;
            const double x = snap(w);
            if (!owner_is_region || dir == 0) o->cfg.fix_x_value = x;
            else                              o->cfg.fix_y_value = x;
            tree_cursor_moved(owner);
        };
        view.on_left_drag  = move;
        view.on_left_click = move;
    } else {
        view.on_left_drag = nullptr;
        view.on_left_click = nullptr;
    }

    std::vector<PlotSeriesInput> series;
    auto& bufs = v.bufs;
    int series_gen = data_gen;
    if (n.kind == TreeNodeKind::Bif1D) {
        auto& d = rt.bif->diagrams[0];
        const auto& r = d.result;
        const auto& src = d.plot_inter_peaks ? r.peak_times : r.bifurcation_points;
        const int np = r.n_pts > 0 ? r.n_pts : 1;
        auto& cache = v.bif_points;
        cache.update(d, {np, (int)src.size(), lo, hi, log, false, swap}, /*finite_only*/ true);
        PlotSeriesInput si;
        si.points = cache.points.data();
        si.n_points = (int)(cache.points.size() / 2);
        si.sorted_x = cache.sorted_x;
        si.color = ic_base_color(0);
        if (d.custom_point_style) si.color.w = d.point_alpha;
        si.label = n.name;
        series.push_back(si);
        series_gen = cache.generation;
    } else if (n.kind == TreeNodeKind::LLE1D) {
        auto& c = rt.lle->curves[0];
        const auto& r = c.result;
        const int np = (int)r.lyapunov.size();
        bufs.resize(1);
        auto& cache = bufs[0];
        if (cache.update({&r, data_gen, np, 0, lo, hi, log, false, c.continuation, swap},
                         [&](std::vector<double>& buf) {
            buf.clear();
            for (int p = 0; p < np; ++p) {
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                const double y = r.lyapunov[p];
                if (!std::isfinite(y)) continue;
                buf.push_back(sweep_value_at(p, np, lo, hi, log, false, c.continuation));
                buf.push_back(y);
            }
            if (swap)
                for (size_t k = 0; k + 1 < buf.size(); k += 2) std::swap(buf[k], buf[k + 1]);
        })) view.series_generation = -1;
        PlotSeriesInput si;
        si.points = cache.data.data();
        si.n_points = (int)(cache.data.size() / 2);
        si.color = ic_base_color(0);
        si.label = n.name;
        series.push_back(si);
    } else {
        auto& c = rt.ls->curves[0];
        const auto& r = c.result;
        const int np = r.n_pts, ne = r.n_exponents;
        if (ne > 0 && np > 0 && (int)r.spectrum.size() == np) {
            bufs.resize(ne);
            for (int j = 0; j < ne; ++j) {
                auto& cache = bufs[j];
                if (cache.update({&r, data_gen, np, j, lo, hi, log, false, c.continuation, swap},
                                 [&](std::vector<double>& buf) {
                    buf.clear();
                    for (int p = 0; p < np; ++p) {
                        if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                        const auto& row = r.spectrum[p];
                        if (j >= (int)row.size() || !std::isfinite(row[j])) continue;
                        buf.push_back(sweep_value_at(p, np, lo, hi, log, false, c.continuation));
                        buf.push_back(row[j]);
                    }
                    if (swap)
                        for (size_t k = 0; k + 1 < buf.size(); k += 2) std::swap(buf[k], buf[k + 1]);
                })) view.series_generation = -1;
                PlotSeriesInput si;
                si.points = cache.data.data();
                si.n_points = (int)(cache.data.size() / 2);
                si.color = ic_base_color(j);
                si.label = ne > 1 ? n.name + " L" + std::to_string(j + 1) : n.name;
                series.push_back(si);
            }
            series_gen = data_gen * 64 + ne;
        }
    }
    if (pkind != ParamPlotKind::Bifurcation && v.custom_line_style) {
        for (auto& curve : series) curve.color.w = v.line_alpha;
    }
    std::vector<bool> init_vis(series.size(), true), glob_vis(series.size(), true);
    if (swap) {

        std::swap(view.x_axis.name, view.y_axis.name);
        view.show_zero_x = view.show_zero_y;
        view.show_zero_y = false;
        series_gen = series_gen * 2 + 1;   // другая раскладка данных — новая заливка VBO
    } else {
        series_gen = series_gen * 2;
    }

    const TreeNodeKind kind = n.kind;
    TreeLeafRuntime* rtp = &rt;
    view.popup_extras = [kind, rtp, busy, &cb, swap, &wp]() {
        bool sw = swap;
        if (ImGui::MenuItem("Swap axes", nullptr, &sw)) {
            wp.swap_axes = sw;   // перефит — на следующем кадре, по смене флага
            ImGui::MarkIniSettingsDirty();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sweep parameter on the vertical axis (e.g. a Y slice\n"
                              "next to the vertical axis of the 2D map).");
        draw_export_menu_item(busy, cb, [kind, rtp](const std::string& path) {
            if      (kind == TreeNodeKind::Bif1D) data_export::export_bif1d(rtp->bif->diagrams[0].result, path);
            else if (kind == TreeNodeKind::LLE1D) data_export::export_lle1d(rtp->lle->curves[0].result, path);
            else                                  data_export::export_ls1d(rtp->ls->curves[0].result, path);
        });
    };
    const bool take_fit = (primary && *fit) || v.gen_fit != data_gen;
    if (primary) *fit = false;
    v.gen_fit = data_gen;
    if (!v.rnd) v.rnd = std::make_unique<PlotRenderer>();
    const ImVec2 avail = ImGui::GetContentRegionAvail(), origin = ImGui::GetCursorScreenPos();
    view.render(*v.rnd, origin, avail, owner_id, series_gen, series, init_vis, glob_vis, take_fit);
    (void)model;
}

} // namespace

std::string tree_controls_title(const std::string& system) {
    return "Tree Controls###tree_controls_sys_" + system;
}

static std::vector<std::pair<int, std::string>> tree_layout_entries(const AppModel& model) {
    const auto& ts = model.tree_session;
    const std::string suffix = tree_system_suffix(model.loaded_name);
    std::vector<std::pair<int, std::string>> windows;
    const auto append_projections = [&windows](int id, const PhaseAnalysisSession& phase, const std::string& tail) {
        for (int i = 0; i < (int)phase.projections.size(); ++i)
            windows.emplace_back(id, projection_window_title(phase, i, tail));
    };
    for (int id : ts.tree.preorder()) {
        const auto* node = ts.tree.find(id);
        if (!node || tree_kind_is_context(node->kind)) continue;
        const auto it = ts.leaves.find(id);
        if (it == ts.leaves.end()) continue;
        const auto& rt = it->second;
        if (node->kind == TreeNodeKind::Phase) {
            if (rt.phase) append_projections(id, *rt.phase, "_tree" + std::to_string(id) + suffix);
            continue;
        }
        windows.emplace_back(id, leaf_window_title(*node, suffix));
        for (int k = 1; k <= (int)node->extra_views.size() && k < kViewsPerLeaf; ++k)
            windows.emplace_back(id, leaf_window_title(*node, suffix, k));
        if (node->kind == TreeNodeKind::Basins && rt.basins && !rt.basins->phase_slots.empty()) {
            const auto& slot = rt.basins->phase_slots[0];
            if (slot && (slot->has_result || slot->phase.in_flight))
                append_projections(id, slot->phase, "_treebpp" + std::to_string(id) + suffix);
        }
    }
    return windows;
}

std::vector<std::string> tree_layout_windows(const AppModel& model) {
    std::vector<std::string> names;
    for (auto& entry : tree_layout_entries(model)) names.push_back(std::move(entry.second));
    return names;
}

void apply_tree_layout_windows(AppModel& model, const std::vector<std::string>& unmatched) {
    auto& ts = model.tree_session;
    const auto entries = tree_layout_entries(model);
    const std::unordered_set<std::string> excluded(unmatched.begin(), unmatched.end());
    std::unordered_set<int> enabled;
    for (const auto& [id, name] : entries) {
        if (excluded.count(name)) continue;
        for (const auto* node = ts.tree.find(id); node && enabled.insert(node->id).second;
             node = ts.tree.find(node->parent)) {}
    }

    for (const auto& node : ts.tree.nodes) {
        const bool in_layout = enabled.count(node.id) != 0;
        if (!in_layout && tree_kind_is_context(node.kind)) continue;
        if (node.enabled != in_layout) ts.toggle_enabled(node.id);
        if (!in_layout) {
            const auto it = ts.leaves.find(node.id);
            if (it != ts.leaves.end() && it->second.in_flight()) it->second.request_cancel();
        }
    }

    // Дополнительные представления одного расчёта не имеют собственного ON/OFF.
    std::vector<std::string> hidden_views;
    for (const auto& [id, name] : entries) {
        if (enabled.count(id) && excluded.count(name)) hidden_views.push_back(name);
    }
    tree_hide_layout_windows(model.loaded_name, hidden_views);
    g_tree_focus_leaf = 0;
    g_tree_focus_view = 0;
}

void tree_hide_layout_windows(const std::string& system, const std::vector<std::string>& windows) {
    if (windows.empty()) hidden_windows().erase(system);
    else hidden_windows()[system] = std::unordered_set<std::string>(windows.begin(), windows.end());
}

bool tree_has_hidden_windows(const std::string& system) {
    const auto& systems = hidden_windows();
    const auto it = systems.find(system);
    return it != systems.end() && !it->second.empty();
}

void tree_forget_views(int leaf_id, int from_view) {
    auto& views = ui_resource<ViewMap, 7701>();
    for (int k = std::max(1, from_view); k < kViewsPerLeaf; ++k) views.erase(leaf_id * kViewsPerLeaf + k);
}

void draw_tree_plot_windows(AppModel& model, const GuiCallbacks& cb) {
    auto& ts = model.tree_session;
    if (ts.tree.nodes.empty()) return;
    auto& views = ui_resource<ViewMap, 7701>();
    static int seen_epoch = 0;
    if (session_reloaded(model, AppModel::AppMode::Tree, seen_epoch)) {
        for (auto& [id, v] : views) {
            if (v.hv) invalidate_view(*v.hv);
            if (v.pv) invalidate_view(*v.pv);
            v.bufs.clear();
            v.bif_points = {};
            v.cmap_seen = -2;
            v.exp_seen = -999;
            if (v.basins_owner) {
                forget_basins_plot(*v.basins_owner);
                v.basins_owner.reset();
            }
        }
    }
    for (auto it = views.begin(); it != views.end();) {
        const TreeNode* vn = ts.tree.find(it->first / kViewsPerLeaf);
        const int k = it->first % kViewsPerLeaf;
        if (!vn || k > (int)vn->extra_views.size() ||
            (it->second.basins_owner && vn->kind != TreeNodeKind::Basins)) {
            if (it->second.basins_owner) forget_basins_plot(*it->second.basins_owner);
            it = views.erase(it);
        } else {
            ++it;
        }
    }

    const std::string& sys = model.loaded_name;
    const std::string sys_sfx = tree_system_suffix(sys);
    const std::string controls = tree_controls_title(sys);
    const auto hidden_it = hidden_windows().find(sys);
    auto* hidden = hidden_it == hidden_windows().end() ? nullptr : &hidden_it->second;
    const auto visible = [hidden](const std::string& title) {
        return !hidden || hidden->count(title) == 0;
    };
    const int sys_delta = sys.empty() ? 0 : (int)(ImHashStr(sys.c_str()) & 0x00FFFFFF);

    for (int id : ts.tree.preorder()) {
        const TreeNode* n = ts.tree.find(id);
        if (!n || tree_kind_is_context(n->kind) || !ts.tree.is_active(id)) continue;
        auto rit = ts.leaves.find(id);
        if (rit == ts.leaves.end()) continue;
        TreeLeafRuntime& rt = rit->second;
        const int owner_id = ((0x7E000 + id) << 4) ^ sys_delta;

        if (n->kind == TreeNodeKind::Phase) {
            if (!rt.phase) continue;
            const bool focus = g_tree_focus_leaf == id;
            const std::string suffix = "_tree" + std::to_string(id) + sys_sfx;
            if (focus && hidden && !rt.phase->projections.empty())
                hidden->erase(projection_window_title(*rt.phase, 0, suffix));
            if (focus) g_tree_focus_leaf = 0;
            const int slot = tree_kind_dim_slot(n->kind);
            const size_t n_before = rt.phase->projections.size();
            draw_projection_windows(*rt.phase, cb,
                [focus, &controls](int idx, const std::string& title) {
                    ensure_docked(title);
                    if (focus && idx == 0) focus_leaf_window(title, controls);
                },
                [&ts, id, slot](int, const std::string&) {
                    if (ImGui::GetWindowDockID() != 0) g_leaf_dock = ImGui::GetWindowDockID();
                    leaf_window_frame(ts, id, slot);
                },
                suffix, owner_id, {}, {}, visible);
            // Закрыли последнее окно ФП — узел выключается, а проекция возвращается, чтобы
            // включение узла снова показало окно.
            if (rt.phase->projections.empty() && n_before > 0) {
                rt.phase->add_projection();
                for (auto& pr : rt.phase->projections) pr.label = n->name + ": " + pr.label;
                ts.toggle_enabled(id);
            } else if (rt.phase->projections.size() != n_before) {
                ts.dirty = true;
            }
            continue;
        }

        const std::string title = leaf_window_title(*n, sys_sfx);
        if (hidden && g_tree_focus_leaf == id && g_tree_focus_view == 0) hidden->erase(title);
        // Крестик основного окна выключает лист: включение в дереве вернёт окно.
        bool main_open = true;
        if (visible(title)) {
            if (begin_leaf_window(ts, *n, title, &main_open, controls)) {
                ImGui::PushID(id);
                LeafViews& v = views[id * kViewsPerLeaf];
                if (tree_kind_is_2d_leaf(n->kind))      draw_2d_leaf(model, ts, *n, rt, v, cb, owner_id, true, nullptr);
                else if (tree_kind_is_1d_leaf(n->kind)) draw_1d_leaf(model, ts, *n, rt, v, cb, owner_id, true);
                else if (n->kind == TreeNodeKind::Basins && rt.basins) {
                    const auto basins_owner = static_cast<unsigned>(owner_id);
                    if (v.basins_owner && *v.basins_owner != basins_owner)
                        forget_basins_plot(*v.basins_owner);
                    v.basins_owner = basins_owner;
                    draw_basins_plot(model, *rt.basins, 0, cb, basins_owner, [&ts] { ts.dirty = true; });
                }
                ImGui::PopID();
            }
            ImGui::End();
        }
        if (!main_open) {
            ts.toggle_enabled(id);
            continue;
        }

        // Дополнительные окна того же результата; крестик окна убирает его.
        int close_k = -1;
        TreeNode* nm = ts.tree.find(id);
        for (int k = 1; nm && k <= (int)nm->extra_views.size() && k < kViewsPerLeaf; ++k) {
            const std::string vt = leaf_window_title(*nm, sys_sfx, k);
            if (hidden && g_tree_focus_leaf == id && g_tree_focus_view == k) hidden->erase(vt);
            if (!visible(vt)) continue;
            bool open = true;
            if (begin_leaf_window(ts, *nm, vt, &open, controls, k)) {
                ImGui::PushID(id * kViewsPerLeaf + k);
                LeafViews& v = views[id * kViewsPerLeaf + k];
                if (tree_kind_is_2d_leaf(nm->kind))
                    draw_2d_leaf(model, ts, *nm, rt, v, cb, owner_id + k, false, &nm->extra_views[k - 1]);
                else if (tree_kind_is_1d_leaf(nm->kind))
                    draw_1d_leaf(model, ts, *nm, rt, v, cb, owner_id + k, false);
                ImGui::PopID();
            }
            ImGui::End();
            if (!open) close_k = k;
        }
        if (nm && close_k > 0) {
            nm->extra_views.erase(nm->extra_views.begin() + (close_k - 1));
            // Окна правее сдвигаются на номер: их виды начинают заново.
            for (int k = close_k; k < kViewsPerLeaf; ++k) views.erase(id * kViewsPerLeaf + k);
            ts.dirty = true;
        }

        // Портреты по бассейнам — окна вкладки Basins для этой сессии.
        // Окна появляются, когда портреты посчитаны или считаются: пустые не загромождают док.
        if (n->kind == TreeNodeKind::Basins && rt.basins && !rt.basins->phase_slots.empty()
            && rt.basins->phase_slots[0]
            && (rt.basins->phase_slots[0]->has_result || rt.basins->phase_slots[0]->phase.in_flight)) {
            const int slot = tree_kind_dim_slot(n->kind);
            draw_basins_phase_windows(model, *rt.basins, 0, cb,
                "_treebpp" + std::to_string(id) + sys_sfx, ((0x7E800 + id) << 4) ^ sys_delta,
                [](int, const std::string& t) { ensure_docked(t); },
                [&ts, id, slot](int, const std::string&) {
                    if (ImGui::GetWindowDockID() != 0) g_leaf_dock = ImGui::GetWindowDockID();
                    leaf_window_frame(ts, id, slot);
                }, static_cast<unsigned>(owner_id), visible);
        }
    }
}

} // namespace ucuda::ui_detail
