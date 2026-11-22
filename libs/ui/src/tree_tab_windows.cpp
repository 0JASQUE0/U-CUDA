#include "tab_tree.h"
#include "gui_inputs.h"
#include "gui_plot.h"
#include "tab_phase.h"
#include "tab_basins.h"
#include <iomanip>
#include <sstream>

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
    int display_exponent = -1;
    int swap_seen = -1;
    int gen_fit   = -1;   // поколение данных, под которое окно уже подогнано
    bool custom_line_style = false;
    float line_width = 1.5f;
    float line_alpha = 1.0f;
};
// Ключ — id листа * 16 + номер окна (0 — основное, 1..15 — дополнительные).
using ViewMap = std::map<int, LeafViews>;
constexpr int kViewsPerLeaf = kTreeMaxExtraViews + 1;

auto& tree_views_by_tab() {
    return ui_resource<std::map<int, ViewMap>, 7701>();
}

std::string leaf_window_title(const TreeSession& session, const TreeNode& node,
                              const std::string& suffix, int view) {
    std::string label = node.name;
    const std::string default_name = tree_kind_title(node.kind);
    if (label == default_name) {
        switch (node.kind) {
        case TreeNodeKind::Bif1D: label = "Bif 1D"; break;
        case TreeNodeKind::Bif2D: label = "Bif 2D"; break;
        case TreeNodeKind::LS1D:  label = "LS 1D";  break;
        case TreeNodeKind::LS2D:  label = "LS 2D";  break;
        default: break;
        }
    }
    const auto sweep = session.sweep_window_label(node.id);
    if (!sweep.empty()) label += ": " + sweep;
    if (view > 0) label += " [" + std::to_string(view + 1) + "]";
    if (session.tree.active_tab().is_copy(tree_leaf_window_key(node.id, view))) label += " (copy)";
    const std::string id = "###tree_" + std::to_string(node.id);
    return label + id + (view > 0 ? "_v" + std::to_string(view) : "") + suffix;
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
    if (w && ctl && w->DockNode && w->DockNode == ctl->DockNode &&
        w->LastFrameActive >= ImGui::GetFrameCount() - 1) return;
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
    if (g_tree_focus_leaf == n.id && g_tree_focus_type == 'w' && g_tree_focus_view == view_k) {
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
    if (v.cmap_seen == -2 || (primary && v.cmap_seen != *cmap)) {
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
    hv.x_axis.name = ts.sweep_axis_label(effective_sweep_x(s));
    hv.y_axis.name = ts.sweep_axis_label(effective_sweep_y(s));
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
    view.x_axis.name = ts.sweep_axis_label(own);

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
            item_w(150.0f);
            bool changed = slider_float_typed("Line width##tree_lyapunov", &v.line_width, 0.5f, 12.0f, "%.1f");
            item_w(150.0f);
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

std::string tree_tab_suffix(const std::string& system, int tab) {
    const std::string scope = tab == 1 ? "" : "_tab_" + std::to_string(tab);
    return scope + (system.empty() ? "" : "_sys_" + system);
}

std::string tree_controls_title(const std::string& system, int tab) {
    return "Tree Controls###tree_controls" + (system.empty() && tab == 1 ? "_sys_" : tree_tab_suffix(system, tab));
}

void tree_forget_extra_views(int leaf, int first_view) {
    for (auto& [tab, views] : tree_views_by_tab())
        for (int index = first_view; index < kViewsPerLeaf; ++index)
            views.erase(leaf * kViewsPerLeaf + index);
}

namespace {
struct TreeWindowEntry {
    int leaf;
    char type;
    int index;
    std::string title;

    std::string key() const { return tree_leaf_window_key(leaf, index, type); }
};

std::string projection_suffix(int leaf, char type, const std::string& suffix) {
    return (type == 'p' ? "_tree" : "_treebpp") + std::to_string(leaf) + suffix;
}

std::vector<TreeWindowEntry> tree_layout_entries(const AppModel& model) {
    const auto& ts = model.tree_session;
    const auto suffix = tree_tab_suffix(model.loaded_name, ts.tree.active_tab().id);
    std::vector<TreeWindowEntry> windows;
    const auto append_projections = [&](int id, const PhaseAnalysisSession& phase, char type) {
        const auto tail = projection_suffix(id, type, suffix);
        for (int i = 0; i < static_cast<int>(phase.projections.size()); ++i)
            windows.push_back({id, type, i, projection_window_title(phase, i, tail)});
    };
    for (int id : ts.tree.preorder()) {
        const auto* node = ts.tree.find(id);
        if (!node || tree_kind_is_context(node->kind)) continue;
        const auto it = ts.leaves.find(id);
        if (it == ts.leaves.end()) continue;
        const auto& rt = it->second;
        if (node->kind == TreeNodeKind::Phase) {
            if (rt.phase) append_projections(id, *rt.phase, 'p');
            continue;
        }
        for (int k = 0; k <= static_cast<int>(node->extra_views.size()) && k < kViewsPerLeaf; ++k)
            windows.push_back({id, 'w', k, leaf_window_title(ts, *node, suffix, k)});
        if (rt.basins && !rt.basins->phase_slots.empty()) {
            const auto& slot = rt.basins->phase_slots[0];
            if (slot) append_projections(id, slot->phase, 'b');
        }
    }
    return windows;
}
} // namespace

std::vector<std::string> tree_layout_windows(const AppModel& model) {
    std::vector<std::string> names;
    for (auto& entry : tree_layout_entries(model)) names.push_back(std::move(entry.title));
    return names;
}

std::string describe_tree_layout_window(const AppModel& model, const std::string& title) {
    for (const auto& entry : tree_layout_entries(model)) {
        if (ImHashStr(entry.title.c_str()) != ImHashStr(title.c_str())) continue;
        std::ostringstream out;
        out << "tree1 " << entry.type << ' ' << entry.index;
        for (const auto& part : model.tree_session.tree.layout_path(entry.leaf))
            out << ' ' << tree_kind_key(part.kind) << ' ' << part.slice_dir << ' ' << part.occurrence
                << ' ' << std::quoted(part.name);
        return out.str();
    }
    return {};
}

std::string resolve_tree_layout_window(AppModel& model, const std::string& descriptor) {
    std::istringstream in(descriptor);
    std::string version;
    char type = 0;
    int index = 0;
    if (!(in >> version >> type >> index) || version != "tree1" || index < 0 ||
        (type == 'w' ? index >= kViewsPerLeaf : index >= 64)) return {};
    std::vector<TreeLayoutNode> path;
    while (in >> std::ws && !in.eof()) {
        TreeLayoutNode part{};
        std::string kind;
        if (path.size() >= 64 || !(in >> kind >> part.slice_dir >> part.occurrence >> std::quoted(part.name)) ||
            !tree_kind_from_key(kind, part.kind)) return {};
        path.push_back(std::move(part));
    }
    if (path.empty()) return {};
    const auto kind = path.back().kind;
    if (type == 'w') {
        if (tree_kind_is_context(kind) || kind == TreeNodeKind::Phase ||
            (index > 0 && !tree_kind_is_1d_leaf(kind) && !tree_kind_is_2d_leaf(kind))) return {};
    } else if (!((type == 'p' && kind == TreeNodeKind::Phase) ||
                 (type == 'b' && kind == TreeNodeKind::Basins))) return {};

    auto& ts = model.tree_session;
    const auto existing = tree_layout_entries(model);
    const int id = ts.ensure_layout_leaf(path);
    if (!id) return {};
    auto& node = *ts.tree.find(id);
    auto& rt = ts.leaves.at(id);
    if (type == 'w' && index > static_cast<int>(node.extra_views.size())) {
        node.extra_views.resize(index, rt.ls ? rt.ls->curves[0].display_exponent_idx : 0);
        ts.dirty = true;
    }
    PhaseAnalysisSession* phase = rt.phase.get();
    if (type == 'b') phase = &rt.basins->phase_slot(0).phase;
    if (phase) {
        while (index >= static_cast<int>(phase->projections.size())) {
            phase->add_projection();
            phase->projections.back().label = node.name + ": " + phase->projections.back().label;
            ts.dirty = true;
        }
    }

    std::string title;
    for (const auto& entry : tree_layout_entries(model)) {
        if (entry.leaf == id && entry.type == type && entry.index == index) title = entry.title;
        if (std::none_of(existing.begin(), existing.end(), [&](const auto& old) {
                return old.leaf == entry.leaf && old.type == entry.type && old.index == entry.index;
            })) {
            for (auto& tab : ts.tree.tabs)
                if (tab.id != ts.tree.active_tab_id && tab.show_enabled_windows)
                    ts.dirty |= tab.set_visible(entry.key(), false);
        }
    }
    return title;
}

void apply_tree_layout_windows(AppModel& model, const std::vector<std::string>& matched) {
    auto& ts = model.tree_session;
    auto& tab = ts.tree.active_tab();
    tab.show_enabled_windows = false;
    tab.shown_windows.clear();
    tab.hidden_windows.clear();
    const std::unordered_set<std::string> placed(matched.begin(), matched.end());
    for (const auto& entry : tree_layout_entries(model)) {
        if (!placed.count(entry.title)) continue;
        ts.tree.register_layout_window(entry.key());
        tab.set_visible(entry.key(), true);
    }
    ts.dirty = true;
    g_tree_focus_leaf = 0;
    g_tree_focus_view = 0;
}

void draw_tree_plot_windows(AppModel& model, const GuiCallbacks& cb) {
    auto& ts = model.tree_session;
    if (ts.tree.nodes.empty()) return;

    auto& by_tab = tree_views_by_tab();
    static int seen_epoch = 0;
    static std::string seen_system;
    if (session_reloaded(model, AppModel::AppMode::Tree, seen_epoch) || seen_system != model.loaded_name) {
        for (auto& [tab, views] : by_tab)
            for (auto& [id, view] : views)
                if (view.basins_owner) forget_basins_plot(*view.basins_owner);
        by_tab.clear();
        seen_system = model.loaded_name;
    }
    const auto& tab = ts.tree.active_tab();
    auto& views = by_tab[tab.id];
    for (auto it = by_tab.begin(); it != by_tab.end();) {
        if (std::any_of(ts.tree.tabs.begin(), ts.tree.tabs.end(), [&](const auto& t) { return t.id == it->first; })) {
            ++it;
        } else {
            for (const auto& [id, view] : it->second)
                if (view.basins_owner) forget_basins_plot(*view.basins_owner);
            it = by_tab.erase(it);
        }
    }
    for (auto it = views.begin(); it != views.end();) {
        const auto* node = ts.tree.find(it->first / kViewsPerLeaf);
        if (!node || it->first % kViewsPerLeaf > static_cast<int>(node->extra_views.size())) {
            if (it->second.basins_owner) forget_basins_plot(*it->second.basins_owner);
            it = views.erase(it);
        } else {
            ++it;
        }
    }

    const auto& sys = model.loaded_name;
    const std::string suffix = tree_tab_suffix(sys, tab.id);
    const std::string controls = tree_controls_title(sys, tab.id);

    for (int id : ts.tree.preorder()) {
        auto* node = ts.tree.find(id);
        if (!node || tree_kind_is_context(node->kind)) continue;
        auto rit = ts.leaves.find(id);
        if (rit == ts.leaves.end()) continue;
        auto& rt = rit->second;
        const int slot = tree_kind_dim_slot(node->kind);
        const int owner_id = static_cast<int>(ImHashStr(("tree_" + std::to_string(id) + suffix).c_str()));
        const auto visible = [&](char type, int index, bool available = true) {
            return ts.workspace_window_visible(id, tree_leaf_window_key(id, index, type), available);
        };
        const auto close = [&](char type, int index) {
            const auto key = tree_leaf_window_key(id, index, type);
            auto& workspace = ts.tree.active_tab();
            ts.dirty |= workspace.remember_window(key);
            ts.dirty |= workspace.set_visible(key, false);
        };
        const auto after_begin = [&ts, id, slot](int, const std::string&) {
            if (ImGui::GetWindowDockID() != 0) g_leaf_dock = ImGui::GetWindowDockID();
            leaf_window_frame(ts, id, slot);
        };
        const auto prepare_projection = [&](char type, int index, const std::string& title) {
            ensure_docked(title);
            if (g_tree_focus_leaf == id && g_tree_focus_type == type && g_tree_focus_view == index) {
                focus_leaf_window(title, controls);
                g_tree_focus_leaf = 0;
            }
        };

        if (rt.phase) {
            const auto tail = projection_suffix(id, 'p', suffix);
            draw_projection_windows(*rt.phase, cb,
                [&](int idx, const std::string& title) { prepare_projection('p', idx, title); },
                after_begin, tail, owner_id, {}, {},
                [&](int index, const std::string&) { return visible('p', index); },
                [&](int index, const std::string&) { close('p', index); });
            continue;
        }

        for (int k = 0; k <= static_cast<int>(node->extra_views.size()) && k < kViewsPerLeaf; ++k) {
            if (!visible('w', k)) continue;
            bool open = true;
            const auto title = leaf_window_title(ts, *node, suffix, k);
            if (begin_leaf_window(ts, *node, title, &open, controls, k)) {
                const int key = id * kViewsPerLeaf + k;
                ImGui::PushID(key);
                auto& v = views[key];
                const bool copy = tab.is_copy(tree_leaf_window_key(id, k));
                const bool primary = k == 0 && tab.id == 1 && !copy;
                const int view_owner = k == 0 ? owner_id : static_cast<int>(ImHashStr(title.c_str()));
                if (tree_kind_is_2d_leaf(node->kind)) {
                    const int exponent = k ? node->extra_views[k - 1] :
                        (rt.ls ? rt.ls->curves[0].display_exponent_idx : -1);
                    if (v.exp_seen == -999) v.display_exponent = exponent;
                    int* exp_store = k && tab.id == 1 && !copy ? &node->extra_views[k - 1] :
                        (primary && rt.ls ? &rt.ls->curves[0].display_exponent_idx : &v.display_exponent);
                    draw_2d_leaf(model, ts, *node, rt, v, cb, view_owner, primary, exp_store);
                } else if (tree_kind_is_1d_leaf(node->kind)) {
                    draw_1d_leaf(model, ts, *node, rt, v, cb, view_owner, primary);
                } else if (rt.basins) {
                    const auto basins_owner = static_cast<unsigned>(owner_id);
                    v.basins_owner = basins_owner;
                    draw_basins_plot(model, *rt.basins, 0, cb, basins_owner, [&ts] { ts.dirty = true; });
                }
                ImGui::PopID();
            }
            ImGui::End();
            if (!open) close('w', k);
        }

        if (!rt.basins || rt.basins->phase_slots.empty() || !rt.basins->phase_slots[0]) continue;
        const auto& phase_slot = *rt.basins->phase_slots[0];
        const bool has_result = phase_slot.has_result || phase_slot.phase.in_flight;
        const auto tail = projection_suffix(id, 'b', suffix);
        draw_basins_phase_windows(model, *rt.basins, 0, cb, tail, static_cast<int>(ImHashStr(tail.c_str())),
            [&](int idx, const std::string& title) { prepare_projection('b', idx, title); }, after_begin,
            static_cast<unsigned>(owner_id),
            [&](int index, const std::string&) { return visible('b', index, has_result); },
            [&](int index, const std::string&) { close('b', index); });
    }
    // Focus requests are scoped to this frame and workspace.
    g_tree_focus_leaf = 0;
    g_tree_focus_view = 0;
}

} // namespace ucuda::ui_detail
