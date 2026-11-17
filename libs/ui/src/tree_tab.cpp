#include "tab_tree.h"
#include "gui_fields.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_plot.h"
#include "tab_phase.h"
#include "tab_basins.h"
#include "ucuda/storage/system_library.h"

namespace ucuda::ui_detail {

namespace {

std::map<int, double>& cursor_timers() {
    static std::map<int, double> t;
    return t;
}

// Сообщение под кнопкой Run (отказы операций), гаснет через 3 с.
std::string g_msg;
double      g_msg_time = -100.0;
void flash(const std::string& m) { g_msg = m; g_msg_time = ImGui::GetTime(); }

std::string axis_name(const TreeSession& ts, bool over_var, int par_i, int var_i, bool over_h) {
    if (over_h) return "h";
    if (over_var)
        return (var_i >= 0 && var_i < (int)ts.vars.size()) ? ts.vars[var_i] + " (IC)" : "x (IC)";
    return (par_i >= 0 && par_i < (int)ts.params.size()) ? ts.params[par_i] : "param";
}

std::string fmt_short(double v) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.4g", v);
    return b;
}

// Короткая подпись справа от имени: что этот узел задаёт.
std::string node_meta(const TreeSession& ts, const TreeNode& n) {
    const auto& c = n.cfg;
    switch (n.kind) {
    case TreeNodeKind::System:
        return c.scheme;
    case TreeNodeKind::Region2D:
        return "(" + axis_name(ts, c.axis_x_over_var, c.axis_x_par_index, c.axis_x_var_index, c.axis_x_over_h)
             + ", " + axis_name(ts, c.axis_y_over_var, c.axis_y_par_index, c.axis_y_var_index, c.axis_y_over_h)
             + ") " + c.resolution_text + "^2";
    case TreeNodeKind::Slice1D: {
        const TreeNode* reg = ts.tree.find(ts.region_of(n.id));
        if (reg && c.inherit_sweep_from_2d) {
            const auto& r = reg->cfg;
            const bool x = n.slice_dir == 0;
            return "along " + axis_name(ts, x ? r.axis_x_over_var : r.axis_y_over_var,
                                        x ? r.axis_x_par_index : r.axis_y_par_index,
                                        x ? r.axis_x_var_index : r.axis_y_var_index,
                                        x ? r.axis_x_over_h : r.axis_y_over_h)
                 + ", N " + c.n_x_1d_text;
        }
        return "along " + axis_name(ts, c.sweep_x_over_var, c.sweep_x_par_index, c.sweep_x_var_index,
                                    c.sweep_x_over_h)
             + " [" + c.sweep_x_lo_text + ", " + c.sweep_x_hi_text + "], N " + c.n_x_1d_text;
    }
    case TreeNodeKind::Point: {
        const TreeNode* reg = ts.tree.find(ts.region_of(n.id));
        if (!reg) return "System values";
        // Что именно точка закрепляет: имена осей региона и их значения.
        const auto& r   = reg->cfg;
        const auto& src = n.follow_cursor ? r : c;
        return axis_name(ts, r.axis_x_over_var, r.axis_x_par_index, r.axis_x_var_index, r.axis_x_over_h)
             + " = " + fmt_short(src.fix_x_value) + ", "
             + axis_name(ts, r.axis_y_over_var, r.axis_y_par_index, r.axis_y_var_index, r.axis_y_over_h)
             + " = " + fmt_short(src.fix_y_value) + (n.follow_cursor ? " (cursor)" : " (pinned)");
    }
    default:
        return {};
    }
}

enum class Act { None, Add, Dup, Del, Toggle, Up, Down, RunNode, NewView, CloseView };
struct Pending { Act act = Act::None; int id = 0; TreeNodeKind kind = TreeNodeKind::System; int dir = 0; };

int  g_rename_id = 0;
char g_rename_buf[128] = {};

void add_menu_items(const TreeNode& n, int child_level, Pending& pend) {
    using K = TreeNodeKind;
    auto item = [&](const char* label, K k, int dir, bool branch) {
        const ImVec4 col = tree_dim_color(tree_kind_dim_slot(k), true);
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        const bool picked = ImGui::Selectable(label);
        ImGui::PopStyleColor();
        ImGui::SameLine(220.0f);
        ImGui::TextDisabled(branch ? "branch" : "leaf");
        if (picked) { pend = { Act::Add, n.id, k, dir }; ImGui::CloseCurrentPopup(); }
    };
    ImGui::TextDisabled("Add to %s -> L%d", n.name.c_str(), child_level);
    ImGui::Separator();
    switch (n.kind) {
    case K::System:
        item("Region 2D", K::Region2D, 0, true);
        item("Slice 1D", K::Slice1D, 0, true);
        item("Point", K::Point, 0, true);
        break;
    case K::Region2D:
        item("Bifurcation 2D", K::Bif2D, 0, false);
        item("LLE 2D", K::LLE2D, 0, false);
        item("Lyapunov Spectrum 2D", K::LS2D, 0, false);
        ImGui::Separator();
        item("Slice 1D X", K::Slice1D, 0, true);
        item("Slice 1D Y", K::Slice1D, 1, true);
        item("Point", K::Point, 0, true);
        break;
    case K::Slice1D:
        item("Bifurcation 1D", K::Bif1D, 0, false);
        item("LLE 1D", K::LLE1D, 0, false);
        item("Lyapunov Spectrum 1D", K::LS1D, 0, false);
        ImGui::Separator();
        item("Point", K::Point, 0, true);
        break;
    case K::Point:
        item("Phase", K::Phase, 0, false);
        item("Basins", K::Basins, 0, false);
        break;
    default: break;
    }
}

void draw_status_dot(AppModel& model, const TreeSession& ts, int id) {
    auto it = ts.leaves.find(id);
    const float r = ImGui::GetTextLineHeight() * 0.22f;
    ImGui::Dummy(ImVec2(r * 2.0f + 2.0f, ImGui::GetTextLineHeight()));
    if (it == ts.leaves.end()) return;
    const auto& q = model.tree_queue();
    const bool queued = std::find(q.begin(), q.end(), id) != q.end();
    ImU32 col;
    const char* tip;
    if (it->second.in_flight()) {
        const float a = 0.5f + 0.5f * std::sin((float)ImGui::GetTime() * 6.0f);
        col = ImGui::GetColorU32(ImVec4(0.35f, 0.65f, 1.0f, 0.4f + 0.6f * a));
        tip = "Computing";
    } else if (queued) {
        col = ImGui::GetColorU32(ImVec4(0.35f, 0.65f, 1.0f, 0.6f));
        tip = "Queued";
    } else if (!it->second.committed.empty() && !it->second.committed_ok) {
        col = ImGui::GetColorU32(ImVec4(0.95f, 0.45f, 0.40f, 1.0f));
        tip = nullptr;
    } else if (ts.leaf_dirty(id)) {
        col = ImGui::GetColorU32(ImVec4(0.86f, 0.80f, 0.45f, 1.0f));
        tip = "Out of date: Run recomputes it";
    } else {
        col = ImGui::GetColorU32(ImVec4(0.30f, 0.79f, 0.55f, 1.0f));
        tip = "Up to date";
    }
    const ImVec2 mn = ImGui::GetItemRectMin(), mx = ImGui::GetItemRectMax();
    const ImVec2 c((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddCircleFilled(c, r, col);
    // На светлом фоне (Light, светлые камуфляжи, Cyan HUD Light) светлые
    // кружки теряются — обводим тёмной рамкой; на тёмном она не нужна.
    const ImVec4 bg = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
    if (0.299f * bg.x + 0.587f * bg.y + 0.114f * bg.z > 0.5f)
        dl->AddCircle(c, r + 0.5f, IM_COL32(20, 20, 20, 200), 0, 1.0f);
    if (ImGui::IsItemHovered()) {
        if (tip) ImGui::SetTooltip("%s", tip);
        else {
            const std::string err = it->second.last_error();
            ImGui::SetTooltip("Failed: %s", err.empty() ? "see the plot window" : err.c_str());
        }
    }
}

float tree_button_width(const char* label) {
    return ImGui::CalcTextSize(label).x + 2.0f * ImGui::GetStyle().FramePadding.x;
}

bool small_icon_button(const char* label, const char* tip, ImVec4 col = ImVec4(0, 0, 0, 0),
                       float width = 0.0f) {
    if (col.w > 0.0f) ImGui::PushStyleColor(ImGuiCol_Text, readable_button_text(col));
    bool r;
    if (width > 0.0f) {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 0.0f));
        r = ImGui::Button(label, ImVec2(width, 0.0f));
        ImGui::PopStyleVar();
    } else {
        r = ImGui::SmallButton(label);
    }
    if (col.w > 0.0f) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    return r;
}

enum TreeColumn { NodeColumn, AddColumn, DuplicateColumn, PowerColumn, CloseColumn, LevelColumn, TreeColumnCount };

void draw_node(AppModel& model, TreeSession& ts, int id, Pending& pend, float power_width) {
    TreeNode* n = ts.tree.find(id);
    if (!n) return;
    const bool ctx    = tree_kind_is_context(n->kind);
    const bool active = ts.tree.is_active(id);
    const int  level  = ts.tree.level(id);
    const ImVec4 col  = tree_dim_color(tree_kind_dim_slot(n->kind), active);

    ImGui::PushID(id);
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(NodeColumn);
    ImGuiTreeNodeFlags fl = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth
                          | ImGuiTreeNodeFlags_AllowOverlap;
    // Лист с доп. окнами раскрывается: под ним строки этих окон.
    const bool has_views = !ctx && !n->extra_views.empty();
    const bool branch    = ctx || has_views;
    if (!branch) fl |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (ts.tree.selected_id == id) fl |= ImGuiTreeNodeFlags_Selected;
    if (branch) ImGui::SetNextItemOpen(n->enabled && n->expanded, ImGuiCond_Always);

    ImGui::PushStyleColor(ImGuiCol_Text, col);
    const bool open = ImGui::TreeNodeEx("##node", fl, "%s", n->name.c_str());
    ImGui::PopStyleColor();
    if (branch && ImGui::IsItemToggledOpen()) {
        if (!n->enabled) flash("Node is disabled - enable it to expand");
        else { n->expanded = open; ts.dirty = true; }
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen()) {
        ts.tree.selected_id = id;
        if (!ctx) { g_tree_focus_leaf = id; g_tree_focus_view = 0; }
    }
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        g_rename_id = id;
        std::snprintf(g_rename_buf, sizeof(g_rename_buf), "%s", n->name.c_str());
    }
    if (ImGui::BeginPopupContextItem("##ctx")) {
        ts.tree.selected_id = id;
        if (ImGui::MenuItem(ctx ? "Run subtree" : "Run node")) pend = { Act::RunNode, id };
        if (tree_kind_is_2d_leaf(n->kind) || tree_kind_is_1d_leaf(n->kind)) {
            if (ImGui::MenuItem("New window", nullptr, false, n->extra_views.size() < 15))
                pend = { Act::NewView, id };
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Another window of the same result, no recomputation\n"
                                  "(e.g. a different Lyapunov exponent of the same 2D map).");
        }
        if (ImGui::MenuItem("Rename")) {
            g_rename_id = id;
            std::snprintf(g_rename_buf, sizeof(g_rename_buf), "%s", n->name.c_str());
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Move up", nullptr, false, n->parent != 0))   pend = { Act::Up, id };
        if (ImGui::MenuItem("Move down", nullptr, false, n->parent != 0)) pend = { Act::Down, id };
        ImGui::Separator();
        if (ImGui::MenuItem("Duplicate", nullptr, false, n->parent != 0)) pend = { Act::Dup, id };
        if (ImGui::MenuItem(n->enabled ? "Disable" : "Enable"))           pend = { Act::Toggle, id };
        if (ImGui::MenuItem("Delete", nullptr, false, n->parent != 0))    pend = { Act::Del, id };
        ImGui::EndPopup();
    }

    // Состояние листа и подпись контекста.
    ImGui::SameLine();
    if (!ctx) draw_status_dot(model, ts, id);
    else {
        const std::string meta = node_meta(ts, *n);
        if (!meta.empty()) ImGui::TextDisabled("%s", meta.c_str());
        if (n->kind == TreeNodeKind::Point && ImGui::IsItemHovered())
            ImGui::SetTooltip("Fixed parameters of this point: its leaves are computed at these values.\n"
                              "(cursor) - follows the region crosshair; (pinned) - set in the node panel.");
    }

    const bool can_add = !tree_addable_children(n->kind).empty();
    const char* power = n->enabled ? "on" : "off";
    ImGui::TableSetColumnIndex(AddColumn);
    if (can_add) {
        if (small_icon_button("+", "Add child", ImVec4(0.30f, 0.79f, 0.55f, 1.0f))) {
            if (!n->enabled) flash("Node is disabled - enable it first");
            else ImGui::OpenPopup("##add");
        }
    }
    if (ImGui::BeginPopup("##add")) {
        add_menu_items(*n, level + 1, pend);
        ImGui::EndPopup();
    }
    ImGui::TableSetColumnIndex(DuplicateColumn);
    if (n->parent != 0) {
        if (small_icon_button("dup", "Duplicate (with subtree)")) pend = { Act::Dup, id };
    }
    ImGui::TableSetColumnIndex(PowerColumn);
    if (small_icon_button(power, n->enabled ? "Disable node" : "Enable node",
                          n->enabled ? ImVec4(0.30f, 0.79f, 0.69f, 1.0f) : ImVec4(0.96f, 0.53f, 0.44f, 1.0f),
                          power_width))
        pend = { Act::Toggle, id };
    ImGui::TableSetColumnIndex(CloseColumn);
    if (n->parent != 0) {
        if (small_icon_button("x", "Delete (with subtree)")) pend = { Act::Del, id };
    }
    ImGui::TableSetColumnIndex(LevelColumn);
    ImGui::TextColored(col, "L%d", level);

    if (ctx && open) {
        for (int c : std::vector<int>(n->children)) draw_node(model, ts, c, pend, power_width);
        ImGui::TreePop();
    }
    if (has_views && open) {
        // Доп. окна листа: клик выводит окно вперёд, x закрывает.
        const bool ls2d = n->kind == TreeNodeKind::LS2D;
        for (int k = 1; k <= (int)n->extra_views.size(); ++k) {
            ImGui::PushID(1000 + k);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(NodeColumn);
            char label[64];
            const int ex = n->extra_views[k - 1];
            if (ls2d && ex < 0)  std::snprintf(label, sizeof(label), "Window %d  (sum)", k + 1);
            else if (ls2d)       std::snprintf(label, sizeof(label), "Window %d  (L%d)", k + 1, ex + 1);
            else                 std::snprintf(label, sizeof(label), "Window %d", k + 1);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            ImGui::TreeNodeEx("##view", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen
                              | ImGuiTreeNodeFlags_SpanAvailWidth, "%s", label);
            ImGui::PopStyleColor();
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                ts.tree.selected_id = id;
                g_tree_focus_leaf = id;
                g_tree_focus_view = k;
            }
            ImGui::TableSetColumnIndex(CloseColumn);
            if (small_icon_button("x", "Close this window")) pend = { Act::CloseView, id, n->kind, k };
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void apply_pending(AppModel& model, TreeSession& ts, const Pending& p) {
    TreeNode* n = ts.tree.find(p.id);
    if (!n) return;
    switch (p.act) {
    case Act::Add: {
        const int nid = ts.add_node(p.id, p.kind, p.dir);
        if (nid) {
            ts.tree.selected_id = nid;
            if (!tree_kind_is_context(p.kind)) g_tree_focus_leaf = nid;
        }
        break;
    }
    case Act::Dup: {
        const int nid = ts.duplicate_node(p.id);
        if (nid) ts.tree.selected_id = nid;
        break;
    }
    case Act::Del:
        if (!model.remove_tree_node(p.id))
            flash("\"" + n->name + "\" is computing - stop it first");
        break;
    case Act::Toggle:
        ts.toggle_enabled(p.id);
        break;
    case Act::Up:   ts.move_node(p.id, -1); break;
    case Act::Down: ts.move_node(p.id, +1); break;
    case Act::NewView: {
        int v = 0;
        auto it = ts.leaves.find(p.id);
        if (it != ts.leaves.end() && it->second.ls && !it->second.ls->curves.empty())
            v = it->second.ls->curves[0].display_exponent_idx;
        n->extra_views.push_back(v);
        ts.dirty = true;
        break;
    }
    case Act::CloseView:
        if (p.dir >= 1 && p.dir <= (int)n->extra_views.size()) {
            n->extra_views.erase(n->extra_views.begin() + (p.dir - 1));
            tree_forget_views(p.id, p.dir);
            ts.dirty = true;
        }
        break;
    case Act::RunNode:
        if (tree_kind_is_context(n->kind)) model.enqueue_tree(ts.dirty_leaves(p.id));
        else if (ts.tree.is_active(p.id))  model.enqueue_tree({ p.id });
        break;
    default: break;
    }
}

// --- Панели параметров ---

// Курсор вдоль оси: InputDouble с запуском автопересчёта по отпусканию.
void cursor_field(const char* label, double& v, int owner) {
    ImGui::SetNextItemWidth(kFieldW);
    ImGui::InputDouble(label, &v, 0.0, 0.0, "%.10g");
    if (ImGui::IsItemDeactivatedAfterEdit()) tree_cursor_moved(owner);
}

bool axis_row(AppModel& model, TreeSession& ts, const char* tag, const char* title,
              int& par, bool& ov, int& vi, bool& oh, bool* other_oh, bool& lg,
              std::string& lo, std::string& hi) {
    bool ch = false;
    const int p0 = par; const bool ov0 = ov, oh0 = oh, lg0 = lg; const int vi0 = vi;
    ImGui::TextUnformatted(title); ImGui::SameLine();
    draw_sweep_target_combo((std::string("##") + tag).c_str(), ts.params, ts.vars, par, ov, vi, oh,
                            other_oh, false, 120.0f,
                            scheme_uses_symmetry(ts.tree.nodes.front().cfg.scheme, ts.custom_schemes),
                            nullptr, BroadcastField::SweepTarget, !model.is_map);
    ch |= InputNumStr((std::string("lo##") + tag).c_str(), lo, kFieldW);
    ch |= InputNumStr((std::string("hi##") + tag).c_str(), hi, kFieldW);
    ImGui::Checkbox((std::string("Log scale##") + tag).c_str(), &lg);
    if (lg) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }
    return ch || p0 != par || ov0 != ov || vi0 != vi || oh0 != oh || lg0 != lg;
}

bool draw_system_panel(AppModel& model, TreeSession& ts, TreeNode& n) {
    bool ch = false;
    auto& c = n.cfg;
    ch |= draw_scheme_combo("Scheme", c.scheme, ts.custom_schemes, {}, &ts.enabled_builtin_schemes,
                            nullptr, model.is_map, &ts.wrapper_schemes, false);
    if (ImGui::CollapsingHeader("Integration##tree_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!model.is_map) ch |= InputNumStr("h", c.h_text, kFieldW);
        if (scheme_uses_symmetry(c.scheme, ts.custom_schemes))
            ch |= InputNumStr("symmetry s", c.symmetry_s, kFieldW);
        if (scheme_uses_cd_fp_iters(c.scheme)) {
            ch |= InputNumStr("FP iterations n", c.cd_fp_iters, kFieldW);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
        }
        if (scheme_uses_newton(c.scheme)) ch |= draw_newton_fields(c.newton, kFieldW, "tree_newton");
        ch |= InputNumStr(model.is_map ? "transient iterations" : "transient time", c.transient_text, kFieldW);
        ch |= InputNumStr(model.is_map ? "iterations" : "computing time", c.t_max_text, kFieldW);
        ch |= InputNumStr("decimator", c.pre_scaller_text, kFieldW);
        ch |= InputNumStr("max value", c.max_value_text, kFieldW);
        ImGui::Separator();
        ch |= draw_adaptive_block("tree_ad", c.adaptive, c.scheme,
                                  kAdUiRaw | kAdUiInterp | kAdUiMaxPts, model.is_map, nullptr);
    }
    const auto ic0 = c.initial_conditions;
    draw_named_num_fields("Initial conditions##tree_ic", ts.vars, c.initial_conditions);
    ch |= ic0 != c.initial_conditions;

    if (ImGui::CollapsingHeader("Parameters##tree_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Параметры осей свипа помечены; правка такого параметра двигает курсор владельца.
        struct AxisUse { int owner; int dir; };
        std::map<std::string, std::vector<AxisUse>> uses;
        for (const auto& m : ts.tree.nodes) {
            if (!ts.tree.is_active(m.id)) continue;
            auto mark = [&](bool ov, bool oh, int pi, int dir) {
                if (ov || oh || pi < 0 || pi >= (int)ts.params.size()) return;
                uses[ts.params[pi]].push_back({ m.id, dir });
            };
            if (m.kind == TreeNodeKind::Region2D) {
                mark(m.cfg.axis_x_over_var, m.cfg.axis_x_over_h, m.cfg.axis_x_par_index, 0);
                mark(m.cfg.axis_y_over_var, m.cfg.axis_y_over_h, m.cfg.axis_y_par_index, 1);
            } else if (m.kind == TreeNodeKind::Slice1D && ts.region_of(m.id) == 0) {
                mark(m.cfg.sweep_x_over_var, m.cfg.sweep_x_over_h, m.cfg.sweep_x_par_index, 0);
            }
        }
        draw_param_tree(&model.param_groups, ts.params, [&](const std::string& p) {
            ImGui::PushID(p.c_str());
            const bool edited = InputNumStr("##v", c.param_values[p], kFieldW);
            ImGui::SameLine();
            draw_field_name(p, &model.field_labels);
            auto it = uses.find(p);
            if (edited) {
                ch = true;
                if (it != uses.end()) {
                    const double v = parse_ratio_or(c.param_values[p], 0.0);
                    for (const auto& u : it->second) {
                        TreeNode* o = ts.tree.find(u.owner);
                        if (!o) continue;
                        (u.dir == 0 ? o->cfg.fix_x_value : o->cfg.fix_y_value) = v;
                        tree_cursor_moved(u.owner);
                    }
                }
            }
            if (it != uses.end()) { ImGui::SameLine(); ImGui::TextDisabled("(swept)"); }
            ImGui::PopID();
        });
    }
    return ch;
}

bool draw_region_panel(AppModel& model, TreeSession& ts, TreeNode& n) {
    bool ch = false;
    auto& c = n.cfg;
    ch |= axis_row(model, ts, "ax", "Axis X:", c.axis_x_par_index, c.axis_x_over_var, c.axis_x_var_index,
                   c.axis_x_over_h, &c.axis_y_over_h, c.axis_x_log, c.axis_x_lo_text, c.axis_x_hi_text);
    ch |= axis_row(model, ts, "ay", "Axis Y:", c.axis_y_par_index, c.axis_y_over_var, c.axis_y_var_index,
                   c.axis_y_over_h, &c.axis_x_over_h, c.axis_y_log, c.axis_y_lo_text, c.axis_y_hi_text);
    ch |= InputNumStr("Resolution", c.resolution_text, kFieldW);
    const int wv0 = c.bif_writable_var;
    draw_writable_var_combo(ts.vars, c.bif_writable_var, "Bif variable##tree_wv");
    ch |= wv0 != c.bif_writable_var;
    ImGui::SeparatorText("Cursor");
    cursor_field("x##cur", c.fix_x_value, n.id);
    cursor_field("y##cur", c.fix_y_value, n.id);
    ImGui::TextDisabled("Middle drag or Shift+drag on a map moves it.");
    return ch;
}

bool draw_slice_panel(AppModel& model, TreeSession& ts, TreeNode& n) {
    bool ch = false;
    auto& c = n.cfg;
    const TreeNode* reg = ts.tree.find(ts.region_of(n.id));
    ImGui::Text("Direction: along %s", n.slice_dir == 0 ? "X" : "Y");
    if (!reg) ImGui::BeginDisabled();
    ch |= ImGui::Checkbox("Inherit axis from the region", &c.inherit_sweep_from_2d);
    if (!reg) { ImGui::EndDisabled(); ImGui::SameLine(); ImGui::TextDisabled("(no region above)"); }
    const bool inheriting = reg && c.inherit_sweep_from_2d;
    if (inheriting) ImGui::BeginDisabled();
    bool other_oh = false;
    ch |= axis_row(model, ts, "sw", "Axis:", c.sweep_x_par_index, c.sweep_x_over_var, c.sweep_x_var_index,
                   c.sweep_x_over_h, &other_oh, c.sweep_x_log, c.sweep_x_lo_text, c.sweep_x_hi_text);
    if (inheriting) ImGui::EndDisabled();
    ch |= InputNumStr("N", c.n_x_1d_text, kFieldW);
    ImGui::SeparatorText("1D integrator");
    bool own = n.own_integration;
    if (ImGui::Checkbox("Own integration##l1d", &own)) {
        if (own) ts.enable_own_integration(n.id);
        else     n.own_integration = false;
        ch = true;
    }
    if (n.own_integration) {
        if (!model.is_map) ch |= InputNumStr("h##l1d", c.l1d_h_text, kFieldW);
        ch |= InputNumStr(model.is_map ? "transient iterations##l1d" : "transient time##l1d", c.l1d_transient_text, kFieldW);
        ch |= InputNumStr(model.is_map ? "iterations##l1d" : "computing time##l1d", c.l1d_t_max_text, kFieldW);
    } else {
        const CustomTabSharedConfig& root = ts.tree.nodes.front().cfg;
        ImGui::TextDisabled("Inherited: h %s, transient %s, time %s",
                            root.h_text.c_str(), root.transient_text.c_str(), root.t_max_text.c_str());
    }
    ch |= ImGui::Checkbox("Continuation", &c.continuation_1d_enabled);
    ch |= ImGui::Checkbox("Inter-peak intervals on Y (Bif)", &c.plot_inter_peaks_1d);
    ImGui::SeparatorText("Cursor");
    ch |= ImGui::Checkbox("Live (recompute when the cursor moves)", &n.live);
    if (reg) {
        ch |= ImGui::Checkbox("Follow the region cursor", &n.follow_cursor);
        if (!n.follow_cursor) {
            ImGui::TextDisabled("Pinned point for the other axis:");
            cursor_field("x##pin", c.fix_x_value, n.id);
            cursor_field("y##pin", c.fix_y_value, n.id);
        }
    } else {
        cursor_field("position##cur", c.fix_x_value, n.id);
    }
    return ch;
}

bool draw_point_panel(TreeSession& ts, TreeNode& n) {
    bool ch = false;
    const TreeNode* reg = ts.tree.find(ts.region_of(n.id));
    ch |= ImGui::Checkbox("Live (recompute when the cursor moves)", &n.live);
    if (!reg) { ImGui::TextDisabled("No region above: the point uses System values."); return ch; }
    ch |= ImGui::Checkbox("Follow the region cursor", &n.follow_cursor);
    if (n.follow_cursor) {
        ImGui::TextDisabled("At (%s, %s)", fmt_short(reg->cfg.fix_x_value).c_str(),
                            fmt_short(reg->cfg.fix_y_value).c_str());
        if (ImGui::Button("Pin here")) {
            n.follow_cursor = false;
            n.cfg.fix_x_value = reg->cfg.fix_x_value;
            n.cfg.fix_y_value = reg->cfg.fix_y_value;
            ch = true;
        }
    } else {
        cursor_field("x##pin", n.cfg.fix_x_value, n.id);
        cursor_field("y##pin", n.cfg.fix_y_value, n.id);
    }
    return ch;
}

bool draw_leaf_integration(AppModel& model, TreeSession& ts, TreeNode& n,
                           PhaseAnalysisSession* phase = nullptr) {
    bool ch = false;
    ImGui::SeparatorText("Integration");
    bool own = n.own_integration;
    if (ImGui::Checkbox(phase ? "Own integration" : "Own integration and initial conditions", &own)) {
        if (own) ts.enable_own_integration(n.id);
        else     n.own_integration = false;
        ch = true;
    }
    if (n.own_integration) {
        auto& c = n.cfg;
        if (!model.is_map) ch |= InputNumStr("h##own", c.h_text, kFieldW);
        ch |= InputNumStr(model.is_map ? "transient iterations##own" : "transient time##own", c.transient_text, kFieldW);
        ch |= InputNumStr(model.is_map ? "iterations##own" : "computing time##own", c.t_max_text, kFieldW);
        if (!phase) {
            ch |= InputNumStr("decimator##own", c.pre_scaller_text, kFieldW);
            const auto ic0 = c.initial_conditions;
            draw_named_num_fields("Initial conditions##own_ic", ts.vars, c.initial_conditions,
                                  n.kind == TreeNodeKind::Basins ? "Values for non-axis variables." : nullptr);
            ch |= ic0 != c.initial_conditions;
        }
    } else {
        const CustomTabSharedConfig s = ts.effective_shared(n.id);
        const bool is1d = tree_kind_is_1d_leaf(n.kind);
        ImGui::TextDisabled("Inherited: h %s, transient %s, time %s",
                            (is1d ? s.l1d_h_text : s.h_text).c_str(),
                            (is1d ? s.l1d_transient_text : s.transient_text).c_str(),
                            (is1d ? s.l1d_t_max_text : s.t_max_text).c_str());
    }
    if (phase) ch |= InputNumStr("Decimation (every Nth point)##own", phase->decimation, kFieldW);
    return ch;
}

bool draw_leaf_panel(AppModel& model, TreeSession& ts, TreeNode& n) {
    bool ch = false;
    auto it = ts.leaves.find(n.id);
    if (it == ts.leaves.end()) return false;
    auto& rt = it->second;
    // Переменная БД живёт в регионе (или в System), ей же наследуется срез.
    auto writable_owner = [&]() -> CustomTabSharedConfig& {
        TreeNode* reg = ts.tree.find(ts.region_of(n.id));
        return reg ? reg->cfg : ts.tree.nodes.front().cfg;
    };
    switch (n.kind) {
    case TreeNodeKind::Bif2D: case TreeNodeKind::Bif1D: {
        auto& d = rt.bif->diagrams[0];
        auto& wo = writable_owner();
        const int wv0 = wo.bif_writable_var;
        draw_writable_var_combo(ts.vars, wo.bif_writable_var, "Bif variable##tree_leaf_wv");
        ch |= wv0 != wo.bif_writable_var;
        if (n.kind == TreeNodeKind::Bif2D) {
            ch |= InputNumStr("DBSCAN eps", d.eps_dbscan_text, kFieldW);
            ch |= InputNumStr("mult peak", d.mult_peak_text, kFieldW);
            ch |= InputNumStr("mult interval", d.mult_interval_text, kFieldW);
        }
        break;
    }
    case TreeNodeKind::LLE2D: case TreeNodeKind::LLE1D:
        ch |= InputNumStr("eps", rt.lle->curves[0].eps_text, kFieldW);
        ch |= InputNumStr("NT", rt.lle->curves[0].nt_text, kFieldW);
        break;
    case TreeNodeKind::LS2D: case TreeNodeKind::LS1D:
        ch |= InputNumStr("eps", rt.ls->curves[0].eps_text, kFieldW);
        ch |= InputNumStr("NT", rt.ls->curves[0].nt_text, kFieldW);
        break;
    case TreeNodeKind::Phase: {
        const int id = n.id;
        // Recompute и Auto recompute — через очередь дерева: прямой расчёт взял бы поля панели
        // без контекста узла.
        draw_phase_controls(*rt.phase, nullptr, &model, nullptr,
                            [&model, id]() { model.enqueue_tree({ id }); }, false,
                            [&]() {
                                const bool edited = draw_leaf_integration(model, ts, n, rt.phase.get());
                                ch |= edited;
                                return edited;
                            });
        break;
    }
    case TreeNodeKind::Basins: {
        // Как во вкладке Basins: оси по НУ, сетка, фичи DBSCAN, перекластеризация, портреты.
        auto& b  = *rt.basins;
        auto& bc = b.configs[0];
        if (!ts.vars.empty()) {
            if (bc.axis_x_var < 0 || bc.axis_x_var >= (int)ts.vars.size()) bc.axis_x_var = 0;
            if (bc.axis_y_var < 0 || bc.axis_y_var >= (int)ts.vars.size())
                bc.axis_y_var = ts.vars.size() > 1 ? 1 : 0;
            const std::vector<const char*> items = c_str_list(ts.vars);
            auto symlog = [&](const char* id, bool& on) {
                ImGui::SameLine();
                ch |= ImGui::Checkbox(id, &on);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("|x| log-spaced on [lo, hi] on both sides of 0\n"
                                      "(nodes -hi..-lo, +lo..+hi, monotone).\n"
                                      "lo, hi > 0; Resolution even.");
            };
            ImGui::SetNextItemWidth(kComboW);
            ch |= ImGui::Combo("Axis X (IC)", &bc.axis_x_var, items.data(), (int)items.size());
            symlog("\xC2\xB1log##tbas_xsym", bc.axis_x_symlog);
            ch |= InputNumStr(bc.axis_x_symlog ? "|X| lo" : "X lo", bc.axis_x_lo_text, kFieldW);
            ch |= InputNumStr(bc.axis_x_symlog ? "|X| hi" : "X hi", bc.axis_x_hi_text, kFieldW);
            ImGui::SetNextItemWidth(kComboW);
            ch |= ImGui::Combo("Axis Y (IC)", &bc.axis_y_var, items.data(), (int)items.size());
            symlog("\xC2\xB1log##tbas_ysym", bc.axis_y_symlog);
            ch |= InputNumStr(bc.axis_y_symlog ? "|Y| lo" : "Y lo", bc.axis_y_lo_text, kFieldW);
            ch |= InputNumStr(bc.axis_y_symlog ? "|Y| hi" : "Y hi", bc.axis_y_hi_text, kFieldW);
        }
        ch |= InputNumStr("Resolution", bc.n_pts_text, kFieldW);
        const int wv0 = bc.writable_var;
        draw_writable_var_combo(ts.vars, bc.writable_var, "Writable var##tree_bas_wv");
        ch |= wv0 != bc.writable_var;
        if (ImGui::CollapsingHeader("Features (DBSCAN axes + plot data)", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (bc.feature1 < 0 || bc.feature1 >= BF_FEATURE_COUNT) bc.feature1 = BF_FEATURE1_DEFAULT;
            if (bc.feature2 < 0 || bc.feature2 >= BF_FEATURE_COUNT) bc.feature2 = BF_FEATURE2_DEFAULT;
            ImGui::SetNextItemWidth(220);
            ch |= ImGui::Combo("Feature 1##tbas", &bc.feature1, kBasinFeatureNames, BF_FEATURE_COUNT);
            ImGui::SameLine();
            ch |= InputNumStr("mult##tbas_f1", bc.mult_feature1_text, 80);
            ImGui::SetNextItemWidth(220);
            ch |= ImGui::Combo("Feature 2##tbas", &bc.feature2, kBasinFeatureNames, BF_FEATURE_COUNT);
            ImGui::SameLine();
            ch |= InputNumStr("mult##tbas_f2", bc.mult_feature2_text, 80);
        }
        ch |= InputNumStr("DBSCAN eps", bc.eps_dbscan_text, kFieldW);
        ImGui::SameLine();
        const bool can_recluster = !ts.any_in_flight() && bc.last_run_ok && !bc.result.avg_peaks.empty()
                                && !bc.result.avg_intervals.empty() && !bc.result.helpful_array.empty();
        if (!can_recluster) ImGui::BeginDisabled();
        if (ImGui::Button("Clustering")) model.recluster_tree_basins(n.id);
        if (!can_recluster) ImGui::EndDisabled();
        ImGui::TextDisabled("Clustering radius in (Feature 1, Feature 2) space.");
        if (ImGui::CollapsingHeader("Phase portraits")) draw_basins_phase_controls(model, b, 0);
        if (bc.last_run_ok)
            ImGui::TextColored(readable_text(ImVec4(0.4f, 1.0f, 0.4f, 1.0f)), "OK: %dx%d, %d clusters (+ %d FP clusters)",
                               bc.result.n_pts, bc.result.n_pts, bc.result.n_clusters, -bc.result.min_cluster_idx);
        break;
    }
    default: break;
    }

    if (!rt.phase) ch |= draw_leaf_integration(model, ts, n);
    if (!rt.in_flight() && !rt.committed.empty() && !rt.committed_ok) {
        const std::string err = rt.last_error();
        if (!err.empty()) ImGui::TextColored(readable_text(ImVec4(1.0f, 0.5f, 0.0f, 1.0f)), "%s", err.c_str());
    }
    return ch;
}

void draw_params_panel(AppModel& model, TreeSession& ts) {
    TreeNode* n = ts.tree.find(ts.tree.selected_id);
    if (!n) { ImGui::TextDisabled("Select a node."); return; }
    const ImVec4 col = tree_dim_color(tree_kind_dim_slot(n->kind), ts.tree.is_active(n->id));
    ImGui::TextColored(col, "%s", n->name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("L%d  %s  %s", ts.tree.level(n->id), tree_kind_title(n->kind),
                        n->enabled ? "" : "(disabled)");
    ImGui::Separator();
    ImGui::PushID(n->id);
    bool ch = false;
    switch (n->kind) {
    case TreeNodeKind::System:   ch = draw_system_panel(model, ts, *n); break;
    case TreeNodeKind::Region2D: ch = draw_region_panel(model, ts, *n); break;
    case TreeNodeKind::Slice1D:  ch = draw_slice_panel(model, ts, *n);  break;
    case TreeNodeKind::Point:    ch = draw_point_panel(ts, *n);         break;
    default:                     ch = draw_leaf_panel(model, ts, *n);   break;
    }
    ImGui::PopID();
    if (ch) ts.dirty = true;
}

} // namespace

void tree_cursor_moved(int owner_id) {
    cursor_timers()[owner_id] = ImGui::GetTime();
}

ImVec4 tree_dim_color(int slot, bool active) {
    if (!active) return ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
    if (slot <= 0) return ImGui::GetStyleColorVec4(ImGuiCol_Text);
    // Тот же сдвиг оттенка от цвета Header, что у уровней Custom (2D, 1D, точка).
    const ImVec4 base = ImGui::GetStyleColorVec4(ImGuiCol_Header);
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    float nh = h + (45.0f / 360.0f) * (float)(slot - 1);
    nh -= std::floor(nh);
    ImVec4 col(0, 0, 0, 1.0f);
    ImGui::ColorConvertHSVtoRGB(nh, std::max(0.75f, s), std::max(0.85f, v), col.x, col.y, col.z);
    return readable_text(col);
}

void load_tree_session(AppModel& model, SystemLibrary& lib) {
    model.tree_session = TreeSession{};
    model.clear_tree_queue();
    cursor_timers().clear();
    model.start_tree_analysis();
    if (!model.loaded_name.empty()) {
        const std::string j = lib.load_session(model.loaded_name, "_last_tree");
        if (!j.empty() && !session_from_json_tree(j, model.tree_session))
            model.session_load_warning = "saved session '_last_tree' is corrupt - ignored";
    }
    model.tree_session.dirty = false;
}

void draw_tree_controls(AppModel& model, SystemLibrary& lib) {
    (void)lib;
    auto& ts = model.tree_session;
    if (ts.tree.nodes.empty()) { ImGui::TextDisabled("Load a system first."); return; }
    ts.clamp_cursors();

    // Автопересчёт по курсору: пауза после последнего сдвига, движок и очередь свободны.
    if (!ts.any_in_flight() && model.tree_queue().empty()) {
        const double now = ImGui::GetTime();
        for (auto& [owner, t] : cursor_timers()) {
            if (t <= 0.0 || now - t <= kCustomSliderDebounceSec) continue;
            t = 0.0;
            if (ts.tree.find(owner)) model.enqueue_tree(ts.cursor_leaves(owner));
        }
    }

    // Run / Stop.
    bool run_now = ImGui::Button("Run");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compute every out-of-date node (Ctrl+R)");
    if (ImGui::GetIO().KeyCtrl && !ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_now = true;
    if (run_now) model.enqueue_tree(ts.dirty_leaves(ts.tree.nodes.front().id));
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        model.clear_tree_queue();
        ts.request_cancel_all();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu queued)", model.tree_queue().size());
    if (ImGui::GetTime() - g_msg_time < 3.0)
        ImGui::TextColored(readable_text(ImVec4(0.96f, 0.53f, 0.44f, 1.0f)), "%s", g_msg.c_str());

    // Дерево сверху, параметры снизу, между ними сплиттер.
    static float tree_fraction = 0.5f;
    constexpr float splitter_h = 6.0f;
    const float panes_h = std::max(8.0f, ImGui::GetContentRegionAvail().y
                                  - splitter_h - 2.0f * ImGui::GetStyle().ItemSpacing.y);
    const float min_pane_h = std::min(90.0f, panes_h * 0.5f);
    const float tree_h = std::clamp(panes_h * tree_fraction, min_pane_h, panes_h - min_pane_h);

    Pending pend;
    if (ImGui::BeginChild("##tree_nodes", ImVec2(0, tree_h), ImGuiChildFlags_Borders)) {
        const float power_width = std::max(tree_button_width("on"), tree_button_width("off"));
        if (ImGui::BeginTable("##tree_tbl", TreeColumnCount, ImGuiTableFlags_RowBg | ImGuiTableFlags_NoPadOuterX)) {
            ImGui::TableSetupColumn("node", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("add", ImGuiTableColumnFlags_WidthFixed, tree_button_width("+"));
            ImGui::TableSetupColumn("duplicate", ImGuiTableColumnFlags_WidthFixed, tree_button_width("dup"));
            ImGui::TableSetupColumn("power", ImGuiTableColumnFlags_WidthFixed, power_width);
            ImGui::TableSetupColumn("close", ImGuiTableColumnFlags_WidthFixed, tree_button_width("x"));
            ImGui::TableSetupColumn("level", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("L10").x);
            draw_node(model, ts, ts.tree.nodes.front().id, pend, power_width);
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    if (pend.act != Act::None) apply_pending(model, ts, pend);

    ImGui::InvisibleButton("##tree_split", ImVec2(-1, splitter_h));
    if (ImGui::IsItemActive()) {
        const float new_h = tree_h + ImGui::GetIO().MouseDelta.y;
        tree_fraction = std::clamp(new_h, min_pane_h, panes_h - min_pane_h) / panes_h;
    }
    if (ImGui::IsItemHovered() || ImGui::IsItemActive()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    {
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        const float y = (a.y + b.y) * 0.5f;
        ImGui::GetWindowDrawList()->AddLine(ImVec2(a.x, y), ImVec2(b.x, y),
            ImGui::GetColorU32(ImGui::IsItemActive() ? ImGuiCol_SeparatorActive : ImGuiCol_Separator), 2.0f);
    }

    if (ImGui::BeginChild("##tree_params", ImVec2(0, 0), ImGuiChildFlags_Borders)) {
        draw_params_panel(model, ts);
    }
    ImGui::EndChild();

    // Переименование по двойному клику.
    if (g_rename_id) {
        ImGui::OpenPopup("Rename node##tree");
        if (ImGui::BeginPopupModal("Rename node##tree", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            const bool enter = ImGui::InputText("##name", g_rename_buf, sizeof(g_rename_buf),
                                                ImGuiInputTextFlags_EnterReturnsTrue);
            if (enter || ImGui::Button("OK")) {
                if (TreeNode* n = ts.tree.find(g_rename_id); n && g_rename_buf[0]) {
                    n->name = g_rename_buf;
                    ts.dirty = true;
                }
                g_rename_id = 0;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                g_rename_id = 0;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
}

} // namespace ucuda::ui_detail
