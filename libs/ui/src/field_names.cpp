#include "gui_internal.h"
// Имена полей формулами, подписи, комбо переменных, деревья групп параметров.
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
std::string field_label_auto(const std::string& name) { return name_to_latex(name); }
}


namespace ucuda::ui_detail {
static bool g_field_latex = true;
}


namespace ucuda::ui_detail {
static const FieldLabels* g_field_labels = nullptr;
}


namespace ucuda::ui_detail {
void set_field_latex(bool on) { g_field_latex = on; }
}


namespace ucuda::ui_detail {
void set_field_labels(const FieldLabels* labels) { g_field_labels = labels; }
}

namespace ucuda::ui_detail {
static SystemLibrary* g_field_lib = nullptr;
}


namespace ucuda::ui_detail {
void set_field_library(SystemLibrary* lib) { g_field_lib = lib; }
}


namespace ucuda::ui_detail {
const SystemRecord* field_lib_record(const SystemEditor* m, bool reload) {
    static SystemRecord rec;
    static std::string name;
    static bool ok = false;
    if (!g_field_lib || !m || m->loaded_name.empty()) return nullptr;
    if (reload || name != m->loaded_name) {
        name = m->loaded_name;
        ok = false;
        try { rec = g_field_lib->load(name); ok = true; } catch (...) {}
    }
    return ok ? &rec : nullptr;
}
}


namespace ucuda::ui_detail {
bool field_restore_menu_items(const SystemEditor* model, BroadcastField f, const std::string& name,
                              std::map<std::string, std::string>& values, const std::vector<std::string>& names) {
    if (f != BroadcastField::Param && f != BroadcastField::InitCondition) return false;
    const SystemRecord* r = field_lib_record(model, ImGui::IsWindowAppearing());
    if (!r) {
        ImGui::BeginDisabled();
        ImGui::MenuItem("Restore from library (system not in the library)");
        ImGui::EndDisabled();
        return false;
    }
    const auto& src = f == BroadcastField::Param ? r->param_values : r->init_conditions;
    bool changed = false;
    {
        auto it = src.find(name);
        auto cur = values.find(name);
        const bool has = it != src.end();
        const bool same = has && cur != values.end() && cur->second == it->second;
        const std::string lbl = has ? "Restore from library (" + it->second + ")" : "Restore from library (no value)";
        if (ImGui::MenuItem(lbl.c_str(), nullptr, false, has && !same)) {
            values[name] = it->second;
            changed = true;
        }
    }
    int differ = 0;
    for (const std::string& n : names) {
        auto it = src.find(n);
        if (it == src.end()) continue;
        auto cur = values.find(n);
        if (cur == values.end() || cur->second != it->second) ++differ;
    }
    const std::string all = std::string("Restore all ")
                          + (f == BroadcastField::Param ? "parameters" : "initial conditions")
                          + " from library (" + std::to_string(differ) + " differ)";
    if (ImGui::MenuItem(all.c_str(), nullptr, false, differ > 0)) {
        for (const std::string& n : names) {
            auto it = src.find(n);
            if (it != src.end()) values[n] = it->second;
        }
        changed = true;
    }
    return changed;
}
}


namespace ucuda::ui_detail {
bool field_restore_popup(const char* popup_id, const SystemEditor* model, BroadcastField f, const std::string& name,
                         std::map<std::string, std::string>& values, const std::vector<std::string>& names) {
    if (!ImGui::BeginPopupContextItem(popup_id)) return false;
    const bool changed = field_restore_menu_items(model, f, name, values, names);
    ImGui::EndPopup();
    return changed;
}
}


namespace ucuda::ui_detail {
std::string field_label(const FieldLabels* labels, const std::string& name) {
    if (!labels) labels = g_field_labels;   // подписи системы, загруженной в модель
    if (labels) {
        auto it = labels->find(name);
        if (it != labels->end() && !it->second.empty()) return it->second;
    }
    return field_label_auto(name);
}
}


namespace ucuda::ui_detail {
MathExtent field_tex_extent(const std::string& tex, float fs) {
    static std::unordered_map<std::string, MathExtent> cache;
    const std::string key = tex + '\x01' + std::to_string((int)(fs * 8.0f));
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, math_formula_extent(tex.c_str(), fs)).first;
    return it->second;
}
}


namespace ucuda::ui_detail {
float field_tex_size() { return std::floor(ImGui::GetFontSize() * 1.2f + 0.5f); }
}


namespace ucuda::ui_detail {
void field_tex_draw(ImDrawList* dl, ImVec2 p, const std::string& tex, float box_h, bool fit ) {
    float fs = field_tex_size();
    MathExtent ext = field_tex_extent(tex, fs);
    float y = p.y + std::max(0.0f, (box_h - (ext.ascent + ext.descent)) * 0.5f);
    if (fit) {
        // Кегль UI-текста и его базовая линия: индекс уходит ниже строки, но в пределах отступа рамки.
        const float ui = ImGui::GetFontSize();
        const float base = p.y + (box_h - ui) * 0.5f + 0.8f * ui;   // ~ascent строки UI-шрифта
        y = std::max(p.y + 1.0f, base - ext.ascent);
    }
    math_formula_draw(dl, ImVec2(p.x, y), ImGui::GetColorU32(ImGuiCol_Text), tex.c_str(), fs);
}
}


namespace ucuda::ui_detail {
std::string g_combo_preview;
}


namespace ucuda::ui_detail {
void draw_field_name(const std::string& name, const FieldLabels* labels) {
    if (!g_field_latex) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(name.c_str());
        return;
    }
    const std::string tex = field_label(labels, name);
    const MathExtent ext = field_tex_extent(tex, field_tex_size());
    const float fh = ImGui::GetFrameHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(std::max(ext.width, 1.0f), std::max(fh, ext.ascent + ext.descent)));
    if (ImGui::IsItemVisible()) field_tex_draw(ImGui::GetWindowDrawList(), p, tex, fh);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
}
}


namespace ucuda::ui_detail {
bool var_combo_begin(const char* id, const std::string& cur, bool cur_is_name) {
    g_combo_preview.clear();
    if (!g_field_latex || !cur_is_name) return ImGui::BeginCombo(id, cur.c_str());
    g_combo_preview = field_label(nullptr, cur);
    return ImGui::BeginCombo(id, "", ImGuiComboFlags_CustomPreview);
}
}


namespace ucuda::ui_detail {
void var_combo_preview() {
    if (g_combo_preview.empty()) return;
    const std::string tex = std::move(g_combo_preview);
    g_combo_preview.clear();
    if (ImGui::BeginComboPreview()) {
        // Курсор превью стоит на отступе рамки; формулу вписываем во всю высоту рамки.
        const ImVec2 c = ImGui::GetCursorScreenPos();
        field_tex_draw(ImGui::GetWindowDrawList(), ImVec2(c.x, c.y - ImGui::GetStyle().FramePadding.y), tex,
                       ImGui::GetFrameHeight(), true);
        ImGui::EndComboPreview();
    }
}
}


namespace ucuda::ui_detail {
bool var_selectable(const std::string& name, bool selected) {
    if (!g_field_latex) return ImGui::Selectable(name.c_str(), selected);
    const std::string tex = field_label(nullptr, name);
    const MathExtent ext = field_tex_extent(tex, field_tex_size());
    const float h = std::max(ImGui::GetTextLineHeight(), ext.ascent + ext.descent);
    const bool clicked = ImGui::Selectable(("##vs_" + name).c_str(), selected, 0, ImVec2(std::max(ext.width, 1.0f), h));
    if (ImGui::IsItemVisible()) field_tex_draw(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), tex, h);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
    return clicked;
}
}


namespace ucuda::ui_detail {
std::string plot_var_name(const std::string& name) {
    return g_field_latex && plot_math_enabled() ? field_label(nullptr, name) : name;
}
}


namespace ucuda::ui_detail {
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note,
                                  AppModel* bc,
                                  BroadcastField field,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (note) ImGui::TextDisabled("%s", note);
    // Группы и подписи — у системы, загруженной в модель (bc); без модели — плоский список имён.
    ParamGroups* groups = !bc ? nullptr
                        : field == BroadcastField::Param ? &bc->param_groups
                        : field == BroadcastField::InitCondition ? &bc->ic_groups : nullptr;
    const FieldLabels* labels = bc ? &bc->field_labels : nullptr;
    int hidden = 0;
    const std::vector<std::string> shown = bc && field == BroadcastField::InitCondition ? bc->ic_fields(names, &hidden) : names;
    if (hidden > 0) ImGui::TextDisabled("%d algebraic variables hidden (Settings)", hidden);
    draw_param_tree(groups, shown, [&](const std::string& n) {
        ImGui::PushID(n.c_str());
        const bool off = overwritten && overwritten(n);
        if (off) ImGui::BeginDisabled();
        InputNumStr("##v", values[n], kFieldW,
                    [&] { field_apply_all_menu(bc, field, n, n, values[n],
                                               [&] { field_restore_menu_items(bc, field, n, values, names); }); });
        ImGui::SameLine();
        draw_field_name(n, labels);
        if (off) ImGui::EndDisabled();
        ImGui::PopID();
    });
}
}


namespace ucuda::ui_detail {
int pg_count(PgNode& n, const std::set<std::string>& present) {
    n.count = 0;
    if (n.g) for (const std::string& p : n.g->params) n.count += present.count(p) ? 1 : 0;
    for (PgNode& k : n.kids) n.count += pg_count(k, present);
    return n.count;
}
}


namespace ucuda::ui_detail {
constexpr const char* kPgDragType = "UCUDA_PG_NODE";
}


namespace ucuda::ui_detail {
std::string pg_payload(const ParamGroups* list, const std::string& path) {
    std::string s(sizeof(list), '\0');
    std::memcpy(&s[0], &list, sizeof(list));
    return s + path;
}
}


namespace ucuda::ui_detail {
void pg_drag_node(ParamGroups* list, const std::string& path, const std::string& shown, PgDrop& drop) {
    if (!list) return;
    if (ImGui::BeginDragDropSource()) {
        const std::string pl = pg_payload(list, path);
        ImGui::SetDragDropPayload(kPgDragType, pl.data(), pl.size());
        ImGui::TextUnformatted(shown.c_str());
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        const ImGuiPayload* pk = ImGui::GetDragDropPayload();
        if (pk && pk->IsDataType(kPgDragType) && pk->DataSize > (int)sizeof(list)) {
            const ParamGroups* src = nullptr;
            std::memcpy(&src, pk->Data, sizeof(src));
            const std::string from((const char*)pk->Data + sizeof(src), (size_t)pk->DataSize - sizeof(src));
            if (src == list && from != path && pg_parent(from) == pg_parent(path)) {
                const ImRect r(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
                const bool after = ImGui::GetMousePos().y > (r.Min.y + r.Max.y) * 0.5f;
                // Линия вставки вместо подсветки всего узла.
                const float y = after ? r.Max.y : r.Min.y;
                ImGui::GetWindowDrawList()->AddLine(ImVec2(r.Min.x, y), ImVec2(r.Max.x, y),
                                                    ImGui::GetColorU32(ImGuiCol_DragDropTarget), 2.0f);
                if (ImGui::AcceptDragDropPayload(kPgDragType, ImGuiDragDropFlags_AcceptNoDrawDefaultRect))
                    drop = PgDrop{ from, path, after };
            }
        }
        ImGui::EndDragDropTarget();
    }
}
}


namespace ucuda::ui_detail {
void pg_collect(const PgNode& n, const std::set<std::string>& present, std::vector<std::string>& out) {
    if (n.g) for (const std::string& p : n.g->params) if (present.count(p)) out.push_back(p);
    for (const PgNode& k : n.kids) pg_collect(k, present, out);
}
}


namespace ucuda::ui_detail {
bool pg_header(const std::string& label, const std::string& id, bool in_table, ParamGroups* list,
               const std::string& seg, PgDrop& drop, const PgMenu& menu,
               const std::function<std::vector<std::string>()>& names_of) {
    if (in_table) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); }
    const ImGuiTreeNodeFlags fl = in_table ? ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_LabelSpanAllColumns
                                           : ImGuiTreeNodeFlags_SpanAvailWidth;
    const bool open = ImGui::TreeNodeEx((label + "##pg_" + id).c_str(), fl);
    pg_drag_node(list, id, seg, drop);
    if (menu && ImGui::BeginPopupContextItem(("##pgm_" + id).c_str())) {
        menu(names_of());
        ImGui::EndPopup();
    }
    return open;
}
}


namespace ucuda::ui_detail {
void pg_draw(const PgNode& n, const std::set<std::string>& present,
             const std::function<void(const std::string&)>& item, bool in_table, ParamGroups* list, PgDrop& drop,
             const PgMenu& menu) {
    if (n.g)
        for (const std::string& p : n.g->params)
            if (present.count(p)) item(p);
    for (const PgNode& k : n.kids) {
        if (k.count == 0) continue;
        auto names_of = [&] { std::vector<std::string> v; pg_collect(k, present, v); return v; };
        if (pg_header(k.seg + "  (" + std::to_string(k.count) + ")", k.path, in_table, list, k.seg, drop, menu,
                      names_of)) {
            pg_draw(k, present, item, in_table, list, drop, menu);
            ImGui::TreePop();
        }
    }
}
}


namespace ucuda::ui_detail {
void draw_param_tree(ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table,
                     const std::function<void(const std::vector<std::string>&)>& group_menu) {
    if (groups == nullptr || groups->empty()) {
        for (const std::string& n : names) item(n);
        return;
    }
    const std::set<std::string> present(names.begin(), names.end());
    PgNode root;
    std::set<std::string> grouped;
    for (const ParamGroup& g : *groups) {
        PgNode* cur = &root;
        size_t pos = 0;
        while (pos <= g.path.size()) {
            size_t e = g.path.find('/', pos);
            if (e == std::string::npos) e = g.path.size();
            const std::string seg = g.path.substr(pos, e - pos);
            pos = e + 1;
            auto it = std::find_if(cur->kids.begin(), cur->kids.end(), [&](const PgNode& k) { return k.seg == seg; });
            if (it == cur->kids.end()) {
                PgNode k;
                k.seg = seg;
                k.path = g.path.substr(0, e);
                cur->kids.push_back(std::move(k));
                it = cur->kids.end() - 1;
            }
            cur = &*it;
        }
        cur->g = &g;
        grouped.insert(g.params.begin(), g.params.end());
    }
    if (pg_count(root, present) == 0) {   // ни одна группа к этим именам не относится
        for (const std::string& n : names) item(n);
        return;
    }
    // У параметров и НУ бывают одноимённые группы ("Capacitors") в одном окне — id по списку.
    ImGui::PushID(groups);
    PgDrop drop;
    pg_draw(root, present, item, in_table, groups, drop, group_menu);
    std::vector<std::string> other;
    for (const std::string& n : names) if (!grouped.count(n)) other.push_back(n);
    if (!other.empty() && pg_header("Other  (" + std::to_string(other.size()) + ")", "\x01other", in_table,
                                    nullptr, "Other", drop, group_menu, [&] { return other; })) {
        for (const std::string& n : other) item(n);
        ImGui::TreePop();
    }
    ImGui::PopID();
    // Узлы держат указатели в groups — менять порядок только после отрисовки.
    if (!drop.from.empty()) pg_reorder(*groups, drop.from, drop.to, drop.after);
}
}


namespace ucuda::ui_detail {
void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint) {
    if (!ImGui::CollapsingHeader(header)) return;
    ImGui::Checkbox(checkbox_label, &enabled);
    InputTextStr(path_id, path);
    ImGui::TextDisabled("%s", hint);
}
}


namespace ucuda::ui_detail {
void draw_error_box(const char* id, const std::string& err,
                           int lines) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
    const ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * (float)lines);
    ImGui::InputTextMultiline(id, const_cast<char*>(err.c_str()), err.size() + 1,
                              sz, ImGuiInputTextFlags_ReadOnly);
}
}


namespace ucuda::ui_detail {
void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write) {
    if (!ImGui::MenuItem("Export data...", nullptr, false, !busy)) return;
    if (!cb.pick_save_file_csv) return;
    const std::string path = cb.pick_save_file_csv();
    if (!path.empty() && write) write(path);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo, std::string* y_hi) {
    ViewRangeTarget t;
    t.name = std::move(name);
    t.x_lo = x_lo; t.x_hi = x_hi; t.y_lo = y_lo; t.y_hi = y_hi;
    t.relocate = [x_lo, x_hi, y_lo, y_hi]() {
        ViewRangeSlots s;
        s.x_lo = x_lo; s.x_hi = x_hi; s.y_lo = y_lo; s.y_hi = y_hi;
        return s;
    };
    return t;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string fmt_view_bound(double v, double span) {
    const double tol = std::abs(span) * 1e-6;
    if (tol > 0.0) {
        char buf[64];
        for (int prec = 6; prec <= 17; ++prec) {
            std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
            if (std::abs(parse_ratio_or(buf, v) - v) <= tol) return std::string(buf);
        }
    }
    return fmt_num_shortest(v);
}
}
