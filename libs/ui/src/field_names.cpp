#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_fields.h"
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
void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value,
                                 const std::function<void()>& extra) {
    if (!model) return;
    // ID popup'а обязан быть свой у каждого поля: в блоке Integration шесть
    // полей лежат на ОДНОМ уровне ID-стека, и общий идентификатор склеил бы
    // их меню в одно.
    const std::string popup_id = "##apply_all_" + label + name;
    if (!ImGui::BeginPopupContextItem(popup_id.c_str())) return;
    ImGui::TextDisabled("%s = %s", label.c_str(), value.c_str());
    ImGui::Separator();
    if (extra) {
        extra();
        ImGui::Separator();
    }

    // Сухой прогон: сколько конфигов реально РАЗОЙДЁТСЯ с этим значением.
    // Показываем числом, но НЕ гасим по нему команду: серый пункт читается как
    // подпись, и "0 differ" выглядит как сломанная рассылка, хотя на деле все
    // цели уже с этим значением.
    auto differ_count = [&](const std::vector<BroadcastTarget>* only) {
        return model->broadcast_field(f, name, value, /*apply*/false, label, only, "");
    };
    // Применяет и кладёт отчёт в шапку: адресаты бывают не открыты, и без
    // строки сверху "применилось" неотличимо от "ничего не произошло".
    auto apply_now = [&](const std::vector<BroadcastTarget>* only, const std::string& note) {
        const int n = model->broadcast_field(f, name, value, /*apply*/true, label, only, note);
        model->undo_note = n > 0
            ? ("Applied: " + label + " = " + value + " -> " + note)
            : ("Nothing to apply: " + note + " already at " + value);
        model->undo_note_time = ImGui::GetTime();
    };
    // Подпись склеиваем, а НЕ прогоняем вторым проходом через printf: в шаблон
    // подставляется имя группы, и попади в него '%', результат стал бы
    // спецификацией формата с несуществующим аргументом. Сегодня имена — жёсткие
    // литералы из group_of, но держать это на соглашении незачем.
    auto count_and_item = [&](const std::string& title,
                              const std::vector<BroadcastTarget>* only,
                              const std::string& note) {
        const std::string lbl = title + " (" + std::to_string(differ_count(only)) + " differ)";
        if (ImGui::MenuItem(lbl.c_str())) apply_now(only, note);
    };

    const std::vector<BroadcastTarget> all = model->broadcast_targets(f);

    // 1. Свой тип. Источник берём из панели, которая сейчас рисуется
    //    (broadcast_source_*): по app_mode его не вывести — вкладка Parametric
    //    покрывает сразу Bifurcation, LLE и LS.
    std::vector<BroadcastTarget> same_group;
    std::string group;
    for (const auto& t : all)
        if (t.tab == model->broadcast_source_tab && t.idx == model->broadcast_source_idx) {
            group = t.group;
            break;
        }
    if (!group.empty()) {
        for (const auto& t : all)
            if (t.group == group) same_group.push_back(t);
    }
    if (same_group.size() > 1)
        count_and_item("Apply to all " + group, &same_group, "all " + group);

    // 2. Выбор галочками. Маска живёт в модели и пересобирается, когда
    //    сменилось поле или изменился состав целей (диаграмму добавили/закрыли).
    if (ImGui::BeginMenu("Apply to...")) {
        // Подпись состава: поле + имена целей по порядку. Меняется состав —
        // маска собирается заново, а не переезжает на соседние конфиги.
        std::string sig = std::to_string((int)f);
        for (const auto& t : all) { sig += '\x1f'; sig += t.label; }
        if (model->broadcast_pick_sig != sig ||
            model->broadcast_pick.size() != all.size()) {
            model->broadcast_pick.assign(all.size(), '0');
            model->broadcast_pick_sig = sig;
            // По умолчанию отмечаем свой тип: это и подсказка, что здесь
            // выбирают, и самый частый набор.
            for (size_t i = 0; i < all.size(); ++i)
                if (all[i].group == group) model->broadcast_pick[i] = '1';
        }
        if (ImGui::SmallButton("All")) model->broadcast_pick.assign(all.size(), '1');
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) model->broadcast_pick.assign(all.size(), '0');

        // Кнопка применения — СВЕРХУ, до списка: галочек бывает под два десятка,
        // и внизу она уезжает за край экрана. Кнопка, а не пункт меню: пункт в
        // конце списка читается как подпись, а не как действие.
        {
            std::vector<BroadcastTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->broadcast_pick[i] == '1') sel.push_back(all[i]);
            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                char note[64];
                std::snprintf(note, sizeof(note), "%d selected", (int)sel.size());
                apply_now(&sel, note);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::SameLine();
            if (sel.empty()) {
                ImGui::TextDisabled("nothing ticked");
            } else {
                const int d = differ_count(&sel);
                if (d > 0) ImGui::TextDisabled("%d differ", d);
                else       ImGui::TextDisabled("already at this value");
            }
        }
        ImGui::Separator();

        std::string shown_group;
        for (size_t i = 0; i < all.size(); ++i) {
            if (all[i].group != shown_group) {
                shown_group = all[i].group;
                ImGui::SeparatorText(shown_group.c_str());
            }
            ImGui::PushID((int)i);
            bool on = (model->broadcast_pick[i] == '1');
            if (ImGui::Checkbox("##pick", &on))
                model->broadcast_pick[i] = on ? '1' : '0';
            ImGui::SameLine();
            ImGui::TextUnformatted(all[i].label.c_str());
            ImGui::PopID();
        }

        ImGui::EndMenu();
    }

    // 3. Прежний адресат: активный конфиг каждой вкладки (only = nullptr).
    count_and_item("Apply to all calculation tabs", nullptr, "all tabs");

    draw_undo_menu_item(model);
    ImGui::EndPopup();
}
}


namespace ucuda::ui_detail {
void draw_param_groups_window(SystemEditor& m, const std::vector<std::string>& params,
                              const std::vector<std::string>& vars, const char* id) {
    AppModel::ParamGroupsUi& u = m.pg_ui;
    if (!u.open) return;
    ImGui::SetNextWindowSize(ImVec2(760.0f, 620.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin((std::string("Parameter groups##") + id).c_str(), &u.open)) { ImGui::End(); return; }
    ImGui::TextDisabled("Only how the fields are laid out and labelled in the tabs; the computation does not change.\n"
                        "Nested groups: '/' in the name, e.g. OPA2134/XU1. Drag a group row to reorder it among\n"
                        "its siblings; drag a name onto a group row to move it there.");
    if (ImGui::BeginTabBar("##pg_kind")) {
        const int was = u.kind;
        auto req = [&](int k) { return u.kind_req == k ? ImGuiTabItemFlags_SetSelected : 0; };
        if (ImGui::BeginTabItem("Parameters", nullptr, req(0))) { u.kind = 0; ImGui::EndTabItem(); }
        if (!vars.empty() && ImGui::BeginTabItem("Initial conditions", nullptr, req(1))) { u.kind = 1; ImGui::EndTabItem(); }
        ImGui::EndTabBar();
        u.kind_req = -1;
        if (u.kind != was) { u.sel.clear(); u.target.clear(); u.rename_from.clear(); }
    }
    if (vars.empty()) u.kind = 0;
    ParamGroups& G = u.kind == 1 ? m.ic_groups : m.param_groups;
    const std::vector<std::string>& names = u.kind == 1 ? vars : params;

    InputTextStrHint("##pg_new", u.new_path, "new group, e.g. OPA2134/XU1", 300.0f);
    ImGui::SameLine();
    if (ImGui::Button("Create group")) {
        const std::string p = pg_normalize_path(u.new_path);
        if (!p.empty()) { pg_get(G, p); u.target = p; }
        u.new_path.clear();
    }

    const std::vector<std::string> rows = pg_rows(G);
    const std::set<std::string> present(names.begin(), names.end());
    auto count_in = [&](const std::string& path) {
        int n = 0;
        for (const ParamGroup& g : G)
            if (pg_in_subtree(g.path, path))
                for (const std::string& p : g.params) n += present.count(p) ? 1 : 0;
        return n;
    };

    // Перетаскивание: строка группы (среди соседей) и имя из таблицы (на строку группы).
    static const char* const kRowDrag = "UCUDA_PG_ROW";
    static const char* const kNameDrag = "UCUDA_PG_NAME";
    std::string drop_from, drop_to, drop_name;
    bool drop_after = false;

    ImGui::SeparatorText("Groups");
    if (rows.empty()) ImGui::TextDisabled("No groups: the tabs show a flat list.");
    std::string del, ren_from, ren_to;
    if (!rows.empty() && ImGui::BeginChild("##pg_rows", ImVec2(0, std::min(240.0f, 26.0f * (float)rows.size() + 8.0f)),
                                           ImGuiChildFlags_Borders)) {
        for (const std::string& r : rows) {
            ImGui::PushID(r.c_str());
            const int depth = (int)std::count(r.begin(), r.end(), '/');
            const size_t sl = r.rfind('/');
            const std::string seg = sl == std::string::npos ? r : r.substr(sl + 1);
            ImGui::Indent(16.0f * (float)depth + 1.0f);
            if (u.rename_from == r) {
                InputTextStr("##rn", u.rename_to, 260.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("OK")) { ren_from = r; ren_to = u.rename_to; }
                ImGui::SameLine();
                if (ImGui::SmallButton("Cancel")) u.rename_from.clear();
            } else {
                const std::string text = seg + "  (" + std::to_string(count_in(r)) + ")";
                ImGui::Selectable(text.c_str(), u.target == r, 0, ImVec2(330.0f - 16.0f * (float)depth, 0));
                if (ImGui::IsItemClicked()) u.target = r;
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload(kRowDrag, r.data(), r.size());
                    ImGui::TextUnformatted(seg.c_str());
                    ImGui::EndDragDropSource();
                }
                if (ImGui::BeginDragDropTarget()) {
                    const ImGuiPayload* pk = ImGui::GetDragDropPayload();
                    const ImRect rc(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
                    if (pk && pk->IsDataType(kRowDrag)) {
                        const std::string from((const char*)pk->Data, (size_t)pk->DataSize);
                        if (from != r && pg_parent(from) == pg_parent(r)) {
                            const bool after = ImGui::GetMousePos().y > (rc.Min.y + rc.Max.y) * 0.5f;
                            const float y = after ? rc.Max.y : rc.Min.y;
                            ImGui::GetWindowDrawList()->AddLine(ImVec2(rc.Min.x, y), ImVec2(rc.Max.x, y),
                                                                ImGui::GetColorU32(ImGuiCol_DragDropTarget), 2.0f);
                            if (ImGui::AcceptDragDropPayload(kRowDrag, ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
                                drop_from = from; drop_to = r; drop_after = after;
                            }
                        }
                    } else if (const ImGuiPayload* pn = ImGui::AcceptDragDropPayload(kNameDrag)) {
                        drop_name.assign((const char*)pn->Data, (size_t)pn->DataSize);
                        drop_to = r;
                    }
                    ImGui::EndDragDropTarget();
                }
                ImGui::SameLine(360.0f);
                if (ImGui::SmallButton("Rename")) { u.rename_from = r; u.rename_to = r; }
                ImGui::SameLine();
                if (ImGui::SmallButton("Delete")) del = r;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Removes the group and its subgroups; their fields stay ungrouped.");
            }
            ImGui::Unindent(16.0f * (float)depth + 1.0f);
            ImGui::PopID();
        }
    }
    if (!rows.empty()) ImGui::EndChild();
    if (!drop_from.empty()) pg_reorder(G, drop_from, drop_to, drop_after);
    if (!drop_name.empty()) {
        // Брошено выделенное имя — едут все выделенные.
        if (u.sel.count(drop_name)) {
            for (const std::string& p : names) if (u.sel.count(p)) pg_move(G, p, drop_to);
            u.sel.clear();
        } else pg_move(G, drop_name, drop_to);
    }
    if (!del.empty()) { pg_remove(G, del); if (pg_in_subtree(u.target, del)) u.target.clear(); }
    if (!ren_from.empty()) {
        pg_rename(G, ren_from, ren_to);
        u.rename_from.clear();
        u.target.clear();
    }

    ImGui::SeparatorText(u.kind == 1 ? "Initial conditions" : "Parameters");
    InputTextStr("filter##pg", u.filter, 200.0f);
    std::vector<std::string> shown;
    for (const std::string& p : names)
        if (u.filter.empty() || p.find(u.filter) != std::string::npos
            || pg_group_of(G, p).find(u.filter) != std::string::npos) shown.push_back(p);
    ImGui::SameLine();
    if (ImGui::Button("Select shown")) u.sel.insert(shown.begin(), shown.end());
    ImGui::SameLine();
    if (ImGui::Button("Clear selection")) u.sel.clear();

    // Список целей: все строки групп + "без группы".
    auto group_combo = [&](const char* label, std::string& cur) {
        bool changed = false;
        if (ImGui::BeginCombo(label, cur.empty() ? "(no group)" : cur.c_str())) {
            if (ImGui::Selectable("(no group)", cur.empty())) { cur.clear(); changed = true; }
            for (const std::string& r : rows)
                if (ImGui::Selectable(r.c_str(), cur == r)) { cur = r; changed = true; }
            ImGui::EndCombo();
        }
        return changed;
    };
    ImGui::SetNextItemWidth(260.0f);
    group_combo("##pg_target", u.target);
    ImGui::SameLine();
    int nsel = 0;
    for (const std::string& p : names) nsel += u.sel.count(p) ? 1 : 0;
    ImGui::BeginDisabled(nsel == 0);
    if (ImGui::Button(("Move selected (" + std::to_string(nsel) + ")").c_str())) {
        for (const std::string& p : names)
            if (u.sel.count(p)) pg_move(G, p, u.target);
        u.sel.clear();
    }
    ImGui::EndDisabled();

    if (ImGui::BeginTable("##pg_params", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 24.0f);
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("label (LaTeX)", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("shown as", ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableSetupColumn("group");
        ImGui::TableHeadersRow();
        ImGuiListClipper clip;
        clip.Begin((int)shown.size());
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                const std::string& p = shown[(size_t)i];
                ImGui::PushID(p.c_str());
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                bool s = u.sel.count(p) > 0;
                if (ImGui::Checkbox("##s", &s)) { if (s) u.sel.insert(p); else u.sel.erase(p); }
                ImGui::TableSetColumnIndex(1);
                ImGui::Selectable(p.c_str(), false);
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload(kNameDrag, p.data(), p.size());
                    const int k = u.sel.count(p) ? nsel : 1;
                    if (k > 1) ImGui::Text("%s (+%d)", p.c_str(), k - 1);
                    else ImGui::TextUnformatted(p.c_str());
                    ImGui::EndDragDropSource();
                }
                ImGui::TableSetColumnIndex(2);
                {
                    auto it = m.field_labels.find(p);
                    std::string lab = it == m.field_labels.end() ? std::string() : it->second;
                    ImGui::SetNextItemWidth(-1.0f);
                    const std::string hint = field_label_auto(p);
                    if (InputTextStrHint("##lab", lab, hint.c_str())) {
                        if (lab.find_first_not_of(" \t") == std::string::npos) m.field_labels.erase(p);
                        else m.field_labels[p] = lab;
                    }
                }
                ImGui::TableSetColumnIndex(3);
                draw_field_name(p, &m.field_labels);
                ImGui::TableSetColumnIndex(4);
                std::string g = pg_group_of(G, p);
                ImGui::SetNextItemWidth(-1.0f);
                if (group_combo("##g", g)) pg_move(G, p, g);
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
    ImGui::End();
}
}
