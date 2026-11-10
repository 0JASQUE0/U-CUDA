#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_plot.h"
// Диапазоны видов, вкладки дока, панели графиков, стили точек, палитры, панель тепловой карты.
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
[[nodiscard]] std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) {
    std::vector<ViewRangeTarget> out;
    auto add = [&out](ViewRangeTarget t, const char* kind, const std::string& group) {
        if (!t.relocate || (!t.x_lo && !t.y_lo)) return;   // цель без полей не нужна
        t.kind  = kind;
        t.group = group;
        out.push_back(std::move(t));
    };

    using B = BifurcationDiagramConfig;
    auto& bd = model.bifurcation_session.diagrams;
    for (int i = 0; i < (int)bd.size(); ++i) {
        if (bd[(size_t)i].mode_2d)
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::param_lo_2_text, &B::param_hi_2_text),
                "Parametric 2D", "Bifurcation 2D");
        else if (bd[(size_t)i].colored_1d)
            // Colored 1D: по Y окно гистограммы, и запись туда включает
            // ручной режим — ровно как на самой диаграмме.
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::colored_1d_ymin_text, &B::colored_1d_ymax_text,
                          &B::colored_1d_custom_y),
                "Parametric 1D", "Bifurcation colored 1D");
        else
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text),
                "Parametric 1D", "Bifurcation 1D");
    }

    using L = LLECurveConfig;
    auto& lle = model.lle_session.curves;
    for (int i = 0; i < (int)lle.size(); ++i) {
        if (lle[(size_t)i].mode_2d)
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text,
                          &L::param_lo_2_text, &L::param_hi_2_text),
                "Parametric 2D", "LLE 2D");
        else
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text),
                "Parametric 1D", "LLE 1D");
    }

    using S = LSCurveConfig;
    auto& ls = model.ls_session.curves;
    for (int i = 0; i < (int)ls.size(); ++i) {
        if (ls[(size_t)i].mode_2d)
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text,
                          &S::param_lo_2_text, &S::param_hi_2_text),
                "Parametric 2D", "LS 2D");
        else
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text),
                "Parametric 1D", "LS 1D");
    }

    using M = SignalMetricsConfig;
    auto& met = model.metrics_session.configs;
    for (int i = 0; i < (int)met.size(); ++i) {
        if (met[(size_t)i].mode_2d)
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text,
                          &M::param_lo_2_text, &M::param_hi_2_text),
                "Parametric 2D", "Metrics 2D");
        else
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text),
                "Parametric 1D", "Metrics 1D");
    }

    // DFT — свой тип: по X свип параметра, а по Y частоты, и подставлять туда
    // диапазон параметра со второй оси 2D-карты нельзя.
    using D = Dft1DConfig;
    auto& dft = model.dft1d_session.configs;
    for (int i = 0; i < (int)dft.size(); ++i)
        add(vr_target(dft, i, &D::param_lo_text, &D::param_hi_text,
                      &D::freq_lo_text, &D::freq_hi_text),
            "1D DFT", "1D DFT");

    // Basins и FastSync — сетка начальных условий, а не параметрический свип.
    using BA = BasinsConfig;
    auto& bas = model.basins_session.configs;
    for (int i = 0; i < (int)bas.size(); ++i)
        add(vr_target(bas, i, &BA::axis_x_lo_text, &BA::axis_x_hi_text,
                      &BA::axis_y_lo_text, &BA::axis_y_hi_text),
            "IC grid", "Basins");

    using F = FastSyncConfig;
    auto& fs = model.fastsync_session.configs;
    for (int i = 0; i < (int)fs.size(); ++i)
        add(vr_target(fs, i, &F::axis_x_lo_text, &F::axis_x_hi_text,
                      &F::axis_y_lo_text, &F::axis_y_hi_text),
            "IC grid", "Fast Synchro");

    using O = OrderConfig;
    auto& ord = model.order_session.configs;
    for (int i = 0; i < (int)ord.size(); ++i) {
        if (ord[(size_t)i].two_d)
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text,
                          &O::axis_y_lo_text, &O::axis_y_hi_text),
                "Order 2D", "Order 2D");
        else
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text),
                "Order 1D", "Order 1D");
    }

    add(vr_target_fixed("Shared config",
                        &model.custom_session.shared.axis_x_lo_text,
                        &model.custom_session.shared.axis_x_hi_text,
                        &model.custom_session.shared.axis_y_lo_text,
                        &model.custom_session.shared.axis_y_hi_text),
        "Custom", "Custom");
    return out;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) {
    std::vector<ViewRangeTarget> all = all_view_range_targets(model);
    std::string kind;
    for (const auto& t : all)
        if (t.x_lo && t.x_lo == source.x_lo) { kind = t.kind; break; }
    std::vector<ViewRangeTarget> out;
    if (kind.empty()) return out;
    for (auto& t : all)
        if (t.kind == kind) {
            if (t.x_lo == source.x_lo) out.insert(out.begin(), std::move(t));
            else                       out.push_back(std::move(t));
        }
    return out;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) {
    const AxisInfo& ax = swapped ? y : x;   // ось ДАННЫХ X
    const AxisInfo& ay = swapped ? x : y;   // ось ДАННЫХ Y
    // invert — только про отрисовку, view_min/view_max остаются упорядоченными;
    // min/max здесь страховка от конфига, пришедшего из чужой руки.
    ViewRangeValues v;
    // ±log-ось (бассейны): вид живёт в координатах узлов, а значения по ним идут немонотонно
    // (-hi..-lo, +lo..+hi) — диапазоном [lo, hi] его не записать, такую ось не трогаем.
    if (!ax.symlog) {
        const double span = ax.view_max - ax.view_min;
        v.has_x = true;
        v.x_lo = fmt_view_bound((std::min)(ax.view_min, ax.view_max), span);
        v.x_hi = fmt_view_bound((std::max)(ax.view_min, ax.view_max), span);
    }
    if (src_has_y && !ay.symlog) {
        const double span = ay.view_max - ay.view_min;
        v.has_y = true;
        v.y_lo = fmt_view_bound((std::min)(ay.view_min, ay.view_max), span);
        v.y_hi = fmt_view_bound((std::max)(ay.view_min, ay.view_max), span);
    }
    return v;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeValues values_from_target(const ViewRangeTarget& t) {
    ViewRangeValues v;
    v.source = "fields";
    if (t.x_lo && t.x_hi) { v.has_x = true; v.x_lo = *t.x_lo; v.x_hi = *t.x_hi; }
    if (t.y_lo && t.y_hi) { v.has_y = true; v.y_lo = *t.y_lo; v.y_hi = *t.y_hi; }
    return v;
}
}


namespace ucuda::ui_detail {
void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) {
    if (v.has_x && t.x_lo && t.x_hi) { *t.x_lo = v.x_lo; *t.x_hi = v.x_hi; }
    if (v.has_y && t.y_lo && t.y_hi) {
        *t.y_lo = v.y_lo; *t.y_hi = v.y_hi;
        if (t.y_manual) *t.y_manual = true;
    }
}
}


namespace ucuda::ui_detail {
void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) {
    if (targets.empty()) return;
    count = (std::min)(count, targets.size());

    // Снимок полей цели. Держим не адреса, а способ найти их заново.
    struct Snap {
        std::function<ViewRangeSlots()> relocate;
        bool has_x = false, has_y = false, has_flag = false;
        std::string x_lo, x_hi, y_lo, y_hi;
        bool flag = false;
    };
    auto capture = [](const ViewRangeTarget& t) {
        Snap p;
        p.relocate = t.relocate;
        if (t.x_lo && t.x_hi) { p.has_x = true; p.x_lo = *t.x_lo; p.x_hi = *t.x_hi; }
        if (t.y_lo && t.y_hi) { p.has_y = true; p.y_lo = *t.y_lo; p.y_hi = *t.y_hi; }
        if (t.y_manual)       { p.has_flag = true; p.flag = *t.y_manual; }
        return p;
    };
    // Отмена и повтор — одна процедура, отличаются только снимком.
    auto writer = [](std::vector<Snap> snaps) {
        return [snaps = std::move(snaps)]() {
            for (const auto& p : snaps) {
                const ViewRangeSlots s = p.relocate();
                if (p.has_x && s.x_lo && s.x_hi) { *s.x_lo = p.x_lo; *s.x_hi = p.x_hi; }
                if (p.has_y && s.y_lo && s.y_hi) { *s.y_lo = p.y_lo; *s.y_hi = p.y_hi; }
                if (p.has_flag && s.y_manual)    *s.y_manual = p.flag;
            }
        };
    };

    std::vector<Snap> before, after;
    for (size_t i = 0; i < count; ++i) {
        const ViewRangeTarget& t = targets[i];
        Snap p = capture(t);
        // Пишем ТОЛЬКО то, что умеем вернуть: цель без relocate откатить нечем,
        // и запись в неё была бы дырой в отмене — поля изменились, а Ctrl+Z про
        // них не знает. Сегодня такие цели не появляются (обе фабрики relocate
        // ставят), но инвариант лучше держать кодом, чем соглашением.
        if (!p.relocate || (!p.has_x && !p.has_y)) continue;
        before.push_back(std::move(p));
        apply_view_values(t, vals);
    }
    if (!model || before.empty()) return;
    // "После" снимаем по тем же целям и уже после записи — так повтор кладёт
    // ровно то, что команда посчитала, без второго прохода по осям.
    for (size_t i = 0; i < count; ++i) {
        Snap p = capture(targets[i]);
        if (p.relocate && (p.has_x || p.has_y)) after.push_back(std::move(p));
    }

    std::string what = std::string("calculation range from ") + vals.source;
    if (count > 1)                     what += " (" + std::to_string((int)count) + " diagrams)";
    else if (!targets[0].name.empty()) what += " (" + targets[0].name + ")";
    model->push_undo(what, writer(std::move(before)), writer(std::move(after)));
    // Отчёт в шапку — как у рассылки полей. Команда правит и диаграммы в других
    // окнах, которые сейчас не видны, и без строки это неотличимо от "ничего не
    // произошло".
    model->undo_note = "Applied: " + what;
    model->undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool view_range_bound(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && (targets[0].x_lo != nullptr || targets[0].y_lo != nullptr);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && targets[0].y_lo != nullptr;
}
}


namespace ucuda::ui_detail {
void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model) {
    if (!view_range_bound(targets)) { draw_undo_menu_item(model); return; }
    ImGui::Separator();
    std::string one = "Calculation range from view";
    if (targets.size() > 1) one += " (" + targets[0].name + ")";
    if (ImGui::MenuItem(one.c_str(), "Ctrl+T"))
        run_view_range_cmd(targets, 1,
                           values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Writes the visible axis range into this diagram's sweep\n"
                          "range fields. Press Run to recompute the zoomed region\n"
                          "at full resolution.");
    // Ctrl+Shift+T — все диаграммы ТОГО ЖЕ типа, а не только наложенные в этом
    // окне: приблизив 2D-бифуркацию, обычно хотят увидеть тот же участок и на
    // 2D LLE с 2D LS. Набор считается по общему перечислению (см.
    // view_range_same_kind), поэтому в него попадают и другие окна.
    if (model) {
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        if (kin.size() > 1) {
            // Сразу после Ctrl+Z / Ctrl+Shift+Z источником становятся ПОЛЯ, а
            // не вид: поля только что откатили, картинка осталась зумленной, и
            // разослать "из вида" значило бы отменить отмену. Пишем это прямо
            // в пункте меню — угадывать источник пользователь не должен.
            const bool from_fields = model->undo_just_happened();
            char lbl[160];
            std::snprintf(lbl, sizeof(lbl), "Calculation range from %s - all %s (%d)",
                          from_fields ? "fields" : "view",
                          kin[0].kind.c_str(), (int)kin.size());
            if (ImGui::MenuItem(lbl, "Ctrl+Shift+T")) {
                const ViewRangeValues v = from_fields
                        ? values_from_target(kin[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(kin, kin.size(), v, model);
            }
            if (from_fields && ImGui::IsItemHovered())
                ImGui::SetTooltip("Undo just restored the sweep fields, so this sends\n"
                                  "THOSE values on. The plot is still zoomed; sending the\n"
                                  "view instead would undo the undo.");
        }

        // Точный выбор галочками — для случаев, когда "тот же тип" не тот набор:
        // например, подтянуть к 2D-карте ещё и Basins, у которых оси другого
        // смысла и в автоматический набор они не входят.
        std::vector<ViewRangeTarget> all = all_view_range_targets(*model);
        if (!all.empty() && ImGui::BeginMenu("Calculation range to...")) {
            std::string sig;
            for (const auto& t : all) { sig += '\x1f'; sig += t.group; sig += t.name; }
            if (model->vr_pick_sig != sig || model->vr_pick.size() != all.size()) {
                model->vr_pick.assign(all.size(), '0');
                model->vr_pick_sig = sig;
                // По умолчанию отмечаем свой тип: это и подсказка, что здесь
                // выбирают, и самый частый набор.
                for (size_t i = 0; i < all.size(); ++i)
                    for (const auto& k : kin)
                        if (all[i].x_lo && all[i].x_lo == k.x_lo) model->vr_pick[i] = '1';
            }
            if (ImGui::SmallButton("All"))  model->vr_pick.assign(all.size(), '1');
            ImGui::SameLine();
            if (ImGui::SmallButton("None")) model->vr_pick.assign(all.size(), '0');

            std::vector<ViewRangeTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->vr_pick[i] == '1') sel.push_back(all[i]);

            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                const ViewRangeValues v = model->undo_just_happened()
                        ? values_from_target(targets[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(sel, sel.size(), v, model);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::Separator();

            std::string shown_group;
            for (size_t i = 0; i < all.size(); ++i) {
                if (all[i].group != shown_group) {
                    shown_group = all[i].group;
                    ImGui::SeparatorText(shown_group.c_str());
                }
                ImGui::PushID((int)i);
                bool on = (model->vr_pick[i] == '1');
                if (ImGui::Checkbox("##pick", &on))
                    model->vr_pick[i] = on ? '1' : '0';
                ImGui::SameLine();
                ImGui::TextUnformatted(all[i].name.c_str());
                ImGui::PopID();
            }
            ImGui::EndMenu();
        }
    }
    draw_undo_menu_item(model);
}
}


namespace ucuda::ui_detail {
void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model) {   // без умолчаний — см. draw_view_range_menu
    if (!active || !view_range_bound(targets)) return;
    const ImGuiIO& io = ImGui::GetIO();
    // С Alt не реагируем вовсе: Ctrl+Alt+T — не наше сочетание, и тихо
    // трактовать его как Ctrl+T значит переписать диапазон по случайному нажатию.
    if (!io.KeyCtrl || io.KeyAlt || io.WantTextInput) return;
    if (!ImGui::IsKeyPressed(ImGuiKey_T, false)) return;
    if (io.KeyShift && model) {
        // Ctrl+Shift+T — весь тип целиком (см. draw_view_range_menu). Если
        // источник в общем перечислении не нашёлся, падаем на прежнее
        // поведение — наложенные в этом окне диаграммы.
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        const std::vector<ViewRangeTarget>& dst = kin.empty() ? targets : kin;
        // Сразу после отката источник — поля, а не вид (см. меню).
        const ViewRangeValues v = model->undo_just_happened()
                                ? values_from_target(dst[0])
                                : values_from_axes(x, y, swapped, src_y_is_input(targets));
        run_view_range_cmd(dst, dst.size(), v, model);
        return;
    }
    run_view_range_cmd(targets, 1,
                       values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool plot_window_active() {
    return ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows |
                                  ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
}
}


namespace ucuda::ui_detail {
void draw_label_rename(const char* label_with_id, std::string& label,
                              float width) {
    ImGui::SetNextItemWidth(width);
    InputTextStr(label_with_id, label);
}
}


namespace ucuda::ui_detail {
void track_dock_selection(AppModel& model) {
    const int mode = (int)model.app_mode;
    std::vector<std::string>& mem = model.dock_selected_by_mode[mode];
    if (model.dock_prev_mode != mode) {
        model.dock_prev_mode      = mode;
        model.dock_restore_frames = 3;
        model.dock_restore_budget = 12;
    }
    if (model.dock_restore_frames > 0) {
        --model.dock_restore_frames;
        bool corrected = false;
        for (const std::string& name : mem) {
            ImGuiWindow* w = ImGui::FindWindowByName(name.c_str());
            if (w == nullptr || !w->Active || w->DockNode == nullptr || w->DockNode->TabBar == nullptr) continue;
            ImGuiTabBar* tb = w->DockNode->TabBar;
            if (tb->SelectedTabId != w->TabId && ImGui::TabBarFindTabByID(tb, w->TabId) != nullptr) {
                tb->NextSelectedTabId = w->TabId;
                // Узел снова выбирает окно с фокусом (NavWindow), а фокус после входа у окна,
                // появившегося последним: без переноса фокуса выбор возвращается через кадр.
                ImGui::FocusWindow(w);
                corrected = true;
            }
        }
        // ImGui отдаёт узел окну, появившемуся последним, не сразу: на третьем кадре после
        // входа (Parametric: LLE 2D держится два кадра, затем выбран Metrics 2D). Поправка
        // в последнем кадре без продления проигрывает этому переключению, поэтому, пока
        // выбор приходится править, восстановление идёт ещё два кадра (не больше бюджета).
        if (corrected && model.dock_restore_frames < 2 && model.dock_restore_budget > 0) {
            model.dock_restore_budget -= 2 - model.dock_restore_frames;
            model.dock_restore_frames  = 2;
        }
        return;
    }
    mem.clear();
    for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows)
        if (w->Active && w->DockIsActive && w->DockNode != nullptr && w->DockNode->TabBar != nullptr
            && w->DockNode->TabBar->SelectedTabId == w->TabId)
            mem.push_back(w->Name);
}
}


namespace ucuda::ui_detail {
TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select) {
    TabBarResult r;
    // Выбор после закрытия вкладки (между кадрами — в хранилище ImGui, ключ от bar_id).
    // Под номер закрытой вкладки встаёт следующий конфиг, ImGui видит его как новую вкладку
    // и по AutoSelectNewTabs выбрал бы его, даже если закрывали не активную.
    ImGuiStorage* st = ImGui::GetStateStorage();
    const ImGuiID pending_key = ImGui::GetID(bar_id) ^ 0x5e1ec7u;
    const int pending = st->GetInt(pending_key, -1);
    if (pending >= 0) {
        st->SetInt(pending_key, -1);
        if (request_select < 0 && pending < n) request_select = pending;
    }
    if (!ImGui::BeginTabBar(bar_id, ImGuiTabBarFlags_AutoSelectNewTabs |
                                    ImGuiTabBarFlags_FittingPolicyScroll))
        return r;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        bool open = true;
        // ID завязан на индекс, а видимая часть — на label: пользователь видит
        // свежее имя сразу после редактирования.
        const std::string tab_id = label(i) + "###" + item_id_prefix + std::to_string(i);
        // Запрещаем закрывать вкладку, чей расчёт сейчас идёт, и ЕДИНСТВЕННУЮ
        // вкладку. Пустой список ни одна сессия не переживает: загрузчики при
        // следующем входе (перезапуск, смена системы) кладут дефолтный конфиг,
        // "+" на пустом списке — тоже дефолт, и настройки последней вкладки
        // терялись. Так же давно устроен NetworkSession::remove_config.
        const bool can_close = !(in_flight && running_index == i) && n > 1;
        ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
        if (request_select == i) flags |= ImGuiTabItemFlags_SetSelected;
        if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr, flags)) {
            r.active = i;
            if (body) body(i);
            ImGui::EndTabItem();
        }
        if (!open) r.to_remove = i;
        ImGui::PopID();
    }
    // Кнопка "+" справа: Trailing удерживает её в конце, NoTooltip убирает
    // дефолтную подсказку.
    if (!in_flight && on_add) {
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing |
                                      ImGuiTabItemFlags_NoTooltip))
            on_add();
    }
    ImGui::EndTabBar();
    if (r.to_remove >= 0) {
        // Активная остаётся активной (её номер сдвигается, если она правее закрытой);
        // закрыли саму активную — выбрать ту, что встала на её место (или последнюю).
        const int a = r.active, i = r.to_remove;
        const int next = (a > i) ? a - 1 : (a == i || a < 0) ? std::min(i, n - 2) : a;
        st->SetInt(pending_key, next);
    }
    return r;
}
}


namespace ucuda::ui_detail {
void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued) {
    const ImGuiIO& io = ImGui::GetIO();

    bool do_run = false;
    if (busy) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(run_w, 0));
        ImGui::EndDisabled();
    } else {
        if (no_active) ImGui::BeginDisabled();
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(run_w, 0));
        if (no_active) ImGui::EndDisabled();
    }
    if (!busy && !no_active && io.KeyCtrl && !io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        do_run = true;
    if (do_run && on_run) on_run();

    if (extra_after_run) extra_after_run();

    auto& picks_by_popup = ui_resource<std::map<std::string, std::vector<std::vector<bool>>>, 3023>();
    std::vector<std::vector<bool>>& picks = picks_by_popup[popup_id];
    if (picks.size() != groups.size()) picks.assign(groups.size(), {});
    for (size_t g = 0; g < groups.size(); ++g)
        if (picks[g].size() != (size_t)groups[g].n)
            picks[g].assign((size_t)groups[g].n, true);

    auto run_all_marked = [&]() {
        for (size_t g = 0; g < groups.size(); ++g)
            for (size_t i = 0; i < picks[g].size(); ++i)
                if (picks[g][i] && groups[g].enqueue) groups[g].enqueue((int)i);
        if (start_queue) start_queue();
    };
    if (!block_run_all && io.KeyCtrl && io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_all_marked();

    ImGui::SameLine();
    if (block_run_all) ImGui::BeginDisabled();
    if (ImGui::Button("Run all... (Ctrl+Shift+R)")) ImGui::OpenPopup(popup_id);
    if (block_run_all) ImGui::EndDisabled();
    if (queued > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", queued);
    }
    if (ImGui::BeginPopup(popup_id)) {
        ImGui::TextDisabled("Sequential (one CUDA context).");
        for (size_t g = 0; g < groups.size(); ++g) {
            if (picks[g].empty()) continue;
            if (groups[g].title) ImGui::SeparatorText(groups[g].title);
            for (size_t i = 0; i < picks[g].size(); ++i) {
                bool v = picks[g][i];
                const std::string lbl = groups[g].label((int)i) + "###" +
                                        groups[g].pick_id + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[g][i] = v;
            }
        }
        ImGui::Separator();
        if (ImGui::Button("All"))  { for (auto& p : picks) for (auto&& b : p) b = true;  }
        ImGui::SameLine();
        if (ImGui::Button("None")) { for (auto& p : picks) for (auto&& b : p) b = false; }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            run_all_marked();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}
}


namespace ucuda::ui_detail {
void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind) {
    view.pad_x       = false;   // данные вплотную к боковым рамкам
    view.show_zero_x = false;
    switch (kind) {
    case ParamPlotKind::Bifurcation:
        view.points_mode       = true;
        view.point_size_px     = 2.0f;
        view.imdraw_lines      = false;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "X";
        break;
    case ParamPlotKind::LLE:
    case ParamPlotKind::LS:
        view.points_mode       = false;   // непрерывная линия
        view.line_thickness_px = 1.5f;
        // glLineWidth в core-profile клампится драйвером до 1px, поэтому
        // линии идут через ImDrawList (см. Plot2DView::imdraw_lines).
        view.imdraw_lines      = true;
        view.show_zero_y       = true;    // λ=0 — граница хаос/порядок
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "lambda";
        break;
    case ParamPlotKind::Metrics:
        // Линия, как у LLE, но без нулевой линии: у амплитуд и частот ноль
        // ничего не разделяет. Имя Y ставит draw_metrics_plot по выбранной метрике.
        view.points_mode       = false;
        view.line_thickness_px = 1.5f;
        view.imdraw_lines      = true;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "metric";
        break;
    }
}
}


namespace ucuda::ui_detail {
bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body) {
    const std::string sfx = id_suffix;
    bool changed = false;
    if (ImGui::Checkbox((std::string(label) + "##" + sfx).c_str(), &enabled))
        changed = true;

    const std::string pop_id = "##stylepop_" + sfx;
    if (enabled) {
        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::ArrowButton(("##styleopen_" + sfx).c_str(), ImGuiDir_Down))
            ImGui::OpenPopup(pop_id.c_str());
    }
    if (ImGui::BeginPopup(pop_id.c_str())) {
        if (body) changed |= body();
        ImGui::EndPopup();
    }
    return changed;
}
}


namespace ucuda::ui_detail {
void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip) {
    const std::string s = sfx;
    draw_style_toolbar("Custom line style", sfx, pr.custom_line_style,
        [&pr, &s]() {
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Line width##" + s).c_str(), &pr.line_width, 0.1f, 5.0f, "%.2f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + s).c_str(), &pr.alpha, 0.0f, 1.0f, "%.2f");
            return ch;
        });
    ImGui::SameLine();
    ImGui::Checkbox(("Points##" + s).c_str(), &pr.draw_points);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\nWith Custom line style on too: points on top of the line.", points_tip);
    if (pr.draw_points) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat(("size##" + s + "pt").c_str(), &pr.point_size, 0.5f, 8.0f, "%.1f");
        if (pr.custom_line_style) {   // своя альфа точек — при Custom line style (у линии своя)
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            ImGui::SliderFloat(("alpha##" + s + "pt").c_str(), &pr.point_alpha, 0.0f, 1.0f, "%.2f");
        }
    }
    projection_views(pr).view2d->imdraw_lines      = pr.custom_line_style;
    projection_views(pr).view2d->gpu_thick_lines   = true;
    projection_views(pr).view2d->line_thickness_px = pr.line_width;
    projection_views(pr).view2d->points_mode       = pr.draw_points;
    projection_views(pr).view2d->points_with_lines = pr.draw_points && pr.custom_line_style;
    projection_views(pr).view2d->point_size_px     = pr.point_size;
    projection_views(pr).view2d->point_alpha       = (pr.draw_points && pr.custom_line_style) ? pr.point_alpha : -1.0f;
}
}


namespace ucuda::ui_detail {
bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) {
    const std::string sfx = id_suffix;
    return draw_style_toolbar("Custom point style", id_suffix, bd.custom_point_style,
        [&bd, &sfx]() {
            if (bd.point_marker < 0 || bd.point_marker >= kPointMarkerCount) bd.point_marker = 0;
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::Combo(("Marker##" + sfx).c_str(), &bd.point_marker,
                               kPointMarkerNames, kPointMarkerCount);
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Point size##" + sfx).c_str(), &bd.point_size,
                                     0.5f, 12.0f, "%.1f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + sfx).c_str(), &bd.point_alpha,
                                     0.0f, 1.0f, "%.2f");
            return ch;
        });
}
}


namespace ucuda::ui_detail {
void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd) {
    view.point_marker  = bd.custom_point_style ? bd.point_marker : -1;
    view.point_size_px = bd.custom_point_style ? bd.point_size   : 2.0f;
}
}


namespace ucuda::ui_detail {
void apply_snap_x(Plot2DView& view, double lo, double hi, int n) {
    view.snap_x_to_grid = true;
    // Вырожденный диапазон — это не сетка: снап по нему сажал бы курсор на
    // узлы, которых в данных нет. Проверка здесь, в единственной точке входа.
    if (n > 1 && hi != lo) {
        view.snap_x_min = lo;
        view.snap_x_max = hi;
        view.snap_x_n   = n;
    } else {
        view.snap_x_n = 0;
    }
}
}


namespace ucuda::ui_detail {
[[nodiscard]] double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation) {
    if (continuation || reverse) {
        // Та же функция, что у cont-ядер и cont_sweep_value: конвенция continuation
        // живёт в ucuda_node_value_cont (configCUDA.h). Гард lo/hi > 0 на лог-ветке —
        // сюда может долететь живой чекбокс без прогона, а log10(<=0) отравил бы ось NaN'ом.
        const bool log_ok_cont = log_scale && lo > 0.0 && hi > 0.0;
        return (double)ucuda_node_value_cont(k, n, (numb)lo, (numb)hi, log_ok_cont, reverse);
    }
    if (log_scale && lo > 0.0 && hi > 0.0)
        return (double)ucuda_node_value_log(k, n, (numb)lo, (numb)hi);
    return (double)ucuda_node_value(k, n, (numb)lo, (numb)hi);
}
}


namespace ucuda::ui_detail {
void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) {
    int  shared_kind = -2;   // -2 = ещё не видели, -1 = смешанные
    int  shared_idx  = -2;
    bool shared_log = false, log_seen = false, log_mismatch = false;

    for (const auto& m : members) {
        const int kidx = (m.kind == 2) ? -1 : m.index;
        if (shared_kind == -2) { shared_kind = m.kind; shared_idx = kidx; }
        else if (shared_kind != m.kind || shared_idx != kidx) { shared_kind = -1; shared_idx = -1; }

        if (!log_seen) { shared_log = m.log_scale; log_seen = true; }
        else if (shared_log != m.log_scale) { log_mismatch = true; }
    }

    view.x_fit_use_explicit = false;
    view.x_axis.log_scale = shared_log && !log_mismatch;

    if (shared_kind == 0 && shared_idx < 0)
        view.x_axis.name = "s";
    else if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)params.size())
        view.x_axis.name = params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)vars.size())
        view.x_axis.name = vars[shared_idx] + " (IC)";
    else if (shared_kind == 2)
        view.x_axis.name = "h";
    else
        view.x_axis.name = "parameter";
}
}


namespace ucuda::ui_detail {
void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text) {
    if (have_result && res_n > 1 && res_hi != res_lo) {
        apply_snap_x(view, res_lo, res_hi, res_n);
        return;
    }
    // Без прогона сетку берём из полей конфига — но ТОЛЬКО если они реально
    // прочитались. Раньше здесь стояли дефолты parse_ratio_or(lo, 0.0) и
    // (hi, 1.0): на непрочитанных полях снап включался с чужой сеткой [0, 1],
    // и курсор садился на её узлы (k/(n-1)) вместо значений свипа.
    const int n = parse_int_or(n_text, 0);
    double lo = 0.0, hi = 0.0;
    if (n > 1 && parse_num_ok(lo_text, lo) && parse_num_ok(hi_text, hi) && lo != hi)
        apply_snap_x(view, lo, hi, n);
    else
        apply_snap_x(view, 0.0, 1.0, 0);   // n = 0 -> снап выключен
}
}


namespace ucuda::ui_detail {
void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb) {
    if (!ImGui::BeginMenu("Export data...")) return;
    bool any = false;
    for (int i = 0; i < n; ++i) {
        if (!ready(i)) continue;
        any = true;
        char item[192];
        std::snprintf(item, sizeof(item), "%s##exp_%s_%d", label(i).c_str(), id_tag, i);
        if (ImGui::MenuItem(item, nullptr, false, !busy(i))) {
            if (cb.pick_save_file_csv) {
                std::string path = cb.pick_save_file_csv();
                if (!path.empty()) do_export(i, path);
            }
        }
    }
    if (!any) ImGui::TextDisabled("(no completed runs)");
    ImGui::EndMenu();
}
}


namespace ucuda::ui_detail {
HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx) {
    auto& slot = map[idx];
    if (!slot) {
        slot = std::make_unique<HeatmapView>();
        slot->colormap = (HeatmapColormap)colormap_id_or(cfg_colormap, app_default_colormap);
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
}
}


namespace ucuda::ui_detail {
void invalidate_view(HeatmapView& hv) { hv.invalidate(); }
}


namespace ucuda::ui_detail {
void invalidate_view(Plot2DView& v)   { v.view_valid = false;  v.series_generation = -1; }
}


namespace ucuda::ui_detail {
void invalidate_view(Plot3DView& v)   { v.view_valid = false;  v.series_generation = -1; }
}


namespace ucuda::ui_detail {
bool session_reloaded(const AppModel& model, AppModel::AppMode mode, int& seen) {
    const int epoch = model.session_epoch[(int)mode];
    if (seen == epoch) return false;
    seen = epoch;
    return true;
}
}


namespace ucuda::ui_detail {
std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series) {
    auto& cache = ui_resource<std::map<int, std::vector<std::vector<double>>>, 3024>();
    auto& bufs = cache[window_id];
    if (bufs.size() != n_series) bufs.assign(n_series, {});
    return bufs;
}
}


namespace ucuda::ui_detail {
std::vector<double>& window_node_buf(int window_id) {
    auto& cache = ui_resource<std::map<int, std::vector<double>>, 3025>();
    return cache[window_id];
}
}


namespace ucuda::ui_detail {
void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation) {
    auto& nb = window_node_buf(window_id);
    nb.clear();
    if (npts > 1) {
        nb.reserve((size_t)npts);
        for (int k = 0; k < npts; ++k)
            nb.push_back(sweep_value_at(k, npts, lo, hi, log_scale, reverse, continuation));
    }
    view.snap_x_nodes      = nb.empty() ? nullptr : nb.data();
    view.snap_x_node_count = (int)nb.size();
}
}


namespace ucuda::ui_detail {
void colormap_strip_item(int id, float w, float h) {
    constexpr int kSeg = 24;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    for (int i = 0; i < kSeg; ++i) {
        const float t0 = (float)i / kSeg, t1 = (float)(i + 1) / kSeg;
        const ImU32 c0 = cmap_sample_id(t0, id);
        const ImU32 c1 = cmap_sample_id(t1, id);
        // +0.5 к правой границе — иначе между сегментами проступают щели
        // на дробном UI-скейле.
        dl->AddRectFilledMultiColor(ImVec2(p.x + w * t0, p.y),
                                    ImVec2(p.x + w * t1 + 0.5f, p.y + h),
                                    c0, c1, c1, c0);
    }
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(ImGuiCol_Border));
    ImGui::Dummy(ImVec2(w, h));
}
}


namespace ucuda::ui_detail {
bool colormap_combo(const char* label, int* id, float width) {
    bool changed = false;
    const float h = ImGui::GetTextLineHeight();
    const float strip_w = h * 4.0f;

    ImGui::SetNextItemWidth(width);
    // HeightLarge = 20 строк вместо дефолтных 8: набор карт пользователь
    // набирает сам, но и десяток в попапе размером с восемь строк листать
    // неудобно.
    if (!ImGui::BeginCombo(label, colormap_label(*id), ImGuiComboFlags_HeightLarge))
        return false;

    const std::vector<int>& ids = picker_colormap_ids();
    // Показываем текущую карту первой, если её нет во включённом наборе.
    const bool orphan = std::find(ids.begin(), ids.end(), *id) == ids.end();
    auto row = [&](int cid) {
        ImGui::PushID(cid);
        const bool sel = (cid == *id);
        // Selectable без явной ширины растягивается на всю строку — он служит
        // подложкой-хитбоксом, а градиент и подпись рисуются поверх. SameLine()
        // после него встал бы у ПРАВОГО края (offset_from_start_x==0 = «после
        // предыдущего элемента»), поэтому возвращаем курсор в начало строки.
        const ImVec2 p0 = ImGui::GetCursorPos();
        if (ImGui::Selectable("##pick", sel, 0, ImVec2(0, h))) {
            if (!sel) { *id = cid; changed = true; }
        }
        if (sel) ImGui::SetItemDefaultFocus();
        ImGui::SetCursorPos(p0);
        colormap_strip_item(cid, strip_w, h);
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x);
        ImGui::TextUnformatted(colormap_label(cid));
        ImGui::PopID();
    };
    if (orphan) { row(*id); ImGui::Separator(); }
    for (int cid : ids) row(cid);
    // Все галочки сняты — пикер пуст. Текущая карта показана выше отдельной
    // строкой (orphan), так что состояние не тупиковое, но подсказать надо.
    if (ids.empty())
        ImGui::TextDisabled("(no colormaps ticked — pick some in Settings)");

    ImGui::EndCombo();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o) {
    bool changed = false;

    int cmap_idx = (int)hv.colormap;
    if (colormap_combo("Colormap", &cmap_idx, ImGui::GetFontSize() * 12.0f)) {
        hv.colormap = (HeatmapColormap)cmap_idx;
        if (o.persist_colormap) o.persist_colormap(cmap_idx);
        changed = true;
    }

    if (o.extras) o.extras();

    ImGui::SameLine();
    if (ImGui::Checkbox("Autoscale color", &hv.autoscale)) changed = true;
    if (!hv.autoscale) {
        ImGui::SameLine();
        if (InputNumStr("vmin", hv.manual_vmin_text, 80)) changed = true;
        hv.manual_vmin = (float)parse_ratio_or(hv.manual_vmin_text, hv.manual_vmin);
        ImGui::SameLine();
        if (InputNumStr("vmax", hv.manual_vmax_text, 80)) changed = true;
        hv.manual_vmax = (float)parse_ratio_or(hv.manual_vmax_text, hv.manual_vmax);
    }

    if (o.show_swap) {
        ImGui::SameLine();
        if (ImGui::Button(hv.swap_axes ? "Swap axes (on)" : "Swap axes")) {
            hv.swap_axes = !hv.swap_axes;
            changed = true;
        }
    }

    if (o.extras_tail) o.extras_tail();
    return changed;
}
}


namespace ucuda::ui_detail {
void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick) {
    if (n_exponents <= 0) return;
    const int N = n_exponents;
    if (hv.display_exponent_idx != -1 &&
        (hv.display_exponent_idx < 0 || hv.display_exponent_idx >= N))
        hv.display_exponent_idx = 0;

    const std::string preview = (hv.display_exponent_idx == -1)
        ? "sum L_i" : ("L" + std::to_string(hv.display_exponent_idx + 1));
    auto pick = [&](int j) {
        hv.display_exponent_idx = j;
        if (on_pick) on_pick(j);
    };
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::BeginCombo("Exponent", preview.c_str())) {
        for (int j = 0; j < N; ++j) {
            std::string lbl = "L" + std::to_string(j + 1);
            if (ImGui::Selectable(lbl.c_str(), hv.display_exponent_idx == j)) pick(j);
        }
        ImGui::Separator();
        if (ImGui::Selectable("sum L_i", hv.display_exponent_idx == -1)) pick(-1);
        ImGui::EndCombo();
    }
}
}


namespace ucuda::ui_detail {
void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen) {
    const size_t plane_size = (size_t)cact.result_2d.n_pts * (size_t)cact.result_2d.n_pts;
    const int    N          = cact.result_2d.n_exponents;

    if (k == -1) {
        if (cact.sum_cache_gen != cact.data_generation_2d) {
            cact.sum_cache.assign(plane_size, 0.0);
            double smin =  std::numeric_limits<double>::infinity();
            double smax = -std::numeric_limits<double>::infinity();
            for (size_t c2 = 0; c2 < plane_size; ++c2) {
                // Всё, что не колебательный режим (FP/unbound) — под sentinel:
                // раньше проверялось flags < 0, но у LS unbound теперь 0.
                if (c2 < cact.result_2d.flags.size() &&
                    !regime_is_oscillation(cact.result_2d.flags[c2])) {
                    cact.sum_cache[c2] = kSentinelNoData;
                    continue;
                }
                double sum = 0.0;
                for (int j = 0; j < N; ++j)
                    sum += cact.result_2d.values[(size_t)j * plane_size + c2];
                cact.sum_cache[c2] = sum;
                if (sum < smin) smin = sum;
                if (sum > smax) smax = sum;
            }
            cact.sum_cache_min = std::isfinite(smin) ? smin : 0.0;
            cact.sum_cache_max = std::isfinite(smax) ? smax : 0.0;
            cact.sum_cache_gen = cact.data_generation_2d;
        }
        plane = cact.sum_cache.data();
        vmin  = cact.sum_cache_min;
        vmax  = cact.sum_cache_max;
        gen   = cact.data_generation_2d * 64 + N;
    } else {
        plane = cact.result_2d.values.data() + (size_t)k * plane_size;
        vmin  = (k >= 0 && k < (int)cact.result_2d.min_val.size()) ? cact.result_2d.min_val[k] : 0.0;
        vmax  = (k >= 0 && k < (int)cact.result_2d.max_val.size()) ? cact.result_2d.max_val[k] : 0.0;
        gen   = cact.data_generation_2d * 64 + k;
    }
}
}


namespace ucuda::ui_detail {
void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) {
    lo = res_lo;
    hi = res_hi;
    if (hi == lo) {
        lo = parse_ratio_or(lo_text, 0.0);
        hi = parse_ratio_or(hi_text, 1.0);
    }
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


namespace ucuda::ui_detail {
void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc, int* stopped ) {
    fp = unb = osc = 0;
    if (stopped) *stopped = 0;
    for (int f : flags) {
        if      (regime_is_cancelled(f))   { if (stopped) ++*stopped; }
        else if (regime_is_fixed_point(f)) ++fp;
        else if (regime_is_unbound(f))     ++unb;
        else                               ++osc;
    }
}
}


namespace ucuda::ui_detail {
std::function<const char*(int, int)> regime_cell_label(const std::vector<int>& flags,
                                                              int row_stride,
                                                              bool with_oscillation) {
    return [&flags, row_stride, with_oscillation](int ix, int iy) -> const char* {
        const size_t k = (size_t)iy * (size_t)row_stride + (size_t)ix;
        if (k >= flags.size()) return nullptr;
        if (!with_oscillation && regime_is_oscillation(flags[k])) return nullptr;
        return regime_label(flags[k]);
    };
}
}


namespace ucuda::ui_detail {
std::function<bool(int, int)> cancelled_cell_mask(const std::vector<int>& flags, int row_stride) {
    return [&flags, row_stride](int ix, int iy) -> bool {
        const size_t k = (size_t)iy * (size_t)row_stride + (size_t)ix;
        return k < flags.size() && regime_is_cancelled(flags[k]);
    };
}
}


namespace ucuda::ui_detail {
void draw_adaptive_stats(const std::vector<double>& st) {
    if (st.size() < 4) return;
    double acc = 0, rej = 0, frc = 0, hsum = 0, hmin = 0, hmax = 0;
    size_t n = 0;
    for (size_t i = 0; i + 3 < st.size(); i += 4) {
        acc += st[i]; rej += st[i + 1]; frc += st[i + 2];
        const double hm = st[i + 3];
        if (hm > 0) {
            if (n == 0 || hm < hmin) hmin = hm;
            if (n == 0 || hm > hmax) hmax = hm;
            hsum += hm; ++n;
        }
    }
    const double att = acc + rej;
    ImGui::TextDisabled("Adaptive step: mean h %.3g (%.3g..%.3g over points), %.0f accepted, "
                        "%.0f rejected (%.1f%%), %.0f forced",
                        n ? hsum / (double)n : 0.0, hmin, hmax, acc, rej,
                        att > 0 ? 100.0 * rej / att : 0.0, frc);
}
}


namespace ucuda::ui_detail {
void draw_regime_summary(const std::vector<int>& flags, RegimeCounts* cache, int gen) {
    int fp = 0, unb = 0, osc = 0, stopped = 0;
    if (!cache) {
        count_regimes(flags, fp, unb, osc, &stopped);
    } else {
        if (cache->gen != gen || cache->data != (const void*)flags.data() || cache->size != flags.size()) {
            count_regimes(flags, cache->fp, cache->unb, cache->osc, &cache->stopped);
            cache->gen = gen; cache->data = flags.data(); cache->size = flags.size();
        }
        fp = cache->fp; unb = cache->unb; osc = cache->osc; stopped = cache->stopped;
    }
    if (stopped > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                           "Stopped: %d of %d points computed, the rest are grey / blank",
                           (int)flags.size() - stopped, (int)flags.size());
    if (fp == 0 && unb == 0) return;
    ImGui::TextDisabled("regimes: %d %s / %d %s / %d %s",
                        fp,  regime_label(REGIME_FIXED_POINT),
                        unb, regime_label(REGIME_UNBOUND),
                        osc, regime_label(REGIME_OSCILLATION));
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string ic_legend_text(const AnalysisResult& res, size_t k) {
    if (k < res.snapshot.ic_flat.size() && !res.snapshot.ic_flat[k].empty()) {
        const std::vector<double>& v = res.snapshot.ic_flat[k];
        std::string txt = "(";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) txt += ",";
            txt += fmt_tick(v[i]);
        }
        txt += ")";
        return txt;
    }
    return (k < res.ic_text.size()) ? res.ic_text[k] : std::string();
}
}


namespace ucuda::ui_detail {
void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id) {
    if (vars.empty()) return;
    const int N = (int)vars.size();
    // Допустимые значения: -1 (combination) или [0, N-1]. Старые сейвы могли
    // иметь невалидные индексы — clamp в 0.
    if (writable_var < -1 || writable_var >= N) writable_var = 0;
    // Для системы из одной переменной combination сворачивается до x[0] —
    // показывать отдельную опцию бессмысленно, форсим single-var.
    if (N == 1 && writable_var == -1) writable_var = 0;

    // Подпись комбинации — общая combo_var_label (раньше здесь жила её копия).
    const std::string preview =
        (writable_var == -1) ? combo_var_label(vars) : vars[writable_var];

    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo(combo_id, preview.c_str())) {
        for (int i = 0; i < N; ++i) {
            bool sel = writable_var == i;
            if (ImGui::Selectable(vars[i].c_str(), sel)) writable_var = i;
        }
        if (N >= 2) {
            ImGui::Separator();
            const std::string lbl = combo_var_label(vars);
            bool sel_combo = writable_var == -1;
            if (ImGui::Selectable(lbl.c_str(), sel_combo)) writable_var = -1;
        }
        ImGui::EndCombo();
    }
}
}
