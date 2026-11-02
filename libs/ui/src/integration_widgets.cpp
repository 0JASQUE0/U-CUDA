#include "gui_internal.h"
// Блок Integration: Apply to..., выбор схемы, адаптивный шаг, библиотека регуляторов, Newton; списки схем и методов ДАУ по умолчанию.
using namespace ucuda::ui_detail;

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
[[nodiscard]] std::string scheme_group_title(const BuiltinScheme& b) {
    return b.group ? std::string(b.group) : "Order " + std::to_string(b.order);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_is_map(const std::string& name) {
    return name == kMapSchemeName;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<const char*> c_str_list(const std::vector<std::string>& v) {
    std::vector<const char*> out;
    out.reserve(v.size());
    for (const auto& s : v) out.push_back(s.c_str());
    return out;
}
}


namespace ucuda::ui_detail {
static const AppModel* g_scheme_masks = nullptr;
}


namespace ucuda::ui_detail {
void set_scheme_masks(const AppModel* m) { g_scheme_masks = m; }
}


namespace ucuda::ui_detail {
bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick,
                              const std::vector<std::string>* enabled_builtins,
                              AppModel* bc,
                              bool is_map,
                              const std::vector<std::string>* wrapper_schemes,
                              bool broadcast) {
    // Nothing to choose for a discrete map: the step is the right-hand side.
    if (is_map) {
        scheme = kMapSchemeName;
        ImGui::TextDisabled("Map (x_{n+1} = f(x_n)) - no scheme");
        return false;
    }
    bool picked = false;
    auto choose = [&](const std::string& nm) {
        scheme = nm;
        if (on_pick) on_pick(nm);
        picked = true;
    };
    // ДАУ: сессия получила только схемы ДАУ (enabled_builtins_from_record) — их и показываем,
    // схемы ОДУ на такой системе не строятся. Скрытые в Settings не предлагаются (выбранная остаётся).
    if (enabled_builtins && !enabled_builtins->empty()
        && (dae_scheme_from_name((*enabled_builtins)[0], nullptr) || (*enabled_builtins)[0] == kBdfSchemeDae)) {
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo(label, scheme.c_str())) {
            ImGui::SeparatorText("DAE");
            bool ms_header = false;
            for (const auto& nm : *enabled_builtins) {
                if (g_scheme_masks && g_scheme_masks->dae_method_hidden(nm) && scheme != nm) continue;
                if (scheme_is_multistep(nm) && !ms_header) {
                    ImGui::SeparatorText("multistep");
                    ms_header = true;
                }
                if (ImGui::Selectable(nm.c_str(), scheme == nm)) choose(nm);
                if (scheme_is_multistep(nm) && ImGui::IsItemHovered())
                    ImGui::SetTooltip("BDF orders 1-5 (DASSL form), its own driver; index 3 goes through GGL.\n"
                                      "Adaptive step: variable step and order. Fixed step: step h, order 1..5\n"
                                      "ramped up, no step controller. Newton: the tab's Newton fields.");
            }
            ImGui::EndCombo();
        }
        return picked;
    }
    // enabled_builtins null = no filter. Empty = no built-in is ticked: that is still
    // "no filter" only while the combo would otherwise be empty; a system that offers
    // custom KRS or wrappers (e.g. Izhikevich, whose reset lives in custom schemes)
    // gets just those. Currently-selected built-in stays visible even if not enabled,
    // so sessions saved with a now-disabled scheme aren't stranded.
    bool has_other = wrapper_schemes && !wrapper_schemes->empty();
    for (const auto& cs : custom_schemes) if (cs.enabled) { has_other = true; break; }
    const bool filter = enabled_builtins && (!enabled_builtins->empty() || has_other);
    auto is_enabled = [&](const char* m) {
        if (!filter) return true;
        for (const auto& n : *enabled_builtins) if (n == m) return true;
        return false;
    };
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, wrapper_display_name(scheme).c_str())) {
        // Заголовок порядка печатаем лениво — перед ПЕРВОЙ видимой схемой
        // группы, иначе отфильтрованная группа оставила бы пустую шапку.
        std::string shown_group;
        for (const auto& b : kBuiltinSchemes) {
            // Выключенная у системы или скрытая в Settings схема не предлагается, выбранная остаётся.
            if ((!is_enabled(b.name) || (g_scheme_masks && g_scheme_masks->scheme_hidden(b.name)))
                && scheme != b.name) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group = scheme_group_title(b);
            }
            if (ImGui::Selectable(b.name, scheme == b.name)) choose(b.name);
        }
        // Многошаговый BDF 1-5 — свой драйвер, флага scheme_* у него нет: показывается всегда
        // (enabled_builtins_from_record его тоже всегда добавляет), только с адаптивным шагом.
        if (is_enabled(kBdfSchemeOde) || scheme == kBdfSchemeOde) {
            ImGui::SeparatorText("Multistep (adaptive step only)");
            if (ImGui::Selectable(kBdfSchemeOde, scheme == kBdfSchemeOde)) choose(kBdfSchemeOde);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("BDF orders 1-5 with variable step (DASSL form), its own driver.\n"
                                  "Needs Integration -> Adaptive step; stiff problems.");
        }
        // Своя шапка, а не голая черта: у групп порядка и у обёрток заголовки
        // есть, и без неё кастомные читались как хвост последней Order-группы.
        // Выключенная галочкой КРС не предлагается, но уже выбранная остаётся
        // видна — как и у встроенных, чтобы сессия не осиротела.
        bool custom_header = false;
        for (const auto& cs : custom_schemes) {
            if (!cs.enabled && scheme != cs.name) continue;
            if (!custom_header) { ImGui::SeparatorText("Custom"); custom_header = true; }
            if (ImGui::Selectable((cs.name + "##custom").c_str(), scheme == cs.name))
                choose(cs.name);
        }
        // Обёртки — отдельной группой: паспортного порядка в таблице у них
        // нет (у Extr он считается из базы и числа стадий, у Comp вообще
        // известен только в рантайме), поэтому в группировку по Order выше
        // они не встают.
        // Сессии получают только включённые обёртки; выбранная, но выключенная
        // добавляется обратно по той же причине, что и КРС выше.
        std::vector<std::string> sorted;
        if (wrapper_schemes) sorted = *wrapper_schemes;
        {
            ExtrapolationSpec esp;
            CompositionSpec   csp;
            if ((parse_extrapolation_name(scheme, &esp) || parse_composition_name(scheme, &csp))
                && std::find(sorted.begin(), sorted.end(), scheme) == sorted.end())
                sorted.push_back(scheme);
        }
        if (!sorted.empty()) {
            ImGui::SeparatorText("Wrappers");
            // Порядок sort_wrapper_schemes: список сессии мог приехать из файла
            // в старом порядке добавления. Подзаголовок — на смене группы
            // "тип + опорник".
            sort_wrapper_schemes(sorted);
            std::string shown_wgroup;
            for (const auto& nm : sorted) {
                const std::string wg = wrapper_group_title(nm);
                if (wg != shown_wgroup) {
                    if (!wg.empty()) ImGui::TextDisabled("%s", wg.c_str());
                    shown_wgroup = wg;
                }
                const std::string shown = wrapper_display_name(nm);
                if (ImGui::Selectable((shown + "##" + nm).c_str(), scheme == nm))
                    choose(nm);
                // Коэффициенты прямо в списке выбора: именно здесь схему и
                // выбирают, а знать, какие числа за ней стоят, надо ДО того.
                // Значения параметров берутся у bc, когда он есть: символьная
                // gamma без них раскрыта не будет, и отчёт честно скажет "?".
                if (ImGui::IsItemHovered()) {
                    static const std::map<std::string, std::string> kNoParams;
                    const std::string rep = wrapper_coefficients_text(
                        nm, custom_schemes, bc ? bc->param_values : kNoParams);
                    if (!rep.empty())      ImGui::SetTooltip("%s", rep.c_str());
                    else if (shown != nm)  ImGui::SetTooltip("%s", nm.c_str());
                }
            }
        }
        ImGui::EndCombo();
    }
    // Меню вешаем ПОСЛЕ комбо: последним элементом остаётся оно само, и ПКМ
    // попадает по нему, а не по строке раскрытого списка.
    // broadcast = false — комбо не основной схемы (эталон вкладки Order):
    // рассылка BroadcastField::Scheme записала бы значение в ОСНОВНУЮ схему
    // чужих конфигов. PushID(label): ID popup'а собирается из "scheme", и две
    // комбо схемы на одном уровне ID-стека склеивали бы меню в одно окно,
    // рисуя его содержимое дважды (конфликт ID у галочек "Apply to...").
    if (broadcast) {
        ImGui::PushID(label);
        field_apply_all_menu(bc, BroadcastField::Scheme, {}, "scheme", scheme);
        ImGui::PopID();
    }
    return picked;
}
}


namespace ucuda::ui_detail {
void draw_sweep_target_combo(const char* label,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars,
                                    int& par_index, bool& over_var, int& var_index,
                                    bool& over_h,
                                    bool* other_over_h,
                                    bool note_when_empty,
                                    float width,
                                    bool allow_s,
                                    AppModel* bc,
                                    BroadcastField which,
                                    bool allow_h,
                                    int* ad_axis,
                                    const AdaptiveSettings* ad) {
    // Адаптивный шаг: h — шаг вывода, его не свипают; вместо него ось может свипать
    // настройку шага (*ad_axis — AdaptiveAxisKind движка, 0 — обычная цель).
    const bool ad_on = ad_axis && ad && ad->enabled;
    if (ad_axis && !ad_on) *ad_axis = 0;
    if (ad_on) allow_h = false;
    const bool ad_sel = ad_on && *ad_axis != 0;
    // A map has no step, so there is nothing to sweep over.
    if (!allow_h) over_h = false;
    if (params.empty() && vars.empty() && note_when_empty) {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
        return;
    }
    // Кламп индексов: сохранённая сессия могла прийти от системы с другим
    // числом переменных/параметров. Отрицательный par_index — не мусор, а цель
    // "s"; сбрасываем его только там, где схема a[0] не читает (или сессия
    // пришла от схемы, которая читала).
    if (par_index < 0 && !allow_s) par_index = 0;
    if (par_index >= (int)params.size()) par_index = 0;
    if (var_index < 0 || var_index >= (int)vars.size())   var_index = 0;

    auto ad_name = [&](int kind) -> std::string {
        if (kind == kAdAxisRtol) return "rtol";
        if (kind == kAdAxisAtol) return "atol";
        if (kind == kAdAxisTol)  return "tol (atol = rtol x ratio)";
        AdaptiveCtrlResolved cr;
        std::string cerr;
        const bool known = ad && adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr);
        const int k = kind - kAdAxisCtrl;
        return "controller: " + ((known && k >= 0 && k < (int)cr.par.size()) ? cr.par[(size_t)k] : std::string("?"));
    };
    const std::string preview =
          ad_sel                      ? ad_name(*ad_axis)
        : over_h                      ? std::string("dt (h)")
        : (over_var && !vars.empty()) ? (vars[var_index] + " (IC)")
        : (par_index < 0)             ? std::string("s (a[0])")
        : (!params.empty())           ? params[par_index]
                                      : std::string("?");
    // Значение для ПКМ-рассылки: четыре поля цели одной строкой (см.
    // encode_sweep_target). Меню вешается на комбо в ОБОИХ случаях — и когда
    // список раскрыт, и когда нет: оно принадлежит комбо, а не его строкам.
    auto apply_all_menu = [&]() {
        field_apply_all_menu(bc, which, {}, label,
            encode_sweep_target(par_index, over_var, var_index, over_h));
    };
    ImGui::SetNextItemWidth(width);
    if (!ImGui::BeginCombo(label, preview.c_str())) { apply_all_menu(); return; }

    for (int i = 0; i < (int)params.size(); ++i) {
        const bool sel = !ad_sel && !over_var && !over_h && par_index == i;
        if (ImGui::Selectable(params[i].c_str(), sel)) {
            par_index = i; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (!params.empty() && !vars.empty()) ImGui::Separator();
    for (int i = 0; i < (int)vars.size(); ++i) {
        const bool sel = !ad_sel && over_var && !over_h && var_index == i;
        if (ImGui::Selectable((vars[i] + " (IC)").c_str(), sel)) {
            var_index = i; over_var = true; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }

    if (allow_s) {
        ImGui::Separator();
        if (ImGui::Selectable("s (a[0])", !ad_sel && !over_var && !over_h && par_index < 0)) {
            par_index = -1; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (allow_h) {
        ImGui::Separator();
        if (ImGui::Selectable("dt (h)", over_h)) {
            over_h = true; over_var = false;
            if (other_over_h) *other_over_h = false;   // ровно одна ось = h
        }
    }
    if (ad_on) {
        ImGui::Separator();
        auto item = [&](int kind) {
            if (ImGui::Selectable(ad_name(kind).c_str(), *ad_axis == kind)) {
                *ad_axis = kind; over_var = false; over_h = false;
            }
        };
        item(kAdAxisRtol); item(kAdAxisAtol); item(kAdAxisTol);
        AdaptiveCtrlResolved cr;
        std::string cerr;
        if (adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr))
            for (int k = 0; k < (int)cr.par.size(); ++k) item(kAdAxisCtrl + k);
    }
    ImGui::EndCombo();
    apply_all_menu();
}
}


namespace ucuda::ui_detail {
std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) {
    std::vector<AdaptiveSettings*> v;
    v.push_back(&m.phase_session.adaptive);
    // Custom: источник — общий конфиг вкладки (apply_shared_to_* раздаёт его уровням при
    // запуске); его Phase — вместе с ним, как это делает draw_shared_config.
    v.push_back(&m.custom_session.shared.adaptive);
    v.push_back(&m.custom_session.phase_session.adaptive);
    for (auto& d : m.bifurcation_session.diagrams) v.push_back(&d.adaptive);
    for (auto& c : m.lle_session.curves)           v.push_back(&c.adaptive);
    for (auto& c : m.ls_session.curves)            v.push_back(&c.adaptive);
    for (auto& c : m.basins_session.configs)       v.push_back(&c.adaptive);
    for (auto& c : m.metrics_session.configs)      v.push_back(&c.adaptive);
    return v;
}
}


namespace ucuda::ui_detail {
void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) {
    std::vector<AdaptiveSettings> old;
    for (AdaptiveSettings* p : collect_adaptive_slots(m)) { old.push_back(*p); *p = a; }
    AppModel* mp = &m;
    m.push_undo("adaptive step settings to all tabs",
        [mp, old]() {
            std::vector<AdaptiveSettings*> v = collect_adaptive_slots(*mp);
            if (v.size() == old.size())
                for (size_t i = 0; i < v.size(); ++i) *v[i] = old[i];
        },
        [mp, a]() { for (AdaptiveSettings* p : collect_adaptive_slots(*mp)) *p = a; });
}
}


namespace ucuda::ui_detail {
std::vector<std::pair<std::string, std::string>> user_ctrl_list() {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::lock_guard<std::mutex> lk(L.mu);
    std::vector<std::pair<std::string, std::string>> v;
    for (const AdaptiveUserCtrl& u : L.items) {
#ifdef UCUDA_AD_HAIRER_ONLY
        if (u.kind == kUserCtrlFilter) continue;   // встроенный Filter не собран
#endif
        v.emplace_back(u.name, u.tip);
    }
    return v;
}
}


namespace ucuda::ui_detail {
static const char* const kCtrlBodyHelp =
    "void ucuda_ctrl_custom(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) { <body> }\n"
    "\n"
    "in.h          size of this attempt;  in.t - t at the start of the step\n"
    "in.y0[i]      solution at t;  in.y1[i] - higher-order solution at t + h\n"
    "in.yerr[i]    error estimate y1 - yhat; in.nlo estimates of in.n values each\n"
    "              (DOP853: in.yerr[in.n + i] is the 3rd-order one)\n"
    "in.rtol, in.atol[i], in.c[k] - parameters of this controller, in.q - estimator order\n"
    "in.hmin, in.hmax (+inf: no limit; the driver clamps to it anyway), in.nrej - rejected attempts\n"
    "m.h[j], m.err[j]  accepted steps and their errors, j = 0 the latest (3 kept)\n"
    "m.nacc        accepted steps so far;  m.user[0..7] - free memory kept between steps\n"
    "o.err         error of the attempt (1 = on the tolerance): the norm or its square (Hairer\n"
    "              reports err^2); the driver only checks that it is finite; goes to the log\n"
    "o.accept      1 - take the step, 0 - retry it with o.h\n"
    "o.h           next step, or the step of the retry\n"
    "\n"
    "in.k[0..7]    constants computed by the prepare section (0 if not set)\n"
    "\n"
    "Prepare section (optional), once per trajectory:\n"
    "void ucuda_ctrl_custom_prep(const numb* c, int q, numb* k) { <prepare> }\n"
    "c[] - the parameters, q - estimator order, k[0..7] - constants for the body.\n"
    "Put divisions and powers that depend only on c[] and q here: double division\n"
    "costs as much as several RHS evaluations on GeForce cards.\n"
    "\n"
    "Helpers: ucuda_ctl_err(in) - RMS norm of the estimate, sc_i = atol_i + rtol max(|y0_i|, |y1_i|);\n"
    "ucuda_ctl_lg2err2(in, err2) - log2(err^2) in float, err^2 into err2, no sqrt (\"Hairer fast\");\n"
    "ucuda_ctl_lg2err2_numb(in, err2) - the same in numb (\"Hairer numb\");\n"
    "ucuda_ctl_pow(x, y) - fast x^y (float);  ucuda_ctl_log2(x), ucuda_ctl_exp2(f) - float;\n"
    "ucuda_clamp(x, lo, hi);  ucuda_fmin/fmax (double), ucuda_fminf/fmaxf (float).  numb is double.\n"
    "Index arrays (m.user, in.k, ...) with constants: a computed index puts the state of\n"
    "the GPU thread into local memory.\n"
    "The driver then clamps h to h_max, takes the step at h_min when it would go below\n"
    "(or after max rejects), and retries with at most 0.9 h.";
}


namespace ucuda::ui_detail {
void draw_ctrl_library_window(AppModel& m) {
    // Результат фонового Check — и при закрытом окне, чтобы future не висел.
    if (m.ctrl_lib_check_future.valid()
        && m.ctrl_lib_check_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::pair<bool, std::string> r = m.ctrl_lib_check_future.get();
        m.ctrl_lib_check_ok  = r.first;
        m.ctrl_lib_check_log = r.second;
    }
    if (!m.show_ctrl_library) return;
    ImGui::SetNextWindowSize(ImVec2(940.0f, 640.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Step controller library", &m.show_ctrl_library)) { ImGui::End(); return; }

    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    bool edited = false;
    std::string check_body;
    bool do_check = false;
    {
        std::lock_guard<std::mutex> lk(L.mu);
        std::vector<AdaptiveUserCtrl>& items = L.items;
        if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;

        auto taken = [&](const std::string& s, int self) {
            if (adaptive_find_builtin(s)) return true;
            for (int i = 0; i < (int)items.size(); ++i) if (i != self && items[(size_t)i].name == s) return true;
            return false;
        };
        auto add = [&](AdaptiveUserCtrl u) {
            std::string base = u.name.empty() ? std::string("controller") : u.name;
            u.name = base;
            for (int k = 2; taken(u.name, -1); ++k) u.name = base + " " + std::to_string(k);
            items.push_back(u);
            m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        };

        // Слева — список и кнопки.
        ImGui::BeginChild("##ctrl_list", ImVec2(230.0f, 0.0f), ImGuiChildFlags_Borders);
        if (ImGui::Button("New...")) ImGui::OpenPopup("##ctrl_new");
        if (ImGui::BeginPopup("##ctrl_new")) {
            if (ImGui::Selectable("empty C body")) {
                AdaptiveUserCtrl u;
                u.name = "my controller";
                u.body = "const numb err = ucuda_ctl_err(in);\n"
                         "o.err = err;\n"
                         "o.accept = err <= 1;\n"
                         "o.h = in.h * ucuda_clamp((numb)0.9 * ucuda_ctl_pow(err, (numb)-1 / (in.q + 1)),\n"
                         "                         (numb)0.2, o.accept ? (numb)5 : (numb)1);\n";
                add(u);
            }
#ifndef UCUDA_AD_HAIRER_ONLY   // фильтры — параметры встроенного Filter, а он не собран
            if (ImGui::Selectable("empty filter")) {
                AdaptiveUserCtrl u;
                u.name = "my filter"; u.kind = kUserCtrlFilter;
                for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                add(u);
            }
#endif
            int np = 0;
            const AdaptiveCtrlPreset* pr = adaptive_ctrl_presets(&np);
            ImGui::SeparatorText("C body presets");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlBody) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#ifndef UCUDA_AD_HAIRER_ONLY
            ImGui::SeparatorText("Soderlind filters");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlFilter) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#endif
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(m.ctrl_lib_sel < 0);
        if (ImGui::Button("Copy") && m.ctrl_lib_sel >= 0) add(items[(size_t)m.ctrl_lib_sel]);
        ImGui::SameLine();
        if (ImGui::Button("Delete") && m.ctrl_lib_sel >= 0) {
            items.erase(items.begin() + m.ctrl_lib_sel);
            if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        for (int i = 0; i < (int)items.size(); ++i) {
            ImGui::PushID(i);
            const AdaptiveUserCtrl& u = items[(size_t)i];
            const std::string label = u.name + (u.kind == kUserCtrlFilter ? "  (filter)" : "");
            if (ImGui::Selectable(label.c_str(), m.ctrl_lib_sel == i)) m.ctrl_lib_sel = i;
            if (!u.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", u.tip.c_str());
            ImGui::PopID();
        }
        if (items.empty()) ImGui::TextDisabled("Empty. New... adds a controller\nfrom scratch or from a preset.");
        ImGui::EndChild();

        // Справа — выбранная запись.
        ImGui::SameLine();
        ImGui::BeginChild("##ctrl_edit", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        if (m.ctrl_lib_sel >= 0) {
            AdaptiveUserCtrl& u = items[(size_t)m.ctrl_lib_sel];
            const ImVec4 warn(1.0f, 0.55f, 0.3f, 1.0f);
            edited |= InputTextStr("name", u.name, 260.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Tabs refer to a controller by its name: after a rename they\n"
                                  "report the old name as unknown until you pick it again.");
            if (u.name.empty()) ImGui::TextColored(warn, "The name is empty.");
            else if (taken(u.name, m.ctrl_lib_sel))
                ImGui::TextColored(warn, "The name is taken (a built-in controller or another entry).");
            int kind = u.kind;
#ifdef UCUDA_AD_HAIRER_ONLY
            // Фильтр — параметры встроенного Filter, который в этой сборке не собран: такую
            // запись (из старой библиотеки) можно только перевести в C body или удалить.
            if (u.kind == kUserCtrlFilter) {
                ImGui::TextColored(warn, "Soderlind filters are not built in this version: the tabs do not offer it.");
                if (ImGui::SmallButton("make it a C body")) kind = kUserCtrlBody;
            }
#else
            ImGui::RadioButton("C body", &kind, kUserCtrlBody);
            ImGui::SameLine();
            ImGui::RadioButton("Soderlind filter", &kind, kUserCtrlFilter);
#endif
            if (kind != u.kind) {
                u.kind = kind;
                if (kind == kUserCtrlFilter) {
                    u.par_values.clear();
                    for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                } else if (u.par_names.size() != u.par_values.size()) {
                    u.par_values.resize(u.par_names.size());
                }
                edited = true;
            }
            edited |= InputTextStr("description", u.tip, 520.0f);

            if (u.kind == kUserCtrlFilter) {
#ifndef UCUDA_AD_HAIRER_ONLY   // иначе встроенного Filter нет (adaptive_find_builtin -> nullptr)
                const AdaptiveCtrlInfo* fi = adaptive_find_builtin("Filter");
                ImGui::TextDisabled("%s", fi->tip);
                if ((int)u.par_values.size() != fi->npar) {
                    const std::vector<double> d = adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7);
                    u.par_values.resize((size_t)fi->npar);
                    for (int i = 0; i < fi->npar; ++i)
                        if (u.par_values[(size_t)i].empty()) u.par_values[(size_t)i] = fmt_num_shortest(d[(size_t)i]);
                    edited = true;
                }
                for (int i = 0; i < fi->npar; ++i) {
                    ImGui::PushID(i);
                    edited |= InputTextStr(fi->par[i], u.par_values[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
#endif
            } else {
                bool hr = u.hairer_rules != 0;
                if (ImGui::Checkbox("Hairer's initial step and end-point rule", &hr)) { u.hairer_rules = hr; edited = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("On: initial step as hinit of dopri5 / dop853 and the last step\n"
                                      "stretched when x + 1.01 h passes the end (as the built-in Hairer).\n"
                                      "Off: scipy's initial step, the step is cut exactly at the end.");
                ImGui::Text("Parameters c[k] (name, default):");
                if (u.par_values.size() != u.par_names.size()) u.par_values.resize(u.par_names.size());
                int remove = -1;
                for (int i = 0; i < (int)u.par_names.size(); ++i) {
                    ImGui::PushID(i);
                    ImGui::Text("c[%d]", i);
                    ImGui::SameLine(60.0f);
                    edited |= InputTextStr("##pn", u.par_names[(size_t)i], 140.0f);
                    ImGui::SameLine();
                    edited |= InputTextStr("##pv", u.par_values[(size_t)i], kFieldW);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x")) remove = i;
                    ImGui::PopID();
                }
                if (remove >= 0) {
                    u.par_names.erase(u.par_names.begin() + remove);
                    u.par_values.erase(u.par_values.begin() + remove);
                    edited = true;
                }
                ImGui::BeginDisabled(u.par_names.size() >= (size_t)UCUDA_CTL_NPAR);
                if (ImGui::SmallButton("+ parameter")) {
                    u.par_names.push_back("p" + std::to_string(u.par_names.size()));
                    u.par_values.push_back("0");
                    edited = true;
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled("(up to %d; each is also a sweep axis)", UCUDA_CTL_NPAR);

                ImGui::Text("Prepare:");
                ImGui::SameLine();
                ImGui::TextDisabled("once per trajectory: k[0..7] from c[] and q, read by the body as in.k[]");
                edited |= InputTextMultilineStr("##prep", u.prep,
                                                ImVec2(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 6.5f));
                // Проверка и её результат — по паре (подготовка, тело), как их собирает модуль.
                const std::string packed = adaptive_ctrl_pack(u.prep, u.body);
                const bool checked = !m.ctrl_lib_check_body.empty() && m.ctrl_lib_check_body == packed;
                ImGui::Text("Body:");
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCtrlBodyHelp);
                ImGui::SameLine();
                const bool checking = m.ctrl_lib_check_future.valid();
                ImGui::BeginDisabled(checking);
                if (ImGui::SmallButton("Check")) { do_check = true; check_body = packed; }
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Compile the prepare section and the body with NVRTC (no run). The CPU\n"
                                      "build with cl.exe happens on the first CPU run of Analysis.");
                if (checked) {
                    ImGui::SameLine();
                    if (checking)                 ImGui::TextDisabled("checking...");
                    else if (m.ctrl_lib_check_ok) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "compiles");
                    else                          ImGui::TextColored(warn, "does not compile:");
                }
                edited |= InputTextMultilineStr("##body", u.body, ImVec2(-1.0f, m.ctrl_lib_body_h));
                draw_resize_handle("##body_resize", m.ctrl_lib_body_h);
                if (checked && !m.ctrl_lib_check_log.empty())
                    ImGui::InputTextMultiline("##check_log", const_cast<char*>(m.ctrl_lib_check_log.c_str()),
                                              m.ctrl_lib_check_log.size() + 1, ImVec2(-1.0f, 110.0f),
                                              ImGuiInputTextFlags_ReadOnly);
                if (ImGui::TreeNode("API")) {
                    ImGui::TextUnformatted(kCtrlBodyHelp);
                    ImGui::TreePop();
                }
            }
        } else {
            ImGui::TextDisabled("Select a controller on the left, or add one with New...");
            ImGui::TextDisabled("Controllers of this library appear in the \"controller\" list of every tab.");
        }
        ImGui::EndChild();
    }
    if (do_check && !m.ctrl_lib_check_future.valid()) {
        // NVRTC в фоне: окно не замирает на время сборки. Результат забирается в начале кадра.
        m.ctrl_lib_check_body = check_body;
        m.ctrl_lib_check_ok   = false;
        m.ctrl_lib_check_log.clear();
        m.ctrl_lib_check_future = std::async(std::launch::async, [check_body]() {
            std::string log;
            const bool ok = check_controller_body(check_body, log);
            return std::make_pair(ok, log);
        });
    }
    if (edited) m.ctrl_lib_dirty = true;
    if (m.ctrl_lib_dirty && !ImGui::IsAnyItemActive()) {
        std::string err;
        m.ctrl_lib_status = save_ctrl_library(&err) ? std::string() : err;
        m.ctrl_lib_dirty = false;
    }
    if (!m.ctrl_lib_status.empty())
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", m.ctrl_lib_status.c_str());
    ImGui::End();
}
}


namespace ucuda::ui_detail {
bool draw_adaptive_block(const char* id, AdaptiveSettings& a, const std::string& scheme,
                                int opts, bool is_map, AppModel* bc) {
    if (is_map) return false;   // у отображения шага нет
    ImGui::PushID(id);
    bool changed = false;
    const bool ok = adaptive_scheme_name_ok(scheme);

    int mode = a.enabled ? 1 : 0;
    changed |= ImGui::RadioButton("Fixed step", &mode, 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(!ok && !a.enabled);
    changed |= ImGui::RadioButton("Adaptive step", &mode, 1);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Step size chosen by an error controller from the embedded\n"
                          "error estimate of the scheme. Needs RK45, DOPRI78, DOP853,\n"
                          "GBS 2-4 ... or an extrapolation Extr(...) with at least two stages:\n"
                          "its first K-1 stages give the lower-order solution, and the\n"
                          "dense output between step nodes is cubic Hermite.");
    a.enabled = (mode == 1);

    if (a.enabled) {
        if (!ok)
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                               "The scheme has no error estimate: choose RK45, DOPRI78, DOP853, GBS 2-4 ...\n"
                               "or Extr(...) with at least two stages.");
        if (opts & kAdUiRaw) {
            int out = a.raw_nodes ? 1 : 0;
            changed |= ImGui::RadioButton("uniform grid", &out, 0);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output on a uniform time grid through the dense output of\n"
                                  "the scheme; the field h is the output step.");
            ImGui::SameLine();
            changed |= ImGui::RadioButton("step nodes", &out, 1);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output exactly at the accepted steps (including the steps\n"
                                  "shortened at the end of the transient and of the interval);\n"
                                  "the decimator keeps every N-th node, h is not used.");
            a.raw_nodes = (out == 1);
        }
        changed |= InputNumStr("rtol", a.rtol, kFieldW);
        changed |= InputNumListStr("atol", a.atol, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("One value for all variables, or one per variable separated by ';'.");
        changed |= InputNumStr("h0", a.h0, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Initial step. Empty: chosen automatically.");
        changed |= InputNumStr("h_min", a.hmin, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Smallest step. A step that would go below it is taken at h_min\n"
                              "anyway and counted as forced. Empty: the larger of 10 ulp(t) and\n"
                              "1e-12 x the length of the whole interval (transient + record).\n"
                              "A set value is raised to 10 ulp(t) where it is smaller: below that\n"
                              "t + h_min == t and the step would not advance time.");
        changed |= InputNumStr("h_max", a.hmax, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Largest step. Empty: no limit.");
        changed |= InputNumStr("max rejects", a.max_rej, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Rejections allowed per step. After this many in a row the next\n"
                              "attempt is made at h_min and taken unconditionally. Empty: no limit.\n"
                              "0: no retries - every attempt is taken, a rejection only shrinks the\n"
                              "next step (counted as forced); the tolerance becomes a target, not a bound.");

        int nb = 0;
        const AdaptiveCtrlInfo* bi = adaptive_builtin_ctrls(&nb);
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo("controller", a.ctrl.c_str())) {
            auto pick = [&](const std::string& name) {
                if (a.ctrl != name) { a.ctrl = name; a.ctrl_params.clear(); changed = true; }
            };
            for (int i = 0; i < nb; ++i) {
                if (ImGui::Selectable(bi[i].name, a.ctrl == bi[i].name)) pick(bi[i].name);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", bi[i].tip);
            }
            const std::vector<std::pair<std::string, std::string>> users = user_ctrl_list();
            if (!users.empty()) ImGui::SeparatorText("library");
            for (size_t i = 0; i < users.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::Selectable(users[i].first.c_str(), a.ctrl == users[i].first)) pick(users[i].first);
                if (!users[i].second.empty() && ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", users[i].second.c_str());
                ImGui::PopID();
            }
            ImGui::Separator();
            // Окно библиотеки — флаг в AppModel; без модели (bc == nullptr) пункта нет.
            if (bc != nullptr && ImGui::Selectable("edit library...")) bc->show_ctrl_library = true;
            ImGui::EndCombo();
        }
        AdaptiveCtrlResolved cur;
        std::string cur_err;
        if (!adaptive_resolve_ctrl(a.ctrl, adaptive_scheme_q(scheme), cur, cur_err)) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s", cur_err.c_str());
        } else {
            const int npar = (int)cur.par.size();
            const std::vector<double>& def = cur.def;
            if ((int)a.ctrl_params.size() != npar) {
                std::string line;
                for (int i = 0; i < npar; ++i) {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "%s%s = %g", i ? ", " : "", cur.par[(size_t)i].c_str(),
                                  i < (int)def.size() ? def[(size_t)i] : 0.0);
                    line += buf;
                }
                if (npar > 0) {
                    ImGui::TextDisabled("%s", line.c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("edit")) {
                        a.ctrl_params.clear();
                        for (int i = 0; i < npar; ++i) a.ctrl_params.push_back(fmt_num_shortest(def[(size_t)i]));
                        changed = true;
                    }
                }
            } else {
                for (int i = 0; i < npar; ++i) {
                    ImGui::PushID(i);
                    changed |= InputNumStr(cur.par[(size_t)i].c_str(), a.ctrl_params[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
                if (ImGui::SmallButton("defaults")) { a.ctrl_params.clear(); changed = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(cur.user ? "Back to the defaults stored in the controller library."
                                               : "Back to the defaults for this scheme.");
            }
        }
        if ((opts & kAdUiInterp) && a.raw_nodes) {
            const char* items[] = { "node", "parabola", "Hermite by f" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("peak interpolation", &a.peak_interp, items, IM_ARRAYSIZE(items));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How a peak (in Metrics also a min / max) is placed between step nodes:\n"
                                  "node - the largest node itself;\n"
                                  "parabola - vertex of the parabola through three nodes\n"
                                  "  (the fixed-step formula generalised to unequal steps);\n"
                                  "Hermite by f - cubic on the step where dx/dt changes sign,\n"
                                  "  built from the nodes and the exact derivatives.");
        }
        if ((opts & kAdUiMaxPts) && a.raw_nodes)
            changed |= InputNumStr("max points", a.max_points, kFieldW);
        if (opts & kAdUiRenorm) {
            const char* items[] = { "exactly at k*NT", "first node after NT" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("renormalization", &a.lyap_renorm, items, IM_ARRAYSIZE(items));
        }
    }
    if (opts & kAdUiMinMax) {
        changed |= ImGui::Checkbox("min/max by interpolated extrema in Fixed", &a.minmax_interp_fixed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("With the adaptive step min/max always use interpolated extrema: on the\n"
                              "uniform grid the vertex of the parabola through three samples (as peaks\n"
                              "do), on step nodes the peak interpolation. This applies the parabola to\n"
                              "the fixed step too.");
    }
    if (bc && a.enabled) {
        if (ImGui::SmallButton("apply to all tabs")) broadcast_adaptive(*bc, a);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Copy these step settings to every tab that has them\n"
                              "(Analysis, Custom, Bifurcation, LLE, LS, Basins, Metrics). Ctrl+Z undoes.");
    }
    ImGui::PopID();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_newton_fields(NewtonSettings& n, float w, const char* id) {
    bool changed = false;
    ImGui::PushID(id);
    int jac_mode = n.full ? 1 : 0;
    ImGui::SetNextItemWidth(w * 2.0f);
    if (ImGui::Combo("Newton jacobian", &jac_mode, "modified (once per step)\0full (every iteration)\0")) {
        n.full = (jac_mode == 1);
        changed = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("modified: Jacobian and LU reused across iterations and\n"
                          "refreshed when the correction stops shrinking.\n"
                          "Cheaper on GPU and the right default.\n"
                          "full: rebuilt every iteration. Quadratic convergence,\n"
                          "most robust at large h, ~k times the cost.\n"
                          "Passed at run time as a[M+4]: no recompilation.");
    changed |= InputNumStr("Newton tolerance", n.tol, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Newton stops once ||dX|| falls below this (a[M+2], run time).\n"
                          "DAE methods also stop at ||dX|| < tol * |X| (relative).\n"
                          "With numb = float it also stops at the precision floor\n"
                          "||dX|| < 16 eps |X|: below that float cannot go.");
    changed |= InputNumStr("Newton max iterations", n.max_iters, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Hard cap on Newton passes per step (a[M+3], run time).\n"
                          "2-3 is typical from the explicit-Euler predictor.\n"
                          "A step that hits the cap is accepted unconverged: on a DAE\n"
                          "the constraint then drifts. DAE: full Newton, ~25 iterations.");
    ImGui::PopID();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f,
                                   AppModel* bc,
                                   bool is_map) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return false;
    bool changed = false;
    auto menu = [&](BroadcastField kind, const char* label, const std::string& v) {
        return [bc, kind, label, &v] { field_apply_all_menu(bc, kind, {}, label, v); };
    };
    // h is pinned to 1 for a map (see default_h_from_record), so the field is
    // gone and the engine's "time" is exactly the iteration count.
    const char* t_max_label     = is_map ? "iterations"           : "computing time";
    const char* transient_label = is_map ? "transient iterations" : "transient time";
    if (f.h && !is_map) changed |= InputNumStr("h", *f.h, kFieldW,
                                    menu(BroadcastField::StepH, "h", *f.h));
    if (f.h && !is_map && f.adaptive && f.adaptive->enabled) {
        ImGui::SameLine();
        // LLE/LS (перенормировка в узлах шага) h не используют вовсе.
        ImGui::TextDisabled((f.adaptive_opts & kAdUiRenorm) ? "(not used)"
                            : f.adaptive->raw_nodes ? "(not used on step nodes)" : "(output step)");
    }
    if (f.symmetry_s && scheme_uses_symmetry(scheme, custom_schemes))
        changed |= InputNumStr("symmetry s", *f.symmetry_s, kFieldW,
                               menu(BroadcastField::SymmetryS, "symmetry s", *f.symmetry_s));
    if (f.cd_fp_iters && scheme_uses_cd_fp_iters(scheme)) {
        changed |= InputNumStr("FP iterations n", *f.cd_fp_iters, kFieldW,
                               menu(BroadcastField::CdFpIters, "FP iterations n", *f.cd_fp_iters));
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
    }
    if (f.newton && scheme_uses_newton(scheme))
        changed |= draw_newton_fields(*f.newton, kFieldW);
    if (f.t_max)      changed |= InputNumStr(t_max_label, *f.t_max,       kFieldW,
                               menu(BroadcastField::TMax, t_max_label, *f.t_max));
    if (f.transient)   changed |= InputNumStr(transient_label, *f.transient,   kFieldW,
                               menu(BroadcastField::Transient, transient_label, *f.transient));
    if (f.pre_scaller) changed |= InputNumStr("decimator",      *f.pre_scaller, kFieldW,
                               menu(BroadcastField::PreScaller, "decimator", *f.pre_scaller));
    if (f.max_value)   changed |= InputNumStr("max value",      *f.max_value,   kFieldW,
                               menu(BroadcastField::MaxValue, "max value", *f.max_value));
    if (f.adaptive) {
        ImGui::Separator();
        changed |= draw_adaptive_block(header, *f.adaptive, scheme, f.adaptive_opts, is_map, bc);
    }
    return changed;
}
}





std::vector<std::string> default_hidden_schemes() {
    static const char* const kShown[] = { "Euler", "Explicit Midpoint", "CD", "RK4", "DOPRI78", "GBS (n=2)" };
    std::vector<std::string> out;
    for (const auto& b : kBuiltinSchemes)
        if (std::find_if(std::begin(kShown), std::end(kShown),
                         [&](const char* s) { return std::strcmp(s, b.name) == 0; }) == std::end(kShown))
            out.push_back(b.name);
    return out;
}


std::vector<std::string> dae_method_list() {
    std::vector<std::string> v = dae_scheme_names();
    v.push_back(kBdfSchemeDae);
    return v;
}


std::vector<std::string> default_hidden_dae_methods() {
    std::vector<std::string> out;
    for (const std::string& n : dae_method_list())
        if (n != "DAE Trapezoidal" && n != kBdfSchemeDae) out.push_back(n);
    return out;
}
