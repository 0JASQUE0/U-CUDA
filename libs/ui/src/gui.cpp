#include "gui_internal.h"
using namespace ucuda::ui_detail;
void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    // полноэкранный dockspace-хост
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("MainHost", nullptr, host_flags);
    ImGui::PopStyleVar(2);

    refresh_auto_labels(model);
    prune_projection_views();
    // переключатель режимов
    int mode = (int)model.app_mode;
    ImGui::RadioButton("Library", &mode, (int)AppModel::AppMode::Library); ImGui::SameLine();
    ImGui::RadioButton("Phase analysis", &mode, (int)AppModel::AppMode::Analysis); ImGui::SameLine();
    ImGui::RadioButton("Parametric", &mode, (int)AppModel::AppMode::Parametric); ImGui::SameLine();
    ImGui::RadioButton("1D DFT", &mode, (int)AppModel::AppMode::Dft1D); ImGui::SameLine();
    ImGui::RadioButton("Basins", &mode, (int)AppModel::AppMode::Basins); ImGui::SameLine();
    ImGui::RadioButton("Fast Synchro", &mode, (int)AppModel::AppMode::FastSync); ImGui::SameLine();
    ImGui::RadioButton("Custom", &mode, (int)AppModel::AppMode::Custom); ImGui::SameLine();
    ImGui::RadioButton("Order", &mode, (int)AppModel::AppMode::Order); ImGui::SameLine();
    ImGui::RadioButton("Settings", &mode, (int)AppModel::AppMode::Settings);

    // Правый край ряда вкладок, в координатах окна. Нужен системному комбо
    // ниже: оно центрируется, но не должно наезжать на вкладки. Снимаем
    // ИМЕННО ЗДЕСЬ, а не в самом комбо, потому что между ними рисуется
    // индикатор занятости — тот уводит курсор к правому краю окна, и
    // измеренный там "конец вкладок" выталкивал бы комбо за край экрана
    // на всё время расчёта.
    const float mode_tabs_end_x = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x
                                + ImGui::GetScrollX();

    // Битый файл сессии. Висит до следующей УСПЕШНОЙ загрузки, а не до конца
    // кадра: иначе сообщение о том, что настройки не восстановились, мелькнуло
    // бы один раз при старте и пропало. Раньше эта ошибка не показывалась
    // вообще — результат разбора игнорировался во всех точках вызова.
    if (!model.session_load_warning.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s",
                           model.session_load_warning.c_str());
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The saved session file could not be parsed and was ignored.\n"
                              "Settings from the system record are in use instead.\n"
                              "Running an analysis and switching systems will overwrite the file.");
    }

    // Индикатор компьюта — справа по границе окна, виден во всех режимах. Layout:
    // [text] [progress bar] [Stop] для отменяемых сессий, только [text] для phase и для
    // персистентного "Done/Cancelled". Stop заодно осушает parametric_queue и basins_queue,
    // чтобы остаток пачки не стартовал сам.
    BusyInfo busy;                    // подпись/старт/отмена/прогресс — см. busy_from
    bool   show_done    = false;      // not in flight — show persistent last-run info
    bool   last_ok      = true;       // for show_done: true = green "Done", false = red "Cancelled"
    double done_seconds = 0.0;
    int    basins_phase = 0;          // 1 = sim, 2 = cluster; 0 = not basins or not started

    {
        auto& bif = model.bifurcation_session;
        auto& lle = model.lle_session;
        auto& lsp = model.ls_session;
        auto& dft = model.dft1d_session;
        auto& bas = model.basins_session;
        auto& fsy = model.fastsync_session;
        auto& ord = model.order_session;
        auto& cus = model.custom_session;

    if (bif.in_flight)
        busy = busy_from(bif, BusyKind::Bif,
                         running_label(bif.diagrams, bif.running_diagram_index, "bifurcation"));
    else if (lle.in_flight)
        busy = busy_from(lle, BusyKind::LLE,
                         running_label(lle.curves, lle.running_curve_index, "LLE"));
    else if (lsp.in_flight)
        busy = busy_from(lsp, BusyKind::LS,
                         running_label(lsp.curves, lsp.running_curve_index, "LS"));
    else if (dft.in_flight)
        busy = busy_from(dft, BusyKind::Dft1D,
                         running_label(dft.configs, dft.running_config_index, "dft1d"));
    else if (model.phase_session.in_flight) {
        // У phase нет ни отмены, ни прогресса — только время старта.
        busy.kind  = BusyKind::Phase;
        busy.what  = "phase";
        busy.start = model.phase_session.compute_start_time;
    }
    else if (bas.in_flight) {
        busy = busy_from(bas, BusyKind::Basins,
                         running_label(bas.configs, bas.running_config_index, "basins"));
        // Своя, только у Basins: 1 = моделирование, 2 = кластеризация.
        if (bas.progress_phase_token)
            basins_phase = bas.progress_phase_token->load(std::memory_order_relaxed);
    }
    else if (fsy.in_flight)
        busy = busy_from(fsy, BusyKind::FastSync,
                         running_label(fsy.configs, fsy.running_config_index, "fastsync"));
    else if (ord.in_flight)
        busy = busy_from(ord, BusyKind::Order,
                         running_label(ord.configs, ord.running_config_index, "order"));
    else if (cus.any_in_flight()) {
        // Custom-tab: queue is drained serially, so at most one sub-session is
        // in-flight at any time — pick whichever one it is and surface its
        // label/progress under a "Custom: ..." prefix.
        if (cus.bif_session.in_flight)
            busy = busy_from(cus.bif_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.bif_session.diagrams,
                                                        cus.bif_session.running_diagram_index, "Bif"));
        else if (cus.lle_session.in_flight)
            busy = busy_from(cus.lle_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.lle_session.curves,
                                                        cus.lle_session.running_curve_index, "LLE"));
        else if (cus.ls_session.in_flight)
            busy = busy_from(cus.ls_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.ls_session.curves,
                                                        cus.ls_session.running_curve_index, "LS"));
        else if (cus.phase_session.in_flight) {
            busy.what  = "Custom: phase";
            busy.start = cus.phase_session.compute_start_time;
        }
        else if (cus.basins_session.in_flight)
            busy = busy_from(cus.basins_session, BusyKind::Custom,
                             "Custom: " + running_label(cus.basins_session.configs,
                                                        cus.basins_session.running_config_index, "basins"));
        busy.kind = BusyKind::Custom;   // в том числе для phase-ветки выше
    }
    else {
        // Nothing in flight — pick the session whose last run finished most
        // recently (across the cancellable ones) and show persistent info.
        // Под-сессии Custom-вкладки живут отдельно от одноимённых сессий
        // Parametric/Basins и тоже обязаны сюда попадать: без них после счёта
        // в Custom плашка показывала время ПРОШЛОГО запуска из Parametric.
        // Метки у них свои ("Custom 2D", "Custom 1D X", ...), так что префикс
        // не нужен.
        struct DoneCand { BusyKind kind; std::chrono::steady_clock::time_point ts; const std::string* label; bool ok; double secs; };
        DoneCand candidates[10] = {
            { BusyKind::Bif,    model.bifurcation_session.last_run_completed_at,
              &model.bifurcation_session.last_run_label,
              model.bifurcation_session.last_run_succeeded,
              model.bifurcation_session.last_run_seconds },
            { BusyKind::LLE,    model.lle_session.last_run_completed_at,
              &model.lle_session.last_run_label,
              model.lle_session.last_run_succeeded,
              model.lle_session.last_run_seconds },
            { BusyKind::LS,     model.ls_session.last_run_completed_at,
              &model.ls_session.last_run_label,
              model.ls_session.last_run_succeeded,
              model.ls_session.last_run_seconds },
            { BusyKind::Dft1D,  model.dft1d_session.last_run_completed_at,
              &model.dft1d_session.last_run_label,
              model.dft1d_session.last_run_succeeded,
              model.dft1d_session.last_run_seconds },
            { BusyKind::Basins, model.basins_session.last_run_completed_at,
              &model.basins_session.last_run_label,
              model.basins_session.last_run_succeeded,
              model.basins_session.last_run_seconds },
            { BusyKind::Order,  model.order_session.last_run_completed_at,
              &model.order_session.last_run_label,
              model.order_session.last_run_succeeded,
              model.order_session.last_run_seconds },
            { BusyKind::Custom, cus.bif_session.last_run_completed_at,
              &cus.bif_session.last_run_label,
              cus.bif_session.last_run_succeeded,
              cus.bif_session.last_run_seconds },
            { BusyKind::Custom, cus.lle_session.last_run_completed_at,
              &cus.lle_session.last_run_label,
              cus.lle_session.last_run_succeeded,
              cus.lle_session.last_run_seconds },
            { BusyKind::Custom, cus.ls_session.last_run_completed_at,
              &cus.ls_session.last_run_label,
              cus.ls_session.last_run_succeeded,
              cus.ls_session.last_run_seconds },
            { BusyKind::Custom, cus.basins_session.last_run_completed_at,
              &cus.basins_session.last_run_label,
              cus.basins_session.last_run_succeeded,
              cus.basins_session.last_run_seconds },
        };
        const DoneCand* best = nullptr;
        for (const auto& c : candidates) {
            if (c.label->empty()) continue;
            if (!best || c.ts > best->ts) best = &c;
        }
        if (best) {
            busy.what    = *best->label;
            busy.kind    = best->kind;
            show_done    = true;
            last_ok      = best->ok;
            done_seconds = best->secs;
        }
    }
    }   // область видимости ссылок на сессии

    if (busy.kind != BusyKind::None) {
        char text[200];
        // Phase suffix appears only for basins while running (not on "Cancelling"
        // or "Done" — those reflect overall state, not the current sub-phase).
        const char* phase_suffix = "";
        if (busy.kind == BusyKind::Basins && !show_done && !busy.cancelling) {
            if      (basins_phase == 1) phase_suffix = " (1/2 sim)";
            else if (basins_phase == 2) phase_suffix = " (2/2 cluster)";
        }
        if (show_done) {
            std::snprintf(text, sizeof(text), "%s %s in %.1fs",
                          last_ok ? "Done" : "Cancelled",
                          busy.what.c_str(), done_seconds);
        } else if (busy.cancelling) {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy.start).count();
            std::snprintf(text, sizeof(text), "Cancelling %s... %.1fs", busy.what.c_str(), secs);
        } else {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy.start).count();
            // Queue suffix: prefer the queue that matches the running session
            // (parametric for Bif/LLE/LS, basins for Basins, fastsync for FastSync).
            size_t queue_n = 0;
            if      (busy.kind == BusyKind::Basins)   queue_n = model.basins_queue.size();
            else if (busy.kind == BusyKind::FastSync) queue_n = model.fastsync_queue.size();
            else if (busy.kind == BusyKind::Dft1D)    queue_n = model.dft1d_queue.size();
            else if (busy.kind == BusyKind::Custom)   queue_n = model.custom_queue.size();
            else                                      queue_n = model.parametric_queue.size();
            if (queue_n > 0)
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs (+%zu)",
                              busy.what.c_str(), phase_suffix, secs, queue_n);
            else
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs",
                              busy.what.c_str(), phase_suffix, secs);
        }

        const bool show_stop = (busy.kind != BusyKind::Phase) &&
                               !show_done && !busy.cancelling;
        const bool show_bar  = show_stop;  // bar only when running & not cancelling

        const float pad      = ImGui::GetStyle().ItemSpacing.x;
        const float bar_w    = 120.0f;
        const float bar_h    = ImGui::GetTextLineHeight();
        const float text_w   = ImGui::CalcTextSize(text).x;
        const float stop_w   = show_stop
                               ? (ImGui::CalcTextSize("Stop").x +
                                  ImGui::GetStyle().FramePadding.x * 2.0f)
                               : 0.0f;
        float total_w = text_w + 12.0f;
        if (show_bar)  total_w += bar_w + pad;
        if (show_stop) total_w += stop_w + pad;

        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowSize().x - total_w);

        ImVec4 col;
        if (show_done) {
            col = last_ok ? ImVec4(0.55f, 0.95f, 0.55f, 1.0f)   // green
                          : ImVec4(0.95f, 0.45f, 0.45f, 1.0f);  // red
        } else {
            col = ImVec4(1.0f, 0.85f, 0.25f, 1.0f);              // yellow (running/cancelling)
        }
        ImGui::TextColored(col, "%s", text);

        if (show_bar) {
            ImGui::SameLine();
            const float f = std::clamp(busy.progress, 0.0f, 1.0f);
            ImGui::ProgressBar(f, ImVec2(bar_w, bar_h), "");
        }
        if (show_stop) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.55f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.90f, 0.25f, 0.25f, 1.0f));
            if (ImGui::SmallButton("Stop")) {
                switch (busy.kind) {
                    case BusyKind::Bif:      model.bifurcation_session.request_cancel(); break;
                    case BusyKind::LLE:      model.lle_session.request_cancel();         break;
                    case BusyKind::LS:       model.ls_session.request_cancel();          break;
                    case BusyKind::Dft1D:    model.dft1d_session.request_cancel();       break;
                    case BusyKind::Basins:   model.basins_session.request_cancel();      break;
                    case BusyKind::FastSync: model.fastsync_session.request_cancel();    break;
                    case BusyKind::Order:    model.order_session.request_cancel();       break;
                    case BusyKind::Custom:   model.custom_session.request_cancel_all();  break;
                    default: break;
                }
                // Drain all batch queues so remaining items don't auto-start.
                model.parametric_queue.clear();
                model.dft1d_queue.clear();
                model.basins_queue.clear();
                model.fastsync_queue.clear();
                model.custom_queue.clear();
            }
            ImGui::PopStyleColor(3);
        }
    }

    // Global system picker — centered on the top-bar. One combo replaces the
    // five per-tab System: combos that used to live in each draw_*_controls.
    // Disabled while ANY session is in-flight so a switch can't race a worker
    // that will apply its result to the already-swapped session.
    {
        bool any_in_flight =
              model.phase_session.in_flight
           || model.bifurcation_session.in_flight
           || model.lle_session.in_flight
           || model.ls_session.in_flight
           || model.dft1d_session.in_flight
           || model.basins_session.in_flight
           || model.fastsync_session.in_flight
           || model.order_session.in_flight
           || model.custom_session.any_in_flight();
        const float combo_w = 240.0f;
        std::string preview = model.name.empty() ? std::string("(select system)") : model.name;
        // Центрируем, но не левее конца ряда вкладок: с добавлением Order ряд
        // дорос до ширины, на которой жёсткий центр наезжал на "Settings".
        // Край ряда снят выше (mode_tabs_end_x) — здесь курсор уже сдвинут
        // индикатором занятости к правому краю окна и мерить по нему нельзя.
        const float after_tabs = mode_tabs_end_x + ImGui::GetStyle().ItemSpacing.x * 2.0f;
        float cx = (ImGui::GetWindowSize().x - combo_w) * 0.5f;
        if (cx < after_tabs) cx = after_tabs;
        ImGui::SameLine();
        ImGui::SetCursorPosX(cx);
        ImGui::SetNextItemWidth(combo_w);
        if (any_in_flight) ImGui::BeginDisabled();
        if (ImGui::BeginCombo("##topsyssel", preview.c_str())) {
            for (const auto& nm : lib.list()) {
                if (ImGui::Selectable(nm.c_str(), model.name == nm))
                    apply_system_switch(model, lib, nm);
            }
            ImGui::EndCombo();
        }
        if (any_in_flight) ImGui::EndDisabled();
    }
    // При входе в Analysis/Parametric решаем, нужно ли (пере)инициализировать сессию. Init идёт,
    // когда: 1) система сменилась относительно той, для которой собиралась session; 2) session ещё
    // не инициализирована (vars пустой) — несохранённые системы, где model.name =
    // loaded_system_name = ""; 3) у model сменился алфавит / vars_text и session.vars/params уже не
    // совпадают с model.known_vars/known_params. Случай 3 раньше требовал перезапуска приложения.
    bool entering_phase  = (AppModel::AppMode)mode == AppModel::AppMode::Analysis &&
                           model.app_mode != AppModel::AppMode::Analysis;
    bool entering_par    = (AppModel::AppMode)mode == AppModel::AppMode::Parametric &&
                           model.app_mode != AppModel::AppMode::Parametric;
    bool entering_dft1d  = (AppModel::AppMode)mode == AppModel::AppMode::Dft1D &&
                           model.app_mode != AppModel::AppMode::Dft1D;
    bool entering_basins = (AppModel::AppMode)mode == AppModel::AppMode::Basins &&
                           model.app_mode != AppModel::AppMode::Basins;
    bool entering_fastsync = (AppModel::AppMode)mode == AppModel::AppMode::FastSync &&
                             model.app_mode != AppModel::AppMode::FastSync;
    bool entering_custom = (AppModel::AppMode)mode == AppModel::AppMode::Custom &&
                           model.app_mode != AppModel::AppMode::Custom;
    bool entering_order  = (AppModel::AppMode)mode == AppModel::AppMode::Order &&
                           model.app_mode != AppModel::AppMode::Order;
    if (entering_phase || entering_par || entering_dft1d || entering_basins || entering_fastsync
        || entering_custom || entering_order) {
        // обновим known_vars/known_params из живого алфавита, чтобы сравнение
        // ниже было против актуального состояния
        model.refresh_symbols();
    }
    auto phase_need_init = model.phase_session.loaded_system_name != model.name
                        || model.phase_session.vars.empty()
                        || model.phase_session.vars   != model.known_vars
                        || model.phase_session.params != model.known_params;
    auto par_need_init   = model.bifurcation_session.loaded_system_name != model.name
                        || model.bifurcation_session.vars.empty()
                        || model.bifurcation_session.vars   != model.known_vars
                        || model.bifurcation_session.params != model.known_params;
    if (entering_phase && phase_need_init) {
        model.start_phase_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
        }
    }
    if (entering_par && par_need_init) {
        model.start_parametric_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, j, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
        }
    }
    auto dft1d_need_init = model.dft1d_session.loaded_system_name != model.name
                        || model.dft1d_session.vars.empty()
                        || model.dft1d_session.vars   != model.known_vars
                        || model.dft1d_session.params != model.known_params;
    if (entering_dft1d && dft1d_need_init) {
        model.start_dft1d_analysis();
        if (!model.loaded_name.empty()) {
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
        }
    }
    auto basins_need_init = model.basins_session.loaded_system_name != model.name
                         || model.basins_session.vars.empty()
                         || model.basins_session.vars   != model.known_vars
                         || model.basins_session.params != model.known_params;
    if (entering_basins && basins_need_init) {
        model.start_basins_analysis();
        if (!model.loaded_name.empty()) {
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
        }
    }
    // Order: своего _last_order.json пока нет, поэтому после init'а просто
    // остаются дефолты из записи библиотеки — ничего не восстанавливаем.
    auto order_need_init = model.order_session.loaded_system_name != model.name
                        || model.order_session.vars.empty()
                        || model.order_session.vars   != model.known_vars
                        || model.order_session.params != model.known_params;
    if (entering_order && order_need_init) model.start_order_analysis();
    auto fastsync_need_init = model.fastsync_session.loaded_system_name != model.name
                           || model.fastsync_session.vars.empty()
                           || model.fastsync_session.vars   != model.known_vars
                           || model.fastsync_session.params != model.known_params;
    if (entering_fastsync && fastsync_need_init) {
        model.start_fastsync_analysis();
        if (!model.loaded_name.empty()) {
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
        }
    }
    auto custom_need_init = model.custom_session.loaded_system_name != model.name
                         || model.custom_session.vars.empty()
                         || model.custom_session.vars   != model.known_vars
                         || model.custom_session.params != model.known_params;
    if (entering_custom && custom_need_init) {
        // Hard reset — mirrors the reset in apply_system_switch for the
        // Custom case. Prevents stale sub-session results / signature cache
        // from a previously-loaded system leaking into the fresh init.
        model.custom_session = CustomSession{};
        model.start_custom_analysis();
        if (!model.loaded_name.empty()) {
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
        }
    }
    // Persist AppMode change so the next launch restores this tab. Compare
    // BEFORE overwriting so we only write on real transitions (not on every
    // frame while sitting in the same tab).
    if ((AppModel::AppMode)mode != model.app_mode) {
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_app_mode = mode;
        save_app_config(get_exe_dir_with_sep(), cfg);
    }
    model.app_mode = (AppModel::AppMode)mode;
    ImGui::Separator();

    // Custom mode has its own two-pane layout (Controls | Workspace, split by
    // a draggable splitter) instead of the shared MainDockspace. Capture the
    // region below the AppMode radios so that layout can occupy exactly it.
    ImVec2 custom_area_pos{}, custom_area_size{};
    if (model.app_mode == AppModel::AppMode::Custom) {
        custom_area_pos  = ImGui::GetCursorScreenPos();
        custom_area_size = ImGui::GetContentRegionAvail();
    } else {
        // dockspace для содержимого — используется всеми режимами, кроме Custom.
        ImGuiID dockspace_id = ImGui::GetID("MainDockspace");
        ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_None);
    }

    ImGui::End(); // MainHost

    if (model.app_mode == AppModel::AppMode::Library) {
        // Library mode: list view by default, editor view (System/Parameters sub-tabs +
        // Save/Cancel) after Edit or Add new. Switching top-level tabs during edit is safe — the
        // editor works on model.library_edit_buffer, a scratch AppModel, so it never touches the
        // system active in Parametric/Phase/Basins/FastSync.
        if (ImGui::Begin("Editor")) {
            if (model.library_edit_mode == AppModel::LibraryEditMode::None)
                draw_library_list(model, lib);
            else
                draw_library_editor(model, lib, cb);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Analysis) {
        // режим анализа: панель настроек + окна проекций (докаются пользователем)
        if (ImGui::Begin("Controls")) {
            draw_phase_controls(model.phase_session, [&model, &lib]() {
                if (!model.loaded_name.empty()) {
                    model.from_record(lib.load(model.loaded_name));   // reference from disk
                    model.start_phase_analysis();
                }
            });
        }
        ImGui::End();
        draw_projection_windows(model.phase_session, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Parametric) {
        // Parametric mode: controls window (Bif/LLE/LS config tabs + "Plot
        // windows" management, unchanged in shape) + a dynamic list of plot
        // windows the user opens/closes/docks freely, like Phase's
        // projection windows.
        if (ImGui::Begin("Parametric Controls")) {
            draw_parametric_controls(model, lib);
        }
        ImGui::End();
        draw_parametric_plot_windows(model, lib, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Dft1D) {
        // DFT1D mode: controls window (system picker + Run/Run all + config
        // tab bar + Plot windows manager) + a dynamic list of heatmap plot
        // windows, mirrors Parametric mode's shape.
        if (ImGui::Begin("DFT1D Controls")) {
            draw_dft1d_controls(model, lib);
        }
        ImGui::End();
        draw_dft1d_plot_windows(model, lib, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Basins) {
        if (ImGui::Begin("Basins Controls")) {
            draw_basins_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Basins of Attraction")) {
            draw_basins_plot(model, lib, cb);
        }
        ImGui::End();
        // Окна фазовых портретов активного config'а (2D / time domain).
        draw_basins_phase_windows(model, cb);
    }
    else if (model.app_mode == AppModel::AppMode::FastSync) {
        if (ImGui::Begin("FastSync Controls")) {
            draw_fastsync_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Fast Synchro")) {
            draw_fastsync_plot(model, cb);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Order) {
        // Order mode: панель настроек + одно окно графика (кривая p или
        // 2D-карта — по тому, включён ли в конфиге 2D map).
        if (ImGui::Begin("Order Controls")) {
            draw_order_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Accuracy order")) {
            draw_order_plot(model, cb);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Custom) {
        // Custom mode: split-region layout (Controls | Workspace) drawn into
        // the area captured above. Plot windows live in per-tab dockspaces
        // inside the Workspace panel and are moved between tabs via native
        // ImGui drag&drop (see draw_custom_mode_layout).
        draw_custom_mode_layout(model, lib, cb, custom_area_pos, custom_area_size);
    }
    else { // AppMode::Settings
        // Settings состоит из независимых виджетов, а save_app_config пишет файл целиком —
        // поэтому read-modify-write, как в draw_top_bar. Раньше каждый обработчик собирал AppConfig
        // с нуля и терял last_app_mode / last_system_name, а слайдер UI scale и чекбокс шрифта —
        // ещё и dark_theme (тема сбрасывалась в Dark при правке масштаба).
        auto persist_settings = [](const AppModel& m) {
            AppConfig cfg;
            load_app_config(get_exe_dir_with_sep(), cfg);
            cfg.ui_scale_override      = m.ui_scale_override;
            cfg.use_builtin_font       = m.use_builtin_font;
            cfg.heatmap_colormap       = m.heatmap_colormap;
            cfg.basins_colormap        = m.basins_colormap;
            cfg.basins_avgpk_colormap  = m.basins_avgpk_colormap;
            cfg.basins_avgint_colormap = m.basins_avgint_colormap;
            cfg.basins_states_colormap = m.basins_states_colormap;
            cfg.slancm_enabled         = m.slancm_enabled;
            cfg.tick_precision         = m.tick_precision;
            cfg.dark_theme             = m.dark_theme;
            cfg.peak                   = m.peak;
            cfg.nvrtc_fmad             = m.nvrtc_fmad;
            save_app_config(get_exe_dir_with_sep(), cfg);
        };

        if (ImGui::Begin("Settings")) {
            ImGui::Text("Interface scale");
            ImGui::TextDisabled("Auto-detected at startup from glfwGetMonitorContentScale.");
            ImGui::TextDisabled("Override persists in _app_config.json next to exe.");
            ImGui::Separator();

            ImGui::TextUnformatted("UI scale:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            // Применяем НЕ во время drag'а, а на отпускание (IsItemDeactivatedAfterEdit) —
            // иначе UI пересобирается на каждом кадре, виджет уходит из-под курсора.
            static float ui_slider_value = -1.0f;
            if (ui_slider_value < 0.0f) ui_slider_value = model.effective_ui_scale();
            if (!ImGui::IsAnyItemActive())
                ui_slider_value = model.effective_ui_scale();
            ImGui::SliderFloat("##ui_scale", &ui_slider_value, 0.5f, 3.0f, "%.2fx");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                model.ui_scale_override = ui_slider_value;
                persist_settings(model);
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto")) {
                model.ui_scale_override = 0.0f;
                ui_slider_value = model.ui_scale_auto;
                persist_settings(model);
            }
            ImGui::TextDisabled("Auto detected: %.2fx   |   Override: %s",
                model.ui_scale_auto,
                model.ui_scale_override > 0 ?
                    (std::to_string(model.ui_scale_override) + "x").c_str() :
                    "(off)");

            ImGui::Separator();
            ImGui::Text("Font");
            bool use_builtin = model.use_builtin_font;
            if (ImGui::Checkbox("Use built-in font (ProggyClean)", &use_builtin)) {
                model.use_builtin_font = use_builtin;
                persist_settings(model);
            }
            ImGui::TextDisabled("Off: Windows Segoe UI TTF (recommended, crisp at any scale).");
            ImGui::TextDisabled("On: built-in bitmap ProggyClean (compact, pixel-perfect at 1x/2x/3x).");

            ImGui::Separator();
            ImGui::Text("Axes");
            int tp = model.tick_precision;
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("Tick precision (digits)", &tp, 2, 10)) {
                model.tick_precision = tp;
                set_tick_precision(tp);
                persist_settings(model);
            }
            ImGui::TextDisabled("Significant digits in axis tick and colorbar labels.");

            ImGui::Separator();
            ImGui::Text("Theme");
            // Радио по Dark/Light. apply_ui_scale в app_main.cpp ловит изменение
            // через applied_dark_theme и пересобирает style + scale.
            int theme_idx = model.dark_theme ? 0 : 1;
            bool theme_changed = false;
            if (ImGui::RadioButton("Dark", &theme_idx, 0)) theme_changed = true;
            ImGui::SameLine();
            if (ImGui::RadioButton("Light", &theme_idx, 1)) theme_changed = true;
            if (theme_changed) {
                model.dark_theme = (theme_idx == 0);
                persist_settings(model);
            }
            ImGui::TextDisabled("Color palette for ImGui controls. Plots use their own colormap.");

            // Colormaps: какие из 200 карт slanCM показывать в пикере.
            // Держать в combo все 200 неудобно, поэтому набор набирается
            // галочками здесь и живёт в _app_config.json (одна маска на
            // приложение, как и остальные настройки этой вкладки).
            ImGui::Separator();
            ImGui::Text("Colormaps");
            ImGui::TextDisabled("200 colormaps from slanCM (MATLAB File Exchange #120088),");
            ImGui::TextDisabled("numbered as in that library. Ticked ones show up in the");
            ImGui::TextDisabled("Colormap picker above every heatmap. Viridis / Inferno /");
            ImGui::TextDisabled("Turbo / Gray are built in and always available.");

            // Битая или отсутствующая маска (конфиг от старой версии, ручная
            // правка JSON) — молча чинится дефолтом, а не роняет вкладку.
            if (model.slancm_enabled.size() != (size_t)kSlanCmCount) {
                model.slancm_enabled = default_enabled_slancm();
                set_enabled_slancm(model.slancm_enabled);
            }
            auto apply_cmap_mask = [&]() {
                set_enabled_slancm(model.slancm_enabled);
                persist_settings(model);
            };

            InputTextStr("Filter##cmap_filter", model.colormap_filter, 220.0f);
            ImGui::SameLine();
            ImGui::TextDisabled("name, number or category");

            // Фильтрованный список номеров. 200 сравнений на кадр дешевле, чем
            // кэш с инвалидацией по каждому нажатию в поле фильтра.
            std::vector<int> shown;
            shown.reserve(kSlanCmCount);
            {
                auto lower = [](std::string s) {
                    // Имена и категории slanCM — чистый ASCII, поэтому обходимся
                    // без <cctype> и без вопросов к локали.
                    for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
                    return s;
                };
                const std::string q = lower(model.colormap_filter);
                for (int n = 1; n <= kSlanCmCount; ++n) {
                    if (q.empty()) { shown.push_back(n); continue; }
                    const std::string hay = lower(std::string(slancm_name(n)) + ' '
                        + slancm_category_name(slancm_category(n)) + ' ' + std::to_string(n));
                    if (hay.find(q) != std::string::npos) shown.push_back(n);
                }
            }

            // Кнопки работают по ВИДИМОМУ списку: с фильтром "Diverging" это
            // даёт «включить всю категорию» одним нажатием, без фильтра —
            // обычные select all / clear all по всем 200.
            auto set_shown = [&](char v) {
                for (int n : shown) model.slancm_enabled[n - 1] = v;
                apply_cmap_mask();
            };
            if (ImGui::Button("Select all")) set_shown('1');
            ImGui::SameLine();
            if (ImGui::Button("Clear all")) set_shown('0');
            ImGui::SameLine();
            if (ImGui::Button("Reset to default")) {
                model.slancm_enabled = default_enabled_slancm();
                apply_cmap_mask();
            }
            ImGui::SameLine();
            {
                const int n_on = (int)std::count(model.slancm_enabled.begin(),
                                                 model.slancm_enabled.end(), '1');
                if (shown.size() == (size_t)kSlanCmCount)
                    ImGui::Text("%d / %d selected", n_on, kSlanCmCount);
                else
                    ImGui::Text("%d / %d selected   (buttons apply to the %d shown)",
                                n_on, kSlanCmCount, (int)shown.size());
            }

            {
                const float em      = ImGui::GetFontSize();
                const float strip_w = em * 6.0f;
                const float col_num   = em * 2.2f;
                const float col_strip = em * 4.4f;
                const float col_name  = col_strip + strip_w + em * 0.6f;
                const float col_cat   = col_name + em * 11.0f;
                const float row_h     = ImGui::GetFrameHeightWithSpacing();

                ImGui::BeginChild("##cmap_list", ImVec2(0, row_h * 12.0f), true);
                // Клиппер: 200 строк с градиентами рисовать каждый кадр незачем,
                // видно от силы дюжину. Высота строк одинаковая (её задаёт
                // чекбокс), поэтому хватает Begin(count) без items_height.
                ImGuiListClipper clip;
                clip.Begin((int)shown.size());
                while (clip.Step()) {
                    for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                        const int n = shown[i];
                        ImGui::PushID(n);
                        bool on = (model.slancm_enabled[n - 1] == '1');
                        if (ImGui::Checkbox("##on", &on)) {
                            model.slancm_enabled[n - 1] = on ? '1' : '0';
                            apply_cmap_mask();
                        }
                        ImGui::SameLine(col_num);
                        ImGui::TextDisabled("%d", n);
                        ImGui::SameLine(col_strip);
                        colormap_strip_item(slancm_id(n), strip_w, ImGui::GetFrameHeight());
                        ImGui::SameLine(col_name);
                        ImGui::TextUnformatted(slancm_name(n));
                        ImGui::SameLine(col_cat);
                        ImGui::TextDisabled("%s", slancm_category_name(slancm_category(n)));
                        ImGui::PopID();
                    }
                }
                ImGui::EndChild();
                if (shown.empty())
                    ImGui::TextDisabled("Nothing matches the filter.");
            }

            // Peak detection & regime thresholds — knobs configCUDA.h.
            // Уходят в NVRTC как #define, поэтому правка = перекомпиляция ядер
            // на следующем Run (set_peak_config бампает cache-epoch).
            ImGui::Separator();
            ImGui::Text("Peak detection & regime thresholds");
            ImGui::TextDisabled("Applies to every GPU calculation: bifurcations, LLE/LS,");
            ImGui::TextDisabled("basins, DFT. Changing a value recompiles the NVRTC");
            ImGui::TextDisabled("kernels on the next Run.");

            PeakConfig& pk = model.peak;
            bool pk_changed = false;

            pk_changed |= ImGui::Checkbox("Calculate peaks##do_calc_peaks", &pk.do_calculate_peaks);
            ImGui::TextDisabled("Off: the raw trajectory is passed downstream instead of its peaks.");
            pk_changed |= ImGui::Checkbox("Parabolic interpolation of peaks##do_interp_peaks", &pk.do_interpolate_peaks);
            ImGui::TextDisabled("Sub-sample peak position/amplitude from the 3-point parabola.");

            // Числовые поля — тем же виджетом, что и во вкладках анализа:
            // текстовый ввод с ↑/↓ по разряду, запятой вместо точки и дробями
            // "a/b", без навязанного "%.3e". Значение снимается на commit'е,
            // см. InputNumStrCommit.
            pk_changed |= InputNumStrCommit("eps fixed point##eps_fp",
                                            model.peak_eps_fixed_point_text,
                                            pk.eps_fixed_point, 220);
            ImGui::TextDisabled("Max sum|x(n)-x(n-1)| still reported as regime -1 (fixed point).");

            pk_changed |= InputNumStrCommit("eps peak delta##eps_pd",
                                            model.peak_eps_peak_delta_text,
                                            pk.eps_peak_delta, 220);
            ImGui::TextDisabled("Min rise/fall against a neighbour to accept a sample as a peak.");

            pk_changed |= InputNumStrCommit("eps inter-peak delta##eps_ipd",
                                            model.peak_eps_interPeak_delta_text,
                                            pk.eps_interPeak_delta, 220);
            ImGui::TextDisabled("Min interspike interval; closer peaks are merged. 0 = filter off.");

            pk_changed |= InputNumStrCommit("peak threshold##pk_thr",
                                            model.peak_threshold_text,
                                            pk.peak_threshold, 220);
            ImGui::TextDisabled("Peaks below this value are ignored.");

            // max peaks целочисленный, но поле — то же самое. Диапазон режем ДО
            // приведения к int: пользователь может ввести 1e30, а это уже UB на
            // касте. Ниже clamp_peak_config всё равно отработает — здесь только
            // защита самого каста.
            double max_peaks_val = (double)pk.max_amount_of_peaks;
            if (InputNumStrCommit("max peaks##pk_max", model.peak_max_amount_text,
                                  max_peaks_val, 220)) {
                max_peaks_val = std::max((double)kPeakCountMin,
                                std::min((double)kPeakCountMax, max_peaks_val));
                pk.max_amount_of_peaks = (int)std::llround(max_peaks_val);
                pk_changed = true;
            }
            ImGui::TextDisabled("Per-trajectory cap (%d..%d). Also sizes the device peak",
                                kPeakCountMin, kPeakCountMax);
            ImGui::TextDisabled("and interval buffers allocated for every sweep point.");
            if (pk.max_amount_of_peaks > 5000)
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "Large values raise GPU memory use for 2D bifurcation / basins.");

            if (ImGui::Button("Reset to configCUDA.h defaults")) {
                pk = PeakConfig{};
                model.sync_peak_text();
                pk_changed = true;
            }

            if (pk_changed) {
                const PeakConfig before = pk;
                clamp_peak_config(pk);
                // Кламп мог поправить введённое (отрицательный eps, потолок max
                // peaks) — тогда текст в поле уже врёт про конфиг, пересеиваем
                // буферы из значений.
                if (before.eps_fixed_point     != pk.eps_fixed_point ||
                    before.eps_peak_delta      != pk.eps_peak_delta ||
                    before.eps_interPeak_delta != pk.eps_interPeak_delta ||
                    before.max_amount_of_peaks != pk.max_amount_of_peaks)
                    model.sync_peak_text();
                set_peak_config(pk);       // бампает epoch -> PTX-кэши инвалидируются
                persist_settings(model);
            }

            // GPU floating point — опция компиляции NVRTC, а не #define, но логика та же: смена
            // значения = другой PTX, кэши модулей обоих движков инвалидируются ключом (см.
            // parametric_engine.h). Настройка ОДНА на приложение сознательно: карта и фазовый
            // портрет по её ячейке обязаны считать одинаково.
            ImGui::Separator();
            ImGui::Text("GPU floating point");
            ImGui::TextDisabled("Applies to every NVRTC kernel: bifurcations, LLE/LS, basins,");
            ImGui::TextDisabled("DFT, fast synchro AND phase portraits. Changing it recompiles");
            ImGui::TextDisabled("the kernels on the next Run.");

            if (ImGui::Checkbox("FMA contraction (--fmad)##nvrtc_fmad", &model.nvrtc_fmad)) {
                set_nvrtc_fmad(model.nvrtc_fmad);
                persist_settings(model);
            }
            if (model.nvrtc_fmad) {
                ImGui::TextDisabled("On (NVRTC default): a*b+c is fused, one rounding instead");
                ImGui::TextDisabled("of two. Faster and the usual choice.");
            } else {
                ImGui::TextDisabled("Off: multiply and add round separately. Closer to the CPU");
                ImGui::TextDisabled("fallback and to external references (MATLAB), slightly slower.");
            }
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "Results computed with different settings are not bit-comparable;");
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "on fractal basin boundaries they may differ visibly. The value in");
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "effect at Run time is written into every exported _config.csv.");
        }
        ImGui::End();
    }
}