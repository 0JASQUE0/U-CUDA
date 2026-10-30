#include "gui_internal.h"

// Главное окно: переключатель режимов Library / Analysis / Parametric
void gui_process_deferred(AppModel& model, const GuiCallbacks& cb) {
    if (!model.pending_netlist_export) return;
    model.pending_netlist_export = false;
    if (!cb.pick_save_file_netlist) return;

    PhaseAnalysisSession& s = model.phase_session;
    if (!s.result.circuit_valid) return;

    const std::string path = cb.pick_save_file_netlist();
    if (path.empty()) return;

    NetlistOptions no;
    no.title      = "U-CUDA synthesized circuit";
    no.var_names  = s.vars;
    no.rhs_text   = s.sys.rhs;
    no.scale      = s.result.circuit_scale;
    no.x0_volts   = s.result.circuit_x0;
    no.opamp      = s.result.circuit_opamp;
    no.h_ode      = s.result.circuit_h_ode;
    no.t_end_ode  = s.result.circuit_t_end_ode;
    no.dialect    = s.circuit_netlist_poly ? NetlistDialect::Poly : NetlistDialect::Behavioral;

    const std::string text = emit_spice_netlist(s.result.circuit_graph, no);
    std::ofstream f(path, std::ios::binary);
    if (f) { f << text; s.result.circuit_status = "netlist written to " + path; }
    else     s.result.circuit_status = "cannot write " + path;
}

// Автосохранение вкладки Order. Раньше _last_order.json писался только по
// завершении расчёта, а _last_order_windows.json — по флагу, который ставили
// лишь добавление/удаление окон и правка их членов. Всё прочее — новая или
// удалённая вкладка, переименование, схема, поля, вид карты, колормапа,
// лог-оси — до файла не доходило, пока не досчитается очередной Run, а при
// выходе без Run терялось. Хуже того, файлы расходились между собой: окна
// сохранялись со ссылкой на вкладку, которой в сохранённых конфигах ещё нет.
//
// Теперь оба файла пишутся, как только сериализация отличается от последней
// записанной. Сериализация Order — несколько килобайт текста, сравнивать её
// каждый кадр дешевле, чем расставлять флаг по сотне мест правки (включая
// ПКМ-рассылку в конфиги Order с чужих вкладок).
//
// Писать можно только сессию ТЕКУЩЕЙ системы: после смены системы на другой
// вкладке order_session держит старую, пока в Order не зайдут снова, и без
// этой проверки её содержимое легло бы в файл новой системы.
static void autosave_order_session(AppModel& model, SystemLibrary& lib) {
    model.order_plot_windows_dirty = false;   // флаг поглощён сравнением ниже
    if (model.loaded_name.empty()) return;
    const OrderAnalysisSession& s = model.order_session;
    if (s.loaded_system_name.empty() || s.loaded_system_name != model.name) return;

    if (model.order_autosave_target != model.loaded_name) {
        model.order_autosave_target = model.loaded_name;
        model.order_autosave_json.clear();
        model.order_windows_autosave_json.clear();
    }
    std::string jo = session_to_json_order(s);
    if (jo != model.order_autosave_json) {
        lib.save_session(model.loaded_name, "_last_order", jo);
        model.order_autosave_json = std::move(jo);
    }
    std::string jw = session_to_json_order_windows(model.order_plot_windows);
    if (jw != model.order_windows_autosave_json) {
        lib.save_session(model.loaded_name, "_last_order_windows", jw);
        model.order_windows_autosave_json = std::move(jw);
    }
}

void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    set_field_latex(model.field_latex);
    set_field_labels(&model.field_labels);
    set_field_library(&lib);
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

    // custom_schemes — единственное поле, которое может отредактироваться в System tab БЕЗ
    // переключения режима (без start_*_analysis). Чтобы scheme combo в Phase/Parametric/Basins/
    // FastSync увидел свежий список сразу после "+ Add custom scheme" или правки тела схемы,
    // синкаем копию live → сессии каждый кадр. Раньше Basins был пропущен полностью, а FastSync
    // синкался только при активном окне, из-за чего отредактированное cs.body не доходило до
    // compute_krs_for_scheme на момент Run и NVRTC брал устаревший body из кеша.
    model.phase_session.custom_schemes       = model.custom_schemes;
    model.bifurcation_session.custom_schemes = model.custom_schemes;
    model.lle_session.custom_schemes         = model.custom_schemes;
    model.ls_session.custom_schemes          = model.custom_schemes;
    model.dft1d_session.custom_schemes       = model.custom_schemes;
    model.basins_session.custom_schemes      = model.custom_schemes;
    model.fastsync_session.custom_schemes    = model.custom_schemes;
    model.order_session.custom_schemes       = model.custom_schemes;
    model.network_session.custom_schemes     = model.custom_schemes;
    // Custom tab owns 5 isolated sub-sessions — same per-frame sync applies.
    model.custom_session.custom_schemes                = model.custom_schemes;
    model.custom_session.bif_session.custom_schemes    = model.custom_schemes;
    model.custom_session.lle_session.custom_schemes    = model.custom_schemes;
    model.custom_session.ls_session.custom_schemes     = model.custom_schemes;
    model.custom_session.phase_session.custom_schemes  = model.custom_schemes;
    model.custom_session.basins_session.custom_schemes = model.custom_schemes;
    // Экстраполяционные обёртки — тем же покадровым синком и по тем же
    // причинам: собранная в Library схема должна появиться в комбо сразу.
    const std::vector<std::string> wrappers_on = model.enabled_wrapper_schemes();   // галочки списка обёрток
    model.phase_session.wrapper_schemes       = wrappers_on;
    model.bifurcation_session.wrapper_schemes = wrappers_on;
    model.lle_session.wrapper_schemes         = wrappers_on;
    model.ls_session.wrapper_schemes          = wrappers_on;
    model.dft1d_session.wrapper_schemes       = wrappers_on;
    model.basins_session.wrapper_schemes      = wrappers_on;
    model.fastsync_session.wrapper_schemes    = wrappers_on;
    model.order_session.wrapper_schemes       = wrappers_on;
    model.network_session.wrapper_schemes     = wrappers_on;
    model.custom_session.wrapper_schemes                = wrappers_on;
    model.custom_session.bif_session.wrapper_schemes    = wrappers_on;
    model.custom_session.lle_session.wrapper_schemes    = wrappers_on;
    model.custom_session.ls_session.wrapper_schemes     = wrappers_on;
    model.custom_session.phase_session.wrapper_schemes  = wrappers_on;
    model.custom_session.basins_session.wrapper_schemes = wrappers_on;
    // Включённые встроенные схемы — тем же покадровым синком. Раньше они
    // доезжали до сессий только через propagate_to_sessions (сохранение
    // активной системы в Library) или load_from_record, а Order и Network
    // propagate не видел вовсе: отмеченная и сохранённая схема не появлялась
    // в их комбо до перезагрузки системы. Пустой список = фильтра нет, как и
    // у enabled_builtins_from_record.
    {
        // ДАУ: флагов scheme_* у неё нет, а схемы ОДУ на ней не строятся — список
        // свой (иначе этот синк каждый кадр затирал бы его схемами ОДУ).
        std::vector<std::string> enabled_now;
        if (model.dae_native()) {
            for (const auto& n : dae_scheme_names())
                if (model.dae_scheme_on(n)) enabled_now.push_back(n);
            if (model.dae_scheme_on(kBdfSchemeDae)) enabled_now.push_back(kBdfSchemeDae);   // как enabled_builtins_from_record
        }
        else {
            for (const auto& b : kBuiltinSchemes)
                if (model.*b.flag) enabled_now.push_back(b.name);
            // флага нет: всегда, как enabled_builtins_from_record (пустой список встроенных остаётся пустым —
            // у draw_scheme_combo это "без фильтра")
            if (!enabled_now.empty()) enabled_now.push_back(kBdfSchemeOde);
        }
        // BDF 1-5 — только там, где есть адаптивный шаг (Analysis, БД, LLE/LS, Metrics, бассейны); DFT, FastSync,
        // Order и Network многошагового драйвера не имеют — им список без него.
        std::vector<std::string> enabled_fixed;
        for (const auto& n : enabled_now) if (!scheme_is_multistep(n)) enabled_fixed.push_back(n);
        for (std::vector<std::string>* dst : {
                 &model.phase_session.enabled_builtin_schemes,
                 &model.bifurcation_session.enabled_builtin_schemes,
                 &model.lle_session.enabled_builtin_schemes,
                 &model.ls_session.enabled_builtin_schemes,
                 &model.metrics_session.enabled_builtin_schemes,
                 &model.basins_session.enabled_builtin_schemes,
                 &model.custom_session.enabled_builtin_schemes,
                 &model.custom_session.bif_session.enabled_builtin_schemes,
                 &model.custom_session.lle_session.enabled_builtin_schemes,
                 &model.custom_session.ls_session.enabled_builtin_schemes,
                 &model.custom_session.phase_session.enabled_builtin_schemes,
                 &model.custom_session.basins_session.enabled_builtin_schemes })
            if (*dst != enabled_now) *dst = enabled_now;
        for (std::vector<std::string>* dst : {
                 &model.dft1d_session.enabled_builtin_schemes,
                 &model.fastsync_session.enabled_builtin_schemes,
                 &model.order_session.enabled_builtin_schemes,
                 &model.network_session.enabled_builtin_schemes })
            if (*dst != enabled_fixed) *dst = enabled_fixed;
    }

    // Auto-labels: pre-frame refresh so all label-consumers (tab-bar names,
    // plot legend, window title, Plot windows section) see the same value.
    refresh_auto_labels(model);

    // poll'им async-расчёты независимо от текущего режима — чтобы при
    // возврате в этот режим пользователь сразу увидел готовый результат
    if (model.bifurcation_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_parametric",
                             session_to_json_parametric(model.bifurcation_session));
    }
    if (model.lle_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_lle",
                             session_to_json_lle(model.lle_session));
    }
    if (model.ls_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_ls",
                             session_to_json_ls(model.ls_session));
    }
    {
        // Индекс запоминаем ДО poll: он сбрасывает running_config_index.
        const int met_idx = model.metrics_session.running_config_index;
        if (model.metrics_session.poll()) {
            if (!model.loaded_name.empty())
                lib.save_session(model.loaded_name, "_last_metrics",
                                 session_to_json_metrics(model.metrics_session));
            // Вкладка новая: у сохранённых раскладок окон под неё окна нет, и
            // посчитанный результат иначе некуда было бы смотреть. Окно заводится,
            // только если конфиг не показан ни в одном окне своей размерности.
            auto& cfgs = model.metrics_session.configs;
            if (met_idx >= 0 && met_idx < (int)cfgs.size() &&
                (cfgs[met_idx].mode_2d ? cfgs[met_idx].last_run_2d_ok : cfgs[met_idx].last_run_ok)) {
                const bool m2d = cfgs[met_idx].mode_2d;
                bool shown = false;
                for (const auto& w : model.parametric_plot_windows)
                    if (w.kind == ParametricPlotWindow::Kind::Metrics && w.mode_2d == m2d &&
                        std::find(w.members.begin(), w.members.end(), met_idx) != w.members.end())
                        shown = true;
                if (!shown) {
                    ParametricPlotWindow* target = nullptr;
                    if (!m2d)   // 1D-окно оверлеит кривые — дописываемся в имеющееся
                        for (auto& w : model.parametric_plot_windows)
                            if (w.kind == ParametricPlotWindow::Kind::Metrics && !w.mode_2d) { target = &w; break; }
                    if (target) {
                        target->members.push_back(met_idx);
                        model.parametric_plot_windows_dirty = true;
                    } else {
                        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Metrics, m2d, { met_idx });
                    }
                }
            }
        }
    }
    // Plot windows list is UI-only (not tied to any session's poll()) — save
    // whenever add/remove/membership-edit touched it this frame, rather than
    // every frame.
    if (model.parametric_plot_windows_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_parametric_windows",
                             session_to_json_parametric_windows(model.parametric_plot_windows));
        model.parametric_plot_windows_dirty = false;
    }
    // DFT1D: independent multi-config session (own queue, own poll) — same
    // save-after-poll + dirty-plot-windows pattern as above.
    if (model.dft1d_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_dft1d",
                             session_to_json_dft1d(model.dft1d_session));
    }
    if (model.dft1d_plot_windows_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_dft1d_windows",
                             session_to_json_dft1d_windows(model.dft1d_plot_windows));
        model.dft1d_plot_windows_dirty = false;
    }
    if (model.phase_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last",
                             session_to_json(model.phase_session));
    }
    // Continuation timer: after each frame finishes and once the debounce
    // window elapses, schedule the next chunk. Runs off the per-frame poll —
    // no separate thread, no wakeup while the previous chunk is still async.
    {
        auto& ps = model.phase_session;
        if (ps.continuation_active && !ps.continuation_paused && !ps.in_flight) {
            const auto  now      = std::chrono::steady_clock::now();
            const double delay_ms = std::max(0.0, parse_num(ps.continuation_delay_ms, 50.0));
            const auto  due      = ps.continuation_last_frame +
                                   std::chrono::milliseconds((long long)delay_ms);
            if (now >= due) {
                ps.continuation_last_frame = now;
                ps.recompute_async();
            }
        }
    }
    // Basins: один config на сессию. Сохраняем JSON каждый кадр (после poll
    // - но также при изменении полей в controls). Здесь только after-poll save.
    if (model.basins_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
    }
    // Фазовые портреты по бассейнам: свой poll + свой тик очереди. Идут
    // независимо от текущего режима — вернувшись в Basins, пользователь сразу
    // видит готовый результат (как и у всех остальных сессий).
    if (model.basins_session.poll_phase()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
    }
    basins_phase_tick(model);
    model.basins_session.start_next_phase();
    if (model.basins_session.phase_settings_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
        model.basins_session.phase_settings_dirty = false;
    }
    // FastSync: poll worker future; on completion persist session JSON.
    // Без этого вызова in_flight никогда не сбрасывается → "Running" висит вечно.
    if (model.fastsync_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_fastsync",
                             session_to_json_fastsync(model.fastsync_session));
    }
    // Order: тот же контракт, что у FastSync — без poll() in_flight никогда не
    // сбросится и "Running" повиснет навсегда. Сессию и окна пишет
    // autosave_order_session — по любому изменению, а не по завершении.
    model.order_session.poll();
    autosave_order_session(model, lib);
    // Network: тот же контракт, что у Order — без poll() in_flight никогда не
    // сбросится и "Running" повиснет навсегда.
    if (model.network_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_network",
                             session_to_json_network(model.network_session));
    }
    // Custom tab: aggregate poll of all 5 sub-sessions; one bundle save on any completion, чтобы
    // не переписывать _last_custom.json пять раз. Сохраняем и на любой мутации workspace
    // (add/close/rename таба, drag сплиттера, перенос окна между табами) — с одним лишь
    // сохранением по завершении компьюта закрытый таб воскресал после перезапуска.
    bool custom_dirty = model.custom_session.poll_all();
    if (model.custom_session.workspace.dirty) {
        custom_dirty = true;
        model.custom_session.workspace.dirty = false;
    }
    if (custom_dirty) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_custom",
                             session_to_json_custom(model.custom_session));
        // When the whole pipeline is drained (queue empty AND nothing left
        // running), promote pending → committed for each level's signature
        // so the next Run knows which levels are still up-to-date.
        if (model.custom_queue.empty() && !model.custom_session.any_in_flight())
            model.custom_session.commit_pending_signatures();
    }

    // Tick parametric-очереди: если ни одна из BD/LLE/LS не in_flight и в
    // очереди есть элементы — берём следующий и стартуем. start_next сам
    // проверяет условие и безопасен к вызову каждый кадр.
    model.start_next_in_parametric_queue();
    // То же для dft1d-очереди (независимая).
    model.start_next_in_dft1d_queue();
    // То же для basins-очереди (независимая).
    model.start_next_in_basins_queue();
    // То же для fastsync-очереди (независимая).
    model.start_next_in_fastsync_queue();
    // То же для order-очереди (независимая).
    model.start_next_in_order_queue();
    // То же для network-очереди (независимая).
    model.start_next_in_network_queue();
    // Custom tab has its own queue (2D → 1D → Phase/Basins pipeline).
    model.start_next_in_custom_queue();
    // Фоновая компиляция под текущие настройки параметрики, чтобы первый Run не ждал NVRTC.
    model.poll_parametric_prewarm();

    // Ctrl+Z / Ctrl+Shift+Z — отмена и повтор МАССОВЫХ команд: "Calculation
    // range from view" (Ctrl+T / Ctrl+Shift+T) и "Apply to all calculation tabs".
    // Обычные правки полей в стек не идут. Стек свой у каждой вкладки, и
    // работают хоткеи по ОТКРЫТОЙ (см. AppModel::undo_stacks).
    // Условие !WantTextInput принципиально: внутри активного InputText Ctrl+Z —
    // это undo самого ImGui (stb_textedit, 99 шагов), и перехват здесь сломал бы
    // набор текста. Alt исключён, чтобы не срабатывать на Ctrl+Alt+Z.
    {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && !io.KeyAlt && !io.WantTextInput &&
            ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
            if (io.KeyShift) do_redo(model);
            else             do_undo(model);
        }
    }

    // переключатель режимов. Скрытые в Settings вкладки пропускаем; SameLine
    // ставится ПЕРЕД кнопкой, иначе скрытая оставляла бы разрыв в ряду.
    // Если активной оказалась скрытая вкладка (маску принесли из конфига или
    // правкой файла) — уводим в Library: рисовать её содержимое без кнопки
    // в ряду значит запереть пользователя в режиме, из которого не видно выхода.
    for (const auto& t : kModeTabs)
        if (!t.always && t.mode == model.app_mode && model.tab_hidden(t.id))
            model.app_mode = AppModel::AppMode::Library;
    int mode = (int)model.app_mode;
    {
        bool first_tab = true;
        for (const auto& t : kModeTabs) {
            if (!t.always && model.tab_hidden(t.id)) continue;
            if (!first_tab) ImGui::SameLine();
            first_tab = false;
            ImGui::RadioButton(t.label, &mode, (int)t.mode);
        }
    }

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
    else if (model.metrics_session.in_flight)
        busy = busy_from(model.metrics_session, BusyKind::Metrics,
                         running_label(model.metrics_session.configs,
                                       model.metrics_session.running_config_index, "metrics"));
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
        DoneCand candidates[11] = {
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
            { BusyKind::Metrics, model.metrics_session.last_run_completed_at,
              &model.metrics_session.last_run_label,
              model.metrics_session.last_run_succeeded,
              model.metrics_session.last_run_seconds },
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

    // Оценка времени счёта траекторий (time_estimate.h). Пока идёт расчёт — его собственная,
    // снятая один раз на старте: правка настроек под следующий прогон её не трогает. В простое —
    // по настройкам открытой вкладки.
    TrajTimeEstimator::Result est;
    bool show_est = false;
    {
        // Время завершённого расчёта уточняет поправку оценщика на настоящие ядра. Смотрим саму
        // сессию, а не плашку Done: очередь запускает следующий элемент в том же кадре, и кадра
        // простоя между ними может не быть.
        if (model.traj_learn_pending) {
            auto learn_from = [&](const auto& s) {
                if (s.last_run_completed_at <= model.traj_running_start) return;   // ещё идёт
                if (s.last_run_succeeded)
                    model.traj_estimator.learn(model.traj_running_work, s.last_run_seconds);
                model.traj_learn_pending = false;
            };
            switch ((BusyKind)model.traj_running_kind) {
            case BusyKind::LLE:    learn_from(model.lle_session);    break;
            case BusyKind::LS:     learn_from(model.ls_session);     break;
            case BusyKind::Basins: learn_from(model.basins_session); break;
            default:               model.traj_learn_pending = false; break;
            }
        }
        const bool running = busy.kind != BusyKind::None && !show_done;
        if (running) {
            if (model.traj_running_start != busy.start) {
                model.traj_running_start = busy.start;
                model.traj_running_work  = running_traj_work(model);
                model.traj_running_kind  = (int)busy.kind;
                model.traj_learn_pending = model.traj_running_work.valid;
            }
            est = model.traj_estimator.estimate(model.traj_running_work, false);
            show_est = model.traj_running_work.valid || !model.traj_running_work.why.empty();
        } else {
            const auto now = std::chrono::steady_clock::now();
            if (now - model.traj_idle_at > std::chrono::milliseconds(300)) {
                model.traj_idle_at   = now;
                model.traj_idle_work = current_tab_traj_work(model);
            }
            est = model.traj_estimator.estimate(model.traj_idle_work, true);
            show_est = model.traj_idle_work.valid || !model.traj_idle_work.why.empty();
        }
    }
    char   est_text[64] = "";
    ImVec4 est_col(0.55f, 0.55f, 0.55f, 1.0f);
    if (show_est) {
        if (est.state == TrajTimeEstimator::State::Ready) {
            std::snprintf(est_text, sizeof(est_text), "est. %s%s", est.rough ? "~" : "",
                          traj_format_seconds(est.seconds).c_str());
            est_col = ImVec4(0.55f, 0.78f, 1.0f, 1.0f);
        } else if (est.state == TrajTimeEstimator::State::Pending) {
            std::snprintf(est_text, sizeof(est_text), "est. ...");
        } else {
            std::snprintf(est_text, sizeof(est_text), "est. n/a");
        }
    }
    auto draw_est = [&]() {
        ImGui::TextColored(est_col, "%s", est_text);
        if (ImGui::IsItemHovered() && !est.detail.empty())
            ImGui::SetTooltip("Estimated trajectory integration time\n%s\nRight-click: recalibrate the GPU.",
                              est.detail.c_str());
        if (ImGui::BeginPopupContextItem("##traj_est_ctx")) {
            if (ImGui::MenuItem("Recalibrate GPU")) model.traj_estimator.recalibrate();
            ImGui::EndPopup();
        }
    };

    if (busy.kind == BusyKind::None && show_est) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowSize().x - ImGui::CalcTextSize(est_text).x - 12.0f);
        draw_est();
    }

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
            else if (busy.kind == BusyKind::Order)    queue_n = model.order_queue.size();
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
        if (show_est)  total_w += ImGui::CalcTextSize(est_text).x + pad;

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
        if (show_est) {
            ImGui::SameLine();
            draw_est();
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
                    case BusyKind::Metrics:  model.metrics_session.request_cancel();     break;
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
                model.order_queue.clear();
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
           || model.metrics_session.in_flight
           || model.dft1d_session.in_flight
           || model.basins_session.in_flight
           || model.fastsync_session.in_flight
           || model.order_session.in_flight
           || model.network_session.in_flight
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
        if (ImGui::BeginCombo("##topsyssel", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
            // Группы по вкладкам Library: ODE, DAE, MAP. ДАУ считает пока только
            // Analysis (схемы DAE BDF1 / Trapezoidal); другие вкладки откажут при Run.
            const LibraryListCache& cache = library_list_cached(lib);
            int listed = 0;
            for (int group = 0; group < kLibTabCount; ++group) {
                const bool dae = (LibTab)group == LibTab::Dae;
                bool header = false;
                for (size_t i = 0; i < cache.names.size(); ++i) {
                    const std::string& nm = cache.names[i];
                    if (cache.tab[i] != group) continue;
                    // Ticked off in the Library tab. The active system stays listed
                    // regardless, so the combo never previews a name it can't show.
                    if (model.system_hidden(nm) && model.name != nm) continue;
                    if (!header) { ImGui::SeparatorText(kLibTabName[group]); header = true; }
                    ++listed;
                    if (ImGui::Selectable(nm.c_str(), model.name == nm))
                        apply_system_switch(model, lib, nm);
                    if (dae && ImGui::IsItemHovered())
                        ImGui::SetTooltip("DAE: DAE methods in Analysis on the GPU;\n"
                                          "with 'Solve as ODE' (Library -> DAE) — every tab and ODE scheme");
                }
            }
            if (listed == 0) ImGui::TextDisabled("(all systems hidden in Library)");
            ImGui::EndCombo();
        }
        if (any_in_flight) ImGui::EndDisabled();

        // Что сделала последняя массовая команда / Ctrl+Z / Ctrl+Shift+Z.
        // Держим несколько секунд: "Apply to..." правит и невидимые сейчас
        // вкладки, и без строки в шапке это неотличимо от "ничего не произошло".
        //
        // Рисуем СТРОГО ПОСЛЕ комбо. Раньше строка стояла между рядом вкладок и
        // комбо и ломала ему раскладку: край ряда (mode_tabs_end_x) снимается
        // ДО неё, а комбо центрируется абсолютным SetCursorPosX — длинное
        // сообщение заезжало ровно туда, куда становится комбо. Здесь же оно
        // занимает свободное место справа и ни на что не влияет.
        if (!model.undo_note.empty()) {
            if (ImGui::GetTime() - model.undo_note_time > 4.0) {
                model.undo_note.clear();
            } else {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", model.undo_note.c_str());
            }
        }
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
    bool entering_network = (AppModel::AppMode)mode == AppModel::AppMode::Network &&
                            model.app_mode != AppModel::AppMode::Network;
    if (entering_phase || entering_par || entering_dft1d || entering_basins || entering_fastsync
        || entering_custom || entering_order || entering_network) {
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
            std::string jm = lib.load_session(model.loaded_name, "_last_metrics");
            apply_session_json(model, jm, model.metrics_session, session_from_json_metrics, "_last_metrics");
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
    auto order_need_init = model.order_session.loaded_system_name != model.name
                        || model.order_session.vars.empty()
                        || model.order_session.vars   != model.known_vars
                        || model.order_session.params != model.known_params;
    if (entering_order && order_need_init) {
        model.start_order_analysis();
        if (!model.loaded_name.empty()) {
            std::string jo = lib.load_session(model.loaded_name, "_last_order");
            apply_session_json(model, jo, model.order_session, session_from_json_order, "_last_order");
            std::string jw = lib.load_session(model.loaded_name, "_last_order_windows");
            model.load_or_init_order_plot_windows(jw);
            order_autosave_seed(model);
        }
    }
    auto network_need_init = model.network_session.loaded_system_name != model.name
                          || model.network_session.vars.empty()
                          || model.network_session.vars   != model.known_vars
                          || model.network_session.params != model.known_params;
    if (entering_network && network_need_init) {
        model.start_network_analysis();
        if (!model.loaded_name.empty()) {
            std::string jn = lib.load_session(model.loaded_name, "_last_network");
            apply_session_json(model, jn, model.network_session, session_from_json_network, "_last_network");
        }
    }
    // Окна могли не появиться вовсе: несохранённая система (loaded_name пуст,
    // грузить нечего), сессия из старой сборки без файла окон, вкладка открыта
    // впервые. Проверяем каждый кадр, а не только на входе, — иначе Order
    // рисовал бы пустой экран без единого окна и без способа его завести.
    if ((AppModel::AppMode)mode == AppModel::AppMode::Order
        && model.order_plot_windows.empty()
        && !model.order_session.configs.empty())
        model.load_or_init_order_plot_windows({});
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
        //
        // Две вкладки: ODE (бывшая Editor — "###Editor" сохраняет её место в
        // imgui.ini) и DAE. Редактор открывается в той вкладке, к которой
        // относится черновик; вторая в это время показывает свой список.
        // Три вкладки: ODE (бывшая Editor), DAE, MAP.
        const bool editing = model.library_edit_mode != AppModel::LibraryEditMode::None;
        const LibTab edit_tab = lib_tab_of(model.library_edit_buffer->is_dae,
                                           model.library_edit_buffer->is_map);
        static const char* const kLibWindow[kLibTabCount] =
            { "ODE###Editor", "DAE###LibraryDAE", "MAP###LibraryMAP" };
        ImGuiID ode_dock = 0;
        for (int t = 0; t < kLibTabCount; ++t) {
            // Впервые — в тот же узел дока, что и ODE, чтобы встать рядом вкладкой.
            if (t > 0 && ode_dock != 0) ImGui::SetNextWindowDockID(ode_dock, ImGuiCond_FirstUseEver);
            if (ImGui::Begin(kLibWindow[t])) {
                if (editing && edit_tab == (LibTab)t) draw_library_editor(model, lib, cb);
                else                                  draw_library_list(model, lib, (LibTab)t);
            }
            if (t == 0) ode_dock = ImGui::GetWindowDockID();
            ImGui::End();
        }
    }
    else if (model.app_mode == AppModel::AppMode::Analysis) {
        // режим анализа: панель настроек + окна проекций (докаются пользователем)
        if (ImGui::Begin("Controls")) {
            draw_phase_controls(model.phase_session, [&model, &lib]() {
                if (!model.loaded_name.empty()) {
                    model.from_record(lib.load(model.loaded_name));   // reference from disk
                    model.start_phase_analysis();
                }
            }, &model, &cb);
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
        // Order mode: панель настроек + динамический список окон графика
        // (кривая p, кривая ошибки, 2D-карта, диаграмма производительности).
        if (ImGui::Begin("Order Controls")) {
            draw_order_controls(model, lib);
        }
        ImGui::End();
        draw_order_plot_windows(model, cb);
    }
    else if (model.app_mode == AppModel::AppMode::Network) {
        // Network mode: панель настроек + граф + кривые узлов + растр.
        // Окна фиксированные (в отличие от Order): собирать из них
        // произвольные комбинации пока незачем, а смотреть на сеть сразу в
        // трёх разрезах нужно всегда.
        static std::map<int, std::unique_ptr<PlotRenderer>> net_renderers;
        static Plot2DView   net_series_view;
        static HeatmapView  net_raster_view;
        static int seen_epoch = 0;
        if (session_reloaded(model, AppModel::AppMode::Network, seen_epoch)) {
            invalidate_view(net_series_view);
            invalidate_view(net_raster_view);
        }
        auto& rnd_series = net_renderers[0];
        if (!rnd_series) rnd_series = std::make_unique<PlotRenderer>();
        auto& rnd_raster = net_renderers[1];
        if (!rnd_raster) rnd_raster = std::make_unique<PlotRenderer>();

        if (ImGui::Begin("Network Controls")) {
            draw_network_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Network Graph")) {
            draw_network_graph_window(model);
        }
        ImGui::End();
        if (ImGui::Begin("Node Series")) {
            draw_network_series_window(model, *rnd_series, net_series_view);
        }
        ImGui::End();
        if (ImGui::Begin("Spatiotemporal")) {
            draw_network_raster_window(model, *rnd_raster, net_raster_view);
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
            cfg.plot_math_font         = m.plot_math_font;
            cfg.field_latex            = m.field_latex;
            cfg.show_alg_ic            = m.show_alg_ic;
            cfg.plot_font_scale        = m.plot_font_scale;
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
            cfg.nvrtc_rdc              = m.nvrtc_rdc;
            cfg.gpu_block_size         = m.gpu_block_size;
            cfg.hidden_tabs            = m.hidden_tabs;
            cfg.hidden_schemes         = m.hidden_schemes;
            cfg.hidden_dae_methods     = m.hidden_dae_methods;
            cfg.hidden_systems         = m.hidden_systems;
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

            float plot_fs = model.plot_font_scale;
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderFloat("Plot font size", &plot_fs, 0.5f, 3.0f, "%.2fx")) {
                set_plot_font_scale(plot_fs);
                model.plot_font_scale = plot_font_scale();   // клампнутое значение
                persist_settings(model);
            }
            ImGui::TextDisabled("Size of every label on the plots (ticks, axis names,");
            ImGui::TextDisabled("legend, colorbar), relative to the UI font. Plot margins");
            ImGui::TextDisabled("follow it, so labels stay inside the diagram block.");

            bool math_font = model.plot_math_font;
            if (ImGui::Checkbox("LaTeX-style labels on plots", &math_font)) {
                model.plot_math_font = math_font;
                persist_settings(model);
            }
            ImGui::TextDisabled("On: serif math typesetting for every label on the plots —");
            ImGui::TextDisabled("\"sigma\" prints as a greek glyph, x1 as x with a subscript,");
            ImGui::TextDisabled("variables go italic, 1e-05 becomes a power of ten.");
            ImGui::TextDisabled("Computer Modern (the TeX face) is picked up from fonts\\ if");
            ImGui::TextDisabled("present; otherwise Times New Roman. Off: plain UI font.");

            bool field_tex = model.field_latex;
            if (ImGui::Checkbox("Use LaTeX for parameters and variables", &field_tex)) {
                model.field_latex = field_tex;
                persist_settings(model);
            }
            ImGui::TextDisabled("On: names next to the parameter and initial-condition fields and in the");
            ImGui::TextDisabled("projection lists are typeset as formulas (vc_c1 -> vc with subscript C1,");
            ImGui::TextDisabled("sigma -> greek), or with the labels set in Groups.... Off: the plain names.");

            bool alg_ic = model.show_alg_ic;
            if (ImGui::Checkbox("Initial conditions of algebraic variables (DAE)", &alg_ic)) {
                model.show_alg_ic = alg_ic;
                persist_settings(model);
            }
            ImGui::TextDisabled("Off: the tabs show initial conditions of the state variables x only. The algebraic");
            ImGui::TextDisabled("z are recomputed anyway (DC operating point / projection onto g = 0), their fields");
            ImGui::TextDisabled("are only Newton's starting guess. The library editor always shows all of them.");

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

            // Видимость вкладок и схем. Обе маски — чисто интерфейсные: на
            // расчёты, сохранённые сессии и system.json они не влияют, только
            // убирают лишнее из виду. Живут в _app_config.json, одна на
            // приложение, как и остальные настройки этой вкладки.
            ImGui::Separator();
            ImGui::Text("Tabs");
            ImGui::TextDisabled("Which analysis tabs appear in the top row. Library and");
            ImGui::TextDisabled("Settings are always shown. Hiding a tab does not touch its");
            ImGui::TextDisabled("sessions: ticking it back brings everything as it was.");
            for (const auto& t : kModeTabs) {
                if (t.always) continue;
                bool on = !model.tab_hidden(t.id);
                ImGui::PushID(t.id);
                if (ImGui::Checkbox(t.label, &on)) {
                    AppModel::set_name_hidden(model.hidden_tabs, t.id, !on);
                    persist_settings(model);
                }
                ImGui::PopID();
            }

            ImGui::Separator();
            ImGui::Text("Integration schemes");
            ImGui::TextDisabled("Which built-in schemes are offered in \"Schemes to generate\"");
            ImGui::TextDisabled("on the System tab. Unticking one only hides the checkbox:");
            ImGui::TextDisabled("a system that already has the scheme enabled keeps generating");
            ImGui::TextDisabled("it, and the System tab says so.");
            {
                auto set_all_schemes = [&](bool shown) {
                    for (const auto& b : kBuiltinSchemes)
                        AppModel::set_name_hidden(model.hidden_schemes, b.name, !shown);
                    persist_settings(model);
                };
                if (ImGui::Button("Show all##schemes")) set_all_schemes(true);
                ImGui::SameLine();
                if (ImGui::Button("Hide all##schemes")) set_all_schemes(false);
                ImGui::SameLine();
                int n_on = 0;
                for (const auto& b : kBuiltinSchemes)
                    if (!model.scheme_hidden(b.name)) ++n_on;
                ImGui::Text("%d / %d shown", n_on, kBuiltinSchemeCount);

                // Раскладка — как в System tab: группа порядка одной строкой,
                // чтобы список читался теми же блоками, что и там.
                std::string  shown_group;
                bool first_in_line = true;
                for (const auto& b : kBuiltinSchemes) {
                    if (scheme_group_title(b) != shown_group) {
                        ImGui::SeparatorText(scheme_group_title(b).c_str());
                        shown_group   = scheme_group_title(b);
                        first_in_line = true;
                    }
                    if (!first_in_line) ImGui::SameLine();
                    first_in_line = false;
                    bool on = !model.scheme_hidden(b.name);
                    ImGui::PushID(b.name);
                    if (ImGui::Checkbox(b.name, &on)) {
                        AppModel::set_name_hidden(model.hidden_schemes, b.name, !on);
                        persist_settings(model);
                    }
                    ImGui::PopID();
                    if (b.tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", b.tooltip);
                }
            }

            // То же для методов ДАУ: список "DAE methods to generate" редактора ДАУ.
            ImGui::Separator();
            ImGui::Text("DAE methods");
            ImGui::TextDisabled("Which DAE methods are offered in the DAE editor. Unticking one");
            ImGui::TextDisabled("only hides the checkbox, as for the integration schemes.");
            {
                const std::vector<std::string> dae = dae_method_list();
                auto set_all_dae = [&](bool shown) {
                    for (const auto& n : dae) AppModel::set_name_hidden(model.hidden_dae_methods, n, !shown);
                    persist_settings(model);
                };
                if (ImGui::Button("Show all##dae_methods")) set_all_dae(true);
                ImGui::SameLine();
                if (ImGui::Button("Hide all##dae_methods")) set_all_dae(false);
                ImGui::SameLine();
                int n_on = 0;
                for (const auto& n : dae)
                    if (!model.dae_method_hidden(n)) ++n_on;
                ImGui::Text("%d / %d shown", n_on, (int)dae.size());
                for (size_t i = 0; i < dae.size(); ++i) {
                    if (i % 4 != 0) ImGui::SameLine();
                    bool on = !model.dae_method_hidden(dae[i]);
                    ImGui::PushID(("dae_m_" + dae[i]).c_str());
                    if (ImGui::Checkbox(dae[i].c_str(), &on)) {
                        AppModel::set_name_hidden(model.hidden_dae_methods, dae[i], !on);
                        persist_settings(model);
                    }
                    ImGui::PopID();
                }
            }

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

            ImGui::Separator();
            ImGui::Text("Kernel compilation");
            ImGui::TextDisabled("Parametric analyses only; phase portraits build their own tiny");
            ImGui::TextDisabled("kernel and are unaffected.");

            if (ImGui::Checkbox("Split compilation (-rdc + link)##nvrtc_rdc", &model.nvrtc_rdc)) {
                set_nvrtc_rdc(model.nvrtc_rdc);
                persist_settings(model);
            }
            if (model.nvrtc_rdc) {
                ImGui::TextDisabled("On: the NonLinAnal library compiles once and is reused, only");
                ImGui::TextDisabled("the step is rebuilt when the scheme changes. Measured on a 1D");
                ImGui::TextDisabled("bifurcation: 62 ms per scheme switch instead of 1700 ms.");
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "The step is called instead of being inlined, so FMA contraction");
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "across that boundary is gone. Measured on Lorenz: 13 of 14 schemes");
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "bit-identical, Complex Implicit Euler differs by up to 3e-10 (with");
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "FMA off both agree exactly). Maps and phase portraits may therefore");
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                    "disagree in the last digits on such schemes.");
            } else {
                ImGui::TextDisabled("Off: every kernel is built as one translation unit with only");
                ImGui::TextDisabled("the library kernels the analysis launches (the rest are cut,");
                ImGui::TextDisabled("same code as the full build). The step is inlined and results");
                ImGui::TextDisabled("stay bit-comparable with older runs; a scheme change costs a");
                ImGui::TextDisabled("few seconds (Lorenz, 11 analyses: 14 s instead of 95 s).");
            }

            // GPU launch width. В отличие от FMA и peak-knobs, на результат не влияет вовсе:
            // это аргумент cuLaunchKernel, а не опция компиляции, поэтому кэши модулей не
            // инвалидируются и перекомпиляции нет — новое значение действует со следующего Run.
            ImGui::Separator();
            ImGui::Text("GPU launch width");
            ImGui::TextDisabled("Threads per CUDA block. Affects speed only, never the numbers:");
            ImGui::TextDisabled("kernels are not recompiled, the new value applies on the next Run.");

            // Значения кратны варпу. Ширину сверх этого списка не предлагаем: за 256 занятость
            // упирается в регистры, а shared-бюджет режет блок обратно (см. launch_block_size).
            static const int kWidths[]      = { 32, 64, 96, 128, 192, 256 };
            static const char* kWidthNames[] = { "32 (default)", "64", "96", "128", "192", "256" };
            int widthIdx = 0;
            for (int i = 0; i < IM_ARRAYSIZE(kWidths); ++i)
                if (kWidths[i] == model.gpu_block_size) { widthIdx = i; break; }
            if (ImGui::Combo("Threads per block##gpu_block_size", &widthIdx, kWidthNames, IM_ARRAYSIZE(kWidthNames))) {
                model.gpu_block_size = clamp_gpu_block_size(kWidths[widthIdx]);
                set_gpu_block_size(model.gpu_block_size);
                persist_settings(model);
            }
            ImGui::TextDisabled("Measured on a 30-SM Turing card: on large grids every width is");
            ImGui::TextDisabled("within a few percent, while on small 1D sweeps a wide block");
            ImGui::TextDisabled("leaves most SMs idle and loses up to 2.5x. Occupancy here is");
            ImGui::TextDisabled("capped by registers, not by blocks per SM, so 32 is a fine");
            ImGui::TextDisabled("default -- try wider only on a different GPU.");
            ImGui::TextDisabled("Wide systems are capped back down to fit 48 KB of shared memory");
            ImGui::TextDisabled("per block, so the value here is a request, not a guarantee.");

            ImGui::Separator();
            ImGui::Text("Step controllers");
#ifdef UCUDA_AD_HAIRER_ONLY
            ImGui::TextDisabled("Your own adaptive step controllers (C bodies), shared by all systems");
            ImGui::TextDisabled("and tabs (library\\step_controllers.json).");
#else
            ImGui::TextDisabled("Your own adaptive step controllers: C bodies and named Soderlind");
            ImGui::TextDisabled("filters, shared by all systems and tabs (library\\step_controllers.json).");
#endif
            if (ImGui::Button("Step controller library...")) model.show_ctrl_library = true;
        }
        ImGui::End();
    }
    track_dock_selection(model);
    draw_ctrl_library_window(model);
}