#include "ucuda/application/app_model.h"
#include "ucuda/storage/system_library.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/domain/num_parse.h"
#include <algorithm>
static void basins_phase_tick(AppModel& model) {
    BasinsAnalysisSession& s = model.basins_session;
    s.ensure_phase_slots();
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    for (int i = 0; i < (int)s.configs.size(); ++i) {
        BasinsConfig&    c  = s.configs[(size_t)i];
        BasinsPhaseSlot& sl = *s.phase_slots[(size_t)i];
        const std::string sig = build_basins_phase_signature(c);
        if (sig != sl.pending_signature) {
            sl.pending_signature = sig;
            sl.pending_since     = now;
        }
        if (sig == sl.run_signature) continue;         // актуально
        if (!c.pp_autorun) continue;
        if (!c.last_run_ok || c.result.basin_idx.empty()) continue;
        if (sl.phase.in_flight) continue;
        if (now - sl.pending_since < 0.3) continue;   // ещё печатают
        s.request_phase_run(i);
    }
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

void AppModel::tick(SystemLibrary& lib) {
    poll();
    auto& model = *this;
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
    model.phase_session.wrapper_schemes       = model.enabled_wrapper_schemes();
    model.bifurcation_session.wrapper_schemes = model.enabled_wrapper_schemes();
    model.lle_session.wrapper_schemes         = model.enabled_wrapper_schemes();
    model.ls_session.wrapper_schemes          = model.enabled_wrapper_schemes();
    model.dft1d_session.wrapper_schemes       = model.enabled_wrapper_schemes();
    model.basins_session.wrapper_schemes      = model.enabled_wrapper_schemes();
    model.fastsync_session.wrapper_schemes    = model.enabled_wrapper_schemes();
    model.order_session.wrapper_schemes       = model.enabled_wrapper_schemes();
    model.network_session.wrapper_schemes     = model.enabled_wrapper_schemes();
    model.custom_session.wrapper_schemes                = model.enabled_wrapper_schemes();
    model.custom_session.bif_session.wrapper_schemes    = model.enabled_wrapper_schemes();
    model.custom_session.lle_session.wrapper_schemes    = model.enabled_wrapper_schemes();
    model.custom_session.ls_session.wrapper_schemes     = model.enabled_wrapper_schemes();
    model.custom_session.phase_session.wrapper_schemes  = model.enabled_wrapper_schemes();
    model.custom_session.basins_session.wrapper_schemes = model.enabled_wrapper_schemes();

    // Auto-labels: pre-frame refresh so all label-consumers (tab-bar names,
    // plot legend, window title, Plot windows section) see the same value.


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

}
