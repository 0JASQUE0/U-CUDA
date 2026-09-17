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

void AppModel::tick(SystemLibrary& lib) {
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
    // Custom tab owns 5 isolated sub-sessions — same per-frame sync applies.
    model.custom_session.custom_schemes                = model.custom_schemes;
    model.custom_session.bif_session.custom_schemes    = model.custom_schemes;
    model.custom_session.lle_session.custom_schemes    = model.custom_schemes;
    model.custom_session.ls_session.custom_schemes     = model.custom_schemes;
    model.custom_session.phase_session.custom_schemes  = model.custom_schemes;
    model.custom_session.basins_session.custom_schemes = model.custom_schemes;

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
    // сбросится и "Running" повиснет навсегда. Сессию на диск не пишем: своего
    // _last_order.json у вкладки пока нет.
    model.order_session.poll();
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
    // Custom tab has its own queue (2D → 1D → Phase/Basins pipeline).
    model.start_next_in_custom_queue();

}
