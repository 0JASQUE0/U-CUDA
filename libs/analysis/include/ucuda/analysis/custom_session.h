#pragma once
#include "ucuda/domain/custom_session.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/equations/adaptive_params.h"

// Сессия = сохраняемое состояние (CustomState, domain) + выполнение: future, отмена, прогресс,
// запуск и опрос. Сериализация (storage) видит только CustomState.
struct CustomSession : CustomState {
    // Owned sub-sessions (isolated from AppModel::bifurcation_session etc.).
    BifurcationAnalysisSession         bif_session;
    LLEAnalysisSession                 lle_session;
    LyapunovSpectrumAnalysisSession    ls_session;
    PhaseAnalysisSession               phase_session;
    // Фазовые портреты по бассейнам в Custom не предусмотрены — это решение, а не недоделка.
    // Панель «Phase portraits» и её окна живут только во вкладке Basins
    // (draw_basins_phase_controls / draw_basins_phase_windows / basins_phase_tick завязаны на
    // AppModel::basins_session). Поэтому здесь basins_session.phase_slots всегда пуст — поле общее
    // с типом, а не забытая инициализация.
    BasinsAnalysisSession              basins_session;
    // Non-copyable: sub-sessions hold std::future.
    CustomSession() = default;
    CustomSession(CustomSession&&) = default;
    CustomSession& operator=(CustomSession&&) = default;
    CustomSession(const CustomSession&) = delete;
    CustomSession& operator=(const CustomSession&) = delete;
    // Seed shared config from the current system record and populate each
    // sub-session with the fixed 3-slot layout (2D, 1D-X, 1D-Y for Bif/LLE/LS).
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    // Push queue items for enabled sub-tasks at this level. Idempotent — safe
    // to call from Run buttons or from auto-recompute.
    void enqueue_level_2d(std::deque<CustomQueueItem>& q) const;
    void enqueue_level_1d(std::deque<CustomQueueItem>& q) const;
    void enqueue_level_3 (std::deque<CustomQueueItem>& q) const;
    // Enqueue only the L1D slices affected by an axis change: `x_slices` pushes
    // Bif1D_X/LLE1D_X/LS1D_X (they depend on fix_y), `y_slices` pushes Bif1D_Y/LLE1D_Y/LS1D_Y (they
    // depend on fix_x). Used by the slider auto-recompute so an isolated fix_x drag doesn't re-run
    // (and re-autofit) the fix_x-independent X-slice.
    void enqueue_level_1d_partial(std::deque<CustomQueueItem>& q,
                                  bool x_slices, bool y_slices) const;
    // Aggregate in-flight status — true if ANY sub-session is currently
    // computing. Used to gate the queue drainer.
    bool any_in_flight() const;
    // Cancel every sub-session (Stop button in the top-bar).
    void request_cancel_all();
    // Aggregate poll — polls each sub-session. Returns true if ANY completed
    // this frame (caller writes _last_custom on that signal).
    bool poll_all();
    // Called by the drainer when the pipeline drains (queue empty + no
    // sub-session in flight). Promotes each level's `pending` signature
    // → `committed` and marks committed_ok from the sub-session's
    // last_run_* flags. Called from draw_gui after poll_all returns true.
    void commit_pending_signatures();
};

// Level-signature builders. Each returns a string that reflects EVERY input consumed by that
// level's Run — text of every numeric field, map serialisations, enable flags, sweep target
// indices. Two invocations with the same shared+cs return equal strings iff the downstream compute
// would land on identical data. Used by the Run dirty-tracking (skip level if the signature matches
// the last committed one).
std::string build_l2d_signature(const CustomTabSharedConfig& s, const CustomSession& cs);
std::string build_l1d_signature(const CustomTabSharedConfig& s, const CustomSession& cs);
std::string build_l3_signature (const CustomTabSharedConfig& s, const CustomSession& cs);

// Вспомогательные функции сессий (реализация в analysis; типы — в ucuda/domain).

// Shared -> sub-session field synchronisation
//
// Called exactly once immediately BEFORE run_async for each sub-item; never
// on every frame (would clobber per-type options the user edits in the
// detail panel — LS::display_exponent_idx, LLE::eps, etc.).
//
// dir semantics for 1D variants:
//   0 = X-slice (sweep along shared X, Y fixed at fix_y_value)
//   1 = Y-slice (sweep along shared Y, X fixed at fix_x_value)
void apply_shared_to_bif2d(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c);
void apply_shared_to_bif1d(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c, int dir);
void apply_shared_to_lle2d(const CustomTabSharedConfig& s, LLECurveConfig& c);
// c2d — слот [0] (2D-конфиг) той же подсессии. Специфика LLE/LS (eps, NT) живёт только там и
// редактируется в панели Level 2D, поэтому срез обязан забирать её оттуда — ровно как
// bif_writable_var. Без этого слоты [1]/[2] считались с тем, что им досталось от add_curve при
// загрузке системы, и правка «eps» / «NT» действовала на карту, но не на срез.
void apply_shared_to_lle1d(const CustomTabSharedConfig& s, const LLECurveConfig& c2d,
                           LLECurveConfig& c, int dir);
void apply_shared_to_ls2d (const CustomTabSharedConfig& s, LSCurveConfig&  c);
void apply_shared_to_ls1d (const CustomTabSharedConfig& s, const LSCurveConfig& c2d,
                           LSCurveConfig&  c, int dir);   // c2d — см. apply_shared_to_lle1d
void apply_shared_to_phase (const CustomTabSharedConfig& s, PhaseAnalysisState& ph, const std::vector<std::string>& vars);
void apply_shared_to_basins(const CustomTabSharedConfig& s, BasinsConfig& c);

EffectiveSweep effective_sweep_x(const CustomTabSharedConfig& s);
EffectiveSweep effective_sweep_y(const CustomTabSharedConfig& s);
