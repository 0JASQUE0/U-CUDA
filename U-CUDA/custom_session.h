#pragma once
#include "analysis_session.h"
#include "system_record.h"
#include "parametric_engine.h"
#include <deque>
#include <string>
#include <vector>
#include <map>

// Custom tab — hierarchical pipeline that stitches together five existing analysis sub-sessions
// with a shared configuration on top.
//
// Levels (each independently toggleable via enable-checkbox, so skipping any level works out of
// the box):
//   Level 2D: Bif2D + LLE2D + LS2D on a shared axis_x/axis_y/N grid.
//   Level 1D: up to 6 slices — X- and Y-direction for each of Bif/LLE/LS — cutting through the
//             fix_x/fix_y point inside the 2D range.
//   Level 3:  Phase portrait + Time-domain OR Basins (radio choice).
//
// Sub-sessions are OWNED by CustomSession (not shared with Parametric/Analysis/Basins tabs), so
// switching AppMode tabs never cross-contaminates state.
//
// Per-type sub-session slot layout (fixed to keep run indices stable):
//   diagrams[0]/curves[0]/configs[0] = 2D config  (mode_2d = true)
//   diagrams[1]/curves[1]/configs[1] = 1D X-slice (mode_2d = false, dir=X)
//   diagrams[2]/curves[2]/configs[2] = 1D Y-slice (mode_2d = false, dir=Y)
// (LS/LLE follow the same pattern via add_curve; Basins is single-config.)

struct CustomTabSharedConfig {
    // Level-enable flags
    bool level_2d_enabled    = true;
    bool level_1d_enabled    = true;
    bool level_phase_enabled = true;

    // Integrator group (shared across all levels)
    std::string scheme           = "Euler";
    std::string symmetry_s       = "0.5";
    std::string h_text           = "0.01";
    std::string t_max_text       = "100";
    std::string transient_text   = "100";
    std::string pre_scaller_text = "1";
    std::string max_value_text   = "1e6";

    // ---- Initial conditions + RHS parameters (both maps mirror
    // BifurcationDiagramConfig::initial_conditions / param_values) ----
    std::map<std::string, std::string> initial_conditions;
    std::map<std::string, std::string> param_values;

    // Level 2D: shared sweep for all three types
    int         axis_x_par_index    = 0;
    bool        axis_x_over_var     = false;
    int         axis_x_var_index    = 0;
    // Свип по шагу интегрирования вместо параметра/НУ. Взаимоисключающе с
    // over_var, и ровно ОДНА из осей может быть h: 2D-ядра принимают один
    // hSweepAxis (см. sweep_over_h_2 в analysis_session.h). UI это соблюдает.
    bool        axis_x_over_h       = false;
    std::string axis_x_lo_text      = "0";
    std::string axis_x_hi_text      = "1";
    // Лог-сетка по оси (любой sweep target — param/IC/h), см.
    // BifurcationDiagramConfig::log_scale. Требует lo>0 и hi>0 — иначе
    // движок отказывает на Run. Это свойство ОСИ, поэтому 1D-срез вдоль
    // той же оси наследует его вместе с par/lo/hi (см. EffectiveSweep).
    bool        axis_x_log          = false;
    int         axis_y_par_index    = 1;
    bool        axis_y_over_var     = false;
    int         axis_y_var_index    = 0;
    bool        axis_y_over_h       = false;
    std::string axis_y_lo_text      = "0";
    std::string axis_y_hi_text      = "1";
    bool        axis_y_log          = false;   // см. axis_x_log
    // Shared 2D grid resolution — the underlying NVRTC kernels
    // (`getValueByIdx` etc.) require a square N×N grid, so one field drives
    // both axes. Matches Analysis-tab's single "Resolution" field.
    std::string resolution_text     = "64";
    bool        bif2d_enabled       = true;
    bool        lle2d_enabled       = true;
    bool        ls2d_enabled        = true;

    // Переменная, по которой строится БД (BifurcationDiagramConfig::writable_var). Живёт в
    // Level 2D и оттуда наследуется ОБОИМИ 1D-срезами: срез — это разрез той же карты, строить его
    // по другой переменной бессмысленно. У LLE/LS своей writable_var нет (λ — скаляр на точку).
    // -1 = комбинация x[0] + pi*x[1] + e*x[2] (draw_writable_var_combo). Работает и в 1D, и в 2D:
    // обе ветки идут через один loopCalculateDiscreteModel_int, где сентинел обработан явно.
    int         bif_writable_var    = 0;

    // Per-type options that don't fit into the sub-session config directly
    // (options that DO live in sub-session config, like DBSCAN eps, are left
    // there and edited in the L2D detail panel).

    // Level 1D: 6 independent slices + inheritance flag
    bool inherit_sweep_from_2d = true;
    // Own X-direction sweep (used when inherit_sweep_from_2d == false, OR
    // when Level 2D is disabled).
    int         sweep_x_par_index  = 0;
    bool        sweep_x_over_var   = false;
    int         sweep_x_var_index  = 0;
    bool        sweep_x_over_h     = false;   // см. axis_x_over_h
    std::string sweep_x_lo_text    = "0";
    std::string sweep_x_hi_text    = "1";
    bool        sweep_x_log        = false;   // см. axis_x_log
    std::string n_x_1d_text        = "500";
    // Own Y-direction sweep.
    int         sweep_y_par_index  = 1;
    bool        sweep_y_over_var   = false;
    int         sweep_y_var_index  = 0;
    bool        sweep_y_over_h     = false;   // см. axis_x_over_h
    std::string sweep_y_lo_text    = "0";
    std::string sweep_y_hi_text    = "1";
    bool        sweep_y_log        = false;   // см. axis_x_log
    std::string n_y_1d_text        = "500";

    // Per-L1D integrator overrides. L1D is cheap and interactive, so users often want finer h /
    // longer TT / shorter CT than L2D (or Phase). Seeded from shared defaults on load_from_record
    // and edited independently in the L1D detail panel regardless of `inherit_sweep_from_2d` — that
    // flag only controls the sweep axis (par/lo/hi), not integrator params.
    std::string l1d_h_text          = "0.01";
    std::string l1d_transient_text  = "100";
    std::string l1d_t_max_text      = "100";

    // 6 per-type-per-direction enable flags.
    bool bif1d_x_enabled = true;
    bool bif1d_y_enabled = false;
    bool lle1d_x_enabled = false;
    bool lle1d_y_enabled = false;
    bool ls1d_x_enabled  = false;
    bool ls1d_y_enabled  = false;

    // Continuation for all 1D slices (matches BifurcationDiagramConfig).
    bool continuation_1d_enabled = false;

    // По Y на Bif-срезах: false — значения пиков, true — межпиковые интервалы
    // (BifurcationDiagramConfig::plot_inter_peaks). Чисто отображение: оба массива приходят из
    // ОДНОГО прогона, пересчёт не нужен, поэтому поле не входит в l1d-сигнатуру и пишется в слоты
    // сразу при клике, а не на Run. В 2D аналога нет (там период через DBSCAN).
    bool plot_inter_peaks_1d = false;

    // Slice position — clamped to current effective sweep ranges.
    double fix_x_value = 0.0;
    double fix_y_value = 0.0;

    // Auto-recompute (Level 1D only — cheap and interactive). Level 2D is
    // manual on purpose (expensive N_x*N_y). Phase auto-recompute is on the
    // sub-session itself (PhaseAnalysisSession::auto_recompute).
    bool auto_recompute_1d = true;

    // Debounce state for slider drags: (re)start the 1D recompute only after the slider settles for
    // ~200 ms. Split per axis so we re-run ONLY the slice whose data actually depends on the moved
    // axis — the X-slice sweeps X and pins Y at fix_y (so it depends on fix_y), the Y-slice is the
    // mirror. Without the split a fix_x drag re-ran both slices, and the X-slice recompute —
    // data-identical to the previous run — fired an autofit that clobbered the user's manual zoom.
    // Heatmap drags bump BOTH.
    double last_fix_x_change_time = 0.0;
    double last_fix_y_change_time = 0.0;

    // Level 3: Phase / Basins selector + drill-down policy
    int  level3_kind             = 0;      // 0 = Phase, 1 = Basins
    bool autorun_on_drilldown    = true;
};

// One item in the Custom pipeline queue — mirrors the per-mode QueueItem
// structs in app_model.h (Bif/LLE/LS/Basins each have their own kind of item).
struct CustomQueueItem {
    enum class Kind {
        Bif2D, LLE2D, LS2D,
        Bif1D_X, Bif1D_Y,
        LLE1D_X, LLE1D_Y,
        LS1D_X, LS1D_Y,
        Phase, Basins
    };
    Kind kind = Kind::Bif2D;
};

// Workspace tabs (Custom-mode UI layout container)
//
// Each tab hosts its own DockSpace. Plot windows dock into per-tab spaces
// via ImGui's DockBuilder API (see draw_custom_mode_layout in gui.cpp);
// switching between tabs shows/hides the corresponding set of docked
// windows, without orphaning them (KeepAliveOnly submit for inactive tabs).
//
// Only metadata lives here — the actual dock layout inside each tab is
// owned by ImGui and persisted in imgui.ini keyed off the tab's stable id.
struct WorkspaceTab {
    int         id   = 0;   // Stable per session — used to derive DockSpace ID.
    std::string name;       // User-editable label shown in the tab bar.
};

struct CustomWorkspace {
    std::vector<WorkspaceTab> tabs;                // Always non-empty (ensure_default).
    int                       active_tab_id = 1;   // Currently visible tab.
    int                       next_tab_id   = 2;   // Monotonic id allocator.
    float                     controls_width = 380.0f; // Splitter position (px).

    // Transient (not serialised) — set by the workspace UI when any persistable field changes
    // (add/close/rename tab, splitter drag, cross-tab window move, active tab switch). Consumed once
    // per frame by draw_gui to write _last_custom.json; without it the workspace metadata was saved
    // only on a sub-session compute completion, so e.g. closing a tab wouldn't survive a restart.
    bool                      dirty = false;

    void ensure_default() {
        if (tabs.empty()) {
            tabs.push_back({1, "Tab 1"});
            active_tab_id = 1;
            next_tab_id   = 2;
            dirty         = true;
        }
    }
};

// Coordinator that owns per-type sub-sessions + the shared config. Kept in
// AppModel as a value member. Non-copyable (sub-sessions hold std::future).
struct CustomSession {
    // Identity — mirrors what other sessions carry, kept in sync with the
    // owning AppModel after start_custom_analysis().
    std::vector<std::string> vars;
    std::vector<std::string> params;
    System                   sys;
    std::vector<CustomScheme> custom_schemes;
    std::vector<std::string> enabled_builtin_schemes;
    std::string              loaded_system_name;

    CustomTabSharedConfig shared;

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

    // Layout generation for the "Reset windows layout" button (Custom-tab
    // plot windows include this into their ImGui-ID so docking treats them
    // as new after a reset). Mirrors PhaseAnalysisSession::layout_generation.
    int layout_generation = 0;

    // Workspace tabs — visible only in Custom mode.
    CustomWorkspace workspace;

    // Cached last-error string for GUI status; sub-session errors are still
    // rendered from their own last_error fields.
    std::string last_error;

    // Dirty-tracking for the Run button. Every level owns a "committed" signature — a string built
    // from every input the level cares about (integrator + IC + params + sweep axis + resolution +
    // enable flags + level-specific bits like eps/exponent). Enqueueing the level bumps its pending
    // signature; a successful sub-session completion promotes pending → committed. Run skips a level
    // when its current build matches committed and the last commit was OK — so hitting Run after
    // changing only N_x1d recomputes only L1D. Per-level Run buttons (if added later) bypass this.
    struct LevelSig {
        std::string committed;
        std::string pending;
        bool committed_ok = false;
        bool pending_armed = false;
    };
    LevelSig sig_l2d, sig_l1d, sig_l3;

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
void apply_shared_to_phase (const CustomTabSharedConfig& s, PhaseAnalysisSession& ph, const std::vector<std::string>& vars);
void apply_shared_to_basins(const CustomTabSharedConfig& s, BasinsConfig& c);

// Effective 1D sweep ranges — takes shared 2D fields when inherit is on and
// L2D is enabled, otherwise takes L1D's own sweep_* fields.
struct EffectiveSweep {
    int         par_index;
    bool        over_var;
    int         var_index;
    bool        over_h;      // свип по шагу; взаимоисключающе с over_var
    bool        log_scale;   // лог-сетка по этой оси (см. axis_x_log)
    std::string lo_text;
    std::string hi_text;
    std::string n_pts_text;
};
EffectiveSweep effective_sweep_x(const CustomTabSharedConfig& s);
EffectiveSweep effective_sweep_y(const CustomTabSharedConfig& s);

// Level-signature builders. Each returns a string that reflects EVERY input consumed by that
// level's Run — text of every numeric field, map serialisations, enable flags, sweep target
// indices. Two invocations with the same shared+cs return equal strings iff the downstream compute
// would land on identical data. Used by the Run dirty-tracking (skip level if the signature matches
// the last committed one).
std::string build_l2d_signature(const CustomTabSharedConfig& s, const CustomSession& cs);
std::string build_l1d_signature(const CustomTabSharedConfig& s, const CustomSession& cs);
std::string build_l3_signature (const CustomTabSharedConfig& s, const CustomSession& cs);
