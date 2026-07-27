#include "custom_session.h"
#include <algorithm>
#include <cstdio>

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Format a double as text with enough precision to survive round-tripping
// through the string-based InputText fields the rest of the codebase uses.
std::string fmt_d(double v) {
    char buf[64];
    // %.6g strips float→double round-trip noise from SliderFloat inputs.
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

// Copy the shared integrator/IC/params block into any per-run config that
// holds the same-named string fields. Templated to avoid repeating six times.
template <typename Cfg>
void copy_integrator_and_state(const CustomTabSharedConfig& s, Cfg& c) {
    c.scheme           = s.scheme;
    c.symmetry_s       = s.symmetry_s;
    c.h_text           = s.h_text;
    c.t_max_text       = s.t_max_text;
    c.transient_text   = s.transient_text;
    c.max_value_text   = s.max_value_text;
    // Not every config has pre_scaller_text (LLE/LS don't); apply where it
    // exists via a separate overload set below.
    c.initial_conditions = s.initial_conditions;
    c.param_values       = s.param_values;
}

// Overload to also carry pre_scaller_text for configs that have it.
void copy_prescaller(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c) { c.pre_scaller_text = s.pre_scaller_text; }
void copy_prescaller(const CustomTabSharedConfig& s, BasinsConfig&           c) { c.pre_scaller_text = s.pre_scaller_text; }
void copy_prescaller(const CustomTabSharedConfig&,   LLECurveConfig&) { /* no such field */ }
void copy_prescaller(const CustomTabSharedConfig&,   LSCurveConfig&)  { /* no such field */ }

} // namespace

// ============================================================================
// Effective sweep ranges (respects inherit_sweep_from_2d + level_2d_enabled)
// ============================================================================

EffectiveSweep effective_sweep_x(const CustomTabSharedConfig& s) {
    EffectiveSweep e;
    if (s.inherit_sweep_from_2d && s.level_2d_enabled) {
        // Inherit: sweep AXIS (par/lo/hi) comes from L2D. Resolution
        // stays L1D's own — L1D is cheap and interactive, users often
        // want higher N there than the expensive N_x·N_y L2D grid.
        e.par_index  = s.axis_x_par_index;
        e.over_var   = s.axis_x_over_var;
        e.var_index  = s.axis_x_var_index;
        e.over_h     = s.axis_x_over_h;
        e.lo_text    = s.axis_x_lo_text;
        e.hi_text    = s.axis_x_hi_text;
    } else {
        e.par_index  = s.sweep_x_par_index;
        e.over_var   = s.sweep_x_over_var;
        e.var_index  = s.sweep_x_var_index;
        e.over_h     = s.sweep_x_over_h;
        e.lo_text    = s.sweep_x_lo_text;
        e.hi_text    = s.sweep_x_hi_text;
    }
    e.n_pts_text = s.n_x_1d_text;
    return e;
}

EffectiveSweep effective_sweep_y(const CustomTabSharedConfig& s) {
    EffectiveSweep e;
    if (s.inherit_sweep_from_2d && s.level_2d_enabled) {
        e.par_index  = s.axis_y_par_index;
        e.over_var   = s.axis_y_over_var;
        e.var_index  = s.axis_y_var_index;
        e.over_h     = s.axis_y_over_h;
        e.lo_text    = s.axis_y_lo_text;
        e.hi_text    = s.axis_y_hi_text;
    } else {
        e.par_index  = s.sweep_y_par_index;
        e.over_var   = s.sweep_y_over_var;
        e.var_index  = s.sweep_y_var_index;
        e.over_h     = s.sweep_y_over_h;
        e.lo_text    = s.sweep_y_lo_text;
        e.hi_text    = s.sweep_y_hi_text;
    }
    e.n_pts_text = s.n_y_1d_text;
    return e;
}

// ============================================================================
// apply_shared_to_* — one function per (type, mode) combination.
// ============================================================================

void apply_shared_to_bif2d(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c) {
    copy_integrator_and_state(s, c);
    copy_prescaller(s, c);
    c.mode_2d           = true;
    c.colored_1d        = false;
    c.param_index       = s.axis_x_par_index;
    c.sweep_over_var    = s.axis_x_over_var;
    c.var_sweep_index   = s.axis_x_var_index;
    c.sweep_over_h      = s.axis_x_over_h;
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.resolution_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.sweep_over_h_2    = s.axis_y_over_h;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
    // eps_dbscan_text stays on the sub-config (edited in the L2D detail panel).
}

void apply_shared_to_bif1d(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    copy_prescaller(s, c);
    // L1D-specific integrator overrides (finer h / different TT/CT than L2D).
    c.h_text         = s.l1d_h_text;
    c.transient_text = s.l1d_transient_text;
    c.t_max_text     = s.l1d_t_max_text;
    c.mode_2d      = false;
    c.colored_1d   = false;
    c.continuation = s.continuation_1d_enabled;

    EffectiveSweep sweep_x = effective_sweep_x(s);
    EffectiveSweep sweep_y = effective_sweep_y(s);

    if (dir == 0) {
        // X-slice: sweep along X, fix Y at fix_y_value.
        c.param_index     = sweep_x.par_index;
        c.sweep_over_var  = sweep_x.over_var;
        c.var_sweep_index = sweep_x.var_index;
        c.sweep_over_h    = sweep_x.over_h;
        c.param_lo_text   = sweep_x.lo_text;
        c.param_hi_text   = sweep_x.hi_text;
        c.n_pts_text      = sweep_x.n_pts_text;
        if (!sweep_y.over_var && sweep_y.par_index >= 0 && sweep_y.par_index < (int)s.param_values.size()) {
            // Overwrite the fixed-axis param value in the per-config copy.
            // (Uses the parameter NAME from shared.param_values ordered by
            // params[] — caller supplies params via CustomSession::params.)
        }
        // Pin the fixed axis value via param_values below (see apply-time
        // wrapping in run_next_in_custom_queue where param name is known).
    } else {
        // Y-slice: sweep along Y, fix X at fix_x_value.
        c.param_index     = sweep_y.par_index;
        c.sweep_over_var  = sweep_y.over_var;
        c.var_sweep_index = sweep_y.var_index;
        c.sweep_over_h    = sweep_y.over_h;
        c.param_lo_text   = sweep_y.lo_text;
        c.param_hi_text   = sweep_y.hi_text;
        c.n_pts_text      = sweep_y.n_pts_text;
    }
}

void apply_shared_to_lle2d(const CustomTabSharedConfig& s, LLECurveConfig& c) {
    copy_integrator_and_state(s, c);
    c.mode_2d           = true;
    c.param_index       = s.axis_x_par_index;
    c.sweep_over_var    = s.axis_x_over_var;
    c.var_sweep_index   = s.axis_x_var_index;
    c.sweep_over_h      = s.axis_x_over_h;
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.resolution_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.sweep_over_h_2    = s.axis_y_over_h;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
    // eps_text / nt_text stay on the sub-config.
}

void apply_shared_to_lle1d(const CustomTabSharedConfig& s, LLECurveConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    c.h_text         = s.l1d_h_text;
    c.transient_text = s.l1d_transient_text;
    c.t_max_text     = s.l1d_t_max_text;
    c.mode_2d = false;
    EffectiveSweep sweep = (dir == 0) ? effective_sweep_x(s) : effective_sweep_y(s);
    c.param_index     = sweep.par_index;
    c.sweep_over_var  = sweep.over_var;
    c.var_sweep_index = sweep.var_index;
    c.sweep_over_h    = sweep.over_h;
    c.param_lo_text   = sweep.lo_text;
    c.param_hi_text   = sweep.hi_text;
    c.n_pts_text      = sweep.n_pts_text;
}

void apply_shared_to_ls2d(const CustomTabSharedConfig& s, LSCurveConfig& c) {
    copy_integrator_and_state(s, c);
    c.mode_2d           = true;
    c.param_index       = s.axis_x_par_index;
    c.sweep_over_var    = s.axis_x_over_var;
    c.var_sweep_index   = s.axis_x_var_index;
    c.sweep_over_h      = s.axis_x_over_h;
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.resolution_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.sweep_over_h_2    = s.axis_y_over_h;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
}

void apply_shared_to_ls1d(const CustomTabSharedConfig& s, LSCurveConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    c.h_text         = s.l1d_h_text;
    c.transient_text = s.l1d_transient_text;
    c.t_max_text     = s.l1d_t_max_text;
    c.mode_2d = false;
    EffectiveSweep sweep = (dir == 0) ? effective_sweep_x(s) : effective_sweep_y(s);
    c.param_index     = sweep.par_index;
    c.sweep_over_var  = sweep.over_var;
    c.var_sweep_index = sweep.var_index;
    c.sweep_over_h    = sweep.over_h;
    c.param_lo_text   = sweep.lo_text;
    c.param_hi_text   = sweep.hi_text;
    c.n_pts_text      = sweep.n_pts_text;
}

void apply_shared_to_phase(const CustomTabSharedConfig& s, PhaseAnalysisSession& ph,
                           const std::vector<std::string>& /*vars*/) {
    // Shared config drives the integrator + parameter values on Run.
    // IC-sets are owned by PhaseAnalysisSession and edited via the L3
    // Phase controls panel (parity with Analysis tab), so the shared
    // "Initial conditions" block affects only BD/LLE/LS/Basins — not
    // phase, which typically wants several ICs the shared field can't
    // express. Ensure at least one slot exists so recompute_async has
    // something to run.
    ph.scheme     = s.scheme;
    ph.symmetry_s = s.symmetry_s;
    ph.step_h     = s.h_text;
    ph.sim_time   = s.t_max_text;
    ph.skip_time  = s.transient_text;
    ph.param_values = s.param_values;
    if (ph.ic_sets.empty()) {
        InitialConditionSet ic;
        ic.label = "IC 1";
        ph.ic_sets.push_back(std::move(ic));
    }
}

void apply_shared_to_basins(const CustomTabSharedConfig& s, BasinsConfig& c) {
    copy_integrator_and_state(s, c);
    copy_prescaller(s, c);
    // axis_x_var/axis_y_var are IC-space, edited in the Basins detail panel.
    // Shared param_values (with fix_x/fix_y overrides for the L2D axes)
    // are applied by the queue driver, which knows the parameter names.
}

// ============================================================================
// CustomSession members
// ============================================================================

void CustomSession::load_from_record(const SystemRecord& r,
                                     const std::vector<std::string>& vars_,
                                     const std::vector<std::string>& params_) {
    vars   = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;

    // Hard-reset shared to struct defaults BEFORE seeding from the record.
    // Previously we only overrode a subset of fields (scheme, symmetry, h,
    // ICs, param_values, default sweep indices) — everything else (fix_x/y,
    // sweep ranges, resolution, level/sub-type enables, inherit flag, log
    // scales, level3_kind, etc.) leaked from the PREVIOUS system when the
    // new system had no saved _last_custom.json. That fed the wrong
    // fix_x/fix_y and axis targets straight into the kernel via
    // apply_shared_to_bif2d/pin_param and made the second-system Run look
    // "similar but wrong". Now defaults come from the struct itself and the
    // explicit assignments below override just what the record specifies.
    shared = CustomTabSharedConfig{};

    // Seed shared config from the record's defaults.
    shared.scheme         = "Euler";
    shared.symmetry_s     = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    shared.h_text         = r.step_h.empty() ? std::string("0.01") : r.step_h;
    // Seed L1D integrator overrides from the same shared defaults; user
    // can drift them apart later in the L1D detail panel.
    shared.l1d_h_text         = shared.h_text;
    shared.l1d_transient_text = shared.transient_text;
    shared.l1d_t_max_text     = shared.t_max_text;
    shared.initial_conditions.clear();
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        shared.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : std::string();
    }
    shared.param_values.clear();
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        shared.param_values[p] = (it != r.param_values.end()) ? it->second : std::string();
    }

    // Choose sensible default sweep axes (params[0]/params[1] if available).
    if (params.size() >= 1) shared.axis_x_par_index = 0;
    if (params.size() >= 2) shared.axis_y_par_index = 1;
    shared.sweep_x_par_index = shared.axis_x_par_index;
    shared.sweep_y_par_index = shared.axis_y_par_index;

    // Default fix_x/y to the MIDPOINT of the effective sweep range instead
    // of leaving them at struct-default 0.0. On a fresh (never-visited)
    // system, 0.0 on the pinned axis pushed many systems into degenerate
    // trajectories, and 1D-Run appeared to "produce nothing" until the user
    // dragged the fix slider off zero. Midpoint of default 0..1 is 0.5 —
    // a much better neutral starting point.
    auto safe_parse = [](const std::string& s, double def) -> double {
        if (s.empty()) return def;
        try { return std::stod(s); } catch (...) { return def; }
    };
    {
        EffectiveSweep esx = effective_sweep_x(shared);
        EffectiveSweep esy = effective_sweep_y(shared);
        shared.fix_x_value = (safe_parse(esx.lo_text, 0.0) + safe_parse(esx.hi_text, 1.0)) * 0.5;
        shared.fix_y_value = (safe_parse(esy.lo_text, 0.0) + safe_parse(esy.hi_text, 1.0)) * 0.5;
    }

    // Seed sub-sessions. Each Bif/LLE/LS gets 3 slots: [0]=2D, [1]=1D-X, [2]=1D-Y.
    bif_session.load_from_record(r, vars, params);
    bif_session.add_diagram();  // slot 1
    bif_session.add_diagram();  // slot 2
    if (bif_session.diagrams.size() >= 3) {
        bif_session.diagrams[0].label = "Custom 2D";        bif_session.diagrams[0].label_is_manual = true;
        bif_session.diagrams[1].label = "Custom 1D X";      bif_session.diagrams[1].label_is_manual = true;
        bif_session.diagrams[2].label = "Custom 1D Y";      bif_session.diagrams[2].label_is_manual = true;
    }

    lle_session.load_from_record(r, vars, params);
    lle_session.add_curve();
    lle_session.add_curve();
    if (lle_session.curves.size() >= 3) {
        lle_session.curves[0].label = "Custom LLE 2D";     lle_session.curves[0].label_is_manual = true;
        lle_session.curves[1].label = "Custom LLE 1D X";   lle_session.curves[1].label_is_manual = true;
        lle_session.curves[2].label = "Custom LLE 1D Y";   lle_session.curves[2].label_is_manual = true;
    }

    ls_session.load_from_record(r, vars, params);
    ls_session.add_curve();
    ls_session.add_curve();
    if (ls_session.curves.size() >= 3) {
        ls_session.curves[0].label = "Custom LS 2D";       ls_session.curves[0].label_is_manual = true;
        ls_session.curves[1].label = "Custom LS 1D X";     ls_session.curves[1].label_is_manual = true;
        ls_session.curves[2].label = "Custom LS 1D Y";     ls_session.curves[2].label_is_manual = true;
    }

    phase_session.load_from_record(r, vars, params);
    basins_session.load_from_record(r, vars, params);

    loaded_system_name = r.name;
    last_error.clear();
}

void CustomSession::enqueue_level_2d(std::deque<CustomQueueItem>& q) const {
    if (!shared.level_2d_enabled) return;
    if (shared.bif2d_enabled) q.push_back({ CustomQueueItem::Kind::Bif2D });
    if (shared.lle2d_enabled) q.push_back({ CustomQueueItem::Kind::LLE2D });
    if (shared.ls2d_enabled ) q.push_back({ CustomQueueItem::Kind::LS2D  });
}

void CustomSession::enqueue_level_1d(std::deque<CustomQueueItem>& q) const {
    if (!shared.level_1d_enabled) return;
    if (shared.bif1d_x_enabled) q.push_back({ CustomQueueItem::Kind::Bif1D_X });
    if (shared.bif1d_y_enabled) q.push_back({ CustomQueueItem::Kind::Bif1D_Y });
    if (shared.lle1d_x_enabled) q.push_back({ CustomQueueItem::Kind::LLE1D_X });
    if (shared.lle1d_y_enabled) q.push_back({ CustomQueueItem::Kind::LLE1D_Y });
    if (shared.ls1d_x_enabled)  q.push_back({ CustomQueueItem::Kind::LS1D_X  });
    if (shared.ls1d_y_enabled)  q.push_back({ CustomQueueItem::Kind::LS1D_Y  });
}

void CustomSession::enqueue_level_1d_partial(std::deque<CustomQueueItem>& q,
                                             bool x_slices, bool y_slices) const {
    if (!shared.level_1d_enabled) return;
    if (x_slices) {
        if (shared.bif1d_x_enabled) q.push_back({ CustomQueueItem::Kind::Bif1D_X });
        if (shared.lle1d_x_enabled) q.push_back({ CustomQueueItem::Kind::LLE1D_X });
        if (shared.ls1d_x_enabled)  q.push_back({ CustomQueueItem::Kind::LS1D_X  });
    }
    if (y_slices) {
        if (shared.bif1d_y_enabled) q.push_back({ CustomQueueItem::Kind::Bif1D_Y });
        if (shared.lle1d_y_enabled) q.push_back({ CustomQueueItem::Kind::LLE1D_Y });
        if (shared.ls1d_y_enabled)  q.push_back({ CustomQueueItem::Kind::LS1D_Y  });
    }
}

void CustomSession::enqueue_level_3(std::deque<CustomQueueItem>& q) const {
    if (!shared.level_phase_enabled) return;
    q.push_back({ shared.level3_kind == 0 ? CustomQueueItem::Kind::Phase
                                          : CustomQueueItem::Kind::Basins });
}

bool CustomSession::any_in_flight() const {
    return bif_session.in_flight
        || lle_session.in_flight
        || ls_session.in_flight
        || phase_session.in_flight
        || basins_session.in_flight;
}

void CustomSession::request_cancel_all() {
    bif_session.request_cancel();
    lle_session.request_cancel();
    ls_session.request_cancel();
    // PhaseAnalysisSession has no cancel_token in the current codebase — poll
    // will still finish when the worker returns; nothing else to do here.
    basins_session.request_cancel();
}

bool CustomSession::poll_all() {
    bool any = false;
    if (bif_session.poll())     any = true;
    if (lle_session.poll())     any = true;
    if (ls_session.poll())      any = true;
    if (phase_session.poll())   any = true;
    if (basins_session.poll())  any = true;
    return any;
}

// ============================================================================
// Level signatures — string blobs summarising every input a level consumes.
// Two identical strings ⇒ downstream compute lands on identical data ⇒ the
// Run drainer can skip the level.
// ============================================================================
namespace {
    // Append map<string, string> sorted by key for deterministic output.
    void sig_append_map(std::string& o, const char* tag,
                        const std::map<std::string, std::string>& m) {
        o += tag; o += "{";
        for (const auto& kv : m) { o += kv.first; o += "="; o += kv.second; o += ";"; }
        o += "}";
    }
    void sig_append_int(std::string& o, const char* tag, int v) {
        o += tag; o += "="; o += std::to_string(v); o += ";";
    }
    void sig_append_bool(std::string& o, const char* tag, bool v) {
        o += tag; o += "="; o += (v ? "1" : "0"); o += ";";
    }
    void sig_append_str(std::string& o, const char* tag, const std::string& v) {
        o += tag; o += "="; o += v; o += ";";
    }
    void sig_append_double(std::string& o, const char* tag, double v) {
        char b[64]; std::snprintf(b, sizeof(b), "%.17g", v);
        o += tag; o += "="; o += b; o += ";";
    }
    // Shared integrator + IC + params — feed all three levels.
    void sig_append_shared_state(std::string& o, const CustomTabSharedConfig& s) {
        sig_append_str(o, "sc", s.scheme);
        sig_append_str(o, "sym", s.symmetry_s);
        sig_append_str(o, "h",   s.h_text);
        sig_append_str(o, "tt",  s.transient_text);
        sig_append_str(o, "ct",  s.t_max_text);
        sig_append_str(o, "dec", s.pre_scaller_text);
        sig_append_str(o, "mv",  s.max_value_text);
        sig_append_map(o, "ic", s.initial_conditions);
        sig_append_map(o, "pv", s.param_values);
    }
}

std::string build_l2d_signature(const CustomTabSharedConfig& s, const CustomSession& cs) {
    std::string o; o.reserve(512);
    sig_append_bool(o, "en", s.level_2d_enabled);
    sig_append_shared_state(o, s);
    sig_append_int (o, "axp",  s.axis_x_par_index);
    sig_append_bool(o, "axv",  s.axis_x_over_var);
    sig_append_bool(o, "axh",  s.axis_x_over_h);
    sig_append_int (o, "axvi", s.axis_x_var_index);
    sig_append_str (o, "axlo", s.axis_x_lo_text);
    sig_append_str (o, "axhi", s.axis_x_hi_text);
    sig_append_int (o, "ayp",  s.axis_y_par_index);
    sig_append_bool(o, "ayv",  s.axis_y_over_var);
    sig_append_bool(o, "ayh",  s.axis_y_over_h);
    sig_append_int (o, "ayvi", s.axis_y_var_index);
    sig_append_str (o, "aylo", s.axis_y_lo_text);
    sig_append_str (o, "ayhi", s.axis_y_hi_text);
    sig_append_str (o, "res",  s.resolution_text);
    sig_append_bool(o, "b2",   s.bif2d_enabled);
    sig_append_bool(o, "l2",   s.lle2d_enabled);
    sig_append_bool(o, "s2",   s.ls2d_enabled);
    if (!cs.bif_session.diagrams.empty())
        sig_append_str(o, "beps", cs.bif_session.diagrams[0].eps_dbscan_text);
    if (!cs.lle_session.curves.empty()) {
        sig_append_str(o, "leps", cs.lle_session.curves[0].eps_text);
        sig_append_str(o, "lnt",  cs.lle_session.curves[0].nt_text);
    }
    if (!cs.ls_session.curves.empty()) {
        sig_append_str(o, "seps", cs.ls_session.curves[0].eps_text);
        sig_append_str(o, "snt",  cs.ls_session.curves[0].nt_text);
    }
    return o;
}

std::string build_l1d_signature(const CustomTabSharedConfig& s, const CustomSession& cs) {
    (void)cs;
    std::string o; o.reserve(512);
    sig_append_bool(o, "en", s.level_1d_enabled);
    sig_append_shared_state(o, s);
    // L1D integrator overrides.
    sig_append_str(o, "lh",  s.l1d_h_text);
    sig_append_str(o, "ltt", s.l1d_transient_text);
    sig_append_str(o, "lct", s.l1d_t_max_text);
    // Effective sweep — resolves inherit vs own automatically.
    EffectiveSweep sx = effective_sweep_x(s);
    EffectiveSweep sy = effective_sweep_y(s);
    sig_append_int (o, "sxp",  sx.par_index);
    sig_append_bool(o, "sxv",  sx.over_var);
    sig_append_bool(o, "sxh",  sx.over_h);
    sig_append_int (o, "sxvi", sx.var_index);
    sig_append_str (o, "sxlo", sx.lo_text);
    sig_append_str (o, "sxhi", sx.hi_text);
    sig_append_str (o, "sxn",  sx.n_pts_text);
    sig_append_int (o, "syp",  sy.par_index);
    sig_append_bool(o, "syv",  sy.over_var);
    sig_append_bool(o, "syh",  sy.over_h);
    sig_append_int (o, "syvi", sy.var_index);
    sig_append_str (o, "sylo", sy.lo_text);
    sig_append_str (o, "syhi", sy.hi_text);
    sig_append_str (o, "syn",  sy.n_pts_text);
    // Fix positions — X-slice pins Y at fix_y, Y-slice pins X at fix_x.
    sig_append_double(o, "fx", s.fix_x_value);
    sig_append_double(o, "fy", s.fix_y_value);
    sig_append_bool(o, "cnt", s.continuation_1d_enabled);
    sig_append_bool(o, "bx", s.bif1d_x_enabled);
    sig_append_bool(o, "by", s.bif1d_y_enabled);
    sig_append_bool(o, "lx", s.lle1d_x_enabled);
    sig_append_bool(o, "ly", s.lle1d_y_enabled);
    sig_append_bool(o, "sx", s.ls1d_x_enabled);
    sig_append_bool(o, "sy", s.ls1d_y_enabled);
    return o;
}

std::string build_l3_signature(const CustomTabSharedConfig& s, const CustomSession& cs) {
    std::string o; o.reserve(512);
    sig_append_bool(o, "en", s.level_phase_enabled);
    sig_append_int (o, "kind", s.level3_kind);
    sig_append_shared_state(o, s);
    if (s.level3_kind == 0) {
        // Phase: sim/skip/decimation + all ic_sets (each IC is a map<var,text>).
        sig_append_str(o, "st", cs.phase_session.sim_time);
        sig_append_str(o, "sk", cs.phase_session.skip_time);
        sig_append_str(o, "dc", cs.phase_session.decimation);
        for (size_t i = 0; i < cs.phase_session.ic_sets.size(); ++i) {
            std::string tag = "ic" + std::to_string(i);
            sig_append_map(o, tag.c_str(), cs.phase_session.ic_sets[i].values);
        }
    } else if (!cs.basins_session.configs.empty()) {
        const auto& bc = cs.basins_session.configs[0];
        sig_append_int(o, "bxv",  bc.axis_x_var);
        sig_append_str(o, "bxlo", bc.axis_x_lo_text);
        sig_append_str(o, "bxhi", bc.axis_x_hi_text);
        sig_append_int(o, "byv",  bc.axis_y_var);
        sig_append_str(o, "bylo", bc.axis_y_lo_text);
        sig_append_str(o, "byhi", bc.axis_y_hi_text);
        sig_append_str(o, "bn",   bc.n_pts_text);
        sig_append_int(o, "f1",   bc.feature1);
        sig_append_int(o, "f2",   bc.feature2);
        sig_append_str(o, "beps", bc.eps_dbscan_text);
    }
    return o;
}

void CustomSession::commit_pending_signatures() {
    // Read each level's success from its sub-sessions' last_run flags.
    // Sticky flags — reflect the most recent completed run on that slot,
    // exactly what we want to attribute to the pending commit.
    bool l2d_ok = false;
    if (shared.bif2d_enabled && !bif_session.diagrams.empty()
        && bif_session.diagrams[0].last_run_2d_ok) l2d_ok = true;
    if (shared.lle2d_enabled && !lle_session.curves.empty()
        && lle_session.curves[0].last_run_2d_ok)   l2d_ok = true;
    if (shared.ls2d_enabled  && !ls_session.curves.empty()
        && ls_session.curves[0].last_run_2d_ok)    l2d_ok = true;

    bool l1d_ok = false;
    auto slot_ok = [](auto& coll, size_t i) {
        return coll.size() > i && coll[i].last_run_ok;
    };
    if (shared.bif1d_x_enabled && slot_ok(bif_session.diagrams, 1)) l1d_ok = true;
    if (shared.bif1d_y_enabled && slot_ok(bif_session.diagrams, 2)) l1d_ok = true;
    if (shared.lle1d_x_enabled && slot_ok(lle_session.curves,   1)) l1d_ok = true;
    if (shared.lle1d_y_enabled && slot_ok(lle_session.curves,   2)) l1d_ok = true;
    if (shared.ls1d_x_enabled  && slot_ok(ls_session.curves,    1)) l1d_ok = true;
    if (shared.ls1d_y_enabled  && slot_ok(ls_session.curves,    2)) l1d_ok = true;

    bool l3_ok = false;
    if (shared.level3_kind == 0) l3_ok = phase_session.result.ok;
    else l3_ok = !basins_session.configs.empty() && basins_session.configs[0].last_run_ok;

    auto commit = [](LevelSig& sig, bool ok) {
        if (!sig.pending_armed) return;
        sig.committed     = sig.pending;
        sig.committed_ok  = ok;
        sig.pending_armed = false;
        sig.pending.clear();
    };
    commit(sig_l2d, l2d_ok);
    commit(sig_l1d, l1d_ok);
    commit(sig_l3,  l3_ok);
}
