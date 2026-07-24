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
    std::snprintf(buf, sizeof(buf), "%.10g", v);
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
        e.par_index  = s.axis_x_par_index;
        e.over_var   = s.axis_x_over_var;
        e.var_index  = s.axis_x_var_index;
        e.lo_text    = s.axis_x_lo_text;
        e.hi_text    = s.axis_x_hi_text;
        e.n_pts_text = s.n_x_text;
    } else {
        e.par_index  = s.sweep_x_par_index;
        e.over_var   = s.sweep_x_over_var;
        e.var_index  = s.sweep_x_var_index;
        e.lo_text    = s.sweep_x_lo_text;
        e.hi_text    = s.sweep_x_hi_text;
        e.n_pts_text = s.n_x_1d_text;
    }
    return e;
}

EffectiveSweep effective_sweep_y(const CustomTabSharedConfig& s) {
    EffectiveSweep e;
    if (s.inherit_sweep_from_2d && s.level_2d_enabled) {
        e.par_index  = s.axis_y_par_index;
        e.over_var   = s.axis_y_over_var;
        e.var_index  = s.axis_y_var_index;
        e.lo_text    = s.axis_y_lo_text;
        e.hi_text    = s.axis_y_hi_text;
        e.n_pts_text = s.n_y_text;
    } else {
        e.par_index  = s.sweep_y_par_index;
        e.over_var   = s.sweep_y_over_var;
        e.var_index  = s.sweep_y_var_index;
        e.lo_text    = s.sweep_y_lo_text;
        e.hi_text    = s.sweep_y_hi_text;
        e.n_pts_text = s.n_y_1d_text;
    }
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
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.n_x_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
    // eps_dbscan_text stays on the sub-config (edited in the L2D detail panel).
}

void apply_shared_to_bif1d(const CustomTabSharedConfig& s, BifurcationDiagramConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    copy_prescaller(s, c);
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
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.n_x_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
    // eps_text / nt_text stay on the sub-config.
}

void apply_shared_to_lle1d(const CustomTabSharedConfig& s, LLECurveConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    c.mode_2d = false;
    EffectiveSweep sweep = (dir == 0) ? effective_sweep_x(s) : effective_sweep_y(s);
    c.param_index     = sweep.par_index;
    c.sweep_over_var  = sweep.over_var;
    c.var_sweep_index = sweep.var_index;
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
    c.param_lo_text     = s.axis_x_lo_text;
    c.param_hi_text     = s.axis_x_hi_text;
    c.n_pts_text        = s.n_x_text;
    c.param_index_2     = s.axis_y_par_index;
    c.sweep_over_var_2  = s.axis_y_over_var;
    c.var_sweep_index_2 = s.axis_y_var_index;
    c.param_lo_2_text   = s.axis_y_lo_text;
    c.param_hi_2_text   = s.axis_y_hi_text;
}

void apply_shared_to_ls1d(const CustomTabSharedConfig& s, LSCurveConfig& c, int dir) {
    copy_integrator_and_state(s, c);
    c.mode_2d = false;
    EffectiveSweep sweep = (dir == 0) ? effective_sweep_x(s) : effective_sweep_y(s);
    c.param_index     = sweep.par_index;
    c.sweep_over_var  = sweep.over_var;
    c.var_sweep_index = sweep.var_index;
    c.param_lo_text   = sweep.lo_text;
    c.param_hi_text   = sweep.hi_text;
    c.n_pts_text      = sweep.n_pts_text;
}

void apply_shared_to_phase(const CustomTabSharedConfig& s, PhaseAnalysisSession& ph,
                           const std::vector<std::string>& vars) {
    ph.scheme     = s.scheme;
    ph.symmetry_s = s.symmetry_s;
    ph.step_h     = s.h_text;
    ph.sim_time   = s.t_max_text;
    ph.skip_time  = s.transient_text;
    ph.param_values = s.param_values;

    if (s.phase_use_shared_ic) {
        // Keep as many ic_sets as the user has, but rewrite ic_sets[0] values
        // from shared IC (default sandbox drill-down slot).
        if (ph.ic_sets.empty()) {
            InitialConditionSet ic;
            ic.label = "IC 1";
            for (const auto& v : vars) ic.values[v] = "";
            ph.ic_sets.push_back(std::move(ic));
        }
        auto& ic0 = ph.ic_sets[0];
        for (const auto& v : vars) {
            auto it = s.initial_conditions.find(v);
            ic0.values[v] = (it != s.initial_conditions.end()) ? it->second : std::string();
        }
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

    // Seed shared config from the record's defaults.
    shared.scheme         = "Euler";
    shared.symmetry_s     = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    shared.h_text         = r.step_h.empty() ? std::string("0.01") : r.step_h;
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
