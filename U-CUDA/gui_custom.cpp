#include "gui_internal.h"

// Global system switch — fired from the top-bar combo. Loads the record and re-inits the
// CURRENT tab (mirrors what each per-tab combo used to do); other tabs re-init on entry via
// the block in draw_gui. Non-static: also called from app_main.cpp on startup to restore the
// last-used system (see gui.h::apply_system_switch).
void apply_system_switch(AppModel& model, SystemLibrary& lib,
                         const std::string& name)
{
    try {
        model.from_record(lib.load(name));
        switch (model.app_mode) {
        case AppModel::AppMode::Analysis: {
            model.start_phase_analysis();
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
            break;
        }
        case AppModel::AppMode::Parametric: {
            model.start_parametric_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, jb, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jm = lib.load_session(model.loaded_name, "_last_metrics");
            apply_session_json(model, jm, model.metrics_session, session_from_json_metrics, "_last_metrics");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Dft1D: {
            model.start_dft1d_analysis();
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Basins: {
            model.start_basins_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
            break;
        }
        case AppModel::AppMode::FastSync: {
            model.start_fastsync_analysis();
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
            break;
        }
        case AppModel::AppMode::Custom: {
            // Hard reset + engine wipe. Sub-session buffers, signature caches,
            // AND the CUDA-side PTX cache in parametric_engine all need to
            // clear on system switch — otherwise the second-system Run either
            // produced a stale-looking result or zeroed out the first one.
            model.custom_session = CustomSession{};
            model.parametric_engine.reset();
            model.start_custom_analysis();
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
            break;
        }
        case AppModel::AppMode::Order: {
            model.start_order_analysis();
            std::string jo = lib.load_session(model.loaded_name, "_last_order");
            apply_session_json(model, jo, model.order_session, session_from_json_order, "_last_order");
            std::string jw = lib.load_session(model.loaded_name, "_last_order_windows");
            model.load_or_init_order_plot_windows(jw);
            order_autosave_seed(model);
            break;
        }
        case AppModel::AppMode::Network: {
            model.start_network_analysis();
            std::string jn = lib.load_session(model.loaded_name, "_last_network");
            apply_session_json(model, jn, model.network_session, session_from_json_network, "_last_network");
            break;
        }
        case AppModel::AppMode::Library:
        case AppModel::AppMode::Settings:
        default:
            break;
        }
        // Persist the choice so the next launch restores this system. Read-
        // modify-write so we don't clobber unrelated fields (colormaps, UI
        // scale, etc.). Silent on failure — best-effort UX polish.
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_system_name = model.name;
        save_app_config(get_exe_dir_with_sep(), cfg);
    } catch (...) {}
}

// Custom tab (master-detail pipeline) — see custom_session.h

namespace {

// Комбо выбора цели свипа и разбор числовых полей — общие для всего файла
// (draw_sweep_target_combo / parse_ratio_or, см. верх файла). Раньше здесь
// лежали их локальные копии: комбо со своим стилем подписей ("IC x" вместо
// "x (IC)") и обёртка parse_num_default поверх того же парсера.

// Shared config panel

void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) {
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c      = cs.shared;
    const auto& vars   = cs.vars;
    const auto& params = cs.params;
    auto& phase  = cs.phase_session;
    ImGui::SeparatorText("Shared config");

    // Scheme combo — mirrors draw_diagram_controls / draw_lle_controls layout, so users get
    // the same picker with built-ins + custom schemes. On change, propagate to Phase (L3 shows
    // the same picker) and rebuild its KRS body so the next Phase Run uses the new integrator;
    // L1D/L2D pick it up via copy_integrator_and_state on Run.
    draw_scheme_combo("Scheme", c.scheme, custom_schemes,
        [&phase, &custom_schemes](const std::string& nm) {
            phase.scheme = nm;
            phase.regenerate_krs();
            // Custom КРС и экстраполяционные обёртки в Custom-вкладке считаются только на GPU.
            if (scheme_needs_compiled_body(nm, custom_schemes)) phase.use_gpu = true;
        },
        &cs.enabled_builtin_schemes, &model, model.is_map, &cs.wrapper_schemes);

    // Integration group — mirrors the "Integration##bd_int" collapsing header in
    // draw_diagram_controls (per-line InputNumStr with comma→dot + ↑/↓). Each edited field is
    // mirrored into L1D's own override (`l1d_h_text` for step) and into `phase_session`, so the
    // typed value shows up in the L1D and L3 panels without waiting for Run. L1D keeps
    // independent transient/computing-time overrides — step is intentionally the ONLY per-L1D
    // field kept in sync with shared.
    if (ImGui::CollapsingHeader("Integration##custom_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!model.is_map)
            if (InputNumStr("h",          c.h_text, kFieldW)) {
                c.l1d_h_text  = c.h_text;
                phase.step_h  = c.h_text;
            }
        if (scheme_uses_symmetry(c.scheme, custom_schemes))
            if (InputNumStr("symmetry s", c.symmetry_s, kFieldW))
                phase.symmetry_s = c.symmetry_s;
        if (scheme_uses_cd_fp_iters(c.scheme)) {
            if (InputNumStr("FP iterations n", c.cd_fp_iters, kFieldW))
                phase.cd_fp_iters = c.cd_fp_iters;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
        }
        if (scheme_uses_newton(c.scheme))
            if (draw_newton_fields(c.newton, kFieldW, "custom_newton"))
                phase.newton = c.newton;
        // TT before CT: transient runs first, computing-time is what's
        // actually sampled after — order matches conceptual flow.
        if (InputNumStr(model.is_map ? "transient iterations" : "transient time",
                        c.transient_text, kFieldW))
            phase.skip_time = c.transient_text;
        if (InputNumStr(model.is_map ? "iterations" : "computing time",
                        c.t_max_text, kFieldW))
            phase.sim_time  = c.t_max_text;
        if (InputNumStr("decimator",      c.pre_scaller_text, kFieldW))
            phase.decimation = c.pre_scaller_text;
        InputNumStr("max value",      c.max_value_text, kFieldW); // Phase has no analogue
        // Адаптивный шаг — общий для всех уровней; Phase получает его сразу, как и h.
        ImGui::Separator();
        if (draw_adaptive_block("custom_ad", c.adaptive, c.scheme,
                                kAdUiRaw | kAdUiInterp | kAdUiMaxPts, model.is_map, &model))
            phase.adaptive = c.adaptive;
    }

    // Initial conditions — one InputNumStr per line, matching draw_diagram_controls.
    // Not propagated to Phase: phase.ic_sets is multi-IC (mulistability) and
    // is edited in the L3 Phase panel; the shared IC block drives BD/LLE/LS/Basins.
    draw_named_num_fields("Initial conditions##custom_ic", vars, c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);

    // Parameters — same per-line layout; skip disable for params that are
    // swept on any enabled level (they get their values from the sweep).
    // Non-swept params are also mirrored to phase.param_values so the L3
    // Phase panel Parameters section shows the same live values.
    if (ImGui::CollapsingHeader("Parameters##custom_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        // All params stay editable — the sweep overwrites swept-axis values at run time anyway,
        // and the manual value is useful for initial-frame edits and non-sweeping levels (L3
        // Phase). Swept params are MARKED with a "(swept)" tag so it is visible at a glance
        // which values the sweep will replace.
        std::vector<bool> is_swept(params.size(), false);
        const bool any_2d   = c.bif2d_enabled || c.lle2d_enabled || c.ls2d_enabled;
        const bool any_1d_x = c.bif1d_x_enabled || c.lle1d_x_enabled || c.ls1d_x_enabled;
        const bool any_1d_y = c.bif1d_y_enabled || c.lle1d_y_enabled || c.ls1d_y_enabled;
        // over_h проверяется наравне с over_var: при свипе по шагу par_index
        // остаётся от прежнего выбора, и без этого условия "(swept)" висело на
        // постороннем параметре — том, чей индекс случайно лежал в par_index.
        auto mark = [&](int par_i, bool over_var, bool over_h) {
            if (over_var || over_h) return;
            if (par_i < 0 || par_i >= (int)params.size()) return;
            is_swept[par_i] = true;
        };
        if (c.level_2d_enabled && any_2d) {
            mark(c.axis_x_par_index, c.axis_x_over_var, c.axis_x_over_h);
            mark(c.axis_y_par_index, c.axis_y_over_var, c.axis_y_over_h);
        }
        if (c.level_1d_enabled) {
            if (any_1d_x) {
                EffectiveSweep esx = effective_sweep_x(c);
                mark(esx.par_index, esx.over_var, esx.over_h);
            }
            if (any_1d_y) {
                EffectiveSweep esy = effective_sweep_y(c);
                mark(esy.par_index, esy.over_var, esy.over_h);
            }
        }
        // Effective sweep axes cached once — used by both the (swept) label
        // logic below AND the crosshair sync when the user edits the value
        // of a sweep-axis param manually.
        EffectiveSweep esx = effective_sweep_x(c);
        EffectiveSweep esy = effective_sweep_y(c);
        std::map<std::string, int> par_idx;
        for (int i = 0; i < (int)params.size(); ++i) par_idx.emplace(params[i], i);
        draw_param_tree(&model.param_groups, params, [&](const std::string& p) {
            const int i = par_idx.at(p);
            ImGui::PushID(p.c_str());
            const bool edited = InputNumStr("##v", c.param_values[p], kFieldW);
            ImGui::SameLine();
            draw_field_name(p, &model.field_labels);
            if (edited) {
                phase.param_values[p] = c.param_values[p];
                // Bump debounce timers so the auto-recompute path (L1D
                // partial + Phase, gated by auto_recompute_1d /
                // autorun_on_drilldown) picks up the manual edit — same
                // path a slider drag settle goes through, no Run needed.
                double t = ImGui::GetTime();
                c.last_fix_x_change_time = t;
                c.last_fix_y_change_time = t;
                // If this param IS the effective sweep axis, move the
                // crosshair to the new pinned value so 1D slice plots and
                // 2D heatmaps show it there immediately.
                double new_val = parse_ratio_or(c.param_values[p], 0.0);
                if (!esx.over_var && !esx.over_h && esx.par_index == i) c.fix_x_value = new_val;
                if (!esy.over_var && !esy.over_h && esy.par_index == i) c.fix_y_value = new_val;
            }
            if (is_swept[i]) {
                ImGui::SameLine();
                ImGui::TextDisabled("(swept)");
            }
            ImGui::PopID();
        });
    }
}

// Level 2D detail

void draw_level2d_detail(AppModel& model, CustomSession& cs) {
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    ImGui::Checkbox("Bif",  &c.bif2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE",  &c.lle2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS",   &c.ls2d_enabled);

    // Axis X — combo (sweep target) + lo/hi/N on separate lines so the
    // per-field digit-step (↑/↓) and comma→dot filter work reliably (they
    // don't when InputText's are packed onto one SameLine chain).
    ImGui::TextUnformatted("Axis X:"); ImGui::SameLine();
    draw_sweep_target_combo("##ax", cs.params, cs.vars,
                            c.axis_x_par_index, c.axis_x_over_var, c.axis_x_var_index,
                            c.axis_x_over_h, &c.axis_y_over_h,
                            /*note_when_empty*/ false, 120.0f,
                            scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map);
    InputNumStr("lo##ax", c.axis_x_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                          "sweep lo", c.axis_x_lo_text); });
    InputNumStr("hi##ax", c.axis_x_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                          "sweep hi", c.axis_x_hi_text); });
    // Log-сетка по оси. Свойство оси, поэтому X-срез Level 1D наследует его
    // вместе с par/lo/hi (см. EffectiveSweep::log_scale).
    ImGui::Checkbox("Log scale##ax_log", &c.axis_x_log);
    if (c.axis_x_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    ImGui::TextUnformatted("Axis Y:"); ImGui::SameLine();
    draw_sweep_target_combo("##ay", cs.params, cs.vars,
                            c.axis_y_par_index, c.axis_y_over_var, c.axis_y_var_index,
                            c.axis_y_over_h, &c.axis_x_over_h,
                            /*note_when_empty*/ false, 120.0f,
                            scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                            &model, BroadcastField::SweepTarget2, !model.is_map);
    InputNumStr("lo##ay", c.axis_y_lo_text, kFieldW);
    InputNumStr("hi##ay", c.axis_y_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##ay_log", &c.axis_y_log);
    if (c.axis_y_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Shared N×N grid resolution — kernel `getValueByIdx` requires a square
    // grid, so one field drives both axes (matches Analysis tab).
    InputNumStr("Resolution##a", c.resolution_text, kFieldW);

    // Переменная БД — задаётся здесь и наследуется обоими 1D-срезами (см.
    // CustomTabSharedConfig::bif_writable_var). Рисуется независимо от
    // bif2d_enabled: при выключенном Bif-2D её всё равно читает Level 1D.
    // Комбинация (-1) работает и в 2D, и в 1D — одно и то же ядро.
    ImGui::Separator();
    draw_writable_var_combo(cs.vars, c.bif_writable_var, "Bif variable##custom_wv");

    // Per-type options edited directly on the sub-session's slot [0] (2D config).
    ImGui::Separator();
    if (c.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
        InputNumStr("Bif DBSCAN eps", cs.bif_session.diagrams[0].eps_dbscan_text, kFieldW);
        // Множители осей кластеризации — см. draw_diagram_controls.
        InputNumStr("Bif mult peak",     cs.bif_session.diagrams[0].mult_peak_text, kFieldW);
        InputNumStr("Bif mult interval", cs.bif_session.diagrams[0].mult_interval_text, kFieldW);
    }
    // eps/NT у LLE и LS живут в слоте [0] и наследуются 1D-срезами (см.
    // apply_shared_to_lle1d / _ls1d), поэтому поля показываем и когда сама
    // карта выключена, но включён хотя бы один её срез — иначе их негде править.
    if ((c.lle2d_enabled || c.lle1d_x_enabled || c.lle1d_y_enabled) &&
        !cs.lle_session.curves.empty()) {
        InputNumStr("LLE eps", cs.lle_session.curves[0].eps_text, kFieldW);
        InputNumStr("LLE NT",  cs.lle_session.curves[0].nt_text, kFieldW);
    }
    if ((c.ls2d_enabled || c.ls1d_x_enabled || c.ls1d_y_enabled) &&
        !cs.ls_session.curves.empty()) {
        InputNumStr("LS eps", cs.ls_session.curves[0].eps_text, kFieldW);
        InputNumStr("LS NT",  cs.ls_session.curves[0].nt_text, kFieldW);
    }
    ImGui::Separator();

    // Run buttons live on the top row of draw_custom_controls — this panel
    // is edit-only.
}

// Level 1D detail

void draw_level1d_detail(CustomSession& cs) {
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    if (!c.level_2d_enabled) ImGui::BeginDisabled();
    ImGui::Checkbox("Inherit sweep from Level 2D", &c.inherit_sweep_from_2d);
    if (!c.level_2d_enabled) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(Level 2D off - L1D uses its own sweep)");
    }

    const bool inheriting = c.inherit_sweep_from_2d && c.level_2d_enabled;
    EffectiveSweep sx = effective_sweep_x(c);
    EffectiveSweep sy = effective_sweep_y(c);

    // Inherit disables ONLY the sweep axis (par target + lo/hi + log), leaving N and the
    // L1D-specific integrator fields (h/TT/CT below) always editable — L1D is cheap and
    // interactive, so a finer grid or longer transient than L2D makes sense even on a shared axis.
    // В inherit-режиме виджеты кормим КОПИЯМИ эффективного свипа: править их всё равно нельзя
    // (BeginDisabled), зато в панели видно то, что реально пойдёт в расчёт. Раньше показывались
    // собственные sweep_*, которые в этом режиме не используются, — панель врала об осях среза.
    auto sweep_row = [&](const char* label, const char* tag,
                         const EffectiveSweep& eff,
                         int& own_par, bool& own_ov, int& own_vi,
                         bool& own_oh, bool& own_log,
                         std::string& own_lo, std::string& own_hi,
                         bool* other_oh, std::string& n_text) {
        int         par = inheriting ? eff.par_index : own_par;
        bool        ov  = inheriting ? eff.over_var  : own_ov;
        int         vi  = inheriting ? eff.var_index : own_vi;
        bool        oh  = inheriting ? eff.over_h    : own_oh;
        bool        lg  = inheriting ? eff.log_scale : own_log;
        std::string lo  = inheriting ? eff.lo_text   : own_lo;
        std::string hi  = inheriting ? eff.hi_text   : own_hi;
        bool        other_copy = other_oh ? *other_oh : false;

        ImGui::TextUnformatted(label); ImGui::SameLine();
        if (inheriting) ImGui::BeginDisabled();
        draw_sweep_target_combo((std::string("##") + tag + "p").c_str(),
                                cs.params, cs.vars, par, ov, vi, oh,
                                inheriting ? &other_copy : other_oh,
                                /*note_when_empty*/ false, 120.0f,
                                scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                                nullptr, BroadcastField::SweepTarget,
                                /*allow_h*/ !scheme_is_map(c.scheme));
        InputNumStr((std::string("lo##") + tag).c_str(), lo, kFieldW);
        InputNumStr((std::string("hi##") + tag).c_str(), hi, kFieldW);
        ImGui::Checkbox((std::string("Log scale##") + tag + "log").c_str(), &lg);
        if (lg) {
            ImGui::SameLine();
            ImGui::TextDisabled(inheriting ? "(from Level 2D)" : "(lo/hi > 0)");
        }
        if (inheriting) ImGui::EndDisabled();
        InputNumStr((std::string("N##") + tag).c_str(), n_text, kFieldW);

        // Обратная запись только в своём режиме — в inherit-режиме источник
        // истины остаётся за осями Level 2D.
        if (!inheriting) {
            own_par = par; own_ov  = ov; own_vi = vi;
            own_oh  = oh;  own_log = lg;
            own_lo  = lo;  own_hi  = hi;
        }
    };

    sweep_row("X-sweep:", "sx", sx,
              c.sweep_x_par_index, c.sweep_x_over_var, c.sweep_x_var_index,
              c.sweep_x_over_h, c.sweep_x_log,
              c.sweep_x_lo_text, c.sweep_x_hi_text,
              &c.sweep_y_over_h, c.n_x_1d_text);

    sweep_row("Y-sweep:", "sy", sy,
              c.sweep_y_par_index, c.sweep_y_over_var, c.sweep_y_var_index,
              c.sweep_y_over_h, c.sweep_y_log,
              c.sweep_y_lo_text, c.sweep_y_hi_text,
              &c.sweep_x_over_h, c.n_y_1d_text);

    // Строки выше могли поменять собственный свип L1D — перечитываем, чтобы
    // ползунки ниже снапились по сетке ЭТОГО кадра, а не прошлого.
    sx = effective_sweep_x(c);
    sy = effective_sweep_y(c);

    // L1D-specific integrator overrides — order TT before CT (per user
    // convention: transient runs first, then computing time is what's
    // actually sampled).
    ImGui::Separator();
    ImGui::TextUnformatted("L1D integrator (independent of shared):");
    InputNumStr("h##l1d",              c.l1d_h_text, kFieldW);
    InputNumStr("transient time##l1d", c.l1d_transient_text, kFieldW);
    InputNumStr("computing time##l1d", c.l1d_t_max_text, kFieldW);

    ImGui::Separator();
    // Slice sliders — clamped to the current effective ranges. Values snap to the FINER of the two
    // grids per axis: L2D `resolution_text` (N×N heatmap pixels) and L1D `n_{x,y}_1d_text` (slice
    // samples). When the user cranks 1D resolution up (typical — L1D is cheap, 2D is expensive),
    // the slider gets finer too, so the crosshair on the X-slice plot lands on X-slice data points
    // instead of drifting between them; on the heatmap it may then sit between pixel columns, an
    // acceptable trade-off. SliderScalar<Double> stays in double throughout (SliderFloat would
    // downcast and land 0.2 as 0.20000000298023224). Drag only moves the crosshair; recompute
    // fires on IsItemDeactivatedAfterEdit.
    double fx_lo = parse_ratio_or(sx.lo_text, 0.0);
    double fx_hi = parse_ratio_or(sx.hi_text, 1.0);
    double fy_lo = parse_ratio_or(sy.lo_text, 0.0);
    double fy_hi = parse_ratio_or(sy.hi_text, 1.0);
    if (fx_hi < fx_lo) std::swap(fx_lo, fx_hi);
    if (fy_hi < fy_lo) std::swap(fy_lo, fy_hi);
    // Discrete slider via index-value trick: SliderScalar<Int> steps by 1 (thumb snaps hard, no
    // continuous drift), value = grid index. The format string is a LITERAL world-coord string (no
    // % specifier), so the bubble shows "0.158730" instead of "5". Bubble text is precomputed from
    // the CURRENT idx before SliderScalar runs — during drag it shows the previous frame's snapped
    // world value (1-frame lag on the text only; the thumb always sits on a grid node).
    int n_2d   = parse_int_or(c.resolution_text, 64); if (n_2d   < 2) n_2d   = 64;
    int n_1d_x = parse_int_or(c.n_x_1d_text,     64); if (n_1d_x < 2) n_1d_x = 64;
    int n_1d_y = parse_int_or(c.n_y_1d_text,     64); if (n_1d_y < 2) n_1d_y = 64;
    int n_snap_x = (n_1d_x > n_2d) ? n_1d_x : n_2d;
    int n_snap_y = (n_1d_y > n_2d) ? n_1d_y : n_2d;
        // При log-свипе узлы распределены лог-равномерно (sweep_value_at / getValueByIdx_log),
        // поэтому и шаг ползунка обязан идти по логарифму — иначе thumb прилипает к линейным
        // позициям, которых в данных нет, и крестик уезжает с узла. log при lo<=0 невалиден
        // (движок такой Run отклонит) — деградируем на линейную сетку, чтобы до Run не было NaN.
    auto log_ok = [](bool log_scale, double lo, double hi) {
        return log_scale && lo > 0.0 && hi > 0.0;
    };
    auto idx_from_world = [&](double v, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return 0;
        double i_d;
        if (log_ok(log_scale, lo, hi)) {
            if (!(v > 0.0)) return 0;
            const double l0 = std::log10(lo), l1 = std::log10(hi);
            i_d = (std::log10(v) - l0) / ((l1 - l0) / (double)(n - 1));
        } else {
            i_d = (v - lo) / ((hi - lo) / (double)(n - 1));
        }
        return std::clamp((int)std::round(i_d), 0, n - 1);
    };
    auto world_from_idx = [&](int i, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return lo;
        return sweep_value_at(i, n, lo, hi, log_ok(log_scale, lo, hi),
                              /*reverse*/ false, /*continuation*/ false);
    };

    // Step-arrows + slider row. Arrows walk idx by ±1 (repeat on hold),
    // slider is the discrete int-index widget from above. Same pattern
    // for X and Y — factored into a lambda.
    auto arrow_row = [&](const char* id,
                         int& idx, int idx_min, int idx_max,
                         double lo, double hi, int n_snap, bool log_scale,
                         double& fix_value,
                         double& timer_to_bump,
                         const char* slider_label) {
        ImGui::PushID(id);
        ImGui::PushButtonRepeat(true);
        bool step_changed = false;
        if (ImGui::ArrowButton("l", ImGuiDir_Left)) {
            if (idx > idx_min) { --idx; step_changed = true; }
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (ImGui::ArrowButton("r", ImGuiDir_Right)) {
            if (idx < idx_max) { ++idx; step_changed = true; }
        }
        ImGui::PopButtonRepeat();
        ImGui::PopID();
        ImGui::SameLine();
        char fmt[64];
        std::snprintf(fmt, sizeof(fmt), "%.6g", world_from_idx(idx, lo, hi, n_snap, log_scale));
        ImGui::SetNextItemWidth(240.0f);
        ImGui::SliderScalar(slider_label, ImGuiDataType_S32, &idx,
                            &idx_min, &idx_max, fmt);
        bool released     = ImGui::IsItemDeactivatedAfterEdit();
        bool slider_edit  = ImGui::IsItemActive() || released;
        // Only overwrite fix_value on real user interaction: otherwise the idx_from_world →
        // world_from_idx round-trip would re-snap a fix_value that arrived from the heatmap (on the
        // coarser 2D grid) to the nearest slider-grid node — potentially the finer 1D grid —
        // silently drifting it off the 2D pixel the user just clicked.
        if (slider_edit || step_changed)
            fix_value = world_from_idx(idx, lo, hi, n_snap, log_scale);
        // Arrow clicks fire the same debounce path as slider release —
        // step-changed → immediate commit, no need to wait for release.
        // Bump only the caller-supplied timer so the settled branch can
        // enqueue just the slices that actually depend on this axis.
        if (released || step_changed)
            timer_to_bump = ImGui::GetTime();
    };

    int idx_x = idx_from_world(c.fix_x_value, fx_lo, fx_hi, n_snap_x, sx.log_scale);
    arrow_row("##fix_x_arr", idx_x, 0, n_snap_x - 1,
              fx_lo, fx_hi, n_snap_x, sx.log_scale,
              c.fix_x_value, c.last_fix_x_change_time, "fix X");

    int idx_y = idx_from_world(c.fix_y_value, fy_lo, fy_hi, n_snap_y, sy.log_scale);
    arrow_row("##fix_y_arr", idx_y, 0, n_snap_y - 1,
              fy_lo, fy_hi, n_snap_y, sy.log_scale,
              c.fix_y_value, c.last_fix_y_change_time, "fix Y");

    ImGui::Separator();
    ImGui::TextUnformatted("Enable slices:");
    ImGui::Checkbox("Bif-X", &c.bif1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("Bif-Y", &c.bif1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-X", &c.lle1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-Y", &c.lle1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS-X",  &c.ls1d_x_enabled);  ImGui::SameLine();
    ImGui::Checkbox("LS-Y",  &c.ls1d_y_enabled);

    ImGui::Separator();
    // Переменная БД — read-only эхо настройки Level 2D (срез строится по той
    // же переменной, что и карта). Меняется в панели Level 2D.
    {
        const char* wv_name =
            (c.bif_writable_var == -1) ? "combination"
          : (c.bif_writable_var >= 0 && c.bif_writable_var < (int)cs.vars.size())
                ? cs.vars[c.bif_writable_var].c_str() : "?";
        ImGui::TextDisabled("Bif variable: %s (from Level 2D)", wv_name);
    }
    // Так же и eps/NT для LLE/LS-срезов: read-only эхо слота 2D, правятся в
    // панели Level 2D (см. apply_shared_to_lle1d — срез считается с ними же).
    if ((c.lle1d_x_enabled || c.lle1d_y_enabled) && !cs.lle_session.curves.empty()) {
        ImGui::TextDisabled("LLE eps / NT: %s / %s (from Level 2D)",
                            cs.lle_session.curves[0].eps_text.c_str(),
                            cs.lle_session.curves[0].nt_text.c_str());
    }
    if ((c.ls1d_x_enabled || c.ls1d_y_enabled) && !cs.ls_session.curves.empty()) {
        ImGui::TextDisabled("LS eps / NT: %s / %s (from Level 2D)",
                            cs.ls_session.curves[0].eps_text.c_str(),
                            cs.ls_session.curves[0].nt_text.c_str());
    }
    // Отображение Y на Bif-срезах. Пересчёт не нужен — peak_times и
    // bifurcation_points приходят из одного прогона, поэтому пишем флаг прямо
    // в слоты [1]/[2] и просим автофит; Run не требуется.
    if (ImGui::Checkbox("Plot inter-peaks instead of peak values",
                        &c.plot_inter_peaks_1d)) {
        for (int slot = 1; slot <= 2; ++slot) {
            if ((int)cs.bif_session.diagrams.size() <= slot) break;
            cs.bif_session.diagrams[slot].plot_inter_peaks = c.plot_inter_peaks_1d;
            cs.bif_session.diagrams[slot].fit_request      = true;
        }
        cs.workspace.dirty = true;   // → пересохранение _last_custom.json
    }

    ImGui::Separator();
    ImGui::Checkbox("Continuation (1D)", &c.continuation_1d_enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-recompute 1D on slider/sweep change", &c.auto_recompute_1d);
}

// Level 3 detail

void draw_level3_detail(AppModel& model, CustomSession& cs) {
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.
    ImGui::RadioButton("Phase + Time-domain", &c.level3_kind, 0); ImGui::SameLine();
    ImGui::RadioButton("Basins",              &c.level3_kind, 1);

    ImGui::Separator();
    ImGui::Checkbox("Auto-run", &c.autorun_on_drilldown);

    ImGui::Separator();
    if (c.level3_kind == 0) {
        // Phase: the full IC-sets / integrator / projections UI is embedded right here, so
        // everything lives inside the pipeline's L3 config (no separate top-level window). The
        // Analysis tab uses the same helper with model.phase_session; here we pass Custom's own
        // isolated cs.phase_session.
        draw_phase_controls(cs.phase_session, nullptr, &model);
    } else {
        // Basins sub-panel: IC-space axes + features. All of scheme/h/t_max/
        // etc. come from shared; DBSCAN eps stays per-config.
        if (cs.basins_session.configs.empty()) {
            ImGui::TextDisabled("Basins config not initialised.");
        } else {
            auto& bc = cs.basins_session.configs[0];
            ImGui::TextUnformatted("IC-space X var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##bxv",
                bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_x_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_x_var == i))
                        bc.axis_x_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##bx", bc.axis_x_lo_text, kFieldW);
            InputNumStr("hi##bx", bc.axis_x_hi_text, kFieldW);

            ImGui::TextUnformatted("IC-space Y var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##byv",
                bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_y_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_y_var == i))
                        bc.axis_y_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##by", bc.axis_y_lo_text, kFieldW);
            InputNumStr("hi##by", bc.axis_y_hi_text, kFieldW);
            InputNumStr("N##bxy", bc.n_pts_text, kFieldW);

            // Переменная, чью траекторию читает feature-экстрактор. Своя, а не от Level 2D: у
            // Basins другая задача (оси — IC-пространство), и в отдельной вкладке Basins этот
            // выбор тоже свой. Без комбо в Custom всегда шла первая переменная.
            draw_writable_var_combo(cs.vars, bc.writable_var, "Writable var##custom_bas_wv");

            ImGui::InputInt("feature1", &bc.feature1);
            ImGui::InputInt("feature2", &bc.feature2);
            InputNumStr("DBSCAN eps", bc.eps_dbscan_text, kFieldW);
        }
    }
}

// Pipeline column (master)

// draw_pipeline_column / level_status_badge жили здесь и не вызывались ни
// разу: колонка со списком уровней была заменена на level_header внутри
// draw_custom_controls, а вся её логика статусов переписана там заново.
// Удалены как мёртвый код.

} // namespace (draw helpers)

static void draw_custom_controls(AppModel& model, SystemLibrary& lib) {
    (void)lib;
    auto& cs = model.custom_session;
    auto& c  = cs.shared;

    // Top row: Run + Stop + status, also bound to Ctrl+R (тот же шорткат, что у локального Run
    // в Phase/Bif controls). Dirty-tracked: строим сигнатуру каждого уровня, сравниваем с
    // последней закоммиченной и пропускаем enqueue, если ничего не менялось И прошлый прогон
    // удался. Новые сигнатуры кладутся в `pending_armed`, успешный poll промоутит
    // pending → committed (см. poll_and_commit ниже).
    bool run_now = ImGui::Button("Run");
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_now = true;
    if (run_now) {
        auto try_enqueue = [&](CustomSession::LevelSig& sig,
                               const std::string& current,
                               std::function<void()> do_enqueue) {
            if (current == sig.committed && sig.committed_ok) return;
            do_enqueue();
            sig.pending = current;
            sig.pending_armed = true;
        };
        try_enqueue(cs.sig_l2d, build_l2d_signature(c, cs),
                    [&](){ cs.enqueue_level_2d(model.custom_queue); });
        try_enqueue(cs.sig_l1d, build_l1d_signature(c, cs),
                    [&](){ cs.enqueue_level_1d(model.custom_queue); });
        try_enqueue(cs.sig_l3,  build_l3_signature (c, cs),
                    [&](){ cs.enqueue_level_3 (model.custom_queue); });
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop / clear queue")) {
        model.custom_session.request_cancel_all();
        model.custom_queue.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu queued)", model.custom_queue.size());

    // Auto-recompute 1D on slider settle, plus Phase (если L3=Phase и включён
    // autorun_on_drilldown), чтобы отпускание крестика или ползунка давало отклик сразу.
    // Дебаунс 200 мс + отсутствие расчёта в полёте. Per-axis таймеры
    // (last_fix_{x,y}_change_time) позволяют ставить в очередь ТОЛЬКО срезы, зависящие от
    // сдвинутой оси: drag fix_x пересчитывает Y-срезы (они фиксируют X при fix_x), drag fix_y —
    // X-срезы. Без разделения нетронутый срез пересчитывался бы с теми же данными и вызывал
    // autofit, сбрасывающий ручной зум. Drag хитмапы двигает ОБА таймера.
    if (c.level_1d_enabled && !cs.any_in_flight()) {
        double now = ImGui::GetTime();
        // 500 ms debounce: long enough that stepping a numeric field with
        // the arrow buttons (multiple ticks in quick succession) coalesces
        // into a single recompute instead of firing on every click.
        const double debounce_s = kCustomSliderDebounceSec;
        bool x_settled = c.last_fix_x_change_time > 0.0
                      && now - c.last_fix_x_change_time > debounce_s;
        bool y_settled = c.last_fix_y_change_time > 0.0
                      && now - c.last_fix_y_change_time > debounce_s;
        bool queue_free = model.custom_queue.empty();
        if ((x_settled || y_settled) && queue_free) {
            if (c.auto_recompute_1d) {
                // fix_x moved → Y-slices depend on fix_x. fix_y → X-slices.
                cs.enqueue_level_1d_partial(model.custom_queue,
                    /*x_slices*/ y_settled,
                    /*y_slices*/ x_settled);
            }
            // Pin the shared param_values from the EFFECTIVE sweep axes. Раньше это жило внутри
            // `if (autorun_on_drilldown && ...)` и всегда брало оси L2D, поэтому 1D-standalone
            // свип (inherit off) не пробрасывал drag ползунка обратно в общую секцию Parameters.
            // Теперь пиним безусловно на settle, по effective_sweep_x/y — корректно и для
            // L2D-inherit, и для L1D-standalone.
            // Ось может свипаться по параметру, по НУ или по шагу h; в общие поля отражаем только
            // param-случай (НУ и h подставляются в под-конфиги перед Run — pin_fixed_* в
            // app_model.cpp), а для h дополнительно обновляем общий h_text, чтобы панель
            // показывала реальный шаг расчёта.
            // Формат round-trip (fmt_num_shortest), а НЕ %.6g: v — значение узла сетки от
            // ucuda_node_value, уходящее в ядро через parse_num, и шесть цифр отрезали ~10 знаков,
            // из-за чего drill-down считался в параметре, в котором ячейка не считалась.
            auto pin_axis = [&](const EffectiveSweep& e, double v) {
                if (e.over_h) {
                    if (v > 0.0) c.l1d_h_text = fmt_num_shortest(v);
                    return;
                }
                if (e.over_var) return;
                if (e.par_index < 0 || e.par_index >= (int)cs.params.size()) return;
                c.param_values[cs.params[e.par_index]] = fmt_num_shortest(v);
            };
            EffectiveSweep esx = effective_sweep_x(c);
            EffectiveSweep esy = effective_sweep_y(c);
            if (x_settled) pin_axis(esx, c.fix_x_value);
            if (y_settled) pin_axis(esy, c.fix_y_value);
            if (c.autorun_on_drilldown && c.level_phase_enabled) {
                model.custom_queue.push_back({ c.level3_kind == 0
                    ? CustomQueueItem::Kind::Phase
                    : CustomQueueItem::Kind::Basins });
            }
            if (x_settled) c.last_fix_x_change_time = 0.0;
            if (y_settled) c.last_fix_y_change_time = 0.0;
        }
    }

    // Shared config panel — always visible at the top.
    draw_shared_config(model, cs, cs.custom_schemes);

    ImGui::Separator();

    // Each level = enable-checkbox + CollapsingHeader + status label; headers keep the
    // open/closed state the user left them in (SetNextItemOpen with FirstUseEver opens L2D on
    // the very first appearance, after that ImGui's own state wins). Several headers may be
    // open at once — there is no forced current level.
    // Level colour ladder: base = theme's Header colour, HUE rotated by +45°/level so the three
    // markers land on distinct points of the wheel (dark theme: blue → violet → magenta).
    // Saturation used to be index-tied, which read as one colour fading; both saturation and
    // value are now pinned to the base so all three keep the theme's brightness. Swapping `base`
    // for a user-picked primary makes the rotation cascade automatically.
    auto level_marker_color = [](int level_idx) {
        const ImVec4 base = ImGui::GetStyleColorVec4(ImGuiCol_Header);
        float h, s, v;
        ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
        const int   idx      = (level_idx >= 0 && level_idx < 3) ? level_idx : 0;
        const float hue_step = 45.0f / 360.0f;   // ImGui uses 0..1 for hue.
        float nh = h + hue_step * (float)idx;
        nh -= std::floor(nh);                    // wrap into [0, 1).
        // Lift saturation to full so the markers pop against ambient body
        // background (base Header colour is quite pale in default themes).
        const float ns = std::max(0.75f, s);
        const float nv = std::max(0.85f, v);
        ImVec4 col(0, 0, 0, 1.0f);
        ImGui::ColorConvertHSVtoRGB(nh, ns, nv, col.x, col.y, col.z);
        return col;
    };

    auto level_header = [&](int level_idx,
                            const char* enable_id,
                            bool& enabled,
                            const char* title,
                            bool running, bool has_result,
                            bool default_open,
                            const std::function<void()>& body) {
        ImGui::Checkbox(enable_id, &enabled);
        ImGui::SameLine();
        // Tinted status suffix on the header title itself so it's visible
        // even when the header is collapsed.
        const char* status =
            !enabled ? "off" :
            running  ? "running" :
            has_result ? "ok" : "idle";
        ImVec4 status_col(0.6f, 0.6f, 0.6f, 1.0f);
        if      (!std::strcmp(status, "running")) status_col = ImVec4(1.0f, 0.85f, 0.35f, 1.0f);
        else if (!std::strcmp(status, "ok"))      status_col = ImVec4(0.5f, 0.9f,  0.5f,  1.0f);
        else if (!std::strcmp(status, "off"))     status_col = ImVec4(0.5f, 0.5f,  0.5f,  1.0f);
        char header_label[128];
        std::snprintf(header_label, sizeof(header_label), "%s##hdr_%s", title, enable_id);
        ImGui::SetNextItemOpen(default_open, ImGuiCond_FirstUseEver);
        bool open = ImGui::CollapsingHeader(header_label);
        // Status label on the SAME line as the header (rendered AFTER the
        // header so it sits inline).
        ImGui::SameLine();
        ImGui::TextColored(status_col, "[%s]", status);
        if (open) {
            // Colored left-side bar spanning the body — always in peripheral
            // vision, so whichever setting you're looking at, the coloured
            // stripe on the left tells you which level it belongs to. Bar
            // is drawn AFTER the body so its height matches actual content.
            const float bar_w   = 2.5f;         // 1.5× thinner than before.
            const float bar_gap = 8.0f;         // px between bar and content.
            const float indent  = bar_w + bar_gap;
            const ImVec2 top_screen = ImGui::GetCursorScreenPos();
            ImGui::Indent(indent);
            body();
            ImGui::Unindent(indent);
            const ImVec2 bot_screen = ImGui::GetCursorScreenPos();
            // Pull bar bottom up by ItemSpacing so it doesn't run into the
            // next level's checkbox row (cursor after body sits at the row
            // start of what comes next).
            const float  bar_bot_y  = bot_screen.y - ImGui::GetStyle().ItemSpacing.y;
            const ImU32  col        = ImGui::ColorConvertFloat4ToU32(
                                          level_marker_color(level_idx));
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(top_screen.x,          top_screen.y),
                ImVec2(top_screen.x + bar_w,  bar_bot_y),
                col, 1.5f);
        }
    };

    // Derive status flags (same logic that used to live in draw_pipeline_column).
    bool level2_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index == 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   == 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    == 0);
    bool level2_hasres  =
        (!cs.bif_session.diagrams.empty() && cs.bif_session.diagrams[0].last_run_2d_ok) ||
        (!cs.lle_session.curves.empty()   && cs.lle_session.curves[0].last_run_2d_ok)   ||
        (!cs.ls_session.curves.empty()    && cs.ls_session.curves[0].last_run_2d_ok);
    bool level1_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index > 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   > 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    > 0);
    bool level1_hasres = false;
    if (cs.bif_session.diagrams.size() > 1 && cs.bif_session.diagrams[1].last_run_ok) level1_hasres = true;
    if (cs.bif_session.diagrams.size() > 2 && cs.bif_session.diagrams[2].last_run_ok) level1_hasres = true;
    if (cs.lle_session.curves.size()   > 1 && cs.lle_session.curves[1].last_run_ok)   level1_hasres = true;
    if (cs.lle_session.curves.size()   > 2 && cs.lle_session.curves[2].last_run_ok)   level1_hasres = true;
    if (cs.ls_session.curves.size()    > 1 && cs.ls_session.curves[1].last_run_ok)    level1_hasres = true;
    if (cs.ls_session.curves.size()    > 2 && cs.ls_session.curves[2].last_run_ok)    level1_hasres = true;
    bool level3_running = cs.phase_session.in_flight || cs.basins_session.in_flight;
    bool level3_hasres  = cs.phase_session.result.ok ||
                          (!cs.basins_session.configs.empty() && cs.basins_session.configs[0].last_run_ok);
    const char* l3_title = c.level3_kind == 0 ? "Level 3 - Phase / Time-domain" : "Level 3 - Basins";

    level_header(0, "##en_l2d", c.level_2d_enabled, "Level 2D - Bif / LLE / LS",
                 level2_running, level2_hasres, /*default_open=*/true,
                 [&](){ draw_level2d_detail(model, cs); });
    level_header(1, "##en_l1d", c.level_1d_enabled, "Level 1D - slices",
                 level1_running, level1_hasres, /*default_open=*/false,
                 [&](){ draw_level1d_detail(cs); });
    level_header(2, "##en_l3", c.level_phase_enabled, l3_title,
                 level3_running, level3_hasres, /*default_open=*/false,
                 [&](){ draw_level3_detail(model, cs); });
}

// Custom-tab plot windows: minimal renderer for the pipeline output. Full-featured versions
// with colormap toolbars, colorbar controls and colored-1D density are TODO; this MVP shows
// results and wires drill-down clicks / crosshairs.
namespace {

// Helper: bind slider fix_x/fix_y as crosshair on the given heatmap view,
// and install a drill-down click callback that updates shared.param_values
// + fix_x/y and optionally enqueues a Phase/Basins run.
void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) {
    hv.crosshair_x = cs.shared.level_1d_enabled ? cs.shared.fix_x_value
                                                : std::numeric_limits<double>::quiet_NaN();
    hv.crosshair_y = cs.shared.level_1d_enabled ? cs.shared.fix_y_value
                                                : std::numeric_limits<double>::quiet_NaN();
    // Drag callback: fires every frame while LMB is held inside the plot (HeatmapView
    // on_left_drag). Cheap — just moves the crosshair via fix_x/fix_y so the shared crosshair on
    // every 2D window follows the cursor. Recompute (Phase/Basins enqueue + param_values update)
    // is deferred to on_left_click on release.
    hv.on_left_drag = [&cs](int, int, double snap_x, double snap_y) {
        cs.shared.fix_x_value = snap_x;
        cs.shared.fix_y_value = snap_y;
        // Heatmap drag moves BOTH axes → bump both timers so the settled
        // branch re-runs X-slices AND Y-slices.
        double t = ImGui::GetTime();
        cs.shared.last_fix_x_change_time = t;
        cs.shared.last_fix_y_change_time = t;
    };
    hv.on_left_click = [&cs, &q](int, int, double snap_x, double snap_y) {
        auto& s = cs.shared;
        // Update fix_x/fix_y (also drives crosshair on other 2D windows).
        s.fix_x_value = snap_x;
        s.fix_y_value = snap_y;
        double t = ImGui::GetTime();
        s.last_fix_x_change_time = t;
        s.last_fix_y_change_time = t;
        // Update shared.param_values so any later Phase/Basins run reads the drilled-down
        // location. Only param-sweeps are pinned; var-sweeps stay as IC edits (handled by the
        // pipeline drainer). snap_x/snap_y — значения узлов от ucuda_node_value (та же функция,
        // что у ядра), поэтому пишем их round-trip форматом: стоявший здесь %.6g ронял точность
        // до 6 цифр, и портрет по клику считался рядом с пикселем, а не в нём.
        // Проверка over_h обязательна: при свипе по шагу over_var == false (выбор
        // "dt (h)" в комбо гасит его), par_index остаётся от прежнего выбора, и без
        // этого условия значение ШАГА писалось в посторонний параметр. Сам шаг
        // подставляется в конфиги из fix_x/fix_y при запуске (pin_swept_h в
        // custom_session.cpp, pin_fixed_h в app_model.cpp), отдельного поля тут не надо.
        if (!s.axis_x_over_var && !s.axis_x_over_h && s.axis_x_par_index >= 0 &&
            s.axis_x_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_x_par_index]] = fmt_num_shortest(snap_x);
        }
        if (!s.axis_y_over_var && !s.axis_y_over_h && s.axis_y_par_index >= 0 &&
            s.axis_y_par_index < (int)cs.params.size()) {
            s.param_values[cs.params[s.axis_y_par_index]] = fmt_num_shortest(snap_y);
        }
        if (s.autorun_on_drilldown && s.level_phase_enabled) {
            q.push_back({ s.level3_kind == 0 ? CustomQueueItem::Kind::Phase
                                             : CustomQueueItem::Kind::Basins });
        }
    };
}

} // namespace

// Custom Workspace — split-region layout for Custom AppMode.
//
// Per frame: AppMode radios on top (MainHost); below, Custom Controls panel | splitter |
// tab bar [ Tab 1 | Tab 2 | + ] over a per-tab DockSpace that hosts the plot windows.
//
// Docking model:
//   - Each tab owns one DockSpace with a STABLE id from tab.id via ws_dock_id() — deliberately
//     NOT ImGui::GetID(str), which hashes with the current window-ID stack and would differ
//     inside BeginTabItem vs outside; we submit the same dockspace both ways (KeepAliveOnly
//     before BeginTabBar, real inside BeginTabItem of the active tab).
//   - ALL per-tab dockspaces are submitted every frame with ImGuiDockNodeFlags_KeepAliveOnly so
//     ImGui keeps the nodes alive and windows docked in inactive tabs don't orphan to floating.
//   - Plot windows are docked via DockBuilderDockWindow on first appearance; imgui.ini persists
//     the internal split layout across sessions.
//   - Cross-tab drag&drop: BeginDragDropSource on each plot title bar + BeginDragDropTarget on
//     each tab item → drop calls DockBuilderDockWindow(name, target_ds) and switches tab.

// Stable dockspace ID for a workspace tab: independent of ImGui's ID-stack scope (same id
// inside or outside a window) and unlikely to collide with other IDs. Includes a hash of the
// loaded system name so imgui.ini keeps a separate dock layout per system — otherwise switching
// systems leaked one system's plot placement into another's tabs.
[[nodiscard]] static inline ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}

// Suffix appended to every Custom-mode plot window title so imgui.ini keys
// its dock state per system. Empty when no system is loaded (edge case;
// windows behave as before). "##sys_<name>" — the "##" makes ImGui strip
// the suffix from the visible title while including it in the hashed ID.
[[nodiscard]] static inline std::string custom_win_suffix(const std::string& sys) {
    return sys.empty() ? std::string{} : ("##sys_" + sys);
}

// Drag&drop payload key — passed via ImGui's built-in drag-drop channel.
// Payload data: `const char* name` of the docked window title.
static const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";

// Forward decls — the layout function calls these; they live further down.
static void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);

static void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) {
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (ImGui::Begin("Custom Controls##panel", nullptr, flags)) {
        draw_custom_controls(model, lib);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// Runtime rename state — Custom mode is the only place we rename tabs, so
// keeping it as a file-scope static (not persisted) is fine.
struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};
static WsRenameState g_ws_rename;

static void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) {
    auto& ws = cs.workspace;
    ws.ensure_default();

    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    // No NoBringToFrontOnFocus / NoNavFocus here — clicks on Controls/Workspace
    // widgets were being swallowed with those on; keeping the panels pinned via
    // NoMove/NoResize + SetNextWindowPos/Size is enough to make them "regions".
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoCollapse
                           | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    if (!ImGui::Begin("Custom Workspace##panel", nullptr, flags)) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    // Tab bar with +, close, rename, and cross-tab drop targets.
    // NOTE ON DOCKING: the real DockSpace for the active tab is submitted AFTER EndTabBar, not
    // inside the BeginTabItem body. Submitting DockSpace(id) inside BeginTabItem pushes an
    // ID-stack entry, so ImGui creates an auto-child host "…/DockSpace_XXX" whose host context
    // differs from the KeepAliveOnly submit outside — that mismatch orphaned docked windows on
    // tab switch (plots vanishing when returning to a tab). Real + keep-alive from the same
    // context (workspace panel window) keeps the host consistent.
    int close_id = 0;
    int switch_to_id = 0;
    ImGuiTabBarFlags tb_flags = ImGuiTabBarFlags_Reorderable
                              | ImGuiTabBarFlags_TabListPopupButton
                              | ImGuiTabBarFlags_AutoSelectNewTabs
                              | ImGuiTabBarFlags_FittingPolicyScroll;
    if (ImGui::BeginTabBar("##custom_ws_tabs", tb_flags)) {
        const bool allow_close = (int)ws.tabs.size() > 1;
        for (size_t i = 0; i < ws.tabs.size(); ++i) {
            WorkspaceTab& tab = ws.tabs[i];
            // Stable tab ID via "##ws_tab_<id>" suffix so rename changes
            // only the visible label, not the internal ImGui key.
            std::string label = tab.name + "##ws_tab_" + std::to_string(tab.id);
            bool keep = true;
            bool item_open = ImGui::BeginTabItem(label.c_str(),
                                                 allow_close ? &keep : nullptr,
                                                 ImGuiTabItemFlags_None);
            // Drop-target for cross-tab window move. Accepts our own payload (WS_DRAG_PAYLOAD,
            // `const char*` name) for a future explicit drag handle, and ImGui's native window
            // docking payload (IMGUI_PAYLOAD_TYPE_WINDOW, `ImGuiWindow*`), which fires when the
            // user drags a docked plot's tab out of its dockspace onto ours — that is what makes
            // native drag&drop between tabs work.
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(WS_DRAG_PAYLOAD)) {
                    const char* win_name = (const char*)p->Data;
                    if (win_name && *win_name) {
                        ImGui::DockBuilderDockWindow(win_name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_WINDOW)) {
                    ImGuiWindow* dragged = *(ImGuiWindow**)p->Data;
                    if (dragged && dragged->Name) {
                        ImGui::DockBuilderDockWindow(dragged->Name, ws_dock_id(tab.id, sys));
                        switch_to_id = tab.id;
                        ws.dirty = true;
                    }
                }
                ImGui::EndDragDropTarget();
            }
            // Right-click menu (rename / close).
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Rename")) {
                    g_ws_rename.target_id = tab.id;
                    std::snprintf(g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                  "%s", tab.name.c_str());
                }
                if (allow_close && ImGui::MenuItem("Close tab")) {
                    close_id = tab.id;
                }
                ImGui::EndPopup();
            }
            if (item_open) {
                if (ws.active_tab_id != tab.id) {
                    ws.active_tab_id = tab.id;
                    ws.dirty = true;
                }
                // Empty body — real DockSpace is submitted AFTER EndTabBar
                // (see NOTE ON DOCKING above).
                ImGui::EndTabItem();
            }
            if (!keep) close_id = tab.id;
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing
                                    | ImGuiTabItemFlags_NoTooltip)) {
            WorkspaceTab nt;
            // id keeps monotonically incrementing — it seeds the DockSpace node id, and reusing
            // one could collide with imgui.ini state of a previously-closed tab. The display name
            // instead picks the lowest unused "Tab N" so the counter doesn't grow to 20+.
            nt.id = ws.next_tab_id++;
            int label_num = 1;
            auto name_taken = [&](int n){
                std::string s = "Tab " + std::to_string(n);
                return std::any_of(ws.tabs.begin(), ws.tabs.end(),
                                   [&](const WorkspaceTab& t){ return t.name == s; });
            };
            while (name_taken(label_num)) ++label_num;
            nt.name = "Tab " + std::to_string(label_num);
            ws.tabs.push_back(nt);
            ws.active_tab_id = nt.id;
            ws.dirty = true;
        }
        // Sync ws.tabs order with ImGui's internal reordered order: the user can drag tabs
        // (Reorderable), but ImGui only shuffles its own Tabs array. Without this our vector keeps
        // the original order, so plots docked into "the third tab visually" land in whichever tab
        // id is third in ws.tabs on restart — the sync fixes both the forgotten tab order and the
        // wrong-tab plots.
        if (ImGuiTabBar* tb = ImGui::GetCurrentTabBar()) {
            if (tb->Tabs.Size >= 2 && (int)ws.tabs.size() == tb->Tabs.Size) {
                std::vector<WorkspaceTab> reordered;
                reordered.reserve(ws.tabs.size());
                bool all_matched = true;
                for (int i = 0; i < tb->Tabs.Size; ++i) {
                    ImGuiID tid = tb->Tabs[i].ID;
                    auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                        [&](const WorkspaceTab& t){
                            std::string label = t.name + "##ws_tab_" + std::to_string(t.id);
                            return ImHashStr(label.c_str()) == tid;
                        });
                    if (it == ws.tabs.end()) { all_matched = false; break; }
                    reordered.push_back(*it);
                }
                if (all_matched) {
                    // Only mark dirty when actual order changed, to avoid a
                    // needless _last_custom.json rewrite every frame.
                    bool changed = false;
                    for (size_t i = 0; i < ws.tabs.size(); ++i) {
                        if (ws.tabs[i].id != reordered[i].id) { changed = true; break; }
                    }
                    if (changed) {
                        ws.tabs = std::move(reordered);
                        ws.dirty = true;
                    }
                }
            }
        }
        ImGui::EndTabBar();
    }

    // Submit all workspace dockspaces from the SAME host context (workspace panel, outside
    // BeginTabItem). The active tab gets a real submit that renders and hosts its docked windows;
    // inactive tabs get KeepAliveOnly so their windows stay docked (invisible until switched to)
    // instead of orphaning to floating state.
    for (const auto& tab : ws.tabs) {
        if (tab.id != ws.active_tab_id) {
            ImGui::DockSpace(ws_dock_id(tab.id, sys), ImVec2(0, 0),
                             ImGuiDockNodeFlags_KeepAliveOnly);
        }
    }
    if (std::any_of(ws.tabs.begin(), ws.tabs.end(),
                    [&](const WorkspaceTab& t){ return t.id == ws.active_tab_id; })) {
        ImGui::DockSpace(ws_dock_id(ws.active_tab_id, sys), ImVec2(0, 0),
                         ImGuiDockNodeFlags_None);
    }

    // Deferred tab close — move every window still docked in the closing
    // tab's dockspace into the fallback tab (front). DockBuilderRemoveNode
    // moves them off first, so we manually re-dock instead.
    if (close_id != 0) {
        auto it = std::find_if(ws.tabs.begin(), ws.tabs.end(),
                               [close_id](const WorkspaceTab& t){ return t.id == close_id; });
        if (it != ws.tabs.end() && ws.tabs.size() > 1) {
            ws.tabs.erase(it);
            ws.dirty = true;
            int fallback = ws.tabs.front().id;
            // Find all windows docked in the closing node and re-dock them
            // into the fallback. Iterate a copy of the node's windows list
            // because DockBuilderDockWindow mutates it under us.
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(ws_dock_id(close_id, sys));
            if (node) {
                // Walk the whole subtree (splits) and re-dock every window
                // encountered. Simple recursive lambda.
                std::vector<std::string> to_move;
                std::function<void(ImGuiDockNode*)> walk = [&](ImGuiDockNode* n){
                    if (!n) return;
                    for (int wi = 0; wi < n->Windows.Size; ++wi) {
                        if (n->Windows[wi] && n->Windows[wi]->Name)
                            to_move.emplace_back(n->Windows[wi]->Name);
                    }
                    walk(n->ChildNodes[0]);
                    walk(n->ChildNodes[1]);
                };
                walk(node);
                for (const auto& name : to_move)
                    ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(fallback, sys));
                // Now discard the closed node's split hierarchy.
                ImGui::DockBuilderRemoveNode(ws_dock_id(close_id, sys));
            }
            if (ws.active_tab_id == close_id) ws.active_tab_id = fallback;
        }
    }
    if (switch_to_id != 0) ws.active_tab_id = switch_to_id;

    ImGui::End();
    ImGui::PopStyleVar();

    // Rename modal — centered on viewport, Esc = Cancel, auto-reset if the user clicks outside.
    // IMPORTANT: SetNextWindowPos must be INSIDE the "opening" branch — unconditional, it left
    // the next-window-pos flag dangling and got applied to the first plot window created that
    // frame, which appeared floating at viewport center like a dark overlay.
    if (g_ws_rename.target_id != 0) {
        ImGui::OpenPopup("Rename tab##custom_ws");
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    }
    if (ImGui::BeginPopupModal("Rename tab##custom_ws", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::SetKeyboardFocusHere();
        bool commit = ImGui::InputText("##ws_rename",
                                       g_ws_rename.buf, sizeof(g_ws_rename.buf),
                                       ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::Button("OK") || commit) {
            for (auto& t : ws.tabs) {
                if (t.id == g_ws_rename.target_id) {
                    std::string s = g_ws_rename.buf;
                    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
                    if (!s.empty() && s != t.name) {
                        t.name = s;
                        ws.dirty = true;
                    }
                    break;
                }
            }
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            g_ws_rename.target_id = 0;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    } else if (g_ws_rename.target_id != 0) {
        // Popup was closed by ImGui itself (clicked outside etc.). Clear the
        // trigger flag so we don't immediately re-open.
        g_ws_rename.target_id = 0;
    }
}

void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size) {
    auto& ws = model.custom_session.workspace;
    ws.ensure_default();

    if (area_size.x < 100.0f || area_size.y < 40.0f) return;

    // Clamp splitter to sane bounds for the current viewport.
    const float min_ctrl   = 200.0f;
    const float min_ws     = 300.0f;
    const float splitter_w = 6.0f;
    float max_ctrl = std::max(min_ctrl, area_size.x - min_ws - splitter_w);
    if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
    if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;

    ImVec2 ctrl_pos  = area_pos;
    ImVec2 ctrl_size(ws.controls_width, area_size.y);
    ImVec2 split_pos(area_pos.x + ws.controls_width, area_pos.y);
    ImVec2 split_size(splitter_w, area_size.y);
    ImVec2 ws_pos  (split_pos.x + splitter_w, area_pos.y);
    ImVec2 ws_size (area_size.x - ws.controls_width - splitter_w, area_size.y);

    // Splitter (invisible button + drawn rect).
    ImGui::SetNextWindowPos(split_pos);
    ImGui::SetNextWindowSize(split_size);
    // NoBackground — splitter draws its own filled rect manually, so ImGui's
    // window background (which showed as the "dark strip" artefact on click)
    // is redundant. NoBringToFrontOnFocus swallows the drag input entirely,
    // so we can't use it here (unlike on Controls/Workspace panels).
    ImGuiWindowFlags split_flags = ImGuiWindowFlags_NoTitleBar
                                 | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoMove
                                 | ImGuiWindowFlags_NoScrollbar
                                 | ImGuiWindowFlags_NoScrollWithMouse
                                 | ImGuiWindowFlags_NoCollapse
                                 | ImGuiWindowFlags_NoNavFocus
                                 | ImGuiWindowFlags_NoBackground
                                 | ImGuiWindowFlags_NoDocking;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("##CustomSplitter", nullptr, split_flags);
    ImGui::InvisibleButton("##split_btn", split_size);
    if (ImGui::IsItemActive()) {
        float dx = ImGui::GetIO().MouseDelta.x;
        if (dx != 0.0f) {
            ws.controls_width += dx;
            if (ws.controls_width < min_ctrl) ws.controls_width = min_ctrl;
            if (ws.controls_width > max_ctrl) ws.controls_width = max_ctrl;
            ws.dirty = true;
        }
    }
    ImU32 col = ImGui::IsItemActive()  ? ImGui::GetColorU32(ImGuiCol_SeparatorActive)
              : ImGui::IsItemHovered() ? ImGui::GetColorU32(ImGuiCol_SeparatorHovered)
                                       : ImGui::GetColorU32(ImGuiCol_Separator);
    ImGui::GetWindowDrawList()->AddRectFilled(split_pos,
        ImVec2(split_pos.x + split_size.x, split_pos.y + split_size.y), col);
    if (ImGui::IsItemHovered() || ImGui::IsItemActive())
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::End();
    ImGui::PopStyleVar(3);

    draw_custom_controls_panel(model, lib, ctrl_pos, ctrl_size);
    draw_custom_workspace_panel(model.custom_session, model.loaded_name, ws_pos, ws_size);
    draw_custom_plot_windows(model, lib, cb);
}

static void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    (void)lib; (void)cb;
    auto& cs = model.custom_session;
    auto& q  = model.custom_queue;
    auto& ws = cs.workspace;
    // Per-system suffix appended to every plot window title. imgui.ini keys
    // dock state per window name, so without a per-system suffix the same
    // plot titles (e.g. "Custom Bif 2D") would share layout across systems.
    // Empty when no system is loaded — safe (title unchanged from before).
    const std::string& sys    = model.loaded_name;
    const std::string  suffix = custom_win_suffix(sys);
    // Per-system delta XOR'd into every plot's owner_id: HeatmapView / PlotRenderer cache their
    // GPU texture per owner_id, so without it both systems shared one texture and the FIRST
    // system's pixels stayed on screen while the SECOND system's data sat unrendered underneath.
    // Low bits of the system-name hash — enough entropy without colliding with the hand-picked
    // owner_id constants below.
    const int sys_owner_delta = sys.empty() ? 0
                              : (int)(ImHashStr(sys.c_str()) & 0x00FFFFFF);

    // Persistent per-window renderers + views. Indexed by fixed slots so their
    // ImGui-IDs stay stable across frames. Hoisted to the top of the function
    // (together with the L1D-side statics and the init-seen sentinels) so a
    // single system-change block below can reset the whole cache tier.
    static std::array<std::unique_ptr<PlotRenderer>, 3> renderers_2d;
    static std::array<HeatmapView, 3>                   heatmaps;
    static std::array<int, 3>                           hm_init_cmap_seen = { -2, -2, -2 };
    static int                                          ls_init_exp_seen  = -999;
    static std::array<std::unique_ptr<Plot2DView>,  6>  l1_views;
    static std::array<std::unique_ptr<PlotRenderer>, 6> l1_renderers;
    static std::array<std::vector<std::vector<double>>, 6> l1_bufs;
    // Custom Basins (Level-3 kind=1) HeatmapView + renderer — hoisted so the
    // system-change block below can reset them alongside the other statics.
    static HeatmapView                                  bsn_hv;
    static PlotRenderer                                 bsn_renderer;
    for (auto& r : renderers_2d) if (!r) r = std::make_unique<PlotRenderer>();

    // System-change detection. Slot-indexed static caches (HeatmapView view_valid, autofit ranges,
    // seen colormap; Plot2DView view limits and cached VBO series_generation) leaked across systems
    // because slot 0 of Chen and Rossler shared one HeatmapView instance. On rename of the loaded
    // system, wipe the per-slot view / seen state so the new one starts with fresh autofit and
    // colormap seeding. renderers_* (PlotRenderer/GPU textures + FBOs) are reused: the owner_id XOR
    // with sys_owner_delta already makes PlotRenderer treat the slot as a new cache entry, and
    // destroying them here would leak GPU memory tied to the GL context.
    static std::string last_system_for_plots;
    if (last_system_for_plots != sys) {
        last_system_for_plots = sys;
        // HeatmapView / Plot2DView are non-copyable — reset only the fields that gate rendering
        // (view_valid / data_gen_cached / series_generation), which are the actual leak. Plot2DView
        // caches its VBO by (owner_id, series_generation), and the owner_id XOR alone wasn't enough:
        // the second system's data_generation=1 matched the first's cached series_generation=1, so
        // Plot2DView kept rendering the FIRST system's VBO. Colormap / autoscale / manual v-limits
        // stay, so user prefs aren't wiped mid-session.
        for (auto& hv : heatmaps) { hv.view_valid = false; hv.data_gen_cached = -1; }
        for (auto& v  : l1_views) if (v) { v->view_valid = false; v->series_generation = -1; }
        for (auto& b  : l1_bufs)  b.clear();
        bsn_hv.view_valid = false; bsn_hv.data_gen_cached = -1;
        hm_init_cmap_seen = { -2, -2, -2 };
        ls_init_exp_seen  = -999;
    }

    // Clamp fix_x/fix_y to the current effective sweep range every frame. custom_session.h
    // documents this ("clamped to current effective sweep ranges") but it was never implemented —
    // a stale fix outside the range fed the kernel via apply_shared_to_bif1d/pin_param and produced
    // degenerate 1D results (all-zero or empty on Run) until the user dragged the slider back in.
    {
        EffectiveSweep esx = effective_sweep_x(cs.shared);
        EffectiveSweep esy = effective_sweep_y(cs.shared);
        double x_lo = parse_ratio_or(esx.lo_text, 0.0), x_hi = parse_ratio_or(esx.hi_text, 1.0);
        double y_lo = parse_ratio_or(esy.lo_text, 0.0), y_hi = parse_ratio_or(esy.hi_text, 1.0);
        if (x_hi < x_lo) std::swap(x_lo, x_hi);
        if (y_hi < y_lo) std::swap(y_lo, y_hi);
        if (cs.shared.fix_x_value < x_lo) cs.shared.fix_x_value = x_lo;
        if (cs.shared.fix_x_value > x_hi) cs.shared.fix_x_value = x_hi;
        if (cs.shared.fix_y_value < y_lo) cs.shared.fix_y_value = y_lo;
        if (cs.shared.fix_y_value > y_hi) cs.shared.fix_y_value = y_hi;
    }

    // Dock a plot window into the active tab ONLY when there is no dock memory for it — neither at
    // runtime (`window->DockId`) NOR in imgui.ini (`ImGuiWindowSettings::DockId`). This lets a
    // reloaded session restore its per-tab layout from ini on the very first frame, when the window
    // hasn't been Begun yet (FindWindowByName returns null) but ImGui has the settings ready.
    // Overriding a persisted DockId here dumped all plots into the active tab on startup.
    // Skipped while the mouse is held down, so an in-flight drag isn't yanked.
    auto ensure_docked = [&](const std::string& name) {
        if (ImGui::GetIO().MouseDown[0]) return;
        ImGuiID persisted_dock = 0;
        if (ImGuiWindow* w = ImGui::FindWindowByName(name.c_str())) {
            persisted_dock = w->DockId;
        } else if (ImGuiWindowSettings* s =
                       ImGui::FindWindowSettingsByID(ImHashStr(name.c_str()))) {
            persisted_dock = s->DockId;   // will be applied on window's next Begin
        }
        if (persisted_dock != 0) return;
        ImGui::DockBuilderDockWindow(name.c_str(), ws_dock_id(ws.active_tab_id, sys));
    };

    // Cross-tab window move is handled entirely via native ImGui drag&drop: drop a plot's tab onto
    // a target tab in the workspace tab bar (BeginDragDropTarget in draw_custom_workspace_panel
    // accepts IMGUI_PAYLOAD_TYPE_WINDOW). No MMB menu here — MMB is claimed by heatmap/1D-plot
    // views for crosshair-drag / slice movement.

    auto axis_name = [&](bool over_var, int par_i, int var_i, bool over_h) -> std::string {
        if (over_h)   return "h";
        if (over_var) return (var_i >= 0 && var_i < (int)cs.vars.size())
                             ? cs.vars[var_i] + " (IC)" : "x";
        return (par_i >= 0 && par_i < (int)cs.params.size()) ? cs.params[par_i] : "param";
    };
    const std::string ax_x = axis_name(cs.shared.axis_x_over_var,
                                       cs.shared.axis_x_par_index,
                                       cs.shared.axis_x_var_index,
                                       cs.shared.axis_x_over_h);
    const std::string ax_y = axis_name(cs.shared.axis_y_over_var,
                                       cs.shared.axis_y_par_index,
                                       cs.shared.axis_y_var_index,
                                       cs.shared.axis_y_over_h);

    // Level 2D heatmaps
    struct L2Slot {
        std::string title;   // includes per-system suffix so imgui.ini isolates layout per system
        bool        show;
        bool        has_data;
        int         n_pts;
        const double* values;
        double lo_x, hi_x, lo_y, hi_y;
        double vmin, vmax;
        int    data_gen;
        int    owner_id;
        const std::vector<int>* flags;   // режим ячейки для подсказки (regime_cell_label)
    };
    L2Slot slots[3] = {};
    if (cs.shared.level_2d_enabled) {
        if (cs.shared.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
            auto& d = cs.bif_session.diagrams[0];
            slots[0] = { "Custom Bif 2D" + suffix, true, d.last_run_2d_ok && !d.result_2d.values.empty(),
                         d.result_2d.n_pts, d.result_2d.values.data(),
                         d.result_2d.param_lo, d.result_2d.param_hi,
                         d.result_2d.param_lo_2, d.result_2d.param_hi_2,
                         d.result_2d.min_val, d.result_2d.max_val,
                         d.data_generation_2d, 0xCB1F2D ^ sys_owner_delta, &d.result_2d.flags };
        }
        if (cs.shared.lle2d_enabled && !cs.lle_session.curves.empty()) {
            auto& c = cs.lle_session.curves[0];
            slots[1] = { "Custom LLE 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, c.result_2d.values.data(),
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         c.result_2d.min_val, c.result_2d.max_val,
                         c.data_generation_2d, 0xCE1E2D ^ sys_owner_delta, &c.result_2d.flags };
        }
        if (cs.shared.ls2d_enabled && !cs.ls_session.curves.empty()) {
            auto& c = cs.ls_session.curves[0];
            // Fallback plane pointer = first exponent (L1). The real plane for has_data / render is
            // picked later in the toolbar block, so has_data must NOT depend on the currently
            // selected exponent — otherwise picking "sum L_i" flips the slot to "No data yet"
            // because the fallback here goes nullptr.
            const double* values_default = c.result_2d.values.empty()
                ? nullptr : c.result_2d.values.data();
            double vmin = c.result_2d.min_val.empty() ? 0.0 : c.result_2d.min_val[0];
            double vmax = c.result_2d.max_val.empty() ? 1.0 : c.result_2d.max_val[0];
            slots[2] = { "Custom LS 2D" + suffix, true, c.last_run_2d_ok && !c.result_2d.values.empty(),
                         c.result_2d.n_pts, values_default,
                         c.result_2d.param_lo, c.result_2d.param_hi,
                         c.result_2d.param_lo_2, c.result_2d.param_hi_2,
                         vmin, vmax, c.data_generation_2d, 0xC152D0 ^ sys_owner_delta,
                         &c.result_2d.flags };
        }
    }
    // One-time init of per-slot HeatmapView colormap from the sub-session config (or app default).
    // Runs whenever the sub-session-level slot's colormap changes, so a slot's first appearance
    // picks up the persisted choice while later toolbar picks update both places.
    // (hm_init_cmap_seen is hoisted to the top of this function so the system-change reset can wipe it.)
    auto init_cmap_from_config = [&](int i, int cfg_cmap) {
        if (hm_init_cmap_seen[i] == cfg_cmap) return;
        hm_init_cmap_seen[i] = cfg_cmap;
        heatmaps[i].colormap =
            (HeatmapColormap)colormap_id_or(cfg_cmap, model.heatmap_colormap);
    };
    // Sync LS exponent choice from the sub-session's persisted display_exponent_idx (sentinel -1 =
    // sum L_i) on first appearance — otherwise session_from_json_custom loads the pref but the
    // HeatmapView silently starts at 0 (L1) until the user re-picks. Same "seen" pattern as the
    // colormap sync, so later user picks aren't clobbered. (ls_init_exp_seen hoisted to the top.)
    if (!cs.ls_session.curves.empty()) {
        int cfg_exp = cs.ls_session.curves[0].display_exponent_idx;
        if (ls_init_exp_seen != cfg_exp) {
            ls_init_exp_seen = cfg_exp;
            heatmaps[2].display_exponent_idx = cfg_exp;
        }
    }
    if (!cs.bif_session.diagrams.empty())
        init_cmap_from_config(0, cs.bif_session.diagrams[0].colormap_idx);
    if (!cs.lle_session.curves.empty())
        init_cmap_from_config(1, cs.lle_session.curves[0].colormap_idx);
    if (!cs.ls_session.curves.empty())
        init_cmap_from_config(2, cs.ls_session.curves[0].colormap_idx);

    // Автоскейл 2D-карт. Тот же узор, что в Parametric (draw_bifurcation_plot:
    // "bool fit = bdact.fit_request_2d; if (fit) ... = false"): запрос живёт в
    // конфиге подсессии и взводится приходом результата. Здесь в render уходило
    // жёстко false, поэтому после первого прогона HeatmapView фиксировал вид
    // (view_valid) и больше его не трогал: пересчёт в ДРУГОМ диапазоне рисовался
    // в старых границах — при расширении диапазона это выглядело как "новый
    // расчёт не сработал", потому что на экране оставался прежний узкий кусок.
    //
    // Забираем флаг у самой отрисовки, а не при сборке slots: карты сложены в
    // один док, у невидимой вкладки Begin вернёт false, и съеденный за неё
    // запрос пропал бы (ровно та же ловушка, что у Projection::fit_pending).
    auto take_fit_2d = [&cs](int slot) {
        bool* f = nullptr;
        if      (slot == 0 && !cs.bif_session.diagrams.empty())
            f = &cs.bif_session.diagrams[0].fit_request_2d;
        else if (slot == 1 && !cs.lle_session.curves.empty())
            f = &cs.lle_session.curves[0].fit_request_2d;
        else if (slot == 2 && !cs.ls_session.curves.empty())
            f = &cs.ls_session.curves[0].fit_request_2d;
        if (!f || !*f) return false;
        *f = false;
        return true;
    };
    for (int i = 0; i < 3; ++i) {
        if (!slots[i].show) continue;
        ensure_docked(slots[i].title);
        if (ImGui::Begin(slots[i].title.c_str())) {
            HeatmapView& hv = heatmaps[i];

            // Toolbar — mirrors draw_bifurcation_plot / draw_ls_plot layout
            // (colormap combo + autoscale + vmin/vmax + swap axes + LS
            // exponent picker for slot 2).
            ImGui::PushID(i);

            // LS-плоскость по выбранной экспоненте. Заполняется внутри
            // extras-колбэка тулбара (там же, где рисуется picker), а
            // используется ниже при render — поэтому объявлено до тулбара.
            const double* ls_plane = nullptr;
            double        ls_vmin = 0.0, ls_vmax = 1.0;
            int           ls_gen  = slots[i].data_gen;

            {
                HeatmapToolbarOpts topts;
                topts.persist_colormap = [&](int cm) {
                    // Persist per-slot in the owning sub-session config, so a
                    // saved _last_custom.json restores the choice.
                    if      (i == 0 && !cs.bif_session.diagrams.empty())
                        cs.bif_session.diagrams[0].colormap_idx = cm;
                    else if (i == 1 && !cs.lle_session.curves.empty())
                        cs.lle_session.curves[0].colormap_idx = cm;
                    else if (i == 2 && !cs.ls_session.curves.empty())
                        cs.ls_session.curves[0].colormap_idx = cm;
                    hm_init_cmap_seen[i] = cm;
                };
                topts.extras = [&]() {
                    if (i != 2 || cs.ls_session.curves.empty()) return;
                    auto& cact = cs.ls_session.curves[0];
                    if (!cact.last_run_2d_ok || cact.result_2d.n_exponents <= 0) return;
                    draw_ls_exponent_picker(hv, cact.result_2d.n_exponents,
                                            [&](int j) { cact.display_exponent_idx = j; });
                    ls_resolve_plane(cact, hv.display_exponent_idx,
                                     ls_plane, ls_vmin, ls_vmax, ls_gen);
                };
                draw_heatmap_toolbar(hv, topts);
            }

            if (!slots[i].has_data) {
                ImGui::TextDisabled("No data yet. Press Run / Run Level 2D.");
                ImGui::PopID();
            } else {
                hv.x_axis.name = ax_x;
                hv.y_axis.name = ax_y;
                // Лог-оси — как в Parametric (hb.x_axis.log_scale = bd.log_scale): движок
                // раскладывает узлы лог-равномерно, ось обязана это повторить, иначе тики и
                // snap курсора врут. Флаг берём из конфига слота, а не из живого shared:
                // конфиг получает log вместе с диапазоном на Run (apply_shared_to_*), поэтому
                // ось и данные всегда описывают один прогон.
                if (i == 0 && !cs.bif_session.diagrams.empty()) {
                    hv.x_axis.log_scale = cs.bif_session.diagrams[0].log_scale;
                    hv.y_axis.log_scale = cs.bif_session.diagrams[0].log_scale_2;
                } else if (i == 1 && !cs.lle_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.lle_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.lle_session.curves[0].log_scale_2;
                } else if (i == 2 && !cs.ls_session.curves.empty()) {
                    hv.x_axis.log_scale = cs.ls_session.curves[0].log_scale;
                    hv.y_axis.log_scale = cs.ls_session.curves[0].log_scale_2;
                }
                wire_2d_heatmap_interaction(hv, cs, q);

                // Export menu on right-click (parity with Parametric).
                const bool bd_busy = (i == 0) && cs.bif_session.in_flight
                                     && cs.bif_session.is_2d_run
                                     && cs.bif_session.running_diagram_index == 0;
                const bool lle_busy = (i == 1) && cs.lle_session.in_flight
                                      && cs.lle_session.is_2d_run
                                      && cs.lle_session.running_curve_index == 0;
                const bool ls_busy = (i == 2) && cs.ls_session.in_flight
                                     && cs.ls_session.is_2d_run
                                     && cs.ls_session.running_curve_index == 0;
                // Ctrl+T: цель — ОБЩИЙ config вкладки, а не конфиг подсессии.
                // Подсессии всё равно пересобираются из него срезами
                // apply_shared_to_* на каждом Run, и запись в них пропала бы.
                const std::vector<ViewRangeTarget> vrt = {
                    vr_target_fixed("Shared config",
                        &cs.shared.axis_x_lo_text, &cs.shared.axis_x_hi_text,
                        &cs.shared.axis_y_lo_text, &cs.shared.axis_y_hi_text) };
                hv.popup_extras = [i, &cs, &cb, bd_busy, lle_busy, ls_busy, &vrt, &model, &hv]() {
                    const bool busy = bd_busy || lle_busy || ls_busy;
                    draw_export_menu_item(busy, cb, [i, &cs](const std::string& path) {
                        if      (i == 0 && !cs.bif_session.diagrams.empty())
                            data_export::export_bif2d(cs.bif_session.diagrams[0].result_2d, path);
                        else if (i == 1 && !cs.lle_session.curves.empty())
                            data_export::export_lle2d(cs.lle_session.curves[0].result_2d, path);
                        else if (i == 2 && !cs.ls_session.curves.empty())
                            data_export::export_ls2d(cs.ls_session.curves[0].result_2d, path);
                    });
                    draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
                };

                // Route through LS-specific plane/vmin/vmax if the user
                // picked a different exponent; otherwise use the pre-baked
                // slot values.
                const double* values_ptr = (i == 2 && ls_plane) ? ls_plane : slots[i].values;
                double        vmin_use   = (i == 2 && ls_plane) ? ls_vmin  : slots[i].vmin;
                double        vmax_use   = (i == 2 && ls_plane) ? ls_vmax  : slots[i].vmax;
                int           gen_use    = (i == 2 && ls_plane) ? ls_gen   : slots[i].data_gen;
                if (slots[i].flags) {
                    hv.cell_regime = regime_cell_label(*slots[i].flags, slots[i].n_pts);
                    hv.cell_not_computed = cancelled_cell_mask(*slots[i].flags, slots[i].n_pts);
                }
                if (i == 0) {   // Bif 2D: значение — период, -1 / 0 — коды режима (как в Parametric)
                    hv.value_name = "period";
                    hv.value_is_regime_code = true;
                }

                ImVec2 avail  = ImGui::GetContentRegionAvail();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                hv.render(*renderers_2d[i], origin, avail,
                          slots[i].owner_id, gen_use,
                          slots[i].n_pts, slots[i].n_pts,
                          values_ptr,
                          slots[i].lo_x, slots[i].hi_x,
                          slots[i].lo_y, slots[i].hi_y,
                          vmin_use, vmax_use,
                          take_fit_2d(i));
                handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(),
                               hv.swap_axes, &model);
                ImGui::PopID();
            }
        }
        ImGui::End();
    }

    // Level 1D slice plots (six independently toggleable)
    // Kind enum for the slot dispatch. Each slot picks its data from the
    // owning sub-session's slot [1] (X-slice) or [2] (Y-slice).
    enum class L1Kind { Bif, LLE, LS };
    struct L1Slot {
        std::string title;   // includes per-system suffix
        bool        show = false;
        L1Kind      kind = L1Kind::Bif;
        int         cfg_idx = -1;   // 1 = X-slice, 2 = Y-slice inside sub-session
        double      fix_pt = 0.0;   // world X of the OTHER-axis crosshair
    };
    L1Slot lslots[6] = {};
    if (cs.shared.level_1d_enabled) {
        double fx = cs.shared.fix_x_value;
        double fy = cs.shared.fix_y_value;
        if (cs.shared.bif1d_x_enabled) lslots[0] = { "Custom Bif 1D (X-slice)" + suffix, true, L1Kind::Bif, 1, fx };
        if (cs.shared.bif1d_y_enabled) lslots[1] = { "Custom Bif 1D (Y-slice)" + suffix, true, L1Kind::Bif, 2, fy };
        if (cs.shared.lle1d_x_enabled) lslots[2] = { "Custom LLE 1D (X-slice)" + suffix, true, L1Kind::LLE, 1, fx };
        if (cs.shared.lle1d_y_enabled) lslots[3] = { "Custom LLE 1D (Y-slice)" + suffix, true, L1Kind::LLE, 2, fy };
        if (cs.shared.ls1d_x_enabled)  lslots[4] = { "Custom LS 1D (X-slice)"  + suffix, true, L1Kind::LS,  1, fx };
        if (cs.shared.ls1d_y_enabled)  lslots[5] = { "Custom LS 1D (Y-slice)"  + suffix, true, L1Kind::LS,  2, fy };
    }

    // (l1_views / l1_renderers / l1_bufs statics hoisted to the top of the
    // function so the system-change reset can wipe them uniformly.)

    for (int i = 0; i < 6; ++i) {
        if (!lslots[i].show) continue;
        if (!l1_views[i])     l1_views[i]     = std::make_unique<Plot2DView>();
        if (!l1_renderers[i]) l1_renderers[i] = std::make_unique<PlotRenderer>();
        Plot2DView&  view = *l1_views[i];
        PlotRenderer& rnd = *l1_renderers[i];

        ensure_docked(lslots[i].title);
        if (!ImGui::Begin(lslots[i].title.c_str())) { ImGui::End(); continue; }
        ImGui::PushID(i);

        // Resolve owning config + result + real sweep-range from *_text
        // fields (result.param_lo/hi is often 0..1 default until engine
        // fills it — falling back to config text keeps the axis correct
        // when the user set 0.1..0.35 on the sweep).
        const auto axis_label_for_slot = [&](int cfg_idx) -> std::string {
            // cfg_idx 1 = X-slice → sweep_x_par_index or axis_x if inherit;
            // cfg_idx 2 = Y-slice → sweep_y_*.
            EffectiveSweep e = (cfg_idx == 1) ? effective_sweep_x(cs.shared)
                                              : effective_sweep_y(cs.shared);
            if (e.over_h) return "h";
            if (e.over_var)
                return (e.var_index >= 0 && e.var_index < (int)cs.vars.size())
                        ? cs.vars[e.var_index] + " (IC)" : "x (IC)";
            return (e.par_index >= 0 && e.par_index < (int)cs.params.size())
                    ? cs.params[e.par_index] : "param";
        };

        // ЕДИНАЯ конфигурация вида — та же функция, что применяет Parametric.
        // Раньше здесь стоял свой набор присваиваний, из-за чего одинаковые
        // диаграммы отличались между вкладками (см. configure_param_plot_view).
        const ParamPlotKind pkind =
            (lslots[i].kind == L1Kind::Bif) ? ParamPlotKind::Bifurcation
          : (lslots[i].kind == L1Kind::LLE) ? ParamPlotKind::LLE
                                            : ParamPlotKind::LS;
        configure_param_plot_view(view, pkind);

        bool ok = false;
        int  data_gen = 0;
        double param_lo = 0.0, param_hi = 1.0;
        int    n_pts = 0;
        // Лог-масштаб оси среза берём из КОНФИГА слота, а не из shared: конфиг получает log
        // вместе с param_lo/hi в apply_shared_to_bif1d на Run, поэтому ось и диапазон описывают
        // один прогон. Живой чекбокс в панели поменяет ось после Run (он входит в l1d-сигнатуру).
        bool   slice_log = false;
        // Y-axis label + series bookkeeping filled per-kind below.

        if (lslots[i].kind == L1Kind::Bif) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.bif_session.diagrams.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& d = cs.bif_session.diagrams[idx];
            ok = d.last_run_ok;
            data_gen = d.data_generation;
            param_lo = parse_ratio_or(d.param_lo_text, 0.0);
            param_hi = parse_ratio_or(d.param_hi_text, 1.0);
            n_pts = d.result.n_pts;
            slice_log = d.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            view.y_axis.name = (d.writable_var >= 0 && d.writable_var < (int)cs.vars.size())
                               ? cs.vars[d.writable_var] : "X";
            // Как в Parametric: при plot_inter_peaks по Y идут интервалы
            // между пиками, а не сама переменная.
            if (d.plot_inter_peaks) view.y_axis.name += " interval";
            // Custom point style — тот же тулбар и та же семантика, что в
            // Parametric (состояние живёт в самой БД, поэтому вкладки не
            // расходятся). Здесь член ровно один (X- или Y-срез),
            // синхронизировать нечего.
            if (draw_point_style_toolbar(d, "custombd"))
                cs.workspace.dirty = true;   // → пересохранение _last_custom.json
            apply_point_style(view, d);
        } else if (lslots[i].kind == L1Kind::LLE) {
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.lle_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.lle_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = (int)c.result.lyapunov.size();
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
        } else { // LS
            int idx = lslots[i].cfg_idx;
            if (idx < 0 || idx >= (int)cs.ls_session.curves.size()) { ImGui::PopID(); ImGui::End(); continue; }
            auto& c = cs.ls_session.curves[idx];
            ok = c.last_run_ok;
            data_gen = c.data_generation;
            param_lo = parse_ratio_or(c.param_lo_text, 0.0);
            param_hi = parse_ratio_or(c.param_hi_text, 1.0);
            n_pts = c.result.n_pts;
            slice_log = c.log_scale;
            view.x_axis.name = axis_label_for_slot(idx);
            // LS shows every exponent as its own coloured line (parity with
            // draw_ls_plot in Parametric) — no exponent picker here; series
            // are built in the render block below.
        }

        if (!ok) {
            ImGui::TextDisabled("No data yet.");
            ImGui::PopID(); ImGui::End(); continue;
        }

        // Autofit по данным, как в Parametric (configure_sweep_x_axis).
        view.x_fit_use_explicit = false;
        // Лог-ось — как в Parametric; узлы движка лежат лог-равномерно, ось
        // и snap обязаны это повторить. log при lo<=0 невозможен (движок
        // такой Run отклоняет), но проверку держим здесь тоже — конфиг мог
        // прийти из старой сессии.
        view.x_axis.log_scale = slice_log && std::min(param_lo, param_hi) > 0.0;

        // Sweep-position crosshair — vertical line at fix_x (X-slice)
        // or fix_y (Y-slice), synced with the slider / drag on 2D heatmaps.
        // Hidden if Level 1D isn't the driver (falls out to NaN → no draw).
        view.crosshair_x = cs.shared.level_1d_enabled
                           ? lslots[i].fix_pt
                           : std::numeric_limits<double>::quiet_NaN();
        // Красим вертикальный крест по тому свипу, которому принадлежит срез: один цвет на 2D
        // хитмапе и её 1D срезе показывает, на какую ось смотришь.
        //  cfg_idx == 1 → X-срез → совпадает с вертикальной X-линией хитмапы
        //  cfg_idx == 2 → Y-срез → совпадает с горизонтальной Y-линией
        view.crosshair_x_color = (lslots[i].cfg_idx == 2)
                                  ? 0xFFFF9028u   // orange = Y sweep
                                  : 0xFF50A0FFu;  // blue   = X sweep

        // Перетаскивание креста: MMB или Shift+LMB внутри плота двигает соответствующий fix_*
        // вдоль оси свипа этого среза (cfg_idx 1 → fix_x, 2 → fix_y). Отпускание запускает общий
        // авто-пересчёт (Level 1D + Phase) через last_fix_{x,y}_change_time — тот же дебаунс,
        // что у drag'а L2D-хитмапы и fix-слайдеров, чтобы все три источника вели себя одинаково.
        const bool slice_is_x = (lslots[i].cfg_idx == 1);
        // Snap crosshair drag to this slice's OWN grid nodes — the sampled
        // points the 1D compute actually produced — so the crosshair always
        // lands on a data point rather than drifting between them.
        // Captured by value (fresh lambda per frame, so no lifetime hazard).
        int   snap_n = parse_int_or(slice_is_x ? cs.shared.n_x_1d_text
                                              : cs.shared.n_y_1d_text, 2);
        if (snap_n < 2) snap_n = 2;
        const double snap_lo   = std::min(param_lo, param_hi);
        const double snap_hi   = std::max(param_lo, param_hi);
        const double snap_step = (snap_hi - snap_lo) / (double)(snap_n - 1);
        // При log-сетке узлы не равноудалены — индекс и позицию считаем по
        // логарифму (та же формула, что у движка), иначе крестик садится
        // между реальными точками.
        const bool snap_log = view.x_axis.log_scale && snap_lo > 0.0 && snap_hi > 0.0;
        // Конвенция узлов зависит от того, шёл ли срез continuation'ом: у него
        // своя формула (см. sweep_value_at). Флаг берём общий по вкладке — им
        // же apply_shared_to_bif1d проставляет continuation в под-конфиг.
        const bool snap_cont = cs.shared.continuation_1d_enabled;
        auto snap_to_grid = [snap_lo, snap_hi, snap_step, snap_n, snap_log, snap_cont](double w) {
            if (snap_step <= 0.0) return w;
            int i;
            if (snap_log) {
                if (!(w > 0.0)) return snap_lo;
                const double l0 = std::log10(snap_lo), l1 = std::log10(snap_hi);
                i = (int)std::round((std::log10(w) - l0) / ((l1 - l0) / (double)(snap_n - 1)));
            } else {
                i = (int)std::round((w - snap_lo) / snap_step);
            }
            i = std::clamp(i, 0, snap_n - 1);
            const double s = sweep_value_at(i, snap_n, snap_lo, snap_hi, snap_log,
                                            /*reverse*/ false, snap_cont);
            return std::clamp(s, snap_lo, snap_hi);
        };
        // Тот же snap и для тиков осей / hover-readout, что и в Parametric —
        // раньше в Custom он стоял только на crosshair-драге, поэтому tooltip
        // здесь показывал промежуточные X, которых в данных нет.
        apply_snap_x(view, snap_lo, snap_hi, snap_n);
        view.on_left_drag = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };
        view.on_left_click = [&cs, slice_is_x, snap_to_grid](double world_x) {
            double w = snap_to_grid(world_x);
            if (slice_is_x) {
                cs.shared.fix_x_value = w;
                cs.shared.last_fix_x_change_time = ImGui::GetTime();
            } else {
                cs.shared.fix_y_value = w;
                cs.shared.last_fix_y_change_time = ImGui::GetTime();
            }
        };

        // Build series. Bif = one scatter series with all peak samples;
        // LLE = one line; LS = N line series, one per exponent (parity
        // with draw_ls_plot).
        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        auto& bufs = l1_bufs[i];
        bufs.clear();

        int series_gen = data_gen;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            const auto& r = d.result;
            // Как в Parametric: источник — peak_times при plot_inter_peaks,
            // иначе сами точки бифуркации. Раньше Custom всегда брал
            // bifurcation_points, из-за чего тоггл «inter-peaks» на него не влиял.
            const auto& source = d.plot_inter_peaks ? r.peak_times
                                                    : r.bifurcation_points;
            int n = r.n_pts > 0 ? r.n_pts : 1;
            bufs.emplace_back();
            auto& buf = bufs.back();
            for (int p = 0; p < (int)source.size(); ++p) {
                // Diverged-точки пропускаем (в Parametric так и было; здесь
                // проверки не было, и разошедшиеся точки попадали на график).
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, d.continuation);
                for (double y : source[p]) {
                    if (!std::isfinite(y)) continue;
                    buf.push_back((double)px);
                    buf.push_back((double)y);
                }
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = (int)(buf.size() / 2);
            si.color = ic_base_color(0);
            if (d.custom_point_style) si.color.w = d.point_alpha;
            si.label = d.label.empty() ? "bd" : d.label;
            series_in.push_back(si);
            series_gen = data_gen * 2 + (d.plot_inter_peaks ? 1 : 0);
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = (int)r.lyapunov.size();
            bufs.emplace_back();
            auto& buf = bufs.back();
            buf.reserve(n * 2);
            int total_pts = 0;
            for (int p = 0; p < n; ++p) {
                // Diverged / NaN пропускаем — как в Parametric. Раньше здесь
                // проверок не было, поэтому sentinel-значения расхождения
                // рисовались как выброс.
                if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                double y = r.lyapunov[p];
                if (!std::isfinite(y)) continue;
                double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                           /*reverse*/ false, c.continuation);
                buf.push_back((double)px);
                buf.push_back((double)y);
                ++total_pts;
            }
            PlotSeriesInput si;
            si.points = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color = ic_base_color(0);
            si.label = c.label.empty() ? "LLE" : c.label;
            series_in.push_back(si);
        } else { // LS — N series, one per exponent (parity with draw_ls_plot).
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            const auto& r = c.result;
            int n = r.n_pts;
            int N = r.n_exponents;
            if (N > 0 && n > 0 && (int)r.spectrum.size() == n) {
                bufs.reserve(N);
                for (int j = 0; j < N; ++j) {
                    bufs.emplace_back();
                    auto& buf = bufs.back();
                    buf.reserve(n * 2);
                    int total_pts = 0;
                    for (int p = 0; p < n; ++p) {
                        if (p < (int)r.flags.size() && !regime_is_oscillation(r.flags[p])) continue;
                        if (p >= (int)r.spectrum.size()) continue;
                        const auto& row = r.spectrum[p];
                        if (j >= (int)row.size()) continue;
                        double px = sweep_value_at(p, n, param_lo, param_hi, slice_log,
                                                   /*reverse*/ false, c.continuation);
                        double y = row[j];
                        if (!std::isfinite(y)) continue;
                        buf.push_back((double)px);
                        buf.push_back((double)y);
                        ++total_pts;
                    }
                    PlotSeriesInput si;
                    si.points = buf.empty() ? nullptr : buf.data();
                    si.n_points = total_pts;
                    si.color = ic_base_color(j);
                    // Как в Parametric: "<label> Lj" при N>1, иначе <label>.
                    const std::string base = c.label.empty() ? "LS" : c.label;
                    si.label = (N > 1) ? (base + " L" + std::to_string(j + 1)) : base;
                    series_in.push_back(si);
                }
                series_gen = data_gen * 64 + N;  // rebuild VBO if N changes
            }
        }
        init_vis.assign(series_in.size(), true);
        glob_vis.assign(series_in.size(), true);

        // Right-click "Export data..." — паритет с Parametric, где он есть у
        // всех 1D-графиков. В Custom его не было ни на одном из 6 слотов.
        // Ctrl+T: срез идёт по своей оси общего config'а (слот 1 — X, 2 — Y),
        // по Y — сама считаемая величина, входа в расчёте у неё нет.
        //
        // Писать нужно в ДЕЙСТВУЮЩИЕ поля, а не всегда в sweep_*. При
        // inherit_sweep_from_2d (значение по умолчанию) effective_sweep_x/y
        // читают axis_*, а sweep_* лежат мёртвым грузом: запись туда не меняла
        // ни сам срез, ни 2D-карту, и Ctrl+T выглядел как ничего не делающий.
        // Через axis_* диапазон заодно приезжает на соответствующую ось L2D —
        // это и есть "та же зона" на двумерной диаграмме.
        const bool  vr_slice_is_x = (lslots[i].cfg_idx == 1);
        const bool  vr_inherited  = cs.shared.inherit_sweep_from_2d
                                 && cs.shared.level_2d_enabled;
        const std::vector<ViewRangeTarget> vrt = { vr_target_fixed(
            vr_inherited ? (vr_slice_is_x ? "Shared config (2D X axis)"
                                          : "Shared config (2D Y axis)")
                         : (vr_slice_is_x ? "Shared config (X slice)"
                                          : "Shared config (Y slice)"),
            vr_inherited ? (vr_slice_is_x ? &cs.shared.axis_x_lo_text
                                          : &cs.shared.axis_y_lo_text)
                         : (vr_slice_is_x ? &cs.shared.sweep_x_lo_text
                                          : &cs.shared.sweep_y_lo_text),
            vr_inherited ? (vr_slice_is_x ? &cs.shared.axis_x_hi_text
                                          : &cs.shared.axis_y_hi_text)
                         : (vr_slice_is_x ? &cs.shared.sweep_x_hi_text
                                          : &cs.shared.sweep_y_hi_text)) };
        view.popup_extras = [i, &lslots, &cs, &cb, &vrt, &model, &view]() {
            const bool busy = (lslots[i].kind == L1Kind::Bif) ? cs.bif_session.in_flight
                            : (lslots[i].kind == L1Kind::LLE) ? cs.lle_session.in_flight
                                                              : cs.ls_session.in_flight;
            draw_export_menu_item(busy, cb, [i, &lslots, &cs](const std::string& path) {
                const int ci = lslots[i].cfg_idx;
                if (lslots[i].kind == L1Kind::Bif) {
                    if (ci >= 0 && ci < (int)cs.bif_session.diagrams.size())
                        data_export::export_bif1d(cs.bif_session.diagrams[ci].result, path);
                } else if (lslots[i].kind == L1Kind::LLE) {
                    if (ci >= 0 && ci < (int)cs.lle_session.curves.size())
                        data_export::export_lle1d(cs.lle_session.curves[ci].result, path);
                } else {
                    if (ci >= 0 && ci < (int)cs.ls_session.curves.size())
                        data_export::export_ls1d(cs.ls_session.curves[ci].result, path);
                }
            });
            draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
        };

        // Autofit whenever the underlying result changed (bif/lle/ls each
        // set fit_request in apply_*_result on completion of run_async).
        bool fit = false;
        if (lslots[i].kind == L1Kind::Bif) {
            auto& d = cs.bif_session.diagrams[lslots[i].cfg_idx];
            fit = d.fit_request; if (fit) d.fit_request = false;
        } else if (lslots[i].kind == L1Kind::LLE) {
            auto& c = cs.lle_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        } else {
            auto& c = cs.ls_session.curves[lslots[i].cfg_idx];
            fit = c.fit_request; if (fit) c.fit_request = false;
        }

        ImVec2 avail  = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        view.render(rnd, origin, avail,
                    /*owner_id*/ (0xC10000 + i) ^ sys_owner_delta,
                    series_gen,
                    series_in, init_vis, glob_vis, fit);
        handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
        ImGui::PopID();
        ImGui::End();
    }

    // --- Level 3: phase controls live inside the L3 config panel (see
    // draw_level3_detail). Here we only spawn the projection windows so
    // 2D / 3D / TimeDomain plots dock alongside the other custom plots.
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 0) {
        // Auto-place phase projections into the active tab on first appearance. Suffix + owner_id
        // delta both keep per-system isolation: suffix for imgui.ini dock state, delta for the
        // SHARED PlotRenderer cache (otherwise Rossler's projection 0 saw Chen's cached FBO).
        // Диаграмма признаков рисуется в осях кластеризации того же 2D-конфига, чьи eps/множители
        // стоят в Shared config — иначе по ней нельзя судить, что сольёт dbscan (FeatureClusterParams).
        FeatureClusterParams clust;
        if (!cs.bif_session.diagrams.empty()) {
            const auto& bd = cs.bif_session.diagrams[0];
            clust.valid         = true;
            clust.mult_peak     = parse_ratio_or(bd.mult_peak_text,     (double)mult_peak);
            clust.mult_interval = parse_ratio_or(bd.mult_interval_text, (double)mult_interval);
            clust.eps           = parse_ratio_or(bd.eps_dbscan_text,    0.1);
        }
        draw_projection_windows(cs.phase_session, cb,
            [&](int /*idx*/, const std::string& title) { ensure_docked(title); },
            {}, suffix, sys_owner_delta, {}, clust);
    }
    // Level 3 Basins window (unchanged HeatmapView minimal renderer)
    if (cs.shared.level_phase_enabled && cs.shared.level3_kind == 1) {
        // Автоскейл — как у 2D-карт выше (см. take_fit_2d): сюда тоже уходило
        // жёсткое false, и карта, посчитанная в новой сетке НУ, рисовалась в
        // границах предыдущего прогона.
        auto take_fit_basins = [&cs]() {
            if (cs.basins_session.configs.empty()) return false;
            bool& f = cs.basins_session.configs[0].fit_request;
            if (!f) return false;
            f = false;
            return true;
        };
        std::string basins_title = "Custom Basins" + suffix;
        ensure_docked(basins_title);
        if (ImGui::Begin(basins_title.c_str())) {
                // Basins — HeatmapView of basin_idx (cluster id) — simpler
                // than draw_basins_plot's toolbar (feature switch stays in
                // detail panel), matches user request for a visible result.
                auto& bsn = cs.basins_session;
                if (bsn.in_flight) {
                    ImGui::TextDisabled("Computing basins...");
                } else if (bsn.configs.empty() || !bsn.configs[0].last_run_ok) {
                    ImGui::TextDisabled("No basins result yet.");
                } else {
                    // bsn_hv / bsn_renderer hoisted to top of function for
                    // system-change reset (see the reset block above).
                    const auto& bc = bsn.configs[0];
                    const auto& r  = bc.result;
                    if (r.n_pts <= 0 || r.basin_idx.empty()) {
                        ImGui::TextDisabled("Basins result empty.");
                    } else {
                        // Convert int cluster ids to double for HeatmapView.
                        static std::vector<double> bsn_values;
                        bsn_values.resize(r.basin_idx.size());
                        double vmin =  std::numeric_limits<double>::infinity();
                        double vmax = -std::numeric_limits<double>::infinity();
                        for (size_t k = 0; k < r.basin_idx.size(); ++k) {
                            double v = (double)r.basin_idx[k];
                            bsn_values[k] = v;
                            if (v < vmin) vmin = v;
                            if (v > vmax) vmax = v;
                        }
                        if (!std::isfinite(vmin)) { vmin = 0.0; vmax = 1.0; }
                        // Тот же тулбар, что и у остальных хитмап: раньше здесь был только combo
                        // Colormap, без autoscale/vmin/vmax и Swap axes, хотя диаграмма ровно такая
                        // же. Выбор colormap'а теперь персистится в config (слот 0 = таб "Basins") —
                        // раньше он жил только в static-view и терялся при перезапуске.
                        {
                            auto& bcfg = bsn.configs[0];
                            const int cm = colormap_id_or(bcfg.colormap_idx[0],
                                                          model.basins_colormap);
                            bsn_hv.colormap = (HeatmapColormap)cm;

                            HeatmapToolbarOpts topts;
                            topts.persist_colormap =
                                [&bcfg](int picked) { bcfg.colormap_idx[0] = picked; };
                            draw_heatmap_toolbar(bsn_hv, topts);
                        }
                        bsn_hv.x_axis.name = (bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_x_var] : "x";
                        bsn_hv.y_axis.name = (bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size())
                                              ? cs.vars[bc.axis_y_var] : "y";
                        bsn_hv.cell_regime = regime_cell_label(r.helpful_array, r.n_pts);
                        bsn_hv.value_name = "basin";
                        bsn_hv.value_is_regime_code = true;
                        ImVec2 avail  = ImGui::GetContentRegionAvail();
                        ImVec2 origin = ImGui::GetCursorScreenPos();
                        bsn_hv.render(bsn_renderer, origin, avail,
                                      /*owner_id*/ 0xCBA51E5 ^ sys_owner_delta,
                                      bc.data_generation,
                                      r.n_pts, r.n_pts,
                                      bsn_values.data(),
                                      r.axis_x_lo, r.axis_x_hi,
                                      r.axis_y_lo, r.axis_y_hi,
                                      vmin, vmax,
                                      take_fit_basins());
                    }
                }
            }
        ImGui::End();
    }
}

// Объём работы идущего расчёта (time_estimate.h) — по running-индексу его сессии. Порядок
// проверок тот же, что у цепочки busy в draw_gui: показывается оценка того расчёта, чья подпись
// стоит в индикаторе. Phase, FastSync, Order — без оценки.
[[nodiscard]] TrajWork running_traj_work(const AppModel& m) {
    const CustomSession& cus = m.custom_session;
    if (m.bifurcation_session.in_flight)
        return m.bifurcation_session.traj_work(m.bifurcation_session.running_diagram_index);
    if (m.lle_session.in_flight)     return m.lle_session.traj_work(m.lle_session.running_curve_index);
    if (m.ls_session.in_flight)      return m.ls_session.traj_work(m.ls_session.running_curve_index);
    if (m.metrics_session.in_flight) return m.metrics_session.traj_work(m.metrics_session.running_config_index);
    if (m.dft1d_session.in_flight)   return m.dft1d_session.traj_work(m.dft1d_session.running_config_index);
    if (m.phase_session.in_flight)   return {};
    if (m.basins_session.in_flight)  return m.basins_session.traj_work(m.basins_session.running_config_index);
    if (m.fastsync_session.in_flight || m.order_session.in_flight) return {};
    if (cus.bif_session.in_flight)    return cus.bif_session.traj_work(cus.bif_session.running_diagram_index);
    if (cus.lle_session.in_flight)    return cus.lle_session.traj_work(cus.lle_session.running_curve_index);
    if (cus.ls_session.in_flight)     return cus.ls_session.traj_work(cus.ls_session.running_curve_index);
    if (cus.basins_session.in_flight) return cus.basins_session.traj_work(cus.basins_session.running_config_index);
    return {};
}

// Объём работы по настройкам открытой вкладки: активный конфиг активного анализа.
[[nodiscard]] TrajWork current_tab_traj_work(const AppModel& m) {
    switch (m.app_mode) {
    case AppModel::AppMode::Parametric:
        switch (m.parametric_active_analysis) {
        case 0:  return m.bifurcation_session.traj_work(m.bifurcation_session.active_diagram_index);
        case 1:  return m.lle_session.traj_work(m.lle_session.active_curve_index);
        case 2:  return m.ls_session.traj_work(m.ls_session.active_curve_index);
        case 3:  return m.metrics_session.traj_work(m.metrics_session.active_config_index);
        default: return {};
        }
    case AppModel::AppMode::Dft1D:  return m.dft1d_session.traj_work(m.dft1d_session.active_config_index);
    case AppModel::AppMode::Basins: return m.basins_session.traj_work(m.basins_session.active_config_index);
    default: return {};
    }
}
