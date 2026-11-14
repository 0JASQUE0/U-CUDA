#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_fields.h"
#include "gui_plot.h"
#include "tab_custom.h"
#include "tab_phase.h"
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 AppModel& model) {
    hv.crosshair_x = cs.shared.level_1d_enabled ? cs.shared.fix_x_value
                                                : std::numeric_limits<double>::quiet_NaN();
    hv.crosshair_y = cs.shared.level_1d_enabled ? cs.shared.fix_y_value
                                                : std::numeric_limits<double>::quiet_NaN();
    // Drag callback: fires every frame while LMB is held inside the plot (HeatmapView
    // on_left_drag). Cheap — just moves the crosshair via fix_x/fix_y so the shared crosshair on
    // every 2D window follows the cursor. param_values update is deferred to on_left_click on
    // release; Phase/Basins enqueue — to the settled branch in draw_custom_controls (or to
    // on_left_click when the 1D level is off and that branch never runs).
    hv.on_left_drag = [&cs](int, int, double snap_x, double snap_y) {
        cs.shared.fix_x_value = snap_x;
        cs.shared.fix_y_value = snap_y;
        // Heatmap drag moves BOTH axes → bump both timers so the settled
        // branch re-runs X-slices AND Y-slices.
        double t = ImGui::GetTime();
        cs.shared.last_fix_x_change_time = t;
        cs.shared.last_fix_y_change_time = t;
    };
    hv.on_left_click = [&cs, &model](int, int, double snap_x, double snap_y) {
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
        // При включённом 1D ФП/бассейны ставит ветка автопересчёта в draw_custom_controls по
        // сдвинутым выше таймерам; второй enqueue здесь гонял расчёт дважды в той же точке.
        if (!s.level_1d_enabled && s.autorun_on_drilldown && s.level_phase_enabled) {
            model.enqueue_custom({ s.level3_kind == 0 ? CustomQueueItem::Kind::Phase
                                             : CustomQueueItem::Kind::Basins });
            cs.arm_auto_recompute_signatures(false, false, true);
        }
    };
}
}

namespace ucuda::ui_detail {
[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) {
    ImGuiID sys_hash = sys.empty() ? 0u : ImHashStr(sys.c_str());
    return (ImGuiID)0xD5000000u ^ (ImGuiID)tab_id ^ sys_hash;
}
}

namespace ucuda::ui_detail {
[[nodiscard]] std::string custom_win_suffix(const std::string& sys) {
    return sys.empty() ? std::string{} : ("##sys_" + sys);
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void draw_custom_controls(AppModel& model, SystemLibrary& lib) {
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
                    [&](){ model.enqueue_custom_level_2d(); });
        try_enqueue(cs.sig_l1d, build_l1d_signature(c, cs),
                    [&](){ model.enqueue_custom_level_1d(); });
        try_enqueue(cs.sig_l3,  build_l3_signature (c, cs),
                    [&](){ model.enqueue_custom_level_3(); });
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop / clear queue")) {
        model.custom_session.request_cancel_all();
        model.clear_custom_queue();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu queued)", model.custom_queue().size());

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
        bool queue_free = model.custom_queue().empty();
        if ((x_settled || y_settled) && queue_free) {
            if (c.auto_recompute_1d) {
                // fix_x moved → Y-slices depend on fix_x. fix_y → X-slices.
                model.enqueue_custom_level_1d_partial(
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
            const bool run_l3 = c.autorun_on_drilldown && c.level_phase_enabled;
            if (run_l3) {
                model.enqueue_custom({ c.level3_kind == 0
                    ? CustomQueueItem::Kind::Phase
                    : CustomQueueItem::Kind::Basins });
            }
            cs.arm_auto_recompute_signatures(c.auto_recompute_1d && y_settled,
                                             c.auto_recompute_1d && x_settled, run_l3);
            // Нечего считать (у сдвинутой оси нет включённых срезов) — коммит иначе ждал бы
            // чужого завершения расчёта.
            if (model.custom_queue().empty()) cs.commit_pending_signatures();
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
}
