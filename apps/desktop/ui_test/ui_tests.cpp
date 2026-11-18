// Compiled UI tests of the desktop app. Run: U-CUDA.exe --ui-test all (or a name filter).
// Add a test: IM_REGISTER_TEST(engine, "<category>", "<name>")->TestFunc = [](ImGuiTestContext* ctx) { ... };
// Item paths start at a window: "//MainHost/Phase analysis"; "**/Label" searches any depth.
#include "ui_test_host.h"

#include "ui_test_modes.h"

#include "imgui.h"
#include "imgui_te_engine.h"
#include "imgui_te_context.h"

#include "ucuda/application/app_model.h"
#include "ucuda/domain/configCUDA.h"
#include "ucuda/domain/num_parse.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <vector>

namespace ui_test {

void register_sweep_tests(ImGuiTestEngine* engine);   // ui_tests_sweep.cpp

void register_tests(ImGuiTestEngine* engine) {
    ImGuiTest* t = nullptr;

    t = IM_REGISTER_TEST(engine, "smoke", "startup");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->Yield(5);
        IM_CHECK(ctx->WindowInfo("//MainHost").Window != nullptr);
        IM_CHECK(ctx->ItemExists("//MainHost/Library"));
        IM_CHECK(ctx->ItemExists("//MainHost/Settings"));
    };

    // Every visible mode button switches AppModel::app_mode and the next frames draw without errors.
    t = IM_REGISTER_TEST(engine, "smoke", "mode_tabs");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* model = app_model();
        IM_CHECK_NO_RET(model != nullptr);
        if (!model) return;
        const AppModel::AppMode initial = model->app_mode;
        ctx->SetRef("//MainHost");
        for (const ModeButton& b : kModeButtons) {
            if (!ctx->ItemExists(b.label)) {
                ctx->LogInfo("mode button '%s' hidden, skipped", b.label);
                continue;
            }
            ctx->ItemClick(b.label);
            ctx->Yield(3);
            IM_CHECK_EQ((int)model->app_mode, (int)b.mode);
        }
        for (const ModeButton& b : kModeButtons)
            if (b.mode == initial && ctx->ItemExists(b.label)) ctx->ItemClick(b.label);
    };

    // Parametric/Bifurcation "Param lo" -> "Apply to..." -> All -> "Apply to N selected":
    // every bifurcation config (active or not) receives the value. Needs a system with
    // two bifurcation configs (Chen: run with a script that loads it).
    t = IM_REGISTER_TEST(engine, "parametric", "apply_param_lo_to_all");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        ctx->SetRef("//MainHost");
        ctx->ItemClick("Parametric");
        ctx->Yield(3);
        ctx->SetRef("//Parametric Controls");
        auto log_configs = [&](const char* when) {
            for (size_t i = 0; i < m->bifurcation_session.diagrams.size(); ++i) {
                const auto& d = m->bifurcation_session.diagrams[i];
                ctx->LogInfo("[%s] bif config %d \"%s\": lo=%s hi=%s 2d=%d", when, (int)i, d.label.c_str(),
                             d.param_lo_text.c_str(), d.param_hi_text.c_str(), (int)d.mode_2d);
            }
        };
        log_configs("before");
        ctx->ItemInputValue("**/Param lo", "31");
        ctx->ItemClick("**/Param lo", ImGuiMouseButton_Right);
        ctx->Yield(2);
        // The "active config of each tab" item is gone: targets are chosen only in "Apply to...".
        ImGuiTestItemList items;
        ctx->GatherItems(&items, "//##Popup_a4503e11", 1);
        for (const ImGuiTestItemInfo& it : items)
            IM_CHECK(std::strncmp(it.DebugLabel, "Apply to active", 15) != 0 &&
                     std::strncmp(it.DebugLabel, "Apply to all calc", 17) != 0);
        ctx->ItemClick("//##Popup_a4503e11/Apply to...");
        ctx->Yield(2);
        ctx->ItemClick("//###Menu_00/All");
        ctx->Yield(2);
        items.Clear();
        ctx->GatherItems(&items, "//###Menu_00", 1);
        ImGuiID apply_id = 0;
        for (const ImGuiTestItemInfo& it : items)
            if (std::strncmp(it.DebugLabel, "Apply to ", 9) == 0) { apply_id = it.ID; ctx->LogInfo("button: %s", it.DebugLabel); }
        IM_CHECK_NO_RET(apply_id != 0);
        if (apply_id) ctx->ItemClick(apply_id);
        ctx->Yield(3);
        log_configs("after");
        const auto& ds = m->bifurcation_session.diagrams;
        IM_CHECK_NO_RET(ds.size() >= 2);
        for (const auto& d : ds) IM_CHECK_STR_EQ(d.param_lo_text.c_str(), "31");
    };

    // Basins -> "Run portraits": the IC of every portrait must be bit for bit the start point the
    // map kernel used for that cell (BasinsResult::axis_*_nodes, computed by the same module).
    // Linear and ±log grids. Also checks that the host node formula now equals the kernel's.
    // Needs a loaded ODE system (Chen) and a GPU.
    t = IM_REGISTER_TEST(engine, "basins", "portrait_ic_bitwise");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        ctx->SetRef("//MainHost");
        ctx->ItemClick("Basins");   // the tab loads its session from the system on first visit
        ctx->Yield(10);
        BasinsAnalysisSession& s = m->basins_session;
        ctx->LogInfo("[basins] configs %d, vars %d", (int)s.configs.size(), (int)s.vars.size());
        IM_CHECK_NO_RET(!s.configs.empty() && s.vars.size() >= 2);
        if (s.configs.empty() || s.vars.size() < 2) return;
        auto wait = [&](auto done, int max_frames) {
            for (int f = 0; f < max_frames && !done(); ++f) { ctx->Yield(); ctx->SleepNoSkip(0.01f, 0.01f); }
            return done();
        };
        auto bits_eq = [](double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; };
        for (int symlog = 0; symlog < 2; ++symlog) {
            BasinsConfig& c = s.configs[0];
            c.n_pts_text = "120";
            c.t_max_text = "50"; c.transient_text = "20";
            c.pp_t_max_text = "5"; c.pp_transient_text = "0";
            c.pp_max_attractors_text = "8";
            c.pp_use_gpu = true;
            c.axis_x_symlog = c.axis_y_symlog = (symlog != 0);
            c.axis_x_lo_text = symlog ? "0.01" : "-17.3"; c.axis_x_hi_text = symlog ? "20" : "13.1";
            c.axis_y_lo_text = symlog ? "0.02" : "-9.7";  c.axis_y_hi_text = symlog ? "25" : "21.9";
            IM_CHECK_NO_RET(m->run_basins(0));
            IM_CHECK_NO_RET(wait([&] { return !s.in_flight; }, 6000));
            IM_CHECK_NO_RET(c.last_run_ok);
            if (!c.last_run_ok) { ctx->LogError("basins: %s", c.last_error.c_str()); return; }
            const BasinsResult& r = c.result;
            const int n = r.n_pts;
            IM_CHECK_EQ_NO_RET((int)r.axis_x_nodes.size(), n);
            IM_CHECK_EQ_NO_RET((int)r.axis_y_nodes.size(), n);
            if ((int)r.axis_x_nodes.size() != n || (int)r.axis_y_nodes.size() != n) return;
            // Host formula vs kernel nodes: both grids must match exactly (linear — explicit fma,
            // ±log — ucuda_log10_det / ucuda_exp10_det instead of the library log10/pow).
            int host_diff = 0, old_diff = 0;
            for (int k = 0; k < n; ++k) {
                const double t = (double)k / (double)(n - 1);
                const double hx = symlog ? (double)ucuda_node_value_symlog(k, n, r.axis_x_lo, r.axis_x_hi)
                                         : (double)ucuda_node_value(k, n, r.axis_x_lo, r.axis_x_hi);
                if (!bits_eq(hx, r.axis_x_nodes[(size_t)k])) ++host_diff;
                if (!symlog && !bits_eq((1.0 - t) * r.axis_x_lo + t * r.axis_x_hi, r.axis_x_nodes[(size_t)k])) ++old_diff;
            }
            ctx->LogInfo("[basins] %s: host formula differs from kernel in %d/%d X nodes (old lerp copy: %d)",
                         symlog ? "±log" : "linear", host_diff, n, old_diff);
            IM_CHECK_EQ(host_diff, 0);
            // Portraits.
            BasinsPhaseSlot& sl = s.phase_slot(0);
            s.request_phase_run(0);
            IM_CHECK_NO_RET(wait([&] { return s.phase_queue.empty() && !sl.phase.in_flight; }, 6000));
            IM_CHECK_NO_RET(!sl.cells.empty() && sl.cells.size() == sl.phase.ic_sets.size());
            const std::string& vx = s.vars[(size_t)r.axis_x_var];
            const std::string& vy = s.vars[(size_t)r.axis_y_var];
            int mismatches = 0;
            for (size_t k = 0; k < sl.cells.size(); ++k) {
                const int cell = sl.cells[k], ix = cell % n, iy = cell / n;
                const auto& vals = sl.phase.ic_sets[k].values;
                const double x = parse_num(vals.at(vx), 0.0), y = parse_num(vals.at(vy), 0.0);
                const bool ok = bits_eq(x, r.axis_x_nodes[(size_t)ix]) && bits_eq(y, r.axis_y_nodes[(size_t)iy]);
                if (!ok) ++mismatches;
                ctx->LogInfo("[basins] %s cell (%d,%d) basin %s: x=%s y=%s %s", symlog ? "±log" : "linear", ix, iy,
                             sl.phase.ic_sets[k].label.c_str(), vals.at(vx).c_str(), vals.at(vy).c_str(),
                             ok ? "== kernel" : "!= KERNEL");
            }
            IM_CHECK_EQ(mismatches, 0);
            IM_CHECK(sl.has_result);
        }
        // Back to the linear defaults the system had (the session is the isolated copy).
        BasinsConfig& c = s.configs[0];
        c.axis_x_symlog = c.axis_y_symlog = false;
    };

    // Diagnostic: the same basins config computed by a Tree leaf and by the Basins tab, twice each
    // (the first run includes NVRTC). Logs wall time, frame count and frame time while running;
    // the engine prints its own compile/sim/DBSCAN split.
    t = IM_REGISTER_TEST(engine, "tree", "basins_speed");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        ctx->SetRef("//MainHost");
        ctx->ItemClick("Basins");   // loads the Basins session (reference config)
        ctx->Yield(10);
        ctx->ItemClick("Tree");
        ctx->Yield(10);
        TreeSession& ts = m->tree_session;
        IM_CHECK_NO_RET(!ts.tree.nodes.empty());
        if (ts.tree.nodes.empty()) return;
        int leaf = 0;
        for (const auto& n : ts.tree.nodes) if (n.kind == TreeNodeKind::Basins) { leaf = n.id; break; }
        const bool own_tree = (leaf == 0);   // an existing leaf (user's session) is measured as is
        if (!leaf) {   // Basins lives under a Point
            int point = 0;
            for (const auto& n : ts.tree.nodes) if (n.kind == TreeNodeKind::Point) { point = n.id; break; }
            if (!point) point = ts.add_node(ts.tree.nodes.front().id, TreeNodeKind::Point);
            leaf = ts.add_node(point, TreeNodeKind::Basins);
        }
        ts.ensure_leaf_runtimes();
        IM_CHECK_NO_RET(leaf != 0 && ts.leaves.count(leaf) && ts.leaves[leaf].basins);
        if (!leaf || !ts.leaves[leaf].basins) return;
        BasinsAnalysisSession& tb = *ts.leaves[leaf].basins;
        if (own_tree) { tb.configs[0].n_pts_text = "300"; tb.configs[0].pp_autorun = false; }
        ctx->LogInfo("[speed] leaf pp_autorun %d, pp tt %s ct %s, max attractors %s", (int)tb.configs[0].pp_autorun,
                     tb.configs[0].pp_transient_text.c_str(), tb.configs[0].pp_t_max_text.c_str(),
                     tb.configs[0].pp_max_attractors_text.c_str());
        const BasinsConfig tab_own = m->basins_session.configs.empty() ? BasinsConfig{} : m->basins_session.configs[0];
        // Integration of the Basins tab defaults (Chen: two basins), so both sides do real work.
        if (own_tree && !m->basins_session.configs.empty()) {
            const BasinsConfig& ref = m->basins_session.configs[0];
            for (auto& n : ts.tree.nodes) {
                CustomTabSharedConfig& c = n.cfg;
                c.scheme = ref.scheme; c.symmetry_s = ref.symmetry_s; c.cd_fp_iters = ref.cd_fp_iters;
                c.newton = ref.newton; c.h_text = ref.h_text; c.adaptive = ref.adaptive;
                c.transient_text = "50"; c.t_max_text = "100";
                c.max_value_text = ref.max_value_text; c.pre_scaller_text = ref.pre_scaller_text;
                c.initial_conditions = ref.initial_conditions; c.param_values = ref.param_values;
                // The point pins the two region axes (Chen: a, b) to its cursor: put it on the tab's values.
                auto pv = [&](const char* p) { auto it = ref.param_values.find(p); return it == ref.param_values.end() ? 0.0 : parse_num(it->second, 0.0); };
                c.fix_x_value = pv("a"); c.fix_y_value = pv("b");
                c.axis_x_lo_text = "0"; c.axis_x_hi_text = "60";   // clamp_cursors keeps them inside
                c.axis_y_lo_text = "0"; c.axis_y_hi_text = "10";
                n.own_integration = false;
            }
        }
        ctx->LogInfo("[speed] tree leaf %d, scheme %s h %s tt %s ct %s", leaf, ts.tree.nodes.front().cfg.scheme.c_str(),
                     ts.tree.nodes.front().cfg.h_text.c_str(), ts.tree.nodes.front().cfg.transient_text.c_str(),
                     ts.tree.nodes.front().cfg.t_max_text.c_str());
        using clk = std::chrono::steady_clock;
        // A user tree: compute every other leaf first, so the Tree tab draws what it normally shows.
        auto fill_tree = [&] {
            std::vector<int> others;
            for (int id : ts.dirty_leaves(ts.tree.nodes.front().id)) if (id != leaf) others.push_back(id);
            ctx->LogInfo("[speed] computing %d other leaves first", (int)others.size());
            m->enqueue_tree(others);
            for (int f = 0; f < 2000000 && (ts.any_in_flight() || !m->tree_queue().empty()); ++f) ctx->Yield();
        };
        auto measure = [&](const char* what, auto start, auto busy) {
            const auto t0 = clk::now();
            start();
            int frames = 0;
            double worst = 0.0;
            auto prev = clk::now();
            for (; frames < 200000; ++frames) {
                ctx->Yield();
                const auto now = clk::now();
                worst = std::max(worst, std::chrono::duration<double>(now - prev).count());
                prev = now;
                if (!busy()) break;
            }
            const double sec = std::chrono::duration<double>(clk::now() - t0).count();
            const double ms = frames ? 1000.0 * sec / frames : 0.0;
            ctx->LogInfo("[speed] %s: %.3f s wall, %d frames (%.1f ms/frame, worst %.1f ms)", what, sec, frames,
                         ms, 1000.0 * worst);
            return ms;
        };
        if (!own_tree) {
            const auto t0 = clk::now();
            fill_tree();
            ctx->LogInfo("[speed] other leaves: %.1f s", std::chrono::duration<double>(clk::now() - t0).count());
        }
        // Basins portraits (pp_autorun) start ~0.3 s after the map and run beside the next one.
        auto portraits_busy = [&] {
            for (const auto& sl : tb.phase_slots) if (sl && sl->phase.in_flight) return true;
            return !tb.phase_queue.empty();
        };
        auto wait_portraits = [&](const char* what) {
            const auto t0 = clk::now();
            while (std::chrono::duration<double>(clk::now() - t0).count() < 1.0) ctx->Yield();
            int f = 0;
            for (; f < 2000000 && portraits_busy(); ++f) ctx->Yield();
            ctx->LogInfo("[speed] %s: portraits done %.2f s after the map (%d frames), %d trajectories", what,
                         std::chrono::duration<double>(clk::now() - t0).count(), f,
                         tb.phase_slots.empty() ? 0 : (int)tb.phase_slots[0]->phase.result.trajectories.size());
        };
        for (int rep = 0; rep < 3; ++rep) {
            const char* names[] = { "tree run 1", "tree run 2", "tree run 2b" };
            const double ms = measure(names[rep],
                    [&] { m->enqueue_tree({leaf}); },
                    [&] { return ts.any_in_flight() || !m->tree_queue().empty(); });
            // The main loop throttles rendering while a GPU job runs (app_main.cpp, ~10 FPS), so
            // the window does not steal the GPU from the kernel. Tree was missing from that list:
            // frames came every 1.5-3 ms, and on a full-screen layout basins took ~3x longer.
            if (rep > 0) IM_CHECK_GT(ms, 50.0);
            wait_portraits(names[rep]);
        }
        const BasinsConfig leaf_cfg = tb.configs[0];
        for (const auto& kv : leaf_cfg.param_values) ctx->LogInfo("[speed] leaf param %s = %s", kv.first.c_str(), kv.second.c_str());
        for (const auto& kv : leaf_cfg.initial_conditions) ctx->LogInfo("[speed] leaf ic %s = %s", kv.first.c_str(), kv.second.c_str());
        for (const auto& kv : m->basins_session.configs[0].param_values) ctx->LogInfo("[speed] tab param %s = %s", kv.first.c_str(), kv.second.c_str());
        ctx->LogInfo("[speed] leaf axes %d [%s,%s] %d [%s,%s], max %s, clusters %d", leaf_cfg.axis_x_var,
                     leaf_cfg.axis_x_lo_text.c_str(), leaf_cfg.axis_x_hi_text.c_str(), leaf_cfg.axis_y_var,
                     leaf_cfg.axis_y_lo_text.c_str(), leaf_cfg.axis_y_hi_text.c_str(), leaf_cfg.max_value_text.c_str(),
                     leaf_cfg.result.n_clusters);
        ctx->LogInfo("[speed] leaf cfg: scheme %s h %s tt %s ct %s n %s adaptive %d, ok %d",
                     leaf_cfg.scheme.c_str(), leaf_cfg.h_text.c_str(), leaf_cfg.transient_text.c_str(),
                     leaf_cfg.t_max_text.c_str(), leaf_cfg.n_pts_text.c_str(), (int)leaf_cfg.adaptive.enabled,
                     (int)leaf_cfg.last_run_ok);
        ctx->SetRef("//MainHost");
        ctx->ItemClick("Basins");
        ctx->Yield(10);
        BasinsAnalysisSession& s = m->basins_session;
        IM_CHECK_NO_RET(!s.configs.empty());
        if (s.configs.empty()) return;
        s.configs[0] = leaf_cfg;
        s.configs[0].pp_autorun = false;
        for (int rep = 0; rep < 2; ++rep)
            measure(rep ? "basins tab run 2" : "basins tab run 1",
                    [&] { m->run_basins(0); },
                    [&] { return s.in_flight; });
        // Tree once more with the Tree tab hidden: separates GUI cost from compute.
        measure("tree run 3 (Basins tab shown)",
                [&] { m->enqueue_tree({leaf}); },
                [&] { return ts.any_in_flight() || !m->tree_queue().empty(); });
        // The tab's own config (its parameters / axes / IC) for comparison.
        s.configs[0] = tab_own;
        s.configs[0].pp_autorun = false;
        ctx->LogInfo("[speed] tab own cfg: scheme %s h %s tt %s ct %s n %s", tab_own.scheme.c_str(), tab_own.h_text.c_str(),
                     tab_own.transient_text.c_str(), tab_own.t_max_text.c_str(), tab_own.n_pts_text.c_str());
        measure("basins tab, its own config",
                [&] { m->run_basins(0); },
                [&] { return s.in_flight; });
    };

    // While any tab computes on the GPU, the main loop must throttle rendering (~10 FPS, see
    // AppModel::any_gpu_job_in_flight): a window drawing at full speed shares the GPU with the kernel
    // (Tree basins took ~3x longer). Starts each tab's job and checks the frame time while it runs.
    // Jobs that cannot start on the loaded system, or end within 3 frames, are reported as skipped.
    t = IM_REGISTER_TEST(engine, "throttle", "all_tabs");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        ctx->SetRef("//MainHost");
        // DFT, FastSync, Order, Network and Neural are hidden by default: show every tab.
        const std::vector<std::string> hidden_before = m->hidden_tabs;
        m->hidden_tabs.clear();
        ctx->Yield(3);
        for (const ModeButton& b : kModeButtons) {  // every tab loads its session on the first visit
            if (ctx->ItemExists(b.label)) { ctx->ItemClick(b.label); ctx->Yield(5); }
            else ctx->LogInfo("[throttle] tab button '%s' hidden", b.label);
        }
        auto why = [](const auto& s) -> std::string {
            if (s.configs.empty()) return "no configs";
            return "configs " + std::to_string(s.configs.size()) + ", last_error '" + s.configs[0].last_error + "'";
        };
        ctx->LogInfo("[throttle] DFT: %s", why(m->dft1d_session).c_str());
        ctx->LogInfo("[throttle] FastSync: %s", why(m->fastsync_session).c_str());
        ctx->LogInfo("[throttle] Order: %s", why(m->order_session).c_str());
        ctx->LogInfo("[throttle] Network: %s", why(m->network_session).c_str());
        ctx->LogInfo("[throttle] Neural: %s", why(m->neural_session).c_str());
        using clk = std::chrono::steady_clock;
        struct Job { const char* name; std::function<bool()> start; };
        std::vector<Job> jobs = {
            { "Phase",       [&] { return m->phase_session.recompute_async(); } },
            { "Bifurcation", [&] { return m->run_bifurcation(0); } },
            { "LLE",         [&] { return m->run_lle(0); } },
            { "LS",          [&] { return m->run_ls(0); } },
            { "Metrics",     [&] { return m->run_metrics(0); } },
            { "DFT 1D",      [&] { return m->run_dft1d(0); } },
            { "Basins",      [&] { return m->run_basins(0); } },
            { "FastSync",    [&] { return m->run_fastsync(0); } },
            { "Order",       [&] { return m->run_order(0); } },
            { "Network",     [&] { return m->run_network(0); } },
            { "Neural",      [&] { return m->run_neural(0); } },
            { "Custom",      [&] { m->enqueue_custom_level_2d(); return true; } },
        };
        int checked = 0;
        for (const Job& j : jobs) {
            for (int f = 0; f < 600 && m->any_gpu_job_in_flight(); ++f) ctx->Yield();   // previous job's tail
            if (!j.start()) { ctx->LogInfo("[throttle] %s: did not start (skipped)", j.name); continue; }
            const auto t0 = clk::now();
            int frames = 0;
            ctx->Yield();   // the queue starts the job on the next tick
            while (m->any_gpu_job_in_flight() && frames < 100000) { ctx->Yield(); ++frames; }
            const double sec = std::chrono::duration<double>(clk::now() - t0).count();
            if (frames < 3) { ctx->LogInfo("[throttle] %s: %.2f s, %d frames - too short (skipped)", j.name, sec, frames); continue; }
            const double ms = 1000.0 * sec / frames;
            ctx->LogInfo("[throttle] %s: %.2f s, %d frames, %.1f ms/frame", j.name, sec, frames, ms);
            IM_CHECK_GT(ms, 50.0);
            ++checked;
        }
        ctx->LogInfo("[throttle] checked %d tabs", checked);
        IM_CHECK_GT(checked, 3);
        m->hidden_tabs = hidden_before;
    };

    register_sweep_tests(engine);
}

} // namespace ui_test
