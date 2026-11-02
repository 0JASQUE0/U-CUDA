#pragma once
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/analysis/order_session.h"
#include "ucuda/analysis/network_session.h"   // net_coupling_body, NetCouplingLaw: законы связи Neural
#include "ucuda/equations/codegen.hpp"
#include "builtin_cases.h"
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace regression {
struct AnalysisCase { const char* name; bool cpu; bool gpu; };
inline const std::vector<AnalysisCase>& analysis_cases() {
    static const std::vector<AnalysisCase> cases = {
        {"bifurcation_1d", true, true}, {"bifurcation_2d", false, true},
        {"lle_1d", true, true}, {"lle_2d", false, true}, {"ls_1d", true, true}, {"ls_2d", false, true},
        {"dft", true, true}, {"metrics", true, true}, {"basins", false, true}, {"basins_recluster", true, false},
        {"fastsync", false, true}, {"order", true, true}, {"performance", true, true},
        {"stability", true, true}, {"network", false, true}, {"neural", false, true}};
    return cases;
}
template<class R> inline void valid_result(const R& r) {
    if (!r.ok || r.cancelled) throw std::runtime_error("Analysis failed: " + r.error);
}
template<class R> inline void sweep_request(R& q, const std::string& body) {
    q.krs_body = body; q.amountOfX = 2; q.initial_conditions = {1, 1};
    q.base_values = {.5, 1, 4, 1e-12, 25, 0};
    q.n_pts = 4; q.h = .01; q.t_max = 2; q.transient_time = 0;
    q.sweep_over_var = true; q.var_sweep_index = 0; q.param_lo = .9; q.param_hi = 1.1;
}
template<class R> inline void grid_request(R& q, const std::string& body) {
    sweep_request(q, body); q.sweep_over_var_2 = true; q.var_sweep_index_2 = 1;
    q.param_lo_2 = .9; q.param_hi_2 = 1.1;
}
inline std::vector<double> analysis_workload(ParametricEngine& engine, const std::string& name, bool gpu,
                                             double* kernel_seconds = nullptr, bool verify = true,
                                             Scheme method = Scheme::RK4, bool matrix = false) {
    System sys; sys.vars = {"x", "y"}; sys.params = {"p"}; sys.rhs = {"-p*x", "-2*p*y"};
    const auto body = codegen_scheme(sys, method);
    int order = 0; bool symmetric = false;
    if (!builtin_scheme_traits(builtin_display_name(method), &order, &symmetric)) throw std::runtime_error("Unknown analysis method");
    // First-order methods at h=.01 have O(h) bias in the continuous spectrum.
    const double spectrum_tolerance = order == 1 ? .04 : 1e-3;
    const std::vector<double> a = {.5, 1, 4, 1e-12, 25, 0};
    std::vector<double> values;
    if (name == "bifurcation_1d") {
        Bifurcation1DRequest q; sweep_request(q, body); q.use_cpu = !gpu; q.emit_all_samples = true;
        const auto r = engine.run_bifurcation_1d(q); valid_result(r);
        if (verify && (r.flags.size() != 4 || r.bifurcation_points.size() != 4)) throw std::runtime_error("Bifurcation shape");
        for (const auto& row : r.bifurcation_points) values.insert(values.end(), row.begin(), row.end());
    } else if (name == "bifurcation_2d") {
        Bifurcation2DRequest q; grid_request(q, body); const auto r = engine.run_bifurcation_2d(q); valid_result(r);
        values = r.values; if (verify && values.size() != 16) throw std::runtime_error("Bifurcation grid shape");
    } else if (name == "lle_1d") {
        LLE1DRequest q; sweep_request(q, body); q.use_cpu = !gpu; q.NT = .1; q.eps = 1e-4;
        q.sweep_over_var = false; q.param_index = 1; q.param_lo = q.param_hi = 1;
        const auto r = engine.run_lle_1d(q); valid_result(r); values = r.lyapunov;
        if (verify && values.size() != 4) throw std::runtime_error("LLE shape");
        if (verify) for (double v : values) if (v > -.9 || v < -2.1) throw std::runtime_error("LLE outside analytic spectrum");
    } else if (name == "lle_2d") {
        LLE2DRequest q; grid_request(q, body); q.NT = .1; q.eps = 1e-4;
        const auto r = engine.run_lle_2d(q); valid_result(r); values = r.values;
        if (verify && values.size() != 16) throw std::runtime_error("LLE grid shape");
    } else if (name == "ls_1d") {
        LS1DRequest q; sweep_request(q, body); q.use_cpu = !gpu; q.NT = .1; q.eps = 1e-4;
        q.sweep_over_var = false; q.param_index = 1; q.param_lo = q.param_hi = 1;
        const auto r = engine.run_ls_1d(q); valid_result(r);
        if (verify && r.spectrum.size() != 4) throw std::runtime_error("LS shape");
        for (const auto& row : r.spectrum) {
            if (verify && (row.size() != 2 || std::fabs(row[0] + row[1] + 3) > spectrum_tolerance)) throw std::runtime_error("LS trace invariant");
            values.insert(values.end(), row.begin(), row.end());
        }
    } else if (name == "ls_2d") {
        LS2DRequest q; grid_request(q, body); q.NT = .1; q.eps = 1e-4;
        const auto r = engine.run_ls_2d(q); valid_result(r); values = r.values;
        if (verify && values.size() != 32) throw std::runtime_error("LS grid shape");
        if (verify) for (int i = 0; i < 16; ++i) if (std::fabs(values[i] + values[16 + i] + 3) > spectrum_tolerance) throw std::runtime_error("LS grid trace invariant");
    } else if (name == "dft") {
        Dft1DRequest q; sweep_request(q, body); q.use_cpu = !gpu; q.n_freq = 8;
        const auto r = engine.run_dft_1d(q); valid_result(r); values = r.ak_cos;
        if (verify && values.size() != 32) throw std::runtime_error("DFT shape");
        values.insert(values.end(), r.bk_sin.begin(), r.bk_sin.end());
    } else if (name == "metrics") {
        SignalMetricsRequest q; sweep_request(q, body); q.use_cpu = !gpu;
        q.metric_mask = (1 << SIGM_MEAN) | (1 << SIGM_MIN) | (1 << SIGM_MAX);
        const auto r = engine.run_signal_metrics(q); valid_result(r);
        for (int metric : {SIGM_MEAN, SIGM_MIN, SIGM_MAX}) {
            if (verify && r.values[metric].size() != 4) throw std::runtime_error("Metrics shape");
            values.insert(values.end(), r.values[metric].begin(), r.values[metric].end());
        }
        if (verify) for (double v : values) if (v < 0 || v > 1.2) throw std::runtime_error("Decay metric outside analytic range");
    } else if (name == "basins") {
        BasinsRequest q; q.krs_body = body; q.amountOfX = 2; q.initial_conditions = {1, 1}; q.base_values = a;
        q.n_pts = 4; q.h = .01; q.t_max = 2; q.transient_time = 0;
        const auto r = engine.run_basins(q); valid_result(r);
        if (verify && r.basin_idx.size() != 16) throw std::runtime_error("Basins shape");
        std::map<int, int> labels; int next = 1;
        for (int id : r.basin_idx) {
            if (id == 0) values.push_back(0);
            else { if (!labels.count(id)) labels[id] = next++; values.push_back((id < 0 ? -1 : 1) * labels[id]); }
        }
    } else if (name == "basins_recluster") {
        BasinsReclusterRequest q; q.krs_body = body; q.amountOfX = 2; q.n_pts = 4; q.eps_dbscan = .1;
        q.helpful_array.assign(16, 1);
        for (int i = 0; i < 16; ++i) { q.avg_peaks.push_back((i < 8 ? 0 : 1) + .01 * (i % 4)); q.avg_intervals.push_back(i < 8 ? 2 : 4); }
        const auto r = engine.run_basins_recluster(q); valid_result(r);
        if (verify && r.basin_idx.size() != 16) throw std::runtime_error("Recluster shape");
        for (int i = 0; i < 16; ++i) {
            if (verify && (r.basin_idx[i] == r.basin_idx[0]) != (i < 8)) throw std::runtime_error("Recluster known partition");
            values.push_back(i < 8 ? 1 : 2);
        }
    } else if (name == "fastsync") {
        FastSyncRequest q; q.krs_body = body; q.amountOfX = 2; q.values = a;
        q.ic_master = {1, 1}; q.ic_slave = {1.01, 1.01}; q.k_forward = {1, 1}; q.k_backward = {1, 1};
        q.h = .01; q.t_max = .1; q.window = .05; q.iter_of_synchr = 2; q.transient_time = 0;
        const auto r = engine.run_fastsync(q); valid_result(r); values = r.sync_error;
    } else if (name == "network") {
        NetworkRequest q; q.krs_body = body; q.amountOfX = 2; q.n_nodes = 16; q.amountOfValues = (int)a.size();
        q.h = .01; q.t_max = 1; q.transient = 0; q.pre_scaller = 10;
        if (kernel_seconds) {
            // One substantial launch makes event timing insensitive to tiny-kernel overhead.
            q.n_nodes = 256; q.h = .0001; q.pre_scaller = 1000; q.steps_per_launch = 10000;
        }
        q.edge_start.resize(q.n_nodes + 1, 0);
        q.measure_gpu_time = kernel_seconds != nullptr;
        for (int i = 0; i < q.n_nodes; ++i) { q.values.insert(q.values.end(), a.begin(), a.end()); q.initial_conditions.insert(q.initial_conditions.end(), {1, 1}); }
        const auto r = engine.run_network(q); valid_result(r); values = r.data;
        if (verify && (r.n_diverged || r.n_points < 2 || r.n_nodes != q.n_nodes || (kernel_seconds && r.launch_count != 1)))
            throw std::runtime_error("Network shape/divergence/launch count");
        const double network_tolerance = order == 1 ? .003 : order == 2 ? 2e-5 : 1e-7;
        if (verify && std::fabs(values[values.size() - 2] - std::exp(-1.0)) > network_tolerance) throw std::runtime_error("Network analytic solution");
        if (verify && kernel_seconds) {
            const auto offset = values.size() - (size_t)q.n_nodes * 2;
            for (int node = 0; node < q.n_nodes; ++node)
                if (std::fabs(values[offset + node * 2] - std::exp(-1.0)) > network_tolerance ||
                    std::fabs(values[offset + node * 2 + 1] - std::exp(-2.0)) > network_tolerance)
                    throw std::runtime_error("Network batched analytic solution");
        }
        if (kernel_seconds) *kernel_seconds = r.gpu_ms / 1000;
    } else if (name == "neural") {
        // Несвязанные узлы (дуг нет): каждый узел — та же линейная релаксация, расписание stage всегда RK4.
        NeuralRequest q; q.schedule = NeuralSchedule::Stage; q.krs_body = body; q.rhs_body = codegen_rhs_body(sys);
        q.coupling_body = net_coupling_body(sys, {NetCouplingLaw{}});
        q.amountOfX = 2; q.n_nodes = 64; q.amountOfValues = (int)a.size();
        q.h = .01; q.t_max = 1; q.transient = 0; q.pre_scaller = 10; q.keep_final_state = true;
        q.edge_start.assign(q.n_nodes + 1, 0);
        for (int i = 0; i < q.n_nodes; ++i) { q.values.insert(q.values.end(), a.begin(), a.end()); q.initial_conditions.insert(q.initial_conditions.end(), {1, 1}); }
        q.observe.nodes = {0, q.n_nodes - 1}; q.observe.vars = {0, 1}; q.observe.pop_start = {0, q.n_nodes};
        const auto r = engine.run_neural(q); valid_result(r);
        if (verify && (r.n_diverged || r.n_points < 2 || r.n_nodes != q.n_nodes || r.final_state.size() != (size_t)q.n_nodes * 2))
            throw std::runtime_error("Neural shape/divergence");
        if (verify)
            for (int node = 0; node < q.n_nodes; ++node)
                if (std::fabs(r.final_state[node * 2] - std::exp(-1.0)) > 1e-7 || std::fabs(r.final_state[node * 2 + 1] - std::exp(-2.0)) > 1e-7)
                    throw std::runtime_error("Neural analytic solution");
        values = r.final_state; values.insert(values.end(), r.means.begin(), r.means.end());
    } else if (name == "order" || name == "performance" || name == "stability") {
        const bool stability = name == "stability";
        SystemRecord record; record.vars_text = "x, y"; record.params_text = stability ? "a, b, c, d" : "p";
        record.init_conditions = {{"x", "1"}, {"y", "1"}}; record.param_values = {{"p", "1"}};
        OrderAnalysisSession session; session.load_from_record(record, sys.vars, stability ? std::vector<std::string>{"a", "b", "c", "d"} : sys.params);
        session.sys = sys;
        if (stability) { session.sys.params = {"a", "b", "c", "d"}; session.sys.rhs = {"a*x+b*y", "c*x+d*y"}; }
        if (session.configs.empty()) session.add_config();
        auto& c = session.configs.front(); c.scheme = "Euler"; c.use_gpu = gpu;
        if (matrix) c.scheme = builtin_display_name(method);
        c.calc_kind = stability ? kOrderCalcStab : name == "performance" ? kOrderCalcPerf : kOrderCalcOrder;
        c.axis_x_lo_text = "0.01"; c.axis_x_hi_text = "0.02"; c.axis_x_n_text = "3"; c.t_max_text = "1";
        c.perf_repeats_text = "3"; c.perf_warmup_text = "1";
        if (!session.run_async(engine, 0)) throw std::runtime_error(c.last_error);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
        while (!session.poll()) {
            if (std::chrono::steady_clock::now() > deadline) { session.request_cancel(); throw std::runtime_error("Order timeout"); }
            std::this_thread::yield();
        }
        if (stability) { if (!c.stab_last_run_ok) throw std::runtime_error(c.last_error); values = c.stab_result.rho; }
        else if (name == "performance") { if (!c.perf_last_run_ok) throw std::runtime_error(c.last_error); values = c.perf_result.e1; }
        else { if (!c.last_run_ok) throw std::runtime_error(c.last_error);
            if (matrix) { values = c.result.e1; for (double e : values) if (e < 0) throw std::runtime_error("Negative convergence error"); }
            else { values = c.result.p;
                if (verify) for (double p : values) if (std::fabs(p - 1) > .1) throw std::runtime_error("Euler convergence order"); }
        }
    } else throw std::runtime_error("Unknown analysis scenario");
    if (verify && values.empty()) throw std::runtime_error("Analysis produced no testable values: " + name);
    if (verify) for (double v : values) if (!std::isfinite(v)) throw std::runtime_error("Non-finite analysis result: " + name);
    return values;
}
}
