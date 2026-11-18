#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/cuda/circuit_nvrtc.h"
#include "ucuda/platform/platform_paths.h"
#include <cuda.h>
#include <cmath>
#include <cstring>
#include <iostream>

std::string exe_dir() { return platform::executable_dir(); }
int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cout << "No CUDA device available; skipping GPU test\n";
        return 77;
    }
    NvrtcEngine engine;
    if (!engine.compile("X[0] += h * a[0];", 1)) {
        std::cerr << engine.error() << '\n'; return 1;
    }
    std::vector<std::vector<std::vector<double>>> out;
    if (!engine.run_phase_portraits({1.0}, 1, {2.0}, 0.25, 4, 0, out)) {
        std::cerr << engine.error() << '\n'; return 1;
    }
    if (out.size() != 1 || out[0].size() != 4 || std::abs(out[0][3][0] - 2.5) > 1e-12) {
        std::cerr << "Incorrect GPU trajectory\n"; return 1;
    }
    // The complex path loads configCUDA.h from the runtime kernels directory.
    if (!engine.compile("ucmplx z(X[0], 0); X[0] = z.re + h * a[0];", 1)) {
        std::cerr << engine.error() << '\n'; return 1;
    }
    ParametricEngine sweep;
    Bifurcation1DRequest request;
    request.krs_body = "X[0] += h * (a[0] - X[0]);";
    request.amountOfX = 1;
    request.initial_conditions = {0.0};
    request.base_values = {0.5};
    request.n_pts = 2;
    request.h = 0.01;
    request.t_max = 1.0;
    auto result = sweep.run_bifurcation_1d(request);
    if (!result.ok) { std::cerr << result.error << '\n'; return 1; }
    // Basins: the kernel returns the IC nodes it started the cells from (ucudaBasinsAxisNodes);
    // portraits take their IC from them. On a linear axis the host formula must give the same
    // bits: NVRTC contracts the node into an fma, and the host copy without it differed in ~1/3
    // of the nodes, so a portrait could start in another basin.
    auto basins_nodes_ok = [](const char* tag) {
        BasinsRequest b;
        b.krs_body = "X[0] += h * (-X[0]); X[1] += h * (-X[1]);";
        b.amountOfX = 2;
        b.initial_conditions = {0.0, 0.0};
        b.base_values = {0.5};
        b.axis_x_var = 0; b.axis_y_var = 1;
        b.axis_x_lo = -17.3; b.axis_x_hi = 13.1;
        b.axis_y_lo = -9.7;  b.axis_y_hi = 21.9;
        b.n_pts = 120;
        b.h = 0.01; b.t_max = 1.0; b.transient_time = 0.0;
        b.pre_scaller = 1; b.max_value = 1e6; b.eps_dbscan = 0.5;
        ParametricEngine eng;
        BasinsResult r = eng.run_basins(b);
        if (!r.ok) { std::cerr << tag << " basins: " << r.error << '\n'; return false; }
        if ((int)r.axis_x_nodes.size() != b.n_pts || (int)r.axis_y_nodes.size() != b.n_pts) {
            std::cerr << tag << " basins: kernel returned no axis nodes\n"; return false;
        }
        for (int k = 0; k < b.n_pts; ++k) {
            const double hx = (double)ucuda_node_value(k, b.n_pts, b.axis_x_lo, b.axis_x_hi);
            const double hy = (double)ucuda_node_value(k, b.n_pts, b.axis_y_lo, b.axis_y_hi);
            if (std::memcmp(&hx, &r.axis_x_nodes[(size_t)k], sizeof hx) != 0 ||
                std::memcmp(&hy, &r.axis_y_nodes[(size_t)k], sizeof hy) != 0) {
                std::cerr.precision(17);
                std::cerr << tag << " basins: node " << k << " host " << hx << ' ' << hy << " kernel "
                          << r.axis_x_nodes[(size_t)k] << ' ' << r.axis_y_nodes[(size_t)k] << '\n';
                return false;
            }
        }
        return true;
    };
    if (!basins_nodes_ok("monolithic")) return 1;
    // Parameter nodes as the kernel really used them: with X0 = (X1 > 0 ? a1 : 0), X1 = -X1 every
    // peak of X0 is exactly the point's parameter. The host node functions (crosshair, drill-down,
    // slice parameter, CSV) must give the same bits on linear and log grids, classic and
    // continuation (both directions). log10/pow of CUDA and of the host CRT differ, so the log
    // nodes are built from ucuda_log10_det / ucuda_exp10_det.
    auto param_nodes_ok = [](bool log_scale, bool cont, bool rev) {
        Bifurcation1DRequest q;
        q.krs_body = "X[0] = (X[1] > 0) ? a[1] : 0; X[1] = -X[1];";
        q.amountOfX = 2;
        q.initial_conditions = {0.0, 1.0};
        q.base_values = {0.5, 0.0};
        q.param_index = 1;
        q.log_scale = log_scale;
        q.continuation = cont; q.continuation_reverse = rev;
        q.param_lo = log_scale ? 0.0123 : -17.3; q.param_hi = log_scale ? 47.9 : 13.1;
        q.n_pts = 333;
        q.h = 1.0; q.t_max = 40.0; q.transient_time = 0.0;
        q.writable_var = 0;
        ParametricEngine eng;
        const Bifurcation1DResult r = eng.run_bifurcation_1d(q);
        const char* tag = log_scale ? (cont ? (rev ? "log cont rev" : "log cont") : "log")
                                    : (cont ? (rev ? "lin cont rev" : "lin cont") : "lin");
        if (!r.ok) { std::cerr << tag << ": " << r.error << '\n'; return false; }
        int checked = 0;
        for (int k = 0; k < q.n_pts && k < (int)r.bifurcation_points.size(); ++k) {
            const double hv = cont ? (double)ucuda_node_value_cont(k, q.n_pts, q.param_lo, q.param_hi, log_scale, rev)
                        : log_scale ? (double)ucuda_node_value_log(k, q.n_pts, q.param_lo, q.param_hi)
                                    : (double)ucuda_node_value(k, q.n_pts, q.param_lo, q.param_hi);
            for (double v : r.bifurcation_points[(size_t)k]) {
                if (hv <= 0.0 && !log_scale) break;   // non-positive a: no peak equals the parameter
                if (std::memcmp(&hv, &v, sizeof hv) != 0) {
                    std::cerr.precision(17);
                    std::cerr << tag << ": point " << k << " kernel parameter " << v << ", host node " << hv << '\n';
                    return false;
                }
                ++checked;
            }
        }
        if (checked < q.n_pts / 3) { std::cerr << tag << ": too few peaks (" << checked << ")\n"; return false; }
        return true;
    };
    for (int lg = 0; lg < 2; ++lg)
        if (!param_nodes_ok(lg, false, false) || !param_nodes_ok(lg, true, false) || !param_nodes_ok(lg, true, true))
            return 1;
    // Separate NVRTC compilation (Settings -> GPU): the step and the library are
    // separate programs, each needing every header configCUDA.h pulls in.
    set_nvrtc_rdc(true);
    ParametricEngine split;
    auto split_result = split.run_bifurcation_1d(request);
    const bool split_basins = basins_nodes_ok("rdc");
    set_nvrtc_rdc(false);
    if (!split_result.ok) { std::cerr << "rdc: " << split_result.error << '\n'; return 1; }
    if (!split_basins) return 1;
    // Circuit kernels must also resolve resources beside the executable on both OSes.
    bool circuit_ok = false;
    const std::string circuit_report = circuit_nvrtc_selftest(&circuit_ok);
    if (!circuit_ok) { std::cerr << circuit_report; return 1; }
    std::cout << "CUDA execution and NVRTC resource loading passed\n";
}
