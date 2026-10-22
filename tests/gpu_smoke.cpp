#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/cuda/circuit_nvrtc.h"
#include "ucuda/platform/platform_paths.h"
#include <cuda.h>
#include <cmath>
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
    // Separate NVRTC compilation (Settings -> GPU): the step and the library are
    // separate programs, each needing every header configCUDA.h pulls in.
    set_nvrtc_rdc(true);
    ParametricEngine split;
    auto split_result = split.run_bifurcation_1d(request);
    set_nvrtc_rdc(false);
    if (!split_result.ok) { std::cerr << "rdc: " << split_result.error << '\n'; return 1; }
    // Circuit kernels must also resolve resources beside the executable on both OSes.
    bool circuit_ok = false;
    const std::string circuit_report = circuit_nvrtc_selftest(&circuit_ok);
    if (!circuit_ok) { std::cerr << circuit_report; return 1; }
    std::cout << "CUDA execution and NVRTC resource loading passed\n";
}
