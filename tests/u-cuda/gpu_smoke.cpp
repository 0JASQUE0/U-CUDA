// GPU smoke: NVRTC компилирует тело шага, модуль грузится, считается минимальная траектория;
// комплексный путь находит configCUDA.h в kernels\ рядом с exe; свип БД собирается и считает.
#include "test_util.h"
#include "nvrtc_engine.h"
#include "parametric_engine.h"
#include <cuda.h>

int main() {
    return ut::run_main("gpu_smoke", []() -> int {
        if (cuInit(0) != CUDA_SUCCESS) { std::printf("no CUDA device: skip\n"); return ut::kSkip; }
        NvrtcEngine engine;
        ut::require(engine.compile("X[0] += h * a[0];", 1), "compile: " + engine.error());
        std::vector<std::vector<std::vector<double>>> out;
        ut::require(engine.run_phase_portraits({ 1.0 }, 1, { 2.0 }, 0.25, 4, 0, out), "run: " + engine.error());
        ut::require(out.size() == 1 && out[0].size() == 4, "trajectory shape");
        ut::approx(out[0][3][0], 2.5, 1e-12, "x after 3 steps");

        ut::require(engine.compile("ucmplx z(X[0], 0); X[0] = z.re + h * a[0];", 1),
                    "complex body (configCUDA.h from kernels\\): " + engine.error());

        ParametricEngine sweep;
        Bifurcation1DRequest rq;
        rq.krs_body = "X[0] += h * (a[1] - X[0]);";
        rq.amountOfX = 1;
        rq.initial_conditions = { 0.0 };
        rq.base_values = { 0.5, 1.0 };
        rq.param_index = 1;
        rq.param_lo = 0.5; rq.param_hi = 1.5;
        rq.n_pts = 2;
        rq.h = 0.01;
        rq.t_max = 1.0;
        const auto r = sweep.run_bifurcation_1d(rq);
        ut::require(r.ok, "BD 1D: " + r.error);
        return 0;
    });
}
