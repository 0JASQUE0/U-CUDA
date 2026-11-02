#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/platform/platform_paths.h"
#include "support/analysis_workloads.h"
#include "support/builtin_cases.h"
#include "support/solver_workloads.h"
#include <benchmark/benchmark.h>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <memory>

std::string exe_dir() { return platform::executable_dir(); }
namespace {
// Google Benchmark invokes a callback again for calibration and repetitions.
// Keep the current warm workload alive across those callbacks. Switching cases
// releases it, so the full suite does not retain hundreds of CUDA modules.
std::string active_key;
std::shared_ptr<void> active_workload;
template<class T, class Factory> T& warm_workload(const std::string& key, Factory factory) {
    if (active_key != key || !active_workload) {
        active_workload.reset(); active_key = key; active_workload = factory();
    }
    return *std::static_pointer_cast<T>(active_workload);
}
struct PhaseModule { KrsCpuStep step; NvrtcEngine engine; bool ready = false; double setup_ms = 0; };
void phase(benchmark::State& state, int method, bool gpu) {
    try {
        System sys; sys.vars = {"x", "y"}; sys.rhs = {"-x", "-2*y"};
        const auto body = codegen_scheme(sys, (Scheme)method);
        const std::vector<double> a = {.5, 4, 1e-12, 25, 0}; const double ic[] = {1, 1};
        auto& module = warm_workload<PhaseModule>("phase." + std::to_string(method) + (gpu ? ".gpu" : ".cpu"),
            [] { return std::make_shared<PhaseModule>(); });
        auto& step = module.step; auto& engine = module.engine;
        if (!module.ready) {
            std::vector<KrsCpuDiag> diags;
            const auto cold = std::chrono::steady_clock::now();
            if (gpu ? !engine.compile(body, 2) : !step.compile(body, 2, (int)a.size(), diags))
                throw std::runtime_error("Scheme compilation failed: " + engine.error());
            module.setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold).count();
            module.ready = true;
        }
        std::vector<std::vector<double>> cpu;
        std::vector<std::vector<std::vector<double>>> device;
        auto run = [&] {
            const bool ok = gpu ? engine.run_phase_portraits({1, 1}, 1, a, .001, 1001, 0, device)
                : computePhasePortraitCPU_custom(step.fn(), ic, 2, a.data(), (int)a.size(), .001, 1001, 0, cpu);
            if (!ok) throw std::runtime_error("Phase calculation failed");
        };
        run();
        auto verify = [&] {
            const auto& result = gpu ? device.at(0) : cpu;
            if (result.size() != 1001 || result.back().size() != 2 ||
                !std::isfinite(result.back()[0]) || std::fabs(result.back()[0] - std::exp(-1.0)) > 3e-4)
                throw std::runtime_error("Benchmark failed its analytic oracle");
            for (size_t i = 0; i < result.size(); ++i) {
                if (result[i].size() != 2) throw std::runtime_error("Benchmark trajectory shape changed");
                for (int j = 0; j < 2; ++j)
                    if (!std::isfinite(result[i][j]) || std::fabs(result[i][j] - std::exp(-(j+1)*.001*i)) > (j ? 4e-4 : 3e-4))
                        throw std::runtime_error("Benchmark trajectory failed its analytic oracle");
            }
        };
        verify();
        for (auto _ : state) {
            run();
            benchmark::DoNotOptimize(gpu ? device[0].back()[0] : cpu.back()[0]);
        }
        verify();
        state.counters["module_setup_ms"] = module.setup_ms;
        state.counters["input_steps"] = 1000;
        state.counters["input_method"] = method;
        state.counters["input_dimension"] = 2;
        state.counters["input_trajectories"] = 1;
        state.counters["input_workload_version"] = 1;
    } catch (const std::exception& e) { state.SkipWithError(e.what()); }
}
}
int main(int argc, char** argv) {
    const char* requestedBackend = std::getenv("UCUDA_BENCHMARK_BACKEND");
    const std::string backend = requestedBackend ? requestedBackend : "all";
    for (const auto& c : regression::builtin_cases()) {
        if (c.scheme == Scheme::Map) continue;
        const int i = (int)c.scheme;
        for (bool gpu : {false, true}) {
            if (backend != "all" && backend != (gpu ? "gpu" : "cpu")) continue;
            const auto name = std::string(gpu ? "gpu_phase_" : "cpu_phase_") + c.name;
            benchmark::RegisterBenchmark(name.c_str(), [i, gpu](benchmark::State& s) { phase(s, i, gpu); })
                ->UseRealTime()->Unit(benchmark::kMicrosecond);
        }
    }
    for (const auto& c : regression::analysis_cases()) {
        for (bool gpu : {false, true}) {
            if (backend != "all" && backend != (gpu ? "gpu" : "cpu")) continue;
            if (!(gpu ? c.gpu : c.cpu)) continue;
            const std::string family = c.name;
            const auto name = std::string(gpu ? "gpu_analysis_" : "cpu_analysis_") + family;
            benchmark::RegisterBenchmark(name.c_str(), [family, gpu](benchmark::State& state) {
                try {
                    auto& engine = warm_workload<ParametricEngine>("analysis." + family + (gpu ? ".gpu" : ".cpu"),
                        [] { return std::make_shared<ParametricEngine>(); });
                    regression::analysis_workload(engine, family, gpu);
                    for (auto _ : state) {
                        const auto values = regression::analysis_workload(engine, family, gpu, nullptr, false);
                        benchmark::DoNotOptimize(values.data());
                    }
                    regression::analysis_workload(engine, family, gpu);
                    state.counters["input_workload_version"] = 1;
                } catch (const std::exception& e) { state.SkipWithError(e.what()); }
            })->UseRealTime()->Unit(benchmark::kMicrosecond);
        }
    }
    for (const auto& method : regression::solver_cases()) for (bool gpu : {false, true}) {
        if (backend != "all" && backend != (gpu ? "gpu" : "cpu")) continue;
        const auto name = std::string(gpu ? "gpu_solver_" : "cpu_solver_") + regression::solver_case_id(method);
        benchmark::RegisterBenchmark(name.c_str(), [method, gpu](benchmark::State& state) {
            try {
                auto& solver = warm_workload<regression::SolverWorkload>("solver." + method + (gpu ? ".gpu" : ".cpu"),
                    [&] { return std::make_shared<regression::SolverWorkload>(method, gpu); });
                auto result = solver.run(); solver.verify(result);
                for (auto _ : state) { result = solver.run(); benchmark::DoNotOptimize(result.data()); }
                solver.verify(result);
                state.counters["input_workload_version"] = 1;
                state.counters["input_steps"] = 100;
            } catch (const std::exception& e) { state.SkipWithError(e.what()); }
        })->UseRealTime()->Unit(benchmark::kMicrosecond);
    }
    if (backend != "cpu") benchmark::RegisterBenchmark("gpu_network_kernel", [](benchmark::State& state) {
        try {
            auto& engine = warm_workload<ParametricEngine>("network.kernel", [] { return std::make_shared<ParametricEngine>(); });
            double warm_seconds = 0;
            regression::analysis_workload(engine, "network", true, &warm_seconds);
            for (auto _ : state) {
                double seconds = 0;
                regression::analysis_workload(engine, "network", true, &seconds, false);
                if (!(seconds > 0)) throw std::runtime_error("Missing CUDA event measurement");
                state.SetIterationTime(seconds);
            }
            regression::analysis_workload(engine, "network", true, &warm_seconds);
            state.counters["input_workload_version"] = 2;
            state.counters["input_nodes"] = 256;
            state.counters["input_steps"] = 10000;
        } catch (const std::exception& e) { state.SkipWithError(e.what()); }
    })->UseManualTime()->Unit(benchmark::kMicrosecond);
    for (bool gpu : {false, true}) {
        if (backend != "all" && backend != (gpu ? "gpu" : "cpu")) continue;
        const std::string name = gpu ? "gpu_cold_compile_RK4" : "cpu_cold_compile_RK4";
        benchmark::RegisterBenchmark(name.c_str(), [gpu](benchmark::State& state) {
            try {
                active_workload.reset(); active_key.clear();
                System sys; sys.vars = {"x", "y"}; sys.rhs = {"-x", "-2*y"};
                const auto body = codegen_scheme(sys, Scheme::RK4);
                for (auto _ : state) {
                    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
                    const auto source = body + "\n// independent cold program " + std::to_string(unique) + "\n";
                    if (gpu) {
                        NvrtcEngine engine;
                        if (!engine.compile(source, 2)) throw std::runtime_error(engine.error());
                    } else {
                        KrsCpuStep step; std::vector<KrsCpuDiag> diags;
                        if (!step.compile(source, 2, 5, diags)) throw std::runtime_error("Cold native compilation failed");
                    }
                }
                state.counters["input_method"] = (int)Scheme::RK4;
                state.counters["input_dimension"] = 2;
                state.counters["input_workload_version"] = 1;
            } catch (const std::exception& e) { state.SkipWithError(e.what()); }
        })->UseRealTime()->Unit(benchmark::kMillisecond);
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    active_workload.reset(); active_key.clear();
    benchmark::Shutdown();
}
