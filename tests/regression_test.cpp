#include "support/artifacts.h"
#include "support/analysis_workloads.h"
#include "support/builtin_cases.h"
#include "support/solver_workloads.h"
#include "ucuda/cuda/nvrtc_engine.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/numerics/integrator.h"
#include "ucuda/numerics/krs_cpu.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/analysis/network_session.h"
#include <gtest/gtest.h>
#include <cuda.h>
#include <cmath>
#include <utility>
#include <regex>
#include <set>

std::string exe_dir() { return platform::executable_dir(); }
namespace {
using regression::Json;
std::vector<double> parameters() { return {.5, 4, 1e-12, 25, 0}; }
void compile(KrsCpuStep& step, const std::string& body) {
    std::vector<KrsCpuDiag> diags;
    if (!step.compile(body, 2, (int)parameters().size(), diags)) {
        std::string message;
        for (const auto& d : diags) message += d.message + '\n';
        throw std::runtime_error(message);
    }
}
void fixed_case(int method, bool gpu) {
    const Scheme scheme = (Scheme)method;
    System system; system.vars = {"x", "y"};
    system.rhs = scheme == Scheme::Map ? std::vector<std::string>{"x+0.125", "y-0.25"}
                                      : std::vector<std::string>{"-x", "-2*y"};
    const auto body = codegen_scheme(system, scheme);
    ASSERT_FALSE(body.empty());
    KrsCpuStep step; ASSERT_NO_THROW(compile(step, body));
    const auto a = parameters(); const double ic[] = {1, 1};
    constexpr double h = .001;
    const int count = scheme == Scheme::Map ? 9 : 101;
    std::vector<std::vector<double>> cpu;
    ASSERT_TRUE(computePhasePortraitCPU_custom(step.fn(), ic, 2, a.data(), (int)a.size(), h, count, 0, cpu));
    ASSERT_EQ(cpu.size(), count);
    std::vector<std::vector<double>> actual = cpu;
    NvrtcEngine engine;
    if (gpu) {
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS) << "GPU coverage requires a real CUDA device";
        ASSERT_TRUE(engine.compile(body, 2)) << engine.error();
        std::vector<std::vector<std::vector<double>>> out;
        ASSERT_TRUE(engine.run_phase_portraits({1, 1}, 1, a, h, count, 0, out)) << engine.error();
        ASSERT_EQ(out.size(), 1); ASSERT_EQ(out[0].size(), count); actual = out[0];
    }
    std::vector<double> values;
    for (int i = 0; i < count; ++i) {
        ASSERT_EQ(actual[i].size(), 2);
        for (int k = 0; k < 2; ++k) {
            ASSERT_TRUE(std::isfinite(actual[i][k]));
            const double expected = scheme == Scheme::Map ? 1 + i * (k ? -.25 : .125) : std::exp(-(k + 1) * i * h);
            ASSERT_NEAR(actual[i][k], expected, scheme == Scheme::Map ? 0 : 2e-4) << "point " << i;
            ASSERT_NEAR(actual[i][k], cpu[i][k], 1e-10);
            values.push_back(actual[i][k]);
        }
    }
    Json inputs = {{"method", method}, {"rhs", system.rhs}, {"initial", {1, 1}}, {"parameters", a}, {"h", h}, {"count", count}};
    const bool new_cd = scheme == Scheme::CDFast || scheme == Scheme::CDFP || scheme == Scheme::CDFastFP;
    if (new_cd) {
        std::vector<double> errors;
        for (int n : {10, 20, 40}) {
            std::vector<std::vector<double>> trajectory;
            if (gpu) {
                std::vector<std::vector<std::vector<double>>> result;
                ASSERT_TRUE(engine.run_phase_portraits({1, 1}, 1, a, 1.0 / n, n + 1, 0, result)) << engine.error();
                trajectory = result.at(0);
            } else {
                ASSERT_TRUE(computePhasePortraitCPU_custom(step.fn(), ic, 2, a.data(), (int)a.size(), 1.0 / n, n + 1, 0, trajectory));
            }
            errors.push_back(std::fabs(trajectory.back()[0] - std::exp(-1.0)));
        }
        for (size_t i = 1; i < errors.size(); ++i) {
            const double order = std::log2(errors[i - 1] / errors[i]);
            ASSERT_GT(order, 1.8); ASSERT_LT(order, 2.2);
        }
        inputs["convergence_steps"] = {10, 20, 40};
        values.insert(values.end(), errors.begin(), errors.end());
    }
    regression::snapshot("fixed." + std::string(gpu ? "gpu." : "cpu.") + std::to_string(method), inputs, values, new_cd ? 2 : 1);
}
class FixedMethods : public ::testing::TestWithParam<std::pair<int, bool>> {};
TEST_P(FixedMethods, AnalyticTrajectoryAndBackendAgreement) { fixed_case(GetParam().first, GetParam().second); }
std::vector<std::pair<int, bool>> methods() {
    std::vector<std::pair<int, bool>> result;
    for (const auto& c : regression::builtin_cases())
        for (bool gpu : {false, true}) result.emplace_back((int)c.scheme, gpu);
    return result;
}
INSTANTIATE_TEST_SUITE_P(AllBuiltins, FixedMethods, ::testing::ValuesIn(methods()), [](const auto& info) {
    return std::string(info.param.second ? "gpu_" : "cpu_") + regression::builtin_case_name(info.param.first);
});
TEST(SmokeCpu, RK4) { fixed_case((int)Scheme::RK4, false); }
TEST(SmokeGpu, RK4) { fixed_case((int)Scheme::RK4, true); }
class SolverMethods : public ::testing::TestWithParam<std::pair<std::string, bool>> {};
TEST_P(SolverMethods, AnalyticSolutionOrConstraintAndBackendAgreement) {
    const auto& [name, gpu] = GetParam();
    if (gpu) ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    regression::SolverWorkload solver(name, gpu);
    std::vector<std::vector<double>> result;
    ASSERT_NO_THROW(result = solver.run());
    ASSERT_NO_THROW(solver.verify(result));
    if (gpu) {
        regression::SolverWorkload reference(name, false);
        const auto cpu = reference.run(); reference.verify(cpu);
        ASSERT_EQ(result.size(), cpu.size());
        for (size_t i = 0; i < result.size(); ++i) for (size_t j = 0; j < result[i].size(); ++j)
            ASSERT_NEAR(result[i][j], cpu[i][j], 1e-8) << name << " point " << i;
    }
    std::vector<double> values;
    for (const auto& point : result) values.insert(values.end(), point.begin(), point.end());
    regression::snapshot("solver." + std::string(gpu ? "gpu." : "cpu.") + regression::solver_case_id(name),
        {{"method", name}, {"workload_version", 1}, {"h", .001}, {"count", 101}}, values);
}
std::vector<std::pair<std::string, bool>> solvers() {
    std::vector<std::pair<std::string, bool>> result;
    for (const auto& name : regression::solver_cases()) for (bool gpu : {false, true}) result.emplace_back(name, gpu);
    return result;
}
INSTANTIATE_TEST_SUITE_P(AllSolvers, SolverMethods, ::testing::ValuesIn(solvers()), [](const auto& info) {
    return std::string(info.param.second ? "gpu_" : "cpu_") + regression::solver_case_id(info.param.first);
});
TEST(ConvergenceCpu, EulerAndRK4RetainOrder) {
    System sys; sys.vars = {"x", "y"}; sys.rhs = {"-x", "-2*y"};
    const auto a = parameters(); const double ic[] = {1, 1};
    for (Scheme scheme : {Scheme::Euler, Scheme::RK4}) {
        KrsCpuStep step; ASSERT_NO_THROW(compile(step, codegen_scheme(sys, scheme)));
        std::vector<double> errors;
        for (int n : {50, 100, 200}) {
            std::vector<std::vector<double>> out;
            ASSERT_TRUE(computePhasePortraitCPU_custom(step.fn(), ic, 2, a.data(), (int)a.size(), 1.0 / n, n + 1, 0, out));
            errors.push_back(std::fabs(out.back()[0] - std::exp(-1.0)));
        }
        const double order = std::log2(errors[0] / errors[1]);
        EXPECT_GT(order, scheme == Scheme::Euler ? .95 : 3.8);
        EXPECT_LT(order, scheme == Scheme::Euler ? 1.1 : 4.2);
        EXPECT_LT(errors[2], errors[1]);
        regression::snapshot("convergence.cpu." + std::to_string((int)scheme),
            {{"method", (int)scheme}, {"steps", {50, 100, 200}}}, errors);
    }
}
TEST(CompilerCpu, RejectsOutOfRangeAndSupportsMove) {
    KrsCpuStep step; std::vector<KrsCpuDiag> diags;
    EXPECT_FALSE(step.compile("X[2] += h;", 2, 5, diags)); EXPECT_FALSE(diags.empty());
    ASSERT_TRUE(step.compile("X[0] += h;", 2, 5, diags));
    KrsCpuStep moved(std::move(step)); EXPECT_NE(moved.fn(), nullptr); EXPECT_EQ(step.fn(), nullptr);
    step = std::move(moved); EXPECT_NE(step.fn(), nullptr); EXPECT_EQ(moved.fn(), nullptr);
}
TEST(CompilerCpu, ExtendedPrecisionsRetainSubDoubleIncrementsAcrossTheModuleBoundary) {
    const double delta = std::ldexp(1., -80);
    for (const auto precision : {KrsCpuPrec::DD, KrsCpuPrec::QD}) {
        KrsCpuStep step; std::vector<KrsCpuDiag> diags;
        ASSERT_TRUE(step.compile("X[0] += a[0];", 1, 1, precision, diags));
        EXPECT_EQ(step.fn(), nullptr);
        double difference = 0;
        if (precision == KrsCpuPrec::DD) {
            ASSERT_NE(step.fn_dd(), nullptr); EXPECT_EQ(step.fn_qd(), nullptr);
            ucuda::dd x[1] = {1.}, a[1] = {delta}, h(.1);
            for (int i = 0; i < 16; ++i) step.fn_dd()(x, a, &h);
            difference = (double)(x[0] - ucuda::dd(1.));
        } else {
            ASSERT_NE(step.fn_qd(), nullptr); EXPECT_EQ(step.fn_dd(), nullptr);
            ucuda::qd x[1] = {1.}, a[1] = {delta}, h(.1);
            for (int i = 0; i < 16; ++i) step.fn_qd()(x, a, &h);
            difference = (double)(x[0] - ucuda::qd(1.));
        }
        EXPECT_EQ(difference, 16 * delta);
        regression::snapshot("precision.cpu." + std::to_string((int)precision),
            {{"precision", (int)precision}, {"increment", delta}, {"repetitions", 16}}, {difference});
    }
}
TEST(SerializationCpu, LegacyFixturesPreserveValuesAndNewNewtonOptionsRoundTrip) {
    const auto root = std::filesystem::path(UCUDA_TEST_SOURCE) / "tests/fixtures/library/Dahlquist 2D/sessions";
    auto read = [&](const char* name) { std::ifstream in(root / name); if (!in) throw std::runtime_error("Missing session fixture");
        return std::string(std::istreambuf_iterator<char>(in), {}); };
    PhaseAnalysisSession phase; ASSERT_TRUE(session_from_json(read("legacy-phase.json"), phase));
    EXPECT_EQ(phase.step_h, "0.02"); EXPECT_EQ(phase.sim_time, "12"); EXPECT_EQ(phase.projections.size(), 1u);
    OrderAnalysisSession order; ASSERT_TRUE(session_from_json_order(read("_last_order.json"), order));
    ASSERT_EQ(order.configs.size(), 1u); EXPECT_EQ(order.configs[0].label, "legacy order");
    EXPECT_EQ(order.configs[0].axis_x_n_text, "6"); EXPECT_FALSE(order.configs[0].use_gpu);
    order.configs[0].newton.tol = "3e-13"; order.configs[0].newton.max_iters = "31"; order.configs[0].newton.full = false;
    const auto saved = session_to_json_order(order);
    OrderAnalysisSession loaded; ASSERT_TRUE(session_from_json_order(saved, loaded));
    EXPECT_EQ(loaded.configs[0].newton.tol, "3e-13"); EXPECT_EQ(loaded.configs[0].newton.max_iters, "31");
    EXPECT_FALSE(loaded.configs[0].newton.full); EXPECT_EQ(session_to_json_order(loaded), saved);
    NetworkSession network; ASSERT_TRUE(session_from_json_network(read("_last_network.json"), network));
    ASSERT_EQ(network.configs.size(), 1u); EXPECT_EQ(network.configs[0].h_text, "0.025");
    EXPECT_EQ(network.configs[0].nodes.size(), 2u); EXPECT_EQ(network.configs[0].nodes[1].label, "B");
    const auto net = session_to_json_network(network);
    NetworkSession netLoaded; ASSERT_TRUE(session_from_json_network(net, netLoaded));
    EXPECT_EQ(session_to_json_network(netLoaded), net);
}
void cancelled_sweep_recovers(bool gpu) {
    System sys; sys.vars = {"x", "y"}; sys.params = {"p"}; sys.rhs = {"-p*x", "-2*p*y"};
    ParametricEngine engine;
    Bifurcation1DRequest request; regression::sweep_request(request, codegen_scheme(sys, Scheme::RK4));
    request.use_cpu = !gpu; request.emit_all_samples = true;
    request.cancel = std::make_shared<std::atomic<bool>>(true);
    const auto stopped = engine.run_bifurcation_1d(request);
    EXPECT_TRUE(stopped.cancelled); EXPECT_FALSE(stopped.ok);
    request.cancel->store(false);
    const auto recovered = engine.run_bifurcation_1d(request);
    ASSERT_TRUE(recovered.ok) << recovered.error; EXPECT_FALSE(recovered.cancelled);
    EXPECT_EQ(recovered.bifurcation_points.size(), 4u);
}
TEST(CancellationCpu, CancelledSweepReleasesResourcesAndEngineCanRunAgain) { cancelled_sweep_recovers(false); }
TEST(CancellationGpu, CancelledSweepReleasesResourcesAndEngineCanRunAgain) {
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS); cancelled_sweep_recovers(true);
}
TEST(CacheGpu, SwitchingModulesAndCompileFailurePreserveCachedResults) {
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    NvrtcEngine engine;
    auto run = [&](const char* body) {
        if (!engine.compile(body, 1)) throw std::runtime_error(engine.error());
        std::vector<std::vector<std::vector<double>>> out;
        if (!engine.run_phase_portraits({1}, 1, {.5}, .125, 9, 0, out)) throw std::runtime_error(engine.error());
        return out;
    };
    const auto initial = run("X[0] += h;");
    const auto other = run("X[0] -= h;"); EXPECT_NE(initial, other);
    EXPECT_FALSE(engine.compile("this is invalid CUDA;", 1));
    EXPECT_EQ(run("X[0] += h;"), initial);
}
void explicit_time_trajectory(bool gpu) {
    System sys; sys.vars = {"x"}; sys.rhs = {"t"};
    const auto body = codegen_scheme(sys, Scheme::RK4);
    const std::vector<double> a = parameters();
    std::vector<std::vector<double>> result;
    if (gpu) {
        NvrtcEngine engine; ASSERT_TRUE(engine.compile(body, 1)) << engine.error();
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::vector<std::vector<std::vector<double>>> out;
            ASSERT_TRUE(engine.run_phase_portraits({0}, 1, a, .01, 101, 20, out)) << engine.error();
            result = out.at(0);
            for (size_t i = 0; i < result.size(); ++i) EXPECT_NEAR(result[i][0], .5 * std::pow(.2 + .01*i, 2), 1e-12);
        }
    } else {
        KrsCpuStep step; std::vector<KrsCpuDiag> diags;
        ASSERT_TRUE(step.compile(body, 1, (int)a.size(), diags)); ASSERT_TRUE(step.has_time());
        const double ic = 0;
        for (int repeat = 0; repeat < 2; ++repeat) {
            step.step_index(0);
            ASSERT_TRUE(computePhasePortraitCPU_custom(step.fn(), &ic, 1, a.data(), (int)a.size(), .01, 101, 20, result));
            for (size_t i = 0; i < result.size(); ++i) EXPECT_NEAR(result[i][0], .5 * std::pow(.2 + .01*i, 2), 1e-12);
        }
    }
    ASSERT_EQ(result.size(), 101u);
    regression::snapshot(std::string("time.") + (gpu ? "gpu" : "cpu"), {{"rhs", "t"}, {"h", .01}, {"skip", 20}, {"count", 101}}, {result.front()[0], result.back()[0]});
    EXPECT_THROW(codegen_scheme(sys, Scheme::ComplexCD), std::runtime_error);
}
TEST(TimeCpu, NonAutonomousStagesTransientAndRepeatedRunStartAtTheCorrectTime) { explicit_time_trajectory(false); }
TEST(TimeGpu, NonAutonomousStagesTransientAndRepeatedRunStartAtTheCorrectTime) {
    ASSERT_EQ(cuInit(0), CUDA_SUCCESS); explicit_time_trajectory(true);
}
TEST(MatrixCpu, EveryBuiltinEnumIsEnumerated) {
    EXPECT_EQ(methods().size(), 2u * regression::builtin_cases().size());
    EXPECT_EQ((int)Scheme::Euler, 0);
    const std::filesystem::path file = std::filesystem::path(UCUDA_TEST_SOURCE) / "libs/equations/include/ucuda/equations/codegen.hpp";
    std::ifstream in(file); ASSERT_TRUE(in.good());
    const std::string source{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const auto begin = source.find("enum class Scheme {"); ASSERT_NE(begin, std::string::npos);
    const auto end = source.find("};", begin); ASSERT_NE(end, std::string::npos);
    std::istringstream names(source.substr(begin + 19, end - begin - 19));
    size_t count = 0;
    std::set<std::string> declared, covered;
    for (std::string name; std::getline(names, name, ',');)
        if (name.find_first_not_of(" \r\n\t") != std::string::npos) {
            name.erase(0, name.find_first_not_of(" \r\n\t"));
            name.erase(name.find_last_not_of(" \r\n\t") + 1);
            declared.insert(name); ++count;
        }
    for (const auto& c : regression::builtin_cases()) {
        EXPECT_TRUE(covered.insert(c.name).second);
        const auto display = regression::builtin_display_name(c.scheme);
        EXPECT_EQ(scheme_from_name(display), c.scheme);
        if (c.scheme != Scheme::Map) {
            int order = 0; bool symmetric = false;
            EXPECT_TRUE(builtin_scheme_traits(display, &order, &symmetric)); EXPECT_GT(order, 0);
        }
    }
    EXPECT_EQ(declared, covered);
    EXPECT_EQ(count, regression::builtin_cases().size()) << "New enum values need explicit test matrix coverage";
}
TEST(MatrixCpu, EveryAnalysisFacadeHasAWorkload) {
    const auto path = std::filesystem::path(UCUDA_TEST_SOURCE) / "libs/cuda/include/ucuda/cuda/parametric_engine.h";
    std::ifstream in(path); ASSERT_TRUE(in.good());
    const std::string source{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    std::set<std::string> declared, covered;
    const std::regex method("\\b\\w+Result\\s+(run_\\w+)\\s*\\(");
    for (auto i = std::sregex_iterator(source.begin(), source.end(), method); i != std::sregex_iterator(); ++i)
        declared.insert((*i)[1].str());
    for (const auto& c : regression::analysis_cases()) {
        std::string name = c.name;
        covered.insert(name == "dft" ? "run_dft_1d" : name == "metrics" ? "run_signal_metrics" : "run_" + name);
        EXPECT_TRUE(c.cpu || c.gpu);
    }
    EXPECT_EQ(declared, covered) << "New analysis APIs need a test and benchmark scenario";
}
class MethodAnalysisMatrix : public ::testing::TestWithParam<std::pair<int, bool>> {};
TEST_P(MethodAnalysisMatrix, SupportedAnalysisPaths) {
    const auto [value, gpu] = GetParam();
    const auto method = (Scheme)value;
    ParametricEngine engine;
    for (const auto& family : regression::analysis_cases()) {
        if (!(gpu ? family.gpu : family.cpu)) continue;
        const std::string name = family.name;
        SCOPED_TRACE(regression::builtin_display_name(method) + "/" + name);
        std::vector<double> values;
        ASSERT_NO_THROW(values = regression::analysis_workload(engine, name, gpu, nullptr, true, method, true));
        regression::snapshot("matrix." + std::string(gpu ? "gpu." : "cpu.") + std::to_string(value) + "." + name,
            {{"method", regression::builtin_display_name(method)}, {"analysis", name}, {"workload_version", 1}}, values);
    }
}
std::vector<std::pair<int, bool>> analysis_matrix_methods() {
    auto values = methods();
    values.erase(std::remove_if(values.begin(), values.end(), [](const auto& v) { return v.first == (int)Scheme::Map; }), values.end());
    return values;
}
INSTANTIATE_TEST_SUITE_P(FullMatrix, MethodAnalysisMatrix, ::testing::ValuesIn(analysis_matrix_methods()), [](const auto& info) {
    return std::string(info.param.second ? "gpu_" : "cpu_") + regression::builtin_case_name(info.param.first);
});
TEST(ValidationCpu, RejectsUnsupportedInitialConditionLyapunovSweeps) {
    System sys; sys.vars = {"x", "y"}; sys.params = {"p"}; sys.rhs = {"-p*x", "-2*p*y"};
    ParametricEngine engine;
    LLE1DRequest lle; regression::sweep_request(lle, codegen_scheme(sys, Scheme::RK4)); lle.use_cpu = true;
    const auto lr = engine.run_lle_1d(lle); EXPECT_FALSE(lr.ok); EXPECT_FALSE(lr.error.empty());
    LS1DRequest ls; regression::sweep_request(ls, codegen_scheme(sys, Scheme::RK4)); ls.use_cpu = true;
    const auto sr = engine.run_ls_1d(ls); EXPECT_FALSE(sr.ok); EXPECT_FALSE(sr.error.empty());
}
class AnalysisFamilies : public ::testing::TestWithParam<std::pair<std::string, bool>> {};
TEST_P(AnalysisFamilies, ReferenceWorkload) {
    const auto& [name, gpu] = GetParam();
    if (gpu) ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    ParametricEngine engine;
    std::vector<double> values;
    ASSERT_NO_THROW(values = regression::analysis_workload(engine, name, gpu));
    regression::snapshot("analysis." + std::string(gpu ? "gpu." : "cpu.") + name,
        {{"workload", name}, {"workload_version", 1}, {"model", "linear_decay_p1"}, {"backend", gpu ? "gpu" : "cpu"}}, values);
}
std::vector<std::pair<std::string, bool>> families() {
    std::vector<std::pair<std::string, bool>> result;
    for (const auto& c : regression::analysis_cases()) {
        if (c.cpu) result.emplace_back(c.name, false);
        if (c.gpu) result.emplace_back(c.name, true);
    }
    return result;
}
INSTANTIATE_TEST_SUITE_P(AllAnalyses, AnalysisFamilies, ::testing::ValuesIn(families()), [](const auto& info) {
    return std::string(info.param.second ? "gpu_" : "cpu_") + info.param.first;
});
}
