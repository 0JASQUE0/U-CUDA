#include "ucuda/equations/codegen.hpp"
#include "ucuda/cuda/rqa.h"
#include "ucuda/numerics/integrator.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/analysis/order_session.h"
#include "ucuda/analysis/network_session.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/platform/platform_paths.h"
#include "../support/builtin_cases.h"
#include <nlohmann/json.hpp>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::string exe_dir() { return platform::executable_dir(); }
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Json number(double value) {
    if (std::isfinite(value)) return value;
    return std::isnan(value) ? "NaN" : value > 0 ? "+Inf" : "-Inf";
}
std::string read(const fs::path& file) {
    std::ifstream in(file, std::ios::binary); require(bool(in), "Cannot read observation fixture");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
Json rqa_observations() {
    require(cuInit(0) == CUDA_SUCCESS, "A real CUDA GPU is required");
    std::vector<std::vector<double>> trajectory;
    for (int i = 0; i < 96; ++i) trajectory.push_back({std::cos(i * .1), std::sin(i * .1), i * .005});
    Json observations = Json::array();
    for (auto source : {rqa::Source::StateVector, rqa::Source::Embedding})
        for (auto norm : {rqa::Norm::Euclidean, rqa::Norm::Maximum, rqa::Norm::Manhattan})
            for (auto mode : {rqa::EpsMode::Absolute, rqa::EpsMode::FracMaxDist, rqa::EpsMode::FracStd, rqa::EpsMode::TargetRR}) {
                rqa::Config cfg; cfg.source = source; cfg.norm = norm; cfg.eps_mode = mode;
                cfg.var = 0; cfg.m = 3; cfg.tau = 2; cfg.points = 96; cfg.eps = .35;
                cfg.eps_frac = .25; cfg.target_rr = .1; cfg.network_measures = true;
                rqa::Result result;
                require(rqa::compute(trajectory, .01, cfg, result) && result.ok, "RQA calculation failed");
                require(result.n > 2 && result.dist.size() == size_t(result.n) * result.n, "RQA output shape");
                require(std::isfinite(result.metrics.RR) && result.metrics.RR >= 0 && result.metrics.RR <= 1, "RQA recurrence rate");
                for (int i = 0; i < result.n; ++i) {
                    require(result.dist[size_t(i) * result.n + i] == 0, "RQA distance diagonal");
                    for (int j = 0; j < result.n; ++j)
                        require(result.dist[size_t(i) * result.n + j] == result.dist[size_t(j) * result.n + i], "RQA distance symmetry");
                }
                require(result.n == (source == rqa::Source::StateVector ? 96 : 92), "RQA embedding size");
                size_t recurrent = 0, allowed = 0;
                for (int i = 0; i < result.n; ++i) for (int j = 0; j < result.n; ++j) {
                    double expected = 0;
                    for (int coordinate = 0; coordinate < 3; ++coordinate) {
                        const auto x = source == rqa::Source::StateVector ? trajectory[i][coordinate] : trajectory[i + coordinate * cfg.tau][0];
                        const auto y = source == rqa::Source::StateVector ? trajectory[j][coordinate] : trajectory[j + coordinate * cfg.tau][0];
                        const auto delta = std::fabs(x - y);
                        if (norm == rqa::Norm::Maximum) expected = (std::max)(expected, delta);
                        else expected += norm == rqa::Norm::Euclidean ? delta * delta : delta;
                    }
                    if (norm == rqa::Norm::Euclidean) expected = std::sqrt(expected);
                    require(std::fabs(result.dist[size_t(i) * result.n + j] - expected) < 1e-12, "RQA independent distance oracle");
                    if (std::abs(i - j) > cfg.theiler) {
                        ++allowed;
                        recurrent += result.dist[size_t(i) * result.n + j] <= result.eps_used;
                    }
                }
                require(std::fabs(result.metrics.RR - double(recurrent) / allowed) < 1e-15, "RQA independent recurrence rate oracle");
                const auto& m = result.metrics;
                Json metrics = Json::array();
                for (double value : {m.RR,m.DET,m.L,m.L_max,m.DIV,m.ENTR,m.RATIO,m.LAM,m.TT,m.V_max,m.V_ENTR,m.TREND,
                                     m.T1,m.T2,m.W,m.W_max,m.RTE,m.clustering,m.transitivity}) metrics.push_back(number(value));
                observations.push_back({{"source", (int)source}, {"norm", (int)norm}, {"eps_mode", (int)mode},
                    {"n", result.n}, {"dt", result.dt}, {"t0", result.t0}, {"t1", result.t1},
                    {"eps_used", result.eps_used}, {"dist_max", result.dist_max}, {"distances", result.dist},
                    {"metrics", metrics}, {"network_valid", m.network_valid}});
            }
    rqa::Config invalid; rqa::Result failure;
    require(!rqa::compute(trajectory, .01, invalid, failure) && !failure.error.empty(), "RQA must reject missing source");
    observations.push_back({{"rejection", failure.error}});
    invalid.source = rqa::Source::StateVector;
    require(!rqa::compute({}, .01, invalid, failure) && !failure.error.empty(), "RQA must reject empty input");
    observations.push_back({{"rejection", failure.error}});
    return {{"trajectory", trajectory}, {"cases", observations}};
}
Json serialization() {
    SystemRecord record; record.name = "Parity fixture"; record.vars_text = "x,y"; record.params_text = "p";
    record.init_conditions = {{"x","1"},{"y","-2"}}; record.param_values = {{"p","0.75"}};
    record.newton_tol = "3e-12"; record.newton_max_iters = "17"; record.newton_full = false;
    const std::vector<std::string> variables = {"x","y"}, parameters = {"p"};
    Json result = Json::object();
#define SESSION(type, name, write, load) { \
    type original; original.load_from_record(record, variables, parameters); \
    const auto raw = write(original); type first; first.load_from_record(record, variables, parameters); \
    require(load(raw, first), "Cannot reload serialized session"); const auto normalized = write(first); \
    type second; second.load_from_record(record, variables, parameters); \
    require(load(normalized, second) && write(second) == normalized, "Session round trip changed"); \
    result[name] = {{"initial", raw}, {"normalized", normalized}}; }
    SESSION(PhaseAnalysisSession, "phase", session_to_json, session_from_json);
    SESSION(BifurcationAnalysisSession, "bifurcation", session_to_json_parametric, session_from_json_parametric);
    SESSION(LLEAnalysisSession, "lle", session_to_json_lle, session_from_json_lle);
    SESSION(LyapunovSpectrumAnalysisSession, "ls", session_to_json_ls, session_from_json_ls);
    SESSION(SignalMetricsAnalysisSession, "metrics", session_to_json_metrics, session_from_json_metrics);
    SESSION(Dft1DAnalysisSession, "dft", session_to_json_dft1d, session_from_json_dft1d);
    SESSION(BasinsAnalysisSession, "basins", session_to_json_basins, session_from_json_basins);
    SESSION(FastSyncAnalysisSession, "fastsync", session_to_json_fastsync, session_from_json_fastsync);
    SESSION(OrderAnalysisSession, "order", session_to_json_order, session_from_json_order);
    SESSION(NetworkSession, "network", session_to_json_network, session_from_json_network);
    SESSION(CustomSession, "custom", session_to_json_custom, session_from_json_custom);
#undef SESSION
    result["parametric_windows"] = session_to_json_parametric_windows(std::vector<ParametricPlotWindow>(1));
    result["order_windows"] = session_to_json_order_windows(std::vector<OrderPlotWindow>(1));
    result["dft_windows"] = session_to_json_dft1d_windows(std::vector<Dft1DPlotWindow>(1));
    return result;
}
Json generated_sources() {
    Json result = Json::object();
    for (int model = 0; model < 3; ++model) {
        System system; system.vars = {"x","y"}; system.params = {"p"};
        system.rhs = model == 0 ? std::vector<std::string>{"-p*x","-2*y"} : model == 1
            ? std::vector<std::string>{"y","-x-p*y+x*x"} : std::vector<std::string>{"p*cos(t)-x","-2*y"};
        for (const auto& method : regression::builtin_cases()) {
            const auto key = std::to_string(model) + "." + method.name;
            try { result[key] = {{"source", codegen_scheme(system, method.scheme)}}; }
            catch (const std::exception& e) { result[key] = {{"rejection", e.what()}}; }
        }
    }
    return result;
}
Json interpreted_cpu() {
    Json result = Json::object();
    const double initial[] = {1,1}, parameters[] = {.5,4,1e-12,25,0};
    for (const auto& method : regression::builtin_cases()) {
        const bool map = method.scheme == Scheme::Map;
        System sys; sys.vars = {"x","y"}; sys.is_map = map;
        sys.rhs = map ? std::vector<std::string>{"x+0.125","y-0.25"} : std::vector<std::string>{"-x","-2*y"};
        SystemEvaluator evaluator(sys);
        const auto scheme = int_scheme_from_string(regression::builtin_display_name(method.scheme));
        std::vector<std::vector<double>> trajectory;
        const int total = map ? 9 : 101, skip = map ? 3 : 10;
        require(computePhasePortraitCPU(evaluator, scheme, initial, 2, parameters, 5, .001, total, skip, trajectory), "Interpreted CPU trajectory failed");
        require(trajectory.size() == size_t(total), "Interpreted CPU shape");
        for (int i = 0; i < total; ++i) for (int k = 0; k < 2; ++k) {
            const double expected = map ? 1 + (i + skip) * (k ? -.25 : .125) : std::exp(-(k + 1) * (i + skip) * .001);
            require(std::fabs(trajectory[i][k] - expected) <= (map ? 0 : 3e-4), "Interpreted CPU analytic solution");
        }
        Json item = {{"fixed", trajectory}};
        if (int_scheme_supports_adaptive(scheme)) {
            const auto code = codegen_adaptive(sys, method.scheme);
            const bool hermite = method.scheme == Scheme::DOPRI78 || method.scheme == Scheme::DOPRI78Legacy;
            std::vector<double> grid_errors;
            for (double max_step : {.05,.025}) {
                AdaptiveSettings settings; settings.enabled = true; settings.rtol = "1e-8"; settings.atol = "1e-11";
                settings.hmax = max_step == .05 ? "0.05" : "0.025";
                UcudaAdaptParams p{}; std::string error;
                require(adaptive_build_params(settings, code, 2, .11, p, error), "Interpreted adaptive configuration failed");
                for (bool raw : {false,true}) {
                    std::vector<std::vector<double>> adaptive; std::vector<double> times, log; AdaptiveStats stats; double final_h = 0;
                    require(computePhasePortraitCPU_adaptive(evaluator, scheme, initial, 2, parameters, p, raw, .01, .1, .001,
                        101, 1024, 64, adaptive, times, log, stats, final_h), "Interpreted adaptive calculation failed");
                    require(raw ? adaptive.size() == times.size() && !times.empty() : adaptive.size() == 101 && times.empty(),
                        "Interpreted adaptive output shape");
                    double max_error = 0;
                    for (size_t i = 0; i < adaptive.size(); ++i) for (int k = 0; k < 2; ++k) {
                        const double time = raw ? times[i] : .01 + i * .001, expected = std::exp(-(k + 1) * time);
                        const double difference = std::fabs(adaptive[i][k] - expected);
                        max_error = (std::max)(max_error, difference);
                        // Cubic Hermite interpolation: |f''''| h^4 / 384. For exp(-lambda*t),
                        // |f''''| <= lambda^4; add a separate endpoint error budget of 2*rtol.
                        const double bound = hermite && !raw ? std::pow((k + 1) * max_step, 4) / 384 + 2e-8 : 2e-7;
                        if (difference > bound) {
                            std::ostringstream message; message.precision(17);
                            message << "Interpreted adaptive analytic solution: " << method.name << " raw=" << raw << " point=" << i
                                << " variable=" << k << " t=" << time << " actual=" << adaptive[i][k] << " expected=" << expected;
                            throw std::runtime_error(message.str());
                        }
                    }
                    if (!raw) grid_errors.push_back(max_error);
                    item[std::string(raw ? "adaptive_nodes_" : "adaptive_grid_") + settings.hmax] = {{"trajectory", adaptive}, {"times", times}, {"log", log},
                        {"final_h", final_h}, {"statistics", {stats.nacc,stats.nrej,stats.nforced,stats.nrhs,stats.hmin,stats.hmax,stats.hmean}},
                        {"diverged", stats.diverged}, {"truncated", stats.truncated}};
                }
            }
            if (hermite) {
                const double order = std::log2(grid_errors[0] / grid_errors[1]);
                require(order > 3.5 && order < 4.5, "Cubic Hermite dense output convergence");
                item["dense_convergence_order"] = order;
            }
        }
        result[method.name] = item;
    }
    return result;
}
Json disk_formats(const fs::path& directory) {
    fs::create_directories(directory);
    Json result = Json::object();
    data_export::Bif1DSnapshot bif; bif.values = {.5, 1. / 3, 2.125}; bif.initial_conditions = {.25, -1.5};
    bif.tMax = 1.25; bif.transientTime = .125; bif.h = .0078125; bif.preScaller = 2;
    bif.writableVar = 1; bif.indexOfMutVar = 1; bif.range_lo = -1.25; bif.range_hi = 2.5;
    const auto bif_file = directory / "bif.csv";
    { std::ofstream out(bif_file); data_export::write_bif1d_config(out, bif); }
    result["bif_config"] = read(bif_file);
    const double peaks[] = {1e-12, 123456789.12345}, times[] = {.5, 1.25};
    { std::ofstream out(bif_file); data_export::write_bif1d_rows(out, 1. / 3, 2, peaks, times);
      data_export::write_bif1d_rows(out, .75, 0, nullptr, nullptr);
      data_export::write_bif1d_rows(out, 1.25, -1, nullptr, nullptr); }
    result["bif_rows"] = read(bif_file);
    data_export::Dft1DSnapshot dft; dft.values = bif.values; dft.initial_conditions = bif.initial_conditions;
    dft.tMax = 2; dft.h = .01; dft.preScaller = 2; dft.range_lo = .5; dft.range_hi = 1.5;
    dft.freq_lo = .25; dft.freq_hi = 1.25; dft.n_freq = 3;
    const auto dft_file = directory / "dft.csv";
    { std::ofstream out(dft_file); data_export::write_dft1d_config(out, dft); }
    result["dft_config"] = read(dft_file);
    const double matrix[] = {1. / 3, -0., 1e-100, -2.5, 1e120, .125};
    { std::ofstream out(dft_file); data_export::write_dft1d_header(out, dft); data_export::write_dft1d_matrix(out, matrix, 0, 2, 3); }
    result["dft_rows"] = read(dft_file);
    const auto folder = directory.generic_u8string() + "/";
    const auto config_file = directory / "_app_config.json";
    AppConfig cfg; cfg.ui_scale_override = 1.25f; cfg.use_builtin_font = true; cfg.nvrtc_fmad = false; cfg.gpu_block_size = 64;
    require(save_app_config(folder, cfg), "Cannot save app configuration");
    result["app_config_initial"] = read(config_file);
    AppConfig loaded; require(load_app_config(folder, loaded) && save_app_config(folder, loaded), "Cannot reload app configuration");
    const auto normalized = read(config_file); result["app_config_normalized"] = normalized;
    AppConfig again; require(load_app_config(folder, again) && save_app_config(folder, again) && read(config_file) == normalized,
        "App configuration round trip changed");
    return result;
}
Json compatibility_observations() {
    SystemRecord record; record.vars_text = "x,y"; record.params_text = "p";
    record.init_conditions = {{"x","1"},{"y","1"}}; record.param_values = {{"p","1"}};
    PhaseAnalysisSession session;
    session.load_from_record(record, {"x","y"}, {"p"});
    session.sys.vars = {"x","y"}; session.sys.params = {"p"}; session.sys.rhs = {"-p*x","-2*y"};
    const std::string ordinary = R"({"step_h":"0.01","scheme":"RK4"})";
    require(session_from_json(ordinary, session), "Cannot load compatibility probe");
    const bool immediately_generated = !session.krs_code.empty();
    session.regenerate_krs();
    const auto generated = session.krs_code;
    require(!generated.empty(), "Deferred KRS regeneration failed");
    const std::string retired = R"({"scheme":"RK4","adaptive":{"enabled":true,"method":1,"rtol":"1e-8","atol":"1e-11"}})";
    require(session_from_json(retired, session), "Cannot load retired-format probe");
    return {{"ordinary_session", {{"input", ordinary}, {"immediately_generated", immediately_generated}, {"generated_after_regeneration", generated}}},
        {"retired_local_adaptive", {{"input", retired}, {"enabled", session.adaptive.enabled}, {"rtol", session.adaptive.rtol}}}};
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 3 && std::string(argv[1]) == "--compatibility-report"), "Unknown observation arguments");
        const char* root = std::getenv("UCUDA_TEST_RESULT_DIR");
        const auto directory = root && *root ? fs::u8path(root).parent_path() / "parity-observations"
            : fs::temp_directory_path() / ("ucuda-parity-observation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup {
            fs::path path; bool temporary;
            ~Cleanup() { std::error_code ec; if (temporary && path.parent_path() == fs::temp_directory_path(ec) &&
                path.filename().string().rfind("ucuda-parity-observation-", 0) == 0) fs::remove_all(path, ec); }
        } cleanup{directory, !(root && *root)};
        fs::create_directories(directory);
        const auto output = directory / "parity.observations.json";
        const Json result = {{"schema", 1}, {"generated_sources", generated_sources()},
            {"serialization", serialization()}, {"rqa", rqa_observations()}, {"interpreted_cpu", interpreted_cpu()},
            {"disk_formats", disk_formats(directory / "formats")}};
        std::ofstream out(output, std::ios::binary);
        require(bool(out) && bool(out << result.dump(2) << '\n'), "Cannot write observations");
        if (argc == 3) {
            std::ofstream compatibility(fs::u8path(argv[2]), std::ios::binary);
            require(bool(compatibility) && bool(compatibility << compatibility_observations().dump(2) << '\n'), "Cannot write compatibility report");
        }
        std::cout << "108 code generation cases, 11 session kinds, 3 window formats, 24 RQA cases, "
            "36 interpreted fixed and 16 adaptive paths, CSV/settings and rejections passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
