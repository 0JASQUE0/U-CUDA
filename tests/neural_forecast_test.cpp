#include "ucuda/analysis/neural_session.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/platform/platform_paths.h"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

std::string exe_dir() { return platform::executable_dir(); }

namespace {
constexpr int fit_samples = 2000;
constexpr int forecast_samples = 600;
constexpr int channels = 3;
constexpr double source_dt = 0.02;
constexpr double source_lle = 0.906;
constexpr std::array<int, 5> starts{2200, 2900, 3600, 4300, 5000};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

NeuralConfig fixture() {
    NeuralConfig c;
    c.schedule = NeuralSchedule::Stage;
    c.h_text = "0.1";
    c.input_hold_text = "1";
    c.t_max_text = std::to_string(fit_samples);
    c.seed_text = "1";
    c.coupling_text = "1";
    auto& pop = c.populations.emplace_back();
    pop.size_text = "96";
    pop.param_values = {{"b", "0"}};
    pop.param_spread = {{"b", "0.5"}};
    pop.initial_conditions = {{"x", "0"}};
    auto& projection = c.projections.emplace_back();
    projection.degree_text = "12";
    projection.weight_text = "1";
    projection.weight_spread_text = "2";
    c.laws = {{"tanh", {"w*tanh(x_j)"}}};
    for (int k = 0; k < channels; ++k) {
        auto& source = c.sources.emplace_back();
        source.kind = NeuralSourceKind::System;
        source.vars = {"x", "y", "z"};
        source.rhs = {"10*(y-x)", "x*(28-z)-y", "x*y-8*z/3"};
        source.initial_conditions = {{"x", "1"}, {"y", "1"}, {"z", "1"}};
        source.variable = k;
        source.name = source.vars[k];
        source.step_text = "0.01";
        source.time_scale_text = "0.02";
        c.links.push_back({k, 0, "1", "0.8", "0.33"});
    }
    c.task.enabled = true;
    c.task.kind = NeuralTaskKind::Forecast;
    c.task.washout_text = "200";
    c.task.gap_text = "50";
    c.task.features_text = "96";
    c.task.forecast_length_text = std::to_string(forecast_samples);
    c.task.lyapunov_text = "0.906";
    return neural_task_config(c);
}

NeuralPlan reservoir_plan(NeuralConfig config, int end) {
    System sys;
    sys.vars = {"x"};
    sys.params = {"b"};
    sys.rhs = {"-x+b"};
    // Sample time is one network time unit. Keep the normalisation from the fixed fitting record.
    config.t_max_text = std::to_string(end);
    return neural_plan(sys, {}, sys.vars, sys.params, config);
}

void evaluate(ParametricEngine& engine) {
    const NeuralConfig config = fixture();
    const auto plan = [&](int end) { return reservoir_plan(config, end); };
    const auto reference = plan(starts.back() + forecast_samples);
    std::vector<double> weights, lambda;
    int ar_order = 0, nvar_order = 0;
    std::array<std::vector<double>, channels> horizons;
    std::cout << std::setprecision(8) << "start_sample,source_time,reservoir,ar,nvar,reservoir_censored,ar_censored,nvar_censored\n";
    for (int start : starts) {
        auto teacher = plan(start);
        auto closed = plan(start + forecast_samples);
        require(teacher.hash == reference.hash && closed.hash == reference.hash, "graph changed between starts");
        require(teacher.request.input.samples.size() >= static_cast<size_t>(start) * channels &&
                teacher.request.input.samples.size() <= reference.request.input.samples.size() &&
                std::equal(teacher.request.input.samples.begin(), teacher.request.input.samples.end(),
                           reference.request.input.samples.begin()), "input history or normalisation changed between starts");
        teacher.request.keep_checkpoint = true;
        const NeuralResult history = engine.run_neural(teacher.request);
        require(history.ok && history.checkpoint, "teacher-forced run failed: " + history.error);
        closed.request.resume = history.checkpoint;
        NeuralTaskResult result;
        neural_forecast_prepare(config, teacher.request, history, closed.request, result, fit_samples);
        require(result.t0 == start && result.points == forecast_samples + 1, "wrong forecast window");
        require(result.ar_order > 0 && result.nvar_order > 0, "missing closed-loop baseline");
        require(std::equal(result.truth.begin(), result.truth.end(), reference.request.input.samples.begin() + start * channels),
                "forecast truth is misaligned or differently normalised");
        if (weights.empty()) {
            weights = closed.request.feedback.w;
            lambda = result.lambda;
            ar_order = result.ar_order;
            nvar_order = result.nvar_order;
        }
        require(weights == closed.request.feedback.w && lambda == result.lambda &&
                ar_order == result.ar_order && nvar_order == result.nvar_order, "fitting changed with the test start");
        if (start == starts.front()) {
            auto changed = plan(start + forecast_samples);
            for (size_t i = static_cast<size_t>(start) * channels; i < changed.request.input.samples.size(); ++i)
                changed.request.input.samples[i] += 100;
            NeuralTaskResult check;
            neural_forecast_prepare(config, teacher.request, history, changed.request, check, fit_samples);
            const auto same = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
            require(check.truth != result.truth && changed.request.feedback.w == weights &&
                    std::equal(check.predicted.begin(), check.predicted.begin() + channels, result.predicted.begin()) &&
                    std::equal(check.ar_predicted.begin(), check.ar_predicted.end(), result.ar_predicted.begin(), result.ar_predicted.end(), same) &&
                    std::equal(check.nvar_predicted.begin(), check.nvar_predicted.end(), result.nvar_predicted.begin(), result.nvar_predicted.end(), same),
                    "future truth influenced fitting or initialisation");
        }
        const NeuralResult prediction = engine.run_neural(closed.request);
        require(prediction.ok && !prediction.feedback_rows.empty(), "closed-loop run failed: " + prediction.error);
        neural_forecast_finish(prediction, result);
        const std::array<double, channels> times{result.valid_time, result.ar_valid_time, result.nvar_valid_time};
        std::cout << start << ',' << start * source_dt;
        for (int model = 0; model < channels; ++model) {
            require(std::isfinite(times[model]) && times[model] >= 0 && times[model] <= result.points,
                    "invalid forecast horizon");
            horizons[model].push_back(times[model] * source_dt * source_lle);
            std::cout << ',' << horizons[model].back();
        }
        for (double time : times) std::cout << ',' << (time == result.points);
        std::cout << '\n';
    }
    std::cout << "model,mean,min,max (Lyapunov times)\n";
    const std::array<const char*, channels> names{"reservoir", "ar", "nvar"};
    for (int model = 0; model < channels; ++model) {
        const auto& h = horizons[model];
        const auto range = std::minmax_element(h.begin(), h.end());
        std::cout << names[model] << ',' << std::accumulate(h.begin(), h.end(), 0.0) / h.size()
                  << ',' << *range.first << ',' << *range.second << '\n';
    }
}

void check_closed_loop_lyapunov(ParametricEngine& engine) {
    const double dt = 0.25, gain = 0.5;
    const double multiplier = 1 + gain * dt;
    System sys;
    sys.vars = {"x"};
    sys.rhs = {"0"};
    NeuralConfig c;
    c.populations.resize(1);
    c.populations[0].size_text = "1";
    c.populations[0].initial_conditions = {{"x", "1"}};
    c.input_hold_text = "0.25";
    c.t_max_text = "8";
    c.laws = {NetCouplingLaw{}};
    c.sources.resize(2);
    for (auto& source : c.sources) {
        source.kind = NeuralSourceKind::Constant;
        source.amplitude_text = "0";
    }
    c.links = {{1, 0, "1", "0", "1"}};
    for (auto schedule : {NeuralSchedule::Stage, NeuralSchedule::Heun, NeuralSchedule::Post, NeuralSchedule::Old}) {
        c.schedule = schedule;
        for (int hold : {2, 4}) {
            c.h_text = hold == 2 ? "0.125" : "0.0625";
            c.pre_scaller_text = std::to_string(hold);
            auto plan = neural_plan(sys, {}, sys.vars, sys.params, c);
            NeuralRequest& req = plan.request;
            req.feedback = {3, {0}, {1}, {gain, 0}};
            const NeuralResult plain = engine.run_neural(req);
            require(plain.ok && !plain.diverged && plain.feedback_rows.size() == 30, "linear feedback: " + plain.error);
            for (size_t j = 0; j < plain.feedback_rows.size(); ++j)
                require(std::abs(plain.feedback_rows[j] - gain * std::pow(multiplier, (double)j)) < 1e-10,
                        "linear closed-loop signal disagrees with the exact sample map");
            req.lyapunov.enabled = true;
            req.lyapunov.eps = 1e-5;
            req.lyapunov.interval = 0.5;
            req.lyapunov.settle = 0.5;
            req.steps_per_sync = hold + 1;
            const NeuralResult twin = engine.run_neural(req);
            require(twin.ok && !twin.diverged && twin.lyapunov.size() == 1 && twin.lyap_blocks == 13,
                    "linear closed-loop Lyapunov: " + twin.error);
            require(twin.feedback_rows == plain.feedback_rows && twin.traces == plain.traces, "clone changed the primary forecast");
            require(std::abs(twin.lyapunov.front() - std::log(multiplier) / dt) < 1e-8,
                    "closed-loop exponent disagrees with the exact sample map");
            req.lyapunov.interval = req.h;
            const NeuralResult off_grid = engine.run_neural(req);
            require(!off_grid.ok && off_grid.error.find("sample time") != std::string::npos,
                    "an interval off the sample grid was accepted");
            std::cout << neural_schedule_name(schedule) << ", hold=" << hold
                      << ", plain_ms=" << plain.wall_ms << ", clone_ms=" << twin.wall_ms << '\n';
        }
    }
}

void attractor_stats(const char* model, double h, int length, const std::vector<double>& samples,
                     size_t first, size_t last, double lle) {
    for (int k = 0; k < channels; ++k) {
        double sum = 0;
        for (size_t i = first; i < last; ++i) {
            const double v = samples[i * channels + k];
            require(std::isfinite(v), "incomplete attractor record");
            sum += v;
        }
        const double mean = sum / (last - first);
        double variance = 0, lag = 0;
        for (size_t i = first; i < last; ++i) {
            const double d = samples[i * channels + k] - mean;
            variance += d * d;
            if (i > first) lag += d * (samples[(i - 1) * channels + k] - mean);
        }
        std::cout << model << ',' << h << ',' << length << ',' << k << ',' << mean << ','
                  << std::sqrt(variance / (last - first)) << ',' << (variance > 0 ? lag / variance : 0) << ',' << lle << '\n';
    }
}

void evaluate_attractor(ParametricEngine& engine) {
    NeuralConfig config = fixture();
    constexpr int start = starts.front(), settle = 500, duration = 4000, interval = 5;
    config.task.forecast_length_text = std::to_string(duration);
    auto teacher = reservoir_plan(config, start);
    auto preparation = reservoir_plan(config, start + duration);
    const auto reference_input = preparation.request.input.samples;
    const auto history = engine.run_neural(teacher.request);
    require(history.ok && !history.diverged, "attractor fitting run failed: " + history.error);
    NeuralTaskResult targets;
    neural_forecast_prepare(config, teacher.request, history, preparation.request, targets, fit_samples);
    NeuralFeedback feedback = preparation.request.feedback;
    // Both steps use this fixed readout; evaluate it at the first autonomous sample on-device.
    feedback.from_row = start;

    std::cout << std::setprecision(8) << "model,h,autonomous_samples,channel,mean,sd,lag1,lle_in_source_time\n";
    for (int length : {2000, duration})
        attractor_stats("Lorenz", 0.01, length, targets.truth, settle, length + 1, source_lle);
    for (const char* h : {"0.1", "0.05"}) {
        config.h_text = h;
        config.pre_scaller_text = std::string(h) == "0.1" ? "10" : "20";
        auto autonomous = reservoir_plan(config, start + duration);
        auto& req = autonomous.request;
        require(autonomous.hash == teacher.hash && req.observe.nodes == teacher.request.observe.nodes &&
                req.input.samples == reference_input,
                "the step study changed the graph, features or input");
        req.feedback = feedback;
        req.lyapunov.enabled = true;
        req.lyapunov.interval = interval;
        req.lyapunov.settle = settle;
        const NeuralResult run = engine.run_neural(req);
        require(run.ok && !run.diverged && run.lyap_blocks == (duration - settle) / interval &&
                run.lyap_log.size() == (size_t)run.lyap_blocks && run.feedback_rows.size() == targets.truth.size(),
                "autonomous attractor run failed: " + run.error);
        std::cerr << "Reservoir h=" << req.h << ": " << run.wall_ms << " ms, " << run.lyap_blocks << " measured intervals\n";
        for (int length : {2000, duration}) {
            const int count = (length - settle) / interval;
            const double sum = std::accumulate(run.lyap_log.begin(), run.lyap_log.begin() + count, 0.0);
            const double lle = sum / (count * run.lyap_dt * source_dt);
            require(std::isfinite(lle), "invalid autonomous exponent");
            attractor_stats("reservoir", req.h, length, run.feedback_rows, settle, length + 1, lle);
        }
    }
}

// Both equations are driven by this prescribed time function, sampled every 0.25 units.
double prescribed_input(int sample) {
    constexpr double pi = 3.14159265358979323846;
    return 0.15 + 0.8 * std::sin(2 * pi * 0.25 * (sample * 0.25));
}

std::vector<double> monolithic_pair(double h, double eps) {
    const int steps_per_sample = (int)std::lround(0.25 / h);
    std::array<double, 2> x{1.5, 1.5 + eps};
    std::vector<double> growth;
    for (int sample = 0; sample < 64; ++sample) {
        const double input = prescribed_input(sample);
        const auto rhs = [input](double v) { return -0.7 * v - 0.2 * v * v * v + input; };
        for (int step = 0; step < steps_per_sample; ++step) {
            for (double& v : x) {
                const double a = rhs(v);
                const double b = rhs(v + h * a / 2);
                const double c = rhs(v + h * b / 2);
                const double d = rhs(v + h * c);
                v += h / 6 * (a + 2 * b + 2 * c + d);
            }
        }
        if ((sample + 1) % 2 != 0) continue;
        const double delta = x[1] - x[0];
        if (sample + 1 > 16) growth.push_back(std::log(std::abs(delta) / eps));
        x[1] = x[0] + std::copysign(eps, delta);
    }
    return growth;
}

void check_conditional_lyapunov(ParametricEngine& engine) {
    constexpr double eps = 1e-6, interval = 0.5;
    const auto reference = monolithic_pair(0.0078125, eps);
    const double expected = std::accumulate(reference.begin(), reference.end(), 0.0) / (reference.size() * interval);
    require(reference.size() == 24 && std::isfinite(expected) && expected < -0.7, "invalid contracting reference");
    System sys;
    sys.vars = {"x"};
    sys.rhs = {"-0.7*x-0.2*x*x*x"};
    NeuralConfig c;
    c.schedule = NeuralSchedule::Stage;
    c.populations.resize(1);
    c.populations[0].size_text = "1";
    c.populations[0].initial_conditions = {{"x", "1.5"}};
    c.laws = {NetCouplingLaw{}};
    auto& source = c.sources.emplace_back();
    source.kind = NeuralSourceKind::Sine;
    source.amplitude_text = "0.8";
    source.frequency_text = "0.25";
    source.offset_text = "0.15";
    c.links = {{0, 0, "1", "0", "1"}};
    c.input_hold_text = "0.25";
    c.transient_text = "4";
    c.t_max_text = "12";
    std::cout << std::setprecision(10) << "monolithic CPU reference: conditional LLE=" << expected << '\n';
    for (int hold : {2, 4}) {
        c.h_text = hold == 2 ? "0.125" : "0.0625";
        c.pre_scaller_text = std::to_string(hold);
        auto plan = neural_plan(sys, {}, sys.vars, sys.params, c);
        auto& req = plan.request;
        require(req.input.stage_rows == 1 && req.input.samples.size() == 65, "wrong prescribed input grid");
        for (size_t row = 0; row < req.input.samples.size(); ++row)
            require(std::abs(req.input.samples[row] - prescribed_input((int)row)) < 1e-14, "prescribed inputs differ");
        const NeuralResult plain = engine.run_neural(req);
        req.lyapunov.enabled = true;
        req.lyapunov.eps = eps;
        req.lyapunov.interval = interval;
        req.lyapunov.settle = 4;
        req.steps_per_sync = hold + 1;
        const NeuralResult run = engine.run_neural(req);
        require(plain.ok && run.ok && !run.diverged && run.traces == plain.traces, "conditional clone changed the driven trajectory");
        require(run.lyap_log.size() == reference.size() && run.lyapunov.size() == 1, "conditional exponent has the wrong window");
        const auto matched_grid = monolithic_pair(req.h, eps);
        for (size_t block = 0; block < matched_grid.size(); ++block)
            require(std::abs(run.lyap_log[block] - matched_grid[block]) < 2e-6, "growth disagrees with the monolithic pair");
        const double error = std::abs(run.lyapunov.front() - expected);
        require(error < (hold == 2 ? 1e-5 : 1e-6), "conditional exponent disagrees with the refined CPU reference");
        std::cout << "GPU h=" << req.h << ", conditional LLE=" << run.lyapunov.front()
                  << ", error=" << error << ", run_ms=" << run.wall_ms << '\n';
    }
}

void check_reservoir_replicas(ParametricEngine& engine) {
    constexpr int horizon = 64;
    constexpr double fraction = 1e-5;
    // Keep the same training-prefix normalisation as the forecast, including past this short test.
    auto plan = reservoir_plan(fixture(), fit_samples);
    auto& req = plan.request;
    req.t_max = horizon;
    std::array<NeuralResult, 3> replicas;
    const auto initial = req.initial_conditions;
    const size_t nodes = req.observe.nodes.size();
    require(nodes == 96 && req.observe.vars.size() == 1, "the replica test needs the full reservoir state");
    for (int i = 0; i < (int)replicas.size(); ++i) {
        for (size_t j = 0; j < initial.size(); ++j) req.initial_conditions[j] = initial[j] + i - 1;
        replicas[i] = engine.run_neural(req);
        require(replicas[i].ok && !replicas[i].diverged && replicas[i].traces.size() == (horizon + 1) * nodes,
                "driven replica did not complete");
    }
    std::vector<double> distance(horizon + 1);
    for (size_t a = 0; a < replicas.size(); ++a) for (size_t b = a + 1; b < replicas.size(); ++b) {
        for (size_t p = 0; p < distance.size(); ++p) {
            double sum = 0;
            for (size_t j = 0; j < nodes; ++j) {
                const double d = replicas[a].traces[p * nodes + j] - replicas[b].traces[p * nodes + j];
                sum += d * d;
            }
            require(std::isfinite(sum), "non-finite separation");
            distance[p] = (std::max)(distance[p], std::sqrt(sum / nodes));
        }
    }
    require(distance.front() == 2, "replicas did not start apart");
    const double threshold = distance.front() * fraction;
    size_t settled = distance.size();
    while (settled > 0 && distance[settled - 1] <= threshold) --settled;
    require(settled <= horizon - 8, "replicas did not forget their initial states for the final eight samples");
    std::cout << "network_time,max_pairwise_rms\n";
    for (int time : {0, 4, 8, 16, 32, horizon}) std::cout << time << ',' << distance[time] << '\n';
    std::cout << "first sample staying below " << threshold << " through " << horizon << ": " << settled << '\n';

    req.initial_conditions = initial;
    req.transient = 16;
    req.t_max = horizon - req.transient;
    req.lyapunov.enabled = true;
    req.lyapunov.interval = 1;
    req.lyapunov.settle = req.transient;
    const NeuralResult run = engine.run_neural(req);
    require(run.ok && !run.diverged && run.lyap_blocks == 48 && run.lyapunov.size() == 1 &&
            std::isfinite(run.lyapunov.front()) && run.lyapunov.front() < 0, "expected a negative conditional reservoir exponent");
    std::cout << "reservoir conditional LLE per network time: " << run.lyapunov.front() << '\n';
}
}

int main(int argc, char** argv) {
    try {
        const bool attractor = argc == 2 && std::string(argv[1]) == "--attractor";
        const bool linear = argc == 2 && std::string(argv[1]) == "--lyapunov";
        const bool conditional = argc == 2 && std::string(argv[1]) == "--conditional";
        require(argc == 1 || attractor || linear || conditional,
                "usage: neural_forecast_test [--attractor | --lyapunov | --conditional]");
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; forecast check skipped\n"; return 77; }
        ParametricEngine engine;
        if (attractor) evaluate_attractor(engine);
        else if (linear) check_closed_loop_lyapunov(engine);
        else if (conditional) {
            check_conditional_lyapunov(engine);
            check_reservoir_replicas(engine);
        }
        else evaluate(engine);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural forecast test failed: " << e.what() << '\n';
        return 1;
    }
}
