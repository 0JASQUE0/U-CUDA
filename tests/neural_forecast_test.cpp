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
}

int main(int argc, char** argv) {
    try {
        const bool attractor = argc == 2 && std::string(argv[1]) == "--attractor";
        const bool linear = argc == 2 && std::string(argv[1]) == "--lyapunov";
        require(argc == 1 || attractor || linear, "usage: neural_forecast_test [--attractor | --lyapunov]");
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; forecast check skipped\n"; return 77; }
        ParametricEngine engine;
        if (attractor) evaluate_attractor(engine);
        else if (linear) check_closed_loop_lyapunov(engine);
        else evaluate(engine);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural forecast test failed: " << e.what() << '\n';
        return 1;
    }
}
