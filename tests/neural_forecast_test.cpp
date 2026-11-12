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

void evaluate(ParametricEngine& engine) {
    const NeuralConfig config = fixture();
    System sys;
    sys.vars = {"x"};
    sys.params = {"b"};
    sys.rhs = {"-x+b"};
    auto plan = [&](int end) {
        NeuralConfig run = config;
        // Sample time is one network time unit. Keep the normalisation from the fixed fitting record.
        run.t_max_text = std::to_string(end);
        return neural_plan(sys, {}, sys.vars, sys.params, run);
    };

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
}

int main() {
    try {
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; forecast check skipped\n"; return 77; }
        ParametricEngine engine;
        evaluate(engine);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural forecast test failed: " << e.what() << '\n';
        return 1;
    }
}
