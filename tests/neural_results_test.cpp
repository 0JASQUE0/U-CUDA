#include "ucuda/application/app_model.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

std::string exe_dir() { return platform::executable_dir(); }

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void close_to(double a, double b, const char* why) { require(std::abs(a - b) <= 1e-12, why); }

void spike_statistics() {
    NeuralResult r;
    r.n_nodes = 5; r.steps_done = 16; r.h = 0.5;
    r.spike_step = {0, 0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 10, 12};
    r.spike_node = {0, 2, 3, 1, 0, 2, 2, 3, 0, 0, 1,  0,  0};
    const auto s = neural_spike_statistics(r, {0, 2, 4, 5}, {0, 6});
    require(s.complete && s.populations.size() == 3, "complete statistics expected");
    const auto& a = s.populations[0];
    const auto& b = s.populations[1];
    require(a.intervals == 4 && a.neurons_with_cv == 1, "intervals crossed a trial reset or included a sparse neuron");
    close_to(a.mean_isi, 1.375, "wrong ISI in physical time");
    close_to(a.mean_cv, std::sqrt(0.6875) / 2.75, "wrong CV of unequal intervals");
    require(b.intervals == 3 && b.neurons_with_cv == 1, "CV requires two intervals per neuron");
    close_to(b.mean_isi, 4.0 / 3.0, "wrong pooled ISI mean");
    close_to(b.mean_cv, 0, "regular spike train must give CV 0");
    require(b.cv.counts.front() == 1 && s.populations[2].isi.counts.empty() && s.populations[2].cv.counts.empty(), "empty population histogram");
    for (const auto& p : s.populations) {
        require(std::accumulate(p.isi.counts.begin(), p.isi.counts.end(), 0ULL) == p.intervals, "ISI histogram lost intervals");
        require(std::accumulate(p.cv.counts.begin(), p.cv.counts.end(), 0ULL) == (unsigned long long)p.neurons_with_cv,
                "CV histogram lost neurons");
    }
    r.spikes_dropped = 1;
    require(!neural_spike_statistics(r, {0, 5}, {}).complete, "truncated record must not report ISI or CV");
    r.spikes_dropped = 0; r.spike_node[0] = -1;
    require(!neural_spike_statistics(r, {0, 5}, {}).complete, "invalid node id must be rejected");
}

void binned_rates() {
    NeuralConfig c;
    c.pop_start = {0, 2, 4}; c.instance_start = {0, 4}; c.pops_per_instance = 2;
    c.result.h = 0.5; c.result.steps_done = 16;
    c.result.spike_step = {0, 5, 15}; c.result.spike_node = {0, 1, 3};
    const auto rates = neural_population_rates(c, 0, 3.0);
    require(rates.points == 3 && rates.values.size() == 6, "wrong number of rate bins");
    close_to(rates.values[0], 1.0 / 6, "population A, first bin");
    close_to(rates.values[2], 1.0 / 6, "spike on bin boundary");
    close_to(rates.values[5], 1.0 / 4, "partial final bin or final spike is wrong");
    require(neural_population_rates(c, 1, 3).values.empty(), "invalid instance unexpectedly has a rate");
}

bool has(const NeuralConfig& c, NeuralResultView view) {
    const auto views = neural_result_views(c);
    return std::find(views.begin(), views.end(), view) != views.end();
}

void availability_and_key() {
    using View = NeuralResultView;
    NeuralConfig c;
    require(neural_result_views(c).empty(), "no successful result must have no tabs");
    const auto before = neural_settings_key(c);
    c.csv_output_path = "another.csv"; c.csv_save_enabled = true; c.model_text = "last run";
    c.graph_hash = "last run"; c.readout.w = {3};
    require(neural_settings_key(c) == before, "output paths, previous result or an inactive model made the result stale");
    c.h_text = "0.125";
    require(neural_settings_key(c) != before, "step change did not make result stale");
    c.task.enabled = true; c.task.use_saved = true;
    const auto applying = neural_settings_key(c);
    c.readout.w[0] = 4;
    require(neural_settings_key(c) != applying, "changing active readout weights did not invalidate result");
    c.result.ok = true; c.result.spike_count = {0}; c.schedule = NeuralSchedule::Stage;
    require(has(c, View::Activity) && has(c, View::SpikeStatistics) && !has(c, View::Means), "activity must come from the recorded run");
    c.result.spike_count.clear(); c.result.means = {1}; c.result.order = {0.5}; c.schedule = NeuralSchedule::Spike;
    require(!has(c, View::Activity) && has(c, View::Means) && has(c, View::Synchronisation), "continuous result was relabelled as spikes");
    c.task_result.forecast = true; c.task_result.names = {"x", "y"}; c.task_result.test = {0, 0};
    c.task_result.points = 2; c.task_result.truth = c.task_result.predicted = {0, 1, 2, 3};
    require(has(c, View::Readout) && has(c, View::ForecastPhase), "forecast views missing");
    c.sweep_values = {0, 1}; c.sweep2_values = {2, 3};
    require(has(c, View::Sweep), "sweep data must have a view");
}

void wait_for_run(NeuralSession& s) {
    require(s.run_future.wait_for(std::chrono::seconds(45)) == std::future_status::ready, "timed out waiting for Neural");
    require(s.poll(), "Neural poll did not finish");
}

void result_provenance() {
    AppModel model([](const std::vector<unsigned char>&) { return std::string{}; });
    auto& s = model.neural_session;
    s.sys.vars = {"x"}; s.vars = s.sys.vars; s.sys.rhs = {"-x"};
    NeuralConfig setup;
    setup.label = "first experiment"; setup.h_text = "0.125"; setup.t_max_text = "1";
    setup.laws = {NetCouplingLaw{}}; setup.populations.resize(1);
    setup.populations[0].name = "Before edit"; setup.populations[0].size_text = "2";
    setup.populations[0].initial_conditions = {{"x", "1"}};
    s.configs = {setup, setup}; s.configs[1].label = "queued experiment";

    const auto key = model.neural_result_settings_key(0);
    require(model.run_neural(0), "could not start Neural run");
    auto& first = s.configs[0];
    first.schedule = NeuralSchedule::Spike;
    first.populations[0].name = "After edit";
    wait_for_run(s);
    require(first.last_run_ok, "original Neural run failed");
    require(first.result_context.settings_key == key && first.result_context.populations == std::vector<std::string>{"Before edit"},
            "run used editable labels or settings");
    require(model.neural_result_settings_key(0) != key && has(first, NeuralResultView::Means) && !has(first, NeuralResultView::Activity),
            "view cannot distinguish old continuous result from new spike settings");
    const int previous = first.data_generation;
    first.populations[0].size_text = "0";
    require(model.run_neural(0), "could not test failed run");
    wait_for_run(s);
    require(!first.last_run_ok && first.data_generation == previous && first.result_context.settings_key == key,
            "a failed run replaced the last successful result");

    model.enqueue_neural(1);
    require(model.start_next_in_neural_queue(), "queue did not start");
    wait_for_run(s);
    const auto& second = s.configs[1];
    require(second.last_run_ok && second.result_context.settings_key == model.neural_result_settings_key(1)
            && second.result_context.label == "queued experiment" && second.data_generation != previous,
            "queued result did not retain its own identity");
    const int second_id = second.data_generation;
    model.remove_neural_config(0);
    require(s.configs.size() == 1 && s.configs[0].data_generation == second_id
            && s.configs[0].result_context.label == "queued experiment", "reindexing lost the result association");

    auto& spiking = s.configs[0];
    spiking.schedule = NeuralSchedule::Spike;
    spiking.populations[0].spike_source = true;
    spiking.populations[0].rate_text = "8"; // p = 1 per step, two perfectly regular spike trains
    spiking.keep_checkpoint = true;
    require(model.run_neural(0), "could not start regular-spike run");
    wait_for_run(s);
    require(spiking.last_run_ok && has(spiking, NeuralResultView::Activity) && spiking.spike_statistics.complete,
            "successful spiking run did not publish Activity and statistics");
    require(spiking.result_context.trials.count == 0, "continuous spiking must not have trial bands");
    const auto& stats = spiking.spike_statistics.populations.at(0);
    require(stats.intervals == 14 && stats.neurons_with_cv == 2, "wrong integrated ISI/CV sample counts");
    close_to(stats.mean_isi, 0.125, "wrong integrated ISI mean");
    close_to(stats.mean_cv, 0, "wrong integrated CV");

    spiking.t_max_text = "1.5";
    require(model.run_neural(0, true), "could not continue regular-spike run");
    wait_for_run(s);
    require(spiking.last_run_ok && spiking.result.spike_step.size() == 24 && spiking.spike_statistics.complete
            && spiking.spike_statistics.populations.at(0).intervals == 22, "continuation lost earlier spikes or the boundary interval");
    spiking.checkpoint_from_result = false; // equivalent to loading the checkpoint from disk without the matching record
    spiking.t_max_text = "2";
    require(model.run_neural(0, true), "could not continue with missing history");
    wait_for_run(s);
    require(spiking.last_run_ok && spiking.result_context.partial_history && !spiking.spike_statistics.complete
            && spiking.result.spike_step.size() == 8, "missing history must not be merged or presented as complete");
    close_to(neural_population_rates(spiking, 0, 0).t0, 1.5, "rate plot must start at the first recorded step");
}

void benchmark_statistics() {
    NeuralResult r;
    r.n_nodes = 100000; r.h = 0.125; r.steps_done = 20;
    r.spike_step.reserve(2000000); r.spike_node.reserve(2000000);
    for (int step = 0; step < r.steps_done; ++step)
        for (int node = 0; node < r.n_nodes; ++node) {
            r.spike_step.push_back(step); r.spike_node.push_back(node);
        }
    std::vector<double> times;
    for (int run = 0; run < 5; ++run) {
        const auto start = std::chrono::steady_clock::now();
        const auto s = neural_spike_statistics(r, {0, 80000, 100000}, {});
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        require(s.complete && s.populations[0].intervals == 19 * 80000ULL && s.populations[1].intervals == 19 * 20000ULL,
                "benchmark interval counts are wrong");
    }
    std::sort(times.begin(), times.end());
    std::cout << "ISI/CV: 2,000,000 events, 100,000 neurons, median " << times[times.size() / 2] << " ms\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--gpu") {
            if (cuInit(0) != CUDA_SUCCESS) return 77;
            result_provenance();
        } else if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            benchmark_statistics();
        } else {
            require(argc == 1, "usage: neural_results_test [--gpu | --benchmark]");
            spike_statistics(); binned_rates(); availability_and_key();
        }
        std::cout << "Neural results checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
