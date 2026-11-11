#include "ucuda/analysis/neural_session.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/storage/system_library.h"
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::string exe_dir() { return platform::executable_dir(); }

namespace {
void require(bool pass, const std::string& message) {
    if (!pass) throw std::runtime_error(message);
}

struct TempReadout {
    std::filesystem::path root, file;
    TempReadout() {
        root = std::filesystem::temp_directory_path() /
               ("ucuda-classifier-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(root), "could not create private readout directory");
        file = root / "readout.json";
    }
    ~TempReadout() {
        std::error_code ec;
        std::filesystem::remove(file, ec);
        std::filesystem::remove(root, ec);
    }
};

SystemRecord lif_record() {
    SystemRecord r;
    r.name = "LIF LSM validation";
    r.vars_text = "v";
    r.mode = "Plain"; r.plain_text = "v' = -v/4";
    r.scheme_rk4 = true; r.step_h = "0.125";
    r.init_conditions = {{"v", "0"}};
    return r;
}

NeuralConfig fixture(bool spike) {
    NeuralConfig c;
    c.label = spike ? "LIF LSM validation" : "State classifier validation";
    c.schedule = spike ? NeuralSchedule::Spike : NeuralSchedule::Stage;
    c.h_text = "0.125"; c.input_hold_text = "0.125";
    c.protocol = NeuralProtocol::Trials; c.trials_text = "12";
    c.rest_text = "0.5"; c.stimulus_text = "2"; c.reset_state = true;
    c.threshold_text = "1"; c.reset_text = {"0"}; c.refractory_text = "0.25";
    c.populations.resize(1);
    c.populations[0].name = "Reservoir"; c.populations[0].size_text = "4";
    c.populations[0].initial_conditions = {{"v", "0"}};
    c.projections.resize(1);
    c.projections[0].rule = NeuralRule::Ring;
    c.projections[0].weight_text = "0.05";
    c.projections[0].delay_text = "0.25";
    c.coupling_text = "1";
    c.laws = {NetCouplingLaw{}};
    c.receptors = {NeuralReceptor{}};
    c.sources.resize(3);
    c.sources[0].name = "Low current"; c.sources[0].kind = NeuralSourceKind::Constant;
    c.sources[0].amplitude_text = "0.125";
    c.sources[1].name = "High current"; c.sources[1].kind = NeuralSourceKind::Constant;
    c.sources[1].amplitude_text = "2";
    auto& ramp = c.sources[2];
    ramp.name = "Auxiliary ramp"; ramp.kind = NeuralSourceKind::System;
    ramp.vars = {"x"}; ramp.rhs = {"1"}; ramp.skip_text = "0";
    c.links = {{0, 0, "1", "0", "0.75"}, {2, 0, "0.02", "0", "0.25"}};
    c.task.enabled = true; c.task.kind = NeuralTaskKind::Classification;
    c.task.classes = {0, 1}; c.task.bins_text = "2"; c.task.features_text = "4";
    c.task.train_text = "0.5"; c.task.val_text = "0.25";
    return c;
}

NeuralSession session(bool spike) {
    NeuralSession s;
    s.sys.vars = {"v"}; s.sys.rhs = {"-v/4"}; s.vars = s.sys.vars;
    s.configs = {fixture(spike)};
    return s;
}

void run(NeuralSession& s, ParametricEngine& engine) {
    require(s.run_async(engine, 0), "could not start classifier session");
    require(s.run_future.wait_for(std::chrono::seconds(45)) == std::future_status::ready, "classifier run timed out");
    require(s.poll(), "classifier run did not complete");
}

void legacy_and_normalisation() {
    // A v1 session could keep the regression toggle set while in Classification.
    const std::string json = R"({"configs":[{"task":{"kind":"classification","use_saved":true},"readout":{
        "graph_hash":"old", "M":1,"K":1,"targets":[0],"nodes":[0],"mean":[0],"scale":[1],"w":[2,3],"lambda":[0.1]}}]})";
    NeuralState old;
    require(session_from_json_neural(json, old) && old.configs.size() == 1, "old Neural JSON did not load");
    const auto& c = old.configs[0];
    require(c.task.use_saved && !c.task.use_saved_classifier && c.classifier.empty(), "old regression setting must not activate classifier replay");
    require(!c.readout.is_classifier() && c.readout.w == std::vector<double>({2, 3}), "old regression weights lost");
    const TempReadout file;
    neural_readout_export(file.file.string(), c.readout);
    const auto regression = neural_readout_import(file.file.string());
    require(!regression.is_classifier() && regression.w == c.readout.w, "v1 regression export changed");

    auto model = fixture(false);
    model.sources[0].kind = NeuralSourceKind::System;
    model.sources[0].vars = {"x"}; model.sources[0].rhs = {"1"}; model.sources[0].skip_text = "0";
    const auto prepared = neural_task_config(model);
    // Six training trials: class 0 sees three cycles of 2.5, auxiliary input sees six stimuli of 2.
    // At the RK4 half step (1/16), the mean of the first N ramp samples is (N-1)/32.
    require(prepared.sources[0].norm_fixed && std::abs(prepared.sources[0].norm_mean - 119.0 / 32) < 1e-12,
            "class scaler included validation or test trials");
    require(prepared.sources[2].norm_fixed && std::abs(prepared.sources[2].norm_mean - 191.0 / 32) < 1e-12,
            "auxiliary scaler included validation or test trials");
    require(prepared.sources[0].norm_sd > 0 && prepared.sources[2].norm_sd > 0, "bad training scaler");
    std::cout << "Legacy Neural JSON and training-only normalisation passed\n";
}

void verify_trial_reset(NeuralSession& trained, ParametricEngine& engine) {
    const auto& c = trained.configs[0];
    const auto job = neural_task_config(c);
    auto plan = neural_plan(trained.sys, {}, trained.vars, trained.params, job);
    require(plan.request.reset_at.size() == 12, "expected one reset per trial");
    const long long steps = 20;
    const size_t row_width = (size_t)plan.request.input.channels * plan.request.input.stage_rows;
    // Replaying isolated trials must match the log in the combined run, including ring and refractory reset.
    for (int tr : {1, 5, 11}) {
        auto single = plan.request;
        const auto begin = plan.request.input.samples.begin() + (size_t)tr * steps * row_width;
        single.input.samples.assign(begin, begin + (size_t)steps * row_width);
        single.t_max = steps * single.h; single.reset_at.clear();
        const auto isolated = engine.run_neural(single);
        require(isolated.ok && isolated.spikes_dropped == 0, "isolated LIF trial: " + isolated.error);
        std::vector<long long> step;
        std::vector<int> node;
        for (size_t i = 0; i < c.result.spike_step.size(); ++i) {
            const long long at = c.result.spike_step[i] - tr * steps;
            if (at >= 0 && at < steps) { step.push_back(at); node.push_back(c.result.spike_node[i]); }
        }
        require(step == isolated.spike_step && node == isolated.spike_node, "trial reset differs from isolated LIF run");
    }
}

void replay(bool spike, ParametricEngine& engine) {
    auto trained = session(spike);
    auto& t = trained.configs[0];
    t.task.use_saved = true; // Must be ignored by classification, even for legacy sessions.
    run(trained, engine);
    require(t.last_run_ok, t.last_error);
    const auto& timeline = t.result_context.trials;
    require(timeline.first_step == 0 && timeline.rest_steps == 4 && timeline.stimulus_steps == 16 && timeline.count == 12,
            "trial timing for Activity differs from the fixture protocol");
    require(timeline.classes == neural_class_labels(t) && timeline.class_names == t.task_result.class_names,
            "trial annotations differ from the classes used in the run");
    require(!t.task_result.model.empty() && t.task_result.model.is_classifier(), "train did not produce a classifier");
    require(t.task_result.model.nodes.size() == 4 && t.task_result.model.M == 8, "wrong feature layout");
    require(t.task_result.test == std::vector<double>{1.0}, "fixture classes were not separated");
    if (spike) {
        verify_trial_reset(trained, engine);
        int rest_spikes = 0, low_spikes = 0, high_spikes = 0;
        const long long cycle = timeline.rest_steps + timeline.stimulus_steps;
        for (long long step : t.result.spike_step) {
            const long long at = step - timeline.first_step;
            require(at >= 0 && at / cycle < timeline.count, "spike outside the annotated trials");
            if (at % cycle < timeline.rest_steps) ++rest_spikes;
            else if (timeline.classes[(size_t)(at / cycle)] == 0) ++low_spikes;
            else ++high_spikes;
        }
        require(rest_spikes == 0 && low_spikes == 0 && high_spikes == 30, "LIF fixture response does not match its trial bands");
        std::cout << "Spikes by annotated interval: rest " << rest_spikes << ", low " << low_spikes << ", high " << high_spikes << '\n';
    }
    t.classifier = t.task_result.model;
    t.task.use_saved_classifier = true;
    const auto json = session_to_json_neural(trained);
    auto saved = session(spike);
    require(session_from_json_neural(json, saved) && session_to_json_neural(saved) == json, "classifier JSON round trip differs");
    auto& s = saved.configs[0];
    require(s.classifier.w == t.classifier.w && s.classifier.class_input_norm == t.classifier.class_input_norm,
            "weights or scaler changed after loading");
    const TempReadout file;
    neural_readout_export(file.file.string(), s.classifier);
    s.classifier = neural_readout_import(file.file.string());
    require(session_to_json_neural(saved) == json, "v2 classifier export changed the saved model");
    run(saved, engine);
    require(s.last_run_ok, s.last_error);
    require(s.result_context.trials.classes == timeline.classes && s.result_context.trials.class_names == timeline.class_names,
            "classifier replay changed the trial annotations");
    const auto& actual = s.task_result;
    require(actual.applied && actual.model.empty() && actual.val.empty() && actual.lambda.empty() && actual.delay_test.empty(),
            "application must not fit the classifier or the baseline");
    require(actual.truth == t.task_result.truth && actual.predicted == t.task_result.predicted
            && actual.confusion == t.task_result.confusion && actual.test == t.task_result.test, "replayed test differs");
    require(saved.configs[0].input_view == trained.configs[0].input_view, "replayed input differs");

    // Decisive check: changing only the saved weights changes the prediction instead of fitting a new classifier.
    std::fill(s.classifier.w.begin(), s.classifier.w.end(), 0.0);
    s.classifier.w[(size_t)s.classifier.M] = 1.0;
    run(saved, engine);
    require(s.last_run_ok && s.task_result.applied, s.last_error);
    require(s.task_result.predicted == std::vector<double>(s.task_result.truth.size(), 0.0)
            && s.task_result.predicted != t.task_result.predicted, "replay ignored saved weights");

    const auto reject = [&](const std::string& error) {
        run(saved, engine);
        require(!s.last_run_ok && s.last_error.find(error) != std::string::npos, "expected " + error + ", got: " + s.last_error);
    };
    s.classifier = t.classifier;
    s.task.bins_text = "1"; reject("features differ"); s.task.bins_text = "2";
    std::swap(s.task.classes[0], s.task.classes[1]); reject("class order differs"); std::swap(s.task.classes[0], s.task.classes[1]);
    s.task.data_seed_text = "87"; reject("trial protocol"); s.task.data_seed_text = t.task.data_seed_text;
    s.reset_state = false; reject("trial protocol"); s.reset_state = true;
    s.sources[1].amplitude_text = "3"; reject("trial protocol"); s.sources[1].amplitude_text = "2";
    s.coupling_text = "0.5"; reject("another graph"); s.coupling_text = t.coupling_text;
    s.task.control_k0 = true; reject("K = 0 control");

    std::cout << t.label << ": accuracy " << t.task_result.test[0] << ", confusion:";
    for (double n : t.task_result.confusion) std::cout << ' ' << n;
    std::cout << "; " << t.result.spike_step.size() << " spikes; classifier replay passed\n";
}

void write_fixture(const std::string& path) {
    require(std::filesystem::create_directory(path), "fixture directory must not exist and its parent must exist");
    SystemLibrary library(path);
    const auto name = library.save(lif_record());
    library.save_session(name, "_last_neural", session_to_json_neural(session(true)));
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-lif-library") { write_fixture(argv[2]); return 0; }
        const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
        require(cpu || argc == 1, "usage: neural_classifier_test [--cpu | --write-lif-library NEW_DIRECTORY]");
        legacy_and_normalisation();
        if (cpu) return 0;
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; classifier replay skipped\n"; return 77; }
        ParametricEngine engine;
        replay(false, engine);
        replay(true, engine);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural classifier test failed: " << e.what() << '\n';
        return 1;
    }
}
