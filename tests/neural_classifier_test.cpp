#include "ucuda/analysis/neural_session.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/storage/neural_dataset_io.h"
#include "ucuda/storage/system_library.h"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
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

void dataset_csv() {
    const TempReadout temp;
    const auto load = [&](const std::string& csv) {
        std::ofstream file(temp.file, std::ios::binary);
        file << csv;
        file.close();
        require(file.good(), "could not write the dataset fixture");
        return load_neural_dataset_csv(temp.file);
    };
    const std::string header = "\xef\xbb\xbf" "id,split,label,group,p0,p1\r\n";
    const std::string samples = "b,test,digit-1,,0.25,1\r\n"
                                "a,train,digit-0,writer-a,0,0.5\r\n"
                                "c,validation,digit-0,writer-b,1,0\r\n";
    const auto data = load(header + samples);
    require(data.channels == std::vector<std::string>{"p0", "p1"} &&
            data.values == std::vector<double>{0.25, 1, 0, 0.5, 1, 0}, "dataset channel order changed");
    require(data.samples.size() == 3 && data.samples[0].id == "b" && data.samples[0].label == "digit-1" &&
            data.samples[0].split == NeuralDatasetSplit::Test && data.samples[1].group == "writer-a" &&
            data.samples[1].split == NeuralDatasetSplit::Train &&
            data.samples[2].split == NeuralDatasetSplit::Validation, "dataset metadata or file order changed");
    const auto reject = [&](const std::string& csv, const std::string& reason) {
        try { load(csv); }
        catch (const std::runtime_error& e) {
            require(std::string(e.what()).find(reason) != std::string::npos, e.what());
            return;
        }
        throw std::runtime_error("invalid dataset accepted: " + reason);
    };
    reject(header + samples + "a,train,digit-1,,0,1", "unique");
    reject(header + samples + "d,test,digit-1,writer-a,0,1", "cross the split");
    reject(header + samples + "d,train,digit-1,,nan,1", "finite");
    reject(header + samples + "d,train,digit-1,,0", "number of columns");
    reject(header + "a,train,digit-0,,0,1", "each contain samples");
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

NeuralSession external_session(bool spike) {
    auto s = session(spike);
    auto& c = s.configs.front();
    c.projections.clear();
    c.sources.clear();
    c.task.classes.clear();
    c.links = {{0, 0, "1", "0", "0.5"}, {1, 0, "1", "0", "0.5"}};
    const TempReadout temp;
    std::ofstream file(temp.file);
    file << "id,split,label,group,pixel0,pixel1\n";
    int id = 0;
    for (int i = 0; i < 6; ++i) {
        for (const char* split : {"test", "train", "validation"})
            file << id++ << ',' << split << (i % 2 ? ",B,,2,0\n" : ",A,,0,2\n");
    }
    file.close();
    require(file.good(), "could not write external classification fixture");
    c.task.dataset = std::make_shared<NeuralDataset>(load_neural_dataset_csv(temp.file));
    return s;
}

void external_trial_grid() {
    auto s = external_session(false);
    auto& c = s.configs.front();
    c.task.dataset_path = "samples.csv";
    NeuralState loaded;
    require(session_from_json_neural(session_to_json_neural(s), loaded) &&
            loaded.configs.size() == 1 && loaded.configs.front().task.dataset_path == c.task.dataset_path &&
            !loaded.configs.front().task.dataset,
            "dataset path did not round-trip without embedding sample data");
    bool needs_load = false;
    try { neural_task_config(loaded.configs.front()); }
    catch (const std::runtime_error& e) { needs_load = std::string(e.what()).find("load the external dataset") != std::string::npos; }
    require(needs_load, "a missing dataset should stop the run");
    const auto job = neural_task_config(c);
    const auto plan = neural_plan(s.sys, {}, s.vars, s.params, job);
    const auto& request = plan.request;
    require(job.trials_text == "18" && request.reset_at.size() == 18 && request.reset_at[1] == 20,
            "external samples must determine trial count and reset boundaries");
    require(request.input.stage_rows == 1 && request.input.node_channel.size() == 4, "wrong external channel grid");
    const auto& samples = request.input.samples;
    require(samples[0] == 0 && samples[1] == 0 && samples[8] == 0 && samples[9] == 2 &&
            samples[48] == 2 && samples[49] == 0, "external samples were not held over the intended stimulus");
    auto relabelled = std::make_shared<NeuralDataset>(*c.task.dataset);
    for (auto& sample : relabelled->samples) sample.label = sample.label == "A" ? "B" : "A";
    c.task.dataset = relabelled;
    const auto other = neural_plan(s.sys, {}, s.vars, s.params, neural_task_config(c));
    require(other.request.input.samples == samples && other.request.input.node_channel == request.input.node_channel &&
            other.request.input.node_gain == request.input.node_gain, "class labels affected external input");
    require(neural_class_labels(neural_task_config(c)).front() == 1, "external labels did not follow explicit sample order");

    c.task.dataset_frames = 2;
    c.stimulus_text = "0.25";
    c.links.resize(1);
    c.links.front().fraction_text = "1";
    const auto sequential = neural_plan(s.sys, {}, s.vars, s.params, neural_task_config(c));
    require(sequential.request.input.channels == 1 && sequential.request.input.samples[4] == 0 &&
            sequential.request.input.samples[5] == 2 && sequential.request.input.samples[6] == 0,
            "external vector frames must arrive in column order, with zero in the next rest");
    require(session_from_json_neural(session_to_json_neural(s), loaded) &&
            loaded.configs.front().task.dataset_frames == 2, "external frame count did not round-trip");
    neural_validate_dataset_presentation(c);
    c.stimulus_text = "0.5";
    try {
        neural_validate_dataset_presentation(c);
        throw std::logic_error("an extra input step was accepted in frame mode");
    } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find("frame count times the input sample time") != std::string::npos,
                "wrong diagnostic for a stimulus that does not match the frame grid");
    }
    c.task.dataset_frames = 3;
    try {
        neural_validate_dataset_presentation(c);
        throw std::logic_error("a dataset width incompatible with the frame count was accepted");
    } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find("width must be divisible") != std::string::npos,
                "wrong diagnostic for an incompatible frame count");
    }
}

void external_result_snapshot(ParametricEngine& engine) {
    auto s = external_session(false);
    auto& c = s.configs.front();
    auto data = std::make_shared<NeuralDataset>(*c.task.dataset);
    std::fill(data->values.begin(), data->values.end(), 0.0);
    c.task.dataset = data;
    run(s, engine);
    require(c.last_run_ok, c.last_error);
    c.task.dataset.reset();
    const auto& result = c.task_result;
    require(result.test_sample_ids == std::vector<std::string>{"0", "3", "6", "9", "12", "15"},
            "the completed run lost its original test sample IDs");
    require(!result.external_test_key.empty(), "the completed test split cannot be compared");
    const auto views = neural_result_views(c);
    require(!views.empty() && views.front() == NeuralResultView::Readout &&
            std::find(views.begin(), views.end(), NeuralResultView::Input) != views.end() &&
            std::find(views.begin(), views.end(), NeuralResultView::Nodes) != views.end(),
            "classification must retain input and node views");
    c.sweep_values = {1, 2};
    require(neural_result_views(c).back() == NeuralResultView::Sweep, "classification must retain sweep results");
    c.sweep_values.clear();
    const int predicted = (int)result.predicted.front();
    const auto samples = neural_classification_errors(result, 0, 1 - predicted, predicted);
    const auto cell = (size_t)(1 - predicted) * 2 + predicted;
    require(!samples.empty() && samples.size() == (size_t)result.confusion[cell],
            "the drill-down does not reproduce the selected confusion cell");
    for (size_t sample : samples)
        require(data->samples[3 * sample].id == result.test_sample_ids[sample] &&
                data->samples[3 * sample].label == result.class_names[(size_t)(1 - predicted)],
                "an error opened a sample from the wrong class or split");
    require(neural_classification_errors(result, 1, 1 - predicted, predicted).empty() &&
            neural_classification_errors(result, 0, predicted, predicted).empty(),
            "invalid instance or diagonal cell gave a drill-down");
    const std::string key = result.external_test_key;
    auto changed = std::make_shared<NeuralDataset>(*data);
    changed->values[0] = 0.5;
    c.task.dataset = changed;
    run(s, engine);
    require(c.last_run_ok && c.task_result.external_test_key != key,
            "different test vectors would be grouped in the same comparison");
}

void external_replay(bool spike, ParametricEngine& engine) {
    auto s = external_session(spike);
    auto& c = s.configs.front();
    run(s, engine);
    require(c.last_run_ok, c.last_error);
    const auto trained = c.task_result;
    require(c.result_context.sources == std::vector<std::string>{"pixel0", "pixel1"},
            "the result lost the dataset channel labels");
    if (spike) {
        std::array<std::vector<std::pair<long long, int>>, 18> events;
        for (size_t i = 0; i < c.result.spike_step.size(); ++i) {
            const auto step = c.result.spike_step[i];
            events.at((size_t)(step / 20)).emplace_back(step % 20, c.result.spike_node[i]);
        }
        require(!events[0].empty() && events[0] == events[2] && events[0] == events[6],
                "repeated external samples did not reset spiking state");
    } else {
        const auto& traces = c.result.traces;
        const size_t nodes = c.result.observed.size();
        require(nodes == 4 && c.result.n_points == 91, "unexpected feature grid");
        require(std::equal(traces.begin() + 3 * nodes, traces.begin() + 6 * nodes, traces.begin() + 13 * nodes),
                "repeated external samples did not reset continuous state");
    }
    require(trained.class_names == std::vector<std::string>{"A", "B"} &&
            trained.test == std::vector<double>{1} && trained.confusion == std::vector<double>{3, 0, 0, 3},
            "external classifier did not distinguish the two vectors");
    require(!trained.model.empty(), "external classifier was not saved in the result");
    if (!spike) {
        const auto batched = neural_classify_batches(engine, s.sys, {}, c, 5);
        require(batched.task.predicted == trained.predicted && batched.task.confusion == trained.confusion &&
                batched.task.model.w == trained.model.w && batched.task.model.class_protocol == trained.model.class_protocol,
                "batch extraction changed the classifier");
    }
    c.classifier = trained.model;
    const TempReadout saved;
    neural_readout_export(saved.file.string(), c.classifier);
    c.classifier = neural_readout_import(saved.file.string());
    c.task.use_saved_classifier = true;
    run(s, engine);
    require(c.last_run_ok && c.task_result.applied && c.task_result.predicted == trained.predicted &&
            c.task_result.confusion == trained.confusion, "saved external classifier did not reproduce the test");
    if (!spike) {
        const auto replayed = neural_classify_batches(engine, s.sys, {}, c, 7);
        require(replayed.task.applied && replayed.task.predicted == trained.predicted,
                "batch replay did not match the saved classifier");
    }

    auto changed = std::make_shared<NeuralDataset>(*c.task.dataset);
    changed->samples.front().label = "B";
    c.task.dataset = changed;
    run(s, engine);
    require(!c.last_run_ok && c.last_error.find("saved classifier") != std::string::npos,
            "saved classifier accepted changed test metadata");

    c.task.use_saved_classifier = false;
    changed->values.front() = 1;
    run(s, engine);
    require(c.last_run_ok && c.task_result.model.w == trained.model.w &&
            c.task_result.model.mean == trained.model.mean && c.task_result.model.scale == trained.model.scale &&
            c.task_result.model.lambda == trained.model.lambda, "held-out data changed the trained classifier");
}

NeuralSession mnist_session() {
    constexpr int pixels = 784;
    auto s = session(false);
    auto& c = s.configs.front();
    c.label = "MNIST pilot";
    c.populations[0].size_text = "784";
    c.projections = {NeuralProjection{}};
    auto& projection = c.projections.front();
    projection.degree_text = "16";
    projection.weight_spread_text = "2";
    c.laws = {{"tanh", {"w*tanh(v_j)"}}};
    c.links.clear();
    for (int q = 0; q < pixels; ++q) {
        c.links.push_back({q, 0, "1", "0", "0.0012755102040816326"});
    }
    c.task.features_text = "784";
    c.task.bins_text = "1";
    return s;
}

void mnist_pilot(const std::filesystem::path& path, ParametricEngine& engine) {
    constexpr int pixels = 784;
    auto s = mnist_session();
    auto& c = s.configs.front();
    c.task.dataset = std::make_shared<NeuralDataset>(load_neural_dataset_csv(path));
    require(c.task.dataset->channels.size() == pixels && c.task.dataset->samples.size() == 300,
            "expected the fixed 300-image MNIST pilot");
    for (int q = 0; q < pixels; ++q)
        require(c.task.dataset->channels[q] == "pixel" + std::to_string(q), "wrong MNIST pixel order");
    double raw_accuracy = -1;
    for (const char* seed : {"1", "2", "3"}) {
        c.seed_text = seed;
        c.coupling_text = "0";
        const auto control = neural_plan(s.sys, {}, s.vars, s.params, neural_task_config(c));
        auto routed = control.request.input.node_channel;
        std::sort(routed.begin(), routed.end());
        require(routed.size() == pixels, "unexpected MNIST node count");
        for (int q = 0; q < pixels; ++q) require(routed[q] == q, "MNIST routing must cover every channel once");

        c.coupling_text = "0.5";
        const auto recurrent = neural_plan(s.sys, {}, s.vars, s.params, neural_task_config(c));
        const auto& a = control.request;
        const auto& b = recurrent.request;
        require(a.edge_start == b.edge_start && a.edge_src == b.edge_src && a.edge_w.size() == pixels * 16 &&
                std::all_of(a.edge_w.begin(), a.edge_w.end(), [](double w) { return w == 0; }) &&
                std::any_of(b.edge_w.begin(), b.edge_w.end(), [](double w) { return w != 0; }),
                "MNIST control and recurrent graph differ beyond edge weights");
        require(a.initial_conditions == b.initial_conditions && a.reset_at == b.reset_at &&
                a.input.node_channel == b.input.node_channel && a.input.node_gain == b.input.node_gain &&
                a.input.samples == b.input.samples && a.observe.nodes == b.observe.nodes,
                "MNIST comparison changed input, initial states, resets or features");

        std::vector<double> truth;
        for (const char* coupling : {"0", "0.5"}) {
            c.coupling_text = coupling;
            c.task.use_saved_classifier = false;
            run(s, engine);
            require(c.last_run_ok, c.last_error);
            const auto trained = c.task_result;
            require(trained.features == pixels && trained.points == 50 && trained.class_names.size() == 10 &&
                    trained.test.size() == 1 && std::isfinite(trained.test.front()) && trained.delay_test.size() == 1 &&
                    trained.confusion.size() == 100, "incomplete MNIST pilot result");
            if (truth.empty()) truth = trained.truth;
            require(truth == trained.truth, "MNIST test labels changed between variants");
            if (raw_accuracy < 0) raw_accuracy = trained.delay_test.front();
            require(raw_accuracy == trained.delay_test.front(), "raw pixel accuracy changed with graph or seed");

            c.classifier = trained.model;
            const TempReadout saved;
            neural_readout_export(saved.file.string(), c.classifier);
            c.classifier = neural_readout_import(saved.file.string());
            c.task.use_saved_classifier = true;
            run(s, engine);
            require(c.last_run_ok && c.task_result.applied && c.task_result.predicted == trained.predicted &&
                    c.task_result.confusion == trained.confusion, "MNIST classifier replay changed the test result");
            std::cout << "MNIST seed=" << seed << " K=" << coupling << " accuracy=" << trained.test.front()
                      << " raw=" << raw_accuracy << " validation=" << trained.val.front()
                      << " lambda=" << trained.lambda.front() << '\n';
            std::cout << "confusion rows (true digit, predicted digit):\n";
            for (int i = 0; i < 10; ++i)
                for (int j = 0; j < 10; ++j)
                    std::cout << trained.confusion[(size_t)i * 10 + j] << (j == 9 ? '\n' : ',');
        }
    }
}

void write_fixture(const std::string& path) {
    require(std::filesystem::create_directory(path), "fixture directory must not exist and its parent must exist");
    SystemLibrary library(path);
    const auto name = library.save(lif_record());
    library.save_session(name, "_last_neural", session_to_json_neural(session(true)));
}

void write_mnist_sessions(const std::filesystem::path& folder, const std::string& dataset) {
    require(std::filesystem::create_directory(folder), "MNIST configuration directory must be new");
    auto s = mnist_session();
    auto& c = s.configs.front();
    c.task.dataset_path = dataset;
    // Each image is constant over the stimulus. This keeps the input grid on the rest/stimulus boundaries.
    c.input_hold_text = "0.5";
    std::ofstream system(folder / "system.json");
    system << record_to_json(lif_record());
    system.close();
    require(system.good(), "cannot write MNIST system record");
    for (const char* coupling : {"0", "0.5"}) {
        c.coupling_text = coupling;
        std::ofstream output(folder / (std::string("mnist-k") + coupling + ".json"));
        output << session_to_json_neural(s);
        output.close();
        require(output.good(), "cannot write MNIST batch session");
    }
}

void write_mnist_column_sessions(const std::filesystem::path& folder, const std::string& dataset) {
    require(std::filesystem::create_directory(folder), "MNIST configuration directory must be new");
    auto s = mnist_session();
    auto& c = s.configs.front();
    c.label = "MNIST columns";
    c.task.dataset_path = dataset;
    c.task.dataset_frames = 28;
    c.populations[0].size_text = "280";
    c.task.features_text = "280";
    c.task.bins_text = "4";
    c.rest_text = "0";
    c.stimulus_text = "3.5";
    c.links.clear();
    for (int row = 0; row < 28; ++row)
        c.links.push_back({row, 0, "4", "0", "0.03571428571428571"});
    auto record = lif_record();
    record.name = "MNIST column reservoir";
    record.plain_text = "v' = -2*v";
    std::ofstream system(folder / "system.json");
    system << record_to_json(record);
    system.close();
    require(system.good(), "cannot write MNIST column system record");
    for (const char* coupling : {"0", "0.5"}) {
        c.coupling_text = coupling;
        std::ofstream output(folder / (std::string("mnist-k") + coupling + ".json"));
        output << session_to_json_neural(s);
        output.close();
        require(output.good(), "cannot write MNIST column session");
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-lif-library") { write_fixture(argv[2]); return 0; }
        if (argc == 4 && std::string(argv[1]) == "--write-mnist-sessions") {
            write_mnist_sessions(std::filesystem::u8path(argv[2]), argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--write-mnist-column-sessions") {
            write_mnist_column_sessions(std::filesystem::u8path(argv[2]), argv[3]);
            return 0;
        }
        const bool cpu = argc == 2 && std::string(argv[1]) == "--cpu";
        const bool mnist = argc == 3 && std::string(argv[1]) == "--mnist-pilot";
        require(cpu || mnist || argc == 1, "usage: neural_classifier_test [--cpu | --write-lif-library NEW_DIRECTORY | --mnist-pilot CSV]");
        if (mnist) {
            if (cuInit(0) != CUDA_SUCCESS) return 77;
            ParametricEngine engine;
            mnist_pilot(std::filesystem::u8path(argv[2]), engine);
            return 0;
        }
        legacy_and_normalisation();
        dataset_csv();
        external_trial_grid();
        if (cpu) return 0;
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; classifier replay skipped\n"; return 77; }
        ParametricEngine engine;
        replay(false, engine);
        replay(true, engine);
        external_replay(false, engine);
        external_replay(true, engine);
        external_result_snapshot(engine);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural classifier test failed: " << e.what() << '\n';
        return 1;
    }
}
