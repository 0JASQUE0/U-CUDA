#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_integration.h"
#include "gui_fields.h"
#include "gui_plot.h"
#include "tab_library.h"
#include "tab_neural.h"
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ucuda::ui_detail {

// Вкладка Neural: описание сети популяциями и проекциями. Узлы и рёбра не показываются: их слишком много,
// вместо них статистика графа и наблюдаемые величины.

static constexpr size_t kNeuralRasterPoints = 2000000;   // GL рисует столько точек без задержки кадра
enum { kNeuralMeans, kNeuralNodes, kNeuralSync, kNeuralSweep, kNeuralInput, kNeuralLyap, kNeuralRate,
       kNeuralRaster, kNeuralWeights, kNeuralReadout, kNeuralForecastError, kNeuralForecastPhase, kNeuralWindowCount };
static const char* const kNeuralRules[] = { "Probability", "Fixed in-degree", "Chain", "Ring", "Grid", "King graph", "Small world" };
static const char* const kNeuralNorms[] = { "None", "This projection", "All inputs of the node" };
static const NeuralSchedule kNeuralSchedules[] = { NeuralSchedule::Stage, NeuralSchedule::Heun, NeuralSchedule::Post, NeuralSchedule::Spike };
static const NeuralSweepKind kNeuralSweeps[] = { NeuralSweepKind::Coupling, NeuralSweepKind::Parameter,
                                                 NeuralSweepKind::ParameterSpread, NeuralSweepKind::Replica,
                                                 NeuralSweepKind::InputGain };
static const NeuralSourceKind kNeuralSources[] = { NeuralSourceKind::Constant, NeuralSourceKind::Sine, NeuralSourceKind::Square,
                                                   NeuralSourceKind::Noise, NeuralSourceKind::Bits, NeuralSourceKind::System };

static void draw_neural_populations(NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Populations", ImGuiTreeNodeFlags_DefaultOpen)) return;
    int remove = -1;
    for (size_t i = 0; i < c.populations.size(); ++i) {
        NeuralPopulation& p = c.populations[i];
        ImGui::PushID((int)i);
        const std::string title = p.name + " (" + p.size_text + ")###pop";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            InputTextStr("name", p.name, kFieldW);
            InputNumStr("size", p.size_text, kFieldW);
            if (c.schedule == NeuralSchedule::Spike) {
                ImGui::Checkbox("spike source", &p.spike_source);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Nodes without dynamics: each fires at random with this rate, independently.");
                if (p.spike_source) {
                    InputNumStr("rate, per unit of time", p.rate_text, kFieldW);
                    InputNumStr("from", p.source_from_text, kFieldW);
                    InputNumStr("to (0 = to the end)", p.source_to_text, kFieldW);
                }
            }
            draw_named_num_fields("Parameters", s.params, p.param_values);
            // Id поля — имя переменной, поэтому блоки разброса нужно отделить от блоков значений.
            ImGui::PushID("spread");
            draw_named_num_fields("Parameter spread", s.params, p.param_spread, "Half-width of a uniform spread around the value.");
            ImGui::PopID();
            draw_named_num_fields("Initial conditions", s.vars, p.initial_conditions);
            ImGui::PushID("spread");
            draw_named_num_fields("Initial spread", s.vars, p.ic_spread,
                                  "Zero spread on identical nodes: a diffusive network starts synchronised and stays so.");
            ImGui::PopID();
            if (c.populations.size() > 1 && ImGui::Button("Remove population")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) {
        c.populations.erase(c.populations.begin() + remove);
        c.projections.erase(std::remove_if(c.projections.begin(), c.projections.end(),
            [&](const NeuralProjection& p) { return p.from == remove || p.to == remove; }), c.projections.end());
        for (NeuralProjection& p : c.projections) { p.from -= p.from > remove; p.to -= p.to > remove; }
    }
    if (ImGui::Button("Add population")) {
        NeuralPopulation p = c.populations.empty() ? neural_default_population(nullptr, s.vars, s.params) : c.populations.back();
        p.name = std::string(1, (char)('A' + (char)(c.populations.size() % 26)));
        c.populations.push_back(std::move(p));
    }
}

static bool neural_pop_combo(const char* label, int& index, const NeuralConfig& c) {
    std::vector<const char*> names;
    for (const NeuralPopulation& p : c.populations) names.push_back(p.name.c_str());
    ImGui::SetNextItemWidth(kFieldW);
    return ImGui::Combo(label, &index, names.data(), (int)names.size());
}

static void draw_neural_projections(NeuralConfig& c, const std::vector<std::string>& vars) {
    if (!ImGui::CollapsingHeader("Projections", ImGuiTreeNodeFlags_DefaultOpen)) return;
    int remove = -1;
    for (size_t i = 0; i < c.projections.size(); ++i) {
        NeuralProjection& p = c.projections[i];
        ImGui::PushID((int)i + 1000);
        const char* from = p.from >= 0 && p.from < (int)c.populations.size() ? c.populations[(size_t)p.from].name.c_str() : "?";
        const char* to = p.to >= 0 && p.to < (int)c.populations.size() ? c.populations[(size_t)p.to].name.c_str() : "?";
        const std::string title = std::string(from) + " -> " + to + ": " + neural_rule_name(p.rule) + "###proj";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            neural_pop_combo("from", p.from, c);
            neural_pop_combo("to", p.to, c);
            int rule = (int)p.rule;
            ImGui::SetNextItemWidth(kFieldW + 60.0f);
            if (ImGui::Combo("rule", &rule, kNeuralRules, IM_ARRAYSIZE(kNeuralRules))) p.rule = (NeuralRule)rule;
            if (p.rule == NeuralRule::Probability) InputNumStr("probability", p.p_text, kFieldW);
            if (p.rule == NeuralRule::FixedDegree) InputNumStr("in-degree", p.degree_text, kFieldW);
            if (p.rule == NeuralRule::Ring || p.rule == NeuralRule::SmallWorld) InputNumStr("neighbours each side", p.k_text, kFieldW);
            if (p.rule == NeuralRule::SmallWorld) InputNumStr("rewire probability", p.rewire_text, kFieldW);
            if (p.rule == NeuralRule::Grid || p.rule == NeuralRule::King) {
                InputNumStr("width", p.width_text, kFieldW);
                ImGui::Checkbox("periodic (torus)", &p.periodic);
            }
            if (p.rule != NeuralRule::Probability && p.rule != NeuralRule::FixedDegree)
                ImGui::Checkbox("directed arcs", &p.directed);
            InputNumStr("weight", p.weight_text, kFieldW);
            InputNumStr("relative weight spread", p.weight_spread_text, kFieldW);
            int norm = (int)p.norm;
            ImGui::SetNextItemWidth(kFieldW + 60.0f);
            if (ImGui::Combo("normalisation", &norm, kNeuralNorms, IM_ARRAYSIZE(kNeuralNorms))) p.norm = (NeuralNorm)norm;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Divide the weight by the number of inputs: of this projection,\n"
                                  "or of the node over all projections, as on the Network tab.");
            if (p.norm != NeuralNorm::None) ImGui::Checkbox("expected degree", &p.norm_expected);
            if (c.schedule == NeuralSchedule::Spike) {
                std::vector<const char*> synapses;
                for (const NeuralReceptor& q : c.receptors) synapses.push_back(q.name.c_str());
                ImGui::SetNextItemWidth(kFieldW);
                ImGui::Combo("synapse", &p.receptor, synapses.data(), (int)synapses.size());
                InputNumStr("delay", p.delay_text, kFieldW);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("A multiple of h. 0 = one step, the shortest possible.");
                std::vector<const char*> rules = { "constant weight" };
                for (const NeuralPlasticRule& q : c.plastic_rules) rules.push_back(q.name.c_str());
                int rule = p.plastic + 1;
                if (rule < 0 || rule >= (int)rules.size()) rule = 0;
                ImGui::SetNextItemWidth(kFieldW);
                if (ImGui::Combo("plasticity", &rule, rules.data(), (int)rules.size())) p.plastic = rule - 1;
            }
            else {
                if (c.schedule == NeuralSchedule::Stage) {
                    ImGui::Checkbox("device on the link", &p.device);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Instead of a coupling law: an element with its own state, see the Device block.\nThe arc weight is the gain of its current.");
                }
                if (p.device && c.schedule == NeuralSchedule::Stage) {
                    ImGui::Checkbox("one physical element", &p.one_element);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("On: an arc and its opposite arc are one element, the current leaves the source and enters the receiver.\n"
                                          "Off: every arc is its own element, the current only enters the receiver.");
                    const std::vector<const char*> items = c_str_list(vars);
                    if (p.device_var < 0 || p.device_var >= (int)items.size()) p.device_var = 0;
                    ImGui::SetNextItemWidth(kFieldW);
                    ImGui::Combo("device variable", &p.device_var, items.data(), (int)items.size());
                }
                else {
                    std::vector<const char*> laws;
                    for (const NetCouplingLaw& l : c.laws) laws.push_back(l.name.c_str());
                    ImGui::SetNextItemWidth(kFieldW);
                    ImGui::Combo("coupling law", &p.law, laws.data(), (int)laws.size());
                }
            }
            if (ImGui::Button("Remove projection")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) c.projections.erase(c.projections.begin() + remove);
    if (ImGui::Button("Add projection")) c.projections.push_back(NeuralProjection{});
}

static void draw_neural_spiking(NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Spiking", ImGuiTreeNodeFlags_DefaultOpen)) return;
    const std::vector<const char*> items = c_str_list(s.vars);
    if (c.threshold_var < 0 || c.threshold_var >= (int)items.size()) c.threshold_var = 0;
    ImGui::SetNextItemWidth(kFieldW);
    ImGui::Combo("threshold variable", &c.threshold_var, items.data(), (int)items.size());
    InputTextStr("threshold", c.threshold_text, kFieldW + 120.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A spike when the variable is at or above this expression, checked after every step.");
    c.reset_text.resize(s.vars.size());
    for (size_t k = 0; k < s.vars.size(); ++k) InputTextStr((s.vars[k] + " after a spike").c_str(), c.reset_text[k], kFieldW + 120.0f);
    ImGui::TextDisabled("Empty = the variable does not change. All resets read the state before the spike.");
    InputNumStr("refractory time", c.refractory_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A multiple of h. The node is held after a spike, its synapses keep decaying.");

    ImGui::SeparatorText("Synapses");
    int remove = -1;
    for (size_t i = 0; i < c.receptors.size(); ++i) {
        NeuralReceptor& q = c.receptors[i];
        ImGui::PushID((int)i + 5000);
        const std::string title = q.name + "###syn";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            InputTextStr("name", q.name, kFieldW);
            InputNumStr("decay time", q.tau_text, kFieldW);
            if (q.target < 0 || q.target >= (int)items.size()) q.target = 0;
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::Combo("acts on", &q.target, items.data(), (int)items.size());
            ImGui::Checkbox("conductance", &q.conductance);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off: adds gain * s to the equation.\nOn: adds gain * s * (reversal - variable).");
            if (q.conductance) InputTextStr("reversal", q.reversal_text, kFieldW + 120.0f);
            InputTextStr("gain", q.gain_text, kFieldW + 120.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("An expression over the parameters of the node, for example 1/Cm.");
            if (ImGui::Button("Remove synapse")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) {
        c.receptors.erase(c.receptors.begin() + remove);
        for (NeuralProjection& p : c.projections) { if (p.receptor == remove) p.receptor = 0; else p.receptor -= p.receptor > remove; }
    }
    if (c.receptors.size() < 8 && ImGui::Button("Add synapse")) {
        NeuralReceptor q;
        q.name = "syn " + std::to_string(c.receptors.size() + 1);
        c.receptors.push_back(std::move(q));
    }

    ImGui::SeparatorText("Plasticity");
    int removeRule = -1;
    for (size_t i = 0; i < c.plastic_rules.size(); ++i) {
        NeuralPlasticRule& q = c.plastic_rules[i];
        ImGui::PushID((int)i + 6000);
        const std::string title = q.name + "###rule";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            InputTextStr("name", q.name, kFieldW);
            InputNumStr("A+ (potentiation)", q.a_plus_text, kFieldW);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("In units of the arc weight. A spike of the receiver adds A+ * trace of the source.");
            InputNumStr("A- (depression)", q.a_minus_text, kFieldW);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("In units of the arc weight. An arriving spike takes A- * trace of the receiver.\nAt equal times the receiver goes first, so the arc is depressed.");
            InputNumStr("tau+", q.tau_plus_text, kFieldW);
            InputNumStr("tau-", q.tau_minus_text, kFieldW);
            InputNumStr("w min", q.w_min_text, kFieldW);
            InputNumStr("w max", q.w_max_text, kFieldW);
            InputNumStr("mu+", q.mu_plus_text, kFieldW);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("0: additive rule with hard bounds. 1: multiplicative rule, soft bounds.");
            InputNumStr("mu-", q.mu_minus_text, kFieldW);
            ImGui::Checkbox("nearest spike only", &q.nearest);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Off: all pairs count, the trace adds up.\nOn: the trace is 1 after a spike.");
            if (ImGui::Button("Remove rule")) removeRule = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (removeRule >= 0) {
        c.plastic_rules.erase(c.plastic_rules.begin() + removeRule);
        for (NeuralProjection& p : c.projections) { if (p.plastic == removeRule) p.plastic = -1; else p.plastic -= p.plastic > removeRule; }
    }
    if (ImGui::Button("Add rule")) {
        NeuralPlasticRule q;
        q.name = "stdp " + std::to_string(c.plastic_rules.size() + 1);
        c.plastic_rules.push_back(std::move(q));
    }
    if (!c.sweep.enabled)
        draw_csv_output_block("CSV output##neural_spike_csv", "Save spikes to file", c.csv_save_enabled, "##neural_spike_csv_path", c.csv_output_path,
                              "One row per spike: step, time, node. Written after every run.");
}

static void draw_neural_device(NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Device")) return;
    ImGui::TextDisabled("State s, voltage v = x source - x receiver, current I enters the receiver. Stage schedule only.");
    for (NeuralDevicePreset q : { NeuralDevicePreset::Vteam, NeuralDevicePreset::Team }) {
        if (q != NeuralDevicePreset::Vteam) ImGui::SameLine();
        if (ImGui::Button(neural_device_preset_name(q))) c.device = neural_device_preset(q);
    }
    InputTextStr("state", c.device.state_name, kFieldW);
    InputTextStr("d state / dt", c.device.rhs_text, kFieldW + 240.0f);
    InputTextStr("current I", c.device.current_text, kFieldW + 240.0f);
    InputTextStr("parameters", c.device.params_text, kFieldW + 240.0f);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("name = value, separated by commas.");
    InputNumStr("initial state", c.device.initial_text, kFieldW);
    InputNumStr("state min", c.device.min_text, kFieldW);
    InputNumStr("state max", c.device.max_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("A hard bound at the end of the step, on top of the window in the equation.\nEach time it acts is counted below the results.");
}

static void draw_neural_checkpoint(AppModel& model, NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Checkpoint")) return;
    ImGui::Checkbox("keep a checkpoint at the end of the run", &c.keep_checkpoint);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("State, weights, traces, delayed spikes and the time. Does not work with R or the Lyapunov exponent.");
    if (c.keep_checkpoint && c.schedule != NeuralSchedule::Spike && c.order_var >= 0) ImGui::TextDisabled("R is not computed while a checkpoint is kept.");
    if (c.checkpoint) ImGui::Text("step %lld, %.1f MB", c.checkpoint->step, (double)c.checkpoint->data.size() / (1024.0 * 1024.0));
    else ImGui::TextDisabled("No checkpoint yet.");
    const bool can = c.checkpoint != nullptr && !s.in_flight;
    if (!can) ImGui::BeginDisabled();
    if (ImGui::Button("Continue")) {
        model.run_neural(s.active_config_index, true);
    }
    if (!can) ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Runs from the checkpoint to the computing time: raise it first.\nParameters may change. The graph, h, the decimation, the transient and the recorded nodes may not.");
    InputTextStr("file", c.checkpoint_path, kFieldW + 240.0f);
    if (!c.checkpoint) ImGui::BeginDisabled();
    if (ImGui::Button("Save to file")) {
        try { neural_checkpoint_save(c.checkpoint_path, *c.checkpoint); }
        catch (const std::exception& e) { c.last_error = e.what(); }
    }
    if (!c.checkpoint) ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Load from file")) {
        try { c.checkpoint = neural_checkpoint_load(c.checkpoint_path); c.checkpoint_from_result = false; }
        catch (const std::exception& e) { c.last_error = e.what(); }
    }
}

static void draw_neural_laws(NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Coupling laws")) return;
    ImGui::TextDisabled("x_j is the source, x or x_i the receiver, K or w the arc weight.");
    int remove = -1;
    for (size_t i = 0; i < c.laws.size(); ++i) {
        NetCouplingLaw& l = c.laws[i];
        l.expr.resize(s.vars.size());
        ImGui::PushID((int)i + 2000);
        InputTextStr("law name", l.name, kFieldW);
        for (size_t k = 0; k < s.vars.size(); ++k)
            InputTextStr((s.vars[k] + "'##law").c_str(), l.expr[k], kFieldW + 120.0f);
        if (c.laws.size() > 1 && ImGui::Button("Remove law")) remove = (int)i;
        ImGui::PopID();
        ImGui::Separator();
    }
    if (remove >= 0) {
        c.laws.erase(c.laws.begin() + remove);
        for (NeuralProjection& p : c.projections) { if (p.law == remove) p.law = 0; else p.law -= p.law > remove; }
    }
    if (ImGui::Button("Add law")) {
        NetCouplingLaw l = net_default_law(s.vars);
        l.name = "law " + std::to_string(c.laws.size() + 1);
        c.laws.push_back(std::move(l));
    }
}

// Снимок системы библиотеки: дальше источник от записи не зависит.
static void neural_take_system(SystemLibrary& lib, const std::string& name, NeuralSource& src, std::string& error) {
    try {
        const SystemRecord r = lib.load(name);
        AppModel tmp(nullptr);
        tmp.from_record(r);
        if (!tmp.start_neural_analysis()) throw std::runtime_error("the system does not parse");
        neural_source_set_system(src, name, r, tmp.neural_session.sys);
        error.clear();
    } catch (const std::exception& e) {
        error = "input source: " + std::string(e.what());
    }
}

static void draw_neural_input(SystemLibrary& lib, NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Input", ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (!s.vars.empty()) {
        const std::vector<const char*> items = c_str_list(s.vars);
        if (c.input_var < 0 || c.input_var >= (int)items.size()) c.input_var = 0;
        ImGui::SetNextItemWidth(kFieldW);
        ImGui::Combo("input variable", &c.input_var, items.data(), (int)items.size());
    }
    InputNumStr("sample time", c.input_hold_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Each sample is held this long, a multiple of h. 0 = a new sample every step.");
    int remove = -1;
    for (size_t i = 0; i < c.sources.size(); ++i) {
        NeuralSource& q = c.sources[i];
        ImGui::PushID((int)i + 3000);
        const std::string title = q.name + ": " + neural_source_name(q.kind) + "###src";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            InputTextStr("name", q.name, kFieldW);
            ImGui::SetNextItemWidth(kFieldW + 100.0f);
            if (ImGui::BeginCombo("source", neural_source_name(q.kind))) {
                for (NeuralSourceKind v : kNeuralSources)
                    if (ImGui::Selectable(neural_source_name(v), v == q.kind)) q.kind = v;
                ImGui::EndCombo();
            }
            if (q.kind == NeuralSourceKind::System) {
                ImGui::SetNextItemWidth(kFieldW + 100.0f);
                if (ImGui::BeginCombo("system", q.system_name.empty() ? "(pick a system)" : q.system_name.c_str())) {
                    for (const std::string& name : library_list_cached(lib).names)
                        if (ImGui::Selectable(name.c_str(), name == q.system_name)) neural_take_system(lib, name, q, c.last_error);
                    ImGui::EndCombo();
                }
                if (!q.vars.empty()) {
                    const std::vector<const char*> items = c_str_list(q.vars);
                    if (q.variable < 0 || q.variable >= (int)items.size()) q.variable = 0;
                    ImGui::SetNextItemWidth(kFieldW);
                    ImGui::Combo("signal", &q.variable, items.data(), (int)items.size());
                }
                InputNumStr("integration step", q.step_text, kFieldW);
                InputNumStr("time scale", q.time_scale_text, kFieldW);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Source time per unit of network time.");
                InputNumStr("source transient", q.skip_text, kFieldW);
                ImGui::Checkbox("normalise to zero mean, unit deviation", &q.normalize);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Each source by its own mean and deviation. For classes of one system this hides\n"
                                      "the difference in amplitude and level: the network has to tell the dynamics apart.");
                // Снимок из библиотеки правится здесь же: классы одной системы различаются параметрами.
                if (!q.params.empty() || !q.vars.empty()) {
                    if (ImGui::TreeNode("parameters and initial conditions")) {
                        for (const std::string& n : q.params) InputNumStr((n + "##srcpar").c_str(), q.param_values[n], kFieldW);
                        for (const std::string& n : q.vars) InputNumStr((n + "(0)##srcic").c_str(), q.initial_conditions[n], kFieldW);
                        ImGui::TreePop();
                    }
                }
            }
            InputNumStr(q.kind == NeuralSourceKind::Noise ? "deviation" : "amplitude", q.amplitude_text, kFieldW);
            if (q.kind == NeuralSourceKind::Sine || q.kind == NeuralSourceKind::Square) {
                InputNumStr("frequency", q.frequency_text, kFieldW);
                InputNumStr("phase, rad", q.phase_text, kFieldW);
            }
            InputNumStr("offset", q.offset_text, kFieldW);
            if (ImGui::Button("Remove source")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) {
        c.sources.erase(c.sources.begin() + remove);
        c.links.erase(std::remove_if(c.links.begin(), c.links.end(), [&](const NeuralLink& l) { return l.source == remove; }), c.links.end());
        for (NeuralLink& l : c.links) l.source -= l.source > remove;
    }
    if (ImGui::Button("Add source")) {
        NeuralSource q;
        q.name = "input " + std::to_string(c.sources.size() + 1);
        c.sources.push_back(std::move(q));
    }
    remove = -1;
    std::vector<const char*> sources, pops;
    for (const NeuralSource& q : c.sources) sources.push_back(q.name.c_str());
    for (const NeuralPopulation& p : c.populations) pops.push_back(p.name.c_str());
    for (size_t i = 0; i < c.links.size(); ++i) {
        NeuralLink& l = c.links[i];
        ImGui::PushID((int)i + 4000);
        const char* from = l.source >= 0 && l.source < (int)sources.size() ? sources[(size_t)l.source] : "?";
        const char* to = l.population >= 0 && l.population < (int)pops.size() ? pops[(size_t)l.population] : "?";
        const std::string title = std::string(from) + " -> " + to + "###link";
        if (ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::Combo("from", &l.source, sources.data(), (int)sources.size());
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::Combo("to population", &l.population, pops.data(), (int)pops.size());
            InputNumStr("gain", l.gain_text, kFieldW);
            InputNumStr("relative gain spread", l.gain_spread_text, kFieldW);
            InputNumStr("fraction of nodes", l.fraction_text, kFieldW);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Picked at random by the seed. A node takes one channel,\nfractions of one population must add up to 1 or less.");
            if (ImGui::Button("Remove link")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) c.links.erase(c.links.begin() + remove);
    if (c.sources.empty()) ImGui::BeginDisabled();
    if (ImGui::Button("Add link")) c.links.push_back(NeuralLink{});
    if (c.sources.empty()) ImGui::EndDisabled();
}

// Классификация: попытка = образец, источник = класс.
static void draw_neural_classes(NeuralConfig& c) {
    NeuralTask& t = c.task;
    t.classes.erase(std::remove_if(t.classes.begin(), t.classes.end(), [&](int q) { return q < 0 || q >= (int)c.sources.size(); }), t.classes.end());
    if (c.sources.size() < 2) { ImGui::TextDisabled("Add two or more input sources in Input: each one can be a class."); return; }
    ImGui::TextUnformatted("classes");
    for (int q = 0; q < (int)c.sources.size(); ++q) {
        auto it = std::find(t.classes.begin(), t.classes.end(), q);
        bool on = it != t.classes.end();
        if (ImGui::Checkbox((c.sources[(size_t)q].name + "##class" + std::to_string(q)).c_str(), &on)) {
            if (on) t.classes.push_back(q); else t.classes.erase(it);
            std::sort(t.classes.begin(), t.classes.end());
        }
    }
    InputNumStr("feature nodes", t.features_text, kFieldW);
    InputNumStr("feature points per stimulus", t.bins_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The stimulus is cut into this many equal parts; a part gives the state of every feature node at its end,\n"
                          "or with spikes the spike count of the node inside it. 1 = the state at the end of the stimulus.");
    InputNumStr("train share", t.train_text, kFieldW);
    InputNumStr("validation share", t.val_text, kFieldW);
    InputNumStr("data seed", t.data_seed_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Order of the classes over the trials. The seed of the network does not change it.");
    try {
        const std::vector<int> labels = neural_class_labels(c);
        ImGui::TextDisabled("%zu trials = samples, %zu classes, every class in every part of the split.", labels.size(), t.classes.size());
    } catch (const std::exception& e) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s", e.what());
    }
    ImGui::TextDisabled("A trial is one sample: the protocol is separate trials, the rest before the stimulus is the washout.\n"
                        "During the stimulus every class source carries the signal of the trial's class: link one of them.\n"
                        "A system gives every trial a new piece of its trajectory; its parameters are in Input.\n"
                        "Train, validation and test go in time order; ridge on one-hot, lambda by validation accuracy.");
}

// Memory capacity и XOR: цель берётся из самого входа.
static void draw_neural_delayed(NeuralConfig& c) {
    NeuralTask& t = c.task;
    if (c.sources.empty()) { ImGui::TextDisabled("Add an input source in Input and link it to the network."); return; }
    if (t.source < 0 || t.source >= (int)c.sources.size()) t.source = 0;
    ImGui::SetNextItemWidth(kFieldW);
    if (ImGui::BeginCombo("input", c.sources[(size_t)t.source].name.c_str())) {
        for (int i = 0; i < (int)c.sources.size(); ++i)
            if (ImGui::Selectable(c.sources[(size_t)i].name.c_str(), i == t.source)) t.source = i;
        ImGui::EndCombo();
    }
    InputNumStr("largest delay, samples", t.max_delay_text, kFieldW);
    InputNumStr("washout, samples", t.washout_text, kFieldW);
    InputNumStr("train share", t.train_text, kFieldW);
    InputNumStr("validation share", t.val_text, kFieldW);
    InputNumStr("gap, samples", t.gap_text, kFieldW);
    InputNumStr("feature nodes", t.features_text, kFieldW);
    if (c.schedule == NeuralSchedule::Spike) InputNumStr("spike filter time", t.filter_text, kFieldW);
    const NeuralSourceKind k = c.sources[(size_t)t.source].kind;
    if (t.kind == NeuralTaskKind::Memory) {
        ImGui::TextDisabled("Target d is the input d samples back, d = 1..largest delay; r^2 of each on the test, MC is their sum.\n"
                            "The classic protocol drives the network with white noise: MC <= number of feature nodes.");
        if (k != NeuralSourceKind::Noise && k != NeuralSourceKind::Bits)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "A smooth input is predictable from its own past: MC then counts that too. Gaussian noise is the usual input.");
    }
    else {
        ImGui::TextDisabled("Target d is the XOR of the signs of the input d and d + 1 samples back; accuracy by the threshold 0.5,\n"
                            "chance 0.5. A linear model on the delayed input cannot do XOR: it is shown as the control of nonlinearity.");
        if (k != NeuralSourceKind::Bits && k != NeuralSourceKind::Noise)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Use Random bits as the input: the bit is the sign of the sample.");
    }
    ImGui::TextDisabled("One sample is the input sample time. The washout is raised to the largest delay + 1 when shorter.");
}

// Автономный прогноз: выход учится на следующем отсчёте входа, затем сам становится входом.
static void draw_neural_forecast(NeuralConfig& c) {
    NeuralTask& t = c.task;
    const std::vector<int> chans = neural_forecast_channels(c);
    if (chans.empty()) { ImGui::TextDisabled("Link an input source to the network: the readout learns and then replaces every linked input."); return; }
    std::string list;
    bool smooth = true;
    for (int q : chans) {
        list += (list.empty() ? "" : ", ") + c.sources[(size_t)q].name;
        smooth = smooth && c.sources[(size_t)q].kind != NeuralSourceKind::Noise && c.sources[(size_t)q].kind != NeuralSourceKind::Bits;
    }
    ImGui::Text("forecast of %s", list.c_str());
    if (!smooth) ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "A random input cannot be forecast: link a system or a periodic source.");
    InputNumStr("forecast length, samples", t.forecast_length_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("After t max of Protocol the network runs on with its own readout as the input.");
    InputNumStr("error threshold", t.forecast_threshold_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Error = RMS over the inputs of (forecast - truth) / deviation of the input on the training part.\n"
                          "Valid time = how long the error stays below this; 0.5 is common.");
    InputNumStr("ridge lambda", t.lambda_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("0 = chosen on the validation part by the error one sample ahead.\n"
                          "That choice is often too small for a stable closed loop: try 1e-4 .. 1e-1 by hand.");
    InputNumStr("Lyapunov exponent of the source", t.lyapunov_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Optional, in the time of the source (Lorenz 28, 10, 8/3: 0.906). Shows the valid time in Lyapunov times.");
    InputNumStr("washout, samples", t.washout_text, kFieldW);
    InputNumStr("train share", t.train_text, kFieldW);
    InputNumStr("validation share", t.val_text, kFieldW);
    InputNumStr("gap, samples", t.gap_text, kFieldW);
    InputNumStr("feature nodes", t.features_text, kFieldW);
    ImGui::TextDisabled("t max of Protocol is the training run: the true input drives the network, the readout learns the next\n"
                        "input sample (washout, train, gap, validation, gap, test one sample ahead). A second run goes on from\n"
                        "its end with the readout as the input. Compared in the same closed loop with a linear autoregression\n"
                        "(the floor: is nonlinearity needed) and NVAR, delays and their products (the strong baseline).\n"
                        "One sample is the input sample time; the transient must be 0; no sweep, no spiking network yet.");
}

static void draw_neural_saved(NeuralConfig& c, bool classifier);

static void draw_neural_readout(NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Readout")) return;
    NeuralTask& t = c.task;
    ImGui::Checkbox("use a readout on the response", &t.enabled);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Ridge on the state of the nodes, or on filtered spikes.\n"
                          "Recover or predict: targets are variables of the input system, the record is split in time order\n"
                          "(washout, train, gap, validation, gap, test), NRMSE is reported on the test.\n"
                          "Classify: every trial is a sample of a class, accuracy is reported on the test trials.");
    if (!t.enabled) return;
    int kind = (int)t.kind;
    static const char* const kinds[] = { "Recover or predict the input system", "Classify the stimulus",
                                         "Memory capacity: recall the delayed input", "XOR of two delayed bits of the input",
                                         "Autonomous forecast: the readout drives the input" };
    ImGui::SetNextItemWidth(kFieldW + 160.0f);
    if (ImGui::Combo("task", &kind, kinds, IM_ARRAYSIZE(kinds))) t.kind = (NeuralTaskKind)kind;
    if (t.kind == NeuralTaskKind::Forecast) { draw_neural_forecast(c); return; }
    ImGui::Checkbox("control: the same network with K = 0", &t.control_k0);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A second run with the same nodes, input, feature nodes and data, but every recurrent link is 0.\n"
                          "The difference to the main run is what the links add; the readout of the control is trained anew.\n"
                          "Not with a sweep: sweep over K instead.");
    if (t.kind == NeuralTaskKind::Classification) { draw_neural_classes(c); draw_neural_saved(c, true); return; }
    if (t.kind == NeuralTaskKind::Memory || t.kind == NeuralTaskKind::Xor) { draw_neural_delayed(c); return; }
    std::vector<int> systems;
    for (int i = 0; i < (int)c.sources.size(); ++i) if (c.sources[(size_t)i].kind == NeuralSourceKind::System) systems.push_back(i);
    if (systems.empty()) { ImGui::TextDisabled("Add an input source that is a system from the library."); return; }
    if (std::find(systems.begin(), systems.end(), t.source) == systems.end()) t.source = systems[0];
    ImGui::SetNextItemWidth(kFieldW);
    if (ImGui::BeginCombo("input system", c.sources[(size_t)t.source].name.c_str())) {
        for (int i : systems) if (ImGui::Selectable(c.sources[(size_t)i].name.c_str(), i == t.source)) { t.source = i; t.targets.clear(); }
        ImGui::EndCombo();
    }
    const NeuralSource& src = c.sources[(size_t)t.source];
    for (int k = 0; k < (int)src.vars.size(); ++k) {
        auto it = std::find(t.targets.begin(), t.targets.end(), k);
        bool on = it != t.targets.end();
        if (k) ImGui::SameLine();
        if (ImGui::Checkbox((src.vars[(size_t)k] + "##target").c_str(), &on)) {
            if (on) t.targets.push_back(k); else t.targets.erase(it);
            std::sort(t.targets.begin(), t.targets.end());
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("targets");
    InputNumStr("horizon, samples", t.horizon_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("0: the target at the same moment. Above 0: this many input samples ahead.");
    InputNumStr("washout, samples", t.washout_text, kFieldW);
    InputNumStr("train share", t.train_text, kFieldW);
    InputNumStr("validation share", t.val_text, kFieldW);
    InputNumStr("gap, samples", t.gap_text, kFieldW);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Dropped between the parts, so that one window of the record does not lie on both sides.");
    InputNumStr("feature nodes", t.features_text, kFieldW);
    if (c.schedule == NeuralSchedule::Spike) InputNumStr("spike filter time", t.filter_text, kFieldW);
    ImGui::TextDisabled("The input sample time sets the samples. The transient must be 0.\n"
                        "With a sweep every network trains its own readout: see Sweep in Neural Results.");
    draw_neural_saved(c, false);
}

static void draw_neural_saved(NeuralConfig& c, bool classifier) {
    NeuralReadoutSaved& saved = classifier ? c.classifier : c.readout;
    bool& apply = classifier ? c.task.use_saved_classifier : c.task.use_saved;
    ImGui::SeparatorText(classifier ? "Saved classifier" : "Saved readout");
    const bool have = !saved.empty(), fresh = !c.task_result.model.empty() && c.task_result.model.is_classifier() == classifier;
    if (!have) ImGui::TextDisabled("None yet: train and press Keep.");
    else {
        if (classifier)
            ImGui::Text("%zu classes, %zu feature nodes, %d windows; test accuracy %.3f when trained",
                        saved.names.size(), saved.nodes.size(), saved.class_bins, saved.test.empty() ? 0.0 : saved.test[0]);
        else for (size_t k = 0; k < saved.names.size(); ++k)
            ImGui::Text("%s from %d feature nodes, test NRMSE %.4f when trained", saved.names[k].c_str(), saved.M, k < saved.test.size() ? saved.test[k] : 0.0);
        ImGui::TextDisabled("graph hash %s", saved.graph_hash.c_str());
    }
    const bool disable_apply = !have && !apply;
    if (disable_apply) ImGui::BeginDisabled();
    ImGui::Checkbox(classifier ? "use the saved classifier, no training" : "use the saved readout, no training", &apply);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", classifier ? "Run repeats the same held-out trials with the saved weights and reports accuracy and the confusion matrix.\n"
                          "Use the same network, sources, features and trial settings. Turn off sweep and the K = 0 control."
                          : "Run applies the saved weights to the whole record after the washout and reports NRMSE there.\n"
                          "The graph, the feature nodes, the targets, the horizon and the sample time must match the training;\n"
                          "node parameters and the source transient (another piece of the same attractor) may change;\n"
                          "other initial conditions or parameters of the input system come from the library: edit and pick it again.\n"
                          "The input is normalised with the mean and deviation of the training.");
    if (disable_apply) ImGui::EndDisabled();
    if (!fresh) ImGui::BeginDisabled();
    if (ImGui::Button("Keep")) saved = c.task_result.model;
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", classifier ? "Keeps the classifier trained by the last run. It is written with the session.\nTrain without a sweep to keep one classifier."
                          : "Keeps the readout trained by the last run, in place of the saved one. It is written with the session.\n"
                          "A sweep trains one readout per network, so there is nothing to keep after it: run without the sweep.\n"
                          "A saved readout can be applied to a sweep over a node parameter or the input gain (one graph);\n"
                          "a sweep over K or the seed changes the graph.");
    if (!fresh) ImGui::EndDisabled();
    ImGui::SameLine();
    if (!have) ImGui::BeginDisabled();
    if (ImGui::Button("Forget")) { saved = NeuralReadoutSaved{}; apply = false; }
    if (!have) ImGui::EndDisabled();
    InputTextStr("file##readout", c.readout_path, kFieldW + 240.0f);
    if (!have) ImGui::BeginDisabled();
    if (ImGui::Button("Export")) {
        try { neural_readout_export(c.readout_path, saved); }
        catch (const std::exception& e) { c.last_error = e.what(); }
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("JSON with the weights, the feature normalisation, the feature nodes,\nthe input normalisation and the formula that applies them.");
    if (!have) ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Load##readout")) {
        // Загружают, чтобы применить: Forget перед этим снял галочку, без неё Run молча обучил бы новый выход.
        try {
            auto loaded = neural_readout_import(c.readout_path);
            if (loaded.is_classifier() != classifier) throw std::runtime_error("the file is for a different readout task");
            saved = std::move(loaded); apply = true; c.last_error.clear();
        }
        catch (const std::exception& e) { c.last_error = e.what(); }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Loads weights from Export and turns on their use for the selected task.");
}

static void draw_neural_protocol(NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Protocol", ImGuiTreeNodeFlags_DefaultOpen)) return;
    int kind = (int)c.protocol;
    static const char* const kinds[] = { "One continuous run", "Separate trials" };
    ImGui::SetNextItemWidth(kFieldW + 100.0f);
    if (ImGui::Combo("protocol", &kind, kinds, IM_ARRAYSIZE(kinds))) c.protocol = (NeuralProtocol)kind;
    if (c.protocol != NeuralProtocol::Trials) return;
    InputNumStr("trials", c.trials_text, kFieldW);
    InputNumStr("rest before the stimulus", c.rest_text, kFieldW);
    InputNumStr("stimulus", c.stimulus_text, kFieldW);
    ImGui::Checkbox("restart each trial from the initial conditions", &c.reset_state);
    if (c.reset_state) ImGui::TextDisabled("The order parameter is off: a phase across a reset has no meaning.");
    ImGui::TextDisabled("The record lasts trials x (rest + stimulus) = %s x (%s + %s) = %g time units;\n"
                        "the t max field of Integration is not used with trials.",
                        c.trials_text.c_str(), c.rest_text.c_str(), c.stimulus_text.c_str(), neural_record_time(c));
}

static void draw_neural_sweep_axis(NeuralSession& s, NeuralConfig& c, NeuralSweep& w) {
    ImGui::SetNextItemWidth(kFieldW + 60.0f);
    if (ImGui::BeginCombo("sweep over", neural_sweep_name(w.kind))) {
        for (NeuralSweepKind v : kNeuralSweeps)
            if (ImGui::Selectable(neural_sweep_name(v), v == w.kind)) w.kind = v;
        ImGui::EndCombo();
    }
    if (w.kind == NeuralSweepKind::Parameter || w.kind == NeuralSweepKind::ParameterSpread) {
        std::vector<const char*> names = { "all populations" };
        for (const NeuralPopulation& p : c.populations) names.push_back(p.name.c_str());
        int pop = w.population + 1;
        if (pop < 0 || pop >= (int)names.size()) pop = 0;
        ImGui::SetNextItemWidth(kFieldW + 60.0f);
        if (ImGui::Combo("in", &pop, names.data(), (int)names.size())) w.population = pop - 1;
        if (w.param.empty() && !s.params.empty()) w.param = s.params[0];
        ImGui::SetNextItemWidth(kFieldW);
        if (ImGui::BeginCombo("parameter", w.param.c_str())) {
            for (const std::string& n : s.params)
                if (ImGui::Selectable(n.c_str(), n == w.param)) w.param = n;
            ImGui::EndCombo();
        }
    }
    if (w.kind != NeuralSweepKind::Replica) {
        InputNumStr("from", w.lo_text, kFieldW);
        InputNumStr("to", w.hi_text, kFieldW);
        ImGui::Checkbox("log spacing", &w.log);
    }
    InputNumStr("points", w.points_text, kFieldW);
}

static void draw_neural_sweep(NeuralSession& s, NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Sweep", ImGuiTreeNodeFlags_DefaultOpen)) return;
    ImGui::Checkbox("sweep this experiment", &c.sweep.enabled);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Every value runs as a separate network in one launch.\n"
                          "A network that diverges stops alone, the others go on.");
    if (!c.sweep.enabled) return;
    draw_neural_sweep_axis(s, c, c.sweep);
    ImGui::Checkbox("second axis", &c.sweep2.enabled);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Every pair of values is a network: the number of networks is the product of the points.\n"
                          "Replicas on this axis give an error bar for each value of the first one.");
    if (c.sweep2.enabled) {
        ImGui::PushID("sweep2");
        draw_neural_sweep_axis(s, c, c.sweep2);
        ImGui::PopID();
    }
    draw_csv_output_block("CSV output##neural_csv", "Save to file", c.csv_save_enabled, "##neural_csv_path", c.csv_output_path,
                          "One row per network and population. Written after every run.");
}

static void draw_neural_lyapunov(NeuralConfig& c) {
    if (!ImGui::CollapsingHeader("Lyapunov exponent")) return;
    ImGui::Checkbox("largest exponent of the whole network", &c.lyap_on);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A perturbed copy of every network runs next to it, with the same parameters, input and links.\n"
                          "Memory and time double from the moment the copy starts.");
    if (!c.lyap_on) return;
    if (c.protocol == NeuralProtocol::Trials && c.reset_state) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Does not work with a restart of each trial.");
    }
    InputNumStr("perturbation per coordinate", c.lyap_eps_text, kFieldW);
    InputNumStr("renormalisation interval", c.lyap_interval_text, kFieldW);
    InputNumStr("direction settling time", c.lyap_settle_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Runs before the record, inside the transient: renormalised, not summed.\n"
                          "Close exponents turn the direction slowly: slow modes of many nodes need thousands.");
}

void draw_neural_controls(AppModel& model, SystemLibrary& lib) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No system loaded. Pick a system in the list above.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) s.active_config_index = 0;

    {
        RunAllGroup g;
        g.pick_id = "pneural_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.enqueue_neural(i); };
        draw_run_and_run_all("##run_all_neural", s.in_flight, false, s.in_flight, 160.0f,
                             [&model, &s]() {
                                 model.run_neural(s.active_config_index);
                             },
                             {},
                             { g },
                             [&model]() { model.start_next_in_neural_queue(); },
                             model.neural_queue().size());
        ImGui::SameLine();
        if (!s.in_flight) ImGui::BeginDisabled();
        if (ImGui::Button("Stop", ImVec2(90, 0))) { s.request_cancel(); model.clear_neural_queue(); }
        if (!s.in_flight) ImGui::EndDisabled();
    }
    ImGui::Separator();

    const TabBarResult tabs = draw_config_tab_bar(
        "##neural_tabs", "neural_tab_", (int)s.configs.size(), s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; }, {}, [&s]() { s.add_config(); });
    if (tabs.active >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_neural_config(tabs.to_remove);
    if (s.active_config_index >= (int)s.configs.size()) s.active_config_index = (int)s.configs.size() - 1;
    NeuralConfig& c = s.configs[(size_t)s.active_config_index];

    draw_label_rename("Label##neural", c.label);
    // Ошибка сверху: внизу под свёрнутыми разделами её не видно, а старые числа выглядят как результат.
    if (!c.last_error.empty()) draw_error_box("##neural_err", c.last_error);
    if (c.hash_mismatch && ImGui::Button("Rebuild as a new experiment")) {
        c.graph_hash.clear(); c.graph_recipe.clear();
        c.hash_mismatch = false; c.last_error.clear();
    }

    ImGui::SetNextItemWidth(kFieldW + 100.0f);
    if (ImGui::BeginCombo("coupling in time", neural_schedule_name(c.schedule))) {
        for (NeuralSchedule v : kNeuralSchedules)
            if (ImGui::Selectable(neural_schedule_name(v), v == c.schedule)) c.schedule = v;
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Stage: coupling in every RK4 stage, order 4, four exchanges per step.\n"
                          "Heun: order 1 on trajectories, synchronisation threshold nearly exact, two exchanges.\n"
                          "Post: order 1, threshold shifted by about -h K^2, one exchange.\n"
                          "Heun and Post are checked on diffusive coupling of Hindmarsh-Rose pairs only.\n"
                          "Spikes: nodes interact only through spike events with a delay.");
    const bool spiking = c.schedule == NeuralSchedule::Spike;
    if (spiking)
        ImGui::TextDisabled("Node step: RK4, synapses by the exact exponential.");
    else if (c.schedule != NeuralSchedule::Stage)
        draw_scheme_combo("node scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                          nullptr, s.sys.is_map, &s.wrapper_schemes);
    else
        ImGui::TextDisabled("Node step: RK4 (the only scheme with coupling in stages so far).");

    IntegrationFields f;
    f.h = &c.h_text; f.symmetry_s = &c.symmetry_s; f.cd_fp_iters = &c.cd_fp_iters; f.newton = &c.newton;
    f.t_max = &c.t_max_text; f.transient = &c.transient_text; f.pre_scaller = &c.pre_scaller_text; f.max_value = &c.max_value_text;
    draw_integration_block("Integration", c.schedule == NeuralSchedule::Stage || spiking ? std::string("RK4") : c.scheme,
                           s.custom_schemes, f, nullptr, s.sys.is_map);

    if (ImGui::CollapsingHeader("Network", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("coupling K", c.coupling_text, kFieldW);
        InputNumStr("seed", c.seed_text, kFieldW);
        InputNumStr("observed nodes per population", c.observed_per_pop_text, kFieldW);
        if (!spiking) {
            std::vector<const char*> names = { "off" };
            for (const std::string& v : s.vars) names.push_back(v.c_str());
            int order = c.order_var + 1;
            if (order < 0 || order >= (int)names.size()) order = 0;
            ImGui::SetNextItemWidth(kFieldW);
            if (ImGui::Combo("order parameter variable", &order, names.data(), (int)names.size())) c.order_var = order - 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("R(t) per population. The phase of a node grows by 2 pi between upward threshold\n"
                                  "crossings, found on the output grid: keep the output step far below the period.\n"
                                  "There is no R before the first and after the last crossings; a node gone silent\n"
                                  "lowers R. R is per population and ignores amplitudes: see the spread.");
            if (c.order_var >= 0) InputNumStr("order parameter threshold", c.order_threshold_text, kFieldW);
        }
        for (size_t k = 0; k < s.vars.size(); ++k) {
            auto it = std::find(c.obs_vars_off.begin(), c.obs_vars_off.end(), (int)k);
            bool on = it == c.obs_vars_off.end();
            if (k) ImGui::SameLine();
            if (ImGui::Checkbox((s.vars[k] + "##rec").c_str(), &on)) {
                if (on) c.obs_vars_off.erase(it); else c.obs_vars_off.push_back((int)k);
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("recorded variables");
        InputNumStr("output limit, MB", c.output_limit_text, kFieldW);
    }
    if (spiking) draw_neural_spiking(s, c);
    draw_neural_input(lib, s, c);
    draw_neural_protocol(c);
    draw_neural_readout(c);
    draw_neural_sweep(s, c);
    if (!spiking) draw_neural_lyapunov(c);
    draw_neural_populations(s, c);
    draw_neural_projections(c, s.vars);
    if (!spiking) draw_neural_laws(s, c);
    if (c.schedule == NeuralSchedule::Stage) draw_neural_device(c);
    draw_neural_checkpoint(model, s, c);

    if (c.arcs > 0 || c.last_run_ok) {
        ImGui::Separator();
        if (!c.last_run_ok && c.result.ok)
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "The last run failed (see the error at the top): the numbers below are from the run before.");
        ImGui::Text("%d nodes, %lld arcs, in-degree %d..%d", c.result.n_nodes, c.arcs, c.degree_min, c.degree_max);
        ImGui::Text("graph %.2f s, run %.2f s, %.0f MB on the GPU", c.build_seconds, c.result.wall_ms / 1000.0,
                    (double)c.result.device_bytes / (1024.0 * 1024.0));
        ImGui::TextDisabled("graph hash %s", c.graph_hash.c_str());
        if (c.sweep_values.empty())
            for (size_t p = 0; p < c.pop_rate.size() && p < c.result_context.populations.size(); ++p)
                ImGui::Text("%s: %.4g spikes per node per unit of time", c.result_context.populations[p].c_str(), c.pop_rate[p]);
        const NeuralTaskResult& q = c.task_result;
        const size_t T = q.classify || q.by_delay || q.forecast ? 0 : q.names.size();
        if (q.forecast && q.points > 0 && q.dt > 0.0) {
            const auto valid_text = [&q](double v) {
                char b[160];
                const long long n = std::llround(v / q.dt);
                if (n >= q.points) std::snprintf(b, sizeof b, "at least %.4g (all %lld samples)", v, n);
                else std::snprintf(b, sizeof b, "%.4g (%lld samples)", v, n);
                std::string s = b;
                if (q.lyapunov > 0.0) { std::snprintf(b, sizeof b, ", %.2f Lyapunov times", v * q.time_scale * q.lyapunov); s += b; }
                return s;
            };
            ImGui::Text("autonomous forecast: valid time %s", valid_text(q.valid_time).c_str());
            if (q.ar_order > 0)
                ImGui::TextDisabled("linear autoregression of order %d in the same closed loop: %s", q.ar_order, valid_text(q.ar_valid_time).c_str());
            if (q.nvar_order > 0)
                ImGui::TextDisabled("NVAR, %d delays of every input and their products, same closed loop: %s", q.nvar_order, valid_text(q.nvar_valid_time).c_str());
            for (size_t k = 0; k < q.names.size() && k < q.test.size(); ++k)
                ImGui::TextDisabled("%s one sample ahead under the true input: test NRMSE %.3g, validation %.3g, lambda %g",
                                    q.names[k].c_str(), q.test[k], q.val[k], q.lambda[k]);
            ImGui::TextDisabled("Error threshold %.3g. See Forecast in Neural Results.", q.threshold);
        }
        if (q.by_delay && !q.test.empty()) {
            const size_t KO = q.names.size();
            const bool memory = !q.names.empty() && q.names[0] == "MC";
            const int D = (int)KO - (memory ? 1 : 0);
            // У свипа значения по сетям в окнах Neural Sweep и Neural Map; здесь сеть 0 и разброс MC.
            if (memory) {
                double lo = std::numeric_limits<double>::infinity(), hi = -lo;
                for (int i = 0; i < q.instances; ++i) if (std::isfinite(q.test[(size_t)i * KO])) { lo = (std::min)(lo, q.test[(size_t)i * KO]); hi = (std::max)(hi, q.test[(size_t)i * KO]); }
                if (q.instances > 1) ImGui::Text("memory capacity MC %.2f .. %.2f over %d networks, delays 1..%d", lo, hi, q.instances, D);
                else ImGui::Text("memory capacity MC = %.2f over delays 1..%d, %d feature nodes", q.test[0], D, q.features);
                if (q.control_test.size() == KO) ImGui::TextDisabled("the same network with K = 0: MC = %.2f", q.control_test[0]);
            }
            else {
                std::string line = "XOR accuracy by delay:";
                char b[48];
                for (int d = 0; d < (std::min)(D, 6); ++d) { std::snprintf(b, sizeof b, " %d: %.3f", d + 1, q.test[(size_t)d]); line += b; }
                ImGui::TextUnformatted(line.c_str());
                if (q.control_test.size() == KO) {
                    line = "K = 0:";
                    for (int d = 0; d < (std::min)(D, 6); ++d) { std::snprintf(b, sizeof b, " %d: %.3f", d + 1, q.control_test[(size_t)d]); line += b; }
                    ImGui::TextDisabled("%s", line.c_str());
                }
                if (!q.delay_test.empty()) {
                    line = "linear on " + std::to_string(q.delay_length) + " delayed inputs:";
                    for (int d = 0; d < (std::min)(D, 6); ++d) { std::snprintf(b, sizeof b, " %d: %.3f", d + 1, q.delay_test[(size_t)d]); line += b; }
                    ImGui::TextDisabled("%s", line.c_str());
                }
                ImGui::TextDisabled("Chance 0.5. See XOR in Neural Results.");
            }
        }
        if (q.classify && q.control_test.size() == 1) ImGui::TextDisabled("the same network with K = 0: accuracy %.3f", q.control_test[0]);
        if (T > 0 && q.control_test.size() == T)
            for (size_t k = 0; k < T; ++k) ImGui::TextDisabled("%s with K = 0: test NRMSE %.4f", q.names[k].c_str(), q.control_test[k]);
        if (q.classify && !q.test.empty()) {
            // Лучшая сеть у точности наибольшая.
            int best = -1, finite = 0;
            double lo = std::numeric_limits<double>::infinity();
            for (int i = 0; i < q.instances; ++i) {
                const double v = q.test[(size_t)i];
                if (!std::isfinite(v)) continue;
                ++finite;
                lo = (std::min)(lo, v);
                if (best < 0 || v > q.test[(size_t)best]) best = i;
            }
            if (best < 0) ImGui::Text("classification: no network gave a readout");
            else if (q.instances > 1)
                ImGui::Text("classification: test accuracy %.3f .. %.3f over %d of %d networks, best is network %d",
                            lo, q.test[(size_t)best], finite, q.instances, best);
            else if (q.applied) ImGui::Text("saved classifier: test accuracy %.3f, %d test trials", q.test[0], q.points);
            else ImGui::Text("classification: test accuracy %.3f, validation %.3f, lambda %g, %d test trials",
                             q.test[0], q.val[0], q.lambda[0], q.points);
            if (q.delay_test.empty()) ImGui::TextDisabled("chance %.3f", q.chance);
            else ImGui::TextDisabled("chance %.3f, ridge on %d stimulus samples %.3f", q.chance, q.delay_length, q.delay_test[0]);
        }
        for (size_t k = 0; k < T; ++k) {
            if (q.instances > 1) {
                // Сводка свипа: лучшая сеть и где она, остальное в окнах Neural Sweep и Neural Map.
                int best = -1, finite = 0;
                double hi = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < q.instances; ++i) {
                    const double v = q.test[(size_t)i * T + k];
                    if (!std::isfinite(v)) continue;
                    ++finite;
                    hi = (std::max)(hi, v);
                    if (best < 0 || v < q.test[(size_t)best * T + k]) best = i;
                }
                if (best < 0) ImGui::Text("%s: no network gave a readout", q.names[k].c_str());
                else ImGui::Text("%s: %s NRMSE %.4f .. %.4f over %d of %d networks, best is network %d", q.names[k].c_str(),
                                 q.applied ? "saved readout," : "test", q.test[(size_t)best * T + k], hi, finite, q.instances, best);
            }
            else if (q.applied) ImGui::Text("%s: NRMSE %.4f of the saved readout after the washout", q.names[k].c_str(), q.test[k]);
            else ImGui::Text("%s: test NRMSE %.4f, validation %.4f, lambda %g", q.names[k].c_str(), q.test[k], q.val[k], q.lambda[k]);
            if (!q.applied && k < q.delay_test.size()) { ImGui::SameLine(); ImGui::TextDisabled("delays of the input (%d): %.4f", q.delay_length, q.delay_test[k]); }
        }
        if (!q.note.empty()) ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Readout: %s", q.note.c_str());
        if (c.result.device_clips > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "The device state bound acted %lld times.", c.result.device_clips);
        if (c.result.spikes_dropped > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "The spike log is cut: %lld events omitted. Total counts are still exact.", c.result.spikes_dropped);
        const int nInst = (int)c.result.instance_stop.size();
        if (c.result.lyapunov.size() == 1)
            ImGui::Text("largest Lyapunov exponent %.6g, %d blocks", c.result.lyapunov[0], c.result.lyap_blocks);
        else if (!c.result.lyapunov.empty()) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            for (double v : c.result.lyapunov) if (std::isfinite(v)) { lo = (std::min)(lo, v); hi = (std::max)(hi, v); }
            if (hi >= lo) ImGui::Text("largest Lyapunov exponent %.4g .. %.4g over %d networks", lo, hi, nInst);
            ImGui::TextDisabled("See the Lyapunov Exponent window.");
        }
        if (c.result.diverged && nInst > 1) {
            int stopped = 0;
            for (long long v : c.result.instance_stop) stopped += v >= 0;
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%d of %d networks diverged, the others ran on", stopped, nInst);
        }
        else if (c.result.diverged)
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%d node(s) diverged, the run stopped at step %lld", c.result.n_diverged, c.result.steps_done);
    }
    if (!c.model_text.empty() && !s.sys.rhs.empty() && c.model_text != neural_model_text(s.sys))
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "The system changed since this experiment was last run.");
}

// Номер показываемого экземпляра и что с ним стало.
static void neural_instance_picker(const NeuralConfig& c, NeuralViewState& vs, const char* id) {
    const int nInst = (int)c.instance_start.size() - 1;
    vs.instance = std::clamp(vs.instance, 0, (std::max)(0, nInst - 1));
    if (nInst <= 1) return;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140.0f);
    ImGui::SliderInt(id, &vs.instance, 0, nInst - 1);
    const int n2 = (std::max)(1, (int)c.sweep2_values.size());
    if (vs.instance / n2 < (int)c.sweep_values.size()) {
        ImGui::SameLine();
        if (c.sweep2_values.empty()) ImGui::Text("value %g", c.sweep_values[(size_t)vs.instance]);
        else ImGui::Text("values %g, %g", c.sweep_values[(size_t)(vs.instance / n2)], c.sweep2_values[(size_t)(vs.instance % n2)]);
    }
    if (vs.instance < (int)c.result.instance_stop.size() && c.result.instance_stop[(size_t)vs.instance] >= 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "diverged at t = %g",
                           (double)c.result.instance_stop[(size_t)vs.instance] * c.result.h);
    }
}

static void choose_result_metric(const char* id, int& selected, const std::vector<std::pair<int, const char*>>& choices) {
    if (choices.empty()) return;
    auto current = std::find_if(choices.begin(), choices.end(), [&](const auto& p) { return p.first == selected; });
    if (current == choices.end()) { current = choices.begin(); selected = current->first; }
    if (choices.size() == 1) { ImGui::TextUnformatted(current->second); return; }
    ImGui::SetNextItemWidth(175.0f);
    if (ImGui::BeginCombo(id, current->second)) {
        for (const auto& p : choices) if (ImGui::Selectable(p.second, selected == p.first)) selected = p.first;
        ImGui::EndCombo();
    }
}

static std::vector<std::pair<int, const char*>> sweep_metrics(const NeuralConfig& c) {
    std::vector<std::pair<int, const char*>> choices;
    if (!c.result.spike_count.empty()) {
        if (!c.pop_rate.empty()) choices.emplace_back(0, "Spike rate");
        if (!c.task_result.test.empty()) choices.emplace_back(1, "Readout quality");
    } else {
        if (!c.sweep_order.empty()) choices.emplace_back(0, "Order parameter R");
        if (!c.sweep_spread.empty()) choices.emplace_back(1, "Spread across nodes");
        if (!c.sweep_lyap.empty()) choices.emplace_back(2, "Lyapunov exponent");
        if (!c.task_result.test.empty()) choices.emplace_back(3, "Readout quality");
    }
    return choices;
}

// Окна вкладки: кривые от времени для выбранного экземпляра, сводка свипа от значения.
static void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, int what,
                                      const std::vector<PlotXBand>* trial_bands = nullptr) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty() || s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) {
        ImGui::TextDisabled("No system loaded.");
        return;
    }
    NeuralConfig& c = s.configs[(size_t)s.active_config_index];
    const NeuralResult& r = c.result;
    const auto& ctx = c.result_context;
    NeuralViewState& vs = model.neural_view;
    // Сессия другой системы начинает data_generation заново: без номера сессии кэши ниже отдают данные прошлой.
    const int epoch = model.session_epoch[(int)AppModel::AppMode::Neural];
    const int nVars = (int)r.observed_vars.size(), K = c.pops_per_instance;
    const int nInst = (int)c.instance_start.size() - 1, nPops = (std::max)(0, nInst) * K;
    if (!r.observed_vars.empty() && std::find(r.observed_vars.begin(), r.observed_vars.end(), vs.series_var) == r.observed_vars.end())
        vs.series_var = r.observed_vars.front();

    if (what == kNeuralInput) {
        const int ch = (int)ctx.sources.size();
        if (ch == 0 || c.input_view.empty()) { ImGui::TextDisabled("No input yet: add a source and a link, then press Run."); return; }
    }
    const bool spikes = !r.spike_count.empty();
    const bool sync = what == kNeuralSync, sweep = what == kNeuralSweep, input = what == kNeuralInput, lyapWindow = what == kNeuralLyap, rate = what == kNeuralRate, raster = what == kNeuralRaster, weights = what == kNeuralWeights;
    const bool forecastError = what == kNeuralForecastError, phase = what == kNeuralForecastPhase;
    const bool readout = what == kNeuralReadout || forecastError || phase;
    if ((rate || raster) && r.spike_count.empty()) { ImGui::TextDisabled("Pick the Spikes schedule and press Run."); return; }
    if (rate) {
        InputNumStr("bin (0 = auto)", vs.rate_bin_text, kFieldW);
    }
    if (sweep) choose_result_metric("##neural_sweep_metric", vs.sweep_kind, sweep_metrics(c));
    if (sync) {
        std::vector<std::pair<int, const char*>> choices;
        if (!r.order.empty()) choices.emplace_back(0, "Order parameter R");
        if (!r.spread.empty()) choices.emplace_back(1, "Spread across nodes");
        choose_result_metric("##neural_sync_metric", vs.sync_kind, choices);
    }
    const NeuralTaskResult& task = c.task_result;
    // Номер NRMSE в окне свипа: у импульсной сети список короче.
    const bool readoutSweep = sweep && vs.sweep_kind == (spikes ? 1 : 3);
    const int taskT = (int)task.names.size();
    if (readoutSweep && task.classify) {
        ImGui::SameLine();
        ImGui::TextDisabled("chance %.3f", task.chance);
        if (!task.delay_test.empty()) { ImGui::SameLine(); ImGui::TextDisabled(", ridge on the stimulus samples %.3f", task.delay_test[0]); }
    }
    else if (readoutSweep && taskT > 0) {
        const std::vector<const char*> items = c_str_list(task.names);
        if (vs.readout_target >= taskT) vs.readout_target = 0;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("target##readout_sweep", &vs.readout_target, items.data(), taskT);
        if ((size_t)vs.readout_target < task.delay_test.size()) {
            ImGui::SameLine();
            ImGui::TextDisabled("delays of the input: %.4f", task.delay_test[(size_t)vs.readout_target]);
        }
    }
    const bool lyap = lyapWindow || (sweep && !spikes && vs.sweep_kind == 2);
    if (lyapWindow) {
        if (r.lyapunov.empty()) {
            ImGui::TextDisabled("Turn on the Lyapunov exponent in Neural Controls and press Run.");
            return;
        }
        ImGui::Text("lambda = %.6g", r.lyapunov[(size_t)(std::min)(vs.instance, (int)r.lyapunov.size() - 1)]);
        ImGui::SameLine();
        ImGui::TextDisabled("%d blocks of %g", r.lyap_blocks, r.lyap_dt);
    }
    // Показатель от значения свипа рисуется как сводка свипа.
    const bool byValue = sweep;
    const bool spread = (sync && vs.sync_kind == 1) || (sweep && !spikes && vs.sweep_kind == 1);
    const int taskInst = (std::max)(0, (std::min)(vs.instance, task.instances - 1));
    if (readout && task.classify) {
        // Матрица ошибок теста: строка = истинный класс, столбец = ответ.
        const int C = (int)task.class_names.size();
        const size_t at = (size_t)taskInst * C * C;
        if (C < 1 || task.confusion.size() < at + (size_t)C * C || !std::isfinite(task.confusion[at])) {
            ImGui::TextDisabled(task.note.empty() ? "This network gave no readout." : task.note.c_str());
            return;
        }
        ImGui::Text("test accuracy %.3f on %d trials, chance %.3f", task.test[(size_t)taskInst], task.points, task.chance);
        if (!task.delay_test.empty()) { ImGui::SameLine(); ImGui::TextDisabled("ridge on %d stimulus samples %.3f", task.delay_length, task.delay_test[0]); }
        ImGui::TextDisabled("Rows: the true class, columns: the answer of the readout.");
        if (ImGui::BeginTable("##neural_confusion", C + 2, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("true \\ answer");
            for (int k = 0; k < C; ++k) ImGui::TableSetupColumn(task.class_names[(size_t)k].c_str());
            ImGui::TableSetupColumn("right");
            ImGui::TableHeadersRow();
            for (int i = 0; i < C; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(task.class_names[(size_t)i].c_str());
                double row = 0.0;
                for (int k = 0; k < C; ++k) row += task.confusion[at + (size_t)i * C + k];
                for (int k = 0; k < C; ++k) {
                    ImGui::TableNextColumn();
                    const double v = task.confusion[at + (size_t)i * C + k];
                    if (i == k) ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "%.0f", v);
                    else if (v > 0.0) ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%.0f", v);
                    else ImGui::TextDisabled("0");
                }
                ImGui::TableNextColumn();
                ImGui::Text("%.0f / %.0f", task.confusion[at + (size_t)i * C + i], row);
            }
            ImGui::EndTable();
        }
        return;
    }
    // Memory и XOR: кривая значения от задержки d, сеть, та же сеть при K = 0 и линейная модель по задержкам.
    const bool delayMemory = readout && task.by_delay && !task.names.empty() && task.names[0] == "MC";
    const int delayFirst = delayMemory ? 1 : 0, delayD = (int)task.names.size() - delayFirst;
    const bool delayCtl = readout && task.by_delay && task.control_test.size() == task.names.size() && task.instances == 1;
    const bool delayLin = readout && task.by_delay && (int)task.delay_test.size() == delayD;
    if (readout && task.by_delay && !task.test.empty()) {
        if (delayMemory) ImGui::Text("MC = %.2f over delays 1..%d", task.test[(size_t)taskInst * task.names.size()], delayD);
        else ImGui::TextUnformatted("XOR of the input signs d and d + 1 samples back, chance 0.5");
        ImGui::SameLine();
        ImGui::TextDisabled("%d feature nodes, test part", task.features);
    }
    // Прогноз: настоящий вход, сеть, авторегрессия и NVAR по входам, последний пункт = их ошибки с порогом.
    const bool forecastAr = readout && task.forecast && task.ar_order > 0 && task.ar_predicted.size() == task.truth.size()
                            && task.ar_error.size() == (size_t)task.points;
    const bool forecastNvar = readout && task.forecast && task.nvar_order > 0 && task.nvar_predicted.size() == task.truth.size()
                              && task.nvar_error.size() == (size_t)task.points;
    enum { kFcTrue, kFcNet, kFcAr, kFcNvar, kFcThreshold };
    std::vector<int> forecastKinds;
    if (readout && !forecastError && !phase && !task.names.empty() && !task.by_delay) {
        std::vector<std::string> labels = task.names;
        const std::vector<const char*> items = c_str_list(labels);
        if (vs.readout_target >= (int)items.size()) vs.readout_target = 0;
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("target##readout", &vs.readout_target, items.data(), (int)items.size());
        ImGui::SameLine();
        if (task.forecast)
            ImGui::TextDisabled("autonomous from t = %.6g, valid time %.4g, error threshold %.3g, %d feature nodes",
                                task.t0, task.valid_time, task.threshold, task.features);
        else
            ImGui::TextDisabled(task.applied ? "saved readout, record after the washout, NRMSE %.4f, %d feature nodes"
                                             : "test part only, NRMSE %.4f, %d feature nodes",
                                task.test[(size_t)taskInst * items.size() + (size_t)vs.readout_target], task.features);
    }
    if (!sweep && !input && !lyap && !rate && !raster && !weights && !readout && !r.observed_vars.empty() && (!sync || spread)) {
        std::vector<std::pair<int, const char*>> choices;
        for (int v : r.observed_vars) if (v >= 0 && v < (int)ctx.variables.size()) choices.emplace_back(v, ctx.variables[(size_t)v].c_str());
        if (sync) ImGui::SameLine();
        choose_result_metric("##neural_variable", vs.series_var, choices);
    }
    const int recorded = (int)(std::find(r.observed_vars.begin(), r.observed_vars.end(), vs.series_var) - r.observed_vars.begin());

    // Точка p кривой k.
    std::function<double(int)> xof = [&](int p) { return r.t0 + r.dt * (double)p; };
    std::function<double(int, int)> series_xof;
    std::function<double(int, int)> yof;
    int curves = 0, points = r.n_points;
    std::vector<int> nodes;
    std::string yname = vs.series_var >= 0 && vs.series_var < (int)ctx.variables.size() ? ctx.variables[(size_t)vs.series_var] : "state";
    const std::string none = "No data yet - press Run on the Neural tab.";
    std::string empty;
    const int p0 = vs.instance * K;
    double max_rate = 0.0;
    const double typed_bin = rate ? parse_num(vs.rate_bin_text, 0.0) : 0.0;
    const double rate_bin = std::isfinite(typed_bin) ? typed_bin : 0.0;
    if (input) {
        const int ch = (int)ctx.sources.size();
        curves = ch; points = (int)(c.input_view.size() / (size_t)ch);
        xof = [&c](int p) { return c.input_view_t0 + c.input_view_dt * (double)p; };
        yof = [&c, ch](int k, int p) { return c.input_view[(size_t)p * ch + k]; };
        yname = "input";
    }
    else if (rate) {
        static NeuralBinnedRates table;
        static std::tuple<int, int, int, double> cached;
        static double cached_max = 0.0;
        const auto key = std::make_tuple(epoch, c.data_generation, vs.instance, rate_bin);
        if (cached != key) {
            cached = key;
            table = neural_population_rates(c, vs.instance, rate_bin);
            cached_max = table.values.empty() ? 0.0 : *std::max_element(table.values.begin(), table.values.end());
        }
        max_rate = cached_max;
        points = table.points;
        curves = K;
        if (points < 1) empty = none;
        xof = [](int p) {
            const double begin = table.t0 + (double)p * table.bin_width;
            return 0.5 * (begin + (std::min)(table.t_end, begin + table.bin_width));
        };
        yof = [K](int k, int p) { return table.values[(size_t)p * K + k]; };
        yname = "spikes / neuron / time";
        if (table.bin_width > 0) { ImGui::SameLine(); ImGui::TextDisabled("width %.4g", table.bin_width); }
    }
    else if (raster) {
        // Точка на спайк, серия на популяцию. Сверх лимита показывается каждый stride-й узел: его спайки все, строки не рвутся.
        static std::vector<int> shown;
        static std::tuple<int, int, int> shownKey;
        static int stride = 1;
        const auto key = std::make_tuple(epoch, c.data_generation, vs.instance);
        const int i0 = c.instance_start[(size_t)vs.instance], i1 = c.instance_start[(size_t)vs.instance + 1];
        if (shownKey != key) {
            shownKey = key;
            size_t inside = 0;
            for (int node : r.spike_node) inside += node >= i0 && node < i1;
            stride = (int)((inside + kNeuralRasterPoints - 1) / kNeuralRasterPoints);
            if (stride < 1) stride = 1;
            shown.clear();
            for (size_t e = 0; e < r.spike_step.size(); ++e)
                if (r.spike_node[e] >= i0 && r.spike_node[e] < i1 && (r.spike_node[e] - i0) % stride == 0) shown.push_back((int)e);
        }
        curves = K; points = (int)shown.size();
        if (K < 1) empty = none;
        xof = [&r](int p) { return (double)(r.spike_step[(size_t)shown[(size_t)p]] + 1) * r.h; };
        yof = [&r, &c, p0, i0](int k, int p) {
            const int node = r.spike_node[(size_t)shown[(size_t)p]];
            return node >= c.pop_start[(size_t)p0 + k] && node < c.pop_start[(size_t)p0 + k + 1] ? (double)(node - i0) : std::numeric_limits<double>::quiet_NaN();
        };
        yname = "node";
        ImGui::TextDisabled(stride > 1 ? "%zu displayed spikes (every %d-th node)" : "%zu spikes", shown.size(), stride);
    }
    else if (readout) {
        if (task.names.empty()) empty = "Turn on the readout in Neural Controls and press Run.";
        else if (task.by_delay) {
            const size_t at = (size_t)taskInst * task.names.size() + (size_t)delayFirst;
            const int first = delayFirst;
            curves = 1 + (delayCtl ? 1 : 0) + (delayLin ? 1 : 0); points = delayD;
            xof = [](int p) { return (double)(p + 1); };
            yof = [&task, at, first, delayCtl](int k, int p) {
                if (k == 0) return task.test[at + (size_t)p];
                if (k == 1 && delayCtl) return task.control_test[(size_t)(first + p)];
                return task.delay_test[(size_t)p];
            };
            yname = delayMemory ? "r^2" : "accuracy";
        }
        else if (phase) {
            const int columns = (int)task.names.size();
            if (columns < 2 || task.truth.size() != (size_t)columns * task.points || task.predicted.size() != task.truth.size()) empty = none;
            else {
                curves = 2; points = task.points;
                series_xof = [&task, columns](int k, int p) { return (k == 0 ? task.truth : task.predicted)[(size_t)p * columns]; };
                yof = [&task, columns](int k, int p) { return (k == 0 ? task.truth : task.predicted)[(size_t)p * columns + 1]; };
                yname = task.names[1];
            }
        }
        else if (task.forecast) {
            const int K2 = (int)task.names.size(), v = (std::min)(vs.readout_target, K2 - 1);
            points = task.points;
            xof = [&task](int p) { return task.t0 + task.dt * (double)p; };
            forecastKinds = { forecastError ? (int)kFcNet : (int)kFcTrue };
            if (!forecastError) forecastKinds.push_back(kFcNet);
            if (forecastAr) forecastKinds.push_back(kFcAr);
            if (forecastNvar) forecastKinds.push_back(kFcNvar);
            if (forecastError) forecastKinds.push_back(kFcThreshold);
            curves = (int)forecastKinds.size();
            if (task.error.size() != (size_t)task.points) empty = "The forecast run gave no forecast.";
            else if (forecastError) {
                yof = [&task, kinds = forecastKinds](int k, int p) {
                    switch (kinds[(size_t)k]) {
                    case kFcNet:  return task.error[(size_t)p];
                    case kFcAr:   return task.ar_error[(size_t)p];
                    case kFcNvar: return task.nvar_error[(size_t)p];
                    default:      return task.threshold;
                    }
                };
                yname = "error";
            }
            else {
                yof = [&task, K2, v, kinds = forecastKinds](int k, int p) {
                    const size_t at = (size_t)p * K2 + (size_t)v;
                    switch (kinds[(size_t)k]) {
                    case kFcTrue: return task.truth[at];
                    case kFcNet:  return task.predicted[at];
                    case kFcAr:   return task.ar_predicted[at];
                    default:      return task.nvar_predicted[at];
                    }
                };
                yname = task.names[(size_t)v];
            }
        }
        else {
            const int K2 = (int)task.names.size(), v = vs.readout_target;
            const size_t at = (size_t)taskInst * task.points * K2;
            curves = 2; points = task.points;
            xof = [&task](int p) { return task.t0 + task.dt * (double)p; };
            yof = [&task, K2, v, at](int k, int p) { return k == 0 ? task.truth[(size_t)p * K2 + v] : task.predicted[at + (size_t)p * K2 + v]; };
            yname = task.names[(size_t)v];
        }
    }
    else if (weights) {
        const std::vector<double>& d = !r.weight_trace.empty() ? r.weight_trace : r.device_trace;
        const int W = c.watched_per_instance;
        if (d.empty() || W < 1 || nInst < 1 || d.size() != (size_t)r.n_points * nInst * W) empty = "Give a projection a plasticity rule or a device, then press Run.";
        else {
            curves = W;
            const int at = vs.instance * W, row = nInst * W;
            yof = [&d, at, row](int k, int p) { return d[(size_t)p * row + at + k]; };
            yname = !r.weight_trace.empty() ? std::string("weight") : "device state " + ctx.device_state;
            if (!r.weight_trace.empty()) ImGui::TextDisabled("Weights of %d plastic arcs picked evenly, source -> receiver.", W);
            else ImGui::TextDisabled("State of the device between two nodes. In the presets 0 = Ron (strong link), 1 = Roff (weak link).");
            if (c.weight_summary.size() == (size_t)nInst * 3) {
                const double* q = c.weight_summary.data() + (size_t)vs.instance * 3;
                ImGui::TextDisabled("all plastic arcs at the end: min %.5g, mean %.5g, max %.5g", q[0], q[1], q[2]);
            }
        }
    }
    else if (readoutSweep) {
        // Кривая на значение второй оси, как у показателя: один выход на сеть.
        const int n2 = (std::max)(1, (int)c.sweep2_values.size());
        if (c.sweep_values.empty()) empty = "Turn on the sweep and press Run.";
        else if (taskT == 0 || task.instances != (int)c.sweep_values.size() * n2) empty = "Turn on the readout in Neural Controls and press Run.";
        else {
            const int v = (std::min)(vs.readout_target, taskT - 1);
            curves = n2; points = (int)c.sweep_values.size();
            xof = [&c](int p) { return c.sweep_values[(size_t)p]; };
            yof = [&task, n2, taskT, v](int k, int p) { return task.test[((size_t)p * n2 + k) * taskT + v]; };
            yname = task.metric.empty() ? task.names[(size_t)v] : task.metric + " " + task.names[(size_t)v];
        }
    }
    else if (!byValue && !lyap && (r.n_points <= 0 || nPops <= 0)) empty = none;
    else if (!sweep && !lyap && !(sync && !spread) && recorded >= nVars) empty = "This variable is not recorded.";
    else if (byValue) {
        // Кривая на значение второй оси; у показателя одно значение на сеть, а не на популяцию.
        const std::vector<double>& d = sweep && spikes ? c.pop_rate : lyap ? c.sweep_lyap : spread ? c.sweep_spread : c.sweep_order;
        const int perNet = lyap ? 1 : K, n2 = (std::max)(1, (int)c.sweep2_values.size());
        if (c.sweep_values.empty()) empty = "Turn on the sweep and press Run.";
        else if (lyap && d.empty()) empty = "Turn on the Lyapunov exponent in Neural Controls and press Run.";
        else if (d.size() != (size_t)nInst * perNet) empty = none;
        else {
            curves = n2 * perNet; points = (int)c.sweep_values.size();
            xof = [&c](int p) { return c.sweep_values[(size_t)p]; };
            yof = [&d, perNet, n2](int k, int p) { return d[((size_t)p * n2 + k / perNet) * perNet + k % perNet]; };
            yname = sweep && spikes ? "rate" : lyap ? "lambda" : spread ? "spread" : "R";
        }
    }
    else if (lyap) {
        if (r.lyap_blocks < 1) empty = "The record holds no whole block.";
        else {
            // Накопленное среднее: видно, сошлась ли оценка на этой длине записи.
            static std::vector<double> running;
            running.assign((size_t)r.lyap_blocks, 0.0);
            double sum = 0.0;
            for (int b = 0; b < r.lyap_blocks; ++b) {
                sum += r.lyap_log[(size_t)b * nInst + vs.instance];
                running[(size_t)b] = sum / ((double)(b + 1) * r.lyap_dt);
            }
            curves = 1; points = r.lyap_blocks;
            xof = [&r](int p) { return r.t0 + r.lyap_dt * (double)(p + 1); };
            yof = [](int, int p) { return running[(size_t)p]; };
            yname = "lambda";
        }
    }
    else if (what == kNeuralMeans || (sync && spread)) {
        const std::vector<double>& d = sync ? r.spread : r.means;
        if (d.size() != (size_t)r.n_points * nPops * nVars) empty = none;
        else {
            curves = K;
            const int v = recorded;
            yof = [&d, nPops, nVars, p0, v](int k, int p) { return d[((size_t)p * nPops + p0 + k) * nVars + v]; };
            if (sync) yname = "spread " + yname;
        }
    }
    else if (sync) {
        if (r.order.size() != (size_t)r.n_points * nPops) empty = "The order parameter is off.";
        else {
            curves = K;
            yof = [&r, nPops, p0](int k, int p) { return r.order[(size_t)p * nPops + p0 + k]; };
            yname = "R";
        }
    }
    else {
        const size_t nSel = r.observed.size();
        if (r.traces.size() != (size_t)r.n_points * nSel * nVars) empty = none;
        else {
            const int i0 = c.instance_start[(size_t)vs.instance], i1 = c.instance_start[(size_t)vs.instance + 1];
            for (size_t q = 0; q < nSel; ++q) if (r.observed[q] >= i0 && r.observed[q] < i1) nodes.push_back((int)q);
            curves = (int)nodes.size();
            const int v = recorded;
            yof = [&r, &nodes, nSel, nVars, v](int k, int p) { return r.traces[((size_t)p * nSel + nodes[(size_t)k]) * nVars + v]; };
            if (curves == 0) empty = "No nodes are observed.";
        }
    }
    if (!empty.empty()) { ImGui::TextDisabled("%s", empty.c_str()); return; }

    // Вид держит указатели между кадрами.
    static std::vector<std::vector<double>> buffers[kNeuralWindowCount];
    using PlotKey = std::tuple<int, int, int, int, int, int, int, int, double>;
    static PlotKey filled[kNeuralWindowCount];
    static int generation[kNeuralWindowCount] = {};
    std::vector<std::vector<double>>& bufs = buffers[what];
    const PlotKey key = {epoch, c.data_generation, vs.series_var, vs.instance, vs.sync_kind, vs.sweep_kind,
                         curves, readout || readoutSweep ? vs.readout_target : 0, rate_bin};
    const bool fresh = filled[what] != key || (int)bufs.size() != curves;
    if (fresh) {
        bufs.assign((size_t)curves, {});
        for (int k = 0; k < curves; ++k) {
            std::vector<double>& b = bufs[(size_t)k];
            b.reserve((size_t)(raster ? points / (std::max)(1, curves) : points) * 2);
            for (int p = 0; p < points; ++p) {
                const double v = yof(k, p);
                const double x = series_xof ? series_xof(k, p) : xof(p);
                if (!std::isfinite(v) || !std::isfinite(x)) continue;
                b.push_back(x);
                b.push_back(v);
            }
        }
        filled[what] = key;
        ++generation[what];
    }
    const int gen = generation[what];
    std::vector<PlotSeriesInput> series;
    std::vector<bool> vis((size_t)curves, true);
    size_t longest = 0;
    for (int k = 0; k < curves; ++k) {
        const std::vector<double>& b = bufs[(size_t)k];
        longest = (std::max)(longest, b.size() / 2);
        PlotSeriesInput si;
        si.points = b.data();
        si.n_points = (int)(b.size() / 2);
        const float t = curves > 1 ? (float)k / (float)(curves - 1) : 0.0f;
        si.color = ImGui::ColorConvertU32ToFloat4(cmap_sample_id(t, model.heatmap_colormap));
        si.label = what == kNeuralNodes ? std::to_string(r.observed[(size_t)nodes[(size_t)k]])
                 : input ? ctx.sources[(size_t)k]
                 : weights ? (k < (int)c.watch_labels.size() ? c.watch_labels[(size_t)k] : std::to_string(k))
                 : readout && task.by_delay ? std::string(k == 0 ? "network" : k == 1 && delayCtl ? "K = 0" : "linear on delays")
                 : readout && task.forecast && k < (int)forecastKinds.size()
                     ? std::string(forecastKinds[(size_t)k] == kFcTrue ? "true" : forecastKinds[(size_t)k] == kFcNet ? "network"
                                 : forecastKinds[(size_t)k] == kFcAr ? "autoregression" : forecastKinds[(size_t)k] == kFcNvar ? "NVAR" : "threshold")
                 : readout ? std::string(k == 0 ? "true" : phase ? "forecast" : "readout")
                 : readoutSweep ? std::string(task.classify ? "accuracy" : "NRMSE")
                 : lyap ? std::string("lambda") : ctx.populations[(size_t)(k % K)];
        if (byValue && !c.sweep2_values.empty()) {
            const int perNet = lyap || readoutSweep ? 1 : K;
            char v[32];
            std::snprintf(v, sizeof v, " %g", c.sweep2_values[(size_t)(k / perNet)]);
            si.label += v;
        }
        series.push_back(si);
    }
    view.x_axis.name = byValue ? neural_sweep_name(ctx.sweep.kind) : phase ? task.names[0] : readout && task.by_delay ? "delay d, samples" : "t";
    view.y_axis.name = yname;
    if (longest == 0 && !raster) {
        ImGui::TextDisabled(readoutSweep ? "No network gave a readout: see the note in Neural Controls."
                            : sync || sweep ? "No finite values. R needs two threshold crossings of every node: run longer."
                                            : "No finite values.");
        return;
    }
    // Без запаса вырожденный диапазон (одна точка, все значения равны) даёт ось нулевой длины.
    view.pad_x = longest <= 1;
    // Ось времени по записи, а не по данным: у R края могут быть пустыми, окна должны совпадать по t.
    const double t_end = rate || raster ? (double)r.steps_done * r.h : r.t0 + r.dt * (double)(r.n_points - 1);
    view.x_fit_use_explicit = !byValue && !phase && (readout ? task.points > 1 : t_end > r.t0);
    view.x_fit_min = readout ? task.t0 : rate || raster ? (std::max)(r.t0, (double)ctx.first_record_step * r.h) : r.t0;
    view.x_fit_max = readout ? task.t0 + task.dt * (double)(task.points - 1) : t_end;
    if (readout && task.by_delay) { view.x_fit_use_explicit = true; view.x_fit_min = 1.0; view.x_fit_max = (double)(std::max)(2, delayD); }
    view.pad_y = true;
    view.show_zero_x = view.show_zero_y = false;
    view.point_markers = byValue || (readout && task.by_delay);
    // Клик по точке свипа выбирает её экземпляр для окон от времени; вторая ось остаётся как была.
    const int sweepN2 = (std::max)(1, (int)c.sweep2_values.size());
    view.left_click_plain = byValue;
    view.snap_x_to_grid = byValue;
    view.snap_x_nodes = byValue ? c.sweep_values.data() : nullptr;
    view.snap_x_node_count = byValue ? (int)c.sweep_values.size() : 0;
    view.on_left_click = nullptr;
    if (byValue) {
        view.on_left_click = [&vs, &c, sweepN2](double x) {
            int best = 0;
            for (int i = 1; i < (int)c.sweep_values.size(); ++i)
                if (std::fabs(c.sweep_values[(size_t)i] - x) < std::fabs(c.sweep_values[(size_t)best] - x)) best = i;
            vs.instance = best * sweepN2 + vs.instance % sweepN2;
        };
        if (vs.instance >= 0 && vs.instance < nInst) {
            if (c.sweep2_values.empty()) ImGui::Text("instance %d: %g", vs.instance, c.sweep_values[(size_t)vs.instance]);
            else ImGui::Text("instance %d: %g, %g", vs.instance, c.sweep_values[(size_t)(vs.instance / sweepN2)], c.sweep2_values[(size_t)(vs.instance % sweepN2)]);
        }
    }
    view.points_mode = raster;
    view.point_size_px = 2.0f;
    view.point_marker = -1;
    // Вид один на R и разброс: log10(R) на линейной оси дал бы отрицательные значения.
    bool* spread_log = sync ? &vs.spread_log[0] : sweep ? &vs.spread_log[1] : nullptr;
    if (spread_log) view.y_axis.display_log = spread && *spread_log;
    if ((raster || rate) && fresh) {
        // Activity navigation is horizontal by default. Set meaningful fixed Y
        // bounds once per record/instance (or when the rate bin changes).
        const int nodes_in_instance = c.instance_start[(size_t)vs.instance + 1] - c.instance_start[(size_t)vs.instance];
        view.y_axis.view_min = raster ? -0.5 : 0.0;
        view.y_axis.view_max = raster ? (double)nodes_in_instance - 0.5 : max_rate > 0 ? 1.05 * max_rate : 1.0;
        view.y_axis.user_range = false;
        view.y_axis.display_log = false;
        view.y_axis.lock = true;
    }
    else if ((sync || sweep) && !spread && !lyap && !(sweep && spikes) && !readoutSweep) {
        view.y_axis.view_min = -0.05; view.y_axis.view_max = 1.05;
        view.y_axis.lock = true;
    }
    else if (fresh) view.y_axis.lock = false;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, 0xBE7101 + what, gen, series, vis, vis, fresh, trial_bands);
    if (spread_log && spread) *spread_log = view.y_axis.display_log;
}

static void draw_neural_map_window(AppModel& model, PlotRenderer& renderer, std::map<int, std::unique_ptr<HeatmapView>>& maps) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty() || s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) {
        ImGui::TextDisabled("No system loaded.");
        return;
    }
    const int idx = s.active_config_index;
    NeuralConfig& c = s.configs[(size_t)idx];
    const auto& ctx = c.result_context;
    NeuralViewState& vs = model.neural_view;
    const int n1 = (int)c.sweep_values.size(), n2 = (int)c.sweep2_values.size(), K = c.pops_per_instance, nInst = n1 * n2;
    if (n2 == 0 || K < 1) {
        ImGui::TextDisabled("Turn on a sweep with a second axis and press Run.");
        return;
    }
    HeatmapView& hv = get_or_create_heatmap(maps, idx, -1, model.heatmap_colormap);
    const bool spikes = !c.result.spike_count.empty();
    choose_result_metric("##neural_map_metric", vs.map_kind, sweep_metrics(c));
    const NeuralTaskResult& task = c.task_result;
    const bool readout = vs.map_kind == (spikes ? 1 : 3);
    const int T = (int)task.names.size();
    if (readout && T > 0 && !task.classify) {
        const std::vector<const char*> items = c_str_list(task.names);
        if (vs.readout_target >= T) vs.readout_target = 0;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("target##readout_map", &vs.readout_target, items.data(), T);
    }
    if (vs.map_pop >= K) vs.map_pop = 0;
    if (K > 1 && !readout && (spikes || vs.map_kind != 2)) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::BeginCombo("##neural_map_pop", ctx.populations[(size_t)vs.map_pop].c_str())) {
            for (int p = 0; p < K; ++p)
                if (ImGui::Selectable(ctx.populations[(size_t)p].c_str(), p == vs.map_pop)) vs.map_pop = p;
            ImGui::EndCombo();
        }
    }
    if (!spikes && vs.map_kind == 1) { ImGui::SameLine(); ImGui::Checkbox("log10##neural_map", &vs.map_log); }
    if (vs.instance >= 0 && vs.instance < nInst) {
        ImGui::SameLine();
        ImGui::Text("instance %d: %g, %g", vs.instance, c.sweep_values[(size_t)(vs.instance / n2)], c.sweep2_values[(size_t)(vs.instance % n2)]);
    }
    draw_heatmap_toolbar(hv);

    const bool lyap = !spikes && vs.map_kind == 2, spread = !spikes && vs.map_kind == 1, log10_spread = spread && vs.map_log;
    // У выхода «популяция» ячейки это цель: test лежит как [сеть][цель].
    const std::vector<double>& d = readout ? task.test : spikes ? c.pop_rate : lyap ? c.sweep_lyap : spread ? c.sweep_spread : c.sweep_order;
    const int perNet = readout ? T : lyap ? 1 : K, pop = readout ? (std::min)(vs.readout_target, T - 1) : lyap ? 0 : vs.map_pop;
    if (perNet < 1 || d.size() != (size_t)nInst * perNet) {
        ImGui::TextDisabled(readout ? "Turn on the readout in Neural Controls and press Run."
                            : lyap ? "Turn on the Lyapunov exponent in Neural Controls and press Run." : "No data yet - press Run on the Neural tab.");
        return;
    }
    // Вид читает значения каждый кадр ради шкалы цвета, поэтому буфер живёт между кадрами.
    static std::vector<double> cells;
    cells.assign((size_t)nInst, kSentinelNoData);
    double vmin = std::numeric_limits<double>::infinity(), vmax = -vmin;
    for (int i1 = 0; i1 < n1; ++i1)
        for (int i2 = 0; i2 < n2; ++i2) {
            double v = d[(size_t)(i1 * n2 + i2) * perNet + pop];
            if (log10_spread) v = v > 0.0 ? std::log10(v) : std::numeric_limits<double>::quiet_NaN();
            if (!std::isfinite(v)) continue;
            cells[(size_t)i2 * n1 + i1] = v;
            vmin = (std::min)(vmin, v);
            vmax = (std::max)(vmax, v);
        }
    if (!std::isfinite(vmin)) { ImGui::TextDisabled(readout ? "No network gave a readout: see the note in Neural Controls." : "No finite values."); return; }
    if (!lyap && !spread && !spikes && !readout) { vmin = 0.0; vmax = 1.0; }

    hv.x_axis.name = neural_sweep_name(ctx.sweep.kind);
    hv.y_axis.name = neural_sweep_name(ctx.sweep2.kind);
    hv.x_axis.log_scale = ctx.sweep.log && c.sweep_values.front() > 0.0;
    hv.y_axis.log_scale = ctx.sweep2.log && c.sweep2_values.front() > 0.0;
    // Вид хранит указатель на имя между кадрами.
    static std::string value_name;
    value_name = readout ? (task.metric.empty() ? task.names[(size_t)pop] : task.metric + " " + task.names[(size_t)pop]) : spikes ? "rate" : lyap ? "lambda" : log10_spread ? "log10 spread" : spread ? "spread" : "R";
    hv.value_name = value_name.c_str();
    hv.left_click_plain = true;
    hv.on_left_click = [&vs, n2](int ix, int iy, double, double) { vs.instance = ix * n2 + iy; };

    double x0 = c.sweep_values.front(), x1 = c.sweep_values.back(), y0 = c.sweep2_values.front(), y1 = c.sweep2_values.back();
    if (x1 <= x0) x1 = x0 + 1.0;
    if (y1 <= y0) y1 = y0 + 1.0;
    const int epoch = model.session_epoch[(int)AppModel::AppMode::Neural];
    static std::tuple<int, int, int, int, bool> previous;
    static int generation = 0;
    const auto key = std::make_tuple(epoch, c.data_generation, vs.map_kind, pop, log10_spread);
    const bool fit = std::get<0>(previous) != epoch || std::get<1>(previous) != c.data_generation;
    if (previous != key) { previous = key; ++generation; }
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail, 0xBE71A000 + idx, generation, n1, n2, cells.data(), x0, x1, y0, y1, vmin, vmax, fit);
}

namespace {
struct LinkedNeuralPlots {
    std::pair<int, int> record = {-1, -1};
    AxisInfo x;
    std::map<std::string, ImVec4> colors;
};

struct NeuralHistogramPlot {
    Plot2DView view;
    std::unique_ptr<PlotRenderer> renderer;
    std::vector<double> points;
    std::tuple<int, int, size_t> key;
    int generation = 0;
};

struct NeuralResultsUI {
    std::array<std::unique_ptr<PlotRenderer>, kNeuralWindowCount> renderers;
    std::array<Plot2DView, kNeuralWindowCount> plots;
    std::unique_ptr<PlotRenderer> map_renderer;
    std::map<int, std::unique_ptr<HeatmapView>> maps;
    LinkedNeuralPlots activity, forecast;
    NeuralHistogramPlot isi, cv;
    int epoch = -1;
    int statistics_population = 0;
};

ImVec4 trial_stimulus_color(int label, size_t classes, int colormap) {
    const float t = label >= 0 && classes > 1 ? 0.15f + 0.7f * (float)label / (float)(classes - 1) : 0.6f;
    return ImGui::ColorConvertU32ToFloat4(cmap_sample_id(t, colormap));
}

std::vector<PlotXBand> trial_bands_in_view(const NeuralConfig& c, int instance, const AxisInfo& x, int colormap) {
    const auto& grid = c.result_context.trials;
    const auto& r = c.result;
    const double h = r.h, cycle = (double)grid.rest_steps + (double)grid.stimulus_steps;
    if (grid.count < 1 || !(h > 0) || !(cycle > 0)) return {};
    const long long stop = instance < (int)r.instance_stop.size() ? r.instance_stop[(size_t)instance] : -1;
    const double recorded_start = (std::max)(r.t0, (double)c.result_context.first_record_step * h);
    const double recorded_end = (double)(stop >= 0 ? stop : r.steps_done) * h;
    const double visible_start = (std::max)(recorded_start, (std::min)(x.view_min, x.view_max));
    const double visible_end = (std::min)(recorded_end, (std::max)(x.view_min, x.view_max));
    if (!(visible_end > visible_start)) return {};
    const auto index = [&](double t) { return (t / h - (double)grid.first_step) / cycle; };
    const int first = (int)std::clamp(std::floor(index(visible_start)), 0.0, (double)grid.count);
    const int last = (int)std::clamp(std::ceil(index(visible_end)), 0.0, (double)grid.count);
    if (last - first > 512) return {}; // zoom in before drawing individual labels on very long records
    std::vector<PlotXBand> bands;
    bands.reserve((size_t)(last - first) * 2);
    for (int i = first; i < last; ++i) {
        const double trial_step = (double)grid.first_step + i * cycle;
        const double start = trial_step * h, stimulus = (trial_step + (double)grid.rest_steps) * h;
        const double end = (trial_step + cycle) * h;
        const std::string number = "Trial " + std::to_string(i + 1);
        const int label = i < (int)grid.classes.size() ? grid.classes[(size_t)i] : -1;
        const bool named = label >= 0 && label < (int)grid.class_names.size();
        const std::string stimulus_name = named ? grid.class_names[(size_t)label] : "Stimulus";
        const auto color = trial_stimulus_color(named ? label : -1, grid.class_names.size(), colormap);
        if (grid.rest_steps > 0 && stimulus > visible_start && start < visible_end)
            bands.push_back({(std::max)(start, recorded_start), (std::min)(stimulus, recorded_end),
                             ImVec4(0.55f, 0.55f, 0.55f, 0.52f), "Rest", "", number + " · Rest"});
        if (grid.stimulus_steps > 0 && end > visible_start && stimulus < visible_end)
            bands.push_back({(std::max)(stimulus, recorded_start), (std::min)(end, recorded_end), color,
                             std::to_string(i + 1) + " · " + stimulus_name, "#" + std::to_string(i + 1),
                             number + " · Stimulus" + (named ? ": " + stimulus_name : "")});
    }
    return bands;
}

void draw_spike_histogram(NeuralHistogramPlot& plot, const std::tuple<int, int, size_t>& key, int id,
                          const char* units, const char* count_label, const NeuralSpikeHistogram& hist, float height, int colormap) {
    if (hist.counts.empty()) return;
    const bool fresh = plot.key != key || plot.points.empty();
    if (fresh) {
        plot.key = key;
        ++plot.generation;
        plot.points.clear();
        const auto last = std::find_if(hist.counts.rbegin(), hist.counts.rend(), [](auto count) { return count != 0; });
        const size_t occupied_end = (size_t)(last.base() - hist.counts.begin());
        // Keep one empty bin after the data so a narrow distribution does not
        // get buried at the edge of a long, empty tail.
        const size_t bins = (std::min)(hist.counts.size(), occupied_end + 1);
        plot.points.reserve(2 * bins);
        for (size_t i = 0; i < bins; ++i) {
            plot.points.push_back(((double)i + 0.5) * hist.bin_width);
            plot.points.push_back((double)hist.counts[i]);
        }
        plot.view.x_axis.user_range = plot.view.y_axis.user_range = false;
    }
    if (!plot.renderer) plot.renderer = std::make_unique<PlotRenderer>();
    auto& view = plot.view;
    view.x_axis.name = units;
    view.y_axis.name = count_label;
    view.show_legend = false;
    view.pad_x = false;
    view.pad_y = view.y_axis.display_log;
    view.show_zero_x = view.show_zero_y = false;
    PlotSeriesInput series;
    series.points = plot.points.data();
    series.n_points = (int)(plot.points.size() / 2);
    series.color = ImGui::ColorConvertU32ToFloat4(cmap_sample_id(0.55f, colormap));
    series.label = count_label;
    series.bar_width = hist.bin_width;
    view.render(*plot.renderer, ImGui::GetCursorScreenPos(), ImVec2(ImGui::GetContentRegionAvail().x, height),
                id, plot.generation, {series}, {true}, {true}, fresh);
}

void draw_spike_statistics(const NeuralConfig& c, int instance, NeuralResultsUI& ui, int colormap) {
    const auto& stats = c.spike_statistics;
    if (!stats.complete) {
        ImGui::TextWrapped(c.result.spikes_dropped > 0 ? "ISI and CV are unavailable: the spike log was truncated. Raise the output limit and run again."
                         : "ISI and CV are unavailable: this result does not have a complete spike record.");
        return;
    }
    std::vector<std::pair<int, const char*>> pops;
    for (int p = 0; p < c.pops_per_instance; ++p) pops.emplace_back(p, c.result_context.populations[(size_t)p].c_str());
    choose_result_metric("##statistics_population", ui.statistics_population, pops);
    const size_t pos = (size_t)instance * c.pops_per_instance + ui.statistics_population;
    if (pos >= stats.populations.size()) return;
    const auto& pop = stats.populations[pos];
    ImGui::TextDisabled("From the full spike log. Intervals never cross a state reset.");
    const float height = (std::max)(155.0f, 0.5f * ImGui::GetContentRegionAvail().y - 38.0f);
    const auto key = std::make_tuple(ui.epoch, c.data_generation, pos);
    if (pop.intervals) {
        ImGui::Text("ISI: %llu intervals, mean %.5g", pop.intervals, pop.mean_isi);
        draw_spike_histogram(ui.isi, key, 0x4E7101, "Interspike interval (time units)", "Intervals", pop.isi, height, colormap);
    } else ImGui::TextDisabled("ISI needs at least two spikes from the same neuron between resets.");
    if (pop.neurons_with_cv) {
        ImGui::Text("CV: %d neurons, mean %.5g", pop.neurons_with_cv, pop.mean_cv);
        draw_spike_histogram(ui.cv, key, 0x4E7102, "Coefficient of variation", "Neurons", pop.cv, height, colormap);
    } else ImGui::TextDisabled("CV needs at least two within-trial intervals from the same neuron.");
}
} // namespace

void draw_neural_results(AppModel& model) {
    auto& s = model.neural_session;
    const int idx = s.active_config_index;
    if (idx < 0 || idx >= (int)s.configs.size()) { ImGui::TextDisabled("No system loaded."); return; }
    const auto& c = s.configs[(size_t)idx];
    const auto& r = c.result;
    const auto& ctx = c.result_context;
    const auto available = neural_result_views(c);
    if (available.empty()) { ImGui::TextDisabled("No results for this experiment. Press Run in Neural Controls."); return; }
    ImGui::TextUnformatted(ctx.label.c_str());
    if (ctx.settings_key != model.neural_result_settings_key(idx) || ctx.mixed_settings)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Previous settings — showing the last successful result.");
    if (ctx.partial_history) ImGui::TextDisabled("Continued from a checkpoint; the earlier recording is unavailable.");
    if (r.spikes_dropped > 0)
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Spike log incomplete: %lld events omitted. Raster and binned rates are partial.", r.spikes_dropped);
    neural_instance_picker(c, model.neural_view, "Network instance");

    auto& ui = ui_resource<NeuralResultsUI, 3020>();
    const int epoch = model.session_epoch[(int)AppModel::AppMode::Neural];
    if (ui.epoch != epoch) {
        ui.epoch = epoch;
        for (auto& view : ui.plots) invalidate_view(view);
        ui.maps.clear();
    }
    const auto draw = [&](int which, const std::vector<PlotXBand>* bands = nullptr) {
        auto& renderer = ui.renderers[(size_t)which];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();
        draw_neural_series_window(model, *renderer, ui.plots[(size_t)which], which, bands);
    };
    const auto stack = [&](int top, int bottom, LinkedNeuralPlots& linked, double first, double last, bool same_colors) {
        const auto record = std::make_pair(epoch, c.data_generation);
        if (linked.record != record) {
            linked.record = record;
            linked.x = {};
            linked.x.view_min = first;
            linked.x.view_max = last > first ? last : first + 1.0;
        }
        if (ImGui::SmallButton("Reset time axis")) {
            linked.x.view_min = first;
            linked.x.view_max = last > first ? last : first + 1.0;
            linked.x.display_log = false;
        }
        const bool show_trials = top == kNeuralRaster && ctx.trials.count > 0;
        if (show_trials)
            ImGui::TextWrapped("Gray: rest. Color: stimulus. Hover the strip for trial and class; zoom in if the labels are crowded.");
        if (show_trials && !ctx.trials.class_names.empty() && ctx.trials.class_names.size() <= 6) {
            const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
            ImGui::TextUnformatted("Stimulus classes:");
            for (size_t k = 0; k < ctx.trials.class_names.size(); ++k) {
                const auto& label = ctx.trials.class_names[k];
                if (ImGui::GetItemRectMax().x + ImGui::CalcTextSize(label.c_str()).x + ImGui::GetStyle().ItemSpacing.x
                    < right) ImGui::SameLine();
                ImGui::TextColored(trial_stimulus_color((int)k, ctx.trials.class_names.size(), model.heatmap_colormap), "%s", label.c_str());
            }
        }
        const float height = (std::max)(200.0f, 0.5f * (ImGui::GetContentRegionAvail().y - ImGui::GetStyle().ItemSpacing.y));
        float left_top, left_bottom, t, right, b;
        const auto margin = [&](int which, float& left) {
            const auto& view = ui.plots[(size_t)which];
            plot_2d_margins_for(view.y_axis, view.y_axis.name.c_str(), left, t, right, b);
        };
        margin(top, left_top); margin(bottom, left_bottom);
        for (const int which : {top, bottom}) {
            auto& view = ui.plots[(size_t)which];
            ImGui::PushID(which);
            if (ImGui::BeginChild("##stacked_neural_plot", ImVec2(0, height), ImGuiChildFlags_None,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                view.shared_margin_left = (std::max)(left_top, left_bottom);
                view.shared_x_range = true;
                view.x_axis = linked.x;
                if (same_colors) view.series_color_override = linked.colors;
                const auto bands = show_trials ? trial_bands_in_view(c, model.neural_view.instance, view.x_axis, model.heatmap_colormap)
                                               : std::vector<PlotXBand>{};
                draw(which, show_trials ? &bands : nullptr);
                linked.x = view.x_axis;
                if (same_colors) linked.colors = view.series_color_override;
            }
            ImGui::EndChild();
            ImGui::PopID();
        }
    };

    const auto& task = c.task_result;
    using View = NeuralResultView;
    if (ImGui::BeginTabBar("##neural_result_views")) {
        const auto tab = [&](const char* title, auto show) {
            if (ImGui::BeginTabItem(title)) { show(); ImGui::EndTabItem(); }
        };
        for (View view : available) switch (view) {
        case View::Activity:
            tab("Activity", [&] { stack(kNeuralRaster, kNeuralRate, ui.activity,
                                       (std::max)(r.t0, (double)ctx.first_record_step * r.h), (double)r.steps_done * r.h, true); });
            break;
        case View::SpikeStatistics:
            tab("Spike statistics", [&] { draw_spike_statistics(c, model.neural_view.instance, ui, model.heatmap_colormap); });
            break;
        case View::Readout:
            if (task.forecast) tab("Forecast", [&] { stack(kNeuralReadout, kNeuralForecastError, ui.forecast,
                                                           task.t0, task.t0 + task.dt * (task.points - 1), false); });
            else tab(task.classify ? "Classification" : task.by_delay ? (task.names[0] == "MC" ? "Memory" : "XOR") : "Readout",
                     [&] { draw(kNeuralReadout); });
            break;
        case View::ForecastPhase: tab("Phase portrait", [&] { draw(kNeuralForecastPhase); }); break;
        case View::Means: tab("Population means", [&] { draw(kNeuralMeans); }); break;
        case View::Nodes: tab("Observed nodes", [&] { draw(kNeuralNodes); }); break;
        case View::Synchronisation: tab("Synchronisation", [&] { draw(kNeuralSync); }); break;
        case View::Lyapunov: tab("Lyapunov", [&] { draw(kNeuralLyap); }); break;
        case View::Input: tab("Input", [&] { draw(kNeuralInput); }); break;
        case View::Weights: tab("Weights", [&] { draw(kNeuralWeights); }); break;
        case View::Sweep:
            tab("Sweep", [&] {
                if (c.sweep2_values.empty()) draw(kNeuralSweep);
                else {
                    if (!ui.map_renderer) ui.map_renderer = std::make_unique<PlotRenderer>();
                    draw_neural_map_window(model, *ui.map_renderer, ui.maps);
                }
            });
            break;
        }
        ImGui::EndTabBar();
    }
}

}  // namespace ucuda::ui_detail
