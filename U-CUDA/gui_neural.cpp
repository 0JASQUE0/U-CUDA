#include "gui_internal.h"

// Вкладка Neural: описание сети популяциями и проекциями. Узлы и рёбра не показываются: их слишком много,
// вместо них статистика графа и наблюдаемые величины.

static const char* const kNeuralRules[] = { "Probability", "Fixed in-degree", "Chain", "Ring", "Grid", "King graph", "Small world" };
static const char* const kNeuralNorms[] = { "None", "This projection", "All inputs of the node" };
static const NeuralSchedule kNeuralSchedules[] = { NeuralSchedule::Stage, NeuralSchedule::Heun, NeuralSchedule::Post };
static const NeuralSweepKind kNeuralSweeps[] = { NeuralSweepKind::Coupling, NeuralSweepKind::Parameter,
                                                 NeuralSweepKind::ParameterSpread, NeuralSweepKind::Replica,
                                                 NeuralSweepKind::InputGain };
static const NeuralSourceKind kNeuralSources[] = { NeuralSourceKind::Constant, NeuralSourceKind::Sine, NeuralSourceKind::Square,
                                                   NeuralSourceKind::Noise, NeuralSourceKind::System };

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

static void draw_neural_projections(NeuralConfig& c) {
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
            std::vector<const char*> laws;
            for (const NetCouplingLaw& l : c.laws) laws.push_back(l.name.c_str());
            ImGui::SetNextItemWidth(kFieldW);
            ImGui::Combo("coupling law", &p.law, laws.data(), (int)laws.size());
            if (ImGui::Button("Remove projection")) remove = (int)i;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (remove >= 0) c.projections.erase(c.projections.begin() + remove);
    if (ImGui::Button("Add projection")) c.projections.push_back(NeuralProjection{});
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
    ImGui::TextDisabled("Recorded time %g replaces t max.", neural_record_time(c));
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
        g.enqueue = [&model](int i) { model.neural_queue.push_back({ i }); };
        draw_run_and_run_all("##run_all_neural", s.in_flight, false, s.in_flight, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             {},
                             { g },
                             [&model]() { model.start_next_in_neural_queue(); },
                             model.neural_queue.size());
        ImGui::SameLine();
        if (!s.in_flight) ImGui::BeginDisabled();
        if (ImGui::Button("Stop", ImVec2(90, 0))) { s.request_cancel(); model.neural_queue.clear(); }
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
                          "Heun and Post are checked on diffusive coupling of Hindmarsh-Rose pairs only.");
    if (c.schedule != NeuralSchedule::Stage)
        draw_scheme_combo("node scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                          nullptr, s.sys.is_map, &s.wrapper_schemes);
    else
        ImGui::TextDisabled("Node step: RK4 (the only scheme with coupling in stages so far).");

    IntegrationFields f;
    f.h = &c.h_text; f.symmetry_s = &c.symmetry_s; f.cd_fp_iters = &c.cd_fp_iters; f.newton = &c.newton;
    f.t_max = &c.t_max_text; f.transient = &c.transient_text; f.pre_scaller = &c.pre_scaller_text; f.max_value = &c.max_value_text;
    draw_integration_block("Integration", c.schedule == NeuralSchedule::Stage ? std::string("RK4") : c.scheme,
                           s.custom_schemes, f, nullptr, s.sys.is_map);

    if (ImGui::CollapsingHeader("Network", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("coupling K", c.coupling_text, kFieldW);
        InputNumStr("seed", c.seed_text, kFieldW);
        InputNumStr("observed nodes per population", c.observed_per_pop_text, kFieldW);
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
    draw_neural_input(lib, s, c);
    draw_neural_protocol(c);
    draw_neural_sweep(s, c);
    draw_neural_lyapunov(c);
    draw_neural_populations(s, c);
    draw_neural_projections(c);
    draw_neural_laws(s, c);

    if (c.arcs > 0 || c.last_run_ok) {
        ImGui::Separator();
        ImGui::Text("%d nodes, %lld arcs, in-degree %d..%d", c.result.n_nodes, c.arcs, c.degree_min, c.degree_max);
        ImGui::Text("graph %.2f s, run %.2f s, %.0f MB on the GPU", c.build_seconds, c.result.wall_ms / 1000.0,
                    (double)c.result.device_bytes / (1024.0 * 1024.0));
        ImGui::TextDisabled("graph hash %s", c.graph_hash.c_str());
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
    if (!c.last_error.empty()) {
        ImGui::Separator();
        draw_error_box("##neural_err", c.last_error);
    }
    if (c.hash_mismatch && ImGui::Button("Rebuild as a new experiment")) {
        c.graph_hash.clear(); c.graph_recipe.clear();
        c.hash_mismatch = false; c.last_error.clear();
    }
}

// Номер показываемого экземпляра и что с ним стало.
static void neural_instance_picker(const NeuralConfig& c, NeuralViewState& vs, const char* id) {
    const int nInst = (int)c.instance_start.size() - 1;
    if (vs.instance >= nInst) vs.instance = (std::max)(0, nInst - 1);
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

// Окна вкладки: кривые от времени для выбранного экземпляра, сводка свипа от значения.
void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, int what) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty() || s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) {
        ImGui::TextDisabled("No system loaded.");
        return;
    }
    NeuralConfig& c = s.configs[(size_t)s.active_config_index];
    const NeuralResult& r = c.result;
    NeuralViewState& vs = model.neural_view;
    const int nVars = (int)r.observed_vars.size(), K = c.pops_per_instance;
    const int recorded = (int)(std::find(r.observed_vars.begin(), r.observed_vars.end(), vs.series_var) - r.observed_vars.begin());
    const int nInst = (int)c.instance_start.size() - 1, nPops = (std::max)(0, nInst) * K;
    if (vs.series_var < 0 || vs.series_var >= (int)s.vars.size()) vs.series_var = 0;

    if (what == kNeuralInput) {
        const int ch = (int)c.sources.size();
        if (ch == 0 || c.input_view.empty()) { ImGui::TextDisabled("No input yet: add a source and a link, then press Run."); return; }
    }
    const bool sync = what == kNeuralSync, sweep = what == kNeuralSweep, input = what == kNeuralInput, lyapWindow = what == kNeuralLyap;
    if (sync || sweep) {
        static const char* const kinds[] = { "Order parameter R", "Spread across nodes", "Lyapunov exponent" };
        ImGui::SetNextItemWidth(170.0f);
        ImGui::Combo(sweep ? "##neural_sweep_kind" : "##neural_sync_kind", sweep ? &vs.sweep_kind : &vs.sync_kind, kinds, sweep ? 3 : 2);
    }
    const bool lyap = lyapWindow || (sweep && vs.sweep_kind == 2);
    if (lyapWindow) {
        if (r.lyapunov.empty()) {
            ImGui::TextDisabled("Turn on the Lyapunov exponent in Neural Controls and press Run.");
            return;
        }
        if (r.lyapunov.size() > 1) {
            static const char* const kinds[] = { "Against the sweep value", "Running mean of one network" };
            ImGui::SetNextItemWidth(210.0f);
            ImGui::Combo("##neural_lyap_kind", &vs.lyap_kind, kinds, IM_ARRAYSIZE(kinds));
            ImGui::SameLine();
        }
        ImGui::Text("lambda = %.6g", r.lyapunov[(size_t)(std::min)(vs.instance, (int)r.lyapunov.size() - 1)]);
        ImGui::SameLine();
        ImGui::TextDisabled("%d blocks of %g", r.lyap_blocks, r.lyap_dt);
    }
    // Показатель от значения свипа рисуется как сводка свипа.
    const bool byValue = sweep || (lyapWindow && r.lyapunov.size() > 1 && vs.lyap_kind == 0);
    const bool spread = (sync && vs.sync_kind == 1) || (sweep && vs.sweep_kind == 1);
    if (!sweep && !input && !lyap && !s.vars.empty() && (!sync || spread)) {
        const std::vector<const char*> items = c_str_list(s.vars);
        if (sync) ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo(what == kNeuralMeans ? "variable##means" : what == kNeuralNodes ? "variable##nodes" : "variable##sync",
                     &vs.series_var, items.data(), (int)items.size());
    }
    if (!byValue && !input) neural_instance_picker(c, vs, what == kNeuralMeans ? "instance##means" : what == kNeuralNodes ? "instance##nodes" : lyap ? "instance##lyap" : "instance##sync");

    // Точка p кривой k.
    std::function<double(int)> xof = [&](int p) { return r.t0 + r.dt * (double)p; };
    std::function<double(int, int)> yof;
    int curves = 0, points = r.n_points;
    std::vector<int> nodes;
    std::string yname = !s.vars.empty() ? s.vars[(size_t)vs.series_var] : std::string();
    const std::string none = "No data yet - press Run on the Neural tab.";
    std::string empty;
    const int p0 = vs.instance * K;
    if (input) {
        const int ch = (int)c.sources.size();
        curves = ch; points = (int)(c.input_view.size() / (size_t)ch);
        xof = [&c](int p) { return c.input_view_t0 + c.input_view_dt * (double)p; };
        yof = [&c, ch](int k, int p) { return c.input_view[(size_t)p * ch + k]; };
        yname = "input";
    }
    else if (!lyap && (r.n_points <= 0 || nPops <= 0)) empty = none;
    else if (!sweep && !lyap && !(sync && !spread) && recorded >= nVars) empty = "This variable is not recorded.";
    else if (byValue) {
        // Кривая на значение второй оси; у показателя одно значение на сеть, а не на популяцию.
        const std::vector<double>& d = lyap ? c.sweep_lyap : spread ? c.sweep_spread : c.sweep_order;
        const int perNet = lyap ? 1 : K, n2 = (std::max)(1, (int)c.sweep2_values.size());
        if (c.sweep_values.empty()) empty = "Turn on the sweep and press Run.";
        else if (lyap && d.empty()) empty = "Turn on the Lyapunov exponent in Neural Controls and press Run.";
        else if (d.size() != (size_t)nInst * perNet) empty = none;
        else {
            curves = n2 * perNet; points = (int)c.sweep_values.size();
            xof = [&c](int p) { return c.sweep_values[(size_t)p]; };
            yof = [&d, perNet, n2](int k, int p) { return d[((size_t)p * n2 + k / perNet) * perNet + k % perNet]; };
            yname = lyap ? "lambda" : spread ? "spread" : "R";
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
    static int filled[kNeuralWindowCount] = {};
    std::vector<std::vector<double>>& bufs = buffers[what];
    const int gen = (((((c.data_generation * 131 + vs.series_var) * 131 + s.active_config_index) * 131 + vs.instance) * 131 + vs.sync_kind * 3 + vs.sweep_kind) * 131 + curves) * 131 + vs.lyap_kind;
    const bool fresh = (int)bufs.size() != curves || filled[what] != gen + 1;
    if (fresh) {
        bufs.assign((size_t)curves, {});
        for (int k = 0; k < curves; ++k) {
            std::vector<double>& b = bufs[(size_t)k];
            b.reserve((size_t)points * 2);
            for (int p = 0; p < points; ++p) {
                const double v = yof(k, p);
                if (!std::isfinite(v)) continue;
                b.push_back(xof(p));
                b.push_back(v);
            }
        }
        filled[what] = gen + 1;
    }
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
                 : input ? c.sources[(size_t)k].name
                 : lyap ? std::string("lambda") : c.populations[(size_t)(k % K)].name;
        if (byValue && !c.sweep2_values.empty()) {
            const int perNet = lyap ? 1 : K;
            char v[32];
            std::snprintf(v, sizeof v, " %g", c.sweep2_values[(size_t)(k / perNet)]);
            si.label += v;
        }
        series.push_back(si);
    }
    view.x_axis.name = byValue ? neural_sweep_name(c.sweep.kind) : "t";
    view.y_axis.name = yname;
    if (longest == 0) {
        ImGui::TextDisabled(sync || sweep ? "No finite values. R needs two threshold crossings of every node: run longer."
                                          : "No finite values.");
        return;
    }
    // Без запаса вырожденный диапазон (одна точка, все значения равны) даёт ось нулевой длины.
    view.pad_x = longest <= 1;
    // Ось времени по записи, а не по данным: у R края могут быть пустыми, окна должны совпадать по t.
    const double t_end = r.t0 + r.dt * (double)(r.n_points - 1);
    view.x_fit_use_explicit = !byValue && t_end > r.t0;
    view.x_fit_min = r.t0;
    view.x_fit_max = t_end;
    view.pad_y = true;
    view.show_zero_x = view.show_zero_y = false;
    view.point_markers = byValue;
    // Вид один на R и разброс: log10(R) на линейной оси дал бы отрицательные значения.
    bool* spread_log = sync ? &vs.spread_log[0] : sweep ? &vs.spread_log[1] : nullptr;
    if (spread_log) view.y_axis.display_log = spread && *spread_log;
    if ((sync || sweep) && !spread && !lyap) {
        view.y_axis.view_min = -0.05; view.y_axis.view_max = 1.05;
        view.y_axis.lock = true;
    }
    else if (fresh) view.y_axis.lock = false;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, 0xBE7101 + what, gen, series, vis, vis, fresh);
    if (spread_log && spread) *spread_log = view.y_axis.display_log;
}

void draw_neural_map_window(AppModel& model, PlotRenderer& renderer, std::map<int, std::unique_ptr<HeatmapView>>& maps) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty() || s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) {
        ImGui::TextDisabled("No system loaded.");
        return;
    }
    const int idx = s.active_config_index;
    NeuralConfig& c = s.configs[(size_t)idx];
    NeuralViewState& vs = model.neural_view;
    const int n1 = (int)c.sweep_values.size(), n2 = (int)c.sweep2_values.size(), K = c.pops_per_instance, nInst = n1 * n2;
    if (n2 == 0 || K < 1) {
        ImGui::TextDisabled("Turn on a sweep with a second axis and press Run.");
        return;
    }
    HeatmapView& hv = get_or_create_heatmap(maps, idx, -1, model.heatmap_colormap);
    static const char* const kinds[] = { "Order parameter R", "Spread across nodes", "Lyapunov exponent" };
    ImGui::SetNextItemWidth(170.0f);
    ImGui::Combo("##neural_map_kind", &vs.map_kind, kinds, IM_ARRAYSIZE(kinds));
    if (vs.map_pop >= K) vs.map_pop = 0;
    if (K > 1 && vs.map_kind != 2) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::BeginCombo("##neural_map_pop", c.populations[(size_t)vs.map_pop].name.c_str())) {
            for (int p = 0; p < K; ++p)
                if (ImGui::Selectable(c.populations[(size_t)p].name.c_str(), p == vs.map_pop)) vs.map_pop = p;
            ImGui::EndCombo();
        }
    }
    if (vs.map_kind == 1) { ImGui::SameLine(); ImGui::Checkbox("log10##neural_map", &vs.map_log); }
    draw_heatmap_toolbar(hv);

    const bool lyap = vs.map_kind == 2, spread = vs.map_kind == 1, log10_spread = spread && vs.map_log;
    const std::vector<double>& d = lyap ? c.sweep_lyap : spread ? c.sweep_spread : c.sweep_order;
    const int perNet = lyap ? 1 : K, pop = lyap ? 0 : vs.map_pop;
    if (d.size() != (size_t)nInst * perNet) {
        ImGui::TextDisabled(lyap ? "Turn on the Lyapunov exponent in Neural Controls and press Run." : "No data yet - press Run on the Neural tab.");
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
    if (!std::isfinite(vmin)) { ImGui::TextDisabled("No finite values."); return; }
    if (!lyap && !spread) { vmin = 0.0; vmax = 1.0; }

    hv.x_axis.name = neural_sweep_name(c.sweep.kind);
    hv.y_axis.name = neural_sweep_name(c.sweep2.kind);
    hv.x_axis.log_scale = c.sweep.log && c.sweep_values.front() > 0.0;
    hv.y_axis.log_scale = c.sweep2.log && c.sweep2_values.front() > 0.0;
    hv.value_name = lyap ? "lambda" : log10_spread ? "log10 spread" : spread ? "spread" : "R";
    hv.on_left_click = [&vs, n2](int ix, int iy, double, double) { vs.instance = ix * n2 + iy; };

    double x0 = c.sweep_values.front(), x1 = c.sweep_values.back(), y0 = c.sweep2_values.front(), y1 = c.sweep2_values.back();
    if (x1 <= x0) x1 = x0 + 1.0;
    if (y1 <= y0) y1 = y0 + 1.0;
    const int gen = (((c.data_generation * 131 + idx) * 131 + vs.map_kind) * 131 + pop) * 131 + (log10_spread ? 1 : 0);
    static int shown = -1;
    const bool fit = shown != c.data_generation * 131 + idx;
    shown = c.data_generation * 131 + idx;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail, 0xBE71A000 + idx, gen, n1, n2, cells.data(), x0, x1, y0, y1, vmin, vmax, fit);
}
