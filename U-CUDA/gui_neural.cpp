#include "gui_internal.h"

// Вкладка Neural: описание сети популяциями и проекциями. Узлы и рёбра не показываются: их слишком много,
// вместо них статистика графа и наблюдаемые величины.

static const char* const kNeuralRules[] = { "Probability", "Fixed in-degree", "Chain", "Ring", "Grid", "King graph", "Small world" };
static const char* const kNeuralNorms[] = { "None", "This projection", "All inputs of the node" };
static const NeuralSchedule kNeuralSchedules[] = { NeuralSchedule::Stage, NeuralSchedule::Heun, NeuralSchedule::Post };

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

void draw_neural_controls(AppModel& model, SystemLibrary& /*lib*/) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No system loaded. Pick a system in the list above.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) s.active_config_index = 0;

    const ImGuiIO& io = ImGui::GetIO();
    if (s.in_flight) ImGui::BeginDisabled();
    bool do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    if (s.in_flight) ImGui::EndDisabled();
    if (!s.in_flight && io.KeyCtrl && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_R, false)) do_run = true;
    if (do_run) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, s.active_config_index);
    }
    ImGui::SameLine();
    if (!s.in_flight) ImGui::BeginDisabled();
    if (ImGui::Button("Stop", ImVec2(90, 0))) s.request_cancel();
    if (!s.in_flight) ImGui::EndDisabled();
    if (s.in_flight && s.progress_token) {
        ImGui::SameLine();
        ImGui::ProgressBar(s.progress_token->load(std::memory_order_relaxed), ImVec2(-1, 0));
    }
    ImGui::Separator();

    const TabBarResult tabs = draw_config_tab_bar(
        "##neural_tabs", "neural_tab_", (int)s.configs.size(), s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; }, {}, [&s]() { s.add_config(); });
    if (tabs.active >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) s.remove_config(tabs.to_remove);
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
    }
    draw_neural_populations(s, c);
    draw_neural_projections(c);
    draw_neural_laws(s, c);

    if (c.arcs > 0 || c.last_run_ok) {
        ImGui::Separator();
        ImGui::Text("%d nodes, %lld arcs, in-degree %d..%d", c.result.n_nodes, c.arcs, c.degree_min, c.degree_max);
        ImGui::Text("graph %.2f s, run %.2f s, %.0f MB on the GPU", c.build_seconds, c.result.wall_ms / 1000.0,
                    (double)c.result.device_bytes / (1024.0 * 1024.0));
        ImGui::TextDisabled("graph hash %s", c.graph_hash.c_str());
        if (c.result.diverged)
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

// Кривые средних по популяциям или выбранных узлов.
void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, bool means) {
    NeuralSession& s = model.neural_session;
    if (s.configs.empty() || s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) {
        ImGui::TextDisabled("No system loaded.");
        return;
    }
    NeuralConfig& c = s.configs[(size_t)s.active_config_index];
    const NeuralResult& r = c.result;
    NeuralViewState& vs = model.neural_view;
    if (!s.vars.empty()) {
        const std::vector<const char*> items = c_str_list(s.vars);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo(means ? "variable##means" : "variable##nodes", &vs.series_var, items.data(), (int)items.size());
    }
    if (vs.series_var < 0 || vs.series_var >= (int)s.vars.size()) vs.series_var = 0;
    const int nVars = (int)s.vars.size();
    const int curves = means ? (int)c.populations.size() : (int)r.observed.size();
    const std::vector<double>& data = means ? r.means : r.traces;
    if (r.n_points <= 0 || curves <= 0 || data.size() != (size_t)r.n_points * curves * nVars) {
        ImGui::TextDisabled("No data yet - press Run on the Neural tab.");
        return;
    }
    // Вид держит указатели между кадрами.
    static std::vector<std::vector<double>> buffers[2];
    std::vector<std::vector<double>>& bufs = buffers[means ? 0 : 1];
    if ((int)bufs.size() != curves) bufs.assign((size_t)curves, {});
    std::vector<PlotSeriesInput> series;
    std::vector<bool> vis((size_t)curves, true);
    for (int k = 0; k < curves; ++k) {
        std::vector<double>& b = bufs[(size_t)k];
        b.clear();
        b.reserve((size_t)r.n_points * 2);
        for (int p = 0; p < r.n_points; ++p) {
            const double v = data[((size_t)p * curves + k) * nVars + vs.series_var];
            if (!std::isfinite(v)) continue;
            b.push_back(r.t0 + r.dt * (double)p);
            b.push_back(v);
        }
        PlotSeriesInput si;
        si.points = b.data();
        si.n_points = (int)(b.size() / 2);
        const float t = curves > 1 ? (float)k / (float)(curves - 1) : 0.0f;
        si.color = ImGui::ColorConvertU32ToFloat4(cmap_sample_id(t, model.heatmap_colormap));
        si.label = means ? c.populations[(size_t)k].name : std::to_string(r.observed[(size_t)k]);
        series.push_back(si);
    }
    view.x_axis.name = "t";
    view.y_axis.name = s.vars[(size_t)vs.series_var];
    view.pad_x = view.pad_y = false;
    view.show_zero_x = view.show_zero_y = false;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const int gen = (c.data_generation * 131 + vs.series_var) * 131 + s.active_config_index * 7919;
    view.render(renderer, origin, avail, means ? 0xBE7101 : 0xBE7102, gen, series, vis, vis, false);
}
