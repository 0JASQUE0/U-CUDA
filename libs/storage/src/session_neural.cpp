#include "session_json_internal.h"

using namespace ucuda::storage_detail;

// Neural-сессия. Перенесено из U-CUDA/session_io.cpp (main, d30f0ac) без изменений в телах.

static void write_neural_config(std::ostringstream& o, const NeuralConfig& c) {
    o << "{\"label\":"; jstr(o, c.label);
    o << ",\"schedule\":" << (int)c.schedule;
    o << ",\"scheme\":"; jstr(o, c.scheme);
    o << ",\"symmetry_s\":"; jstr(o, c.symmetry_s);
    o << ",\"cd_fp_iters\":"; jstr(o, c.cd_fp_iters);
    o << ",\"newton_tol\":"; jstr(o, c.newton.tol);
    o << ",\"newton_max_iters\":"; jstr(o, c.newton.max_iters);
    o << ",\"newton_full\":" << (c.newton.full ? "true" : "false");
    o << ",\"h_text\":"; jstr(o, c.h_text);
    o << ",\"t_max_text\":"; jstr(o, c.t_max_text);
    o << ",\"transient_text\":"; jstr(o, c.transient_text);
    o << ",\"pre_scaller_text\":"; jstr(o, c.pre_scaller_text);
    o << ",\"max_value_text\":"; jstr(o, c.max_value_text);
    o << ",\"coupling_text\":"; jstr(o, c.coupling_text);
    o << ",\"seed_text\":"; jstr(o, c.seed_text);
    o << ",\"observed_per_pop_text\":"; jstr(o, c.observed_per_pop_text);
    o << ",\"order_var\":" << c.order_var;
    o << ",\"order_threshold\":"; jstr(o, c.order_threshold_text);
    o << ",\"sweep_on\":" << (c.sweep.enabled ? "true" : "false");
    o << ",\"sweep_kind\":" << (int)c.sweep.kind;
    o << ",\"sweep_pop\":" << c.sweep.population;
    o << ",\"sweep_param\":"; jstr(o, c.sweep.param);
    o << ",\"sweep_lo\":"; jstr(o, c.sweep.lo_text);
    o << ",\"sweep_hi\":"; jstr(o, c.sweep.hi_text);
    o << ",\"sweep_points\":"; jstr(o, c.sweep.points_text);
    o << ",\"sweep_log\":" << (c.sweep.log ? "true" : "false");
    o << ",\"sweep2_on\":" << (c.sweep2.enabled ? "true" : "false");
    o << ",\"sweep2_kind\":" << (int)c.sweep2.kind;
    o << ",\"sweep2_pop\":" << c.sweep2.population;
    o << ",\"sweep2_param\":"; jstr(o, c.sweep2.param);
    o << ",\"sweep2_lo\":"; jstr(o, c.sweep2.lo_text);
    o << ",\"sweep2_hi\":"; jstr(o, c.sweep2.hi_text);
    o << ",\"sweep2_points\":"; jstr(o, c.sweep2.points_text);
    o << ",\"sweep2_log\":" << (c.sweep2.log ? "true" : "false");
    o << ",\"lyap_on\":" << (c.lyap_on ? "true" : "false");
    o << ",\"lyap_eps\":"; jstr(o, c.lyap_eps_text);
    o << ",\"lyap_interval\":"; jstr(o, c.lyap_interval_text);
    o << ",\"lyap_settle\":"; jstr(o, c.lyap_settle_text);
    o << ",\"csv_save\":" << (c.csv_save_enabled ? "true" : "false");
    o << ",\"csv_path\":"; jstr(o, c.csv_output_path);
    o << ",\"obs_vars_off\":[";
    for (size_t i = 0; i < c.obs_vars_off.size(); ++i) o << (i ? "," : "") << c.obs_vars_off[i];
    o << "],\"output_limit\":"; jstr(o, c.output_limit_text);
    o << ",\"input_var\":" << c.input_var;
    o << ",\"input_hold\":"; jstr(o, c.input_hold_text);
    o << ",\"protocol\":" << (int)c.protocol;
    o << ",\"trials\":"; jstr(o, c.trials_text);
    o << ",\"rest\":"; jstr(o, c.rest_text);
    o << ",\"stimulus\":"; jstr(o, c.stimulus_text);
    o << ",\"reset_state\":" << (c.reset_state ? "true" : "false");
    const auto strings = [&o](const std::vector<std::string>& v) {
        o << "[";
        for (size_t k = 0; k < v.size(); ++k) { if (k) o << ","; jstr(o, v[k]); }
        o << "]";
    };
    o << ",\"sources\":[";
    for (size_t i = 0; i < c.sources.size(); ++i) {
        const NeuralSource& s = c.sources[i];
        if (i) o << ",";
        o << "{\"name\":"; jstr(o, s.name);
        o << ",\"kind\":" << (int)s.kind;
        o << ",\"amplitude\":"; jstr(o, s.amplitude_text);
        o << ",\"frequency\":"; jstr(o, s.frequency_text);
        o << ",\"phase\":"; jstr(o, s.phase_text);
        o << ",\"offset\":"; jstr(o, s.offset_text);
        o << ",\"system\":"; jstr(o, s.system_name);
        o << ",\"vars\":"; strings(s.vars);
        o << ",\"params\":"; strings(s.params);
        o << ",\"rhs\":"; strings(s.rhs);
        o << ",\"latex\":" << (s.latex ? "true" : "false");
        o << ",\"ic\":"; jmap(o, s.initial_conditions);
        o << ",\"values\":"; jmap(o, s.param_values);
        o << ",\"variable\":" << s.variable;
        o << ",\"step\":"; jstr(o, s.step_text);
        o << ",\"time_scale\":"; jstr(o, s.time_scale_text);
        o << ",\"skip\":"; jstr(o, s.skip_text);
        o << ",\"normalize\":" << (s.normalize ? "true" : "false") << "}";
    }
    o << "],\"links\":[";
    for (size_t i = 0; i < c.links.size(); ++i) {
        const NeuralLink& l = c.links[i];
        if (i) o << ",";
        o << "{\"source\":" << l.source << ",\"population\":" << l.population;
        o << ",\"gain\":"; jstr(o, l.gain_text);
        o << ",\"gain_spread\":"; jstr(o, l.gain_spread_text);
        o << ",\"fraction\":"; jstr(o, l.fraction_text);
        o << "}";
    }
    o << "]";
    o << ",\"graph_version\":" << kNeuralGraphVersion;
    o << ",\"graph_recipe\":"; jstr(o, c.graph_recipe);
    o << ",\"graph_hash\":"; jstr(o, c.graph_hash);
    o << ",\"model_text\":"; jstr(o, c.model_text);
    o << ",\"laws\":[";
    for (size_t i = 0; i < c.laws.size(); ++i) {
        if (i) o << ",";
        o << "{\"name\":"; jstr(o, c.laws[i].name);
        o << ",\"expr\":[";
        for (size_t k = 0; k < c.laws[i].expr.size(); ++k) {
            if (k) o << ",";
            jstr(o, c.laws[i].expr[k]);
        }
        o << "]}";
    }
    o << "],\"populations\":[";
    for (size_t i = 0; i < c.populations.size(); ++i) {
        const NeuralPopulation& p = c.populations[i];
        if (i) o << ",";
        o << "{\"name\":"; jstr(o, p.name);
        o << ",\"size\":"; jstr(o, p.size_text);
        o << ",\"params\":"; jmap(o, p.param_values);
        o << ",\"param_spread\":"; jmap_nonempty(o, p.param_spread);
        o << ",\"ic\":"; jmap(o, p.initial_conditions);
        o << ",\"ic_spread\":"; jmap_nonempty(o, p.ic_spread);
        if (p.spike_source) {
            o << ",\"spike_source\":true,\"rate\":"; jstr(o, p.rate_text);
            o << ",\"source_from\":"; jstr(o, p.source_from_text);
            o << ",\"source_to\":"; jstr(o, p.source_to_text);
        }
        o << "}";
    }
    o << "],\"projections\":[";
    for (size_t i = 0; i < c.projections.size(); ++i) {
        const NeuralProjection& p = c.projections[i];
        if (i) o << ",";
        o << "{\"from\":" << p.from << ",\"to\":" << p.to << ",\"rule\":" << (int)p.rule;
        o << ",\"p\":"; jstr(o, p.p_text);
        o << ",\"degree\":"; jstr(o, p.degree_text);
        o << ",\"k\":"; jstr(o, p.k_text);
        o << ",\"width\":"; jstr(o, p.width_text);
        o << ",\"rewire\":"; jstr(o, p.rewire_text);
        o << ",\"periodic\":" << (p.periodic ? "true" : "false");
        o << ",\"directed\":" << (p.directed ? "true" : "false");
        o << ",\"weight\":"; jstr(o, p.weight_text);
        o << ",\"weight_spread\":"; jstr(o, p.weight_spread_text);
        o << ",\"norm\":" << (int)p.norm;
        o << ",\"norm_expected\":" << (p.norm_expected ? "true" : "false");
        o << ",\"law\":" << p.law;
        // Импульсные поля пишутся только у импульсных сессий: старые файлы не меняются.
        if (c.schedule == NeuralSchedule::Spike) { o << ",\"receptor\":" << p.receptor << ",\"delay\":"; jstr(o, p.delay_text); }
        if (p.plastic >= 0) o << ",\"plastic\":" << p.plastic;
        if (p.device) o << ",\"device\":true,\"one_element\":" << (p.one_element ? "true" : "false") << ",\"device_var\":" << p.device_var;
        o << "}";
    }
    o << "]";
    if (c.schedule == NeuralSchedule::Spike) {
        o << ",\"threshold_var\":" << c.threshold_var;
        o << ",\"threshold\":"; jstr(o, c.threshold_text);
        o << ",\"reset\":"; strings(c.reset_text);
        o << ",\"refractory\":"; jstr(o, c.refractory_text);
        o << ",\"receptors\":[";
        for (size_t i = 0; i < c.receptors.size(); ++i) {
            const NeuralReceptor& q = c.receptors[i];
            if (i) o << ",";
            o << "{\"name\":"; jstr(o, q.name);
            o << ",\"tau\":"; jstr(o, q.tau_text);
            o << ",\"target\":" << q.target;
            o << ",\"conductance\":" << (q.conductance ? "true" : "false");
            o << ",\"reversal\":"; jstr(o, q.reversal_text);
            o << ",\"gain\":"; jstr(o, q.gain_text);
            o << "}";
        }
        o << "]";
    }
    // Пишутся только при наличии: старые файлы не меняются.
    if (!c.plastic_rules.empty()) {
        o << ",\"plastic_rules\":[";
        for (size_t i = 0; i < c.plastic_rules.size(); ++i) {
            const NeuralPlasticRule& q = c.plastic_rules[i];
            if (i) o << ",";
            o << "{\"name\":"; jstr(o, q.name);
            o << ",\"a_plus\":"; jstr(o, q.a_plus_text);
            o << ",\"a_minus\":"; jstr(o, q.a_minus_text);
            o << ",\"tau_plus\":"; jstr(o, q.tau_plus_text);
            o << ",\"tau_minus\":"; jstr(o, q.tau_minus_text);
            o << ",\"w_min\":"; jstr(o, q.w_min_text);
            o << ",\"w_max\":"; jstr(o, q.w_max_text);
            o << ",\"mu_plus\":"; jstr(o, q.mu_plus_text);
            o << ",\"mu_minus\":"; jstr(o, q.mu_minus_text);
            o << ",\"nearest\":" << (q.nearest ? "true" : "false") << "}";
        }
        o << "]";
    }
    if (!c.device.rhs_text.empty()) {
        o << ",\"device_state\":"; jstr(o, c.device.state_name);
        o << ",\"device_rhs\":"; jstr(o, c.device.rhs_text);
        o << ",\"device_current\":"; jstr(o, c.device.current_text);
        o << ",\"device_params\":"; jstr(o, c.device.params_text);
        o << ",\"device_initial\":"; jstr(o, c.device.initial_text);
        o << ",\"device_min\":"; jstr(o, c.device.min_text);
        o << ",\"device_max\":"; jstr(o, c.device.max_text);
    }
    if (c.task.enabled) {
        o << ",\"task\":{\"source\":" << c.task.source << ",\"targets\":[";
        for (size_t i = 0; i < c.task.targets.size(); ++i) o << (i ? "," : "") << c.task.targets[i];
        o << "],\"horizon\":"; jstr(o, c.task.horizon_text);
        o << ",\"washout\":"; jstr(o, c.task.washout_text);
        o << ",\"gap\":"; jstr(o, c.task.gap_text);
        o << ",\"train\":"; jstr(o, c.task.train_text);
        o << ",\"val\":"; jstr(o, c.task.val_text);
        o << ",\"filter\":"; jstr(o, c.task.filter_text);
        o << ",\"features\":"; jstr(o, c.task.features_text);
        o << "}";
    }
    if (c.keep_checkpoint) o << ",\"keep_checkpoint\":true";
    if (!c.checkpoint_path.empty()) { o << ",\"checkpoint_path\":"; jstr(o, c.checkpoint_path); }
    o << "}";
}

static void read_neural_list(JP& p, const std::function<void(const std::string&)>& field) {
    p.expect('[');
    if (p.opt(']')) return;
    while (true) {
        p.expect('{');
        if (!p.opt('}')) {
            while (true) {
                std::string k = p.str(); p.expect(':');
                field(k);
                if (p.opt(',')) continue;
                p.expect('}'); break;
            }
        }
        field("");
        if (p.opt(',')) continue;
        p.expect(']'); break;
    }
}

static bool read_neural_field(JP& p, NeuralConfig& c, const std::string& key) {
    if      (key == "label")            c.label = p.str();
    else if (key == "schedule")         c.schedule = (NeuralSchedule)std::stoi(p.str_or_num());
    else if (key == "scheme")           c.scheme = p.str();
    else if (key == "symmetry_s")       c.symmetry_s = p.str();
    else if (key == "cd_fp_iters")      c.cd_fp_iters = p.str();
    else if (key == "newton_tol")       c.newton.tol = p.str();
    else if (key == "newton_max_iters") c.newton.max_iters = p.str();
    else if (key == "newton_full")      c.newton.full = p.boolean();
    else if (key == "h_text")           c.h_text = p.str();
    else if (key == "t_max_text")       c.t_max_text = p.str();
    else if (key == "transient_text")   c.transient_text = p.str();
    else if (key == "pre_scaller_text") c.pre_scaller_text = p.str();
    else if (key == "max_value_text")   c.max_value_text = p.str();
    else if (key == "coupling_text")    c.coupling_text = p.str();
    else if (key == "seed_text")        c.seed_text = p.str();
    else if (key == "observed_per_pop_text") c.observed_per_pop_text = p.str();
    else if (key == "order_var")        c.order_var = std::stoi(p.str_or_num());
    else if (key == "order_threshold")  c.order_threshold_text = p.str();
    else if (key == "sweep_on")         c.sweep.enabled = p.boolean();
    else if (key == "sweep_kind")       c.sweep.kind = (NeuralSweepKind)std::stoi(p.str_or_num());
    else if (key == "sweep_pop")        c.sweep.population = std::stoi(p.str_or_num());
    else if (key == "sweep_param")      c.sweep.param = p.str();
    else if (key == "sweep_lo")         c.sweep.lo_text = p.str();
    else if (key == "sweep_hi")         c.sweep.hi_text = p.str();
    else if (key == "sweep_points")     c.sweep.points_text = p.str();
    else if (key == "sweep_log")        c.sweep.log = p.boolean();
    else if (key == "sweep2_on")        c.sweep2.enabled = p.boolean();
    else if (key == "sweep2_kind")      c.sweep2.kind = (NeuralSweepKind)std::stoi(p.str_or_num());
    else if (key == "sweep2_pop")       c.sweep2.population = std::stoi(p.str_or_num());
    else if (key == "sweep2_param")     c.sweep2.param = p.str();
    else if (key == "sweep2_lo")        c.sweep2.lo_text = p.str();
    else if (key == "sweep2_hi")        c.sweep2.hi_text = p.str();
    else if (key == "sweep2_points")    c.sweep2.points_text = p.str();
    else if (key == "sweep2_log")       c.sweep2.log = p.boolean();
    else if (key == "lyap_on")          c.lyap_on = p.boolean();
    else if (key == "lyap_eps")         c.lyap_eps_text = p.str();
    else if (key == "lyap_interval")    c.lyap_interval_text = p.str();
    else if (key == "lyap_settle")      c.lyap_settle_text = p.str();
    else if (key == "csv_save")         c.csv_save_enabled = p.boolean();
    else if (key == "csv_path")         c.csv_output_path = p.str();
    else if (key == "obs_vars_off") {
        p.expect('[');
        if (!p.opt(']')) while (true) { c.obs_vars_off.push_back(std::stoi(p.str_or_num())); if (p.opt(',')) continue; p.expect(']'); break; }
    }
    else if (key == "output_limit")     c.output_limit_text = p.str();
    else if (key == "input_var")        c.input_var = std::stoi(p.str_or_num());
    else if (key == "input_hold")       c.input_hold_text = p.str();
    else if (key == "protocol")         c.protocol = (NeuralProtocol)std::stoi(p.str_or_num());
    else if (key == "trials")           c.trials_text = p.str();
    else if (key == "rest")             c.rest_text = p.str();
    else if (key == "stimulus")         c.stimulus_text = p.str();
    else if (key == "reset_state")      c.reset_state = p.boolean();
    else if (key == "sources") {
        c.sources.clear();
        NeuralSource q;
        const auto strings = [&p](std::vector<std::string>& v) {
            p.expect('[');
            if (!p.opt(']')) while (true) { v.push_back(p.str()); if (p.opt(',')) continue; p.expect(']'); break; }
        };
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.sources.push_back(q); q = NeuralSource{}; }
            else if (k == "name") q.name = p.str();
            else if (k == "kind") q.kind = (NeuralSourceKind)std::stoi(p.str_or_num());
            else if (k == "amplitude") q.amplitude_text = p.str();
            else if (k == "frequency") q.frequency_text = p.str();
            else if (k == "phase") q.phase_text = p.str();
            else if (k == "offset") q.offset_text = p.str();
            else if (k == "system") q.system_name = p.str();
            else if (k == "vars") strings(q.vars);
            else if (k == "params") strings(q.params);
            else if (k == "rhs") strings(q.rhs);
            else if (k == "latex") q.latex = p.boolean();
            else if (k == "ic") q.initial_conditions = p.map_ss();
            else if (k == "values") q.param_values = p.map_ss();
            else if (k == "variable") q.variable = std::stoi(p.str_or_num());
            else if (k == "step") q.step_text = p.str();
            else if (k == "time_scale") q.time_scale_text = p.str();
            else if (k == "skip") q.skip_text = p.str();
            else if (k == "normalize") q.normalize = p.boolean();
            else p.skip_value();
        });
    }
    else if (key == "links") {
        c.links.clear();
        NeuralLink q;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.links.push_back(q); q = NeuralLink{}; }
            else if (k == "source") q.source = std::stoi(p.str_or_num());
            else if (k == "population") q.population = std::stoi(p.str_or_num());
            else if (k == "gain") q.gain_text = p.str();
            else if (k == "gain_spread") q.gain_spread_text = p.str();
            else if (k == "fraction") q.fraction_text = p.str();
            else p.skip_value();
        });
    }
    else if (key == "graph_recipe")     c.graph_recipe = p.str();
    else if (key == "graph_hash")       c.graph_hash = p.str();
    else if (key == "model_text")       c.model_text = p.str();
    else if (key == "laws") {
        c.laws.clear();
        NetCouplingLaw l;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.laws.push_back(l); l = NetCouplingLaw{}; }
            else if (k == "name") l.name = p.str();
            else if (k == "expr") {
                p.expect('[');
                if (!p.opt(']')) while (true) { l.expr.push_back(p.str()); if (p.opt(',')) continue; p.expect(']'); break; }
            }
            else p.skip_value();
        });
    }
    else if (key == "populations") {
        c.populations.clear();
        NeuralPopulation q;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.populations.push_back(q); q = NeuralPopulation{}; }
            else if (k == "name") q.name = p.str();
            else if (k == "size") q.size_text = p.str();
            else if (k == "params") q.param_values = p.map_ss();
            else if (k == "param_spread") q.param_spread = p.map_ss();
            else if (k == "ic") q.initial_conditions = p.map_ss();
            else if (k == "ic_spread") q.ic_spread = p.map_ss();
            else if (k == "spike_source") q.spike_source = p.boolean();
            else if (k == "rate") q.rate_text = p.str();
            else if (k == "source_from") q.source_from_text = p.str();
            else if (k == "source_to") q.source_to_text = p.str();
            else p.skip_value();
        });
    }
    else if (key == "projections") {
        c.projections.clear();
        NeuralProjection q;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.projections.push_back(q); q = NeuralProjection{}; }
            else if (k == "from") q.from = std::stoi(p.str_or_num());
            else if (k == "to") q.to = std::stoi(p.str_or_num());
            else if (k == "rule") q.rule = (NeuralRule)std::stoi(p.str_or_num());
            else if (k == "p") q.p_text = p.str();
            else if (k == "degree") q.degree_text = p.str();
            else if (k == "k") q.k_text = p.str();
            else if (k == "width") q.width_text = p.str();
            else if (k == "rewire") q.rewire_text = p.str();
            else if (k == "periodic") q.periodic = p.boolean();
            else if (k == "directed") q.directed = p.boolean();
            else if (k == "weight") q.weight_text = p.str();
            else if (k == "weight_spread") q.weight_spread_text = p.str();
            else if (k == "norm") q.norm = (NeuralNorm)std::stoi(p.str_or_num());
            else if (k == "norm_expected") q.norm_expected = p.boolean();
            else if (k == "law") q.law = std::stoi(p.str_or_num());
            else if (k == "receptor") q.receptor = std::stoi(p.str_or_num());
            else if (k == "delay") q.delay_text = p.str();
            else if (k == "plastic") q.plastic = std::stoi(p.str_or_num());
            else if (k == "device") q.device = p.boolean();
            else if (k == "one_element") q.one_element = p.boolean();
            else if (k == "device_var") q.device_var = std::stoi(p.str_or_num());
            else p.skip_value();
        });
    }
    else if (key == "plastic_rules") {
        c.plastic_rules.clear();
        NeuralPlasticRule q;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.plastic_rules.push_back(q); q = NeuralPlasticRule{}; }
            else if (k == "name") q.name = p.str();
            else if (k == "a_plus") q.a_plus_text = p.str();
            else if (k == "a_minus") q.a_minus_text = p.str();
            else if (k == "tau_plus") q.tau_plus_text = p.str();
            else if (k == "tau_minus") q.tau_minus_text = p.str();
            else if (k == "w_min") q.w_min_text = p.str();
            else if (k == "w_max") q.w_max_text = p.str();
            else if (k == "mu_plus") q.mu_plus_text = p.str();
            else if (k == "mu_minus") q.mu_minus_text = p.str();
            else if (k == "nearest") q.nearest = p.boolean();
            else p.skip_value();
        });
    }
    else if (key == "task") {
        c.task.enabled = true;
        p.expect('{');
        if (!p.opt('}')) {
            while (true) {
                const std::string k = p.str();
                p.expect(':');
                if (k == "source") c.task.source = std::stoi(p.str_or_num());
                else if (k == "targets") {
                    c.task.targets.clear();
                    p.expect('[');
                    if (!p.opt(']')) while (true) { c.task.targets.push_back(std::stoi(p.str_or_num())); if (p.opt(',')) continue; p.expect(']'); break; }
                }
                else if (k == "horizon") c.task.horizon_text = p.str();
                else if (k == "washout") c.task.washout_text = p.str();
                else if (k == "gap") c.task.gap_text = p.str();
                else if (k == "train") c.task.train_text = p.str();
                else if (k == "val") c.task.val_text = p.str();
                else if (k == "filter") c.task.filter_text = p.str();
                else if (k == "features") c.task.features_text = p.str();
                else p.skip_value();
                if (p.opt(',')) continue;
                p.expect('}');
                break;
            }
        }
    }
    else if (key == "device_state")     c.device.state_name = p.str();
    else if (key == "device_rhs")       c.device.rhs_text = p.str();
    else if (key == "device_current")   c.device.current_text = p.str();
    else if (key == "device_params")    c.device.params_text = p.str();
    else if (key == "device_initial")   c.device.initial_text = p.str();
    else if (key == "device_min")       c.device.min_text = p.str();
    else if (key == "device_max")       c.device.max_text = p.str();
    else if (key == "keep_checkpoint")  c.keep_checkpoint = p.boolean();
    else if (key == "checkpoint_path")  c.checkpoint_path = p.str();
    else if (key == "threshold_var")    c.threshold_var = std::stoi(p.str_or_num());
    else if (key == "threshold")        c.threshold_text = p.str();
    else if (key == "refractory")       c.refractory_text = p.str();
    else if (key == "reset") {
        c.reset_text.clear();
        p.expect('[');
        if (!p.opt(']')) while (true) { c.reset_text.push_back(p.str()); if (p.opt(',')) continue; p.expect(']'); break; }
    }
    else if (key == "receptors") {
        c.receptors.clear();
        NeuralReceptor q;
        read_neural_list(p, [&](const std::string& k) {
            if (k.empty()) { c.receptors.push_back(q); q = NeuralReceptor{}; }
            else if (k == "name") q.name = p.str();
            else if (k == "tau") q.tau_text = p.str();
            else if (k == "target") q.target = std::stoi(p.str_or_num());
            else if (k == "conductance") q.conductance = p.boolean();
            else if (k == "reversal") q.reversal_text = p.str();
            else if (k == "gain") q.gain_text = p.str();
            else p.skip_value();
        });
    }
    else return false;
    return true;
}

std::string session_to_json_neural(const NeuralState& s) {
    std::ostringstream o;
    o << "{\n  \"active_config_index\":" << s.active_config_index << ",\n  \"configs\":[";
    for (size_t i = 0; i < s.configs.size(); ++i) {
        if (i) o << ",";
        o << "\n    ";
        write_neural_config(o, s.configs[i]);
    }
    if (!s.configs.empty()) o << "\n  ";
    o << "]\n}\n";
    return o.str();
}

bool session_from_json_neural(const std::string& json, NeuralState& s) {
    try {
        JP p(json);
        p.expect('{');
        if (p.opt('}')) return true;
        std::vector<NeuralConfig> configs;
        int active = 0;
        while (true) {
            std::string key = p.str();
            p.expect(':');
            if (key == "configs") {
                p.expect('[');
                if (!p.opt(']')) {
                    while (true) {
                        p.expect('{');
                        NeuralConfig c;
                        if (!p.opt('}')) {
                            while (true) {
                                std::string k2 = p.str(); p.expect(':');
                                if (!read_neural_field(p, c, k2)) p.skip_value();
                                if (p.opt(',')) continue;
                                p.expect('}'); break;
                            }
                        }
                        if (c.laws.empty()) c.laws.push_back(net_default_law(s.vars));
                        configs.push_back(std::move(c));
                        if (p.opt(',')) continue;
                        p.expect(']'); break;
                    }
                }
            }
            else if (key == "active_config_index") active = std::stoi(p.str_or_num());
            else p.skip_value();
            if (p.opt(',')) continue;
            p.expect('}'); break;
        }
        if (!configs.empty()) s.configs = std::move(configs);
        s.active_config_index = (active >= 0 && active < (int)s.configs.size()) ? active : 0;
        s.running_config_index = -1;
        return true;
    } catch (...) {
        return false;
    }
}
