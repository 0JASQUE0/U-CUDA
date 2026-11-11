#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/platform/platform_paths.h"
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::string exe_dir() { return platform::executable_dir(); }

namespace {
using Event = std::pair<long long, int>;
constexpr int variables = 3;
constexpr int parameters = 5; // rest potential, membrane tau, DC drive, threshold, reset
struct Edge { int src, dst, receptor, delay; double weight; };
struct Case {
    std::string name;
    NeuralRequest request;
    std::vector<Edge> edges;
    int steps = 0;
};
struct Reference {
    std::vector<Event> events;
    std::vector<double> traces; // includes the initial point, then each completed step
    std::vector<int> counts;
};
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}
void expect_close(double actual, double expected, double tolerance, const std::string& where) {
    require(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance,
            where + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}
Case base_case(const std::string& name, int nodes, int steps) {
    Case c;
    c.name = name; c.steps = steps;
    auto& q = c.request;
    q.schedule = NeuralSchedule::Spike;
    q.amountOfX = variables; q.amountOfValues = parameters; q.n_nodes = nodes;
    q.rhs_body = "F[0] = (a[0] - X[0]) / a[1] + a[2] + X[1] + X[2]; F[1] = 0; F[2] = 0;";
    q.spike.spike_body = "if (X[0] >= a[3]) { X[0] = a[4]; spiked = true; }";
    q.h = 0.125; q.t_max = steps * q.h; q.pre_scaller = 1;
    q.keep_final_state = true;
    q.spike.receptors = 2; q.spike.tau = {5, 10};
    q.spike.refractory_steps.assign(nodes, 0);
    q.spike.log_capacity = nodes * steps;
    q.observe.nodes.resize(nodes);
    std::iota(q.observe.nodes.begin(), q.observe.nodes.end(), 0);
    q.observe.vars = {0, 1, 2};
    q.edge_start.assign(nodes + 1, 0);
    for (int i = 0; i < nodes; ++i) {
        q.values.insert(q.values.end(), {-60, 20, 0, -50, -60});
        q.initial_conditions.insert(q.initial_conditions.end(), {-60, 0, 0});
    }
    return c;
}
void graph(Case& c) {
    auto& q = c.request;
    q.edge_src.clear(); q.edge_w.clear(); q.edge_law.clear(); q.spike.edge_delay.clear();
    for (int dst = 0; dst < q.n_nodes; ++dst) {
        q.edge_start[dst] = (int)q.edge_src.size();
        for (const auto& e : c.edges) if (e.dst == dst) {
            q.edge_src.push_back(e.src); q.edge_w.push_back(e.weight);
            q.edge_law.push_back(e.receptor); q.spike.edge_delay.push_back(e.delay);
        }
    }
    q.edge_start.back() = (int)q.edge_src.size();
}

// Closed-form LIF transition for exponentially decaying input currents.
// No generated RHS, RK stages, device ring, fixed-point summation or production code is reused.
// Events are placed in an absolute-time delivery table. This is deliberately a different algorithm.
Reference analytic(const Case& c) {
    const auto& q = c.request;
    const int n = q.n_nodes;
    std::vector<double> state = q.initial_conditions;
    std::vector<int> held(n, 0);
    std::vector<double> deliveries((size_t)(c.steps + 1) * n * 2, 0);
    Reference r;
    r.counts.assign(n, 0); r.traces = state;
    for (int step = 0; step < c.steps; ++step) {
        for (int node = 0; node < n; ++node) {
            double* x = state.data() + node * variables;
            const double* a = q.values.data() + node * parameters;
            for (int rec = 0; rec < 2; ++rec)
                x[rec + 1] += deliveries[((size_t)step * n + node) * 2 + rec];
            bool fired = false;
            if (held[node] > 0) --held[node];
            else {
                const double em = std::exp(-q.h / a[1]);
                double v = a[0] + (x[0] - a[0]) * em - a[2] * a[1] * std::expm1(-q.h / a[1]);
                for (int rec = 0; rec < 2; ++rec) {
                    const double delta = 1 / a[1] - 1 / q.spike.tau[rec];
                    const double integral = delta == 0 ? q.h : std::expm1(q.h * delta) / delta;
                    v += x[rec + 1] * em * integral;
                }
                fired = v >= a[3];
                x[0] = fired ? a[4] : v;
                if (fired) held[node] = q.spike.refractory_steps[node];
            }
            for (int rec = 0; rec < 2; ++rec) x[rec + 1] *= std::exp(-q.h / q.spike.tau[rec]);
            if (fired) {
                r.events.emplace_back(step, node); ++r.counts[node];
                for (const auto& e : c.edges) if (e.src == node && step + e.delay < c.steps)
                    deliveries[((size_t)(step + e.delay) * n + e.dst) * 2 + e.receptor] += e.weight;
            }
        }
        r.traces.insert(r.traces.end(), state.begin(), state.end());
    }
    return r;
}
double point(const Reference& r, const Case& c, int completed, int node, int var) {
    return r.traces[((size_t)completed * c.request.n_nodes + node) * variables + var];
}
std::vector<Case> cases() {
    // Exact equality at the END of a step: >= must fire; two full held steps follow.
    auto boundary = base_case("threshold_reset_refractory", 3, 9);
    boundary.request.values = {0, 20, 0, 0, 0, 0, 20, 0, 0, -1, -1, 20, 0, 0, -1};
    boundary.request.initial_conditions = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    boundary.request.spike.refractory_steps = {2, 2, 0};
    // Node 2 starts AT threshold but drops below it during the step: no initial-time spike.

    auto pulses = base_case("simultaneous_signed_deliveries", 6, 24);
    for (int i = 0; i < 2; ++i) {
        const double p[] = {0, 20, 0, 0, -1};
        std::copy(p, p + parameters, pulses.request.values.begin() + i * parameters);
        pulses.request.initial_conditions[i * variables] = 0;
        pulses.request.spike.refractory_steps[i] = 100;
    }
    // Two arrivals to one channel, a negative current, exact signed cancellation,
    // minimum delay 1, a second delayed pulse, and an arrival while the target is refractory.
    pulses.edges = {{0, 2, 0, 1, .5}, {1, 2, 0, 1, 1}, {1, 2, 1, 1, -.5},
                    {0, 3, 0, 2, .5}, {1, 3, 0, 2, -.5}, {0, 2, 0, 5, 2},
                    {0, 1, 0, 2, 2}, {0, 4, 0, 3, 1}, {0, 5, 1, 4, -1}};

    auto recurrent = base_case("recurrent_cuba", 8, 512);
    for (int i = 0; i < 8; ++i) {
        recurrent.request.values[i * parameters + 2] = .75 + .0625 * (i % 3);
        recurrent.request.initial_conditions[i * variables] = -50.125 - i;
        recurrent.request.spike.refractory_steps[i] = 2 + i % 3;
        recurrent.edges.push_back({i, (i + 1) % 8, 0, 2, 2});
        recurrent.edges.push_back({i, (i + 3) % 8, 1, 3, -1});
        recurrent.edges.push_back({i, (i + 5) % 8, 0, 5, .5});
    }
    // Exercise nonzero initial synapses and the equal membrane/synaptic tau limit.
    recurrent.request.initial_conditions[1] = .5;
    recurrent.request.initial_conditions[2] = -.25;
    recurrent.request.values[7 * parameters + 1] = 5;
    std::vector<Case> result = {boundary, pulses, recurrent};
    for (auto& c : result) graph(c);
    return result;
}
void oracle_checks(const std::vector<Case>& fixtures) {
    const auto& b = fixtures[0];
    const auto r = analytic(b);
    require(r.events == std::vector<Event>{{0, 0}, {0, 1}, {3, 0}, {6, 0}}, "pinned boundary events");
    expect_close(point(r, b, 3, 1, 0), -1, 0, "reset held for two complete steps");
    expect_close(point(r, b, 4, 1, 0), -std::exp(-b.request.h / 20), 1e-14, "release after refractory period");
    const auto& p = fixtures[1];
    const auto s = analytic(p);
    require(s.events == std::vector<Event>{{0, 0}, {0, 1}}, "pinned pulse events");
    expect_close(point(s, p, 1, 2, 1), 0, 0, "no zero-delay delivery");
    expect_close(point(s, p, 2, 2, 1), 1.5 * std::exp(-p.request.h / 5), 1e-14, "simultaneous excitatory arrivals");
    expect_close(point(s, p, 2, 2, 2), -.5 * std::exp(-p.request.h / 10), 1e-14, "inhibitory arrival");
    expect_close(point(s, p, 3, 3, 1), 0, 0, "signed cancellation");
    expect_close(point(s, p, 3, 1, 0), -1, 0, "arrival must not release refractory membrane");
    expect_close(point(s, p, 3, 1, 1), 2 * std::exp(-p.request.h / 5), 1e-14, "synapse evolves during refractory period");
    const auto t = analytic(fixtures[2]);
    require(t.events.size() >= 16, "recurrent fixture must produce repeated spikes");
    auto shifted = p;
    for (auto& e : shifted.edges) ++e.delay;
    require(analytic(shifted).traces != s.traces, "oracle must distinguish a one-step delay error");
    auto shortened = b;
    shortened.request.spike.refractory_steps[0] = 1;
    require(analytic(shortened).events != r.events, "oracle must distinguish a refractory off-by-one");
}
void compare(const Case& c, const Reference& ref, const NeuralResult& actual) {
    const auto prefix = c.name + ": ";
    require(actual.ok, prefix + actual.error);
    require(!actual.cancelled && !actual.diverged && actual.n_diverged == 0, prefix + "incomplete dynamics");
    require(actual.steps_done == c.steps, prefix + "step count");
    require(actual.spikes_dropped == 0, prefix + "truncated spike log");
    require(actual.spike_node.size() == actual.spike_step.size(), prefix + "event array shape");
    std::vector<Event> events;
    for (size_t i = 0; i < actual.spike_step.size(); ++i) events.emplace_back(actual.spike_step[i], actual.spike_node[i]);
    require(events == ref.events, prefix + "(step, neuron) events differ from analytic reference");
    require(actual.spike_count == ref.counts, prefix + "per-neuron counts");
    const bool recording = !c.request.observe.nodes.empty();
    require(recording ? (actual.n_points == c.steps + 1 && actual.traces.size() == ref.traces.size())
                      : (actual.n_points == 0 && actual.traces.empty()), prefix + "trace shape");
    // The oracle integrates exactly; the engine uses RK4 for V and exact exponential currents.
    // Fixed h=0.125 ms, tau>=5 ms, 64 ms duration: bound allows truncation, not a whole step shift.
    for (size_t i = 0; recording && i < ref.traces.size(); ++i)
        expect_close(actual.traces[i], ref.traces[i], i % variables == 0 ? 2e-6 : 2e-11,
             prefix + "trace scalar " + std::to_string(i));
    const size_t width = (size_t)c.request.n_nodes * variables;
    require(actual.final_state.size() == width, prefix + "final state shape");
    for (size_t i = 0; i < width; ++i)
        expect_close(actual.final_state[i], ref.traces[ref.traces.size() - width + i], i % variables == 0 ? 2e-6 : 2e-11,
             prefix + "final state scalar " + std::to_string(i));
}
}

int main(int argc, char** argv) {
    try {
        const bool cpu = argc == 2 && std::string(argv[1]) == "--oracle-only";
        require(argc == 1 || cpu, "usage: neural_cuba_test [--oracle-only]");
        const auto fixtures = cases();
        oracle_checks(fixtures);
        if (cpu) { std::cout << "CUBA analytic oracle: pinned boundary and delivery checks passed\n"; return 0; }
        if (cuInit(0) != CUDA_SUCCESS) { std::cout << "No CUDA device; CUBA GPU test skipped\n"; return 77; }
        ParametricEngine engine;
        for (const auto& c : fixtures) {
            const auto ref = analytic(c);
            const auto first = engine.run_neural(c.request);
            compare(c, ref, first);
            // Different blocks and host chunks must not change the trajectory or log.
            auto alternate = c.request;
            alternate.block_size = 2; alternate.steps_per_sync = 7; alternate.spike.window = 1;
            const auto second = engine.run_neural(alternate);
            compare(c, ref, second);
            require(same_bits(first.traces, second.traces) && same_bits(first.final_state, second.final_state)
                    && first.spike_step == second.spike_step && first.spike_node == second.spike_node,
                    c.name + ": chunk/window/block invariance");
            // Per-step recording limits the launch window to one. Disable recording to
            // exercise actual multi-step launches (minimum recurrent delay is two).
            auto batched = c;
            batched.request.observe = {};
            const auto third = engine.run_neural(batched.request);
            compare(batched, ref, third);
            require(same_bits(first.final_state, third.final_state) && first.spike_step == third.spike_step
                    && first.spike_node == third.spike_node, c.name + ": recording/window invariance");
            if (c.name == "recurrent_cuba") require(third.launches < first.launches, "multi-step window was not exercised");
            std::cout << c.name << ": " << first.spike_step.size() << " spikes; analytic and chunk checks passed\n";
        }
        auto invalid = fixtures[1].request;
        invalid.spike.edge_delay[0] = 0;
        const auto rejected = engine.run_neural(invalid);
        require(!rejected.ok && rejected.error.find("delay") != std::string::npos, "zero-delay edge must be rejected");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CUBA regression failed: " << e.what() << '\n';
        return 1;
    }
}
