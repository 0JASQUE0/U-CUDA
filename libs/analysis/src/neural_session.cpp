#include "ucuda/analysis/neural_session.h"
#include "ucuda/domain/num_parse.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <array>
#include "ucuda/numerics/neural_readout.h"

namespace {

void log_neural_run(const char* label, bool ok, double secs) {
    std::printf("[run] %s %s in %.2fs\n", ok ? "Done" : "Cancelled", (label && *label) ? label : "(unnamed)", secs);
    std::fflush(stdout);
}

// Запас от int в CSR и от памяти хоста: полный граф на 100 000 узлов отказывает здесь, а не в malloc.
constexpr long long kMaxArcs = 500000000LL;

double num_or(const std::string& s, double fallback) {
    return s.empty() ? fallback : parse_num(s, fallback);
}
int int_or(const std::string& s, int fallback) {
    const double v = num_or(s, (double)fallback);
    if (!std::isfinite(v) || std::fabs(v) > 2.0e9) throw std::runtime_error("number out of range: " + s);
    return (int)std::llround(v);
}

uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// SplitMix64: один поток на (seed, a, b), порядок обхода на числа не влияет. Начало проходит через mix:
// при линейном по a, b начале соседние узлы получали одну последовательность со сдвигом.
struct Stream {
    uint64_t s;
    Stream(uint64_t seed, uint64_t a, uint64_t b) {
        s = mix64(seed + 0x9E3779B97F4A7C15ULL);
        s = mix64(s ^ (a + 1) * 0xC2B2AE3D27D4EB4FULL);
        s = mix64(s ^ (b + 1) * 0x165667B19E3779F9ULL);
    }
    uint64_t next() { return mix64(s += 0x9E3779B97F4A7C15ULL); }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
    int below(int n) { return (int)(uni() * (double)n); }
};

// Шаги времени, которое обязано быть кратно h: иначе граница входа или попытки легла бы внутрь шага.
long long steps_of(double t, double h, const char* what) {
    const double n = t / h;
    if (!std::isfinite(n) || n < 0.0 || std::fabs(n - std::round(n)) > 1e-9 * (std::max)(1.0, std::fabs(n)))
        throw std::runtime_error(std::string(what) + " must be a multiple of h");
    return std::llround(n);
}

// Значение источника на одном отсчёте; у системы это заранее посчитанный ряд.
struct SourceSeries {
    const NeuralSource* src = nullptr;
    int index = 0;
    uint64_t seed = 0;
    double a = 1.0, f = 0.0, phase = 0.0, offset = 0.0, h = 0.0;
    long long half_steps = 0, shift = 0;   // меандр: фронты через half_steps шагов, 0 = без фронтов
    std::vector<double> series;    // система: по одному значению на активный отсчёт
    // tau в шагах, sub = 0, 1, 2 это t, t + h/2, t + h, при отсчёте на шаг всегда 0. Ступенчатые источники от sub не зависят.
    double value(long long tau, int sub, bool three, long long sample, size_t active) const {
        const double two_pi = 6.283185307179586476925286766559;
        switch (src->kind) {
        case NeuralSourceKind::Constant: return offset + a;
        case NeuralSourceKind::Sine:     return offset + a * std::sin(two_pi * f * ((double)tau + 0.5 * sub) * h + phase);
        case NeuralSourceKind::Square: {
            if (half_steps == 0) return offset + (std::sin(phase) >= 0.0 ? a : -a);
            const long long k = tau + shift, q = (k >= 0 ? k : k - half_steps + 1) / half_steps;
            return offset + (q % 2 == 0 ? a : -a);
        }
        case NeuralSourceKind::Noise: {
            Stream g(seed ^ 0x4E4F4953ULL, (uint64_t)index, (uint64_t)sample);
            const double u = 1.0 - g.uni(), v = g.uni();
            return offset + a * std::sqrt(-2.0 * std::log(u)) * std::cos(two_pi * v);
        }
        case NeuralSourceKind::System:   return offset + a * series[three ? 2 * active + (size_t)sub : active];
        case NeuralSourceKind::Bits: {
            Stream g(seed ^ 0x42495453ULL, (uint64_t)index, (uint64_t)sample);
            return offset + (g.uni() < 0.5 ? a : -a);
        }
        }
        return 0.0;
    }
};

// RK4 снимка системы: active отсчётов через dt времени источника, после skip.
std::vector<double> source_system_series(const NeuralSource& s, size_t active, double dt, size_t per_sample = 1, double* stats = nullptr) {
    if (s.rhs.empty()) throw std::runtime_error("input source '" + s.name + "' has no system: pick one from the library");
    if (s.variable < 0 || s.variable >= (int)s.vars.size()) throw std::runtime_error("input source '" + s.name + "': bad variable");
    System sys;
    sys.vars = s.vars; sys.params = s.params; sys.rhs = s.rhs;
    // Снимки до появления флага: по самому тексту.
    sys.latex = s.latex;
    for (const std::string& e : s.rhs) if (e.find('\\') != std::string::npos) sys.latex = true;
    const SystemEvaluator ev(sys);
    const int n = ev.dim();
    std::vector<double> a(s.params.size() + 1 + kSchemeTailSlots, 0.0), x((size_t)n), k1(x), k2(x), k3(x), k4(x), y(x);
    for (size_t j = 0; j < s.params.size(); ++j) {
        auto it = s.param_values.find(s.params[j]);
        a[j + 1] = it == s.param_values.end() ? 0.0 : num_or(it->second, 0.0);
    }
    for (int i = 0; i < n; ++i) {
        auto it = s.initial_conditions.find(s.vars[(size_t)i]);
        x[(size_t)i] = it == s.initial_conditions.end() ? 0.0 : num_or(it->second, 0.0);
    }
    const double step = num_or(s.step_text, 0.01), skip = num_or(s.skip_text, 0.0);
    if (!(step > 0.0) || !(dt > 0.0)) throw std::runtime_error("input source '" + s.name + "': bad step");
    const long long sub = (long long)std::ceil(dt / step - 1e-9), first = (long long)std::ceil(skip / step - 1e-9);
    const double h = dt / (double)(std::max)(1LL, sub), h0 = skip > 0.0 && first > 0 ? skip / (double)first : 0.0;
    const auto advance = [&](double q) {
        ev.eval(x.data(), a.data(), k1.data());
        for (int i = 0; i < n; ++i) y[(size_t)i] = x[(size_t)i] + 0.5 * q * k1[(size_t)i];
        ev.eval(y.data(), a.data(), k2.data());
        for (int i = 0; i < n; ++i) y[(size_t)i] = x[(size_t)i] + 0.5 * q * k2[(size_t)i];
        ev.eval(y.data(), a.data(), k3.data());
        for (int i = 0; i < n; ++i) y[(size_t)i] = x[(size_t)i] + q * k3[(size_t)i];
        ev.eval(y.data(), a.data(), k4.data());
        for (int i = 0; i < n; ++i) x[(size_t)i] += q / 6.0 * (k1[(size_t)i] + 2.0 * k2[(size_t)i] + 2.0 * k3[(size_t)i] + k4[(size_t)i]);
    };
    for (long long q = 0; q < first; ++q) advance(h0);
    std::vector<double> out(active);
    for (size_t q = 0; q < active; ++q) {
        out[q] = x[(size_t)s.variable];
        if (!std::isfinite(out[q])) throw std::runtime_error("input source '" + s.name + "' diverged");
        for (long long m = 0; m < (std::max)(1LL, sub); ++m) advance(h);
    }
    if (s.normalize && active > 1) {
        double m = s.norm_mean, v = s.norm_sd;
        if (!s.norm_fixed) {
            const size_t used = s.normalize_first > 0 ? (std::min)(active, (size_t)s.normalize_first * per_sample) : active;
            m = 0.0; v = 0.0;
            for (size_t q = 0; q < used; ++q) m += out[q];
            m /= (double)used;
            for (size_t q = 0; q < used; ++q) v += (out[q] - m) * (out[q] - m);
            v = std::sqrt(v / (double)used);
        }
        if (!(v > 0.0)) throw std::runtime_error("input source '" + s.name + "' is constant, it cannot be normalised");
        if (stats) { stats[0] = m; stats[1] = v; }
        for (double& e : out) e = (e - m) / v;
    }
    return out;
}

// Отсчёты входа и какие узлы их получают; шаги сброса протокола.
void fill_input(const NeuralConfig& c, const NeuralGraph& g, uint64_t seed, NeuralRequest& r) {
    const long long skip = steps_of(r.transient, r.h, "the transient");
    const long long total = skip + (long long)std::llround(r.t_max / r.h);
    const bool trials = c.protocol == NeuralProtocol::Trials;
    long long rest = 0, stim = 0, cycle = 0, n_trials = 0;
    if (trials) {
        rest = steps_of(num_or(c.rest_text, 0.0), r.h, "the rest time");
        stim = steps_of(num_or(c.stimulus_text, 0.0), r.h, "the stimulus time");
        cycle = rest + stim;
        n_trials = int_or(c.trials_text, 0);
        if (n_trials < 1 || cycle < 1) throw std::runtime_error("a protocol needs at least one trial of non-zero length");
        if (c.reset_state) for (long long k = 0; k < n_trials; ++k) r.reset_at.push_back(skip + k * cycle);
    }
    if (c.sources.empty() || c.links.empty()) return;

    NeuralInput& in = r.input;
    if (c.input_var < 0 || c.input_var >= r.amountOfX) throw std::runtime_error("the input variable is out of range");
    in.var = c.input_var;
    const double hold_t = num_or(c.input_hold_text, 0.0);
    in.hold_steps = hold_t > 0.0 ? (std::max)(1LL, steps_of(hold_t, r.h, "the input sample time")) : 1;
    in.channels = (int)c.sources.size();
    // Отсчёт на шаг и связь в стадиях: гладкий источник видит время своей стадии, иначе порядок по нему первый.
    // Прогноз подаёт выход строкой на отсчёт: гладкий вход внутри шага сеть сама не посчитает.
    const bool closed = c.task.enabled && c.task.kind == NeuralTaskKind::Forecast;
    const bool three = (c.schedule == NeuralSchedule::Stage || c.schedule == NeuralSchedule::Spike) && in.hold_steps == 1 && !closed;
    in.stage_rows = three ? 3 : 1;
    const long long rows = total / in.hold_steps + 1;
    if ((double)rows * in.stage_rows * in.channels * sizeof(double) > 4.0e9) throw std::runtime_error("the input needs too many samples: raise the sample time");
    in.samples.assign((size_t)rows * in.stage_rows * in.channels, 0.0);

    // Активен ли отсчёт и сколько времени идёт его стимул. Граница отсчёта лежит на шаге, поэтому всё в шагах.
    const auto active_at = [&](long long row, long long& tau, long long& trial) {
        const long long step = row * in.hold_steps;
        tau = step; trial = 0;
        if (!trials) return true;
        if (step < skip) return false;
        trial = (step - skip) / cycle;
        if (trial >= n_trials) return false;
        const long long o = (step - skip) % cycle;
        tau = o - rest;
        return o >= rest;
    };
    size_t n_active = 0;
    for (long long row = 0; row < rows; ++row) { long long tau, trial; n_active += active_at(row, tau, trial); }
    const bool classify = c.task.enabled && c.task.kind == NeuralTaskKind::Classification;
    std::vector<int> class_of((size_t)in.channels, -1), labels;
    if (classify) {
        if (!trials) throw std::runtime_error("classification needs the protocol of separate trials");
        for (size_t k = 0; k < c.task.classes.size(); ++k) {
            const int q = c.task.classes[k];
            if (q < 0 || q >= in.channels) throw std::runtime_error("a class points to a missing input source");
            class_of[(size_t)q] = (int)k;
        }
        labels = neural_class_labels(c);
    }
    const auto make_series = [&](int q) {
        const NeuralSource& s = c.sources[(size_t)q];
        SourceSeries v;
        v.src = &s; v.index = q; v.seed = seed;
        v.a = num_or(s.amplitude_text, 1.0); v.f = num_or(s.frequency_text, 0.0);
        v.phase = num_or(s.phase_text, 0.0); v.offset = num_or(s.offset_text, 0.0);
        v.h = r.h;
        if (s.kind == NeuralSourceKind::Square && v.f != 0.0) {
            // Фронт только на границе шага: внутри шага RK4 должен видеть константу, а знак sin около нуля решает округление.
            const double half = 0.5 / (v.f * r.h), sh = v.phase / (6.283185307179586476925286766559 * v.f * r.h);
            v.half_steps = std::llround(half);
            v.shift = std::llround(sh);
            if (v.half_steps < 1 || std::fabs(half - (double)v.half_steps) > 1e-9 * half)
                throw std::runtime_error("input source '" + s.name + "': half a period is not a multiple of h");
            if (std::fabs(sh - (double)v.shift) > 1e-6)
                throw std::runtime_error("input source '" + s.name + "': the phase puts the edges between steps");
            if (v.half_steps % in.hold_steps != 0 || v.shift % in.hold_steps != 0)
                throw std::runtime_error("input source '" + s.name + "': the edges fall inside an input sample");
        }
        return v;
    };
    const auto system_series = [&](const NeuralSource& s, size_t n) {
        const double dt = (double)in.hold_steps * r.h * num_or(s.time_scale_text, 1.0);
        return source_system_series(s, three ? 2 * n + 1 : n, three ? 0.5 * dt : dt, three ? 2 : 1);
    };
    for (int q = 0; q < in.channels; ++q) {
        if (class_of[(size_t)q] >= 0) continue;
        const NeuralSource& s = c.sources[(size_t)q];
        SourceSeries v = make_series(q);
        if (s.kind == NeuralSourceKind::System) v.series = system_series(s, n_active);
        size_t active = 0;
        for (long long row = 0; row < rows; ++row) {
            long long tau, trial;
            if (!active_at(row, tau, trial)) continue;
            for (int j = 0; j < in.stage_rows; ++j)
                in.samples[((size_t)row * in.stage_rows + j) * in.channels + q] = v.value(tau, j, three, row, active);
            ++active;
        }
    }
    if (classify) {
        // Попытка j класса берёт кусок его траектории с отсчёта j * cycle_rows + rest_rows: система идёт и в паузе, невидимо.
        if (rest % in.hold_steps != 0 || stim % in.hold_steps != 0)
            throw std::runtime_error("the rest and the stimulus must be multiples of the input sample time");
        const long long rest_rows = rest / in.hold_steps, cycle_rows = cycle / in.hold_steps;
        const int C = (int)c.task.classes.size();
        std::vector<long long> segment((size_t)n_trials), count((size_t)C, 0);
        for (long long i = 0; i < n_trials; ++i) segment[(size_t)i] = count[(size_t)labels[(size_t)i]]++;
        std::vector<SourceSeries> cls((size_t)C);
        for (int k = 0; k < C; ++k) {
            const int q = c.task.classes[(size_t)k];
            cls[(size_t)k] = make_series(q);
            if (c.sources[(size_t)q].kind == NeuralSourceKind::System)
                cls[(size_t)k].series = system_series(c.sources[(size_t)q], (size_t)(std::max(1LL, count[(size_t)k]) * cycle_rows));
        }
        for (long long row = 0; row < rows; ++row) {
            long long tau, trial;
            if (!active_at(row, tau, trial)) continue;
            const int L = labels[(size_t)trial];
            const SourceSeries& v = cls[(size_t)L];
            const size_t at = (size_t)(segment[(size_t)trial] * cycle_rows + rest_rows + tau / in.hold_steps);
            for (int j = 0; j < in.stage_rows; ++j) {
                const double x = v.value(tau, j, three, row, at);
                for (int q : c.task.classes) in.samples[((size_t)row * in.stage_rows + j) * in.channels + (size_t)q] = x;
            }
        }
    }

    in.node_channel.assign((size_t)g.N, -1);
    in.node_gain.assign((size_t)g.N, 0.0);
    const int nPops = (int)g.pop_start.size() - 1;
    for (int pop = 0; pop < nPops; ++pop) {
        const int i0 = g.pop_start[(size_t)pop], n = g.pop_start[(size_t)pop + 1] - i0;
        std::vector<int> order((size_t)n);
        std::iota(order.begin(), order.end(), i0);
        Stream gp(seed ^ 0x494E5055ULL, (uint64_t)pop, 0);
        for (int i = n - 1; i > 0; --i) std::swap(order[(size_t)i], order[(size_t)gp.below(i + 1)]);
        long long at = 0;
        double taken = 0.0;
        for (size_t q = 0; q < c.links.size(); ++q) {
            const NeuralLink& l = c.links[q];
            if (l.source < 0 || l.source >= in.channels) throw std::runtime_error("an input link points to a missing source");
            if (l.population < 0 || l.population >= nPops) throw std::runtime_error("an input link points to a missing population");
            if (l.population != pop) continue;
            const double frac = num_or(l.fraction_text, 1.0);
            if (!(frac >= 0.0 && frac <= 1.0)) throw std::runtime_error("an input fraction must be in [0, 1]");
            taken += frac;
            if (taken > 1.0 + 1e-9) throw std::runtime_error("input fractions of one population add up to more than 1");
            const long long m = (std::min)((long long)n - at, std::llround(frac * n));
            const double gain = num_or(l.gain_text, 1.0), spread = num_or(l.gain_spread_text, 0.0);
            for (long long j = 0; j < m; ++j) {
                const int node = order[(size_t)(at + j)];
                Stream gg(seed ^ 0x4741494EULL, (uint64_t)q, (uint64_t)node);
                in.node_channel[(size_t)node] = l.source;
                in.node_gain[(size_t)node] = gain * (spread != 0.0 ? 1.0 + spread * (2.0 * gg.uni() - 1.0) : 1.0);
            }
            at += m;
        }
    }
}

// Среднее по времени конечных значений; у остановленного экземпляра NaN, а не среднее по началу.
void neural_sweep_summary(NeuralOutcome& out, int order_var) {
    const NeuralResult& r = out.result;
    const int nInst = (int)r.instance_stop.size(), K = out.pops_per_instance;
    if (nInst < 1 || K < 1 || r.n_points < 1) return;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int nPops = nInst * K, nRec = (int)r.observed_vars.size();
    // Разброс той переменной, по которой считается R, если она записана.
    const int want = order_var >= 0 ? order_var : 0;
    const int var = (int)(std::find(r.observed_vars.begin(), r.observed_vars.end(), want) - r.observed_vars.begin());
    out.sweep_order.assign((size_t)nPops, nan);
    out.sweep_spread.assign((size_t)nPops, nan);
    for (int q = 0; q < nPops; ++q) {
        if (r.instance_stop[(size_t)(q / K)] >= 0) continue;
        double so = 0.0, ss = 0.0;
        int no = 0, ns = 0;
        for (int p = 0; p < r.n_points; ++p) {
            if (!r.order.empty()) {
                const double v = r.order[(size_t)p * nPops + q];
                if (std::isfinite(v)) { so += v; ++no; }
            }
            if (!r.spread.empty() && var < nRec) {
                const double v = r.spread[((size_t)p * nPops + q) * nRec + var];
                if (std::isfinite(v)) { ss += v; ++ns; }
            }
        }
        if (no > 0) out.sweep_order[(size_t)q] = so / no;
        if (ns > 0) out.sweep_spread[(size_t)q] = ss / ns;
    }
}

struct Rows {
    std::vector<long long> start;   // [n_to + 1]
    std::vector<int> src;           // локальные номера источника
    double expected = 0.0;          // ожидаемая степень
};

// Округлённое вниз log(1 - u)/log(1 - p) не обходит каждую пару: O(дуг), а не O(N^2).
void row_probability(Stream& g, int n_from, int self, double p, std::vector<int>& row) {
    if (p <= 0.0) return;
    if (p >= 1.0) { for (int s = 0; s < n_from; ++s) if (s != self) row.push_back(s); return; }
    const double lq = std::log1p(-p);
    for (long long s = -1;;) {
        const double step = std::floor(std::log1p(-g.uni()) / lq);
        if (step >= (double)n_from) break;
        s += 1 + (long long)step;
        if (s >= n_from) break;
        if ((int)s != self) row.push_back((int)s);
    }
}
// m различных чисел из [0, avail), self пропускается; большая степень считается через дополнение.
void row_fixed(Stream& g, int n_from, int self, int d, std::vector<int>& row, std::vector<int>& tmp) {
    const int avail = n_from - (self >= 0 ? 1 : 0);
    const bool invert = d > avail / 2;
    const int m = invert ? avail - d : d;
    tmp.clear();
    while ((int)tmp.size() < m) {
        for (int need = m - (int)tmp.size(); need > 0; --need) tmp.push_back(g.below(avail));
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    }
    const auto real = [&](int v) { return (self >= 0 && v >= self) ? v + 1 : v; };
    if (!invert) { for (int v : tmp) row.push_back(real(v)); return; }
    size_t at = 0;
    for (int v = 0; v < avail; ++v) {
        if (at < tmp.size() && tmp[at] == v) { ++at; continue; }
        row.push_back(real(v));
    }
}
void row_regular(const NeuralProjection& p, int n, int w, int h, int k, int i, std::vector<int>& row) {
    if (p.rule == NeuralRule::Chain) {
        if (i > 0) row.push_back(i - 1);
        if (!p.directed && i + 1 < n) row.push_back(i + 1);
    } else if (p.rule == NeuralRule::Ring) {
        for (int j = 1; j <= k; ++j) {
            row.push_back((i + n - j % n) % n);
            if (!p.directed) row.push_back((i + j) % n);
        }
    } else {
        // Как у Network: тор замыкается только при стороне больше 2, иначе дуга уже есть.
        const int x = i % w, y = i / w;
        const auto at = [&](int ax, int ay) {
            if (ax < 0 || ax >= w) { if (!p.periodic || w <= 2) return -1; ax = (ax + w) % w; }
            if (ay < 0 || ay >= h) { if (!p.periodic || h <= 2) return -1; ay = (ay + h) % h; }
            return ay * w + ax;
        };
        const auto add = [&](int dx, int dy) { const int v = at(x + dx, y + dy); if (v >= 0) row.push_back(v); };
        add(-1, 0); add(0, -1);
        if (!p.directed) { add(1, 0); add(0, 1); }
        if (p.rule == NeuralRule::King) {
            add(-1, -1); add(1, -1);
            if (!p.directed) { add(1, 1); add(-1, 1); }
        }
    }
}
// Уоттс-Строгац: кольцо, затем конец ребра переносится с вероятностью p без петель и повторов.
void rows_small_world(const NeuralProjection& p, uint64_t seed, int proj, int n, int k, double rewire, Rows& r) {
    struct E { int a, b; };
    std::vector<E> edges;
    std::unordered_set<long long> seen;
    const auto key = [&](int a, int b) { if (a > b) std::swap(a, b); return (long long)a * n + b; };
    edges.reserve((size_t)n * k);
    seen.reserve((size_t)n * k * 2);
    for (int i = 0; i < n; ++i)
        for (int j = 1; j <= k; ++j) {
            const int b = (i + j) % n;
            if (b != i && seen.insert(key(i, b)).second) edges.push_back({ i, b });
        }
    Stream g(seed, (uint64_t)proj, 0x535749ULL);
    for (E& e : edges) {
        if (g.uni() >= rewire) continue;
        for (int attempt = 0; attempt < 32; ++attempt) {
            const int cand = g.below(n);
            if (cand == e.a || seen.count(key(e.a, cand))) continue;
            seen.erase(key(e.a, e.b));
            seen.insert(key(e.a, cand));
            e.b = cand;
            break;
        }
    }
    std::vector<long long> count((size_t)n + 1, 0);
    for (const E& e : edges) { ++count[(size_t)e.b + 1]; if (!p.directed) ++count[(size_t)e.a + 1]; }
    for (int i = 0; i < n; ++i) count[(size_t)i + 1] += count[(size_t)i];
    r.start = count;
    r.src.resize((size_t)count[(size_t)n]);
    for (const E& e : edges) {
        r.src[(size_t)count[(size_t)e.b]++] = e.a;
        if (!p.directed) r.src[(size_t)count[(size_t)e.a]++] = e.b;
    }
    for (int i = 0; i < n; ++i)
        std::sort(r.src.begin() + (std::ptrdiff_t)r.start[(size_t)i], r.src.begin() + (std::ptrdiff_t)r.start[(size_t)i + 1]);
    r.expected = (double)r.src.size() / (double)n;
}

Rows build_rows(const NeuralProjection& p, int proj, uint64_t seed, int n_from, int n_to) {
    Rows r;
    const bool same = p.from == p.to;
    const bool regular = p.rule != NeuralRule::Probability && p.rule != NeuralRule::FixedDegree;
    if (regular && !same) throw std::runtime_error("regular topologies need a projection inside one population");

    double prob = 0.0;
    int d = 0, k = 1, w = 1, h = 1;
    double rewire = 0.0;
    if (p.rule == NeuralRule::Probability) {
        prob = num_or(p.p_text, 0.0);
        if (!(prob >= 0.0 && prob <= 1.0)) throw std::runtime_error("probability must be in [0, 1]");
        r.expected = prob * (double)(n_from - (same ? 1 : 0));
    } else if (p.rule == NeuralRule::FixedDegree) {
        d = int_or(p.degree_text, 0);
        if (d < 0 || d > n_from - (same ? 1 : 0)) throw std::runtime_error("the degree does not fit the source population");
        r.expected = d;
    } else if (p.rule == NeuralRule::Ring || p.rule == NeuralRule::SmallWorld) {
        k = int_or(p.k_text, 1);
        if (k < 1 || 2 * k >= n_to) throw std::runtime_error("neighbours each side must be between 1 and (size - 1) / 2");
    } else if (p.rule == NeuralRule::Grid || p.rule == NeuralRule::King) {
        w = int_or(p.width_text, 1);
        if (w < 1 || n_to % w != 0) throw std::runtime_error("the population size must divide by the lattice width");
        h = n_to / w;
    }
    if (p.rule == NeuralRule::SmallWorld) {
        rewire = num_or(p.rewire_text, 0.0);
        if (!(rewire >= 0.0 && rewire <= 1.0)) throw std::runtime_error("rewire probability must be in [0, 1]");
        rows_small_world(p, seed, proj, n_to, k, rewire, r);
        return r;
    }
    if ((double)n_to * r.expected > (double)kMaxArcs)
        throw std::runtime_error("a projection would hold more than " + std::to_string(kMaxArcs) + " arcs");

    r.start.assign((size_t)n_to + 1, 0);
    const size_t guess = (size_t)((double)n_to * (r.expected > 0.0 ? r.expected : 8.0) * 1.05);
    r.src.reserve(guess);
    std::vector<int> row, tmp;
    for (int i = 0; i < n_to; ++i) {
        row.clear();
        Stream g(seed, (uint64_t)proj, (uint64_t)i);
        const int self = same ? i : -1;
        if (p.rule == NeuralRule::Probability) row_probability(g, n_from, self, prob, row);
        else if (p.rule == NeuralRule::FixedDegree) row_fixed(g, n_from, self, d, row, tmp);
        else {
            row_regular(p, n_to, w, h, k, i, row);
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            row.erase(std::remove(row.begin(), row.end(), i), row.end());
        }
        r.src.insert(r.src.end(), row.begin(), row.end());
        r.start[(size_t)i + 1] = (long long)r.src.size();
        if ((long long)r.src.size() > kMaxArcs)
            throw std::runtime_error("a projection would hold more than " + std::to_string(kMaxArcs) + " arcs");
    }
    if (regular) r.expected = (double)r.src.size() / (double)n_to;
    return r;
}

// FNV-1a по словам: хеш графа в канонической нумерации.
struct Fnv {
    uint64_t h = 0xCBF29CE484222325ULL;
    void add(uint64_t v) { h = (h ^ v) * 0x100000001B3ULL; }
};
std::string hex64(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return b;
}
}  // namespace

const char* neural_rule_name(NeuralRule r) {
    switch (r) {
    case NeuralRule::Probability: return "Probability";
    case NeuralRule::FixedDegree: return "Fixed in-degree";
    case NeuralRule::Chain:       return "Chain";
    case NeuralRule::Ring:        return "Ring";
    case NeuralRule::Grid:        return "Grid";
    case NeuralRule::King:        return "King graph";
    case NeuralRule::SmallWorld:  return "Small world";
    }
    return "?";
}
const char* neural_schedule_name(NeuralSchedule s) {
    switch (s) {
    case NeuralSchedule::Stage: return "Stage (RK4, 4 exchanges)";
    case NeuralSchedule::Heun:  return "Heun (2 exchanges)";
    case NeuralSchedule::Post:  return "Post (1 exchange)";
    case NeuralSchedule::Old:   return "Old state (Network)";
    case NeuralSchedule::Spike: return "Spikes (events with a delay)";
    }
    return "?";
}
void neural_source_set_system(NeuralSource& src, const std::string& name, const SystemRecord& r, const System& sys) {
    if (sys.rhs.empty()) throw std::runtime_error("the system does not parse");
    if (sys.is_map || sys.dae_nx >= 0) throw std::runtime_error("an input source must be an ODE system");
    src.system_name = name;
    src.vars = sys.vars; src.params = sys.params; src.rhs = sys.rhs; src.latex = sys.latex;
    src.initial_conditions = r.init_conditions; src.param_values = r.param_values;
    src.variable = 0;
}

const char* neural_sweep_name(NeuralSweepKind k) {
    switch (k) {
    case NeuralSweepKind::Coupling:        return "Coupling K";
    case NeuralSweepKind::Parameter:       return "Parameter";
    case NeuralSweepKind::ParameterSpread: return "Parameter spread";
    case NeuralSweepKind::Replica:         return "Replicas (seed)";
    case NeuralSweepKind::InputGain:       return "Input gain";
    }
    return "?";
}
const char* neural_source_name(NeuralSourceKind k) {
    switch (k) {
    case NeuralSourceKind::Constant: return "Constant";
    case NeuralSourceKind::Sine:     return "Sine";
    case NeuralSourceKind::Square:   return "Square";
    case NeuralSourceKind::Noise:    return "Gaussian noise";
    case NeuralSourceKind::Bits:     return "Random bits";
    case NeuralSourceKind::System:   return "System from the library";
    }
    return "?";
}

static constexpr int kNeuralWatched = 16;                 // дуг и устройств на сеть в окне Neural Weights
static constexpr char kNeuralCheckpointMagic[9] = "UCUDACK1";

// Время в целых шагах: событие стоит на сетке. floor_steps = наименьшее допустимое, 0 у задержки это один шаг.
static int spike_steps(const std::string& text, double h, const char* what, int floor_steps) {
    const double t = num_or(text, 0.0);
    if (!(t >= 0.0) || !(h > 0.0)) throw std::runtime_error(std::string(what) + " must not be negative");
    const long long n = std::llround(t / h);
    if (std::fabs((double)n * h - t) > 1e-9 * (std::max)(t, h)) throw std::runtime_error(std::string(what) + " must be a multiple of h");
    if (n > 2000000000LL) throw std::runtime_error(std::string(what) + " is too long");
    return (std::max)((int)n, floor_steps);
}

NeuralGraph neural_build_graph(const NeuralConfig& c) {
    NeuralGraph g;
    const int nPops = (int)c.populations.size();
    if (nPops == 0) throw std::runtime_error("the network has no populations");
    g.pop_start.assign((size_t)nPops + 1, 0);
    long long total = 0;
    for (int p = 0; p < nPops; ++p) {
        const int n = int_or(c.populations[(size_t)p].size_text, 0);
        if (n < 1) throw std::runtime_error("a population must have at least one node");
        total += n;
        if (total > 2000000000LL) throw std::runtime_error("too many nodes");
        g.pop_start[(size_t)p + 1] = (int)total;
    }
    g.N = (int)total;
    const double K = num_or(c.coupling_text, 0.0);
    const uint64_t seed = (uint64_t)int_or(c.seed_text, 1);
    const bool spike = c.schedule == NeuralSchedule::Spike;
    const double h = parse_num(c.h_text, 0.01);

    std::vector<Rows> rows(c.projections.size());
    bool devices = false;
    for (size_t q = 0; q < c.projections.size(); ++q) {
        const NeuralProjection& p = c.projections[q];
        if (p.from < 0 || p.from >= nPops || p.to < 0 || p.to >= nPops) throw std::runtime_error("a projection points to a missing population");
        if (spike) {
            if (p.receptor < 0 || p.receptor >= (int)c.receptors.size()) throw std::runtime_error("a projection uses a missing synapse: add one in the Spiking block");
            if (c.populations[(size_t)p.to].spike_source) throw std::runtime_error("a spike source cannot receive arcs");
            if (p.plastic < -1 || p.plastic >= (int)c.plastic_rules.size()) throw std::runtime_error("a projection uses a missing plasticity rule");
        }
        else if (!p.device && (p.law < 0 || p.law >= (int)c.laws.size())) throw std::runtime_error("a projection uses a missing coupling law");
        if (p.device) {
            if (c.schedule != NeuralSchedule::Stage) throw std::runtime_error("a device on a link needs the Stage schedule");
            devices = true;
        }
        rows[q] = build_rows(p, (int)q, seed, g.pop_start[(size_t)p.from + 1] - g.pop_start[(size_t)p.from],
                             g.pop_start[(size_t)p.to + 1] - g.pop_start[(size_t)p.to]);
    }
    long long arcs = 0;
    for (const Rows& r : rows) arcs += (long long)r.src.size();
    if (arcs > kMaxArcs) throw std::runtime_error("the network would hold more than " + std::to_string(kMaxArcs) + " arcs");

    std::vector<double> expected_in((size_t)nPops, 0.0);
    for (size_t q = 0; q < c.projections.size(); ++q) expected_in[(size_t)c.projections[q].to] += rows[q].expected;

    g.start.assign((size_t)g.N + 1, 0);
    g.src.reserve((size_t)arcs); g.w.reserve((size_t)arcs); g.law.reserve((size_t)arcs);
    struct Arc { int src, law, delay, rule; double w; };
    std::vector<Arc> row;
    struct Link { int a, b, proj; double gain; };
    std::vector<Link> links;
    g.degree_min = std::numeric_limits<int>::max(); g.degree_max = 0;
    for (int pop = 0; pop < nPops; ++pop) {
        const int n = g.pop_start[(size_t)pop + 1] - g.pop_start[(size_t)pop];
        for (int t = 0; t < n; ++t) {
            row.clear();
            long long degree = 0;
            for (size_t q = 0; q < c.projections.size(); ++q)
                if (c.projections[q].to == pop) degree += rows[q].start[(size_t)t + 1] - rows[q].start[(size_t)t];
            for (size_t q = 0; q < c.projections.size(); ++q) {
                const NeuralProjection& p = c.projections[q];
                if (p.to != pop) continue;
                const Rows& r = rows[q];
                const long long a0 = r.start[(size_t)t], a1 = r.start[(size_t)t + 1];
                if (a0 == a1) continue;
                double div = 1.0;
                if (p.norm == NeuralNorm::Projection) div = p.norm_expected ? r.expected : (double)(a1 - a0);
                else if (p.norm == NeuralNorm::Total) div = p.norm_expected ? expected_in[(size_t)pop] : (double)degree;
                if (!(div > 0.0)) div = 1.0;
                const double base = K * num_or(p.weight_text, 1.0) / div;
                const double spread = num_or(p.weight_spread_text, 0.0);
                Stream gw(seed ^ 0x57454947ULL, (uint64_t)q, (uint64_t)t);
                const int off = g.pop_start[(size_t)p.from];
                const int law = spike ? p.receptor : p.law, delay = spike ? spike_steps(p.delay_text, h, "a delay", 1) : 0;
                for (long long a = a0; a < a1; ++a) {
                    const double f = spread != 0.0 ? 1.0 + spread * (2.0 * gw.uni() - 1.0) : 1.0;
                    if (p.device) links.push_back({ r.src[(size_t)a] + off, g.pop_start[(size_t)pop] + t, (int)q, base * f });
                    else row.push_back({ r.src[(size_t)a] + off, law, delay, spike ? p.plastic : -1, base * f });
                }
            }
            std::stable_sort(row.begin(), row.end(), [](const Arc& a, const Arc& b) { return a.src < b.src; });
            const int node = g.pop_start[(size_t)pop] + t;
            g.start[(size_t)node] = (int)g.src.size();
            for (const Arc& a : row) { g.src.push_back(a.src); g.law.push_back(a.law); g.w.push_back(a.w); if (spike) { g.delay.push_back(a.delay); g.rule.push_back(a.rule); } }
            g.degree_min = std::min(g.degree_min, (int)row.size());
            g.degree_max = std::max(g.degree_max, (int)row.size());
        }
    }
    g.start[(size_t)g.N] = (int)g.src.size();

    // Один элемент: встречная дуга той же проекции уже описана первой, вторая отбрасывается.
    if (devices) {
        std::unordered_set<uint64_t> seen;
        for (const Link& l : links) {
            const NeuralProjection& p = c.projections[(size_t)l.proj];
            if (p.one_element) {
                const uint64_t back = ((uint64_t)(uint32_t)l.b << 32) | (uint32_t)l.a;
                if (seen.count(back ^ ((uint64_t)l.proj << 60))) continue;
                seen.insert((((uint64_t)(uint32_t)l.a << 32) | (uint32_t)l.b) ^ ((uint64_t)l.proj << 60));
            }
            g.dev_a.push_back(l.a); g.dev_b.push_back(l.b); g.dev_var.push_back(p.device_var);
            g.dev_two_way.push_back(p.one_element ? 1 : 0); g.dev_gain.push_back(l.gain);
        }
    }

    Fnv f;
    f.add((uint64_t)g.N);
    for (int v : g.pop_start) f.add((uint64_t)(uint32_t)v);
    for (int v : g.start) f.add((uint64_t)(uint32_t)v);
    for (size_t e = 0; e < g.src.size(); ++e) {
        uint64_t bits;
        std::memcpy(&bits, &g.w[e], sizeof bits);
        f.add(((uint64_t)(uint32_t)g.src[e] << 8) | (uint64_t)(uint8_t)g.law[e]);
        f.add(bits);
        if (spike) f.add((uint64_t)(uint32_t)g.delay[e]);
    }
    for (size_t e = 0; e < g.dev_a.size(); ++e) {
        uint64_t bits;
        std::memcpy(&bits, &g.dev_gain[e], sizeof bits);
        f.add(((uint64_t)(uint32_t)g.dev_a[e] << 32) | (uint32_t)g.dev_b[e]);
        f.add(((uint64_t)(uint32_t)g.dev_var[e] << 1) | (uint64_t)g.dev_two_way[e]);
        f.add(bits);
    }
    g.hash = hex64(f.h);
    return g;
}

std::string neural_graph_recipe(const NeuralConfig& c) {
    std::ostringstream o;
    o << "v" << kNeuralGraphVersion << "|K=" << c.coupling_text << "|seed=" << c.seed_text << "|laws=" << c.laws.size();
    for (const NeuralPopulation& p : c.populations) o << "|pop:" << p.size_text;
    for (const NeuralProjection& p : c.projections)
        o << "|proj:" << p.from << ">" << p.to << ":" << (int)p.rule << ":" << p.p_text << ":" << p.degree_text << ":" << p.k_text
          << ":" << p.width_text << ":" << p.rewire_text << ":" << p.periodic << p.directed << ":" << p.weight_text << ":"
          << p.weight_spread_text << ":" << (int)p.norm << p.norm_expected << ":" << p.law;
    if (c.schedule == NeuralSchedule::Spike) {
        o << "|spike:h=" << c.h_text;
        for (const NeuralProjection& p : c.projections) o << ":" << p.receptor << "," << p.delay_text;
    }
    bool anyDevice = false;
    for (const NeuralProjection& p : c.projections) anyDevice = anyDevice || p.device;
    if (anyDevice) {
        o << "|dev";
        for (const NeuralProjection& p : c.projections) o << ":" << p.device << p.one_element << p.device_var;
    }
    const auto axis = [&o](const NeuralSweep& s, const char* tag) {
        o << tag << (int)s.kind << ":" << s.population << ":" << s.param << ":" << s.lo_text
          << ":" << s.hi_text << ":" << s.points_text << ":" << s.log;
    };
    if (c.sweep.enabled) {
        axis(c.sweep, "|sweep:");
        if (c.sweep2.enabled) axis(c.sweep2, "|sweep2:");
    }
    return o.str();
}

std::string neural_model_text(const System& sys) {
    std::string out;
    for (const auto& r : sys.rhs) out += r + ";";
    return out;
}

NeuralPopulation neural_default_population(const SystemRecord* r, const std::vector<std::string>& vars,
                                           const std::vector<std::string>& params) {
    NeuralPopulation p;
    for (const auto& name : params) {
        std::string v;
        if (r) { auto it = r->param_values.find(name); if (it != r->param_values.end()) v = it->second; }
        p.param_values[name] = v;
    }
    for (const auto& name : vars) {
        std::string v;
        if (r) { auto it = r->init_conditions.find(name); if (it != r->init_conditions.end()) v = it->second; }
        p.initial_conditions[name] = v;
        p.ic_spread[name] = "0.1";   // из одинаковых НУ диффузионная сеть стартует синхронной
    }
    return p;
}

const char* neural_device_preset_name(NeuralDevicePreset p) {
    return p == NeuralDevicePreset::Team ? "TEAM (current threshold)" : "VTEAM (voltage threshold)";
}

NeuralDeviceModel neural_device_preset(NeuralDevicePreset p) {
    NeuralDeviceModel m;
    m.current_text = "v/(Ron + (Roff - Ron)*m)";
    if (p == NeuralDevicePreset::Vteam) {
        m.rhs_text = "koff*max(v/voff - 1, 0)^aoff*exp(-exp((m - 0.9)/wc)) - kon*max(v/von - 1, 0)^aon*exp(-exp((0.1 - m)/wc))";
        m.params_text = "koff = 0.02, kon = 0.02, aoff = 2, aon = 2, voff = 0.5, von = -0.5, Ron = 1, Roff = 10, wc = 0.05";
    } else {
        m.rhs_text = "koff*max(v/(Ron + (Roff - Ron)*m)/ioff - 1, 0)^aoff*exp(-exp((m - 0.9)/wc)) - kon*max(v/(Ron + (Roff - Ron)*m)/ion - 1, 0)^aon*exp(-exp((0.1 - m)/wc))";
        m.params_text = "koff = 0.02, kon = 0.02, aoff = 2, aon = 2, ioff = 0.2, ion = -0.2, Ron = 1, Roff = 10, wc = 0.05";
    }
    return m;
}

// Модель устройства: правая часть и ток печатаются обычным кодгеном, параметры идут за v.
static void fill_devices(const NeuralConfig& c, const NeuralGraph& g, NeuralRequest& r) {
    const NeuralDeviceModel& m = c.device;
    if (m.rhs_text.empty() || m.current_text.empty()) throw std::runtime_error("the device has no equations: pick a preset in the Device block");
    System s;
    s.vars = { m.state_name };
    s.params = { "v" };
    std::vector<double> values;
    std::stringstream list(m.params_text);
    std::string item;
    while (std::getline(list, item, ',')) {
        const size_t eq = item.find('=');
        const auto trim = [](std::string t) {
            const size_t a = t.find_first_not_of(" \t"), b = t.find_last_not_of(" \t");
            return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
        };
        if (trim(item).empty()) continue;
        if (eq == std::string::npos) throw std::runtime_error("device parameters must read name = value");
        s.params.push_back(trim(item.substr(0, eq)));
        values.push_back(num_or(trim(item.substr(eq + 1)), 0.0));
    }
    System cur = s;
    s.rhs = { m.rhs_text };
    cur.rhs = { m.current_text };
    r.devices.states = 1;
    r.devices.params = (int)values.size();
    r.devices.rhs_body = codegen_rhs_body(s);
    r.devices.current_body = codegen_rhs_body(cur);
    r.devices.s_min = { num_or(m.min_text, 0.0) };
    r.devices.s_max = { num_or(m.max_text, 1.0) };
    r.devices.a = g.dev_a; r.devices.b = g.dev_b; r.devices.var = g.dev_var; r.devices.two_way = g.dev_two_way; r.devices.gain = g.dev_gain;
    const double s0 = num_or(m.initial_text, 0.5);
    r.devices.initial.assign(g.dev_a.size(), s0);
    r.devices.values.reserve(g.dev_a.size() * values.size());
    for (size_t d = 0; d < g.dev_a.size(); ++d) r.devices.values.insert(r.devices.values.end(), values.begin(), values.end());
    for (int d = 0; d < (std::min)(kNeuralWatched, (int)g.dev_a.size()); ++d) r.devices.watch.push_back(d);
}

// Всё, чем импульсная сеть отличается от непрерывной: система с синапсами, правило спайка, источники, задержки.
static void fill_spiking(const System& sys, const std::vector<std::string>& vars, const NeuralConfig& c, const NeuralGraph& g, uint64_t seed, NeuralRequest& r) {
    if (sys.is_map) throw std::runtime_error("a spiking network needs a flow, not a map");
    if (c.lyap_on) throw std::runtime_error("the Lyapunov exponent does not work on a spiking network");
    if (c.threshold_var < 0 || c.threshold_var >= (int)vars.size()) throw std::runtime_error("the threshold variable is missing");
    if (c.threshold_text.empty()) throw std::runtime_error("the threshold is empty");
    const int R = (int)c.receptors.size();
    if (R > 8) throw std::runtime_error("no more than 8 synapses");

    std::vector<std::string> names, terms;
    std::vector<int> target;
    for (int k = 0; k < R; ++k) {
        const NeuralReceptor& q = c.receptors[(size_t)k];
        if (q.target < 0 || q.target >= (int)vars.size()) throw std::runtime_error("synapse " + q.name + " points to a missing variable");
        const double tau = num_or(q.tau_text, 0.0);
        if (!(tau > 0.0)) throw std::runtime_error("synapse " + q.name + ": the time constant must be positive");
        r.spike.tau.push_back(tau);
        const std::string name = "ucudasyn" + std::to_string(k), gain = q.gain_text.empty() ? std::string("1") : q.gain_text;
        const std::string reversal = q.reversal_text.empty() ? std::string("0") : q.reversal_text;
        names.push_back(name);
        target.push_back(q.target);
        terms.push_back("(" + gain + ")*" + name + (q.conductance ? "*((" + reversal + ") - " + vars[(size_t)q.target] + ")" : std::string()));
    }
    const System spk = codegen_spike_system(sys, names, target, terms);
    r.rhs_body = codegen_rhs_body(spk);
    std::vector<std::string> reset = c.reset_text;
    reset.resize(spk.vars.size());
    r.spike.spike_body = codegen_spike_body(spk, c.threshold_var, c.threshold_text, reset);
    r.spike.receptors = R;
    r.spike.seed = seed;
    r.spike.edge_delay = g.delay;
    r.spike.refractory_steps.assign((size_t)g.N, spike_steps(c.refractory_text, r.h, "the refractory time", 0));
    r.amountOfX = (int)spk.vars.size();
    r.edge_law = g.law;
    if (c.input_var < 0 || c.input_var >= (int)vars.size()) throw std::runtime_error("the input variable is out of range");
    if (std::any_of(g.rule.begin(), g.rule.end(), [](int q) { return q >= 0; })) {
        for (const NeuralPlasticRule& q : c.plastic_rules) {
            NeuralStdpRule o;
            o.a_plus = num_or(q.a_plus_text, 0.0); o.a_minus = num_or(q.a_minus_text, 0.0);
            o.tau_plus = num_or(q.tau_plus_text, 20.0); o.tau_minus = num_or(q.tau_minus_text, 20.0);
            o.w_min = num_or(q.w_min_text, 0.0); o.w_max = num_or(q.w_max_text, 1.0);
            o.mu_plus = num_or(q.mu_plus_text, 0.0); o.mu_minus = num_or(q.mu_minus_text, 0.0);
            o.nearest = q.nearest;
            r.spike.rules.push_back(o);
        }
        r.spike.edge_rule = g.rule;
        // Несколько дуг на всю сеть, равномерно: показать динамику, а не хранить все веса во времени.
        std::vector<int> plastic;
        for (int e = 0; e < (int)g.rule.size(); ++e) if (g.rule[(size_t)e] >= 0) plastic.push_back(e);
        const int m = (std::min)(kNeuralWatched, (int)plastic.size());
        for (int k = 0; k < m; ++k) r.spike.watch_arcs.push_back(plastic[(size_t)((long long)k * (long long)plastic.size() / m)]);
    }

    for (size_t pop = 0; pop < c.populations.size(); ++pop) {
        const NeuralPopulation& p = c.populations[pop];
        if (!p.spike_source) continue;
        if (r.spike.source_p.empty()) {
            r.spike.source_p.assign((size_t)g.N, -1.0);
            r.spike.source_from.assign((size_t)g.N, 0);
            r.spike.source_to.assign((size_t)g.N, 0);
            r.spike.source_seed.resize((size_t)g.N);
            for (int i = 0; i < g.N; ++i) r.spike.source_seed[(size_t)i] = neural_source_seed(seed, i);
        }
        const double rate = num_or(p.rate_text, 0.0), to = num_or(p.source_to_text, 0.0);
        if (!(rate >= 0.0) || rate * r.h > 1.0) throw std::runtime_error("population " + p.name + ": the rate must lie between 0 and 1 / h");
        const int from = spike_steps(p.source_from_text, r.h, "the start of a source", 0);
        const int stop = to > 0.0 ? spike_steps(p.source_to_text, r.h, "the end of a source", 0) : 2147483647;
        for (int i = g.pop_start[pop]; i < g.pop_start[pop + 1]; ++i) {
            r.spike.source_p[(size_t)i] = rate * r.h;
            r.spike.source_from[(size_t)i] = from;
            r.spike.source_to[(size_t)i] = stop;
        }
    }
}

NeuralRequest neural_build_request(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                                   const std::vector<std::string>& vars, const std::vector<std::string>& params,
                                   const NeuralConfig& c, const NeuralGraph& g) {
    if (sys.dae_nx >= 0) throw std::runtime_error("DAE systems are not supported on this tab yet");
    const bool spike = c.schedule == NeuralSchedule::Spike;
    if (sys.is_map && c.schedule == NeuralSchedule::Stage) throw std::runtime_error("a map has no stages: pick another schedule");
    NeuralRequest r;
    r.schedule = c.schedule;
    const bool stage = c.schedule == NeuralSchedule::Stage;
    r.krs_body = spike ? std::string(" ") : compute_krs_for_scheme(custom_schemes, sys, stage ? std::string("RK4") : c.scheme);
    if (r.krs_body.empty()) throw std::runtime_error(krs_empty_error(custom_schemes, sys, stage ? std::string("RK4") : c.scheme));
    if (stage) r.rhs_body = codegen_rhs_body(sys);
    if (!spike) r.coupling_body = net_coupling_body(sys, c.laws);
    r.n_laws = spike ? 1 : (int)c.laws.size();
    r.amountOfX = (int)vars.size();
    r.amountOfValues = (int)params.size() + 1 + kSchemeTailSlots;
    r.n_nodes = g.N;
    r.edge_start = g.start; r.edge_src = g.src; r.edge_w = g.w;
    if (r.n_laws > 1) r.edge_law = g.law;

    const uint64_t seed = (uint64_t)int_or(c.seed_text, 1);
    r.h = parse_num(c.h_text, 0.01);
    if (spike) fill_spiking(sys, vars, c, g, seed, r);
    if (!g.dev_a.empty()) fill_devices(c, g, r);
    r.keep_checkpoint = c.keep_checkpoint;
    r.values.assign((size_t)g.N * (size_t)r.amountOfValues, 0.0);
    r.initial_conditions.assign((size_t)g.N * (size_t)r.amountOfX, 0.0);
    const double s_default = num_or(c.symmetry_s, 0.5);
    for (size_t pop = 0; pop < c.populations.size(); ++pop) {
        const NeuralPopulation& p = c.populations[pop];
        std::vector<double> value(params.size()), spread(params.size()), x0(vars.size()), dx(vars.size());
        const auto get = [](const std::map<std::string, std::string>& m, const std::string& k) {
            auto it = m.find(k);
            return it == m.end() ? std::string() : it->second;
        };
        for (size_t j = 0; j < params.size(); ++j) { value[j] = num_or(get(p.param_values, params[j]), 0.0); spread[j] = num_or(get(p.param_spread, params[j]), 0.0); }
        for (size_t k = 0; k < vars.size(); ++k) { x0[k] = num_or(get(p.initial_conditions, vars[k]), 0.0); dx[k] = num_or(get(p.ic_spread, vars[k]), 0.0); }
        for (int i = g.pop_start[pop]; i < g.pop_start[pop + 1]; ++i) {
            double* a = r.values.data() + (size_t)i * r.amountOfValues;
            a[0] = s_default;
            scheme_fill_tail(a + params.size() + 1, c.cd_fp_iters, c.newton);
            Stream gp(seed ^ 0x50415241ULL, (uint64_t)pop, (uint64_t)(i - g.pop_start[pop]));
            for (size_t j = 0; j < params.size(); ++j)
                a[j + 1] = value[j] + (spread[j] != 0.0 ? spread[j] * (2.0 * gp.uni() - 1.0) : 0.0);
            Stream gi(seed ^ 0x49435331ULL, (uint64_t)pop, (uint64_t)(i - g.pop_start[pop]));
            for (size_t k = 0; k < vars.size(); ++k)
                r.initial_conditions[(size_t)i * r.amountOfX + k] = x0[k] + (dx[k] != 0.0 ? dx[k] * (2.0 * gi.uni() - 1.0) : 0.0);
        }
    }
    r.t_max = neural_record_time(c);
    r.transient = parse_num(c.transient_text, 0.0);
    r.pre_scaller = std::max(1, int_or(c.pre_scaller_text, 1));
    r.max_value = parse_num(c.max_value_text, 1.0e6);
    fill_input(c, g, seed, r);
    r.lyapunov.enabled = c.lyap_on;
    r.lyapunov.eps = num_or(c.lyap_eps_text, 1.0e-8);
    r.lyapunov.interval = num_or(c.lyap_interval_text, 1.0);
    r.lyapunov.settle = num_or(c.lyap_settle_text, 100.0);
    r.lyapunov.seed = seed;

    for (size_t k = 0; k < vars.size(); ++k)
        if (std::find(c.obs_vars_off.begin(), c.obs_vars_off.end(), (int)k) == c.obs_vars_off.end()) r.observe.vars.push_back((int)k);
    r.observe.pop_start = g.pop_start;
    r.observe.spread = !spike;
    // Пересечения порога в точку не входят, поэтому при контрольной точке R не считается.
    r.observe.order_var = !spike && r.reset_at.empty() && !c.keep_checkpoint && c.order_var < (int)vars.size() ? c.order_var : -1;
    r.observe.order_threshold = num_or(c.order_threshold_text, 0.0);
    const double limit_mb = num_or(c.output_limit_text, 1024.0);
    if (!(limit_mb > 0.0)) throw std::runtime_error("the output limit must be positive");
    r.max_output_bytes = (size_t)(limit_mb * 1024.0 * 1024.0);
    const int per_pop = std::max(0, int_or(c.observed_per_pop_text, 8));
    for (size_t pop = 0; pop + 1 < g.pop_start.size(); ++pop) {
        const int n = g.pop_start[pop + 1] - g.pop_start[pop], m = std::min(per_pop, n);
        for (int j = 0; j < m; ++j) r.observe.nodes.push_back(g.pop_start[pop] + (int)((long long)j * n / m));
    }
    return r;
}

double neural_record_time(const NeuralConfig& c) {
    if (c.protocol == NeuralProtocol::Continuous) return parse_num(c.t_max_text, 100.0);
    return num_or(c.trials_text, 0.0) * (num_or(c.rest_text, 0.0) + num_or(c.stimulus_text, 0.0));
}

static std::vector<double> sweep_axis_values(const NeuralConfig& c, const NeuralSweep& s) {
    std::vector<double> out;
    if (!s.enabled) return out;
    const int n = int_or(s.points_text, 0);
    if (n < 1) throw std::runtime_error("a sweep needs at least one point");
    out.resize((size_t)n);
    if (s.kind == NeuralSweepKind::Replica) {
        const int seed = int_or(c.seed_text, 1);
        for (int i = 0; i < n; ++i) out[(size_t)i] = (double)seed + i;
        return out;
    }
    const double lo = num_or(s.lo_text, 0.0), hi = num_or(s.hi_text, 0.0);
    if (!std::isfinite(lo) || !std::isfinite(hi)) throw std::runtime_error("the sweep range is not a number");
    if (s.log && !(lo > 0.0 && hi > 0.0)) throw std::runtime_error("a log sweep needs a positive range");
    for (int i = 0; i < n; ++i) {
        const double f = n > 1 ? (double)i / (double)(n - 1) : 0.0;
        out[(size_t)i] = s.log ? std::exp(std::log(lo) + f * (std::log(hi) - std::log(lo))) : lo + f * (hi - lo);
    }
    if (n > 1) out.back() = hi;
    return out;
}

std::vector<double> neural_sweep_values(const NeuralConfig& c) { return sweep_axis_values(c, c.sweep); }

std::vector<double> neural_sweep2_values(const NeuralConfig& c) {
    return c.sweep.enabled ? sweep_axis_values(c, c.sweep2) : std::vector<double>{};
}

static void sweep_apply(NeuralConfig& o, const NeuralSweep& s, int index, double value) {
    const std::string text = fmt_num_shortest(value);
    if (s.kind == NeuralSweepKind::Coupling) o.coupling_text = text;
    else if (s.kind == NeuralSweepKind::Replica) o.seed_text = std::to_string(int_or(o.seed_text, 1) + index);
    else if (s.kind == NeuralSweepKind::InputGain) {
        if (o.links.empty()) throw std::runtime_error("the sweep over the input gain needs an input link");
        for (NeuralLink& l : o.links) l.gain_text = text;
    }
    else {
        if (s.param.empty()) throw std::runtime_error("the sweep has no parameter");
        if (s.population >= (int)o.populations.size()) throw std::runtime_error("the sweep points to a missing population");
        for (int p = 0; p < (int)o.populations.size(); ++p) {
            if (s.population >= 0 && s.population != p) continue;
            auto& m = s.kind == NeuralSweepKind::Parameter ? o.populations[(size_t)p].param_values : o.populations[(size_t)p].param_spread;
            m[s.param] = text;
        }
    }
}

NeuralConfig neural_sweep_instance(const NeuralConfig& c, int index, double value) {
    NeuralConfig o = c;
    o.sweep.enabled = false;
    sweep_apply(o, c.sweep, index, value);
    return o;
}

std::vector<double> neural_source_samples(const NeuralSource& s, int variable, size_t n, double dt) {
    NeuralSource raw = s;
    raw.variable = variable;
    raw.normalize = false;
    return source_system_series(raw, n, dt);
}

static ReadoutSplit task_split(const NeuralTask& t, size_t total) {
    const int wash = int_or(t.washout_text, 0), gap = int_or(t.gap_text, 0);
    if (wash < 0 || gap < 0) throw std::runtime_error("the washout and the gap must not be negative");
    return readout_split(total, (size_t)wash, num_or(t.train_text, 0.6), num_or(t.val_text, 0.2), (size_t)gap);
}
static long long task_hold(const NeuralConfig& c) {
    const double hold = num_or(c.input_hold_text, 0.0);
    return hold > 0.0 ? (std::max)(1LL, steps_of(hold, parse_num(c.h_text, 0.01), "the input sample time")) : 1;
}

// Классификация: части по попыткам, без прогрева и пропуска: пауза перед стимулом уже отделяет попытки.
static ReadoutSplit class_split(const NeuralConfig& c) {
    const long long n = int_or(c.trials_text, 0);
    if (n < 3) throw std::runtime_error("classification needs more trials");
    return readout_split((size_t)n, 0, num_or(c.task.train_text, 0.6), num_or(c.task.val_text, 0.2), 0);
}

std::vector<int> neural_class_labels(const NeuralConfig& c) {
    const int C = (int)c.task.classes.size();
    if (C < 2) throw std::runtime_error("classification needs at least two classes");
    const ReadoutSplit sp = class_split(c);
    std::vector<int> labels(sp.test1);
    Stream g((uint64_t)int_or(c.task.data_seed_text, 1), 0x4C4142454CULL, 0);
    const size_t cut[4] = { 0, sp.train1, sp.val1, sp.test1 };
    for (int part = 0; part < 3; ++part) {
        const size_t lo = cut[part], n = cut[part + 1] - lo;
        if (n < (size_t)C) throw std::runtime_error("every part of the split needs a trial of every class: raise the number of trials");
        for (size_t i = 0; i < n; ++i) labels[lo + i] = (int)(i % (size_t)C);
        for (size_t i = n; i > 1; --i) std::swap(labels[lo + i - 1], labels[lo + (size_t)g.below((int)i)]);
    }
    return labels;
}

struct ClassSteps { long long rest = 0, stim = 0, cycle = 0, bins = 1, bin = 0; };
static ClassSteps class_steps(const NeuralConfig& c) {
    const double h = parse_num(c.h_text, 0.01);
    ClassSteps s;
    s.rest = steps_of(num_or(c.rest_text, 0.0), h, "the rest time");
    s.stim = steps_of(num_or(c.stimulus_text, 0.0), h, "the stimulus time");
    s.cycle = s.rest + s.stim;
    s.bins = int_or(c.task.bins_text, 1);
    if (s.bins < 1 || s.stim < s.bins || s.stim % s.bins != 0) throw std::runtime_error("the stimulus must split into the feature points by whole steps");
    s.bin = s.stim / s.bins;
    return s;
}

static NeuralConfig class_config(const NeuralConfig& c) {
    NeuralConfig o = c;
    const NeuralTask& t = c.task;
    if (c.protocol != NeuralProtocol::Trials) throw std::runtime_error("classification needs the protocol of separate trials");
    if (num_or(c.transient_text, 0.0) != 0.0) throw std::runtime_error("set the transient to 0: the rest before each stimulus is the washout");
    if (c.lyap_on) throw std::runtime_error("a readout does not work with the Lyapunov exponent");
    // Галочка сохранённого выхода остаётся от режима восстановления и здесь не видна: классификация её не читает.
    o.task.use_saved = false;
    for (size_t k = 0; k < t.classes.size(); ++k) {
        if (t.classes[k] < 0 || t.classes[k] >= (int)c.sources.size()) throw std::runtime_error("a class points to a missing input source");
        if (std::count(t.classes.begin(), t.classes.end(), t.classes[k]) > 1) throw std::runtime_error("a source is a class twice");
    }
    bool linked = false;
    for (const NeuralLink& l : c.links) linked = linked || std::count(t.classes.begin(), t.classes.end(), l.source) > 0;
    if (!linked) throw std::runtime_error("link one of the class sources to the network");
    const ClassSteps s = class_steps(c);
    const std::vector<int> labels = neural_class_labels(c);
    const ReadoutSplit sp = class_split(c);
    const long long hold = task_hold(c);
    // Нормировка системы-класса только по её кускам в попытках обучения: они у класса первые.
    for (size_t k = 0; k < t.classes.size(); ++k) {
        long long n_train = 0;
        for (size_t i = 0; i < sp.train1; ++i) n_train += labels[i] == (int)k;
        o.sources[(size_t)t.classes[k]].normalize_first = n_train * (s.cycle / hold);
    }
    const bool spike = c.schedule == NeuralSchedule::Spike;
    const int M = (std::max)(1, int_or(t.features_text, 1000));
    // Точки признаков: s.rest + (b + 1) * s.bin внутри каждой попытки, сетка записи кратна обоим.
    const long long pre = std::gcd(s.rest, s.bin);
    o.pre_scaller_text = std::to_string(pre);
    o.order_var = -1;
    o.observed_per_pop_text = spike ? "0" : std::to_string(M);
    o.obs_vars_off.clear();
    for (int k = 0; k < 64; ++k) if (k != c.input_var || spike) o.obs_vars_off.push_back(k);
    const double nets = (double)(std::max<size_t>)(1, neural_sweep_values(c).size()) * (double)(std::max<size_t>)(1, neural_sweep2_values(c).size());
    const double points = (double)labels.size() * (double)s.cycle / (double)pre + 1.0;
    const double mb = spike ? 16.0 : points * M * (double)c.populations.size() * nets * sizeof(double) / (1024.0 * 1024.0) * 1.2 + 16.0;
    o.output_limit_text = fmt_num_shortest((std::max)(num_or(c.output_limit_text, 1024.0), mb));
    return o;
}

std::vector<int> neural_forecast_channels(const NeuralConfig& c) {
    std::vector<int> ch;
    for (const NeuralLink& l : c.links) if (l.source >= 0 && l.source < (int)c.sources.size()) ch.push_back(l.source);
    std::sort(ch.begin(), ch.end());
    ch.erase(std::unique(ch.begin(), ch.end()), ch.end());
    return ch;
}

// Прогноз: запись на отсчётах входа, цель = следующая строка каждого подключённого входа.
static NeuralConfig forecast_config(const NeuralConfig& c) {
    const NeuralTask& t = c.task;
    if (c.protocol != NeuralProtocol::Continuous) throw std::runtime_error("a readout needs the continuous protocol");
    if (num_or(c.transient_text, 0.0) != 0.0) throw std::runtime_error("a readout has its own washout: set the transient to 0");
    if (c.lyap_on) throw std::runtime_error("a readout does not work with the Lyapunov exponent");
    if (c.sweep.enabled) throw std::runtime_error("a forecast drives one network: turn off the sweep");
    if (c.schedule == NeuralSchedule::Spike) throw std::runtime_error("a forecast needs a network of states: a spiking readout cannot drive the input yet");
    const std::vector<int> ch = neural_forecast_channels(c);
    if (ch.empty()) throw std::runtime_error("a forecast needs an input link: the readout replaces the linked inputs");
    if (int_or(t.forecast_length_text, 0) < 1) throw std::runtime_error("the forecast length must be at least 1 sample");
    if (!(num_or(t.forecast_threshold_text, 0.5) > 0.0)) throw std::runtime_error("the error threshold must be positive");
    if (!(num_or(t.lambda_text, 0.0) >= 0.0)) throw std::runtime_error("lambda must not be negative");
    NeuralConfig o = c;
    // Галочки других задач в панели прогноза не видны: от них ничего не зависит.
    o.task.use_saved = false;
    o.task.control_k0 = false;
    o.task.horizon_text = "0";
    const long long hold = task_hold(c);
    const double h = parse_num(c.h_text, 0.01);
    const long long total = std::llround(neural_record_time(c) / h) / hold;
    // Конец обучения на границе отсчёта: с неё второй прогон продолжает уже с выходом на входе.
    o.t_max_text = fmt_num_shortest((double)(total * hold) * h);
    const size_t train1 = task_split(o.task, (size_t)(std::max)(0LL, total)).train1;
    for (int q : ch) {
        NeuralSource& s = o.sources[(size_t)q];
        if (s.kind == NeuralSourceKind::System) { s.normalize = true; s.normalize_first = (long long)train1; }
    }
    const int M = (std::max)(1, int_or(t.features_text, 1000));
    o.pre_scaller_text = std::to_string(hold);
    o.order_var = -1;
    o.observed_per_pop_text = std::to_string(M);
    o.obs_vars_off.clear();
    for (int k = 0; k < 64; ++k) if (k != c.input_var) o.obs_vars_off.push_back(k);
    const double mb = (double)(total + int_or(t.forecast_length_text, 0) + 1) * M * (double)c.populations.size() * sizeof(double) / (1024.0 * 1024.0) * 1.2 + 16.0;
    o.output_limit_text = fmt_num_shortest((std::max)(num_or(c.output_limit_text, 1024.0), mb));
    return o;
}

NeuralConfig neural_task_config(const NeuralConfig& c) {
    NeuralConfig o = c;
    const NeuralTask& t = c.task;
    if (!t.enabled) return o;
    if (t.kind == NeuralTaskKind::Forecast) return forecast_config(c);
    if (t.control_k0 && c.sweep.enabled) throw std::runtime_error("the K = 0 control runs without a sweep: sweep over K instead");
    if (t.control_k0 && t.use_saved && t.kind == NeuralTaskKind::Regression)
        throw std::runtime_error("the K = 0 control trains its own readout: turn off the saved readout");
    if (t.kind == NeuralTaskKind::Classification) return class_config(c);
    const bool delayed = t.kind == NeuralTaskKind::Memory || t.kind == NeuralTaskKind::Xor;
    if (c.protocol != NeuralProtocol::Continuous) throw std::runtime_error("a readout needs the continuous protocol");
    if (num_or(c.transient_text, 0.0) != 0.0) throw std::runtime_error("a readout has its own washout: set the transient to 0");
    if (c.lyap_on) throw std::runtime_error("a readout does not work with the Lyapunov exponent");
    if (t.source < 0 || t.source >= (int)c.sources.size()) throw std::runtime_error("the readout points to a missing input source");
    if (!delayed && c.sources[(size_t)t.source].kind != NeuralSourceKind::System)
        throw std::runtime_error("a readout needs an input source that is a system: targets come from it");
    if (c.links.empty()) throw std::runtime_error("a readout needs an input link");
    if (delayed) {
        bool linked = false;
        for (const NeuralLink& l : c.links) linked = linked || l.source == t.source;
        if (!linked) throw std::runtime_error("link the input source of the task to the network");
        if (int_or(t.max_delay_text, 40) < 1) throw std::runtime_error("the largest delay must be at least 1");
        o.task.use_saved = false;
        o.task.horizon_text = "0";
        o.task.washout_text = std::to_string((std::max)(int_or(t.washout_text, 0), int_or(t.max_delay_text, 40) + 1));
    }
    else {
        if (t.targets.empty()) throw std::runtime_error("pick at least one target variable");
        for (int v : t.targets) if (v < 0 || v >= (int)c.sources[(size_t)t.source].vars.size()) throw std::runtime_error("a target variable is missing");
    }
    const long long hold = task_hold(c);
    const double h = parse_num(c.h_text, 0.01);
    const long long total = std::llround(neural_record_time(c) / h) / hold;
    if (int_or(t.horizon_text, 0) < 0) throw std::runtime_error("the horizon must not be negative");
    const bool spike = c.schedule == NeuralSchedule::Spike;
    const int M = (std::max)(1, int_or(t.features_text, 1000));
    NeuralSource& src = o.sources[(size_t)t.source];
    src.normalize = true;
    if (o.task.use_saved) {
        // Всё, что задаёт признаки и цели, проверяется до расчёта: иначе ошибка выйдет только после него.
        const NeuralReadoutSaved& m = c.readout;
        if (m.empty()) throw std::runtime_error("there is no saved readout: train one and press Keep");
        if (m.spikes != spike) throw std::runtime_error("the saved readout was trained on a different schedule (spikes or states)");
        if (m.source != t.source || m.targets != t.targets) throw std::runtime_error("the input system or the targets differ from the saved readout");
        if (m.horizon != int_or(t.horizon_text, 0)) throw std::runtime_error("the horizon differs from the saved readout");
        if (m.features_requested != M) throw std::runtime_error("the number of feature nodes differs from the saved readout");
        if (!spike && m.feature_var != c.input_var) throw std::runtime_error("the node variable differs from the saved readout");
        if (spike && num_or(t.filter_text, 20.0) != m.filter) throw std::runtime_error("the spike filter time differs from the saved readout");
        if (std::fabs((double)hold * h - m.sample_dt) > 1e-12 * m.sample_dt) throw std::runtime_error("the input sample time differs from the saved readout");
        if (total - (long long)m.horizon <= (long long)int_or(t.washout_text, 0)) throw std::runtime_error("the record is not longer than the washout");
        src.norm_fixed = true;
        src.norm_mean = m.input_mean; src.norm_sd = m.input_sd;
    }
    else src.normalize_first = (long long)task_split(o.task, (size_t)(std::max)(0LL, total - (long long)int_or(o.task.horizon_text, 0))).train1;
    o.pre_scaller_text = std::to_string(hold);
    o.order_var = -1;
    o.observed_per_pop_text = spike ? "0" : std::to_string(M);
    o.obs_vars_off.clear();
    for (int k = 0; k < 64; ++k) if (k != c.input_var || spike) o.obs_vars_off.push_back(k);
    const double nets = (double)(std::max<size_t>)(1, neural_sweep_values(c).size()) * (double)(std::max<size_t>)(1, neural_sweep2_values(c).size());
    const double mb = (double)(total + 1) * M * (double)c.populations.size() * nets * sizeof(double) / (1024.0 * 1024.0) * 1.2 + 16.0;
    o.output_limit_text = fmt_num_shortest((std::max)(num_or(c.output_limit_text, 1024.0), mb));
    return o;
}

// Задержки входа: признак отсчёта p видит только строки до p, как и сеть.
static std::vector<double> task_delays(const NeuralRequest& r, int channel, size_t points, int D) {
    const size_t sub = (size_t)r.input.stage_rows, ch = (size_t)r.input.channels;
    std::vector<double> F(points * (size_t)D, 0.0);
    for (size_t t = 0; t < points; ++t)
        for (int d = 0; d < D; ++d) if ((long long)t - 1 - d >= 0) F[t * D + d] = r.input.samples[(t - 1 - (size_t)d) * sub * ch + (size_t)channel];
    return F;
}

// Классификация: строка признаков на попытку, точка b стимула даёт столбцы i * B + b.
static NeuralTaskResult class_evaluate(const NeuralConfig& c, const NeuralRequest& r, const NeuralResult& res) {
    const NeuralTask& t = c.task;
    NeuralTaskResult out;
    out.classify = true;
    out.higher_better = true;
    out.metric.clear();
    out.names = { "accuracy" };
    const int C = (int)t.classes.size();
    for (int q : t.classes) out.class_names.push_back(c.sources[(size_t)q].name);
    const std::vector<int> labels = neural_class_labels(c);
    const ReadoutSplit sp = class_split(c);
    const ClassSteps s = class_steps(c);
    const size_t n = labels.size(), B = (size_t)s.bins;
    std::vector<double> Y(n * (size_t)C, 0.0);
    for (size_t i = 0; i < n; ++i) Y[i * C + (size_t)labels[i]] = 1.0;
    const bool spike = c.schedule == NeuralSchedule::Spike;
    if (spike && res.spikes_dropped > 0) throw std::runtime_error("the spike log was cut, the features would be wrong");
    const std::vector<int> inst = r.instance_start.empty() ? std::vector<int>{ 0, r.n_nodes } : r.instance_start;
    const int nInst = (int)inst.size() - 1, P = (int)c.populations.size();
    const auto features = [&](int q, std::vector<double>& F) {
        if (spike) {
            // Спайки в отрезке (начало, конец] точки: прошлая попытка в признаки не протекает.
            const int want = (std::max)(1, int_or(t.features_text, 1000));
            int node0 = -1, M = 0;
            for (int p = 0; p < P && node0 < 0; ++p)
                if (!c.populations[(size_t)p].spike_source) {
                    node0 = r.observe.pop_start[(size_t)(q * P + p)];
                    M = (std::min)(want, r.observe.pop_start[(size_t)(q * P + p + 1)] - node0);
                }
            if (node0 < 0 || M < 1) throw std::runtime_error("the readout needs a population that is not a spike source");
            F.assign(n * (size_t)M * B, 0.0);
            for (size_t e = 0; e < res.spike_step.size(); ++e) {
                const int i = res.spike_node[e] - node0;
                if (i < 0 || i >= M) continue;
                const long long at = res.spike_step[e] + 1, tr = (at - 1) / s.cycle, o = at - tr * s.cycle;
                if (tr >= (long long)n || o <= s.rest) continue;
                F[((size_t)tr * M + (size_t)i) * B + (size_t)((o - s.rest - 1) / s.bin)] += 1.0;
            }
            return M;
        }
        const size_t nSel = res.observed.size();
        std::vector<size_t> cols;
        for (size_t i = 0; i < nSel; ++i) if (res.observed[i] >= inst[(size_t)q] && res.observed[i] < inst[(size_t)q + 1]) cols.push_back(i);
        const int M = (int)cols.size();
        if (M < 1) throw std::runtime_error("the network has no observed nodes");
        F.assign(n * (size_t)M * B, 0.0);
        for (size_t tr = 0; tr < n; ++tr)
            for (size_t b = 0; b < B; ++b) {
                const long long step = (long long)tr * s.cycle + s.rest + (long long)(b + 1) * s.bin;
                const size_t p = (size_t)(step / r.pre_scaller);
                if (res.traces.size() < (p + 1) * nSel) throw std::runtime_error("the record is shorter than the trials");
                for (int i = 0; i < M; ++i) F[(tr * M + (size_t)i) * B + b] = res.traces[p * nSel + cols[(size_t)i]];
            }
        return M;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const size_t T = sp.test1 - sp.test0;
    out.instances = nInst;
    out.points = (int)T;
    for (size_t i = sp.test0; i < sp.test1; ++i) out.truth.push_back((double)labels[i]);
    std::vector<size_t> share((size_t)C, 0);
    for (size_t i = sp.test0; i < sp.test1; ++i) ++share[(size_t)labels[i]];
    out.chance = (double)*std::max_element(share.begin(), share.end()) / (double)T;
    out.val.assign((size_t)nInst, nan); out.test.assign((size_t)nInst, nan); out.lambda.assign((size_t)nInst, nan);
    out.predicted.assign((size_t)nInst * T, nan);
    out.confusion.assign((size_t)nInst * C * C, nan);
    const std::vector<double> lambdas = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    const auto score = [&](const ReadoutModel& m, const std::vector<double>& F, double* predicted, double* confusion) {
        const std::vector<int> cls = readout_argmax(readout_predict(m, F, sp.test0, sp.test1), C);
        size_t right = 0;
        for (size_t i = 0; i < T; ++i) {
            right += cls[i] == labels[sp.test0 + i];
            if (predicted) predicted[i] = (double)cls[i];
        }
        if (confusion) {
            std::fill(confusion, confusion + (size_t)C * C, 0.0);
            for (size_t i = 0; i < T; ++i) confusion[(size_t)labels[sp.test0 + i] * C + (size_t)cls[i]] += 1.0;
        }
        return (double)right / (double)T;
    };
    for (int q = 0; q < nInst; ++q) {
        try {
            if (nInst > 1 && q < (int)res.instance_stop.size() && res.instance_stop[(size_t)q] >= 0) throw std::runtime_error("the network diverged");
            std::vector<double> F;
            const int M = features(q, F);
            if (q == 0) out.features = M;
            const ReadoutModel m = readout_fit(F, Y, M * (int)B, C, sp, lambdas, true);
            out.val[(size_t)q] = m.val_accuracy;
            out.lambda[(size_t)q] = m.lambda[0];
            out.test[(size_t)q] = score(m, F, out.predicted.data() + (size_t)q * T, out.confusion.data() + (size_t)q * C * C);
        } catch (const std::exception& e) {
            if (nInst == 1) throw;
            if (out.note.empty()) out.note = "network " + std::to_string(q) + ": " + e.what();
        }
    }
    // Простая модель: тот же ridge на самих отсчётах стимула, до 50 равномерно; канал класса у всех сетей свипа общий.
    const long long hold = r.input.hold_steps, stim_rows = s.stim / hold;
    const int D = (int)(std::min)(stim_rows, 50LL);
    const size_t sub = (size_t)r.input.stage_rows, ch = (size_t)r.input.channels;
    std::vector<double> G(n * (size_t)D);
    for (size_t tr = 0; tr < n; ++tr)
        for (int d = 0; d < D; ++d) {
            const long long row = ((long long)tr * s.cycle + s.rest) / hold + ((long long)(d + 1) * stim_rows) / D - 1;
            G[tr * D + (size_t)d] = r.input.samples[(size_t)row * sub * ch + (size_t)t.classes[0]];
        }
    try {
        const ReadoutModel m = readout_fit(G, Y, D, C, sp, lambdas, true);
        out.delay_test = { score(m, G, nullptr, nullptr) };
        out.delay_length = D;
    } catch (const std::exception&) {}
    return out;
}

NeuralTaskResult neural_task_evaluate(const NeuralConfig& c, const NeuralRequest& r, const NeuralResult& res) {
    if (c.task.kind == NeuralTaskKind::Classification) return class_evaluate(c, r, res);
    if (c.task.kind == NeuralTaskKind::Forecast) throw std::runtime_error("a forecast is evaluated by its own two runs");
    NeuralTask t = c.task;
    NeuralTaskResult out;
    // Memory и Xor: цели из самого входа, задержка d считается от последней строки входа, которую видела точка.
    const bool memory = t.kind == NeuralTaskKind::Memory, xr = t.kind == NeuralTaskKind::Xor, delayed = memory || xr;
    const int D = delayed ? int_or(t.max_delay_text, 40) : 0;
    if (delayed && D < 1) throw std::runtime_error("the largest delay must be at least 1");
    if (delayed) {
        t.use_saved = false;
        t.washout_text = std::to_string((std::max)(int_or(t.washout_text, 0), D + 1));
    }
    const int horizon = delayed ? 0 : int_or(t.horizon_text, 0), K = delayed ? D : (int)t.targets.size();
    // Импульсная сеть точек не пишет, признаки идут из журнала, поэтому отсчёты считаются от длины записи.
    const size_t points = (size_t)(std::llround(r.t_max / r.h) / r.pre_scaller) - (size_t)horizon;
    const NeuralSource& src = c.sources[(size_t)t.source];
    // Шаг источника считается как в fill_input: хаотичному источнику другой последний бит даёт другую траекторию.
    const double dt = (double)r.input.hold_steps * r.h * num_or(src.time_scale_text, 1.0);
    const bool three = r.input.stage_rows == 3;
    std::vector<double> Y(points * (size_t)K, 0.0);
    if (delayed) {
        // Точка p видела строки входа 0..p-1 (pre_scaller = hold), поэтому задержка 1 это строка p - 1.
        const size_t row_stride = (size_t)r.input.stage_rows * (size_t)r.input.channels;
        const auto u = [&](long long row) { return r.input.samples[(size_t)row * row_stride + (size_t)t.source]; };
        for (size_t p = 0; p < points; ++p)
            for (int d = 1; d <= D; ++d) {
                const long long a = (long long)p - d;
                if (a - 1 < 0) continue;
                Y[p * K + (size_t)(d - 1)] = memory ? u(a) : (double)((u(a) > 0.0) != (u(a - 1) > 0.0));
            }
        if (memory) out.names.push_back("MC");
        for (int d = 1; d <= D; ++d) out.names.push_back((memory ? "r2 d=" : "accuracy d=") + std::to_string(d));
        out.by_delay = true;
        out.higher_better = true;
        out.metric.clear();
    }
    else for (int k = 0; k < K; ++k) {
        const size_t need = points + (size_t)horizon;
        const std::vector<double> v = neural_source_samples(src, t.targets[(size_t)k], three ? 2 * need + 1 : need, three ? 0.5 * dt : dt);
        for (size_t q = 0; q < points; ++q) Y[q * K + k] = v[(q + (size_t)horizon) * (three ? 2 : 1)];
        out.names.push_back(src.vars[(size_t)t.targets[(size_t)k]]);
    }
    // Число значений на сеть: у Memory первое это сумма r^2.
    const int KO = (int)out.names.size();
    const bool spike = c.schedule == NeuralSchedule::Spike;
    if (spike && res.spikes_dropped > 0) throw std::runtime_error("the spike log was cut, the features would be wrong");
    const std::vector<int> inst = r.instance_start.empty() ? std::vector<int>{ 0, r.n_nodes } : r.instance_start;
    const int nInst = (int)inst.size() - 1, P = (int)c.populations.size();
    // Признаки экземпляра q и номера их узлов внутри экземпляра: по ним сохранённый выход узнаёт свои узлы.
    const auto features = [&](int q, std::vector<double>& F, std::vector<int>& nodes) {
        nodes.clear();
        if (spike) {
            const int want = (std::max)(1, int_or(t.features_text, 1000));
            int node0 = -1, M = 0;
            for (int p = 0; p < P && node0 < 0; ++p)
                if (!c.populations[(size_t)p].spike_source) {
                    node0 = r.observe.pop_start[(size_t)(q * P + p)];
                    M = (std::min)(want, r.observe.pop_start[(size_t)(q * P + p + 1)] - node0);
                }
            if (node0 < 0 || M < 1) throw std::runtime_error("the readout needs a population that is not a spike source");
            F = readout_spike_features(res.spike_step, res.spike_node, node0, M, 0, r.pre_scaller, (int)points, num_or(t.filter_text, 20.0) / r.h);
            for (int i = 0; i < M; ++i) nodes.push_back(node0 + i - inst[(size_t)q]);
            return M;
        }
        const size_t nSel = res.observed.size();
        if (nSel < 1 || res.traces.size() < points * nSel) throw std::runtime_error("the record has no features");
        std::vector<size_t> cols;
        for (size_t i = 0; i < nSel; ++i)
            if (res.observed[i] >= inst[(size_t)q] && res.observed[i] < inst[(size_t)q + 1]) { cols.push_back(i); nodes.push_back(res.observed[i] - inst[(size_t)q]); }
        const int M = (int)cols.size();
        if (M < 1) throw std::runtime_error("the network has no observed nodes");
        F.resize(points * (size_t)M);
        for (size_t p = 0; p < points; ++p)
            for (int i = 0; i < M; ++i) F[p * (size_t)M + (size_t)i] = res.traces[p * nSel + cols[(size_t)i]];
        return M;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const size_t wash = (size_t)int_or(t.washout_text, 0);
    const ReadoutSplit sp = t.use_saved ? ReadoutSplit{} : task_split(t, points);
    const size_t from = t.use_saved ? wash : sp.test0;
    out.instances = nInst;
    out.applied = t.use_saved;
    out.dt = (double)r.pre_scaller * r.h;
    out.points = (int)(points - from);
    out.t0 = (double)from * out.dt;
    // Кривые по задержкам рядов не хранят: окно рисует значение от d, а ряд на 40 целей весил бы десятки МБ на сеть.
    if (!delayed) out.truth.assign(Y.begin() + (std::ptrdiff_t)(from * K), Y.begin() + (std::ptrdiff_t)(points * K));
    out.val.assign((size_t)nInst * KO, nan); out.test.assign((size_t)nInst * KO, nan); out.lambda.assign((size_t)nInst * KO, nan);
    if (!delayed) out.predicted.assign((size_t)nInst * out.points * K, nan);
    const std::vector<double> lambdas = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    ReadoutModel m0;
    std::vector<int> nodes0;
    for (int q = 0; q < nInst; ++q) {
        // Отказ одной сети свипа (разошлась, молчит) не отменяет остальные: у неё NaN и первая причина в заметке.
        try {
            if (nInst > 1 && q < (int)res.instance_stop.size() && res.instance_stop[(size_t)q] >= 0) throw std::runtime_error("the network diverged");
            std::vector<double> F;
            std::vector<int> nodes;
            const int M = features(q, F, nodes);
            if (q == 0) out.features = M;
            ReadoutModel m;
            if (t.use_saved) {
                const NeuralReadoutSaved& s = c.readout;
                if (nodes != s.nodes) throw std::runtime_error("the feature nodes differ from the saved readout");
                m.M = s.M; m.K = s.K; m.mean = s.mean; m.scale = s.scale; m.w = s.w;
                for (int k = 0; k < K; ++k) out.lambda[(size_t)q * K + k] = s.lambda[(size_t)k];
            }
            else {
                m = readout_fit(F, Y, M, K, sp, lambdas);
                const int o = memory ? 1 : 0;
                for (int k = 0; k < K; ++k) {
                    if (!delayed) out.val[(size_t)q * KO + k] = m.val_nrmse[(size_t)k];
                    out.lambda[(size_t)q * KO + o + k] = m.lambda[(size_t)k];
                }
            }
            const std::vector<double> p = readout_predict(m, F, from, points);
            if (delayed) {
                const double* y = Y.data() + from * K;
                double mc = 0.0;
                for (int k = 0; k < K; ++k) {
                    const double v = memory ? readout_r2(y + k, p.data() + k, (size_t)K, (size_t)out.points)
                                            : readout_bit_accuracy(y + k, p.data() + k, (size_t)K, (size_t)out.points);
                    out.test[(size_t)q * KO + (memory ? 1 : 0) + k] = v;
                    if (std::isfinite(v)) mc += v;
                }
                if (memory) out.test[(size_t)q * KO] = mc;
            }
            else {
                std::copy(p.begin(), p.end(), out.predicted.begin() + (std::ptrdiff_t)((size_t)q * out.points * K));
                for (int k = 0; k < K; ++k)
                    out.test[(size_t)q * K + k] = readout_nrmse(out.truth.data() + k, p.data() + k, (size_t)K, (size_t)out.points);
            }
            if (q == 0) { m0 = std::move(m); nodes0 = std::move(nodes); }
        } catch (const std::exception& e) {
            if (nInst == 1) throw;
            if (out.note.empty()) out.note = "network " + std::to_string(q) + ": " + e.what();
        }
    }
    if (t.use_saved || memory) return out;
    if (xr) {
        // Линейная модель по задержанным входам XOR не решает: её accuracy около 0.5, это и есть контроль нелинейности.
        const std::vector<double> G = task_delays(r, t.source, points, D + 2);
        try {
            const ReadoutModel q = readout_fit(G, Y, D + 2, K, sp, lambdas);
            const std::vector<double> e = readout_predict(q, G, from, points);
            for (int k = 0; k < K; ++k) out.delay_test.push_back(readout_bit_accuracy(Y.data() + from * K + k, e.data() + k, (size_t)K, (size_t)out.points));
            out.delay_length = D + 2;
        } catch (const std::exception&) {}
        return out;
    }
    if (nInst == 1) {
    // Нормировку входа считает fill_input внутри плана; повтор того же ряда по тем же отсчётам даёт те же биты.
    double stats[2] = { 0.0, 1.0 };
    const size_t used = (size_t)src.normalize_first * (three ? 2 : 1);
    source_system_series(src, used, three ? 0.5 * dt : dt, three ? 2 : 1, stats);
    NeuralReadoutSaved& s = out.model;
    s.spikes = spike;
    s.feature_var = c.input_var;
    s.features_requested = (std::max)(1, int_or(t.features_text, 1000));
    s.filter = num_or(t.filter_text, 20.0);
    s.sample_dt = out.dt;
    s.source = t.source; s.horizon = horizon;
    s.targets = t.targets; s.names = out.names;
    s.input_mean = stats[0]; s.input_sd = stats[1];
    s.nodes = nodes0;
    s.M = m0.M; s.K = m0.K; s.mean = m0.mean; s.scale = m0.scale; s.w = m0.w;
    s.lambda = m0.lambda; s.val = m0.val_nrmse; s.test = out.test;
    }
    // Простая модель той же задачи: без неё число ни о чём не говорит. Вход у сетей свипа общий, поэтому она одна.
    double best = std::numeric_limits<double>::infinity();
    for (int D : {1, 5, 10, 20, 50}) {
        if ((size_t)D >= sp.train1 - sp.train0) break;
        const std::vector<double> G = task_delays(r, t.source, points, D);
        const ReadoutModel q = readout_fit(G, Y, D, K, sp, lambdas);
        double v = 0.0;
        for (double e : q.val_nrmse) v += e;
        if (v >= best) continue;
        best = v;
        out.delay_length = D;
        const std::vector<double> e = readout_predict(q, G, sp.test0, sp.test1);
        out.delay_test.clear();
        for (int k = 0; k < K; ++k) out.delay_test.push_back(readout_nrmse(out.truth.data() + k, e.data() + k, (size_t)K, (size_t)out.points));
    }
    return out;
}

// Ошибка прогноза по отсчётам в долях СКО цели на обучении; время верного прогноза = отсчётов до первого превышения порога.
static void forecast_error(const NeuralTaskResult& o, const std::vector<double>& pred, std::vector<double>& e, double& valid) {
    const size_t K = o.names.size(), n = (size_t)o.points;
    e.assign(n, std::numeric_limits<double>::quiet_NaN());
    size_t good = n;
    for (size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < K; ++k) {
            const double d = (pred[j * K + k] - o.truth[j * K + k]) / o.target_sd[k];
            s += d * d;
        }
        e[j] = std::sqrt(s / (double)K);
        if (good == n && !(e[j] <= o.threshold)) good = j;
    }
    // Весь прогноз в пороге: время = вся длина, окно пишет его как нижнюю границу.
    valid = (double)good * o.dt;
}

void neural_forecast_prepare(const NeuralConfig& c, const NeuralRequest& r1, const NeuralResult& res1, NeuralRequest& r2, NeuralTaskResult& out) {
    const NeuralTask& t = c.task;
    const std::vector<int> chans = neural_forecast_channels(c);
    const size_t K = chans.size(), ch = (size_t)r1.input.channels;
    if (K < 1 || ch < 1 || r1.input.stage_rows != 1 || r2.input.stage_rows != 1 || (size_t)r2.input.channels != ch)
        throw std::runtime_error("the forecast needs an input of one row per sample");
    // Точка S видела строки до S - 1: точки 0..S-1 учат выход, точка S даёт первую строку прогноза.
    const size_t S = (size_t)(std::llround(r1.t_max / r1.h) / r1.pre_scaller);
    const size_t L = (size_t)int_or(t.forecast_length_text, 500);
    if (r2.input.samples.size() / ch < S + L + 1) throw std::runtime_error("the forecast run is shorter than the forecast");
    if (res1.observed_vars.size() != 1) throw std::runtime_error("the forecast reads one node variable");
    const size_t nSel = res1.observed.size();
    if (nSel < 1 || res1.traces.size() < (S + 1) * nSel) throw std::runtime_error("the record has no features");
    const int M = (int)nSel;
    const std::vector<double>& F = res1.traces;
    for (size_t i = 0; i < nSel; ++i) if (!std::isfinite(F[S * nSel + i])) throw std::runtime_error("the network diverged before the forecast");
    std::vector<double> Y(S * K);
    for (size_t p = 0; p < S; ++p)
        for (size_t k = 0; k < K; ++k) Y[p * K + k] = r1.input.samples[p * ch + (size_t)chans[k]];
    const ReadoutSplit sp = task_split(t, S);
    const double lam = num_or(t.lambda_text, 0.0);
    const std::vector<double> lambdas = lam > 0.0 ? std::vector<double>{ lam } : std::vector<double>{ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
    const ReadoutModel m = readout_fit(F, Y, M, (int)K, sp, lambdas);

    out = NeuralTaskResult{};
    out.forecast = true;
    out.features = M;
    out.threshold = num_or(t.forecast_threshold_text, 0.5);
    out.lyapunov = num_or(t.lyapunov_text, 0.0);
    for (size_t k = 0; k < K; ++k) {
        const NeuralSource& s = c.sources[(size_t)chans[k]];
        out.names.push_back(s.name);
        if (k == 0 || s.kind == NeuralSourceKind::System) out.time_scale = num_or(s.time_scale_text, 1.0);
    }
    // Под настоящим входом, на один отсчёт вперёд: с этим числом сравнивается прогноз.
    const size_t nt = sp.test1 - sp.test0;
    const std::vector<double> one = readout_predict(m, F, sp.test0, sp.test1);
    for (size_t k = 0; k < K; ++k) {
        out.val.push_back(m.val_nrmse[k]);
        out.test.push_back(readout_nrmse(Y.data() + sp.test0 * K + k, one.data() + k, K, nt));
        out.lambda.push_back(m.lambda[k]);
        double mu = 0.0, v = 0.0;
        const size_t n = sp.train1 - sp.train0;
        for (size_t p = sp.train0; p < sp.train1; ++p) mu += Y[p * K + k];
        mu /= (double)n;
        for (size_t p = sp.train0; p < sp.train1; ++p) v += (Y[p * K + k] - mu) * (Y[p * K + k] - mu);
        out.target_sd.push_back(v > 0.0 ? std::sqrt(v / (double)n) : 1.0);
    }

    // Настоящее продолжение: строки S..S+L второго запроса, пока S не подменена прогнозом.
    out.dt = (double)r1.pre_scaller * r1.h;
    out.t0 = (double)S * out.dt;
    out.points = (int)(L + 1);
    out.truth.resize((L + 1) * K);
    for (size_t j = 0; j <= L; ++j)
        for (size_t k = 0; k < K; ++k) out.truth[j * K + k] = r2.input.samples[(S + j) * ch + (size_t)chans[k]];
    const std::vector<double> first = readout_predict(m, F, S, S + 1);
    out.predicted.assign((L + 1) * K, std::numeric_limits<double>::quiet_NaN());
    for (size_t k = 0; k < K; ++k) {
        out.predicted[k] = first[k];
        r2.input.samples[S * ch + (size_t)chans[k]] = first[k];
    }
    // Остальные строки считает выход на GPU: нормировка признаков свёрнута в веса.
    NeuralFeedback& fb = r2.feedback;
    fb.from_row = (long long)S + 1;
    fb.cols.resize((size_t)M);
    std::iota(fb.cols.begin(), fb.cols.end(), 0);
    fb.channels = chans;
    fb.w.assign(K * (size_t)(M + 1), 0.0);
    for (size_t k = 0; k < K; ++k) {
        const double* w = m.w.data() + k * (size_t)(M + 1);
        double b = w[M];
        for (int i = 0; i < M; ++i) {
            const double a = w[i] / m.scale[(size_t)i];
            fb.w[k * (size_t)(M + 1) + (size_t)i] = a;
            if (a != 0.0) b -= a * m.mean[(size_t)i];
        }
        fb.w[k * (size_t)(M + 1) + (size_t)M] = b;
    }

    // Сравнения выбирают lambda сами: поле lambda подбирают под сеть. Квадратичным признакам гладкого сигнала нужна и 1e-10.
    const std::vector<double> base_lambdas = { 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
    // Авторегрессия по прошлым строкам всех целей: порядок по валидации на один отсчёт, затем тот же замкнутый режим.
    const auto ar_row = [&](const std::vector<double>& rows, size_t p, int D, double* g) {
        for (size_t k = 0; k < K; ++k)
            for (int d = 1; d <= D; ++d) g[k * (size_t)D + (size_t)(d - 1)] = p >= (size_t)d ? rows[(p - (size_t)d) * K + k] : 0.0;
    };
    double best = std::numeric_limits<double>::infinity();
    ReadoutModel ar;
    for (int D : {1, 2, 5, 10, 20, 50}) {
        if ((size_t)D * K + 1 >= sp.train1 - sp.train0) break;
        std::vector<double> G(S * K * (size_t)D);
        for (size_t p = 0; p < S; ++p) ar_row(Y, p, D, G.data() + p * K * (size_t)D);
        try {
            ReadoutModel q = readout_fit(G, Y, D * (int)K, (int)K, sp, base_lambdas);
            double v = 0.0;
            for (double e : q.val_nrmse) v += e;
            if (!(v < best)) continue;
            best = v;
            ar = std::move(q);
            out.ar_order = D;
        } catch (const std::exception&) {}
    }
    if (out.ar_order > 0) {
        const int D = out.ar_order;
        std::vector<double> rows(Y);
        rows.resize((S + L + 1) * K);
        std::vector<double> g(K * (size_t)D);
        out.ar_predicted.resize((L + 1) * K);
        for (size_t j = 0; j <= L; ++j) {
            ar_row(rows, S + j, D, g.data());
            const std::vector<double> y = readout_predict(ar, g, 0, 1);
            for (size_t k = 0; k < K; ++k) rows[(S + j) * K + k] = out.ar_predicted[j * K + k] = y[k];
        }
        forecast_error(out, out.ar_predicted, out.ar_error, out.ar_valid_time);
    }

    // NVAR: линейная часть как у авторегрессии, к ней все попарные произведения. Цель = приращение строки:
    // приращение мало и гладко, а вес 1 на прошлом отсчёте ridge не прижимает.
    const auto nvar_size = [K](int D) { const size_t n = K * (size_t)D; return n + n * (n + 1) / 2; };
    const auto nvar_row = [&](const std::vector<double>& rows, size_t p, int D, double* g) {
        const size_t n = K * (size_t)D;
        ar_row(rows, p, D, g);
        size_t at = n;
        for (size_t a = 0; a < n; ++a)
            for (size_t b = a; b < n; ++b) g[at++] = g[a] * g[b];
    };
    std::vector<double> dY(S * K, 0.0);
    for (size_t p = 1; p < S; ++p)
        for (size_t k = 0; k < K; ++k) dY[p * K + k] = Y[p * K + k] - Y[(p - 1) * K + k];
    best = std::numeric_limits<double>::infinity();
    ReadoutModel nv;
    for (int D : {1, 2, 3, 4}) {
        const size_t Mn = nvar_size(D);
        if (Mn + 1 >= sp.train1 - sp.train0) break;
        std::vector<double> G(S * Mn);
        for (size_t p = 0; p < S; ++p) nvar_row(Y, p, D, G.data() + p * Mn);
        try {
            ReadoutModel q = readout_fit(G, dY, (int)Mn, (int)K, sp, base_lambdas);
            double v = 0.0;
            for (double e : q.val_nrmse) v += e;
            if (!(v < best)) continue;
            best = v;
            nv = std::move(q);
            out.nvar_order = D;
        } catch (const std::exception&) {}
    }
    if (out.nvar_order > 0 && S >= 1) {
        const int D = out.nvar_order;
        std::vector<double> rows(Y);
        rows.resize((S + L + 1) * K);
        std::vector<double> g(nvar_size(D));
        out.nvar_predicted.resize((L + 1) * K);
        for (size_t j = 0; j <= L; ++j) {
            const size_t p = S + j;
            nvar_row(rows, p, D, g.data());
            const std::vector<double> d = readout_predict(nv, g, 0, 1);
            for (size_t k = 0; k < K; ++k) {
                // Разошедшийся прогноз дальше не считается: квадраты переполнили бы double, порог он уже перешёл.
                const double y = rows[(p - 1) * K + k] + d[k];
                rows[p * K + k] = out.nvar_predicted[j * K + k] = std::isfinite(y) && std::fabs(y) < 1e100 ? y : std::numeric_limits<double>::quiet_NaN();
            }
        }
        forecast_error(out, out.nvar_predicted, out.nvar_error, out.nvar_valid_time);
    }
}

void neural_forecast_finish(const NeuralResult& res2, NeuralTaskResult& out) {
    const size_t K = out.names.size(), L = (size_t)out.points - 1;
    for (size_t j = 1; j <= L && j * K <= res2.feedback_rows.size(); ++j)
        for (size_t k = 0; k < K; ++k) out.predicted[j * K + k] = res2.feedback_rows[(j - 1) * K + k];
    forecast_error(out, out.predicted, out.error, out.valid_time);
}

void neural_checkpoint_save(const std::string& path, const NeuralCheckpoint& c) {
    std::ofstream f(path, std::ios::binary);
    const uint64_t step = (uint64_t)c.step, shape = c.shape.size(), size = c.data.size();
    f.write(kNeuralCheckpointMagic, 8);
    f.write((const char*)&step, 8); f.write((const char*)&shape, 8); f.write((const char*)&size, 8);
    f.write(c.shape.data(), (std::streamsize)shape);
    f.write(c.data.data(), (std::streamsize)size);
    if (!f) throw std::runtime_error("could not write " + path);
}

std::shared_ptr<NeuralCheckpoint> neural_checkpoint_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8] = {};
    uint64_t step = 0, shape = 0, size = 0;
    f.read(magic, 8);
    f.read((char*)&step, 8); f.read((char*)&shape, 8); f.read((char*)&size, 8);
    if (!f || std::memcmp(magic, kNeuralCheckpointMagic, 8) != 0) throw std::runtime_error(path + " is not a checkpoint");
    auto c = std::make_shared<NeuralCheckpoint>();
    c->step = (long long)step;
    c->shape.resize((size_t)shape);
    c->data.resize((size_t)size);
    f.read(c->shape.data(), (std::streamsize)shape);
    f.read(c->data.data(), (std::streamsize)size);
    if (!f) throw std::runtime_error(path + " is cut short");
    return c;
}

void neural_merge_resumed(const NeuralResult& before, NeuralResult& after) {
    const auto fill = [](const std::vector<double>& from, std::vector<double>& to) {
        for (size_t k = 0; k < from.size() && k < to.size(); ++k) if (std::isnan(to[k])) to[k] = from[k];
    };
    fill(before.traces, after.traces);
    fill(before.means, after.means);
    fill(before.spread, after.spread);
    fill(before.weight_trace, after.weight_trace);
    fill(before.device_trace, after.device_trace);
    after.spike_step.insert(after.spike_step.begin(), before.spike_step.begin(), before.spike_step.end());
    after.spike_node.insert(after.spike_node.begin(), before.spike_node.begin(), before.spike_node.end());
    after.spikes_dropped += before.spikes_dropped;
}

std::string neural_spike_csv(const NeuralConfig& c) {
    std::ostringstream o;
    o.precision(17);
    o << "step,time,node\n";
    const NeuralResult& r = c.result;
    for (size_t e = 0; e < r.spike_step.size(); ++e) o << r.spike_step[e] << "," << (double)(r.spike_step[e] + 1) * r.h << "," << r.spike_node[e] << "\n";
    return o.str();
}

std::string neural_sweep_csv(const NeuralConfig& c) {
    std::ostringstream o;
    if (c.sweep_values.empty() || c.pops_per_instance < 1) return o.str();
    o.precision(17);
    const int K = c.pops_per_instance, n2 = (std::max)(1, (int)c.sweep2_values.size()), nInst = (int)c.result.instance_stop.size();
    const bool two = !c.sweep2_values.empty(), spikes = c.schedule == NeuralSchedule::Spike;
    const NeuralTaskResult& task = c.task_result;
    const int T = task.instances == nInst ? (int)task.names.size() : 0;
    o << neural_sweep_name(c.sweep.kind) << (two ? std::string(",") + neural_sweep_name(c.sweep2.kind) : std::string())
      << ",population," << (spikes ? "rate" : "R,spread,lyapunov") << ",stop_step";
    for (int k = 0; k < T; ++k) {
        std::string col = task.metric.empty() ? task.names[(size_t)k] : "nrmse_" + task.names[(size_t)k];
        std::replace(col.begin(), col.end(), ' ', '_');
        o << "," << col;
    }
    o << "\n";
    for (int q = 0; q < nInst; ++q) {
        for (int p = 0; p < K; ++p) {
            const size_t at = (size_t)q * K + p;
            const auto cell = [&o](const std::vector<double>& v, size_t i) {
                o << ",";
                if (i < v.size() && !std::isnan(v[i])) o << v[i];
            };
            o << c.sweep_values[(size_t)(q / n2)];
            if (two) o << "," << c.sweep2_values[(size_t)(q % n2)];
            o << "," << c.populations[(size_t)p].name;
            if (spikes) cell(c.pop_rate, at);
            else {
                cell(c.sweep_order, at);
                cell(c.sweep_spread, at);
                cell(c.sweep_lyap, (size_t)q);
            }
            o << "," << c.result.instance_stop[(size_t)q];
            for (int k = 0; k < T; ++k) cell(task.test, (size_t)q * T + k);
            o << "\n";
        }
    }
    return o.str();
}

// Подписи кривых Neural Weights: дуга или устройство как популяция[номер], номера внутри популяции.
static std::vector<std::string> neural_watch_labels(const NeuralConfig& c, const NeuralRequest& r) {
    const std::vector<int>& ps = r.observe.pop_start;
    const auto node = [&](int i) {
        const int q = (int)(std::upper_bound(ps.begin(), ps.end(), i) - ps.begin()) - 1;
        return c.populations[(size_t)q].name + "[" + std::to_string(i - ps[(size_t)q]) + "]";
    };
    std::vector<std::string> out;
    for (int e : r.spike.watch_arcs) {
        const int dst = (int)(std::upper_bound(r.edge_start.begin(), r.edge_start.end(), e) - r.edge_start.begin()) - 1;
        out.push_back(node(r.edge_src[(size_t)e]) + " -> " + node(dst));
    }
    for (int d : r.devices.watch) out.push_back(node(r.devices.a[(size_t)d]) + " - " + node(r.devices.b[(size_t)d]));
    return out;
}

// Вторая часть метки контрольной точки: продолжение пишет точки по этой же сетке, иначе склейка с прошлой записью неверна.
static std::string neural_record_tag(const NeuralRequest& r) {
    Fnv f;
    for (int v : r.observe.nodes) f.add((uint64_t)(uint32_t)v);
    for (int v : r.observe.vars) f.add((uint64_t)(uint32_t)v);
    for (int v : r.spike.watch_arcs) f.add((uint64_t)(uint32_t)v);
    for (int v : r.devices.watch) f.add((uint64_t)(uint32_t)v);
    std::ostringstream o;
    o.precision(17);
    o << "|" << r.h << ":" << r.transient << ":" << r.pre_scaller << ":" << hex64(f.h);
    return o.str();
}

NeuralPlan neural_plan(const System& sys, const std::vector<CustomScheme>& custom_schemes,
                       const std::vector<std::string>& vars, const std::vector<std::string>& params,
                       const NeuralConfig& c) {
    NeuralPlan plan;
    plan.pops_per_instance = (int)c.populations.size();
    plan.sweep_values = neural_sweep_values(c);
    plan.sweep2_values = neural_sweep2_values(c);
    // Вход для окна: не больше 4096 точек на канал, каналы первого экземпляра.
    const auto take_view = [&](const NeuralRequest& r) {
        if (r.input.channels <= 0) return;
        const int ch = (int)c.sources.size();
        // Только запись, без transient: окно сравнивается с остальными по одной оси времени.
        const size_t sub = (size_t)r.input.stage_rows, rows = r.input.samples.size() / (size_t)r.input.channels / sub;
        const long long skip = std::llround(r.transient / r.h);
        const size_t first = (std::min)(rows, (size_t)((skip + r.input.hold_steps - 1) / r.input.hold_steps));
        const size_t step = (std::max<size_t>)(1, (rows - first) / 4096);
        plan.input_view_t0 = (double)first * (double)r.input.hold_steps * r.h;
        plan.input_view_dt = (double)step * (double)r.input.hold_steps * r.h;
        for (size_t row = first; row < rows; row += step)
            for (int q = 0; q < ch; ++q) plan.input_view.push_back(r.input.samples[row * sub * r.input.channels + q]);
    };
    if (plan.sweep_values.empty()) {
        const NeuralGraph g = neural_build_graph(c);
        plan.request = neural_build_request(sys, custom_schemes, vars, params, c, g);
        take_view(plan.request);
        plan.hash = g.hash;
        plan.instance_hashes = { g.hash };
        plan.arcs = (long long)g.src.size();
        plan.degree_min = g.degree_min; plan.degree_max = g.degree_max;
        plan.watched_per_instance = (int)(plan.request.spike.watch_arcs.size() + plan.request.devices.watch.size() * (size_t)plan.request.devices.states);
        plan.watch_labels = neural_watch_labels(c, plan.request);
        plan.request.checkpoint_tag = plan.hash + neural_record_tag(plan.request);
        return plan;
    }
    // Экземпляры подряд: номера узлов и популяций сдвигаются, дуга наружу экземпляра появиться не может.
    NeuralRequest& a = plan.request;
    Fnv f;
    long long nodes = 0;
    std::vector<std::vector<double>> input_blocks;   // разные отсчёты у экземпляров: каналы идут подряд
    plan.degree_min = std::numeric_limits<int>::max();
    const int n2 = (std::max)(1, (int)plan.sweep2_values.size()), nTotal = (int)plan.sweep_values.size() * n2;
    for (int i = 0; i < nTotal; ++i) {
        NeuralConfig one = neural_sweep_instance(c, i / n2, plan.sweep_values[(size_t)(i / n2)]);
        if (!plan.sweep2_values.empty()) sweep_apply(one, c.sweep2, i % n2, plan.sweep2_values[(size_t)(i % n2)]);
        const NeuralGraph g = neural_build_graph(one);
        NeuralRequest r = neural_build_request(sys, custom_schemes, vars, params, one, g);
        if (nodes + r.n_nodes > 2000000000LL) throw std::runtime_error("the sweep holds too many nodes");
        plan.arcs += (long long)r.edge_src.size();
        if (plan.arcs > kMaxArcs) throw std::runtime_error("the sweep would hold more than " + std::to_string(kMaxArcs) + " arcs");
        const int off = (int)nodes, arc_off = (int)a.edge_src.size();
        for (const char ch : g.hash) f.add((uint64_t)(uint8_t)ch);
        plan.instance_hashes.push_back(g.hash);
        plan.degree_min = (std::min)(plan.degree_min, g.degree_min);
        plan.degree_max = (std::max)(plan.degree_max, g.degree_max);
        const unsigned long long lyap_seed = r.lyapunov.seed;
        if (i == 0) {
            plan.watched_per_instance = (int)(r.spike.watch_arcs.size() + r.devices.watch.size() * (size_t)r.devices.states);
            plan.watch_labels = neural_watch_labels(c, r);
            a = std::move(r);
            a.instance_start = { 0, a.n_nodes };
            a.lyapunov.instance_seed = { lyap_seed };
            nodes = a.n_nodes;
            take_view(a);
            input_blocks.push_back(a.input.samples);
            continue;
        }
        a.lyapunov.instance_seed.push_back(lyap_seed);
        if (r.input.channels > 0) {
            a.input.node_gain.insert(a.input.node_gain.end(), r.input.node_gain.begin(), r.input.node_gain.end());
            int shift = -1;
            for (size_t q = 0; q < input_blocks.size() && shift < 0; ++q)
                if (input_blocks[q] == r.input.samples) shift = (int)q * r.input.channels;
            if (shift < 0) { shift = (int)input_blocks.size() * r.input.channels; input_blocks.push_back(r.input.samples); }
            for (int v : r.input.node_channel) a.input.node_channel.push_back(v < 0 ? -1 : v + shift);
        }
        a.values.insert(a.values.end(), r.values.begin(), r.values.end());
        a.initial_conditions.insert(a.initial_conditions.end(), r.initial_conditions.begin(), r.initial_conditions.end());
        a.edge_start.pop_back();
        for (int v : r.edge_start) a.edge_start.push_back(v + arc_off);
        for (int v : r.edge_src) a.edge_src.push_back(v + off);
        a.edge_w.insert(a.edge_w.end(), r.edge_w.begin(), r.edge_w.end());
        a.edge_law.insert(a.edge_law.end(), r.edge_law.begin(), r.edge_law.end());
        const auto append = [](auto& to, const auto& from) { to.insert(to.end(), from.begin(), from.end()); };
        append(a.spike.edge_delay, r.spike.edge_delay);
        append(a.spike.refractory_steps, r.spike.refractory_steps);
        append(a.spike.source_p, r.spike.source_p);
        append(a.spike.source_from, r.spike.source_from);
        append(a.spike.source_to, r.spike.source_to);
        append(a.spike.source_seed, r.spike.source_seed);
        // У каждого экземпляра свои правила: значение свипа может их менять.
        const int rule_off = (int)a.spike.rules.size(), dev_off = (int)a.devices.a.size();
        append(a.spike.rules, r.spike.rules);
        for (int v : r.spike.edge_rule) a.spike.edge_rule.push_back(v < 0 ? -1 : v + rule_off);
        for (int v : r.spike.watch_arcs) a.spike.watch_arcs.push_back(v + arc_off);
        for (int v : r.devices.a) a.devices.a.push_back(v + off);
        for (int v : r.devices.b) a.devices.b.push_back(v + off);
        append(a.devices.var, r.devices.var);
        append(a.devices.two_way, r.devices.two_way);
        append(a.devices.gain, r.devices.gain);
        append(a.devices.values, r.devices.values);
        append(a.devices.initial, r.devices.initial);
        for (int v : r.devices.watch) a.devices.watch.push_back(v + dev_off);
        a.observe.pop_start.pop_back();
        for (int v : r.observe.pop_start) a.observe.pop_start.push_back(v + off);
        for (int v : r.observe.nodes) a.observe.nodes.push_back(v + off);
        nodes += r.n_nodes;
        a.n_nodes = (int)nodes;
        a.instance_start.push_back(a.n_nodes);
    }
    if (input_blocks.size() > 1) {
        const int ch = a.input.channels, K = (int)input_blocks.size();
        const size_t rows = input_blocks[0].size() / (size_t)ch;
        std::vector<double>& out = a.input.samples;
        out.assign(rows * (size_t)ch * K, 0.0);
        for (int q = 0; q < K; ++q)
            for (size_t row = 0; row < rows; ++row)
                for (int m = 0; m < ch; ++m) out[(row * K + q) * ch + m] = input_blocks[(size_t)q][row * ch + m];
        a.input.channels = ch * K;
    }
    plan.hash = hex64(f.h);
    plan.request.checkpoint_tag = plan.hash + neural_record_tag(plan.request);
    return plan;
}

void NeuralSession::load_from_record(const SystemRecord& r, const std::vector<std::string>& vars_,
                                     const std::vector<std::string>& params_) {
    vars = vars_; params = params_;
    enabled_builtin_schemes = enabled_builtins_from_record(r);
    NeuralConfig c;
    c.label = "Experiment 1";
    c.h_text = default_h_from_record(r);
    c.scheme = default_scheme_from_record(r, c.scheme);
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    c.cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    c.newton = newton_from_record(r);
    c.laws.push_back(net_default_law(vars));
    c.populations.push_back(neural_default_population(&r, vars, params));
    c.populations[0].name = "A";
    NeuralProjection p;
    c.projections.push_back(p);
    configs.clear();
    configs.push_back(std::move(c));
    active_config_index = 0;
    running_config_index = -1;
}

void NeuralSession::add_config() {
    NeuralConfig c;
    if (!configs.empty()) {
        c = configs.back();
        c.result = NeuralResult{};
        c.instance_start.clear(); c.sweep_values.clear(); c.sweep2_values.clear(); c.sweep_order.clear(); c.sweep_spread.clear();
        c.sweep_lyap.clear(); c.input_view.clear(); c.pop_rate.clear(); c.weight_summary.clear();
        c.task_result = NeuralTaskResult{};
        c.checkpoint.reset();
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.graph_hash.clear(); c.graph_recipe.clear();
    } else {
        c.laws.push_back(net_default_law(vars));
        c.populations.push_back(neural_default_population(nullptr, vars, params));
    }
    c.label = "Experiment " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
}

void NeuralSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size() || configs.size() == 1) return;
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    if (running_config_index > i) --running_config_index;
    if (active_config_index >= (int)configs.size()) active_config_index = (int)configs.size() - 1;
}

bool NeuralSession::run_async(ParametricEngine& engine, int config_idx, bool resume) {
    if (in_flight || config_idx < 0 || config_idx >= (int)configs.size()) return false;
    NeuralConfig& c = configs[(size_t)config_idx];
    if (resume && !c.checkpoint) return false;
    c.last_error.clear();
    c.hash_mismatch = false;
    last_run_label.clear();

    NeuralConfig snap = c;
    snap.result = NeuralResult{};
    cancel_token = std::make_shared<std::atomic<bool>>(false);
    progress_token = std::make_shared<std::atomic<float>>(0.0f);
    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();

    // Граф на 10 млн дуг строится секунды: всё это в рабочем потоке, а не в кадре.
    run_future = std::async(std::launch::async,
        [&engine, snap = std::move(snap), sys_copy = sys, cs = custom_schemes, v = vars, q = params, resume,
         cancel = cancel_token, progress = progress_token]() {
        NeuralOutcome out;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            const NeuralConfig job = neural_task_config(snap);
            NeuralPlan plan = neural_plan(sys_copy, cs, v, q, job);
            out.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            out.hash = plan.hash;
            out.recipe = neural_graph_recipe(snap);
            out.arcs = plan.arcs;
            out.degree_min = plan.degree_min; out.degree_max = plan.degree_max;
            if (!snap.graph_hash.empty() && snap.graph_recipe == out.recipe && snap.graph_hash != plan.hash) {
                out.hash_mismatch = true;
                out.result.error = "this recipe no longer gives the saved graph (hash " + snap.graph_hash + ", now " + plan.hash
                                 + "): the generator or its library changed";
                return out;
            }
            plan.request.cancel = cancel; plan.request.progress = progress;
            if (resume) { plan.request.resume = snap.checkpoint; plan.request.observe.order_var = -1; }
            // Признаки и цели выхода берутся от начала записи, у продолжения его нет.
            if (job.task.enabled && resume) throw std::runtime_error("a readout does not work with Continue yet");
            if (job.task.enabled && job.task.use_saved)
                for (size_t q = 0; q < plan.instance_hashes.size(); ++q)
                    if (plan.instance_hashes[q] != job.readout.graph_hash)
                        throw std::runtime_error("the saved readout was trained on another graph (hash " + job.readout.graph_hash + ", now "
                                                 + plan.instance_hashes[q] + (plan.instance_hashes.size() > 1 ? " in network " + std::to_string(q) : std::string()) + ")");
            out.resumed = resume;
            out.watched_per_instance = plan.watched_per_instance;
            out.watch_labels = plan.watch_labels;
            const bool forecast = job.task.enabled && job.task.kind == NeuralTaskKind::Forecast;
            // Прогноз продолжает первый прогон с его конца: снимок нужен всегда, а не только по флажку.
            const bool keep = plan.request.keep_checkpoint;
            if (forecast) plan.request.keep_checkpoint = true;
            out.result = engine.run_neural(plan.request);
            if (forecast && out.result.ok) {
                NeuralConfig longer = job;
                const double h = plan.request.h;
                const long long steps = std::llround(plan.request.t_max / h) + (long long)int_or(job.task.forecast_length_text, 500) * plan.request.pre_scaller;
                longer.t_max_text = fmt_num_shortest((double)steps * h);
                NeuralPlan p2 = neural_plan(sys_copy, cs, v, q, longer);
                if (p2.hash != plan.hash) throw std::runtime_error("the forecast run built another graph");
                p2.request.cancel = cancel; p2.request.progress = progress;
                p2.request.resume = out.result.checkpoint;
                p2.request.keep_checkpoint = keep;
                neural_forecast_prepare(job, plan.request, out.result, p2.request, out.task_result);
                NeuralResult r2 = engine.run_neural(p2.request);
                if (r2.ok) {
                    neural_forecast_finish(r2, out.task_result);
                    neural_merge_resumed(out.result, r2);
                }
                else if (!r2.cancelled) r2.error = "the forecast run failed: " + r2.error;
                out.result = std::move(r2);
            }
            out.instance_start = plan.request.instance_start;
            if (out.instance_start.empty()) out.instance_start = { 0, plan.request.n_nodes };
            out.pops_per_instance = plan.pops_per_instance;
            out.pop_start = plan.request.observe.pop_start;
            if (!out.result.spike_count.empty() && out.result.ok) {
                const double span = (double)out.result.steps_done * out.result.h - out.result.t0;
                for (size_t p = 0; p + 1 < out.pop_start.size(); ++p) {
                    long long sum = 0;
                    for (int i = out.pop_start[p]; i < out.pop_start[p + 1]; ++i) sum += out.result.spike_count[(size_t)i];
                    const int K = out.pops_per_instance, q = K > 0 ? (int)p / K : 0;
                    const bool stopped = q < (int)out.result.instance_stop.size() && out.result.instance_stop[(size_t)q] >= 0;
                    out.pop_rate.push_back(stopped ? std::numeric_limits<double>::quiet_NaN()
                                                   : span > 0.0 ? (double)sum / (double)(out.pop_start[p + 1] - out.pop_start[p]) / span : 0.0);
                }
            }
            if (out.result.ok && !plan.request.spike.edge_rule.empty()) {
                const std::vector<int>& inst = out.instance_start;
                for (size_t q2 = 0; q2 + 1 < inst.size(); ++q2) {
                    double lo = std::numeric_limits<double>::infinity(), hi = -lo, sum = 0.0;
                    long long n = 0;
                    for (int e = plan.request.edge_start[(size_t)inst[q2]]; e < plan.request.edge_start[(size_t)inst[q2 + 1]]; ++e) {
                        if (plan.request.spike.edge_rule[(size_t)e] < 0) continue;
                        const double w = out.result.final_weights[(size_t)e];
                        lo = (std::min)(lo, w); hi = (std::max)(hi, w); sum += w; ++n;
                    }
                    out.weight_summary.push_back(lo); out.weight_summary.push_back(n ? sum / (double)n : 0.0); out.weight_summary.push_back(hi);
                }
            }
            if (out.result.ok && job.task.enabled && !forecast) {
                out.task_result = neural_task_evaluate(job, plan.request, out.result);
                if (!out.task_result.applied) out.task_result.model.graph_hash = plan.hash;
                if (job.task.control_k0) {
                    // Та же сеть, тот же вход и те же узлы, только K = 0: разница с основным прогоном это вклад рекуррентных связей.
                    NeuralConfig zero = job;
                    zero.coupling_text = "0";
                    NeuralPlan p0 = neural_plan(sys_copy, cs, v, q, zero);
                    p0.request.cancel = cancel; p0.request.progress = progress;
                    const NeuralResult r0 = engine.run_neural(p0.request);
                    if (r0.cancelled) out.task_result.note = "the K = 0 control was stopped";
                    else if (!r0.ok) out.task_result.note = "the K = 0 control failed: " + r0.error;
                    else {
                        const NeuralTaskResult t0 = neural_task_evaluate(zero, p0.request, r0);
                        out.task_result.control_test = t0.test;
                        out.task_result.control_val = t0.val;
                    }
                }
            }
            out.sweep_values = std::move(plan.sweep_values);
            out.sweep2_values = std::move(plan.sweep2_values);
            out.sweep_lyap = out.result.lyapunov;
            out.input_view = std::move(plan.input_view);
            out.input_view_t0 = plan.input_view_t0;
            out.input_view_dt = plan.input_view_dt;
            neural_sweep_summary(out, plan.request.observe.order_var);
        } catch (const std::exception& e) {
            // Обучение выхода идёт после удачного расчёта, его отказ не должен выглядеть успехом.
            out.result.ok = false;
            out.result.error = e.what();
        }
        return out;
    });
    return true;
}

void NeuralSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}

bool NeuralSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false; running_config_index = -1;
        cancel_token.reset(); progress_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    const int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[(size_t)idx].label.empty())
                            ? configs[(size_t)idx].label : std::string("network");
    NeuralOutcome out = run_future.get();
    const bool cancelled = out.result.cancelled;
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        NeuralConfig& c = configs[(size_t)idx];
        c.last_run_ok = out.result.ok;
        c.last_error = out.result.ok ? std::string() : out.result.error;
        c.hash_mismatch = out.hash_mismatch;
        if (!out.hash.empty() && !out.hash_mismatch) {
            c.graph_hash = out.hash; c.graph_recipe = out.recipe;
            c.model_text = neural_model_text(sys);
            c.arcs = out.arcs; c.degree_min = out.degree_min; c.degree_max = out.degree_max;
            c.build_seconds = out.build_seconds;
        }
        if (out.result.ok) {
            if (out.resumed) neural_merge_resumed(c.result, out.result);
            c.result = std::move(out.result);
            c.checkpoint = c.result.checkpoint;
            c.result.checkpoint.reset();
            c.watched_per_instance = out.watched_per_instance;
            c.watch_labels = std::move(out.watch_labels);
            c.weight_summary = std::move(out.weight_summary);
            c.task_result = std::move(out.task_result);
            c.instance_start = std::move(out.instance_start);
            c.sweep_values = std::move(out.sweep_values);
            c.sweep2_values = std::move(out.sweep2_values);
            c.sweep_order = std::move(out.sweep_order);
            c.sweep_spread = std::move(out.sweep_spread);
            c.sweep_lyap = std::move(out.sweep_lyap);
            c.input_view = std::move(out.input_view);
            c.input_view_t0 = out.input_view_t0;
            c.input_view_dt = out.input_view_dt;
            c.pops_per_instance = out.pops_per_instance;
            c.pop_start = std::move(out.pop_start);
            c.pop_rate = std::move(out.pop_rate);
            ++c.data_generation;
            const bool spikes = c.schedule == NeuralSchedule::Spike && c.sweep_values.empty();
            if (c.csv_save_enabled && !c.csv_output_path.empty() && (spikes || !c.sweep_values.empty())) {
                std::ofstream f(c.csv_output_path, std::ios::binary);
                f << (spikes ? neural_spike_csv(c) : neural_sweep_csv(c));
                if (!f) c.last_error = "could not write " + c.csv_output_path;
            }
        }
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_neural_run(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    return true;
}
