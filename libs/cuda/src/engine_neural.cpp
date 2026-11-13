#include "engine_impl.h"
using namespace ucuda::cuda_detail;

// Neural — одна сеть на много блоков (resources/kernels/neural.template.cu). Перенесено из
// U-CUDA/parametric_engine.cpp (main, d30f0ac) без изменений в телах.

namespace {

// Neural: блок и группа средних подобраны замерами; синхронизация по числу обновлений узлов и дуг,
// порядка 30 мс работы между проверками отмены и разлёта.
constexpr int    kNeuralDefaultBlock = 16;
constexpr int    kNeuralObserveBlock = 64;
constexpr int    kNeuralMeanGroup    = 256;
constexpr double kNeuralWorkPerSync  = 2.0e8;
constexpr size_t kNeuralFlushBytes   = 64u * 1024u * 1024u;
constexpr int    kNeuralMaxReceptors = 8;    // NEURAL_REC_MAX в шаблоне
constexpr int    kNeuralSpikeMaxWindow = 64; // запуск не должен расти до watchdog и задерживать запись

// Возмущение клона: число от номера узла внутри экземпляра, поэтому значение свипа = отдельный запуск.
double neural_lyap_normal(uint64_t seed, uint64_t node, uint64_t var) {
    const auto mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    uint64_t s = mix(mix(seed + 0x9E3779B97F4A7C15ULL) ^ (node + 1) * 0xC2B2AE3D27D4EB4FULL);
    s = mix(s ^ (var + 1) * 0x165667B19E3779F9ULL);
    const double u1 = ((double)(mix(s + 0x9E3779B97F4A7C15ULL) >> 11) + 0.5) / 9007199254740992.0;
    const double u2 = ((double)(mix(s + 0x3C6EF372FE94F82AULL) >> 11) + 0.5) / 9007199254740992.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// R(t) по спискам пересечений порога: times[start[i]..start[i + 1]) у узла i по возрастанию. Фаза линейна между
// соседними пересечениями, косинус и синус идут поворотом, а не sincos на точку.
void neural_order_from_crossings(const std::vector<size_t>& start, const std::vector<double>& times,
                                 const std::vector<int>& pop_start, int n_points, double t0, double dt, double t_end,
                                 std::vector<double>& out) {
    const int N = (int)start.size() - 1, nPops = (int)pop_start.size() - 1;
    const int T = (std::max)(1, (std::min)((int)std::thread::hardware_concurrency(), N));
    struct Part { int pop0 = 0, pops = 0; std::vector<double> c, s, n, silent; };
    std::vector<Part> parts((size_t)T);
    const auto work = [&](int j) {
        Part& a = parts[(size_t)j];
        const int i0 = (int)((long long)N * j / T), i1 = (int)((long long)N * (j + 1) / T);
        if (i0 >= i1) return;
        const auto pop_of = [&](int i) { return (int)(std::upper_bound(pop_start.begin(), pop_start.end(), i) - pop_start.begin()) - 1; };
        a.pop0 = pop_of(i0);
        a.pops = pop_of(i1 - 1) - a.pop0 + 1;
        const size_t cells = (size_t)a.pops * n_points;
        a.c.assign(cells, 0.0); a.s.assign(cells, 0.0); a.n.assign(cells, 0.0); a.silent.assign(cells + a.pops, 0.0);
        const double two_pi = 6.283185307179586476925286766559;
        for (int i = i0; i < i1; ++i) {
            const int q = pop_of(i) - a.pop0;
            double* cs = a.c.data() + (size_t)q * n_points;
            double* sn = a.s.data() + (size_t)q * n_points;
            double* nn = a.n.data() + (size_t)q * n_points;
            const size_t m = start[(size_t)i + 1] - start[(size_t)i];
            const double* ts = times.data() + start[(size_t)i];
            double widest = 0.0;
            for (size_t k = 0; k + 1 < m; ++k) {
                const double a0 = ts[k], b0 = ts[k + 1], len = b0 - a0;
                widest = (std::max)(widest, len);
                const long long p0 = (std::max)(0LL, (long long)std::ceil((a0 - t0) / dt));
                const double step = two_pi * dt / len, cd = std::cos(step), sd = std::sin(step);
                double phi = two_pi * (t0 + (double)p0 * dt - a0) / len, c = std::cos(phi), s = std::sin(phi);
                for (long long p = p0; p < n_points && t0 + (double)p * dt < b0; ++p) {
                    cs[p] += c; sn[p] += s; nn[p] += 1.0;
                    const double c1 = c * cd - s * sd;
                    s = s * cd + c * sd; c = c1;
                }
            }
            // Замолчал до конца запуска: дольше двух своих самых длинных интервалов. Иначе следующего пересечения просто не дождались.
            const double last = m > 0 ? ts[m - 1] : t0;
            if (m == 0 || (m >= 2 && t_end - last > 2.0 * widest)) {
                const long long from = m == 0 ? 0 : (std::max)(0LL, (long long)std::ceil((last - t0) / dt));
                if (from < n_points) a.silent[(size_t)q * (n_points + 1) + (size_t)from] += 1.0;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int j = 1; j < T; ++j) pool.emplace_back(work, j);
    work(0);
    for (std::thread& t : pool) t.join();

    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.assign((size_t)n_points * nPops, nan);
    std::vector<double> c((size_t)nPops * n_points, 0.0), s(c), n(c), silent((size_t)nPops * (n_points + 1), 0.0);
    for (const Part& a : parts) {
        for (int q = 0; q < a.pops; ++q) {
            const size_t to = (size_t)(a.pop0 + q) * n_points, from = (size_t)q * n_points;
            for (int p = 0; p < n_points; ++p) { c[to + p] += a.c[from + p]; s[to + p] += a.s[from + p]; n[to + p] += a.n[from + p]; }
            for (int p = 0; p <= n_points; ++p) silent[(size_t)(a.pop0 + q) * (n_points + 1) + p] += a.silent[(size_t)q * (n_points + 1) + p];
        }
    }
    for (int q = 0; q < nPops; ++q) {
        const double size = (double)(pop_start[(size_t)q + 1] - pop_start[(size_t)q]);
        double quiet = 0.0;
        for (int p = 0; p < n_points; ++p) {
            quiet += silent[(size_t)q * (n_points + 1) + p];
            if (t0 + (double)p * dt > t_end) break;
            const double have = n[(size_t)q * n_points + p], den = have + quiet;
            if (have <= 0.0 || den < size) continue;
            const double cv = c[(size_t)q * n_points + p], sv = s[(size_t)q * n_points + p];
            out[(size_t)p * nPops + q] = std::sqrt(cv * cv + sv * sv) / den;
        }
    }
}

    // Контрольная точка: буферы устройства подряд, порядок задаёт вызывающий.
static bool neural_ck_put(NeuralCheckpoint& c, const void* dev, size_t bytes) {
    const size_t at = c.data.size();
    c.data.resize(at + bytes);
    return bytes == 0 || cudaMemcpy(c.data.data() + at, dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

static bool neural_ck_get(const NeuralCheckpoint& c, size_t& at, void* dev, size_t bytes) {
    if (at + bytes > c.data.size()) return false;
    if (bytes != 0 && cudaMemcpy(dev, c.data.data() + at, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    at += bytes;
    return true;
}

    // Причина отказа, пусто = форма та же.
static std::string neural_ck_refusal(const std::string& have, const std::string& want) {
    if (have == want) return {};
    if (have.substr(0, have.find('|')) == want.substr(0, want.find('|')))
        return "the time step, the decimation, the transient or the recorded nodes changed since the checkpoint";
    return "the checkpoint has another network shape";
}

static void neural_ck_put_host(NeuralCheckpoint& c, const void* src, size_t bytes) {
    const char* p = static_cast<const char*>(src);
    c.data.insert(c.data.end(), p, p + bytes);
}

static bool neural_ck_get_host(const NeuralCheckpoint& c, size_t& at, void* dst, size_t bytes) {
    if (at + bytes > c.data.size()) return false;
    if (bytes != 0) std::memcpy(dst, c.data.data() + at, bytes);
    at += bytes;
    return true;
}

}  // namespace

bool ParametricEngine::Impl::compile_neural_module(bool activate, const NeuralRequest& req, std::string& err) {
        cuCtxSetCurrent(context);
        const int sched = (int)req.schedule;
        const int inst = req.instance_start.size() > 2 ? 1 : 0;
        const std::string key = hash_key(req.krs_body, req.amountOfX) + ":neural:v"
                              + std::to_string(req.amountOfValues) + ":s" + std::to_string(sched)
                              + ":l" + std::to_string(req.n_laws)
                              + ":i" + std::to_string(inst)
                              + ":c" + std::to_string(std::hash<std::string>{}(req.coupling_body))
                              + ":r" + std::to_string(std::hash<std::string>{}(req.rhs_body))
                              + ":p" + std::to_string(req.spike.receptors) + ":" + std::to_string(std::hash<std::string>{}(req.spike.spike_body))
                              + ":q" + std::to_string(req.spike.edge_rule.empty() ? 0 : 1)
                              + ":d" + std::to_string(req.devices.states) + ":" + std::to_string(req.devices.params)
                              + ":" + std::to_string(std::hash<std::string>{}(req.devices.rhs_body + "|" + req.devices.current_body));
        const bool spike = req.schedule == NeuralSchedule::Spike;
        return compile_into(pool_neural, key, cached_neural, activate, [&](CachedNeuralModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template_neural), "neural.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(req.amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(req.amountOfValues) },
                                { "{{NEURAL_SCHED}}",     std::to_string(sched) },
                                { "{{NEURAL_LAWS}}",      std::to_string(req.n_laws) },
                                { "{{NEURAL_INST}}",      std::to_string(inst) },
                                { "{{NEURAL_REC}}",       std::to_string(req.spike.receptors) },
                                { "{{NEURAL_PLASTIC}}",   req.spike.edge_rule.empty() ? "0" : "1" },
                                { "{{NEURAL_DEV}}",       std::to_string(req.devices.states) },
                                { "{{NEURAL_DEV_P}}",     std::to_string(req.devices.params) },
                                { "{{DEV_RHS_BODY}}",     req.devices.rhs_body },
                                { "{{DEV_CURRENT_BODY}}", req.devices.current_body },
                                { "{{SPIKE_BODY}}",       req.spike.spike_body },
                                { "{{KRS_BODY}}",         req.krs_body },
                                { "{{KRS_RHS_BODY}}",     req.rhs_body },
                                { "{{COUPLING_BODY}}",    req.coupling_body } },
                              {}, mod, lowered, err))
                return false;
            fresh.key = key;
            fresh.module = mod;
            if (!module_fn(mod, spike ? "neuralSpikeKernel" : "neuralStepKernel", fresh.step, err)
                || !module_fn(mod, "neuralObserveKernel", fresh.observe, err)
                || !module_fn(mod, "neuralMeanKernel", fresh.mean, err)
                || !module_fn(mod, "neuralOrderKernel", fresh.order, err)
                || !module_fn(mod, "neuralLyapKernel", fresh.lyap, err)
                || !module_fn(mod, "neuralWeightKernel", fresh.weight, err)
                || !module_fn(mod, "neuralFeedbackKernel", fresh.feedback, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

// Импульсная сеть: свой цикл, общие с непрерывными только наблюдения и сборка модуля.
NeuralResult ParametricEngine::Impl::run_neural_spike(const NeuralRequest& req) {
        // Раскладки совпадают с шаблоном.
        struct SpikeRule { numb aPlus, aMinus, hPlus, hMinus, wMin, wMax, muPlus, muMinus, gain; int nearest, pad; };
        struct WeightArgs { const numb* w; const int* arcs; numb* out; int n, pad; };
        struct SpikeArgs {
            const numb* values; numb* state;
            const int* outStart; const int* outDst; const int* outPack; const unsigned long long* outW;
            unsigned long long* ring; const int* refrSteps; int* refr; int* spikes;
            int* logCount; int* logNode; int* logStep;
            const numb* srcP; const int* srcFrom; const int* srcTo; const unsigned long long* srcSeed;
            const int* inChannel; const numb* inGain; const numb* inSamples; const numb* invW; int* status; int* badFlag;
            const int* plStart; const int* plSrc; const int* plPack; const int* plRule; const SpikeRule* rules;
            numb* plW; numb* plPre; numb* plPost; int* plLast; int* stamp;
            numb decay[kNeuralMaxReceptors]; numb decayHalf[kNeuralMaxReceptors]; numb h; numb maxValue;
            long long step0; long long logBase; long long inHold; long long inLast;
            int nNodes; int slots; int steps; int logCap; int inChannels; int inSub; int inVar; int pad;
        };
        struct ObserveArgs { const numb* state; const int* nodes; const int* vars; numb* out; int nNodes, nSel, nVars, pad; };
        struct MeanArgs { const numb* state; const int* groupStart; const int* groupPop; const int* vars; numb* out; numb* out2; int nNodes, nGroups, nVars, pad; };

        NeuralResult res;
        auto fail = [&](const std::string& msg) -> NeuralResult& { res.error = msg; return res; };
        const auto wall0 = std::chrono::steady_clock::now();

        const NeuralSpiking& sp = req.spike;
        const int N = req.n_nodes, nx = req.amountOfX, R = sp.receptors;
        if (N <= 0) return fail("the network has no nodes");
        if (nx <= 0 || nx > kMaxAmountOfX) return fail("amountOfX out of range");
        if (req.amountOfValues <= 0 || req.amountOfValues > kMaxAmountOfValues) return fail("amountOfValues out of range");
        if (R < 0 || R > kNeuralMaxReceptors || R >= nx) return fail("receptors out of range");
        if ((int)sp.tau.size() != R) return fail("tau needs one value per receptor");
        for (double t : sp.tau) if (!(t > 0.0)) return fail("a synaptic time constant must be positive");
        if (req.values.size() != (size_t)N * (size_t)req.amountOfValues) return fail("values.size() != n_nodes * amountOfValues");
        if (req.initial_conditions.size() != (size_t)N * nx) return fail("initial_conditions.size() != n_nodes * amountOfX");
        if (req.edge_start.size() != (size_t)N + 1) return fail("edge_start.size() != n_nodes + 1");
        const int n_arcs = req.edge_start.back();
        if (n_arcs < 0 || (int)req.edge_src.size() != n_arcs || (int)req.edge_w.size() != n_arcs) return fail("the CSR arrays disagree on the number of arcs");
        if (n_arcs > 0 && R == 0) return fail("arcs need a receptor");
        if ((int)sp.edge_delay.size() != n_arcs) return fail("edge_delay needs one delay per arc");
        if (n_arcs > 0 && (int)req.edge_law.size() != n_arcs) return fail("edge_law must give the receptor of every arc");
        const bool plastic = !sp.edge_rule.empty();
        const int chans = R + (plastic ? 1 : 0);
        if (plastic) {
            if ((int)sp.edge_rule.size() != n_arcs) return fail("edge_rule needs one rule per arc");
            for (const NeuralStdpRule& q : sp.rules)
                if (!(q.tau_plus > 0.0) || !(q.tau_minus > 0.0) || !(q.w_max > q.w_min) || q.a_plus < 0.0 || q.a_minus < 0.0 || q.mu_plus < 0.0 || q.mu_minus < 0.0)
                    return fail("bad STDP rule");
            for (int q : sp.edge_rule) if (q < -1 || q >= (int)sp.rules.size()) return fail("an arc points to a missing STDP rule");
            for (int e = 0; e < n_arcs; ++e) {
                const int q = sp.edge_rule[(size_t)e];
                if (q >= 0 && (req.edge_w[(size_t)e] < sp.rules[(size_t)q].w_min || req.edge_w[(size_t)e] > sp.rules[(size_t)q].w_max)) return fail("a start weight is outside the bounds of its rule");
            }
        }
        for (int e : sp.watch_arcs) if (!plastic || e < 0 || e >= n_arcs) return fail("a watched arc does not exist");
        if (req.rhs_body.empty() || sp.spike_body.empty()) return fail("a spiking network needs rhs_body and spike_body");
        if (!(req.h > 0.0) || !(req.t_max > 0.0) || req.transient < 0.0 || req.pre_scaller < 1) return fail("bad time settings");
        if (step_body_uses_time(req.rhs_body)) return fail("explicit time t is not supported in the large network engine yet");
        const NeuralInput& in = req.input;
        if (in.channels > 0) {
            if (in.hold_steps < 1 || in.var < 0 || in.var >= nx - R) return fail("bad input settings");
            if (in.stage_rows != 1 && (in.stage_rows != 3 || in.hold_steps != 1)) return fail("stage_rows = 3 needs hold_steps = 1");
            if (in.samples.empty() || in.samples.size() % ((size_t)in.channels * in.stage_rows) != 0) return fail("input samples are not whole rows");
            if (in.node_channel.size() != (size_t)N || in.node_gain.size() != (size_t)N) return fail("input maps need one entry per node");
            for (int c : in.node_channel) if (c >= in.channels) return fail("input channel out of range");
        }
        if (req.lyapunov.enabled) return fail("the Lyapunov exponent does not work on a spiking network");
        if (req.feedback.from_row >= 0) return fail("a closed-loop readout does not work on a spiking network yet");
        if (req.devices.states > 0) return fail("devices on links need the stage schedule");
        for (size_t q = 0; q < req.reset_at.size(); ++q)
            if (req.reset_at[q] < 0 || (q > 0 && req.reset_at[q] <= req.reset_at[q - 1])) return fail("reset steps must increase");
        std::vector<int> instStart = req.instance_start;
        if (instStart.empty()) instStart = { 0, N };
        const int nInst = (int)instStart.size() - 1;
        if (nInst < 1 || instStart.front() != 0 || instStart.back() != N) return fail("instances must cover all nodes");
        if (req.observe.order_var >= 0 || req.observe.spread) return fail("R and the spread do not work on a spiking network");
        if (!sp.refractory_steps.empty() && sp.refractory_steps.size() != (size_t)N) return fail("refractory_steps needs one value per node");
        const bool sources = !sp.source_p.empty();
        if (sources && (sp.source_p.size() != (size_t)N || sp.source_from.size() != (size_t)N || sp.source_to.size() != (size_t)N)) return fail("source arrays need one value per node");
        for (int i = 0; i < N; ++i)
            if (req.edge_start[(size_t)i] > req.edge_start[(size_t)i + 1]) return fail("edge_start is not monotonic");

        // Приёмники хранят входящие дуги, рассылке нужны исходящие.
        std::vector<int> outStart((size_t)N + 1, 0), outDst((size_t)n_arcs), outPack((size_t)n_arcs);
        std::vector<unsigned long long> outW((size_t)n_arcs);
        int minDelay = (std::numeric_limits<int>::max)(), maxDelay = 0;
        std::vector<double> maxW((size_t)nInst, 0.0);
        std::vector<int> instOf((size_t)N);
        for (int q = 0; q < nInst; ++q) {
            if (instStart[(size_t)q] >= instStart[(size_t)q + 1]) return fail("an empty instance");
            for (int i = instStart[(size_t)q]; i < instStart[(size_t)q + 1]; ++i) instOf[(size_t)i] = q;
        }
        for (int dst = 0; dst < N; ++dst)
            for (int e = req.edge_start[(size_t)dst]; e < req.edge_start[(size_t)dst + 1]; ++e)
                if (req.edge_src[(size_t)e] >= 0 && req.edge_src[(size_t)e] < N && instOf[(size_t)req.edge_src[(size_t)e]] != instOf[(size_t)dst]) return fail("an arc joins two instances");
        for (int e = 0; e < n_arcs; ++e) {
            const int src = req.edge_src[(size_t)e], d = sp.edge_delay[(size_t)e], rec = req.edge_law[(size_t)e];
            if (src < 0 || src >= N) return fail("an arc points outside the node list");
            if (d < 1) return fail("a delay must be at least one step");
            if (rec < 0 || rec >= R) return fail("an arc points to a missing receptor");
            ++outStart[(size_t)src + 1];
            minDelay = (std::min)(minDelay, d); maxDelay = (std::max)(maxDelay, d);
            // Пластичная дуга пишет в кольцо только факт прихода.
            if (plastic && sp.edge_rule[(size_t)e] >= 0) continue;
            double& top = maxW[(size_t)instOf[(size_t)src]];
            top = (std::max)(top, std::fabs(req.edge_w[(size_t)e]));
        }
        // Запас 2^22 наибольших весов в один слот, точность 2^-40 наибольшего; дуги не выходят из экземпляра.
        std::vector<int> wExp((size_t)nInst, 0);
        for (int q = 0; q < nInst; ++q) if (maxW[(size_t)q] > 0.0) wExp[(size_t)q] = 40 - std::ilogb(maxW[(size_t)q]) - 1;
        for (int i = 0; i < N; ++i) outStart[(size_t)i + 1] += outStart[(size_t)i];
        {
            std::vector<int> put(outStart.begin(), outStart.end() - 1);
            for (int dst = 0; dst < N; ++dst)
                for (int e = req.edge_start[(size_t)dst]; e < req.edge_start[(size_t)dst + 1]; ++e) {
                    const int at = put[(size_t)req.edge_src[(size_t)e]]++;
                    outDst[(size_t)at] = dst;
                    const bool arcPlastic = plastic && sp.edge_rule[(size_t)e] >= 0;
                    outPack[(size_t)at] = sp.edge_delay[(size_t)e] * chans + (arcPlastic ? R : req.edge_law[(size_t)e]);
                    outW[(size_t)at] = arcPlastic ? 1ULL : (unsigned long long)std::llround(std::ldexp(req.edge_w[(size_t)e], wExp[(size_t)instOf[(size_t)dst]]));
                }
        }
        // Пластичные дуги по приёмнику, как в запросе: порядок прохода и суммы фиксирован.
        std::vector<int> plStart((size_t)N + 1, 0), plSrc, plPack, plRule, plArc((size_t)(plastic ? n_arcs : 0), -1);
        std::vector<numb> plW;
        if (plastic) {
            for (int dst = 0; dst < N; ++dst) {
                for (int e = req.edge_start[(size_t)dst]; e < req.edge_start[(size_t)dst + 1]; ++e) {
                    if (sp.edge_rule[(size_t)e] < 0) continue;
                    plArc[(size_t)e] = (int)plSrc.size();
                    plSrc.push_back(req.edge_src[(size_t)e]);
                    plPack.push_back(sp.edge_delay[(size_t)e] * R + req.edge_law[(size_t)e]);
                    plRule.push_back(sp.edge_rule[(size_t)e]);
                    plW.push_back((numb)req.edge_w[(size_t)e]);
                }
                plStart[(size_t)dst + 1] = (int)plSrc.size();
            }
        }
        std::vector<int> watch;
        for (int e : sp.watch_arcs) {
            if (plArc[(size_t)e] < 0) return fail("a watched arc is not plastic");
            watch.push_back(plArc[(size_t)e]);
        }
        const int nWatch = (int)watch.size(), nPl = (int)plSrc.size();

        int window = n_arcs > 0 ? (std::min)(minDelay, kNeuralSpikeMaxWindow) : kNeuralSpikeMaxWindow;
        if (sp.window > 0) {
            if (n_arcs > 0 && sp.window > minDelay) return fail("the window cannot exceed the shortest delay");
            window = sp.window;
        }
        // Запуск идёт без согласования шагов: спайк конца окна не должен попасть в слот, который сосед ещё не прочёл.
        const int slots = maxDelay + window;
        if (R > 0 && (double)slots * chans > (double)(std::numeric_limits<int>::max)()) return fail("delays are too long");
        const size_t ringBytes = (size_t)slots * chans * (size_t)N * sizeof(unsigned long long);

        const NeuralObservation& ob = req.observe;
        for (int v : ob.vars) if (v < 0 || v >= nx) return fail("observed variable out of range");
        for (int i : ob.nodes) if (i < 0 || i >= N) return fail("observed node out of range");
        const int nPops = ob.pop_start.empty() ? 0 : (int)ob.pop_start.size() - 1;
        if (nPops > 0 && (ob.pop_start.front() != 0 || ob.pop_start.back() != N)) return fail("populations must cover all nodes");

        const long long skipSteps = (long long)steps_from_time_size_t(req.transient, req.h);
        const long long workSteps = (long long)steps_from_time_size_t(req.t_max, req.h);
        const long long totalSteps = skipSteps + workSteps;
        if (totalSteps <= 0) return fail("t_max / h gives no steps");

        const int nVars = (int)ob.vars.size(), nSel = nVars ? (int)ob.nodes.size() : 0, nMean = nVars ? nPops : 0;
        const long long nPointsLL = (nSel + nMean + nWatch > 0) ? workSteps / req.pre_scaller + 1 : 0;
        const size_t rowElems = ((size_t)nSel + (size_t)nMean) * (size_t)nVars + (size_t)nWatch;
        if ((double)nPointsLL * (double)rowElems * sizeof(double) > (double)req.max_output_bytes)
            return fail("the recorded output is above max_output_bytes: raise the decimation or select less");
        const int nPoints = (int)nPointsLL;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync
        if (!compile_neural_module(true, req, err)) return fail(err);

        const int flushRows = nPoints > 0 ? (int)std::min<long long>(nPoints, std::max<long long>(1, (long long)(kNeuralFlushBytes / std::max<size_t>(1, rowElems * sizeof(numb))))) : 0;
        const int logCap = sp.log_capacity > 0 ? sp.log_capacity : (int)std::clamp((long long)N * 64, 1LL << 20, 1LL << 24);
        const size_t logLimit = req.max_output_bytes / (sizeof(long long) + sizeof(int));

        OrderDevBuf d_values, d_state, d_oStart, d_oDst, d_oPack, d_oW, d_ring, d_refrSteps, d_refr, d_spikes,
                    d_logCount, d_logNode, d_logStep, d_srcP, d_srcFrom, d_srcTo, d_srcSeed, d_bad, d_status, d_invW,
                    d_inCh, d_inGain, d_inRows, d_sel, d_vars, d_gStart, d_gPop, d_out,
                    d_plStart, d_plSrc, d_plPack, d_plRule, d_rules, d_plW, d_plPre, d_plPost, d_plLast, d_stamp, d_watch;
        auto dev = [&](OrderDevBuf& b, size_t bytes, const char* what) {
            res.device_bytes += bytes;
            return b.alloc(std::max<size_t>(bytes, sizeof(numb)), what, err);
        };
        auto up = [&](OrderDevBuf& b, const void* src, size_t bytes, const char* what) {
            if (bytes == 0) return true;
            if (cudaMemcpy(b.p, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) return true;
            err = std::string("memcpy ") + what; return false;
        };
        const size_t stampBytes = plastic ? (size_t)slots * (size_t)N * sizeof(int) : 0;
        std::vector<numb> soa((size_t)N * nx);
        for (int i = 0; i < N; ++i) for (int k = 0; k < nx; ++k)
            soa[(size_t)k * N + i] = (numb)req.initial_conditions[(size_t)i * nx + k];
        {
            const std::vector<numb> hv = to_numb(req.values);
            std::vector<int> refrSteps = sp.refractory_steps;
            refrSteps.resize((size_t)N, 0);
            std::vector<numb> invW((size_t)N);
            for (int i = 0; i < N; ++i) invW[(size_t)i] = (numb)std::ldexp(1.0, -wExp[(size_t)instOf[(size_t)i]]);
            if (!dev(d_invW, invW.size() * sizeof(numb), "invW") || !up(d_invW, invW.data(), invW.size() * sizeof(numb), "invW")) return fail(err);
            if (!dev(d_values, hv.size() * sizeof(numb), "values") || !up(d_values, hv.data(), hv.size() * sizeof(numb), "values")
                || !dev(d_state, soa.size() * sizeof(numb), "state") || !up(d_state, soa.data(), soa.size() * sizeof(numb), "state")
                || !dev(d_oStart, outStart.size() * sizeof(int), "outStart") || !up(d_oStart, outStart.data(), outStart.size() * sizeof(int), "outStart")
                || !dev(d_oDst, outDst.size() * sizeof(int), "outDst") || !up(d_oDst, outDst.data(), outDst.size() * sizeof(int), "outDst")
                || !dev(d_oPack, outPack.size() * sizeof(int), "outPack") || !up(d_oPack, outPack.data(), outPack.size() * sizeof(int), "outPack")
                || !dev(d_oW, outW.size() * sizeof(unsigned long long), "outW") || !up(d_oW, outW.data(), outW.size() * sizeof(unsigned long long), "outW")
                || !dev(d_refrSteps, (size_t)N * sizeof(int), "refrSteps") || !up(d_refrSteps, refrSteps.data(), (size_t)N * sizeof(int), "refrSteps")
                || !dev(d_ring, ringBytes, "ring") || !dev(d_refr, (size_t)N * sizeof(int), "refr") || !dev(d_spikes, (size_t)N * sizeof(int), "spikes")
                || !dev(d_logCount, sizeof(int), "logCount") || !dev(d_logNode, (size_t)logCap * sizeof(int), "logNode")
                || !dev(d_logStep, (size_t)logCap * sizeof(int), "logStep") || !dev(d_bad, sizeof(int), "badFlag")
                || !dev(d_status, (size_t)N * sizeof(int), "status")) return fail(err);
            if (in.channels > 0) {
                const std::vector<numb> rows = to_numb(in.samples), gain = to_numb(in.node_gain);
                if (!dev(d_inRows, rows.size() * sizeof(numb), "inSamples") || !up(d_inRows, rows.data(), rows.size() * sizeof(numb), "inSamples")
                    || !dev(d_inCh, (size_t)N * sizeof(int), "inChannel") || !up(d_inCh, in.node_channel.data(), (size_t)N * sizeof(int), "inChannel")
                    || !dev(d_inGain, (size_t)N * sizeof(numb), "inGain") || !up(d_inGain, gain.data(), (size_t)N * sizeof(numb), "inGain")) return fail(err);
            }
            if (sources) {
                std::vector<unsigned long long> seeds = sp.source_seed;
                if (seeds.empty()) {
                    seeds.resize((size_t)N);
                    for (int i = 0; i < N; ++i) seeds[(size_t)i] = neural_source_seed(sp.seed, i);
                }
                if (seeds.size() != (size_t)N) return fail("source_seed needs one value per node");
                if (!dev(d_srcSeed, (size_t)N * sizeof(unsigned long long), "srcSeed") || !up(d_srcSeed, seeds.data(), (size_t)N * sizeof(unsigned long long), "srcSeed")) return fail(err);
                const std::vector<numb> p = to_numb(sp.source_p);
                if (!dev(d_srcP, p.size() * sizeof(numb), "srcP") || !up(d_srcP, p.data(), p.size() * sizeof(numb), "srcP")
                    || !dev(d_srcFrom, (size_t)N * sizeof(int), "srcFrom") || !up(d_srcFrom, sp.source_from.data(), (size_t)N * sizeof(int), "srcFrom")
                    || !dev(d_srcTo, (size_t)N * sizeof(int), "srcTo") || !up(d_srcTo, sp.source_to.data(), (size_t)N * sizeof(int), "srcTo")) return fail(err);
            }
        }
        if (plastic) {
            std::vector<SpikeRule> rules(sp.rules.size());
            for (size_t q = 0; q < rules.size(); ++q) {
                const NeuralStdpRule& s = sp.rules[q];
                rules[q] = SpikeRule{ (numb)s.a_plus, (numb)s.a_minus, (numb)(req.h / s.tau_plus), (numb)(req.h / s.tau_minus), (numb)s.w_min, (numb)s.w_max,
                                      (numb)s.mu_plus, (numb)s.mu_minus, (numb)s.gain, s.nearest ? 1 : 0, 0 };
            }
            const size_t iBytes = (size_t)nPl * sizeof(int), nBytes = (size_t)nPl * sizeof(numb);
            if (!dev(d_plStart, plStart.size() * sizeof(int), "plStart") || !up(d_plStart, plStart.data(), plStart.size() * sizeof(int), "plStart")
                || !dev(d_plSrc, iBytes, "plSrc") || !up(d_plSrc, plSrc.data(), iBytes, "plSrc")
                || !dev(d_plPack, iBytes, "plPack") || !up(d_plPack, plPack.data(), iBytes, "plPack")
                || !dev(d_plRule, iBytes, "plRule") || !up(d_plRule, plRule.data(), iBytes, "plRule")
                || !dev(d_rules, rules.size() * sizeof(SpikeRule), "rules") || !up(d_rules, rules.data(), rules.size() * sizeof(SpikeRule), "rules")
                || !dev(d_plW, nBytes, "plW") || !up(d_plW, plW.data(), nBytes, "plW")
                || !dev(d_plPre, nBytes, "plPre") || !dev(d_plPost, nBytes, "plPost") || !dev(d_plLast, iBytes, "plLast")
                || !dev(d_stamp, stampBytes, "stamp")) return fail(err);
            if (nWatch > 0 && (!dev(d_watch, (size_t)nWatch * sizeof(int), "watch") || !up(d_watch, watch.data(), (size_t)nWatch * sizeof(int), "watch"))) return fail(err);
            if (cudaMemset(d_plPre.p, 0, nBytes) != cudaSuccess || cudaMemset(d_plPost.p, 0, nBytes) != cudaSuccess
                || cudaMemset(d_plLast.p, 0, iBytes) != cudaSuccess || cudaMemset(d_stamp.p, 0xFF, stampBytes) != cudaSuccess) return fail("memset plastic buffers");
        }
        if ((ringBytes > 0 && cudaMemset(d_ring.p, 0, ringBytes) != cudaSuccess) || cudaMemset(d_refr.p, 0, (size_t)N * sizeof(int)) != cudaSuccess
            || cudaMemset(d_spikes.p, 0, (size_t)N * sizeof(int)) != cudaSuccess || cudaMemset(d_logCount.p, 0, sizeof(int)) != cudaSuccess
            || cudaMemset(d_bad.p, 0, sizeof(int)) != cudaSuccess || cudaMemset(d_status.p, 0, (size_t)N * sizeof(int)) != cudaSuccess) return fail("memset spike buffers");

        std::vector<int> gStart, gPop;
        if (nMean > 0) {
            for (int p = 0; p < nPops; ++p)
                for (int i = ob.pop_start[(size_t)p]; i < ob.pop_start[(size_t)p + 1]; i += kNeuralMeanGroup) { gStart.push_back(i); gPop.push_back(p); }
            gStart.push_back(N);
            if (!dev(d_gStart, gStart.size() * sizeof(int), "groupStart") || !dev(d_gPop, gPop.size() * sizeof(int), "groupPop")
                || !up(d_gStart, gStart.data(), gStart.size() * sizeof(int), "groupStart") || !up(d_gPop, gPop.data(), gPop.size() * sizeof(int), "groupPop")) return fail(err);
        }
        if (nVars > 0 && (!dev(d_vars, (size_t)nVars * sizeof(int), "vars") || !up(d_vars, ob.vars.data(), (size_t)nVars * sizeof(int), "vars"))) return fail(err);
        if (nSel > 0 && (!dev(d_sel, (size_t)nSel * sizeof(int), "selected") || !up(d_sel, ob.nodes.data(), (size_t)nSel * sizeof(int), "selected"))) return fail(err);
        if (flushRows > 0 && !dev(d_out, (size_t)flushRows * rowElems * sizeof(numb), "out")) return fail(err);

        std::vector<double> traces, means, weightTrace;
        if (nSel > 0) traces.assign((size_t)nPoints * nSel * nVars, std::numeric_limits<double>::quiet_NaN());
        if (nMean > 0) means.assign((size_t)nPoints * nMean * nVars, std::numeric_limits<double>::quiet_NaN());
        if (nWatch > 0) weightTrace.assign((size_t)nPoints * nWatch, std::numeric_limits<double>::quiet_NaN());
        std::vector<numb> hostRows((size_t)flushRows * rowElems);
        int rowsOnDevice = 0, rowsFlushed = 0;
        const auto flush = [&]() {
            if (rowsOnDevice == 0) return true;
            if (cudaMemcpy(hostRows.data(), d_out.p, (size_t)rowsOnDevice * rowElems * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (int r = 0; r < rowsOnDevice; ++r) {
                const numb* row = hostRows.data() + (size_t)r * rowElems;
                const size_t p = (size_t)rowsFlushed + r;
                for (size_t e = 0; e < (size_t)nSel * nVars; ++e) traces[p * nSel * nVars + e] = (double)row[e];
                const numb* pops = row + (size_t)nSel * nVars;
                for (int q = 0; q < nMean; ++q) {
                    const double size = (double)(ob.pop_start[(size_t)q + 1] - ob.pop_start[(size_t)q]);
                    for (int v = 0; v < nVars; ++v) means[(p * nMean + q) * nVars + v] = (double)pops[(size_t)q * nVars + v] / size;
                }
                const numb* weights = pops + (size_t)nMean * nVars;
                for (int q = 0; q < nWatch; ++q) weightTrace[p * nWatch + q] = (double)weights[q];
            }
            rowsFlushed += rowsOnDevice;
            rowsOnDevice = 0;
            return true;
        };
        const auto record = [&]() -> bool {
            numb* row = d_out.as<numb>() + (size_t)rowsOnDevice * rowElems;
            if (nSel > 0) {
                ObserveArgs oa{ d_state.as<numb>(), d_sel.as<int>(), d_vars.as<int>(), row, N, nSel, nVars, 0 };
                void* args[] = { &oa };
                const unsigned b = (unsigned)((nSel + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.observe, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural observe): " + cu_err(r); return false; }
                ++res.launches;
            }
            if (nMean > 0) {
                numb* sums = row + (size_t)nSel * nVars;
                if (cudaMemsetAsync(sums, 0, (size_t)nMean * nVars * sizeof(numb), 0) != cudaSuccess) { err = "memset means"; return false; }
                MeanArgs ma{ d_state.as<numb>(), d_gStart.as<int>(), d_gPop.as<int>(), d_vars.as<int>(), sums, nullptr, N, (int)gPop.size(), nVars, 0 };
                void* args[] = { &ma };
                const unsigned b = (unsigned)((gPop.size() + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.mean, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural mean): " + cu_err(r); return false; }
                ++res.launches;
            }
            if (nWatch > 0) {
                WeightArgs wa{ d_plW.as<numb>(), d_watch.as<int>(), row + ((size_t)nSel + nMean) * nVars, nWatch, 0 };
                void* args[] = { &wa };
                const unsigned b = (unsigned)((nWatch + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.weight, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural weight): " + cu_err(r); return false; }
                ++res.launches;
            }
            ++rowsOnDevice;
            return true;
        };

        const int block = req.block_size > 0 ? req.block_size : kNeuralDefaultBlock;
        const unsigned blocks = (unsigned)(((long long)N + block - 1) / block);
        long long syncEvery = (std::max)(1LL, (long long)(kNeuralWorkPerSync / ((double)N * 4.0)));
        if (req.steps_per_sync > 0 && req.steps_per_sync < syncEvery) syncEvery = req.steps_per_sync;
        res.block_size = block;
        res.steps_per_sync = syncEvery;

        SpikeArgs sa{};
        sa.values = d_values.as<numb>(); sa.state = d_state.as<numb>();
        sa.outStart = d_oStart.as<int>(); sa.outDst = d_oDst.as<int>(); sa.outPack = d_oPack.as<int>();
        sa.outW = d_oW.as<unsigned long long>(); sa.ring = d_ring.as<unsigned long long>();
        sa.refrSteps = d_refrSteps.as<int>(); sa.refr = d_refr.as<int>(); sa.spikes = d_spikes.as<int>();
        sa.logCount = d_logCount.as<int>(); sa.logNode = d_logNode.as<int>(); sa.logStep = d_logStep.as<int>();
        sa.srcP = sources ? d_srcP.as<numb>() : nullptr; sa.srcFrom = sources ? d_srcFrom.as<int>() : nullptr; sa.srcTo = sources ? d_srcTo.as<int>() : nullptr;
        sa.srcSeed = sources ? d_srcSeed.as<unsigned long long>() : nullptr;
        sa.inChannel = in.channels > 0 ? d_inCh.as<int>() : nullptr;
        sa.inGain = in.channels > 0 ? d_inGain.as<numb>() : nullptr;
        sa.inSamples = in.channels > 0 ? d_inRows.as<numb>() : nullptr;
        sa.inChannels = in.channels; sa.inSub = in.stage_rows; sa.inVar = in.var; sa.inHold = in.hold_steps;
        sa.inLast = in.channels > 0 ? (long long)(in.samples.size() / ((size_t)in.channels * in.stage_rows)) - 1 : 0;
        sa.invW = d_invW.as<numb>();
        sa.status = d_status.as<int>();
        sa.badFlag = d_bad.as<int>();
        if (plastic) {
            sa.plStart = d_plStart.as<int>(); sa.plSrc = d_plSrc.as<int>(); sa.plPack = d_plPack.as<int>(); sa.plRule = d_plRule.as<int>();
            sa.rules = d_rules.as<SpikeRule>(); sa.plW = d_plW.as<numb>(); sa.plPre = d_plPre.as<numb>(); sa.plPost = d_plPost.as<numb>();
            sa.plLast = d_plLast.as<int>(); sa.stamp = d_stamp.as<int>();
        }
        for (int r = 0; r < R; ++r) {
            sa.decay[r] = (numb)std::exp(-req.h / sp.tau[(size_t)r]);
            sa.decayHalf[r] = (numb)std::exp(-0.5 * req.h / sp.tau[(size_t)r]);
        }
        sa.h = (numb)req.h; sa.maxValue = (numb)req.max_value;
        sa.nNodes = N; sa.slots = slots; sa.logCap = logCap;

        // Журнал копится упакованным: сортировка по шагу и узлу убирает порядок потоков.
        std::vector<unsigned long long> packed;
        std::vector<int> hostNode, hostStep;
        const auto drain_log = [&](long long base) -> bool {
            int n = 0;
            const int zero = 0;
            if (cudaMemcpy(&n, d_logCount.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            const int kept = (std::min)(n, logCap);
            if (base >= skipSteps) res.spikes_dropped += n - kept;
            if (kept > 0 && base >= skipSteps) {
                hostNode.resize((size_t)kept); hostStep.resize((size_t)kept);
                if (cudaMemcpy(hostNode.data(), d_logNode.p, (size_t)kept * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess
                    || cudaMemcpy(hostStep.data(), d_logStep.p, (size_t)kept * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
                for (int e = 0; e < kept; ++e) {
                    if (packed.size() >= logLimit) { res.spikes_dropped += kept - e; break; }
                    packed.push_back(((unsigned long long)(base + hostStep[(size_t)e]) << 32) | (unsigned)hostNode[(size_t)e]);
                }
            }
            return cudaMemcpy(d_logCount.p, &zero, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
        };

        std::vector<long long> instStop((size_t)nInst, -1);
        long long done = 0;
        size_t nextReset = 0;
        const std::string ckShape = "spike:" + std::to_string(N) + ":" + std::to_string(nx) + ":" + std::to_string(slots) + ":" + std::to_string(chans)
                                  + ":" + std::to_string(nPl) + ":" + std::to_string(nInst) + ":" + std::to_string(sizeof(numb)) + ":" + req.checkpoint_tag;
        if (req.resume) {
            const NeuralCheckpoint& c = *req.resume;
            const std::string refusal = neural_ck_refusal(c.shape, ckShape);
            if (!refusal.empty()) return fail(refusal);
            if (c.step > totalSteps) return fail("the checkpoint is past the end of this run");
            size_t at = 0;
            bool ok = neural_ck_get_host(c, at, instStop.data(), instStop.size() * sizeof(long long))
                   && neural_ck_get(c, at, d_state.p, soa.size() * sizeof(numb)) && neural_ck_get(c, at, d_ring.p, ringBytes)
                   && neural_ck_get(c, at, d_refr.p, (size_t)N * sizeof(int)) && neural_ck_get(c, at, d_spikes.p, (size_t)N * sizeof(int))
                   && neural_ck_get(c, at, d_status.p, (size_t)N * sizeof(int));
            if (plastic) ok = ok && neural_ck_get(c, at, d_plW.p, (size_t)nPl * sizeof(numb)) && neural_ck_get(c, at, d_plPre.p, (size_t)nPl * sizeof(numb))
                              && neural_ck_get(c, at, d_plPost.p, (size_t)nPl * sizeof(numb)) && neural_ck_get(c, at, d_plLast.p, (size_t)nPl * sizeof(int))
                              && neural_ck_get(c, at, d_stamp.p, stampBytes);
            if (!ok || at != c.data.size()) return fail("the checkpoint is damaged");
            done = c.step;
            while (nextReset < req.reset_at.size() && req.reset_at[nextReset] < done) ++nextReset;
            if (done >= skipSteps) rowsFlushed = (int)(std::min<long long>)((done - skipSteps) / req.pre_scaller + 1, nPoints);
            for (long long v : instStop) res.diverged = res.diverged || v >= 0;
        } else if (skipSteps == 0 && nPoints > 0 && !record()) return fail(err);

        bool stopped = false;
        while (done < totalSteps && !stopped) {
            // Попытка с чистого листа: и отложенные спайки, и рефрактерность.
            if (nextReset < req.reset_at.size() && req.reset_at[nextReset] == done) {
                ++nextReset;
                if (cudaMemcpy(d_state.p, soa.data(), soa.size() * sizeof(numb), cudaMemcpyHostToDevice) != cudaSuccess
                    || (ringBytes > 0 && cudaMemset(d_ring.p, 0, ringBytes) != cudaSuccess)
                    || cudaMemset(d_refr.p, 0, (size_t)N * sizeof(int)) != cudaSuccess) return fail("memcpy reset");
                // Веса переживают попытку: обучение идёт через попытки.
                if (plastic && (cudaMemset(d_plPre.p, 0, (size_t)nPl * sizeof(numb)) != cudaSuccess || cudaMemset(d_plPost.p, 0, (size_t)nPl * sizeof(numb)) != cudaSuccess
                                || cudaMemset(d_stamp.p, 0xFF, stampBytes) != cudaSuccess)) return fail("memset reset");
            }
            const long long chunk0 = done;
            long long chunkEnd = (std::min)(done + syncEvery, totalSteps);
            if (done < skipSteps) chunkEnd = (std::min)(chunkEnd, skipSteps);
            if (nextReset < req.reset_at.size()) chunkEnd = (std::min)(chunkEnd, req.reset_at[nextReset]);
            sa.logBase = chunk0;
            while (done < chunkEnd) {
                long long to = chunkEnd;
                if (nPoints > 0 && done >= skipSteps) to = (std::min)(to, skipSteps + ((done - skipSteps) / req.pre_scaller + 1) * req.pre_scaller);
                sa.step0 = done;
                sa.steps = (int)(std::min<long long>)(to - done, window);
                void* args[] = { &sa };
                CUresult r = cuLaunchKernel(cached_neural.step, blocks, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(neural spike): " + cu_err(r));
                ++res.launches;
                done += sa.steps;
                if (nPoints > 0 && done >= skipSteps && (done - skipSteps) % req.pre_scaller == 0 && (done - skipSteps) / req.pre_scaller < nPoints) {
                    if (!record()) return fail(err);
                    if (rowsOnDevice == flushRows) {
                        cudaDeviceSynchronize();
                        if (!flush()) return fail("memcpy D2H out");
                    }
                }
            }
            cudaDeviceSynchronize();
            ++res.syncs;
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("neural spike kernel: ") + cudaGetErrorString(ce));
            if (!flush()) return fail("memcpy D2H out");
            if (!drain_log(chunk0)) return fail("memcpy D2H spike log");
            if (done == skipSteps && skipSteps > 0 && cudaMemset(d_spikes.p, 0, (size_t)N * sizeof(int)) != cudaSuccess) return fail("memset spikes");
            if (req.progress) req.progress->store((float)((double)done / (double)totalSteps), std::memory_order_relaxed);
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { res.cancelled = true; stopped = true; break; }
            int bad = 0;
            if (cudaMemcpy(&bad, d_bad.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H flag");
            if (!bad) continue;
            res.diverged = true;
            // Шаг разлёта экземпляра = минимум по его узлам, от длины чанка он не зависит.
            std::vector<int> st((size_t)N);
            if (cudaMemcpy(st.data(), d_status.p, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H status");
            int running = 0;
            for (int q = 0; q < nInst; ++q) {
                if (instStop[(size_t)q] >= 0) continue;
                int first = 0;
                for (int i = instStart[(size_t)q]; i < instStart[(size_t)q + 1]; ++i)
                    if (st[(size_t)i] > 0 && (first == 0 || st[(size_t)i] < first)) first = st[(size_t)i];
                if (first == 0) { ++running; continue; }
                instStop[(size_t)q] = first;
                for (int i = instStart[(size_t)q]; i < instStart[(size_t)q + 1]; ++i) if (st[(size_t)i] == 0) st[(size_t)i] = -1;
            }
            const int zero = 0;
            if (cudaMemcpy(d_status.p, st.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess
                || cudaMemcpy(d_bad.p, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return fail("memcpy H2D status");
            if (running == 0) stopped = true;
        }
        res.steps_done = done;
        res.instance_stop = instStop;
        if (req.keep_checkpoint) {
            cudaDeviceSynchronize();
            auto c = std::make_shared<NeuralCheckpoint>();
            c->step = done;
            c->shape = ckShape;
            neural_ck_put_host(*c, instStop.data(), instStop.size() * sizeof(long long));
            bool ok = neural_ck_put(*c, d_state.p, soa.size() * sizeof(numb)) && neural_ck_put(*c, d_ring.p, ringBytes)
                   && neural_ck_put(*c, d_refr.p, (size_t)N * sizeof(int)) && neural_ck_put(*c, d_spikes.p, (size_t)N * sizeof(int))
                   && neural_ck_put(*c, d_status.p, (size_t)N * sizeof(int));
            if (plastic) ok = ok && neural_ck_put(*c, d_plW.p, (size_t)nPl * sizeof(numb)) && neural_ck_put(*c, d_plPre.p, (size_t)nPl * sizeof(numb))
                              && neural_ck_put(*c, d_plPost.p, (size_t)nPl * sizeof(numb)) && neural_ck_put(*c, d_plLast.p, (size_t)nPl * sizeof(int))
                              && neural_ck_put(*c, d_stamp.p, stampBytes);
            if (!ok) return fail("memcpy D2H checkpoint");
            res.checkpoint = std::move(c);
        }

        // После шага разлёта экземпляр считался до конца чанка: эти спайки зависят от чанка и данных не несут.
        if (res.diverged) {
            packed.erase(std::remove_if(packed.begin(), packed.end(), [&](unsigned long long v) {
                const long long stop = instStop[(size_t)instOf[(size_t)(v & 0xFFFFFFFFULL)]];
                return stop >= 0 && (long long)(v >> 32) >= stop;
            }), packed.end());
        }
        std::sort(packed.begin(), packed.end());
        res.spike_step.resize(packed.size());
        res.spike_node.resize(packed.size());
        for (size_t e = 0; e < packed.size(); ++e) {
            res.spike_step[e] = (long long)(packed[e] >> 32);
            res.spike_node[e] = (int)(packed[e] & 0xFFFFFFFFULL);
        }
        res.spike_count.resize((size_t)N);
        if (cudaMemcpy(res.spike_count.data(), d_spikes.p, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H spikes");

        if (req.keep_final_state) {
            std::vector<numb> soa((size_t)N * nx);
            if (cudaMemcpy(soa.data(), d_state.p, soa.size() * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H state");
            res.final_state.resize(soa.size());
            for (int i = 0; i < N; ++i) for (int k = 0; k < nx; ++k) res.final_state[(size_t)i * nx + k] = (double)soa[(size_t)k * N + i];
        }
        if (plastic) {
            std::vector<numb> w((size_t)nPl);
            if (nPl > 0 && cudaMemcpy(w.data(), d_plW.p, w.size() * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H weights");
            res.final_weights = req.edge_w;
            for (int e = 0; e < n_arcs; ++e) if (plArc[(size_t)e] >= 0) res.final_weights[(size_t)e] = (double)w[(size_t)plArc[(size_t)e]];
        }

        res.n_nodes = N; res.n_vars = nVars; res.n_points = nPoints; res.h = req.h;
        res.t0 = (double)skipSteps * req.h; res.dt = (double)req.pre_scaller * req.h;
        res.observed = nSel > 0 ? ob.nodes : std::vector<int>();
        res.observed_vars = ob.vars;
        if (res.diverged && nPoints > 0) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (int p = 0; p < nPoints; ++p) {
                const long long step = skipSteps + (long long)p * req.pre_scaller;
                for (int q = 0; q < nSel; ++q) {
                    const long long stop = instStop[(size_t)instOf[(size_t)ob.nodes[(size_t)q]]];
                    if (stop >= 0 && step >= stop) for (int v = 0; v < nVars; ++v) traces[((size_t)p * nSel + q) * nVars + v] = nan;
                }
                for (int q = 0; q < nMean; ++q) {
                    const long long stop = instStop[(size_t)instOf[(size_t)ob.pop_start[(size_t)q]]];
                    if (stop >= 0 && step >= stop) for (int v = 0; v < nVars; ++v) means[((size_t)p * nMean + q) * nVars + v] = nan;
                }
                for (int q = 0; q < nWatch; ++q) {
                    const long long stop = instStop[(size_t)instOf[(size_t)req.edge_src[(size_t)sp.watch_arcs[(size_t)q]]]];
                    if (stop >= 0 && step >= stop) weightTrace[(size_t)p * nWatch + q] = nan;
                }
            }
        }
        res.traces = std::move(traces);
        res.means = std::move(means);
        res.weight_trace = std::move(weightTrace);
        res.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
        res.ok = true;
        return res;
    }

NeuralResult ParametricEngine::Impl::run_neural(const NeuralRequest& req) {
        if (req.schedule == NeuralSchedule::Spike) return run_neural_spike(req);
        // Раскладки совпадают с шаблоном.
        struct StepArgs {
            const numb* values; numb* s0; numb* s1; numb* s2; numb* s3;
            const int* edgeStart; const int* edgeSrc; const numb* edgeW; const int* edgeLaw;
            const int* inChannel; const numb* inGain; const numb* inSample;
            int* badFlag; int* status;
            numb h; numb maxValue;
            int nNodes; int phase; int inVar; int step; int nRun; int pad;
            const int* devStart; const int* devList; const int* devA; const int* devB; const int* devVar;
            const numb* devGain; const numb* devValues;
            numb* d0; numb* d1; numb* d2; numb* d3; int* devClips;
            numb devMin[4]; numb devMax[4];
            int nDev; int pad2;
        };
        struct WeightArgs { const numb* w; const int* arcs; numb* out; int n, pad; };
        struct LyapArgs { numb* state; const int* groupStart; numb* dist; const numb* scale; int nNodes, nGroups; };
        struct ObserveArgs { const numb* state; const int* nodes; const int* vars; numb* out; int nNodes, nSel, nVars, pad; };
        struct MeanArgs { const numb* state; const int* groupStart; const int* groupPop; const int* vars; numb* out; numb* out2; int nNodes, nGroups, nVars, pad; };
        struct OrderArgs {
            const numb* column; numb* prev; int* count; int* outNode; numb* outTime;
            numb threshold; numb t; numb dt;
            int nNodes; int first; int capacity; int pad;
        };

        NeuralResult res;
        auto fail = [&](const std::string& msg) -> NeuralResult& { res.error = msg; return res; };
        const auto wall0 = std::chrono::steady_clock::now();

        const int N = req.n_nodes, nx = req.amountOfX;
        const size_t nStates = (size_t)N * (size_t)nx;
        if (N <= 0) return fail("the network has no nodes");
        if (nx <= 0 || nx > kMaxAmountOfX) return fail("amountOfX out of range");
        if (req.amountOfValues <= 0 || req.amountOfValues > kMaxAmountOfValues) return fail("amountOfValues out of range");
        if (req.values.size() != (size_t)N * (size_t)req.amountOfValues) return fail("values.size() != n_nodes * amountOfValues");
        if (req.initial_conditions.size() != nStates) return fail("initial_conditions.size() != n_nodes * amountOfX");
        if (req.edge_start.size() != (size_t)N + 1) return fail("edge_start.size() != n_nodes + 1");
        if (req.n_laws < 1) return fail("n_laws must be >= 1");
        const int n_arcs = req.edge_start.back();
        if (n_arcs < 0 || (int)req.edge_src.size() != n_arcs || (int)req.edge_w.size() != n_arcs)
            return fail("the CSR arrays disagree on the number of arcs");
        if (req.n_laws > 1 && (int)req.edge_law.size() != n_arcs) return fail("edge_law is required with several laws");
        for (int i = 0; i < N; ++i)
            if (req.edge_start[(size_t)i] > req.edge_start[(size_t)i + 1]) return fail("edge_start is not monotonic");
        for (int e = 0; e < n_arcs; ++e)
            if (req.edge_src[(size_t)e] < 0 || req.edge_src[(size_t)e] >= N) return fail("an arc points outside the node list");
        if (req.krs_body.empty()) return fail("krs_body is empty");
        if (req.schedule == NeuralSchedule::Stage && req.rhs_body.empty()) return fail("stage coupling needs rhs_body");
        if (step_body_uses_time(req.krs_body) || step_body_uses_time(req.rhs_body))
            return fail("explicit time t is not supported in the large network engine yet");
        if (!(req.h > 0.0) || !(req.t_max > 0.0) || req.transient < 0.0 || req.pre_scaller < 1) return fail("bad time settings");

        const NeuralInput& in = req.input;
        if (in.channels > 0) {
            if (in.hold_steps < 1 || in.var < 0 || in.var >= nx) return fail("bad input settings");
            if (in.stage_rows != 1 && (in.stage_rows != 3 || req.schedule != NeuralSchedule::Stage || in.hold_steps != 1)) return fail("stage_rows = 3 needs the stage schedule and hold_steps = 1");
            if (in.samples.empty() || in.samples.size() % ((size_t)in.channels * in.stage_rows) != 0) return fail("input samples are not whole rows");
            if (in.node_channel.size() != (size_t)N || in.node_gain.size() != (size_t)N) return fail("input maps need one entry per node");
            for (int c : in.node_channel) if (c >= in.channels) return fail("input channel out of range");
        }
        // Экземпляр не делит дуг и популяций с другими: на этом держится вся изоляция.
        std::vector<int> instStart = req.instance_start;
        if (instStart.empty()) instStart = { 0, N };
        const int nInst = (int)instStart.size() - 1;
        if (nInst < 1 || instStart.front() != 0 || instStart.back() != N) return fail("instances must cover all nodes");
        for (int q = 0; q < nInst; ++q) {
            const int i0 = instStart[(size_t)q], i1 = instStart[(size_t)q + 1];
            if (i0 >= i1) return fail("an empty instance");
            if (nInst > 1)
                for (int e = req.edge_start[(size_t)i0]; e < req.edge_start[(size_t)i1]; ++e)
                    if (req.edge_src[(size_t)e] < i0 || req.edge_src[(size_t)e] >= i1) return fail("an arc joins two instances");
        }
        const auto instance_of = [&](int node) {
            return (int)(std::upper_bound(instStart.begin(), instStart.end(), node) - instStart.begin()) - 1;
        };

        const NeuralDevices& dv = req.devices;
        const int dS = dv.states, dP = dv.params, nDev = dS > 0 ? (int)dv.a.size() : 0;
        std::vector<int> devStart, devList;
        if (dS > 0) {
            if (dS > 4 || dP < 0) return fail("device states or parameters out of range");
            if (req.schedule != NeuralSchedule::Stage) return fail("devices on links need the stage schedule");
            if (req.lyapunov.enabled) return fail("the Lyapunov exponent does not work with devices on links yet");
            if (dv.rhs_body.empty() || dv.current_body.empty()) return fail("a device needs rhs_body and current_body");
            const size_t n = (size_t)nDev;
            if (dv.b.size() != n || dv.var.size() != n || dv.two_way.size() != n || dv.gain.size() != n
                || dv.values.size() != n * dP || dv.initial.size() != n * dS || dv.s_min.size() != (size_t)dS || dv.s_max.size() != (size_t)dS)
                return fail("device arrays disagree on the number of devices");
            for (int q = 0; q < dS; ++q) if (!(dv.s_max[(size_t)q] >= dv.s_min[(size_t)q])) return fail("bad device state bounds");
            devStart.assign((size_t)N + 1, 0);
            for (int d = 0; d < nDev; ++d) {
                const int a = dv.a[(size_t)d], b = dv.b[(size_t)d];
                if (a < 0 || a >= N || b < 0 || b >= N) return fail("a device points outside the node list");
                if (dv.var[(size_t)d] < 0 || dv.var[(size_t)d] >= nx) return fail("a device variable is out of range");
                if (instance_of(a) != instance_of(b)) return fail("a device joins two instances");
                ++devStart[(size_t)b + 1];
                if (dv.two_way[(size_t)d]) ++devStart[(size_t)a + 1];
            }
            for (int i = 0; i < N; ++i) devStart[(size_t)i + 1] += devStart[(size_t)i];
            devList.resize((size_t)devStart.back());
            std::vector<int> put(devStart.begin(), devStart.end() - 1);
            for (int d = 0; d < nDev; ++d) {
                devList[(size_t)put[(size_t)dv.b[(size_t)d]]++] = d * 2 + 1;
                if (dv.two_way[(size_t)d]) devList[(size_t)put[(size_t)dv.a[(size_t)d]]++] = d * 2;
            }
            for (int d : dv.watch) if (d < 0 || d >= nDev) return fail("a watched device does not exist");
        }
        const int nDevWatch = dS > 0 ? (int)dv.watch.size() : 0;
        if ((req.resume || req.keep_checkpoint) && (req.observe.order_var >= 0 || req.lyapunov.enabled)) return fail("a checkpoint does not work with R or the Lyapunov exponent");

        const NeuralObservation& ob = req.observe;
        for (int v : ob.vars) if (v < 0 || v >= nx) return fail("observed variable out of range");
        for (int i : ob.nodes) if (i < 0 || i >= N) return fail("observed node out of range");
        const int nPops = ob.pop_start.empty() ? 0 : (int)ob.pop_start.size() - 1;
        if (nPops > 0) {
            if (ob.pop_start.front() != 0 || ob.pop_start.back() != N) return fail("populations must cover all nodes");
            for (int p = 0; p < nPops; ++p)
                if (ob.pop_start[(size_t)p] >= ob.pop_start[(size_t)p + 1]) return fail("an empty population");
            for (int p = 0; p < nPops; ++p)
                if (instance_of(ob.pop_start[(size_t)p]) != instance_of(ob.pop_start[(size_t)p + 1] - 1)) return fail("a population crosses two instances");
        }
        if (ob.order_var >= nx) return fail("the order parameter variable is out of range");
        if (ob.order_var >= 0 && nPops == 0) return fail("the order parameter needs populations");
        if (ob.order_var >= 0 && !req.reset_at.empty()) return fail("the order parameter does not work across state resets");
        for (size_t q = 0; q < req.reset_at.size(); ++q)
            if (req.reset_at[q] < 0 || (q > 0 && req.reset_at[q] <= req.reset_at[q - 1])) return fail("reset steps must increase");

        const long long skipSteps  = (long long)steps_from_time_size_t(req.transient, req.h);
        const long long workSteps  = (long long)steps_from_time_size_t(req.t_max, req.h);
        const long long totalSteps = skipSteps + workSteps;
        if (totalSteps <= 0) return fail("t_max / h gives no steps");

        // Клоны лежат после основной сети, Nt = шаг столбца SoA. Без показателя Nt = N и всё как было.
        const NeuralLyapunov& ly = req.lyapunov;
        const bool lyap = ly.enabled;
        if (lyap && (long long)N * 2 > (std::numeric_limits<int>::max)() / 2) return fail("too many nodes for the Lyapunov clone");
        if (lyap && (long long)n_arcs * 2 > (std::numeric_limits<int>::max)()) return fail("too many arcs for the Lyapunov clone");
        const int Nt = lyap ? 2 * N : N, arcsT = lyap ? 2 * n_arcs : n_arcs;
        const size_t nStatesT = (size_t)Nt * (size_t)nx;
        const bool closed = req.feedback.from_row >= 0;
        const bool closedLyap = closed && lyap;
        long long lyM = 0, lyNext = 0, lyCountAfter = skipSteps;
        if (lyap) {
            if (!(ly.eps > 0.0) || !(ly.interval > 0.0) || !(ly.settle >= 0.0) ||
                !std::isfinite(ly.eps) || !std::isfinite(ly.interval) || !std::isfinite(ly.settle)) return fail("bad Lyapunov settings");
            if (!req.reset_at.empty()) return fail("the Lyapunov exponent does not work across state resets");
            if (!ly.instance_seed.empty() && (int)ly.instance_seed.size() != nInst) return fail("instance_seed needs one seed per instance");
            if (ly.interval / req.h > (double)workSteps + 0.5) return fail("the record is shorter than the Lyapunov interval");
            lyM = std::llround(ly.interval / req.h);
            if (lyM < 1 || std::fabs((double)lyM * req.h - ly.interval) > 1e-9 * ly.interval)
                return fail("the Lyapunov interval must be a multiple of h");
            if (workSteps / lyM < 1) return fail("the record is shorter than the Lyapunov interval");
            // Перенормировка всегда попадает в начало записи.
            if (!closedLyap) {
                const long long beforeRecord = skipSteps / lyM;
                const double settleIntervals = ly.settle / ((double)lyM * req.h);
                lyNext = skipSteps - (settleIntervals >= (double)beforeRecord ? beforeRecord : std::llround(settleIntervals)) * lyM;
            }
        }

        const int nVars = (int)ob.vars.size(), nSel = nVars ? (int)ob.nodes.size() : 0, nMean = nVars ? nPops : 0;
        const int nOrder = ob.order_var >= 0 ? nPops : 0;
        const bool wantSpread = ob.spread && nMean > 0;
        const long long nPointsLL = (nSel + nMean + nOrder + nDevWatch > 0) ? workSteps / req.pre_scaller + 1 : 0;
        // Строка записи: выбранные узлы, суммы, суммы квадратов, состояния устройств.
        const size_t rowElems = ((size_t)nSel + (size_t)nMean * (wantSpread ? 2 : 1)) * (size_t)nVars + (size_t)nDevWatch * dS;
        if ((double)nPointsLL * (double)rowElems * sizeof(double) > (double)req.max_output_bytes)
            return fail("the recorded output is above max_output_bytes: raise the decimation or select less");
        const int nPoints = (int)nPointsLL;

        // Замкнутый выход: строка входа p считается по записи точки p, поэтому сетки входа и записи совпадают.
        const NeuralFeedback& fb = req.feedback;
        const int fbM = (int)fb.cols.size(), fbK = (int)fb.channels.size();
        if (closed) {
            if (in.channels < 1 || in.stage_rows != 1) return fail("a closed-loop readout needs an input of one row per sample");
            if (skipSteps != 0 || (long long)req.pre_scaller != in.hold_steps) return fail("a closed-loop readout needs no transient and the record on the input samples");
            if (nInst != 1) return fail("a closed-loop readout drives one network");
            if (fbM < 1 || fbK < 1 || fb.w.size() != (size_t)fbK * (size_t)(fbM + 1)) return fail("bad closed-loop readout");
            for (int col : fb.cols) if (col < 0 || (size_t)col >= (size_t)nSel * (size_t)nVars) return fail("a closed-loop feature is not in the record");
            for (int ch : fb.channels) if (ch < 0 || ch >= in.channels) return fail("a closed-loop channel is out of range");
        }
        if (closedLyap) {
            if (in.channels > (std::numeric_limits<int>::max)() / 2) return fail("too many closed-loop channels for the Lyapunov clone");
            if (lyM % in.hold_steps != 0) return fail("the closed-loop Lyapunov interval must be a multiple of the input sample time");
            if (fb.from_row > totalSteps / in.hold_steps) return fail("feedback starts past the Lyapunov record");
            if (in.samples.size() / (size_t)in.channels <= (size_t)(totalSteps / in.hold_steps))
                return fail("the closed-loop Lyapunov record needs a full input grid");
            lyNext = fb.from_row * in.hold_steps;
            const long long intervals = (totalSteps - lyNext) / lyM;
            const double settleIntervals = std::ceil(ly.settle / ((double)lyM * req.h));
            if (!(settleIntervals < (double)intervals)) return fail("the closed-loop record needs a full Lyapunov interval after settling");
            lyCountAfter = lyNext + (long long)settleIntervals * lyM;
        }
        const size_t inputStride = (size_t)in.channels * (closedLyap ? 2 : 1);

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync
        if (!compile_neural_module(true, req, err)) return fail(err);

        const bool stage = req.schedule == NeuralSchedule::Stage, heun = req.schedule == NeuralSchedule::Heun;
        const int phases = stage ? 4 : heun ? 2 : 1;

        // Отсчёты копятся на устройстве и сливаются на хост порциями: память устройства не растёт с числом точек.
        const int flushRows = nPoints > 0 ? (int)std::min<long long>(nPoints, std::max<long long>(1, (long long)(kNeuralFlushBytes / std::max<size_t>(1, rowElems * sizeof(numb))))) : 0;

        OrderDevBuf d_values, d_buf[4], d_eStart, d_eSrc, d_eW, d_eLaw, d_inCh, d_inGain, d_inRows, d_bad, d_status,
                    d_sel, d_vars, d_gStart, d_gPop, d_out, d_prev, d_crossCount, d_crossNode, d_crossTime,
                    d_lyStart, d_lyDist, d_lyScale,
                    d_dev[4], d_devStart, d_devList, d_devA, d_devB, d_devVar, d_devGain, d_devValues, d_devClips, d_devWatch;
        auto dev = [&](OrderDevBuf& b, size_t bytes, const char* what) {
            res.device_bytes += bytes;
            return b.alloc(std::max<size_t>(bytes, sizeof(numb)), what, err);
        };
        auto up = [&](OrderDevBuf& b, const void* src, size_t bytes, const char* what) {
            if (bytes == 0) return true;
            if (cudaMemcpy(b.p, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) return true;
            err = std::string("memcpy ") + what; return false;
        };
        const size_t stateBytes = nStatesT * sizeof(numb);
        const int buffers = stage ? 4 : heun ? 3 : 2;
        if (!dev(d_values, (size_t)Nt * req.amountOfValues * sizeof(numb), "values")) return fail(err);
        for (int k = 0; k < buffers; ++k) if (!dev(d_buf[k], stateBytes, "state")) return fail(err);
        if (!dev(d_eStart, ((size_t)Nt + 1) * sizeof(int), "edgeStart")) return fail(err);
        if (!dev(d_eSrc, (size_t)arcsT * sizeof(int), "edgeSrc")) return fail(err);
        if (!dev(d_eW, (size_t)arcsT * sizeof(numb), "edgeW")) return fail(err);
        if (req.n_laws > 1 && !dev(d_eLaw, (size_t)arcsT * sizeof(int), "edgeLaw")) return fail(err);
        if (!dev(d_bad, sizeof(int), "badFlag")) return fail(err);
        if (!dev(d_status, (size_t)Nt * sizeof(int), "status")) return fail(err);

        // Вторая половина массива: у клона те же значения и связи, соседи сдвинуты на N.
        const auto up_to = [&](void* dst, const void* src, size_t bytes, const char* what) {
            if (bytes == 0 || cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) return true;
            err = std::string("memcpy ") + what; return false;
        };
        std::vector<numb> soa(nStatesT);
        {
            const std::vector<numb> hv = to_numb(req.values), hw = to_numb(req.edge_w);
            for (int i = 0; i < N; ++i) for (int k = 0; k < nx; ++k)
                soa[(size_t)k * Nt + i] = (numb)req.initial_conditions[(size_t)i * nx + k];
            if (!up(d_values, hv.data(), hv.size() * sizeof(numb), "values")) return fail(err);
            for (int k = 0; k < 2; ++k) if (!up(d_buf[k], soa.data(), stateBytes, "state")) return fail(err);
            if (!up(d_eStart, req.edge_start.data(), req.edge_start.size() * sizeof(int), "edgeStart")) return fail(err);
            if (!up(d_eSrc, req.edge_src.data(), (size_t)n_arcs * sizeof(int), "edgeSrc")) return fail(err);
            if (!up(d_eW, hw.data(), (size_t)n_arcs * sizeof(numb), "edgeW")) return fail(err);
            if (req.n_laws > 1 && !up(d_eLaw, req.edge_law.data(), (size_t)n_arcs * sizeof(int), "edgeLaw")) return fail(err);
            if (lyap) {
                std::vector<int> es((size_t)N), src((size_t)n_arcs);
                for (int i = 0; i < N; ++i) es[(size_t)i] = req.edge_start[(size_t)i + 1] + n_arcs;
                for (int e = 0; e < n_arcs; ++e) src[(size_t)e] = req.edge_src[(size_t)e] + N;
                if (!up_to(d_values.as<numb>() + (size_t)N * req.amountOfValues, hv.data(), hv.size() * sizeof(numb), "values")
                    || !up_to(d_eStart.as<int>() + (size_t)N + 1, es.data(), (size_t)N * sizeof(int), "edgeStart")
                    || !up_to(d_eSrc.as<int>() + n_arcs, src.data(), (size_t)n_arcs * sizeof(int), "edgeSrc")
                    || !up_to(d_eW.as<numb>() + n_arcs, hw.data(), (size_t)n_arcs * sizeof(numb), "edgeW")
                    || (req.n_laws > 1 && !up_to(d_eLaw.as<int>() + n_arcs, req.edge_law.data(), (size_t)n_arcs * sizeof(int), "edgeLaw")))
                    return fail(err);
            }
        }
        if (cudaMemset(d_status.p, 0, (size_t)Nt * sizeof(int)) != cudaSuccess || cudaMemset(d_bad.p, 0, sizeof(int)) != cudaSuccess)
            return fail("memset status");

        const size_t devBytes = (size_t)nDev * dS * sizeof(numb);
        std::vector<int> devWatch;
        if (dS > 0) {
            std::vector<numb> s0((size_t)nDev * dS);
            for (int d = 0; d < nDev; ++d) for (int k = 0; k < dS; ++k) s0[(size_t)k * nDev + d] = (numb)dv.initial[(size_t)d * dS + k];
            const std::vector<numb> gain = to_numb(dv.gain), values = to_numb(dv.values);
            const size_t iBytes = (size_t)nDev * sizeof(int);
            for (int k = 0; k < 4; ++k) if (!dev(d_dev[k], devBytes, "device state")) return fail(err);
            if (!up(d_dev[0], s0.data(), devBytes, "device state")
                || !dev(d_devStart, devStart.size() * sizeof(int), "devStart") || !up(d_devStart, devStart.data(), devStart.size() * sizeof(int), "devStart")
                || !dev(d_devList, devList.size() * sizeof(int), "devList") || !up(d_devList, devList.data(), devList.size() * sizeof(int), "devList")
                || !dev(d_devA, iBytes, "devA") || !up(d_devA, dv.a.data(), iBytes, "devA")
                || !dev(d_devB, iBytes, "devB") || !up(d_devB, dv.b.data(), iBytes, "devB")
                || !dev(d_devVar, iBytes, "devVar") || !up(d_devVar, dv.var.data(), iBytes, "devVar")
                || !dev(d_devGain, gain.size() * sizeof(numb), "devGain") || !up(d_devGain, gain.data(), gain.size() * sizeof(numb), "devGain")
                || !dev(d_devValues, values.size() * sizeof(numb), "devValues") || !up(d_devValues, values.data(), values.size() * sizeof(numb), "devValues")
                || !dev(d_devClips, sizeof(int), "devClips")) return fail(err);
            if (cudaMemset(d_devClips.p, 0, sizeof(int)) != cudaSuccess) return fail("memset device clips");
            for (int d : dv.watch) for (int k = 0; k < dS; ++k) devWatch.push_back(k * nDev + d);
            if (!devWatch.empty() && (!dev(d_devWatch, devWatch.size() * sizeof(int), "devWatch") || !up(d_devWatch, devWatch.data(), devWatch.size() * sizeof(int), "devWatch"))) return fail(err);
        }

        std::vector<numb> inRows;
        long long nInRows = 0;
        if (in.channels > 0) {
            inRows = to_numb(in.samples);
            nInRows = (long long)(in.samples.size() / ((size_t)in.channels * in.stage_rows));
            if (closedLyap) {
                if (inRows.size() > inRows.max_size() / 2) return fail("too many samples for the closed-loop Lyapunov clone");
                std::vector<numb> both(inRows.size() * 2);
                for (size_t row = 0; row < (size_t)nInRows; ++row) {
                    const numb* src = inRows.data() + row * in.channels;
                    numb* dst = both.data() + row * inputStride;
                    std::copy_n(src, in.channels, dst);
                    std::copy_n(src, in.channels, dst + in.channels);
                }
                inRows.swap(both);
            }
            const std::vector<numb> gain = to_numb(in.node_gain);
            if (!dev(d_inRows, inRows.size() * sizeof(numb), "inSamples")) return fail(err);
            if (!dev(d_inCh, (size_t)Nt * sizeof(int), "inChannel")) return fail(err);
            if (!dev(d_inGain, (size_t)Nt * sizeof(numb), "inGain")) return fail(err);
            if (!up(d_inRows, inRows.data(), inRows.size() * sizeof(numb), "inSamples")) return fail(err);
            if (!up(d_inCh, in.node_channel.data(), (size_t)N * sizeof(int), "inChannel")) return fail(err);
            if (!up(d_inGain, gain.data(), (size_t)N * sizeof(numb), "inGain")) return fail(err);
            if (lyap) {
                std::vector<int> cloneChannels;
                if (closedLyap) {
                    cloneChannels = in.node_channel;
                    for (int& channel : cloneChannels) if (channel >= 0) channel += in.channels;
                }
                if (!up_to(d_inCh.as<int>() + N, closedLyap ? cloneChannels.data() : in.node_channel.data(), (size_t)N * sizeof(int), "inChannel")
                    || !up_to(d_inGain.as<numb>() + N, gain.data(), (size_t)N * sizeof(numb), "inGain")) return fail(err);
            }
        }
        OrderDevBuf d_fbCols, d_fbCh, d_fbW, d_cloneRow;
        if (closed && (!dev(d_fbCols, (size_t)fbM * sizeof(int), "feedbackCols") || !up(d_fbCols, fb.cols.data(), (size_t)fbM * sizeof(int), "feedbackCols")
                       || !dev(d_fbCh, (size_t)fbK * sizeof(int), "feedbackChannels") || !up(d_fbCh, fb.channels.data(), (size_t)fbK * sizeof(int), "feedbackChannels")
                       || !dev(d_fbW, fb.w.size() * sizeof(double), "feedbackW") || !up(d_fbW, fb.w.data(), fb.w.size() * sizeof(double), "feedbackW")))
            return fail(err);
        if (closedLyap && !dev(d_cloneRow, (size_t)nSel * nVars * sizeof(numb), "clone feedback row")) return fail(err);

        std::vector<int> lyStart, lyInst;
        if (lyap) {
            for (int q = 0; q < nInst; ++q)
                for (int i = instStart[(size_t)q]; i < instStart[(size_t)q + 1]; i += kNeuralMeanGroup) { lyStart.push_back(i); lyInst.push_back(q); }
            lyStart.push_back(N);
            if (!dev(d_lyStart, lyStart.size() * sizeof(int), "lyapStart") || !dev(d_lyDist, lyInst.size() * sizeof(numb), "lyapDist")
                || !dev(d_lyScale, lyInst.size() * sizeof(numb), "lyapScale")) return fail(err);
            if (!up(d_lyStart, lyStart.data(), lyStart.size() * sizeof(int), "lyapStart")) return fail(err);
        }

        // Популяции режутся на группы, чтобы atomicAdd шёл один на группу.
        std::vector<int> gStart, gPop;
        if (nMean > 0) {
            for (int p = 0; p < nPops; ++p)
                for (int i = ob.pop_start[(size_t)p]; i < ob.pop_start[(size_t)p + 1]; i += kNeuralMeanGroup) { gStart.push_back(i); gPop.push_back(p); }
            gStart.push_back(N);
            if (!dev(d_gStart, gStart.size() * sizeof(int), "groupStart") || !dev(d_gPop, gPop.size() * sizeof(int), "groupPop")) return fail(err);
            if (!up(d_gStart, gStart.data(), gStart.size() * sizeof(int), "groupStart") || !up(d_gPop, gPop.data(), gPop.size() * sizeof(int), "groupPop")) return fail(err);
        }
        if (nVars > 0) {
            if (!dev(d_vars, (size_t)nVars * sizeof(int), "vars") || !up(d_vars, ob.vars.data(), (size_t)nVars * sizeof(int), "vars")) return fail(err);
        }
        if (nSel > 0) {
            if (!dev(d_sel, (size_t)nSel * sizeof(int), "selected") || !up(d_sel, ob.nodes.data(), (size_t)nSel * sizeof(int), "selected")) return fail(err);
        }
        if (flushRows > 0 && !dev(d_out, (size_t)flushRows * rowElems * sizeof(numb), "out")) return fail(err);
        // Список пересечений: за запись узел даёт не больше одного, поэтому хост забирает его до переполнения.
        const int crossCap = (int)(std::max)((long long)N * 2, std::clamp((long long)N * 64, 1LL << 20, 1LL << 26));
        const size_t crossLimit = req.max_output_bytes / (sizeof(int) + sizeof(double));
        if (nOrder > 0) {
            const int zero = 0;
            if (!dev(d_prev, (size_t)N * sizeof(numb), "orderPrev") || !dev(d_crossCount, sizeof(int), "crossCount")
                || !dev(d_crossNode, (size_t)crossCap * sizeof(int), "crossNode")
                || !dev(d_crossTime, (size_t)crossCap * sizeof(numb), "crossTime")) return fail(err);
            if (!up(d_crossCount, &zero, sizeof(int), "crossCount")) return fail(err);
        }
        std::vector<int> crossNode;
        std::vector<double> crossTime;
        long long crossWorst = 0;
        bool crossFull = false;
        double crossEnd = (double)(skipSteps + (long long)(nPoints > 0 ? nPoints - 1 : 0) * req.pre_scaller) * req.h;
        const auto flush_cross = [&]() {
            int n = 0;
            const int zero = 0;
            if (cudaMemcpy(&n, d_crossCount.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            n = (std::min)(n, crossCap);
            if (n > 0) {
                std::vector<int> hn((size_t)n);
                std::vector<numb> ht((size_t)n);
                if (cudaMemcpy(hn.data(), d_crossNode.p, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess
                    || cudaMemcpy(ht.data(), d_crossTime.p, (size_t)n * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
                crossNode.insert(crossNode.end(), hn.begin(), hn.end());
                crossTime.insert(crossTime.end(), ht.begin(), ht.end());
            }
            crossWorst = 0;
            return cudaMemcpy(d_crossCount.p, &zero, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
        };

        std::vector<double> traces, means, spreads, orders, deviceTrace;
        if (!devWatch.empty()) deviceTrace.assign((size_t)nPoints * devWatch.size(), std::numeric_limits<double>::quiet_NaN());
        if (nSel > 0) traces.assign((size_t)nPoints * nSel * nVars, std::numeric_limits<double>::quiet_NaN());
        if (nMean > 0) means.assign((size_t)nPoints * nMean * nVars, std::numeric_limits<double>::quiet_NaN());
        if (wantSpread) spreads.assign((size_t)nPoints * nMean * nVars, std::numeric_limits<double>::quiet_NaN());
        std::vector<numb> hostRows((size_t)flushRows * rowElems);
        int rowsOnDevice = 0, rowsFlushed = 0;
        const auto flush = [&]() {
            if (rowsOnDevice == 0) return true;
            if (cudaMemcpy(hostRows.data(), d_out.p, (size_t)rowsOnDevice * rowElems * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (int r = 0; r < rowsOnDevice; ++r) {
                const numb* row = hostRows.data() + (size_t)r * rowElems;
                const size_t p = (size_t)rowsFlushed + r;
                for (size_t e = 0; e < (size_t)nSel * nVars; ++e) traces[p * nSel * nVars + e] = (double)row[e];
                const numb* pops = row + (size_t)nSel * nVars;
                for (int q = 0; q < nMean; ++q) {
                    const double size = (double)(ob.pop_start[(size_t)q + 1] - ob.pop_start[(size_t)q]);
                    for (int v = 0; v < nVars; ++v) means[(p * nMean + q) * nVars + v] = (double)pops[(size_t)q * nVars + v] / size;
                }
                const numb* tail = pops + (size_t)nMean * nVars;
                if (wantSpread)
                    for (int q = 0; q < nMean; ++q) {
                        const double size = (double)(ob.pop_start[(size_t)q + 1] - ob.pop_start[(size_t)q]);
                        for (int v = 0; v < nVars; ++v) {
                            const double m = (double)pops[(size_t)q * nVars + v] / size, m2 = (double)tail[(size_t)q * nVars + v] / size;
                            spreads[(p * nMean + q) * nVars + v] = std::sqrt((std::max)(0.0, m2 - m * m));
                        }
                    }
                const numb* devs = tail + (wantSpread ? (size_t)nMean * nVars : 0);
                for (size_t q = 0; q < devWatch.size(); ++q) deviceTrace[p * devWatch.size() + q] = (double)devs[q];
            }
            rowsFlushed += rowsOnDevice;
            rowsOnDevice = 0;
            return true;
        };

        const int block = req.block_size > 0 ? req.block_size : kNeuralDefaultBlock;
        unsigned blocks = (unsigned)(((long long)N + block - 1) / block);
        const double workPerStep = ((double)Nt + (double)arcsT) * phases;
        long long syncEvery = (long long)(kNeuralWorkPerSync / workPerStep);
        if (syncEvery < 1) syncEvery = 1;
        if (req.steps_per_sync > 0 && req.steps_per_sync < syncEvery) syncEvery = req.steps_per_sync;
        res.block_size = block;
        res.steps_per_sync = syncEvery;

        StepArgs sa{};
        sa.values = d_values.as<numb>();
        sa.edgeStart = d_eStart.as<int>(); sa.edgeSrc = d_eSrc.as<int>(); sa.edgeW = d_eW.as<numb>();
        sa.edgeLaw = req.n_laws > 1 ? d_eLaw.as<int>() : nullptr;
        sa.inChannel = in.channels > 0 ? d_inCh.as<int>() : nullptr;
        sa.inGain = in.channels > 0 ? d_inGain.as<numb>() : nullptr;
        sa.badFlag = d_bad.as<int>(); sa.status = d_status.as<int>();
        sa.h = (numb)req.h; sa.maxValue = (numb)req.max_value;
        sa.nNodes = Nt; sa.nRun = N; sa.inVar = in.var;
        if (dS > 0) {
            sa.devStart = d_devStart.as<int>(); sa.devList = d_devList.as<int>(); sa.devA = d_devA.as<int>(); sa.devB = d_devB.as<int>();
            sa.devVar = d_devVar.as<int>(); sa.devGain = d_devGain.as<numb>(); sa.devValues = d_devValues.as<numb>();
            sa.d0 = d_dev[0].as<numb>(); sa.d1 = d_dev[1].as<numb>(); sa.d2 = d_dev[2].as<numb>(); sa.d3 = d_dev[3].as<numb>();
            sa.devClips = d_devClips.as<int>(); sa.nDev = nDev;
            for (int k = 0; k < dS; ++k) { sa.devMin[k] = (numb)dv.s_min[(size_t)k]; sa.devMax[k] = (numb)dv.s_max[(size_t)k]; }
        }

        std::vector<long long> instStop((size_t)nInst, -1);

        // Посев и перенормировка идут на границе чанка, поэтому от его длины не зависят.
        bool lySeeded = false;
        std::vector<double> lyLog;
        const auto lyap_seed = [&](numb* state) -> bool {
            std::vector<numb> host(nStatesT);
            if (cudaMemcpy(host.data(), state, stateBytes, cudaMemcpyDeviceToHost) != cudaSuccess) { err = "memcpy D2H state"; return false; }
            for (int q = 0; q < nInst; ++q) {
                const int i0 = instStart[(size_t)q], i1 = instStart[(size_t)q + 1];
                std::vector<double> dir((size_t)(i1 - i0) * nx);
                double norm = 0.0;
                const uint64_t seed = ly.instance_seed.empty() ? ly.seed : ly.instance_seed[(size_t)q];
                for (int i = i0; i < i1; ++i) for (int k = 0; k < nx; ++k) {
                    const double v = neural_lyap_normal(seed, (uint64_t)(i - i0), (uint64_t)k);
                    dir[(size_t)(i - i0) * nx + k] = v;
                    norm += v * v;
                }
                const double c = ly.eps * std::sqrt((double)(i1 - i0) * nx / norm);
                for (int i = i0; i < i1; ++i) for (int k = 0; k < nx; ++k)
                    host[(size_t)k * Nt + N + i] = host[(size_t)k * Nt + i] + (numb)(c * dir[(size_t)(i - i0) * nx + k]);
            }
            return up_to(state, host.data(), stateBytes, "state");
        };
        const auto lyap_renorm = [&](numb* state, bool counted) -> bool {
            const int nGroups = (int)lyInst.size();
            const unsigned b = (unsigned)((nGroups + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
            LyapArgs la{ state, d_lyStart.as<int>(), d_lyDist.as<numb>(), nullptr, N, nGroups };
            void* args[] = { &la };
            CUresult r = cuLaunchKernel(cached_neural.lyap, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural lyap): " + cu_err(r); return false; }
            std::vector<numb> dist((size_t)nGroups), scale((size_t)nGroups, (numb)1);
            cudaDeviceSynchronize();
            if (cudaMemcpy(dist.data(), d_lyDist.p, dist.size() * sizeof(numb), cudaMemcpyDeviceToHost) != cudaSuccess) { err = "memcpy D2H lyap"; return false; }
            std::vector<double> sum((size_t)nInst, 0.0), factor((size_t)nInst, 1.0);
            for (int g = 0; g < nGroups; ++g) sum[(size_t)lyInst[(size_t)g]] += (double)dist[(size_t)g];
            for (int q = 0; q < nInst; ++q) {
                double lg = std::numeric_limits<double>::quiet_NaN();
                if (instStop[(size_t)q] < 0 && std::isfinite(sum[(size_t)q])) {
                    const double d0 = ly.eps * std::sqrt((double)(instStart[(size_t)q + 1] - instStart[(size_t)q]) * nx), d = std::sqrt(sum[(size_t)q]);
                    lg = d > 0.0 ? std::log(d / d0) : -std::numeric_limits<double>::infinity();
                    factor[(size_t)q] = d > 0.0 ? d0 / d : 0.0;
                }
                if (counted) lyLog.push_back(lg);
            }
            for (int g = 0; g < nGroups; ++g) scale[(size_t)g] = (numb)factor[(size_t)lyInst[(size_t)g]];
            if (!up_to(d_lyScale.p, scale.data(), scale.size() * sizeof(numb), "lyapScale")) return false;
            la.scale = d_lyScale.as<numb>();
            r = cuLaunchKernel(cached_neural.lyap, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural lyap): " + cu_err(r); return false; }
            return true;
        };
        // Первое событие сеет клонов, остальные перенормируют. В записи считается блок, закончившийся после её начала.
        const auto lyap_event = [&](long long step, numb* state) -> bool {
            if (!lySeeded) {
                if (!lyap_seed(state)) return false;
                lySeeded = true;
                sa.nRun = Nt;
                blocks = (unsigned)(((long long)Nt + block - 1) / block);
            } else if (!lyap_renorm(state, step > lyCountAfter)) return false;
            lyNext += lyM;
            return true;
        };

        bool trackFirst = true;
        const auto track = [&](const numb* state, double t) -> bool {
            if (crossFull) return true;
            if (crossWorst + N > crossCap) {
                cudaDeviceSynchronize();
                if (!flush_cross()) { err = "memcpy crossings"; return false; }
                if (crossNode.size() >= crossLimit) { crossFull = true; crossEnd = t; return true; }
            }
            OrderArgs oa{ state + (size_t)ob.order_var * Nt, d_prev.as<numb>(), d_crossCount.as<int>(), d_crossNode.as<int>(), d_crossTime.as<numb>(),
                          (numb)ob.order_threshold, (numb)t, (numb)((double)req.pre_scaller * req.h), N, trackFirst ? 1 : 0, crossCap, 0 };
            void* args[] = { &oa };
            const unsigned b = (unsigned)((N + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
            CUresult r = cuLaunchKernel(cached_neural.order, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural order): " + cu_err(r); return false; }
            ++res.launches;
            crossWorst += N;
            trackFirst = false;
            return true;
        };
        const auto record = [&](const numb* state) -> bool {
            numb* row = d_out.as<numb>() + (size_t)rowsOnDevice * rowElems;
            if (nSel > 0) {
                ObserveArgs oa{ state, d_sel.as<int>(), d_vars.as<int>(), row, Nt, nSel, nVars, 0 };
                void* args[] = { &oa };
                const unsigned b = (unsigned)((nSel + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.observe, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural observe): " + cu_err(r); return false; }
                ++res.launches;
            }
            if (nMean > 0) {
                numb* sums = row + (size_t)nSel * nVars;
                numb* squares = wantSpread ? sums + (size_t)nMean * nVars : nullptr;
                if (cudaMemsetAsync(sums, 0, (size_t)nMean * nVars * (wantSpread ? 2 : 1) * sizeof(numb), 0) != cudaSuccess) { err = "memset means"; return false; }
                MeanArgs ma{ state, d_gStart.as<int>(), d_gPop.as<int>(), d_vars.as<int>(), sums, squares, Nt, (int)gPop.size(), nVars, 0 };
                void* args[] = { &ma };
                const unsigned b = (unsigned)((gPop.size() + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.mean, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural mean): " + cu_err(r); return false; }
                ++res.launches;
            }
            if (nOrder > 0) {
                const long long point = (long long)rowsFlushed + rowsOnDevice;
                if (!track(state, (double)(skipSteps + point * req.pre_scaller) * req.h)) return false;
            }
            if (!devWatch.empty()) {
                numb* tail = row + ((size_t)nSel + (size_t)nMean * (wantSpread ? 2 : 1)) * nVars;
                WeightArgs wa{ d_dev[0].as<numb>(), d_devWatch.as<int>(), tail, (int)devWatch.size(), 0 };
                void* args[] = { &wa };
                const unsigned b = (unsigned)((devWatch.size() + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
                CUresult r = cuLaunchKernel(cached_neural.weight, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
                if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural device watch): " + cu_err(r); return false; }
                ++res.launches;
            }
            ++rowsOnDevice;
            return true;
        };
        // Сразу после записи точки p, до её сброса на хост: строка входа p нужна уже следующему шагу.
        struct FeedbackArgs { const numb* row; const int* cols; const int* channels; const double* w; numb* out; int M, K; };
        constexpr unsigned kFeedbackBlock = 256;   // NEURAL_FEEDBACK_BLOCK в шаблоне
        const auto feed_row = [&](const numb* row, numb* input) -> bool {
            FeedbackArgs fa{ row, d_fbCols.as<int>(), d_fbCh.as<int>(), d_fbW.as<double>(), input, fbM, fbK };
            void* args[] = { &fa };
            CUresult r = cuLaunchKernel(cached_neural.feedback, (unsigned)fbK, 1, 1, kFeedbackBlock, 1, 1, 0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural feedback): " + cu_err(r); return false; }
            ++res.launches;
            return true;
        };
        long long fedLast = -1;
        const auto feed = [&](long long point) -> bool {
            if (!closed || point < fb.from_row || point >= nInRows) return true;
            if (!feed_row(d_out.as<numb>() + (size_t)(rowsOnDevice - 1) * rowElems,
                          d_inRows.as<numb>() + (size_t)point * inputStride)) return false;
            fedLast = point;
            return true;
        };
        const auto feed_clone = [&](long long point, const numb* state) -> bool {
            if (point >= nInRows) return true;
            ObserveArgs oa{ state + N, d_sel.as<int>(), d_vars.as<int>(), d_cloneRow.as<numb>(), Nt, nSel, nVars, 0 };
            void* args[] = { &oa };
            const unsigned b = (unsigned)((nSel + kNeuralObserveBlock - 1) / kNeuralObserveBlock);
            const CUresult r = cuLaunchKernel(cached_neural.observe, b, 1, 1, kNeuralObserveBlock, 1, 1, 0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) { err = "cuLaunchKernel(neural clone observe): " + cu_err(r); return false; }
            ++res.launches;
            return feed_row(d_cloneRow.as<numb>(), d_inRows.as<numb>() + (size_t)point * inputStride + in.channels);
        };

        long long done = 0;
        size_t nextReset = 0;
        const std::string ckShape = "cont:" + std::to_string((int)req.schedule) + ":" + std::to_string(N) + ":" + std::to_string(nx) + ":" + std::to_string(nDev)
                                  + ":" + std::to_string(dS) + ":" + std::to_string(nInst) + ":" + std::to_string(sizeof(numb)) + ":" + req.checkpoint_tag;
        if (req.resume) {
            const NeuralCheckpoint& c = *req.resume;
            const std::string refusal = neural_ck_refusal(c.shape, ckShape);
            if (!refusal.empty()) return fail(refusal);
            if (c.step > totalSteps) return fail("the checkpoint is past the end of this run");
            done = c.step;
            size_t at = 0;
            int clips = 0;
            if (!neural_ck_get_host(c, at, instStop.data(), instStop.size() * sizeof(long long)) || !neural_ck_get_host(c, at, &clips, sizeof(int))
                || !neural_ck_get(c, at, d_buf[stage ? 0 : (int)(done & 1)].p, stateBytes) || !neural_ck_get(c, at, d_status.p, (size_t)N * sizeof(int))
                || !neural_ck_get(c, at, d_dev[0].p, devBytes) || at != c.data.size()) return fail("the checkpoint is damaged");
            if (dS > 0 && !up(d_devClips, &clips, sizeof(int), "devClips")) return fail(err);
            while (nextReset < req.reset_at.size() && req.reset_at[nextReset] < done) ++nextReset;
            if (done >= skipSteps) rowsFlushed = (int)(std::min<long long>)((done - skipSteps) / req.pre_scaller + 1, nPoints);
            for (long long v : instStop) res.diverged = res.diverged || v >= 0;
        } else {
            if (skipSteps == 0 && nPoints > 0 && (!record(d_buf[0].as<numb>()) || !feed(0))) return fail(err);
            // Интервал, на который попадает начало записи, начался раньше: пересечения считаются и в transient.
            if (skipSteps > 0 && nOrder > 0 && skipSteps % req.pre_scaller == 0 && !track(d_buf[0].as<numb>(), 0.0)) return fail(err);
        }

        bool stopped = false;
        while (done < totalSteps && !stopped) {
            if (nextReset < req.reset_at.size() && req.reset_at[nextReset] == done) {
                ++nextReset;
                if (cudaMemcpy(d_buf[stage ? 0 : (int)(done & 1)].p, soa.data(), stateBytes, cudaMemcpyHostToDevice) != cudaSuccess)
                    return fail("memcpy reset");
            }
            if (lyap && lyNext == done) {
                numb* state = d_buf[stage ? 0 : (int)(done & 1)].as<numb>();
                if (!lyap_event(done, state)) return fail(err);
                // The input for the next interval must use the renormalised clone.
                if (closedLyap && !feed_clone(done / req.pre_scaller, state)) return fail(err);
            }
            long long left = totalSteps - done;
            if (nextReset < req.reset_at.size()) left = (std::min)(left, req.reset_at[nextReset] - done);
            if (lyap) left = (std::min)(left, lyNext - done);
            const long long count = (std::min)(syncEvery, left);
            for (long long c = 0; c < count; ++c) {
                const int odd = stage ? 0 : (int)(done & 1);
                sa.s0 = d_buf[odd].as<numb>();
                sa.s1 = d_buf[stage ? 1 : 1 - odd].as<numb>();
                sa.s2 = (stage || heun) ? d_buf[2].as<numb>() : nullptr;
                sa.s3 = stage ? d_buf[3].as<numb>() : nullptr;
                const long long inRow = in.channels > 0 ? std::min<long long>(done / in.hold_steps, nInRows - 1) : 0;
                sa.step = (int)(std::min<long long>)(done + 1, (std::numeric_limits<int>::max)());
                void* args[] = { &sa };
                for (int ph = 0; ph < phases; ++ph) {
                    sa.phase = ph;
                    if (in.channels > 0) {
                        const int sub = in.stage_rows == 3 ? (ph == 0 ? 0 : ph == 3 ? 2 : 1) : 0;
                        sa.inSample = d_inRows.as<numb>() + ((size_t)inRow * in.stage_rows + sub) * inputStride;
                    }
                    CUresult r = cuLaunchKernel(cached_neural.step, blocks, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
                    if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(neural): " + cu_err(r));
                    ++res.launches;
                }
                ++done;
                if (nOrder > 0 && done < skipSteps && (skipSteps - done) % req.pre_scaller == 0
                    && !track(d_buf[stage ? 0 : (int)(done & 1)].as<numb>(), (double)done * req.h)) return fail(err);
                if (nPoints > 0 && done >= skipSteps && (done - skipSteps) % req.pre_scaller == 0
                    && (done - skipSteps) / req.pre_scaller < nPoints) {
                    const numb* state = d_buf[stage ? 0 : (int)(done & 1)].as<numb>();
                    if (!record(state) || !feed((done - skipSteps) / req.pre_scaller)) return fail(err);
                    if (closedLyap && lySeeded && done != lyNext && done < totalSteps &&
                        !feed_clone(done / req.pre_scaller, state)) return fail(err);
                    if (rowsOnDevice == flushRows) {
                        cudaDeviceSynchronize();
                        if (!flush()) return fail("memcpy D2H out");
                    }
                }
            }
            cudaDeviceSynchronize();
            ++res.syncs;
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("neural kernel: ") + cudaGetErrorString(ce));
            if (!flush()) return fail("memcpy D2H out");
            if (req.progress) req.progress->store((float)((double)done / (double)totalSteps), std::memory_order_relaxed);
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { res.cancelled = true; stopped = true; break; }
            int bad = 0;
            if (cudaMemcpy(&bad, d_bad.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H flag");
            if (!bad) continue;
            res.diverged = true;
            if (nInst == 1) { instStop[0] = done; stopped = true; continue; }
            // Шаг разлёта экземпляра = минимум по его узлам, от длины чанка он не зависит.
            std::vector<int> st((size_t)Nt);
            if (cudaMemcpy(st.data(), d_status.p, (size_t)Nt * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H status");
            int running = 0;
            for (int q = 0; q < nInst; ++q) {
                if (instStop[(size_t)q] >= 0) continue;
                const int i0 = instStart[(size_t)q], i1 = instStart[(size_t)q + 1];
                int first = 0;
                for (int shift = 0; shift < Nt; shift += N)
                    for (int i = i0 + shift; i < i1 + shift; ++i)
                        if (st[(size_t)i] > 0 && (first == 0 || st[(size_t)i] < first)) first = st[(size_t)i];
                if (first == 0) { ++running; continue; }
                instStop[(size_t)q] = first;
                for (int shift = 0; shift < Nt; shift += N)
                    for (int i = i0 + shift; i < i1 + shift; ++i) if (st[(size_t)i] == 0) st[(size_t)i] = -1;
            }
            int zero = 0;
            if (cudaMemcpy(d_status.p, st.data(), (size_t)Nt * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess
                || cudaMemcpy(d_bad.p, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return fail("memcpy H2D status");
            if (running == 0) stopped = true;
        }
        res.steps_done = done;
        if (closed && fb.from_row < nInRows) {
            cudaDeviceSynchronize();
            const size_t first = (size_t)fb.from_row, n = (size_t)(nInRows - fb.from_row);
            const size_t rowBytes = (size_t)in.channels * sizeof(numb);
            std::vector<numb> rows(n * in.channels);
            const numb* src = d_inRows.as<numb>() + first * inputStride;
            const auto copied = closedLyap ? cudaMemcpy2D(rows.data(), rowBytes, src, inputStride * sizeof(numb), rowBytes, n, cudaMemcpyDeviceToHost)
                                           : cudaMemcpy(rows.data(), src, rows.size() * sizeof(numb), cudaMemcpyDeviceToHost);
            if (copied != cudaSuccess)
                return fail("memcpy D2H closed-loop input");
            // Строки после остановки не считались: там лежит заданный вход, а не выход сети.
            res.feedback_rows.assign(n * (size_t)fbK, std::numeric_limits<double>::quiet_NaN());
            for (size_t p = 0; p < n && (long long)(first + p) <= fedLast; ++p)
                for (int k = 0; k < fbK; ++k) res.feedback_rows[p * (size_t)fbK + (size_t)k] = (double)rows[p * in.channels + (size_t)fb.channels[(size_t)k]];
        }
        res.device_trace = std::move(deviceTrace);
        if (dS > 0) {
            int clips = 0;
            cudaDeviceSynchronize();
            std::vector<numb> s((size_t)nDev * dS);
            if (cudaMemcpy(&clips, d_devClips.p, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess
                || cudaMemcpy(s.data(), d_dev[0].p, devBytes, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H devices");
            res.device_clips = clips;
            res.final_devices.resize(s.size());
            for (int d = 0; d < nDev; ++d) for (int k = 0; k < dS; ++k) res.final_devices[(size_t)d * dS + k] = (double)s[(size_t)k * nDev + d];
        }
        if (req.keep_checkpoint) {
            cudaDeviceSynchronize();
            auto c = std::make_shared<NeuralCheckpoint>();
            c->step = done;
            c->shape = ckShape;
            const int clips = (int)res.device_clips;
            neural_ck_put_host(*c, instStop.data(), instStop.size() * sizeof(long long));
            neural_ck_put_host(*c, &clips, sizeof(int));
            if (!neural_ck_put(*c, d_buf[stage ? 0 : (int)(done & 1)].p, stateBytes) || !neural_ck_put(*c, d_status.p, (size_t)N * sizeof(int))
                || !neural_ck_put(*c, d_dev[0].p, devBytes)) return fail("memcpy D2H checkpoint");
            res.checkpoint = std::move(c);
        }

        if (lyap && !stopped && !res.cancelled && lyNext == done) {
            if (!lyap_event(done, d_buf[stage ? 0 : (int)(done & 1)].as<numb>())) return fail(err);
            cudaDeviceSynchronize();
        }

        res.status.assign((size_t)Nt, 0);
        if (cudaMemcpy(res.status.data(), d_status.p, (size_t)Nt * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("memcpy D2H status");
        for (int i = 0; i < N; ++i) res.status[(size_t)i] = (res.status[(size_t)i] > 0 || res.status[(size_t)(Nt - N + i)] > 0) ? 1 : 0;
        res.status.resize((size_t)N);
        for (int v : res.status) res.n_diverged += v;

        if (lyap) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            res.lyap_dt = (double)lyM * req.h;
            res.lyap_blocks = (int)(lyLog.size() / (size_t)nInst);
            res.lyapunov.assign((size_t)nInst, nan);
            for (int q = 0; q < nInst && res.lyap_blocks > 0; ++q) {
                if (instStop[(size_t)q] >= 0) continue;
                double sum = 0.0;
                for (int b = 0; b < res.lyap_blocks; ++b) sum += lyLog[(size_t)b * nInst + q];
                res.lyapunov[(size_t)q] = sum / ((double)res.lyap_blocks * res.lyap_dt);
            }
            res.lyap_log = std::move(lyLog);
        }

        if (nOrder > 0) {
            cudaDeviceSynchronize();
            if (!flush_cross()) return fail("memcpy crossings");
            std::vector<size_t> at((size_t)N + 1, 0);
            for (int i : crossNode) ++at[(size_t)i + 1];
            for (int i = 0; i < N; ++i) at[(size_t)i + 1] += at[(size_t)i];
            std::vector<double> sorted(crossTime.size());
            std::vector<size_t> put(at.begin(), at.end() - 1);
            for (size_t e = 0; e < crossNode.size(); ++e) sorted[put[(size_t)crossNode[e]]++] = crossTime[e];
            neural_order_from_crossings(at, sorted, ob.pop_start, nPoints, (double)skipSteps * req.h,
                                        (double)req.pre_scaller * req.h, (std::min)(crossEnd, (double)done * req.h), orders);
        }

        // Записанное после шага разлёта зависит от чанка и данных не несёт.
        if (nInst > 1 && nPoints > 0) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            std::vector<long long> stopOfPop((size_t)nPops, -1), stopOfSel((size_t)nSel, -1);
            for (int q = 0; q < nPops; ++q) stopOfPop[(size_t)q] = instStop[(size_t)instance_of(ob.pop_start[(size_t)q])];
            for (int q = 0; q < nSel; ++q) stopOfSel[(size_t)q] = instStop[(size_t)instance_of(ob.nodes[(size_t)q])];
            for (int p = 0; p < nPoints; ++p) {
                const long long step = skipSteps + (long long)p * req.pre_scaller;
                const size_t at = (size_t)p;
                for (int q = 0; q < nSel; ++q)
                    if (stopOfSel[(size_t)q] >= 0 && step >= stopOfSel[(size_t)q])
                        for (int v = 0; v < nVars; ++v) traces[(at * nSel + q) * nVars + v] = nan;
                for (int q = 0; q < nPops; ++q) {
                    if (stopOfPop[(size_t)q] < 0 || step < stopOfPop[(size_t)q]) continue;
                    for (int v = 0; v < nVars && nMean > 0; ++v) {
                        means[(at * nMean + q) * nVars + v] = nan;
                        if (wantSpread) spreads[(at * nMean + q) * nVars + v] = nan;
                    }
                    if (nOrder > 0) orders[at * nOrder + q] = nan;
                }
            }
        }

        if (req.keep_final_state) {
            std::vector<numb> soa(nStatesT);
            if (cudaMemcpy(soa.data(), d_buf[stage ? 0 : (int)(done & 1)].p, stateBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H state");
            res.final_state.resize(nStates);
            for (int i = 0; i < N; ++i) for (int k = 0; k < nx; ++k)
                res.final_state[(size_t)i * nx + k] = (double)soa[(size_t)k * Nt + i];
        }
        res.observed = ob.nodes;
        res.observed_vars = ob.vars;
        res.traces = std::move(traces);
        res.means = std::move(means);
        res.spread = std::move(spreads);
        res.order = std::move(orders);
        res.instance_stop = std::move(instStop);
        res.n_nodes = N; res.n_vars = nx; res.n_points = nPoints;
        res.h = req.h; res.t0 = (double)skipSteps * req.h; res.dt = (double)req.pre_scaller * req.h;
        res.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
        res.ok = !res.cancelled;
        return res;
    }
