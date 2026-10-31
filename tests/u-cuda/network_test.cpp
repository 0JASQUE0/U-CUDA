#include "test_util.h"
#include "network_fixtures.h"
#include "network_session.h"
#include <cuda.h>
#include <nvrtc.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
void cuda_ok(CUresult r) { ut::require(r == CUDA_SUCCESS, "CUDA driver error " + std::to_string(r)); }

std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    ut::require(bool(f), "cannot read " + p.u8string());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void write(const fs::path& p, const std::string& bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(bytes.data(), bytes.size());
    ut::require(bool(f), "cannot write " + p.u8string());
}
template<class T> void append(std::string& b, const T& x) {
    b.append(reinterpret_cast<const char*>(&x), sizeof x);
}
template<class T> void append_vec(std::string& b, const std::vector<T>& v) {
    append(b, uint64_t(v.size()));
    if (!v.empty()) b.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
void append_text(std::string& b, const std::string& s) { append(b, uint64_t(s.size())); b += s; }

std::string input_bytes(const NetworkRequest& q) {
    std::string b = "UCUDA_NETWORK_INPUT_V1";
    append_text(b, q.krs_body); append_text(b, q.coupling_body);
    append(b, q.amountOfX); append(b, q.n_nodes); append(b, q.amountOfValues);
    append_vec(b, q.values); append_vec(b, q.initial_conditions);
    append_vec(b, q.edge_start); append_vec(b, q.edge_src);
    append_vec(b, q.edge_w); append_vec(b, q.edge_law);
    append(b, q.h); append(b, q.t_max); append(b, q.transient);
    append(b, q.pre_scaller); append(b, q.max_points); append(b, q.max_value);
    return b;
}
std::string result_bytes(const NetworkResult& r) {
    std::string b = "UCUDA_NETWORK_RESULT_V1";
    append(b, r.n_nodes); append(b, r.n_vars); append(b, r.n_points);
    append(b, r.h); append(b, r.t0); append(b, r.dt); append(b, r.pre_scaller);
    append(b, r.n_diverged); append_vec(b, r.status); append_vec(b, r.data);
    append_vec(b, r.vmin); append_vec(b, r.vmax);
    return b;
}
void equal_bytes(const std::string& a, const std::string& b, const std::string& label) {
    ut::require(a.size() == b.size(), label + ": byte count differs");
    const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
    ut::require(mismatch.first == a.end(), label + ": mismatch at byte "
                + std::to_string(mismatch.first - a.begin()));
}
NetworkResult run(ParametricEngine& engine, const NetworkRequest& q) {
    auto r = engine.run_network(q);
    ut::require(r.ok && !r.cancelled, "network run: " + r.error);
    ut::require(r.n_diverged == 0 && !r.data.empty(), "empty/diverged network");
    for (double v : r.data) ut::require(std::isfinite(v), "non-finite trajectory");
    return r;
}

NetworkRequest request(const NetworkFixture& f, Scheme scheme, NetTopology topology, int n, bool mixed) {
    NetworkRequest q;
    q.krs_body = codegen_scheme(f.sys, scheme);
    q.amountOfX = int(f.sys.vars.size()); q.n_nodes = n;
    q.amountOfValues = int(f.parameters.size()) + 1 + kSchemeTailSlots;
    for (int i = 0; i < n; ++i) {
        std::vector<double> a{0.5};
        a.insert(a.end(), f.parameters.begin(), f.parameters.end());
        // Heterogeneous last parameter; exact binary IC offsets, no std random distributions.
        a[f.parameters.size()] += (i % 5) * 0.002;
        const size_t base = a.size(); a.resize(base + kSchemeTailSlots);
        scheme_fill_tail(a.data() + base, "4", NewtonSettings{});
        q.values.insert(q.values.end(), a.begin(), a.end());
        for (size_t k = 0; k < f.initial.size(); ++k)
            q.initial_conditions.push_back(f.initial[k] + ((i * 17 + int(k) * 7) % 31 - 15) / 128.0);
    }
    NetworkConfig c;
    c.topology = topology; c.gen_n_text = std::to_string(n);
    c.gen_directed = mixed; c.gen_k_text = "2";
    c.gen_seed_text = "271828"; c.gen_p_text = "0.25";
    c.normalize_coupling = !mixed;
    ut::require(net_generate_topology(c, f.sys.vars).empty(), "topology generator");
    c.laws = {net_default_law(f.sys.vars)};
    if (mixed) {
        NetCouplingLaw law; law.name = "nonlinear"; law.expr.resize(f.sys.vars.size());
        const auto& x = f.sys.vars.front();
        law.expr[0] = "K*(" + x + "_j-" + x + "_i)/(1+" + x + "_j*" + x + "_j)";
        c.laws.push_back(law);
        for (size_t e = 0; e < c.edges.size(); ++e) {
            c.edges[e].law = int(e % 2);
            c.edges[e].weight_text = (e % 3 == 0) ? "0.5" : "1.25";
        }
    }
    q.coupling_body = net_coupling_body(f.sys, c.laws);
    net_build_csr(c, 0.2, q.edge_start, q.edge_src, q.edge_w, q.edge_law);
    q.h = 0.01; q.t_max = 2.57; q.transient = 0.13;
    q.pre_scaller = 3; q.max_points = 20000;
    return q;
}

std::string environment(CUdevice dev) {
    char name[256]{}; cuda_ok(cuDeviceGetName(name, sizeof name, dev));
    CUuuid uuid{}; cuda_ok(cuDeviceGetUuid(&uuid, dev));
    int driver = 0, major = 0, minor = 0;
    cuda_ok(cuDriverGetVersion(&driver));
    ut::require(nvrtcVersion(&major, &minor) == NVRTC_SUCCESS, "nvrtcVersion");
    std::ostringstream s;
    s << "network baseline v1\nGPU=" << name << "\nUUID=";
    for (unsigned char c : uuid.bytes) s << std::hex << std::setw(2) << std::setfill('0') << int(c);
    s << std::dec << "\ndriver=" << driver << "\nnvrtc=" << major << '.' << minor
      << "\nMSVC=" << _MSC_FULL_VER << "\nnumb_bytes=" << sizeof(numb)
      << "\nRDC=1; default FMA; Release /O2 /MD\n";
    // Capture exact runtime sources. A different kernel deployment invalidates bitwise comparison.
    const fs::path kernels = fs::u8path(exe_dir()) / "kernels";
    for (const char* file : {"network.template.cu", "cudaLibrary.cu", "cudaLibrary.cuh", "cudaMacros.cuh", "configCUDA.h"}) {
        s << "\nSOURCE " << file << '\n' << read(kernels / file);
    }
    return s.str();
}
struct Stats { double minimum, median, maximum; };
Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return {v.front(), v[v.size()/2], v.back()};
}

void benchmark(ParametricEngine& engine, const fs::path& out) {
#ifdef UCUDA_NETWORK_DIAGNOSTICS
    std::ofstream csv(out);
    ut::require(bool(csv), "cannot open benchmark CSV");
    csv << "model,nodes,arcs,steps,points,launches,series,repeat,gpu_ms,full_ms\n";
    for (const auto& f : network_fixtures()) for (int n : {16, 64, 256}) {
        auto q = request(f, Scheme::RK4, NetTopology::Ring, n, false);
        q.transient = 0; q.t_max = 10.24; q.pre_scaller = 4;
        q.measure_gpu_time = true;
        std::vector<double> gpu, full;
        // Independent series, each warmed for >=300 ms, not isolated cold launches.
        for (int series = 0; series < 3; ++series) {
            auto warm = Clock::now();
            do { run(engine, q); } while (Clock::now() - warm < std::chrono::milliseconds(300));
            for (int repeat = 0; repeat < 9; ++repeat) {
                auto start = Clock::now(); const auto r = engine.run_network(q);
                double ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
                ut::require(r.ok && !r.cancelled && r.n_diverged == 0, "benchmark run: " + r.error);
                for (double value : r.data) ut::require(std::isfinite(value), "benchmark non-finite");
                gpu.push_back(r.gpu_ms); full.push_back(ms);
                csv << f.name << ',' << n << ',' << q.edge_src.size() << ",1024," << r.n_points
                    << ',' << r.launch_count << ',' << series << ',' << repeat << ','
                    << std::setprecision(12) << r.gpu_ms << ',' << ms << '\n';
            }
        }
        auto g = stats(gpu), w = stats(full);
        std::printf("BENCH %s N=%d GPU min/median/max %.4f/%.4f/%.4f ms; full %.4f/%.4f/%.4f ms\n",
            f.name, n, g.minimum, g.median, g.maximum, w.minimum, w.median, w.maximum);
    }
    ut::require(bool(csv), "benchmark write failed");
#else
    ut::require(false, "benchmark requires rebuilt diagnostic API");
#endif
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    return ut::run_main("network", [&]() -> int {
        std::string mode = argc > 1 ? argv[1] : "--check";
        ut::require(mode == "--check" || mode == "--record" || mode == "--compare" || mode == "--benchmark", "unknown mode");
        ut::require((mode == "--check" && argc == 1) || (mode != "--check" && argc == 3),
                    "usage: network.exe [--record DIR | --compare DIR | --benchmark CSV]");
        if (cuInit(0) != CUDA_SUCCESS) return ut::kSkip;
        CUdevice dev; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return ut::kSkip;
        set_nvrtc_rdc(true);
        const auto env = environment(dev);
        ParametricEngine engine;
        if (mode == "--benchmark") { benchmark(engine, fs::u8path(argv[2])); return 0; }
        fs::path dir;
        if (argc == 3) dir = fs::u8path(argv[2]);
        if (mode == "--record") {
            ut::require(!fs::exists(dir), "baseline directory exists; choose a new path");
            fs::create_directories(dir); write(dir / "environment.txt", env);
            write(dir / "build.txt", std::string(__DATE__) + " " + __TIME__ + "\n");
        }
        if (mode == "--compare") equal_bytes(read(dir / "environment.txt"), env, "baseline environment/sources");
        int cases = 0;
        for (const auto& f : network_fixtures())
            for (auto scheme : {Scheme::Euler, Scheme::RK4, Scheme::CD})
                for (auto topology : {NetTopology::Ring, NetTopology::Star, NetTopology::Chain, NetTopology::SmallWorld}) {
                    bool mixed = topology == NetTopology::Chain || topology == NetTopology::SmallWorld;
                    auto q = request(f, scheme, topology, 16, mixed);
                    // Includes zero transient and automatic decimation, in addition to non-aligned boundaries.
                    if (topology == NetTopology::Star) { q.transient = 0; q.max_points = 37; }
                    std::string id = std::string(f.name) + "_" + std::to_string(int(scheme)) + "_" + std::to_string(int(topology));
                    const auto ref = run(engine, q);
                    const auto bytes = result_bytes(ref);
                    equal_bytes(bytes, result_bytes(run(engine, q)), id + " repeat");
                    if (mode == "--record") {
                        write(dir / (id + ".input.bin"), input_bytes(q));
                        write(dir / (id + ".result.bin"), bytes);
                    }
                    if (mode == "--compare") {
                        equal_bytes(read(dir / (id + ".input.bin")), input_bytes(q), id + " inputs");
                        equal_bytes(read(dir / (id + ".result.bin")), bytes, id + " baseline");
                    }
#ifdef UCUDA_NETWORK_DIAGNOSTICS
                    for (long long chunk : {1LL, 7LL, 13LL, 64LL, 1000000LL}) {
                        auto split = q; split.steps_per_launch = chunk;
                        const auto r = run(engine, split);
                        equal_bytes(bytes, result_bytes(r), id + " chunk=" + std::to_string(chunk));
                        if (chunk == 1) ut::require(r.launch_count == 270 - (topology == NetTopology::Star ? 13 : 0),
                                                    "one launch per step, including transient");
                    }
                    auto timed = q; timed.measure_gpu_time = true;
                    const auto r = run(engine, timed);
                    equal_bytes(bytes, result_bytes(r), id + " timing");
                    ut::require(r.gpu_ms > 0 && r.launch_count > 0, "missing timing");
#endif
                    ++cases; std::printf("  %s: OK (%s)\n", id.c_str(), mode.c_str());
                }
        // Independent scalar CPU RK4 reference on an isolated HR neuron.
        // A longer horizon catches an omitted -y in the recovery equation.
        auto hr = request(network_fixtures()[1], Scheme::RK4, NetTopology::Ring, 16, false);
        hr.n_nodes = 1; hr.values.resize(hr.amountOfValues);
        hr.initial_conditions = {-1.3, -7.0, 3.0};
        hr.edge_start = {0, 0}; hr.edge_src.clear(); hr.edge_w.clear(); hr.edge_law.clear();
        hr.transient = 0; hr.t_max = 10.24; hr.pre_scaller = 1024;
        const auto single = run(engine, hr);
        ut::require(single.data.size() == 6, "HR reference output shape");
        ut::approx(single.data[3], -1.1423619163157845, 1e-9, "HR reference x");
        ut::approx(single.data[4], -5.723316649310541, 1e-9, "HR reference y");
        ut::approx(single.data[5], 2.9135830388666357, 1e-9, "HR reference z");
#ifdef UCUDA_NETWORK_DIAGNOSTICS
        auto invalid = request(network_fixtures()[0], Scheme::RK4, NetTopology::Ring, 16, false);
        invalid.steps_per_launch = -1;
        ut::require(!engine.run_network(invalid).ok, "negative chunk must fail");
#endif
        std::printf("%d cases, mode=%s\n", cases, mode.c_str());
        return 0;
    });
}
