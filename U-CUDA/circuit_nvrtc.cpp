#include "circuit_nvrtc.h"

#define NOMINMAX          // иначе макросы min/max из windows.h ломают std::min/std::max
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <nvrtc.h>

namespace {

std::string exe_dir() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    std::string p(buf, n);
    const size_t pos = p.find_last_of("\\/");
    return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
}

// Тот же приём, что в parametric_engine: NVRTC спотыкается и о BOM, и о не-ASCII
// внутри комментариев, потому что разбирает файл как device-код.
std::string read_template(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "failed to open " + path; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF
                      && (unsigned char)s[1] == 0xBB
                      && (unsigned char)s[2] == 0xBF) s.erase(0, 3);
    for (char& c : s) if ((unsigned char)c >= 0x80) c = ' ';
    return s;
}

const char* cu_err(CUresult r) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
}

}  // namespace

CircuitGpuRunner::~CircuitGpuRunner() { release(); }

void CircuitGpuRunner::release() {
    if (module_) { cuModuleUnload((CUmodule)module_); module_ = nullptr; }
    func_ = nullptr;
    if (device_ >= 0) { cuDevicePrimaryCtxRelease(device_); device_ = -1; }
}

bool CircuitGpuRunner::build(const CircuitGraph& g, const CircuitSolverConfig& scfg,
                             const EmitConfig& ec, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    release();

    if (!build_mna_layout(g, scfg, lay_, err)) return false;
    time_scale_ = g.time_scale;
    n_vars_     = (int)g.var_node.size();

    std::vector<MatEntry> pat;
    mna_pattern(lay_, pat);
    if (!plan_elimination(pat, lay_.size, plan_, err)) return false;
    if (!emit_circuit_step(lay_, plan_, ec, emit_, err)) return false;
    comp_ = emit_.comp;
    // vsat живёт в comp, но у эмиттера его взять неоткуда: он знает раскладку,
    // а не модель. Дописываем последним элементом ровно там, где он его ждёт.
    if (!scfg.opamp.ideal && !comp_.empty()) comp_.back() = scfg.opamp.vsat;

    std::string terr;
    std::string src = read_template(exe_dir() + "/kernels/circuit.template.cu", terr);
    if (src.empty()) return fail("circuit nvrtc: " + terr);
    const std::string ph = "{{CIRCUIT_BODY}}";
    const size_t at = src.find(ph);
    if (at == std::string::npos) return fail("circuit nvrtc: placeholder {{CIRCUIT_BODY}} not found");
    src.replace(at, ph.size(), emit_.body);
    source_ = src;

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) return fail(std::string("circuit nvrtc: cuInit - ") + cu_err(r));
    CUdevice dev = 0;
    if ((r = cuDeviceGet(&dev, 0)) != CUDA_SUCCESS)
        return fail(std::string("circuit nvrtc: cuDeviceGet - ") + cu_err(r));
    CUcontext ctx = nullptr;
    // Первичный контекст, а не свой: NvrtcEngine держит собственный, и второй
    // созданный рядом ломал бы текущий контекст потока при каждом переключении.
    if ((r = cuDevicePrimaryCtxRetain(&ctx, dev)) != CUDA_SUCCESS)
        return fail(std::string("circuit nvrtc: primary context - ") + cu_err(r));
    device_ = dev;
    cuCtxSetCurrent(ctx);

    int cc_major = 0, cc_minor = 0;
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
    const char* opts[] = { arch, "--std=c++14" };

    nvrtcProgram prog = nullptr;
    if (nvrtcCreateProgram(&prog, source_.c_str(), "circuit.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
        return fail("circuit nvrtc: nvrtcCreateProgram failed");
    const nvrtcResult cr = nvrtcCompileProgram(prog, 2, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t n = 0; nvrtcGetProgramLogSize(prog, &n);
        std::string log(n ? n - 1 : 0, '\0');
        if (n > 1) nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        return fail("circuit nvrtc: compilation failed\n" + log);
    }
    size_t ptxn = 0; nvrtcGetPTXSize(prog, &ptxn);
    std::string ptx(ptxn, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    if ((r = cuModuleLoadData(&mod, ptx.c_str())) != CUDA_SUCCESS)
        return fail(std::string("circuit nvrtc: module load - ") + cu_err(r));
    module_ = mod;
    CUfunction fn = nullptr;
    if ((r = cuModuleGetFunction(&fn, mod, "circuitKernel")) != CUDA_SUCCESS)
        return fail(std::string("circuit nvrtc: kernel not found - ") + cu_err(r));
    func_ = fn;

    cuFuncGetAttribute(&regs_, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
    cuFuncGetAttribute(&lmem_, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
    return true;
}

bool CircuitGpuRunner::run(const std::vector<double>& x0_flat, int n_threads,
                           double h_ode, double t_end_ode, double sample_every_ode,
                           std::vector<double>& out, std::vector<int>& fail_out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!func_) return fail("circuit nvrtc: kernel is not built");
    if (n_threads <= 0) return fail("circuit nvrtc: thread count must be positive");
    if ((int)x0_flat.size() < n_threads * n_vars_)
        return fail("circuit nvrtc: initial condition array is too short");
    if (h_ode <= 0.0 || t_end_ode <= 0.0 || sample_every_ode <= 0.0)
        return fail("circuit nvrtc: step, horizon and sampling interval must be positive");

    const double    h_sec  = h_ode / time_scale_;
    const int       nsteps = (int)std::llround(t_end_ode / h_ode);
    const int       every  = std::max(1, (int)std::llround(sample_every_ode / h_ode));
    const long long nsamp  = nsteps / every;
    if (nsamp <= 0) return fail("circuit nvrtc: horizon is shorter than one sample");

    out.assign((size_t)nsamp * n_vars_ * n_threads, 0.0);
    fail_out.assign((size_t)n_threads, 0);

    CUdeviceptr d_comp = 0, d_x0 = 0, d_out = 0, d_fail = 0;
    CUresult r = CUDA_SUCCESS;
    auto cleanup = [&]() {
        if (d_comp) cuMemFree(d_comp);
        if (d_x0)   cuMemFree(d_x0);
        if (d_out)  cuMemFree(d_out);
        if (d_fail) cuMemFree(d_fail);
    };
    auto guard = [&](CUresult rr, const char* what) {
        if (rr == CUDA_SUCCESS) return true;
        cleanup();
        fail(std::string("circuit nvrtc: ") + what + " - " + cu_err(rr));
        return false;
    };

    if (!guard(cuMemAlloc(&d_comp, comp_.size() * sizeof(double)), "alloc comp")) return false;
    if (!guard(cuMemAlloc(&d_x0, (size_t)n_threads * n_vars_ * sizeof(double)), "alloc x0")) return false;
    if (!guard(cuMemAlloc(&d_out, out.size() * sizeof(double)), "alloc out")) return false;
    if (!guard(cuMemAlloc(&d_fail, (size_t)n_threads * sizeof(int)), "alloc fail")) return false;
    if (!guard(cuMemcpyHtoD(d_comp, comp_.data(), comp_.size() * sizeof(double)), "copy comp")) return false;
    if (!guard(cuMemcpyHtoD(d_x0, x0_flat.data(), (size_t)n_threads * n_vars_ * sizeof(double)), "copy x0")) return false;

    const int block = 32;   // замеренный оптимум проекта; на мелкой сетке широкий блок проигрывает
    const int grid  = (n_threads + block - 1) / block;
    void* args[] = { &d_comp, &d_x0, &d_out, &n_threads, (void*)&h_sec,
                     (void*)&nsteps, (void*)&every, (void*)&nsamp, &d_fail };
    r = cuLaunchKernel((CUfunction)func_, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr);
    if (!guard(r, "launch")) return false;
    if (!guard(cuCtxSynchronize(), "synchronize")) return false;
    if (!guard(cuMemcpyDtoH(out.data(), d_out, out.size() * sizeof(double)), "copy out")) return false;
    if (!guard(cuMemcpyDtoH(fail_out.data(), d_fail, (size_t)n_threads * sizeof(int)), "copy fail")) return false;

    cleanup();
    return true;
}

// --- самопроверка -------------------------------------------------------------

std::string circuit_nvrtc_selftest(bool* ok) {
    std::ostringstream o;
    bool good = true;
    auto check = [&](bool cond, const std::string& what) {
        o << (cond ? "  ok   " : "  FAIL ") << what << "\n";
        if (!cond) good = false;
    };

    System sys;
    sys.vars   = { "u", "v", "w" };
    sys.params = { "sigma", "r", "b" };
    sys.rhs    = { "sigma*(v - u)", "r*u - v - 20*u*w", "5*u*v - b*w" };
    const std::vector<double> vals = { 0.0, 16.0, 45.6, 4.0 };

    std::string err;
    std::vector<PolyRhs> poly;
    if (!extract_quadratic(sys, vals, PolyExtractConfig(), poly, &err)) {
        if (ok) *ok = false; return "  FAIL " + err + "\n";
    }
    CircuitGraph g;
    if (!synthesize_circuit(sys, poly, SynthesisConfig(), g, &err)) {
        if (ok) *ok = false; return "  FAIL " + err + "\n";
    }

    CircuitSolverConfig scfg;
    scfg.newton_fixed = 3;          // GPU не ветвится, CPU обязан делать столько же
    EmitConfig ec;
    ec.newton_iters    = scfg.newton_fixed;
    ec.newton_max_step = scfg.newton_max_step;
    ec.pivot_min       = scfg.pivot_min;

    CircuitGpuRunner gpu;
    if (!gpu.build(g, scfg, ec, &err)) { if (ok) *ok = false; return "  FAIL " + err + "\n"; }

    char buf[224];
    std::snprintf(buf, sizeof(buf), "kernel built: %d registers, %d bytes of local memory, comp[%d]",
                  gpu.registers(), gpu.local_bytes(), (int)gpu.components().size());
    o << "  info " << buf << "\n";
    check(gpu.local_bytes() == 0, "no register spilling");
    check(gpu.registers() < 256, "register count is under the hardware ceiling");

    const double h = 1.0e-4, t_end = 1.0, samp = 0.01;
    const std::vector<double> x0 = { 0.1, 0.1, 0.1 };
    std::vector<double> got;
    std::vector<int>    failed;
    if (!gpu.run(x0, 1, h, t_end, samp, got, failed, &err)) {
        check(false, "kernel run: " + err);
        if (ok) *ok = good;
        return o.str();
    }
    check(failed[0] == 0, "static pivot order held (no small-pivot flag)");

    CircuitRunResult cpu = circuit_simulate(g, scfg, x0, h, t_end, samp);
    check(cpu.ok, std::string("CPU reference run") + (cpu.ok ? "" : ": " + cpu.error));
    if (cpu.ok) {
        const size_t n = std::min(cpu.t.size(), got.size() / 3);
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (int v = 0; v < 3; ++v)
                worst = std::max(worst, std::fabs(got[i * 3 + (size_t)v] - cpu.x[(size_t)v][i]));
        std::snprintf(buf, sizeof(buf),
            "GPU vs CPU pointwise over t <= %.0f: max |dV| = %.3e", t_end, worst);
        // Порог арифметический, а не физический: у GPU включена FMA-контракция,
        // а порядок исключения статический против частичного пивотинга на CPU.
        check(worst < 1.0e-9, buf);
    }

    // Номиналы — аргумент ядра, а не литералы: смена значения обязана менять
    // результат БЕЗ перекомпиляции. Иначе фаза 6 невозможна.
    std::vector<double> got2;
    gpu.components()[0] *= 1.05;
    if (gpu.run(x0, 1, h, t_end, samp, got2, failed, &err)) {
        double diff = 0.0;
        for (size_t i = 0; i < std::min(got.size(), got2.size()); ++i)
            diff = std::max(diff, std::fabs(got[i] - got2[i]));
        std::snprintf(buf, sizeof(buf),
            "component value is a runtime argument (5%% change moved the result by %.3g V)", diff);
        check(diff > 1.0e-6, buf);
    } else {
        check(false, "re-run with changed component: " + err);
    }

    if (ok) *ok = good;
    o << (good ? "NVRTC SELFTEST PASSED\n" : "NVRTC SELFTEST FAILED\n");
    return o.str();
}
