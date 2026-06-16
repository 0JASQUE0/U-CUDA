#include "parametric_engine.h"
//
// Реализация ParametricEngine: NVRTC-компиляция шаблона + хост-оркестрация
// для 1D-бифуркации. Кэш PTX по хэшу (krs_body, amountOfX) — если КРС не
// менялся, при следующем запуске переиспользуется уже загруженный модуль.
//

#include <cuda.h>
#include <nvrtc.h>
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>

namespace {

constexpr int kBlockSize         = 256;
constexpr int kMaxAmountOfX      = 32;
constexpr int kMaxAmountOfValues = 64;   // должно совпадать с LOCAL_VALUES_MAX в шаблоне

std::string exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0) return ".";
    std::string p(buf, n);
    auto pos = p.find_last_of("\\/");
    return (pos == std::string::npos) ? std::string(".") : p.substr(0, pos);
}

std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "не удалось открыть файл"; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string hash_key(const std::string& krs_body, int amountOfX) {
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" + std::to_string(amountOfX);
}

std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}

std::string nvrtc_err(nvrtcResult r) {
    const char* s = nvrtcGetErrorString(r);
    return s ? std::string(s) : ("nvrtcResult " + std::to_string((int)r));
}

}  // namespace

// ---------------------------------------------------------------------------

struct ParametricEngine::Impl {
    bool       inited   = false;
    CUcontext  context  = nullptr;
    CUdevice   device   = 0;
    int        cc_major = 0;
    int        cc_minor = 0;

    // Кэш одного скомпилированного модуля для bifurcation1D.
    // Ключ — хэш krs_body + размерность. Если запрос с другим ключом — пересобираем.
    struct Bif1dCache {
        std::string key;
        CUmodule    module = nullptr;
        CUfunction  kernel = nullptr;
    } bif1d;

    ~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            release_bif1d();
            cuCtxDestroy(context);
        }
    }

    void release_bif1d() {
        if (bif1d.module) {
            cuModuleUnload(bif1d.module);
            bif1d.module = nullptr;
            bif1d.kernel = nullptr;
            bif1d.key.clear();
        }
    }

    bool ensure_init(std::string& err) {
        if (inited) return true;
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { err = "cuInit: " + cu_err(r); return false; }
        r = cuDeviceGet(&device, 0);
        if (r != CUDA_SUCCESS) { err = "cuDeviceGet: " + cu_err(r); return false; }
#if CUDA_VERSION >= 13000
        r = cuCtxCreate(&context, nullptr, 0, device);
#else
        r = cuCtxCreate(&context, 0, device);
#endif
        if (r != CUDA_SUCCESS) { err = "cuCtxCreate: " + cu_err(r); return false; }
        cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        inited = true;
        return true;
    }

    // Подгружает шаблон, подставляет {{...}}, компилирует через NVRTC,
    // грузит PTX в модуль, достаёт указатель на kernel. Кэш по ключу.
    bool compile_bif1d(const std::string& krs_body, int amountOfX, std::string& err) {
        std::string key = hash_key(krs_body, amountOfX);
        if (bif1d.module && bif1d.key == key) return true;  // cache hit
        release_bif1d();

        std::string tmpl_path = exe_dir() + "\\kernels\\bifurcation1d.template.cu";
        std::string load_err;
        std::string tmpl = read_text_file(tmpl_path, load_err);
        if (!load_err.empty()) {
            err = "load template (" + tmpl_path + "): " + load_err;
            return false;
        }

        std::string src = tmpl;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "bifurcation1d.cu", 0, nullptr, nullptr);
        if (nr != NVRTC_SUCCESS) { err = "nvrtcCreateProgram: " + nvrtc_err(nr); return false; }

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
        const char* opts[] = { arch };

        nr = nvrtcCompileProgram(prog, 1, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile failed:\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&bif1d.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx: " + cu_err(r); return false; }

        r = cuModuleGetFunction(&bif1d.kernel, bif1d.module, "bifurcation1d_kernel");
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(bifurcation1d_kernel): " + cu_err(r);
            release_bif1d();
            return false;
        }

        bif1d.key = key;
        return true;
    }

    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;

        // ---- валидация запроса ----
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)
            return fail("amountOfX must be in [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if (req.base_values.empty() ||
            (int)req.base_values.size() > kMaxAmountOfValues)
            return fail("base_values size must be in [1," + std::to_string(kMaxAmountOfValues) + "]");
        if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                                    return fail("param_index out of range");
        if (req.writable_var < 0 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts        <= 0)   return fail("n_pts must be > 0");
        if (req.h            <= 0.0) return fail("h must be > 0");
        if (req.t_max        <= 0.0) return fail("t_max must be > 0");
        if (req.transient_time < 0.0)return fail("transient_time must be >= 0");
        if (req.pre_scaller  <= 0)   return fail("pre_scaller must be > 0");

        // ---- CUDA init + контекст текущим ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // ---- компиляция (или cache hit) ----
        if (!compile_bif1d(req.krs_body, req.amountOfX, err)) return fail(err);

        // ---- производные величины ----
        int transient_steps   = (int)std::ceil(req.transient_time / req.h);
        int total_record_iter = (int)std::ceil(req.t_max          / req.h);
        int record_steps      = total_record_iter / req.pre_scaller;
        if (record_steps <= 0)
            return fail("computed record_steps == 0 (t_max/h/pre_scaller слишком малы)");

        // ---- выделение device-памяти ----
        CUdeviceptr d_ic = 0, d_vals = 0, d_out = 0, d_flags = 0;
        size_t ic_bytes    = (size_t)req.amountOfX     * sizeof(double);
        size_t vals_bytes  = req.base_values.size()    * sizeof(double);
        size_t out_bytes   = (size_t)req.n_pts * (size_t)record_steps * sizeof(double);
        size_t flags_bytes = (size_t)req.n_pts         * sizeof(int);

        auto cleanup = [&]() {
            if (d_ic)    cuMemFree(d_ic);
            if (d_vals)  cuMemFree(d_vals);
            if (d_out)   cuMemFree(d_out);
            if (d_flags) cuMemFree(d_flags);
        };

        CUresult r;
        if ((r = cuMemAlloc(&d_ic,    ic_bytes))    != CUDA_SUCCESS) { res.error = "cuMemAlloc d_ic: "    + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_vals,  vals_bytes))  != CUDA_SUCCESS) { res.error = "cuMemAlloc d_vals: "  + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_out,   out_bytes))   != CUDA_SUCCESS) {
            res.error = "cuMemAlloc d_out (" + std::to_string(out_bytes) + " B): " + cu_err(r);
            cleanup(); return res;
        }
        if ((r = cuMemAlloc(&d_flags, flags_bytes)) != CUDA_SUCCESS) { res.error = "cuMemAlloc d_flags: " + cu_err(r); cleanup(); return res; }

        // ---- H2D ----
        cuMemcpyHtoD(d_ic,   req.initial_conditions.data(), ic_bytes);
        cuMemcpyHtoD(d_vals, req.base_values.data(),        vals_bytes);

        // ---- упаковка аргументов ядра (порядок строго как в kernel-сигнатуре) ----
        int    amount_of_values_arg = (int)req.base_values.size();
        int    param_index_arg      = req.param_index;
        double param_lo_arg         = req.param_lo;
        double param_hi_arg         = req.param_hi;
        int    n_pts_arg            = req.n_pts;
        int    writable_var_arg     = req.writable_var;
        double h_arg                = req.h;
        int    transient_steps_arg  = transient_steps;
        int    record_steps_arg     = record_steps;
        int    pre_scaller_arg      = req.pre_scaller;
        double max_value_arg        = req.max_value;

        void* args[] = {
            &d_ic, &d_vals,
            &amount_of_values_arg,
            &param_index_arg,
            &param_lo_arg, &param_hi_arg,
            &n_pts_arg,
            &writable_var_arg,
            &h_arg,
            &transient_steps_arg,
            &record_steps_arg,
            &pre_scaller_arg,
            &max_value_arg,
            &d_out, &d_flags
        };

        // ---- launch ----
        unsigned int blocks = (unsigned int)((req.n_pts + kBlockSize - 1) / kBlockSize);
        r = cuLaunchKernel(bif1d.kernel,
                           blocks, 1, 1,
                           kBlockSize, 1, 1,
                           /*shared*/ 0, /*stream*/ nullptr,
                           args, nullptr);
        if (r != CUDA_SUCCESS) { res.error = "cuLaunchKernel: " + cu_err(r); cleanup(); return res; }

        r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) { res.error = "cuCtxSynchronize (kernel error): " + cu_err(r); cleanup(); return res; }

        // ---- D2H ----
        std::vector<double> h_out  ((size_t)req.n_pts * (size_t)record_steps);
        std::vector<int>    h_flags((size_t)req.n_pts);
        cuMemcpyDtoH(h_out.data(),   d_out,   out_bytes);
        cuMemcpyDtoH(h_flags.data(), d_flags, flags_bytes);

        cleanup();

        // ---- упаковка результата ----
        res.ok            = true;
        res.n_pts         = req.n_pts;
        res.record_steps  = record_steps;
        res.flags         = std::move(h_flags);
        res.bifurcation_points.resize(req.n_pts);
        for (int i = 0; i < req.n_pts; ++i) {
            const double* row = h_out.data() + (size_t)i * (size_t)record_steps;
            res.bifurcation_points[i].assign(row, row + record_steps);
        }
        return res;
    }
};

// ---------------------------------------------------------------------------

ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
}
