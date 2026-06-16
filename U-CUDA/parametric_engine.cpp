#include "parametric_engine.h"
//
// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).
//

#include <cuda.h>
#include <nvrtc.h>
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>

namespace {

constexpr int kBlockSize         = 32;     // как в NonLinAnal::bifurcation1D
constexpr int kMaxAmountOfX      = 32;
constexpr int kMaxAmountOfValues = 64;

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
    if (!f) { err = "не удалось открыть " + path; return {}; }
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

}  // namespace

// ---------------------------------------------------------------------------

struct ParametricEngine::Impl {
    bool       inited   = false;
    CUcontext  context  = nullptr;
    CUdevice   device   = 0;
    int        cc_major = 0;
    int        cc_minor = 0;

    // Закэшированные тексты NonLinAnal headers (читаются один раз)
    std::string src_cudaLibrary_cu;
    std::string src_cudaLibrary_cuh;
    std::string src_cudaMacros_cuh;
    std::string src_configCUDA_h;
    std::string src_template;
    bool srcs_loaded = false;

    struct CachedModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_traj = nullptr;  // calculateDiscreteModelCUDA
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
    };
    CachedModule cached;

    ~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            release_module();
            cuCtxDestroy(context);
        }
    }

    void release_module() {
        if (cached.module) {
            cuModuleUnload(cached.module);
            cached.module = nullptr;
            cached.kernel_traj = nullptr;
            cached.kernel_peak = nullptr;
            cached.key.clear();
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

    bool load_sources(std::string& err) {
        if (srcs_loaded) return true;
        std::string root = exe_dir() + "\\kernels\\";
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h      = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        srcs_loaded = true;
        return true;
    }

    bool compile_if_needed(const std::string& krs_body, int amountOfX, std::string& err) {
        std::string key = hash_key(krs_body, amountOfX);
        if (cached.module && cached.key == key) return true;  // cache hit
        release_module();

        if (!load_sources(err)) return false;

        // Подстановка плейсхолдеров в шаблон
        std::string src = src_template;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);

        // Виртуальные заголовки для NVRTC
        const char* header_sources[] = {
            src_cudaLibrary_cu.c_str(),
            src_cudaLibrary_cuh.c_str(),
            src_cudaMacros_cuh.c_str(),
            src_configCUDA_h.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
        };
        constexpr int n_headers = 4;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "bifurcation1d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram: ") + nvrtcGetErrorString(nr); return false; }

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h и т.п.).
        // Берём CUDA_PATH из окружения (его проставляет CUDA Toolkit installer).
        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "переменная окружения CUDA_PATH не задана — NVRTC не найдёт math_constants.h "
                  "(установи CUDA Toolkit или задай CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = {
            arch,
            std_opt.c_str(),
            "-default-device",
            cuda_include_opt.c_str()
        };

        nr = nvrtcCompileProgram(prog, 4, opts);
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

        CUresult r = cuModuleLoadDataEx(&cached.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx: " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached.kernel_traj, cached.module, "calculateDiscreteModelCUDA");
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(calculateDiscreteModelCUDA): " + cu_err(r);
            release_module(); return false;
        }
        r = cuModuleGetFunction(&cached.kernel_peak, cached.module, "peakFinderCUDA");
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(peakFinderCUDA): " + cu_err(r);
            release_module(); return false;
        }

        cached.key = key;
        return true;
    }

    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        // ---- валидация ----
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        // base_values уже идёт со сдвигом +1 (a[0] зарезервирован):
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                                                                    return fail("param_index вне диапазона");
        if (req.writable_var < 0 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var вне диапазона");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

        // ---- init + контекст ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);

        // ---- компиляция или cache hit ----
        if (!compile_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);

        // ---- производные величины (как в hostLibrary::bifurcation1D) ----
        int    nPts                 = req.n_pts;
        int    nPtsLimiter          = nPts;                  // без chunking
        int    amountOfPointsInBlock= (int)(req.t_max / req.h / req.pre_scaller);
        int    amountOfPointsForSkip= (int)(req.transient_time / req.h);
        int    amountOfInitialConditions = req.amountOfX;
        int    amountOfValues       = (int)req.base_values.size();
        int    preScaller           = req.pre_scaller;
        int    writableVar          = req.writable_var;
        int    dimension            = 1;     // 1D-бифуркация
        bool   par_or_var           = true;  // меняем параметр (не НУ)

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        // ---- device alloc ----
        CUdeviceptr d_data = 0, d_ranges = 0, d_indices = 0, d_ic = 0, d_values = 0;
        CUdeviceptr d_outPeaks = 0, d_timeOfPeaks = 0, d_amountOfPeaks = 0;
        size_t bytes_data    = (size_t)nPtsLimiter * amountOfPointsInBlock * sizeof(double);
        size_t bytes_ranges  = 2 * sizeof(double);
        size_t bytes_indices = sizeof(int);
        size_t bytes_ic      = (size_t)amountOfInitialConditions * sizeof(double);
        size_t bytes_values  = (size_t)amountOfValues * sizeof(double);
        size_t bytes_peaks   = bytes_data;                    // up to same size as data
        size_t bytes_amount  = (size_t)nPtsLimiter * sizeof(int);

        auto cleanup = [&]() {
            if (d_data)         cuMemFree(d_data);
            if (d_ranges)       cuMemFree(d_ranges);
            if (d_indices)      cuMemFree(d_indices);
            if (d_ic)           cuMemFree(d_ic);
            if (d_values)       cuMemFree(d_values);
            if (d_outPeaks)     cuMemFree(d_outPeaks);
            if (d_timeOfPeaks)  cuMemFree(d_timeOfPeaks);
            if (d_amountOfPeaks)cuMemFree(d_amountOfPeaks);
        };

        CUresult r;
        if ((r = cuMemAlloc(&d_data,          bytes_data))    != CUDA_SUCCESS) { res.error = "cuMemAlloc data ("    + std::to_string(bytes_data) + "B): " + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_ranges,        bytes_ranges))  != CUDA_SUCCESS) { res.error = "cuMemAlloc ranges: "  + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_indices,       bytes_indices)) != CUDA_SUCCESS) { res.error = "cuMemAlloc indices: " + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_ic,            bytes_ic))      != CUDA_SUCCESS) { res.error = "cuMemAlloc ic: "      + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_values,        bytes_values))  != CUDA_SUCCESS) { res.error = "cuMemAlloc values: "  + cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_outPeaks,      bytes_peaks))   != CUDA_SUCCESS) { res.error = "cuMemAlloc outPeaks: "+ cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_timeOfPeaks,   bytes_peaks))   != CUDA_SUCCESS) { res.error = "cuMemAlloc timeOfPeaks: "+ cu_err(r); cleanup(); return res; }
        if ((r = cuMemAlloc(&d_amountOfPeaks, bytes_amount))  != CUDA_SUCCESS) { res.error = "cuMemAlloc amountOfPeaks: "+ cu_err(r); cleanup(); return res; }

        // ---- H2D ----
        double h_ranges[2] = { req.param_lo, req.param_hi };
        int    h_indices[1] = { req.param_index };   // уже 1-индексированный
        cuMemcpyHtoD(d_ranges,  h_ranges,                    bytes_ranges);
        cuMemcpyHtoD(d_indices, h_indices,                   bytes_indices);
        cuMemcpyHtoD(d_ic,      req.initial_conditions.data(), bytes_ic);
        cuMemcpyHtoD(d_values,  req.base_values.data(),      bytes_values);

        // ---- launch calculateDiscreteModelCUDA ----
        // Сигнатура (cudaLibrary.cu:906):
        //   const int nPts, const int nPtsLimiter, const size_t sizeOfBlock,
        //   const size_t amountOfCalculatedPoints, const size_t amountOfPointsForSkip,
        //   const int dimension, numb* ranges, const numb h, int* indicesOfMutVars,
        //   numb* initialConditions, const int amountOfInitialConditions,
        //   const numb* values, const int amountOfValues,
        //   const size_t amountOfIterations, const int preScaller,
        //   const int writableVar, const numb maxValue,
        //   numb* data, int* maxValueCheckerArray, const bool Par_or_Var
        size_t sizeOfBlock          = (size_t)amountOfPointsInBlock;
        size_t amountOfCalculated   = 0;
        size_t amountOfPointsForSkip_s = (size_t)amountOfPointsForSkip;
        size_t amountOfIterations   = (size_t)amountOfPointsInBlock;
        double h_arg                = req.h;
        double maxValue_arg         = req.max_value;

        void* args_traj[] = {
            &nPts,
            &nPtsLimiter,
            &sizeOfBlock,
            &amountOfCalculated,
            &amountOfPointsForSkip_s,
            &dimension,
            &d_ranges,
            &h_arg,
            &d_indices,
            &d_ic,
            &amountOfInitialConditions,
            &d_values,
            &amountOfValues,
            &amountOfIterations,
            &preScaller,
            &writableVar,
            &maxValue_arg,
            &d_data,
            &d_amountOfPeaks,
            &par_or_var
        };

        unsigned int grid = (unsigned int)((nPtsLimiter + kBlockSize - 1) / kBlockSize);
        unsigned int shared = (unsigned int)((amountOfInitialConditions + amountOfValues) * sizeof(double) * kBlockSize);

        r = cuLaunchKernel(cached.kernel_traj,
                           grid, 1, 1, kBlockSize, 1, 1,
                           shared, nullptr, args_traj, nullptr);
        if (r != CUDA_SUCCESS) { res.error = "cuLaunchKernel(traj): " + cu_err(r); cleanup(); return res; }

        r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) { res.error = "cuCtxSynchronize after traj: " + cu_err(r); cleanup(); return res; }

        // ---- launch peakFinderCUDA ----
        // Сигнатура (cudaLibrary.cu:1553):
        //   numb* data, const size_t sizeOfBlock, const int amountOfBlocks,
        //   int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, const numb timeStep
        double timeStep_arg = req.h * (double)preScaller;
        void* args_peak[] = {
            &d_data,
            &sizeOfBlock,
            &nPtsLimiter,
            &d_amountOfPeaks,
            &d_outPeaks,
            &d_timeOfPeaks,
            &timeStep_arg
        };
        r = cuLaunchKernel(cached.kernel_peak,
                           grid, 1, 1, kBlockSize, 1, 1,
                           0, nullptr, args_peak, nullptr);
        if (r != CUDA_SUCCESS) { res.error = "cuLaunchKernel(peak): " + cu_err(r); cleanup(); return res; }

        r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) { res.error = "cuCtxSynchronize after peak: " + cu_err(r); cleanup(); return res; }

        // ---- D2H ----
        std::vector<double> h_outPeaks((size_t)nPtsLimiter * amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);
        cuMemcpyDtoH(h_outPeaks.data(),      d_outPeaks,      bytes_peaks);
        cuMemcpyDtoH(h_amountOfPeaks.data(), d_amountOfPeaks, bytes_amount);

        cleanup();

        // ---- упаковка ----
        res.ok           = true;
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.flags        = std::move(h_amountOfPeaks);
        res.bifurcation_points.resize(nPts);
        for (int i = 0; i < nPts; ++i) {
            int n = res.flags[i];
            if (n <= 0) { res.bifurcation_points[i].clear(); continue; }
            if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
            const double* row = h_outPeaks.data() + (size_t)i * amountOfPointsInBlock;
            res.bifurcation_points[i].assign(row, row + n);
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
