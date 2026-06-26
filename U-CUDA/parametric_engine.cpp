#include "parametric_engine.h"
//
// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

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
    std::string s = ss.str();
    // UTF-8 BOM: 0xEF 0xBB 0xBF в начале — NVRTC от него спотыкается.
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        s.erase(0, 3);
    }
    // NVRTC не любит не-ASCII символы даже внутри комментариев (компилит как
    // device code, не предполагает многобайтовых юникод-точек). Заменяем все
    // байты со старшим битом на пробел — это безвредно для комментариев и
    // не задевает строковые литералы из ASCII.
    for (char& c : s) {
        if ((unsigned char)c >= 0x80) c = ' ';
    }
    return s;
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

// Локальная host-копия __device__ __host__ getValueByIdx из cudaLibrary.cu:1266.
// Для 1D-бифуркации valueNumber всегда 0, поэтому просто линейная интерполяция.
// Нужна для расчёта значения параметра в CSV (host-side).
inline double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    if (nPts <= 1) return lo;
    return lo + (hi - lo) * (double)idx / (double)(nPts - 1);
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
    // curand_kernel.h перехвачен inline-stub'ом в каждом template'е (kernels/*.cu)
    // + `#define CURAND_KERNEL_H_` блокирует реальный header. Virtual header'а
    // здесь больше нет — иначе он повторно объявлял бы curandState_t.
    std::string src_configCUDA_h;
    std::string src_template;        // bifurcation1d.template.cu
    std::string src_template_cont;   // bifurcation1d_cont.template.cu
    std::string src_template_lle;    // lle1d.template.cu
    std::string src_template_lle_2d; // lle2d.template.cu
    std::string src_template_ls;     // ls1d.template.cu
    std::string src_template_ls_2d;  // ls2d.template.cu
    std::string src_template_bif2d;  // bifurcation2d.template.cu
    bool srcs_loaded = false;

    struct CachedModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_traj = nullptr;  // calculateDiscreteModelCUDA
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
    };
    CachedModule cached;          // bif1d kernels (traj + peak)

    // Отдельный модуль для LLE — другой шаблон, другой kernel.
    struct CachedLleModule {
        std::string key;
        CUmodule    module      = nullptr;
        CUfunction  kernel_lle  = nullptr;  // LLEKernelCUDA
    };
    CachedLleModule cached_lle;

    // LLE-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLleModule cached_lle_2d;

    // LS — отдельный модуль/кэш (третий слот PTX).
    struct CachedLsModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_ls  = nullptr;   // LSKernelCUDA
    };
    CachedLsModule cached_ls;

    // LS-2D — отдельный шаблон/par_or_var-ветка, тот же kernel.
    CachedLsModule cached_ls_2d;

    // Continuation bif1d — четвёртый слот. Кернел single-thread, плюс
    // peakFinderCUDA из того же модуля (он определён в cudaLibrary.cu).
    struct CachedContModule {
        std::string key;
        CUmodule    module     = nullptr;
        CUfunction  kernel_cont = nullptr;  // bifurcation1dContinuationKernel
        CUfunction  kernel_peak = nullptr;  // peakFinderCUDA
    };
    CachedContModule cached_cont;

    // Bif-2D — отдельный шаблон (bifurcation2d.template.cu), три kernel'а:
    // calculateDiscreteModelCUDA + peakFinderCUDA + dbscanCUDA.
    struct CachedBif2dModule {
        std::string  key;
        CUmodule     module        = nullptr;
        CUfunction   kernel_traj   = nullptr;   // calculateDiscreteModelCUDA
        CUfunction   kernel_peak   = nullptr;   // peakFinderCUDA
        CUfunction   kernel_dbscan = nullptr;   // dbscanCUDA
    };
    CachedBif2dModule cached_bif2d;

    ~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            release_module();
            release_lle_module();
            release_lle_2d_module();
            release_ls_module();
            release_ls_2d_module();
            release_cont_module();
            release_bif2d_module();
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

    void release_lle_module() {
        if (cached_lle.module) {
            cuModuleUnload(cached_lle.module);
            cached_lle.module = nullptr;
            cached_lle.kernel_lle = nullptr;
            cached_lle.key.clear();
        }
    }

    void release_lle_2d_module() {
        if (cached_lle_2d.module) {
            cuModuleUnload(cached_lle_2d.module);
            cached_lle_2d.module = nullptr;
            cached_lle_2d.kernel_lle = nullptr;
            cached_lle_2d.key.clear();
        }
    }

    void release_ls_module() {
        if (cached_ls.module) {
            cuModuleUnload(cached_ls.module);
            cached_ls.module = nullptr;
            cached_ls.kernel_ls = nullptr;
            cached_ls.key.clear();
        }
    }

    void release_ls_2d_module() {
        if (cached_ls_2d.module) {
            cuModuleUnload(cached_ls_2d.module);
            cached_ls_2d.module = nullptr;
            cached_ls_2d.kernel_ls = nullptr;
            cached_ls_2d.key.clear();
        }
    }

    void release_cont_module() {
        if (cached_cont.module) {
            cuModuleUnload(cached_cont.module);
            cached_cont.module = nullptr;
            cached_cont.kernel_cont = nullptr;
            cached_cont.kernel_peak = nullptr;
            cached_cont.key.clear();
        }
    }

    void release_bif2d_module() {
        if (cached_bif2d.module) {
            cuModuleUnload(cached_bif2d.module);
            cached_bif2d.module        = nullptr;
            cached_bif2d.kernel_traj   = nullptr;
            cached_bif2d.kernel_peak   = nullptr;
            cached_bif2d.kernel_dbscan = nullptr;
            cached_bif2d.key.clear();
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
        src_template          = read_text_file(root + "bifurcation1d.template.cu",      e); if (!e.empty()) { err = e; return false; }
        src_template_cont     = read_text_file(root + "bifurcation1d_cont.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_template_lle      = read_text_file(root + "lle1d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_lle_2d   = read_text_file(root + "lle2d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_ls       = read_text_file(root + "ls1d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_ls_2d    = read_text_file(root + "ls2d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_bif2d    = read_text_file(root + "bifurcation2d.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h      = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        srcs_loaded = true;
        return true;
    }

    bool compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err) {
        // Если worker-thread унаследовал чужой контекст (NvrtcEngine, например),
        // компиляция и загрузка модуля прицепят символы не в тот контекст. Жёстко
        // выставляем наш перед NVRTC/CU-вызовами.
        cuCtxSetCurrent(context);
        // par_or_var в kernel'е cudaLibrary.cu — compile-time макрос. Поэтому
        // включаем его в hash-key: param-sweep и IC-sweep кешируются отдельно.
        std::string key = hash_key(krs_body, amountOfX) + ":pov" + std::to_string(par_or_var);
        if (cached.module && cached.key == key) return true;  // cache hit
        release_module();

        if (!load_sources(err)) return false;

        // Подстановка плейсхолдеров в шаблон
        std::string src = src_template;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

        // Виртуальные заголовки для NVRTC. curand_kernel.h-stub НЕ нужен здесь:
        // inline-stub в template'е + `#define CURAND_KERNEL_H_` блокируют как
        // реальный header (по -I path), так и любой повторный inject.
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

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        nvrtcAddNameExpression(prog, "calculateDiscreteModelCUDA");
        nvrtcAddNameExpression(prog, "peakFinderCUDA");

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

        // Mangled-имена для обоих kernel-ов. Нужно скопировать в свои строки
        // ДО nvrtcDestroyProgram — после destroy указатели становятся невалидны.
        const char* mangled_traj_ptr = nullptr;
        const char* mangled_peak_ptr = nullptr;
        nvrtcGetLoweredName(prog, "calculateDiscreteModelCUDA", &mangled_traj_ptr);
        nvrtcGetLoweredName(prog, "peakFinderCUDA",             &mangled_peak_ptr);
        std::string mangled_traj = mangled_traj_ptr ? mangled_traj_ptr : "calculateDiscreteModelCUDA";
        std::string mangled_peak = mangled_peak_ptr ? mangled_peak_ptr : "peakFinderCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx: " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached.kernel_traj, cached.module, mangled_traj.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_traj + "): " + cu_err(r);
            release_module(); return false;
        }
        r = cuModuleGetFunction(&cached.kernel_peak, cached.module, mangled_peak.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_peak + "): " + cu_err(r);
            release_module(); return false;
        }

        cached.key = key;
        return true;
    }

    // =========================================================================
    // run_bif1d — порт NonLinAnal::bifurcation1D из hostLibrary.cu:165-655.
    // Идея: брать оригинальный код почти как есть, чтобы при обновлениях
    // NonLinAnal перенос был механическим diff → patch. Изменения, которые
    // ОБЯЗАТЕЛЬНЫ (помечены комментарием [ADAPT]):
    //   - <<<grid, block, shared>>>(...) → cuLaunchKernel(CUfunction, ...) —
    //     потому что наш kernel-модуль скомпилирован NVRTC'ом во время работы
    //     и недоступен по compile-time символу
    //   - gpuErrorCheck(...) — у NonLinAnal он зовёт exit() при ошибке; здесь
    //     заменено локальным макросом BIF_CHECK, который пишет в res.error и
    //     возвращает результат с cleanup'ом
    //   - OUT_FILE_PATH приходит из req.csv_output_path (пусто = CSV не пишем)
    //   - continuation_bif1D == 1 ветка отключена — она использует host-side
    //     calculateDiscreteModel (default Lorenz из cudaLibrary.cu:120), а не
    //     user's KRS; для нашего адаптера это не сработает
    //   - calculate_mean_med_freq / calculate_mean_and_variance отключены —
    //     соответствующие kernel-ы (MeanAndMedianFreqCUDA и т.д.) не входят
    //     в NVRTC-bundle. Включим по мере необходимости
    //   - Результат пишется в Bifurcation1DResult (память + CSV), не в файл
    // =========================================================================
    Bifurcation1DResult run_bif1d(const Bifurcation1DRequest& req) {
        // Continuation требует sequential x-carry — это совсем другой путь
        // (single-thread kernel). Отказываем при IC-sweep (не имеет смысла:
        // continuation подразумевает param как непрерывный параметр).
        if (req.continuation) {
            if (req.sweep_over_var) {
                Bifurcation1DResult r;
                r.error = "continuation требует param-sweep, не IC-sweep";
                return r;
            }
            return run_bif1d_continuation(req);
        }

        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        // ---- валидация ----
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        // base_values уже идёт со сдвигом +1 (a[0] зарезервирован):
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index вне диапазона");
        } else {
            if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index вне диапазона");
        }
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
        if (!compile_if_needed(req.krs_body, req.amountOfX,
                               req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ====================================================================
        // ПОРТ NonLinAnal::bifurcation1D (hostLibrary.cu:165-655).
        // Локальные имена мапятся на аргументы функции NonLinAnal для удобства
        // дифа — слева name из req, справа name как в hostLibrary.
        // ====================================================================
        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[2]                        = { req.param_lo, req.param_hi };
        // Sweep target:
        //   param-sweep → indicesOfMutVars[0] = 1-based индекс параметра (a[]),
        //                 par_or_var_arg = true  → kernel пишет в localValues.
        //   var-sweep   → indicesOfMutVars[0] = 0-based индекс переменной (X[]),
        //                 par_or_var_arg = false → kernel пишет в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // Константы из configCUDA.h — у NonLinAnal они constexpr,
        // здесь литералы (значения те же).
        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;
        // [ADAPT] continuation_bif1D, calculate_mean_med_freq отключены —
        // см. комментарий в шапке функции.

        // --- amountOfPointsInBlock / amountOfPointsForSkip — порт строк 196-200 NL ---
        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        // --- Memory budget (порт строк 202-244 NL) ---
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory = (size_t)((double)freeMemory * 0.92);

        size_t memPerSystem =
            3 * (size_t)amountOfPointsInBlock * sizeof(double) +  // d_data, d_outPeaks, d_timeOfPeaks
            2 * sizeof(double) +                                  // d_meanFreq, d_medianFreq (зарезервировано)
            sizeof(int);                                          // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(double) +
            sizeof(int) +
            (size_t)amountOfInitialConditions * sizeof(double) +
            (size_t)amountOfValues * sizeof(double);
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        nPtsLimiter = (nPtsLimiter / blockSize_setup) * blockSize_setup;
        if (nPtsLimiter == 0)
            return fail("not enough GPU memory: per-system buffer too large");
        size_t originalNPtsLimiter = nPtsLimiter;

        // --- Host buffers (порт строк 257-264 NL) ---
        // h_data/h_meanFreq/h_medianFreq/h_localX/h_localValues нужны только для
        // continuation_bif1D и mean/median — мы их не используем.
        std::vector<double> h_outPeaks   (nPtsLimiter * (size_t)amountOfPointsInBlock);
        std::vector<double> h_timeOfPeaks(nPtsLimiter * (size_t)amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);

        // --- Device buffers (порт строк 297-306 NL, без d_meanFreq/d_medianFreq) ---
        double* d_data              = nullptr;
        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        double* d_outPeaks          = nullptr;
        double* d_timeOfPeaks       = nullptr;

        auto cleanup = [&]() {
            if (d_data)              cudaFree(d_data);
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_outPeaks)          cudaFree(d_outPeaks);
            if (d_timeOfPeaks)       cudaFree(d_timeOfPeaks);
        };

        // [ADAPT] gpuErrorCheck → BIF_CHECK: пишем в res.error и выходим с cleanup
        #define BIF_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        BIF_CHECK(cudaMalloc((void**)&d_data,              nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_data");
        BIF_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(double)),                                          "cudaMalloc d_ranges");
        BIF_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        BIF_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),          "cudaMalloc d_initialConditions");
        BIF_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),                     "cudaMalloc d_values");
        BIF_CHECK(cudaMalloc((void**)&d_outPeaks,          nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_outPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_timeOfPeaks,       nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_timeOfPeaks");
        BIF_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");

        // --- H2D констант (порт строк 314-319 NL) ---
        BIF_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(double),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),            cudaMemcpyHostToDevice), "memcpy d_values");
        BIF_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // --- Config CSV (порт строк 331-376 NL — упрощённо, только если путь задан) ---
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "1D classical bifurcation\n";
                cfg << "Parameter estimation\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != amountOfInitialConditions - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = "       << tMax << "\n";
                cfg << "TT = "       << transientTime << "\n";
                cfg << "h = "        << h << "\n";
                cfg << "decimator = " << preScaller << "\n";
                cfg << "indexVar for peakfinder = " << writableVar << "\n";
                cfg << "indexPar for estimation = " << indicesOfMutVars[0] << "\n";
                cfg << "start value = " << ranges[0] << ", stop value = " << ranges[1] << "\n";
            }
            // обнуляем основной файл данных
            std::ofstream trunc(OUT_FILE_PATH);
            trunc.close();
        }

        // --- результат-аккумулятор (для GUI) ---
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});

        // --- Главный цикл (порт строк 396-630 NL) ---
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            // последний чанк может быть меньше
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            int blockSize = 32;
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // [ADAPT] <<<>>> → cuLaunchKernel.
            // continuation_bif1D == 0 ветка (classical). Continuation отключена.
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t sizeOfBlock_s             = (size_t)amountOfPointsInBlock;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            double h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            double maxValue_arg              = maxValue;
            bool   par_or_var_arg            = !req.sweep_over_var; // true=param, false=IC

            void* args_traj[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &sizeOfBlock_s,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfInitialConditions_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_data,
                &d_amountOfPeaks,
                &par_or_var_arg
            };

            unsigned int shared = (unsigned int)((amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize);

            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_traj,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_traj, nullptr),
                         "cuLaunchKernel(traj)");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after traj");

            // peakFinderCUDA
            double timeStep_arg = h * (double)preScaller;
            void* args_peak[] = {
                &d_data,
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_outPeaks,
                &d_timeOfPeaks,
                &timeStep_arg
            };
            BIF_CHECK_CU(cuLaunchKernel(cached.kernel_peak,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args_peak, nullptr),
                         "cuLaunchKernel(peak)");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after peak");

            // D2H (порт строк 538-540 NL)
            BIF_CHECK(cudaMemcpy(h_outPeaks.data(),       d_outPeaks,       nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double), cudaMemcpyDeviceToHost), "memcpy h_outPeaks");
            BIF_CHECK(cudaMemcpy(h_amountOfPeaks.data(),  d_amountOfPeaks,  nPtsLimiter * sizeof(int),                                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            BIF_CHECK(cudaMemcpy(h_timeOfPeaks.data(),    d_timeOfPeaks,    nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double), cudaMemcpyDeviceToHost), "memcpy h_timeOfPeaks");
            BIF_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // --- CSV + аккумуляция результата (порт строк 574-608 NL) ---
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }

            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                int    npeaks     = h_amountOfPeaks[k];

                // CSV — формат NonLinAnal: param, peak, time
                if (out.is_open()) {
                    if (npeaks == 0) {
                        out << param_val << ", " << 0 << ", " << 0 << '\n';
                    } else if (npeaks == -1) {
                        out << param_val << ", " << 0 << ", " << -1 << '\n';
                    } else {
                        for (size_t j = 0; j < (size_t)npeaks; ++j) {
                            out << param_val << ", "
                                << h_outPeaks   [k * (size_t)amountOfPointsInBlock + j] << ", "
                                << h_timeOfPeaks[k * (size_t)amountOfPointsInBlock + j] << '\n';
                        }
                    }
                }

                // В память для GUI: значения пиков и межпиковые интервалы
                res.flags[global_idx] = npeaks;
                auto& dst_peaks = res.bifurcation_points[global_idx];
                auto& dst_times = res.peak_times[global_idx];
                if (npeaks > 0) {
                    int n = npeaks;
                    if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
                    const double* peakRow = h_outPeaks.data()    + k * (size_t)amountOfPointsInBlock;
                    const double* timeRow = h_timeOfPeaks.data() + k * (size_t)amountOfPointsInBlock;
                    dst_peaks.assign(peakRow, peakRow + n);
                    dst_times.assign(timeRow, timeRow + n);
                } else {
                    dst_peaks.clear();
                    dst_times.clear();
                }
            }

            if (out.is_open()) out.close();
        }

        cleanup();
        #undef BIF_CHECK
        #undef BIF_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_lle_if_needed — отдельная компиляция для LLE-шаблона.
    // Ключ кэша = hash(krs_body + amountOfX + "lle"), чтобы PTX от bif1d
    // не путался с LLE даже при одной и той же KRS.
    // =========================================================================
    bool compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        // par_or_var в kernel — compile-time макрос (см. bif1d). Два PTX-модуля
        // кешируются отдельно: param-sweep и IC-sweep.
        std::string key = hash_key(krs_body, amountOfX) + ":lle:pov" + std::to_string(par_or_var);
        if (cached_lle.module && cached_lle.key == key) return true;
        release_lle_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_lle;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

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
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "lle1d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(lle): ") + nvrtcGetErrorString(nr); return false; }

        nvrtcAddNameExpression(prog, "LLEKernelCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "CUDA_PATH не задан (нужен для curand_kernel.h)";
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
            err = "NVRTC compile failed (lle):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mangled_lle_ptr = nullptr;
        nvrtcGetLoweredName(prog, "LLEKernelCUDA", &mangled_lle_ptr);
        std::string mangled_lle = mangled_lle_ptr ? mangled_lle_ptr : "LLEKernelCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_lle.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(lle): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_lle.kernel_lle, cached_lle.module, mangled_lle.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_lle + "): " + cu_err(r);
            release_lle_module(); return false;
        }

        cached_lle.key = key;
        return true;
    }

    // =========================================================================
    // run_lle_1d — порт NonLinAnal::LLE1D из hostLibrary.cu:2261-2511. Та же
    // diff-friendly стратегия, что у run_bif1d: оригинальное имена слева,
    // req-имена справа. [ADAPT]-комментарии — отличия от NonLinAnal.
    //   - <<<>>> → cuLaunchKernel (NVRTC-модуль).
    //   - gpuErrorCheck → локальный LLE_CHECK с cleanup'ом, без exit().
    //   - OUT_FILE_PATH — req.csv_output_path; пусто = без файла.
    //   - Результат пишется в LLE1DResult (память + опц. CSV).
    // =========================================================================
    LLE1DResult run_lle_1d(const LLE1DRequest& req) {
        LLE1DResult res;
        auto fail = [&](const std::string& msg) -> LLE1DResult& { res.error = msg; return res; };

        // ---- валидация ----
        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index вне диапазона");
        } else {
            if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index вне диапазона");
        }
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_if_needed(req.krs_body, req.amountOfX,
                                   req.sweep_over_var ? 0 : 1, err)) return fail(err);

        // ===== ПОРТ NonLinAnal::LLE1D (hostLibrary.cu:2261-2511) =====
        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[2]                        = { req.param_lo, req.param_hi };
        // Sweep target: см. bif1d. При IC-свипе индекс — 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        // amountOfPointsInBlock = tMax / NT — число NT-блоков интегрирования.
        int amountOfPointsInBlock = (int)(tMax / NT);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        // Memory budget — мирор NonLinAnal LLE1D:2291-2299 (консервативно).
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory = (size_t)((double)freeMemory * 0.5);

        size_t nPtsLimiter = freeMemory / (sizeof(double) * (size_t)amountOfPointsInBlock);
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<double> h_lleResult(nPtsLimiter);

        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        double* d_lleResult         = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };

        #define LLE_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        LLE_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(double)),                                "cudaMalloc d_ranges");
        LLE_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),"cudaMalloc d_initialConditions");
        LLE_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),           "cudaMalloc d_values");
        LLE_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(double)),                      "cudaMalloc d_lleResult");

        LLE_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(double),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Опциональный config CSV (порт NonLinAnal LLE1D:2355-2389)
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "1D LLE\nParameter estimation\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != amountOfInitialConditions - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = " << tMax << "\nNT = " << NT << "\nTT = " << transientTime << "\n";
                cfg << "h = "  << h    << "\neps = " << eps << "\n";
                cfg << "indexPar = " << indicesOfMutVars[0] << "\n";
                cfg << "start value = " << ranges[0] << ", stop value = " << ranges[1] << "\n";
            }
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // Результат-аккумулятор
        res.n_pts    = nPts;
        res.param_lo = ranges[0];
        res.param_hi = ranges[1];
        res.lyapunov.assign(nPts, 0.0);
        res.flags.assign(nPts, 0);

        // Главный цикл (порт NonLinAnal LLE1D:2403-2496)
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize по формуле NonLinAnal (hostLibrary.cu:2419), cap=32
            int blockSize = (int)std::ceil((1024.0 * 32.0) / ((double)(3 * amountOfInitialConditions + amountOfValues) * (double)sizeof(double)));
            if (blockSize < 1) blockSize = 1;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA (cudaLibrary.cu:2379)
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            double NT_arg                    = NT;
            double tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            int    amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            double h_arg                     = h;
            double eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            double maxValue_arg              = maxValue;

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult
            };

            // Shared = (3 * amountOfIC + amountOfValues) * sizeof(double) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(double) * blockSize);

            LLE_CHECK_CU(cuLaunchKernel(cached_lle.kernel_lle,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(lle)");
            LLE_CHECK(cudaDeviceSynchronize(), "sync after lle");

            LLE_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, nPtsLimiter * sizeof(double), cudaMemcpyDeviceToHost),
                      "memcpy h_lleResult");
            LLE_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // Аккумулируем + опциональный CSV (порт NonLinAnal LLE1D:2478-2492)
            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);
                double v          = h_lleResult[k];

                res.lyapunov[global_idx] = v;
                // 999 / -999 — спец-флаги из kernel'а (нет аттрактора / разошлось)
                res.flags[global_idx] = (v == 999.0 || v == -999.0) ? -1 : 1;

                if (out.is_open()) out << param_val << ", " << v << '\n';
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LLE_CHECK
        #undef LLE_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_lle_2d_if_needed — отдельный PTX-кэш для LLE-2D. Структура та
    // же, что у compile_lle_if_needed; отличия только в исходнике шаблона
    // (lle2d.template.cu) и в маркере ключа кэша (":lle2d"). Сам kernel
    // (LLEKernelCUDA) — тот же, ловится по тому же имени.
    // =========================================================================
    bool compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":lle2d:pov" + std::to_string(par_or_var);
        if (cached_lle_2d.module && cached_lle_2d.key == key) return true;
        release_lle_2d_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_lle_2d;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

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
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "lle2d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(lle2d): ") + nvrtcGetErrorString(nr); return false; }

        nvrtcAddNameExpression(prog, "LLEKernelCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "CUDA_PATH не задан (нужен для curand_kernel.h)";
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
            err = "NVRTC compile failed (lle2d):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mangled_ptr = nullptr;
        nvrtcGetLoweredName(prog, "LLEKernelCUDA", &mangled_ptr);
        std::string mangled = mangled_ptr ? mangled_ptr : "LLEKernelCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_lle_2d.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(lle2d): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_lle_2d.kernel_lle, cached_lle_2d.module, mangled.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled + "): " + cu_err(r);
            release_lle_2d_module(); return false;
        }

        cached_lle_2d.key = key;
        return true;
    }

    // =========================================================================
    // run_lle_2d — λ(p1, p2) на квадратной сетке. Порт NonLinAnal::LLE2D
    // (hostLibrary.cu:2514) на NVRTC-engine. Отличия от LLE1D:
    //   - ranges[4], indicesOfMutVars[2];
    //   - dimension=2;
    //   - chunking по cell'ам (n_pts × n_pts может не влезть в память за раз);
    //   - результат — плоский row-major массив n_pts × n_pts.
    // Идея par_or_var (compile-time):
    //   - mixed_mode=true → par_or_var=2 (см. cudaLibrary.cu:2439);
    //   - иначе оба свипа одного типа: par_or_var=1 если обе оси param,
    //     0 если обе оси IC; смешанные комбинации без mixed_mode не поддержаны
    //     (validator отказывает — kernel-ветка под это не предусмотрена).
    // =========================================================================
    LLE2DResult run_lle_2d(const LLE2DRequest& req) {
        LLE2DResult res;
        auto fail = [&](const std::string& msg) -> LLE2DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне допустимого диапазона");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");

        // par_or_var — compile-time. Логика маппинга см. parametric_engine.h.
        // Дополнительно нужен swap_xy: если X=param, Y=IC, передаём в kernel с
        // X↔Y и потом транспонируем результат на хосте.
        auto check_param = [&](int p1based) -> bool {
            return p1based > 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;   // что передаём в indicesOfMutVars[0/1]
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;

        if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = req.sweep_over_var ? 0 : 1;
            if (par_or_var == 1) {
                if (!check_param(req.param_index))    return fail("param_index (ось X) вне диапазона");
                if (!check_param(req.param_index_2))  return fail("param_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))    return fail("var_sweep_index (ось X) вне диапазона");
                if (!check_var(req.var_sweep_index_2))  return fail("var_sweep_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            // X=IC, Y=param — нативно соответствует ветке par_or_var=2 kernel'а.
            par_or_var = 2;
            if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
            if (!check_param(req.param_index_2))   return fail("param_index_2 (ось Y) вне диапазона");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            // X=param, Y=IC — kernel напрямую не поддерживает, свопаем оси
            // под капотом и транспонируем результат при выгрузке.
            par_or_var = 2;
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
            if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
            // В kernel: ось 1 (IC) = пользовательский Y, ось 2 (param) = X.
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
        }

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_lle_2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;       // сторона сетки
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[4]                        = { ranges_lo_x, ranges_hi_x,
                                                    ranges_lo_y, ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — мирор NonLinAnal LLE2D (hostLibrary.cu:2535-2547).
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory = (size_t)((double)freeMemory * 0.5);

        size_t nPtsLimiter = freeMemory / (sizeof(double) * (size_t)amountOfPointsInBlock);
        if (nPtsLimiter == 0)                  nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)         nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<double> h_lleResult(nPtsLimiter);

        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        double* d_lleResult         = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lleResult)         cudaFree(d_lleResult);
        };

        #define LLE2_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LLE2_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        LLE2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(double)),                                "cudaMalloc d_ranges");
        LLE2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LLE2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),"cudaMalloc d_initialConditions");
        LLE2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),           "cudaMalloc d_values");
        LLE2_CHECK(cudaMalloc((void**)&d_lleResult,         nPtsLimiter * sizeof(double)),                      "cudaMalloc d_lleResult");

        LLE2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(double),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LLE2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LLE2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_ic");
        LLE2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),            cudaMemcpyHostToDevice), "memcpy d_values");
        LLE2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Опциональный config CSV (порт LLE2D:2575-2611).
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "2D LLE\n";
                if (par_or_var == 1) cfg << "Parameter estimation\n";
                if (par_or_var == 0) cfg << "Initial conditions estimation\n";
                if (par_or_var == 2) cfg << "Mixed: x=IC, y=parameter\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != amountOfInitialConditions - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = " << tMax << "\nNT = " << NT << "\nTT = " << transientTime << "\n";
                cfg << "h = " << h << "\neps = " << eps << "\n";
                cfg << "indices = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
                cfg << "axis1: " << ranges[0] << " .. " << ranges[1] << "\n";
                cfg << "axis2: " << ranges[2] << " .. " << ranges[3] << "\n";
                cfg << "n_pts = " << nPts << "x" << nPts << "\n";
            }
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        // В result диапазоны храним всегда «пользовательские» (X = первая ось
        // запроса), не свопнутые ranges[] — чтобы GUI рисовал оси правильно
        // независимо от того, делал ли engine внутренний свап.
        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        std::ofstream out;
        if (!OUT_FILE_PATH.empty()) {
            out.open(OUT_FILE_PATH);
            if (out.is_open()) out << std::setprecision(set_precision);
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            int blockSize = (int)std::ceil((1024.0 * 32.0) / ((double)(3 * amountOfInitialConditions + amountOfValues) * (double)sizeof(double)));
            if (blockSize < 1) blockSize = 1;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // Аргументы LLEKernelCUDA — те же 21 параметр, что и в LLE1D,
            // но dimension=2 и амбулатура индекса nPts остаётся "сторона сетки".
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            double NT_arg                    = NT;
            double tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            int    amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            double h_arg                     = h;
            double eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            double maxValue_arg              = maxValue;

            void* args[] = {
                &nPts_arg,
                &nPtsLimiter_arg,
                &NT_arg,
                &tMax_arg,
                &sizeOfBlock_arg,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_arg,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &eps_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_arg,
                &d_values,
                &amountOfValues_arg,
                &amountOfIterations_arg,
                &preScaller_arg,
                &writableVar_arg,
                &maxValue_arg,
                &d_lleResult
            };

            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions + amountOfValues)
                                                 * sizeof(double) * blockSize);

            LLE2_CHECK_CU(cuLaunchKernel(cached_lle_2d.kernel_lle,
                                         gridSize, 1, 1, blockSize, 1, 1,
                                         shared, nullptr, args, nullptr),
                          "cuLaunchKernel(lle2d)");
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after lle2d");

            LLE2_CHECK(cudaMemcpy(h_lleResult.data(), d_lleResult, cur_limiter * sizeof(double), cudaMemcpyDeviceToHost),
                       "memcpy h_lleResult");
            LLE2_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                double v          = h_lleResult[k];
                // Транспонирование при свопе: kernel_idx = kix + n*kiy, в
                // системе пользователя ix=kiy, iy=kix → user_idx = kix*n + kiy.
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                res.values[out_idx] = v;
                res.flags[out_idx]  = (v == 999.0 || v == -999.0) ? -1 : 1;
                if (out.is_open()) out << v << ((k + 1 < cur_limiter) ? ", " : "\n");
            }
        }
        if (out.is_open()) out.close();

        // Авто-нормализация для colormap'а: min/max по валидным значениям.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < total_cells; ++k) {
            int f = res.flags[k];
            double v = res.values[k];
            if (f < 0) continue;
            if (!std::isfinite(v)) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        if (std::isfinite(vmin) && std::isfinite(vmax)) {
            res.min_val = vmin;
            res.max_val = vmax;
        } else {
            res.min_val = 0.0;
            res.max_val = 0.0;
        }

        cleanup();
        #undef LLE2_CHECK
        #undef LLE2_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_ls_if_needed — третий шаблон. Ключ кэша помечен ":ls".
    // =========================================================================
    bool compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls:pov" + std::to_string(par_or_var);
        if (cached_ls.module && cached_ls.key == key) return true;
        release_ls_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_ls;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

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
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "ls1d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(ls): ") + nvrtcGetErrorString(nr); return false; }

        nvrtcAddNameExpression(prog, "LSKernelCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "CUDA_PATH не задан (нужен для curand_kernel.h)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device", cuda_include_opt.c_str() };

        nr = nvrtcCompileProgram(prog, 4, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile failed (ls):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mangled_ptr = nullptr;
        nvrtcGetLoweredName(prog, "LSKernelCUDA", &mangled_ptr);
        std::string mangled = mangled_ptr ? mangled_ptr : "LSKernelCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_ls.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(ls): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_ls.kernel_ls, cached_ls.module, mangled.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled + "): " + cu_err(r);
            release_ls_module(); return false;
        }

        cached_ls.key = key;
        return true;
    }

    // =========================================================================
    // run_ls_1d — порт NonLinAnal::LS1D (hostLibrary.cu:2698-2868). Той же
    // стратегией, что run_bif1d / run_lle_1d. Per-system результат — вектор
    // длины amountOfX (один Ляпунов на каждую переменную).
    //
    // Memory budget per thread у LS значительно больше: shared = (3*N + 2*N^2
    // + nValues) * sizeof(double) * blockSize. blockSize подбирается из 32K
    // shared-лимита (как делает NonLinAnal).
    // =========================================================================
    LS1DResult run_ls_1d(const LS1DRequest& req) {
        LS1DResult res;
        auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index вне диапазона");
        } else {
            if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index вне диапазона");
        }
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_ls_if_needed(req.krs_body, req.amountOfX,
                                  req.sweep_over_var ? 0 : 1, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[2]                        = { req.param_lo, req.param_hi };
        // Sweep target: param-sweep → 1-based в localValues; IC-sweep → 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        int amountOfPointsForSkip = (int)(transientTime / h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        // Memory budget — мирор NonLinAnal LS1D:2719-2727 (агрессивно делит /16,
        // т.к. per-system memory ~ N).
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory /= 16;

        size_t perSystemBytes = sizeof(double) * (size_t)amountOfPointsInBlock * (size_t)amountOfInitialConditions;
        if (perSystemBytes == 0) perSystemBytes = sizeof(double);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult хранит nPtsLimiter × N row-major.
        std::vector<double> h_lsResult(nPtsLimiter * (size_t)amountOfInitialConditions);

        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        double* d_lsResult          = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };

        #define LS_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        LS_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(double)),                                                          "cudaMalloc d_ranges");
        LS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                                             "cudaMalloc d_indicesOfMutVars");
        LS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),                          "cudaMalloc d_initialConditions");
        LS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),                                     "cudaMalloc d_values");
        LS_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(double)),            "cudaMalloc d_lsResult");

        LS_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(double),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_ic");
        LS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),            cudaMemcpyHostToDevice), "memcpy d_values");
        LS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Опциональный config CSV
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "1D LS\nParameter estimation\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != amountOfInitialConditions - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = " << tMax << "\nNT = " << NT << "\nTT = " << transientTime << "\n";
                cfg << "h = "  << h    << "\neps = " << eps << "\n";
                cfg << "indexPar = " << indicesOfMutVars[0] << "\n";
                cfg << "start value = " << ranges[0] << ", stop value = " << ranges[1] << "\n";
            }
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        res.n_pts       = nPts;
        res.n_exponents = amountOfInitialConditions;
        res.param_lo    = ranges[0];
        res.param_hi    = ranges[1];
        res.spectrum.assign(nPts, std::vector<double>(amountOfInitialConditions, 0.0));
        res.flags.assign(nPts, 0);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize: 32K shared / per-thread, cap=32 (порт LS1D:2821-2824)
            int blockSizeMax = (int)(32000 / ((double)(3 * amountOfInitialConditions
                                + 2 * amountOfInitialConditions * amountOfInitialConditions
                                + amountOfValues) * (double)sizeof(double)));
            int blockSize = blockSizeMax;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            double NT_arg                    = NT;
            double tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            int    amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            double h_arg                     = h;
            double eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            double maxValue_arg              = maxValue;

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult
            };

            // Shared = (3N + 2N² + nValues) * sizeof(double) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions
                                                 + 2 * amountOfInitialConditions * amountOfInitialConditions
                                                 + amountOfValues)
                                                * sizeof(double) * blockSize);

            LS_CHECK_CU(cuLaunchKernel(cached_ls.kernel_ls,
                                       gridSize, 1, 1, blockSize, 1, 1,
                                       shared, nullptr, args, nullptr),
                        "cuLaunchKernel(ls)");
            LS_CHECK(cudaDeviceSynchronize(), "sync after ls");

            LS_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "memcpy h_lsResult");
            LS_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            std::ofstream out;
            if (!OUT_FILE_PATH.empty()) {
                out.open(OUT_FILE_PATH, std::ios::app);
                if (out.is_open()) out << std::setprecision(set_precision);
            }
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = originalNPtsLimiter * iter + k;
                double param_val  = getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);

                // первая экспонента используется как ground-truth для флага
                double first = h_lsResult[k * (size_t)amountOfInitialConditions + 0];
                res.flags[global_idx] = (first == 999.0 || first == -999.0) ? -1 : 1;

                auto& row = res.spectrum[global_idx];
                for (int j = 0; j < amountOfInitialConditions; ++j) {
                    row[j] = h_lsResult[k * (size_t)amountOfInitialConditions + j];
                }

                if (out.is_open()) {
                    out << param_val;
                    for (int j = 0; j < amountOfInitialConditions; ++j)
                        out << ", " << row[j];
                    out << '\n';
                }
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LS_CHECK
        #undef LS_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_ls_2d_if_needed — отдельный шаблон ls2d.template.cu, тот же kernel
    // LSKernelCUDA. Ключ кэша помечен ":ls2d:" — изолирован от ":ls:" slot'а.
    // =========================================================================
    bool compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls2d:pov" + std::to_string(par_or_var);
        if (cached_ls_2d.module && cached_ls_2d.key == key) return true;
        release_ls_2d_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_ls_2d;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

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
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "ls2d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(ls2d): ") + nvrtcGetErrorString(nr); return false; }

        nvrtcAddNameExpression(prog, "LSKernelCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH) {
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
            }
        }
        if (cuda_include_opt.empty()) {
            err = "CUDA_PATH не задан (нужен для curand_kernel.h)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device", cuda_include_opt.c_str() };

        nr = nvrtcCompileProgram(prog, 4, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile failed (ls2d):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mangled_ptr = nullptr;
        nvrtcGetLoweredName(prog, "LSKernelCUDA", &mangled_ptr);
        std::string mangled = mangled_ptr ? mangled_ptr : "LSKernelCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_ls_2d.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(ls2d): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_ls_2d.kernel_ls, cached_ls_2d.module, mangled.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled + "): " + cu_err(r);
            release_ls_2d_module(); return false;
        }

        cached_ls_2d.key = key;
        return true;
    }

    // =========================================================================
    // run_ls_2d — спектр Ляпунова на сетке n_pts × n_pts. Гибрид run_ls_1d
    // (per-system буфер размером N экспонент) и run_lle_2d (swap_xy логика,
    // chunking по cell'ам). На каждую ячейку kernel возвращает N значений; D2H
    // копирует cur_limiter * N doubles, host распаковывает по плоскостям —
    // values[k*n*n + iy*n + ix] = k-я экспонента в ячейке (ix, iy).
    // =========================================================================
    LS2DResult run_ls_2d(const LS2DRequest& req) {
        LS2DResult res;
        auto fail = [&](const std::string& msg) -> LS2DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне допустимого диапазона");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("base_values слишком много");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");

        // par_or_var + swap_xy: точная копия логики из run_lle_2d.
        auto check_param = [&](int p1based) -> bool {
            return p1based > 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;

        if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = req.sweep_over_var ? 0 : 1;
            if (par_or_var == 1) {
                if (!check_param(req.param_index))    return fail("param_index (ось X) вне диапазона");
                if (!check_param(req.param_index_2))  return fail("param_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))    return fail("var_sweep_index (ось X) вне диапазона");
                if (!check_var(req.var_sweep_index_2))  return fail("var_sweep_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
            if (!check_param(req.param_index_2))   return fail("param_index_2 (ось Y) вне диапазона");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
            if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
        }

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_ls_2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        const double tMax                       = req.t_max;
        const double NT                         = req.NT;
        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const double eps                        = req.eps;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[4]                        = { ranges_lo_x, ranges_hi_x,
                                                    ranges_lo_y, ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;
        const int N = amountOfInitialConditions;

        // Memory budget — мирор run_ls_1d (агрессивно делит /16, т.к. per-system
        // память ~N). total_cells заменяет nPts.
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess)
            return fail("cudaMemGetInfo failed");
        freeMemory /= 16;

        size_t perSystemBytes = sizeof(double) * (size_t)amountOfPointsInBlock * (size_t)N;
        if (perSystemBytes == 0) perSystemBytes = sizeof(double);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)             nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)    nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult — nPtsLimiter × N row-major (как в run_ls_1d).
        std::vector<double> h_lsResult(nPtsLimiter * (size_t)N);

        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        double* d_lsResult          = nullptr;

        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };

        #define LS2_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define LS2_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        LS2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(double)),                                "cudaMalloc d_ranges");
        LS2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LS2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)N * sizeof(double)),                        "cudaMalloc d_initialConditions");
        LS2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),           "cudaMalloc d_values");
        LS2_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)N * sizeof(double)),          "cudaMalloc d_lsResult");

        LS2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(double),                       cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                          cudaMemcpyHostToDevice), "memcpy d_indices");
        LS2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)N * sizeof(double),               cudaMemcpyHostToDevice), "memcpy d_ic");
        LS2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),  cudaMemcpyHostToDevice), "memcpy d_values");
        LS2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Опциональный config CSV.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "2D LS\n";
                if (par_or_var == 1) cfg << "Parameter estimation\n";
                if (par_or_var == 0) cfg << "Initial conditions estimation\n";
                if (par_or_var == 2) cfg << "Mixed: x=IC, y=parameter\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) {
                    cfg << values[kk];
                    if (kk != amountOfValues - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "X0[" << N << "] = { ";
                for (int kk = 0; kk < N; ++kk) {
                    cfg << initialConditions[kk];
                    if (kk != N - 1) cfg << ", "; else cfg << " }\n";
                }
                cfg << "CT = " << tMax << "\nNT = " << NT << "\nTT = " << transientTime << "\n";
                cfg << "h = " << h << "\neps = " << eps << "\n";
                cfg << "indices = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
                cfg << "axis1: " << ranges[0] << " .. " << ranges[1] << "\n";
                cfg << "axis2: " << ranges[2] << " .. " << ranges[3] << "\n";
                cfg << "n_pts = " << nPts << "x" << nPts << "\n";
                cfg << "exponents = " << N << "\n";
            }
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        res.n_pts       = nPts;
        res.n_exponents = N;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.param_lo_2  = req.param_lo_2;
        res.param_hi_2  = req.param_hi_2;
        res.values.assign((size_t)N * total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        std::ofstream out;
        if (!OUT_FILE_PATH.empty()) {
            out.open(OUT_FILE_PATH);
            if (out.is_open()) out << std::setprecision(set_precision);
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: тот же расчёт что в run_ls_1d (32K shared / per-thread).
            int blockSizeMax = (int)(32000 / ((double)(3 * N + 2 * N * N + amountOfValues)
                                              * (double)sizeof(double)));
            int blockSize = blockSizeMax;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            double NT_arg                    = NT;
            double tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            int    amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            double h_arg                     = h;
            double eps_arg                   = eps;
            int    amountOfIC_arg            = N;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            double maxValue_arg              = maxValue;

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult
            };

            unsigned int shared = (unsigned int)((3 * N + 2 * N * N + amountOfValues)
                                                 * sizeof(double) * blockSize);

            LS2_CHECK_CU(cuLaunchKernel(cached_ls_2d.kernel_ls,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(ls2d)");
            LS2_CHECK(cudaDeviceSynchronize(), "sync after ls2d");

            LS2_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                 cur_limiter * (size_t)N * sizeof(double),
                                 cudaMemcpyDeviceToHost),
                      "memcpy h_lsResult");
            LS2_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            // Распаковка: для каждой ячейки чанка — N экспонент. Layout
            // values[k * total_cells + out_idx] — k-я плоскость contiguous.
            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                // Первая экспонента — ground-truth для флага (как в run_ls_1d).
                double first = h_lsResult[k * (size_t)N + 0];
                int flag = (first == 999.0 || first == -999.0) ? -1 : 1;
                res.flags[out_idx] = flag;
                for (int j = 0; j < N; ++j) {
                    double v = h_lsResult[k * (size_t)N + j];
                    res.values[(size_t)j * total_cells + out_idx] = v;
                }
                if (out.is_open()) {
                    for (int j = 0; j < N; ++j) {
                        out << h_lsResult[k * (size_t)N + j];
                        if (j + 1 < N) out << ", ";
                    }
                    out << '\n';
                }
            }
        }
        if (out.is_open()) out.close();

        // Per-plane min/max — autoscale colormap'а в GUI выбирает их по
        // активной экспоненте. Diverged-ячейки (flag<0) и 999/-999/NaN
        // исключаются.
        res.min_val.assign((size_t)N, 0.0);
        res.max_val.assign((size_t)N, 0.0);
        for (int j = 0; j < N; ++j) {
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < total_cells; ++c) {
                if (res.flags[c] < 0) continue;
                double v = res.values[(size_t)j * total_cells + c];
                if (!std::isfinite(v))         continue;
                if (v == 999.0 || v == -999.0) continue;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            if (std::isfinite(vmin) && std::isfinite(vmax)) {
                res.min_val[j] = vmin;
                res.max_val[j] = vmax;
            }
        }

        cleanup();
        #undef LS2_CHECK
        #undef LS2_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_bif1d_cont_if_needed — отдельный модуль (single-thread sequential
    // continuation kernel + peakFinderCUDA). Cache key с суффиксом :cont.
    // =========================================================================
    bool compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":cont";
        if (cached_cont.module && cached_cont.key == key) return true;
        release_cont_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_cont;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);

        const char* header_sources[] = {
            src_cudaLibrary_cu.c_str(),
            src_cudaLibrary_cuh.c_str(),
            src_cudaMacros_cuh.c_str(),
            src_configCUDA_h.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu", "cudaLibrary.cuh", "cudaMacros.cuh", "configCUDA.h",
        };
        constexpr int n_headers = 4;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "bifurcation1d_cont.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(cont): ") + nvrtcGetErrorString(nr); return false; }

        // bifurcation1dContinuationKernel — extern "C" (имя не мангается).
        // peakFinderCUDA — обычный C++ символ, регистрируем для mangled-имени.
        nvrtcAddNameExpression(prog, "peakFinderCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);
        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
        }
        if (cuda_include_opt.empty()) {
            err = "CUDA_PATH не задан";
            nvrtcDestroyProgram(&prog); return false;
        }
        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device", cuda_include_opt.c_str() };

        nr = nvrtcCompileProgram(prog, 4, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile failed (cont):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mangled_peak_ptr = nullptr;
        nvrtcGetLoweredName(prog, "peakFinderCUDA", &mangled_peak_ptr);
        std::string mangled_peak = mangled_peak_ptr ? mangled_peak_ptr : "peakFinderCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_cont.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(cont): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_cont.kernel_cont, cached_cont.module,
                                "bifurcation1dContinuationKernel");
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(bifurcation1dContinuationKernel): " + cu_err(r);
            release_cont_module(); return false;
        }
        r = cuModuleGetFunction(&cached_cont.kernel_peak, cached_cont.module, mangled_peak.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + mangled_peak + "): " + cu_err(r);
            release_cont_module(); return false;
        }

        cached_cont.key = key;
        return true;
    }

    // =========================================================================
    // run_bif1d_continuation — single-thread sequential. Каждый параметр
    // стартует с конечного x[] предыдущего. Direction (forward/reverse)
    // обрабатывается в kernel'е. PeakFinderCUDA вызывается из того же модуля.
    // =========================================================================
    Bifurcation1DResult run_bif1d_continuation(const Bifurcation1DRequest& req) {
        Bifurcation1DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)      return fail("amountOfX вне диапазона");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("base_values слишком много");
        if (req.param_index <= 0 || req.param_index >= (int)req.base_values.size())
                                                                     return fail("param_index вне диапазона");
        if (req.writable_var < 0 || req.writable_var >= req.amountOfX)
                                                                     return fail("writable_var вне диапазона");
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller должно быть > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err))
            return fail(err);

        const int amountOfPointsInBlock = (int)(req.t_max / req.h / req.pre_scaller);
        const int amountOfPointsForSkip = (int)(req.transient_time / req.h);
        if (amountOfPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts = req.n_pts;
        double* d_data       = nullptr;
        double* d_baseValues = nullptr;
        double* d_baseX      = nullptr;
        int*    d_amountOfPeaks = nullptr;
        double* d_outPeaks   = nullptr;
        double* d_timeOfPeaks= nullptr;

        auto cleanup = [&]() {
            if (d_data)       cudaFree(d_data);
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_amountOfPeaks) cudaFree(d_amountOfPeaks);
            if (d_outPeaks)   cudaFree(d_outPeaks);
            if (d_timeOfPeaks)cudaFree(d_timeOfPeaks);
        };
        #define C_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define C_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        const size_t dataBytes  = (size_t)nPts * (size_t)amountOfPointsInBlock * sizeof(double);
        C_CHECK(cudaMalloc((void**)&d_data,         dataBytes),                                       "cudaMalloc d_data");
        C_CHECK(cudaMalloc((void**)&d_baseValues,   (size_t)req.base_values.size() * sizeof(double)), "cudaMalloc d_baseValues");
        C_CHECK(cudaMalloc((void**)&d_baseX,        (size_t)req.amountOfX * sizeof(double)),          "cudaMalloc d_baseX");
        C_CHECK(cudaMalloc((void**)&d_amountOfPeaks,(size_t)nPts * sizeof(int)),                      "cudaMalloc d_amountOfPeaks");
        C_CHECK(cudaMalloc((void**)&d_outPeaks,     dataBytes),                                       "cudaMalloc d_outPeaks");
        C_CHECK(cudaMalloc((void**)&d_timeOfPeaks,  dataBytes),                                       "cudaMalloc d_timeOfPeaks");

        C_CHECK(cudaMemcpy(d_baseValues, req.base_values.data(),
                           req.base_values.size() * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_baseValues");
        C_CHECK(cudaMemcpy(d_baseX, req.initial_conditions.data(),
                           (size_t)req.amountOfX * sizeof(double), cudaMemcpyHostToDevice), "memcpy d_baseX");
        C_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // --- Launch continuation kernel ---
        int    nPts_arg              = nPts;
        double lo_arg                = req.param_lo;
        double hi_arg                = req.param_hi;
        bool   reverse_arg           = req.continuation_reverse;
        int    mutParamIdx_arg       = req.param_index;
        int    amountOfValues_arg    = (int)req.base_values.size();
        int    amountOfX_arg         = req.amountOfX;
        double h_arg                 = req.h;
        int    transient_steps_arg   = amountOfPointsForSkip;
        int    sizeOfBlock_arg       = amountOfPointsInBlock;
        int    preScaller_arg        = req.pre_scaller;
        int    writableVar_arg       = req.writable_var;
        double maxValue_arg          = req.max_value;

        void* cont_args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg,
            &mutParamIdx_arg,
            &d_baseValues, &amountOfValues_arg,
            &d_baseX,      &amountOfX_arg,
            &h_arg, &transient_steps_arg, &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_data, &d_amountOfPeaks
        };
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_cont,
                                  1, 1, 1, 1, 1, 1, 0, nullptr, cont_args, nullptr),
                   "cuLaunchKernel(cont)");
        C_CHECK(cudaDeviceSynchronize(), "sync after cont kernel");

        // --- Launch peakFinderCUDA на полученные данные ---
        // Сигнатура: (numb* data, size_t sizeOfBlock, int amountOfBlocks,
        //             int* amountOfPeaks, numb* outPeaks, numb* timeOfPeaks, numb h)
        size_t sizeOfBlock_s = (size_t)amountOfPointsInBlock;
        int    nBlocks       = nPts;
        double timeStep      = req.h * (double)req.pre_scaller;
        void* peak_args[] = {
            &d_data, &sizeOfBlock_s, &nBlocks,
            &d_amountOfPeaks, &d_outPeaks, &d_timeOfPeaks, &timeStep
        };
        int    peakBlock = 32;
        int    peakGrid  = (nPts + peakBlock - 1) / peakBlock;
        C_CHECK_CU(cuLaunchKernel(cached_cont.kernel_peak,
                                  peakGrid, 1, 1, peakBlock, 1, 1,
                                  0, nullptr, peak_args, nullptr),
                   "cuLaunchKernel(peak)");
        C_CHECK(cudaDeviceSynchronize(), "sync after peak");

        // --- D2H ---
        std::vector<double> h_outPeaks   ((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<double> h_timeOfPeaks((size_t)nPts * (size_t)amountOfPointsInBlock);
        std::vector<int>    h_amountOfPeaks((size_t)nPts);
        C_CHECK(cudaMemcpy(h_outPeaks.data(),    d_outPeaks,    dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_outPeaks");
        C_CHECK(cudaMemcpy(h_timeOfPeaks.data(), d_timeOfPeaks, dataBytes,             cudaMemcpyDeviceToHost), "memcpy out h_timeOfPeaks");
        C_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks, nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy out h_amountOfPeaks");
        C_CHECK(cudaDeviceSynchronize(), "sync after D2H");

        // --- Заполнение Result ---
        res.n_pts        = nPts;
        res.record_steps = amountOfPointsInBlock;
        res.param_lo     = req.param_lo;
        res.param_hi     = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.flags.assign(nPts, 0);
        res.bifurcation_points.assign(nPts, {});
        res.peak_times.assign(nPts, {});
        for (int j = 0; j < nPts; ++j) {
            int n = h_amountOfPeaks[j];
            res.flags[j] = n;
            if (n > 0) {
                if (n > amountOfPointsInBlock) n = amountOfPointsInBlock;
                const double* pr = h_outPeaks.data()    + (size_t)j * amountOfPointsInBlock;
                const double* tr = h_timeOfPeaks.data() + (size_t)j * amountOfPointsInBlock;
                res.bifurcation_points[j].assign(pr, pr + n);
                res.peak_times[j].assign(tr, tr + n);
            }
        }

        cleanup();
        #undef C_CHECK
        #undef C_CHECK_CU
        res.ok = true;
        return res;
    }

    // =========================================================================
    // compile_bif2d_if_needed — шаблон bifurcation2d.template.cu, три kernel'а:
    // calculateDiscreteModelCUDA + peakFinderCUDA + dbscanCUDA.
    // Cache key: ":bif2d:" + par_or_var.
    // =========================================================================
    bool compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":bif2d:" + std::to_string(par_or_var);
        if (cached_bif2d.module && cached_bif2d.key == key) return true;
        release_bif2d_module();

        if (!load_sources(err)) return false;

        std::string src = src_template_bif2d;
        src = replace_all(src, "{{AMOUNT_OF_X}}", std::to_string(amountOfX));
        src = replace_all(src, "{{KRS_BODY}}",    krs_body);
        src = replace_all(src, "{{PAR_OR_VAR}}",  std::to_string(par_or_var));

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
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), "bifurcation2d.cu",
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) { err = std::string("nvrtcCreateProgram(bif2d): ") + nvrtcGetErrorString(nr); return false; }

        nvrtcAddNameExpression(prog, "calculateDiscreteModelCUDA");
        nvrtcAddNameExpression(prog, "peakFinderCUDA");
        nvrtcAddNameExpression(prog, "dbscanCUDA");

        char arch[64];
        snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        std::string cuda_include_opt;
        {
            char buf[MAX_PATH];
            DWORD nlen = GetEnvironmentVariableA("CUDA_PATH", buf, MAX_PATH);
            if (nlen > 0 && nlen < MAX_PATH)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
        }
        if (cuda_include_opt.empty()) {
            err = "переменная окружения CUDA_PATH не задана";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device", cuda_include_opt.c_str() };
        nr = nvrtcCompileProgram(prog, 4, opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = "NVRTC compile(bif2d) failed:\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const char* mp_traj = nullptr, *mp_peak = nullptr, *mp_dbscan = nullptr;
        nvrtcGetLoweredName(prog, "calculateDiscreteModelCUDA", &mp_traj);
        nvrtcGetLoweredName(prog, "peakFinderCUDA",             &mp_peak);
        nvrtcGetLoweredName(prog, "dbscanCUDA",                 &mp_dbscan);
        std::string mangled_traj   = mp_traj   ? mp_traj   : "calculateDiscreteModelCUDA";
        std::string mangled_peak   = mp_peak   ? mp_peak   : "peakFinderCUDA";
        std::string mangled_dbscan = mp_dbscan ? mp_dbscan : "dbscanCUDA";

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        nvrtcDestroyProgram(&prog);

        CUresult r = cuModuleLoadDataEx(&cached_bif2d.module, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadDataEx(bif2d): " + cu_err(r); return false; }

        r = cuModuleGetFunction(&cached_bif2d.kernel_traj,   cached_bif2d.module, mangled_traj.c_str());
        if (r != CUDA_SUCCESS) { err = "cuModuleGetFunction(" + mangled_traj + "): " + cu_err(r); release_bif2d_module(); return false; }
        r = cuModuleGetFunction(&cached_bif2d.kernel_peak,   cached_bif2d.module, mangled_peak.c_str());
        if (r != CUDA_SUCCESS) { err = "cuModuleGetFunction(" + mangled_peak + "): " + cu_err(r); release_bif2d_module(); return false; }
        r = cuModuleGetFunction(&cached_bif2d.kernel_dbscan, cached_bif2d.module, mangled_dbscan.c_str());
        if (r != CUDA_SUCCESS) { err = "cuModuleGetFunction(" + mangled_dbscan + "): " + cu_err(r); release_bif2d_module(); return false; }

        cached_bif2d.key = key;
        return true;
    }

    // =========================================================================
    // run_bif2d — порт NonLinAnal::bifurcation2D (hostLibrary.cu:898-1724).
    // Три kernel'а: calculateDiscreteModelCUDA → peakFinderCUDA → dbscanCUDA.
    // Результат: число DBSCAN-кластеров на ячейку = период системы.
    // [ADAPT]: те же адаптации, что у run_bif1d + run_lle_2d (см. их комментарии).
    // =========================================================================
    Bifurcation2DResult run_bif2d(const Bifurcation2DRequest& req) {
        Bifurcation2DResult res;
        auto fail = [&](const std::string& msg) -> Bifurcation2DResult& { res.error = msg; return res; };

        // ---- валидация ----
        if (req.krs_body.empty())                                    return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("base_values слишком много");
        if (req.n_pts <= 0)          return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)            return fail("h должно быть > 0");
        if (req.t_max <= 0.0)        return fail("t_max должно быть > 0");
        if (req.transient_time < 0)  return fail("transient_time должно быть >= 0");
        if (req.pre_scaller <= 0)    return fail("pre_scaller должно быть > 0");
        if (req.eps_dbscan <= 0.0)   return fail("eps_dbscan должно быть > 0");
        if (req.writable_var < 0 || req.writable_var >= req.amountOfX) return fail("writable_var вне диапазона");

        // ---- par_or_var + swap_xy (та же логика что у run_lle_2d) ----
        auto check_param = [&](int p1based) -> bool {
            return p1based > 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int    idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool   swap_xy = false;

        if (req.sweep_over_var == req.sweep_over_var_2) {
            par_or_var = req.sweep_over_var ? 0 : 1;
            if (par_or_var == 1) {
                if (!check_param(req.param_index))   return fail("param_index (ось X) вне диапазона");
                if (!check_param(req.param_index_2)) return fail("param_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.param_index;
                idx_axis_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
                if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
                idx_axis_x = req.var_sweep_index;
                idx_axis_y = req.var_sweep_index_2;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index))  return fail("var_sweep_index (ось X) вне диапазона");
            if (!check_param(req.param_index_2))  return fail("param_index_2 (ось Y) вне диапазона");
            idx_axis_x = req.var_sweep_index;
            idx_axis_y = req.param_index_2;
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else {
            par_or_var = 2;
            if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
            if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
            idx_axis_x = req.var_sweep_index_2;
            idx_axis_y = req.param_index;
            ranges_lo_x = req.param_lo_2; ranges_hi_x = req.param_hi_2;
            ranges_lo_y = req.param_lo;   ranges_hi_y = req.param_hi;
            swap_xy = true;
        }

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif2d_if_needed(req.krs_body, req.amountOfX, par_or_var, err)) return fail(err);

        // ---- локальные переменные (порт hostLibrary.cu:bifurcation2D) ----
        const int    nPts                       = req.n_pts;
        const double tMax                       = req.t_max;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const double* initialConditions         = req.initial_conditions.data();
        double ranges[4]                        = { ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double* values                    = req.base_values.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int  blockSize_setup          = 32;
        constexpr int  set_precision            = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        int amountOfPointsForSkip = (int)(transientTime / h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller слишком малы)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // Memory budget — порт hostLibrary.cu:930-950.
        size_t freeMemory = 0, totalMemory = 0;
        if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess) return fail("cudaMemGetInfo failed");
        freeMemory = (size_t)((double)freeMemory * 0.92);

        size_t baseMemPerSystem = (size_t)amountOfPointsInBlock * 3 * sizeof(double) + 2 * sizeof(int);
        size_t memConstants     = (4 + (size_t)amountOfInitialConditions + (size_t)amountOfValues) * sizeof(double) + 2 * sizeof(int);
        if (memConstants >= freeMemory) return fail("not enough GPU memory for constants");

        size_t nPtsLimiter = (freeMemory - memConstants) / baseMemPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)             nPtsLimiter = total_cells;
        nPtsLimiter = (nPtsLimiter / blockSize_setup) * blockSize_setup;
        if (nPtsLimiter == 0) return fail("not enough GPU memory: per-system buffer too large");
        size_t originalNPtsLimiter = nPtsLimiter;

        // ---- host buffers ----
        std::vector<int> h_dbscanResult(nPtsLimiter);

        // ---- device buffers ----
        double* d_data              = nullptr;
        double* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        double* d_initialConditions = nullptr;
        double* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        double* d_intervals         = nullptr;
        double* d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;

        auto cleanup = [&]() {
            if (d_data)              cudaFree(d_data);
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_intervals)         cudaFree(d_intervals);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
        };

        #define BIF2D_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BIF2D_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)

        BIF2D_CHECK(cudaMalloc((void**)&d_data,              nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_data");
        BIF2D_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(double)),                                           "cudaMalloc d_ranges");
        BIF2D_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BIF2D_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(double)),           "cudaMalloc d_initialConditions");
        BIF2D_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(double)),                      "cudaMalloc d_values");
        BIF2D_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BIF2D_CHECK(cudaMalloc((void**)&d_intervals,         nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_intervals");
        BIF2D_CHECK(cudaMalloc((void**)&d_helpfulArray,      nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(double)), "cudaMalloc d_helpfulArray");
        BIF2D_CHECK(cudaMalloc((void**)&d_dbscanResult,      nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_dbscanResult");

        BIF2D_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(double),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BIF2D_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BIF2D_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(double),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BIF2D_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(double),             cudaMemcpyHostToDevice), "memcpy d_values");
        BIF2D_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Config CSV (опционально).
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            if (cfg.is_open()) {
                cfg << std::setprecision(set_precision);
                cfg << "2D bifurcation (DBSCAN)\n";
                if (par_or_var == 1) cfg << "Parameter estimation\n";
                if (par_or_var == 0) cfg << "Initial conditions estimation\n";
                if (par_or_var == 2) cfg << "Mixed: x=IC, y=parameter\n";
                cfg << "a[" << amountOfValues << "] = { ";
                for (int kk = 0; kk < amountOfValues; ++kk) { cfg << values[kk]; if (kk != amountOfValues-1) cfg << ", "; else cfg << " }\n"; }
                cfg << "X0[" << amountOfInitialConditions << "] = { ";
                for (int kk = 0; kk < amountOfInitialConditions; ++kk) { cfg << initialConditions[kk]; if (kk != amountOfInitialConditions-1) cfg << ", "; else cfg << " }\n"; }
                cfg << "CT = " << tMax << "\nTT = " << transientTime << "\nh = " << h << "\ndecimator = " << preScaller << "\n";
                cfg << "eps_DBSCAN = " << eps_dbscan << "\n";
                cfg << "indexVar for peakfinder = " << writableVar << "\n";
                cfg << "indices = " << indicesOfMutVars[0] << ", " << indicesOfMutVars[1] << "\n";
                cfg << "axis1: " << ranges[0] << " .. " << ranges[1] << "\n";
                cfg << "axis2: " << ranges[2] << " .. " << ranges[3] << "\n";
                cfg << "n_pts = " << nPts << "x" << nPts << "\n";
            }
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        res.n_pts      = nPts;
        res.param_lo   = req.param_lo;
        res.param_hi   = req.param_hi;
        res.param_lo_2 = req.param_lo_2;
        res.param_hi_2 = req.param_hi_2;
        res.values.assign(total_cells, 0.0);
        res.flags.assign(total_cells,  0);

        std::ofstream out;
        if (!OUT_FILE_PATH.empty()) {
            out.open(OUT_FILE_PATH);
            if (out.is_open()) out << std::setprecision(set_precision);
        }

        // ---- Главный цикл по чанкам (порт hostLibrary.cu:1212-1692) ----
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            int blockSize = blockSize_setup;
            int gridSize  = (int)((cur_limiter + blockSize - 1) / blockSize);

            // 1. calculateDiscreteModelCUDA (dimension=2)
            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t sizeOfBlock_s             = (size_t)amountOfPointsInBlock;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            double h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            double maxValue_arg              = maxValue;
            bool   par_or_var_arg            = (par_or_var != 0);  // true=param, false=IC (runtime hint, kernel использует compile-time макрос)

            void* args_traj[] = {
                &nPts_int,
                &nPtsLimiter_int,
                &sizeOfBlock_s,
                &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s,
                &dimension_arg,
                &d_ranges,
                &h_arg,
                &d_indicesOfMutVars,
                &d_initialConditions,
                &amountOfIC_int,
                &d_values,
                &amountOfValues_int,
                &amountOfIterations_arg,
                &preScaller_int,
                &writableVar_int,
                &maxValue_arg,
                &d_data,
                &d_amountOfPeaks,
                &par_or_var_arg
            };
            unsigned int shared_traj = (unsigned int)((amountOfInitialConditions + amountOfValues) * sizeof(double) * blockSize);
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_traj,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          shared_traj, nullptr, args_traj, nullptr),
                           "cuLaunchKernel(bif2d traj)");
            BIF2D_CHECK(cudaDeviceSynchronize(), "sync after traj");

            // 2. peakFinderCUDA — d_data передаётся и как data, и как outPeaks (in-place)
            double timeStep_arg = h * (double)preScaller;
            void* args_peak[] = {
                &d_data,
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_data,        // outPeaks — in-place поверх траектории
                &d_intervals,
                &timeStep_arg
            };
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_peak,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, args_peak, nullptr),
                           "cuLaunchKernel(bif2d peak)");
            BIF2D_CHECK(cudaDeviceSynchronize(), "sync after peak");

            // 3. dbscanCUDA
            double eps_arg = eps_dbscan;
            void* args_dbscan[] = {
                &d_data,
                &sizeOfBlock_s,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_intervals,
                &d_helpfulArray,
                &eps_arg,
                &d_dbscanResult
            };
            BIF2D_CHECK_CU(cuLaunchKernel(cached_bif2d.kernel_dbscan,
                                          gridSize, 1, 1, blockSize, 1, 1,
                                          0, nullptr, args_dbscan, nullptr),
                           "cuLaunchKernel(bif2d dbscan)");
            BIF2D_CHECK(cudaDeviceSynchronize(), "sync after dbscan");

            // D2H: только d_dbscanResult (число кластеров = период).
            BIF2D_CHECK(cudaMemcpy(h_dbscanResult.data(), d_dbscanResult,
                                   cur_limiter * sizeof(int), cudaMemcpyDeviceToHost),
                        "memcpy h_dbscanResult");

            for (size_t k = 0; k < cur_limiter; ++k) {
                size_t kernel_idx = originalNPtsLimiter * iter + k;
                int    period     = h_dbscanResult[k];
                size_t out_idx;
                if (swap_xy) {
                    size_t kix = kernel_idx % (size_t)nPts;
                    size_t kiy = kernel_idx / (size_t)nPts;
                    out_idx = kix * (size_t)nPts + kiy;
                } else {
                    out_idx = kernel_idx;
                }
                double v = (period < 0) ? -1.0 : (double)period;
                res.values[out_idx] = v;
                res.flags[out_idx]  = (period < 0) ? -1 : 1;
                if (out.is_open()) out << period << ((k + 1 < cur_limiter) ? ", " : "\n");
            }
        }
        if (out.is_open()) out.close();

        // Авто-нормализация colormap.
        double vmin =  std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < total_cells; ++k) {
            if (res.flags[k] < 0) continue;
            double v = res.values[k];
            if (!std::isfinite(v) || v < 0.0) continue;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        res.min_val = std::isfinite(vmin) ? vmin : 0.0;
        res.max_val = std::isfinite(vmax) ? vmax : 0.0;

        cleanup();
        #undef BIF2D_CHECK
        #undef BIF2D_CHECK_CU
        res.ok = true;
        return res;
    }
};

// ---------------------------------------------------------------------------

ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
}

Bifurcation2DResult ParametricEngine::run_bifurcation_2d(const Bifurcation2DRequest& req) {
    return impl_->run_bif2d(req);
}

LLE1DResult ParametricEngine::run_lle_1d(const LLE1DRequest& req) {
    return impl_->run_lle_1d(req);
}

LS1DResult ParametricEngine::run_ls_1d(const LS1DRequest& req) {
    return impl_->run_ls_1d(req);
}

LLE2DResult ParametricEngine::run_lle_2d(const LLE2DRequest& req) {
    return impl_->run_lle_2d(req);
}

LS2DResult ParametricEngine::run_ls_2d(const LS2DRequest& req) {
    return impl_->run_ls_2d(req);
}
