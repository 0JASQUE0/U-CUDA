#include "engine_impl.h"
using namespace ucuda::cuda_detail;


ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}


Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    Bifurcation1DResult res = impl_->run_bif1d(req);
    // Диапазон прогона для X точек в GUI — у всех путей, не только у continuation.
    res.param_lo = req.param_lo;
    res.param_hi = req.param_hi;
    return res;
}


Dft1DResult ParametricEngine::run_dft_1d(const Dft1DRequest& req) {
    return impl_->run_dft_1d(req);
}


Bifurcation2DResult ParametricEngine::run_bifurcation_2d(const Bifurcation2DRequest& req) {
    return impl_->run_bif2d(req);
}


LLE1DResult ParametricEngine::run_lle_1d(const LLE1DRequest& req) {
    return impl_->run_lle_1d(req);
}


LS1DResult ParametricEngine::run_ls_1d(const LS1DRequest& req) {
    LS1DResult res = impl_->run_ls_1d(req);
    trim_ls_spectrum(res, lyap_dim(req.krs_body, req.amountOfX));
    return res;
}


LLE2DResult ParametricEngine::run_lle_2d(const LLE2DRequest& req) {
    return impl_->run_lle_2d(req);
}


LS2DResult ParametricEngine::run_ls_2d(const LS2DRequest& req) {
    LS2DResult res = impl_->run_ls_2d(req);
    trim_ls_spectrum(res, lyap_dim(req.krs_body, req.amountOfX));
    return res;
}


BasinsResult ParametricEngine::run_basins(const BasinsRequest& req) {
    return impl_->run_basins(req);
}


BasinsReclusterResult ParametricEngine::run_basins_recluster(const BasinsReclusterRequest& req) {
    return impl_->run_basins_recluster(req);
}


PerfResult ParametricEngine::run_performance(const PerfRequest& req) {
    return impl_->run_performance(req);
}


OrderResult ParametricEngine::run_order(const OrderRequest& req) {
    return impl_->run_order(req);
}


StabilityResult ParametricEngine::run_stability(const StabilityRequest& req) {
    return impl_->run_stability(req);
}


FastSyncResult ParametricEngine::run_fastsync(const FastSyncRequest& req) {
    return impl_->run_fastsync(req);
}


void ParametricEngine::prewarm(const Bifurcation1DRequest& req) { impl_->prewarm_bif1d(req); }

void ParametricEngine::prewarm(const Bifurcation2DRequest& req) { impl_->prewarm_bif2d(req); }

void ParametricEngine::prewarm(const LLE1DRequest& req)         { impl_->prewarm_lle1d(req); }

void ParametricEngine::prewarm(const LLE2DRequest& req)         { impl_->prewarm_lle2d(req); }

void ParametricEngine::prewarm(const LS1DRequest& req)          { impl_->prewarm_ls1d(req); }

void ParametricEngine::prewarm(const LS2DRequest& req)          { impl_->prewarm_ls2d(req); }

void ParametricEngine::prewarm(const BasinsRequest& req)        { impl_->prewarm_basins(req); }

void ParametricEngine::prewarm(const SignalMetricsRequest& req) { impl_->prewarm_metrics(req); }

const char* signal_metric_name(int m) {
    switch (m) {
        case SIGM_MAX:               return "Max";
        case SIGM_MIN:               return "Min";
        case SIGM_RANGE:             return "Max - Min";
        case SIGM_MEAN:              return "Mean";
        case SIGM_MEAN_FREQ:         return "Mean freq";
        case SIGM_MEDIAN_FREQ:       return "Median freq";
        case SIGM_VARIANCE:   return "Variance";
        case SIGM_HJORTH_MOBILITY:   return "Mobility";
        case SIGM_HJORTH_COMPLEXITY: return "Complexity";
        case SIGM_INT_MAX:           return "Int. max";
        case SIGM_INT_MIN:           return "Int. min";
        case SIGM_INT_RANGE:         return "Int. max - min";
        case SIGM_INT_MEAN:          return "Int. mean";
        case SIGM_VOLUME:            return "Volume";
        default:                     return "?";
    }
}

const char* signal_metric_axis_label(int m) {
    switch (m) {
        case SIGM_MAX:               return "max(x)";
        case SIGM_MIN:               return "min(x)";
        case SIGM_RANGE:             return "max(x) - min(x)";
        case SIGM_MEAN:              return "mean(x)";
        case SIGM_MEAN_FREQ:         return "mean freq, 1/t";
        case SIGM_MEDIAN_FREQ:       return "median freq, 1/t";
        case SIGM_VARIANCE:   return "variance = var(x)";
        case SIGM_HJORTH_MOBILITY:   return "Hjorth mobility, rad/t";
        case SIGM_HJORTH_COMPLEXITY: return "Hjorth complexity";
        case SIGM_INT_MAX:           return "max(T), t";
        case SIGM_INT_MIN:           return "min(T), t";
        case SIGM_INT_RANGE:         return "max(T) - min(T), t";
        case SIGM_INT_MEAN:          return "mean(T), t";
        case SIGM_VOLUME:            return "volume = prod(max x_i - min x_i)";
        default:                     return "";
    }
}

bool signal_metrics_write_csv(const SignalMetricsResult& r, const std::string& path, std::string& err) {
    std::ofstream out(path);
    if (!out.is_open()) { err = "cannot open " + path; return false; }
    out << std::setprecision(15);

    const int n = r.n_pts;
    auto node = [n, &r](int i, double lo, double hi, bool lg) {
        if (r.continuation) return cont_sweep_value(i, n, lo, hi, r.continuation_reverse, lg);
        return lg ? getValueByIdx_log_local((size_t)i, n, lo, hi)
                  : getValueByIdx_local((size_t)i, n, lo, hi);
    };

    out << (r.dimension == 2 ? "x,y,regime" : "x,regime");
    for (int m : kSignalMetricDisplayOrder)
        if ((r.metric_mask >> m) & 1) out << "," << signal_metric_name(m);
    out << "\n";

    const size_t total = r.dimension == 2 ? (size_t)n * (size_t)n : (size_t)n;
    for (size_t k = 0; k < total; ++k) {
        const int ix = (int)(k % (size_t)n);
        const int iy = (int)(k / (size_t)n);
        out << node(ix, r.param_lo, r.param_hi, r.log_scale);
        if (r.dimension == 2) out << "," << node(iy, r.param_lo_2, r.param_hi_2, r.log_scale_2);
        out << "," << (k < r.flags.size() ? r.flags[k] : REGIME_UNBOUND);
        for (int m : kSignalMetricDisplayOrder) {
            if (!((r.metric_mask >> m) & 1)) continue;
            const double v = k < r.values[m].size() ? r.values[m][k]
                                                    : std::numeric_limits<double>::quiet_NaN();
            out << ",";
            if (std::isfinite(v)) out << v;
            else                  out << "NaN";
        }
        out << "\n";
    }
    if (!out.good()) { err = "write failed: " + path; return false; }
    return true;
}

namespace ucuda::cuda_detail {
const char* ptx_numb_type() { return sizeof(numb) == 8 ? "f64" : "f32"; }
}

namespace ucuda::cuda_detail {
double ptx_op_weight(const std::string& op) {
    std::vector<std::string> parts;
    size_t s = 0;
    for (size_t i = 0; i <= op.size(); ++i)
        if (i == op.size() || op[i] == '.') { parts.push_back(op.substr(s, i - s)); s = i + 1; }
    const std::string ty = ptx_numb_type();
    bool typed = false, approx = false;
    for (size_t i = 1; i < parts.size(); ++i) {
        if (parts[i] == ty)       typed  = true;
        if (parts[i] == "approx") approx = true;
    }
    if (!typed) return 0.0;
    const std::string& b = parts[0];
    if (b == "div" || b == "sqrt" || b == "rcp") return approx ? 1.0 : 8.0;
    if (b == "add" || b == "sub" || b == "mul" || b == "fma" || b == "mad" || b == "neg" ||
        b == "abs" || b == "min" || b == "max" || b == "setp" || b == "cvt" || b == "copysign" ||
        b == "testp" || b == "rsqrt" || b == "sin" || b == "cos" || b == "ex2" || b == "lg2" ||
        b == "tanh")
        return 1.0;
    return 0.0;
}
}

namespace ucuda::cuda_detail {
double ptx_function_cost(const std::string& ptx, const std::string& fn, bool& loops) {
    loops = false;
    size_t pos = 0;
    for (;;) {
        pos = ptx.find(fn + "(", pos);
        if (pos == std::string::npos) return 0.0;
        // Определение, а не объявление: за списком параметров идёт '{', а не ';'.
        const size_t rp = ptx.find(')', pos);
        if (rp == std::string::npos) return 0.0;
        size_t q = rp + 1;
        while (q < ptx.size() && std::isspace((unsigned char)ptx[q])) ++q;
        if (q < ptx.size() && ptx[q] == '{') { pos = q; break; }
        pos = rp;
    }

    double cost = 0.0;
    int depth = 0;
    std::set<std::string> labels;
    std::istringstream in(ptx.substr(pos));
    std::string line;
    while (std::getline(in, line)) {
        const size_t a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos) continue;
        std::string t = line.substr(a);
        const size_t cm = t.find("//");
        if (cm != std::string::npos) t.erase(cm);
        for (char c : t) {
            if (c == '{') ++depth;
            else if (c == '}') --depth;
        }
        if (depth == 0) break;
        if (t.empty() || t[0] == '{' || t[0] == '}' || t[0] == '.') continue;
        if (t[0] == '$') {
            const size_t c = t.find(':');
            if (c != std::string::npos) labels.insert(t.substr(0, c));
            continue;
        }
        if (t[0] == '@') {   // предикат: "@%p1 bra ..."
            const size_t sp = t.find_first_of(" \t");
            if (sp == std::string::npos) continue;
            const size_t nx = t.find_first_not_of(" \t", sp);
            if (nx == std::string::npos) continue;
            t.erase(0, nx);
        }
        const std::string op = t.substr(0, t.find_first_of(" \t;"));
        if (op.compare(0, 3, "bra") == 0) {
            const size_t l = t.find('$');
            if (l != std::string::npos) {
                const std::string tgt = t.substr(l, t.find_first_of(" \t;", l) - l);
                if (labels.count(tgt)) loops = true;
            }
            continue;
        }
        cost += ptx_op_weight(op);
    }
    return cost;
}
}

namespace ucuda::cuda_detail {
bool estimate_device(CUdevice& dev, int& cc_major, int& cc_minor, std::string& err) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { err = "cuInit: " + cu_err(r); return false; }
    r = cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) { err = "cuDeviceGet: " + cu_err(r); return false; }
    cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    return true;
}
}

namespace ucuda::cuda_detail {
std::string calib_source() {
    std::string src = "#define AMOUNTOFX 3\n#include \"configCUDA.h\"\n";
    const char* names[2]  = { "Cheap", "Rk4" };
    const char* bodies[2] = { kCalibBodyCheap, kCalibBody };
    for (int k = 0; k < 2; ++k) {
        src += std::string("__device__ __forceinline__ void ucuda_calib_step") + names[k]
             + "(numb* X, const numb* a, const numb h) {\n" + bodies[k] + "}\n"
             + "extern \"C\" __global__ void ucudaCalib" + names[k]
             + "(numb* out, const numb* ag, numb h, int steps, int n, numb maxValue) {\n"
               "    extern __shared__ numb sh[];\n"
               "    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
               "    if (i >= n) return;\n"
               "    numb* X = sh + threadIdx.x * 3;\n"
               "    numb* a = sh + blockDim.x * 3 + threadIdx.x * 4;\n"
               "    for (int j = 0; j < 4; ++j) a[j] = ag[j];\n"
               "    X[0] = (numb)1 + (numb)1e-3 * (numb)(i & 1023); X[1] = (numb)1; X[2] = (numb)1;\n"
               "    numb* data = out + (size_t)i * " + std::to_string(kCalibRing) + ";\n"
               "    int untilCheck = 0;\n"
               "    for (int s = 0; s < steps; ++s) {\n"
               "        data[s & " + std::to_string(kCalibRing - 1) + "] = X[0];\n"
               "        ucuda_calib_step" + names[k] + "(X, a, h);\n"
               "        if (--untilCheck < 0) {\n"
               "            untilCheck = 99;\n"
               "            const numb c = abs(X[0]) + abs(X[1]) + abs(X[2]);\n"
               "            if (isnan(c) || isinf(c) || c > maxValue) return;\n"
               "        }\n"
               "    }\n"
               "}\n";
    }
    return src;
}
}

bool krs_step_cost(const std::string& krs_body, int amountOfX, KrsStepCost& out, std::string& err) {
    out = KrsStepCost();
    CUdevice dev = 0;
    int cc_major = 0, cc_minor = 0;
    if (!estimate_device(dev, cc_major, cc_minor, err)) return false;
    std::string rerr;
    const std::string cfg = peak_config_defines() + read_text_file(exe_dir() + "\\kernels\\configCUDA.h", rerr);
    if (!rerr.empty()) { err = rerr; return false; }

    // Шаг со стадией предрасчёта (CD fast) считается без неё: в горячих циклах она идёт раз на
    // траекторию (krs_prep.h).
    KrsPrepSplit sp;
    const bool prep = krs_split_prep(krs_body, sp);
    const std::string fn = prep ? "krs_step_p" : "krs_step";
    const std::string src = "#define AMOUNTOFX " + std::to_string(amountOfX) + "\n"
                            "#include \"configCUDA.h\"\n"
                            "extern \"C\" __device__ void " + fn + "("
                          + (prep ? krs_step_p_fn_params() : "numb* X, const numb* a, const numb h")
                          + ") {\n" + (prep ? krs_step_p_fn_body(sp) : krs_body) + "\n}\n";
    std::string ptx;
    std::vector<std::string> ignored;
    if (!nvrtc_compile_ptx(cc_major, cc_minor, src, "krs_cost.cu", { cfg.c_str() }, { "configCUDA.h" },
                           {}, true, ptx, ignored, err))
        return false;
    out.ops = ptx_function_cost(ptx, fn, out.loops);
    if (out.ops <= 0.0) {
        err = std::string("no ") + ptx_numb_type() + " arithmetic found in the step";
        return false;
    }
    return true;
}

std::string gpu_calibration_path() { return exe_dir() + "\\gpu_calibration.txt"; }

bool gpu_device_info(std::string& name, int& sm_count, std::string& err) {
    CUdevice dev = 0;
    int cc_major = 0, cc_minor = 0;
    if (!estimate_device(dev, cc_major, cc_minor, err)) return false;
    char buf[256] = {};
    cuDeviceGetName(buf, (int)sizeof buf, dev);
    name = buf;
    sm_count = 0;
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    return sm_count > 0;
}

bool gpu_calibrate(GpuCalibration& out, std::string& err) {
    out = GpuCalibration();
    KrsStepCost cost[2];
    if (!krs_step_cost(kCalibBodyCheap, 3, cost[0], err)) return false;
    if (!krs_step_cost(kCalibBody,      3, cost[1], err)) return false;
    if (cost[1].ops <= cost[0].ops) { err = "calibration steps have equal cost"; return false; }
    CUdevice dev = 0;
    int cc_major = 0, cc_minor = 0;
    if (!estimate_device(dev, cc_major, cc_minor, err)) return false;
    if (!gpu_device_info(out.device, out.sm_count, err)) return false;
    out.numb_bytes = (int)sizeof(numb);
    out.calib_ops  = cost[1].ops;

    std::string rerr;
    const std::string cfg = peak_config_defines() + read_text_file(exe_dir() + "\\kernels\\configCUDA.h", rerr);
    if (!rerr.empty()) { err = rerr; return false; }
    std::string ptx;
    std::vector<std::string> ignored;
    if (!nvrtc_compile_ptx(cc_major, cc_minor, calib_source(), "calib.cu", { cfg.c_str() }, { "configCUDA.h" },
                           {}, false, ptx, ignored, err))
        return false;

    // Первичный контекст: тот же, что у cudart в GUI, — отдельный стоил бы сотни МБ видеопамяти.
    CUcontext ctx = nullptr;
    CUresult r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS) { err = "cuDevicePrimaryCtxRetain: " + cu_err(r); return false; }
    cuCtxPushCurrent(ctx);

    const int sm        = out.sm_count;
    const int kWarpsSat = 32;   // варпов на SM в насыщенном прогоне
    CUmodule    mod   = nullptr;
    CUfunction  fn[2] = { nullptr, nullptr };
    CUdeviceptr d_out = 0, d_a = 0;
    CUstream    st    = nullptr;
    CUevent     e0    = nullptr, e1 = nullptr;
    bool ok = false;
    do {
        if ((r = cuModuleLoadData(&mod, ptx.c_str())) != CUDA_SUCCESS) { err = "cuModuleLoadData(calib): " + cu_err(r); break; }
        if ((r = cuModuleGetFunction(&fn[0], mod, "ucudaCalibCheap")) != CUDA_SUCCESS ||
            (r = cuModuleGetFunction(&fn[1], mod, "ucudaCalibRk4")) != CUDA_SUCCESS) { err = "cuModuleGetFunction(calib): " + cu_err(r); break; }
        if ((r = cuMemAlloc(&d_out, (size_t)32 * sm * kWarpsSat * kCalibRing * sizeof(numb))) != CUDA_SUCCESS) { err = "cuMemAlloc(calib): " + cu_err(r); break; }
        if ((r = cuMemAlloc(&d_a, 4 * sizeof(numb))) != CUDA_SUCCESS) { err = "cuMemAlloc(calib): " + cu_err(r); break; }
        const numb av[4] = { (numb)0.5, (numb)10, (numb)28, (numb)8 / (numb)3 };
        if ((r = cuMemcpyHtoD(d_a, av, sizeof av)) != CUDA_SUCCESS) { err = "cuMemcpyHtoD(calib): " + cu_err(r); break; }
        if ((r = cuStreamCreate(&st, CU_STREAM_NON_BLOCKING)) != CUDA_SUCCESS) { err = "cuStreamCreate(calib): " + cu_err(r); break; }
        cuEventCreate(&e0, CU_EVENT_DEFAULT);
        cuEventCreate(&e1, CU_EVENT_DEFAULT);

        numb h = (numb)0.001;
        numb max_value = (numb)1e6;
        // Один прогон ядра k: warps варпов на SM (блоки по варпу), steps шагов; время в мс.
        auto run = [&](int k, int warps, int steps, double& ms) -> bool {
            int n = 32 * sm * warps;
            void* args[] = { &d_out, &d_a, &h, &steps, &n, &max_value };
            cuEventRecord(e0, st);
            CUresult rr = cuLaunchKernel(fn[k], (unsigned)(sm * warps), 1, 1, 32, 1, 1,
                                         (unsigned)(32 * 7 * sizeof(numb)), st, args, nullptr);
            if (rr == CUDA_SUCCESS) rr = cuEventRecord(e1, st);
            if (rr == CUDA_SUCCESS) rr = cuEventSynchronize(e1);
            if (rr != CUDA_SUCCESS) { err = "calibration launch: " + cu_err(rr); return false; }
            float f = 0.0f;
            cuEventElapsedTime(&f, e0, e1);
            ms = f;
            return true;
        };
        // Шагов на ~40 мс: короче — шум запуска, длиннее — риск сброса драйвером (TDR) на
        // медленной карте. Подбор заодно начинает поднимать частоты.
        auto fit_steps = [&](int k, int warps, int& steps) -> bool {
            steps = 256;
            double ms = 0.0;
            for (int it = 0; it < 16; ++it) {
                if (!run(k, warps, steps, ms)) return false;
                if (ms >= 10.0 || steps > (1 << 26)) break;
                steps *= 4;
            }
            steps = (std::max)(1, (int)(std::min)(1e9, steps * 40.0 / (std::max)(ms, 1e-3)));
            return true;
        };
        // [ядро][0 — насыщение, 1 — один варп на SM]
        int    steps[2][2] = {};
        double best[2][2]  = { { 1e300, 1e300 }, { 1e300, 1e300 } };
        const int warps[2] = { kWarpsSat, 1 };
        bool run_ok = true;
        for (int k = 0; k < 2 && run_ok; ++k)
            for (int m = 0; m < 2 && run_ok; ++m) run_ok = fit_steps(k, warps[m], steps[k][m]);
        if (!run_ok) break;

        // Прогрев: ~300 мс насыщенных запусков подряд, иначе карта не выходит из низкого
        // P-state и первый замер завышен в 2-3 раза.
        double ms = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        while (run_ok && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(300))
            run_ok = run(1, kWarpsSat, steps[1][0], ms);
        for (int rep = 0; rep < 5 && run_ok; ++rep)
            for (int k = 0; k < 2 && run_ok; ++k)
                for (int m = 0; m < 2 && run_ok; ++m) {
                    run_ok = run(k, warps[m], steps[k][m], ms);
                    best[k][m] = (std::min)(best[k][m], ms / steps[k][m]);   // мс на шаг
                }
        if (!run_ok) break;

        // Шаг стоит (ops + c0) * ns: наклон по двум шагам — цена операции, остаток — добавка c0.
        double ns[2], c0[2];
        for (int m = 0; m < 2; ++m) {
            const double slope = (best[1][m] - best[0][m]) / (cost[1].ops - cost[0].ops);
            if (slope > 0.0) {
                ns[m] = slope * 1e6;
                c0[m] = (std::max)(0.0, best[0][m] / slope - cost[0].ops);
            } else {   // шум перекрыл разницу: без добавки, по дорогому шагу
                ns[m] = best[1][m] * 1e6 / cost[1].ops;
                c0[m] = 0.0;
            }
        }
        out.ns_thr = ns[0] / kWarpsSat;
        out.c0_thr = c0[0];
        out.ns_lat = ns[1];
        out.c0_lat = c0[1];
        ok = out.ns_thr > 0.0 && out.ns_lat > 0.0;
        if (!ok) err = "calibration produced a zero time";
    } while (false);

    if (e0)    cuEventDestroy(e0);
    if (e1)    cuEventDestroy(e1);
    if (st)    cuStreamDestroy(st);
    if (d_out) cuMemFree(d_out);
    if (d_a)   cuMemFree(d_a);
    if (mod)   cuModuleUnload(mod);
    cuCtxPopCurrent(nullptr);
    cuDevicePrimaryCtxRelease(dev);
    return ok;
}
