#include "engine_impl.h"
using namespace ucuda::cuda_detail;

namespace ucuda::cuda_detail {
int launch_block_size(size_t sharedPerThread) {
    int b = get_gpu_block_size();
    while (b > kGpuBlockSizeMin && sharedPerThread * (size_t)b > 48u * 1024u)
        b /= 2;
    return b;
}
}

namespace ucuda::cuda_detail {
void add_profiling_nvrtc_opts(std::vector<const char*>& opts, std::string& storage) {
    char v[16];
    unsigned long n = env_value("UCUDA_NVRTC_LINEINFO", v, sizeof v);
    if (n > 0 && n < sizeof v && v[0] == '1') opts.push_back("--generate-line-info");
    n = env_value("UCUDA_NVRTC_MAXRREG", v, sizeof v);
    if (n > 0 && n < sizeof v) { storage = std::string("--maxrregcount=") + v; opts.push_back(storage.c_str()); }
}
}

namespace ucuda::cuda_detail {
bool nvrtc_prune_kernels() {
    char v[8];
    const unsigned long n = env_value("UCUDA_NVRTC_PRUNE", v, sizeof v);
    return !(n > 0 && n < sizeof v && v[0] == '0');
}
}

namespace ucuda::cuda_detail {
std::string prune_kernel_defines(const std::vector<const char*>& name_exprs, const std::string& src) {
    std::string d = "#define UCUDA_PRUNE_KERNELS\n";
    for (const char* e : name_exprs) { d += "#define UCUDA_KEEP_"; d += e; d += '\n'; }
    if (src.find("loopCalculateDiscreteModel_int(") != std::string::npos) d += "#define UCUDA_KEEP_LOOP\n";
    if (src.find("loopCalculateDiscreteModelPeaks_int(") != std::string::npos) d += "#define UCUDA_KEEP_PEAKS_LOOP\n";
    return d;
}
}

namespace ucuda::cuda_detail {
void dump_module_ptx(const char* src_name, const std::string& src, const std::string& ptx) {
    char dir[4096];
    const unsigned long n = env_value("UCUDA_NVRTC_PTX_DIR", dir, 4096);
    if (n == 0 || n >= 4096) return;
    static std::atomic<int> seq{0};
    const std::string base = std::string(dir, n) + UCUDA_PATH_SEP + src_name + "_" + std::to_string(seq++);
    for (const auto& [ext, text] : { std::pair<const char*, const std::string*>{ ".cu", &src }, { ".ptx", &ptx } }) {
        FILE* f = nullptr;
#ifdef _WIN32
        if (fopen_s(&f, (base + ext).c_str(), "wb") == 0 && f) { std::fwrite(text->data(), 1, text->size(), f); std::fclose(f); }
#else
        f = std::fopen((base + ext).c_str(), "wb");
        if (f) { std::fwrite(text->data(), 1, text->size(), f); std::fclose(f); }
#endif
    }
}
}

namespace ucuda::cuda_detail {
const char* nvrtc_fmad_opt() {
    return get_nvrtc_fmad() ? "--fmad=true" : "--fmad=false";
}
}

namespace ucuda::cuda_detail {
bool gpu_free_budget(double reserve, size_t& out_bytes) {
    size_t freeMemory = 0, totalMemory = 0;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess) return false;
    out_bytes = (size_t)((double)freeMemory * reserve);
    return true;
}
}

namespace ucuda::cuda_detail {
int par_or_var_2d(bool sweep_h_x, bool sweep_h_y, bool sweep_var_x, bool sweep_var_y) {
    if (sweep_h_x || sweep_h_y) return (sweep_h_x ? sweep_var_y : sweep_var_x) ? 0 : 1;
    if (sweep_var_x == sweep_var_y) return sweep_var_x ? 0 : 1;
    return 2;
}
}

namespace ucuda::cuda_detail {
int lyap_warm_blocks(double vt, double NT)
{
    return vt > 0 ? (int)ucuda_steps_per_block((numb)vt, (numb)NT) : 0;
}
}

namespace ucuda::cuda_detail {
bool cancel_requested(const std::shared_ptr<std::atomic<bool>>& c)
{
    return c && c->load(std::memory_order_relaxed);
}
}

namespace ucuda::cuda_detail {
void mark_cancelled_tail(std::vector<int>& flags, size_t j)
{
    for (size_t k = j; k < flags.size(); ++k) flags[k] = REGIME_CANCELLED;
}
}

namespace ucuda::cuda_detail {
void fill_nan_tail(std::vector<double>& v, size_t from)
{
    for (size_t k = from; k < v.size(); ++k) v[k] = std::numeric_limits<double>::quiet_NaN();
}
}

namespace ucuda::cuda_detail {
double lyap_value_from_kernel(double v, int& flag)
{
    if (v == (double)UCUDA_CANCELLED_VALUE) { flag = REGIME_CANCELLED; return std::numeric_limits<double>::quiet_NaN(); }
    if (v == 999.0 || v == -999.0)         { flag = REGIME_UNBOUND;   return std::numeric_limits<double>::quiet_NaN(); }
    flag = REGIME_OSCILLATION;
    return v;
}
}

namespace ucuda::cuda_detail {
size_t steps_from_time_size_t(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (size_t)s;
}
}

namespace ucuda::cuda_detail {
int steps_from_time_int(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (s >= 2147483647.0) ? 2147483647 : (int)s;
}
}

namespace ucuda::cuda_detail {
bool dae_z_axis(const std::string& body, int var) {
    const int nx = dae_nx_from_body(body);
    return nx > 0 && var >= nx;
}
}

namespace ucuda::cuda_detail {
void dae_point_start(const KrsCpuStep& step, const std::string& body, numb* x, const numb* a) {
    if (dae_nx_from_body(body) > 0) step.fn()(x, a, (numb)0);
}
}

namespace ucuda::cuda_detail {
double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) {
    return (double)ucuda_node_value_cont(j, nPts, (numb)lo, (numb)hi, log_scale, reverse);
}
}

namespace ucuda::cuda_detail {
std::vector<numb> to_numb(const std::vector<double>& v) {
    return std::vector<numb>(v.begin(), v.end());
}
}

namespace ucuda::cuda_detail {
void fill_bif1d_snapshot(data_export::Bif1DSnapshot& sn, const Bifurcation1DRequest& req,
                                bool continuation, bool on_cpu) {
    sn.values.assign(req.base_values.begin(), req.base_values.end());
    sn.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
    sn.tMax          = req.t_max;
    sn.transientTime = req.transient_time;
    sn.h             = req.h;
    sn.gpu_fmad      = get_nvrtc_fmad();
    sn.preScaller    = req.pre_scaller;
    sn.writableVar   = req.writable_var;
    sn.indexOfMutVar = req.sweep_over_var ? req.var_sweep_index : req.param_index;
    sn.range_lo      = req.param_lo;
    sn.range_hi      = req.param_hi;
    sn.log_scale     = req.log_scale;
    sn.continuation  = continuation;
    sn.continuation_reverse = continuation && req.continuation_reverse;
    sn.on_cpu        = on_cpu;
    sn.step_control  = req.adaptive.enabled ? req.adaptive.desc : std::string();
}
}

namespace ucuda::cuda_detail {
std::string exe_dir() { return platform::executable_dir(); }
}

namespace ucuda::cuda_detail {
std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "failed to open " + path; return {}; }
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
}

namespace ucuda::cuda_detail {
std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}
}

std::string hash_key(const std::string& krs_body, int amountOfX) {
    // peak_config_epoch(): knobs configCUDA.h уходят в NVRTC как #define, т.е. при их смене тот
    // же КРС даёт другой PTX. Без epoch в ключе кэши модулей отдали бы старый модуль, и настройка
    // не применилась бы до перезапуска приложения.
    // fmad — то же самое, но через опцию компиляции. Здесь в ключ идёт САМО значение, а не счётчик
    // поколений: флаг бинарный, поэтому возврат к прежнему положению переиспользует уже собранный
    // модуль вместо лишней перекомпиляции.
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" +
           std::to_string(amountOfX) + ":pk" + std::to_string(peak_config_epoch()) +
           ":fm" + (get_nvrtc_fmad() ? "1" : "0") +
           ":rdc" + (get_nvrtc_rdc() ? "1" : "0");
}

namespace ucuda::cuda_detail {
std::string lyap_nx_prefix(const std::string& krs_body) {
    const int nx = dae_nx_from_body(krs_body);
    return nx > 0 ? "#define UCUDA_LYAP_NX " + std::to_string(nx) + "\n" : std::string();
}
}

namespace ucuda::cuda_detail {
bool ad_is_bdf(const AdaptiveRequest& ad) {
    return ad.prefix.find("UCUDA_BDF_DRIVER") != std::string::npos;
}
}

namespace ucuda::cuda_detail {
std::string time_prefix(const std::string& krs_body) {
    return step_body_uses_time(krs_body) ? kNonautonomousDefine : std::string();
}
}

namespace ucuda::cuda_detail {
int lyap_dim(const std::string& krs_body, int N) {
    const int nx = dae_nx_from_body(krs_body);
    return (nx > 0 && nx < N) ? nx : N;
}
}

namespace ucuda::cuda_detail {
std::string peak_config_defines() {
    const PeakConfig c = get_peak_config();
    std::ostringstream o;
    o << std::setprecision(17);
    o << "// --- injected by parametric_engine (GUI Settings) ---\n"
      << "#define doCalculatePeaks "    << (c.do_calculate_peaks   ? 1 : 0) << "\n"
      << "#define doInterpolatePeaks "  << (c.do_interpolate_peaks ? 1 : 0) << "\n"
      << "#define eps_fixed_point ((numb)"     << c.eps_fixed_point     << ")\n"
      << "#define eps_peak_delta ((numb)"      << c.eps_peak_delta      << ")\n"
      << "#define eps_interPeak_delta ((numb)" << c.eps_interPeak_delta << ")\n"
      << "#define peak_threshold ((numb)"      << c.peak_threshold      << ")\n"
      << "#define max_amount_of_peaks "        << c.max_amount_of_peaks << "\n";
    return o.str();
}
}

namespace ucuda::cuda_detail {
std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}
}

namespace ucuda::cuda_detail {
double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    // Одна арифметика с ядром — ucuda_node_value (configCUDA.h). Раньше здесь стояло
    // lo + (hi-lo)*idx/(n-1): при idx = nPts-1 это не давало РОВНО hi, и CSV сообщал параметр, в
    // котором точка не считалась. Вырожденный nPts == 1 теперь тоже как в ядре (hi, а не lo).
    // Касты явные: при numb = float сетка обязана считаться во float — ровно так, как её увидит
    // ядро (ranges уходят туда через to_numb).
    return (double)ucuda_node_value((int)idx, nPts, (numb)lo, (numb)hi);
}
}

namespace ucuda::cuda_detail {
double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) {
    // См. getValueByIdx_local: общая формула вместо копии. Здесь копия ещё и
    // умножала до деления, а ядро (getValueByIdxLog) — наоборот.
    return (double)ucuda_node_value_log((int)idx, nPts, (numb)lo, (numb)hi);
}
}

namespace ucuda::cuda_detail {
void write_curve1d_csv(const LLE1DResult& r, const std::string& path) {
    if (!path.empty()) data_export::export_lle1d(r, path);
}
}

namespace ucuda::cuda_detail {
void write_curve1d_csv(const LS1DResult& r, const std::string& path, int K) {
    if (path.empty()) return;
    if (K >= r.n_exponents) { data_export::export_ls1d(r, path); return; }
    LS1DResult t = r;
    t.n_exponents = K;
    for (auto& v : t.spectrum) if ((int)v.size() > K) v.resize((size_t)K);
    data_export::export_ls1d(t, path);
}
}

namespace ucuda::cuda_detail {
void dae_project(const KrsCpuStep& step, const numb* x, numb* v, const numb* a, int k, int n) {
    if (k >= n) return;
    for (int i = k; i < n; ++i) v[i] = x[i];
    step.fn()(v, a, (numb)0);
}
}

namespace ucuda::cuda_detail {
void metrics_finish(SignalMetricsResult& res, const SignalMetricsRequest& req) {
    for (int m = 0; m < SIGM_COUNT; ++m) {
        double lo =  std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (double v : res.values[m]) {
            if (!std::isfinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        res.min_val[m] = std::isfinite(lo) ? lo : 0.0;
        res.max_val[m] = std::isfinite(hi) ? hi : 0.0;
    }
    if (!req.csv_output_path.empty()) {
        std::string e;
        if (!signal_metrics_write_csv(res, req.csv_output_path, e))
            std::fprintf(stderr, "[metrics] CSV: %s\n", e.c_str());
    }
    res.ok = true;
    res.cancelled = cancel_requested(req.cancel);   // ok + cancelled = частичный результат (Stop)
}
}

int EngineContext::progress_stride_for(size_t stepsPerPoint) {
        int s = (int)(stepsPerPoint / 64);
        s -= s % kLoopCheckBlock;
        if (s < kLoopCheckBlock) s = kLoopCheckBlock;
        return s;
    }

ParametricEngine::Impl::~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            drain_pool(pool_bif1d);
            drain_pool(pool_lle);
            drain_pool(pool_lle_2d);
            drain_pool(pool_ls);
            drain_pool(pool_ls_2d);
            drain_pool(pool_cont);
            drain_pool(pool_lle_cont);
            drain_pool(pool_ls_cont);
            drain_pool(pool_dft_cont);
            drain_pool(pool_dft_hsweep);
            drain_pool(pool_bif2d);
            drain_pool(pool_basins);
            drain_pool(pool_ad);
            drain_pool(pool_ad_cont);
            drain_pool(pool_lyap_ad);
            drain_pool(pool_lyap_ad_cont);
            drain_pool(pool_metrics_ad);
            drain_pool(pool_fs_attr);
            drain_pool(pool_fs_grid);
            drain_pool(pool_order);
            drain_pool(pool_network);
            drain_pool(pool_neural);
            drain_pool(pool_metrics);
            cuCtxDestroy(context);
        }
    }

bool EngineContext::ensure_init(std::string& err) {
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

bool EngineSources::load_sources(std::string& err) {
        // Перечитывать файлы с диска надо один раз, а вот текст configCUDA.h
        // пересобирать — на каждую смену peak-настроек (см. peak_config_defines).
        const uint64_t ep = peak_config_epoch();
        if (srcs_loaded && srcs_peak_epoch == ep) return true;
        if (srcs_loaded) {
            src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
            srcs_peak_epoch  = ep;
            return true;
        }
        std::string root = exe_dir() + UCUDA_PATH_SEP "kernels" UCUDA_PATH_SEP;
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu",      e); if (!e.empty()) { err = e; return false; }
        src_template_cont     = read_text_file(root + "bifurcation1d_cont.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_template_lle_cont = read_text_file(root + "lle1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_ls_cont  = read_text_file(root + "ls1d_cont.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_dft_cont = read_text_file(root + "dft1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_dft_hsweep = read_text_file(root + "dft1d_hsweep.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_lle      = read_text_file(root + "lle1d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_lle_2d   = read_text_file(root + "lle2d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_ls       = read_text_file(root + "ls1d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_ls_2d    = read_text_file(root + "ls2d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_bif2d    = read_text_file(root + "bifurcation2d.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_basins   = read_text_file(root + "basins.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_fs_attr  = read_text_file(root + "fastsync_attr.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_fs_grid  = read_text_file(root + "fastsync_grid.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_order    = read_text_file(root + "order.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_network  = read_text_file(root + "network.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_neural   = read_text_file(root + "neural.template.cu",           e); if (!e.empty()) { err = e; return false; }
        src_template_metrics  = read_text_file(root + "signal_metrics.template.cu",   e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_ucuda_adaptive_cuh = read_text_file(root + "ucuda_adaptive.cuh",       e); if (!e.empty()) { err = e; return false; }
        src_ucuda_bdf_cuh     = read_text_file(root + "ucuda_bdf.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_adaptive_part     = read_text_file(root + "adaptive_part.cu",          e); if (!e.empty()) { err = e; return false; }
        src_metrics_adaptive_part = read_text_file(root + "metrics_adaptive_part.cu", e); if (!e.empty()) { err = e; return false; }
        src_lyap_adaptive_part = read_text_file(root + "lyapunov_adaptive_part.cu", e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h_raw  = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
        srcs_peak_epoch  = ep;
        srcs_loaded = true;
        return true;
    }

EngineSources::SrcSnapshot EngineSources::snapshot_sources(const std::string& tmpl) const {
        return { tmpl, src_cudaLibrary_cu, src_cudaLibrary_cuh, src_cudaMacros_cuh, src_configCUDA_h,
                 src_ucuda_adaptive_cuh, src_ucuda_bdf_cuh };
    }

const EngineModuleBuilder::KrsFn* EngineModuleBuilder::krs_fn(const std::string& ph) {
        for (const KrsFn& f : kKrsFns) if (ph == f.ph) return &f;
        return nullptr;
    }

const EngineModuleBuilder::KrsFn* EngineModuleBuilder::krs_fn(const std::string& ph, bool nonauto) {
        if (nonauto)
            for (const KrsFn& f : kKrsFnsT) if (ph == f.ph) return &f;
        return krs_fn(ph);
    }

bool EngineModuleBuilder::compile_ptx(const std::string& src, const char* src_name,
                     const std::vector<const char*>& headers_src,
                     const std::vector<const char*>& headers_name,
                     const std::vector<const char*>& name_exprs,
                     bool rdc, std::string& out_ptx, std::vector<std::string>& lowered,
                     std::string& err) {
        return nvrtc_compile_ptx(cc_major, cc_minor, src, src_name, headers_src, headers_name,
                                 name_exprs, rdc, out_ptx, lowered, err);
    }

bool EngineModuleBuilder::lib_ptx_for(const SrcSnapshot& snap, const char* src_name, const std::string& lib_key,
                     const std::vector<std::pair<std::string, std::string>>& subs,
                     const std::vector<const char*>& name_exprs,
                     CachedLibPtx& out, std::string& err) {
        std::unique_lock<std::mutex> lk(ptx_mu);
        for (;;) {
            if (pool_lib_ptx.take(lib_key, out)) return true;
            if (lib_in_flight.insert(lib_key).second) break;
            lib_ptx_done.wait(lk);
        }
        lk.unlock();

        std::string src = snap.tmpl;
        for (const auto& sub : subs) {
            if (krs_fn(sub.first)) continue;
            src = replace_all(src, sub.first, sub.second);
        }
        std::string decls;
        const bool nonauto = snap.tmpl.find(kNonautonomousDefine) != std::string::npos;
        for (const KrsFn& f0 : kKrsFns) {
            const KrsFn& f = *krs_fn(f0.ph, nonauto);
            if (src.find(f.ph) == std::string::npos) continue;
            src = replace_all(src, f.ph, f.call);
            decls += std::string("\nextern \"C\" __device__ void ") + f.name + "(" + f.params + ");";
        }
        src = replace_all(src, "#include \"cudaLibrary.cuh\"", "#include \"cudaLibrary.cuh\"" + decls);

        const std::vector<const char*> hs = { snap.lib_cu.c_str(), snap.lib_cuh.c_str(),
                                              snap.macros_cuh.c_str(), snap.config_h.c_str(),
                                              snap.adaptive_cuh.c_str(), snap.bdf_cuh.c_str() };
        const std::vector<const char*> hn = { "cudaLibrary.cu", "cudaLibrary.cuh",
                                              "cudaMacros.cuh", "configCUDA.h", "ucuda_adaptive.cuh", "ucuda_bdf.cuh" };
        CachedLibPtx fresh;
        fresh.key = lib_key;
        const bool ok = compile_ptx(src, src_name, hs, hn, name_exprs, true,
                                    fresh.ptx, fresh.lowered, err);

        lk.lock();
        lib_in_flight.erase(lib_key);
        lib_ptx_done.notify_all();
        if (!ok) return false;
        std::vector<CachedLibPtx> evicted;
        pool_lib_ptx.insert(fresh, std::string(), evicted);
        out = fresh;
        return true;
    }

bool EngineModuleBuilder::krs_ptx_for(const SrcSnapshot& snap,
                     const std::vector<std::pair<const KrsFn*, std::string>>& fns,
                     const std::string& amount_of_x, const std::string& krs_key,
                     std::string& out_ptx, std::string& err) {
        {
            std::lock_guard<std::mutex> lk(ptx_mu);
            CachedKrsPtx hit;
            if (pool_krs_ptx.take(krs_key, hit)) { out_ptx = hit.ptx; return true; }
        }

        std::string src = "#define AMOUNTOFX " + amount_of_x + "\n"
                          "#include \"configCUDA.h\"\n";
        for (const auto& fb : fns)
            src += std::string("extern \"C\" __device__ void ") + fb.first->name + "(" + fb.first->params
                 + ") {\n" + fb.second + "\n}\n";
        const std::vector<const char*> hs = { snap.config_h.c_str() };
        const std::vector<const char*> hn = { "configCUDA.h" };
        std::vector<std::string> ignored;
        if (!compile_ptx(src, "krs.cu", hs, hn, {}, true, out_ptx, ignored, err)) return false;

        std::lock_guard<std::mutex> lk(ptx_mu);
        CachedKrsPtx fresh;
        fresh.key = krs_key;
        fresh.ptx = out_ptx;
        std::vector<CachedKrsPtx> evicted;
        pool_krs_ptx.insert(fresh, std::string(), evicted);
        return true;
    }

bool EngineModuleBuilder::link_and_load(const std::string& lib_ptx, const std::string& krs_ptx,
                       CUmodule& out_module, std::string& err) {
        CUlinkState st = nullptr;
        // UCUDA_NVRTC_LINEINFO=1 (см. build_module): строки исходника переживают и линковку.
        // UCUDA_NVRTC_MAXRREG — тот же потолок регистров: распределяет их именно линковщик.
        CUjit_option jopt[2];
        void*        jval[2];
        unsigned     nj = 0;
        char li[16];
        unsigned long lin = env_value("UCUDA_NVRTC_LINEINFO", li, sizeof li);
        if (lin > 0 && lin < sizeof li && li[0] == '1') { jopt[nj] = CU_JIT_GENERATE_LINE_INFO; jval[nj++] = (void*)(size_t)1; }
        lin = env_value("UCUDA_NVRTC_MAXRREG", li, sizeof li);
        if (lin > 0 && lin < sizeof li) { jopt[nj] = CU_JIT_MAX_REGISTERS; jval[nj++] = (void*)(size_t)std::atoi(li); }
        CUresult r = cuLinkCreate(nj, nj ? jopt : nullptr, nj ? jval : nullptr, &st);
        if (r != CUDA_SUCCESS) { err = "cuLinkCreate: " + cu_err(r); return false; }
        auto fail = [&](const std::string& what, CUresult rr) {
            err = what + ": " + cu_err(rr);
            cuLinkDestroy(st);
            return false;
        };
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)lib_ptx.data(), lib_ptx.size() + 1,
                          "lib", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(lib)", r);
        r = cuLinkAddData(st, CU_JIT_INPUT_PTX, (void*)krs_ptx.data(), krs_ptx.size() + 1,
                          "krs", 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) return fail("cuLinkAddData(krs)", r);

        void* cubin = nullptr; size_t cubin_size = 0;
        r = cuLinkComplete(st, &cubin, &cubin_size);
        if (r != CUDA_SUCCESS) return fail("cuLinkComplete", r);

        CUmodule mod = nullptr;
        r = cuModuleLoadData(&mod, cubin);
        cuLinkDestroy(st);
        if (r != CUDA_SUCCESS) { err = "cuModuleLoadData(linked): " + cu_err(r); return false; }
        out_module = mod;
        return true;
    }

bool EngineModuleBuilder::build_module(const SrcSnapshot& snap_in,
                      const char* src_name,
                      const std::vector<std::pair<std::string, std::string>>& subs,
                      const std::vector<const char*>& name_exprs,
                      CUmodule& out_module,
                      std::vector<std::string>& lowered,
                      std::string& err) {
        // Явное время t (этап 2): номер шага проведён через ядра БД, LLE/LS, Metrics, бассейнов и
        // DFT на фиксированном шаге — их модули приходят с UCUDA_NONAUTONOMOUS (time_prefix).
        // Остальным (адаптивные свипы, FastSync, Order, Network) — понятный отказ, а не
        // "identifier ucuda_t is undefined" от NVRTC.
        const bool nonauto = snap_in.tmpl.find(kNonautonomousDefine) != std::string::npos;
        for (const auto& sub : subs)
            if ((sub.first == "{{KRS_BODY}}" || sub.first == "{{KRS_RHS_BODY}}")
                && step_body_uses_time(sub.second) && !nonauto) {
                err = "explicit time t: FastSync and Network do not support it yet";
                return false;
            }
        // Стадия предрасчёта (krs_prep.h): у шага с блоком констант ("CD fast") горячие
        // циклы cudaLibrary.cu зовут ucudaKrsPrep / calculateDiscreteModelP. Их определения
        // встают в шаблон: одной единицей трансляции — перед #include "cudaLibrary.cu"
        // (сам calculateDiscreteModel остаётся самодостаточным для прочих вызовов), при
        // раздельной сборке — обёртками над krs_prep / krs_step_p модуля КРС сразу после
        // #include "ucuda/cuda/cudaLibrary.cuh". Шаблон меняется, значит меняется и ключ библиотеки.
        KrsPrepSplit prep_sp;
        bool has_prep = false;
        for (const auto& sub : subs)
            if (sub.first == "{{KRS_BODY}}") has_prep = krs_split_prep(sub.second, prep_sp);
        SrcSnapshot snap = snap_in;
        // Раздельная сборка: библиотека без КРС компилируется один раз на (шаблон, размерность,
        // настройки) и переиспользуется, шаг живёт в своём модуле, склейка — cuLink. Замерено на
        // bifurcation2d + implicit midpoint: смена КРС 0.25 с против ~15 с одной единицей
        // трансляции. Цена — вызов вместо инлайна: на RTX 2060 SUPER от -1% до +7% времени счёта,
        // результаты побитово те же (FP64 идёт 1/32 скорости, накладные прячутся в её тени).
        if (get_nvrtc_rdc()) {
            std::vector<std::pair<const KrsFn*, std::string>> krs_fns;
            std::string krs_all, amount_of_x = "3";
            std::string lib_key = src_name;
            static const KrsFn kPrepFn  = { "", "krs_prep",   krs_prep_fn_params(),   "" };
            static const KrsFn kStepPFn = { "", "krs_step_p", krs_step_p_fn_params(), "" };
            if (has_prep) {
                krs_fns.emplace_back(&kPrepFn,  krs_prep_fn_body(prep_sp));
                krs_fns.emplace_back(&kStepPFn, krs_step_p_fn_body(prep_sp));
                // Тела обеих выводятся из KRS_BODY, который ниже и так входит в ключ.
                if (!krs_insert_after_line(snap.tmpl, "#include \"cudaLibrary.cuh\"",
                                           krs_prep_lib_defs(prep_sp))) {
                    // Шаблон без этой строки: шаг остаётся самодостаточным, циклы
                    // берут пустой предрасчёт из cudaLibrary.cu.
                    krs_fns.clear();
                    snap.tmpl = snap_in.tmpl;
                }
            }
            for (const auto& sub : subs) {
                if (const KrsFn* f = krs_fn(sub.first, nonauto)) {
                    krs_fns.emplace_back(f, sub.second);
                    krs_all += f->name; krs_all += '\x1f'; krs_all += sub.second; krs_all += '\x1f';
                    continue;
                }
                if (sub.first == "{{AMOUNT_OF_X}}") amount_of_x = sub.second;
                lib_key += ''; lib_key += sub.first; lib_key += '='; lib_key += sub.second;
            }
            lib_key += get_nvrtc_fmad() ? "|fm1" : "|fm0";
            lib_key += "|pk" + std::to_string(peak_config_epoch());
            // Шаблон и запрошенные ядра — тоже часть ключа. Имя исходника их не определяет:
            // FastSync собирает "fastsync.cu" и из fs_attr, и из fs_grid с одинаковыми подстановками,
            // и On Grid получал библиотеку On Attractor — чужие ядра (картинка-мусор, illegal
            // address) или lowered короче name_exprs (чтение за концом в cuModuleGetFunction).
            lib_key += "|t" + std::to_string(std::hash<std::string>{}(snap.tmpl));
            for (const char* e : name_exprs) { lib_key += '|'; lib_key += e; }

            std::string krs_err;
            std::string krs_ptx;
            const std::string krs_key = std::to_string(std::hash<std::string>{}(krs_all))
                                      + ":" + amount_of_x
                                      + (get_nvrtc_fmad() ? ":fm1" : ":fm0")
                                      + ":pk" + std::to_string(peak_config_epoch());
            if (!krs_ptx_for(snap, krs_fns, amount_of_x, krs_key, krs_ptx, krs_err)) {
                // Не компилируется САМ шаг — это ошибка пользователя, и монолитный путь выдал бы
                // ту же самую. Отдаём как есть, без второго захода на те же грабли.
                err = krs_err;
                return false;
            }

            CachedLibPtx lib;
            std::string lib_err;
            if (lib_ptx_for(snap, src_name, lib_key, subs, name_exprs, lib, lib_err)
                && lib.lowered.size() == name_exprs.size()
                && link_and_load(lib.ptx, krs_ptx, out_module, lib_err)) {
                lowered = lib.lowered;
                return true;
            }
            // Библиотека или линковка — не пользовательская ошибка: собираем по-старому, одной
            // единицей трансляции. Медленно, но работает везде, где работало раньше.
            std::fprintf(stderr, "[nvrtc] rdc split failed (%s), falling back to one TU\n",
                         lib_err.c_str());
        }

        // snap_in, а не snap: после неудачной раздельной сборки в snap.tmpl остались бы
        // её обёртки над krs_prep / krs_step_p.
        std::string src = snap_in.tmpl;
        for (const auto& sub : subs) src = replace_all(src, sub.first, sub.second);
        if (has_prep)
            krs_insert_before_line(src, "#include \"cudaLibrary.cu\"", krs_prep_inline_defs(prep_sp));
        if (nvrtc_prune_kernels())
            krs_insert_before_line(src, "#include \"cudaLibrary.cu\"", prune_kernel_defines(name_exprs, src));

        // Виртуальные заголовки для NVRTC. curand_kernel.h-stub НЕ нужен:
        // inline-stub в шаблонах + `#define CURAND_KERNEL_H_` блокируют как
        // реальный header (по -I path), так и любой повторный inject.
        const char* header_sources[] = {
            snap.lib_cu.c_str(),
            snap.lib_cuh.c_str(),
            snap.macros_cuh.c_str(),
            snap.config_h.c_str(),
            snap.adaptive_cuh.c_str(),
            snap.bdf_cuh.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
            "ucuda_adaptive.cuh",
            "ucuda_bdf.cuh",
        };
        constexpr int n_headers = 6;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), src_name,
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) {
            err = std::string("nvrtcCreateProgram(") + src_name + "): " + nvrtcGetErrorString(nr);
            return false;
        }

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        for (const char* sym : name_exprs) nvrtcAddNameExpression(prog, sym);

        char arch[64];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h,
        // curand_kernel.h и т.п.). Берём CUDA_PATH из окружения — его проставляет
        // CUDA Toolkit installer.
        std::string cuda_include_opt;
        {
            char buf[4096];
#ifdef _WIN32
            unsigned long nlen = env_value("CUDA_PATH", buf, 4096);
            if (nlen > 0 && nlen < 4096)
                cuda_include_opt = std::string("-I") + std::string(buf, nlen) + "\\include";
#else
            // POSIX: CUDA_PATH is optional; fall back to the toolkit found by CMake.
            (void)buf;
            const std::string inc = platform::cuda_include_dir();
            if (!inc.empty()) cuda_include_opt = "-I" + inc;
#endif
        }
        if (cuda_include_opt.empty()) {
            err = "the CUDA_PATH environment variable is not set - NVRTC will not find math_constants.h "
                  "(install the CUDA Toolkit or set CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        std::string std_opt = "--std=c++17";
        std::vector<const char*> opts = { arch, std_opt.c_str(), "-default-device",
                                          cuda_include_opt.c_str(), nvrtc_fmad_opt() };
        std::string dev_opt_storage;
        add_profiling_nvrtc_opts(opts, dev_opt_storage);

        nr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = std::string("NVRTC compile failed (") + src_name + "):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);
        dump_module_ptx(src_name, src, ptx);

        // Mangled-имена копируем в свои строки ДО nvrtcDestroyProgram — после
        // destroy указатели становятся невалидны.
        lowered.clear();
        lowered.reserve(name_exprs.size());
        for (const char* sym : name_exprs) {
            const char* lo = nullptr;
            nvrtcGetLoweredName(prog, sym, &lo);
            lowered.push_back(lo ? lo : sym);
        }
        nvrtcDestroyProgram(&prog);

        // Грузим во временную переменную: при неудаче у вызывающего слот кэша
        // остаётся нетронутым, и release_*_module() не получит мусорный handle.
        CUmodule mod = nullptr;
        // UCUDA_NVRTC_MAXRREG (add_profiling_nvrtc_opts): регистры распределяет JIT драйвера при
        // загрузке PTX, поэтому потолок — и сюда.
        CUjit_option mopt[1] = { CU_JIT_MAX_REGISTERS };
        void*        mval[1] = { nullptr };
        unsigned     nm = 0;
        {
            char v[16];
            const unsigned long n = env_value("UCUDA_NVRTC_MAXRREG", v, sizeof v);
            if (n > 0 && n < sizeof v) { mval[0] = (void*)(size_t)std::atoi(v); nm = 1; }
        }
        CUresult r = cuModuleLoadDataEx(&mod, ptx.c_str(), nm, nm ? mopt : nullptr, nm ? mval : nullptr);
        if (r != CUDA_SUCCESS) {
            err = std::string("cuModuleLoadDataEx(") + src_name + "): " + cu_err(r);
            return false;
        }
        out_module = mod;
        return true;
    }

bool EngineModuleBuilder::module_fn(CUmodule mod, const std::string& name, CUfunction& out, std::string& err) {
        CUresult r = cuModuleGetFunction(&out, mod, name.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + name + "): " + cu_err(r);
            return false;
        }
        return true;
    }

bool ParametricEngine::Impl::compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err, bool activate ) {
        // Если worker-thread унаследовал чужой контекст (NvrtcEngine, например),
        // компиляция и загрузка модуля прицепят символы не в тот контекст. Жёстко
        // выставляем наш перед NVRTC/CU-вызовами.
        cuCtxSetCurrent(context);
        // par_or_var в kernel'е cudaLibrary.cu — compile-time макрос. Поэтому
        // включаем его в hash-key: param-sweep и IC-sweep кешируются отдельно.
        std::string key = hash_key(krs_body, amountOfX) + ":pov" + std::to_string(par_or_var);
        return compile_into(pool_bif1d, key, cached, activate, [&](CachedModule& fresh) {
            // DFT_custom уже присутствует в этом же модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) — регистрируем его тоже, чтобы run_dft_1d
            // мог переиспользовать этот кэш без отдельной компиляции.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template), "bifurcation1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelCUDA", "peakFinderCUDA", "DFT_custom_pf",
                                "calculateDiscreteModelPeaksCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_traj,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_peak,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_dft,   err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_fused, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        // Узлы шага — свой модуль без плотного выхода (UCUDA_AD_NO_DENSE): состояние нити
        // легче на D, Xp, Fp, а стадии живут только внутри попытки. Своё имя исходника — от
        // него зависит ключ PTX библиотеки при раздельной сборке.
        const bool nodes = ad_nodes_module(ad);
        const std::string key = hash_key(bodies, amountOfX) + ":ad:pov" + std::to_string(par_or_var)
                              + (nodes ? ":nodes" : "");
        return compile_into(pool_ad, key, cached_ad, activate, [&](CachedAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = (nodes ? std::string(kAdNodesModulePrefix) : std::string())
                                   + src_template_bif2d + "\n" + src_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), nodes ? "adaptive_sweep_nodes.cu" : "adaptive_sweep.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelPeaksAdCUDA", "calculateDiscreteModelAvgPeaksAdCUDA",
                                "dbscanCUDA",
                                "dbscanWarpCUDA", "ucudaMaxPeaksCUDA", "dbscanWarp2DCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            CUfunction* slots[] = { &fresh.kernel_peaks, &fresh.kernel_avg, &fresh.kernel_dbscan,
                                    &fresh.kernel_dbscan_warp, &fresh.kernel_max_peaks, &fresh.kernel_dbscan_warp2d };
            for (int i = 0; i < 6; ++i)
                if (!module_fn(mod, mg[(size_t)i], *slots[i], err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":ad_cont";
        return compile_into(pool_ad_cont, key, cached_ad_cont, activate, [&](CachedSimpleContModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_CONT_KERNEL 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "adaptive_cont.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      "1" },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelPeaksAdContCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lyap_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":lyap_ad:pov" + std::to_string(par_or_var);
        return compile_into(pool_lyap_ad, key, cached_lyap_ad, activate, [&](CachedLyapAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_NO_DENSE 1\n"
                                     "#define UCUDA_AD_LYAPUNOV 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part + "\n" + src_lyap_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "lyapunov_adaptive.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad_is_bdf(ad) ? ad.dprep : std::string() },
                                { "{{KRS_DEVAL_BODY}}",  "" },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "lyapunovLLEAdCUDA", "lyapunovLSAdCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err) || !module_fn(mod, mg[1], fresh.kernel_ls, err)) {
                cuModuleUnload(mod); return false;
            }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lyap_ad_cont_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                        std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        const std::string key = hash_key(bodies, amountOfX) + ":lyap_ad_cont";
        return compile_into(pool_lyap_ad_cont, key, cached_lyap_ad_cont, activate, [&](CachedLyapAdModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n#define UCUDA_AD_NO_DENSE 1\n"
                                     "#define UCUDA_AD_LYAPUNOV 1\n#define UCUDA_LYAP_AD_CONT_ONLY 1\n"
                                   + src_template_bif2d + "\n" + src_adaptive_part + "\n" + src_lyap_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), "lyapunov_adaptive_cont.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      "1" },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad_is_bdf(ad) ? ad.dprep : std::string() },
                                { "{{KRS_DEVAL_BODY}}",  "" },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "lyapunovLLEAdContCUDA", "lyapunovLSAdContCUDA" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err) || !module_fn(mod, mg[1], fresh.kernel_ls, err)) {
                cuModuleUnload(mod); return false;
            }
            return true;
        }, err);
    }

std::string ParametricEngine::Impl::ad_axis_range_error(int kind, double lo, double hi) {
        if ((kind == kAdAxisRtol || kind == kAdAxisAtol || kind == kAdAxisTol) && !(lo > 0.0 && hi > 0.0))
            return "lo/hi must be > 0 on the rtol / atol axis";
        return {};
    }

std::string ParametricEngine::Impl::ad_axes_overlap_error(int kx, int ky) {
        auto fields = [](int k) -> unsigned {
            if (k == kAdAxisRtol) return 1u;
            if (k == kAdAxisAtol) return 2u;
            if (k == kAdAxisTol)  return 3u;
            if (k >= kAdAxisCtrl && k < kAdAxisCtrl + UCUDA_CTL_NPAR) return 4u << (k - kAdAxisCtrl);
            return 0u;
        };
        if (fields(kx) & fields(ky))
            return "both axes sweep the same step setting (rtol / atol / tol / controller parameter)";
        return {};
    }

bool ParametricEngine::Impl::ad_nodes_module(const AdaptiveRequest& ad) {
        return ad.raw_nodes && ad.axis_kind[0] == kAdAxisSystem && ad.axis_kind[1] == kAdAxisSystem;
    }

void* ParametricEngine::Impl::ad_param_arg(const AdaptiveRequest& ad, UcudaAdaptParams** d_ptr) {
        return ad_nodes_module(ad) ? (void*)const_cast<UcudaAdaptParams*>(&ad.params) : (void*)d_ptr;
    }

CUresult ParametricEngine::Impl::launch_lyap_ad(bool ls, AdDeviceArgs& a, const AdaptiveRequest& ad, int nPts, int cur,
                            size_t calculated, int dimension, numb* d_ranges, int* d_idx, numb* d_ic,
                            int nIC, numb* d_vals, int nVals, double maxValue, numb* d_res, int logAxisMask,
                            double transientTime, double tMax, double NT, double eps, double vectorTransient,
                            RunSignals& sig, int progressStride) {
        const size_t sharedPerThread = (size_t)ucuda_shared_stride(nIC, nVals) * sizeof(numb);
        int blockSize = launch_block_size(sharedPerThread);
        if (blockSize < 1) blockSize = 1;
        const int gridSize  = (cur + blockSize - 1) / blockSize;
        numb   maxValue_a = (numb)maxValue, tolRatio = (numb)ad.tol_ratio, tTr = (numb)transientTime;
        numb   tMax_a = (numb)tMax, NT_a = (numb)NT, eps_a = (numb)eps;
        int    nBlocks = (int)(tMax / NT), renorm = ad.lyap_renorm;
        int    nWarm = lyap_warm_blocks(vectorTransient, NT);
        size_t units = kAdProgressUnits;
        UcudaAdaptParams* dP = a.params;
        int*   dAx = a.axis;
        numb*  dSt = a.stats;
        int*   d_cancel = sig.cancelArg();
        int*   d_prog   = sig.progressArg();
        void* args[] = { &nPts, &cur, &calculated, &dimension, &d_ranges, &d_idx, &d_ic, &nIC, &d_vals, &nVals,
                         &maxValue_a, &d_res, &logAxisMask, &dP, &dAx, &tolRatio, &tTr, &tMax_a, &NT_a,
                         &nBlocks, &nWarm, &eps_a, &renorm, &d_cancel, &d_prog, &progressStride, &units, &dSt };
        return cuLaunchKernel(ls ? cached_lyap_ad.kernel_ls : cached_lyap_ad.kernel_lle,
                              gridSize, 1, 1, blockSize, 1, 1, (unsigned int)(sharedPerThread * blockSize),
                              nullptr, args, nullptr);
    }

bool ParametricEngine::Impl::compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        // par_or_var в kernel — compile-time макрос (см. bif1d). Два PTX-модуля
        // кешируются отдельно: param-sweep и IC-sweep.
        std::string key = hash_key(krs_body, amountOfX) + ":lle:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle, key, cached_lle, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_lle), "lle1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":lle2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_lle_2d, key, cached_lle_2d, activate, [&](CachedLleModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_lle_2d), "lle2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LLEKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_lle, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls, key, cached_ls, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_ls), "ls1d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls2d:pov" + std::to_string(par_or_var);
        return compile_into(pool_ls_2d, key, cached_ls_2d, activate, [&](CachedLsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_ls_2d), "ls2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "LSKernelCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_ls, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":cont";
        return compile_into(pool_cont, key, cached_cont, activate, [&](CachedContModule& fresh) {
            // bifurcation1dContinuationKernel — extern "C" (имя не мангается), поэтому в name_exprs
            // не идёт и берётся из модуля напрямую. peakFinderCUDA и DFT_custom — обычные C++
            // символы, нужны mangled-варианты; DFT_custom уже в этом модуле (шаблон #include'ит
            // cudaLibrary.cu целиком) и нужен continuation-ветке run_dft_1d.
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_cont), "bifurcation1d_cont.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "peakFinderCUDA", "DFT_custom_pf" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "bifurcation1dContinuationKernel", fresh.kernel_cont, err))
                { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[0], fresh.kernel_peak, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dft,  err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_simple_cont_if_needed(ModuleLru<CachedSimpleContModule>& pool,
                                       CachedSimpleContModule& slot,
                                       const std::string& tmpl,
                                       const char* kernel_name,
                                       const char* src_name,
                                       const char* key_suffix,
                                       const std::string& krs_body, int amountOfX,
                                       std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + key_suffix;
        return compile_into(pool, key, slot, true, [&](CachedSimpleContModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;   // пуст: единственный символ — extern "C"
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + tmpl), src_name,
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              {},
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, kernel_name, fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":bif2d:" + std::to_string(par_or_var);
        return compile_into(pool_bif2d, key, cached_bif2d, activate, [&](CachedBif2dModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_bif2d), "bifurcation2d.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelPeaksCUDA", "dbscanCUDA",
                                "dbscanWarpCUDA", "ucudaMaxPeaksCUDA", "dbscanWarp2DCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused,  err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_dbscan, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[2], fresh.kernel_dbscan_warp, err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[3], fresh.kernel_max_peaks,   err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[4], fresh.kernel_dbscan_warp2d, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_metrics_if_needed(const std::string& krs_body, int amountOfX,
                                   int par_or_var, std::string& err, bool activate ,
                                   bool minmax ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":metrics:" + std::to_string(par_or_var)
                        + (minmax ? ":mm" : "");
        return compile_into(pool_metrics, key, cached_metrics, activate, [&](CachedMetricsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            const std::string tmpl = minmax ? "#define SIGM_MINMAX_INTERP 1\n" + src_template_metrics
                                            : src_template_metrics;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + tmpl), minmax ? "signal_metrics_mm.cu" : "signal_metrics.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body },
                                { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                              { "calculateDiscreteModelMetricsCUDA", "signalMetricsContinuationKernel" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel,      err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_cont, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_metrics_ad_if_needed(const AdaptiveRequest& ad, const std::string& krs_body, int amountOfX,
                                      int par_or_var, std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string bodies = krs_body;
        for (const std::string* b : { &ad.prefix, &ad.rhs, &ad.emb, &ad.dprep, &ad.deval, &ad.ctrl_body }) { bodies += '\x1f'; bodies += *b; }
        // Узлы шага — модуль без плотного выхода, как у БД (см. compile_ad_if_needed).
        const bool nodes = ad_nodes_module(ad);
        const std::string key = hash_key(bodies, amountOfX) + ":metrics_ad:pov" + std::to_string(par_or_var)
                              + (nodes ? ":nodes" : "");
        return compile_into(pool_metrics_ad, key, cached_metrics_ad, activate, [&](CachedMetricsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            // Ядра свипов БД и бассейнов из adaptive_part.cu метрикам не нужны — только время ptxas.
            const std::string tmpl = std::string(nodes ? kAdNodesModulePrefix : "")
                                   + "#define SIGM_MINMAX_INTERP 1\n#define UCUDA_AD_NO_SWEEP_KERNELS 1\n"
                                   + src_template_metrics + "\n"
                                   + src_adaptive_part + "\n" + src_metrics_adaptive_part;
            if (!build_module(snapshot_sources(ad.prefix + time_prefix(ad.rhs) + tmpl), nodes ? "signal_metrics_ad_nodes.cu" : "signal_metrics_ad.cu",
                              { { "{{AMOUNT_OF_X}}",     std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",        krs_body },
                                { "{{PAR_OR_VAR}}",      std::to_string(par_or_var) },
                                { "{{KRS_RHS_BODY}}",    ad.rhs },
                                { "{{KRS_EMB_BODY}}",    ad.emb },
                                { "{{KRS_DPREP_BODY}}",  ad.dprep },
                                { "{{KRS_DEVAL_BODY}}",  ad.deval },
                                { "{{CTRL_CUSTOM}}",     adaptive_ctrl_source(ad.ctrl_body) } },
                              { "calculateDiscreteModelMetricsAdCUDA", "signalMetricsContinuationAdKernel" },
                              mod, mg, err))
                return false;
            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel,      err)) { cuModuleUnload(mod); return false; }
            if (!module_fn(mod, mg[1], fresh.kernel_cont, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }
