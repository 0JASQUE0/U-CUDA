#include "engine_impl.h"


ParametricEngine::ParametricEngine()  : impl_(std::make_unique<Impl>()) {}
ParametricEngine::~ParametricEngine() = default;

Bifurcation1DResult ParametricEngine::run_bifurcation_1d(const Bifurcation1DRequest& req) {
    return impl_->run_bif1d(req);
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
    return impl_->run_ls_1d(req);
}

LLE2DResult ParametricEngine::run_lle_2d(const LLE2DRequest& req) {
    return impl_->run_lle_2d(req);
}

LS2DResult ParametricEngine::run_ls_2d(const LS2DRequest& req) {
    return impl_->run_ls_2d(req);
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

// ---------------------------------------------------------------------------
// Signal metrics: имена и CSV.

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

// Одна таблица на прогон, «длинный» формат: строка = точка свипа,
// колонки x [, y], regime, затем посчитанные метрики. Значения осей — те же
// узлы сетки, что видело ядро (ucuda_node_value / _log), а не интерполяция
// поверх них — см. getValueByIdx_local.
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
