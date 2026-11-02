#include "engine_impl.h"
using namespace ucuda::cuda_detail;
// Metrics: оси свипа, классический и продолжающий счёт на GPU, CPU и адаптивный шаг; прогрев модулей.

std::string ParametricEngine::Impl::metrics_axes(const SignalMetricsRequest& req, MetricsAxes& a) {
        auto check_param = [&](int p) { return p >= 0 && p < (int)req.base_values.size(); };
        auto check_var   = [&](int v) { return v >= 0 && v < req.amountOfX; };

        if (req.sweep_over_var && dae_z_axis(req.krs_body, req.var_sweep_index)) return kDaeZAxisMsg;
        if (req.dimension == 2 && req.sweep_over_var_2 && dae_z_axis(req.krs_body, req.var_sweep_index_2)) return kDaeZAxisMsg;
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return "log scale requires param lo/hi > 0 (X axis)";

        // Адаптивный шаг: h не свипается; ось настройки шага разбирается как ось h,
        // но h-осью для ядра не становится (hSweepAxis = -1).
        const bool ad   = req.adaptive.enabled;
        const bool ad_x = ad && req.adaptive.axis_kind[0] != kAdAxisSystem;
        const bool ad_y = ad && req.adaptive.axis_kind[1] != kAdAxisSystem;
        if (ad && (req.sweep_over_h || req.sweep_over_h_2))
            return "the step h is not swept with the adaptive step: sweep rtol / atol instead";
        const bool over_h  = req.sweep_over_h   || ad_x;
        const bool over_h2 = req.sweep_over_h_2 || ad_y;
        auto any_sign = [&](int axis) { return ad && req.adaptive.axis_kind[axis] >= kAdAxisCtrl; };

        if (req.dimension == 1) {
            if (over_h) {
                if (!any_sign(0) && (req.param_lo <= 0.0 || req.param_hi <= 0.0))
                    return "lo/hi must be > 0 on the axis of h / rtol / atol";
                a.hSweepAxis = ad ? -1 : 0;
                a.par_or_var = 1;
                a.idx_x = 0;
            } else if (req.sweep_over_var) {
                if (!check_var(req.var_sweep_index)) return "var_sweep_index out of range";
                a.par_or_var = 0;
                a.idx_x = req.var_sweep_index;
            } else {
                if (!check_param(req.param_index)) return "param_index out of range";
                a.par_or_var = 1;
                a.idx_x = req.param_index;
            }
            a.lo_x = req.param_lo; a.hi_x = req.param_hi;
            a.lo_y = 0.0;          a.hi_y = 1.0;
            a.logAxisMask = req.log_scale ? 1 : 0;
            return {};
        }

        if (ad) {
            const std::string e3 = ad_axes_overlap_error(req.adaptive.axis_kind[0], req.adaptive.axis_kind[1]);
            if (!e3.empty()) return e3;
        }
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return "log scale requires param lo/hi > 0 (Y axis)";
        if (req.sweep_over_h && req.sweep_over_h_2)
            return "only one axis can sweep over dt (h)";

        bool log_x = req.log_scale, log_y = req.log_scale_2;
        if (over_h && over_h2) {
            // Обе оси — настройки шага (скажем, rtol x atol): система не свипается.
            a.par_or_var = 1;
            a.idx_x = 0; a.idx_y = 0;
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
            a.logAxisMask = (log_x ? 1 : 0) | (log_y ? 2 : 0);
            return {};
        }
        a.par_or_var = par_or_var_2d(over_h, over_h2,
                                     req.sweep_over_var, req.sweep_over_var_2);
        if (over_h || over_h2) {
            a.hSweepAxis = ad ? -1 : (over_h ? 0 : 1);
            if (over_h) {
                if (a.par_or_var == 1) { if (!check_param(req.param_index_2))   return "param_index_2 (Y axis) out of range"; a.idx_y = req.param_index_2; }
                else                   { if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range"; a.idx_y = req.var_sweep_index_2; }
                a.idx_x = 0;
            } else {
                if (a.par_or_var == 1) { if (!check_param(req.param_index))     return "param_index (X axis) out of range"; a.idx_x = req.param_index; }
                else                   { if (!check_var(req.var_sweep_index))   return "var_sweep_index (X axis) out of range"; a.idx_x = req.var_sweep_index; }
                a.idx_y = 0;
            }
            const double h_lo = over_h ? req.param_lo : req.param_lo_2;
            const double h_hi = over_h ? req.param_hi : req.param_hi_2;
            if (!any_sign(over_h ? 0 : 1) && (h_lo <= 0.0 || h_hi <= 0.0))
                return std::string("lo/hi must be > 0 on the axis of h / rtol / atol (axis ")
                       + (over_h ? "X" : "Y") + ")";
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
            if (a.par_or_var == 1) {
                if (!check_param(req.param_index))   return "param_index (X axis) out of range";
                if (!check_param(req.param_index_2)) return "param_index_2 (Y axis) out of range";
                a.idx_x = req.param_index; a.idx_y = req.param_index_2;
            } else {
                if (!check_var(req.var_sweep_index))   return "var_sweep_index (X axis) out of range";
                if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range";
                a.idx_x = req.var_sweep_index; a.idx_y = req.var_sweep_index_2;
            }
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else if (req.sweep_over_var && !req.sweep_over_var_2) {
            if (!check_var(req.var_sweep_index))  return "var_sweep_index (X axis) out of range";
            if (!check_param(req.param_index_2))  return "param_index_2 (Y axis) out of range";
            a.idx_x = req.var_sweep_index; a.idx_y = req.param_index_2;
            a.lo_x = req.param_lo;   a.hi_x = req.param_hi;
            a.lo_y = req.param_lo_2; a.hi_y = req.param_hi_2;
        } else {
            if (!check_var(req.var_sweep_index_2)) return "var_sweep_index_2 (Y axis) out of range";
            if (!check_param(req.param_index))     return "param_index (X axis) out of range";
            a.idx_x = req.var_sweep_index_2; a.idx_y = req.param_index;
            a.lo_x = req.param_lo_2; a.hi_x = req.param_hi_2;
            a.lo_y = req.param_lo;   a.hi_y = req.param_hi;
            a.swap_xy = true;
            log_x = req.log_scale_2; log_y = req.log_scale;
        }
        a.logAxisMask = (log_x ? 1 : 0) | (log_y ? 2 : 0);
        return {};
    }

SignalMetricsResult ParametricEngine::Impl::run_signal_metrics(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        if (req.adaptive.enabled && !req.adaptive.setup_error.empty()) return fail(req.adaptive.setup_error);
        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.dimension != 1 && req.dimension != 2)                return fail("dimension must be 1 or 2");
        if (req.n_pts <= 0)          return fail("n_pts must be > 0");
        if (req.h <= 0.0)            return fail("h must be > 0");
        if (req.t_max <= 0.0)        return fail("t_max must be > 0");
        if (req.transient_time < 0)  return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)    return fail("pre_scaller must be > 0");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");
        const int mask = req.metric_mask & kSignalMetricAllMask;
        if (mask == 0)               return fail("no metrics selected");

        MetricsAxes ax;
        {
            const std::string e = metrics_axes(req, ax);
            if (!e.empty()) return fail(e);
        }

        // Continuation и CPU — только 1D; обе ветки уходят до ensure_init
        // (CPU-ветке CUDA не нужна вовсе).
        const bool ad = req.adaptive.enabled;
        if (req.continuation) {
            if (req.dimension != 1)  return fail("continuation works in 1D only");
            if (req.sweep_over_var)  return fail("continuation requires a param or h sweep, not an IC sweep");
            // Адаптивный: CPU — нативная цепочка (run_metrics_cpu_ad), GPU — ядро в одну нить.
            if (ad)                  return req.use_cpu ? run_metrics_cpu_ad(req, ax) : run_metrics_continuation_ad(req);
            if (req.use_cpu)         return run_metrics_cpu(req, true);
            return run_metrics_continuation_gpu(req);
        }
        if (req.use_cpu) {
            if (req.dimension != 1)  return fail("CPU computation works in 1D only");
            return ad ? run_metrics_cpu_ad(req, ax) : run_metrics_cpu(req, false);
        }

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (ad) {
            if (!compile_metrics_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, ax.par_or_var, err)) return fail(err);
        }
        else if (!compile_metrics_if_needed(req.krs_body, req.amountOfX, ax.par_or_var, err, true,
                                            req.minmax_interp)) return fail(err);

        const int    nPts          = req.n_pts;
        const int    dimension     = req.dimension;
        const int    amountOfIC    = req.amountOfX;
        const int    amountOfValues= (int)req.base_values.size();
        const int    preScaller    = req.pre_scaller;
        const std::vector<numb> ic_staged     = to_numb(req.initial_conditions);
        const std::vector<numb> values_staged = to_numb(req.base_values);
        numb ranges[4]           = { (numb)ax.lo_x, (numb)ax.hi_x, (numb)ax.lo_y, (numb)ax.hi_y };
        int  indicesOfMutVars[2] = { ax.idx_x, ax.idx_y };

        // Буфер — под худший случай шага (минимальный h на h-оси), см. run_bif2d.
        const double worstCaseH = (ax.hSweepAxis == 0) ? ranges[0] : (ax.hSweepAxis == 1) ? ranges[2] : req.h;
        int amountOfPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / preScaller);
        // Узлы шага: отсчётов сетки нет, строка интервалов — под потолок пиков (как у БД).
        const bool adRaw = ad && req.adaptive.raw_nodes;
        if (adRaw) amountOfPointsInBlock = (int)max_amount_of_peaks + 1;
        const size_t amountOfPointsForSkip = steps_from_time_size_t(req.transient_time, req.h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Интервалы нужны одной медиане; "+1" — стадия интервалов съедает один
        // сырой пик (см. run_bif2d).
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (size_t)amountOfPointsInBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)amountOfPointsInBlock
                                   : (size_t)max_amount_of_peaks + 1;
        const int    peakCapacity  = (int)peakStride;

        const size_t total_cells = dimension == 2 ? (size_t)nPts * (size_t)nPts : (size_t)nPts;

        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");
        const size_t memPerCell  = (size_t)SIGM_COUNT * sizeof(numb) + sizeof(int)
                                 + (needIntervals ? peakStride * sizeof(numb) : 0);
        const size_t memConstants = (4 + (size_t)amountOfIC + (size_t)amountOfValues) * sizeof(numb) + 2 * sizeof(int);
        if (memConstants >= freeMemory) return fail("not enough GPU memory for constants");

        size_t nPtsLimiter = (freeMemory - memConstants) / memPerCell;
        if (nPtsLimiter < 32)          nPtsLimiter = 32;
        if (nPtsLimiter > total_cells) nPtsLimiter = total_cells;
        const size_t chunk = nPtsLimiter;
        const size_t amountOfIteration = (total_cells + chunk - 1) / chunk;

        const size_t stepsPerCell   = ad ? kAdProgressUnits
                                         : amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;

        numb* d_ranges    = nullptr;
        int*  d_indices   = nullptr;
        numb* d_ic        = nullptr;
        numb* d_values    = nullptr;
        numb* d_intervals = nullptr;
        numb* d_out       = nullptr;
        int*  d_flags     = nullptr;
        RunSignals sig;
        CUstream stream = nullptr;
        AdDeviceArgs adArgs;                  // адаптивный шаг: параметры, оси, статистика

        auto cleanup = [&]() {
            adArgs.release();
            if (d_ranges)    cudaFree(d_ranges);
            if (d_indices)   cudaFree(d_indices);
            if (d_ic)        cudaFree(d_ic);
            if (d_values)    cudaFree(d_values);
            if (d_intervals) cudaFree(d_intervals);
            if (d_out)       cudaFree(d_out);
            if (d_flags)     cudaFree(d_flags);
            sig.release();
            if (stream)      cuStreamDestroy(stream);
        };

        #define SIGM_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define SIGM_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define SIGM_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        SIGM_CHECK(cudaMalloc((void**)&d_ranges,  4 * sizeof(numb)),                           "cudaMalloc d_ranges");
        SIGM_CHECK(cudaMalloc((void**)&d_indices, 2 * sizeof(int)),                            "cudaMalloc d_indices");
        SIGM_CHECK(cudaMalloc((void**)&d_ic,      (size_t)amountOfIC * sizeof(numb)),          "cudaMalloc d_ic");
        SIGM_CHECK(cudaMalloc((void**)&d_values,  (size_t)amountOfValues * sizeof(numb)),      "cudaMalloc d_values");
        SIGM_CHECK(cudaMalloc((void**)&d_out,     chunk * (size_t)SIGM_COUNT * sizeof(numb)),  "cudaMalloc d_out");
        SIGM_CHECK(cudaMalloc((void**)&d_flags,   chunk * sizeof(int)),                        "cudaMalloc d_flags");
        if (needIntervals)
            SIGM_CHECK(cudaMalloc((void**)&d_intervals, chunk * peakStride * sizeof(numb)),    "cudaMalloc d_intervals");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        if (ad) {
            // swap_xy — только при двух системных осях, коды осей идут как есть.
            const int kinds[2] = { req.adaptive.axis_kind[0], req.adaptive.axis_kind[1] };
            if (!adArgs.alloc(req.adaptive, kinds, chunk, res.error)) { cleanup(); return res; }
            res.ad_stats.assign(total_cells * 4, 0.0);
        }

        SIGM_CHECK(cudaMemcpy(d_ranges,  ranges,               4 * sizeof(numb),                      cudaMemcpyHostToDevice), "memcpy d_ranges");
        SIGM_CHECK(cudaMemcpy(d_indices, indicesOfMutVars,     2 * sizeof(int),                       cudaMemcpyHostToDevice), "memcpy d_indices");
        SIGM_CHECK(cudaMemcpy(d_ic,      ic_staged.data(),     (size_t)amountOfIC * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy d_ic");
        SIGM_CHECK(cudaMemcpy(d_values,  values_staged.data(), (size_t)amountOfValues * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_values");
        SIGM_CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

        res.dimension   = dimension;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;   res.param_hi   = req.param_hi;
        res.param_lo_2  = req.param_lo_2; res.param_hi_2 = req.param_hi_2;
        res.log_scale   = req.log_scale;  res.log_scale_2 = req.log_scale_2;
        res.metric_mask = mask;
        // Точки чанков, до которых не дошёл Stop, остаются пустыми.
        res.flags.assign(total_cells, REGIME_CANCELLED);
        for (int m = 0; m < SIGM_COUNT; ++m)
            if ((mask >> m) & 1) res.values[m].assign(total_cells, std::numeric_limits<double>::quiet_NaN());

        std::vector<numb> h_row(chunk);
        std::vector<int>  h_flags(chunk);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            if (iter > 0 && cancel_requested(req.cancel)) break;   // Stop: отдаём посчитанное
            SIGM_CANCEL_CHECK();
            if (req.progress) req.progress->store((float)((double)(chunk * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            const size_t cur = (iter == amountOfIteration - 1) ? total_cells - chunk * iter : chunk;

            const size_t sharedPerThread = (size_t)ucuda_shared_stride(amountOfIC, amountOfValues) * sizeof(numb);
            const int blockSize = launch_block_size(sharedPerThread);
            const int gridSize  = (int)((cur + blockSize - 1) / blockSize);

            int    nPts_arg        = nPts;
            int    limiter_arg     = (int)cur;
            size_t calculated_arg  = iter * chunk;
            size_t skip_arg        = amountOfPointsForSkip;
            int    dimension_arg   = dimension;
            numb   h_arg           = (numb)req.h;
            int    amountOfIC_arg  = amountOfIC;
            int    amountOfVal_arg = amountOfValues;
            size_t iterations_arg  = (size_t)amountOfPointsInBlock;
            int    preScaller_arg  = preScaller;
            int    writableVar_arg = req.writable_var;
            numb   maxValue_arg    = (numb)req.max_value;
            size_t peakStride_arg  = peakStride;
            int    peakCap_arg     = peakCapacity;
            size_t metricStride_arg= cur;
            int    mask_arg        = mask;
            int    hSweep_arg      = ax.hSweepAxis;
            numb   transient_arg   = (numb)req.transient_time;
            numb   tMax_arg        = (numb)req.t_max;
            int    logMask_arg     = ax.logAxisMask;
            int*   d_cancel_arg    = sig.cancelArg();
            int*   d_progress_arg  = sig.progressArg();
            int    progStride_arg  = progressStride;

            void* args[] = {
                &nPts_arg, &limiter_arg, &calculated_arg, &skip_arg, &dimension_arg,
                &d_ranges, &h_arg, &d_indices, &d_ic, &amountOfIC_arg,
                &d_values, &amountOfVal_arg, &iterations_arg, &preScaller_arg,
                &writableVar_arg, &maxValue_arg,
                &d_intervals, &peakStride_arg, &peakCap_arg,
                &d_out, &metricStride_arg, &mask_arg, &d_flags,
                &hSweep_arg, &transient_arg, &tMax_arg, &logMask_arg,
                &d_cancel_arg, &d_progress_arg, &progStride_arg
            };
            sig.resetTicks();
            const unsigned int shared = (unsigned int)(sharedPerThread * blockSize);
            if (ad) {
                UcudaAdaptParams* d_adp_arg = adArgs.params;
                int*   d_axis_arg        = adArgs.axis;
                numb   tolRatio_arg      = (numb)req.adaptive.tol_ratio;
                numb   dtOut_arg         = (numb)req.h;
                size_t progressUnits_arg = kAdProgressUnits;
                numb*  d_adStats_arg     = adArgs.stats;
                int    raw_arg           = adRaw ? 1 : 0;
                int    interp_arg        = req.adaptive.peak_interp;
                void* args_ad[] = {
                    &nPts_arg, &limiter_arg, &calculated_arg, &dimension_arg,
                    &d_ranges, &d_indices, &d_ic, &amountOfIC_arg,
                    &d_values, &amountOfVal_arg, &writableVar_arg, &maxValue_arg,
                    &d_intervals, &peakStride_arg, &peakCap_arg,
                    &d_out, &metricStride_arg, &mask_arg, &d_flags, &logMask_arg,
                    ad_param_arg(req.adaptive, &d_adp_arg), &d_axis_arg, &tolRatio_arg, &transient_arg, &tMax_arg, &dtOut_arg,
                    &preScaller_arg, &iterations_arg, &raw_arg, &interp_arg,
                    &d_cancel_arg, &d_progress_arg, &progStride_arg,
                    &progressUnits_arg, &d_adStats_arg
                };
                SIGM_CHECK_CU(cuLaunchKernel(cached_metrics_ad.kernel,
                                             gridSize, 1, 1, blockSize, 1, 1,
                                             shared, stream, args_ad, nullptr),
                              "cuLaunchKernel(signal metrics adaptive)");
            }
            else
            SIGM_CHECK_CU(cuLaunchKernel(cached_metrics.kernel,
                                         gridSize, 1, 1, blockSize, 1, 1,
                                         shared, stream, args, nullptr),
                          "cuLaunchKernel(signal metrics)");

            if (!wait_with_signals(stream, sig, req.cancel, req.progress,
                                   (double)(chunk * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            SIGM_CHECK(cudaStreamSynchronize(stream), "sync stream");
            // Stop посреди чанка: прерванные точки помечены REGIME_CANCELLED, метрики у них NaN.

            // Индекс ядра -> индекс результата: в 2D при swap_xy ядро шло по осям
            // в обратном порядке, результат всегда в раскладке GUI [iy*n + ix].
            auto out_index = [&](size_t k) -> size_t {
                const size_t kernel_idx = chunk * iter + k;
                if (!ax.swap_xy) return kernel_idx;
                const size_t kix = kernel_idx % (size_t)nPts;
                const size_t kiy = kernel_idx / (size_t)nPts;
                return kix * (size_t)nPts + kiy;
            };

            SIGM_CHECK(cudaMemcpy(h_flags.data(), d_flags, cur * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
            for (size_t k = 0; k < cur; ++k) res.flags[out_index(k)] = h_flags[k];
            if (ad) {
                std::vector<numb> h_st(cur * 4);
                SIGM_CHECK(cudaMemcpy(h_st.data(), adArgs.stats, cur * 4 * sizeof(numb), cudaMemcpyDeviceToHost),
                           "memcpy adaptive stats");
                for (size_t k = 0; k < cur; ++k)
                    for (int q = 0; q < 4; ++q) res.ad_stats[out_index(k) * 4 + q] = (double)h_st[k * 4 + q];
            }

            for (int m = 0; m < SIGM_COUNT; ++m) {
                if (!((mask >> m) & 1)) continue;
                SIGM_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * cur, cur * sizeof(numb),
                                      cudaMemcpyDeviceToHost), "memcpy metrics");
                std::vector<double>& dst = res.values[m];
                for (size_t k = 0; k < cur; ++k) dst[out_index(k)] = (double)h_row[k];
            }
        }

        cleanup();
        #undef SIGM_CHECK
        #undef SIGM_CHECK_CU
        #undef SIGM_CANCEL_CHECK

        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_continuation_gpu(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_metrics_if_needed(req.krs_body, req.amountOfX, 1, err, true, req.minmax_interp)) return fail(err);

        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
        const int sizeOfBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (sizeOfBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (size_t)sizeOfBlock < (size_t)max_amount_of_peaks + 1
                                   ? (size_t)sizeOfBlock : (size_t)max_amount_of_peaks + 1;

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_intervals  = nullptr;
        numb* d_out        = nullptr;
        int*  d_flags      = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_intervals)  cudaFree(d_intervals);
            if (d_out)        cudaFree(d_out);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define SMC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SMC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SMC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)),        "cudaMalloc d_baseValues");
        SMC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),         "cudaMalloc d_baseX");
        SMC_CHECK(cudaMalloc((void**)&d_out,        (size_t)SIGM_COUNT * (size_t)nPts * sizeof(numb)), "cudaMalloc d_out");
        SMC_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                   "cudaMalloc d_flags");
        if (needIntervals)
            SMC_CHECK(cudaMalloc((void**)&d_intervals, peakStride * sizeof(numb)),                 "cudaMalloc d_intervals");
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        const std::vector<numb> baseValues_staged = to_numb(req.base_values);
        const std::vector<numb> baseX_staged      = to_numb(req.initial_conditions);
        SMC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged.data(), req.base_values.size() * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseValues");
        SMC_CHECK(cudaMemcpy(d_baseX, baseX_staged.data(), (size_t)req.amountOfX * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseX");
        SMC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        int  nPts_arg        = nPts;
        numb lo_arg          = (numb)req.param_lo;
        numb hi_arg          = (numb)req.param_hi;
        int  reverse_arg     = req.continuation_reverse ? 1 : 0;
        int  logScale_arg    = req.log_scale ? 1 : 0;
        int  sweepIsH_arg    = req.sweep_over_h ? 1 : 0;
        int  mutParamIdx_arg = req.param_index;
        int  amountOfVal_arg = (int)req.base_values.size();
        int  amountOfX_arg   = req.amountOfX;
        numb h_arg           = (numb)req.h;
        numb tMax_arg        = (numb)req.t_max;
        numb transient_arg   = (numb)req.transient_time;
        int  sizeOfBlock_arg = sizeOfBlock;
        int  preScaller_arg  = req.pre_scaller;
        int  writableVar_arg = req.writable_var;
        numb maxValue_arg    = (numb)req.max_value;
        int  peakCap_arg     = (int)peakStride;
        int  mask_arg        = mask;
        int* d_cancel_arg    = sig.cancelArg();
        int* d_progress_arg  = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfVal_arg, &d_baseX, &amountOfX_arg,
            &h_arg, &tMax_arg, &transient_arg, &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_intervals, &peakCap_arg,
            &d_out, &mask_arg, &d_flags,
            &d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SMC_CHECK_CU(cuLaunchKernel(cached_metrics.kernel_cont, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                     "cuLaunchKernel(metrics cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SMC_CHECK(cudaDeviceSynchronize(), "sync after metrics cont");
        // Stop посреди цепочки: ядро пометило хвост REGIME_CANCELLED / NaN, начало цело.

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = true;
        res.continuation_reverse = req.continuation_reverse;

        std::vector<int>  h_flags((size_t)nPts);
        std::vector<numb> h_row((size_t)nPts);
        SMC_CHECK(cudaMemcpy(h_flags.data(), d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        res.flags.assign(h_flags.begin(), h_flags.end());
        for (int m = 0; m < SIGM_COUNT; ++m) {
            if (!((mask >> m) & 1)) continue;
            SMC_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * (size_t)nPts, (size_t)nPts * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy metrics");
            res.values[m].assign(h_row.begin(), h_row.end());
        }

        cleanup();
        #undef SMC_CHECK
        #undef SMC_CHECK_CU
        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_cpu_ad(const SignalMetricsRequest& req, const MetricsAxes& ax) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };
        const AdaptiveRequest& ad = req.adaptive;
        const bool cont = req.continuation;
        if (req.dimension != 1) return fail("CPU computation works in 1D only");
        if (cont && !(ad.axis_kind[0] != kAdAxisSystem) && (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
            return fail("param_index out of range");

        AdaptiveCpuModule mod;
        {
            std::vector<KrsCpuDiag> diags;
            if (!mod.compile(ad.rhs, ad.emb, ad.dprep, ad.deval, ad.ctrl_body, req.amountOfX,
                             ad.prefix + peak_config_defines(), diags) || !mod.metrics()) {
                std::string e = "CPU adaptive module:";
                for (const KrsCpuDiag& d : diags)
                    e += "\n" + (d.line > 0 ? "line " + std::to_string(d.line) + ": " : std::string()) + d.message;
                if (diags.empty()) e += " no dense output for this scheme";
                return fail(e);
            }
        }

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   raw   = ad.raw_nodes;   // узлы шага: h не участвует, интервалы — под потолок пиков
        const size_t iters = raw ? 0 : (size_t)std::ceil(req.t_max / req.h / (double)req.pre_scaller);
        if (!raw && (iters == 0 || iters > (size_t)(std::numeric_limits<int>::max)()))
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (raw || iters >= (size_t)max_amount_of_peaks + 1)
                                   ? (size_t)max_amount_of_peaks + 1 : iters;
        const int    peakCapacity  = (int)peakStride;
        const std::vector<double> values = req.base_values, x0 = req.initial_conditions;
        const double lo = cont ? req.param_lo : ax.lo_x, hi = cont ? req.param_hi : ax.hi_x;
        const int    mutIdx = cont ? (ad.axis_kind[0] != kAdAxisSystem ? 0 : req.param_index) : ax.idx_x;

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = cont;
        res.continuation_reverse = cont && req.continuation_reverse;
        // Точки, до которых не дошёл Stop, остаются пустыми.
        res.flags.assign((size_t)nPts, REGIME_CANCELLED);
        for (int m = 0; m < SIGM_COUNT; ++m)
            if ((mask >> m) & 1) res.values[m].assign((size_t)nPts, std::numeric_limits<double>::quiet_NaN());
        res.ad_stats.assign((size_t)nPts * 4, 0.0);

        // Stop до запуска — прежний отказ, а не пустой частичный результат.
        if (cancel_requested(req.cancel)) { res.cancelled = true; return fail("Cancelled by user"); }
        volatile int cancelFlag = 0;
        std::atomic<int> donePts{ 0 };
        std::atomic<bool> finished{ false };
        int chainProg = 0;
        std::thread watcher([&]() {
            while (!finished.load(std::memory_order_relaxed)) {
                if (req.cancel && req.cancel->load(std::memory_order_relaxed)) cancelFlag = 1;
                const int got = cont ? *(volatile int*)&chainProg : donePts.load(std::memory_order_relaxed);
                if (req.progress) req.progress->store((float)got / (float)nPts, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        auto call = [&](int i0, int i1, double* iv, double* om, int* fl, double* st, int* prog) {
            return mod.metrics()(cont ? 1 : 0, i0, i1, nPts, lo, hi, req.continuation_reverse ? 1 : 0,
                                 req.log_scale ? 1 : 0, ax.par_or_var, mutIdx, values.data(), (int)values.size(),
                                 x0.data(), &ad.params, ad.axis_kind[0], ad.tol_ratio, req.writable_var,
                                 req.max_value, iv, (unsigned long long)peakStride, peakCapacity, om, mask, fl,
                                 req.transient_time, req.t_max, req.h, req.pre_scaller, (unsigned long long)iters,
                                 raw ? 1 : 0, ad.peak_interp, st, &cancelFlag, prog);
        };
        bool aborted = false;
        if (cont) {
            std::vector<double> iv(needIntervals ? peakStride : 0), om((size_t)SIGM_COUNT * nPts), st((size_t)nPts * 4);
            // Stop рвёт цепочку: хвост, до которого она не дошла, остаётся пустым.
            std::vector<int> fl((size_t)nPts, REGIME_CANCELLED);
            std::fill(om.begin(), om.end(), std::numeric_limits<double>::quiet_NaN());
            aborted = !call(0, nPts, needIntervals ? iv.data() : nullptr, om.data(), fl.data(), st.data(), &chainProg);
            for (int j = 0; j < nPts; ++j) res.flags[(size_t)j] = fl[(size_t)j];
            for (int m = 0; m < SIGM_COUNT; ++m)
                if ((mask >> m) & 1)
                    for (int j = 0; j < nPts; ++j) res.values[m][(size_t)j] = om[(size_t)m * nPts + j];
            res.ad_stats = st;
        } else {
            // Куски точек — потокам по мере освобождения; строки кусков не пересекаются.
            constexpr int kChunk = 4;
            unsigned hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 1;
            const int nChunks  = (nPts + kChunk - 1) / kChunk;
            const int nWorkers = (int)std::min<unsigned>(hw, (unsigned)nChunks);
            std::atomic<int> nextChunk{ 0 };
            std::atomic<bool> stop{ false };
            auto worker = [&]() {
                std::vector<double> iv(needIntervals ? (size_t)kChunk * peakStride : 0);
                std::vector<double> om((size_t)SIGM_COUNT * kChunk), st((size_t)kChunk * 4);
                std::vector<int> fl((size_t)kChunk);
                for (;;) {
                    const int c = nextChunk.fetch_add(1, std::memory_order_relaxed);
                    if (c >= nChunks || stop.load(std::memory_order_relaxed)) return;
                    const int i0 = c * kChunk, i1 = (std::min)(nPts, i0 + kChunk), rows = i1 - i0;
                    int prog = 0;
                    // Stop рвёт кусок: строки, до которых он не дошёл, остаются пустыми.
                    std::fill(fl.begin(), fl.end(), REGIME_CANCELLED);
                    std::fill(om.begin(), om.end(), std::numeric_limits<double>::quiet_NaN());
                    const bool whole = call(i0, i1, needIntervals ? iv.data() : nullptr, om.data(), fl.data(), st.data(), &prog) != 0;
                    if (!whole) stop.store(true);
                    for (int r = 0; r < rows; ++r) {
                        res.flags[(size_t)(i0 + r)] = fl[(size_t)r];
                        for (int q = 0; q < 4; ++q) res.ad_stats[(size_t)(i0 + r) * 4 + q] = st[(size_t)r * 4 + q];
                    }
                    for (int m = 0; m < SIGM_COUNT; ++m)
                        if ((mask >> m) & 1)
                            for (int r = 0; r < rows; ++r) res.values[m][(size_t)(i0 + r)] = om[(size_t)m * rows + r];
                    if (!whole) return;
                    donePts.fetch_add(rows, std::memory_order_relaxed);
                }
            };
            std::vector<std::thread> pool;
            for (int w = 0; w < nWorkers; ++w) pool.emplace_back(worker);
            for (std::thread& t : pool) t.join();
            aborted = stop.load();
        }
        finished.store(true, std::memory_order_relaxed);
        watcher.join();
        (void)aborted;   // Stop: частичный результат, cancelled ставит metrics_finish
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        metrics_finish(res, req);
        return res;
    }

SignalMetricsResult ParametricEngine::Impl::run_metrics_continuation_ad(const SignalMetricsRequest& req) {
        SignalMetricsResult res;
        auto fail = [&](const std::string& msg) -> SignalMetricsResult& { res.error = msg; return res; };

        const AdaptiveRequest& ad = req.adaptive;
        const bool ad_axis = ad.axis_kind[0] != kAdAxisSystem;
        if (!ad.setup_error.empty()) return fail(ad.setup_error);
        if (req.sweep_over_h)
            return fail("the step h is not swept with the adaptive step: sweep rtol / atol instead");
        if (!ad_axis && (req.param_index < 0 || req.param_index >= (int)req.base_values.size()))
            return fail("param_index out of range");
        const bool raw = ad.raw_nodes;   // узлы шага: h не участвует, интервалы — под потолок пиков
        if (!raw && req.h <= 0.0) return fail("h must be > 0");
        const size_t iters = raw ? 0 : (size_t)std::ceil(req.t_max / req.h / (double)req.pre_scaller);
        if (!raw && iters == 0) return fail("amountOfPointsInBlock <= 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_metrics_ad_if_needed(ad, req.krs_body, req.amountOfX, 1, err)) return fail(err);

        const int    nPts  = req.n_pts;
        const int    mask  = req.metric_mask & kSignalMetricAllMask;
        const bool   needIntervals = ((mask >> SIGM_MEDIAN_FREQ) & 1) != 0;
        const size_t peakStride    = (raw || iters >= (size_t)max_amount_of_peaks + 1)
                                   ? (size_t)max_amount_of_peaks + 1 : iters;

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_intervals  = nullptr;
        numb* d_out        = nullptr;
        int*  d_flags      = nullptr;
        AdDeviceArgs adArgs;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            adArgs.release();
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_intervals)  cudaFree(d_intervals);
            if (d_out)        cudaFree(d_out);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define SMA_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SMA_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SMA_CHECK(cudaMalloc((void**)&d_baseValues, (req.base_values.empty() ? 1 : req.base_values.size()) * sizeof(numb)), "cudaMalloc d_baseValues");
        SMA_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),             "cudaMalloc d_baseX");
        SMA_CHECK(cudaMalloc((void**)&d_out,        (size_t)SIGM_COUNT * (size_t)nPts * sizeof(numb)), "cudaMalloc d_out");
        SMA_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                       "cudaMalloc d_flags");
        if (needIntervals)
            SMA_CHECK(cudaMalloc((void**)&d_intervals, peakStride * sizeof(numb)),                     "cudaMalloc d_intervals");
        {
            const int kinds[2] = { ad.axis_kind[0], kAdAxisSystem };
            if (!adArgs.alloc(ad, kinds, (size_t)nPts, res.error)) { cleanup(); return res; }
        }
        if (!sig.alloc(res.error)) { cleanup(); return res; }

        const std::vector<numb> baseValues_staged = to_numb(req.base_values);
        const std::vector<numb> baseX_staged      = to_numb(req.initial_conditions);
        if (!baseValues_staged.empty())
            SMA_CHECK(cudaMemcpy(d_baseValues, baseValues_staged.data(), baseValues_staged.size() * sizeof(numb),
                                 cudaMemcpyHostToDevice), "memcpy d_baseValues");
        SMA_CHECK(cudaMemcpy(d_baseX, baseX_staged.data(), (size_t)req.amountOfX * sizeof(numb),
                             cudaMemcpyHostToDevice), "memcpy d_baseX");
        SMA_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        int    nPts_arg        = nPts;
        numb   lo_arg          = (numb)req.param_lo;
        numb   hi_arg          = (numb)req.param_hi;
        int    reverse_arg     = req.continuation_reverse ? 1 : 0;
        int    logScale_arg    = req.log_scale ? 1 : 0;
        int    mutParamIdx_arg = ad_axis ? 0 : req.param_index;
        int    amountOfVal_arg = (int)req.base_values.size();
        int    writableVar_arg = req.writable_var;
        numb   maxValue_arg    = (numb)req.max_value;
        int    peakCap_arg     = (int)peakStride;
        int    mask_arg        = mask;
        UcudaAdaptParams* d_adp_arg = adArgs.params;
        int*   d_axis_arg      = adArgs.axis;
        numb   tolRatio_arg    = (numb)ad.tol_ratio;
        numb   transient_arg   = (numb)req.transient_time;
        numb   tRec_arg        = (numb)req.t_max;
        numb   dtOut_arg       = (numb)req.h;
        int    preScaller_arg  = req.pre_scaller;
        size_t iters_arg       = iters;
        int    raw_arg         = raw ? 1 : 0;
        int    interp_arg      = ad.peak_interp;
        int*   d_cancel_arg    = sig.cancelArg();
        int*   d_progress_arg  = sig.progressArg();
        numb*  d_adStats_arg   = adArgs.stats;
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfVal_arg, &d_baseX, &writableVar_arg, &maxValue_arg,
            &d_intervals, &peakCap_arg, &d_out, &mask_arg, &d_flags,
            &d_adp_arg, &d_axis_arg, &tolRatio_arg, &transient_arg, &tRec_arg, &dtOut_arg,
            &preScaller_arg, &iters_arg, &raw_arg, &interp_arg, &d_cancel_arg, &d_progress_arg, &d_adStats_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SMA_CHECK_CU(cuLaunchKernel(cached_metrics_ad.kernel_cont, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                     "cuLaunchKernel(metrics adaptive continuation)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SMA_CHECK(cudaDeviceSynchronize(), "sync after metrics adaptive continuation");
        // Stop посреди цепочки: ядро пометило хвост REGIME_CANCELLED / NaN, начало цело.

        res.dimension   = 1;
        res.n_pts       = nPts;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.log_scale   = req.log_scale;
        res.metric_mask = mask;
        res.continuation         = true;
        res.continuation_reverse = req.continuation_reverse;

        std::vector<int>  h_flags((size_t)nPts);
        std::vector<numb> h_row((size_t)nPts);
        std::vector<numb> h_st((size_t)nPts * 4);
        SMA_CHECK(cudaMemcpy(h_flags.data(), d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        res.flags.assign(h_flags.begin(), h_flags.end());
        for (int m = 0; m < SIGM_COUNT; ++m) {
            if (!((mask >> m) & 1)) continue;
            SMA_CHECK(cudaMemcpy(h_row.data(), d_out + (size_t)m * (size_t)nPts, (size_t)nPts * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy metrics");
            res.values[m].assign(h_row.begin(), h_row.end());
        }
        SMA_CHECK(cudaMemcpy(h_st.data(), adArgs.stats, (size_t)nPts * 4 * sizeof(numb), cudaMemcpyDeviceToHost),
                  "memcpy adaptive stats");
        res.ad_stats.assign(h_st.begin(), h_st.end());

        cleanup();
        #undef SMA_CHECK
        #undef SMA_CHECK_CU
        metrics_finish(res, req);
        return res;
    }

bool ParametricEngine::Impl::compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err, bool activate ) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":basins";
        return compile_into(pool_basins, key, cached_basins, activate, [&](CachedBasinsModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> mg;
            if (!build_module(snapshot_sources(time_prefix(krs_body) + lyap_nx_prefix(krs_body) + src_template_basins), "basins.cu",
                              { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                                { "{{KRS_BODY}}",    krs_body } },
                              { "calculateDiscreteModelAvgPeaksCUDA" },
                              mod, mg, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, mg[0], fresh.kernel_fused, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::basins_dbscan_host(const numb* X, const numb* Y, const int* H, size_t n, double eps,
                                   const std::shared_ptr<std::atomic<bool>>& cancel,
                                   const std::shared_ptr<std::atomic<float>>& progress,
                                   int* labels, int& n_pos, int& n_neg) {
        n_pos = 0; n_neg = 0;
        numb eps2 = (numb)eps;
        eps2 = eps2 * eps2;   // в numb, как eps * eps в CUDA_dbscan_kernel
        const bool fmad = get_nvrtc_fmad();
        auto near_pt = [&](size_t a, size_t b) {
            const numb dx = X[b] - X[a];
            const numb dy = Y[b] - Y[a];
            const numb d2 = fmad ? (numb)std::fma(dx, dx, dy * dy) : dx * dx + dy * dy;
            return d2 <= eps2;
        };
        auto cancelled = [&]() { return cancel && cancel->load(std::memory_order_relaxed); };

        // Ячейка: сторона s чуть меньше eps/√2 (запас 1e-4 перекрывает округления и numb = float).
        // Ключ считается в double; при |x/s| > 2^32 погрешность ключа съела бы запас, такие точки
        // (и вообще нечисловые признаки) идут мимо сетки.
        const double s   = eps / std::sqrt(2.0) * (1.0 - 1e-4);
        const double lim = 4294967296.0;
        struct Key { long long cx, cy; uint32_t i; };
        std::vector<Key> keys;     keys.reserve(n);
        std::vector<uint32_t> offgrid;   // конечные, но вне сетки: перебор против всех
        std::vector<int> node(n, -1);   // узел union-find точки; -1 — Unbound (не участвует)
        for (size_t i = 0; i < n; ++i) {
            if (H[i] == REGIME_UNBOUND) continue;
            if (!std::isfinite((double)X[i]) || !std::isfinite((double)Y[i])) continue;   // узел получит ниже; соседей у NaN/inf нет
            const double qx = (double)X[i] / s, qy = (double)Y[i] / s;
            if (!(std::fabs(qx) <= lim && std::fabs(qy) <= lim)) { offgrid.push_back((uint32_t)i); continue; }
            keys.push_back({ (long long)std::floor(qx), (long long)std::floor(qy), (uint32_t)i });
        }
        std::sort(keys.begin(), keys.end(), [](const Key& a, const Key& b) {
            return a.cx != b.cx ? a.cx < b.cx : (a.cy != b.cy ? a.cy < b.cy : a.i < b.i);
        });
        struct Cell { long long cx, cy; size_t b, e; double x0, x1, y0, y1; };
        std::vector<Cell> cells;
        for (size_t k = 0; k < keys.size();) {
            size_t e = k;
            Cell c{ keys[k].cx, keys[k].cy, k, k, INFINITY, -INFINITY, INFINITY, -INFINITY };
            for (; e < keys.size() && keys[e].cx == c.cx && keys[e].cy == c.cy; ++e) {
                const size_t i = keys[e].i;
                c.x0 = (std::min)(c.x0, (double)X[i]); c.x1 = (std::max)(c.x1, (double)X[i]);
                c.y0 = (std::min)(c.y0, (double)Y[i]); c.y1 = (std::max)(c.y1, (double)Y[i]);
                node[i] = (int)cells.size();
            }
            c.e = e;
            cells.push_back(c);
            k = e;
        }
        int nodes = (int)cells.size();
        for (uint32_t i : offgrid) node[i] = nodes++;
        for (size_t i = 0; i < n; ++i)
            if (H[i] != REGIME_UNBOUND && node[i] < 0) node[i] = nodes++;   // NaN/inf: свой узел навсегда

        std::vector<int> parent((size_t)nodes);
        for (int v = 0; v < nodes; ++v) parent[(size_t)v] = v;
        auto find = [&](int v) {
            while (parent[(size_t)v] != v) { parent[(size_t)v] = parent[(size_t)parent[(size_t)v]]; v = parent[(size_t)v]; }
            return v;
        };
        auto unite = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a != b) { if (a < b) std::swap(a, b); parent[(size_t)a] = b; }
        };

        // Пары соседних ячеек. Окно 5x5 без центра, каждая неупорядоченная пара — один раз (сосед
        // с большим ключом). Пара, уже связанная через другие ячейки, не проверяется: плотные
        // кластеры сливаются за первые же проверки. Перед перебором точек — отсев по
        // прямоугольникам ячеек с тем же запасом.
        const double gapMax2 = eps * eps * (1.0 + 1e-4);
        auto box_gap2 = [](double ax0, double ax1, double ay0, double ay1,
                           double bx0, double bx1, double by0, double by1) {
            const double gx = (std::max)(0.0, (std::max)(ax0 - bx1, bx0 - ax1));
            const double gy = (std::max)(0.0, (std::max)(ay0 - by1, by0 - ay1));
            return gx * gx + gy * gy;
        };
        auto find_cell = [&](long long cx, long long cy) -> int {
            auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(cx, cy),
                [](const Cell& c, const std::pair<long long, long long>& k) {
                    return c.cx != k.first ? c.cx < k.first : c.cy < k.second;
                });
            return (it != cells.end() && it->cx == cx && it->cy == cy) ? (int)(it - cells.begin()) : -1;
        };
        for (size_t a = 0; a < cells.size(); ++a) {
            if ((a & 1023) == 0) {
                if (cancelled()) return false;
                if (progress) progress->store(0.9f * float(a) / float(cells.size()), std::memory_order_relaxed);
            }
            const Cell& A = cells[a];
            for (int ox = 0; ox <= 2; ++ox)
                for (int oy = -2; oy <= 2; ++oy) {
                    if (ox == 0 && oy <= 0) continue;
                    const int b = find_cell(A.cx + ox, A.cy + oy);
                    if (b < 0 || find((int)a) == find(b)) continue;
                    const Cell& B = cells[(size_t)b];
                    if (box_gap2(A.x0, A.x1, A.y0, A.y1, B.x0, B.x1, B.y0, B.y1) > gapMax2) continue;
                    bool hit = false;
                    for (size_t p = A.b; p < A.e && !hit; ++p) {
                        const size_t i = keys[p].i;
                        const double xi = (double)X[i], yi = (double)Y[i];
                        if (box_gap2(xi, xi, yi, yi, B.x0, B.x1, B.y0, B.y1) > gapMax2) continue;
                        for (size_t q = B.b; q < B.e; ++q)
                            if (near_pt(i, keys[q].i)) { hit = true; break; }
                    }
                    if (hit) unite((int)a, b);
                }
        }
        // Точки вне сетки: перебор против всех конечных точек (на практике их нет).
        for (uint32_t u : offgrid) {
            if (cancelled()) return false;
            for (size_t j = 0; j < n; ++j)
                if (j != u && H[j] != REGIME_UNBOUND && near_pt(u, j)) unite(node[u], node[j]);
        }

        // Метки: компоненты с FP — по возрастанию наименьшего индекса FP, затем остальные — по
        // наименьшему индексу Osc. Первая встреча корня при обходе по возрастанию индекса и есть
        // этот наименьший индекс.
        std::vector<int> labelOf((size_t)nodes, 0);
        for (size_t i = 0; i < n; ++i)
            if (H[i] == REGIME_FIXED_POINT) { int& l = labelOf[(size_t)find(node[i])]; if (l == 0) l = --n_neg; }
        for (size_t i = 0; i < n; ++i)
            if (H[i] == REGIME_OSCILLATION) { int& l = labelOf[(size_t)find(node[i])]; if (l == 0) l = ++n_pos; }
        for (size_t i = 0; i < n; ++i)
            labels[i] = node[i] >= 0 ? labelOf[(size_t)find(node[i])] : 0;
        if (progress) progress->store(1.0f, std::memory_order_relaxed);
        return true;
    }

bool ParametricEngine::Impl::basins_dbscan_grid(const numb* d_X, const numb* d_Y, const int* d_helpful, int* d_labels,
                            size_t total_cells, double eps,
                            const std::shared_ptr<std::atomic<bool>>& cancel,
                            const std::shared_ptr<std::atomic<float>>& progress,
                            int& n_pos, int& n_neg, std::string& err) {
        err.clear();
        const size_t n = total_cells;
        std::vector<numb> X(n), Y(n);
        std::vector<int>  H(n), labels(n);
        cudaError_t ce = cudaMemcpy(X.data(), d_X, n * sizeof(numb), cudaMemcpyDeviceToHost);
        if (ce == cudaSuccess) ce = cudaMemcpy(Y.data(), d_Y, n * sizeof(numb), cudaMemcpyDeviceToHost);
        if (ce == cudaSuccess) ce = cudaMemcpy(H.data(), d_helpful, n * sizeof(int), cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) { err = std::string("CUDA memcpy dbscan D2H: ") + cudaGetErrorString(ce); return false; }
        if (!basins_dbscan_host(X.data(), Y.data(), H.data(), n, eps, cancel, progress, labels.data(), n_pos, n_neg))
            return false;
        ce = cudaMemcpy(d_labels, labels.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { err = std::string("CUDA memcpy dbscan labels H2D: ") + cudaGetErrorString(ce); return false; }
        return true;
    }

bool ParametricEngine::Impl::compile_fs_module(const std::string& src_template,
                           const std::string& tag, // ":fs_attr" или ":fs_grid"
                           int amountOfX,
                           const std::string& krs_body,
                           int type_of_synch_v, int error_estim_v, double fs_error_trs_v,
                           const std::vector<const char*>& expr_kernels,
                           ModuleLru<CachedFastSyncModule>& pool,
                           CachedFastSyncModule& slot,
                           std::string& err,
                           bool activate )
    {
        cuCtxSetCurrent(context);
        char trs_buf[64];
        std::snprintf(trs_buf, sizeof(trs_buf), "%.17g", fs_error_trs_v);
        std::string key = hash_key(krs_body, amountOfX) + tag
                        + ":t" + std::to_string(type_of_synch_v)
                        + ":e" + std::to_string(error_estim_v)
                        + ":r" + trs_buf;
        return compile_into(pool, key, slot, activate, [&](CachedFastSyncModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template), "fastsync.cu",
                              { { "{{AMOUNT_OF_X}}",   std::to_string(amountOfX) },
                                { "{{TYPE_OF_SYNCH}}", std::to_string(type_of_synch_v) },
                                { "{{ERROR_ESTIM}}",   std::to_string(error_estim_v) },
                                { "{{FS_ERROR_TRS}}",  std::string(trs_buf) },
                                { "{{KRS_BODY}}",      krs_body } },
                              expr_kernels,
                              mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;

            // Связываем по expr_kernels индексу.
            for (size_t i = 0; i < expr_kernels.size(); ++i) {
                CUfunction f = nullptr;
                if (!module_fn(mod, lowered[i], f, err)) { cuModuleUnload(mod); return false; }
                const std::string sym_name = expr_kernels[i];
                if      (sym_name == "fillFSMasterTrajectory")                   fresh.kernel_fs_fill = f;
                else if (sym_name == "calculateDiscreteModelforFastSynchroCUDA") fresh.kernel_fs_traj = f;
                else if (sym_name == "calculateDiscreteModelICCforFastSynchro")  fresh.kernel_fs_grid = f;
            }
            return true;
        }, err);
    }

bool ParametricEngine::Impl::compile_order_module(bool activate, int amountOfX, int amountOfValues,
                              const std::string& krs_body, const std::string& ref_body,
                              std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело эталона входит в ключ: с ним и без него это РАЗНЫЙ модуль (ветка
        // эталона вырезается препроцессором), да и сам эталонный метод меняется.
        const std::string key = hash_key(krs_body, amountOfX) + ":order:v"
                              + std::to_string(amountOfValues)
                              + ":ref" + std::to_string(std::hash<std::string>{}(ref_body));
        return compile_into(pool_order, key, cached_order, activate, [&](CachedOrderModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            // orderEstimateKernel объявлено extern "C" — в name_exprs не нужно.
            if (!build_module(snapshot_sources(time_prefix(krs_body + ref_body) + src_template_order), "order.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{REF_ENABLED}}",      ref_body.empty() ? "0" : "1" },
                                { "{{KRS_REF_BODY}}",     ref_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "orderEstimateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            // Ядро замера лежит в том же модуле: Performance нужны ОБА прохода,
            // и второй компиляции ради этого быть не должно.
            if (!module_fn(mod, "perfIntegrateKernel", fresh.kernel_perf, err)) { cuModuleUnload(mod); return false; }
            // И ядро областей устойчивости — оттуда же. Отдельного модуля ему не
            // нужно: тело КРС в этом уже лежит, а регистров чужим расчётам оно
            // не стоит, регистровый бюджет считается на ядро.
            if (!module_fn(mod, "stabilityRegionKernel", fresh.kernel_stab, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }

void ParametricEngine::Impl::prewarm_ad_cpu(const AdaptiveRequest& ad, int amountOfX) {
        if (!ad.setup_error.empty()) return;
        AdaptiveCpuModule mod;
        std::vector<KrsCpuDiag> diags;
        (void)mod.compile(ad.rhs, ad.emb, ad.dprep, ad.deval, ad.ctrl_body, amountOfX, ad.prefix + peak_config_defines(), diags);
    }

void ParametricEngine::Impl::prewarm_metrics(const SignalMetricsRequest& req) {
        if (req.use_cpu) {   // Fixed: CPU-шаг собирает cl.exe при Run; адаптивный — общая CPU-DLL
            if (req.adaptive.enabled) prewarm_ad_cpu(req.adaptive, req.amountOfX);
            return;
        }
        MetricsAxes ax;
        if (!metrics_axes(req, ax).empty()) return;
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        if (req.adaptive.enabled)
            compile_metrics_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, ax.par_or_var, err, false);
        else
            compile_metrics_if_needed(req.krs_body, req.amountOfX, ax.par_or_var, err, false, req.minmax_interp);
    }
