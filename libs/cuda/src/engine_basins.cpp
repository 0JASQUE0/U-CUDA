#include "engine_impl.h"
using namespace ucuda::cuda_detail;

BasinsResult ParametricEngine::Impl::run_basins(const BasinsRequest& req) {
        BasinsResult res;
        auto fail = [&](const std::string& msg) -> BasinsResult& { res.error = msg; return res; };
        if (req.adaptive.enabled && !req.adaptive.setup_error.empty()) return fail(req.adaptive.setup_error);

        if (req.krs_body.empty())                                  return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)   return fail("amountOfX out of the allowed range");
        if ((int)req.initial_conditions.size() != req.amountOfX)   return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)      return fail("too many base_values");
        if (req.axis_x_var < 0 || req.axis_x_var >= req.amountOfX) return fail("axis_x_var out of range");
        if (req.axis_y_var < 0 || req.axis_y_var >= req.amountOfX) return fail("axis_y_var out of range");
        if (req.axis_x_var == req.axis_y_var)                      return fail("axis_x_var == axis_y_var (pick different variables)");
        if (dae_z_axis(req.krs_body, req.axis_x_var) || dae_z_axis(req.krs_body, req.axis_y_var)) return fail(kDaeZAxisMsg);
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX) return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        if (req.eps_dbscan <= 0.0)  return fail("eps_dbscan must be > 0");
        if (req.axis_x_symlog || req.axis_y_symlog) {
            // m = n_pts/2 узлов на полуось, и формула делит на m - 1. При нечётном n_pts легаси
            // клал лишний узел за пределы [lo, hi], поэтому нечётное не принимаем.
            if (req.n_pts < 4 || (req.n_pts % 2) != 0)
                return fail("±log grid needs an even Resolution >= 4");
            if (req.axis_x_symlog && !(req.axis_x_lo > 0.0 && req.axis_x_hi > 0.0))
                return fail("±log X: lo and hi are |x| bounds and must be > 0");
            if (req.axis_y_symlog && !(req.axis_y_lo > 0.0 && req.axis_y_hi > 0.0))
                return fail("±log Y: lo and hi are |y| bounds and must be > 0");
        }
        const int logAxisMask = (req.axis_x_symlog ? (1 << (UCUDA_SYMLOG_SHIFT + 0)) : 0)
                              | (req.axis_y_symlog ? (1 << (UCUDA_SYMLOG_SHIFT + 1)) : 0);

        // Время фаз печатается в консоль в конце расчёта: компиляция модуля, счёт
        // (траектории + признаки), DBSCAN и весь run_basins.
        using bas_clock = std::chrono::steady_clock;
        const auto bas_t0 = bas_clock::now();
        auto bas_secs = [](bas_clock::time_point a, bas_clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        const bool ad = req.adaptive.enabled;   // адаптивный шаг: ядра из адаптивного модуля
        if (ad) {
            if (!compile_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, 0, err)) return fail(err);
        }
        else if (!compile_basins_if_needed(req.krs_body, req.amountOfX, err)) return fail(err);
        const auto bas_t_compiled = bas_clock::now();

        const int    nPts                       = req.n_pts;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)req.axis_x_lo, (numb)req.axis_x_hi,
                                                    (numb)req.axis_y_lo, (numb)req.axis_y_hi };
        int    indicesOfMutVars[2]              = { req.axis_x_var, req.axis_y_var };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const double tMax                       = req.t_max;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const double eps_dbscan                 = req.eps_dbscan;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        // blockSize_setup здесь больше нет: ширина запуска берётся из настройки
        // (launch_block_size), а полом бюджета памяти у бассейнов он не служил.
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0");

        size_t total_cells = (size_t)nPts * (size_t)nPts;

        // По памяти чанк ничем не ограничен: буферы, росшие вместе с ним,
        // ушли вместе с траекторией, а выходные массивы выделяются на всю сетку.
        size_t nPtsLimiter = total_cells;

        size_t originalNPtsLimiter = nPtsLimiter;

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        int*    d_helpfulArray      = nullptr;
        int*    d_dbscanResult      = nullptr;
        RunSignals sig;   // прогресс и отмена в mapped-памяти
        numb* d_avgPeaks          = nullptr;
        numb* d_avgIntervals      = nullptr;
        AdDeviceArgs adArgs;                  // адаптивный шаг: параметры, оси, статистика

        auto cleanup = [&]() {
            adArgs.release();
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_helpfulArray)      cudaFree(d_helpfulArray);
            if (d_dbscanResult)      cudaFree(d_dbscanResult);
            sig.release();
            if (d_avgPeaks)          cudaFree(d_avgPeaks);
            if (d_avgIntervals)      cudaFree(d_avgIntervals);
        };

        #define BAS_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define BAS_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        BAS_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                           "cudaMalloc d_ranges");
        BAS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                              "cudaMalloc d_indicesOfMutVars");
        BAS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),           "cudaMalloc d_initialConditions");
        BAS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                      "cudaMalloc d_values");
        BAS_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                    "cudaMalloc d_amountOfPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_helpfulArray,      total_cells * sizeof(int)),                                    "cudaMalloc d_helpfulArray");
        BAS_CHECK(cudaMalloc((void**)&d_dbscanResult,      total_cells * sizeof(int)),                                    "cudaMalloc d_dbscanResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        if (ad) {
            const int kinds[2] = { kAdAxisSystem, kAdAxisSystem };
            if (!adArgs.alloc(req.adaptive, kinds, total_cells, res.error)) { cleanup(); return res; }
        }
        const size_t stepsPerCell   = ad ? kAdProgressUnits
                                         : amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerCell);
        const double ticksPerCell   = (double)(stepsPerCell / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerCell;
        BAS_CHECK(cudaMalloc((void**)&d_avgPeaks,          total_cells * sizeof(numb)),                                 "cudaMalloc d_avgPeaks");
        BAS_CHECK(cudaMalloc((void**)&d_avgIntervals,      total_cells * sizeof(numb)),                                 "cudaMalloc d_avgIntervals");

        BAS_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                                  cudaMemcpyHostToDevice), "memcpy d_ranges");
        BAS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                                     cudaMemcpyHostToDevice), "memcpy d_indices");
        BAS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_ic");
        BAS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),             cudaMemcpyHostToDevice), "memcpy d_values");
        BAS_CHECK(cudaMemset(d_dbscanResult, 0, total_cells * sizeof(int)),  "memset d_dbscanResult");
        BAS_CHECK(cudaMemset(d_helpfulArray, 0, total_cells * sizeof(int)),  "memset d_helpfulArray");
        BAS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        // Snapshot CSV-relevant fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.eps_dbscan    = eps_dbscan;
        res.snapshot.writableVar   = req.writable_var;
        res.snapshot.axis_x_var    = req.axis_x_var;
        res.snapshot.axis_y_var    = req.axis_y_var;
        res.snapshot.axis_x_lo     = req.axis_x_lo;
        res.snapshot.axis_x_hi     = req.axis_x_hi;
        res.snapshot.axis_y_lo     = req.axis_y_lo;
        res.snapshot.axis_y_hi     = req.axis_y_hi;
        res.snapshot.axis_x_symlog = req.axis_x_symlog;
        res.snapshot.axis_y_symlog = req.axis_y_symlog;
        res.snapshot.n_pts         = nPts;
        res.snapshot.feature1      = req.feature1;
        res.snapshot.feature2      = req.feature2;
        res.snapshot.mult1         = (double)req.mult1;
        res.snapshot.mult2         = (double)req.mult2;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_basins_config(cfg, res.snapshot);
        }

        // Pre-write the 2-line ranges header into each of the 4 data files.
        // Body is appended after compute (see the post-loop block below).
        if (!OUT_FILE_PATH.empty()) {
            auto write_ranges = [&](const std::string& path) {
                std::ofstream o(path);
                data_export::write_basins_ranges(o, res.snapshot);
            };
            write_ranges(OUT_FILE_PATH);
            write_ranges(OUT_FILE_PATH + "_1.csv");
            write_ranges(OUT_FILE_PATH + "_2.csv");
            write_ranges(OUT_FILE_PATH + "_3.csv");
        }

        // 1. Цикл по chunk'ам: traj-kernel + avg-peak-kernel
        // Two-phase progress: phase 1 here (sim), phase 2 (DBSCAN) below. Each
        // phase reports its own 0..1 fraction; GUI shows the phase label.
        if (req.progress_phase) req.progress_phase->store(1, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            BAS_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerCell / ticksTotal), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: ceil(48K / ((N + nValues) * sizeof(numb))), cap — настройка
            // (Settings -> GPU launch); формула остаётся вторым потолком, см. run_lle_1d.
            const size_t sharedPerThread = (size_t)(amountOfInitialConditions + amountOfValues) * sizeof(numb);
            int blockSize = (int)std::ceil((1024.0 * 48.0) / (double)sharedPerThread);
            if (blockSize < 1)                blockSize = 1;
            const int blockSizeCap = launch_block_size(sharedPerThread);
            if (blockSize > blockSizeCap)     blockSize = blockSizeCap;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            // calculateDiscreteModelAvgPeaksCUDA: интегрирование и фичи в одном ядре.
            // h-свипа и лог-осей у бассейнов нет в запросе вовсе, ядро зашивает их
            // константами -1 / 0 — ровно тем, что сюда передавалась раздельная пара.
            int    nPts_arg                  = nPts;
            int    nPtsLimiter_int           = (int)cur_limiter;
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            int    amountOfIC_int            = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = req.writable_var;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = false;   // compile-time par_or_var=0 в шаблоне
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int  feature1_int = req.feature1;
            int  feature2_int = req.feature2;
            numb mult1_v      = req.mult1;
            numb mult2_v      = req.mult2;
            // Сигналы пока выключены: прогресс и Cancel у бассейнов ещё по чанкам.
            int*   d_cancel_arg              = sig.cancelArg();
            int*   d_progress_arg            = sig.progressArg();
            int    progressStride_arg        = progressStride;
            sig.resetTicks();

            // Offset-указатели: чанк пишет в свою часть общей сетки.
            int*  d_helpful_chunk    = d_helpfulArray  + iter * originalNPtsLimiter;
            numb* d_avg_peak_chunk   = d_avgPeaks      + iter * originalNPtsLimiter;
            numb* d_avg_interv_chunk = d_avgIntervals  + iter * originalNPtsLimiter;

            int logAxisMask_arg = logAxisMask;
            void* args_basins[] = {
                &nPts_arg, &nPtsLimiter_int, &amountOfCalculatedPoints,
                &amountOfPointsForSkip_s, &dimension_arg, &d_ranges, &h_arg,
                &d_indicesOfMutVars, &d_initialConditions, &amountOfIC_int,
                &d_values, &amountOfValues_int, &amountOfIterations_arg,
                &preScaller_int, &writableVar_int, &maxValue_arg,
                &d_avg_peak_chunk, &d_avg_interv_chunk, &d_helpful_chunk,
                &par_or_var_arg, &transientTime_arg, &tMax_arg,
                &feature1_int, &feature2_int, &mult1_v, &mult2_v,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };
            unsigned int shared_traj = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);
            if (ad) {
                UcudaAdaptParams* d_adp_arg = adArgs.params;
                int*   d_axis_arg        = adArgs.axis;
                numb   tolRatio_arg      = (numb)req.adaptive.tol_ratio;
                numb   dtOut_arg         = h;
                int    raw_arg           = req.adaptive.raw_nodes ? 1 : 0;
                int    interp_arg        = req.adaptive.peak_interp;
                size_t progressUnits_arg = kAdProgressUnits;
                numb*  d_adStats_arg     = adArgs.stats + iter * originalNPtsLimiter * 4;
                void* args_ad[] = {
                    &nPts_arg, &nPtsLimiter_int, &amountOfCalculatedPoints, &dimension_arg,
                    &d_ranges, &d_indicesOfMutVars, &d_initialConditions, &amountOfIC_int,
                    &d_values, &amountOfValues_int, &writableVar_int, &maxValue_arg,
                    &d_avg_peak_chunk, &d_avg_interv_chunk, &d_helpful_chunk,
                    &feature1_int, &feature2_int, &mult1_v, &mult2_v,
                    ad_param_arg(req.adaptive, &d_adp_arg), &d_axis_arg, &tolRatio_arg, &transientTime_arg, &tMax_arg, &dtOut_arg,
                    &preScaller_int, &amountOfIterations_arg, &raw_arg, &interp_arg,
                    &d_cancel_arg, &d_progress_arg, &progressStride_arg,
                    &progressUnits_arg, &d_adStats_arg, &logAxisMask_arg
                };
                BAS_CHECK_CU(cuLaunchKernel(cached_ad.kernel_avg,
                                            gridSize, 1, 1, blockSize, 1, 1,
                                            shared_traj, nullptr, args_ad, nullptr),
                             "cuLaunchKernel(basins adaptive)");
            }
            else
            BAS_CHECK_CU(cuLaunchKernel(cached_basins.kernel_fused,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared_traj, nullptr, args_basins, nullptr),
                         "cuLaunchKernel(basins traj+features)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerCell,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            BAS_CHECK(cudaDeviceSynchronize(), "sync after traj+features");
            BAS_CANCEL_CHECK();
        }

        if (ad) {
            std::vector<numb> h_adStats(total_cells * 4);
            BAS_CHECK(cudaMemcpy(h_adStats.data(), adArgs.stats, total_cells * 4 * sizeof(numb),
                                 cudaMemcpyDeviceToHost), "memcpy adaptive stats");
            res.ad_stats.assign(h_adStats.begin(), h_adStats.end());
        }

        const auto bas_t_sim = bas_clock::now();

        // 2. DBSCAN — на CPU сеткой в пространстве признаков (basins_dbscan_grid). Прогресс фазы —
        // доля обработанных ячеек сетки.
        if (req.progress_phase) req.progress_phase->store(2, std::memory_order_relaxed);
        if (req.progress)       req.progress->store(0.0f, std::memory_order_relaxed);
        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        if (!basins_dbscan_grid(d_avgPeaks, d_avgIntervals, d_helpfulArray, d_dbscanResult,
                                total_cells, eps_dbscan, req.cancel, req.progress,
                                amountOfClusters, amountOfNegativeClusters, res.error)) {
            if (res.error.empty()) { res.cancelled = true; res.error = "Cancelled by user"; }
            cleanup(); return res;
        }
        const auto bas_t_dbscan = bas_clock::now();

        // 3. D2H результаты
        res.n_pts       = nPts;
        res.axis_x_lo   = req.axis_x_lo;
        res.axis_x_hi   = req.axis_x_hi;
        res.axis_y_lo   = req.axis_y_lo;
        res.axis_y_hi   = req.axis_y_hi;
        res.axis_x_var  = req.axis_x_var;
        res.axis_y_var  = req.axis_y_var;
        res.axis_x_symlog = req.axis_x_symlog;
        res.axis_y_symlog = req.axis_y_symlog;
        res.basin_idx.assign(total_cells, 0);
        res.avg_peaks.assign(total_cells, 0.0);
        res.avg_intervals.assign(total_cells, 0.0);
        res.helpful_array.assign(total_cells, 0);

        BAS_CHECK(cudaMemcpy(res.basin_idx.data(),     d_dbscanResult, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy basin_idx");
        {   // numb на устройстве -> double в Result.
            std::vector<numb> ap_(total_cells), ai_(total_cells);
            BAS_CHECK(cudaMemcpy(ap_.data(), d_avgPeaks,     total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_peaks");
            BAS_CHECK(cudaMemcpy(ai_.data(), d_avgIntervals, total_cells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy avg_intervals");
            res.avg_peaks.assign(ap_.begin(), ap_.end());
            res.avg_intervals.assign(ai_.begin(), ai_.end());
        }
        BAS_CHECK(cudaMemcpy(res.helpful_array.data(), d_helpfulArray, total_cells * sizeof(int),    cudaMemcpyDeviceToHost), "memcpy helpful_array");

        // 4. Сводки + CSV
        res.n_clusters       = amountOfClusters;
        res.min_cluster_idx  = amountOfNegativeClusters;

        // Per-field min/max (исключая 999/-999/NaN).
        auto compute_minmax = [&](const std::vector<double>& v, double& mn, double& mx) {
            mn = std::numeric_limits<double>::infinity();
            mx = -std::numeric_limits<double>::infinity();
            for (double x : v) {
                if (!std::isfinite(x)) continue;
                if (x == 999.0 || x == -999.0) continue;
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            if (!std::isfinite(mn) || !std::isfinite(mx)) { mn = 0.0; mx = 0.0; }
        };
        compute_minmax(res.avg_peaks,     res.avg_peaks_min,     res.avg_peaks_max);
        compute_minmax(res.avg_intervals, res.avg_intervals_min, res.avg_intervals_max);

        // Append the n_pts × n_pts grid for each of the 4 fields, after the
        // 2-line ranges header already written above. Layout matches the
        // shared writers data_export uses for the GUI right-click export.
        if (!OUT_FILE_PATH.empty()) {
            auto append_int = [&](const std::string& path, const int* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_int(o, nPts, data);
            };
            auto append_double = [&](const std::string& path, const double* data) {
                std::ofstream o(path, std::ios::app);
                if (!o.is_open()) return;
                o << std::setprecision(set_precision);
                data_export::write_basins_grid_double(o, nPts, data);
            };
            append_int   (OUT_FILE_PATH,             res.basin_idx.data());
            append_double(OUT_FILE_PATH + "_1.csv",  res.avg_peaks.data());
            append_double(OUT_FILE_PATH + "_2.csv",  res.avg_intervals.data());
            append_int   (OUT_FILE_PATH + "_3.csv",  res.helpful_array.data());
        }

        cleanup();
        #undef BAS_CHECK
        #undef BAS_CHECK_CU
        std::fprintf(stderr, "[basins] %dx%d %s: compile %.3f s, sim %.3f s, DBSCAN %.3f s, total %.3f s (clusters +%d/%d)\n",
                     nPts, nPts, ad ? "adaptive" : "fixed",
                     bas_secs(bas_t0, bas_t_compiled), bas_secs(bas_t_compiled, bas_t_sim),
                     bas_secs(bas_t_sim, bas_t_dbscan), bas_secs(bas_t0, bas_clock::now()),
                     amountOfClusters, amountOfNegativeClusters);
        res.ok = true;
        return res;
    }

BasinsReclusterResult ParametricEngine::Impl::run_basins_recluster(const BasinsReclusterRequest& req) {
        BasinsReclusterResult res;
        auto fail = [&](const std::string& msg) -> BasinsReclusterResult& {
            res.error = msg; return res;
        };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of the allowed range");
        if (req.n_pts <= 0)                                      return fail("n_pts must be > 0");
        if (req.eps_dbscan <= 0.0)                               return fail("eps_dbscan must be > 0");
        const size_t total_cells = (size_t)req.n_pts * (size_t)req.n_pts;
        if (req.avg_peaks.size()     != total_cells)             return fail("avg_peaks: size does not match n_pts²");
        if (req.avg_intervals.size() != total_cells)             return fail("avg_intervals: size does not match n_pts²");
        if (req.helpful_array.size() != total_cells)             return fail("helpful_array: size does not match n_pts²");

        // DBSCAN считается на CPU (basins_dbscan_host): ни GPU, ни модуль бассейнов здесь не нужны.
        // Признаки — в numb, как их видел бы расчёт на устройстве.
        const auto brc_t0 = std::chrono::steady_clock::now();
        const std::vector<numb> X = to_numb(req.avg_peaks);
        const std::vector<numb> Y = to_numb(req.avg_intervals);
        if (req.progress) req.progress->store(0.0f, std::memory_order_relaxed);
        int amountOfClusters         = 0;
        int amountOfNegativeClusters = 0;
        res.basin_idx.assign(total_cells, 0);
        const auto brc_t_db0 = std::chrono::steady_clock::now();
        if (!basins_dbscan_host(X.data(), Y.data(), req.helpful_array.data(), total_cells, req.eps_dbscan,
                                req.cancel, req.progress, res.basin_idx.data(),
                                amountOfClusters, amountOfNegativeClusters)) {
            res.basin_idx.clear();
            res.cancelled = true; res.error = "Cancelled by user";
            return res;
        }
        const auto brc_t_db1 = std::chrono::steady_clock::now();
        res.n_pts = req.n_pts;
        res.n_clusters      = amountOfClusters;
        res.min_cluster_idx = amountOfNegativeClusters;

        std::fprintf(stderr, "[basins recluster] %dx%d: DBSCAN %.3f s, total %.3f s (clusters +%d/%d)\n",
                     req.n_pts, req.n_pts,
                     std::chrono::duration<double>(brc_t_db1 - brc_t_db0).count(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - brc_t0).count(),
                     amountOfClusters, amountOfNegativeClusters);
        res.ok = true;
        return res;
    }

void ParametricEngine::Impl::prewarm_basins(const BasinsRequest& req) {
        std::string err;
        if (!ensure_init(err)) return;
        cuCtxSetCurrent(context);
        if (req.adaptive.enabled) compile_ad_if_needed(req.adaptive, req.krs_body, req.amountOfX, 0, err, false);
        else compile_basins_if_needed(req.krs_body, req.amountOfX, err, false);
    }
