#include "engine_impl.h"



    // Ждём, пока отданная в stream работа закончится, обновляя прогресс из тиков
    // и передавая Cancel в ядро. В теле цикла не должно быть вызовов CUDA, кроме
    // cudaStreamQuery — см. комментарий к RunSignals.
    bool ParametricEngine::Impl::wait_with_signals(cudaStream_t stream, RunSignals& sig,
                           const std::shared_ptr<std::atomic<bool>>& cancel,
                           const std::shared_ptr<std::atomic<float>>& progress,
                           double ticksBefore, double ticksTotal, std::string& err) const
    {
        bool cancelSent = false;
        for (;;) {
            cudaError_t q = cudaStreamQuery(stream);
            if (q == cudaSuccess) break;
            if (q != cudaErrorNotReady) {
                err = std::string("CUDA stream query: ") + cudaGetErrorString(q);
                return false;
            }
            if (progress && ticksTotal > 0.0) {
                double f = (ticksBefore + (double)sig.ticks()) / ticksTotal;
                if (f > 1.0) f = 1.0;
                progress->store((float)f, std::memory_order_relaxed);
            }
            if (!cancelSent && cancel && cancel->load(std::memory_order_relaxed)) {
                sig.raiseCancel();
                cancelSent = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        return true;
    }


    // run_ls_1d — порт NonLinAnal::LS1D той же стратегией, что run_bif1d / run_lle_1d.
    // Per-system результат — вектор длины amountOfX (один Ляпунов на переменную).
    // Memory budget на поток у LS значительно больше: shared = (3*N + 2*N^2 + nValues) *
    // sizeof(numb) * blockSize, и blockSize подбирается из 32K shared-лимита (как в NonLinAnal).
    LS1DResult ParametricEngine::Impl::run_ls_1d(const LS1DRequest& req) {
        LS1DResult res;
        auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                   return fail("krs_body пуст");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX вне [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("base_values слишком много");
        if (req.sweep_over_h) {
            if (req.param_lo <= 0.0 || req.param_hi <= 0.0)
                return fail("h lo/hi должны быть > 0 при sweep_over_h");
        } else if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index вне диапазона");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index вне диапазона");
        }
        if (req.n_pts <= 0)         return fail("n_pts должно быть > 0");
        if (req.h <= 0.0)           return fail("h должно быть > 0");
        if (req.t_max <= 0.0)       return fail("t_max должно быть > 0");
        if (req.transient_time < 0) return fail("transient_time должно быть >= 0");
        if (req.NT <= 0.0)          return fail("NT должно быть > 0");
        if (req.eps <= 0.0)         return fail("eps должно быть > 0");
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0");

        // CPU-ветки — до ensure_init, CUDA им не нужна. Ограничения те же, что
        // и у LLE (см. run_lle_1d).
        if (req.continuation) {
            if (req.sweep_over_var) return fail("continuation требует param-sweep, не IC-sweep");
            return req.use_cpu ? run_ls1d_cpu(req, /*continuation*/ true)
                               : run_ls1d_continuation_gpu(req);
        }
        if (req.use_cpu) {
            if (req.sweep_over_var) return fail("CPU-ветка поддерживает только param-sweep");
            return run_ls1d_cpu(req, /*continuation*/ false);
        }

        // dt-sweep: см. run_lle_1d -- t_max/transient_time/NT фиксированы, число
        // шагов пересчитывается на GPU из h per-thread.
        const int hSweepAxis = req.sweep_over_h ? 0 : -1;
        const int logAxisMask = req.log_scale ? 1 : 0;

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
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        // Sweep target: param-sweep → 1-based в localValues; IC-sweep → 0-based в localX.
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        // Memory budget — мирор NonLinAnal LS1D:2719-2727 (агрессивно делит /16,
        // т.к. per-system memory ~ N).
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lsResult (N numb): траектория
        // не хранится. Прежняя формула умножала это на длину блока и брала
        // 1/16 свободной памяти — вместе это резало чанк без причины.
        size_t perSystemBytes = sizeof(numb) * (size_t)amountOfInitialConditions;
        if (perSystemBytes == 0) perSystemBytes = sizeof(numb);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)            nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)  nPtsLimiter = (size_t)nPts;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult хранит nPtsLimiter × N row-major.
        std::vector<numb> h_lsResult(nPtsLimiter * (size_t)amountOfInitialConditions);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lsResult          = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };
            sig.release();

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
        #define LS_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LS_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                                          "cudaMalloc d_ranges");
        LS_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                                             "cudaMalloc d_indicesOfMutVars");
        LS_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),                          "cudaMalloc d_initialConditions");
        LS_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                                     "cudaMalloc d_values");
        LS_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(numb)),            "cudaMalloc d_lsResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        LS_CHECK(cudaMemcpy(d_ranges,            ranges,            2 * sizeof(numb),                                 cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  1 * sizeof(int),                                    cudaMemcpyHostToDevice), "memcpy d_indices");
        LS_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        LS_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        LS_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        // Snapshot CSV-relevant request fields for GUI right-click export.
        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_ls1d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        res.n_pts       = nPts;
        res.n_exponents = amountOfInitialConditions;
        res.param_lo    = ranges[0];
        res.param_hi    = ranges[1];
        res.spectrum.assign(nPts, std::vector<double>(amountOfInitialConditions, 0.0));
        res.flags.assign(nPts, 0);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LS_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // blockSize: 32K shared / per-thread, cap=32 (порт LS1D:2821-2824)
            int blockSizeMax = (int)(32000 / ((double)(3 * amountOfInitialConditions
                                + 2 * amountOfInitialConditions * amountOfInitialConditions
                                + amountOfValues) * (double)sizeof(numb)));
            int blockSize = blockSizeMax;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)nPtsLimiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 1;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = amountOfInitialConditions;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult, &hSweepAxis_arg, &transientTime_arg, &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            // Shared = (3N + 2N² + nValues) * sizeof(numb) * blockSize
            unsigned int shared = (unsigned int)((3 * amountOfInitialConditions
                                                 + 2 * amountOfInitialConditions * amountOfInitialConditions
                                                 + amountOfValues)
                                                * sizeof(numb) * blockSize);

            LS_CHECK_CU(cuLaunchKernel(cached_ls.kernel_ls,
                                       gridSize, 1, 1, blockSize, 1, 1,
                                       shared, nullptr, args, nullptr),
                        "cuLaunchKernel(ls)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LS_CHECK(cudaDeviceSynchronize(), "sync after ls");

            LS_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                nPtsLimiter * (size_t)amountOfInitialConditions * sizeof(numb),
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
                double param_val  = req.log_scale ? getValueByIdx_log_local(global_idx, nPts, ranges[0], ranges[1])
                                                   : getValueByIdx_local(global_idx, nPts, ranges[0], ranges[1]);

                // первая экспонента используется как ground-truth для флага
                double first = h_lsResult[k * (size_t)amountOfInitialConditions + 0];
                // 999/-999 — kernel-sentinel расходимости; FP LS не различает.
                res.flags[global_idx] = (first == 999.0 || first == -999.0) ? REGIME_UNBOUND
                                                                            : REGIME_OSCILLATION;

                auto& row = res.spectrum[global_idx];
                for (int j = 0; j < amountOfInitialConditions; ++j) {
                    row[j] = h_lsResult[k * (size_t)amountOfInitialConditions + j];
                }

                if (out.is_open())
                    data_export::write_ls1d_row(out, param_val, row.data(),
                                                amountOfInitialConditions);
            }
            if (out.is_open()) out.close();
        }

        cleanup();
        #undef LS_CHECK
        #undef LS_CHECK_CU
        res.ok = true;
        return res;
    }


    // run_ls_2d — спектр Ляпунова на сетке n_pts × n_pts: гибрид run_ls_1d (per-system буфер на N
    // экспонент) и run_lle_2d (swap_xy, chunking по ячейкам). Kernel возвращает N значений на
    // ячейку; D2H копирует cur_limiter * N doubles, host распаковывает по плоскостям —
    // values[k*n*n + iy*n + ix] = k-я экспонента в ячейке (ix, iy).
    LS2DResult ParametricEngine::Impl::run_ls_2d(const LS2DRequest& req) {
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
            return p1based >= 0 && p1based < (int)req.base_values.size();
        };
        auto check_var = [&](int v0based) -> bool {
            return v0based >= 0 && v0based < req.amountOfX;
        };

        int par_or_var;
        int  idx_axis_x, idx_axis_y;
        double ranges_lo_x, ranges_hi_x, ranges_lo_y, ranges_hi_y;
        bool swap_xy = false;
        int  hSweepAxis = -1;
        bool log_axis_x = req.log_scale, log_axis_y = req.log_scale_2;  // см. run_lle_2d

        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось X)");
        if (req.log_scale_2 && !(req.param_lo_2 > 0.0 && req.param_hi_2 > 0.0))
            return fail("log scale требует param lo/hi > 0 (ось Y)");

        if (req.sweep_over_h && req.sweep_over_h_2)
            return fail("sweep_over_h и sweep_over_h_2 не могут быть true одновременно");

        if (req.sweep_over_h || req.sweep_over_h_2) {
            // См. run_lle_2d -- симметрично по слотам, swap_xy не нужен.
            hSweepAxis = req.sweep_over_h ? 0 : 1;
            bool other_is_var = req.sweep_over_h ? req.sweep_over_var_2 : req.sweep_over_var;
            par_or_var = other_is_var ? 0 : 1;

            if (req.sweep_over_h) {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index_2))   return fail("param_index_2 (ось Y) вне диапазона");
                    idx_axis_y = req.param_index_2;
                } else {
                    if (!check_var(req.var_sweep_index_2)) return fail("var_sweep_index_2 (ось Y) вне диапазона");
                    idx_axis_y = req.var_sweep_index_2;
                }
                idx_axis_x = 0;
            } else {
                if (par_or_var == 1) {
                    if (!check_param(req.param_index))     return fail("param_index (ось X) вне диапазона");
                    idx_axis_x = req.param_index;
                } else {
                    if (!check_var(req.var_sweep_index))   return fail("var_sweep_index (ось X) вне диапазона");
                    idx_axis_x = req.var_sweep_index;
                }
                idx_axis_y = 0;
            }
            ranges_lo_x = req.param_lo;   ranges_hi_x = req.param_hi;
            ranges_lo_y = req.param_lo_2; ranges_hi_y = req.param_hi_2;
        } else if (req.sweep_over_var == req.sweep_over_var_2) {
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
            log_axis_x = req.log_scale_2; log_axis_y = req.log_scale;
        }

        int logAxisMask = (log_axis_x ? 1 : 0) | (log_axis_y ? 2 : 0);

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
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[4]                        = { (numb)ranges_lo_x, (numb)ranges_hi_x,
                                                    (numb)ranges_lo_y, (numb)ranges_hi_y };
        int    indicesOfMutVars[2]              = { idx_axis_x, idx_axis_y };
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / NT);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);

        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max / NT слишком малы)");

        size_t total_cells = (size_t)nPts * (size_t)nPts;
        const int N = amountOfInitialConditions;

        // Memory budget — мирор run_ls_1d (агрессивно делит /16, т.к. per-system
        // память ~N). total_cells заменяет nPts.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.9, freeMemory)) return fail("cudaMemGetInfo failed");

        // На точку реально выделяется ТОЛЬКО d_lsResult (N numb): траектория
        // не хранится. Прежняя формула умножала это на длину блока и брала
        // 1/16 свободной памяти — вместе это резало чанк без причины.
        size_t perSystemBytes = sizeof(numb) * (size_t)N;
        if (perSystemBytes == 0) perSystemBytes = sizeof(numb);
        size_t nPtsLimiter = freeMemory / perSystemBytes;
        if (nPtsLimiter == 0)             nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > total_cells)    nPtsLimiter = total_cells;
        size_t originalNPtsLimiter = nPtsLimiter;

        // h_lsResult — nPtsLimiter × N row-major (как в run_ls_1d).
        std::vector<numb> h_lsResult(nPtsLimiter * (size_t)N);

        numb* d_ranges            = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        numb* d_lsResult          = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_ranges)            cudaFree(d_ranges);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_lsResult)          cudaFree(d_lsResult);
        };
            sig.release();

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
        #define LS2_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        LS2_CHECK(cudaMalloc((void**)&d_ranges,            4 * sizeof(numb)),                                "cudaMalloc d_ranges");
        LS2_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  2 * sizeof(int)),                                   "cudaMalloc d_indicesOfMutVars");
        LS2_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)N * sizeof(numb)),                        "cudaMalloc d_initialConditions");
        LS2_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),           "cudaMalloc d_values");
        LS2_CHECK(cudaMalloc((void**)&d_lsResult,          nPtsLimiter * (size_t)N * sizeof(numb)),          "cudaMalloc d_lsResult");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        // Шагов на точку: транзиент плюс NT-блоки по NT/h шагов.
        // Тики ставит только ведущая траектория (см. ядро), поэтому копии не считаем.
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock * steps_from_time_size_t(NT, h);
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)total_cells * ticksPerPoint;

        LS2_CHECK(cudaMemcpy(d_ranges,            ranges,            4 * sizeof(numb),                       cudaMemcpyHostToDevice), "memcpy d_ranges");
        LS2_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,  2 * sizeof(int),                          cudaMemcpyHostToDevice), "memcpy d_indices");
        LS2_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)N * sizeof(numb),               cudaMemcpyHostToDevice), "memcpy d_ic");
        LS2_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),  cudaMemcpyHostToDevice), "memcpy d_values");
        LS2_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)total_cells / (double)nPtsLimiter);

        res.n_pts       = nPts;
        res.n_exponents = N;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.param_lo_2  = req.param_lo_2;
        res.param_hi_2  = req.param_hi_2;
        res.values.assign((size_t)N * total_cells, 0.0);
        res.flags.assign(total_cells, 0);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + N);
        res.snapshot.par_or_var    = par_or_var;
        res.snapshot.tMax          = tMax;
        res.snapshot.NT            = NT;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.eps           = eps;
        res.snapshot.indexOfMutVar  = req.sweep_over_var   ? req.var_sweep_index
                                                           : req.param_index;
        res.snapshot.indexOfMutVar2 = req.sweep_over_var_2 ? req.var_sweep_index_2
                                                           : req.param_index_2;
        res.snapshot.range1_lo = req.param_lo;   res.snapshot.range1_hi = req.param_hi;
        res.snapshot.range2_lo = req.param_lo_2; res.snapshot.range2_hi = req.param_hi_2;
        res.snapshot.n_pts       = nPts;
        res.snapshot.n_exponents = N;

        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_ls2d_config(cfg, res.snapshot);
            std::ofstream trunc(OUT_FILE_PATH); trunc.close();
        }

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            LS2_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            size_t cur_limiter = originalNPtsLimiter;
            if (iter == amountOfIteration - 1)
                cur_limiter = total_cells - (originalNPtsLimiter * iter);

            // blockSize: тот же расчёт что в run_ls_1d (32K shared / per-thread).
            int blockSizeMax = (int)(32000 / ((double)(3 * N + 2 * N * N + amountOfValues)
                                              * (double)sizeof(numb)));
            int blockSize = blockSizeMax;
            if (blockSize > blockSize_setup) blockSize = blockSize_setup;
            if (blockSize < 1)               blockSize = 1;
            int gridSize = (int)((cur_limiter + blockSize - 1) / blockSize);

            int    nPts_arg                  = nPts;
            int    nPtsLimiter_arg           = (int)cur_limiter;
            numb NT_arg                    = NT;
            numb tMax_arg                  = tMax;
            int    sizeOfBlock_arg           = amountOfPointsInBlock;
            int    amountOfCalculatedPoints  = (int)(iter * originalNPtsLimiter);
            size_t amountOfPointsForSkip_arg = amountOfPointsForSkip;
            int    dimension_arg             = 2;
            numb h_arg                     = h;
            numb eps_arg                   = eps;
            int    amountOfIC_arg            = N;
            int    amountOfValues_arg        = amountOfValues;
            int    amountOfIterations_arg    = (int)(tMax / NT);
            int    preScaller_arg            = 1;
            int    writableVar_arg           = 0;
            numb maxValue_arg              = maxValue;
            int    hSweepAxis_arg            = hSweepAxis;
            numb transientTime_arg         = transientTime;
            int    logAxisMask_arg           = logAxisMask;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &NT_arg, &tMax_arg, &sizeOfBlock_arg,
                &amountOfCalculatedPoints, &amountOfPointsForSkip_arg, &dimension_arg,
                &d_ranges, &h_arg, &eps_arg, &d_indicesOfMutVars, &d_initialConditions,
                &amountOfIC_arg, &d_values, &amountOfValues_arg,
                &amountOfIterations_arg, &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &d_lsResult, &hSweepAxis_arg, &transientTime_arg, &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)((3 * N + 2 * N * N + amountOfValues)
                                                 * sizeof(numb) * blockSize);

            LS2_CHECK_CU(cuLaunchKernel(cached_ls_2d.kernel_ls,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args, nullptr),
                         "cuLaunchKernel(ls2d)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            LS2_CHECK(cudaDeviceSynchronize(), "sync after ls2d");

            LS2_CHECK(cudaMemcpy(h_lsResult.data(), d_lsResult,
                                 cur_limiter * (size_t)N * sizeof(numb),
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
                int flag = (first == 999.0 || first == -999.0) ? REGIME_UNBOUND
                                                               : REGIME_OSCILLATION;
                res.flags[out_idx] = flag;
                for (int j = 0; j < N; ++j) {
                    double v = h_lsResult[k * (size_t)N + j];
                    res.values[(size_t)j * total_cells + out_idx] = v;
                }
            }
        }

        // Write the data file once, AFTER the chunked loop — see data_export
        // header for why the grid layout is centralized there.
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream out(OUT_FILE_PATH);
            if (out.is_open()) {
                out << std::setprecision(set_precision);
                data_export::write_ls2d_cells(out, nPts, N, res.values.data());
            }
        }

        // Per-plane min/max — autoscale colormap'а в GUI выбирает их по
        // активной экспоненте. Diverged-ячейки (flag<0) и 999/-999/NaN
        // исключаются.
        res.min_val.assign((size_t)N, 0.0);
        res.max_val.assign((size_t)N, 0.0);
        for (int j = 0; j < N; ++j) {
            double vmin =  std::numeric_limits<double>::infinity();
            double vmax = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < total_cells; ++c) {
                if (!regime_is_oscillation(res.flags[c])) continue;
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


    LS1DResult ParametricEngine::Impl::run_ls1d_continuation_gpu(const LS1DRequest& req) {
        LS1DResult res;
        auto fail = [&](const std::string& msg) -> LS1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(cached_ls_cont, src_template_ls_cont,
                                           "ls1dContinuationKernel", "ls1d_cont.cu", ":lscont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int nPts = req.n_pts;
        const int N    = req.amountOfX;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_result     = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_result)     cudaFree(d_result);
            sig.release();
        };
        #define SC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define SC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        SC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        SC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)N * sizeof(numb)),              "cudaMalloc baseX");
        SC_CHECK(cudaMalloc((void**)&d_result,     (size_t)nPts * N * sizeof(numb)),       "cudaMalloc result");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        SC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        SC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)N * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = N;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, NT_arg = req.NT, tMax_arg = req.t_max;
        numb transientTime_arg = req.transient_time, eps_arg = req.eps, maxValue_arg = req.max_value;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &NT_arg, &tMax_arg, &transientTime_arg,
            &eps_arg, &maxValue_arg, &d_result
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        SC_CHECK_CU(cuLaunchKernel(cached_ls_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(ls cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        SC_CHECK(cudaDeviceSynchronize(), "sync after ls cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> host((size_t)nPts * N);
        SC_CHECK(cudaMemcpy(host.data(), d_result, host.size() * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy result");
        cleanup();
        #undef SC_CHECK
        #undef SC_CHECK_CU

        res.n_pts       = nPts;
        res.n_exponents = N;
        res.param_lo    = req.param_lo;
        res.param_hi    = req.param_hi;
        res.continuation_reverse = req.continuation_reverse;
        res.spectrum.assign(nPts, std::vector<double>((size_t)N, 0.0));
        res.flags.assign(nPts, 0);
        for (int j = 0; j < nPts; ++j) {
            bool ok = true;
            for (int k = 0; k < N; ++k) {
                const double v = host[(size_t)j * N + k];
                res.spectrum[j][k] = v;
                if (!std::isfinite(v)) ok = false;
            }
            // NaN у continuation-ядра = расходимость (FP оно не различает).
            res.flags[j] = ok ? REGIME_OSCILLATION : REGIME_UNBOUND;
        }
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }