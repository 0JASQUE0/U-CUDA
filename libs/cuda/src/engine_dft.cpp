#include "engine_impl.h"



    // run_dft1d_continuation_gpu — выделенное ядро (dft1d_cont.template.cu),
    // а не связка bifurcation1dContinuationKernel + DFT_custom: под h-свипом у
    // каждой точки своя длина блока, свой шаг и своё окно, чего DFT_custom
    // выразить не может (у него один sizeOfBlock/h/window на запуск).
    Dft1DResult ParametricEngine::Impl::run_dft1d_continuation_gpu(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_dft_cont, cached_dft_cont, src_template_dft_cont,
                                           "dft1dContinuationKernel", "dft1d_cont.cu", ":dftcont",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const double worstCaseH = req.sweep_over_h
                                ? ((req.param_lo < req.param_hi) ? req.param_lo : req.param_hi)
                                : req.h;
        if (worstCaseH <= 0.0) return fail("h must be > 0 (for an h-sweep, over the whole range)");
        const int maxPointsInBlock = (int)std::ceil(req.t_max / worstCaseH / req.pre_scaller);
        if (maxPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts  = req.n_pts;
        const int nFreq = req.n_freq;

        numb* d_baseValues = nullptr; numb* d_baseX = nullptr;
        numb* d_data = nullptr; numb* d_ak = nullptr; numb* d_bk = nullptr;
        int*    d_flags = nullptr;
        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_data)       cudaFree(d_data);
            if (d_ak)         cudaFree(d_ak);
            if (d_bk)         cudaFree(d_bk);
            if (d_flags)      cudaFree(d_flags);
            sig.release();
        };
        #define DC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define DC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)

        const size_t freqCells = (size_t)nPts * (size_t)nFreq;
        DC_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)), "cudaMalloc baseValues");
        DC_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),  "cudaMalloc baseX");
        // Один блок, а не nPts x block: kernel однопоточный, больше не нужно.
        DC_CHECK(cudaMalloc((void**)&d_data,       (size_t)maxPointsInBlock * sizeof(numb)), "cudaMalloc data");
        DC_CHECK(cudaMalloc((void**)&d_ak,         freqCells * sizeof(numb)),                "cudaMalloc AkCOS");
        DC_CHECK(cudaMalloc((void**)&d_bk,         freqCells * sizeof(numb)),                "cudaMalloc BkSIN");
        DC_CHECK(cudaMalloc((void**)&d_flags,      (size_t)nPts * sizeof(int)),                "cudaMalloc flags");
        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        DC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        DC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseX");

        int    nPts_arg = nPts, reverse_arg = req.continuation_reverse ? 1 : 0;
        int    logScale_arg = req.log_scale ? 1 : 0, sweepIsH_arg = req.sweep_over_h ? 1 : 0;
        int    mutParamIdx_arg = req.param_index;
        int    amountOfValues_arg = (int)req.base_values.size(), amountOfX_arg = req.amountOfX;
        numb lo_arg = req.param_lo, hi_arg = req.param_hi;
        numb h_arg = req.h, tMax_arg = req.t_max, transientTime_arg = req.transient_time;
        int    preScaller_arg = req.pre_scaller, writableVar_arg = req.writable_var;
        numb maxValue_arg = req.max_value;
        int    nFreq_arg = nFreq;
        numb freqLo_arg = req.freq_lo, freqHi_arg = req.freq_hi;
        int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
        int    windowType_arg = req.window_type;
        int    maxBlock_arg = maxPointsInBlock;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg, &logScale_arg, &sweepIsH_arg,
            &mutParamIdx_arg, &d_baseValues, &amountOfValues_arg,
            &d_baseX, &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &preScaller_arg, &writableVar_arg, &maxValue_arg,
            &nFreq_arg, &freqLo_arg, &freqHi_arg, &logFreqAxis_arg, &windowType_arg,
            &maxBlock_arg, &d_data, &d_ak, &d_bk, &d_flags
            ,&d_cancel_arg, &d_progress_arg
        };
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        DC_CHECK_CU(cuLaunchKernel(cached_dft_cont.kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
                    "cuLaunchKernel(dft cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        DC_CHECK(cudaDeviceSynchronize(), "sync after dft cont");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }

        std::vector<numb> ak_host(freqCells), bk_host(freqCells);
        res.flags.assign(nPts, 0);
        DC_CHECK(cudaMemcpy(ak_host.data(), d_ak, freqCells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy AkCOS");
        DC_CHECK(cudaMemcpy(bk_host.data(), d_bk, freqCells * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy BkSIN");
        res.ak_cos.assign(ak_host.begin(), ak_host.end());   // numb -> double
        res.bk_sin.assign(bk_host.begin(), bk_host.end());
        DC_CHECK(cudaMemcpy(res.flags.data(),  d_flags, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy flags");
        cleanup();
        #undef DC_CHECK
        #undef DC_CHECK_CU

        res.n_pts   = nPts;
        res.n_freq  = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = req.freq_lo;
        res.freq_hi  = req.freq_hi;
        res.continuation_reverse = req.continuation_reverse;
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }


    // run_dft1d_hsweep_gpu — классический (без continuation) свип по шагу h. Отдельно от
    // run_dft1d_classical, потому что связка calculateDiscreteModelCUDA + DFT_custom тут
    // неприменима: DFT_custom берёт одну длину блока, одно готовое окно и один шаг дискретизации
    // на весь запуск, а под h-свипом все три — per-point. Ядро dft1dHSweepKernel делает траекторию
    // и DFT в одном месте, поток на точку (в отличие от однопоточного dft1dContinuationKernel:
    // здесь точки независимы).
    // Буферы считаются по worst case — наименьшему h диапазона, дающему больше всего сэмплов; так
    // же поступают run_dft1d_cpu и run_dft1d_continuation_gpu. Chunking по nPtsLimiter — как в
    // run_dft1d_classical / run_bif1d.
    Dft1DResult ParametricEngine::Impl::run_dft1d_hsweep_gpu(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        // Валидация — как в run_dft1d_classical, минус param_index/var_sweep_index:
        // свипуемая величина здесь h, индексы параметра/НУ не участвуют вовсе
        // (именно поэтому старый путь ломался на системах без параметров).
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");
        // param_lo/hi (= границы h) проверены на положительность в run_dft_1d.

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_simple_cont_if_needed(pool_dft_hsweep, cached_dft_hsweep, src_template_dft_hsweep,
                                           "dft1dHSweepKernel", "dft1d_hsweep.cu", ":dfthsweep",
                                           req.krs_body, req.amountOfX, err))
            return fail(err);

        const int    nPts  = req.n_pts;
        const int    nFreq = req.n_freq;
        const double worstCaseH = (req.param_lo < req.param_hi) ? req.param_lo : req.param_hi;
        if (worstCaseH <= 0.0) return fail("h lo/hi must be > 0 when sweeping over dt (h)");

        // Самый мелкий шаг диапазона задаёт длину буфера. Считаем в double и
        // проверяем до сужения в int: t_max/h_lo легко перевалит за 2^31, и без
        // этой проверки пользователь получил бы отрицательный размер вместо
        // внятного сообщения.
        const double maxBlockD = std::ceil(req.t_max / worstCaseH / (double)req.pre_scaller);
        if (!(maxBlockD >= 1.0))
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");
        if (maxBlockD > 2.0e9)
            return fail("h lo too small: block > 2e9 samples (raise h lo, t_max or pre_scaller)");
        const int maxPointsInBlock = (int)maxBlockD;

        // --- Memory budget: как в run_dft1d_classical, но на систему берётся
        // worst-case длина блока. d_window тут нет вовсе — окно ядро считает
        // на лету, его длина у каждой точки своя.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        const size_t memPerSystem =
            (size_t)maxPointsInBlock * sizeof(numb) +   // d_data
            2 * (size_t)nFreq * sizeof(numb) +           // d_AkCOS + d_BkSIN
            sizeof(int);                                   // d_flags
        const size_t memConstants =
            (size_t)req.amountOfX * sizeof(numb) +
            req.base_values.size() * sizeof(numb);
        constexpr double SAFETY_FACTOR = 0.9;
        const size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        const size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter == 0)
            return fail("not enough GPU memory even for a single h-sweep point "
                        "(block " + std::to_string(maxPointsInBlock) + " samples)");
        if (nPtsLimiter > (size_t)nPts) nPtsLimiter = (size_t)nPts;
        const size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_AkCOS(nPtsLimiter * (size_t)nFreq);
        std::vector<numb> h_BkSIN(nPtsLimiter * (size_t)nFreq);
        std::vector<int>  h_flags(nPtsLimiter);

        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        numb* d_data       = nullptr;
        numb* d_AkCOS      = nullptr;
        numb* d_BkSIN      = nullptr;
        int*  d_flags      = nullptr;

        auto cleanup = [&]() {
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_data)       cudaFree(d_data);
            if (d_AkCOS)      cudaFree(d_AkCOS);
            if (d_BkSIN)      cudaFree(d_BkSIN);
            if (d_flags)      cudaFree(d_flags);
        };

        #define DH_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DH_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DH_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        const int amountOfValues = (int)req.base_values.size();

        DH_CHECK(cudaMalloc((void**)&d_baseValues, req.base_values.size() * sizeof(numb)),        "cudaMalloc baseValues");
        DH_CHECK(cudaMalloc((void**)&d_baseX,      (size_t)req.amountOfX * sizeof(numb)),         "cudaMalloc baseX");
        DH_CHECK(cudaMalloc((void**)&d_data,       nPtsLimiter * (size_t)maxPointsInBlock * sizeof(numb)), "cudaMalloc d_data");
        DH_CHECK(cudaMalloc((void**)&d_AkCOS,      nPtsLimiter * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_AkCOS");
        DH_CHECK(cudaMalloc((void**)&d_BkSIN,      nPtsLimiter * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_BkSIN");
        DH_CHECK(cudaMalloc((void**)&d_flags,      nPtsLimiter * sizeof(int)),                      "cudaMalloc d_flags");

        DH_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                            req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy baseValues");
        DH_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                            (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice),  "memcpy baseX");
        DH_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Снимок — как у classical; indexOfMutVar = -1 помечает "свипуется h,
        // а не элемент a[]/X[]" (data_export пишет его как есть).
        res.snapshot.values.assign(req.base_values.begin(), req.base_values.end());
        res.snapshot.initial_conditions.assign(req.initial_conditions.begin(), req.initial_conditions.end());
        res.snapshot.tMax          = req.t_max;
        res.snapshot.transientTime = req.transient_time;
        res.snapshot.h             = req.h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = req.pre_scaller;
        res.snapshot.writableVar   = req.writable_var;
        res.snapshot.indexOfMutVar = -1;
        res.snapshot.range_lo      = req.param_lo;
        res.snapshot.range_hi      = req.param_hi;
        res.snapshot.n_freq        = nFreq;
        res.snapshot.freq_lo       = req.freq_lo;
        res.snapshot.freq_hi       = req.freq_hi;
        res.snapshot.window_type   = req.window_type;

        const std::string& OUT_FILE_PATH = req.csv_output_path;
        constexpr int set_precision = 15;
        std::ofstream akFile, bkFile;
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            akFile.open(OUT_FILE_PATH + "_AkCOS.csv");
            bkFile.open(OUT_FILE_PATH + "_BkSIN.csv");
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            akFile.close();
            bkFile.close();
        }

        res.n_pts    = nPts;
        res.n_freq   = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = req.freq_lo;
        res.freq_hi  = req.freq_hi;
        res.flags.assign(nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        const size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            DH_CANCEL_CHECK();
            if (req.progress) req.progress->store(float(iter) / float(amountOfIteration), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // Ядро h-свипа динамическую shared не берёт (launch с 0) — настройка идёт как есть.
            const int blockSize = launch_block_size(0);
            const int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_arg                 = nPts;
            int    nPtsLimiter_arg          = (int)nPtsLimiter;
            int    amountOfCalculatedPoints = (int)(iter * originalNPtsLimiter);
            numb   lo_arg                   = (numb)req.param_lo;
            numb   hi_arg                   = (numb)req.param_hi;
            int    logScale_arg             = req.log_scale ? 1 : 0;
            int    amountOfValues_arg       = amountOfValues;
            int    amountOfX_arg            = req.amountOfX;
            numb   tMax_arg                 = (numb)req.t_max;
            numb   transientTime_arg        = (numb)req.transient_time;
            int    preScaller_arg           = req.pre_scaller;
            int    writableVar_arg          = req.writable_var;
            numb   maxValue_arg             = (numb)req.max_value;
            int    nFreq_arg                = nFreq;
            numb   freqLo_arg               = (numb)req.freq_lo;
            numb   freqHi_arg               = (numb)req.freq_hi;
            int    logFreqAxis_arg          = req.freq_log_scale ? 1 : 0;
            int    windowType_arg           = req.window_type;
            int    maxBlock_arg             = maxPointsInBlock;

            void* args[] = {
                &nPts_arg, &nPtsLimiter_arg, &amountOfCalculatedPoints,
                &lo_arg, &hi_arg, &logScale_arg,
                &d_baseValues, &amountOfValues_arg,
                &d_baseX, &amountOfX_arg,
                &tMax_arg, &transientTime_arg,
                &preScaller_arg, &writableVar_arg, &maxValue_arg,
                &nFreq_arg, &freqLo_arg, &freqHi_arg, &logFreqAxis_arg, &windowType_arg,
                &maxBlock_arg, &d_data, &d_AkCOS, &d_BkSIN, &d_flags
            };

            DH_CHECK_CU(cuLaunchKernel(cached_dft_hsweep.kernel,
                                       gridSize, 1, 1, blockSize, 1, 1,
                                       0, nullptr, args, nullptr),
                        "cuLaunchKernel(dft h-sweep)");
            DH_CHECK(cudaDeviceSynchronize(), "sync after dft h-sweep");

            DH_CHECK(cudaMemcpy(h_AkCOS.data(), d_AkCOS, nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_AkCOS");
            DH_CHECK(cudaMemcpy(h_BkSIN.data(), d_BkSIN, nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_BkSIN");
            DH_CHECK(cudaMemcpy(h_flags.data(), d_flags, nPtsLimiter * sizeof(int),                    cudaMemcpyDeviceToHost), "memcpy h_flags");
            DH_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            if (!OUT_FILE_PATH.empty()) {
                akFile.open(OUT_FILE_PATH + "_AkCOS.csv", std::ios::app);
                bkFile.open(OUT_FILE_PATH + "_BkSIN.csv", std::ios::app);
                if (akFile.is_open()) akFile << std::setprecision(set_precision);
                if (bkFile.is_open()) bkFile << std::setprecision(set_precision);
                const std::vector<double> akWide(h_AkCOS.begin(), h_AkCOS.end());
                const std::vector<double> bkWide(h_BkSIN.begin(), h_BkSIN.end());
                data_export::write_dft1d_matrix(akFile, akWide.data(), 0, (int)nPtsLimiter, nFreq);
                data_export::write_dft1d_matrix(bkFile, bkWide.data(), 0, (int)nPtsLimiter, nFreq);
                akFile.close();
                bkFile.close();
            }

            const size_t global_offset = originalNPtsLimiter * iter;
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                const size_t global_idx = global_offset + k;
                res.flags[global_idx] = h_flags[k];
                std::copy(h_AkCOS.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_AkCOS.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.ak_cos.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
                std::copy(h_BkSIN.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_BkSIN.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.bk_sin.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
            }
        }

        cleanup();
        #undef DH_CHECK
        #undef DH_CHECK_CU
        #undef DH_CANCEL_CHECK
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);
        res.ok = true;
        return res;
    }


    // run_dft_1d — 1D DFT (порт bifurcation_DFT_1D из hostLibrary.cu:4900-5315).
    // Диспетчер: continuation требует param-sweep (та же причина, что и у
    // run_bif1d) — делегирует в run_dft1d_continuation, иначе classical.
    Dft1DResult ParametricEngine::Impl::run_dft_1d(const Dft1DRequest& req) {
        // Ограничения и порядок — как у Bif/LLE/LS 1D: continuation требует
        // param- или h-свипа (цепочка по IC бессмысленна), log-сетка по
        // параметру требует положительных границ. CPU-ветки не трогают CUDA,
        // поэтому стоят до ensure_init внутри самих функций.
        if (req.log_scale && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
            Dft1DResult r;
            r.error = "log scale requires param lo/hi > 0";
            return r;
        }
        // h-свип: границы — это сам шаг, нулевой/отрицательный шаг не считается
        // ни на одном из путей. Проверка здесь, а не в каждой ветке, как у
        // log_scale выше (см. те же проверки в run_bif1d / run_lle_1d).
        if (req.sweep_over_h && !(req.param_lo > 0.0 && req.param_hi > 0.0)) {
            Dft1DResult r;
            r.error = "h lo/hi must be > 0 when sweeping over dt (h)";
            return r;
        }
        if (req.continuation) {
            if (req.sweep_over_var) {
                Dft1DResult r;
                r.error = "continuation requires a param sweep, not an IC sweep";
                return r;
            }
            return req.use_cpu ? run_dft1d_cpu(req, /*continuation*/ true)
                               : run_dft1d_continuation_gpu(req);
        }
        if (req.use_cpu) return run_dft1d_cpu(req, /*continuation*/ false);
        // Классический GPU-свип по h идёт отдельным ядром: связка
        // calculateDiscreteModelCUDA + DFT_custom тут не работает, потому что
        // DFT_custom принимает одну длину блока, одно готовое окно и один шаг
        // дискретизации на весь запуск, а под h-свипом все три — per-point.
        if (req.sweep_over_h) return run_dft1d_hsweep_gpu(req);
        return run_dft1d_classical(req);
    }


    // Общая для classical/continuation: строит оконную функцию длиной sizeOfBlock. DFT_custom
    // принимает готовое окно аргументом и НЕ считает его сам. Формулы и нумерация типов — в
    // cpu_build_window выше (была отдельная копия тех же формул; после добавления Blackman и
    // Blackman-Harris держать две копии в одном файле смысла нет).
    void ParametricEngine::Impl::build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) {
        cpu_build_window(out, sizeOfBlock, window_type);
    }


    // run_dft1d_classical — порт classical-ветки bifurcation_DFT_1D. Реюзает
    // cached.kernel_traj (calculateDiscreteModelCUDA, тот же PTX-кэш что и
    // run_bif1d) для генерации сырых траекторий, затем cached.kernel_dft
    // (DFT_custom) вместо peakFinderCUDA. Chunked по nPtsLimiter как run_bif1d.
    Dft1DResult ParametricEngine::Impl::run_dft1d_classical(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        // валидация (как run_bif1d + n_freq/freq range)
        if (req.krs_body.empty())                                   return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)     return fail("amountOfX out of [1," + std::to_string(kMaxAmountOfX) + "]");
        if ((int)req.initial_conditions.size() != req.amountOfX)    return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)       return fail("too many base_values");
        if (req.sweep_over_var) {
            if (req.var_sweep_index < 0 || req.var_sweep_index >= req.amountOfX)
                return fail("var_sweep_index out of range");
        } else {
            if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                return fail("param_index out of range");
        }
        // writable_var == -1 — sentinel "combination" (см. loopCalculateDiscreteModel_int
        // / Bifurcation1DRequest::writable_var) — тот же calculateDiscreteModelCUDA,
        // так что DFT1D поддерживает её точно так же.
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                    return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_if_needed(req.krs_body, req.amountOfX,
                               req.sweep_over_var ? 0 : 1, err)) return fail(err);

        const double tMax                       = req.t_max;
        const int    nPts                       = req.n_pts;
        const int    nFreq                      = req.n_freq;
        const double h                          = req.h;
        const int    amountOfInitialConditions  = req.amountOfX;
        const std::vector<numb> ic_staged_       = to_numb(req.initial_conditions);
        const numb*   initialConditions         = ic_staged_.data();
        numb   ranges[2]                        = { (numb)req.param_lo, (numb)req.param_hi };
        numb   rangesFreq[2]                    = { (numb)req.freq_lo, (numb)req.freq_hi };
        int    indicesOfMutVars[1]              = { req.sweep_over_var
                                                    ? req.var_sweep_index
                                                    : req.param_index };
        const int    writableVar                = req.writable_var;
        const double maxValue                   = req.max_value;
        const double transientTime              = req.transient_time;
        const std::vector<numb> values_staged_   = to_numb(req.base_values);
        const numb*   values                    = values_staged_.data();
        const int    amountOfValues             = (int)req.base_values.size();
        const int    preScaller                 = req.pre_scaller;
        const std::string& OUT_FILE_PATH        = req.csv_output_path;

        constexpr int blockSize_setup = 32;
        constexpr int set_precision   = 15;

        int amountOfPointsInBlock = (int)(tMax / h / preScaller);
        size_t amountOfPointsForSkip = steps_from_time_size_t(transientTime, h);
        if (amountOfPointsInBlock <= 0)
            return fail("computed amountOfPointsInBlock <= 0 (t_max/h/pre_scaller too small)");

        // Memory budget: как run_bif1d, но выход DFT (AkCOS/BkSIN, nPtsLimiter*n_freq каждый)
        // обычно намного меньше, чем outPeaks/timeOfPeaks (nPtsLimiter*amountOfPointsInBlock) —
        // n_freq почти всегда << amountOfPointsInBlock. d_window константен и с nPtsLimiter не
        // масштабируется.
        size_t freeMemory = 0;
        if (!gpu_free_budget(0.92, freeMemory)) return fail("cudaMemGetInfo failed");

        size_t memPerSystem =
            (size_t)amountOfPointsInBlock * sizeof(numb) +      // d_data
            2 * (size_t)nFreq * sizeof(numb) +                   // d_AkCOS + d_BkSIN (per-point share)
            sizeof(int);                                           // d_amountOfPeaks
        size_t memConstants =
            2 * sizeof(numb) +                                    // d_ranges
            2 * sizeof(numb) +                                    // d_rangesFreq
            sizeof(int) +                                           // d_indicesOfMutVars
            (size_t)amountOfInitialConditions * sizeof(numb) +
            (size_t)amountOfValues * sizeof(numb) +
            (size_t)amountOfPointsInBlock * sizeof(numb);         // d_window
        constexpr double SAFETY_FACTOR = 0.9;
        size_t safeFree = (size_t)((double)freeMemory * SAFETY_FACTOR);
        if (memConstants >= safeFree) return fail("not enough GPU memory for constants");
        size_t availableMemory = safeFree - memConstants;

        size_t nPtsLimiter = availableMemory / memPerSystem;
        if (nPtsLimiter < (size_t)blockSize_setup) nPtsLimiter = (size_t)blockSize_setup;
        if (nPtsLimiter > (size_t)nPts)            nPtsLimiter = (size_t)nPts;
        // Округления вниз до кратного blockSize_setup здесь больше нет — по той же причине, что в
        // run_bif1d: ядра сами отсекают лишние потоки, а при n_pts < 32 округление давало 0, и Run
        // падал с сообщением про нехватку памяти, которая была ни при чём.
        if (nPtsLimiter == 0) return fail("n_pts must be > 0");
        size_t originalNPtsLimiter = nPtsLimiter;

        std::vector<numb> h_AkCOS(nPtsLimiter * (size_t)nFreq);
        std::vector<numb> h_BkSIN(nPtsLimiter * (size_t)nFreq);
        std::vector<int>    h_amountOfPeaks(nPtsLimiter);
        std::vector<numb> h_window;
        build_window(h_window, amountOfPointsInBlock, req.window_type);

        numb* d_data              = nullptr;
        numb* d_ranges            = nullptr;
        numb* d_rangesFreq        = nullptr;
        int*    d_indicesOfMutVars  = nullptr;
        numb* d_initialConditions = nullptr;
        numb* d_values            = nullptr;
        int*    d_amountOfPeaks     = nullptr;
        numb* d_AkCOS             = nullptr;
        numb* d_BkSIN             = nullptr;
        numb* d_window            = nullptr;

        RunSignals sig;   // прогресс и отмена в mapped-памяти
        auto cleanup = [&]() {
            if (d_data)              cudaFree(d_data);
            if (d_ranges)            cudaFree(d_ranges);
            if (d_rangesFreq)        cudaFree(d_rangesFreq);
            if (d_indicesOfMutVars)  cudaFree(d_indicesOfMutVars);
            if (d_initialConditions) cudaFree(d_initialConditions);
            if (d_values)            cudaFree(d_values);
            if (d_amountOfPeaks)     cudaFree(d_amountOfPeaks);
            if (d_AkCOS)             cudaFree(d_AkCOS);
            if (d_BkSIN)             cudaFree(d_BkSIN);
            if (d_window)            cudaFree(d_window);
        };
            sig.release();

        #define DFT_CHECK(call, where) do { \
            cudaError_t _e = (call); \
            if (_e != cudaSuccess) { \
                res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DFT_CHECK_CU(call, where) do { \
            CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { \
                res.error = std::string(where) + ": " + cu_err(_r); \
                cleanup(); return res; \
            } \
        } while(0)
        #define DFT_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        DFT_CHECK(cudaMalloc((void**)&d_data,              nPtsLimiter * (size_t)amountOfPointsInBlock * sizeof(numb)), "cudaMalloc d_data");
        DFT_CHECK(cudaMalloc((void**)&d_ranges,            2 * sizeof(numb)),                                          "cudaMalloc d_ranges");
        DFT_CHECK(cudaMalloc((void**)&d_rangesFreq,        2 * sizeof(numb)),                                          "cudaMalloc d_rangesFreq");
        DFT_CHECK(cudaMalloc((void**)&d_indicesOfMutVars,  1 * sizeof(int)),                                             "cudaMalloc d_indicesOfMutVars");
        DFT_CHECK(cudaMalloc((void**)&d_initialConditions, (size_t)amountOfInitialConditions * sizeof(numb)),          "cudaMalloc d_initialConditions");
        DFT_CHECK(cudaMalloc((void**)&d_values,            (size_t)amountOfValues * sizeof(numb)),                     "cudaMalloc d_values");
        DFT_CHECK(cudaMalloc((void**)&d_amountOfPeaks,     nPtsLimiter * sizeof(int)),                                   "cudaMalloc d_amountOfPeaks");
        DFT_CHECK(cudaMalloc((void**)&d_AkCOS,             nPtsLimiter * (size_t)nFreq * sizeof(numb)),                "cudaMalloc d_AkCOS");
        DFT_CHECK(cudaMalloc((void**)&d_BkSIN,             nPtsLimiter * (size_t)nFreq * sizeof(numb)),                "cudaMalloc d_BkSIN");
        DFT_CHECK(cudaMalloc((void**)&d_window,            (size_t)amountOfPointsInBlock * sizeof(numb)),             "cudaMalloc d_window");
        if (!sig.alloc(res.error)) { cleanup(); return res; }
        const size_t stepsPerPoint  = amountOfPointsForSkip + (size_t)amountOfPointsInBlock;
        const int    progressStride = progress_stride_for(stepsPerPoint);
        const double ticksPerPoint  = (double)(stepsPerPoint / (size_t)progressStride);
        const double ticksTotal     = (double)nPts * ticksPerPoint;

        DFT_CHECK(cudaMemcpy(d_ranges,            ranges,             2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_ranges");
        DFT_CHECK(cudaMemcpy(d_rangesFreq,        rangesFreq,         2 * sizeof(numb),                                cudaMemcpyHostToDevice), "memcpy d_rangesFreq");
        DFT_CHECK(cudaMemcpy(d_indicesOfMutVars,  indicesOfMutVars,   1 * sizeof(int),                                   cudaMemcpyHostToDevice), "memcpy d_indices");
        DFT_CHECK(cudaMemcpy(d_initialConditions, initialConditions, (size_t)amountOfInitialConditions * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_ic");
        DFT_CHECK(cudaMemcpy(d_values,            values,            (size_t)amountOfValues * sizeof(numb),            cudaMemcpyHostToDevice), "memcpy d_values");
        DFT_CHECK(cudaMemcpy(d_window,            h_window.data(),   (size_t)amountOfPointsInBlock * sizeof(numb),     cudaMemcpyHostToDevice), "memcpy d_window");
        DFT_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        size_t amountOfIteration = (size_t)std::ceil((double)nPts / (double)nPtsLimiter);

        res.snapshot.values.assign(values, values + amountOfValues);
        res.snapshot.initial_conditions.assign(initialConditions,
            initialConditions + amountOfInitialConditions);
        res.snapshot.tMax          = tMax;
        res.snapshot.transientTime = transientTime;
        res.snapshot.h             = h;
        res.snapshot.gpu_fmad      = get_nvrtc_fmad();
        res.snapshot.preScaller    = preScaller;
        res.snapshot.writableVar   = writableVar;
        res.snapshot.indexOfMutVar = indicesOfMutVars[0];
        res.snapshot.range_lo      = ranges[0];
        res.snapshot.range_hi      = ranges[1];
        res.snapshot.n_freq        = nFreq;
        res.snapshot.freq_lo       = rangesFreq[0];
        res.snapshot.freq_hi       = rangesFreq[1];
        res.snapshot.window_type   = req.window_type;

        std::ofstream akFile, bkFile;
        if (!OUT_FILE_PATH.empty()) {
            std::ofstream cfg(OUT_FILE_PATH + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            akFile.open(OUT_FILE_PATH + "_AkCOS.csv");
            bkFile.open(OUT_FILE_PATH + "_BkSIN.csv");
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            akFile.close();
            bkFile.close();
        }

        res.n_pts  = nPts;
        res.n_freq = nFreq;
        res.param_lo = ranges[0];
        res.param_hi = ranges[1];
        res.freq_lo  = rangesFreq[0];
        res.freq_hi  = rangesFreq[1];
        res.flags.assign(nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        for (size_t iter = 0; iter < amountOfIteration; ++iter) {
            DFT_CANCEL_CHECK();
            // Тик-шкала, а не доля чанков: иначе на стыке бар дёргается назад (см. wait_with_signals).
            if (req.progress) req.progress->store((float)((double)(originalNPtsLimiter * iter) * ticksPerPoint / ticksTotal), std::memory_order_relaxed);
            if (iter == amountOfIteration - 1)
                nPtsLimiter = nPts - (originalNPtsLimiter * iter);

            // Обе стадии (траектория и DFT_custom) запускаются одной шириной, поэтому режем её
            // по shared более требовательной из них — траектории.
            int blockSize = launch_block_size(
                (size_t)ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb));
            int gridSize  = (int)((nPtsLimiter + blockSize - 1) / blockSize);

            int    nPts_int                  = nPts;
            int    nPtsLimiter_int           = (int)nPtsLimiter;
            size_t sizeOfBlock_s             = (size_t)amountOfPointsInBlock;  // for kernel_traj (size_t param)
            int    sizeOfBlock_i             = amountOfPointsInBlock;          // for DFT_custom (int param — see gotcha #3)
            size_t amountOfCalculatedPoints  = iter * originalNPtsLimiter;
            size_t amountOfPointsForSkip_s   = (size_t)amountOfPointsForSkip;
            int    dimension                 = 1;
            numb h_arg                     = h;
            int    amountOfInitialConditions_int = amountOfInitialConditions;
            int    amountOfValues_int        = amountOfValues;
            size_t amountOfIterations_arg    = (size_t)amountOfPointsInBlock;
            int    preScaller_int            = preScaller;
            int    writableVar_int           = writableVar;
            numb maxValue_arg              = maxValue;
            bool   par_or_var_arg            = !req.sweep_over_var;
            // h-свип сюда не доходит: run_dft_1d уводит его в run_dft1d_hsweep_gpu (DFT_custom
            // берёт одну длину блока, одно окно и один шаг на весь запуск — под per-point h этого
            // мало). Здесь ось всегда param/IC, поэтому hSweepAxis выключен, а actualIterations не
            // нужен: длина блока одна на весь запуск, в отличие от run_bif1d, где его читает
            // peakFinderCUDA.
            int    hSweepAxis_arg            = -1;
            numb transientTime_arg         = transientTime;
            numb tMax_arg                  = tMax;
            int*   d_actualIterations        = nullptr;
            // Лог-сетка по оси параметра — бит 0 (в 1D ось одна), как в run_bif1d.
            int    logAxisMask_arg           = req.log_scale ? 1 : 0;

            int*   d_cancel_arg       = sig.cancelArg();
            int*   d_progress_arg     = sig.progressArg();
            int    progressStride_arg = progressStride;
            sig.resetTicks();

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
                &par_or_var_arg,
                &hSweepAxis_arg,
                &transientTime_arg,
                &tMax_arg,
                &d_actualIterations,
                &logAxisMask_arg,
                &d_cancel_arg, &d_progress_arg, &progressStride_arg
            };

            unsigned int shared = (unsigned int)(ucuda_shared_stride(amountOfInitialConditions, amountOfValues) * sizeof(numb) * blockSize);

            DFT_CHECK_CU(cuLaunchKernel(cached.kernel_traj,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        shared, nullptr, args_traj, nullptr),
                         "cuLaunchKernel(traj)");
            if (!wait_with_signals(0, sig, req.cancel, req.progress,
                                   (double)(originalNPtsLimiter * iter) * ticksPerPoint,
                                   ticksTotal, res.error)) { cleanup(); return res; }
            DFT_CHECK(cudaDeviceSynchronize(), "sync after traj");
            DFT_CANCEL_CHECK();

            // DFT_custom(data, sizeOfBlock, amountOfBlocks, checkerArray, AkCOS,
            // BkSIN, rangesFreq, window, nFreq, h, logFreqAxis) — h здесь ШАГ
            // МЕЖДУ decimated сэмплами (h*preScaller), как и у peakFinderCUDA
            // (см. run_bif1d).
            int    nFreq_int      = nFreq;
            numb timeStep_arg   = h * (double)preScaller;
            int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
            void* args_dft[] = {
                &d_data,
                &sizeOfBlock_i,
                &nPtsLimiter_int,
                &d_amountOfPeaks,
                &d_AkCOS,
                &d_BkSIN,
                &d_rangesFreq,
                &d_window,
                &nFreq_int,
                &timeStep_arg,
                &logFreqAxis_arg
            };
            DFT_CHECK_CU(cuLaunchKernel(cached.kernel_dft,
                                        gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args_dft, nullptr),
                         "cuLaunchKernel(dft)");
            DFT_CHECK(cudaDeviceSynchronize(), "sync after dft");

            DFT_CHECK(cudaMemcpy(h_AkCOS.data(),         d_AkCOS,          nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_AkCOS");
            DFT_CHECK(cudaMemcpy(h_BkSIN.data(),         d_BkSIN,          nPtsLimiter * (size_t)nFreq * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy h_BkSIN");
            DFT_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks,  nPtsLimiter * sizeof(int),                    cudaMemcpyDeviceToHost), "memcpy h_amountOfPeaks");
            DFT_CHECK(cudaDeviceSynchronize(), "sync after D2H");

            if (!OUT_FILE_PATH.empty()) {
                akFile.open(OUT_FILE_PATH + "_AkCOS.csv", std::ios::app);
                bkFile.open(OUT_FILE_PATH + "_BkSIN.csv", std::ios::app);
                if (akFile.is_open()) akFile << std::setprecision(set_precision);
                if (bkFile.is_open()) bkFile << std::setprecision(set_precision);
                // Экспорт принимает double* — расширяем numb-буфер на месте.
                const std::vector<double> akWide(h_AkCOS.begin(), h_AkCOS.end());
                const std::vector<double> bkWide(h_BkSIN.begin(), h_BkSIN.end());
                data_export::write_dft1d_matrix(akFile, akWide.data(), 0, (int)nPtsLimiter, nFreq);
                data_export::write_dft1d_matrix(bkFile, bkWide.data(), 0, (int)nPtsLimiter, nFreq);
                akFile.close();
                bkFile.close();
            }

            size_t global_offset = originalNPtsLimiter * iter;
            for (size_t k = 0; k < nPtsLimiter; ++k) {
                size_t global_idx = global_offset + k;
                res.flags[global_idx] = h_amountOfPeaks[k];
                std::copy(h_AkCOS.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_AkCOS.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.ak_cos.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
                std::copy(h_BkSIN.begin() + (ptrdiff_t)(k * (size_t)nFreq),
                          h_BkSIN.begin() + (ptrdiff_t)((k + 1) * (size_t)nFreq),
                          res.bk_sin.begin() + (ptrdiff_t)(global_idx * (size_t)nFreq));
            }
        }

        cleanup();
        #undef DFT_CHECK
        #undef DFT_CHECK_CU
        #undef DFT_CANCEL_CHECK
        res.ok = true;
        return res;
    }


    // run_dft1d_continuation — реюзает bifurcation1dContinuationKernel как есть: он уже пишет
    // полную decimated-траекторию в d_data и флаги в d_amountOfPeaks для ВСЕГО nPts за один
    // монолитный запуск (kernels/bifurcation1d_cont.template.cu). Второй проход — DFT_custom вместо
    // peakFinderCUDA над теми же буферами. Монолитно, без chunking, как run_bif1d_continuation.
    Dft1DResult ParametricEngine::Impl::run_dft1d_continuation(const Dft1DRequest& req) {
        Dft1DResult res;
        auto fail = [&](const std::string& msg) -> Dft1DResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                    return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX)      return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX)     return fail("initial_conditions.size() != amountOfX");
        if ((int)req.base_values.size() > kMaxAmountOfValues)        return fail("too many base_values");
        if (req.param_index < 0 || req.param_index >= (int)req.base_values.size())
                                                                     return fail("param_index out of range");
        // writable_var == -1 — sentinel "combination" (см. run_bif1d_continuation).
        if (req.writable_var < -1 || req.writable_var >= req.amountOfX)
                                                                     return fail("writable_var out of range");
        if (req.n_pts <= 0)         return fail("n_pts must be > 0");
        if (req.n_freq <= 0)        return fail("n_freq must be > 0");
        if (req.freq_hi <= req.freq_lo) return fail("freq_hi must be > freq_lo");
        if (req.freq_log_scale && !(req.freq_lo > 0.0 && req.freq_hi > 0.0))
            return fail("log scale over frequency requires freq_lo/freq_hi > 0");
        if (req.h <= 0.0)           return fail("h must be > 0");
        if (req.t_max <= 0.0)       return fail("t_max must be > 0");
        if (req.transient_time < 0) return fail("transient_time must be >= 0");
        if (req.pre_scaller <= 0)   return fail("pre_scaller must be > 0");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        if (!compile_bif1d_cont_if_needed(req.krs_body, req.amountOfX, err))
            return fail(err);

        const int amountOfPointsInBlock = (int)(req.t_max / req.h / req.pre_scaller);
        // amountOfPointsForSkip здесь не нужен: в этой ветке транзиент уходит в
        // ядро как время (transientTime_arg), и число шагов ядро считает само —
        // по своему hLocal, который в h-свипе меняется от точки к точке.
        if (amountOfPointsInBlock <= 0) return fail("amountOfPointsInBlock <= 0");

        const int nPts  = req.n_pts;
        const int nFreq = req.n_freq;
        numb   rangesFreq[2] = { (numb)req.freq_lo, (numb)req.freq_hi };

        std::vector<numb> h_window;
        build_window(h_window, amountOfPointsInBlock, req.window_type);

        numb* d_data       = nullptr;
        numb* d_baseValues = nullptr;
        numb* d_baseX      = nullptr;
        int*    d_amountOfPeaks = nullptr;
        numb* d_AkCOS      = nullptr;
        numb* d_BkSIN      = nullptr;
        numb* d_rangesFreq = nullptr;
        numb* d_window     = nullptr;

        RunSignals sig;   // однопоточное ядро: тик на точку
        auto cleanup = [&]() {
            if (d_data)       cudaFree(d_data);
            if (d_baseValues) cudaFree(d_baseValues);
            if (d_baseX)      cudaFree(d_baseX);
            if (d_amountOfPeaks) cudaFree(d_amountOfPeaks);
            if (d_AkCOS)      cudaFree(d_AkCOS);
            if (d_BkSIN)      cudaFree(d_BkSIN);
            if (d_rangesFreq) cudaFree(d_rangesFreq);
            if (d_window)     cudaFree(d_window);
            sig.release();
        };
        #define DFTC_CHECK(call, where) do { cudaError_t _e = (call); \
            if (_e != cudaSuccess) { res.error = std::string("CUDA ") + (where) + ": " + cudaGetErrorString(_e); cleanup(); return res; } } while(0)
        #define DFTC_CHECK_CU(call, where) do { CUresult _r = (call); \
            if (_r != CUDA_SUCCESS) { res.error = std::string(where) + ": " + cu_err(_r); cleanup(); return res; } } while(0)
        #define DFTC_CANCEL_CHECK() do { \
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) { \
                res.cancelled = true; \
                res.error = "Cancelled by user"; \
                cleanup(); return res; \
            } \
        } while(0)

        const size_t dataBytes = (size_t)nPts * (size_t)amountOfPointsInBlock * sizeof(numb);
        DFTC_CHECK(cudaMalloc((void**)&d_data,         dataBytes),                                       "cudaMalloc d_data");
        DFTC_CHECK(cudaMalloc((void**)&d_baseValues,   (size_t)req.base_values.size() * sizeof(numb)), "cudaMalloc d_baseValues");
        DFTC_CHECK(cudaMalloc((void**)&d_baseX,        (size_t)req.amountOfX * sizeof(numb)),          "cudaMalloc d_baseX");
        DFTC_CHECK(cudaMalloc((void**)&d_amountOfPeaks,(size_t)nPts * sizeof(int)),                      "cudaMalloc d_amountOfPeaks");
        DFTC_CHECK(cudaMalloc((void**)&d_AkCOS,        (size_t)nPts * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_AkCOS");
        DFTC_CHECK(cudaMalloc((void**)&d_BkSIN,        (size_t)nPts * (size_t)nFreq * sizeof(numb)),   "cudaMalloc d_BkSIN");
        DFTC_CHECK(cudaMalloc((void**)&d_rangesFreq,   2 * sizeof(numb)),                               "cudaMalloc d_rangesFreq");
        DFTC_CHECK(cudaMalloc((void**)&d_window,       (size_t)amountOfPointsInBlock * sizeof(numb)),  "cudaMalloc d_window");

        // Вход приходит из Request в double — сужаем до numb на границе.
        const std::vector<numb> baseValues_staged_ = to_numb(req.base_values);
        const std::vector<numb> baseX_staged_      = to_numb(req.initial_conditions);
        DFTC_CHECK(cudaMemcpy(d_baseValues, baseValues_staged_.data(),
                           req.base_values.size() * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseValues");
        DFTC_CHECK(cudaMemcpy(d_baseX, baseX_staged_.data(),
                           (size_t)req.amountOfX * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_baseX");
        DFTC_CHECK(cudaMemcpy(d_rangesFreq, rangesFreq, 2 * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_rangesFreq");
        DFTC_CHECK(cudaMemcpy(d_window, h_window.data(),
                           (size_t)amountOfPointsInBlock * sizeof(numb), cudaMemcpyHostToDevice), "memcpy d_window");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after H2D");

        // Launch continuation kernel (тот же, что у run_bif1d_continuation)
        // У DFT1D нет h-свипа и log-сетки по параметру, поэтому соответствующие
        // флаги ядра выключены, а actualIterations не нужен (длина блока одна
        // на все точки) — передаём nullptr, ядро это допускает.
        int    nPts_arg              = nPts;
        numb lo_arg                = req.param_lo;
        numb hi_arg                = req.param_hi;
        bool   reverse_arg           = req.continuation_reverse;
        int    logScale_arg          = 0;
        int    sweepIsH_arg          = 0;
        int    mutParamIdx_arg       = req.param_index;
        int    amountOfValues_arg    = (int)req.base_values.size();
        int    amountOfX_arg         = req.amountOfX;
        numb h_arg                 = req.h;
        numb tMax_arg              = req.t_max;
        numb transientTime_arg     = req.transient_time;
        int    sizeOfBlock_arg       = amountOfPointsInBlock;
        int    preScaller_arg        = req.pre_scaller;
        int    writableVar_arg       = req.writable_var;
        numb maxValue_arg          = req.max_value;
        int*   d_actualIterations    = nullptr;

        if (!sig.alloc(res.error)) { cleanup(); return res; }
        int* d_cancel_arg   = sig.cancelArg();
        int* d_progress_arg = sig.progressArg();
        void* cont_args[] = {
            &nPts_arg, &lo_arg, &hi_arg, &reverse_arg,
            &logScale_arg, &sweepIsH_arg, &mutParamIdx_arg,
            &d_baseValues, &amountOfValues_arg,
            &d_baseX,      &amountOfX_arg,
            &h_arg, &tMax_arg, &transientTime_arg,
            &sizeOfBlock_arg, &preScaller_arg,
            &writableVar_arg, &maxValue_arg,
            &d_data, &d_amountOfPeaks, &d_actualIterations,
            &d_cancel_arg, &d_progress_arg
        };
        DFTC_CANCEL_CHECK();
        DFTC_CHECK_CU(cuLaunchKernel(cached_cont.kernel_cont,
                                  1, 1, 1, 1, 1, 1, 0, nullptr, cont_args, nullptr),
                   "cuLaunchKernel(cont)");
        if (!wait_with_signals(0, sig, req.cancel, req.progress, 0.0, (double)nPts, res.error))
            { cleanup(); return res; }
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after cont kernel");
        if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
            res.cancelled = true; res.error = "Cancelled by user"; cleanup(); return res;
        }
        if (req.progress) req.progress->store(0.5f, std::memory_order_relaxed);

        // DFT_custom вместо peakFinderCUDA, над теми же d_data/d_amountOfPeaks
        int    sizeOfBlock_i   = amountOfPointsInBlock;  // int, не size_t — см. gotcha #3
        int    nFreq_int       = nFreq;
        // numb, а не double — DFT_custom ждёт numb h; см. пояснение в
        // run_bif1d_continuation.
        numb timeStep          = (numb)(req.h * (double)req.pre_scaller);
        int    logFreqAxis_arg = req.freq_log_scale ? 1 : 0;
        void* dft_args[] = {
            &d_data, &sizeOfBlock_i, &nPts_arg,
            &d_amountOfPeaks, &d_AkCOS, &d_BkSIN, &d_rangesFreq, &d_window,
            &nFreq_int, &timeStep, &logFreqAxis_arg
        };
        int    dftBlock = 32;
        int    dftGrid  = (nPts + dftBlock - 1) / dftBlock;
        DFTC_CHECK_CU(cuLaunchKernel(cached_cont.kernel_dft,
                                  dftGrid, 1, 1, dftBlock, 1, 1,
                                  0, nullptr, dft_args, nullptr),
                   "cuLaunchKernel(dft)");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after dft");

        res.n_pts  = nPts;
        res.n_freq = nFreq;
        res.param_lo = req.param_lo;
        res.param_hi = req.param_hi;
        res.freq_lo  = rangesFreq[0];
        res.freq_hi  = rangesFreq[1];
        res.continuation_reverse = req.continuation_reverse;
        res.flags.assign((size_t)nPts, 0);
        res.ak_cos.assign((size_t)nPts * (size_t)nFreq, 0.0);
        res.bk_sin.assign((size_t)nPts * (size_t)nFreq, 0.0);

        std::vector<int> h_amountOfPeaks((size_t)nPts);
        {   // Приёмник D2H — numb, наружу расширяем в double-хранилище Result.
            const size_t cells_ = (size_t)nPts * (size_t)nFreq;
            std::vector<numb> ak_(cells_), bk_(cells_);
            DFTC_CHECK(cudaMemcpy(ak_.data(), d_AkCOS, cells_ * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy out d_AkCOS");
            DFTC_CHECK(cudaMemcpy(bk_.data(), d_BkSIN, cells_ * sizeof(numb), cudaMemcpyDeviceToHost), "memcpy out d_BkSIN");
            res.ak_cos.assign(ak_.begin(), ak_.end());
            res.bk_sin.assign(bk_.begin(), bk_.end());
        }
        DFTC_CHECK(cudaMemcpy(h_amountOfPeaks.data(), d_amountOfPeaks, (size_t)nPts * sizeof(int), cudaMemcpyDeviceToHost), "memcpy out d_amountOfPeaks");
        DFTC_CHECK(cudaDeviceSynchronize(), "sync after D2H");
        for (int j = 0; j < nPts; ++j) res.flags[(size_t)j] = h_amountOfPeaks[(size_t)j];
        if (req.progress) req.progress->store(1.0f, std::memory_order_relaxed);

        if (!req.csv_output_path.empty()) {
            res.snapshot.values = req.base_values;
            res.snapshot.initial_conditions = req.initial_conditions;
            res.snapshot.tMax          = req.t_max;
            res.snapshot.transientTime = req.transient_time;
            res.snapshot.h             = req.h;
            res.snapshot.gpu_fmad      = get_nvrtc_fmad();
            res.snapshot.preScaller    = req.pre_scaller;
            res.snapshot.writableVar   = req.writable_var;
            res.snapshot.indexOfMutVar = req.param_index;
            res.snapshot.range_lo      = req.param_lo;
            res.snapshot.range_hi      = req.param_hi;
            res.snapshot.n_freq        = nFreq;
            res.snapshot.freq_lo       = rangesFreq[0];
            res.snapshot.freq_hi       = rangesFreq[1];
            res.snapshot.window_type   = req.window_type;

            std::ofstream cfg(req.csv_output_path + "_config.csv");
            data_export::write_dft1d_config(cfg, res.snapshot);
            std::ofstream akFile(req.csv_output_path + "_AkCOS.csv");
            std::ofstream bkFile(req.csv_output_path + "_BkSIN.csv");
            akFile << std::setprecision(15);
            bkFile << std::setprecision(15);
            data_export::write_dft1d_header(akFile, res.snapshot);
            data_export::write_dft1d_header(bkFile, res.snapshot);
            data_export::write_dft1d_matrix(akFile, res.ak_cos.data(), 0, nPts, nFreq);
            data_export::write_dft1d_matrix(bkFile, res.bk_sin.data(), 0, nPts, nFreq);
        }

        cleanup();
        #undef DFTC_CHECK
        #undef DFTC_CHECK_CU
        #undef DFTC_CANCEL_CHECK
        res.ok = true;
        return res;
    }
