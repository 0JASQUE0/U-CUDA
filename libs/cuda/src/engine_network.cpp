#include "engine_impl.h"
using namespace ucuda::cuda_detail;
// Network: сеть связанных осцилляторов (связь расщеплением после шага узла).

NetworkResult ParametricEngine::Impl::run_network(const NetworkRequest& req) {
        NetworkResult res;
        auto fail = [&](const std::string& msg) -> NetworkResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if (req.n_nodes <= 0)                                    return fail("the network has no nodes");
        if (req.n_nodes > kMaxNetworkNodes)
            return fail("too many nodes: the whole network runs in one block, the cap is "
                        + std::to_string(kMaxNetworkNodes));
        if (req.amountOfValues <= 0 || req.amountOfValues > kMaxAmountOfValues)
            return fail("amountOfValues out of range");
        if ((int)req.values.size() != req.n_nodes * req.amountOfValues)
            return fail("values.size() != n_nodes * amountOfValues");
        if ((int)req.initial_conditions.size() != req.n_nodes * req.amountOfX)
            return fail("initial_conditions.size() != n_nodes * amountOfX");
        if ((int)req.edge_start.size() != req.n_nodes + 1)
            return fail("edge_start.size() != n_nodes + 1");
        const int n_arcs = req.edge_start.back();
        if (n_arcs < 0 || (int)req.edge_src.size() != n_arcs
            || (int)req.edge_w.size() != n_arcs || (int)req.edge_law.size() != n_arcs)
            return fail("the CSR arrays disagree on the number of arcs");
        for (int i = 0; i < req.n_nodes; ++i)
            if (req.edge_start[(size_t)i] > req.edge_start[(size_t)i + 1])
                return fail("edge_start is not monotonic");
        for (int e = 0; e < n_arcs; ++e)
            if (req.edge_src[(size_t)e] < 0 || req.edge_src[(size_t)e] >= req.n_nodes)
                return fail("an arc points outside the node list");
        if (!(req.h > 0.0))     return fail("h must be > 0");
        if (!(req.t_max > 0.0)) return fail("t_max must be > 0");
        if (req.transient < 0.0) return fail("transient must be >= 0");
        if (req.pre_scaller < 1) return fail("preScaller must be >= 1");
        if (req.max_points < 2)  return fail("max_points must be >= 2");
        if (req.steps_per_launch < 0) return fail("steps_per_launch must be >= 0");

        // Состояние всей сети живёт в динамической shared — отсюда потолок.
        const size_t shared_bytes = (size_t)req.n_nodes * (size_t)req.amountOfX * sizeof(numb);
        if (shared_bytes > kNetworkSharedCap)
            return fail("the network state does not fit into shared memory ("
                        + std::to_string(shared_bytes) + " B > " + std::to_string(kNetworkSharedCap)
                        + " B): reduce the node count or the system dimension");

        const long long skipSteps  = (long long)steps_from_time_size_t(req.transient, req.h);
        const long long workSteps  = (long long)steps_from_time_size_t(req.t_max, req.h);
        const long long totalSteps = skipSteps + workSteps;
        if (totalSteps <= 0) return fail("t_max / h gives no steps");
        if ((double)totalSteps > 1.0e12)
            return fail("t_max / h too large: the work does not fit into a reasonable time");

        // Прореживание: пользовательское, но поднятое до того, при котором
        // записанное влезает в max_points. Поднимать молча нельзя — фактическое
        // значение уезжает в результат и показывается в UI.
        int preScaller = req.pre_scaller;
        {
            const long long need = workSteps / preScaller + 1;
            if (need > (long long)req.max_points) {
                const long long k = (workSteps + (long long)req.max_points - 2)
                                  / ((long long)req.max_points - 1);
                if (k > preScaller) preScaller = (int)k;
            }
        }
        const long long nPointsLL = workSteps / preScaller + 1;
        if (nPointsLL <= 0 || nPointsLL > 2147483647LL) return fail("the recorded point count is out of range");
        const int nPoints = (int)nPointsLL;

        const size_t out_elems = (size_t)nPoints * (size_t)req.n_nodes * (size_t)req.amountOfX;
        const size_t out_bytes = out_elems * sizeof(numb);
        if (out_bytes > kNetworkOutCap)
            return fail("the recorded trajectory would take " + std::to_string(out_bytes >> 20)
                        + " MB: raise the decimation or lower t_max");

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        if (!compile_network_module(true, req.amountOfX, req.amountOfValues,
                                    req.krs_body, req.coupling_body, err)) return fail(err);

        OrderDevBuf d_values, d_state, d_eStart, d_eSrc, d_eW, d_eLaw, d_out, d_status;
        if (!d_values.alloc(req.values.size() * sizeof(numb), "values", err)) return fail(err);
        if (!d_state .alloc(req.initial_conditions.size() * sizeof(numb), "state", err)) return fail(err);
        if (!d_eStart.alloc(req.edge_start.size() * sizeof(int), "edgeStart", err)) return fail(err);
        if (!d_eSrc  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeSrc", err)) return fail(err);
        if (!d_eW    .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(numb), "edgeW",   err)) return fail(err);
        if (!d_eLaw  .alloc((size_t)(n_arcs > 0 ? n_arcs : 1) * sizeof(int),  "edgeLaw", err)) return fail(err);
        if (!d_out   .alloc(out_bytes, "out", err)) return fail(err);
        if (!d_status.alloc((size_t)req.n_nodes * sizeof(int), "status", err)) return fail(err);

        {
            const std::vector<numb> hv  = to_numb(req.values);
            const std::vector<numb> hic = to_numb(req.initial_conditions);
            const std::vector<numb> hw  = to_numb(req.edge_w);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                if (bytes == 0) return true;
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_values.p, hv.data(),  hv.size()  * sizeof(numb), "values"))    return fail(err);
            if (!up(d_state.p,  hic.data(), hic.size() * sizeof(numb), "state"))     return fail(err);
            if (!up(d_eStart.p, req.edge_start.data(), req.edge_start.size() * sizeof(int), "edgeStart")) return fail(err);
            if (!up(d_eSrc.p,   req.edge_src.data(),   (size_t)n_arcs * sizeof(int),  "edgeSrc")) return fail(err);
            if (!up(d_eW.p,     hw.data(),             (size_t)n_arcs * sizeof(numb), "edgeW"))   return fail(err);
            if (!up(d_eLaw.p,   req.edge_law.data(),   (size_t)n_arcs * sizeof(int),  "edgeLaw")) return fail(err);
        }
        // 0xFF по всем байтам — это NaN и для float, и для double: точки после
        // разлёта обязаны читаться как «не посчитано», а не как ноль.
        if (cudaMemset(d_out.p, 0xFF, out_bytes) != cudaSuccess) return fail("memset out");
        if (cudaMemset(d_status.p, 0, (size_t)req.n_nodes * sizeof(int)) != cudaSuccess)
            return fail("memset status");

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Один блок = одна сеть, поток = узел. Идущие сверх узлов потоки стоят
        // на тех же __syncthreads, что и рабочие, и ничего не считают.
        int blockSize = ((req.n_nodes + 31) / 32) * 32;
        if (blockSize > 1024) blockSize = 1024;

        // Чанкование против watchdog'а (TDR): расчёт последователен по времени,
        // поэтому режем его по шагам, а состояние переносится через d_state.
        const double workPerStep = (double)req.n_nodes * (1.0 + (double)n_arcs / (double)req.n_nodes);
        long long stepsPerLaunch = (long long)(kNetworkWorkBudget / (workPerStep > 1.0 ? workPerStep : 1.0));
        if (stepsPerLaunch < 1)          stepsPerLaunch = 1;
        if (stepsPerLaunch > totalSteps) stepsPerLaunch = totalSteps;

        if (req.steps_per_launch > 0 && req.steps_per_launch < stepsPerLaunch)
            stepsPerLaunch = req.steps_per_launch;

        // Event pairs are allocated only for an explicit benchmark request.
        // Record the end before host polling/status copies, so these are excluded.
        struct NetworkEvents {
            CUevent begin = nullptr, end = nullptr;
            ~NetworkEvents() {
                if (end && cuEventDestroy(end) != CUDA_SUCCESS)
                    std::fprintf(stderr, "network: cuEventDestroy(end) failed\n");
                if (begin && cuEventDestroy(begin) != CUDA_SUCCESS)
                    std::fprintf(stderr, "network: cuEventDestroy(begin) failed\n");
            }
        } events;
        if (req.measure_gpu_time) {
            CUresult r = cuEventCreate(&events.begin, CU_EVENT_DEFAULT);
            if (r != CUDA_SUCCESS) return fail("network event create: " + cu_err(r));
            r = cuEventCreate(&events.end, CU_EVENT_DEFAULT);
            if (r != CUDA_SUCCESS) return fail("network event create: " + cu_err(r));
        }

        const int    progressStride = progress_stride_for((size_t)totalSteps);
        const double ticksTotal     = (double)totalSteps / (double)progressStride;
        double ticksDone = 0.0;

        for (long long base = 0; base < totalSteps; base += stepsPerLaunch) {
            const long long left  = totalSteps - base;
            const long long count = (stepsPerLaunch < left) ? stepsPerLaunch : left;

            numb*     values_arg = d_values.as<numb>();
            numb*     state_arg  = d_state.as<numb>();
            int*      eStart_arg = d_eStart.as<int>();
            int*      eSrc_arg   = d_eSrc.as<int>();
            numb*     eW_arg     = d_eW.as<numb>();
            int*      eLaw_arg   = d_eLaw.as<int>();
            int       nNodes_arg = req.n_nodes;
            numb      h_arg      = (numb)req.h;
            long long base_arg   = base;
            long long count_arg  = count;
            long long skip_arg   = skipSteps;
            int       pre_arg    = preScaller;
            int       nPts_arg   = nPoints;
            numb      maxV_arg   = (numb)req.max_value;
            numb*     out_arg    = d_out.as<numb>();
            int*      status_arg = d_status.as<int>();
            int*      cancel_arg = sig.cancelArg();
            int*      prog_arg   = sig.progressArg();
            int       stride_arg = progressStride;

            void* args[] = {
                &values_arg, &state_arg, &eStart_arg, &eSrc_arg, &eW_arg, &eLaw_arg,
                &nNodes_arg, &h_arg, &base_arg, &count_arg, &skip_arg, &pre_arg, &nPts_arg,
                &maxV_arg, &out_arg, &status_arg, &cancel_arg, &prog_arg, &stride_arg
            };

            sig.resetTicks();
            if (req.measure_gpu_time) {
                CUresult r = cuEventRecord(events.begin, nullptr);
                if (r != CUDA_SUCCESS) return fail("network event begin: " + cu_err(r));
            }
            CUresult r = cuLaunchKernel(cached_network.kernel, 1, 1, 1, blockSize, 1, 1,
                                        (unsigned)shared_bytes, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(network): " + cu_err(r));
            ++res.launch_count;
            if (req.measure_gpu_time) {
                r = cuEventRecord(events.end, nullptr);
                if (r != CUDA_SUCCESS) return fail("network event end: " + cu_err(r));
            }
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("network kernel: ") + cudaGetErrorString(ce));

            if (req.measure_gpu_time) {
                float elapsed = 0.0f; // CUDA event API requires float milliseconds.
                r = cuEventElapsedTime(&elapsed, events.begin, events.end);
                if (r != CUDA_SUCCESS) return fail("network event elapsed: " + cu_err(r));
                res.gpu_ms += elapsed;
            }

            ticksDone += (double)count / (double)progressStride;

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            // Разлетелась сеть — остальные чанки досчитывать нечего: ядро на
            // них всё равно сразу выйдет, а пользователь ждал бы впустую.
            std::vector<int> st((size_t)req.n_nodes, 0);
            if (cudaMemcpy(st.data(), d_status.p, st.size() * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H status");
            bool diverged = false;
            for (int v : st) if (v != NET_ST_OK) diverged = true;
            if (diverged) break;
        }

        res.status.assign((size_t)req.n_nodes, 0);
        if (cudaMemcpy(res.status.data(), d_status.p, res.status.size() * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return fail("memcpy D2H status");
        for (int v : res.status) if (v != NET_ST_OK) ++res.n_diverged;

        {
            std::vector<numb> host(out_elems);
            if (cudaMemcpy(host.data(), d_out.p, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("memcpy D2H out");
            res.data.resize(out_elems);
            res.vmin.assign((size_t)req.amountOfX, 0.0);
            res.vmax.assign((size_t)req.amountOfX, 0.0);
            std::vector<bool> seen((size_t)req.amountOfX, false);
            for (size_t i = 0; i < out_elems; ++i) {
                const double v = (double)host[i];
                res.data[i] = v;
                if (!std::isfinite(v)) continue;
                const size_t k = i % (size_t)req.amountOfX;
                if (!seen[k]) { res.vmin[k] = res.vmax[k] = v; seen[k] = true; }
                else { if (v < res.vmin[k]) res.vmin[k] = v; if (v > res.vmax[k]) res.vmax[k] = v; }
            }
        }

        res.n_nodes     = req.n_nodes;
        res.n_vars      = req.amountOfX;
        res.n_points    = nPoints;
        res.h           = req.h;
        res.t0          = (double)skipSteps * req.h;
        res.dt          = (double)preScaller * req.h;
        res.pre_scaller = preScaller;
        res.ok          = true;
        return res;
    }
