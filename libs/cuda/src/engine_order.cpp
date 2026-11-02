#include "engine_impl.h"
using namespace ucuda::cuda_detail;

bool ParametricEngine::Impl::order_axis_nodes(const OrderAxis& ax, const char* what,
                                 std::vector<double>& out, std::string& err)
    {
        const int n = (ax.kind == OrderAxisKind::None) ? 1 : ax.n_pts;
        if (n <= 0)    { err = std::string(what) + ": the number of points must be > 0"; return false; }
        if (n > 100000){ err = std::string(what) + ": the number of points is too large"; return false; }
        out.assign((size_t)n, 0.0);
        if (ax.kind == OrderAxisKind::None) { out[0] = 0.0; return true; }
        if (n == 1) { out[0] = ax.lo; return true; }
        if (ax.log_scale) {
            if (!(ax.lo > 0.0) || !(ax.hi > 0.0)) {
                err = std::string(what) + ": log scale requires both bounds > 0";
                return false;
            }
            const double k = std::log(ax.hi / ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo * std::exp(k * (double)i);
        } else {
            const double d = (ax.hi - ax.lo) / (double)(n - 1);
            for (int i = 0; i < n; ++i) out[(size_t)i] = ax.lo + d * (double)i;
        }
        return true;
    }

long long ParametricEngine::Impl::order_steps_for(double h, double tMax, bool snap) {
        if (!(h > 0.0)) return 1;
        long long N = snap ? (long long)(tMax / h + 0.5) : (long long)(tMax / h);
        return N < 1 ? 1 : N;
    }

OrderResult ParametricEngine::Impl::run_order(const OrderRequest& req) {
        OrderResult res;
        res.axis_x = req.axis_x;
        res.axis_y = req.axis_y;
        auto fail = [&](const std::string& msg) -> OrderResult& { res.error = msg; return res; };

        if (req.krs_body.empty())                                return fail("krs_body is empty");
        if (req.amountOfX <= 0 || req.amountOfX > kMaxAmountOfX) return fail("amountOfX out of range");
        if ((int)req.initial_conditions.size() != req.amountOfX) return fail("initial_conditions.size() != amountOfX");
        if (req.values.empty())                                  return fail("values is empty (at least a[0] is required)");
        if ((int)req.values.size() > kMaxAmountOfValues)         return fail("too many values");
        if (!(req.h > 0.0))                                      return fail("h must be > 0");
        if (!(req.t_max > 0.0))                                  return fail("t_max must be > 0");
        if (req.axis_x.kind == OrderAxisKind::None)              return fail("the X axis is not set");

        const int amountOfValues = (int)req.values.size();
        auto check_axis_index = [&](const OrderAxis& ax, const char* what) -> bool {
            if (ax.kind != OrderAxisKind::Value) return true;
            if (ax.index < 0 || ax.index >= amountOfValues) {
                res.error = std::string(what) + ": parameter index outside a[]";
                return false;
            }
            return true;
        };
        if (!check_axis_index(req.axis_x, "X axis")) return res;
        if (!check_axis_index(req.axis_y, "Y axis")) return res;
        if (req.axis_x.kind == OrderAxisKind::H && req.axis_y.kind == OrderAxisKind::H)
            return fail("both axes cannot sweep h");
        if (req.axis_x.kind == OrderAxisKind::Value && req.axis_y.kind == OrderAxisKind::Value
            && req.axis_x.index == req.axis_y.index)
            return fail("both axes sweep the same parameter");

        std::string err;
        if (!order_axis_nodes(req.axis_x, "X axis", res.axis_x_vals, err)) return fail(err);
        if (!order_axis_nodes(req.axis_y, "Y axis", res.axis_y_vals, err)) return fail(err);

        res.n_pts_x = (int)res.axis_x_vals.size();
        res.n_pts_y = (req.axis_y.kind == OrderAxisKind::None) ? 1 : (int)res.axis_y_vals.size();
        if (req.axis_y.kind == OrderAxisKind::None) res.axis_y_vals.assign(1, 0.0);

        const size_t total_cells = (size_t)res.n_pts_x * (size_t)res.n_pts_y;
        if (total_cells == 0) return fail("empty grid");

        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_fastsync

        // Эталон включается только когда есть И тело, И положительное число
        // подшагов: одно без другого — это просьба посчитать пустоту.
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);

        // Работа по ячейкам: h ячейки зависит только от той оси, что свипует h.
        auto cell_h = [&](int ix, int iy) -> double {
            double h = req.h;
            if (req.axis_x.kind == OrderAxisKind::H) h = res.axis_x_vals[(size_t)ix];
            if (req.axis_y.kind == OrderAxisKind::H) h = res.axis_y_vals[(size_t)iy];
            return h;
        };
        long long maxStepsPerCell = 1;
        double    totalSteps      = 0.0;
        for (int iy = 0; iy < res.n_pts_y; ++iy)
            for (int ix = 0; ix < res.n_pts_x; ++ix) {
                const long long N = order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps);
                if (N > maxStepsPerCell) maxStepsPerCell = N;
                totalSteps += (double)N;
            }
        // 1e15 шагов — это уже «никогда не досчитается», и (long long) ниже
        // всё равно переполнится на произведении с числом ячеек.
        if (totalSteps > 1.0e15) return fail("t_max / h too large: the work does not fit into a reasonable time");

        const int    progressStride = progress_stride_for((size_t)maxStepsPerCell);
        const double ticksTotal     = totalSteps / (double)progressStride;

        // Чанкование по ячейкам — не ради памяти (выход 4 числа на ячейку), а
        // ради TDR: один запуск на всю сетку при большом t_max/h легко
        // перевалит за watchdog. Бюджет — примерно столько вычислений правой
        // части, сколько GPU успевает за доли секунды; 7 вызовов КРС на
        // грубый шаг (1 + 2 + 4).
        const double kWorkBudget = 2.0e8;
        size_t cellsPerLaunch = (size_t)(kWorkBudget / (7.0 * (double)maxStepsPerCell));
        if (cellsPerLaunch < 256)          cellsPerLaunch = 256;
        if (cellsPerLaunch > total_cells)  cellsPerLaunch = total_cells;

        OrderDevBuf d_axisX, d_axisY, d_X0, d_values;
        OrderDevBuf d_p, d_e1, d_e2, d_eref, d_h, d_status;
        if (!d_axisX .alloc(res.axis_x_vals.size() * sizeof(numb), "axisXVals", err)) return fail(err);
        if (!d_axisY .alloc(res.axis_y_vals.size() * sizeof(numb), "axisYVals", err)) return fail(err);
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",        err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",    err)) return fail(err);
        if (!d_p     .alloc(total_cells * sizeof(numb), "outP",      err)) return fail(err);
        if (!d_e1    .alloc(total_cells * sizeof(numb), "outE1",     err)) return fail(err);
        if (!d_e2    .alloc(total_cells * sizeof(numb), "outE2",     err)) return fail(err);
        if (!d_eref  .alloc(total_cells * sizeof(numb), "outERef",   err)) return fail(err);
        if (!d_h     .alloc(total_cells * sizeof(numb), "outH",      err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);

        {
            const std::vector<numb> ax = to_numb(res.axis_x_vals);
            const std::vector<numb> ay = to_numb(res.axis_y_vals);
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_axisX.p,  ax.data(), ax.size() * sizeof(numb), "axisXVals")) return fail(err);
            if (!up(d_axisY.p,  ay.data(), ay.size() * sizeof(numb), "axisYVals")) return fail(err);
            if (!up(d_X0.p,     x0.data(), x0.size() * sizeof(numb), "X0"))        return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        RunSignals sig;
        if (!sig.alloc(err)) return fail(err);
        struct SigGuard { RunSignals& s; ~SigGuard() { s.release(); } } sig_guard{ sig };

        // Множитель порога полки округления. Само сравнение в ядре —
        // floorEps * |решение| * sqrt(4N): шум вычитания копится по шагам, так
        // что абсолютный порог обязан расти вместе с их числом (см. коммент в
        // order.template.cu). Здесь только константа при нём.
        const numb floorEps = (numb)2 * std::numeric_limits<numb>::epsilon();

        double ticksDone = 0.0;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int    nPtsX_arg  = res.n_pts_x;
            int    nPtsY_arg  = res.n_pts_y;
            int    nCells_arg = (int)count;
            int    offset_arg = (int)offset;
            numb*  axX_arg    = d_axisX.as<numb>();
            numb*  axY_arg    = d_axisY.as<numb>();
            int    axXKind    = (int)req.axis_x.kind;
            int    axXIndex   = req.axis_x.index;
            int    axYKind    = (int)req.axis_y.kind;
            int    axYIndex   = req.axis_y.index;
            numb*  X0_arg     = d_X0.as<numb>();
            numb*  values_arg = d_values.as<numb>();
            numb   hBase_arg  = (numb)req.h;
            numb   tMax_arg   = (numb)req.t_max;
            int    snap_arg   = req.snap_steps    ? 1 : 0;
            int    endp_arg   = req.endpoint_only ? 1 : 0;
            numb   maxV_arg   = (numb)req.max_value;
            numb   feps_arg   = floorEps;
            int    refSub_arg = ref_on ? req.ref_substeps : 0;
            numb*  outP_arg   = d_p.as<numb>();
            numb*  outE1_arg  = d_e1.as<numb>();
            numb*  outE2_arg  = d_e2.as<numb>();
            numb*  outERef_arg = d_eref.as<numb>();
            numb*  outH_arg   = d_h.as<numb>();
            int*   outSt_arg  = d_status.as<int>();
            int*   cancel_arg = sig.cancelArg();
            int*   prog_arg   = sig.progressArg();
            int    stride_arg = progressStride;

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &axX_arg, &axY_arg,
                &axXKind, &axXIndex, &axYKind, &axYIndex,
                &X0_arg, &values_arg,
                &hBase_arg, &tMax_arg, &snap_arg, &endp_arg, &maxV_arg, &feps_arg,
                &refSub_arg,
                &outP_arg, &outE1_arg, &outE2_arg, &outERef_arg, &outH_arg, &outSt_arg,
                &cancel_arg, &prog_arg, &stride_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            sig.resetTicks();
            CUresult r = cuLaunchKernel(cached_order.kernel, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(order): " + cu_err(r));
            if (!wait_with_signals(0, sig, req.cancel, req.progress, ticksDone, ticksTotal, err))
                return fail(err);
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("order kernel: ") + cudaGetErrorString(ce));

            // Работа чанка в тиках — сумма по его ячейкам, а не count*средняя:
            // при свипе по h соседние ячейки различаются на порядки.
            for (size_t c = offset; c < offset + count; ++c) {
                const int ix = (int)(c % (size_t)res.n_pts_x);
                const int iy = (res.n_pts_y > 1) ? (int)(c / (size_t)res.n_pts_x) : 0;
                ticksDone += (double)order_steps_for(cell_h(ix, iy), req.t_max, req.snap_steps)
                           / (double)progressStride;
            }

            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hp(total_cells), he1(total_cells), he2(total_cells),
                              herf(total_cells), hh(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_p.p,      hp.data(),         total_cells * sizeof(numb), "p"))      return fail(err);
            if (!dn(d_e1.p,     he1.data(),        total_cells * sizeof(numb), "e1"))     return fail(err);
            if (!dn(d_e2.p,     he2.data(),        total_cells * sizeof(numb), "e2"))     return fail(err);
            if (!dn(d_eref.p,   herf.data(),       total_cells * sizeof(numb), "eRef"))   return fail(err);
            if (!dn(d_h.p,      hh.data(),         total_cells * sizeof(numb), "h"))      return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);

            res.p.resize(total_cells); res.e1.resize(total_cells);
            res.e2.resize(total_cells); res.h_eff.resize(total_cells);
            res.e_ref.resize(total_cells);
            bool first_p = true, first_e = true, first_er = true;
            for (size_t i = 0; i < total_cells; ++i) {
                res.p[i]     = (double)hp[i];
                res.e1[i]    = (double)he1[i];
                res.e2[i]    = (double)he2[i];
                res.e_ref[i] = (double)herf[i];
                res.h_eff[i] = (double)hh[i];
                switch (res.status[i]) {
                    case ORDER_ST_DIVERGED:   ++res.n_diverged;   break;
                    case ORDER_ST_FLOOR:      ++res.n_floor;      break;
                    case ORDER_ST_NOCONTRACT: ++res.n_nocontract; break;
                    default:                  ++res.n_ok;         break;
                }
                // Диапазоны — только по чистым ячейкам (см. OrderResult::p_min).
                if (res.status[i] != ORDER_ST_OK) continue;
                if (std::isfinite(res.p[i])) {
                    if (first_p) { res.p_min = res.p_max = res.p[i]; first_p = false; }
                    else { if (res.p[i] < res.p_min) res.p_min = res.p[i];
                           if (res.p[i] > res.p_max) res.p_max = res.p[i]; }
                }
                if (std::isfinite(res.e1[i]) && res.e1[i] > 0.0) {
                    if (first_e) { res.e1_min = res.e1_max = res.e1[i]; first_e = false; }
                    else { if (res.e1[i] < res.e1_min) res.e1_min = res.e1[i];
                           if (res.e1[i] > res.e1_max) res.e1_max = res.e1[i]; }
                }
                if (std::isfinite(res.e_ref[i]) && res.e_ref[i] > 0.0) {
                    if (first_er) { res.eref_min = res.eref_max = res.e_ref[i]; first_er = false; }
                    else { if (res.e_ref[i] < res.eref_min) res.eref_min = res.e_ref[i];
                           if (res.e_ref[i] > res.eref_max) res.eref_max = res.e_ref[i]; }
                }
            }
        }

        res.ok = true;
        return res;
    }

// Stability: область устойчивости схемы; Performance: замер точность — затраты.

StabilityResult ParametricEngine::Impl::run_stability(const StabilityRequest& req) {
        StabilityResult res;
        auto fail = [&](const std::string& msg) -> StabilityResult& { res.error = msg; return res; };

        const std::string bad = stability_validate(req);
        if (!bad.empty()) return fail(bad);
        stability_fill_axes(req, res);

        const int    amountOfValues = (int)req.values.size();
        const size_t total_cells    = (size_t)res.n_pts_x * (size_t)res.n_pts_y;

        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки прошлого прогона, см. run_order

        // Модуль общий с Order: тот же ключ, та же компиляция, эталон выключен.
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  std::string(), err)) return fail(err);
        if (cached_order.kernel_stab == nullptr)
            return fail("stabilityRegionKernel not found in the module");

        OrderDevBuf d_sig, d_om, d_values, d_rho, d_status, d_R;
        if (!d_sig   .alloc(res.sigma_vals.size() * sizeof(numb), "sigmaVals", err)) return fail(err);
        if (!d_om    .alloc(res.omega_vals.size() * sizeof(numb), "omegaVals", err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",   err)) return fail(err);
        if (!d_rho   .alloc(total_cells * sizeof(numb), "outRho",    err)) return fail(err);
        if (!d_status.alloc(total_cells * sizeof(int),  "outStatus", err)) return fail(err);
        if (!d_R     .alloc(4 * total_cells * sizeof(numb), "outR",  err)) return fail(err);

        {
            const std::vector<numb> hs = to_numb(res.sigma_vals);
            const std::vector<numb> ho = to_numb(res.omega_vals);
            const std::vector<numb> va = to_numb(req.values);
            auto up = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { err = std::string("memcpy ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!up(d_sig.p,    hs.data(), hs.size() * sizeof(numb), "sigmaVals")) return fail(err);
            if (!up(d_om.p,     ho.data(), ho.size() * sizeof(numb), "omegaVals")) return fail(err);
            if (!up(d_values.p, va.data(), va.size() * sizeof(numb), "values"))    return fail(err);
        }

        const size_t cellsPerLaunch = 1u << 16;
        const size_t nLaunches = (total_cells + cellsPerLaunch - 1) / cellsPerLaunch;
        for (size_t L = 0; L < nLaunches; ++L) {
            const size_t offset = L * cellsPerLaunch;
            const size_t count  = (offset + cellsPerLaunch > total_cells) ? (total_cells - offset) : cellsPerLaunch;

            int   nPtsX_arg  = res.n_pts_x;
            int   nPtsY_arg  = res.n_pts_y;
            int   nCells_arg = (int)count;
            int   offset_arg = (int)offset;
            numb* sig_arg    = d_sig.as<numb>();
            numb* om_arg     = d_om.as<numb>();
            numb* values_arg = d_values.as<numb>();
            int   ia_arg = req.idx_a, ib_arg = req.idx_b, ic_arg = req.idx_c, id_arg = req.idx_d;
            numb  k_arg = (numb)req.k, r_arg = (numb)req.r, h_arg = (numb)req.h;
            numb* rho_arg = d_rho.as<numb>();
            int*  st_arg  = d_status.as<int>();
            numb* R_arg   = d_R.as<numb>();

            void* args[] = {
                &nPtsX_arg, &nPtsY_arg, &nCells_arg, &offset_arg,
                &sig_arg, &om_arg, &values_arg,
                &ia_arg, &ib_arg, &ic_arg, &id_arg,
                &k_arg, &r_arg, &h_arg,
                &rho_arg, &st_arg, &R_arg
            };

            const int blockSize = 64;
            const int gridSize  = (int)((count + blockSize - 1) / blockSize);
            CUresult r = cuLaunchKernel(cached_order.kernel_stab, gridSize, 1, 1, blockSize, 1, 1,
                                        0, nullptr, args, nullptr);
            if (r != CUDA_SUCCESS) return fail("cuLaunchKernel(stability): " + cu_err(r));
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return fail(std::string("stability kernel: ") + cudaGetErrorString(ce));

            if (req.progress)
                req.progress->store((float)((double)(offset + count) / (double)total_cells),
                                    std::memory_order_relaxed);
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }
        }

        {
            std::vector<numb> hrho(total_cells);
            res.status.assign(total_cells, 0);
            auto dn = [&](void* src, void* dst, size_t bytes, const char* what) -> bool {
                cudaError_t e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (e != cudaSuccess) { err = std::string("memcpy D2H ") + what + ": " + cudaGetErrorString(e); return false; }
                return true;
            };
            if (!dn(d_rho.p,    hrho.data(),       total_cells * sizeof(numb), "rho"))    return fail(err);
            if (!dn(d_status.p, res.status.data(), total_cells * sizeof(int),  "status")) return fail(err);
            std::vector<numb> hR(4 * total_cells);
            if (!dn(d_R.p, hR.data(), hR.size() * sizeof(numb), "R")) return fail(err);

            res.rho.resize(total_cells);
            for (size_t i = 0; i < total_cells; ++i) res.rho[i] = (double)hrho[i];

            // Ошибка шага — на хосте, той же функцией, что в CPU-ветке.
            res.err.assign(total_cells, std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < total_cells; ++i) {
                if (res.status[i] != STAB_ST_OK) continue;
                const size_t ix = i % (size_t)res.n_pts_x, iy = i / (size_t)res.n_pts_x;
                res.err[i] = stability_step_error(res.sigma_vals[ix], res.omega_vals[iy],
                                                  req.k, req.r, req.h,
                                                  (double)hR[i],                   (double)hR[total_cells + i],
                                                  (double)hR[2 * total_cells + i], (double)hR[3 * total_cells + i]);
            }
            stability_summarize(res);
        }

        res.ok = true;
        return res;
    }

PerfResult ParametricEngine::Impl::run_performance(const PerfRequest& req) {
        PerfResult res;
        res.axis = req.axis;
        auto fail = [&](const std::string& msg) -> PerfResult& { res.error = msg; return res; };

        if (req.axis.kind == OrderAxisKind::None) return fail("the axis is not set");
        if (req.repeats  < 1) return fail("the number of measurements must be >= 1");
        if (req.replicas < 1) return fail("the number of replicas must be >= 1");
        if (req.warmup   < 0) return fail("the number of warmup launches cannot be negative");

        // ---- Проход 1: ошибки (не засекается) ----
        OrderRequest oreq;
        oreq.krs_body           = req.krs_body;
        oreq.amountOfX          = req.amountOfX;
        oreq.initial_conditions = req.initial_conditions;
        oreq.values             = req.values;
        oreq.axis_x             = req.axis;
        oreq.axis_y             = OrderAxis{};          // kind == None -> одномерная сетка
        oreq.h                  = req.h;
        oreq.t_max              = req.t_max;
        oreq.snap_steps         = req.snap_steps;
        oreq.endpoint_only      = req.endpoint_only;
        oreq.max_value          = req.max_value;
        oreq.ref_krs_body       = req.ref_krs_body;
        oreq.ref_substeps       = req.ref_substeps;
        oreq.cancel             = req.cancel;

        // Прогресс order-прохода отдаётся в [0, 0.5]: без пересчёта бар
        // пробежал бы от нуля до конца дважды. run_order синхронный, поэтому
        // масштабированием занимается отдельный поток, живущий ровно на время
        // вызова.
        std::shared_ptr<std::atomic<float>> sub_prog;
        std::atomic<bool> watch_stop{ false };
        std::thread watcher;
        if (req.progress) {
            sub_prog = std::make_shared<std::atomic<float>>(0.0f);
            oreq.progress = sub_prog;
            watcher = std::thread([&req, &sub_prog, &watch_stop]() {
                while (!watch_stop.load(std::memory_order_relaxed)) {
                    req.progress->store(0.5f * sub_prog->load(std::memory_order_relaxed),
                                        std::memory_order_relaxed);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            });
        }
        OrderResult ores = run_order(oreq);
        if (watcher.joinable()) { watch_stop.store(true, std::memory_order_relaxed); watcher.join(); }

        if (ores.cancelled) { res.cancelled = true; return res; }
        if (!ores.ok) return fail(ores.error.empty() ? std::string("the order pass failed") : ores.error);

        res.n_pts     = ores.n_pts_x;
        res.axis_vals = ores.axis_x_vals;
        res.e1        = ores.e1;
        res.e2        = ores.e2;
        res.e_ref     = ores.e_ref;
        res.p         = ores.p;
        res.h_eff     = ores.h_eff;
        res.status    = ores.status;
        res.n_ok         = ores.n_ok;
        res.n_diverged   = ores.n_diverged;
        res.n_floor      = ores.n_floor;
        res.n_nocontract = ores.n_nocontract;
        res.repeats   = req.repeats;
        res.warmup    = req.warmup;
        res.replicas  = req.replicas;

        const int n = res.n_pts;
        if (n <= 0) return fail("empty grid");
        const double qnan = std::numeric_limits<double>::quiet_NaN();
        res.t_min.assign((size_t)n, qnan);
        res.t_max.assign((size_t)n, qnan);
        res.t_avg.assign((size_t)n, qnan);
        res.n_steps.assign((size_t)n, 0);

        // ---- Проход 2: замер ----
        std::string err;
        if (!ensure_init(err)) return fail(err);
        cuCtxSetCurrent(context);
        cudaGetLastError();   // сброс sticky-ошибки, см. run_order

        const int amountOfValues = (int)req.values.size();
        const bool ref_on = !req.ref_krs_body.empty() && req.ref_substeps > 0;
        if (!compile_order_module(true, req.amountOfX, amountOfValues, req.krs_body,
                                  ref_on ? req.ref_krs_body : std::string(), err)) return fail(err);
        if (cached_order.kernel_perf == nullptr) return fail("perfIntegrateKernel not found in the module");

        OrderDevBuf d_X0, d_values, d_out;
        if (!d_X0    .alloc((size_t)req.amountOfX  * sizeof(numb), "X0",      err)) return fail(err);
        if (!d_values.alloc((size_t)amountOfValues * sizeof(numb), "values",  err)) return fail(err);
        if (!d_out   .alloc((size_t)req.replicas * (size_t)req.amountOfX * sizeof(numb), "perfOut", err)) return fail(err);

        std::vector<numb> h_vals = to_numb(req.values);
        {
            const std::vector<numb> x0 = to_numb(req.initial_conditions);
            cudaError_t e = cudaMemcpy(d_X0.p, x0.data(), x0.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy X0: ") + cudaGetErrorString(e));
            e = cudaMemcpy(d_values.p, h_vals.data(), h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
            if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
        }

        // Время меряет само ядро (%globaltimer вокруг цикла шагов, см. perfIntegrateKernel):
        // события вокруг запуска захватывали ещё и задержку запуска (~6-8 мкс на WDDM).
        OrderDevBuf d_tspan;
        if (!d_tspan.alloc(2 * sizeof(unsigned long long), "perfTimer", err)) return fail(err);

        // y(T) узла — для E(T) против эталона y*(T) (run_performance_any), как у адаптивного.
        res.y_end.assign((size_t)n, std::vector<double>());

        // Ширина блока — 32: измерено, что дальше упирается в регистры, а не в
        // лимит блоков на SM. При replicas == 1 это один активный warp.
        const int blockSize = 32;
        const int gridSize  = (req.replicas + blockSize - 1) / blockSize;
        bool first_node = true;

        for (int i = 0; i < n; ++i) {
            if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                res.cancelled = true;
                return res;
            }

            const double h_node = res.h_eff[(size_t)i];
            if (!(h_node > 0.0) || !std::isfinite(h_node)) continue;   // узел без времени: см. status

            // Свип по параметру меняет a[] от узла к узлу. Загрузка делается
            // здесь, ДО прогрева, и в замер не попадает.
            if (req.axis.kind == OrderAxisKind::Value) {
                const int vi = req.axis.index;
                if (vi >= 0 && vi < amountOfValues) {
                    h_vals[(size_t)vi] = (numb)res.axis_vals[(size_t)i];
                    cudaError_t e = cudaMemcpy(d_values.p, h_vals.data(),
                                               h_vals.size() * sizeof(numb), cudaMemcpyHostToDevice);
                    if (e != cudaSuccess) return fail(std::string("memcpy values: ") + cudaGetErrorString(e));
                }
            }

            const long long N = order_steps_for(h_node, req.t_max, req.snap_steps);
            res.n_steps[(size_t)i] = N;

            int       nt_arg  = req.replicas;
            numb*     X0_arg  = d_X0.as<numb>();
            numb*     val_arg = d_values.as<numb>();
            numb      h_arg   = (numb)h_node;
            long long n_arg   = N;
            numb*     out_arg = d_out.as<numb>();
            unsigned long long* ts_none = nullptr;
            unsigned long long* ts_arg  = d_tspan.as<unsigned long long>();
            void* args_warm[] = { &nt_arg, &X0_arg, &val_arg, &h_arg, &n_arg, &out_arg, &ts_none };
            void* args[]      = { &nt_arg, &X0_arg, &val_arg, &h_arg, &n_arg, &out_arg, &ts_arg };

            auto launch = [&](const char* what, void** a) -> bool {
                CUresult r = cuLaunchKernel(cached_order.kernel_perf, gridSize, 1, 1, blockSize, 1, 1,
                                            0, nullptr, a, nullptr);
                if (r != CUDA_SUCCESS) { err = std::string("cuLaunchKernel(") + what + "): " + cu_err(r); return false; }
                return true;
            };
            auto sync = [&]() -> bool {
                cudaError_t ce = cudaDeviceSynchronize();
                if (ce != cudaSuccess) { err = std::string("perf kernel: ") + cudaGetErrorString(ce); return false; }
                return true;
            };

            // Прогрев по времени: GPU поднимает частоту только под сплошной нагрузкой, а
            // короткие запуски с синхронизацией между ними её не дают — без этого одно и то же
            // ядро мерилось то на холостой частоте, то на рабочей (до 3-5 раз разницы).
            // Пачка запусков подряд, без синхронизаций: ~300 мс перед первым узлом, ~20 мс
            // перед каждым следующим. Потом — warmup запусков из настроек, как раньше.
            if (!perf_burst_warmup(first_node ? 300.0 : 20.0,
                                   [&]() { return launch("perf warmup", args_warm); }, sync))
                return fail(err);
            first_node = false;
            for (int w = 0; w < req.warmup; ++w)
                if (!launch("perf warmup", args_warm) || !sync()) return fail(err);

            double tmin = 0.0, tmax = 0.0, tsum = 0.0;
            int got = 0;
            for (int rep = 0; rep < req.repeats; ++rep) {
                const unsigned long long ts_init[2] = { ~0ULL, 0ULL };
                cudaError_t ce = cudaMemcpy(d_tspan.p, ts_init, sizeof(ts_init), cudaMemcpyHostToDevice);
                if (ce != cudaSuccess) return fail(std::string("memcpy timer: ") + cudaGetErrorString(ce));
                if (!launch("perf", args) || !sync()) return fail(err);
                unsigned long long ts[2] = { 0, 0 };
                ce = cudaMemcpy(ts, d_tspan.p, sizeof(ts), cudaMemcpyDeviceToHost);
                if (ce != cudaSuccess) return fail(std::string("memcpy timer: ") + cudaGetErrorString(ce));
                const double us = ts[1] >= ts[0] ? (double)(ts[1] - ts[0]) * 1.0e-3 : 0.0;
                if (got == 0) { tmin = tmax = us; }
                else { if (us < tmin) tmin = us; if (us > tmax) tmax = us; }
                tsum += us;
                ++got;

                if (req.cancel && req.cancel->load(std::memory_order_relaxed)) {
                    res.cancelled = true;
                    return res;
                }
            }
            if (got > 0) {
                res.t_min[(size_t)i] = tmin;
                res.t_max[(size_t)i] = tmax;
                res.t_avg[(size_t)i] = tsum / (double)got;
            }

            // y(T) первой реплики — только если узел кончился ровно в t_max (fit h to t_max):
            // иначе сравнивать с y*(T) нечего, E(T) узла остаётся пустым.
            if ((double)N * h_node > 0.0
                && std::fabs((double)N * h_node - req.t_max) <= 1e-9 * (std::max)(1.0, std::fabs(req.t_max))) {
                std::vector<numb> y((size_t)req.amountOfX);
                cudaError_t ce = cudaMemcpy(y.data(), d_out.p, y.size() * sizeof(numb), cudaMemcpyDeviceToHost);
                if (ce != cudaSuccess) return fail(std::string("memcpy y(T): ") + cudaGetErrorString(ce));
                res.y_end[(size_t)i].assign(y.begin(), y.end());
            }

            if (req.progress)
                req.progress->store(0.5f + 0.5f * (float)(i + 1) / (float)n, std::memory_order_relaxed);
        }

        // Диапазоны для автоскейла: только узлы, у которых есть И время, И
        // конечная положительная ошибка — точка графика существует лишь тогда,
        // когда есть обе координаты.
        bool first_t = true, first_e1 = true, first_e2 = true, first_er = true;
        for (int i = 0; i < n; ++i) {
            const double ta = res.t_avg[(size_t)i];
            if (!std::isfinite(ta)) continue;
            const double lo = res.t_min[(size_t)i], hi = res.t_max[(size_t)i];
            if (first_t) { res.t_lo = lo; res.t_hi = hi; first_t = false; }
            else { if (lo < res.t_lo) res.t_lo = lo; if (hi > res.t_hi) res.t_hi = hi; }
            const double e1 = res.e1[(size_t)i], e2 = res.e2[(size_t)i];
            if (std::isfinite(e1) && e1 > 0.0) {
                if (first_e1) { res.e1_min = res.e1_max = e1; first_e1 = false; }
                else { if (e1 < res.e1_min) res.e1_min = e1; if (e1 > res.e1_max) res.e1_max = e1; }
            }
            if (std::isfinite(e2) && e2 > 0.0) {
                if (first_e2) { res.e2_min = res.e2_max = e2; first_e2 = false; }
                else { if (e2 < res.e2_min) res.e2_min = e2; if (e2 > res.e2_max) res.e2_max = e2; }
            }
            const double er = ((size_t)i < res.e_ref.size())
                                  ? res.e_ref[(size_t)i]
                                  : std::numeric_limits<double>::quiet_NaN();
            if (std::isfinite(er) && er > 0.0) {
                if (first_er) { res.eref_min = res.eref_max = er; first_er = false; }
                else { if (er < res.eref_min) res.eref_min = er; if (er > res.eref_max) res.eref_max = er; }
            }
        }

        res.ok = true;
        return res;
    }

bool ParametricEngine::Impl::compile_network_module(bool activate, int amountOfX, int amountOfValues,
                                const std::string& krs_body, const std::string& coupling_body,
                                std::string& err)
    {
        cuCtxSetCurrent(context);
        // Тело связи входит в ключ: оно подставляется в switch внутри ядра, и
        // смена закона связи — такая же перекомпиляция, как смена КРС.
        const std::string key = hash_key(krs_body, amountOfX) + ":net:v"
                              + std::to_string(amountOfValues)
                              + ":c" + std::to_string(std::hash<std::string>{}(coupling_body));
        return compile_into(pool_network, key, cached_network, activate, [&](CachedNetworkModule& fresh) {
            CUmodule mod = nullptr;
            std::vector<std::string> lowered;
            if (!build_module(snapshot_sources(src_template_network), "network.cu",
                              { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                                { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                                { "{{KRS_BODY}}",         krs_body },
                                { "{{COUPLING_BODY}}",    coupling_body } },
                              {}, mod, lowered, err))
                return false;

            fresh.key    = key;
            fresh.module = mod;
            if (!module_fn(mod, "networkIntegrateKernel", fresh.kernel, err)) { cuModuleUnload(mod); return false; }
            return true;
        }, err);
    }
