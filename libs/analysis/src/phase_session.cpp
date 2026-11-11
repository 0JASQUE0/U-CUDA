#include "session_internal.h"
#include "ucuda/cuda/rqa.h"
using namespace ucuda::analysis_detail;


void PhaseAnalysisSession::add_ic() {
    InitialConditionSet ic;
    ic.label = "IC " + std::to_string(ic_sets.size() + 1);
    for (const auto& v : vars) ic.values[v] = "";
    ic_sets.push_back(ic);
}


void PhaseAnalysisSession::remove_ic(int i) {
    if (i >= 0 && i < (int)ic_sets.size()) ic_sets.erase(ic_sets.begin() + i);
}


void PhaseAnalysisSession::remove_projection(int i) {
    if (i >= 0 && i < (int)projections.size()) {
        projections.erase(projections.begin() + i);
    }
}


void PhaseAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);
    step_h = default_h_from_record(r);
    scheme = default_scheme_from_record(r, scheme);
    symmetry_s = r.symmetry_s.empty() ? "0.5" : r.symmetry_s;
    cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    newton = newton_from_record(r);
    sim_time = "50";
    skip_time = "10";

    param_values.clear();
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }

    ic_sets.clear();
    InitialConditionSet ic;
    ic.label = "IC 1";
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        ic.values[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    ic_sets.push_back(ic);

    // одна проекция по умолчанию
    projections.clear();
    add_projection();
    // add_projection() reads sys.is_map, but sys is only assigned after this
    // call (see AppModel::start_phase_analysis) — take it from the record here.
    for (auto& p : projections) p.draw_points = r.is_map;

    result = AnalysisResult{};
}

namespace ucuda::analysis_detail {


// Ветка адаптивного шага для compute_phase_portrait: GPU (ядро phase_kernel_ad) или
// CPU (computePhasePortraitCPU_adaptive) — один и тот же драйвер. raw[k] — траектория
// НУ k, times[k] — моменты её точек на узлах шага (на равномерной сетке пусто).
// false — ошибка настройки или компиляции, текст уже в result.error; расхождение
// траектории ошибкой не считается и уходит в calc_ok / calc_err, как у постоянного шага.
bool run_adaptive_phase(const PhaseRunInputs& in, int dim, int N,
                               const std::vector<double>& ic_flat, const std::vector<double>& a,
                               double h, double tsim, double tskip, int total,
                               std::vector<std::vector<std::vector<double>>>& raw,
                               std::vector<std::vector<double>>& times,
                               AnalysisResult& result, bool& calc_ok, std::string& calc_err) {
    if (!adaptive_scheme_name_ok(in.scheme)) {
        result.error = adaptive_scheme_hint();
        return false;
    }
    // Многошаговый BDF 1-5 — свой драйвер (multistep_bdf.h), CPU, тот же вывод, что у адаптивных схем.
    if (scheme_is_multistep(in.scheme)) {
        const bool dae = in.sys.dae_nx >= 0;
        if (dae != (in.scheme == kBdfSchemeDae)) {
            result.error = dae ? "a DAE needs DAE BDF 1-5" : "DAE BDF 1-5 needs a DAE system";
            return false;
        }
        BdfSettings bs;
        bs.rtol = parse_val(in.adaptive.rtol, 1e-6);
        bs.atol.clear();
        {
            std::string cur;
            for (char c : in.adaptive.atol + ",") {
                if (c == ',' || c == ';') { if (!cur.empty()) bs.atol.push_back(parse_val(cur, 1e-8)); cur.clear(); }
                else if (c != ' ') cur += c;
            }
            if (bs.atol.empty()) bs.atol.push_back(1e-8);
        }
        bs.h0 = parse_val(in.adaptive.h0, 0.0);
        if (!in.adaptive.hmax.empty()) bs.hmax = parse_val(in.adaptive.hmax, bs.hmax);
        if (!in.adaptive.hmin.empty()) bs.hmin = parse_val(in.adaptive.hmin, 0.0);
        if (!in.adaptive.enabled) bs.hfix = h;   // переключатель Fixed step: шаг h без регулятора
        {   // Ньютон — хвост a[] вкладки (scheme_fill_tail): [n, tol, max_iters, full] после p_M
            const size_t ns = (size_t)newton_slot(in.sys);
            if (ns + 2 < a.size()) {
                bs.newton_tol = a[ns]; bs.newton_max = (int)a[ns + 1]; bs.newton_full = a[ns + 2] != 0;
            }
        }
        const double t_skip = in.skip_transient ? tskip : 0.0;
        const bool rawm = in.adaptive.raw_nodes && in.adaptive.enabled;   // постоянный шаг — сетка через h
        int max_pts = parse_num_int(in.adaptive.max_points, 1000000);
        max_pts = std::max(2, std::min(max_pts, 50000000));
        result.adaptive  = true;
        result.raw_nodes = rawm;
        result.step_log.assign((size_t)N, {});
        result.step_stats.assign((size_t)N, AdaptiveStats());
        raw.assign((size_t)N, {});
        times.assign((size_t)N, {});
        if (in.use_gpu) {
            // GPU: тот же алгоритм (kernels/ucuda_bdf.cuh), тела F и J — кодоген над системой после GGL
            System sp;
            std::string e;
            if (!bdf_prepare_system(in.sys, sp, &e)) { result.error = e; return false; }
            BdfGpuRequest rq;
            try { codegen_bdf_bodies(sp, rq.res, rq.jac); }
            catch (const std::exception& ex) { result.error = std::string("BDF codegen: ") + ex.what(); return false; }
            const int ng = (int)sp.vars.size();
            rq.amountOfX = ng;
            rq.nx = sp.dae_nx < 0 ? ng : sp.dae_nx;
            rq.kidx.assign((size_t)ng, 1);
            for (int i = 0; i < ng && i < (int)sp.dae_index.size(); ++i) rq.kidx[(size_t)i] = std::max(1, sp.dae_index[(size_t)i]);
            for (int i = 0; i < ng; ++i) rq.atol.push_back(bs.atol[(size_t)std::min(i, (int)bs.atol.size() - 1)]);
            rq.rtol = bs.rtol; rq.h0 = bs.h0; rq.hmax = std::isfinite(bs.hmax) ? bs.hmax : 0.0; rq.maxk = bs.max_order;
            rq.hmin = bs.hmin; rq.hfix = bs.hfix;
            rq.newton_max = bs.newton_max; rq.newton_full = bs.newton_full; rq.newton_tol = bs.newton_tol;
            rq.N = N;
            rq.ic_flat.assign((size_t)N * ng, 0.0);   // mu системы GGL — с нуля
            for (int k = 0; k < N; ++k)
                for (int i = 0; i < dim; ++i) rq.ic_flat[(size_t)k * ng + i] = ic_flat[(size_t)k * dim + i];
            rq.values = a;
            rq.raw = rawm; rq.t_skip = t_skip; rq.t_rec = tsim; rq.dt = h; rq.total = total; rq.max_pts = max_pts;
            rq.log_cap = kAdaptLogCap;   // журнал попыток — проекция "Step size"
            rq.cancel = in.cancel;
            PhaseAdaptiveResult out;
            if (!computePhasePortraitsBdfNVRTC(rq, out, &e)) {
                if (e == kNvrtcCancelled) { result.cancelled = true; result.error = e; }
                else result.error = "GPU: " + e;
                return false;
            }
            for (int k = 0; k < N; ++k) {
                raw[(size_t)k] = std::move(out.traj[(size_t)k]);
                for (auto& row : raw[(size_t)k]) row.resize((size_t)dim);   // без mu
                if (rawm) times[(size_t)k] = std::move(out.times[(size_t)k]);
                result.step_log[(size_t)k] = std::move(out.log[(size_t)k]);
                result.step_stats[(size_t)k] = out.stats[(size_t)k];
                if (out.hmin_fail[(size_t)k]) {
                    result.error = bs.hfix > 0 ? "BDF (fixed step): Newton does not converge - lower h or raise Newton max iters"
                                 : bs.hmin > 0 ? "BDF: Newton does not converge at h_min - lower h_min"
                                               : "BDF: step size below h_min";
                    return false;
                }
            }
        } else
        for (int k = 0; k < N; ++k) {
            BdfStats bst;
            std::string e;
            if (!bdf_integrate(in.sys, a.data(), (int)a.size(), &ic_flat[(size_t)k * dim], bs, t_skip, tsim, h, total,
                               rawm, max_pts, raw[(size_t)k], times[(size_t)k], bst, &e, in.cancel.get(),
                               &result.step_log[(size_t)k], kAdaptLogCap)) {
                result.error = e;
                return false;
            }
            AdaptiveStats& as = result.step_stats[(size_t)k];
            as.nacc = (double)bst.nacc; as.nrej = (double)bst.nrej; as.nforced = (double)bst.nforced;
            as.nrhs = (double)bst.nrhs;
            as.hmin = bst.hmin; as.hmax = bst.hmax; as.hmean = bst.nacc ? bst.hsum / (double)bst.nacc : 0.0;
            as.diverged = bst.diverged; as.truncated = bst.truncated;
            if (in.cancel && in.cancel->load(std::memory_order_relaxed)) break;
        }
        if (in.cancel && in.cancel->load(std::memory_order_relaxed)) {
            result.cancelled = true;
            result.error = kNvrtcCancelled;
            return false;
        }
        for (int k = 0; k < N; ++k) {
            const AdaptiveStats& st = result.step_stats[(size_t)k];
            if (st.diverged) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[(size_t)k].label + "' diverged (nan/inf)"; }
            else if (st.truncated) {
                calc_ok = false;
                calc_err = "trajectory '" + in.ic_sets[(size_t)k].label + "' reached max points before the end of the interval";
            }
        }
        return true;
    }
    AdaptiveCode code;
    try { code = adaptive_code_for_scheme(in.custom_schemes, in.sys, in.scheme); }
    catch (const std::exception& e) { result.error = std::string("adaptive codegen: ") + e.what(); return false; }

    const double t_skip = in.skip_transient ? tskip : 0.0;
    UcudaAdaptParams P;
    std::string perr, ctrl_body;
    if (!adaptive_build_params(in.adaptive, code, dim, t_skip + tsim, P, perr, &ctrl_body)) {
        result.error = perr;
        return false;
    }
    const bool rawm = in.adaptive.raw_nodes;
    int max_pts = parse_num_int(in.adaptive.max_points, 1000000);
    if (max_pts < 2) max_pts = 2;
    if (max_pts > 50000000) max_pts = 50000000;

    result.adaptive  = true;
    result.raw_nodes = rawm;
    result.step_log.assign((size_t)N, {});
    result.step_stats.assign((size_t)N, AdaptiveStats());
    raw.assign((size_t)N, {});
    times.assign((size_t)N, {});

    if (in.use_gpu) {
        PhaseAdaptiveRequest rq;
        rq.rhs = code.rhs; rq.emb = code.emb; rq.dprep = code.dprep; rq.deval = code.deval;
        rq.ctrl_body = ctrl_body;
        rq.amountOfX = dim; rq.ic_flat = ic_flat; rq.N = N; rq.values = a; rq.params = P;
        rq.raw = rawm; rq.t_skip = t_skip; rq.t_rec = tsim; rq.dt = h; rq.total = total;
        rq.max_pts = max_pts; rq.log_cap = kAdaptLogCap;
        rq.cancel = in.cancel;
        PhaseAdaptiveResult out;
        std::string err;
        if (!computePhasePortraitsAdaptiveNVRTC(rq, out, &err)) {
            if (err == kNvrtcCancelled) { result.cancelled = true; result.error = err; }
            else result.error = "GPU: " + err;
            return false;
        }
        raw = std::move(out.traj);
        if (rawm) times = std::move(out.times);
        result.step_log   = std::move(out.log);
        result.step_stats = out.stats;
    }
    else if (in.sys.dae_nx >= 0 || !scheme_supports_adaptive(scheme_from_name(in.scheme)) || system_uses_time(in.sys)
             || !in.sys.out_exprs.empty() || !in.sys.zb_g.empty()) {   // z (Solve as ODE) — тоже: вычислитель их не знает
        // Экстраполятор (Extr, GBS): CPU-драйвер exe считает по таблицам Бутчера и тело базы в
        // стадиях не выразит. Те же тела, что у GPU, собирает cl.exe в DLL адаптивного модуля
        // (AdaptiveCpuModule, кэш на диске), траектория — ucuda_cpu_ad_phase, копия phase_kernel_ad;
        // регулятор (и пользовательский) вкомпилирован туда же. Явное t — тоже сюда: exe-драйвер
        // (вычислитель правых частей) времени не знает, а тела модуля его получают от драйвера.
        AdaptiveCpuModule mod;
        std::vector<KrsCpuDiag> diags;
        if (!mod.compile(code.rhs, code.emb, code.dprep, code.deval, ctrl_body, dim, "", diags) || !mod.phase()) {
            result.error = "adaptive " + in.scheme + " (CPU build):";
            for (const KrsCpuDiag& d : diags)
                result.error += "\n" + (d.line > 0 ? "line " + std::to_string(d.line) + ": " : std::string()) + d.message;
            return false;
        }
        // Буфер DLL — под весь потолок сразу (cap точек): max points до 5e7 — это гигабайты,
        // и нехватку памяти нужно отдать текстом, а не исключением из потока расчёта.
        const int cap = rawm ? max_pts : total;
        std::vector<double> data, tm, lg;
        try {
            data.resize((size_t)std::max(cap, 1) * dim);
            tm.resize(rawm ? (size_t)max_pts : 0);
            lg.resize((size_t)kAdaptLogCap * 4);
        } catch (const std::bad_alloc&) {
            result.error = "adaptive " + in.scheme + " (CPU): not enough memory for "
                         + std::to_string(cap) + " output points; lower max points or the record time";
            return false;
        }
        // Отмена: DLL смотрит volatile-флаг, его ставит этот наблюдатель по токену сессии.
        volatile int cflag = 0;
        std::atomic<bool> done{ false };
        std::thread watcher([&]() {
            while (!done.load(std::memory_order_relaxed)) {
                if (in.cancel && in.cancel->load(std::memory_order_relaxed)) cflag = 1;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
        struct JoinOnExit {
            std::atomic<bool>& d; std::thread& t;
            ~JoinOnExit() { d.store(true, std::memory_order_relaxed); t.join(); }
        } join_on_exit{ done, watcher };
        for (int k = 0; k < N; ++k) {
            if (in.cancel && in.cancel->load(std::memory_order_relaxed)) break;
            int nlog = 0;
            double st[9] = {}, fh = 0;
            const int c = mod.phase()(&ic_flat[(size_t)k * dim], a.data(), &P, t_skip, tsim, h, total, rawm ? 1 : 0,
                                      max_pts, kAdaptLogCap, data.data(), rawm ? tm.data() : nullptr, lg.data(),
                                      &nlog, st, &fh, &cflag);
            std::vector<std::vector<double>>& tr = raw[(size_t)k];
            tr.assign((size_t)c, std::vector<double>((size_t)dim));
            for (int i = 0; i < c; ++i)
                for (int j = 0; j < dim; ++j) tr[(size_t)i][(size_t)j] = data[(size_t)i * dim + j];
            if (rawm) times[(size_t)k].assign(tm.begin(), tm.begin() + c);
            result.step_log[(size_t)k].assign(lg.begin(), lg.begin() + (size_t)nlog * 4);
            AdaptiveStats& as = result.step_stats[(size_t)k];
            as.nacc = st[0]; as.nrej = st[1]; as.nforced = st[2]; as.nrhs = st[3];
            as.hmin = st[4]; as.hmax = st[5]; as.hmean = st[6];
            as.diverged = st[7] != 0; as.truncated = st[8] != 0;
        }
    }
    else {
        std::unique_ptr<SystemEvaluator> ev;
        try { ev.reset(new SystemEvaluator(in.sys)); }
        catch (const std::exception& e) { result.error = std::string("parse error: ") + e.what(); return false; }
        const IntScheme isch = int_scheme_from_string(in.scheme);
        // Пользовательский регулятор на CPU — тело, собранное cl.exe (кэш на диске).
        CtrlCpuFn ctrl;
        if (!ctrl_body.empty()) {
            std::vector<KrsCpuDiag> diags;
            if (!ctrl.compile(ctrl_body, diags)) {
                result.error = "step controller '" + in.adaptive.ctrl + "' (CPU build):";
                for (const KrsCpuDiag& d : diags)
                    result.error += "\n" + (d.line > 0 ? "line " + std::to_string(d.line) + ": " : std::string())
                                  + d.message;
                return false;
            }
        }
        for (int k = 0; k < N; ++k) {
            double fh = 0;
            computePhasePortraitCPU_adaptive(*ev, isch, &ic_flat[(size_t)k * dim], dim, a.data(), P,
                                             rawm, t_skip, tsim, h, total, max_pts, kAdaptLogCap,
                                             raw[(size_t)k], times[(size_t)k],
                                             result.step_log[(size_t)k], result.step_stats[(size_t)k], fh,
                                             ctrl.fn(), ctrl.prep(), in.cancel.get());
            if (in.cancel && in.cancel->load(std::memory_order_relaxed)) break;
        }
    }
    if (in.cancel && in.cancel->load(std::memory_order_relaxed)) {
        result.cancelled = true;
        result.error = kNvrtcCancelled;
        return false;
    }
    for (int k = 0; k < N; ++k) {
        const AdaptiveStats& st = result.step_stats[(size_t)k];
        if (st.diverged) {
            calc_ok = false;
            calc_err = "trajectory '" + in.ic_sets[(size_t)k].label + "' diverged (nan/inf)";
        }
        else if (st.truncated) {
            calc_ok = false;
            calc_err = "trajectory '" + in.ic_sets[(size_t)k].label
                     + "' reached max points before the end of the interval";
        }
    }
    trim_step_log_to_record(result, t_skip);
    return true;
}
}

namespace ucuda::analysis_detail {


AnalysisResult compute_phase_portrait(const PhaseRunInputs& in) {
    AnalysisResult result;
    auto _t0 = std::chrono::high_resolution_clock::now();

    int dim = (int)in.vars.size();
    if (dim < 1) { result.error = "no variables"; return result; }
    double h = parse_val(in.step_h, 0.01);
    if (h <= 0) { result.error = "step h must be > 0"; return result; }
    double tsim = parse_val(in.sim_time, 50.0);
    double tskip = parse_val(in.skip_time, 0.0);
    int total = (int)(tsim / h); if (total <= 0) total = 1;
    int skip = in.skip_transient ? (int)(tskip / h) : 0;
    if (skip < 0) skip = 0;

    // параметры: a[0] зарезервирован под CD-коэф. симметрии, a[1..] = params
    int nparams = (int)in.params.size();
    std::vector<double> a(nparams + 1 + kSchemeTailSlots, 0.0);   // [s, p_1..p_M, n, Newton]
    a[0] = parse_val(in.symmetry_s, 0.5);
    scheme_fill_tail(a.data() + nparams + 1, in.cd_fp_iters, in.newton); // a[M+1..M+4]: CD FP n, Newton
    for (int j = 0; j < nparams; ++j) {
        auto it = in.param_values.find(in.params[j]);
        a[1 + j] = parse_val(it != in.param_values.end() ? it->second : "", 0.0);
    }

    int dec = parse_num_int(in.decimation, 1); if (dec < 1) dec = 1;

    int N = (int)in.ic_sets.size();
    if (N < 1) { result.error = "no initial conditions"; return result; }

    std::vector<double> ic_flat((size_t)N * dim, 0.0);
    // Continuation seeds ic_flat from the previous chunk's final state; skip
    // parsing so a Lorenz-scale double reaches the kernel bit-exact.
    const bool use_override = (int)in.ic_override.size() == N &&
                              (in.ic_override.empty() || (int)in.ic_override[0].size() >= dim);
    for (int k = 0; k < N; ++k) {
        if (use_override) {
            for (int i = 0; i < dim; ++i)
                ic_flat[(size_t)k * dim + i] = in.ic_override[k][i];
        }
        else {
            const auto& ic = in.ic_sets[k];
            for (int i = 0; i < dim; ++i) {
                auto it = ic.values.find(in.vars[i]);
                ic_flat[(size_t)k * dim + i] = parse_val(it != ic.values.end() ? it->second : "", 0.0);
            }
        }
    }

    // ДАУ: согласованные НУ — z из g(x0, z, 0) = 0 при заданных x0 (индекс 1). Поля НУ не
    // меняются: заданное z — лишь старт Ньютона. Продолжение (ic_override) несёт состояние,
    // уже согласованное шагом, — его не трогаем.
    if (in.sys.dae_nx >= 0 && in.sys.dae_nx < dim && !use_override) {
        int solved = 0;
        std::string why, done;   // done — что сделано при индексе >= 2 (скрытые связи), пусто при индексе 1
        for (int k = 0; k < N; ++k) {
            std::string w;
            if (dae_consistent_z(in.sys, a.data(), (int)a.size(), 0.0, &ic_flat[(size_t)k * dim], &w)) {
                ++solved;
                if (done.empty()) done = w;
            }
            else if (why.empty()) why = w;
        }
        result.dae_status = solved == N
            ? (done.empty() ? std::string("initial z made consistent: g(x0, z) = 0") : "initial values: " + done)
            : "initial z: " + why;
    }

    std::vector<std::vector<std::vector<double>>> raw;
    bool calc_ok = true;
    std::string calc_err;

    // Адаптивный шаг — своя ветка расчёта; признаки, RQA и прореживание ниже общие.
    // У отображения шага нет, и переключатель для него игнорируется.
    std::vector<std::vector<double>> ad_times;
    // Явное время t на CPU: фиксированный шаг — через krs_cpu (номер шага ведёт DLL, см.
    // KrsCpuStep::step_index), адаптивный — через DLL адаптивного модуля (run_adaptive_phase).
    const bool timed = system_uses_time(in.sys);
    if ((in.adaptive.enabled || scheme_is_multistep(in.scheme)) && !in.sys.is_map) {
        if (!run_adaptive_phase(in, dim, N, ic_flat, a, h, tsim, tskip, total, raw, ad_times,
                                result, calc_ok, calc_err))
            return result;
    }
    else if (in.use_gpu) {
        if (in.krs_code.empty()) {
            result.error = in.krs_error.empty() ? "no KRS generated (check system)" : in.krs_error;
            return result;
        }
        calc_ok = computePhasePortraitsNVRTC(in.krs_code, dim,
            ic_flat.data(), N, a.data(), (int)a.size(),
            h, total, skip, raw, &calc_err, in.step0);
        if (!calc_ok) {
            result.error = "GPU: " + calc_err;
            return result;
        }
    }
    // ДАУ на CPU: тело DAE-метода — обычный шаг над [x; z], его собирает в нативную функцию
    // krs_cpu, как пользовательскую КРС (вычислитель правых частей ДАУ не считает).
    // Явное время и z-выходы (Solve as ODE) — тоже здесь: вычислитель правых частей их не знает.
    else if (in.krs_is_custom || in.sys.dae_nx >= 0 || timed || !in.sys.out_exprs.empty() || !in.sys.zb_g.empty()) {
        // Custom КРС: тело — сырой C с объявлениями, циклами и локальными
        // массивами. SystemEvaluator интерпретирует только ВЫРАЖЕНИЯ правых
        // частей, поэтому тело компилируется в нативную функцию шага
        // (см. krs_cpu.h) и гоняется тем же циклом, что и встроенные схемы.
        // Пустое тело (Extr над базой, на которой отказал кодген) собралось бы в шаг,
        // который ничего не делает, — застывшая траектория вместо причины.
        if (in.krs_code.empty()) {
            result.error = in.krs_error.empty() ? "no KRS generated (check system)" : in.krs_error;
            return result;
        }
        KrsCpuStep step;
        std::vector<KrsCpuDiag> diags;
        if (!step.compile(in.krs_code, dim, (int)a.size(), diags)) {
            std::string msg = "CPU custom KRS:";
            for (const auto& d : diags) {
                msg += "\n";
                if (d.line > 0) msg += "line " + std::to_string(d.line) + ": ";
                msg += d.message;
            }
            result.error = msg;
            return result;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            // Явное t: траектория с t = step0 * h (0, кроме continuation-куска).
            step.step_index(in.step0);
            bool ok = computePhasePortraitCPU_custom(step.fn(),
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }
    else {
        std::unique_ptr<SystemEvaluator> evaluator;
        try {
            evaluator.reset(new SystemEvaluator(in.sys));
        }
        catch (const std::exception& e) {
            result.error = std::string("parse error: ") + e.what();
            return result;
        }
        // A map ignores the scheme field - see compute_krs_for_scheme.
        const IntScheme isch = in.sys.is_map ? IntScheme::Map
                                             : int_scheme_from_string(in.scheme);
        // Те же отказы, что у GPU-пути на кодгене (cd_check_complex_safe, jac_over), —
        // сюда кодген не доходит. Без них комплексная схема молча считала бы atan2/fmod
        // нулём, а неявная без якобиана выдавала бы "diverged (nan/inf)" вместо причины.
        // Именно отказ, а не исключение: у future этого расчёта нет catch.
        if (int_scheme_complex_state(isch) && !evaluator->complex_unsafe_function().empty()) {
            result.error = in.scheme + ": function " + evaluator->complex_unsafe_function()
                         + " is not defined in complex arithmetic - pick another scheme";
            return result;
        }
        if ((isch == IntScheme::ImplicitEuler || isch == IntScheme::ImplicitMidpoint
             || isch == IntScheme::ComplexIEuler) && !evaluator->has_jacobian()) {
            result.error = in.scheme + ": the right-hand side is not differentiable (floor/ceil/fmod), "
                           "no Jacobian can be built - pick an explicit scheme";
            return result;
        }
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            bool ok = computePhasePortraitCPU(*evaluator, isch,
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }

    // Наблюдаемые вне состояния (ДАУ, сведённая к ОДУ): z(x) в каждой точке, дописанные
    // после координат состояния. Тем же вычислителем правых частей: выражения — как rhs над
    // x и a[] (списки выровнены нулями/фиктивными переменными до одной длины).
    if (!in.obs_exprs.empty()) {
        System os;
        os.vars = in.sys.vars;
        os.params = in.sys.params;
        os.latex = in.sys.latex;
        os.rhs = in.obs_exprs;
        const size_t nz = in.obs_exprs.size(), L = std::max(os.vars.size(), nz);
        for (size_t j = os.vars.size(); j < L; ++j) os.vars.push_back("ucuda_obs_pad" + std::to_string(j));
        while (os.rhs.size() < L) os.rhs.push_back("0");
        if (system_uses_time(os)) {
            result.dae_status = "z observables use explicit time t: not computed yet";
        } else {
            try {
                SystemEvaluator ev(os);
                std::vector<double> xin(L, 0.0), zout(L, 0.0);
                for (auto& traj : raw)
                    for (auto& pt : traj) {
                        for (int i = 0; i < dim && i < (int)pt.size(); ++i) xin[(size_t)i] = pt[(size_t)i];
                        ev.eval(xin.data(), a.data(), zout.data());
                        pt.resize((size_t)dim);
                        for (size_t j = 0; j < nz; ++j) pt.push_back(zout[j]);
                    }
            } catch (const std::exception& e) {
                result.dae_status = std::string("z observables: ") + e.what();
            }
        }
    }

    // Пики — БЕЗ лимита max_amount_of_peaks: из них же строится диаграмма Continuation, и лимит
    // обрезал каждый её кусок (пики после первых N пропадали). Диаграмма Feature, сверяемая
    // с ядром бифуркации, режет до лимита сама (features_cap).
    result.features.resize(N);
    result.features_cap = get_peak_config().max_amount_of_peaks;
    // Признакам на узлах шага нужна точная производная в узле — та же f(x), что ядро
    // берёт из S.F0. Траектория могла прийти и с GPU, поэтому вычислитель свой.
    std::unique_ptr<SystemEvaluator> node_ev;
    if (result.raw_nodes) {
        try { node_ev.reset(new SystemEvaluator(in.sys)); }
        catch (const std::exception&) { node_ev.reset(); }
    }
    // Preallocated so a divergent trajectory (empty traj[]) still leaves a
    // zero-length slot at the same index as ic_sets[k].
    result.final_states.assign(N, std::vector<double>{});
    // Последняя ЗАПИСАННАЯ точка: запись идёт до шага (run_trajectory, phase_kernel), так что
    // это состояние после skip + total - 1 шагов, и следующий кусок начинается с него же.
    // Длительность куска — (точек - 1) шагов; на узлах адаптивного шага — по их времени.
    {
        size_t pts = 0;
        for (const auto& traj : raw) pts = std::max(pts, traj.size());
        result.final_step = in.step0 + (long long)skip + (pts > 0 ? (long long)pts - 1 : 0);
        result.final_t    = pts > 1 ? (double)(pts - 1) * h : 0.0;
        if (result.raw_nodes) {
            result.final_t = 0.0;
            for (const auto& tk : ad_times)
                if (tk.size() > 1) result.final_t = std::max(result.final_t, tk.back() - tk.front());
        }
    }
    // Адаптивный шаг и BDF начинают время с нуля и стартового t не принимают: у системы
    // с явным t каждый continuation-кусок начинался бы с t = 0.
    if (in.step0 > 0 && timed && (in.adaptive.enabled || scheme_is_multistep(in.scheme)) && !in.sys.is_map)
        result.note = "Continuation: explicit time t restarts at 0 in every chunk with the adaptive step / BDF; "
                      "use a fixed step for a time-dependent system.";
    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        std::vector<std::vector<double>>& traj = raw[k];
        if (!traj.empty()) result.final_states[k] = traj.back();

        // Признаки (пики + интервалы) считаем по ПРОРЕЖЕННОМУ ряду и с шагом h*decimator — ровно
        // тот сигнал и тот шаг, что получает peakFinderCUDA в 2D-бифуркации (timeStep = h *
        // preScaller в parametric_engine).
        // Раньше здесь брался полноразрешающий raw с шагом h, «чтобы decimator не прорядил сами
        // пики». Формально так и было, но диаграмма из-за этого переставала предсказывать карту:
        // критерии пика (data[i]-data[i-1] > eps_peak_delta, data[i] >= data[i+1]) и интерполяция
        // вершины по трём соседям чувствительны к шагу дискретизации, и на вчетверо более редкой
        // сетке ядро находило другой набор пиков. Совпадение с DBSCAN важнее разрешения: диаграмма
        // нужна, чтобы подбирать eps и множители. При decimator == 1 обе ветки эквивалентны.
        //
        // Слотов dim + 1: последний (индекс dim) — комбинация переменных, тот же ряд, что ядро
        // строит при writable_var == -1 (x0 + pi*x1 + euler*x2, с деградацией при dim < 3).
        // Константы берём из configCUDA.h, чтобы диаграмма и бифуркация шли по одному ряду.
        result.features[k].resize((size_t)dim + 1);
        // На узлах адаптивного шага ряд неравномерный, и поиск пиков по трём соседним
        // отсчётам к нему неприменим — там работает find_peaks_host_nodes (ниже).
        if (!result.raw_nodes) {
            const size_t stride = (dec > 1) ? (size_t)dec : (size_t)1;
            std::vector<double> series;
            series.reserve(traj.size() / stride + 1);
            for (int v = 0; v < dim; ++v) {
                series.clear();
                for (size_t t = 0; t < traj.size(); t += stride) {
                    const auto& pt = traj[t];
                    series.push_back(v < (int)pt.size() ? pt[v] : 0.0);
                }
                result.features[k][v] = find_peaks_host(series, h * (double)stride, /*apply_cap*/ false);
            }

            series.clear();
            for (size_t t = 0; t < traj.size(); t += stride) {
                const auto& pt = traj[t];
                double c = pt.size() > 0 ? pt[0] : 0.0;
                if (dim >= 2 && pt.size() > 1) c += ::pi    * pt[1];
                if (dim >= 3 && pt.size() > 2) c += ::euler * pt[2];
                series.push_back(c);
            }
            result.features[k][(size_t)dim] = find_peaks_host(series, h * (double)stride, /*apply_cap*/ false);
        }
        else if (node_ev && k < (int)ad_times.size()) {
            // Узлы — как в ucudaAdRecordNodes: каждый dec-й (считая начальный) и последний.
            const std::vector<double>& tk = ad_times[(size_t)k];
            const size_t m = std::min(traj.size(), tk.size());
            std::vector<size_t> sel;
            sel.reserve(m / (size_t)dec + 2);
            for (size_t i = 0; i < m; ++i)
                if (dec <= 1 || i % (size_t)dec == 0 || i + 1 == m) sel.push_back(i);
            const size_t ns = sel.size();
            std::vector<double> F(ns * (size_t)dim), ts(ns), xs(ns), ds(ns);
            for (size_t j = 0; j < ns; ++j) {
                ts[j] = tk[sel[j]];
                node_ev->eval(traj[sel[j]].data(), a.data(), &F[j * (size_t)dim]);
            }
            const int interp = in.adaptive.peak_interp;
            for (int v = 0; v <= dim; ++v) {
                for (size_t j = 0; j < ns; ++j) {
                    const double* X = traj[sel[j]].data();
                    const double* f = &F[j * (size_t)dim];
                    if (v < dim) { xs[j] = X[v]; ds[j] = f[v]; continue; }
                    // Комбинация — та же, что ucudaAdObs; линейна, производная — от f.
                    double c = X[0], dcomb = f[0];
                    if (dim >= 2) { c += ::pi * X[1];    dcomb += ::pi * f[1]; }
                    if (dim >= 3) { c += ::euler * X[2]; dcomb += ::euler * f[2]; }
                    xs[j] = c; ds[j] = dcomb;
                }
                result.features[k][(size_t)v] = find_peaks_host_nodes(xs, ds, ts, interp, /*apply_cap*/ false);
            }
        }

        // RQA по этой траектории — ДО прореживания и до move(traj) ниже. Считаем по полному
        // ряду с шагом h: decimator задуман как настройка отрисовки, и наследовать его здесь
        // значило бы молча менять частоту дискретизации, от которой RQA зависит напрямую.
        // Прореживание до Config::points делает сам rqa::compute и возвращает честный dt.
        for (const RqaJob& job : in.rqa_jobs) {
            if (job.ic != k) continue;
            RqaOutcome o;
            o.job = job;
            if (traj.empty()) o.res.error = "RQA: trajectory diverged";
            else if (result.raw_nodes)
                o.res.error = "RQA needs a uniform time grid: switch the adaptive output to the uniform grid";
            else rqa::compute(traj, h, job.cfg, o.res);
            result.rqa_out.push_back(std::move(o));
        }

        std::vector<std::vector<double>> dtraj;
        if (dec <= 1) dtraj = std::move(traj);
        else {
            dtraj.reserve(traj.size() / dec + 1);
            for (size_t t = 0; t < traj.size(); t += dec) dtraj.push_back(traj[t]);
        }
        result.trajectories.push_back(std::move(dtraj));
        if (result.raw_nodes) {
            const std::vector<double>& tk = ad_times[(size_t)k];
            std::vector<double> dtimes;
            if (dec <= 1) dtimes = tk;
            else {
                dtimes.reserve(tk.size() / dec + 1);
                for (size_t t = 0; t < tk.size(); t += dec) dtimes.push_back(tk[t]);
            }
            result.times.push_back(std::move(dtimes));
        }
        result.labels.push_back(ic.label);
        result.visible.push_back(ic.visible);

        std::string txt = "(";
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            std::string val = it != ic.values.end() ? it->second : "";
            if (val.empty()) val = "0";
            txt += val; if (i + 1 < dim) txt += ",";
        }
        txt += ")";
        result.ic_text.push_back(txt);
    }

    result.ok = result.trajectories.size() > 0;
    if (!calc_ok && result.ok) result.error = calc_err;

    // Snapshot of CSV-relevant inputs for the GUI right-click export.
    result.snapshot.vars      = in.vars;
    result.snapshot.params    = in.params;
    result.snapshot.a         = a;
    result.snapshot.scheme    = in.sys.is_map ? std::string("Map") : in.scheme;
    result.snapshot.is_map    = in.sys.is_map;
    result.snapshot.h         = h;
    // Настройка FMA на момент прогона — в CSV, см. Bif1DSnapshot::gpu_fmad.
    // Пишется одинаково для GPU- и CPU-ветки: это состояние настройки, а не
    // утверждение об устройстве (на CPU NVRTC не участвует вовсе).
    result.snapshot.gpu_fmad  = get_nvrtc_fmad();
    result.snapshot.t_max     = tsim;
    result.snapshot.t_skip    = tskip;
    result.snapshot.decimator = dec;
    if (result.adaptive) {
        const AdaptiveSettings& ad = in.adaptive;
        std::string d = "adaptive, controller " + ad.ctrl + ", rtol " + ad.rtol + ", atol " + ad.atol;
        if (!ad.h0.empty())   d += ", h0 " + ad.h0;
        if (!ad.hmin.empty()) d += ", h_min " + ad.hmin;
        if (!ad.hmax.empty()) d += ", h_max " + ad.hmax;
        if (!ad.max_rej.empty()) d += ", max rejects " + ad.max_rej;
        d += ad.raw_nodes ? ", output at step nodes" : ", output on the uniform grid (h = output step)";
        result.snapshot.adaptive = d;
    }
    // Snapshot IC from the user's panel text — not ic_flat, which continuation overwrites
    // with the previous chunk's final X[] so the legend would flicker every frame.
    result.snapshot.ic_flat.assign(N, std::vector<double>(dim, 0.0));
    result.snapshot.ic_labels.reserve(N);
    for (int k = 0; k < N; ++k) {
        const auto& ic = in.ic_sets[k];
        for (int i = 0; i < dim; ++i) {
            auto it = ic.values.find(in.vars[i]);
            result.snapshot.ic_flat[k][i] = parse_val(it != ic.values.end() ? it->second : "", 0.0);
        }
        result.snapshot.ic_labels.push_back(in.ic_sets[k].label);
    }

    if (in.circuit_show && result.ok)
        append_circuit_trajectory(in, a, h, tsim, dec, result);

    auto _t1 = std::chrono::high_resolution_clock::now();
    double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    if (result.error.empty())
        result.error = "recompute: " + std::to_string(_ms) + " ms";

    return result;
}
}

namespace ucuda::analysis_detail {


// Снапшот текущей session в PhaseRunInputs. Делается на главном потоке.
PhaseRunInputs snapshot_phase(PhaseAnalysisSession& s) {
    // krs_code должен быть свежим. Раньше регенерировали только при пустом кеше — это ломало
    // правки тела custom-scheme: первый Run пёк krs_code из старого cs.body, дальнейшие правки в
    // Library на него не влияли. Регенерация дешёвая (codegen или просто копия cs.body), поэтому
    // делаем её каждый Run.
    s.regenerate_krs();
    PhaseRunInputs in;
    in.vars         = s.vars;
    in.params       = s.params;
    in.step_h       = s.step_h;
    in.sim_time     = s.sim_time;
    in.skip_time    = s.skip_time;
    in.scheme       = s.scheme;
    in.symmetry_s   = s.symmetry_s;
    in.cd_fp_iters  = s.cd_fp_iters;
    in.newton  = s.newton;
    in.decimation   = s.decimation;
    in.adaptive     = s.adaptive;
    in.use_gpu      = s.use_gpu;
    in.obs_exprs    = s.obs_exprs;
    in.sys          = s.sys;
    in.krs_code     = s.krs_code;
    in.krs_error    = s.krs_error;
    in.krs_is_custom = false;
    // For a map krs_code is always codegen output; custom KRS do not apply.
    if (!s.sys.is_map) {
        for (const auto& cs : s.custom_schemes)
            if (cs.name == s.scheme) { in.krs_is_custom = true; break; }
        // Экстраполяционная обёртка — тоже "сырой текст" в смысле этой ветки:
        // тело базы завёрнуто в лямбду и стадийные циклы, интерпретатору
        // выражений такое не разобрать. Зато CPU компилирует тем же cl.exe
        // БУКВАЛЬНО тот же текст, который NVRTC собирает для GPU, — совпадение
        // путей тут точнее, чем у любой встроенной схемы.
        if (!in.krs_is_custom && parse_extrapolation_name(s.scheme, nullptr))
            in.krs_is_custom = true;
    }
    in.custom_schemes = s.custom_schemes;
    in.param_values = s.param_values;
    in.ic_sets      = s.ic_sets;
    in.circuit_show        = s.circuit_show;
    in.circuit_ideal_opamp = s.circuit_ideal_opamp;
    in.circuit_target_volt = s.circuit_target_volt;
    in.circuit_gbw_mhz     = s.circuit_gbw_mhz;
    in.circuit_vsat        = s.circuit_vsat;
    in.circuit_substeps    = s.circuit_substeps;
    // Continuation: on frames >= 1 resume from the previous chunk's final X[]
    // and skip the transient (already burned in). Size mismatch (user added or
    // removed an IC mid-run) falls back to first-frame semantics.
    if (s.continuation_active && !s.continuation_first_frame &&
        s.continuation_state.size() == s.ic_sets.size()) {
        in.ic_override    = s.continuation_state;
        in.skip_transient = false;
        in.step0          = s.continuation_step;
    }

    // RQA-задачи из открытых окон типа RecurrencePlot. Дедуп по (ic, cfg): два окна с одинаковыми
    // настройками (например «то же, но бинарный вид») обязаны стоить один расчёт, а не два.
    // Проверки на "окно закрыто" тут нет намеренно: крестик окна проекции УДАЛЯЕТ проекцию
    // (pr_to_remove в draw_projection_windows), состояния "есть, но закрыта" не существует —
    // Projection::open остался от старой схемы и никем не читается.
    for (const Projection& pr : s.projections) {
        if (s.continuation_active && !pr.rqa_in_continuation) continue;
        RqaJob job;
        if (!make_rqa_job(pr, job)) continue;
        if (job.ic < 0 || job.ic >= (int)s.ic_sets.size()) continue;
        bool dup = false;
        for (const RqaJob& have : in.rqa_jobs) if (have == job) { dup = true; break; }
        if (!dup) in.rqa_jobs.push_back(job);
    }
    return in;
}
}


void PhaseAnalysisSession::recompute() {
    PhaseRunInputs in = snapshot_phase(*this);
    result = compute_phase_portrait(in);
    fit_request = true;
    data_generation++;
    if (!continuation_mode) rebuild_peaks_from_result();
}


bool PhaseAnalysisSession::recompute_async() {
    if (in_flight) return false;
    PhaseRunInputs in = snapshot_phase(*this);
    cancel_token = std::make_shared<std::atomic<bool>>(false);
    in.cancel    = cancel_token;
    in_flight = true;
    compute_start_time = std::chrono::steady_clock::now();
    recompute_future = std::async(std::launch::async, [in = std::move(in)]() {
        return compute_phase_portrait(in);
    });
    return true;
}


void PhaseAnalysisSession::trim_continuation_peaks() {
    const size_t cap_pairs = (size_t)std::clamp(continuation_peaks_cap,
                                                kContPeaksCapMin, kContPeaksCapMax);
    for (auto& per_var : continuation_peaks)
        for (std::vector<double>& buf : per_var) {
            const size_t cur_pairs = buf.size() / 2;
            if (cur_pairs <= cap_pairs) continue;
            const size_t drop_pairs = cur_pairs - cap_pairs;
            buf.erase(buf.begin(), buf.begin() + (std::ptrdiff_t)(drop_pairs * 2));
        }
}


// Пики result.features -> continuation_peaks, время сдвинуто на base.
// Буферы по IC пересоздаются, если число IC или переменных поменялось.
void PhaseAnalysisSession::append_result_peaks(double base) {
    const int N = (int)ic_sets.size();
    if ((int)continuation_peaks.size() != N) continuation_peaks.assign((size_t)N, {});
    for (int k = 0; k < N && k < (int)result.features.size(); ++k) {
        auto& per_var_out = continuation_peaks[(size_t)k];
        const auto& per_var_in = result.features[(size_t)k];
        if (per_var_out.size() != per_var_in.size())
            per_var_out.assign(per_var_in.size(), {});
        for (size_t v = 0; v < per_var_in.size(); ++v) {
            const FeaturePoints& fp = per_var_in[v];
            std::vector<double>& buf = per_var_out[v];
            // Отсчёт — от опорного (первого) пика куска: интервалы идут от него. Раньше отсчёт
            // шёл от начала куска, и все пики куска съезжали влево на first_time, а сам
            // опорный пик терялся — на стыках continuation-кусков были провалы.
            if (!std::isfinite(fp.first_time)) continue;   // в куске нет пиков
            double cumt = base + fp.first_time;
            const size_t n = fp.peaks.size() < fp.intervals.size()
                           ? fp.peaks.size() : fp.intervals.size();
            buf.reserve(buf.size() + n * 2 + 2);
            if (std::isfinite(fp.first_peak)) {
                buf.push_back(cumt);
                buf.push_back(fp.first_peak);
            }
            for (size_t p = 0; p < n; ++p) {
                if (!std::isfinite(fp.peaks[p]) || !std::isfinite(fp.intervals[p])) continue;
                cumt += fp.intervals[p];
                buf.push_back((double)cumt);
                buf.push_back((double)fp.peaks[p]);
            }
        }
    }
    trim_continuation_peaks();
}


// Без continuation диаграмма показывает пики ОДНОГО прогона: время от начала
// записи (после transient), как у первого кадра continuation. Буфер при этом
// собирается заново, а не копится, — иначе каждый Recompute склеивал бы
// прогоны с разными параметрами в одну ось времени.
void PhaseAnalysisSession::rebuild_peaks_from_result() {
    continuation_peaks.clear();
    if (result.ok) append_result_peaks(0.0);
    ++continuation_peaks_gen;
}


void PhaseAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool PhaseAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!recompute_future.valid()) { in_flight = false; cancel_token.reset(); return false; }
    if (recompute_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    AnalysisResult r = recompute_future.get();
    in_flight = false;
    cancel_token.reset();
    if (r.cancelled) {
        // Прежние графики остаются; цикл continuation (если шёл) останавливается.
        result.error = r.error;
        continuation_active = false;
        continuation_paused = false;
        return true;
    }
    result = std::move(r);
    fit_request = true;
    data_generation++;
    // Carry the final X[] forward for the next continuation chunk. Only
    // adopt on success so a divergent frame doesn't corrupt the seed.
    if (continuation_active && result.ok &&
        result.final_states.size() == ic_sets.size()) {
        // Длительность куска — до его последней точки, с которой начнётся следующий
        // (snapshot.t_max = Time(s) была длиннее на шаг: стыки уезжали на h за кусок).
        const double chunk_t = result.final_t;
        const double base    = continuation_elapsed;
        continuation_elapsed += chunk_t;
        append_result_peaks(base);
        ++continuation_peaks_gen;

        continuation_step        = result.final_step;
        continuation_state       = std::move(result.final_states);
        continuation_first_frame = false;
    }
    // Режим включён, но цикл остановлен (Stop): накопленное не трогаем.
    else if (!continuation_mode) {
        rebuild_peaks_from_result();
    }
    return true;
}


void PhaseAnalysisSession::regenerate_krs() {
    // Раньше здесь лежала вторая копия резолвера, и добавить разбор имени
    // пришлось бы в оба места: панель KRS рисует именно krs_code, а Run идёт
    // через compute_krs_for_scheme. Разъехавшись, они показывали бы одно, а
    // считали другое — поэтому теперь резолвер ровно один.
    krs_code = compute_krs_for_scheme(custom_schemes, sys, scheme, &krs_error);
    prewarm_gpu();
}


void PhaseAnalysisSession::prewarm_gpu() {
    // Фоновый прогрев NVRTC-кэша под новую систему/метод, не дожидаясь Run. Новый прогрев не
    // запускаем, пока предыдущий не закончился: присвоение prewarm_future иначе заблокировало бы
    // ЭТОТ (UI) поток на деструкторе предыдущей async-future. Best-effort — если не успеваем
    // прогреть, настоящий Run просто скомпилирует синхронно.
    bool prewarm_busy = prewarm_future.valid() &&
        prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    if (!use_gpu || prewarm_busy) return;
    const int dim = (int)vars.size();
    if (adaptive.enabled && !sys.is_map) {
        // Тела и регулятор — как в run_adaptive_phase: ключ кэша ядра — ровно они и размерность.
        // Холодная сборка адаптивного ядра Analysis — 5-6 с; без прогрева их ждал первый Run.
        if (!adaptive_scheme_name_ok(scheme)) return;
        PhaseAdaptiveRequest rq;
        try {
            const AdaptiveCode code = adaptive_code_for_scheme(custom_schemes, sys, scheme);
            UcudaAdaptParams P;
            std::string err;
            if (!adaptive_build_params(adaptive, code, dim, 1.0, P, err, &rq.ctrl_body)) return;
            rq.rhs = code.rhs; rq.emb = code.emb; rq.dprep = code.dprep; rq.deval = code.deval;
        } catch (const std::exception&) { return; }
        rq.amountOfX = dim;
        prewarm_future = std::async(std::launch::async, [rq] { prewarmPhasePortraitsAdaptiveNVRTC(rq); });
        return;
    }
    if (krs_code.empty()) return;
    std::string krs = krs_code;
    prewarm_future = std::async(std::launch::async, [krs, dim] {
        prewarmPhasePortraitsNVRTC(krs, dim);
    });
}
