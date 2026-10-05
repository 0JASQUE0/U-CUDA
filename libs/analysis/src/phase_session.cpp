#include "session_internal.h"

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
    else if (!scheme_supports_adaptive(scheme_from_name(in.scheme))) {
        // Экстраполятор (Extr, GBS): CPU-драйвер exe считает по таблицам Бутчера и тело базы в
        // стадиях не выразит. Те же тела, что у GPU, собирает cl.exe в DLL адаптивного модуля
        // (AdaptiveCpuModule, кэш на диске), траектория — ucuda_cpu_ad_phase, копия phase_kernel_ad;
        // регулятор (и пользовательский) вкомпилирован туда же.
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
    return true;
}


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
    std::vector<double> a(nparams + 1, 0.0);
    a[0] = parse_val(in.symmetry_s, 0.5);
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

    std::vector<std::vector<std::vector<double>>> raw;
    bool calc_ok = true;
    std::string calc_err;

    // Адаптивный шаг — своя ветка расчёта; признаки, RQA и прореживание ниже общие.
    // У отображения шага нет, и переключатель для него игнорируется.
    std::vector<std::vector<double>> ad_times;
    if (in.adaptive.enabled && !in.sys.is_map) {
        if (!run_adaptive_phase(in, dim, N, ic_flat, a, h, tsim, tskip, total, raw, ad_times,
                                result, calc_ok, calc_err))
            return result;
    }
    else if (in.use_gpu) {
        if (in.krs_code.empty()) {
            result.error = "no KRS generated (check system)";
            return result;
        }
        calc_ok = computePhasePortraitsNVRTC(in.krs_code, dim,
            ic_flat.data(), N, a.data(), (int)a.size(),
            h, total, skip, raw, &calc_err);
        if (!calc_ok) {
            result.error = "GPU: " + calc_err;
            return result;
        }
    }
    else if (in.krs_is_custom) {
        // Custom КРС: тело — сырой C с объявлениями, циклами и локальными
        // массивами. SystemEvaluator интерпретирует только ВЫРАЖЕНИЯ правых
        // частей, поэтому тело компилируется в нативную функцию шага
        // (см. krs_cpu.h) и гоняется тем же циклом, что и встроенные схемы.
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
        raw.resize(N);
        for (int k = 0; k < N; ++k) {
            // A map ignores the scheme field - see compute_krs_for_scheme.
            const IntScheme isch = in.sys.is_map ? IntScheme::Map
                                                 : int_scheme_from_string(in.scheme);
            bool ok = computePhasePortraitCPU(*evaluator, isch,
                &ic_flat[(size_t)k * dim], dim, a.data(), (int)a.size(),
                h, total, skip, raw[k]);
            if (!ok) { calc_ok = false; calc_err = "trajectory '" + in.ic_sets[k].label + "' diverged (nan/inf)"; }
        }
    }

    result.features.resize(N);
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
                result.features[k][v] = find_peaks_host(series, h * (double)stride);
            }

            series.clear();
            for (size_t t = 0; t < traj.size(); t += stride) {
                const auto& pt = traj[t];
                double c = pt.size() > 0 ? pt[0] : 0.0;
                if (dim >= 2 && pt.size() > 1) c += ::pi    * pt[1];
                if (dim >= 3 && pt.size() > 2) c += ::euler * pt[2];
                series.push_back(c);
            }
            result.features[k][(size_t)dim] = find_peaks_host(series, h * (double)stride);
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
                result.features[k][(size_t)v] = find_peaks_host_nodes(xs, ds, ts, interp);
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
    in.decimation   = s.decimation;
    in.adaptive     = s.adaptive;
    in.use_gpu      = s.use_gpu;
    in.sys          = s.sys;
    in.krs_code     = s.krs_code;
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
            double cumt = base;
            const size_t n = fp.peaks.size() < fp.intervals.size()
                           ? fp.peaks.size() : fp.intervals.size();
            buf.reserve(buf.size() + n * 2);
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
        const double chunk_t = result.snapshot.t_max;
        const double base    = continuation_elapsed;
        continuation_elapsed += chunk_t;
        append_result_peaks(base);
        ++continuation_peaks_gen;

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
    krs_code = compute_krs_for_scheme(custom_schemes, sys, scheme);
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
