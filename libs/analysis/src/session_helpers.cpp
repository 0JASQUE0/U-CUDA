#include "session_internal.h"

namespace ucuda::analysis_detail {












// Слот в a[] для цели свипа по параметру. param_index конфига — 0-based индекс
// в списке параметров системы, а в a[] параметры сдвинуты на единицу, потому
// что a[0] занят коэффициентом симметрии s. Отдельное значение -1 означает
// "свипать сам s": он и есть a[0], поэтому движку хватает индекса и никакого
// отдельного флага (в отличие от sweep_over_h / sweep_over_var) не нужно.
// Пункт "s (a[0])" в UI показывается только у схем, которые a[0] читают.
int sweep_param_slot(int idx, int nparams) {
    if (idx < 0) return 0;
    return (idx < nparams) ? idx + 1 : 1;
}


// Mirrors the top-bar indicator text to the console on completion so users
// who keep stdout visible can see timings without watching the title bar.
void log_run_completed(const char* label, bool ok, double secs) {
    std::printf("[run] %s %s in %.2fs\n",
                ok ? "Done" : "Cancelled",
                (label && *label) ? label : "(unnamed)",
                secs);
    std::fflush(stdout);
}


// Host-порт peakFinder из cudaLibrary.cu.
//
// Зачем копия, а не вызов оригинала: peakFinder объявлен как __device__ __host__ в
// cudaLibrary.cuh, который парсит только nvcc, а этот файл собирает MSVC. Тот же приём уже
// применён для getValueByIdx (getValueByIdx_local в parametric_engine.cpp).
//
// Отличия от оригинала — только в способе доступа к настройкам: пороги берутся из PeakConfig (GUI
// Settings), а не из constexpr-дефолтов configCUDA.h, потому что на GPU те же значения инжектятся
// как #define перед компиляцией (peak_config_defines). Логика поиска пиков, интерполяция вершины
// и фильтр по eps_interPeak_delta повторены один в один — иначе Feature diagram показывала бы не
// те точки, что реально уходят в DBSCAN 2D-бифуркации.
//
// dt — шаг ПО ВРЕМЕНИ между соседними отсчётами `data`. Вызывающий передаёт h*decimator и сам ряд,
// прореженный тем же шагом: признаки должны считаться по тому же сигналу, что уходит в
// peakFinderCUDA. Возвращает пары (значение пика; интервал от предыдущего удержанного пика).
FeaturePoints find_peaks_host(const std::vector<double>& data, double dt) {
    FeaturePoints out;
    const PeakConfig pc = get_peak_config();
    if (!pc.do_calculate_peaks) return out;

    const size_t n = data.size();
    if (n < 5) return out;   // оригинал сканирует [2, n-2)

    std::vector<double> peaks, times;   // times — в индексах, как в оригинале

    for (size_t i = 2; i + 2 < n; ++i) {
        if (!(data[i] - data[i - 1] > pc.eps_peak_delta
              && data[i] > pc.peak_threshold
              && data[i] >= data[i + 1])) continue;

        for (size_t j = i; j + 2 < n; ++j) {
            // Строго больше следующей — плато шло вверх, пика не было.
            if (data[j] < data[j + 1]) { i = j + 1; break; }

            if (data[j] - data[j + 1] > pc.eps_peak_delta) {
                if (pc.do_interpolate_peaks) {
                    const double denom = data[j - 1] - 2.0 * data[j] + data[j + 1];
                    double delta = 0.0;
                    if (std::fabs(denom) > 1e-12)
                        delta = 0.5 * (data[j - 1] - data[j + 1]) / denom;
                    peaks.push_back(data[j] - 0.25 * (data[j - 1] - data[j + 1]) * delta);
                    times.push_back((double)(j - 1) + delta);
                } else {
                    peaks.push_back(data[j]);
                    times.push_back((double)(j - 1));
                }
                i = j + 1;   // два пика подряд невозможны
                break;
            }
        }
    }

    // Второй проход: пара (пик; интервал до него) + фильтр слишком близких.
    // Опора не сдвигается на отброшенный пик — как в оригинале.
    if (peaks.size() > 1) {
        double anchor = times[0];
        for (size_t i = 1; i < peaks.size(); ++i) {
            const double cur   = times[i];
            const double delta = (cur - anchor) * dt;
            if (delta >= pc.eps_interPeak_delta) {
                out.peaks.push_back(peaks[i]);
                out.intervals.push_back(delta);
                anchor = cur;
            }
        }
    }

    const size_t cap = (size_t)std::max(0, pc.max_amount_of_peaks);
    if (out.peaks.size() > cap) {
        out.peaks.resize(cap);
        out.intervals.resize(cap);
    }
    return out;
}


// Парсит значение (число или дробь a/b) в double; пусто -> fallback.
double parse_val(const std::string& s, double fallback) {
    return parse_num(s, fallback);   // тело — в num_parse.h, одно на проект
}





// генерация КРС из системы по выбранному методу (для NVRTC)
Scheme scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return Scheme::EulerCromer;
    if (s == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (s == "RK4")               return Scheme::RK4;
    if (s == "DOPRI78")           return Scheme::DOPRI78;
    if (s == "CD")                return Scheme::CD;
    if (s == "Complex CD")        return Scheme::ComplexCD;
    if (s == "Complex CD4")       return Scheme::ComplexCD4;
    if (s == "Implicit Euler")    return Scheme::ImplicitEuler;
    if (s == "Implicit Midpoint") return Scheme::ImplicitMidpoint;
    if (s == "SEMP")              return Scheme::SEMP;
    if (s == "SIMP")              return Scheme::SIMP;
    if (s == "D")                 return Scheme::D;
    if (s == "Complex Implicit Euler") return Scheme::ComplexIEuler;
    return Scheme::Euler;
}


// Парсит double, понимая дробь "a/b" — поведение симметрично с parse_val
// (см. строки 10–18 этого же файла, которой пользуется Phase). Иначе
// Parametric получал бы "8/3" как 8, а Phase как 2.667.
double parse_d(const std::string& s, double def) {
    return parse_num(s, def);   // историческое имя, см. num_parse.h
}

int parse_i(const std::string& s, int def) {
    return parse_num_int(s, def);   // историческое имя, см. num_parse.h
}


// Координаты обхода NxN-сетки по спирали из центра (порт MATLAB spiral_coords_from_center).
// Стартовая клетка — округление к верху центра: для N=5 это (2,2), для N=4 — (1,1), 0-based.
// Дальше right→down→left→up с увеличением шага на каждом цикле полуоборотов; точки за границей
// сетки пропускаются, поэтому набирается ровно N*N валидных индексов. Возвращает row-major
// индексы row*N + col.
std::vector<int> spiral_coords_from_center(int N) {
    std::vector<int> out;
    if (N <= 0) return out;
    const int total = N * N;
    out.reserve((size_t)total);
    int r = (N - 1) / 2;
    int c = (N - 1) / 2;
    out.push_back(r * N + c);
    static const int dr[4] = { 0, 1, 0, -1 };
    static const int dc[4] = { 1, 0, -1, 0 };
    int dir = 0;
    int step = 1;
    while ((int)out.size() < total) {
        for (int k = 0; k < 2; ++k) {
            for (int t = 0; t < step; ++t) {
                r += dr[dir];
                c += dc[dir];
                if (r >= 0 && r < N && c >= 0 && c < N) {
                    out.push_back(r * N + c);
                    if ((int)out.size() >= total) return out;
                }
            }
            dir = (dir + 1) & 3;
        }
        ++step;
    }
    return out;
}


LS1DRequest build_ls1d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) {
    LS1DRequest req;
    req.continuation         = c.continuation;
    req.continuation_reverse = c.continuation_reverse;
    req.use_cpu              = !c.use_gpu;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme, c.adaptive);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    req.base_values[0] = parse_d(c.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }
    req.param_index = sweep_param_slot(c.param_index, nparams);
    req.sweep_over_var = c.sweep_over_var;
    req.sweep_over_h   = c.sweep_over_h;
    req.log_scale      = c.log_scale;
    req.var_sweep_index = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                          ? c.var_sweep_index : 0;

    req.param_lo       = parse_d(c.param_lo_text, 0.0);
    req.param_hi       = parse_d(c.param_hi_text, 1.0);
    if (req.param_hi < req.param_lo) std::swap(req.param_lo, req.param_hi);
    req.n_pts          = parse_i(c.n_pts_text, 500);
    req.h              = parse_d(c.h_text, 0.01);
    req.t_max          = parse_d(c.t_max_text, 100.0);
    req.transient_time = parse_d(c.transient_text, 100.0);
    req.max_value      = parse_d(c.max_value_text, 1.0e6);
    req.NT             = parse_d(c.nt_text,  1.0);
    req.eps            = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}


void apply_ls1d_result(LSCurveConfig& c, LS1DResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}


// 2D-Request: те же общие поля + поля второй оси (param_index_2 / var_sweep_index_2).
LS2DRequest build_ls2d_request(const LyapunovSpectrumAnalysisSession& s,
                                      const LSCurveConfig& c) {
    LS2DRequest req;
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme, c.adaptive);
    req.amountOfX = (int)s.vars.size();

    req.initial_conditions.resize(req.amountOfX);
    for (int i = 0; i < req.amountOfX; ++i) {
        auto it = c.initial_conditions.find(s.vars[i]);
        req.initial_conditions[i] = (it != c.initial_conditions.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    int nparams = (int)s.params.size();
    req.base_values.assign((size_t)nparams + 1, 0.0);
    req.base_values[0] = parse_d(c.symmetry_s, 0.5); // CD: коэф. симметрии (a[0])
    for (int i = 0; i < nparams; ++i) {
        auto it = c.param_values.find(s.params[i]);
        req.base_values[i + 1] = (it != c.param_values.end()) ? parse_d(it->second, 0.0) : 0.0;
    }

    req.sweep_over_var     = c.sweep_over_var;
    req.sweep_over_var_2   = c.sweep_over_var_2;
    req.sweep_over_h       = c.sweep_over_h;
    req.sweep_over_h_2     = c.sweep_over_h_2;
    req.log_scale          = c.log_scale;
    req.log_scale_2        = c.log_scale_2;
    req.param_index        = sweep_param_slot(c.param_index, nparams);
    req.param_index_2      = sweep_param_slot(c.param_index_2, nparams);
    req.var_sweep_index    = (c.var_sweep_index >= 0 && c.var_sweep_index < req.amountOfX)
                             ? c.var_sweep_index : 0;
    req.var_sweep_index_2  = (c.var_sweep_index_2 >= 0 && c.var_sweep_index_2 < req.amountOfX)
                             ? c.var_sweep_index_2 : 0;

    req.param_lo           = parse_d(c.param_lo_text,   0.0);
    req.param_hi           = parse_d(c.param_hi_text,   1.0);
    req.param_lo_2         = parse_d(c.param_lo_2_text, 0.0);
    req.param_hi_2         = parse_d(c.param_hi_2_text, 1.0);
    if (req.param_hi   < req.param_lo)   std::swap(req.param_lo,   req.param_hi);
    if (req.param_hi_2 < req.param_lo_2) std::swap(req.param_lo_2, req.param_hi_2);
    req.n_pts              = parse_i(c.n_pts_text, 200);
    req.h                  = parse_d(c.h_text, 0.01);
    req.t_max              = parse_d(c.t_max_text, 100.0);
    req.transient_time     = parse_d(c.transient_text, 100.0);
    req.max_value          = parse_d(c.max_value_text, 1.0e6);
    req.NT                 = parse_d(c.nt_text,  1.0);
    req.eps                = parse_d(c.eps_text, 1.0e-4);
    req.csv_output_path    = c.csv_save_enabled ? c.csv_output_path : std::string{};
    return req;
}


void apply_ls2d_result(LSCurveConfig& c, LS2DResult&& r) {
    c.result_2d = std::move(r);
    c.last_run_2d_ok = c.result_2d.ok;
    if (!c.result_2d.ok) c.last_error = c.result_2d.error;
    // Clamp выбранной экспоненты под новый n_exponents (если N изменился).
    // -1 ("sum L_i") не трогаем — сумма валидна для любого N.
    if (c.result_2d.n_exponents > 0 &&
        c.display_exponent_idx != -1 &&
        c.display_exponent_idx >= c.result_2d.n_exponents) {
        c.display_exponent_idx = 0;
    }
    c.data_generation_2d++;
    c.fit_request_2d = true;
}
}
using namespace ucuda::analysis_detail;


std::vector<std::string> enabled_builtins_from_record(const SystemRecord& r) {
    std::vector<std::string> out;
    // Порядок — как в комбо схем: по порядку точности.
    if (r.scheme_euler)    out.emplace_back("Euler");
    if (r.scheme_cromer)   out.emplace_back("Euler-Cromer");
    if (r.scheme_dmethod)  out.emplace_back("D");
    if (r.scheme_ieuler)   out.emplace_back("Implicit Euler");
    if (r.scheme_midpoint) out.emplace_back("Explicit Midpoint");
    if (r.scheme_imidpoint) out.emplace_back("Implicit Midpoint");
    if (r.scheme_cd)       out.emplace_back("CD");
    if (r.scheme_ccd)      out.emplace_back("Complex CD");
    if (r.scheme_cieuler)  out.emplace_back("Complex Implicit Euler");
    if (r.scheme_semp)     out.emplace_back("SEMP");
    if (r.scheme_simp)     out.emplace_back("SIMP");
    if (r.scheme_rk4)      out.emplace_back("RK4");
    if (r.scheme_ccd4)     out.emplace_back("Complex CD4");
    if (r.scheme_dopri78)  out.emplace_back("DOPRI78");
    return out;
}

 // namespace

bool make_rqa_job(const Projection& pr, RqaJob& out)
{
    if (pr.type != ProjType::RecurrencePlot) return false;
    if (pr.rqa_source == rqa::Source::None)  return false;

    out = RqaJob();
    out.ic = pr.rqa_ic;
    rqa::Config& c = out.cfg;
    c.source   = pr.rqa_source;
    c.var      = pr.rqa_var;
    c.m        = (int)parse_val(pr.rqa_m_text,   3.0);
    c.tau      = (int)parse_val(pr.rqa_tau_text, 1.0);
    c.norm     = pr.rqa_norm;
    c.eps_mode = pr.rqa_eps_mode;
    c.eps      = (numb)parse_val(pr.rqa_eps_text,      0.1);
    c.eps_frac = (numb)parse_val(pr.rqa_eps_frac_text, 0.1);
    // Поле RR в UI — проценты, а Config::target_rr — доля.
    c.target_rr = (numb)(parse_val(pr.rqa_rr_text, 5.0) / 100.0);
    c.theiler  = (int)parse_val(pr.rqa_theiler_text, 1.0);
    c.l_min    = (int)parse_val(pr.rqa_lmin_text,    2.0);
    c.v_min    = (int)parse_val(pr.rqa_vmin_text,    2.0);
    c.points   = (int)parse_val(pr.rqa_points_text,  2048.0);
    c.network_measures = pr.rqa_network;
    return true;
}


// BifurcationAnalysisSession

// Резолвит КРС по имени scheme: сперва среди custom_schemes (имя имеет
// приоритет над built-in, что блокируется в System tab), иначе генерирует
// через codegen_scheme. Чистая функция — зовётся при сборке Request в
// момент Run (не персистится). Переиспользуется bifurcation и LLE.
std::string compute_krs_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                          const System& sys,
                                          const std::string& scheme, const adaptive::Options& options) {
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme) {
            try { return codegen_adaptive(sys,scheme,cs.body,true,options); }
            catch (const std::exception& e) { return std::string("#error ")+e.what()+"\n"; }
        }
    }
    if (sys.rhs.empty()) return {};
    try {
        return codegen_adaptive(sys, scheme,
            options.method == 1 ? std::string{} : codegen_scheme(sys, scheme_from_string(scheme)),
            false, options);
    }
    catch (const std::exception& e) {
        return std::string("#error ")+e.what()+"\n";
    }
}

using namespace ucuda::analysis_detail;
const rqa::Result* find_rqa(const AnalysisResult& res, const RqaJob& job)
{
    for (const RqaOutcome& o : res.rqa_out)
        if (o.job == job) return &o.res;
    return nullptr;
}
