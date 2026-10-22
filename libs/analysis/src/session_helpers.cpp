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


// Host-порт PeakStreamNU из kernels/adaptive_part.cu: пики на узлах адаптивного шага.
// Копия по той же причине, что и find_peaks_host выше, и повторяет оригинал один в один:
// пик — смена знака производной сигнала с + на - между соседними узлами; принимается,
// если вершина поднялась над минимумом сигнала после предыдущего пика не меньше чем на
// eps_peak_delta и выше peak_threshold. Вершина — по interp: 0 — больший из двух узлов,
// 1 — парабола по трём последним узлам с неравными шагами, 2 — кубический Эрмит на шаге
// со сменой знака по значениям и точным производным. Время пика абсолютное, интервалы и
// их фильтр eps_interPeak_delta — как у find_peaks_host.
//
// x, d, t — значение сигнала, его производная и время узлов; прореживание (каждый dec-й
// узел и последний) уже сделано вызывающим — тем же правилом, что в ucudaAdRecordNodes.
FeaturePoints find_peaks_host_nodes(const std::vector<double>& x, const std::vector<double>& d,
                                           const std::vector<double>& t, int interp) {
    FeaturePoints out;
    const PeakConfig pc = get_peak_config();
    if (!pc.do_calculate_peaks) return out;
    const size_t n = std::min(x.size(), std::min(d.size(), t.size()));
    if (n < 2) return out;
    const size_t cap = (size_t)std::max(0, pc.max_amount_of_peaks);

    bool   have_anchor = false;
    double anchor = 0.0;
    double run_min = x[0];
    for (size_t i = 1; i < n && out.peaks.size() < cap; ++i) {
        const double t1 = t[i - 1], x1 = x[i - 1], d1 = d[i - 1];
        const double tc = t[i], xc = x[i], dc = d[i];
        if (d1 > 0 && dc <= 0) {
            double pv = x1, pt = t1;
            if (xc > x1) { pv = xc; pt = tc; }
            if (interp == 1 && i >= 2) {
                const double h1 = t1 - t[i - 2], h2 = tc - t1;
                if (h1 > 0 && h2 > 0) {
                    const double e1 = (x1 - x[i - 2]) / h1, e2 = (xc - x1) / h2;
                    const double c = (e2 - e1) / (h1 + h2);
                    const double b = (e1 * h2 + e2 * h1) / (h1 + h2);
                    if (c < 0) {
                        double tau = -b / (2 * c);
                        if (tau < -h1) tau = -h1;
                        if (tau > h2)  tau = h2;
                        pv = x1 + tau * (b + c * tau);
                        pt = t1 + tau;
                    }
                }
            }
            else if (interp == 2) {
                const double h = tc - t1;
                if (h > 0) {
                    const double D  = xc - x1;
                    const double c1 = h * d1;
                    const double c2 = 3 * D - h * (2 * d1 + dc);
                    const double c3 = h * (d1 + dc) - 2 * D;
                    const double A = 3 * c3, B = 2 * c2, C = c1;
                    double th = -1;
                    if (std::fabs(A) <= 1e-14 * (std::fabs(B) + std::fabs(C))) {
                        if (B != 0) th = -C / B;
                    } else {
                        const double disc = B * B - 4 * A * C;
                        if (disc >= 0) {
                            const double sq = std::sqrt(disc);
                            const double q  = -0.5 * (B + (B >= 0 ? sq : -sq));
                            const double r1 = q / A;
                            const double r2 = (q != 0) ? C / q : r1;
                            if (r1 >= 0 && r1 <= 1 && B + 2 * A * r1 < 0)      th = r1;
                            else if (r2 >= 0 && r2 <= 1 && B + 2 * A * r2 < 0) th = r2;
                        }
                    }
                    if (th >= 0 && th <= 1) {
                        pv = x1 + th * (c1 + th * (c2 + th * c3));
                        pt = t1 + th * h;
                    }
                }
            }
            if (pv - run_min >= pc.eps_peak_delta && pv > pc.peak_threshold) {
                if (!have_anchor) { anchor = pt; have_anchor = true; }
                else if (pt - anchor >= pc.eps_interPeak_delta) {
                    out.peaks.push_back(pv);
                    out.intervals.push_back(pt - anchor);
                    anchor = pt;
                }
                run_min = xc;
            }
        }
        if (xc < run_min) run_min = xc;
    }
    return out;
}


// Парсит значение (число или дробь a/b) в double; пусто -> fallback.
double parse_val(const std::string& s, double fallback) {
    return parse_num(s, fallback);   // тело — в num_parse.h, одно на проект
}





// Чистая функция: входы → AnalysisResult. Не трогает session, поэтому её
// безопасно звать с любого потока.
// Схемная траектория поверх ОДУ. Кладётся ещё одной строкой в trajectories: оттуда
// она сама доедет до всех проекций, легенды и экспорта, без правок в отрисовке.
//
// Масштаб ИЗМЕРЯЕТСЯ по уже посчитанной траектории ОДУ, а не подбирается: схема
// живёт в вольтах, и у Лоренца из библиотеки без этого вышло бы +-45 В. Наружу
// траектория домножается на масштаб обратно — иначе две кривые в одном окне не
// с чем было бы сравнивать.
void append_circuit_trajectory(const PhaseRunInputs& in,
                                      const std::vector<double>& a,
                                      double h, double tsim, int dec,
                                      AnalysisResult& result) {
    auto note = [&](const std::string& m) { result.circuit_status = m; };

    if (result.trajectories.empty() || result.trajectories[0].size() < 2) {
        note("circuit: no ODE trajectory to take the scale from");
        return;
    }
    const int dim = (int)in.vars.size();

    std::string err;
    std::vector<PolyRhs> poly;
    if (!extract_quadratic(in.sys, a, PolyExtractConfig(), poly, &err)) { note(err); return; }

    ScalePlan sp;
    if (!plan_amplitude_scaling(result.trajectories[0], parse_val(in.circuit_target_volt, 3.0),
                                sp, &err)) { note(err); return; }

    std::vector<PolyRhs> scaled;
    apply_amplitude_scaling(poly, sp, scaled);

    CircuitGraph g;
    if (!synthesize_circuit(in.sys, scaled, SynthesisConfig(), g, &err)) { note(err); return; }

    CircuitSolverConfig cfg;
    cfg.opamp.ideal  = in.circuit_ideal_opamp;
    cfg.opamp.gbw_hz = parse_val(in.circuit_gbw_mhz, 3.0) * 1.0e6;
    cfg.opamp.vsat   = parse_val(in.circuit_vsat, 13.0);

    // Схема стартует из ПЕРВОЙ ЗАПИСАННОЙ точки ОДУ, а не из НУ пользователя, и
    // своего транзиента не проходит.
    //
    // Иначе обе траектории шли бы через transient (по умолчанию 100 единиц времени)
    // каждая своим методом, и при положительном ляпуновском показателе к концу
    // транзиента оказывались бы в НИКАК НЕ СВЯЗАННЫХ точках аттрактора. Поточечное
    // сравнение там невозможно в принципе, и наложение теряло бы весь смысл.
    std::vector<double> x0((size_t)dim, 0.0);
    {
        const std::vector<double>& p0 = result.trajectories[0][0];
        // Границу берём по КАЖДОМУ контейнеру отдельно: sp.s и p0 сейчас одной
        // длины по построению, но полагаться на это — значит оставить мину.
        for (int i = 0; i < dim && i < (int)p0.size() && i < (int)sp.s.size(); ++i) {
            const double si = sp.s[(size_t)i] != 0.0 ? sp.s[(size_t)i] : 1.0;
            x0[(size_t)i] = p0[(size_t)i] / si;
        }
    }

    const double sample = h * (dec > 0 ? dec : 1);
    int sub = (int)parse_val(in.circuit_substeps, 10.0);
    if (sub < 1) sub = 1;
    const double hc = h / sub;
    CircuitRunResult run = circuit_simulate(g, cfg, x0, hc, tsim, sample);
    if (!run.ok) { note(run.error); return; }

    // Сколько координат реально можно взять. dim — из in.vars, то есть из списка
    // переменных СЕССИИ, а run.x и sp.s сидят на in.sys.vars и на размерности точки
    // траектории. Эти списки выводятся РАЗНЫМИ путями (см. CLAUDE.md про
    // refresh_symbols против build_system) и способны разойтись; тогда обращение по
    // dim читало бы за границей вектора векторов, доставая мусорный std::vector —
    // то есть портило бы кучу, а падало бы позже и в чужом месте.
    const int ndim = (int)std::min(std::min((size_t)dim, run.x.size()), sp.s.size());
    if (ndim < dim)
        note("circuit: session lists " + std::to_string(dim) + " variables but the system has "
             + std::to_string(ndim) + "; the extra ones are left at zero");

    std::vector<std::vector<double>> traj;
    traj.reserve(run.t.size());
    for (size_t k = 0; k < run.t.size(); ++k) {
        // Длина точки остаётся dim: проекции адресуют координаты по индексам из
        // списка сессии, и короткая точка увела бы за границу уже отрисовку.
        std::vector<double> p((size_t)dim, 0.0);
        for (int i = 0; i < ndim; ++i) {
            if (k >= run.x[(size_t)i].size()) break;
            p[(size_t)i] = run.x[(size_t)i][k] * sp.s[(size_t)i];
        }
        traj.push_back(std::move(p));
    }

    result.circuit_valid     = true;
    result.circuit_graph     = g;
    result.circuit_scale     = sp.s;
    result.circuit_x0        = x0;
    result.circuit_opamp     = cfg.opamp;
    result.circuit_h_ode     = hc;
    result.circuit_t_end_ode = tsim;

    result.trajectories.push_back(std::move(traj));
    result.labels.push_back(in.circuit_ideal_opamp ? "circuit (ideal)" : "circuit (one-pole)");
    result.visible.push_back(true);
    result.ic_text.push_back("circuit");

    char buf[256];
    std::snprintf(buf, sizeof(buf), "circuit: %d nodes, %d op-amps, %d multipliers; scale",
                  (int)g.nodes.size(), (int)g.opamps.size(), (int)g.multipliers.size());
    std::string msg = buf;
    for (int i = 0; i < dim && i < (int)sp.s.size(); ++i) {
        std::snprintf(buf, sizeof(buf), " %s/%.3g", in.vars[(size_t)i].c_str(), sp.s[(size_t)i]);
        msg += buf;
    }
    std::snprintf(buf, sizeof(buf), ", step %.3g (%dx), Newton %.2f iter/step",
                  hc, sub,
                  (double)run.stats.newton_iters /
                  (double)(run.stats.steps ? run.stats.steps : 1));
    msg += buf;

    // Докуда кривые ещё совпадают. На хаосе это главная величина: дальше расхождение
    // растёт по построению, и смотреть надо на форму аттрактора, а не на точки.
    {
        const std::vector<std::vector<double>>& ode = result.trajectories[0];
        const std::vector<std::vector<double>>& cir = result.trajectories.back();
        double span = 0.0;
        for (int i = 0; i < ndim; ++i) span = std::max(span, sp.s[(size_t)i]);
        const double tol = 0.02 * span * 3.0;   // 2% от характерной амплитуды
        size_t k = 0;
        for (; k < ode.size() && k < cir.size(); ++k) {
            double d = 0.0;
            for (int i = 0; i < dim && i < (int)ode[k].size(); ++i)
                d = std::max(d, std::fabs(ode[k][(size_t)i] - cir[k][(size_t)i]));
            if (d > tol) break;
        }
        std::snprintf(buf, sizeof(buf), "; tracks the ODE for %.3g time units", (double)k * sample);
        msg += buf;
    }
    note(msg);
}


// генерация КРС из системы по выбранному методу (для NVRTC)
Scheme scheme_from_string(const std::string& s) {
    if (s == "Euler-Cromer")      return Scheme::EulerCromer;
    if (s == "Explicit Midpoint") return Scheme::ExplicitMidpoint;
    if (s == "RK4")               return Scheme::RK4;
    if (s == "DOPRI78")           return Scheme::DOPRI78;
    if (s == "DOPRI78 (legacy)")  return Scheme::DOPRI78Legacy;
    if (s == "RK45")              return Scheme::RK45;
    if (s == "DOP853")            return Scheme::DOP853;
    if (s == "CD")                return Scheme::CD;
    if (s == "Complex CD")        return Scheme::ComplexCD;
    if (s == "Complex CD4")       return Scheme::ComplexCD4;
    if (s == "CCD4 (o4s3)")       return Scheme::ComplexCD4S3;
    if (s == "CCD4 (o4s4)")       return Scheme::ComplexCD4S4;
    if (s == "CCD4 (s-split)_{0-1}") return Scheme::ComplexCD4SS01;
    if (s == "CCD4 (s-split)_{1-0}") return Scheme::ComplexCD4SS10;
    if (s == "CD_{1-0}") return Scheme::CD10;
    if (s == "Complex CD_{1-0}") return Scheme::ComplexCD10;
    if (s == "Complex CD4_{1-0}") return Scheme::ComplexCD4_10;
    if (s == "CCD4 (o4s3)_{1-0}") return Scheme::ComplexCD4S3_10;
    if (s == "CCD4 (o4s4)_{1-0}") return Scheme::ComplexCD4S4_10;
    if (s == "Implicit Euler")    return Scheme::ImplicitEuler;
    if (s == "Implicit Midpoint") return Scheme::ImplicitMidpoint;
    if (s == "SEMP")              return Scheme::SEMP;
    if (s == "SIMP")              return Scheme::SIMP;
    if (s == "D")                 return Scheme::D;
    if (s == "Complex Implicit Euler") return Scheme::ComplexIEuler;
    if (s == "GBS (n=2)") return Scheme::GBS;
    if (s == "GBS 2-4") return Scheme::GBS24;
    if (s == "GBS 2-4-6") return Scheme::GBS246;
    if (s == "GBS 2-4-6-8") return Scheme::GBS2468;
    if (s == "GBS 2-4-6-8-10") return Scheme::GBS246810;
    if (s == "GBS 2-4-6-8-10-12") return Scheme::GBS24681012;
    if (s == "Map")               return Scheme::Map;
    return Scheme::Euler;
}


// BifurcationAnalysisSession

// Резолвит КРС по имени scheme: сперва среди custom_schemes (имя имеет
// приоритет над built-in, что блокируется в System tab), затем разбирает имя
// как экстраполяционную обёртку "Extr(<база>|n...)", иначе генерирует через
// codegen_scheme. Чистая функция — зовётся при сборке Request в момент Run
// (не персистится). Переиспользуется bifurcation и LLE.
//
// Тело, порядок и симметричность базы обёртки (Extr, Comp, адаптивный Extr): у кастомной
// КРС их объявил автор, у встроенной берём из паспорта. Неизвестная база — отказ, а не
// тихий откат на Эйлер. Одна функция на все обёртки — иначе они разъехались бы по тому,
// что считают допустимой базой.
// Кодген отказывает исключением с читаемой причиной (cd_check_complex_safe,
// jac_check_differentiable, ...). Наверх уходит пустая КРС, а причина — в *err:
// без неё пользователь видел только "КРС пуста" и гадал, что не так с системой.
template <class F>
static std::string krs_try(std::string* err, F&& f) {
    try { return f(); }
    catch (const std::exception& e) { if (err) *err = e.what(); }
    catch (...) { if (err) *err = "code generation failed"; }
    return {};
}

bool resolve_wrapper_base(const std::vector<CustomScheme>& custom_schemes, const System& sys,
                                 const std::string& base_name, std::string (*gen)(const System&, Scheme),
                                 std::string& body, int& p, bool& sym, std::string* err) {
    p = 1; sym = false;
    for (const auto& cs : custom_schemes) {
        if (cs.name == base_name) {
            body = cs.body; p = cs.order; sym = cs.symmetric;
            return !body.empty();
        }
    }
    if (!builtin_scheme_traits(base_name, &p, &sym)) return false;
    body = krs_try(err, [&] { return gen(sys, scheme_from_string(base_name)); });
    return !body.empty();
}


// gen отличает GPU-форму от CPU-эквивалента: обе проходят ОДИН И ТОТ ЖЕ разбор
// имени, поэтому панель KRS и движок не могут понять имя по-разному.
std::string krs_for_scheme_impl(const std::vector<CustomScheme>& custom_schemes,
                                       const System& sys,
                                       const std::string& scheme,
                                       std::string (*gen)(const System&, Scheme),
                                       std::string* err) {
    if (err) err->clear();
    auto refuse = [err](std::string why) -> std::string {
        if (err) *err = std::move(why);
        return {};
    };
    // Nothing to choose for a map: the step is always the same whatever the
    // scheme field says (a stale session JSON may hold a previous ODE's scheme).
    if (sys.is_map) {
        if (sys.rhs.empty()) return {};
        return krs_try(err, [&] { return gen(sys, Scheme::Map); });
    }
    for (const auto& cs : custom_schemes) {
        if (cs.name == scheme)
            return cs.body.empty() ? refuse("custom KRS '" + scheme + "' has an empty body") : cs.body;
    }
    if (sys.rhs.empty()) return {};

    // Порядок и симметричность базы: у кастомной КРС их объявил автор, у
    // встроенной берём из паспорта. Неизвестная база — отказ, а не тихий
    // откат на Эйлер: пустая КРС наверху превращается в честную ошибку.
    // Общее для ОБЕИХ обёрток — иначе Extr и Comp разъехались бы по тому,
    // что считают допустимой базой. Отказ кодгена на базе (Extr(Complex CD|...)
    // при atan2) несёт свою причину; иначе база просто не найдена.
    auto resolve_base = [&](const char* kind, const std::string& base_name, std::string& body,
                            int& p, bool& sym) -> bool {
        if (resolve_wrapper_base(custom_schemes, sys, base_name, gen, body, p, sym, err)) return true;
        if (err && err->empty())
            *err = std::string(kind) + " base '" + base_name + "' is unknown or has no step body";
        return false;
    };

    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme, &spec) && spec.re_at_output) {
        // ExtrZ: нужно комплексное ядро базы, а оно есть только у встроенных
        // комплексных CD-схем. Кастомная КРС с тем же именем перекрыла бы
        // встроенную (см. выше) — её ядро не выделить, поэтому отказ.
        for (const auto& cs : custom_schemes)
            if (cs.name == spec.base)
                return refuse("ExtrZ needs a built-in complex base, '" + spec.base + "' is a custom KRS");
        int p = 1; bool sym = false;
        if (!scheme_has_complex_core(spec.base) || !builtin_scheme_traits(spec.base, &p, &sym))
            return refuse("ExtrZ base '" + spec.base + "' has no complex core");
        return krs_try(err, [&] {
            const std::string core = codegen_scheme_complex_core(sys, scheme_from_string(spec.base));
            return wrap_extrapolation_complex(core, (int)sys.vars.size(), spec.n, p, spec.base);
        });
    }
    if (parse_extrapolation_name(scheme, &spec)) {
        int  p = 1; bool sym = false; std::string base_body;
        if (!resolve_base("extrapolation", spec.base, base_body, p, sym)) return {};
        return krs_try(err, [&] {
            return wrap_extrapolation(base_body, (int)sys.vars.size(),
                                      spec.n, p, sym, spec.base);
        });
    }

    CompositionSpec cspec;
    if (parse_composition_name(scheme, &cspec)) {
        int  p = 1; bool sym = false; std::string base_body;
        if (!resolve_base("composition", cspec.base, base_body, p, sym)) return {};
        return krs_try(err, [&] {
            return wrap_composition(base_body, sys, cspec.gammas, p, sym, cspec.base);
        });
    }

    return krs_try(err, [&] { return gen(sys, scheme_from_string(scheme)); });
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


// Адаптивный шаг вкладки -> AdaptiveRequest движка. Ошибка настройки (схема без
// оценки ошибки, неразборный rtol...) не бросается, а уходит в setup_error: движок
// вернёт её вместо расчёта, как любую другую ошибку запуска. span — длина всего
// интервала интегрирования (транзиент + запись), нужна выбору h0. ax0 / ax1 —
// коды адаптивных осей (AdaptiveAxisKind), 0 — обычная цель свипа.
void fill_adaptive_request(const std::vector<CustomScheme>& custom_schemes,
                                  const System& sys, const std::string& scheme,
                                  const AdaptiveSettings& st, int n, double span,
                                  int ax0, int ax1, AdaptiveRequest& out) {
    out = AdaptiveRequest();
    if (!st.enabled || sys.is_map) return;
    out.enabled      = true;
    out.raw_nodes    = st.raw_nodes;
    out.peak_interp  = st.peak_interp;
    out.axis_kind[0] = ax0;
    out.axis_kind[1] = ax1;
    if (!adaptive_scheme_name_ok(scheme)) {
        out.setup_error = adaptive_scheme_hint();
        return;
    }
    AdaptiveCode code;
    try { code = adaptive_code_for_scheme(custom_schemes, sys, scheme); }
    catch (const std::exception& e) { out.setup_error = std::string("adaptive codegen: ") + e.what(); return; }
    std::string err;
    if (!adaptive_build_params(st, code, n, span, out.params, err, &out.ctrl_body)) { out.setup_error = err; return; }
    out.rhs = code.rhs; out.emb = code.emb; out.dprep = code.dprep; out.deval = code.deval;
    // Ось tol держит отношение atol/rtol текущих настроек.
    out.tol_ratio = out.params.rtol > 0 ? out.params.atol[0] / out.params.rtol : 1.0;
    out.lyap_renorm = st.lyap_renorm;
    out.desc = "adaptive " + scheme + ", controller " + st.ctrl + ", rtol " + st.rtol + ", atol " + st.atol
             + (st.hmin.empty() ? std::string() : ", h_min " + st.hmin)
             + (st.hmax.empty() ? std::string() : ", h_max " + st.hmax)
             + (st.max_rej.empty() ? std::string() : ", max rejects " + st.max_rej);
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
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
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
    req.vector_transient = parse_d(c.vtr_text, 0.0);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, c.scheme, c.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, c.ad_axis, 0, req.adaptive);
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
    req.krs_body  = compute_krs_for_scheme(s.custom_schemes, s.sys, c.scheme);
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
    req.vector_transient   = parse_d(c.vtr_text, 0.0);
    req.csv_output_path    = c.csv_save_enabled ? c.csv_output_path : std::string{};
    fill_adaptive_request(s.custom_schemes, s.sys, c.scheme, c.adaptive, req.amountOfX,
                          req.transient_time + req.t_max, c.ad_axis, c.ad_axis_2, req.adaptive);
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


std::string default_h_from_record(const SystemRecord& r) {
    if (r.is_map) return "1";
    return r.step_h.empty() ? std::string("0.01") : r.step_h;
}


std::string default_scheme_from_record(const SystemRecord& r, std::string current) {
    return r.is_map ? std::string("Map") : std::move(current);
}


bool default_all_iterates_from_record(const SystemRecord& r) {
    return r.is_map;
}


std::vector<std::string> enabled_builtins_from_record(const SystemRecord& r) {
    std::vector<std::string> out;
    // A map has no scheme: its only step is the right-hand side itself.
    if (r.is_map) { out.emplace_back("Map"); return out; }
    // Порядок — как в комбо схем: по порядку точности.
    if (r.scheme_euler)    out.emplace_back("Euler");
    if (r.scheme_cromer)   out.emplace_back("Euler-Cromer");
    if (r.scheme_dmethod)  out.emplace_back("D");
    if (r.scheme_ieuler)   out.emplace_back("Implicit Euler");
    if (r.scheme_midpoint) out.emplace_back("Explicit Midpoint");
    if (r.scheme_imidpoint) out.emplace_back("Implicit Midpoint");
    if (r.scheme_cd)       out.emplace_back("CD");
    if (r.scheme_cd10) out.emplace_back("CD_{1-0}");
    if (r.scheme_ccd)      out.emplace_back("Complex CD");
    if (r.scheme_ccd10) out.emplace_back("Complex CD_{1-0}");
    if (r.scheme_cieuler)  out.emplace_back("Complex Implicit Euler");
    if (r.scheme_semp)     out.emplace_back("SEMP");
    if (r.scheme_simp)     out.emplace_back("SIMP");
    if (r.scheme_rk4)      out.emplace_back("RK4");
    if (r.scheme_ccd4)     out.emplace_back("Complex CD4");
    if (r.scheme_ccd4_10) out.emplace_back("Complex CD4_{1-0}");
    if (r.scheme_ccd4s3)   out.emplace_back("CCD4 (o4s3)");
    if (r.scheme_ccd4s3_10) out.emplace_back("CCD4 (o4s3)_{1-0}");
    if (r.scheme_ccd4s4)   out.emplace_back("CCD4 (o4s4)");
    if (r.scheme_ccd4s4_10) out.emplace_back("CCD4 (o4s4)_{1-0}");
    if (r.scheme_ccd4ss01) out.emplace_back("CCD4 (s-split)_{0-1}");
    if (r.scheme_ccd4ss10) out.emplace_back("CCD4 (s-split)_{1-0}");
    if (r.scheme_rk45)     out.emplace_back("RK45");
    if (r.scheme_dopri78)  out.emplace_back("DOPRI78");
    if (r.scheme_dopri78_legacy) out.emplace_back("DOPRI78 (legacy)");
    if (r.scheme_dop853)   out.emplace_back("DOP853");
    if (r.scheme_gbs) out.emplace_back("GBS (n=2)");
    if (r.scheme_gbs24) out.emplace_back("GBS 2-4");
    if (r.scheme_gbs246) out.emplace_back("GBS 2-4-6");
    if (r.scheme_gbs2468) out.emplace_back("GBS 2-4-6-8");
    if (r.scheme_gbs246810) out.emplace_back("GBS 2-4-6-8-10");
    if (r.scheme_gbs24681012) out.emplace_back("GBS 2-4-6-8-10-12");
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


std::string compute_krs_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                          const System& sys,
                                          const std::string& scheme,
                                          std::string* err) {
    return krs_for_scheme_impl(custom_schemes, sys, scheme, &codegen_scheme, err);
}


std::string krs_empty_error(const std::vector<CustomScheme>& custom_schemes,
                            const System& sys, const std::string& scheme) {
    std::string err;
    compute_krs_for_scheme(custom_schemes, sys, scheme, &err);
    return err.empty() ? "krs_code is empty (no valid system or scheme)" : err;
}

// CPU-форма того же шага — только для отладочной панели, нигде не
// компилируется (CPU-интегратор гоняет байткод AST, а не этот текст).
// Экстраполяция оборачивает CPU-эквивалент базы той же wrap_extrapolation,
// так что обе панели показывают одну и ту же структуру стадий.
std::string compute_krs_cpu_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                       const System& sys,
                                       const std::string& scheme) {
    return krs_for_scheme_impl(custom_schemes, sys, scheme, &codegen_scheme_cpu_equivalent, nullptr);
}


// SignalMetricsAnalysisSession — устройство LLEAnalysisSession, один future на
// оба режима (run_signal_metrics различает их по request.dimension).


AdaptiveCode adaptive_code_for_scheme(const std::vector<CustomScheme>& custom_schemes,
                                      const System& sys, const std::string& scheme) {
    // Кастомная КРС с этим именем перекрывает встроенную (как в krs_for_scheme_impl), а её
    // текст непрозрачен: оценки ошибки у неё нет.
    for (const auto& cs : custom_schemes)
        if (cs.name == scheme) throw std::runtime_error(adaptive_scheme_hint());
    const Scheme sch = scheme_from_name(scheme);
    if (scheme_supports_adaptive(sch)) return codegen_adaptive(sys, sch);
    if (const int K = gbs_stage_count(sch); K >= 2) return codegen_adaptive_gbs(sys, K);
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme, &spec)) {
        if (spec.n.size() < 2)
            throw std::runtime_error("adaptive step needs an extrapolation with at least two stages");
        if (spec.re_at_output) {
            // ExtrZ — как в krs_for_scheme_impl: нужно комплексное ядро встроенной CD-схемы; кастомная
            // КРС с тем же именем перекрыла бы встроенную, а её ядро не выделить.
            for (const auto& cs : custom_schemes)
                if (cs.name == spec.base)
                    throw std::runtime_error("ExtrZ needs a built-in complex base, '" + spec.base + "' is a custom KRS");
            int p = 1; bool sym = false;
            if (!scheme_has_complex_core(spec.base) || !builtin_scheme_traits(spec.base, &p, &sym))
                throw std::runtime_error("ExtrZ base '" + spec.base + "' has no complex core");
            return codegen_adaptive_extrapolation_complex(
                sys, codegen_scheme_complex_core(sys, scheme_from_string(spec.base)), spec.n, p, spec.base,
                scheme_rhs_per_step(sys, spec.base));
        }
        std::string body, why; int p = 1; bool sym = false;
        if (!resolve_wrapper_base(custom_schemes, sys, spec.base, &codegen_scheme, body, p, sym, &why))
            throw std::runtime_error(!why.empty() ? why
                : "extrapolation base '" + spec.base + "' is unknown or has no step body");
        // f базы: кастомная КРС непрозрачна — 0 (счётчик видит только F1); встроенная — по
        // scheme_rhs_per_step, CD-семейство в том числе.
        bool base_custom = false;
        for (const auto& cs : custom_schemes) if (cs.name == spec.base) base_custom = true;
        return codegen_adaptive_extrapolation(sys, body, spec.n, p, sym, spec.base,
                                              base_custom ? 0.0 : scheme_rhs_per_step(sys, spec.base));
    }
    throw std::runtime_error(adaptive_scheme_hint());
}
std::function<void(ParametricEngine&)> BifurcationAnalysisSession::prewarm_task(int diagram_idx) const {
    if (diagram_idx < 0 || diagram_idx >= (int)diagrams.size()) return {};
    const BifurcationDiagramConfig& bd = diagrams[diagram_idx];
    if (bd.mode_2d) {
        Bifurcation2DRequest req = build_bif2d_request(*this, bd);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    Bifurcation1DRequest req = build_bif1d_request(*this, bd);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}
std::function<void(ParametricEngine&)> LLEAnalysisSession::prewarm_task(int curve_idx) const {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return {};
    const LLECurveConfig& c = curves[curve_idx];
    if (c.mode_2d) {
        LLE2DRequest req = build_lle2d_request(*this, c);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    LLE1DRequest req = build_lle1d_request(*this, c);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}
std::function<void(ParametricEngine&)> LyapunovSpectrumAnalysisSession::prewarm_task(int curve_idx) const {
    if (curve_idx < 0 || curve_idx >= (int)curves.size()) return {};
    const LSCurveConfig& c = curves[curve_idx];
    if (c.mode_2d) {
        LS2DRequest req = build_ls2d_request(*this, c);
        if (req.krs_body.empty()) return {};
        return [req](ParametricEngine& e) { e.prewarm(req); };
    }
    LS1DRequest req = build_ls1d_request(*this, c);
    if (req.krs_body.empty()) return {};
    return [req](ParametricEngine& e) { e.prewarm(req); };
}

const rqa::Result* find_rqa(const AnalysisResult& res, const RqaJob& job)
{
    for (const RqaOutcome& o : res.rqa_out)
        if (o.job == job) return &o.res;
    return nullptr;
}
