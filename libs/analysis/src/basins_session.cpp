#include "session_internal.h"

namespace ucuda::analysis_detail {


BasinsRequest build_basins_request(const BasinsAnalysisSession& s,
                                          const BasinsConfig& c) {
    BasinsRequest req;
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

    req.axis_x_var      = (c.axis_x_var >= 0 && c.axis_x_var < req.amountOfX) ? c.axis_x_var : 0;
    req.axis_y_var      = (c.axis_y_var >= 0 && c.axis_y_var < req.amountOfX) ? c.axis_y_var : (req.amountOfX > 1 ? 1 : 0);
    req.axis_x_lo       = parse_d(c.axis_x_lo_text, -10.0);
    req.axis_x_hi       = parse_d(c.axis_x_hi_text,  10.0);
    req.axis_y_lo       = parse_d(c.axis_y_lo_text, -10.0);
    req.axis_y_hi       = parse_d(c.axis_y_hi_text,  10.0);
    if (req.axis_x_hi < req.axis_x_lo) std::swap(req.axis_x_lo, req.axis_x_hi);
    if (req.axis_y_hi < req.axis_y_lo) std::swap(req.axis_y_lo, req.axis_y_hi);
    req.n_pts           = parse_i(c.n_pts_text, 200);
    // -1 — combination (см. loopCalculateDiscreteModel_int / draw_writable_var_combo).
    req.writable_var    = (c.writable_var >= -1 && c.writable_var < req.amountOfX) ? c.writable_var : 0;
    req.h               = parse_d(c.h_text, 0.01);
    req.t_max           = parse_d(c.t_max_text, 1000.0);
    req.transient_time  = parse_d(c.transient_text, 10000.0);
    req.pre_scaller     = std::max(1, parse_i(c.pre_scaller_text, 1));
    req.max_value       = parse_d(c.max_value_text, 1.0e6);
    req.eps_dbscan      = parse_d(c.eps_dbscan_text, 0.5);
    req.csv_output_path = c.csv_save_enabled ? c.csv_output_path : std::string{};

    // Feature dispatch для avgPeakFinderCUDA. Кэшируем сейчас, чтобы worker
    // не зависел от GUI-state. Кламп в валидный диапазон на случай ручного
    // редактирования сессии.
    int f1 = c.feature1, f2 = c.feature2;
    if (f1 < 0 || f1 >= BF_FEATURE_COUNT) f1 = BF_FEATURE1_DEFAULT;
    if (f2 < 0 || f2 >= BF_FEATURE_COUNT) f2 = BF_FEATURE2_DEFAULT;
    req.feature1 = f1;
    req.feature2 = f2;
    req.mult1    = (numb)parse_d(c.mult_feature1_text, 1.0);
    req.mult2    = (numb)parse_d(c.mult_feature2_text, 1.0);
    return req;
}


// Сама таблица kBasinFeatureNames и её проверки (длина, только-ASCII) живут в
// analysis_session.h рядом с enum BasinFeature: static_assert'у нужен constexpr-массив, а
// constexpr в namespace scope несовместим с extern-объявлением.


void apply_basins_result(BasinsConfig& c, BasinsResult&& r) {
    c.result = std::move(r);
    c.last_run_ok = c.result.ok;
    if (!c.result.ok) c.last_error = c.result.error;
    c.data_generation++;
    c.fit_request = true;
}








// Значение узла сетки — зеркало getValueByIdx (cudaLibrary.cu): та же
// lerp-форма и те же вырожденные случаи, иначе восстановленное НУ не совпадёт
// с тем, что реально считало ядро для этой ячейки.
double basins_grid_value(int point_idx, int n_pts, double lo, double hi) {
    if (n_pts <= 0) return lo;
    if (n_pts == 1) return hi;
    const double t = (double)point_idx / (double)(n_pts - 1);
    return (1.0 - t) * lo + t * hi;
}


std::string basins_fmt_num(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}


// Перенос настроек basins-config'а в фазовую подсессию. Зовётся ровно перед
// запуском (не каждый кадр), как apply_shared_to_* в Custom-табе.
void apply_basins_config_to_phase(const BasinsAnalysisSession& s,
                                         const BasinsConfig& c,
                                         PhaseAnalysisSession& ph) {
    ph.vars           = s.vars;
    ph.params         = s.params;
    ph.sys            = s.sys;
    ph.custom_schemes = s.custom_schemes;
    // Схема и шаг ОБЯЗАНЫ совпадать с расчётом карты — иначе траектория не
    // обязана прийти в тот же аттрактор, что и ячейка сетки.
    ph.scheme       = c.scheme;
    ph.adaptive = c.adaptive;
    ph.symmetry_s   = c.symmetry_s;
    ph.step_h       = c.h_text;
    ph.param_values = c.param_values;
    // Своё время — единственное, чем портрет отличается от карты.
    ph.sim_time   = c.pp_t_max_text;
    ph.skip_time  = c.pp_transient_text;
    ph.decimation = c.pp_prescaller_text;
    ph.use_gpu    = c.pp_use_gpu;
    // Автозапуском рулит слой Basins (pp_autorun + очередь), собственный
    // auto_recompute подсессии должен молчать.
    ph.auto_recompute = false;
}
}
using namespace ucuda::analysis_detail;


// BasinsAnalysisSession — N configs на сессию (по образцу
// BifurcationAnalysisSession::diagrams). 5 inner tabs (Basins/AvgPk/AvgInt/
// States/Scatter) живут per-config через BasinsConfig::active_plot_tab.

void BasinsAnalysisSession::load_from_record(const SystemRecord& r,
    const std::vector<std::string>& vars_,
    const std::vector<std::string>& params_) {
    vars = vars_;
    params = params_;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    BasinsConfig c;
    c.label = "Basins 1";
    c.h_text = r.step_h.empty() ? std::string("0.01") : r.step_h;
    c.symmetry_s = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;

    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        c.param_values[p] = (it != r.param_values.end()) ? it->second : "";
    }
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        c.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : "";
    }
    c.axis_x_var = 0;
    c.axis_y_var = (vars.size() > 1) ? 1 : 0;
    c.writable_var = 0;

    configs.clear();
    configs.push_back(std::move(c));
    active_config_index = 0;
    running_config_index = -1;
    // Слоты фазовых портретов пересоздаются с нуля: старые ссылались на
    // конфиги (и систему) предыдущей записи библиотеки.
    phase_slots.clear();
    phase_queue.clear();
    ensure_phase_slots();
}


void BasinsAnalysisSession::add_config() {
    BasinsConfig c;
    if (!configs.empty()) {
        // Глубокая копия последнего config — пользователь обычно правит 1-2 поля.
        c = configs.back();
        c.result = BasinsResult{};
        c.last_run_ok = false;
        c.last_error.clear();
        c.data_generation = 0;
        c.fit_request = false;
    } else {
        for (const auto& v : vars) c.initial_conditions[v] = "";
        for (const auto& p : params) c.param_values[p] = "";
    }
    c.label = "Basins " + std::to_string(configs.size() + 1);
    configs.push_back(std::move(c));
    active_config_index = (int)configs.size() - 1;
    ensure_phase_slots();   // новому config'у — свой слот портретов
}


void BasinsAnalysisSession::remove_config(int i) {
    if (i < 0 || i >= (int)configs.size()) return;
    // Запрещаем удалять config, чей расчёт сейчас идёт — иначе worker положит
    // результат в несуществующий слот при poll().
    if (in_flight && running_config_index == i) return;
    configs.erase(configs.begin() + i);
    // Слот портретов удаляем ПО ТОМУ ЖЕ индексу (а не хвостовой через
    // ensure_phase_slots) — иначе позиционная привязка слотов к configs
    // разъедется на всех элементах правее i.
    if (i < (int)phase_slots.size()) phase_slots.erase(phase_slots.begin() + i);
    for (auto it = phase_queue.begin(); it != phase_queue.end(); ) {
        if (*it == i) it = phase_queue.erase(it);
        else ++it;
    }
    for (auto& q : phase_queue) if (q > i) --q;
    if (in_flight && running_config_index > i) running_config_index--;
    if (active_config_index >= (int)configs.size())
        active_config_index = (int)configs.size() - 1;
    if (active_config_index < 0) active_config_index = 0;
}


bool BasinsAnalysisSession::run(ParametricEngine& engine, int config_idx) {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;
    BasinsConfig& c = configs[config_idx];
    // Keep previous last_run_ok so the old heatmap stays visible until
    // apply_basins_result swaps in fresh data.
    c.last_error.clear();

    BasinsRequest req = build_basins_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code пуст (нет валидной системы или scheme)";
        return false;
    }
    BasinsResult r = engine.run_basins(req);
    bool ok = r.ok;
    apply_basins_result(c, std::move(r));
    return ok;
}


bool BasinsAnalysisSession::run_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    BasinsConfig& c = configs[config_idx];
    // Keep previous last_run_ok so the old heatmap stays visible until
    // apply_basins_result swaps in fresh data.
    c.last_error.clear();
    last_run_label.clear();
    cancel_token         = std::make_shared<std::atomic<bool>>(false);
    progress_token       = std::make_shared<std::atomic<float>>(0.0f);
    progress_phase_token = std::make_shared<std::atomic<int>>(0);

    BasinsRequest req = build_basins_request(*this, c);
    if (req.krs_body.empty()) {
        c.last_error = "krs_code пуст (нет валидной системы или scheme)";
        cancel_token.reset();
        progress_token.reset();
        progress_phase_token.reset();
        return false;
    }
    req.cancel         = cancel_token;
    req.progress       = progress_token;
    req.progress_phase = progress_phase_token;

    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();
    run_future = std::async(std::launch::async, [&engine, req = std::move(req)]() {
        return engine.run_basins(req);
    });
    return true;
}


bool BasinsAnalysisSession::run_recluster_async(ParametricEngine& engine, int config_idx) {
    if (in_flight) return false;
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;

    BasinsConfig& c = configs[config_idx];
    if (!c.last_run_ok || c.result.avg_peaks.empty() ||
        c.result.avg_intervals.empty() || c.result.helpful_array.empty() ||
        c.result.n_pts <= 0) {
        c.last_error = "Нет кэша фич — запусти полный Run сначала.";
        return false;
    }

    // Берём krs_body/amountOfX/eps_dbscan из text-полей (как build_basins_request
    // для обычного запуска). Кэш-ключ модуля совпадёт с предыдущим run_basins,
    // если система не менялась — compile_basins_if_needed обойдётся no-op'ом.
    BasinsRequest sim_req = build_basins_request(*this, c);
    if (sim_req.krs_body.empty()) {
        c.last_error = "krs_code пуст (нет валидной системы или scheme)";
        return false;
    }

    BasinsReclusterRequest req;
    req.krs_body      = std::move(sim_req.krs_body);
    req.amountOfX     = sim_req.amountOfX;
    req.n_pts         = c.result.n_pts;
    req.eps_dbscan    = sim_req.eps_dbscan;
    req.avg_peaks     = c.result.avg_peaks;
    req.avg_intervals = c.result.avg_intervals;
    req.helpful_array = c.result.helpful_array;

    c.last_error.clear();
    last_run_label.clear();
    cancel_token         = std::make_shared<std::atomic<bool>>(false);
    progress_token       = std::make_shared<std::atomic<float>>(0.0f);
    // Для recluster sim-фазы нет → ставим phase=2, чтобы топ-бар сразу показал
    // "(2/2 cluster)" и progress соответствовал фазе кластеризации.
    progress_phase_token = std::make_shared<std::atomic<int>>(2);
    req.cancel   = cancel_token;
    req.progress = progress_token;

    in_flight = true;
    running_config_index = config_idx;
    compute_start_time = std::chrono::steady_clock::now();

    // Снапшот текущего result'а — async-задача вернёт его с заменённым
    // basin_idx/n_clusters. Так poll() через apply_basins_result сохранит
    // все остальные поля (фичи, axis_*, snapshot) без копирования вручную.
    BasinsResult base = c.result;
    base.snapshot.eps_dbscan = req.eps_dbscan;

    run_future = std::async(std::launch::async,
        [&engine, req = std::move(req), base = std::move(base)]() mutable {
            BasinsReclusterResult rc = engine.run_basins_recluster(req);
            BasinsResult out = std::move(base);
            out.ok        = rc.ok;
            out.cancelled = rc.cancelled;
            out.error     = std::move(rc.error);
            if (rc.ok) {
                out.basin_idx       = std::move(rc.basin_idx);
                out.n_clusters      = rc.n_clusters;
                out.min_cluster_idx = rc.min_cluster_idx;
            }
            return out;
        });
    return true;
}


void BasinsAnalysisSession::request_cancel() {
    if (cancel_token) cancel_token->store(true, std::memory_order_relaxed);
}


bool BasinsAnalysisSession::poll() {
    if (!in_flight) return false;
    if (!run_future.valid()) {
        in_flight = false;
        running_config_index = -1;
        cancel_token.reset(); progress_token.reset(); progress_phase_token.reset();
        return false;
    }
    if (run_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;

    BasinsResult r = run_future.get();
    bool cancelled = r.cancelled;
    int idx = running_config_index;
    std::string label = (idx >= 0 && idx < (int)configs.size() && !configs[idx].label.empty())
                            ? configs[idx].label : std::string("basins");
    if (!cancelled && idx >= 0 && idx < (int)configs.size()) {
        apply_basins_result(configs[idx], std::move(r));
    }
    last_run_completed_at = std::chrono::steady_clock::now();
    last_run_seconds = std::chrono::duration<double>(last_run_completed_at - compute_start_time).count();
    last_run_label = label;
    last_run_succeeded = !cancelled;
    log_run_completed(last_run_label.c_str(), last_run_succeeded, last_run_seconds);
    in_flight = false;
    running_config_index = -1;
    cancel_token.reset();
    progress_token.reset();
    progress_phase_token.reset();
    return true;
}


std::vector<int> renumber_basins_spiral(const std::vector<int>& src, int N,
                                        int& out_n_pos, int& out_n_neg) {
    out_n_pos = 0;
    out_n_neg = 0;
    std::vector<int> dst(src.size(), 0);
    if ((int)src.size() != N * N || N <= 0) return dst;
    const std::vector<int> order = spiral_coords_from_center(N);
    std::map<int, int> pos_map, neg_map;
    int pos_next = 1, neg_next = 1;
    for (int idx : order) {
        const int v = src[(size_t)idx];
        if (v > 0) {
            auto it = pos_map.find(v);
            if (it == pos_map.end()) { it = pos_map.emplace(v, pos_next++).first; }
            dst[(size_t)idx] = it->second;
        } else if (v < 0) {
            auto it = neg_map.find(v);
            if (it == neg_map.end()) { it = neg_map.emplace(v, -(neg_next++)).first; }
            dst[(size_t)idx] = it->second;
        }
        // v == 0 — diverged, dst остаётся 0.
    }
    out_n_pos = pos_next - 1;
    out_n_neg = neg_next - 1;
    return dst;
}


std::string build_basins_phase_run_signature(const BasinsConfig& c) {
    std::string o;
    o.reserve(256);
    auto add = [&o](const std::string& v) { o += v; o += '|'; };
    add(c.pp_t_max_text);
    add(c.pp_transient_text);
    add(c.pp_prescaller_text);
    add(c.pp_use_gpu ? "1" : "0");
    add(c.scheme);
    add(c.symmetry_s);
    add(c.h_text);
    for (const auto& kv : c.param_values)       { o += kv.first; o += '='; o += kv.second; o += ';'; }
    o += '|';
    // НУ не-осевых переменных: ячейка та же, а стартовая точка другая.
    for (const auto& kv : c.initial_conditions) { o += kv.first; o += '='; o += kv.second; o += ';'; }
    return o;
}


std::string build_basins_phase_signature(const BasinsConfig& c) {
    std::string o = build_basins_phase_run_signature(c);
    o += '|';
    o += std::to_string(c.data_generation);
    o += '|';
    o += (c.renumber_spiral ? "1" : "0");
    o += '|';
    o += c.pp_max_attractors_text;
    return o;
}




BasinsPhaseSlot& BasinsAnalysisSession::phase_slot(int i) {
    ensure_phase_slots();
    // Страховка на случай вызова без единого config'а: отдаём общий пустой
    // слот, а не висячую ссылку. Все реальные пути сюда с пустым списком не
    // приходят (GUI проверяет configs.empty() раньше).
    static BasinsPhaseSlot fallback;
    if (phase_slots.empty()) return fallback;
    if (i < 0) i = 0;
    if (i >= (int)phase_slots.size()) i = (int)phase_slots.size() - 1;
    return *phase_slots[(size_t)i];
}


bool BasinsAnalysisSession::rebuild_phase_ics(int config_idx) {
    ensure_phase_slots();
    if (config_idx < 0 || config_idx >= (int)configs.size()) return false;
    BasinsConfig&    c  = configs[(size_t)config_idx];
    BasinsPhaseSlot& sl = *phase_slots[(size_t)config_idx];

    std::vector<int> new_cells;
    std::vector<int> new_ids;

    const int    n   = c.result.n_pts;
    const size_t tot = (size_t)n * (size_t)n;
    const int*   ids = (c.last_run_ok && n > 0 && c.result.basin_idx.size() == tot)
                           ? basins_display_ids(c) : nullptr;
    if (ids) {
        // id -> ячейки. 0 (расходящиеся решения) не рисуем вообще.
        std::map<int, std::vector<int>> by_id;
        for (size_t k = 0; k < tot; ++k) {
            const int id = ids[k];
            if (id == 0) continue;
            by_id[id].push_back((int)k);
        }
        // Порядок: положительные по возрастанию (1,2,3...), затем отрицательные
        // по модулю (-1,-2,-3...). Ограничитель — предохранитель от «нашлась
        // тысяча бассейнов», поэтому режем именно по этому порядку, без
        // сортировки по площади.
        std::vector<int> order;
        order.reserve(by_id.size());
        for (const auto& kv : by_id) if (kv.first > 0) order.push_back(kv.first);
        for (auto it = by_id.rbegin(); it != by_id.rend(); ++it)
            if (it->first < 0) order.push_back(it->first);

        int limit = parse_num_int(c.pp_max_attractors_text, 1);
        if (limit < 1) limit = 1;
        if ((int)order.size() > limit) order.resize((size_t)limit);

        for (int id : order) {
            const std::vector<int>& cells = by_id[id];
            if (cells.empty()) continue;
            std::mt19937 rng(kBasinsPhaseSeed + (unsigned)id * 2654435761u);
            std::uniform_int_distribution<size_t> dist(0, cells.size() - 1);
            new_cells.push_back(cells[dist(rng)]);
            new_ids.push_back(id);
        }
    }

    // Тот же НАБОР ячеек, но, возможно, в другом порядке — типичный случай при toggle «Renumber
    // (spiral)»: кластеры те же, изменились только их номера, а значит и порядок обхода
    // «положительные по возрастанию, затем отрицательные». Считаем перестановку: perm[k] = индекс
    // старой серии, которая должна встать на новое место k. O(n²), но n <= max attractors.
    std::vector<int> perm;
    bool same_set = (new_cells.size() == sl.cells.size());
    if (same_set) {
        perm.assign(new_cells.size(), -1);
        std::vector<char> used(sl.cells.size(), 0);
        for (size_t k = 0; k < new_cells.size(); ++k) {
            int found = -1;
            for (size_t j = 0; j < sl.cells.size(); ++j)
                if (!used[j] && sl.cells[j] == new_cells[k]) { found = (int)j; break; }
            if (found < 0) { same_set = false; break; }
            used[(size_t)found] = 1;
            perm[k] = found;
        }
    }
    const bool cells_changed = !same_set;

    // Видимость переживает перестройку. Ключ — ЯЧЕЙКА, а не подпись: при
    // переномерации подписи меняются, а бассейн остаётся тем же, и снятая
    // пользователем галка не должна возвращаться сама собой.
    std::map<int, bool> prev_visible;
    for (size_t k = 0; k < sl.cells.size() && k < sl.phase.ic_sets.size(); ++k)
        prev_visible[sl.cells[k]] = sl.phase.ic_sets[k].visible;

    sl.cells = new_cells;

    std::vector<InitialConditionSet> ics;
    ics.reserve(new_cells.size());
    for (size_t k = 0; k < new_cells.size(); ++k) {
        const int cell = new_cells[k];
        const int id   = new_ids[k];
        const int ix   = (n > 0) ? (cell % n) : 0;
        const int iy   = (n > 0) ? (cell / n) : 0;

        InitialConditionSet ic;
        ic.label = "Basin " + std::to_string(id) + (id < 0 ? " (FP)" : " (Osc)");
        for (const auto& v : vars) {
            auto it = c.initial_conditions.find(v);
            ic.values[v] = (it != c.initial_conditions.end()) ? it->second : std::string();
        }
        // Осевые переменные перекрываем координатами узла сетки.
        if (c.result.axis_x_var >= 0 && c.result.axis_x_var < (int)vars.size())
            ic.values[vars[(size_t)c.result.axis_x_var]] =
                basins_fmt_num(basins_grid_value(ix, n, c.result.axis_x_lo, c.result.axis_x_hi));
        if (c.result.axis_y_var >= 0 && c.result.axis_y_var < (int)vars.size())
            ic.values[vars[(size_t)c.result.axis_y_var]] =
                basins_fmt_num(basins_grid_value(iy, n, c.result.axis_y_lo, c.result.axis_y_hi));

        auto pv = prev_visible.find(cell);
        ic.visible = (pv != prev_visible.end()) ? pv->second : true;
        ics.push_back(std::move(ic));
    }
    sl.phase.ic_sets = std::move(ics);

    // Набор ячеек не изменился: траектории уже посчитаны — переставляем их под
    // новый порядок и переподписываем на месте. Ни одного запуска GPU.
    AnalysisResult& res = sl.phase.result;
    const size_t n_new = sl.phase.ic_sets.size();
    if (!cells_changed && sl.has_result && res.trajectories.size() == n_new) {
        // Перестановка без глубокого копирования: траектории переезжают
        // move'ом (их суммарный объём — десятки мегабайт).
        auto permute = [&perm, n_new](auto& vec) {
            if (vec.size() != n_new) return;
            auto tmp = std::move(vec);
            vec.clear();
            vec.resize(n_new);
            for (size_t k = 0; k < n_new; ++k) vec[k] = std::move(tmp[(size_t)perm[k]]);
        };
        permute(res.trajectories);
        permute(res.visible);
        permute(res.ic_text);
        permute(res.snapshot.ic_flat);
        permute(res.snapshot.ic_labels);
        res.labels.resize(n_new);
        for (size_t k = 0; k < n_new; ++k)
            res.labels[k] = sl.phase.ic_sets[k].label;
        // Перестановка серий обязана инвалидировать GPU-кэш Plot2DView, иначе
        // точки останутся от прежнего порядка (кэш ключуется по data_generation).
        sl.phase.data_generation++;
    }
    return cells_changed;
}


void BasinsAnalysisSession::request_phase_run(int config_idx) {
    if (config_idx < 0 || config_idx >= (int)configs.size()) return;
    for (int q : phase_queue) if (q == config_idx) return;   // уже в очереди
    phase_queue.push_back(config_idx);
}


bool BasinsAnalysisSession::start_next_phase() {
    ensure_phase_slots();
    if (phase_queue.empty()) return false;
    // Один расчёт портретов за раз на всю сессию.
    for (const auto& sl : phase_slots)
        if (sl && sl->phase.in_flight) return false;

    while (!phase_queue.empty()) {
        const int i = phase_queue.front();
        if (i < 0 || i >= (int)configs.size()) { phase_queue.pop_front(); continue; }
        // Карта этого config'а ещё считается — ждём: НУ пока не финальные
        // (кластеры и их номера появятся только после DBSCAN).
        if (in_flight && running_config_index == i) return false;

        phase_queue.pop_front();
        BasinsConfig&    c  = configs[(size_t)i];
        BasinsPhaseSlot& sl = *phase_slots[(size_t)i];
        const bool cells_changed = rebuild_phase_ics(i);
        const std::string run_sig = build_basins_phase_run_signature(c);

        if (sl.phase.ic_sets.empty()) {
            // Бассейнов не найдено — рисовать нечего; помечаем как обработанное,
            // чтобы autorun не долбился в пустоту каждый кадр.
            sl.phase.result = AnalysisResult{};
            sl.has_result   = false;
            sl.cells.clear();
            sl.run_signature        = build_basins_phase_signature(c);
            sl.run_inputs_signature = run_sig;
            continue;
        }
        // Ячейки те же и входы расчёта не менялись — траектории заведомо
        // идентичны (типичный случай: toggle «Renumber (spiral)» без
        // срабатывания ограничителя). rebuild_phase_ics уже переписал подписи
        // на месте; GPU не трогаем.
        if (!cells_changed && sl.has_result && sl.run_inputs_signature == run_sig) {
            sl.run_signature = build_basins_phase_signature(c);
            continue;
        }
        apply_basins_config_to_phase(*this, c, sl.phase);
        if (sl.phase.recompute_async()) {
            sl.run_signature        = build_basins_phase_signature(c);
            sl.run_inputs_signature = run_sig;
            return true;
        }
    }
    return false;
}


bool BasinsAnalysisSession::poll_phase() {
    ensure_phase_slots();
    bool any = false;
    for (auto& sl : phase_slots) {
        if (!sl) continue;
        if (sl->phase.poll()) {
            sl->has_result = sl->phase.result.ok;
            any = true;
        }
    }
    return any;
}

using namespace ucuda::analysis_detail;
void ensure_basins_spiral_cache(BasinsConfig& c) {
    if (c.basin_idx_spiral_gen == c.data_generation
        && c.basin_idx_spiral.size() == c.result.basin_idx.size()) return;
    int n_pos = 0, n_neg = 0;
    c.basin_idx_spiral = renumber_basins_spiral(c.result.basin_idx, c.result.n_pts,
                                                n_pos, n_neg);
    c.n_clusters_spiral      = n_pos;
    c.min_cluster_idx_spiral = -n_neg;
    c.basin_idx_spiral_gen   = c.data_generation;
}

using namespace ucuda::analysis_detail;
const int* basins_display_ids(BasinsConfig& c) {
    if (c.result.basin_idx.empty()) return nullptr;
    if (!c.renumber_spiral) return c.result.basin_idx.data();
    ensure_basins_spiral_cache(c);
    if (c.basin_idx_spiral.size() != c.result.basin_idx.size())
        return c.result.basin_idx.data();
    return c.basin_idx_spiral.data();
}
