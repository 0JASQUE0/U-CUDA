#include "ucuda/application/app_model.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/domain/num_parse.h"   // fmt_num_shortest — round-trip формат для пинов узлов
#include "ucuda/application/session_persistence.h"
#include "ucuda/application/time_estimate.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/cuda/runtime_settings.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "ucuda/equations/sysparse.hpp"
#include "ucuda/equations/dae_ir.h"       // build_system для ДАУ
#include "ucuda/equations/dae_devices.h"  // dae_dc_operating_point
#include "ucuda/numerics/kicad_import.h"
#include "ucuda/numerics/spice_parse.h"  // import_circuit
#include <filesystem>
#include <stdexcept>


// Cross-analysis batch queue

namespace {

// Всё, от чего зависит КЛЮЧ кэша модуля, и ничего больше: схема, система, размерность и вид свипа
// по каждой оси. Диапазоны, число точек, время интегрирования в ключ не входят — поэтому набор
// чисел в полях прогрев не перезапускает, а переключение схемы перезапускает.
std::string prewarm_session_id(const System& sys, const std::vector<std::string>& vars,
                               const std::vector<CustomScheme>& custom_schemes) {
    std::string joined;
    for (const std::string& r : sys.rhs) { joined += r; joined += '\x1f'; }
    // Тело кастомной схемы правится без смены имени, поэтому в идентификатор идёт и оно.
    for (const CustomScheme& cs : custom_schemes) { joined += cs.name; joined += cs.body; joined += '\x1f'; }
    return std::to_string(std::hash<std::string>{}(joined)) + ":" + std::to_string(vars.size());
}

template <class Cfg>
std::string prewarm_sig(const std::string& session_id, const Cfg& c) {
    std::string s = session_id;
    s += '\x1f';
    s += c.scheme;
    s += c.mode_2d              ? "|2d" : "|1d";
    s += c.sweep_over_var       ? 'v' : '-';
    s += c.sweep_over_h         ? 'h' : '-';
    s += c.sweep_over_var_2     ? 'V' : '-';
    s += c.sweep_over_h_2       ? 'H' : '-';
    s += c.continuation         ? 'c' : '-';
    s += c.use_gpu              ? 'g' : '-';
    // Настройки пиков и fmad тоже входят в ключ модуля (см. hash_key в parametric_engine.cpp).
    s += ":pk" + std::to_string(peak_config_epoch());
    s += get_nvrtc_fmad() ? ":fm1" : ":fm0";
    return s;
}

}  // namespace

// Прогрев ПЕРВОГО запуска. Зовётся каждый кадр рядом с тиками очередей; почти всегда это
// несколько сравнений строк и выход.
void AppModel::poll_parametric_prewarm() {
    // Не соревнуемся с идущим расчётом: пока он считает, хвост очереди греет
    // prewarm_rest_of_parametric_queue, а лишняя компиляция отняла бы у него CPU.
    if (bifurcation_session.in_flight || lle_session.in_flight || ls_session.in_flight ||
        metrics_session.in_flight) return;
    if (!parametric_queue_.empty()) return;
    if (parametric_prewarm_future.valid() &&
        parametric_prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    const auto now = std::chrono::steady_clock::now();
    std::function<void(ParametricEngine&)> task;

    // Скан одной сессии. Решение «что и когда греть» живёт в PrewarmWatch::pick (prewarm_watch.h),
    // здесь — только сбор сигнатур и сборка задачи. Задача строится ЗДЕСЬ, на UI-потоке (внутри
    // codegen), фоновому потоку остаётся NVRTC.
    auto scan = [&](PrewarmWatch& w, const std::string& session_id,
                    size_t count, auto&& sig_of, auto&& make_task) {
        if (task) return;
        std::vector<std::string> current;
        current.reserve(count);
        for (size_t i = 0; i < count; ++i) current.push_back(sig_of(i, session_id));
        const int idx = w.pick(current, now, std::chrono::milliseconds(400));
        if (idx < 0) return;
        auto t = make_task(idx);
        if (t) task = std::move(t);
    };

    scan(prewarm_watch_bif,
         prewarm_session_id(bifurcation_session.sys, bifurcation_session.vars,
                            bifurcation_session.custom_schemes),
         bifurcation_session.diagrams.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, bifurcation_session.diagrams[i]); },
         [&](int i) { return bifurcation_session.prewarm_task(i); });
    scan(prewarm_watch_lle,
         prewarm_session_id(lle_session.sys, lle_session.vars, lle_session.custom_schemes),
         lle_session.curves.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, lle_session.curves[i]); },
         [&](int i) { return lle_session.prewarm_task(i); });
    scan(prewarm_watch_ls,
         prewarm_session_id(ls_session.sys, ls_session.vars, ls_session.custom_schemes),
         ls_session.curves.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, ls_session.curves[i]); },
         [&](int i) { return ls_session.prewarm_task(i); });
    scan(prewarm_watch_metrics,
         prewarm_session_id(metrics_session.sys, metrics_session.vars, metrics_session.custom_schemes),
         metrics_session.configs.size(),
         [&](size_t i, const std::string& id) { return prewarm_sig(id, metrics_session.configs[i]); },
         [&](int i) { return metrics_session.prewarm_task(i); });

    if (!task) return;
    // Движок создаётся тут же: CUDA-контекст он поднимает лениво, уже в фоновом потоке.
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    ParametricEngine* eng = parametric_engine.get();
    parametric_prewarm_future = std::async(std::launch::async,
                                           [eng, task = std::move(task)] { task(*eng); });
}

// Прогрев хвоста очереди. Запросы собираются ЗДЕСЬ, на UI-потоке: фоновому потоку нельзя читать
// сессии, их правит пользователь.
void AppModel::prewarm_rest_of_parametric_queue() {
    if (!parametric_engine || parametric_queue_.empty()) return;
    // Присваивание поверх работающего future заблокировало бы UI-поток в его деструкторе — в этом
    // случае просто пропускаем: не прогрелось, значит Run скомпилирует сам (см. analysis_session).
    if (parametric_prewarm_future.valid() &&
        parametric_prewarm_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    std::vector<std::function<void(ParametricEngine&)>> tasks;
    for (const ParametricQueueItem& it : parametric_queue_) {
        std::function<void(ParametricEngine&)> t;
        switch (it.kind) {
        case ParametricQueueItem::Kind::Bifurcation: t = bifurcation_session.prewarm_task(it.index); break;
        case ParametricQueueItem::Kind::LLE:         t = lle_session.prewarm_task(it.index);         break;
        case ParametricQueueItem::Kind::LS:          t = ls_session.prewarm_task(it.index);          break;
        case ParametricQueueItem::Kind::Metrics:     t = metrics_session.prewarm_task(it.index);     break;
        }
        if (t) tasks.push_back(std::move(t));
        if ((int)tasks.size() >= kPrewarmAhead) break;
    }
    if (tasks.empty()) return;

    ParametricEngine* eng = parametric_engine.get();
    parametric_prewarm_future = std::async(std::launch::async, [eng, tasks = std::move(tasks)] {
        for (const auto& task : tasks) task(*eng);
    });
}

bool AppModel::start_next_in_parametric_queue() {
    if (bifurcation_session.in_flight ||
        lle_session.in_flight ||
        ls_session.in_flight ||
        metrics_session.in_flight) return false;
    if (parametric_queue_.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    while (!parametric_queue_.empty()) {
        ParametricQueueItem it = parametric_queue_.front();
        parametric_queue_.pop_front();
        bool ok = false;
        switch (it.kind) {
        case ParametricQueueItem::Kind::Bifurcation:
            if (it.index >= 0 && it.index < (int)bifurcation_session.diagrams.size())
                ok = bifurcation_session.run_async(*parametric_engine, it.index);
            break;
        case ParametricQueueItem::Kind::LLE:
            if (it.index >= 0 && it.index < (int)lle_session.curves.size())
                ok = lle_session.run_async(*parametric_engine, it.index);
            break;
        case ParametricQueueItem::Kind::LS:
            if (it.index >= 0 && it.index < (int)ls_session.curves.size())
                ok = ls_session.run_async(*parametric_engine, it.index);
            break;
        case ParametricQueueItem::Kind::Metrics:
            if (it.index >= 0 && it.index < (int)metrics_session.configs.size())
                ok = metrics_session.run_async(*parametric_engine, it.index);
            break;
        }
        if (ok) { prewarm_rest_of_parametric_queue(); return true; }
        // ok == false (например, krs пуст / индекс плохой) — last_error
        // выставлен соответствующим run_async; идём дальше.
    }
    return false;
}

// Custom-tab pipeline queue drainer

namespace {
// Format a double for insertion into a text-based numeric input field.
// Все три вызывающих (pin_fixed_param / _h / _ic) пиннят ЗНАЧЕНИЕ УЗЛА сетки, пришедшее из snap'а
// по карте или по 1D-графику, поэтому формат обязан быть round-trip: значение уходит в ядро через
// parse_num, и шесть значащих цифр (стояли здесь раньше) отрезали ~10 знаков — drill-down считался
// не в том параметре, в котором посчитана ячейка. См. fmt_num_shortest в num_parse.h.
std::string fmt_num_for_input(double v) {
    return fmt_num_shortest(v);
}

// Закреплённая ось (та, что держится фиксированной, пока свипуется другая)
// может идти по параметру, по начальному условию или по ШАГУ — и «зафиксировать»
// её значит подставить значение в РАЗНЫЕ поля конфига. Все три подстановки
// вызываются вместе и решают по одной и той же EffectiveSweep: срабатывает
// ровно одна.
//
// Флаги приходят одним объектом намеренно. Раньше каждая функция получала свои
// флаги отдельными аргументами, и pin_fixed_param проверяла только over_var —
// а при свипе по шагу over_var как раз false (см. draw_sweep_target_combo:
// выбор "dt (h)" ставит over_h и гасит over_var). В результате значение ШАГА
// уходило в параметр с индексом par_index — обычно первый в списке, — и в
// панели Parameters на глазах менялся посторонний параметр.
void pin_fixed_param(std::map<std::string,std::string>& pv,
                     const std::vector<std::string>& params,
                     const EffectiveSweep& e, double value) {
    if (e.over_var || e.over_h) return;
    if (e.par_index < 0 || e.par_index >= (int)params.size()) return;
    pv[params[e.par_index]] = fmt_num_for_input(value);
}

void pin_fixed_h(std::string& h_text, const EffectiveSweep& e, double value) {
    if (!e.over_h) return;
    if (value <= 0.0) return;             // шаг <= 0 бессмыслен, оставляем прежний
    h_text = fmt_num_for_input(value);
}

void pin_fixed_ic(std::map<std::string,std::string>& ic,
                  const std::vector<std::string>& vars,
                  const EffectiveSweep& e, double value) {
    if (!e.over_var || e.over_h) return;
    if (e.var_index < 0 || e.var_index >= (int)vars.size()) return;
    ic[vars[e.var_index]] = fmt_num_for_input(value);
}
} // namespace

bool AppModel::start_next_in_custom_queue() {
    if (custom_session.any_in_flight()) return false;
    if (custom_queue_.empty()) return false;
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();

    using K = CustomQueueItem::Kind;
    auto& cs = custom_session;
    const auto& shared = cs.shared;

    while (!custom_queue_.empty()) {
        CustomQueueItem it = custom_queue_.front();
        custom_queue_.pop_front();
        bool ok = false;

        switch (it.kind) {
        case K::Bif2D:
            if (cs.bif_session.diagrams.size() > 0) {
                apply_shared_to_bif2d(shared, cs.bif_session.diagrams[0]);
                ok = cs.bif_session.run_async(*parametric_engine, 0);
            }
            break;
        case K::LLE2D:
            if (cs.lle_session.curves.size() > 0) {
                apply_shared_to_lle2d(shared, cs.lle_session.curves[0]);
                ok = cs.lle_session.run_async(*parametric_engine, 0);
            }
            break;
        case K::LS2D:
            if (cs.ls_session.curves.size() > 0) {
                apply_shared_to_ls2d(shared, cs.ls_session.curves[0]);
                ok = cs.ls_session.run_async(*parametric_engine, 0);
            }
            break;

        case K::Bif1D_X:
            if (cs.bif_session.diagrams.size() > 1) {
                auto& c = cs.bif_session.diagrams[1];
                apply_shared_to_bif1d(shared, c, 0);
                // Pin the OTHER (Y) axis at fix_y.
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.bif_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::Bif1D_Y:
            if (cs.bif_session.diagrams.size() > 2) {
                auto& c = cs.bif_session.diagrams[2];
                apply_shared_to_bif1d(shared, c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.bif_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LLE1D_X:
            if (cs.lle_session.curves.size() > 1) {
                auto& c = cs.lle_session.curves[1];
                apply_shared_to_lle1d(shared, cs.lle_session.curves[0], c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.lle_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LLE1D_Y:
            if (cs.lle_session.curves.size() > 2) {
                auto& c = cs.lle_session.curves[2];
                apply_shared_to_lle1d(shared, cs.lle_session.curves[0], c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.lle_session.run_async(*parametric_engine, 2);
            }
            break;
        case K::LS1D_X:
            if (cs.ls_session.curves.size() > 1) {
                auto& c = cs.ls_session.curves[1];
                apply_shared_to_ls1d(shared, cs.ls_session.curves[0], c, 0);
                EffectiveSweep swy = effective_sweep_y(shared);
                pin_fixed_param(c.param_values, cs.params, swy, shared.fix_y_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swy, shared.fix_y_value);
                pin_fixed_h    (c.h_text, swy, shared.fix_y_value);
                ok = cs.ls_session.run_async(*parametric_engine, 1);
            }
            break;
        case K::LS1D_Y:
            if (cs.ls_session.curves.size() > 2) {
                auto& c = cs.ls_session.curves[2];
                apply_shared_to_ls1d(shared, cs.ls_session.curves[0], c, 1);
                EffectiveSweep swx = effective_sweep_x(shared);
                pin_fixed_param(c.param_values, cs.params, swx, shared.fix_x_value);
                pin_fixed_ic   (c.initial_conditions, cs.vars, swx, shared.fix_x_value);
                pin_fixed_h    (c.h_text, swx, shared.fix_x_value);
                ok = cs.ls_session.run_async(*parametric_engine, 2);
            }
            break;

        case K::Phase: {
            apply_shared_to_phase(shared, cs.phase_session, cs.vars);
            cs.phase_session.regenerate_krs();
            ok = cs.phase_session.recompute_async();
            break;
        }
        case K::Basins:
            if (cs.basins_session.configs.size() > 0) {
                auto& c = cs.basins_session.configs[0];
                apply_shared_to_basins(shared, c);
                ok = cs.basins_session.run_async(*parametric_engine, 0);
            }
            break;
        }

        if (ok) return true;
        // ok == false — skip and try the next item.
    }
    return false;
}

namespace {
// Помощник: убрать из очереди элементы с (kind == k && index == removed),
// для оставшихся того же kind: index > removed → --index.
void cleanup_queue_after_removal(std::deque<ParametricQueueItem>& q,
                                 ParametricQueueItem::Kind k, int removed) {
    for (auto it = q.begin(); it != q.end(); ) {
        if (it->kind == k && it->index == removed) it = q.erase(it);
        else ++it;
    }
    for (auto& it : q)
        if (it.kind == k && it.index > removed) --it.index;
}

// Помощник: у окон того же kind убрать `removed` из members, у оставшихся
// индексов > removed сделать --index. Окно, у которого members опустел
// (был последний diagram/curve, его удалили) — удаляется целиком, чтобы не
// висело пустое окно без данных.
void cleanup_plot_windows_after_removal(std::vector<ParametricPlotWindow>& wins,
                                        ParametricPlotWindow::Kind k, int removed) {
    for (auto& w : wins) {
        if (w.kind != k) continue;
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), removed), m.end());
        for (auto& idx : m) if (idx > removed) --idx;
    }
    wins.erase(std::remove_if(wins.begin(), wins.end(),
                              [k](const ParametricPlotWindow& w) {
                                  return w.kind == k && w.members.empty();
                              }),
               wins.end());
}
} // namespace


// Общий ход очередей вкладок с одной сессией: снять следующий элемент и запустить его.
// Элемент с устаревшим индексом или отказом run_async (last_error уже выставлен) пропускается.
template <class Session, class Item>
static bool run_next_queued(Session& s, std::deque<Item>& q, std::unique_ptr<ParametricEngine>& engine) {
    if (s.in_flight) return false;
    if (q.empty()) return false;
    if (!engine) engine = std::make_unique<ParametricEngine>();
    while (!q.empty()) {
        Item it = q.front();
        q.pop_front();
        if (it.index >= 0 && it.index < (int)s.configs.size()) {
            if (s.run_async(*engine, it.index)) return true;
        }
    }
    return false;
}

bool AppModel::start_next_in_dft1d_queue() {
    return run_next_queued(dft1d_session, dft1d_queue_, parametric_engine);
}

bool AppModel::start_next_in_basins_queue() {
    return run_next_queued(basins_session, basins_queue_, parametric_engine);
}

bool AppModel::start_next_in_order_queue() {
    return run_next_queued(order_session, order_queue_, parametric_engine);
}

bool AppModel::start_next_in_network_queue() {
    return run_next_queued(network_session, network_queue_, parametric_engine);
}

bool AppModel::start_next_in_neural_queue() {
    return run_next_queued(neural_session, neural_queue_, parametric_engine);
}

bool AppModel::start_next_in_fastsync_queue() {
    return run_next_queued(fastsync_session, fastsync_queue_, parametric_engine);
}

void AppModel::remove_bifurcation_diagram(int i) {
    bifurcation_session.remove_diagram(i);
    cleanup_queue_after_removal(parametric_queue_, ParametricQueueItem::Kind::Bifurcation, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::Bifurcation, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_lle_curve(int i) {
    lle_session.remove_curve(i);
    cleanup_queue_after_removal(parametric_queue_, ParametricQueueItem::Kind::LLE, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::LLE, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_metrics_config(int i) {
    metrics_session.remove_config(i);
    cleanup_queue_after_removal(parametric_queue_, ParametricQueueItem::Kind::Metrics, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::Metrics, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_ls_curve(int i) {
    ls_session.remove_curve(i);
    cleanup_queue_after_removal(parametric_queue_, ParametricQueueItem::Kind::LS, i);
    cleanup_plot_windows_after_removal(parametric_plot_windows, ParametricPlotWindow::Kind::LS, i);
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_basins_config(int i) {
    basins_session.remove_config(i);
    // Cleanup basins_queue_: drop items pointing at the removed index, shift
    // index > i down by one.
    for (auto it = basins_queue_.begin(); it != basins_queue_.end(); ) {
        if (it->index == i) it = basins_queue_.erase(it);
        else ++it;
    }
    for (auto& it : basins_queue_)
        if (it.index > i) --it.index;
}

void AppModel::remove_fastsync_config(int i) {
    fastsync_session.remove_config(i);
    // Cleanup fastsync_queue_: drop items pointing at the removed index, shift
    // index > i down by one.
    for (auto it = fastsync_queue_.begin(); it != fastsync_queue_.end(); ) {
        if (it->index == i) it = fastsync_queue_.erase(it);
        else ++it;
    }
    for (auto& it : fastsync_queue_)
        if (it.index > i) --it.index;
}

void AppModel::remove_order_config(int i) {
    order_session.remove_config(i);

    // Окна графиков: выкинуть i из members, сдвинуть большие индексы,
    // выбросить окна, оставшиеся вовсе без членов (как у DFT1D).
    for (auto& w : order_plot_windows) {
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), i), m.end());
        for (auto& idx : m) if (idx > i) --idx;
    }
    order_plot_windows.erase(std::remove_if(order_plot_windows.begin(), order_plot_windows.end(),
                                            [](const OrderPlotWindow& w) { return w.members.empty(); }),
                             order_plot_windows.end());
    order_plot_windows_dirty = true;
    // Та же чистка очереди, что у Basins/FastSync: выкидываем элементы,
    // указывающие на удалённый конфиг, и сдвигаем те, что были правее.
    for (auto it = order_queue_.begin(); it != order_queue_.end(); ) {
        if (it->index == i) it = order_queue_.erase(it);
        else ++it;
    }
    for (auto& it : order_queue_)
        if (it.index > i) --it.index;
}

void AppModel::remove_network_config(int i) {
    network_session.remove_config(i);
    // Та же чистка очереди, что у Order: выкидываем элементы, указывающие на
    // удалённый конфиг, и сдвигаем те, что были правее.
    for (auto it = network_queue_.begin(); it != network_queue_.end(); ) {
        if (it->index == i) it = network_queue_.erase(it);
        else ++it;
    }
    for (auto& it : network_queue_)
        if (it.index > i) --it.index;
}

void AppModel::remove_neural_config(int i) {
    neural_session.remove_config(i);
    for (auto it = neural_queue_.begin(); it != neural_queue_.end(); ) {
        if (it->index == i) it = neural_queue_.erase(it);
        else ++it;
    }
    for (auto& it : neural_queue_)
        if (it.index > i) --it.index;
}


// ---- Команды UI: запуск на движке, очереди ----

AppModel::AppModel(OcrFn ocr)
    : SystemEditor(std::move(ocr)), library_edit_buffer(std::make_unique<SystemEditor>(ocr_)) {}

AppModel::~AppModel() = default;

std::string TrajTimeEstimator::cost_key(const TrajWork& w) const {
    return std::to_string(std::hash<std::string>{}(w.krs_body)) + ":" + std::to_string(w.amountOfX)
         + (get_nvrtc_fmad() ? ":fm1" : ":fm0");
}

ParametricEngine& AppModel::ensure_engine() {
    if (!parametric_engine) parametric_engine = std::make_unique<ParametricEngine>();
    return *parametric_engine;
}

bool AppModel::run_bifurcation(int idx)  { return bifurcation_session.run_async(ensure_engine(), idx); }
bool AppModel::run_lle(int idx)          { return lle_session.run_async(ensure_engine(), idx); }
bool AppModel::run_ls(int idx)           { return ls_session.run_async(ensure_engine(), idx); }
bool AppModel::run_metrics(int idx)      { return metrics_session.run_async(ensure_engine(), idx); }
bool AppModel::run_dft1d(int idx)        { return dft1d_session.run_async(ensure_engine(), idx); }
bool AppModel::run_basins(int idx)       { return basins_session.run_async(ensure_engine(), idx); }
bool AppModel::recluster_basins(int idx) { return basins_session.run_recluster_async(ensure_engine(), idx); }
bool AppModel::run_fastsync(int idx)     { return fastsync_session.run_async(ensure_engine(), idx); }
bool AppModel::run_order(int idx)        { return order_session.run_async(ensure_engine(), idx); }
bool AppModel::run_network(int idx)      { return network_session.run_async(ensure_engine(), idx); }
bool AppModel::run_neural(int idx, bool resume) { return neural_session.run_async(ensure_engine(), idx, resume); }

void AppModel::reset_engine() {
    if (parametric_prewarm_future.valid()) parametric_prewarm_future.wait();
    parametric_engine.reset();
}

void AppModel::enqueue_parametric(ParametricQueueItem item) { parametric_queue_.push_back(item); }
void AppModel::enqueue_dft1d(int idx)    { dft1d_queue_.push_back({idx}); }
void AppModel::enqueue_basins(int idx)   { basins_queue_.push_back({idx}); }
void AppModel::enqueue_fastsync(int idx) { fastsync_queue_.push_back({idx}); }
void AppModel::enqueue_order(int idx)    { order_queue_.push_back({idx}); }
void AppModel::enqueue_network(int idx)  { network_queue_.push_back({idx}); }
void AppModel::enqueue_neural(int idx)   { neural_queue_.push_back({idx}); }
void AppModel::enqueue_custom(CustomQueueItem item) { custom_queue_.push_back(item); }

void AppModel::enqueue_custom_level_2d() { custom_session.enqueue_level_2d(custom_queue_); }
void AppModel::enqueue_custom_level_1d() { custom_session.enqueue_level_1d(custom_queue_); }
void AppModel::enqueue_custom_level_3()  { custom_session.enqueue_level_3(custom_queue_); }
void AppModel::enqueue_custom_level_1d_partial(bool x_slices, bool y_slices) {
    custom_session.enqueue_level_1d_partial(custom_queue_, x_slices, y_slices);
}

void AppModel::clear_neural_queue() { neural_queue_.clear(); }
void AppModel::clear_custom_queue() { custom_queue_.clear(); }

void AppModel::clear_all_queues() {
    parametric_queue_.clear();
    dft1d_queue_.clear();
    basins_queue_.clear();
    fastsync_queue_.clear();
    order_queue_.clear();
    network_queue_.clear();
    neural_queue_.clear();
    custom_queue_.clear();
}
