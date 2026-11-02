#include "ucuda/application/app_model.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/domain/num_parse.h"   // fmt_num_shortest — round-trip формат для пинов узлов
#include "ucuda/application/session_persistence.h"
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


void AppModel::add_parametric_plot_window(ParametricPlotWindow::Kind kind, bool mode_2d,
                                          std::vector<int> initial_members,
                                          bool colored_1d) {
    ParametricPlotWindow w;
    w.kind = kind;
    w.mode_2d = mode_2d;
    w.colored_1d = colored_1d;
    w.members = std::move(initial_members);
    // A 2D/Colored-1D window shows exactly one heatmap — enforce single-member
    // here too, not just in the UI, since this is the one place all windows
    // go through.
    if ((w.mode_2d || w.colored_1d) && w.members.size() > 1) w.members.resize(1);
    w.id = next_parametric_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;   // fresh window → auto-label
    parametric_plot_windows.push_back(std::move(w));
    parametric_plot_windows_dirty = true;
}

void AppModel::remove_parametric_plot_window(int pos) {
    if (pos < 0 || pos >= (int)parametric_plot_windows.size()) return;
    parametric_plot_windows.erase(parametric_plot_windows.begin() + pos);
    parametric_plot_windows_dirty = true;
}

void AppModel::load_or_init_parametric_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_parametric_windows(json, parametric_plot_windows);
        // Bump next-id past the highest loaded id so subsequent Add window
        // calls don't collide with existing ones (which would give two rows
        // the same PushID and trigger ImGui's "conflicting ID" warning).
        int max_id = 0;
        for (const auto& w : parametric_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_parametric_plot_window_id)
            next_parametric_plot_window_id = max_id + 1;
        return;
    }
    parametric_plot_windows.clear();

    // 1D: one window overlaying every 1D diagram/curve of a kind (reproduces
    // the old "everything on one shared plot" behavior). 2D: one window PER
    // diagram/curve, since a 2D window can only ever show a single heatmap.
    std::vector<int> bd1;
    for (size_t i = 0; i < bifurcation_session.diagrams.size(); ++i) {
        const auto& bd = bifurcation_session.diagrams[i];
        if (bd.mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, true, { (int)i });
        else if (bd.colored_1d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, { (int)i }, true);
        else
            bd1.push_back((int)i);
    }
    if (!bd1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, bd1);

    std::vector<int> lle1;
    for (size_t i = 0; i < lle_session.curves.size(); ++i) {
        if (lle_session.curves[i].mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::LLE, true, { (int)i });
        else
            lle1.push_back((int)i);
    }
    if (!lle1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::LLE, false, lle1);

    std::vector<int> ls1;
    for (size_t i = 0; i < ls_session.curves.size(); ++i) {
        if (ls_session.curves[i].mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::LS, true, { (int)i });
        else
            ls1.push_back((int)i);
    }
    if (!ls1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::LS, false, ls1);

    std::vector<int> met1;
    for (size_t i = 0; i < metrics_session.configs.size(); ++i) {
        if (metrics_session.configs[i].mode_2d)
            add_parametric_plot_window(ParametricPlotWindow::Kind::Metrics, true, { (int)i });
        else
            met1.push_back((int)i);
    }
    if (!met1.empty()) add_parametric_plot_window(ParametricPlotWindow::Kind::Metrics, false, met1);
}

void AppModel::remove_dft1d_config(int i) {
    dft1d_session.remove_config(i);
    // Cleanup dft1d_queue: drop items pointing at the removed index, shift
    // index > i down by one (same pattern as remove_basins_config).
    for (auto it = dft1d_queue.begin(); it != dft1d_queue.end(); ) {
        if (it->index == i) it = dft1d_queue.erase(it);
        else ++it;
    }
    for (auto& it : dft1d_queue)
        if (it.index > i) --it.index;

    // Plot windows: drop `i` from members, shift indices > i down by one;
    // drop windows left with no members (same pattern as
    // cleanup_plot_windows_after_removal, but DFT1D windows have no `kind`).
    for (auto& w : dft1d_plot_windows) {
        auto& m = w.members;
        m.erase(std::remove(m.begin(), m.end(), i), m.end());
        for (auto& idx : m) if (idx > i) --idx;
    }
    dft1d_plot_windows.erase(std::remove_if(dft1d_plot_windows.begin(), dft1d_plot_windows.end(),
                                            [](const Dft1DPlotWindow& w) { return w.members.empty(); }),
                             dft1d_plot_windows.end());
    dft1d_plot_windows_dirty = true;
}

void AppModel::add_dft1d_plot_window(std::vector<int> initial_members) {
    Dft1DPlotWindow w;
    w.members = std::move(initial_members);
    // A DFT1D window always shows exactly one config's heatmap (unlike
    // Parametric's classic-1D windows, which can overlay several) — enforce
    // here too, not just in the "Members..." popup's radio buttons.
    if (w.members.size() > 1) w.members.resize(1);
    w.id = next_dft1d_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;   // fresh window → auto-label
    dft1d_plot_windows.push_back(std::move(w));
    dft1d_plot_windows_dirty = true;
}

void AppModel::remove_dft1d_plot_window(int pos) {
    if (pos < 0 || pos >= (int)dft1d_plot_windows.size()) return;
    dft1d_plot_windows.erase(dft1d_plot_windows.begin() + pos);
    dft1d_plot_windows_dirty = true;
}

void AppModel::load_or_init_dft1d_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_dft1d_windows(json, dft1d_plot_windows);
        int max_id = 0;
        for (const auto& w : dft1d_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_dft1d_plot_window_id)
            next_dft1d_plot_window_id = max_id + 1;
        return;
    }
    dft1d_plot_windows.clear();
    // One window per config, same as Bifurcation's 2D/colored-1D default —
    // a DFT heatmap isn't overlay-friendly, so there is no shared-window case.
    for (size_t i = 0; i < dft1d_session.configs.size(); ++i)
        add_dft1d_plot_window({ (int)i });
}

void AppModel::add_order_plot_window(OrderPlotWindow::Kind kind, std::vector<int> initial_members) {
    OrderPlotWindow w;
    w.kind    = kind;
    w.members = std::move(initial_members);
    // Карта — хитмапа: наложить две друг на друга нельзя, поэтому лишние
    // члены отсекаются здесь, а не только радиокнопками в попапе.
    if (kind == OrderPlotWindow::Kind::Map && w.members.size() > 1) w.members.resize(1);
    w.id = next_order_plot_window_id++;
    w.label = "Plot " + std::to_string(w.id);
    w.label_is_manual = false;
    // Порядок p линеен по построению — логарифмировать его нечего.
    if (kind == OrderPlotWindow::Kind::P) w.y_log = false;
    order_plot_windows.push_back(std::move(w));
    order_plot_windows_dirty = true;
}

void AppModel::remove_order_plot_window(int pos) {
    if (pos < 0 || pos >= (int)order_plot_windows.size()) return;
    order_plot_windows.erase(order_plot_windows.begin() + pos);
    order_plot_windows_dirty = true;
}

void AppModel::load_or_init_order_plot_windows(const std::string& json) {
    if (!json.empty()) {
        session_from_json_order_windows(json, order_plot_windows);
        int max_id = 0;
        for (const auto& w : order_plot_windows)
            if (w.id > max_id) max_id = w.id;
        if (max_id >= next_order_plot_window_id)
            next_order_plot_window_id = max_id + 1;

        // Сверка с конфигами. Прежние сборки писали окна и конфиги в разные
        // моменты (окна — по правке, конфиги — только после Run), и файлы
        // расходились: окно ссылалось на вкладку, которой в сохранённых
        // конфигах нет. Ссылки за пределы — выбросить (окно, оставшееся от
        // этого пустым, тоже), дубликаты — схлопнуть, у карты — один член.
        // Вкладку, не лежащую ни в одном окне, НЕ возвращаем: её могли убрать
        // из окон намеренно, через Members.
        const int nc = (int)order_session.configs.size();
        for (auto it = order_plot_windows.begin(); it != order_plot_windows.end(); ) {
            auto& m = it->members;
            const bool had = !m.empty();
            std::vector<int> kept;
            for (int idx : m)
                if (idx >= 0 && idx < nc && std::find(kept.begin(), kept.end(), idx) == kept.end())
                    kept.push_back(idx);
            if (it->kind == OrderPlotWindow::Kind::Map && kept.size() > 1) kept.resize(1);
            m = std::move(kept);
            if (had && m.empty()) { it = order_plot_windows.erase(it); continue; }
            ++it;
        }
        if (!order_plot_windows.empty()) return;
    }
    order_plot_windows.clear();
    next_order_plot_window_id = 1;
    // Дефолт — одно окно с кривой p, в котором лежат все сеточные конфиги:
    // именно так вкладка и выглядела до появления окон.
    std::vector<int> curves, maps, perfs;
    for (size_t i = 0; i < order_session.configs.size(); ++i) {
        const OrderConfig& c = order_session.configs[i];
        if (c.calc_kind == kOrderCalcPerf)      perfs.push_back((int)i);
        // Область устойчивости — всегда карта, своего вида окна у неё нет:
        // в окне Map лежит ровно один член, что хитмапе и нужно.
        else if (c.calc_kind == kOrderCalcStab) maps.push_back((int)i);
        else if (c.two_d)                       maps.push_back((int)i);
        else                                    curves.push_back((int)i);
    }
    if (!curves.empty()) add_order_plot_window(OrderPlotWindow::Kind::P, curves);
    if (!perfs.empty())  add_order_plot_window(OrderPlotWindow::Kind::Perf, perfs);
    for (int m : maps)   add_order_plot_window(OrderPlotWindow::Kind::Map, { m });
    if (order_plot_windows.empty())
        add_order_plot_window(OrderPlotWindow::Kind::P, {});
}
