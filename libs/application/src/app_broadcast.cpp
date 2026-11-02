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


namespace {

// Цель свипа одной оси. Пишется и читается ЦЕЛИКОМ: половина применённой цели
// (скажем, par_index без снятого over_h) — это не цель, а мусор. Поэтому по
// общему строковому интерфейсу рассылки она едет одной строкой, см.
// encode_sweep / decode_sweep.
struct SweepTargetSlot {
    int*  par_index = nullptr;
    bool* over_var  = nullptr;
    int*  var_index = nullptr;
    bool* over_h    = nullptr;
    // Списки имён ЭТОЙ вкладки — для проверки границ при разборе: у сессии,
    // собранной до правки алфавита, индексов может не хватать.
    const std::vector<std::string>* params = nullptr;
    const std::vector<std::string>* vars   = nullptr;
    [[nodiscard]] bool valid() const { return par_index && over_var && var_index && over_h; }
};

// Куда broadcast_field может писать у одного конфига. nullptr = поля у этого
// конфига нет.
struct BroadcastSlots {
    std::string* h           = nullptr;
    std::string* symmetry    = nullptr;
    std::string* cd_fp_iters = nullptr;
    std::string* t_max       = nullptr;
    std::string* transient   = nullptr;
    std::string* pre_scaller = nullptr;
    std::string* max_value   = nullptr;
    std::map<std::string, std::string>* params = nullptr;
    std::map<std::string, std::string>* ics    = nullptr;
    std::string* scheme     = nullptr;
    std::string* sweep_lo   = nullptr;
    std::string* sweep_hi   = nullptr;
    std::string* sweep_lo_2 = nullptr;
    std::string* sweep_hi_2 = nullptr;
    std::string* resolution = nullptr;
    SweepTargetSlot sweep;     // первая ось свипа
    SweepTargetSlot sweep_2;   // вторая (2D-режим)
    // Подпись конфига (nullptr — у вкладки её нет: Phase и Shared config
    // Custom'а одни на вкладку). Отмена сверяет её, чтобы не записать в чужой
    // конфиг, вставший на освободившийся индекс.
    const std::string* label = nullptr;
};

// Цель свипа <-> строка. "par:2" — параметр с индексом 2, "var:0" — начальное
// условие, "h" — шаг интегрирования, "s" — коэффициент симметрии a[0]
// (par_index < 0). Лог-шкала сюда НЕ входит: это отдельный чекбокс рядом с
// комбо, и ПКМ по комбо про него ничего не обещает.
std::string encode_sweep(const SweepTargetSlot& s) {
    if (!s.valid()) return {};
    return encode_sweep_target(*s.par_index, *s.over_var, *s.var_index, *s.over_h);
}

// false = строка не разобралась или индекс за пределами списков ЭТОЙ вкладки;
// тогда поля не трогаются вовсе.
bool decode_sweep(const SweepTargetSlot& s, const std::string& v) {
    if (!s.valid() || v.empty()) return false;
    auto set = [&](int par, bool ov, int vi, bool oh) {
        *s.par_index = par; *s.over_var = ov; *s.var_index = vi; *s.over_h = oh;
        return true;
    };
    if (v == "h") return set(*s.par_index, false, *s.var_index, true);
    if (v == "s") return set(-1, false, *s.var_index, false);
    const bool is_var = (v.compare(0, 4, "var:") == 0);
    const bool is_par = (v.compare(0, 4, "par:") == 0);
    if (!is_var && !is_par) return false;
    int idx = 0;
    try { idx = std::stoi(v.substr(4)); } catch (...) { return false; }
    if (idx < 0) return false;
    const std::vector<std::string>* list = is_var ? s.vars : s.params;
    if (!list || idx >= (int)list->size()) return false;   // алфавит вкладки короче
    return is_var ? set(*s.par_index, true, idx, false)
                  : set(idx, false, *s.var_index, false);
}

constexpr BroadcastTab kBroadcastTabs[] = {
    BroadcastTab::Phase,  BroadcastTab::Bifurcation, BroadcastTab::LLE,
    BroadcastTab::LS,     BroadcastTab::Dft1D,       BroadcastTab::Basins,
    BroadcastTab::FastSync, BroadcastTab::Custom,    BroadcastTab::Order,
    BroadcastTab::Metrics,
};

// Элемент вектора конфигов или nullptr (пустая сессия, индекс из битого
// сохранения или из записи отмены, пережившей закрытие вкладки).
template <class T>
T* config_at(std::vector<T>& v, int idx) {
    return (idx >= 0 && idx < (int)v.size()) ? &v[(size_t)idx] : nullptr;
}

// Слоты конфига вкладки. idx < 0 — взять активный конфиг (обычный путь
// broadcast'а); idx >= 0 — именно этот (путь отмены). В resolved_idx уходит
// фактический индекс, чтобы запись отмены знала, куда возвращаться.
// false = конфига нет.
bool slots_for(AppModel& m, BroadcastTab tab, int idx,
               BroadcastSlots& out, int& resolved_idx) {
    out = BroadcastSlots{};
    switch (tab) {
    case BroadcastTab::Phase: {
        // Имена полей у Phase исторически свои (step_h / sim_time / skip_time),
        // а НУ лежат списком наборов: несколько траекторий в одних осях — весь
        // смысл вкладки, поэтому трогаем только ПЕРВЫЙ набор, тот, которому
        // соответствует единственное НУ остальных вкладок.
        resolved_idx = 0;
        out.h         = &m.phase_session.step_h;
        out.symmetry  = &m.phase_session.symmetry_s;
        out.cd_fp_iters = &m.phase_session.cd_fp_iters;
        out.t_max     = &m.phase_session.sim_time;
        out.transient = &m.phase_session.skip_time;
        out.params    = &m.phase_session.param_values;
        out.scheme    = &m.phase_session.scheme;
        if (!m.phase_session.ic_sets.empty())
            out.ics = &m.phase_session.ic_sets[0].values;
        return true;
    }
    case BroadcastTab::Bifurcation: {
        auto& v = m.bifurcation_session.diagrams;
        resolved_idx = (idx < 0) ? m.bifurcation_session.active_diagram_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.bifurcation_session.params, &m.bifurcation_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.bifurcation_session.params, &m.bifurcation_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::LLE: {
        auto& v = m.lle_session.curves;
        resolved_idx = (idx < 0) ? m.lle_session.active_curve_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.lle_session.params, &m.lle_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.lle_session.params, &m.lle_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::LS: {
        auto& v = m.ls_session.curves;
        resolved_idx = (idx < 0) ? m.ls_session.active_curve_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.ls_session.params, &m.ls_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.ls_session.params, &m.ls_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Dft1D: {
        auto& v = m.dft1d_session.configs;
        resolved_idx = (idx < 0) ? m.dft1d_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text; out.sweep_hi = &c->param_hi_text;
        out.resolution = &c->n_pts_text;   // Resolution X; по Y у DFT частоты
        out.sweep = { &c->param_index, &c->sweep_over_var, &c->var_sweep_index,
                      &c->sweep_over_h, &m.dft1d_session.params, &m.dft1d_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Basins: {
        auto& v = m.basins_session.configs;
        resolved_idx = (idx < 0) ? m.basins_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        // Свипа параметра у Basins нет: обе оси — сетка начальных условий.
        // n_pts_text при этом СТОРОНА этой сетки, и рассылка Resolution с
        // 1D-вкладки поднимет её в квадрате — счётчик "(N differ)" и Ctrl+Z
        // на этот случай и нужны.
        out.resolution = &c->n_pts_text;
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::FastSync: {
        // НУ у FastSync нет: обе оси сетки — это и есть начальные условия.
        auto& v = m.fastsync_session.configs;
        resolved_idx = (idx < 0) ? m.fastsync_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values;
        out.scheme = &c->scheme;
        out.resolution = &c->n_pts_text;   // сторона сетки, см. Basins
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Custom: {
        // Писать надо в Shared config, а не в подсессии — те всё равно
        // пересобираются из него срезами apply_shared_to_* на каждом Run.
        resolved_idx = 0;
        auto& c = m.custom_session.shared;
        out.h = &c.h_text; out.symmetry = &c.symmetry_s; out.cd_fp_iters = &c.cd_fp_iters;
        out.t_max = &c.t_max_text; out.transient = &c.transient_text;
        out.pre_scaller = &c.pre_scaller_text; out.max_value = &c.max_value_text;
        out.params = &c.param_values; out.ics = &c.initial_conditions;
        out.scheme = &c.scheme;
        // Свип вкладки — оси ОБЩЕГО config'а (уровень 2D). Подсессии
        // пересобираются из него срезами apply_shared_to_* на каждом Run.
        out.sweep_lo = &c.axis_x_lo_text;   out.sweep_hi = &c.axis_x_hi_text;
        out.sweep_lo_2 = &c.axis_y_lo_text; out.sweep_hi_2 = &c.axis_y_hi_text;
        out.resolution = &c.resolution_text;
        out.sweep   = { &c.axis_x_par_index, &c.axis_x_over_var, &c.axis_x_var_index,
                        &c.axis_x_over_h, &m.custom_session.params, &m.custom_session.vars };
        out.sweep_2 = { &c.axis_y_par_index, &c.axis_y_over_var, &c.axis_y_var_index,
                        &c.axis_y_over_h, &m.custom_session.params, &m.custom_session.vars };
        return true;
    }
    case BroadcastTab::Metrics: {
        auto& v = m.metrics_session.configs;
        resolved_idx = (idx < 0) ? m.metrics_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.transient = &c->transient_text;
        out.pre_scaller = &c->pre_scaller_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        out.sweep_lo = &c->param_lo_text;     out.sweep_hi = &c->param_hi_text;
        out.sweep_lo_2 = &c->param_lo_2_text; out.sweep_hi_2 = &c->param_hi_2_text;
        out.resolution = &c->n_pts_text;
        out.sweep   = { &c->param_index,   &c->sweep_over_var,   &c->var_sweep_index,
                        &c->sweep_over_h,   &m.metrics_session.params, &m.metrics_session.vars };
        out.sweep_2 = { &c->param_index_2, &c->sweep_over_var_2, &c->var_sweep_index_2,
                        &c->sweep_over_h_2, &m.metrics_session.params, &m.metrics_session.vars };
        out.label = &c->label;
        return true;
    }
    case BroadcastTab::Order: {
        auto& v = m.order_session.configs;
        resolved_idx = (idx < 0) ? m.order_session.active_config_index : idx;
        auto* c = config_at(v, resolved_idx);
        if (!c) return false;
        out.h = &c->h_text; out.symmetry = &c->symmetry_s; out.cd_fp_iters = &c->cd_fp_iters;
        out.t_max = &c->t_max_text; out.max_value = &c->max_value_text;
        out.params = &c->param_values; out.ics = &c->initial_conditions;
        out.scheme = &c->scheme;
        // Диапазоны и цель осей у Order своей кодировки (axis_x_target,
        // kOrderTargetH) и с параметрическим свипом не совпадают — пропускаем.
        out.label = &c->label;
        return true;
    }
    }
    return false;
}

// Строковое поле слотов по BroadcastField. nullptr — либо поле у вкладки
// отсутствует, либо это одна из карт (Param / InitCondition), их разбирает
// caller.
std::string* string_slot(const BroadcastSlots& s, BroadcastField f) {
    switch (f) {
        case BroadcastField::StepH:      return s.h;
        case BroadcastField::SymmetryS:  return s.symmetry;
        case BroadcastField::CdFpIters:  return s.cd_fp_iters;
        case BroadcastField::TMax:       return s.t_max;
        case BroadcastField::Transient:  return s.transient;
        case BroadcastField::PreScaller: return s.pre_scaller;
        case BroadcastField::MaxValue:   return s.max_value;
        case BroadcastField::Scheme:     return s.scheme;
        case BroadcastField::SweepLo:    return s.sweep_lo;
        case BroadcastField::SweepHi:    return s.sweep_hi;
        case BroadcastField::SweepLo2:   return s.sweep_lo_2;
        case BroadcastField::SweepHi2:   return s.sweep_hi_2;
        case BroadcastField::Resolution: return s.resolution;
        default:                         return nullptr;
    }
}

// Цель свипа по BroadcastField. valid() == false — у вкладки такой оси нет.
const SweepTargetSlot* sweep_slot(const BroadcastSlots& s, BroadcastField f) {
    if (f == BroadcastField::SweepTarget)  return &s.sweep;
    if (f == BroadcastField::SweepTarget2) return &s.sweep_2;
    return nullptr;
}

// Сколько конфигов у вкладки. У Phase и Custom он один на вкладку.
int config_count(AppModel& m, BroadcastTab tab) {
    switch (tab) {
        case BroadcastTab::Phase:       return 1;
        case BroadcastTab::Bifurcation: return (int)m.bifurcation_session.diagrams.size();
        case BroadcastTab::LLE:         return (int)m.lle_session.curves.size();
        case BroadcastTab::LS:          return (int)m.ls_session.curves.size();
        case BroadcastTab::Dft1D:       return (int)m.dft1d_session.configs.size();
        case BroadcastTab::Basins:      return (int)m.basins_session.configs.size();
        case BroadcastTab::FastSync:    return (int)m.fastsync_session.configs.size();
        case BroadcastTab::Custom:      return 1;
        case BroadcastTab::Order:       return (int)m.order_session.configs.size();
        case BroadcastTab::Metrics:     return (int)m.metrics_session.configs.size();
    }
    return 0;
}

// Тип конфига — то, по чему работает "применить ко всем такого же типа".
// 1D и 2D разведены сознательно: это разные диаграммы с разным смыслом осей.
std::string group_of(AppModel& m, BroadcastTab tab, int idx) {
    switch (tab) {
        case BroadcastTab::Phase: return "Phase analysis";
        case BroadcastTab::Bifurcation: {
            auto* c = config_at(m.bifurcation_session.diagrams, idx);
            return (c && c->mode_2d) ? "Bifurcation 2D" : "Bifurcation 1D";
        }
        case BroadcastTab::LLE: {
            auto* c = config_at(m.lle_session.curves, idx);
            return (c && c->mode_2d) ? "LLE 2D" : "LLE 1D";
        }
        case BroadcastTab::LS: {
            auto* c = config_at(m.ls_session.curves, idx);
            return (c && c->mode_2d) ? "LS 2D" : "LS 1D";
        }
        case BroadcastTab::Dft1D:    return "1D DFT";
        case BroadcastTab::Basins:   return "Basins";
        case BroadcastTab::FastSync: return "Fast Synchro";
        case BroadcastTab::Custom:   return "Custom";
        case BroadcastTab::Order:    return "Order";
        case BroadcastTab::Metrics: {
            auto* c = config_at(m.metrics_session.configs, idx);
            return (c && c->mode_2d) ? "Metrics 2D" : "Metrics 1D";
        }
    }
    return {};
}

std::map<std::string, std::string>* map_slot(const BroadcastSlots& s, BroadcastField f) {
    if (f == BroadcastField::Param)         return s.params;
    if (f == BroadcastField::InitCondition) return s.ics;
    return nullptr;
}

} // namespace

std::string encode_sweep_target(int par_index, bool over_var, int var_index, bool over_h) {
    if (over_h)        return "h";
    if (over_var)      return "var:" + std::to_string(var_index);
    if (par_index < 0) return "s";
    return "par:" + std::to_string(par_index);
}

void AppModel::push_undo(std::string description, std::function<void()> undo,
                         std::function<void()> redo) {
    if (!undo) return;
    auto& st = undo_stacks[app_mode];
    st.push_back({ std::move(description), std::move(undo), std::move(redo) });
    // Запись держит прежние значения полей, а не всю сессию, поэтому 200 штук
    // на вкладку стоят копейки.
    while (st.size() > kUndoDepth) st.pop_front();
    undo_just_happened_tabs[app_mode] = false;
    // Новая команда обрывает ветку повтора — как в любом редакторе: вернуть
    // отменённое после того, как поверх сделали другое, уже некуда.
    redo_stacks[app_mode].clear();
}

namespace {
// Снять верхнюю запись стека этой вкладки, применить её сторону (undo/redo) и
// переложить в противоположный стек. Обе команды отличаются только этим.
bool pop_apply_move(std::map<AppModel::AppMode, std::deque<UndoRecord>>& from,
                    std::map<AppModel::AppMode, std::deque<UndoRecord>>& to,
                    AppModel::AppMode tab, bool forward,
                    const char* verb, std::string& note) {
    auto it = from.find(tab);
    if (it == from.end() || it->second.empty()) return false;
    UndoRecord r = std::move(it->second.back());
    it->second.pop_back();
    const std::function<void()>& act = forward ? r.redo : r.undo;
    if (act) act();
    note = std::string(verb) + ": " + r.description;
    auto& dst = to[tab];
    dst.push_back(std::move(r));
    while (dst.size() > AppModel::kUndoDepth) dst.pop_front();
    return true;
}
} // namespace

bool AppModel::undo_last() {
    const bool ok = pop_apply_move(undo_stacks, redo_stacks, app_mode, /*forward*/false,
                                   "Undone", undo_note);
    if (ok) undo_just_happened_tabs[app_mode] = true;
    return ok;
}

bool AppModel::redo_last() {
    const bool ok = pop_apply_move(redo_stacks, undo_stacks, app_mode, /*forward*/true,
                                   "Redone", undo_note);
    if (ok) undo_just_happened_tabs[app_mode] = true;
    return ok;
}

bool AppModel::undo_just_happened() const {
    auto it = undo_just_happened_tabs.find(app_mode);
    return it != undo_just_happened_tabs.end() && it->second;
}

const std::string* AppModel::undo_top() const {
    auto it = undo_stacks.find(app_mode);
    return (it == undo_stacks.end() || it->second.empty())
         ? nullptr : &it->second.back().description;
}

const std::string* AppModel::redo_top() const {
    auto it = redo_stacks.find(app_mode);
    // Повторять нечего, если у записи нет обратного хода.
    if (it == redo_stacks.end() || it->second.empty()) return nullptr;
    return it->second.back().redo ? &it->second.back().description : nullptr;
}

// Есть ли у этого конфига такое поле вообще. Нужно перечислению целей: вкладка
// без поля не должна появляться в списке галочек и обещать применение.
static bool has_field(const BroadcastSlots& s, BroadcastField f) {
    if (map_slot(s, f)) return true;
    if (const SweepTargetSlot* sw = sweep_slot(s, f)) return sw->valid();
    return string_slot(s, f) != nullptr;
}

std::vector<BroadcastTarget> AppModel::broadcast_targets(BroadcastField f) {
    std::vector<BroadcastTarget> out;
    for (BroadcastTab tab : kBroadcastTabs) {
        const int n = config_count(*this, tab);
        for (int i = 0; i < n; ++i) {
            BroadcastSlots slots;
            int idx = -1;
            if (!slots_for(*this, tab, i, slots, idx)) continue;
            if (!has_field(slots, f)) continue;
            BroadcastTarget t;
            t.tab   = tab;
            t.idx   = idx;
            t.group = group_of(*this, tab, idx);
            // У Phase и Custom конфиг один и подписи не имеет — тогда именем
            // служит сам тип, иначе в списке галочек была бы пустая строка.
            t.label = slots.label ? *slots.label : t.group;
            out.push_back(std::move(t));
        }
    }
    return out;
}

int AppModel::broadcast_field(BroadcastField f, const std::string& name,
                              const std::string& value, bool apply,
                              const std::string& label,
                              const std::vector<BroadcastTarget>* only,
                              const std::string& scope_note) {
    // Прежнее состояние одного поля — для стека отмены. Адрес не храним:
    // вкладку с конфигом могут закрыть, и указатель повиснет. Храним, где
    // искать (вкладка + индекс + label) и что вернуть.
    struct Prev {
        BroadcastTab tab;
        int          idx;
        std::string  label;    // пусто = у вкладки конфигов нет (Phase, Custom)
        bool         existed;  // для карт: был ли ключ. Нет — откат его удалит
        std::string  value;
    };
    std::vector<Prev> prev;
    int n = 0;

    // Одна цель. req_idx < 0 — активный конфиг вкладки (обход "по всем
    // вкладкам"); иначе именно этот конфиг (адресный список из меню).
    auto touch = [&](BroadcastTab tab, int req_idx) {
        BroadcastSlots slots;
        int idx = -1;
        if (!slots_for(*this, tab, req_idx, slots, idx)) return;
        const std::string cfg_label = slots.label ? *slots.label : std::string();

        if (auto* m = map_slot(slots, f)) {
            auto it = m->find(name);
            if (it != m->end()) {
                if (it->second == value) return;
                ++n;
                if (apply) {
                    prev.push_back({ tab, idx, cfg_label, true, it->second });
                    it->second = value;
                }
            } else {
                // Ключа нет — конфиг собран до того, как параметр появился в
                // системе. Это расхождение; заводим ключ только когда реально
                // применяем, иначе сухой прогон менял бы состояние.
                ++n;
                if (apply) {
                    prev.push_back({ tab, idx, cfg_label, false, std::string() });
                    (*m)[name] = value;
                }
            }
            return;
        }

        if (const SweepTargetSlot* sw = sweep_slot(slots, f)) {
            if (!sw->valid()) return;
            const std::string cur = encode_sweep(*sw);
            if (cur == value) return;
            // Проверяем разбор ДО подсчёта: если индекс за пределами алфавита
            // этой вкладки, применить нечего, и показывать её в "(N differ)"
            // значит обещать то, чего не будет.
            SweepTargetSlot probe = *sw;
            int  p_par = *sw->par_index, p_vi = *sw->var_index;
            bool p_ov  = *sw->over_var,  p_oh = *sw->over_h;
            probe.par_index = &p_par; probe.over_var = &p_ov;
            probe.var_index = &p_vi;  probe.over_h   = &p_oh;
            if (!decode_sweep(probe, value)) return;
            ++n;
            if (apply) {
                prev.push_back({ tab, idx, cfg_label, true, cur });
                decode_sweep(*sw, value);
            }
            return;
        }

        std::string* dst = string_slot(slots, f);
        if (!dst || *dst == value) return;
        ++n;
        if (apply) {
            prev.push_back({ tab, idx, cfg_label, true, *dst });
            *dst = value;
        }
    };

    if (only) { for (const auto& t : *only) touch(t.tab, t.idx); }
    else      { for (BroadcastTab tab : kBroadcastTabs) touch(tab, -1); }

    if (apply && !prev.empty()) {
        // Одна запись на команду, а не по одной на конфиг: пользователь нажал
        // один пункт меню и ждёт, что один Ctrl+Z его отменит.
        const std::string& shown = label.empty() ? name : label;
        const std::string what = (shown.empty() ? ("value " + value) : (shown + " = " + value))
                               + " -> " + scope_note;

        // Отмена и повтор — одна и та же процедура, отличаются только списком
        // значений: назад кладём снятое, вперёд — то, что команда написала.
        auto writer = [this, f, name](std::vector<Prev> snaps) {
            return [this, f, name, snaps = std::move(snaps)]() {
                for (const auto& p : snaps) {
                    BroadcastSlots slots;
                    int idx = -1;
                    if (!slots_for(*this, p.tab, p.idx, slots, idx)) continue;
                    // На записанном индексе мог оказаться другой конфиг
                    // (вкладку закрыли, соседние сдвинулись) — тогда
                    // пропускаем, а не переписываем чужое.
                    if (!p.label.empty() && (!slots.label || *slots.label != p.label)) continue;
                    if (auto* m = map_slot(slots, f)) {
                        if (p.existed) (*m)[name] = p.value;
                        else           m->erase(name);
                    } else if (const SweepTargetSlot* sw = sweep_slot(slots, f)) {
                        decode_sweep(*sw, p.value);
                    } else if (std::string* dst = string_slot(slots, f)) {
                        *dst = p.value;
                    }
                }
            };
        };
        std::vector<Prev> after;
        after.reserve(prev.size());
        for (const auto& p : prev) after.push_back({ p.tab, p.idx, p.label, true, value });
        push_undo(what, writer(std::move(prev)), writer(std::move(after)));
    }
    return n;
}
