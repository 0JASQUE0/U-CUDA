#include "ucuda/analysis/tree_session.h"
#include "ucuda/analysis/custom_session.h"
#include "ucuda/domain/num_parse.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>

// --- TreeLeafRuntime ---

bool TreeLeafRuntime::in_flight() const {
    return (bif && bif->in_flight) || (lle && lle->in_flight) || (ls && ls->in_flight)
        || (phase && phase->in_flight) || (basins && basins->in_flight);
}

bool TreeLeafRuntime::last_ok() const {
    if (bif && !bif->diagrams.empty())
        return bif->diagrams[0].mode_2d ? bif->diagrams[0].last_run_2d_ok : bif->diagrams[0].last_run_ok;
    if (lle && !lle->curves.empty())
        return lle->curves[0].mode_2d ? lle->curves[0].last_run_2d_ok : lle->curves[0].last_run_ok;
    if (ls && !ls->curves.empty())
        return ls->curves[0].mode_2d ? ls->curves[0].last_run_2d_ok : ls->curves[0].last_run_ok;
    if (phase)  return phase->result.ok;
    if (basins && !basins->configs.empty()) return basins->configs[0].last_run_ok;
    return false;
}

std::string TreeLeafRuntime::last_error() const {
    if (bif && !bif->diagrams.empty()) {
        const auto& c = bif->diagrams[0];
        return c.mode_2d ? c.result_2d.error : c.last_error;
    }
    if (lle && !lle->curves.empty()) {
        const auto& c = lle->curves[0];
        return c.mode_2d ? c.result_2d.error : c.last_error;
    }
    if (ls && !ls->curves.empty()) {
        const auto& c = ls->curves[0];
        return c.mode_2d ? c.result_2d.error : c.last_error;
    }
    if (phase)  return phase->result.error;
    if (basins && !basins->configs.empty()) return basins->configs[0].last_error;
    return {};
}

void TreeLeafRuntime::request_cancel() {
    if (bif)    bif->request_cancel();
    if (lle)    lle->request_cancel();
    if (ls)     ls->request_cancel();
    if (phase)  phase->request_cancel();
    if (basins) basins->request_cancel();
}

// --- Пины ---

void tree_pin_axis(CustomTabSharedConfig& s, const std::vector<std::string>& params,
                   const std::vector<std::string>& vars, const EffectiveSweep& e, double value,
                   bool l1d_h) {
    if (e.over_h) {
        if (value > 0.0) (l1d_h ? s.l1d_h_text : s.h_text) = fmt_num_shortest(value);
        return;
    }
    if (e.over_var) {
        if (e.var_index >= 0 && e.var_index < (int)vars.size())
            s.initial_conditions[vars[e.var_index]] = fmt_num_shortest(value);
        return;
    }
    if (e.par_index >= 0 && e.par_index < (int)params.size())
        s.param_values[params[e.par_index]] = fmt_num_shortest(value);
}

namespace {

// Ось среза хранится в sweep_x_* узла Slice1D независимо от направления.
EffectiveSweep slice_own_axis(const CustomTabSharedConfig& c) {
    EffectiveSweep e;
    e.par_index = c.sweep_x_par_index;
    e.over_var  = c.sweep_x_over_var;
    e.var_index = c.sweep_x_var_index;
    e.over_h    = c.sweep_x_over_h;
    e.log_scale = c.sweep_x_log;
    e.lo_text   = c.sweep_x_lo_text;
    e.hi_text   = c.sweep_x_hi_text;
    e.n_pts_text = c.n_x_1d_text;
    return e;
}

void put_sweep(CustomTabSharedConfig& s, int dir, const EffectiveSweep& e) {
    if (dir == 0) {
        s.sweep_x_par_index = e.par_index; s.sweep_x_over_var = e.over_var;
        s.sweep_x_var_index = e.var_index; s.sweep_x_over_h = e.over_h;
        s.sweep_x_log = e.log_scale; s.sweep_x_lo_text = e.lo_text; s.sweep_x_hi_text = e.hi_text;
        s.n_x_1d_text = e.n_pts_text;
    } else {
        s.sweep_y_par_index = e.par_index; s.sweep_y_over_var = e.over_var;
        s.sweep_y_var_index = e.var_index; s.sweep_y_over_h = e.over_h;
        s.sweep_y_log = e.log_scale; s.sweep_y_lo_text = e.lo_text; s.sweep_y_hi_text = e.hi_text;
        s.n_y_1d_text = e.n_pts_text;
    }
}

EffectiveSweep region_axis(const CustomTabSharedConfig& c, int dir) {
    EffectiveSweep e;
    if (dir == 0) {
        e.par_index = c.axis_x_par_index; e.over_var = c.axis_x_over_var;
        e.var_index = c.axis_x_var_index; e.over_h = c.axis_x_over_h;
        e.log_scale = c.axis_x_log; e.lo_text = c.axis_x_lo_text; e.hi_text = c.axis_x_hi_text;
    } else {
        e.par_index = c.axis_y_par_index; e.over_var = c.axis_y_over_var;
        e.var_index = c.axis_y_var_index; e.over_h = c.axis_y_over_h;
        e.log_scale = c.axis_y_log; e.lo_text = c.axis_y_lo_text; e.hi_text = c.axis_y_hi_text;
    }
    e.n_pts_text = c.resolution_text;
    return e;
}

double sweep_mid(const EffectiveSweep& e) {
    const double lo = parse_num(e.lo_text, 0.0);
    const double hi = parse_num(e.hi_text, 1.0);
    if (e.log_scale && lo > 0.0 && hi > 0.0) return std::sqrt(lo * hi);
    return (lo + hi) * 0.5;
}

// --- Подписи (по образцу build_l*_signature в custom_session.cpp) ---
void sa(std::string& o, const char* t, const std::string& v) { o += t; o += '='; o += v; o += ';'; }
void sa(std::string& o, const char* t, int v)  { sa(o, t, std::to_string(v)); }
void sab(std::string& o, const char* t, bool v) { sa(o, t, std::string(v ? "1" : "0")); }
void sa_map(std::string& o, const char* t, const std::map<std::string, std::string>& m,
            const std::set<std::string>& skip) {
    o += t; o += '{';
    for (const auto& kv : m) {
        if (skip.count(kv.first)) continue;
        o += kv.first; o += '='; o += kv.second; o += ';';
    }
    o += '}';
}
void sa_axis(std::string& o, const char* t, const EffectiveSweep& e) {
    o += t; o += '[';
    sa(o, "p", e.par_index); sab(o, "v", e.over_var); sa(o, "vi", e.var_index);
    sab(o, "h", e.over_h); sab(o, "lg", e.log_scale); sa(o, "lo", e.lo_text); sa(o, "hi", e.hi_text);
    sa(o, "n", e.n_pts_text);
    o += ']';
}
void sa_adaptive(std::string& o, const AdaptiveSettings& a) {
    sab(o, "ae", a.enabled); sab(o, "arn", a.raw_nodes); sa(o, "art", a.rtol); sa(o, "aat", a.atol);
    sa(o, "ah0", a.h0); sa(o, "ahn", a.hmin); sa(o, "ahx", a.hmax); sa(o, "amr", a.max_rej);
    sa(o, "act", a.ctrl);
    for (const auto& p : a.ctrl_params) sa(o, "acp", p);
    sa(o, "api", a.peak_interp); sa(o, "amp", a.max_points); sa(o, "alr", a.lyap_renorm);
    sab(o, "amm", a.minmax_interp_fixed);
}
// Значение оси свипа задаёт сетка, а не поле: такие параметры и НУ в подпись не идут.
void skip_axis(std::set<std::string>& pv, std::set<std::string>& ic, const EffectiveSweep& e,
               const std::vector<std::string>& params, const std::vector<std::string>& vars) {
    if (e.over_h) return;
    if (e.over_var) {
        if (e.var_index >= 0 && e.var_index < (int)vars.size()) ic.insert(vars[e.var_index]);
    } else if (e.par_index >= 0 && e.par_index < (int)params.size()) {
        pv.insert(params[e.par_index]);
    }
}

std::string unique_child_name(const AnalysisTree& t, int parent, const std::string& base, int skip_id = 0) {
    auto taken = [&](const std::string& n) {
        const TreeNode* p = t.find(parent);
        if (!p) return false;
        for (int c : p->children) {
            if (c == skip_id) continue;
            const TreeNode* cn = t.find(c);
            if (cn && cn->name == n) return true;
        }
        return false;
    };
    // Суффикс « (N)» у оригинала не наращиваем: копия «LLE (2)» даёт «LLE (3)», а не «LLE (2) (2)».
    std::string stem = base;
    const size_t sp = stem.rfind(" (");
    if (sp != std::string::npos && !stem.empty() && stem.back() == ')') {
        bool digits = sp + 2 < stem.size() - 1;
        for (size_t i = sp + 2; i + 1 < stem.size(); ++i)
            if (!std::isdigit((unsigned char)stem[i])) digits = false;
        if (digits) stem = stem.substr(0, sp);
    }
    if (!taken(stem)) return stem;
    for (int k = 2; ; ++k) {
        std::string n = stem + " (" + std::to_string(k) + ")";
        if (!taken(n)) return n;
    }
}

} // namespace

// --- Загрузка ---

void TreeSession::load_from_record(const SystemRecord& r,
                                   const std::vector<std::string>& vars_,
                                   const std::vector<std::string>& params_) {
    vars   = vars_;
    params = params_;
    record = r;
    custom_schemes = r.custom_schemes;
    enabled_builtin_schemes = enabled_builtins_from_record(r);

    // Затравка — та же, что у CustomSession::load_from_record.
    CustomTabSharedConfig seed{};
    seed.scheme      = default_scheme_from_record(r, "Euler");
    seed.symmetry_s  = r.symmetry_s.empty() ? std::string("0.5") : r.symmetry_s;
    seed.cd_fp_iters = r.cd_fp_iters.empty() ? std::string("4") : r.cd_fp_iters;
    seed.newton      = newton_from_record(r);
    seed.h_text      = default_h_from_record(r);
    seed.l1d_h_text         = seed.h_text;
    seed.l1d_transient_text = seed.transient_text;
    seed.l1d_t_max_text     = seed.t_max_text;
    for (const auto& v : vars) {
        auto it = r.init_conditions.find(v);
        seed.initial_conditions[v] = (it != r.init_conditions.end()) ? it->second : std::string();
    }
    for (const auto& p : params) {
        auto it = r.param_values.find(p);
        seed.param_values[p] = (it != r.param_values.end()) ? it->second : std::string();
    }
    if (params.size() >= 1) seed.axis_x_par_index = 0;
    if (params.size() >= 2) seed.axis_y_par_index = 1;
    seed.fix_x_value = sweep_mid(region_axis(seed, 0));
    seed.fix_y_value = sweep_mid(region_axis(seed, 1));

    leaves.clear();
    tree = AnalysisTree{};
    TreeNode root;
    root.id   = tree.next_id++;
    root.kind = TreeNodeKind::System;
    root.name = tree_kind_title(TreeNodeKind::System);
    root.cfg  = seed;
    tree.nodes.push_back(std::move(root));

    // Раскладка по умолчанию — как в Custom: карты, X-срез с БД, выключенный Y-срез, ФП.
    const int reg = add_node(tree.nodes.front().id, TreeNodeKind::Region2D);
    add_node(reg, TreeNodeKind::Bif2D);
    add_node(reg, TreeNodeKind::LLE2D);
    add_node(reg, TreeNodeKind::LS2D);
    const int sx = add_node(reg, TreeNodeKind::Slice1D, 0);
    add_node(sx, TreeNodeKind::Bif1D);
    const int sy = add_node(reg, TreeNodeKind::Slice1D, 1);
    add_node(sy, TreeNodeKind::Bif1D);
    if (TreeNode* n = tree.find(sy)) { n->enabled = false; n->was_expanded = n->expanded; n->expanded = false; }
    const int pt = add_node(reg, TreeNodeKind::Point);
    add_node(pt, TreeNodeKind::Phase);
    tree.selected_id = reg;

    loaded_system_name = r.name;
    dirty = true;
}

void TreeSession::propagate_identity() {
    auto give = [&](auto& s) {
        s.custom_schemes = custom_schemes;
        s.wrapper_schemes = wrapper_schemes;
        s.enabled_builtin_schemes = enabled_builtin_schemes;
        s.loaded_system_name = loaded_system_name;
    };
    for (auto& [id, rt] : leaves) {
        if (rt.bif)    { give(*rt.bif);    rt.bif->sys = sys; }
        if (rt.lle)    { give(*rt.lle);    rt.lle->sys = sys; }
        if (rt.ls)     { give(*rt.ls);     rt.ls->sys = sys; }
        if (rt.basins) { give(*rt.basins); rt.basins->sys = sys; }
        if (rt.phase) {
            rt.phase->custom_schemes = custom_schemes;
            rt.phase->wrapper_schemes = wrapper_schemes;
            rt.phase->enabled_builtin_schemes = enabled_builtin_schemes;
            rt.phase->loaded_system_name = loaded_system_name;
            rt.phase->sys = sys;
        }
    }
}

void TreeSession::ensure_leaf_runtimes() {
    for (const auto& n : tree.nodes) {
        if (tree_kind_is_context(n.kind)) continue;
        auto& rt = leaves[n.id];
        switch (n.kind) {
        case TreeNodeKind::Bif2D: case TreeNodeKind::Bif1D:
            if (!rt.bif) {
                rt.bif = std::make_unique<BifurcationAnalysisSession>();
                rt.bif->load_from_record(record, vars, params);
            }
            break;
        case TreeNodeKind::LLE2D: case TreeNodeKind::LLE1D:
            if (!rt.lle) {
                rt.lle = std::make_unique<LLEAnalysisSession>();
                rt.lle->load_from_record(record, vars, params);
            }
            break;
        case TreeNodeKind::LS2D: case TreeNodeKind::LS1D:
            if (!rt.ls) {
                rt.ls = std::make_unique<LyapunovSpectrumAnalysisSession>();
                rt.ls->load_from_record(record, vars, params);
            }
            break;
        case TreeNodeKind::Phase:
            if (!rt.phase) {
                rt.phase = std::make_unique<PhaseAnalysisSession>();
                rt.phase->load_from_record(record, vars, params);
                rt.phase->observables.clear();
                rt.phase->obs_exprs.clear();
                // Окна проекций разных точек иначе не различить по заголовку.
                for (auto& pr : rt.phase->projections) pr.label = n.name + ": " + pr.label;
            }
            break;
        case TreeNodeKind::Basins:
            if (!rt.basins) {
                rt.basins = std::make_unique<BasinsAnalysisSession>();
                rt.basins->load_from_record(record, vars, params);
            }
            break;
        default: break;
        }
    }
    // Узлов больше нет — подсессии тоже.
    for (auto it = leaves.begin(); it != leaves.end();) {
        if (!tree.find(it->first)) it = leaves.erase(it); else ++it;
    }
    propagate_identity();
}

// --- Структура ---

int TreeSession::add_node(int parent, TreeNodeKind kind, int slice_dir) {
    TreeNode* p = tree.find(parent);
    if (!p) return 0;
    const auto allowed = tree_addable_children(p->kind);
    if (std::find(allowed.begin(), allowed.end(), kind) == allowed.end()) return 0;

    TreeNode n;
    n.id     = tree.next_id++;
    n.parent = parent;
    n.kind   = kind;
    n.slice_dir = slice_dir;
    // Поля контекста берутся у ближайшего предка: вложенный регион наследует оси родителя.
    n.cfg = p->cfg;
    std::string base = tree_kind_title(kind);
    if (kind == TreeNodeKind::Region2D && p->kind == TreeNodeKind::System) {
        n.cfg = tree.nodes.front().cfg;
    }
    if (kind == TreeNodeKind::Slice1D) {
        const int reg = tree.nearest(parent, TreeNodeKind::Region2D);
        const TreeNode* rn = tree.find(reg);
        const CustomTabSharedConfig& src = rn ? rn->cfg : tree.nodes.front().cfg;
        EffectiveSweep own = region_axis(src, slice_dir);
        own.n_pts_text = "500";
        put_sweep(n.cfg, 0, own);
        n.cfg.inherit_sweep_from_2d = rn != nullptr;
        const CustomTabSharedConfig& root = tree.nodes.front().cfg;
        n.cfg.l1d_h_text         = root.h_text;
        n.cfg.l1d_transient_text = root.transient_text;
        n.cfg.l1d_t_max_text     = root.t_max_text;
        n.cfg.fix_x_value = sweep_mid(own);
        base += slice_dir == 0 ? " X" : " Y";
    }
    if (kind == TreeNodeKind::Point) n.live = true;
    n.name = unique_child_name(tree, parent, base);
    const int id = n.id;
    p->children.push_back(id);
    p->expanded = true;
    tree.nodes.push_back(std::move(n));
    if (!tree_kind_is_context(kind)) ensure_leaf_runtimes();
    dirty = true;
    return id;
}

int TreeSession::duplicate_node(int id) {
    const TreeNode* src = tree.find(id);
    if (!src || src->parent == 0) return 0;
    const int parent = src->parent;
    std::map<int, int> remap;
    const std::vector<int> ids = tree.subtree(id);
    for (int old : ids) remap[old] = tree.next_id++;
    std::vector<TreeNode> copies;
    for (int old : ids) {
        TreeNode c = *tree.find(old);
        c.id = remap[old];
        c.parent = (old == id) ? parent : remap[c.parent];
        for (int& ch : c.children) ch = remap[ch];
        copies.push_back(std::move(c));
    }
    copies.front().name = unique_child_name(tree, parent, copies.front().name);
    for (auto& c : copies) tree.nodes.push_back(std::move(c));
    // Вставка сразу после оригинала.
    TreeNode* p = tree.find(parent);
    auto it = std::find(p->children.begin(), p->children.end(), id);
    p->children.insert(it == p->children.end() ? it : it + 1, remap[id]);
    ensure_leaf_runtimes();
    // Конфиги листов копируются без признака «посчитан»: копия сразу грязная.
    for (int old : ids) {
        auto so = leaves.find(old);
        auto sn = leaves.find(remap[old]);
        if (so == leaves.end() || sn == leaves.end()) continue;
        auto& a = so->second; auto& b = sn->second;
        if (a.bif && b.bif)       static_cast<BifurcationAnalysisState&>(*b.bif) = *a.bif;
        if (a.lle && b.lle)       static_cast<LLEAnalysisState&>(*b.lle) = *a.lle;
        if (a.ls && b.ls)         static_cast<LyapunovSpectrumAnalysisState&>(*b.ls) = *a.ls;
        // Projection некопируема: у ФП переносим только входы расчёта, проекции — по умолчанию.
        if (a.phase && b.phase) {
            b.phase->ic_sets    = a.phase->ic_sets;
            b.phase->decimation = a.phase->decimation;
        }
        if (a.basins && b.basins) static_cast<BasinsAnalysisState&>(*b.basins) = *a.basins;
    }
    propagate_identity();
    dirty = true;
    return remap[id];
}

bool TreeSession::subtree_in_flight(int id) const {
    for (int n : tree.subtree(id)) {
        auto it = leaves.find(n);
        if (it != leaves.end() && it->second.in_flight()) return true;
    }
    return false;
}

bool TreeSession::remove_node(int id) {
    const TreeNode* n = tree.find(id);
    if (!n || n->parent == 0) return false;
    if (subtree_in_flight(id)) return false;
    const int parent = n->parent;
    const std::vector<int> ids = tree.subtree(id);
    const std::set<int> gone(ids.begin(), ids.end());
    if (TreeNode* p = tree.find(parent))
        p->children.erase(std::remove(p->children.begin(), p->children.end(), id), p->children.end());
    tree.nodes.erase(std::remove_if(tree.nodes.begin(), tree.nodes.end(),
                                    [&](const TreeNode& x) { return gone.count(x.id) > 0; }),
                     tree.nodes.end());
    for (int g : ids) leaves.erase(g);
    if (gone.count(tree.selected_id)) tree.selected_id = parent;
    dirty = true;
    return true;
}

void TreeSession::toggle_enabled(int id) {
    TreeNode* n = tree.find(id);
    if (!n) return;
    n->enabled = !n->enabled;
    if (!n->enabled) { n->was_expanded = n->expanded; n->expanded = false; }
    else             { n->expanded = n->was_expanded; }
    dirty = true;
}

void TreeSession::move_node(int id, int delta) {
    const TreeNode* n = tree.find(id);
    if (!n || n->parent == 0) return;
    TreeNode* p = tree.find(n->parent);
    auto it = std::find(p->children.begin(), p->children.end(), id);
    if (it == p->children.end()) return;
    const std::ptrdiff_t pos = it - p->children.begin();
    const std::ptrdiff_t to  = pos + delta;
    if (to < 0 || to >= (std::ptrdiff_t)p->children.size()) return;
    std::swap(p->children[pos], p->children[to]);
    dirty = true;
}

// --- Контекст ---

int TreeSession::region_of(int id) const {
    return tree.nearest(id, TreeNodeKind::Region2D);
}

CustomTabSharedConfig TreeSession::effective_shared(int leaf_id) const {
    CustomTabSharedConfig s = tree.nodes.empty() ? CustomTabSharedConfig{} : tree.nodes.front().cfg;
    const TreeNode* leaf = tree.find(leaf_id);
    if (!leaf) return s;
    const TreeNode* reg = tree.find(tree.nearest(leaf_id, TreeNodeKind::Region2D));
    const TreeNode* sl  = tree.find(tree.nearest(leaf_id, TreeNodeKind::Slice1D));
    const TreeNode* pt  = tree.find(tree.nearest(leaf_id, TreeNodeKind::Point));

    s.level_2d_enabled = reg != nullptr;
    s.inherit_sweep_from_2d = reg != nullptr;
    if (reg) {
        const auto& r = reg->cfg;
        s.axis_x_par_index = r.axis_x_par_index; s.axis_x_over_var = r.axis_x_over_var;
        s.axis_x_var_index = r.axis_x_var_index; s.axis_x_over_h = r.axis_x_over_h;
        s.axis_x_log = r.axis_x_log; s.axis_x_lo_text = r.axis_x_lo_text; s.axis_x_hi_text = r.axis_x_hi_text;
        s.axis_y_par_index = r.axis_y_par_index; s.axis_y_over_var = r.axis_y_over_var;
        s.axis_y_var_index = r.axis_y_var_index; s.axis_y_over_h = r.axis_y_over_h;
        s.axis_y_log = r.axis_y_log; s.axis_y_lo_text = r.axis_y_lo_text; s.axis_y_hi_text = r.axis_y_hi_text;
        s.resolution_text  = r.resolution_text;
        s.bif_writable_var = r.bif_writable_var;
        s.fix_x_value = r.fix_x_value;
        s.fix_y_value = r.fix_y_value;
        // Своя ось среза без наследования подставляется ниже; вторая ось — ось региона.
        put_sweep(s, 0, region_axis(r, 0));
        put_sweep(s, 1, region_axis(r, 1));
    }
    int dir = 0;
    if (sl) {
        dir = sl->slice_dir;
        const auto& c = sl->cfg;
        // Интегратор среза: свой (галка Own integration) или общий System.
        if (sl->own_integration) {
            s.l1d_h_text = c.l1d_h_text;
            s.l1d_transient_text = c.l1d_transient_text;
            s.l1d_t_max_text = c.l1d_t_max_text;
        } else {
            s.l1d_h_text = s.h_text;
            s.l1d_transient_text = s.transient_text;
            s.l1d_t_max_text = s.t_max_text;
        }
        s.continuation_1d_enabled = c.continuation_1d_enabled;
        s.plot_inter_peaks_1d = c.plot_inter_peaks_1d;
        const bool inherit = reg && c.inherit_sweep_from_2d;
        EffectiveSweep own = slice_own_axis(c);
        if (inherit) {
            // Ось — региона, разрешение — своё (как EffectiveSweep при inherit).
            EffectiveSweep ax = region_axis(reg->cfg, dir);
            ax.n_pts_text = own.n_pts_text;
            own = ax;
        }
        s.inherit_sweep_from_2d = false;   // эффективные оси уже разложены в sweep_*
        put_sweep(s, dir, own);
        if (dir == 0) s.n_x_1d_text = own.n_pts_text; else s.n_y_1d_text = own.n_pts_text;
        if (!reg) {
            (dir == 0 ? s.fix_x_value : s.fix_y_value) = c.fix_x_value;
        } else if (!sl->follow_cursor) {
            s.fix_x_value = c.fix_x_value;
            s.fix_y_value = c.fix_y_value;
        }
    }
    if (pt && !pt->follow_cursor) {
        s.fix_x_value = pt->cfg.fix_x_value;
        s.fix_y_value = pt->cfg.fix_y_value;
    }
    if (leaf->own_integration && !tree_kind_is_context(leaf->kind)) {
        const auto& c = leaf->cfg;
        s.h_text = s.l1d_h_text = c.h_text;
        s.transient_text = s.l1d_transient_text = c.transient_text;
        s.t_max_text = s.l1d_t_max_text = c.t_max_text;
        s.pre_scaller_text   = c.pre_scaller_text;
        s.initial_conditions = c.initial_conditions;
    }

    // Пины закреплённых осей.
    const EffectiveSweep ex = effective_sweep_x(s);
    const EffectiveSweep ey = effective_sweep_y(s);
    if (tree_kind_is_1d_leaf(leaf->kind)) {
        if (reg) {
            const int other = 1 - dir;
            tree_pin_axis(s, params, vars, other == 0 ? ex : ey,
                          other == 0 ? s.fix_x_value : s.fix_y_value, /*l1d_h*/ true);
        }
    } else if (tree_kind_is_0d_leaf(leaf->kind)) {
        if (sl) {
            tree_pin_axis(s, params, vars, dir == 0 ? ex : ey,
                          dir == 0 ? s.fix_x_value : s.fix_y_value, false);
            if (reg) {
                const int other = 1 - dir;
                tree_pin_axis(s, params, vars, other == 0 ? ex : ey,
                              other == 0 ? s.fix_x_value : s.fix_y_value, false);
            }
        } else if (reg) {
            tree_pin_axis(s, params, vars, ex, s.fix_x_value, false);
            tree_pin_axis(s, params, vars, ey, s.fix_y_value, false);
        }
    }
    return s;
}

std::string TreeSession::leaf_signature(int leaf_id) const {
    const TreeNode* leaf = tree.find(leaf_id);
    auto rit = leaves.find(leaf_id);
    if (!leaf || rit == leaves.end()) return {};
    const auto& rt = rit->second;
    const CustomTabSharedConfig s = effective_shared(leaf_id);
    const bool is2d = tree_kind_is_2d_leaf(leaf->kind);
    const bool is1d = tree_kind_is_1d_leaf(leaf->kind);
    const TreeNode* sl = tree.find(tree.nearest(leaf_id, TreeNodeKind::Slice1D));
    const int dir = sl ? sl->slice_dir : 0;
    const EffectiveSweep ex = effective_sweep_x(s);
    const EffectiveSweep ey = effective_sweep_y(s);

    std::set<std::string> skip_pv, skip_ic;
    bool skip_h = false;
    if (is2d) {
        skip_axis(skip_pv, skip_ic, ex, params, vars);
        skip_axis(skip_pv, skip_ic, ey, params, vars);
        skip_h = ex.over_h || ey.over_h;
    } else if (is1d) {
        const EffectiveSweep& own = dir == 0 ? ex : ey;
        skip_axis(skip_pv, skip_ic, own, params, vars);
        skip_h = own.over_h;
    }

    std::string o;
    o.reserve(768);
    sa(o, "k", tree_kind_key(leaf->kind));
    sa(o, "sc", s.scheme); sa(o, "sym", s.symmetry_s); sa(o, "fpn", s.cd_fp_iters);
    sa(o, "ntol", s.newton.tol); sa(o, "nit", s.newton.max_iters); sab(o, "nfull", s.newton.full);
    if (is1d) {
        if (!skip_h) sa(o, "h", s.l1d_h_text);
        sa(o, "tt", s.l1d_transient_text); sa(o, "ct", s.l1d_t_max_text);
    } else {
        if (!skip_h) sa(o, "h", s.h_text);
        sa(o, "tt", s.transient_text); sa(o, "ct", s.t_max_text);
    }
    sa(o, "dec", s.pre_scaller_text); sa(o, "mv", s.max_value_text);
    sa_adaptive(o, s.adaptive);
    sa_map(o, "ic", s.initial_conditions, skip_ic);
    sa_map(o, "pv", s.param_values, skip_pv);

    if (is2d) {
        sa_axis(o, "ax", ex); sa_axis(o, "ay", ey);
        sa(o, "res", s.resolution_text);
    } else if (is1d) {
        sa_axis(o, "own", dir == 0 ? ex : ey);
        sab(o, "cnt", s.continuation_1d_enabled);
    }

    if (rt.bif && !rt.bif->diagrams.empty()) {
        const auto& c = rt.bif->diagrams[0];
        sa(o, "wv", s.bif_writable_var);
        if (is2d) {
            sa(o, "beps", c.eps_dbscan_text); sa(o, "bmp", c.mult_peak_text);
            sa(o, "bmi", c.mult_interval_text);
        }
    }
    if (rt.lle && !rt.lle->curves.empty()) {
        sa(o, "leps", rt.lle->curves[0].eps_text); sa(o, "lnt", rt.lle->curves[0].nt_text);
    }
    if (rt.ls && !rt.ls->curves.empty()) {
        sa(o, "seps", rt.ls->curves[0].eps_text); sa(o, "snt", rt.ls->curves[0].nt_text);
    }
    if (rt.phase) {
        sa(o, "dc", rt.phase->decimation);
        for (size_t i = 0; i < rt.phase->ic_sets.size(); ++i)
            sa_map(o, ("ic" + std::to_string(i)).c_str(), rt.phase->ic_sets[i].values, {});
    }
    if (rt.basins && !rt.basins->configs.empty()) {
        const auto& b = rt.basins->configs[0];
        sa(o, "bxv", b.axis_x_var); sa(o, "bxlo", b.axis_x_lo_text); sa(o, "bxhi", b.axis_x_hi_text);
        sa(o, "byv", b.axis_y_var); sa(o, "bylo", b.axis_y_lo_text); sa(o, "byhi", b.axis_y_hi_text);
        sab(o, "bxs", b.axis_x_symlog); sab(o, "bys", b.axis_y_symlog);
        sa(o, "bn", b.n_pts_text); sa(o, "bwv", b.writable_var);
        sa(o, "f1", b.feature1); sa(o, "f2", b.feature2);
        sa(o, "mf1", b.mult_feature1_text); sa(o, "mf2", b.mult_feature2_text);
        sa(o, "beps", b.eps_dbscan_text);
    }
    return o;
}

bool TreeSession::leaf_dirty(int leaf_id) const {
    auto it = leaves.find(leaf_id);
    if (it == leaves.end()) return false;
    const auto& rt = it->second;
    if (!rt.committed_ok) return true;
    return rt.committed != leaf_signature(leaf_id);
}

std::vector<int> TreeSession::dirty_leaves(int subtree_root) const {
    std::vector<int> out;
    for (int id : tree.subtree(subtree_root)) {
        const TreeNode* n = tree.find(id);
        if (!n || tree_kind_is_context(n->kind)) continue;
        if (!tree.is_active(id)) continue;
        if (leaf_dirty(id)) out.push_back(id);
    }
    return out;
}

int TreeSession::cursor_owner(int id) const {
    const int reg = region_of(id);
    return reg ? reg : tree.nearest(id, TreeNodeKind::Slice1D);
}

std::vector<int> TreeSession::cursor_leaves(int region_id) const {
    std::vector<int> out;
    for (int id : tree.subtree(region_id)) {
        const TreeNode* n = tree.find(id);
        if (!n || tree_kind_is_context(n->kind) || tree_kind_is_2d_leaf(n->kind)) continue;
        if (cursor_owner(id) != region_id || !tree.is_active(id)) continue;
        // Все контексты между листом и регионом обязаны следовать за курсором и быть живыми.
        bool follows = true;
        for (const TreeNode* c = tree.find(n->parent); c && c->id != region_id;
             c = tree.find(c->parent)) {
            if (!c->follow_cursor || !c->live) { follows = false; break; }
        }
        if (follows && leaf_dirty(id)) out.push_back(id);
    }
    return out;
}

// --- Расчёт ---

bool TreeSession::start_leaf(int leaf_id, ParametricEngine& engine) {
    const TreeNode* leaf = tree.find(leaf_id);
    auto it = leaves.find(leaf_id);
    if (!leaf || it == leaves.end()) return false;
    auto& rt = it->second;
    if (rt.in_flight()) return false;
    const CustomTabSharedConfig s = effective_shared(leaf_id);
    const TreeNode* sl = tree.find(tree.nearest(leaf_id, TreeNodeKind::Slice1D));
    const int dir = sl ? sl->slice_dir : 0;
    rt.pending = leaf_signature(leaf_id);
    rt.pending_armed = true;
    // Подпись в плашке расчёта и в легенде — имя узла.
    auto name_cfg = [&](auto& cfg) { cfg.label = leaf->name; cfg.label_is_manual = true; };
    if (rt.bif && !rt.bif->diagrams.empty()) name_cfg(rt.bif->diagrams[0]);
    if (rt.lle && !rt.lle->curves.empty())   name_cfg(rt.lle->curves[0]);
    if (rt.ls && !rt.ls->curves.empty())     name_cfg(rt.ls->curves[0]);
    if (rt.basins && !rt.basins->configs.empty()) rt.basins->configs[0].label = leaf->name;

    bool ok = false;
    switch (leaf->kind) {
    case TreeNodeKind::Bif2D:
        if (rt.bif && !rt.bif->diagrams.empty()) {
            apply_shared_to_bif2d(s, rt.bif->diagrams[0]);
            ok = rt.bif->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::Bif1D:
        if (rt.bif && !rt.bif->diagrams.empty()) {
            apply_shared_to_bif1d(s, rt.bif->diagrams[0], dir);
            ok = rt.bif->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::LLE2D:
        if (rt.lle && !rt.lle->curves.empty()) {
            apply_shared_to_lle2d(s, rt.lle->curves[0]);
            ok = rt.lle->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::LLE1D:
        if (rt.lle && !rt.lle->curves.empty()) {
            const LLECurveConfig own = rt.lle->curves[0];   // eps/NT — свои у листа
            apply_shared_to_lle1d(s, own, rt.lle->curves[0], dir);
            ok = rt.lle->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::LS2D:
        if (rt.ls && !rt.ls->curves.empty()) {
            apply_shared_to_ls2d(s, rt.ls->curves[0]);
            ok = rt.ls->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::LS1D:
        if (rt.ls && !rt.ls->curves.empty()) {
            const LSCurveConfig own = rt.ls->curves[0];
            apply_shared_to_ls1d(s, own, rt.ls->curves[0], dir);
            ok = rt.ls->run_async(engine, 0);
        }
        break;
    case TreeNodeKind::Phase:
        if (rt.phase) {
            apply_shared_to_phase(s, *rt.phase, vars);
            rt.phase->regenerate_krs();
            ok = rt.phase->recompute_async();
        }
        break;
    case TreeNodeKind::Basins:
        if (rt.basins && !rt.basins->configs.empty()) {
            apply_shared_to_basins(s, rt.basins->configs[0]);
            ok = rt.basins->run_async(engine, 0);
        }
        break;
    default: break;
    }
    if (!ok) rt.pending_armed = false;
    return ok;
}

bool TreeSession::any_in_flight() const {
    for (const auto& [id, rt] : leaves) if (rt.in_flight()) return true;
    return false;
}

void TreeSession::request_cancel_all() {
    for (auto& [id, rt] : leaves) rt.request_cancel();
}

bool TreeSession::poll_all() {
    bool any = false;
    for (auto& [id, rt] : leaves) {
        bool done = false;
        // Отменённый ФП оставляет прошлый result.ok и не двигает data_generation.
        const int phase_gen = rt.phase ? rt.phase->data_generation : 0;
        if (rt.bif    && rt.bif->poll())    done = true;
        if (rt.lle    && rt.lle->poll())    done = true;
        if (rt.ls     && rt.ls->poll())     done = true;
        if (rt.phase  && rt.phase->poll())  done = true;
        if (rt.basins && rt.basins->poll()) done = true;
        if (!done) continue;
        any = true;
        if (rt.pending_armed) {
            const bool cancelled = rt.phase && rt.phase->data_generation == phase_gen;
            rt.committed     = rt.pending;
            rt.committed_ok  = rt.last_ok() && !cancelled;
            rt.pending_armed = false;
            rt.pending.clear();
        }
    }
    return any;
}

bool TreeSession::tick_basins_phase(double now) {
    bool done = false;
    for (auto& [id, rt] : leaves) {
        if (!rt.basins) continue;
        auto& b = *rt.basins;
        b.ensure_phase_slots();
        if (b.poll_phase()) done = true;
        // Автозапуск — как basins_phase_tick вкладки Basins (tick.cpp).
        for (int i = 0; i < (int)b.configs.size(); ++i) {
            BasinsConfig&    c  = b.configs[(size_t)i];
            BasinsPhaseSlot& sl = *b.phase_slots[(size_t)i];
            const std::string sig = build_basins_phase_signature(c);
            if (sig != sl.pending_signature) { sl.pending_signature = sig; sl.pending_since = now; }
            if (sig == sl.run_signature || !c.pp_autorun) continue;
            if (!c.last_run_ok || c.result.basin_idx.empty() || sl.phase.in_flight) continue;
            if (now - sl.pending_since < 0.3) continue;
            b.request_phase_run(i);
        }
        b.start_next_phase();
        if (b.phase_settings_dirty) { b.phase_settings_dirty = false; dirty = true; }
    }
    return done;
}

void TreeSession::enable_own_integration(int leaf_id) {
    TreeNode* n = tree.find(leaf_id);
    if (!n || n->own_integration) return;
    if (n->kind == TreeNodeKind::Slice1D) {
        // Свой 1D-интегратор среза стартует с унаследованных значений System.
        const CustomTabSharedConfig& root = tree.nodes.front().cfg;
        n->cfg.l1d_h_text         = root.h_text;
        n->cfg.l1d_transient_text = root.transient_text;
        n->cfg.l1d_t_max_text     = root.t_max_text;
        n->own_integration = true;
        dirty = true;
        return;
    }
    if (tree_kind_is_context(n->kind)) return;
    const CustomTabSharedConfig s = effective_shared(leaf_id);
    const bool is1d = tree_kind_is_1d_leaf(n->kind);
    n->cfg.h_text             = is1d ? s.l1d_h_text : s.h_text;
    n->cfg.transient_text     = is1d ? s.l1d_transient_text : s.transient_text;
    n->cfg.t_max_text         = is1d ? s.l1d_t_max_text : s.t_max_text;
    n->cfg.pre_scaller_text   = s.pre_scaller_text;
    n->cfg.initial_conditions = s.initial_conditions;
    n->own_integration = true;
    dirty = true;
}

void TreeSession::clamp_cursors() {
    for (auto& n : tree.nodes) {
        if (n.kind != TreeNodeKind::Region2D) continue;
        for (int d = 0; d < 2; ++d) {
            const EffectiveSweep e = region_axis(n.cfg, d);
            double lo = parse_num(e.lo_text, 0.0), hi = parse_num(e.hi_text, 1.0);
            if (hi < lo) std::swap(lo, hi);
            double& v = d == 0 ? n.cfg.fix_x_value : n.cfg.fix_y_value;
            v = std::clamp(v, lo, hi);
        }
    }
}
