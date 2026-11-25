// Вкладка Tree без GUI:
//  - дерево по умолчанию повторяет раскладку Custom, а конфиги листов совпадают с теми, что
//    собирает Custom при тех же полях (2D-карта и X-срез с пином второй оси);
//  - подписи: сдвиг курсора по X пачкает только Y-срезы и точку, правка параметра без свипа —
//    все листья, правка значения свипуемого параметра — ничего;
//  - cursor_leaves отбирает живые следующие за курсором листья;
//  - duplicate / remove / move и очередь AppModel (без повторов, удалённые узлы уходят);
//  - круг JSON -> дерево -> JSON, битый и чужой файл не трогают сессию;
//  - 1D-интегратор среза наследуется от System, Own integration и старые файлы.
#include "ucuda/application/app_model.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/domain/num_parse.h"
#include "ucuda/platform/platform_paths.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

std::string exe_dir() { return platform::executable_dir(); }

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

const std::vector<std::string> kVars   = { "x", "y", "z" };
const std::vector<std::string> kParams = { "a", "b", "c" };

SystemRecord make_record() {
    SystemRecord r;
    r.name = "tree-test";
    r.param_values = { { "a", "0.2" }, { "b", "0.2" }, { "c", "5.7" } };
    r.init_conditions = { { "x", "1" }, { "y", "1" }, { "z", "1" } };
    return r;
}

int find_kind(const TreeSession& ts, TreeNodeKind k, int nth = 0) {
    for (const auto& n : ts.tree.nodes)
        if (n.kind == k && nth-- == 0) return n.id;
    return 0;
}

void commit_all(TreeSession& ts) {
    for (auto& [id, rt] : ts.leaves) {
        rt.committed = ts.leaf_signature(id);
        rt.committed_ok = true;
    }
}

void default_tree_matches_custom() {
    TreeSession ts;
    ts.load_from_record(make_record(), kVars, kParams);
    const int reg = find_kind(ts, TreeNodeKind::Region2D);
    const int sx  = find_kind(ts, TreeNodeKind::Slice1D, 0);
    const int sy  = find_kind(ts, TreeNodeKind::Slice1D, 1);
    const int pt  = find_kind(ts, TreeNodeKind::Point);
    require(reg && sx && sy && pt, "default tree misses a context");
    require(ts.tree.find(sx)->slice_dir == 0 && ts.tree.find(sy)->slice_dir == 1, "slice directions");
    require(!ts.tree.find(sy)->enabled, "Y slice is off by default, as in Custom");
    require(ts.leaves.size() == 6, "default tree has 6 leaves");
    require(ts.tree.level(find_kind(ts, TreeNodeKind::Phase)) == 3, "Phase sits at L3");

    CustomSession cs;
    cs.load_from_record(make_record(), kVars, kParams);
    const auto& sh = cs.shared;

    // 2D-карта: те же оси, сетка и значения.
    const int bif2d = find_kind(ts, TreeNodeKind::Bif2D);
    BifurcationDiagramConfig a, b;
    apply_shared_to_bif2d(ts.effective_shared(bif2d), a);
    apply_shared_to_bif2d(sh, b);
    require(a.param_index == b.param_index && a.param_index_2 == b.param_index_2
            && a.param_lo_text == b.param_lo_text && a.param_hi_2_text == b.param_hi_2_text
            && a.n_pts_text == b.n_pts_text && a.h_text == b.h_text && a.scheme == b.scheme
            && a.param_values == b.param_values && a.initial_conditions == b.initial_conditions,
            "Bif2D config differs from Custom");

    // X-срез: свип по X, Y закреплён в курсоре (в Custom это делает pin_fixed_param очереди).
    const int bif1d = ts.tree.find(sx)->children.front();
    BifurcationDiagramConfig c, d;
    apply_shared_to_bif1d(ts.effective_shared(bif1d), c, 0);
    apply_shared_to_bif1d(sh, d, 0);
    d.param_values[kParams[effective_sweep_y(sh).par_index]] = fmt_num_shortest(sh.fix_y_value);
    require(c.param_index == d.param_index && c.param_lo_text == d.param_lo_text
            && c.param_hi_text == d.param_hi_text && c.n_pts_text == d.n_pts_text
            && c.h_text == d.h_text && c.param_values == d.param_values,
            "Bif1D X-slice config differs from Custom");
}

void signatures_follow_the_cursor() {
    TreeSession ts;
    ts.load_from_record(make_record(), kVars, kParams);
    const int reg = find_kind(ts, TreeNodeKind::Region2D);
    const int sx  = find_kind(ts, TreeNodeKind::Slice1D, 0);
    const int sy  = find_kind(ts, TreeNodeKind::Slice1D, 1);
    const int bx  = ts.tree.find(sx)->children.front();
    const int by  = ts.tree.find(sy)->children.front();
    const int ph  = find_kind(ts, TreeNodeKind::Phase);
    const int b2  = find_kind(ts, TreeNodeKind::Bif2D);
    require(ts.leaf_dirty(b2), "a never computed leaf is dirty");

    commit_all(ts);
    require(ts.dirty_leaves(ts.tree.nodes.front().id).empty(), "everything is clean after commit");

    ts.tree.find(reg)->cfg.fix_x_value += 0.125;
    require(!ts.leaf_dirty(b2), "cursor X does not touch 2D");
    require(!ts.leaf_dirty(bx), "cursor X does not touch the X slice");
    require(ts.leaf_dirty(by), "cursor X moves the Y slice");
    require(ts.leaf_dirty(ph), "cursor X moves the point");

    // Живые следующие за курсором листья: Y-срез выключен, остаётся точка.
    auto leaves = ts.cursor_leaves(reg);
    require(leaves == std::vector<int>{ ph }, "cursor leaves with the Y slice off");
    ts.toggle_enabled(sy);
    leaves = ts.cursor_leaves(reg);
    require(leaves.size() == 2 && std::find(leaves.begin(), leaves.end(), by) != leaves.end(),
            "cursor leaves with the Y slice on");
    ts.tree.find(find_kind(ts, TreeNodeKind::Point))->live = false;
    require(ts.cursor_leaves(reg) == std::vector<int>{ by }, "a non-live point is left for Run");

    commit_all(ts);
    ts.tree.find(reg)->cfg.fix_y_value += 0.125;
    require(ts.leaf_dirty(bx) && !ts.leaf_dirty(by) && !ts.leaf_dirty(b2), "cursor Y moves only the X slice");

    // Значение свипуемого параметра в System перекрывают сетка и курсор.
    commit_all(ts);
    ts.tree.nodes.front().cfg.param_values["a"] = "0.37";
    require(ts.dirty_leaves(ts.tree.nodes.front().id).empty(), "swept parameter value is not an input");
    ts.tree.nodes.front().cfg.param_values["c"] = "6";
    require(ts.leaf_dirty(b2) && ts.leaf_dirty(bx) && ts.leaf_dirty(by) && ts.leaf_dirty(ph),
            "a non-swept parameter dirties every leaf");

    // Свой интегратор листа: стартует с унаследованного, дальше живёт сам; пины осей остаются.
    commit_all(ts);
    ts.enable_own_integration(ph);
    require(!ts.leaf_dirty(ph), "switching to own integration keeps the inputs");
    ts.tree.find(ph)->cfg.t_max_text = "777";
    ts.tree.find(ph)->cfg.param_values["a"] = "999";   // параметры листа не перекрывают System
    const CustomTabSharedConfig own = ts.effective_shared(ph);
    require(own.t_max_text == "777" && ts.leaf_dirty(ph) && !ts.leaf_dirty(b2), "own time is the leaf's");
    require(own.param_values.at("a") == fmt_num_shortest(ts.tree.find(reg)->cfg.fix_x_value),
            "the point pin still sets the swept parameter");
    ts.tree.nodes.front().cfg.h_text = "0.002";
    require(ts.effective_shared(ph).h_text != "0.002", "System h no longer reaches an own-integration leaf");

    // Правка контекста пачкает только его поддерево.
    commit_all(ts);
    ts.tree.find(sx)->cfg.n_x_1d_text = "321";
    require(ts.leaf_dirty(bx) && !ts.leaf_dirty(by) && !ts.leaf_dirty(b2), "slice N dirties its leaves only");
}

void structure_and_queue() {
    AppModel model([](const std::vector<unsigned char>&) { return std::string{}; });
    TreeSession& ts = model.tree_session;
    ts.load_from_record(make_record(), kVars, kParams);
    const int root = ts.tree.nodes.front().id;
    const int reg  = find_kind(ts, TreeNodeKind::Region2D);
    const size_t nodes0 = ts.tree.nodes.size(), leaves0 = ts.leaves.size();

    require(ts.add_node(reg, TreeNodeKind::Phase) == 0, "a Phase leaf cannot hang under a region");
    require(ts.add_node(reg, TreeNodeKind::Region2D) == 0, "no nested regions");
    const int copy = ts.duplicate_node(reg);
    require(copy != 0 && ts.tree.find(copy)->name == "Region 2D (2)", "duplicate name");
    const auto& kids = ts.tree.find(root)->children;
    require(kids.size() == 2 && kids[0] == reg && kids[1] == copy, "copy goes right after the original");
    require(ts.tree.nodes.size() == nodes0 * 2 - 1 && ts.leaves.size() == leaves0 * 2, "subtree copied");
    require(ts.leaf_dirty(ts.tree.find(copy)->children.front()), "a copy is dirty");
    const int again = ts.duplicate_node(copy);
    require(ts.tree.find(again)->name == "Region 2D (3)", "suffix is replaced, not stacked");
    require(model.remove_tree_node(again), "remove the second copy");

    // Очередь: повторы пропускаются, удалённые листья уходят вместе с узлом.
    const int copy_leaf = ts.tree.find(copy)->children.front();
    model.enqueue_tree({ copy_leaf, copy_leaf, ts.tree.find(reg)->children.front() });
    require(model.tree_queue().size() == 2, "tree queue skips duplicates");
    require(model.remove_tree_node(copy), "remove the copy");
    require(model.tree_queue().size() == 1 && !ts.tree.find(copy_leaf) && ts.leaves.size() == leaves0,
            "removed subtree leaves the queue and the runtimes");
    require(!model.remove_tree_node(root), "the root cannot be removed");

    const int pt = find_kind(ts, TreeNodeKind::Point);
    const auto order0 = ts.tree.find(reg)->children;
    ts.move_node(pt, -1);
    const auto& order1 = ts.tree.find(reg)->children;
    require(order1[order1.size() - 2] == pt, "move up");
    ts.move_node(order0.front(), -1);
    require(ts.tree.find(reg)->children.front() == order0.front(), "first child stays first");
    model.clear_all_queues();
    require(model.tree_queue().empty(), "clear_all_queues drains the tree queue");
}

void json_round_trip() {
    TreeSession a;
    a.load_from_record(make_record(), kVars, kParams);
    const int reg = find_kind(a, TreeNodeKind::Region2D);
    a.tree.find(reg)->cfg.fix_x_value = 0.123456789012345;
    a.tree.find(reg)->name = "Wide \"map\"";
    a.add_node(find_kind(a, TreeNodeKind::Point), TreeNodeKind::Basins);
    a.leaves[find_kind(a, TreeNodeKind::LLE2D)].lle->curves[0].eps_text = "3e-5";
    // Свой интегратор и НУ листа бассейнов, доп. окна LS 2D — тоже в файле.
    const int bas = find_kind(a, TreeNodeKind::Basins);
    a.enable_own_integration(bas);
    a.tree.find(bas)->cfg.t_max_text = "2000";
    a.tree.find(bas)->cfg.initial_conditions["z"] = "16";
    a.tree.find(find_kind(a, TreeNodeKind::LS2D))->extra_views = { 1, -1 };
    const std::string j1 = session_to_json_tree(a);

    TreeSession b;
    b.load_from_record(make_record(), kVars, kParams);
    require(session_from_json_tree(j1, b), "tree JSON parses");
    require(session_to_json_tree(b) == j1, "tree JSON round trip changes the file");
    require(b.tree.find(reg)->cfg.fix_x_value == 0.123456789012345, "cursor keeps every digit");
    require(b.leaves[find_kind(b, TreeNodeKind::LLE2D)].lle->curves[0].eps_text == "3e-5", "leaf config restored");
    require(b.leaves.size() == a.leaves.size(), "leaf runtimes restored");
    require(b.tree.find(bas)->own_integration && b.tree.find(bas)->cfg.t_max_text == "2000",
            "own integration restored");
    require(b.tree.find(find_kind(b, TreeNodeKind::LS2D))->extra_views == std::vector<int>({ 1, -1 }),
            "extra windows restored");

    // Битый файл и неизвестный вид узла не трогают сессию.
    const std::string before = session_to_json_tree(b);
    require(!session_from_json_tree(j1.substr(0, j1.size() / 2), b), "truncated JSON is rejected");
    std::string alien = j1;
    const size_t k = alien.find("\"region2d\"");
    alien.replace(k, 10, "\"hologram\"");
    require(!session_from_json_tree(alien, b), "unknown node kind is rejected");
    require(session_to_json_tree(b) == before, "failed load keeps the session");
}

// 1D-интегратор среза: по умолчанию — System, галка Own integration — свои поля;
// старый файл без ключа грузится со своими полями (счёт старых сессий не меняется).
void slice_integrator_inherits() {
    TreeSession ts;
    ts.load_from_record(make_record(), kVars, kParams);
    const int sx = find_kind(ts, TreeNodeKind::Slice1D, 0);
    const int bif1d = ts.tree.find(sx)->children.front();
    require(!ts.tree.find(sx)->own_integration, "new slice inherits the integrator");
    ts.tree.nodes.front().cfg.transient_text = "1700";
    ts.tree.find(sx)->cfg.l1d_transient_text = "100";
    require(ts.effective_shared(bif1d).l1d_transient_text == "1700", "slice inherits System transient");

    ts.enable_own_integration(sx);
    require(ts.tree.find(sx)->own_integration, "slice own integration on");
    require(ts.tree.find(sx)->cfg.l1d_transient_text == "1700", "own integrator starts from inherited values");
    ts.tree.find(sx)->cfg.l1d_transient_text = "100";
    require(ts.effective_shared(bif1d).l1d_transient_text == "100", "own slice integrator is used");

    const std::string j = session_to_json_tree(ts);
    TreeSession b;
    b.load_from_record(make_record(), kVars, kParams);
    require(session_from_json_tree(j, b) && b.tree.find(sx)->own_integration, "slice own integration restored");

    ts.tree.find(sx)->own_integration = false;
    std::string legacy = session_to_json_tree(ts);
    const size_t at = legacy.find(",\"own_integration\":false", legacy.find("\"slice1d\""));
    require(at != std::string::npos, "slice writes own_integration");
    legacy.erase(at, std::string(",\"own_integration\":false").size());
    TreeSession c;
    c.load_from_record(make_record(), kVars, kParams);
    require(session_from_json_tree(legacy, c), "legacy tree JSON parses");
    require(c.tree.find(sx)->own_integration, "legacy slice keeps its own integrator");
    require(c.effective_shared(bif1d).l1d_transient_text == "100", "legacy slice keeps its values");
}

void layout_uses_one_tree_and_separate_workspaces() {
    TreeSession source, target;
    source.load_from_record(make_record(), kVars, kParams);
    target.load_from_record(make_record(), kVars, kParams);
    const int point = find_kind(source, TreeNodeKind::Point);
    const int basins = source.add_node(point, TreeNodeKind::Basins);
    const auto path = source.tree.layout_path(basins);
    require(path.size() == 3 && path.back().kind == TreeNodeKind::Basins, "layout stores the path to its leaf");
    const int target_point = find_kind(target, TreeNodeKind::Point);
    target.tree.find(target_point)->cfg.fix_x_value = 9.0;
    const size_t initial_nodes = target.tree.nodes.size();
    const int leaf = target.ensure_layout_leaf(path);
    require(leaf > 0 && target.tree.nodes.size() == initial_nodes + 1, "layout creates only the missing leaf");
    require(target.tree.find(leaf)->parent == target_point &&
            target.tree.find(target_point)->cfg.fix_x_value == 9.0, "layout reuses the existing context and settings");
    require(target.leaves.at(leaf).basins != nullptr && target.tree.needs_review(leaf), "new calculation has defaults and a review flag");
    require(target.ensure_layout_leaf(path) == leaf && target.tree.nodes.size() == initial_nodes + 1,
            "applying the same layout reuses the calculation");

    auto missing_branch = path;
    missing_branch.front().name = "Layout region";
    const int other = target.ensure_layout_leaf(missing_branch);
    require(other && other != leaf && target.tree.nodes.size() == initial_nodes + 4,
            "layout creates missing ancestors for a distinct branch");
    const int other_point = target.tree.find(other)->parent;
    target.tree.find(other)->needs_review = false;
    require(target.tree.needs_review(other) && target.tree.find(other_point)->needs_review,
            "unreviewed context still blocks the leaf");
    auto invalid = path;
    invalid[0].name = "Do not create";
    invalid[1].kind = TreeNodeKind::System;
    require(!target.ensure_layout_leaf(invalid) && target.tree.nodes.size() == initial_nodes + 4,
            "invalid layout path leaves the tree untouched");

    const auto key = std::to_string(leaf) + "w0";
    auto& tree = target.tree;
    tree.tabs.push_back({2, "Basins", false, {}, {}});
    tree.next_tab_id = 3;
    tree.active_tab_id = 2;
    require(tree.tabs.front().shows(key, true) && !tree.active_tab().shows(key, true),
            "legacy workspace shows enabled windows and new workspace starts empty");
    tree.active_tab().set_visible(key, true);
    tree.tabs.front().set_visible(key, false);
    require(!tree.tabs.front().shows(key, true) && tree.active_tab().shows(key, true),
            "closing one window copy leaves the other visible");
    require(tree.find(leaf)->enabled && target.leaves.at(leaf).basins != nullptr, "visibility leaves analysis alone");
    const auto dirty = target.dirty_leaves(tree.nodes.front().id);
    require(std::find(dirty.begin(), dirty.end(), leaf) == dirty.end(), "Run skips calculations awaiting review");

    const auto json = session_to_json_tree(target);
    TreeSession restored;
    restored.load_from_record(make_record(), kVars, kParams);
    require(session_from_json_tree(json, restored) && session_to_json_tree(restored) == json,
            "workspace order, selected tab, windows and review flags round-trip");
    require(restored.tree.active_tab().id == 2 && restored.tree.active_tab().shows(key, false) &&
            !restored.tree.tabs.front().shows(key, true) && restored.tree.needs_review(leaf),
            "workspace copies and review state survive reload");
    restored.tree.find(leaf)->needs_review = false;
    const auto ready = restored.dirty_leaves(restored.tree.nodes.front().id);
    require(std::find(ready.begin(), ready.end(), leaf) != ready.end(), "confirmed leaf can be queued");

    const std::string legacy = R"({"version":1,"next_id":2,"selected_id":1,"nodes":[
        {"id":1,"parent":0,"kind":"system","name":"System","children":[]}]})";
    require(session_from_json_tree(legacy, restored) && restored.tree.tabs.size() == 1 &&
            restored.tree.active_tab().id == 1 && restored.tree.active_tab().shows(key, true) &&
            !restored.tree.needs_review(1), "old sessions open on their original enabled-window workspace");
}

void closed_tree_windows_can_be_restored_and_extra_views_removed() {
    TreeSession ts;
    ts.load_from_record(make_record(), kVars, kParams);
    const int leaf = find_kind(ts, TreeNodeKind::LS2D);
    auto& tree = ts.tree;
    tree.tabs.push_back({2, "Copy", false, {}, {}});
    tree.tabs.back().set_visible(tree_leaf_window_key(leaf), true);

    tree.active_tab().set_visible(tree_leaf_window_key(leaf), false);
    require(!ts.leaf_window_visible(leaf) && tree.is_active(leaf), "closing a window leaves the calculation enabled");
    require(tree.tabs.back().shows(tree_leaf_window_key(leaf), true), "closing one copy leaves the other visible");
    require(ts.open_or_add_leaf_window(leaf) == 0 && ts.leaf_window_visible(leaf) &&
            tree.find(leaf)->extra_views.empty(), "New window restores the main view before adding a duplicate");
    require(ts.open_or_add_leaf_window(leaf) == 1 && ts.leaf_window_visible(leaf, 1) &&
            tree.find(leaf)->expanded, "adding a second view exposes its row in the tree");
    require(ts.open_or_add_leaf_window(leaf) == 2, "another extra view is added");
    tree.active_tab().set_visible(tree_leaf_window_key(leaf), false);
    require(!ts.workspace_window_visible(leaf, tree_leaf_window_key(leaf)) &&
            ts.workspace_window_visible(leaf, tree_leaf_window_key(leaf, 1)) &&
            tree.is_active(leaf), "turning the main window off leaves the extra window visible");
    ts.toggle_enabled(leaf);
    require(!ts.workspace_window_visible(leaf, tree_leaf_window_key(leaf, 1)),
            "disabling the calculation hides its windows");
    ts.toggle_enabled(leaf);
    require(ts.workspace_window_visible(leaf, tree_leaf_window_key(leaf, 1)) &&
            !ts.workspace_window_visible(leaf, tree_leaf_window_key(leaf)),
            "reenabling the calculation restores each window's on/off choice");

    tree.active_tab().set_visible(tree_leaf_window_key(leaf, 1), false);
    require(!ts.leaf_window_visible(leaf, 1) && ts.show_leaf_window(leaf, 1) &&
            ts.leaf_window_visible(leaf, 1), "the tree can restore a closed extra view");
    tree.find(leaf)->extra_views = {0, 2};
    tree.active_tab().set_visible(tree_leaf_window_key(leaf, 2), false);
    tree.tabs.back().set_visible(tree_leaf_window_key(leaf, 2), true);
    require(ts.remove_extra_view(leaf, 1) && tree.find(leaf)->extra_views == std::vector<int>{2},
            "the tree deletes a view row and preserves the remaining view settings");
    require(!ts.leaf_window_visible(leaf, 1) && tree.tabs.back().shows(tree_leaf_window_key(leaf, 1), true) &&
            !tree.tabs.back().shows(tree_leaf_window_key(leaf, 2), true),
            "removing a view preserves each tab's visibility for the remaining view");
    require(!ts.remove_extra_view(leaf, 0) && !ts.remove_extra_view(leaf, 2) && tree.is_active(leaf),
            "removing an invalid view leaves the main calculation intact");
}

void layout_copies_and_sweep_titles() {
    TreeSession ts;
    ts.load_from_record(make_record(), kVars, kParams);
    auto& tree = ts.tree;
    const int plane = find_kind(ts, TreeNodeKind::Bif2D);
    const int x = find_kind(ts, TreeNodeKind::Bif1D, 0);
    const int y = find_kind(ts, TreeNodeKind::Bif1D, 1);
    require(ts.sweep_window_label(plane) == "X: a, Y: b" &&
            ts.sweep_window_label(x) == "a (X)" && ts.sweep_window_label(y) == "b (Y)",
            "titles distinguish 2D sweep and both 1D slices");
    auto& region = *tree.find(find_kind(ts, TreeNodeKind::Region2D));
    region.cfg.axis_x_over_var = true;
    region.cfg.axis_x_var_index = 2;
    region.cfg.axis_y_over_h = true;
    require(ts.sweep_window_label(plane) == "X: z (IC), Y: h" &&
            ts.sweep_window_label(x) == "z (IC) (X)" && ts.sweep_window_label(y) == "h (Y)",
            "titles follow edits to initial-condition and step axes");

    const auto key = tree_leaf_window_key(plane);
    const auto nodes = tree.nodes.size();
    require(tree.register_layout_window(key) && !tree.active_tab().is_copy(key),
            "a first layout window is the original");
    tree.active_tab().set_visible(key, false);
    tree.tabs.push_back({2, "Copy", false, {}, {}});
    tree.active_tab_id = 2;
    tree.next_tab_id = 3;
    require(tree.register_layout_window(key) && tree.active_tab().is_copy(key) &&
            !tree.register_layout_window(key) && tree.nodes.size() == nodes,
            "the second tab gets one reusable copy of the same calculation even if its original is off");
    tree.active_tab().set_visible(key, true);
    const auto both_tabs = std::vector<std::size_t>{0, 1};
    require(tree.window_tabs(key, true) == both_tabs, "the shared tree lists the closed original and its open copy");
    tree.active_tab_id = 1;
    require(tree.window_tabs(key, true) == both_tabs, "switching tabs keeps both window rows in the shared tree");
    tree.active_tab_id = 2;
    const auto json = session_to_json_tree(ts);
    TreeSession restored;
    restored.load_from_record(make_record(), kVars, kParams);
    require(session_from_json_tree(json, restored) && session_to_json_tree(restored) == json &&
            restored.tree.active_tab().is_copy(key), "layout copies round-trip");
    require(restored.tree.active_tab().remove_copy(key) &&
            !restored.tree.active_tab().shows(key, true) &&
            restored.tree.window_tabs(key, true) == std::vector<std::size_t>{0} &&
            restored.tree.tabs.front().layout_windows == std::vector<std::string>{key},
            "deleting the copy preserves the original layout window");
}

} // namespace

int main() {
    try {
        default_tree_matches_custom();
        signatures_follow_the_cursor();
        structure_and_queue();
        json_round_trip();
        slice_integrator_inherits();
        layout_uses_one_tree_and_separate_workspaces();
        closed_tree_windows_can_be_restored_and_extra_views_removed();
        layout_copies_and_sweep_titles();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("tree: OK\n");
    return 0;
}
