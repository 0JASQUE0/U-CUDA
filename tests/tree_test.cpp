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

} // namespace

int main() {
    try {
        default_tree_matches_custom();
        signatures_follow_the_cursor();
        structure_and_queue();
        json_round_trip();
        slice_integrator_inherits();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("tree: OK\n");
    return 0;
}
