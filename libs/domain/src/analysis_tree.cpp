#include "ucuda/domain/analysis_tree.h"
#include <algorithm>
#include <functional>

std::string tree_leaf_window_key(int leaf, int view, char type) {
    return std::to_string(leaf) + type + std::to_string(view);
}

bool TreeWorkspaceTab::shows(const std::string& key, bool enabled) const {
    const auto has = [&key](const auto& keys) {
        return std::find(keys.begin(), keys.end(), key) != keys.end();
    };
    return has(shown_windows) || (enabled && show_enabled_windows && !has(hidden_windows));
}

bool TreeWorkspaceTab::owns_window(const std::string& key, bool available) const {
    return shows(key, available) ||
           std::find(layout_windows.begin(), layout_windows.end(), key) != layout_windows.end();
}

bool TreeWorkspaceTab::remember_window(const std::string& key) {
    if (std::find(layout_windows.begin(), layout_windows.end(), key) != layout_windows.end()) return false;
    layout_windows.push_back(key);
    return true;
}

bool TreeWorkspaceTab::set_visible(const std::string& key, bool visible) {
    auto& add = visible ? shown_windows : hidden_windows;
    auto& remove = visible ? hidden_windows : shown_windows;
    remove.erase(std::remove(remove.begin(), remove.end(), key), remove.end());
    if (std::find(add.begin(), add.end(), key) != add.end()) return false;
    add.push_back(key);
    return true;
}

bool TreeWorkspaceTab::is_copy(const std::string& key) const {
    return std::find(layout_copies.begin(), layout_copies.end(), key) != layout_copies.end();
}

bool TreeWorkspaceTab::remove_copy(const std::string& key) {
    const auto copy = std::find(layout_copies.begin(), layout_copies.end(), key);
    if (copy == layout_copies.end()) return false;
    layout_copies.erase(copy);
    layout_windows.erase(std::remove(layout_windows.begin(), layout_windows.end(), key), layout_windows.end());
    set_visible(key, false);
    return true;
}

bool AnalysisTree::register_layout_window(const std::string& key) {
    auto& active = active_tab();
    if (!active.remember_window(key)) return false;
    for (const auto& tab : tabs) {
        if (tab.id == active.id || !tab.owns_window(key, true)) continue;
        active.layout_copies.push_back(key);
        break;
    }
    return true;
}

std::vector<std::size_t> AnalysisTree::window_tabs(const std::string& key, bool available) const {
    std::vector<std::size_t> owners;
    for (std::size_t i = 0; i < tabs.size(); ++i)
        if (tabs[i].owns_window(key, available)) owners.push_back(i);
    if (owners.empty() && !tabs.empty()) {
        const auto first = std::min_element(tabs.begin(), tabs.end(), [](const auto& a, const auto& b) {
            return a.id < b.id;
        });
        owners.push_back(static_cast<std::size_t>(first - tabs.begin()));
    }
    return owners;
}

TreeWorkspaceTab& AnalysisTree::active_tab() {
    return const_cast<TreeWorkspaceTab&>(static_cast<const AnalysisTree&>(*this).active_tab());
}

const TreeWorkspaceTab& AnalysisTree::active_tab() const {
    for (const auto& tab : tabs)
        if (tab.id == active_tab_id) return tab;
    return tabs.front();
}

std::vector<TreeLayoutNode> AnalysisTree::layout_path(int id) const {
    std::vector<TreeLayoutNode> path;
    for (const TreeNode* n = find(id); n && n->parent; n = find(n->parent)) {
        const TreeNode* parent = find(n->parent);
        if (!parent) return {};
        TreeLayoutNode part{n->kind, n->name, n->slice_dir};
        for (int sibling : parent->children) {
            if (sibling == n->id) break;
            const auto* other = find(sibling);
            if (other && other->kind == part.kind && other->name == part.name &&
                other->slice_dir == part.slice_dir) ++part.occurrence;
        }
        path.push_back(std::move(part));
    }
    std::reverse(path.begin(), path.end());
    return path;
}

bool AnalysisTree::needs_review(int id) const {
    for (const TreeNode* n = find(id); n; n = find(n->parent))
        if (n->needs_review) return true;
    return false;
}

namespace {
struct KindInfo { TreeNodeKind kind; const char* key; const char* title; int slot; };
const KindInfo kKinds[] = {
    { TreeNodeKind::System,   "system",   "System",                 0 },
    { TreeNodeKind::Region2D, "region2d", "Region 2D",              1 },
    { TreeNodeKind::Slice1D,  "slice1d",  "Slice 1D",               2 },
    { TreeNodeKind::Point,    "point",    "Point",                  3 },
    { TreeNodeKind::Bif2D,    "bif2d",    "Bifurcation 2D",         1 },
    { TreeNodeKind::LLE2D,    "lle2d",    "LLE 2D",                 1 },
    { TreeNodeKind::LS2D,     "ls2d",     "Lyapunov Spectrum 2D",   1 },
    { TreeNodeKind::Bif1D,    "bif1d",    "Bifurcation 1D",         2 },
    { TreeNodeKind::LLE1D,    "lle1d",    "LLE 1D",                 2 },
    { TreeNodeKind::LS1D,     "ls1d",     "Lyapunov Spectrum 1D",   2 },
    { TreeNodeKind::Phase,    "phase",    "Phase",                  3 },
    { TreeNodeKind::Basins,   "basins",   "Basins",                 3 },
};
const KindInfo& info(TreeNodeKind k) {
    for (const auto& i : kKinds) if (i.kind == k) return i;
    return kKinds[0];
}
} // namespace

const char* tree_kind_key(TreeNodeKind k)   { return info(k).key; }
const char* tree_kind_title(TreeNodeKind k) { return info(k).title; }
int tree_kind_dim_slot(TreeNodeKind k)      { return info(k).slot; }

bool tree_kind_from_key(const std::string& key, TreeNodeKind& out) {
    for (const auto& i : kKinds)
        if (key == i.key) { out = i.kind; return true; }
    return false;
}

TreeNode* AnalysisTree::find(int id) {
    for (auto& n : nodes) if (n.id == id) return &n;
    return nullptr;
}
const TreeNode* AnalysisTree::find(int id) const {
    for (const auto& n : nodes) if (n.id == id) return &n;
    return nullptr;
}

int AnalysisTree::level(int id) const {
    int lv = 0;
    const TreeNode* n = find(id);
    while (n && n->parent != 0) { ++lv; n = find(n->parent); }
    return lv;
}

bool AnalysisTree::is_active(int id) const {
    for (const TreeNode* n = find(id); n; n = n->parent ? find(n->parent) : nullptr)
        if (!n->enabled) return false;
    return find(id) != nullptr;
}

int AnalysisTree::nearest(int id, TreeNodeKind k) const {
    for (const TreeNode* n = find(id); n; n = n->parent ? find(n->parent) : nullptr)
        if (n->kind == k) return n->id;
    return 0;
}

std::vector<int> AnalysisTree::subtree(int id) const {
    std::vector<int> out;
    std::function<void(int)> walk = [&](int cur) {
        const TreeNode* n = find(cur);
        if (!n) return;
        out.push_back(cur);
        for (int c : n->children) walk(c);
    };
    walk(id);
    return out;
}

std::vector<int> AnalysisTree::preorder() const {
    if (nodes.empty()) return {};
    return subtree(nodes.front().id);
}

std::vector<TreeNodeKind> tree_addable_children(TreeNodeKind parent) {
    using K = TreeNodeKind;
    switch (parent) {
    case K::System:   return { K::Region2D, K::Slice1D, K::Point };
    // Вложенный регион вернётся вместе с «New region from view»: без неё он не отличается
    // от второго региона под System.
    case K::Region2D: return { K::Bif2D, K::LLE2D, K::LS2D, K::Slice1D, K::Point };
    case K::Slice1D:  return { K::Bif1D, K::LLE1D, K::LS1D, K::Point };
    case K::Point:    return { K::Phase, K::Basins };
    default:          return {};
    }
}
