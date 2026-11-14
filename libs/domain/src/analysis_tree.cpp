#include "ucuda/domain/analysis_tree.h"
#include <functional>

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
