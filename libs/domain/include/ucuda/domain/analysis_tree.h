#pragma once
#include "ucuda/domain/custom_session.h"
#include <string>
#include <vector>

// Вкладка Tree: дерево анализа. Контексты (System, Region2D, Slice1D, Point) задают область
// параметров и ничего не считают; листья считают через свои подсессии (analysis/tree_session.h).
// Эффективный контекст листа собирается в CustomTabSharedConfig, чтобы расчёт шёл теми же
// apply_shared_to_*, что и в Custom.

enum class TreeNodeKind {
    System = 0, Region2D, Slice1D, Point,
    Bif2D, LLE2D, LS2D, Bif1D, LLE1D, LS1D, Phase, Basins
};

[[nodiscard]] inline bool tree_kind_is_context(TreeNodeKind k) {
    return k == TreeNodeKind::System || k == TreeNodeKind::Region2D
        || k == TreeNodeKind::Slice1D || k == TreeNodeKind::Point;
}
[[nodiscard]] inline bool tree_kind_is_2d_leaf(TreeNodeKind k) {
    return k == TreeNodeKind::Bif2D || k == TreeNodeKind::LLE2D || k == TreeNodeKind::LS2D;
}
[[nodiscard]] inline bool tree_kind_is_1d_leaf(TreeNodeKind k) {
    return k == TreeNodeKind::Bif1D || k == TreeNodeKind::LLE1D || k == TreeNodeKind::LS1D;
}
[[nodiscard]] inline bool tree_kind_is_0d_leaf(TreeNodeKind k) {
    return k == TreeNodeKind::Phase || k == TreeNodeKind::Basins;
}

// Строковые ключи — для JSON и подписей; порядок enum в файл не пишется.
[[nodiscard]] const char* tree_kind_key(TreeNodeKind k);
[[nodiscard]] bool tree_kind_from_key(const std::string& key, TreeNodeKind& out);
// Имя по умолчанию (английское, для UI).
[[nodiscard]] const char* tree_kind_title(TreeNodeKind k);
// Слот размерности для цвета: 0 System, 1 — 2D, 2 — 1D, 3 — 0D. Лист красится по родителю.
[[nodiscard]] int tree_kind_dim_slot(TreeNodeKind k);

struct TreeNode {
    int          id     = 0;
    int          parent = 0;          // 0 у корня
    TreeNodeKind kind   = TreeNodeKind::System;
    std::string  name;
    bool         enabled      = true;
    bool         expanded     = true;
    bool         was_expanded = true;
    // Slice1D и Point: закрепление из курсора региона; false — свои пины в cfg.fix_x/fix_y.
    bool         follow_cursor = true;
    // Slice1D и Point: листья пересчитываются сами после движения курсора.
    bool         live = true;
    // Slice1D: 0 — вдоль X региона, 1 — вдоль Y.
    int          slice_dir = 0;
    // Поля контекста; смысл зависит от kind (см. effective_shared в tree_session.cpp):
    //   System   — интегратор, НУ, параметры, pre_scaller, max_value;
    //   Region2D — axis_x_*, axis_y_*, resolution_text, bif_writable_var, курсор fix_x/fix_y;
    //   Slice1D  — inherit_sweep_from_2d, своя ось в sweep_x_*, n_x_1d_text, l1d_*,
    //              continuation_1d_enabled, plot_inter_peaks_1d; курсор вдоль оси в fix_x
    //              (срез под System) или пин второй оси (follow_cursor = false);
    //   Point    — пины fix_x/fix_y при follow_cursor = false.
    CustomTabSharedConfig cfg;
    std::vector<int> children;
};

struct AnalysisTree {
    std::vector<TreeNode> nodes;      // корень — nodes[0], id = 1
    int next_id     = 1;
    int selected_id = 0;

    [[nodiscard]] TreeNode*       find(int id);
    [[nodiscard]] const TreeNode* find(int id) const;
    [[nodiscard]] int  level(int id) const;            // глубина: корень 0
    [[nodiscard]] bool is_active(int id) const;        // узел и все предки включены
    // Ближайший предок (включая сам узел) данного вида; 0 — нет.
    [[nodiscard]] int  nearest(int id, TreeNodeKind k) const;
    // Поддерево в порядке обхода (сам узел первым).
    [[nodiscard]] std::vector<int> subtree(int id) const;
    // Все узлы в порядке обхода от корня.
    [[nodiscard]] std::vector<int> preorder() const;
};

// Допустимые дети для «+» (раздел 6.2 спецификации).
[[nodiscard]] std::vector<TreeNodeKind> tree_addable_children(TreeNodeKind parent);
