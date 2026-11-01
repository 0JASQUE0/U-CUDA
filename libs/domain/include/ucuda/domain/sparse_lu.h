#pragma once
// Разреженная LU по структуре (этап 6 плана DAE): план исключения считается ОДИН РАЗ на хосте по
// портрету матрицы — без чисел, — а в ядро идёт прямолинейный код. Пользуются схемный решатель
// (circuit_codegen / circuit_emit, портрет из MnaLayout) и методы ДАУ в codegen.cpp (портрет из
// символьного якобиана). Header-only: codegen.cpp и пробники берут его без лишних .obj.

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Позиция в матрице.
struct MatEntry {
    int row = 0, col = 0;
};

struct EliminationStep {
    int pivot_row = 0;
    int pivot_col = 0;
    // Строки, которые этот шаг обновляет, и столбцы, по которым идёт обновление.
    std::vector<int> rows;
    std::vector<int> cols;
};

struct EliminationPlan {
    std::vector<EliminationStep> steps;
    int       n            = 0;
    long long nnz_original = 0;
    long long nnz_fill_in  = 0;
    long long nnz_factor   = 0;   // ненулей в L+U
    long long nnz_u        = 0;   // ненулей в U: её хранить обязаны, обратная подстановка читает
    long long nnz_l        = 0;   // ненулей в L: при исключении на [A|b] хранить НЕ надо
    long long peak_active  = 0;   // пик ненулей активной подматрицы
    long long mul_ops      = 0;   // умножений-сложений в факторизации
    std::vector<MatEntry> filled; // портрет ПОСЛЕ fill-in: эмиттер обязан объявить и их
    int       max_row_len  = 0;
    bool      diagonal_free = false;  // есть ли строки без диагонального элемента
};

// Жадный Марковиц по структуре. Диагональным пивотингом обойтись нельзя:
// строка-ограничение идеального ОУ (и связь ДАУ индекса >= 2) несёт клетки только на чужих
// столбцах, и её диагональ структурно нулевая.
// diag_first (может быть nullptr): строки i с diag_first[i] != 0 исключаются ПЕРВЫМИ и только по
// своей диагонали (порядок между ними — по Марковицу). У ДАУ это строки x матрицы I - h J: их
// диагональ ~1 при любом состоянии, а ведущий элемент выбирается по структуре, не по числам, —
// без этой фазы Марковиц мог взять клетку связи 2x маятника, которая проходит через ноль.
inline bool plan_elimination(const std::vector<MatEntry>& pattern, int n,
                             EliminationPlan& plan, std::string* err,
                             const std::vector<char>* diag_first = nullptr) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (n <= 0) return fail("sparse LU: empty matrix");

    std::vector<std::set<int>> row(static_cast<size_t>(n)), col(static_cast<size_t>(n));
    for (const MatEntry& e : pattern) {
        if (e.row < 0 || e.row >= n || e.col < 0 || e.col >= n)
            return fail("sparse LU: pattern entry out of range");
        row[(size_t)e.row].insert(e.col);
        col[(size_t)e.col].insert(e.row);
    }

    // Портрет ПОСЛЕ fill-in: row/col по ходу исключения опустошаются, а эмиттеру
    // нужны все клетки, которые хоть раз существовали.
    std::set<std::pair<int, int>> all_entries;
    for (const MatEntry& e : pattern) all_entries.insert(std::make_pair(e.row, e.col));

    plan = EliminationPlan();
    plan.n = n;
    plan.nnz_original = (long long)pattern.size();
    for (int i = 0; i < n; ++i)
        if (!row[(size_t)i].count(i)) { plan.diagonal_free = true; break; }

    std::vector<char> row_done(static_cast<size_t>(n), 0), col_done(static_cast<size_t>(n), 0);
    auto first_phase_left = [&]() {
        if (!diag_first) return false;
        for (int i = 0; i < n; ++i)
            if ((*diag_first)[(size_t)i] && !row_done[(size_t)i] && !col_done[(size_t)i]) return true;
        return false;
    };

    for (int step = 0; step < n; ++step) {
        // Жадный Марковиц: минимум (r-1)(c-1) по всем оставшимся клеткам (в первой фазе — только
        // по диагоналям строк diag_first).
        const bool phase1 = first_phase_left();
        long long best = -1;
        int bi = -1, bj = -1;
        for (int i = 0; i < n; ++i) {
            if (row_done[(size_t)i]) continue;
            if (phase1 && !(*diag_first)[(size_t)i]) continue;
            const long long r = (long long)row[(size_t)i].size();
            for (int j : row[(size_t)i]) {
                if (col_done[(size_t)j]) continue;
                if (phase1 && j != i) continue;
                const long long c = (long long)col[(size_t)j].size();
                const long long m = (r - 1) * (c - 1);
                if (best < 0 || m < best) { best = m; bi = i; bj = j; }
            }
        }
        if (bi < 0) return fail("sparse LU: matrix is structurally singular - "
                                "a row or column has no remaining entry");

        EliminationStep es;
        es.pivot_row = bi;
        es.pivot_col = bj;
        for (int j : row[(size_t)bi]) if (!col_done[(size_t)j] && j != bj) es.cols.push_back(j);
        for (int i : col[(size_t)bj]) if (!row_done[(size_t)i] && i != bi) es.rows.push_back(i);

        plan.mul_ops   += (long long)es.rows.size() * (long long)es.cols.size();
        plan.nnz_factor += (long long)es.rows.size() + (long long)es.cols.size() + 1;
        plan.nnz_u      += (long long)es.cols.size() + 1;
        plan.nnz_l      += (long long)es.rows.size();
        plan.max_row_len = std::max(plan.max_row_len, (int)row[(size_t)bi].size());

        for (int i : es.rows) {
            for (int j : es.cols) {
                if (row[(size_t)i].insert(j).second) {
                    col[(size_t)j].insert(i);
                    all_entries.insert(std::make_pair(i, j));
                    ++plan.nnz_fill_in;
                }
            }
        }

        row_done[(size_t)bi] = 1;
        col_done[(size_t)bj] = 1;
        {
            long long active = 0;
            for (int i = 0; i < n; ++i)
                if (!row_done[(size_t)i])
                    for (int j : row[(size_t)i]) if (!col_done[(size_t)j]) ++active;
            plan.peak_active = std::max(plan.peak_active, active);
        }
        for (int j : row[(size_t)bi]) col[(size_t)j].erase(bi);
        for (int i : col[(size_t)bj]) row[(size_t)i].erase(bj);
        plan.steps.push_back(std::move(es));
    }
    for (const std::pair<int, int>& e : all_entries)
        plan.filled.push_back(MatEntry{ e.first, e.second });
    return true;
}

inline std::string elimination_report(const EliminationPlan& p) {
    std::ostringstream o;
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "  matrix %dx%d, density %.1f%%\n"
        "  nonzeros: %lld original + %lld fill-in\n"
        "  factor holds %lld scalars (U %lld + L %lld), %lld multiply-adds\n"
        "  peak active submatrix %lld, longest pivot row %d, zero diagonal: %s\n"
        "  working set on [A|b]: U %lld + rhs %d = %lld doubles = %lld registers\n",
        p.n, p.n, 100.0 * (double)p.nnz_original / ((double)p.n * p.n),
        p.nnz_original, p.nnz_fill_in, p.nnz_factor, p.nnz_u, p.nnz_l, p.mul_ops,
        p.peak_active, p.max_row_len, p.diagonal_free ? "yes" : "no",
        p.nnz_u, p.n, p.nnz_u + p.n, 2 * (p.nnz_u + p.n));
    o << buf;
    return o.str();
}
