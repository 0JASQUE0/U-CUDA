#include "circuit_codegen.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>

namespace {

// Тот же обход, что в CircuitSolver::assemble, но пишутся только ПОЗИЦИИ.
// Держать эти два места синхронными обязан circuit_codegen_selftest.
struct PatternSink {
    const MnaLayout*      lay;
    std::vector<MatEntry>* out;

    void put(int r, int c) {
        if (r >= 0 && c >= 0) out->push_back(MatEntry{ r, c });
    }
    void block(int a, int b) {           // симметричный штамп двухполюсника
        const int ia = lay->idx(a), ib = lay->idx(b);
        put(ia, ia); put(ia, ib); put(ib, ia); put(ib, ib);
    }
};

}  // namespace

void mna_pattern(const MnaLayout& lay, std::vector<MatEntry>& out) {
    out.clear();
    PatternSink s{ &lay, &out };

    for (const CircuitResistor& r : lay.graph.resistors) {
        if (r.siemens == 0.0) continue;   // разрыв: клетки не появляются вовсе
        s.block(r.a, r.b);
    }
    for (const MnaLayout::Cap& c : lay.caps) s.block(c.a, c.b);

    for (size_t k = 0; k < lay.branches.size(); ++k) {
        const MnaLayout::Branch& br = lay.branches[k];
        const int row = lay.n_v + (int)k;

        s.put(lay.idx(br.out), row);

        switch (br.kind) {
        case MnaLayout::BranchKind::VSource:
            s.put(row, lay.idx(lay.graph.vsources[(size_t)br.elem].node));
            break;
        case MnaLayout::BranchKind::OpAmpIdeal: {
            const CircuitOpAmp& o = lay.graph.opamps[(size_t)br.elem];
            s.put(row, lay.idx(o.in_plus));
            s.put(row, lay.idx(o.in_minus));
            break;
        }
        case MnaLayout::BranchKind::OpAmpReal: {
            const CircuitOpAmp& o = lay.graph.opamps[(size_t)br.elem];
            s.put(lay.idx(br.pole), lay.idx(o.in_plus));
            s.put(lay.idx(br.pole), lay.idx(o.in_minus));
            s.put(lay.idx(br.pole), lay.idx(br.pole));
            s.put(row, lay.idx(br.out));
            s.put(row, lay.idx(br.pole));
            break;
        }
        case MnaLayout::BranchKind::Multiplier: {
            const CircuitMultiplier& m = lay.graph.multipliers[(size_t)br.elem];
            s.put(row, lay.idx(m.out));
            s.put(row, lay.idx(m.x1)); s.put(row, lay.idx(m.x2));
            s.put(row, lay.idx(m.y1)); s.put(row, lay.idx(m.y2));
            s.put(row, lay.idx(m.z));
            break;
        }
        }
    }

    std::sort(out.begin(), out.end(), [](const MatEntry& a, const MatEntry& b) {
        return a.row != b.row ? a.row < b.row : a.col < b.col;
    });
    out.erase(std::unique(out.begin(), out.end(), [](const MatEntry& a, const MatEntry& b) {
        return a.row == b.row && a.col == b.col;
    }), out.end());
}

bool plan_elimination(const std::vector<MatEntry>& pattern, int n,
                      EliminationPlan& plan, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (n <= 0) return fail("circuit codegen: empty matrix");

    std::vector<std::set<int>> row(static_cast<size_t>(n)), col(static_cast<size_t>(n));
    for (const MatEntry& e : pattern) {
        if (e.row < 0 || e.row >= n || e.col < 0 || e.col >= n)
            return fail("circuit codegen: pattern entry out of range");
        row[(size_t)e.row].insert(e.col);
        col[(size_t)e.col].insert(e.row);
    }

    plan = EliminationPlan();
    plan.n = n;
    plan.nnz_original = (long long)pattern.size();
    for (int i = 0; i < n; ++i)
        if (!row[(size_t)i].count(i)) { plan.diagonal_free = true; break; }

    std::vector<char> row_done(static_cast<size_t>(n), 0), col_done(static_cast<size_t>(n), 0);

    for (int step = 0; step < n; ++step) {
        // Жадный Марковиц: минимум (r-1)(c-1) по всем оставшимся клеткам.
        long long best = -1;
        int bi = -1, bj = -1;
        for (int i = 0; i < n; ++i) {
            if (row_done[(size_t)i]) continue;
            const long long r = (long long)row[(size_t)i].size();
            for (int j : row[(size_t)i]) {
                if (col_done[(size_t)j]) continue;
                const long long c = (long long)col[(size_t)j].size();
                const long long m = (r - 1) * (c - 1);
                if (best < 0 || m < best) { best = m; bi = i; bj = j; }
            }
        }
        if (bi < 0) return fail("circuit codegen: matrix is structurally singular - "
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
    return true;
}

std::string elimination_report(const EliminationPlan& p) {
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

// --- самопроверка -------------------------------------------------------------

std::string circuit_codegen_selftest(bool* ok) {
    std::ostringstream o;
    bool good = true;
    auto check = [&](bool cond, const std::string& what) {
        o << (cond ? "  ok   " : "  FAIL ") << what << "\n";
        if (!cond) good = false;
    };

    System sys;
    sys.vars   = { "u", "v", "w" };
    sys.params = { "sigma", "r", "b" };
    sys.rhs    = { "sigma*(v - u)", "r*u - v - 20*u*w", "5*u*v - b*w" };
    const std::vector<double> vals = { 0.0, 16.0, 45.6, 4.0 };

    PolyExtractConfig pc;
    std::vector<PolyRhs> poly;
    std::string err;
    if (!extract_quadratic(sys, vals, pc, poly, &err)) { if (ok) *ok = false; return "  FAIL " + err + "\n"; }
    SynthesisConfig sc;
    CircuitGraph g;
    if (!synthesize_circuit(sys, poly, sc, g, &err)) { if (ok) *ok = false; return "  FAIL " + err + "\n"; }

    for (int mode = 0; mode < 2; ++mode) {
        CircuitSolverConfig cfg;
        cfg.opamp.ideal = (mode == 0);
        o << (cfg.opamp.ideal ? "ideal op-amps:\n" : "one-pole op-amps:\n");

        MnaLayout lay;
        if (!build_mna_layout(g, cfg, lay, &err)) { check(false, err); continue; }

        std::vector<MatEntry> pat;
        mna_pattern(lay, pat);

        EliminationPlan plan;
        if (!plan_elimination(pat, lay.size, plan, &err)) { check(false, err); continue; }
        o << elimination_report(plan);

        // Портрет обязан покрывать то, что реально штампует решатель.
        CircuitSolver s;
        if (!s.build(g, cfg, &err)) { check(false, err); continue; }
        s.set_initial_variables({ 0.1, 0.1, 0.1 });
        for (int i = 0; i < 200; ++i) {                 // подальше от нулевого состояния
            if (!s.step(1.0e-4 / g.time_scale, &err)) { check(false, err); break; }
        }
        std::vector<char> covered((size_t)lay.size * lay.size, 0);
        for (const MatEntry& e : pat) covered[(size_t)e.row * lay.size + e.col] = 1;

        int missing = 0;
        const std::vector<double>& J = s.debug_assemble_jacobian(1.0e-4 / g.time_scale);
        for (int i = 0; i < lay.size; ++i)
            for (int j = 0; j < lay.size; ++j)
                if (J[(size_t)i * lay.size + j] != 0.0 && !covered[(size_t)i * lay.size + j])
                    ++missing;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "pattern covers every stamped entry (%d missed)", missing);
        check(missing == 0, buf);
    }

    if (ok) *ok = good;
    o << (good ? "CODEGEN SELFTEST PASSED\n" : "CODEGEN SELFTEST FAILED\n");
    return o.str();
}
