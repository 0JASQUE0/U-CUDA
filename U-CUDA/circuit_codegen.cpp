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

    // Нулевую проводимость НЕ пропускаем: структура запекается в код, а число
    // приходит в рантайме. Выкинув клетку сейчас, мы потребовали бы перекомпиляции
    // от простой смены параметра системы.
    for (const CircuitResistor& r : lay.graph.resistors) s.block(r.a, r.b);
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

// plan_elimination и elimination_report — в sparse_lu.h.

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
