#include "circuit_codegen.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>

// Печать шага схемного решателя прямолинейным кодом.
//
// Обход обязан повторять CircuitSolver::assemble операция в операцию: это ТРЕТЬЕ
// место с той же логикой штампов (после assemble и mna_pattern), и разъехаться им
// не даёт только сверка GPU против CPU на идентичных входах. Любая правка штампа
// правится во всех трёх.

namespace {

struct Emitter {
    const MnaLayout*  lay;
    std::ostringstream o;
    std::set<std::pair<int, int>> declared;   // объявленные клетки матрицы
    std::vector<double>* comp;

    // Имя неизвестной. Земля печатается нулём, а не переменной.
    std::string V(int node) const {
        const int i = lay->idx(node);
        if (i < 0) return "0.0";
        char b[32]; std::snprintf(b, sizeof(b), "v%d", i);
        return b;
    }
    static std::string A(int r, int c) {
        char b[48]; std::snprintf(b, sizeof(b), "a%d_%d", r, c);
        return b;
    }
    static std::string F(int r) {
        char b[32]; std::snprintf(b, sizeof(b), "f%d", r);
        return b;
    }
    // Разность двух узлов с подавлением слагаемого-земли: "(v3 - 0.0)" читается плохо
    // и мешает компилятору не больше, чем нам, но текст короче в разы.
    std::string diff(int a, int b) const {
        const std::string va = V(a), vb = V(b);
        if (vb == "0.0") return va;
        if (va == "0.0") return "(-" + vb + ")";
        return "(" + va + " - " + vb + ")";
    }

    void addj(int r, int c, const std::string& e) {
        if (r < 0 || c < 0) return;
        if (!declared.count(std::make_pair(r, c))) return;   // клетки нет в портрете
        o << "        " << A(r, c) << " += " << e << ";\n";
    }
    void addf(int r, const std::string& e) {
        if (r < 0) return;
        o << "        " << F(r) << " += " << e << ";\n";
    }
    int push_comp(double v) {
        comp->push_back(v);
        return (int)comp->size() - 1;
    }
};

std::string cdbl(double v) {
    char b[40];
    std::snprintf(b, sizeof(b), "%.17g", v);
    std::string s = b;
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos
        && s.find("inf") == std::string::npos && s.find("nan") == std::string::npos)
        s += ".0";
    return s;
}

}  // namespace

bool emit_circuit_step(const MnaLayout& lay, const EliminationPlan& plan,
                       const EmitConfig& ec, EmittedKernel& out, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (plan.n != lay.size) return fail("circuit emit: elimination plan does not match the layout");
    if (ec.newton_iters < 1) return fail("circuit emit: at least one Newton iteration is required");

    out = EmittedKernel();
    out.size   = lay.size;
    out.n_caps = (int)lay.caps.size();
    out.n_vars = (int)lay.graph.var_node.size();

    Emitter em;
    em.lay  = &lay;
    em.comp = &out.comp;
    for (const MatEntry& e : plan.filled) em.declared.insert(std::make_pair(e.row, e.col));

    // --- упаковка номиналов: порядок ФИКСИРОВАН и повторяется хостом -------------
    std::vector<int> r_comp, c_comp;
    for (const CircuitResistor& r : lay.graph.resistors) r_comp.push_back(em.push_comp(r.siemens));
    for (const MnaLayout::Cap& c : lay.caps)             c_comp.push_back(em.push_comp(c.farad));
    int gm_c = -1, gp_c = -1, vsat_c = -1;
    bool has_real = false;
    for (const MnaLayout::Branch& b : lay.branches)
        if (b.kind == MnaLayout::BranchKind::OpAmpReal) has_real = true;
    if (has_real) {
        gm_c   = em.push_comp(lay.gm);
        gp_c   = em.push_comp(1.0 / lay.rp);
        vsat_c = em.push_comp(0.0);   // заполняется хостом из OpAmpModel::vsat
    }
    out.n_comp = (int)out.comp.size();

    std::ostringstream& o = em.o;

    // --- одна итерация Ньютона ---------------------------------------------------
    o << "    for (int it = 0; it < " << ec.newton_iters << "; ++it) {\n";

    o << "        // сборка: только клетки портрета, матрицы как массива нет\n";
    for (const MatEntry& e : plan.filled)
        o << "        double " << Emitter::A(e.row, e.col) << " = 0.0;\n";
    for (int r = 0; r < lay.size; ++r)
        o << "        double " << Emitter::F(r) << " = 0.0;\n";

    for (size_t i = 0; i < lay.graph.resistors.size(); ++i) {
        const CircuitResistor& r = lay.graph.resistors[i];
        const int ia = lay.idx(r.a), ib = lay.idx(r.b);
        char g[32]; std::snprintf(g, sizeof(g), "g%d", (int)i);
        o << "        const double " << g << " = comp[" << r_comp[i] << "];\n";
        o << "        const double i" << g << " = " << g << " * " << em.diff(r.a, r.b) << ";\n";
        em.addf(ia, std::string("i") + g);
        em.addf(ib, std::string("-i") + g);
        em.addj(ia, ia, g); em.addj(ia, ib, std::string("-") + g);
        em.addj(ib, ia, std::string("-") + g); em.addj(ib, ib, g);
    }

    for (size_t i = 0; i < lay.caps.size(); ++i) {
        const MnaLayout::Cap& c = lay.caps[i];
        const int ia = lay.idx(c.a), ib = lay.idx(c.b);
        char cg[32]; std::snprintf(cg, sizeof(cg), "cg%d", (int)i);
        o << "        const double " << cg << " = alpha * comp[" << c_comp[i] << "];\n";
        o << "        const double ic" << i << " = " << cg << " * " << em.diff(c.a, c.b)
          << " - hist" << i << ";\n";
        em.addf(ia, "ic" + std::to_string(i));
        em.addf(ib, "-ic" + std::to_string(i));
        em.addj(ia, ia, cg); em.addj(ia, ib, std::string("-") + cg);
        em.addj(ib, ia, std::string("-") + cg); em.addj(ib, ib, cg);
    }

    for (size_t k = 0; k < lay.branches.size(); ++k) {
        const MnaLayout::Branch& br = lay.branches[k];
        const int row = lay.n_v + (int)k;
        // Ток ветви — это неизвестная с индексом row, то есть v{row}.
        em.addf(lay.idx(br.out), "v" + std::to_string(row));
        em.addj(lay.idx(br.out), row, "1.0");

        switch (br.kind) {
        case MnaLayout::BranchKind::VSource: {
            const CircuitVSource& s = lay.graph.vsources[(size_t)br.elem];
            o << "        " << Emitter::F(row) << " += " << em.V(s.node) << " - "
              << cdbl(s.volt) << ";\n";
            em.addj(row, lay.idx(s.node), "1.0");
            break;
        }
        case MnaLayout::BranchKind::OpAmpIdeal: {
            const CircuitOpAmp& a = lay.graph.opamps[(size_t)br.elem];
            o << "        " << Emitter::F(row) << " += " << em.diff(a.in_plus, a.in_minus) << ";\n";
            em.addj(row, lay.idx(a.in_plus),  "1.0");
            em.addj(row, lay.idx(a.in_minus), "-1.0");
            break;
        }
        case MnaLayout::BranchKind::OpAmpReal: {
            const CircuitOpAmp& a = lay.graph.opamps[(size_t)br.elem];
            o << "        const double gin" << k << " = comp[" << gm_c << "] * "
              << em.diff(a.in_plus, a.in_minus) << ";\n";
            em.addf(lay.idx(br.pole), "-gin" + std::to_string(k));
            em.addj(lay.idx(br.pole), lay.idx(a.in_plus),  std::string("-comp[") + std::to_string(gm_c) + "]");
            em.addj(lay.idx(br.pole), lay.idx(a.in_minus), std::string("comp[") + std::to_string(gm_c) + "]");
            em.addf(lay.idx(br.pole), "comp[" + std::to_string(gp_c) + "] * " + em.V(br.pole));
            em.addj(lay.idx(br.pole), lay.idx(br.pole), std::string("comp[") + std::to_string(gp_c) + "]");
            o << "        const double th" << k << " = tanh(" << em.V(br.pole)
              << " / comp[" << vsat_c << "]);\n";
            o << "        " << Emitter::F(row) << " += " << em.V(br.out) << " - comp["
              << vsat_c << "] * th" << k << ";\n";
            em.addj(row, lay.idx(br.out),  "1.0");
            em.addj(row, lay.idx(br.pole), "-(1.0 - th" + std::to_string(k) + " * th"
                                            + std::to_string(k) + ")");
            break;
        }
        case MnaLayout::BranchKind::Multiplier: {
            const CircuitMultiplier& m = lay.graph.multipliers[(size_t)br.elem];
            o << "        const double dx" << k << " = " << em.diff(m.x1, m.x2) << ";\n";
            o << "        const double dy" << k << " = " << em.diff(m.y1, m.y2) << ";\n";
            o << "        " << Emitter::F(row) << " += " << em.V(m.out) << " - (dx" << k
              << " * dy" << k << " * 0.1 + " << em.V(m.z) << ");\n";
            em.addj(row, lay.idx(m.out), "1.0");
            em.addj(row, lay.idx(m.x1), "-dy" + std::to_string(k) + " * 0.1");
            em.addj(row, lay.idx(m.x2),  "dy" + std::to_string(k) + " * 0.1");
            em.addj(row, lay.idx(m.y1), "-dx" + std::to_string(k) + " * 0.1");
            em.addj(row, lay.idx(m.y2),  "dx" + std::to_string(k) + " * 0.1");
            em.addj(row, lay.idx(m.z),  "-1.0");
            break;
        }
        }
    }

    // Решаем J*d = -F, поэтому невязку переворачиваем один раз здесь.
    o << "\n        // J * d = -F\n";
    for (int r = 0; r < lay.size; ++r) o << "        " << Emitter::F(r) << " = -" << Emitter::F(r) << ";\n";

    // --- исключение по статическому порядку --------------------------------------
    o << "\n        // прямой ход: порядок и позиции fill-in посчитаны на хосте\n";
    for (const EliminationStep& st : plan.steps) {
        const std::string piv = Emitter::A(st.pivot_row, st.pivot_col);
        if (st.rows.empty()) {
            o << "        if (fabs(" << piv << ") < " << cdbl(ec.pivot_min) << ") bad = 1;\n";
            continue;
        }
        o << "        if (fabs(" << piv << ") < " << cdbl(ec.pivot_min) << ") bad = 1;\n";
        o << "        { const double ip = 1.0 / " << piv << ";\n";
        for (int i : st.rows) {
            o << "          const double m" << i << " = " << Emitter::A(i, st.pivot_col) << " * ip;\n";
            for (int j : st.cols)
                o << "          " << Emitter::A(i, j) << " -= m" << i << " * "
                  << Emitter::A(st.pivot_row, j) << ";\n";
            o << "          " << Emitter::F(i) << " -= m" << i << " * " << Emitter::F(st.pivot_row) << ";\n";
        }
        o << "        }\n";
    }

    // --- обратный ход -------------------------------------------------------------
    o << "\n        // обратный ход в обратном порядке шагов\n";
    for (int i = 0; i < lay.size; ++i) o << "        double d" << i << ";\n";
    for (size_t s = plan.steps.size(); s-- > 0;) {
        const EliminationStep& st = plan.steps[s];
        o << "        d" << st.pivot_col << " = (" << Emitter::F(st.pivot_row);
        for (int j : st.cols) o << " - " << Emitter::A(st.pivot_row, j) << " * d" << j;
        o << ") / " << Emitter::A(st.pivot_row, st.pivot_col) << ";\n";
    }

    o << "\n        // limiting: потолок поправки за итерацию\n";
    for (int i = 0; i < lay.size; ++i) {
        o << "        d" << i << " = fmin(fmax(d" << i << ", " << cdbl(-ec.newton_max_step)
          << "), " << cdbl(ec.newton_max_step) << ");\n";
        o << "        v" << i << " += d" << i << ";\n";
    }
    o << "    }\n";

    // --- обвязка потока: объявления, НУ, шаг по времени, история, выборка --------
    //
    // Печатается здесь, а не в шаблоне, потому что зависит от раскладки: шаблон,
    // знающий про индексы узлов, разъехался бы с MnaLayout при первой же правке.
    std::ostringstream pro, epi;

    pro << "    // неизвестные скалярами, а не массивом: только так они лягут в регистры\n";
    for (int i = 0; i < lay.size; ++i) pro << "    double v" << i << " = 0.0;\n";
    for (size_t i = 0; i < lay.caps.size(); ++i)
        pro << "    double q" << i << " = 0.0, q2_" << i << " = 0.0;\n";

    pro << "\n    // НУ садятся на ЗАРЯД: выход интегратора держит ОУ, свободной переменной он не является\n";
    for (size_t i = 0; i < lay.caps.size(); ++i) {
        for (size_t v = 0; v < lay.graph.var_node.size(); ++v) {
            if (lay.caps[i].b != lay.graph.var_node[v]) continue;
            pro << "    q" << i << " = -comp[" << c_comp[i] << "] * x0[tid * " << out.n_vars
                << " + " << v << "];\n";
            pro << "    q2_" << i << " = q" << i << ";\n";
        }
    }

    pro << "\n    for (int step = 0; step < n_steps; ++step) {\n";
    pro << "        // первый шаг по BDF1: второй точки истории ещё нет\n";
    pro << "        const double alpha = (step == 0) ? (1.0 / h) : (1.5 / h);\n";
    for (size_t i = 0; i < lay.caps.size(); ++i)
        pro << "        const double hist" << i << " = (step == 0) ? (q" << i
            << " / h) : ((4.0 * q" << i << " - q2_" << i << ") / (2.0 * h));\n";
    pro << "\n";

    epi << "\n        // история зарядов\n";
    for (size_t i = 0; i < lay.caps.size(); ++i) {
        epi << "        q2_" << i << " = q" << i << ";\n";
        epi << "        q" << i << " = comp[" << c_comp[i] << "] * "
            << em.diff(lay.caps[i].a, lay.caps[i].b) << ";\n";
    }
    epi << "\n        if ((step + 1) % sample_every == 0) {\n";
    epi << "            const long long s = (long long)((step + 1) / sample_every - 1);\n";
    epi << "            double* dst = out + ((long long)tid * n_samples + s) * " << out.n_vars << ";\n";
    for (int v = 0; v < out.n_vars; ++v)
        epi << "            dst[" << v << "] = " << em.V(lay.graph.var_node[(size_t)v]) << ";\n";
    epi << "        }\n";
    epi << "    }\n";

    out.body = pro.str() + o.str() + epi.str();
    return true;
}
