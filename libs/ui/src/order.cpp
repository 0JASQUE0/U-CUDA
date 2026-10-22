#include "gui_internal.h"

namespace ucuda::ui_detail {


// ---------------------------------------------------------------------------
// Order — окна графиков (см. OrderPlotWindow в app_model.h).
//
// Окно знает, ЧТО оно рисует (kind), и какие конфиги в нём лежат (members).
// Одномерные виды оверлеят сколько угодно вкладок — ради сравнения схем между
// собой вкладка и заведена; карта p(x, y) остаётся одиночной.

const char* order_window_kind_name(OrderPlotWindow::Kind k) {
    switch (k) {
        case OrderPlotWindow::Kind::P:     return "Order p";
        case OrderPlotWindow::Kind::Error: return "Error";
        case OrderPlotWindow::Kind::Map:   return "Map 2D";
        case OrderPlotWindow::Kind::Perf:  return "Performance";
    }
    return "Order";
}


// Может ли конфиг попасть в окно этого вида. Performance показывает только
// замеры времени, карта — только двумерные прогоны, p/ошибка — одномерные.
bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) {
    switch (k) {
        case OrderPlotWindow::Kind::Perf: return c.calc_kind == kOrderCalcPerf;
        case OrderPlotWindow::Kind::Map:  return c.calc_kind == kOrderCalcStab
                                              || (c.calc_kind == kOrderCalcOrder && c.two_d);
        default:                          return c.calc_kind == kOrderCalcOrder && !c.two_d;
    }
}


// Оттенок базового цвета серии: min темнее, max светлее, avg — как есть.
// Три кривые одной вкладки обязаны читаться как одна группа.
//
// k < 1 — подмешать чёрного, k > 1 — БЕЛОГО, а не умножить на k с клампом:
// умножение на светлых цветах палитры (жёлтый, голубой) упиралось в 1.0 по
// каждому каналу, и max получался неотличим от avg — а у самых светлых карт
// сливались все три. Примесь белого светлеет всегда, независимо от того, как
// близко канал к насыщению.
ImVec4 order_shade(ImVec4 c, float k) {
    auto mix = [](float v, float to, float t) { return v + (to - v) * t; };
    auto ch  = [&](float v) {
        const float t = std::min(1.0f, std::fabs(k - 1.0f));
        return std::min(1.0f, std::max(0.0f, (k < 1.0f) ? mix(v, 0.0f, t) : mix(v, 1.0f, t)));
    };
    return ImVec4(ch(c.x), ch(c.y), ch(c.z), c.w);
}


// Форма маркера по порядковому номеру кривой внутри вкладки (min/avg/max или
// E1/E2). Второй признак принадлежности помимо оттенка: на плотной сетке узлов
// и при десятке кривых в окне цвета всё равно сближаются.
int order_marker_for(int within) {
    static const int kSeq[3] = { (int)PointMarker::Circle,
                                 (int)PointMarker::TriangleDown,
                                 (int)PointMarker::TriangleUp };
    return kSeq[(within < 0 ? 0 : within) % 3];
}


// Базовый цвет члена окна. Палитры КАЧЕСТВЕННЫЕ (см. kOrderPalette* в
// app_model.h): номер в списке — номер вкладки в окне, и соседние номера
// обязаны читаться как разные цвета, а не как соседние оттенки.
//
// High contrast — двенадцать насыщенных тонов средней светлоты: слишком
// светлые (лимон, розовый) из ходовых "distinct color" наборов выброшены,
// потому что order_shade делает из них ещё и осветлённый вариант для max, а
// тот на светлой теме плота уже не виден. Okabe-Ito взят целиком, кроме
// чёрного: на тёмной теме он сливается с фоном, поэтому заменён на серый.
ImVec4 order_base_color(int idx, int palette) {
    static const ImU32 kContrast[12] = {
        IM_COL32(0xE6, 0x19, 0x4B, 255),   // алый
        IM_COL32(0x43, 0x63, 0xD8, 255),   // синий
        IM_COL32(0x3C, 0xB4, 0x4B, 255),   // зелёный
        IM_COL32(0xF5, 0x82, 0x31, 255),   // оранжевый
        IM_COL32(0x91, 0x1E, 0xB4, 255),   // фиолетовый
        IM_COL32(0x00, 0xA0, 0xB0, 255),   // бирюзовый
        IM_COL32(0xF0, 0x32, 0xE6, 255),   // пурпурный
        IM_COL32(0x8B, 0x69, 0x14, 255),   // охра
        IM_COL32(0x7F, 0xBF, 0x00, 255),   // травяной
        IM_COL32(0x00, 0x77, 0xBB, 255),   // стальной
        IM_COL32(0xA0, 0x52, 0x2D, 255),   // сиена
        IM_COL32(0x60, 0x60, 0x60, 255),   // серый
    };
    static const ImU32 kOkabe[8] = {
        IM_COL32(0xE6, 0x9F, 0x00, 255),
        IM_COL32(0x56, 0xB4, 0xE9, 255),
        IM_COL32(0x00, 0x9E, 0x73, 255),
        IM_COL32(0xF0, 0xE4, 0x42, 255),
        IM_COL32(0x00, 0x72, 0xB2, 255),
        IM_COL32(0xD5, 0x5E, 0x00, 255),
        IM_COL32(0xCC, 0x79, 0xA7, 255),
        IM_COL32(0x55, 0x55, 0x55, 255),
    };
    if (idx < 0) idx = 0;
    if (palette == kOrderPaletteContrast)
        return ImGui::ColorConvertU32ToFloat4(kContrast[(size_t)(idx % 12)]);
    if (palette == kOrderPaletteOkabe)
        return ImGui::ColorConvertU32ToFloat4(kOkabe[(size_t)(idx % 8)]);
    return ic_base_color(idx);
}


// Короткая сводка по статусам прогона — содержательная часть диаграммы:
// левый обвал (потеря устойчивости) и правая полка (машинная точность) видны
// именно как счётчики, а не только как форма кривой.
void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) {
    ImGui::TextDisabled("%s:", label);
    ImGui::SameLine(); ImGui::Text("ok %d", n_ok);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "| diverged %d", n_div);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "| floor %d", n_floor);
    ImGui::SameLine(); ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f), "| p<=0 %d", n_noc);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "floor    - E2 sank into the machine precision of the solution: the right\n"
            "           plateau, the order is not measurable any further.\n"
            "p<=0     - the differences stopped contracting: loss of\n"
            "           stability, the left edge of the working window in h.\n"
            "diverged - nan/inf or |X| > max value; such cells are NaN.");
}


// Снимок настроек вкладки для <path>_config.csv: движок его не собирает —
// OrderResult/PerfResult не носят ни схемы, ни параметров системы.
data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) {
    data_export::OrderSnapshot sn;
    sn.scheme      = c.scheme;
    sn.axis_x_name = s.axis_target_label(c.axis_x_target);
    sn.axis_y_name = two_d ? s.axis_target_label(c.axis_y_target) : std::string();
    sn.var_names   = s.vars;
    sn.param_names = s.params;
    for (const auto& v : s.vars) {
        auto it = c.initial_conditions.find(v);
        sn.initial_conditions.push_back(it != c.initial_conditions.end() ? parse_num(it->second, 0.0) : 0.0);
    }
    sn.values.push_back(parse_num(c.symmetry_s, 0.5));
    for (const auto& p : s.params) {
        auto it = c.param_values.find(p);
        sn.values.push_back(it != c.param_values.end() ? parse_num(it->second, 0.0) : 0.0);
    }
    sn.h             = parse_num(c.h_text, 0.01);
    sn.t_max         = parse_num(c.t_max_text, 10.0);
    sn.max_value     = parse_num(c.max_value_text, 1.0e6);
    sn.snap_steps    = c.snap_steps;
    sn.endpoint_only = c.endpoint_only;
    sn.use_gpu       = c.use_gpu;
    sn.gpu_fmad      = gpu_fmad;
    sn.gpu_rdc       = gpu_rdc;
    // Режим замера пишем ФАКТИЧЕСКИЙ, из результата: поля вкладки могли
    // измениться уже после прогона.
    sn.repeats  = c.perf_result.repeats;
    sn.warmup   = c.perf_result.warmup;
    sn.replicas = c.perf_result.replicas;
    // Эталон пишем только когда он в результате ЕСТЬ: поля вкладки могли
    // измениться после прогона, а файл описывает прогон, а не поля.
    const bool had_ref = std::any_of(c.perf_result.e_ref.begin(), c.perf_result.e_ref.end(),
                                     [](double v) { return std::isfinite(v) && v > 0.0; });
    if (had_ref) {
        sn.ref_scheme   = c.perf_ref_scheme;
        sn.ref_substeps = std::max(0, (int)parse_num(c.perf_ref_substeps_text, 4.0));
    }
    // Настройки шага — тоже из прогона (у адаптивного Eref нет вовсе, только E(T)).
    if (c.perf_result.adaptive) sn.adaptive_desc = c.perf_result.ad_desc;
    // Настройки теста устойчивости — тоже фактические, из результата; в снимке
    // остаётся лишь раскладка слотов, которой в результате нет.
    sn.stab_k = c.stab_result.k;
    sn.stab_r = c.stab_result.r;
    sn.stab_h = c.stab_result.h;
    sn.stab_pref_tol = parse_num(c.stab_pref_tol_text, 1.0);
    {
        const int slots[4] = { c.stab_idx_a, c.stab_idx_b, c.stab_idx_c, c.stab_idx_d };
        const char* names[4] = { "a", "b", "c", "d" };
        std::string acc;
        for (int i = 0; i < 4; ++i) {
            if (i) acc += ", ";
            acc += names[i];
            acc += " = ";
            const int pi = slots[i] - 1;
            if (pi >= 0 && pi < (int)s.params.size()) acc += s.params[(size_t)pi] + " ";
            acc += "a[" + std::to_string(slots[i]) + "]";
        }
        sn.stab_slots = acc;
    }
    return sn;
}


// ПКМ по графику: по пункту на каждую лежащую в окне вкладку — в оверлее
// «просто экспортировать» нечего, файлы у вкладок разные.
void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) {
    const OrderAnalysisSession& s = model.order_session;
    const bool perf = (win.kind == OrderPlotWindow::Kind::Perf);
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        const bool stab = (c.calc_kind == kOrderCalcStab);
        const bool has = stab ? c.stab_last_run_ok : (perf ? c.perf_last_run_ok : c.last_run_ok);
        const std::string item = "Export " + c.label + "...";
        if (!ImGui::MenuItem(item.c_str(), nullptr, false, has && !busy)) continue;
        if (!cb.pick_save_file_csv) continue;
        const std::string path = cb.pick_save_file_csv();
        if (path.empty()) continue;
        const data_export::OrderSnapshot sn =
            order_snapshot(s, c, win.kind == OrderPlotWindow::Kind::Map,
                           model.nvrtc_fmad, model.nvrtc_rdc);
        if      (stab) data_export::export_stability(c.stab_result, sn, path);
        else if (perf) data_export::export_perf(c.perf_result, sn, path);
        else           data_export::export_order(c.result, sn, path);
    }
}


// Карта области устойчивости. Живёт в окне того же вида (Map): хитмапа с одним
// членом — ровно то, что ей нужно, заводить ради неё отдельный вид окна было бы
// дублированием всей развески вкладок по окнам.
void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) {
    OrderAnalysisSession& s = model.order_session;
    const StabilityResult& r = c.stab_result;

    ImGui::TextDisabled("show:"); ImGui::SameLine();
    ImGui::RadioButton("1 - rho##stabview", &win.stab_view, 1); ImGui::SameLine();
    ImGui::RadioButton("rho##stabview", &win.stab_view, 0); ImGui::SameLine();
    ImGui::RadioButton("region##stabview", &win.stab_view, 2); ImGui::SameLine();
    ImGui::RadioButton("log10 err##stabview", &win.stab_view, 3);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "1 - rho   - positive means stable. The default, because this is the\n"
            "            quantity the reference MATLAB script plots, and the\n"
            "            colour scale is capped at [0, 1] the same way its\n"
            "            contour levels are: everything unstable collapses into\n"
            "            the bottom band, which makes the rho = 1 boundary a\n"
            "            sharp edge instead of a gradient.\n"
            "rho       - the spectral radius itself, scaled over the computed\n"
            "            range: use it to see HOW unstable the outside is.\n"
            "region    - the plain mask rho <= 1: the stability region itself.\n"
            "log10 err - the relative error of one step against the exact\n"
            "            operator, ||e^{hA} - R|| / ||e^{hA}||: the field the\n"
            "            preference region is cut from.\n"
            "\n"
            "The colour range of any of them can be pinned by hand in the\n"
            "toolbar below.");

    // Цвета слоя берутся у самой вью: образец рядом с чекбоксом обязан
    // совпадать с тем, что нарисовано на карте.
    ImGui::Checkbox("preference region##stabpref", &win.stab_show_pref);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Overlay the preference region on the map: cells with rho <= 1 AND\n"
            "a one-step relative error ||e^{hA} - R|| / ||e^{hA}|| not above the\n"
            "tolerance set in \"Stability region\" (Fedoseev et al., 2022).\n"
            "The stability region says the solution does not blow up; the\n"
            "preference region says it also still resembles the exact one.\n"
            "For A-stable schemes the difference is dramatic: the whole left\n"
            "half-plane is stable, only a small part of it is preferred.");
    // Цвета слоя живут на окне (сохраняются вместе с ним) и отдаются вью каждый
    // кадр; перезаливку текстуры при смене цвета вью делает сама.
    hv.overlay_fill_color = win.stab_pref_fill;
    hv.overlay_edge_color = win.stab_pref_edge;
    hv.overlay_edge_width = win.stab_pref_edge_w;
    ImGui::SameLine();
    // Образец — контур поверх заливки, как область и выглядит на карте.
    if (ImGui::ColorButton("##stabpref_swatch", ImGui::ColorConvertU32ToFloat4(win.stab_pref_edge),
                           ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_AlphaPreviewHalf,
                           ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
        ImGui::OpenPopup("stabpref_colors");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Colour of the preference region - click to change.");
    if (ImGui::BeginPopup("stabpref_colors")) {
        // Редактор работает с float[4]; источник правды — IM_COL32 на окне,
        // обратно он пишется только при правке.
        auto edit = [](const char* label, unsigned& col) {
            ImVec4 v = ImGui::ColorConvertU32ToFloat4(col);
            float f[4] = { v.x, v.y, v.z, v.w };
            if (ImGui::ColorEdit4(label, f, ImGuiColorEditFlags_AlphaBar
                                          | ImGuiColorEditFlags_AlphaPreviewHalf))
                col = ImGui::ColorConvertFloat4ToU32(ImVec4(f[0], f[1], f[2], f[3]));
        };
        edit("fill##stabpref", win.stab_pref_fill);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Inside of the region. Keep the alpha low: the map\n"
                              "underneath has to stay readable.");
        edit("edge##stabpref", win.stab_pref_edge);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cells on the boundary of the region (a neighbour is\n"
                              "outside it) - the outline.");
        ImGui::SetNextItemWidth(ImGui::CalcItemWidth());
        ImGui::SliderInt("edge width##stabpref", &win.stab_pref_edge_w, 0, 10, "%d cells",
                         ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Thickness of the outline in cells of the grid: a cell\n"
                              "of the region is painted with the edge colour when it is\n"
                              "at most this many steps from a cell outside. 0 = fill only.\n"
                              "Counted in cells, not pixels, so the outline stays glued\n"
                              "to the grid at any zoom; at a coarse grid 1 is already thick.");
        if (ImGui::Button("Reset##stabpref")) {
            const OrderPlotWindow defaults;
            win.stab_pref_fill   = defaults.stab_pref_fill;
            win.stab_pref_edge   = defaults.stab_pref_edge;
            win.stab_pref_edge_w = defaults.stab_pref_edge_w;
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    ImGui::Checkbox("paper colours##stabpaper", &win.stab_paper_cmap);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Colour the map the way the figures of Fedoseev et al. (2022) do:\n"
            "a grey scale from black to white over the colour range, and WHITE\n"
            "for everything outside it - below as well as above.\n"
            "\n"
            "On the \"1 - rho\" view (range [0, 1]) this gives a white unstable\n"
            "outside, a black rho = 1 boundary and an inside that lightens\n"
            "towards rho = 0: the darker, the closer to losing stability.\n"
            "\n"
            "While this is on, the colormap chosen in the toolbar is not used\n"
            "(it is kept and comes back when you turn this off).");
    // Раскраска отдаётся вью каждый кадр: HeatmapView общий для всех видов
    // окна, и снятая галочка обязана вернуть пользовательскую палитру сразу.
    // Серая шкала — slanCM #28 "gray", чистое чёрное -> белое.
    hv.colormap_override = win.stab_paper_cmap ? slancm_id(28) : 0;
    hv.oor_custom        = win.stab_paper_cmap;
    for (int q = 0; q < 3; ++q) hv.oor_below[q] = hv.oor_above[q] = 1.0f;

    if (!c.stab_last_run_ok || r.rho.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Маска предпочтительности нужна и для счётчика, поэтому строится всегда, а
    // не только при включённом слое. Допуск читается с вкладки каждый кадр:
    // в расчёт он не входит, и его правка обязана доходить до карты сразу.
    const double pref_tol = parse_num(c.stab_pref_tol_text, 1.0);
    std::vector<unsigned char> pref_mask(r.rho.size(), 0);
    int n_pref = 0;
    for (size_t i = 0; i < r.rho.size(); ++i)
        if (stability_cell_preferred(r, i, pref_tol)) { pref_mask[i] = 1; ++n_pref; }

    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    const int n_cells = r.n_ok + r.n_bad;
    ImGui::Text("%s: %d stable, %d preferred (err <= %g) of %d",
                c.label.c_str(), r.n_stable, n_pref, pref_tol, n_cells);
    if (r.n_bad > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "(%d not computed)", r.n_bad);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cells where the test matrix could not be built or the step\n"
                              "returned nan/inf.");
    }
    ImGui::TextDisabled("k = %g, r = %g, h = %g; the boundary is rho = 1", r.k, r.r, r.h);

    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&win](int cm) { win.colormap_idx = cm; };
        draw_heatmap_toolbar(hv, topts);
    }

    // Потолок шкалы. Неустойчивые углы карты уходят в rho на много порядков, и
    // без ограничения вся рабочая окрестность границы становится одноцветной.
    // Ячейки выше потолка не пропадают — упираются в край палитры, ровно как
    // на карте порядка.
    const double rho_cap = (r.rho_max > 2.0) ? 2.0 : r.rho_max;

    std::vector<double> buf(r.rho.size());
    double vmin = 0.0, vmax = 1.0;
    if (win.stab_view == 2) {
        for (size_t i = 0; i < r.rho.size(); ++i)
            buf[i] = std::isfinite(r.rho[i]) ? (r.rho[i] <= 1.0 ? 1.0 : 0.0)
                                             : std::numeric_limits<double>::quiet_NaN();
        vmin = 0.0; vmax = 1.0;
    } else if (win.stab_view == 1) {
        for (size_t i = 0; i < r.rho.size(); ++i) buf[i] = 1.0 - r.rho[i];
        // Диапазон ровно тот, что задают уровни contourf(X, Y, 1-Rh, [1 0:0.01:1])
        // в исходном скрипте: шкала кончается на нуле, и всё неустойчивое
        // (1 - rho < 0) ложится в нижнюю полосу одним цветом. Это не потеря
        // данных, а то, ради чего картинка и строится: граница rho = 1
        // становится краем палитры, а не серединой градиента. Кому нужна
        // глубина неустойчивости — рядом есть вид rho.
        vmin = 0.0;
        vmax = 1.0;
    } else if (win.stab_view == 3) {
        // Точный ноль ошибки бывает (в начале координат шаг любой схемы — это
        // I), и log10 превратил бы его в -inf, то есть в «нет данных». Пол
        // 1e-17 — ниже eps double, отличить его от настоящего нуля нечем.
        // +inf (точная экспонента выродилась) остаётся нечисловым и красится
        // как непосчитанная ячейка. Потолок 1e2 — по той же причине, что rho_cap:
        // иначе дальний край карты съедает всю шкалу, а интересна окрестность
        // допуска.
        const double floor_err = 1e-17;
        for (size_t i = 0; i < r.rho.size(); ++i) {
            const double e = (i < r.err.size()) ? r.err[i] : std::numeric_limits<double>::quiet_NaN();
            buf[i] = std::isfinite(e) ? std::log10(std::max(e, floor_err))
                                      : std::numeric_limits<double>::quiet_NaN();
        }
        vmin = std::log10(std::max(r.err_min, floor_err));
        vmax = std::min(std::log10(std::max(r.err_max, floor_err)), 2.0);
    } else {
        buf = r.rho;
        vmin = r.rho_min;
        vmax = rho_cap;
    }
    if (!(vmax > vmin)) { vmin -= 0.5; vmax += 0.5; }

    // Слой предпочтительности. Поколение текстуры слоя меняется и с данными,
    // и с допуском: допуск правится на лету, и без него в ключе слой остался
    // бы от старого порога.
    if (win.stab_show_pref) {
        std::uint64_t tol_bits = 0;
        std::memcpy(&tol_bits, &pref_tol, sizeof(tol_bits));
        hv.overlay_mask = pref_mask.data();
        hv.overlay_generation = (int)((unsigned)c.stab_data_generation * 1000003u
                                      ^ (unsigned)(tol_bits ^ (tol_bits >> 32)));
    }

    hv.x_axis.name = "sigma";
    hv.y_axis.name = "omega";
    hv.x_axis.log_scale = false;
    hv.y_axis.log_scale = false;

    const int sig = c.stab_data_generation * 8 + win.stab_view;
    bool fit = c.stab_fit_request;
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }
    if (c.stab_fit_request) c.stab_fit_request = false;

    // Ctrl+T меняет сами границы сетки: у этой карты оси и есть диапазоны расчёта.
    const std::vector<ViewRangeTarget> vrt = {
        vr_target(s.configs, mi,
                  &OrderConfig::stab_sig_lo_text, &OrderConfig::stab_sig_hi_text,
                  &OrderConfig::stab_om_lo_text,  &OrderConfig::stab_om_hi_text) };
    hv.popup_extras = [&vrt, &model, &hv, &win, &cb, &s]() {
        draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
        ImGui::Separator();
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail,
              /*owner_id*/ 0x0BDE0000 + win.id,
              sig,
              r.n_pts_x, r.n_pts_y, buf.data(),
              r.sigma_lo, r.sigma_hi, r.omega_lo, r.omega_hi,
              vmin, vmax, fit);
    handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(), hv.swap_axes, &model);
}


// 2D-карта: ровно один член, хитмапа p или log10 E1.
void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) {
    OrderAnalysisSession& s = model.order_session;
    if (win.members.empty()) { ImGui::TextDisabled("The window is empty - add a tab via \"Members...\"."); return; }
    const int mi = win.members[0];
    if (mi < 0 || mi >= (int)s.configs.size()) { ImGui::TextDisabled("The tab has been removed."); return; }
    OrderConfig& c = s.configs[(size_t)mi];
    // Раскраску «как в статье» ставит только карта устойчивости, а вью у окна
    // одна на все его члены: без сброса подмена палитры пережила бы смену
    // члена окна на обычную карту порядка.
    hv.colormap_override = 0;
    hv.oor_custom        = false;
    if (c.calc_kind == kOrderCalcStab) {
        draw_order_stability_map(model, win, cb, renderer, hv, c, mi);
        return;
    }
    const OrderResult& r = c.result;

    ImGui::Checkbox("show error", &win.map_error);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off - the map of the order p. On - log10 E1.");
    ImGui::SameLine();
    draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);

    if (!c.last_run_ok || r.p.empty() || r.n_pts_y <= 1) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    {
        HeatmapToolbarOpts topts;
        topts.persist_colormap = [&win](int cm) { win.colormap_idx = cm; };
        draw_heatmap_toolbar(hv, topts);
    }

    // Ошибка гуляет на много порядков — в карту идёт log10, иначе одна ячейка
    // растягивает шкалу и остальное становится однотонным.
    std::vector<double> buf;
    const std::vector<double>* vals = &r.p;
    if (win.map_error) {
        buf.resize(r.e1.size());
        for (size_t i = 0; i < r.e1.size(); ++i)
            buf[i] = (std::isfinite(r.e1[i]) && r.e1[i] > 0.0)
                         ? std::log10(r.e1[i]) : std::numeric_limits<double>::quiet_NaN();
        vals = &buf;
    }
    // Диапазон цвета — по ячейкам со статусом OK (движок отдаёт их в p_min/p_max
    // и e1_min/e1_max). Полка округления и разошедшиеся ячейки исключены
    // намеренно: одна такая растягивает шкалу так, что рабочая область
    // становится одноцветной. Они не пропадают — упираются в край палитры.
    double vmin = 0.0, vmax = 0.0;
    if (win.map_error) {
        vmin = (r.e1_min > 0.0) ? std::log10(r.e1_min) : 0.0;
        vmax = (r.e1_max > 0.0) ? std::log10(r.e1_max) : 0.0;
    } else {
        vmin = r.p_min;
        vmax = r.p_max;
    }
    if (!(vmax > vmin)) {
        bool first = true;
        for (double v : *vals) {
            if (!std::isfinite(v)) continue;
            if (first) { vmin = vmax = v; first = false; }
            else { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
        }
        if (!(vmax > vmin)) { vmin -= 0.5; vmax += 0.5; }
    }

    hv.x_axis.name = s.axis_target_label(c.axis_x_target);
    hv.y_axis.name = s.axis_target_label(c.axis_y_target);
    hv.x_axis.log_scale = r.axis_x.log_scale;
    hv.y_axis.log_scale = r.axis_y.log_scale;

    const int sig = c.data_generation * 2 + (win.map_error ? 1 : 0);
    bool fit = c.fit_request;
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }
    if (c.fit_request) c.fit_request = false;

    // Ctrl+T: обе оси карты — диапазоны свипа (по умолчанию X — сам шаг h).
    const std::vector<ViewRangeTarget> vrt = {
        vr_target(s.configs, mi,
                  &OrderConfig::axis_x_lo_text, &OrderConfig::axis_x_hi_text,
                  &OrderConfig::axis_y_lo_text, &OrderConfig::axis_y_hi_text) };
    hv.popup_extras = [&vrt, &model, &hv, &win, &cb, &s]() {
        draw_view_range_menu(vrt, hv.x_axis, hv.y_axis, hv.swap_axes, &model);
        ImGui::Separator();
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    hv.render(renderer, origin, avail,
              /*owner_id*/ 0x0BDE0000 + win.id,
              sig,
              r.n_pts_x, r.n_pts_y, vals->data(),
              r.axis_x.lo, r.axis_x.hi, r.axis_y.lo, r.axis_y.hi,
              vmin, vmax, fit);
    handle_view_range_keys(vrt, hv.x_axis, hv.y_axis, plot_window_active(), hv.swap_axes, &model);
}


// Одномерное окно: p(x), E(x) или «время vs ошибка». Кривые всех членов
// лежат в одних осях.
void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) {
    OrderAnalysisSession& s = model.order_session;
    const bool is_perf  = (win.kind == OrderPlotWindow::Kind::Perf);
    const bool is_error = (win.kind == OrderPlotWindow::Kind::Error);
    const bool is_p     = (win.kind == OrderPlotWindow::Kind::P);
    // Есть ли в окне адаптивная вкладка: без неё окно Performance — ровно прежнее
    // (тулбар, оси, подсказки точек), без осей «точность — затраты».
    bool any_ad = false;
    for (int mi : win.members)
        if (mi >= 0 && mi < (int)s.configs.size())
            any_ad |= s.configs[(size_t)mi].adaptive.enabled || s.configs[(size_t)mi].perf_result.adaptive;

    // ---- Тулбар ----
    ImGui::Checkbox("log X", &win.x_log);
    if (!is_p) {
        ImGui::SameLine();
        ImGui::Checkbox("log Y", &win.y_log);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The logarithm is taken of THE VALUES THEMSELVES while the axis stays linear:\n"
                "Plot2DView can only do log on X. The slope of the line in log-log\n"
                "does not change because of that, but the axis labels read as log10.");
    }
    if (is_error) { ImGui::SameLine(); ImGui::Checkbox("show E2", &win.show_e2); }
    if (is_p)     { ImGui::SameLine(); ImGui::Checkbox("nominal order line", &win.show_nominal); }
    if (is_perf) {
        ImGui::SameLine(); ImGui::TextDisabled("| X:"); ImGui::SameLine();
        ImGui::RadioButton("E1##perrsrc", &win.error_source, 0); ImGui::SameLine();
        ImGui::RadioButton("E2##perrsrc", &win.error_source, 1); ImGui::SameLine();
        ImGui::RadioButton("Eref##perrsrc", &win.error_source, 2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "The difference against the reference solution, max|y_h - y_ref|:\n"
                "the actual error of the scheme at step h, rather than the\n"
                "Richardson estimate that E1 and E2 are. The reference method is\n"
                "chosen in the tab's settings (DOPRI78 by default).");
        ImGui::SameLine();
        ImGui::TextDisabled("| t:"); ImGui::SameLine();
        ImGui::RadioButton("us##ptu", &win.time_unit, 0); ImGui::SameLine();
        ImGui::RadioButton("ms##ptu", &win.time_unit, 1); ImGui::SameLine();
        ImGui::TextDisabled("|"); ImGui::SameLine();
        ImGui::Checkbox("min", &win.show_min); ImGui::SameLine();
        ImGui::Checkbox("avg", &win.show_avg); ImGui::SameLine();
        ImGui::Checkbox("max", &win.show_max);

        // Вторая строка — «точность — затраты»: ошибка в T против эталона и другие оси
        // затрат. Есть всегда: E(T) постоянного шага считается против того же y*(T), что у
        // адаптивного (reference y*(T) в настройках вкладки), так что и окно из одних Fixed-
        // вкладок можно строить в тех же осях, что смешанное.
        {
        ImGui::TextDisabled("X:"); ImGui::SameLine();
        ImGui::RadioButton("E(T)##perrsrc", &win.error_source, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Error at t_max, max|y(T) - y*(T)|, against DOP853 in dd/qd on the CPU.\n"
                "Adaptive tabs always have it (their only error). Fixed tabs: with\n"
                "\"reference y*(T)\" set in the tab and \"fit h to t_max\" on - the same\n"
                "y*(T), so fixed-step and adaptive curves share one diagram; without it,\n"
                "Eref at t_max (\"endpoint only\" and \"fit h to t_max\").");
        ImGui::SameLine();
        ImGui::TextDisabled("| Y:"); ImGui::SameLine();
        ImGui::RadioButton("time##pcost", &win.perf_cost, 0); ImGui::SameLine();
        ImGui::RadioButton("f evals##pcost", &win.perf_cost, 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Evaluations of the right-hand side per run (one = all N components):\n"
                              "the controller's count for the adaptive step (rejected attempts\n"
                              "included), f per step x steps for the fixed one. CD family: passes x\n"
                              "(explicit half 1 + implicit half: 1 per component linear in its own\n"
                              "variable, 4 iterations otherwise, / N); Extr: substeps x the base;\n"
                              "Comp: stages x the base; GBS: sum n_k + 1. Unknown for Newton-based\n"
                              "implicit schemes and custom KRS - their curves are left out.\n"
                              "min / avg / max apply to time only.");
        ImGui::SameLine();
        ImGui::RadioButton("steps##pcost", &win.perf_cost, 2);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Accepted steps per run (the rejected ones are in the node tooltip).");
        }
    }

    // Стиль кривых — тот же тулбар-с-попапом, что у точек бифуркационных
    // диаграмм и у Node Series: выключенный режим даёт прежний вид.
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    {
        const std::string sfx = "ordplot" + std::to_string(win.id);
        // Палитра стоит В СТРОКЕ, а не в попапе: попап открывается только при
        // включённом custom_line_style, а цвет кривых нужен и без него.
        if (win.palette < 0 || win.palette > kOrderPaletteOkabe)
            win.palette = kOrderPaletteContrast;
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.0f);
        ImGui::Combo(("##pal" + sfx).c_str(), &win.palette,
                     kOrderPaletteNames, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The colour codes the TAB, the marker shape codes the curve inside it\n"
                              "(min / avg / max). Both extra palettes are qualitative: adjacent\n"
                              "numbers are deliberately far apart in hue, unlike the heatmap maps.");
        ImGui::SameLine();
        draw_style_toolbar("Custom line style", sfx.c_str(), win.custom_line_style,
            [&win, &sfx]() {
                bool ch = false;
                ImGui::SetNextItemWidth(150);
                ch |= ImGui::SliderFloat(("Line width##" + sfx).c_str(), &win.line_width,
                                         0.5f, 8.0f, "%.1f");
                ImGui::SetNextItemWidth(150);
                ch |= ImGui::SliderFloat(("Alpha##" + sfx).c_str(), &win.line_alpha,
                                         0.05f, 1.0f, "%.2f");
                ch |= ImGui::Checkbox(("Points instead of lines##" + sfx).c_str(), &win.points_mode);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Points go through GL_POINTS, and the hover tooltip hangs on\n"
                                      "the ImDrawList markers - on the performance diagram it is\n"
                                      "lost in this mode, together with the h / error / steps of a node.");
                if (win.points_mode) {
                    if (win.point_marker < 0 || win.point_marker >= kPointMarkerCount)
                        win.point_marker = 0;
                    ImGui::SetNextItemWidth(150);
                    ch |= ImGui::Combo(("Marker##" + sfx).c_str(), &win.point_marker,
                                       kPointMarkerNames, kPointMarkerCount);
                    ImGui::SetNextItemWidth(150);
                    ch |= ImGui::SliderFloat(("Point size##" + sfx).c_str(), &win.point_size,
                                             0.5f, 12.0f, "%.1f");
                } else {
                    ch |= ImGui::Checkbox(("Mark data points##" + sfx).c_str(), &win.node_markers);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("On the performance diagram the markers are always on:\n"
                                          "neither the step nor the time itself lies on the axes,\n"
                                          "and the tooltip is attached to the marker.");
                    ImGui::SetNextItemWidth(150);
                    ch |= ImGui::SliderFloat(("Marker size##" + sfx).c_str(), &win.point_size,
                                             1.0f, 12.0f, "%.1f");
                }
                ch |= ImGui::Checkbox(("Marker shape per curve##" + sfx).c_str(), &win.vary_markers);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("min / avg / max (and E1 / E2) get different shapes:\n"
                                      "one tab is one hue, and on a dense grid the hue alone\n"
                                      "is not enough to tell the curves apart.");
                return ch;
            });
    }

    const bool   ylog       = win.y_log && !is_p;
    const double time_scale = (is_perf && win.time_unit == 1) ? 1.0e-3 : 1.0;

    // ---- Сбор кривых ----
    // Буферы локальные: render() заливает VBO прямо в вызове, держать их между
    // кадрами не нужно (и нельзя — членов у окна произвольное число).
    std::vector<std::vector<double>> bufs;
    // Величина «за точкой» для подсказки при наведении, по буферу на кривую и
    // в том же порядке. Пусто — у этой кривой подсказки нет (у диаграммы
    // порядка h лежит прямо на оси X, спрашивать нечего). Выравнивание с bufs
    // держится resize'ом ниже, а не парным push_back: точек добавления кривых
    // пять, и забыть одну было бы легко.
    std::vector<std::vector<double>> tags;
    std::vector<std::string>         labels;
    std::vector<ImVec4>              colors;
    // Форма маркера узла на кривую; выравнивание с bufs держится так же, как у
    // colors — push_back в каждой точке добавления кривой.
    std::vector<int>                 markers;
    std::vector<ViewRangeTarget>     vrt;
    bool   any_data = false, fit = false;
    double xlo = 0.0, xhi = 0.0;
    bool   got_x = false;
    int    sig = (int)win.kind * 1000003;

    auto note_x = [&](double x) {
        if (!got_x) { xlo = xhi = x; got_x = true; }
        else { if (x < xlo) xlo = x; if (x > xhi) xhi = x; }
    };

    for (size_t j = 0; j < win.members.size(); ++j) {
        const int mi = win.members[j];
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        OrderConfig& c = s.configs[(size_t)mi];
        const ImVec4 base = order_base_color((int)j, win.palette);

        if (is_perf) {
            const PerfResult& r = c.perf_result;
            sig = sig * 131 + mi * 17 + c.perf_data_generation;
            if (!c.perf_last_run_ok || r.n_pts <= 0) continue;
            if (c.perf_fit_request) { fit = true; c.perf_fit_request = false; }
            any_data = true;

            // Точка существует, только когда есть ОБЕ координаты: ошибка узла
            // и его затраты. Сортировка по X обязательна — ось X здесь не сетка
            // узлов, а посчитанная величина, и монотонность её не гарантирована.
            const double qn = std::numeric_limits<double>::quiet_NaN();
            auto at = [qn](const std::vector<double>& v, int i) { return (size_t)i < v.size() ? v[(size_t)i] : qn; };
            // tsrc — время, которое попадает в подсказку: у кривых времени своё (min / avg /
            // max), как было до осей f и шагов; у них — среднее.
            auto add_cost_series = [&](const std::function<double(int)>& cost, const std::vector<double>& tsrc,
                                       const std::string& suffix, float shade_k, int marker_slot) {
                // Теги — h (или tol у адаптивного), ошибка, время, шаги, f, отказы. Они
                // обязаны ехать вместе с точкой через сортировку по X: после неё позиция
                // в массиве уже ничего не говорит о том, какому узлу точка принадлежала.
                struct Pt { double x, y, tg[6]; };
                std::vector<Pt> pts;
                pts.reserve((size_t)r.n_pts);
                for (int i = 0; i < r.n_pts; ++i) {
                    // Eref / E(T) есть не в каждом результате: эталон можно было и не
                    // просить, а старые прогоны про него вовсе не знают.
                    const double e = (win.error_source == 3) ? at(r.e_end, i)
                                   : (win.error_source == 2) ? at(r.e_ref, i)
                                   : (win.error_source == 1) ? at(r.e2, i) : at(r.e1, i);
                    const double cv = cost(i);
                    if (!std::isfinite(e) || !std::isfinite(cv)) continue;
                    if (win.x_log && !(e > 0.0)) continue;
                    if (ylog && !(cv > 0.0)) continue;
                    const double ns = ((size_t)i < r.n_steps.size()) ? (double)r.n_steps[(size_t)i] : qn;
                    // e и cv кладём в теги ОТДЕЛЬНО от координат: в лог-режиме в
                    // координатах лежат log10. В подсказке хочется исходные числа.
                    Pt pt{ win.x_log ? std::log10(e) : e, ylog ? std::log10(cv) : cv,
                           { r.adaptive ? at(r.axis_vals, i) : at(r.h_eff, i), e,
                             at(tsrc, i) * time_scale, ns, at(r.n_rhs, i), at(r.n_rej, i) } };
                    pts.push_back(pt);
                }
                if (pts.empty()) return;
                std::sort(pts.begin(), pts.end(),
                          [](const Pt& a, const Pt& b) { return a.x < b.x; });
                std::vector<double> xy, tg;
                xy.reserve(pts.size() * 2);
                tg.reserve(pts.size() * (any_ad ? 6 : 4));
                for (const auto& pr : pts) {
                    xy.push_back(pr.x); xy.push_back(pr.y);
                    for (int q = 0; q < (any_ad ? 6 : 4); ++q) tg.push_back(pr.tg[q]);   // Fixed: h, E, t, steps
                    note_x(pr.x);
                }
                bufs.push_back(std::move(xy));
                tags.resize(bufs.size());
                tags.back() = std::move(tg);
                labels.push_back(c.label + (suffix.empty() ? std::string() : " " + suffix));
                colors.push_back(order_shade(base, shade_k));
                markers.push_back(win.vary_markers ? order_marker_for(marker_slot) : -1);
            };
            const std::string kind_tag = r.adaptive ? std::string("(adaptive)") : std::string();
            if (win.perf_cost == 1) {
                add_cost_series([&](int i) { return at(r.n_rhs, i); }, r.t_avg, kind_tag, 1.0f, 1);
            } else if (win.perf_cost == 2) {
                add_cost_series([&](int i) {
                    return (size_t)i < r.n_steps.size() && r.n_steps[(size_t)i] > 0 ? (double)r.n_steps[(size_t)i] : qn;
                }, r.t_avg, kind_tag, 1.0f, 1);
            } else {
                const std::string sp = kind_tag.empty() ? std::string() : kind_tag + " ";
                if (win.show_min) add_cost_series([&](int i) { return at(r.t_min, i) * time_scale; }, r.t_min, sp + "min", 0.55f, 0);
                if (win.show_avg) add_cost_series([&](int i) { return at(r.t_avg, i) * time_scale; }, r.t_avg, sp + "avg", 1.0f,  1);
                if (win.show_max) add_cost_series([&](int i) { return at(r.t_max, i) * time_scale; }, r.t_max, sp + "max", 1.5f,  2);
            }
            continue;
        }

        const OrderResult& r = c.result;
        sig = sig * 131 + mi * 17 + c.data_generation;
        if (!c.last_run_ok || r.p.empty() || r.n_pts_y > 1) continue;
        if (c.fit_request) { fit = true; c.fit_request = false; }
        any_data = true;

        // Ctrl+T: по X идёт свип, и его границы — поля конфига. По Y входа
        // в расчёте нет (p и E — результат), поэтому цель односторонняя.
        vrt.push_back(vr_target(s.configs, mi,
                                &OrderConfig::axis_x_lo_text, &OrderConfig::axis_x_hi_text));

        const int n = std::min((int)r.axis_x_vals.size(), (int)r.p.size());
        auto add_node_series = [&](int which, const std::string& suffix,
                                   float shade_k, int marker_slot) {
            std::vector<double> xy;
            xy.reserve((size_t)n * 2);
            for (int i = 0; i < n; ++i) {
                const double x = r.axis_x_vals[(size_t)i];
                if (win.x_log && !(x > 0.0)) continue;
                double y;
                if (which == 0) {
                    // Ячейки floor / p<=0 НЕ выбрасываем: их провал и есть ответ
                    // на вопрос, где метод ещё держит свой порядок. Прячем
                    // только diverged — там ничего не посчитано.
                    y = r.p[(size_t)i];
                    if (!std::isfinite(y)) continue;
                } else {
                    const double e = (which == 1) ? r.e1[(size_t)i] : r.e2[(size_t)i];
                    if (!std::isfinite(e) || (ylog && !(e > 0.0))) continue;
                    y = ylog ? std::log10(e) : e;
                }
                xy.push_back(x); xy.push_back(y); note_x(x);
            }
            if (xy.empty()) return;
            bufs.push_back(std::move(xy));
            labels.push_back(suffix.empty() ? c.label : (c.label + " " + suffix));
            colors.push_back(order_shade(base, shade_k));
            markers.push_back(win.vary_markers ? order_marker_for(marker_slot) : -1);
        };

        if (is_error) {
            add_node_series(1, "E1", 1.0f,  0);   // circle
            if (win.show_e2) add_node_series(2, "E2", 0.55f, 1);   // triangle down
        } else {
            add_node_series(0, "", 1.0f, 0);
            if (win.show_nominal) {
                const int nom = builtin_scheme_order(c.scheme);
                if (nom > 0 && !r.axis_x_vals.empty()) {
                    double a = r.axis_x_vals.front(), b = r.axis_x_vals.back();
                    if (b < a) std::swap(a, b);
                    if (!win.x_log || (a > 0.0 && b > 0.0)) {
                        std::vector<double> xy = { a, (double)nom, b, (double)nom };
                        bufs.push_back(std::move(xy));
                        labels.push_back(c.label + " nominal " + std::to_string(nom));
                        colors.push_back(ImVec4(0.6f, 0.6f, 0.6f, 0.9f));
                        markers.push_back(-1);   // горизонталь, узлов у неё нет
                    }
                }
            }
        }
    }

    // Сводка по статусам — по строке на член окна.
    for (size_t j = 0; j < win.members.size(); ++j) {
        const int mi = win.members[j];
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        const OrderConfig& c = s.configs[(size_t)mi];
        if (is_perf) {
            if (!c.perf_last_run_ok) continue;
            const PerfResult& r = c.perf_result;
            draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);
            ImGui::SameLine();
            ImGui::TextDisabled("| %d measurements x %d replicas, %d warmup",
                                r.repeats, r.replicas, r.warmup);
            if (win.error_source == 2) {
                ImGui::SameLine();
                const bool has_ref = std::any_of(r.e_ref.begin(), r.e_ref.end(),
                                                 [](double v) { return std::isfinite(v) && v > 0.0; });
                if (has_ref)
                    ImGui::TextDisabled("| reference: %s x%s",
                                        c.perf_ref_scheme.c_str(), c.perf_ref_substeps_text.c_str());
                else
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                                       r.adaptive ? "| an adaptive tab has only E(T)"
                                                  : "| no reference in this result: set reference substeps > 0 and run again");
            } else if (win.error_source == 3) {
                ImGui::SameLine();
                const bool any_e = std::any_of(r.e_end.begin(), r.e_end.end(),
                                               [](double v) { return std::isfinite(v); });
                if (any_e && r.ref_prec > 0)
                    ImGui::TextDisabled("| y*(T): DOP853 in %s, %lld steps, its own error ~%.1e",
                                        r.ref_prec == 2 ? "qd" : "dd", r.ref_steps, r.ref_err);
                else if (any_e)
                    ImGui::TextDisabled("| E(T) = Eref at t_max: %s x%s",
                                        c.perf_ref_scheme.c_str(), c.perf_ref_substeps_text.c_str());
                else
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f),
                                       r.adaptive ? "| no y*(T) in this result: run again"
                                                  : "| no E(T): set \"reference y*(T)\" (or \"endpoint only\") with \"fit h to t_max\" and run again");
            } else if (r.adaptive) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.4f, 1.0f), "| an adaptive tab has only E(T)");
            }
            if (r.adaptive) {
                double rej = 0, acc = 0;
                for (int i = 0; i < r.n_pts; ++i) {
                    if ((size_t)i < r.n_rej.size() && std::isfinite(r.n_rej[(size_t)i])) rej += r.n_rej[(size_t)i];
                    if ((size_t)i < r.n_steps.size()) acc += (double)r.n_steps[(size_t)i];
                }
                ImGui::SameLine();
                ImGui::TextDisabled("| tol %.0e..%.0e, rejected %.1f%% of attempts",
                                    r.axis_vals.empty() ? 0.0 : r.axis_vals.front(),
                                    r.axis_vals.empty() ? 0.0 : r.axis_vals.back(),
                                    acc + rej > 0 ? 100.0 * rej / (acc + rej) : 0.0);
            }
        } else {
            if (!c.last_run_ok) continue;
            const OrderResult& r = c.result;
            draw_order_status_line(c.label.c_str(), r.n_ok, r.n_diverged, r.n_floor, r.n_nocontract);
        }
    }

    if (win.members.empty()) {
        ImGui::TextDisabled("The window is empty - add a tab via \"Members...\".");
        return;
    }
    if (!any_data || bufs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Толщина > 1 px существует только на пути ImDrawList: glLineWidth в core
    // OpenGL драйверы клампят до 1.0 (то же решение, что у LLE/LS и Network).
    const bool pts = win.custom_line_style && win.points_mode;
    view.imdraw_lines = !pts;
    view.show_zero_x  = false;
    // Линия y = 0 осмысленна только на графике порядка: ниже неё разности
    // перестали сокращаться. В логарифме ошибки нуль — это E = 1, рисовать
    // его незачем.
    view.show_zero_y  = is_p;
    view.points_mode  = pts;
    view.point_marker  = pts ? win.point_marker : -1;
    view.point_size_px = pts ? win.point_size : 2.0f;
    view.line_thickness_px = win.custom_line_style ? win.line_width : 1.5f;

    // Подписи осей. У членов окна ось X по построению одна и та же величина,
    // поэтому имя берём у первого valid-члена.
    std::string xname = "x";
    for (int mi : win.members) {
        if (mi < 0 || mi >= (int)s.configs.size()) continue;
        xname = is_perf ? ((win.error_source == 3) ? "E(T)" : (win.error_source == 2) ? "Eref"
                            : (win.error_source == 1) ? "E2" : "E1")
                        : s.axis_target_label(s.configs[(size_t)mi].axis_x_target);
        break;
    }
    // На диаграмме производительности лог по X берётся ЛОГАРИФМОМ ДАННЫХ, а не
    // логарифмическим масштабом оси — как это давно сделано по Y. Причина в
    // подписях: на лог-оси тики печатаются как сами значения, и на
    // double-double ошибка уходит к 1e-30, где формат схлопывает их в "0" и
    // ось становится нечитаемой. log10(E) — обычные числа вроде -16, они
    // помещаются всегда. Исходное E остаётся в подсказке узла (тег "E1").
    // На диаграмме порядка по X лежит сетка свипа, там масштаб оси и остаётся.
    if (is_perf && win.x_log) xname = "log10 " + xname;
    view.x_axis.name = xname;
    if (is_perf) {
        const char* unit = (win.time_unit == 1) ? "ms" : "us";
        const std::string yn = win.perf_cost == 1 ? std::string("f evals")
                             : win.perf_cost == 2 ? std::string("steps")
                                                  : std::string("t, ") + unit;
        view.y_axis.name = ylog ? "log10 " + yn : yn;
    } else {
        view.y_axis.name = is_error ? (ylog ? "log10 E" : "E") : "p";
    }
    view.x_axis.log_scale = !is_perf && win.x_log;

    // Явный X-диапазон по всем кривым окна: без него Plot2DView берёт границы
    // клампа из bbox VBO, а там на лог-оси лежит log10(x). На диаграмме
    // производительности ось линейная, и note_x собирал уже те же значения,
    // что легли в VBO, — единицы сходятся и тут.
    view.x_fit_use_explicit = got_x && (xhi > xlo);
    view.x_fit_min = xlo;
    view.x_fit_max = xhi;

    // Маркеры узлов нужны именно на диаграмме производительности: ни шага, ни
    // самого времени в лог-режиме на осях нет. На диаграмме порядка h лежит
    // прямо на оси X, спрашивать нечего.
    view.point_markers = !pts && (is_perf || (win.custom_line_style && win.node_markers));
    view.point_marker_px = win.custom_line_style ? win.point_size : 3.5f;
    if (is_perf) {
        const char* ename = (win.error_source == 3) ? "E(T)" : (win.error_source == 2) ? "Eref"
                          : (win.error_source == 1) ? "E2" : "E1";
        const char* unit  = (win.time_unit == 1) ? "ms" : "us";
        if (any_ad) view.point_tag_names = { "h / tol", ename, std::string("t, ") + unit, "steps", "f evals", "rejected" };
        else        view.point_tag_names = { "h", ename, std::string("t, ") + unit, "steps" };
    }

    std::vector<PlotSeriesInput> series;
    std::vector<bool> init_vis, glob_vis;
    series.reserve(bufs.size());
    tags.resize(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        PlotSeriesInput si;
        si.points   = bufs[i].empty() ? nullptr : bufs[i].data();
        si.n_points = (int)(bufs[i].size() / 2);
        si.color    = colors[i];
        if (win.custom_line_style) si.color.w = win.line_alpha;
        si.label    = labels[i];
        if (i < markers.size()) si.node_marker = markers[i];
        // Длины обязаны сойтись: если кривая отфильтровалась иначе, чем
        // собирались теги, лучше остаться без подсказки, чем показать чужое h.
        const int tag_cols = (int)view.point_tag_names.size();
        if (tag_cols > 0 && !tags[i].empty() &&
            tags[i].size() == (size_t)si.n_points * (size_t)tag_cols) {
            si.point_tags      = tags[i].data();
            si.point_tag_count = tag_cols;
        }
        series.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(true);
    }

    // Сигнатура рисуемого: новые данные любого члена, смена масштаба любой
    // оси, состав кривых. Служит и поколением для перезаливки VBO, и триггером
    // autofit — log10 E и E отличаются на четырнадцать порядков, а p от них обоих.
    sig = sig * 131 + (win.x_log ? 1 : 0) + (ylog ? 2 : 0) + (win.show_e2 ? 4 : 0)
        + (win.show_nominal ? 8 : 0) + (win.show_min ? 16 : 0) + (win.show_avg ? 32 : 0)
        + (win.show_max ? 64 : 0) + win.error_source * 128 + win.time_unit * 512
        + win.perf_cost * 1024 + (int)bufs.size() * 4096;
    // Стиль в сигнатуру НЕ входит намеренно: содержимое VBO от него не
    // зависит (толщина, маркер, альфа — решения момента рисования), а смена
    // сигнатуры тянет за собой autofit и сбросила бы текущий масштаб вида.
    if (win.plot_sig != sig) { fit = true; win.plot_sig = sig; }

    // Ctrl+T есть только там, где по X лежит вход расчёта: на диаграмме
    // производительности ось X — посчитанная ошибка, задать её нельзя.
    view.popup_extras = [&vrt, &model, &view, &win, &cb, &s]() {
        if (!vrt.empty()) {
            draw_view_range_menu(vrt, view.x_axis, view.y_axis, /*swapped*/ false, &model);
            ImGui::Separator();
        }
        draw_order_export_items(model, win, cb, s.in_flight);
    };

    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail,
                /*owner_id*/ 0x0BDE1000 + win.id, sig,
                series, init_vis, glob_vis, fit);
    if (!vrt.empty())
        handle_view_range_keys(vrt, view.x_axis, view.y_axis, plot_window_active(), /*swapped*/ false, &model);
}


void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb) {
    auto& renderers = ui_resource<std::map<int, std::unique_ptr<PlotRenderer>>, 13>();
    auto& heatmaps = ui_resource<std::map<int, std::unique_ptr<HeatmapView>>, 14>();
    auto& curves = ui_resource<std::map<int, std::unique_ptr<Plot2DView>>, 15>();

    int to_remove = -1;
    for (int i = 0; i < (int)model.order_plot_windows.size(); ++i) {
        OrderPlotWindow& win = model.order_plot_windows[i];

        auto& renderer = renderers[win.id];
        if (!renderer) renderer = std::make_unique<PlotRenderer>();

        // Автоподпись держит вид окна: пользователь переименовал — не трогаем.
        if (!win.label_is_manual)
            win.label = std::string(order_window_kind_name(win.kind)) + " " + std::to_string(win.id);

        std::string title = win.label + "###orderwin" + std::to_string(win.id)
                          + "_g" + std::to_string(model.order_layout_generation);
        bool open = true;
        float ox = 60.0f + (float)(i % 5) * 35.0f, oy = 80.0f + (float)(i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(win.id);
            if (win.kind == OrderPlotWindow::Kind::Map) {
                auto& hv = heatmaps[win.id];
                if (!hv) {
                    hv = std::make_unique<HeatmapView>();
                    hv->colormap = (HeatmapColormap)colormap_id_or(win.colormap_idx, model.heatmap_colormap);
                }
                draw_order_map_window(model, win, cb, *renderer, *hv);
            } else {
                auto& cv = curves[win.id];
                if (!cv) cv = std::make_unique<Plot2DView>();
                draw_order_curve_window(model, win, cb, *renderer, *cv);
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) to_remove = i;
    }
    if (to_remove >= 0) {
        const int id = model.order_plot_windows[to_remove].id;
        renderers.erase(id);
        heatmaps.erase(id);
        curves.erase(id);
        model.remove_order_plot_window(to_remove);
    }
}


void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Union used by the Run / Run all... buttons below to disable themselves
    // while any sub-analysis (Bif/LLE/LS) is computing. Kept after the tab's
    // System combo was moved to the top-bar; the top-bar combo has its own
    // wider union across all 7 sessions.
    bool any_in_flight = model.bifurcation_session.in_flight
                      || model.lle_session.in_flight
                      || model.ls_session.in_flight
                      || model.metrics_session.in_flight;

    // Run (active sub-tab, active config) + batch Run all. Кнопка Run единая для Bif/LLE/LS,
    // диспатчится по parametric_active_analysis (0=Bif, 1=LLE, 2=LS) к соответствующему
    // active_*_index. Run all — глобальный по всем трём спискам: попап показывает чекбоксы
    // для каждой диаграммы / кривой / спектра, Run кладёт отмеченные в
    // model.parametric_queue и стартует первый; draw_gui тикает очередь после poll'ов.
    {
        const int kind = model.parametric_active_analysis;
        int  active_idx = -1;
        bool no_active  = false;
        if (kind == 0) {
            active_idx = model.bifurcation_session.active_diagram_index;
            no_active  = model.bifurcation_session.diagrams.empty();
        } else if (kind == 1) {
            active_idx = model.lle_session.active_curve_index;
            no_active  = model.lle_session.curves.empty();
        } else if (kind == 2) {
            active_idx = model.ls_session.active_curve_index;
            no_active  = model.ls_session.curves.empty();
        } else if (kind == 3) {
            active_idx = model.metrics_session.active_config_index;
            no_active  = model.metrics_session.configs.empty();
        }

        std::vector<RunAllGroup> groups;
        groups.push_back(RunAllGroup{
            "Bifurcation", "pbd_", (int)model.bifurcation_session.diagrams.size(),
            [&model](int i) { return model.bifurcation_session.diagrams[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, i});
            } });
        groups.push_back(RunAllGroup{
            "LLE", "plle_", (int)model.lle_session.curves.size(),
            [&model](int i) { return model.lle_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, i});
            } });
        groups.push_back(RunAllGroup{
            "LS", "pls_", (int)model.ls_session.curves.size(),
            [&model](int i) { return model.ls_session.curves[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::LS, i});
            } });
        groups.push_back(RunAllGroup{
            "Metrics", "pmet_", (int)model.metrics_session.configs.size(),
            [&model](int i) { return model.metrics_session.configs[i].label; },
            [&model](int i) {
                model.parametric_queue.push_back({ParametricQueueItem::Kind::Metrics, i});
            } });

        // Run all блокируется ТОЛЬКО на время расчёта: списков три, и пустой
        // активный не повод запрещать запуск остальных (в отличие от вкладок
        // с одним списком, где block_run_all включает и «нет конфигов»).
        draw_run_and_run_all("##run_all_parametric", any_in_flight, no_active,
                             any_in_flight, 140.0f,
                             [&model, kind, active_idx]() {
                                 if (active_idx < 0) return;
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 if      (kind == 0) model.bifurcation_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 1) model.lle_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 2) model.ls_session.run_async(*model.parametric_engine, active_idx);
                                 else if (kind == 3) model.metrics_session.run_async(*model.parametric_engine, active_idx);
                             },
                             {}, groups,
                             [&model]() { model.start_next_in_parametric_queue(); },
                             model.parametric_queue.size());
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("##parm_top")) {
        if (ImGui::BeginTabItem("Bifurcation")) {
            model.parametric_active_analysis = 0;
            draw_bifurcation_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LLE")) {
            model.parametric_active_analysis = 1;
            draw_lle_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LS")) {
            model.parametric_active_analysis = 2;
            draw_ls_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Metrics")) {
            model.parametric_active_analysis = 3;
            draw_metrics_controls(model, lib);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    // Plot windows: dynamic list, mirrors Phase's Projections section. A sibling of the
    // Bif/LLE/LS tabs above (not nested in any of them) — always visible, since a plot window
    // can be any (kind, dimension) combo. Each row has its own Type combo so the chart type
    // can be changed after creation instead of being fixed at "Add".
    ImGui::Separator();
    ImGui::SeparatorText("Plot windows");

    int win_to_remove = -1;
    for (int i = 0; i < (int)model.parametric_plot_windows.size(); ++i) {
        ParametricPlotWindow& win = model.parametric_plot_windows[i];
        ImGui::PushID(win.id);
        if (draw_parametric_window_setup_row(model, win)) win_to_remove = i;
        ImGui::PopID();
    }
    if (win_to_remove >= 0) model.remove_parametric_plot_window(win_to_remove);

    if (ImGui::Button("Add window")) {
        model.add_parametric_plot_window(ParametricPlotWindow::Kind::Bifurcation, false, {});
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { model.parametric_layout_generation++; }
}


// Загрузка Order из файлов засевает кэш автосохранения (autosave_order_session)
// только что загруженным состоянием: файл перепишется при первой настоящей
// правке, а не сразу после загрузки. Иначе битый файл, отвергнутый
// apply_session_json, молча заменился бы умолчаниями, хотя пользователь ещё
// ничего не менял.
void order_autosave_seed(AppModel& model) {
    model.order_autosave_target       = model.loaded_name;
    model.order_autosave_json         = session_to_json_order(model.order_session);
    model.order_windows_autosave_json = session_to_json_order_windows(model.order_plot_windows);
}


// Global system switch — fired from the top-bar combo. Loads the record and re-inits the
// CURRENT tab (mirrors what each per-tab combo used to do); other tabs re-init on entry via
// the block in draw_gui. Non-static: also called from app_main.cpp on startup to restore the
// last-used system (see gui.h::apply_system_switch).
}
using namespace ucuda::ui_detail;
void apply_system_switch(AppModel& model, SystemLibrary& lib,
                         const std::string& name)
{
    try {
        model.from_record(lib.load(name));
        switch (model.app_mode) {
        case AppModel::AppMode::Analysis: {
            model.start_phase_analysis();
            std::string j = lib.load_session(model.loaded_name, "_last");
            apply_session_json(model, j, model.phase_session, session_from_json, "_last");
            break;
        }
        case AppModel::AppMode::Parametric: {
            model.start_parametric_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
            apply_session_json(model, jb, model.bifurcation_session, session_from_json_parametric, "_last_parametric");
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            apply_session_json(model, jl, model.lle_session, session_from_json_lle, "_last_lle");
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            apply_session_json(model, js, model.ls_session, session_from_json_ls, "_last_ls");
            std::string jm = lib.load_session(model.loaded_name, "_last_metrics");
            apply_session_json(model, jm, model.metrics_session, session_from_json_metrics, "_last_metrics");
            std::string jw = lib.load_session(model.loaded_name, "_last_parametric_windows");
            model.load_or_init_parametric_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Dft1D: {
            model.start_dft1d_analysis();
            std::string jd = lib.load_session(model.loaded_name, "_last_dft1d");
            apply_session_json(model, jd, model.dft1d_session, session_from_json_dft1d, "_last_dft1d");
            std::string jw = lib.load_session(model.loaded_name, "_last_dft1d_windows");
            model.load_or_init_dft1d_plot_windows(jw);
            break;
        }
        case AppModel::AppMode::Basins: {
            model.start_basins_analysis();
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            apply_session_json(model, jb, model.basins_session, session_from_json_basins, "_last_basins");
            break;
        }
        case AppModel::AppMode::FastSync: {
            model.start_fastsync_analysis();
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            apply_session_json(model, jf, model.fastsync_session, session_from_json_fastsync, "_last_fastsync");
            break;
        }
        case AppModel::AppMode::Custom: {
            // Hard reset + engine wipe. Sub-session buffers, signature caches,
            // AND the CUDA-side PTX cache in parametric_engine all need to
            // clear on system switch — otherwise the second-system Run either
            // produced a stale-looking result or zeroed out the first one.
            model.custom_session = CustomSession{};
            model.parametric_engine.reset();
            model.start_custom_analysis();
            std::string jc = lib.load_session(model.loaded_name, "_last_custom");
            apply_session_json(model, jc, model.custom_session, session_from_json_custom, "_last_custom");
            break;
        }
        case AppModel::AppMode::Order: {
            model.start_order_analysis();
            std::string jo = lib.load_session(model.loaded_name, "_last_order");
            apply_session_json(model, jo, model.order_session, session_from_json_order, "_last_order");
            std::string jw = lib.load_session(model.loaded_name, "_last_order_windows");
            model.load_or_init_order_plot_windows(jw);
            order_autosave_seed(model);
            break;
        }
        case AppModel::AppMode::Network: {
            model.start_network_analysis();
            std::string jn = lib.load_session(model.loaded_name, "_last_network");
            apply_session_json(model, jn, model.network_session, session_from_json_network, "_last_network");
            break;
        }
        case AppModel::AppMode::Library:
        case AppModel::AppMode::Settings:
        default:
            break;
        }
        // Persist the choice so the next launch restores this system. Read-
        // modify-write so we don't clobber unrelated fields (colormaps, UI
        // scale, etc.). Silent on failure — best-effort UX polish.
        AppConfig cfg;
        load_app_config(get_exe_dir_with_sep(), cfg);
        cfg.last_system_name = model.name;
        save_app_config(get_exe_dir_with_sep(), cfg);
    } catch (...) {}
}
namespace ucuda::ui_detail {



// Custom tab (master-detail pipeline) — see custom_session.h



// Комбо выбора цели свипа и разбор числовых полей — общие для всего файла
// (draw_sweep_target_combo / parse_ratio_or, см. верх файла). Раньше здесь
// лежали их локальные копии: комбо со своим стилем подписей ("IC x" вместо
// "x (IC)") и обёртка parse_num_default поверх того же парсера.

// Shared config panel

void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) {
    // Чей конфиг правит эта панель — для ПКМ-рассылки "во все такие же".
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c      = cs.shared;
    const auto& vars   = cs.vars;
    const auto& params = cs.params;
    auto& phase  = cs.phase_session;
    ImGui::SeparatorText("Shared config");

    // Scheme combo — mirrors draw_diagram_controls / draw_lle_controls layout, so users get
    // the same picker with built-ins + custom schemes. On change, propagate to Phase (L3 shows
    // the same picker) and rebuild its KRS body so the next Phase Run uses the new integrator;
    // L1D/L2D pick it up via copy_integrator_and_state on Run.
    draw_scheme_combo("Scheme", c.scheme, custom_schemes,
        [&phase, &custom_schemes](const std::string& nm) {
            phase.scheme = nm;
            phase.regenerate_krs();
            // Custom КРС и экстраполяционные обёртки в Custom-вкладке считаются только на GPU.
            if (scheme_needs_compiled_body(nm, custom_schemes)) phase.use_gpu = true;
        },
        &cs.enabled_builtin_schemes, &model, model.is_map, &cs.wrapper_schemes);

    // Integration group — mirrors the "Integration##bd_int" collapsing header in
    // draw_diagram_controls (per-line InputNumStr with comma→dot + ↑/↓). Each edited field is
    // mirrored into L1D's own override (`l1d_h_text` for step) and into `phase_session`, so the
    // typed value shows up in the L1D and L3 panels without waiting for Run. L1D keeps
    // independent transient/computing-time overrides — step is intentionally the ONLY per-L1D
    // field kept in sync with shared.
    if (ImGui::CollapsingHeader("Integration##custom_int", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!model.is_map)
            if (InputNumStr("h",          c.h_text, kFieldW)) {
                c.l1d_h_text  = c.h_text;
                phase.step_h  = c.h_text;
            }
        if (scheme_uses_symmetry(c.scheme, custom_schemes))
            if (InputNumStr("symmetry s", c.symmetry_s, kFieldW))
                phase.symmetry_s = c.symmetry_s;
        // TT before CT: transient runs first, computing-time is what's
        // actually sampled after — order matches conceptual flow.
        if (InputNumStr(model.is_map ? "transient iterations" : "transient time",
                        c.transient_text, kFieldW))
            phase.skip_time = c.transient_text;
        if (InputNumStr(model.is_map ? "iterations" : "computing time",
                        c.t_max_text, kFieldW))
            phase.sim_time  = c.t_max_text;
        if (InputNumStr("decimator",      c.pre_scaller_text, kFieldW))
            phase.decimation = c.pre_scaller_text;
        InputNumStr("max value",      c.max_value_text, kFieldW); // Phase has no analogue
        // Адаптивный шаг — общий для всех уровней; Phase получает его сразу, как и h.
        ImGui::Separator();
        if (draw_adaptive_block("custom_ad", c.adaptive, c.scheme,
                                kAdUiRaw | kAdUiInterp | kAdUiMaxPts, model.is_map, &model))
            phase.adaptive = c.adaptive;
    }

    // Initial conditions — one InputNumStr per line, matching draw_diagram_controls.
    // Not propagated to Phase: phase.ic_sets is multi-IC (mulistability) and
    // is edited in the L3 Phase panel; the shared IC block drives BD/LLE/LS/Basins.
    draw_named_num_fields("Initial conditions##custom_ic", vars, c.initial_conditions,
                          nullptr, &model, BroadcastField::InitCondition);

    // Parameters — same per-line layout; skip disable for params that are
    // swept on any enabled level (they get their values from the sweep).
    // Non-swept params are also mirrored to phase.param_values so the L3
    // Phase panel Parameters section shows the same live values.
    if (ImGui::CollapsingHeader("Parameters##custom_par", ImGuiTreeNodeFlags_DefaultOpen)) {
        // All params stay editable — the sweep overwrites swept-axis values at run time anyway,
        // and the manual value is useful for initial-frame edits and non-sweeping levels (L3
        // Phase). Swept params are MARKED with a "(swept)" tag so it is visible at a glance
        // which values the sweep will replace.
        std::vector<bool> is_swept(params.size(), false);
        const bool any_2d   = c.bif2d_enabled || c.lle2d_enabled || c.ls2d_enabled;
        const bool any_1d_x = c.bif1d_x_enabled || c.lle1d_x_enabled || c.ls1d_x_enabled;
        const bool any_1d_y = c.bif1d_y_enabled || c.lle1d_y_enabled || c.ls1d_y_enabled;
        // over_h проверяется наравне с over_var: при свипе по шагу par_index
        // остаётся от прежнего выбора, и без этого условия "(swept)" висело на
        // постороннем параметре — том, чей индекс случайно лежал в par_index.
        auto mark = [&](int par_i, bool over_var, bool over_h) {
            if (over_var || over_h) return;
            if (par_i < 0 || par_i >= (int)params.size()) return;
            is_swept[par_i] = true;
        };
        if (c.level_2d_enabled && any_2d) {
            mark(c.axis_x_par_index, c.axis_x_over_var, c.axis_x_over_h);
            mark(c.axis_y_par_index, c.axis_y_over_var, c.axis_y_over_h);
        }
        if (c.level_1d_enabled) {
            if (any_1d_x) {
                EffectiveSweep esx = effective_sweep_x(c);
                mark(esx.par_index, esx.over_var, esx.over_h);
            }
            if (any_1d_y) {
                EffectiveSweep esy = effective_sweep_y(c);
                mark(esy.par_index, esy.over_var, esy.over_h);
            }
        }
        // Effective sweep axes cached once — used by both the (swept) label
        // logic below AND the crosshair sync when the user edits the value
        // of a sweep-axis param manually.
        EffectiveSweep esx = effective_sweep_x(c);
        EffectiveSweep esy = effective_sweep_y(c);
        for (int i = 0; i < (int)params.size(); ++i) {
            const auto& p = params[i];
            ImGui::PushID(p.c_str());
            if (InputNumStr(p.c_str(), c.param_values[p], kFieldW)) {
                phase.param_values[p] = c.param_values[p];
                // Bump debounce timers so the auto-recompute path (L1D
                // partial + Phase, gated by auto_recompute_1d /
                // autorun_on_drilldown) picks up the manual edit — same
                // path a slider drag settle goes through, no Run needed.
                double t = ImGui::GetTime();
                c.last_fix_x_change_time = t;
                c.last_fix_y_change_time = t;
                // If this param IS the effective sweep axis, move the
                // crosshair to the new pinned value so 1D slice plots and
                // 2D heatmaps show it there immediately.
                double new_val = parse_ratio_or(c.param_values[p], 0.0);
                if (!esx.over_var && !esx.over_h && esx.par_index == i) c.fix_x_value = new_val;
                if (!esy.over_var && !esy.over_h && esy.par_index == i) c.fix_y_value = new_val;
            }
            if (is_swept[i]) {
                ImGui::SameLine();
                ImGui::TextDisabled("(swept)");
            }
            ImGui::PopID();
        }
    }
}


// Level 2D detail

void draw_level2d_detail(AppModel& model, CustomSession& cs) {
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    ImGui::Checkbox("Bif",  &c.bif2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE",  &c.lle2d_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS",   &c.ls2d_enabled);

    // Axis X — combo (sweep target) + lo/hi/N on separate lines so the
    // per-field digit-step (↑/↓) and comma→dot filter work reliably (they
    // don't when InputText's are packed onto one SameLine chain).
    ImGui::TextUnformatted("Axis X:"); ImGui::SameLine();
    draw_sweep_target_combo("##ax", cs.params, cs.vars,
                            c.axis_x_par_index, c.axis_x_over_var, c.axis_x_var_index,
                            c.axis_x_over_h, &c.axis_y_over_h,
                            /*note_when_empty*/ false, 120.0f,
                            scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                            &model, BroadcastField::SweepTarget, !model.is_map);
    InputNumStr("lo##ax", c.axis_x_lo_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepLo, {},
                                          "sweep lo", c.axis_x_lo_text); });
    InputNumStr("hi##ax", c.axis_x_hi_text, kFieldW,
                [&]{ field_apply_all_menu(&model, BroadcastField::SweepHi, {},
                                          "sweep hi", c.axis_x_hi_text); });
    // Log-сетка по оси. Свойство оси, поэтому X-срез Level 1D наследует его
    // вместе с par/lo/hi (см. EffectiveSweep::log_scale).
    ImGui::Checkbox("Log scale##ax_log", &c.axis_x_log);
    if (c.axis_x_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    ImGui::TextUnformatted("Axis Y:"); ImGui::SameLine();
    draw_sweep_target_combo("##ay", cs.params, cs.vars,
                            c.axis_y_par_index, c.axis_y_over_var, c.axis_y_var_index,
                            c.axis_y_over_h, &c.axis_x_over_h,
                            /*note_when_empty*/ false, 120.0f,
                            scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                            &model, BroadcastField::SweepTarget2, !model.is_map);
    InputNumStr("lo##ay", c.axis_y_lo_text, kFieldW);
    InputNumStr("hi##ay", c.axis_y_hi_text, kFieldW);
    ImGui::Checkbox("Log scale##ay_log", &c.axis_y_log);
    if (c.axis_y_log) { ImGui::SameLine(); ImGui::TextDisabled("(lo/hi > 0)"); }

    // Shared N×N grid resolution — kernel `getValueByIdx` requires a square
    // grid, so one field drives both axes (matches Analysis tab).
    InputNumStr("Resolution##a", c.resolution_text, kFieldW);

    // Переменная БД — задаётся здесь и наследуется обоими 1D-срезами (см.
    // CustomTabSharedConfig::bif_writable_var). Рисуется независимо от
    // bif2d_enabled: при выключенном Bif-2D её всё равно читает Level 1D.
    // Комбинация (-1) работает и в 2D, и в 1D — одно и то же ядро.
    ImGui::Separator();
    draw_writable_var_combo(cs.vars, c.bif_writable_var, "Bif variable##custom_wv");

    // Per-type options edited directly on the sub-session's slot [0] (2D config).
    ImGui::Separator();
    if (c.bif2d_enabled && !cs.bif_session.diagrams.empty()) {
        InputNumStr("Bif DBSCAN eps", cs.bif_session.diagrams[0].eps_dbscan_text, kFieldW);
        // Множители осей кластеризации — см. draw_diagram_controls.
        InputNumStr("Bif mult peak",     cs.bif_session.diagrams[0].mult_peak_text, kFieldW);
        InputNumStr("Bif mult interval", cs.bif_session.diagrams[0].mult_interval_text, kFieldW);
    }
    // eps/NT у LLE и LS живут в слоте [0] и наследуются 1D-срезами (см.
    // apply_shared_to_lle1d / _ls1d), поэтому поля показываем и когда сама
    // карта выключена, но включён хотя бы один её срез — иначе их негде править.
    if ((c.lle2d_enabled || c.lle1d_x_enabled || c.lle1d_y_enabled) &&
        !cs.lle_session.curves.empty()) {
        InputNumStr("LLE eps", cs.lle_session.curves[0].eps_text, kFieldW);
        InputNumStr("LLE NT",  cs.lle_session.curves[0].nt_text, kFieldW);
    }
    if ((c.ls2d_enabled || c.ls1d_x_enabled || c.ls1d_y_enabled) &&
        !cs.ls_session.curves.empty()) {
        InputNumStr("LS eps", cs.ls_session.curves[0].eps_text, kFieldW);
        InputNumStr("LS NT",  cs.ls_session.curves[0].nt_text, kFieldW);
    }
    ImGui::Separator();

    // Run buttons live on the top row of draw_custom_controls — this panel
    // is edit-only.
}


// Level 1D detail

void draw_level1d_detail(CustomSession& cs) {
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.

    if (!c.level_2d_enabled) ImGui::BeginDisabled();
    ImGui::Checkbox("Inherit sweep from Level 2D", &c.inherit_sweep_from_2d);
    if (!c.level_2d_enabled) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(Level 2D off - L1D uses its own sweep)");
    }

    const bool inheriting = c.inherit_sweep_from_2d && c.level_2d_enabled;
    EffectiveSweep sx = effective_sweep_x(c);
    EffectiveSweep sy = effective_sweep_y(c);

    // Inherit disables ONLY the sweep axis (par target + lo/hi + log), leaving N and the
    // L1D-specific integrator fields (h/TT/CT below) always editable — L1D is cheap and
    // interactive, so a finer grid or longer transient than L2D makes sense even on a shared axis.
    // В inherit-режиме виджеты кормим КОПИЯМИ эффективного свипа: править их всё равно нельзя
    // (BeginDisabled), зато в панели видно то, что реально пойдёт в расчёт. Раньше показывались
    // собственные sweep_*, которые в этом режиме не используются, — панель врала об осях среза.
    auto sweep_row = [&](const char* label, const char* tag,
                         const EffectiveSweep& eff,
                         int& own_par, bool& own_ov, int& own_vi,
                         bool& own_oh, bool& own_log,
                         std::string& own_lo, std::string& own_hi,
                         bool* other_oh, std::string& n_text) {
        int         par = inheriting ? eff.par_index : own_par;
        bool        ov  = inheriting ? eff.over_var  : own_ov;
        int         vi  = inheriting ? eff.var_index : own_vi;
        bool        oh  = inheriting ? eff.over_h    : own_oh;
        bool        lg  = inheriting ? eff.log_scale : own_log;
        std::string lo  = inheriting ? eff.lo_text   : own_lo;
        std::string hi  = inheriting ? eff.hi_text   : own_hi;
        bool        other_copy = other_oh ? *other_oh : false;

        ImGui::TextUnformatted(label); ImGui::SameLine();
        if (inheriting) ImGui::BeginDisabled();
        draw_sweep_target_combo((std::string("##") + tag + "p").c_str(),
                                cs.params, cs.vars, par, ov, vi, oh,
                                inheriting ? &other_copy : other_oh,
                                /*note_when_empty*/ false, 120.0f,
                                scheme_uses_symmetry(c.scheme, cs.custom_schemes),
                                nullptr, BroadcastField::SweepTarget,
                                /*allow_h*/ !scheme_is_map(c.scheme));
        InputNumStr((std::string("lo##") + tag).c_str(), lo, kFieldW);
        InputNumStr((std::string("hi##") + tag).c_str(), hi, kFieldW);
        ImGui::Checkbox((std::string("Log scale##") + tag + "log").c_str(), &lg);
        if (lg) {
            ImGui::SameLine();
            ImGui::TextDisabled(inheriting ? "(from Level 2D)" : "(lo/hi > 0)");
        }
        if (inheriting) ImGui::EndDisabled();
        InputNumStr((std::string("N##") + tag).c_str(), n_text, kFieldW);

        // Обратная запись только в своём режиме — в inherit-режиме источник
        // истины остаётся за осями Level 2D.
        if (!inheriting) {
            own_par = par; own_ov  = ov; own_vi = vi;
            own_oh  = oh;  own_log = lg;
            own_lo  = lo;  own_hi  = hi;
        }
    };

    sweep_row("X-sweep:", "sx", sx,
              c.sweep_x_par_index, c.sweep_x_over_var, c.sweep_x_var_index,
              c.sweep_x_over_h, c.sweep_x_log,
              c.sweep_x_lo_text, c.sweep_x_hi_text,
              &c.sweep_y_over_h, c.n_x_1d_text);

    sweep_row("Y-sweep:", "sy", sy,
              c.sweep_y_par_index, c.sweep_y_over_var, c.sweep_y_var_index,
              c.sweep_y_over_h, c.sweep_y_log,
              c.sweep_y_lo_text, c.sweep_y_hi_text,
              &c.sweep_x_over_h, c.n_y_1d_text);

    // Строки выше могли поменять собственный свип L1D — перечитываем, чтобы
    // ползунки ниже снапились по сетке ЭТОГО кадра, а не прошлого.
    sx = effective_sweep_x(c);
    sy = effective_sweep_y(c);

    // L1D-specific integrator overrides — order TT before CT (per user
    // convention: transient runs first, then computing time is what's
    // actually sampled).
    ImGui::Separator();
    ImGui::TextUnformatted("L1D integrator (independent of shared):");
    InputNumStr("h##l1d",              c.l1d_h_text, kFieldW);
    InputNumStr("transient time##l1d", c.l1d_transient_text, kFieldW);
    InputNumStr("computing time##l1d", c.l1d_t_max_text, kFieldW);

    ImGui::Separator();
    // Slice sliders — clamped to the current effective ranges. Values snap to the FINER of the two
    // grids per axis: L2D `resolution_text` (N×N heatmap pixels) and L1D `n_{x,y}_1d_text` (slice
    // samples). When the user cranks 1D resolution up (typical — L1D is cheap, 2D is expensive),
    // the slider gets finer too, so the crosshair on the X-slice plot lands on X-slice data points
    // instead of drifting between them; on the heatmap it may then sit between pixel columns, an
    // acceptable trade-off. SliderScalar<Double> stays in double throughout (SliderFloat would
    // downcast and land 0.2 as 0.20000000298023224). Drag only moves the crosshair; recompute
    // fires on IsItemDeactivatedAfterEdit.
    double fx_lo = parse_ratio_or(sx.lo_text, 0.0);
    double fx_hi = parse_ratio_or(sx.hi_text, 1.0);
    double fy_lo = parse_ratio_or(sy.lo_text, 0.0);
    double fy_hi = parse_ratio_or(sy.hi_text, 1.0);
    if (fx_hi < fx_lo) std::swap(fx_lo, fx_hi);
    if (fy_hi < fy_lo) std::swap(fy_lo, fy_hi);
    // Discrete slider via index-value trick: SliderScalar<Int> steps by 1 (thumb snaps hard, no
    // continuous drift), value = grid index. The format string is a LITERAL world-coord string (no
    // % specifier), so the bubble shows "0.158730" instead of "5". Bubble text is precomputed from
    // the CURRENT idx before SliderScalar runs — during drag it shows the previous frame's snapped
    // world value (1-frame lag on the text only; the thumb always sits on a grid node).
    int n_2d   = parse_int_or(c.resolution_text, 64); if (n_2d   < 2) n_2d   = 64;
    int n_1d_x = parse_int_or(c.n_x_1d_text,     64); if (n_1d_x < 2) n_1d_x = 64;
    int n_1d_y = parse_int_or(c.n_y_1d_text,     64); if (n_1d_y < 2) n_1d_y = 64;
    int n_snap_x = (n_1d_x > n_2d) ? n_1d_x : n_2d;
    int n_snap_y = (n_1d_y > n_2d) ? n_1d_y : n_2d;
        // При log-свипе узлы распределены лог-равномерно (sweep_value_at / getValueByIdx_log),
        // поэтому и шаг ползунка обязан идти по логарифму — иначе thumb прилипает к линейным
        // позициям, которых в данных нет, и крестик уезжает с узла. log при lo<=0 невалиден
        // (движок такой Run отклонит) — деградируем на линейную сетку, чтобы до Run не было NaN.
    auto log_ok = [](bool log_scale, double lo, double hi) {
        return log_scale && lo > 0.0 && hi > 0.0;
    };
    auto idx_from_world = [&](double v, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return 0;
        double i_d;
        if (log_ok(log_scale, lo, hi)) {
            if (!(v > 0.0)) return 0;
            const double l0 = std::log10(lo), l1 = std::log10(hi);
            i_d = (std::log10(v) - l0) / ((l1 - l0) / (double)(n - 1));
        } else {
            i_d = (v - lo) / ((hi - lo) / (double)(n - 1));
        }
        return std::clamp((int)std::round(i_d), 0, n - 1);
    };
    auto world_from_idx = [&](int i, double lo, double hi, int n, bool log_scale) {
        if (n < 2 || hi <= lo) return lo;
        return sweep_value_at(i, n, lo, hi, log_ok(log_scale, lo, hi),
                              /*reverse*/ false, /*continuation*/ false);
    };

    // Step-arrows + slider row. Arrows walk idx by ±1 (repeat on hold),
    // slider is the discrete int-index widget from above. Same pattern
    // for X and Y — factored into a lambda.
    auto arrow_row = [&](const char* id,
                         int& idx, int idx_min, int idx_max,
                         double lo, double hi, int n_snap, bool log_scale,
                         double& fix_value,
                         double& timer_to_bump,
                         const char* slider_label) {
        ImGui::PushID(id);
        ImGui::PushButtonRepeat(true);
        bool step_changed = false;
        if (ImGui::ArrowButton("l", ImGuiDir_Left)) {
            if (idx > idx_min) { --idx; step_changed = true; }
        }
        ImGui::SameLine(0.0f, 2.0f);
        if (ImGui::ArrowButton("r", ImGuiDir_Right)) {
            if (idx < idx_max) { ++idx; step_changed = true; }
        }
        ImGui::PopButtonRepeat();
        ImGui::PopID();
        ImGui::SameLine();
        char fmt[64];
        std::snprintf(fmt, sizeof(fmt), "%.6g", world_from_idx(idx, lo, hi, n_snap, log_scale));
        ImGui::SetNextItemWidth(240.0f);
        ImGui::SliderScalar(slider_label, ImGuiDataType_S32, &idx,
                            &idx_min, &idx_max, fmt);
        bool released     = ImGui::IsItemDeactivatedAfterEdit();
        bool slider_edit  = ImGui::IsItemActive() || released;
        // Only overwrite fix_value on real user interaction: otherwise the idx_from_world →
        // world_from_idx round-trip would re-snap a fix_value that arrived from the heatmap (on the
        // coarser 2D grid) to the nearest slider-grid node — potentially the finer 1D grid —
        // silently drifting it off the 2D pixel the user just clicked.
        if (slider_edit || step_changed)
            fix_value = world_from_idx(idx, lo, hi, n_snap, log_scale);
        // Arrow clicks fire the same debounce path as slider release —
        // step-changed → immediate commit, no need to wait for release.
        // Bump only the caller-supplied timer so the settled branch can
        // enqueue just the slices that actually depend on this axis.
        if (released || step_changed)
            timer_to_bump = ImGui::GetTime();
    };

    int idx_x = idx_from_world(c.fix_x_value, fx_lo, fx_hi, n_snap_x, sx.log_scale);
    arrow_row("##fix_x_arr", idx_x, 0, n_snap_x - 1,
              fx_lo, fx_hi, n_snap_x, sx.log_scale,
              c.fix_x_value, c.last_fix_x_change_time, "fix X");

    int idx_y = idx_from_world(c.fix_y_value, fy_lo, fy_hi, n_snap_y, sy.log_scale);
    arrow_row("##fix_y_arr", idx_y, 0, n_snap_y - 1,
              fy_lo, fy_hi, n_snap_y, sy.log_scale,
              c.fix_y_value, c.last_fix_y_change_time, "fix Y");

    ImGui::Separator();
    ImGui::TextUnformatted("Enable slices:");
    ImGui::Checkbox("Bif-X", &c.bif1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("Bif-Y", &c.bif1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-X", &c.lle1d_x_enabled); ImGui::SameLine();
    ImGui::Checkbox("LLE-Y", &c.lle1d_y_enabled); ImGui::SameLine();
    ImGui::Checkbox("LS-X",  &c.ls1d_x_enabled);  ImGui::SameLine();
    ImGui::Checkbox("LS-Y",  &c.ls1d_y_enabled);

    ImGui::Separator();
    // Переменная БД — read-only эхо настройки Level 2D (срез строится по той
    // же переменной, что и карта). Меняется в панели Level 2D.
    {
        const char* wv_name =
            (c.bif_writable_var == -1) ? "combination"
          : (c.bif_writable_var >= 0 && c.bif_writable_var < (int)cs.vars.size())
                ? cs.vars[c.bif_writable_var].c_str() : "?";
        ImGui::TextDisabled("Bif variable: %s (from Level 2D)", wv_name);
    }
    // Так же и eps/NT для LLE/LS-срезов: read-only эхо слота 2D, правятся в
    // панели Level 2D (см. apply_shared_to_lle1d — срез считается с ними же).
    if ((c.lle1d_x_enabled || c.lle1d_y_enabled) && !cs.lle_session.curves.empty()) {
        ImGui::TextDisabled("LLE eps / NT: %s / %s (from Level 2D)",
                            cs.lle_session.curves[0].eps_text.c_str(),
                            cs.lle_session.curves[0].nt_text.c_str());
    }
    if ((c.ls1d_x_enabled || c.ls1d_y_enabled) && !cs.ls_session.curves.empty()) {
        ImGui::TextDisabled("LS eps / NT: %s / %s (from Level 2D)",
                            cs.ls_session.curves[0].eps_text.c_str(),
                            cs.ls_session.curves[0].nt_text.c_str());
    }
    // Отображение Y на Bif-срезах. Пересчёт не нужен — peak_times и
    // bifurcation_points приходят из одного прогона, поэтому пишем флаг прямо
    // в слоты [1]/[2] и просим автофит; Run не требуется.
    if (ImGui::Checkbox("Plot inter-peaks instead of peak values",
                        &c.plot_inter_peaks_1d)) {
        for (int slot = 1; slot <= 2; ++slot) {
            if ((int)cs.bif_session.diagrams.size() <= slot) break;
            cs.bif_session.diagrams[slot].plot_inter_peaks = c.plot_inter_peaks_1d;
            cs.bif_session.diagrams[slot].fit_request      = true;
        }
        cs.workspace.dirty = true;   // → пересохранение _last_custom.json
    }

    ImGui::Separator();
    ImGui::Checkbox("Continuation (1D)", &c.continuation_1d_enabled);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-recompute 1D on slider/sweep change", &c.auto_recompute_1d);
}


// Level 3 detail

void draw_level3_detail(AppModel& model, CustomSession& cs) {
    model.broadcast_source_tab = BroadcastTab::Custom;
    model.broadcast_source_idx = 0;
    auto& c = cs.shared;
    // Header title carries the level name — no SeparatorText here.
    ImGui::RadioButton("Phase + Time-domain", &c.level3_kind, 0); ImGui::SameLine();
    ImGui::RadioButton("Basins",              &c.level3_kind, 1);

    ImGui::Separator();
    ImGui::Checkbox("Auto-run", &c.autorun_on_drilldown);

    ImGui::Separator();
    if (c.level3_kind == 0) {
        // Phase: the full IC-sets / integrator / projections UI is embedded right here, so
        // everything lives inside the pipeline's L3 config (no separate top-level window). The
        // Analysis tab uses the same helper with model.phase_session; here we pass Custom's own
        // isolated cs.phase_session.
        draw_phase_controls(cs.phase_session, nullptr, &model);
    } else {
        // Basins sub-panel: IC-space axes + features. All of scheme/h/t_max/
        // etc. come from shared; DBSCAN eps stays per-config.
        if (cs.basins_session.configs.empty()) {
            ImGui::TextDisabled("Basins config not initialised.");
        } else {
            auto& bc = cs.basins_session.configs[0];
            ImGui::TextUnformatted("IC-space X var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##bxv",
                bc.axis_x_var >= 0 && bc.axis_x_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_x_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_x_var == i))
                        bc.axis_x_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##bx", bc.axis_x_lo_text, kFieldW);
            InputNumStr("hi##bx", bc.axis_x_hi_text, kFieldW);

            ImGui::TextUnformatted("IC-space Y var:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0f);
            if (ImGui::BeginCombo("##byv",
                bc.axis_y_var >= 0 && bc.axis_y_var < (int)cs.vars.size()
                    ? cs.vars[bc.axis_y_var].c_str() : "var")) {
                for (int i = 0; i < (int)cs.vars.size(); ++i)
                    if (ImGui::Selectable(cs.vars[i].c_str(), bc.axis_y_var == i))
                        bc.axis_y_var = i;
                ImGui::EndCombo();
            }
            InputNumStr("lo##by", bc.axis_y_lo_text, kFieldW);
            InputNumStr("hi##by", bc.axis_y_hi_text, kFieldW);
            InputNumStr("N##bxy", bc.n_pts_text, kFieldW);

            // Переменная, чью траекторию читает feature-экстрактор. Своя, а не от Level 2D: у
            // Basins другая задача (оси — IC-пространство), и в отдельной вкладке Basins этот
            // выбор тоже свой. Без комбо в Custom всегда шла первая переменная.
            draw_writable_var_combo(cs.vars, bc.writable_var, "Writable var##custom_bas_wv");

            ImGui::InputInt("feature1", &bc.feature1);
            ImGui::InputInt("feature2", &bc.feature2);
            InputNumStr("DBSCAN eps", bc.eps_dbscan_text, kFieldW);
        }
    }
}


// Pipeline column (master)

// draw_pipeline_column / level_status_badge жили здесь и не вызывались ни
// разу: колонка со списком уровней была заменена на level_header внутри
// draw_custom_controls, а вся её логика статусов переписана там заново.
// Удалены как мёртвый код.

 // namespace (draw helpers)

void draw_custom_controls(AppModel& model, SystemLibrary& lib) {
    (void)lib;
    auto& cs = model.custom_session;
    auto& c  = cs.shared;

    // Top row: Run + Stop + status, also bound to Ctrl+R (тот же шорткат, что у локального Run
    // в Phase/Bif controls). Dirty-tracked: строим сигнатуру каждого уровня, сравниваем с
    // последней закоммиченной и пропускаем enqueue, если ничего не менялось И прошлый прогон
    // удался. Новые сигнатуры кладутся в `pending_armed`, успешный poll промоутит
    // pending → committed (см. poll_and_commit ниже).
    bool run_now = ImGui::Button("Run");
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_now = true;
    if (run_now) {
        auto try_enqueue = [&](CustomSession::LevelSig& sig,
                               const std::string& current,
                               std::function<void()> do_enqueue) {
            if (current == sig.committed && sig.committed_ok) return;
            do_enqueue();
            sig.pending = current;
            sig.pending_armed = true;
        };
        try_enqueue(cs.sig_l2d, build_l2d_signature(c, cs),
                    [&](){ cs.enqueue_level_2d(model.custom_queue); });
        try_enqueue(cs.sig_l1d, build_l1d_signature(c, cs),
                    [&](){ cs.enqueue_level_1d(model.custom_queue); });
        try_enqueue(cs.sig_l3,  build_l3_signature (c, cs),
                    [&](){ cs.enqueue_level_3 (model.custom_queue); });
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop / clear queue")) {
        model.custom_session.request_cancel_all();
        model.custom_queue.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu queued)", model.custom_queue.size());

    // Auto-recompute 1D on slider settle, plus Phase (если L3=Phase и включён
    // autorun_on_drilldown), чтобы отпускание крестика или ползунка давало отклик сразу.
    // Дебаунс 200 мс + отсутствие расчёта в полёте. Per-axis таймеры
    // (last_fix_{x,y}_change_time) позволяют ставить в очередь ТОЛЬКО срезы, зависящие от
    // сдвинутой оси: drag fix_x пересчитывает Y-срезы (они фиксируют X при fix_x), drag fix_y —
    // X-срезы. Без разделения нетронутый срез пересчитывался бы с теми же данными и вызывал
    // autofit, сбрасывающий ручной зум. Drag хитмапы двигает ОБА таймера.
    if (c.level_1d_enabled && !cs.any_in_flight()) {
        double now = ImGui::GetTime();
        // 500 ms debounce: long enough that stepping a numeric field with
        // the arrow buttons (multiple ticks in quick succession) coalesces
        // into a single recompute instead of firing on every click.
        const double debounce_s = kCustomSliderDebounceSec;
        bool x_settled = c.last_fix_x_change_time > 0.0
                      && now - c.last_fix_x_change_time > debounce_s;
        bool y_settled = c.last_fix_y_change_time > 0.0
                      && now - c.last_fix_y_change_time > debounce_s;
        bool queue_free = model.custom_queue.empty();
        if ((x_settled || y_settled) && queue_free) {
            if (c.auto_recompute_1d) {
                // fix_x moved → Y-slices depend on fix_x. fix_y → X-slices.
                cs.enqueue_level_1d_partial(model.custom_queue,
                    /*x_slices*/ y_settled,
                    /*y_slices*/ x_settled);
            }
            // Pin the shared param_values from the EFFECTIVE sweep axes. Раньше это жило внутри
            // `if (autorun_on_drilldown && ...)` и всегда брало оси L2D, поэтому 1D-standalone
            // свип (inherit off) не пробрасывал drag ползунка обратно в общую секцию Parameters.
            // Теперь пиним безусловно на settle, по effective_sweep_x/y — корректно и для
            // L2D-inherit, и для L1D-standalone.
            // Ось может свипаться по параметру, по НУ или по шагу h; в общие поля отражаем только
            // param-случай (НУ и h подставляются в под-конфиги перед Run — pin_fixed_* в
            // app_model.cpp), а для h дополнительно обновляем общий h_text, чтобы панель
            // показывала реальный шаг расчёта.
            // Формат round-trip (fmt_num_shortest), а НЕ %.6g: v — значение узла сетки от
            // ucuda_node_value, уходящее в ядро через parse_num, и шесть цифр отрезали ~10 знаков,
            // из-за чего drill-down считался в параметре, в котором ячейка не считалась.
            auto pin_axis = [&](const EffectiveSweep& e, double v) {
                if (e.over_h) {
                    if (v > 0.0) c.l1d_h_text = fmt_num_shortest(v);
                    return;
                }
                if (e.over_var) return;
                if (e.par_index < 0 || e.par_index >= (int)cs.params.size()) return;
                c.param_values[cs.params[e.par_index]] = fmt_num_shortest(v);
            };
            EffectiveSweep esx = effective_sweep_x(c);
            EffectiveSweep esy = effective_sweep_y(c);
            if (x_settled) pin_axis(esx, c.fix_x_value);
            if (y_settled) pin_axis(esy, c.fix_y_value);
            if (c.autorun_on_drilldown && c.level_phase_enabled) {
                model.custom_queue.push_back({ c.level3_kind == 0
                    ? CustomQueueItem::Kind::Phase
                    : CustomQueueItem::Kind::Basins });
            }
            if (x_settled) c.last_fix_x_change_time = 0.0;
            if (y_settled) c.last_fix_y_change_time = 0.0;
        }
    }

    // Shared config panel — always visible at the top.
    draw_shared_config(model, cs, cs.custom_schemes);

    ImGui::Separator();

    // Each level = enable-checkbox + CollapsingHeader + status label; headers keep the
    // open/closed state the user left them in (SetNextItemOpen with FirstUseEver opens L2D on
    // the very first appearance, after that ImGui's own state wins). Several headers may be
    // open at once — there is no forced current level.
    // Level colour ladder: base = theme's Header colour, HUE rotated by +45°/level so the three
    // markers land on distinct points of the wheel (dark theme: blue → violet → magenta).
    // Saturation used to be index-tied, which read as one colour fading; both saturation and
    // value are now pinned to the base so all three keep the theme's brightness. Swapping `base`
    // for a user-picked primary makes the rotation cascade automatically.
    auto level_marker_color = [](int level_idx) {
        const ImVec4 base = ImGui::GetStyleColorVec4(ImGuiCol_Header);
        float h, s, v;
        ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
        const int   idx      = (level_idx >= 0 && level_idx < 3) ? level_idx : 0;
        const float hue_step = 45.0f / 360.0f;   // ImGui uses 0..1 for hue.
        float nh = h + hue_step * (float)idx;
        nh -= std::floor(nh);                    // wrap into [0, 1).
        // Lift saturation to full so the markers pop against ambient body
        // background (base Header colour is quite pale in default themes).
        const float ns = std::max(0.75f, s);
        const float nv = std::max(0.85f, v);
        ImVec4 col(0, 0, 0, 1.0f);
        ImGui::ColorConvertHSVtoRGB(nh, ns, nv, col.x, col.y, col.z);
        return col;
    };

    auto level_header = [&](int level_idx,
                            const char* enable_id,
                            bool& enabled,
                            const char* title,
                            bool running, bool has_result,
                            bool default_open,
                            const std::function<void()>& body) {
        ImGui::Checkbox(enable_id, &enabled);
        ImGui::SameLine();
        // Tinted status suffix on the header title itself so it's visible
        // even when the header is collapsed.
        const char* status =
            !enabled ? "off" :
            running  ? "running" :
            has_result ? "ok" : "idle";
        ImVec4 status_col(0.6f, 0.6f, 0.6f, 1.0f);
        if      (!std::strcmp(status, "running")) status_col = ImVec4(1.0f, 0.85f, 0.35f, 1.0f);
        else if (!std::strcmp(status, "ok"))      status_col = ImVec4(0.5f, 0.9f,  0.5f,  1.0f);
        else if (!std::strcmp(status, "off"))     status_col = ImVec4(0.5f, 0.5f,  0.5f,  1.0f);
        char header_label[128];
        std::snprintf(header_label, sizeof(header_label), "%s##hdr_%s", title, enable_id);
        ImGui::SetNextItemOpen(default_open, ImGuiCond_FirstUseEver);
        bool open = ImGui::CollapsingHeader(header_label);
        // Status label on the SAME line as the header (rendered AFTER the
        // header so it sits inline).
        ImGui::SameLine();
        ImGui::TextColored(status_col, "[%s]", status);
        if (open) {
            // Colored left-side bar spanning the body — always in peripheral
            // vision, so whichever setting you're looking at, the coloured
            // stripe on the left tells you which level it belongs to. Bar
            // is drawn AFTER the body so its height matches actual content.
            const float bar_w   = 2.5f;         // 1.5× thinner than before.
            const float bar_gap = 8.0f;         // px between bar and content.
            const float indent  = bar_w + bar_gap;
            const ImVec2 top_screen = ImGui::GetCursorScreenPos();
            ImGui::Indent(indent);
            body();
            ImGui::Unindent(indent);
            const ImVec2 bot_screen = ImGui::GetCursorScreenPos();
            // Pull bar bottom up by ItemSpacing so it doesn't run into the
            // next level's checkbox row (cursor after body sits at the row
            // start of what comes next).
            const float  bar_bot_y  = bot_screen.y - ImGui::GetStyle().ItemSpacing.y;
            const ImU32  col        = ImGui::ColorConvertFloat4ToU32(
                                          level_marker_color(level_idx));
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(top_screen.x,          top_screen.y),
                ImVec2(top_screen.x + bar_w,  bar_bot_y),
                col, 1.5f);
        }
    };

    // Derive status flags (same logic that used to live in draw_pipeline_column).
    bool level2_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index == 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   == 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    == 0);
    bool level2_hasres  =
        (!cs.bif_session.diagrams.empty() && cs.bif_session.diagrams[0].last_run_2d_ok) ||
        (!cs.lle_session.curves.empty()   && cs.lle_session.curves[0].last_run_2d_ok)   ||
        (!cs.ls_session.curves.empty()    && cs.ls_session.curves[0].last_run_2d_ok);
    bool level1_running =
        (cs.bif_session.in_flight && cs.bif_session.running_diagram_index > 0) ||
        (cs.lle_session.in_flight && cs.lle_session.running_curve_index   > 0) ||
        (cs.ls_session.in_flight  && cs.ls_session.running_curve_index    > 0);
    bool level1_hasres = false;
    if (cs.bif_session.diagrams.size() > 1 && cs.bif_session.diagrams[1].last_run_ok) level1_hasres = true;
    if (cs.bif_session.diagrams.size() > 2 && cs.bif_session.diagrams[2].last_run_ok) level1_hasres = true;
    if (cs.lle_session.curves.size()   > 1 && cs.lle_session.curves[1].last_run_ok)   level1_hasres = true;
    if (cs.lle_session.curves.size()   > 2 && cs.lle_session.curves[2].last_run_ok)   level1_hasres = true;
    if (cs.ls_session.curves.size()    > 1 && cs.ls_session.curves[1].last_run_ok)    level1_hasres = true;
    if (cs.ls_session.curves.size()    > 2 && cs.ls_session.curves[2].last_run_ok)    level1_hasres = true;
    bool level3_running = cs.phase_session.in_flight || cs.basins_session.in_flight;
    bool level3_hasres  = cs.phase_session.result.ok ||
                          (!cs.basins_session.configs.empty() && cs.basins_session.configs[0].last_run_ok);
    const char* l3_title = c.level3_kind == 0 ? "Level 3 - Phase / Time-domain" : "Level 3 - Basins";

    level_header(0, "##en_l2d", c.level_2d_enabled, "Level 2D - Bif / LLE / LS",
                 level2_running, level2_hasres, /*default_open=*/true,
                 [&](){ draw_level2d_detail(model, cs); });
    level_header(1, "##en_l1d", c.level_1d_enabled, "Level 1D - slices",
                 level1_running, level1_hasres, /*default_open=*/false,
                 [&](){ draw_level1d_detail(cs); });
    level_header(2, "##en_l3", c.level_phase_enabled, l3_title,
                 level3_running, level3_hasres, /*default_open=*/false,
                 [&](){ draw_level3_detail(model, cs); });
}


}
