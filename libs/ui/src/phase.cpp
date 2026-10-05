#include "gui_internal.h"

namespace ucuda::ui_detail {


// Единый счётчик режимов для статусных строк всех панелей. Раньше каждая панель считала
// "diverged" по flags[k] < 0, и подпись значила разное: у Bif1D/Bif2D отрицательный флаг —
// это fixed point, а у LLE/LS — расходимость. Теперь коды канонические (REGIME_* в
// configCUDA.h) и категории раскладываются одинаково везде.
void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc) {
    fp = unb = osc = 0;
    for (int f : flags) {
        if      (regime_is_fixed_point(f)) ++fp;
        else if (regime_is_unbound(f))     ++unb;
        else                               ++osc;
    }
}


// Подпись под "OK: ..." — печатается только если есть что показать кроме
// колебательного режима (иначе строка была бы шумом на каждом успешном run'е).
// Сводка адаптивного шага по свипу: st — по 4 числа на точку (принятые, отвергнутые,
// вынужденные шаги, средний h), пусто при постоянном шаге.
void draw_adaptive_stats(const std::vector<double>& st) {
    if (st.size() < 4) return;
    double acc = 0, rej = 0, frc = 0, hsum = 0, hmin = 0, hmax = 0;
    size_t n = 0;
    for (size_t i = 0; i + 3 < st.size(); i += 4) {
        acc += st[i]; rej += st[i + 1]; frc += st[i + 2];
        const double hm = st[i + 3];
        if (hm > 0) {
            if (n == 0 || hm < hmin) hmin = hm;
            if (n == 0 || hm > hmax) hmax = hm;
            hsum += hm; ++n;
        }
    }
    const double att = acc + rej;
    ImGui::TextDisabled("Adaptive step: mean h %.3g (%.3g..%.3g over points), %.0f accepted, "
                        "%.0f rejected (%.1f%%), %.0f forced",
                        n ? hsum / (double)n : 0.0, hmin, hmax, acc, rej,
                        att > 0 ? 100.0 * rej / att : 0.0, frc);
}


void draw_regime_summary(const std::vector<int>& flags) {
    int fp = 0, unb = 0, osc = 0;
    count_regimes(flags, fp, unb, osc);
    if (fp == 0 && unb == 0) return;
    ImGui::TextDisabled("regimes: %d fixed point / %d unbound / %d oscillation",
                        fp, unb, osc);
}


// Подпись НУ для легенды (галка "Legend shows initial conditions"). Берём ЧИСЛА, реально
// ушедшие в расчёт (result.snapshot.ic_flat), и форматируем через fmt_tick, т.е. с Tick
// precision из Settings: так сгенерированные НУ (узлы сетки бассейнов пишутся с %.17g) не
// разносят легенду на пол-экрана, а смена точности применяется сразу, без пересчёта.
// Fallback на сохранённый result.ic_text — если снапшота нет.
[[nodiscard]] std::string ic_legend_text(const AnalysisResult& res, size_t k) {
    if (k < res.snapshot.ic_flat.size() && !res.snapshot.ic_flat[k].empty()) {
        const std::vector<double>& v = res.snapshot.ic_flat[k];
        std::string txt = "(";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) txt += ",";
            txt += fmt_tick(v[i]);
        }
        txt += ")";
        return txt;
    }
    return (k < res.ic_text.size()) ? res.ic_text[k] : std::string();
}


void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin ,
                                    const ProjHookFn& after_begin  ,
                                    const std::string& title_suffix ,
                                    int owner_id_delta ,
                                    const PhaseStyleFn& style_fn ,
                                    const FeatureClusterParams& clust ) {
    const AnalysisResult& res = s.result;
    // Lambda installed on every projection view that has data — right-click "Export data..."
    // writes the full double-precision trajectory set (all coords for all ICs) + a _config.csv
    // sidecar. Ставится и на Phase3D: Plot3DView тоже поддерживает popup_extras, поэтому
    // экспорт доступен во всех трёх типах проекций одинаково.
    const bool phase_busy = s.in_flight;
    auto phase_popup_extras = [&res, &cb, phase_busy]() {
        const bool has_data = res.ok && !res.trajectories.empty();
        draw_export_menu_item(!has_data || phase_busy, cb, [&res](const std::string& p) {
            data_export::export_phase(res, res.snapshot, p);
        });
    };
    // Свой offscreen-рендерер (FBO/текстура) на КАЖДУЮ проекцию: иначе все окна показывали бы
    // одну общую текстуру (геометрию последней отрисованной). PlotRenderer некопируемый — храним
    // через unique_ptr и подгоняем под число проекций. Кэш ключуется по owner_id_delta: у функции
    // появился второй потребитель (Basins → фазовые портреты, свой delta на каждый config), и с
    // общим вектором переключение между ними пересоздавало бы FBO каждый кадр; Analysis передаёт 0.
    // Давно не рисовавшиеся владельцы вытесняются: каждый держит по FBO на проекцию (мегабайты
    // видеопамяти), а число систем/config'ов не ограничено.
    struct RendererBucket {
        std::vector<std::unique_ptr<PlotRenderer>> v;
        int last_frame = 0;
    };
    auto& renderers_by_owner = ui_resource<std::map<int, RendererBucket>, 1>();
    const int cur_frame = ImGui::GetFrameCount();
    for (auto it = renderers_by_owner.begin(); it != renderers_by_owner.end(); ) {
        if (it->first != owner_id_delta &&
            cur_frame - it->second.last_frame > kRendererEvictFrames)
            it = renderers_by_owner.erase(it);
        else
            ++it;
    }
    RendererBucket& bucket = renderers_by_owner[owner_id_delta];
    bucket.last_frame = cur_frame;
    std::vector<std::unique_ptr<PlotRenderer>>& renderers = bucket.v;
    if ((int)renderers.size() != (int)s.projections.size()) {
        renderers.clear();
        for (size_t k = 0; k < s.projections.size(); ++k)
            renderers.push_back(std::make_unique<PlotRenderer>());
    }
    // Запрос автоскейла раздаём по проекциям и сразу гасим сессионный: дальше
    // каждая проекция отвечает за свой флаг сама (см. Projection::fit_pending).
    if (s.fit_request) {
        for (auto& p : s.projections) p.fit_pending = true;
        s.fit_request = false;
    }
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        PlotRenderer& renderer = *renderers[i]; // рендерер этой проекции
        // ε-окружность включает ТОЛЬКО ветка FeatureDiagram ниже. Гасим её здесь каждый кадр:
        // view2d переживает смену типа проекции (объект переиспользуется, см. `if (!projection_views(pr).view2d)`),
        // и без сброса кружок остался бы висеть на фазовом портрете после переключения комбо.
        if (projection_views(pr).view2d) projection_views(pr).view2d->hover_circle_r = std::numeric_limits<double>::quiet_NaN();
        std::string title = pr.label + "##proj" + std::to_string(i) + "_g" + std::to_string(s.layout_generation) + title_suffix;
        bool open = true; // крестик закрытия
        // Начальные позиция и размер (только при первом появлении).
        // Каскад слева-сверху: каждое следующее окно чуть смещено.
        // ИЗМЕНИТЬ РАЗМЕР МОЖНО ЗДЕСЬ: ImVec2(ширина, высота) в пикселях.
        float ox = 60.0f + (i % 5) * 35.0f;
        float oy = 80.0f + (i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (before_begin) before_begin(i, title);
        if (ImGui::Begin(title.c_str(), &open)) {
            if (after_begin) after_begin(i, title);
            ImGui::PushID(i);   // разделить ID внутренних виджетов между окнами проекций
            if (!res.ok || res.trajectories.empty()) {
                ImGui::TextDisabled("No data. Press Recompute.");
            }
            ////////////////////////////////////////
            else if (pr.type == ProjType::Phase2D) {
                int ax = pr.axis_x, ay = pr.axis_y;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    // создать вьюер при первой отрисовке
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();

                    // обновить имена осей
                    projection_views(pr).view2d->x_axis.name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    projection_views(pr).view2d->y_axis.name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    // No zero-axis on phase 2D — x=0 / y=0 have no meaning
                    // for a state-space trajectory.
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = false;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). По дефолту выключено → быстрый
                    // GL shader-line путь (1px, α=1). Текущая отрисовка не ломается.
                    draw_proj_line_style(pr, "phase2d",
                        "Draw iterates as markers instead of a polyline.\n"
                        "Default for discrete maps: consecutive iterates are\n"
                        "not joined by a continuous path, so the connecting\n"
                        "segments are an artefact.");

                    // Серии: для каждой траектории берём координаты по (ax, ay). Буфер
                    // локальный — render() ниже забирает точки синхронно. static тут был
                    // бесполезен (clear()+resize() всё равно убивает capacity) и опасен:
                    // три вызова draw_projection_windows делили бы один буфер.
                    std::vector<std::vector<double>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.trajectories.size());

                    // видимость: глобальная — из живых галочек НУ (меняется без recompute)
                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];

                        // Per-IC стиль (Basins → фазовые портреты). as_point:
                        // траектория бассейна-равновесия после транзиента стоит
                        // в одной точке, поэтому кладём в буфер ТОЛЬКО последнюю
                        // — один маркер вместо десятков тысяч совпадающих точек.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);
                        const bool as_point  = has_style && st.as_point;

                        if (as_point) {
                            if (!traj.empty()) {
                                const auto& pt = traj.back();
                                buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        } else {
                            buf.reserve(traj.size() * 2);
                            for (const auto& pt : traj) {
                                buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                                buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            }
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color = has_style ? st.color : ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (rendering: ImDrawList использует color.w как alpha).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        if (as_point) {
                            si.points_override = 1;
                            si.point_marker    = st.marker;
                            si.point_size_px   = st.point_size;
                        }
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 100 + ax * 10 + ay;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Диаграмма признаков: точечный график (значение пика; интервал до
            // него) по переменной pr.axis_x. Пики уже посчитаны на worker'е
            // (AnalysisResult::features) тем же алгоритмом, что и GPU-peakFinder,
            // поэтому здесь только раскладка в буферы — пересчёта нет.
            else if (pr.type == ProjType::FeatureDiagram) {
                const int av = pr.axis_x;
                if (!res.ok || res.features.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();

                    // Оси кластеризации: те же множители, что уходят в dbscan.
                    // Без DBSCAN-конфига (clust.valid == false) остаются 1 и 1,
                    // т.е. сырые признаки, как было.
                    const double mp = clust.valid ? clust.mult_peak     : 1.0;
                    const double mi = clust.valid ? clust.mult_interval : 1.0;
                    auto mult_suffix = [](double m) -> std::string {
                        if (m == 1.0) return std::string();
                        char b[32]; std::snprintf(b, sizeof(b), " x %g", m);
                        return std::string(b);
                    };

                    // av == vars.size() — комбинация (см. комбо-бокс выше).
                    const int nv_ax = (int)s.vars.size();
                    std::string ax_name;
                    if (s.vars.empty())                     ax_name = "x";
                    else if (nv_ax >= 2 && av == nv_ax)     ax_name = combo_var_label(s.vars);
                    else                                    ax_name = s.vars[av < nv_ax ? av : 0];

                    projection_views(pr).view2d->x_axis.name = "peak " + ax_name + mult_suffix(mp);
                    projection_views(pr).view2d->y_axis.name = "IPI" + mult_suffix(mi);   // interpeak interval
                    // Интервал неотрицателен, а пики часто лежат вокруг нуля —
                    // нулевая линия по Y тут осмысленна, по X нет.
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = true;
                    // ε-окружность под курсором: радиус eps в ТЕХ ЖЕ осях, в которых
                    // уложены точки (множители уже применены), поэтому накрытые ею пики —
                    // то, что dbscan сольёт в один кластер. Без DBSCAN-конфига радиуса нет
                    // → NaN, ничего не рисуется. Ставится каждый кадр: immediate mode.
                    projection_views(pr).view2d->hover_circle_r = (clust.valid && clust.eps > 0.0)
                        ? clust.eps
                        : std::numeric_limits<double>::quiet_NaN();

                    draw_style_toolbar("Custom point style", "featdiag", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##featdiag", &pr.line_width, 0.5f, 8.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##featdiag",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    // Всегда точки, линий между пиками нет: соседние пики
                    // соединять нечем — это облако признаков, а не траектория.
                    projection_views(pr).view2d->imdraw_lines = false;

                    // Локальный буфер — см. series_data в ветке Phase2D выше.
                    std::vector<std::vector<double>> feat_data(res.features.size());
                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.features.size());
                    std::vector<bool> init_vis(res.features.size(), true);
                    std::vector<bool> glob_vis(res.features.size(), true);

                    size_t total_pts = 0;
                    for (size_t k = 0; k < res.features.size(); ++k) {
                        auto& buf = feat_data[k];
                        const auto& per_var = res.features[k];
                        if (av >= 0 && av < (int)per_var.size()) {
                            const FeaturePoints& fp = per_var[(size_t)av];
                            const size_t n = fp.peaks.size() < fp.intervals.size()
                                           ? fp.peaks.size() : fp.intervals.size();
                            buf.reserve(n * 2);
                            for (size_t p = 0; p < n; ++p) {
                                if (!std::isfinite(fp.peaks[p]) || !std::isfinite(fp.intervals[p]))
                                    continue;
                                buf.push_back((double)(fp.peaks[p]     * mp));
                                buf.push_back((double)(fp.intervals[p] * mi));
                            }
                            total_pts += buf.size() / 2;
                        }

                        std::string lab = (k < res.labels.size()) ? res.labels[k]
                                                                  : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        PlotSeriesInput si;
                        si.points   = buf.empty() ? nullptr : buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color    = has_style ? st.color : ic_base_color((int)k);
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.points_override = 1;
                        si.point_marker    = (int)PointMarker::Circle;
                        si.point_size_px   = pr.custom_line_style ? pr.line_width : 3.0f;
                        si.label = lab;
                        series_in.push_back(si);

                        glob_vis[k] = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        init_vis[k] = true;
                    }

                    if (total_pts == 0)
                        ImGui::TextDisabled("No peaks found (check transient / peak thresholds in Settings).");

                    // Множители и eps входят в gen-токен: иначе Plot2DView отдал бы
                    // уже залитый VBO, и правка коэффициентов не меняла бы картинку —
                    // ровно тот класс «настройка молча не применяется», что уже
                    // случался с --fmad.
                    int data_gen = s.data_generation * 100 + av;
                    if (clust.valid) {
                        auto mix = [](int acc, double v) {
                            char b[32]; std::snprintf(b, sizeof(b), "%.9g", v);
                            for (const char* p = b; *p; ++p)
                                acc = acc * 131 + (int)(unsigned char)*p;
                            return acc;
                        };
                        data_gen = mix(mix(mix(data_gen, clust.mult_peak),
                                           clust.mult_interval), clust.eps);
                    }

                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::TimeDomain) {
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();
                    // No zero-axis for time-domain — the X-axis is time,
                    // y=0 rarely coincides with a meaningful reference.
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = false;

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). Дефолт — быстрый GL shader-line
                    // путь (1px, α=1). Аналогично Phase2D.
                    draw_proj_line_style(pr, "timedomain",
                        "Draw the samples as markers instead of a polyline\n"
                        "(on step nodes: where the adaptive step actually landed).");

                    double h = parse_ratio_or(s.step_h, 0.01);  if (h <= 0) h = 0.01;
                    int dec = parse_int_or(s.decimation, 1);    if (dec < 1) dec = 1;
                    double dt = h * dec;
                    int nvars = (int)s.vars.size();

                    // Слотов nvars + 1: последний — комбинация (см. блок галочек).
                    if ((int)pr.show_var.size() != nvars + 1) {
                        std::vector<bool> prev = pr.show_var;
                        pr.show_var.assign((size_t)nvars + 1, true);
                        for (size_t j = 0; j < prev.size() && j < (size_t)nvars; ++j)
                            pr.show_var[j] = prev[j];
                        pr.show_var[(size_t)nvars] = false;
                    }

                    // step_h == 1 for a map, so dt above equals decimation: the
                    // X axis already counts iterations, only the label changes.
                    projection_views(pr).view2d->x_axis.name = s.sys.is_map ? "n" : "t";
                    projection_views(pr).view2d->y_axis.name = "value";

                    projection_views(pr).view2d->pad_x = false;
                    projection_views(pr).view2d->show_zero_x = false;

                    // серии: одна на (траектория k, видимая переменная vi).
                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<double>> series_data;

                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis;
                    std::vector<bool> glob_vis;

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        if (traj.empty()) continue;
                        int n = (int)traj.size();
                        // видимость НУ — из живой галочки (без recompute)
                        bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;

                        // Per-IC цвет (Basins → фазовые портреты). as_point тут
                        // намеренно игнорируется: во временной развёртке даже
                        // равновесие — это линия (константа), маркер не нужен.
                        PhaseSeriesStyle st;
                        const bool has_style = style_fn && style_fn((int)k, st);

                        // vi == nvars — слот комбинации; для dim < 2 он не
                        // предлагается и пропускается.
                        for (int vi = 0; vi <= nvars; ++vi) {
                            const bool is_combo = (vi == nvars);
                            if (is_combo && nvars < 2) continue;
                            if (vi < (int)pr.show_var.size() && !pr.show_var[vi]) continue;

                            series_data.emplace_back();
                            auto& buf = series_data.back();
                            buf.reserve(n * 2);
                            // Узлы адаптивного шага идут неравномерно: время — из
                            // результата, а не t * dt.
                            const std::vector<double>* tk =
                                (res.raw_nodes && k < res.times.size()) ? &res.times[k] : nullptr;
                            for (int t = 0; t < n; ++t) {
                                buf.push_back(tk && t < (int)tk->size() ? (*tk)[(size_t)t] : (double)(t * dt));
                                buf.push_back((double)(is_combo
                                    ? combo_var_value(traj[t], nvars)
                                    : traj[t][vi < (int)traj[t].size() ? vi : 0]));
                            }

                            std::string base = (k < res.labels.size()) ? res.labels[k] : ("IC" + std::to_string(k + 1));
                            std::string who = s.legend_show_ic ? ic_legend_text(res, k) : base;
                            std::string lab = (is_combo ? combo_var_label(s.vars) : s.vars[vi])
                                            + " [" + who + "]";

                            // Оттенок комбинации берём из расширенной палитры
                            // (nvars + 1), чтобы цвета самих переменных остались
                            // теми же, что были до появления слота.
                            const int shade_den = is_combo ? nvars + 1 : nvars;
                            PlotSeriesInput si;
                            si.points = buf.data();
                            si.n_points = n;
                            si.color = has_style ? shade_of(st.color, vi, shade_den)
                                                 : ic_var_shade((int)k, vi, shade_den);
                            // В custom_line_style режиме применяем α к цвету
                            // (ImDrawList использует color.w как alpha).
                            if (pr.custom_line_style) si.color.w = pr.alpha;
                            si.label = lab;
                            series_in.push_back(si);
                            init_vis.push_back(true);   // локальная (легенда) стартует с видимости НУ
                            glob_vis.push_back(ic_vis);   // глобальная = живая галочка НУ
                        }
                    }

                    int data_gen = s.data_generation * 1000 + (int)series_in.size();

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Recurrence plot + RQA. Матрица считается на GPU в воркере вместе с траекториями
            // (см. rqa::compute в hostLibrary.cu); здесь только показ. Переключение
            // Distance <-> Recurrence GPU не трогает: бинарная матрица получается порогом по
            // уже посчитанной dist, поэтому оно мгновенное.
            else if (pr.type == ProjType::RecurrencePlot) {
                if (!projection_views(pr).viewrp) {
                    projection_views(pr).viewrp = std::make_unique<HeatmapView>();
                    // Тот же приём, что у get_or_create_heatmap: сохранённый в проекции выбор
                    // имеет приоритет, -1 отдаёт дефолт. Общий model.heatmap_colormap сюда не
                    // тянем — он шарится Bif/LLE/LS, и RQA к ним не привязана (как и Basins).
                    projection_views(pr).viewrp->colormap =
                        (HeatmapColormap)colormap_id_or(pr.rqa_colormap, (int)kDefaultColormap);
                    projection_views(pr).viewrp->reverse_colormap = pr.rqa_reverse_cmap;
                    projection_views(pr).viewrp->autoscale         = pr.rqa_autoscale;
                    projection_views(pr).viewrp->manual_vmin_text  = pr.rqa_cbar_vmin_text;
                    projection_views(pr).viewrp->manual_vmax_text  = pr.rqa_cbar_vmax_text;
                }

                // "(?)" с тултипом — в gui.cpp нет общего HelpMarker'а, идиома тут
                // IsItemHovered + SetTooltip (см. подсказки к схемам КРС и к шкале дБ).
                auto hint = [](const char* t) {
                    ImGui::SameLine(); ImGui::TextDisabled("(?)");
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", t);
                };

                RqaJob job;
                const bool have_job = make_rqa_job(pr, job);
                const rqa::Result* rq = have_job ? find_rqa(res, job) : nullptr;

                // ---------------- toolbar ----------------
                ImGui::SetNextItemWidth(150);
                const char* vmnames[] = { "Distance matrix", "Recurrence plot" };
                ImGui::Combo("##rqvm", &pr.rqa_view_mode, vmnames, IM_ARRAYSIZE(vmnames));
                ImGui::SameLine();
                if (ImGui::Button("RQA settings...")) ImGui::OpenPopup("rqa_cfg");
                ImGui::SameLine();
                if (!have_job) {
                    ImGui::TextDisabled("%s", "choose a state-vector source in the Projections list");
                }
                else if (!rq) {
                    // Через "%s": TextDisabled принимает формат, и текст сюда приходит переменной.
                    ImGui::TextDisabled("%s", s.in_flight ? "computing..."
                                                          : "no result for these settings - press Recompute");
                }
                else if (!rq->ok) {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", rq->error.c_str());
                }
                else {
                    // n и dt печатаются всегда: ряд прорежен до n точек, и длины линий ниже —
                    // В ОТСЧЁТАХ этой выборки, а не в секундах.
                    ImGui::Text("n=%d   dt=%.4g s   eps=%.5g   RR=%.3f%%",
                                rq->n, rq->dt, rq->eps_used, 100.0 * rq->metrics.RR);
                }

                // ---------------- параметры ----------------
                if (ImGui::BeginPopup("rqa_cfg")) {
                    bool ch = false;
                    ImGui::SeparatorText("Embedding");
                    const char* snames[] = { "choose source...", "Full state vector", "Delay embedding" };
                    ImGui::SetNextItemWidth(180);
                    int si = (int)pr.rqa_source;
                    if (ImGui::Combo("source", &si, snames, IM_ARRAYSIZE(snames))) {
                        pr.rqa_source = (rqa::Source)si; ch = true;
                    }
                    if (pr.rqa_source == rqa::Source::Embedding) {
                        const int nv = (int)s.vars.size();
                        ImGui::SetNextItemWidth(180);
                        if (ImGui::BeginCombo("variable", s.vars.empty() ? "-" : s.vars[pr.rqa_var < nv ? pr.rqa_var : 0].c_str())) {
                            for (int k = 0; k < nv; ++k)
                                if (ImGui::Selectable(s.vars[k].c_str(), pr.rqa_var == k)) { pr.rqa_var = k; ch = true; }
                            ImGui::EndCombo();
                        }
                        ch |= InputNumStr("m (dimension)", pr.rqa_m_text, 100.0f);
                        ch |= InputNumStr("tau (samples)", pr.rqa_tau_text, 100.0f);
                    }

                    ImGui::SeparatorText("Threshold");
                    const char* nnames[] = { "Euclidean (L2)", "Maximum (Linf)", "Manhattan (L1)" };
                    ImGui::SetNextItemWidth(180);
                    int ni = (int)pr.rqa_norm;
                    if (ImGui::Combo("norm", &ni, nnames, IM_ARRAYSIZE(nnames))) { pr.rqa_norm = (rqa::Norm)ni; ch = true; }
                    const char* enames[] = { "absolute", "fraction of max distance",
                                             "fraction of sigma", "target recurrence rate" };
                    ImGui::SetNextItemWidth(180);
                    int ei = (int)pr.rqa_eps_mode;
                    if (ImGui::Combo("eps mode", &ei, enames, IM_ARRAYSIZE(enames))) { pr.rqa_eps_mode = (rqa::EpsMode)ei; ch = true; }
                    if (pr.rqa_eps_mode == rqa::EpsMode::Absolute)
                        ch |= InputNumStr("eps", pr.rqa_eps_text, 100.0f);
                    else if (pr.rqa_eps_mode == rqa::EpsMode::TargetRR)
                        ch |= InputNumStr("target RR, %", pr.rqa_rr_text, 100.0f);
                    else
                        ch |= InputNumStr("fraction", pr.rqa_eps_frac_text, 100.0f);

                    ImGui::SeparatorText("Colors");
                    // Отдельно от `ch`: цвет — это про отрисовку, пересчитывать из-за него RQA
                    // незачем (и в continuation это был бы лишний кадр GPU на каждый клик).
                    int cmap_idx = (int)projection_views(pr).viewrp->colormap;
                    if (colormap_combo("colormap", &cmap_idx, ImGui::GetFontSize() * 12.0f)) {
                        projection_views(pr).viewrp->colormap = (HeatmapColormap)cmap_idx;
                        pr.rqa_colormap = cmap_idx;
                    }
                    ImGui::Checkbox("reverse colormap", &projection_views(pr).viewrp->reverse_colormap);
                    hint("Inverts the scale. For the distance matrix this switches which end of "
                         "the scale reads as closeness of points; for the binary one, which level is dark.");

                    ImGui::Checkbox("autoscale color", &projection_views(pr).viewrp->autoscale);
                    hint("On: the scale is taken from the data - 0..max(D) for the distance matrix and "
                         "0..1 for the binary one. Off: the bounds are set manually in the fields below.\n"
                         "A manual range is useful for comparing two matrices on one scale, but "
                         "on the binary view it breaks the match between levels 0/1 and the colorbar bands.");
                    if (!projection_views(pr).viewrp->autoscale) {
                        // Ровно как в draw_heatmap_toolbar: текст парсится каждый кадр, значение
                        // кладётся в float-поля вью (parse_ratio_or понимает и дроби вида 1/3).
                        InputNumStr("vmin", projection_views(pr).viewrp->manual_vmin_text, 100.0f);
                        projection_views(pr).viewrp->manual_vmin =
                            (float)parse_ratio_or(projection_views(pr).viewrp->manual_vmin_text, projection_views(pr).viewrp->manual_vmin);
                        InputNumStr("vmax", projection_views(pr).viewrp->manual_vmax_text, 100.0f);
                        projection_views(pr).viewrp->manual_vmax =
                            (float)parse_ratio_or(projection_views(pr).viewrp->manual_vmax_text, projection_views(pr).viewrp->manual_vmax);
                    }

                    ImGui::SeparatorText("Lines");
                    ch |= InputNumStr("Theiler window", pr.rqa_theiler_text, 100.0f);
                    hint("Pairs with |i-j| <= w are excluded entirely. For flows this is "
                        "mandatory: without the window, points adjacent in time give false diagonals and inflate DET/LAM.");
                    ch |= InputNumStr("l_min", pr.rqa_lmin_text, 100.0f);
                    ch |= InputNumStr("v_min", pr.rqa_vmin_text, 100.0f);

                    ImGui::SeparatorText("Sampling");
                    ch |= InputNumStr("RP points (n)", pr.rqa_points_text, 100.0f);
                    hint("Side of the matrix. The trajectory is decimated uniformly to n points; the metrics "
                        "are computed on THE SAME matrix that is drawn. The ceiling is 4096 (4096^2 doubles = "
                        "134 MB both on the GPU and in host memory).\n"
                        "Measured on an RTX: 512 ~ 3 ms, 1024 ~ 6 ms, 2048 ~ 20 ms, 4096 ~ 70 ms.");
                    ch |= ImGui::Checkbox("network measures (clustering, transitivity)", &pr.rqa_network);
                    hint("Counting triangles of the recurrence network: O(RR*n^3). Measured: +30 ms at "
                        "n=2048, +85 ms at n=4096 on top of the usual computation. That is why it is off by default.");
                    ch |= ImGui::Checkbox("compute in continuation mode", &pr.rqa_in_continuation);
                    hint("In continuation a frame comes once every ~50 ms, and RQA at n=2048 costs ~20 ms of that "
                        "(at 4096 it is 70 ms, i.e. the frame no longer holds). If it does not fit, clear the "
                        "checkbox: trajectories keep being computed, RQA freezes until the next ordinary Recompute.");

                    if (ch) s.rqa_dirty = true;
                    ImGui::EndPopup();
                }

                // Инверсию можно переключить и из right-click меню хитмапы, поэтому её текущее
                // состояние забираем из вью, а не наоборот (см. комментарий у rqa_reverse_cmap).
                pr.rqa_reverse_cmap = projection_views(pr).viewrp->reverse_colormap;
                pr.rqa_autoscale    = projection_views(pr).viewrp->autoscale;
                pr.rqa_cbar_vmin_text = projection_views(pr).viewrp->manual_vmin_text;
                pr.rqa_cbar_vmax_text = projection_views(pr).viewrp->manual_vmax_text;

                if (!rq || !rq->ok) {
                    ImGui::TextDisabled("No RQA data.");
                }
                else {
                    // ---------------- метрики ----------------
                    // NaN печатается прочерком, а не нулём: "линий такой длины не нашлось" и
                    // "линии есть, но метрика равна нулю" — разные утверждения.
                    if (ImGui::CollapsingHeader("Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
                        const rqa::Metrics& M = rq->metrics;
                        struct Row { const char* name; double val; const char* tip; };
                        const Row rows[] = {
                            { "RR",     M.RR,     "Recurrence rate: fraction of recurrent pairs outside the Theiler window" },
                            { "DET",    M.DET,    "Determinism: fraction of points lying on diagonals of length >= l_min" },
                            { "L",      M.L,      "Mean diagonal line length, in samples" },
                            { "L_max",  M.L_max,  "Longest diagonal (excluding the LOI), in samples" },
                            { "DIV",    M.DIV,    "1 / L_max" },
                            { "ENTR",   M.ENTR,   "Shannon entropy of the diagonal length distribution, nats" },
                            { "RATIO",  M.RATIO,  "DET / RR" },
                            { "LAM",    M.LAM,    "Laminarity: fraction of points on verticals of length >= v_min" },
                            { "TT",     M.TT,     "Trapping time: mean vertical, in samples" },
                            { "V_max",  M.V_max,  "Longest vertical, in samples" },
                            { "V_ENTR", M.V_ENTR, "Entropy of the vertical length distribution, nats" },
                            { "TREND",  M.TREND,  "Slope of the linear regression of RR on diagonal index (RR per sample)" },
                            { "T1",     M.T1,     "Recurrence time of the first kind, in samples" },
                            { "T2",     M.T2,     "Recurrence time of the second kind (from the starts of vertical blocks)" },
                            { "W",      M.W,      "Mean white vertical length, in samples" },
                            { "W_max",  M.W_max,  "Longest white vertical, in samples" },
                            { "RTE",    M.RTE,    "Recurrence time entropy, normalised to [0,1]" },
                        };
                        if (ImGui::BeginTable("rqa_metrics", 6,
                                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
                            for (int r = 0; r < IM_ARRAYSIZE(rows); ++r) {
                                if (r % 3 == 0) ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("%s", rows[r].name);
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", rows[r].tip);
                                ImGui::TableNextColumn();
                                if (std::isnan(rows[r].val)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", rows[r].val);
                            }
                            if (M.network_valid) {
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn(); ImGui::TextDisabled("C");
                                ImGui::TableNextColumn();
                                if (std::isnan(M.clustering)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", M.clustering);
                                ImGui::TableNextColumn(); ImGui::TextDisabled("Trans");
                                ImGui::TableNextColumn();
                                if (std::isnan(M.transitivity)) ImGui::TextDisabled("--");
                                else ImGui::Text("%.5g", M.transitivity);
                            }
                            ImGui::EndTable();
                        }
                    }

                    // ---------------- сама матрица ----------------
                    const int n = rq->n;
                    const double* values = rq->dist.data();
                    double vlo = 0.0, vhi = rq->dist_max;

                    if (pr.rqa_view_mode == 1) {
                        // Бинарная матрица строится порогом по dist и кэшируется в проекции:
                        // GPU для смены режима не нужен. Ключ кэша — data_generation: любое новое
                        // значение dist приходит только вместе с новым расчётом.
                        if (pr.rqa_binary_gen != s.data_generation ||
                            pr.rqa_binary.size() != rq->dist.size()) {
                            pr.rqa_binary.resize(rq->dist.size());
                            for (size_t t = 0; t < rq->dist.size(); ++t)
                                pr.rqa_binary[t] = (rq->dist[t] <= rq->eps_used) ? 1.0 : 0.0;
                            pr.rqa_binary_gen = s.data_generation;
                        }
                        values = pr.rqa_binary.data();
                        vlo = 0.0; vhi = 1.0;
                    }

                    // discrete форсим ТОЛЬКО в кадр смены режима: иначе каждый кадр затирался бы
                    // ручной toggle "Discrete colorbar" из right-click меню HeatmapView.
                    if (pr.rqa_view_mode_applied != pr.rqa_view_mode) {
                        projection_views(pr).viewrp->discrete = (pr.rqa_view_mode == 1);
                        projection_views(pr).viewrp->discrete_levels = (pr.rqa_view_mode == 1) ? 2 : 0;
                        projection_views(pr).viewrp->discrete_levels_text = (pr.rqa_view_mode == 1) ? "2" : "0";
                        pr.rqa_view_mode_applied = pr.rqa_view_mode;
                    }

                    projection_views(pr).viewrp->x_axis.name = s.sys.is_map ? "n_j" : "t_j, s";
                    projection_views(pr).viewrp->y_axis.name = s.sys.is_map ? "n_i" : "t_i, s";
                    // Свой экспорт вместо phase_popup_extras: из этого окна нужны матрица и
                    // метрики, а не траектории. Траекторный экспорт оставлен вторым пунктом —
                    // он тут тоже осмыслен (RQA считалась именно по ним).
                    projection_views(pr).viewrp->popup_extras = [&res, &cb, phase_busy, rq, &job]() {
                        if (ImGui::MenuItem("Export RQA...", nullptr, false, !phase_busy)) {
                            if (cb.pick_save_file_csv) {
                                const std::string path = cb.pick_save_file_csv();
                                if (!path.empty())
                                    data_export::export_rqa(*rq, job.cfg, job.ic, res.snapshot, path);
                            }
                        }
                        const bool has_traj = res.ok && !res.trajectories.empty();
                        draw_export_menu_item(!has_traj || phase_busy, cb, [&res](const std::string& p) {
                            data_export::export_phase(res, res.snapshot, p);
                        });
                    };

                    // Режим входит в data_generation: при переключении Distance <-> Recurrence
                    // текстуру надо перезалить, хотя расчёт тот же.
                    const int data_gen = s.data_generation * 2 + pr.rqa_view_mode;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    projection_views(pr).viewrp->render(renderer, origin, avail,
                                      (i ^ owner_id_delta) ^ 0x5251A0, data_gen,
                                      n, n, values,
                                      rq->t0, rq->t1, rq->t0, rq->t1,
                                      vlo, vhi, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::Phase3D) {
                int ax = pr.axis_x, ay = pr.axis_y, az = pr.axis_z;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!projection_views(pr).view3d) projection_views(pr).view3d = std::make_unique<Plot3DView>();

                    // Toolbar над плотом: opt-in custom line styling (толщина + α). В 3D нет
                    // ImDrawList-fallback (потерялся бы depth-sorting), толщина идёт через
                    // glLineWidth и может клампиться драйвером, α точно уходит в шейдер.
                    // При выключенной фиче восстанавливаем прежний хардкод 1.5f.
                    draw_style_toolbar("Custom line style", "phase3d", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Line width##phase3d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##phase3d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    ImGui::SameLine();
                    ImGui::Checkbox("Points##phase3d", &pr.draw_points);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Draw iterates as markers instead of a polyline.\n"
                                          "Default for discrete maps.");
                    if (pr.draw_points) {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(110);
                        ImGui::SliderFloat("size##phase3dpt", &pr.point_size, 0.5f, 8.0f, "%.1f");
                        if (pr.custom_line_style) {   // своя альфа точек (см. draw_proj_line_style)
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(110);
                            ImGui::SliderFloat("alpha##phase3dpt", &pr.point_alpha, 0.0f, 1.0f, "%.2f");
                        }
                    }
                    projection_views(pr).view3d->point_alpha = (pr.draw_points && pr.custom_line_style) ? pr.point_alpha : -1.0f;
                    projection_views(pr).view3d->line_thickness_px = pr.custom_line_style ? pr.line_width : 1.5f;
                    projection_views(pr).view3d->custom_line_style = pr.custom_line_style;
                    projection_views(pr).view3d->points_mode       = pr.draw_points;
                    projection_views(pr).view3d->points_with_lines = pr.draw_points && pr.custom_line_style;
                    projection_views(pr).view3d->point_size_px     = pr.point_size;

                    projection_views(pr).view3d->x_name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    projection_views(pr).view3d->y_name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    projection_views(pr).view3d->z_name = s.vars.empty() ? "z" : s.vars[az < (int)s.vars.size() ? az : 0];

                    // Локальный буфер — см. ветку Phase2D выше.
                    std::vector<std::vector<float>> series_data(res.trajectories.size());

                    std::vector<PlotSeriesInput3D> series_in;
                    series_in.reserve(res.trajectories.size());

                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 3);
                        for (const auto& pt : traj) {
                            buf.push_back((double)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((double)pt[ay < (int)pt.size() ? ay : 0]);
                            buf.push_back((double)pt[az < (int)pt.size() ? az : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic) lab = ic_legend_text(res, k);

                        PlotSeriesInput3D si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 3);
                        si.color = ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (draw_line_3d прокидывает color[3] в шейдер как альфа).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 1000 + ax * 100 + ay * 10 + az;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    projection_views(pr).view3d->popup_extras = phase_popup_extras;
                    projection_views(pr).view3d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            else if (pr.type == ProjType::ContinuationDiagram) {
                const int av = pr.axis_x;
                if (s.continuation_peaks.empty()) {
                    ImGui::TextDisabled("No data. Press Recompute (Continuation (live) accumulates across runs).");
                }
                else {
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();

                    const int nv_ax = (int)s.vars.size();
                    std::string ax_name;
                    if (s.vars.empty())                     ax_name = "x";
                    else if (nv_ax >= 2 && av == nv_ax)     ax_name = combo_var_label(s.vars);
                    else                                    ax_name = s.vars[av < nv_ax ? av : 0];

                    projection_views(pr).view2d->x_axis.name = s.sys.is_map ? "n" : "t, s";
                    projection_views(pr).view2d->y_axis.name = "peak " + ax_name;
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = true;

                    draw_style_toolbar("Custom point style", "contdiag", pr.custom_line_style,
                        [&pr]() {
                            bool ch = false;
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Point size##contdiag", &pr.line_width, 0.5f, 8.0f, "%.1f");
                            ImGui::SetNextItemWidth(150);
                            ch |= ImGui::SliderFloat("Alpha##contdiag",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                            return ch;
                        });
                    projection_views(pr).view2d->imdraw_lines = false;

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(s.continuation_peaks.size());
                    std::vector<bool> init_vis(s.continuation_peaks.size(), true);
                    std::vector<bool> glob_vis(s.continuation_peaks.size(), true);

                    size_t total_pts = 0;
                    for (size_t k = 0; k < s.continuation_peaks.size(); ++k) {
                        const auto& per_var = s.continuation_peaks[k];
                        const double* pts = nullptr;
                        int          np  = 0;
                        if (av >= 0 && av < (int)per_var.size()) {
                            const auto& buf = per_var[(size_t)av];
                            pts = buf.empty() ? nullptr : buf.data();
                            np  = (int)(buf.size() / 2);
                            total_pts += (size_t)np;
                        }

                        std::string lab = (k < res.labels.size()) ? res.labels[k]
                                                                  : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic && k < res.ic_text.size())
                            lab = ic_legend_text(res, k);

                        PlotSeriesInput si;
                        si.points   = pts;
                        si.n_points = np;
                        si.color    = ic_base_color((int)k);
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.points_override = 1;
                        si.point_marker    = (int)PointMarker::Circle;
                        si.point_size_px   = pr.custom_line_style ? pr.line_width : 3.0f;
                        si.label = lab;
                        series_in.push_back(si);

                        glob_vis[k] = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        init_vis[k] = true;
                    }

                    if (total_pts == 0)
                        ImGui::TextDisabled("Waiting for peaks... (adjust transient / peak thresholds if needed).");

                    int data_gen = s.continuation_peaks_gen * 100 + av;

                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // h(t) адаптивного шага: принятые шаги линией, отвергнутые попытки —
            // крестиками, вынужденные (на h_min) — квадратами. Данные — лог попыток.
            else if (pr.type == ProjType::StepSize) {
                if (!res.ok || !res.adaptive || res.step_log.empty()) {
                    ImGui::TextDisabled("No data. The step-size plot needs the adaptive step "
                                        "(Step size -> Adaptive step).");
                }
                else {
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();
                    // log Y — пересчётом в log10 h (лог-оси Y у Plot2DView нет), как у окон Order.
                    const bool prev_ylog = pr.y_log;
                    ImGui::Checkbox("log Y##stepsize", &pr.y_log);
                    if (pr.y_log != prev_ylog) pr.fit_pending = true;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Y axis in log10 h: steps spread over orders of magnitude\n"
                                          "(stiff parts, rejections near h_min) become readable.");
                    ImGui::SameLine();
                    draw_proj_line_style(pr, "stepsize",
                        "Draw the accepted steps as markers instead of a line.");
                    projection_views(pr).view2d->x_axis.name = "t";
                    projection_views(pr).view2d->y_axis.name = pr.y_log ? "log10 h" : "h";
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = false;

                    std::vector<std::vector<double>> bufs;
                    bufs.reserve(res.step_log.size() * 3);   // указатели серий не должны переехать
                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis, glob_vis;
                    for (size_t k = 0; k < res.step_log.size(); ++k) {
                        const std::vector<double>& lg = res.step_log[k];
                        std::vector<double> acc, rej, frc;
                        for (size_t r = 0; r + 4 <= lg.size(); r += 4) {
                            const double t = lg[r], code = lg[r + 3];
                            double hh = lg[r + 1];
                            if (pr.y_log) {
                                if (!(hh > 0)) continue;   // log10 не определён
                                hh = std::log10(hh);
                            }
                            if (code == 0) { rej.push_back(t); rej.push_back(hh); continue; }
                            acc.push_back(t); acc.push_back(hh);
                            if (code == 2) { frc.push_back(t); frc.push_back(hh); }
                        }
                        const std::string who = (k < res.labels.size()) ? res.labels[k]
                                                                        : ("IC" + std::to_string(k + 1));
                        const bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        struct Part { std::vector<double>* v; const char* name; int marker; ImVec4 col; };
                        const Part parts[3] = {
                            { &acc, "h",        -1, ic_base_color((int)k) },
                            { &rej, "rejected", (int)PointMarker::Cross,  ImVec4(0.90f, 0.30f, 0.30f, 1.0f) },
                            { &frc, "forced",   (int)PointMarker::Square, ImVec4(1.00f, 0.65f, 0.20f, 1.0f) },
                        };
                        for (const Part& part : parts) {
                            if (part.v->empty()) continue;
                            bufs.push_back(std::move(*part.v));
                            PlotSeriesInput si;
                            si.points = bufs.back().data();
                            si.n_points = (int)(bufs.back().size() / 2);
                            si.color = part.col;
                            if (part.marker >= 0) {
                                si.points_override = 1;
                                si.point_marker    = part.marker;
                                si.point_size_px   = 4.0f;
                            }
                            else if (pr.custom_line_style) si.color.w = pr.alpha;   // α — линии, не маркерам
                            si.label = std::string(part.name) + " [" + who + "]";
                            series_in.push_back(si);
                            init_vis.push_back(true);
                            glob_vis.push_back(ic_vis);
                        }
                    }
                    int data_gen = s.data_generation * 1000 + (pr.y_log ? 507 : 7);   // log Y — другие данные
                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            // Отказы адаптивного шага: у каждого принятого шага — сколько попыток ему
            // отвергли (маркер, шаги без отказов не рисуются; вынужденные — квадратом), и
            // скользящее среднее по kWin принятым шагам — доля лишних вычислений схемы.
            // Точка стоит в начале шага. Данные — лог попыток, как у Step size.
            else if (pr.type == ProjType::Rejections) {
                if (!res.ok || !res.adaptive || res.step_log.empty()) {
                    ImGui::TextDisabled("No data. The rejections plot needs the adaptive step "
                                        "(Step size -> Adaptive step).");
                }
                else {
                    if (!projection_views(pr).view2d) projection_views(pr).view2d = std::make_unique<Plot2DView>();
                    draw_proj_line_style(pr, "rejections",
                        "Draw the moving mean as markers instead of a line.");
                    projection_views(pr).view2d->x_axis.name = "t";
                    projection_views(pr).view2d->y_axis.name = "rejected per step";
                    projection_views(pr).view2d->show_zero_x = false;
                    projection_views(pr).view2d->show_zero_y = false;

                    constexpr int kWin = 100;
                    std::vector<std::vector<double>> bufs;
                    bufs.reserve(res.step_log.size() * 3);   // указатели серий не должны переехать
                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis, glob_vis;
                    for (size_t k = 0; k < res.step_log.size(); ++k) {
                        const std::vector<double>& lg = res.step_log[k];
                        std::vector<double> rej, frc, mean;
                        int cnt = 0, wsum = 0, wn = 0, wpos = 0;
                        int win[kWin] = {};
                        for (size_t r = 0; r + 4 <= lg.size(); r += 4) {
                            const double code = lg[r + 3];
                            if (code == 0) { ++cnt; continue; }
                            const double t = lg[r];
                            if (code == 2) { frc.push_back(t); frc.push_back(cnt); }
                            else if (cnt > 0) { rej.push_back(t); rej.push_back(cnt); }
                            wsum += cnt - (wn == kWin ? win[wpos] : 0);
                            win[wpos] = cnt; wpos = (wpos + 1) % kWin;
                            if (wn < kWin) ++wn;
                            mean.push_back(t); mean.push_back((double)wsum / wn);
                            cnt = 0;
                        }
                        const std::string who = (k < res.labels.size()) ? res.labels[k]
                                                                        : ("IC" + std::to_string(k + 1));
                        const bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        struct Part { std::vector<double>* v; const char* name; int marker; ImVec4 col; };
                        const Part parts[3] = {
                            { &mean, "mean of 100 steps", -1, ic_base_color((int)k) },
                            { &rej,  "rejected",          (int)PointMarker::Cross,  ImVec4(0.90f, 0.30f, 0.30f, 1.0f) },
                            { &frc,  "forced",            (int)PointMarker::Square, ImVec4(1.00f, 0.65f, 0.20f, 1.0f) },
                        };
                        for (const Part& part : parts) {
                            if (part.v->empty()) continue;
                            bufs.push_back(std::move(*part.v));
                            PlotSeriesInput si;
                            si.points = bufs.back().data();
                            si.n_points = (int)(bufs.back().size() / 2);
                            si.color = part.col;
                            if (part.marker >= 0) {
                                si.points_override = 1;
                                si.point_marker    = part.marker;
                                si.point_size_px   = 4.0f;
                            }
                            else if (pr.custom_line_style) si.color.w = pr.alpha;   // α — линии, не маркерам
                            si.label = std::string(part.name) + " [" + who + "]";
                            series_in.push_back(si);
                            init_vis.push_back(true);
                            glob_vis.push_back(ic_vis);
                        }
                    }
                    int data_gen = s.data_generation * 1000 + 8;
                    ImVec2 avail  = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();
                    projection_views(pr).view2d->popup_extras = phase_popup_extras;
                    projection_views(pr).view2d->render(renderer, origin, avail, i ^ owner_id_delta, data_gen,
                        series_in, init_vis, glob_vis, pr.fit_pending);
                }
            }
            ImGui::PopID();
            // Окно реально отрисовалось в этом кадре — автоскейл до него дошёл.
            // Гасим здесь, а не после цикла: у невидимых окон Begin вернул
            // false, и их запрос должен дожить до кадра, когда их покажут.
            // Ветка "No data" тоже гасит: фитить нечего, а приход результата
            // взведёт fit_request заново.
            pr.fit_pending = false;
        }
        ImGui::End();
        if (!open) pr_to_remove = i; // окно закрыто крестиком
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
}


// Combo "Writable var": переменные + (через разделитель) комбинация
// x[0] + pi*x[1] + euler*x[2] (при |vars|=2 без e*x[2], при |vars|=1 — сама переменная).
// Sentinel в hostside int: -1 = комбинация. Используется в Bifurcation и Basins, где
// kernel пишет data[i] через loopCalculateDiscreteModel_int — он умеет оба режима.
void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id) {
    if (vars.empty()) return;
    const int N = (int)vars.size();
    // Допустимые значения: -1 (combination) или [0, N-1]. Старые сейвы могли
    // иметь невалидные индексы — clamp в 0.
    if (writable_var < -1 || writable_var >= N) writable_var = 0;
    // Для системы из одной переменной combination сворачивается до x[0] —
    // показывать отдельную опцию бессмысленно, форсим single-var.
    if (N == 1 && writable_var == -1) writable_var = 0;

    // Подпись комбинации — общая combo_var_label (раньше здесь жила её копия).
    const std::string preview =
        (writable_var == -1) ? combo_var_label(vars) : vars[writable_var];

    ImGui::SetNextItemWidth(220);
    if (ImGui::BeginCombo(combo_id, preview.c_str())) {
        for (int i = 0; i < N; ++i) {
            bool sel = writable_var == i;
            if (ImGui::Selectable(vars[i].c_str(), sel)) writable_var = i;
        }
        if (N >= 2) {
            ImGui::Separator();
            const std::string lbl = combo_var_label(vars);
            bool sel_combo = writable_var == -1;
            if (ImGui::Selectable(lbl.c_str(), sel_combo)) writable_var = -1;
        }
        ImGui::EndCombo();
    }
}
}
