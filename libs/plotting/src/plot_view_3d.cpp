#include "ucuda/plotting/plot_view_3d.h"
#include "imgui_internal.h"   // ImGui::MarkIniSettingsDirty (масштаб осей пишется в imgui.ini)
#include <algorithm>
#include <cmath>
#include <cstdio>

void Plot3DView::effective_scale(float aspect, float out[3]) const {
    out[0] = out[1] = out[2] = 1.0f;
    float b[6];
    if (fit_axes_to_window && series_cache_.bbox(b[0], b[1], b[2], b[3], b[4], b[5])) {
        // z — на высоту поля, x и y — на его ширину. Вырожденная ось (плоский портрет) остаётся 1.
        const float len[3] = { aspect, aspect, 1.0f };
        for (int i = 0; i < 3; ++i) {
            const float range = b[2 * i + 1] - b[2 * i];
            if (range > 1e-20f) out[i] = len[i] / range;
        }
        // Общий множитель не важен (ортографическая камера), но держим мир около единиц:
        // нормируем на масштаб z, чтобы смена размеров окна не уводила дистанцию камеры.
        const float ref = out[2];
        for (int i = 0; i < 3; ++i) out[i] /= ref;
    }
    for (int i = 0; i < 3; ++i) out[i] *= axis_scale[i] > 0.0f ? axis_scale[i] : 1.0f;
}

void Plot3DView::do_autofit() {
    float xmn, xmx, ymn, ymx, zmn, zmx;
    if (series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) {
        // Камера живёт в масштабированном мире — подгоняем по масштабированному bbox.
        effective_scale(aspect_, scale_);
        camera.fit_to_bbox(xmn * scale_[0], xmx * scale_[0], ymn * scale_[1], ymx * scale_[1],
                           zmn * scale_[2], zmx * scale_[2]);
        if (fit_axes_to_window) {
            // fit_to_bbox берёт расстояние по диагонали с запасом и не знает формы окна — в
            // широком поле растянутый портрет остался бы мелким. Ортографическая проекция
            // линейна по 1/distance: проецируем углы bbox при distance = 1 и берём наибольшую
            // NDC-координату — это и есть нужное расстояние (плюс 8 % поля).
            camera.aspect = aspect_;
            const float keep = camera.distance;
            camera.distance = 1.0f;
            float m[16];
            camera.build_mvp(m);
            const float b[6] = { xmn * scale_[0], xmx * scale_[0], ymn * scale_[1],
                                 ymx * scale_[1], zmn * scale_[2], zmx * scale_[2] };
            float need = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const float x = b[0 + (k & 1)], y = b[2 + ((k >> 1) & 1)], z = b[4 + ((k >> 2) & 1)];
                need = (std::max)(need, std::fabs(m[0] * x + m[4] * y + m[8] * z + m[12]));
                need = (std::max)(need, std::fabs(m[1] * x + m[5] * y + m[9] * z + m[13]));
            }
            camera.distance = need > 1e-12f ? need * 1.08f : keep;
        }
        view_valid = true;
    }
}

void Plot3DView::rebuild_axis_cache() {
    float xmn, xmx, ymn, ymx, zmn, zmx;
    if (!series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) return;

    // если bbox не изменился - не пересобираем
    if (axis_cache_.size() == 3 &&
        axis_bbox_[0] == xmn && axis_bbox_[1] == xmx &&
        axis_bbox_[2] == ymn && axis_bbox_[3] == ymx &&
        axis_bbox_[4] == zmn && axis_bbox_[5] == zmx) return;

    axis_cache_.clear();
    // Ось X: от (xmn, ymn, zmn) до (xmx, ymn, zmn)
    {
        float pts[6] = { xmn, ymn, zmn, xmx, ymn, zmn };
        axis_cache_.upload(pts, 2);
    }
    // Ось Y: от (xmn, ymn, zmn) до (xmn, ymx, zmn)
    {
        float pts[6] = { xmn, ymn, zmn, xmn, ymx, zmn };
        axis_cache_.upload(pts, 2);
    }
    // Ось Z: от (xmn, ymn, zmn) до (xmn, ymn, zmx)
    {
        float pts[6] = { xmn, ymn, zmn, xmn, ymn, zmx };
        axis_cache_.upload(pts, 2);
    }
    axis_bbox_[0] = xmn; axis_bbox_[1] = xmx;
    axis_bbox_[2] = ymn; axis_bbox_[3] = ymx;
    axis_bbox_[4] = zmn; axis_bbox_[5] = zmx;
}

void Plot3DView::render(PlotRenderer& renderer,
    ImVec2 block_origin, ImVec2 avail_size,
    int owner_id,
    int data_generation,
    const std::vector<PlotSeriesInput3D>& series_in,
    const std::vector<bool>& init_visible,
    const std::vector<bool>& global_visible,
    bool fit_request)
{
    // Toolbar окна (plot_axis.h) — ссылка здесь: в попапе текущим окном будет сам попап.
    PlotWindowPrefs& wp = plot_window_prefs_current();
    const bool axes_on   = show_axes && wp.axes3d;
    const bool legend_on = show_legend && wp.legend;
    // Масштаб осей хранится на окно (imgui.ini). Переключили режим (меню этого или другого
    // кадра, загрузка ini) — заново вписываем портрет.
    if (fit_axes_to_window != wp.fit3d) view_valid = false;
    fit_axes_to_window = wp.fit3d;
    for (int i = 0; i < 3; ++i) axis_scale[i] = wp.scale3d[i];

    // 1. Пересобираем кэш
    if (data_generation != series_generation) {
        bool count_changed = ((int)visible.size() != (int)series_in.size());
        series_cache_.clear();
        for (const auto& s : series_in)
            series_cache_.upload(s.points, s.n_points, s.values);
        series_generation = data_generation;

        if (count_changed) {
            visible.assign(series_in.size(), true);
            for (size_t k = 0; k < series_in.size() && k < init_visible.size(); ++k)
                visible[k] = init_visible[k];
        }
    }
    if (visible.size() != series_in.size())
        visible.resize(series_in.size(), true);

    auto eff_visible = [&](int k) -> bool {
        bool loc = (k < (int)visible.size()) ? visible[k] : true;
        bool glob = (k < (int)global_visible.size()) ? global_visible[k] : true;
        return loc && glob;
        };

    // Цвет серии: пользовательский (меню легенды) поверх цвета вызывающего, альфа — его.
    auto series_color = [&](const PlotSeriesInput3D& s) -> ImVec4 {
        auto it = wp.series_colors.find(s.label);
        if (it == wp.series_colors.end()) return s.color;
        return ImVec4(it->second.x, it->second.y, it->second.z, s.color.w);
        };

    // Записи легенды — до размеров поля: положение *outside забирает у него место.
    std::vector<LegendEntry> legend_entries;
    ImVec2 legend_sz(0.0f, 0.0f);
    if (legend_on) {
        legend_entries.reserve(series_in.size());
        for (const auto& s : series_in) {
            LegendEntry e{ s.label, series_color(s) };
            e.color.w = 1.0f;   // см. Plot2DView: ярлык не гаснет вместе с кривой
            legend_entries.push_back(e);
        }
        legend_sz = legend_size(legend_entries, global_visible, wp.legend_place.columns);
    }

    // Размер поля нужен до автофита: от его соотношения сторон зависит масштаб осей.
    const float margin = wp.fit_pane ? plot_compact_margin() : 4.0f;
    float margin_l = margin, margin_t = margin, margin_r = margin, margin_b = margin;
    if (legend_on)
        legend_reserve_margins(wp.legend_place.location, legend_sz, margin_l, margin_t, margin_r, margin_b);
    int plot_w = (std::max)(64, (int)(avail_size.x - margin_l - margin_r));
    int plot_h = (std::max)(64, (int)(avail_size.y - margin_t - margin_b));
    // Окно растянули при «Fit axes to window» — заново вписываем портрет (если автофит не
    // выключен: тогда пользовательский вид не трогаем, только масштаб осей ниже).
    const float aspect_now = (float)plot_w / (float)plot_h;
    if (fit_axes_to_window && auto_fit && aspect_now != aspect_) view_valid = false;
    aspect_ = aspect_now;

    // 2. Автофит. Запрос вызывающего (новые данные, смена осей) — только при auto_fit; первый
    // кадр и «Auto fit» из меню / двойной клик (view_valid = false) — всегда.
    if (!view_valid || (fit_request && auto_fit)) do_autofit();
    // Масштаб осей поменялся (окно растянули, новые данные, множитель в меню) — переносим
    // точку взгляда камеры в новый масштаб, чтобы вид не уезжал с портрета.
    {
        float s[3];
        effective_scale(aspect_, s);
        for (int i = 0; i < 3; ++i)
            if (s[i] != scale_[i]) {
                camera.target[i] *= s[i] / scale_[i];
                scale_[i] = s[i];
            }
    }

    // 3. Отступы и размеры (3D не нужен margin под подписи осей, они лежат внутри -
    //    но небольшой зазор от краёв оставляем, чтобы рамка ImGui не сливалась с полем)
    // Fit to pane — узкое поле, как у 2D и карт; без него — прежний зазор (размеры — выше).

    ImGui::Dummy(avail_size);
    ImVec2 img_pos = ImVec2(block_origin.x + margin_l, block_origin.y + margin_t);
    last_img_pos  = img_pos;
    last_img_size = ImVec2((float)plot_w, (float)plot_h);

    // 4. Рендер в FBO, с depth
    camera.aspect = (float)plot_w / (float)plot_h;
    float mvp[16];
    camera.build_mvp(mvp);
    // Масштаб осей — справа: MVP · diag(sx, sy, sz). Дальше всё (оси, подписи, OIT-глубина)
    // идёт в координатах данных через этот же mvp.
    for (int i = 0; i < 3; ++i)
        for (int r = 0; r < 4; ++r) mvp[i * 4 + r] *= scale_[i];

    float bgc[4];
    plot_bg_color(bgc[0], bgc[1], bgc[2], bgc[3]);
    // 3D-сцена чуть темнее dark / чуть тусклее light, чтобы axes-стрелки
    // оставались читабельными в обоих случаях.
    // Пользовательский фон и чёрный фон High Contrast берём как есть.
    if (!plot_bg_is_custom()) {
        if (plot_light_theme()) { bgc[0] = 0.965f; bgc[1] = 0.965f; bgc[2] = 0.965f; }
        else                    { bgc[0] = 0.050f; bgc[1] = 0.050f; bgc[2] = 0.080f; }
    }

    // Кадр перерисовывается, только если что-то из того, что в него входит, поменялось: у
    // каждой проекции свой PlotRenderer, и неподвижная сцена (камера стоит, данные те же)
    // иначе гоняла бы все сегменты через geometry shader каждый кадр ImGui.
    unsigned long long key = 1469598103934665603ull;   // FNV-1a
    auto mix = [&key](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { key ^= b[i]; key *= 1099511628211ull; }
    };
    auto mixv = [&mix](const auto& v) { mix(&v, sizeof(v)); };
    {
        const Plot3DView* self = this;
        mixv(self); mixv(plot_w); mixv(plot_h); mix(bgc, sizeof(bgc)); mix(mvp, sizeof(mvp));
        mixv(series_generation); mixv(axes_on); mixv(line_thickness_px);
        mixv(custom_line_style); mixv(points_mode); mixv(points_with_lines);
        mixv(point_size_px); mixv(point_alpha); mixv(point_marker);
        const int ns = (int)series_cache_.size();
        mixv(ns);
        for (int k = 0; k < ns; ++k) {
            const bool vis = eff_visible(k);
            mixv(vis);
            if (!vis || k >= (int)series_in.size()) continue;
            const PlotSeriesInput3D& s = series_in[k];
            const ImVec4 c = series_color(s);
            mixv(c.x); mixv(c.y); mixv(c.z); mixv(c.w);
            mixv(s.colormap); mixv(s.cmin); mixv(s.cmax); mixv(s.cmap_reverse);
        }
        if (key == 0) key = 1;   // 0 — «содержимого нет» у рендерера
    }
    const bool frame_cached = renderer.content_key() == key &&
                              renderer.width() == plot_w && renderer.height() == plot_h;

    if (!frame_cached) {
    renderer.begin_frame(plot_w, plot_h, bgc[0], bgc[1], bgc[2], bgc[3], /*with_depth=*/true);
    // Оси (X=красная, Y=зелёная, Z=синяя) — ДО данных: они непрозрачны и пишут глубину, поэтому
    // траектория за осью ею закрывается, а перед ней — рисуется поверх. Полупрозрачные данные
    // глубину не пишут (draw_line_3d), и оси, нарисованные после них, ложились бы поверх всего.
    if (axes_on) {
        rebuild_axis_cache();
        if (axis_cache_.size() == 3) {
            float col_x[4] = { 1.0f, 0.4f, 0.4f, 1.0f };
            float col_y[4] = { 0.4f, 1.0f, 0.4f, 1.0f };
            float col_z[4] = { 0.4f, 0.6f, 1.0f, 1.0f };
            const GpuLineSeries3D& ax = axis_cache_.get(0);
            const GpuLineSeries3D& ay = axis_cache_.get(1);
            const GpuLineSeries3D& az = axis_cache_.get(2);
            if (ax.valid()) renderer.draw_line_3d(ax.vbo, ax.point_count, mvp, col_x, 2.0f);
            if (ay.valid()) renderer.draw_line_3d(ay.vbo, ay.point_count, mvp, col_y, 2.0f);
            if (az.valid()) renderer.draw_line_3d(az.vbo, az.point_count, mvp, col_z, 2.0f);
        }
    }
    // Два прохода: сначала непрозрачное (пишет глубину), затем полупрозрачное через OIT —
    // оно закрывается непрозрачным и перекрывается между сериями по глубине. Без OIT
    // полупрозрачные серии ложились бы друг на друга в порядке рисования: серия, нарисованная
    // последней (IC 1), оказывалась поверх, даже находясь внутри другой.
    // pass 0 — непрозрачные элементы, pass 1 — полупрозрачные.
    // dry — только посчитать, есть ли что рисовать в проходе (без OIT-накопителей, если нечего).
    auto draw_series = [&](int pass, bool dry) -> int {
        int drawn = 0;
        for (int k = (int)series_cache_.size() - 1; k >= 0; --k) {
            if (!eff_visible(k)) continue;
            const GpuLineSeries3D& g = series_cache_.get(k);
            if (!g.valid()) continue;
            ImVec4 c = (k < (int)series_in.size()) ? series_color(series_in[k]) : ImVec4(1, 1, 1, 1);
            float color[4] = { c.x, c.y, c.z, c.w };
            // Элемент с альфой a — в этом проходе? Учитывает его и говорит, рисовать ли.
            auto take = [&](float a) {
                if ((a < 1.0f) != (pass == 1)) return false;
                ++drawn;
                return !dry;
            };
            // Тонкий путь (без Custom line style) α не смешивает — для него линия непрозрачна.
            const float line_a = custom_line_style ? color[3] : 1.0f;
            if (g.vbo_val && k < (int)series_in.size()) {
                const PlotSeriesInput3D& s = series_in[k];
                if (take(c.w))
                    renderer.draw_line_3d_cmap(g.vbo, g.vbo_val, g.point_count, mvp, s.colormap,
                                               s.cmin, s.cmax, s.cmap_reverse, c.w, line_thickness_px);
            }
            else if (points_mode) {
                // Линия под точками; в одном проходе порядок не важен — решает глубина.
                if (points_with_lines && take(line_a))
                    renderer.draw_line_3d(g.vbo, g.point_count, mvp, color, line_thickness_px, custom_line_style);
                float pcolor[4] = { c.x, c.y, c.z, point_alpha >= 0.0f ? point_alpha : c.w };
                if (take(pcolor[3]))
                    renderer.draw_points_3d(g.vbo, g.point_count, mvp, pcolor, point_size_px, point_marker);
            }
            else if (take(line_a))
                renderer.draw_line_3d(g.vbo, g.point_count, mvp, color,
                                      line_thickness_px, custom_line_style);
        }
        return drawn;
    };
    draw_series(0, false);
    if (draw_series(1, true) > 0) {
        // Диапазон оконной глубины сцены — по 8 углам bbox (вес OIT считается вдоль него).
        float zn = 0.0f, zf = 1.0f;
        float b[6];
        if (series_cache_.bbox(b[0], b[1], b[2], b[3], b[4], b[5])) {
            float lo = 1.0f, hi = 0.0f;
            for (int m = 0; m < 8; ++m) {
                const float x = b[0 + (m & 1)], y = b[2 + ((m >> 1) & 1)], z = b[4 + ((m >> 2) & 1)];
                const float cz = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
                const float cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
                if (cw <= 1e-12f) { lo = 0.0f; continue; }   // угол за камерой — от ближней плоскости
                const float wz = std::clamp(0.5f * cz / cw + 0.5f, 0.0f, 1.0f);
                lo = (std::min)(lo, wz);
                hi = (std::max)(hi, wz);
            }
            if (hi > lo) { zn = lo; zf = hi; }
        }
        const bool oit = renderer.begin_oit(zn, zf);
        draw_series(1, false);
        if (oit) renderer.end_oit();
    }
    renderer.end_frame();
    renderer.set_content_key(key);
    }

    // 5. Вывод FBO
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddImage((ImTextureID)(intptr_t)renderer.texture_id(),
        img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        ImVec2(0, 1), ImVec2(1, 0));

    // Подписи осей (текстом поверх картинки, через ImDrawList)
    {
        if (axes_on) {
            float xmn, xmx, ymn, ymx, zmn, zmx;
            if (series_cache_.bbox(xmn, xmx, ymn, ymx, zmn, zmx)) {
                ImU32 col_x = IM_COL32(255, 120, 120, 180);
                ImU32 col_y = IM_COL32(120, 255, 120, 180);
                ImU32 col_z = IM_COL32(140, 170, 255, 180);

                // Отступ от конца оси (в экранных пикселях)
                const float text_offset = 6.0f;

                auto draw_axis_label = [&](float ex, float ey, float ez,
                    const char* name, ImU32 color)
                    {
                        float zclip = 0;
                        ImVec2 p = project_to_screen(mvp, ex, ey, ez,
                            img_pos, (float)plot_w, (float)plot_h, &zclip);
                        if (zclip > 1.0f) return; // точка за плоскостью отсечения
                        ImVec2 ts = plot_text_size(name);
                        plot_text(dl, ImVec2(p.x + text_offset, p.y - ts.y * 0.5f), color, name);
                    };

                draw_axis_label(xmx, ymn, zmn, x_name.c_str(), col_x);
                draw_axis_label(xmn, ymx, zmn, y_name.c_str(), col_y);
                draw_axis_label(xmn, ymn, zmx, z_name.c_str(), col_z);
            }
        }
    }

    // 6. Легенда. ПКМ по квадрату — меню цвета, по строке мимо — меню плота (как в 2D).
    LegendRightClick legend_rclick;
    if (legend_on) {
        // best в 3D не разрешается (данные — в проекции камеры): остаётся northeast.
        LegendFrame frame;
        frame.plot_min  = img_pos;
        frame.plot_max  = ImVec2(img_pos.x + plot_w, img_pos.y + plot_h);
        frame.block_min = block_origin;
        frame.block_max = ImVec2(block_origin.x + avail_size.x, block_origin.y + avail_size.y);
        draw_legend(dl, frame, wp.legend_place, legend_entries, visible, global_visible, owner_id,
                    LegendPass::Both, &legend_rclick);
    }

    // 7. Зона взаимодействия (одна кнопка на всё поле)
    ImGui::SetCursorScreenPos(img_pos);
    char id_buf[48];
    std::snprintf(id_buf, sizeof(id_buf), "##plot3d_%d", owner_id);
    ImGui::InvisibleButton(id_buf, ImVec2((float)plot_w, (float)plot_h),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool plot_h_ov = ImGui::IsItemHovered();
    bool plot_a = ImGui::IsItemActive();
    bool plot_dbl = plot_h_ov && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    // 8. Рамка
    dl->AddRect(img_pos, ImVec2(img_pos.x + plot_w, img_pos.y + plot_h),
        plot_col_border(), 0.0f, 0, 1.0f);

    // 9. Двойной клик - автофит
    if (plot_dbl) view_valid = false;

    // 10. Управление: orbit правой, pan левой, zoom колесом, popup-меню по короткому клику правой
    static bool rzoom_pending[64] = { false }; // отличаем drag правой от короткого клика.
    // (в production лучше вынести pending в поле класса, как в 2D)
    // Индексируем этот static по owner_id - иначе плоты мешали бы друг другу.
    if (owner_id < 0 || owner_id >= 64) owner_id = 0;

    static float rzoom_start_x[64] = { 0 };
    static float rzoom_start_y[64] = { 0 };

    ImGuiIO& io = ImGui::GetIO();
    const float drag_threshold = plot_rmb_drag_threshold();   // общий порог щелчка ПКМ (plot_axis.h)

    if (plot_h_ov || plot_a) {
        // Orbit правой
        if (plot_a && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
            camera.orbit(io.MouseDelta.x, io.MouseDelta.y);
        }
        // Pan левой
        if (plot_a && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
            camera.pan(io.MouseDelta.x, io.MouseDelta.y, plot_w, plot_h);
        }
        // Zoom колесом
        if (plot_h_ov && io.MouseWheel != 0.0f) {
            camera.zoom(std::pow(0.85f, io.MouseWheel));
        }
    }

    // Короткий клик правой (без drag) - popup-меню
    if (!rzoom_pending[owner_id] && plot_h_ov && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        rzoom_pending[owner_id] = true;
        rzoom_start_x[owner_id] = io.MousePos.x;
        rzoom_start_y[owner_id] = io.MousePos.y;
    }
    if (rzoom_pending[owner_id] && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        float dx = io.MousePos.x - rzoom_start_x[owner_id];
        float dy = io.MousePos.y - rzoom_start_y[owner_id];
        bool was_drag = std::sqrt(dx * dx + dy * dy) > drag_threshold;
        rzoom_pending[owner_id] = false;
        if (!was_drag) {
            char pop_id[48];
            std::snprintf(pop_id, sizeof(pop_id), "##plot3d_menu_%d", owner_id);
            ImGui::OpenPopup(pop_id);
        }
    }

    // 11. Контекстное меню (и ПКМ по строке легенды мимо квадрата — легенда забирает hover себе).
    char pop_id[48];
    std::snprintf(pop_id, sizeof(pop_id), "##plot3d_menu_%d", owner_id);
    if (legend_rclick.row_other) ImGui::OpenPopup(pop_id);
    if (ImGui::BeginPopup(pop_id)) {
        if (ImGui::MenuItem("Auto fit")) view_valid = false;
        ImGui::MenuItem("Auto fit on new data", nullptr, &auto_fit);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("On: every recompute fits the view to the data.\n"
                              "Off: the view stays where you left it; Auto fit above and\n"
                              "a double click still fit it.");
        if (ImGui::BeginMenu("Axis scale")) {
            bool changed = ImGui::MenuItem("Fit axes to window", nullptr, &wp.fit3d);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("On: the z range fills the window height, x and y fill its width.\n"
                                  "Off: all axes in data units (true proportions).");
            const char* names[3] = { x_name.c_str(), y_name.c_str(), z_name.c_str() };
            for (int i = 0; i < 3; ++i) {
                ImGui::PushID(i);
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.0f);
                changed |= ImGui::DragFloat(names[i][0] ? names[i] : "axis", &wp.scale3d[i], 0.01f, 0.01f,
                                            100.0f, "x %.2f",
                                            ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
                ImGui::PopID();
            }
            if (ImGui::MenuItem("Reset scale")) {
                wp.scale3d[0] = wp.scale3d[1] = wp.scale3d[2] = 1.0f;
                wp.fit3d = false;
                view_valid = false;
                changed = true;
            }
            if (changed) ImGui::MarkIniSettingsDirty();
            ImGui::EndMenu();
        }
        ImGui::Separator();
        // Легенда и оси — галки окна (сохраняются в imgui.ini); show_legend / show_axes вида —
        // разрешение вызывающего.
        plot_toolbar_menu();   // элементы тулбара окна (plot_axis.h)
        plot_show_menu(wp, (show_legend ? PlotShowLegend : 0u) | (show_axes ? PlotShowAxes3D : 0u) |
                           (external_colorbar ? PlotShowColorbar : 0u));
        if (show_legend) plot_legend_menu(wp, false);
        ImGui::Separator();
        if (ImGui::MenuItem("Copy image to clipboard")) {
            request_plot_screenshot(block_origin,
                ImVec2(block_origin.x + avail_size.x + screenshot_extra_right,
                       block_origin.y + avail_size.y));
        }
        // Caller-injected пункты (например, "Export data..."). Зеркалит тот же
        // хук у Plot2DView/HeatmapView — до его появления Phase 3D был
        // единственной диаграммой без экспорта из контекстного меню.
        if (popup_extras) {
            ImGui::Separator();
            popup_extras();
        }
        ImGui::EndPopup();
    }

    // 12. Цвет серии — ПКМ по квадрату легенды (общее с 2D меню, legend_color_popup).
    char col_pop[48];
    std::snprintf(col_pop, sizeof(col_pop), "##plot3d_color_%d", owner_id);
    if (legend_rclick.swatch_index >= 0 && legend_rclick.swatch_index < (int)series_in.size()) {
        legend_color_target_ = series_in[legend_rclick.swatch_index].label;
        ImGui::OpenPopup(col_pop);
    }
    ImVec4 cur(1, 1, 1, 1);
    for (const auto& s : series_in)
        if (s.label == legend_color_target_) { cur = series_color(s); break; }
    // Цвета — общие на окно и сохраняются в imgui.ini (PlotWindowPrefs::series_colors).
    const std::map<std::string, ImVec4> before = wp.series_colors;
    if (legend_color_popup(col_pop, legend_color_target_, wp.series_colors, cur))
        plot_series_color_spread(wp, legend_color_target_);
    plot_series_colors_changed(wp, before);
}
