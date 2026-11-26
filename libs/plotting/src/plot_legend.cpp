#include "ucuda/plotting/plot_legend.h"
#include "ucuda/plotting/plot_axis.h"
#include "imgui_internal.h"   // MarkIniSettingsDirty (недавние цвета в imgui.ini)
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace {
constexpr int   kLegendMaxRows   = 20;
constexpr float kLegendMarkerW   = 14.0f;
constexpr float kLegendGap       = 4.0f;    // квадрат — подпись
constexpr float kLegendPad       = 6.0f;    // поле внутри рамки
constexpr float kLegendInset     = 6.0f;    // отступ внутренних положений от края поля графика
constexpr float kLegendOutsideGap = 6.0f;   // outside: зазор легенда — оси
constexpr float kLegendEdge      = 2.0f;    // outside: отступ от края блока

float legend_row_h() { return plot_text_line_height() + 2.0f; }

float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

ImU32 with_alpha(ImU32 c, float a) {
    const unsigned base = (c >> IM_COL32_A_SHIFT) & 0xFFu;
    const unsigned aa = (unsigned)std::lround(base * clamp01(a));
    return (c & ~IM_COL32_A_MASK) | (aa << IM_COL32_A_SHIFT);
}

// Новое место при протяжке: ручное положение, доли свободного хода внутри поля графика.
void legend_move_to(LegendPlacement& place, ImVec2 origin, ImVec2 size, const LegendFrame& f) {
    const float free_x = (f.plot_max.x - f.plot_min.x) - size.x;
    const float free_y = (f.plot_max.y - f.plot_min.y) - size.y;
    place.location = kLegendNone;
    place.pos[0] = free_x > 0.0f ? clamp01((origin.x - f.plot_min.x) / free_x) : 0.0f;
    place.pos[1] = free_y > 0.0f ? clamp01((origin.y - f.plot_min.y) / free_y) : 0.0f;
}

constexpr float kLegendColGap = 12.0f;   // между столбцами

// Раскладка записей: показываемые записи (не скрытые глобально, не больше kLegendMaxRows строк
// на столбец), по столбцам сверху вниз; ширина столбца — по его самой длинной подписи.
struct LegendLayout {
    std::vector<int>   idx;            // индексы записей в порядке показа
    int                rows = 0, cols = 0;
    std::vector<float> col_x, col_w;   // смещение столбца от левого поля и его ширина
    ImVec2             size{ 0.0f, 0.0f };
};

LegendLayout legend_layout(const std::vector<LegendEntry>& entries, const std::vector<bool>& global_visible,
                           int columns) {
    LegendLayout L;
    for (int k = 0; k < (int)entries.size(); ++k)
        if (k >= (int)global_visible.size() || global_visible[k]) L.idx.push_back(k);
    if (L.idx.empty()) return L;
    L.cols = std::clamp(columns, 1, kLegendMaxColumns);
    L.cols = std::min(L.cols, (int)L.idx.size());
    if ((int)L.idx.size() > kLegendMaxRows * L.cols) L.idx.resize((size_t)kLegendMaxRows * L.cols);
    L.rows = ((int)L.idx.size() + L.cols - 1) / L.cols;
    L.cols = ((int)L.idx.size() + L.rows - 1) / L.rows;   // 5 записей в 4 столбца -> 2 строки, 3 столбца
    float x = 0.0f;
    for (int c = 0; c < L.cols; ++c) {
        float text_w = 0.0f;
        for (int r = 0; r < L.rows; ++r) {
            const int i = c * L.rows + r;
            if (i >= (int)L.idx.size()) break;
            text_w = std::max(text_w, plot_text_size(entries[L.idx[i]].label.c_str()).x);
        }
        if (c > 0) x += kLegendColGap;
        L.col_x.push_back(x);
        L.col_w.push_back(kLegendMarkerW + kLegendGap + text_w);
        x += L.col_w.back();
    }
    L.size = ImVec2(x + kLegendPad * 2.0f, L.rows * legend_row_h() + kLegendPad * 2.0f);
    return L;
}
}  // namespace

const char* legend_location_name(int loc) {
    static const char* const names[kLegendLocationCount] = {
        "northeast", "northwest", "southeast", "southwest",
        "north", "south", "east", "west",
        "northoutside", "southoutside", "eastoutside", "westoutside",
        "northeastoutside", "northwestoutside", "southeastoutside", "southwestoutside",
        "best", "none",
    };
    return (loc >= 0 && loc < kLegendLocationCount) ? names[loc] : "?";
}

bool legend_location_outside(int loc) {
    return loc >= kLegendNorthOutside && loc <= kLegendSouthWestOutside;
}

bool legend_placement_default(const LegendPlacement& p) {
    const LegendPlacement d;
    return p.location == d.location && p.box_line == d.box_line && p.box_fill == d.box_fill &&
           p.line_alpha == d.line_alpha && p.fill_alpha == d.fill_alpha && p.columns == d.columns;
}

ImVec2 legend_size(const std::vector<LegendEntry>& entries, const std::vector<bool>& global_visible,
                   int columns) {
    return legend_layout(entries, global_visible, columns).size;
}

void legend_reserve_margins(int loc, ImVec2 size, float& left, float& top, float& right, float& bottom) {
    if (size.x <= 0.0f || size.y <= 0.0f) return;
    switch (loc) {
    case kLegendNorthOutside: top    += size.y + kLegendOutsideGap; break;
    case kLegendSouthOutside: bottom += size.y + kLegendOutsideGap; break;
    case kLegendEastOutside: case kLegendNorthEastOutside: case kLegendSouthEastOutside:
        right += size.x + kLegendOutsideGap; break;
    case kLegendWestOutside: case kLegendNorthWestOutside: case kLegendSouthWestOutside:
        left  += size.x + kLegendOutsideGap; break;
    default: break;
    }
}

ImVec2 legend_origin(int loc, const LegendPlacement& place, ImVec2 s, const LegendFrame& f) {
    if (loc == kLegendBest) loc = (f.best >= 0 && f.best < kLegendBest) ? f.best : kLegendNorthEast;
    const ImVec2 P0 = f.plot_min, P1 = f.plot_max, B0 = f.block_min, B1 = f.block_max;
    const float  cx = 0.5f * (P0.x + P1.x) - 0.5f * s.x;
    const float  cy = 0.5f * (P0.y + P1.y) - 0.5f * s.y;
    const float  in_l = P0.x + kLegendInset, in_r = P1.x - s.x - kLegendInset;
    const float  in_t = P0.y + kLegendInset, in_b = P1.y - s.y - kLegendInset;
    const float  out_l = B0.x + kLegendEdge, out_r = B1.x - s.x - kLegendEdge;
    switch (loc) {
    case kLegendNorthWest:        return ImVec2(in_l, in_t);
    case kLegendSouthEast:        return ImVec2(in_r, in_b);
    case kLegendSouthWest:        return ImVec2(in_l, in_b);
    case kLegendNorth:            return ImVec2(cx, in_t);
    case kLegendSouth:            return ImVec2(cx, in_b);
    case kLegendEast:             return ImVec2(in_r, cy);
    case kLegendWest:             return ImVec2(in_l, cy);
    case kLegendNorthOutside:     return ImVec2(cx, B0.y + kLegendEdge);
    case kLegendSouthOutside:     return ImVec2(cx, B1.y - s.y - kLegendEdge);
    case kLegendEastOutside:      return ImVec2(out_r, cy);
    case kLegendWestOutside:      return ImVec2(out_l, cy);
    case kLegendNorthEastOutside: return ImVec2(out_r, P0.y);
    case kLegendNorthWestOutside: return ImVec2(out_l, P0.y);
    case kLegendSouthEastOutside: return ImVec2(out_r, P1.y - s.y);
    case kLegendSouthWestOutside: return ImVec2(out_l, P1.y - s.y);
    case kLegendNone: {
        const float free_x = std::max(0.0f, (P1.x - P0.x) - s.x);
        const float free_y = std::max(0.0f, (P1.y - P0.y) - s.y);
        return ImVec2(std::floor(P0.x + clamp01(place.pos[0]) * free_x),
                      std::floor(P0.y + clamp01(place.pos[1]) * free_y));
    }
    case kLegendNorthEast:
    default:                      return ImVec2(in_r, in_t);
    }
}

void draw_legend(ImDrawList* dl,
    const LegendFrame& frame, LegendPlacement& place,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id,
    LegendPass pass,
    LegendRightClick* out_clicks)
{
    // Оба прохода сразу — сначала клики (протяжка меняет place), потом визуал уже на новом
    // месте: иначе легенда отставала бы от курсора на кадр.
    if (pass == LegendPass::Both) {
        draw_legend(dl, frame, place, entries, visible, global_visible, owner_id, LegendPass::Interact, out_clicks);
        draw_legend(dl, frame, place, entries, visible, global_visible, owner_id, LegendPass::Draw, out_clicks);
        return;
    }
    const bool do_visual   = (pass != LegendPass::Interact);
    const bool do_interact = (pass != LegendPass::Draw);

    const LegendLayout layout = legend_layout(entries, global_visible, place.columns);
    const ImVec2 size = layout.size;
    if (size.x <= 0.0f) return; // рисовать нечего
    const float row_h    = legend_row_h();
    const float marker_w = kLegendMarkerW;
    const float gap      = kLegendGap;
    const float pad      = kLegendPad;

    ImVec2 leg_min = legend_origin(place.location, place, size, frame);
    ImVec2 leg_max = ImVec2(leg_min.x + size.x, leg_min.y + size.y);

    // Протяжка ЛКМ: начало — по нажатию на строку или на подложку, сдвиг — от угла легенды
    // в момент нажатия. Состояние — в хранилище окна ImGui (между кадрами, на легенду).
    char bg_id[48];
    std::snprintf(bg_id, sizeof(bg_id), "##legend_bg_%d", owner_id);
    ImGuiStorage* st = ImGui::GetStateStorage();
    const ImGuiID key = ImGui::GetID(bg_id);
    const ImGuiID k_drag = key ^ 0x9E3779B1u, k_x = key ^ 0x85EBCA6Bu, k_y = key ^ 0xC2B2AE35u;
    auto drag_item = [&]() {
        if (ImGui::IsItemActivated() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            st->SetBool(k_drag, false);
            st->SetFloat(k_x, leg_min.x);
            st->SetFloat(k_y, leg_min.y);
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // До порога перетаскивания ImGui дельта нулевая: щелчок не сдвигает легенду.
            const ImVec2 d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            if (d.x != 0.0f || d.y != 0.0f || st->GetBool(k_drag)) {
                st->SetBool(k_drag, true);
                legend_move_to(place, ImVec2(st->GetFloat(k_x) + d.x, st->GetFloat(k_y) + d.y), size, frame);
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            }
        }
        if (ImGui::IsItemDeactivated() && st->GetBool(k_drag))
            ImGui::MarkIniSettingsDirty();
    };

    if (do_visual) {
        // Подложка — цвет фона плота (совпадает и с чёрным фоном High Contrast, и с
        // пользовательским фоном): на dark — почти-чёрная, на light — почти-белая;
        // непрозрачность — Box из меню легенды.
        float pr = 0, pg = 0, pb = 0, pa = 1;
        plot_bg_color(pr, pg, pb, pa);
        const ImVec4 bg = plot_bg_is_custom() ? ImVec4(pr, pg, pb, 1.0f)
                        : plot_light_theme()  ? ImVec4(248 / 255.0f, 248 / 255.0f, 248 / 255.0f, 1.0f)
                                              : ImVec4( 20 / 255.0f,  20 / 255.0f,  25 / 255.0f, 1.0f);
        const float fill_a = clamp01(place.fill_alpha);
        if (place.box_fill && fill_a > 0.0f)
            dl->AddRectFilled(leg_min, leg_max,
                              ImGui::ColorConvertFloat4ToU32(ImVec4(bg.x, bg.y, bg.z, fill_a)), 4.0f);
        if (place.box_line && place.line_alpha > 0.0f)
            dl->AddRect(leg_min, leg_max, with_alpha(plot_col_border(), place.line_alpha), 4.0f);
    }

    for (int i = 0; i < (int)layout.idx.size(); ++i) {
        const int k   = layout.idx[i];          // скрытые глобально в раскладку не попали
        const int col = i / layout.rows, row = i % layout.rows;
        const float col_x = leg_min.x + pad + layout.col_x[col];
        bool v = (k < (int)visible.size()) ? visible[k] : true;
        ImVec4 c = entries[k].color;
        if (!v) c.w *= 0.35f;
        ImU32 cu = ImGui::ColorConvertFloat4ToU32(c);
        ImU32 text_cu;
        if (v) text_cu = plot_col_text();
        else {
            // Dimmed text для невидимых entries: половина alpha от обычного.
            ImU32 base = plot_col_text();
            unsigned aa = (base >> IM_COL32_A_SHIFT) & 0xFFu;
            aa = aa * 110u / 255u;
            text_cu = (base & ~IM_COL32_A_MASK) | (aa << IM_COL32_A_SHIFT);
        }
        float row_y = leg_min.y + pad + row * row_h;
        ImVec2 mmin = ImVec2(col_x, row_y + 2);
        ImVec2 mmax = ImVec2(mmin.x + marker_w, row_y + row_h - 2);
        if (do_visual) {
            dl->AddRectFilled(mmin, mmax, cu);
            // Рамка на наведении именно КВАДРАТА — подсказка, что ПКМ по нему
            // (и только по нему, не по подписи) открывает выбор цвета.
            if (out_clicks && ImGui::IsMouseHoveringRect(mmin, mmax, false))
                dl->AddRect(mmin, mmax, plot_col_text(), 0.0f, 0, 1.5f);
            plot_text(dl, ImVec2(mmax.x + gap, row_y), text_cu, entries[k].label.c_str());
        }
        ImVec2 emin = ImVec2(col_x, row_y);
        ImVec2 emax = ImVec2(col_x + layout.col_w[col], row_y + row_h);

        if (do_interact) {
            ImGui::SetCursorScreenPos(emin);
            char btn_id[48];
            std::snprintf(btn_id, sizeof(btn_id), "##legend_%d_%d", owner_id, k);
            // ПКМ включаем явно: по умолчанию InvisibleButton слушает только ЛКМ.
            // Меню плота от этого не страдает — его InvisibleButton подан ПОЗЖЕ и
            // под легендой hover не получает (первый item в кадре выигрывает).
            const bool released = ImGui::InvisibleButton(btn_id, ImVec2(emax.x - emin.x, emax.y - emin.y),
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            drag_item();
            // Гасим серию по ОТПУСКАНИЮ ЛКМ и только если легенду не тянули: нажатие на строку
            // может оказаться началом протяжки.
            if (released && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !st->GetBool(k_drag)) {
                if (k < (int)visible.size()) visible[k] = !visible[k];
            }
            // Меню цвета — только по цветному квадрату. Промах мимо него (по
            // подписи или пустому месту строки) не должен пропадать втуне:
            // сообщаем об этом отдельным флагом, и плот покажет своё обычное
            // меню — как если бы легенды в этом месте не было.
            if (out_clicks && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                if (ImGui::IsMouseHoveringRect(mmin, mmax, false))
                    out_clicks->swatch_index = k;
                else
                    out_clicks->row_other = true;
            }
        } else if (do_visual) {
            // Draw-only pass: hover highlight без InvisibleButton (тот был
            // создан в Interact-проходе, повторно нельзя — коллизия ID).
            if (ImGui::IsMouseHoveringRect(emin, emax, false))
                dl->AddRectFilled(emin, emax, IM_COL32(255, 255, 255, 18));
        }
    }

    // Подложка целиком (поля вокруг строк) — тоже ручка протяжки. Подана ПОСЛЕ строк: те
    // поданы раньше и hover над собой оставляют себе. Плот подаёт свою кнопку ещё позже,
    // так что ЛКМ по легенде не тянет вид, а ПКМ по полям — его обычное меню.
    if (do_interact) {
        ImGui::SetCursorScreenPos(leg_min);
        ImGui::InvisibleButton(bg_id, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        drag_item();
        if (ImGui::IsItemHovered() && !ImGui::IsMouseDown(ImGuiMouseButton_Left))
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        if (out_clicks && ImGui::IsItemClicked(ImGuiMouseButton_Right))
            out_clicks->row_other = true;
    }
}

std::vector<ImVec4>& legend_recent_colors() {
    static std::vector<ImVec4> v;
    return v;
}

namespace {
constexpr int kRecentMax = 8;

void remember_colour(const ImVec4& c) {
    auto& r = legend_recent_colors();
    const auto same = [&](const ImVec4& o) {
        return std::fabs(o.x - c.x) < 1e-3f && std::fabs(o.y - c.y) < 1e-3f && std::fabs(o.z - c.z) < 1e-3f;
    };
    r.erase(std::remove_if(r.begin(), r.end(), same), r.end());
    r.insert(r.begin(), ImVec4(c.x, c.y, c.z, 1.0f));
    if ((int)r.size() > kRecentMax) r.resize(kRecentMax);
    ImGui::MarkIniSettingsDirty();
}

ImVec4 hex_colour(unsigned rgb) {
    return ImVec4(((rgb >> 16) & 0xFF) / 255.0f, ((rgb >> 8) & 0xFF) / 255.0f, (rgb & 0xFF) / 255.0f, 1.0f);
}

// MATLAB orderedcolors: "gem12" (R2025a, светлая тема) и "glow12" (тёмная тема); первые
// 7 — палитры "gem" / "glow". Сняты с образцов документации MathWorks (colordots-*.png).
constexpr unsigned kGem[]  = { 0x1171BE, 0xDD5400, 0xEDB120, 0x8516D1, 0x3BAA32, 0x2FBEEF, 0xD1048B,
                               0xFFD60A, 0x6582FD, 0xFF453A, 0x00A3A3, 0xCB845D };
constexpr unsigned kGlow[] = { 0x268CDD, 0xF57729, 0xFFE864, 0xC05CFB, 0x49DB40, 0x6CF4FF, 0xF267C5,
                               0xFEC04C, 0x7DA9FF, 0xFF7A74, 0x1FCFBE, 0xDC996C };
// ImPlot "Deep" (Seaborn deep) — палитра серий U-CUDA до перехода на gem12.
constexpr unsigned kDeep[] = { 0x4C72B0, 0xDD8452, 0x55A868, 0xC44E52, 0x8172B3,
                               0x937860, 0xDA8BC3, 0x8C8C8C, 0xCCB974, 0x64B5CD };
// Основные цвета MATLAB: r g b c m y k w.
constexpr unsigned kBasic[] = { 0xFF0000, 0x00FF00, 0x0000FF, 0x00FFFF, 0xFF00FF, 0xFFFF00, 0x000000, 0xFFFFFF };

// Ряд квадратов с подписью слева; true и out — по клику на квадрат.
bool swatch_row(const char* name, const ImVec4* colours, int n, ImVec4& out) {
    bool picked = false;
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("%s", name);
    ImGui::SameLine(52.0f);
    ImGui::PushID(name);
    const float sz = ImGui::GetFrameHeight();
    for (int i = 0; i < n; ++i) {
        if (i > 0) ImGui::SameLine(0.0f, 4.0f);
        ImGui::PushID(i);
        if (ImGui::ColorButton("##sw", colours[i], ImGuiColorEditFlags_NoAlpha, ImVec2(sz, sz))) {
            out = colours[i];
            picked = true;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    return picked;
}

template <size_t N>
bool swatch_row(const char* name, const unsigned (&hex)[N], ImVec4& out) {
    ImVec4 c[N];
    for (size_t i = 0; i < N; ++i) c[i] = hex_colour(hex[i]);
    return swatch_row(name, c, (int)N, out);
}

struct SeriesPalette { const char* name; const unsigned* hex; int n; };
constexpr SeriesPalette kSeriesPalettes[kSeriesPaletteCount] = {
    { "MATLAB gem12",  kGem,  (int)(sizeof(kGem)  / sizeof(kGem[0]))  },
    { "MATLAB glow12", kGlow, (int)(sizeof(kGlow) / sizeof(kGlow[0])) },
    { "ImPlot Deep",   kDeep, (int)(sizeof(kDeep) / sizeof(kDeep[0])) },
};
int g_series_palette = kSeriesPaletteGem12;
}  // namespace

const char* series_palette_name(int palette) {
    return (palette >= 0 && palette < kSeriesPaletteCount) ? kSeriesPalettes[palette].name : "?";
}

void set_series_palette(int palette) {
    g_series_palette = (palette >= 0 && palette < kSeriesPaletteCount) ? palette : kSeriesPaletteGem12;
}

int series_palette() { return g_series_palette; }

int series_palette_size(int palette) {
    return (palette >= 0 && palette < kSeriesPaletteCount) ? kSeriesPalettes[palette].n : 0;
}

ImVec4 series_palette_color(int palette, int index) {
    if (palette < 0 || palette >= kSeriesPaletteCount) palette = kSeriesPaletteGem12;
    const SeriesPalette& p = kSeriesPalettes[palette];
    return hex_colour(p.hex[((index % p.n) + p.n) % p.n]);
}

ImVec4 default_series_color(int index) {
    return series_palette_color(g_series_palette, index);
}

bool legend_color_popup(const char* pop_id, const std::string& target,
                        std::map<std::string, ImVec4>& overrides, const ImVec4& current) {
    // Цвет из пипетки уходит в «недавние», когда правка закончена (мышь отпущена).
    static bool   pending = false;
    static ImVec4 pending_colour;
    if (!ImGui::BeginPopup(pop_id)) {
        if (pending) { remember_colour(pending_colour); pending = false; }
        return false;
    }
    bool spread = false;
    ImGui::TextUnformatted(target.empty() ? "(series)" : target.c_str());
    ImGui::Separator();
    float f[3] = { current.x, current.y, current.z };
    ImGui::SetNextItemWidth(260.0f);
    // Override — только после реальной правки: открыть меню и закрыть — цвет остаётся палитровым.
    if (ImGui::ColorEdit3("colour##legend_color", f) && !target.empty()) {
        overrides[target] = ImVec4(f[0], f[1], f[2], 1.0f);
        pending = true;
        pending_colour = overrides[target];
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Click the swatch for the picker; drag the fields or type RGB / HSV / Hex.\n"
                          "Transparency is set by the Alpha sliders of the plot.");
    if (pending && !ImGui::IsAnyItemActive() && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        remember_colour(pending_colour);
        pending = false;
    }

    ImGui::Separator();
    ImVec4 picked;
    bool   chosen = false;
    chosen |= swatch_row("gem",   kGem,   picked);
    chosen |= swatch_row("glow",  kGlow,  picked);
    chosen |= swatch_row("deep",  kDeep,  picked);
    chosen |= swatch_row("basic", kBasic, picked);
    const auto& recent = legend_recent_colors();
    if (!recent.empty()) {
        const std::vector<ImVec4> copy = recent;   // remember_colour ниже меняет список
        chosen |= swatch_row("recent", copy.data(), (int)copy.size(), picked);
    }
    if (chosen && !target.empty()) {
        overrides[target] = picked;
        remember_colour(picked);
        // Тот же цвет, что уже стоит, — правки нет, и plot_series_colors_changed её не увидит;
        // Ctrl+Shift+клик по квадрату всё равно разносит цвет по вкладке.
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && io.KeyShift) spread = true;
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reset this colour", nullptr, false, overrides.count(target) != 0)) {
        overrides.erase(target);
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::MenuItem("Reset all colours", nullptr, false, !overrides.empty()))
        overrides.clear();
    ImGui::Separator();
    if (ImGui::MenuItem("Apply to all plots of this tab", nullptr, false, !target.empty()))
        spread = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Same colour for \"%s\" in every plot of this tab.\n"
                          "Ctrl+Shift while editing or resetting does the same.",
                          target.c_str());
    ImGui::EndPopup();
    return spread;
}

bool legend_placement_menu(LegendPlacement& place, bool has_best, bool& spread_all) {
    spread_all = false;
    if (!ImGui::BeginMenu("Legend")) return false;
    bool changed = false;
    auto loc_item = [&](int loc) {
        if (ImGui::MenuItem(legend_location_name(loc), nullptr, place.location == loc) && place.location != loc) {
            place.location = loc;
            changed = true;
        }
    };
    if (ImGui::BeginMenu("Location")) {
        // Выбор не закрывает меню: положения обычно перебирают, глядя на график.
        ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
        ImGui::TextDisabled("Inside");
        for (int loc = kLegendNorthEast; loc <= kLegendWest; ++loc) loc_item(loc);
        ImGui::Separator();
        ImGui::TextDisabled("Outside (the plot shrinks)");
        for (int loc = kLegendNorthOutside; loc <= kLegendSouthWestOutside; ++loc) loc_item(loc);
        ImGui::Separator();
        if (!has_best) ImGui::BeginDisabled();
        loc_item(kLegendBest);
        if (!has_best) ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(has_best ? "Inside position that covers the fewest data points."
                                       : "Not available in this plot.");
        loc_item(kLegendNone);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Manual position: drag the legend with the left mouse button.\n"
                              "Dragging switches any location to this one.");
        ImGui::PopItemFlag();
        ImGui::EndMenu();
    }
    ImGui::Separator();
    // Галка и ползунки не закрывают меню: Box обычно подбирают несколькими правками подряд.
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    changed |= ImGui::MenuItem("Box line", nullptr, &place.box_line);
    ImGui::PopItemFlag();
    const float w = ImGui::GetFontSize() * 9.0f;
    if (!place.box_line) ImGui::BeginDisabled();
    float line_pct = place.line_alpha * 100.0f;
    ImGui::SetNextItemWidth(w);
    if (ImGui::SliderFloat("Line opacity", &line_pct, 0.0f, 100.0f, "%.0f %%", ImGuiSliderFlags_AlwaysClamp)) {
        place.line_alpha = line_pct / 100.0f;
        changed = true;
    }
    if (!place.box_line) ImGui::EndDisabled();
    ImGui::PushItemFlag(ImGuiItemFlags_AutoClosePopups, false);
    changed |= ImGui::MenuItem("Box fill", nullptr, &place.box_fill);
    ImGui::PopItemFlag();
    if (!place.box_fill) ImGui::BeginDisabled();
    float fill_pct = place.fill_alpha * 100.0f;
    ImGui::SetNextItemWidth(w);
    if (ImGui::SliderFloat("Fill opacity", &fill_pct, 0.0f, 100.0f, "%.0f %%", ImGuiSliderFlags_AlwaysClamp)) {
        place.fill_alpha = fill_pct / 100.0f;
        changed = true;
    }
    if (!place.box_fill) ImGui::EndDisabled();
    ImGui::Separator();
    ImGui::SetNextItemWidth(w);
    if (ImGui::SliderInt("Columns", &place.columns, 1, kLegendMaxColumns, "%d", ImGuiSliderFlags_AlwaysClamp))
        changed = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Number of legend columns (MATLAB NumColumns).\n"
                          "Entries fill each column top to bottom, columns go left to right.");
    ImGui::Separator();
    if (ImGui::MenuItem("Reset legend", nullptr, false, !legend_placement_default(place))) {
        place = LegendPlacement{};
        changed = true;
    }
    if (ImGui::MenuItem("Apply to all plots of this tab")) spread_all = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Same location and box in every plot of this tab.");
    ImGui::EndMenu();
    return changed || spread_all;
}
