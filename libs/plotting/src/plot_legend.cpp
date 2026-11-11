#include "ucuda/plotting/plot_legend.h"
#include "ucuda/plotting/plot_axis.h"
#include "imgui_internal.h"   // MarkIniSettingsDirty (недавние цвета в imgui.ini)
#include <cstdio>
#include <cmath>
#include <algorithm>

void draw_legend(ImDrawList* dl,
    ImVec2 img_pos, float plot_w,
    const std::vector<LegendEntry>& entries,
    std::vector<bool>& visible,
    const std::vector<bool>& global_visible,
    int owner_id,
    LegendPass pass,
    LegendRightClick* out_clicks)
{
    const bool do_visual   = (pass != LegendPass::Interact);
    const bool do_interact = (pass != LegendPass::Draw);

    auto is_global_vis = [&](int k) -> bool {
        return (k < (int)global_visible.size()) ? global_visible[k] : true;
        };

    float line_h = plot_text_line_height();
    float row_h = line_h + 2;
    float marker_w = 14;
    float gap = 4;

    // Ширина панели — по самой длинной подписи среди показываемых записей
    float max_text_w = 0;
    int visible_count = 0;
    for (int k = 0; k < (int)entries.size(); ++k) {
        if (!is_global_vis(k)) continue;
        max_text_w = std::max(max_text_w, plot_text_size(entries[k].label.c_str()).x);
        ++visible_count;
    }
    if (visible_count == 0) return; // рисовать нечего

    float pad = 6;
    float entry_w = marker_w + gap + max_text_w;
    float legend_w = entry_w + pad * 2;
    int nrows = std::min(visible_count, 20);
    float legend_h = nrows * row_h + pad * 2;

    ImVec2 leg_min = ImVec2(img_pos.x + plot_w - legend_w - 6, img_pos.y + 6);
    ImVec2 leg_max = ImVec2(leg_min.x + legend_w, leg_min.y + legend_h);

    if (do_visual) {
        // Bg+border легенды зависит от темы: на dark — почти-чёрный полупрозрачный
        // блок, на light — почти-белый. Без этого в light-теме легенда оставалась
        // чёрным пятном поверх белого плота.
        ImU32 bg_col     = plot_light_theme() ? IM_COL32(248, 248, 248, 200)
                                              : IM_COL32( 20,  20,  25, 180);
        ImU32 border_col = plot_col_border();
        dl->AddRectFilled(leg_min, leg_max, bg_col, 4.0f);
        dl->AddRect(leg_min, leg_max, border_col, 4.0f);
    }

    int row = 0;
    for (int k = 0; k < (int)entries.size(); ++k) {
        if (!is_global_vis(k)) continue;   // скрытые глобально не показываем
        if (row >= 20) break;
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
        ImVec2 mmin = ImVec2(leg_min.x + pad, row_y + 2);
        ImVec2 mmax = ImVec2(mmin.x + marker_w, row_y + row_h - 2);
        if (do_visual) {
            dl->AddRectFilled(mmin, mmax, cu);
            // Рамка на наведении именно КВАДРАТА — подсказка, что ПКМ по нему
            // (и только по нему, не по подписи) открывает выбор цвета.
            if (out_clicks && ImGui::IsMouseHoveringRect(mmin, mmax, false))
                dl->AddRect(mmin, mmax, plot_col_text(), 0.0f, 0, 1.5f);
            plot_text(dl, ImVec2(mmax.x + gap, row_y), text_cu, entries[k].label.c_str());
        }
        ImVec2 emin = ImVec2(leg_min.x + pad, row_y);
        ImVec2 emax = ImVec2(leg_max.x - pad, row_y + row_h);

        if (do_interact) {
            ImGui::SetCursorScreenPos(emin);
            char btn_id[48];
            std::snprintf(btn_id, sizeof(btn_id), "##legend_%d_%d", owner_id, k);
            // ПКМ включаем явно: по умолчанию InvisibleButton слушает только ЛКМ.
            // Меню плота от этого не страдает — его InvisibleButton подан ПОЗЖЕ и
            // под легендой hover не получает (первый item в кадре выигрывает).
            ImGui::InvisibleButton(btn_id, ImVec2(emax.x - emin.x, emax.y - emin.y),
                ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            if (do_visual && ImGui::IsItemHovered())
                dl->AddRectFilled(emin, emax, IM_COL32(255, 255, 255, 18));
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
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
        ++row;
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
