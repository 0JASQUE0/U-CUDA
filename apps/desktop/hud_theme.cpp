#include "hud_theme.h"

#include "imgui.h"
#include "imgui_internal.h"   // GImGui->Windows, ImGuiWindow
#include <glad/glad.h>

#include <algorithm>
#include <cmath>

namespace hud_theme {

namespace {

constexpr float kPi = 3.14159265358979f;

// Цвета поля. Тёмный вариант: основа почти чёрная с синевой, подсветка
// бирюзово-голубая, центры ядер почти белые. Светлый: ледяная основа,
// линии и ядра глубокого бирюзового цвета.
struct Look {
    ImVec4 accent;   // соты, кольца, рамки
    ImVec4 core;     // центры ядер и импульс в канале
    ImVec4 base;     // сплошная основа под окном и срезы углов
    ImU32  scan;     // строки развёртки
};
const Look kDarkLook  = { ImVec4(0.0f, 229.0f / 255.0f, 1.0f, 1), ImVec4(0.80f, 0.98f, 1.0f, 1),
                          ImVec4(1 / 255.0f, 4 / 255.0f, 7 / 255.0f, 1), IM_COL32(0, 0, 0, 60) };
const Look kLightLook = { ImVec4(0.0f, 0.50f, 0.64f, 1), ImVec4(0.0f, 0.36f, 0.48f, 1),
                          ImVec4(0.91f, 0.965f, 0.98f, 1), 0 };   // светлая — без развёртки
const Look* g_look = &kDarkLook;

inline ImU32 with_alpha(ImVec4 c, float a) {
    c.w = std::clamp(a, 0.0f, 1.0f);
    return ImGui::ColorConvertFloat4ToU32(c);
}
inline ImU32 cyan(float a) { return with_alpha(g_look->accent, a); }
inline ImU32 ice(float a)  { return with_alpha(g_look->core, a); }
inline ImU32 base_col()    { return ImGui::ColorConvertFloat4ToU32(g_look->base); }

// Шутка для коллег: видна, только если присмотреться.
const char* kNote  = "up to 3x faster than CUDAynamics";

struct Core { ImVec2 c; float s; };

// Два ядра по краям главного окна: правое верхнее и левое нижнее.
void cores_for(const ImGuiViewport* vp, Core out[2]) {
    out[0] = { ImVec2(vp->Pos.x + vp->Size.x * 0.88f, vp->Pos.y + vp->Size.y * 0.20f), 1.0f  };
    out[1] = { ImVec2(vp->Pos.x + vp->Size.x * 0.09f, vp->Pos.y + vp->Size.y * 0.84f), 0.75f };
}

// Яркость поля в точке: близость ядер, бегущие от них ударные волны и
// диагональная развёртка сканера. sc — масштаб интерфейса.
float field_energy(ImVec2 p, const Core* cores, float t, float sc) {
    float e = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const Core& k = cores[i];
        const float dx = p.x - k.c.x, dy = p.y - k.c.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        e += k.s * std::exp(-d / (160.0f * sc));
        const float period = 460.0f * sc;
        const float front = std::fmod(t * 120.0f * sc * k.s + i * 0.5f * period, period);
        const float w = (d - front) / (16.0f * sc);
        e += k.s * 0.9f * std::exp(-w * w) * std::exp(-d / (380.0f * sc));
    }
    const float span = 1100.0f * sc;
    float sw = std::fmod(p.x * 0.6f + p.y - t * 200.0f * sc, span);
    if (sw < 0) sw += span;
    const float u = (sw - span * 0.5f) / (30.0f * sc);
    e += 0.25f * std::exp(-u * u);
    return e;
}

// Псевдослучайная фаза соты: вспышки отдельных сот не синхронны.
inline float cell_phase(int col, int row) {
    const float h = std::sin(col * 12.9898f + row * 78.233f) * 43758.5453f;
    return (h - std::floor(h)) * 2.0f * kPi;
}

void hex_points(ImVec2 c, float r, ImVec2 out[6]) {
    for (int k = 0; k < 6; ++k) {
        const float a = kPi / 180.0f * (60.0f * k - 30.0f);
        out[k] = ImVec2(c.x + r * std::cos(a), c.y + r * std::sin(a));
    }
}

void dashed_arc(ImDrawList* dl, ImVec2 c, float r, float a0, float span,
                const float* pattern, int n, ImU32 col, float thick) {
    float a = a0, end = a0 + span;
    int k = 0;
    while (a < end) {
        const float on = pattern[k % n] / r, off = pattern[(k + 1) % n] / r;
        const float b = std::min(a + on, end);
        dl->PathArcTo(c, r, a, b, std::max(2, (int)((b - a) * r / 4.0f)));
        dl->PathStroke(col, 0, thick);
        a = b + off;
        k += 2;
    }
}

bool is_descendant(const ImGuiWindow* w, const ImGuiWindow* root) {
    for (; w; w = w->ParentWindow)
        if (w == root) return true;
    return false;
}

bool wants_backdrop(const ImGuiWindow* w) {
    const ImGuiWindowFlags skip = ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup |
                                  ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoBackground;
    return w && w->Active && !w->Hidden && !(w->Flags & skip) && w->Size.x > 0.0f && w->Size.y > 0.0f;
}

} // namespace

void apply_style(ImGuiStyle& st, bool light) {
    st.WindowRounding = st.ChildRounding = st.FrameRounding = st.PopupRounding = 0.0f;
    st.ScrollbarRounding = st.GrabRounding = st.TabRounding = 0.0f;
    st.WindowBorderSize = 1.0f;
    st.FrameBorderSize  = 1.0f;
    st.PopupBorderSize  = 1.0f;
    const ImVec4 a = light ? kLightLook.accent : kDarkLook.accent;
    auto acc = [&](float alpha) { return ImVec4(a.x, a.y, a.z, alpha); };
    // Галочка и активный бегунок: на тёмном — почти белые, на светлом — темнее акцента.
    const ImVec4 strong = light ? ImVec4(0.00f, 0.32f, 0.42f, 1.0f) : ImVec4(0.65f, 1.00f, 1.00f, 1.0f);
    ImVec4* c = st.Colors;
    c[ImGuiCol_Border]           = acc(light ? 0.45f : 0.35f);
    c[ImGuiCol_BorderShadow]     = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_Separator]        = acc(light ? 0.40f : 0.30f);
    c[ImGuiCol_SeparatorHovered] = acc(0.75f);
    c[ImGuiCol_SeparatorActive]  = acc(1.00f);
    c[ImGuiCol_CheckMark]        = strong;
    c[ImGuiCol_SliderGrab]       = acc(0.85f);
    c[ImGuiCol_SliderGrabActive] = strong;
    c[ImGuiCol_TabSelectedOverline] = acc(1.00f);
    c[ImGuiCol_TitleBgActive]    = light ? ImVec4(0.66f, 0.87f, 0.92f, 1.00f) : ImVec4(0.00f, 0.20f, 0.26f, 1.00f);
    c[ImGuiCol_PlotLines]        = acc(1.00f);
    c[ImGuiCol_PlotHistogram]    = acc(0.85f);
    c[ImGuiCol_TextSelectedBg]   = acc(light ? 0.25f : 0.30f);
    c[ImGuiCol_DockingPreview]   = acc(light ? 0.45f : 0.55f);
    c[ImGuiCol_NavCursor]        = acc(1.00f);
}

void backdrop_extremes(bool light, float out[2][3]) {
    // Самое тёмное и самое светлое место поля под вуалью окна: основа и
    // яркая сота у ядра (линия с заливкой ~35 % цвета подсветки).
    const Look& l = light ? kLightLook : kDarkLook;
    for (int k = 0; k < 3; ++k) {
        const float b = (&l.base.x)[k], a = (&l.accent.x)[k];
        out[0][k] = b;
        out[1][k] = b + (a - b) * 0.35f;
    }
}

void Painter::prepare_frame(float scale) {
    font_ = ImGui::GetFont();
    const float base = ImGui::GetFontSize();
    note_size_  = std::round(base * 0.85f);
    (void)scale;
    // Измерение текста запекает недостающие глифы этого кегля сейчас, внутри
    // кадра, а не после Render, когда обновление атласа уже собрано.
    font_->CalcTextSizeA(note_size_,  FLT_MAX, 0.0f, kNote);
}

ImDrawList* Painter::next_list(ImDrawData* dd) {
    if (used_ == pool_.size())
        pool_.push_back(std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData()));
    ImDrawList* dl = pool_[used_++].get();
    dl->_ResetForNewFrame();
    // Текстура атласа шрифтов, как у списков окон: на ней белый пиксель
    // заливок, сглаживание линий и глифы. Без неё список рисовал текст (и
    // линии) пустой текстурой, и подписи у ядер не было видно.
    dl->PushTexture(ImGui::GetIO().Fonts->TexRef);
    dl->PushClipRect(dd->DisplayPos,
                     ImVec2(dd->DisplayPos.x + dd->DisplaySize.x, dd->DisplayPos.y + dd->DisplaySize.y));
    return dl;
}

void Painter::inject(ImDrawData* dd, float sc, double time, bool computing, bool show_note, bool light) {
    g_look = light ? &kLightLook : &kDarkLook;
    // Часы анимации. Скорость: 1 без расчёта, 0.3 в первые секунды расчёта,
    // 0 дальше; к цели подтягивается экспоненциально (~0.6 с), без скачков.
    {
        const double dt = last_time_ < 0.0 ? 0.0 : std::clamp(time - last_time_, 0.0, 0.5);
        last_time_ = time;
        if (computing && compute_start_ < 0.0) compute_start_ = time;
        if (!computing) compute_start_ = -1.0;
        const float target = !computing ? 1.0f : (time - compute_start_ < 6.0 ? 0.3f : 0.0f);
        speed_ += (target - speed_) * (1.0f - (float)std::exp(-dt / 0.6));
        if (speed_ < 0.002f && target == 0.0f) speed_ = 0.0f;
        anim_time_ = std::fmod(anim_time_ + dt * speed_, 3600.0);
    }
    if (!dd || dd->CmdListsCount == 0 || sc <= 0.0f) return;
    ImGuiContext& g = *GImGui;

    // Спрайт сияния: белый круг с плавно спадающей прозрачностью, цвет задаётся
    // вершинами AddImage.
    if (!glow_tried_) {
        glow_tried_ = true;
        const int n = 64;
        std::vector<unsigned char> px((size_t)n * n * 4);
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const float fx = (x + 0.5f) / n * 2 - 1, fy = (y + 0.5f) / n * 2 - 1;
                const float r = std::min(1.0f, std::sqrt(fx * fx + fy * fy));
                const float a = std::pow(1.0f - r, 2.2f);
                unsigned char* p = &px[((size_t)y * n + x) * 4];
                p[0] = p[1] = p[2] = 255;
                p[3] = (unsigned char)std::lround(a * 255.0f);
            }
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, n, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        glow_tex_ = tex;
    }
    const ImTextureRef glow((ImTextureID)(intptr_t)glow_tex_);

    const float t = (float)anim_time_;   // < 3600: фазы без потери точности
    Core cores[2];
    cores_for(ImGui::GetMainViewport(), cores);

    // Энергоканал от нижнего ядра к верхнему: короткий отрезок под 30°,
    // горизонталь и подъём под 60° — по направлениям сотовой сетки.
    ImVec2 chan[4];
    chan[0] = cores[1].c;
    chan[1] = ImVec2(cores[1].c.x + 110.0f * sc, cores[1].c.y - 64.0f * sc);
    chan[3] = cores[0].c;
    chan[2] = ImVec2(cores[0].c.x - (chan[1].y - cores[0].c.y) * 0.577f, chan[1].y);
    float seg[3], total = 0.0f;
    for (int k = 0; k < 3; ++k) {
        seg[k] = std::hypot(chan[k + 1].x - chan[k].x, chan[k + 1].y - chan[k].y);
        total += seg[k];
    }
    ImVec2 pulse = chan[0];
    {
        float q = std::fmod(t * 0.25f, 1.0f) * total;
        int k = 0;
        while (k < 2 && q > seg[k]) { q -= seg[k]; ++k; }
        const float f = seg[k] > 0 ? q / seg[k] : 0.0f;
        pulse = ImVec2(chan[k].x + (chan[k + 1].x - chan[k].x) * f, chan[k].y + (chan[k + 1].y - chan[k].y) * f);
    }

    // Окно, чей правый нижний угол ближе всех к углу главного окна.
    const ImGuiWindow* corner_win = nullptr;
    {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        const ImVec2 br(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
        float best = 1e30f;
        for (ImGuiWindow* w : g.Windows) {
            if (!wants_backdrop(w)) continue;
            const float dx = br.x - (w->Pos.x + w->Size.x), dy = br.y - (w->Pos.y + w->Size.y);
            const float d = dx * dx + dy * dy;
            if (d < best) { best = d; corner_win = w; }
        }
    }

    const float R  = 13.0f * sc;
    const float HW = 1.7320508f * R;
    const float RS = 1.5f * R;

    auto paint_under = [&](ImDrawList* dl, const ImGuiWindow* w) {
        const ImVec2 a = w->Pos, b(w->Pos.x + w->Size.x, w->Pos.y + w->Size.y);
        dl->PushClipRect(a, b, true);
        dl->AddRectFilled(a, b, base_col());

        for (const Core& k : cores) {
            const float r = 240.0f * sc * k.s;
            dl->AddImage(glow, ImVec2(k.c.x - r, k.c.y - r), ImVec2(k.c.x + r, k.c.y + r),
                         ImVec2(0, 0), ImVec2(1, 1), cyan(0.30f + 0.07f * std::sin(t * 2.0f)));
        }

        // Соты: только попадающие в окно.
        const int r0 = (int)std::floor((a.y - R) / RS), r1 = (int)std::ceil((b.y + R) / RS);
        const int c0 = (int)std::floor((a.x - HW) / HW), c1 = (int)std::ceil((b.x + HW) / HW);
        ImVec2 pts[6];
        for (int row = r0; row <= r1; ++row) {
            const float cy = row * RS;
            const float off = (row & 1) ? HW * 0.5f : 0.0f;
            for (int col = c0; col <= c1; ++col) {
                const ImVec2 c(col * HW + off, cy);
                float e = field_energy(c, cores, t, sc);
                const float s = std::sin(cell_phase(col, row) + t * 0.9f);
                if (s > 0.0f) { const float s2 = s * s, s4 = s2 * s2, s8 = s4 * s4, s16 = s8 * s8;
                                e += 0.55f * s16 * s16 * s8 * s4; }   // s^60: короткая вспышка
                e = std::min(e, 1.0f);
                hex_points(c, R - 1.2f * sc, pts);
                if (e > 0.45f) dl->AddConvexPolyFilled(pts, 6, cyan(0.14f * (e - 0.45f) / 0.55f));
                // Мягкие края сот: широкий бледный проход под тонким — дешёвая
                // замена размытию (настоящее потребовало бы рендера в текстуру).
                const float la = 0.07f + 0.55f * e;
                dl->AddPolyline(pts, 6, cyan(la * 0.30f), ImDrawFlags_Closed, 3.5f * sc);
                dl->AddPolyline(pts, 6, cyan(la * 0.45f), ImDrawFlags_Closed, 1.0f);
            }
        }

        dl->AddPolyline(chan, 4, cyan(0.40f), 0, 1.0f * sc);
        const float pr = 20.0f * sc;
        dl->AddImage(glow, ImVec2(pulse.x - pr, pulse.y - pr), ImVec2(pulse.x + pr, pulse.y + pr),
                     ImVec2(0, 0), ImVec2(1, 1), ice(0.90f));

        // Кольца прицела вокруг ядер: сплошное, пунктир и штрих-пунктир,
        // соседние вращаются навстречу.
        static const float dash1[] = { 3, 6 };
        static const float dash2[] = { 20, 10, 4, 10 };
        for (const Core& k : cores) {
            const float base[3] = { 34.0f, 52.0f, 78.0f };
            for (int j = 0; j < 3; ++j) {
                const float r = base[j] * k.s * sc;
                const float dir = (j & 1) ? -1.0f : 1.0f;
                const float a0 = dir * t * (0.6f + 0.3f * j);
                const ImU32 col = cyan(0.70f - 0.18f * j);
                if (j == 0) {
                    dl->PathArcTo(k.c, r, a0, a0 + kPi * 1.6f, 48);
                    dl->PathStroke(col, 0, 1.0f * sc);
                } else {
                    float pat[4];
                    const float* src = j == 1 ? dash1 : dash2;
                    const int n = j == 1 ? 2 : 4;
                    for (int q = 0; q < n; ++q) pat[q] = src[q] * sc;
                    dashed_arc(dl, k.c, r, a0, kPi * 1.6f, pat, n, col, 1.0f * sc);
                }
            }
            hex_points(k.c, 8.0f * k.s * sc, pts);
            dl->AddConvexPolyFilled(pts, 6, ice(0.80f + 0.20f * std::sin(t * 6.0f)));
            const float gr = 60.0f * sc * k.s;
            dl->AddImage(glow, ImVec2(k.c.x - gr, k.c.y - gr), ImVec2(k.c.x + gr, k.c.y + gr),
                         ImVec2(0, 0), ImVec2(1, 1), ice(0.55f));
        }


        // Строки развёртки ЭЛТ — только по фону, под содержимым окна: поверх
        // графиков, текста и полей ввода они мешали.
        const float step = std::max(3.0f, std::round(3.0f * sc));
        for (float y = std::floor(a.y / step) * step; g_look->scan != 0 && y < b.y; y += step)
            dl->AddRectFilled(ImVec2(a.x, y), ImVec2(b.x, y + 1.0f), g_look->scan);
        dl->PopClipRect();
    };

    auto paint_over = [&](ImDrawList* dl, const ImGuiWindow* w) {
        const ImVec2 a = w->Pos, b(w->Pos.x + w->Size.x, w->Pos.y + w->Size.y);
        if (b.x - a.x < 24.0f * sc || b.y - a.y < 24.0f * sc) return;
        dl->PushClipRect(a, b, true);
        const bool focused = is_descendant(g.NavWindow, w);
        const float th = std::max(1.0f, std::round(sc));

        if (focused)
            for (int k = 3; k >= 1; --k) {
                const float in = k * th;
                dl->AddRect(ImVec2(a.x + in, a.y + in), ImVec2(b.x - in, b.y - in),
                            cyan(0.10f * (4 - k)), 0.0f, 0, th);
            }
        dl->AddRect(ImVec2(a.x + 0.5f, a.y + 0.5f), ImVec2(b.x - 0.5f, b.y - 0.5f),
                    cyan(focused ? 0.90f : 0.30f), 0.0f, 0, 1.0f);

        // Срезанные углы: левый верхний всегда, правый нижний — если там нет
        // уголка изменения размера (пристыкованное окно или NoResize).
        const float ch = 7.0f * sc;
        const ImU32 edge = cyan(focused ? 0.95f : 0.45f);
        dl->AddTriangleFilled(a, ImVec2(a.x + ch, a.y), ImVec2(a.x, a.y + ch), base_col());
        dl->AddLine(ImVec2(a.x + ch, a.y + 0.5f), ImVec2(a.x + 0.5f, a.y + ch), edge, th);
        const bool cut_br = w->DockIsActive || (w->Flags & ImGuiWindowFlags_NoResize);
        if (cut_br) {
            dl->AddTriangleFilled(b, ImVec2(b.x - ch, b.y), ImVec2(b.x, b.y - ch), base_col());
            dl->AddLine(ImVec2(b.x - ch, b.y - 0.5f), ImVec2(b.x - 0.5f, b.y - ch), edge, th);
        }

        // Скобки прицела в двух других углах.
        const float L = 14.0f * sc, bt = 2.0f * th, o = 1.0f;
        const ImU32 br = cyan(focused ? 0.90f : 0.40f);
        dl->AddLine(ImVec2(b.x - L, a.y + o), ImVec2(b.x - o, a.y + o), br, bt);
        dl->AddLine(ImVec2(b.x - o, a.y + o), ImVec2(b.x - o, a.y + L), br, bt);
        dl->AddLine(ImVec2(a.x + o, b.y - L), ImVec2(a.x + o, b.y - o), br, bt);
        dl->AddLine(ImVec2(a.x + o, b.y - o), ImVec2(a.x + L, b.y - o), br, bt);

        // Шкала с засечками по нижней кромке и бегущие наклонные полоски у
        // правого нижнего угла.
        const float step = 8.0f * sc;
        int i = 0;
        for (float x = a.x + 22.0f * sc; x < b.x - 90.0f * sc; x += step, ++i) {
            const float hgt = (i % 5 == 0 ? 5.0f : 2.5f) * sc;
            dl->AddLine(ImVec2(std::floor(x) + 0.5f, b.y - 1.0f), ImVec2(std::floor(x) + 0.5f, b.y - 1.0f - hgt),
                        cyan(0.28f), 1.0f);
        }
        const int lit = (int)std::floor(t * 4.0f);
        for (int k = 0; k < 6; ++k) {
            const float x0 = b.x - (cut_br ? 20.0f : 26.0f) * sc - (5 - k) * 9.0f * sc;
            const float y0 = b.y - 7.0f * sc, y1 = b.y - 2.0f * sc;
            const ImVec2 q[4] = { ImVec2(x0 + 2.0f * sc, y0), ImVec2(x0 + 7.0f * sc, y0),
                                  ImVec2(x0 + 5.0f * sc, y1), ImVec2(x0, y1) };
            dl->AddConvexPolyFilled(q, 4, cyan(0.15f + 0.45f * (float)((k + lit) % 6) / 5.0f));
        }
        // Шутка для коллег — только у окна в правом нижнем углу экрана, над
        // полосками, мелко и тускло.
        if (show_note && w == corner_win && font_) {
            const ImVec2 ns = font_->CalcTextSizeA(note_size_, FLT_MAX, 0.0f, kNote);
            const float right = b.x - (cut_br ? 20.0f : 26.0f) * sc + 7.0f * sc;
            dl->AddText(font_, note_size_,
                        ImVec2(std::floor(right - ns.x), std::floor(b.y - 9.0f * sc - ns.y)),
                        cyan(0.35f), kNote);
        }
        dl->PopClipRect();
    };

    used_ = 0;
    ImVector<ImDrawList*> out;
    out.reserve(dd->CmdListsCount * 3 + 1);
    const ImGuiWindow* pending = nullptr;   // окно, чьи рамки ещё не выведены
    auto flush = [&]() {
        if (!pending) return;
        ImDrawList* dl = next_list(dd);
        paint_over(dl, pending);
        out.push_back(dl);
        pending = nullptr;
    };
    for (ImDrawList* list : dd->CmdLists) {
        const ImGuiWindow* win = nullptr;
        for (ImGuiWindow* w : g.Windows)
            if (w->DrawList == list) { win = w; break; }
        // Рамки окна — после него и его дочерних окон, но до следующего
        // корневого окна или всплывающего меню.
        if (pending && !is_descendant(win, pending)) flush();
        if (wants_backdrop(win)) {
            ImDrawList* dl = next_list(dd);
            paint_under(dl, win);
            out.push_back(dl);
            pending = win;
        }
        out.push_back(list);
    }
    flush();

    for (size_t k = 0; k < used_; ++k) {
        dd->TotalVtxCount += pool_[k]->VtxBuffer.Size;
        dd->TotalIdxCount += pool_[k]->IdxBuffer.Size;
    }
    dd->CmdLists.swap(out);
    dd->CmdListsCount = dd->CmdLists.Size;
}

void Painter::release() {
    pool_.clear();
    used_ = 0;
    if (glow_tex_) {
        GLuint t = glow_tex_;
        glDeleteTextures(1, &t);
        glow_tex_ = 0;
    }
    glow_tried_ = false;
    font_ = nullptr;
}

} // namespace hud_theme
