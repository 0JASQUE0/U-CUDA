#pragma once
// Общие примитивы анимированных сцен под окнами (Dark Side, Light Side):
// детерминированный хеш и генератор, цвет, сияние спрайтом, луч с ореолом.
// Сцена — чистая функция времени t: под каждым окном рисуется заново и должна
// совпадать на стыках окон.

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hud_theme::scene {

constexpr float kPi = 3.14159265358979f;

// Целочисленный хеш в [0, 1).
inline float hash01(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}
inline float hash01(uint32_t a, uint32_t b) { return hash01(a * 0x9E3779B1U ^ (b + 0x7F4A7C15U)); }
inline float fract(float x) { return x - std::floor(x); }

struct Rng {
    uint32_t s;
    float next() { s = s * 1664525U + 1013904223U; return hash01(s); }
};

inline ImU32 rgba(float r, float g, float b, float a) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, std::clamp(a, 0.0f, 1.0f)));
}

inline void glow_at(ImDrawList* dl, ImTextureRef glow, ImVec2 c, float r, ImU32 col) {
    dl->AddImage(glow, ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), ImVec2(0, 0), ImVec2(1, 1), col);
}

inline bool inside(ImVec2 p, ImVec2 a, ImVec2 b, float m) {
    return p.x > a.x - m && p.x < b.x + m && p.y > a.y - m && p.y < b.y + m;
}

// Цвет энергии: тело луча, раскалённое ядро, огонь вспышек.
struct Tint {
    float body[3], core[3], flame[3];
    ImU32 b(float a) const { return rgba(body[0], body[1], body[2], a); }
    ImU32 c(float a) const { return rgba(core[0], core[1], core[2], a); }
    ImU32 f(float a) const { return rgba(flame[0], flame[1], flame[2], a); }
};

// Луч с ореолом: широкие бледные проходы под узким ярким ядром — дешёвая
// замена размытию, как у краёв сот в Cyan HUD.
inline void beam(ImDrawList* dl, const ImVec2* p, int n, float w, float a, float core_w, const Tint& tn) {
    if (a <= 0.0f) return;
    dl->AddPolyline(p, n, tn.b(0.16f * a), 0, w * 2.6f);
    dl->AddPolyline(p, n, tn.b(0.35f * a), 0, w * 1.6f);
    dl->AddPolyline(p, n, tn.b(0.85f * a), 0, w);
    if (core_w > 0.0f) dl->AddPolyline(p, n, tn.c(0.95f * a), 0, core_w);
}
inline void beam(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float w, float a, float core_w, const Tint& tn) {
    const ImVec2 p[2] = { p0, p1 };
    beam(dl, p, 2, w, a, core_w, tn);
}

} // namespace hud_theme::scene
