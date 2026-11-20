#include "hud_jedi.h"
#include "hud_scene.h"

namespace hud_theme {

namespace {

using namespace scene;

// Голубые залпы и синий клинок: на светлом небе ядро не белое (белое
// терялось бы), а бледно-голубое.
constexpr Tint kBoltTint  { {0.00f, 0.36f, 1.00f}, {0.70f, 0.88f, 1.00f}, {0.45f, 0.75f, 1.00f} };
constexpr Tint kSaberTint { {0.00f, 0.30f, 0.98f}, {0.78f, 0.92f, 1.00f}, {0.40f, 0.70f, 1.00f} };
constexpr Tint kSaber2Tint { {0.05f, 0.72f, 0.20f}, {0.70f, 1.00f, 0.72f}, {0.30f, 0.90f, 0.40f} };

// Цвет неба: к нему тянутся дальние объекты (воздушная перспектива).
constexpr float kSky[3] = { 0.78f, 0.88f, 1.00f };

inline ImU32 hazed(float r, float g, float b, float a, float h) {
    return rgba(r + (kSky[0] - r) * h, g + (kSky[1] - g) * h, b + (kSky[2] - b) * h, a);
}

// Небо по высоте вьюпорта: насыщенно-голубое вверху, светлое у горизонта.
ImU32 sky_at(float f) {
    f = std::clamp(f, 0.0f, 1.0f);
    return rgba(0.64f + 0.27f * f, 0.79f + 0.16f * f, 0.99f + 0.01f * f, 1.0f);
}

// Звенья крестокрылых истребителей. dir = +1 — слева направо. s — масштаб
// (меньше = дальше), haze — сколько цвета неба подмешано.
struct Flight { float v, off, s, haze; int n, dir; uint32_t seed; };
const Flight kFlights[3] = {
    { 0.030f, 0.55f, 0.55f, 0.45f, 2, +1, 9 },    // дальнее звено
    { 0.050f, 0.20f, 0.75f, 0.20f, 1, -1, 13 },   // одиночка навстречу
    { 0.042f, 0.00f, 1.00f, 0.00f, 3, +1, 5 },    // ведущее звено, ближе всех
};

struct Pose { ImVec2 c; float L, cs, sn; bool flip; };

// Точка корабля: x вперёд к носу, y вниз (кабина сверху), в долях длины.
inline ImVec2 W(const Pose& p, float x, float y) {
    const float yy = p.flip ? -y : y;
    return ImVec2(p.c.x + p.L * (x * p.cs - yy * p.sn), p.c.y + p.L * (x * p.sn + yy * p.cs));
}

// Центр ведущего: проход через экран с плавным покачиванием; на каждом
// проходе своя высота и наклон траектории.
ImVec2 leader_at(const Flight& f, float t, ImVec2 P, ImVec2 S) {
    const float u = t * f.v + f.off;
    const float q = fract(u);
    const uint32_t pass = (uint32_t)(int)std::floor(u);
    const float x = f.dir > 0 ? -0.35f + q * 1.7f : 1.35f - q * 1.7f;
    const float y0 = 0.22f + 0.50f * hash01(f.seed, pass);
    const float slope = (hash01(f.seed, pass + 77U) - 0.5f) * 0.30f;
    const float y = y0 + slope * q + 0.035f * std::sin(q * 9.0f + f.seed);
    return ImVec2(P.x + S.x * x, P.y + S.y * y);
}

Pose fighter_pose(const Flight& f, int m, float t, ImVec2 P, ImVec2 S) {
    const ImVec2 c0 = leader_at(f, t, P, S), c1 = leader_at(f, t + 0.05f, P, S);
    float dx = c1.x - c0.x, dy = c1.y - c0.y;
    const float d = std::hypot(dx, dy);
    if (d < 1e-4f || d > S.x * 0.2f) { dx = (float)f.dir; dy = 0.0f; }   // стык проходов
    else { dx /= d; dy /= d; }
    const float L = std::min(S.x * 0.16f, S.y * 0.30f) * f.s;
    // Ведомые — уступом назад, выше и ниже ведущего, со своим покачиванием.
    float bx = 0.0f, by = 0.0f;
    if (m > 0) { bx = -1.0f; by = (m == 1 ? -0.75f : 0.75f); }
    by += 0.05f * std::sin(t * 1.7f + m * 2.1f);
    const float fl = dx < 0.0f ? -1.0f : 1.0f;   // зеркальный: кабина остаётся сверху
    const ImVec2 c(c0.x + L * (bx * dx - by * fl * dy), c0.y + L * (bx * dy + by * fl * dx));
    return { c, L * (m > 0 ? 0.9f : 1.0f), dx, dy, dx < 0.0f };
}

// Крылья открыты буквой X: вид сбоку-сзади, ближние крылья длиннее дальних.
constexpr ImVec2 kNearWing[4] = { {-0.48f, -0.02f}, {-0.12f, -0.02f}, {-0.34f, -0.44f}, {-0.50f, -0.44f} };
constexpr ImVec2 kFarWing[4]  = { {-0.44f, -0.04f}, {-0.18f, -0.04f}, {-0.36f, -0.34f}, {-0.48f, -0.34f} };

void wing(ImDrawList* dl, const Pose& p, const ImVec2* w, float sy, ImU32 fill, ImU32 line, float th) {
    ImVec2 q[4];
    for (int i = 0; i < 4; ++i) q[i] = W(p, w[i].x, w[i].y * sy);
    dl->AddConvexPolyFilled(q, 4, fill);
    dl->AddPolyline(q, 4, line, ImDrawFlags_Closed, th);
}

// Концы пушек ближних крыльев (сверху и снизу) — отсюда идут залпы.
inline ImVec2 gun_tip(const Pose& p, float sy) { return W(p, 0.12f, -0.445f * sy); }

void draw_fighter(ImDrawList* dl, ImTextureRef glow, const Pose& p, float h, float sc, float t, int id) {
    const float th = std::max(1.0f, 1.5f * sc * p.L / 120.0f);
    const ImU32 hull  = hazed(0.95f, 0.96f, 0.97f, 1.0f, h);
    const ImU32 hullF = hazed(0.64f, 0.69f, 0.77f, 1.0f, h);
    const ImU32 line  = hazed(0.10f, 0.16f, 0.26f, 1.0f, h);
    const ImU32 lineF = hazed(0.22f, 0.30f, 0.42f, 1.0f, h);
    const ImU32 strip = hazed(0.16f, 0.42f, 0.88f, 1.0f, h);
    const ImU32 gun   = hazed(0.18f, 0.22f, 0.30f, 1.0f, h);

    // Дальние крылья и их пушки.
    for (float sy : { -1.0f, 1.0f }) {
        wing(dl, p, kFarWing, sy, hullF, lineF, th);
        dl->AddLine(W(p, -0.48f, -0.35f * sy), W(p, 0.02f, -0.35f * sy), gun, 1.6f * th);
    }
    // Выхлоп: голубое сияние у сопел и тающий след.
    const float fl = 0.85f + 0.15f * std::sin(t * 23.0f + id);
    for (float sy : { -1.0f, 1.0f }) {
        const ImVec2 e = W(p, -0.52f, 0.075f * sy);
        beam(dl, e, W(p, -1.30f, 0.075f * sy), 0.05f * p.L, 0.35f * (1.0f - h), 0.0f, kBoltTint);
        glow_at(dl, glow, e, 0.16f * p.L, hazed(0.35f, 0.65f, 1.0f, 0.90f * fl, h));
    }
    // Фюзеляж с длинным носом, кабина.
    const ImVec2 body[7] = { W(p, -0.50f, -0.085f), W(p, 0.00f, -0.070f), W(p, 0.50f, -0.020f), W(p, 0.54f, 0.0f),
                             W(p, 0.50f, 0.020f), W(p, 0.00f, 0.070f), W(p, -0.50f, 0.085f) };
    dl->AddConvexPolyFilled(body, 7, hull);
    dl->AddPolyline(body, 7, line, ImDrawFlags_Closed, th);
    dl->AddLine(W(p, 0.18f, -0.035f), W(p, 0.18f, 0.035f), strip, 2.0f * th);
    dl->AddLine(W(p, 0.26f, -0.030f), W(p, 0.26f, 0.030f), strip, 2.0f * th);
    const ImVec2 cab[4] = { W(p, -0.14f, -0.068f), W(p, 0.06f, -0.060f), W(p, 0.00f, -0.110f), W(p, -0.11f, -0.115f) };
    dl->AddConvexPolyFilled(cab, 4, hazed(0.12f, 0.22f, 0.34f, 1.0f, h));
    // Ближние крылья: синяя полоса у законцовки, пушка вдоль кромки.
    for (float sy : { -1.0f, 1.0f }) {
        wing(dl, p, kNearWing, sy, hull, line, th);
        auto lerp = [](ImVec2 u, ImVec2 v, float k) { return ImVec2(u.x + (v.x - u.x) * k, u.y + (v.y - u.y) * k); };
        const ImVec2* w = kNearWing;
        ImVec2 s[4] = { lerp(w[0], w[3], 0.62f), lerp(w[1], w[2], 0.62f), lerp(w[1], w[2], 0.80f), lerp(w[0], w[3], 0.80f) };
        for (ImVec2& v : s) v = W(p, v.x, v.y * sy);
        dl->AddConvexPolyFilled(s, 4, strip);
        dl->AddLine(W(p, -0.50f, -0.445f * sy), gun_tip(p, sy), gun, 2.2f * th);
    }
}

} // namespace

void paint_jedi_scene(ImDrawList* dl, const ImGuiViewport* vp, ImVec2 a, ImVec2 b,
                      float sc, float t, ImTextureRef glow) {
    const ImVec2 P = vp->Pos, S = vp->Size;
    if (S.x <= 0.0f || S.y <= 0.0f) return;

    // Небо: градиент по высоте вьюпорта, одинаковый под всеми окнами.
    dl->AddRectFilledMultiColor(a, b, sky_at((a.y - P.y) / S.y), sky_at((a.y - P.y) / S.y),
                                sky_at((b.y - P.y) / S.y), sky_at((b.y - P.y) / S.y));

    // Солнце с медленно вращающимися лучами.
    {
        const ImVec2 c(P.x + S.x * 0.15f, P.y + S.y * 0.13f);
        const float R = std::hypot(S.x, S.y);
        for (int i = 0; i < 14; ++i) {
            const float an = i / 14.0f * 2.0f * kPi + t * 0.03f;
            const float w = 0.045f + 0.03f * hash01((uint32_t)i, 3U);
            const float al = 0.13f + 0.07f * std::sin(t * 0.8f + i * 1.7f);
            const ImVec2 p1(c.x + std::cos(an - w) * R, c.y + std::sin(an - w) * R);
            const ImVec2 p2(c.x + std::cos(an + w) * R, c.y + std::sin(an + w) * R);
            dl->AddTriangleFilled(c, p1, p2, rgba(1.0f, 1.0f, 1.0f, al));
        }
        glow_at(dl, glow, c, S.y * 0.45f, rgba(1.0f, 1.0f, 1.0f, 0.95f));
        glow_at(dl, glow, c, S.y * 0.12f, rgba(1.0f, 1.0f, 1.0f, 1.0f));
        dl->AddCircleFilled(c, S.y * 0.035f, rgba(1.0f, 1.0f, 1.0f, 1.0f), 48);
    }

    // Окольцованная планета: задняя половина кольца, диск, передняя половина.
    {
        const ImVec2 c(P.x + S.x * 0.84f, P.y + S.y * 0.30f);
        const float r = S.y * 0.13f, rx = r * 1.95f, ry = r * 0.36f, rot = -0.32f;
        const float cr = std::cos(rot), sr = std::sin(rot);
        auto ring = [&](float f0, float f1, float al) {
            ImVec2 pts[49];
            for (int i = 0; i <= 48; ++i) {
                const float f = f0 + (f1 - f0) * i / 48.0f;
                const float x = std::cos(f) * rx, y = std::sin(f) * ry;
                pts[i] = ImVec2(c.x + x * cr - y * sr, c.y + x * sr + y * cr);
            }
            dl->AddPolyline(pts, 49, rgba(0.62f, 0.76f, 0.95f, 0.45f * al), 0, 9.0f * sc);
            dl->AddPolyline(pts, 49, rgba(0.36f, 0.56f, 0.88f, 0.75f * al), 0, 2.5f * sc);
        };
        ring(kPi, 2.0f * kPi, 0.8f);
        glow_at(dl, glow, c, r * 1.5f, rgba(0.70f, 0.85f, 1.0f, 0.55f));
        dl->AddCircleFilled(c, r, rgba(0.50f, 0.66f, 0.90f, 1.0f), 96);
        for (int i = -2; i <= 2; ++i) {   // облачные пояса
            const float yy = i * 0.32f;
            dl->AddEllipse(ImVec2(c.x, c.y + yy * r), ImVec2(std::sqrt(1.0f - yy * yy) * r * 0.98f, r * 0.07f),
                           rgba(0.80f, 0.88f, 0.98f, 0.45f), 0.0f, 64, 2.0f * sc);
        }
        for (int k = 1; k <= 4; ++k) {   // тень к правому нижнему краю
            const float o = 0.12f * r * k / 1.4142f;
            dl->AddCircleFilled(ImVec2(c.x + o, c.y + o), r * (1.0f - 0.12f * k), rgba(0.40f, 0.55f, 0.82f, 0.22f), 72);
        }
        dl->PathArcTo(c, r * 0.97f, kPi * 0.95f, kPi * 1.55f, 32);   // освещённый край
        dl->PathStroke(rgba(1.0f, 1.0f, 1.0f, 0.85f), 0, 2.5f * sc);
        dl->AddCircle(c, r, rgba(0.36f, 0.52f, 0.80f, 0.70f), 96, 1.2f * sc);
        ring(0.0f, kPi, 1.0f);
        // Спутник.
        const ImVec2 m(P.x + S.x * 0.70f, P.y + S.y * 0.11f);
        dl->AddCircleFilled(m, S.y * 0.022f, rgba(0.74f, 0.83f, 0.95f, 1.0f), 32);
        dl->AddCircle(m, S.y * 0.022f, rgba(0.45f, 0.60f, 0.85f, 0.7f), 32, 1.0f * sc);
    }

    // Облака: гряды мягких пятен плывут влево, у нижнего края — плотнее.
    for (uint32_t i = 0; i < 9; ++i) {
        const float v = 0.004f + 0.006f * hash01(i, 21);
        const float x = 1.25f - fract(hash01(i, 22) + v * t) * 1.5f;
        const float y = 0.55f + 0.42f * hash01(i, 23);
        const float s = (0.6f + 0.6f * hash01(i, 24)) * S.y * 0.07f;
        for (uint32_t k = 0; k < 6; ++k) {
            const ImVec2 q(P.x + S.x * x + (hash01(i * 7 + k, 25) - 0.5f) * s * 5.0f,
                           P.y + S.y * y + (hash01(i * 7 + k, 26) - 0.5f) * s * 1.2f);
            const float r = s * (1.0f + 1.2f * hash01(i * 7 + k, 27));
            if (!inside(q, a, b, r)) continue;
            glow_at(dl, glow, q, r, rgba(1.0f, 1.0f, 1.0f, 0.75f));
        }
    }

    // Два световых клинка, синий и зелёный: медленно сходятся и расходятся,
    // в точке касания — искры. Рукояти — сталь с тёмными кольцами.
    {
        const float cx = P.x + S.x * 0.45f, cy = P.y + S.y * 0.94f, L = S.y * 0.55f;
        const float a1 = -1.10f + 0.28f * std::sin(t * 0.6f);
        const float a2 = -2.04f - 0.28f * std::sin(t * 0.6f + 0.5f);
        const ImVec2 h[2] = { ImVec2(cx - 0.30f * L, cy), ImVec2(cx + 0.30f * L, cy) };
        const ImVec2 dir[2] = { ImVec2(std::cos(a1), std::sin(a1)), ImVec2(std::cos(a2), std::sin(a2)) };
        const Tint* tint[2] = { &kSaberTint, &kSaber2Tint };
        const float fl = 0.90f + 0.10f * std::sin(t * 37.0f) * std::sin(t * 23.0f);
        for (int i = 0; i < 2; ++i) {
            const ImVec2 d = dir[i], hlt = h[i];
            beam(dl, hlt, ImVec2(hlt.x + d.x * L, hlt.y + d.y * L), 8.0f * sc, fl, 2.8f * sc, *tint[i]);
            const ImVec2 back(hlt.x - d.x * 34.0f * sc, hlt.y - d.y * 34.0f * sc);
            dl->AddLine(hlt, back, rgba(0.58f, 0.62f, 0.68f, 1.0f), 6.0f * sc);
            for (int q = 1; q < 4; ++q) {
                const float o0 = q * 8.0f * sc, o1 = o0 + 2.5f * sc;
                dl->AddLine(ImVec2(hlt.x - d.x * o0, hlt.y - d.y * o0), ImVec2(hlt.x - d.x * o1, hlt.y - d.y * o1),
                            rgba(0.18f, 0.22f, 0.28f, 1.0f), 7.0f * sc);
            }
            glow_at(dl, glow, hlt, 14.0f * sc, tint[i]->f(0.7f * fl));
        }
        // Пересечение отрезков клинков: искры, ярче у концов.
        const ImVec2 r(dir[0].x * L, dir[0].y * L), s(dir[1].x * L, dir[1].y * L);
        const float den = r.x * s.y - r.y * s.x;
        if (std::fabs(den) > 1e-6f) {
            const ImVec2 qp(h[1].x - h[0].x, h[1].y - h[0].y);
            const float u = (qp.x * s.y - qp.y * s.x) / den, w = (qp.x * r.y - qp.y * r.x) / den;
            if (u >= 0.0f && u <= 1.0f && w >= 0.0f && w <= 1.0f) {
                const ImVec2 ip(h[0].x + r.x * u, h[0].y + r.y * u);
                const float k = 0.45f + 0.55f * std::clamp((u - 0.6f) / 0.35f, 0.0f, 1.0f);
                Rng rng{ (uint32_t)(int)std::floor(t * 30.0f) * 2654435761U };
                for (int j = 0; j < 8; ++j) {
                    const float an = rng.next() * 2.0f * kPi, len = rng.next() * 30.0f * sc * k;
                    beam(dl, ip, ImVec2(ip.x + std::cos(an) * len, ip.y + std::sin(an) * len), 1.3f * sc, k,
                         0.8f * sc, j & 1 ? kSaber2Tint : kSaberTint);
                }
                glow_at(dl, glow, ip, 26.0f * sc * k, rgba(1.0f, 1.0f, 1.0f, 0.9f));
            }
        }
    }

    // Истребители, дальние первыми, и их залпы.
    for (int fi = 0; fi < 3; ++fi) {
        const Flight& f = kFlights[fi];
        for (int m = 0; m < f.n; ++m) {
            const int id = fi * 4 + m;
            const Pose p = fighter_pose(f, m, t, P, S);
            if (inside(p.c, a, b, p.L * 1.6f)) draw_fighter(dl, glow, p, f.haze, sc, t, id);

            // Залп k: пара болтов из концов пушек в момент tf; живут 0.8 с.
            const float every = 1.3f, life = 0.8f;
            const int k1 = (int)std::floor(t / every), k0 = (int)std::floor((t - life) / every) - 1;
            for (int k = k0; k <= k1; ++k) {
                const uint32_t key = (uint32_t)k * 7919U + (uint32_t)id * 131U + 3U;
                if (hash01(key, 1) < 0.35f) continue;   // не каждый раз
                const float tf = (k + 0.7f * hash01(key, 2)) * every;
                const float age = t - tf;
                if (age < 0.0f || age > life) continue;
                const Pose sp = fighter_pose(f, m, tf, P, S);
                const ImVec2 dir(sp.cs, sp.sn);
                const float spd = 1.3f * S.x, len = 0.35f * sp.L;
                for (float sy : { -1.0f, 1.0f }) {
                    const ImVec2 o = gun_tip(sp, sy);
                    const ImVec2 q(o.x + dir.x * spd * age, o.y + dir.y * spd * age);
                    if (!inside(q, a, b, len + 20.0f * sc)) continue;
                    beam(dl, q, ImVec2(q.x - dir.x * len, q.y - dir.y * len), 2.4f * sc * f.s + 0.6f * sc,
                         1.0f - f.haze * 0.6f, 1.0f * sc, kBoltTint);
                }
                if (age < 0.08f)   // вспышка у дульного среза (по текущему положению)
                    for (float sy : { -1.0f, 1.0f })
                        glow_at(dl, glow, gun_tip(p, sy), 0.10f * p.L, kBoltTint.f(1.0f - age / 0.08f));
            }
        }
    }

    // Пылинки поднимаются и мерцают.
    for (uint32_t i = 0; i < 80; ++i) {
        const float v = 0.01f + 0.03f * hash01(i, 31);
        const float ph = hash01(i, 32) * 2.0f * kPi;
        const ImVec2 q(P.x + hash01(i, 33) * S.x + std::sin(t * 0.7f + ph) * 10.0f * sc,
                       P.y + S.y * (1.0f - fract(hash01(i, 34) + v * t)));
        if (!inside(q, a, b, 6.0f * sc)) continue;
        const float f = 0.5f + 0.5f * std::sin(t * 3.0f + ph);
        const float r = (0.8f + 1.6f * hash01(i, 35)) * sc;
        glow_at(dl, glow, q, r * 3.5f, rgba(0.35f, 0.62f, 1.0f, 0.25f * f));
        dl->AddCircleFilled(q, r, rgba(0.30f, 0.55f, 0.95f, 0.25f + 0.45f * f), 8);
    }
}

} // namespace hud_theme
