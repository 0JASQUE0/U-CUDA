#include "hud_sith.h"
#include "hud_scene.h"

namespace hud_theme {

namespace {

using namespace scene;

// Палитра сцены: чистый красный без синего канала (с синим он уходил в
// розовый), раскалённое оранжево-красное ядро, огонь искр.
inline ImU32 red(float a)  { return rgba(0.88f, 0.00f, 0.00f, a); }
inline ImU32 hot(float a)  { return rgba(1.00f, 0.44f, 0.31f, a); }
inline ImU32 fire(float a) { return rgba(1.00f, 0.16f, 0.00f, a); }

constexpr Tint kRedTint   { {0.88f, 0.00f, 0.00f}, {1.00f, 0.44f, 0.31f}, {1.00f, 0.16f, 0.00f} };
constexpr Tint kGreenTint { {0.08f, 0.92f, 0.12f}, {0.72f, 1.00f, 0.66f}, {0.30f, 1.00f, 0.20f} };
constexpr Tint kBlueTint  { {0.30f, 0.62f, 1.00f}, {0.88f, 0.96f, 1.00f}, {0.50f, 0.80f, 1.00f} };

// По умолчанию лучи этой сцены красные.
void beam(ImDrawList* dl, const ImVec2* p, int n, float w, float a, float core_w) {
    scene::beam(dl, p, n, w, a, core_w, kRedTint);
}
void beam(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float w, float a, float core_w) {
    scene::beam(dl, p0, p1, w, a, core_w, kRedTint);
}
using scene::beam;

// Молния: середины отрезков сдвигаются на ±25 % длины, 6 уровней — 65 точек.
constexpr int kBoltPts = 65;
void make_bolt(Rng& r, ImVec2 p0, ImVec2 p1, ImVec2* out) {
    out[0] = p0;
    out[kBoltPts - 1] = p1;
    for (int step = kBoltPts - 1; step > 1; step /= 2)
        for (int i = 0; i + step < kBoltPts; i += step) {
            const ImVec2 u = out[i], v = out[i + step];
            const float L = std::hypot(v.x - u.x, v.y - u.y) * 0.25f;
            out[i + step / 2] = ImVec2((u.x + v.x) * 0.5f + (r.next() - 0.5f) * L,
                                       (u.y + v.y) * 0.5f + (r.next() - 0.5f) * L);
        }
}

// Крейсеры: дальний (меньше, медленнее) и ближний. Летят справа налево,
// x в долях ширины от 1.3 до -0.5; на каждом проходе новая высота.
struct ShipDef { float s, v, off; uint32_t seed; };
const ShipDef kShips[2] = { { 0.6f, 0.012f, 0.50f, 11 }, { 1.0f, 0.018f, 0.25f, 23 } };

struct ShipPose { ImVec2 nose; float L, h; };
ShipPose ship_pose(const ShipDef& d, float t, ImVec2 P, ImVec2 S) {
    const float u = t * d.v / 1.8f + d.off;
    const float x = 1.3f - fract(u) * 1.8f;
    const float y = 0.28f + 0.42f * hash01(d.seed, (uint32_t)(int)std::floor(u));
    const float L = std::min(S.x * 0.28f, S.y * 0.60f) * d.s;
    return { ImVec2(P.x + S.x * x, P.y + S.y * y), L, L * 0.32f };
}

void draw_ship(ImDrawList* dl, ImTextureRef glow, const ShipPose& p, float s, float sc, float t, bool gray) {
    const float x = p.nose.x, y = p.nose.y, L = p.L, h = p.h;
    const ImVec2 tail_up(x + L, y - h * 0.5f), tail_dn(x + L, y + h * 0.5f);
    // gray — стальной корпус (Dark Side II), иначе тёмно-багровый.
    dl->AddTriangleFilled(p.nose, tail_up, tail_dn, gray ? rgba(0.22f, 0.23f, 0.25f, 1.0f) : rgba(0.11f, 0.05f, 0.043f, 1.0f));
    dl->AddTriangle(p.nose, tail_up, tail_dn, gray ? rgba(0.66f, 0.68f, 0.72f, 0.85f) : red(0.70f), 1.3f * sc);
    dl->AddLine(p.nose, ImVec2(x + L, y), gray ? rgba(0.56f, 0.58f, 0.62f, 0.75f) : rgba(0.50f, 0.28f, 0.24f, 0.70f), 1.0f * sc);
    // Надстройка с мостиком у кормы.
    const ImU32 deck = gray ? rgba(0.34f, 0.35f, 0.38f, 1.0f) : rgba(0.17f, 0.085f, 0.07f, 1.0f);
    dl->AddRectFilled(ImVec2(x + L * 0.70f, y - h * 0.42f), ImVec2(x + L * 0.84f, y - h * 0.22f), deck);
    dl->AddRectFilled(ImVec2(x + L * 0.74f, y - h * 0.55f), ImVec2(x + L * 0.81f, y - h * 0.41f), deck);
    // Дюзы.
    for (int i = -1; i <= 1; ++i) {
        const ImVec2 e(x + L + 2.0f * sc, y + i * h * 0.25f);
        const float r = (2.5f + std::sin(t * 20.0f + i)) * sc * s;
        glow_at(dl, glow, e, r * 6.0f, fire(0.55f));
        dl->AddCircleFilled(e, r, hot(0.95f));
    }
    // Огни по корпусу.
    for (int i = 0; i < 14; ++i) {
        const float u = 0.25f + i * 0.05f;
        const ImVec2 q(x + L * u, y - h * 0.5f * u + (i % 3) * 2.0f * sc);
        dl->AddRectFilled(q, ImVec2(q.x + sc, q.y + sc), rgba(1.0f, 0.24f, 0.12f, 0.70f));
    }
}

} // namespace

void paint_sith_scene(ImDrawList* dl, const ImGuiViewport* vp, ImVec2 a, ImVec2 b,
                      float sc, float t, ImTextureRef glow, bool alt_energy) {
    const ImVec2 P = vp->Pos, S = vp->Size;
    // Вариант II: зелёный главный калибр станции и синие молнии.
    const Tint& laser = alt_energy ? kGreenTint : kRedTint;
    const Tint& bolt  = alt_energy ? kBlueTint : kRedTint;
    if (S.x <= 0.0f || S.y <= 0.0f) return;

    // Звёзды: тёплые, мерцают вразнобой.
    for (uint32_t i = 0; i < 220; ++i) {
        const ImVec2 q(P.x + hash01(i, 1) * S.x, P.y + hash01(i, 2) * S.y);
        if (!inside(q, a, b, 2.0f)) continue;
        const float f = 0.5f + 0.5f * std::sin(t * 2.0f + hash01(i, 3) * 2.0f * kPi);
        const float s = (0.6f + 1.3f * hash01(i, 4)) * sc;
        dl->AddRectFilled(q, ImVec2(q.x + s, q.y + s), rgba(1.0f, 0.86f + 0.08f * f, 0.80f + 0.08f * f, 0.40f + 0.6f * f));
    }

    // Багровая виньетка, «дышит» с периодом ~5 с: четыре полосы по краям с
    // градиентом от края к центру.
    {
        const float p = 0.5f + 0.5f * std::sin(t * 1.2f);
        const ImU32 edge = rgba(0.50f + 0.35f * p, 0.0f, 0.0f, 0.50f), none = rgba(0.45f, 0.0f, 0.0f, 0.0f);
        const float bx = S.x * 0.22f, by = S.y * 0.25f;
        const ImVec2 Q(P.x + S.x, P.y + S.y);
        dl->AddRectFilledMultiColor(P, ImVec2(P.x + bx, Q.y), edge, none, none, edge);
        dl->AddRectFilledMultiColor(ImVec2(Q.x - bx, P.y), Q, none, edge, edge, none);
        dl->AddRectFilledMultiColor(P, ImVec2(Q.x, P.y + by), edge, edge, none, none);
        dl->AddRectFilledMultiColor(ImVec2(P.x, Q.y - by), Q, none, none, edge, edge);
    }

    // Тлеющие искры поднимаются вверх, как пепел.
    for (uint32_t i = 0; i < 70; ++i) {
        const float v = 0.02f + 0.05f * hash01(i, 11);
        const float ph = hash01(i, 12) * 2.0f * kPi;
        const ImVec2 q(P.x + hash01(i, 13) * S.x + std::sin(t + ph) * 8.0f * sc,
                       P.y + S.y * (1.0f - fract(hash01(i, 14) + v * t)));
        if (!inside(q, a, b, 12.0f * sc)) continue;
        const float f = 0.5f + 0.5f * std::sin(t * 4.0f + ph);
        const float r = (0.6f + 2.0f * hash01(i, 15)) * sc;
        if (r > 1.6f * sc) glow_at(dl, glow, q, r * 4.0f, fire(0.18f * f));
        dl->AddCircleFilled(q, r, rgba(1.0f, 0.08f + 0.24f * f, 0.0f, 0.30f + 0.6f * f), 8);
    }

    // Боевая станция: тёмная сфера с подсветкой по краю, экваториальной
    // бороздой и кратером главного калибра. Цикл 8 с: заряд 4–6 с, выстрел 6–7 с.
    {
        const ImVec2 c(P.x + S.x * 0.60f, P.y + S.y * 0.24f);
        const float r = S.y * 0.15f;
        const bool gray = alt_energy;   // Dark Side II: серая сталь вместо багрового
        glow_at(dl, glow, c, r * 1.6f, gray ? rgba(0.60f, 0.62f, 0.66f, 0.12f) : red(0.20f));
        dl->AddCircleFilled(c, r, gray ? rgba(0.11f, 0.115f, 0.125f, 1.0f) : rgba(0.040f, 0.014f, 0.012f, 1.0f), 96);
        // Объём: светлые круги, смещённые к верхнему левому краю (свет оттуда).
        for (int k = 1; k <= 5; ++k) {
            const float o = 0.10f * r * k / 1.4142f;
            dl->AddCircleFilled(ImVec2(c.x - o, c.y - o), r * (1.0f - 0.15f * k), gray ? rgba(0.52f, 0.54f, 0.58f, 0.20f) : rgba(0.30f, 0.17f, 0.15f, 0.22f), 72);
        }
        dl->AddCircle(c, r, gray ? rgba(0.72f, 0.74f, 0.78f, 0.70f) : red(0.65f), 96, 1.5f * sc);
        for (int i = -3; i <= 3; ++i) {
            if (i == 0) continue;
            const float yy = i * 0.25f;
            dl->AddEllipse(ImVec2(c.x, c.y + yy * r), ImVec2(std::sqrt(1.0f - yy * yy) * r, r * 0.05f),
                           gray ? rgba(0.46f, 0.47f, 0.51f, 0.45f) : rgba(0.55f, 0.28f, 0.22f, 0.40f), 0.0f, 64, 0.7f * sc);
        }
        dl->AddEllipse(c, ImVec2(r, r * 0.12f), gray ? rgba(0.04f, 0.04f, 0.045f, 1.0f) : rgba(0.01f, 0.0f, 0.0f, 1.0f), 0.0f, 96, 2.0f * sc);

        const float T = std::fmod(t, 8.0f);
        const ImVec2 d(c.x - r * 0.42f, c.y - r * 0.40f);
        const float dr = r * 0.24f;
        dl->AddCircleFilled(d, dr, gray ? rgba(0.07f, 0.075f, 0.08f, 1.0f) : rgba(0.04f, 0.016f, 0.016f, 1.0f), 48);
        dl->AddCircle(d, dr, gray ? rgba(0.32f, 0.33f, 0.36f, 1.0f) : rgba(0.16f, 0.07f, 0.06f, 1.0f), 48, 1.0f * sc);
        const float ch = (T > 4.0f && T < 6.0f) ? (T - 4.0f) / 2.0f : 0.0f;
        const bool shoot = T >= 6.0f && T < 7.0f;
        const ImVec2 focus(d.x - dr * 1.4f, d.y + dr * 0.9f);
        if (ch > 0.0f || shoot) {
            const float k = shoot ? 1.0f : ch;
            glow_at(dl, glow, d, dr * (1.0f + 1.2f * k), laser.b(k));
            dl->AddCircleFilled(d, dr * (0.25f + 0.3f * k), shoot ? laser.c(1.0f) : laser.f(k), 32);
            for (int j = 0; j < 6; ++j) {
                const float an = j / 6.0f * 2.0f * kPi + t;
                beam(dl, ImVec2(d.x + std::cos(an) * dr * 0.95f, d.y + std::sin(an) * dr * 0.95f), focus,
                     1.4f * sc, k, 0.0f, laser);
            }
        }
        if (shoot) {
            const float k = T - 6.0f;
            const ImVec2 tgt(P.x + S.x * 0.12f, P.y + S.y * 0.95f);
            const float u = std::min(1.0f, k * 4.0f);
            const ImVec2 end(focus.x + (tgt.x - focus.x) * u, focus.y + (tgt.y - focus.y) * u);
            beam(dl, focus, end, 9.0f * sc, 0.9f, 2.5f * sc, laser);
            glow_at(dl, glow, focus, 30.0f * sc, laser.c(0.8f));
            if (k > 0.25f) {
                glow_at(dl, glow, tgt, (40.0f + 120.0f * k) * sc, laser.f(0.7f * (1.0f - k)));
                dl->AddRectFilled(a, b, laser.b(0.08f * (1.0f - k)));   // отсвет по всему экрану
            }
        }
    }

    // Крейсеры (дальний первым) и их красные залпы.
    for (const ShipDef& d : kShips) {
        const ShipPose pose = ship_pose(d, t, P, S);
        if (pose.nose.x < P.x + S.x + 40.0f * sc && pose.nose.x + pose.L > P.x - 40.0f * sc)
            draw_ship(dl, glow, pose, d.s, sc, t, alt_energy);

        // Выстрел k — в момент tf, из точки корпуса корабля в тот момент; живёт 2 с.
        const float every = 0.55f, life = 2.0f;
        const int k1 = (int)std::floor(t / every), k0 = (int)std::floor((t - life) / every) - 1;
        for (int k = k0; k <= k1; ++k) {
            const uint32_t key = (uint32_t)k * 7919U + d.seed;
            const float tf = (k + 0.6f * hash01(key, 1)) * every;
            const float age = t - tf;
            if (age < 0.0f || age > life) continue;
            const ShipPose sp = ship_pose(d, tf, P, S);
            const ImVec2 o(sp.nose.x + sp.L * (0.2f + 0.4f * hash01(key, 2)),
                           sp.nose.y + (hash01(key, 3) - 0.5f) * sp.h * 0.4f);
            const float an = kPi + (hash01(key, 4) - 0.5f) * 0.9f;
            const ImVec2 dir(std::cos(an), std::sin(an));
            const float spd = 0.6f * S.x;
            // Часть выстрелов взрывается в пути.
            const bool boom = hash01(key, 5) < 0.3f;
            const float te = 0.4f + 1.2f * hash01(key, 6);
            if (boom && age > te) {
                const float g = (age - te) / 0.3f;
                if (g >= 1.0f) continue;
                const ImVec2 q(o.x + dir.x * spd * te, o.y + dir.y * spd * te);
                glow_at(dl, glow, q, (4.0f + 18.0f * g) * sc, fire(1.0f - g));
                continue;
            }
            const ImVec2 q(o.x + dir.x * spd * age, o.y + dir.y * spd * age);
            if (!inside(q, a, b, 30.0f * sc)) continue;
            beam(dl, q, ImVec2(q.x - dir.x * 22.0f * sc, q.y - dir.y * 22.0f * sc), 2.2f * sc, 1.0f, 1.0f * sc);
        }
    }

    // Два скрещённых плазменных клинка: сходятся и расходятся, как ножницы;
    // в точке скрещивания — искры, в момент удара у концов — вспышка.
    {
        const float cx = P.x + S.x * 0.47f, cy = P.y + S.y * 0.72f, L = S.y * 0.42f;
        const float a1 = -1.0f + 0.45f * std::sin(t * 1.3f);
        const float a2 = -2.14f - 0.45f * std::sin(t * 1.3f + 0.3f);
        const ImVec2 h[2] = { ImVec2(cx - 0.35f * L, cy + 0.25f * L), ImVec2(cx + 0.35f * L, cy + 0.25f * L) };
        const ImVec2 dir[2] = { ImVec2(std::cos(a1), std::sin(a1)), ImVec2(std::cos(a2), std::sin(a2)) };
        const float fl = 0.88f + 0.12f * std::sin(t * 41.0f) * std::sin(t * 27.0f);
        for (int i = 0; i < 2; ++i) {
            const ImVec2 e(h[i].x + dir[i].x * L, h[i].y + dir[i].y * L);
            beam(dl, h[i], e, 6.0f * sc, fl, 2.6f * sc);
            // Рукоять: металл с тёмными кольцами, яркий излучатель.
            const ImVec2 back(h[i].x - dir[i].x * 30.0f * sc, h[i].y - dir[i].y * 30.0f * sc);
            dl->AddLine(h[i], back, rgba(0.35f, 0.32f, 0.31f, 1.0f), 5.0f * sc);
            for (int q = 1; q < 4; ++q) {
                const float o0 = q * 7.0f * sc, o1 = o0 + 2.0f * sc;
                dl->AddLine(ImVec2(h[i].x - dir[i].x * o0, h[i].y - dir[i].y * o0),
                            ImVec2(h[i].x - dir[i].x * o1, h[i].y - dir[i].y * o1),
                            rgba(0.10f, 0.08f, 0.08f, 1.0f), 6.0f * sc);
            }
            glow_at(dl, glow, h[i], 10.0f * sc, hot(0.6f * fl));
        }
        // Пересечение отрезков клинков.
        const ImVec2 r = ImVec2(dir[0].x * L, dir[0].y * L), s = ImVec2(dir[1].x * L, dir[1].y * L);
        const float den = r.x * s.y - r.y * s.x;
        if (std::fabs(den) > 1e-6f) {
            const ImVec2 qp(h[1].x - h[0].x, h[1].y - h[0].y);
            const float u = (qp.x * s.y - qp.y * s.x) / den, w = (qp.x * r.y - qp.y * r.x) / den;
            if (u >= 0.0f && u <= 1.0f && w >= 0.0f && w <= 1.0f) {
                const ImVec2 ip(h[0].x + r.x * u, h[0].y + r.y * u);
                const float hit = std::clamp((u - 0.6f) / 0.35f, 0.0f, 1.0f);   // удар у концов
                const float k = 0.35f + 0.65f * hit;
                Rng rng{ (uint32_t)(int)std::floor(t * 30.0f) * 2654435761U };
                for (int j = 0; j < 8; ++j) {
                    const float an = rng.next() * 2.0f * kPi, len = rng.next() * 30.0f * sc * k;
                    beam(dl, ip, ImVec2(ip.x + std::cos(an) * len, ip.y + std::sin(an) * len), 1.2f * sc, k, 0.8f * sc);
                }
                glow_at(dl, glow, ip, (14.0f + 30.0f * hit) * sc, hot(0.5f + 0.4f * hit));
            }
        }
    }

    // Разряды-молнии: раз в 2–5 с от края экрана, живут 0.45 с, дрожат.
    {
        const float period = 3.4f;
        const int n = (int)std::floor(t / period);
        for (int m = n - 1; m <= n; ++m) {
            const uint32_t key = (uint32_t)m * 104729U + 7U;
            const float t0 = m * period + hash01(key, 1) * (period - 0.5f);
            const float age = t - t0;
            if (age < 0.0f || age > 0.45f) continue;
            Rng rng{ key };
            const bool left = rng.next() < 0.5f;
            const ImVec2 p0(left ? P.x : P.x + S.x, P.y + rng.next() * S.y);
            const ImVec2 p1(P.x + S.x * (0.3f + 0.4f * rng.next()), P.y + rng.next() * S.y);
            ImVec2 main[kBoltPts], br[kBoltPts];
            make_bolt(rng, p0, p1, main);
            const ImVec2 bp = main[20];
            make_bolt(rng, bp, ImVec2(bp.x + (rng.next() - 0.5f) * S.x * 0.3f, bp.y + S.y * 0.3f), br);
            const float flick = hash01((uint32_t)(int)std::floor(t * 30.0f), key) > 0.25f ? 1.0f : 0.3f;
            const float al = (1.0f - age / 0.45f) * flick;
            beam(dl, main, kBoltPts, 3.0f * sc, al, 1.0f * sc, bolt);
            beam(dl, br, kBoltPts, 1.5f * sc, al * 0.8f, 0.0f, bolt);
        }
    }
}

} // namespace hud_theme
