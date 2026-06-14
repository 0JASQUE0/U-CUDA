#include "plot_camera_3d.h"
#include <cmath>
#include <algorithm>
#include "imgui.h"

static void mat_identity(float m[16]) {
    for (int i = 0; i < 16; ++i) m[i] = 0;
    m[0] = m[5] = m[10] = m[15] = 1;
}

static void mat_mul(const float a[16], const float b[16], float out[16]) {
    float r[16];
    for (int c = 0; c < 4; ++c) {
        for (int r0 = 0; r0 < 4; ++r0) {
            r[c * 4 + r0] = a[0 * 4 + r0] * b[c * 4 + 0]
                + a[1 * 4 + r0] * b[c * 4 + 1]
                + a[2 * 4 + r0] * b[c * 4 + 2]
                + a[3 * 4 + r0] * b[c * 4 + 3];
        }
    }
    for (int i = 0; i < 16; ++i) out[i] = r[i];
}

void PlotCamera3D::build_mvp(float out[16]) const {
    // Позиция камеры (orbit вокруг target по углам theta, phi).
    float cx = target[0] + distance * std::cos(phi) * std::cos(theta);
    float cy = target[1] + distance * std::cos(phi) * std::sin(theta);
    float cz = target[2] + distance * std::sin(phi);

    // forward = target - cam
    float fx = target[0] - cx;
    float fy = target[1] - cy;
    float fz = target[2] - cz;
    float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (fl < 1e-6f) fl = 1.0f;
    fx /= fl; fy /= fl; fz /= fl;

    // up = Z вверх; right = forward x up
    float ux_temp = 0.0f, uy_temp = 0.0f, uz_temp = 1.0f;
    float rx = fy * uz_temp - fz * uy_temp;
    float ry = fz * ux_temp - fx * uz_temp;
    float rz = fx * uy_temp - fy * ux_temp;
    float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (rl < 1e-6f) { rx = 1; ry = 0; rz = 0; }
    else { rx /= rl; ry /= rl; rz /= rl; }
    // up = right x forward (ортонормируем)
    float upx = ry * fz - rz * fy;
    float upy = rz * fx - rx * fz;
    float upz = rx * fy - ry * fx;

    // View matrix (look-at), column-major
    float V[16];
    V[0] = rx;  V[1] = upx; V[2] = -fx; V[3] = 0;
    V[4] = ry;  V[5] = upy; V[6] = -fy; V[7] = 0;
    V[8] = rz;  V[9] = upz; V[10] = -fz; V[11] = 0;
    V[12] = -(rx * cx + ry * cy + rz * cz);
    V[13] = -(upx * cx + upy * cy + upz * cz);
    V[14] = (fx * cx + fy * cy + fz * cz);
    V[15] = 1;

    // Ортографическая проекция: размер окна пропорционален distance,
    // aspect компенсирует ширину/высоту.
    float h = distance * 0.6f;
    float w = h * aspect;
    float n = -distance * 10.0f;
    float f = distance * 10.0f;
    float P[16];
    for (int i = 0; i < 16; ++i) P[i] = 0;
    P[0] = 1.0f / w;
    P[5] = 1.0f / h;
    P[10] = -2.0f / (f - n);
    P[14] = -(f + n) / (f - n);
    P[15] = 1.0f;

    mat_mul(P, V, out);
}

void PlotCamera3D::orbit(float dx_pixels, float dy_pixels) {
    const float speed = 0.005f;
    theta -= dx_pixels * speed;
    phi += dy_pixels * speed;
    const float lim = 1.5533f; // ~89 deg
    if (phi > lim) phi = lim;
    if (phi < -lim) phi = -lim;
}

void PlotCamera3D::pan(float dx_pixels, float dy_pixels, int viewport_w, int viewport_h) {
    // Считаем "размер мира на пиксель" исходя из ortho-окна.
    float h = distance * 0.6f;
    float world_per_px_y = (2.0f * h) / std::max(1, viewport_h);
    float world_per_px_x = world_per_px_y * (float)viewport_w / std::max(1, viewport_h);

    // right и up в мировых координатах (повторяем расчёт из build_mvp).
    float cx = std::cos(phi) * std::cos(theta);
    float cy = std::cos(phi) * std::sin(theta);
    float cz = std::sin(phi);
    float fx = -cx, fy = -cy, fz = -cz;

    float ux_temp = 0.0f, uy_temp = 0.0f, uz_temp = 1.0f;
    float rx = fy * uz_temp - fz * uy_temp;
    float ry = fz * ux_temp - fx * uz_temp;
    float rz = fx * uy_temp - fy * ux_temp;
    float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (rl < 1e-6f) { rx = 1; ry = 0; rz = 0; }
    else { rx /= rl; ry /= rl; rz /= rl; }
    float upx = ry * fz - rz * fy;
    float upy = rz * fx - rx * fz;
    float upz = rx * fy - ry * fx;

    target[0] -= rx * dx_pixels * world_per_px_x - upx * dy_pixels * world_per_px_y;
    target[1] -= ry * dx_pixels * world_per_px_x - upy * dy_pixels * world_per_px_y;
    target[2] -= rz * dx_pixels * world_per_px_x - upz * dy_pixels * world_per_px_y;
}

void PlotCamera3D::zoom(float factor) {
    distance *= factor;
    if (distance < 0.001f) distance = 0.001f;
    if (distance > 1e9f)   distance = 1e9f;
}

void PlotCamera3D::fit_to_bbox(float xmin, float xmax, float ymin, float ymax,
    float zmin, float zmax) {
    target[0] = 0.5f * (xmin + xmax);
    target[1] = 0.5f * (ymin + ymax);
    target[2] = 0.5f * (zmin + zmax);

    float dx = xmax - xmin;
    float dy = ymax - ymin;
    float dz = zmax - zmin;
    float diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (diag < 1e-6f) diag = 1.0f;
    distance = diag * 1.2f;
}

ImVec2 project_to_screen(const float mvp[16], float x, float y, float z,
    ImVec2 plot_pos, float plot_w, float plot_h,
    float* out_z_clip)
{
    // clip = mvp * (x,y,z,1)
    float cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    float cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    float cz = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
    float cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (std::abs(cw) < 1e-12f) cw = 1.0f;
    float nx = cx / cw; // NDC: -1..1
    float ny = cy / cw;
    float nz = cz / cw;
    if (out_z_clip) *out_z_clip = nz;
    // в пиксели: NDC (-1..1) X?(plot_x .. plot_x+plot_w), Y перевернут
    float sx = plot_pos.x + (nx * 0.5f + 0.5f) * plot_w;
    float sy = plot_pos.y + (1.0f - (ny * 0.5f + 0.5f)) * plot_h;
    return ImVec2(sx, sy);
}