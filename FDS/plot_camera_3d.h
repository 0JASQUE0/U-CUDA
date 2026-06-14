#pragma once
#include "imgui.h"

// Орбитальная камера для 3D-вьюера.
// Параметризация: камера смотрит на target_, отдалена на distance_, углы:
//   theta - азимут (вращение вокруг Z), радианы.
//   phi   - элевация (от плоскости XY вверх), радианы. Ограничена (-pi/2+eps, pi/2-eps).
// Проекция - ортографическая (поле зрения - distance_ задаёт размер ortho-окна).
class PlotCamera3D {
public:
    // углы и расстояние
    float theta = 0.6f;   // ~35 градусов
    float phi = 0.45f;  // ~25 градусов
    float distance = 3.0f;
    float target[3] = { 0.0f, 0.0f, 0.0f };

    // Соотношение сторон viewport'а (width/height) - нужно для ortho.
    // Адаптер выставляет перед build_mvp.
    float aspect = 1.0f;

    // Собрать MVP-матрицу (column-major, GL).
    void build_mvp(float out[16]) const;

    // Орбитальное вращение от движения мыши в пикселях.
    void orbit(float dx_pixels, float dy_pixels);

    // Пан в плоскости экрана.
    void pan(float dx_pixels, float dy_pixels, int viewport_w, int viewport_h);

    // Зум (factor > 1 - отдалить, factor < 1 - приблизить).
    void zoom(float factor);

    // Установить начальный вид: камера смотрит на центр bbox, distance подобран
    // так, чтобы bbox целиком помещался.
    void fit_to_bbox(float xmin, float xmax, float ymin, float ymax,
        float zmin, float zmax);
};

// Проекция 3D-точки в экранные координаты, используя MVP-матрицу.
// plot_pos/plot_w/plot_h - экранная область, куда рисуется FBO-картинка.
// Возвращает позицию в пикселях ImGui. z_clip - глубина в clip-space (-1..1),
// > 1.0 значит точка за дальней плоскостью (можно не рисовать).
ImVec2 project_to_screen(const float mvp[16], float x, float y, float z,
    ImVec2 plot_pos, float plot_w, float plot_h,
    float* out_z_clip = nullptr);