#pragma once
#include <glad/glad.h>
#include <vector>

// Одна GPU-серия: VBO с вершинами вида float[2] (x, y).
//
// Почему наружу double, а внутрь float. Контекст приложения — OpenGL 3.3 core, а
// 64-битные вершинные атрибуты (dvec2 / glVertexAttribLPointer) появились только в
// 4.1; glVertexAttribPointer с GL_DOUBLE в 3.3 молча конвертирует значение во float
// на входе. То есть в шейдер double не передать в принципе, и приведение неизбежно.
//
// Но раньше оно стояло НЕ здесь, а в три десятка мест упаковки точек, то есть
// задолго до GL. При сильном приближении это убивало разрешение: для окна
// a in [0.154548376405 .. 0.154548935927] шаг сетки 5.6e-10 в 27 раз меньше ULP
// float на этом модуле (1.5e-8), и 1000 точек свипа схлопывались в 39 различных.
//
// Теперь приведение делается ЗДЕСЬ и только здесь, причём после вычитания начала
// координат В DOUBLE: координаты становятся малыми по модулю, и float их разрешает
// (те же 1000 точек дают 1000 различных значений). Начало координат общее на весь
// набор и должно совпадать с тем, в котором вызывающий строит матрицу проекции.
struct GpuLineSeries {
    GLuint vbo = 0;
    int    point_count = 0;
    bool valid() const { return vbo != 0 && point_count > 0; }
};

// Набор GPU-серий с CPU-копией bbox — чтобы autofit не читал обратно из GL.
class GpuLineSeriesSet {
public:
    GpuLineSeriesSet() = default;
    ~GpuLineSeriesSet() { clear(); }
    GpuLineSeriesSet(const GpuLineSeriesSet&) = delete;
    GpuLineSeriesSet& operator=(const GpuLineSeriesSet&) = delete;

    // Заливает серию из плоского массива double[2] (x,y - x,y - ...).
    // points - указатель на n_points * 2 double'ов, координаты АБСОЛЮТНЫЕ.
    // origin_x/origin_y вычитаются перед приведением к float (см. шапку файла);
    // bbox при этом остаётся в абсолютных координатах, чтобы autofit не съезжал.
    // Возвращает индекс созданной серии.
    int upload(const double* points, int n_points, double origin_x, double origin_y);

    // Перегрузка для уже собранного вектора пар.
    int upload(const std::vector<double>& xy_pairs, double origin_x, double origin_y) {
        return upload(xy_pairs.data(), (int)(xy_pairs.size() / 2), origin_x, origin_y);
    }

    const GpuLineSeries& get(int index) const;
    int size() const { return (int)series_.size(); }
    void clear();

    // Общий bbox по всем сериям, в АБСОЛЮТНЫХ координатах. false, если серий нет.
    bool bbox(double& xmin, double& xmax, double& ymin, double& ymax) const;

    // То же, но учитывает только серии, для которых is_visible(k) == true.
    // Возвращает false, если ни одной видимой серии не нашлось.
    bool bbox_filtered(double& xmin, double& xmax, double& ymin, double& ymax,
                       const std::vector<bool>& visible_mask) const;

private:
    std::vector<GpuLineSeries> series_;
    std::vector<double>        bbox_data_; // [k*4 + 0..3], абсолютные координаты
};