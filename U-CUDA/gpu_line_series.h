#pragma once
#include <glad/glad.h>
#include <vector>

// Одна GPU-серия: VBO с массивом точек float[2] (x, y).
struct GpuLineSeries {
    GLuint vbo = 0;
    int    point_count = 0;
    bool valid() const { return vbo != 0 && point_count > 0; }
};

// Кеш набора GPU-серий с CPU-копией bbox для быстрого autofit.
class GpuLineSeriesSet {
public:
    GpuLineSeriesSet() = default;
    ~GpuLineSeriesSet() { clear(); }
    GpuLineSeriesSet(const GpuLineSeriesSet&) = delete;
    GpuLineSeriesSet& operator=(const GpuLineSeriesSet&) = delete;

    // Залить серию из готового массива float[2] (x,y - x,y - ...).
    // points - указатель на n_points * 2 float'ов.
    // Возвращает индекс созданной серии.
    int upload(const float* points, int n_points);

    // Удобная перегрузка для std::vector<ImVec2>-подобного источника.
    int upload(const std::vector<float>& xy_pairs) {
        return upload(xy_pairs.data(), (int)(xy_pairs.size() / 2));
    }

    const GpuLineSeries& get(int index) const;
    int size() const { return (int)series_.size(); }
    void clear();

    // Общий bbox по всем сериям. Возвращает false если пусто.
    bool bbox(float& xmin, float& xmax, float& ymin, float& ymax) const;

private:
    std::vector<GpuLineSeries> series_;
    std::vector<float>         bbox_data_; // [k*4 + 0..3]
};