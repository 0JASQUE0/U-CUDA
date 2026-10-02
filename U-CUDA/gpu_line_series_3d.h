#pragma once
#include <glad/glad.h>
#include <vector>

struct GpuLineSeries3D {
    GLuint vbo = 0;
    GLuint vbo_val = 0;   // float на вершину для колормапы (0 — серия одного цвета)
    int    point_count = 0;
    bool valid() const { return vbo != 0 && point_count > 0; }
};

class GpuLineSeriesSet3D {
public:
    GpuLineSeriesSet3D() = default;
    ~GpuLineSeriesSet3D() { clear(); }
    GpuLineSeriesSet3D(const GpuLineSeriesSet3D&) = delete;
    GpuLineSeriesSet3D& operator=(const GpuLineSeriesSet3D&) = delete;

    // points - указатель на n_points * 3 float (x, y, z подряд).
    // values - n_points float для окраски по колормапе (nullptr — без неё).
    int upload(const float* points, int n_points, const float* values = nullptr);

    const GpuLineSeries3D& get(int index) const;
    int size() const { return (int)series_.size(); }
    void clear();

    // Общий bbox по всем сериям (xmin, xmax, ymin, ymax, zmin, zmax).
    bool bbox(float& xmin, float& xmax, float& ymin, float& ymax,
        float& zmin, float& zmax) const;

private:
    std::vector<GpuLineSeries3D> series_;
    std::vector<float>           bbox_data_; // [k*6 + 0..5] = xmin,xmax,ymin,ymax,zmin,zmax
};