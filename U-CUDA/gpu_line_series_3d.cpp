#include "gpu_line_series_3d.h"
#include <limits>
#include <algorithm>

int GpuLineSeriesSet3D::upload(const float* points, int n_points) {
    GpuLineSeries3D s;
    s.point_count = n_points;
    float xmin = std::numeric_limits<float>::infinity();
    float xmax = -std::numeric_limits<float>::infinity();
    float ymin = std::numeric_limits<float>::infinity();
    float ymax = -std::numeric_limits<float>::infinity();
    float zmin = std::numeric_limits<float>::infinity();
    float zmax = -std::numeric_limits<float>::infinity();

    if (n_points > 0 && points) {
        for (int i = 0; i < n_points; ++i) {
            float x = points[i * 3 + 0];
            float y = points[i * 3 + 1];
            float z = points[i * 3 + 2];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        glGenBuffers(1, &s.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
        glBufferData(GL_ARRAY_BUFFER, n_points * 3 * sizeof(float),
            points, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    else {
        xmin = xmax = ymin = ymax = zmin = zmax = 0;
    }

    bbox_data_.push_back(xmin); bbox_data_.push_back(xmax);
    bbox_data_.push_back(ymin); bbox_data_.push_back(ymax);
    bbox_data_.push_back(zmin); bbox_data_.push_back(zmax);
    series_.push_back(s);
    return (int)series_.size() - 1;
}

const GpuLineSeries3D& GpuLineSeriesSet3D::get(int index) const {
    static const GpuLineSeries3D empty;
    if (index < 0 || index >= (int)series_.size()) return empty;
    return series_[index];
}

void GpuLineSeriesSet3D::clear() {
    for (auto& s : series_)
        if (s.vbo) glDeleteBuffers(1, &s.vbo);
    series_.clear();
    bbox_data_.clear();
}

bool GpuLineSeriesSet3D::bbox(float& xmin, float& xmax, float& ymin, float& ymax,
    float& zmin, float& zmax) const {
    if (series_.empty()) return false;
    xmin = std::numeric_limits<float>::infinity();
    xmax = -std::numeric_limits<float>::infinity();
    ymin = std::numeric_limits<float>::infinity();
    ymax = -std::numeric_limits<float>::infinity();
    zmin = std::numeric_limits<float>::infinity();
    zmax = -std::numeric_limits<float>::infinity();
    bool any = false;
    for (size_t k = 0; k < series_.size(); ++k) {
        if (!series_[k].valid()) continue;
        any = true;
        xmin = std::min(xmin, bbox_data_[k * 6 + 0]);
        xmax = std::max(xmax, bbox_data_[k * 6 + 1]);
        ymin = std::min(ymin, bbox_data_[k * 6 + 2]);
        ymax = std::max(ymax, bbox_data_[k * 6 + 3]);
        zmin = std::min(zmin, bbox_data_[k * 6 + 4]);
        zmax = std::max(zmax, bbox_data_[k * 6 + 5]);
    }
    return any;
}