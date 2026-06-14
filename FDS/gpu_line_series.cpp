#include "gpu_line_series.h"
#include <limits>
#include <algorithm>

int GpuLineSeriesSet::upload(const float* points, int n_points) {
    GpuLineSeries s;
    s.point_count = n_points;
    float xmin = std::numeric_limits<float>::infinity();
    float xmax = -std::numeric_limits<float>::infinity();
    float ymin = std::numeric_limits<float>::infinity();
    float ymax = -std::numeric_limits<float>::infinity();

    if (n_points > 0 && points) {
        for (int i = 0; i < n_points; ++i) {
            float x = points[i * 2 + 0];
            float y = points[i * 2 + 1];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
        }
        glGenBuffers(1, &s.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
        glBufferData(GL_ARRAY_BUFFER, n_points * 2 * sizeof(float),
            points, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    else {
        xmin = xmax = ymin = ymax = 0;
    }

    bbox_data_.push_back(xmin);
    bbox_data_.push_back(xmax);
    bbox_data_.push_back(ymin);
    bbox_data_.push_back(ymax);
    series_.push_back(s);
    return (int)series_.size() - 1;
}

const GpuLineSeries& GpuLineSeriesSet::get(int index) const {
    static const GpuLineSeries empty;
    if (index < 0 || index >= (int)series_.size()) return empty;
    return series_[index];
}

void GpuLineSeriesSet::clear() {
    for (auto& s : series_)
        if (s.vbo) glDeleteBuffers(1, &s.vbo);
    series_.clear();
    bbox_data_.clear();
}

bool GpuLineSeriesSet::bbox(float& xmin, float& xmax, float& ymin, float& ymax) const {
    if (series_.empty()) return false;
    xmin = std::numeric_limits<float>::infinity();
    xmax = -std::numeric_limits<float>::infinity();
    ymin = std::numeric_limits<float>::infinity();
    ymax = -std::numeric_limits<float>::infinity();
    bool any = false;
    for (size_t k = 0; k < series_.size(); ++k) {
        if (!series_[k].valid()) continue;
        any = true;
        xmin = std::min(xmin, bbox_data_[k * 4 + 0]);
        xmax = std::max(xmax, bbox_data_[k * 4 + 1]);
        ymin = std::min(ymin, bbox_data_[k * 4 + 2]);
        ymax = std::max(ymax, bbox_data_[k * 4 + 3]);
    }
    return any;
}