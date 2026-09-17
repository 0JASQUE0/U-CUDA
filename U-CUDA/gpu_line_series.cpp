#include "gpu_line_series.h"
#include <limits>
#include <vector>
#include <algorithm>

int GpuLineSeriesSet::upload(const double* points, int n_points,
                             double origin_x, double origin_y) {
    GpuLineSeries s;
    s.point_count = n_points;
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();

    if (n_points > 0 && points) {
        // bbox — по АБСОЛЮТНЫМ значениям (его читает autofit), а в VBO уходят
        // смещённые: вычитание в double, приведение к float уже после него.
        std::vector<float> shifted((size_t)n_points * 2);
        for (int i = 0; i < n_points; ++i) {
            const double x = points[i * 2 + 0];
            const double y = points[i * 2 + 1];
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            shifted[(size_t)i * 2 + 0] = (float)(x - origin_x);
            shifted[(size_t)i * 2 + 1] = (float)(y - origin_y);
        }
        glGenBuffers(1, &s.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
        glBufferData(GL_ARRAY_BUFFER, n_points * 2 * sizeof(float),
            shifted.data(), GL_STATIC_DRAW);
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

bool GpuLineSeriesSet::bbox(double& xmin, double& xmax, double& ymin, double& ymax) const {
    if (series_.empty()) return false;
    xmin = std::numeric_limits<double>::infinity();
    xmax = -std::numeric_limits<double>::infinity();
    ymin = std::numeric_limits<double>::infinity();
    ymax = -std::numeric_limits<double>::infinity();
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

bool GpuLineSeriesSet::bbox_filtered(double& xmin, double& xmax, double& ymin, double& ymax,
                                     const std::vector<bool>& visible_mask) const {
    if (series_.empty()) return false;
    xmin = std::numeric_limits<double>::infinity();
    xmax = -std::numeric_limits<double>::infinity();
    ymin = std::numeric_limits<double>::infinity();
    ymax = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (size_t k = 0; k < series_.size(); ++k) {
        if (!series_[k].valid()) continue;
        // Если для серии нет записи в маске — считаем видимой (back-compat).
        bool vis = (k < visible_mask.size()) ? visible_mask[k] : true;
        if (!vis) continue;
        any = true;
        xmin = std::min(xmin, bbox_data_[k * 4 + 0]);
        xmax = std::max(xmax, bbox_data_[k * 4 + 1]);
        ymin = std::min(ymin, bbox_data_[k * 4 + 2]);
        ymax = std::max(ymax, bbox_data_[k * 4 + 3]);
    }
    return any;
}