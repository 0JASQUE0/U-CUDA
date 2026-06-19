#pragma once
#include <glad/glad.h>
#include <vector>

// ���� GPU-�����: VBO � �������� ����� float[2] (x, y).
struct GpuLineSeries {
    GLuint vbo = 0;
    int    point_count = 0;
    bool valid() const { return vbo != 0 && point_count > 0; }
};

// ��� ������ GPU-����� � CPU-������ bbox ��� �������� autofit.
class GpuLineSeriesSet {
public:
    GpuLineSeriesSet() = default;
    ~GpuLineSeriesSet() { clear(); }
    GpuLineSeriesSet(const GpuLineSeriesSet&) = delete;
    GpuLineSeriesSet& operator=(const GpuLineSeriesSet&) = delete;

    // ������ ����� �� �������� ������� float[2] (x,y - x,y - ...).
    // points - ��������� �� n_points * 2 float'��.
    // ���������� ������ ��������� �����.
    int upload(const float* points, int n_points);

    // ������� ���������� ��� std::vector<ImVec2>-��������� ���������.
    int upload(const std::vector<float>& xy_pairs) {
        return upload(xy_pairs.data(), (int)(xy_pairs.size() / 2));
    }

    const GpuLineSeries& get(int index) const;
    int size() const { return (int)series_.size(); }
    void clear();

    // ����� bbox �� ���� ������. ���������� false ���� �����.
    bool bbox(float& xmin, float& xmax, float& ymin, float& ymax) const;

    // То же, но учитывает только серии, для которых is_visible(k) == true.
    // Возвращает false, если ни одной видимой серии не нашлось.
    bool bbox_filtered(float& xmin, float& xmax, float& ymin, float& ymax,
                       const std::vector<bool>& visible_mask) const;

private:
    std::vector<GpuLineSeries> series_;
    std::vector<float>         bbox_data_; // [k*4 + 0..3]
};