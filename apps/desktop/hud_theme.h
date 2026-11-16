#pragma once
// Тема Cyan HUD: вместо текстуры камуфляжа под окнами рисуется анимированное
// сотовое «силовое поле» (энергетические ядра, ударные волны, развёртка
// сканера, вспышки сот, энергоканал между ядрами), поверх окон — HUD-рамки
// (срезанные углы, скобки, засечки, свечение активного окна) и строки
// развёртки ЭЛТ. Всё строится заново каждый кадр из ImDrawList-ов, которые
// вставляются в ImDrawData после ImGui::Render (как подкладка камуфляжа).

#include <memory>
#include <vector>

struct ImDrawData;
struct ImDrawList;
struct ImFont;
struct ImGuiStyle;

namespace hud_theme {

// Поправки стиля поверх палитры темы: нулевые скругления, рамки у полей.
void apply_style(ImGuiStyle& st);

class Painter {
public:
    Painter() = default;
    Painter(const Painter&) = delete;
    Painter& operator=(const Painter&) = delete;

    // Между NewFrame и Render: запекает глифы подписей нужного кегля, пока
    // атлас шрифтов можно менять.
    void prepare_frame(float scale);
    // После Render: вставляет подкладки под корневые окна, рамки после них и
    // строки развёртки поверх всего.
    // computing — идёт расчёт на GPU: кадров всего ~10 в секунду, поэтому
    // анимация плавно замедляется (мелкие шаги между кадрами не дёргаются), а
    // на долгом расчёте останавливается совсем; после расчёта разгоняется.
    // show_note — выводить шутку про CUDAynamics (только на вкладке Settings).
    void inject(ImDrawData* dd, float scale, double time, bool computing, bool show_note);
    // До ImGui::DestroyContext (деструктор ImDrawList читает данные контекста).
    void release();

private:
    unsigned glow_tex_ = 0;   // GLuint радиального спрайта сияния
    bool     glow_tried_ = false;
    ImFont*  font_ = nullptr;
    float    note_size_ = 0.0f;
    std::vector<std::unique_ptr<ImDrawList>> pool_;
    size_t   used_ = 0;
    double   last_time_ = -1.0;      // реальное время прошлого кадра
    double   anim_time_ = 0.0;       // время анимации (идёт со скоростью speed_)
    float    speed_ = 1.0f;
    double   compute_start_ = -1.0;  // начало текущего расчёта, -1 — расчёта нет
    ImDrawList* next_list(ImDrawData* dd);
};

} // namespace hud_theme
