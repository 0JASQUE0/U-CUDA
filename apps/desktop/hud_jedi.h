#pragma once
// Сцена темы Light Side под окнами: светлое небо с солнцем и лучами, облака,
// окольцованная планета, звенья крестокрылых истребителей с голубыми залпами,
// синий световой клинок и пылинки. Чистая функция времени t, как сцена Dark Side.

#include "imgui.h"

namespace hud_theme {

// vp — главный вьюпорт (координаты сцены), [a, b] — прямоугольник окна, sc —
// масштаб интерфейса, glow — радиальный спрайт сияния (белый, цвет вершинами).
void paint_jedi_scene(ImDrawList* dl, const ImGuiViewport* vp, ImVec2 a, ImVec2 b,
                      float sc, float t, ImTextureRef glow);

} // namespace hud_theme
