#pragma once
// Вкладка Neural (большие сети): настройки и окна.
#include "gui_common.h"

namespace ucuda::ui_detail {
// Neural — большие сети: окно настроек и кривые (means: средние по популяциям, иначе выбранные узлы).
void draw_neural_controls(AppModel& model, SystemLibrary& lib);
void draw_neural_results(AppModel& model);
}
