#pragma once
// Вкладка FastSync: настройки и график.
#include "gui_common.h"

namespace ucuda::ui_detail {
void draw_fastsync_controls(AppModel& model, SystemLibrary& lib);


void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb);
}
