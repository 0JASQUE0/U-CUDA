#pragma once
// Вкладка Neural (большие сети): настройки и окна.
#include "gui_common.h"

namespace ucuda::ui_detail {
// Neural — большие сети: окно настроек и кривые (means: средние по популяциям, иначе выбранные узлы).
void draw_neural_controls(AppModel& model, SystemLibrary& lib);
enum { kNeuralMeans = 0, kNeuralNodes, kNeuralSync, kNeuralSweep, kNeuralInput, kNeuralLyap, kNeuralRate, kNeuralRaster, kNeuralWeights, kNeuralReadout, kNeuralWindowCount };
void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, int what);
// Карта свипа по двум осям; клик по ячейке выбирает сеть в окнах от времени.
void draw_neural_map_window(AppModel& model, PlotRenderer& renderer, std::map<int, std::unique_ptr<HeatmapView>>& maps);
}
