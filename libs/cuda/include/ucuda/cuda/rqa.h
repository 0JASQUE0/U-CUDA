#pragma once
// RQA на GPU: расчёт (реализация — host_rqa.cu, NVCC). Типы Config / Result — в
// ucuda/domain/rqa.h; этот заголовок host-safe, его включает обычный cl.exe.
#include "ucuda/domain/rqa.h"
#include <vector>

namespace rqa {

// traj — [step][coord], как AnalysisResult::trajectories[ic], но ЖЕЛАТЕЛЬНО непрореженная:
// прореживание до Config::points делается здесь, и dt возвращается честный.
// dt_traj — шаг между соседними точками traj в секундах (h * decimator, если traj уже прорежена).
// Возвращает out.ok; при ошибке текст в out.error.
bool compute(const std::vector<std::vector<double>>& traj,
             double dt_traj,
             const Config& cfg,
             Result& out);

}  // namespace rqa
