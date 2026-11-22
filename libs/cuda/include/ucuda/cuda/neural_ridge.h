#pragma once

#include "ucuda/numerics/neural_readout.h"

// Inputs use the same row-major layout and split as readout_fit.
ReadoutModel neural_ridge_gpu(const std::vector<double>& features, const std::vector<double>& targets,
                            int columns, int classes, const ReadoutSplit& split,
                            const std::vector<double>& lambdas, const std::atomic<bool>* cancel = nullptr);
