#pragma once

#include "ucuda/domain/neural_dataset.h"
#include <filesystem>

// CSV of dense labelled samples. Invalid or unreadable files throw std::runtime_error.
NeuralDataset load_neural_dataset_csv(const std::filesystem::path& path);
