#pragma once

#include <string>
#include <vector>

enum class NeuralDatasetSplit { Train, Validation, Test };

struct NeuralSample {
    std::string id;
    std::string label;
    std::string group;
    NeuralDatasetSplit split = NeuralDatasetSplit::Train;
};

struct NeuralDataset {
    std::vector<std::string> channels;
    std::vector<NeuralSample> samples;
    std::vector<double> values;  // [sample][channel], in input file order
};
