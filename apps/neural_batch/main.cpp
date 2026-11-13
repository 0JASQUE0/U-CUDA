#include "ucuda/analysis/neural_session.h"
#include "ucuda/application/system_editor.h"
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/platform/platform_paths.h"
#include "ucuda/storage/neural_dataset_io.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/storage/system_library.h"
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>

std::string exe_dir() { return platform::executable_dir(); }

namespace {
struct BatchSystemEditor : SystemEditor {
    BatchSystemEditor() : SystemEditor({}) {}
    using SystemEditor::build_system;
};

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input JSON");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int integer(const std::string& text) {
    int value = 0;
    const char* end = text.data() + text.size();
    const auto r = std::from_chars(text.data(), end, value);
    if (r.ec != std::errc{} || r.ptr != end || value < 0) throw std::runtime_error("expected a nonnegative integer");
    return value;
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 6) throw std::runtime_error(
            "usage: ucuda_neural_batch SYSTEM.json SESSION.json BATCH_SIZE OUTPUT.csv SEED [SEED ...]");
        const auto record = record_from_json(read_file(std::filesystem::u8path(argv[1])));
        const auto session_path = std::filesystem::u8path(argv[2]);
        NeuralState saved;
        if (!session_from_json_neural(read_file(session_path), saved) || saved.configs.empty())
            throw std::runtime_error("cannot load Neural session");
        const int batch = integer(argv[3]);
        if (batch == 0) throw std::runtime_error("batch size must be positive");
        std::vector<int> seeds;
        for (int arg = 5; arg < argc; ++arg) seeds.push_back(integer(argv[arg]));

        auto config = saved.configs.at((size_t)saved.active_config_index);
        if (config.task.dataset_path.empty()) throw std::runtime_error("the session must specify a dataset CSV");
        const auto dataset_path = session_path.parent_path() / std::filesystem::u8path(config.task.dataset_path);
        config.task.dataset = std::make_shared<NeuralDataset>(load_neural_dataset_csv(dataset_path));
        std::array<size_t, 3> counts{};
        for (const auto& sample : config.task.dataset->samples) ++counts[(size_t)sample.split];
        BatchSystemEditor editor;
        editor.from_record(record);
        const System sys = editor.build_system();
        ParametricEngine engine;
        const auto output_path = std::filesystem::u8path(argv[4]);
        if (std::filesystem::exists(output_path)) throw std::runtime_error("output CSV already exists");
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open output CSV");
        output << "seed,status,batch_size,train,validation,test,graph_hash,accuracy,raw_accuracy,lambda,extraction_ms,peak_device_bytes,confusion_row_major,validation_accuracy\n";
        output.flush();
        output << std::setprecision(17);
        bool failed = false;
        for (int seed : seeds) {
            config.seed_text = std::to_string(seed);
            try {
                const auto result = neural_classify_batches(engine, sys, record.custom_schemes, config, (size_t)batch);
                const auto& task = result.task;
                if (task.test.size() != 1 || !std::isfinite(task.test.front())) throw std::runtime_error("incomplete classification");
                output << seed << ",ok," << batch << ',' << counts[0] << ',' << counts[1] << ',' << counts[2]
                       << ',' << result.graph_hash << ',' << task.test.front() << ',';
                if (!task.delay_test.empty()) output << task.delay_test.front();
                output << ',';
                if (!task.lambda.empty()) output << task.lambda.front();
                output << ',' << result.extraction_ms << ',' << result.peak_device_bytes << ',';
                for (size_t i = 0; i < task.confusion.size(); ++i) output << (i ? " " : "") << task.confusion[i];
                output << ',';
                if (!task.val.empty()) output << task.val.front();
                output << '\n';
            } catch (const std::exception& e) {
                failed = true;
                std::cerr << "seed " << seed << ": " << e.what() << '\n';
                output << seed << ",failed," << batch << ',' << counts[0] << ',' << counts[1] << ',' << counts[2] << ",,,,,,,,\n";
            }
            output.flush();
            if (!output) throw std::runtime_error("failed to write batch results");
        }
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Neural batch: " << e.what() << '\n';
        return 1;
    }
}
