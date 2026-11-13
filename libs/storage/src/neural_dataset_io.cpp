#include "ucuda/storage/neural_dataset_io.h"
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace {
constexpr size_t max_line_bytes = 1024 * 1024;
constexpr size_t max_values = 64 * 1024 * 1024;
constexpr size_t max_samples = 1000000;

[[noreturn]] void invalid(size_t line, const char* reason) {
    throw std::runtime_error("Neural dataset CSV, line " + std::to_string(line) + ": " + reason);
}

std::vector<std::string_view> fields(std::string_view line, size_t number) {
    if (line.find('"') != std::string_view::npos || line.find('\0') != std::string_view::npos)
        invalid(number, "quoted fields and NUL bytes are unsupported");
    std::vector<std::string_view> result;
    for (;;) {
        const size_t end = line.find(',');
        auto value = line.substr(0, end);
        const size_t first = value.find_first_not_of(" \t");
        if (first == std::string_view::npos) value = {};
        else value = value.substr(first, value.find_last_not_of(" \t") - first + 1);
        result.push_back(value);
        if (end == std::string_view::npos) return result;
        line.remove_prefix(end + 1);
    }
}

bool read_line(std::istream& file, std::string& line, size_t number) {
    if (!std::getline(file, line)) {
        if (file.bad() || !file.eof()) invalid(number, "read failed");
        return false;
    }
    if (line.size() > max_line_bytes) invalid(number, "row exceeds the size limit");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}
}

NeuralDataset load_neural_dataset_csv(const std::filesystem::path& path) {
    constexpr auto max_file_bytes = 512ull * 1024 * 1024;
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > max_file_bytes) throw std::runtime_error("Neural dataset CSV is unreadable or exceeds 512 MiB");
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open Neural dataset CSV");

    NeuralDataset data;
    std::string line;
    if (!read_line(file, line, 1)) invalid(1, "missing header");
    std::string_view header = line;
    if (header.substr(0, 3) == "\xef\xbb\xbf") header.remove_prefix(3);
    const auto columns = fields(header, 1);
    if (columns.size() < 5 || columns[0] != "id" || columns[1] != "split" ||
        columns[2] != "label" || columns[3] != "group")
        invalid(1, "expected id,split,label,group followed by one or more channel names");
    std::unordered_set<std::string> channel_names;
    for (size_t col = 4; col < columns.size(); ++col) {
        if (columns[col].empty() || !channel_names.emplace(columns[col]).second)
            invalid(1, "channel names must be nonempty and unique");
        data.channels.emplace_back(columns[col]);
    }

    std::unordered_set<std::string> ids;
    std::unordered_map<std::string, NeuralDatasetSplit> groups;
    std::array<size_t, 3> split_counts{};
    for (size_t number = 2; read_line(file, line, number); ++number) {
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        const auto row = fields(line, number);
        if (row.size() != data.channels.size() + 4) invalid(number, "wrong number of columns");
        if (data.samples.size() == max_samples || data.values.size() > max_values - data.channels.size())
            invalid(number, "dataset exceeds the sample or value limit");
        if (row[0].empty() || !ids.emplace(row[0]).second) invalid(number, "sample ID must be nonempty and unique");
        if (row[2].empty()) invalid(number, "missing class label");
        NeuralDatasetSplit split;
        if (row[1] == "train") split = NeuralDatasetSplit::Train;
        else if (row[1] == "validation") split = NeuralDatasetSplit::Validation;
        else if (row[1] == "test") split = NeuralDatasetSplit::Test;
        else invalid(number, "split must be train, validation or test");
        ++split_counts[static_cast<size_t>(split)];
        if (!row[3].empty()) {
            const auto group = groups.try_emplace(std::string(row[3]), split);
            if (group.first->second != split) invalid(number, "related samples cross the split");
        }
        data.samples.push_back({std::string(row[0]), std::string(row[2]), std::string(row[3]), split});
        for (size_t col = 4; col < row.size(); ++col) {
            if (row[col].empty()) invalid(number, "missing channel value");
            const char* end = row[col].data() + row[col].size();
            double value = 0;
            const auto parsed = std::from_chars(row[col].data(), end, value);
            if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(value))
                invalid(number, "channel values must be finite decimal numbers");
            data.values.push_back(value);
        }
    }
    for (size_t count : split_counts)
        if (count == 0) invalid(1, "train, validation and test must each contain samples");
    return data;
}
