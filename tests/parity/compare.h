#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

namespace parity {
using Json = nlohmann::json;
namespace fs = std::filesystem;
struct Comparison {
    bool exact = true;
    double absolute = 0, relative = 0;
    size_t numbers = 0, bit_differences = 0, failures = 0;
    double max_absolute = 0, max_relative = 0;
    Json details = Json::array();
    void fail(const std::string& path, const std::string& reason) {
        ++failures;
        if (details.size() < 16) details.push_back({{"path", path}, {"reason", reason}});
    }
    void visit(const Json& reference, const Json& actual, const std::string& path = "") {
        if (reference.type() != actual.type()) { fail(path, "JSON type changed"); return; }
        if (reference.is_object()) {
            for (auto it = reference.begin(); it != reference.end(); ++it) {
                if (!actual.contains(it.key())) fail(path + "/" + it.key(), "Missing field");
                else visit(it.value(), actual.at(it.key()), path + "/" + it.key());
            }
            for (auto it = actual.begin(); it != actual.end(); ++it)
                if (!reference.contains(it.key())) fail(path + "/" + it.key(), "Unexpected field");
        } else if (reference.is_array()) {
            if (reference.size() != actual.size()) { fail(path, "Array size changed"); return; }
            for (size_t i = 0; i < reference.size(); ++i) visit(reference[i], actual[i], path + "/" + std::to_string(i));
        } else if (reference.is_number_float()) {
            ++numbers;
            const double a = reference.get<double>(), b = actual.get<double>();
            if (!std::isfinite(a) || !std::isfinite(b)) { fail(path, "Non-finite number"); return; }
            if (std::memcmp(&a, &b, sizeof(double)) == 0) return;
            ++bit_differences;
            const double difference = std::fabs(a - b), scale = (std::max)(std::fabs(a), std::fabs(b));
            max_absolute = (std::max)(max_absolute, difference);
            if (scale > 0) max_relative = (std::max)(max_relative, difference / scale);
            if (exact || difference > absolute + relative * std::fabs(a)) {
                fail(path, "Floating-point result changed");
                if (details.size() < 16) details.push_back({{"path", path}, {"reference", a}, {"actual", b}});
            }
        } else {
            if (reference.is_number()) ++numbers;
            if (reference != actual) fail(path, "Value changed");
        }
    }
    Json report() const {
        return {{"status", failures ? "different" : bit_differences ? "equivalent" : "identical"},
            {"numbers", numbers}, {"bit_differences", bit_differences}, {"failures", failures},
            {"max_absolute_difference", max_absolute}, {"max_relative_difference", max_relative}, {"details", details}};
    }
};
inline Json read(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return Json::parse(in);
}
inline std::set<fs::path> files(const fs::path& root) {
    if (!fs::is_directory(root)) throw std::runtime_error("Missing output directory " + root.string());
    std::set<fs::path> paths;
    for (const auto& item : fs::recursive_directory_iterator(root))
        if (item.is_regular_file() && item.path().extension() == ".json") paths.insert(fs::relative(item.path(), root));
    return paths;
}
inline Json compare(const fs::path& reference, const fs::path& actual, bool exact, double absolute, double relative) {
    auto paths = files(reference); const auto current = files(actual); paths.insert(current.begin(), current.end());
    if (paths.empty()) throw std::runtime_error("No results to compare");
    Json cases = Json::object(); size_t failures = 0, identical = 0, equivalent = 0, numbers = 0, bits = 0;
    double max_absolute = 0, max_relative = 0;
    for (const auto& path : paths) {
        Comparison c; c.exact = exact; c.absolute = absolute; c.relative = relative;
        if (!fs::exists(reference / path)) c.fail("", "Unexpected result file");
        else if (!fs::exists(actual / path)) c.fail("", "Missing result file");
        else c.visit(read(reference / path), read(actual / path));
        cases[path.generic_u8string()] = c.report();
        failures += c.failures != 0; identical += !c.failures && !c.bit_differences; equivalent += !c.failures && c.bit_differences;
        numbers += c.numbers; bits += c.bit_differences;
        max_absolute = (std::max)(max_absolute, c.max_absolute); max_relative = (std::max)(max_relative, c.max_relative);
    }
    return {{"status", failures ? "different" : equivalent ? "equivalent" : "identical"},
        {"exact", exact}, {"absolute_tolerance", absolute}, {"relative_tolerance", relative},
        {"files", paths.size()}, {"identical_files", identical}, {"equivalent_files", equivalent}, {"different_files", failures},
        {"numbers", numbers}, {"bit_differences", bits}, {"max_absolute_difference", max_absolute}, {"max_relative_difference", max_relative},
        {"cases", cases}};
}
}
