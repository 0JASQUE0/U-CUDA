#pragma once
#include <nlohmann/json.hpp>
#include "ucuda/cuda/runtime_settings.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace regression {
using Json = nlohmann::json;
namespace fs = std::filesystem;
inline std::string& test_context() { static thread_local std::string value; return value; }

inline Json read_json(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    return Json::parse(in);
}
inline void write_json(const fs::path& path, const Json& value) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out || !(out << std::setw(2) << value << '\n'))
        throw std::runtime_error("Cannot write " + path.string());
}
inline double median(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("No timing samples");
    for (double v : values)
        if (!std::isfinite(v) || v <= 0) throw std::runtime_error("Invalid timing sample");
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
inline void validate_config(const Json& c) {
    if (c.at("schema") != 1) throw std::runtime_error("Unsupported test configuration schema");
    const auto& p = c.at("performance");
    for (const char* key : {"slowdown_limit", "warmup_seconds", "minimum_seconds", "max_relative_spread"}) {
        const double v = p.at(key).get<double>();
        if (!std::isfinite(v) || v <= 0) throw std::runtime_error(std::string("Invalid ") + key);
    }
    if (!p.at("series").is_number_integer() || !p.at("repetitions").is_number_integer() ||
        p.at("series").get<int>() < 3 || p.at("repetitions").get<int>() < 3)
        throw std::runtime_error("At least three series and repetitions are required");
    for (const char* key : {"absolute", "relative"}) {
        const double v = c.at("numerical").at(key).get<double>();
        if (!std::isfinite(v) || v < 0) throw std::runtime_error("Invalid numerical tolerance");
    }
    if (!c.at("cases").is_object()) throw std::runtime_error("cases must be an object");
    for (auto it = c.at("cases").begin(); it != c.at("cases").end(); ++it) {
        for (auto k = it.value().begin(); k != it.value().end(); ++k) {
            if (k.key() != "absolute" && k.key() != "relative" && k.key() != "slowdown_limit")
                throw std::runtime_error("Unknown case override: " + k.key());
            const double v = k.value().get<double>();
            if (!std::isfinite(v) || v < 0) throw std::runtime_error("Invalid case override");
        }
    }
}
inline double setting(const Json& c, const std::string& id, const char* section, const char* key) {
    if (c.at("cases").contains(id) && c.at("cases").at(id).contains(key))
        return c.at("cases").at(id).at(key).get<double>();
    return c.at(section).at(key).get<double>();
}

// Independent assertions run before this function. A snapshot never replaces an oracle.
inline void snapshot(const std::string& id, const Json& inputs, const std::vector<double>& values, int version = 1) {
    if (version < 1) throw std::runtime_error("Invalid scenario version");
    for (double v : values)
        if (!std::isfinite(v)) throw std::runtime_error("Non-finite snapshot: " + id);
    const char* output = std::getenv("UCUDA_TEST_RESULT_DIR");
    if (!output || !*output) return;
    if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos)
        throw std::runtime_error("Invalid snapshot ID");
    const auto path = fs::path(output) / (id + ".json");
    Json recordedInputs = inputs;
    recordedInputs["execution"] = {{"fma", get_nvrtc_fmad()}, {"rdc", get_nvrtc_rdc()}, {"gpu_block_size", get_gpu_block_size()}};
    Json tests = Json::array();
    if (fs::exists(path)) {
        const auto previous = read_json(path);
        if (previous.at("version") != version || previous.at("inputs") != recordedInputs || previous.at("values") != values)
            throw std::runtime_error("Conflicting snapshots share an ID: " + id);
        tests = previous.value("tests", Json::array());
    }
    const auto& context = test_context();
    if (!context.empty() && std::find(tests.begin(), tests.end(), context) == tests.end()) tests.push_back(context);
    write_json(path, {{"id", id}, {"version", version}, {"inputs", recordedInputs}, {"values", values}, {"passed", true}, {"tests", tests}});
}

inline Json compare_case(const Json& current, const Json& base, const Json& config, bool compatible) {
    const auto id = current.at("id").get<std::string>();
    Json result = {{"id", id}, {"status", "pass"}};
    if (!current.value("passed", false) || !base.value("passed", false)) {
        result["status"] = "failed"; result["reason"] = "Independent checks failed"; return result;
    }
    if (current.at("version") != base.at("version") || current.at("inputs") != base.at("inputs")) {
        result["status"] = "incompatible"; result["reason"] = "Scenario inputs/version changed"; return result;
    }
    const auto a = current.at("values").get<std::vector<double>>();
    const auto b = base.at("values").get<std::vector<double>>();
    if (a.size() != b.size()) { result["status"] = "failed"; result["reason"] = "Output shape changed"; return result; }
    double worst = 0;
    const double abs = setting(config, id, "numerical", "absolute");
    const double rel = setting(config, id, "numerical", "relative");
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            result["status"] = "failed"; result["reason"] = "Non-finite result"; return result;
        }
        worst = (std::max)(worst, std::fabs(a[i] - b[i]));
        if (std::fabs(a[i] - b[i]) > abs + rel * std::fabs(b[i])) result["status"] = "failed";
    }
    result["max_absolute_difference"] = worst;
    if (current.contains("series") || base.contains("series")) {
        if (!compatible) { result["status"] = "incompatible"; result["reason"] = "Timing environment changed"; return result; }
        if (!current.contains("series") || !base.contains("series")) {
            result["status"] = "incompatible"; result["reason"] = "Missing timing series"; return result;
        }
        std::vector<double> cm, bm;
        bool noisy = false;
        bool currentSeries = true;
        const double spread = config.at("performance").at("max_relative_spread");
        for (const auto* series : {&current.at("series"), &base.at("series")}) {
            std::vector<double> medians;
            for (const auto& samples : *series) medians.push_back(median(samples.get<std::vector<double>>()));
            if (medians.size() < 3) throw std::runtime_error("Incomplete timing series");
            const double m = median(medians);
            const auto mm = std::minmax_element(medians.begin(), medians.end());
            noisy |= (*mm.second - *mm.first) / m > spread;
            if (currentSeries) cm = medians; else bm = medians;
            currentSeries = false;
        }
        const double limit = setting(config, id, "performance", "slowdown_limit");
        result["series_medians"] = cm; result["baseline_series_medians"] = bm;
        result["slowdown_limit"] = limit;
        const double referenceMedian = median(bm);
        const bool slow = std::all_of(cm.begin(), cm.end(), [&](double value) { return value > referenceMedian * (1 + limit); });
        result["time_ratio"] = median(cm) / median(bm);
        if (noisy && result["status"] == "pass") result["status"] = "inconclusive";
        else if (slow) { result["status"] = "failed"; result["reason"] = "Sustained slowdown"; }
    }
    return result;
}
}
