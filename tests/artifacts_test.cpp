#include "support/artifacts.h"
#include "support/process.h"
#include <gtest/gtest.h>
#include <chrono>
#include <limits>

namespace {
using namespace regression;
Json config() { return {{"schema", 1}, {"performance", {{"slowdown_limit", .1}, {"warmup_seconds", .3},
    {"minimum_seconds", .1}, {"series", 3}, {"repetitions", 9}, {"max_relative_spread", .1}}},
    {"numerical", {{"absolute", 1e-12}, {"relative", 1e-10}}}, {"cases", Json::object()}}; }
Json sample() { return {{"id", "case"}, {"version", 1}, {"inputs", {{"h", .01}}}, {"values", {1., 2.}}, {"passed", true}}; }
TEST(ArtifactPolicyCpu, DetectsNumericalCorruptionAndShapeChange) {
    const auto base = sample(); auto run = base;
    run["values"][0] = 1.01; EXPECT_EQ(compare_case(run, base, config(), true)["status"], "failed");
    run = base; run["values"].push_back(3); EXPECT_EQ(compare_case(run, base, config(), true)["status"], "failed");
    run = base; run["passed"] = false; EXPECT_EQ(compare_case(run, base, config(), true)["status"], "failed");
}
TEST(ArtifactPolicyCpu, DetectsSlowdownAndAcceptsImprovement) {
    auto base = sample(); base["series"] = {{10., 10., 10.}, {10., 10., 10.}, {10., 10., 10.}};
    EXPECT_EQ(compare_case(base, base, config(), true)["status"], "pass");
    auto run = base; run["series"] = {{12., 12., 12.}, {12., 12., 12.}, {12., 12., 12.}};
    EXPECT_EQ(compare_case(run, base, config(), true)["status"], "failed");
    auto c = config(); c["cases"]["case"]["slowdown_limit"] = .25;
    EXPECT_EQ(compare_case(run, base, c, true)["status"], "pass");
    run["series"] = {{8., 8., 8.}, {8., 8., 8.}, {8., 8., 8.}};
    EXPECT_EQ(compare_case(run, base, config(), true)["status"], "pass");
    EXPECT_EQ(compare_case(run, base, config(), false)["status"], "incompatible");
    run["series"] = {{12., 12., 12.}, {12., 12., 12.}, {12., 12., 12.}};
    run["series"].push_back({12., 12., 12.}); run["series"].push_back({12., 12., 12.});
    EXPECT_EQ(compare_case(run, base, config(), true)["status"], "failed");
}
TEST(ArtifactPolicyCpu, RejectsChangedInputsAndUnstableTiming) {
    auto base = sample(); auto run = base; run["inputs"]["h"] = .02;
    EXPECT_EQ(compare_case(run, base, config(), true)["status"], "incompatible");
    base["series"] = {{10., 10., 10.}, {10., 10., 10.}, {10., 10., 10.}};
    run = base; run["series"][1] = {20., 20., 20.};
    EXPECT_EQ(compare_case(run, base, config(), true)["status"], "inconclusive");
}
TEST(ArtifactPolicyCpu, ValidatesConfiguration) {
    auto c = config(); EXPECT_NO_THROW(validate_config(c));
    c["performance"]["slowdown_limit"] = -1; EXPECT_THROW(validate_config(c), std::runtime_error);
    c = config(); c["performance"]["series"] = 0; EXPECT_THROW(validate_config(c), std::runtime_error);
    c = config(); c["performance"]["series"] = 3.5; EXPECT_THROW(validate_config(c), std::runtime_error);
    c = config(); c["cases"]["case"]["relative"] = -1; EXPECT_THROW(validate_config(c), std::runtime_error);
    EXPECT_THROW(median({}), std::runtime_error);
    EXPECT_THROW(median({std::numeric_limits<double>::quiet_NaN()}), std::runtime_error);
}
TEST(ArtifactPolicyCpu, BaselineAcceptanceIsExplicitVersionedAndSelective) {
    const auto parent = fs::temp_directory_path();
    const auto root = parent / ("ucuda baseline test проверка " + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(fs::create_directory(root));
    struct Cleanup {
        fs::path root, parent;
        ~Cleanup() { std::error_code ec; if (root.parent_path() == parent && root.filename().string().rfind("ucuda baseline test ", 0) == 0) fs::remove_all(root, ec); }
    } cleanup{root, parent};
    const auto cfg = root / "config.json", baseline = root / "active.json";
    write_json(cfg, config());
    Json run = {{"schema", 1}, {"checks_passed", true}, {"environment", {{"machine", "synthetic"}}},
        {"configuration", config()},
        {"revision", "independent-policy-fixture"}, {"profile", "regression|cpu||"}, {"mode", "regression"}, {"backend", "cpu"},
        {"selection", {{"cases", ""}, {"benchmarks", ""}}}, {"tests", {"fixture", "other_fixture"}},
        {"cases", {{"case", sample()}}}};
    run["cases"]["case"]["tests"] = {"fixture"};
    auto other = sample(); other["id"] = "other"; other["values"] = {7.}; other["tests"] = {"other_fixture"};
    other["series"] = Json::array();
    for (int i = 0; i < 3; ++i) other["series"].push_back(std::vector<double>(9, 10.));
    run["cases"]["other"] = other;
    const auto first = root / "first", second = root / "second";
    write_json(first / "run.json", run);
    auto command = [&](std::vector<std::string> args) {
        args.insert(args.begin(), UCUDA_TEST_TOOL);
        args.insert(args.end(), {"--config", cfg.string(), "--baseline", baseline.string()});
        return process(args, root / "command.log");
    };
    const std::string initialReason = "Проверенный исходный эталон";
    ASSERT_EQ(command({"baseline", "accept", "--run", first.string(), "--reason", initialReason}), 0)
        << text_file(root / "command.log");
    const auto initial = read_baseline(baseline);
    const auto initialManifest = read_json(baseline);
    EXPECT_EQ(initialManifest.at("case_directory"), "active-data");
    EXPECT_TRUE(initialManifest.at("cases").at("case").contains("artifact"));
    EXPECT_EQ(initial.at("reason"), initialReason);
    EXPECT_EQ(command({"compare", "--run", first.string()}), 0);
    EXPECT_EQ(read_baseline(baseline), initial) << "Ordinary comparisons must not change baselines";
    run["cases"]["case"]["values"][0] = 1.1;
    run["cases"]["other"]["values"][0] = 99.;
    write_json(second / "run.json", run);
    EXPECT_NE(command({"compare", "--run", second.string()}), 0);
    EXPECT_EQ(read_baseline(baseline), initial);
    const auto selectedFile = root / "selected.json"; write_json(selectedFile, Json::array({"case"}));
    ASSERT_EQ(command({"baseline", "accept", "--run", second.string(), "--cases-file", selectedFile.string(), "--reason", "Intentional fixture result change"}), 0)
        << text_file(root / "command.log");
    const auto updated = read_baseline(baseline);
    EXPECT_EQ(updated.at("cases").at("other"), initial.at("cases").at("other")) << "Selective acceptance changed an unselected case";
    EXPECT_EQ(read_json(baseline).at("cases").at("other"), initialManifest.at("cases").at("other"))
        << "Unchanged cases must share the same immutable artifact across versions";
    EXPECT_EQ(updated.at("previous"), initial.at("id"));
    EXPECT_EQ(read_baseline(baseline.parent_path() / "active-history" / (initial.at("id").get<std::string>() + ".json")), initial);
    run["cases"]["other"] = other; write_json(second / "run.json", run);
    EXPECT_EQ(command({"compare", "--run", second.string()}), 0);
    auto subset = run; subset["cases"].erase("other"); subset["tests"] = {"fixture"};
    subset["profile"] = "regression|cpu|filtered-fixture"; subset["selection"]["cases"] = "fixture";
    const auto filtered = root / "filtered"; write_json(filtered / "run.json", subset);
    EXPECT_EQ(command({"compare", "--run", filtered.string()}), 0) << text_file(root / "command.log");
    subset["cases"].erase("case"); write_json(filtered / "run.json", subset);
    EXPECT_NE(command({"compare", "--run", filtered.string()}), 0);
    auto slow = run;
    for (auto& series : slow["cases"]["other"]["series"]) series = std::vector<double>(9, 12.);
    write_json(filtered / "run.json", slow);
    EXPECT_NE(command({"compare", "--run", filtered.string()}), 0);
    EXPECT_NE(text_file(root / "command.log").find("Sustained slowdown"), std::string::npos);
    auto noisy = run; noisy["cases"]["other"]["series"][1] = std::vector<double>(9, 20.);
    write_json(filtered / "run.json", noisy);
    EXPECT_NE(command({"baseline", "accept", "--run", filtered.string(), "--cases", "other", "--reason", "Unstable timing fixture"}), 0);
    EXPECT_EQ(read_baseline(baseline), updated);
    auto changedInputs = run; changedInputs["cases"]["case"]["inputs"]["h"] = .02;
    write_json(filtered / "run.json", changedInputs);
    EXPECT_NE(command({"baseline", "accept", "--run", filtered.string(), "--cases", "case", "--reason", "Missing scenario version"}), 0);
    EXPECT_EQ(read_baseline(baseline), updated);
    run["checks_passed"] = false; write_json(second / "run.json", run);
    EXPECT_NE(command({"compare", "--run", second.string()}), 0);
    EXPECT_NE(command({"baseline", "accept", "--run", second.string(), "--reason", "Invalid run"}), 0);
    EXPECT_EQ(read_baseline(baseline), updated);
    run["checks_passed"] = true; run["cases"] = Json::object(); write_json(second / "run.json", run);
    EXPECT_NE(command({"compare", "--run", second.string()}), 0) << "Missing results must not pass";
    run["cases"]["case"] = sample(); run["environment"]["machine"] = "different";
    write_json(second / "run.json", run);
    EXPECT_NE(command({"baseline", "accept", "--run", second.string(), "--reason", "Wrong environment"}), 0);
    EXPECT_EQ(read_baseline(baseline), updated);
    changedInputs["cases"]["case"]["version"] = 2; write_json(filtered / "run.json", changedInputs);
    EXPECT_EQ(command({"baseline", "accept", "--run", filtered.string(), "--cases", "case", "--reason", "Versioned input change"}), 0)
        << text_file(root / "command.log");
    EXPECT_EQ(read_baseline(baseline).at("cases").at("case").at("version"), 2);

    // Partial acceptance must expose complete correctness profiles while noisy
    // performance cases stay unavailable. A newer accepted workload remains covered.
    const auto profiles = root / "profiles.json", complete = root / "complete";
    Json full = {{"schema", 1}, {"checks_passed", true}, {"environment", {{"machine", "synthetic"}}},
        {"configuration", config()}, {"revision", "profile-policy-fixture"},
        {"profile", "all|all||"}, {"mode", "all"}, {"backend", "all"},
        {"tests", {"fixture"}}, {"test_labels", {{"fixture", {"smoke", "regression", "cpu"}}}},
        {"cases", {{"case", sample()}, {"perf.cpu_fixture", other}}}};
    full["cases"]["case"]["tests"] = {"fixture"};
    full["cases"]["perf.cpu_fixture"]["id"] = "perf.cpu_fixture";
    write_json(complete / "run.json", full);
    auto profileCommand = [&](std::vector<std::string> args) {
        args.insert(args.begin(), {UCUDA_TEST_TOOL, "baseline", "accept"});
        args.insert(args.end(), {"--config", cfg.string(), "--baseline", profiles.string(), "--reason", "Partial profile policy fixture"});
        return process(args, root / "command.log");
    };
    ASSERT_EQ(profileCommand({"--run", complete.string(), "--cases", "case"}), 0) << text_file(root / "command.log");
    auto partial = read_baseline(profiles);
    EXPECT_TRUE(partial.at("profiles").contains("regression|all||"));
    EXPECT_TRUE(partial.at("profiles").contains("smoke|cpu||"));
    EXPECT_FALSE(partial.at("profiles").contains("performance|all||"));
    EXPECT_FALSE(partial.at("profiles").contains("all|all||"));
    auto newer = full;
    newer["profile"] = "performance|cpu|filtered";
    newer["cases"].erase("case");
    newer["cases"]["perf.cpu_fixture"]["version"] = 2;
    newer["cases"]["perf.cpu_fixture"]["inputs"]["h"] = .005;
    const auto replacement = root / "replacement"; write_json(replacement / "run.json", newer);
    ASSERT_EQ(profileCommand({"--run", replacement.string()}), 0) << text_file(root / "command.log");
    ASSERT_EQ(profileCommand({"--run", complete.string(), "--cases", "case"}), 0) << text_file(root / "command.log");
    partial = read_baseline(profiles);
    EXPECT_TRUE(partial.at("profiles").contains("performance|all||"));
    EXPECT_TRUE(partial.at("profiles").contains("all|all||"));
    EXPECT_EQ(partial.at("cases").at("perf.cpu_fixture").at("version"), 2);
}
}
