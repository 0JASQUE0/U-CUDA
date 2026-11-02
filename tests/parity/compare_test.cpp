#include "compare.h"
#include <gtest/gtest.h>
#include <chrono>

TEST(ParityComparatorCpu, ExactBitsTypesIntegersAndExplicitTolerance) {
    parity::Comparison exact;
    exact.visit(parity::Json{{"value", 1.}}, parity::Json{{"value", 1.}});
    EXPECT_EQ(exact.failures, 0u); EXPECT_EQ(exact.bit_differences, 0u);
    exact.visit(0., -0.); EXPECT_EQ(exact.failures, 1u); EXPECT_EQ(exact.bit_differences, 1u);
    parity::Comparison tolerance; tolerance.exact = false; tolerance.absolute = 1e-12;
    tolerance.visit(1., 1. + 1e-13); EXPECT_EQ(tolerance.failures, 0u); EXPECT_EQ(tolerance.bit_differences, 1u);
    tolerance.visit(1., 1.01); EXPECT_GT(tolerance.failures, 0u);
    parity::Comparison integer;
    integer.visit(parity::Json(uint64_t(1) << 63), parity::Json((uint64_t(1) << 63) + 1));
    EXPECT_EQ(integer.failures, 1u);
    integer.visit(1, 1.); EXPECT_EQ(integer.failures, 2u);
    integer.visit(parity::Json::array({1, 2}), parity::Json::array({1})); EXPECT_EQ(integer.failures, 3u);
    integer.visit(parity::Json{{"a", 1}}, parity::Json{{"b", 1}}); EXPECT_EQ(integer.failures, 5u);
}
TEST(ParityComparatorCpu, MissingEmptyAndModifiedOutputsCannotPass) {
    const auto root = std::filesystem::temp_directory_path() / ("ucuda-parity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path root; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(root, ec); } } cleanup{root};
    const auto reference = root / "reference", actual = root / "actual";
    std::filesystem::create_directories(reference); std::filesystem::create_directories(actual);
    EXPECT_THROW(parity::compare(reference, actual, true, 0, 0), std::runtime_error);
    { std::ofstream out(reference / "case.json"); out << "{\"values\":[1.0]}"; }
    EXPECT_EQ(parity::compare(reference, actual, true, 0, 0).at("different_files"), 1);
    std::filesystem::copy_file(reference / "case.json", actual / "case.json");
    EXPECT_EQ(parity::compare(reference, actual, true, 0, 0).at("status"), "identical");
    { std::ofstream out(actual / "case.json"); out << "{\"values\":[1.25]}"; }
    EXPECT_EQ(parity::compare(reference, actual, true, 0, 0).at("different_files"), 1);
}
