#pragma once
#include "ucuda/platform/platform_paths.h"
#include "ucuda/cuda/parametric_engine.h"
#ifdef UCUDA_TEST_ARTIFACTS
#include "../support/artifacts.h"
#ifdef UCUDA_TEST_GTEST
#include <gtest/gtest.h>
#endif
#endif
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <chrono>
#ifndef _MSC_FULL_VER
#define _MSC_FULL_VER 0
#endif
std::string exe_dir() { return platform::executable_dir(); }
namespace ut {
inline std::filesystem::path fixture_library() {
 struct Copy {
  std::filesystem::path root;
  Copy() {
   root = std::filesystem::temp_directory_path() / ("ucuda-test-library-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
   if (!std::filesystem::create_directory(root)) throw std::runtime_error("Temporary test directory already exists");
   std::filesystem::copy(std::filesystem::path(exe_dir()) / "library_template", root, std::filesystem::copy_options::recursive);
#ifdef UCUDA_TEST_SOURCE
   const auto fixtures = std::filesystem::path(UCUDA_TEST_SOURCE) / "tests/fixtures/library";
   std::filesystem::copy(fixtures, root, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
#endif
  }
  ~Copy() {
   std::error_code ec;
   if (root.parent_path() == std::filesystem::temp_directory_path(ec) && root.filename().string().rfind("ucuda-test-library-", 0) == 0)
    std::filesystem::remove_all(root, ec);
  }
 };
 static Copy copy; return copy.root;
}


inline void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

// "name: value vs ref (tol)" — для сообщений о расхождении чисел.
inline std::string num(const char* name, double v, double ref, double tol) {
    char b[256];
    std::snprintf(b, sizeof b, "%s: %.10g vs %.10g (tol %.1e)", name, v, ref, tol);
    return b;
}

inline void approx(double v, double ref, double tol, const char* name) {
    require(std::isfinite(v) && std::fabs(v - ref) <= tol, num(name, v, ref, tol));
#ifdef UCUDA_TEST_ARTIFACTS
    static size_t index = 0;
    std::string scope = UCUDA_TEST_SUITE;
#ifdef UCUDA_TEST_GTEST
    if (const auto* test = ::testing::UnitTest::GetInstance()->current_test_info())
        scope += std::string(".") + test->test_suite_name() + "." + test->name();
#endif
    regression::snapshot("upstream." + scope + "." + std::to_string(index++),
        {{"assertion", name}, {"reference", ref}, {"absolute_tolerance", tol}}, {v});
#endif
}

// Код 77 — пропуск (нет CUDA), как SKIP_RETURN_CODE в CTest.
constexpr int kSkip = 77;
inline void set_environment(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value ? value : "") != 0) throw std::runtime_error("Cannot set test environment");
#else
    if ((value ? setenv(name, value, 1) : unsetenv(name)) != 0) throw std::runtime_error("Cannot set test environment");
#endif
}

// Запуск тела main() с перехватом: 0 — прошло, 1 — упало.
template <class F>
int run_main(const char* title, F&& f) {
    try {
#ifdef UCUDA_TEST_ARTIFACTS
        if (regression::test_context().empty()) regression::test_context() = "upstream_" UCUDA_TEST_SUITE;
#endif
        const int rc = f();
        if (rc == 0) std::printf("%s: passed\n", title);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: FAILED: %s\n", title, e.what());
        return 1;
    }
}

}  // namespace ut
