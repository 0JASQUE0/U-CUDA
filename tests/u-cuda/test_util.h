#pragma once
// Общее для тестов U-CUDA (MSBuild-проект, без CMake): проверка, заглушки символов GUI-части,
// helpers. Тесты собирает и запускает run_tests.bat против объектников Release.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef near
#undef far

// Символы, которые объектники движка ждут от приложения (как runner/stubs.cpp скилла ucuda-headless).
// Тест, слинкованный со всем приложением (architecture), берёт basin_feature_name из gui.obj.
#ifndef UT_FULL_APP
const char* basin_feature_name(int) { return "f"; }
#endif
std::string exe_dir() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n);
    const size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? std::string(".") : p.substr(0, k);
}

namespace ut {

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
}

// Код 77 — пропуск (нет CUDA), как SKIP_RETURN_CODE в CTest.
constexpr int kSkip = 77;

// Запуск тела main() с перехватом: 0 — прошло, 1 — упало.
template <class F>
int run_main(const char* title, F&& f) {
    try {
        const int rc = f();
        if (rc == 0) std::printf("%s: passed\n", title);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: FAILED: %s\n", title, e.what());
        return 1;
    }
}

}  // namespace ut
