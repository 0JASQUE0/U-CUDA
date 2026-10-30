// CPU-путь КРС (krs_cpu): cl.exe собирает тело шага в DLL, она считает, повторная сборка берётся
// из кэша, индекс за пределами X отвергается до компиляции, ошибка компилятора — с номером строки.
// (В CMake-форке это часть platform_test; процессы/OCR там Linux-only и здесь не проверяются.)
#include "test_util.h"
#include "krs_cpu.h"

int main() {
    return ut::run_main("krs_cpu", []() -> int {
        KrsCpuStep step;
        std::vector<KrsCpuDiag> diags;
        ut::require(step.compile("X[0] += h * a[0];", 1, 1, diags), "compile CPU scheme");
        numb x[] = { 1 }, a[] = { 2 };
        step.fn()(x, a, (numb)0.25);
        ut::approx(x[0], 1.5, 1e-12, "CPU step");
        ut::require(step.compile("X[0] += h * a[0];", 1, 1, diags), "cached CPU scheme");
        diags.clear();
        ut::require(!step.compile("X[1] = 0;", 1, 1, diags) && !diags.empty(), "out-of-bounds X[1] accepted");
        diags.clear();
        ut::require(!step.compile("X[0] = missing_symbol;", 1, 1, diags), "unknown symbol compiled");
        ut::require(!diags.empty() && diags[0].line == 1, "compiler diagnostic without line 1");
        return 0;
    });
}
