#pragma once
#include <string>
#include <vector>

// Считает N фазовых траекторий на GPU через NVRTC (параллельно, поток на НУ).
// krs_body — тело calculateDiscreteModel (то, что выдаёт codegen_scheme).
// Движок компилируется лениво и кэшируется по тексту КРС: если krs_body
// не изменился с прошлого вызова — повторная компиляция не делается.
//
//   ic_flat   — начальные условия всех НУ, плоско: ic_flat[k*amountOfX + i]
//   N         — число траекторий (наборов НУ)
//   values    — параметры a[] (со сдвигом, a[0] зарезервирован), ОБЩИЕ для всех
//   out       — [N][total][amountOfX]
// Возвращает false при ошибке; err (если не nullptr) получает текст.
bool computePhasePortraitsNVRTC(
    const std::string& krs_body,
    int amountOfX,
    const double* ic_flat, int N,
    const double* values, int amountOfValues,
    double h, int totalPoints, int skipPoints,
    std::vector<std::vector<std::vector<double>>>& outTrajectories,
    std::string* err = nullptr);