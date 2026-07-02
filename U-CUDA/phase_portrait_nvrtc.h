#pragma once
#include <string>
#include <vector>

// Считает N фазовых траекторий на GPU через NVRTC (параллельно, поток на НУ).
// krs_body — тело calculateDiscreteModel (то, что выдаёт codegen_scheme).
// Движок компилируется лениво и кэширует несколько последних скомпилированных
// вариантов (по тексту КРС + размерности): переключение между недавно
// использованными системами/методами анализа не требует повторной компиляции.
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

// Фоновый (best-effort) прогрев кэша: компилирует krs_body/amountOfX прямо
// сейчас, синхронно в вызывающем потоке — асинхронность (не блокировать GUI)
// на совести вызывающего (см. PhaseAnalysisSession::regenerate_krs, который
// зовёт это из std::async сразу при смене системы/метода интегрирования).
// К моменту, когда пользователь нажмёт Run/Recompute, модуль уже будет в
// кэше NvrtcEngine — computePhasePortraitsNVRTC() возьмёт cache hit.
// Ошибки не репортит: если прогрев не успел или упал, тот же krs_body
// перекомпилируется (и покажет тот же NVRTC-лог) на настоящем Run.
void prewarmPhasePortraitsNVRTC(const std::string& krs_body, int amountOfX);