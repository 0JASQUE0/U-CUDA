#pragma once
// Расчётные службы, которые интерфейс вызывает напрямую (проверки, файлы схем), без GPU-задачи.
// ui видит только этот заголовок, а не cuda / numerics: границу проверяет tests/layer_rules.cmake.
#include <string>
#include <vector>

struct PhaseAnalysisSession;

// Найден ли компилятор для CPU-путей (cl.exe). При false в why_not причина.
bool cpu_compiler_available(std::string* why_not = nullptr);

// Обращения X[k] / a[k] с константным индексом за пределами системы (тело пользовательской КРС).
struct BodyIndexDiag {
    int         line = 0;
    std::string message;
};
std::vector<BodyIndexDiag> check_body_indices(const std::string& body, int amount_of_x, int amount_of_values);

// Сборка C-тела регулятора шага через NVRTC (кнопка Check библиотеки регуляторов). Блокирует поток.
bool check_controller_body(const std::string& body, std::string& log);

// Время изменения файла схемы (слежение Watch во вкладке DAE); 0, если файла нет.
long long circuit_file_mtime(const std::string& path);

// SPICE-нетлист синтезированной схемы из результата Analysis в файл path; итог в s.result.circuit_status.
void write_synthesized_netlist(PhaseAnalysisSession& s, const std::string& path);
