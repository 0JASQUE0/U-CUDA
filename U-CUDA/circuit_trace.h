#pragma once
#include <fstream>
#include <string>

// ВРЕМЕННО: локализация падения на схемном пути. Снять, когда причина найдена.
//
// Пишется с немедленным сбросом на диск, потому что смысл именно в последней
// строке перед падением: процесс до закрытия файла не доживёт.
inline void circuit_trace(const char* where, const std::string& extra = std::string()) {
    std::ofstream f("circuit_trace.log", std::ios::app);
    if (!f) return;
    f << where;
    if (!extra.empty()) f << " | " << extra;
    f << std::endl;
}
