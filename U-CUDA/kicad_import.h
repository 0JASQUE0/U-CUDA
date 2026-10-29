#pragma once
// Источник схемы вкладки DAE: SPICE-netlist (.cir/.net/.sp/.spice) или схема KiCad
// (.kicad_sch). Схему KiCad в netlist переводит сам KiCad:
//   kicad-cli sch export netlist --format spice -o <tmp> <файл>
// (kicad-cli есть с KiCad 7). Код внутри KiCad не нужен.

#include <string>

// Путь к kicad-cli.exe: UCUDA_KICAD_CLI, затем PATH, затем
// Program Files\KiCad\<версия>\bin и %LOCALAPPDATA%\Programs\KiCad\<версия>\bin
// (берётся старшая версия). Пусто — не найден.
std::string kicad_find_cli();

bool circuit_source_is_kicad(const std::string& path);

// Текст netlist'а по пути (UTF-8). Для .kicad_sch зовёт kicad-cli; его вывод — в log.
bool circuit_source_load(const std::string& path, std::string& netlist,
                         std::string* log, std::string* err);

// Время изменения файла (для слежения за ним); -1 — файла нет.
long long circuit_source_mtime(const std::string& path);
