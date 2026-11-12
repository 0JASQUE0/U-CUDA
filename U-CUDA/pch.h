#pragma once
// Предкомпилированный заголовок классической сборки (U-CUDA.sln). Подключается ко всем .cpp
// через /FI (ForcedIncludeFiles в U-CUDA.vcxproj), исходники его не включают — CMake-сборка
// тестов идёт без него, и код обязан собираться и так.
//
// Сюда — только тяжёлые и стабильные заголовки: стандартная библиотека и ImGui. Заголовки
// проекта (app_model.h и т.п.) меняются часто, а любая правка PCH пересобирает всё.
// Замер 2026-11-12, полная пересборка на 12 потоках: 82 с, из них 62 % времени компилятора —
// разбор заголовков (c1xx), по 5–10 с на файл.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
