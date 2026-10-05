#pragma once
// Прогрев GPU перед замером времени (Order -> Performance).
//
// Видеокарта поднимает частоту только под сплошной нагрузкой. Короткие запуски с
// синхронизацией после каждого её не дают, и одно и то же ядро мерилось то на холостой
// частоте, то на рабочей — на Lorenz RK45 до 3-5 раз разницы (16 против 3.5 мкс на шаг).
// Поэтому перед замером ядро ставится в очередь пачками БЕЗ синхронизации между запусками,
// пока не пройдёт target_ms стенного времени.
//
// launch() — поставить один запуск в очередь, sync() — дождаться всех; оба возвращают
// false при ошибке (текст ошибки вызывающий кладёт сам).
#include <algorithm>
#include <chrono>
#include <cmath>

template <class Launch, class Sync>
bool perf_burst_warmup(double target_ms, Launch launch, Sync sync) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto elapsed_ms = [&]() {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    if (!launch() || !sync()) return false;
    // Скобки вокруг std::max/min — от макросов max/min из <windows.h>.
    const double one = (std::max)(elapsed_ms(), 1e-3);   // один запуск вместе с синхронизацией
    while (elapsed_ms() < target_ms) {
        // Пачка на ~10 мс по оценке первого запуска, не больше 1000 штук: очередь драйвера
        // не раздувается, а длинное ядро не уводит прогрев далеко за цель.
        const double left = target_ms - elapsed_ms();
        const int k = (int)std::clamp(std::ceil((std::min)(left, 10.0) / one), 1.0, 1000.0);
        for (int j = 0; j < k; ++j)
            if (!launch()) return false;
        if (!sync()) return false;
    }
    return true;
}
