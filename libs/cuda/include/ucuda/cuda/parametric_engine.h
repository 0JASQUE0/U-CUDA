#pragma once
#include "ucuda/domain/analysis_types.h"
#include "ucuda/domain/neural_types.h"     // NeuralRequest / NeuralResult: большие сети
#include "ucuda/cuda/runtime_settings.h"
class ParametricEngine {
public:
    ParametricEngine();
    ~ParametricEngine();
    ParametricEngine(const ParametricEngine&) = delete;
    ParametricEngine& operator=(const ParametricEngine&) = delete;

    // 1D-бифуркация. Если КРС не изменился с прошлого запуска — переиспользуется
    // скомпилированный PTX (кэш по хэшу krs_body + amountOfX).
    Bifurcation1DResult run_bifurcation_1d(const Bifurcation1DRequest& req);

    // 1D DFT — параметрическое преобразование Фурье. Использует тот же PTX-кэш
    // (classical и continuation), что и run_bifurcation_1d — DFT_custom уже
    // скомпилирован в оба модуля, просто ищется дополнительно.
    Dft1DResult run_dft_1d(const Dft1DRequest& req);

    // 2D-бифуркация — период(p1, p2) через DBSCAN на квадратной сетке.
    Bifurcation2DResult run_bifurcation_2d(const Bifurcation2DRequest& req);

    // 1D-LLE. Отдельный PTX-кэш (другое шаблоное тело и другой kernel).
    LLE1DResult run_lle_1d(const LLE1DRequest& req);

    // 1D-LS — спектр Ляпунова (N экспонент на точку параметра).
    LS1DResult run_ls_1d(const LS1DRequest& req);

    // 2D-LLE — λ(p1, p2) на квадратной сетке. Отдельный PTX-кэш (другая
    // ветка par_or_var, тот же kernel LLEKernelCUDA).
    LLE2DResult run_lle_2d(const LLE2DRequest& req);

    // 2D-LS — полный спектр N экспонент в каждой ячейке n_pts × n_pts.
    LS2DResult run_ls_2d(const LS2DRequest& req);

    // Basins of attraction — карта на сетке IC. 4 output-поля + cluster ids.
    BasinsResult run_basins(const BasinsRequest& req);

    // DBSCAN-only прогон по уже посчитанным фичам (для перекластеризации при
    // смене eps_dbscan без пересчёта траекторий).
    BasinsReclusterResult run_basins_recluster(const BasinsReclusterRequest& req);

    // Fast Synchro — режим выбирается через req.mode (0 = traj, 1 = grid).
    FastSyncResult run_fastsync(const FastSyncRequest& req);

    // Order — оценка порядка точности (Эйткен/Ричардсон на тройке h, h/2, h/4).
    // Один вызов обслуживает и 1D-диаграмму, и 2D-карту: различает их axis_y.kind.
    OrderResult run_order(const OrderRequest& req);

    // Performance — время счёта vs ошибка по той же одномерной сетке.
    PerfResult run_performance(const PerfRequest& req);

    // Stability — область устойчивости схемы на двумерной задаче Дальквиста.
    StabilityResult run_stability(const StabilityRequest& req);

    // Network — интегрирование сети связанных осцилляторов.
    NetworkResult run_network(const NetworkRequest& req);
    NeuralResult run_neural(const NeuralRequest& req);

    // Signal metrics — max/min/mean, частоты по пикам и параметры Хьорта
    // на 1D-отрезке или 2D-сетке свипа.
    SignalMetricsResult run_signal_metrics(const SignalMetricsRequest& req);

    // Компилирует модуль под этот запрос, ничего не считая: тот же ключ кэша, что возьмёт
    // соответствующий run_*, поэтому Run потом просто найдёт готовый модуль. Зовётся из фонового
    // потока, пока пользователь ещё настраивает параметры или пока считается предыдущая задача;
    // ошибки глотаются (не смогли прогреть — Run скомпилирует сам). Ждать не нужно: если Run
    // обгонит прогрев того же ключа, он дождётся его результата вместо второй компиляции.
    void prewarm(const Bifurcation1DRequest& req);
    void prewarm(const Bifurcation2DRequest& req);
    void prewarm(const LLE1DRequest& req);
    void prewarm(const LLE2DRequest& req);
    void prewarm(const LS1DRequest& req);
    void prewarm(const LS2DRequest& req);
    void prewarm(const BasinsRequest& req);
    void prewarm(const SignalMetricsRequest& req);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- Свободные функции движка (реализация в cuda) ----
// Стабильность: общие для GPU- и CPU-ветки проверки и сводка.
// Проверка запроса и раскладка сетки — общие для GPU- и CPU-ветки, иначе одни
// и те же настройки принимались бы на одном устройстве и отвергались на
// другом. stability_validate возвращает пустую строку, когда всё в порядке.
std::string stability_validate(const StabilityRequest& req);
void        stability_fill_axes(const StabilityRequest& req, StabilityResult& res);

// Счётчики, число устойчивых ячеек и диапазон rho по чистым ячейкам. Функция
// общая для GPU- и CPU-ветки: разойдись они здесь, одна и та же карта
// подписывалась бы по-разному в зависимости от устройства.
void stability_summarize(StabilityResult& res);

// Относительная ошибка одного шага на тестовой матрице узла (sigma, omega):
// ||e^{hA} - R||_2 / ||e^{hA}||_2, где A строится из (sigma, omega, k, r) по
// тем же формулам, что в ядре, а R = [[R00, R01], [R10, R11]] — матрица
// усиления схемы. Спектр A ровно sigma +- i*omega, поэтому экспонента берётся
// в замкнутом виде:
//   e^{hA} = e^{h*sigma} * (cos(h*omega) I + h*sinc(h*omega) (A - sigma I)).
// Считается в double при любой точности CPU-ветки: сравнение идёт с допуском
// порядка единицы, и разрядов double здесь с запасом.
// Возвращает +inf, когда e^{hA} вырождается в 0 или inf, и NaN, когда
// матрицу построить нельзя или R не конечна.
double stability_step_error(double sigma, double omega, double k, double r, double h,
                            double R00, double R01, double R10, double R11);

// CSV одной таблицей: x [, y], regime, метрики из маски. Зовёт и движок
// (csv_output_path), и ПКМ-экспорт с графика.
bool signal_metrics_write_csv(const SignalMetricsResult& r, const std::string& path, std::string& err);

// Без контекста: только NVRTC. Тяжёлый (~0.2-1 с) — звать из фонового потока.
bool krs_step_cost(const std::string& krs_body, int amountOfX, KrsStepCost& out, std::string& err);

bool gpu_device_info(std::string& name, int& sm_count, std::string& err);
// <каталог exe>\gpu_calibration.txt — хранилище калибровок (time_estimate.h).
std::string gpu_calibration_path();
// Занимает GPU на ~0.5-1 с. Звать, когда расчётов нет: иначе замер завышен.
bool gpu_calibrate(GpuCalibration& out, std::string& err);
