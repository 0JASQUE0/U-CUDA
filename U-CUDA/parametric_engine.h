#pragma once
//
// parametric_engine — фасад для параметрического анализа через NVRTC.
//
// Цель: изолировать остальной проект (GUI, AppModel, session_io) от внутренностей
// hostLibrary.cu / cudaLibrary.cu / main_NonLinAnal.cu. Эти файлы дорабатываются
// внешне и иногда меняются — здесь стабильный, проектный API, который при
// обновлении NonLinAnal правится в одном-двух местах, а не по всему коду.
//
// Архитектура:
//   1. Adapter (этот файл + .cpp): хранит стабильный API
//      run_bifurcation_1d/2d/lle/etc. и реализацию через NVRTC.
//   2. Шаблоны ядер: U-CUDA/kernels/*.template.cu — текстовые файлы,
//      копируются Post-Build Event в OutDir рядом с exe. NVRTC грузит и
//      подставляет КРС-тело + размерность.
//   3. Кэш PTX по хэшу (КРС + размерность) — чтобы не перекомпилировать
//      при смене только параметров анализа.
//

#include <memory>
#include <string>
#include <vector>

struct Bifurcation1DRequest {
    // КРС
    std::string krs_body;                    // тело calculateDiscreteModel из codegen
    int amountOfX = 0;                       // размерность системы (число переменных)

    // Начальные условия и базовые значения параметров системы
    std::vector<double> initial_conditions;  // длина == amountOfX
    std::vector<double> base_values;         // все параметры системы

    // Изменяемый параметр
    int param_index = 0;                     // индекс в base_values
    double param_lo = 0.0;
    double param_hi = 1.0;
    int n_pts = 1000;                        // разрешение по параметру

    // Запись результата
    int writable_var = 0;                    // индекс X-переменной, которую пишем

    // Интегрирование
    double h = 0.01;                         // шаг
    double transient_time = 0.0;             // burn-in (отбрасывается)
    double t_max = 0.0;                      // время записи после transient
    int pre_scaller = 1;                     // писать каждую pre_scaller-тую точку

    // Защита от расхождения
    double max_value = 1.0e6;

    // Если не пусто — engine запишет CSV с результатами по тому же формату,
    // что и NonLinAnal::bifurcation1D (для publication-quality пост-процессинга).
    // Пустая строка = ничего не пишем (только в памяти).
    std::string csv_output_path;
};

struct Bifurcation1DResult {
    bool ok = false;
    std::string error;

    int n_pts = 0;
    int record_steps = 0;

    // bifurcation_points[i] — значения writable_var для i-го значения параметра.
    // Длина каждого внутреннего вектора == record_steps.
    std::vector<std::vector<double>> bifurcation_points;

    // peak_times[i] — межпиковые интервалы (timeOfPeaks в NonLinAnal).
    // peak_times[i][j] = время от j-го пика до (j+1)-го для параметра i.
    // Та же длина, что и bifurcation_points[i].
    std::vector<std::vector<double>> peak_times;

    // flags[i]: 1 — расчёт прошёл, -1 — траектория разошлась за max_value.
    std::vector<int> flags;
};

// ============================================================================
// LLE (Largest Lyapunov Exponent) 1D — свип параметра, λ(param).
// Алгоритм: Wolf/Benettin с малым возмущением. Реализован в NonLinAnal
// (cudaLibrary.cu:LLEKernelCUDA), engine подключает его через NVRTC и
// шаблон kernels/lle1d.template.cu.
// ============================================================================

struct LLE1DRequest {
    std::string krs_body;
    int amountOfX = 0;

    std::vector<double> initial_conditions;  // длина == amountOfX
    std::vector<double> base_values;         // все параметры системы (с a[0]=0)

    int param_index = 0;                     // индекс в base_values (1-based)
    double param_lo = 0.0;
    double param_hi = 1.0;
    int n_pts = 500;

    // Интегрирование (как в bif1d).
    double h = 0.01;
    double transient_time = 0.0;
    double t_max = 100.0;

    // LLE-специфика: длина одного блока интегрирования между ренормализациями
    // в единицах времени (NT), и размер возмущения (eps).
    double NT  = 1.0;
    double eps = 1.0e-4;

    // Защита от расхождения.
    double max_value = 1.0e6;

    // Опциональный CSV.
    std::string csv_output_path;
};

struct LLE1DResult {
    bool ok = false;
    std::string error;

    int n_pts = 0;
    // lyapunov[i] — оценка λ для i-го значения параметра. Спец-значения:
    //   +999  — траектория x(t) сошла с аттрактора на transient'е (kernel-флаг),
    //   -999  — траектория разошлась за maxValue (kernel-флаг),
    //   nan/inf — численная проблема (потом фильтруется на стороне GUI).
    std::vector<double> lyapunov;
    // flags[i]: 1 (ok) / -1 (diverged) — для совместимости с UI-агрегацией.
    std::vector<int>    flags;
};

class ParametricEngine {
public:
    ParametricEngine();
    ~ParametricEngine();
    ParametricEngine(const ParametricEngine&) = delete;
    ParametricEngine& operator=(const ParametricEngine&) = delete;

    // 1D-бифуркация. Если КРС не изменился с прошлого запуска — переиспользуется
    // скомпилированный PTX (кэш по хэшу krs_body + amountOfX).
    Bifurcation1DResult run_bifurcation_1d(const Bifurcation1DRequest& req);

    // 1D-LLE. Отдельный PTX-кэш (другое шаблоное тело и другой kernel).
    LLE1DResult run_lle_1d(const LLE1DRequest& req);

    // TODO (следующие шаги): run_bifurcation_2d, run_lyapunov_spectrum_1d, ...

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
