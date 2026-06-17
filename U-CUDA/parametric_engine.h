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

    // flags[i]: 1 — расчёт прошёл, -1 — траектория разошлась за max_value.
    std::vector<int> flags;
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

    // TODO (следующие шаги): run_bifurcation_2d, run_lle_1d, run_basins_of_attraction, ...

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
