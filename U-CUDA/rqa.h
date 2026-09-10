#pragma once
#include "configCUDA.h"
#include <string>
#include <vector>

// RQA (Recurrence Quantification Analysis) для одной траектории.
//
// Почему заголовок отдельный, а реализация — в hostLibrary.cu: считается всё на GPU, а вызывает
// расчёт analysis_session.cpp (обычный cl.exe). Включить туда hostLibrary.cuh нельзя — он тянет
// cudaLibrary.cuh с __device__-квалификаторами, которых MSVC не понимает. Здесь же только POD-типы
// и numb из configCUDA.h (тот host-safe, см. гарды в его шапке).
//
// Матрица считается ОДИН раз на конфиг: порог eps применяется уже к готовой D на устройстве, но
// смена eps требует нового вызова compute() — пересборка D стоит O(n^2*d) и на n=2048 это единицы
// миллисекунд, отдельный «пересчитать только линии» путь того не стоит.
namespace rqa {

// Из чего строится вектор состояния. Дефолта НЕТ намеренно: для ОДУ честнее полный вектор, но в
// публикациях по RQA принят Такенс, и молчаливый выбор за пользователя тут врёт о методе.
// Порядок = порядок в комбо-боксе и в сохранённых сессиях (пишется как int) — дополнять только
// в конец.
enum class Source : int {
    None = 0,          // не выбрано — compute() вернёт ошибку, GUI просит выбрать
    StateVector = 1,   // (x1..xn) как есть, без эмбеддинга
    Embedding = 2      // Такенс по одной переменной: (v[i], v[i+tau], ..., v[i+(m-1)tau])
};

enum class Norm : int { Euclidean = 0, Maximum = 1, Manhattan = 2 };

// Как выбирается порог eps.
enum class EpsMode : int {
    Absolute = 0,     // eps как есть
    FracMaxDist = 1,  // eps = eps_frac * max(D)
    FracStd = 2,      // eps = eps_frac * среднеквадратичное отклонение точек от центра масс
    TargetRR = 3      // eps подбирается так, чтобы RR == target_rr (квантиль распределения D)
};

struct Config {
    Source  source = Source::None;
    int     var = 0;             // Embedding: индекс переменной; StateVector: не используется
    int     m = 3;               // размерность эмбеддинга
    int     tau = 1;             // задержка эмбеддинга, В ОТСЧЁТАХ прореженного ряда
    Norm    norm = Norm::Euclidean;
    EpsMode eps_mode = EpsMode::TargetRR;
    numb    eps = 0.1;           // Absolute
    numb    eps_frac = 0.1;      // FracMaxDist / FracStd
    numb    target_rr = 0.05;    // TargetRR (доля, не проценты)
    int     theiler = 1;         // окно Тейлера: пары с |i-j| <= theiler исключены ПОЛНОСТЬЮ
    int     l_min = 2;           // мин. длина диагональной линии (DET, L, ENTR)
    int     v_min = 2;           // мин. длина вертикальной линии (LAM, TT, V_ENTR)
    int     points = 2048;       // верхняя граница n; ряд равномерно прореживается до неё
    bool    network_measures = false;  // clustering/transitivity — O(RR*n^3), по умолчанию выкл.
};

// Все метрики. NaN = «не определена на этих данных» (нет ни одной линии нужной длины и т.п.).
// Насыщенных заглушек тут нет намеренно: пустой знаменатель честнее показать как NaN, чем как 0.
struct Metrics {
    double RR = 0.0;       // recurrence rate
    double DET = 0.0;      // determinism
    double L = 0.0;        // средняя длина диагональной линии
    double L_max = 0.0;
    double DIV = 0.0;      // 1 / L_max
    double ENTR = 0.0;     // энтропия Шеннона распределения диагональных линий
    double RATIO = 0.0;    // DET / RR
    double LAM = 0.0;      // laminarity
    double TT = 0.0;       // trapping time (средняя вертикаль)
    double V_max = 0.0;
    double V_ENTR = 0.0;
    double TREND = 0.0;    // дрейф RR по диагоналям (наклон линейной регрессии)
    double T1 = 0.0;       // время возврата 1-го рода
    double T2 = 0.0;       // время возврата 2-го рода (по началам вертикальных блоков)
    double W = 0.0;        // средняя длина БЕЛОЙ вертикальной линии
    double W_max = 0.0;
    double RTE = 0.0;      // recurrence time entropy (нормированная энтропия белых линий)
    double clustering = 0.0;    // средний коэффициент кластеризации сети рекуррентности
    double transitivity = 0.0;
    bool   network_valid = false;  // false => clustering/transitivity не считались
};

struct Result {
    bool        ok = false;
    std::string error;

    int    n = 0;          // фактическая сторона матрицы (<= Config::points)
    double t0 = 0.0;       // время первой точки выборки, с
    double t1 = 0.0;       // время последней точки выборки, с
    double dt = 0.0;       // шаг выборки, с (t1-t0)/(n-1)
    double eps_used = 0.0; // фактически применённый порог
    double dist_max = 0.0; // max(D) — для UI и для режима FracMaxDist

    // Матрица расстояний n*n, row-major (idx = i*n + j). Именно её ждёт HeatmapView::render;
    // бинарную матрицу рекуррентности GUI получает порогом по этой же (сравнение с eps_used),
    // чтобы переключение Distance <-> Recurrence не гоняло GPU.
    // Тип double, а не numb: HeatmapView принимает const double*, и при смене typedef numb на
    // float расширение должно быть ВИДНО здесь, а не молча происходить внутри.
    std::vector<double> dist;

    Metrics metrics;
};

// Побитовое равенство конфигов. Нужно воркеру: два окна проекций с одинаковыми настройками
// должны считаться ОДИН раз, а результат ищется по конфигу, а не по индексу проекции (индексы
// разъезжаются, если пользователь добавил или удалил окно, пока шёл расчёт).
inline bool operator==(const Config& a, const Config& b)
{
    return a.source == b.source && a.var == b.var && a.m == b.m && a.tau == b.tau
        && a.norm == b.norm && a.eps_mode == b.eps_mode
        && a.eps == b.eps && a.eps_frac == b.eps_frac && a.target_rr == b.target_rr
        && a.theiler == b.theiler && a.l_min == b.l_min && a.v_min == b.v_min
        && a.points == b.points && a.network_measures == b.network_measures;
}
inline bool operator!=(const Config& a, const Config& b) { return !(a == b); }

// traj — [step][coord], как AnalysisResult::trajectories[ic], но ЖЕЛАТЕЛЬНО непрореженная:
// прореживание до Config::points делается здесь, и dt возвращается честный.
// dt_traj — шаг между соседними точками traj в секундах (h * decimator, если traj уже прорежена).
// Возвращает out.ok; при ошибке текст в out.error.
bool compute(const std::vector<std::vector<double>>& traj,
             double dt_traj,
             const Config& cfg,
             Result& out);

}  // namespace rqa
