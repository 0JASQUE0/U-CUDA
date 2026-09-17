#pragma once
#include "ucuda/cuda/parametric_engine.h"
#include "ucuda/storage/data_export.h"
#include "ucuda/equations/adaptive_codegen.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include "ucuda/platform/platform_paths.h"
#include "ucuda/numerics/krs_cpu.h"   // CPU-ветка continuation: КРС -> нативная функция шага
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace ucuda::cuda_detail {



// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).






















// Глобальный PeakConfig — единственный источник истины для GUI-настраиваемых knobs
// configCUDA.h. Пишет UI-поток (Settings), читают worker-потоки при компиляции NVRTC и в
// CPU-ветках, поэтому под мьютексом. Epoch бампается на каждую запись и входит в cache-key
// модулей (hash_key) — иначе после смены настройки переиспользовался бы старый PTX.


// Строка опции для nvrtcCompileProgram. Литералы статические, поэтому указатель
// живёт дольше вызова.
const char* nvrtc_fmad_opt() ;


// gpu_free_budget — свободная память GPU с применённым запасом; общий первый шаг всех расчётов
// чанкования. Возвращает false, если cudaMemGetInfo недоступен (вызывающие превращают это в
// свой fail("cudaMemGetInfo failed")).
//
// `reserve` — доля свободной памяти, которую анализу разрешено занять; единого значения нет и
// сведено оно намеренно НЕ было:
//     0.92  — bifurcation 1D/2D, DFT и h-свипы (сверху ещё SAFETY_FACTOR 0.9 и вычет
//             memConstants у вызывающего, т.е. фактически ~0.83)
//     0.9   — basins
//     0.5   — LLE 1D/2D и fastsync-сетка
//     1/16  — LS 1D/2D: per-system память там ~N (буферы возмущённых траекторий), но 1/16 всё
//             равно агрессивнее, чем следует из одной этой оценки.
// Происхождение 0.5 и 1/16 не восстановлено — это следы отладки проблем с памятью, а не расчёт,
// поэтому числа сохранены как есть. Менять только после замера на реальных сетках; выигрыш
// реальный (LS сейчас берёт шестнадцатую часть памяти и делает в разы больше чанков).
bool gpu_free_budget(double reserve, size_t& out_bytes) ;

  // namespace



constexpr int kBlockSize         = 32;
     // как в NonLinAnal::bifurcation1D
constexpr int kMaxAmountOfX      = 32;

constexpr int kMaxAmountOfValues = 64;


// Зеркало constexpr-констант configCUDA.h. Сам заголовок сюда не включается — он читается как
// ТЕКСТ и уходит в NVRTC (src_configCUDA_h), поэтому значения дублируются литералами, как уже
// сделано в run_bif1d. При правке configCUDA.h эти значения надо править вместе с ним.
// Peak-knobs (doCalculatePeaks / eps_* / peak_threshold / max_amount_of_peaks) здесь больше НЕ
// дублируются: они настраиваются из GUI, и CPU-ветки читают их через get_peak_config(). Заодно
// ушло расхождение — kEpsFixedPoint был 1e-8 против 1e-6 в configCUDA.h, из-за чего CPU-ветка
// ловила fixed point не там, где GPU.
constexpr int    kCheckInterval      = 100;
      // CHECK_INTERVAL
constexpr double kPi    = 3.1415926535897932384626433832795;

constexpr double kEuler = 2.7182818284590452353602874713527;


// Число шагов интегрирования из времени: transient_time / h, NT / h и т.п.
//
// Прямой (int)(t / h) — UB, когда частное не влезает в int: TT=1e5 при h=1e-5 дают 1e10. На
// практике получался мусор или отрицательное число, т.е. транзиент молча не отрабатывал вовсе —
// худший вид ошибки, потому что расчёт выглядел успешным.
//
// Две версии, и выбор между ними НЕ косметический: он определяется типом приёмника. Значение
// уходит в ядро через void*[] в cuLaunchKernel, где тип обязан совпадать с параметром БАЙТ В
// БАЙТ — компилятор там ничего не проверит, рассогласование соберётся молча и развалит буфер
// аргументов на запуске.
//   ..._size_t — число ШАГОВ интегрирования; вся цепочка расширена до size_t (параметры ядер
//                amountOfPointsForSkip, device-функции loopCalculateDiscreteModel / _int со
//                счётчиками циклов, CPU-порт cpu_loop_model). Потолка нет — рабочая версия.
//   ..._int    — только под settleBlocks = TT / NT: это число NT-БЛОКОВ, а не шагов, оно живёт в
//                host-side счётчиках и на порядки меньше. INT_MAX здесь недостижим на осмысленных
//                входах, но каст всё равно идёт через проверку, а не вслепую.
 size_t steps_from_time_size_t(double t, double h)
;


 int steps_from_time_int(double t, double h)
;


// Порт loopCalculateDiscreteModel_int (cudaLibrary.cu) на CPU. Отличие одно: там размерность —
// compile-time AMOUNTOFX, здесь она приходит параметром (в NVRTC-сборке AMOUNTOFX как раз
// раскрывается в amountOfX, т.е. численно это одно и то же).
// Возврат — REGIME_* из configCUDA.h (1 = oscillation, -1 = fixed point, 0 = unbound). Проверка
// расходимости — раз в kCheckInterval итераций, как на GPU.
int cpu_loop_model(KrsCpuStep::StepFn step,
                   numb* x, const numb* a, numb h,
                   size_t iterations, int amountOfX, int preScaller,
                   int writableVar, numb maxValue,
                   numb* data)
;


// Порт peakFinder (cudaLibrary.cu) на один блок данных. Индексация здесь блок-локальная
// (startDataIndex == 0): CPU-ветка обрабатывает точки свипа потоково и держит один блок, а не
// матрицу nPts x record_steps. Логика — построчная копия, включая параболическую интерполяцию,
// сдвиг пиков влево на один и пересчёт timeOfPeaks в межпиковые интервалы.
int cpu_peak_finder(const numb* data, size_t amountOfPoints,
                    numb* outPeaks, numb* timeOfPeaks, numb h)
;


// Генератор случайных чисел — побайтовая копия stub'а из шаблонов
// kernels/lle1d.template.cu (splitmix64-инициализация + LCG). Копия, а не
// повторная реализация: направления начального возмущения обязаны совпадать
// с GPU-веткой, иначе первые точки λ разошлись бы из-за разного «щупа».
// Значение свипа в точке j continuation-цепочки. Одна формула на все три
// анализа (Bif/LLE/LS) и на оба устройства — GPU-ядра считают её же.
// reverse — идём от hi к lo (гистерезис), log_scale — логарифмическая сетка.
// Тело — общая с ядрами реализация (ucuda_node_value_cont в configCUDA.h):
// та же функция, которую после правки зовут kernels/*_cont.template.cu.
// Раньше здесь стояла копия в double, а ядро считало в numb — в double-режиме
// одно и то же, при numb = float расходились.
 double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) ;


// Request/Result — публичный интерфейс в double, а ядра работают в numb.
// Конвертация живёт ровно на границе: перед H2D сужаем, после D2H расширяем.
// Когда numb == double это no-op копия, поэтому обе конфигурации ведут себя
// одинаково и без #ifdef.
std::vector<numb> to_numb(const std::vector<double>& v) ;


struct CpuRandState { unsigned long long state; };


void cpu_curand_init(unsigned long long seed, unsigned long long sequence,
                     unsigned long long offset, CpuRandState* s) ;


float cpu_curand_uniform(CpuRandState* s) ;


// run_bif1d_continuation_cpu — CPU-двойник run_bif1d_continuation.
//
// Зачем: GPU-версия гоняет ВЕСЬ свип в ОДНОМ потоке (gridDim = blockDim = 1, см.
// bifurcation1d_cont.template.cu) — continuation последователен по построению, каждая точка
// стартует с конечного x[] предыдущей, и распараллелить это нельзя, не сломав гистерезис.
// Одиночный GPU-поток на зависимой FP64-цепочке — заведомо медленное железо.
// КРС компилируется в нативную функцию шага через krs_cpu.h: это работает и для custom, и для
// встроенных схем (req.krs_body в обоих случаях обычный C — у встроенных вывод codegen_scheme).
//
// Два отличия от GPU-двойника, оба в лучшую сторону: память — один блок траектории вместо
// матрицы nPts x record_steps (GPU держит её трижды: d_data + d_outPeaks + d_timeOfPeaks); отмена
// и прогресс работают на каждой точке свипа, а не только до запуска монолитного kernel'а.
Bifurcation1DResult run_bif1d_continuation_cpu(const Bifurcation1DRequest& req) ;


// run_bif1d_cpu — CPU-порт КЛАССИЧЕСКОГО (без continuation) 1D-свипа, двойник run_bif1d. Точки
// независимы: каждая стартует с одних и тех же НУ и базовых параметров, цепочки здесь нет.
// Флаги повторяют связку calculateDiscreteModelCUDA + peakFinderCUDA: kernel пишет
// checker = REGIME_* транзиента, основной участок считается только при OSCILLATION или
// FIXED_POINT, а peakFinderCUDA пропускает FP и UNBOUND, оставляя их коды в массиве.
// Continuation-ветка теперь ведёт себя так же (раньше метила unbound как -1).
// Свипуемая величина — параметр, НУ или сам шаг h, как на GPU; сетка берётся из cont_sweep_value
// (reverse только у continuation), как у CPU-веток LLE/LS/DFT.
Bifurcation1DResult run_bif1d_cpu(const Bifurcation1DRequest& req) ;


std::string exe_dir() ;


std::string read_text_file(const std::string& path, std::string& err) ;


std::string replace_all(std::string s, const std::string& from, const std::string& to) ;


std::string hash_key(const std::string& krs_body, int amountOfX) ;


// Блок #define'ов, дописываемый ПЕРЕД текстом виртуального configCUDA.h (тот оборачивает свои
// дефолты в #ifndef). Один инжект покрывает все 14 шаблонов — они тянут configCUDA.h транзитивно
// через cudaLibrary.cuh. Приведение к (numb) внутри макроса безопасно: разворачивается он только
// в местах использования (cudaLibrary.cu), т.е. заведомо после typedef numb.
std::string peak_config_defines() ;


std::string cu_err(CUresult r) ;


// Локальная host-копия __device__ __host__ getValueByIdx из cudaLibrary.cu:1266.
// Для 1D-бифуркации valueNumber всегда 0, поэтому просто линейная интерполяция.
// Нужна для расчёта значения параметра в CSV (host-side).
 double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) ;


// Host-копия getValueByIdx_log из cudaLibrary.cu -- та же формула, что кернел
// реально использовал для этой точки при log_scale=true. Нужна, чтобы CSV
// (см. getValueByIdx_local) сообщал то самое значение параметра, что было
// просимулировано, а не линейную интерполяцию поверх log-распределённой сетки.
 double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) ;


// run_lle1d_cpu — LLE(param) на CPU. Один код обслуживает два режима.
//
// continuation = false (классический свип): построчный порт LLEKernelCUDA — то, что GPU делает в
//   каждом потоке. Точки независимы: x сбрасывается на initial_conditions, щуп берётся из RNG с
//   subsequence = индекс точки (ровно как idx в kernel'е), прогревается transient_time, дальше
//   цикл Бенеттина. Нужен для сверки CPU-ядра с GPU и для счёта без GPU.
//
// continuation = true: точки выстроены в цепочку, переносится не только траектория x[], но и САМ
//   щуп y[] — к концу точки он уже развёрнут вдоль направления максимального растяжения, и
//   следующей точке не надо разворачивать его заново из случайного направления, отчего кривая
//   lambda(param) выходит глаже. GPU-двойника у режима нет и не будет: цепочка последовательна по
//   построению и выродилась бы в один поток. Прогрев отрабатывается на КАЖДОЙ точке, но начиная
//   со второй идёт NT-блоками с перенормировкой — щуп остаётся прицепленным и не теряет
//   ориентацию, поэтому transient_time сохраняет прежний смысл, а результаты сравнимы с классикой.
//
// Внутри точки алгоритм в обоих режимах одинаков: Бенеттин с одним вектором возмущения,
// перенормировка каждые NT единиц времени, lambda = sum(log(|dX|/eps)) / tMax.
LLE1DResult run_lle1d_cpu(const LLE1DRequest& req, bool continuation) ;


// Оконная функция для DFT. Все варианты — косинусные суммы
//   w(n) = a0 - a1*cos(g) + a2*cos(2g) - a3*cos(3g),   g = 2*pi*n/(N-1),
// так что таблицы коэффициентов хватает на все типы:
//   0 = None (rectangular)
//   1 = Hanning (default)        -31 дБ, спад -18 дБ/окт
//   2 = Hamming                  -43 дБ, спад  -6 дБ/окт
//   3 = Blackman                 -58 дБ, спад -18 дБ/окт
//   4 = Blackman-Harris (4 чл.)  -92 дБ, спад  -6 дБ/окт
// Уровень боковых лепестков — это и есть динамический диапазон: на Hanning'е
// субгармоники слабее -31 дБ тонут в утечке от основной частоты, что в каскаде
// удвоений периода отрезает всё после третьего-четвёртого удвоения.
//
// Окно нормируется на единичное среднее. Все четыре пути DFT (DFT_custom,
// cpu_dft_block и оба template-ядра) делят сумму на длину блока, а не на
// sum(w), поэтому без нормировки абсолютная амплитуда спектра зависела бы от
// выбора окна — когерентное усиление тут 1.0 / 0.5 / 0.54 / 0.42 / 0.36.
// Форма спектра от нормировки не зависит.
//
// Дублируется в dft_window (dft1d_cont.template.cu) и dft_hsweep_window
// (dft1d_hsweep.template.cu): при h-свипе и в continuation-режиме длина блока
// своя в каждой точке, и окно приходится строить прямо на GPU.
void cpu_build_window(std::vector<numb>& out, int sizeOfBlock, int window_type) ;


// Порт DFT_custom (cudaLibrary.cu) на один блок. Рекуррентный поворот вектора (cos_n, sin_n)
// вместо cos/sin на каждой итерации, со сбросом на точное значение раз в RESET_INTERVAL — как на
// GPU, включая float-точность в самом сбросе (там cosf/sinf).
void cpu_dft_block(const numb* data, int sizeOfBlock, const numb* window,
                   int nFreq, numb freq_lo, numb freq_hi, bool logFreqAxis,
                   numb h, numb* akcos, numb* bksin)
;


// Порты projectionOperator и gramSchmidtProcess (cudaLibrary.cu:2832 и :2848).
// Оригиналы объявлены __device__ __host__ и уже работают с runtime-размерностью,
// но живут в .cu под nvcc — из обычного .cpp их не подключить, поэтому копия.
void cpu_projection_operator(const numb* a, const numb* b, numb* minuend, int n) ;


// a — входные векторы (n штук по n компонент, row-major), b — ортонормированный
// результат. denominators (опц.) получает длины ДО нормировки — из них LS и
// собирает показатели.
void cpu_gram_schmidt(const numb* a, numb* b, int n, numb* denominators = nullptr) ;


// run_ls1d_cpu — спектр Ляпунова LS(param) на CPU, оба режима (см. run_lle1d_cpu).
// Внутри точки — построчный порт LSKernelCUDA: Бенеттин с N векторами возмущения и
// ортогонализацией Грама-Шмидта каждые NT единиц времени,
// lambda_k = sum(log(denominators[k]/eps)) / tMax.
// В continuation переносятся траектория x[] и ВСЕ N щупов y[] — к концу точки они уже выстроены
// вдоль собственных направлений растяжения. Отличие от LLE только в размере состояния (там один
// щуп, здесь N плюс ортогонализация), поэтому точка дороже примерно в N раз.
LS1DResult run_ls1d_cpu(const LS1DRequest& req, bool continuation) ;


// run_dft1d_cpu — 1D DFT на CPU, оба режима (см. run_lle1d_cpu).
// continuation = false: точки независимы, x сбрасывается на initial_conditions (при IC-свипе
//   свипуемая координата подменяется). continuation = true: x[] переносится с предыдущей точки,
//   как в Bifurcation1D — тангенциальных векторов здесь нет.
// Внутри точки: transient, запись блока writable_var, окно, DFT (cpu_dft_block — порт
// DFT_custom). При h-свипе число сэмплов и окно пересчитываются под шаг точки; буфер выделен под
// худший случай.
Dft1DResult run_dft1d_cpu(const Dft1DRequest& req, bool continuation) ;
}
