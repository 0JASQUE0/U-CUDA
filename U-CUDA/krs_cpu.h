#pragma once
#include <string>
#include <vector>
#include "configCUDA.h"      // typedef numb — CPU считает в той же точности, что GPU
#include "kernels/ucuda_hp.h" // double-double для режима расширенной точности

// CPU-исполнение пользовательских КРС (custom KRS).
//
// На GPU тело КРС — сырой C/CUDA, который NVRTC подставляет в
// calculateDiscreteModel. CPU-путь (integrator.cpp) работает на
// SystemEvaluator — интерпретаторе ВЫРАЖЕНИЙ правых частей; операторы C
// (объявления, циклы, ветвления, локальные массивы) он не понимает, а
// реальные схемы в library/ ими пользуются вовсю (см. Chua Matreshka).
//
// Здесь тело компилируется в нативную функцию шага тем же способом, каким
// GPU компилирует его в kernel: генерируем исходник, отдаём компилятору,
// получаем указатель. Компилятор — cl.exe из установленной Visual Studio
// (той же, которой собирается проект), находится через vswhere + vcvars64.
// Компилируем как C++, а не как C: тогда abs(double), bool/true/false и
// объявления в любом месте блока работают ровно как в CUDA.

// Точность арифметики скомпилированного шага.
//   Double — numb как везде в проекте (совпадает с GPU бит в бит);
//   DD     — ucuda::dd, мантисса 106 бит (~32 цифры), eps 4.9e-32;
//   QD     — ucuda::qd, мантисса 212 бит (~62 цифры), eps 1.2e-63.
// Тело КРС и компилятор во всех трёх случаях одни и те же, подменён только
// тип: полка округления в оценке порядка опускается вместе с eps.
enum class KrsCpuPrec { Double = 0, DD = 1, QD = 2 };

// Диагностика по телу КРС. line — 1-based номер строки В ТЕЛЕ (0 = не
// привязано к строке).
struct KrsCpuDiag {
    int         line = 0;
    std::string message;
};

// Статическая проверка обращений X[k] / a[k] с КОНСТАНТНЫМ индексом.
//   amountOfX      — число переменных системы; валидно X[0..amountOfX-1].
//   amountOfValues — размер массива a: a[0] — symmetry s, a[1..M] — параметры,
//                    т.е. (число параметров + 1).
// Индексы-выражения (X[i], k[i][j]) не проверяются — их значение известно
// только в рантайме. Токенизация идёт по ЦЕЛЫМ идентификаторам, поэтому
// X1[0] и a_param не считаются обращением к X / a: в реальных схемах они
// встречаются на каждом шагу, и наивный поиск "X[" врал бы.
// Возвращает true, если ошибок нет.
bool krs_cpu_check_indices(const std::string& body,
                           int amountOfX, int amountOfValues,
                           std::vector<KrsCpuDiag>& diags);

// Доступен ли CPU-бэкенд (найден ли компилятор). При false в why_not
// кладётся причина — GUI показывает её пользователю.
bool krs_cpu_backend_available(std::string* why_not = nullptr);

// Скомпилированное тело КРС. Владеет загруженной DLL; указатель из fn()
// валиден, пока жив объект.
class KrsCpuStep {
public:
    // Сигнатура совпадает с calculateDiscreteModel на GPU — включая тип numb:
    // тело обязано считаться в той же точности, иначе CPU и GPU разъедутся уже
    // на уровне типа, а не на уровне порядка операций.
    using StepFn = void (*)(numb* X, const numb* a, numb h);
    // Тот же шаг в расширенной точности. h передаётся УКАЗАТЕЛЕМ: dd — это
    // 16-байтовый агрегат, и его передача по значению через границу DLL
    // зависит от соглашения вызова, а не от типа. Указатель убирает вопрос.
    using StepFnDD = void (*)(ucuda::dd* X, const ucuda::dd* a, const ucuda::dd* h);
    using StepFnQD = void (*)(ucuda::qd* X, const ucuda::qd* a, const ucuda::qd* h);

    KrsCpuStep() = default;
    ~KrsCpuStep();
    KrsCpuStep(KrsCpuStep&&) noexcept;
    KrsCpuStep& operator=(KrsCpuStep&&) noexcept;
    KrsCpuStep(const KrsCpuStep&) = delete;
    KrsCpuStep& operator=(const KrsCpuStep&) = delete;

    // Проверяет индексы, генерирует исходник, компилирует, грузит DLL.
    // Ошибки (наши и компиляторские, с номерами строк тела) складываются в
    // diags. false -> fn() остаётся nullptr.
    // Результат компиляции кэшируется на диске по хэшу тела: повторный Run
    // с той же схемой берёт готовую DLL и не платит за компиляцию.
    bool compile(const std::string& body, int amountOfX, int amountOfValues,
                 std::vector<KrsCpuDiag>& diags);
    // То же, но с выбором точности. Кэш у режимов раздельный (точность входит
    // в ключ), поэтому переключение туда-обратно не пересобирает.
    bool compile(const std::string& body, int amountOfX, int amountOfValues,
                 KrsCpuPrec prec, std::vector<KrsCpuDiag>& diags);

    // Непустым будет ровно один из трёх — тот, что отвечает точности сборки.
    StepFn   fn()    const { return fn_; }
    StepFnDD fn_dd() const { return fn_dd_; }
    StepFnQD fn_qd() const { return fn_qd_; }
    explicit operator bool() const {
        return fn_ != nullptr || fn_dd_ != nullptr || fn_qd_ != nullptr;
    }

private:
    void     release();
    void*    module_ = nullptr;   // HMODULE
    StepFn   fn_     = nullptr;
    StepFnDD fn_dd_  = nullptr;
    StepFnQD fn_qd_  = nullptr;
};

// Пользовательский регулятор шага (C body из библиотеки регуляторов, adaptive_settings.h)
// для CPU-драйвера адаптивного шага: тело оборачивается в функцию над раскладкой
// kernels/ucuda_adaptive.cuh — ту же, что ucuda_ctrl_custom на GPU, — и собирается cl.exe
// в DLL (кэш на диске, как у КРС). Номера строк в diags — в координатах тела.
struct UcudaCtlIn;
struct UcudaCtlMem;
struct UcudaCtlOut;
class CtrlCpuFn {
public:
    using Fn = void (*)(const UcudaCtlIn*, UcudaCtlMem*, UcudaCtlOut*);
    CtrlCpuFn() = default;
    ~CtrlCpuFn();
    CtrlCpuFn(const CtrlCpuFn&) = delete;
    CtrlCpuFn& operator=(const CtrlCpuFn&) = delete;
    bool compile(const std::string& body, std::vector<KrsCpuDiag>& diags);
    Fn   fn() const { return fn_; }
private:
    void* module_ = nullptr;   // HMODULE
    Fn    fn_     = nullptr;
};

// Адаптивный шаг на CPU нативным кодом: адаптивные тела схемы (AdaptiveCode), регулятор
// (C body из библиотеки или пусто), kernels/adaptive_part.cu с драйвером ucuda_adaptive.cuh
// (вместе с разделом UCUDA_AD_LYAPUNOV) и PeakStream из cudaLibrary.cu собираются cl.exe в
// одну DLL (кэш на диске, как у КРС). Текст алгоритма — тот же, что у GPU-ядер, поэтому CPU
// и GPU сравнимы. Только double.
struct UcudaAdaptParams;
class AdaptiveCpuModule {
public:
    // y(T) от ic: y[n], stats[8] = nacc, nrej, nforced, nrhs, hmin, hmax, hmean, diverged.
    using EndpointFn = int (*)(const double* ic, const double* a, const UcudaAdaptParams* P, double T,
                               double* y, double* stats);
    // Свип LLE (ls = 0) / LS (ls = 1) цепочкой точек — ucuda_lyap_chain (continuation = 1)
    // или классически (0). result[nPts * NC] (NaN — разлёт), stats[nPts * 4].
    using LyapFn = void (*)(int ls, int continuation, int nPts, double lo, double hi, int reverse,
                            int logScale, int mutParamIdx, const double* baseValues, int amountOfValues,
                            const double* baseX, const UcudaAdaptParams* P, int axisKind, double tolRatio,
                            double tTr, double NT, int nBlocks, int nWarm, double eps, int renorm,
                            double maxValue, double* result, double* stats, const volatile int* cancel,
                            int* progress);
    // БД 1D (то же, что calculateDiscreteModelPeaksAdCUDA / ...AdContCUDA). Классика
    // (continuation = 0) — точки [i0, i1) сетки nPts, строки выходов — с i0 (строка
    // idx - i0); continuation — цепочка всех nPts точек (i0, i1 не читаются). flags — число
    // пиков или код режима, stats[4] на точку. Возвращает 0 при отмене.
    using BifFn = int (*)(int continuation, int i0, int i1, int nPts, double lo, double hi, int reverse,
                          int logScale, int sweepVar, int mutIdx, const double* baseValues, int amountOfValues,
                          const double* baseX, const UcudaAdaptParams* P, int axisKind, double tolRatio,
                          int writableVar, double maxValue, double* outPeaks, double* timeOfPeaks, int* flags,
                          unsigned long long peakStride, int peakCapacity, double transientTime, double tRec,
                          double dtOut, int preScaller, unsigned long long iters, int raw, int interp,
                          double* stats, const volatile int* cancel, int* progress);
    AdaptiveCpuModule() = default;
    ~AdaptiveCpuModule();
    AdaptiveCpuModule(const AdaptiveCpuModule&) = delete;
    AdaptiveCpuModule& operator=(const AdaptiveCpuModule&) = delete;
    // dprep / deval пустые — модуль без плотного выхода (UCUDA_AD_NO_DENSE) и без входа БД.
    // prelude — #define'ы перед configCUDA.h (настройки пиков: peak_config_defines движка).
    bool compile(const std::string& rhs, const std::string& emb, const std::string& dprep,
                 const std::string& deval, const std::string& ctrl_body, int amountOfX,
                 const std::string& prelude, std::vector<KrsCpuDiag>& diags);
    EndpointFn endpoint() const { return endpoint_; }
    LyapFn     lyap()     const { return lyap_; }
    BifFn      bif()      const { return bif_; }
private:
    void*      module_   = nullptr;   // HMODULE
    EndpointFn endpoint_ = nullptr;
    LyapFn     lyap_     = nullptr;
    BifFn      bif_      = nullptr;
};
