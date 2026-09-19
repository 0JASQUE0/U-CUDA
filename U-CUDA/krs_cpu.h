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
//   DD     — ucuda::dd, мантисса 106 бит (~32 цифры). То же тело КРС, тот же
//            компилятор, подменён только тип: eps падает с 2.2e-16 до 4.9e-32,
//            и оценка порядка перестаёт упираться в полку округления.
enum class KrsCpuPrec { Double = 0, DD = 1 };

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

    // Непустым будет ровно один из двух — тот, что отвечает точности сборки.
    StepFn   fn()    const { return fn_; }
    StepFnDD fn_dd() const { return fn_dd_; }
    explicit operator bool() const { return fn_ != nullptr || fn_dd_ != nullptr; }

private:
    void     release();
    void*    module_ = nullptr;   // HMODULE
    StepFn   fn_     = nullptr;
    StepFnDD fn_dd_  = nullptr;
};
