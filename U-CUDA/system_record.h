#pragma once
#include <string>
#include <vector>
#include <map>

// Именованная пользовательская КРС. Это "ещё один scheme" — встаёт в scheme
// combo рядом с Euler/Cromer/Midpoint/RK4. Имя не должно совпадать с built-in.
// body — сырой C/CUDA-текст, подставляется напрямую в NVRTC-шаблон вместо
// codegen-вывода. Доступны X[0..N-1], a[1..M], h, AMOUNTOFX, обычная математика.
struct CustomScheme {
    std::string name;
    std::string body;

    // Паспорт схемы — нужен ТОЛЬКО когда эта КРС берётся опорной для
    // экстраполяционной обёртки "Extr(<имя>|...)": из порядка и симметричности
    // считаются веса стадий. У встроенных схем то же самое отдаёт
    // builtin_scheme_traits, но про кастомную КРС кодоген не знает ничего —
    // текст тела он не разбирает, так что сказать может только автор.
    // Ошибиться не страшно: неверные order/symmetric не дают неправильных
    // чисел, они лишь гасят не те члены разложения, и порядок обёртки
    // опускается до порядка самой базы.
    // symmetric = разложение ошибки идёт только по ЧЁТНЫМ степеням h; это
    // верно для самосопряжённых схем (Phi* ∘ Phi), и каждая стадия тогда
    // добавляет не один порядок, а два.
    int  order     = 1;
    bool symmetric = false;
};

// Полная запись системы в библиотеке: весь ввод (для редактирования) +
// значения по умолчанию (все опциональны: пустая строка = не задано).
struct SystemRecord {
    // метаданные
    std::string name;            // имя (для списка); если пусто — автоимя при сохранении
    std::string note;            // заметка/ссылка на статью

    // ввод (для редактирования)
    // Discrete map x_{n+1} = f(x_n): no integration scheme, h is pinned to 1.
    bool        is_map = false;
    std::string mode;            // "Image" | "LaTeX" | "Plain"
    std::string latex_text;
    std::string plain_text;
    std::string alphabet_text;       // legacy: один список (vars + params вперемешку)
    // Явные списки переменных и параметров. Приоритет над alphabet_text:
    // если оба непустые — используются они напрямую; alphabet_text остаётся
    // у старых записей как legacy fallback (либо когда equations должны
    // сами вывести vars/params).
    std::string vars_text;
    std::string params_text;
    bool        use_aux_funcs = false;
    std::string func_defs_text;
    std::string param_order;     // "AsInAlphabet" | "AsInSystem"

    // выбранные методы
    bool scheme_euler    = false;
    bool scheme_cromer   = false;
    bool scheme_midpoint = false;
    bool scheme_rk4      = false;
    bool scheme_dopri78  = false;
    bool scheme_cd       = false;
    bool scheme_ccd      = false;   // Complex CD (комплексные полушаги)
    bool scheme_ccd4     = false;   // Complex CD4 (два CD с шагами gamma*h / conj)
    bool scheme_ieuler   = false;   // Implicit Euler (Ньютон по символьному якобиану)
    bool scheme_imidpoint = false;  // Implicit Midpoint (то же, стадия Y = (X + X_next)/2)
    bool scheme_semp     = false;   // SEMP (средняя точка, явная стадия на h1 = s*h)
    bool scheme_simp     = false;   // SIMP (то же, стадия диагонально-неявная)
    bool scheme_dmethod  = false;   // D (диагонально-неявный шаг на полный h)
    bool scheme_cieuler  = false;   // Complex Implicit Euler (два неявных Эйлера, tau = h*(1±i)/2)

    // Настройки Ньютона для двух неявных схем. Живут на уровне системы, а не
    // конфига анализа: AppModel::build_system() — единственная фабрика System,
    // и её результат копируется во все сессии, так что кодген видит их сам.
    // newton_full: false = якобиан и LU один раз за шаг (модифицированный Ньютон),
    // true = пересчёт на каждой итерации (полный Ньютон).
    bool        newton_full      = false;
    std::string newton_tol       = "1e-10";
    std::string newton_max_iters = "8";

    // Коэффициент симметрии s для CD-методов (передаётся в kernel как a[0]).
    // 0.5 = классический симметричный CD. Читается схемами "CD" и "Complex CD"
    // (а также пользовательскими КРС, чьё тело обращается к a[0]).
    std::string symmetry_s = "0.5";

    // Пользовательские именованные КРС (см. CustomScheme выше).
    std::vector<CustomScheme> custom_schemes;

    // Собранные пользователем обёртки — ТОЛЬКО имена вида "Extr(RK4|1,2,4)"
    // или "Comp(CD|g1,g2,g1)". Хранить рядом нечего: имя самоописательно, и тело
    // пересобирается резолвером при каждой правке системы (в отличие от
    // custom_schemes, где текст заморожен). Опорной может быть и встроенная
    // схема, и кастомная КРС из списка выше.
    std::vector<std::string> wrapper_schemes;

    // значения по умолчанию (всё опционально, пустое = не задано)
    std::string step_h;          // шаг дискретизации (строка, пустая = не задано)

    // начальные условия по имени переменной: "x" -> "1.0" (пустое = не задано)
    std::map<std::string, std::string> init_conditions;

    // значения параметров по имени: "sigma" -> "10" (пустое = не задано)
    std::map<std::string, std::string> param_values;
};
