#pragma once
// adaptive_settings.h — настройки адаптивного шага одной расчётной вкладки
// (блок Integration) и их перевод в параметры ядра UcudaAdaptParams.
//
// Fixed (enabled == false) — прежний путь без изменений. Adaptive работает
// только для схем со встроенной оценкой ошибки (scheme_supports_adaptive).
// Сетка вывода: равномерная (плотный выход; шаг вывода — поле h вкладки,
// decimator прореживает её как раньше) или узлы самого шага (raw_nodes).

#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "configCUDA.h"
#include "codegen.hpp"
#include "num_parse.h"
#define UCUDA_ADAPT_LAYOUT_ONLY
#include "kernels/ucuda_adaptive.cuh"
#undef UCUDA_ADAPT_LAYOUT_ONLY

struct AdaptiveSettings {
    bool        enabled   = false;          // Fixed / Adaptive
    bool        raw_nodes = false;          // сетка вывода: равномерная / узлы шага
    std::string rtol      = "1e-8";
    std::string atol      = "1e-10";        // одно число или по переменной через запятую
    std::string h0;                         // пусто — автоматически
    std::string hmin;                       // пусто — max(10 ulp(t), 1e-12 * длина интервала)
    std::string hmax;                       // пусто — без ограничения
    std::string max_rej;                    // отказов подряд до шага с h_min; пусто — без предела,
                                            //   0 — без повторов (каждая попытка принимается)
    std::string ctrl      = "Hairer";       // регулятор (встроенный или из библиотеки)
    std::vector<std::string> ctrl_params;   // пусто — значения по умолчанию для схемы
    int         peak_interp = 2;            // сырые узлы: 0 — узел, 1 — парабола, 2 — Эрмит по f
    std::string max_points = "1000000";     // Analysis, сырые узлы: потолок узлов на траекторию
    int         lyap_renorm = 0;            // LLE/LS: 0 — ровно в k*NT, 1 — в первом узле после NT
    bool        minmax_interp_fixed = false;// Metrics: min/max по интерполированным экстремумам и в Fixed
};

inline bool operator==(const AdaptiveSettings& a, const AdaptiveSettings& b) {
    return a.enabled == b.enabled && a.raw_nodes == b.raw_nodes && a.rtol == b.rtol
        && a.atol == b.atol && a.h0 == b.h0 && a.hmin == b.hmin && a.hmax == b.hmax
        && a.max_rej == b.max_rej
        && a.ctrl == b.ctrl && a.ctrl_params == b.ctrl_params && a.peak_interp == b.peak_interp
        && a.max_points == b.max_points && a.lyap_renorm == b.lyap_renorm
        && a.minmax_interp_fixed == b.minmax_interp_fixed;
}
inline bool operator!=(const AdaptiveSettings& a, const AdaptiveSettings& b) { return !(a == b); }

// Схемы со встроенной оценкой ошибки — по имени, как их пишут сессии и UI.
inline bool adaptive_scheme_name_ok(const std::string& scheme) {
    return scheme == "RK45" || scheme == "DOP853" || scheme == "DOPRI78" || scheme == "DOPRI78 (legacy)";
}
// Порядок оценщика по имени схемы (нужен значениям по умолчанию регулятора Хайрера).
inline int adaptive_scheme_q(const std::string& scheme) {
    return scheme == "RK45" ? 4 : 7;
}

// ---- Встроенные регуляторы ------------------------------------------------------
struct AdaptiveCtrlInfo {
    const char* name;                    // имя в UI и в сессии
    int         id;                      // UCUDA_CTRL_*
    int         npar;
    const char* par[UCUDA_CTL_NPAR];     // имена параметров
    const char* tip;                     // подсказка
};

inline const AdaptiveCtrlInfo* adaptive_builtin_ctrls(int* count) {
    static const AdaptiveCtrlInfo k[] = {
        { "Hairer", UCUDA_CTRL_HAIRER, 5, { "safe", "fac1", "fac2", "beta", "kbeta" },
          "Hairer's dopri5 / dop853 controller (PI with facold^beta),\n"
          "Hairer's initial step and end-point rule (x + 1.01h > xend).\n"
          "Accepts err <= 1. Defaults follow the scheme: dopri5 for RK45,\n"
          "dop853 for the 8th-order methods." },
        { "SciPy", UCUDA_CTRL_SCIPY, 3, { "safety", "min_factor", "max_factor" },
          "scipy.integrate RK45 / DOP853 controller: elementary,\n"
          "scipy's initial step, accepts err < 1, no growth after a rejection." },
        { "I", UCUDA_CTRL_I, 3, { "safety", "facmin", "facmax" },
          "Elementary (integral) controller:\n"
          "h_new = h * clamp(safety * err^(-1/(q+1)), facmin, facmax)." },
        { "PI", UCUDA_CTRL_PI, 5, { "safety", "facmin", "facmax", "kI", "kP" },
          "Gustafsson PI controller:\n"
          "h_new = h * (th/err)^(kI/k) * (err_prev/err)^(kP/k), k = q+1,\n"
          "th = safety^k: the error it settles to (as the elementary controller)." },
        { "Filter", UCUDA_CTRL_FILTER, 9,
          { "safety", "facmin", "facmax", "b1", "b2", "b3", "a2", "a3", "limiter" },
          "Soderlind digital filter:\n"
          "rho = (th/e_n)^(b1/k) (th/e_n-1)^(b2/k) (th/e_n-2)^(b3/k) rho_n-1^(-a2) rho_n-2^(-a3),\n"
          "k = q+1, th = safety^k (target error); limiter = 1 smooths with 1 + atan(rho - 1).\n"
          "Defaults: H211b (b = 4)." },
    };
    if (count) *count = (int)(sizeof(k) / sizeof(k[0]));
    return k;
}

inline const AdaptiveCtrlInfo* adaptive_find_builtin(const std::string& name) {
    int n = 0;
    const AdaptiveCtrlInfo* k = adaptive_builtin_ctrls(&n);
    for (int i = 0; i < n; ++i) if (name == k[i].name) return &k[i];
    return nullptr;
}

// ---- Библиотека пользовательских регуляторов ------------------------------------
// Одна на приложение (файл step_controllers.json в каталоге library, загрузка и запись —
// session_io). Регулятор выбирается по имени (AdaptiveSettings::ctrl), имена встроенных
// заняты. Два вида записей:
//   C body — тело функции
//              void ucuda_ctrl_custom(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o)
//            (UCUDA_CTRL_CUSTOM). На GPU NVRTC собирает его вместе с ядром, на CPU — cl.exe
//            (krs_cpu). Вход, память и выход — структуры kernels/ucuda_adaptive.cuh; там же
//            помощники ucuda_ctl_err (норма ошибки), ucuda_ctl_pow, ucuda_clamp. Параметры
//            c[] — со своими именами и значениями по умолчанию (они же — оси свипа);
//   filter — именованный набор из 9 параметров встроенного фильтра Сёдерлинда (Filter):
//            без кода и без компиляции.
enum AdaptiveUserCtrlKind { kUserCtrlBody = 0, kUserCtrlFilter = 1 };

// Пользовательский регулятор, собранный для CPU (CtrlCpuFn в krs_cpu.h).
typedef void (*UcudaCtrlCustomFn)(const UcudaCtlIn*, UcudaCtlMem*, UcudaCtlOut*);

struct AdaptiveUserCtrl {
    std::string name;
    int         kind = kUserCtrlBody;
    std::string body;                      // C body
    std::vector<std::string> par_names;    // C body: имена c[0..], не больше UCUDA_CTL_NPAR
    std::vector<std::string> par_values;   // значения по умолчанию (filter — 9 чисел Filter)
    int         hairer_rules = 0;          // C body: h0 и последний шаг — 0 как в scipy, 1 как у Хайрера
    std::string tip;                       // описание (подсказка в списке)
};

inline bool operator==(const AdaptiveUserCtrl& a, const AdaptiveUserCtrl& b) {
    return a.name == b.name && a.kind == b.kind && a.body == b.body && a.par_names == b.par_names
        && a.par_values == b.par_values && a.hairer_rules == b.hairer_rules && a.tip == b.tip;
}
inline bool operator!=(const AdaptiveUserCtrl& a, const AdaptiveUserCtrl& b) { return !(a == b); }

// Записи правит только UI-поток (под mu); расчётные потоки берут копию записи по имени
// (adaptive_find_user_ctrl), поэтому правка во время расчёта его не задевает.
struct AdaptiveCtrlLibrary {
    std::mutex                    mu;
    std::vector<AdaptiveUserCtrl> items;
    std::string                   path;    // пусто — не загружена, записывать некуда
};

inline AdaptiveCtrlLibrary& adaptive_ctrl_library() {
    static AdaptiveCtrlLibrary lib;
    return lib;
}

inline bool adaptive_find_user_ctrl(const std::string& name, AdaptiveUserCtrl* out) {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::lock_guard<std::mutex> lk(L.mu);
    for (const AdaptiveUserCtrl& u : L.items)
        if (u.name == name) { if (out) *out = u; return true; }
    return false;
}

// Шаблоны новых записей: встроенные Hairer и SciPy, выписанные C-телом (отправная точка
// для своих), и известные фильтры Сёдерлинда (2003) с их полюсами.
struct AdaptiveCtrlPreset {
    const char* name;
    int         kind;
    const char* pars;      // C body: имена через запятую
    const char* values;    // значения по умолчанию через запятую
    int         hairer_rules;
    const char* tip;
    const char* body;
};

inline const AdaptiveCtrlPreset* adaptive_ctrl_presets(int* count) {
    static const AdaptiveCtrlPreset k[] = {
        { "Hairer (C)", kUserCtrlBody, "safe, fac1, fac2, beta, kbeta", "0.9, 0.333, 6, 0, 0.2", 1,
          "Hairer's dopri5 / dop853 controller written out as a C body\n"
          "(the built-in \"Hairer\"; defaults of dop853, for dopri5: 0.9, 0.2, 10, 0.04, 0.75).",
          "// Hairer's dopri5 / dop853 step-size control (PI with facold^beta).\n"
          "// c[0] safe, c[1] fac1, c[2] fac2, c[3] beta, c[4] kbeta; m.user[0] keeps facold.\n"
          "const numb safe = in.c[0], fac1 = in.c[1], fac2 = in.c[2];\n"
          "const numb beta = in.c[3], kbeta = in.c[4];\n"
          "const numb err = ucuda_ctl_err(in);          // RMS norm, 1 = on the tolerance\n"
          "const numb facold = m.nacc > 0 ? m.user[0] : (numb)1e-4;\n"
          "const numb expo1 = (numb)1 / (in.q + 1) - kbeta * beta;\n"
          "const numb fac11 = ucuda_ctl_pow(err, expo1);\n"
          "numb fac = fac11 / ucuda_ctl_pow(facold, beta);\n"
          "fac = fmax(1 / fac2, fmin(1 / fac1, fac / safe));\n"
          "o.err = err;\n"
          "if (err <= 1) {\n"
          "    o.accept = 1;\n"
          "    m.user[0] = fmax(err, (numb)1e-4);\n"
          "    numb hnew = in.h / fac;\n"
          "    if (in.hmax > 0 && hnew > in.hmax) hnew = in.hmax;\n"
          "    if (in.nrej > 0) hnew = fmin(hnew, in.h);   // no growth right after a rejection\n"
          "    o.h = hnew;\n"
          "} else {\n"
          "    o.accept = 0;\n"
          "    o.h = in.h / fmin(1 / fac1, fac11 / safe);\n"
          "}\n" },
        { "SciPy (C)", kUserCtrlBody, "safety, min_factor, max_factor", "0.9, 0.2, 10", 0,
          "scipy.integrate RK45 / DOP853 controller written out as a C body\n"
          "(the built-in \"SciPy\").",
          "// scipy.integrate RungeKutta._step_impl: elementary controller.\n"
          "// c[0] safety, c[1] min_factor, c[2] max_factor.\n"
          "const numb safety = in.c[0], min_factor = in.c[1], max_factor = in.c[2];\n"
          "const numb err = ucuda_ctl_err(in);          // RMS norm, 1 = on the tolerance\n"
          "const numb expo = (numb)-1 / (in.q + 1);\n"
          "o.err = err;\n"
          "if (err < 1) {\n"
          "    numb factor = err == 0 ? max_factor : fmin(max_factor, safety * ucuda_ctl_pow(err, expo));\n"
          "    if (in.nrej > 0) factor = fmin((numb)1, factor);   // no growth right after a rejection\n"
          "    o.h = in.h * factor;\n"
          "    o.accept = 1;\n"
          "} else {\n"
          "    o.h = in.h * fmax(min_factor, safety * ucuda_ctl_pow(err, expo));\n"
          "    o.accept = 0;\n"
          "}\n" },
        { "H211b", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.25, 0.25, 0, 0.25, 0, 1", 0,
          "Soderlind H211b, b = 4: kbeta = (1/4, 1/4), alpha2 = 1/4. Poles 0, 1/2.", "" },
        { "H0211", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.5, 0.5, 0, 0.5, 0, 1", 0,
          "Soderlind H0211 (H211b with b = 2): deadbeat, poles 0, 0.", "" },
        { "H211PI", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.1666666666666667, 0.1666666666666667, 0, 0, 0, 1", 0,
          "Soderlind H211PI: kbeta = (1/6, 1/6). Poles 1/2, 1/3.", "" },
        { "H312b", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.125, 0.25, 0.125, 0.375, 0.125, 1", 0,
          "Soderlind H312b, b = 8: kbeta = (1, 2, 1)/8, alpha = (3, 1)/8. Poles 0, 0, 1/2.", "" },
        { "H0312", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.25, 0.5, 0.25, 0.75, 0.25, 1", 0,
          "Soderlind H0312 (H312b with b = 4): deadbeat, poles 0, 0, 0.", "" },
        { "H312PID", kUserCtrlFilter, "",
          "0.9, 0.2, 5, 0.05555555555555556, 0.1111111111111111, 0.05555555555555556, 0, 0, 1", 0,
          "Soderlind H312PID: kbeta = (1/18, 1/9, 1/18).", "" },
        { "H321", kUserCtrlFilter, "",
          "0.9, 0.2, 5, 0.3333333333333333, 0.05555555555555556, -0.2777777777777778, -0.8333333333333334, -0.1666666666666667, 1", 0,
          "Soderlind H321: kbeta = (1/3, 1/18, -5/18), alpha = (-5/6, -1/6)\n"
          "in the sign convention of the Filter formula. Poles 1/3, 1/2, 2/3.", "" },
        { "H0321", kUserCtrlFilter, "", "0.9, 0.2, 5, 1.25, 0.5, -0.75, -0.25, -0.75, 1", 0,
          "Soderlind H0321: kbeta = (5/4, 1/2, -3/4), alpha = (-1/4, -3/4). Deadbeat.", "" },
        { "PI3040", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.7, -0.4, 0, 0, 0, 1", 0,
          "PI.3.4 (Gustafsson): kbeta = (0.7, -0.4). Poles 0.8, -0.5.", "" },
        { "PI4020", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.6, -0.2, 0, 0, 0, 1", 0,
          "PI.4.2: kbeta = (0.6, -0.2).", "" },
        { "PI3333", kUserCtrlFilter, "", "0.9, 0.2, 5, 0.6666666666666666, -0.3333333333333333, 0, 0, 0, 1", 0,
          "PI.3.3: kbeta = (2/3, -1/3).", "" },
    };
    if (count) *count = (int)(sizeof(k) / sizeof(k[0]));
    return k;
}

// Список через запятую -> элементы без крайних пробелов (пустой текст -> пустой список).
inline std::vector<std::string> adaptive_split_list(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto push = [&]() {
        size_t b = 0, e = cur.size();
        while (b < e && std::isspace((unsigned char)cur[b])) ++b;
        while (e > b && std::isspace((unsigned char)cur[e - 1])) --e;
        out.push_back(cur.substr(b, e - b));
        cur.clear();
    };
    bool any = false;
    for (char c : text) { if (!std::isspace((unsigned char)c)) any = true; if (c == ',') push(); else cur += c; }
    if (any) push();
    else out.clear();
    return out;
}

inline AdaptiveUserCtrl adaptive_ctrl_from_preset(const AdaptiveCtrlPreset& p) {
    AdaptiveUserCtrl u;
    u.name = p.name; u.kind = p.kind; u.body = p.body; u.hairer_rules = p.hairer_rules; u.tip = p.tip;
    u.par_names  = adaptive_split_list(p.pars);
    u.par_values = adaptive_split_list(p.values);
    return u;
}

// Значения параметров по умолчанию. q — порядок оценщика схемы: у Хайрера
// dopri5 (q = 4) и dop853 (q = 7) настроены по-разному.
inline std::vector<double> adaptive_ctrl_defaults(int id, int q) {
    switch (id) {
    case UCUDA_CTRL_HAIRER:
        return q <= 4 ? std::vector<double>{ 0.9, 0.2, 10.0, 0.04, 0.75 }
                      : std::vector<double>{ 0.9, 0.333, 6.0, 0.0, 0.2 };
    case UCUDA_CTRL_SCIPY:  return { 0.9, 0.2, 10.0 };
    case UCUDA_CTRL_I:      return { 0.9, 0.2, 5.0 };
    case UCUDA_CTRL_PI:     return { 0.9, 0.2, 5.0, 0.3, 0.4 };
    case UCUDA_CTRL_FILTER: return { 0.9, 0.2, 5.0, 0.25, 0.25, 0.0, 0.25, 0.0, 1.0 };
    default:                return {};
    }
}

// Регулятор по имени: встроенный или из библиотеки — всё, что нужно UI и сборке.
struct AdaptiveCtrlResolved {
    int  id = UCUDA_CTRL_HAIRER;          // UCUDA_CTRL_*
    bool user = false;                    // из библиотеки
    std::vector<std::string> par;         // имена параметров c[]
    std::vector<double>      def;         // значения по умолчанию (для схемы с порядком q)
    std::string body;                     // C body (id == UCUDA_CTRL_CUSTOM)
    bool hairer_rules = false;            // h0 и последний шаг по Хайреру
    std::string tip;
};

inline bool adaptive_resolve_ctrl(const std::string& name, int q, AdaptiveCtrlResolved& r, std::string& err) {
    r = AdaptiveCtrlResolved();
    if (const AdaptiveCtrlInfo* ci = adaptive_find_builtin(name)) {
        r.id = ci->id;
        for (int i = 0; i < ci->npar; ++i) r.par.push_back(ci->par[i]);
        r.def = adaptive_ctrl_defaults(ci->id, q);
        r.hairer_rules = ci->id == UCUDA_CTRL_HAIRER;
        r.tip = ci->tip;
        return true;
    }
    AdaptiveUserCtrl u;
    if (!adaptive_find_user_ctrl(name, &u)) {
        err = "unknown step controller '" + name + "' (not in the controller library)";
        return false;
    }
    r.user = true;
    r.tip = u.tip;
    if (u.kind == kUserCtrlFilter) {
        const AdaptiveCtrlInfo* fi = adaptive_find_builtin("Filter");
        r.id = UCUDA_CTRL_FILTER;
        for (int i = 0; i < fi->npar; ++i) r.par.push_back(fi->par[i]);
        r.def = adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, q);
    } else {
        r.id = UCUDA_CTRL_CUSTOM;
        r.body = u.body;
        r.hairer_rules = u.hairer_rules != 0;
        if (u.par_names.size() > (size_t)UCUDA_CTL_NPAR) {
            err = "controller '" + name + "': at most " + std::to_string(UCUDA_CTL_NPAR) + " parameters";
            return false;
        }
        r.par = u.par_names;
        r.def.assign(u.par_names.size(), 0.0);
    }
    for (size_t i = 0; i < u.par_values.size() && i < r.def.size(); ++i)
        if (!parse_num_checked(u.par_values[i], r.def[i])) {
            err = "controller '" + name + "': default of " + r.par[i] + " is not a number";
            return false;
        }
    return true;
}

// Текст пользовательского регулятора для модуля ядра: ставится между раскладкой
// ucuda_adaptive.cuh (UCUDA_ADAPT_LAYOUT_ONLY) и его драйвером. Пустое тело — пустой
// текст (встроенные регуляторы). line_directive — нумерация ошибок в координатах тела
// (только когда за функцией ничего нет: #line действует до конца файла).
inline std::string adaptive_ctrl_source(const std::string& body, bool line_directive = false) {
    if (body.empty()) return std::string();
    std::string s = "#define UCUDA_HAS_CUSTOM_CTRL 1\n"
                    "__device__ __host__ __forceinline__ void ucuda_ctrl_custom(const UcudaCtlIn& in, "
                    "UcudaCtlMem& m, UcudaCtlOut& o) {\n"
                    "    (void)in; (void)m; (void)o;\n";
    if (line_directive) s += "#line 1 \"controller\"\n";
    s += body;
    s += "\n}\n";
    return s;
}

// Разбор atol: одно значение на все переменные или ровно n через запятую.
inline bool adaptive_parse_atol(const std::string& text, int n, std::vector<double>& out, std::string& err) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : text) {
        if (c == ',' || c == ';') { parts.push_back(cur); cur.clear(); }
        else cur += c;
    }
    parts.push_back(cur);
    out.clear();
    for (const std::string& p : parts) {
        double v = 0;
        if (!parse_num_checked(p, v) || !(v >= 0)) { err = "atol: '" + p + "' is not a non-negative number"; return false; }
        out.push_back(v);
    }
    if (out.size() == 1) { out.assign((size_t)n, out[0]); return true; }
    if ((int)out.size() != n) {
        err = "atol: give one value or one per variable (" + std::to_string(n) + ")";
        return false;
    }
    return true;
}

// Необязательное положительное поле: пусто — 0 (значит «по умолчанию»).
inline bool adaptive_parse_opt(const std::string& text, const char* name, double& out, std::string& err) {
    out = 0;
    bool blank = true;
    for (char c : text) if (!std::isspace((unsigned char)c)) { blank = false; break; }
    if (blank) return true;
    if (!parse_num_checked(text, out) || !(out > 0)) { err = std::string(name) + " must be empty or > 0"; return false; }
    return true;
}

// Собирает параметры ядра. code — AdaptiveCode схемы (q, nlo, счётчики f),
// n — размерность, span — длина всего интервала интегрирования (нужна h0).
// ctrl_body — тело пользовательского регулятора (C body из библиотеки), иначе пусто:
// его нужно вкомпилировать в ядро (adaptive_ctrl_source).
inline bool adaptive_build_params(const AdaptiveSettings& s, const AdaptiveCode& code, int n,
                                  double span, UcudaAdaptParams& P, std::string& err,
                                  std::string* ctrl_body = nullptr) {
    if (ctrl_body) ctrl_body->clear();
    std::memset(&P, 0, sizeof(P));
    if (n < 1 || n > UCUDA_AD_MAXN) { err = "adaptive step: dimension out of range"; return false; }
    double rtol = 0;
    if (!parse_num_checked(s.rtol, rtol) || !(rtol >= 0)) { err = "rtol must be a number >= 0"; return false; }
    std::vector<double> atol;
    if (!adaptive_parse_atol(s.atol, n, atol, err)) return false;
    bool any_tol = rtol > 0;
    for (double v : atol) any_tol |= v > 0;
    if (!any_tol) { err = "rtol and atol cannot both be zero"; return false; }
    double h0 = 0, hmin = 0, hmax = 0;
    if (!adaptive_parse_opt(s.h0, "h0", h0, err)) return false;
    if (!adaptive_parse_opt(s.hmin, "h_min", hmin, err)) return false;
    if (!adaptive_parse_opt(s.hmax, "h_max", hmax, err)) return false;
    if (hmin > 0 && hmax > 0 && hmin > hmax) { err = "h_min > h_max"; return false; }
    double maxrej = 0;
    bool maxrej_blank = true;
    for (char ch : s.max_rej) if (!std::isspace((unsigned char)ch)) { maxrej_blank = false; break; }
    if (!maxrej_blank && (!parse_num_checked(s.max_rej, maxrej) || !(maxrej >= 0) || maxrej > 1e9
                          || (double)(long long)maxrej != maxrej)) {
        err = "max rejects must be empty or a whole number >= 0";
        return false;
    }

    AdaptiveCtrlResolved cr;
    if (!adaptive_resolve_ctrl(s.ctrl, code.q, cr, err)) return false;
    std::vector<double> c = cr.def;
    if (!s.ctrl_params.empty()) {
        if (s.ctrl_params.size() != cr.par.size()) { err = "controller parameters: wrong count"; return false; }
        for (size_t i = 0; i < cr.par.size(); ++i)
            if (!parse_num_checked(s.ctrl_params[i], c[i])) {
                err = "controller parameter " + cr.par[i] + " is not a number";
                return false;
            }
    }
    if (cr.id == UCUDA_CTRL_CUSTOM) {
        bool blank = true;
        for (char ch : cr.body) if (!std::isspace((unsigned char)ch)) { blank = false; break; }
        if (blank) { err = "controller '" + s.ctrl + "' has an empty body"; return false; }
        if (ctrl_body) *ctrl_body = cr.body;
    }

    P.rtol = rtol;
    for (int i = 0; i < n; ++i) P.atol[i] = atol[i];
    P.h0 = h0; P.hmin = hmin; P.hmax = hmax; P.span = span;
    P.maxrej = maxrej_blank ? 0 : (maxrej == 0 ? UCUDA_AD_NO_RETRY : (int)maxrej);
    for (size_t i = 0; i < c.size() && i < UCUDA_CTL_NPAR; ++i) P.c[i] = c[i];
    P.q = code.q; P.nlo = code.nlo; P.ctrl = cr.id;
    P.clip   = cr.hairer_rules ? 1 : 0;
    P.h0mode = cr.hairer_rules ? 1 : 0;
    P.emb_rhs = code.emb_rhs; P.dprep_rhs = code.dprep_rhs;
    return true;
}

// Статистика траектории (копия UcudaAdaptStats для хоста и отрисовки).
struct AdaptiveStats {
    double nacc = 0, nrej = 0, nforced = 0, nrhs = 0;
    double hmin = 0, hmax = 0, hmean = 0;
    bool   diverged = false;
    bool   truncated = false;   // сырые узлы: упёрлись в max_points раньше конца
};
