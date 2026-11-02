#pragma once
// Редактор системы (вкладка Library): текст уравнений, схемы, построители Extr/Comp, редактор ДАУ,
// OCR, генерация кода и запись в библиотеку. Часть AppModel (AppModel : SystemEditor : AppSettings).
#include "ucuda/application/app_settings.h"
#include "ucuda/analysis/analysis_session.h"   // compute_krs_for_scheme в generate()
#include "ucuda/domain/plot_window.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/equations/sysparse.hpp"
#include "ucuda/platform/image_source.h"
#include "ucuda/domain/system_record.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <future>
#include <mutex>
#include <atomic>
#include <deque>
#include <regex>
#include <algorithm>

// Функция-распознаватель: PNG-байты -> LaTeX. Внедряется снаружи,
// чтобы модель не зависела от OcrClient напрямую (и была тестируема).
using OcrFn = std::function<std::string(const std::vector<unsigned char>&)>;

// Способ ввода системы.
enum class InputMode { Image, Latex, Plain };

// Состояние распознавания (для UI-индикации).
enum class OcrState { Idle, Running, Done, Failed };

struct SystemEditor : AppSettings {
public:
    explicit SystemEditor(OcrFn ocr) : ocr_(std::move(ocr)) {}

    // редактируемые UI-поля (UI читает/пишет напрямую)
    // Discrete map x_{n+1} = f(x_n): no integration scheme, h is pinned to 1.
    bool is_map = false;
    // ДАУ: вкладка Library → DAE. alg_text — алгебраические переменные (z),
    // vars_text у ДАУ — дифференциальные (x).
    bool is_dae = false;
    std::string alg_text;
    std::string dae_index_text;    // "x=1, lambda=3", см. SystemRecord
    bool dae_as_ode = false;       // ДАУ -> ОДУ подстановкой z (см. SystemRecord)
    std::string dae_stab_alpha = "10";   // уровень C: (d/dt + alpha)^m g = 0 (см. SystemRecord)
    std::string circuit_source;    // ДАУ из схемы: путь к .kicad_sch / .cir (см. SystemRecord)
    std::string circuit_netlist;   // копия netlist'а на момент импорта
    bool circuit_ideal_opamps = false;   // ОУ — идеальные (см. SystemRecord)
    bool circuit_no_jcaps = false, circuit_ideal_diodes = false, circuit_ideal_bjts = false;   // см. SystemRecord
    // Группы параметров (см. SystemRecord / param_groups.h) и состояние окна их правки.
    ParamGroups param_groups;
    ParamGroups ic_groups;        // то же для начальных условий
    FieldLabels field_labels;     // свои подписи полей (LaTeX)
    struct ParamGroupsUi {
        bool open = false;
        int kind = 0;             // вкладка окна: 0 — параметры, 1 — начальные условия
        int kind_req = -1;        // кнопка "Groups..." просит открыть эту вкладку (один кадр)
        std::string new_path, filter, target, rename_from, rename_to;
        std::set<std::string> sel;
    } pg_ui;
    // Импорт схемы circuit_source: перезаписывает уравнения, x, z, параметры, НУ и шаг.
    // false + err — файл не читается, kicad-cli не сработал или схема не поддержана.
    bool import_circuit(std::string* report, std::string* err);
    // Запись схемы без групп НУ (сохранена до них): группы и подписи берутся из повторного разбора
    // circuit_netlist (уравнения и значения не трогаются); что не нашлось — x в "States", z в
    // "Algebraic (computed)". Подписи импорта дописываются только там, где своей нет.
    void circuit_layout_defaults();
    // Имена для полей начальных условий: vars без алгебраических z (alg_text), если show_alg_ic
    // выключен; *hidden — сколько убрано. Значения z всё равно пересчитываются (рабочая точка /
    // проекция на g = 0), их НУ — лишь старт Ньютона. Редактор записи показывает все.
    std::vector<std::string> ic_fields(const std::vector<std::string>& vars, int* hidden = nullptr) const;
    // Рабочая точка ДАУ (x' = 0, g = 0; у схемы — расчёт по постоянному току) из текущих полей:
    // старт — НУ записи, результат пишется в init_conditions всех переменных.
    bool dae_operating_point(std::string* report, std::string* err);
    // ДАУ решается схемами ОДУ (сведение включено) — тогда всё как у ОДУ.
    bool dae_native() const { return is_dae && !dae_as_ode; }
    InputMode mode = InputMode::Image;
    std::string latex_text;        // распознанный/введённый LaTeX (правится в UI)
    std::string plain_text;        // обычный синтаксис (режим Plain)
    std::string alphabet_text;     // legacy: один список (vars+params вперемешку)
    // Явные списки переменных и параметров. Если оба непустые — приоритет
    // над alphabet_text. Удобнее визуально, явно отделяет переменные от
    // параметров. alphabet_text остаётся для старых сохранённых систем.
    std::string vars_text;
    std::string params_text;
    bool scheme_euler = false;
    bool scheme_cromer = false;
    bool scheme_midpoint = false;
    bool scheme_rk4 = false;
    bool scheme_dopri78 = false;
    bool scheme_dopri78_legacy = false;   // DOPRI78 (legacy): дроби статьи без уточнения
    bool scheme_rk45 = false;
    bool scheme_dop853 = false;
    bool scheme_cd = false;
    bool scheme_cdfast = false;   // CD fast: CD с предрасчётом констант шага
    // CD FP, CD fast FP: неявная половина простыми итерациями (cd_fp_iters)
    bool scheme_cdfp = false;
    bool scheme_cdfastfp = false;
    // Complex CD: та же композиция и тот же слот a[0], но полушаги комплексные:
    // h1 = s*h + i*h*sqrt(3)/6, h2 = (1-s)*h - i*h*sqrt(3)/6.
    bool scheme_ccd = false;
    // Complex CD4: два прохода CD с шагами gamma*h и conj(gamma)*h — порядок 4.
    // Требует s = 0.5 (иначе внутренний CD несимметричен и порядок падает до 1).
    bool scheme_ccd4 = false;
    // CCD4 (o4s3) / CCD4 (o4s4): симметричные (палиндромные) композиции того же
    // CD, тоже порядок 4 и тоже только при s = 0.5. Complex CD4 симметричной
    // быть не может — палиндром из двух стадий даёт лишь порядок 2.
    bool scheme_ccd4s3 = false;
    bool scheme_ccd4s4 = false;
    // CCD4 (s-split)_{0-1} / _{1-0}: два прохода CD с вещественными шагами h/2
    // и комплексным разбиением s1 = (1-i)/3, s2 = 1 - s1 — порядок 4. {0-1}:
    // в проходе явная половина первой, {1-0}: неявная. a[0] не читают.
    bool scheme_ccd4ss01 = false;
    bool scheme_ccd4ss10 = false;
    // Варианты {1-0} (неявная половина первой) для CD, Complex CD, Complex CD4,
    // CCD4 (o4s3), CCD4 (o4s4); порядок и требование s = 0.5 те же.
    bool scheme_cd10 = false;
    bool scheme_ccd10 = false;
    bool scheme_ccd4_10 = false;
    bool scheme_ccd4s3_10 = false;
    bool scheme_ccd4s4_10 = false;
    // Неявные схемы: Ньютон по символьному якобиану, решается ВСЯ связанная
    // система (в отличие от CD, диагонально-неявного). Отсюда A-устойчивость.
    bool scheme_ieuler = false;
    bool scheme_imidpoint = false;
    // SEMP / SIMP: средняя точка с последовательной стадией на полушаге
    // h1 = s*h (s = a[0], слот CD). У SEMP стадия явная, у SIMP —
    // диагонально-неявная. Порядок 2 только при s = 0.5.
    bool scheme_semp = false;
    bool scheme_simp = false;
    // D: та же диагонально-неявная стадия, но как самостоятельный шаг на
    // полный h — метод первого порядка, a[0] не читает.
    bool scheme_dmethod = false;
    // Complex Implicit Euler: композиция двух неявных Эйлеров с сопряжёнными
    // комплексными шагами tau1 = h*(s + i/2), tau2 = h*(1 - s - i/2), s = a[0].
    // Порядок 2 только при s = 0.5 — коэффициенты вынуждены, см. codegen.hpp.
    bool scheme_cieuler = false;
    // ГБШ: GBS (n=2) — опорник, GBS 2-4 ... 2-4-6-8-10-12 — экстраполяторы
    // с одной цепочкой leapfrog на стадию (см. Scheme::GBS в codegen.hpp).
    bool scheme_gbs = false;
    bool scheme_gbs24 = false;
    bool scheme_gbs246 = false;
    bool scheme_gbs2468 = false;
    bool scheme_gbs246810 = false;
    bool scheme_gbs24681012 = false;

    // Настройки Ньютона (см. SystemRecord). Едут в System через build_system().
    bool        newton_full      = false;
    std::string newton_tol       = "1e-10";
    std::string newton_max_iters = "8";
    // Число простых итераций CD FP / CD fast FP по умолчанию для вкладок
    // (как symmetry_s: копируется в конфиги при загрузке системы).
    std::string cd_fp_iters      = "4";

    // Коэффициент симметрии s для CD-методов (передаётся в kernel как a[0]).
    std::string symmetry_s = "0.5";

    // Пользовательские именованные КРС (выбираются в scheme combo сессий
    // вместе с built-in). Имя не должно совпадать с built-in.
    std::vector<CustomScheme> custom_schemes;

    // Wrapper schemes the user has assembled -- names only, and BOTH kinds:
    // "Extr(<base>|n...)" and "Comp(<base>|g...)". One list, because the
    // resolver dispatches on the name prefix; that is what lets a second
    // combinator cost no new plumbing. The body is not stored -- it is rebuilt
    // from the name, so editing the system never leaves a stale wrapper.
    std::vector<std::string> wrapper_schemes;
    // Выключенные обёртки, канонические имена (см. SystemRecord::disabled_wrappers).
    std::vector<std::string> disabled_wrappers;
    // Выключенные методы ДАУ (см. SystemRecord::dae_schemes_off).
    std::vector<std::string> dae_schemes_off;
    bool dae_scheme_on(const std::string& n) const {
        return std::find(dae_schemes_off.begin(), dae_schemes_off.end(), n) == dae_schemes_off.end();
    }
    bool wrapper_enabled(const std::string& name) const {
        const std::string c = wrapper_canonical_name(name);
        for (const auto& d : disabled_wrappers) if (d == c) return false;
        return true;
    }
    void set_wrapper_enabled(const std::string& name, bool on) {
        const std::string c = wrapper_canonical_name(name);
        disabled_wrappers.erase(std::remove(disabled_wrappers.begin(), disabled_wrappers.end(), c),
                                disabled_wrappers.end());
        if (!on) disabled_wrappers.push_back(c);
    }
    // То, что видят комбо схем во вкладках: только включённые обёртки.
    std::vector<std::string> enabled_wrapper_schemes() const {
        std::vector<std::string> out;
        for (const auto& w : wrapper_schemes) if (wrapper_enabled(w)) out.push_back(w);
        return out;
    }
    // Свёрнут ли код кастомной КРС в System tab; ключ — имя схемы, как у
    // custom_scheme_editor_h. Нет записи — развёрнут.
    std::map<std::string, bool> custom_scheme_code_collapsed;
    // Строка списка обёрток, которую сейчас переименовывают, и набираемое имя.
    // Держать его в самом имени схемы по ходу набора нельзя: имя нормализуется
    // при разборе, и пробел, набранный в "Suzuki 17", исчезал бы на следующем
    // же кадре. -1 — никого не переименовывают.
    int         wrapper_rename_index = -1;
    std::string wrapper_rename_text;
    bool        wrapper_rename_focus = false;   // поставить фокус в поле на след. кадре

    // Состояние конструктора обёрток в System tab. Живёт здесь, а не в
    // рисующей функции: ImGui перерисовывает панель каждый кадр, локальные
    // переменные между кадрами не сохраняются.
    std::string extr_builder_base = "RK4";
    int         extr_builder_stages = 3;
    // Подшаги стадий; используются первые extr_builder_stages элементов.
    // Дефолт — гармоническая последовательность: при равном порядке она
    // дешевле удвоения (сумма n меньше).
    // Размер массива — kExtrMaxStages: конструктор пишет в первые
    // extr_builder_stages элементов, и потолок стадий и потолок массива
    // обязаны быть одним и тем же числом.
    int         extr_builder_n[kExtrMaxStages] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    // Имя собираемой обёртки. Пустое — имени не давали, и в списке схема будет
    // называться своим описанием, как до появления этого поля.
    std::string extr_builder_label;
    // ExtrZ: Re только на выходе экстраполяции (база — встроенная комплексная
    // CD-схема). См. ExtrapolationSpec::re_at_output.
    bool        extr_builder_re_output = false;

    // Composition builder. Coefficients are TEXT, not numbers: a stage may be
    // a parameter name or an expression over parameters ("g1", "1-2*g1"), and
    // that is the whole point -- a coefficient that is a parameter is sweepable
    // from the Order tab. Default is the Yoshida triple jump over a symmetric
    // base, written symbolically so the map works out of the box.
    std::string comp_builder_base = "CD";
    int         comp_builder_stages = 3;
    std::string comp_builder_g[kCompMaxStages] = { "g1", "1-2*g1", "g1" };
    // Имя собираемой композиции; см. extr_builder_label. Для опубликованных
    // методов оно и есть единственная читаемая подпись: семнадцать
    // коэффициентов по восемнадцать цифр ни в комбо, ни в легенду не влезают.
    std::string comp_builder_label;
    // Итог последней загрузки коэффициентов из файла: текст строки статуса под
    // кнопкой и её цвет. Живёт здесь, а не в рисующей функции: ImGui
    // перерисовывает панель каждый кадр, локальная переменная не доживёт.
    std::string comp_load_status;
    bool        comp_load_failed = false;

    // порядок индексации параметров a[1..]
    ParamOrder param_order = ParamOrder::AsInAlphabet;

    // вспомогательные функции (опция). Если включено, func_defs_text парсится
    // и инлайнится в правые части. Напр.: "h(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)"
    bool use_aux_funcs = false;
    std::string func_defs_text;

    // Editor heights (px) for Library multiline inputs. Per-session only.
    float latex_editor_h = 90.0f;
    float plain_editor_h = 90.0f;
    float funcs_editor_h = 60.0f;
    float gen_code_editor_h = 220.0f;
    std::map<std::string, float> custom_scheme_editor_h;   // keyed by scheme name

    // метаданные для библиотеки
    std::string name;
    std::string note;

    // значения по умолчанию (всё опционально, пустая строка = не задано)
    std::string step_h;
    std::map<std::string, std::string> init_conditions; // var -> value
    std::map<std::string, std::string> param_values;    // param -> value

    // имена переменных/параметров, полученные при последнем refresh_symbols()
    std::vector<std::string> known_vars;
    std::vector<std::string> known_params;

    // имя, под которым система сейчас сохранена на диске (для переименования).
    // Пусто = система ещё не сохранена/загружена под именем.
    std::string loaded_name;

    // Редактор ДАУ: результат parse_dae для текущего текста черновика. Разбор
    // идёт только при смене ввода (key), не каждый кадр — в нём regex.
    struct DaeEditorState {
        std::string key;                    // ввод, по которому получен результат
        bool        ok = false;
        std::string message;                // ошибка или сводка (индексы, t)
        std::vector<std::string> vars;      // x, затем z — для полей НУ
        std::vector<std::string> params;
        bool        reducible = false;      // сводится к ОДУ подстановкой z (dae_reduce_to_ode)
        std::string reduce_msg;             // z(x) или почему не сводится
        bool        level_c = false;        // сводится только уровнем C (индекс 2-3, стабилизация alpha)
        std::string latex;                  // уравнения для превью: \dot{x} = f, 0 = g (по строке)
        int         latex_lines = 0;
        // Импорт схемы: отчёт / ошибка последнего импорта, слежение за файлом.
        std::string circuit_report, circuit_error;
        std::string circuit_path;           // путь, для которого запомнен circuit_mtime
        long long   circuit_mtime = -1;
        double      circuit_poll_t = 0.0;   // ImGui::GetTime() последней проверки
        bool        circuit_watch = true;   // перечитывать при сохранении файла
        bool        circuit_import_now = false;
    };
    DaeEditorState dae_editor;

    // результат генерации
    std::string generated_code;    // итоговый код всех выбранных схем
    std::string error_message;     // ошибка парсинга/генерации (для показа в UI)

    // Превью правых частей над X[..]/a[..] под полем ввода Library — то, из
    // чего кодогенератор собирает шаг любой схемы (codegen_rhs). Считается
    // лениво: разбор уравнений на каждом кадре дорог, поэтому пересчёт идёт
    // только когда меняется что-то из входов build_system() (см. key).
    struct RhsPreview {
        std::string key;     // снимок входов, по которому посчитано
        std::string text;    // листинг: легенда X[i]/a[j] и по строке на уравнение
        std::string error;   // ошибка разбора, если text пуст
    };
    const RhsPreview& rhs_preview() const;
    mutable RhsPreview rhs_preview_;

    // запуск фонового OCR
    // Не блокирует UI. Источник изображения передаётся владением.
    void start_ocr(std::unique_ptr<ImageSource> src) {
        if (ocr_state_ == OcrState::Running) return; // одно распознавание за раз
        ocr_state_ = OcrState::Running;
        ocr_error_.clear();
        auto fn = ocr_;
        ImageSource* raw = src.release();
        ocr_future_ = std::async(std::launch::async, [fn, raw]() -> std::string {
            std::unique_ptr<ImageSource> owner(raw);
            auto png = owner->get_png();      // чтение файла/буфера
            return fn(png);                   // вызов OCR (долгий)
            });
    }

    // Вызывается КАЖДЫЙ кадр из UI-потока. Забирает результат, когда готов.
    // Возвращает true, если состояние изменилось (UI может отреагировать).
    bool poll() {
        if (ocr_state_ != OcrState::Running) return false;
        if (!ocr_future_.valid()) { ocr_state_ = OcrState::Idle; return false; }
        if (ocr_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return false;
        try {
            std::string raw = ocr_future_.get();   // успех
            latex_text = format_latex(raw);
            mode = InputMode::Latex;
            ocr_state_ = OcrState::Done;
        }
        catch (const std::exception& e) {
            ocr_error_ = e.what();
            ocr_state_ = OcrState::Failed;
        }
        return true;
    }

    OcrState ocr_state() const { return ocr_state_; }
    const std::string& ocr_error() const { return ocr_error_; }

    // генерация кода из текущих полей
    // Возвращает true при успехе, иначе заполняет error_message.
    bool generate() {
        error_message.clear();
        generated_code.clear();
        try {
            System sys = build_system();
            std::string out;
            // ДАУ: тела методов ДАУ над [x; z]. Метод, который к системе не применим (CD и явные
            // при индексе >= 2), печатает причину, а не обрывает остальные.
            if (dae_native()) {
                bool any = false;
                for (const auto& n : dae_scheme_names()) {
                    if (!dae_scheme_on(n) || dae_method_hidden(n)) continue;
                    any = true;
                    DaeScheme ds;
                    dae_scheme_from_name(n, &ds);
                    out += "// ===== " + n + " =====\n";
                    try { out += codegen_dae_scheme(sys, ds); }
                    catch (const std::exception& e) { out += "// (not available: " + std::string(e.what()) + ")\n"; }
                    out += "\n";
                }
                // BDF 1-5: тела постоянного шага нет — драйвер (kernels/ucuda_bdf.cuh) берёт невязку и якобиан
                if (dae_scheme_on(kBdfSchemeDae) && !dae_method_hidden(kBdfSchemeDae)) {
                    any = true;
                    out += "// ===== " + std::string(kBdfSchemeDae) + " (multistep driver kernels/ucuda_bdf.cuh: "
                           "residual F = [f; g] and Jacobian J; adaptive step only) =====\n";
                    try {
                        System sp = sys;   // как bdf_prepare_system (multistep_bdf.h): индекс 3 -> GGL
                        std::string why;
                        if (dae_structural_index(sys) > 2 && !dae_ggl_system(sys, sp, &why))
                            throw std::runtime_error("index 3 via GGL: " + why);
                        if (sp.vars.size() != sys.vars.size())
                            out += "// index 3: GGL adds " + std::to_string(sp.vars.size() - sys.vars.size())
                                 + " multiplier(s) to the state\n";
                        std::string res, jac;
                        codegen_bdf_bodies(sp, res, jac);
                        out += "/* F */\n" + res + "/* J */\n" + jac;
                    }
                    catch (const std::exception& e) { out += "// (not available: " + std::string(e.what()) + ")\n"; }
                    out += "\n";
                }
                if (!any) { error_message = "no DAE method selected"; return false; }
                generated_code = out;
                return true;
            }
            // A map has exactly one step body and no scheme to pick from.
            if (is_map) {
                generated_code = "// ===== Map =====\n"
                               + codegen_scheme(sys, Scheme::Map) + "\n";
                return true;
            }
            struct Item { bool on; const char* name; Scheme s; };
            // Порядок — как в UI: по порядку точности (1, 2, 4, 8).
            const Item items[] = {
                { scheme_euler,    "Euler",             Scheme::Euler },
                { scheme_cromer,   "Euler-Cromer",      Scheme::EulerCromer },
                { scheme_dmethod,  "D",                 Scheme::D },
                { scheme_ieuler,   "Implicit Euler",    Scheme::ImplicitEuler },
                { scheme_midpoint, "Explicit Midpoint", Scheme::ExplicitMidpoint },
                { scheme_imidpoint,"Implicit Midpoint", Scheme::ImplicitMidpoint },
                { scheme_cd,       "CD",                Scheme::CD },
                { scheme_cd10, "CD_{1-0}", Scheme::CD10 },
                { scheme_cdfast,   "CD fast",           Scheme::CDFast },
                { scheme_cdfp,     "CD FP",             Scheme::CDFP },
                { scheme_cdfastfp, "CD fast FP",        Scheme::CDFastFP },
                { scheme_ccd,      "Complex CD",        Scheme::ComplexCD },
                { scheme_ccd10, "Complex CD_{1-0}", Scheme::ComplexCD10 },
                { scheme_cieuler,  "Complex Implicit Euler", Scheme::ComplexIEuler },
                { scheme_semp,     "SEMP",              Scheme::SEMP },
                { scheme_simp,     "SIMP",              Scheme::SIMP },
                { scheme_rk4,      "RK4",               Scheme::RK4 },
                { scheme_ccd4,     "Complex CD4",       Scheme::ComplexCD4 },
                { scheme_ccd4_10, "Complex CD4_{1-0}", Scheme::ComplexCD4_10 },
                { scheme_ccd4s3,   "CCD4 (o4s3)",       Scheme::ComplexCD4S3 },
                { scheme_ccd4s3_10, "CCD4 (o4s3)_{1-0}", Scheme::ComplexCD4S3_10 },
                { scheme_ccd4s4,   "CCD4 (o4s4)",       Scheme::ComplexCD4S4 },
                { scheme_ccd4s4_10, "CCD4 (o4s4)_{1-0}", Scheme::ComplexCD4S4_10 },
                { scheme_ccd4ss01, "CCD4 (s-split)_{0-1}", Scheme::ComplexCD4SS01 },
                { scheme_ccd4ss10, "CCD4 (s-split)_{1-0}", Scheme::ComplexCD4SS10 },
                { scheme_rk45,     "RK45",              Scheme::RK45 },
                { scheme_dopri78,  "DOPRI78",           Scheme::DOPRI78 },
                { scheme_dopri78_legacy, "DOPRI78 (legacy)", Scheme::DOPRI78Legacy },
                { scheme_dop853,   "DOP853",            Scheme::DOP853 },
                { scheme_gbs, "GBS (n=2)", Scheme::GBS },
                { scheme_gbs24, "GBS 2-4", Scheme::GBS24 },
                { scheme_gbs246, "GBS 2-4-6", Scheme::GBS246 },
                { scheme_gbs2468, "GBS 2-4-6-8", Scheme::GBS2468 },
                { scheme_gbs246810, "GBS 2-4-6-8-10", Scheme::GBS246810 },
                { scheme_gbs24681012, "GBS 2-4-6-8-10-12", Scheme::GBS24681012 },
            };
            bool any = false;
            for (const auto& it : items) {
                if (!it.on || scheme_hidden(it.name)) continue;
                any = true;
                out += "// ===== ";
                out += it.name;
                out += " =====\n";
                // ДАУ как ОДУ (уровень B) берёт не все схемы — у неё отказ печатается на месте.
                if (is_dae) {
                    try { out += codegen_scheme(sys, it.s); }
                    catch (const std::exception& e) { out += "// (not available: " + std::string(e.what()) + ")\n"; }
                } else {
                    out += codegen_scheme(sys, it.s);
                }
                out += "\n";
            }
            // custom-схемы — добавляем их код как есть для preview
            for (const auto& cs : custom_schemes) {
                if (cs.body.empty() || !cs.enabled) continue;
                any = true;
                out += "// ===== ";
                out += cs.name;
                out += " (custom) =====\n";
                out += cs.body;
                out += "\n";
            }
            // Обёртки Extr/ExtrZ/Comp — тем же резолвером, что уходит в ядро на
            // Run, поэтому превью совпадает с тем, что посчитается. Пустое тело —
            // база не нашлась или не разобралась; говорим об этом прямо.
            for (const auto& w : wrapper_schemes) {
                if (!wrapper_enabled(w)) continue;
                any = true;
                const std::string shown = wrapper_display_name(w);
                out += "// ===== ";
                out += shown;
                out += " (wrapper) =====\n";
                if (shown != w) { out += "// "; out += w; out += "\n"; }
                std::string why;
                const std::string body = compute_krs_for_scheme(custom_schemes, sys, w, &why);
                out += body.empty() ? "// (could not build: "
                                      + (why.empty() ? std::string("base scheme missing or invalid") : why)
                                      + ")\n"
                                    : body;
                out += "\n";
            }
            if (!any) { error_message = "no scheme selected"; return false; }
            generated_code = out;
            return true;
        }
        catch (const std::exception& e) {
            error_message = e.what();
            return false;
        }
    }

    // Парсит систему и обновляет known_vars/known_params, синхронизируя
    // словари значений (сохраняет уже введённые, убирает исчезнувшие символы).
    // Возвращает true при успехе; при ошибке парсинга кладёт её в error_message.
    bool refresh_symbols();

    // Выгрузка/загрузка состояния в запись библиотеки.
    SystemRecord to_record() const;
    void from_record(const SystemRecord& r);

    // Сброс всех полей к пустому состоянию (новая система с нуля).
    void clear();

protected:
    OcrFn ocr_;

    // Форматирует сырой LaTeX в единообразный вид: каждое уравнение на своей строке, без
    // \begin{aligned}/\end{aligned} и без \\, все формы производных нормализуются к \dot{X}.
    // Поддерживаемые исходные формы: \frac{dX}{dt}, \dfrac{dX}{dt} (+ варианты с \tau, \theta,
    // пробелами), X' (prime), dX/dt (slash-form).
    static std::string format_latex(const std::string& raw) {
        std::string s = raw;
        // 1) Снимаем обёртки окружений.
        auto remove_all = [&](const std::string& token) {
            size_t p;
            while ((p = s.find(token)) != std::string::npos) s.erase(p, token.size());
            };
        for (const char* env : { "aligned","array","cases","equation","gather","split","matrix" }) {
            remove_all(std::string("\\begin{") + env + "}");
            remove_all(std::string("\\end{") + env + "}");
        }
        remove_all("\\left\\{");
        remove_all("\\right.");
        // align-табы и `{}=`-обёртки правой части — типичный OCR-мусор.
        remove_all("&");
        // \mathrm{d}, \mathrm{t} → d, t (чтобы \frac-нормализация ниже поймала).
        auto replace_all = [&](const std::string& from, const std::string& to) {
            size_t p = 0;
            while ((p = s.find(from, p)) != std::string::npos) {
                s.replace(p, from.size(), to);
                p += to.size();
            }
            };
        replace_all("\\mathrm{d}", "d");
        replace_all("\\mathrm{t}", "t");
        replace_all("\\mathrm d",  "d");
        replace_all("\\mathrm t",  "t");
        replace_all("{{}=", "=");   // OCR-артефакт "{{}=..."
        replace_all("{}=",  "=");

        // 2) Нормализация производных к \dot{X}. X = одна буква, \greek или {group}.
        //    std::regex с raw-strings ECMAScript-flavor по умолчанию.
        try {
            static const std::regex re_frac(
                R"(\\d?frac\s*\{\s*d\s*([A-Za-z]|\\[A-Za-z]+|\{[^{}]*\})\s*\}\s*\{\s*d\s*(t|\\tau|\\theta)\s*\})");
            static const std::regex re_slash(
                R"(d\s*([A-Za-z]|\\[A-Za-z]+)\s*/\s*d\s*(t|\\tau|\\theta))");
            static const std::regex re_prime(
                R"(([A-Za-z]|\\[A-Za-z]+)\s*\')");
            s = std::regex_replace(s, re_frac,  R"(\dot{$1})");
            s = std::regex_replace(s, re_slash, R"(\dot{$1})");
            s = std::regex_replace(s, re_prime, R"(\dot{$1})");
        } catch (const std::regex_error&) {
            // если regex недоступен по какой-то причине — оставляем как было
        }

        // 3) Разбить по "\\" и по переводам строк.
        std::vector<std::string> eqs;
        size_t start = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == '\\') {
                eqs.push_back(s.substr(start, i - start));
                i += 1; start = i + 1;
            } else if (s[i] == '\n' || s[i] == '\r') {
                eqs.push_back(s.substr(start, i - start));
                start = i + 1;
            }
        }
        eqs.push_back(s.substr(start));

        // 4) Trim, снять обёртки `{...}` вокруг всего уравнения, собрать со `\\` в конце.
        std::vector<std::string> clean;
        for (auto& e : eqs) {
            size_t a = e.find_first_not_of(" \t\n\r");
            size_t b = e.find_last_not_of(" \t\n\r");
            if (a == std::string::npos) continue;
            std::string eq = e.substr(a, b - a + 1);
            // снять внешние `{ ... }` (артефакт OCR-вывода).
            while (eq.size() >= 2 && eq.front() == '{' && eq.back() == '}') {
                // только если фигурные сбалансированы строго на концах.
                int depth = 0; bool ok = true;
                for (size_t k = 0; k < eq.size(); ++k) {
                    if (eq[k] == '{') ++depth;
                    else if (eq[k] == '}') {
                        --depth;
                        if (depth == 0 && k + 1 < eq.size()) { ok = false; break; }
                    }
                }
                if (!ok) break;
                eq = eq.substr(1, eq.size() - 2);
                size_t a2 = eq.find_first_not_of(" \t");
                size_t b2 = eq.find_last_not_of(" \t");
                if (a2 == std::string::npos) { eq.clear(); break; }
                eq = eq.substr(a2, b2 - a2 + 1);
            }
            bool only_braces = eq.find_first_not_of("{}clr| \t") == std::string::npos;
            if (only_braces) continue;
            // снять trailing punctuation типа " ,"/" ." (артефакты OCR).
            while (!eq.empty() && (eq.back() == ',' || eq.back() == '.' || eq.back() == ';' || eq.back() == ' ' || eq.back() == '\t'))
                eq.pop_back();
            if (eq.empty()) continue;
            clean.push_back(eq);
        }
        if (clean.empty()) return raw;
        std::string out;
        for (size_t i = 0; i < clean.size(); ++i) {
            if (i) out += "\n";
            out += clean[i];
            if (i + 1 < clean.size()) out += " \\\\";
        }
        return out;
    }
    std::future<std::string> ocr_future_;
    std::atomic<OcrState> ocr_state_{ OcrState::Idle };
    std::string ocr_error_;

    // парсит алфавит из строки "x, y, z" -> вектор
    std::vector<std::string> parse_alphabet() const {
        // Если alphabet_text пуст, склеиваем явные vars+params (нужно для
        // build_system / parse_system_from_latex с новым форматом). Хватает
        // одного непустого списка: у системы без параметров params_text пуст,
        // и требование «оба непустые» оставляло её вовсе без алфавита.
        const std::string& src =
            !alphabet_text.empty()
                ? alphabet_text
                : (!vars_text.empty() || !params_text.empty()
                    ? (combined_alpha_ = vars_text + "," + params_text)
                    : alphabet_text);
        std::vector<std::string> out;
        std::string cur;
        for (char c : src) {
            if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == ';') {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            }
            else cur += c;
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }
    // буфер для склеенного алфавита (используется в parse_alphabet)
    mutable std::string combined_alpha_;

    // строит System из текущего режима ввода
    System build_system() const; // реализовано в .cpp (зависит от sysparse)
};
