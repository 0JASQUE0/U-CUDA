#pragma once
#include "codegen.hpp"
#include "sysparse.hpp"
#include "image_source.h"
#include "system_record.h"
#include "analysis_session.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <mutex>
#include <atomic>
#include <deque>
#include <regex>

// Функция-распознаватель: PNG-байты -> LaTeX. Внедряется снаружи,
// чтобы модель не зависела от OcrClient напрямую (и была тестируема).
using OcrFn = std::function<std::string(const std::vector<unsigned char>&)>;

// Способ ввода системы.
enum class InputMode { Image, Latex, Plain };

// Один элемент общей parametric-очереди: какой анализ + индекс в его сессии.
// Очередь живёт в AppModel и драйнится в draw_gui после polls. См. план
// «Cross-analysis batch Run all».
struct ParametricQueueItem {
    enum class Kind { Bifurcation, LLE, LS };
    Kind kind = Kind::Bifurcation;
    int  index = 0;
};

// Один элемент basins-очереди — индекс config'а в basins_session.configs.
// Basins живёт в отдельной очереди (а не в parametric_queue), потому что
// сейчас параметрика и basins не пересекаются по UI: parametric "Run all"
// пушит BD/LLE/LS, а basins "Run all" — свои configs. Драйнится тем же
// движком, но независимым тиком в draw_gui.
struct BasinsQueueItem {
    int index = 0;
};

// Состояние распознавания (для UI-индикации).
enum class OcrState { Idle, Running, Done, Failed };

// Модель приложения: всё состояние + логика. НЕ знает про ImGui.
// UI читает поля и вызывает методы; долгие операции идут в фоне.
class AppModel {
public:
    explicit AppModel(OcrFn ocr) : ocr_(std::move(ocr)) {}

    // ---- редактируемые UI-поля (UI читает/пишет напрямую) ----
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
    bool scheme_cd = false;

    // Коэффициент симметрии s для CD-метода (передаётся в kernel как a[0]).
    std::string symmetry_s = "0.5";

    // Пользовательские именованные КРС (выбираются в scheme combo сессий
    // вместе с built-in). Имя не должно совпадать с built-in.
    std::vector<CustomScheme> custom_schemes;

    // порядок индексации параметров a[1..]
    ParamOrder param_order = ParamOrder::AsInAlphabet;

    // вспомогательные функции (опция). Если включено, func_defs_text парсится
    // и инлайнится в правые части. Напр.: "h(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)"
    bool use_aux_funcs = false;
    std::string func_defs_text;

    // --- метаданные для библиотеки ---
    std::string name;
    std::string note;

    // --- значения по умолчанию (всё опционально, пустая строка = не задано) ---
    std::string step_h;
    std::map<std::string, std::string> init_conditions; // var -> value
    std::map<std::string, std::string> param_values;    // param -> value

    // имена переменных/параметров, полученные при последнем refresh_symbols()
    std::vector<std::string> known_vars;
    std::vector<std::string> known_params;

    // имя, под которым система сейчас сохранена на диске (для переименования).
    // Пусто = система ещё не сохранена/загружена под именем.
    std::string loaded_name;

    // --- режим приложения и сессия анализа (слой 2) ---
    // режим верхнего уровня: библиотека, фазовый анализ, параметрический,
    // бассейны притяжения или настройки.
    enum class AppMode { Library, Analysis, Parametric, Basins, Settings };
    AppMode app_mode = AppMode::Library;

    // сессия анализа фазовых портретов ("песочница": изменения не сохраняются)
    PhaseAnalysisSession phase_session;

    // Parametric mode держит три независимых анализа (1D bif / LLE / LS),
    // переключаемых верхними табами в Parametric Controls. Каждый —
    // песочница: своя копия системы, свои Run-кнопки, свой плот, свой
    // _last_*.json. Результаты сессии не share'ят между собой.
    BifurcationAnalysisSession bifurcation_session;
    LLEAnalysisSession         lle_session;
    LyapunovSpectrumAnalysisSession ls_session;

    // Basins of attraction — отдельный AppMode со своим Run и 5-плотным окном.
    BasinsAnalysisSession      basins_session;

    // 0=Bifurcation, 1=LLE, 2=LS. Активный sub-tab; используется для top-bar
    // indicator. По умолчанию — Bifurcation.
    int parametric_active_analysis = 0;

    // UI scale (DPI-aware). `ui_scale_auto` выставляется однократно при старте
    // из glfwGetMonitorContentScale. `ui_scale_override` — слайдер в GUI;
    // 0.0 означает «использовать auto». Эффективный scale (см. helper) идёт
    // в apply_ui_scale в app_main каждый кадр — если изменился, шрифт и
    // ImGui-style пересоздаются.
    float ui_scale_auto     = 1.0f;
    float ui_scale_override = 0.0f;
    float effective_ui_scale() const {
        return ui_scale_override > 0.0f ? ui_scale_override : ui_scale_auto;
    }

    // Выбор шрифта. false (дефолт) → TTF Segoe UI (антиалиаc, читаемо на 1080p
    // и крупных DPI). true → bitmap ProggyClean (компактный, классический ImGui).
    // Чекбокс — в Settings; персистится в _app_config.json.
    bool use_builtin_font = false;

    // Последний выбранный colormap для HeatmapView (LLE-2D и пр.).
    // 0=Viridis, 1=Inferno, 2=Turbo, 3=Gray. Дефолт — Viridis.
    // Персистится в _app_config.json. Combo «Colormap» над хитмапой пишет
    // сюда; static HeatmapView в draw_lle_plot читает при первом создании.
    int heatmap_colormap = 0;

    // Кол-во значащих цифр в подписях тиков осей и colorbar'а (2-10).
    // Зеркалит AppConfig::tick_precision; при изменении в Settings вызывается
    // set_tick_precision() из plot_axis.h, чтобы fmt_tick() сразу подхватил.
    int tick_precision = 4;

    // движок параметрики (NVRTC + NonLinAnal). Лениво создаётся при первом Run.
    std::unique_ptr<ParametricEngine> parametric_engine;

    // Cross-analysis batch queue. Run all... popup пушит сюда выбранные конфиги
    // BD/LLE/LS. start_next_in_parametric_queue() драйнит её серийно (engine один).
    std::deque<ParametricQueueItem> parametric_queue;

    // Если ни одна из трёх сессий не in_flight и очередь не пуста — снимает
    // элемент с фронта и запускает run_async соответствующей сессии.
    // Возвращает true если что-то стартовало. Безопасно вызывать каждый кадр.
    bool start_next_in_parametric_queue();

    // Basins batch queue — независимая от parametric_queue (см. BasinsQueueItem).
    std::deque<BasinsQueueItem> basins_queue;
    bool start_next_in_basins_queue();

    // Удаление конфига + чистка очереди (убрать item с этим индексом, у
    // оставшихся того же kind сдвинуть index > i на -1). Используется GUI
    // вместо прямого session.remove_*, чтобы очередь не указывала на
    // удалённые слоты.
    void remove_bifurcation_diagram(int i);
    void remove_lle_curve(int i);
    void remove_ls_curve(int i);
    void remove_basins_config(int i);

    // Подготовить сессию анализа из ТЕКУЩЕЙ системы (после refresh_symbols).
    // Копирует параметры/НУ в сессию; изменения в сессии не идут в библиотеку.
    bool start_phase_analysis();
    // Инициализирует ВСЕ parametric-сессии (bif/LLE/LS) из текущей системы —
    // пользователь ждёт одинаковые vars/params в любом верхнем табе.
    bool start_parametric_analysis();
    // Инициализирует basins-сессию из текущей системы.
    bool start_basins_analysis();

    // ---- результат генерации ----
    std::string generated_code;    // итоговый код всех выбранных схем
    std::string error_message;     // ошибка парсинга/генерации (для показа в UI)

    // ---- запуск фонового OCR ----
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

    // ---- генерация кода из текущих полей ----
    // Возвращает true при успехе, иначе заполняет error_message.
    bool generate() {
        error_message.clear();
        generated_code.clear();
        try {
            System sys = build_system();
            std::string out;
            struct Item { bool on; const char* name; Scheme s; };
            const Item items[] = {
                { scheme_euler,    "Euler",             Scheme::Euler },
                { scheme_cromer,   "Euler-Cromer",      Scheme::EulerCromer },
                { scheme_midpoint, "Explicit Midpoint", Scheme::ExplicitMidpoint },
                { scheme_rk4,      "RK4",               Scheme::RK4 },
                { scheme_dopri78,  "DOPRI78",           Scheme::DOPRI78 },
                { scheme_cd,       "CD",                Scheme::CD },
            };
            bool any = false;
            for (const auto& it : items) {
                if (!it.on) continue;
                any = true;
                out += "// ===== ";
                out += it.name;
                out += " =====\n";
                out += codegen_scheme(sys, it.s);
                out += "\n";
            }
            // custom-схемы — добавляем их код как есть для preview
            for (const auto& cs : custom_schemes) {
                if (cs.body.empty()) continue;
                any = true;
                out += "// ===== ";
                out += cs.name;
                out += " (custom) =====\n";
                out += cs.body;
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

private:
    OcrFn ocr_;

    // Форматирует сырой LaTeX в единообразный вид: каждое уравнение на своей
    // строке, без \begin{aligned}/\end{aligned} и без \\, все формы
    // производных нормализуются к \dot{X}.
    // Поддерживаемые исходные формы производных:
    //   \frac{dX}{dt}, \dfrac{dX}{dt} (+ варианты с \tau, \theta, пробелами)
    //   X'                   (prime)
    //   dX/dt                (slash-form)
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
        // Если alphabet_text пуст, но есть явные vars+params, склеиваем их
        // (нужно для build_system / parse_system_from_latex с новым форматом).
        const std::string& src =
            !alphabet_text.empty()
                ? alphabet_text
                : (!vars_text.empty() && !params_text.empty()
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