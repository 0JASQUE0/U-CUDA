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

// Функция-распознаватель: PNG-байты -> LaTeX. Внедряется снаружи,
// чтобы модель не зависела от OcrClient напрямую (и была тестируема).
using OcrFn = std::function<std::string(const std::vector<unsigned char>&)>;

// Способ ввода системы.
enum class InputMode { Image, Latex, Plain };

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
    std::string alphabet_text;     // алфавит через запятую: "x,y,z,sigma,rho,beta"
    bool scheme_euler = false;
    bool scheme_cromer = false;
    bool scheme_midpoint = false;
    bool scheme_rk4 = false;

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
    std::map<std::string, std::string> param_min;       // param -> min
    std::map<std::string, std::string> param_max;       // param -> max

    // имена переменных/параметров, полученные при последнем refresh_symbols()
    std::vector<std::string> known_vars;
    std::vector<std::string> known_params;

    // имя, под которым система сейчас сохранена на диске (для переименования).
    // Пусто = система ещё не сохранена/загружена под именем.
    std::string loaded_name;

    // --- режим приложения и сессия анализа (слой 2) ---
    // режим верхнего уровня: библиотека, фазовый анализ или параметрический
    enum class AppMode { Library, Analysis, Parametric };
    AppMode app_mode = AppMode::Library;

    // сессия анализа фазовых портретов ("песочница": изменения не сохраняются)
    PhaseAnalysisSession phase_session;

    // сессия параметрического анализа (1D-бифуркация) — независимая песочница
    ParametricAnalysisSession parametric_session;

    // движок параметрики (NVRTC + NonLinAnal). Лениво создаётся при первом Run.
    std::unique_ptr<ParametricEngine> parametric_engine;

    // Подготовить сессию анализа из ТЕКУЩЕЙ системы (после refresh_symbols).
    // Копирует параметры/НУ в сессию; изменения в сессии не идут в библиотеку.
    bool start_phase_analysis();
    bool start_parametric_analysis();

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

    // Форматирует сырой LaTeX в читаемый вид:
    //   \begin{aligned}
    //    eq1 \\
    //    eq2 \\
    //   \end{aligned}
    // Если окружения нет — просто разбивает по \\ с отступом.
    static std::string format_latex(const std::string& raw) {
        std::string s = raw;
        // вырезаем существующие окружения-обёртки, оставив только уравнения
        auto remove_all = [&](const std::string& token) {
            size_t p;
            while ((p = s.find(token)) != std::string::npos) s.erase(p, token.size());
            };
        // снять \begin{...}/\end{...} aligned|array|cases и \left\{ \right.
        for (const char* env : { "aligned","array","cases" }) {
            remove_all(std::string("\\begin{") + env + "}");
            remove_all(std::string("\\end{") + env + "}");
        }
        remove_all("\\left\\{");
        remove_all("\\right.");
        // спецификатор столбцов array, напр. {c} или {r l}
        // (грубо: убираем одиночную {...} из коротких буквенных групп в начале)

        // разбить по "\\"
        std::vector<std::string> eqs;
        size_t start = 0;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '\\' && s[i + 1] == '\\') {
                eqs.push_back(s.substr(start, i - start));
                i += 1; start = i + 1;
            }
        }
        eqs.push_back(s.substr(start));

        // собрать в желаемом формате
        std::string out = "\\begin{aligned}\n";
        bool any = false;
        for (auto& e : eqs) {
            // trim
            size_t a = e.find_first_not_of(" \t\n\r");
            size_t b = e.find_last_not_of(" \t\n\r");
            if (a == std::string::npos) continue;
            std::string eq = e.substr(a, b - a + 1);
            // пропустить остаток вроде одиночного спецификатора столбцов "{c}"
            bool only_braces = eq.find_first_not_of("{}clr| \t") == std::string::npos;
            if (only_braces) continue;
            out += " " + eq + " \\\\\n";
            any = true;
        }
        out += "\\end{aligned}";
        if (!any) return raw; // не смогли разобрать — вернём как есть
        return out;
    }
    std::future<std::string> ocr_future_;
    std::atomic<OcrState> ocr_state_{ OcrState::Idle };
    std::string ocr_error_;

    // парсит алфавит из строки "x, y, z" -> вектор
    std::vector<std::string> parse_alphabet() const {
        std::vector<std::string> out;
        std::string cur;
        for (char c : alphabet_text) {
            if (c == ',' || c == ' ' || c == '\t' || c == '\n') {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            }
            else cur += c;
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    // строит System из текущего режима ввода
    System build_system() const; // реализовано в .cpp (зависит от sysparse)
};