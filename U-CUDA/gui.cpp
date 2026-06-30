#include "gui.h"
#include "imgui.h"
#include "implot.h"
#include "implot3d.h"
#include "plot_renderer.h"
#include "session_io.h"
#include "plot_view_2d.h"
#include "plot_view_3d.h"
#include "heatmap_view.h"
#include "app_config.h"
#include <map>
#include <memory>

// Возвращает директорию exe со слешем в конце. Реализована в app_main.cpp
// (там же используется для resolve_python_exe / library_dir).
extern std::string exe_dir();
static std::string get_exe_dir_with_sep() {
    std::string d = exe_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}

// Базовый цвет траектории по индексу НУ (единый для 2D/3D/time domain).
static ImVec4 ic_base_color(int ic_index) {
    return ImPlot::GetColormapColor(ic_index);
}

// Оттенок базового цвета по насыщенности: для переменной vi из nv внутри
// одного НУ. vi=0 — самый насыщенный, дальше бледнее. Используется в time domain,
// чтобы переменные одного НУ были видимо родственны (один тон), но различимы.
static ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    ImVec4 base = ic_base_color(ic_index);
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <cstring>
#include <algorithm>

// ---- helpers: std::string <-> ImGui ----
static bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 4096);
    buf[str.size()] = '\0';
    bool changed = ImGui::InputTextMultiline(label, buf.data(), buf.size(), size);
    if (changed) str = buf.data();
    return changed;
}
static bool InputTextStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 1024);
    buf[str.size()] = '\0';
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}

// Перехватываем символы ДО того, как ImGui положит их в буфер.
// Так замена ',' → '.' происходит в момент ввода и НЕ модифицирует строку
// между кадрами — иначе ImGui::InputText на каждом следующем кадре видит
// внешнюю подмену буфера и возвращает changed=true, отчего auto_recompute
// триггерится непрерывно при наличии запятой в поле.
static int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}

// Проверка: парсится ли строка как число (или валидная дробь "a/b")?
// Важно: std::stod НЕ кидает на "5asdfaxcv" — он парсит ведущее "5"
// и тихо игнорирует остальное. Поэтому проверяем pos — что вся строка
// (после возможных пробелов) реально была сконвертирована.
// Пустая считается валидной (дефолт подставится дальше).
// "8/3" — валидная дробь, "2/x" / "8/0" / "8/" / "5asdfaxcv" — нет.
static bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;

    // Полностью ли строка v сконвертирована в число (плюс trailing whitespace)?
    auto parse_complete = [](const std::string& v) -> bool {
        if (v.empty()) return false;
        try {
            size_t pos = 0;
            std::stod(v, &pos);
            for (size_t i = pos; i < v.size(); ++i)
                if (!std::isspace(static_cast<unsigned char>(v[i]))) return false;
            return true;
        } catch (...) { return false; }
    };

    size_t slash = s.find('/');
    if (slash != std::string::npos) {
        std::string num = s.substr(0, slash);
        std::string den = s.substr(slash + 1);
        if (!parse_complete(num) || !parse_complete(den)) return false;
        try {
            // знаменатель не должен быть нулём
            return std::stod(den) != 0.0;
        } catch (...) { return false; }
    }
    return parse_complete(s);
}

static bool InputNumStr(const char* label, std::string& str, float width = 0.0f) {
    std::vector<char> buf(str.begin(), str.end());
    buf.resize(str.size() + 1024);
    buf[str.size()] = '\0';
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter, filter_comma_to_dot);
    if (changed) str = buf.data();
    // Inline-предупреждение, если содержимое не парсится как число.
    // Default из engine'а (0) всё равно применится, но пользователю
    // явно сигналим, что введённое значение игнорируется.
    if (!is_numeric_string(str)) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, using default");
    }
    return changed;
}

// ============================================================
// Вкладка System: ввод системы, методы, генерация кода
// ============================================================
static void draw_system_tab(AppModel& model, const GuiCallbacks& cb) {
    model.poll(); // забрать результат OCR, если готов

    // режим ввода
    ImGui::Text("Input mode:");
    ImGui::SameLine();
    int mode = (int)model.mode;
    ImGui::RadioButton("Image", &mode, (int)InputMode::Image); ImGui::SameLine();
    ImGui::RadioButton("LaTeX", &mode, (int)InputMode::Latex); ImGui::SameLine();
    ImGui::RadioButton("Plain", &mode, (int)InputMode::Plain);
    model.mode = (InputMode)mode;
    ImGui::Separator();

    // источник картинки
    if (model.mode == InputMode::Image) {
        if (ImGui::Button("Choose image file...")) {
            if (cb.pick_image_file) {
                std::string path = cb.pick_image_file();
                if (!path.empty()) model.start_ocr(std::make_unique<FileImageSource>(path));
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Paste from clipboard"))
            model.start_ocr(std::make_unique<ClipboardImageSource>());
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V, false))
            model.start_ocr(std::make_unique<ClipboardImageSource>());
        ImGui::SameLine();
        switch (model.ocr_state()) {
        case OcrState::Running: ImGui::TextColored(ImVec4(1, 1, 0, 1), "Recognizing..."); break;
        case OcrState::Done:    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Recognized"); break;
        case OcrState::Failed:  ImGui::TextColored(ImVec4(1, 0, 0, 1), "OCR failed: %s", model.ocr_error().c_str()); break;
        default: ImGui::TextDisabled("(no image)"); break;
        }
        ImGui::TextDisabled("Tip: Win+Shift+S to snip, then Ctrl+V or 'Paste from clipboard'.");
    }

    // поле ввода
    if (model.mode == InputMode::Image || model.mode == InputMode::Latex) {
        ImGui::Text("LaTeX (editable - fix OCR errors here):");
        InputTextMultilineStr("##latex", model.latex_text, ImVec2(-1, 90));
        if (ImGui::CollapsingHeader("LaTeX format examples")) {
            ImGui::TextDisabled(
                "Each equation on its own line, LHS must have a derivative:\n"
                "  \\dot{x} = \\sigma(y-x) \\\\\n  \\dot{y} = x(\\rho-z)-y\n"
                "Supported: \\frac{a}{b}, x^{2}, \\sin x, \\sin^{2} x, \\cdot, |x|,\n"
                "  subscripts x_{m}, greek \\sigma. Derivatives: \\dot{x}, x', dx/dt.");
        }
    }
    else {
        ImGui::Text("Equations (plain syntax):");
        InputTextMultilineStr("##plain", model.plain_text, ImVec2(-1, 90));
        if (ImGui::CollapsingHeader("Plain format examples")) {
            ImGui::TextDisabled(
                "  \\dot{x} = sigma*(y - x) \\\\\n  \\dot{y} = x*(rho - z) - y\n"
                "Use * for multiplication, ^ for powers. LHS needs \\dot{x}= or x'=.");
        }
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // вспомогательные функции
    ImGui::Checkbox("Use auxiliary functions", &model.use_aux_funcs);
    if (model.use_aux_funcs) {
        ImGui::Text("Function definitions (one per line, e.g. h(x) = m_1 x + ...):");
        InputTextMultilineStr("##funcs", model.func_defs_text, ImVec2(-1, 60));
        if (ImGui::CollapsingHeader("Auxiliary function examples")) {
            ImGui::TextDisabled(
                "h(x) = m_1 x + \\frac{1}{2}(m_0-m_1)(|x+1| - |x-1|)\n"
                "Then call h(x) in equations. Body is inlined.\n"
                "IMPORTANT: function params (m_0, m_1) must be in the alphabet too.");
        }
    }

    // ----- Variables / Parameters (новый раздельный формат) -----
    ImGui::Text("Variables:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Auto-detect")) {
        // Scan latex_text (or plain_text если режим Plain) и заполнить vars/params.
        const std::string& src = (model.mode == InputMode::Plain)
                                 ? model.plain_text
                                 : model.latex_text;
        DetectedAlphabet det = detect_alphabet(src);
        auto join = [](const std::vector<std::string>& v) {
            std::string out;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) out += ", ";
                out += v[i];
            }
            return out;
        };
        model.vars_text   = join(det.vars);
        model.params_text = join(det.params);
    }
    ImGui::TextDisabled("Comma-sep, e.g. x,y,z. These are X[0..N-1] in KRS.");
    InputTextStr("##vars_text", model.vars_text);

    ImGui::Text("Parameters:");
    ImGui::TextDisabled("Comma-sep, e.g. sigma,rho,beta. These are a[1..M] in KRS.");
    InputTextStr("##params_text", model.params_text);

    // ----- Legacy: единый алфавит -----
    if (ImGui::CollapsingHeader("Legacy: single alphabet field")) {
        ImGui::TextDisabled("Used by older systems where vars/params live in one list.\n"
                            "If Variables AND Parameters above are both filled, this is ignored.");
        InputTextStr("##alphabet", model.alphabet_text);
    }

    // порядок параметров
    ImGui::Text("Parameter order in a[]:");
    ImGui::SameLine();
    int porder = (int)model.param_order;
    ImGui::RadioButton("as in alphabet", &porder, (int)ParamOrder::AsInAlphabet); ImGui::SameLine();
    ImGui::RadioButton("as in system", &porder, (int)ParamOrder::AsInSystem);
    model.param_order = (ParamOrder)porder;

    ImGui::Separator();

    // методы
    ImGui::Text("Schemes to generate:");
    if (ImGui::Button("Select all")) model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))  model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = model.scheme_dopri78 = model.scheme_cd = false;
    ImGui::Checkbox("Euler", &model.scheme_euler); ImGui::SameLine();
    ImGui::Checkbox("Euler-Cromer", &model.scheme_cromer); ImGui::SameLine();
    ImGui::Checkbox("Explicit Midpoint", &model.scheme_midpoint); ImGui::SameLine();
    ImGui::Checkbox("RK4", &model.scheme_rk4); ImGui::SameLine();
    ImGui::Checkbox("DOPRI78", &model.scheme_dopri78); ImGui::SameLine();
    ImGui::Checkbox("CD", &model.scheme_cd);

    // ----- Custom KRS schemes (raw C/CUDA код вместо codegen) -----
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Custom KRS schemes",
        model.custom_schemes.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled(
            "Raw C/CUDA in calculateDiscreteModel body. Available: X[0..N-1] (vars),\n"
            "a[1..M] (params), h (step), AMOUNTOFX. if/for/while + math functions OK.");

        // существующие схемы
        int to_delete = -1;
        for (int i = 0; i < (int)model.custom_schemes.size(); ++i) {
            auto& cs = model.custom_schemes[i];
            ImGui::PushID(i);
            ImGui::SetNextItemWidth(220);
            std::string name_label = "name##cs_name_" + std::to_string(i);
            InputTextStr(name_label.c_str(), cs.name);
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) to_delete = i;
            std::string body_label = "##cs_body_" + std::to_string(i);
            InputTextMultilineStr(body_label.c_str(), cs.body, ImVec2(-1, 100));
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (to_delete >= 0) model.custom_schemes.erase(model.custom_schemes.begin() + to_delete);

        // блокируем добавление с уже существующим/built-in именем
        static const char* builtin_names[] = {
            "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD"
        };
        if (ImGui::Button("+ Add custom scheme")) {
            // подобрать уникальное имя "Custom N"
            int n = (int)model.custom_schemes.size() + 1;
            std::string candidate;
            auto name_clash = [&](const std::string& nm) {
                for (const char* b : builtin_names) if (nm == b) return true;
                for (const auto& cs : model.custom_schemes) if (cs.name == nm) return true;
                return false;
            };
            do { candidate = "Custom " + std::to_string(n++); } while (name_clash(candidate));
            CustomScheme cs; cs.name = candidate;
            model.custom_schemes.push_back(std::move(cs));
        }

        // подсветка конфликтов
        for (const auto& cs : model.custom_schemes) {
            for (const char* b : builtin_names) {
                if (cs.name == b) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                        "  '%s' conflicts with a built-in scheme name; rename it.",
                        cs.name.c_str());
                    break;
                }
            }
        }
        // дубликаты между custom
        for (size_t i = 0; i < model.custom_schemes.size(); ++i) {
            for (size_t j = i + 1; j < model.custom_schemes.size(); ++j) {
                if (!model.custom_schemes[i].name.empty() &&
                    model.custom_schemes[i].name == model.custom_schemes[j].name) {
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                        "  duplicate custom name '%s'.",
                        model.custom_schemes[i].name.c_str());
                }
            }
        }
    }

    if (ImGui::Button("Generate")) model.generate();
    if (!model.error_message.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s", model.error_message.c_str());
    }

    if (!model.generated_code.empty()) {
        ImGui::Separator();
        ImGui::Text("Generated code:");
        if (ImGui::Button("Copy")) {
            if (cb.set_clipboard_text) cb.set_clipboard_text(model.generated_code);
        }
        ImGui::InputTextMultiline("##code",
            (char*)model.generated_code.c_str(), model.generated_code.size() + 1,
            ImVec2(-1, 220), ImGuiInputTextFlags_ReadOnly);
    }
}

// ============================================================
// Вкладка Parameters: НУ, значения/диапазоны параметров, шаг
// ============================================================
static void draw_parameters_tab(AppModel& model) {
    ImGui::TextDisabled("Parameter fields appear after parsing the system.");
    if (ImGui::Button("Refresh from system")) {
        model.refresh_symbols();
    }
    if (!model.error_message.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", model.error_message.c_str());
    }
    ImGui::Separator();

    // шаг дискретизации
    ImGui::Text("Discretization step h:");
    ImGui::SameLine();
    InputNumStr("##step_h", model.step_h, 120);
    ImGui::TextDisabled("(leave empty to skip)");

    ImGui::Spacing();

    // начальные условия
    if (!model.known_vars.empty()) {
        ImGui::SeparatorText("Initial conditions");
        for (const auto& v : model.known_vars) {
            ImGui::Text("%s(0) =", v.c_str());
            ImGui::SameLine();
            std::string id = "##ic_" + v;
            InputNumStr(id.c_str(), model.init_conditions[v], 140);
        }
    }

    ImGui::Spacing();

    // параметры: значение
    if (!model.known_params.empty()) {
        ImGui::SeparatorText("Parameters (value)");
        if (ImGui::BeginTable("params", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("value");
            ImGui::TableHeadersRow();
            for (const auto& p : model.known_params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                { std::string id = "##val_" + p; InputNumStr(id.c_str(), model.param_values[p], 100); }
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Empty fields are left unset.");
    }

    if (model.known_vars.empty() && model.known_params.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No symbols yet. Enter a system and press 'Refresh from system'.");
    }
}

// ============================================================
// Вкладка Library: имя, заметка, сохранение, список систем
// ============================================================
static void draw_library_tab(AppModel& model, SystemLibrary& lib) {
    // кнопка начать новую систему с нуля (сбрасывает loaded_name)
    if (ImGui::Button("New (clear all)")) {
        model.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("clears all fields to start a fresh system");
    ImGui::Separator();

    // имя и заметка
    ImGui::Text("System name:");
    InputTextStr("##name", model.name);
    ImGui::Text("Note (reference, link, comments):");
    InputTextMultilineStr("##note", model.note, ImVec2(-1, 50));

    if (ImGui::Button("Save")) {
        try {
            bool renaming = (!model.loaded_name.empty() && model.loaded_name != model.name);
            // проверка уникальности: имя занято другой системой?
            // (занято, если есть система с таким именем, и это не та, что редактируем)
            if (model.name != model.loaded_name
                && lib.exists(model.name)) {
                model.error_message = "Name '" + model.name + "' already exists";
            }
            else {
                if (renaming && lib.exists(model.loaded_name)) {
                    lib.rename(model.loaded_name, model.name);
                }
                model.name = lib.save(model.to_record());
                model.loaded_name = model.name;
                model.error_message.clear();
            }
        }
        catch (const std::exception& e) { model.error_message = e.what(); }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save as copy")) {
        // сохранить отдельную копию, не трогая текущую систему на диске
        SystemRecord r = model.to_record();
        r.name = (model.name.empty() ? std::string("Untitled") : model.name) + " (copy)";
        try { lib.save(r); }
        catch (const std::exception& e) { model.error_message = e.what(); }
    }
    if (!model.error_message.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "%s", model.error_message.c_str());
    }

    ImGui::Separator();
    ImGui::Text("Saved systems:");

    // список с кнопками Load / Duplicate / Delete
    std::vector<std::string> names = lib.list();
    if (names.empty()) {
        ImGui::TextDisabled("(library is empty)");
    }
    else {
        if (ImGui::BeginTable("libtbl", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("");
            ImGui::TableSetupColumn("");
            ImGui::TableSetupColumn("");
            for (const auto& n : names) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", n.c_str());
                // Бэйдж "N custom schemes" если у системы они есть.
                int cs_count = 0;
                try { cs_count = (int)lib.load(n).custom_schemes.size(); } catch (...) {}
                if (cs_count > 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.6f, 0.85f, 1.0f, 1.0f),
                        " [%d custom]", cs_count);
                }
                ImGui::TableSetColumnIndex(1);
                {
                    std::string id = "Load##" + n;
                    if (ImGui::SmallButton(id.c_str())) {
                        try { model.from_record(lib.load(n)); }
                        catch (const std::exception& e) { model.error_message = e.what(); }
                    }
                }
                ImGui::TableSetColumnIndex(2);
                {
                    std::string id = "Duplicate##" + n;
                    if (ImGui::SmallButton(id.c_str())) {
                        try { lib.duplicate(n); }
                        catch (const std::exception& e) { model.error_message = e.what(); }
                    }
                }
                ImGui::TableSetColumnIndex(3);
                {
                    std::string id = "Delete##" + n;
                    if (ImGui::SmallButton(id.c_str())) lib.remove(n);
                }
            }
            ImGui::EndTable();
        }
    }
}


// ============================================================
// РЕЖИМ АНАЛИЗА: пространство фазовых портретов
// ============================================================

// Панель настроек сессии: параметры (общие), НУ (список), проекции (список),
// время/шаг, метод (заглушка), кнопка пересчёта.
static void draw_phase_controls(AppModel& model, SystemLibrary& lib) {
    PhaseAnalysisSession& s = model.phase_session;

    bool changed = false;

    ImGui::Text("Phase portrait analysis");
    ImGui::TextDisabled("Changes here are NOT saved to the library (sandbox).");

    // выбор системы из библиотеки
    ImGui::Text("System:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string current = model.name.empty() ? "(current)" : model.name;
    // Во время async-расчёта смена системы запрещена — результат worker'а
    // применится к уже подменённой сессии.
    if (s.in_flight) ImGui::BeginDisabled();
    if (ImGui::BeginCombo("##syssel", current.c_str())) {
        for (const auto& nm : lib.list()) {
            if (ImGui::Selectable(nm.c_str(), model.name == nm)) {
                try {
                    model.from_record(lib.load(nm));   // загрузить систему (базовые значения)
                    model.start_phase_analysis();      // подготовить сессию из эталона
                    // поверх эталона — последняя рабочая сессия, если есть
                    std::string j = lib.load_session(model.loaded_name, "_last");
                    if (!j.empty()) {
                        session_from_json(j, model.phase_session);
                    }
                }
                catch (...) {}
            }
        }
        ImGui::EndCombo();
    }
    if (s.in_flight) ImGui::EndDisabled();
    ImGui::Separator();

    // --- именованные сессии (для текущей системы) ---
    if (!model.loaded_name.empty()) {
        ImGui::Spacing(); ImGui::Spacing();
        ImGui::Text("Session:"); ImGui::SameLine();
        static int sel_session = -1;            // индекс выбранной в комбобоксе
        static char session_name[128] = "";     // имя для "Save as"
        std::vector<std::string> sess = lib.list_sessions(model.loaded_name);

        // загрузка состояния выбранной сессии (вызывается при выборе/при Save поверх)
        auto load_session_into = [&](const std::string& name) {
            std::string j = lib.load_session(model.loaded_name, name);
            if (!j.empty()) {
                session_from_json(j, model.phase_session);
                model.phase_session.fit_request = true; // оси подстроятся при recompute
            }
            };

        // комбобокс выбора сессии — загружается сразу при выборе (как система)
        ImGui::SetNextItemWidth(160);
        const char* preview = (sel_session >= 0 && sel_session < (int)sess.size())
            ? sess[sel_session].c_str() : "(select)";
        if (ImGui::BeginCombo("##sesssel", preview)) {
            for (int k = 0; k < (int)sess.size(); ++k)
                if (ImGui::Selectable(sess[k].c_str(), sel_session == k)) {
                    sel_session = k;
                    model.error_message.clear();      // действие -> чистим ошибку
                    load_session_into(sess[k]);       // грузим сразу (recompute вручную)
                }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        // перезаписать ВЫБРАННУЮ сессию текущим состоянием (редактирование сессии)
        if (ImGui::Button("Save##sess")) {
            model.error_message.clear();
            if (sel_session >= 0 && sel_session < (int)sess.size())
                lib.save_session(model.loaded_name, sess[sel_session], session_to_json(s));
            else
                model.error_message = "No session selected";
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete##sess")) {
            model.error_message.clear();
            if (sel_session >= 0 && sel_session < (int)sess.size()) {
                lib.remove_session(model.loaded_name, sess[sel_session]);
                sel_session = -1;
            }
        }

        // сохранить текущее состояние под НОВЫМ именем
        ImGui::SetNextItemWidth(160);
        ImGui::InputText("##sessname", session_name, sizeof(session_name));
        ImGui::SameLine();
        if (ImGui::Button("Save as##sess")) {
            model.error_message.clear();
            std::string nm = session_name;
            size_t a = nm.find_first_not_of(" \t");
            size_t b = nm.find_last_not_of(" \t");
            nm = (a == std::string::npos) ? "" : nm.substr(a, b - a + 1);
            if (nm.empty())
                model.error_message = "Session name is empty";
            else if (nm == "_last")
                model.error_message = "'_last' is reserved";
            else if (lib.has_session(model.loaded_name, nm))
                model.error_message = "Session '" + nm + "' already exists";
            else {
                lib.save_session(model.loaded_name, nm, session_to_json(s));
                session_name[0] = '\0';
            }
        }
        if (!model.error_message.empty())
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", model.error_message.c_str());
        ImGui::Spacing(); ImGui::Spacing();
        ImGui::Separator();
    }

    // метод моделирования + пользовательские схемы из системы
    ImGui::Text("Method:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(160);
    static const char* methods[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    auto is_custom_scheme = [&](const std::string& nm) {
        for (const auto& cs : s.custom_schemes) if (cs.name == nm) return true;
        return false;
    };
    if (ImGui::BeginCombo("##method", s.scheme.c_str())) {
        for (auto m : methods)
            if (ImGui::Selectable(m, s.scheme == m)) {
                s.scheme = m; s.regenerate_krs(); changed = true;
            }
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), s.scheme == cs.name)) {
                s.scheme = cs.name;
                s.use_gpu = true;   // custom требует GPU — CPU-эвалуатор их не понимает
                s.regenerate_krs();
                changed = true;
            }
        ImGui::EndCombo();
    }
    if (is_custom_scheme(s.scheme) && !s.use_gpu) {
        // Если юзер вручную выключил GPU при custom — это не сработает.
        // Подсвечиваем и не даём так стрелять себе в ногу.
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "(custom requires GPU)");
        s.use_gpu = true;
    }

    // время, шаг, децимация
    ImGui::Text("Step h:"); ImGui::SameLine();
    changed |= InputNumStr("##sh", s.step_h, 80); ImGui::SameLine();
    ImGui::Text("Time(s):"); ImGui::SameLine();
    changed |= InputNumStr("##st", s.sim_time, 70); ImGui::SameLine();
    ImGui::Text("Skip(s):"); ImGui::SameLine();
    changed |= InputNumStr("##ssk", s.skip_time, 70);
    if (s.scheme == "CD") {
        ImGui::Text("Symmetry s:"); ImGui::SameLine();
        changed |= InputNumStr("##sym", s.symmetry_s, 70);
    }
    ImGui::Text("Decimation (every Nth point):"); ImGui::SameLine();
    changed |= InputNumStr("##dec", s.decimation, 70);
    // шаг/время/децимация влияют на ось времени и сами данные: при их смене
    // просим автоскейл, чтобы time domain не "скакал" со старыми пределами.
    if (changed) s.fit_request = true;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // переключатели
    ImGui::Checkbox("Auto recompute", &s.auto_recompute); ImGui::SameLine();
    ImGui::Checkbox("Legend shows initial conditions", &s.legend_show_ic); ImGui::SameLine();
    ImGui::Checkbox("GPU", &s.use_gpu);

    ImGui::Separator();

    // параметры (общие на все проекции)
    if (!s.params.empty()) {
        ImGui::SeparatorText("Parameters");
        if (ImGui::BeginTable("aparams", 2, ImGuiTableFlags_SizingFixedFit)) {
            for (const auto& p : s.params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                std::string id = "##ap_" + p;
                changed |= InputNumStr(id.c_str(), s.param_values[p], 110);
            }
            ImGui::EndTable();
        }
    }

    // начальные условия (несколько, мультистабильность)
    ImGui::SeparatorText("Initial conditions");
    int ic_to_remove = -1;
    for (int i = 0; i < (int)s.ic_sets.size(); ++i) {
        InitialConditionSet& ic = s.ic_sets[i];
        ImGui::PushID(i);
        ImGui::Checkbox("##vis", &ic.visible); ImGui::SameLine();
        ImGui::SetNextItemWidth(70);
        InputTextStr("##label", ic.label); ImGui::SameLine();
        for (const auto& v : s.vars) {
            ImGui::Text("%s:", v.c_str()); ImGui::SameLine();
            std::string id = "##icv_" + v;
            changed |= InputNumStr(id.c_str(), ic.values[v], 60); ImGui::SameLine();
        }
        if (ImGui::SmallButton("X")) ic_to_remove = i;
        ImGui::PopID();
    }
    if (ic_to_remove >= 0) { s.remove_ic(ic_to_remove); changed = true; }
    if (ImGui::Button("Add initial condition")) { s.add_ic(); }

    // проекции
    ImGui::SeparatorText("Projections");
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        ImGui::PushID(1000 + i);
        ImGui::SetNextItemWidth(90);
        InputTextStr("##plabel", pr.label); ImGui::SameLine();
        // тип проекции
        ImGui::SetNextItemWidth(110);
        const char* tnames[] = { "Phase 2D", "Time domain", "Phase 3D" };
        int t = (int)pr.type;
        if (ImGui::Combo("##ptype", &t, tnames, 3)) { pr.type = (ProjType)t; s.fit_request = true; }
        ImGui::SameLine();

        if (pr.type == ProjType::Phase2D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##px", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(55);
            if (ImGui::BeginCombo("##py", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
        }
        else if (pr.type == ProjType::Phase3D) {
            ImGui::Text("X:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3x", s.vars.empty() ? "-" : s.vars[pr.axis_x < (int)s.vars.size() ? pr.axis_x : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_x == k)) { pr.axis_x = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Y:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3y", s.vars.empty() ? "-" : s.vars[pr.axis_y < (int)s.vars.size() ? pr.axis_y : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_y == k)) { pr.axis_y = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
            ImGui::SameLine(); ImGui::Text("Z:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            if (ImGui::BeginCombo("##p3z", s.vars.empty() ? "-" : s.vars[pr.axis_z < (int)s.vars.size() ? pr.axis_z : 0].c_str())) {
                for (int k = 0; k < (int)s.vars.size(); ++k)
                    if (ImGui::Selectable(s.vars[k].c_str(), pr.axis_z == k)) { pr.axis_z = k; s.fit_request = true; }
                ImGui::EndCombo();
            }
        }
        else { // TimeDomain — галочки переменных
            // синхронизируем размер show_var
            if ((int)pr.show_var.size() != (int)s.vars.size())
                pr.show_var.assign(s.vars.size(), true);
            ImGui::Text("vars:"); ImGui::SameLine();
            for (int k = 0; k < (int)s.vars.size(); ++k) {
                bool v = pr.show_var[k];
                if (ImGui::Checkbox(s.vars[k].c_str(), &v)) { pr.show_var[k] = v; }
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) pr_to_remove = i;
        ImGui::PopID();
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
    if (ImGui::Button("Add projection")) { s.add_projection(); }
    ImGui::SameLine();
    if (ImGui::Button("Reset windows layout")) { s.layout_generation++; }

    ImGui::Separator();
    // Debug panel: show what kernel/integrator actually computes. Useful for
    // comparing CPU vs GPU runs and catching parsing surprises.
    if (ImGui::CollapsingHeader("Debug: KRS body & parsed values")) {
        ImGui::TextDisabled("KRS body — what NVRTC compiles for GPU. On CPU the same RHS\n"
                            "is parsed into an AST and interpreted. For CD the CPU and GPU\n"
                            "algorithms differ (see the second panel). Values below are\n"
                            "the exact doubles both paths parse.");

        if (ImGui::Button("Regenerate")) s.regenerate_krs();
        ImGui::SameLine();
        ImGui::TextDisabled("(rebuilds KRS from the current system/scheme)");

        // GPU KRS body (what NVRTC compiles).
        ImGui::SeparatorText("KRS (GPU, NVRTC)");
        std::string body_gpu = s.krs_code.empty()
            ? std::string("(empty — press Regenerate or change Method)")
            : s.krs_code;
        std::vector<char> buf_gpu(body_gpu.begin(), body_gpu.end());
        buf_gpu.resize(body_gpu.size() + 1);
        buf_gpu[body_gpu.size()] = '\0';
        ImGui::InputTextMultiline("##krs_gpu", buf_gpu.data(), buf_gpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);

        // CPU equivalent: identical to GPU for Euler/RK4/etc; for CD it's the
        // 4-simple-iterations form that integrator.cpp::step_cd actually runs.
        ImGui::SeparatorText("KRS (CPU equivalent)");
        std::string body_cpu;
        try {
            body_cpu = codegen_scheme_cpu_equivalent(s.sys, scheme_from_name(s.scheme));
        } catch (...) {
            body_cpu = "(generation failed)";
        }
        if (body_cpu.empty()) body_cpu = "(empty — same as GPU for non-CD schemes)";
        std::vector<char> buf_cpu(body_cpu.begin(), body_cpu.end());
        buf_cpu.resize(body_cpu.size() + 1);
        buf_cpu[body_cpu.size()] = '\0';
        ImGui::InputTextMultiline("##krs_cpu", buf_cpu.data(), buf_cpu.size(),
            ImVec2(-1, 200), ImGuiInputTextFlags_ReadOnly);
        if (s.scheme != "CD")
            ImGui::TextDisabled("(for %s CPU and GPU evaluate the same AST — texts match)", s.scheme.c_str());
        else
            ImGui::TextDisabled("(for CD: GPU uses analytic solve for linear vars; CPU always uses 4 iterations)");

        ImGui::SeparatorText("Parsed inputs (double, %.17g)");
        auto parse = [](const std::string& v, double def) -> double {
            if (v.empty()) return def;
            size_t sl = v.find('/');
            if (sl != std::string::npos) {
                double n = std::atof(v.substr(0, sl).c_str());
                double d = std::atof(v.substr(sl + 1).c_str());
                if (d != 0) return n / d;
            }
            return std::atof(v.c_str());
        };

        ImGui::Text("h        = %.17g", parse(s.step_h, 0.01));
        ImGui::Text("a[0] (s) = %.17g", parse(s.symmetry_s, 0.5));
        for (size_t j = 0; j < s.params.size(); ++j) {
            auto it = s.param_values.find(s.params[j]);
            double v = (it != s.param_values.end()) ? parse(it->second, 0.0) : 0.0;
            ImGui::Text("a[%zu] (%s) = %.17g", j + 1, s.params[j].c_str(), v);
        }
        for (size_t k = 0; k < s.ic_sets.size(); ++k) {
            ImGui::Text("%s:", s.ic_sets[k].label.c_str());
            for (size_t i = 0; i < s.vars.size(); ++i) {
                auto it = s.ic_sets[k].values.find(s.vars[i]);
                double v = (it != s.ic_sets[k].values.end()) ? parse(it->second, 0.0) : 0.0;
                ImGui::Text("  X[%zu] (%s) = %.17g", i, s.vars[i].c_str(), v);
            }
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Reset to defaults")) {
        try {
            if (!model.loaded_name.empty()) {
                model.from_record(lib.load(model.loaded_name)); // эталон с диска
                model.start_phase_analysis();
            }
        }
        catch (...) {}
    }

    ImGui::Separator();
    // Recompute по кнопке или Ctrl+R, дисейблится во время async-расчёта.
    // Авто-сохранение _last делается в draw_gui::poll(), когда результат готов.
    bool do_recompute = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Recomputing...", ImVec2(-1, 0));
        ImGui::EndDisabled();
    }
    else {
        do_recompute = ImGui::Button("Recompute (Ctrl+R)", ImVec2(-1, 0));
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false)) do_recompute = true;
        if (s.auto_recompute && changed) do_recompute = true;
    }
    if (do_recompute) {
        s.recompute_async();
    }

    if (!s.result.error.empty())
        ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "%s", s.result.error.c_str());
}

// Рисует окна проекций (каждая — отдельное docking-окно с графиком).
static void draw_projection_windows(AppModel& model) {
    PhaseAnalysisSession& s = model.phase_session;
    const AnalysisResult& res = s.result;
    // Свой offscreen-рендерер (FBO/текстура) на КАЖДУЮ проекцию: иначе все окна
    // показывали бы одну общую текстуру (геометрию последней отрисованной).
    // PlotRenderer некопируемый -> храним через unique_ptr, подгоняем под число проекций.
    static std::vector<std::unique_ptr<PlotRenderer>> renderers;
    if ((int)renderers.size() != (int)s.projections.size()) {
        renderers.clear();
        for (size_t k = 0; k < s.projections.size(); ++k)
            renderers.push_back(std::make_unique<PlotRenderer>());
    }
    int pr_to_remove = -1;
    for (int i = 0; i < (int)s.projections.size(); ++i) {
        Projection& pr = s.projections[i];
        PlotRenderer& renderer = *renderers[i]; // рендерер этой проекции
        std::string title = pr.label + "##proj" + std::to_string(i) + "_g" + std::to_string(s.layout_generation);
        bool open = true; // крестик закрытия
        // Начальные позиция и размер (только при первом появлении).
        // Каскад слева-сверху: каждое следующее окно чуть смещено.
        // ИЗМЕНИТЬ РАЗМЕР МОЖНО ЗДЕСЬ: ImVec2(ширина, высота) в пикселях.
        float ox = 60.0f + (i % 5) * 35.0f;
        float oy = 80.0f + (i % 5) * 35.0f;
        ImGui::SetNextWindowPos(ImVec2(ox, oy), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &open)) {
            ImGui::PushID(i);   // разделить ID внутренних виджетов между окнами проекций
            if (!res.ok || res.trajectories.empty()) {
                ImGui::TextDisabled("No data. Press Recompute.");
            }
            ////////////////////////////////////////
            else if (pr.type == ProjType::Phase2D) {
                int ax = pr.axis_x, ay = pr.axis_y;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    // создать вьюер при первой отрисовке
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    // обновить имена осей
                    pr.view2d->x_axis.name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    pr.view2d->y_axis.name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];

                    // Toolbar над плотом: opt-in custom line styling (ImDrawList-путь
                    // с настраиваемой толщиной + α). По дефолту выключено → быстрый
                    // GL shader-line путь (1px, α=1). Текущая отрисовка не ломается.
                    ImGui::Checkbox("Custom line style##phase2d", &pr.custom_line_style);
                    if (pr.custom_line_style) {
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Line width##phase2d", &pr.line_width, 0.1f, 5.0f, "%.2f");
                        ImGui::SameLine(); ImGui::SetNextItemWidth(120);
                        ImGui::SliderFloat("Alpha##phase2d",      &pr.alpha,      0.0f, 1.0f, "%.2f");
                    }
                    pr.view2d->imdraw_lines      = pr.custom_line_style;
                    pr.view2d->line_thickness_px = pr.line_width;

                    // подготовить серии: для каждой траектории выбираем координаты по (ax, ay)
                    // храним float-массивы локально в статике, чтобы указатели жили до конца кадра
                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();
                    series_data.resize(res.trajectories.size());

                    std::vector<PlotSeriesInput> series_in;
                    series_in.reserve(res.trajectories.size());

                    // видимость: глобальная — из живых галочек НУ (меняется без recompute)
                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 2);
                        for (const auto& pt : traj) {
                            buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic && k < res.ic_text.size()) lab = res.ic_text[k];

                        PlotSeriesInput si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 2);
                        si.color = ic_base_color((int)k);
                        // В custom_line_style режиме применяем α к цвету IC
                        // (rendering: ImDrawList использует color.w как alpha).
                        if (pr.custom_line_style) si.color.w = pr.alpha;
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 100 + ax * 10 + ay;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->render(renderer, origin, avail, i, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            else if (pr.type == ProjType::TimeDomain) {
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view2d) pr.view2d = std::make_unique<Plot2DView>();

                    double h = atof(s.step_h.c_str()); if (h <= 0) h = 0.01;
                    int dec = atoi(s.decimation.c_str()); if (dec < 1) dec = 1;
                    double dt = h * dec;
                    int nvars = (int)s.vars.size();

                    // синхронизируем show_var с числом переменных
                    if ((int)pr.show_var.size() != nvars)
                        pr.show_var.assign(nvars, true);

                    pr.view2d->x_axis.name = "t";
                    pr.view2d->y_axis.name = "value";

                    pr.view2d->pad_x = false;
                    pr.view2d->show_zero_x = false;

                    // серии: одна на (траектория k, видимая переменная vi)
                    // храним буферы в статике, чтобы указатели жили до render
                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();

                    std::vector<PlotSeriesInput> series_in;
                    std::vector<bool> init_vis;
                    std::vector<bool> glob_vis;

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        if (traj.empty()) continue;
                        int n = (int)traj.size();
                        // видимость НУ — из живой галочки (без recompute)
                        bool ic_vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;

                        for (int vi = 0; vi < nvars; ++vi) {
                            if (vi < (int)pr.show_var.size() && !pr.show_var[vi]) continue;

                            series_data.emplace_back();
                            auto& buf = series_data.back();
                            buf.reserve(n * 2);
                            for (int t = 0; t < n; ++t) {
                                buf.push_back((float)(t * dt));
                                buf.push_back((float)traj[t][vi < (int)traj[t].size() ? vi : 0]);
                            }

                            std::string base = (k < res.labels.size()) ? res.labels[k] : ("IC" + std::to_string(k + 1));
                            std::string who = (s.legend_show_ic && k < res.ic_text.size()) ? res.ic_text[k] : base;
                            std::string lab = s.vars[vi] + " [" + who + "]";

                            PlotSeriesInput si;
                            si.points = buf.data();
                            si.n_points = n;
                            si.color = ic_var_shade((int)k, vi, nvars);
                            si.label = lab;
                            series_in.push_back(si);
                            init_vis.push_back(true);   // локальная (легенда) стартует с видимости НУ
                            glob_vis.push_back(ic_vis);   // глобальная = живая галочка НУ
                        }
                    }

                    int data_gen = s.data_generation * 1000 + (int)series_in.size();

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view2d->render(renderer, origin, avail, i, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            else if (pr.type == ProjType::Phase3D) {
                int ax = pr.axis_x, ay = pr.axis_y, az = pr.axis_z;
                if (!res.ok || res.trajectories.empty()) {
                    ImGui::TextDisabled("No data.");
                }
                else {
                    if (!pr.view3d) pr.view3d = std::make_unique<Plot3DView>();

                    pr.view3d->x_name = s.vars.empty() ? "x" : s.vars[ax < (int)s.vars.size() ? ax : 0];
                    pr.view3d->y_name = s.vars.empty() ? "y" : s.vars[ay < (int)s.vars.size() ? ay : 0];
                    pr.view3d->z_name = s.vars.empty() ? "z" : s.vars[az < (int)s.vars.size() ? az : 0];

                    static std::vector<std::vector<float>> series_data;
                    series_data.clear();
                    series_data.resize(res.trajectories.size());

                    std::vector<PlotSeriesInput3D> series_in;
                    series_in.reserve(res.trajectories.size());

                    std::vector<bool> init_vis(res.trajectories.size(), true);
                    std::vector<bool> glob_vis(res.trajectories.size(), true);

                    for (size_t k = 0; k < res.trajectories.size(); ++k) {
                        const auto& traj = res.trajectories[k];
                        auto& buf = series_data[k];
                        buf.reserve(traj.size() * 3);
                        for (const auto& pt : traj) {
                            buf.push_back((float)pt[ax < (int)pt.size() ? ax : 0]);
                            buf.push_back((float)pt[ay < (int)pt.size() ? ay : 0]);
                            buf.push_back((float)pt[az < (int)pt.size() ? az : 0]);
                        }
                        std::string lab = (k < res.labels.size()) ? res.labels[k] : ("IC " + std::to_string(k + 1));
                        if (s.legend_show_ic && k < res.ic_text.size()) lab = res.ic_text[k];

                        PlotSeriesInput3D si;
                        si.points = buf.data();
                        si.n_points = (int)(buf.size() / 3);
                        si.color = ic_base_color((int)k);
                        si.label = lab;
                        series_in.push_back(si);

                        bool vis = (k < s.ic_sets.size()) ? s.ic_sets[k].visible : true;
                        glob_vis[k] = vis;
                        init_vis[k] = true;
                    }

                    int data_gen = s.data_generation * 1000 + ax * 100 + ay * 10 + az;

                    ImVec2 avail = ImGui::GetContentRegionAvail();
                    ImVec2 origin = ImGui::GetCursorScreenPos();

                    pr.view3d->render(renderer, origin, avail, i, data_gen,
                        series_in, init_vis, glob_vis, s.fit_request);
                }
            }
            ImGui::PopID();
        }
        ImGui::End();
        if (!open) pr_to_remove = i; // окно закрыто крестиком
    }
    if (pr_to_remove >= 0) s.remove_projection(pr_to_remove);
    // автоскейл применён ко всем окнам в этом кадре — сбрасываем запрос
    s.fit_request = false;
}

// ============================================================
// Parametric: контролы + scatter-plot 1D-бифуркации через наш GL-renderer
// ============================================================
// Рисует контролы одной БД внутри её таба. Возвращает true, если пользователь
// нажал Run для этой БД (внешний код может также взвести Run через Ctrl+R).
static bool draw_diagram_controls(BifurcationAnalysisSession& s, int idx) {
    BifurcationDiagramConfig& bd = s.diagrams[idx];

    ImGui::SetNextItemWidth(160);
    InputTextStr("Label", bd.label);
    ImGui::SameLine();
    ImGui::Checkbox("visible", &bd.visible);
    ImGui::Separator();

    // ----- Scheme (built-in + custom) -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", bd.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, bd.scheme == m)) bd.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), bd.scheme == cs.name))
                bd.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // ----- Sweep target (parameter ИЛИ initial condition) -----
    // Один combo с разделителем: сверху параметры, снизу переменные (IC).
    // Выбор переменной → BD строится по начальному условию (par_or_var = 0
    // в engine), runtime-флаг — никакой пересборки PTX.
    if (!s.params.empty() || !s.vars.empty()) {
        // Бочки валидации индексов на случай старых сохранений.
        if (bd.param_index < 0 || bd.param_index >= (int)s.params.size())
            bd.param_index = 0;
        if (bd.var_sweep_index < 0 || bd.var_sweep_index >= (int)s.vars.size())
            bd.var_sweep_index = 0;

        std::string preview;
        if (bd.sweep_over_var && !s.vars.empty())
            preview = s.vars[bd.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[bd.param_index];
        else
            preview = "?";

        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !bd.sweep_over_var && bd.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    bd.sweep_over_var = false;
                    bd.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = bd.sweep_over_var && bd.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    bd.sweep_over_var = true;
                    bd.var_sweep_index = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr("Param lo", bd.param_lo_text, 120);
    InputNumStr("Param hi", bd.param_hi_text, 120);

    // Continuation: каждая следующая точка стартует с конечного x[] предыдущей
    // (single-thread GPU-kernel; см. parametric_engine::run_bif1d_continuation).
    // При активации справа появляются radio forward/backward — для гистерезиса.
    // В 2D-режиме отключён: run_bif2d использует свой 3-kernel pipeline, флаг
    // continuation им игнорируется — лочим UI и принудительно сбрасываем флаги,
    // чтобы скрытое состояние не утекло в следующий 1D-Run.
    if (bd.mode_2d) {
        bd.continuation = false;
        bd.continuation_reverse = false;
    }
    {
        ImGui::BeginDisabled(bd.mode_2d);
        bool cont = bd.continuation;
        if (ImGui::Checkbox("Continuation", &cont)) bd.continuation = cont;
        if (bd.continuation) {
            ImGui::SameLine();
            int dir = bd.continuation_reverse ? 1 : 0;
            ImGui::RadioButton("forward",  &dir, 0); ImGui::SameLine();
            ImGui::RadioButton("backward", &dir, 1);
            bd.continuation_reverse = (dir == 1);
        }
        ImGui::EndDisabled();
        if (bd.mode_2d)
            ImGui::TextDisabled("(unavailable in 2D mode)");
    }

    ImGui::Separator();

    // ----- 2D mode: хитмап «период»(p1, p2) через DBSCAN -----
    ImGui::Checkbox("2D mode (period heatmap)", &bd.mode_2d);
    if (bd.mode_2d) {
        ImGui::Indent();
        if (!s.params.empty() || !s.vars.empty()) {
            if (bd.param_index_2 < 0 || bd.param_index_2 >= (int)s.params.size())
                bd.param_index_2 = 0;
            if (bd.var_sweep_index_2 < 0 || bd.var_sweep_index_2 >= (int)s.vars.size())
                bd.var_sweep_index_2 = 0;
            std::string preview2;
            if (bd.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[bd.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[bd.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !bd.sweep_over_var_2 && bd.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        bd.sweep_over_var_2 = false;
                        bd.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = bd.sweep_over_var_2 && bd.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        bd.sweep_over_var_2 = true;
                        bd.var_sweep_index_2 = i;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr("Param2 lo",  bd.param_lo_2_text, 120);
        InputNumStr("Param2 hi",  bd.param_hi_2_text, 120);
        InputNumStr("DBSCAN eps", bd.eps_dbscan_text, 120);
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();

    // ----- Variable + resolution + inter-peaks -----
    if (!s.vars.empty()) {
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());
        if (bd.writable_var < 0 || bd.writable_var >= (int)s.vars.size()) bd.writable_var = 0;
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Writable var", &bd.writable_var, items.data(), (int)items.size());
    }
    InputNumStr("Resolution", bd.n_pts_text, 120);
    if (!bd.mode_2d)
        if (ImGui::Checkbox("Plot inter-peaks instead of peak values", &bd.plot_inter_peaks))
            bd.fit_request = true;

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              bd.h_text,           120);
    if (bd.scheme == "CD")
        InputNumStr("symmetry s", bd.symmetry_s,       120);
    InputNumStr("computing time", bd.t_max_text,       120);
    InputNumStr("transient time", bd.transient_text,   120);
    InputNumStr("decimator",      bd.pre_scaller_text, 120);
    InputNumStr("max value",      bd.max_value_text,   120);

    ImGui::Separator();
    ImGui::Text("CSV output:");
    ImGui::Checkbox("Save to file", &bd.csv_save_enabled);
    InputTextStr("##csv_path", bd.csv_output_path);
    ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");

    ImGui::Separator();
    ImGui::Text("Initial conditions:");
    for (const auto& v : s.vars) {
        std::string id = "##bd_ic_" + v;
        ImGui::PushID(v.c_str());
        InputNumStr(v.c_str(), bd.initial_conditions[v], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Parameters:");
    for (const auto& p : s.params) {
        ImGui::PushID(p.c_str());
        InputNumStr(p.c_str(), bd.param_values[p], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    // Кнопка Run этой БД. Дисейблится во время async-расчёта (любой БД сессии).
    bool do_run = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(160, 0));
        ImGui::EndDisabled();
    }
    else {
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    }

    if (bd.mode_2d) {
        if (bd.last_run_2d_ok) {
            int total = (int)bd.result_2d.flags.size();
            int diverged = 0;
            for (int f : bd.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, period(min..max) = %.0f..%.0f",
                bd.result_2d.n_pts, bd.result_2d.n_pts,
                bd.result_2d.min_val, bd.result_2d.max_val);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!bd.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##par_err_2d",
                const_cast<char*>(bd.last_error.c_str()),
                bd.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (bd.last_run_ok) {
            int diverged = 0, total_peaks = 0, max_peaks = 0;
            for (int f : bd.result.flags) {
                if (f < 0) ++diverged;
                else { total_peaks += f; if (f > max_peaks) max_peaks = f; }
            }
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, peaks total=%d (max per param=%d)",
                bd.result.n_pts, total_peaks, max_peaks);
            if (diverged) ImGui::TextDisabled("(%d/%d trajectories diverged)", diverged, bd.result.n_pts);
        }
        else if (!bd.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##par_err",
                const_cast<char*>(bd.last_error.c_str()),
                bd.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return do_run;
}

static void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) {
    BifurcationAnalysisSession& s = model.bifurcation_session;

    // Tab bar: одна вкладка на БД + кнопка "+" справа для добавления новой
    // БД (копия последней). Активная вкладка хранится в s.active_diagram_index
    // и используется Ctrl+R.
    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##bd_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.diagrams.size(); ++i) {
            BifurcationDiagramConfig& bd = s.diagrams[i];
            ImGui::PushID(i);
            bool open = true;
            // ID завязан на индекс, но label — на bd.label, чтобы пользователь
            // видел свежее имя сразу после редактирования.
            std::string tab_id = bd.label + "###bd_tab_" + std::to_string(i);
            // Запрещаем закрывать таб, чей расчёт сейчас идёт.
            bool can_close = !(s.in_flight && s.running_diagram_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                if (draw_diagram_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        // Кнопка "+ Add" справа. ImGuiTabItemFlags_Trailing удерживает её
        // в конце; NoTooltip убирает дефолтную подсказку.
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_diagram();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_diagram_index = active_now;
    if (to_remove >= 0) model.remove_bifurcation_diagram(to_remove);

    // Ctrl+R на уровне всей панели — запускает Run для активной вкладки.
    if (!s.in_flight && ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        run_idx = s.active_diagram_index;
    }

    if (run_idx >= 0 && run_idx < (int)s.diagrams.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
}

static void draw_bifurcation_plot(AppModel& model) {
    BifurcationAnalysisSession& s = model.bifurcation_session;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::unique_ptr<Plot2DView> view;
    // Per-diagram HeatmapView. HeatmapView::data_gen_cached is keyed only by
    // data_generation, not owner_id, so a single shared instance would bleed
    // textures between diagrams that happen to share data_generation_2d
    // (e.g. both at =1 after one Run each). Map by owner_id with lazy-init —
    // each diagram keeps its own upload, zoom/pan and colormap state.
    static std::map<unsigned, std::unique_ptr<HeatmapView>> heatmap_bd_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    if (!view) {
        view = std::make_unique<Plot2DView>();
        view->points_mode = true;
        view->show_legend = true;    // мульти-БД → нужна легенда (как в Phase)
        view->point_size_px = 2.0f;
        view->pad_x = false;         // данные впритык к боковым границам плота
        view->x_axis.name = "parameter";
        view->y_axis.name = "X";
    }
    auto get_bd_heatmap = [&](unsigned oid) -> HeatmapView& {
        auto& slot = heatmap_bd_map[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
            int cm = model.heatmap_colormap;
            if (cm >= 0 && cm <= 3) slot->colormap = (HeatmapColormap)cm;
        }
        return *slot;
    };

    if (s.diagrams.empty()) {
        ImGui::TextDisabled("No diagrams yet.");
        return;
    }

    // Активная БД решает, что рисовать. Если mode_2d=true — heatmap_bd.
    {
        int act = s.active_diagram_index;
        if (act >= 0 && act < (int)s.diagrams.size()) {
            BifurcationDiagramConfig& bdact = s.diagrams[act];
            if (bdact.mode_2d) {
                // Per-diagram heatmap by owner_id. Base + active index = unique.
                const unsigned bd_oid = 0xBD2D0000u + (unsigned)act;
                HeatmapView& hb = get_bd_heatmap(bd_oid);

                // Colormap combo + autoscale над плотом.
                static const char* cmap_names[] = { "Viridis", "Inferno", "Turbo", "Gray" };
                int cmap_idx = (int)hb.colormap;
                ImGui::SetNextItemWidth(140);
                if (ImGui::Combo("Colormap##bdhm", &cmap_idx, cmap_names, IM_ARRAYSIZE(cmap_names))) {
                    hb.colormap = (HeatmapColormap)cmap_idx;
                    model.heatmap_colormap = cmap_idx;
                    AppConfig cfg;
                    cfg.ui_scale_override = model.ui_scale_override;
                    cfg.use_builtin_font  = model.use_builtin_font;
                    cfg.heatmap_colormap  = cmap_idx;
                    cfg.basins_colormap        = model.basins_colormap;
                    cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                    cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                    cfg.basins_states_colormap = model.basins_states_colormap;
                    save_app_config(get_exe_dir_with_sep(), cfg);
                }
                ImGui::SameLine();
                ImGui::Checkbox("Autoscale color##bdhm", &hb.autoscale);
                if (!hb.autoscale) {
                    ImGui::SameLine(); ImGui::SetNextItemWidth(80);
                    ImGui::InputFloat("vmin##bdhm", &hb.manual_vmin, 0.0f, 0.0f, "%.4g");
                    ImGui::SameLine(); ImGui::SetNextItemWidth(80);
                    ImGui::InputFloat("vmax##bdhm", &hb.manual_vmax, 0.0f, 0.0f, "%.4g");
                }
                ImGui::SameLine();
                if (ImGui::Button(hb.swap_axes ? "Swap axes (on)##bdhm" : "Swap axes##bdhm"))
                    hb.swap_axes = !hb.swap_axes;

                if (!bdact.last_run_2d_ok || bdact.result_2d.values.empty()) {
                    ImGui::TextDisabled("No 2D data yet. Press Run.");
                    return;
                }

                auto ax_name = [&](bool sweep_var, int p_idx, int v_idx) -> std::string {
                    if (sweep_var)
                        return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
                    return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
                };
                hb.x_axis.name = ax_name(bdact.sweep_over_var,   bdact.param_index,   bdact.var_sweep_index);
                hb.y_axis.name = ax_name(bdact.sweep_over_var_2, bdact.param_index_2, bdact.var_sweep_index_2);

                bool fit = bdact.fit_request_2d;
                if (fit) bdact.fit_request_2d = false;

                ImVec2 avail = ImGui::GetContentRegionAvail();
                ImVec2 origin = ImGui::GetCursorScreenPos();
                hb.render(*renderer, origin, avail,
                          /*owner_id*/ bd_oid, bdact.data_generation_2d,
                          bdact.result_2d.n_pts, bdact.result_2d.n_pts,
                          bdact.result_2d.values.data(),
                          bdact.result_2d.param_lo,   bdact.result_2d.param_hi,
                          bdact.result_2d.param_lo_2, bdact.result_2d.param_hi_2,
                          bdact.result_2d.min_val, bdact.result_2d.max_val,
                          fit);
                return;
            }
        }
    }

    // Имеется ли хотя бы одна БД с готовыми данными?
    bool any_data = false;
    for (const auto& bd : s.diagrams)
        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) { any_data = true; break; }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    auto safe_stod = [](const std::string& v, double def) -> double {
        if (v.empty()) return def;
        size_t slash = v.find('/');
        if (slash != std::string::npos) {
            double num = std::atof(v.substr(0, slash).c_str());
            double den = std::atof(v.substr(slash + 1).c_str());
            if (den != 0) return num / den;
        }
        return std::atof(v.c_str());
    };

    // Подписи осей + X-fit диапазон: пробегаем по видимым БД, собираем общий
    // sweep-таргет и union (lo, hi) sweep-диапазонов. X-ось должна охватывать
    // ВСЁ что свипалось — даже если часть параметров не дала точек.
    int shared_kind = -2;   // -2 init, -1 mixed, 0 param, 1 var
    int shared_idx  = -2;
    int shared_var_idx = -2;
    double x_fit_lo = 0.0, x_fit_hi = 0.0;
    bool   x_fit_any = false;
    for (const auto& bd : s.diagrams) {
        if (!bd.visible || !bd.last_run_ok) continue;
        int kind = bd.sweep_over_var ? 1 : 0;
        int idx  = bd.sweep_over_var ? bd.var_sweep_index : bd.param_index;
        if (shared_kind == -2) { shared_kind = kind; shared_idx = idx; }
        else if (shared_kind != kind || shared_idx != idx) {
            shared_kind = -1; shared_idx = -1;
        }
        if (shared_var_idx == -2) shared_var_idx = bd.writable_var;
        else if (shared_var_idx != bd.writable_var) shared_var_idx = -1;
        // X-range: continuation сохраняет lo/hi в result, иначе берём из текстов.
        double lo = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                    ? bd.result.param_lo : safe_stod(bd.param_lo_text, 0.0);
        double hi = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                    ? bd.result.param_hi : safe_stod(bd.param_hi_text, 1.0);
        double a = std::min(lo, hi), b = std::max(lo, hi);
        if (!x_fit_any) { x_fit_lo = a; x_fit_hi = b; x_fit_any = true; }
        else { x_fit_lo = std::min(x_fit_lo, a); x_fit_hi = std::max(x_fit_hi, b); }
    }
    view->x_fit_use_explicit = x_fit_any;
    view->x_fit_min = x_fit_lo;
    view->x_fit_max = x_fit_hi;
    if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)s.params.size())
        view->x_axis.name = s.params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)s.vars.size())
        view->x_axis.name = s.vars[shared_idx] + " (IC)";
    else
        view->x_axis.name = "parameter";
    view->y_axis.name = (shared_var_idx >= 0 && shared_var_idx < (int)s.vars.size())
                          ? s.vars[shared_var_idx] : std::string("X");

    // Локальные буферы (по одному на серию). static, чтобы указатели жили
    // до конца кадра (PlotSeriesInput хранит сырые указатели).
    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != s.diagrams.size()) bufs.assign(s.diagrams.size(), {});

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis;
    std::vector<bool> glob_vis;
    series_in.reserve(s.diagrams.size());
    init_vis.reserve(s.diagrams.size());
    glob_vis.reserve(s.diagrams.size());

    bool any_fit = false;
    int  data_gen = 0;

    for (int i = 0; i < (int)s.diagrams.size(); ++i) {
        BifurcationDiagramConfig& bd = s.diagrams[i];
        auto& buf = bufs[i];
        buf.clear();
        int total_pts = 0;

        if (bd.last_run_ok && !bd.result.bifurcation_points.empty()) {
            const auto& source = bd.plot_inter_peaks ? bd.result.peak_times
                                                     : bd.result.bifurcation_points;
            // Continuation result хранит param_lo/hi-снапшот. Classical путь
            // его не заполняет — fallback на текущие текстовые поля.
            double lo = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_lo : safe_stod(bd.param_lo_text, 0.0);
            double hi = (bd.continuation && bd.result.param_hi != bd.result.param_lo)
                        ? bd.result.param_hi : safe_stod(bd.param_hi_text, 1.0);
            bool rev = bd.result.continuation_reverse;
            int npts = bd.result.n_pts;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)bd.result.flags.size() && bd.result.flags[k] < 0) continue;
                double x;
                if (rev)
                    x = (npts > 1) ? (hi - (hi - lo) * (double)k / (double)(npts - 1)) : hi;
                else
                    x = (npts > 1) ? (lo + (hi - lo) * (double)k / (double)(npts - 1)) : lo;
                if (k >= (int)source.size()) continue;
                for (double y : source[k]) {
                    buf.push_back((float)x);
                    buf.push_back((float)y);
                    ++total_pts;
                }
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color(i);
        si.label    = bd.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(bd.visible);

        // data_gen накапливает per-BD generation + toggle inter_peaks
        // (как было в одно-БД версии — чтобы тоггл триггерил перерисовку VBO).
        data_gen = data_gen * 31 + bd.data_generation * 2 + (bd.plot_inter_peaks ? 1 : 0);
        if (bd.fit_request) { any_fit = true; bd.fit_request = false; }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    view->render(*renderer, origin, avail, /*owner_id*/ 0xBE0F1D, data_gen,
                 series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// LLE: контролы (per-curve в табе) + line-plot λ(param)
// ============================================================

// Контролы одной LLE-кривой. Возвращает true, если пользователь нажал Run
// для этой кривой.
static bool draw_lle_curve_controls(LLEAnalysisSession& s, int idx) {
    LLECurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(160);
    InputTextStr("Label", c.label);
    ImGui::SameLine();
    ImGui::Checkbox("visible", &c.visible);
    ImGui::Separator();

    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC). См. BD.
    if (!s.params.empty() || !s.vars.empty()) {
        if (c.param_index < 0 || c.param_index >= (int)s.params.size())
            c.param_index = 0;
        if (c.var_sweep_index < 0 || c.var_sweep_index >= (int)s.vars.size())
            c.var_sweep_index = 0;
        std::string preview;
        if (c.sweep_over_var && !s.vars.empty())
            preview = s.vars[c.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[c.param_index];
        else
            preview = "?";
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !c.sweep_over_var && c.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    c.sweep_over_var = false;
                    c.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = c.sweep_over_var && c.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    c.sweep_over_var = true;
                    c.var_sweep_index = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr("Param lo", c.param_lo_text, 120);
    InputNumStr("Param hi", c.param_hi_text, 120);
    InputNumStr("Resolution", c.n_pts_text, 120);

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. analysis_session.h:LLECurveConfig коммент)
    // — Resolution выше применяется и к X, и к Y.
    ImGui::Checkbox("2D mode (heatmap)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        // Sweep target для второй оси — комбо, симметрично первой.
        if (!s.params.empty() || !s.vars.empty()) {
            if (c.param_index_2 < 0 || c.param_index_2 >= (int)s.params.size())
                c.param_index_2 = 0;
            if (c.var_sweep_index_2 < 0 || c.var_sweep_index_2 >= (int)s.vars.size())
                c.var_sweep_index_2 = 0;
            std::string preview2;
            if (c.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[c.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[c.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !c.sweep_over_var_2 && c.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        c.sweep_over_var_2 = false;
                        c.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = c.sweep_over_var_2 && c.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        c.sweep_over_var_2 = true;
                        c.var_sweep_index_2 = i;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr("Param2 lo", c.param_lo_2_text, 120);
        InputNumStr("Param2 hi", c.param_hi_2_text, 120);
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).");
        ImGui::Unindent();
    }

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              c.h_text,         120);
    if (c.scheme == "CD")
        InputNumStr("symmetry s", c.symmetry_s,     120);
    InputNumStr("computing time", c.t_max_text,     120);
    InputNumStr("transient time", c.transient_text, 120);
    InputNumStr("max value",      c.max_value_text, 120);

    ImGui::Separator();
    ImGui::Text("LLE (Wolf/Benettin):");
    InputNumStr("eps", c.eps_text, 120);
    InputNumStr("NT",  c.nt_text,  120);
    ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                        "between renormalizations (in time units).");

    ImGui::Separator();
    ImGui::Text("CSV output:");
    ImGui::Checkbox("Save to file", &c.csv_save_enabled);
    InputTextStr("##lle_csv_path", c.csv_output_path);
    ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");

    ImGui::Separator();
    ImGui::Text("Initial conditions:");
    for (const auto& v : s.vars) {
        ImGui::PushID(v.c_str());
        InputNumStr(v.c_str(), c.initial_conditions[v], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Parameters:");
    for (const auto& p : s.params) {
        ImGui::PushID(p.c_str());
        InputNumStr(p.c_str(), c.param_values[p], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    bool do_run = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(160, 0));
        ImGui::EndDisabled();
    }
    else {
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    }

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            int total = (int)c.result_2d.flags.size();
            int diverged = 0;
            for (int f : c.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, lambda(min..max) = %.4g..%.4g",
                c.result_2d.n_pts, c.result_2d.n_pts,
                c.result_2d.min_val, c.result_2d.max_val);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##lle_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (c.last_run_ok) {
            int diverged = 0;
            for (int f : c.result.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, lambda-curve computed", c.result.n_pts);
            if (diverged) ImGui::TextDisabled("(%d/%d points diverged)", diverged, c.result.n_pts);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##lle_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return do_run;
}

// Tab bar по LLE-кривым + кнопка «+» (копирует последнюю).
static void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LLEAnalysisSession& s = model.lle_session;

    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##lle_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.curves.size(); ++i) {
            LLECurveConfig& c = s.curves[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = c.label + "###lle_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_curve_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                if (draw_lle_curve_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_curve();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_curve_index = active_now;
    if (to_remove >= 0) model.remove_lle_curve(to_remove);

    if (!s.in_flight && ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        run_idx = s.active_curve_index;
    }

    if (run_idx >= 0 && run_idx < (int)s.curves.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
}

// Plot LLE: линии (points_mode=false). Каждая кривая — λ(param).
// При mode_2d=true у активной кривой вместо линий рисуется HeatmapView
// (квадратная сетка λ(p1, p2) с colormap'ом).
static void draw_lle_plot(AppModel& model) {
    LLEAnalysisSession& s = model.lle_session;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::unique_ptr<Plot2DView> view;
    // Per-curve HeatmapView (see draw_bifurcation_plot for the rationale —
    // shared cache bleeds textures between curves with equal data_generation_2d).
    static std::map<unsigned, std::unique_ptr<HeatmapView>> heatmap_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    if (!view) {
        view = std::make_unique<Plot2DView>();
        view->points_mode = false;   // LLE — непрерывная линия
        view->show_legend = true;
        view->line_thickness_px = 1.5f;
        view->imdraw_lines = true;   // данные поверх осей + надёжная толщина
        view->pad_x = false;         // кривая впритык к боковым границам плота
        view->x_axis.name = "parameter";
        view->y_axis.name = "lambda";
    }
    auto get_lle_heatmap = [&](unsigned oid) -> HeatmapView& {
        auto& slot = heatmap_map[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
            int cm = model.heatmap_colormap;
            if (cm >= 0 && cm <= 3) slot->colormap = (HeatmapColormap)cm;
        }
        return *slot;
    };

    if (s.curves.empty()) {
        ImGui::TextDisabled("No curves yet.");
        return;
    }

    // Активная кривая решает, что рисовать. Если её mode_2d=true — heatmap;
    // иначе старый line-plot со всеми кривыми.
    int act = s.active_curve_index;
    if (act < 0 || act >= (int)s.curves.size()) act = 0;
    LLECurveConfig& cact = s.curves[act];

    if (cact.mode_2d) {
        // ------- HEATMAP -------
        // Per-curve heatmap by owner_id.
        const unsigned lle_oid = 0xBE110000u + (unsigned)act;
        HeatmapView& heatmap = get_lle_heatmap(lle_oid);

        // Combo для выбора colormap'а — над плотом.
        static const char* cmap_names[] = { "Viridis", "Inferno", "Turbo", "Gray" };
        int cmap_idx = (int)heatmap.colormap;
        ImGui::SetNextItemWidth(140);
        if (ImGui::Combo("Colormap", &cmap_idx, cmap_names, IM_ARRAYSIZE(cmap_names))) {
            heatmap.colormap = (HeatmapColormap)cmap_idx;
            model.heatmap_colormap = cmap_idx;
            // Персистим в _app_config.json — выбор сохранится между запусками.
            AppConfig cfg;
            cfg.ui_scale_override = model.ui_scale_override;
            cfg.use_builtin_font  = model.use_builtin_font;
            cfg.heatmap_colormap  = cmap_idx;
            cfg.basins_colormap        = model.basins_colormap;
            cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
            cfg.basins_avgint_colormap = model.basins_avgint_colormap;
            cfg.basins_states_colormap = model.basins_states_colormap;
            save_app_config(get_exe_dir_with_sep(), cfg);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Autoscale color", &heatmap.autoscale);
        if (!heatmap.autoscale) {
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("vmin", &heatmap.manual_vmin, 0.0f, 0.0f, "%.4g");
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("vmax", &heatmap.manual_vmax, 0.0f, 0.0f, "%.4g");
        }
        ImGui::SameLine();
        if (ImGui::Button(heatmap.swap_axes ? "Swap axes (on)##llehm" : "Swap axes##llehm"))
            heatmap.swap_axes = !heatmap.swap_axes;

        if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
            ImGui::TextDisabled("No 2D data yet. Press Run.");
            return;
        }

        // Подписи осей по реальным selected-полям свипа.
        auto axis_name_for = [&](bool sweep_var, int p_idx, int v_idx) -> std::string {
            if (sweep_var) {
                return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
            }
            return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
        };
        heatmap.x_axis.name = axis_name_for(cact.sweep_over_var,   cact.param_index,   cact.var_sweep_index);
        heatmap.y_axis.name = axis_name_for(cact.sweep_over_var_2, cact.param_index_2, cact.var_sweep_index_2);

        bool fit = cact.fit_request_2d;
        if (fit) cact.fit_request_2d = false;

        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        heatmap.render(*renderer, origin, avail,
                       /*owner_id*/ lle_oid, cact.data_generation_2d,
                       cact.result_2d.n_pts, cact.result_2d.n_pts,
                       cact.result_2d.values.data(),
                       cact.result_2d.param_lo,   cact.result_2d.param_hi,
                       cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                       cact.result_2d.min_val, cact.result_2d.max_val,
                       fit);
        return;
    }

    bool any_data = false;
    for (const auto& c : s.curves)
        if (c.last_run_ok && !c.result.lyapunov.empty()) { any_data = true; break; }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    auto safe_stod = [](const std::string& v, double def) -> double {
        if (v.empty()) return def;
        size_t slash = v.find('/');
        if (slash != std::string::npos) {
            double num = std::atof(v.substr(0, slash).c_str());
            double den = std::atof(v.substr(slash + 1).c_str());
            if (den != 0) return num / den;
        }
        return std::atof(v.c_str());
    };

    // Подпись X + X-fit диапазон. См. BD: ось охватывает union(lo, hi)
    // всех видимых кривых, не зависит от того, где есть/нет точек.
    int shared_kind = -2;   // -2 init, -1 mixed, 0 param, 1 var
    int shared_idx  = -2;
    double x_fit_lo = 0.0, x_fit_hi = 0.0;
    bool   x_fit_any = false;
    for (const auto& c : s.curves) {
        if (!c.visible || !c.last_run_ok) continue;
        int kind = c.sweep_over_var ? 1 : 0;
        int idx  = c.sweep_over_var ? c.var_sweep_index : c.param_index;
        if (shared_kind == -2) { shared_kind = kind; shared_idx = idx; }
        else if (shared_kind != kind || shared_idx != idx) {
            shared_kind = -1; shared_idx = -1;
        }
        double lo = c.result.param_lo, hi = c.result.param_hi;
        if (hi == lo) { lo = safe_stod(c.param_lo_text, 0.0); hi = safe_stod(c.param_hi_text, 1.0); }
        double a = std::min(lo, hi), b = std::max(lo, hi);
        if (!x_fit_any) { x_fit_lo = a; x_fit_hi = b; x_fit_any = true; }
        else { x_fit_lo = std::min(x_fit_lo, a); x_fit_hi = std::max(x_fit_hi, b); }
    }
    view->x_fit_use_explicit = x_fit_any;
    view->x_fit_min = x_fit_lo;
    view->x_fit_max = x_fit_hi;
    if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)s.params.size())
        view->x_axis.name = s.params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)s.vars.size())
        view->x_axis.name = s.vars[shared_idx] + " (IC)";
    else
        view->x_axis.name = "parameter";
    view->y_axis.name = "lambda";

    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != s.curves.size()) bufs.assign(s.curves.size(), {});

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(s.curves.size());
    init_vis.reserve(s.curves.size());
    glob_vis.reserve(s.curves.size());

    bool any_fit = false;
    int  data_gen = 0;

    for (int i = 0; i < (int)s.curves.size(); ++i) {
        LLECurveConfig& c = s.curves[i];
        auto& buf = bufs[i];
        buf.clear();
        int total_pts = 0;

        if (c.last_run_ok && !c.result.lyapunov.empty()) {
            // X считаются по тому диапазону, с которым реально шёл Run, а не
            // по текущим полям GUI — иначе кривая «прыгает» при редактировании
            // param_lo/hi до следующего Run.
            double lo = c.result.param_lo;
            double hi = c.result.param_hi;
            int npts = c.result.n_pts;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)c.result.flags.size() && c.result.flags[k] < 0) continue;
                double x = (npts > 1) ? (lo + (hi - lo) * (double)k / (double)(npts - 1)) : lo;
                double y = c.result.lyapunov[k];
                if (!std::isfinite(y)) continue;
                buf.push_back((float)x);
                buf.push_back((float)y);
                ++total_pts;
            }
        }

        PlotSeriesInput si;
        si.points   = buf.empty() ? nullptr : buf.data();
        si.n_points = total_pts;
        si.color    = ic_base_color(i);
        si.label    = c.label;
        series_in.push_back(si);
        init_vis.push_back(true);
        glob_vis.push_back(c.visible);

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view->render(*renderer, origin, avail, /*owner_id*/ 0xBE11E5, data_gen,
                 series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// LS: контролы (per-curve в табе) + line-plot λ_k(param), N экспонент
// на один спектр-«прогон». UX зеркало LLE.
// ============================================================

static bool draw_ls_curve_controls(LyapunovSpectrumAnalysisSession& s, int idx) {
    LSCurveConfig& c = s.curves[idx];

    ImGui::SetNextItemWidth(160);
    InputTextStr("Label", c.label);
    ImGui::SameLine();
    ImGui::Checkbox("visible", &c.visible);
    ImGui::Separator();

    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // Sweep target: параметры + разделитель + переменные (IC). См. BD.
    if (!s.params.empty() || !s.vars.empty()) {
        if (c.param_index < 0 || c.param_index >= (int)s.params.size())
            c.param_index = 0;
        if (c.var_sweep_index < 0 || c.var_sweep_index >= (int)s.vars.size())
            c.var_sweep_index = 0;
        std::string preview;
        if (c.sweep_over_var && !s.vars.empty())
            preview = s.vars[c.var_sweep_index] + " (IC)";
        else if (!s.params.empty())
            preview = s.params[c.param_index];
        else
            preview = "?";
        ImGui::SetNextItemWidth(160);
        if (ImGui::BeginCombo("Sweep", preview.c_str())) {
            for (int i = 0; i < (int)s.params.size(); ++i) {
                bool sel = !c.sweep_over_var && c.param_index == i;
                if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                    c.sweep_over_var = false;
                    c.param_index = i;
                }
            }
            if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
            for (int i = 0; i < (int)s.vars.size(); ++i) {
                std::string lbl = s.vars[i] + " (IC)";
                bool sel = c.sweep_over_var && c.var_sweep_index == i;
                if (ImGui::Selectable(lbl.c_str(), sel)) {
                    c.sweep_over_var = true;
                    c.var_sweep_index = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
    }
    InputNumStr("Param lo", c.param_lo_text, 120);
    InputNumStr("Param hi", c.param_hi_text, 120);
    InputNumStr("Resolution", c.n_pts_text, 120);

    ImGui::Separator();
    // 2D-режим. Сетка квадратная (см. LLE 2D — то же ограничение getValueByIdx).
    ImGui::Checkbox("2D mode (heatmap of one exponent)", &c.mode_2d);
    if (c.mode_2d) {
        ImGui::Indent();
        if (!s.params.empty() || !s.vars.empty()) {
            if (c.param_index_2 < 0 || c.param_index_2 >= (int)s.params.size())
                c.param_index_2 = 0;
            if (c.var_sweep_index_2 < 0 || c.var_sweep_index_2 >= (int)s.vars.size())
                c.var_sweep_index_2 = 0;
            std::string preview2;
            if (c.sweep_over_var_2 && !s.vars.empty())
                preview2 = s.vars[c.var_sweep_index_2] + " (IC)";
            else if (!s.params.empty())
                preview2 = s.params[c.param_index_2];
            else
                preview2 = "?";
            ImGui::SetNextItemWidth(160);
            if (ImGui::BeginCombo("Sweep Y", preview2.c_str())) {
                for (int i = 0; i < (int)s.params.size(); ++i) {
                    bool sel = !c.sweep_over_var_2 && c.param_index_2 == i;
                    if (ImGui::Selectable(s.params[i].c_str(), sel)) {
                        c.sweep_over_var_2 = false;
                        c.param_index_2 = i;
                    }
                }
                if (!s.params.empty() && !s.vars.empty()) ImGui::Separator();
                for (int i = 0; i < (int)s.vars.size(); ++i) {
                    std::string lbl = s.vars[i] + " (IC)";
                    bool sel = c.sweep_over_var_2 && c.var_sweep_index_2 == i;
                    if (ImGui::Selectable(lbl.c_str(), sel)) {
                        c.sweep_over_var_2 = true;
                        c.var_sweep_index_2 = i;
                    }
                }
                ImGui::EndCombo();
            }
        }
        InputNumStr("Param2 lo", c.param_lo_2_text, 120);
        InputNumStr("Param2 hi", c.param_hi_2_text, 120);
        ImGui::TextDisabled("Grid is square (Resolution applies to both axes).\nAll N exponents computed; switch in plot window.");
        ImGui::Unindent();
    }

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              c.h_text,         120);
    if (c.scheme == "CD")
        InputNumStr("symmetry s", c.symmetry_s,     120);
    InputNumStr("computing time", c.t_max_text,     120);
    InputNumStr("transient time", c.transient_text, 120);
    InputNumStr("max value",      c.max_value_text, 120);

    ImGui::Separator();
    ImGui::Text("LS (Wolf/Benettin + Gram-Schmidt):");
    InputNumStr("eps", c.eps_text, 120);
    InputNumStr("NT",  c.nt_text,  120);
    ImGui::TextDisabled("eps = initial perturbation magnitude; NT = block length\n"
                        "between renormalizations (in time units).");

    ImGui::Separator();
    ImGui::Text("CSV output:");
    ImGui::Checkbox("Save to file", &c.csv_save_enabled);
    InputTextStr("##ls_csv_path", c.csv_output_path);
    ImGui::TextDisabled("Path is kept even when save is off. Also writes <path>_config.csv.");

    ImGui::Separator();
    ImGui::Text("Initial conditions:");
    for (const auto& v : s.vars) {
        ImGui::PushID(v.c_str());
        InputNumStr(v.c_str(), c.initial_conditions[v], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Parameters:");
    for (const auto& p : s.params) {
        ImGui::PushID(p.c_str());
        InputNumStr(p.c_str(), c.param_values[p], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    bool do_run = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(160, 0));
        ImGui::EndDisabled();
    }
    else {
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    }

    if (c.mode_2d) {
        if (c.last_run_2d_ok) {
            int total = (int)c.result_2d.flags.size();
            int diverged = 0;
            for (int f : c.result_2d.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: %dx%d heatmap, %d exponents",
                c.result_2d.n_pts, c.result_2d.n_pts, c.result_2d.n_exponents);
            if (diverged) ImGui::TextDisabled("(%d/%d cells diverged)", diverged, total);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##ls_err_2d",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    } else {
        if (c.last_run_ok) {
            int diverged = 0;
            for (int f : c.result.flags) if (f < 0) ++diverged;
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: n_pts=%d, n_exponents=%d", c.result.n_pts, c.result.n_exponents);
            if (diverged) ImGui::TextDisabled("(%d/%d points diverged)", diverged, c.result.n_pts);
        }
        else if (!c.last_error.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
            ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
            ImGui::InputTextMultiline("##ls_err",
                const_cast<char*>(c.last_error.c_str()),
                c.last_error.size() + 1,
                sz,
                ImGuiInputTextFlags_ReadOnly);
        }
    }
    return do_run;
}

static void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;

    int active_now = -1;
    int run_idx = -1;
    int to_remove = -1;
    if (ImGui::BeginTabBar("##ls_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.curves.size(); ++i) {
            LSCurveConfig& c = s.curves[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = c.label + "###ls_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_curve_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                if (draw_ls_curve_controls(s, i)) run_idx = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_curve();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_curve_index = active_now;
    if (to_remove >= 0) model.remove_ls_curve(to_remove);

    if (!s.in_flight && ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        run_idx = s.active_curve_index;
    }

    if (run_idx >= 0 && run_idx < (int)s.curves.size()) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, run_idx);
    }
}

// Plot LS: каждая кривая раскладывается на N лiний (по числу экспонент).
// Серия: spectrum_idx * N + exponent_idx. Цвета через ic_base_color(seq).
// При mode_2d=true у активной кривой вместо линий рисуется HeatmapView с
// одной выбранной экспонентой; combo "Exponent" над хитмапой переключает
// плоскость без повторного Run.
static void draw_ls_plot(AppModel& model) {
    LyapunovSpectrumAnalysisSession& s = model.ls_session;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::unique_ptr<Plot2DView> view;
    // Per-curve HeatmapView (see draw_bifurcation_plot rationale).
    static std::map<unsigned, std::unique_ptr<HeatmapView>> heatmap_ls_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    if (!view) {
        view = std::make_unique<Plot2DView>();
        view->points_mode = false;
        view->show_legend = true;
        view->line_thickness_px = 1.5f;
        view->imdraw_lines = true;
        view->pad_x = false;
        view->x_axis.name = "parameter";
        view->y_axis.name = "lambda";
    }
    auto get_ls_heatmap = [&](unsigned oid) -> HeatmapView& {
        auto& slot = heatmap_ls_map[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
            int cm = model.heatmap_colormap;
            if (cm >= 0 && cm <= 3) slot->colormap = (HeatmapColormap)cm;
        }
        return *slot;
    };

    if (s.curves.empty()) {
        ImGui::TextDisabled("No spectra yet.");
        return;
    }

    // Активная кривая решает, что рисовать. mode_2d → heatmap одной экспоненты.
    int act = s.active_curve_index;
    if (act < 0 || act >= (int)s.curves.size()) act = 0;
    LSCurveConfig& cact = s.curves[act];

    if (cact.mode_2d) {
        // Per-curve heatmap by owner_id.
        const unsigned ls_oid = 0x15A20000u + (unsigned)act;
        HeatmapView& heatmap_ls = get_ls_heatmap(ls_oid);

        // Combo colormap (persisted в _app_config.json, как у LLE/BD).
        static const char* cmap_names[] = { "Viridis", "Inferno", "Turbo", "Gray" };
        int cmap_idx = (int)heatmap_ls.colormap;
        ImGui::SetNextItemWidth(140);
        if (ImGui::Combo("Colormap##lshm", &cmap_idx, cmap_names, IM_ARRAYSIZE(cmap_names))) {
            heatmap_ls.colormap = (HeatmapColormap)cmap_idx;
            model.heatmap_colormap = cmap_idx;
            AppConfig cfg;
            cfg.ui_scale_override = model.ui_scale_override;
            cfg.use_builtin_font  = model.use_builtin_font;
            cfg.heatmap_colormap  = cmap_idx;
            cfg.basins_colormap        = model.basins_colormap;
            cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
            cfg.basins_avgint_colormap = model.basins_avgint_colormap;
            cfg.basins_states_colormap = model.basins_states_colormap;
            cfg.tick_precision         = model.tick_precision;
            save_app_config(get_exe_dir_with_sep(), cfg);
        }

        // Combo exponent: λ₁..λ_N. Active при наличии данных; clamp idx под N.
        if (cact.last_run_2d_ok && cact.result_2d.n_exponents > 0) {
            int N = cact.result_2d.n_exponents;
            if (cact.display_exponent_idx < 0 || cact.display_exponent_idx >= N)
                cact.display_exponent_idx = 0;
            std::vector<std::string> names;
            names.reserve(N);
            for (int j = 0; j < N; ++j) names.push_back("L" + std::to_string(j + 1));
            std::vector<const char*> cnames;
            cnames.reserve(N);
            for (auto& s2 : names) cnames.push_back(s2.c_str());
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::Combo("Exponent##lshm", &cact.display_exponent_idx, cnames.data(), N);
        }

        ImGui::SameLine();
        ImGui::Checkbox("Autoscale color##lshm", &heatmap_ls.autoscale);
        if (!heatmap_ls.autoscale) {
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("vmin##lshm", &heatmap_ls.manual_vmin, 0.0f, 0.0f, "%.4g");
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            ImGui::InputFloat("vmax##lshm", &heatmap_ls.manual_vmax, 0.0f, 0.0f, "%.4g");
        }
        ImGui::SameLine();
        if (ImGui::Button(heatmap_ls.swap_axes ? "Swap axes (on)##lshm" : "Swap axes##lshm"))
            heatmap_ls.swap_axes = !heatmap_ls.swap_axes;

        if (!cact.last_run_2d_ok || cact.result_2d.values.empty()) {
            ImGui::TextDisabled("No 2D data yet. Press Run.");
            return;
        }

        // Подписи осей по реальным selected-полям свипа.
        auto ax_name = [&](bool sweep_var, int p_idx, int v_idx) -> std::string {
            if (sweep_var)
                return (v_idx >= 0 && v_idx < (int)s.vars.size()) ? (s.vars[v_idx] + " (IC)") : "x";
            return (p_idx >= 0 && p_idx < (int)s.params.size()) ? s.params[p_idx] : "param";
        };
        heatmap_ls.x_axis.name = ax_name(cact.sweep_over_var,   cact.param_index,   cact.var_sweep_index);
        heatmap_ls.y_axis.name = ax_name(cact.sweep_over_var_2, cact.param_index_2, cact.var_sweep_index_2);

        bool fit = cact.fit_request_2d;
        if (fit) cact.fit_request_2d = false;

        // Указатель на нужную плоскость + её per-plane min/max.
        int k = cact.display_exponent_idx;
        size_t plane_size = (size_t)cact.result_2d.n_pts * (size_t)cact.result_2d.n_pts;
        const double* plane_ptr = cact.result_2d.values.data() + (size_t)k * plane_size;
        double vmin = (k >= 0 && k < (int)cact.result_2d.min_val.size()) ? cact.result_2d.min_val[k] : 0.0;
        double vmax = (k >= 0 && k < (int)cact.result_2d.max_val.size()) ? cact.result_2d.max_val[k] : 0.0;

        // data_generation для VBO-кэша: смешиваем поколение чанка + индекс
        // экспоненты, чтобы переключение перезалило текстуру.
        int gen = cact.data_generation_2d * 64 + k;

        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        heatmap_ls.render(*renderer, origin, avail,
                          /*owner_id*/ ls_oid, gen,
                          cact.result_2d.n_pts, cact.result_2d.n_pts,
                          plane_ptr,
                          cact.result_2d.param_lo,   cact.result_2d.param_hi,
                          cact.result_2d.param_lo_2, cact.result_2d.param_hi_2,
                          vmin, vmax,
                          fit);
        return;
    }

    bool any_data = false;
    for (const auto& c : s.curves)
        if (c.last_run_ok && !c.result.spectrum.empty()) { any_data = true; break; }
    if (!any_data) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    auto safe_stod = [](const std::string& v, double def) -> double {
        if (v.empty()) return def;
        size_t slash = v.find('/');
        if (slash != std::string::npos) {
            double num = std::atof(v.substr(0, slash).c_str());
            double den = std::atof(v.substr(slash + 1).c_str());
            if (den != 0) return num / den;
        }
        return std::atof(v.c_str());
    };

    // Подпись X + X-fit диапазон. См. BD/LLE: ось охватывает union(lo, hi)
    // всех видимых спектров, не зависит от наличия точек.
    int shared_kind = -2;   // -2 init, -1 mixed, 0 param, 1 var
    int shared_idx  = -2;
    double x_fit_lo = 0.0, x_fit_hi = 0.0;
    bool   x_fit_any = false;
    for (const auto& c : s.curves) {
        if (!c.visible || !c.last_run_ok) continue;
        int kind = c.sweep_over_var ? 1 : 0;
        int idx  = c.sweep_over_var ? c.var_sweep_index : c.param_index;
        if (shared_kind == -2) { shared_kind = kind; shared_idx = idx; }
        else if (shared_kind != kind || shared_idx != idx) {
            shared_kind = -1; shared_idx = -1;
        }
        double lo = c.result.param_lo, hi = c.result.param_hi;
        if (hi == lo) { lo = safe_stod(c.param_lo_text, 0.0); hi = safe_stod(c.param_hi_text, 1.0); }
        double a = std::min(lo, hi), b = std::max(lo, hi);
        if (!x_fit_any) { x_fit_lo = a; x_fit_hi = b; x_fit_any = true; }
        else { x_fit_lo = std::min(x_fit_lo, a); x_fit_hi = std::max(x_fit_hi, b); }
    }
    view->x_fit_use_explicit = x_fit_any;
    view->x_fit_min = x_fit_lo;
    view->x_fit_max = x_fit_hi;
    if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)s.params.size())
        view->x_axis.name = s.params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)s.vars.size())
        view->x_axis.name = s.vars[shared_idx] + " (IC)";
    else
        view->x_axis.name = "parameter";

    // Подсчитываем общее число серий — sum(n_exponents per curve).
    size_t total_series = 0;
    for (const auto& c : s.curves) {
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();
        total_series += (size_t)N;
    }

    static std::vector<std::vector<float>> bufs;
    if (bufs.size() != total_series) bufs.assign(total_series, {});

    std::vector<PlotSeriesInput> series_in;
    std::vector<bool> init_vis, glob_vis;
    series_in.reserve(total_series);
    init_vis.reserve(total_series);
    glob_vis.reserve(total_series);

    bool any_fit = false;
    int  data_gen = 0;
    size_t buf_cursor = 0;
    int    series_idx = 0;

    for (int i = 0; i < (int)s.curves.size(); ++i) {
        LSCurveConfig& c = s.curves[i];
        int N = c.result.n_exponents > 0 ? c.result.n_exponents : (int)s.vars.size();

        // X — по диапазону, с которым шёл Run (см. LLE-плот).
        double lo = c.result.param_lo;
        double hi = c.result.param_hi;
        int npts = c.result.n_pts;
        bool have = c.last_run_ok && !c.result.spectrum.empty();

        for (int j = 0; j < N; ++j) {
            auto& buf = bufs[buf_cursor++];
            buf.clear();
            int total_pts = 0;

            if (have) {
                for (int k = 0; k < npts; ++k) {
                    if (k < (int)c.result.flags.size() && c.result.flags[k] < 0) continue;
                    if (k >= (int)c.result.spectrum.size()) continue;
                    const auto& row = c.result.spectrum[k];
                    if (j >= (int)row.size()) continue;
                    double x = (npts > 1) ? (lo + (hi - lo) * (double)k / (double)(npts - 1)) : lo;
                    double y = row[j];
                    if (!std::isfinite(y)) continue;
                    buf.push_back((float)x);
                    buf.push_back((float)y);
                    ++total_pts;
                }
            }

            PlotSeriesInput si;
            si.points   = buf.empty() ? nullptr : buf.data();
            si.n_points = total_pts;
            si.color    = ic_base_color(series_idx++);
            // label: "<spectrum> λj" если N>1, иначе просто <spectrum>.
            std::string lab = (N > 1)
                ? (c.label + " " + std::string("\xCE\xBB") /* UTF-8 λ — но шрифт не покажет, fallback */)
                : c.label;
            // Для совместимости со шрифтом без glyph'а — английская подпись.
            lab = (N > 1) ? (c.label + " L" + std::to_string(j + 1)) : c.label;
            si.label    = lab;
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(c.visible);
        }

        data_gen = data_gen * 31 + c.data_generation;
        if (c.fit_request) { any_fit = true; c.fit_request = false; }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();
    view->render(*renderer, origin, avail, /*owner_id*/ 0x15A1E0, data_gen,
                 series_in, init_vis, glob_vis, any_fit);
}

// ============================================================
// Basins of attraction: controls + 5-plot window (inner tab-bar).
// ============================================================

static void draw_basins_controls(AppModel& model, SystemLibrary& lib) {
    BasinsAnalysisSession& s = model.basins_session;

    ImGui::Text("Basins of attraction");
    ImGui::TextDisabled("DBSCAN clustering in (avgPeak, avgInterval) plane.");

    // ----- System picker (как у Parametric) -----
    // Смена системы во время async-расчёта запрещена.
    ImGui::Text("System:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string current = model.name.empty() ? "(current)" : model.name;
    if (s.in_flight) ImGui::BeginDisabled();
    if (ImGui::BeginCombo("##basins_syssel", current.c_str())) {
        for (const auto& nm : lib.list()) {
            if (ImGui::Selectable(nm.c_str(), model.name == nm)) {
                try {
                    model.from_record(lib.load(nm));
                    model.start_basins_analysis();
                    std::string jb = lib.load_session(model.loaded_name, "_last_basins");
                    if (!jb.empty())
                        session_from_json_basins(jb, model.basins_session);
                }
                catch (...) {}
            }
        }
        ImGui::EndCombo();
    }
    if (s.in_flight) ImGui::EndDisabled();
    ImGui::Separator();

    // Tab bar: одна вкладка на Basins-config + "+" для add. Зеркалит
    // draw_bifurcation_controls. Активная вкладка хранится в
    // s.active_config_index и используется Ctrl+R + плотами.
    int active_now = -1;
    int to_remove = -1;
    bool run_pressed_in_tab = false;
    if (ImGui::BeginTabBar("##basins_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            BasinsConfig& bc = s.configs[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = bc.label + "###basins_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_config_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+",
                                     ImGuiTabItemFlags_Trailing |
                                     ImGuiTabItemFlags_NoTooltip)) {
                s.add_config();
            }
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_config_index = active_now;
    if (to_remove >= 0) model.remove_basins_config(to_remove);
    (void)run_pressed_in_tab;

    if (s.configs.empty()) {
        ImGui::TextDisabled("No basins configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];

    // Inline rename для активной вкладки.
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s", c.label.c_str());
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputText("Label##basins_label", buf, sizeof(buf)))
            c.label = buf;
    }
    ImGui::Separator();

    // ----- Scheme -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    ImGui::Separator();

    // ----- Axes (X, Y по двум IC-переменным) -----
    if (!s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());

        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis X (IC)", &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, 120);
        InputNumStr("X hi", c.axis_x_hi_text, 120);

        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis Y (IC)", &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, 120);
        InputNumStr("Y hi", c.axis_y_hi_text, 120);
    } else {
        ImGui::TextDisabled("No variables (load a system first)");
    }
    InputNumStr("Resolution", c.n_pts_text, 120);

    // ----- Writable var (для peak finder) -----
    if (!s.vars.empty()) {
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());
        if (c.writable_var < 0 || c.writable_var >= (int)s.vars.size()) c.writable_var = 0;
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Writable var", &c.writable_var, items.data(), (int)items.size());
    }

    // ----- Feature selection -----
    // 12 фич (см. BF_* в configCUDA.h / enum BasinFeature в analysis_session.h).
    // Feature 1 пишется в outAvgPeaks-буфер (X-координата DBSCAN), Feature 2 —
    // в AvgTimeOfPeaks-буфер (Y-координата). Множители применяются ПОСЛЕ
    // вычисления фичи и нужны для масштабирования кластеризации.
    ImGui::Separator();
    ImGui::Text("Features (DBSCAN axes + plot data):");
    static const char* feat_names[] = {
        "Avg peaks",             "Avg intervals",
        "RMS peaks",             "RMS intervals",
        "StDev peaks",           "StDev intervals",
        "sign\xc2\xb7log10|avg peaks|", "sign\xc2\xb7log10|avg intervals|",
        "log10 RMS peaks",       "log10 RMS intervals",
        "log10 StDev peaks",     "log10 StDev intervals",
    };
    if (c.feature1 < 0 || c.feature1 >= BF_FEATURE_COUNT) c.feature1 = BF_FEATURE1_DEFAULT;
    if (c.feature2 < 0 || c.feature2 >= BF_FEATURE_COUNT) c.feature2 = BF_FEATURE2_DEFAULT;
    ImGui::SetNextItemWidth(220);
    ImGui::Combo("Feature 1##bas", &c.feature1, feat_names, IM_ARRAYSIZE(feat_names));
    ImGui::SameLine();
    InputNumStr("mult##bas_f1", c.mult_feature1_text, 80);
    ImGui::SetNextItemWidth(220);
    ImGui::Combo("Feature 2##bas", &c.feature2, feat_names, IM_ARRAYSIZE(feat_names));
    ImGui::SameLine();
    InputNumStr("mult##bas_f2", c.mult_feature2_text, 80);

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              c.h_text,           120);
    if (c.scheme == "CD")
        InputNumStr("symmetry s", c.symmetry_s,       120);
    InputNumStr("computing time", c.t_max_text,       120);
    InputNumStr("transient time", c.transient_text,   120);
    InputNumStr("decimator",      c.pre_scaller_text, 120);
    InputNumStr("max value",      c.max_value_text,   120);

    ImGui::Separator();
    InputNumStr("DBSCAN eps", c.eps_dbscan_text, 120);
    ImGui::TextDisabled("Clustering radius in (Feature 1, Feature 2) space.");

    ImGui::Separator();
    ImGui::Text("CSV output:");
    ImGui::Checkbox("Save to file", &c.csv_save_enabled);
    InputTextStr("##basins_csv_path", c.csv_output_path);
    ImGui::TextDisabled("Writes 4 files: <path>, _1.csv (Feature 1), _2.csv (Feature 2), _3.csv (states).");

    ImGui::Separator();
    ImGui::Text("Initial conditions (for non-axis variables):");
    for (const auto& v : s.vars) {
        ImGui::PushID(v.c_str());
        InputNumStr(v.c_str(), c.initial_conditions[v], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Parameters:");
    for (const auto& p : s.params) {
        ImGui::PushID(p.c_str());
        InputNumStr(p.c_str(), c.param_values[p], 120);
        ImGui::PopID();
    }

    ImGui::Separator();
    bool do_run = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(160, 0));
        ImGui::EndDisabled();
    }
    else {
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    }
    if (!s.in_flight && ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        do_run = true;
    }
    if (do_run) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, s.active_config_index);
    }

    // Batch "Run all..." across basin configs. Pushes selected indices into
    // model.basins_queue; draw_gui ticks the queue after polls.
    ImGui::SameLine();
    if (s.in_flight) ImGui::BeginDisabled();
    if (ImGui::Button("Run all..."))
        ImGui::OpenPopup("##run_all_basins");
    if (s.in_flight) ImGui::EndDisabled();
    if (!model.basins_queue.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", model.basins_queue.size());
    }
    if (ImGui::BeginPopup("##run_all_basins")) {
        static std::vector<bool> picks;
        if (picks.size() != s.configs.size()) picks.assign(s.configs.size(), true);

        ImGui::TextDisabled("Sequential (one CUDA context).");
        for (size_t i = 0; i < picks.size(); ++i) {
            bool v = picks[i];
            std::string lbl = s.configs[i].label + "###pbasins_" + std::to_string(i);
            if (ImGui::Checkbox(lbl.c_str(), &v)) picks[i] = v;
        }

        ImGui::Separator();
        if (ImGui::Button("All"))  { for (auto&& b : picks) b = true;  }
        ImGui::SameLine();
        if (ImGui::Button("None")) { for (auto&& b : picks) b = false; }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            for (size_t i = 0; i < picks.size(); ++i)
                if (picks[i])
                    model.basins_queue.push_back({(int)i});
            model.start_next_in_basins_queue();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (c.last_run_ok) {
        int total = (int)c.result.basin_idx.size();
        int diverged = 0;
        for (int f : c.result.helpful_array) if (f == 0) ++diverged;
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: %dx%d, %d clusters (+ %d FP clusters); %d cells unbound",
            c.result.n_pts, c.result.n_pts,
            c.result.n_clusters, -c.result.min_cluster_idx, diverged);
    } else if (!c.last_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
        ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 12);
        ImGui::InputTextMultiline("##basins_err",
            const_cast<char*>(c.last_error.c_str()),
            c.last_error.size() + 1,
            sz,
            ImGuiInputTextFlags_ReadOnly);
    }
}

// Plot Basins: inner tab-bar по 5 представлениям. Heatmap-views — per
// (config × tab) в std::map по owner_id. HeatmapView/Plot2DView хранят
// data_gen_cached внутри без учёта owner_id, поэтому переиспользовать один
// view между разными configs нельзя (после Run All у двух configs одинаковый
// data_generation=1 → cache не invalidate'тся и на чужой вкладке показывается
// предыдущий buffer). Map с lazy-init решает это и сохраняет независимый
// zoom/pan per (config, tab).
static void draw_basins_plot(AppModel& model) {
    BasinsAnalysisSession& s = model.basins_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    BasinsConfig& c = s.configs[s.active_config_index];
    // Owner IDs зависят от индекса config — каждый basin имеет независимый
    // zoom/pan per inner tab. Схема: 0x1BA50000 + cfg*5 + tab (max 50 configs).
    const unsigned base_oid = 0x1BA50000u + (unsigned)s.active_config_index * 5u;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_basins, hm_avgpk, hm_avgint, hm_states;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  scatter_views;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    auto get_hm = [](std::map<unsigned, std::unique_ptr<HeatmapView>>& m, unsigned oid,
                     bool discrete_default) -> HeatmapView& {
        auto& slot = m[oid];
        if (!slot) {
            slot = std::make_unique<HeatmapView>();
            if (discrete_default) slot->discrete = true;
        }
        return *slot;
    };
    auto get_scatter = [](unsigned oid) -> Plot2DView& {
        auto& slot = scatter_views[oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode = true;
            slot->show_legend = false;
            slot->point_size_px = 3.0f;
            slot->pad_x = false;
        }
        return *slot;
    };
    HeatmapView& hm_basins_v = get_hm(hm_basins, base_oid + 0u, /*discrete*/ true);
    HeatmapView& hm_avgpk_v  = get_hm(hm_avgpk,  base_oid + 1u, false);
    HeatmapView& hm_avgint_v = get_hm(hm_avgint, base_oid + 2u, false);
    HeatmapView& hm_states_v = get_hm(hm_states, base_oid + 3u, false);
    Plot2DView&  scatter_v   = get_scatter(base_oid + 4u);

    if (!c.last_run_ok || c.result.basin_idx.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    // Inner tab-bar — переключение по 5 видам.
    // Имена 2-го и 3-го табов нейтральные — они показывают выбранную фичу
    // (Feature 1/2 из BasinsConfig), которая может быть не "avg peaks/interval".
    const char* tab_names[5] = { "Basins", "Feature 1", "Feature 2", "States", "Scatter" };
    if (ImGui::BeginTabBar("##basins_inner")) {
        for (int t = 0; t < 5; ++t) {
            if (ImGui::BeginTabItem(tab_names[t])) {
                c.active_plot_tab = t;
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    // Combo выбора colormap — свой для каждого heatmap-таба (Basins/AvgPk/
    // AvgInt/States). Каждый выбор пишется в свой field в AppModel и
    // персистится в _app_config.json. Все эти поля независимы от
    // model.heatmap_colormap (тот шарится Bif/LLE/LS) — смена здесь их не
    // затрагивает, и наоборот. Scatter-таб использует Plot2DView, без
    // colormap.
    {
        static const char* cmap_names[] = { "Viridis", "Inferno", "Turbo", "Gray" };
        int* field = nullptr;
        const char* combo_id = nullptr;
        HeatmapView* active_hm = nullptr;
        switch (c.active_plot_tab) {
            case 0: field = &model.basins_colormap;        combo_id = "Colormap##bas";     active_hm = &hm_basins_v; break;
            case 1: field = &model.basins_avgpk_colormap;  combo_id = "Colormap##bas_pk";  active_hm = &hm_avgpk_v;  break;
            case 2: field = &model.basins_avgint_colormap; combo_id = "Colormap##bas_int"; active_hm = &hm_avgint_v; break;
            case 3: field = &model.basins_states_colormap; combo_id = "Colormap##bas_st";  active_hm = &hm_states_v; break;
            default: break;  // Scatter (4) — без combo
        }
        if (field) {
            int cmap_idx = *field;
            if (cmap_idx < 0 || cmap_idx > 3) cmap_idx = 2;
            ImGui::SetNextItemWidth(140);
            if (ImGui::Combo(combo_id, &cmap_idx, cmap_names, IM_ARRAYSIZE(cmap_names))) {
                *field = cmap_idx;
                AppConfig cfg;
                cfg.ui_scale_override      = model.ui_scale_override;
                cfg.use_builtin_font       = model.use_builtin_font;
                cfg.heatmap_colormap       = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            // Swap axes — per-tab. Сессионный toggle (не персистится),
            // повторный клик возвращает исходную ориентацию.
            if (active_hm) {
                ImGui::SameLine();
                if (ImGui::Button(active_hm->swap_axes ? "Swap axes (on)##bas_sw" : "Swap axes##bas_sw"))
                    active_hm->swap_axes = !active_hm->swap_axes;
            }
        }
    }

    int n = c.result.n_pts;
    size_t total = (size_t)n * (size_t)n;
    double xlo = c.result.axis_x_lo, xhi = c.result.axis_x_hi;
    double ylo = c.result.axis_y_lo, yhi = c.result.axis_y_hi;

    auto axis_name = [&](int var_idx) -> std::string {
        if (var_idx >= 0 && var_idx < (int)s.vars.size())
            return s.vars[var_idx] + "(0)";
        return std::string("x");
    };
    std::string ax_x = axis_name(c.result.axis_x_var);
    std::string ax_y = axis_name(c.result.axis_y_var);

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    if (c.active_plot_tab == 0) {
        // Basins idx — turbo-discrete через GL_NEAREST. Spectrum [min..max].
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) buf[k] = (double)c.result.basin_idx[k];
        // Cluster IDs in basin_idx:
        //   min_cluster_idx..-1 — FP-clusters (always present when negative)
        //   0                   — diverged/unbound cells (helpful_array[i] == 0)
        //   1..n_clusters       — oscillatory clusters
        // When no FP-clusters exist (min_cluster_idx == 0) and no cell
        // diverged, "cluster 0" isn't a real ID; shift vmin to 1 so the
        // colorbar doesn't show a phantom band. If diverged cells exist,
        // keep vmin = 0 so they get their own color band.
        bool has_diverged = false;
        for (int f : c.result.helpful_array)
            if (f == 0) { has_diverged = true; break; }
        double vmin;
        if (c.result.min_cluster_idx < 0)        vmin = (double)c.result.min_cluster_idx;
        else if (has_diverged)                   vmin = 0.0;
        else                                     vmin = 1.0;
        double vmax = (double)c.result.n_clusters;
        if (vmax < vmin) vmax = vmin;
        {
            int cm = model.basins_colormap;
            hm_basins_v.colormap = (HeatmapColormap)((cm >= 0 && cm <= 3) ? cm : 2);
        }
        hm_basins_v.x_axis.name = ax_x;
        hm_basins_v.y_axis.name = ax_y;
        hm_basins_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 0u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          vmin, vmax, fit);
    }
    else if (c.active_plot_tab == 1) {
        {
            int cm = model.basins_avgpk_colormap;
            hm_avgpk_v.colormap = (HeatmapColormap)((cm >= 0 && cm <= 3) ? cm : 2);
        }
        hm_avgpk_v.x_axis.name = ax_x;
        hm_avgpk_v.y_axis.name = ax_y;
        hm_avgpk_v.render(*renderer, origin, avail,
                         /*owner_id*/ base_oid + 1u, c.data_generation,
                         n, n, c.result.avg_peaks.data(),
                         xlo, xhi, ylo, yhi,
                         c.result.avg_peaks_min, c.result.avg_peaks_max, fit);
    }
    else if (c.active_plot_tab == 2) {
        {
            int cm = model.basins_avgint_colormap;
            hm_avgint_v.colormap = (HeatmapColormap)((cm >= 0 && cm <= 3) ? cm : 2);
        }
        hm_avgint_v.x_axis.name = ax_x;
        hm_avgint_v.y_axis.name = ax_y;
        hm_avgint_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 2u, c.data_generation,
                          n, n, c.result.avg_intervals.data(),
                          xlo, xhi, ylo, yhi,
                          c.result.avg_intervals_min, c.result.avg_intervals_max, fit);
    }
    else if (c.active_plot_tab == 3) {
        // States: helpful_array → 3 категории. Маппим в дискретные значения
        // и рендерим через HeatmapView в Turbo (3 равноотстоящие точки):
        //   1 (Osc)     → 0   (синий конец turbo)
        //   -1 (FP)     → 1   (зелёный середина)
        //   0 (Unbound) → 2   (красный конец)
        // Это не точные MATLAB-цвета, но 3 различимые категории.
        static std::vector<double> buf;
        buf.resize(total);
        for (size_t k = 0; k < total; ++k) {
            int v = c.result.helpful_array[k];
            buf[k] = (v == 1) ? 0.0 : (v == -1 ? 1.0 : 2.0);
        }
        {
            int cm = model.basins_states_colormap;
            hm_states_v.colormap = (HeatmapColormap)((cm >= 0 && cm <= 3) ? cm : 2);
        }
        hm_states_v.x_axis.name = ax_x;
        hm_states_v.y_axis.name = ax_y;
        hm_states_v.render(*renderer, origin, avail,
                          /*owner_id*/ base_oid + 3u, c.data_generation,
                          n, n, buf.data(),
                          xlo, xhi, ylo, yhi,
                          0.0, 2.0, fit);
        // Подсказка под плотом — числовые уровни, цвет зависит от выбранной colormap.
        ImGui::TextDisabled("Levels: 0=Osc, 1=FixedPoint, 2=Unbound");
    }
    else if (c.active_plot_tab == 4) {
        // Scatter (avgPeak, avgInterval), точки сгруппированы по basin_idx.
        // Каждый кластер — своя серия (PlotSeriesInput с собственным цветом).
        int min_id = c.result.min_cluster_idx;
        int max_id = c.result.n_clusters;
        int n_total_clusters = max_id - min_id + 1;
        if (n_total_clusters < 1) n_total_clusters = 1;

        // Сгруппируем точки по basin_idx.
        std::map<int, std::vector<float>> bufs;
        int valid_pts = 0;
        for (size_t k = 0; k < total; ++k) {
            int id = c.result.basin_idx[k];
            double xp = c.result.avg_peaks[k];
            double yp = c.result.avg_intervals[k];
            if (!std::isfinite(xp) || !std::isfinite(yp)) continue;
            if (xp == 999.0 || xp == -999.0 || yp == 999.0 || yp == -999.0) continue;
            bufs[id].push_back((float)xp);
            bufs[id].push_back((float)yp);
            ++valid_pts;
        }
        if (valid_pts == 0) {
            ImGui::TextDisabled("No valid (avgPeak, avgInterval) points.");
            return;
        }

        // Static-буферы должны жить весь кадр (Plot2DView хранит сырые указатели).
        static std::vector<std::vector<float>> series_buffers;
        static std::vector<std::string>        series_labels;
        series_buffers.clear();
        series_labels.clear();
        series_buffers.reserve(bufs.size());
        series_labels.reserve(bufs.size());

        std::vector<PlotSeriesInput> series_in;
        std::vector<bool> init_vis, glob_vis;
        for (auto& kv : bufs) {
            int id = kv.first;
            series_buffers.push_back(std::move(kv.second));
            series_labels.push_back("c" + std::to_string(id));
            // Цвет per-cluster через ic_base_color (golden-ratio hash). Сдвиг
            // на (id - min_id) — чтобы FP-кластеры (отрицательные) и Osc
            // (положительные) тоже разделялись.
            PlotSeriesInput si;
            si.points   = series_buffers.back().empty() ? nullptr : series_buffers.back().data();
            si.n_points = (int)(series_buffers.back().size() / 2);
            si.color    = ic_base_color(id - min_id);
            si.label    = series_labels.back();
            series_in.push_back(si);
            init_vis.push_back(true);
            glob_vis.push_back(true);
        }
        (void)n_total_clusters;
        // Имена осей scatter'а — выбранные фичи. Должны быть синхронизированы
        // с feat_names в draw_basins_controls (тот же порядок BF_*).
        static const char* feat_names_plot[] = {
            "Avg peaks",             "Avg intervals",
            "RMS peaks",             "RMS intervals",
            "StDev peaks",           "StDev intervals",
            "sign\xc2\xb7log10|avg peaks|", "sign\xc2\xb7log10|avg intervals|",
            "log10 RMS peaks",       "log10 RMS intervals",
            "log10 StDev peaks",     "log10 StDev intervals",
        };
        int f1 = (c.feature1 >= 0 && c.feature1 < BF_FEATURE_COUNT) ? c.feature1 : BF_FEATURE1_DEFAULT;
        int f2 = (c.feature2 >= 0 && c.feature2 < BF_FEATURE_COUNT) ? c.feature2 : BF_FEATURE2_DEFAULT;
        scatter_v.x_axis.name = feat_names_plot[f1];
        scatter_v.y_axis.name = feat_names_plot[f2];
        scatter_v.render(*renderer, origin, avail,
                             /*owner_id*/ base_oid + 4u, c.data_generation,
                             series_in, init_vis, glob_vis, fit);
    }
}

// ============================================================
// Fast Synchro Controls + Plot — recurrent synchronization (anti-sync).
// Mode 0 = On Attractor (trajectory + per-point error); Mode 1 = On Grid.
// ============================================================
static void draw_fastsync_controls(AppModel& model, SystemLibrary& lib) {
    FastSyncAnalysisSession& s = model.fastsync_session;

    ImGui::Text("Fast Synchro");
    ImGui::TextDisabled("Recurrent synchronization analysis (anti-sync error).");

    // System picker.
    ImGui::Text("System:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string current = model.name.empty() ? "(current)" : model.name;
    if (s.in_flight) ImGui::BeginDisabled();
    if (ImGui::BeginCombo("##fs_syssel", current.c_str())) {
        for (const auto& nm : lib.list()) {
            if (ImGui::Selectable(nm.c_str(), model.name == nm)) {
                try {
                    model.from_record(lib.load(nm));
                    model.start_fastsync_analysis();
                    std::string jb = lib.load_session(model.loaded_name, "_last_fastsync");
                    if (!jb.empty())
                        session_from_json_fastsync(jb, model.fastsync_session);
                } catch (...) {}
            }
        }
        ImGui::EndCombo();
    }
    if (s.in_flight) ImGui::EndDisabled();
    ImGui::Separator();

    // Tab bar для config'ов.
    int active_now = -1, to_remove = -1;
    if (ImGui::BeginTabBar("##fs_tabs",
                           ImGuiTabBarFlags_Reorderable |
                           ImGuiTabBarFlags_AutoSelectNewTabs |
                           ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < (int)s.configs.size(); ++i) {
            FastSyncConfig& fc = s.configs[i];
            ImGui::PushID(i);
            bool open = true;
            std::string tab_id = fc.label + "###fs_tab_" + std::to_string(i);
            bool can_close = !(s.in_flight && s.running_config_index == i);
            if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr)) {
                active_now = i;
                ImGui::EndTabItem();
            }
            if (!open) to_remove = i;
            ImGui::PopID();
        }
        if (!s.in_flight) {
            if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip))
                s.add_config();
        }
        ImGui::EndTabBar();
    }
    if (active_now >= 0) s.active_config_index = active_now;
    if (to_remove >= 0)  model.remove_fastsync_config(to_remove);

    if (s.configs.empty()) {
        ImGui::TextDisabled("No FastSync configs. Press '+' to add one.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    // Label rename.
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s", c.label.c_str());
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputText("Label##fs_label", buf, sizeof(buf))) c.label = buf;
    }
    ImGui::Separator();

    // ----- Mode -----
    ImGui::Text("Mode:");
    ImGui::RadioButton("On Attractor", &c.mode, 0); ImGui::SameLine();
    ImGui::RadioButton("On Grid",      &c.mode, 1);
    ImGui::Separator();

    // ----- Scheme -----
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4", "DOPRI78", "CD" };
    ImGui::SetNextItemWidth(160);
    if (ImGui::BeginCombo("Scheme", c.scheme.c_str())) {
        for (auto m : schemes)
            if (ImGui::Selectable(m, c.scheme == m)) c.scheme = m;
        if (!s.custom_schemes.empty()) ImGui::Separator();
        for (const auto& cs : s.custom_schemes)
            if (ImGui::Selectable((cs.name + " (custom)").c_str(), c.scheme == cs.name))
                c.scheme = cs.name;
        ImGui::EndCombo();
    }
    if (c.scheme == "CD") InputNumStr("symmetry s", c.symmetry_s, 120);
    ImGui::Separator();

    // ----- Mode-specific axes -----
    if (c.mode == 1 && !s.vars.empty()) {
        if (c.axis_x_var < 0 || c.axis_x_var >= (int)s.vars.size()) c.axis_x_var = 0;
        if (c.axis_y_var < 0 || c.axis_y_var >= (int)s.vars.size())
            c.axis_y_var = (s.vars.size() > 1) ? 1 : 0;
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());

        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis X (slave IC)", &c.axis_x_var, items.data(), (int)items.size());
        InputNumStr("X lo", c.axis_x_lo_text, 120);
        InputNumStr("X hi", c.axis_x_hi_text, 120);
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Axis Y (slave IC)", &c.axis_y_var, items.data(), (int)items.size());
        InputNumStr("Y lo", c.axis_y_lo_text, 120);
        InputNumStr("Y hi", c.axis_y_hi_text, 120);
        InputNumStr("Resolution", c.n_pts_text, 120);
        ImGui::Separator();
    }
    else if (c.mode == 0 && !s.vars.empty()) {
        // На траектории — какие 2 переменные показывать как X/Y фазового портрета.
        std::vector<const char*> items;
        items.reserve(s.vars.size());
        for (const auto& v : s.vars) items.push_back(v.c_str());
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Display X var", &c.axis_x_var, items.data(), (int)items.size());
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Display Y var", &c.axis_y_var, items.data(), (int)items.size());
        ImGui::Separator();
    }

    ImGui::Text("Integration:");
    InputNumStr("h",                c.h_text,              120);
    if (c.mode == 0) {
        InputNumStr("t_max",        c.t_max_text,          120);
        InputNumStr("transient",    c.transient_text,      120);
    }
    InputNumStr("window",           c.window_text,         120);
    InputNumStr("iter of synch",    c.iter_of_synchr_text, 120);
    InputNumStr("decimator",        c.pre_scaller_text,    120);
    InputNumStr("max value",        c.max_value_text,      120);
    ImGui::Separator();

    // ----- Runtime knobs -----
    ImGui::Text("Synchro runtime:");
    static const char* tos_names[] = { "Unidirectional", "Bidirectional" };
    ImGui::SetNextItemWidth(160);
    ImGui::Combo("Type of synch.", &c.type_of_synch, tos_names, IM_ARRAYSIZE(tos_names));
    static const char* ee_names[] = {
        "0: RMS on last iter",
        "1: # iters to reach FS_error_trs",
        "2: RMS at last point"
    };
    ImGui::SetNextItemWidth(280);
    ImGui::Combo("Error estim.", &c.error_estim, ee_names, IM_ARRAYSIZE(ee_names));
    InputNumStr("FS error trs.", c.fs_error_trs_text, 120);
    ImGui::Separator();

    // ----- Per-var IC + coupling -----
    auto draw_var_block = [&](const char* title, std::map<std::string, std::string>& m, const char* id_prefix) {
        ImGui::Text("%s", title);
        for (const auto& v : s.vars) {
            std::string pid = std::string(id_prefix) + v;
            ImGui::PushID(pid.c_str());
            InputNumStr(v.c_str(), m[v], 120);
            ImGui::PopID();
        }
        ImGui::Separator();
    };
    draw_var_block("Master initial conditions:",                                   c.ic_master,  "icm_");
    draw_var_block("Slave initial conditions:",                                    c.ic_slave,   "ics_");
    draw_var_block("K forward (coupling on forward step, h>0):",                   c.k_forward,  "kf_");
    draw_var_block("K backward (coupling on backward step, h<0):",                 c.k_backward, "kb_");

    ImGui::Text("Parameters:");
    for (const auto& p : s.params) {
        ImGui::PushID(p.c_str());
        InputNumStr(p.c_str(), c.param_values[p], 120);
        ImGui::PopID();
    }
    ImGui::Separator();

    // (Visualization controls — colormap / autoscale / cmin-cmax / swap /
    // line width / alpha — теперь живут в toolbar'е над плотом, см.
    // draw_fastsync_plot. Здесь только compute-параметры.)
    ImGui::Separator();

    // ----- Run -----
    bool do_run = false;
    if (s.in_flight) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(160, 0));
        ImGui::EndDisabled();
    } else {
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(160, 0));
    }
    if (!s.in_flight && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_R, false))
        do_run = true;
    if (do_run) {
        if (!model.parametric_engine) model.parametric_engine = std::make_unique<ParametricEngine>();
        s.run_async(*model.parametric_engine, s.active_config_index);
    }
    ImGui::SameLine();
    if (s.in_flight && ImGui::Button("Cancel")) s.request_cancel();

    if (c.last_run_ok) {
        if (c.mode == 0)
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: traj %d pts, sync_err [%.4g, %.4g]",
                c.result.n_pts_traj, c.result.min_val, c.result.max_val);
        else
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                "OK: grid %dx%d, sync_err [%.4g, %.4g]",
                c.result.n_pts_grid, c.result.n_pts_grid,
                c.result.min_val, c.result.max_val);
    } else if (!c.last_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
        ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * 10);
        ImGui::InputTextMultiline("##fs_err",
            const_cast<char*>(c.last_error.c_str()),
            c.last_error.size() + 1,
            sz, ImGuiInputTextFlags_ReadOnly);
    }
}

// Plot Fast Synchro: либо colored trajectory (mode 0), либо heatmap (mode 1).
static void draw_fastsync_plot(AppModel& model) {
    FastSyncAnalysisSession& s = model.fastsync_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;
    FastSyncConfig& c = s.configs[s.active_config_index];

    // ---- Visualization toolbar (как у basins / Bif-2D / LLE-2D) ----
    {
        static const char* cmap_names[] = { "Viridis", "Inferno", "Turbo", "Gray" };
        if (c.colormap_idx < 0 || c.colormap_idx > 3) c.colormap_idx = 2;
        ImGui::SetNextItemWidth(140);
        ImGui::Combo("Colormap##fs", &c.colormap_idx, cmap_names, IM_ARRAYSIZE(cmap_names));
        ImGui::SameLine();
        ImGui::Checkbox("Autoscale color##fs", &c.autoscale_color);
        if (!c.autoscale_color) {
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            InputNumStr("vmin##fs", c.c_min_text, 80);
            ImGui::SameLine(); ImGui::SetNextItemWidth(80);
            InputNumStr("vmax##fs", c.c_max_text, 80);
        }
        if (c.mode == 0) {
            ImGui::SameLine(); ImGui::SetNextItemWidth(120);
            ImGui::SliderFloat("Line width##fs", &c.line_width, 0.1f, 5.0f, "%.2f");
            ImGui::SameLine(); ImGui::SetNextItemWidth(120);
            ImGui::SliderFloat("Alpha##fs",      &c.alpha,      0.0f, 1.0f, "%.2f");
            // Выбор Display X/Y живёт в Controls-панели (рядом с другими compute-
            // параметрами) — дублировать здесь не нужно.
        } else {
            ImGui::SameLine();
            if (ImGui::Button(c.swap_axes ? "Swap axes (on)##fs" : "Swap axes##fs"))
                c.swap_axes = !c.swap_axes;
        }
    }

    if (!c.last_run_ok) {
        ImGui::TextDisabled("No data yet. Press Run.");
        return;
    }

    const unsigned base_oid = 0x5F50000u + (unsigned)s.active_config_index;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::map<unsigned, std::unique_ptr<HeatmapView>> hm_map;
    static std::map<unsigned, std::unique_ptr<Plot2DView>>  traj_map;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();

    bool fit = c.fit_request;
    if (fit) c.fit_request = false;
    ImVec2 avail  = ImGui::GetContentRegionAvail();
    ImVec2 origin = ImGui::GetCursorScreenPos();

    auto var_name = [&](int idx) -> std::string {
        if (idx >= 0 && idx < (int)s.vars.size()) return s.vars[idx];
        return std::string("x");
    };

    auto parse_d_local = [](const std::string& s, double def) -> double {
        if (s.empty()) return def;
        try { return std::stod(s); } catch (...) { return def; }
    };
    // cmin/cmax: при autoscale берём диапазон актуальных значений из result.
    double cmin, cmax;
    if (c.autoscale_color) {
        cmin = c.result.min_val;
        cmax = c.result.max_val;
        if (!(cmax > cmin)) cmax = cmin + 1.0;
    } else {
        cmin = parse_d_local(c.c_min_text, -12.0);
        cmax = parse_d_local(c.c_max_text,   0.0);
    }
    HeatmapColormap cmap = (HeatmapColormap)((c.colormap_idx >= 0 && c.colormap_idx <= 3) ? c.colormap_idx : 2);

    if (c.mode == 0) {
        // Colored trajectory + manual colorbar справа.
        const int nX = c.result.amountOfX_traj;
        if (nX <= 0 || c.result.n_pts_traj <= 0) {
            ImGui::TextDisabled("No trajectory data.");
            return;
        }
        int vx = (c.axis_x_var >= 0 && c.axis_x_var < nX) ? c.axis_x_var : 0;
        int vy = (c.axis_y_var >= 0 && c.axis_y_var < nX) ? c.axis_y_var : (nX > 1 ? 1 : 0);

        // Резервируем место справа под colorbar (mirror HeatmapView layout).
        const float colorbar_w   = 18.0f;
        const float colorbar_gap = 12.0f;
        const float tick_len     = 4.0f;
        const float tick_text_gap = 2.0f;

        // Прикинем максимальную ширину tick-подписей colorbar'а.
        std::string lbl_lo = fmt_tick(cmin);
        std::string lbl_hi = fmt_tick(cmax);
        float max_tick_w = std::max(ImGui::CalcTextSize(lbl_lo.c_str()).x,
                                    ImGui::CalcTextSize(lbl_hi.c_str()).x);
        float cb_total = colorbar_w + colorbar_gap + tick_len + tick_text_gap + max_tick_w + 6.0f;

        ImVec2 plot_avail(std::max(64.0f, avail.x - cb_total), avail.y);

        auto& slot = traj_map[base_oid];
        if (!slot) {
            slot = std::make_unique<Plot2DView>();
            slot->points_mode  = false;
            slot->show_legend  = false;
            slot->imdraw_lines = true;
            slot->pad_x = true; slot->pad_y = true;
        }
        Plot2DView& v = *slot;
        v.x_axis.name = var_name(vx);
        v.y_axis.name = var_name(vy);
        v.line_thickness_px = c.line_width;

        // Если поменялись axes — форсим (a) fit, чтобы view нашёл новый bbox;
        // (b) re-upload GPU-кэша точек, иначе series_cache_.bbox() даст старый
        // диапазон. Поэтому передаём генерационный токен, зависящий от (vx,vy).
        static std::map<unsigned, std::pair<int,int>> last_axes;
        auto it_ax = last_axes.find(base_oid);
        if (it_ax == last_axes.end() || it_ax->second.first != vx || it_ax->second.second != vy) {
            v.view_valid = false;     // force autofit
            last_axes[base_oid] = { vx, vy };
        }

        // Пересобираем XY/values из полного буфера (без decimator'а — рисуем
        // все точки, ImDrawList сегмент-за-сегментом справляется).
        int n_in = c.result.n_pts_traj;
        static std::vector<float> xy_buf;
        static std::vector<float> err_buf;
        xy_buf.resize((size_t)n_in * 2);
        err_buf.resize((size_t)n_in);
        for (int i = 0; i < n_in; ++i) {
            xy_buf[2*i + 0] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vx];
            xy_buf[2*i + 1] = (float)c.result.traj_full[(size_t)i * nX + (size_t)vy];
            err_buf[i]      = (float)c.result.sync_error[i];
        }

        std::vector<PlotSeriesInput> series_in(1);
        series_in[0].points   = xy_buf.data();
        series_in[0].n_points = n_in;
        series_in[0].color    = ImVec4(1, 1, 1, c.alpha);  // .w → alpha-multiplier на cmap_sample()
        series_in[0].label    = "trajectory";
        series_in[0].values   = err_buf.data();
        series_in[0].colormap = cmap;
        series_in[0].cmin     = (float)cmin;
        series_in[0].cmax     = (float)cmax;
        std::vector<bool> vis(1, true);
        // Synthetic generation token: меняется при смене (data_generation, vx, vy),
        // чтобы Plot2DView::series_cache_ перезалил GPU-буфер → bbox()/autofit
        // подхватили новую X/Y проекцию.
        int gen_token = c.data_generation * 1000 + vx * 10 + vy;
        v.render(*renderer, origin, plot_avail,
                 /*owner_id*/ (int)base_oid, gen_token,
                 series_in, vis, vis, fit);

        // ---- Manual colorbar справа ----
        // Зеркалит HeatmapView::render section 9. Plot2DView использует
        // те же margin_left/top/bottom (78/20/46) — берём отсюда.
        const float margin_left   = 78.0f;
        const float margin_top    = 20.0f;
        const float margin_bottom = 46.0f;
        float plot_w = std::max(64.0f, plot_avail.x - margin_left - 20.0f);
        float plot_h = std::max(64.0f, plot_avail.y - margin_top  - margin_bottom);
        float cb_x = origin.x + margin_left + plot_w + colorbar_gap;
        float cb_y = origin.y + margin_top;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const int n_strips = 64;
        for (int k = 0; k < n_strips; ++k) {
            float t0 = (float)k / n_strips;
            float t1 = (float)(k + 1) / n_strips;
            float y0 = cb_y + plot_h * (1.0f - t1);
            float y1 = cb_y + plot_h * (1.0f - t0);
            ImU32 col = cmap_sample((t0 + t1) * 0.5f, cmap);
            dl->AddRectFilled(ImVec2(cb_x, y0), ImVec2(cb_x + colorbar_w, y1), col);
        }
        dl->AddRect(ImVec2(cb_x, cb_y), ImVec2(cb_x + colorbar_w, cb_y + plot_h),
                    IM_COL32(120, 120, 130, 200));
        ImU32 col_text = IM_COL32(220, 220, 230, 255);
        float font_h = ImGui::GetFontSize();
        // Метки на 5 равномерно распределённых уровнях.
        const int n_ticks = 5;
        for (int k = 0; k < n_ticks; ++k) {
            float t = (float)k / (n_ticks - 1);
            double v = cmin + t * (cmax - cmin);
            float y = cb_y + plot_h * (1.0f - t);
            dl->AddLine(ImVec2(cb_x + colorbar_w, y),
                        ImVec2(cb_x + colorbar_w + tick_len, y),
                        IM_COL32(120, 120, 130, 200));
            std::string s = fmt_tick(v);
            dl->AddText(ImVec2(cb_x + colorbar_w + tick_len + tick_text_gap,
                               y - font_h * 0.5f),
                        col_text, s.c_str());
        }
    }
    else {
        // Heatmap.
        auto& slot = hm_map[base_oid];
        if (!slot) slot = std::make_unique<HeatmapView>();
        HeatmapView& h = *slot;
        h.colormap = cmap;
        h.autoscale = c.autoscale_color;
        h.manual_vmin = (float)cmin;
        h.manual_vmax = (float)cmax;
        h.swap_axes = c.swap_axes;
        h.x_axis.name = var_name(c.result.axis_x_var) + "(0)";
        h.y_axis.name = var_name(c.result.axis_y_var) + "(0)";
        h.render(*renderer, origin, avail,
                 /*owner_id*/ (int)base_oid, c.data_generation,
                 c.result.n_pts_grid, c.result.n_pts_grid,
                 c.result.heatmap.data(),
                 c.result.axis_x_lo, c.result.axis_x_hi,
                 c.result.axis_y_lo, c.result.axis_y_hi,
                 c.result.min_val, c.result.max_val,
                 fit);
    }
}

// ============================================================
// Parametric Controls dispatcher — верхние табы Bif / LLE / LS.
// ============================================================
static void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Общий селектор системы — одна система на все три sub-анализа (Bif/LLE/LS).
    // Во время async-расчёта смена системы запрещена — иначе worker применит
    // результат к уже подменённой сессии.
    bool any_in_flight = model.bifurcation_session.in_flight
                      || model.lle_session.in_flight
                      || model.ls_session.in_flight;
    ImGui::Text("System:"); ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    std::string current = model.name.empty() ? "(current)" : model.name;
    if (any_in_flight) ImGui::BeginDisabled();
    if (ImGui::BeginCombo("##par_syssel", current.c_str())) {
        for (const auto& nm : lib.list()) {
            if (ImGui::Selectable(nm.c_str(), model.name == nm)) {
                try {
                    model.from_record(lib.load(nm));
                    model.start_parametric_analysis();
                    // Поверх эталона — последние рабочие сейвы для обеих сессий.
                    std::string jb = lib.load_session(model.loaded_name, "_last_parametric");
                    if (!jb.empty())
                        session_from_json_parametric(jb, model.bifurcation_session);
                    std::string jl = lib.load_session(model.loaded_name, "_last_lle");
                    if (!jl.empty())
                        session_from_json_lle(jl, model.lle_session);
                    std::string js = lib.load_session(model.loaded_name, "_last_ls");
                    if (!js.empty())
                        session_from_json_ls(js, model.ls_session);
                }
                catch (...) {}
            }
        }
        ImGui::EndCombo();
    }
    if (any_in_flight) ImGui::EndDisabled();
    ImGui::Separator();

    // Batch Run all — global across BD/LLE/LS. Popup shows checkboxes for
    // every configured diagram/curve/spectrum; Run pushes selected to
    // model.parametric_queue and starts the first one. draw_gui ticks the
    // queue after polls.
    if (any_in_flight) ImGui::BeginDisabled();
    if (ImGui::Button("Run all..."))
        ImGui::OpenPopup("##run_all_parametric");
    if (any_in_flight) ImGui::EndDisabled();
    if (!model.parametric_queue.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", model.parametric_queue.size());
    }
    if (ImGui::BeginPopup("##run_all_parametric")) {
        static std::vector<bool> picks_bd, picks_lle, picks_ls;
        auto fit = [&](std::vector<bool>& v, size_t n) {
            if (v.size() != n) v.assign(n, true);
        };
        fit(picks_bd,  model.bifurcation_session.diagrams.size());
        fit(picks_lle, model.lle_session.curves.size());
        fit(picks_ls,  model.ls_session.curves.size());

        ImGui::TextDisabled("Sequential (one CUDA context).");

        if (!picks_bd.empty()) {
            ImGui::SeparatorText("Bifurcation");
            for (size_t i = 0; i < picks_bd.size(); ++i) {
                bool v = picks_bd[i];
                std::string lbl = model.bifurcation_session.diagrams[i].label
                                  + "###pbd_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_bd[i] = v;
            }
        }
        if (!picks_lle.empty()) {
            ImGui::SeparatorText("LLE");
            for (size_t i = 0; i < picks_lle.size(); ++i) {
                bool v = picks_lle[i];
                std::string lbl = model.lle_session.curves[i].label
                                  + "###plle_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_lle[i] = v;
            }
        }
        if (!picks_ls.empty()) {
            ImGui::SeparatorText("LS");
            for (size_t i = 0; i < picks_ls.size(); ++i) {
                bool v = picks_ls[i];
                std::string lbl = model.ls_session.curves[i].label
                                  + "###pls_" + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks_ls[i] = v;
            }
        }

        ImGui::Separator();
        if (ImGui::Button("All")) {
            for (auto&& b : picks_bd)  b = true;
            for (auto&& b : picks_lle) b = true;
            for (auto&& b : picks_ls)  b = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("None")) {
            for (auto&& b : picks_bd)  b = false;
            for (auto&& b : picks_lle) b = false;
            for (auto&& b : picks_ls)  b = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            for (size_t i = 0; i < picks_bd.size(); ++i)
                if (picks_bd[i])
                    model.parametric_queue.push_back({ParametricQueueItem::Kind::Bifurcation, (int)i});
            for (size_t i = 0; i < picks_lle.size(); ++i)
                if (picks_lle[i])
                    model.parametric_queue.push_back({ParametricQueueItem::Kind::LLE, (int)i});
            for (size_t i = 0; i < picks_ls.size(); ++i)
                if (picks_ls[i])
                    model.parametric_queue.push_back({ParametricQueueItem::Kind::LS, (int)i});
            model.start_next_in_parametric_queue();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    if (ImGui::BeginTabBar("##parm_top")) {
        if (ImGui::BeginTabItem("Bifurcation")) {
            model.parametric_active_analysis = 0;
            draw_bifurcation_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LLE")) {
            model.parametric_active_analysis = 1;
            draw_lle_controls(model, lib);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LS")) {
            model.parametric_active_analysis = 2;
            draw_ls_controls(model, lib);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

// ============================================================
// Главное окно: переключатель режимов Library / Analysis / Parametric
// ============================================================
void draw_gui(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) {
    // полноэкранный dockspace-хост
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("MainHost", nullptr, host_flags);
    ImGui::PopStyleVar(2);

    // custom_schemes — единственное поле, которое может отредактироваться
    // в System tab БЕЗ переключения режима (т.е. без start_*_analysis).
    // Чтобы scheme combo в Phase/Parametric увидел свежий список сразу
    // после "+ Add custom scheme", синкаем копию live → сессии каждый кадр.
    model.phase_session.custom_schemes = model.custom_schemes;
    model.bifurcation_session.custom_schemes = model.custom_schemes;
    model.lle_session.custom_schemes         = model.custom_schemes;
    model.ls_session.custom_schemes          = model.custom_schemes;

    // poll'им async-расчёты независимо от текущего режима — чтобы при
    // возврате в этот режим пользователь сразу увидел готовый результат
    if (model.bifurcation_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_parametric",
                             session_to_json_parametric(model.bifurcation_session));
    }
    if (model.lle_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_lle",
                             session_to_json_lle(model.lle_session));
    }
    if (model.ls_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_ls",
                             session_to_json_ls(model.ls_session));
    }
    if (model.phase_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last",
                             session_to_json(model.phase_session));
    }
    // Basins: один config на сессию. Сохраняем JSON каждый кадр (после poll
    // - но также при изменении полей в controls). Здесь только after-poll save.
    if (model.basins_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_basins",
                             session_to_json_basins(model.basins_session));
    }
    // FastSync: poll worker future; on completion persist session JSON.
    // Без этого вызова in_flight никогда не сбрасывается → "Running" висит вечно.
    if (model.fastsync_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last_fastsync",
                             session_to_json_fastsync(model.fastsync_session));
    }

    // Tick parametric-очереди: если ни одна из BD/LLE/LS не in_flight и в
    // очереди есть элементы — берём следующий и стартуем. start_next сам
    // проверяет условие и безопасен к вызову каждый кадр.
    model.start_next_in_parametric_queue();
    // То же для basins-очереди (независимая).
    model.start_next_in_basins_queue();

    // переключатель режимов
    int mode = (int)model.app_mode;
    ImGui::RadioButton("Library", &mode, (int)AppModel::AppMode::Library); ImGui::SameLine();
    ImGui::RadioButton("Phase analysis", &mode, (int)AppModel::AppMode::Analysis); ImGui::SameLine();
    ImGui::RadioButton("Parametric", &mode, (int)AppModel::AppMode::Parametric); ImGui::SameLine();
    ImGui::RadioButton("Basins", &mode, (int)AppModel::AppMode::Basins); ImGui::SameLine();
    ImGui::RadioButton("Fast Synchro", &mode, (int)AppModel::AppMode::FastSync); ImGui::SameLine();
    ImGui::RadioButton("Settings", &mode, (int)AppModel::AppMode::Settings);

    // Индикатор компьюта — справа по правой границе окна, виден во всех режимах.
    // Layout: [text] [progress bar] [Stop] for in-flight cancellable sessions;
    // [text] only for phase or for "Done/Cancelled" persistent state. Stop also
    // drains parametric_queue and basins_queue so remaining batch items
    // don't auto-start.
    enum class BusyKind { None, Bif, LLE, LS, Basins, Phase, FastSync };
    BusyKind busy_kind = BusyKind::None;
    std::string busy_what;
    std::chrono::steady_clock::time_point busy_start;
    bool busy_cancelling = false;     // user already pressed Stop, waiting for engine
    bool show_done = false;           // not in flight — show persistent last-run info
    bool last_ok = true;              // for show_done: true = green "Done", false = red "Cancelled"
    double done_seconds = 0.0;
    float  progress_fraction = 0.0f;  // 0..1 from session.progress_token
    int    basins_phase = 0;          // 1 = sim, 2 = cluster; 0 = not basins or not started

    if (model.bifurcation_session.in_flight) {
        int ri = model.bifurcation_session.running_diagram_index;
        busy_what = (ri >= 0 && ri < (int)model.bifurcation_session.diagrams.size())
                        ? model.bifurcation_session.diagrams[ri].label : "bifurcation";
        busy_start = model.bifurcation_session.compute_start_time;
        busy_kind = BusyKind::Bif;
        busy_cancelling = model.bifurcation_session.cancel_token &&
                          model.bifurcation_session.cancel_token->load(std::memory_order_relaxed);
        if (model.bifurcation_session.progress_token)
            progress_fraction = model.bifurcation_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.lle_session.in_flight) {
        int ri = model.lle_session.running_curve_index;
        busy_what = (ri >= 0 && ri < (int)model.lle_session.curves.size())
                        ? model.lle_session.curves[ri].label : "LLE";
        busy_start = model.lle_session.compute_start_time;
        busy_kind = BusyKind::LLE;
        busy_cancelling = model.lle_session.cancel_token &&
                          model.lle_session.cancel_token->load(std::memory_order_relaxed);
        if (model.lle_session.progress_token)
            progress_fraction = model.lle_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.ls_session.in_flight) {
        int ri = model.ls_session.running_curve_index;
        busy_what = (ri >= 0 && ri < (int)model.ls_session.curves.size())
                        ? model.ls_session.curves[ri].label : "LS";
        busy_start = model.ls_session.compute_start_time;
        busy_kind = BusyKind::LS;
        busy_cancelling = model.ls_session.cancel_token &&
                          model.ls_session.cancel_token->load(std::memory_order_relaxed);
        if (model.ls_session.progress_token)
            progress_fraction = model.ls_session.progress_token->load(std::memory_order_relaxed);
    }
    else if (model.phase_session.in_flight) {
        busy_what  = "phase";
        busy_start = model.phase_session.compute_start_time;
        busy_kind  = BusyKind::Phase;
    }
    else if (model.basins_session.in_flight) {
        int ri = model.basins_session.running_config_index;
        busy_what = (ri >= 0 && ri < (int)model.basins_session.configs.size() &&
                     !model.basins_session.configs[ri].label.empty())
                        ? model.basins_session.configs[ri].label
                        : std::string("basins");
        busy_start = model.basins_session.compute_start_time;
        busy_kind  = BusyKind::Basins;
        busy_cancelling = model.basins_session.cancel_token &&
                          model.basins_session.cancel_token->load(std::memory_order_relaxed);
        if (model.basins_session.progress_token)
            progress_fraction = model.basins_session.progress_token->load(std::memory_order_relaxed);
        if (model.basins_session.progress_phase_token)
            basins_phase = model.basins_session.progress_phase_token->load(std::memory_order_relaxed);
    }
    else if (model.fastsync_session.in_flight) {
        int ri = model.fastsync_session.running_config_index;
        busy_what = (ri >= 0 && ri < (int)model.fastsync_session.configs.size() &&
                     !model.fastsync_session.configs[ri].label.empty())
                        ? model.fastsync_session.configs[ri].label
                        : std::string("fastsync");
        busy_start = model.fastsync_session.compute_start_time;
        busy_kind  = BusyKind::FastSync;
        busy_cancelling = model.fastsync_session.cancel_token &&
                          model.fastsync_session.cancel_token->load(std::memory_order_relaxed);
        if (model.fastsync_session.progress_token)
            progress_fraction = model.fastsync_session.progress_token->load(std::memory_order_relaxed);
    }
    else {
        // Nothing in flight — pick the session whose last run finished most
        // recently (across the 4 cancellable ones) and show persistent info.
        struct DoneCand { BusyKind kind; std::chrono::steady_clock::time_point ts; const std::string* label; bool ok; double secs; };
        DoneCand candidates[4] = {
            { BusyKind::Bif,    model.bifurcation_session.last_run_completed_at,
              &model.bifurcation_session.last_run_label,
              model.bifurcation_session.last_run_succeeded,
              model.bifurcation_session.last_run_seconds },
            { BusyKind::LLE,    model.lle_session.last_run_completed_at,
              &model.lle_session.last_run_label,
              model.lle_session.last_run_succeeded,
              model.lle_session.last_run_seconds },
            { BusyKind::LS,     model.ls_session.last_run_completed_at,
              &model.ls_session.last_run_label,
              model.ls_session.last_run_succeeded,
              model.ls_session.last_run_seconds },
            { BusyKind::Basins, model.basins_session.last_run_completed_at,
              &model.basins_session.last_run_label,
              model.basins_session.last_run_succeeded,
              model.basins_session.last_run_seconds },
        };
        const DoneCand* best = nullptr;
        for (const auto& c : candidates) {
            if (c.label->empty()) continue;
            if (!best || c.ts > best->ts) best = &c;
        }
        if (best) {
            busy_what    = *best->label;
            busy_kind    = best->kind;
            show_done    = true;
            last_ok      = best->ok;
            done_seconds = best->secs;
        }
    }

    if (busy_kind != BusyKind::None) {
        char text[200];
        // Phase suffix appears only for basins while running (not on "Cancelling"
        // or "Done" — those reflect overall state, not the current sub-phase).
        const char* phase_suffix = "";
        if (busy_kind == BusyKind::Basins && !show_done && !busy_cancelling) {
            if      (basins_phase == 1) phase_suffix = " (1/2 sim)";
            else if (basins_phase == 2) phase_suffix = " (2/2 cluster)";
        }
        if (show_done) {
            std::snprintf(text, sizeof(text), "%s %s in %.1fs",
                          last_ok ? "Done" : "Cancelled",
                          busy_what.c_str(), done_seconds);
        } else if (busy_cancelling) {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy_start).count();
            std::snprintf(text, sizeof(text), "Cancelling %s... %.1fs", busy_what.c_str(), secs);
        } else {
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - busy_start).count();
            // Queue suffix: prefer the queue that matches the running session
            // (parametric for Bif/LLE/LS, basins for Basins).
            size_t queue_n = 0;
            if (busy_kind == BusyKind::Basins) queue_n = model.basins_queue.size();
            else                               queue_n = model.parametric_queue.size();
            if (queue_n > 0)
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs (+%zu)",
                              busy_what.c_str(), phase_suffix, secs, queue_n);
            else
                std::snprintf(text, sizeof(text), "Computing %s%s... %.1fs",
                              busy_what.c_str(), phase_suffix, secs);
        }

        const bool show_stop = (busy_kind != BusyKind::Phase) &&
                               !show_done && !busy_cancelling;
        const bool show_bar  = show_stop;  // bar only when running & not cancelling

        const float pad      = ImGui::GetStyle().ItemSpacing.x;
        const float bar_w    = 120.0f;
        const float bar_h    = ImGui::GetTextLineHeight();
        const float text_w   = ImGui::CalcTextSize(text).x;
        const float stop_w   = show_stop
                               ? (ImGui::CalcTextSize("Stop").x +
                                  ImGui::GetStyle().FramePadding.x * 2.0f)
                               : 0.0f;
        float total_w = text_w + 12.0f;
        if (show_bar)  total_w += bar_w + pad;
        if (show_stop) total_w += stop_w + pad;

        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowSize().x - total_w);

        ImVec4 col;
        if (show_done) {
            col = last_ok ? ImVec4(0.55f, 0.95f, 0.55f, 1.0f)   // green
                          : ImVec4(0.95f, 0.45f, 0.45f, 1.0f);  // red
        } else {
            col = ImVec4(1.0f, 0.85f, 0.25f, 1.0f);              // yellow (running/cancelling)
        }
        ImGui::TextColored(col, "%s", text);

        if (show_bar) {
            ImGui::SameLine();
            float f = progress_fraction;
            if (f < 0.0f) f = 0.0f;
            if (f > 1.0f) f = 1.0f;
            ImGui::ProgressBar(f, ImVec2(bar_w, bar_h), "");
        }
        if (show_stop) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.55f, 0.15f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.75f, 0.20f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.90f, 0.25f, 0.25f, 1.0f));
            if (ImGui::SmallButton("Stop")) {
                switch (busy_kind) {
                    case BusyKind::Bif:    model.bifurcation_session.request_cancel(); break;
                    case BusyKind::LLE:    model.lle_session.request_cancel();         break;
                    case BusyKind::LS:     model.ls_session.request_cancel();          break;
                    case BusyKind::Basins: model.basins_session.request_cancel();      break;
                    default: break;
                }
                // Drain both queues so remaining batch items don't auto-start.
                model.parametric_queue.clear();
                model.basins_queue.clear();
            }
            ImGui::PopStyleColor(3);
        }
    }

    // Имя выбранной системы — по центру топ-бара. Рисуем после radio-кнопок
    // и busy-индикатора, чтобы не сдвигать их разметку (SetCursorPosX выходит
    // из cursor-flow). Если система не загружена — показываем плейсхолдер.
    {
        std::string sys_label = model.name.empty()
            ? std::string("(no system loaded)")
            : (std::string("System: ") + model.name);
        ImVec2 ts = ImGui::CalcTextSize(sys_label.c_str());
        float cx  = (ImGui::GetWindowSize().x - ts.x) * 0.5f;
        // У radio-кнопок Y был установлен по первому SameLine'у; берём ту же Y,
        // что у текущего курсора в начале строки (до индикатора).
        // SetCursorPosX переносит только по X — Y остаётся в текущей строке.
        ImGui::SameLine();
        ImGui::SetCursorPosX(cx);
        ImColor col = model.name.empty()
            ? ImColor(150, 150, 160, 200)
            : ImColor(220, 220, 230, 255);
        ImGui::TextColored(col, "%s", sys_label.c_str());
    }
    // При входе в Analysis/Parametric решаем, нужно ли (пере)инициализировать
    // сессию. Init происходит когда:
    //   1) система сменилась относительно той, для которой session была собрана;
    //   2) session ещё ни разу не была инициализирована (vars пустой) — это
    //      случай несохранённых систем, где model.name = loaded_system_name = "";
    //   3) у model сменился алфавит / vars_text, и session.vars/params уже
    //      не совпадают с актуальным model.known_vars/known_params.
    // Случай 3 раньше требовал перезапуска приложения, чтобы подхватить новый
    // алфавит — теперь подхватывается при следующем входе в режим.
    bool entering_phase  = (AppModel::AppMode)mode == AppModel::AppMode::Analysis &&
                           model.app_mode != AppModel::AppMode::Analysis;
    bool entering_par    = (AppModel::AppMode)mode == AppModel::AppMode::Parametric &&
                           model.app_mode != AppModel::AppMode::Parametric;
    bool entering_basins = (AppModel::AppMode)mode == AppModel::AppMode::Basins &&
                           model.app_mode != AppModel::AppMode::Basins;
    bool entering_fastsync = (AppModel::AppMode)mode == AppModel::AppMode::FastSync &&
                             model.app_mode != AppModel::AppMode::FastSync;
    if (entering_phase || entering_par || entering_basins || entering_fastsync) {
        // обновим known_vars/known_params из живого алфавита, чтобы сравнение
        // ниже было против актуального состояния
        model.refresh_symbols();
    }
    auto phase_need_init = model.phase_session.loaded_system_name != model.name
                        || model.phase_session.vars.empty()
                        || model.phase_session.vars   != model.known_vars
                        || model.phase_session.params != model.known_params;
    auto par_need_init   = model.bifurcation_session.loaded_system_name != model.name
                        || model.bifurcation_session.vars.empty()
                        || model.bifurcation_session.vars   != model.known_vars
                        || model.bifurcation_session.params != model.known_params;
    if (entering_phase && phase_need_init) {
        model.start_phase_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last");
            if (!j.empty()) session_from_json(j, model.phase_session);
        }
    }
    if (entering_par && par_need_init) {
        model.start_parametric_analysis();
        if (!model.loaded_name.empty()) {
            std::string j = lib.load_session(model.loaded_name, "_last_parametric");
            if (!j.empty()) session_from_json_parametric(j, model.bifurcation_session);
            std::string jl = lib.load_session(model.loaded_name, "_last_lle");
            if (!jl.empty()) session_from_json_lle(jl, model.lle_session);
            std::string js = lib.load_session(model.loaded_name, "_last_ls");
            if (!js.empty()) session_from_json_ls(js, model.ls_session);
        }
    }
    auto basins_need_init = model.basins_session.loaded_system_name != model.name
                         || model.basins_session.vars.empty()
                         || model.basins_session.vars   != model.known_vars
                         || model.basins_session.params != model.known_params;
    if (entering_basins && basins_need_init) {
        model.start_basins_analysis();
        if (!model.loaded_name.empty()) {
            std::string jb = lib.load_session(model.loaded_name, "_last_basins");
            if (!jb.empty()) session_from_json_basins(jb, model.basins_session);
        }
    }
    auto fastsync_need_init = model.fastsync_session.loaded_system_name != model.name
                           || model.fastsync_session.vars.empty()
                           || model.fastsync_session.vars   != model.known_vars
                           || model.fastsync_session.params != model.known_params;
    if (entering_fastsync && fastsync_need_init) {
        model.start_fastsync_analysis();
        if (!model.loaded_name.empty()) {
            std::string jf = lib.load_session(model.loaded_name, "_last_fastsync");
            if (!jf.empty()) session_from_json_fastsync(jf, model.fastsync_session);
        }
    }
    model.app_mode = (AppModel::AppMode)mode;
    ImGui::Separator();

    // dockspace для содержимого
    ImGuiID dockspace_id = ImGui::GetID("MainDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_None);

    ImGui::End(); // MainHost

    if (model.app_mode == AppModel::AppMode::Library) {
        // режим библиотеки: окно с вкладками System/Parameters/Library
        if (ImGui::Begin("Editor")) {
            if (ImGui::BeginTabBar("tabs")) {
                if (ImGui::BeginTabItem("System")) { draw_system_tab(model, cb); ImGui::EndTabItem(); }
                if (ImGui::BeginTabItem("Parameters")) { draw_parameters_tab(model); ImGui::EndTabItem(); }
                if (ImGui::BeginTabItem("Library")) { draw_library_tab(model, lib); ImGui::EndTabItem(); }
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Analysis) {
        // режим анализа: панель настроек + окна проекций (докаются пользователем)
        if (ImGui::Begin("Controls")) {
            draw_phase_controls(model, lib);
        }
        ImGui::End();
        draw_projection_windows(model);
    }
    else if (model.app_mode == AppModel::AppMode::Parametric) {
        if (ImGui::Begin("Parametric Controls")) {
            draw_parametric_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Bifurcation 1D")) {
            draw_bifurcation_plot(model);
        }
        ImGui::End();
        if (ImGui::Begin("LLE 1D")) {
            draw_lle_plot(model);
        }
        ImGui::End();
        if (ImGui::Begin("Lyapunov Spectrum")) {
            draw_ls_plot(model);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::Basins) {
        if (ImGui::Begin("Basins Controls")) {
            draw_basins_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Basins of Attraction")) {
            draw_basins_plot(model);
        }
        ImGui::End();
    }
    else if (model.app_mode == AppModel::AppMode::FastSync) {
        if (ImGui::Begin("FastSync Controls")) {
            draw_fastsync_controls(model, lib);
        }
        ImGui::End();
        if (ImGui::Begin("Fast Synchro")) {
            draw_fastsync_plot(model);
        }
        ImGui::End();
    }
    else { // AppMode::Settings
        if (ImGui::Begin("Settings")) {
            ImGui::Text("Interface scale");
            ImGui::TextDisabled("Auto-detected at startup from glfwGetMonitorContentScale.");
            ImGui::TextDisabled("Override persists in _app_config.json next to exe.");
            ImGui::Separator();

            ImGui::TextUnformatted("UI scale:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            // Применяем НЕ во время drag'а, а на отпускание (IsItemDeactivatedAfterEdit) —
            // иначе UI пересобирается на каждом кадре, виджет уходит из-под курсора.
            static float ui_slider_value = -1.0f;
            if (ui_slider_value < 0.0f) ui_slider_value = model.effective_ui_scale();
            if (!ImGui::IsAnyItemActive())
                ui_slider_value = model.effective_ui_scale();
            ImGui::SliderFloat("##ui_scale", &ui_slider_value, 0.5f, 3.0f, "%.2fx");
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                model.ui_scale_override = ui_slider_value;
                AppConfig cfg;
                cfg.ui_scale_override = ui_slider_value;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto")) {
                model.ui_scale_override = 0.0f;
                ui_slider_value = model.ui_scale_auto;
                AppConfig cfg;
                cfg.ui_scale_override = 0.0f;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::TextDisabled("Auto detected: %.2fx   |   Override: %s",
                model.ui_scale_auto,
                model.ui_scale_override > 0 ?
                    (std::to_string(model.ui_scale_override) + "x").c_str() :
                    "(off)");

            ImGui::Separator();
            ImGui::Text("Font");
            bool use_builtin = model.use_builtin_font;
            if (ImGui::Checkbox("Use built-in font (ProggyClean)", &use_builtin)) {
                model.use_builtin_font = use_builtin;
                // Сохраняем ОБА поля чтобы не сбросить override.
                AppConfig cfg;
                cfg.ui_scale_override = model.ui_scale_override;
                cfg.use_builtin_font  = use_builtin;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = model.tick_precision;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::TextDisabled("Off: Windows Segoe UI TTF (recommended, crisp at any scale).");
            ImGui::TextDisabled("On: built-in bitmap ProggyClean (compact, pixel-perfect at 1x/2x/3x).");

            ImGui::Separator();
            ImGui::Text("Axes");
            int tp = model.tick_precision;
            ImGui::SetNextItemWidth(220);
            if (ImGui::SliderInt("Tick precision (digits)", &tp, 2, 10)) {
                model.tick_precision = tp;
                set_tick_precision(tp);
                AppConfig cfg;
                cfg.ui_scale_override = model.ui_scale_override;
                cfg.use_builtin_font  = model.use_builtin_font;
                cfg.heatmap_colormap  = model.heatmap_colormap;
                cfg.basins_colormap        = model.basins_colormap;
                cfg.basins_avgpk_colormap  = model.basins_avgpk_colormap;
                cfg.basins_avgint_colormap = model.basins_avgint_colormap;
                cfg.basins_states_colormap = model.basins_states_colormap;
                cfg.tick_precision         = tp;
                save_app_config(get_exe_dir_with_sep(), cfg);
            }
            ImGui::TextDisabled("Significant digits in axis tick and colorbar labels.");
        }
        ImGui::End();
    }
}