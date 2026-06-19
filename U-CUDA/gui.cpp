#include "gui.h"
#include "imgui.h"
#include "implot.h"
#include "implot3d.h"
#include "plot_renderer.h"
#include "session_io.h"
#include "plot_view_2d.h"
#include "plot_view_3d.h"

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
    if (ImGui::Button("Select all")) model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))  model.scheme_euler = model.scheme_cromer = model.scheme_midpoint = model.scheme_rk4 = false;
    ImGui::Checkbox("Euler", &model.scheme_euler); ImGui::SameLine();
    ImGui::Checkbox("Euler-Cromer", &model.scheme_cromer); ImGui::SameLine();
    ImGui::Checkbox("Explicit Midpoint", &model.scheme_midpoint); ImGui::SameLine();
    ImGui::Checkbox("RK4", &model.scheme_rk4);

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
            "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4"
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

    // параметры: значение, min, max
    if (!model.known_params.empty()) {
        ImGui::SeparatorText("Parameters (value / min / max)");
        // заголовки колонок
        if (ImGui::BeginTable("params", 4, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("value");
            ImGui::TableSetupColumn("min");
            ImGui::TableSetupColumn("max");
            ImGui::TableHeadersRow();
            for (const auto& p : model.known_params) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%s", p.c_str());
                ImGui::TableSetColumnIndex(1);
                { std::string id = "##val_" + p; InputNumStr(id.c_str(), model.param_values[p], 100); }
                ImGui::TableSetColumnIndex(2);
                { std::string id = "##min_" + p; InputNumStr(id.c_str(), model.param_min[p], 100); }
                ImGui::TableSetColumnIndex(3);
                { std::string id = "##max_" + p; InputNumStr(id.c_str(), model.param_max[p], 100); }
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Empty fields are left unset. Min/max are for bifurcation sweeps.");
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
    static const char* methods[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4" };
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
    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4" };
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

    // ----- Parameter sweep -----
    if (!s.params.empty()) {
        std::vector<const char*> items;
        items.reserve(s.params.size());
        for (const auto& p : s.params) items.push_back(p.c_str());
        if (bd.param_index < 0 || bd.param_index >= (int)s.params.size()) bd.param_index = 0;
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Parameter", &bd.param_index, items.data(), (int)items.size());
    }
    else {
        ImGui::TextDisabled("No parameters (select a system first)");
    }
    InputNumStr("Param lo", bd.param_lo_text, 120);
    InputNumStr("Param hi", bd.param_hi_text, 120);
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
    if (ImGui::Checkbox("Plot inter-peaks instead of peak values", &bd.plot_inter_peaks))
        bd.fit_request = true;

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              bd.h_text,           120);
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
    if (to_remove >= 0) s.remove_diagram(to_remove);

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
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    if (!view) {
        view = std::make_unique<Plot2DView>();
        view->points_mode = true;
        view->show_legend = true;    // мульти-БД → нужна легенда (как в Phase)
        view->point_size_px = 2.0f;
        view->x_axis.name = "parameter";
        view->y_axis.name = "X";
    }

    if (s.diagrams.empty()) {
        ImGui::TextDisabled("No diagrams yet.");
        return;
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

    // Подписи осей: если ВСЕ видимые БД с данными свипают тот же параметр /
    // пишут ту же переменную — показываем имя; иначе generic "parameter"/"X".
    int shared_param_idx = -2;  // -2 = ещё не инициализировано
    int shared_var_idx   = -2;
    for (const auto& bd : s.diagrams) {
        if (!bd.visible || !bd.last_run_ok) continue;
        if (shared_param_idx == -2) shared_param_idx = bd.param_index;
        else if (shared_param_idx != bd.param_index) shared_param_idx = -1;
        if (shared_var_idx == -2) shared_var_idx = bd.writable_var;
        else if (shared_var_idx != bd.writable_var) shared_var_idx = -1;
    }
    view->x_axis.name = (shared_param_idx >= 0 && shared_param_idx < (int)s.params.size())
                          ? s.params[shared_param_idx] : std::string("parameter");
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
            double lo = safe_stod(bd.param_lo_text, 0.0);
            double hi = safe_stod(bd.param_hi_text, 1.0);
            int npts = bd.result.n_pts;
            for (int k = 0; k < npts; ++k) {
                if (k < (int)bd.result.flags.size() && bd.result.flags[k] < 0) continue;
                double x = (npts > 1) ? (lo + (hi - lo) * (double)k / (double)(npts - 1)) : lo;
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

    static const char* schemes[] = { "Euler", "Euler-Cromer", "Explicit Midpoint", "RK4" };
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

    if (!s.params.empty()) {
        std::vector<const char*> items;
        items.reserve(s.params.size());
        for (const auto& p : s.params) items.push_back(p.c_str());
        if (c.param_index < 0 || c.param_index >= (int)s.params.size()) c.param_index = 0;
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("Parameter", &c.param_index, items.data(), (int)items.size());
    }
    else {
        ImGui::TextDisabled("No parameters (select a system first)");
    }
    InputNumStr("Param lo", c.param_lo_text, 120);
    InputNumStr("Param hi", c.param_hi_text, 120);
    InputNumStr("Resolution", c.n_pts_text, 120);

    ImGui::Separator();
    ImGui::Text("Integration:");
    InputNumStr("h",              c.h_text,         120);
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

    if (c.last_run_ok) {
        int diverged = 0;
        for (int f : c.result.flags) if (f < 0) ++diverged;
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
            "OK: n_pts=%d, λ-curve computed", c.result.n_pts);
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
    if (to_remove >= 0) s.remove_curve(to_remove);

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
static void draw_lle_plot(AppModel& model) {
    LLEAnalysisSession& s = model.lle_session;
    static std::unique_ptr<PlotRenderer> renderer;
    static std::unique_ptr<Plot2DView> view;
    if (!renderer) renderer = std::make_unique<PlotRenderer>();
    if (!view) {
        view = std::make_unique<Plot2DView>();
        view->points_mode = false;   // LLE — непрерывная линия
        view->show_legend = true;
        view->x_axis.name = "parameter";
        view->y_axis.name = "lambda";
    }

    if (s.curves.empty()) {
        ImGui::TextDisabled("No curves yet.");
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

    // Подпись X: если все видимые кривые свипают тот же параметр — имя,
    // иначе generic "parameter".
    int shared_param_idx = -2;
    for (const auto& c : s.curves) {
        if (!c.visible || !c.last_run_ok) continue;
        if (shared_param_idx == -2) shared_param_idx = c.param_index;
        else if (shared_param_idx != c.param_index) shared_param_idx = -1;
    }
    view->x_axis.name = (shared_param_idx >= 0 && shared_param_idx < (int)s.params.size())
                          ? s.params[shared_param_idx] : std::string("parameter");
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
            double lo = safe_stod(c.param_lo_text, 0.0);
            double hi = safe_stod(c.param_hi_text, 1.0);
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
// Parametric Controls dispatcher — верхние табы Bif / LLE / LS.
// ============================================================
static void draw_parametric_controls(AppModel& model, SystemLibrary& lib) {
    ImGui::Text("Parametric analysis");
    ImGui::TextDisabled("Per-thread parameter sweep via NVRTC + NonLinAnal kernels.");

    // Общий селектор системы — одна система на все три sub-анализа (Bif/LLE/LS).
    // Во время async-расчёта смена системы запрещена — иначе worker применит
    // результат к уже подменённой сессии.
    bool any_in_flight = model.bifurcation_session.in_flight || model.lle_session.in_flight;
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
                }
                catch (...) {}
            }
        }
        ImGui::EndCombo();
    }
    if (any_in_flight) ImGui::EndDisabled();
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
            ImGui::TextDisabled("Lyapunov spectrum: coming soon.");
            ImGui::TextDisabled("(NonLinAnal already has LSKernelCUDA — will be enabled");
            ImGui::TextDisabled(" in a follow-up PR: unguard + UI/engine wrapper.)");
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
    if (model.phase_session.poll()) {
        if (!model.loaded_name.empty())
            lib.save_session(model.loaded_name, "_last",
                             session_to_json(model.phase_session));
    }

    // переключатель режимов
    int mode = (int)model.app_mode;
    ImGui::RadioButton("Library", &mode, (int)AppModel::AppMode::Library); ImGui::SameLine();
    ImGui::RadioButton("Phase analysis", &mode, (int)AppModel::AppMode::Analysis); ImGui::SameLine();
    ImGui::RadioButton("Parametric", &mode, (int)AppModel::AppMode::Parametric);

    // Индикатор компьюта — справа по правой границе окна, виден во всех режимах.
    std::string busy_what;
    std::chrono::steady_clock::time_point busy_start;
    if (model.bifurcation_session.in_flight) {
        int ri = model.bifurcation_session.running_diagram_index;
        if (ri >= 0 && ri < (int)model.bifurcation_session.diagrams.size())
            busy_what = model.bifurcation_session.diagrams[ri].label;
        else
            busy_what = "bifurcation";
        busy_start = model.bifurcation_session.compute_start_time;
    }
    else if (model.lle_session.in_flight) {
        int ri = model.lle_session.running_curve_index;
        if (ri >= 0 && ri < (int)model.lle_session.curves.size())
            busy_what = model.lle_session.curves[ri].label;
        else
            busy_what = "LLE";
        busy_start = model.lle_session.compute_start_time;
    }
    else if (model.phase_session.in_flight) {
        busy_what  = "phase";
        busy_start = model.phase_session.compute_start_time;
    }
    if (!busy_what.empty()) {
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - busy_start).count();
        char text[128];
        std::snprintf(text, sizeof(text), "Computing %s... %.1fs", busy_what.c_str(), secs);
        float text_w = ImGui::CalcTextSize(text).x;
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetWindowSize().x - text_w - 12.0f);
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.25f, 1.0f), "%s", text);
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
    bool entering_phase = (AppModel::AppMode)mode == AppModel::AppMode::Analysis &&
                          model.app_mode != AppModel::AppMode::Analysis;
    bool entering_par   = (AppModel::AppMode)mode == AppModel::AppMode::Parametric &&
                          model.app_mode != AppModel::AppMode::Parametric;
    if (entering_phase || entering_par) {
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
    else { // AppMode::Parametric
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
            ImGui::TextDisabled("Coming soon.");
            ImGui::TextDisabled("LSKernelCUDA already exists in NonLinAnal — will be");
            ImGui::TextDisabled("enabled in a follow-up PR (unguard + UI/engine wrapper).");
        }
        ImGui::End();
    }
}