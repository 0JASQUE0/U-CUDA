#include "gui_internal.h"

// Вкладка System: ввод системы, методы, генерация кода
// Generate button, error text and the generated-code viewer. Extracted because
// the System tab stops earlier for a map - no schemes and no custom KRS.
static void draw_generated_code_block(AppModel& model, const GuiCallbacks& cb) {
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
            ImVec2(-1, model.gen_code_editor_h), ImGuiInputTextFlags_ReadOnly);
        draw_resize_handle("##code_resize", model.gen_code_editor_h);
    }
}

// --- Коэффициенты обёртки: то, что иначе остаётся чёрным ящиком -------------
//
// И у Extr, и у Comp имя схемы описывает ВХОД (база, подшаги, выражения для
// gamma), а в КРС работают ВЫВЕДЕННЫЕ из него числа: веса Ричардсона alpha_k
// вычисляются, символьные gamma раскрываются значениями параметров. Увидеть их
// можно было ровно в одном месте — в конструкторе, и только пока в нём набран
// ровно этот набор; в списке собранных схем и в комбо выбора не было ничего.
//
// Отчёт собирается ТЕКСТОМ, а не рисуется таблицей, по одной причине: одни и
// те же числа нужны на экране, в тултипе комбо и в буфере обмена (коэффициенты
// уезжают в статью). Три способа нарисовать одно и то же разъехались бы.
std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values) {
    // Паспорт базы: встроенная — из таблицы, кастомная — со слов автора.
    auto base_traits = [&](const std::string& base, int* p, bool* sym) {
        if (builtin_scheme_traits(base, p, sym)) return true;
        for (const auto& cs : custom_schemes)
            if (cs.name == base) { *p = cs.order; *sym = cs.symmetric; return true; }
        return false;
    };
    char buf[256];
    std::string o;

    ExtrapolationSpec esp;
    CompositionSpec   csp;

    if (parse_extrapolation_name(name, &esp)) {
        int p = 1; bool sym = false;
        const bool known = base_traits(esp.base, &p, &sym);
        sym = extrapolation_symmetric(esp.re_at_output, sym);
        const int  K = (int)esp.n.size();
        long long  cost = 0;
        for (int v : esp.n) cost += v;

        o += name;
        o += "\n";
        if (esp.re_at_output)
            o += "ExtrZ: complex state across all substeps, Re taken ONCE on the output;\n"
                 "the real part expands in even powers only, so the weights are symmetric\n";
        std::snprintf(buf, sizeof(buf), "base %s: order %d, %s%s",
                      esp.base.c_str(), p,
                      sym ? "symmetric" : "non-symmetric",
                      known ? "" : "  (UNRESOLVED - the figures below assume order 1)");
        o += buf; o += "\n";
        std::snprintf(buf, sizeof(buf),
                      "%d stages, %lld base steps per macro-step  ->  order %d",
                      K, cost, extrapolation_order(K, p, sym));
        o += buf; o += "\n";
        // Что именно гасят эти веса: разложение идёт по u_k = 1/n_k у обычной
        // базы и по 1/n_k^2 у симметричной — отсюда и разный прирост порядка.
        o += sym ? "the weights cancel h^p, h^(p+2), ... (expansion in even powers only)\n"
                 : "the weights cancel h^p, h^(p+1), ...\n";
        o += "\n";

        const std::vector<double> alpha = extrapolation_weights(esp.n, p, sym);
        std::vector<std::string>  exact;
        const bool has_exact = extrapolation_weights_exact(esp.n, p, sym, &exact);

        o += has_exact ? "  k   n_k   substep     alpha_k (exact)   alpha_k (double)\n"
                       : "  k   n_k   substep     alpha_k\n";
        double asum = 0.0, amax = 0.0;
        for (int k = 0; k < K; ++k) {
            asum += alpha[(size_t)k];
            if (std::fabs(alpha[(size_t)k]) > amax) amax = std::fabs(alpha[(size_t)k]);
            if (has_exact)
                std::snprintf(buf, sizeof(buf), "  %-3d %-5d h/%-9d %-17s %.17g",
                              k, esp.n[(size_t)k], esp.n[(size_t)k],
                              exact[(size_t)k].c_str(), alpha[(size_t)k]);
            else
                std::snprintf(buf, sizeof(buf), "  %-3d %-5d h/%-9d %.17g",
                              k, esp.n[(size_t)k], esp.n[(size_t)k], alpha[(size_t)k]);
            o += buf; o += "\n";
        }
        std::snprintf(buf, sizeof(buf), "sum(alpha) = %.17g   max|alpha| = %.6g", asum, amax);
        o += buf; o += "\n";
        if (!has_exact)
            o += "the exact fractions do not fit in 2^53 - the kernel gets the doubles above\n";
        if (amax > 20.0) {
            std::snprintf(buf, sizeof(buf),
                          "cancellation: about %.1f decimal digits are lost in the weighted sum\n",
                          std::log10(amax));
            o += buf;
        }
        return o;
    }

    if (parse_composition_name(name, &csp)) {
        int p = 1; bool sym = false;
        const bool known = base_traits(csp.base, &p, &sym);
        const int  K = (int)csp.gammas.size();

        o += name;
        o += "\n";
        std::snprintf(buf, sizeof(buf), "base %s: order %d, %s%s",
                      csp.base.c_str(), p,
                      sym ? "symmetric" : "non-symmetric",
                      known ? "" : "  (UNRESOLVED)");
        o += buf; o += "\n";
        std::snprintf(buf, sizeof(buf), "%d stages, %d base steps per macro-step", K, K);
        o += buf; o += "\n";
        o += "\n";

        // Значения — при ТЕКУЩИХ параметрах. Для постоянного коэффициента это
        // то же самое число, для символьного — единственный способ его увидеть.
        std::vector<double> val;
        std::string         verr;
        const bool all_val = composition_gamma_values(csp, param_values, &val, &verr);

        o += "  k   gamma (as written)        value                 stage step\n";
        double gsum = 0.0, gcube = 0.0, gmin = 0.0;
        bool   sums_ok = true;
        for (int k = 0; k < K; ++k) {
            const double v = val[(size_t)k];
            if (std::isfinite(v)) {
                gsum += v; gcube += v * v * v;
                if (v < gmin) gmin = v;
            } else sums_ok = false;
            char vb[64], sb[64];
            if (std::isfinite(v)) {
                std::snprintf(vb, sizeof(vb), "%.17g", v);
                std::snprintf(sb, sizeof(sb), "%.6g*h", v);
            } else {
                std::snprintf(vb, sizeof(vb), "?");
                std::snprintf(sb, sizeof(sb), "?");
            }
            std::snprintf(buf, sizeof(buf), "  %-3d %-25s %-21s %s",
                          k, csp.gammas[(size_t)k].c_str(), vb, sb);
            o += buf; o += "\n";
        }
        if (!all_val) {
            o += "\n";
            o += "not every stage is numeric: ";
            o += verr.empty() ? std::string("a coefficient did not resolve") : verr;
            o += "\n";
            o += "(give the parameter a value on the tab, or type a number in its place)\n";
        }
        if (sums_ok) {
            o += "\n";
            std::snprintf(buf, sizeof(buf), "sum(gamma)   = %.17g%s",
                          gsum, std::fabs(gsum - 1.0) < 1e-12
                                    ? "   (consistent)"
                                    : "   NOT 1 - this integrates a rescaled field");
            o += buf; o += "\n";
            std::snprintf(buf, sizeof(buf), "sum(gamma^3) = %.6g", gcube);
            o += buf;
            if (sym && std::fabs(gcube) < 1e-12) {
                std::snprintf(buf, sizeof(buf), "   ->  expected order %d", p + 2);
                o += buf;
            } else if (sym) {
                std::snprintf(buf, sizeof(buf), "   not 0 - the order stays %d", p);
                o += buf;
            } else {
                o += "   (the base is not symmetric: the p+2 conditions do not apply)";
            }
            o += "\n";
            if (gmin < 0.0) {
                std::snprintf(buf, sizeof(buf),
                              "most negative stage: %.6g*h - a step BACKWARD in time", gmin);
                o += buf; o += "\n";
            }
        }
        return o;
    }

    return std::string();   // не обёртка: выводить из имени нечего
}

// Отчёт на экран. Моноширинного шрифта в проекте нет, поэтому колонки держатся
// пробелами, а блок печатается ПОСТРОЧНО через TextUnformatted: TextWrapped
// сломал бы выравнивание переносом, а один многострочный Text — отступами.
static void draw_wrapper_coefficients(const std::string& name,
                                      const std::vector<CustomScheme>& custom_schemes,
                                      const std::map<std::string, std::string>& param_values,
                                      const char* id) {
    const std::string rep = wrapper_coefficients_text(name, custom_schemes, param_values);
    if (rep.empty()) return;
    ImGui::PushID(id);
    if (ImGui::SmallButton("Copy coefficients")) ImGui::SetClipboardText(rep.c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Copies the whole block below as plain text.");
    for (size_t pos = 0; pos <= rep.size(); ) {
        size_t eol = rep.find('\n', pos);
        if (eol == std::string::npos) eol = rep.size();
        ImGui::TextDisabled("%s", rep.substr(pos, eol - pos).c_str());
        pos = eol + 1;
    }
    ImGui::PopID();
}

// Таблица коэффициентов — СТОЛБЦОМ, по строке на стадию: в ряд она не читается
// уже на пяти полях, а у Comp их до сорока. Высота ограничена max_rows
// строками, дальше список прокручивается, иначе сорок полей выдавливают со
// вкладки всё остальное. Открывает child — парный EndChild обязателен.
static void begin_coeff_column(const char* id, int rows, int max_rows) {
    const float row_h = ImGui::GetFrameHeightWithSpacing();
    const int   vis   = rows < max_rows ? rows : max_rows;
    ImGui::BeginChild(id, ImVec2(0.0f, row_h * (float)(vis < 1 ? 1 : vis)
                                       + ImGui::GetStyle().FramePadding.y * 2.0f), true);
}

// Поле имени схемы-обёртки, общее для обоих конструкторов. Пустое поле — это
// не "без имени", а "имя по умолчанию", поэтому подсказкой в нём стоит ровно то
// описание, которое иначе и встанет в список. Возвращает готовую метку: края
// подрезаны, разделители самого имени выброшены.
static std::string wrapper_name_field(const char* id, std::string& text,
                                      const std::string& default_name) {
    if (InputTextStrHint(id, text, default_name.c_str(), ImGui::GetFontSize() * 18.0f)) {
        // Чистим ПРЯМО В ПОЛЕ, иначе '|' виден пользователю, а в имя не
        // попадает. Пробелы по краям при этом не трогаем: их режет только
        // сборка имени, а то "S17 " нельзя было бы набрать.
        std::string keep;
        for (char c : text)
            if (c != '|' && c != ',' && c != '(' && c != ')') keep += c;
        text = keep;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Name shown in the scheme combo, the wrapper list and the\n"
                          "Order tab instead of the coefficients. Leave empty to keep\n"
                          "the generated description. The name is COSMETIC: the step is\n"
                          "rebuilt from the base and the coefficients either way.");
    return wrapper_sanitize_label(text);
}

// Конструктор экстраполяционных обёрток. Собирает ИМЯ вида "Extr(RK4|1,2,4)" —
// тела здесь нет и не хранится нигде: его пересобирает compute_krs_for_scheme
// при каждом обращении, поэтому правка системы обёртку не протухает.
// Схемы, читающие a[0] как коэффициент симметрии, держат паспортный порядок
// только при a[0] = 0.5; проверить это в кодогене нельзя (значение приходит в
// рантайме), поэтому предупреждаем здесь.
static bool extr_base_needs_half_s(const std::string& nm) {
    return nm == "CD" || nm == "CD fast" || nm == "CD FP" || nm == "CD fast FP"
        || nm == "Complex CD" || nm == "Complex CD4"
        || nm == "CCD4 (o4s3)" || nm == "CCD4 (o4s4)"
        || nm == "CD_{1-0}" || nm == "Complex CD_{1-0}" || nm == "Complex CD4_{1-0}"
        || nm == "CCD4 (o4s3)_{1-0}" || nm == "CCD4 (o4s4)_{1-0}"
        || nm == "SEMP" || nm == "SIMP";
}

static void draw_extrapolation_builder(AppModel& model) {
    ImGui::Spacing();
    if (!ImGui::CollapsingHeader("Extrapolated schemes (Richardson)",
        model.wrapper_schemes.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::TextDisabled(
        "Splits the step h into K independent runs of a base scheme: stage k goes\n"
        "from the SAME state in n[k] substeps of h/n[k]. The results are combined so\n"
        "that the first K-1 error terms cancel. Cost is sum(n) base steps per step.\n"
        "Every thread does the same work, so there is no warp divergence.");

    // --- выбор базы -------------------------------------------------------
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo("base scheme", model.extr_builder_base.c_str())) {
        std::string shown_group;
        for (const auto& b : kBuiltinSchemes) {
            if (!(model.*(b.flag)) && model.extr_builder_base != b.name) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group = scheme_group_title(b);
            }
            if (ImGui::Selectable(b.name, model.extr_builder_base == b.name))
                model.extr_builder_base = b.name;
        }
        if (!model.custom_schemes.empty()) ImGui::SeparatorText("Custom");
        for (const auto& cs : model.custom_schemes)
            if (ImGui::Selectable((cs.name + "##custom").c_str(),
                                  model.extr_builder_base == cs.name))
                model.extr_builder_base = cs.name;
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Only schemes ticked above are listed.\n"
                          "A custom KRS can be the base too - its order and symmetry\n"
                          "come from the fields next to its editor.");

    // Паспорт базы: у встроенной из таблицы, у кастомной — со слов автора.
    int  base_p   = 1;
    bool base_sym = false;
    bool base_known = builtin_scheme_traits(model.extr_builder_base, &base_p, &base_sym);
    if (!base_known) {
        for (const auto& cs : model.custom_schemes) {
            if (cs.name == model.extr_builder_base) {
                base_p = cs.order; base_sym = cs.symmetric; base_known = true; break;
            }
        }
    }

    ImGui::Checkbox("Re only on output (ExtrZ)", &model.extr_builder_re_output);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Run every stage in complex arithmetic and take Re ONCE, after the\n"
            "weighted sum - not after each base step. Needs a built-in complex\n"
            "CD scheme as the base (Complex CD, Complex CD4, CCD4 ...).\n"
            "The real part of the stage expansion then has even powers only,\n"
            "so the order is p + 2(K-1) for any such base. Measured on Lorenz:\n"
            "  CCD4 (s-split)_{0-1}: 1,2 -> 6, 1,2,3 -> 8   (plain Extr: 5, 5)\n"
            "  Complex CD4:    1,2 -> 6, 1,2,3 -> 8   (plain Extr: 6, 7)\n"
            "  Complex CD:     1,2 -> 4, 1,2,3 -> 6   (plain Extr: 3, 3)");
    const bool re_out = model.extr_builder_re_output;
    const bool eff_sym = extrapolation_symmetric(re_out, base_sym);

    // --- число стадий и сами n -------------------------------------------
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("stages K", &model.extr_builder_stages)) {
        if (model.extr_builder_stages < kExtrMinStages) model.extr_builder_stages = kExtrMinStages;
        if (model.extr_builder_stages > kExtrMaxStages) model.extr_builder_stages = kExtrMaxStages;
        // Слоты, открывшиеся при увеличении K, хранят прошлое значение (после
        // пресета "1,2,4" это 4 в слоте 3) и рвали бы строгое возрастание.
        // Правим ТОЛЬКО здесь, при смене K: набранное руками n не трогаем.
        for (int k = 1; k < model.extr_builder_stages; ++k)
            if (model.extr_builder_n[k] <= model.extr_builder_n[k - 1])
                model.extr_builder_n[k] = model.extr_builder_n[k - 1] + 1;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%d..%d stages. Each stage adds an order (two over a symmetric\n"
                          "base), but the weights grow with K: past ~8 stages the\n"
                          "cancellation warning below is the real limit, not this one.",
                          kExtrMinStages, kExtrMaxStages);
    const int K = model.extr_builder_stages;

    ImGui::TextUnformatted("substeps n:");
    begin_coeff_column("##extr_n_column", K, kExtrMaxStages);
    for (int k = 0; k < K; ++k) {
        ImGui::PushID(k);
        char lbl[24];
        std::snprintf(lbl, sizeof(lbl), "n[%d]", k);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputInt(lbl, &model.extr_builder_n[k], 0, 0);
        if (model.extr_builder_n[k] < 1) model.extr_builder_n[k] = 1;
        if (model.extr_builder_n[k] > kExtrMaxSubsteps) model.extr_builder_n[k] = kExtrMaxSubsteps;
        ImGui::PopID();
    }
    ImGui::EndChild();

    auto preset = [&](const char* label, const int* vals, int cnt, const char* tip) {
        if (ImGui::SmallButton(label)) {
            model.extr_builder_stages = cnt;
            for (int k = 0; k < cnt; ++k) model.extr_builder_n[k] = vals[k];
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        ImGui::SameLine();
    };
    static const int kHarm3[] = { 1,2,3 }, kHarm4[] = { 1,2,3,4 };
    static const int kDbl3[]  = { 1,2,4 }, kBulirsch[] = { 1,2,4,6 };
    ImGui::TextUnformatted("presets:"); ImGui::SameLine();
    preset("1,2,3",   kHarm3,    3, "Harmonic, 3 stages. Cheapest at this order (6 base steps).");
    preset("1,2,3,4", kHarm4,    4, "Harmonic, 4 stages. 10 base steps.");
    preset("1,2,4",   kDbl3,     3, "Step doubling (Romberg). 7 base steps.");
    preset("1,2,4,6", kBulirsch, 4, "Bulirsch-style. 13 base steps - the harmonic 1,2,3,4\n"
                                    "reaches the same order for 10.");
    ImGui::NewLine();

    // --- валидация и живое превью ----------------------------------------
    std::vector<int> n(model.extr_builder_n, model.extr_builder_n + K);
    std::string problem;
    for (int k = 1; k < K; ++k)
        if (n[k] <= n[k - 1]) { problem = "substep counts must strictly increase"; break; }
    if (!base_known) problem = "base scheme is not resolvable";
    else if (re_out && !scheme_has_complex_core(model.extr_builder_base))
        problem = "Re only on output needs a built-in complex CD scheme as the base";

    const std::string deflt = make_extrapolation_name(model.extr_builder_base, n, {}, re_out);
    const std::string label = wrapper_name_field("scheme name##extr",
                                                 model.extr_builder_label, deflt);
    const std::string new_name = make_extrapolation_name(model.extr_builder_base, n, label, re_out);
    // Совпадение ищем по имени БЕЗ метки: две обёртки с одной базой и одними
    // подшагами — это одна схема, как бы её ни подписали.
    bool duplicate = false;
    for (const auto& nm : model.wrapper_schemes)
        if (wrapper_canonical_name(nm) == deflt) duplicate = true;

    ImGui::Separator();
    if (!problem.empty()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "  %s", problem.c_str());
    }
    else {
        // Стоимость, порядок и сами веса печатает отчёт ниже; здесь остаётся
        // только max|alpha| — по нему выводится предупреждение о сокращении.
        const std::vector<double> alpha = extrapolation_weights(n, base_p, eff_sym);
        double amax = 0.0;
        for (double v : alpha) if (std::fabs(v) > amax) amax = std::fabs(v);

        // Полная раскладка — та же, что у собранной обёртки в списке ниже и в
        // тултипе комбо: пользователь видит ОДНИ И ТЕ ЖЕ числа до нажатия Add
        // и после него.
        draw_wrapper_coefficients(new_name, model.custom_schemes, model.param_values,
                                  "extr_preview");

        // Знакопеременные веса усиливают шум округления пропорционально max|alpha|.
        // Для double и разумных наборов это ничто, но "1,2,3,100" пользователь
        // вбить может, и там сокращение уже съедает разряды.
        if (amax > 20.0)
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                "  max|weight| = %.0f: heavy cancellation, ~%.1f decimal digits lost.",
                amax, std::log10(amax));

        if (extr_base_needs_half_s(model.extr_builder_base)) {
            const double s_now = parse_num(model.symmetry_s, 0.5);
            if (s_now != 0.5)
                ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                    "  '%s' is order %d only at s = 0.5, and s is currently %g:\n"
                    "  the base drops to order 1 and the wrapper gains nothing.",
                    model.extr_builder_base.c_str(), base_p, s_now);
            else
                ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                    "  '%s' holds order %d only while s (a[0]) stays 0.5 - do not sweep it.",
                    model.extr_builder_base.c_str(), base_p);
        }
    }

    ImGui::BeginDisabled(!problem.empty() || duplicate);
    if (ImGui::Button("+ Add extrapolated scheme")) {
        model.wrapper_schemes.push_back(new_name);
        // Список держим упорядоченным (тип, опорник, стадии), а не по
        // времени добавления; индексы съезжают — переименование бросаем.
        sort_wrapper_schemes(model.wrapper_schemes);
        model.wrapper_rename_index = -1;
    }
    ImGui::EndDisabled();
    if (duplicate) {
        ImGui::SameLine();
        ImGui::TextDisabled("(already in the list)");
    }

}

// Читает таблицу коэффициентов композиции из файла в конструктор. Число стадий
// берётся ИЗ ФАЙЛА: у опубликованных методов высокого порядка их 17, 31, 35 —
// выставлять K руками перед загрузкой значило бы считать строки глазами.
// Разбор — parse_composition_coeff_file, здесь только файл и строка статуса.
static void comp_load_coeffs_from_file(AppModel& model, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        model.comp_load_status = "cannot open " + path;
        model.comp_load_failed = true;
        return;
    }
    std::string text, line;
    while (std::getline(in, line)) { text += line; text += '\n'; }

    std::vector<std::string> g;
    std::string err;
    if (!parse_composition_coeff_file(text, &g, &err)) {
        model.comp_load_status = err;
        model.comp_load_failed = true;
        return;
    }

    model.comp_builder_stages = (int)g.size();
    // Хвост чистим: иначе после файла на 17 стадий в слотах 18.. осталась бы
    // прошлая раскладка и всплыла бы при следующем увеличении K.
    for (int k = 0; k < kCompMaxStages; ++k)
        model.comp_builder_g[k] = (k < (int)g.size()) ? g[(size_t)k] : std::string();

    const size_t slash = path.find_last_of("\\/");
    model.comp_load_status = "loaded " + std::to_string(g.size()) + " coefficients from "
        + (slash == std::string::npos ? path : path.substr(slash + 1));
    model.comp_load_failed = false;
}

// Конструктор композиций. Имя вида "Comp(CD|g1,1-2*g1,g1)"; тела, как и у Extr,
// нигде нет — резолвер пересобирает его из имени.
static void draw_composition_builder(AppModel& model, const GuiCallbacks& cb) {
    ImGui::Spacing();
    if (!ImGui::CollapsingHeader("Composed schemes (triple jump / Suzuki)"))
        return;

    ImGui::TextDisabled(
        "Runs the base scheme K times in a row, stage k with step gamma[k]*h.\n"
        "Over a SYMMETRIC base of order p the composition reaches p+2 when\n"
        "sum(gamma) = 1 and sum(gamma^3) = 0. Cost is K base steps per step.\n"
        "A coefficient may be a parameter name - then the Order tab can sweep it.");

    // --- база -------------------------------------------------------------
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo("base scheme##comp", model.comp_builder_base.c_str())) {
        std::string shown_group;
        for (const auto& b : kBuiltinSchemes) {
            if (!(model.*(b.flag)) && model.comp_builder_base != b.name) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group = scheme_group_title(b);
            }
            if (ImGui::Selectable(b.name, model.comp_builder_base == b.name))
                model.comp_builder_base = b.name;
        }
        if (!model.custom_schemes.empty()) ImGui::SeparatorText("Custom");
        for (const auto& cs : model.custom_schemes)
            if (ImGui::Selectable((cs.name + "##custom").c_str(),
                                  model.comp_builder_base == cs.name))
                model.comp_builder_base = cs.name;
        ImGui::EndCombo();
    }

    int  base_p = 1;
    bool base_sym = false;
    bool base_known = builtin_scheme_traits(model.comp_builder_base, &base_p, &base_sym);
    if (!base_known) {
        for (const auto& cs : model.custom_schemes) {
            if (cs.name == model.comp_builder_base) {
                base_p = cs.order; base_sym = cs.symmetric; base_known = true; break;
            }
        }
    }

    // --- стадии -----------------------------------------------------------
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("stages K##comp", &model.comp_builder_stages)) {
        if (model.comp_builder_stages < kCompMinStages) model.comp_builder_stages = kCompMinStages;
        if (model.comp_builder_stages > kCompMaxStages) model.comp_builder_stages = kCompMaxStages;
        // Правка руками обесценивает строку "loaded N coefficients from ...":
        // в таблице уже не то, что приехало из файла.
        model.comp_load_status.clear();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%d..%d stages. Tables of published high-order compositions\n"
                          "are long - load them with the button below instead of typing.",
                          kCompMinStages, kCompMaxStages);
    // Кнопка стоит на строке заголовка таблицы, а не под пресетами: заполнение
    // из файла — это АЛЬТЕРНАТИВА набору коэффициентов руками, и искать её надо
    // там же, где таблица, а не ниже неё.
    ImGui::TextUnformatted("gamma:");
    ImGui::SameLine();
    if (ImGui::Button("Load coefficients from file...")) {
        if (cb.pick_open_file_text) {
            const std::string path = cb.pick_open_file_text();
            if (!path.empty()) comp_load_coeffs_from_file(model, path);
        }
        else {
            model.comp_load_status = "no file dialog available in this build";
            model.comp_load_failed = true;
        }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Plain text, one coefficient per line:\n"
                          "  0.127136927734878585\n"
                          "  0.561702537988802652\n"
                          "  ...\n"
                          "Blank lines, '#'/'%%'/'//' comments, a leading index column\n"
                          "and several comma-separated values per line are all accepted.\n"
                          "Digits are taken VERBATIM - nothing is rounded.\n"
                          "The stage count comes from the file (max %d).", kCompMaxStages);
    if (!model.comp_load_status.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(model.comp_load_failed ? ImVec4(1, 0.5f, 0.3f, 1)
                                                  : ImVec4(0.5f, 1, 0.5f, 1),
                           "%s", model.comp_load_status.c_str());
    }

    // Число стадий читаем ПОСЛЕ кнопки: загрузка файла меняет его прямо в этом
    // кадре, и таблица должна показать семнадцать строк сразу, а не со
    // следующего кадра.
    const int K = model.comp_builder_stages;

    // Двенадцать строк на виду: семнадцать стадий S17o8 уже не влезают, а
    // прокрутка внутри блока оставляет на экране остальной конструктор.
    begin_coeff_column("##comp_gamma_column", K, 12);
    for (int k = 0; k < K; ++k) {
        ImGui::PushID(k);
        char lbl[24];
        std::snprintf(lbl, sizeof(lbl), "gamma[%d]", k);
        // Ширина под 18 значащих цифр: "-0.606074483235848116" не должен
        // прокручиваться внутри поля, его читают глазами и сверяют со статьёй.
        if (InputTextStr(lbl, model.comp_builder_g[k], ImGui::GetFontSize() * 14.0f))
            model.comp_load_status.clear();
        ImGui::PopID();
    }
    ImGui::EndChild();

    // name_suffix != nullptr — пресет опубликованного метода: без имени схема в
    // комбо читалась бы строкой из семнадцати чисел, поэтому пустое имя
    // заполняется "<база> <суффикс>" (так их и называли руками: "CD S17o8").
    auto preset = [&](const char* label, const char* const* vals, int cnt, const char* tip,
                      const char* name_suffix = nullptr) {
        if (ImGui::SmallButton(label)) {
            model.comp_builder_stages = cnt;
            for (int k = 0; k < cnt; ++k) model.comp_builder_g[k] = vals[k];
            // Хвост за cnt чистим по той же причине, что и при загрузке файла:
            // иначе увеличение K вытащит наружу прошлую раскладку.
            for (int k = cnt; k < kCompMaxStages; ++k) model.comp_builder_g[k].clear();
            model.comp_load_status.clear();
            if (name_suffix && model.comp_builder_label.empty())
                model.comp_builder_label =
                    wrapper_sanitize_label(model.comp_builder_base + " " + name_suffix);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
        ImGui::SameLine();
    };
    // Йошида: 2*g^3 + (1-2g)^3 = 0 даёт g = 1/(2 - 2^(1/3)); Судзуки: пятикратная
    // композиция с w = 1/(4 - 4^(1/3)). Цифры выписаны, а не посчитаны на месте —
    // это имя схемы, оно уезжает в JSON и должно быть воспроизводимым.
    static const char* const kYoshida[] = { "1.3512071919596578", "-1.7024143839193155",
                                            "1.3512071919596578" };
    static const char* const kSuzuki[]  = { "0.41449077179437573", "0.41449077179437573",
                                            "-0.6579630871775029",
                                            "0.41449077179437573", "0.41449077179437573" };
    // S17o8 (Kahan & Li, 1997): палиндром из 17 стадий, порядок 8 над
    // симметричной базой 2-го порядка. 18 значащих цифр, как в статье: на них
    // sum(g) = 1 и sum(g^3) = sum(g^5) = sum(g^7) = 0 держатся до ~1e-17.
    static const char* const kS17o8[] = {
        "0.127136927734878585", "0.561702537988802652", "-0.382534719948830204",
        "0.16007605629464744",  "-0.401816374326806946", "0.187366716542278489",
        "0.260708709207792377", "0.290397388125161637",  "-0.606074483235848116",
        "0.290397388125161637", "0.260708709207792377",  "0.187366716542278489",
        "-0.401816374326806946", "0.16007605629464744",  "-0.382534719948830204",
        "0.561702537988802652", "0.127136927734878585" };
    static const char* const kLine[]    = { "g1", "1-2*g1", "g1" };
    static const char* const kFree[]    = { "g1", "g2", "g1" };
    ImGui::TextUnformatted("presets:"); ImGui::SameLine();
    preset("Yoshida", kYoshida, 3, "Triple jump, order 4 from a symmetric order-2 base.\n"
                                   "The middle step is NEGATIVE (-1.70 h): backward in time,\n"
                                   "which is what a real composition pays to kill the h^3 term.");
    preset("Suzuki",  kSuzuki,  5, "Suzuki 5-fold, also order 4 but with a much smaller\n"
                                   "negative step (-0.66 h) - 5 base steps instead of 3.");
    preset("S17o8",   kS17o8,  17, "Kahan & Li (1997): 17-stage palindromic composition,\n"
                                   "order 8 from a symmetric order-2 base (CD at s = 0.5,\n"
                                   "Implicit Midpoint, ...). 17 base steps per step.\n"
                                   "Fills an empty scheme name with \"<base> S17o8\".",
           "S17o8");
    preset("sweep g1", kLine,   3, "Consistency 2*g1 + g2 = 1 built in, so a 1D sweep of g1\n"
                                   "in the Order tab stays inside the triple-jump family.");
    preset("sweep g1,g2", kFree, 3, "Both coefficients free: gives the p(g1, g2) map.\n"
                                    "Declare g1 and g2 as parameters unused by the RHS.");
    ImGui::NewLine();

    // --- валидация и превью ------------------------------------------------
    // Число стадий читаем ЗАНОВО: пресет и загрузка из файла выше меняют его
    // прямо в этом кадре, а K захвачен до них — иначе превью один кадр
    // показывало бы 3 коэффициента вместо семнадцати только что загруженных.
    const int Kn = model.comp_builder_stages;
    std::vector<std::string> g;
    for (int k = 0; k < Kn; ++k) g.push_back(model.comp_builder_g[k]);
    const std::string deflt = make_composition_name(model.comp_builder_base, g);
    const std::string label = wrapper_name_field("scheme name##comp",
                                                 model.comp_builder_label, deflt);
    const std::string new_name = make_composition_name(model.comp_builder_base, g, label);

    CompositionSpec spec;
    std::string problem;
    if (!parse_composition_name(new_name, &spec, &problem)) { /* problem set */ }
    else if (!base_known) problem = "base scheme is not resolvable";
    else problem.clear();

    // Совпадение — по имени БЕЗ метки: подпись схему не меняет.
    bool duplicate = false;
    for (const auto& nm : model.wrapper_schemes)
        if (wrapper_canonical_name(nm) == deflt) duplicate = true;

    ImGui::Separator();
    if (!problem.empty()) {
        ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1), "  %s", problem.c_str());
    }
    else {
        // Имя, коэффициенты, их значения при текущих параметрах и обе суммы —
        // одним блоком, тем же, что показывают список обёрток и тултип комбо.
        draw_wrapper_coefficients(new_name, model.custom_schemes, model.param_values,
                                  "comp_preview");
        if (!base_sym)
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                "  base is not self-adjoint: the p+2 conditions above do not apply to it,\n"
                "  and the published coefficients were derived assuming they do.");

        // Цветные строки — только АЛАРМ; сами числа печатает отчёт выше.
        double gsum = 0.0, gcube = 0.0;
        if (composition_sums(spec, &gsum, &gcube)) {
            if (std::fabs(gsum - 1.0) >= 1e-12)
                ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                    "  sum != 1: this integrates the field scaled by %.6g, i.e. a different\n"
                    "  problem. The measured order stays valid, the solution does not.", gsum);
            else if (base_sym && std::fabs(gcube) < 1e-12)
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "  -> expected order %d", base_p + 2);
        }

        if (extr_base_needs_half_s(model.comp_builder_base)) {
            const double s_now = parse_num(model.symmetry_s, 0.5);
            if (s_now != 0.5)
                ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                    "  '%s' is symmetric only at s = 0.5, and s is currently %g:\n"
                    "  the composition conditions no longer hold.",
                    model.comp_builder_base.c_str(), s_now);
            else
                ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                    "  '%s' stays symmetric only while s (a[0]) is 0.5 - do not sweep it.",
                    model.comp_builder_base.c_str());
        }
    }

    ImGui::BeginDisabled(!problem.empty() || duplicate);
    if (ImGui::Button("+ Add composed scheme")) {
        model.wrapper_schemes.push_back(new_name);
        // Список держим упорядоченным (тип, опорник, стадии), а не по
        // времени добавления; индексы съезжают — переименование бросаем.
        sort_wrapper_schemes(model.wrapper_schemes);
        model.wrapper_rename_index = -1;
    }
    ImGui::EndDisabled();
    if (duplicate) {
        ImGui::SameLine();
        ImGui::TextDisabled("(already in the list)");
    }
}

// Общий список собранных обёрток. Один на оба конструктора: список один, и
// проверка отвалившейся базы у Extr и Comp дословно совпадает.
static void draw_wrapper_list(AppModel& model) {
    if (model.wrapper_schemes.empty()) return;
    ImGui::Spacing();
    if (!ImGui::CollapsingHeader("Assembled wrappers", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    int to_delete = -1;
    // Своя область ID на весь список: строки нумеруются с нуля и здесь, и
    // в "Custom KRS schemes" выше, а CollapsingHeader своей области не
    // открывает (это TreeNodeBehavior без TreePush). Без этого PushID(0) +
    // SmallButton("Delete") в обоих списках даёт один и тот же ID.
    ImGui::PushID("wrapper_list");
    std::string shown_wgroup;   // список отсортирован, см. sort_wrapper_schemes
    for (int i = 0; i < (int)model.wrapper_schemes.size(); ++i) {
        const std::string wg = wrapper_group_title(model.wrapper_schemes[i]);
        if (wg != shown_wgroup) {
            if (!wg.empty()) ImGui::SeparatorText(wg.c_str());
            shown_wgroup = wg;
        }
        ImGui::PushID(i);
        {
            bool on = model.wrapper_enabled(model.wrapper_schemes[i]);
            if (ImGui::Checkbox("##wr_on", &on))
                model.set_wrapper_enabled(model.wrapper_schemes[i], on);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Offer this wrapper in the scheme lists of the tabs\n"
                                  "and include its code in Generate. A tab that already\n"
                                  "uses it keeps it.");
            ImGui::SameLine();
        }
        const std::string shown = wrapper_display_name(model.wrapper_schemes[i]);
        // Строка-раскрывашка, а не просто текст: коэффициенты у пяти стадий
        // занимают полэкрана, а список обычно просматривают целиком.
        // AllowOverlap обязателен в паре со SpanAvailWidth: тот растягивает
        // хитбокс строки до правого края, и без него Delete оказывается ПОД
        // раскрывашкой — клик по кнопке уходил ей и просто раскрывал строку.
        // С флагом приоритет получает предмет, выданный на строке позже,
        // то есть сама кнопка.
        const bool open = ImGui::TreeNodeEx("##row",
                                            ImGuiTreeNodeFlags_SpanAvailWidth
                                                | ImGuiTreeNodeFlags_AllowOverlap,
                                            "%s", shown.c_str());
        if (shown != model.wrapper_schemes[i] && ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", model.wrapper_schemes[i].c_str());
        ImGui::SameLine();
        if (model.wrapper_rename_index == i) {
            // Имя записываем в схему по выходу из поля (Enter или клик мимо),
            // а не на каждый символ: имя — это удостоверение схемы, и менять
            // его на каждой набранной букве незачем.
            const std::string deflt = wrapper_canonical_name(model.wrapper_schemes[i]);
            // Фокус ставим ОДИН раз, на первом кадре после нажатия Rename:
            // иначе поле перехватывало бы клавиатуру у всей вкладки.
            if (model.wrapper_rename_focus) {
                ImGui::SetKeyboardFocusHere();
                model.wrapper_rename_focus = false;
            }
            InputTextStrHint("##rename", model.wrapper_rename_text, deflt.c_str(),
                             ImGui::GetFontSize() * 18.0f);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                model.wrapper_schemes[i] =
                    wrapper_relabel(model.wrapper_schemes[i], model.wrapper_rename_text);
                model.wrapper_rename_index = -1;
            }
            else if (ImGui::IsItemDeactivated()) model.wrapper_rename_index = -1;
        }
        else {
            if (ImGui::SmallButton("Rename")) {
                model.wrapper_rename_index = i;
                model.wrapper_rename_text  = wrapper_label(model.wrapper_schemes[i]);
                model.wrapper_rename_focus = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Give the scheme a short name. Empty restores the\n"
                                  "generated one. Sessions saved under the old name keep\n"
                                  "computing the same thing - the name is cosmetic.");
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) to_delete = i;
        }
        if (open) {
            draw_wrapper_coefficients(model.wrapper_schemes[i], model.custom_schemes,
                                      model.param_values, "coef");
            ImGui::TreePop();
        }

        // Отвалившаяся база (кастомную КРС переименовали или удалили) — на Run
        // это станет пустой КРС, поэтому говорим сразу.
        ExtrapolationSpec esp;
        CompositionSpec   csp;
        std::string base;
        if (parse_extrapolation_name(model.wrapper_schemes[i], &esp)) base = esp.base;
        else if (parse_composition_name(model.wrapper_schemes[i], &csp)) base = csp.base;
        if (!base.empty()) {
            bool ok = builtin_scheme_traits(base, nullptr, nullptr);
            if (!ok)
                for (const auto& cs : model.custom_schemes)
                    if (cs.name == base) { ok = true; break; }
            if (!ok)
                ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                    "  base '%s' no longer exists.", base.c_str());
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    if (to_delete >= 0) {
        // Галочку тоже забываем: заново собранная та же обёртка — включена.
        model.set_wrapper_enabled(model.wrapper_schemes[(size_t)to_delete], true);
        model.wrapper_schemes.erase(model.wrapper_schemes.begin() + to_delete);
        // Индексы за удалённой строкой съехали — переименование бросаем, иначе
        // введённое имя досталось бы соседней схеме.
        model.wrapper_rename_index = -1;
    }
}

// Галочки встроенных схем ОДУ (флаги scheme_* системы): вкладка System редактора ОДУ и редактор
// ДАУ при «Solve as ODE».
static void draw_builtin_scheme_checkboxes(AppModel& model) {
    ImGui::Text("Schemes to generate:");
    // И кнопки, и сам список идут по kBuiltinSchemes и пропускают схемы,
    // скрытые в Settings: скрытая схема не должна ни показываться, ни
    // включаться скопом.
    auto scheme_shown = [&](const BuiltinScheme& b) { return !model.scheme_hidden(b.name); };
    if (ImGui::Button("Select all"))
        for (const auto& b : kBuiltinSchemes) if (scheme_shown(b)) model.*b.flag = true;
    ImGui::SameLine();
    if (ImGui::Button("Clear all"))
        for (const auto& b : kBuiltinSchemes) if (scheme_shown(b)) model.*b.flag = false;

    // Схемы сгруппированы по порядку точности — тот же порядок, что в комбо
    // выбора схемы и в списке генерации (AppModel::generate). Таблица
    // отсортирована по order, поэтому шапка печатается на его смене, и группа
    // целиком скрытых схем не оставляет пустого заголовка.
    // У CD, Complex CD, SEMP и SIMP порядок группы достигается только при
    // s = a[0] = 0.5; при других s все четыре падают до первого.
    {
        std::string  shown_group;
        bool first_in_line = true;
        for (const auto& b : kBuiltinSchemes) {
            if (!scheme_shown(b)) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group   = scheme_group_title(b);
                first_in_line = true;
            }
            if (!first_in_line) ImGui::SameLine();
            first_in_line = false;
            ImGui::Checkbox(b.name, &(model.*b.flag));
            if (b.tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", b.tooltip);
        }
    }

    // Маска из Settings — фильтр интерфейса, флаги системы она не трогает.
    // Поэтому скрытая схема может остаться включённой у системы, и Generate
    // её выпустит, хотя чекбокса рядом уже нет — об этом надо сказать прямо.
    {
        std::string hidden_on;
        for (const auto& b : kBuiltinSchemes) {
            if (!model.scheme_hidden(b.name) || !(model.*b.flag)) continue;
            if (!hidden_on.empty()) hidden_on += ", ";
            hidden_on += b.name;
        }
        if (!hidden_on.empty())
            ImGui::TextDisabled("Hidden in Settings, still enabled for this system: %s",
                                hidden_on.c_str());
    }
}

// Галочки методов ДАУ (список выключенных — AppModel::dae_schemes_off). Порядок — dae_scheme_names.
static void draw_dae_method_checkboxes(AppModel& model) {
    ImGui::Text("DAE methods to generate:");
    if (ImGui::Button("Select all##dae")) model.dae_schemes_off.clear();
    ImGui::SameLine();
    if (ImGui::Button("Clear all##dae")) {
        model.dae_schemes_off = dae_scheme_names();
        model.dae_schemes_off.push_back(kBdfSchemeDae);
    }
    // Группы: неявные (любой индекс), с решением связи при каждом x (только индекс 1), многошаговый.
    // Подсказки — по имени (список методов рос, по номеру они съезжали).
    static const std::map<std::string, const char*> kTips = {
        { "DAE BDF1", "Implicit Euler on [x; z], one Newton per step. Order 1, L-stable, any index." },
        { "DAE Trapezoidal", "Trapezoid in x, constraint at the new node. Order 2, not L-stable (index 3 diverges)." },
        { "DAE TR-BDF2", "TR-BDF2: trapezoid to t + gamma h, then BDF2. Order 2, L-stable." },
        { "DAE SDIRK2", "SDIRK2 (Alexander): two implicit Euler stages with gamma h. Order 2 (z: 1 on index 2), L-stable." },
        { "DAE Implicit Midpoint", "Implicit midpoint: control method, does not hold an index-2 constraint, diverges on index 3." },
        { "DAE Radau IIA (5)", "Radau IIA, 3 stages: order 5 (index 1), x 5 / z 3 (index 2), x 3 / z 2 (index 3).\n"
                               "Newton full = 0 in Integration: RADAU5 simplified Newton (one J per step)." },
        { "DAE BDF1 + GGL", "BDF1 on the GGL-stabilised system (index 3 -> 2, multipliers local to the step)." },
        { "DAE Radau IIA + GGL", "Radau IIA on the GGL-stabilised system: index 3 with x 5 / lambda 3." },
        { "DAE Rosenbrock ROS2", "Rosenbrock ROS2: linearly implicit, one LU per step, no Newton. Order 2, L-stable." },
        { "DAE Rosenbrock ROS3P", "Rosenbrock ROS3P (Lang-Verwer): order 3 in x and z, A-stable, one LU per step." },
        { "DAE Rodas4", "Rodas4 (Hairer-Wanner): order 4, stiffly accurate, L-stable, one LU per step.\n"
                        "With Integration -> Adaptive step: embedded error estimate (index 1)." },
        { "DAE CD", "CD for DAE (level B): explicit forward half with z re-solved after every x update, implicit\n"
                    "reverse half with one Newton on (x_i, z). Order 2 at s = 0.5. Index 1 only." },
        { "DAE CD + Yoshida-4", "Yoshida composition of three CD steps: order 4 at s = 0.5. Index 1 only." },
        { "DAE CD (index 2)", "CD with z constant per half-step, chosen so that g holds at its end. Index 2 only." },
        { "DAE Explicit Euler", "Explicit Euler with z solved from the constraint at every stage. Index 1, non-stiff." },
        { "DAE Explicit Midpoint", "Explicit midpoint with z solved at every stage. Order 2, index 1, non-stiff." },
        { "DAE RK4", "RK4 with z solved at every stage. Order 4, index 1, non-stiff." },
        { kBdfSchemeDae, "BDF orders 1-5 with variable step (DASSL form), its own multistep driver.\n"
                         "Adaptive step only (Analysis on CPU / GPU, Parametric sweeps). Index 1-2 directly,\n"
                         "index 3 through GGL. Generate prints its residual [f; g] and Jacobian." },
    };
    std::vector<std::string> names = dae_scheme_names();
    names.push_back(kBdfSchemeDae);
    bool first_in_line = true;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i == 0) ImGui::SeparatorText("implicit, any index");
        if (names[i] == "DAE CD") { ImGui::SeparatorText("constraint solved whenever x changes, index 1"); first_in_line = true; }
        if (names[i] == kBdfSchemeDae) { ImGui::SeparatorText("multistep, adaptive step only"); first_in_line = true; }
        if (!first_in_line) ImGui::SameLine();
        first_in_line = false;
        bool on = model.dae_scheme_on(names[i]);
        if (ImGui::Checkbox(names[i].c_str(), &on)) {
            auto& off = model.dae_schemes_off;
            off.erase(std::remove(off.begin(), off.end(), names[i]), off.end());
            if (!on) off.push_back(names[i]);
        }
        auto tip = kTips.find(names[i]);
        if (tip != kTips.end() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip->second);
    }
}

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
    // Отображение или ОДУ — решает вкладка Library (MAP / ODE), а не галочка:
    // снятая галочка увела бы черновик в другую вкладку посреди правки.
    if (model.is_map) {
        ImGui::SameLine();
        ImGui::TextDisabled("(discrete map)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("System is given as x_{n+1} = f(x_n): no integration\n"
                              "scheme, no step h, and time is the iteration number n.");
    }
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
        // Поле ввода слева, набранные формулы справа — поровну.
        const float gap   = ImGui::GetStyle().ItemSpacing.x;
        const float x0    = ImGui::GetCursorPosX();
        const float half  = std::floor((ImGui::GetContentRegionAvail().x - gap) * 0.5f);
        ImGui::Text("LaTeX (editable - fix OCR errors here):");
        ImGui::SameLine(x0 + half + gap);
        ImGui::Text("Preview:");
        InputTextMultilineStr("##latex", model.latex_text, ImVec2(half, model.latex_editor_h));
        ImGui::SameLine(0.0f, gap);
        draw_latex_preview("##latex_preview", model.latex_text, ImVec2(-1, model.latex_editor_h));
        draw_resize_handle("##latex_resize", model.latex_editor_h);
        if (ImGui::CollapsingHeader("LaTeX format examples")) {
            if (model.is_map)
                ImGui::TextDisabled(
                    "Each equation on its own line, LHS is the next iterate:\n"
                    "  x_{n+1} = 1 - a x_n^{2} + y_n \\\\\n  y_{n+1} = b x_n\n"
                    "Supported: \\frac{a}{b}, x^{2}, \\sin x, \\sin^{2} x, \\cdot, |x|,\n"
                    "  greek \\sigma. The iteration index (_n, _{n+1}) is stripped.");
            else
                ImGui::TextDisabled(
                    "Each equation on its own line, LHS must have a derivative:\n"
                    "  \\dot{x} = \\sigma(y-x) \\\\\n  \\dot{y} = x(\\rho-z)-y\n"
                    "Supported: \\frac{a}{b}, x^{2}, \\sin x, \\sin^{2} x, \\cdot, |x|,\n"
                    "  subscripts x_{m}, greek \\sigma. Derivatives: \\dot{x}, x', dx/dt.");
        }
    }
    else {
        ImGui::Text("Equations (plain syntax):");
        InputTextMultilineStr("##plain", model.plain_text, ImVec2(-1, model.plain_editor_h));
        draw_resize_handle("##plain_resize", model.plain_editor_h);
        if (ImGui::CollapsingHeader("Plain format examples")) {
            if (model.is_map)
                ImGui::TextDisabled(
                    "  x_{n+1} = 1 - a*x_n^2 + y_n \\\\\n  y_{n+1} = b*x_n\n"
                    "Use * for multiplication, ^ for powers. LHS needs x_{n+1}=.");
            else
                ImGui::TextDisabled(
                    "  \\dot{x} = sigma*(y - x) \\\\\n  \\dot{y} = x*(rho - z) - y\n"
                    "Use * for multiplication, ^ for powers. LHS needs \\dot{x}= or x'=.");
        }
    }
    draw_rhs_preview(model);

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // вспомогательные функции
    ImGui::Checkbox("Use auxiliary functions", &model.use_aux_funcs);
    if (model.use_aux_funcs) {
        ImGui::Text("Function definitions (one per line, e.g. h(x) = m_1 x + ...):");
        InputTextMultilineStr("##funcs", model.func_defs_text, ImVec2(-1, model.funcs_editor_h));
        draw_resize_handle("##funcs_resize", model.funcs_editor_h);
        if (ImGui::CollapsingHeader("Auxiliary function examples")) {
            ImGui::TextDisabled(
                "h(x) = m_1 x + \\frac{1}{2}(m_0-m_1)(|x+1| - |x-1|)\n"
                "Then call h(x) in equations. Body is inlined.\n"
                "IMPORTANT: function params (m_0, m_1) must be in the alphabet too.");
        }
    }

    // Variables / Parameters (новый раздельный формат)
    ImGui::Text("Variables:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Auto-detect")) {
        // Scan latex_text (or plain_text если режим Plain) и заполнить vars/params.
        const std::string& src = (model.mode == InputMode::Plain)
                                 ? model.plain_text
                                 : model.latex_text;
        DetectedAlphabet det = detect_alphabet(src, model.is_map);
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

    // Legacy: единый алфавит
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

    // A map has no scheme: codegen emits exactly one step body.
    if (model.is_map) {
        ImGui::TextDisabled("Discrete map: the step is the right-hand side itself,\n"
                            "so there is no scheme to pick and no step h.");
        ImGui::Spacing();
        draw_generated_code_block(model, cb);
        return;
    }

    // методы
    draw_builtin_scheme_checkboxes(model);

    // Настройки Ньютона на уровне системы — значения по умолчанию для новых вкладок, как
    // symmetry s: каждая вкладка держит свои (NewtonSettings, поля в блоке Integration)
    // и передаёт их в ядро рантайм-слотами a[M+2..M+4].
    if (model.scheme_ieuler || model.scheme_imidpoint) {
        ImGui::Spacing();
        ImGui::Text("Newton solver (defaults for new tabs):");
        int jac_mode = model.newton_full ? 1 : 0;
        ImGui::SetNextItemWidth(kFieldW * 2.0f);
        if (ImGui::Combo("jacobian", &jac_mode, "modified (once per step)\0full (every iteration)\0"))
            model.newton_full = (jac_mode == 1);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("modified: Jacobian and LU reused across iterations and\n"
                              "refreshed when the correction stops shrinking.\n"
                              "Cheaper on GPU and the right default.\n"
                              "full: rebuilt every iteration. Quadratic convergence,\n"
                              "most robust at large h, ~k times the cost.");
        InputNumStr("tolerance", model.newton_tol, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Newton stops once ||dX|| falls below this.");
        InputNumStr("max iterations", model.newton_max_iters, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hard cap on Newton passes per step. 2-3 is typical\n"
                              "from the explicit-Euler predictor.");
    }

    // Custom KRS schemes (raw C/CUDA код вместо codegen)
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Custom KRS schemes",
        model.custom_schemes.empty() ? 0 : ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextDisabled(
            "Raw C/CUDA in calculateDiscreteModel body. Available: X[0..N-1] (vars),\n"
            "a[0] (symmetry s, same slot as CD), a[1..M] (params), h (step), AMOUNTOFX.\n"
            "if/for/while + math functions OK.\n"
            "Runs on GPU (NVRTC) and on CPU (compiled with cl.exe from Visual Studio).");

        // существующие схемы
        int to_delete = -1;
        // Парная к draw_wrapper_list область ID (см. комментарий там).
        ImGui::PushID("custom_krs_list");
        for (int i = 0; i < (int)model.custom_schemes.size(); ++i) {
            auto& cs = model.custom_schemes[i];
            ImGui::PushID(i);
            // Свернуть/развернуть код. Состояние в модели по имени схемы, как
            // и высота редактора: индекс съехал бы при удалении соседней.
            bool& collapsed = model.custom_scheme_code_collapsed[cs.name];
            if (ImGui::ArrowButton("##cs_fold", collapsed ? ImGuiDir_Right : ImGuiDir_Down))
                collapsed = !collapsed;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(collapsed ? "Show the code" : "Hide the code");
            ImGui::SameLine();
            ImGui::Checkbox("##cs_on", &cs.enabled);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Offer this scheme in the scheme lists of the tabs\n"
                                  "and include it in Generate. Unticked, it stays here\n"
                                  "and can still be the base of a wrapper; a tab that\n"
                                  "already uses it keeps it.");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(220);
            std::string name_label = "name##cs_name_" + std::to_string(i);
            InputTextStr(name_label.c_str(), cs.name);
            ImGui::SameLine();
            if (ImGui::SmallButton("Delete")) to_delete = i;
            // Паспорт КРС. Нужен ТОЛЬКО когда эта схема берётся опорной для
            // Extr(...): из порядка и симметричности считаются веса стадий.
            // Кодоген тела не разбирает, так что сказать может только автор.
            ImGui::SetNextItemWidth(90);
            std::string ord_label = "order##cs_ord_" + std::to_string(i);
            if (ImGui::InputInt(ord_label.c_str(), &cs.order)) {
                if (cs.order < 1)  cs.order = 1;
                if (cs.order > 12) cs.order = 12;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Order of accuracy of this step.\n"
                                  "Used only when this scheme is the base of an\n"
                                  "Extr(...) wrapper - it sets the extrapolation\n"
                                  "weights. Getting it wrong does not produce wrong\n"
                                  "numbers, it just cancels the wrong error terms\n"
                                  "and the wrapper falls back to the base order.");
            ImGui::SameLine();
            std::string sym_label = "symmetric##cs_sym_" + std::to_string(i);
            ImGui::Checkbox(sym_label.c_str(), &cs.symmetric);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Tick if the error expansion has EVEN powers of h only.\n"
                                  "True for self-adjoint steps (Phi* composed with Phi):\n"
                                  "implicit midpoint, trapezoid, leapfrog, CD at s = 0.5.\n"
                                  "NOT true for explicit midpoint / RK4 / predictor-corrector.\n"
                                  "A symmetric base gains 2 orders per extrapolation stage\n"
                                  "instead of 1. Same as above: a wrong tick only costs order.");
            if (collapsed) {
                const int n_lines = cs.body.empty()
                    ? 0 : 1 + (int)std::count(cs.body.begin(), cs.body.end(), '\n');
                ImGui::TextDisabled("  code hidden (%d line%s)", n_lines, n_lines == 1 ? "" : "s");
            } else {
                std::string body_label = "##cs_body_" + std::to_string(i);
                float& body_h = model.custom_scheme_editor_h.try_emplace(cs.name, 100.0f).first->second;
                InputTextMultilineStr(body_label.c_str(), cs.body, ImVec2(-1, body_h));
                const std::string resize_id = "##cs_body_resize_" + std::to_string(i);
                draw_resize_handle(resize_id.c_str(), body_h);
            }
            // Проверка обращений X[k] / a[k] с константным индексом против
            // размерности ТЕКУЩЕЙ системы. Статическая и живая — тела короткие,
            // проход по строке стоит копейки. Пока система не распознана
            // (known_vars пуст) молчим, иначе выдали бы ложные ошибки на всё.
            if (!model.known_vars.empty()) {
                std::vector<KrsCpuDiag> diags;
                krs_cpu_check_indices(cs.body,
                                      (int)model.known_vars.size(),
                                      (int)model.known_params.size() + 1,
                                      diags);
                for (const auto& d : diags)
                    ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "  line %d: %s",
                                       d.line, d.message.c_str());
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
        ImGui::PopID();
        if (to_delete >= 0) model.custom_schemes.erase(model.custom_schemes.begin() + to_delete);

        // блокируем добавление с уже существующим/built-in именем
        // (список встроенных — общий kBuiltinSchemes, см. верх файла)
        if (ImGui::Button("+ Add custom scheme")) {
            // подобрать уникальное имя "Custom N"
            int n = (int)model.custom_schemes.size() + 1;
            std::string candidate;
            auto name_clash = [&](const std::string& nm) {
                if (nm == kMapSchemeName) return true;
                for (const auto& b : kBuiltinSchemes) if (nm == b.name) return true;
                for (const auto& cs : model.custom_schemes) if (cs.name == nm) return true;
                return false;
            };
            do { candidate = "Custom " + std::to_string(n++); } while (name_clash(candidate));
            CustomScheme cs; cs.name = candidate;
            model.custom_schemes.push_back(std::move(cs));
        }

        // подсветка конфликтов
        for (const auto& cs : model.custom_schemes) {
            if (cs.name == kMapSchemeName) {
                ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                    "  '%s' is reserved for discrete maps; rename it.", kMapSchemeName);
                continue;
            }
            for (const auto& b : kBuiltinSchemes) {
                if (cs.name == b.name) {
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
        // Символы, ломающие имя обёртки Extr(<база>|n...): по ним резолвер
        // режет строку, так что база с ними стала бы неразрешимой.
        for (const auto& cs : model.custom_schemes) {
            if (cs.name.find_first_of("|()") != std::string::npos) {
                ImGui::TextColored(ImVec4(1, 0.5f, 0.3f, 1),
                    "  '%s' contains | ( or ) - those are reserved by Extr(...) names.",
                    cs.name.c_str());
            }
        }
    }

    draw_extrapolation_builder(model);
    draw_composition_builder(model, cb);
    draw_wrapper_list(model);

    draw_generated_code_block(model, cb);
}

// Вкладка Parameters: НУ, значения/диапазоны параметров, шаг
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

    // Discretization step; always 1 for a map, and not editable there.
    if (!model.is_map) {
        ImGui::Text("Discretization step h:");
        ImGui::SameLine();
        InputNumStr("##step_h", model.step_h, kFieldW);
        ImGui::TextDisabled("(leave empty to skip)");
    }

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

// Library tab: two states — list (table of saved systems + note preview) and editor
// (System + Parameters sub-tabs with Save/Cancel). The editor edits
// AppModel::library_edit_buffer (a scratch AppModel), never the live model, so
// browsing/editing/Cancel never affects the system active in Parametric/Phase/Basins/FastSync.

// Trim leading/trailing ASCII whitespace.
[[nodiscard]] static std::string trim_copy(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
LibTab lib_tab_of(bool is_dae, bool is_map) {
    return is_dae ? LibTab::Dae : is_map ? LibTab::Map : LibTab::Ode;
}
static LibraryListCache g_library_cache;

static void invalidate_library_cache() {
    g_library_cache.valid = false;
    g_library_cache.sel_name.clear();
    g_library_cache.sel_desc.clear();
    g_library_cache.sel_ok = false;
}

const LibraryListCache& library_list_cached(SystemLibrary& lib) {
    const double now = ImGui::GetTime();
    if (g_library_cache.valid && now - g_library_cache.filled_at < kLibraryCacheTtlSec)
        return g_library_cache;

    g_library_cache.names = lib.list();
    g_library_cache.custom_counts.assign(g_library_cache.names.size(), 0);
    g_library_cache.tab.assign(g_library_cache.names.size(), (char)LibTab::Ode);
    for (size_t i = 0; i < g_library_cache.names.size(); ++i) {
        try {
            const SystemRecord r = lib.load(g_library_cache.names[i]);
            g_library_cache.custom_counts[i] = (int)r.custom_schemes.size();
            g_library_cache.tab[i]           = (char)lib_tab_of(r.is_dae, r.is_map);
        } catch (...) {}
    }
    g_library_cache.filled_at = now;
    g_library_cache.valid     = true;
    return g_library_cache;
}

// Persist the Library visibility mask. Read-modify-write, like the other
// call sites: save_app_config rewrites the whole file.
static void persist_hidden_systems(const AppModel& model) {
    AppConfig cfg;
    load_app_config(get_exe_dir_with_sep(), cfg);
    cfg.hidden_systems = model.hidden_systems;
    save_app_config(get_exe_dir_with_sep(), cfg);
}

// Plain-text dump of a library record: everything the editor shows except the
// scheme checkboxes. Lets the Library tab hand the description to the clipboard
// without opening the editor (where a stray edit could be saved over the system).
static std::string describe_record(const SystemRecord& r) {
    std::string o;
    auto line = [&o](const char* key, const std::string& v) {
        if (!v.empty()) o += std::string(key) + ": " + v + "\n";
    };
    auto block = [&o](const char* key, const std::string& v) {
        if (v.empty()) return;
        o += std::string(key) + ":\n" + v;
        if (v.back() != '\n') o += '\n';
    };
    auto map_line = [&o](const char* key, const std::map<std::string, std::string>& m) {
        std::string acc;
        for (const auto& p : m) {
            if (p.second.empty()) continue;
            if (!acc.empty()) acc += ", ";
            acc += p.first + " = " + p.second;
        }
        if (!acc.empty()) o += std::string(key) + ": " + acc + "\n";
    };

    line("Name", r.name);
    o += std::string("Type: ") + (r.is_dae ? "DAE system" : r.is_map ? "discrete map" : "ODE system") + "\n";
    block("Note", r.note);
    block("LaTeX", r.latex_text);
    block("Plain", r.plain_text);
    line(r.is_dae ? "Differential variables" : "Variables", r.vars_text);
    if (r.is_dae) line("Algebraic variables", r.alg_text);
    line("Parameters", r.params_text);
    // Legacy records keep vars and params mixed in one list.
    if (r.vars_text.empty() && r.params_text.empty()) line("Alphabet", r.alphabet_text);
    if (r.use_aux_funcs) block("Aux functions", r.func_defs_text);
    line("Step h", r.step_h);
    map_line("Initial conditions", r.init_conditions);
    map_line("Parameter values", r.param_values);
    if (!r.custom_schemes.empty()) {
        std::string acc;
        for (const auto& cs : r.custom_schemes) {
            if (!acc.empty()) acc += ", ";
            acc += cs.name;
        }
        line("Custom schemes", acc);
    }
    return o;
}

// Save-side logic for the editor: validate name, handle rename for EditExisting, persist
// the scratch buffer via SystemLibrary. Mirrors the save into the live model only if the
// edited system is the one currently active elsewhere — saving any other system must not
// disturb what Parametric/Phase/Basins/FastSync have loaded. On success returns to list
// mode; on failure sets model.edit_error and stays in the editor.
static void library_editor_save(AppModel& model, SystemLibrary& lib) {
    AppModel& buf = *model.library_edit_buffer;
    std::string name = trim_copy(buf.name);
    if (name.empty()) {
        model.edit_error = "Name is required.";
        return;
    }
    buf.name = name;

    try {
        if (model.library_edit_mode == AppModel::LibraryEditMode::EditExisting) {
            const bool renaming = (name != model.edit_original_name);
            if (renaming) {
                if (lib.exists(name)) {
                    model.edit_error = "Name '" + name + "' already exists.";
                    return;
                }
                if (!lib.rename(model.edit_original_name, name)) {
                    model.edit_error = "Rename failed.";
                    return;
                }
                // The visibility tick is keyed by name — carry it to the new one.
                if (model.system_hidden(model.edit_original_name)) {
                    AppModel::set_name_hidden(model.hidden_systems, model.edit_original_name, false);
                    AppModel::set_name_hidden(model.hidden_systems, name, true);
                    persist_hidden_systems(model);
                }
            }
        } else {
            // AddNew
            if (lib.exists(name)) {
                model.edit_error = "Name '" + name + "' already exists.";
                return;
            }
        }
        buf.name = lib.save(buf.to_record());
        invalidate_library_cache();   // список и превью заметки устарели

        const bool editing_active_system =
            model.library_edit_mode == AppModel::LibraryEditMode::EditExisting &&
            !model.loaded_name.empty() &&
            model.edit_original_name == model.loaded_name;
        if (editing_active_system) {
            model.from_record(buf.to_record());
            model.loaded_name = model.name;
            // Push live custom_schemes / sys / vars / params into every session
            // so the next Run in Parametric/Basins/FastSync/Phase picks up the
            // edited state without reopening the tab.
            model.propagate_to_sessions();
            model.generate();
        }

        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::None;
    }
    catch (const std::exception& e) {
        model.edit_error = e.what();
    }
}

// List state: table of saved systems + Add-new + note preview panel.
// Interacting with this view (row select, Edit, Add new) only ever touches
// model.library_edit_buffer — never the live model — so it never changes
// the system currently active in Parametric/Phase/Basins/FastSync.
// tab — какая вкладка рисуется: в ней только записи своего вида (kind в system.json).
void draw_library_list(AppModel& model, SystemLibrary& lib, LibTab tab) {
    // Черновик один на все вкладки: пока он открыт в соседней, Add/Edit отсюда
    // молча затёрли бы несохранённую правку.
    const bool other_tab_editing = model.library_edit_mode != AppModel::LibraryEditMode::None;
    if (other_tab_editing) {
        const AppModel& eb = *model.library_edit_buffer;
        ImGui::TextDisabled("Finish editing in the %s tab first.",
                            kLibTabName[(int)lib_tab_of(eb.is_dae, eb.is_map)]);
    }
    ImGui::BeginDisabled(other_tab_editing);
    const char* add_label = tab == LibTab::Dae ? "Add new DAE system"
                          : tab == LibTab::Map ? "Add new map" : "Add new system";
    const bool add_clicked = ImGui::Button(add_label);
    ImGui::EndDisabled();
    if (add_clicked) {
        model.library_edit_buffer->clear();
        model.library_edit_buffer->is_dae = tab == LibTab::Dae;
        model.library_edit_buffer->is_map = tab == LibTab::Map;
        if (tab == LibTab::Dae) {
            // ДАУ: полный Ньютон и запас итераций. Предиктор z — прежнее значение, и при
            // быстро меняющемся множителе (маятник) квадратичная фаза начинается лишь на
            // ~9-й итерации; 8 итераций модифицированного Ньютона молча принимали
            // несошедшийся шаг, и связь уплывала.
            model.library_edit_buffer->newton_full      = true;
            model.library_edit_buffer->newton_max_iters = "25";
            // «Solve as ODE» берёт схемы ОДУ из флагов scheme_*: RK4 годится и уровню B.
            model.library_edit_buffer->scheme_rk4       = true;
        }
        model.edit_original_name.clear();
        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::AddNew;
    }
    ImGui::Separator();

    // Список и счётчики custom-схем — из кэша (см. library_list_cached):
    // раньше здесь был обход каталога и чтение всех system.json на каждом кадре.
    const LibraryListCache& cache = library_list_cached(lib);
    std::vector<std::string> names;
    std::vector<int>         cs_counts;
    for (size_t i = 0; i < cache.names.size(); ++i) {
        if ((LibTab)cache.tab[i] != tab) continue;
        names.push_back(cache.names[i]);
        cs_counts.push_back(cache.custom_counts[i]);
    }
    if (names.empty()) {
        ImGui::TextDisabled("(no %s systems yet)", kLibTabName[(int)tab]);
        return;
    }

    // Delete confirmation flow: row buttons only set pending_delete + a
    // one-shot open flag; OpenPopup/BeginPopupModal are called at a stable
    // ID-stack level below the table so the popup id stays consistent.
    // По копии на вкладку: все рисуются в одном кадре, общий флаг открыл бы
    // окно подтверждения не в той вкладке.
    static std::string pending_delete_k[kLibTabCount];
    static bool        want_open_confirm_k[kLibTabCount] = {};
    std::string& pending_delete    = pending_delete_k[(int)tab];
    bool&        want_open_confirm = want_open_confirm_k[(int)tab];

    // Visibility mask for the top-bar system picker. Stored as HIDDEN names
    // (AppModel::hidden_systems), so a newly added system shows up ticked.
    ImGui::TextDisabled("Tick a system to keep it in the top-bar picker.");
    ImGui::SameLine();
    if (ImGui::SmallButton("Show all")) {
        for (const auto& n : names) AppModel::set_name_hidden(model.hidden_systems, n, false);
        persist_hidden_systems(model);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Hide all")) {
        for (const auto& n : names) AppModel::set_name_hidden(model.hidden_systems, n, true);
        persist_hidden_systems(model);
    }

    // Name column auto-sized to the longest (name [+ "[N custom]" badge]),
    // instead of proportional stretch, so short names don't waste space.
    float name_col_w = ImGui::CalcTextSize("Name").x;
    for (size_t i = 0; i < names.size(); ++i) {
        float w = ImGui::CalcTextSize(names[i].c_str()).x;
        if (cs_counts[i] > 0) {
            char badge[32];
            std::snprintf(badge, sizeof(badge), " [%d custom]", cs_counts[i]);
            w += ImGui::GetStyle().ItemSpacing.x + ImGui::CalcTextSize(badge).x;
        }
        if (w > name_col_w) name_col_w = w;
    }
    name_col_w += ImGui::GetStyle().CellPadding.x * 2.0f + 8.0f;

    if (ImGui::BeginTable("libtbl", 5,
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit |
        ImGuiTableFlags_NoHostExtendX)) {
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, name_col_w);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("",     ImGuiTableColumnFlags_WidthFixed);

        for (size_t row = 0; row < names.size(); ++row) {
            const std::string& n = names[row];
            ImGui::TableNextRow();
            ImGui::PushID(n.c_str());

            ImGui::TableSetColumnIndex(0);
            bool shown_in_picker = !model.system_hidden(n);
            if (ImGui::Checkbox("##vis", &shown_in_picker)) {
                AppModel::set_name_hidden(model.hidden_systems, n, !shown_in_picker);
                persist_hidden_systems(model);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Show this system in the top-bar picker");

            ImGui::TableSetColumnIndex(1);
            bool selected = (model.library_selected_name == n);
            if (ImGui::Selectable(n.c_str(), selected)) {
                model.library_selected_name = n;
            }
            if (cs_counts[row] > 0) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.6f, 0.85f, 1.0f, 1.0f),
                    " [%d custom]", cs_counts[row]);
            }

            ImGui::TableSetColumnIndex(2);
            ImGui::BeginDisabled(other_tab_editing);
            const bool edit_clicked = ImGui::SmallButton("Edit");
            ImGui::EndDisabled();
            if (edit_clicked) {
                try {
                    model.library_edit_buffer->from_record(lib.load(n));
                    model.edit_original_name = n;
                    model.edit_error.clear();
                    model.library_edit_mode = AppModel::LibraryEditMode::EditExisting;
                }
                catch (const std::exception& e) { model.error_message = e.what(); }
            }

            ImGui::TableSetColumnIndex(3);
            if (ImGui::SmallButton("Duplicate")) {
                try { lib.duplicate(n); invalidate_library_cache(); }
                catch (const std::exception& e) { model.error_message = e.what(); }
            }

            ImGui::TableSetColumnIndex(4);
            if (ImGui::SmallButton("Delete")) {
                pending_delete = n;
                want_open_confirm = true;
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    if (want_open_confirm) {
        ImGui::OpenPopup("Delete system?");
        want_open_confirm = false;
    }

    if (ImGui::BeginPopupModal("Delete system?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Delete \"%s\"?", pending_delete.c_str());
        ImGui::TextDisabled("The system folder (including sessions/) will be removed.");
        ImGui::Separator();
        if (ImGui::Button("Delete", ImVec2(120, 0))) {
            lib.remove(pending_delete);
            invalidate_library_cache();
            // Drop the stale tick so a system re-added under the same name
            // doesn't come back hidden.
            if (model.system_hidden(pending_delete)) {
                AppModel::set_name_hidden(model.hidden_systems, pending_delete, false);
                persist_hidden_systems(model);
            }
            if (model.library_selected_name == pending_delete)
                model.library_selected_name.clear();
            pending_delete.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            pending_delete.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Preview panel for the selected row: note + read-only description, so a
    // system can be read and copied without the editor. Cached by name to
    // avoid a disk read every frame.
    ImGui::Separator();
    if (model.library_selected_name.empty()
        || std::find(names.begin(), names.end(), model.library_selected_name) == names.end()) {
        ImGui::TextDisabled("Click a row to preview and copy its description.");
        return;
    }

    // Запись читается один раз на выбранную систему и живёт в общем кэше
    // (см. LibraryListCache) — сохранение в редакторе его сбрасывает,
    // поэтому отредактированный текст показывается сразу.
    if (g_library_cache.sel_name != model.library_selected_name) {
        try {
            g_library_cache.sel_rec  = lib.load(model.library_selected_name);
            g_library_cache.sel_desc = describe_record(g_library_cache.sel_rec);
            g_library_cache.sel_ok   = true;
        }
        catch (...) {
            g_library_cache.sel_rec  = SystemRecord{};
            g_library_cache.sel_desc.clear();
            g_library_cache.sel_ok   = false;
        }
        g_library_cache.sel_name = model.library_selected_name;
    }

    ImGui::Text("Selected: %s", model.library_selected_name.c_str());
    if (!g_library_cache.sel_ok) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "(cannot read system.json)");
        return;
    }

    const SystemRecord& sel = g_library_cache.sel_rec;
    if (sel.note.empty()) ImGui::TextDisabled("(no note)");
    else                  ImGui::TextWrapped("%s", sel.note.c_str());

    ImGui::Spacing();
    if (ImGui::Button("Copy description")) ImGui::SetClipboardText(g_library_cache.sel_desc.c_str());
    if (!sel.latex_text.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Copy LaTeX")) ImGui::SetClipboardText(sel.latex_text.c_str());
    }
    if (!sel.plain_text.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Copy plain")) ImGui::SetClipboardText(sel.plain_text.c_str());
    }
    if (!sel.note.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Copy note")) ImGui::SetClipboardText(sel.note.c_str());
    }

    // Read-only, so text can also be selected by hand (Ctrl+C) without the
    // editor; ReadOnly keeps the widget from writing back into the cache.
    ImGui::InputTextMultiline("##libdesc", g_library_cache.sel_desc.data(),
                              g_library_cache.sel_desc.size() + 1, ImVec2(-1, 180),
                              ImGuiInputTextFlags_ReadOnly);
}

// Editor state: name/note header + System/Parameters sub-tabs + Save/Cancel.
// Edits go into model.library_edit_buffer, not model itself — Cancel simply
// drops the edit_mode without needing to restore anything on the live model.
void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb) {
    AppModel& buf = *model.library_edit_buffer;
    const bool add_new = (model.library_edit_mode == AppModel::LibraryEditMode::AddNew);
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                       add_new ? "New system" : "Editing");
    if (!add_new) {
        ImGui::SameLine();
        ImGui::TextDisabled("(original: %s)", model.edit_original_name.c_str());
    }

    ImGui::Spacing();
    ImGui::Text("Name:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    InputTextStr("##editor_name", buf.name);
    ImGui::Text("Note:");
    InputTextMultilineStr("##editor_note", buf.note, ImVec2(-1, 60));

    if (!model.edit_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f),
                           "%s", model.edit_error.c_str());
    }

    ImGui::Separator();

    // Редактор рисует System tab по буферу-черновику, а маска схем глобальная
    // и живёт в основной модели — иначе в редакторе показались бы скрытые схемы.
    buf.hidden_schemes = model.hidden_schemes;
    if (buf.is_dae) {
        ImGui::TextWrapped("x' = f(x, z, t),  0 = g(x, z, t). One equation per line: "
                           "x' = ... for each differential variable, 0 = ... for each "
                           "algebraic one. min(a, b) / max(a, b) are available.");
        // Источник — схема: SPICE-netlist или .kicad_sch (через kicad-cli). Импорт перезаписывает
        // уравнения, переменные, параметры (номиналы) и НУ ниже.
        {
            AppModel::DaeEditorState& cst = model.dae_editor;
            ImGui::Text("Circuit:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(-ImGui::GetFontSize() * 15.0f);
            InputTextStr("##dae_circuit", buf.circuit_source);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("SPICE netlist (.cir, .net, .sp) or KiCad schematic (.kicad_sch, needs kicad-cli\n"
                                  "from KiCad 7+). Empty: the equations below are typed by hand.");
            ImGui::SameLine();
            if (ImGui::Button("Open...##dae_circuit") && cb.pick_open_file_circuit) {
                const std::string p = cb.pick_open_file_circuit();
                if (!p.empty()) { buf.circuit_source = p; cst.circuit_import_now = true; }
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(buf.circuit_source.empty());
            if (ImGui::Button("Re-read##dae_circuit")) cst.circuit_import_now = true;
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("Watch##dae_circuit", &cst.circuit_watch);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Re-import when the file is saved (checked once a second).");
            // ОУ: как есть (их .subckt из netlist'а) или идеальные; копия на идеальных — отдельной записью.
            if (ImGui::Checkbox("Ideal op-amps##dae_circuit", &buf.circuit_ideal_opamps) && !buf.circuit_source.empty())
                cst.circuit_import_now = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Off: op-amps (TL072, LM741, ...) are imported as is - their .subckt models\n"
                                  "from the netlist, element by element. On: each is an ideal op-amp (nullor).\n"
                                  "An op-amp without a .subckt in the netlist is ideal either way.");
            if (!buf.circuit_source.empty() && !buf.circuit_ideal_opamps) {
                ImGui::SameLine();
                if (ImGui::Button("Save ideal copy##dae_circuit")) {
                    AppModel copy([](const std::vector<unsigned char>&) { return std::string{}; });
                    copy.from_record(buf.to_record());
                    copy.circuit_ideal_opamps = true;
                    copy.name = (buf.name.empty() ? std::string("Circuit") : buf.name) + " (ideal op-amps)";
                    std::string rep, e;
                    if (copy.import_circuit(&rep, &e)) {
                        try {
                            const std::string saved = lib.save(copy.to_record());
                            invalidate_library_cache();
                            cst.circuit_report = "saved the copy on ideal op-amps as '" + saved + "'\n" + rep;
                            cst.circuit_error.clear();
                        } catch (const std::exception& ex) { cst.circuit_error = std::string("Save ideal copy: ") + ex.what(); }
                    } else cst.circuit_error = "Ideal copy: " + e;
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Save the same circuit with ideal op-amps as a new library system\n"
                                      "(name + \" (ideal op-amps)\"), next to this one with the real models.");
            }
            if (!buf.circuit_source.empty() && cst.circuit_watch) {
                const double now = ImGui::GetTime();
                if (now - cst.circuit_poll_t > 1.0) {
                    cst.circuit_poll_t = now;
                    const long long mt = circuit_source_mtime(buf.circuit_source);
                    if (cst.circuit_path != buf.circuit_source) {   // новый путь: только запомнить время
                        cst.circuit_path = buf.circuit_source;
                        cst.circuit_mtime = mt;
                    } else if (mt >= 0 && mt != cst.circuit_mtime) cst.circuit_import_now = true;
                }
            }
            if (cst.circuit_import_now && !buf.circuit_source.empty()) {
                cst.circuit_import_now = false;
                std::string rep, e;
                if (buf.import_circuit(&rep, &e)) { cst.circuit_report = rep; cst.circuit_error.clear(); }
                else cst.circuit_error = "Import failed: " + e;
                cst.circuit_path = buf.circuit_source;
                cst.circuit_mtime = circuit_source_mtime(buf.circuit_source);
            }
            if (ImGui::Button("DC operating point##dae")) {
                std::string rep, e;
                if (buf.dae_operating_point(&rep, &e)) { cst.circuit_report = rep; cst.circuit_error.clear(); }
                else cst.circuit_error = "DC operating point: " + e;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Initial values from x' = 0, g = 0 (for a circuit: capacitor currents and inductor\n"
                                  "voltages zero). Newton starts from the current initial values.");
            if (!cst.circuit_error.empty() || !cst.circuit_report.empty()) {
                if (!cst.circuit_error.empty())
                    ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", cst.circuit_error.c_str());
                if (!cst.circuit_report.empty() && ImGui::TreeNode("Report##dae_circuit")) {
                    ImGui::TextWrapped("%s", cst.circuit_report.c_str());
                    ImGui::TreePop();
                }
            }
        }
        ImGui::Text("Equations:");
        InputTextMultilineStr("##dae_eqs", buf.plain_text, ImVec2(-1, 160));
        // Вспомогательные функции — как у ОДУ: подставляются в уравнения при разборе.
        ImGui::Checkbox("Use auxiliary functions##dae", &buf.use_aux_funcs);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("One definition per line, e.g. G(x) = Goff + (Gon - Goff)*x.\n"
                              "A call is replaced by the body with the arguments substituted;\n"
                              "a function may call another one defined above or below.");
        if (buf.use_aux_funcs)
            InputTextMultilineStr("##dae_funcs", buf.func_defs_text, ImVec2(-1, 110));
        ImGui::Text("Differential variables (x):"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_vars", buf.vars_text);
        ImGui::Text("Algebraic variables (z):"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_alg", buf.alg_text);
        ImGui::Text("Parameters:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_params", buf.params_text);
        ImGui::Text("Variable indices:"); ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        InputTextStr("##dae_index", buf.dae_index_text);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Optional, e.g. \"x=1, u=2, lambda=3\" (as nind1/2/3 in RADAU5).\n"
                              "Unnamed: differential 1, algebraic 1 if structurally index 1.");

        // Разбор — только при смене ввода.
        AppModel::DaeEditorState& st = model.dae_editor;
        const std::string key = buf.plain_text + '\x1e' + buf.vars_text + '\x1e' + buf.alg_text
                              + '\x1e' + buf.params_text + '\x1e' + buf.dae_index_text
                              + '\x1e' + (buf.use_aux_funcs ? buf.func_defs_text : std::string());
        if (st.key != key) {
            st.key = key;
            try {
                FuncDefs funcs;
                if (buf.use_aux_funcs && !buf.func_defs_text.empty()) funcs = parse_func_defs(buf.func_defs_text);
                const DaeSystem d = parse_dae(buf.plain_text, buf.vars_text, buf.alg_text,
                                              buf.params_text, buf.dae_index_text, funcs);
                st.ok = true;
                st.vars = d.diff;
                st.vars.insert(st.vars.end(), d.alg.begin(), d.alg.end());
                st.params = d.params;
                std::string m = std::to_string(d.diff.size()) + " differential, "
                              + std::to_string(d.alg.size()) + " algebraic";
                if (d.structural_index_hint == 1) m += "; structurally index 1";
                else if (d.structural_index_hint == 2) {
                    m += "; index >= 2 (no constraint contains";
                    for (const auto& z : d.unmatched_alg) m += " " + z;
                    m += ")";
                }
                if (d.uses_t) m += "; uses t";
                m += "\nindices:";
                for (size_t i = 0; i < d.n_vars(); ++i)
                    m += " " + st.vars[i] + "=" + (d.index[i] ? std::to_string(d.index[i]) : "?");
                st.message = m;
                // Превью: \dot{x} = f по дифференциальным, 0 = g по связям — тем же AST, что у кодогена.
                {
                    const System ps = d.as_codegen_system();
                    st.latex.clear();
                    st.latex_lines = 0;
                    std::vector<std::string> fnames;
                    for (const auto& kv : funcs) fnames.push_back(kv.first);
                    // Сами вспомогательные функции — первыми строками (в уравнениях они уже подставлены);
                    // тело может звать другие функции (iM = G(x)*v + iS(v)) — они печатаются вызовами.
                    for (const auto& kv : funcs) {
                        std::string args;
                        for (size_t k = 0; k < kv.second.args.size(); ++k)
                            args += (k ? ", " : "") + name_to_latex(kv.second.args[k]);
                        std::string body;
                        try { body = expr_to_latex(kv.second.body, false, fnames); }
                        catch (const std::exception&) { body = "?"; }
                        st.latex += "\\mathrm{" + kv.first + "}\\left(" + args + "\\right) = " + body + "\n";
                        ++st.latex_lines;
                    }
                    // Уравнения: с функциями — исходные строки (вызовы G(xm) видны как есть), без них —
                    // разобранные f и g. Строка, которая не прочлась, берётся из разбора (с подстановкой).
                    std::vector<std::string> raw_rows;
                    if (!fnames.empty()) {
                        std::istringstream in(buf.plain_text);
                        std::string row;
                        while (std::getline(in, row)) {
                            const size_t a = row.find_first_not_of(" \t\r");
                            if (a == std::string::npos || row[a] == '#' || row[a] == '%' || row.compare(a, 2, "//") == 0) continue;
                            raw_rows.push_back(row.substr(a));
                        }
                    }
                    const bool use_raw = !fnames.empty() && raw_rows.size() == ps.rhs.size();
                    for (size_t i = 0; i < ps.rhs.size(); ++i) {
                        const bool diff = (int)i < ps.dae_nx;
                        std::string ln;
                        std::string lhs = diff ? "\\dot{" + name_to_latex(ps.vars[i]) + "} = " : std::string("0 = ");
                        bool done = false;
                        if (use_raw) {
                            const size_t eq = raw_rows[i].find('=');
                            if (eq != std::string::npos) {
                                std::string l = raw_rows[i].substr(0, eq);
                                l.erase(std::remove_if(l.begin(), l.end(), [](char c) { return c == ' ' || c == '\t'; }), l.end());
                                try {
                                    ln = expr_to_latex(raw_rows[i].substr(eq + 1), false, fnames);
                                    if (l != "0") {
                                        const size_t q = l.find_first_of("'(");
                                        lhs = "\\dot{" + name_to_latex(l.substr(0, q)) + "} = ";
                                    }
                                    done = true;
                                } catch (const std::exception&) {}
                            }
                        }
                        if (!done) {
                            try { ln = expr_to_latex(ps.rhs[i], ps.latex); }
                            catch (const std::exception&) { ln = "?"; }
                        }
                        st.latex += lhs + ln + "\n";
                        ++st.latex_lines;
                    }
                }
                // Уровень A: можно ли решать как ОДУ (z выражены из связей явно).
                System ode;
                std::vector<std::string> zex;
                std::string why;
                st.level_c = false;
                const System dsys = d.as_codegen_system();
                st.reducible = dae_reduce_to_ode(dsys, ode, &zex, &why);
                if (st.reducible) {
                    st.reduce_msg = "explicit z(x):";
                    for (size_t j = 0; j < d.alg.size(); ++j) st.reduce_msg += "\n  " + d.alg[j] + " = " + zex[j];
                } else if (dae_index1_structural(dsys)) {
                    // Уровень B: z из связей Ньютоном в каждом вычислении f.
                    st.reducible = true;
                    st.reduce_msg = "z by Newton in every f (level B): " + why
                                  + "\n  schemes: explicit RK / DOPRI / GBS, Extr/Comp over them, adaptive";
                } else {
                    // Уровень C (индекс 2-3): связи дифференцируются до появления z и стабилизируются.
                    System s1;
                    std::string w2;
                    const int idx = dae_structural_index(dsys);
                    if (idx >= 2 && dae_stabilized_index1(dsys, 1.0, s1, &w2)) {
                        System o2;
                        std::vector<std::string> z2;
                        std::string w3;
                        const bool lev_a = dae_reduce_to_ode(s1, o2, &z2, &w3);
                        st.reducible = true;
                        st.level_c = true;
                        st.reduce_msg = "level C (index " + std::to_string(idx) + "): constraints differentiated until z "
                                        "appears, (d/dt + alpha)^m g = 0;\n  then " + std::string(lev_a ? "explicit z (level A)"
                                        : "z by Newton (level B)") + ". The original constraint is only held by the "
                                        "stabilization (drift ~ e^(-alpha t)).";
                    } else st.reduce_msg = why + (w2.empty() ? std::string() : "; " + w2);
                }
            } catch (const std::exception& e) {
                st.ok = false;
                st.message = e.what();
                st.vars.clear();
                st.params.clear();
                st.reducible = false;
                st.level_c = false;
                st.reduce_msg.clear();
                st.latex.clear();
                st.latex_lines = 0;
            }
        }
        if (st.ok) ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f), "%s", st.message.c_str());
        else       ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", st.message.c_str());
        if (st.ok && st.latex_lines > 0) {
            ImGui::TextDisabled("Equations (LaTeX preview):");
            const float row = std::floor(ImGui::GetFontSize() * 1.3f + 0.5f) * 1.9f;
            draw_latex_preview("##dae_latex", st.latex,
                               ImVec2(-1, std::min(row * (float)st.latex_lines + row * 0.5f, row * 12.0f)));
            if (ImGui::BeginPopupContextItem("##dae_latex_ctx")) {
                if (ImGui::MenuItem("Copy LaTeX") && cb.set_clipboard_text) cb.set_clipboard_text(st.latex);
                ImGui::EndPopup();
            }
        }

        // Этап 3, уровень A: ДАУ -> ОДУ подстановкой z. Снять галочку можно всегда;
        // поставить — только если сведение удаётся.
        if (st.ok) {
            ImGui::BeginDisabled(!st.reducible && !buf.dae_as_ode);
            ImGui::Checkbox("Solve as ODE", &buf.dae_as_ode);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("If z enters its constraint linearly, z(x) is substituted\n"
                                  "into f (level A): an ODE in x for every tab and every ODE scheme.\n"
                                  "Otherwise z is found by Newton from g(x, z) = 0 before every f\n"
                                  "(level B): explicit RK / DOPRI / GBS, Extr/Comp over them, adaptive.\n"
                                  "Index 2-3 (level C): constraints are differentiated until z appears\n"
                                  "and stabilized, then level A or B.\n"
                                  "Either way z stays in the state after x, so bifurcation output,\n"
                                  "metrics and plot axes can use it; LLE/LS perturb x only.");
            ImGui::TextDisabled("%s", st.reduce_msg.c_str());
            if (buf.dae_as_ode && st.level_c) {
                InputNumStr("stabilization alpha##daestab", buf.dae_stab_alpha, 140.0f);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Level C replaces each index-(m+1) constraint by (d/dt + alpha)^m g = 0:\n"
                                      "the drift of g decays like e^(-alpha t). Larger alpha holds g tighter\n"
                                      "but makes the ODE stiffer (explicit schemes need h*alpha small).");
            }
            if (buf.dae_as_ode && !st.reducible)
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "cannot solve as ODE: untick");
        }

        // Начальные условия всех переменных (z тоже: согласовать их — дело
        // решателя, но стартовое приближение Ньютону даёт пользователь) и
        // значения параметров.
        if (st.ok && ImGui::BeginTable("##dae_vals", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Initial conditions");
            for (const auto& v : st.vars)
                InputNumStr((v + "##daeic").c_str(), buf.init_conditions[v], 140.0f, {}, true);
            ImGui::TableNextColumn();
            ImGui::TextDisabled("Parameter values");
            for (const auto& p : st.params)
                InputNumStr((p + "##daepv").c_str(), buf.param_values[p], 140.0f, {}, true);
            InputNumStr("h##daeh", buf.step_h, 140.0f, {}, true);
            // Ньютон схем ДАУ — значения по умолчанию для вкладок (как у неявных схем ОДУ).
            ImGui::TextDisabled("Newton (defaults for the tabs)");
            InputNumStr("tol##daent", buf.newton_tol, 140.0f);
            InputNumStr("max iters##daeni", buf.newton_max_iters, 140.0f);
            ImGui::Checkbox("full Newton##daenf", &buf.newton_full);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Rebuild the Jacobian every iteration. A step that runs out of\n"
                                  "iterations is accepted unconverged and the constraint drifts.");
            ImGui::EndTable();
        }
        // Методы: при «Solve as ODE» система — ОДУ (схемы ОДУ, флаги scheme_*), иначе — методы ДАУ.
        ImGui::Separator();
        if (buf.dae_as_ode) draw_builtin_scheme_checkboxes(buf);
        else                draw_dae_method_checkboxes(buf);
        ImGui::Spacing();
        draw_generated_code_block(buf, cb);
    }
    else if (ImGui::BeginTabBar("editor_tabs")) {
        if (ImGui::BeginTabItem("System")) {
            draw_system_tab(buf, cb);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Parameters")) {
            draw_parameters_tab(buf);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();

    const bool save_clicked = ImGui::Button("Save", ImVec2(120, 0));
    const bool save_shortcut = ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false);
    if (save_clicked || save_shortcut) {
        library_editor_save(model, lib);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
        model.edit_error.clear();
        model.library_edit_mode = AppModel::LibraryEditMode::None;
    }
}

// Ось фазовой проекции: переменная (0..nv-1) или наблюдаемая вне состояния (nv + j, ДАУ,
// сведённая к ОДУ — значения дописаны к точкам после координат состояния).
std::string phase_axis_name(const PhaseAnalysisSession& s, int ax) {
    const int nv = (int)s.vars.size();
    if (ax >= 0 && ax < nv) return s.vars[(size_t)ax];
    if (ax >= nv && ax - nv < (int)s.observables.size()) return s.observables[(size_t)(ax - nv)];
    return s.vars.empty() ? std::string("-") : s.vars[0];
}
void phase_axis_combo(const char* id, PhaseAnalysisSession& s, int& axis) {
    if (!ImGui::BeginCombo(id, phase_axis_name(s, axis).c_str())) return;
    for (int k = 0; k < (int)s.vars.size(); ++k)
        if (ImGui::Selectable(s.vars[(size_t)k].c_str(), axis == k)) { axis = k; s.fit_request = true; }
    if (!s.observables.empty()) {
        ImGui::SeparatorText("z (observables)");
        const int nv = (int)s.vars.size();
        for (int j = 0; j < (int)s.observables.size(); ++j)
            if (ImGui::Selectable(s.observables[(size_t)j].c_str(), axis == nv + j)) { axis = nv + j; s.fit_request = true; }
    }
    ImGui::EndCombo();
}
