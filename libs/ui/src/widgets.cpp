#include "gui_internal.h"
using namespace ucuda::ui_detail;


namespace ucuda::ui_detail {
void draw_generated_code_block(AppModel& model, const GuiCallbacks& cb) {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void draw_wrapper_coefficients(const std::string& name,
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
}


namespace ucuda::ui_detail {
void begin_coeff_column(const char* id, int rows, int max_rows) {
    const float row_h = ImGui::GetFrameHeightWithSpacing();
    const int   vis   = rows < max_rows ? rows : max_rows;
    ImGui::BeginChild(id, ImVec2(0.0f, row_h * (float)(vis < 1 ? 1 : vis)
                                       + ImGui::GetStyle().FramePadding.y * 2.0f), true);
}
}


namespace ucuda::ui_detail {
std::string wrapper_name_field(const char* id, std::string& text,
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
}


namespace ucuda::ui_detail {
bool extr_base_needs_half_s(const std::string& nm) {
    return nm == "CD" || nm == "CD fast" || nm == "CD FP" || nm == "CD fast FP"
        || nm == "Complex CD" || nm == "Complex CD4"
        || nm == "CCD4 (o4s3)" || nm == "CCD4 (o4s4)"
        || nm == "CD_{1-0}" || nm == "Complex CD_{1-0}" || nm == "Complex CD4_{1-0}"
        || nm == "CCD4 (o4s3)_{1-0}" || nm == "CCD4 (o4s4)_{1-0}"
        || nm == "SEMP" || nm == "SIMP";
}
}


namespace ucuda::ui_detail {
void draw_extrapolation_builder(AppModel& model) {
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
}


namespace ucuda::ui_detail {
void comp_load_coeffs_from_file(AppModel& model, const std::string& path) {
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
}


namespace ucuda::ui_detail {
void draw_composition_builder(AppModel& model, const GuiCallbacks& cb) {
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
}


namespace ucuda::ui_detail {
void draw_wrapper_list(AppModel& model) {
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
}


namespace ucuda::ui_detail {
void draw_system_tab(AppModel& model, const GuiCallbacks& cb) {

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
}


namespace ucuda::ui_detail {
void draw_parameters_tab(AppModel& model) {
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
        if (ImGui::Button("Groups...##sys_icg")) { model.pg_ui.open = true; model.pg_ui.kind_req = 1; }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Group and label the initial-condition fields (layout only)");
        if (ImGui::BeginTable("ics", 2, ImGuiTableFlags_SizingFixedFit)) {
            draw_param_tree(&model.ic_groups, model.ic_fields(model.known_vars), [&](const std::string& v) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); draw_field_name(v, &model.field_labels);
                ImGui::TableSetColumnIndex(1);
                { std::string id = "##ic_" + v; InputNumStr(id.c_str(), model.init_conditions[v], 140, [&] {
                    field_restore_popup("##ic_restore", &model, BroadcastField::InitCondition, v,
                                        model.init_conditions, model.known_vars);
                }); }
            }, true);
            ImGui::EndTable();
        }
    }

    ImGui::Spacing();

    // параметры: значение
    if (!model.known_params.empty()) {
        ImGui::SeparatorText("Parameters (value)");
        if (ImGui::Button("Groups...##sys_pg")) { model.pg_ui.open = true; model.pg_ui.kind_req = 0; }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Group and label the parameter fields (layout only)");
        if (ImGui::BeginTable("params", 2, ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("value");
            ImGui::TableHeadersRow();
            draw_param_tree(&model.param_groups, model.known_params, [&](const std::string& p) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); draw_field_name(p, &model.field_labels);
                ImGui::TableSetColumnIndex(1);
                { std::string id = "##val_" + p; InputNumStr(id.c_str(), model.param_values[p], 100, [&] {
                    field_restore_popup("##val_restore", &model, BroadcastField::Param, p,
                                        model.param_values, model.known_params);
                }); }
            }, true);
            ImGui::EndTable();
        }
        draw_param_groups_window(model, model.known_params, model.known_vars, "sys");
        ImGui::TextDisabled("Empty fields are left unset.");
    }

    if (model.known_vars.empty() && model.known_params.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No symbols yet. Enter a system and press 'Refresh from system'.");
    }
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string trim_copy(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
}


namespace ucuda::ui_detail {
void invalidate_library_cache() {
    g_library_cache.valid = false;
    g_library_cache.sel_name.clear();
    g_library_cache.sel_desc.clear();
    g_library_cache.sel_ok = false;
}
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
void persist_hidden_systems(const AppModel& model) {
    AppConfig cfg;
    load_app_config(get_exe_dir_with_sep(), cfg);
    cfg.hidden_systems = model.hidden_systems;
    save_app_config(get_exe_dir_with_sep(), cfg);
}
}


namespace ucuda::ui_detail {
std::string describe_record(const SystemRecord& r) {
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
}


namespace ucuda::ui_detail {
void library_editor_save(AppModel& model, SystemLibrary& lib) {
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
}


namespace ucuda::ui_detail {
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
}


namespace ucuda::ui_detail {
extern std::string exe_dir();
}


namespace ucuda::ui_detail {
std::string get_exe_dir_with_sep() {
    std::string d = platform::executable_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}
}


namespace ucuda::ui_detail {
ImVec4 ic_base_color(int ic_index) {
    return ImPlot::GetColormapColor(ic_index);
}
}


namespace ucuda::ui_detail {
ImVec4 shade_of(ImVec4 base, int vi, int nv) {
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}
}


namespace ucuda::ui_detail {
ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    return shade_of(ic_base_color(ic_index), vi, nv);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars) {
    if (vars.size() >= 3) return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
    if (vars.size() == 2) return vars[0] + " + pi*" + vars[1];
    return vars.empty() ? std::string("x") : vars[0];
}
}


namespace ucuda::ui_detail {
static_assert(::pi == static_cast<numb>(num_parse_detail::kPi),
              "pi in num_parse.h diverged from configCUDA.h");
}


namespace ucuda::ui_detail {
[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars) {
    double v = !pt.empty() ? pt[0] : 0.0;
    if (nvars >= 2 && pt.size() > 1) v += ::pi    * pt[1];
    if (nvars >= 3 && pt.size() > 2) v += ::euler * pt[2];
    return v;
}
}


namespace ucuda::ui_detail {
std::vector<char>& input_scratch(const std::string& str, size_t extra) {
    static thread_local std::vector<char> buf;
    buf.assign(str.begin(), str.end());
    buf.resize(str.size() + extra);
    buf[str.size()] = '\0';
    return buf;
}
}


namespace ucuda::ui_detail {
int input_str_resize(ImGuiInputTextCallbackData* d) {
    if (d->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string* s = static_cast<std::string*>(d->UserData);
        s->resize((size_t)d->BufTextLen);
        d->Buf = s->data();
    }
    return 0;
}
}


namespace ucuda::ui_detail {
bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    return ImGui::InputTextMultiline(label, str.data(), str.capacity() + 1, size,
                                     ImGuiInputTextFlags_CallbackResize, input_str_resize, &str);
}
}


namespace ucuda::ui_detail {
void draw_resize_handle(const char* id, float& height,
                               float min_h, float max_h) {
    // InvisibleButton refuses non-positive dimensions — resolve -1 to the actual width.
    const float w = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton(id, ImVec2(w > 1.0f ? w : 1.0f, 8.0f));
    const bool active  = ImGui::IsItemActive();
    const bool hovered = ImGui::IsItemHovered();
    if (active) {
        height += ImGui::GetIO().MouseDelta.y;
        if (height < min_h) height = min_h;
        if (height > max_h) height = max_h;
    }
    if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    const ImVec2 pmin = ImGui::GetItemRectMin();
    const ImVec2 pmax = ImGui::GetItemRectMax();
    const ImU32 col = ImGui::GetColorU32(active  ? ImGuiCol_SeparatorActive
                                       : hovered ? ImGuiCol_SeparatorHovered
                                                 : ImGuiCol_Separator);
    const float y = (pmin.y + pmax.y) * 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(pmin.x, y), ImVec2(pmax.x, y), col, 3.0f);
}
}


namespace ucuda::ui_detail {
void draw_latex_preview(const char* id, const std::string& src, ImVec2 size) {
    std::vector<std::string> lines;
    std::string cur;
    auto push = [&] {
        const size_t a = cur.find_first_not_of(" \t\r");
        if (a != std::string::npos) lines.push_back(cur.substr(a));
        cur.clear();
    };
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '\n') { push(); continue; }
        if (src[i] == '\\' && i + 1 < src.size() && src[i + 1] == '\\') { push(); ++i; continue; }
        cur += src[i];
    }
    push();

    ImGui::BeginChild(id, size, ImGuiChildFlags_Borders | ImGuiChildFlags_FrameStyle,
                      ImGuiWindowFlags_HorizontalScrollbar);
    const float fs  = std::floor(ImGui::GetFontSize() * 1.3f + 0.5f);
    const ImU32 col = ImGui::GetColorU32(ImGuiCol_Text);
    ImDrawList* dl  = ImGui::GetWindowDrawList();
    bool any = false;
    for (const std::string& ln : lines) {
        const MathExtent ext = math_formula_extent(ln.c_str(), fs);
        if (ext.width <= 0.0f) continue;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(ext.width, ext.ascent + ext.descent + fs * 0.2f));
        if (ImGui::IsItemVisible()) math_formula_draw(dl, p, col, ln.c_str(), fs);
        any = true;
    }
    if (!any) ImGui::TextDisabled("(formula preview)");
    ImGui::EndChild();
}
}


namespace ucuda::ui_detail {
void draw_rhs_preview(const AppModel& model) {
    const std::string& src =
        (model.mode == InputMode::Plain) ? model.plain_text : model.latex_text;
    if (src.find_first_not_of(" \t\r\n") == std::string::npos) return;
    if (!ImGui::CollapsingHeader("Kernel form: X[..], a[..]", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    const AppModel::RhsPreview& pv = model.rhs_preview();
    if (pv.text.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "Parse error: %s", pv.error.c_str());
        return;
    }
    // Длинная правая часть переносится, а не уезжает за край, поэтому высоту
    // меряем с тем же переносом (ширина поля минус отступы и полоса прокрутки).
    const ImGuiStyle& st = ImGui::GetStyle();
    const float wrap_w = ImGui::GetContentRegionAvail().x - st.FramePadding.x * 2.0f
                       - st.ScrollbarSize;
    const float text_h = ImGui::CalcTextSize(pv.text.c_str(), nullptr, false,
                                             std::max(wrap_w, 50.0f)).y;
    const float h = std::min(text_h + st.FramePadding.y * 2.0f + 2.0f, 320.0f);
    ImGui::InputTextMultiline("##rhs_preview", (char*)pv.text.c_str(), pv.text.size() + 1,
                              ImVec2(-1, h),
                              ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
}
}


namespace ucuda::ui_detail {
bool InputTextStr(const char* label, std::string& str, float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    bool changed = ImGui::InputTextWithHint(label, hint, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;
    double v = 0.0;
    return parse_num_checked(s, v);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool is_custom_scheme(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    for (const auto& cs : custom_schemes) if (cs.name == scheme_name) return true;
    return false;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_needs_compiled_body(const std::string& scheme_name,
                              const std::vector<CustomScheme>& custom_schemes) {
    return is_custom_scheme(scheme_name, custom_schemes)
        || parse_extrapolation_name(scheme_name, nullptr)
        || parse_composition_name(scheme_name, nullptr);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool custom_scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    static const std::regex re(R"(a\s*\[\s*0\s*\])");
    for (const auto& cs : custom_schemes)
        if (cs.name == scheme_name) return std::regex_search(cs.body, re);
    return false;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_uses_symmetry(const std::string& scheme_name,
                                         const std::vector<CustomScheme>& custom_schemes) {
    // Экстраполяционная обёртка читает a[0] ровно тогда, когда его читает база:
    // её тело — это тело базы, повторённое по стадиям. Без этой развёртки выбор
    // "Extr(CD|1,2)" прятал бы поле s, которым обёрнутый CD пользуется.
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_symmetry(spec.base, custom_schemes);
    // Same for a composition: its body is the base repeated with scaled steps,
    // so hiding s under "Comp(CD|...)" would hide a knob the inner CD reads.
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_symmetry(cspec.base, custom_schemes);
    return scheme_name == "CD" || scheme_name == "CD fast"
        || scheme_name == "CD FP" || scheme_name == "CD fast FP"
        || scheme_name == "Complex CD" || scheme_name == "Complex CD4"
        || scheme_name == "CCD4 (o4s3)" || scheme_name == "CCD4 (o4s4)"
        || scheme_name == "CD_{1-0}" || scheme_name == "Complex CD_{1-0}"
        || scheme_name == "Complex CD4_{1-0}" || scheme_name == "CCD4 (o4s3)_{1-0}"
        || scheme_name == "CCD4 (o4s4)_{1-0}"
        || scheme_name == "Complex Implicit Euler"
        || scheme_name == "SEMP" || scheme_name == "SIMP"
        || custom_scheme_uses_symmetry(scheme_name, custom_schemes);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_uses_cd_fp_iters(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_cd_fp_iters(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_cd_fp_iters(cspec.base);
    return scheme_name == "CD FP" || scheme_name == "CD fast FP";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_uses_newton(const std::string& scheme_name) {
    ExtrapolationSpec spec;
    if (parse_extrapolation_name(scheme_name, &spec))
        return scheme_uses_newton(spec.base);
    CompositionSpec cspec;
    if (parse_composition_name(scheme_name, &cspec))
        return scheme_uses_newton(cspec.base);
    return scheme_name == "Implicit Euler" || scheme_name == "Implicit Midpoint"
        || scheme_name == "Complex Implicit Euler"
        || dae_scheme_from_name(scheme_name, nullptr)    // DAE BDF1 / Trapezoidal
        || scheme_is_multistep(scheme_name);             // BDF 1-5: корректор — тот же Ньютон
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h) {
    if (sweep_over_h) return "h";
    if (sweep_over_var) {
        if (var_idx >= 0 && var_idx < (int)vars.size()) return vars[var_idx] + " (IC)";
        return "var";
    }
    // -1 — цель "s": коэффициент симметрии живёт в a[0], а не в списке
    // параметров системы, поэтому имени для него в params[] нет.
    if (param_idx < 0) return "s";
    if (param_idx < (int)params.size()) return params[param_idx];
    return "param";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string label_identity(const std::string& s) {
    const size_t k = s.rfind(" [");
    return k == std::string::npos ? s : s.substr(0, k);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE"
                    : w.kind == ParametricPlotWindow::Kind::LS  ? "LS" : "Metrics";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    return sweep_range_label(auto_axis_name(params, vars, c.param_index, c.sweep_over_var,
                                            c.var_sweep_index, c.sweep_over_h),
                             c.param_lo_text, c.param_hi_text);
}
}


namespace ucuda::ui_detail {
void refresh_auto_labels(AppModel& model) {
    for (auto& bd : model.bifurcation_session.diagrams)
        if (!bd.label_is_manual)
            bd.label = auto_label_sweep(bd, model.bifurcation_session.params, model.bifurcation_session.vars);
    for (auto& c : model.lle_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.lle_session.params, model.lle_session.vars);
    for (auto& c : model.ls_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.ls_session.params, model.ls_session.vars);
    for (auto& c : model.metrics_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.metrics_session.params, model.metrics_session.vars);
    for (auto& c : model.dft1d_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_dft1d(c, model.dft1d_session.params, model.dft1d_session.vars);
    for (auto& w : model.parametric_plot_windows)
        if (!w.label_is_manual)
            w.label = auto_label_window(w);
    for (auto& w : model.dft1d_plot_windows)
        if (!w.label_is_manual)
            w.label = "DFT 1D";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) {
    const double a = parse_ratio_or(v, 0.0);
    const double b = parse_ratio_or(v, 1.0);
    if (a != b) return false;
    out = a;
    return true;
}
}


namespace ucuda::ui_detail {
bool InputNumStr(const char* label, std::string& str, float width,
                        const std::function<void()>& ctx_menu,
                        bool allow_empty) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    // CallbackHistory — ↑/↓ в активном InputText, обрабатываем в digit_step_input_callback.
    // CallbackCharFilter — прежняя замена запятой на точку, тоже в нём.
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
        digit_step_input_callback);
    if (changed) str = buf.data();
    if (ctx_menu) ctx_menu();

    // Inline-предупреждение, если содержимое не парсится как число.
    // Default из engine'а (0) всё равно применится, но пользователю
    // явно сигналим, что введённое значение игнорируется.
    if (!(allow_empty && str.empty()) && !is_numeric_string(str)) {
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, using default");
    }
    return changed;
}
}


namespace ucuda::ui_detail {
int list_digit_step_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory) return 0;
    int dir = 0;
    if (data->EventKey == ImGuiKey_UpArrow)   dir = +1;
    if (data->EventKey == ImGuiKey_DownArrow) dir = -1;
    if (dir == 0) return 0;
    const std::string text(data->Buf, data->Buf + data->BufTextLen);
    const int cur = std::clamp(data->CursorPos, 0, (int)text.size());
    int b = cur, e = cur;
    while (b > 0 && text[b - 1] != ',' && text[b - 1] != ';') --b;
    while (e < (int)text.size() && text[e] != ',' && text[e] != ';') ++e;
    while (b < e && text[b] == ' ') ++b;
    while (e > b && text[e - 1] == ' ') --e;
    if (b >= e) return 0;
    std::string new_tok;
    int new_cur = 0;
    if (!DigitInput::ComputeStep(text.substr(b, e - b), std::clamp(cur - b, 0, e - b), dir,
                                 new_tok, new_cur))
        return 0;
    data->DeleteChars(b, e - b);
    data->InsertChars(b, new_tok.c_str());
    const int pos = std::clamp(b + new_cur, 0, data->BufTextLen);
    data->CursorPos = data->SelectionStart = data->SelectionEnd = pos;
    return 0;
}
}


namespace ucuda::ui_detail {
bool InputNumListStr(const char* label, std::string& str, float width ) {
    std::vector<char>& buf = input_scratch(str, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    const bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackHistory, list_digit_step_callback);
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width) {
    std::vector<char>& buf = input_scratch(text, 1024);
    if (width > 0) ImGui::SetNextItemWidth(width);
    if (ImGui::InputText(label, buf.data(), buf.size(),
                         ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
                         digit_step_input_callback))
        text = buf.data();

    // Снимаем ДО возможного TextColored: тот станет last item, и IsItem* уже
    // спрашивал бы про него.
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    const bool valid     = !text.empty() && is_numeric_string(text);
    if (!valid)
        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
            "  invalid number, keeping previous value");

    if (committed && valid) {
        value = parse_ratio_or(text, value);
        return true;
    }
    return false;
}
}


namespace ucuda::ui_detail {
void do_undo(AppModel& m) {
    if (m.undo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void do_redo(AppModel& m) {
    if (m.redo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void draw_undo_menu_item(AppModel* m) {
    if (!m) return;
    const std::string* u = m->undo_top();
    const std::string* r = m->redo_top();
    if (!u && !r) return;
    ImGui::Separator();
    if (u) {
        const std::string lbl = "Undo: " + *u;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Z")) do_undo(*m);
    }
    if (r) {
        const std::string lbl = "Redo: " + *r;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Shift+Z")) do_redo(*m);
    }
}
}


namespace ucuda::ui_detail {
void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value,
                                 const std::function<void()>& extra) {
    if (!model) return;
    // ID popup'а обязан быть свой у каждого поля: в блоке Integration шесть
    // полей лежат на ОДНОМ уровне ID-стека, и общий идентификатор склеил бы
    // их меню в одно.
    const std::string popup_id = "##apply_all_" + label + name;
    if (!ImGui::BeginPopupContextItem(popup_id.c_str())) return;
    ImGui::TextDisabled("%s = %s", label.c_str(), value.c_str());
    ImGui::Separator();
    if (extra) {
        extra();
        ImGui::Separator();
    }

    // Сухой прогон: сколько конфигов реально РАЗОЙДЁТСЯ с этим значением.
    // Показываем числом, но НЕ гасим по нему команду: серый пункт читается как
    // подпись, и "0 differ" выглядит как сломанная рассылка, хотя на деле все
    // цели уже с этим значением.
    auto differ_count = [&](const std::vector<BroadcastTarget>* only) {
        return model->broadcast_field(f, name, value, /*apply*/false, label, only, "");
    };
    // Применяет и кладёт отчёт в шапку: адресаты бывают не открыты, и без
    // строки сверху "применилось" неотличимо от "ничего не произошло".
    auto apply_now = [&](const std::vector<BroadcastTarget>* only, const std::string& note) {
        const int n = model->broadcast_field(f, name, value, /*apply*/true, label, only, note);
        model->undo_note = n > 0
            ? ("Applied: " + label + " = " + value + " -> " + note)
            : ("Nothing to apply: " + note + " already at " + value);
        model->undo_note_time = ImGui::GetTime();
    };
    // Подпись склеиваем, а НЕ прогоняем вторым проходом через printf: в шаблон
    // подставляется имя группы, и попади в него '%', результат стал бы
    // спецификацией формата с несуществующим аргументом. Сегодня имена — жёсткие
    // литералы из group_of, но держать это на соглашении незачем.
    auto count_and_item = [&](const std::string& title,
                              const std::vector<BroadcastTarget>* only,
                              const std::string& note) {
        const std::string lbl = title + " (" + std::to_string(differ_count(only)) + " differ)";
        if (ImGui::MenuItem(lbl.c_str())) apply_now(only, note);
    };

    const std::vector<BroadcastTarget> all = model->broadcast_targets(f);

    // 1. Свой тип. Источник берём из панели, которая сейчас рисуется
    //    (broadcast_source_*): по app_mode его не вывести — вкладка Parametric
    //    покрывает сразу Bifurcation, LLE и LS.
    std::vector<BroadcastTarget> same_group;
    std::string group;
    for (const auto& t : all)
        if (t.tab == model->broadcast_source_tab && t.idx == model->broadcast_source_idx) {
            group = t.group;
            break;
        }
    if (!group.empty()) {
        for (const auto& t : all)
            if (t.group == group) same_group.push_back(t);
    }
    if (same_group.size() > 1)
        count_and_item("Apply to all " + group, &same_group, "all " + group);

    // 2. Выбор галочками. Маска живёт в модели и пересобирается, когда
    //    сменилось поле или изменился состав целей (диаграмму добавили/закрыли).
    if (ImGui::BeginMenu("Apply to...")) {
        // Подпись состава: поле + имена целей по порядку. Меняется состав —
        // маска собирается заново, а не переезжает на соседние конфиги.
        std::string sig = std::to_string((int)f);
        for (const auto& t : all) { sig += '\x1f'; sig += t.label; }
        if (model->broadcast_pick_sig != sig ||
            model->broadcast_pick.size() != all.size()) {
            model->broadcast_pick.assign(all.size(), '0');
            model->broadcast_pick_sig = sig;
            // По умолчанию отмечаем свой тип: это и подсказка, что здесь
            // выбирают, и самый частый набор.
            for (size_t i = 0; i < all.size(); ++i)
                if (all[i].group == group) model->broadcast_pick[i] = '1';
        }
        if (ImGui::SmallButton("All")) model->broadcast_pick.assign(all.size(), '1');
        ImGui::SameLine();
        if (ImGui::SmallButton("None")) model->broadcast_pick.assign(all.size(), '0');

        // Кнопка применения — СВЕРХУ, до списка: галочек бывает под два десятка,
        // и внизу она уезжает за край экрана. Кнопка, а не пункт меню: пункт в
        // конце списка читается как подпись, а не как действие.
        {
            std::vector<BroadcastTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->broadcast_pick[i] == '1') sel.push_back(all[i]);
            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                char note[64];
                std::snprintf(note, sizeof(note), "%d selected", (int)sel.size());
                apply_now(&sel, note);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::SameLine();
            if (sel.empty()) {
                ImGui::TextDisabled("nothing ticked");
            } else {
                const int d = differ_count(&sel);
                if (d > 0) ImGui::TextDisabled("%d differ", d);
                else       ImGui::TextDisabled("already at this value");
            }
        }
        ImGui::Separator();

        std::string shown_group;
        for (size_t i = 0; i < all.size(); ++i) {
            if (all[i].group != shown_group) {
                shown_group = all[i].group;
                ImGui::SeparatorText(shown_group.c_str());
            }
            ImGui::PushID((int)i);
            bool on = (model->broadcast_pick[i] == '1');
            if (ImGui::Checkbox("##pick", &on))
                model->broadcast_pick[i] = on ? '1' : '0';
            ImGui::SameLine();
            ImGui::TextUnformatted(all[i].label.c_str());
            ImGui::PopID();
        }

        ImGui::EndMenu();
    }

    // 3. Прежний адресат: активный конфиг каждой вкладки (only = nullptr).
    count_and_item("Apply to all calculation tabs", nullptr, "all tabs");

    draw_undo_menu_item(model);
    ImGui::EndPopup();
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string scheme_group_title(const BuiltinScheme& b) {
    return b.group ? std::string(b.group) : "Order " + std::to_string(b.order);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool scheme_is_map(const std::string& name) {
    return name == kMapSchemeName;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<const char*> c_str_list(const std::vector<std::string>& v) {
    std::vector<const char*> out;
    out.reserve(v.size());
    for (const auto& s : v) out.push_back(s.c_str());
    return out;
}
}


namespace ucuda::ui_detail {
static const AppModel* g_scheme_masks = nullptr;
}


namespace ucuda::ui_detail {
void set_scheme_masks(const AppModel* m) { g_scheme_masks = m; }
}


namespace ucuda::ui_detail {
bool draw_scheme_combo(const char* label, std::string& scheme,
                              const std::vector<CustomScheme>& custom_schemes,
                              const std::function<void(const std::string&)>& on_pick,
                              const std::vector<std::string>* enabled_builtins,
                              AppModel* bc,
                              bool is_map,
                              const std::vector<std::string>* wrapper_schemes,
                              bool broadcast) {
    // Nothing to choose for a discrete map: the step is the right-hand side.
    if (is_map) {
        scheme = kMapSchemeName;
        ImGui::TextDisabled("Map (x_{n+1} = f(x_n)) - no scheme");
        return false;
    }
    bool picked = false;
    auto choose = [&](const std::string& nm) {
        scheme = nm;
        if (on_pick) on_pick(nm);
        picked = true;
    };
    // ДАУ: сессия получила только схемы ДАУ (enabled_builtins_from_record) — их и показываем,
    // схемы ОДУ на такой системе не строятся. Скрытые в Settings не предлагаются (выбранная остаётся).
    if (enabled_builtins && !enabled_builtins->empty()
        && (dae_scheme_from_name((*enabled_builtins)[0], nullptr) || (*enabled_builtins)[0] == kBdfSchemeDae)) {
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo(label, scheme.c_str())) {
            ImGui::SeparatorText("DAE");
            bool ms_header = false;
            for (const auto& nm : *enabled_builtins) {
                if (g_scheme_masks && g_scheme_masks->dae_method_hidden(nm) && scheme != nm) continue;
                if (scheme_is_multistep(nm) && !ms_header) {
                    ImGui::SeparatorText("multistep");
                    ms_header = true;
                }
                if (ImGui::Selectable(nm.c_str(), scheme == nm)) choose(nm);
                if (scheme_is_multistep(nm) && ImGui::IsItemHovered())
                    ImGui::SetTooltip("BDF orders 1-5 (DASSL form), its own driver; index 3 goes through GGL.\n"
                                      "Adaptive step: variable step and order. Fixed step: step h, order 1..5\n"
                                      "ramped up, no step controller. Newton: the tab's Newton fields.");
            }
            ImGui::EndCombo();
        }
        return picked;
    }
    // enabled_builtins null = no filter. Empty = no built-in is ticked: that is still
    // "no filter" only while the combo would otherwise be empty; a system that offers
    // custom KRS or wrappers (e.g. Izhikevich, whose reset lives in custom schemes)
    // gets just those. Currently-selected built-in stays visible even if not enabled,
    // so sessions saved with a now-disabled scheme aren't stranded.
    bool has_other = wrapper_schemes && !wrapper_schemes->empty();
    for (const auto& cs : custom_schemes) if (cs.enabled) { has_other = true; break; }
    const bool filter = enabled_builtins && (!enabled_builtins->empty() || has_other);
    auto is_enabled = [&](const char* m) {
        if (!filter) return true;
        for (const auto& n : *enabled_builtins) if (n == m) return true;
        return false;
    };
    ImGui::SetNextItemWidth(kComboW);
    if (ImGui::BeginCombo(label, wrapper_display_name(scheme).c_str())) {
        // Заголовок порядка печатаем лениво — перед ПЕРВОЙ видимой схемой
        // группы, иначе отфильтрованная группа оставила бы пустую шапку.
        std::string shown_group;
        for (const auto& b : kBuiltinSchemes) {
            // Выключенная у системы или скрытая в Settings схема не предлагается, выбранная остаётся.
            if ((!is_enabled(b.name) || (g_scheme_masks && g_scheme_masks->scheme_hidden(b.name)))
                && scheme != b.name) continue;
            if (scheme_group_title(b) != shown_group) {
                ImGui::SeparatorText(scheme_group_title(b).c_str());
                shown_group = scheme_group_title(b);
            }
            if (ImGui::Selectable(b.name, scheme == b.name)) choose(b.name);
        }
        // Многошаговый BDF 1-5 — свой драйвер, флага scheme_* у него нет: показывается всегда
        // (enabled_builtins_from_record его тоже всегда добавляет), только с адаптивным шагом.
        if (is_enabled(kBdfSchemeOde) || scheme == kBdfSchemeOde) {
            ImGui::SeparatorText("Multistep (adaptive step only)");
            if (ImGui::Selectable(kBdfSchemeOde, scheme == kBdfSchemeOde)) choose(kBdfSchemeOde);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("BDF orders 1-5 with variable step (DASSL form), its own driver.\n"
                                  "Needs Integration -> Adaptive step; stiff problems.");
        }
        // Своя шапка, а не голая черта: у групп порядка и у обёрток заголовки
        // есть, и без неё кастомные читались как хвост последней Order-группы.
        // Выключенная галочкой КРС не предлагается, но уже выбранная остаётся
        // видна — как и у встроенных, чтобы сессия не осиротела.
        bool custom_header = false;
        for (const auto& cs : custom_schemes) {
            if (!cs.enabled && scheme != cs.name) continue;
            if (!custom_header) { ImGui::SeparatorText("Custom"); custom_header = true; }
            if (ImGui::Selectable((cs.name + "##custom").c_str(), scheme == cs.name))
                choose(cs.name);
        }
        // Обёртки — отдельной группой: паспортного порядка в таблице у них
        // нет (у Extr он считается из базы и числа стадий, у Comp вообще
        // известен только в рантайме), поэтому в группировку по Order выше
        // они не встают.
        // Сессии получают только включённые обёртки; выбранная, но выключенная
        // добавляется обратно по той же причине, что и КРС выше.
        std::vector<std::string> sorted;
        if (wrapper_schemes) sorted = *wrapper_schemes;
        {
            ExtrapolationSpec esp;
            CompositionSpec   csp;
            if ((parse_extrapolation_name(scheme, &esp) || parse_composition_name(scheme, &csp))
                && std::find(sorted.begin(), sorted.end(), scheme) == sorted.end())
                sorted.push_back(scheme);
        }
        if (!sorted.empty()) {
            ImGui::SeparatorText("Wrappers");
            // Порядок sort_wrapper_schemes: список сессии мог приехать из файла
            // в старом порядке добавления. Подзаголовок — на смене группы
            // "тип + опорник".
            sort_wrapper_schemes(sorted);
            std::string shown_wgroup;
            for (const auto& nm : sorted) {
                const std::string wg = wrapper_group_title(nm);
                if (wg != shown_wgroup) {
                    if (!wg.empty()) ImGui::TextDisabled("%s", wg.c_str());
                    shown_wgroup = wg;
                }
                const std::string shown = wrapper_display_name(nm);
                if (ImGui::Selectable((shown + "##" + nm).c_str(), scheme == nm))
                    choose(nm);
                // Коэффициенты прямо в списке выбора: именно здесь схему и
                // выбирают, а знать, какие числа за ней стоят, надо ДО того.
                // Значения параметров берутся у bc, когда он есть: символьная
                // gamma без них раскрыта не будет, и отчёт честно скажет "?".
                if (ImGui::IsItemHovered()) {
                    static const std::map<std::string, std::string> kNoParams;
                    const std::string rep = wrapper_coefficients_text(
                        nm, custom_schemes, bc ? bc->param_values : kNoParams);
                    if (!rep.empty())      ImGui::SetTooltip("%s", rep.c_str());
                    else if (shown != nm)  ImGui::SetTooltip("%s", nm.c_str());
                }
            }
        }
        ImGui::EndCombo();
    }
    // Меню вешаем ПОСЛЕ комбо: последним элементом остаётся оно само, и ПКМ
    // попадает по нему, а не по строке раскрытого списка.
    // broadcast = false — комбо не основной схемы (эталон вкладки Order):
    // рассылка BroadcastField::Scheme записала бы значение в ОСНОВНУЮ схему
    // чужих конфигов. PushID(label): ID popup'а собирается из "scheme", и две
    // комбо схемы на одном уровне ID-стека склеивали бы меню в одно окно,
    // рисуя его содержимое дважды (конфликт ID у галочек "Apply to...").
    if (broadcast) {
        ImGui::PushID(label);
        field_apply_all_menu(bc, BroadcastField::Scheme, {}, "scheme", scheme);
        ImGui::PopID();
    }
    return picked;
}
}


namespace ucuda::ui_detail {
void draw_sweep_target_combo(const char* label,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars,
                                    int& par_index, bool& over_var, int& var_index,
                                    bool& over_h,
                                    bool* other_over_h,
                                    bool note_when_empty,
                                    float width,
                                    bool allow_s,
                                    AppModel* bc,
                                    BroadcastField which,
                                    bool allow_h,
                                    int* ad_axis,
                                    const AdaptiveSettings* ad) {
    // Адаптивный шаг: h — шаг вывода, его не свипают; вместо него ось может свипать
    // настройку шага (*ad_axis — AdaptiveAxisKind движка, 0 — обычная цель).
    const bool ad_on = ad_axis && ad && ad->enabled;
    if (ad_axis && !ad_on) *ad_axis = 0;
    if (ad_on) allow_h = false;
    const bool ad_sel = ad_on && *ad_axis != 0;
    // A map has no step, so there is nothing to sweep over.
    if (!allow_h) over_h = false;
    if (params.empty() && vars.empty() && note_when_empty) {
        ImGui::TextDisabled("No parameters/variables (select a system first)");
        return;
    }
    // Кламп индексов: сохранённая сессия могла прийти от системы с другим
    // числом переменных/параметров. Отрицательный par_index — не мусор, а цель
    // "s"; сбрасываем его только там, где схема a[0] не читает (или сессия
    // пришла от схемы, которая читала).
    if (par_index < 0 && !allow_s) par_index = 0;
    if (par_index >= (int)params.size()) par_index = 0;
    if (var_index < 0 || var_index >= (int)vars.size())   var_index = 0;

    auto ad_name = [&](int kind) -> std::string {
        if (kind == kAdAxisRtol) return "rtol";
        if (kind == kAdAxisAtol) return "atol";
        if (kind == kAdAxisTol)  return "tol (atol = rtol x ratio)";
        AdaptiveCtrlResolved cr;
        std::string cerr;
        const bool known = ad && adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr);
        const int k = kind - kAdAxisCtrl;
        return "controller: " + ((known && k >= 0 && k < (int)cr.par.size()) ? cr.par[(size_t)k] : std::string("?"));
    };
    const std::string preview =
          ad_sel                      ? ad_name(*ad_axis)
        : over_h                      ? std::string("dt (h)")
        : (over_var && !vars.empty()) ? (vars[var_index] + " (IC)")
        : (par_index < 0)             ? std::string("s (a[0])")
        : (!params.empty())           ? params[par_index]
                                      : std::string("?");
    // Значение для ПКМ-рассылки: четыре поля цели одной строкой (см.
    // encode_sweep_target). Меню вешается на комбо в ОБОИХ случаях — и когда
    // список раскрыт, и когда нет: оно принадлежит комбо, а не его строкам.
    auto apply_all_menu = [&]() {
        field_apply_all_menu(bc, which, {}, label,
            encode_sweep_target(par_index, over_var, var_index, over_h));
    };
    ImGui::SetNextItemWidth(width);
    if (!ImGui::BeginCombo(label, preview.c_str())) { apply_all_menu(); return; }

    for (int i = 0; i < (int)params.size(); ++i) {
        const bool sel = !ad_sel && !over_var && !over_h && par_index == i;
        if (ImGui::Selectable(params[i].c_str(), sel)) {
            par_index = i; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (!params.empty() && !vars.empty()) ImGui::Separator();
    for (int i = 0; i < (int)vars.size(); ++i) {
        const bool sel = !ad_sel && over_var && !over_h && var_index == i;
        if (ImGui::Selectable((vars[i] + " (IC)").c_str(), sel)) {
            var_index = i; over_var = true; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }

    if (allow_s) {
        ImGui::Separator();
        if (ImGui::Selectable("s (a[0])", !ad_sel && !over_var && !over_h && par_index < 0)) {
            par_index = -1; over_var = false; over_h = false;
            if (ad_axis) *ad_axis = 0;
        }
    }
    if (allow_h) {
        ImGui::Separator();
        if (ImGui::Selectable("dt (h)", over_h)) {
            over_h = true; over_var = false;
            if (other_over_h) *other_over_h = false;   // ровно одна ось = h
        }
    }
    if (ad_on) {
        ImGui::Separator();
        auto item = [&](int kind) {
            if (ImGui::Selectable(ad_name(kind).c_str(), *ad_axis == kind)) {
                *ad_axis = kind; over_var = false; over_h = false;
            }
        };
        item(kAdAxisRtol); item(kAdAxisAtol); item(kAdAxisTol);
        AdaptiveCtrlResolved cr;
        std::string cerr;
        if (adaptive_resolve_ctrl(ad->ctrl, 7, cr, cerr))
            for (int k = 0; k < (int)cr.par.size(); ++k) item(kAdAxisCtrl + k);
    }
    ImGui::EndCombo();
    apply_all_menu();
}
}


namespace ucuda::ui_detail {
std::vector<AdaptiveSettings*> collect_adaptive_slots(AppModel& m) {
    std::vector<AdaptiveSettings*> v;
    v.push_back(&m.phase_session.adaptive);
    // Custom: источник — общий конфиг вкладки (apply_shared_to_* раздаёт его уровням при
    // запуске); его Phase — вместе с ним, как это делает draw_shared_config.
    v.push_back(&m.custom_session.shared.adaptive);
    v.push_back(&m.custom_session.phase_session.adaptive);
    for (auto& d : m.bifurcation_session.diagrams) v.push_back(&d.adaptive);
    for (auto& c : m.lle_session.curves)           v.push_back(&c.adaptive);
    for (auto& c : m.ls_session.curves)            v.push_back(&c.adaptive);
    for (auto& c : m.basins_session.configs)       v.push_back(&c.adaptive);
    for (auto& c : m.metrics_session.configs)      v.push_back(&c.adaptive);
    return v;
}
}


namespace ucuda::ui_detail {
void broadcast_adaptive(AppModel& m, const AdaptiveSettings& a) {
    std::vector<AdaptiveSettings> old;
    for (AdaptiveSettings* p : collect_adaptive_slots(m)) { old.push_back(*p); *p = a; }
    AppModel* mp = &m;
    m.push_undo("adaptive step settings to all tabs",
        [mp, old]() {
            std::vector<AdaptiveSettings*> v = collect_adaptive_slots(*mp);
            if (v.size() == old.size())
                for (size_t i = 0; i < v.size(); ++i) *v[i] = old[i];
        },
        [mp, a]() { for (AdaptiveSettings* p : collect_adaptive_slots(*mp)) *p = a; });
}
}


namespace ucuda::ui_detail {
std::vector<std::pair<std::string, std::string>> user_ctrl_list() {
    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    std::lock_guard<std::mutex> lk(L.mu);
    std::vector<std::pair<std::string, std::string>> v;
    for (const AdaptiveUserCtrl& u : L.items) {
#ifdef UCUDA_AD_HAIRER_ONLY
        if (u.kind == kUserCtrlFilter) continue;   // встроенный Filter не собран
#endif
        v.emplace_back(u.name, u.tip);
    }
    return v;
}
}


namespace ucuda::ui_detail {
static const char* const kCtrlBodyHelp =
    "void ucuda_ctrl_custom(const UcudaCtlIn& in, UcudaCtlMem& m, UcudaCtlOut& o) { <body> }\n"
    "\n"
    "in.h          size of this attempt;  in.t - t at the start of the step\n"
    "in.y0[i]      solution at t;  in.y1[i] - higher-order solution at t + h\n"
    "in.yerr[i]    error estimate y1 - yhat; in.nlo estimates of in.n values each\n"
    "              (DOP853: in.yerr[in.n + i] is the 3rd-order one)\n"
    "in.rtol, in.atol[i], in.c[k] - parameters of this controller, in.q - estimator order\n"
    "in.hmin, in.hmax (+inf: no limit; the driver clamps to it anyway), in.nrej - rejected attempts\n"
    "m.h[j], m.err[j]  accepted steps and their errors, j = 0 the latest (3 kept)\n"
    "m.nacc        accepted steps so far;  m.user[0..7] - free memory kept between steps\n"
    "o.err         error of the attempt (1 = on the tolerance): the norm or its square (Hairer\n"
    "              reports err^2); the driver only checks that it is finite; goes to the log\n"
    "o.accept      1 - take the step, 0 - retry it with o.h\n"
    "o.h           next step, or the step of the retry\n"
    "\n"
    "in.k[0..7]    constants computed by the prepare section (0 if not set)\n"
    "\n"
    "Prepare section (optional), once per trajectory:\n"
    "void ucuda_ctrl_custom_prep(const numb* c, int q, numb* k) { <prepare> }\n"
    "c[] - the parameters, q - estimator order, k[0..7] - constants for the body.\n"
    "Put divisions and powers that depend only on c[] and q here: double division\n"
    "costs as much as several RHS evaluations on GeForce cards.\n"
    "\n"
    "Helpers: ucuda_ctl_err(in) - RMS norm of the estimate, sc_i = atol_i + rtol max(|y0_i|, |y1_i|);\n"
    "ucuda_ctl_lg2err2(in, err2) - log2(err^2) in float, err^2 into err2, no sqrt (\"Hairer fast\");\n"
    "ucuda_ctl_lg2err2_numb(in, err2) - the same in numb (\"Hairer numb\");\n"
    "ucuda_ctl_pow(x, y) - fast x^y (float);  ucuda_ctl_log2(x), ucuda_ctl_exp2(f) - float;\n"
    "ucuda_clamp(x, lo, hi);  ucuda_fmin/fmax (double), ucuda_fminf/fmaxf (float).  numb is double.\n"
    "Index arrays (m.user, in.k, ...) with constants: a computed index puts the state of\n"
    "the GPU thread into local memory.\n"
    "The driver then clamps h to h_max, takes the step at h_min when it would go below\n"
    "(or after max rejects), and retries with at most 0.9 h.";
}


namespace ucuda::ui_detail {
void draw_ctrl_library_window(AppModel& m) {
    // Результат фонового Check — и при закрытом окне, чтобы future не висел.
    if (m.ctrl_lib_check_future.valid()
        && m.ctrl_lib_check_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        const std::pair<bool, std::string> r = m.ctrl_lib_check_future.get();
        m.ctrl_lib_check_ok  = r.first;
        m.ctrl_lib_check_log = r.second;
    }
    if (!m.show_ctrl_library) return;
    ImGui::SetNextWindowSize(ImVec2(940.0f, 640.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Step controller library", &m.show_ctrl_library)) { ImGui::End(); return; }

    AdaptiveCtrlLibrary& L = adaptive_ctrl_library();
    bool edited = false;
    std::string check_body;
    bool do_check = false;
    {
        std::lock_guard<std::mutex> lk(L.mu);
        std::vector<AdaptiveUserCtrl>& items = L.items;
        if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;

        auto taken = [&](const std::string& s, int self) {
            if (adaptive_find_builtin(s)) return true;
            for (int i = 0; i < (int)items.size(); ++i) if (i != self && items[(size_t)i].name == s) return true;
            return false;
        };
        auto add = [&](AdaptiveUserCtrl u) {
            std::string base = u.name.empty() ? std::string("controller") : u.name;
            u.name = base;
            for (int k = 2; taken(u.name, -1); ++k) u.name = base + " " + std::to_string(k);
            items.push_back(u);
            m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        };

        // Слева — список и кнопки.
        ImGui::BeginChild("##ctrl_list", ImVec2(230.0f, 0.0f), ImGuiChildFlags_Borders);
        if (ImGui::Button("New...")) ImGui::OpenPopup("##ctrl_new");
        if (ImGui::BeginPopup("##ctrl_new")) {
            if (ImGui::Selectable("empty C body")) {
                AdaptiveUserCtrl u;
                u.name = "my controller";
                u.body = "const numb err = ucuda_ctl_err(in);\n"
                         "o.err = err;\n"
                         "o.accept = err <= 1;\n"
                         "o.h = in.h * ucuda_clamp((numb)0.9 * ucuda_ctl_pow(err, (numb)-1 / (in.q + 1)),\n"
                         "                         (numb)0.2, o.accept ? (numb)5 : (numb)1);\n";
                add(u);
            }
#ifndef UCUDA_AD_HAIRER_ONLY   // фильтры — параметры встроенного Filter, а он не собран
            if (ImGui::Selectable("empty filter")) {
                AdaptiveUserCtrl u;
                u.name = "my filter"; u.kind = kUserCtrlFilter;
                for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                add(u);
            }
#endif
            int np = 0;
            const AdaptiveCtrlPreset* pr = adaptive_ctrl_presets(&np);
            ImGui::SeparatorText("C body presets");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlBody) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#ifndef UCUDA_AD_HAIRER_ONLY
            ImGui::SeparatorText("Soderlind filters");
            for (int i = 0; i < np; ++i) {
                if (pr[i].kind != kUserCtrlFilter) continue;
                if (ImGui::Selectable(pr[i].name)) add(adaptive_ctrl_from_preset(pr[i]));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", pr[i].tip);
            }
#endif
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(m.ctrl_lib_sel < 0);
        if (ImGui::Button("Copy") && m.ctrl_lib_sel >= 0) add(items[(size_t)m.ctrl_lib_sel]);
        ImGui::SameLine();
        if (ImGui::Button("Delete") && m.ctrl_lib_sel >= 0) {
            items.erase(items.begin() + m.ctrl_lib_sel);
            if (m.ctrl_lib_sel >= (int)items.size()) m.ctrl_lib_sel = (int)items.size() - 1;
            edited = true;
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        for (int i = 0; i < (int)items.size(); ++i) {
            ImGui::PushID(i);
            const AdaptiveUserCtrl& u = items[(size_t)i];
            const std::string label = u.name + (u.kind == kUserCtrlFilter ? "  (filter)" : "");
            if (ImGui::Selectable(label.c_str(), m.ctrl_lib_sel == i)) m.ctrl_lib_sel = i;
            if (!u.tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", u.tip.c_str());
            ImGui::PopID();
        }
        if (items.empty()) ImGui::TextDisabled("Empty. New... adds a controller\nfrom scratch or from a preset.");
        ImGui::EndChild();

        // Справа — выбранная запись.
        ImGui::SameLine();
        ImGui::BeginChild("##ctrl_edit", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        if (m.ctrl_lib_sel >= 0) {
            AdaptiveUserCtrl& u = items[(size_t)m.ctrl_lib_sel];
            const ImVec4 warn(1.0f, 0.55f, 0.3f, 1.0f);
            edited |= InputTextStr("name", u.name, 260.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Tabs refer to a controller by its name: after a rename they\n"
                                  "report the old name as unknown until you pick it again.");
            if (u.name.empty()) ImGui::TextColored(warn, "The name is empty.");
            else if (taken(u.name, m.ctrl_lib_sel))
                ImGui::TextColored(warn, "The name is taken (a built-in controller or another entry).");
            int kind = u.kind;
#ifdef UCUDA_AD_HAIRER_ONLY
            // Фильтр — параметры встроенного Filter, который в этой сборке не собран: такую
            // запись (из старой библиотеки) можно только перевести в C body или удалить.
            if (u.kind == kUserCtrlFilter) {
                ImGui::TextColored(warn, "Soderlind filters are not built in this version: the tabs do not offer it.");
                if (ImGui::SmallButton("make it a C body")) kind = kUserCtrlBody;
            }
#else
            ImGui::RadioButton("C body", &kind, kUserCtrlBody);
            ImGui::SameLine();
            ImGui::RadioButton("Soderlind filter", &kind, kUserCtrlFilter);
#endif
            if (kind != u.kind) {
                u.kind = kind;
                if (kind == kUserCtrlFilter) {
                    u.par_values.clear();
                    for (double v : adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7)) u.par_values.push_back(fmt_num_shortest(v));
                } else if (u.par_names.size() != u.par_values.size()) {
                    u.par_values.resize(u.par_names.size());
                }
                edited = true;
            }
            edited |= InputTextStr("description", u.tip, 520.0f);

            if (u.kind == kUserCtrlFilter) {
#ifndef UCUDA_AD_HAIRER_ONLY   // иначе встроенного Filter нет (adaptive_find_builtin -> nullptr)
                const AdaptiveCtrlInfo* fi = adaptive_find_builtin("Filter");
                ImGui::TextDisabled("%s", fi->tip);
                if ((int)u.par_values.size() != fi->npar) {
                    const std::vector<double> d = adaptive_ctrl_defaults(UCUDA_CTRL_FILTER, 7);
                    u.par_values.resize((size_t)fi->npar);
                    for (int i = 0; i < fi->npar; ++i)
                        if (u.par_values[(size_t)i].empty()) u.par_values[(size_t)i] = fmt_num_shortest(d[(size_t)i]);
                    edited = true;
                }
                for (int i = 0; i < fi->npar; ++i) {
                    ImGui::PushID(i);
                    edited |= InputTextStr(fi->par[i], u.par_values[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
#endif
            } else {
                bool hr = u.hairer_rules != 0;
                if (ImGui::Checkbox("Hairer's initial step and end-point rule", &hr)) { u.hairer_rules = hr; edited = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("On: initial step as hinit of dopri5 / dop853 and the last step\n"
                                      "stretched when x + 1.01 h passes the end (as the built-in Hairer).\n"
                                      "Off: scipy's initial step, the step is cut exactly at the end.");
                ImGui::Text("Parameters c[k] (name, default):");
                if (u.par_values.size() != u.par_names.size()) u.par_values.resize(u.par_names.size());
                int remove = -1;
                for (int i = 0; i < (int)u.par_names.size(); ++i) {
                    ImGui::PushID(i);
                    ImGui::Text("c[%d]", i);
                    ImGui::SameLine(60.0f);
                    edited |= InputTextStr("##pn", u.par_names[(size_t)i], 140.0f);
                    ImGui::SameLine();
                    edited |= InputTextStr("##pv", u.par_values[(size_t)i], kFieldW);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x")) remove = i;
                    ImGui::PopID();
                }
                if (remove >= 0) {
                    u.par_names.erase(u.par_names.begin() + remove);
                    u.par_values.erase(u.par_values.begin() + remove);
                    edited = true;
                }
                ImGui::BeginDisabled(u.par_names.size() >= (size_t)UCUDA_CTL_NPAR);
                if (ImGui::SmallButton("+ parameter")) {
                    u.par_names.push_back("p" + std::to_string(u.par_names.size()));
                    u.par_values.push_back("0");
                    edited = true;
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled("(up to %d; each is also a sweep axis)", UCUDA_CTL_NPAR);

                ImGui::Text("Prepare:");
                ImGui::SameLine();
                ImGui::TextDisabled("once per trajectory: k[0..7] from c[] and q, read by the body as in.k[]");
                edited |= InputTextMultilineStr("##prep", u.prep,
                                                ImVec2(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 6.5f));
                // Проверка и её результат — по паре (подготовка, тело), как их собирает модуль.
                const std::string packed = adaptive_ctrl_pack(u.prep, u.body);
                const bool checked = !m.ctrl_lib_check_body.empty() && m.ctrl_lib_check_body == packed;
                ImGui::Text("Body:");
                ImGui::SameLine();
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCtrlBodyHelp);
                ImGui::SameLine();
                const bool checking = m.ctrl_lib_check_future.valid();
                ImGui::BeginDisabled(checking);
                if (ImGui::SmallButton("Check")) { do_check = true; check_body = packed; }
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("Compile the prepare section and the body with NVRTC (no run). The CPU\n"
                                      "build with cl.exe happens on the first CPU run of Analysis.");
                if (checked) {
                    ImGui::SameLine();
                    if (checking)                 ImGui::TextDisabled("checking...");
                    else if (m.ctrl_lib_check_ok) ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "compiles");
                    else                          ImGui::TextColored(warn, "does not compile:");
                }
                edited |= InputTextMultilineStr("##body", u.body, ImVec2(-1.0f, m.ctrl_lib_body_h));
                draw_resize_handle("##body_resize", m.ctrl_lib_body_h);
                if (checked && !m.ctrl_lib_check_log.empty())
                    ImGui::InputTextMultiline("##check_log", const_cast<char*>(m.ctrl_lib_check_log.c_str()),
                                              m.ctrl_lib_check_log.size() + 1, ImVec2(-1.0f, 110.0f),
                                              ImGuiInputTextFlags_ReadOnly);
                if (ImGui::TreeNode("API")) {
                    ImGui::TextUnformatted(kCtrlBodyHelp);
                    ImGui::TreePop();
                }
            }
        } else {
            ImGui::TextDisabled("Select a controller on the left, or add one with New...");
            ImGui::TextDisabled("Controllers of this library appear in the \"controller\" list of every tab.");
        }
        ImGui::EndChild();
    }
    if (do_check && !m.ctrl_lib_check_future.valid()) {
        // NVRTC в фоне: окно не замирает на время сборки. Результат забирается в начале кадра.
        m.ctrl_lib_check_body = check_body;
        m.ctrl_lib_check_ok   = false;
        m.ctrl_lib_check_log.clear();
        m.ctrl_lib_check_future = std::async(std::launch::async, [check_body]() {
            std::string log;
            const bool ok = nvrtc_check_ctrl_body(check_body, log);
            return std::make_pair(ok, log);
        });
    }
    if (edited) m.ctrl_lib_dirty = true;
    if (m.ctrl_lib_dirty && !ImGui::IsAnyItemActive()) {
        std::string err;
        m.ctrl_lib_status = save_ctrl_library(&err) ? std::string() : err;
        m.ctrl_lib_dirty = false;
    }
    if (!m.ctrl_lib_status.empty())
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", m.ctrl_lib_status.c_str());
    ImGui::End();
}
}


namespace ucuda::ui_detail {
bool draw_adaptive_block(const char* id, AdaptiveSettings& a, const std::string& scheme,
                                int opts, bool is_map, AppModel* bc) {
    if (is_map) return false;   // у отображения шага нет
    ImGui::PushID(id);
    bool changed = false;
    const bool ok = adaptive_scheme_name_ok(scheme);

    int mode = a.enabled ? 1 : 0;
    changed |= ImGui::RadioButton("Fixed step", &mode, 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(!ok && !a.enabled);
    changed |= ImGui::RadioButton("Adaptive step", &mode, 1);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("Step size chosen by an error controller from the embedded\n"
                          "error estimate of the scheme. Needs RK45, DOPRI78, DOP853,\n"
                          "GBS 2-4 ... or an extrapolation Extr(...) with at least two stages:\n"
                          "its first K-1 stages give the lower-order solution, and the\n"
                          "dense output between step nodes is cubic Hermite.");
    a.enabled = (mode == 1);

    if (a.enabled) {
        if (!ok)
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f),
                               "The scheme has no error estimate: choose RK45, DOPRI78, DOP853, GBS 2-4 ...\n"
                               "or Extr(...) with at least two stages.");
        if (opts & kAdUiRaw) {
            int out = a.raw_nodes ? 1 : 0;
            changed |= ImGui::RadioButton("uniform grid", &out, 0);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output on a uniform time grid through the dense output of\n"
                                  "the scheme; the field h is the output step.");
            ImGui::SameLine();
            changed |= ImGui::RadioButton("step nodes", &out, 1);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Output exactly at the accepted steps (including the steps\n"
                                  "shortened at the end of the transient and of the interval);\n"
                                  "the decimator keeps every N-th node, h is not used.");
            a.raw_nodes = (out == 1);
        }
        changed |= InputNumStr("rtol", a.rtol, kFieldW);
        changed |= InputNumListStr("atol", a.atol, kFieldW);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("One value for all variables, or one per variable separated by ';'.");
        changed |= InputNumStr("h0", a.h0, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Initial step. Empty: chosen automatically.");
        changed |= InputNumStr("h_min", a.hmin, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Smallest step. A step that would go below it is taken at h_min\n"
                              "anyway and counted as forced. Empty: the larger of 10 ulp(t) and\n"
                              "1e-12 x the length of the whole interval (transient + record).\n"
                              "A set value is raised to 10 ulp(t) where it is smaller: below that\n"
                              "t + h_min == t and the step would not advance time.");
        changed |= InputNumStr("h_max", a.hmax, kFieldW, {}, true);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Largest step. Empty: no limit.");
        changed |= InputNumStr("max rejects", a.max_rej, kFieldW, {}, true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Rejections allowed per step. After this many in a row the next\n"
                              "attempt is made at h_min and taken unconditionally. Empty: no limit.\n"
                              "0: no retries - every attempt is taken, a rejection only shrinks the\n"
                              "next step (counted as forced); the tolerance becomes a target, not a bound.");

        int nb = 0;
        const AdaptiveCtrlInfo* bi = adaptive_builtin_ctrls(&nb);
        ImGui::SetNextItemWidth(kComboW);
        if (ImGui::BeginCombo("controller", a.ctrl.c_str())) {
            auto pick = [&](const std::string& name) {
                if (a.ctrl != name) { a.ctrl = name; a.ctrl_params.clear(); changed = true; }
            };
            for (int i = 0; i < nb; ++i) {
                if (ImGui::Selectable(bi[i].name, a.ctrl == bi[i].name)) pick(bi[i].name);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", bi[i].tip);
            }
            const std::vector<std::pair<std::string, std::string>> users = user_ctrl_list();
            if (!users.empty()) ImGui::SeparatorText("library");
            for (size_t i = 0; i < users.size(); ++i) {
                ImGui::PushID((int)i);
                if (ImGui::Selectable(users[i].first.c_str(), a.ctrl == users[i].first)) pick(users[i].first);
                if (!users[i].second.empty() && ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", users[i].second.c_str());
                ImGui::PopID();
            }
            ImGui::Separator();
            // Окно библиотеки — флаг в AppModel; без модели (bc == nullptr) пункта нет.
            if (bc != nullptr && ImGui::Selectable("edit library...")) bc->show_ctrl_library = true;
            ImGui::EndCombo();
        }
        AdaptiveCtrlResolved cur;
        std::string cur_err;
        if (!adaptive_resolve_ctrl(a.ctrl, adaptive_scheme_q(scheme), cur, cur_err)) {
            ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.3f, 1.0f), "%s", cur_err.c_str());
        } else {
            const int npar = (int)cur.par.size();
            const std::vector<double>& def = cur.def;
            if ((int)a.ctrl_params.size() != npar) {
                std::string line;
                for (int i = 0; i < npar; ++i) {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "%s%s = %g", i ? ", " : "", cur.par[(size_t)i].c_str(),
                                  i < (int)def.size() ? def[(size_t)i] : 0.0);
                    line += buf;
                }
                if (npar > 0) {
                    ImGui::TextDisabled("%s", line.c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("edit")) {
                        a.ctrl_params.clear();
                        for (int i = 0; i < npar; ++i) a.ctrl_params.push_back(fmt_num_shortest(def[(size_t)i]));
                        changed = true;
                    }
                }
            } else {
                for (int i = 0; i < npar; ++i) {
                    ImGui::PushID(i);
                    changed |= InputNumStr(cur.par[(size_t)i].c_str(), a.ctrl_params[(size_t)i], kFieldW);
                    ImGui::PopID();
                }
                if (ImGui::SmallButton("defaults")) { a.ctrl_params.clear(); changed = true; }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(cur.user ? "Back to the defaults stored in the controller library."
                                               : "Back to the defaults for this scheme.");
            }
        }
        if ((opts & kAdUiInterp) && a.raw_nodes) {
            const char* items[] = { "node", "parabola", "Hermite by f" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("peak interpolation", &a.peak_interp, items, IM_ARRAYSIZE(items));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("How a peak (in Metrics also a min / max) is placed between step nodes:\n"
                                  "node - the largest node itself;\n"
                                  "parabola - vertex of the parabola through three nodes\n"
                                  "  (the fixed-step formula generalised to unequal steps);\n"
                                  "Hermite by f - cubic on the step where dx/dt changes sign,\n"
                                  "  built from the nodes and the exact derivatives.");
        }
        if ((opts & kAdUiMaxPts) && a.raw_nodes)
            changed |= InputNumStr("max points", a.max_points, kFieldW);
        if (opts & kAdUiRenorm) {
            const char* items[] = { "exactly at k*NT", "first node after NT" };
            ImGui::SetNextItemWidth(kComboW);
            changed |= ImGui::Combo("renormalization", &a.lyap_renorm, items, IM_ARRAYSIZE(items));
        }
    }
    if (opts & kAdUiMinMax) {
        changed |= ImGui::Checkbox("min/max by interpolated extrema in Fixed", &a.minmax_interp_fixed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("With the adaptive step min/max always use interpolated extrema: on the\n"
                              "uniform grid the vertex of the parabola through three samples (as peaks\n"
                              "do), on step nodes the peak interpolation. This applies the parabola to\n"
                              "the fixed step too.");
    }
    if (bc && a.enabled) {
        if (ImGui::SmallButton("apply to all tabs")) broadcast_adaptive(*bc, a);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Copy these step settings to every tab that has them\n"
                              "(Analysis, Custom, Bifurcation, LLE, LS, Basins, Metrics). Ctrl+Z undoes.");
    }
    ImGui::PopID();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_newton_fields(NewtonSettings& n, float w, const char* id) {
    bool changed = false;
    ImGui::PushID(id);
    int jac_mode = n.full ? 1 : 0;
    ImGui::SetNextItemWidth(w * 2.0f);
    if (ImGui::Combo("Newton jacobian", &jac_mode, "modified (once per step)\0full (every iteration)\0")) {
        n.full = (jac_mode == 1);
        changed = true;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("modified: Jacobian and LU reused across iterations and\n"
                          "refreshed when the correction stops shrinking.\n"
                          "Cheaper on GPU and the right default.\n"
                          "full: rebuilt every iteration. Quadratic convergence,\n"
                          "most robust at large h, ~k times the cost.\n"
                          "Passed at run time as a[M+4]: no recompilation.");
    changed |= InputNumStr("Newton tolerance", n.tol, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Newton stops once ||dX|| falls below this (a[M+2], run time).\n"
                          "DAE methods also stop at ||dX|| < tol * |X| (relative).\n"
                          "With numb = float it also stops at the precision floor\n"
                          "||dX|| < 16 eps |X|: below that float cannot go.");
    changed |= InputNumStr("Newton max iterations", n.max_iters, w);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Hard cap on Newton passes per step (a[M+3], run time).\n"
                          "2-3 is typical from the explicit-Euler predictor.\n"
                          "A step that hits the cap is accepted unconverged: on a DAE\n"
                          "the constraint then drifts. DAE: full Newton, ~25 iterations.");
    ImGui::PopID();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_integration_block(const char* header,
                                   const std::string& scheme,
                                   const std::vector<CustomScheme>& custom_schemes,
                                   const IntegrationFields& f,
                                   AppModel* bc,
                                   bool is_map) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return false;
    bool changed = false;
    auto menu = [&](BroadcastField kind, const char* label, const std::string& v) {
        return [bc, kind, label, &v] { field_apply_all_menu(bc, kind, {}, label, v); };
    };
    // h is pinned to 1 for a map (see default_h_from_record), so the field is
    // gone and the engine's "time" is exactly the iteration count.
    const char* t_max_label     = is_map ? "iterations"           : "computing time";
    const char* transient_label = is_map ? "transient iterations" : "transient time";
    if (f.h && !is_map) changed |= InputNumStr("h", *f.h, kFieldW,
                                    menu(BroadcastField::StepH, "h", *f.h));
    if (f.h && !is_map && f.adaptive && f.adaptive->enabled) {
        ImGui::SameLine();
        // LLE/LS (перенормировка в узлах шага) h не используют вовсе.
        ImGui::TextDisabled((f.adaptive_opts & kAdUiRenorm) ? "(not used)"
                            : f.adaptive->raw_nodes ? "(not used on step nodes)" : "(output step)");
    }
    if (f.symmetry_s && scheme_uses_symmetry(scheme, custom_schemes))
        changed |= InputNumStr("symmetry s", *f.symmetry_s, kFieldW,
                               menu(BroadcastField::SymmetryS, "symmetry s", *f.symmetry_s));
    if (f.cd_fp_iters && scheme_uses_cd_fp_iters(scheme)) {
        changed |= InputNumStr("FP iterations n", *f.cd_fp_iters, kFieldW,
                               menu(BroadcastField::CdFpIters, "FP iterations n", *f.cd_fp_iters));
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kCdFpItersTip);
    }
    if (f.newton && scheme_uses_newton(scheme))
        changed |= draw_newton_fields(*f.newton, kFieldW);
    if (f.t_max)      changed |= InputNumStr(t_max_label, *f.t_max,       kFieldW,
                               menu(BroadcastField::TMax, t_max_label, *f.t_max));
    if (f.transient)   changed |= InputNumStr(transient_label, *f.transient,   kFieldW,
                               menu(BroadcastField::Transient, transient_label, *f.transient));
    if (f.pre_scaller) changed |= InputNumStr("decimator",      *f.pre_scaller, kFieldW,
                               menu(BroadcastField::PreScaller, "decimator", *f.pre_scaller));
    if (f.max_value)   changed |= InputNumStr("max value",      *f.max_value,   kFieldW,
                               menu(BroadcastField::MaxValue, "max value", *f.max_value));
    if (f.adaptive) {
        ImGui::Separator();
        changed |= draw_adaptive_block(header, *f.adaptive, scheme, f.adaptive_opts, is_map, bc);
    }
    return changed;
}
}


namespace ucuda::ui_detail {
std::string field_label_auto(const std::string& name) { return name_to_latex(name); }
}


namespace ucuda::ui_detail {
static bool g_field_latex = true;
}


namespace ucuda::ui_detail {
static const FieldLabels* g_field_labels = nullptr;
}


namespace ucuda::ui_detail {
void set_field_latex(bool on) { g_field_latex = on; }
}


namespace ucuda::ui_detail {
void set_field_labels(const FieldLabels* labels) { g_field_labels = labels; }
}


std::vector<std::string> default_hidden_schemes() {
    static const char* const kShown[] = { "Euler", "Explicit Midpoint", "CD", "RK4", "DOPRI78", "GBS (n=2)" };
    std::vector<std::string> out;
    for (const auto& b : kBuiltinSchemes)
        if (std::find_if(std::begin(kShown), std::end(kShown),
                         [&](const char* s) { return std::strcmp(s, b.name) == 0; }) == std::end(kShown))
            out.push_back(b.name);
    return out;
}


std::vector<std::string> dae_method_list() {
    std::vector<std::string> v = dae_scheme_names();
    v.push_back(kBdfSchemeDae);
    return v;
}


std::vector<std::string> default_hidden_dae_methods() {
    std::vector<std::string> out;
    for (const std::string& n : dae_method_list())
        if (n != "DAE Trapezoidal" && n != kBdfSchemeDae) out.push_back(n);
    return out;
}


namespace ucuda::ui_detail {
static SystemLibrary* g_field_lib = nullptr;
}


namespace ucuda::ui_detail {
void set_field_library(SystemLibrary* lib) { g_field_lib = lib; }
}


namespace ucuda::ui_detail {
const SystemRecord* field_lib_record(const AppModel* m, bool reload) {
    static SystemRecord rec;
    static std::string name;
    static bool ok = false;
    if (!g_field_lib || !m || m->loaded_name.empty()) return nullptr;
    if (reload || name != m->loaded_name) {
        name = m->loaded_name;
        ok = false;
        try { rec = g_field_lib->load(name); ok = true; } catch (...) {}
    }
    return ok ? &rec : nullptr;
}
}


namespace ucuda::ui_detail {
bool field_restore_menu_items(const AppModel* model, BroadcastField f, const std::string& name,
                              std::map<std::string, std::string>& values, const std::vector<std::string>& names) {
    if (f != BroadcastField::Param && f != BroadcastField::InitCondition) return false;
    const SystemRecord* r = field_lib_record(model, ImGui::IsWindowAppearing());
    if (!r) {
        ImGui::BeginDisabled();
        ImGui::MenuItem("Restore from library (system not in the library)");
        ImGui::EndDisabled();
        return false;
    }
    const auto& src = f == BroadcastField::Param ? r->param_values : r->init_conditions;
    bool changed = false;
    {
        auto it = src.find(name);
        auto cur = values.find(name);
        const bool has = it != src.end();
        const bool same = has && cur != values.end() && cur->second == it->second;
        const std::string lbl = has ? "Restore from library (" + it->second + ")" : "Restore from library (no value)";
        if (ImGui::MenuItem(lbl.c_str(), nullptr, false, has && !same)) {
            values[name] = it->second;
            changed = true;
        }
    }
    int differ = 0;
    for (const std::string& n : names) {
        auto it = src.find(n);
        if (it == src.end()) continue;
        auto cur = values.find(n);
        if (cur == values.end() || cur->second != it->second) ++differ;
    }
    const std::string all = std::string("Restore all ")
                          + (f == BroadcastField::Param ? "parameters" : "initial conditions")
                          + " from library (" + std::to_string(differ) + " differ)";
    if (ImGui::MenuItem(all.c_str(), nullptr, false, differ > 0)) {
        for (const std::string& n : names) {
            auto it = src.find(n);
            if (it != src.end()) values[n] = it->second;
        }
        changed = true;
    }
    return changed;
}
}


namespace ucuda::ui_detail {
bool field_restore_popup(const char* popup_id, const AppModel* model, BroadcastField f, const std::string& name,
                         std::map<std::string, std::string>& values, const std::vector<std::string>& names) {
    if (!ImGui::BeginPopupContextItem(popup_id)) return false;
    const bool changed = field_restore_menu_items(model, f, name, values, names);
    ImGui::EndPopup();
    return changed;
}
}


namespace ucuda::ui_detail {
std::string field_label(const FieldLabels* labels, const std::string& name) {
    if (!labels) labels = g_field_labels;   // подписи системы, загруженной в модель
    if (labels) {
        auto it = labels->find(name);
        if (it != labels->end() && !it->second.empty()) return it->second;
    }
    return field_label_auto(name);
}
}


namespace ucuda::ui_detail {
MathExtent field_tex_extent(const std::string& tex, float fs) {
    static std::unordered_map<std::string, MathExtent> cache;
    const std::string key = tex + '\x01' + std::to_string((int)(fs * 8.0f));
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, math_formula_extent(tex.c_str(), fs)).first;
    return it->second;
}
}


namespace ucuda::ui_detail {
float field_tex_size() { return std::floor(ImGui::GetFontSize() * 1.2f + 0.5f); }
}


namespace ucuda::ui_detail {
void field_tex_draw(ImDrawList* dl, ImVec2 p, const std::string& tex, float box_h, bool fit ) {
    float fs = field_tex_size();
    MathExtent ext = field_tex_extent(tex, fs);
    float y = p.y + std::max(0.0f, (box_h - (ext.ascent + ext.descent)) * 0.5f);
    if (fit) {
        // Кегль UI-текста и его базовая линия: индекс уходит ниже строки, но в пределах отступа рамки.
        const float ui = ImGui::GetFontSize();
        const float base = p.y + (box_h - ui) * 0.5f + 0.8f * ui;   // ~ascent строки UI-шрифта
        y = std::max(p.y + 1.0f, base - ext.ascent);
    }
    math_formula_draw(dl, ImVec2(p.x, y), ImGui::GetColorU32(ImGuiCol_Text), tex.c_str(), fs);
}
}


namespace ucuda::ui_detail {
std::string g_combo_preview;
}


namespace ucuda::ui_detail {
void draw_field_name(const std::string& name, const FieldLabels* labels) {
    if (!g_field_latex) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(name.c_str());
        return;
    }
    const std::string tex = field_label(labels, name);
    const MathExtent ext = field_tex_extent(tex, field_tex_size());
    const float fh = ImGui::GetFrameHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(std::max(ext.width, 1.0f), std::max(fh, ext.ascent + ext.descent)));
    if (ImGui::IsItemVisible()) field_tex_draw(ImGui::GetWindowDrawList(), p, tex, fh);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
}
}


namespace ucuda::ui_detail {
bool var_combo_begin(const char* id, const std::string& cur, bool cur_is_name) {
    g_combo_preview.clear();
    if (!g_field_latex || !cur_is_name) return ImGui::BeginCombo(id, cur.c_str());
    g_combo_preview = field_label(nullptr, cur);
    return ImGui::BeginCombo(id, "", ImGuiComboFlags_CustomPreview);
}
}


namespace ucuda::ui_detail {
void var_combo_preview() {
    if (g_combo_preview.empty()) return;
    const std::string tex = std::move(g_combo_preview);
    g_combo_preview.clear();
    if (ImGui::BeginComboPreview()) {
        // Курсор превью стоит на отступе рамки; формулу вписываем во всю высоту рамки.
        const ImVec2 c = ImGui::GetCursorScreenPos();
        field_tex_draw(ImGui::GetWindowDrawList(), ImVec2(c.x, c.y - ImGui::GetStyle().FramePadding.y), tex,
                       ImGui::GetFrameHeight(), true);
        ImGui::EndComboPreview();
    }
}
}


namespace ucuda::ui_detail {
bool var_selectable(const std::string& name, bool selected) {
    if (!g_field_latex) return ImGui::Selectable(name.c_str(), selected);
    const std::string tex = field_label(nullptr, name);
    const MathExtent ext = field_tex_extent(tex, field_tex_size());
    const float h = std::max(ImGui::GetTextLineHeight(), ext.ascent + ext.descent);
    const bool clicked = ImGui::Selectable(("##vs_" + name).c_str(), selected, 0, ImVec2(std::max(ext.width, 1.0f), h));
    if (ImGui::IsItemVisible()) field_tex_draw(ImGui::GetWindowDrawList(), ImGui::GetItemRectMin(), tex, h);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name.c_str());
    return clicked;
}
}


namespace ucuda::ui_detail {
std::string plot_var_name(const std::string& name) {
    return g_field_latex && plot_math_enabled() ? field_label(nullptr, name) : name;
}
}


namespace ucuda::ui_detail {
void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note,
                                  AppModel* bc,
                                  BroadcastField field,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten) {
    if (!ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) return;
    if (note) ImGui::TextDisabled("%s", note);
    // Группы и подписи — у системы, загруженной в модель (bc); без модели — плоский список имён.
    ParamGroups* groups = !bc ? nullptr
                        : field == BroadcastField::Param ? &bc->param_groups
                        : field == BroadcastField::InitCondition ? &bc->ic_groups : nullptr;
    const FieldLabels* labels = bc ? &bc->field_labels : nullptr;
    int hidden = 0;
    const std::vector<std::string> shown = bc && field == BroadcastField::InitCondition ? bc->ic_fields(names, &hidden) : names;
    if (hidden > 0) ImGui::TextDisabled("%d algebraic variables hidden (Settings)", hidden);
    draw_param_tree(groups, shown, [&](const std::string& n) {
        ImGui::PushID(n.c_str());
        const bool off = overwritten && overwritten(n);
        if (off) ImGui::BeginDisabled();
        InputNumStr("##v", values[n], kFieldW,
                    [&] { field_apply_all_menu(bc, field, n, n, values[n],
                                               [&] { field_restore_menu_items(bc, field, n, values, names); }); });
        ImGui::SameLine();
        draw_field_name(n, labels);
        if (off) ImGui::EndDisabled();
        ImGui::PopID();
    });
}
}


namespace ucuda::ui_detail {
int pg_count(PgNode& n, const std::set<std::string>& present) {
    n.count = 0;
    if (n.g) for (const std::string& p : n.g->params) n.count += present.count(p) ? 1 : 0;
    for (PgNode& k : n.kids) n.count += pg_count(k, present);
    return n.count;
}
}


namespace ucuda::ui_detail {
constexpr const char* kPgDragType = "UCUDA_PG_NODE";
}


namespace ucuda::ui_detail {
std::string pg_payload(const ParamGroups* list, const std::string& path) {
    std::string s(sizeof(list), '\0');
    std::memcpy(&s[0], &list, sizeof(list));
    return s + path;
}
}


namespace ucuda::ui_detail {
void pg_drag_node(ParamGroups* list, const std::string& path, const std::string& shown, PgDrop& drop) {
    if (!list) return;
    if (ImGui::BeginDragDropSource()) {
        const std::string pl = pg_payload(list, path);
        ImGui::SetDragDropPayload(kPgDragType, pl.data(), pl.size());
        ImGui::TextUnformatted(shown.c_str());
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        const ImGuiPayload* pk = ImGui::GetDragDropPayload();
        if (pk && pk->IsDataType(kPgDragType) && pk->DataSize > (int)sizeof(list)) {
            const ParamGroups* src = nullptr;
            std::memcpy(&src, pk->Data, sizeof(src));
            const std::string from((const char*)pk->Data + sizeof(src), (size_t)pk->DataSize - sizeof(src));
            if (src == list && from != path && pg_parent(from) == pg_parent(path)) {
                const ImRect r(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
                const bool after = ImGui::GetMousePos().y > (r.Min.y + r.Max.y) * 0.5f;
                // Линия вставки вместо подсветки всего узла.
                const float y = after ? r.Max.y : r.Min.y;
                ImGui::GetWindowDrawList()->AddLine(ImVec2(r.Min.x, y), ImVec2(r.Max.x, y),
                                                    ImGui::GetColorU32(ImGuiCol_DragDropTarget), 2.0f);
                if (ImGui::AcceptDragDropPayload(kPgDragType, ImGuiDragDropFlags_AcceptNoDrawDefaultRect))
                    drop = PgDrop{ from, path, after };
            }
        }
        ImGui::EndDragDropTarget();
    }
}
}


namespace ucuda::ui_detail {
void pg_collect(const PgNode& n, const std::set<std::string>& present, std::vector<std::string>& out) {
    if (n.g) for (const std::string& p : n.g->params) if (present.count(p)) out.push_back(p);
    for (const PgNode& k : n.kids) pg_collect(k, present, out);
}
}


namespace ucuda::ui_detail {
bool pg_header(const std::string& label, const std::string& id, bool in_table, ParamGroups* list,
               const std::string& seg, PgDrop& drop, const PgMenu& menu,
               const std::function<std::vector<std::string>()>& names_of) {
    if (in_table) { ImGui::TableNextRow(); ImGui::TableSetColumnIndex(0); }
    const ImGuiTreeNodeFlags fl = in_table ? ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_LabelSpanAllColumns
                                           : ImGuiTreeNodeFlags_SpanAvailWidth;
    const bool open = ImGui::TreeNodeEx((label + "##pg_" + id).c_str(), fl);
    pg_drag_node(list, id, seg, drop);
    if (menu && ImGui::BeginPopupContextItem(("##pgm_" + id).c_str())) {
        menu(names_of());
        ImGui::EndPopup();
    }
    return open;
}
}


namespace ucuda::ui_detail {
void pg_draw(const PgNode& n, const std::set<std::string>& present,
             const std::function<void(const std::string&)>& item, bool in_table, ParamGroups* list, PgDrop& drop,
             const PgMenu& menu) {
    if (n.g)
        for (const std::string& p : n.g->params)
            if (present.count(p)) item(p);
    for (const PgNode& k : n.kids) {
        if (k.count == 0) continue;
        auto names_of = [&] { std::vector<std::string> v; pg_collect(k, present, v); return v; };
        if (pg_header(k.seg + "  (" + std::to_string(k.count) + ")", k.path, in_table, list, k.seg, drop, menu,
                      names_of)) {
            pg_draw(k, present, item, in_table, list, drop, menu);
            ImGui::TreePop();
        }
    }
}
}


namespace ucuda::ui_detail {
void draw_param_tree(ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table,
                     const std::function<void(const std::vector<std::string>&)>& group_menu) {
    if (groups == nullptr || groups->empty()) {
        for (const std::string& n : names) item(n);
        return;
    }
    const std::set<std::string> present(names.begin(), names.end());
    PgNode root;
    std::set<std::string> grouped;
    for (const ParamGroup& g : *groups) {
        PgNode* cur = &root;
        size_t pos = 0;
        while (pos <= g.path.size()) {
            size_t e = g.path.find('/', pos);
            if (e == std::string::npos) e = g.path.size();
            const std::string seg = g.path.substr(pos, e - pos);
            pos = e + 1;
            auto it = std::find_if(cur->kids.begin(), cur->kids.end(), [&](const PgNode& k) { return k.seg == seg; });
            if (it == cur->kids.end()) {
                PgNode k;
                k.seg = seg;
                k.path = g.path.substr(0, e);
                cur->kids.push_back(std::move(k));
                it = cur->kids.end() - 1;
            }
            cur = &*it;
        }
        cur->g = &g;
        grouped.insert(g.params.begin(), g.params.end());
    }
    if (pg_count(root, present) == 0) {   // ни одна группа к этим именам не относится
        for (const std::string& n : names) item(n);
        return;
    }
    // У параметров и НУ бывают одноимённые группы ("Capacitors") в одном окне — id по списку.
    ImGui::PushID(groups);
    PgDrop drop;
    pg_draw(root, present, item, in_table, groups, drop, group_menu);
    std::vector<std::string> other;
    for (const std::string& n : names) if (!grouped.count(n)) other.push_back(n);
    if (!other.empty() && pg_header("Other  (" + std::to_string(other.size()) + ")", "\x01other", in_table,
                                    nullptr, "Other", drop, group_menu, [&] { return other; })) {
        for (const std::string& n : other) item(n);
        ImGui::TreePop();
    }
    ImGui::PopID();
    // Узлы держат указатели в groups — менять порядок только после отрисовки.
    if (!drop.from.empty()) pg_reorder(*groups, drop.from, drop.to, drop.after);
}
}


namespace ucuda::ui_detail {
void draw_csv_output_block(const char* header, const char* checkbox_label,
                                  bool& enabled, const char* path_id, std::string& path,
                                  const char* hint) {
    if (!ImGui::CollapsingHeader(header)) return;
    ImGui::Checkbox(checkbox_label, &enabled);
    InputTextStr(path_id, path);
    ImGui::TextDisabled("%s", hint);
}
}


namespace ucuda::ui_detail {
void draw_error_box(const char* id, const std::string& err,
                           int lines) {
    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error (selectable, Ctrl+C):");
    const ImVec2 sz(-1.0f, ImGui::GetTextLineHeight() * (float)lines);
    ImGui::InputTextMultiline(id, const_cast<char*>(err.c_str()), err.size() + 1,
                              sz, ImGuiInputTextFlags_ReadOnly);
}
}


namespace ucuda::ui_detail {
void draw_export_menu_item(bool busy, const GuiCallbacks& cb,
                                  const std::function<void(const std::string&)>& write) {
    if (!ImGui::MenuItem("Export data...", nullptr, false, !busy)) return;
    if (!cb.pick_save_file_csv) return;
    const std::string path = cb.pick_save_file_csv();
    if (!path.empty() && write) write(path);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeTarget vr_target_fixed(std::string name,
        std::string* x_lo, std::string* x_hi,
        std::string* y_lo, std::string* y_hi) {
    ViewRangeTarget t;
    t.name = std::move(name);
    t.x_lo = x_lo; t.x_hi = x_hi; t.y_lo = y_lo; t.y_hi = y_hi;
    t.relocate = [x_lo, x_hi, y_lo, y_hi]() {
        ViewRangeSlots s;
        s.x_lo = x_lo; s.x_hi = x_hi; s.y_lo = y_lo; s.y_hi = y_hi;
        return s;
    };
    return t;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string fmt_view_bound(double v, double span) {
    const double tol = std::abs(span) * 1e-6;
    if (tol > 0.0) {
        char buf[64];
        for (int prec = 6; prec <= 17; ++prec) {
            std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
            if (std::abs(parse_ratio_or(buf, v) - v) <= tol) return std::string(buf);
        }
    }
    return fmt_num_shortest(v);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<ViewRangeTarget> all_view_range_targets(AppModel& model) {
    std::vector<ViewRangeTarget> out;
    auto add = [&out](ViewRangeTarget t, const char* kind, const std::string& group) {
        if (!t.relocate || (!t.x_lo && !t.y_lo)) return;   // цель без полей не нужна
        t.kind  = kind;
        t.group = group;
        out.push_back(std::move(t));
    };

    using B = BifurcationDiagramConfig;
    auto& bd = model.bifurcation_session.diagrams;
    for (int i = 0; i < (int)bd.size(); ++i) {
        if (bd[(size_t)i].mode_2d)
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::param_lo_2_text, &B::param_hi_2_text),
                "Parametric 2D", "Bifurcation 2D");
        else if (bd[(size_t)i].colored_1d)
            // Colored 1D: по Y окно гистограммы, и запись туда включает
            // ручной режим — ровно как на самой диаграмме.
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text,
                          &B::colored_1d_ymin_text, &B::colored_1d_ymax_text,
                          &B::colored_1d_custom_y),
                "Parametric 1D", "Bifurcation colored 1D");
        else
            add(vr_target(bd, i, &B::param_lo_text, &B::param_hi_text),
                "Parametric 1D", "Bifurcation 1D");
    }

    using L = LLECurveConfig;
    auto& lle = model.lle_session.curves;
    for (int i = 0; i < (int)lle.size(); ++i) {
        if (lle[(size_t)i].mode_2d)
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text,
                          &L::param_lo_2_text, &L::param_hi_2_text),
                "Parametric 2D", "LLE 2D");
        else
            add(vr_target(lle, i, &L::param_lo_text, &L::param_hi_text),
                "Parametric 1D", "LLE 1D");
    }

    using S = LSCurveConfig;
    auto& ls = model.ls_session.curves;
    for (int i = 0; i < (int)ls.size(); ++i) {
        if (ls[(size_t)i].mode_2d)
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text,
                          &S::param_lo_2_text, &S::param_hi_2_text),
                "Parametric 2D", "LS 2D");
        else
            add(vr_target(ls, i, &S::param_lo_text, &S::param_hi_text),
                "Parametric 1D", "LS 1D");
    }

    using M = SignalMetricsConfig;
    auto& met = model.metrics_session.configs;
    for (int i = 0; i < (int)met.size(); ++i) {
        if (met[(size_t)i].mode_2d)
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text,
                          &M::param_lo_2_text, &M::param_hi_2_text),
                "Parametric 2D", "Metrics 2D");
        else
            add(vr_target(met, i, &M::param_lo_text, &M::param_hi_text),
                "Parametric 1D", "Metrics 1D");
    }

    // DFT — свой тип: по X свип параметра, а по Y частоты, и подставлять туда
    // диапазон параметра со второй оси 2D-карты нельзя.
    using D = Dft1DConfig;
    auto& dft = model.dft1d_session.configs;
    for (int i = 0; i < (int)dft.size(); ++i)
        add(vr_target(dft, i, &D::param_lo_text, &D::param_hi_text,
                      &D::freq_lo_text, &D::freq_hi_text),
            "1D DFT", "1D DFT");

    // Basins и FastSync — сетка начальных условий, а не параметрический свип.
    using BA = BasinsConfig;
    auto& bas = model.basins_session.configs;
    for (int i = 0; i < (int)bas.size(); ++i)
        add(vr_target(bas, i, &BA::axis_x_lo_text, &BA::axis_x_hi_text,
                      &BA::axis_y_lo_text, &BA::axis_y_hi_text),
            "IC grid", "Basins");

    using F = FastSyncConfig;
    auto& fs = model.fastsync_session.configs;
    for (int i = 0; i < (int)fs.size(); ++i)
        add(vr_target(fs, i, &F::axis_x_lo_text, &F::axis_x_hi_text,
                      &F::axis_y_lo_text, &F::axis_y_hi_text),
            "IC grid", "Fast Synchro");

    using O = OrderConfig;
    auto& ord = model.order_session.configs;
    for (int i = 0; i < (int)ord.size(); ++i) {
        if (ord[(size_t)i].two_d)
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text,
                          &O::axis_y_lo_text, &O::axis_y_hi_text),
                "Order 2D", "Order 2D");
        else
            add(vr_target(ord, i, &O::axis_x_lo_text, &O::axis_x_hi_text),
                "Order 1D", "Order 1D");
    }

    add(vr_target_fixed("Shared config",
                        &model.custom_session.shared.axis_x_lo_text,
                        &model.custom_session.shared.axis_x_hi_text,
                        &model.custom_session.shared.axis_y_lo_text,
                        &model.custom_session.shared.axis_y_hi_text),
        "Custom", "Custom");
    return out;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::vector<ViewRangeTarget>
view_range_same_kind(AppModel& model, const ViewRangeTarget& source) {
    std::vector<ViewRangeTarget> all = all_view_range_targets(model);
    std::string kind;
    for (const auto& t : all)
        if (t.x_lo && t.x_lo == source.x_lo) { kind = t.kind; break; }
    std::vector<ViewRangeTarget> out;
    if (kind.empty()) return out;
    for (auto& t : all)
        if (t.kind == kind) {
            if (t.x_lo == source.x_lo) out.insert(out.begin(), std::move(t));
            else                       out.push_back(std::move(t));
        }
    return out;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeValues values_from_axes(const AxisInfo& x, const AxisInfo& y,
                                                      bool swapped, bool src_has_y) {
    const AxisInfo& ax = swapped ? y : x;   // ось ДАННЫХ X
    const AxisInfo& ay = swapped ? x : y;   // ось ДАННЫХ Y
    // invert — только про отрисовку, view_min/view_max остаются упорядоченными;
    // min/max здесь страховка от конфига, пришедшего из чужой руки.
    ViewRangeValues v;
    // ±log-ось (бассейны): вид живёт в координатах узлов, а значения по ним идут немонотонно
    // (-hi..-lo, +lo..+hi) — диапазоном [lo, hi] его не записать, такую ось не трогаем.
    if (!ax.symlog) {
        const double span = ax.view_max - ax.view_min;
        v.has_x = true;
        v.x_lo = fmt_view_bound((std::min)(ax.view_min, ax.view_max), span);
        v.x_hi = fmt_view_bound((std::max)(ax.view_min, ax.view_max), span);
    }
    if (src_has_y && !ay.symlog) {
        const double span = ay.view_max - ay.view_min;
        v.has_y = true;
        v.y_lo = fmt_view_bound((std::min)(ay.view_min, ay.view_max), span);
        v.y_hi = fmt_view_bound((std::max)(ay.view_min, ay.view_max), span);
    }
    return v;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] ViewRangeValues values_from_target(const ViewRangeTarget& t) {
    ViewRangeValues v;
    v.source = "fields";
    if (t.x_lo && t.x_hi) { v.has_x = true; v.x_lo = *t.x_lo; v.x_hi = *t.x_hi; }
    if (t.y_lo && t.y_hi) { v.has_y = true; v.y_lo = *t.y_lo; v.y_hi = *t.y_hi; }
    return v;
}
}


namespace ucuda::ui_detail {
void apply_view_values(const ViewRangeTarget& t, const ViewRangeValues& v) {
    if (v.has_x && t.x_lo && t.x_hi) { *t.x_lo = v.x_lo; *t.x_hi = v.x_hi; }
    if (v.has_y && t.y_lo && t.y_hi) {
        *t.y_lo = v.y_lo; *t.y_hi = v.y_hi;
        if (t.y_manual) *t.y_manual = true;
    }
}
}


namespace ucuda::ui_detail {
void run_view_range_cmd(const std::vector<ViewRangeTarget>& targets, size_t count,
                               const ViewRangeValues& vals, AppModel* model) {
    if (targets.empty()) return;
    count = (std::min)(count, targets.size());

    // Снимок полей цели. Держим не адреса, а способ найти их заново.
    struct Snap {
        std::function<ViewRangeSlots()> relocate;
        bool has_x = false, has_y = false, has_flag = false;
        std::string x_lo, x_hi, y_lo, y_hi;
        bool flag = false;
    };
    auto capture = [](const ViewRangeTarget& t) {
        Snap p;
        p.relocate = t.relocate;
        if (t.x_lo && t.x_hi) { p.has_x = true; p.x_lo = *t.x_lo; p.x_hi = *t.x_hi; }
        if (t.y_lo && t.y_hi) { p.has_y = true; p.y_lo = *t.y_lo; p.y_hi = *t.y_hi; }
        if (t.y_manual)       { p.has_flag = true; p.flag = *t.y_manual; }
        return p;
    };
    // Отмена и повтор — одна процедура, отличаются только снимком.
    auto writer = [](std::vector<Snap> snaps) {
        return [snaps = std::move(snaps)]() {
            for (const auto& p : snaps) {
                const ViewRangeSlots s = p.relocate();
                if (p.has_x && s.x_lo && s.x_hi) { *s.x_lo = p.x_lo; *s.x_hi = p.x_hi; }
                if (p.has_y && s.y_lo && s.y_hi) { *s.y_lo = p.y_lo; *s.y_hi = p.y_hi; }
                if (p.has_flag && s.y_manual)    *s.y_manual = p.flag;
            }
        };
    };

    std::vector<Snap> before, after;
    for (size_t i = 0; i < count; ++i) {
        const ViewRangeTarget& t = targets[i];
        Snap p = capture(t);
        // Пишем ТОЛЬКО то, что умеем вернуть: цель без relocate откатить нечем,
        // и запись в неё была бы дырой в отмене — поля изменились, а Ctrl+Z про
        // них не знает. Сегодня такие цели не появляются (обе фабрики relocate
        // ставят), но инвариант лучше держать кодом, чем соглашением.
        if (!p.relocate || (!p.has_x && !p.has_y)) continue;
        before.push_back(std::move(p));
        apply_view_values(t, vals);
    }
    if (!model || before.empty()) return;
    // "После" снимаем по тем же целям и уже после записи — так повтор кладёт
    // ровно то, что команда посчитала, без второго прохода по осям.
    for (size_t i = 0; i < count; ++i) {
        Snap p = capture(targets[i]);
        if (p.relocate && (p.has_x || p.has_y)) after.push_back(std::move(p));
    }

    std::string what = std::string("calculation range from ") + vals.source;
    if (count > 1)                     what += " (" + std::to_string((int)count) + " diagrams)";
    else if (!targets[0].name.empty()) what += " (" + targets[0].name + ")";
    model->push_undo(what, writer(std::move(before)), writer(std::move(after)));
    // Отчёт в шапку — как у рассылки полей. Команда правит и диаграммы в других
    // окнах, которые сейчас не видны, и без строки это неотличимо от "ничего не
    // произошло".
    model->undo_note = "Applied: " + what;
    model->undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool view_range_bound(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && (targets[0].x_lo != nullptr || targets[0].y_lo != nullptr);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool src_y_is_input(const std::vector<ViewRangeTarget>& targets) {
    return !targets.empty() && targets[0].y_lo != nullptr;
}
}


namespace ucuda::ui_detail {
void draw_view_range_menu(const std::vector<ViewRangeTarget>& targets,
                                 const AxisInfo& x, const AxisInfo& y,
                                 bool swapped, AppModel* model) {
    if (!view_range_bound(targets)) { draw_undo_menu_item(model); return; }
    ImGui::Separator();
    std::string one = "Calculation range from view";
    if (targets.size() > 1) one += " (" + targets[0].name + ")";
    if (ImGui::MenuItem(one.c_str(), "Ctrl+T"))
        run_view_range_cmd(targets, 1,
                           values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Writes the visible axis range into this diagram's sweep\n"
                          "range fields. Press Run to recompute the zoomed region\n"
                          "at full resolution.");
    // Ctrl+Shift+T — все диаграммы ТОГО ЖЕ типа, а не только наложенные в этом
    // окне: приблизив 2D-бифуркацию, обычно хотят увидеть тот же участок и на
    // 2D LLE с 2D LS. Набор считается по общему перечислению (см.
    // view_range_same_kind), поэтому в него попадают и другие окна.
    if (model) {
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        if (kin.size() > 1) {
            // Сразу после Ctrl+Z / Ctrl+Shift+Z источником становятся ПОЛЯ, а
            // не вид: поля только что откатили, картинка осталась зумленной, и
            // разослать "из вида" значило бы отменить отмену. Пишем это прямо
            // в пункте меню — угадывать источник пользователь не должен.
            const bool from_fields = model->undo_just_happened();
            char lbl[160];
            std::snprintf(lbl, sizeof(lbl), "Calculation range from %s - all %s (%d)",
                          from_fields ? "fields" : "view",
                          kin[0].kind.c_str(), (int)kin.size());
            if (ImGui::MenuItem(lbl, "Ctrl+Shift+T")) {
                const ViewRangeValues v = from_fields
                        ? values_from_target(kin[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(kin, kin.size(), v, model);
            }
            if (from_fields && ImGui::IsItemHovered())
                ImGui::SetTooltip("Undo just restored the sweep fields, so this sends\n"
                                  "THOSE values on. The plot is still zoomed; sending the\n"
                                  "view instead would undo the undo.");
        }

        // Точный выбор галочками — для случаев, когда "тот же тип" не тот набор:
        // например, подтянуть к 2D-карте ещё и Basins, у которых оси другого
        // смысла и в автоматический набор они не входят.
        std::vector<ViewRangeTarget> all = all_view_range_targets(*model);
        if (!all.empty() && ImGui::BeginMenu("Calculation range to...")) {
            std::string sig;
            for (const auto& t : all) { sig += '\x1f'; sig += t.group; sig += t.name; }
            if (model->vr_pick_sig != sig || model->vr_pick.size() != all.size()) {
                model->vr_pick.assign(all.size(), '0');
                model->vr_pick_sig = sig;
                // По умолчанию отмечаем свой тип: это и подсказка, что здесь
                // выбирают, и самый частый набор.
                for (size_t i = 0; i < all.size(); ++i)
                    for (const auto& k : kin)
                        if (all[i].x_lo && all[i].x_lo == k.x_lo) model->vr_pick[i] = '1';
            }
            if (ImGui::SmallButton("All"))  model->vr_pick.assign(all.size(), '1');
            ImGui::SameLine();
            if (ImGui::SmallButton("None")) model->vr_pick.assign(all.size(), '0');

            std::vector<ViewRangeTarget> sel;
            for (size_t i = 0; i < all.size(); ++i)
                if (model->vr_pick[i] == '1') sel.push_back(all[i]);

            char btn[96];
            std::snprintf(btn, sizeof(btn), "Apply to %d selected", (int)sel.size());
            if (sel.empty()) ImGui::BeginDisabled();
            if (ImGui::Button(btn)) {
                const ViewRangeValues v = model->undo_just_happened()
                        ? values_from_target(targets[0])
                        : values_from_axes(x, y, swapped, src_y_is_input(targets));
                run_view_range_cmd(sel, sel.size(), v, model);
                ImGui::CloseCurrentPopup();
            }
            if (sel.empty()) ImGui::EndDisabled();
            ImGui::Separator();

            std::string shown_group;
            for (size_t i = 0; i < all.size(); ++i) {
                if (all[i].group != shown_group) {
                    shown_group = all[i].group;
                    ImGui::SeparatorText(shown_group.c_str());
                }
                ImGui::PushID((int)i);
                bool on = (model->vr_pick[i] == '1');
                if (ImGui::Checkbox("##pick", &on))
                    model->vr_pick[i] = on ? '1' : '0';
                ImGui::SameLine();
                ImGui::TextUnformatted(all[i].name.c_str());
                ImGui::PopID();
            }
            ImGui::EndMenu();
        }
    }
    draw_undo_menu_item(model);
}
}


namespace ucuda::ui_detail {
void handle_view_range_keys(const std::vector<ViewRangeTarget>& targets,
                                   const AxisInfo& x, const AxisInfo& y, bool active,
                                   bool swapped, AppModel* model) {   // без умолчаний — см. draw_view_range_menu
    if (!active || !view_range_bound(targets)) return;
    const ImGuiIO& io = ImGui::GetIO();
    // С Alt не реагируем вовсе: Ctrl+Alt+T — не наше сочетание, и тихо
    // трактовать его как Ctrl+T значит переписать диапазон по случайному нажатию.
    if (!io.KeyCtrl || io.KeyAlt || io.WantTextInput) return;
    if (!ImGui::IsKeyPressed(ImGuiKey_T, false)) return;
    if (io.KeyShift && model) {
        // Ctrl+Shift+T — весь тип целиком (см. draw_view_range_menu). Если
        // источник в общем перечислении не нашёлся, падаем на прежнее
        // поведение — наложенные в этом окне диаграммы.
        std::vector<ViewRangeTarget> kin = view_range_same_kind(*model, targets[0]);
        const std::vector<ViewRangeTarget>& dst = kin.empty() ? targets : kin;
        // Сразу после отката источник — поля, а не вид (см. меню).
        const ViewRangeValues v = model->undo_just_happened()
                                ? values_from_target(dst[0])
                                : values_from_axes(x, y, swapped, src_y_is_input(targets));
        run_view_range_cmd(dst, dst.size(), v, model);
        return;
    }
    run_view_range_cmd(targets, 1,
                       values_from_axes(x, y, swapped, src_y_is_input(targets)), model);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool plot_window_active() {
    return ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows |
                                  ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
}
}


namespace ucuda::ui_detail {
void draw_label_rename(const char* label_with_id, std::string& label,
                              float width) {
    ImGui::SetNextItemWidth(width);
    InputTextStr(label_with_id, label);
}
}


namespace ucuda::ui_detail {
void track_dock_selection(AppModel& model) {
    const int mode = (int)model.app_mode;
    std::vector<std::string>& mem = model.dock_selected_by_mode[mode];
    if (model.dock_prev_mode != mode) {
        model.dock_prev_mode      = mode;
        model.dock_restore_frames = 3;
    }
    if (model.dock_restore_frames > 0) {
        --model.dock_restore_frames;
        for (const std::string& name : mem) {
            ImGuiWindow* w = ImGui::FindWindowByName(name.c_str());
            if (w == nullptr || !w->Active || w->DockNode == nullptr || w->DockNode->TabBar == nullptr) continue;
            ImGuiTabBar* tb = w->DockNode->TabBar;
            if (tb->SelectedTabId != w->TabId && ImGui::TabBarFindTabByID(tb, w->TabId) != nullptr)
                tb->NextSelectedTabId = w->TabId;
        }
        return;
    }
    mem.clear();
    for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows)
        if (w->Active && w->DockIsActive && w->DockNode != nullptr && w->DockNode->TabBar != nullptr
            && w->DockNode->TabBar->SelectedTabId == w->TabId)
            mem.push_back(w->Name);
}
}


namespace ucuda::ui_detail {
TabBarResult draw_config_tab_bar(const char* bar_id, const char* item_id_prefix,
                                        int n, bool in_flight, int running_index,
                                        const std::function<std::string(int)>& label,
                                        const std::function<void(int)>& body,
                                        const std::function<void()>& on_add,
                                        int request_select) {
    TabBarResult r;
    // Выбор после закрытия вкладки (между кадрами — в хранилище ImGui, ключ от bar_id).
    // Под номер закрытой вкладки встаёт следующий конфиг, ImGui видит его как новую вкладку
    // и по AutoSelectNewTabs выбрал бы его, даже если закрывали не активную.
    ImGuiStorage* st = ImGui::GetStateStorage();
    const ImGuiID pending_key = ImGui::GetID(bar_id) ^ 0x5e1ec7u;
    const int pending = st->GetInt(pending_key, -1);
    if (pending >= 0) {
        st->SetInt(pending_key, -1);
        if (request_select < 0 && pending < n) request_select = pending;
    }
    if (!ImGui::BeginTabBar(bar_id, ImGuiTabBarFlags_AutoSelectNewTabs |
                                    ImGuiTabBarFlags_FittingPolicyScroll))
        return r;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        bool open = true;
        // ID завязан на индекс, а видимая часть — на label: пользователь видит
        // свежее имя сразу после редактирования.
        const std::string tab_id = label(i) + "###" + item_id_prefix + std::to_string(i);
        // Запрещаем закрывать вкладку, чей расчёт сейчас идёт, и ЕДИНСТВЕННУЮ
        // вкладку. Пустой список ни одна сессия не переживает: загрузчики при
        // следующем входе (перезапуск, смена системы) кладут дефолтный конфиг,
        // "+" на пустом списке — тоже дефолт, и настройки последней вкладки
        // терялись. Так же давно устроен NetworkSession::remove_config.
        const bool can_close = !(in_flight && running_index == i) && n > 1;
        ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
        if (request_select == i) flags |= ImGuiTabItemFlags_SetSelected;
        if (ImGui::BeginTabItem(tab_id.c_str(), can_close ? &open : nullptr, flags)) {
            r.active = i;
            if (body) body(i);
            ImGui::EndTabItem();
        }
        if (!open) r.to_remove = i;
        ImGui::PopID();
    }
    // Кнопка "+" справа: Trailing удерживает её в конце, NoTooltip убирает
    // дефолтную подсказку.
    if (!in_flight && on_add) {
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing |
                                      ImGuiTabItemFlags_NoTooltip))
            on_add();
    }
    ImGui::EndTabBar();
    if (r.to_remove >= 0) {
        // Активная остаётся активной (её номер сдвигается, если она правее закрытой);
        // закрыли саму активную — выбрать ту, что встала на её место (или последнюю).
        const int a = r.active, i = r.to_remove;
        const int next = (a > i) ? a - 1 : (a == i || a < 0) ? std::min(i, n - 2) : a;
        st->SetInt(pending_key, next);
    }
    return r;
}
}


namespace ucuda::ui_detail {
void draw_run_and_run_all(const char* popup_id,
                                 bool busy, bool no_active, bool block_run_all,
                                 float run_w,
                                 const std::function<void()>& on_run,
                                 const std::function<void()>& extra_after_run,
                                 const std::vector<RunAllGroup>& groups,
                                 const std::function<void()>& start_queue,
                                 size_t queued) {
    const ImGuiIO& io = ImGui::GetIO();

    bool do_run = false;
    if (busy) {
        ImGui::BeginDisabled();
        ImGui::Button("Running...", ImVec2(run_w, 0));
        ImGui::EndDisabled();
    } else {
        if (no_active) ImGui::BeginDisabled();
        do_run = ImGui::Button("Run (Ctrl+R)", ImVec2(run_w, 0));
        if (no_active) ImGui::EndDisabled();
    }
    if (!busy && !no_active && io.KeyCtrl && !io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        do_run = true;
    if (do_run && on_run) on_run();

    if (extra_after_run) extra_after_run();

    auto& picks_by_popup = ui_resource<std::map<std::string, std::vector<std::vector<bool>>>, 3023>();
    std::vector<std::vector<bool>>& picks = picks_by_popup[popup_id];
    if (picks.size() != groups.size()) picks.assign(groups.size(), {});
    for (size_t g = 0; g < groups.size(); ++g)
        if (picks[g].size() != (size_t)groups[g].n)
            picks[g].assign((size_t)groups[g].n, true);

    auto run_all_marked = [&]() {
        for (size_t g = 0; g < groups.size(); ++g)
            for (size_t i = 0; i < picks[g].size(); ++i)
                if (picks[g][i] && groups[g].enqueue) groups[g].enqueue((int)i);
        if (start_queue) start_queue();
    };
    if (!block_run_all && io.KeyCtrl && io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_R, false))
        run_all_marked();

    ImGui::SameLine();
    if (block_run_all) ImGui::BeginDisabled();
    if (ImGui::Button("Run all... (Ctrl+Shift+R)")) ImGui::OpenPopup(popup_id);
    if (block_run_all) ImGui::EndDisabled();
    if (queued > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu queued)", queued);
    }
    if (ImGui::BeginPopup(popup_id)) {
        ImGui::TextDisabled("Sequential (one CUDA context).");
        for (size_t g = 0; g < groups.size(); ++g) {
            if (picks[g].empty()) continue;
            if (groups[g].title) ImGui::SeparatorText(groups[g].title);
            for (size_t i = 0; i < picks[g].size(); ++i) {
                bool v = picks[g][i];
                const std::string lbl = groups[g].label((int)i) + "###" +
                                        groups[g].pick_id + std::to_string(i);
                if (ImGui::Checkbox(lbl.c_str(), &v)) picks[g][i] = v;
            }
        }
        ImGui::Separator();
        if (ImGui::Button("All"))  { for (auto& p : picks) for (auto&& b : p) b = true;  }
        ImGui::SameLine();
        if (ImGui::Button("None")) { for (auto& p : picks) for (auto&& b : p) b = false; }
        ImGui::SameLine();
        if (ImGui::Button("Run")) {
            run_all_marked();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}
}


namespace ucuda::ui_detail {
void configure_param_plot_view(Plot2DView& view, ParamPlotKind kind) {
    view.pad_x       = false;   // данные вплотную к боковым рамкам
    view.show_zero_x = false;
    switch (kind) {
    case ParamPlotKind::Bifurcation:
        view.points_mode       = true;
        view.point_size_px     = 2.0f;
        view.imdraw_lines      = false;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "X";
        break;
    case ParamPlotKind::LLE:
    case ParamPlotKind::LS:
        view.points_mode       = false;   // непрерывная линия
        view.line_thickness_px = 1.5f;
        // glLineWidth в core-profile клампится драйвером до 1px, поэтому
        // линии идут через ImDrawList (см. Plot2DView::imdraw_lines).
        view.imdraw_lines      = true;
        view.show_zero_y       = true;    // λ=0 — граница хаос/порядок
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "lambda";
        break;
    case ParamPlotKind::Metrics:
        // Линия, как у LLE, но без нулевой линии: у амплитуд и частот ноль
        // ничего не разделяет. Имя Y ставит draw_metrics_plot по выбранной метрике.
        view.points_mode       = false;
        view.line_thickness_px = 1.5f;
        view.imdraw_lines      = true;
        view.show_zero_y       = false;
        view.x_axis.name       = "parameter";
        view.y_axis.name       = "metric";
        break;
    }
}
}


namespace ucuda::ui_detail {
bool draw_style_toolbar(const char* label, const char* id_suffix,
                               bool& enabled,
                               const std::function<bool()>& body) {
    const std::string sfx = id_suffix;
    bool changed = false;
    if (ImGui::Checkbox((std::string(label) + "##" + sfx).c_str(), &enabled))
        changed = true;

    const std::string pop_id = "##stylepop_" + sfx;
    if (enabled) {
        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::ArrowButton(("##styleopen_" + sfx).c_str(), ImGuiDir_Down))
            ImGui::OpenPopup(pop_id.c_str());
    }
    if (ImGui::BeginPopup(pop_id.c_str())) {
        if (body) changed |= body();
        ImGui::EndPopup();
    }
    return changed;
}
}


namespace ucuda::ui_detail {
void draw_proj_line_style(Projection& pr, const char* sfx, const char* points_tip) {
    const std::string s = sfx;
    draw_style_toolbar("Custom line style", sfx, pr.custom_line_style,
        [&pr, &s]() {
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Line width##" + s).c_str(), &pr.line_width, 0.1f, 5.0f, "%.2f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + s).c_str(), &pr.alpha, 0.0f, 1.0f, "%.2f");
            return ch;
        });
    ImGui::SameLine();
    ImGui::Checkbox(("Points##" + s).c_str(), &pr.draw_points);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\nWith Custom line style on too: points on top of the line.", points_tip);
    if (pr.draw_points) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat(("size##" + s + "pt").c_str(), &pr.point_size, 0.5f, 8.0f, "%.1f");
        if (pr.custom_line_style) {   // своя альфа точек — при Custom line style (у линии своя)
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            ImGui::SliderFloat(("alpha##" + s + "pt").c_str(), &pr.point_alpha, 0.0f, 1.0f, "%.2f");
        }
    }
    projection_views(pr).view2d->imdraw_lines      = pr.custom_line_style;
    projection_views(pr).view2d->line_thickness_px = pr.line_width;
    projection_views(pr).view2d->points_mode       = pr.draw_points;
    projection_views(pr).view2d->points_with_lines = pr.draw_points && pr.custom_line_style;
    projection_views(pr).view2d->point_size_px     = pr.point_size;
    projection_views(pr).view2d->point_alpha       = (pr.draw_points && pr.custom_line_style) ? pr.point_alpha : -1.0f;
}
}


namespace ucuda::ui_detail {
bool draw_point_style_toolbar(BifurcationDiagramConfig& bd, const char* id_suffix) {
    const std::string sfx = id_suffix;
    return draw_style_toolbar("Custom point style", id_suffix, bd.custom_point_style,
        [&bd, &sfx]() {
            if (bd.point_marker < 0 || bd.point_marker >= kPointMarkerCount) bd.point_marker = 0;
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::Combo(("Marker##" + sfx).c_str(), &bd.point_marker,
                               kPointMarkerNames, kPointMarkerCount);
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Point size##" + sfx).c_str(), &bd.point_size,
                                     0.5f, 12.0f, "%.1f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat(("Alpha##" + sfx).c_str(), &bd.point_alpha,
                                     0.0f, 1.0f, "%.2f");
            return ch;
        });
}
}


namespace ucuda::ui_detail {
void apply_point_style(Plot2DView& view, const BifurcationDiagramConfig& bd) {
    view.point_marker  = bd.custom_point_style ? bd.point_marker : -1;
    view.point_size_px = bd.custom_point_style ? bd.point_size   : 2.0f;
}
}


namespace ucuda::ui_detail {
void apply_snap_x(Plot2DView& view, double lo, double hi, int n) {
    view.snap_x_to_grid = true;
    // Вырожденный диапазон — это не сетка: снап по нему сажал бы курсор на
    // узлы, которых в данных нет. Проверка здесь, в единственной точке входа.
    if (n > 1 && hi != lo) {
        view.snap_x_min = lo;
        view.snap_x_max = hi;
        view.snap_x_n   = n;
    } else {
        view.snap_x_n = 0;
    }
}
}


namespace ucuda::ui_detail {
[[nodiscard]] double sweep_value_at(int k, int n, double lo, double hi,
                             bool log_scale, bool reverse, bool continuation) {
    if (continuation || reverse) {
        // Та же функция, что у cont-ядер и cont_sweep_value: конвенция continuation
        // живёт в ucuda_node_value_cont (configCUDA.h). Гард lo/hi > 0 на лог-ветке —
        // сюда может долететь живой чекбокс без прогона, а log10(<=0) отравил бы ось NaN'ом.
        const bool log_ok_cont = log_scale && lo > 0.0 && hi > 0.0;
        return (double)ucuda_node_value_cont(k, n, (numb)lo, (numb)hi, log_ok_cont, reverse);
    }
    if (log_scale && lo > 0.0 && hi > 0.0)
        return (double)ucuda_node_value_log(k, n, (numb)lo, (numb)hi);
    return (double)ucuda_node_value(k, n, (numb)lo, (numb)hi);
}
}


namespace ucuda::ui_detail {
void configure_sweep_x_axis(Plot2DView& view,
                                   const std::vector<SweepAxisMember>& members,
                                   const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars) {
    int  shared_kind = -2;   // -2 = ещё не видели, -1 = смешанные
    int  shared_idx  = -2;
    bool shared_log = false, log_seen = false, log_mismatch = false;

    for (const auto& m : members) {
        const int kidx = (m.kind == 2) ? -1 : m.index;
        if (shared_kind == -2) { shared_kind = m.kind; shared_idx = kidx; }
        else if (shared_kind != m.kind || shared_idx != kidx) { shared_kind = -1; shared_idx = -1; }

        if (!log_seen) { shared_log = m.log_scale; log_seen = true; }
        else if (shared_log != m.log_scale) { log_mismatch = true; }
    }

    view.x_fit_use_explicit = false;
    view.x_axis.log_scale = shared_log && !log_mismatch;

    if (shared_kind == 0 && shared_idx < 0)
        view.x_axis.name = "s";
    else if (shared_kind == 0 && shared_idx >= 0 && shared_idx < (int)params.size())
        view.x_axis.name = params[shared_idx];
    else if (shared_kind == 1 && shared_idx >= 0 && shared_idx < (int)vars.size())
        view.x_axis.name = vars[shared_idx] + " (IC)";
    else if (shared_kind == 2)
        view.x_axis.name = "h";
    else
        view.x_axis.name = "parameter";
}
}


namespace ucuda::ui_detail {
void apply_snap_x_from_config(Plot2DView& view,
                                     bool have_result, double res_lo, double res_hi, int res_n,
                                     const std::string& lo_text,
                                     const std::string& hi_text,
                                     const std::string& n_text) {
    if (have_result && res_n > 1 && res_hi != res_lo) {
        apply_snap_x(view, res_lo, res_hi, res_n);
        return;
    }
    // Без прогона сетку берём из полей конфига — но ТОЛЬКО если они реально
    // прочитались. Раньше здесь стояли дефолты parse_ratio_or(lo, 0.0) и
    // (hi, 1.0): на непрочитанных полях снап включался с чужой сеткой [0, 1],
    // и курсор садился на её узлы (k/(n-1)) вместо значений свипа.
    const int n = parse_int_or(n_text, 0);
    double lo = 0.0, hi = 0.0;
    if (n > 1 && parse_num_ok(lo_text, lo) && parse_num_ok(hi_text, hi) && lo != hi)
        apply_snap_x(view, lo, hi, n);
    else
        apply_snap_x(view, 0.0, 1.0, 0);   // n = 0 -> снап выключен
}
}


namespace ucuda::ui_detail {
void draw_export_submenu(const char* id_tag, int n,
                                const std::function<std::string(int)>& label,
                                const std::function<bool(int)>& ready,
                                const std::function<bool(int)>& busy,
                                const std::function<void(int, const std::string&)>& do_export,
                                const GuiCallbacks& cb) {
    if (!ImGui::BeginMenu("Export data...")) return;
    bool any = false;
    for (int i = 0; i < n; ++i) {
        if (!ready(i)) continue;
        any = true;
        char item[192];
        std::snprintf(item, sizeof(item), "%s##exp_%s_%d", label(i).c_str(), id_tag, i);
        if (ImGui::MenuItem(item, nullptr, false, !busy(i))) {
            if (cb.pick_save_file_csv) {
                std::string path = cb.pick_save_file_csv();
                if (!path.empty()) do_export(i, path);
            }
        }
    }
    if (!any) ImGui::TextDisabled("(no completed runs)");
    ImGui::EndMenu();
}
}


namespace ucuda::ui_detail {
HeatmapView& get_or_create_heatmap(
        std::map<int, std::unique_ptr<HeatmapView>>& map, int idx,
        int cfg_colormap, int app_default_colormap,
        int cfg_exponent_idx) {
    auto& slot = map[idx];
    if (!slot) {
        slot = std::make_unique<HeatmapView>();
        slot->colormap = (HeatmapColormap)colormap_id_or(cfg_colormap, app_default_colormap);
        if (cfg_exponent_idx != kNoExponent) slot->display_exponent_idx = cfg_exponent_idx;
    }
    return *slot;
}
}


namespace ucuda::ui_detail {
void invalidate_view(HeatmapView& hv) { hv.invalidate(); }
}


namespace ucuda::ui_detail {
void invalidate_view(Plot2DView& v)   { v.view_valid = false;  v.series_generation = -1; }
}


namespace ucuda::ui_detail {
void invalidate_view(Plot3DView& v)   { v.view_valid = false;  v.series_generation = -1; }
}


namespace ucuda::ui_detail {
bool session_reloaded(const AppModel& model, AppModel::AppMode mode, int& seen) {
    const int epoch = model.session_epoch[(int)mode];
    if (seen == epoch) return false;
    seen = epoch;
    return true;
}
}


namespace ucuda::ui_detail {
std::vector<std::vector<double>>& window_point_bufs(int window_id, size_t n_series) {
    auto& cache = ui_resource<std::map<int, std::vector<std::vector<double>>>, 3024>();
    auto& bufs = cache[window_id];
    if (bufs.size() != n_series) bufs.assign(n_series, {});
    return bufs;
}
}


namespace ucuda::ui_detail {
std::vector<double>& window_node_buf(int window_id) {
    auto& cache = ui_resource<std::map<int, std::vector<double>>, 3025>();
    return cache[window_id];
}
}


namespace ucuda::ui_detail {
void set_snap_nodes(Plot2DView& view, int window_id,
                           int npts, double lo, double hi,
                           bool log_scale, bool reverse, bool continuation) {
    auto& nb = window_node_buf(window_id);
    nb.clear();
    if (npts > 1) {
        nb.reserve((size_t)npts);
        for (int k = 0; k < npts; ++k)
            nb.push_back(sweep_value_at(k, npts, lo, hi, log_scale, reverse, continuation));
    }
    view.snap_x_nodes      = nb.empty() ? nullptr : nb.data();
    view.snap_x_node_count = (int)nb.size();
}
}


namespace ucuda::ui_detail {
void colormap_strip_item(int id, float w, float h) {
    constexpr int kSeg = 24;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    for (int i = 0; i < kSeg; ++i) {
        const float t0 = (float)i / kSeg, t1 = (float)(i + 1) / kSeg;
        const ImU32 c0 = cmap_sample_id(t0, id);
        const ImU32 c1 = cmap_sample_id(t1, id);
        // +0.5 к правой границе — иначе между сегментами проступают щели
        // на дробном UI-скейле.
        dl->AddRectFilledMultiColor(ImVec2(p.x + w * t0, p.y),
                                    ImVec2(p.x + w * t1 + 0.5f, p.y + h),
                                    c0, c1, c1, c0);
    }
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(ImGuiCol_Border));
    ImGui::Dummy(ImVec2(w, h));
}
}


namespace ucuda::ui_detail {
bool colormap_combo(const char* label, int* id, float width) {
    bool changed = false;
    const float h = ImGui::GetTextLineHeight();
    const float strip_w = h * 4.0f;

    ImGui::SetNextItemWidth(width);
    // HeightLarge = 20 строк вместо дефолтных 8: набор карт пользователь
    // набирает сам, но и десяток в попапе размером с восемь строк листать
    // неудобно.
    if (!ImGui::BeginCombo(label, colormap_label(*id), ImGuiComboFlags_HeightLarge))
        return false;

    const std::vector<int>& ids = picker_colormap_ids();
    // Показываем текущую карту первой, если её нет во включённом наборе.
    const bool orphan = std::find(ids.begin(), ids.end(), *id) == ids.end();
    auto row = [&](int cid) {
        ImGui::PushID(cid);
        const bool sel = (cid == *id);
        // Selectable без явной ширины растягивается на всю строку — он служит
        // подложкой-хитбоксом, а градиент и подпись рисуются поверх. SameLine()
        // после него встал бы у ПРАВОГО края (offset_from_start_x==0 = «после
        // предыдущего элемента»), поэтому возвращаем курсор в начало строки.
        const ImVec2 p0 = ImGui::GetCursorPos();
        if (ImGui::Selectable("##pick", sel, 0, ImVec2(0, h))) {
            if (!sel) { *id = cid; changed = true; }
        }
        if (sel) ImGui::SetItemDefaultFocus();
        ImGui::SetCursorPos(p0);
        colormap_strip_item(cid, strip_w, h);
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x);
        ImGui::TextUnformatted(colormap_label(cid));
        ImGui::PopID();
    };
    if (orphan) { row(*id); ImGui::Separator(); }
    for (int cid : ids) row(cid);
    // Все галочки сняты — пикер пуст. Текущая карта показана выше отдельной
    // строкой (orphan), так что состояние не тупиковое, но подсказать надо.
    if (ids.empty())
        ImGui::TextDisabled("(no colormaps ticked — pick some in Settings)");

    ImGui::EndCombo();
    return changed;
}
}


namespace ucuda::ui_detail {
bool draw_heatmap_toolbar(HeatmapView& hv, const HeatmapToolbarOpts& o) {
    bool changed = false;

    int cmap_idx = (int)hv.colormap;
    if (colormap_combo("Colormap", &cmap_idx, ImGui::GetFontSize() * 12.0f)) {
        hv.colormap = (HeatmapColormap)cmap_idx;
        if (o.persist_colormap) o.persist_colormap(cmap_idx);
        changed = true;
    }

    if (o.extras) o.extras();

    ImGui::SameLine();
    if (ImGui::Checkbox("Autoscale color", &hv.autoscale)) changed = true;
    if (!hv.autoscale) {
        ImGui::SameLine();
        if (InputNumStr("vmin", hv.manual_vmin_text, 80)) changed = true;
        hv.manual_vmin = (float)parse_ratio_or(hv.manual_vmin_text, hv.manual_vmin);
        ImGui::SameLine();
        if (InputNumStr("vmax", hv.manual_vmax_text, 80)) changed = true;
        hv.manual_vmax = (float)parse_ratio_or(hv.manual_vmax_text, hv.manual_vmax);
    }

    if (o.show_swap) {
        ImGui::SameLine();
        if (ImGui::Button(hv.swap_axes ? "Swap axes (on)" : "Swap axes")) {
            hv.swap_axes = !hv.swap_axes;
            changed = true;
        }
    }

    if (o.extras_tail) o.extras_tail();
    return changed;
}
}


namespace ucuda::ui_detail {
void draw_ls_exponent_picker(HeatmapView& hv, int n_exponents,
                                    const std::function<void(int)>& on_pick) {
    if (n_exponents <= 0) return;
    const int N = n_exponents;
    if (hv.display_exponent_idx != -1 &&
        (hv.display_exponent_idx < 0 || hv.display_exponent_idx >= N))
        hv.display_exponent_idx = 0;

    const std::string preview = (hv.display_exponent_idx == -1)
        ? "sum L_i" : ("L" + std::to_string(hv.display_exponent_idx + 1));
    auto pick = [&](int j) {
        hv.display_exponent_idx = j;
        if (on_pick) on_pick(j);
    };
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::BeginCombo("Exponent", preview.c_str())) {
        for (int j = 0; j < N; ++j) {
            std::string lbl = "L" + std::to_string(j + 1);
            if (ImGui::Selectable(lbl.c_str(), hv.display_exponent_idx == j)) pick(j);
        }
        ImGui::Separator();
        if (ImGui::Selectable("sum L_i", hv.display_exponent_idx == -1)) pick(-1);
        ImGui::EndCombo();
    }
}
}


namespace ucuda::ui_detail {
void ls_resolve_plane(LSCurveConfig& cact, int k,
                             const double*& plane, double& vmin, double& vmax, int& gen) {
    const size_t plane_size = (size_t)cact.result_2d.n_pts * (size_t)cact.result_2d.n_pts;
    const int    N          = cact.result_2d.n_exponents;

    if (k == -1) {
        if (cact.sum_cache_gen != cact.data_generation_2d) {
            cact.sum_cache.assign(plane_size, 0.0);
            double smin =  std::numeric_limits<double>::infinity();
            double smax = -std::numeric_limits<double>::infinity();
            for (size_t c2 = 0; c2 < plane_size; ++c2) {
                // Всё, что не колебательный режим (FP/unbound) — под sentinel:
                // раньше проверялось flags < 0, но у LS unbound теперь 0.
                if (c2 < cact.result_2d.flags.size() &&
                    !regime_is_oscillation(cact.result_2d.flags[c2])) {
                    cact.sum_cache[c2] = kSentinelNoData;
                    continue;
                }
                double sum = 0.0;
                for (int j = 0; j < N; ++j)
                    sum += cact.result_2d.values[(size_t)j * plane_size + c2];
                cact.sum_cache[c2] = sum;
                if (sum < smin) smin = sum;
                if (sum > smax) smax = sum;
            }
            cact.sum_cache_min = std::isfinite(smin) ? smin : 0.0;
            cact.sum_cache_max = std::isfinite(smax) ? smax : 0.0;
            cact.sum_cache_gen = cact.data_generation_2d;
        }
        plane = cact.sum_cache.data();
        vmin  = cact.sum_cache_min;
        vmax  = cact.sum_cache_max;
        gen   = cact.data_generation_2d * 64 + N;
    } else {
        plane = cact.result_2d.values.data() + (size_t)k * plane_size;
        vmin  = (k >= 0 && k < (int)cact.result_2d.min_val.size()) ? cact.result_2d.min_val[k] : 0.0;
        vmax  = (k >= 0 && k < (int)cact.result_2d.max_val.size()) ? cact.result_2d.max_val[k] : 0.0;
        gen   = cact.data_generation_2d * 64 + k;
    }
}
}


namespace ucuda::ui_detail {}


namespace ucuda::ui_detail {}


namespace ucuda::ui_detail {
void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) {
    lo = res_lo;
    hi = res_hi;
    if (hi == lo) {
        lo = parse_ratio_or(lo_text, 0.0);
        hi = parse_ratio_or(hi_text, 1.0);
    }
}
}
