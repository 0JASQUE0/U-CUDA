#include "gui_common.h"
#include "gui_inputs.h"
#include "tab_library.h"
// Список систем библиотеки: кэш списка, описание записи, сохранение редактора.
using namespace ucuda::ui_detail;

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
    SystemEditor& buf = *model.library_edit_buffer;
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
        const SystemEditor& eb = *model.library_edit_buffer;
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
