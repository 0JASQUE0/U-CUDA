#include "gui_internal.h"

namespace ucuda::ui_detail {





// Значение переменной var узла node в точке p.
double net_value_at(const NetworkResult& r, int p, int node, int var) {
    if (p < 0 || p >= r.n_points || node < 0 || node >= r.n_nodes
        || var < 0 || var >= r.n_vars) return std::numeric_limits<double>::quiet_NaN();
    return r.data[((size_t)p * (size_t)r.n_nodes + (size_t)node) * (size_t)r.n_vars + (size_t)var];
}





// Степень узла по рёбрам конфига (для таблицы узлов). Считается по рёбрам, а
// не по CSR: CSR строится под расчёт и знать про него редактору незачем.
int net_degree(const NetworkConfig& c, int node) {
    int d = 0;
    for (const NetEdge& e : c.edges) {
        if (e.to == node) ++d;
        if (e.bidirectional && e.from == node) ++d;
    }
    return d;
}





// Любая ручная правка топологии переводит конфиг в Custom: иначе следующее
// нажатие Generate молча стёрло бы правку.
void net_mark_custom(NetworkConfig& c) { c.topology = NetTopology::Custom; }





// Проверка выражений связи без запуска расчёта: тот же путь, что у Run.
std::string net_validate_laws(const NetworkSession& s, const NetworkConfig& c) {
    try {
        net_coupling_body(s.sys, c.laws);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}





void draw_network_topology_block(AppModel& model, NetworkSession& s, NetworkConfig& c) {
    if (!ImGui::CollapsingHeader("Topology", ImGuiTreeNodeFlags_DefaultOpen)) return;

    int topo = (int)c.topology;
    ImGui::SetNextItemWidth(kFieldW + 60.0f);
    if (ImGui::Combo("preset", &topo, kNetTopologyNames, IM_ARRAYSIZE(kNetTopologyNames))) {
        c.topology = (NetTopology)topo;
        if (c.topology != NetTopology::Custom) {
            const std::string err = net_generate_topology(c, s.vars);
            if (!err.empty()) c.last_error = err;
            model.network_view.selected_node = -1;
            model.network_view.selected_edge = -1;
        }
    }

    const bool is_grid = (c.topology == NetTopology::Grid || c.topology == NetTopology::King);
    const bool is_ring = (c.topology == NetTopology::Ring || c.topology == NetTopology::SmallWorld);
    if (c.topology != NetTopology::Custom) {
        if (is_grid) {
            InputNumStr("width",  c.gen_w_text, kFieldW);
            InputNumStr("height", c.gen_h_text, kFieldW);
            ImGui::Checkbox("periodic (torus)", &c.gen_periodic);
        } else {
            InputNumStr("nodes", c.gen_n_text, kFieldW);
        }
        if (is_ring) {
            InputNumStr("neighbours each side", c.gen_k_text, kFieldW);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("k = 1 gives the plain ring; k > 1 connects every node\n"
                                  "to k nearest neighbours on each side.");
        }
        if (c.topology == NetTopology::SmallWorld) {
            InputNumStr("rewire probability", c.gen_p_text, kFieldW);
            InputNumStr("topology seed", c.gen_seed_text, kFieldW);
        }
        ImGui::Checkbox("directed arcs", &c.gen_directed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Off: every edge pulls both ways.\n"
                              "On: the generator emits one-way arcs from -> to.");
        if (ImGui::Button("Generate", ImVec2(120, 0))) {
            const std::string err = net_generate_topology(c, s.vars);
            c.last_error = err;
            model.network_view.selected_node = -1;
            model.network_view.selected_edge = -1;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("rebuilds nodes and edges; per-node overrides are kept");
    } else {
        ImGui::TextDisabled("Custom: edges are yours, the generator will not touch them.");
        ImGui::TextDisabled("Pick a preset above to rebuild the network from scratch.");
    }

    ImGui::Text("%d nodes, %d edges", (int)c.nodes.size(), (int)c.edges.size());
    if ((int)c.nodes.size() > kMaxNetworkNodes)
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "over the %d-node cap: the run will be refused", kMaxNetworkNodes);
}





void draw_network_coupling_block(AppModel& model, NetworkSession& s, NetworkConfig& c) {
    if (!ImGui::CollapsingHeader("Coupling", ImGuiTreeNodeFlags_DefaultOpen)) return;

    InputNumStr("coupling K", c.coupling_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Global multiplier: every edge weight is multiplied by it.\n"
                          "Turn the whole network's coupling strength from one field.");
    ImGui::Checkbox("divide by node degree", &c.normalize_coupling);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("On: a node's coupling terms are divided by how many arcs reach it,\n"
                          "so K is the pull per node and a hub is not dragged N times harder.\n"
                          "Off: the plain Laplacian sum - the hub of a star will blow up first.");

    // Закон на ВСЕ рёбра сразу — рядом с общей силой связи, потому что это тот
    // же вопрос «какая связь у сети», только про вид, а не про величину.
    // Разнородный набор показывается как (mixed) и не трогается, пока из списка
    // не выберут закон.
    {
        int common = c.edges.empty() ? 0 : c.edges.front().law;
        for (const NetEdge& e : c.edges)
            if (e.law != common) { common = -1; break; }
        const char* preview = (common < 0) ? "(mixed)"
                            : (common < (int)c.laws.size() ? c.laws[(size_t)common].name.c_str() : "?");
        ImGui::SetNextItemWidth(kFieldW + 60.0f);
        if (ImGui::BeginCombo("law for all edges", preview)) {
            for (size_t li = 0; li < c.laws.size(); ++li) {
                const bool selected = ((int)li == common);
                if (ImGui::Selectable(c.laws[li].name.c_str(), selected))
                    for (NetEdge& e : c.edges) e.law = (int)li;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Assigns the picked law to every edge at once.\n"
                              "Per-edge exceptions stay available in the Edges table.");
    }

    ImGui::TextDisabled("A law is one expression per equation; empty = no coupling there.");
    ImGui::TextDisabled("Names: %s_j = source node, %s or %s_i = this node, K = edge weight.",
                        s.vars.empty() ? "x" : s.vars[0].c_str(),
                        s.vars.empty() ? "x" : s.vars[0].c_str(),
                        s.vars.empty() ? "x" : s.vars[0].c_str());

    int to_remove = -1;
    for (size_t li = 0; li < c.laws.size(); ++li) {
        NetCouplingLaw& law = c.laws[li];
        ImGui::PushID((int)li);
        law.expr.resize(s.vars.size());
        const std::string head = std::to_string(li) + ": " + law.name + "###law";
        if (ImGui::TreeNodeEx(head.c_str(), li == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            InputTextStr("name", law.name);
            for (size_t k = 0; k < s.vars.size(); ++k) {
                const std::string lbl = "d" + s.vars[k] + " +=";
                ImGui::SetNextItemWidth(260.0f);
                InputTextStr(lbl.c_str(), law.expr[k]);
            }
            if (c.laws.size() > 1 && ImGui::SmallButton("Delete law")) to_remove = (int)li;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    if (to_remove >= 0) {
        c.laws.erase(c.laws.begin() + to_remove);
        for (NetEdge& e : c.edges) {
            if (e.law == to_remove) e.law = 0;
            else if (e.law > to_remove) --e.law;
        }
    }
    if (ImGui::Button("+ Add law")) {
        NetCouplingLaw law = net_default_law(s.vars);
        law.name = "law " + std::to_string(c.laws.size());
        c.laws.push_back(std::move(law));
    }

    const std::string lerr = net_validate_laws(s, c);
    if (!lerr.empty())
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", lerr.c_str());

    // Предел устойчивости расщепления: вклад связи добавляется явным шагом
    // h * sum(w), поэтому при h * sum|w| > 1 узел с наибольшей степенью
    // перелетает через цель и сеть разносит независимо от схемы. Ровно это
    // ловит центр звезды, у которого сумма весов в (N-1) раз больше лучевой.
    {
        std::vector<int> es, esrc, elaw;
        std::vector<double> ew;
        net_build_csr(c, parse_num(c.coupling_text, 1.0), es, esrc, ew, elaw);
        double worst = 0.0;
        int worst_node = -1;
        for (size_t i = 0; i + 1 < es.size(); ++i) {
            double sum = 0.0;
            for (int e = es[i]; e < es[i + 1]; ++e) sum += std::fabs(ew[(size_t)e]);
            if (sum > worst) { worst = sum; worst_node = (int)i; }
        }
        const double hK = parse_num(c.h_text, 0.01) * worst;
        if (hK > 1.0)
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "h * coupling = %.2f at node %d: the split coupling step "
                               "overshoots (needs < 1)", hK, worst_node);
        else if (hK > 0.3)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                               "h * coupling = %.2f at node %d - close to the split step's limit",
                               hK, worst_node);
    }
    (void)model;
}





void draw_network_nodes_table(AppModel& model, NetworkSession& s, NetworkConfig& c) {
    if (!ImGui::CollapsingHeader("Nodes")) return;
    if (c.nodes.empty()) { ImGui::TextDisabled("No nodes."); return; }

    ImGui::TextDisabled("Empty cell = the shared value below. Fill one to detune that node.");

    // Столбец s появляется только у схем, которые читают a[0]: у остальных он
    // был бы полем, которое ни на что не влияет.
    const bool has_s = scheme_uses_symmetry(c.scheme, s.custom_schemes);
    const int n_cols = 2 + (has_s ? 1 : 0) + (int)s.params.size() + (int)s.vars.size();
    const float height = ImGui::GetTextLineHeightWithSpacing() * 12.0f;
    if (ImGui::BeginTable("##net_nodes", n_cols,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
                          ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("node");
        ImGui::TableSetupColumn("deg");
        if (has_s) ImGui::TableSetupColumn("s");
        for (const auto& p : s.params) ImGui::TableSetupColumn(p.c_str());
        for (const auto& v : s.vars)   ImGui::TableSetupColumn((v + "0").c_str());
        ImGui::TableHeadersRow();

        // Прокрутка к узлу, выбранному кликом в графе. Позицию НЕ считаем
        // арифметикой: строка тут выше строки текста (в ней поля ввода), и
        // счёт по высоте текста промахивался тем сильнее, чем дальше узел.
        // Вместо этого заставляем клиппер выдать именно эту строку и просим
        // SetScrollHereY у неё самой.
        const int  scroll_to = model.network_view.selected_node;
        const bool want_scroll = model.network_view.scroll_nodes_to_selected
                              && scroll_to >= 0 && scroll_to < (int)c.nodes.size();
        model.network_view.scroll_nodes_to_selected = false;

        ImGuiListClipper clipper;
        clipper.Begin((int)c.nodes.size());
        if (want_scroll) clipper.IncludeItemByIndex(scroll_to);
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                NetNode& nd = c.nodes[(size_t)i];
                ImGui::PushID(i);
                ImGui::TableNextRow();
                const bool sel = (model.network_view.selected_node == i);
                // Выделение строки — фоном, как в таблице рёбер. Selectable со
                // SpanAllColumns рисовал подсветку ПОВЕРХ полей ввода и всего
                // в одну строку текста, тогда как строка выше — в ней рамки.
                if (sel)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(60, 90, 140, 160));
                ImGui::TableNextColumn();
                // Подпись — индекс: им же узел адресуется в таблице рёбер и
                // подписан на графе. Высота — ровно как у полей ввода справа.
                if (ImGui::Selectable(std::to_string(i).c_str(), sel, ImGuiSelectableFlags_None,
                                      ImVec2(0.0f, ImGui::GetFrameHeight())))
                    model.network_view.selected_node = sel ? -1 : i;
                if (want_scroll && i == scroll_to) ImGui::SetScrollHereY(0.5f);
                ImGui::TableNextColumn();
                ImGui::Text("%d", net_degree(c, i));
                // Правка любой ячейки выделяет узел — так строка таблицы и
                // кружок на графе всегда говорят об одном и том же узле.
                // Поля — через общий InputNumStr: запятая, выражения вроде
                // 8/3 и шаг разряда по стрелкам работают здесь так же, как в
                // остальных полях приложения. Пустое значение валидно (= общее),
                // поэтому предупреждение о нечисле показываем только у непустых.
                auto num_cell = [&](const char* id, std::string& v) {
                    InputNumStr(id, v, 80.0f, {}, /*allow_empty*/ true);
                    if (ImGui::IsItemActivated()) model.network_view.selected_node = i;
                };
                if (has_s) {
                    ImGui::TableNextColumn();
                    num_cell("##s", nd.symmetry_s);
                }
                for (const auto& p : s.params) {
                    ImGui::TableNextColumn();
                    ImGui::PushID(p.c_str());
                    num_cell("##p", nd.param_values[p]);
                    ImGui::PopID();
                }
                for (const auto& v : s.vars) {
                    ImGui::TableNextColumn();
                    ImGui::PushID(v.c_str());
                    num_cell("##v", nd.initial_conditions[v]);
                    ImGui::PopID();
                }
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    if (ImGui::Button("Clear all overrides")) {
        for (NetNode& nd : c.nodes) {
            nd.param_values.clear();
            nd.initial_conditions.clear();
            nd.symmetry_s.clear();
        }
    }
}





void draw_network_edges_table(AppModel& model, NetworkSession& s, NetworkConfig& c) {
    if (!ImGui::CollapsingHeader("Edges")) return;

    std::vector<std::string> law_names;
    for (const NetCouplingLaw& l : c.laws) law_names.push_back(l.name);
    const std::vector<const char*> law_items = c_str_list(law_names);

    const float height = ImGui::GetTextLineHeightWithSpacing() * 10.0f;
    int to_remove = -1;
    if (ImGui::BeginTable("##net_edges", 6,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit,
                          ImVec2(0.0f, height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("from");
        ImGui::TableSetupColumn("to");
        ImGui::TableSetupColumn("both ways");
        ImGui::TableSetupColumn("weight");
        ImGui::TableSetupColumn("law");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();

        // Прокрутка к ребру, выбранному кликом в графе — см. таблицу узлов.
        const int  scroll_to = model.network_view.selected_edge;
        const bool want_scroll = model.network_view.scroll_edges_to_selected
                              && scroll_to >= 0 && scroll_to < (int)c.edges.size();
        model.network_view.scroll_edges_to_selected = false;

        ImGuiListClipper clipper;
        clipper.Begin((int)c.edges.size());
        if (want_scroll) clipper.IncludeItemByIndex(scroll_to);
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                NetEdge& e = c.edges[(size_t)i];
                ImGui::PushID(i);
                ImGui::TableNextRow();
                if (model.network_view.selected_edge == i)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(60, 90, 140, 160));
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(60.0f);
                if (ImGui::InputInt("##f", &e.from, 0, 0)) { net_mark_custom(c); }
                if (ImGui::IsItemActivated()) model.network_view.selected_edge = i;
                // ПОСЛЕ первого виджета строки: SetScrollHereY целится в
                // ПРЕДЫДУЩУЮ поданную строку, и вызов до него увёл бы прокрутку
                // на строку выше.
                if (want_scroll && i == scroll_to) ImGui::SetScrollHereY(0.5f);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(60.0f);
                if (ImGui::InputInt("##t", &e.to, 0, 0)) { net_mark_custom(c); }
                if (ImGui::IsItemActivated()) model.network_view.selected_edge = i;
                ImGui::TableNextColumn();
                if (ImGui::Checkbox("##b", &e.bidirectional)) net_mark_custom(c);
                ImGui::TableNextColumn();
                InputNumStr("##w", e.weight_text, 80.0f);
                if (ImGui::IsItemActivated()) model.network_view.selected_edge = i;
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(120.0f);
                if (!law_items.empty())
                    ImGui::Combo("##l", &e.law, law_items.data(), (int)law_items.size());
                ImGui::TableNextColumn();
                if (ImGui::SmallButton("x")) to_remove = i;
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    if (to_remove >= 0) {
        c.edges.erase(c.edges.begin() + to_remove);
        net_mark_custom(c);
        model.network_view.selected_edge = -1;
    }
    if (ImGui::Button("+ Add edge")) {
        NetEdge e;
        e.from = 0;
        e.to   = (int)c.nodes.size() > 1 ? 1 : 0;
        e.bidirectional = true;
        if (!c.edges.empty()) e.law = c.edges.back().law;   // см. графовый редактор
        c.edges.push_back(e);
        net_mark_custom(c);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear all overrides##edges")) {
        // Топология остаётся, сбрасываются только правки на рёбрах: вес в 1
        // и закон в первый — ровно то, что выдаёт генератор.
        for (NetEdge& e : c.edges) { e.weight_text = "1"; e.law = 0; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear edges")) { c.edges.clear(); net_mark_custom(c); }
    ImGui::SameLine();
    ImGui::TextDisabled("or drag from node to node in the graph window");
    (void)s;
}





void draw_network_controls(AppModel& model, SystemLibrary& /*lib*/) {
    NetworkSession& s = model.network_session;
    if (s.configs.empty()) {
        ImGui::TextDisabled("No system loaded. Pick a system in the list above.");
        return;
    }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size())
        s.active_config_index = 0;

    {
        const bool no_cfg = s.configs.empty();
        RunAllGroup g;
        g.pick_id = "pnet_";
        g.n       = (int)s.configs.size();
        g.label   = [&s](int i) { return s.configs[i].label; };
        g.enqueue = [&model](int i) { model.network_queue.push_back({ i }); };
        draw_run_and_run_all("##run_all_network", s.in_flight, no_cfg,
                             s.in_flight || no_cfg, 160.0f,
                             [&model, &s]() {
                                 if (!model.parametric_engine)
                                     model.parametric_engine = std::make_unique<ParametricEngine>();
                                 s.run_async(*model.parametric_engine, s.active_config_index);
                             },
                             [&s]() {
                                 ImGui::SameLine();
                                 if (!s.in_flight) ImGui::BeginDisabled();
                                 if (ImGui::Button("Stop", ImVec2(90, 0))) s.request_cancel();
                                 if (!s.in_flight) ImGui::EndDisabled();
                             },
                             { g },
                             [&model]() { model.start_next_in_network_queue(); },
                             model.network_queue.size());
    }
    ImGui::Separator();

    const TabBarResult tabs = draw_config_tab_bar(
        "##network_tabs", "network_tab_", (int)s.configs.size(),
        s.in_flight, s.running_config_index,
        [&s](int i) { return s.configs[i].label; },
        {},
        [&s]() { s.add_config(); });
    if (tabs.active    >= 0) s.active_config_index = tabs.active;
    if (tabs.to_remove >= 0) model.remove_network_config(tabs.to_remove);

    if (s.active_config_index >= (int)s.configs.size())
        s.active_config_index = (int)s.configs.size() - 1;
    NetworkConfig& c = s.configs[(size_t)s.active_config_index];

    draw_label_rename("Label##net", c.label);

    draw_scheme_combo("scheme", c.scheme, s.custom_schemes, {}, &s.enabled_builtin_schemes,
                      nullptr, s.sys.is_map, &s.wrapper_schemes);

    IntegrationFields f;
    f.h           = &c.h_text;
    f.symmetry_s  = &c.symmetry_s;
    f.t_max       = &c.t_max_text;
    f.transient   = &c.transient_text;
    f.pre_scaller = &c.pre_scaller_text;
    f.max_value   = &c.max_value_text;
    draw_integration_block("Integration", c.scheme, s.custom_schemes, f, nullptr, s.sys.is_map);
    InputNumStr("max points", c.max_points_text, kFieldW);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Cap on recorded time points. The engine raises the decimator\n"
                          "when the run would exceed it, and reports the value it used.");
    // Связь интегрируется расщеплением — это свойство вкладки, а не настройка,
    // и молчать о нём нельзя: порядок схемы на сети не держится.
    ImGui::TextDisabled("Coupling is split off the step (first order), the node step is not.");

    draw_network_topology_block(model, s, c);
    draw_network_coupling_block(model, s, c);

    draw_named_num_fields("Parameters##net", s.params, c.param_values,
                          "Shared by every node unless the node overrides it.");
    draw_named_num_fields("Initial conditions##net", s.vars, c.initial_conditions,
                          "Shared base; the spread below is added on top.");
    if (ImGui::CollapsingHeader("Initial spread", ImGuiTreeNodeFlags_DefaultOpen)) {
        InputNumStr("spread", c.ic_spread_text, kFieldW);
        InputNumStr("spread seed", c.ic_seed_text, kFieldW);
        ImGui::TextDisabled("Uniform noise in [-spread, +spread] per node and variable.");
        ImGui::TextDisabled("Zero spread means identical nodes: a diffusively coupled");
        ImGui::TextDisabled("network then starts synchronised and stays that way.");
    }

    draw_network_nodes_table(model, s, c);
    draw_network_edges_table(model, s, c);

    if (c.last_run_ok && c.result.n_points > 0) {
        ImGui::Separator();
        ImGui::Text("%d points x %d nodes, dt = %g", c.result.n_points, c.result.n_nodes, c.result.dt);
        if (c.result.pre_scaller != parse_num_int(c.pre_scaller_text, 1))
            ImGui::TextDisabled("decimator raised to %d to fit max points", c.result.pre_scaller);
        if (c.result.n_diverged > 0)
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                               "%d node(s) diverged, the run stopped there", c.result.n_diverged);
    }
    if (!c.last_error.empty()) {
        ImGui::Separator();
        draw_error_box("##net_err", c.last_error);
    }
}





// --- Окно графа ------------------------------------------------------------

void draw_network_graph_window(AppModel& model) {
    NetworkSession& s = model.network_session;
    if (s.configs.empty()) { ImGui::TextDisabled("No system loaded."); return; }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) return;
    NetworkConfig& c = s.configs[(size_t)s.active_config_index];
    NetworkViewState& vs = model.network_view;
    const NetworkResult& r = c.result;
    const bool has_data = c.last_run_ok && r.n_points > 0 && r.n_nodes == (int)c.nodes.size();

    // Панель над холстом: переменная раскраски, курсор времени, воспроизведение.
    if (!s.vars.empty()) {
        const std::vector<const char*> vitems = c_str_list(s.vars);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("colour by", &vs.color_var, vitems.data(), (int)vitems.size());
    }
    if (has_data) {
        ImGui::SameLine();
        if (vs.time_index >= r.n_points) vs.time_index = r.n_points - 1;
        if (vs.time_index < 0) vs.time_index = 0;
        ImGui::SetNextItemWidth(260.0f);
        ImGui::SliderInt("##net_t", &vs.time_index, 0, r.n_points - 1,
                         "t index %d");
        ImGui::SameLine();
        ImGui::Text("t = %g", r.t0 + r.dt * (double)vs.time_index);
        ImGui::SameLine();
        if (ImGui::Button(vs.playing ? "Pause" : "Play", ImVec2(70, 0))) vs.playing = !vs.playing;
        if (vs.playing) {
            // Шаг по времени кадра, а не по кадру: на 144 Гц иначе всё улетает
            // вчетверо быстрее, чем на 30.
            const float dt = ImGui::GetIO().DeltaTime;
            const int adv = (int)(vs.play_fps * dt + 0.5f);
            vs.time_index += (adv > 0) ? adv : 1;
            if (vs.time_index >= r.n_points) vs.time_index = 0;
        }
    } else {
        ImGui::SameLine();
        ImGui::TextDisabled("no data yet - press Run");
    }
    ImGui::SameLine();
    ImGui::Checkbox("labels", &vs.show_labels);
    ImGui::SameLine();
    {
        // Своя колормапа графа: -1 значит «как в настройках приложения», и
        // пока пользователь её не трогал, граф идёт вместе с хитмапами.
        int cmap = (vs.graph_colormap >= 0) ? vs.graph_colormap : model.heatmap_colormap;
        if (colormap_combo("##net_cmap", &cmap, ImGui::GetFontSize() * 10.0f))
            vs.graph_colormap = cmap;
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    if (size.x < 50.0f) size.x = 50.0f;
    if (size.y < 50.0f) size.y = 50.0f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y),
                      IM_COL32(18, 18, 22, 255));
    ImGui::InvisibleButton("##net_canvas", size,
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    const bool canvas_hovered = ImGui::IsItemHovered();
    const ImVec2 mouse = ImGui::GetIO().MousePos;

    const float pad = 24.0f;
    auto to_screen = [&](const NetNode& nd) {
        return ImVec2(origin.x + pad + nd.ui_x * (size.x - 2.0f * pad),
                      origin.y + pad + nd.ui_y * (size.y - 2.0f * pad));
    };
    auto from_screen = [&](ImVec2 p) {
        const float x = (p.x - origin.x - pad) / (size.x - 2.0f * pad);
        const float y = (p.y - origin.y - pad) / (size.y - 2.0f * pad);
        return ImVec2(x < 0 ? 0 : (x > 1 ? 1 : x), y < 0 ? 0 : (y > 1 ? 1 : y));
    };

    const float node_r = (c.nodes.size() > 64) ? 5.0f : ((c.nodes.size() > 24) ? 8.0f : 12.0f);
    const int cmap_id = (vs.graph_colormap >= 0) ? vs.graph_colormap : model.heatmap_colormap;

    // Расстояние от курсора до отрезка — попадание по ребру. Ребро тонкое, и
    // ловить его кликом иначе нечем.
    auto dist_to_segment = [](ImVec2 p, ImVec2 a, ImVec2 b) {
        const float vx = b.x - a.x, vy = b.y - a.y;
        const float wx = p.x - a.x, wy = p.y - a.y;
        const float len2 = vx * vx + vy * vy;
        float t = (len2 > 0.0f) ? (wx * vx + wy * vy) / len2 : 0.0f;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        const float dx = wx - t * vx, dy = wy - t * vy;
        return std::sqrt(dx * dx + dy * dy);
    };

    // Рёбра под узлами.
    int hovered_edge = -1;
    float hovered_edge_d = 6.0f;      // порог попадания в пикселях
    for (size_t i = 0; i < c.edges.size(); ++i) {
        const NetEdge& e = c.edges[i];
        if (e.from < 0 || e.to < 0 || e.from >= (int)c.nodes.size() || e.to >= (int)c.nodes.size())
            continue;
        const ImVec2 a = to_screen(c.nodes[(size_t)e.from]);
        const ImVec2 b = to_screen(c.nodes[(size_t)e.to]);
        if (canvas_hovered) {
            const float d = dist_to_segment(mouse, a, b);
            if (d < hovered_edge_d) { hovered_edge_d = d; hovered_edge = (int)i; }
        }
        const bool sel = ((int)i == vs.selected_edge);
        dl->AddLine(a, b, sel ? IM_COL32(255, 200, 80, 255) : IM_COL32(120, 130, 150, 160),
                    sel ? 2.5f : 1.4f);
        if (!e.bidirectional) {
            // Стрелка у приёмника: направление связи — не украшение, из
            // картинки без него не прочитать, кто кого тянет.
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len > 1.0f) {
                const float ux = dx / len, uy = dy / len;
                const ImVec2 tip(b.x - ux * (node_r + 2.0f), b.y - uy * (node_r + 2.0f));
                const ImVec2 l(tip.x - ux * 9.0f - uy * 5.0f, tip.y - uy * 9.0f + ux * 5.0f);
                const ImVec2 rr(tip.x - ux * 9.0f + uy * 5.0f, tip.y - uy * 9.0f - ux * 5.0f);
                dl->AddTriangleFilled(tip, l, rr, IM_COL32(150, 160, 180, 200));
            }
        }
    }

    // Протяжка нового ребра.
    if (vs.drag_from >= 0 && vs.drag_from < (int)c.nodes.size())
        dl->AddLine(to_screen(c.nodes[(size_t)vs.drag_from]), mouse, IM_COL32(255, 220, 120, 200), 2.0f);

    double lo = 0.0, hi = 1.0;
    if (has_data && vs.color_var < (int)r.vmin.size()) {
        lo = r.vmin[(size_t)vs.color_var];
        hi = r.vmax[(size_t)vs.color_var];
        if (!(hi > lo)) { lo -= 0.5; hi += 0.5; }
    }

    int hovered_node = -1;
    for (size_t i = 0; i < c.nodes.size(); ++i) {
        const ImVec2 p = to_screen(c.nodes[i]);
        const float dx = mouse.x - p.x, dy = mouse.y - p.y;
        if (canvas_hovered && dx * dx + dy * dy <= (node_r + 3.0f) * (node_r + 3.0f))
            hovered_node = (int)i;

        ImU32 col = IM_COL32(110, 120, 140, 255);
        if (has_data) {
            const double v = net_value_at(r, vs.time_index, (int)i, vs.color_var);
            if (std::isfinite(v)) {
                const float t = (float)((v - lo) / (hi - lo));
                col = cmap_sample_id(t < 0 ? 0.0f : (t > 1 ? 1.0f : t), cmap_id);
            } else {
                col = IM_COL32(70, 70, 70, 255);   // разлетелось / не посчитано
            }
        }
        dl->AddCircleFilled(p, node_r, col);
        const bool sel = ((int)i == vs.selected_node);
        dl->AddCircle(p, node_r, sel ? IM_COL32(255, 255, 255, 255) : IM_COL32(30, 30, 35, 255),
                      0, sel ? 2.5f : 1.5f);
        if ((int)i == vs.drag_from)
            dl->AddCircle(p, node_r + 4.0f, IM_COL32(255, 220, 120, 255), 0, 2.0f);
        if (has_data && (int)i < (int)r.status.size() && r.status[i] != NET_ST_OK)
            dl->AddCircle(p, node_r + 3.0f, IM_COL32(255, 80, 80, 255), 0, 2.0f);
        // Подпись — ИНДЕКС узла: тем же числом он адресуется в обеих таблицах.
        if (vs.show_labels && c.nodes.size() <= 64) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%d", (int)i);
            dl->AddText(ImVec2(p.x + node_r + 2.0f, p.y - 7.0f), IM_COL32(200, 200, 210, 220), buf);
        }
    }

    // Мышь. Ребро строится ДВУМЯ кликами, а не протяжкой с зажатой кнопкой:
    // двойной клик по узлу берёт его за источник, следующий клик по другому
    // узлу замыкает ребро. Протяжка требовала держать кнопку через весь холст
    // и срывалась на первом же промахе.
    //
    // Состояние на начало кадра: двойной клик приходит вместе с обычным (ImGui
    // шлёт оба), и без снимка обработчик одиночного успел бы погасить то, что
    // обработчик двойного только что поставил.
    const int  pending      = vs.drag_from;
    const bool dbl          = canvas_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
    const bool clicked      = canvas_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);

    if (clicked) {
        if (pending >= 0) {
            // Ждём второй конец: клик по другому узлу — ребро, всё остальное —
            // отмена. Ребро-дубликат не проверяем: параллельные связи с разными
            // законами имеют смысл, а лишнее видно в таблице.
            if (hovered_node >= 0 && hovered_node != pending) {
                NetEdge e;
                e.from = pending;
                e.to   = hovered_node;
                e.bidirectional = !c.gen_directed;
                // Закон — как у последнего ребра: у однородной сети новое
                // ребро обязано остаться в той же связи, иначе «закон на все
                // рёбра» ломается первой же ручной правкой.
                if (!c.edges.empty()) e.law = c.edges.back().law;
                c.edges.push_back(e);
                net_mark_custom(c);
                vs.selected_edge = (int)c.edges.size() - 1;
                vs.scroll_edges_to_selected = true;
            }
            vs.drag_from = -1;
        } else if (hovered_node >= 0) {
            vs.selected_node = hovered_node;
            vs.scroll_nodes_to_selected = true;
        } else if (hovered_edge >= 0) {
            vs.selected_edge = hovered_edge;
            vs.scroll_edges_to_selected = true;
        } else {
            vs.selected_node = -1;
            vs.selected_edge = -1;
        }
    }

    if (dbl && pending < 0) {
        if (hovered_node >= 0) {
            vs.drag_from = hovered_node;          // взяли источник будущего ребра
        } else if (hovered_edge < 0 && (int)c.nodes.size() < kMaxNetworkNodes) {
            // Пустое место — новый узел ровно под курсором. Переопределений у
            // него нет: он повторяет общие параметры конфига, как и все.
            NetNode nd;
            const ImVec2 uv = from_screen(mouse);
            nd.ui_x = uv.x;
            nd.ui_y = uv.y;
            nd.label = std::to_string(c.nodes.size());
            c.nodes.push_back(std::move(nd));
            net_mark_custom(c);
            vs.selected_node = (int)c.nodes.size() - 1;
            vs.scroll_nodes_to_selected = true;
        }
    }

    // Esc — отменить незаконченное ребро.
    if (vs.drag_from >= 0 && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) vs.drag_from = -1;

    if (vs.drag_from < 0 && vs.selected_node >= 0 && vs.selected_node < (int)c.nodes.size()
        && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && canvas_hovered) {
        const ImVec2 uv = from_screen(mouse);
        c.nodes[(size_t)vs.selected_node].ui_x = uv.x;
        c.nodes[(size_t)vs.selected_node].ui_y = uv.y;
    }

    if (canvas_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        if (hovered_node >= 0) {
            const int dead = hovered_node;
            c.edges.erase(std::remove_if(c.edges.begin(), c.edges.end(),
                                         [dead](const NetEdge& e) { return e.from == dead || e.to == dead; }),
                          c.edges.end());
            for (NetEdge& e : c.edges) {
                if (e.from > dead) --e.from;
                if (e.to   > dead) --e.to;
            }
            c.nodes.erase(c.nodes.begin() + dead);
            net_mark_custom(c);
            vs.selected_node = -1;
            vs.selected_edge = -1;
            vs.drag_from = -1;
            // Узла больше нет, а подсказка ниже ходит по индексу: без сброса она
            // прочитала бы за концом вектора в этом же кадре.
            hovered_node = -1;
        } else if (hovered_edge >= 0 && hovered_edge < (int)c.edges.size()) {
            c.edges.erase(c.edges.begin() + hovered_edge);
            net_mark_custom(c);
            vs.selected_edge = -1;
            hovered_edge = -1;
        }
    }

    // Подсказка о значении под курсором.
    if (canvas_hovered && hovered_node >= 0) {
        ImGui::BeginTooltip();
        ImGui::Text("node %d", hovered_node);
        ImGui::Text("degree %d", net_degree(c, hovered_node));
        if (has_data)
            for (size_t k = 0; k < s.vars.size() && (int)k < r.n_vars; ++k)
                ImGui::Text("%s = %g", s.vars[k].c_str(),
                            net_value_at(r, vs.time_index, hovered_node, (int)k));
        ImGui::EndTooltip();
    } else if (canvas_hovered && hovered_edge >= 0 && hovered_edge < (int)c.edges.size()) {
        const NetEdge& e = c.edges[(size_t)hovered_edge];
        ImGui::BeginTooltip();
        ImGui::Text("edge %d: %d %s %d", hovered_edge, e.from,
                    e.bidirectional ? "<->" : "->", e.to);
        ImGui::Text("weight %s, law %s", e.weight_text.c_str(),
                    (e.law >= 0 && e.law < (int)c.laws.size()) ? c.laws[(size_t)e.law].name.c_str() : "?");
        ImGui::EndTooltip();
    }

    if (has_data) {
        // Шкала цвета — та же, что у графа, и по тем же значениям.
        const std::vector<ColorbarTick> ticks = colorbar_ticks((float)lo, (float)hi, 0);
        const float cb_w = colorbar_total_width(ticks);
        const ImVec2 cb_pos(origin.x + size.x - cb_w - 4.0f, origin.y + 8.0f);
        draw_colorbar(dl, cb_pos, size.y - 16.0f, (float)lo, (float)hi,
                      (HeatmapColormap)cmap_id, false, 0, ticks);
    }

    if (vs.drag_from >= 0)
        ImGui::TextColored(ImVec4(1.0f, 0.86f, 0.47f, 1.0f),
                           "edge from node %d: click the other node (Esc cancels)", vs.drag_from);
    else
        ImGui::TextDisabled("double click node = start edge, double click empty = new node, "
                            "drag = move, right click = delete");
}





// --- Кривые узлов ----------------------------------------------------------

void draw_network_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view) {
    NetworkSession& s = model.network_session;
    if (s.configs.empty()) { ImGui::TextDisabled("No system loaded."); return; }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) return;
    NetworkConfig& c = s.configs[(size_t)s.active_config_index];
    NetworkViewState& vs = model.network_view;
    const NetworkResult& r = c.result;

    if (!s.vars.empty()) {
        const std::vector<const char*> vitems = c_str_list(s.vars);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("variable", &vs.series_var, vitems.data(), (int)vitems.size());
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputInt("first node", &vs.first_series_node, 1, 8);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputInt("count", &vs.series_count, 1, 8);
    if (vs.series_count < 1) vs.series_count = 1;
    if (vs.first_series_node < 0) vs.first_series_node = 0;

    {
        int cmap = (vs.series_colormap >= 0) ? vs.series_colormap : model.heatmap_colormap;
        ImGui::SameLine();
        if (colormap_combo("##net_series_cmap", &cmap, ImGui::GetFontSize() * 10.0f))
            vs.series_colormap = cmap;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Curves are coloured along this map in node order,\n"
                              "so neighbours on the ring stay neighbours in colour.");
    }
    ImGui::SameLine();
    // Стиль линий — тот же тулбар-с-попапом, что у точек бифуркационных
    // диаграмм; выключенный режим даёт ровно прежний вид.
    draw_style_toolbar("Custom line style", "netseries", vs.custom_line_style,
        [&vs]() {
            bool ch = false;
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat("Line width##netseries", &vs.line_width, 0.5f, 8.0f, "%.1f");
            ImGui::SetNextItemWidth(150);
            ch |= ImGui::SliderFloat("Alpha##netseries", &vs.line_alpha, 0.05f, 1.0f, "%.2f");
            ch |= ImGui::Checkbox("Points instead of lines##netseries", &vs.points_mode);
            if (vs.points_mode) {
                if (vs.point_marker < 0 || vs.point_marker >= kPointMarkerCount) vs.point_marker = 0;
                ImGui::SetNextItemWidth(150);
                ch |= ImGui::Combo("Marker##netseries", &vs.point_marker,
                                   kPointMarkerNames, kPointMarkerCount);
                ImGui::SetNextItemWidth(150);
                ch |= ImGui::SliderFloat("Point size##netseries", &vs.point_size, 0.5f, 12.0f, "%.1f");
            } else {
                ch |= ImGui::Checkbox("Mark data points##netseries", &vs.node_markers);
            }
            return ch;
        });

    if (!c.last_run_ok || r.n_points <= 0) {
        ImGui::TextDisabled("No data yet - press Run on the Network tab.");
        return;
    }

    // Буферы живут между кадрами: вид держит указатели на время render().
    static std::vector<std::vector<double>> bufs;
    int first = vs.first_series_node;
    if (first >= r.n_nodes) first = r.n_nodes - 1;
    int count = vs.series_count;
    if (first + count > r.n_nodes) count = r.n_nodes - first;
    if (count < 1) count = 1;
    if ((int)bufs.size() != count) bufs.assign((size_t)count, {});

    std::vector<PlotSeriesInput> series;
    std::vector<bool> vis((size_t)count, true);
    for (int k = 0; k < count; ++k) {
        const int node = first + k;
        std::vector<double>& b = bufs[(size_t)k];
        b.clear();
        b.reserve((size_t)r.n_points * 2);
        for (int p = 0; p < r.n_points; ++p) {
            const double v = net_value_at(r, p, node, vs.series_var);
            if (!std::isfinite(v)) continue;     // хвост после разлёта не рисуем
            b.push_back(r.t0 + r.dt * (double)p);
            b.push_back(v);
        }
        PlotSeriesInput si;
        si.points   = b.data();
        si.n_points = (int)(b.size() / 2);
        const float t = (count > 1) ? (float)k / (float)(count - 1) : 0.0f;
        const ImU32 col = cmap_sample_id(t, (vs.series_colormap >= 0) ? vs.series_colormap
                                                                     : model.heatmap_colormap);
        si.color = ImGui::ColorConvertU32ToFloat4(col);
        if (vs.custom_line_style) si.color.w = vs.line_alpha;
        // Подпись — индекс узла, как на графе и в таблицах.
        si.label = std::to_string(node);
        series.push_back(si);
    }

    // Толщина > 1 px существует только на пути ImDrawList: glLineWidth в core
    // OpenGL драйверы клампят до 1.0 (то же решение, что у LLE/LS).
    view.points_mode       = vs.custom_line_style && vs.points_mode;
    view.point_marker      = (vs.custom_line_style && vs.points_mode) ? vs.point_marker : -1;
    view.point_size_px     = (vs.custom_line_style && vs.points_mode) ? vs.point_size : 2.0f;
    view.line_thickness_px = vs.custom_line_style ? vs.line_width : 1.5f;
    view.imdraw_lines      = vs.custom_line_style && !vs.points_mode && vs.line_width > 1.01f;
    view.point_markers     = vs.custom_line_style && !vs.points_mode && vs.node_markers;

    view.x_axis.name = "t";
    view.y_axis.name = (vs.series_var < (int)s.vars.size()) ? s.vars[(size_t)vs.series_var] : "x";
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    // Поколение данных для вида включает НАСТРОЙКИ набора кривых, а не только
    // номер расчёта: Plot2DView перезаливает VBO лишь на его изменение, и без
    // этого смена переменной или диапазона узлов не доезжала бы до экрана.
    const int series_gen = ((c.data_generation * 131 + vs.series_var) * 131 + first) * 131 + count
                         + s.active_config_index * 7919;
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE7001, series_gen,
                series, vis, vis, false);
}





// --- Растр узел x время ----------------------------------------------------

void draw_network_raster_window(AppModel& model, PlotRenderer& renderer, HeatmapView& view) {
    NetworkSession& s = model.network_session;
    if (s.configs.empty()) { ImGui::TextDisabled("No system loaded."); return; }
    if (s.active_config_index < 0 || s.active_config_index >= (int)s.configs.size()) return;
    NetworkConfig& c = s.configs[(size_t)s.active_config_index];
    NetworkViewState& vs = model.network_view;
    const NetworkResult& r = c.result;

    if (!s.vars.empty()) {
        const std::vector<const char*> vitems = c_str_list(s.vars);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::Combo("variable##raster", &vs.color_var, vitems.data(), (int)vitems.size());
    }
    // Тот же тулбар, что у всех хитмап: колормапа, автоскейл, ручные vmin/vmax.
    // Swap axes здесь нет — оси растра не взаимозаменяемы (время и номер узла).
    {
        HeatmapToolbarOpts o;
        o.show_swap = false;
        draw_heatmap_toolbar(view, o);
    }
    if (!c.last_run_ok || r.n_points <= 0) {
        ImGui::TextDisabled("No data yet - press Run on the Network tab.");
        return;
    }

    // Хитмапе нужен row-major [node][time], у результата — [time][node][var].
    // Транспонируем один раз на поколение данных, а не каждый кадр.
    static std::vector<double> cache;
    static int cache_gen = -1, cache_var = -1, cache_cfg = -1;
    const int gen = c.data_generation;
    if (cache_gen != gen || cache_var != vs.color_var || cache_cfg != s.active_config_index) {
        cache.assign((size_t)r.n_nodes * (size_t)r.n_points, 0.0);
        for (int i = 0; i < r.n_nodes; ++i)
            for (int p = 0; p < r.n_points; ++p)
                cache[(size_t)i * (size_t)r.n_points + (size_t)p] =
                    net_value_at(r, p, i, vs.color_var);
        cache_gen = gen; cache_var = vs.color_var; cache_cfg = s.active_config_index;
    }

    double vlo = 0.0, vhi = 1.0;
    if (vs.color_var < (int)r.vmin.size()) {
        vlo = r.vmin[(size_t)vs.color_var];
        vhi = r.vmax[(size_t)vs.color_var];
        if (!(vhi > vlo)) { vlo -= 0.5; vhi += 0.5; }
    }

    view.x_axis.name = "t";
    view.y_axis.name = "node";
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    view.render(renderer, origin, avail, /*owner_id*/ 0xBE7002,
                (gen * 64 + vs.color_var) * 131 + s.active_config_index,
                r.n_points, r.n_nodes, cache.data(),
                r.t0, r.t0 + r.dt * (double)(r.n_points - 1),
                0.0, (double)(r.n_nodes - 1),
                vlo, vhi, false);
}

}
