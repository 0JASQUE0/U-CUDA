#pragma once
// Имена полей формулами, подписи, комбо переменных, деревья групп параметров, меню полей.
#include "gui_integration.h"

namespace ucuda::ui_detail {
void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value,
                                 const std::function<void()>& extra = {});


void set_field_library(SystemLibrary* lib);

bool field_restore_menu_items(const AppModel* model, BroadcastField f, const std::string& name,
                              std::map<std::string, std::string>& values, const std::vector<std::string>& names);


bool field_restore_popup(const char* popup_id, const AppModel* model, BroadcastField f, const std::string& name,
                         std::map<std::string, std::string>& values, const std::vector<std::string>& names);

void draw_param_tree(ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table = false,
                     const std::function<void(const std::vector<std::string>&)>& group_menu = {});


void draw_param_groups_window(AppModel& m, const std::vector<std::string>& params,
                              const std::vector<std::string>& vars, const char* id);


[[nodiscard]] std::string field_label_auto(const std::string& name);


[[nodiscard]] std::string field_label(const FieldLabels* labels, const std::string& name);


void draw_field_name(const std::string& name, const FieldLabels* labels);


void set_field_latex(bool on);


void set_field_labels(const FieldLabels* labels);


bool var_combo_begin(const char* id, const std::string& cur, bool cur_is_name = true);


void var_combo_preview();


bool var_selectable(const std::string& name, bool selected);


[[nodiscard]] std::string plot_var_name(const std::string& name);


void draw_named_num_fields(const char* header,
                                  const std::vector<std::string>& names,
                                  std::map<std::string, std::string>& values,
                                  const char* note = nullptr,
                                  AppModel* bc = nullptr,
                                  BroadcastField field = BroadcastField::Param,
                                  // Поля, которые расчёт ПЕРЕЗАПИШЕТ сам (область
                                  // устойчивости строит четыре элемента матрицы в
                                  // каждом узле сетки). Такие гасятся, а не молча
                                  // игнорируются: живое поле, ни на что не влияющее,
                                  // читается как настройка, которой оно не является.
                                  const std::function<bool(const std::string&)>& overwritten = {});

const SystemRecord* field_lib_record(const AppModel* m, bool reload) ;


MathExtent field_tex_extent(const std::string& tex, float fs) ;


float field_tex_size() ;


void field_tex_draw(ImDrawList* dl, ImVec2 p, const std::string& tex, float box_h, bool fit = false) ;


struct PgNode {
    std::string seg, path;
    const ParamGroup* g = nullptr;
    std::vector<PgNode> kids;
    int count = 0;   // параметров из names в узле и под ним
};


int pg_count(PgNode& n, const std::set<std::string>& present) ;


struct PgDrop { std::string from, to; bool after = false; };


std::string pg_payload(const ParamGroups* list, const std::string& path) ;


void pg_drag_node(ParamGroups* list, const std::string& path, const std::string& shown, PgDrop& drop) ;


using PgMenu = std::function<void(const std::vector<std::string>&)>;


void pg_collect(const PgNode& n, const std::set<std::string>& present, std::vector<std::string>& out) ;


bool pg_header(const std::string& label, const std::string& id, bool in_table, ParamGroups* list,
               const std::string& seg, PgDrop& drop, const PgMenu& menu,
               const std::function<std::vector<std::string>()>& names_of) ;


void pg_draw(const PgNode& n, const std::set<std::string>& present,
             const std::function<void(const std::string&)>& item, bool in_table, ParamGroups* list, PgDrop& drop,
             const PgMenu& menu) ;
}
