#pragma once
// Имена полей формулами, подписи, комбо переменных, деревья групп параметров, меню полей.
#include "gui_common.h"

namespace ucuda::ui_detail {
void field_apply_all_menu(AppModel* model, BroadcastField f,
                                 const std::string& name, const std::string& label,
                                 const std::string& value,
                                 const std::function<void()>& extra = {});


void set_field_library(SystemLibrary* lib);


// Конфиги вкладки, которую сейчас рисует панель: поле ввода (InputNumStr, InputNumListStr),
// лежащее ВНУТРИ активного конфига, получает в ПКМ "применить во все конфиги вкладки" и
// "сбросить к значению по умолчанию". Тот же член в другом конфиге находим по смещению от
// начала объекта — у объектов одного типа оно одно, поэтому знать имена полей не нужно.
struct CfgFieldScope {
    AppModel* model = nullptr;
    const char* tab = "";                  // для меню и записи отмены
    const char* base = nullptr;            // активный конфиг
    size_t size = 0;                       // sizeof конфига
    const char* def = nullptr;             // конфиг по умолчанию того же типа
    int active = -1;
    std::function<int()> count;
    std::function<char*(int)> at;
    std::function<std::string(int)> label;
};
void push_cfg_scope(CfgFieldScope s);
void pop_cfg_scope();
// Пункты меню для поля str (если оно в активном конфиге). true — значение где-то поменялось.
// label — подпись поля для отчёта и отмены; with_apply = false — только сброс (рассылку
// рисует меню вызывающего).
bool cfg_field_menu_items(std::string* str, const std::string& label, bool with_apply = true);
// Модель области, в чей активный конфиг входит str; nullptr — поле не из конфига вкладки.
AppModel* cfg_field_scope_model(const std::string* str);
// Поле, чьё ПКМ-меню сейчас рисуется (InputNumStr ставит его на время ctx_menu), — чтобы
// field_apply_all_menu дописал в своё меню пункты конфигов вкладки.
std::string* cfg_menu_field();
void set_cfg_menu_field(std::string* str);

template <class Cfg>
struct ConfigFieldScope {
    ConfigFieldScope(AppModel& m, const char* tab, std::vector<Cfg>& v, int active) {
        if (active < 0 || active >= (int)v.size()) return;
        static const Cfg def_cfg{};
        CfgFieldScope s;
        s.model = &m; s.tab = tab;
        s.base = reinterpret_cast<const char*>(&v[(size_t)active]);
        s.size = sizeof(Cfg);
        s.def = reinterpret_cast<const char*>(&def_cfg);
        s.active = active;
        s.count = [&v] { return (int)v.size(); };
        s.at = [&v](int i) { return reinterpret_cast<char*>(&v[(size_t)i]); };
        s.label = [&v](int i) { return v[(size_t)i].label; };
        push_cfg_scope(std::move(s));
        pushed = true;
    }
    ~ConfigFieldScope() { if (pushed) pop_cfg_scope(); }
    ConfigFieldScope(const ConfigFieldScope&) = delete;
    ConfigFieldScope& operator=(const ConfigFieldScope&) = delete;
    bool pushed = false;
};

bool field_restore_menu_items(const SystemEditor* model, BroadcastField f, const std::string& name,
                              std::map<std::string, std::string>& values, const std::vector<std::string>& names);


bool field_restore_popup(const char* popup_id, const SystemEditor* model, BroadcastField f, const std::string& name,
                         std::map<std::string, std::string>& values, const std::vector<std::string>& names);

void draw_param_tree(ParamGroups* groups, const std::vector<std::string>& names,
                     const std::function<void(const std::string&)>& item, bool in_table = false,
                     const std::function<void(const std::vector<std::string>&)>& group_menu = {});


void draw_param_groups_window(SystemEditor& m, const std::vector<std::string>& params,
                              const std::vector<std::string>& vars, const char* id);


[[nodiscard]] std::string field_label_auto(const std::string& name);


[[nodiscard]] std::string field_label(const FieldLabels* labels, const std::string& name);


void draw_field_name(const std::string& name, const FieldLabels* labels);


void set_field_latex(bool on);


void set_field_labels(const FieldLabels* labels);


bool var_combo_begin(const char* id, const std::string& cur, bool cur_is_name = true);


void var_combo_preview();


bool var_selectable(const std::string& name, bool selected);

// Ширина имени переменной так, как его рисуют превью комбо и var_selectable (формулой или текстом).
[[nodiscard]] float var_name_width(const std::string& name);


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

const SystemRecord* field_lib_record(const SystemEditor* m, bool reload) ;


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
