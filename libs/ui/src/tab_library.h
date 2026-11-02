#pragma once
// Вкладка Library: список систем, его кэш, редактор записи.
#include "gui_common.h"

namespace ucuda::ui_detail {
enum class LibTab : char { Ode = 0, Dae = 1, Map = 2 };


static constexpr int kLibTabCount = 3;


static const char* const kLibTabName[kLibTabCount] = { "ODE", "DAE", "MAP" };


LibTab lib_tab_of(bool is_dae, bool is_map);


struct LibraryListCache {
    std::vector<std::string> names;
    std::vector<int>         custom_counts;   // параллелен names
    std::vector<char>        tab;             // параллелен names: LibTab записи
    std::string              sel_name;      // для какой системы прочитано превью ниже
    bool                     sel_ok = false;  // запись прочиталась
    SystemRecord             sel_rec;
    std::string              sel_desc;        // describe_record(sel_rec), готов для буфера обмена
    double                   filled_at = -1.0;
    bool                     valid     = false;
};


const LibraryListCache& library_list_cached(SystemLibrary& lib);


void draw_library_list(AppModel& model, SystemLibrary& lib, LibTab tab);


void draw_library_editor(AppModel& model, SystemLibrary& lib,
                                const GuiCallbacks& cb);


[[nodiscard]] std::string trim_copy(const std::string& s) ;


void invalidate_library_cache() ;


void persist_hidden_systems(const AppModel& model) ;


std::string describe_record(const SystemRecord& r) ;


void library_editor_save(AppModel& model, SystemLibrary& lib) ;


inline LibraryListCache g_library_cache;
}
