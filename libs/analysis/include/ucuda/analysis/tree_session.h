#pragma once
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/domain/analysis_tree.h"
#include "ucuda/domain/system_record.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

class ParametricEngine;

// Подсессия листа: ровно одна из сессий заполнена, у неё один конфиг (индекс 0).
struct TreeLeafRuntime {
    std::unique_ptr<BifurcationAnalysisSession>      bif;
    std::unique_ptr<LLEAnalysisSession>              lle;
    std::unique_ptr<LyapunovSpectrumAnalysisSession> ls;
    std::unique_ptr<PhaseAnalysisSession>            phase;
    std::unique_ptr<BasinsAnalysisSession>           basins;
    // Подпись входов последнего успешного расчёта; pending — запущенного.
    std::string committed;
    std::string pending;
    bool        committed_ok  = false;
    bool        pending_armed = false;

    [[nodiscard]] bool in_flight() const;
    [[nodiscard]] bool last_ok() const;
    [[nodiscard]] std::string last_error() const;
    void request_cancel();
};

struct TreeSession {
    // Идентичность системы — как в CustomState.
    std::vector<std::string>  vars;
    std::vector<std::string>  params;
    System                    sys;
    std::vector<CustomScheme> custom_schemes;
    std::vector<std::string>  wrapper_schemes;
    std::vector<std::string>  enabled_builtin_schemes;
    std::string               loaded_system_name;
    SystemRecord              record;   // для заполнения подсессий новых листов

    AnalysisTree tree;
    std::map<int, TreeLeafRuntime> leaves;

    // Ширина левой панели не нужна: раскладка общая. Флаг пересохранения _last_tree.json.
    bool dirty = false;

    TreeSession() = default;
    TreeSession(TreeSession&&) = default;
    TreeSession& operator=(TreeSession&&) = default;
    TreeSession(const TreeSession&) = delete;
    TreeSession& operator=(const TreeSession&) = delete;

    // Сброс к дереву по умолчанию (как раскладка Custom).
    void load_from_record(const SystemRecord& r,
                          const std::vector<std::string>& vars_,
                          const std::vector<std::string>& params_);
    // Раздать vars/params/sys/схемы во все подсессии.
    void propagate_identity();
    // Создать подсессии для листьев, у которых их нет (после загрузки дерева из файла).
    void ensure_leaf_runtimes();

    // --- Структура ---
    // Новый узел последним ребёнком parent; возвращает id или 0. slice_dir — для Slice1D.
    int  add_node(int parent, TreeNodeKind kind, int slice_dir = 0);
    int  ensure_layout_leaf(const std::vector<TreeLayoutNode>& path);
    [[nodiscard]] bool workspace_window_visible(int id, const std::string& key, bool available = true) const;
    [[nodiscard]] bool leaf_window_visible(int id, int view = 0) const;
    bool show_leaf_window(int id, int view = 0);
    // Restore the primary view if hidden. Otherwise append another view of the same result.
    // Returns the view index, or -1 if the leaf cannot create an extra view.
    int open_or_add_leaf_window(int id);
    bool remove_extra_view(int id, int view);
    // Копия поддерева сразу после оригинала; возвращает id копии или 0.
    int  duplicate_node(int id);
    // Удалить поддерево (не корень). Возвращает false, если что-то из него считается.
    bool remove_node(int id);
    void toggle_enabled(int id);
    void move_node(int id, int delta);   // -1 вверх, +1 вниз среди братьев
    [[nodiscard]] bool subtree_in_flight(int id) const;

    // --- Расчёт ---
    // Эффективный контекст листа: System + ближайший регион + срез + точка, с пинами.
    [[nodiscard]] CustomTabSharedConfig effective_shared(int leaf_id) const;
    [[nodiscard]] std::string sweep_axis_label(const EffectiveSweep& axis) const;
    [[nodiscard]] std::string sweep_window_label(int leaf_id) const;
    // Подпись входов листа (без значений осей, по которым он сам свипует).
    [[nodiscard]] std::string leaf_signature(int leaf_id) const;
    [[nodiscard]] bool leaf_dirty(int leaf_id) const;
    // Активные грязные листья поддерева в порядке обхода.
    [[nodiscard]] std::vector<int> dirty_leaves(int subtree_root) const;
    // Активные грязные листья, чьи срез или точка следуют за курсором владельца и живые.
    [[nodiscard]] std::vector<int> cursor_leaves(int owner_id) const;
    // Собрать конфиг и запустить расчёт листа.
    bool start_leaf(int leaf_id, ParametricEngine& engine);
    [[nodiscard]] bool any_in_flight() const;
    void request_cancel_all();
    // Опросить подсессии; true, если что-то завершилось.
    bool poll_all();

    // ФП по бассейнам у листьев Basins: опрос, автозапуск после паузы, старт очереди.
    // now — секунды монотонных часов. true, если какой-то портрет досчитался.
    bool tick_basins_phase(double now);
    // Включить свой интегратор листа (или 1D-интегратор среза), начав с того, что он наследует сейчас.
    void enable_own_integration(int leaf_id);

    // Ближайший регион-предок (включая сам узел); 0 — нет.
    [[nodiscard]] int region_of(int id) const;
    // Чей курсор управляет узлом: регион, а без него — срез под System.
    [[nodiscard]] int cursor_owner(int id) const;
    // Зажать курсоры регионов в их диапазоны.
    void clamp_cursors();
};

// Те же три подстановки, что в очереди Custom (app_queues.cpp): значение узла сетки в поле,
// которое задаёт закреплённую ось.
void tree_pin_axis(CustomTabSharedConfig& s, const std::vector<std::string>& params,
                   const std::vector<std::string>& vars, const EffectiveSweep& e, double value,
                   bool l1d_h);
