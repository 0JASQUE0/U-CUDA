#pragma once
// Функции вкладок (Phase, Parametric, DFT, Basins, FastSync, Order, Network, Neural, Custom, Library,
// System) и общее состояние окон интерфейса.
#include "gui_plot.h"

namespace ucuda::ui_detail {
std::string wrapper_coefficients_text(
        const std::string& name,
        const std::vector<CustomScheme>& custom_schemes,
        const std::map<std::string, std::string>& param_values);


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


std::string phase_axis_name(const PhaseAnalysisSession& s, int ax);


void phase_axis_combo(const char* id, PhaseAnalysisSession& s, int& axis);


void draw_phase_controls(PhaseAnalysisSession& s,
                                std::function<void()> on_reset_defaults,
                                AppModel* bc = nullptr,
                                const GuiCallbacks* cbp = nullptr);


std::function<const char*(int, int)> regime_cell_label(const std::vector<int>& flags,
                                                              int row_stride,
                                                              bool with_oscillation = false);


std::function<bool(int, int)> cancelled_cell_mask(const std::vector<int>& flags, int row_stride);


void draw_adaptive_stats(const std::vector<double>& st);


void draw_regime_summary(const std::vector<int>& flags, RegimeCounts* cache = nullptr, int gen = 0);


using ProjHookFn = std::function<void(int proj_index, const std::string& title)>;


struct PhaseSeriesStyle {
    ImVec4 color      = ImVec4(1, 1, 1, 1);
    bool   as_point   = false;   // только Phase2D; в time domain всегда линия
    float  point_size = 6.0f;
    int    marker     = (int)PointMarker::Circle;
};


using PhaseStyleFn = std::function<bool(int ic_index, PhaseSeriesStyle& out)>;


struct FeatureClusterParams {
    bool   valid         = false;
    double mult_peak     = 1.0;
    double mult_interval = 1.0;
    double eps           = 0.0;
};


void draw_projection_windows(PhaseAnalysisSession& s, const GuiCallbacks& cb,
                                    const ProjHookFn& before_begin = {},
                                    const ProjHookFn& after_begin  = {},
                                    const std::string& title_suffix = {},
                                    int owner_id_delta = 0,
                                    const PhaseStyleFn& style_fn = {},
                                    const FeatureClusterParams& clust = {});


void draw_writable_var_combo(const std::vector<std::string>& vars,
                                    int& writable_var,
                                    const char* combo_id);


void draw_parametric_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_dft1d_controls(AppModel& model, SystemLibrary& lib);


void draw_dft1d_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_basins_controls(AppModel& model, SystemLibrary& lib);


void draw_basins_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb);


void draw_basins_phase_windows(AppModel& model, const GuiCallbacks& cb);


void basins_phase_tick(AppModel& model);


void draw_fastsync_controls(AppModel& model, SystemLibrary& lib);


void draw_fastsync_plot(AppModel& model, const GuiCallbacks& cb);


int builtin_scheme_order(const std::string& name);


void draw_order_controls(AppModel& model, SystemLibrary& /*lib*/);


void draw_order_plot_windows(AppModel& model, const GuiCallbacks& cb);


static const char* const kNetTopologyNames[] = {
    "Custom", "Chain", "Ring", "Star", "Grid", "King graph", "All-to-all", "Small world"
};


double net_value_at(const NetworkResult& r, int p, int node, int var);


int net_degree(const NetworkConfig& c, int node);


void net_mark_custom(NetworkConfig& c);


std::string net_validate_laws(const NetworkSession& s, const NetworkConfig& c);


void draw_network_controls(AppModel& model, SystemLibrary& /*lib*/);
// Neural — большие сети: окно настроек и кривые (means: средние по популяциям, иначе выбранные узлы).
void draw_neural_controls(AppModel& model, SystemLibrary& lib);
enum { kNeuralMeans = 0, kNeuralNodes, kNeuralSync, kNeuralSweep, kNeuralInput, kNeuralLyap, kNeuralRate, kNeuralRaster, kNeuralWeights, kNeuralReadout, kNeuralWindowCount };
void draw_neural_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view, int what);
// Карта свипа по двум осям; клик по ячейке выбирает сеть в окнах от времени.
void draw_neural_map_window(AppModel& model, PlotRenderer& renderer, std::map<int, std::unique_ptr<HeatmapView>>& maps);
void draw_network_graph_window(AppModel& model);


void draw_network_series_window(AppModel& model, PlotRenderer& renderer, Plot2DView& view);


void draw_network_raster_window(AppModel& model, PlotRenderer& renderer, HeatmapView& view);


void draw_parametric_controls(AppModel& model, SystemLibrary& lib);


// Target — то, что принимает parse: сама сессия или её сохраняемое состояние (*State).
template <class Session, class Target>
static bool apply_session_json(AppModel& model,
                               const std::string& json,
                               Session& dst,
                               bool (*parse)(const std::string&, Target&),
                               const char* what)
{
    if (json.empty()) return true;
    Session probe;
    if (!parse(json, probe)) {
        model.session_load_warning =
            std::string("saved session '") + what + "' is corrupt - ignored";
        std::printf("[session] %s: parse failed, keeping defaults from record\n", what);
        std::fflush(stdout);
        return false;
    }
    if (!parse(json, dst)) {                 // не должно случаться, см. выше
        model.session_load_warning =
            std::string("saved session '") + what + "' failed to apply";
        return false;
    }
    model.session_load_warning.clear();
    return true;
}


void order_autosave_seed(AppModel& model);



void draw_custom_mode_layout(AppModel& model, SystemLibrary& lib,
                                    const GuiCallbacks& cb,
                                    ImVec2 area_pos, ImVec2 area_size);

// Индикатор компьюта в верхней строке. Раньше состояние снималось лестницей из двенадцати
// почти одинаковых веток: каждая доставала из своей сессии одну и ту же четвёрку — подпись
// запущенного конфига, время старта, cancel_token, progress_token. Ветки успели разойтись:
// три подставляли запасное имя только при выходе индекса за границы, три — ещё и при пустом label.
enum class BusyKind { None, Bif, LLE, LS, Dft1D, Basins, Phase, FastSync, Custom, Order, Metrics, Neural };


struct BusyInfo {
    BusyKind    kind = BusyKind::None;
    std::string what;
    std::chrono::steady_clock::time_point start{};
    bool  cancelling = false;   // пользователь нажал Stop, ждём движок
    float progress   = 0.0f;    // 0..1 из session.progress_token
};


template <class Vec>
[[nodiscard]] static std::string running_label(const Vec& items, int idx, const char* fallback) {
    if (idx >= 0 && idx < (int)items.size() && !items[idx].label.empty())
        return items[idx].label;
    return fallback;
}


template <class Session>
[[nodiscard]] static BusyInfo busy_from(const Session& s, BusyKind kind, std::string what) {
    BusyInfo b;
    b.kind  = kind;
    b.what  = std::move(what);
    b.start = s.compute_start_time;
    b.cancelling = s.cancel_token && s.cancel_token->load(std::memory_order_relaxed);
    if (s.progress_token)
        b.progress = s.progress_token->load(std::memory_order_relaxed);
    return b;
}


[[nodiscard]] TrajWork running_traj_work(const AppModel& m);


[[nodiscard]] TrajWork current_tab_traj_work(const AppModel& m);


void sweep_range_from_result(double res_lo, double res_hi,
                                    const std::string& lo_text, const std::string& hi_text,
                                    double& lo, double& hi) ;


static auto& basins_heatmaps() { return ui_resource<std::map<unsigned, std::unique_ptr<HeatmapView>>, 1000>(); }





void count_regimes(const std::vector<int>& flags, int& fp, int& unb, int& osc, int* stopped = nullptr) ;


[[nodiscard]] std::string ic_legend_text(const AnalysisResult& res, size_t k) ;


void draw_shared_config(AppModel& model, CustomSession& cs,
                        const std::vector<CustomScheme>& custom_schemes) ;


void draw_level2d_detail(AppModel& model, CustomSession& cs) ;


void draw_level1d_detail(CustomSession& cs) ;


void draw_level3_detail(AppModel& model, CustomSession& cs) ;


void draw_custom_controls(AppModel& model, SystemLibrary& lib) ;


void wire_2d_heatmap_interaction(HeatmapView& hv, CustomSession& cs,
                                 std::deque<CustomQueueItem>& q) ;


[[nodiscard]] ImGuiID ws_dock_id(int tab_id, const std::string& sys) ;


[[nodiscard]] std::string custom_win_suffix(const std::string& sys) ;


void draw_custom_controls_panel(AppModel& model, SystemLibrary& lib,
                                       ImVec2 pos, ImVec2 size) ;


struct WsRenameState {
    int  target_id = 0;              // 0 = no rename in progress.
    char buf[128]  = {};
};


void draw_custom_workspace_panel(CustomSession& cs,
                                        const std::string& sys,
                                        ImVec2 pos, ImVec2 size) ;


void draw_custom_plot_windows(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb) ;


void draw_dft1d_diagram_controls(AppModel& model, Dft1DAnalysisSession& s, int idx) ;


void draw_dft1d_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                           Dft1DPlotWindow& win,
                           PlotRenderer& renderer,
                           std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void basins_colorbar_range(const BasinsConfig& c, double& vmin, double& vmax) ;


[[nodiscard]] unsigned basins_base_oid(int config_index) ;


ImVec4 basins_id_color(const BasinsConfig& c, const AppModel& model,
                              int config_index, int display_id,
                              double vmin, double vmax) ;


void draw_basins_phase_controls(AppModel& model, int cfg_idx) ;


void draw_network_topology_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_coupling_block(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_nodes_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_network_edges_table(AppModel& model, NetworkSession& s, NetworkConfig& c) ;


void draw_order_target_combo(const char* label, int& target,
                                    const OrderAnalysisSession& s) ;


void draw_order_axis_block(const char* id, const char* title,
                                  const OrderAnalysisSession& s,
                                  int& target, std::string& lo, std::string& hi,
                                  bool& log_grid, std::string* n_text,
                                  bool enabled, const char* n_label = "points") ;


void draw_order_steps_hint(const OrderConfig& c, const OrderAnalysisSession& s) ;


void draw_order_stability_block(AppModel& model, const OrderAnalysisSession& s,
                                       OrderConfig& c) ;


OrderPlotWindow::Kind order_kind_for(const OrderConfig& c) ;


void order_rehome_config(AppModel& model, int idx) ;


void order_float_build_warning() ;


const char* order_window_kind_name(OrderPlotWindow::Kind k) ;


bool order_member_fits(const OrderConfig& c, OrderPlotWindow::Kind k) ;


ImVec4 order_shade(ImVec4 c, float k) ;


int order_marker_for(int within) ;


ImVec4 order_base_color(int idx, int palette) ;


void draw_order_status_line(const char* label, int n_ok, int n_div, int n_floor, int n_noc) ;


data_export::OrderSnapshot order_snapshot(const OrderAnalysisSession& s,
                                                 const OrderConfig& c,
                                                 bool two_d, bool gpu_fmad, bool gpu_rdc) ;


void draw_order_export_items(AppModel& model, const OrderPlotWindow& win,
                                    const GuiCallbacks& cb, bool busy) ;


void draw_order_stability_map(AppModel& model, OrderPlotWindow& win,
                                     const GuiCallbacks& cb,
                                     PlotRenderer& renderer, HeatmapView& hv,
                                     OrderConfig& c, int mi) ;


void draw_order_map_window(AppModel& model, OrderPlotWindow& win,
                                  const GuiCallbacks& cb,
                                  PlotRenderer& renderer, HeatmapView& hv) ;


void draw_order_curve_window(AppModel& model, OrderPlotWindow& win,
                                    const GuiCallbacks& cb,
                                    PlotRenderer& renderer, Plot2DView& view) ;


void draw_diagram_controls(AppModel& model, BifurcationAnalysisSession& s, int idx) ;


void draw_bifurcation_controls(AppModel& model, SystemLibrary& /*lib*/) ;


ParamPlotKind to_param_plot_kind(ParametricPlotWindow::Kind kind) ;


void configure_plot_view(Plot2DView& view, ParametricPlotWindow::Kind kind) ;


void draw_bifurcation_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                                   ParametricPlotWindow& win,
                                   PlotRenderer& renderer, Plot2DView& view,
                                   std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_lle_curve_controls(AppModel& model, LLEAnalysisSession& s, int idx) ;


void draw_lle_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_lle_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                          const ParametricPlotWindow& win,
                          PlotRenderer& renderer, Plot2DView& view,
                          std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


void draw_ls_curve_controls(AppModel& model, LyapunovSpectrumAnalysisSession& s, int idx) ;


void draw_ls_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_ls_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                         const ParametricPlotWindow& win,
                         PlotRenderer& renderer, Plot2DView& view,
                         std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


const char* signal_metric_tooltip(int m) ;


void draw_metrics_config_controls(AppModel& model, SignalMetricsAnalysisSession& s, int idx) ;


void draw_metrics_controls(AppModel& model, SystemLibrary& /*lib*/) ;


void draw_metric_tab_bar(AppModel& model, ParametricPlotWindow& win, int avail_mask) ;


void draw_metrics_plot(AppModel& model, SystemLibrary& lib, const GuiCallbacks& cb,
                              ParametricPlotWindow& win,
                              PlotRenderer& renderer, Plot2DView& view,
                              std::map<int, std::unique_ptr<HeatmapView>>& heatmap_map) ;


struct ParamPlotMatchItem { int index; std::string label; };


[[nodiscard]] std::vector<ParamPlotMatchItem>
parametric_matching_items(AppModel& model, ParametricPlotWindow::Kind kind, bool mode_2d) ;


int parametric_type_index_of(ParametricPlotWindow::Kind kind, bool mode_2d) ;


void parametric_type_from_index(int t, ParametricPlotWindow::Kind& kind, bool& mode_2d) ;


bool draw_parametric_window_setup_row(AppModel& model, ParametricPlotWindow& win) ;


void draw_generated_code_block(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_wrapper_coefficients(const std::string& name,
                                      const std::vector<CustomScheme>& custom_schemes,
                                      const std::map<std::string, std::string>& param_values,
                                      const char* id) ;


void begin_coeff_column(const char* id, int rows, int max_rows) ;


std::string wrapper_name_field(const char* id, std::string& text,
                                      const std::string& default_name) ;


bool extr_base_needs_half_s(const std::string& nm) ;


void draw_extrapolation_builder(SystemEditor& model) ;


void comp_load_coeffs_from_file(SystemEditor& model, const std::string& path) ;


void draw_composition_builder(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_wrapper_list(SystemEditor& model) ;


void draw_builtin_scheme_checkboxes(SystemEditor& model) ;


void draw_dae_method_checkboxes(SystemEditor& model) ;


void draw_system_tab(SystemEditor& model, const GuiCallbacks& cb) ;


void draw_parameters_tab(SystemEditor& model) ;


[[nodiscard]] std::string trim_copy(const std::string& s) ;


void invalidate_library_cache() ;


void persist_hidden_systems(const AppModel& model) ;


std::string describe_record(const SystemRecord& r) ;


void library_editor_save(AppModel& model, SystemLibrary& lib) ;

inline LibraryListCache g_library_cache;


inline WsRenameState g_ws_rename;


inline const char* WS_DRAG_PAYLOAD = "CUSTOM_WS_WIN";


inline const char* kParametricTypeNames[] = {
    "Bifurcation 1D", "Bifurcation 2D", "LLE 1D", "LLE 2D", "LS 1D", "LS 2D",
    "Metrics 1D", "Metrics 2D"
};
}
