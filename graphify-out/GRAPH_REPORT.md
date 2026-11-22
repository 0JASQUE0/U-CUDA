# Graph Report - U-CUDA  (2026-11-22)

## Corpus Check
- Large corpus: 419 files · ~809,745 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 11440 nodes · 25027 edges · 404 communities (375 shown, 29 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 4195 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- application: AppModel
- ui: plot_widgets.cpp
- resources: ucuda_adaptive.cuh
- cuda: EngineModuleCache
- plotting: PlotRenderer
- ui: item_w
- cuda: numb
- application: SystemEditor
- ?: algorithm
- domain: SystemRecord
- cuda: fail
- ui: gui_common.h
- resources: UcudaBdfCore
- resources: ucuda_hp.h
- equations: adaptive_codegen.cpp
- resources: neural.template.cu
- analysis: neural_session.cpp
- ui: tree_tab.cpp
- plotting: Plot2DView
- domain: NeuralConfig
- ui: neural.cpp
- domain: NeuralRequest
- plotting: HeatmapView
- domain: NeuralOutcome
- domain: BifurcationDiagramConfig
- domain: map
- domain: neural_classifier_test.cpp
- desktop: ui_test_script.cpp
- domain: OrderConfig
- plotting: plot_axis.cpp
- cuda: ParametricEngine::Impl
- cuda: engine_runtime.cpp
- domain: CustomTabSharedConfig
- storage: data_export.cpp
- equations: codegen_scheme
- domain: adaptive_settings.h
- domain: BifurcationAnalysisState
- ?: app_model.h
- cuda: KrsCpuStep
- domain: OrderPlotWindow
- cuda: PeakStream
- resources: PeakStreamNU
- numerics: krs_cpu.cpp
- domain: LSCurveConfig
- equations: PN
- domain: BasinsConfig
- desktop: hud_theme.cpp
- plotting: Plot2DView::render
- domain: FastSyncConfig
- domain: Dft1DConfig
- domain: LLECurveConfig
- domain: PhaseAnalysisState
- ui: draw_gui
- numerics: neural_readout.h
- domain: CustomSession
- domain: SignalMetricsConfig
- numerics: SystemEvaluator
- analysis: TreeSession
- domain: NeuralPlan
- numerics: string
- storage: AppConfig
- analysis: tree_session.cpp
- domain: NeuralTask
- tests: artifacts.h
- domain: NeuralTaskResult
- domain: TreeNodeKind
- numerics: Builder
- ui: widgets.cpp
- resources: NeuralSpikeArgs
- plotting: plot_renderer.cpp
- domain: Projection
- domain: configCUDA.h
- numerics: IntScheme
- domain: NeuralResult
- desktop: main
- analysis: session_helpers.cpp
- analysis: custom_session.cpp
- domain: NeuralSource
- analysis: order_session.cpp
- analysis: PhaseRunInputs
- application: app_queues.cpp
- application: AppSettings
- domain: PerfResult
- ?: platform/platform_paths.h
- cuda: EngineSources
- tests: observations.cpp
- analysis: basins_session.cpp
- domain: PerfRequest
- storage: AppModel::tick
- domain: CircuitGraph
- resources: NeuralStepArgs
- analysis: PhaseAnalysisSession
- domain: FastSyncRequest
- storage: session_json.cpp
- ui: combo_items_need
- application: TrajTimeEstimator
- cuda: parametric_engine.cpp
- tests: dae_test.cpp
- equations: Scheme
- tests: spice_test.cpp
- plotting: PlotSeriesInput
- plotting: Plot3DView
- desktop: Sweep
- analysis: BasinsAnalysisSession
- domain: Bifurcation2DRequest
- domain: SignalMetricsRequest
- analysis: neural_results_test.cpp
- analysis: OrderAnalysisSession
- domain: AnalysisResult
- ?: TEST
- analysis: SignalMetricsAnalysisSession
- cuda: PhaseAdaptiveRequest
- domain: NetworkConfig
- domain: Config
- tests: compare
- domain: StabilityRequest
- cuda: AdaptiveStats
- cuda: NvrtcEngine
- domain: LLE2DRequest
- numerics: extract_quadratic
- domain: Metrics
- tests: adaptive_test.cpp
- cuda: RunSignals
- domain: BasinsRequest
- domain: Dft1DRequest
- equations: Parser
- analysis: build_bif1d_request
- equations: sysparse.cpp
- domain: LS2DRequest
- domain: MnaLayout
- domain: FastSyncSnapshot
- tests: neural_cuba_test.cpp
- cuda: CircuitGpuRunner
- domain: StabilityResult
- cuda: nvrtc_engine.cpp
- resources: isinf
- domain: Bifurcation1DRequest
- domain: LLE1DRequest
- numerics: CircuitSolver
- storage: OrderSnapshot
- ui: draw_config_tab_bar
- ui: CfgFieldScope
- desktop: app_main.cpp
- equations: codegen.cpp
- domain: AdaptiveRequest
- domain: LS1DRequest
- domain: AdaptiveSettings
- analysis: TrajWork
- equations: parse_dae
- cuda: BdfGpuRequest
- domain: BasinsAnalysisState
- domain: BasinsResult
- domain: NeuralSpiking
- plotting: Plot3DView::render
- ?: TEST_F
- domain: OrderResult
- domain: NeuralReadoutSaved
- domain: param_groups.h
- ui: draw_composition_builder
- ui: ViewRangeValues
- ui: dock_layout.cpp
- domain: EliminationPlan
- equations: AdaptiveCode
- equations: FuncId
- storage: system_library.cpp
- resources: MetricsAccumNU
- cuda: EngineModuleBuilder
- ocr: ocr_client_win.cpp
- tests: neural_forecast_test.cpp
- cuda: engine_impl.h
- ui: LineStyleFields
- domain: ErrTimeResult
- domain: num_parse.h
- equations: SystemEvaluator::Impl
- ui: LeafViews
- tests: library_test.cpp
- desktop: paint_sith_scene
- ?: main.cpp
- tests: Comparison
- tests: architecture_test.cpp
- application: NetworkViewState
- domain: SignalMetricsAnalysisState
- domain: BasinsSnapshot
- equations: SparseLu
- numerics: CpuAdaptiveKrs
- tests: test_util.h
- desktop: paint_jedi_scene
- application: AppModel::broadcast_field
- cuda: ParametricEngine
- domain: string
- domain: FastSyncResult
- domain: NetworkRequest
- numerics: NetlistOptions
- numerics: .element
- numerics: SpiceImportResult
- numerics: Deck
- plotting: PlotWindowPrefs
- tests: main
- domain: Kind
- analysis: dft_session.cpp
- domain: SignalMetricsResult
- domain: Bif2DSnapshot
- domain: NetworkState
- storage: JP
- tests: SolverWorkload
- analysis: BifurcationAnalysisSession
- analysis: LLEAnalysisSession
- analysis: LyapunovSpectrumAnalysisSession
- analysis: net_generate_topology
- application: AppMode
- application: BroadcastSlots
- domain: OrderRequest
- domain: Dft1DAnalysisState
- domain: ErrTimeRequest
- domain: LS2DSnapshot
- domain: OrderAnalysisState
- numerics: bdf_integrate
- numerics: CondRewriter
- numerics: AdaptiveCpuModule
- application: BroadcastField
- application: DaeEditorState
- application: string
- domain: FastSyncAnalysisState
- domain: NetworkResult
- domain: LLE1DSnapshot
- domain: LLE2DSnapshot
- domain: LS1DSnapshot
- equations: DaeScheme
- equations: adaptive_emb_body
- equations: Node
- numerics: Emitter
- ui: BasinsPlotViews
- ui: draw_custom_workspace_panel
- analysis: Dft1DAnalysisSession
- analysis: TreeLeafRuntime
- application: NeuralViewState
- domain: Bif1DSnapshot
- domain: Dft1DSnapshot
- domain: NeuralGraph
- plotting: plot_camera_3d.cpp
- plotting: plot_legend.cpp
- resources: UcudaCtlConst
- application: string #2
- application: compute_services.cpp
- domain: GpuCalibration
- equations: system_evaluator.cpp
- domain: LS2DResult
- domain: SignalMetric
- desktop: hud_scene.h
- application: .build_system
- domain: Bifurcation2DResult
- domain: Dft1DResult
- domain: LLE2DResult
- domain: CircuitRunResult
- domain: ucuda_node_value
- domain: PhaseSnapshot
- domain: NeuralSweep
- equations: dae_devices.cpp
- plotting: GpuLineSeriesSet
- ui: ScatterClusterCache
- ui: DockWindowFactory
- ui: LNode
- ui: draw_library_list
- tests: gen_lorenz_reference.py
- domain: NeuralDeviceModel
- cuda: CpuMetricsAccum
- domain: BasinFeature
- domain: Bifurcation1DResult
- plotting: AxisInfo
- plotting: plot_series_colors_changed
- domain: net_default_law
- domain: NewtonSettings
- application: CamoTheme
- domain: vector
- domain: LS1DResult
- equations: OpCode
- equations: RosTab
- numerics: BdfStats
- platform: FileImageSource
- platform: Process
- storage: SystemLibrary
- desktop: Handle
- desktop: string
- application: run_next_queued
- cuda: host_rqa.cu
- plotting: PlotSeriesInput3D
- ui: string
- ui: Request
- ui: OrderCurveData
- ui: LibraryListCache
- desktop: CamoPalette
- tools: mnist_subset.py
- analysis: fastsync_session.cpp
- numerics: run_adaptive_phase
- application: BroadcastTab
- application: PrewarmWatch
- cuda: CircuitGpuRunner::build
- cuda: MetricsAxes
- domain: OrderAxis
- numerics: Eval
- plotting: PlotCamera3D
- ui: gui_tree_state.h
- resources: ExtremaInterp
- desktop: Options
- tools: graphify_update.py
- domain: NetTopology
- application: ParamGroupsUi
- domain: CircuitResistor
- numerics: SpiceImportOptions
- storage: record_from_json
- tools: ocr_server.py
- domain: NeuralRule
- application: ParametricQueueItem
- cuda: CpuExtremaInterp
- numerics: EmittedKernel
- numerics: Model
- plotting: GpuLineSeriesSet3D
- storage: session_to_json_order_windows
- resources: UcudaLyapClones
- tests: NetworkFixture
- desktop: Flight
- domain: RegimeCounts
- equations: CdKind
- plotting: PointMarker
- storage: write_network_config
- ui: dock_layout_process
- ui: BifurcationPointGrid
- ui: TreeColumn
- tests: vector
- application: AppModel::load_or_init_dft1d_plot_windows
- application: PendingScreenshot
- application: ThemeColorSlot
- application: AppModel::broadcast_targets
- application: cleanup_plot_windows_after_removal
- domain: CircuitMultiplier
- domain: NetNode
- numerics: circuit_netlist.cpp
- plotting: PlotRenderer::PlotRenderer
- resources: lyapunov_adaptive_part.cu
- tests: regression_main.cpp
- desktop: .next_tab
- analysis: ClassSteps
- application: OcrState
- numerics: Inductor
- plotting: GpuLineSeries3D
- plotting: gpu_line_series.cpp
- plotting: InkBox
- ui: DockLayoutPreset
- resources: fastsync_attr.template.cu
- ?: main
- desktop: ShipDef
- application: BroadcastTarget
- application: ThemeColors
- numerics: RefRow
- numerics: ErkTable
- plotting: digit_input.cpp
- plotting: SeriesPaletteId
- plotting: MathMark
- storage: session_to_json_dft1d_windows
- ui: SplitMemo
- resources: basins.template.cu
- resources: bifurcation1d_cont.template.cu
- resources: bifurcation1d.template.cu
- resources: bifurcation2d.template.cu
- resources: dft1d_cont.template.cu
- resources: dft1d_hsweep.template.cu
- resources: fastsync_grid.template.cu
- resources: lle1d_cont.template.cu
- resources: lle1d.template.cu
- resources: lle2d.template.cu
- resources: ls1d_cont.template.cu
- resources: ls1d.template.cu
- resources: ls2d.template.cu
- application: LibraryEditMode
- application: InputMode
- equations: SchemeRow
- numerics: DopriTable
- numerics: Edge
- tests: Stats
- vcpkg.json: vcpkg.json
- application: OcrFn
- plotting: LogAxisTick
- plotting: PlotRenderer::draw_polyline_2d
- plotting: PrefsField
- application: BasinsQueueItem
- application: Dft1DQueueItem
- application: FastSyncQueueItem
- application: NetworkQueueItem
- application: OrderQueueItem
- plotting: .end_oit
- plotting: .effective_scale
- ?: param_value_at
- U-CUDA.sln: U-CUDA.sln

## God Nodes (most connected - your core abstractions)
1. `numb` - 280 edges
2. `AppModel` - 219 edges
3. `NeuralConfig` - 176 edges
4. `PlotRenderer` - 160 edges
5. `SystemEditor` - 143 edges
6. `Plot2DView` - 115 edges
7. `SystemRecord` - 114 edges
8. `map` - 114 edges
9. `CustomTabSharedConfig` - 101 edges
10. `draw_gui()` - 99 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `texture`  [INFERRED]
  libs/plotting/src/plot_renderer.cpp → apps/desktop/app_main.cpp
- `main()` --calls--> `release_gui_resources()`  [INFERRED]
  apps/desktop/app_main.cpp → libs/ui/src/ui_resources.cpp
- `draw_order_stability_map()` --calls--> `edit`  [INFERRED]
  libs/ui/src/order.cpp → apps/desktop/ui_test/ui_tests_sweep.cpp
- `loopCalculateDiscreteModelMetrics_core()` --calls--> `sampleStep`  [INFERRED]
  resources/kernels/signal_metrics.template.cu → libs/cuda/src/device/fastsync.cuh
- `evaluate()` --calls--> `plan`  [INFERRED]
  tests/neural_forecast_test.cpp → libs/equations/src/expression_internal.h

## Import Cycles
- None detected.

## Communities (404 total, 29 thin omitted)

### Community 0 - "application: AppModel"
Cohesion: 0.01
Nodes (93): AppModel, app_mode, basins_queue_, basins_session, bifurcation_session, broadcast_pick, broadcast_pick_open_all, broadcast_pick_sig (+85 more)

### Community 1 - "ui: plot_widgets.cpp"
Cohesion: 0.04
Nodes (120): regime_is_cancelled(), regime_is_fixed_point(), regime_is_oscillation(), regime_is_unbound(), regime_label(), signal_metric_axis_label(), is_slancm_id(), slancm_id() (+112 more)

### Community 2 - "resources: ucuda_adaptive.cuh"
Cohesion: 0.04
Nodes (128): ucuda_ctrl_custom(), ucuda_ctrl_custom_prep(), attempt(), ucuda_ad_advance_to(), ucuda_ad_apply_axis(), ucuda_ad_dae_project(), ucuda_ad_eval(), ucuda_ad_h0() (+120 more)

### Community 3 - "cuda: EngineModuleCache"
Cohesion: 0.02
Nodes (134): cache_store(), CachedAdModule, kernel_avg, kernel_dbscan, kernel_dbscan_warp, kernel_dbscan_warp2d, kernel_max_peaks, kernel_peaks (+126 more)

### Community 4 - "plotting: PlotRenderer"
Cohesion: 0.02
Nodes (109): PlotRenderer, color_tex_, depth_rbo_, fbo_, fbo_h_, fbo_has_depth_, fbo_w_, frame_clear_alpha_ (+101 more)

### Community 5 - "ui: item_w"
Cohesion: 0.06
Nodes (103): order_steps_for_h(), cpu_compiler_available(), CustomScheme, body, enabled, name, order, symmetric (+95 more)

### Community 6 - "cuda: numb"
Cohesion: 0.05
Nodes (108): log, avgPeakFinderCUDA(), avgPeakFinderCUDA_for2Dbif(), avgPeakFinderCUDA_logMaximas(), calculateDiscreteModelAvgPeaksCUDA(), calculateDiscreteModelforFastSynchroCUDA(), compute_basin_feature(), CUDA_dbscan_kernel() (+100 more)

### Community 7 - "application: SystemEditor"
Cohesion: 0.02
Nodes (106): SystemEditor, alg_text, alphabet_text, cd_fp_iters, circuit_ideal_bjts, circuit_ideal_diodes, circuit_ideal_opamps, circuit_netlist (+98 more)

### Community 8 - "?: algorithm"
Cohesion: 0.05
Nodes (24): log_run_completed(), NetworkSession::poll(), set, AppModel::redo_last(), AppModel::undo_last(), pop_apply_move(), AppModel::add_dft1d_plot_window(), AppModel::add_order_plot_window() (+16 more)

### Community 9 - "domain: SystemRecord"
Cohesion: 0.03
Nodes (92): BasinsAnalysisSession::load_from_record(), BifurcationAnalysisSession::load_from_record(), Dft1DAnalysisSession::load_from_record(), FastSyncAnalysisSession::load_from_record(), LLEAnalysisSession::load_from_record(), LyapunovSpectrumAnalysisSession::load_from_record(), SignalMetricsAnalysisSession::load_from_record(), NetworkSession::load_from_record() (+84 more)

### Community 10 - "cuda: fail"
Cohesion: 0.08
Nodes (56): ParametricEngine::Impl::prewarm_basins(), ParametricEngine::Impl::run_basins(), ParametricEngine::Impl::prewarm_bif1d(), ParametricEngine::Impl::prewarm_bif2d(), ParametricEngine::Impl::run_bif1d(), ParametricEngine::Impl::run_bif1d_continuation(), ParametricEngine::Impl::run_bif1d_continuation_ad(), ParametricEngine::Impl::run_bif1d_cpu_ad() (+48 more)

### Community 11 - "ui: gui_common.h"
Cohesion: 0.04
Nodes (53): write_synthesized_netlist(), basins_base_oid(), basins_id_color(), basins_plot_color_range(), draw_basins_phase_windows(), configure_plot_view(), to_param_plot_kind(), custom_win_suffix() (+45 more)

### Community 12 - "resources: UcudaBdfCore"
Cohesion: 0.04
Nodes (91): ucuda_ad_advance_to(), ucuda_ad_eval(), ucuda_ad_init(), ucuda_ad_restart(), ucuda_ad_step(), ucuda_bdf_begin(), ucuda_bdf_finite(), ucuda_bdf_fixed_collect() (+83 more)

### Community 13 - "resources: ucuda_hp.h"
Cohesion: 0.08
Nodes (81): dft1dContinuationKernel(), dft_window(), abs(), acos(), aint(), asin(), atan(), atan2() (+73 more)

### Community 14 - "equations: adaptive_codegen.cpp"
Cohesion: 0.07
Nodes (90): perf_rhs_per_step(), SystemEditor::circuit_layout_defaults(), cd_fp_slot(), newton_slot(), System, cd_fp_iters, dae_index, dae_nx (+82 more)

### Community 15 - "resources: neural.template.cu"
Cohesion: 0.03
Nodes (93): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t, neuralArcArrive(), neuralArcClip(), neuralArcSpike(), neuralArcTraces() (+85 more)

### Community 16 - "analysis: neural_session.cpp"
Cohesion: 0.08
Nodes (75): build_rows(), class_config(), class_evaluate(), class_input_hold(), class_protocol(), class_split(), class_steps(), dataset_classes() (+67 more)

### Community 17 - "ui: tree_tab.cpp"
Cohesion: 0.05
Nodes (76): TreeSession::enable_own_integration(), tree_kind_is_1d_leaf(), tree_kind_is_context(), TreeNode, cfg, children, enabled, expanded (+68 more)

### Community 18 - "plotting: Plot2DView"
Cohesion: 0.02
Nodes (83): LineIdxCache, built, count, first, key, xys, Plot2DView, crosshair_along_y (+75 more)

### Community 19 - "domain: NeuralConfig"
Cohesion: 0.02
Nodes (87): NeuralConfig, arcs, build_seconds, cd_fp_iters, checkpoint, checkpoint_from_result, checkpoint_path, classifier (+79 more)

### Community 20 - "ui: neural.cpp"
Cohesion: 0.04
Nodes (63): NeuralSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, result_serial, run_future (+55 more)

### Community 21 - "domain: NeuralRequest"
Cohesion: 0.03
Nodes (83): NeuralCheckpoint, data, shape, step, NeuralDevices, a, b, current_body (+75 more)

### Community 22 - "plotting: HeatmapView"
Cohesion: 0.02
Nodes (72): HeatmapView, autoscale, cell_not_computed, cell_regime, colormap, colormap_override, crosshair_x, crosshair_x_color (+64 more)

### Community 23 - "domain: NeuralOutcome"
Cohesion: 0.03
Nodes (77): NeuralOutcome, arcs, build_seconds, context, degree_max, degree_min, hash, hash_mismatch (+69 more)

### Community 24 - "domain: BifurcationDiagramConfig"
Cohesion: 0.03
Nodes (76): BifurcationDiagramConfig, ad_axis, ad_axis_2, adaptive, cd_fp_iters, colored_1d, colored_1d_b_text, colored_1d_built_from (+68 more)

### Community 25 - "domain: map"
Cohesion: 0.07
Nodes (23): AdaptiveAxisKind, kAdAxisAtol, kAdAxisCtrl, kAdAxisRtol, kAdAxisSystem, kAdAxisTol, NetworkNodeStatus, NET_ST_DIVERGED (+15 more)

### Community 26 - "domain: neural_classifier_test.cpp"
Cohesion: 0.06
Nodes (64): NeuralNorm, None, Projection, Total, NeuralProjection, degree_text, delay_text, device (+56 more)

### Community 27 - "desktop: ui_test_script.cpp"
Cohesion: 0.05
Nodes (43): app_library(), app_model(), AppModel, bind_app(), capture_gl(), configure_window(), exe_dir_of(), file_dialog() (+35 more)

### Community 28 - "domain: OrderConfig"
Cohesion: 0.03
Nodes (71): OrderConfig, adaptive, axis_x_hi_text, axis_x_lo_text, axis_x_log, axis_x_n_text, axis_x_target, axis_y_hi_text (+63 more)

### Community 29 - "plotting: plot_axis.cpp"
Cohesion: 0.06
Nodes (66): ImFont, MathExtent, ascent, descent, width, append_layout(), decode_utf8(), digit_step_input_callback() (+58 more)

### Community 30 - "cuda: ParametricEngine::Impl"
Cohesion: 0.03
Nodes (13): ParametricEngine::Impl::run_dft_1d(), OrderDevBuf, p, ParametricEngine::Impl, kAdNodesModulePrefix, kAdProgressUnits, ParametricEngine::Impl::prewarm_lle1d(), ParametricEngine::Impl::prewarm_ls1d() (+5 more)

### Community 31 - "cuda: engine_runtime.cpp"
Cohesion: 0.11
Nodes (52): nvrtc_compile_ptx(), env_value(), compile_into(), ParametricEngine::Impl::compile_basins_if_needed(), ParametricEngine::Impl::compile_fs_module(), ParametricEngine::Impl::compile_order_module(), ParametricEngine::Impl::compile_neural_module(), ParametricEngine::Impl::compile_network_module() (+44 more)

### Community 32 - "domain: CustomTabSharedConfig"
Cohesion: 0.03
Nodes (70): CustomTabSharedConfig, adaptive, auto_recompute_1d, autorun_on_drilldown, axis_x_hi_text, axis_x_lo_text, axis_x_log, axis_x_over_h (+62 more)

### Community 33 - "storage: data_export.cpp"
Cohesion: 0.08
Nodes (52): curve1d_x(), error_estim_name(), export_basins(), export_bif1d(), export_bif2d(), export_dft1d(), export_errt(), export_fastsync() (+44 more)

### Community 34 - "equations: codegen_scheme"
Cohesion: 0.12
Nodes (63): adaptive_outputs_tail(), cd_implicit_sweep_rhs(), codegen_adaptive_gbs(), codegen_rhs(), codegen_spike_system(), dae_scheme_body(), extr_ad_finish(), outputs_tail() (+55 more)

### Community 35 - "domain: adaptive_settings.h"
Cohesion: 0.05
Nodes (61): adaptive_builtin_ctrls(), adaptive_ctrl_defaults(), adaptive_ctrl_effective(), adaptive_ctrl_from_preset(), adaptive_ctrl_pack(), adaptive_ctrl_presets(), adaptive_ctrl_unpack(), adaptive_find_builtin() (+53 more)

### Community 36 - "domain: BifurcationAnalysisState"
Cohesion: 0.03
Nodes (60): BifurcationAnalysisState, active_diagram_index, custom_schemes, diagrams, enabled_builtin_schemes, is_2d_run, last_run_label, last_run_seconds (+52 more)

### Community 37 - "?: app_model.h"
Cohesion: 0.06
Nodes (22): ParametricEngine, ParametricEngine, SystemLibrary, UcudaAdaptParams, UcudaCtlIn, UcudaCtlMem, UcudaCtlOut, exe_dir() (+14 more)

### Community 38 - "cuda: KrsCpuStep"
Cohesion: 0.07
Nodes (40): compute_phase_portrait_unchecked(), find_peaks_host(), find_peaks_host_nodes(), cpu_build_window(), cpu_curand_init(), cpu_curand_uniform(), cpu_dft_block(), cpu_gram_schmidt() (+32 more)

### Community 39 - "domain: OrderPlotWindow"
Cohesion: 0.03
Nodes (61): Dft1DPlotWindow, id, label, label_is_manual, members, Kind, Bifurcation, Error (+53 more)

### Community 40 - "cuda: PeakStream"
Cohesion: 0.04
Nodes (52): PeakStream, anchorTime, capacity, d21, done, emitAll, emitted, endN (+44 more)

### Community 41 - "resources: PeakStreamNU"
Cohesion: 0.06
Nodes (46): ucudaProgressTopUp(), ucuda_shared_stride(), calculateDiscreteModelAvgPeaksAdCUDA(), calculateDiscreteModelPeaksAdContCUDA(), calculateDiscreteModelPeaksAdCUDA(), PeakStreamNU, anchorTime, capacity (+38 more)

### Community 42 - "numerics: krs_cpu.cpp"
Cohesion: 0.08
Nodes (46): f, CpuScalarTraits<numb>, prec, CpuScalarTraits<ucuda::dd>, prec, CpuScalarTraits<ucuda::qd>, prec, KrsCpuDiag (+38 more)

### Community 43 - "domain: LSCurveConfig"
Cohesion: 0.03
Nodes (56): LSCurveConfig, ad_axis, ad_axis_2, adaptive, cd_fp_iters, colormap_idx, continuation, continuation_reverse (+48 more)

### Community 44 - "equations: PN"
Cohesion: 0.13
Nodes (49): codegen_adaptive(), pn_total_dt(), sys_augmented_b(), PN, PrepList, cd_check_complex_safe(), cd_find_complex_unsafe(), cd_fp_iters_of() (+41 more)

### Community 45 - "domain: BasinsConfig"
Cohesion: 0.04
Nodes (55): BasinsConfig, active_plot_tab, adaptive, axis_x_hi_text, axis_x_lo_text, axis_x_symlog, axis_x_var, axis_y_hi_text (+47 more)

### Community 46 - "desktop: hud_theme.cpp"
Cohesion: 0.06
Nodes (42): hud_variant(), apply_style(), backdrop_extremes(), base_col(), cell_phase(), Core, c, s (+34 more)

### Community 47 - "plotting: Plot2DView::render"
Cohesion: 0.10
Nodes (46): ImDrawList, ColorbarTick, frac, label, axis_effective(), axis_px(), dim_grid_col(), fill_col_px() (+38 more)

### Community 48 - "domain: FastSyncConfig"
Cohesion: 0.04
Nodes (53): FastSyncConfig, alpha, autoscale_color, axis_x_hi_text, axis_x_lo_text, axis_x_var, axis_y_hi_text, axis_y_lo_text (+45 more)

### Community 49 - "domain: Dft1DConfig"
Cohesion: 0.04
Nodes (52): Dft1DConfig, cd_fp_iters, colormap_idx, continuation, continuation_reverse, csv_output_path, csv_save_enabled, data_generation (+44 more)

### Community 50 - "domain: LLECurveConfig"
Cohesion: 0.04
Nodes (51): LLECurveConfig, ad_axis, ad_axis_2, adaptive, cd_fp_iters, colormap_idx, continuation, continuation_reverse (+43 more)

### Community 51 - "domain: PhaseAnalysisState"
Cohesion: 0.04
Nodes (52): PhaseAnalysisState, adaptive, auto_recompute, cd_fp_iters, circuit_gbw_mhz, circuit_ideal_opamp, circuit_netlist_poly, circuit_show (+44 more)

### Community 52 - "ui: draw_gui"
Cohesion: 0.05
Nodes (39): default_enabled_slancm(), rebuild_picker_ids(), set_enabled_slancm(), set_field_labels(), set_field_latex(), set_field_library(), busy_from(), BusyInfo (+31 more)

### Community 53 - "numerics: neural_readout.h"
Cohesion: 0.08
Nodes (36): neural_task_evaluate(), check(), check_cancel(), DeviceBuffer, ptr_, neural_ridge_gpu(), cholesky(), cholesky_solve() (+28 more)

### Community 54 - "domain: CustomSession"
Cohesion: 0.04
Nodes (36): CustomSession, basins_session, bif_session, lle_session, ls_session, phase_session, CustomState, custom_schemes (+28 more)

### Community 55 - "domain: SignalMetricsConfig"
Cohesion: 0.04
Nodes (51): SignalMetricsConfig, ad_axis, ad_axis_2, adaptive, cd_fp_iters, colormap_idx, continuation, continuation_reverse (+43 more)

### Community 56 - "numerics: SystemEvaluator"
Cohesion: 0.09
Nodes (36): SystemEvaluator, impl_, complex_cd_pass(), complex_cd_pass_any(), complex_cd_pass_implicit_first(), computePhasePortraitCPU(), int_scheme_complex_state(), lu_factor() (+28 more)

### Community 57 - "analysis: TreeSession"
Cohesion: 0.07
Nodes (26): TreeSession, custom_schemes, dirty, enabled_builtin_schemes, leaves, loaded_system_name, params, record (+18 more)

### Community 58 - "domain: NeuralPlan"
Cohesion: 0.04
Nodes (50): NeuralLink, fraction_text, gain_spread_text, gain_text, population, source, NeuralPlan, arcs (+42 more)

### Community 59 - "numerics: string"
Cohesion: 0.11
Nodes (27): Card, line, text, tok, Ctx, depth, group, pins (+19 more)

### Community 60 - "storage: AppConfig"
Cohesion: 0.06
Nodes (45): AppConfig, basins_avgint_colormap, basins_avgpk_colormap, basins_colormap, basins_states_colormap, camo_theme, contrast_theme, dark_theme (+37 more)

### Community 61 - "analysis: tree_session.cpp"
Cohesion: 0.07
Nodes (33): put_sweep(), region_axis(), sa(), sa_adaptive(), sa_axis(), sa_map(), sab(), skip_axis() (+25 more)

### Community 62 - "domain: NeuralTask"
Cohesion: 0.04
Nodes (47): NeuralDataset, channels, samples, values, NeuralDatasetSplit, Test, Train, Validation (+39 more)

### Community 63 - "tests: artifacts.h"
Cohesion: 0.10
Nodes (25): get_gpu_block_size(), config(), sample(), TEST(), baseline_manifest(), compare_case(), Json, median() (+17 more)

### Community 64 - "domain: NeuralTaskResult"
Cohesion: 0.04
Nodes (48): NeuralBatchClassification, extraction_ms, graph_hash, peak_device_bytes, task, NeuralTaskResult, applied, ar_error (+40 more)

### Community 65 - "domain: TreeNodeKind"
Cohesion: 0.06
Nodes (35): AnalysisTree, next_id, nodes, selected_id, TreeNodeKind, Basins, Bif1D, Bif2D (+27 more)

### Community 66 - "numerics: Builder"
Cohesion: 0.04
Nodes (48): Builder, any_ic, branch, couplings, cur_elem, cur_group, dc_edges, gs (+40 more)

### Community 67 - "ui: widgets.cpp"
Cohesion: 0.10
Nodes (40): auto_label_sweep(), auto_axis_name(), auto_label_dft1d(), auto_label_window(), cfg_field_popup(), combo_var_label(), combo_var_value(), current_width_block() (+32 more)

### Community 68 - "resources: NeuralSpikeArgs"
Cohesion: 0.04
Nodes (49): NeuralSpikeArgs, badFlag, decay, decayHalf, h, inChannel, inChannels, inGain (+41 more)

### Community 69 - "plotting: plot_renderer.cpp"
Cohesion: 0.06
Nodes (29): cmap_lut_sample(), compile(), link_program(), link_program_3(), marker_coverage(), marker_sdf(), oit_out(), operator+() (+21 more)

### Community 70 - "domain: Projection"
Cohesion: 0.04
Nodes (46): next_projection_id(), Projection, alpha, axis_x, axis_y, axis_z, custom_line_style, draw_points (+38 more)

### Community 71 - "domain: configCUDA.h"
Cohesion: 0.16
Nodes (44): calculateDiscreteModelCUDA_H(), getValueByIdxLog(), calculateDiscreteModel_rand(), DFT_custom(), DFT_custom_pf(), rqaDistKernel(), abs(), acos() (+36 more)

### Community 72 - "numerics: IntScheme"
Cohesion: 0.05
Nodes (42): UcudaCtrlCustomFn, UcudaCtrlPrepFn, IntScheme, CD, CD10, CDFast, CDFastFP, CDFP (+34 more)

### Community 73 - "domain: NeuralResult"
Cohesion: 0.05
Nodes (43): NeuralResult, block_size, cancelled, checkpoint, device_bytes, device_clips, device_trace, diverged (+35 more)

### Community 74 - "desktop: main"
Cohesion: 0.06
Nodes (35): apply_camo_palette(), BackdropExtremes, c, valid, enforce_text_contrast(), exe_dir(), inject_window_backdrops(), load_ppm_texture() (+27 more)

### Community 75 - "analysis: session_helpers.cpp"
Cohesion: 0.12
Nodes (29): LyapunovSpectrumAnalysisSession::run(), LyapunovSpectrumAnalysisSession::run_async(), LyapunovSpectrumAnalysisSession::prewarm_task(), adaptive_code_for_scheme(), append_circuit_trajectory(), apply_ls1d_result(), apply_ls2d_result(), build_ls1d_request() (+21 more)

### Community 76 - "analysis: custom_session.cpp"
Cohesion: 0.15
Nodes (34): apply_shared_to_basins(), apply_shared_to_bif1d(), apply_shared_to_bif2d(), apply_shared_to_lle1d(), apply_shared_to_lle2d(), apply_shared_to_ls1d(), apply_shared_to_ls2d(), apply_shared_to_phase() (+26 more)

### Community 77 - "domain: NeuralSource"
Cohesion: 0.05
Nodes (42): neural_source_name(), SourceSeries, a, h, half_steps, index, offset, phase (+34 more)

### Community 78 - "analysis: order_session.cpp"
Cohesion: 0.11
Nodes (33): apply_errt_result(), apply_order_result(), apply_perf_result(), apply_stability_result(), as_d(), build_errt_request(), build_order_request(), build_perf_request() (+25 more)

### Community 79 - "analysis: PhaseRunInputs"
Cohesion: 0.05
Nodes (36): PhaseRunInputs, adaptive, cancel, cd_fp_iters, circuit_gbw_mhz, circuit_ideal_opamp, circuit_show, circuit_substeps (+28 more)

### Community 80 - "application: app_queues.cpp"
Cohesion: 0.05
Nodes (4): AppModel::any_gpu_job_in_flight(), AppModel::run_neural(), AppModel::start_next_in_parametric_queue(), basins_portraits_in_flight()

### Community 81 - "application: AppSettings"
Cohesion: 0.05
Nodes (38): AppSettings, basins_avgint_colormap, basins_avgpk_colormap, basins_colormap, basins_states_colormap, camo_theme, colormap_filter, contrast_theme (+30 more)

### Community 82 - "domain: PerfResult"
Cohesion: 0.05
Nodes (41): PerfResult, ad_desc, adaptive, axis, axis_vals, cancelled, e1, e1_max (+33 more)

### Community 83 - "?: platform/platform_paths.h"
Cohesion: 0.08
Nodes (17): circuit_source_is_kicad(), circuit_source_load(), circuit_source_mtime(), env(), kicad_find_cli(), lower_ext(), narrow(), newest_under() (+9 more)

### Community 84 - "cuda: EngineSources"
Cohesion: 0.05
Nodes (39): EngineSources, src_adaptive_part, src_configCUDA_h, src_configCUDA_h_raw, src_cudaLibrary_cu, src_cudaLibrary_cuh, src_cudaMacros_cuh, src_lyap_adaptive_part (+31 more)

### Community 85 - "tests: observations.cpp"
Cohesion: 0.09
Nodes (31): AnalysisResult, BasinsResult, Bifurcation1DResult, Bifurcation2DResult, Dft1DResult, ErrTimeResult, FastSyncResult, LLE1DResult (+23 more)

### Community 86 - "analysis: basins_session.cpp"
Cohesion: 0.09
Nodes (25): apply_basins_config_to_phase(), apply_basins_result(), basins_display_ids(), basins_fmt_num(), basins_grid_value(), BasinsAnalysisSession::add_config(), BasinsAnalysisSession::phase_projections(), BasinsAnalysisSession::poll() (+17 more)

### Community 87 - "domain: PerfRequest"
Cohesion: 0.06
Nodes (36): run_performance_adaptive(), run_performance_any(), run_performance_cpu(), ParametricEngine::run_performance(), PerfRequest, ad_ctrl_body, ad_desc, ad_emb (+28 more)

### Community 88 - "storage: AppModel::tick"
Cohesion: 0.09
Nodes (21): apply_basins_slots(), session_from_json_basins(), session_from_json_custom(), session_from_json_tree(), session_to_json_basins(), session_to_json_custom(), session_to_json_tree(), AppModel::tick() (+13 more)

### Community 89 - "domain: CircuitGraph"
Cohesion: 0.06
Nodes (37): CircuitSolverConfig, newton_fixed, newton_max_iters, newton_max_step, newton_tol, opamp, opamp_cpole, pivot_min (+29 more)

### Community 90 - "resources: NeuralStepArgs"
Cohesion: 0.05
Nodes (39): NeuralStepArgs, badFlag, d0, d1, d2, d3, devA, devB (+31 more)

### Community 91 - "analysis: PhaseAnalysisSession"
Cohesion: 0.07
Nodes (15): PhaseAnalysisSession, cancel_token, compute_start_time, continuation_last_frame, in_flight, prewarm_future, recompute_future, compute_phase_portrait() (+7 more)

### Community 92 - "domain: FastSyncRequest"
Cohesion: 0.05
Nodes (37): ParametricEngine::run_fastsync(), FastSyncRequest, amountOfX, axis_x_hi, axis_x_lo, axis_x_var, axis_y_hi, axis_y_lo (+29 more)

### Community 93 - "storage: session_json.cpp"
Cohesion: 0.18
Nodes (32): esc(), import_session_ctrl(), jmap(), jstr(), load_ctrl_library(), read_basins_phase_projections(), read_dft1d_field(), read_fastsync_field() (+24 more)

### Community 94 - "ui: combo_items_need"
Cohesion: 0.10
Nodes (22): NetworkSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future, c_str_list() (+14 more)

### Community 95 - "application: TrajTimeEstimator"
Cohesion: 0.07
Nodes (26): Result, detail, rough, seconds, state, State, Pending, Ready (+18 more)

### Community 96 - "cuda: parametric_engine.cpp"
Cohesion: 0.10
Nodes (29): exe_dir(), peak_config_defines(), calib_source(), estimate_device(), gpu_calibrate(), gpu_calibration_path(), gpu_device_info(), krs_step_cost() (+21 more)

### Community 97 - "tests: dae_test.cpp"
Cohesion: 0.17
Nodes (30): adaptive_scheme_name_ok(), codegen_dae_scheme(), ad_params(), adaptive_cpu(), adaptive_doubling_cpu(), adaptive_gpu(), bdf_fixed_cpu(), bdf_fixed_sweep_request() (+22 more)

### Community 98 - "equations: Scheme"
Cohesion: 0.05
Nodes (37): Scheme, CD, CD10, CDFast, CDFastFP, CDFP, ComplexCD, ComplexCD10 (+29 more)

### Community 99 - "tests: spice_test.cpp"
Cohesion: 0.16
Nodes (29): dae_dc_operating_point(), spice_import(), SpiceError, b_conditionals(), chua(), clamp01(), device_simplifications(), devices() (+21 more)

### Community 100 - "plotting: PlotSeriesInput"
Cohesion: 0.06
Nodes (35): HeatmapColormap, LegacyGistNcar, LegacyGistRainbow, LegacyGistStern, LegacyGnuPlot, LegacyGray, LegacyInferno, LegacyNipySpectral (+27 more)

### Community 101 - "plotting: Plot3DView"
Cohesion: 0.05
Nodes (31): Plot3DView, aspect_, auto_fit, axis_bbox_, axis_cache_, axis_scale, camera, custom_line_style (+23 more)

### Community 102 - "desktop: Sweep"
Cohesion: 0.06
Nodes (33): Sweep, action_failed, actions, checked, conflicts, ctx, edit, edit_changed (+25 more)

### Community 103 - "analysis: BasinsAnalysisSession"
Cohesion: 0.06
Nodes (17): BasinsAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, phase_slots, progress_phase_token, progress_token (+9 more)

### Community 104 - "domain: Bifurcation2DRequest"
Cohesion: 0.06
Nodes (35): ParametricEngine::run_bifurcation_2d(), Bifurcation2DRequest, adaptive, amountOfX, base_values, cancel, csv_output_path, emit_all_samples (+27 more)

### Community 105 - "domain: SignalMetricsRequest"
Cohesion: 0.06
Nodes (36): SignalMetricsRequest, adaptive, amountOfX, base_values, cancel, continuation, continuation_reverse, csv_output_path (+28 more)

### Community 106 - "analysis: neural_results_test.cpp"
Cohesion: 0.09
Nodes (31): NeuralBinnedRates, bin_width, points, t0, t_end, values, NeuralResultView, Activity (+23 more)

### Community 107 - "analysis: OrderAnalysisSession"
Cohesion: 0.07
Nodes (16): OrderAnalysisSession, cancel_token, compute_start_time, errt_future, in_flight, last_run_completed_at, perf_future, progress_token (+8 more)

### Community 108 - "domain: AnalysisResult"
Cohesion: 0.06
Nodes (34): AnalysisResult, adaptive, cancelled, circuit_graph, circuit_h_ode, circuit_opamp, circuit_scale, circuit_status (+26 more)

### Community 109 - "?: TEST"
Cohesion: 0.07
Nodes (7): CustomStepFn, computePhasePortraitCPU_custom(), compile(), explicit_time_trajectory(), fixed_case(), parameters(), TEST()

### Community 110 - "analysis: SignalMetricsAnalysisSession"
Cohesion: 0.06
Nodes (14): FastSyncAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future, SignalMetricsAnalysisSession (+6 more)

### Community 111 - "cuda: PhaseAdaptiveRequest"
Cohesion: 0.06
Nodes (31): AdaptiveEndpointRequest, amountOfX, cancel, ctrl_body, emb, ic, params, repeats (+23 more)

### Community 112 - "domain: NetworkConfig"
Cohesion: 0.06
Nodes (34): NetworkConfig, cd_fp_iters, coupling_text, data_generation, edges, gen_directed, gen_h_text, gen_k_text (+26 more)

### Community 113 - "domain: Config"
Cohesion: 0.06
Nodes (33): Config, eps, eps_frac, eps_mode, l_min, m, network_measures, norm (+25 more)

### Community 114 - "tests: compare"
Cohesion: 0.08
Nodes (22): HMODULE, compare(), interp(), main(), Ngspice, ngspice_compare(), h, log (+14 more)

### Community 115 - "domain: StabilityRequest"
Cohesion: 0.07
Nodes (28): run_stability_cpu_t(), ParametricEngine::run_network(), ParametricEngine::run_neural(), ParametricEngine::run_signal_metrics(), stab_norm2(), stability_fill_axes(), stability_step_error(), stability_summarize() (+20 more)

### Community 116 - "cuda: AdaptiveStats"
Cohesion: 0.08
Nodes (29): PhaseAnalysisSession::prewarm_gpu(), AdaptiveEndpointResult, stats, t_avg, t_max, t_min, y_end, PhaseAdaptiveResult (+21 more)

### Community 117 - "cuda: NvrtcEngine"
Cohesion: 0.06
Nodes (24): CacheEntry, kernel, key, module, NvrtcEngine, amountOfX_, cache_, cancel_dev_ (+16 more)

### Community 118 - "domain: LLE2DRequest"
Cohesion: 0.06
Nodes (32): ParametricEngine::run_lle_2d(), LLE2DRequest, adaptive, amountOfX, base_values, cancel, csv_output_path, eps (+24 more)

### Community 119 - "numerics: extract_quadratic"
Cohesion: 0.12
Nodes (26): PolyRhs, terms, PolyTerm, coeff, v1, v2, ScalePlan, s (+18 more)

### Community 120 - "domain: Metrics"
Cohesion: 0.06
Nodes (31): Metrics, clustering, DET, DIV, ENTR, L, L_max, LAM (+23 more)

### Community 121 - "tests: adaptive_test.cpp"
Cohesion: 0.17
Nodes (24): error_, approx(), compile(), cpu_analysis_tests(), cpu_grid(), dense_cpu_tests(), engine_gpu_tests(), exe_dir() (+16 more)

### Community 122 - "cuda: RunSignals"
Cohesion: 0.07
Nodes (19): AdDeviceArgs, axis, params, stats, EngineContext, cc_major, cc_minor, context (+11 more)

### Community 123 - "domain: BasinsRequest"
Cohesion: 0.06
Nodes (31): ParametricEngine::run_basins(), BasinsRequest, adaptive, amountOfX, axis_x_hi, axis_x_lo, axis_x_symlog, axis_x_var (+23 more)

### Community 124 - "domain: Dft1DRequest"
Cohesion: 0.06
Nodes (31): ParametricEngine::run_dft_1d(), Dft1DRequest, amountOfX, base_values, cancel, continuation, continuation_reverse, csv_output_path (+23 more)

### Community 125 - "equations: Parser"
Cohesion: 0.14
Nodes (15): ExtrAdWeights, alpha, diff, Parser, extra_funcs, i, latex, s (+7 more)

### Community 126 - "analysis: build_bif1d_request"
Cohesion: 0.12
Nodes (18): apply_bif1d_result(), apply_bif2d_result(), BifurcationAnalysisSession::poll(), BifurcationAnalysisSession::run(), BifurcationAnalysisSession::run_async(), build_bif1d_request(), build_bif2d_request(), apply_lle1d_result() (+10 more)

### Community 127 - "equations: sysparse.cpp"
Cohesion: 0.13
Nodes (24): SystemEditor::build_system(), DetectedAlphabet, params, vars, FuncDef, args, body, FuncDefs (+16 more)

### Community 128 - "domain: LS2DRequest"
Cohesion: 0.06
Nodes (31): LS2DRequest, adaptive, amountOfX, base_values, cancel, csv_output_path, eps, h (+23 more)

### Community 129 - "domain: MnaLayout"
Cohesion: 0.08
Nodes (27): Branch, elem, kind, out, pole, BranchKind, Multiplier, OpAmpIdeal (+19 more)

### Community 130 - "domain: FastSyncSnapshot"
Cohesion: 0.06
Nodes (31): FastSyncSnapshot, axis_x_hi, axis_x_lo, axis_x_var, axis_y_hi, axis_y_lo, axis_y_var, error_estim (+23 more)

### Community 131 - "tests: neural_cuba_test.cpp"
Cohesion: 0.14
Nodes (28): analytic(), base_case(), Case, edges, name, request, steps, cases() (+20 more)

### Community 132 - "cuda: CircuitGpuRunner"
Cohesion: 0.08
Nodes (23): CircuitGpuRunner, comp_, device_, emit_, func_, lay_, lmem_, module_ (+15 more)

### Community 133 - "domain: StabilityResult"
Cohesion: 0.07
Nodes (29): stability_cell_preferred(), stability_count_preferred(), StabilityResult, cancelled, err, err_max, err_min, error (+21 more)

### Community 134 - "cuda: nvrtc_engine.cpp"
Cohesion: 0.15
Nodes (16): cu_ok(), from_numb(), nvrtc_check_ctrl_body(), nvrtc_ok(), NvrtcEngine::cancel_flag_reset(), NvrtcEngine::compile(), NvrtcEngine::compile_adaptive(), NvrtcEngine::prewarm_adaptive() (+8 more)

### Community 135 - "resources: isinf"
Cohesion: 0.15
Nodes (26): fsBenettinCycleExponent(), loopCalculateDiscreteModelForFastSynchro(), calculateDiscreteModelforFastSynchro(), calculateDiscreteModelICCforFastSynchro(), fsDivergedError(), fsRandSymmetric(), fsSplitMix64(), fsStateDiverged() (+18 more)

### Community 136 - "domain: Bifurcation1DRequest"
Cohesion: 0.07
Nodes (28): ParametricEngine::run_bifurcation_1d(), Bifurcation1DRequest, adaptive, amountOfX, base_values, cancel, continuation, continuation_reverse (+20 more)

### Community 137 - "domain: LLE1DRequest"
Cohesion: 0.07
Nodes (28): ParametricEngine::run_lle_1d(), LLE1DRequest, adaptive, amountOfX, base_values, cancel, continuation, continuation_reverse (+20 more)

### Community 138 - "numerics: CircuitSolver"
Cohesion: 0.08
Nodes (17): CircuitSolver, cfg_, delta_, jac_, lay_, n_v_, pivot_, q_prev_ (+9 more)

### Community 139 - "storage: OrderSnapshot"
Cohesion: 0.07
Nodes (27): OrderSnapshot, adaptive_desc, axis_x_name, axis_y_name, endpoint_only, gpu_fmad, gpu_rdc, h (+19 more)

### Community 140 - "ui: draw_config_tab_bar"
Cohesion: 0.12
Nodes (18): draw_bifurcation_controls(), TabBarResult, active, to_remove, draw_lle_controls(), draw_ls_controls(), draw_metrics_controls(), draw_parametric_controls() (+10 more)

### Community 141 - "ui: CfgFieldScope"
Cohesion: 0.08
Nodes (23): cfg_field_menu_items(), pop_cfg_scope(), push_cfg_scope(), CfgFieldScope, active, at, base, count (+15 more)

### Community 142 - "desktop: app_main.cpp"
Cohesion: 0.15
Nodes (20): com_file_dialog(), dir_exists(), file_exists(), LibraryPaths, library, tmpl, MathFontPaths, italic (+12 more)

### Community 143 - "equations: codegen.cpp"
Cohesion: 0.16
Nodes (20): dae_consistent_hidden(), dae_deriv_array(), dense_solve(), pn_print(), codegen_adaptive_dae(), codegen_adaptive_dae_doubling(), codegen_bdf_bodies(), codegen_rhs_body() (+12 more)

### Community 144 - "domain: AdaptiveRequest"
Cohesion: 0.08
Nodes (22): ParametricEngine::Impl::run_basins_recluster(), ParametricEngine::Impl::basins_dbscan_grid(), ParametricEngine::Impl::basins_dbscan_host(), ParametricEngine::Impl::prewarm_ad_cpu(), ParametricEngine::Impl::ad_nodes_module(), AdaptiveRequest, axis_kind, ctrl_body (+14 more)

### Community 145 - "domain: LS1DRequest"
Cohesion: 0.07
Nodes (28): ParametricEngine::prewarm(), LS1DRequest, adaptive, amountOfX, base_values, cancel, continuation, continuation_reverse (+20 more)

### Community 146 - "domain: AdaptiveSettings"
Cohesion: 0.07
Nodes (27): AdaptiveSettings, atol, ctrl, ctrl_params, ctrl_replaced, enabled, h0, hmax (+19 more)

### Community 147 - "analysis: TrajWork"
Cohesion: 0.15
Nodes (21): BasinsAnalysisSession::traj_work(), BifurcationAnalysisSession::traj_work(), build_metrics_request(), Dft1DAnalysisSession::traj_work(), LLEAnalysisSession::traj_work(), lyap_steps(), LyapunovSpectrumAnalysisSession::traj_work(), make_traj_work() (+13 more)

### Community 148 - "equations: parse_dae"
Cohesion: 0.12
Nodes (20): SystemEditor::dae_operating_point(), SystemEditor::ic_fields(), SystemEditor::import_circuit(), dae_parse_index_text(), DaeSystem, alg, diff, f (+12 more)

### Community 149 - "cuda: BdfGpuRequest"
Cohesion: 0.07
Nodes (27): BdfGpuRequest, amountOfX, atol, cancel, dt, h0, hfix, hmax (+19 more)

### Community 150 - "domain: BasinsAnalysisState"
Cohesion: 0.09
Nodes (22): BasinsAnalysisState, active_config_index, configs, custom_schemes, enabled_builtin_schemes, last_run_label, last_run_seconds, last_run_succeeded (+14 more)

### Community 151 - "domain: BasinsResult"
Cohesion: 0.07
Nodes (27): BasinsResult, ad_stats, avg_intervals, avg_intervals_max, avg_intervals_min, avg_peaks, avg_peaks_max, avg_peaks_min (+19 more)

### Community 152 - "domain: NeuralSpiking"
Cohesion: 0.07
Nodes (27): NeuralSpiking, edge_delay, edge_rule, log_capacity, receptors, refractory_steps, rules, seed (+19 more)

### Community 153 - "plotting: Plot3DView::render"
Cohesion: 0.11
Nodes (20): LegendEntry, color, label, LegendPass, Both, Draw, Interact, LegendRightClick (+12 more)

### Community 154 - "?: TEST_F"
Cohesion: 0.09
Nodes (5): apply_nvrtc_rdc(), set_nvrtc_rdc(), DaeGpu, TEST(), TEST_F()

### Community 155 - "domain: OrderResult"
Cohesion: 0.08
Nodes (26): OrderResult, axis_x, axis_x_vals, axis_y, axis_y_vals, cancelled, e1, e1_max (+18 more)

### Community 156 - "domain: NeuralReadoutSaved"
Cohesion: 0.08
Nodes (26): NeuralReadoutSaved, class_bins, class_input_norm, class_protocol, classes, feature_var, features_requested, filter (+18 more)

### Community 157 - "domain: param_groups.h"
Cohesion: 0.24
Nodes (22): FieldLabels, fl_from_text(), fl_to_text(), ParamGroup, params, path, ParamGroups, pg_find() (+14 more)

### Community 158 - "ui: draw_composition_builder"
Cohesion: 0.16
Nodes (21): preset, BuiltinScheme, group, name, order, SystemEditor, tooltip, scheme_group_title() (+13 more)

### Community 159 - "ui: ViewRangeValues"
Cohesion: 0.08
Nodes (24): HeatmapToolbarOpts, extras, extras_tail, persist_colormap, show_swap, RunAllGroup, enqueue, label (+16 more)

### Community 160 - "ui: dock_layout.cpp"
Cohesion: 0.19
Nodes (20): assign_chain(), axis_len(), capture(), chain_items(), collect_separators(), count_leaves(), dock_install_window_menu(), dock_layout_request() (+12 more)

### Community 161 - "domain: EliminationPlan"
Cohesion: 0.09
Nodes (23): elimination_report(), EliminationPlan, diagonal_free, filled, max_row_len, mul_ops, n, nnz_factor (+15 more)

### Community 162 - "equations: AdaptiveCode"
Cohesion: 0.08
Nodes (23): AdaptiveCode, dense_order, deval, dprep, dprep_rhs, dsize, emb, emb_rhs (+15 more)

### Community 163 - "equations: FuncId"
Cohesion: 0.08
Nodes (25): FuncId, F_ACOS, F_ASIN, F_ATAN, F_ATAN2, F_CBRT, F_CEIL, F_COPYSIGN (+17 more)

### Community 164 - "storage: system_library.cpp"
Cohesion: 0.20
Nodes (18): esc(), kv(), kvbool(), kvmap(), record_to_json(), sessions_dir(), SystemLibrary::dir_for(), SystemLibrary::exists() (+10 more)

### Community 165 - "resources: MetricsAccumNU"
Cohesion: 0.08
Nodes (22): MetricsAccumNU, bMax, bMin, box, d1, fp, i1, i2 (+14 more)

### Community 166 - "cuda: EngineModuleBuilder"
Cohesion: 0.08
Nodes (19): CachedKrsPtx, key, ptx, CachedLibPtx, key, lowered, ptx, EngineModuleBuilder (+11 more)

### Community 167 - "ocr: ocr_client_win.cpp"
Cohesion: 0.13
Nodes (14): OcrClient, hChildStdinWr_, hChildStdoutRd_, hProcess_, rbuf_, b64encode(), H(), OcrClient::OcrClient() (+6 more)

### Community 168 - "tests: neural_forecast_test.cpp"
Cohesion: 0.18
Nodes (20): neural_schedule_name(), NeuralSchedule, Heun, Old, Post, Spike, Stage, attractor_stats() (+12 more)

### Community 169 - "cuda: engine_impl.h"
Cohesion: 0.13
Nodes (9): cache_activate(), cache_publish(), drain_pool(), reset_cancel_mirror_pool(), EngineModuleCache::reset_cancel_mirrors(), ParametricEngine::Impl::~Impl(), ModuleLru, cap_ (+1 more)

### Community 170 - "ui: LineStyleFields"
Cohesion: 0.10
Nodes (23): ProjType, ContinuationDiagram, FeatureDiagram, Phase2D, Phase3D, RecurrencePlot, Rejections, StepSize (+15 more)

### Community 171 - "domain: ErrTimeResult"
Cohesion: 0.08
Nodes (24): ErrTimeResult, cancelled, diverged, e_norm, e_rich, e_var, error, h_eff (+16 more)

### Community 172 - "domain: num_parse.h"
Cohesion: 0.14
Nodes (13): is_letter(), parse_expr(), parse_factor(), parse_primary(), parse_term(), scale_pow10(), si_prefix(), skip_ws() (+5 more)

### Community 173 - "equations: SystemEvaluator::Impl"
Cohesion: 0.08
Nodes (23): Instr, idx, op, val, SystemEvaluator::Impl, complex_unsafe, diag_coef, diag_coef_free (+15 more)

### Community 174 - "ui: LeafViews"
Cohesion: 0.08
Nodes (20): BifurcationPointCache, generation, key, points, sorted_x, SweepCurveCache, LeafViews, basins_owner (+12 more)

### Community 175 - "tests: library_test.cpp"
Cohesion: 0.17
Nodes (19): any_finite(), count(), main(), num(), record_model(), report(), run_custom(), run_system() (+11 more)

### Community 176 - "desktop: paint_sith_scene"
Cohesion: 0.18
Nodes (16): fract(), hash01(), Rng, s, beam(), draw_ship(), fire(), hot() (+8 more)

### Community 177 - "?: main.cpp"
Cohesion: 0.14
Nodes (9): BatchSystemEditor, exe_dir(), integer(), main(), read_file(), fields(), invalid(), load_neural_dataset_csv() (+1 more)

### Community 178 - "tests: Comparison"
Cohesion: 0.13
Nodes (15): compare(), Comparison, absolute, bit_differences, details, exact, failures, max_absolute (+7 more)

### Community 179 - "tests: architecture_test.cpp"
Cohesion: 0.16
Nodes (16): cancellation(), engine_is_public, engine_is_public<T, std::void_t<decltype(std::declval<T&>().parametric_engine)>>, exe_dir(), generated_method_selection(), headless_method_availability(), main(), make_model() (+8 more)

### Community 180 - "application: NetworkViewState"
Cohesion: 0.09
Nodes (23): NetworkViewState, color_var, custom_line_style, drag_from, first_series_node, graph_colormap, line_alpha, line_width (+15 more)

### Community 181 - "domain: SignalMetricsAnalysisState"
Cohesion: 0.11
Nodes (19): SignalMetricsAnalysisState, active_config_index, configs, custom_schemes, enabled_builtin_schemes, is_2d_run, last_run_label, last_run_seconds (+11 more)

### Community 182 - "domain: BasinsSnapshot"
Cohesion: 0.09
Nodes (23): BasinsSnapshot, axis_x_hi, axis_x_lo, axis_x_symlog, axis_x_var, axis_y_hi, axis_y_lo, axis_y_symlog (+15 more)

### Community 183 - "equations: SparseLu"
Cohesion: 0.11
Nodes (18): NameMap, m, SparseLu, Cb, id, n1, nb, nnz (+10 more)

### Community 184 - "numerics: CpuAdaptiveKrs"
Cohesion: 0.10
Nodes (18): CpuAdaptiveKrs, a_host, AE, B, DD, ev, kind, L (+10 more)

### Community 185 - "tests: test_util.h"
Cohesion: 0.15
Nodes (13): add(), compare(), main(), Run, run_all(), seconds, v, tail() (+5 more)

### Community 186 - "desktop: paint_jedi_scene"
Cohesion: 0.20
Nodes (16): draw_fighter(), fighter_pose(), gun_tip(), hazed(), leader_at(), paint_jedi_scene(), Pose, c (+8 more)

### Community 187 - "application: AppModel::broadcast_field"
Cohesion: 0.15
Nodes (18): AppModel::broadcast_field(), AppModel::push_undo(), AppModel::redo_top(), AppModel::undo_top(), decode_sweep(), encode_sweep(), encode_sweep_target(), has_field() (+10 more)

### Community 189 - "domain: string"
Cohesion: 0.10
Nodes (19): ParametricEngine::run_basins_recluster(), BasinsReclusterRequest, amountOfX, avg_intervals, avg_peaks, cancel, eps_dbscan, helpful_array (+11 more)

### Community 190 - "domain: FastSyncResult"
Cohesion: 0.09
Nodes (22): FastSyncResult, amountOfX_traj, axis_x_hi, axis_x_lo, axis_x_var, axis_y_hi, axis_y_lo, axis_y_var (+14 more)

### Community 191 - "domain: NetworkRequest"
Cohesion: 0.09
Nodes (22): NetworkRequest, amountOfValues, amountOfX, cancel, coupling_body, edge_law, edge_src, edge_start (+14 more)

### Community 192 - "numerics: NetlistOptions"
Cohesion: 0.09
Nodes (20): OpAmpModel, a0, gbw_hz, ideal, slew, vsat, NetlistDialect, Behavioral (+12 more)

### Community 193 - "numerics: .element"
Cohesion: 0.22
Nodes (6): IdealJunction, ron, vf, ideal_junction_fit(), at, sanitize()

### Community 194 - "numerics: SpiceImportResult"
Cohesion: 0.10
Nodes (19): SpiceImportResult, alg_text, field_labels, has_ic, ic_groups, init_conditions, node_vars, notes (+11 more)

### Community 195 - "numerics: Deck"
Cohesion: 0.10
Nodes (18): Deck, base_dir, global, ic_nodes, ignored, include_depth, models, subckts (+10 more)

### Community 196 - "plotting: PlotWindowPrefs"
Cohesion: 0.10
Nodes (20): PlotWindowPrefs, axes3d, colorbar, fit3d, fit_pane, grid, grid_on_top, invert_x (+12 more)

### Community 197 - "tests: main"
Cohesion: 0.18
Nodes (15): append(), append_text(), append_vec(), benchmark(), Clock, cuda_ok(), environment(), equal_bytes() (+7 more)

### Community 198 - "domain: Kind"
Cohesion: 0.12
Nodes (19): CustomSession::enqueue_level_1d(), CustomSession::enqueue_level_1d_partial(), CustomSession::enqueue_level_2d(), CustomSession::enqueue_level_3(), AppModel::enqueue_custom(), CustomQueueItem, kind, Kind (+11 more)

### Community 199 - "analysis: dft_session.cpp"
Cohesion: 0.14
Nodes (10): apply_dft1d_result(), build_dft1d_request(), Dft1DAnalysisSession::poll(), Dft1DAnalysisSession::run(), Dft1DAnalysisSession::run_async(), LyapunovSpectrumAnalysisSession::poll(), apply_metrics_result(), SignalMetricsAnalysisSession::poll() (+2 more)

### Community 200 - "domain: SignalMetricsResult"
Cohesion: 0.10
Nodes (20): SignalMetricsResult, ad_stats, cancelled, continuation, continuation_reverse, dimension, error, flags (+12 more)

### Community 201 - "domain: Bif2DSnapshot"
Cohesion: 0.10
Nodes (20): Bif2DSnapshot, eps_dbscan, gpu_fmad, h, indexOfMutVar, indexOfMutVar2, initial_conditions, mult_interval (+12 more)

### Community 202 - "domain: NetworkState"
Cohesion: 0.10
Nodes (19): NetworkMonolithic, record, sys, NetworkState, active_config_index, configs, custom_schemes, enabled_builtin_schemes (+11 more)

### Community 203 - "storage: JP"
Cohesion: 0.25
Nodes (9): drop_foreign_ctrl_axes(), JP, i, s, read_adaptive(), read_basins_field(), read_diagram_field(), read_lle_curve_field() (+1 more)

### Community 204 - "tests: SolverWorkload"
Cohesion: 0.12
Nodes (15): solver_case_id(), solver_cases(), SolverWorkload, a_, ad_, adaptive_, bdf_, engine_ (+7 more)

### Community 205 - "analysis: BifurcationAnalysisSession"
Cohesion: 0.11
Nodes (8): BifurcationAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future, run_future_2d

### Community 206 - "analysis: LLEAnalysisSession"
Cohesion: 0.11
Nodes (8): LLEAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future, run_future_2d

### Community 207 - "analysis: LyapunovSpectrumAnalysisSession"
Cohesion: 0.11
Nodes (8): LyapunovSpectrumAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future, run_future_2d

### Community 208 - "analysis: net_generate_topology"
Cohesion: 0.19
Nodes (14): build_network_request(), edge_key(), net_build_csr(), net_coupling_body(), net_generate_topology(), net_monolithic_snapshot(), NetworkSession::run_async(), node_value() (+6 more)

### Community 209 - "application: AppMode"
Cohesion: 0.11
Nodes (19): AppMode, Analysis, Basins, Custom, Dft1D, FastSync, Library, Network (+11 more)

### Community 210 - "application: BroadcastSlots"
Cohesion: 0.11
Nodes (19): BroadcastSlots, cd_fp_iters, h, ics, label, max_value, params, pre_scaller (+11 more)

### Community 211 - "domain: OrderRequest"
Cohesion: 0.11
Nodes (18): ParametricEngine::run_order(), OrderRequest, amountOfX, axis_x, axis_y, cancel, cpu_prec, endpoint_only (+10 more)

### Community 212 - "domain: Dft1DAnalysisState"
Cohesion: 0.12
Nodes (16): Dft1DAnalysisState, active_config_index, configs, custom_schemes, enabled_builtin_schemes, last_run_label, last_run_seconds, last_run_succeeded (+8 more)

### Community 213 - "domain: ErrTimeRequest"
Cohesion: 0.11
Nodes (19): ErrTimeRequest, amountOfX, cancel, cpu_prec, h, initial_conditions, krs_body, max_points (+11 more)

### Community 214 - "domain: LS2DSnapshot"
Cohesion: 0.11
Nodes (19): LS2DSnapshot, eps, gpu_fmad, h, indexOfMutVar, indexOfMutVar2, initial_conditions, n_exponents (+11 more)

### Community 215 - "domain: OrderAnalysisState"
Cohesion: 0.11
Nodes (16): OrderAnalysisState, active_config_index, auto_compute, configs, custom_schemes, enabled_builtin_schemes, last_run_label, last_run_seconds (+8 more)

### Community 216 - "numerics: bdf_integrate"
Cohesion: 0.13
Nodes (16): bdf_eval(), bdf_integrate(), bdf_prepare_system(), BdfSettings, atol, h0, hfix, hmax (+8 more)

### Community 217 - "numerics: CondRewriter"
Cohesion: 0.36
Nodes (5): CondRewriter, p, V, boolean, s

### Community 218 - "numerics: AdaptiveCpuModule"
Cohesion: 0.15
Nodes (4): AdaptiveCpuModule, module_, AdCpu, mod

### Community 219 - "application: BroadcastField"
Cohesion: 0.11
Nodes (18): BroadcastField, CdFpIters, InitCondition, MaxValue, Param, PreScaller, Resolution, Scheme (+10 more)

### Community 220 - "application: DaeEditorState"
Cohesion: 0.11
Nodes (18): DaeEditorState, circuit_error, circuit_import_now, circuit_mtime, circuit_path, circuit_poll_t, circuit_report, circuit_watch (+10 more)

### Community 221 - "application: string"
Cohesion: 0.13
Nodes (5): RhsPreview, error, key, text, SystemEditor::from_record()

### Community 222 - "domain: FastSyncAnalysisState"
Cohesion: 0.12
Nodes (16): FastSyncAnalysisState, active_config_index, configs, custom_schemes, enabled_builtin_schemes, last_run_label, last_run_seconds, last_run_succeeded (+8 more)

### Community 223 - "domain: NetworkResult"
Cohesion: 0.11
Nodes (18): NetworkResult, cancelled, data, dt, error, gpu_ms, h, launch_count (+10 more)

### Community 224 - "domain: LLE1DSnapshot"
Cohesion: 0.11
Nodes (18): LLE1DSnapshot, continuation, continuation_reverse, eps, gpu_fmad, h, indexOfMutVar, initial_conditions (+10 more)

### Community 225 - "domain: LLE2DSnapshot"
Cohesion: 0.11
Nodes (18): LLE2DSnapshot, eps, gpu_fmad, h, indexOfMutVar, indexOfMutVar2, initial_conditions, n_pts (+10 more)

### Community 226 - "domain: LS1DSnapshot"
Cohesion: 0.11
Nodes (18): LS1DSnapshot, continuation, continuation_reverse, eps, gpu_fmad, h, indexOfMutVar, initial_conditions (+10 more)

### Community 227 - "equations: DaeScheme"
Cohesion: 0.11
Nodes (18): DaeScheme, BDF1, BDF1GGL, CD, CDIndex2, CDYoshida4, ExplicitEuler, ExplicitMidpoint (+10 more)

### Community 228 - "equations: adaptive_emb_body"
Cohesion: 0.16
Nodes (16): MultiDoubleCoef, c, adaptive_err_weights(), EmbSpec, A, b, lc, ld (+8 more)

### Community 229 - "equations: Node"
Cohesion: 0.11
Nodes (17): Kind, Add, Call, Div, Mul, Neg, Num, Pow (+9 more)

### Community 230 - "numerics: Emitter"
Cohesion: 0.22
Nodes (7): cdbl(), emit_circuit_step(), Emitter, comp, declared, lay, o

### Community 231 - "ui: BasinsPlotViews"
Cohesion: 0.11
Nodes (14): BasinsPlotViews, clusters, has_diverged, heatmap_generation, heatmap_source, heatmap_states, heatmap_values, heatmaps (+6 more)

### Community 232 - "ui: draw_custom_workspace_panel"
Cohesion: 0.19
Nodes (12): build(), dock_keep_split_ratios(), dock_layout_has_request(), dock_layout_request_saved(), dock_snap_dragged_split(), find_active_split(), ImGuiWindow, rebuild() (+4 more)

### Community 233 - "analysis: Dft1DAnalysisSession"
Cohesion: 0.12
Nodes (7): Dft1DAnalysisSession, cancel_token, compute_start_time, in_flight, last_run_completed_at, progress_token, run_future

### Community 234 - "analysis: TreeLeafRuntime"
Cohesion: 0.12
Nodes (10): TreeLeafRuntime, basins, bif, committed, committed_ok, lle, ls, pending (+2 more)

### Community 235 - "application: NeuralViewState"
Cohesion: 0.12
Nodes (15): NeuralViewState, instance, map_kind, map_log, map_pop, rate_bin_text, readout_target, series_var (+7 more)

### Community 236 - "domain: Bif1DSnapshot"
Cohesion: 0.12
Nodes (17): Bif1DSnapshot, continuation, continuation_reverse, gpu_fmad, h, indexOfMutVar, initial_conditions, log_scale (+9 more)

### Community 237 - "domain: Dft1DSnapshot"
Cohesion: 0.12
Nodes (16): Dft1DSnapshot, freq_hi, freq_lo, gpu_fmad, h, indexOfMutVar, initial_conditions, n_freq (+8 more)

### Community 238 - "domain: NeuralGraph"
Cohesion: 0.12
Nodes (17): NeuralGraph, degree_max, degree_min, delay, dev_a, dev_b, dev_gain, dev_two_way (+9 more)

### Community 239 - "plotting: plot_camera_3d.cpp"
Cohesion: 0.12
Nodes (10): PlotShowItems, PlotShowAxes2D, PlotShowAxes3D, PlotShowColorbar, PlotShowGrid, PlotShowLegend, PlotShowSmooth, mat_mul() (+2 more)

### Community 240 - "plotting: plot_legend.cpp"
Cohesion: 0.19
Nodes (13): default_series_color(), hex_colour(), legend_color_popup(), remember_colour(), series_palette_color(), series_palette_name(), series_palette_size(), SeriesPalette (+5 more)

### Community 241 - "resources: UcudaCtlConst"
Cohesion: 0.12
Nodes (17): UcudaCtlConst, b1, b2, b3, expo, expo1, facc1, facc2 (+9 more)

### Community 242 - "application: string #2"
Cohesion: 0.18
Nodes (13): scan, AppModel::enqueue_tree(), AppModel::neural_result_settings_key(), AppModel::poll_parametric_prewarm(), fmt_num_for_input(), pin_fixed_h(), pin_fixed_ic(), pin_fixed_param() (+5 more)

### Community 243 - "application: compute_services.cpp"
Cohesion: 0.16
Nodes (10): BodyIndexDiag, line, message, PeakConfig, PhaseAnalysisSession, SignalMetricsResult, check_body_indices(), check_controller_body() (+2 more)

### Community 244 - "domain: GpuCalibration"
Cohesion: 0.17
Nodes (11): TrajTimeEstimator::start_calibration(), GpuCalibration, c0_lat, c0_thr, calib_ops, device, kernel_factor, ns_lat (+3 more)

### Community 245 - "equations: system_evaluator.cpp"
Cohesion: 0.17
Nodes (9): T, run_program(), SystemEvaluator::eval(), SystemEvaluator::eval_complex(), SystemEvaluator::eval_jacobian(), SystemEvaluator::eval_jacobian_complex(), SystemEvaluator::solve_diag_fp(), SystemEvaluator::solve_diag_implicit() (+1 more)

### Community 246 - "domain: LS2DResult"
Cohesion: 0.12
Nodes (16): LS2DResult, ad_stats, cancelled, error, flags, max_val, min_val, n_exponents (+8 more)

### Community 247 - "domain: SignalMetric"
Cohesion: 0.12
Nodes (16): SignalMetric, SIGM_COUNT, SIGM_HJORTH_COMPLEXITY, SIGM_HJORTH_MOBILITY, SIGM_INT_MAX, SIGM_INT_MEAN, SIGM_INT_MIN, SIGM_INT_RANGE (+8 more)

### Community 248 - "desktop: hud_scene.h"
Cohesion: 0.25
Nodes (8): beam(), glow_at(), inside(), rgba(), Tint, body, core, flame

### Community 249 - "application: .build_system"
Cohesion: 0.32
Nodes (12): AppModel::propagate_to_sessions(), AppModel::start_basins_analysis(), AppModel::start_custom_analysis(), AppModel::start_dft1d_analysis(), AppModel::start_fastsync_analysis(), AppModel::start_network_analysis(), AppModel::start_neural_analysis(), AppModel::start_order_analysis() (+4 more)

### Community 250 - "domain: Bifurcation2DResult"
Cohesion: 0.13
Nodes (15): Bifurcation2DResult, ad_stats, cancelled, error, flags, max_val, min_val, n_pts (+7 more)

### Community 251 - "domain: Dft1DResult"
Cohesion: 0.13
Nodes (15): Dft1DResult, ak_cos, bk_sin, cancelled, continuation_reverse, error, flags, freq_hi (+7 more)

### Community 252 - "domain: LLE2DResult"
Cohesion: 0.13
Nodes (15): LLE2DResult, ad_stats, cancelled, error, flags, max_val, min_val, n_pts (+7 more)

### Community 253 - "domain: CircuitRunResult"
Cohesion: 0.13
Nodes (13): CircuitRunResult, error, ok, stats, t, x, CircuitSolverStats, converged_always (+5 more)

### Community 254 - "domain: ucuda_node_value"
Cohesion: 0.20
Nodes (14): ucuda_node_value(), ucuda_node_value_log(), GridInfo, height, width, x_max, x_min, y_max (+6 more)

### Community 255 - "domain: PhaseSnapshot"
Cohesion: 0.13
Nodes (14): PhaseSnapshot, a, adaptive, decimator, gpu_fmad, h, ic_flat, ic_labels (+6 more)

### Community 256 - "domain: NeuralSweep"
Cohesion: 0.13
Nodes (15): NeuralSweep, enabled, hi_text, kind, lo_text, log, param, points_text (+7 more)

### Community 257 - "equations: dae_devices.cpp"
Cohesion: 0.27
Nodes (12): BjtCurrents, ib, ic, bjt_currents(), dev_explim(), diode_current(), ideal_diode_current(), jfet_drain_current() (+4 more)

### Community 258 - "plotting: GpuLineSeriesSet"
Cohesion: 0.16
Nodes (6): GpuLineSeries, point_count, vbo, GpuLineSeriesSet, bbox_data_, series_

### Community 259 - "ui: ScatterClusterCache"
Cohesion: 0.13
Nodes (13): ScatterClusterCache, bufs, gen, ids, labels, renumber, src_idx, src_x (+5 more)

### Community 260 - "ui: DockWindowFactory"
Cohesion: 0.13
Nodes (12): BuildCtx, created, factory, first_leaf, missing, DockWindowFactory, apply_unmatched, available_windows (+4 more)

### Community 261 - "ui: LNode"
Cohesion: 0.14
Nodes (12): LNode, a, axis, b, objects, prefs, ratio, selected (+4 more)

### Community 262 - "ui: draw_library_list"
Cohesion: 0.24
Nodes (11): lib_tab_of(), describe_record(), draw_library_list(), invalidate_library_cache(), library_editor_save(), persist_hidden_systems(), trim_copy(), LibTab (+3 more)

### Community 263 - "tests: gen_lorenz_reference.py"
Cohesion: 0.24
Nodes (8): convergence_check(), integrate(), lle(), main(), ranges(), rhs(), rk4(), trajectory()

### Community 264 - "domain: NeuralDeviceModel"
Cohesion: 0.15
Nodes (14): neural_device_preset(), neural_device_preset_name(), NeuralDeviceModel, current_text, initial_text, max_text, min_text, params_text (+6 more)

### Community 265 - "cuda: CpuMetricsAccum"
Cohesion: 0.14
Nodes (13): CpuMetricsAccum, dPrev, mn, mx, n, s1, s2, sd (+5 more)

### Community 266 - "domain: BasinFeature"
Cohesion: 0.14
Nodes (14): BasinFeature, AvgIntervals, AvgPeaks, LogAvgIntervals, LogAvgPeaks, LogRMSIntervals, LogRMSPeaks, LogStDevIntervals (+6 more)

### Community 267 - "domain: Bifurcation1DResult"
Cohesion: 0.14
Nodes (14): Bifurcation1DResult, ad_stats, bifurcation_points, cancelled, continuation_reverse, error, flags, n_pts (+6 more)

### Community 268 - "plotting: AxisInfo"
Cohesion: 0.14
Nodes (13): AxisInfo, display_log, edge_tick_tol, invert, lock, log_scale, name, symlog (+5 more)

### Community 269 - "plotting: plot_series_colors_changed"
Cohesion: 0.20
Nodes (9): plot_series_color_spread(), plot_series_colors_changed(), plot_show_menu(), prefs_default(), prefs_item(), prefs_read_line(), prefs_read_open(), prefs_write_all() (+1 more)

### Community 270 - "domain: net_default_law"
Cohesion: 0.18
Nodes (11): NetworkSession::add_config(), net_default_law(), NetCouplingLaw, expr, name, NetEdge, bidirectional, from (+3 more)

### Community 271 - "domain: NewtonSettings"
Cohesion: 0.35
Nodes (11): order_newton_settings(), cd_fp_append_slot(), cd_fp_iters_from_text(), newton_iters_from_text(), newton_tol_from_text(), NewtonSettings, full, max_iters (+3 more)

### Community 272 - "application: CamoTheme"
Cohesion: 0.15
Nodes (13): CamoTheme, kCamoCount, kCamoDdpm, kCamoDpm, kCamoFlecktarn, kCamoFlora, kCamoHud, kCamoHudLight (+5 more)

### Community 273 - "domain: vector"
Cohesion: 0.15
Nodes (12): LLE1DResult, ad_stats, cancelled, continuation_reverse, error, flags, lyapunov, n_pts (+4 more)

### Community 274 - "domain: LS1DResult"
Cohesion: 0.15
Nodes (13): LS1DResult, ad_stats, cancelled, continuation_reverse, error, flags, n_exponents, n_pts (+5 more)

### Community 275 - "equations: OpCode"
Cohesion: 0.15
Nodes (13): OpCode, OP_ADD, OP_DIV, OP_FUNC1, OP_FUNC2, OP_MUL, OP_NEG, OP_POW (+5 more)

### Community 276 - "equations: RosTab"
Cohesion: 0.15
Nodes (13): RosTab, a, alpha, c, d2, d3, gamma, gt (+5 more)

### Community 277 - "numerics: BdfStats"
Cohesion: 0.15
Nodes (13): BdfStats, diverged, hmax, hmin, hsum, max_q, nacc, nforced (+5 more)

### Community 278 - "platform: FileImageSource"
Cohesion: 0.17
Nodes (4): ClipboardImageSource, FileImageSource, path, ImageSource

### Community 279 - "platform: Process"
Cohesion: 0.26
Nodes (5): Process, input_, output_, pid_, run()

### Community 280 - "storage: SystemLibrary"
Cohesion: 0.18
Nodes (3): SystemLibrary, dir_, SystemLibrary::duplicate()

### Community 281 - "desktop: Handle"
Cohesion: 0.20
Nodes (5): width, Handle, clipped, id, rect

### Community 282 - "desktop: string"
Cohesion: 0.20
Nodes (3): MenuActions, reset, restore

### Community 283 - "application: run_next_queued"
Cohesion: 0.18
Nodes (8): AppModel::start_next_in_basins_queue(), AppModel::start_next_in_dft1d_queue(), AppModel::start_next_in_fastsync_queue(), AppModel::start_next_in_network_queue(), AppModel::start_next_in_neural_queue(), AppModel::start_next_in_order_queue(), run_next_queued(), run_next_queued_with()

### Community 284 - "cuda: host_rqa.cu"
Cohesion: 0.26
Nodes (9): rqa::compute(), rqa_entropy(), rqa_masked(), rqaDegKernel(), rqaDiagKernel(), rqaHistKernel(), rqaMaxKernel(), rqaTriKernel() (+1 more)

### Community 285 - "plotting: PlotSeriesInput3D"
Cohesion: 0.17
Nodes (10): PlotSeriesInput3D, cmap_reverse, cmax, cmin, color, colormap, label, n_points (+2 more)

### Community 286 - "ui: string"
Cohesion: 0.21
Nodes (10): Builtin, mode, name, text, find_builtin(), id_of(), label_of(), Matcher (+2 more)

### Community 287 - "ui: Request"
Cohesion: 0.17
Nodes (12): ReqKind, Apply, None, Preset, Remove, Save, Request, delay (+4 more)

### Community 288 - "ui: OrderCurveData"
Cohesion: 0.17
Nodes (10): builtin_scheme_order(), OrderCurveData, bufs, colors, got_x, labels, markers, tags (+2 more)

### Community 289 - "ui: LibraryListCache"
Cohesion: 0.17
Nodes (10): LibraryListCache, custom_counts, filled_at, names, sel_desc, sel_name, sel_ok, sel_rec (+2 more)

### Community 290 - "desktop: CamoPalette"
Cohesion: 0.18
Nodes (11): CamoPalette, accent, accent_dark, frame, light, plot, popup, text (+3 more)

### Community 291 - "tools: mnist_subset.py"
Cohesion: 0.35
Nodes (4): archive(), main(), records(), select()

### Community 292 - "analysis: fastsync_session.cpp"
Cohesion: 0.27
Nodes (5): apply_fastsync_result(), build_fastsync_request(), FastSyncAnalysisSession::poll(), FastSyncAnalysisSession::run(), FastSyncAnalysisSession::run_async()

### Community 293 - "numerics: run_adaptive_phase"
Cohesion: 0.25
Nodes (3): run_adaptive_phase(), CtrlCpuFn, module_

### Community 294 - "application: BroadcastTab"
Cohesion: 0.18
Nodes (11): BroadcastTab, Basins, Bifurcation, Custom, Dft1D, FastSync, LLE, LS (+3 more)

### Community 295 - "application: PrewarmWatch"
Cohesion: 0.24
Nodes (6): PrewarmWatch, changed_at, fired, kRemembered, pending, sigs

### Community 296 - "cuda: CircuitGpuRunner::build"
Cohesion: 0.22
Nodes (8): CircuitGpuRunner::build(), CircuitGpuRunner::run(), cu_err(), read_template(), EmitConfig, newton_iters, newton_max_step, pivot_min

### Community 297 - "cuda: MetricsAxes"
Cohesion: 0.18
Nodes (11): MetricsAxes, hi_x, hi_y, hSweepAxis, idx_x, idx_y, lo_x, lo_y (+3 more)

### Community 298 - "domain: OrderAxis"
Cohesion: 0.18
Nodes (11): OrderAxis, hi, index, kind, lo, log_scale, n_pts, OrderAxisKind (+3 more)

### Community 300 - "plotting: PlotCamera3D"
Cohesion: 0.18
Nodes (6): PlotCamera3D, aspect, distance, phi, target, theta

### Community 301 - "ui: gui_tree_state.h"
Cohesion: 0.33
Nodes (5): Pairs, read_line(), read_open(), write_all(), write_pairs()

### Community 302 - "resources: ExtremaInterp"
Cohesion: 0.18
Nodes (9): ExtremaInterp, haveMax, haveMin, __host__, mn, mx, n, y0 (+1 more)

### Community 303 - "desktop: Options"
Cohesion: 0.20
Nodes (10): Options, enabled, filter, height, keep_open, list, out_dir, script (+2 more)

### Community 304 - "tools: graphify_update.py"
Cohesion: 0.27
Nodes (3): community_labels(), main(), module_of()

### Community 305 - "domain: NetTopology"
Cohesion: 0.20
Nodes (10): net_topology_name(), NetTopology, AllToAll, Chain, Custom, Grid, King, Ring (+2 more)

### Community 307 - "application: ParamGroupsUi"
Cohesion: 0.20
Nodes (10): ParamGroupsUi, filter, kind, kind_req, new_path, open, rename_from, rename_to (+2 more)

### Community 308 - "domain: CircuitResistor"
Cohesion: 0.20
Nodes (7): CircuitNode, name, CircuitResistor, a, b, origin, siemens

### Community 309 - "numerics: SpiceImportOptions"
Cohesion: 0.20
Nodes (8): SpiceImportOptions, base_dir, ideal_bjts, ideal_diodes, ideal_opamps, no_junction_caps, nullor_gain_min, vcvs_as_nullor

### Community 310 - "storage: record_from_json"
Cohesion: 0.42
Nodes (4): JParser, i, s, record_from_json()

### Community 311 - "tools: ocr_server.py"
Cohesion: 0.42
Nodes (4): load_model(), log(), main(), recognize()

### Community 312 - "domain: NeuralRule"
Cohesion: 0.22
Nodes (9): neural_rule_name(), NeuralRule, Chain, FixedDegree, Grid, King, Probability, Ring (+1 more)

### Community 313 - "application: ParametricQueueItem"
Cohesion: 0.22
Nodes (9): Kind, Bifurcation, LLE, LS, Metrics, ParametricQueueItem, index, kind (+1 more)

### Community 314 - "cuda: CpuExtremaInterp"
Cohesion: 0.22
Nodes (8): CpuExtremaInterp, haveMax, haveMin, mn, mx, n, y0, y1

### Community 315 - "numerics: EmittedKernel"
Cohesion: 0.22
Nodes (7): EmittedKernel, body, comp, n_caps, n_comp, n_vars, size

### Community 316 - "numerics: Model"
Cohesion: 0.22
Nodes (8): Model, bad, flags, in_subckt, name, p, raw, type

### Community 317 - "plotting: GpuLineSeriesSet3D"
Cohesion: 0.25
Nodes (3): GpuLineSeriesSet3D, bbox_data_, series_

### Community 318 - "storage: session_to_json_order_windows"
Cohesion: 0.42
Nodes (6): read_order_field(), session_from_json_order(), session_from_json_order_windows(), session_to_json_order(), session_to_json_order_windows(), order_autosave_seed()

### Community 319 - "resources: UcudaLyapClones"
Cohesion: 0.22
Nodes (9): UcudaLyapClones, active, Ec, F, Fc, nlo, rtol, y (+1 more)

### Community 320 - "tests: NetworkFixture"
Cohesion: 0.28
Nodes (6): network_fixtures(), NetworkFixture, initial, name, parameters, sys

### Community 321 - "desktop: Flight"
Cohesion: 0.25
Nodes (8): Flight, dir, haze, n, off, s, seed, v

### Community 322 - "domain: RegimeCounts"
Cohesion: 0.25
Nodes (8): RegimeCounts, data, fp, gen, osc, size, stopped, unb

### Community 323 - "equations: CdKind"
Cohesion: 0.25
Nodes (8): CdKind, Cx, Cx4, Cx4s3, Cx4s4, Cx4ss01, Cx4ss10, Real

### Community 324 - "plotting: PointMarker"
Cohesion: 0.25
Nodes (8): PointMarker, Circle, Cross, Diamond, Plus, Square, TriangleDown, TriangleUp

### Community 325 - "storage: write_network_config"
Cohesion: 0.50
Nodes (5): jmap_nonempty(), read_network_field(), session_from_json_network(), session_to_json_network(), write_network_config()

### Community 326 - "ui: dock_layout_process"
Cohesion: 0.43
Nodes (7): dock_layout_menu(), dock_layout_process(), file_stem(), has_file(), layouts_dir(), mode_dir(), refresh_saved()

### Community 327 - "ui: BifurcationPointGrid"
Cohesion: 0.25
Nodes (8): BifurcationPointGrid, columns, hi, lo, log, nodes, reverse, swap_axes

### Community 328 - "ui: TreeColumn"
Cohesion: 0.25
Nodes (8): TreeColumn, AddColumn, CloseColumn, DuplicateColumn, LevelColumn, NodeColumn, PowerColumn, TreeColumnCount

### Community 329 - "tests: vector"
Cohesion: 0.43
Nodes (5): analysis_matrix_methods(), exe_dir(), families(), methods(), solvers()

### Community 331 - "application: AppModel::load_or_init_dft1d_plot_windows"
Cohesion: 0.29
Nodes (3): AppModel::load_or_init_dft1d_plot_windows(), AppModel::load_or_init_order_plot_windows(), AppModel::load_or_init_parametric_plot_windows()

### Community 332 - "application: PendingScreenshot"
Cohesion: 0.29
Nodes (7): PendingScreenshot, active, frames_left, max_x, max_y, min_x, min_y

### Community 333 - "application: ThemeColorSlot"
Cohesion: 0.29
Nodes (7): ThemeColorSlot, kThemeAccent, kThemeBg, kThemeColorCount, kThemeFrame, kThemePlotBg, kThemeText

### Community 334 - "application: AppModel::broadcast_targets"
Cohesion: 0.48
Nodes (5): AppModel::broadcast_targets(), config_at(), config_count(), group_of(), slots_for()

### Community 335 - "application: cleanup_plot_windows_after_removal"
Cohesion: 0.48
Nodes (6): AppModel::remove_bifurcation_diagram(), AppModel::remove_lle_curve(), AppModel::remove_ls_curve(), AppModel::remove_metrics_config(), cleanup_plot_windows_after_removal(), cleanup_queue_after_removal()

### Community 336 - "domain: CircuitMultiplier"
Cohesion: 0.29
Nodes (7): CircuitMultiplier, out, x1, x2, y1, y2, z

### Community 337 - "domain: NetNode"
Cohesion: 0.29
Nodes (7): NetNode, initial_conditions, label, param_values, symmetry_s, ui_x, ui_y

### Community 338 - "numerics: circuit_netlist.cpp"
Cohesion: 0.71
Nodes (5): emit_spice_netlist(), eng(), mult_use(), ndiff(), node()

### Community 339 - "plotting: PlotRenderer::PlotRenderer"
Cohesion: 0.29
Nodes (3): PlotRenderer::ensure_fbo(), PlotRenderer::ensure_polyline_aux(), PlotRenderer::PlotRenderer()

### Community 340 - "resources: lyapunov_adaptive_part.cu"
Cohesion: 0.38
Nodes (4): lyapunovLLEAdContCUDA(), lyapunovLLEAdCUDA(), lyapunovLSAdContCUDA(), lyapunovLSAdCUDA()

### Community 342 - "desktop: .next_tab"
Cohesion: 0.33
Nodes (3): ModeButton, label, mode

### Community 344 - "analysis: ClassSteps"
Cohesion: 0.33
Nodes (6): ClassSteps, bin, bins, cycle, rest, stim

### Community 345 - "application: OcrState"
Cohesion: 0.33
Nodes (5): OcrState, Done, Failed, Idle, Running

### Community 346 - "numerics: Inductor"
Cohesion: 0.33
Nodes (6): Inductor, a, b, name, p, row

### Community 347 - "plotting: GpuLineSeries3D"
Cohesion: 0.33
Nodes (4): GpuLineSeries3D, point_count, vbo, vbo_val

### Community 349 - "plotting: InkBox"
Cohesion: 0.33
Nodes (6): InkBox, skew, top, top_center, x0, x1

### Community 350 - "ui: DockLayoutPreset"
Cohesion: 0.33
Nodes (6): DockLayoutPreset, kDockColumns, kDockGrid, kDockPresetCount, kDockRows, kDockTabs

### Community 351 - "resources: fastsync_attr.template.cu"
Cohesion: 0.47
Nodes (5): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t, fillFSMasterTrajectory()

### Community 352 - "?: main"
Cohesion: 0.40
Nodes (3): main(), read(), rt()

### Community 353 - "desktop: ShipDef"
Cohesion: 0.40
Nodes (5): ShipDef, off, s, seed, v

### Community 354 - "application: BroadcastTarget"
Cohesion: 0.40
Nodes (5): BroadcastTarget, group, idx, label, tab

### Community 356 - "numerics: RefRow"
Cohesion: 0.40
Nodes (5): RefRow, t, u, v, w

### Community 357 - "numerics: ErkTable"
Cohesion: 0.40
Nodes (4): ErkTable, B, M, S

### Community 359 - "plotting: SeriesPaletteId"
Cohesion: 0.40
Nodes (5): SeriesPaletteId, kSeriesPaletteCount, kSeriesPaletteDeep, kSeriesPaletteGem12, kSeriesPaletteGlow12

### Community 360 - "plotting: MathMark"
Cohesion: 0.40
Nodes (5): MathMark, r, x0, x1, y

### Community 362 - "ui: SplitMemo"
Cohesion: 0.40
Nodes (5): SplitMemo, c0, c1, len, ratio

### Community 363 - "resources: basins.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 364 - "resources: bifurcation1d_cont.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 365 - "resources: bifurcation1d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 366 - "resources: bifurcation2d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 367 - "resources: dft1d_cont.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 368 - "resources: dft1d_hsweep.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 369 - "resources: fastsync_grid.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 370 - "resources: lle1d_cont.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 371 - "resources: lle1d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 372 - "resources: lle2d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 373 - "resources: ls1d_cont.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 374 - "resources: ls1d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 375 - "resources: ls2d.template.cu"
Cohesion: 0.60
Nodes (4): curand_init(), curand_uniform(), curandState_t, curandStateXORWOW_t

### Community 376 - "application: LibraryEditMode"
Cohesion: 0.50
Nodes (4): LibraryEditMode, AddNew, EditExisting, None

### Community 377 - "application: InputMode"
Cohesion: 0.50
Nodes (4): InputMode, Image, Latex, Plain

### Community 378 - "equations: SchemeRow"
Cohesion: 0.50
Nodes (4): SchemeRow, name, order, symmetric

### Community 379 - "numerics: DopriTable"
Cohesion: 0.50
Nodes (3): DopriTable, B, M

### Community 380 - "numerics: Edge"
Cohesion: 0.50
Nodes (4): Edge, a, b, name

### Community 381 - "tests: Stats"
Cohesion: 0.50
Nodes (4): Stats, maximum, median, minimum

### Community 382 - "vcpkg.json: vcpkg.json"
Cohesion: 0.50
Nodes (3): dependencies, name, version-string

### Community 384 - "plotting: LogAxisTick"
Cohesion: 0.67
Nodes (3): LogAxisTick, major, value

### Community 386 - "plotting: PrefsField"
Cohesion: 0.67
Nodes (3): PrefsField, key, PlotWindowPrefs

## Knowledge Gaps
- **5651 isolated node(s):** `light`, `veil`, `popup`, `text`, `frame` (+5646 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 6713 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `map` connect `domain: map` to `application: AppModel`, `ui: plot_widgets.cpp`, `ui: item_w`, `application: SystemEditor`, `?: algorithm`, `domain: SystemRecord`, `ui: gui_common.h`, `plotting: plot_series_colors_changed`, `equations: adaptive_codegen.cpp`, `equations: codegen.cpp`, `equations: parse_dae`, `ui: neural.cpp`, `domain: BifurcationDiagramConfig`, `domain: OrderConfig`, `domain: param_groups.h`, `plotting: plot_axis.cpp`, `ui: draw_composition_builder`, `domain: CustomTabSharedConfig`, `domain: BifurcationAnalysisState`, `?: app_model.h`, `storage: system_library.cpp`, `domain: LSCurveConfig`, `domain: BasinsConfig`, `ui: gui_tree_state.h`, `tests: library_test.cpp`, `domain: FastSyncConfig`, `domain: Dft1DConfig`, `application: string #3`, `domain: LLECurveConfig`, `domain: PhaseAnalysisState`, `storage: record_from_json`, `domain: SignalMetricsConfig`, `equations: SparseLu`, `analysis: TreeSession`, `domain: NeuralPlan`, `application: AppModel::broadcast_field`, `numerics: string`, `analysis: tree_session.cpp`, `numerics: Model`, `tests: artifacts.h`, `domain: TreeNodeKind`, `numerics: SpiceImportResult`, `numerics: Builder`, `numerics: Deck`, `plotting: PlotWindowPrefs`, `storage: write_network_config`, `storage: JP`, `analysis: custom_session.cpp`, `domain: NeuralSource`, `analysis: PhaseRunInputs`, `analysis: net_generate_topology`, `application: AppSettings`, `application: BroadcastSlots`, `domain: NetNode`, `storage: session_json.cpp`, `application: TrajTimeEstimator`, `tests: dae_test.cpp`, `tests: spice_test.cpp`, `desktop: Sweep`, `domain: NetworkConfig`, `plotting: plot_legend.cpp`, `application: string #2`, `tests: compare`, `equations: sysparse.cpp`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **What connects `light`, `veil`, `popup` to the rest of the system?**
  _5651 weakly-connected nodes found - possible missing edges._
- **Should `application: AppModel` be split into smaller, more focused modules?**
  _Cohesion score 0.011299435028248588 - nodes in this community are weakly interconnected._
- **Why does `numb` connect `cuda: numb` to `resources: ucuda_adaptive.cuh`, `cuda: nvrtc_engine.cpp`, `resources: isinf`, `cuda: CpuMetricsAccum`, `resources: UcudaBdfCore`, `resources: ucuda_hp.h`, `resources: neural.template.cu`, `domain: AdaptiveRequest`, `cuda: host_rqa.cu`, `cuda: engine_runtime.cpp`, `resources: MetricsAccumNU`, `cuda: KrsCpuStep`, `cuda: PeakStream`, `resources: PeakStreamNU`, `numerics: krs_cpu.cpp`, `resources: ExtremaInterp`, `numerics: CpuAdaptiveKrs`, `cuda: CpuExtremaInterp`, `resources: UcudaLyapClones`, `resources: NeuralSpikeArgs`, `domain: configCUDA.h`, `analysis: order_session.cpp`, `resources: NeuralStepArgs`, `domain: FastSyncRequest`, `resources: fastsync_attr.template.cu`, `domain: Config`, `resources: UcudaCtlConst`, `cuda: RunSignals`, `domain: BasinsRequest`, `domain: ucuda_node_value`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Should `ui: plot_widgets.cpp` be split into smaller, more focused modules?**
  _Cohesion score 0.04495311638168781 - nodes in this community are weakly interconnected._
- **Why does `AppModel` connect `application: AppModel` to `application: BasinsQueueItem`, `application: Dft1DQueueItem`, `application: FastSyncQueueItem`, `application: NetworkQueueItem`, `application: OrderQueueItem`, `analysis: TrajWork`, `ui: neural.cpp`, `domain: map`, `application: run_next_queued`, `?: app_model.h`, `application: BroadcastTab`, `application: PrewarmWatch`, `domain: OrderPlotWindow`, `application: NetworkViewState`, `domain: CustomSession`, `analysis: TreeSession`, `application: ParametricQueueItem`, `application: AppModel::broadcast_field`, `domain: Kind`, `application: AppModel::load_or_init_dft1d_plot_windows`, `application: PendingScreenshot`, `analysis: BifurcationAnalysisSession`, `analysis: LLEAnalysisSession`, `analysis: LyapunovSpectrumAnalysisSession`, `application: app_queues.cpp`, `application: AppMode`, `ui: combo_items_need`, `application: TrajTimeEstimator`, `analysis: BasinsAnalysisSession`, `analysis: Dft1DAnalysisSession`, `analysis: OrderAnalysisSession`, `application: NeuralViewState`, `analysis: SignalMetricsAnalysisSession`, `application: LibraryEditMode`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Should `resources: ucuda_adaptive.cuh` be split into smaller, more focused modules?**
  _Cohesion score 0.03576494427558257 - nodes in this community are weakly interconnected._