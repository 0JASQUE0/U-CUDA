ucuda_library(plotting
    src/digit_input.cpp
    src/gpu_line_series.cpp
    src/gpu_line_series_3d.cpp
    src/heatmap_view.cpp
    src/plot_axis.cpp
    src/plot_camera_3d.cpp
    src/plot_legend.cpp
    src/plot_renderer.cpp
    src/plot_view_2d.cpp
    src/plot_view_3d.cpp
    src/screenshot.cpp
)
target_link_libraries(ucuda_plotting PUBLIC ucuda::domain ucuda_gui_dependencies ucuda_glad PRIVATE ucuda::platform)
