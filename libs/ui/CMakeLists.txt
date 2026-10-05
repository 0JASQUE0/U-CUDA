ucuda_library(ui
    src/basins.cpp
    src/bifurcation.cpp
    src/custom.cpp
    src/dft.cpp
    src/fastsync.cpp
    src/gui.cpp
    src/library.cpp
    src/lyapunov.cpp
    src/network.cpp
    src/order.cpp
    src/parametric_windows.cpp
    src/phase.cpp
    src/ui_resources.cpp
    src/widgets.cpp
    src/workspace.cpp
)
target_link_libraries(ucuda_ui PUBLIC ucuda::application ucuda::storage PRIVATE ucuda::plotting ucuda::numerics)
