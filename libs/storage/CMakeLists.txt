ucuda_library(storage
    src/app_config.cpp
    src/data_export.cpp
    src/session_basins.cpp
    src/session_custom.cpp
    src/session_dft1d.cpp
    src/session_dft1d_windows.cpp
    src/session_fastsync.cpp
    src/session_json.cpp
    src/session_lle.cpp
    src/session_ls.cpp
    src/session_parametric.cpp
    src/session_parametric_windows.cpp
    src/session_phase.cpp
    src/system_library.cpp
)
target_link_libraries(ucuda_storage PUBLIC ucuda::domain PRIVATE ucuda::platform)
