add_library(ucuda_export_format STATIC src/data_export.cpp)
target_include_directories(ucuda_export_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ucuda_export_format PUBLIC ucuda::domain)
target_compile_features(ucuda_export_format PUBLIC cxx_std_17)
if(WIN32)
    target_compile_definitions(ucuda_export_format PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
if(MSVC)
    target_compile_options(ucuda_export_format PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    set_target_properties(ucuda_export_format PROPERTIES COMPILE_PDB_NAME ucuda_export_format)
endif()

ucuda_library(storage
    src/app_config.cpp
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
target_link_libraries(ucuda_storage PUBLIC ucuda::domain PRIVATE ucuda::platform ucuda_export_format)
