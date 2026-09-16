cmake_minimum_required(VERSION 3.25)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(U_CUDA VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(U_CUDA_BUILD_LEGACY "Build the legacy batch calculation executable" OFF)
include(FetchContent)
find_package(CUDAToolkit REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 UC_CUDA_INCLUDE)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# Pin all GUI dependencies to compatible revisions on both platforms.
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG 4806a1924ff6181180bf5e4b8b79ab4394118875 SOURCE_SUBDIR ucuda-no-cmake)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG 4707b245fbcd69075b1a8a74fa8d2435561b3134 SOURCE_SUBDIR ucuda-no-cmake)
FetchContent_Declare(implot3d GIT_REPOSITORY https://github.com/brenocq/implot3d.git
    GIT_TAG 41ae3e447c0de20ecab95d38a4b4dc0835a3efc2 SOURCE_SUBDIR ucuda-no-cmake)
# These libraries are built below, avoiding their demo/test subprojects.
FetchContent_MakeAvailable(imgui implot implot3d)
find_package(glfw3 3.3 QUIET)
if(NOT TARGET glfw)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG a74efa0d5628b74adc0426af4c5710e287fa7c2c)
    FetchContent_MakeAvailable(glfw)
endif()
add_library(ucuda_gui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d.cpp ${implot3d_SOURCE_DIR}/implot3d_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d_meshes.cpp)
target_include_directories(ucuda_gui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR} ${implot3d_SOURCE_DIR})
target_link_libraries(ucuda_gui PUBLIC glfw OpenGL::GL ${CMAKE_DL_LIBS})

set(UC_SOURCES
    analysis_session.cpp custom_session.cpp order_session.cpp app_model.cpp app_config.cpp
    codegen.cpp data_export.cpp digit_input.cpp glad.c gpu_line_series.cpp gpu_line_series_3d.cpp
    gui.cpp image_source.cpp integrator.cpp krs_cpu.cpp nvrtc_engine.cpp ocr_client_win.cpp
    parametric_engine.cpp phase_portrait_nvrtc.cpp plot_axis.cpp plot_camera_3d.cpp plot_legend.cpp
    heatmap_view.cpp plot_renderer.cpp plot_view_2d.cpp plot_view_3d.cpp session_io.cpp
    sysparse.cpp system_library.cpp cudaLibrary.cu cudaMacros.cu hostLibrary.cu)
list(TRANSFORM UC_SOURCES PREPEND "U-CUDA/")
add_library(ucuda_core STATIC ${UC_SOURCES})
target_include_directories(ucuda_core PUBLIC U-CUDA U-CUDA/include)
target_link_libraries(ucuda_core PUBLIC ucuda_gui CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
    Threads::Threads ${CMAKE_DL_LIBS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # NVRTC dlopens its builtins library. RPATH also applies to that indirect
    # lookup, allowing a non-system CUDA Toolkit without LD_LIBRARY_PATH.
    target_link_options(ucuda_core INTERFACE "LINKER:--disable-new-dtags")
endif()
target_compile_definitions(ucuda_core PUBLIC
    U_CUDA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA"
    U_CUDA_TOOLKIT_INCLUDE="${UC_CUDA_INCLUDE}")
if(WIN32)
    target_compile_definitions(ucuda_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(ucuda_core PUBLIC comdlg32 user32 gdi32)
endif()
if(MSVC)
    target_compile_options(ucuda_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    target_compile_options(ucuda_gui PRIVATE /utf-8)
endif()
set_target_properties(ucuda_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(U-CUDA U-CUDA/app_main.cpp)
target_link_libraries(U-CUDA PRIVATE ucuda_core)

function(ucuda_runtime target)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/kernels" "$<TARGET_FILE_DIR:${target}>/kernels"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/cudaLibrary.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/cudaLibrary.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/cudaMacros.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/configCUDA.h" "$<TARGET_FILE_DIR:${target}>/kernels"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/ocr_server.py" "$<TARGET_FILE_DIR:${target}>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA/library_template" "$<TARGET_FILE_DIR:${target}>/library_template"
        VERBATIM)
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS VERBATIM)
    endif()
endfunction()
ucuda_runtime(U-CUDA)
if(U_CUDA_BUILD_LEGACY)
    add_executable(U-CUDA-legacy U-CUDA/main_NonLinAnal.cu)
    target_link_libraries(U-CUDA-legacy PRIVATE ucuda_core)
    ucuda_runtime(U-CUDA-legacy)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(gpu_smoke tests/gpu_smoke.cpp)
    target_link_libraries(gpu_smoke PRIVATE ucuda_core)
    ucuda_runtime(gpu_smoke)
    add_test(NAME gpu_smoke COMMAND gpu_smoke)
    set_tests_properties(gpu_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
if(BUILD_TESTING AND UNIX)
    add_executable(platform_test tests/platform_test.cpp U-CUDA/krs_cpu.cpp U-CUDA/ocr_client_win.cpp)
    target_include_directories(platform_test PRIVATE U-CUDA)
    target_compile_definitions(platform_test PRIVATE U_CUDA_TEST_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/U-CUDA")
    target_link_libraries(platform_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    add_test(NAME platform COMMAND platform_test)
endif()
