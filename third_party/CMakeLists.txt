find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# Pin all GUI dependencies to compatible revisions on both platforms.
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG b61e56346a92cfcaf1f43a545ca37b0b32239654 SOURCE_SUBDIR ucuda-no-cmake)
FetchContent_Declare(implot GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG 524f9fcd48d76c13fdf94c5ffbba8787a1ff7e39 SOURCE_SUBDIR ucuda-no-cmake)
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
add_library(ucuda_gui_dependencies STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d.cpp ${implot3d_SOURCE_DIR}/implot3d_items.cpp
    ${implot3d_SOURCE_DIR}/implot3d_meshes.cpp)
target_include_directories(ucuda_gui_dependencies PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR} ${implot3d_SOURCE_DIR})
target_link_libraries(ucuda_gui_dependencies PUBLIC glfw OpenGL::GL ${CMAKE_DL_LIBS})
if(MSVC)
    # MSBuild expects an actual PDB filename when copying static-library symbols.
    set_target_properties(glfw PROPERTIES COMPILE_PDB_NAME glfw)
    set_target_properties(ucuda_gui_dependencies PROPERTIES COMPILE_PDB_NAME ucuda_gui_dependencies)
endif()


add_library(ucuda_glad STATIC glad/src/glad.c)
if(MSVC)
    set_target_properties(ucuda_glad PROPERTIES COMPILE_PDB_NAME ucuda_glad)
endif()
target_include_directories(ucuda_glad PUBLIC glad/include)
target_link_libraries(ucuda_glad PUBLIC OpenGL::GL ${CMAKE_DL_LIBS})
