ucuda_library(platform
    src/image_source.cpp
)
target_include_directories(ucuda_platform PRIVATE "${PROJECT_SOURCE_DIR}/third_party/stb")
if(WIN32)
    target_link_libraries(ucuda_platform PRIVATE user32 gdi32 comdlg32)
endif()
list(GET CUDAToolkit_INCLUDE_DIRS 0 ucuda_toolkit_include)
target_compile_definitions(ucuda_platform PUBLIC U_CUDA_TOOLKIT_INCLUDE="${ucuda_toolkit_include}")
