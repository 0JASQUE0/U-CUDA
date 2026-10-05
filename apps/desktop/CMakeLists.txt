set(ucuda_desktop_sources app_main.cpp)
if(WIN32)
    list(APPEND ucuda_desktop_sources "${PROJECT_SOURCE_DIR}/resources/windows/U-CUDA.rc")
endif()
add_executable(U-CUDA ${ucuda_desktop_sources})
target_link_libraries(U-CUDA PRIVATE ucuda::ui ucuda::plotting ucuda::ocr)
target_compile_definitions(U-CUDA PRIVATE
    U_CUDA_PROJECT_DIR="${PROJECT_SOURCE_DIR}"
    U_CUDA_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/resources")
ucuda_runtime(U-CUDA)
