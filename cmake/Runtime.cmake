file(GLOB_RECURSE UC_RUNTIME_INPUTS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/resources/kernels/*.cu"
    "${PROJECT_SOURCE_DIR}/libs/cuda/src/*.cu"
    "${PROJECT_SOURCE_DIR}/libs/cuda/src/*.cuh"
    "${PROJECT_SOURCE_DIR}/libs/cuda/include/*.cuh"
    "${PROJECT_SOURCE_DIR}/libs/domain/include/*.h")
function(ucuda_runtime target)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${UC_RUNTIME_INPUTS} "${PROJECT_SOURCE_DIR}/cmake/PackRuntime.cmake")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} "-Dsource_root=${PROJECT_SOURCE_DIR}" "-Ddestination=$<TARGET_FILE_DIR:${target}>" -P "${PROJECT_SOURCE_DIR}/cmake/PackRuntime.cmake"
        VERBATIM)
    if(WIN32)
        file(GLOB cuda_runtime_dlls
            "${CUDAToolkit_BIN_DIR}/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/nvrtc64*.dll" "${CUDAToolkit_BIN_DIR}/nvrtc-builtins64*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/x64/nvrtc64*.dll" "${CUDAToolkit_BIN_DIR}/x64/nvrtc-builtins64*.dll")
        list(FILTER cuda_runtime_dlls EXCLUDE REGEX "\\.alt\\.dll$")
        add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND}
            "-Druntime_dlls=$<TARGET_RUNTIME_DLLS:${target}>;${cuda_runtime_dlls}"
            "-Ddestination=$<TARGET_FILE_DIR:${target}>"
            -P "${PROJECT_SOURCE_DIR}/cmake/CopyRuntimeDlls.cmake" VERBATIM)
    endif()
endfunction()
