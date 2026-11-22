cmake_minimum_required(VERSION 3.25)

if(DEFINED cuda_bin)
    foreach(bin IN ITEMS "${cuda_bin}" "${cuda_bin}/x64")
        file(GLOB ridge_dlls "${bin}/cublas64*.dll" "${bin}/cublasLt64*.dll"
            "${bin}/cusolver64*.dll" "${bin}/cusparse64*.dll" "${bin}/nvJitLink*.dll"
            "${bin}/nvjitlink*.dll")
        list(APPEND runtime_dlls ${ridge_dlls})
    endforeach()
    list(REMOVE_DUPLICATES runtime_dlls)
endif()
# An empty TARGET_RUNTIME_DLLS list is valid for a statically linked GUI.
foreach(dll IN LISTS runtime_dlls)
    if(NOT dll STREQUAL "")
        get_filename_component(dll_name "${dll}" NAME)
        file(COPY_FILE "${dll}" "${destination}/${dll_name}" ONLY_IF_DIFFERENT)
    endif()
endforeach()
