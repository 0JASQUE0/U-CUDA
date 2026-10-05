# An empty TARGET_RUNTIME_DLLS list is valid for a statically linked GUI.
foreach(dll IN LISTS runtime_dlls)
    if(NOT dll STREQUAL "")
        get_filename_component(dll_name "${dll}" NAME)
        file(COPY_FILE "${dll}" "${destination}/${dll_name}" ONLY_IF_DIFFERENT)
    endif()
endforeach()
