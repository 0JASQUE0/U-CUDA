file(MAKE_DIRECTORY "${destination}/kernels")
# Remove the obsolete generated header from packages created by older builds.
file(REMOVE "${destination}/kernels/adaptive_solver.h")
file(COPY "${source_root}/resources/kernels/" DESTINATION "${destination}/kernels")
set(runtime_headers
    libs/cuda/src/cudaLibrary.cu
    libs/cuda/include/ucuda/cuda/cudaLibrary.cuh
    libs/cuda/include/ucuda/cuda/cudaMacros.cuh
    libs/domain/include/ucuda/domain/configCUDA.h
    libs/domain/include/ucuda/domain/sparse_lu.h)
foreach(input IN LISTS runtime_headers)
    file(READ "${source_root}/${input}" content)
    if(input STREQUAL "libs/cuda/src/cudaLibrary.cu")
        set(content "")
        file(READ "${source_root}/libs/cuda/src/device/models.cuh" part)
        string(APPEND content "${part}\n")
        file(READ "${source_root}/libs/cuda/src/device/fastsync.cuh" part)
        string(APPEND content "${part}\n")
        file(READ "${source_root}/libs/cuda/src/device/integration.cuh" part)
        string(APPEND content "${part}\n")
        file(READ "${source_root}/libs/cuda/src/device/peaks.cuh" part)
        string(APPEND content "${part}\n")
        file(READ "${source_root}/libs/cuda/src/device/clustering.cuh" part)
        string(APPEND content "${part}\n")
    endif()
    # Each source has its own UTF-8 BOM; concatenation must not keep interior BOMs.
    string(ASCII 239 187 191 utf8_bom)
    string(REPLACE "${utf8_bom}" "" content "${content}")
    # NVRTC registers these headers by their original flat names.
    string(REGEX REPLACE "ucuda/[a-z_]+/" "" content "${content}")
    get_filename_component(name "${input}" NAME)
    file(WRITE "${destination}/kernels/${name}" "${content}")
endforeach()
file(COPY "${source_root}/tools/ocr/ocr_server.py" DESTINATION "${destination}")
file(COPY "${source_root}/resources/library_template" DESTINATION "${destination}")
file(COPY "${source_root}/resources/fonts/" DESTINATION "${destination}/fonts"
     FILES_MATCHING PATTERN "*.otf" PATTERN "*.ttf" PATTERN "OFL.txt")
file(COPY "${source_root}/resources/themes/" DESTINATION "${destination}/themes"
     FILES_MATCHING PATTERN "*.ppm")
