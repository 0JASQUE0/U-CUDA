ucuda_library(cuda
    src/circuit_nvrtc.cpp
    src/cudaMacros.cu
    src/engine_basins.cpp
    src/engine_bifurcation.cpp
    src/engine_cpu.cpp
    src/engine_dft.cpp
    src/engine_fastsync.cpp
    src/engine_lle.cpp
    src/engine_metrics.cpp
    src/engine_network.cpp
    src/engine_neural.cpp
    src/engine_ls.cpp
    src/engine_order.cpp
    src/engine_results.cpp
    src/engine_runtime.cpp
    src/host_rqa.cu
    src/nvrtc_engine.cpp
    src/parametric_engine.cpp
    src/phase_portrait_nvrtc.cpp
    src/runtime_settings.cpp
)
target_link_libraries(ucuda_cuda PUBLIC ucuda::domain ucuda::numerics PRIVATE ucuda::platform ucuda_export_format CUDA::cudart CUDA::cuda_driver CUDA::nvrtc Threads::Threads)
target_include_directories(ucuda_cuda PRIVATE "${PROJECT_SOURCE_DIR}/libs/cuda/include/ucuda/cuda")
set_target_properties(ucuda_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(MSVC)
    target_link_options(ucuda_cuda INTERFACE /NODEFAULTLIB:LIBCMT)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ucuda_cuda INTERFACE "LINKER:--disable-new-dtags")
endif()

# NVCC-built host library of the legacy batch executable (hostLibrary.cuh): kernels of
# cudaLibrary.cu launched from host_*.cu. The desktop app and the tests run everything through
# NVRTC modules and never link it. Built in every configuration so it keeps compiling.
add_library(ucuda_cuda_legacy STATIC
    src/cudaLibrary.cu
    src/host_basins.cu
    src/host_bifurcation.cu
    src/host_dft.cu
    src/host_fastsync.cu
    src/host_lyapunov.cu
    src/host_trajectory.cu
)
add_library(ucuda::cuda_legacy ALIAS ucuda_cuda_legacy)
target_link_libraries(ucuda_cuda_legacy PUBLIC ucuda::cuda PRIVATE ucuda_export_format CUDA::cudart)
target_include_directories(ucuda_cuda_legacy PRIVATE "${PROJECT_SOURCE_DIR}/libs/cuda/include/ucuda/cuda")
set_target_properties(ucuda_cuda_legacy PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(MSVC)
    set_target_properties(ucuda_cuda_legacy PROPERTIES COMPILE_PDB_NAME ucuda_cuda_legacy)
endif()
