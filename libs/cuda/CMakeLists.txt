ucuda_library(cuda
    src/circuit_nvrtc.cpp
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
if(MSVC)
    target_link_options(ucuda_cuda INTERFACE /NODEFAULTLIB:LIBCMT)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ucuda_cuda INTERFACE "LINKER:--disable-new-dtags")
endif()
