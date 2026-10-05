ucuda_library(numerics
    src/circuit_codegen.cpp
    src/circuit_emit.cpp
    src/circuit_netlist.cpp
    src/circuit_solver.cpp
    src/circuit_synth.cpp
    src/integrator.cpp
    src/krs_cpu.cpp
)
target_link_libraries(ucuda_numerics PUBLIC ucuda::equations PRIVATE ucuda::platform ${CMAKE_DL_LIBS} Threads::Threads)
# ucuda_hp.h (double-double/quad-double) — общий для NVRTC-ядер и CPU-сборки КРС;
# источник истины лежит среди ядер, упаковка runtime копирует его в kernels/.
target_include_directories(ucuda_numerics PUBLIC "${PROJECT_SOURCE_DIR}/resources/kernels")
