ucuda_library(numerics
    src/integrator.cpp
    src/krs_cpu.cpp
)
target_link_libraries(ucuda_numerics PUBLIC ucuda::equations PRIVATE ucuda::platform ${CMAKE_DL_LIBS} Threads::Threads)
