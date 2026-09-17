ucuda_library(analysis
    src/basins_session.cpp
    src/bifurcation_session.cpp
    src/custom_session.cpp
    src/dft_session.cpp
    src/fastsync_session.cpp
    src/lle_session.cpp
    src/ls_session.cpp
    src/order_session.cpp
    src/phase_session.cpp
    src/session_helpers.cpp
)
target_link_libraries(ucuda_analysis PUBLIC ucuda::cuda ucuda::equations PRIVATE ucuda::numerics Threads::Threads)
