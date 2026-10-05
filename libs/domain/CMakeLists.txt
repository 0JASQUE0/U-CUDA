ucuda_library(domain
    src/session_defaults.cpp
)
target_include_directories(ucuda_domain PUBLIC "${PROJECT_SOURCE_DIR}/resources/kernels")
