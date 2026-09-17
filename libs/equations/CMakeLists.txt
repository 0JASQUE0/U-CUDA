ucuda_library(equations
    src/adaptive_codegen.cpp
    src/codegen.cpp
    src/expression_parser.cpp
    src/scheme_codegen.cpp
    src/sysparse.cpp
    src/system_evaluator.cpp
)
target_link_libraries(ucuda_equations PUBLIC ucuda::domain)
