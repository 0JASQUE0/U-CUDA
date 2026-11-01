# Static checks of the module boundaries that the compiler does not enforce:
# a header may compile through a transitive include path even when the layer must not use it.
# Run: cmake -DROOT=<repo> -P tests/layer_rules.cmake
if(NOT ROOT)
    message(FATAL_ERROR "ROOT is not set")
endif()

set(failures "")

# rule(<name> <glob> <regex>): no file under <glob> may contain a line matching <regex>.
function(rule name glob regex)
    file(GLOB_RECURSE files "${ROOT}/${glob}")
    foreach(f IN LISTS files)
        file(STRINGS "${f}" hits REGEX "${regex}")
        foreach(h IN LISTS hits)
            file(RELATIVE_PATH rel "${ROOT}" "${f}")
            string(STRIP "${h}" h)
            set(failures "${failures}\n  ${name}: ${rel}: ${h}")
        endforeach()
    endforeach()
    set(failures "${failures}" PARENT_SCOPE)
endfunction()

set(higher "ucuda/(cuda|numerics|equations|analysis|storage|application|plotting|ui|ocr)/")

# Domain: plain serializable types. Sessions keep only their *State here (futures, tokens and
# run methods live in analysis), so no thread primitives and no engine.
rule("domain includes a higher module" "libs/domain/*.h"   "#include[ \t]+\"${higher}")
rule("domain includes a higher module" "libs/domain/*.cpp" "#include[ \t]+\"${higher}")
rule("domain uses <future>/<thread>"   "libs/domain/*.h"   "#include[ \t]+<(future|thread)>")
rule("domain names an engine"          "libs/domain/*.h"   "^[^/]*(ParametricEngine|NvrtcEngine)")

# Storage serializes domain values only: no compute, application or GUI headers.
rule("storage includes compute/GUI" "libs/storage/*.h"
     "#include[ \t]+\"ucuda/(cuda|numerics|analysis|application|plotting|ui)/")
rule("storage includes compute/GUI" "libs/storage/*.cpp"
     "#include[ \t]+\"ucuda/(cuda|numerics|analysis|application|plotting|ui)/")

# Computation never includes presentation.
foreach(lib cuda numerics equations analysis)
    rule("${lib} includes GUI" "libs/${lib}/*.h"   "#include[ \t]+(\"ucuda/(plotting|ui|application)/|<imgui|<GLFW|\"glad)")
    rule("${lib} includes GUI" "libs/${lib}/*.cpp" "#include[ \t]+(\"ucuda/(plotting|ui|application)/|<imgui|<GLFW|\"glad)")
endforeach()

if(failures)
    message(FATAL_ERROR "Layer rules violated:${failures}")
endif()
message(STATUS "Layer rules: OK")
