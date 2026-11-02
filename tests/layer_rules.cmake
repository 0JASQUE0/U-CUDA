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
# The one kernel header domain may see: the layout part of ucuda_adaptive.cuh (UcudaAdaptParams and the
# controller structures go to the kernel byte for byte, so host and device share one definition).
# adaptive_settings.h includes it under UCUDA_ADAPT_LAYOUT_ONLY; anything else from resources/kernels is a leak.
file(GLOB_RECURSE domain_files "${ROOT}/libs/domain/*.h" "${ROOT}/libs/domain/*.cpp")
foreach(f IN LISTS domain_files)
    file(STRINGS "${f}" hits REGEX "^[ \t]*#[ \t]*include[ \t]+\"[^\"]+\"")
    foreach(h IN LISTS hits)
        string(REGEX REPLACE "^[^\"]*\"([^\"]+)\".*$" "\\1" inc "${h}")
        if(NOT inc MATCHES "^ucuda/domain/" AND NOT inc STREQUAL "ucuda_adaptive.cuh")
            file(RELATIVE_PATH rel "${ROOT}" "${f}")
            set(failures "${failures}\n  domain includes a non-domain header: ${rel}: ${inc}")
        endif()
    endforeach()
endforeach()
# Domain declares only what domain defines. A free function declared in a domain header but
# implemented in analysis, cuda or numerics is an upward dependency the include check cannot see
# and a single-project build links silently: its declaration belongs in the implementing module's
# header. A non-inline top-level declaration in a domain header needs a definition in libs/domain/src.
file(GLOB domain_headers "${ROOT}/libs/domain/include/ucuda/domain/*.h")
file(GLOB domain_sources "${ROOT}/libs/domain/src/*.cpp")
set(domain_defs "")
foreach(f IN LISTS domain_sources)
    file(READ "${f}" txt)
    string(APPEND domain_defs "${txt}")
endforeach()
foreach(f IN LISTS domain_headers)
    file(STRINGS "${f}" decls REGEX "^[A-Za-z_][A-Za-z0-9_:<>,*& ]*[ *&][A-Za-z_][A-Za-z0-9_]*[ \t]*\\([^{]*$")
    foreach(d IN LISTS decls)
        if(d MATCHES "^(inline|static|constexpr|template|typedef|using|struct|class|enum|return|friend|extern|namespace)[ \t]"
           OR d MATCHES "^[^(]*[ \t](inline|constexpr)[ \t]")   # UCUDA_HD inline ...
            continue()
        endif()
        string(REGEX REPLACE "^[^(]*[ *&]([A-Za-z_][A-Za-z0-9_]*)[ \t]*\\(.*$" "\\1" name "${d}")
        string(FIND "${domain_defs}" "${name}(" at)
        if(at EQUAL -1)
            file(RELATIVE_PATH rel "${ROOT}" "${f}")
            set(failures "${failures}\n  domain declares ${name} without defining it in libs/domain/src: ${rel}")
        endif()
    endforeach()
endforeach()

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

# Public headers of analysis and application name the engine only by forward declaration: CUDA and
# numerics stay PRIVATE dependencies, so nothing above application sees them even transitively.
foreach(lib analysis application)
    rule("${lib} public header includes cuda/numerics" "libs/${lib}/include/*.h" "#include[ \t]+\"ucuda/(cuda|numerics)/")
endforeach()

# UI reaches computation only through application (compute_services.h, AppModel): no CUDA runtime,
# NVRTC or CPU compiler paths from a window. Equations (text and DAE structure) stays allowed.
rule("ui includes cuda/numerics" "libs/ui/*.h"   "#include[ \t]+(\"ucuda/(cuda|numerics)/|<(cuda|nvrtc)[A-Za-z_]*\\.h>)")
rule("ui includes cuda/numerics" "libs/ui/*.cpp" "#include[ \t]+(\"ucuda/(cuda|numerics)/|<(cuda|nvrtc)[A-Za-z_]*\\.h>)")

if(failures)
    message(FATAL_ERROR "Layer rules violated:${failures}")
endif()
message(STATUS "Layer rules: OK")
