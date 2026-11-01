include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723)
FetchContent_Declare(googlebenchmark GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG 31c66f47f60d06790a4885a261d682ba339bd100)
# Structured configuration and artifacts are test-only; production gains no JSON dependency.
FetchContent_Declare(test_json GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 65ee68451d8eb2b5f3a30b410476ab83deb3289b)
FetchContent_MakeAvailable(googletest googlebenchmark test_json)
if(MSVC)
    # VS/CUDA generators otherwise emit a directory as the static library's /Fd
    # path, and MSBuild tries to copy that directory as a Debug PDB.
    set_target_properties(benchmark PROPERTIES COMPILE_PDB_NAME "benchmark")
    set_target_properties(benchmark_main PROPERTIES COMPILE_PDB_NAME "benchmark_main")
endif()
