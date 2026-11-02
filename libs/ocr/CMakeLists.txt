ucuda_library(ocr
    src/ocr_client_win.cpp
)
target_link_libraries(ucuda_ocr PRIVATE ucuda::platform Threads::Threads)
