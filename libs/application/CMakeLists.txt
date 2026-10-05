ucuda_library(application
    src/app_model.cpp
    src/tick.cpp
)
target_link_libraries(ucuda_application PUBLIC ucuda::analysis ucuda::platform PRIVATE ucuda::storage ucuda::ocr)
