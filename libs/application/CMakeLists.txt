ucuda_library(application
    src/app_model.cpp
    src/compute_services.cpp
    src/tick.cpp
)
# storage is public: session_persistence.h wraps storage's state serializers for whole sessions.
target_link_libraries(ucuda_application PUBLIC ucuda::analysis ucuda::platform ucuda::storage PRIVATE ucuda::numerics ucuda::ocr)
