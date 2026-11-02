ucuda_library(application
    src/app_broadcast.cpp
    src/app_model.cpp
    src/app_plot_windows.cpp
    src/app_queues.cpp
    src/app_settings.cpp
    src/compute_services.cpp
    src/system_editor.cpp
    src/tick.cpp
)
# storage is public: session_persistence.h wraps storage's state serializers for whole sessions.
target_link_libraries(ucuda_application PUBLIC ucuda::analysis ucuda::platform ucuda::storage PRIVATE ucuda::numerics ucuda::ocr)
