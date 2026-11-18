#pragma once
// UI test mode of the desktop app: Dear ImGui Test Engine drives the real GUI.
// Compiled only with -DU_CUDA_UI_TESTS=ON (CMake); the U-CUDA.sln build never sees it.
// Usage and command reference: apps/desktop/ui_test/README.md.
#include <string>

struct GLFWwindow;
struct ImGuiTestEngine;
struct ImGuiTest;
class AppModel;
class SystemLibrary;

namespace ui_test {

struct Options {
    bool        enabled   = false;
    bool        list      = false;   // --ui-list: print registered tests and exit
    bool        show_ui   = false;   // --ui-show: draw the Test Engine window
    bool        keep_open = false;   // --ui-keep: do not exit when the queue is done
    int         speed     = 0;       // 0 fast, 1 normal, 2 cinematic
    int         width = 1600, height = 900;
    std::string filter;              // --ui-test <filter>
    std::string script;              // --ui-script <file>
    std::string out_dir = "ui_out";  // --ui-out <dir>: screenshots, results.xml
};

// Parses the --ui-* arguments; returns true when UI test mode is on.
bool parse_args(int argc, char** argv);
const Options& options();
bool active();

// App objects the compiled tests may inspect (set once in main).
void bind_app(AppModel* model, SystemLibrary* library);
AppModel*      app_model();
SystemLibrary* app_library();

void configure_window(GLFWwindow* window);   // fixed size instead of maximize
void start();                                // after ImGui context + backends
void draw_windows();                         // inside the frame, after draw_gui
bool wants_max_speed();                      // skip vsync while running fast
bool post_swap();                            // after glfwSwapBuffers; true = quit now
int  stop();                                 // before ImGui::DestroyContext; exit code
void destroy();                              // after ImGui::DestroyContext

// Registration (ui_tests.cpp, ui_test_script.cpp).
void register_tests(ImGuiTestEngine* engine);
ImGuiTest* register_script_test(ImGuiTestEngine* engine, const std::string& path);

} // namespace ui_test
