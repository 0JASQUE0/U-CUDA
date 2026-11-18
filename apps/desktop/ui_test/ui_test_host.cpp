// UI test mode host: owns the Dear ImGui Test Engine instance of the desktop app.
#include "ui_test_host.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_te_engine.h"
#include "imgui_te_internal.h"    // ImGuiTestEngine_PassFilter
#include "imgui_te_coroutine.h"
#include "imgui_te_exporters.h"
#include "imgui_te_ui.h"
#include "imgui_te_utils.h"
#include "imgui_capture_tool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace ui_test {
namespace {

Options         g_opt;
ImGuiTestEngine* g_engine  = nullptr;
AppModel*       g_model   = nullptr;
SystemLibrary*  g_library = nullptr;
bool            g_listed  = false;
int             g_frames  = 0;
std::string     g_results_path;

void print_usage() {
    std::printf(
        "U-CUDA UI test mode (Dear ImGui Test Engine)\n"
        "  --ui-test <filter>   run compiled tests; filter = substring or \"all\" (\"^\" prefix = start match)\n"
        "  --ui-script <file>   run a command script (see apps/desktop/ui_test/README.md)\n"
        "  --ui-list            list compiled tests and exit\n"
        "  --ui-out <dir>       output directory for screenshots and results.xml (default ui_out)\n"
        "  --ui-size <WxH>      window size (default 1600x900)\n"
        "  --ui-speed <fast|normal|cinematic>\n"
        "  --ui-show            show the Test Engine window\n"
        "  --ui-keep            keep the app open after the tests\n"
        "  --ui-fresh           delete _app_config.json and imgui.ini next to the exe before start\n");
}

std::string exe_dir_of(const char* argv0) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(argv0, ec);
    return p.parent_path().string();
}

// glReadPixels of the default framebuffer, rows flipped to top-left origin.
bool capture_gl(ImGuiID, int x, int y, int w, int h, unsigned int* pixels, void* user) {
    GLFWwindow* window = (GLFWwindow*)user;
    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    (void)fb_w;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(x, fb_h - y - h, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    std::vector<unsigned int> row((size_t)w);
    for (int r = 0; r < h / 2; ++r) {
        unsigned int* a = pixels + (size_t)r * w;
        unsigned int* b = pixels + (size_t)(h - 1 - r) * w;
        std::memcpy(row.data(), a, (size_t)w * 4);
        std::memcpy(a, b, (size_t)w * 4);
        std::memcpy(b, row.data(), (size_t)w * 4);
    }
    return true;
}

GLFWwindow* g_window = nullptr;

// Tests must never write into the developer's U-CUDA/library (sessions are saved on every change).
// UI test mode points U_CUDA_LIBRARY at <exe>/ui_library: a copy of that library (read only from
// the source), refreshed by --ui-fresh. A U_CUDA_LIBRARY set by the caller wins.
void isolate_library(const std::string& exe_dir, bool fresh) {
    namespace fs = std::filesystem;
    if (const char* env = std::getenv("U_CUDA_LIBRARY"); env && *env) {
        std::printf("[ui] library (from U_CUDA_LIBRARY): %s\n", env);
        return;
    }
    const fs::path dst = fs::path(exe_dir) / "ui_library";
    std::error_code ec;
    if (fresh) fs::remove_all(dst, ec);
    if (!fs::exists(dst)) {
        fs::path src;
#ifdef U_CUDA_PROJECT_DIR
        src = fs::path(U_CUDA_PROJECT_DIR) / "U-CUDA" / "library";
        if (!fs::exists(src)) src = fs::path(U_CUDA_RESOURCE_DIR) / "library_template";
#endif
        if (src.empty() || !fs::exists(src)) src = fs::path(exe_dir) / "library_template";
        fs::copy(src, dst, fs::copy_options::recursive, ec);
        if (ec) { std::fprintf(stderr, "cannot copy library %s -> %s: %s\n", src.string().c_str(),
                               dst.string().c_str(), ec.message().c_str()); std::exit(2); }
    }
#ifdef _WIN32
    _putenv_s("U_CUDA_LIBRARY", dst.string().c_str());
#else
    setenv("U_CUDA_LIBRARY", dst.string().c_str(), 1);
#endif
    std::printf("[ui] library (isolated copy): %s\n", dst.string().c_str());
    std::fflush(stdout);
}

} // namespace

bool parse_args(int argc, char** argv) {
    bool fresh = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--ui-test")   { g_opt.enabled = true; g_opt.filter = next("--ui-test"); }
        else if (a == "--ui-script") { g_opt.enabled = true; g_opt.script = next("--ui-script"); }
        else if (a == "--ui-list")   { g_opt.enabled = true; g_opt.list = true; }
        else if (a == "--ui-out")    { g_opt.out_dir = next("--ui-out"); }
        else if (a == "--ui-show")   { g_opt.enabled = true; g_opt.show_ui = true; }
        else if (a == "--ui-keep")   { g_opt.keep_open = true; }
        else if (a == "--ui-fresh")  { fresh = true; }
        else if (a == "--ui-size") {
            const std::string v = next("--ui-size");
            if (std::sscanf(v.c_str(), "%dx%d", &g_opt.width, &g_opt.height) != 2) {
                std::fprintf(stderr, "--ui-size expects WxH\n"); std::exit(2);
            }
        } else if (a == "--ui-speed") {
            const std::string v = next("--ui-speed");
            g_opt.speed = v == "cinematic" ? 2 : v == "normal" ? 1 : 0;
        } else if (a == "--ui-help" || a == "--help") { print_usage(); std::exit(0); }
        else if (a.rfind("--ui-", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", a.c_str()); print_usage(); std::exit(2);
        }
    }
    if (g_opt.show_ui && g_opt.filter.empty() && g_opt.script.empty()) g_opt.keep_open = true;
    if (fresh && argc > 0) {
        const std::string dir = exe_dir_of(argv[0]);
        std::error_code ec;
        std::filesystem::remove(dir + "/_app_config.json", ec);
        std::filesystem::remove(dir + "/imgui.ini", ec);
    }
    if (g_opt.enabled) {
        std::error_code ec;
        std::filesystem::create_directories(g_opt.out_dir, ec);
        isolate_library(argc > 0 ? exe_dir_of(argv[0]) : std::string("."), fresh);
    }
    return g_opt.enabled;
}

const Options& options() { return g_opt; }
bool active() { return g_opt.enabled; }

void bind_app(AppModel* model, SystemLibrary* library) { g_model = model; g_library = library; }
AppModel*      app_model()   { return g_model; }
SystemLibrary* app_library() { return g_library; }

void configure_window(GLFWwindow* window) {
    g_window = window;
    glfwRestoreWindow(window);
    glfwSetWindowSize(window, g_opt.width, g_opt.height);
    glfwSetWindowPos(window, 40, 40);
}

void start() {
    g_engine = ImGuiTestEngine_CreateContext();
    ImGuiTestEngineIO& io = ImGuiTestEngine_GetIO(g_engine);
    io.ConfigVerboseLevel        = ImGuiTestVerboseLevel_Info;
    io.ConfigVerboseLevelOnError = ImGuiTestVerboseLevel_Debug;
    io.ConfigRunSpeed            = (ImGuiTestRunSpeed)g_opt.speed;
    io.ConfigLogToTTY            = true;
    io.ConfigSavedSettings       = false;   // keep the app's imgui.ini free of engine state
    io.ConfigCaptureEnabled      = true;
    io.ConfigMouseDrawCursor     = g_opt.speed != 0;
    io.ConfigWatchdogWarning     = 60.0f;
    io.ConfigWatchdogKillTest    = 1800.0f; // GPU runs and the all-tabs sweep take minutes
    io.CoroutineFuncs            = Coroutine_ImplStdThread_GetInterface();
    io.ScreenCaptureFunc         = capture_gl;
    io.ScreenCaptureUserData     = g_window;
    g_results_path = (std::filesystem::path(g_opt.out_dir) / "results.xml").string();
    io.ExportResultsFilename = g_results_path.c_str();
    io.ExportResultsFormat   = ImGuiTestEngineExportFormat_JUnitXml;

    register_tests(g_engine);
    ImGuiTest* script = g_opt.script.empty() ? nullptr : register_script_test(g_engine, g_opt.script);

    ImGuiTestEngine_Start(g_engine, ImGui::GetCurrentContext());
    ImGuiTestEngine_InstallDefaultCrashHandler();

    if (g_opt.list) {
        ImVector<ImGuiTest*> tests;
        ImGuiTestEngine_GetTestList(g_engine, &tests);
        for (ImGuiTest* t : tests) std::printf("%s/%s\n", t->Category, t->Name);
        std::fflush(stdout);
        g_listed = true;
        return;
    }
    // The script goes first: with both options it is the setup of the compiled tests (load a system...).
    if (script) ImGuiTestEngine_QueueTest(g_engine, script, ImGuiTestRunFlags_RunFromCommandLine);
    if (!g_opt.filter.empty()) {
        // Filter syntax of the engine: comma list of name substrings, "^name" = prefix, "-name" = exclude.
        ImVector<ImGuiTest*> tests;
        ImGuiTestEngine_GetTestList(g_engine, &tests);
        for (ImGuiTest* t : tests)
            if (t != script && (g_opt.filter == "all" || ImGuiTestEngine_PassFilter(t, g_opt.filter.c_str())))
                ImGuiTestEngine_QueueTest(g_engine, t, ImGuiTestRunFlags_RunFromCommandLine);
    }
}

void draw_windows() {
    if (g_engine && g_opt.show_ui) ImGuiTestEngine_ShowTestEngineWindows(g_engine, nullptr);
}

bool wants_max_speed() {
    return g_engine && ImGuiTestEngine_GetIO(g_engine).IsRequestingMaxAppSpeed;
}

bool post_swap() {
    if (!g_engine) return false;
    ImGuiTestEngine_PostSwap(g_engine);
    ++g_frames;
    if (g_listed) return true;
    if (g_opt.keep_open) return false;
    const ImGuiTestEngineIO& io = ImGuiTestEngine_GetIO(g_engine);
    return g_frames > 2 && !io.IsRunningTests && ImGuiTestEngine_IsTestQueueEmpty(g_engine);
}

int stop() {
    if (!g_engine) return 0;
    ImGuiTestEngine_Stop(g_engine);   // also exports results.xml
    if (g_listed) return 0;
    ImGuiTestEngineResultSummary sum;
    ImGuiTestEngine_GetResultSummary(g_engine, &sum);
    ImVector<ImGuiTest*> tests;
    ImGuiTestEngine_GetTestList(g_engine, &tests);
    for (ImGuiTest* t : tests) {
        const ImGuiTestStatus s = t->Output.Status;
        if (s == ImGuiTestStatus_Unknown || s == ImGuiTestStatus_Queued) continue;
        std::printf("[ui-result] %-7s %s/%s\n",
                    s == ImGuiTestStatus_Success ? "PASS" : s == ImGuiTestStatus_Error ? "FAIL" : "OTHER",
                    t->Category, t->Name);
    }
    std::printf("[ui-summary] tested %d, passed %d, not run %d; results: %s\n",
                sum.CountTested, sum.CountSuccess, sum.CountInQueue, g_results_path.c_str());
    std::fflush(stdout);
    return (sum.CountTested > 0 && sum.CountTested == sum.CountSuccess && sum.CountInQueue == 0) ? 0 : 1;
}

void destroy() {
    if (g_engine) ImGuiTestEngine_DestroyContext(g_engine);
    g_engine = nullptr;
}

} // namespace ui_test
