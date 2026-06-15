#include "plot_renderer.h"
#include <vector>
#include <cmath>
#include "gpu_line_series_3d.h"
#include "plot_camera_3d.h"
// app_main.cpp - точка входа GUI-приложения: окно GLFW+OpenGL3, цикл ImGui.
// Связывает AppModel, OCR и платформенные колбэки.
#include "imgui.h"
#include "implot.h"
#include "implot3d.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad\glad.h>
#include <GLFW/glfw3.h>

#include "app_model.h"
#include "gui.h"
#include "system_library.h"
#include "ocr_client_win.h"   // OcrClient + b64encode

#include <windows.h>
#include <commdlg.h>          // GetOpenFileName
#include <memory>
#include <string>
#include <cstdio>



#pragma comment(lib, "opengl32.lib")

std::string exe_dir() {
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf);
    return p.substr(0, p.find_last_of("\\/"));
}

static const char* OCR_SCRIPT_NAME = "ocr_server.py";

// Ищем Python в таком порядке:
//   1) переменная окружения U_CUDA_PYTHON (полный путь к python.exe)
//   2) <project_dir>\.venv\Scripts\python.exe (локальное venv проекта)
//   3) "python" — CreateProcess сам найдёт через PATH
static std::string resolve_python_exe(const std::string& project_dir) {
    char env_buf[MAX_PATH];
    DWORD n = GetEnvironmentVariableA("U_CUDA_PYTHON", env_buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return std::string(env_buf, n);

    std::string venv_py = project_dir + ".venv\\Scripts\\python.exe";
    if (GetFileAttributesA(venv_py.c_str()) != INVALID_FILE_ATTRIBUTES)
        return venv_py;

    return "python";
}

static std::string pick_image_file_win() {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "Images\0*.png;*.jpg;*.jpeg;*.bmp\0All\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn)) return std::string(filename);
    return "";
}

static void set_clipboard_win(const std::string& text) {
    if (!OpenClipboard(nullptr)) return;
    EmptyClipboard();
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
    if (h) {
        memcpy(GlobalLock(h), text.c_str(), text.size() + 1);
        GlobalUnlock(h);
        SetClipboardData(CF_TEXT, h);
    }
    CloseClipboard();
}

int main() {
    // ocr_server.py и library\ копируются рядом с exe в Post-Build Event.
    std::string dir = exe_dir() + "\\";
    printf("Wait 10s, podnimaetsya server");
    std::string ocr_script = dir + "ocr_server.py";
    std::string library_dir = dir + "library";

    std::string python_exe = resolve_python_exe(dir);

    std::shared_ptr<OcrClient> ocr;
    std::string ocr_init_error;
    try {
        ocr = std::make_shared<OcrClient>(python_exe, ocr_script);
    }
    catch (const std::exception& e) {
        ocr_init_error = e.what();
    }

    OcrFn ocr_fn = [ocr](const std::vector<unsigned char>& png) -> std::string {
        if (!ocr) throw std::runtime_error("OCR not available");
        return ocr->recognize_base64(b64encode(png));
        };

    AppModel model(ocr_fn);
    model.app_mode = AppModel::AppMode::Analysis;
    model.start_phase_analysis();
    model.alphabet_text = "x,y,z,sigma,rho,beta";

    SystemLibrary library(library_dir);

    GuiCallbacks cb;
    cb.pick_image_file = pick_image_file_win;
    cb.set_clipboard_text = set_clipboard_win;

    // --- инициализация GLFW + OpenGL + ImGui ---
    if (!glfwInit()) return 1;

    // === ИЗМЕНЕНО: запрашиваем OpenGL 3.3 Core ===
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // === ИЗМЕНЕНО: GLSL версия 330 (под Core 3.3) ===
    const char* glsl_version = "#version 330";

    GLFWwindow* window = glfwCreateWindow(1100, 750,
        "Dynamical Systems Code Generator", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMaximizeWindow(window);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // === ДОБАВЛЕНО: инициализация GLAD сразу после MakeContextCurrent ===
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (!ocr_init_error.empty()) {
            ImGui::Begin("OCR warning");
            ImGui::TextColored(ImVec4(1, 0.6f, 0, 1),
                "OCR init failed: %s", ocr_init_error.c_str());
            ImGui::TextDisabled("LaTeX/Plain input still work.");
            ImGui::End();
        }

        draw_gui(model, library, cb);

        ImGui::Render();
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}