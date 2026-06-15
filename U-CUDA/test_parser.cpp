//// test_parser.cpp - проверка возможностей парсера и кодгена на реальных системах.
//// Самодостаточный: без OCR. Каждый кейс печатает разбор и генерирует код.
//#include "codegen.hpp"
//#include "sysparse.hpp"
//#include <iostream>
//#include <vector>
//#include <string>
//
//static int g_pass = 0, g_fail = 0;
//
//// Прогоняет одну систему: парсит, печатает разбор, генерирует все 4 схемы.
//static void run(const std::string& title,
//                const std::string& latex,
//                const std::vector<std::string>& alphabet,
//                bool expect_ok = true) {
//    std::cout << "======================================================\n";
//    std::cout << "TEST: " << title << "\n";
//    std::cout << "LaTeX: " << latex << "\n";
//    std::cout << "alphabet: "; for (auto& a : alphabet) std::cout << a << " "; std::cout << "\n";
//    std::cout << "------------------------------------------------------\n";
//    try {
//        System s = parse_system_from_latex(latex, alphabet);
//        std::cout << "vars:   "; for (auto& v : s.vars)   std::cout << v << " "; std::cout << "\n";
//        std::cout << "params: "; for (auto& p : s.params) std::cout << p << " "; std::cout << "\n";
//        std::cout << "rhs:\n"; for (auto& r : s.rhs) std::cout << "   " << r << "\n";
//
//        // Генерируем все схемы, проверяем что не падают
//        const char* names[] = {"Euler","EulerCromer","ExplicitMidpoint","RK4"};
//        Scheme schemes[] = {Scheme::Euler, Scheme::EulerCromer,
//                            Scheme::ExplicitMidpoint, Scheme::RK4};
//        bool all_ok = true;
//        for (int k = 0; k < 4; ++k) {
//            try {
//                std::string code = codegen_scheme(s, schemes[k]);
//                if (code.empty()) { all_ok = false; std::cout << "  [" << names[k] << "] EMPTY\n"; }
//            } catch (std::exception& e) {
//                all_ok = false;
//                std::cout << "  [" << names[k] << "] CODEGEN FAIL: " << e.what() << "\n";
//            }
//        }
//        // печатаем Euler как образец
//        std::cout << "--- Euler code ---\n" << codegen_scheme(s, Scheme::Euler);
//
//        if (all_ok == expect_ok) { std::cout << "RESULT: PASS\n"; ++g_pass; }
//        else { std::cout << "RESULT: FAIL (codegen)\n"; ++g_fail; }
//    } catch (std::exception& e) {
//        if (!expect_ok) { std::cout << "RESULT: PASS (expected parse error: " << e.what() << ")\n"; ++g_pass; }
//        else { std::cout << "RESULT: FAIL (parse: " << e.what() << ")\n"; ++g_fail; }
//    }
//    std::cout << "\n";
//}
//
//int main() {
//    // === 1. Базовые системы разной размерности ===
//    run("Lorenz 3D (plain mult)",
//        "\\dot{x} = \\sigma(y-x) \\\\ \\dot{y} = x(\\rho-z)-y \\\\ \\dot{z} = xy-\\beta z",
//        {"x","y","z","sigma","rho","beta"});
//
//    run("Logistic 1D",
//        "\\dot{x} = r x (1 - x)",
//        {"x","r"});
//
//    run("Rossler 3D",
//        "\\dot{x} = -y - z \\\\ \\dot{y} = x + a y \\\\ \\dot{z} = b + z(x - c)",
//        {"x","y","z","a","b","c"});
//
//    run("Hyperchaos 4D",
//        "\\dot{x} = a(y-x)+w \\\\ \\dot{y} = xz+b \\\\ \\dot{z} = c-xy \\\\ \\dot{w} = dy-z",
//        {"x","y","z","w","a","b","c","d"});
//
//    // === 2. Производные в разных формах ===
//    run("frac d/dt with mathrm",
//        "\\frac{\\mathrm{d}x}{\\mathrm{d}t} = -x \\\\ \\frac{\\mathrm{d}y}{\\mathrm{d}t} = -y",
//        {"x","y"});
//
//    run("prime notation",
//        "x' = -x \\\\ y' = -y",
//        {"x","y"});
//
//    run("dot greek without braces",
//        "\\dot\\theta = \\omega \\\\ \\dot\\omega = -\\sin\\theta",
//        {"theta","omega"});
//
//    // === 3. Нижние индексы ===
//    run("subscript variables 4D",
//        "\\dot{x}_{m} = a(y_{m}-x_{m})+c x_{m} z_{m}+w_{m} \\\\ "
//        "\\dot{y}_{m} = -d y_{m}-x_{m} z_{m} \\\\ "
//        "\\dot{z}_{m} = -b+x_{m} y_{m} \\\\ "
//        "\\dot{w}_{m} = -e y_{m}",
//        {"x_m","y_m","z_m","w_m","a","b","c","d","e"});
//
//    // === 4. Функции ===
//    run("sin without braces",
//        "\\dot{x} = \\sin x \\\\ \\dot{y} = \\cos y",
//        {"x","y"});
//
//    run("sin squared (power on function)",
//        "\\dot{x} = \\sin^{2} x \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    run("tanh nested arg",
//        "\\dot{x} = \\tanh(a x + b) \\\\ \\dot{y} = -y",
//        {"x","y","a","b"});
//
//    run("exp function",
//        "\\dot{x} = \\exp(-x) \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    // === 5. Дроби ===
//    run("simple fraction",
//        "\\dot{x} = \\frac{x}{1+x} \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    run("nested fraction",
//        "\\dot{x} = \\frac{x}{1 + \\frac{y}{2}} \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    // === 6. Модуль ===
//    run("absolute value pipes",
//        "\\dot{x} = -|x| \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    run("absolute value left/right",
//        "\\dot{x} = -\\left|x\\right| \\\\ \\dot{y} = -y",
//        {"x","y"});
//
//    // === 7. Степени ===
//    run("powers x^2 x^3",
//        "\\dot{x} = -x^{3} + y \\\\ \\dot{y} = x^{2} - y",
//        {"x","y"});
//
//    // === 8. Константы как параметры ===
//    run("e and pi as params",
//        "\\dot{x} = e x + pi \\\\ \\dot{y} = -y",
//        {"x","y","e","pi"});
//
//    // === 9. Неявное умножение и cdot ===
//    run("cdot and implicit mult mixed",
//        "\\dot{x} = a \\cdot x + b y \\\\ \\dot{y} = -y",
//        {"x","y","a","b"});
//
//    run("greek times greek",
//        "\\dot{x} = \\alpha \\beta x \\\\ \\dot{y} = -y",
//        {"x","y","alpha","beta"});
//
//    // === 10. Ожидаемые ошибки (negative tests) ===
//    run("no derivative in LHS (should fail)",
//        "\\sigma(y-x) \\\\ x(\\rho-z)-y",
//        {"x","y","sigma","rho"},
//        /*expect_ok=*/false);
//
//    // === Переменные как функции времени x(t) ===
//    run("variables as functions of time x(t)",
//        "\\begin{aligned} {\\frac{d x ( t )} {d t}} & {{}=a ( y ( t )-x ( t ) )} \\\\ "
//        "{\\frac{d y ( t )} {d t}} & {{}=( c-a ) x ( t )-x ( t ) z ( t )+c y ( t )} \\\\ "
//        "{\\frac{d z ( t )} {d t}} & {{}=x ( t ) y ( t )-b z ( t )} \\\\ \\end{aligned}",
//        {"x","y","z","a","b","c"});
//
//    // === Итог ===
//    std::cout << "======================================================\n";
//    std::cout << "TOTAL: " << g_pass << " passed, " << g_fail << " failed\n";
//    return g_fail == 0 ? 0 : 1;
//}


#include "plot_renderer.h"
#include <vector>
#include <cmath>
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

static const char* PYTHON_EXE =
"C:\\Users\\JASQUE\\AppData\\Local\\Programs\\Python\\Python310\\python.exe";
static const char* OCR_SCRIPT_NAME = "ocr_server.py";

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
#ifdef PROJECT_DIR
    std::string dir = PROJECT_DIR;
#else
    std::string dir = exe_dir() + "\\";
#endif
    std::string ocr_script = dir + "ocr_server.py";
    std::string library_dir = dir + "library";

    std::shared_ptr<OcrClient> ocr;
    std::string ocr_init_error;
    try {
        ocr = std::make_shared<OcrClient>(PYTHON_EXE, ocr_script);
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
    // отладочный вывод версии (можно убрать после проверки)
    printf("OpenGL: %s\n", (const char*)glGetString(GL_VERSION));
    printf("GLSL:   %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

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