#include "ucuda/plotting/plot_renderer.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <cmath>
#include "ucuda/plotting/gpu_line_series_3d.h"
#include "ucuda/plotting/plot_camera_3d.h"
#include "imgui.h"
#include "implot.h"
#include "implot3d.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ucuda/application/app_model.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/storage/session_io.h"
#include "ucuda/plotting/plot_axis.h"
#include "ucuda/ui/gui.h"
#include "ucuda/storage/system_library.h"
#include "ucuda/ocr/ocr_client_win.h"   // OcrClient + b64encode
#include "ucuda/plotting/screenshot.h"
#include "ucuda/platform/image_source.h"     // copy_framebuffer_rect_to_clipboard

#include "ucuda/platform/platform_paths.h"
#include "ucuda/platform/platform_posix.h"
#ifdef _WIN32
#include <shobjidl.h>         // IFileDialog: современный диалог файлов
#endif
#include <memory>
#include <string>
#include <cstdio>
#include <future>
#include <filesystem>
#include <algorithm>

#pragma comment(lib, "opengl32.lib")

std::string exe_dir() { return platform::executable_dir(); }

static const char* OCR_SCRIPT_NAME = "ocr_server.py";

static bool dir_exists(const std::string& p) {
    return std::filesystem::is_directory(p);
}

static bool file_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

// Серифная пара под LaTeX-подписи графиков (plot_text в plot_axis.h).
struct MathFontPaths {
    std::string roman;    // пусто = серифа нет, верстаем текущим UI-шрифтом
    std::string italic;   // пусто = курсива нет, переменные идут прямым
};

// Порядок поиска: env U_CUDA_FONTS, fonts/ рядом с exe (туда их кладёт
// PackRuntime), resources/fonts исходного дерева. В каталоге пробуем известные
// имена Computer Modern / Latin Modern — шрифт достаточно положить в fonts/.
static MathFontPaths resolve_math_font_paths(const std::string& exe_dir_with_sep) {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("U_CUDA_FONTS"); env && *env)
        dirs.push_back(std::string(env) + "/");
    dirs.push_back(exe_dir_with_sep + "fonts/");
#ifdef U_CUDA_RESOURCE_DIR
    dirs.push_back(std::string(U_CUDA_RESOURCE_DIR) + "/fonts/");
#endif

    static const char* kPairs[][2] = {
        { "cmunrm.otf",                   "cmunti.otf"                  },  // Computer Modern Unicode (лежит в репозитории)
        { "cmunrm.ttf",                   "cmunti.ttf"                  },  // он же, если взят из TTF-сборки
        { "lmroman10-regular.otf",        "lmroman10-italic.otf"        },  // Latin Modern
        { "LatinModernRoman-Regular.ttf", "LatinModernRoman-Italic.ttf" },
        { "texgyretermes-regular.otf",    "texgyretermes-italic.otf"    },  // Times-клон из TeX Gyre
    };
    for (const std::string& d : dirs)
        for (const auto& pair : kPairs)
            if (file_exists(d + pair[0]))
                return { d + pair[0],
                         file_exists(d + pair[1]) ? d + pair[1] : std::string() };

    // Times New Roman есть на любой Windows и покрывает латиницу, греческий и
    // кириллицу. Не Computer Modern, но засечки те же по духу.
    MathFontPaths p;
    if (file_exists("C:\\Windows\\Fonts\\times.ttf")) {
        p.roman = "C:\\Windows\\Fonts\\times.ttf";
        if (file_exists("C:\\Windows\\Fonts\\timesi.ttf"))
            p.italic = "C:\\Windows\\Fonts\\timesi.ttf";
    }
    return p;
}

// Paths returned by resolve_library_paths. `library` is the writable
// per-user copy (gitignored in dev, next-to-exe in deployed builds).
// `tmpl` is the git-tracked baseline that seeds `library` on first run.
struct LibraryPaths {
    std::string library;   // writable per-user library
    std::string tmpl;      // read-only baseline (git-tracked in dev)
};

// Resolve the pair {library, template}. The dev layout is detected by the
// presence of library_template/ (which is always in git), so both branches
// resolve deterministically even on a fresh clone where library/ does not
// exist yet — seed_library_if_missing will create it.
//
//   env U_CUDA_LIBRARY set (and points to an existing dir)
//       → library = <env>, tmpl = <env>_template (may or may not exist)
//   dev layout via vcxproj (exe in U-CUDA\x64\{Cfg}\)
//       → library_template lives at ..\..\library_template
//         library sits alongside at ..\..\library
//   dev layout via solution (exe in .\x64\{Cfg}\)
//       → ..\..\U-CUDA\library_template and ..\..\U-CUDA\library
//   deployed (exe with library_template next to it)
//       → <exe_dir>\library and <exe_dir>\library_template
static LibraryPaths resolve_library_paths(const std::string& exe_dir_with_sep) {
    LibraryPaths p;

    if (const char* env = std::getenv("U_CUDA_LIBRARY"); env && *env) {
        p.library = env;
        p.tmpl = p.library + "_template";
        return p;
    }
#ifdef U_CUDA_PROJECT_DIR
    p.tmpl = std::string(U_CUDA_RESOURCE_DIR) + "/library_template";
    p.library = std::string(U_CUDA_PROJECT_DIR) + "/U-CUDA/library";
    if (dir_exists(p.tmpl)) return p;
#endif

    p.library = exe_dir_with_sep + "library";
    p.tmpl    = exe_dir_with_sep + "library_template";
    return p;
}

// Copy the baseline into the writable library on first run. No-op if the
// library dir already exists (user's private data wins) or if the template
// is missing (deployed build without a template — start empty).
static void seed_library_if_missing(const std::string& library_dir,
                                    const std::string& template_dir) {
    if (dir_exists(library_dir)) return;
    if (!dir_exists(template_dir)) return;
    try {
        namespace fs = std::filesystem;
        fs::create_directories(library_dir);
        fs::copy(template_dir, library_dir,
                 fs::copy_options::recursive | fs::copy_options::skip_existing);
    } catch (...) {
        // Best-effort seed. If it fails, SystemLibrary will start empty and
        // the user can Add new / re-copy the template by hand.
    }
}

// Ищем Python в таком порядке:
//   1) переменная окружения U_CUDA_PYTHON (полный путь к python.exe)
//   2) <project_dir>\.venv\Scripts\python.exe (локальное venv проекта)
//   3) "python" — CreateProcess сам найдёт через PATH
static std::string resolve_python_exe(const std::string& project_dir) {
    if (const char* env = std::getenv("U_CUDA_PYTHON"); env && *env) return env;
#ifdef _WIN32
    const char* suffix = ".venv/Scripts/python.exe";
#else
    const char* suffix = ".venv/bin/python";
#endif
    std::vector<std::filesystem::path> roots = {project_dir};
#ifdef U_CUDA_PROJECT_DIR
    roots.emplace_back(std::filesystem::path(U_CUDA_PROJECT_DIR));
#endif
    for (const auto& root : roots) {
        auto p = root / suffix;
        if (std::filesystem::exists(p)) return p.string();
    }
#ifdef _WIN32
    return "python";
#else
    return "python3";
#endif
}

#ifdef _WIN32


// Современный диалог файлов (Common Item Dialog, COM) вместо GetOpenFileName /
// GetSaveFileName.
//
// Устаревшие общие диалоги загружают в процесс расширения оболочки, и сбойное
// расширение роняет хозяина. Симптом был именно такой: падение ВНУТРИ диалога с
// нарушением прав доступа, воспроизводившееся и на чистом main, где никакого
// нового кода нет. Общий диалог единственный на все три вызова, чтобы разойтись
// им было негде.
static std::string com_file_dialog(bool save, const COMDLG_FILTERSPEC* filters,
                                   UINT n_filters, const wchar_t* def_ext) {
    // GLFW уже инициализирует COM apartment-threaded, но зависеть от чужого
    // порядка инициализации не хочется. S_FALSE (уже инициализирован) тоже
    // требует парного CoUninitialize, поэтому проверяем SUCCEEDED, а не S_OK.
    const HRESULT ci = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    const bool need_uninit = SUCCEEDED(ci);

    std::string out;
    IFileDialog* dlg = nullptr;
    HRESULT hr = CoCreateInstance(save ? CLSID_FileSaveDialog : CLSID_FileOpenDialog,
                                  nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg));
    if (SUCCEEDED(hr) && dlg) {
        if (n_filters) dlg->SetFileTypes(n_filters, filters);
        if (def_ext)   dlg->SetDefaultExtension(def_ext);
        DWORD opts = 0;
        if (SUCCEEDED(dlg->GetOptions(&opts))) {
            // FORCEFILESYSTEM: не отдавать виртуальные элементы оболочки, для
            // которых нет пути на диске.
            dlg->SetOptions(opts | FOS_FORCEFILESYSTEM
                            | (save ? FOS_OVERWRITEPROMPT : FOS_FILEMUSTEXIST));
        }
        if (SUCCEEDED(dlg->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item)) && item) {
                PWSTR wpath = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wpath)) && wpath) {
                    // В ANSI, как отдавал прежний диалог: остальной код открывает
                    // файлы УЗКИМИ путями, и смена кодировки здесь сломала бы их.
                    const int n = WideCharToMultiByte(CP_ACP, 0, wpath, -1, nullptr, 0, nullptr, nullptr);
                    if (n > 1) {
                        out.resize((size_t)n - 1);
                        WideCharToMultiByte(CP_ACP, 0, wpath, -1, &out[0], n, nullptr, nullptr);
                    }
                    CoTaskMemFree(wpath);
                }
                item->Release();
            }
        }
        dlg->Release();
    }
    if (need_uninit) CoUninitialize();
    return out;
}

static std::string pick_image_file_win() {
    static const COMDLG_FILTERSPEC f[] = {
        { L"Images", L"*.png;*.jpg;*.jpeg;*.bmp" }, { L"All files", L"*.*" } };
    return com_file_dialog(false, f, 2, nullptr);
}

// Диалог сохранения для правого клика "Export data..." на диаграммах.
// Возвращает выбранный абсолютный путь либо пустую строку при отмене.
static std::string pick_save_csv_file_win() {
    static const COMDLG_FILTERSPEC f[] = {
        { L"CSV", L"*.csv" }, { L"All files", L"*.*" } };
    return com_file_dialog(true, f, 2, L"csv");
}

// Диалог открытия текстового файла: таблица коэффициентов композиции в
// конструкторе схем. Расширения перечислены те, в которых такие таблицы
// обычно и лежат; "All files" оставлен, потому что у выписанной из статьи
// таблицы расширения может не быть вовсе.
static std::string pick_open_text_file_win() {
    static const COMDLG_FILTERSPEC f[] = {
        { L"Coefficient tables", L"*.txt;*.dat;*.csv" }, { L"All files", L"*.*" } };
    return com_file_dialog(false, f, 2, nullptr);
}

// Диалог сохранения SPICE-netlist'а.
static std::string pick_save_netlist_file_win() {
    static const COMDLG_FILTERSPEC f[] = {
        { L"SPICE netlist", L"*.cir;*.net" }, { L"All files", L"*.*" } };
    return com_file_dialog(true, f, 2, L"cir");
}
#else
static std::string pick_file_linux(bool save, const char* filter, const char* ext) {
    std::vector<std::string> args = {"zenity", "--file-selection"};
    if (save) args.insert(args.end(), {"--save", "--confirm-overwrite", std::string("--filename=export.") + ext});
    args.push_back(std::string("--file-filter=") + filter);
    std::string out;
    try {
        if (platform::run(args, out) != 0) return {};
        if (!out.empty() && out.back() == '\n') out.pop_back();
        if (save && !out.empty() && std::filesystem::path(out).extension().empty()) out += std::string(".") + ext;
        return out;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return {}; }
}
static std::string pick_image_file_win() { return pick_file_linux(false, "Images | *.png *.jpg *.jpeg *.bmp", ""); }
static std::string pick_save_csv_file_win() { return pick_file_linux(true, "CSV | *.csv", "csv"); }
static std::string pick_open_text_file_win() { return pick_file_linux(false, "Coefficient tables | *.txt *.dat *.csv", ""); }
static std::string pick_save_netlist_file_win() { return pick_file_linux(true, "SPICE netlist | *.cir *.net", "cir"); }
#endif

static void set_clipboard_win(const std::string& text) {
    glfwSetClipboardString(nullptr, text.c_str());
}

int main() {
    // ocr_server.py, kernels\ и NVRTC-заголовки копируются рядом с exe в
    // Post-Build Event. library\ живёт по другой схеме: baseline лежит в
    // library_template\ (git-tracked), а рабочая library\ — per-user, в
    // gitignore. Первый запуск клонирует template → library; дальше UI
    // пишет только в library\, и git-status остаётся чистым.
    std::string dir = exe_dir() + "/";
    //printf("Wait 10s, podnimaetsya server");
    std::string ocr_script = dir + "ocr_server.py";
    LibraryPaths lib_paths = resolve_library_paths(dir);
    seed_library_if_missing(lib_paths.library, lib_paths.tmpl);
    const std::string& library_dir = lib_paths.library;
    std::string ctrl_error;
    if (!load_ctrl_library(library_dir + "/step_controllers.json", &ctrl_error))
        std::fprintf(stderr, "Step controllers: %s\n", ctrl_error.c_str());

    std::string python_exe = resolve_python_exe(dir);

    // Spawn the Python OCR server on a worker thread. Its ctor blocks ~10s
    // waiting for READY from the model; doing it inline would delay window
    // creation by the same amount. Exception (e.g. python missing) is stored
    // in the future and re-thrown by .get() on first OCR call.
    std::shared_future<std::shared_ptr<OcrClient>> ocr_future =
        std::async(std::launch::async,
            [python_exe, ocr_script]() -> std::shared_ptr<OcrClient> {
                return std::make_shared<OcrClient>(python_exe, ocr_script);
            }).share();

    OcrFn ocr_fn = [ocr_future](const std::vector<unsigned char>& png) -> std::string {
        // First call blocks until READY (or rethrows the ctor's exception).
        // start_ocr runs us on its own worker thread, so the UI stays alive.
        auto client = ocr_future.get();
        if (!client) throw std::runtime_error("OCR not available");
        return client->recognize_base64(b64encode(png));
        };

    // Populated lazily from ocr_future once it resolves; surfaced as a banner.
    std::string ocr_init_error;
    bool ocr_init_checked = false;

    AppModel model(ocr_fn);
    model.app_mode = AppModel::AppMode::Analysis;
    model.start_phase_analysis();
    model.vars_text   = "x, y, z";
    model.params_text = "sigma, rho, beta";

    SystemLibrary library(library_dir);

    GuiCallbacks cb;
    cb.pick_image_file = pick_image_file_win;
    cb.set_clipboard_text = set_clipboard_win;
    cb.pick_save_file_csv = pick_save_csv_file_win;
    cb.pick_save_file_netlist = pick_save_netlist_file_win;
    cb.pick_open_file_text = pick_open_text_file_win;

    // Право-клик "Copy image to clipboard" на любой диаграмме стекается
    // сюда (см. plot_axis.h set_screenshot_request_sink).
    // Здесь только фиксируем rect + отсрочку; сам захват — в главном цикле
    // ниже, после того как right-click popup реально закроется.
    set_screenshot_request_sink([&model](ImVec2 mn, ImVec2 mx) {
        model.pending_screenshot.active = true;
        model.pending_screenshot.frames_left = 2; // popup закрывается на след. кадре, +1 запас
        model.pending_screenshot.min_x = mn.x; model.pending_screenshot.min_y = mn.y;
        model.pending_screenshot.max_x = mx.x; model.pending_screenshot.max_y = mx.y;
    });

    // инициализация GLFW + OpenGL + ImGui
    if (!glfwInit()) return 1;

    // ИЗМЕНЕНО: запрашиваем OpenGL 3.3 Core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // ИЗМЕНЕНО: GLSL версия 330 (под Core 3.3)
    const char* glsl_version = "#version 330";

    GLFWwindow* window = glfwCreateWindow(1100, 750,
        "U-CUDA", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMaximizeWindow(window);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ДОБАВЛЕНО: инициализация GLAD сразу после MakeContextCurrent
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
    // Раскладка окон — рядом с exe, как и _app_config.json. По умолчанию ImGui
    // берёт imgui.ini из текущей папки, а она зависит от способа запуска (у
    // сгенерированного .sln это каталог проекта), и раскладка «терялась».
    // Строка обязана жить до DestroyContext: ImGui хранит только указатель.
    static const std::string imgui_ini_path = dir + "imgui.ini";
    ImGui::GetIO().IniFilename = imgui_ini_path.c_str();
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    // Тема выбирается чуть ниже после load_app_config (model.dark_theme может
    // приехать из настроек). Здесь только дефолтная инициализация.
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // UI scale: auto-detect через GLFW + override из _app_config.json
    {
        float xs = 1.0f, ys = 1.0f;
        if (GLFWmonitor* mon = glfwGetPrimaryMonitor())
            glfwGetMonitorContentScale(mon, &xs, &ys);
        model.ui_scale_auto = (xs > ys) ? xs : ys;
        if (model.ui_scale_auto < 0.5f) model.ui_scale_auto = 1.0f;
    }
    AppConfig app_cfg;
    const bool have_cfg = load_app_config(dir, app_cfg);
    // Набор включённых карт применяем ДО чтения остального конфига: id ниже
    // могут указывать на slanCM, и пикеру список нужен готовым к первому кадру.
    // Вне ветки have_cfg — при первом запуске конфига ещё нет, а дефолтный
    // набор нужен всё равно.
    model.slancm_enabled = (have_cfg && !app_cfg.slancm_enabled.empty())
                         ? app_cfg.slancm_enabled : default_enabled_slancm();
    set_enabled_slancm(model.slancm_enabled);
    if (have_cfg) {
        if (app_cfg.ui_scale_override > 0.0f)
            model.ui_scale_override = app_cfg.ui_scale_override;
        model.use_builtin_font = app_cfg.use_builtin_font;
        model.plot_math_font   = app_cfg.plot_math_font;
        model.plot_font_scale  = app_cfg.plot_font_scale;
        // colormap_id_or валидирует и мигрирует легаси 0..8; на невалидном
        // значении остаётся дефолт модели — как и раньше при непрошедшей проверке.
        model.heatmap_colormap = colormap_id_or(app_cfg.heatmap_colormap, model.heatmap_colormap);
        model.basins_colormap        = colormap_id_or(app_cfg.basins_colormap,        model.basins_colormap);
        model.basins_avgpk_colormap  = colormap_id_or(app_cfg.basins_avgpk_colormap,  model.basins_avgpk_colormap);
        model.basins_avgint_colormap = colormap_id_or(app_cfg.basins_avgint_colormap, model.basins_avgint_colormap);
        model.basins_states_colormap = colormap_id_or(app_cfg.basins_states_colormap, model.basins_states_colormap);
        if (app_cfg.tick_precision >= 2 && app_cfg.tick_precision <= 10)
            model.tick_precision = app_cfg.tick_precision;
        model.dark_theme = app_cfg.dark_theme;

        // Маски видимости из Settings: какие вкладки показывать в верхнем ряду
        // и какие встроенные схемы предлагать в "Schemes to generate".
        // Ставим ДО восстановления last_app_mode — draw_gui уведёт в Library,
        // если сохранённая вкладка теперь скрыта.
        model.hidden_tabs    = app_cfg.hidden_tabs;
        model.hidden_schemes = app_cfg.hidden_schemes;
        model.hidden_systems = app_cfg.hidden_systems;

        // Restore last-used AppMode. Clamp to the valid enum range so a
        // future rename/reorder of AppMode doesn't crash the app on an old
        // config file. Счётчик живёт в самом enum'е (AppModel::kAppModeCount):
        // литерал здесь отстал на Order и гасил его восстановление.
        if (app_cfg.last_app_mode >= 0 && app_cfg.last_app_mode < AppModel::kAppModeCount)
            model.app_mode = (AppModel::AppMode)app_cfg.last_app_mode;

        // Restore last-loaded system: if the name still exists in the library,
        // apply the switch (loads record + inits the current tab's session,
        // matching what the top-bar combo does). Silently skip if the file
        // was renamed/deleted between sessions — user picks manually.
        if (!app_cfg.last_system_name.empty()) {
            auto names = library.list();
            if (std::find(names.begin(), names.end(), app_cfg.last_system_name) != names.end())
                apply_system_switch(model, library, app_cfg.last_system_name);
        }

        // Peak-knobs. Clamp — страховка от правленого вручную конфига:
        // max_amount_of_peaks задаёт размер device-буферов пиков и интервалов
        // на точку свипа, поэтому абсурдное значение съедает всю память GPU.
        model.peak = app_cfg.peak;
        clamp_peak_config(model.peak);
        model.nvrtc_fmad = app_cfg.nvrtc_fmad;
        model.nvrtc_rdc  = app_cfg.nvrtc_rdc;
        model.gpu_block_size = clamp_gpu_block_size(app_cfg.gpu_block_size);
    }
    // Пушим до первого Run: движок создаётся лениво, а конфиг глобальный и
    // должен быть актуален уже на самой первой компиляции NVRTC.
    set_peak_config(model.peak);
    set_nvrtc_fmad(model.nvrtc_fmad);
    set_nvrtc_rdc(model.nvrtc_rdc);
    set_gpu_block_size(model.gpu_block_size);
    model.sync_peak_text();   // текстовые буферы полей Settings — из свежих значений
    set_tick_precision(model.tick_precision);
    // Клампится внутри: конфиг мог быть правлен руками.
    set_plot_font_scale(model.plot_font_scale);
    model.plot_font_scale = plot_font_scale();

    // applied = -1 / opposite → форсируем apply на первом кадре.
    float applied_ui_scale     = -1.0f;
    bool  applied_font_builtin = !model.use_builtin_font;  // !=, форсирует первый apply
    bool  applied_dark_theme   = !model.dark_theme;        // != — то же самое для темы
    bool  applied_math_font    = !model.plot_math_font;    // != — и для LaTeX-подписей
    const MathFontPaths math_fonts = resolve_math_font_paths(dir);
    auto apply_ui_scale = [&](float scale) {
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->Clear();

        ImFont* loaded = nullptr;
        if (!model.use_builtin_font) {
            // Segoe UI: 15px baseline, scale-аппливается через SizePixels параметр.
            // GetGlyphRangesCyrillic() — на случай если в UI появятся русские строки.
            ImFontConfig cfg;
            cfg.OversampleH = 2;   // TTF выглядит лучше с oversample 2 (типичный
            cfg.OversampleV = 1;   // ImGui-дефолт для AddFontFromFileTTF).
            cfg.PixelSnapH  = false;
            const char* font_candidates[] = {
#ifdef _WIN32
                "C:/Windows/Fonts/segoeui.ttf"
#else
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/gnu-free/FreeSans.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#endif
            };
            for (const char* path : font_candidates) {
                if (!std::filesystem::is_regular_file(path)) continue;
                loaded = io.Fonts->AddFontFromFileTTF(path, 15.0f * scale, &cfg,
                                                    io.Fonts->GetGlyphRangesCyrillic());
                if (loaded) break;
            }
        }
        if (!loaded) {
            // ProggyClean — либо явно выбран в Settings, либо segoeui.ttf не нашли.
            // Values тут совпадают с внутренним default'ом ImGui::AddFontDefault(nullptr).
            ImFontConfig cfg;
            cfg.OversampleH  = 1;
            cfg.OversampleV  = 1;
            cfg.PixelSnapH   = true;
            cfg.SizePixels   = 13.0f * scale;
            cfg.GlyphOffset.y = 1.0f * (float)(int)(cfg.SizePixels / 13.0f);
            io.Fonts->AddFontDefault(&cfg);
        }

        // LaTeX-подписи графиков: серифная пара (прямой + курсив) в том же
        // атласе, что и UI-шрифт. Кегль здесь номинальный — plot_text рисует
        // размером текущего шрифта, а 1.92 печёт глифы под запрошенный размер.
        ImFont* math_roman  = nullptr;
        ImFont* math_italic = nullptr;
        if (model.plot_math_font && !math_fonts.roman.empty()) {
            ImFontConfig cfg;
            // Oversample 3 против 2 у UI-шрифта: внутри прогона ImGui двигает
            // глифы дробными advance'ами, а тонкие засечки Times/CM на 15px
            // мылятся от этого заметнее, чем гротеск.
            cfg.OversampleH = 3;
            cfg.OversampleV = 1;
            cfg.PixelSnapH  = false;
            math_roman = io.Fonts->AddFontFromFileTTF(
                math_fonts.roman.c_str(), 15.0f * scale, &cfg);
            if (!math_fonts.italic.empty())
                math_italic = io.Fonts->AddFontFromFileTTF(
                    math_fonts.italic.c_str(), 15.0f * scale, &cfg);
        }
        io.Fonts->Build();
        // Атлас только что пересобран — старые ImFont* мертвы, пушим новые до
        // ближайшего NewFrame.
        set_plot_math_fonts(math_roman, math_italic);
        set_plot_math_enabled(model.plot_math_font);
        // В ImGui 1.92+ backend сам обновит fonts texture при следующем NewFrame
        // через ImTextureData::Status — ручных Destroy/Create не нужно.

        // Стиль каждый раз пересоздаём от дефолтов + выбранной темы. Раньше
        // кэшировали `base_style` после StyleColorsDark — но это блокировало
        // переключение Dark/Light в Settings, потому что новая палитра не
        // попадала в base_style. Перестроение дешёвое (memcpy + цвет-таблица).
        ImGui::GetStyle() = ImGuiStyle{};
        if (model.dark_theme) ImGui::StyleColorsDark();
        else                  ImGui::StyleColorsLight();
        ImGui::GetStyle().ScaleAllSizes(scale);
        // Плот-палитра (текст/сетка/рамки/фон FBO) живёт отдельно от ImGui
        // style — её читают plot_view_2d/3d, heatmap_view, plot_legend и
        // colorbar в gui.cpp. Тригерится одновременно со сменой темы.
        set_plot_light_theme(!model.dark_theme);
    };

    while (!glfwWindowShouldClose(window)) {
        // Все сессии, чей счёт идёт на GPU. Custom-вкладка держит свои расчёты
        // в ОТДЕЛЬНЫХ под-сессиях (custom_session.bif_session и т.д.), а не в
        // model.bifurcation_session, поэтому её надо спрашивать отдельно:
        // иначе при запуске из Custom троттлинг ниже не включался, окно
        // рисовалось на полных ~60 FPS и отнимало GPU у ядра — один и тот же
        // расчёт из Custom и из Parametric заметно расходился по времени.
        // Тот же недосмотр был у 1D DFT и Fast Synchro: их сессии тоже
        // появились после того, как троттлинг был написан.
        bool compute_in_flight =
            model.bifurcation_session.in_flight ||
            model.lle_session.in_flight ||
            model.ls_session.in_flight ||
            model.metrics_session.in_flight ||
            model.basins_session.in_flight ||
            model.dft1d_session.in_flight ||
            model.fastsync_session.in_flight ||
            model.custom_session.any_in_flight();
        // During heavy compute throttle the main loop: glfwWaitEventsTimeout
        // blocks the thread up to 100 ms unless an input event arrives. Drops
        // render rate from ~60 FPS to ~10 FPS, freeing GPU and CPU for CUDA
        // work. Empirically captures most of the win; lower rates (500 ms /
        // 2 FPS) save only ~1 s extra on a 90 s job and make the progress bar
        // visibly jerky. Cancel button reacts immediately (input events wake
        // the thread).
        if (compute_in_flight) glfwWaitEventsTimeout(0.1);
        else                   glfwPollEvents();

        // Применяем масштаб/шрифт ДО NewFrame — иначе пересоздание fonts texture
        // в середине кадра валит RenderDrawData.
        float wanted_scale = model.effective_ui_scale();
        if (wanted_scale != applied_ui_scale ||
            model.use_builtin_font != applied_font_builtin ||
            model.dark_theme       != applied_dark_theme   ||
            model.plot_math_font   != applied_math_font) {
            apply_ui_scale(wanted_scale);
            applied_ui_scale     = wanted_scale;
            applied_font_builtin = model.use_builtin_font;
            applied_dark_theme   = model.dark_theme;
            applied_math_font    = model.plot_math_font;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Once the async OCR ctor finishes, peek at the result so we can show
        // a warning banner if it failed. .get() is safe to call repeatedly on
        // a shared_future; subsequent OCR-invocations from ocr_fn will hit the
        // cached value.
        if (!ocr_init_checked &&
            ocr_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            try { ocr_future.get(); }
            catch (const std::exception& e) { ocr_init_error = e.what(); }
            ocr_init_checked = true;
        }

        if (!ocr_init_error.empty()) {
            ImGui::Begin("OCR warning");
            ImGui::TextColored(ImVec4(1, 0.6f, 0, 1),
                "OCR init failed: %s", ocr_init_error.c_str());
            ImGui::TextDisabled("LaTeX/Plain input still work.");
            ImGui::End();
        }

        model.tick(library);
        draw_gui(model, library, cb);

        ImGui::Render();
        int w, h; glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Отложенный "copy diagram to clipboard" (см. set_screenshot_request_sink
        // выше) — ждём, пока right-click popup, из которого могли вызвать
        // запрос, реально закроется (ImGui закрывает popup только со
        // СЛЕДУЮЩЕГО кадра), иначе он попадёт в захваченную картинку.
        // DPI 1:1 — экранные координаты ImGui берутся как есть, без
        // домножения на io.DisplayFramebufferScale.
        if (model.pending_screenshot.active) {
            if (model.pending_screenshot.frames_left > 0) {
                model.pending_screenshot.frames_left--;
            } else {
                const auto& ps = model.pending_screenshot;
                copy_framebuffer_rect_to_clipboard(
                    (int)ps.min_x, (int)ps.min_y, (int)ps.max_x, (int)ps.max_y);
                model.pending_screenshot.active = false;
            }
        }

        // Действия, которым нужен платформенный модальный диалог: только вне
        // кадра ImGui, иначе его цикл сообщений вызовет рекурсивную отрисовку.
        gui_process_deferred(model, cb);

        glfwSwapBuffers(window);
    }

    release_gui_resources();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
