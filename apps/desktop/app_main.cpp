#include "ucuda/plotting/plot_renderer.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <cmath>
#include "ucuda/application/compute_services.h"   // apply_* — глобальные настройки GPU
#include "ucuda/plotting/gpu_line_series_3d.h"
#include "ucuda/plotting/plot_camera_3d.h"
#include "imgui.h"
#include "imgui_internal.h"   // GImGui->Windows: подкладка узора темы Military
#include "implot.h"
#include "implot3d.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "ucuda/application/app_model.h"
#include "ucuda/storage/app_config.h"
#include "ucuda/application/session_persistence.h"
#include "ucuda/plotting/plot_axis.h"
#include "ucuda/plotting/plot_legend.h"
#include "ucuda/ui/gui.h"
#include "ucuda/ui/gui_tree_state.h"
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

// Бинарный PPM (P6, maxval 255) → GL-текстура (повтор делается раскладкой
// плитками в inject_window_backdrops, не режимом GL_REPEAT). Формат выбран ради
// загрузчика в десяток строк: декодера PNG/JPEG в проекте нет. 0 — файла нет
// или он битый; тема тогда остаётся без узора.
static GLuint load_ppm_texture(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    int w = 0, h = 0, maxval = 0;
    const bool ok = std::fscanf(f, "P6 %d %d %d", &w, &h, &maxval) == 3 &&
                    w > 0 && h > 0 && w <= 4096 && h <= 4096 && maxval == 255 &&
                    std::fgetc(f) != EOF;   // один пробельный символ после заголовка
    std::vector<unsigned char> px(ok ? (size_t)w * h * 3 : 0);
    const bool read = ok && std::fread(px.data(), 1, px.size(), f) == px.size();
    std::fclose(f);
    if (!read) return 0;
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

// Палитры камуфляжных тем, индекс — AppSettings::CamoTheme. Задаются
// несколько опорных цветов, остальные цвета ImGui выводятся из них в
// apply_camo_palette. Фон окон — узор из texture под полупрозрачной вуалью
// veil; без файла текстуры вуаль становится сплошной.
struct CamoPalette {
    const char* texture;   // файл в <exe>/themes/
    bool  light;           // светлая основа (тёмный текст) или тёмная
    float veil[4];         // WindowBg поверх узора
    float popup[3];        // всплывающие окна, меню, подсказки
    float text[3];
    float frame[3];        // поля ввода
    float accent[3];       // кнопки, выделение, вкладки
    float accent_dark[3];  // нажатое, активный заголовок окна
    float title[3];        // заголовки окон
    float plot[3];         // фон графиков
};
static const CamoPalette kCamoPalettes[AppSettings::kCamoCount] = {
    {},
    // Multicam: песок и хаки, оливковый акцент, тёмно-коричневый текст.
    { "multicam.ppm", true,
      {0.88f, 0.84f, 0.74f, 0.42f}, {0.87f, 0.83f, 0.73f}, {0.12f, 0.10f, 0.06f},
      {0.93f, 0.90f, 0.82f}, {0.42f, 0.48f, 0.28f}, {0.33f, 0.39f, 0.20f},
      {0.55f, 0.52f, 0.40f}, {0.93f, 0.91f, 0.85f} },
    // DPM (лесной): тёмная основа, светлый хаки текст, оливковый акцент.
    { "dpm.ppm", false,
      {0.09f, 0.10f, 0.06f, 0.55f}, {0.13f, 0.15f, 0.10f}, {0.91f, 0.89f, 0.78f},
      {0.20f, 0.22f, 0.14f}, {0.47f, 0.55f, 0.27f}, {0.36f, 0.27f, 0.16f},
      {0.24f, 0.27f, 0.16f}, {0.08f, 0.09f, 0.06f} },
    // DDPM (пустынный): светлый песок, коричневый акцент, тёмно-коричневый текст.
    { "ddpm.ppm", true,
      {0.92f, 0.86f, 0.72f, 0.42f}, {0.90f, 0.84f, 0.71f}, {0.17f, 0.10f, 0.05f},
      {0.95f, 0.91f, 0.81f}, {0.60f, 0.41f, 0.24f}, {0.46f, 0.30f, 0.16f},
      {0.67f, 0.55f, 0.39f}, {0.95f, 0.92f, 0.85f} },
    // Flora (ВСР-98): светло-оливковая основа, зелёный акцент, коричневое нажатое,
    // тёмно-зелёный текст.
    { "flora.ppm", true,
      {0.82f, 0.86f, 0.72f, 0.45f}, {0.81f, 0.85f, 0.72f}, {0.09f, 0.13f, 0.07f},
      {0.91f, 0.93f, 0.84f}, {0.32f, 0.44f, 0.27f}, {0.43f, 0.28f, 0.20f},
      {0.47f, 0.54f, 0.37f}, {0.92f, 0.94f, 0.88f} },
};

static void apply_camo_palette(ImVec4* c, const CamoPalette& p, bool has_texture) {
    auto v = [](const float* x, float a) { return ImVec4(x[0], x[1], x[2], a); };
    auto lerp = [](ImVec4 a, ImVec4 b, float t) {
        return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
    };
    const ImVec4 popup = v(p.popup, 1.0f), text = v(p.text, 1.0f), frame = v(p.frame, 1.0f);
    const ImVec4 acc = v(p.accent, 1.0f), accd = v(p.accent_dark, 1.0f), title = v(p.title, 1.0f);
    auto alpha = [](ImVec4 x, float a) { x.w = a; return x; };

    c[ImGuiCol_WindowBg]         = ImVec4(p.veil[0], p.veil[1], p.veil[2], has_texture ? p.veil[3] : 1.0f);
    c[ImGuiCol_DockingEmptyBg]   = alpha(c[ImGuiCol_WindowBg], has_texture ? p.veil[3] * 0.7f : 1.0f);
    c[ImGuiCol_ChildBg]          = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg]          = alpha(popup, 0.98f);
    c[ImGuiCol_MenuBarBg]        = alpha(lerp(title, popup, 0.35f), 0.85f);
    c[ImGuiCol_TitleBg]          = title;
    c[ImGuiCol_TitleBgActive]    = accd;
    c[ImGuiCol_TitleBgCollapsed] = alpha(title, 0.80f);
    c[ImGuiCol_Text]             = text;
    c[ImGuiCol_TextDisabled]     = lerp(text, popup, 0.45f);
    c[ImGuiCol_Border]           = alpha(lerp(text, popup, 0.55f), 0.60f);
    c[ImGuiCol_Separator]        = alpha(lerp(text, popup, 0.50f), 0.60f);
    c[ImGuiCol_SeparatorHovered] = alpha(acc, 0.80f);
    c[ImGuiCol_SeparatorActive]  = accd;
    c[ImGuiCol_FrameBg]          = alpha(frame, 0.85f);
    c[ImGuiCol_FrameBgHovered]   = alpha(lerp(frame, acc, 0.25f), 0.95f);
    c[ImGuiCol_FrameBgActive]    = lerp(frame, acc, 0.40f);
    c[ImGuiCol_Button]           = alpha(acc, 0.55f);
    c[ImGuiCol_ButtonHovered]    = alpha(acc, 0.85f);
    c[ImGuiCol_ButtonActive]     = accd;
    c[ImGuiCol_Header]           = alpha(acc, 0.40f);
    c[ImGuiCol_HeaderHovered]    = alpha(acc, 0.70f);
    c[ImGuiCol_HeaderActive]     = alpha(accd, 0.90f);
    // Галочка контрастна к полю: на светлом поле темнее акцента, на тёмном светлее.
    c[ImGuiCol_CheckMark]        = p.light ? lerp(accd, ImVec4(0, 0, 0, 1), 0.35f)
                                           : lerp(acc, ImVec4(1, 1, 1, 1), 0.35f);
    c[ImGuiCol_SliderGrab]       = alpha(acc, 0.90f);
    c[ImGuiCol_SliderGrabActive] = p.light ? accd : lerp(acc, ImVec4(1, 1, 1, 1), 0.25f);
    c[ImGuiCol_Tab]              = alpha(lerp(title, popup, 0.40f), 0.90f);
    c[ImGuiCol_TabHovered]       = acc;
    c[ImGuiCol_TabSelected]      = lerp(acc, popup, 0.30f);
    c[ImGuiCol_TabSelectedOverline] = accd;
    c[ImGuiCol_TabDimmed]        = alpha(lerp(title, popup, 0.60f), 0.90f);
    c[ImGuiCol_TabDimmedSelected] = lerp(acc, popup, 0.50f);
    c[ImGuiCol_ScrollbarBg]      = alpha(popup, 0.50f);
    c[ImGuiCol_ScrollbarGrab]    = alpha(lerp(title, text, 0.15f), 0.80f);
    c[ImGuiCol_ScrollbarGrabHovered] = alpha(lerp(title, text, 0.30f), 0.90f);
    c[ImGuiCol_ScrollbarGrabActive]  = lerp(title, text, 0.45f);
    c[ImGuiCol_ResizeGrip]       = alpha(acc, 0.25f);
    c[ImGuiCol_ResizeGripHovered] = alpha(acc, 0.67f);
    c[ImGuiCol_ResizeGripActive] = alpha(accd, 0.95f);
    c[ImGuiCol_TableHeaderBg]    = lerp(title, popup, 0.40f);
    c[ImGuiCol_TableBorderStrong] = lerp(text, popup, 0.55f);
    c[ImGuiCol_TableBorderLight] = lerp(text, popup, 0.75f);
    c[ImGuiCol_TextSelectedBg]   = alpha(acc, 0.35f);
    c[ImGuiCol_DockingPreview]   = alpha(acc, 0.60f);
    c[ImGuiCol_NavCursor]        = accd;
    c[ImGuiCol_TextLink]         = p.light ? lerp(accd, ImVec4(0, 0, 0, 1), 0.2f) : lerp(acc, ImVec4(1, 1, 1, 1), 0.3f);
    c[ImGuiCol_DragDropTarget]   = accd;
}

// Подкладка узора под окна (камуфляжные темы). ImGui рисует фон окна внутри Begin,
// и вставить картинку между фоном и содержимым без правки всех 36 вызовов Begin
// нельзя. Поэтому после Render перед списком отрисовки каждого корневого окна
// вставляется отдельный список с узором на всю его площадь, а WindowBg темы
// полупрозрачный и ложится сверху вуалью. Узор непрозрачен, так что окно,
// лежащее поверх другого, его не просвечивает; UV берутся от экранных
// координат, поэтому рисунок непрерывен через границы окон. Дочерние окна
// (ChildBg прозрачный), всплывающие окна и подсказки (PopupBg сплошной)
// пропускаются.
static void inject_window_backdrops(ImDrawData* dd, GLuint tex, float tile_px,
                                    std::vector<std::unique_ptr<ImDrawList>>& pool) {
    if (!dd || dd->CmdListsCount == 0 || tile_px <= 0.0f) return;
    ImGuiContext& g = *GImGui;
    ImVector<ImDrawList*> out;
    out.reserve(dd->CmdListsCount * 2);
    size_t used = 0;
    const ImTextureRef ref((ImTextureID)(intptr_t)tex);
    for (ImDrawList* list : dd->CmdLists) {
        const ImGuiWindow* win = nullptr;
        for (ImGuiWindow* w : g.Windows)
            if (w->DrawList == list) { win = w; break; }
        const ImGuiWindowFlags skip = ImGuiWindowFlags_ChildWindow | ImGuiWindowFlags_Popup |
                                      ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoBackground;
        if (win && win->Active && !win->Hidden && !(win->Flags & skip) &&
            win->Size.x > 0.0f && win->Size.y > 0.0f) {
            if (used == pool.size())
                pool.push_back(std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData()));
            ImDrawList* bd = pool[used++].get();
            bd->_ResetForNewFrame();
            bd->PushClipRect(dd->DisplayPos,
                             ImVec2(dd->DisplayPos.x + dd->DisplaySize.x, dd->DisplayPos.y + dd->DisplaySize.y));
            // Раскладка плитками по экранной сетке с шагом tile_px, UV каждой
            // плитки внутри [0,1]. На GL_REPEAT полагаться нельзя: бэкенд ImGui
            // ставит текстурам CLAMP_TO_EDGE, и за первой плиткой узор
            // растягивался полосами.
            const ImVec2 a = win->Pos;
            const ImVec2 b(win->Pos.x + win->Size.x, win->Pos.y + win->Size.y);
            for (float ty = std::floor(a.y / tile_px) * tile_px; ty < b.y; ty += tile_px) {
                for (float tx = std::floor(a.x / tile_px) * tile_px; tx < b.x; tx += tile_px) {
                    const ImVec2 p0(std::max(a.x, tx), std::max(a.y, ty));
                    const ImVec2 p1(std::min(b.x, tx + tile_px), std::min(b.y, ty + tile_px));
                    if (p1.x <= p0.x || p1.y <= p0.y) continue;
                    bd->AddImage(ref, p0, p1,
                                 ImVec2((p0.x - tx) / tile_px, (p0.y - ty) / tile_px),
                                 ImVec2((p1.x - tx) / tile_px, (p1.y - ty) / tile_px));
                }
            }
            bd->PopClipRect();
            out.push_back(bd);
            dd->TotalVtxCount += bd->VtxBuffer.Size;
            dd->TotalIdxCount += bd->IdxBuffer.Size;
        }
        out.push_back(list);
    }
    dd->CmdLists.swap(out);
    dd->CmdListsCount = dd->CmdLists.Size;
}

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

// Диалог открытия схемы (вкладка DAE): SPICE-netlist или схема KiCad.
static std::string pick_open_circuit_file_win() {
    static const COMDLG_FILTERSPEC f[] = {
        { L"Circuits", L"*.cir;*.net;*.sp;*.spice;*.kicad_sch" }, { L"All files", L"*.*" } };
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
    cb.pick_open_file_circuit = pick_open_circuit_file_win;

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
    tree_state::install();
    plot_install_settings_handler();   // вид диаграмм по окнам ([UcudaPlot] в imgui.ini)
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
    // Маски схем и методов ДАУ: без ключа в конфиге (или без конфига) — набор свежей установки.
    model.hidden_schemes     = app_cfg.has_hidden_schemes ? app_cfg.hidden_schemes : default_hidden_schemes();
    model.hidden_dae_methods = app_cfg.has_hidden_dae_methods ? app_cfg.hidden_dae_methods
                                                              : default_hidden_dae_methods();
    if (have_cfg) {
        if (app_cfg.ui_scale_override > 0.0f)
            model.ui_scale_override = app_cfg.ui_scale_override;
        model.use_builtin_font = app_cfg.use_builtin_font;
        model.plot_math_font   = app_cfg.plot_math_font;
        model.field_latex      = app_cfg.field_latex;
        model.show_alg_ic      = app_cfg.show_alg_ic;
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
        model.series_palette = app_cfg.series_palette;
        model.dark_theme = app_cfg.dark_theme;
        model.contrast_theme = app_cfg.contrast_theme && app_cfg.dark_theme;
        model.camo_theme = app_cfg.dark_theme ? 0 : app_cfg.camo_theme;
        model.theme_colors = AppSettings::theme_colors_from_string(app_cfg.theme_colors);

        // Маски видимости из Settings: какие вкладки показывать в верхнем ряду
        // и какие встроенные схемы предлагать в "Schemes to generate".
        // Ставим ДО восстановления last_app_mode — draw_gui уведёт в Library,
        // если сохранённая вкладка теперь скрыта.
        model.hidden_tabs    = app_cfg.hidden_tabs;
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
    apply_peak_config(model.peak);
    apply_nvrtc_fmad(model.nvrtc_fmad);
    apply_nvrtc_rdc(model.nvrtc_rdc);
    apply_gpu_block_size(model.gpu_block_size);
    model.sync_peak_text();   // текстовые буферы полей Settings — из свежих значений
    set_tick_precision(model.tick_precision);
    set_series_palette(model.series_palette);
    // Клампится внутри: конфиг мог быть правлен руками.
    set_plot_font_scale(model.plot_font_scale);
    model.plot_font_scale = plot_font_scale();

    // applied = -1 / opposite → форсируем apply на первом кадре.
    float applied_ui_scale     = -1.0f;
    bool  applied_font_builtin = !model.use_builtin_font;  // !=, форсирует первый apply
    bool  applied_dark_theme   = !model.dark_theme;        // != — то же самое для темы
    bool  applied_math_font    = !model.plot_math_font;    // != — и для LaTeX-подписей
    bool  applied_contrast     = !model.contrast_theme;
    int   applied_camo         = -1;
    // Узоры камуфляжных тем: текстура грузится при первом включении темы, списки
    // подкладки переиспользуются от кадра к кадру (inject_window_backdrops).
    GLuint camo_tex[AppSettings::kCamoCount]   = {};
    bool   camo_tried[AppSettings::kCamoCount] = {};
    std::vector<std::unique_ptr<ImDrawList>> camo_lists;
    AppSettings::ThemeColors applied_theme_colors = model.theme_colors;
    const MathFontPaths math_fonts = resolve_math_font_paths(dir);

    // Стиль каждый раз пересоздаём от дефолтов + выбранной темы. Раньше
    // кэшировали `base_style` после StyleColorsDark — но это блокировало
    // переключение Dark/Light в Settings, потому что новая палитра не
    // попадала в base_style. Перестроение дешёвое (memcpy + цвет-таблица),
    // поэтому его можно звать на каждом кадре перетаскивания пипетки цвета —
    // в отличие от apply_ui_scale, который ещё и пересобирает шрифты.
    auto apply_theme = [&](float scale) {
        ImGuiStyle& st = ImGui::GetStyle();
        st = ImGuiStyle{};
        const int camo = (!model.dark_theme && model.camo_theme > AppSettings::kCamoNone &&
                          model.camo_theme < AppSettings::kCamoCount) ? model.camo_theme : 0;
        const CamoPalette* pal = camo ? &kCamoPalettes[camo] : nullptr;
        if (pal ? !pal->light : model.dark_theme) ImGui::StyleColorsDark();
        else                                      ImGui::StyleColorsLight();
        ImVec4* c = st.Colors;
        const bool contrast = model.dark_theme && model.contrast_theme;
        if (pal) {
            if (!camo_tried[camo]) {
                camo_tried[camo] = true;
                camo_tex[camo] = load_ppm_texture(dir + "themes/" + pal->texture);
            }
            apply_camo_palette(c, *pal, camo_tex[camo] != 0);
        }
        if (contrast) {
            // High Contrast: фон окон чисто чёрный, текст белый, рамки у всех
            // полей и окон светлые, акцентные заливки непрозрачные.
            const ImVec4 black(0, 0, 0, 1);
            c[ImGuiCol_WindowBg] = c[ImGuiCol_PopupBg] = c[ImGuiCol_MenuBarBg] = black;
            c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgCollapsed] = c[ImGuiCol_ScrollbarBg] = black;
            c[ImGuiCol_DockingEmptyBg] = black;
            c[ImGuiCol_TitleBgActive]  = ImVec4(0.12f, 0.12f, 0.12f, 1);
            c[ImGuiCol_Text]           = ImVec4(1, 1, 1, 1);
            c[ImGuiCol_TextDisabled]   = ImVec4(0.72f, 0.72f, 0.72f, 1);
            c[ImGuiCol_Border]         = ImVec4(0.70f, 0.70f, 0.70f, 1);
            c[ImGuiCol_Separator]      = ImVec4(0.55f, 0.55f, 0.55f, 1);
            c[ImGuiCol_FrameBg]        = ImVec4(0.08f, 0.08f, 0.08f, 1);
            c[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.20f, 1);
            c[ImGuiCol_FrameBgActive]  = ImVec4(0.28f, 0.28f, 0.28f, 1);
            c[ImGuiCol_Button]         = ImVec4(0.10f, 0.30f, 0.62f, 1);
            c[ImGuiCol_Header]         = ImVec4(0.10f, 0.30f, 0.62f, 1);
            c[ImGuiCol_Tab]            = ImVec4(0.10f, 0.10f, 0.10f, 1);
            c[ImGuiCol_TabDimmed]      = ImVec4(0.06f, 0.06f, 0.06f, 1);
            c[ImGuiCol_TableBorderStrong] = ImVec4(0.65f, 0.65f, 0.65f, 1);
            c[ImGuiCol_TableBorderLight]  = ImVec4(0.40f, 0.40f, 0.40f, 1);
            c[ImGuiCol_ScrollbarGrab]  = ImVec4(0.45f, 0.45f, 0.45f, 1);
            st.FrameBorderSize = 1.0f;
            st.PopupBorderSize = 1.0f;
        }

        // Пользовательские основные цвета поверх темы. Производные оттенки
        // (наведение, нажатие, выделение) получаем смешиванием, как это
        // делает сама ImGui в StyleColorsDark/Light для своего синего.
        const auto& tc = model.theme_colors;
        auto rgb  = [&](int s, float a) { return ImVec4(tc.rgb[s][0], tc.rgb[s][1], tc.rgb[s][2], a); };
        auto lerp = [](ImVec4 a, ImVec4 b, float t) {
            return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
        };
        if (tc.set[AppSettings::kThemeBg]) {
            const ImVec4 bg = rgb(AppSettings::kThemeBg, 1.0f);
            c[ImGuiCol_WindowBg] = c[ImGuiCol_MenuBarBg] = c[ImGuiCol_TitleBg] = bg;
            c[ImGuiCol_TitleBgCollapsed] = c[ImGuiCol_ScrollbarBg] = c[ImGuiCol_DockingEmptyBg] = bg;
            c[ImGuiCol_PopupBg] = rgb(AppSettings::kThemeBg, 0.97f);
        }
        if (tc.set[AppSettings::kThemeText]) {
            c[ImGuiCol_Text]         = rgb(AppSettings::kThemeText, 1.0f);
            c[ImGuiCol_TextDisabled] = lerp(c[ImGuiCol_Text], c[ImGuiCol_WindowBg], 0.45f);
            c[ImGuiCol_TextDisabled].w = 1.0f;
        }
        if (tc.set[AppSettings::kThemeFrame]) {
            const ImVec4 f = rgb(AppSettings::kThemeFrame, 1.0f);
            c[ImGuiCol_FrameBg]        = f;
            c[ImGuiCol_FrameBgHovered] = lerp(f, c[ImGuiCol_Text], 0.12f);
            c[ImGuiCol_FrameBgActive]  = lerp(f, c[ImGuiCol_Text], 0.22f);
            c[ImGuiCol_FrameBgHovered].w = c[ImGuiCol_FrameBgActive].w = 1.0f;
        }
        if (tc.set[AppSettings::kThemeAccent]) {
            const int a = AppSettings::kThemeAccent;
            const ImVec4 bg = c[ImGuiCol_WindowBg];
            c[ImGuiCol_Button]            = rgb(a, contrast ? 0.70f : 0.40f);
            c[ImGuiCol_ButtonHovered]     = rgb(a, 1.00f);
            c[ImGuiCol_ButtonActive]      = lerp(rgb(a, 1.0f), ImVec4(0, 0, 0, 1), 0.15f);
            c[ImGuiCol_Header]            = rgb(a, contrast ? 0.60f : 0.31f);
            c[ImGuiCol_HeaderHovered]     = rgb(a, 0.80f);
            c[ImGuiCol_HeaderActive]      = rgb(a, 1.00f);
            c[ImGuiCol_CheckMark]         = rgb(a, 1.00f);
            c[ImGuiCol_SliderGrab]        = lerp(rgb(a, 1.0f), bg, 0.15f);
            c[ImGuiCol_SliderGrabActive]  = rgb(a, 1.00f);
            c[ImGuiCol_TitleBgActive]     = lerp(rgb(a, 1.0f), bg, 0.60f);
            c[ImGuiCol_TabHovered]        = rgb(a, 0.80f);
            c[ImGuiCol_TabSelected]       = lerp(rgb(a, 1.0f), bg, 0.40f);
            c[ImGuiCol_TabSelectedOverline] = rgb(a, 1.00f);
            c[ImGuiCol_TabDimmedSelected] = lerp(rgb(a, 1.0f), bg, 0.65f);
            c[ImGuiCol_SeparatorHovered]  = rgb(a, 0.78f);
            c[ImGuiCol_SeparatorActive]   = rgb(a, 1.00f);
            c[ImGuiCol_ResizeGrip]        = rgb(a, 0.20f);
            c[ImGuiCol_ResizeGripHovered] = rgb(a, 0.67f);
            c[ImGuiCol_ResizeGripActive]  = rgb(a, 0.95f);
            c[ImGuiCol_TextSelectedBg]    = rgb(a, 0.35f);
            c[ImGuiCol_DockingPreview]    = rgb(a, 0.70f);
            c[ImGuiCol_TextLink]          = rgb(a, 1.00f);
            c[ImGuiCol_NavCursor]         = rgb(a, 1.00f);
            c[ImGuiCol_DragDropTarget]    = rgb(a, 1.00f);
        }
        // Фон отмеченного чекбокса — цвет поля, подкрашенный акцентом темы
        // (Button несёт акцент во всех темах, включая пользовательский), а не
        // просто FrameBg: иначе на светлых темах галочка стоит на белом квадрате.
        // В тёмной камуфляжной теме подкраска сильнее; в светлых — слабее,
        // иначе квадрат сливается с вуалью над узором.
        {
            ImVec4 f = c[ImGuiCol_FrameBg], a = c[ImGuiCol_Button];
            const float t = (pal && !pal->light) ? 0.45f : 0.25f;
            c[ImGuiCol_CheckboxSelectedBg] = ImVec4(f.x + (a.x - f.x) * t, f.y + (a.y - f.y) * t,
                                                    f.z + (a.z - f.z) * t, 1.0f);
        }
        st.ScaleAllSizes(scale);

        // Плот-палитра (текст/сетка/рамки/фон FBO) живёт отдельно от ImGui
        // style — её читают plot_view_2d/3d, heatmap_view, plot_legend и
        // colorbar в gui.cpp. Тригерится одновременно со сменой темы.
        set_plot_light_theme(!model.dark_theme);
        set_plot_contrast_theme(contrast);
        const int p = AppSettings::kThemePlotBg;
        if (tc.set[p])
            set_plot_bg_override(true, tc.rgb[p][0], tc.rgb[p][1], tc.rgb[p][2]);
        else if (pal)
            set_plot_bg_override(true, pal->plot[0], pal->plot[1], pal->plot[2]);
        else
            set_plot_bg_override(false, 0, 0, 0);
    };

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

        apply_theme(scale);
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
            applied_contrast     = model.contrast_theme;
            applied_theme_colors = model.theme_colors;
            applied_camo         = model.camo_theme;
        } else if (model.contrast_theme != applied_contrast ||
                   model.camo_theme     != applied_camo     ||
                   model.theme_colors   != applied_theme_colors) {
            // Только цвета: шрифты не трогаем, хватает пересборки стиля.
            apply_theme(applied_ui_scale);
            applied_contrast     = model.contrast_theme;
            applied_camo         = model.camo_theme;
            applied_theme_colors = model.theme_colors;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        tree_state::apply();

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
        if (!model.dark_theme && model.camo_theme > AppSettings::kCamoNone &&
            model.camo_theme < AppSettings::kCamoCount && camo_tex[model.camo_theme] != 0)
            inject_window_backdrops(ImGui::GetDrawData(), camo_tex[model.camo_theme],
                                    512.0f * applied_ui_scale, camo_lists);
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
    // Списки подкладки узора и текстуры камуфляжа — до DestroyContext:
    // деструктор ImDrawList обращается к общим данным контекста (_Data), и
    // после его уничтожения выход из программы падал на чтении памяти.
    camo_lists.clear();
    for (GLuint& t : camo_tex)
        if (t) { glDeleteTextures(1, &t); t = 0; }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
