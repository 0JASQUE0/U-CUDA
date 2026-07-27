#include "krs_cpu.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>

// ============================================================
// Часть 1. Статическая проверка индексов
// ============================================================
namespace {

bool is_ident_start(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool is_ident_char (char c) { return std::isalnum((unsigned char)c) || c == '_'; }

} // namespace

bool krs_cpu_check_indices(const std::string& b, int amountOfX, int amountOfValues,
                           std::vector<KrsCpuDiag>& diags) {
    const size_t n = b.size();
    int  line = 1;
    bool ok   = true;

    for (size_t i = 0; i < n; ) {
        const char c = b[i];
        if (c == '\n') { ++line; ++i; continue; }
        // // -комментарий
        if (c == '/' && i + 1 < n && b[i + 1] == '/') {
            while (i < n && b[i] != '\n') ++i;
            continue;
        }
        // /* */ -комментарий
        if (c == '/' && i + 1 < n && b[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(b[i] == '*' && b[i + 1] == '/')) {
                if (b[i] == '\n') ++line;
                ++i;
            }
            i = (i + 1 < n) ? i + 2 : n;
            continue;
        }
        // строковый / символьный литерал
        if (c == '"' || c == '\'') {
            const char q = c; ++i;
            while (i < n && b[i] != q) {
                if (b[i] == '\\')      { ++i; }
                else if (b[i] == '\n') { ++line; }
                ++i;
            }
            ++i; continue;
        }
        if (!is_ident_start(c)) { ++i; continue; }

        // Идентификатор берём ЦЕЛИКОМ — иначе X1 совпал бы с X, a_param с a.
        const size_t s = i;
        while (i < n && is_ident_char(b[i])) ++i;
        const std::string id = b.substr(s, i - s);
        if (id != "X" && id != "a") continue;

        // '[' сразу за именем
        size_t j = i;
        while (j < n && (b[j] == ' ' || b[j] == '\t')) ++j;
        if (j >= n || b[j] != '[') continue;
        ++j;
        while (j < n && (b[j] == ' ' || b[j] == '\t')) ++j;

        // Индекс должен быть целым литералом ЦЕЛИКОМ до ']'. Всё остальное
        // (X[i], X[k+1]) вычисляется в рантайме и статически непроверяемо.
        const size_t ds = j;
        while (j < n && std::isdigit((unsigned char)b[j])) ++j;
        if (j == ds) continue;
        const size_t de = j;
        while (j < n && (b[j] == ' ' || b[j] == '\t')) ++j;
        if (j >= n || b[j] != ']') continue;

        const long idx = std::strtol(b.substr(ds, de - ds).c_str(), nullptr, 10);
        if (id == "X" && (idx < 0 || idx >= amountOfX)) {
            std::string m = "X[" + std::to_string(idx) + "]: в системе " +
                            std::to_string(amountOfX) + " переменных";
            if (amountOfX > 0)
                m += ", допустимо X[0.." + std::to_string(amountOfX - 1) + "]";
            diags.push_back({ line, m });
            ok = false;
        }
        if (id == "a" && (idx < 0 || idx >= amountOfValues)) {
            std::string m = "a[" + std::to_string(idx) + "]: доступно a[0] (symmetry s)";
            if (amountOfValues > 1)
                m += " и a[1.." + std::to_string(amountOfValues - 1) + "] (параметры)";
            else
                m += "; параметров в системе нет";
            diags.push_back({ line, m });
            ok = false;
        }
    }
    return ok;
}

// ============================================================
// Часть 2. Процессы и файлы
//
// Всё держим в ANSI (CreateProcessA + *A-функции путей): пути мы получаем
// от GetTempPathA / getenv_s / vswhere, и наивная конвертация narrow->wide
// поломала бы их на системах с не-ASCII в путях.
// ============================================================
namespace {

std::mutex g_compile_mtx;   // компиляция и запись в кэш — под одним замком

// cl.exe пишет локализованные сообщения в OEM-кодировке консоли (для русской
// локали — CP866). ImGui рисует UTF-8, поэтому без перекодировки текст ошибки
// в панели превратился бы в кашу.
std::string oem_to_utf8(const std::string& s) {
    if (s.empty()) return s;
    const int wn = MultiByteToWideChar(CP_OEMCP, 0, s.data(), (int)s.size(), nullptr, 0);
    if (wn <= 0) return s;
    std::wstring w((size_t)wn, L'\0');
    MultiByteToWideChar(CP_OEMCP, 0, s.data(), (int)s.size(), &w[0], wn);
    const int un = WideCharToMultiByte(CP_UTF8, 0, w.data(), wn, nullptr, 0, nullptr, nullptr);
    if (un <= 0) return s;
    std::string u((size_t)un, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), wn, &u[0], un, nullptr, nullptr);
    return u;
}

// Запускает cmdline и возвращает его stdout+stderr (нужно для vswhere).
std::string run_capture(const std::string& cmdline) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof sa;
    sa.bInheritHandle = TRUE;

    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return {};
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb         = sizeof si;
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = wr;

    PROCESS_INFORMATION pi{};
    std::vector<char> cmd(cmdline.begin(), cmdline.end());
    cmd.push_back('\0');

    BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr,
                             TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!ok) { CloseHandle(rd); return {}; }

    std::string out;
    char  buf[4096];
    DWORD n = 0;
    while (ReadFile(rd, buf, sizeof buf, &n, nullptr) && n > 0) out.append(buf, n);
    CloseHandle(rd);

    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return out;
}

// Запускает cmdline, stdout+stderr -> log_path (файл остаётся на диске, чтобы
// можно было заглянуть руками) и заодно в out. Возвращает код возврата
// процесса (-1 = не запустился).
//
// Вывод читаем через СВОЙ handle, а не переоткрывая файл: сразу после выхода
// компилятора файл ещё может быть занят (антивирус, не успевший закрыться
// потомок), и fopen отдавал EACCES — лог терялся, а ошибка компиляции
// приходила в UI без номера строки.
int run_logged(const std::string& cmdline, const std::string& log_path,
               std::string& out) {
    out.clear();
    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof sa;
    sa.bInheritHandle = TRUE;

    HANDLE hLog = CreateFileA(log_path.c_str(),
                              GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                              &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hLog == INVALID_HANDLE_VALUE) return -1;

    STARTUPINFOA si{};
    si.cb         = sizeof si;
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = hLog;
    si.hStdError  = hLog;

    PROCESS_INFORMATION pi{};
    std::vector<char> cmd(cmdline.begin(), cmdline.end());
    cmd.push_back('\0');

    BOOL ok = CreateProcessA(nullptr, cmd.data(), nullptr, nullptr,
                             TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    if (!ok) { CloseHandle(hLog); return -1; }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 1;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    SetFilePointer(hLog, 0, nullptr, FILE_BEGIN);
    char  buf[4096];
    DWORD n = 0;
    while (ReadFile(hLog, buf, sizeof buf, &n, nullptr) && n > 0) out.append(buf, n);
    CloseHandle(hLog);
    return (int)rc;
}

// Путь к vcvars64.bat. Пусто = не найден; в why кладётся причина.
// Ищем один раз за сессию: vswhere запускается процессом, дёргать его на
// каждый кадр GUI нельзя.
const std::string& vcvars_path(std::string& why) {
    static std::string s_path;
    static std::string s_why;
    static std::once_flag once;
    std::call_once(once, [] {
        char   pf[MAX_PATH] = { 0 };
        size_t len = 0;
        if (getenv_s(&len, pf, sizeof pf, "ProgramFiles(x86)") != 0 || len == 0) {
            s_why = "не найдена переменная ProgramFiles(x86)";
            return;
        }
        const std::string vswhere = std::string(pf) +
            "\\Microsoft Visual Studio\\Installer\\vswhere.exe";
        if (GetFileAttributesA(vswhere.c_str()) == INVALID_FILE_ATTRIBUTES) {
            s_why = "не найден vswhere.exe (Visual Studio не установлена?)";
            return;
        }
        std::string out = run_capture(
            "\"" + vswhere + "\" -latest -products * "
            "-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 "
            "-property installationPath");
        while (!out.empty() &&
               (out.back() == '\r' || out.back() == '\n' || out.back() == ' '))
            out.pop_back();
        if (out.empty()) {
            s_why = "vswhere не нашёл установку с компонентом C++ (VC Tools)";
            return;
        }
        const std::string vc = out + "\\VC\\Auxiliary\\Build\\vcvars64.bat";
        if (GetFileAttributesA(vc.c_str()) == INVALID_FILE_ATTRIBUTES) {
            s_why = "не найден vcvars64.bat в " + out;
            return;
        }
        s_path = vc;
    });
    why = s_why;
    return s_path;
}

} // namespace

bool krs_cpu_backend_available(std::string* why_not) {
    std::string why;
    if (!vcvars_path(why).empty()) return true;
    if (why_not) *why_not = why.empty() ? "компилятор не найден" : why;
    return false;
}

// ============================================================
// Часть 3. Генерация исходника, компиляция, загрузка
// ============================================================
namespace {

// Версия пролога/командной строки. ВХОДИТ В КЛЮЧ КЭША: иначе после правки
// пролога переиспользовалась бы DLL, собранная старым.
constexpr int kPreludeVersion = 2;

// Пролог перед телом. Компилируется КАК C++ (/TP), поэтому bool / true /
// false родные, а объявления допустимы в любом месте блока — как в CUDA.
//
// `using std::abs` обязателен. Без него <stdlib.h> даёт только целочисленные
// перегрузки (int/long/__int64), и `numb s_abs = abs(sigma);` из схемы
// Burkin Matreshka либо не компилируется (C2668), либо молча обрезает
// значение до целого. В CUDA abs(double) — это double, приводим к тому же.
// min/max для double тоже есть в device-коде CUDA, объявляем их сами.
//
// pi / euler дублируют configCUDA.h; локальное объявление внутри тела
// (см. схему "sine") их просто затеняет — ровно как на GPU.
std::string make_source(const std::string& body, int amountOfX) {
    std::ostringstream o;
    o << "#include <cmath>\n"
         "#include <cstdlib>\n"
         "using std::abs;\n"
         "static inline double min(double x, double y) { return x < y ? x : y; }\n"
         "static inline double max(double x, double y) { return x > y ? x : y; }\n"
         "typedef double numb;\n"
         "#define AMOUNTOFX " << amountOfX << "\n"
         "static const double pi    = 3.1415926535897932384626433832795;\n"
         "static const double euler = 2.7182818284590452353602874713527;\n"
         "extern \"C\" __declspec(dllexport)\n"
         "void krs_step(numb* X, const numb* a, numb h) {\n"
         // Дальше — код пользователя. #line переводит нумерацию компилятора
         // в координаты ТЕЛА, поэтому "krs(12): error" указывает ровно на
         // 12-ю строку в редакторе схемы.
         "#line 1 \"krs\"\n"
      << body << "\n}\n";
    return o.str();
}

unsigned long long hash_key(const std::string& body, int nx, int nv) {
    unsigned long long h = 1469598103934665603ULL;      // FNV-1a
    auto mix = [&](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    mix(body.data(), body.size());
    mix(&nx, sizeof nx);
    mix(&nv, sizeof nv);
    const int ver = kPreludeVersion;
    mix(&ver, sizeof ver);
    return h;
}

std::string cache_dir(unsigned long long key) {
    char tmp[MAX_PATH] = { 0 };
    GetTempPathA(MAX_PATH, tmp);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", key);
    const std::string root = std::string(tmp) + "u-cuda-krs";
    const std::string dir  = root + "\\" + buf + "\\";
    CreateDirectoryA(root.c_str(), nullptr);
    CreateDirectoryA(dir.c_str(),  nullptr);
    return dir;
}

// Разбирает вывод cl.exe. Строки вида
//   krs(12): error C2065: 'foo': undeclared identifier
// превращаются в {12, "error C2065: ..."}. Благодаря `#line 1 "krs"` номера
// уже в координатах тела схемы. Строки со словом error, но без распознанной
// позиции, добавляются с line = 0.
void parse_cl_log(const std::string& log, std::vector<KrsCpuDiag>& diags) {
    std::istringstream in(log);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty()) continue;

        const size_t fat = line.find(": fatal error");
        const size_t err = line.find(": error");
        const size_t pos = (fat != std::string::npos) ? fat : err;
        if (pos == std::string::npos) continue;

        int ln = 0;
        if (pos > 0 && line[pos - 1] == ')') {
            const size_t close = pos - 1;
            const size_t open  = line.rfind('(', close);
            if (open != std::string::npos) {
                std::string num = line.substr(open + 1, close - open - 1);
                const size_t comma = num.find(',');   // может быть "12,5"
                if (comma != std::string::npos) num = num.substr(0, comma);
                bool digits = !num.empty();
                for (char c : num)
                    if (!std::isdigit((unsigned char)c)) { digits = false; break; }
                if (digits) ln = std::atoi(num.c_str());
            }
        }
        diags.push_back({ ln, line.substr(pos + 2) });   // отрезаем ": "
        if (diags.size() >= 20) break;                   // не заливаем UI простынёй
    }
}

} // namespace

KrsCpuStep::~KrsCpuStep() { release(); }

KrsCpuStep::KrsCpuStep(KrsCpuStep&& o) noexcept
    : module_(o.module_), fn_(o.fn_) { o.module_ = nullptr; o.fn_ = nullptr; }

KrsCpuStep& KrsCpuStep::operator=(KrsCpuStep&& o) noexcept {
    if (this != &o) {
        release();
        module_ = o.module_; fn_ = o.fn_;
        o.module_ = nullptr; o.fn_ = nullptr;
    }
    return *this;
}

void KrsCpuStep::release() {
    if (module_) FreeLibrary((HMODULE)module_);
    module_ = nullptr;
    fn_     = nullptr;
}

bool KrsCpuStep::compile(const std::string& body, int amountOfX, int amountOfValues,
                         std::vector<KrsCpuDiag>& diags) {
    release();

    // Индексы проверяем ДО компиляции: в нативном коде выход за границу X[]
    // — это порча стека вызывающего, а не понятная ошибка.
    if (!krs_cpu_check_indices(body, amountOfX, amountOfValues, diags))
        return false;

    std::string why;
    const std::string vcvars = vcvars_path(why);
    if (vcvars.empty()) {
        diags.push_back({ 0, "CPU-компилятор недоступен: " + why });
        return false;
    }

    std::lock_guard<std::mutex> lock(g_compile_mtx);

    const unsigned long long key = hash_key(body, amountOfX, amountOfValues);
    const std::string dir = cache_dir(key);
    const std::string src = dir + "krs.cpp";
    const std::string dll = dir + "krs.dll";
    const std::string log = dir + "build.log";

    // Кэш: та же схема + та же размерность -> DLL уже собрана.
    if (GetFileAttributesA(dll.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const std::string source = make_source(body, amountOfX);
        FILE* f = nullptr;
        if (fopen_s(&f, src.c_str(), "wb") != 0 || !f) {
            diags.push_back({ 0, "не удалось записать " + src });
            return false;
        }
        fwrite(source.data(), 1, source.size(), f);
        fclose(f);

        // /TP — компилировать как C++ (см. комментарий к make_source);
        // /LD — DLL; /O2 — оптимизация (ради неё всё и затевается);
        // /Fe /Fo /Fd — артефакты в каталог кэша, чтобы не сорить рядом с exe.
        //
        // Пути к /Fo и /Fd задаём ПОФАЙЛОВО, а не каталогом: каталог
        // оканчивается на '\', и в "...\dir\" обратный слэш экранирует
        // закрывающую кавычку — аргументы слипаются, cl падает с C1083.
        const std::string cmd =
            "cmd.exe /c \"\"" + vcvars + "\" >nul && cl /nologo /TP /O2 /LD"
            " /Fe:\"" + dll + "\""
            " /Fo:\"" + dir + "krs.obj\""
            " /Fd:\"" + dir + "krs.pdb\""
            " \"" + src + "\"\"";

        std::string build_out;
        const int rc = run_logged(cmd, log, build_out);
        if (rc != 0 || GetFileAttributesA(dll.c_str()) == INVALID_FILE_ATTRIBUTES) {
            parse_cl_log(oem_to_utf8(build_out), diags);
            if (diags.empty())
                diags.push_back({ 0, "cl.exe завершился с кодом " + std::to_string(rc) });
            DeleteFileA(dll.c_str());   // не оставляем полуфабрикат в кэше
            return false;
        }
    }

    HMODULE m = LoadLibraryA(dll.c_str());
    if (!m) {
        diags.push_back({ 0, "не удалось загрузить " + dll });
        return false;
    }
    auto p = (StepFn)GetProcAddress(m, "krs_step");
    if (!p) {
        FreeLibrary(m);
        diags.push_back({ 0, "в собранной DLL нет krs_step" });
        return false;
    }
    module_ = m;
    fn_     = p;
    return true;
}
