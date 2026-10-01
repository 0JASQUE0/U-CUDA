#include "krs_cpu.h"
#include "adaptive_settings.h"   // adaptive_ctrl_source
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>

// exe_dir(): определён в app_main.cpp (Release) / main_NonLinAnal.cu (Debug).
// Нужен, чтобы дать cl.exe путь к configCUDA.h из kernels\ рядом с .exe.
extern std::string exe_dir();

// Часть 1. Статическая проверка индексов
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
            std::string m = "X[" + std::to_string(idx) + "]: the system has " +
                            std::to_string(amountOfX) + " variables";
            if (amountOfX > 0)
                m += ", allowed X[0.." + std::to_string(amountOfX - 1) + "]";
            diags.push_back({ line, m });
            ok = false;
        }
        if (id == "a" && (idx < 0 || idx >= amountOfValues)) {
            std::string m = "a[" + std::to_string(idx) + "]: a[0] (symmetry s) is available";
            if (amountOfValues > 1)
                m += " and a[1.." + std::to_string(amountOfValues - 1) + "] (parameters)";
            else
                m += "; the system has no parameters";
            diags.push_back({ line, m });
            ok = false;
        }
    }
    return ok;
}

// Часть 2. Процессы и файлы
//
// Всё держим в ANSI (CreateProcessA + *A-функции путей): пути мы получаем
// от GetTempPathA / getenv_s / vswhere, и наивная конвертация narrow->wide
// поломала бы их на системах с не-ASCII в путях.
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
            s_why = "the ProgramFiles(x86) variable was not found";
            return;
        }
        const std::string vswhere = std::string(pf) +
            "\\Microsoft Visual Studio\\Installer\\vswhere.exe";
        if (GetFileAttributesA(vswhere.c_str()) == INVALID_FILE_ATTRIBUTES) {
            s_why = "vswhere.exe not found (is Visual Studio installed?)";
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
            s_why = "vswhere found no installation with the C++ component (VC Tools)";
            return;
        }
        const std::string vc = out + "\\VC\\Auxiliary\\Build\\vcvars64.bat";
        if (GetFileAttributesA(vc.c_str()) == INVALID_FILE_ATTRIBUTES) {
            s_why = "vcvars64.bat not found in " + out;
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
    if (why_not) *why_not = why.empty() ? "compiler not found" : why;
    return false;
}

// Часть 3. Генерация исходника, компиляция, загрузка
namespace {

// Версия пролога/командной строки. ВХОДИТ В КЛЮЧ КЭША: иначе после правки
// пролога переиспользовалась бы DLL, собранная старым. Туда же уходит размер
// numb — при смене float<->double в configCUDA.h кэш обязан протухнуть, иначе
// подхватилась бы DLL, собранная в другой точности.
constexpr int kPreludeVersion = 6;

// Пролог перед телом. Компилируется КАК C++ (/TP), поэтому bool / true /
// false родные, а объявления допустимы в любом месте блока — как в CUDA.
//
// numb, AMOUNTOFX, pi, euler и ucmplx приходят из configCUDA.h — того самого
// файла, который NVRTC подставляет в ядро. Раньше пролог объявлял их сам
// (typedef + две константы), и тело, написанное под GPU, на CPU могло не
// собраться: схемы с комплексными полушагами (Complex CD и любая custom КРС на
// её основе) падали с "C2065: ucmplx необъявленный идентификатор", потому что
// тип живёт в заголовке, а сюда он не попадал. Дублировать объявления второй
// раз — тот же класс ошибки в будущем, поэтому берём заголовок целиком: он
// уже лежит рядом с .exe (post-build копирует его в kernels\), и путь к нему
// уходит в cl через /I.
//
// `using std::abs` обязателен. Без него <stdlib.h> даёт только целочисленные
// перегрузки (int/long/__int64), и `numb s_abs = abs(sigma);` из схемы
// Burkin Matreshka либо не компилируется (C2668), либо молча обрезает
// значение до целого. В CUDA abs(double) — это double, приводим к тому же.
// min/max для double тоже есть в device-коде CUDA, объявляем их сами.
//
// AMOUNTOFX определяем ДО include: в configCUDA.h он под #ifndef, наш #define
// выигрывает, как и в NVRTC-шаблонах.
// В расширенных режимах подменяется РОВНО тип: тело схемы и заголовок те же,
// но numb становится ucuda::dd или ucuda::qd. Переопределение идёт макросами
// до include configCUDA.h — тем же способом, каким NVRTC-шаблоны
// переопределяют AMOUNTOFX. Константы pi/euler подменяются вместе с типом:
// double-литерал обрезал бы их до 17 цифр, и расширенная точность кончалась
// бы на первом же pi в правой части.
//
// Сигнатура там другая — h приходит указателем (см. StepFnDD/StepFnQD в
// krs_cpu.h), поэтому первая строка функции распаковывает его в локальный h.
// Тело её не видит: #line ниже, и нумерация ошибок остаётся в координатах
// схемы.
std::string make_source(const std::string& body, int amountOfX, KrsCpuPrec prec) {
    const bool hp = (prec != KrsCpuPrec::Double);
    const char* type  = (prec == KrsCpuPrec::QD) ? "ucuda::qd"   : "ucuda::dd";
    const char* c_pi  = (prec == KrsCpuPrec::QD) ? "ucuda::c_qd_pi" : "ucuda::c_pi";
    const char* c_e   = (prec == KrsCpuPrec::QD) ? "ucuda::c_qd_e"  : "ucuda::c_e";
    std::ostringstream o;
    o << "#include <cmath>\n"
         "#include <cstdlib>\n"
         "using std::abs;\n";
    if (hp) {
        o << "#include \"ucuda_hp.h\"\n"
             "#define UCUDA_NUMB_TYPE " << type  << "\n"
             "#define UCUDA_PI    "     << c_pi  << "\n"
             "#define UCUDA_EULER "     << c_e   << "\n";
    }
    o << "#define AMOUNTOFX " << amountOfX << "\n"
         "#include \"configCUDA.h\"\n"
         "static inline numb min(numb x, numb y) { return x < y ? x : y; }\n"
         "static inline numb max(numb x, numb y) { return x > y ? x : y; }\n"
         "extern \"C\" __declspec(dllexport)\n";
    if (hp) o << "void krs_step(numb* X, const numb* a, const numb* h_ptr) {\n"
                 "    numb h = *h_ptr;\n";
    else    o << "void krs_step(numb* X, const numb* a, numb h) {\n";
         // Дальше — код пользователя. #line переводит нумерацию компилятора
         // в координаты ТЕЛА, поэтому "krs(12): error" указывает ровно на
         // 12-ю строку в редакторе схемы.
    o << "#line 1 \"krs\"\n"
      << body << "\n}\n";
    return o.str();
}

unsigned long long hash_key(const std::string& body, int nx, int nv, KrsCpuPrec prec) {
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
    const int numb_bytes = (int)sizeof(numb);
    mix(&numb_bytes, sizeof numb_bytes);
    // Точность — часть ключа: double- и dd-сборки одного тела лежат рядом и
    // не вытесняют друг друга при переключении режима в UI.
    const int prec_code = (int)prec;
    mix(&prec_code, sizeof prec_code);
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

// Собирает source в DLL каталога кэша key (если её там ещё нет) и возвращает путь к
// ней. Вызывать под g_compile_mtx. Общая часть КРС и регулятора шага.
bool build_cached_dll(const std::string& source, unsigned long long key, std::string& dll,
                      std::vector<KrsCpuDiag>& diags) {
    const std::string dir = cache_dir(key);
    const std::string src = dir + "krs.cpp";
    const std::string log = dir + "build.log";
    dll = dir + "krs.dll";

    // Кэш: тот же исходник -> DLL уже собрана.
    if (GetFileAttributesA(dll.c_str()) != INVALID_FILE_ATTRIBUTES) return true;
    FILE* f = nullptr;
    if (fopen_s(&f, src.c_str(), "wb") != 0 || !f) {
        diags.push_back({ 0, "failed to write " + src });
        return false;
    }
    fwrite(source.data(), 1, source.size(), f);
    fclose(f);

    std::string why;
    const std::string vcvars = vcvars_path(why);

    // /TP — компилировать как C++ (см. комментарий к make_source);
    // /LD — DLL; /O2 — оптимизация (ради неё всё и затевается);
    // /Fe /Fo /Fd — артефакты в каталог кэша, чтобы не сорить рядом с exe.
    //
    // Пути к /Fo и /Fd задаём ПОФАЙЛОВО, а не каталогом: каталог
    // оканчивается на '\', и в "...\dir\" обратный слэш экранирует
    // закрывающую кавычку — аргументы слипаются, cl падает с C1083.
    // /I — каталог с configCUDA.h (его копию post-build кладёт в kernels\
    // рядом с .exe; оттуда же его читает NVRTC-путь, так что CPU и GPU
    // видят один и тот же файл).
    const std::string inc_dir = exe_dir() + "\\kernels";

    const std::string cmd =
        // /fp:precise — умолчание MSVC, но задано явно: double-double
        // держится на безошибочных преобразованиях вида (s - a), и при
        // /fp:fast компилятор вправе свернуть их в ноль. Тогда точность
        // молча упала бы до обычного double, а сборка прошла бы успешно.
        "cmd.exe /c \"\"" + vcvars + "\" >nul && cl /nologo /TP /O2 /fp:precise /LD"
        " /I\"" + inc_dir + "\""
        " /Fe:\"" + dll + "\""
        " /Fo:\"" + dir + "krs.obj\""
        " /Fd:\"" + dir + "krs.pdb\""
        " \"" + src + "\"\"";

    std::string build_out;
    const int rc = run_logged(cmd, log, build_out);
    if (rc != 0 || GetFileAttributesA(dll.c_str()) == INVALID_FILE_ATTRIBUTES) {
        parse_cl_log(oem_to_utf8(build_out), diags);
        if (diags.empty())
            diags.push_back({ 0, "cl.exe exited with code " + std::to_string(rc) });
        DeleteFileA(dll.c_str());   // не оставляем полуфабрикат в кэше
        return false;
    }
    return true;
}

} // namespace

// Часть 4. Пользовательский регулятор шага
//
// Тело — внутрь функции с той же сигнатурой, что ucuda_ctrl_custom на GPU, над
// раскладкой kernels/ucuda_adaptive.cuh (структуры, нормы, помощники). Заголовок
// читается из kernels\ рядом с .exe — тем же файлом, что у NVRTC; его текст входит
// в ключ кэша, иначе после правки заголовка подхватилась бы DLL со старой раскладкой.
namespace {

constexpr int kCtrlPreludeVersion = 1;

std::string make_ctrl_source(const std::string& body) {
    std::ostringstream o;
    o << "#include <cmath>\n"
         "#include <cstdlib>\n"
         "using std::abs;\n"
         "#include \"configCUDA.h\"\n"
         "static inline numb min(numb x, numb y) { return x < y ? x : y; }\n"
         "static inline numb max(numb x, numb y) { return x > y ? x : y; }\n"
         "#define UCUDA_ADAPT_LAYOUT_ONLY\n"
         "#include \"ucuda_adaptive.cuh\"\n"
         "extern \"C\" __declspec(dllexport)\n"
         "void ucuda_ctrl_custom_c(const UcudaCtlIn* in_p, UcudaCtlMem* m_p, UcudaCtlOut* o_p) {\n"
         "    const UcudaCtlIn& in = *in_p; UcudaCtlMem& m = *m_p; UcudaCtlOut& o = *o_p;\n"
         "    (void)in; (void)m; (void)o;\n"
         "#line 1 \"controller\"\n"
      << body << "\n}\n";
    return o.str();
}

unsigned long long ctrl_hash_key(const std::string& body, const std::string& header) {
    unsigned long long h = 1469598103934665603ULL;      // FNV-1a
    auto mix = [&](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    const char tag[] = "step-controller";
    mix(tag, sizeof tag);
    mix(body.data(), body.size());
    mix(header.data(), header.size());
    const int ver = kCtrlPreludeVersion;
    mix(&ver, sizeof ver);
    const int numb_bytes = (int)sizeof(numb);
    mix(&numb_bytes, sizeof numb_bytes);
    return h;
}

} // namespace

// Часть 5. Адаптивный шаг нативным кодом (AdaptiveCpuModule)
//
// Исходник DLL повторяет устройство GPU-модуля свипов: перед kernels/adaptive_part.cu
// (тела схемы, раскладка драйвера, регулятор, драйвер, поиск пиков) стоит то, что на GPU
// даёт шаблон с cudaLibrary.cu, — PeakStream (его текст вырезается из cudaLibrary.cu) и
// значение узла сетки; __device__ / __host__ / __forceinline__ сняты макросами, ядра
// свипов выключены (UCUDA_AD_NO_SWEEP_KERNELS), размерность — AMOUNTOFX этой DLL.
// par_or_var на GPU — макрос модуля, здесь — переменная потока (классика БД её ставит).
namespace {

constexpr int kAdModuleVersion = 2;

// Подстановка плейсхолдера {{name}} во всех вхождениях.
void replace_all(std::string& s, const std::string& from, const std::string& to) {
    for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size()))
        s.replace(p, from.size(), to);
}

std::string strip_bom(std::string s) {
    if (s.size() >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB && (unsigned char)s[2] == 0xBF)
        s.erase(0, 3);
    return s;
}

// struct PeakStream { ... }; из текста cudaLibrary.cu: от строки "struct PeakStream" до
// первой строки "};". Пусто — не нашлось.
std::string extract_peak_stream(const std::string& lib) {
    size_t b = lib.find("\nstruct PeakStream");
    while (b != std::string::npos) {
        const char c = b + 18 < lib.size() ? lib[b + 18] : '\0';
        if (c == '\r' || c == '\n') break;
        b = lib.find("\nstruct PeakStream", b + 1);
    }
    if (b == std::string::npos) return {};
    const size_t e = lib.find("\n};", b);
    if (e == std::string::npos) return {};
    return lib.substr(b + 1, e + 3 - (b + 1)) + "\n";
}

std::string make_ad_module_source(const std::string& rhs, const std::string& emb, const std::string& dprep,
                                  const std::string& deval, const std::string& ctrl_body, int amountOfX,
                                  const std::string& prelude, const std::string& peak_stream,
                                  std::string adaptive_part) {
    const bool dense = !dprep.empty() && !deval.empty();
    replace_all(adaptive_part, "{{KRS_RHS_BODY}}", rhs);
    replace_all(adaptive_part, "{{KRS_EMB_BODY}}", emb);
    replace_all(adaptive_part, "{{KRS_DPREP_BODY}}", dense ? dprep : std::string());
    replace_all(adaptive_part, "{{KRS_DEVAL_BODY}}", dense ? deval : std::string());
    replace_all(adaptive_part, "{{CTRL_CUSTOM}}", adaptive_ctrl_source(ctrl_body));
    std::ostringstream o;
    o << "#include <cmath>\n"
         "#include <cstdlib>\n"
         "#include <cstdint>\n"
         "using std::abs;\n"
         "#define AMOUNTOFX " << amountOfX << "\n"
         "static thread_local int ucuda_cpu_par_or_var = 1;\n"
         "#define par_or_var ucuda_cpu_par_or_var\n"
      << prelude << "\n"
         "#include \"configCUDA.h\"\n"
         "static inline numb min(numb x, numb y) { return x < y ? x : y; }\n"
         "static inline numb max(numb x, numb y) { return x > y ? x : y; }\n"
         "#define __device__\n"
         "#define __host__\n"
         "#define __forceinline__ inline\n"
         "static inline int atomicAdd(int* p, int v) { const int o = *p; *p += v; return o; }\n"
         // getValueByIdx / getValueByIdx_log из cudaLibrary.cu (та же формула узла).
         "static inline numb getValueByIdx(const size_t idx, const int nPts, const numb lo, const numb hi,\n"
         "                                 const int valueNumber) {\n"
         "    if (nPts <= 0) return lo;\n"
         "    if (nPts == 1) return hi;\n"
         "    const int64_t divisor = (valueNumber == 0) ? 1 : nPts;\n"
         "    return ucuda_node_value((int)(((int64_t)idx / divisor) % nPts), nPts, lo, hi);\n"
         "}\n"
         "static inline numb getValueByIdx_log(const int idx, const int nPts, const numb lo, const numb hi,\n"
         "                                     const int valueNumber) {\n"
         "    const int n = (int)((int64_t)((int64_t)idx / pow((numb)nPts, (numb)valueNumber)) % nPts);\n"
         "    return ucuda_node_value_log(n, nPts, lo, hi);\n"
         "}\n"
      << peak_stream
      << "#define UCUDA_AD_NO_SWEEP_KERNELS 1\n"
         "#define UCUDA_AD_LYAPUNOV 1\n";
    if (!dense) o << "#define UCUDA_AD_NO_DENSE 1\n";
    o << adaptive_part << "\n"
         "extern \"C\" __declspec(dllexport)\n"
         "int ucuda_cpu_ad_endpoint(const double* ic, const double* a, const UcudaAdaptParams* P, double T,\n"
         "                          double* y, double* st) {\n"
         "    const UcudaKrsFns K{};\n"
         "    UcudaAdaptState S;\n"
         "    ucuda_ad_init(S, K, AMOUNTOFX, ic, (numb)0, a, *P);\n"
         "    while (S.t < T && !S.diverged) ucuda_ad_step(S, K, a, *P, T);\n"
         "    for (int k = 0; k < AMOUNTOFX; ++k) y[k] = S.X[k];\n"
         "    st[0] = (double)S.st.nacc; st[1] = (double)S.st.nrej; st[2] = (double)S.st.nforced;\n"
         "    st[3] = (double)S.st.nrhs; st[4] = S.st.hmin; st[5] = S.st.hmax;\n"
         "    st[6] = S.st.nacc > 0 ? S.st.hsum / (double)S.st.nacc : 0.0;\n"
         "    st[7] = (double)S.diverged;\n"
         "    return S.diverged ? 0 : 1;\n"
         "}\n"
         "extern \"C\" __declspec(dllexport)\n"
         "void ucuda_cpu_ad_lyap(int ls, int continuation, int nPts, double lo, double hi, int reverse,\n"
         "    int logScale, int mutParamIdx, const double* baseValues, int amountOfValues, const double* baseX,\n"
         "    const UcudaAdaptParams* P, int axisKind, double tolRatio, double tTr, double NT, int nBlocks,\n"
         "    int nWarm, double eps, int renorm, double maxValue, double* result, double* stats,\n"
         "    const volatile int* cancel, int* progress) {\n"
         "    const UcudaKrsFns K{};\n"
         "    if (ls) ucuda_lyap_chain<AMOUNTOFX>(K, continuation, nPts, lo, hi, reverse, logScale, mutParamIdx,\n"
         "        baseValues, amountOfValues, baseX, *P, axisKind, tolRatio, tTr, NT, nBlocks, nWarm, eps, renorm,\n"
         "        maxValue, result, stats, cancel, progress);\n"
         "    else    ucuda_lyap_chain<1>(K, continuation, nPts, lo, hi, reverse, logScale, mutParamIdx,\n"
         "        baseValues, amountOfValues, baseX, *P, axisKind, tolRatio, tTr, NT, nBlocks, nWarm, eps, renorm,\n"
         "        maxValue, result, stats, cancel, progress);\n"
         "}\n";
    if (dense) o << R"CPU(
// БД 1D: классика — calculateDiscreteModelPeaksAdCUDA по точкам [i0, i1), continuation —
// calculateDiscreteModelPeaksAdContCUDA строка в строку.
extern "C" __declspec(dllexport)
int ucuda_cpu_ad_bif(int continuation, int i0, int i1, int nPts, double lo, double hi, int reverse,
    int logScale, int sweepVar, int mutIdx, const double* baseValues, int amountOfValues, const double* baseX,
    const UcudaAdaptParams* Pbase, int axisKind, double tolRatio, int writableVar, double maxValue,
    double* outPeaks, double* timeOfPeaks, int* flags, unsigned long long peakStride, int peakCapacity,
    double transientTime, double tRec, double dtOut, int preScaller, unsigned long long iters, int raw,
    int interp, double* adStats, const volatile int* cancelFlag, int* progress) {
    const UcudaKrsFns K{};
    UcudaAdProgress prog;
    prog.init(nullptr, 0, (numb)0);
    const numb dtS = (numb)dtOut * (numb)preScaller;
    if (!continuation) {
        ucuda_cpu_par_or_var = sweepVar ? 0 : 1;
        const numb ranges[2] = { (numb)lo, (numb)hi };
        const int  mut[1]    = { mutIdx };
        const int  kinds[2]  = { axisKind, UCUDA_AXIS_SYSTEM };
        numb localX[AMOUNTOFX];
        numb localValues[64];   // kMaxAmountOfValues в движке
        for (int idx = i0; idx < i1; ++idx) {
            if (cancelFlag != nullptr && *cancelFlag != 0) return 0;
            const int row = idx - i0;
            UcudaAdaptParams P = *Pbase;
            ucudaSetupSweepPointAd(nPts, 0, idx, 1, ranges, mut, baseX, baseValues, amountOfValues,
                logScale ? 1 : 0, kinds, (numb)tolRatio, localX, localValues, P);
            UcudaAdaptState S;
            ucuda_ad_init(S, K, AMOUNTOFX, localX, (numb)0, localValues, P);
            PeakStream   pu;
            PeakStreamNU pn;
            if (!raw) pu.init(outPeaks, timeOfPeaks, (size_t)row * peakStride, dtS, (size_t)iters, peakCapacity);
            else      pn.init(outPeaks, timeOfPeaks, (size_t)row * peakStride, peakCapacity, interp);
            const int flag  = ucudaAdPeaksPoint(S, K, localValues, P, (numb)transientTime, (numb)tRec, dtS,
                (size_t)iters, raw, preScaller, writableVar, (numb)maxValue, pu, pn, cancelFlag, prog);
            const int count = raw ? pn.count() : pu.count();
            flags[row] = (flag == REGIME_OSCILLATION) ? count : flag;
            ucudaAdWriteStats(adStats, row, S);
            if (progress != nullptr) ++*progress;
        }
        return (cancelFlag != nullptr && *cancelFlag != 0) ? 0 : 1;
    }
    numb x[AMOUNTOFX];
    numb a[64];
    for (int i = 0; i < AMOUNTOFX; ++i) x[i] = baseX[i];
    for (int i = 0; i < amountOfValues && i < 64; ++i) a[i] = baseValues[i];
    UcudaAdaptParams P = *Pbase;
    UcudaAdaptState S;
    for (int j = 0; j < nPts; ++j) {
        if (cancelFlag != nullptr && *cancelFlag != 0) return 0;
        if (progress != nullptr) ++*progress;
        const numb v = ucuda_node_value_cont(j, nPts, (numb)lo, (numb)hi, logScale != 0, reverse != 0);
        if (axisKind == UCUDA_AXIS_SYSTEM) a[mutIdx] = v;
        else ucudaAdApplyStepAxis(axisKind, v, (numb)tolRatio, P);
        if (j == 0) ucuda_ad_init(S, K, AMOUNTOFX, x, (numb)0, a, P);
        else        ucuda_ad_restart(S, K, (numb)0, a, P);
        PeakStream   pu;
        PeakStreamNU pn;
        if (!raw) pu.init(outPeaks, timeOfPeaks, (size_t)j * peakStride, dtS, (size_t)iters, peakCapacity);
        else      pn.init(outPeaks, timeOfPeaks, (size_t)j * peakStride, peakCapacity, interp);
        const int flag  = ucudaAdOut(S.X, (numb)maxValue) ? REGIME_UNBOUND
                        : ucudaAdPeaksPoint(S, K, a, P, (numb)transientTime, (numb)tRec, dtS,
                                            (size_t)iters, raw, preScaller, writableVar, (numb)maxValue, pu, pn,
                                            cancelFlag, prog);
        const int count = raw ? pn.count() : pu.count();
        flags[j] = (flag == REGIME_OSCILLATION) ? count : flag;
        ucudaAdWriteStats(adStats, j, S);
    }
    return (cancelFlag != nullptr && *cancelFlag != 0) ? 0 : 1;
}
)CPU";
    return o.str();
}

bool read_kernel_header(const char* name, std::string& out, std::vector<KrsCpuDiag>& diags) {
    const std::string path = exe_dir() + "\\kernels\\" + name;
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) {
        diags.push_back({ 0, "cannot read " + path });
        return false;
    }
    char buf[4096];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, k);
    fclose(f);
    return true;
}

} // namespace

AdaptiveCpuModule::~AdaptiveCpuModule() {
    if (module_) FreeLibrary((HMODULE)module_);
}

bool AdaptiveCpuModule::compile(const std::string& rhs, const std::string& emb, const std::string& dprep,
                                const std::string& deval, const std::string& ctrl_body, int amountOfX,
                                const std::string& prelude, std::vector<KrsCpuDiag>& diags) {
    if (module_) { FreeLibrary((HMODULE)module_); module_ = nullptr; }
    endpoint_ = nullptr; lyap_ = nullptr; bif_ = nullptr;
    std::string why;
    if (vcvars_path(why).empty()) {
        diags.push_back({ 0, "CPU compiler unavailable: " + why });
        return false;
    }
    std::string header, config, part, lib;
    if (!read_kernel_header("ucuda_adaptive.cuh", header, diags)) return false;
    if (!read_kernel_header("configCUDA.h", config, diags)) return false;
    if (!read_kernel_header("adaptive_part.cu", part, diags)) return false;
    if (!read_kernel_header("cudaLibrary.cu", lib, diags)) return false;
    const std::string peaks = extract_peak_stream(lib);
    if (peaks.empty()) {
        diags.push_back({ 0, "struct PeakStream not found in kernels\\cudaLibrary.cu" });
        return false;
    }
    // Тексты adaptive_part.cu и PeakStream входят в исходник, а с ним — в ключ кэша.
    const std::string source = make_ad_module_source(rhs, emb, dprep, deval, ctrl_body, amountOfX, prelude,
                                                     peaks, strip_bom(part));
    unsigned long long h = 1469598103934665603ULL;      // FNV-1a
    auto mix = [&](const void* p, size_t n) {
        const unsigned char* b = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    };
    const char tag[] = "adaptive-module";
    mix(tag, sizeof tag);
    mix(source.data(), source.size());
    mix(header.data(), header.size());
    mix(config.data(), config.size());
    const int ver = kAdModuleVersion;
    mix(&ver, sizeof ver);

    std::lock_guard<std::mutex> lock(g_compile_mtx);
    std::string dll;
    if (!build_cached_dll(source, h, dll, diags)) return false;
    HMODULE m = LoadLibraryA(dll.c_str());
    if (!m) { diags.push_back({ 0, "failed to load " + dll }); return false; }
    auto pe = GetProcAddress(m, "ucuda_cpu_ad_endpoint");
    auto pl = GetProcAddress(m, "ucuda_cpu_ad_lyap");
    if (!pe || !pl) {
        FreeLibrary(m);
        diags.push_back({ 0, "the built DLL lacks the adaptive-step entry points" });
        return false;
    }
    module_ = m;
    endpoint_ = (EndpointFn)pe;
    lyap_ = (LyapFn)pl;
    bif_ = (BifFn)GetProcAddress(m, "ucuda_cpu_ad_bif");   // только у модуля с плотным выходом
    return true;
}

CtrlCpuFn::~CtrlCpuFn() {
    if (module_) FreeLibrary((HMODULE)module_);
}

bool CtrlCpuFn::compile(const std::string& body, std::vector<KrsCpuDiag>& diags) {
    if (module_) { FreeLibrary((HMODULE)module_); module_ = nullptr; }
    fn_ = nullptr;

    std::string why;
    if (vcvars_path(why).empty()) {
        diags.push_back({ 0, "CPU compiler unavailable: " + why });
        return false;
    }
    std::string header;
    {
        const std::string path = exe_dir() + "\\kernels\\ucuda_adaptive.cuh";
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) {
            diags.push_back({ 0, "cannot read " + path });
            return false;
        }
        char buf[4096];
        size_t k;
        while ((k = fread(buf, 1, sizeof buf, f)) > 0) header.append(buf, k);
        fclose(f);
    }

    std::lock_guard<std::mutex> lock(g_compile_mtx);
    std::string dll;
    if (!build_cached_dll(make_ctrl_source(body), ctrl_hash_key(body, header), dll, diags))
        return false;
    HMODULE m = LoadLibraryA(dll.c_str());
    if (!m) {
        diags.push_back({ 0, "failed to load " + dll });
        return false;
    }
    auto p = GetProcAddress(m, "ucuda_ctrl_custom_c");
    if (!p) {
        FreeLibrary(m);
        diags.push_back({ 0, "the built DLL has no ucuda_ctrl_custom_c" });
        return false;
    }
    module_ = m;
    fn_ = (Fn)p;
    return true;
}

KrsCpuStep::~KrsCpuStep() { release(); }

KrsCpuStep::KrsCpuStep(KrsCpuStep&& o) noexcept
    : module_(o.module_), fn_(o.fn_), fn_dd_(o.fn_dd_), fn_qd_(o.fn_qd_) {
    o.module_ = nullptr; o.fn_ = nullptr; o.fn_dd_ = nullptr; o.fn_qd_ = nullptr;
}

KrsCpuStep& KrsCpuStep::operator=(KrsCpuStep&& o) noexcept {
    if (this != &o) {
        release();
        module_ = o.module_; fn_ = o.fn_; fn_dd_ = o.fn_dd_; fn_qd_ = o.fn_qd_;
        o.module_ = nullptr; o.fn_ = nullptr; o.fn_dd_ = nullptr; o.fn_qd_ = nullptr;
    }
    return *this;
}

void KrsCpuStep::release() {
    if (module_) FreeLibrary((HMODULE)module_);
    module_ = nullptr;
    fn_     = nullptr;
    fn_dd_  = nullptr;
    fn_qd_  = nullptr;
}

bool KrsCpuStep::compile(const std::string& body, int amountOfX, int amountOfValues,
                         std::vector<KrsCpuDiag>& diags) {
    return compile(body, amountOfX, amountOfValues, KrsCpuPrec::Double, diags);
}

bool KrsCpuStep::compile(const std::string& body, int amountOfX, int amountOfValues,
                         KrsCpuPrec prec, std::vector<KrsCpuDiag>& diags) {
    release();

    // Индексы проверяем ДО компиляции: в нативном коде выход за границу X[]
    // — это порча стека вызывающего, а не понятная ошибка.
    if (!krs_cpu_check_indices(body, amountOfX, amountOfValues, diags))
        return false;

    std::string why;
    const std::string vcvars = vcvars_path(why);
    if (vcvars.empty()) {
        diags.push_back({ 0, "CPU compiler unavailable: " + why });
        return false;
    }

    std::lock_guard<std::mutex> lock(g_compile_mtx);

    // Кэш: та же схема + та же размерность -> DLL уже собрана.
    const unsigned long long key = hash_key(body, amountOfX, amountOfValues, prec);
    std::string dll;
    if (!build_cached_dll(make_source(body, amountOfX, prec), key, dll, diags))
        return false;

    HMODULE m = LoadLibraryA(dll.c_str());
    if (!m) {
        diags.push_back({ 0, "failed to load " + dll });
        return false;
    }
    auto p = GetProcAddress(m, "krs_step");
    if (!p) {
        FreeLibrary(m);
        diags.push_back({ 0, "the built DLL has no krs_step" });
        return false;
    }
    module_ = m;
    switch (prec) {
        case KrsCpuPrec::QD: fn_qd_ = (StepFnQD)p; break;
        case KrsCpuPrec::DD: fn_dd_ = (StepFnDD)p; break;
        default:             fn_    = (StepFn)p;   break;
    }
    return true;
}
