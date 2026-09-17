#include "engine_impl.h"
#include "engine_helpers.h"

namespace ucuda::cuda_detail {



// Phase 2: реальная реализация. NVRTC компилит наш шаблон, который через #include
// подтягивает cudaLibrary.cuh / cudaLibrary.cu из NonLinAnal. User's KRS определена
// в шаблоне, default-версия в NonLinAnal закрыта #ifndef __CUDACC_RTC__.
// Host-оркестрация мирорит bifurcation1D из hostLibrary.cu (без chunking для MVP).






















// Глобальный PeakConfig — единственный источник истины для GUI-настраиваемых knobs
// configCUDA.h. Пишет UI-поток (Settings), читают worker-потоки при компиляции NVRTC и в
// CPU-ветках, поэтому под мьютексом. Epoch бампается на каждую запись и входит в cache-key
// модулей (hash_key) — иначе после смены настройки переиспользовался бы старый PTX.


// Строка опции для nvrtcCompileProgram. Литералы статические, поэтому указатель
// живёт дольше вызова.
const char* nvrtc_fmad_opt() {
    return get_nvrtc_fmad() ? "--fmad=true" : "--fmad=false";
}


// gpu_free_budget — свободная память GPU с применённым запасом; общий первый шаг всех расчётов
// чанкования. Возвращает false, если cudaMemGetInfo недоступен (вызывающие превращают это в
// свой fail("cudaMemGetInfo failed")).
//
// `reserve` — доля свободной памяти, которую анализу разрешено занять; единого значения нет и
// сведено оно намеренно НЕ было:
//     0.92  — bifurcation 1D/2D, DFT и h-свипы (сверху ещё SAFETY_FACTOR 0.9 и вычет
//             memConstants у вызывающего, т.е. фактически ~0.83)
//     0.9   — basins
//     0.5   — LLE 1D/2D и fastsync-сетка
//     1/16  — LS 1D/2D: per-system память там ~N (буферы возмущённых траекторий), но 1/16 всё
//             равно агрессивнее, чем следует из одной этой оценки.
// Происхождение 0.5 и 1/16 не восстановлено — это следы отладки проблем с памятью, а не расчёт,
// поэтому числа сохранены как есть. Менять только после замера на реальных сетках; выигрыш
// реальный (LS сейчас берёт шестнадцатую часть памяти и делает в разы больше чанков).
bool gpu_free_budget(double reserve, size_t& out_bytes) {
    size_t freeMemory = 0, totalMemory = 0;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess) return false;
    out_bytes = (size_t)((double)freeMemory * reserve);
    return true;
}


// Число шагов интегрирования из времени: transient_time / h, NT / h и т.п.
//
// Прямой (int)(t / h) — UB, когда частное не влезает в int: TT=1e5 при h=1e-5 дают 1e10. На
// практике получался мусор или отрицательное число, т.е. транзиент молча не отрабатывал вовсе —
// худший вид ошибки, потому что расчёт выглядел успешным.
//
// Две версии, и выбор между ними НЕ косметический: он определяется типом приёмника. Значение
// уходит в ядро через void*[] в cuLaunchKernel, где тип обязан совпадать с параметром БАЙТ В
// БАЙТ — компилятор там ничего не проверит, рассогласование соберётся молча и развалит буфер
// аргументов на запуске.
//   ..._size_t — число ШАГОВ интегрирования; вся цепочка расширена до size_t (параметры ядер
//                amountOfPointsForSkip, device-функции loopCalculateDiscreteModel / _int со
//                счётчиками циклов, CPU-порт cpu_loop_model). Потолка нет — рабочая версия.
//   ..._int    — только под settleBlocks = TT / NT: это число NT-БЛОКОВ, а не шагов, оно живёт в
//                host-side счётчиках и на порядки меньше. INT_MAX здесь недостижим на осмысленных
//                входах, но каст всё равно идёт через проверку, а не вслепую.
size_t steps_from_time_size_t(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (size_t)s;
}


int steps_from_time_int(double t, double h)
{
    if (!(h > 0.0) || !(t > 0.0)) return 0;
    const double s = t / h;
    if (!std::isfinite(s)) return 0;
    return (s >= 2147483647.0) ? 2147483647 : (int)s;
}


// Генератор случайных чисел — побайтовая копия stub'а из шаблонов
// kernels/lle1d.template.cu (splitmix64-инициализация + LCG). Копия, а не
// повторная реализация: направления начального возмущения обязаны совпадать
// с GPU-веткой, иначе первые точки λ разошлись бы из-за разного «щупа».
// Значение свипа в точке j continuation-цепочки. Одна формула на все три
// анализа (Bif/LLE/LS) и на оба устройства — GPU-ядра считают её же.
// reverse — идём от hi к lo (гистерезис), log_scale — логарифмическая сетка.
// Тело — общая с ядрами реализация (ucuda_node_value_cont в configCUDA.h):
// та же функция, которую после правки зовут kernels/*_cont.template.cu.
// Раньше здесь стояла копия в double, а ядро считало в numb — в double-режиме
// одно и то же, при numb = float расходились.
double cont_sweep_value(int j, int nPts, double lo, double hi,
                               bool reverse, bool log_scale) {
    return (double)ucuda_node_value_cont(j, nPts, (numb)lo, (numb)hi, log_scale, reverse);
}


// Request/Result — публичный интерфейс в double, а ядра работают в numb.
// Конвертация живёт ровно на границе: перед H2D сужаем, после D2H расширяем.
// Когда numb == double это no-op копия, поэтому обе конфигурации ведут себя
// одинаково и без #ifdef.
std::vector<numb> to_numb(const std::vector<double>& v) {
    return std::vector<numb>(v.begin(), v.end());
}


std::string exe_dir() { return platform::executable_dir(); }


std::string read_text_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "не удалось открыть " + path; return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    // UTF-8 BOM: 0xEF 0xBB 0xBF в начале — NVRTC от него спотыкается.
    if (s.size() >= 3 &&
        (unsigned char)s[0] == 0xEF &&
        (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF) {
        s.erase(0, 3);
    }
    // NVRTC не любит не-ASCII символы даже внутри комментариев (компилит как
    // device code, не предполагает многобайтовых юникод-точек). Заменяем все
    // байты со старшим битом на пробел — это безвредно для комментариев и
    // не задевает строковые литералы из ASCII.
    for (char& c : s) {
        if ((unsigned char)c >= 0x80) c = ' ';
    }
    return s;
}


std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}


std::string hash_key(const std::string& krs_body, int amountOfX) {
    // peak_config_epoch(): knobs configCUDA.h уходят в NVRTC как #define, т.е. при их смене тот
    // же КРС даёт другой PTX. Без epoch в ключе кэши модулей отдали бы старый модуль, и настройка
    // не применилась бы до перезапуска приложения.
    // fmad — то же самое, но через опцию компиляции. Здесь в ключ идёт САМО значение, а не счётчик
    // поколений: флаг бинарный, поэтому возврат к прежнему положению переиспользует уже собранный
    // модуль вместо лишней перекомпиляции.
    return std::to_string(std::hash<std::string>{}(krs_body)) + ":" +
           std::to_string(amountOfX) + ":pk" + std::to_string(peak_config_epoch()) +
           ":fm" + (get_nvrtc_fmad() ? "1" : "0");
}


// Блок #define'ов, дописываемый ПЕРЕД текстом виртуального configCUDA.h (тот оборачивает свои
// дефолты в #ifndef). Один инжект покрывает все 14 шаблонов — они тянут configCUDA.h транзитивно
// через cudaLibrary.cuh. Приведение к (numb) внутри макроса безопасно: разворачивается он только
// в местах использования (cudaLibrary.cu), т.е. заведомо после typedef numb.
std::string peak_config_defines() {
    const PeakConfig c = get_peak_config();
    std::ostringstream o;
    o << std::setprecision(17);
    o << "// --- injected by parametric_engine (GUI Settings) ---\n"
      << "#define doCalculatePeaks "    << (c.do_calculate_peaks   ? 1 : 0) << "\n"
      << "#define doInterpolatePeaks "  << (c.do_interpolate_peaks ? 1 : 0) << "\n"
      << "#define eps_fixed_point ((numb)"     << c.eps_fixed_point     << ")\n"
      << "#define eps_peak_delta ((numb)"      << c.eps_peak_delta      << ")\n"
      << "#define eps_interPeak_delta ((numb)" << c.eps_interPeak_delta << ")\n"
      << "#define peak_threshold ((numb)"      << c.peak_threshold      << ")\n"
      << "#define max_amount_of_peaks "        << c.max_amount_of_peaks << "\n";
    return o.str();
}


std::string cu_err(CUresult r) {
    const char* name = nullptr;
    cuGetErrorString(r, &name);
    return name ? std::string(name) : ("CUresult " + std::to_string((int)r));
}


// Локальная host-копия __device__ __host__ getValueByIdx из cudaLibrary.cu:1266.
// Для 1D-бифуркации valueNumber всегда 0, поэтому просто линейная интерполяция.
// Нужна для расчёта значения параметра в CSV (host-side).
double getValueByIdx_local(size_t idx, int nPts, double lo, double hi) {
    // Одна арифметика с ядром — ucuda_node_value (configCUDA.h). Раньше здесь стояло
    // lo + (hi-lo)*idx/(n-1): при idx = nPts-1 это не давало РОВНО hi, и CSV сообщал параметр, в
    // котором точка не считалась. Вырожденный nPts == 1 теперь тоже как в ядре (hi, а не lo).
    // Касты явные: при numb = float сетка обязана считаться во float — ровно так, как её увидит
    // ядро (ranges уходят туда через to_numb).
    return (double)ucuda_node_value((int)idx, nPts, (numb)lo, (numb)hi);
}


// Host-копия getValueByIdx_log из cudaLibrary.cu -- та же формула, что кернел
// реально использовал для этой точки при log_scale=true. Нужна, чтобы CSV
// (см. getValueByIdx_local) сообщал то самое значение параметра, что было
// просимулировано, а не линейную интерполяцию поверх log-распределённой сетки.
double getValueByIdx_log_local(size_t idx, int nPts, double lo, double hi) {
    // См. getValueByIdx_local: общая формула вместо копии. Здесь копия ещё и
    // умножала до деления, а ядро (getValueByIdxLog) — наоборот.
    return (double)ucuda_node_value_log((int)idx, nPts, (numb)lo, (numb)hi);
}
}



    // Шаг тика прогресса. Кратен CHECK_INTERVAL, потому что тики ставятся в уже
    // существующей проверке расходимости, и подобран так, чтобы точка отчиталась
    // около 64 раз за свою работу: этого хватает на гладкий бар и не создаёт
    // давки на одном адресе.
    int ParametricEngine::Impl::progress_stride_for(size_t stepsPerPoint) {
        int s = (int)(stepsPerPoint / 64);
        s -= s % kCheckInterval;
        if (s < kCheckInterval) s = kCheckInterval;
        return s;
    }


    ParametricEngine::Impl::~Impl() {
        if (inited) {
            cuCtxSetCurrent(context);
            release_module();
            release_lle_module();
            release_lle_2d_module();
            release_ls_module();
            release_ls_2d_module();
            release_cont_module();
            release_bif2d_module();
            release_basins_module();
            release_fs_attr_module();
            release_fs_grid_module();
            release_order_module();
            cuCtxDestroy(context);
        }
    }


    void ParametricEngine::Impl::release_fs_attr_module() {
        if (cached_fs_attr.module) {
            cuModuleUnload(cached_fs_attr.module);
            cached_fs_attr.module         = nullptr;
            cached_fs_attr.kernel_fs_fill = nullptr;
            cached_fs_attr.kernel_fs_traj = nullptr;
            cached_fs_attr.kernel_fs_grid = nullptr;
            cached_fs_attr.key.clear();
        }
    }

    void ParametricEngine::Impl::release_fs_grid_module() {
        if (cached_fs_grid.module) {
            cuModuleUnload(cached_fs_grid.module);
            cached_fs_grid.module         = nullptr;
            cached_fs_grid.kernel_fs_fill = nullptr;
            cached_fs_grid.kernel_fs_traj = nullptr;
            cached_fs_grid.kernel_fs_grid = nullptr;
            cached_fs_grid.key.clear();
        }
    }


    void ParametricEngine::Impl::release_order_module() {
        if (cached_order.module) {
            cuModuleUnload(cached_order.module);
            cached_order.module = nullptr;
            cached_order.kernel = nullptr;
            cached_order.key.clear();
        }
    }


    void ParametricEngine::Impl::release_module() {
        if (cached.module) {
            cuModuleUnload(cached.module);
            cached.module = nullptr;
            cached.kernel_traj = nullptr;
            cached.kernel_peak = nullptr;
            cached.kernel_fused = nullptr;
            cached.kernel_dft = nullptr;
            cached.key.clear();
        }
    }


    void ParametricEngine::Impl::release_lle_module() {
        if (cached_lle.module) {
            cuModuleUnload(cached_lle.module);
            cached_lle.module = nullptr;
            cached_lle.kernel_lle = nullptr;
            cached_lle.key.clear();
        }
    }


    void ParametricEngine::Impl::release_lle_2d_module() {
        if (cached_lle_2d.module) {
            cuModuleUnload(cached_lle_2d.module);
            cached_lle_2d.module = nullptr;
            cached_lle_2d.kernel_lle = nullptr;
            cached_lle_2d.key.clear();
        }
    }


    void ParametricEngine::Impl::release_ls_module() {
        if (cached_ls.module) {
            cuModuleUnload(cached_ls.module);
            cached_ls.module = nullptr;
            cached_ls.kernel_ls = nullptr;
            cached_ls.key.clear();
        }
    }


    void ParametricEngine::Impl::release_ls_2d_module() {
        if (cached_ls_2d.module) {
            cuModuleUnload(cached_ls_2d.module);
            cached_ls_2d.module = nullptr;
            cached_ls_2d.kernel_ls = nullptr;
            cached_ls_2d.key.clear();
        }
    }


    void ParametricEngine::Impl::release_simple_cont_module(CachedSimpleContModule& m) {
        if (m.module) {
            cuModuleUnload(m.module);
            m.module = nullptr;
            m.kernel = nullptr;
            m.key.clear();
        }
    }


    void ParametricEngine::Impl::release_cont_module() {
        if (cached_cont.module) {
            cuModuleUnload(cached_cont.module);
            cached_cont.module = nullptr;
            cached_cont.kernel_cont = nullptr;
            cached_cont.kernel_peak = nullptr;
            cached_cont.kernel_dft = nullptr;
            cached_cont.key.clear();
        }
    }


    void ParametricEngine::Impl::release_bif2d_module() {
        if (cached_bif2d.module) {
            cuModuleUnload(cached_bif2d.module);
            cached_bif2d.module        = nullptr;
            cached_bif2d.kernel_fused  = nullptr;
            cached_bif2d.kernel_dbscan = nullptr;
            cached_bif2d.key.clear();
        }
    }


    void ParametricEngine::Impl::release_basins_module() {
        if (cached_basins.module) {
            cuModuleUnload(cached_basins.module);
            cached_basins.module              = nullptr;
            cached_basins.kernel_fused        = nullptr;
            cached_basins.kernel_dbscan       = nullptr;
            cached_basins.kernel_search_fixed = nullptr;
            cached_basins.kernel_search_clear = nullptr;
            cached_basins.key.clear();
        }
    }


    bool ParametricEngine::Impl::ensure_init(std::string& err) {
        if (inited) return true;
        CUresult r = cuInit(0);
        if (r != CUDA_SUCCESS) { err = "cuInit: " + cu_err(r); return false; }
        r = cuDeviceGet(&device, 0);
        if (r != CUDA_SUCCESS) { err = "cuDeviceGet: " + cu_err(r); return false; }
#if CUDA_VERSION >= 13000
        r = cuCtxCreate(&context, nullptr, 0, device);
#else
        r = cuCtxCreate(&context, 0, device);
#endif
        if (r != CUDA_SUCCESS) { err = "cuCtxCreate: " + cu_err(r); return false; }
        cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        inited = true;
        return true;
    }


    bool ParametricEngine::Impl::load_sources(std::string& err) {
        // Перечитывать файлы с диска надо один раз, а вот текст configCUDA.h
        // пересобирать — на каждую смену peak-настроек (см. peak_config_defines).
        const uint64_t ep = peak_config_epoch();
        if (srcs_loaded && srcs_peak_epoch == ep) return true;
        if (srcs_loaded) {
            src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
            srcs_peak_epoch  = ep;
            return true;
        }
        std::string root = exe_dir() + "/kernels/";
        std::string e;
        src_template          = read_text_file(root + "bifurcation1d.template.cu",      e); if (!e.empty()) { err = e; return false; }
        src_template_cont     = read_text_file(root + "bifurcation1d_cont.template.cu", e); if (!e.empty()) { err = e; return false; }
        src_template_lle_cont = read_text_file(root + "lle1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_ls_cont  = read_text_file(root + "ls1d_cont.template.cu",          e); if (!e.empty()) { err = e; return false; }
        src_template_dft_cont = read_text_file(root + "dft1d_cont.template.cu",         e); if (!e.empty()) { err = e; return false; }
        src_template_dft_hsweep = read_text_file(root + "dft1d_hsweep.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_lle      = read_text_file(root + "lle1d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_lle_2d   = read_text_file(root + "lle2d.template.cu",              e); if (!e.empty()) { err = e; return false; }
        src_template_ls       = read_text_file(root + "ls1d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_ls_2d    = read_text_file(root + "ls2d.template.cu",               e); if (!e.empty()) { err = e; return false; }
        src_template_bif2d    = read_text_file(root + "bifurcation2d.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_basins   = read_text_file(root + "basins.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_template_fs_attr  = read_text_file(root + "fastsync_attr.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_fs_grid  = read_text_file(root + "fastsync_grid.template.cu",     e); if (!e.empty()) { err = e; return false; }
        src_template_order    = read_text_file(root + "order.template.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cu    = read_text_file(root + "cudaLibrary.cu",            e); if (!e.empty()) { err = e; return false; }
        src_cudaLibrary_cuh   = read_text_file(root + "cudaLibrary.cuh",           e); if (!e.empty()) { err = e; return false; }
        src_cudaMacros_cuh    = read_text_file(root + "cudaMacros.cuh",            e); if (!e.empty()) { err = e; return false; }
        src_adaptive_h = read_text_file(root + "adaptive_solver.h", e); if (!e.empty()) { err=e; return false; }
        src_configCUDA_h_raw  = read_text_file(root + "configCUDA.h",              e); if (!e.empty()) { err = e; return false; }
        src_configCUDA_h = peak_config_defines() + src_configCUDA_h_raw;
        srcs_peak_epoch  = ep;
        srcs_loaded = true;
        return true;
    }


    // build_module — общая часть всех compile_*_if_needed: подстановка плейсхолдеров в шаблон,
    // NVRTC-компиляция с едиными опциями, добыча mangled-имён и загрузка PTX. У вызывающих
    // различаются только шаблон, имя исходника и набор символов.
    // Ради этого всё и сведено: раньше блок был скопирован под каждый анализ, и когда --fmad стал
    // настройкой, флаг проставили в nvrtc_engine.cpp, а копии здесь остались на дефолте NVRTC —
    // карта и фазовый портрет считались разной арифметикой, и на фрактальной границе траектория
    // уходила в другой аттрактор (см. nvrtc_fmad_opt).
    // Требует выставленного контекста и уже загруженных источников (load_sources). При успехе
    // out_module загружен, а lowered содержит по одному имени на каждый вход name_exprs в том же
    // порядке. Символы, объявленные extern "C", в name_exprs передавать не нужно.
    bool ParametricEngine::Impl::build_module(const std::string& tmpl,
                      const char* src_name,
                      const std::vector<std::pair<std::string, std::string>>& subs,
                      const std::vector<const char*>& name_exprs,
                      CUmodule& out_module,
                      std::vector<std::string>& lowered,
                      std::string& err) {
        std::string src = tmpl;
        for (const auto& sub : subs) src = replace_all(src, sub.first, sub.second);
        for (const auto& sub : subs) if (sub.first == "{{KRS_BODY}}" && is_adaptive_body(sub.second)) {
            src = "#define UC_ADAPTIVE_ENSEMBLE 1\n" + src;
            src = replace_all(src, "#include \"cudaLibrary.cuh\"", "#include \"cudaLibrary.cuh\"\n" + adaptive_ensemble_function(sub.second, false) + adaptive_sync_function(sub.second));
        }

        // Виртуальные заголовки для NVRTC. curand_kernel.h-stub НЕ нужен:
        // inline-stub в шаблонах + `#define CURAND_KERNEL_H_` блокируют как
        // реальный header (по -I path), так и любой повторный inject.
        const char* header_sources[] = {
            src_cudaLibrary_cu.c_str(),
            src_cudaLibrary_cuh.c_str(),
            src_cudaMacros_cuh.c_str(),
            src_configCUDA_h.c_str(),
            src_adaptive_h.c_str(),
        };
        const char* header_names[] = {
            "cudaLibrary.cu",
            "cudaLibrary.cuh",
            "cudaMacros.cuh",
            "configCUDA.h",
            "adaptive_solver.h",
        };
        constexpr int n_headers = 5;

        nvrtcProgram prog = nullptr;
        nvrtcResult nr = nvrtcCreateProgram(&prog, src.c_str(), src_name,
                                            n_headers, header_sources, header_names);
        if (nr != NVRTC_SUCCESS) {
            err = std::string("nvrtcCreateProgram(") + src_name + "): " + nvrtcGetErrorString(nr);
            return false;
        }

        // Регистрируем kernel-имена ДО компиляции, чтобы потом через
        // nvrtcGetLoweredName достать их mangled-варианты для cuModuleGetFunction.
        for (const char* sym : name_exprs) nvrtcAddNameExpression(prog, sym);

        char arch[64];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", cc_major, cc_minor);

        // NVRTC по умолчанию не знает CUDA-include путей (math_constants.h,
        // curand_kernel.h и т.п.). Берём CUDA_PATH из окружения — его проставляет
        // CUDA Toolkit installer.
        const std::string include_dir = platform::cuda_include_dir();
        const std::string cuda_include_opt = include_dir.empty() ? "" : "-I" + include_dir;
        if (cuda_include_opt.empty()) {
            err = "переменная окружения CUDA_PATH не задана — NVRTC не найдёт math_constants.h "
                  "(установи CUDA Toolkit или задай CUDA_PATH=...)";
            nvrtcDestroyProgram(&prog);
            return false;
        }

        const std::string cccl_include_opt = "-I" + include_dir + "/cccl";
        std::string std_opt = "--std=c++17";
        const char* opts[] = { arch, std_opt.c_str(), "-default-device",
                               cuda_include_opt.c_str(), cccl_include_opt.c_str(), nvrtc_fmad_opt() };

        nr = nvrtcCompileProgram(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);
        if (nr != NVRTC_SUCCESS) {
            size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
            std::string log;
            if (logsz > 1) { log.resize(logsz); nvrtcGetProgramLog(prog, &log[0]); }
            err = std::string("NVRTC compile failed (") + src_name + "):\n" + log;
            nvrtcDestroyProgram(&prog);
            return false;
        }

        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        std::string ptx(ptxsz, '\0');
        nvrtcGetPTX(prog, &ptx[0]);

        // Mangled-имена копируем в свои строки ДО nvrtcDestroyProgram — после
        // destroy указатели становятся невалидны.
        lowered.clear();
        lowered.reserve(name_exprs.size());
        for (const char* sym : name_exprs) {
            const char* lo = nullptr;
            nvrtcGetLoweredName(prog, sym, &lo);
            lowered.push_back(lo ? lo : sym);
        }
        nvrtcDestroyProgram(&prog);

        // Грузим во временную переменную: при неудаче у вызывающего слот кэша
        // остаётся нетронутым, и release_*_module() не получит мусорный handle.
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr);
        if (r != CUDA_SUCCESS) {
            err = std::string("cuModuleLoadDataEx(") + src_name + "): " + cu_err(r);
            return false;
        }
        out_module = mod;
        return true;
    }


    // Достаёт функцию из загруженного модуля. Отдельная обёртка — чтобы у
    // вызывающих не размножалось одинаковое сообщение об ошибке.
    bool ParametricEngine::Impl::module_fn(CUmodule mod, const std::string& name, CUfunction& out, std::string& err) {
        CUresult r = cuModuleGetFunction(&out, mod, name.c_str());
        if (r != CUDA_SUCCESS) {
            err = "cuModuleGetFunction(" + name + "): " + cu_err(r);
            return false;
        }
        return true;
    }


    bool ParametricEngine::Impl::compile_if_needed(const std::string& krs_body, int amountOfX,
                           int par_or_var, std::string& err) {
        // Если worker-thread унаследовал чужой контекст (NvrtcEngine, например),
        // компиляция и загрузка модуля прицепят символы не в тот контекст. Жёстко
        // выставляем наш перед NVRTC/CU-вызовами.
        cuCtxSetCurrent(context);
        // par_or_var в kernel'е cudaLibrary.cu — compile-time макрос. Поэтому
        // включаем его в hash-key: param-sweep и IC-sweep кешируются отдельно.
        std::string key = hash_key(krs_body, amountOfX) + ":pov" + std::to_string(par_or_var);
        if (cached.module && cached.key == key) return true;  // cache hit
        release_module();

        if (!load_sources(err)) return false;

        // DFT_custom уже присутствует в этом же модуле (шаблон #include'ит
        // cudaLibrary.cu целиком) — регистрируем его тоже, чтобы run_dft_1d
        // мог переиспользовать этот кэш без отдельной компиляции.
        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template, "bifurcation1d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "calculateDiscreteModelCUDA", "peakFinderCUDA", "DFT_custom",
                            "calculateDiscreteModelPeaksCUDA" },
                          mod, mg, err))
            return false;

        cached.module = mod;
        if (!module_fn(mod, mg[0], cached.kernel_traj, err)) { release_module(); return false; }
        if (!module_fn(mod, mg[1], cached.kernel_peak, err)) { release_module(); return false; }
        if (!module_fn(mod, mg[2], cached.kernel_dft,  err)) { release_module(); return false; }
        if (!module_fn(mod, mg[3], cached.kernel_fused, err)) { release_module(); return false; }

        cached.key = key;
        return true;
    }


    // compile_lle_if_needed — отдельная компиляция для LLE-шаблона.
    // Ключ кэша = hash(krs_body + amountOfX + "lle"), чтобы PTX от bif1d
    // не путался с LLE даже при одной и той же KRS.
    bool ParametricEngine::Impl::compile_lle_if_needed(const std::string& krs_body, int amountOfX,
                               int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        // par_or_var в kernel — compile-time макрос (см. bif1d). Два PTX-модуля
        // кешируются отдельно: param-sweep и IC-sweep.
        std::string key = hash_key(krs_body, amountOfX) + ":lle:pov" + std::to_string(par_or_var);
        if (cached_lle.module && cached_lle.key == key) return true;
        release_lle_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_lle, "lle1d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "LLEKernelCUDA" },
                          mod, mg, err))
            return false;

        cached_lle.module = mod;
        if (!module_fn(mod, mg[0], cached_lle.kernel_lle, err)) { release_lle_module(); return false; }

        cached_lle.key = key;
        return true;
    }


    // compile_lle_2d_if_needed — отдельный PTX-кэш для LLE-2D. Структура та
    // же, что у compile_lle_if_needed; отличия только в исходнике шаблона
    // (lle2d.template.cu) и в маркере ключа кэша (":lle2d"). Сам kernel
    // (LLEKernelCUDA) — тот же, ловится по тому же имени.
    bool ParametricEngine::Impl::compile_lle_2d_if_needed(const std::string& krs_body, int amountOfX,
                                  int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":lle2d:pov" + std::to_string(par_or_var);
        if (cached_lle_2d.module && cached_lle_2d.key == key) return true;
        release_lle_2d_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_lle_2d, "lle2d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "LLEKernelCUDA" },
                          mod, mg, err))
            return false;

        cached_lle_2d.module = mod;
        if (!module_fn(mod, mg[0], cached_lle_2d.kernel_lle, err)) { release_lle_2d_module(); return false; }

        cached_lle_2d.key = key;
        return true;
    }


    // compile_ls_if_needed — третий шаблон. Ключ кэша помечен ":ls".
    bool ParametricEngine::Impl::compile_ls_if_needed(const std::string& krs_body, int amountOfX,
                              int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls:pov" + std::to_string(par_or_var);
        if (cached_ls.module && cached_ls.key == key) return true;
        release_ls_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_ls, "ls1d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "LSKernelCUDA" },
                          mod, mg, err))
            return false;

        cached_ls.module = mod;
        if (!module_fn(mod, mg[0], cached_ls.kernel_ls, err)) { release_ls_module(); return false; }

        cached_ls.key = key;
        return true;
    }


    // compile_ls_2d_if_needed — отдельный шаблон ls2d.template.cu, тот же kernel
    // LSKernelCUDA. Ключ кэша помечен ":ls2d:" — изолирован от ":ls:" slot'а.
    bool ParametricEngine::Impl::compile_ls_2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":ls2d:pov" + std::to_string(par_or_var);
        if (cached_ls_2d.module && cached_ls_2d.key == key) return true;
        release_ls_2d_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_ls_2d, "ls2d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "LSKernelCUDA" },
                          mod, mg, err))
            return false;

        cached_ls_2d.module = mod;
        if (!module_fn(mod, mg[0], cached_ls_2d.kernel_ls, err)) { release_ls_2d_module(); return false; }

        cached_ls_2d.key = key;
        return true;
    }


    // compile_bif1d_cont_if_needed — отдельный модуль (single-thread sequential
    // continuation kernel + peakFinderCUDA). Cache key с суффиксом :cont.
    bool ParametricEngine::Impl::compile_bif1d_cont_if_needed(const std::string& krs_body, int amountOfX,
                                      std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":cont";
        if (cached_cont.module && cached_cont.key == key) return true;
        release_cont_module();

        if (!load_sources(err)) return false;

        // bifurcation1dContinuationKernel — extern "C" (имя не мангается), поэтому в name_exprs не
        // идёт и берётся из модуля напрямую. peakFinderCUDA и DFT_custom — обычные C++ символы,
        // нужны mangled-варианты; DFT_custom уже в этом модуле (шаблон #include'ит cudaLibrary.cu
        // целиком) и нужен continuation-ветке run_dft_1d.
        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_cont, "bifurcation1d_cont.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body } },
                          { "peakFinderCUDA", "DFT_custom" },
                          mod, mg, err))
            return false;

        cached_cont.module = mod;
        if (!module_fn(mod, "bifurcation1dContinuationKernel", cached_cont.kernel_cont, err))
            { release_cont_module(); return false; }
        if (!module_fn(mod, mg[0], cached_cont.kernel_peak, err)) { release_cont_module(); return false; }
        if (!module_fn(mod, mg[1], cached_cont.kernel_dft,  err)) { release_cont_module(); return false; }

        cached_cont.key = key;
        return true;
    }


    // compile_simple_cont_if_needed — общий компилятор для одноядерных
    // continuation-модулей (LLE и LS). От compile_bif1d_cont_if_needed
    // отличается только тем, что регистрировать mangled-имена не нужно:
    // единственный нужный символ объявлен extern "C".
    bool ParametricEngine::Impl::compile_simple_cont_if_needed(CachedSimpleContModule& slot,
                                       const std::string& tmpl,
                                       const char* kernel_name,
                                       const char* src_name,
                                       const char* key_suffix,
                                       const std::string& krs_body, int amountOfX,
                                       std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + key_suffix;
        if (slot.module && slot.key == key) return true;
        release_simple_cont_module(slot);

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;   // пуст: единственный символ — extern "C"
        if (!build_module(tmpl, src_name,
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body } },
                          {},
                          mod, mg, err))
            return false;

        slot.module = mod;
        if (!module_fn(mod, kernel_name, slot.kernel, err)) {
            release_simple_cont_module(slot); return false;
        }
        slot.key = key;
        return true;
    }


    // compile_bif2d_if_needed — шаблон bifurcation2d.template.cu, два kernel'а:
    // calculateDiscreteModelPeaksCUDA + dbscanCUDA.
    // Cache key: ":bif2d:" + par_or_var.
    bool ParametricEngine::Impl::compile_bif2d_if_needed(const std::string& krs_body, int amountOfX,
                                 int par_or_var, std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":bif2d:" + std::to_string(par_or_var);
        if (cached_bif2d.module && cached_bif2d.key == key) return true;
        release_bif2d_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_bif2d, "bifurcation2d.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body },
                            { "{{PAR_OR_VAR}}",  std::to_string(par_or_var) } },
                          { "calculateDiscreteModelPeaksCUDA", "dbscanCUDA" },
                          mod, mg, err))
            return false;

        cached_bif2d.module = mod;
        if (!module_fn(mod, mg[0], cached_bif2d.kernel_fused,  err)) { release_bif2d_module(); return false; }
        if (!module_fn(mod, mg[1], cached_bif2d.kernel_dbscan, err)) { release_bif2d_module(); return false; }

        cached_bif2d.key = key;
        return true;
    }


    // compile_basins_if_needed — отдельный шаблон basins.template.cu, регистрирует 5 kernel'ов:
    // calculateDiscreteModelCUDA, avgPeakFinderCUDA и три DBSCAN-kernel'а (cluster,
    // search_fixed_points, search_clear_points). Cache-ключ помечен `:basins`; par_or_var=0
    // захардкожен в шаблоне, поэтому ключа не требует — только хэш krs_body/amountOfX.
    bool ParametricEngine::Impl::compile_basins_if_needed(const std::string& krs_body, int amountOfX,
                                  std::string& err) {
        cuCtxSetCurrent(context);
        std::string key = hash_key(krs_body, amountOfX) + ":basins";
        if (cached_basins.module && cached_basins.key == key) return true;
        release_basins_module();

        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> mg;
        if (!build_module(src_template_basins, "basins.cu",
                          { { "{{AMOUNT_OF_X}}", std::to_string(amountOfX) },
                            { "{{KRS_BODY}}",    krs_body } },
                          { "calculateDiscreteModelAvgPeaksCUDA",
                            "CUDA_dbscan_kernel",
                            "CUDA_dbscan_search_fixed_points_kernel",
                            "CUDA_dbscan_search_clear_points_kernel" },
                          mod, mg, err))
            return false;

        cached_basins.module = mod;
        if (!module_fn(mod, mg[0], cached_basins.kernel_fused,        err)) { release_basins_module(); return false; }
        if (!module_fn(mod, mg[1], cached_basins.kernel_dbscan,       err)) { release_basins_module(); return false; }
        if (!module_fn(mod, mg[2], cached_basins.kernel_search_fixed, err)) { release_basins_module(); return false; }
        if (!module_fn(mod, mg[3], cached_basins.kernel_search_clear, err)) { release_basins_module(); return false; }

        cached_basins.key = key;
        return true;
    }


    // FastSynchro — два режима.

    // Один helper для обоих режимов — параметризован шаблоном и списком
    // kernel-symbols. Ключ кэша включает все substituted-параметры
    // (type_of_synch, error_estim, fs_error_trs), чтобы при их смене recompile
    // действительно случился.
    bool ParametricEngine::Impl::compile_fs_module(const std::string& src_template,
                           const std::string& tag, // ":fs_attr" или ":fs_grid"
                           int amountOfX,
                           const std::string& krs_body,
                           int type_of_synch_v, int error_estim_v, double fs_error_trs_v,
                           const std::vector<const char*>& expr_kernels,
                           CachedFastSyncModule& slot,
                           std::string& err)
    {
        cuCtxSetCurrent(context);
        char trs_buf[64];
        std::snprintf(trs_buf, sizeof(trs_buf), "%.17g", fs_error_trs_v);
        std::string key = hash_key(krs_body, amountOfX) + tag
                        + ":t" + std::to_string(type_of_synch_v)
                        + ":e" + std::to_string(error_estim_v)
                        + ":r" + trs_buf;
        if (slot.module && slot.key == key) return true;
        if (slot.module) {
            cuModuleUnload(slot.module);
            slot.module = nullptr;
            slot.kernel_fs_fill = slot.kernel_fs_traj = slot.kernel_fs_grid = nullptr;
            slot.key.clear();
        }
        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> lowered;
        if (!build_module(src_template, "fastsync.cu",
                          { { "{{AMOUNT_OF_X}}",   std::to_string(amountOfX) },
                            { "{{TYPE_OF_SYNCH}}", std::to_string(type_of_synch_v) },
                            { "{{ERROR_ESTIM}}",   std::to_string(error_estim_v) },
                            { "{{FS_ERROR_TRS}}",  std::string(trs_buf) },
                            { "{{KRS_BODY}}",      krs_body } },
                          expr_kernels,
                          mod, lowered, err))
            return false;

        slot.module = mod;

        // Связываем по expr_kernels индексу.
        for (size_t i = 0; i < expr_kernels.size(); ++i) {
            CUfunction f = nullptr;
            if (!module_fn(mod, lowered[i], f, err)) {
                release_fs_attr_module(); release_fs_grid_module(); return false;
            }
            const std::string sym_name = expr_kernels[i];
            if      (sym_name == "fillFSMasterTrajectory")                        slot.kernel_fs_fill = f;
            else if (sym_name == "calculateDiscreteModelforFastSynchroCUDA")      slot.kernel_fs_traj = f;
            else if (sym_name == "calculateDiscreteModelICCforFastSynchro")       slot.kernel_fs_grid = f;
        }
        slot.key = key;
        return true;
    }


    bool ParametricEngine::Impl::compile_order_module(int amountOfX, int amountOfValues,
                              const std::string& krs_body, std::string& err)
    {
        cuCtxSetCurrent(context);
        const std::string key = hash_key(krs_body, amountOfX) + ":order:v"
                              + std::to_string(amountOfValues);
        if (cached_order.module && cached_order.key == key) return true;
        release_order_module();
        if (!load_sources(err)) return false;

        CUmodule mod = nullptr;
        std::vector<std::string> lowered;
        // orderEstimateKernel объявлено extern "C" — в name_exprs не нужно.
        if (!build_module(src_template_order, "order.cu",
                          { { "{{AMOUNT_OF_X}}",      std::to_string(amountOfX) },
                            { "{{AMOUNT_OF_VALUES}}", std::to_string(amountOfValues) },
                            { "{{KRS_BODY}}",         krs_body } },
                          {}, mod, lowered, err))
            return false;

        cached_order.module = mod;
        CUfunction f = nullptr;
        if (!module_fn(mod, "orderEstimateKernel", f, err)) { release_order_module(); return false; }
        cached_order.kernel = f;
        cached_order.key    = key;
        return true;
    }