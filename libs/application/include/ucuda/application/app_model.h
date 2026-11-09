#pragma once
#include "ucuda/application/system_editor.h"
#include "ucuda/domain/plot_window.h"
#include "ucuda/equations/codegen.hpp"
#include "ucuda/equations/sysparse.hpp"
#include "ucuda/platform/image_source.h"
#include "ucuda/domain/system_record.h"
#include "ucuda/analysis/analysis_session.h"
#include "ucuda/analysis/custom_session.h"
#include "ucuda/analysis/order_session.h"
#include "ucuda/analysis/network_session.h"
#include "ucuda/analysis/neural_session.h"
#include "ucuda/application/prewarm_watch.h"
#include "ucuda/application/time_estimate.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <future>
#include <mutex>
#include <atomic>
#include <deque>
#include <regex>
#include <algorithm>



// Один элемент общей parametric-очереди: какой анализ + индекс в его сессии.
// Очередь обслуживается AppModel::tick после poll. См. план
// «Cross-analysis batch Run all».
struct ParametricQueueItem {
    enum class Kind { Bifurcation, LLE, LS, Metrics };
    Kind kind = Kind::Bifurcation;
    int  index = 0;
};

// One dynamic plot window in Parametric mode (mirrors PhaseAnalysisSession's Projection, but must
// span 3 independent sessions — bifurcation_session / lle_session / ls_session — so it lives on
// AppModel rather than inside any one of them). Replaces the old fixed "Bifurcation 1D"/"LLE 1D"/
// "Lyapunov Spectrum" windows and the per-diagram bool `visible`: a window's `members` list is now
// the only source of "what's shown here", since the same diagram/curve can belong to zero, one, or
// several windows at once.


// Dynamic plot window for DFT1D mode — mirrors ParametricPlotWindow but simplified: only one
// display kind exists (heatmap), so no `kind`/`mode_2d`/`colored_1d` fields. `members` indexes
// dft1d_session.configs and always holds at most one entry — a DFT1D window shows exactly one
// config's heatmap (as Bifurcation's 2D/colored-1D windows do; single-select radio in the
// "Members..." popup, enforced again in add_dft1d_plot_window).


// Палитра БАЗОВЫХ цветов кривых окна Order (OrderPlotWindow::palette).
// Цвет кодирует вкладку-член окна, форма маркера — кривую внутри вкладки
// (min/avg/max, E1/E2), поэтому палитра тут нужна КАЧЕСТВЕННАЯ: соседние
// номера обязаны быть максимально далеки друг от друга. Градиентные карты
// хитмап (HeatmapColormap) для этого не годятся — они по построению непрерывны,
// и соседние кривые получают соседние оттенки.
//   kOrderPaletteApp      — ImPlot::GetColormapColor, как у фазовых портретов;
//   kOrderPaletteContrast — 12 насыщенных тонов средней светлоты, читаются и на
//                           светлой, и на тёмной теме плота;
//   kOrderPaletteOkabe    — Okabe-Ito: 8 цветов, различимых при всех трёх
//                           распространённых видах дальтонизма.


// Один элемент DFT1D-очереди — индекс config'а в dft1d_session.configs.
// Отдельная очередь по той же причине, что и у Basins/FastSync — DFT1D
// независим от parametric/basins по UI.
struct Dft1DQueueItem {
    int index = 0;
};

// Один элемент basins-очереди — индекс config'а в basins_session.configs. Basins живёт в отдельной
// очереди (а не в parametric_queue_), потому что параметрика и basins сейчас не пересекаются по UI:
// parametric "Run all" пушит BD/LLE/LS, а basins "Run all" — свои configs. Драйнится тем же
// движком, но независимым тиком в draw_gui.
struct BasinsQueueItem {
    int index = 0;
};

// Аналогично для FastSync: индекс config'а в fastsync_session.configs.
// Отдельная очередь по той же причине — FastSync независим от parametric/basins.
struct FastSyncQueueItem {
    int index = 0;
};

// Аналогично для Order: индекс config'а в order_session.configs. Своя очередь
// по той же причине, что у Basins/FastSync — вкладка независима по UI, а
// движок всё равно обслуживает её последовательно.
struct OrderQueueItem {
    int index = 0;
};

// Аналогично для Network: индекс config'а в network_session.configs.
struct NetworkQueueItem {
    int index = 0;
};
using NeuralQueueItem = NetworkQueueItem;

// Состояние ОТОБРАЖЕНИЯ вкладки Network. В конфиг ему нельзя: конфиг — это
// то, что уходит в расчёт, а выбранный узел, курсор времени и набор кривых
// результат не меняют (и в очередь "Run all" не должны попадать).
// Отображение вкладки Neural; в расчёт не входит, как у Network.
struct NeuralViewState {
    int series_var = 0;
    int instance = 0;    // экземпляр в окнах от времени
    int sync_kind = 0;   // 0 = R, 1 = разброс по узлам
    int sweep_kind = 0;  // Neural Sweep: то же, 2 = показатель Ляпунова, 3 = NRMSE выхода; у спайков 0 = частота, 1 = NRMSE
    bool spread_log[2] = { true, true };   // окна Synchronisation и Neural Sweep; у R оси всегда линейная
    int lyap_kind = 0;   // 0 = показатель от значения свипа, 1 = накопленное среднее одной сети
    int map_kind = 0;    // Neural Map: как sweep_kind
    int map_pop = 0;
    bool map_log = true; // разброс в log10: он меняется на 7 порядков
    std::string rate_bin_text = "0";   // окно Population Rate, 0 = авто
    int readout_target = 0;            // цель выхода в окнах Neural Readout, Neural Sweep и Neural Map
};

struct NetworkViewState {
    int   selected_node = -1;
    int   selected_edge = -1;
    int   color_var  = 0;       // переменная для раскраски графа и растра
    int   series_var = 0;       // переменная на кривых
    int   time_index = 0;       // точка, по которой раскрашен граф
    bool  playing = false;
    float play_fps = 60.0f;     // точек в секунду при проигрывании
    // Узел, из которого тянется новое ребро. Ставится двойным кликом по узлу
    // и держится до следующего клика (по другому узлу — ребро, по пустому
    // месту или Esc — отмена): протяжка с зажатой кнопкой на тачпаде и при
    // случайном срыве курсора теряла ребро на полпути.
    int   drag_from = -1;
    // Сколько узлов рисовать кривыми и с какого начинать: 256 линий в одном
    // плоте нечитаемы, а GPU-серии на них тратятся всерьёз.
    int   series_count = 8;
    int   first_series_node = 0;
    bool  show_labels = true;

    // Колормапа графа и растра. -1 = общий дефолт приложения; своя нужна
    // потому, что граф и хитмапы параметрики читают разные величины и
    // одинаковая шкала для них — совпадение, а не требование.
    int   graph_colormap = -1;
    int   series_colormap = -1;   // ею красятся кривые Node Series

    // Клик по узлу/ребру в графе просит таблицу прокрутиться к нему. Флаг
    // одноразовый: таблица гасит его в том же кадре, в котором прокрутилась.
    bool  scroll_nodes_to_selected = false;
    bool  scroll_edges_to_selected = false;

    // Стиль кривых Node Series. Живёт здесь, а не во вью: вью пересоздаётся
    // вместе с окном, а настройка должна пережить переключение вкладок.
    bool  custom_line_style = false;
    float line_width   = 1.5f;
    bool  points_mode  = false;
    int   point_marker = 0;
    float point_size   = 3.0f;
    float line_alpha   = 1.0f;
    bool  node_markers = false;
};


// Отложенный запрос «скопировать диаграмму в буфер» (правый клик "Copy image to clipboard", см.
// plot_axis.h request_plot_screenshot). Разбирается один раз в главном цикле app_main.cpp: сам
// glReadPixels ждёт пару кадров после запроса, потому что right-click popup закрывается только на
// СЛЕДУЮЩЕМ кадре (семантика ImGui) — иначе он попадёт в захваченную картинку.
struct PendingScreenshot {
    bool  active = false;
    int   frames_left = 0;
    float min_x = 0, min_y = 0, max_x = 0, max_y = 0; // экранные координаты ImGui
};

// Поле настроек, которое ПКМ по нему умеет разослать по вкладкам
// (см. AppModel::broadcast_field). Имя параметра/переменной передаётся
// отдельной строкой — в enum'е ему места нет.
enum class BroadcastField {
    Param,          // значение параметра по имени
    InitCondition,  // начальное условие по имени переменной
    StepH,
    SymmetryS,
    CdFpIters,      // число простых итераций CD FP (a[M+1])
    TMax,           // computing time
    Transient,      // transient time
    PreScaller,     // decimator
    MaxValue,
    Scheme,         // численный метод
    // Свип. Первая ось есть у Bif / LLE / LS / DFT и у оси X общего config'а
    // Custom'а; вторая — только там, где есть 2D-режим. У Basins и FastSync
    // оси сетки — это начальные условия, а не параметрический свип, и у Order
    // своя кодировка цели, поэтому они эти поля пропускают.
    SweepLo,  SweepHi,
    SweepLo2, SweepHi2,
    SweepTarget,  SweepTarget2,   // что свипуется: параметр / НУ / шаг
    Resolution,     // число узлов свипа; у Basins/FastSync — сторона сетки
};

// Расчётные вкладки в том порядке, в котором их обходит рассылка. Живёт в
// заголовке, потому что адресом цели (вкладка + индекс конфига) оперирует и UI:
// в меню можно выбрать, в какие именно конфиги применять.
enum class BroadcastTab {
    Phase, Bifurcation, LLE, LS, Dft1D, Basins, FastSync, Custom, Order, Metrics
};

// Одна цель рассылки — КОНКРЕТНЫЙ конфиг конкретной вкладки, а не вкладка
// целиком: диаграмм в одной вкладке много, и "применить ко всем Bifurcation 1D"
// адресует именно их.
struct BroadcastTarget {
    BroadcastTab tab = BroadcastTab::Phase;
    int          idx = 0;     // индекс конфига; у Phase и Custom он один, idx = 0
    std::string  label;       // подпись конфига: "BD 1", "LLE 2", ...
    std::string  group;       // тип: "Bifurcation 1D", "LLE 2D", "Basins", ...
};

// Цель свипа одной оси в том виде, в каком её рассылает broadcast_field:
// "par:<i>" — параметр, "var:<i>" — начальное условие, "h" — шаг, "s" —
// коэффициент симметрии a[0]. Четыре поля описывают ОДНО решение
// пользователя, и по строковому интерфейсу рассылки они обязаны ехать
// вместе — половина применённой цели это не цель. Нужна и UI (ПКМ по комбо
// свипа передаёт сюда текущий выбор), и самой рассылке.
std::string encode_sweep_target(int par_index, bool over_var, int var_index, bool over_h);

// Одна запись стека отмены. restore самодостаточен и БЕЗОПАСЕН к перестройке
// вкладок: он захватывает сессии (это члены AppModel, они не переезжают) плюс
// ИНДЕКСЫ конфигов, а адреса полей переразрешает в момент отката — векторы
// конфигов съезжают, когда диаграмму добавляют или закрывают. Там же
// сверяется label конфига: если на записанном индексе оказался другой, поле
// пропускается, а не переписывается вслепую.
struct UndoRecord {
    std::string           description;   // что откатится — видно в меню
    std::function<void()> undo;
    std::function<void()> redo;
};

// Модель приложения: всё состояние + логика. НЕ знает про ImGui.
// UI читает поля и вызывает методы; долгие операции идут в фоне.
class SystemLibrary;

class AppModel : public SystemEditor {
public:
    void tick(SystemLibrary& library);
    // Черновик редактора библиотеки — только редактор системы, без сессий и очередей.
    // Конструктор и деструктор в app_queues.cpp: там ParametricEngine полный тип.
    explicit AppModel(OcrFn ocr);
    ~AppModel();

    // Платформенное состояние screenshot-to-clipboard, см. PendingScreenshot.
    PendingScreenshot pending_screenshot;
    // Экспорт netlist'а откладывается до конца кадра по той же причине, что и
    // скриншот: платформенный модальный диалог крутит СВОЙ цикл сообщений, и
    // вызванный изнутри кадра ImGui он приводит к рекурсивной отрисовке —
    // кадр начинается заново, не закончив предыдущий, состояние ImGui рушится.
    bool pending_netlist_export = false;

    // Окно библиотеки регуляторов шага (draw_ctrl_library_window в gui.cpp). Сами записи —
    // в adaptive_ctrl_library() (adaptive_settings.h), здесь только состояние окна.
    bool        show_ctrl_library   = false;
    int         ctrl_lib_sel        = -1;       // выбранная запись
    bool        ctrl_lib_dirty      = false;    // правка ещё не записана в файл
    std::string ctrl_lib_status;                // ошибка записи файла
    std::string ctrl_lib_check_body;            // тело, для которого сделан Check,
    std::string ctrl_lib_check_log;             //   и его результат
    bool        ctrl_lib_check_ok   = false;
    // Check идёт в фоне (NVRTC — около секунды, UI не ждёт): результат — ok и лог компилятора.
    std::future<std::pair<bool, std::string>> ctrl_lib_check_future;
    float       ctrl_lib_body_h     = 300.0f;

    // Library-tab edit state (in-memory only, not persisted to session)
    // Library sub-tab has two states: a plain list (None) and an editor
    // (EditExisting for the "Edit" action, AddNew for "Add new system").
    // The editor edits into library_edit_buffer (a scratch SystemEditor), never
    // into this model directly, so opening/typing/Cancel never touches the
    // globally active system used by Parametric/Phase/Basins/FastSync. Save
    // validates, rename+saves (EditExisting) or saves (AddNew) the buffer to
    // disk, and only mirrors it into this model if edit_original_name is the
    // currently active system (see library_editor_save in gui.cpp).
    enum class LibraryEditMode { None, EditExisting, AddNew };
    LibraryEditMode library_edit_mode = LibraryEditMode::None;
    std::string     library_selected_name;   // row selected in the list (for note preview)
    std::string     edit_original_name;      // on-disk name at the moment Edit was pressed
    std::string     edit_error;              // inline validation / rename error in the editor
    std::unique_ptr<SystemEditor> library_edit_buffer; // scratch editor the Library editor actually edits

    // режим приложения и сессия анализа (слой 2)
    // режим верхнего уровня: библиотека, фазовый анализ, параметрический,
    // бассейны притяжения или настройки.
    // Order добавлен В КОНЕЦ намеренно: last_app_mode сериализуется в app_config
    // целым числом, и вставка в середину сбросила бы всем сохранённую вкладку.
    // Порядок кнопок в UI задаётся отдельно (см. draw_gui).
    enum class AppMode { Library, Analysis, Parametric, Dft1D, Basins, FastSync, Custom, Settings, Order, Network, Neural };
    AppMode app_mode = AppMode::Library;

    // Число элементов AppMode. Нужно для клампа last_app_mode из конфига:
    // литерал в app_main.cpp отставал от enum'а (стоял 8 при девяти режимах,
    // из-за чего Order не восстанавливался на старте).
    static constexpr int kAppModeCount = 11;

    // Номер пересоздания сессий вкладки (++ в start_*_analysis). Сессия грузится с
    // data_generation = 0, а static-вью вкладки помнят поколение прошлой: по смене номера gui сбрасывает их.
    int session_epoch[kAppModeCount] = {};

    // сессия анализа фазовых портретов ("песочница": изменения не сохраняются)
    PhaseAnalysisSession phase_session;

    // Parametric mode держит три независимых анализа (1D bif / LLE / LS),
    // переключаемых верхними табами в Parametric Controls. Каждый —
    // песочница: своя копия системы, свои Run-кнопки, свой плот, свой
    // _last_*.json. Результаты сессии не share'ят между собой.
    BifurcationAnalysisSession bifurcation_session;
    LLEAnalysisSession         lle_session;
    LyapunovSpectrumAnalysisSession ls_session;
    // Metrics — max/min/mean, частоты по пикам и параметры Хьорта по свипу.
    SignalMetricsAnalysisSession metrics_session;

    // 1D DFT — отдельный AppMode (между Parametric и Basins), своя очередь и
    // список конфигов (см. Dft1DAnalysisSession в analysis_session.h).
    Dft1DAnalysisSession       dft1d_session;

    // Basins of attraction — отдельный AppMode со своим Run и 5-плотным окном.
    BasinsAnalysisSession      basins_session;

    // Fast Synchro — отдельный AppMode (anti-synchro analysis на attractor / IC grid).
    FastSyncAnalysisSession    fastsync_session;

    // Order — оценка порядка точности схемы (p по Эйткену/Ричардсону).
    OrderAnalysisSession       order_session;

    // Network — сеть связанных осцилляторов на задаваемой топологии.
    NetworkSession             network_session;
    NetworkViewState           network_view;
    // Neural — большие сети: популяции и проекции, kernels/neural.template.cu.
    NeuralSession              neural_session;
    NeuralViewState            neural_view;

    // Custom pipeline — hierarchical 2D → 1D → Phase/Basins with a shared
    // config layer and drill-down click bindings. Owns its OWN Bif/LLE/LS/
    // Phase/Basins sub-sessions (see CustomSession); never shares state with
    // parametric_/phase_/basins_session above.
    CustomSession              custom_session;

    // 0=Bifurcation, 1=LLE, 2=LS, 3=Metrics. Активный sub-tab; используется для top-bar
    // indicator. По умолчанию — Bifurcation.
    int parametric_active_analysis = 0;

    // Непустая строка = последняя загрузка сохранённой сессии не прошла
    // (битый / обрезанный _last_*.json). Показывается в шапке и держится до
    // следующей успешной загрузки. Пишет apply_session_json в gui.cpp.
    std::string session_load_warning;

    // ---- Команды UI. Движок и очереди закрыты (private): UI не создаёт движок, не запускает
    // в нём сессии и не правит очереди сам, а только через эти методы (app_queues.cpp).
    // run_*: запуск конфига idx на движке (создаётся лениво); false, если сессия отказала
    // (причина в её last_error).
    bool run_bifurcation(int idx);
    bool run_lle(int idx);
    bool run_ls(int idx);
    bool run_metrics(int idx);
    bool run_dft1d(int idx);
    bool run_basins(int idx);
    bool recluster_basins(int idx);
    bool run_fastsync(int idx);
    bool run_order(int idx);
    bool run_network(int idx);
    bool run_neural(int idx, bool resume = false);
    // Сброс движка вместе с кэшами модулей (смена системы в Custom). Сначала дожидается
    // фонового прогрева: его поток держит ссылку на движок.
    void reset_engine();

    void enqueue_parametric(ParametricQueueItem item);
    void enqueue_dft1d(int idx);
    void enqueue_basins(int idx);
    void enqueue_fastsync(int idx);
    void enqueue_order(int idx);
    void enqueue_network(int idx);
    void enqueue_neural(int idx);
    void enqueue_custom(CustomQueueItem item);
    // Уровни Custom-конвейера ставит в очередь сама сессия (CustomSession::enqueue_level_*).
    void enqueue_custom_level_2d();
    void enqueue_custom_level_1d();
    void enqueue_custom_level_3();
    void enqueue_custom_level_1d_partial(bool x_slices, bool y_slices);
    void clear_neural_queue();
    void clear_custom_queue();
    void clear_all_queues();

    // Чтение очередей (размер для подписей, содержимое для тестов).
    const std::deque<ParametricQueueItem>& parametric_queue() const { return parametric_queue_; }
    const std::deque<Dft1DQueueItem>&      dft1d_queue()      const { return dft1d_queue_; }
    const std::deque<BasinsQueueItem>&     basins_queue()     const { return basins_queue_; }
    const std::deque<FastSyncQueueItem>&   fastsync_queue()   const { return fastsync_queue_; }
    const std::deque<OrderQueueItem>&      order_queue()      const { return order_queue_; }
    const std::deque<NetworkQueueItem>&    network_queue()    const { return network_queue_; }
    const std::deque<NeuralQueueItem>&     neural_queue()     const { return neural_queue_; }
    const std::deque<CustomQueueItem>&     custom_queue()     const { return custom_queue_; }

private:
    // движок параметрики (NVRTC). Лениво создаётся при первом Run.
    std::unique_ptr<ParametricEngine> parametric_engine;
    ParametricEngine& ensure_engine();
public:

    // Оценка времени счёта траекторий (time_estimate.h) — подпись справа от прогресс-бара.
    // Движку не принадлежит: цена шага — только NVRTC, калибровка — на первичном контексте.
    TrajTimeEstimator traj_estimator;
    // Объём работы ИДУЩЕГО расчёта: снимается один раз, в первом кадре после старта (по смене
    // времени старта), и дальше не пересчитывается — пользователь может уже править настройки
    // под следующий прогон.
    TrajWork traj_running_work;
    std::chrono::steady_clock::time_point traj_running_start{};
    // Какой анализ снят (BusyKind из gui.cpp) и ждёт ли его время уточнения поправки
    // (TrajTimeEstimator::learn — только LLE, LS, Basins).
    int  traj_running_kind   = 0;
    bool traj_learn_pending  = false;
    // Объём работы по настройкам открытой вкладки, в простое. Пересобирается не чаще раза в
    // 0.3 с: сборка запроса гоняет кодген.
    TrajWork traj_idle_work;
    std::chrono::steady_clock::time_point traj_idle_at{};

    // Фоновая компиляция модулей для ЕЩЁ НЕ ЗАПУЩЕННЫХ элементов очереди: пока текущая задача
    // занимает GPU, NVRTC занимает свободное ядро CPU, и к своей очереди модуль уже готов.
    // Объявлено ПОСЛЕ движка: деструктор future блокируется до конца потока, а поток держит
    // ссылку на движок — значит future должен умереть первым (члены рушатся в обратном порядке).
    std::future<void> parametric_prewarm_future;
    static constexpr int kPrewarmAhead = 3;   // ограничение и по CPU, и по ёмкости пулов модулей
    void prewarm_rest_of_parametric_queue();

    // Прогрев ПЕРВОГО запуска: очередь помогает со второго элемента, а первый Run платит
    // компиляцию сам. Поэтому следим за настройками, влияющими на ключ модуля (схема, система,
    // вид свипа по осям), и прогреваем, пока пользователь ещё возится с диапазонами.
    PrewarmWatch prewarm_watch_bif, prewarm_watch_lle, prewarm_watch_ls, prewarm_watch_metrics;
    void poll_parametric_prewarm();

    // Cross-analysis batch queue. Run all... popup пушит сюда выбранные конфиги
    // BD/LLE/LS. start_next_in_parametric_queue() драйнит её серийно (engine один).
private:
    std::deque<ParametricQueueItem> parametric_queue_;
public:

    // Если ни одна из трёх сессий не in_flight и очередь не пуста — снимает
    // элемент с фронта и запускает run_async соответствующей сессии.
    // Возвращает true если что-то стартовало. Безопасно вызывать каждый кадр.
    bool start_next_in_parametric_queue();

    // Dynamic plot windows for Parametric mode (see ParametricPlotWindow).
    // Rendered by draw_parametric_plot_windows(); managed (add/remove/
    // rename/membership) by the "Plot windows" section in
    // draw_parametric_controls.
    std::vector<ParametricPlotWindow> parametric_plot_windows;
    int  next_parametric_plot_window_id = 1;
    // Bumped by "Reset windows layout" in draw_parametric_controls — mirrors
    // PhaseAnalysisSession::layout_generation, but shared across all 3 kinds (one button resets
    // every parametric plot window's docking at once). Goes into each window's ImGui title so
    // docking treats them as new windows after a reset. Not persisted (transient, like Phase's).
    int  parametric_layout_generation = 0;
    // Set by add/remove/membership edits on parametric_plot_windows; drained
    // once per frame in draw_gui to trigger a "_last_parametric_windows"
    // session save (avoids writing to disk every single frame).
    bool parametric_plot_windows_dirty = false;

    // Выбранная вкладка каждого узла дока, по режимам (AppMode): имена окон, которые были
    // выбраны в своих узлах, пока режим был на экране. Окна чужого режима не выводятся, ImGui
    // вынимает их из узлов, а при возвращении выбирает в узле окно, появившееся последним
    // (оно же получает фокус). track_dock_selection в gui.cpp восстанавливает выбор. Не пишется.
    std::map<int, std::vector<std::string>> dock_selected_by_mode;
    int  dock_prev_mode      = -1;   // режим прошлого кадра
    int  dock_restore_frames = 0;    // сколько кадров ещё восстанавливать выбор после входа
    int  dock_restore_budget = 0;    // сколько кадров восстановление ещё может продлеваться

    // Add a new plot window of (kind, mode_2d) with the given initial member
    // indices (into the matching session's diagrams/curves list). Assigns a
    // fresh id and a default "Plot N" label.
    void add_parametric_plot_window(ParametricPlotWindow::Kind kind, bool mode_2d,
                                     std::vector<int> initial_members,
                                     bool colored_1d = false);
    // Remove by position in parametric_plot_windows (from the "X" in the
    // Plot windows list, or a closed floating window).
    void remove_parametric_plot_window(int pos);

    // Load parametric_plot_windows from a previously-saved "_last_parametric_windows" JSON (empty
    // json = file never existed for this system). If empty, synthesizes one default window per
    // (kind, mode_2d) combo that has >=1 diagram/curve — reproducing the old "everything overlaid in
    // one shared plot per kind" behaviour as the day-one default. If json is present (even an empty
    // window list, meaning the user deliberately closed everything), respects it as-is.
    void load_or_init_parametric_plot_windows(const std::string& json);

    // DFT1D batch queue — независимая от parametric_queue_ (см. Dft1DQueueItem).
private:
    std::deque<Dft1DQueueItem> dft1d_queue_;
public:
    bool start_next_in_dft1d_queue();

    // Dynamic plot windows for DFT1D mode (see Dft1DPlotWindow) — mirrors
    // parametric_plot_windows, minus the kind/mode_2d/colored_1d dimensions.
    std::vector<Dft1DPlotWindow> dft1d_plot_windows;
    int  next_dft1d_plot_window_id = 1;
    int  dft1d_layout_generation = 0;
    bool dft1d_plot_windows_dirty = false;

    void add_dft1d_plot_window(std::vector<int> initial_members);
    void remove_dft1d_plot_window(int pos);
    void load_or_init_dft1d_plot_windows(const std::string& json);

    // Dynamic plot windows for Order mode (see OrderPlotWindow).
    std::vector<OrderPlotWindow> order_plot_windows;
    int  next_order_plot_window_id = 1;
    int  order_layout_generation = 0;
    bool order_plot_windows_dirty = false;
    // Автосохранение Order (autosave_order_session в gui.cpp): что и для какой
    // системы записано последним. Файл пишется, как только сериализация
    // расходится с записанной, — а не только по завершении расчёта.
    std::string order_autosave_target;
    std::string order_autosave_json;
    std::string order_windows_autosave_json;

    void add_order_plot_window(OrderPlotWindow::Kind kind, std::vector<int> initial_members);
    void remove_order_plot_window(int pos);
    void load_or_init_order_plot_windows(const std::string& json);

    // Basins batch queue — независимая от parametric_queue_ (см. BasinsQueueItem).
private:
    std::deque<BasinsQueueItem> basins_queue_;
public:
    bool start_next_in_basins_queue();

    // FastSync batch queue — независимая (см. FastSyncQueueItem).
private:
    std::deque<FastSyncQueueItem> fastsync_queue_;
public:
    bool start_next_in_fastsync_queue();

    // Order batch queue — независимая (см. OrderQueueItem).
private:
    std::deque<OrderQueueItem> order_queue_;
public:
    bool start_next_in_order_queue();

    // Network batch queue — независимая (см. NetworkQueueItem).
private:
    std::deque<NetworkQueueItem> network_queue_;
public:
    bool start_next_in_network_queue();
private:
    std::deque<NeuralQueueItem> neural_queue_;
public:
    bool start_next_in_neural_queue();

    // Custom-tab pipeline queue — drives the 2D→1D→Phase/Basins ordering.
    // Items are pushed by the Run buttons and by drill-down clicks; drained
    // sequentially by start_next_in_custom_queue() so the shared engine
    // processes them one at a time (like parametric_queue_).
private:
    std::deque<CustomQueueItem> custom_queue_;
public:
    bool start_next_in_custom_queue();

    // Удаление конфига + чистка очереди (убрать item с этим индексом, у
    // оставшихся того же kind сдвинуть index > i на -1). Используется GUI
    // вместо прямого session.remove_*, чтобы очередь не указывала на
    // удалённые слоты.
    void remove_bifurcation_diagram(int i);
    void remove_lle_curve(int i);
    void remove_ls_curve(int i);
    void remove_metrics_config(int i);
    void remove_dft1d_config(int i);
    void remove_basins_config(int i);
    void remove_fastsync_config(int i);
    void remove_order_config(int i);
    void remove_network_config(int i);
    void remove_neural_config(int i);

    // Подготовить сессию анализа из ТЕКУЩЕЙ системы (после refresh_symbols).
    // Копирует параметры/НУ в сессию; изменения в сессии не идут в библиотеку.
    bool start_phase_analysis();
    // Инициализирует ВСЕ parametric-сессии (bif/LLE/LS) из текущей системы —
    // пользователь ждёт одинаковые vars/params в любом верхнем табе.
    bool start_parametric_analysis();
    // Инициализирует dft1d-сессию из текущей системы.
    bool start_dft1d_analysis();
    // Инициализирует basins-сессию из текущей системы.
    bool start_basins_analysis();
    bool start_order_analysis();
    bool start_network_analysis();
    bool start_neural_analysis();
    bool start_fastsync_analysis();
    // Initialise the Custom pipeline: seeds shared config from the record and
    // populates each sub-session with the fixed 3-slot layout (2D/1D-X/1D-Y).
    bool start_custom_analysis();

    // Пробрасывает живое состояние модели (custom_schemes, sys, vars/params) во ВСЕ сессии без
    // сброса per-config настроек. Используется в Save / Save as copy: после правки тела
    // custom-схемы или уравнений системы следующий Run в любом режиме подхватит свежее состояние
    // без переоткрытия вкладки. Phase'у дополнительно сбрасывается krs_code, чтобы regenerate_krs
    // пересчитался от актуального cs.body / sys.
    void propagate_to_sessions();

    // Кладёт value в одноимённое поле АКТИВНОГО конфига каждой расчётной
    // вкладки: Phase, Bifurcation, LLE, LS, DFT, Basins, FastSync, Custom
    // (общий Shared config) и Order. Соседние конфиги ВНУТРИ вкладки не
    // трогает — разные диаграммы одной вкладки на то и заведены, чтобы
    // отличаться.
    // name — имя параметра (Param) или переменной (InitCondition), для
    // остальных полей игнорируется. Вкладка, у которой такого поля нет
    // (у FastSync нет НУ, у Order — transient'а и decimator'а), пропускается
    // молча.
    // apply = false — сухой прогон: ничего не пишет. Возвращает число полей,
    // расходящихся с value (при apply = true — сколько изменил). Меню
    // показывает это число: соседние вкладки не открыты, и без него неясно,
    // сделала команда что-нибудь или нет.
    // label — человекочитаемое имя поля ("h", "computing time", "sigma").
    // Идёт только в подпись записи отмены: у безымянных полей name пустой, и
    // без label в меню Ctrl+Z стояло бы "0.01 applied to all tabs".
    //
    // only — куда писать. nullptr = активный конфиг КАЖДОЙ вкладки (прежнее
    // поведение). Непустой список = ровно эти конфиги: все того же типа либо
    // отмеченные галочками — решает вызывающий. scope_note уходит в подпись
    // отмены, чтобы "sigma = 10 -> all Bifurcation 1D" отличалось от "-> all tabs".
    int broadcast_field(BroadcastField f, const std::string& name,
                        const std::string& value, bool apply,
                        const std::string& label = {},
                        const std::vector<BroadcastTarget>* only = nullptr,
                        const std::string& scope_note = "all tabs");

    // Все конфиги, у которых ЕСТЬ поле f. Порядок — по вкладкам, внутри вкладки
    // по индексу; на нём же строятся маска выбора и подписи в меню.
    [[nodiscard]] std::vector<BroadcastTarget> broadcast_targets(BroadcastField f);

    // Какой конфиг редактирует открытая сейчас панель настроек. Ставит сама
    // панель в начале отрисовки; ПКМ-рассылка берёт отсюда тип ("все
    // Bifurcation 1D") и саму исходную цель. Через app_mode это не выводится:
    // вкладка Parametric покрывает сразу Bifurcation, LLE и LS.
    BroadcastTab broadcast_source_tab = BroadcastTab::Phase;
    int          broadcast_source_idx = 0;

    // Состояние выпадающего списка "Apply to...": галочки по broadcast_targets().
    // Живёт в модели, а не в static внутри draw — immediate-mode UI своё
    // состояние между кадрами не держит (см. CLAUDE.md).
    std::vector<char> broadcast_pick;
    // Подпись СОСТАВА списка, для которого собрана маска (поле + склейка имён
    // целей). Сравнивать один размер нельзя: добавив одну диаграмму и закрыв
    // другую, получаем прежнюю длину при другом составе — и галочки молча
    // указывают не на те конфиги.
    std::string       broadcast_pick_sig;

    // То же для списка "Calculation range to..." в меню диаграммы: галочки по
    // общему перечислению целей Ctrl+T.
    std::vector<char> vr_pick;
    std::string       vr_pick_sig;

    // ---- Отмена массовых команд -------------------------------------------
    // Сюда попадают ровно две команды: "Calculation range from view" (Ctrl+T /
    // Ctrl+Alt+T) и "Apply to all calculation tabs". Обе пишут разом в
    // несколько полей, и вернуть прежнее руками = перенабрать по всем вкладкам.
    // Обычные правки полей в стек НЕ идут: внутри активного InputText свой
    // undo у ImGui (stb_textedit), и вторая механика на той же клавише только
    // мешала бы. Поэтому и хоткей снаружи ловится лишь тогда, когда фокуса в
    // поле нет.
    static constexpr size_t kUndoDepth = 200;

    // Стек на КАЖДУЮ вкладку: Ctrl+Z в Parametric не должен трогать то, что
    // сделали в Basins. Команда попадает в стек вкладки, ОТКРЫТОЙ в момент её
    // выполнения (app_mode), — включая "Apply to all calculation tabs": та
    // пишет во все вкладки, но принадлежит той, где её вызвали, и отменяется
    // оттуда же целиком.
    std::map<AppMode, std::deque<UndoRecord>> undo_stacks;
    std::map<AppMode, std::deque<UndoRecord>> redo_stacks;

    // Подпись последнего отката/повтора + момент времени (ImGui::GetTime) —
    // шапка показывает её пару секунд. Без этого Ctrl+Z вслепую страшнее, чем
    // его отсутствие: вкладка, которую откатили, может быть сейчас не видна.
    std::string undo_note;
    double      undo_note_time = 0.0;

    // Вкладки, где последним действием был откат или повтор и команда
    // "диапазон" после него ещё не выполнялась. Ctrl+Shift+T в таком состоянии
    // берёт значения ИЗ ПОЛЕЙ, а не из вида: отмена вернула поля, но картинка
    // осталась зумленной, и рассылка "из вида" затёрла бы восстановленное тем,
    // что осталось на экране.
    // Флаг по вкладкам, как и сами стеки: откат в Parametric не должен менять
    // смысл хоткея в Basins. Сбрасывается новой командой в ЭТОЙ вкладке.
    std::map<AppMode, bool> undo_just_happened_tabs;

    // То же про ТЕКУЩУЮ вкладку — так его и спрашивает UI.
    [[nodiscard]] bool undo_just_happened() const;

    // redo может быть пустым (команда без обратного хода) — тогда запись
    // отменяется, но не повторяется.
    void push_undo(std::string description, std::function<void()> undo,
                   std::function<void()> redo);

    // Откат / повтор в ТЕКУЩЕЙ вкладке. false = соответствующий стек пуст.
    bool undo_last();
    bool redo_last();

    // Описания того, что сделает следующий Ctrl+Z / Ctrl+Shift+Z в текущей
    // вкладке, или nullptr (стек пуст).
    [[nodiscard]] const std::string* undo_top() const;
    [[nodiscard]] const std::string* redo_top() const;
};
