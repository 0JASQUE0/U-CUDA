#include "gui_common.h"
#include "gui_inputs.h"   // get_exe_dir_with_sep
#include "dock_layout.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ucuda::ui_detail {
namespace {

// ---------------------------------------------------------------- запросы меню

enum class ReqKind { None, Preset, Apply, Save, Remove };
struct Request {
    int         mode   = -1;
    ReqKind     kind   = ReqKind::None;
    int         preset = -1;
    std::string name;
    bool        state_done = false;   // Apply: состояние вкладки уже восстановлено
    int         delay      = 0;       // Apply: кадров до раскладки (ждём новые окна)
};
Request     g_req;
std::string g_status;            // итог последнего действия — строка в меню
char        g_save_name[128] = "";
int         g_list_mode = -1;    // для какой вкладки собран g_saved
std::vector<std::string> g_saved;   // имена сохранённых раскладок вкладки g_list_mode

const char* const kPresetNames[kDockPresetCount] = { "Tabs", "Columns", "Rows", "Grid" };

// Встроенные раскладки: показываются в списке Saved, пока у пользователя нет файла с тем же
// именем (свой файл «Default» их перекрывает). Окна сопоставляются по подписи, так что
// суффикс системы в именах не нужен.
struct Builtin { int mode; const char* name; const char* text; };
// Вид окна в раскладке Default: все галки Show включены, оси не переставлены и не обращены.
#define PREFS_ALL "prefs=toolbar=1;legend=1;xlabel=1;xticks=1;ylabel=1;yticks=1;colorbar=1;axes3d=1;smooth=1;swap=0;invx=0;invy=0;\n"
const Builtin kBuiltins[] = {
    { (int)AppModel::AppMode::Custom, "Default",
      "name=Default\n"
      "tab=Custom\n"
      "state=l2d=1;l1d=1;l3=1;bif2d=1;lle2d=0;ls2d=1;bifx=1;bify=1;llex=0;lley=0;lsx=0;lsy=0;l3kind=0;\n"
      "split y 0.412094\n"
      "split x 0.500331\n"
      "split y 0.505464\n"
      "leaf 1 0\n"
      "window=Custom Bif 1D (X-slice)\n"
      "object=\n" PREFS_ALL
      "leaf 1 0\n"
      "window=Custom Bif 1D (Y-slice)\n"
      "object=\n" PREFS_ALL
      "split x 0.5\n"
      "leaf 1 0\n"
      "window=Projection 2\n"
      "object=[{\"label\":\"Projection 2\",\"type\":3,\"ax\":0,\"ay\":1,\"az\":2,\"cls\":false,\"lw\":1.5,\"al\":1,"
      "\"pts\":false,\"ps\":2,\"show_var\":[true,true,true]}]\n" PREFS_ALL
      "leaf 1 0\n"
      "window=Projection 1\n"
      "object=[{\"label\":\"Projection 1\",\"type\":0,\"ax\":0,\"ay\":1,\"az\":2,\"cls\":true,\"lw\":1.5,\"al\":1,"
      "\"pts\":false,\"ps\":2,\"show_var\":[true,true,true]}]\n" PREFS_ALL
      "split x 0.500331\n"
      "leaf 1 0\n"
      "window=Custom Bif 2D\n"
      "object=\n" PREFS_ALL
      "leaf 1 0\n"
      "window=Custom LS 2D\n"
      "object=\n" PREFS_ALL },
};
#undef PREFS_ALL

const Builtin* find_builtin(int mode, const std::string& name) {
    for (const Builtin& b : kBuiltins)
        if (b.mode == mode && name == b.name) return &b;
    return nullptr;
}

const char* mode_dir(int mode) {
    switch ((AppModel::AppMode)mode) {
    case AppModel::AppMode::Library:    return "Library";
    case AppModel::AppMode::Analysis:   return "Analysis";
    case AppModel::AppMode::Parametric: return "Parametric";
    case AppModel::AppMode::Dft1D:      return "Dft1D";
    case AppModel::AppMode::Basins:     return "Basins";
    case AppModel::AppMode::FastSync:   return "FastSync";
    case AppModel::AppMode::Custom:     return "Custom";
    case AppModel::AppMode::Settings:   return "Settings";
    case AppModel::AppMode::Order:      return "Order";
    case AppModel::AppMode::Network:    return "Network";
    case AppModel::AppMode::Neural:     return "Neural";
    }
    return "Other";
}

fs::path layouts_dir(int mode) {
    return fs::u8path(get_exe_dir_with_sep()) / "layouts" / mode_dir(mode);
}

// Имя файла из имени раскладки: запрещённые в Windows символы — на '_'.
std::string file_stem(const std::string& name) {
    std::string s = name;
    for (char& c : s)
        if (std::strchr("\\/:*?\"<>|", c) || (unsigned char)c < 32) c = '_';
    return s;
}

void refresh_saved(int mode) {
    g_list_mode = mode;
    g_saved.clear();
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(layouts_dir(mode), ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".layout") continue;
        // Отображаемое имя — из файла (в имени файла спецсимволы заменены).
        std::ifstream f(e.path());
        std::string line, name = e.path().stem().u8string();
        while (std::getline(f, line))
            if (line.rfind("name=", 0) == 0) { name = line.substr(5); break; }
        g_saved.push_back(name);
    }
    std::sort(g_saved.begin(), g_saved.end());
    // Встроенные — первыми, если их не перекрыл свой файл.
    for (auto it = std::rbegin(kBuiltins); it != std::rend(kBuiltins); ++it)
        if (it->mode == mode && std::find(g_saved.begin(), g_saved.end(), it->name) == g_saved.end())
            g_saved.insert(g_saved.begin(), it->name);
}

bool has_file(int mode, const std::string& name) {
    std::error_code ec;
    return fs::exists(layouts_dir(mode) / fs::u8path(file_stem(name) + ".layout"), ec);
}

// ---------------------------------------------------------------- дерево раскладки

// Узел сохранённой раскладки: разрез (axis 0 — слева/справа, 1 — сверху/снизу; ratio — доля
// первого ребёнка) или лист со списком окон-вкладок.
struct LNode {
    bool split = false;
    int  axis  = 0;
    float ratio = 0.5f;
    std::unique_ptr<LNode> a, b;
    std::vector<std::string> windows;
    std::vector<std::string> objects;   // настройки объекта окна (DockWindowFactory), "" — нет
    std::vector<std::string> prefs;     // вид окна: галки Show, swap, инверсия (plot_window_prefs_*)
    int selected = 0;
};

// JSON объекта — в одну строку файла раскладки.
std::string one_line(std::string s) {
    for (char& c : s) if (c == '\n' || c == '\r') c = ' ';
    return s;
}

float axis_len(const ImGuiDockNode* n, int axis) { return axis == 0 ? n->Size.x : n->Size.y; }

std::unique_ptr<LNode> capture(const ImGuiDockNode* n, const DockWindowFactory* f) {
    auto out = std::make_unique<LNode>();
    if (n->IsSplitNode()) {
        out->split = true;
        out->axis  = (int)n->SplitAxis;
        const float l0 = axis_len(n->ChildNodes[0], out->axis), l1 = axis_len(n->ChildNodes[1], out->axis);
        out->ratio = (l0 + l1) > 0 ? l0 / (l0 + l1) : 0.5f;
        out->a = capture(n->ChildNodes[0], f);
        out->b = capture(n->ChildNodes[1], f);
        return out;
    }
    // Порядок вкладок — как на панели вкладок узла; без неё — порядок окон узла.
    if (n->TabBar && n->TabBar->Tabs.Size > 0) {
        for (int i = 0; i < n->TabBar->Tabs.Size; ++i) {
            const ImGuiTabItem& t = n->TabBar->Tabs[i];
            if (!t.Window) continue;
            if (t.ID == n->TabBar->SelectedTabId) out->selected = (int)out->windows.size();
            out->windows.push_back(t.Window->Name);
        }
    } else {
        for (int i = 0; i < n->Windows.Size; ++i) out->windows.push_back(n->Windows[i]->Name);
    }
    for (const std::string& w : out->windows) {
        out->objects.push_back(f && f->describe ? one_line(f->describe(w)) : std::string());
        out->prefs.push_back(plot_window_prefs_to_string(w));
    }
    return out;
}

void write_tree(std::ostream& o, const LNode& n) {
    if (n.split) {
        // 9 значащих цифр — доля восстанавливает пиксель и на широком экране.
        o << "split " << (n.axis == 0 ? 'x' : 'y') << ' ' << std::setprecision(9) << n.ratio << '\n';
        write_tree(o, *n.a);
        write_tree(o, *n.b);
        return;
    }
    // На окно — три строки: имя ImGui, настройки его объекта (пусто — окно не создаётся) и вид
    // окна (галки меню: Show, Swap axes, Invert).
    o << "leaf " << n.windows.size() << ' ' << n.selected << '\n';
    for (size_t i = 0; i < n.windows.size(); ++i) {
        o << "window=" << n.windows[i] << '\n';
        o << "object=" << (i < n.objects.size() ? n.objects[i] : std::string()) << '\n';
        o << "prefs="  << (i < n.prefs.size() ? n.prefs[i] : std::string()) << '\n';
    }
}

std::unique_ptr<LNode> read_tree(std::istream& in, int depth = 0) {
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (depth > 64) return nullptr;
        auto n = std::make_unique<LNode>();
        if (line.rfind("split ", 0) == 0) {
            std::istringstream ss(line.substr(6));
            char ax = 'x';
            ss >> ax >> n->ratio;
            n->split = true;
            n->axis  = ax == 'y' ? 1 : 0;
            n->ratio = std::clamp(n->ratio, 0.02f, 0.98f);
            n->a = read_tree(in, depth + 1);
            n->b = read_tree(in, depth + 1);
            if (!n->a || !n->b) return nullptr;
            return n;
        }
        if (line.rfind("leaf ", 0) == 0) {
            std::istringstream ss(line.substr(5));
            size_t count = 0;
            ss >> count >> n->selected;
            // Строки окна: window= и за ним необязательные object= / prefs= (файлы старее
            // хранили не все). Читаем, пока идут строки этого листа.
            while (true) {
                const std::streampos at = in.tellg();
                if (!std::getline(in, line)) break;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("window=", 0) == 0) {
                    if (n->windows.size() == count) { in.seekg(at); break; }
                    n->windows.push_back(line.substr(7));
                    n->objects.resize(n->windows.size());
                    n->prefs.resize(n->windows.size());
                } else if (line.rfind("object=", 0) == 0 && !n->windows.empty()) {
                    n->objects.back() = line.substr(7);
                } else if (line.rfind("prefs=", 0) == 0 && !n->windows.empty()) {
                    n->prefs.back() = line.substr(6);
                } else {
                    in.seekg(at);   // начало следующего узла
                    break;
                }
            }
            n->objects.resize(n->windows.size());
            n->prefs.resize(n->windows.size());
            return n;
        }
        // name=, tab= и прочие заголовки — пропускаем
    }
    return nullptr;
}

// Сохранённое окно -> окно вкладки: полное имя ImGui, иначе видимая подпись (до "##"),
// иначе id (после "###"). Номера окон у другого пользователя могут не совпасть, подпись — да.
std::string label_of(const std::string& name) {
    const size_t p = name.find("##");
    return p == std::string::npos ? name : name.substr(0, p);
}
std::string id_of(const std::string& name) {
    const size_t p = name.find("###");
    return p == std::string::npos ? std::string() : name.substr(p + 3);
}

struct Matcher {
    std::vector<std::string> avail;   // ещё не размещённые окна вкладки
    std::string take(const std::string& saved) {
        auto pick = [&](auto pred) -> std::string {
            for (size_t i = 0; i < avail.size(); ++i)
                if (pred(avail[i])) { std::string r = avail[i]; avail.erase(avail.begin() + (long)i); return r; }
            return {};
        };
        std::string r = pick([&](const std::string& w) { return w == saved; });
        if (r.empty() && !id_of(saved).empty())
            r = pick([&](const std::string& w) { return id_of(w) == id_of(saved); });
        if (r.empty())
            r = pick([&](const std::string& w) { return label_of(w) == label_of(saved); });
        // Окно с тем же именем есть, но не на этом месте (другая вкладка рабочей области Custom,
        // плавающее) — тоже переезжает в раскладку.
        // Только живое окно: ImGui держит объект окна и после удаления проекции, и без
        // проверки удалённое окно считалось открытым — и не создавалось заново.
        if (r.empty())
            if (const ImGuiWindow* w = ImGui::FindWindowByName(saved.c_str()))
                if (w->LastFrameActive >= ImGui::GetFrameCount() - 1) r = saved;
        return r;
    }
};

struct BuildCtx {
    Matcher& m;
    const DockWindowFactory* factory = nullptr;
    int missing = 0;
    int created = 0;
    ImGuiID first_leaf = 0;
};

void build(ImGuiID node, ImVec2 size, const LNode& t, BuildCtx& c) {
    if (t.split) {
        ImGuiID a = 0, b = 0;
        ImGui::DockBuilderSplitNode(node, t.axis == 0 ? ImGuiDir_Left : ImGuiDir_Up, t.ratio, &a, &b);
        // Размеры детей — на той же базе, что у ImGui (DockNodeTreeUpdatePosSize): длина узла
        // минус разделитель, ячейка — округлённая доля этого остатка. Сохранённая доля — тоже
        // от остатка (capture: l0 / (l0 + l1)), поэтому круг «сохранить -> применить» даёт те
        // же пиксели. Раньше доля бралась от длины с разделителем и округлялась вниз; рядом с
        // центральным узлом dockspace ImGui берёт размер ячейки как есть, а не долю, и каждый
        // круг сдвигал границу на 1-2 px.
        const float full  = t.axis == 0 ? size.x : size.y;
        const float avail = std::max(0.0f, full - ImGui::GetStyle().DockingSeparatorSize);
        const float la    = std::round(avail * t.ratio);
        ImVec2 sa = size, sb = size;
        (t.axis == 0 ? sa.x : sa.y) = la;
        (t.axis == 0 ? sb.x : sb.y) = avail - la;
        ImGui::DockBuilderSetNodeSize(a, sa);
        ImGui::DockBuilderSetNodeSize(b, sb);
        build(a, sa, *t.a, c);
        build(b, sb, *t.b, c);
        return;
    }
    if (!c.first_leaf) c.first_leaf = node;
    // Выбранную вкладку стыкуем последней — ImGui делает активной последнюю пристыкованную.
    std::vector<size_t> order(t.windows.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    if (t.selected >= 0 && t.selected < (int)order.size() && order.size() > 1)
        std::rotate(order.begin() + t.selected, order.begin() + t.selected + 1, order.end());
    for (size_t i : order) {
        std::string w = c.m.take(t.windows[i]);
        // Окна нет — создаём его объект по сохранённым настройкам. Окно появится со
        // следующего кадра; DockBuilderDockWindow запоминает узел и для ещё не созданного.
        if (w.empty() && c.factory && c.factory->create && i < t.objects.size() && !t.objects[i].empty()) {
            w = c.factory->create(t.objects[i]);
            if (!w.empty()) ++c.created;
        }
        if (w.empty()) { ++c.missing; continue; }
        // Вид окна — по имени окна этой вкладки (у созданного окна его ещё нет, настройки ждут).
        if (i < t.prefs.size()) plot_window_prefs_from_string(w, t.prefs[i]);
        ImGui::DockBuilderDockWindow(w.c_str(), node);
    }
}

// ---------------------------------------------------------------- встроенные пресеты

// k равных частей узла node (размер size): первая — со стороны dir, остальные — за ней.
// Размер каждой части задаётся явно: доля 1/(k-i) от остатка не годится — ImGui отдаёт
// всё лишнее центральному узлу dockspace, и последняя ячейка выходила вдвое шире.
std::vector<ImGuiID> split_even(ImGuiID node, ImVec2 size, ImGuiDir dir, int k) {
    std::vector<ImGuiID> out;
    const bool horiz = dir == ImGuiDir_Left || dir == ImGuiDir_Right;
    const float cell = (horiz ? size.x : size.y) / (float)k;
    for (int i = 0; i < k - 1; ++i) {
        ImGuiID first = 0, rest = 0;
        ImGui::DockBuilderSplitNode(node, dir, 1.0f / (float)(k - i), &first, &rest);
        const float rest_len = cell * (float)(k - i - 1);
        ImGui::DockBuilderSetNodeSize(first, horiz ? ImVec2(cell, size.y) : ImVec2(size.x, cell));
        ImGui::DockBuilderSetNodeSize(rest,  horiz ? ImVec2(rest_len, size.y) : ImVec2(size.x, rest_len));
        out.push_back(first);
        node = rest;
    }
    out.push_back(node);
    return out;
}

void rebuild(ImGuiID dock_id, ImVec2 size, const char* side_window,
             const std::vector<ImGuiWindow*>& wins, int preset) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, size);
    ImGuiID plots = dock_id;
    ImVec2  plots_size = size;
    if (side_window) {
        ImGuiID right = 0;
        ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Right, 0.30f, &right, &plots);
        const float side_w = std::floor(size.x * 0.30f);
        plots_size = ImVec2(size.x - side_w, size.y);
        ImGui::DockBuilderSetNodeSize(right, ImVec2(side_w, size.y));
        ImGui::DockBuilderSetNodeSize(plots, plots_size);
        ImGui::DockBuilderDockWindow(side_window, right);
    }
    const int n = (int)wins.size();
    if (n > 0) {
        std::vector<ImGuiID> cells;
        switch (preset) {
        case kDockColumns: cells = split_even(plots, plots_size, ImGuiDir_Left, n); break;
        case kDockRows:    cells = split_even(plots, plots_size, ImGuiDir_Up, n);   break;
        case kDockGrid: {
            const int cols = (int)std::ceil(std::sqrt((double)n));
            const int rows = (n + cols - 1) / cols;
            const std::vector<ImGuiID> row_ids = split_even(plots, plots_size, ImGuiDir_Up, rows);
            const ImVec2 row_size(plots_size.x, plots_size.y / (float)rows);
            for (int r = 0; r < rows; ++r) {
                const int in_row = (std::min)(cols, n - r * cols);
                for (ImGuiID c : split_even(row_ids[r], row_size, ImGuiDir_Left, in_row)) cells.push_back(c);
            }
            break;
        }
        default: cells.assign((size_t)n, plots); break;   // Tabs: всё в один узел
        }
        for (int i = 0; i < n; ++i) ImGui::DockBuilderDockWindow(wins[i]->Name, cells[(size_t)i]);
    }
    ImGui::DockBuilderFinish(dock_id);
}

// ---------------------------------------------------------------- Grid «на месте»

bool has_window(const ImGuiDockNode* n, const char* name) {
    if (!name) return false;
    if (n->IsSplitNode()) return has_window(n->ChildNodes[0], name) || has_window(n->ChildNodes[1], name);
    for (int i = 0; i < n->Windows.Size; ++i)
        if (std::strcmp(n->Windows[i]->Name, name) == 0) return true;
    return false;
}

int count_leaves(const ImGuiDockNode* n, const char* skip) {
    if (n->IsSplitNode()) return count_leaves(n->ChildNodes[0], skip) + count_leaves(n->ChildNodes[1], skip);
    return has_window(n, skip) ? 0 : 1;
}

// Цепочка разрезов по одной оси — это одна «строка» (axis x) или «столбец» (axis y):
// её элементы — поддеревья, разрезанные по другой оси, или листы.
void chain_items(ImGuiDockNode* n, int axis, std::vector<ImGuiDockNode*>& out) {
    if (n->IsSplitNode() && (int)n->SplitAxis == axis) {
        chain_items(n->ChildNodes[0], axis, out);
        chain_items(n->ChildNodes[1], axis, out);
    } else {
        out.push_back(n);
    }
}

void set_len(ImGuiDockNode* n, int axis, float len) {
    ImVec2 s = n->Size;
    (axis == 0 ? s.x : s.y) = len;
    ImGui::DockBuilderSetNodeSize(n->ID, s);
}

// Размеры по цепочке снизу вверх: элементу — per (или его прежний размер, если в нём
// панель настроек), разрезу — сумма детей. Возвращает длину n по оси.
float assign_chain(ImGuiDockNode* n, int axis, float per, const char* side) {
    if (n->IsSplitNode() && (int)n->SplitAxis == axis) {
        const float l = assign_chain(n->ChildNodes[0], axis, per, side)
                      + assign_chain(n->ChildNodes[1], axis, per, side);
        set_len(n, axis, l);
        return l;
    }
    const float l = has_window(n, side) ? axis_len(n, axis) : per;
    set_len(n, axis, l);
    return l;
}

// Поддерево графиков: панель настроек (side) отделена разрезом от остального — спускаемся мимо неё.
ImGuiDockNode* plots_root(ImGuiDockNode* n, const char* side) {
    while (n->IsSplitNode()) {
        ImGuiDockNode* a = n->ChildNodes[0];
        ImGuiDockNode* b = n->ChildNodes[1];
        if (has_window(a, side) && count_leaves(a, side) == 0) n = b;
        else if (has_window(b, side) && count_leaves(b, side) == 0) n = a;
        else break;
    }
    return n;
}

// Графики разложены как Tabs, Rows или Columns: одна ячейка или одна цепочка ячеек по
// одной оси. Такую раскладку Grid перестраивает в сетку, а не выравнивает.
bool is_single_chain(ImGuiDockNode* n) {
    if (!n->IsSplitNode()) return true;
    std::vector<ImGuiDockNode*> items;
    chain_items(n, (int)n->SplitAxis, items);
    for (const ImGuiDockNode* it : items)
        if (it->IsSplitNode()) return false;
    return true;
}

// Выравнивает раскладку, не двигая окна: в каждой строке — равная ширина ячеек, в каждом
// столбце — равная высота. Панель настроек сохраняет свой размер.
void equalize(ImGuiDockNode* n, const char* side) {
    if (!n->IsSplitNode()) return;
    const int axis = (int)n->SplitAxis;
    std::vector<ImGuiDockNode*> items;
    chain_items(n, axis, items);
    float fixed = 0.0f;
    int   free_n = 0;
    for (ImGuiDockNode* it : items) {
        if (has_window(it, side)) fixed += axis_len(it, axis);
        else ++free_n;
    }
    if (free_n > 0) {
        const float per = std::floor((axis_len(n, axis) - fixed) / (float)free_n);
        if (per > 0) assign_chain(n, axis, per, side);
    }
    for (ImGuiDockNode* it : items) equalize(it, side);
}

}  // namespace

// ---------------------------------------------------------------- доли разрезов

namespace {
struct SplitMemo {
    ImGuiID c0 = 0, c1 = 0;   // дети, для которых запомнена доля (другие — разрез перестроен)
    float len = -1.0f;        // длина разреза по его оси в прошлом кадре
    float ratio = 0.5f;       // длина первого ребёнка / (длина - разделитель)
};
std::unordered_map<ImGuiID, SplitMemo>& split_memo() {
    static std::unordered_map<ImGuiID, SplitMemo> m;
    return m;
}

bool node_inside(const ImGuiDockNode* p, const ImGuiDockNode* n) {
    for (; p; p = p->ParentNode)
        if (p == n) return true;
    return false;
}

// Положения разделителей той же оси в дереве (начало полосы разделителя), кроме самого n и
// его потомков: те едут вместе с перетаскиваемой границей.
void collect_separators(const ImGuiDockNode* p, const ImGuiDockNode* n, int axis, std::vector<float>& out) {
    if (!p || p->IsLeafNode()) return;
    const ImGuiDockNode* a = p->ChildNodes[0];
    const ImGuiDockNode* b = p->ChildNodes[1];
    if (!a || !b) return;
    if (p != n && (int)p->SplitAxis == axis && a->IsVisible && b->IsVisible && !node_inside(p, n))
        out.push_back(a->Pos[axis] + a->Size[axis]);
    collect_separators(a, n, axis, out);
    collect_separators(b, n, axis, out);
}

// Притяжение перетаскиваемой границы разреза n: к разделителю той же оси в другой строке
// (столбце) — граница встаёт с ним в одну линию, перекрестием, — и к середине узла (деление
// пополам). Только в небольшом радиусе, дальше граница идёт за мышью. Возвращает длину
// первого ребёнка.
float snap_split(const ImGuiDockNode* n, const ImGuiDockNode* root, int axis, float avail, float c0, float min_each) {
    const float range = ImGui::GetFontSize() * 0.6f;
    float best = c0, best_d = range + 1.0f;
    auto consider = [&](float cand) {
        cand = std::round(cand);
        if (cand < min_each || cand > avail - min_each) return;
        const float d = std::fabs(cand - c0);
        if (d <= range && d < best_d) { best = cand; best_d = d; }
    };
    consider(avail * 0.5f);
    std::vector<float> seps;
    collect_separators(root, n, axis, seps);
    for (float s : seps) consider(s - n->Pos[axis]);
    return best;
}

// Узел n уже имеет итоговые Pos/Size (rewritten — их только что переписали мы, и детей надо
// переставить, даже если сам разрез не менялся). Повторяет распределение ImGui
// (DockNodeTreeUpdatePosSize): разделитель DockingSeparatorSize, у невидимого ребёнка всё
// место отдаётся видимому. SizeRef пишется, чтобы ImGui держал эти размеры и в следующих кадрах:
// рядом с центральным узлом он берёт SizeRef соседа как есть, без него — их отношение.
void keep_ratios(ImGuiDockNode* n, bool rewritten, const char* fixed, const ImGuiDockNode* root) {
    if (!n || n->IsLeafNode()) return;
    ImGuiDockNode* a = n->ChildNodes[0];
    ImGuiDockNode* b = n->ChildNodes[1];
    if (!a || !b) return;
    const int axis = (int)n->SplitAxis;
    if (axis != 0 && axis != 1) return;
    SplitMemo& m = split_memo()[n->ID];
    const float len = n->Size[axis];

    if (!(a->IsVisible && b->IsVisible)) {
        if (rewritten)
            for (ImGuiDockNode* c : { a, b }) { c->Pos = n->Pos; c->Size = n->Size; }
        m.len = len;
        keep_ratios(a, rewritten, fixed, root);
        keep_ratios(b, rewritten, fixed, root);
        return;
    }

    const ImGuiStyle& st = ImGui::GetStyle();
    const float avail = std::max(0.0f, len - st.DockingSeparatorSize);
    const bool same = m.c0 == a->ID && m.c1 == b->ID && m.len >= 0.0f;
    // Разрез, отделяющий панель настроек (лист с ней), — по правилам ImGui: панель держит
    // свою ширину в пикселях, когда меняется окно приложения.
    const bool fixed_split = fixed && ((a->IsLeafNode() && has_window(a, fixed)) ||
                                       (b->IsLeafNode() && has_window(b, fixed)));
    const bool changed = same && !fixed_split && std::fabs(len - m.len) > 0.5f && avail > 0.0f;
    const float min_each = std::trunc(std::min(avail, st.WindowMinSize[axis] * 2.0f) * 0.5f);
    float c0 = a->Size[axis];
    if (changed)
        c0 = std::clamp(std::round(avail * m.ratio), min_each, std::max(min_each, avail - min_each));
    else if (avail > 0.0f)
        m.ratio = c0 / avail;   // длина та же: доля — какой её сделали (в т.ч. перетаскиванием)
    m.len = len;
    m.c0 = a->ID;
    m.c1 = b->ID;

    const bool place = rewritten || (changed && c0 != a->Size[axis]);
    if (place) {
        ImVec2 sa = n->Size, sb = n->Size, pb = n->Pos;
        sa[axis] = c0;
        sb[axis] = avail - c0;
        pb[axis] += c0 + st.DockingSeparatorSize;
        a->Pos = n->Pos;  a->Size = sa;  a->SizeRef[axis] = sa[axis];
        b->Pos = pb;      b->Size = sb;  b->SizeRef[axis] = sb[axis];
    }
    keep_ratios(a, place, fixed, root);
    keep_ratios(b, place, fixed, root);
}
}  // namespace

namespace {
// ID разделителя разреза n — как его считает ImGui в DockNodeTreeUpdateSplitter: окно-хост
// dockspace (стек ID — только его ID), PushID(node->ID), GetID("##Splitter").
ImGuiID splitter_id(const ImGuiDockNode* n, const ImGuiWindow* host) {
    const int node_id = (int)n->ID;
    return ImHashStr("##Splitter", 0, ImHashData(&node_id, sizeof(node_id), host->ID));
}

ImGuiDockNode* find_active_split(ImGuiDockNode* n, const ImGuiWindow* host, ImGuiID active) {
    if (!n || n->IsLeafNode()) return nullptr;
    if (splitter_id(n, host) == active) return n;
    if (ImGuiDockNode* r = find_active_split(n->ChildNodes[0], host, active)) return r;
    return find_active_split(n->ChildNodes[1], host, active);
}
}  // namespace

void dock_snap_dragged_split(ImGuiID dock_id) {
    ImGuiContext& g = *GImGui;
    // Смещение захвата, которое ImGui поставил при нажатии: от него считается «естественное»
    // место границы под мышью. Своё на каждое перетаскивание.
    static ImGuiID s_active = 0;
    static float   s_offset = 0.0f;
    if (g.ActiveId != s_active) s_active = 0;
    if (g.ActiveId == 0) return;
    ImGuiDockNode* root = ImGui::DockBuilderGetNode(dock_id);
    if (!root || !root->HostWindow || g.ActiveIdWindow != root->HostWindow) return;
    ImGuiDockNode* n = find_active_split(root, root->HostWindow, g.ActiveId);
    if (!n) return;
    const ImGuiDockNode* a = n->ChildNodes[0];
    const ImGuiDockNode* b = n->ChildNodes[1];
    const int axis = (int)n->SplitAxis;
    if (!a || !b || !a->IsVisible || !b->IsVisible || (axis != 0 && axis != 1)) return;
    if (s_active == 0) { s_active = g.ActiveId; s_offset = g.ActiveIdClickOffset[axis]; }

    // SplitterBehavior ставит начало полосы разделителя в MousePos - ClickOffset + hover_extend
    // (hover_extend = WindowsBorderHoverPadding). Подменой ClickOffset граница встаёт туда, куда
    // её притянуло, — ImGui сам раскладывает окна и рисует полосу уже там, в этом же кадре.
    const ImGuiStyle& st = ImGui::GetStyle();
    const float ext   = g.WindowsBorderHoverPadding;
    const float mouse = g.IO.MousePos[axis];
    const float avail = std::max(0.0f, n->Size[axis] - st.DockingSeparatorSize);
    const float min_each = std::trunc(std::min(avail, st.WindowMinSize[axis] * 2.0f) * 0.5f);
    const float natural = mouse - s_offset + ext - n->Pos[axis];
    const float c0 = snap_split(n, root, axis, avail, natural, min_each);
    g.ActiveIdClickOffset[axis] = mouse + ext - (n->Pos[axis] + c0);
}

void dock_keep_split_ratios(ImGuiID dock_id, const char* fixed_window) {
    ImGuiDockNode* root = ImGui::DockBuilderGetNode(dock_id);
    keep_ratios(root, false, fixed_window, root);
}

// ---------------------------------------------------------------- API

void dock_layout_request(int mode, int preset) {
    g_req = Request{ mode, ReqKind::Preset, preset, {} };
}

void dock_layout_request_saved(int mode, const std::string& name, int delay_frames) {
    g_req = Request{ mode, ReqKind::Apply, -1, name };
    g_req.delay = delay_frames;
}

bool dock_layout_has_request(int mode) { return g_req.mode == mode && g_req.kind != ReqKind::None; }

void dock_layout_menu(int mode) {
    if (ImGui::IsWindowAppearing() || g_list_mode != mode) refresh_saved(mode);

    for (int p = 0; p < kDockPresetCount; ++p) {
        if (ImGui::MenuItem(kPresetNames[p]))
            g_req = Request{ mode, ReqKind::Preset, p, {} };
        if (p == kDockGrid && ImGui::IsItemHovered())
            ImGui::SetTooltip("From Tabs, Rows or Columns: lays the windows out in a grid.\n"
                              "From any other arrangement: windows stay where they are, cells of\n"
                              "every row get the same width and of every column the same height.");
    }

    ImGui::SeparatorText("Saved");
    if (g_saved.empty()) ImGui::TextDisabled("(none yet)");
    for (const std::string& name : g_saved) {
        ImGui::PushID(name.c_str());
        if (has_file(mode, name)) {
            if (ImGui::SmallButton("x")) g_req = Request{ mode, ReqKind::Remove, -1, name };
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete this layout");
        } else {
            // Встроенная раскладка: удалять нечего, место под крестик держим для ровного списка.
            ImGui::BeginDisabled();
            ImGui::SmallButton("x");
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Built-in layout. Save one under the same name to replace it.");
        }
        ImGui::SameLine();
        if (ImGui::MenuItem(name.c_str())) g_req = Request{ mode, ReqKind::Apply, -1, name };
        ImGui::PopID();
    }

    ImGui::SeparatorText("Save current");
    ImGui::SetNextItemWidth(200);
    const bool enter = ImGui::InputTextWithHint("##layout_name", "layout name", g_save_name,
                                                sizeof(g_save_name), ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    const std::string nm = g_save_name;
    const bool exists = std::find(g_saved.begin(), g_saved.end(), nm) != g_saved.end();
    ImGui::BeginDisabled(nm.empty());
    if ((ImGui::Button(exists ? "Overwrite" : "Save") || enter) && !nm.empty())
        g_req = Request{ mode, ReqKind::Save, -1, nm };
    ImGui::EndDisabled();
    ImGui::TextDisabled("Files: layouts\\%s\\*.layout next to the exe", mode_dir(mode));
    if (!g_status.empty()) ImGui::TextWrapped("%s", g_status.c_str());
}

void dock_layout_process(int mode, ImGuiID dock_id, ImVec2 size, const char* side_window,
                         const std::vector<ImGuiWindow*>& wins, const DockWindowFactory* factory) {
    if (g_req.mode != mode || g_req.kind == ReqKind::None) return;
    if (g_req.delay > 0) { --g_req.delay; return; }
    Request r = g_req;
    g_req = {};
    ImGuiDockNode* root = ImGui::DockBuilderGetNode(dock_id);

    switch (r.kind) {
    case ReqKind::Preset:
        // Grid над уже разложенными окнами — выравнивание на месте, без перестановки.
        // Tabs / Rows / Columns (одна цепочка ячеек) — строим сетку заново.
        if (r.preset == kDockGrid && root && !is_single_chain(plots_root(root, side_window))) {
            equalize(root, side_window);
            ImGui::MarkIniSettingsDirty();
            g_status.clear();
        } else {
            rebuild(dock_id, size, side_window, wins, r.preset);
            g_status.clear();
        }
        break;

    case ReqKind::Save: {
        if (!root) { g_status = "Nothing to save: no windows are docked here."; break; }
        std::error_code ec;
        fs::create_directories(layouts_dir(mode), ec);
        const fs::path file = layouts_dir(mode) / fs::u8path(file_stem(r.name) + ".layout");
        std::ofstream o(file, std::ios::binary);
        if (!o) { g_status = "Cannot write " + file.u8string(); break; }
        o << "# U-CUDA window layout. Sizes are fractions of the area, so the layout fits any screen.\n";
        o << "name=" << r.name << "\n";
        o << "tab=" << mode_dir(mode) << "\n";
        if (factory && factory->save_state) o << "state=" << one_line(factory->save_state()) << "\n";
        write_tree(o, *capture(root, factory));
        g_status = "Saved \"" + r.name + "\".";
        refresh_saved(mode);
        break;
    }

    case ReqKind::Apply: {
        // Текст раскладки: свой файл, иначе встроенная с тем же именем.
        const fs::path file = layouts_dir(mode) / fs::u8path(file_stem(r.name) + ".layout");
        std::stringstream in;
        {
            std::ifstream f(file, std::ios::binary);
            if (f) in << f.rdbuf();
            else if (const Builtin* b = find_builtin(mode, r.name)) in << b->text;
        }
        std::unique_ptr<LNode> tree = read_tree(in);
        if (!tree) { g_status = "Cannot read " + file.u8string(); break; }
        // Сначала состояние вкладки (какие окна вообще показываются). Если оно поменялось,
        // новые окна появятся только в следующих кадрах — раскладку откладываем на пару кадров.
        if (!r.state_done && factory && factory->load_state) {
            in.clear();
            in.seekg(0);
            std::string line, state;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("state=", 0) == 0) { state = line.substr(6); break; }
            }
            if (!state.empty() && factory->load_state(state)) {
                r.state_done = true;
                r.delay = 2;
                g_req = r;
                break;
            }
        }
        Matcher m;
        if (side_window) m.avail.push_back(side_window);
        for (ImGuiWindow* w : wins) m.avail.push_back(w->Name);
        ImGui::DockBuilderRemoveNode(dock_id);
        ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dock_id, size);
        BuildCtx c{ m, factory };
        build(dock_id, size, *tree, c);
        // Окна вкладки, которых в раскладке нет, — вкладками в первую ячейку.
        for (const std::string& w : m.avail) ImGui::DockBuilderDockWindow(w.c_str(), c.first_leaf ? c.first_leaf : dock_id);
        ImGui::DockBuilderFinish(dock_id);
        g_status = "Applied \"" + r.name + "\".";
        if (c.created > 0)
            g_status += " Opened " + std::to_string(c.created) + " missing window(s).";
        if (c.missing > 0)
            g_status += " " + std::to_string(c.missing) + " window(s) of the layout could not be opened.";
        if (!m.avail.empty())
            g_status += " " + std::to_string(m.avail.size()) + " extra window(s) went to the first cell.";
        break;
    }

    case ReqKind::Remove: {
        std::error_code ec;
        fs::remove(layouts_dir(mode) / fs::u8path(file_stem(r.name) + ".layout"), ec);
        g_status = ec ? "Cannot delete \"" + r.name + "\"." : "Deleted \"" + r.name + "\".";
        refresh_saved(mode);
        break;
    }

    case ReqKind::None: break;
    }
}

}  // namespace ucuda::ui_detail
