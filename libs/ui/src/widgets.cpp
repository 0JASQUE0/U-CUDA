#include "gui_common.h"
#include "gui_inputs.h"
#include "gui_fields.h"
#include "ucuda/plotting/plot_legend.h"
// Общие поля ввода: текст, числа, списки, LaTeX-превью, цвета НУ, меню отмены.
using namespace ucuda::ui_detail;

namespace ucuda::ui_detail {
extern std::string exe_dir();
}


namespace ucuda::ui_detail {
std::string get_exe_dir_with_sep() {
    std::string d = platform::executable_dir();
    if (!d.empty() && d.back() != '\\' && d.back() != '/') d += "\\";
    return d;
}
}


namespace ucuda::ui_detail {
// Палитра серий по умолчанию — выбор в Settings (set_series_palette, plot_legend.h).
ImVec4 ic_base_color(int ic_index) {
    return default_series_color(ic_index);
}
}


namespace ucuda::ui_detail {
ImVec4 shade_of(ImVec4 base, int vi, int nv) {
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(base.x, base.y, base.z, h, s, v);
    // распределяем насыщенность от 1.0 (vi=0) до ~0.35 (последняя переменная)
    float frac = (nv <= 1) ? 0.0f : (float)vi / (float)(nv - 1);
    float new_s = s * (1.0f - 0.45f * frac); // от s до 0.55*s (было 0.65 -> 0.35, тускло)
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(h, new_s, v, r, g, b);
    return ImVec4(r, g, b, base.w);
}
}


namespace ucuda::ui_detail {
ImVec4 ic_var_shade(int ic_index, int vi, int nv) {
    return shade_of(ic_base_color(ic_index), vi, nv);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string combo_var_label(const std::vector<std::string>& vars) {
    if (vars.size() >= 3) return vars[0] + " + pi*" + vars[1] + " + e*" + vars[2];
    if (vars.size() == 2) return vars[0] + " + pi*" + vars[1];
    return vars.empty() ? std::string("x") : vars[0];
}
}


namespace ucuda::ui_detail {
static_assert(::pi == static_cast<numb>(num_parse_detail::kPi),
              "pi in num_parse.h diverged from configCUDA.h");
}


namespace ucuda::ui_detail {
[[nodiscard]] double combo_var_value(const std::vector<double>& pt, int nvars) {
    double v = !pt.empty() ? pt[0] : 0.0;
    if (nvars >= 2 && pt.size() > 1) v += ::pi    * pt[1];
    if (nvars >= 3 && pt.size() > 2) v += ::euler * pt[2];
    return v;
}
}


namespace ucuda::ui_detail {
std::vector<char>& input_scratch(const std::string& str, size_t extra) {
    static thread_local std::vector<char> buf;
    buf.assign(str.begin(), str.end());
    buf.resize(str.size() + extra);
    buf[str.size()] = '\0';
    return buf;
}
}


namespace ucuda::ui_detail {
int input_str_resize(ImGuiInputTextCallbackData* d) {
    if (d->EventFlag == ImGuiInputTextFlags_CallbackResize) {
        std::string* s = static_cast<std::string*>(d->UserData);
        s->resize((size_t)d->BufTextLen);
        d->Buf = s->data();
    }
    return 0;
}
}


namespace ucuda::ui_detail {
bool InputTextMultilineStr(const char* label, std::string& str, const ImVec2& size) {
    return ImGui::InputTextMultiline(label, str.data(), str.capacity() + 1, size,
                                     ImGuiInputTextFlags_CallbackResize, input_str_resize, &str);
}
}


namespace ucuda::ui_detail {
void draw_resize_handle(const char* id, float& height,
                               float min_h, float max_h) {
    // InvisibleButton refuses non-positive dimensions — resolve -1 to the actual width.
    const float w = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton(id, ImVec2(w > 1.0f ? w : 1.0f, 8.0f));
    const bool active  = ImGui::IsItemActive();
    const bool hovered = ImGui::IsItemHovered();
    if (active) {
        height += ImGui::GetIO().MouseDelta.y;
        if (height < min_h) height = min_h;
        if (height > max_h) height = max_h;
    }
    if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    const ImVec2 pmin = ImGui::GetItemRectMin();
    const ImVec2 pmax = ImGui::GetItemRectMax();
    const ImU32 col = ImGui::GetColorU32(active  ? ImGuiCol_SeparatorActive
                                       : hovered ? ImGuiCol_SeparatorHovered
                                                 : ImGuiCol_Separator);
    const float y = (pmin.y + pmax.y) * 0.5f;
    ImGui::GetWindowDrawList()->AddLine(ImVec2(pmin.x, y), ImVec2(pmax.x, y), col, 3.0f);
}
}


namespace ucuda::ui_detail {
void draw_latex_preview(const char* id, const std::string& src, ImVec2 size) {
    std::vector<std::string> lines;
    std::string cur;
    auto push = [&] {
        const size_t a = cur.find_first_not_of(" \t\r");
        if (a != std::string::npos) lines.push_back(cur.substr(a));
        cur.clear();
    };
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '\n') { push(); continue; }
        if (src[i] == '\\' && i + 1 < src.size() && src[i + 1] == '\\') { push(); ++i; continue; }
        cur += src[i];
    }
    push();

    ImGui::BeginChild(id, size, ImGuiChildFlags_Borders | ImGuiChildFlags_FrameStyle,
                      ImGuiWindowFlags_HorizontalScrollbar);
    const float fs  = std::floor(ImGui::GetFontSize() * 1.3f + 0.5f);
    const ImU32 col = ImGui::GetColorU32(ImGuiCol_Text);
    ImDrawList* dl  = ImGui::GetWindowDrawList();
    bool any = false;
    for (const std::string& ln : lines) {
        const MathExtent ext = math_formula_extent(ln.c_str(), fs);
        if (ext.width <= 0.0f) continue;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(ext.width, ext.ascent + ext.descent + fs * 0.2f));
        if (ImGui::IsItemVisible()) math_formula_draw(dl, p, col, ln.c_str(), fs);
        any = true;
    }
    if (!any) ImGui::TextDisabled("(formula preview)");
    ImGui::EndChild();
}
}


namespace ucuda::ui_detail {
void draw_rhs_preview(const SystemEditor& model) {
    const std::string& src =
        (model.mode == InputMode::Plain) ? model.plain_text : model.latex_text;
    if (src.find_first_not_of(" \t\r\n") == std::string::npos) return;
    if (!ucuda::ui_detail::block_header("Kernel form: X[..], a[..]", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    const AppModel::RhsPreview& pv = model.rhs_preview();
    if (pv.text.empty()) {
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.45f, 0.35f, 1.0f)), "Parse error: %s", pv.error.c_str());
        return;
    }
    // Длинная правая часть переносится, а не уезжает за край, поэтому высоту
    // меряем с тем же переносом (ширина поля минус отступы и полоса прокрутки).
    const ImGuiStyle& st = ImGui::GetStyle();
    const float wrap_w = ImGui::GetContentRegionAvail().x - st.FramePadding.x * 2.0f
                       - st.ScrollbarSize;
    const float text_h = ImGui::CalcTextSize(pv.text.c_str(), nullptr, false,
                                             std::max(wrap_w, 50.0f)).y;
    const float h = std::min(text_h + st.FramePadding.y * 2.0f + 2.0f, 320.0f);
    ImGui::InputTextMultiline("##rhs_preview", (char*)pv.text.c_str(), pv.text.size() + 1,
                              ImVec2(-1, h),
                              ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
}
}


namespace ucuda::ui_detail {
static AppSettings* g_width_store = nullptr;

void set_field_width_store(AppSettings* store) { g_width_store = store; }

float field_width(const char* key, float def) {
    if (!g_width_store) return def;
    auto it = g_width_store->field_widths.find(key);
    return it != g_width_store->field_widths.end() ? it->second : def;
}

float combo_fit_width(float content_w) {
    return std::ceil(content_w + ImGui::GetStyle().FramePadding.x * 2.0f + ImGui::GetFrameHeight());
}

float input_fit_width(float content_w) {
    // +2: курсор ввода в конце строки не должен сдвигать текст.
    return std::ceil(content_w + ImGui::GetStyle().FramePadding.x * 2.0f + 2.0f);
}

// Сколько ширины нужно содержимому полей столбца: максимум за этот и прошлый кадр (поля
// столбца, нарисованные позже, видны по прошлому кадру). По нему работает ПКМ-подгонка.
static std::map<std::string, float> g_need_cur, g_need_prev;
static int g_need_frame = -1;

static void need_roll() {
    const int f = ImGui::GetFrameCount();
    if (f == g_need_frame) return;
    g_need_prev = (f == g_need_frame + 1) ? std::move(g_need_cur) : std::map<std::string, float>{};
    g_need_cur.clear();
    g_need_frame = f;
}

static void need_record(const std::string& key, float need) {
    if (!(need > 0.0f)) return;
    need_roll();
    float& v = g_need_cur[key];
    v = std::max(v, need);
}

static float need_of(const std::string& key) {
    need_roll();
    float w = 0.0f;
    if (auto it = g_need_cur.find(key); it != g_need_cur.end()) w = it->second;
    if (auto it = g_need_prev.find(key); it != g_need_prev.end()) w = std::max(w, it->second);
    return w;
}

// Сколько нужно ТЕКСТУ полей ввода столбца (не пунктам комбо): если строка перестаёт влезать,
// столбец на время расширяется под неё и сужается обратно, когда текст укоротили. Заданную
// пользователем ширину это не трогает.
static std::map<std::string, float> g_grow_cur, g_grow_prev;
static int g_grow_frame = -1;

static void grow_roll() {
    const int f = ImGui::GetFrameCount();
    if (f == g_grow_frame) return;
    g_grow_prev = (f == g_grow_frame + 1) ? std::move(g_grow_cur) : std::map<std::string, float>{};
    g_grow_cur.clear();
    g_grow_frame = f;
}

static void grow_record(const std::string& key, float need) {
    grow_roll();
    float& v = g_grow_cur[key];
    v = std::max(v, need);
}

// Ширина столбца с учётом текста, который в него не влезает. Потолок — 600 px или заданная
// ширина, если она больше: строка в пару тысяч символов не должна выталкивать подписи за окно.
static float grown_width(const std::string& key, float w) {
    grow_roll();
    float need = 0.0f;
    if (auto it = g_grow_cur.find(key); it != g_grow_cur.end()) need = it->second;
    if (auto it = g_grow_prev.find(key); it != g_grow_prev.end()) need = std::max(need, it->second);
    return need > w ? std::min(need, std::max(w, 600.0f)) : w;
}

// Текущий блок полей в каждом окне (последний заголовок block_header / block_separator за кадр).
static std::map<ImGuiID, std::string> g_blocks;
static int g_blocks_frame = -1;

void set_width_block(const char* label) {
    if (g_blocks_frame != ImGui::GetFrameCount()) { g_blocks.clear(); g_blocks_frame = ImGui::GetFrameCount(); }
    g_blocks[ImGui::GetCurrentWindow()->ID] = label ? label : "";
}

static std::string current_width_block() {
    if (g_blocks_frame != ImGui::GetFrameCount()) return {};
    auto it = g_blocks.find(ImGui::GetCurrentWindow()->ID);
    return it != g_blocks.end() ? it->second : std::string();
}

// Ключ столбца по месту вызова: файл + функция + ширина по умолчанию + блок (заголовок, под
// которым поле стоит). Поля одной функции с одинаковой шириной в одном блоке — один столбец
// (Integration, Initial conditions, Parameters, оси min/max — каждый сам по себе).
std::string site_width_key(const char* file, const char* func, float def) {
    std::string f = file ? file : "";
    const size_t slash = f.find_last_of("/\\");
    if (slash != std::string::npos) f.erase(0, slash + 1);
    if (const size_t dot = f.rfind('.'); dot != std::string::npos) f.erase(dot);
    std::string fn = func ? func : "";
    // У лямбд MSVC в имени хэш (<lambda_1a2b...>), меняющийся с правками: берём объемлющую функцию.
    if (const size_t lam = fn.find("::<lambda"); lam != std::string::npos) fn.erase(lam);
    if (const size_t paren = fn.find('('); paren != std::string::npos) fn.erase(paren);
    if (const size_t ns = fn.rfind("::"); ns != std::string::npos) fn.erase(0, ns + 2);
    std::string key = f + '/' + fn + '/' + std::to_string((int)std::lround(def));
    std::string block = current_width_block();
    // В ключе — ни ';' ни '=' (формат строки AppConfig::field_widths). Цифры выкидываем: в
    // заголовке бывают счётчики ("Adaptive step (3 rejected)"), а блок от них меняться не должен.
    for (char& c : block) if (c == ';' || c == '=') c = '_';
    block.erase(std::remove_if(block.begin(), block.end(), [](char c) { return c >= '0' && c <= '9'; }),
                block.end());
    if (!block.empty()) key += '/' + block;
    return key;
}

// Ручка по прямоугольнику поля. id — устойчивый, пока тянут (ширина поля при этом меняется).
static void width_handle_rect(const std::string& key, ImGuiID id, ImVec2 imin, ImVec2 imax,
                              const std::function<float()>& fit) {
    ImGuiWindow* win = ImGui::GetCurrentWindow();
    constexpr float kMinW = 24.0f, kMaxW = 2000.0f;
    // Полоска у правого края поля, в основном в зазоре до следующего поля.
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const ImRect bb(imax.x - 2.0f, imin.y, imax.x + std::max(3.0f, gap - 1.0f), imax.y);
    if (!ImGui::ItemAdd(bb, id, nullptr, ImGuiItemFlags_NoNav)) return;
    bool hovered = false, held = false;
    ImGui::ButtonBehavior(bb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick);

    auto& widths = g_width_store->field_widths;
    auto set_w = [&](float w) {
        widths[key] = std::clamp(std::round(w), kMinW, kMaxW);
        g_width_store->field_widths_dirty = true;
    };
    static ImGuiID drag_id = 0;
    static float drag_w0 = 0.0f;
    const ImGuiIO& io = ImGui::GetIO();
    if (held) {
        // Ширина, с которой начали тащить: ширина самого поля (у комбинации переменных она шире
        // ширины столбца — тогда тащим от неё, а не прыгаем к столбцу).
        if (drag_id != id) { drag_id = id; drag_w0 = imax.x - imin.x; }
        // Без сдвига мыши ширину не трогаем: иначе второй щелчок двойного сразу записал бы
        // ширину обратно после сброса к умолчанию.
        const float dx = io.MousePos.x - io.MouseClickedPos[ImGuiMouseButton_Left].x;
        const float w = std::clamp(std::round(drag_w0 + dx), kMinW, kMaxW);
        auto it = widths.find(key);
        if (dx != 0.0f && (it == widths.end() || it->second != w)) set_w(w);
    } else if (drag_id == id) {
        drag_id = 0;
    }
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        widths.erase(key);
        g_width_store->field_widths_dirty = true;
    }
    // По ОТПУСКАНИЮ, а не по нажатию: контекстные меню полей (BeginPopupContextItem)
    // открываются на отпускании. Подгонка по нажатию сдвигала столбец, к отпусканию под
    // курсором оказывалось соседнее поле — и открывалось его меню.
    // Отпускание после этого гасим: прямоугольник поля с подписью справа ("Param lo") включает
    // зазор, где стоит ручка, и BeginPopupContextItem поля, нарисованного следом, открыл бы
    // его меню тем же щелчком.
    if (hovered && fit && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        set_w(fit());
        ImGui::GetIO().MouseReleased[ImGuiMouseButton_Right] = false;
    }

    if (hovered || held) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        const float x = imax.x + 1.0f;
        win->DrawList->AddLine(ImVec2(x, imin.y + 1.0f), ImVec2(x, imax.y - 1.0f),
                               ImGui::GetColorU32(held ? ImGuiCol_SeparatorActive : ImGuiCol_SeparatorHovered),
                               2.0f);
        if (!held)
            ImGui::SetItemTooltip(fit ? "Drag: width of this column\nRight-click: fit to contents\nDouble-click: default width"
                                      : "Drag: width of this column\nDouble-click: default width");
    }
}

void field_width_handle(const char* key, const std::function<float()>& fit) {
    if (!g_width_store) return;
    ImGuiWindow* win = ImGui::GetCurrentWindow();
    if (win->SkipItems) return;
    // Id от самого поля: в одной строке бывает несколько полей одного столбца (значения НУ).
    const ImGuiID item_id = ImGui::GetItemID();
    const ImGuiID id = ImHashStr(key, 0, item_id ? item_id : win->GetID("##fw"));
    width_handle_rect(key, id, ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), fit);
}

// Ручка ДО поля: прямоугольник поля предсказываем по курсору (поле встанет туда же) и ширине.
// Id — номер вхождения столбца в окне за кадр: от положения зависеть нельзя, левые поля того же
// столбца сдвигают его, пока тянут.
static float width_before_item(const std::string& key, float def) {
    const float w = grown_width(key, field_width(key.c_str(), def));
    if (!g_width_store || def <= 0.0f) return w;
    ImGuiWindow* win = ImGui::GetCurrentWindow();
    if (win->SkipItems) return w;
    static std::map<ImGuiID, int> occ;
    static int occ_frame = -1;
    if (occ_frame != ImGui::GetFrameCount()) { occ.clear(); occ_frame = ImGui::GetFrameCount(); }
    const ImGuiID base = win->GetID(key.c_str());
    const int n = occ[base]++;
    const ImGuiID id = ImHashData(&n, sizeof n, base);
    const ImVec2 p = win->DC.CursorPos;
    const ImGuiNextItemData saved = GImGui->NextItemData;   // ItemAdd ручки сбрасывает NextItemData
    std::function<float()> fit;
    if (need_of(key) > 0.0f) fit = [key] { return need_of(key); };
    width_handle_rect(key, id, p, ImVec2(p.x + w, p.y + ImGui::GetFrameHeight()), fit);
    GImGui->NextItemData = saved;
    return grown_width(key, field_width(key.c_str(), def));
}

// Столбец последнего item_w и место, куда встанет его поле: общее поле ввода, нарисованное сразу
// за item_w, дописывает в этот столбец ширину своего текста (иначе ПКМ было бы не по чему).
static std::string g_pending_key;
static ImVec2 g_pending_pos;
static int g_pending_frame = -1;

void item_w(float def, float need, const char* file, const char* func) {
    g_pending_frame = -1;
    if (def <= 0.0f) { ImGui::SetNextItemWidth(def); return; }   // "до края окна" — не столбец
    const std::string key = site_width_key(file, func, def);
    need_record(key, need);
    g_pending_key = key;
    g_pending_pos = ImGui::GetCursorScreenPos();
    g_pending_frame = ImGui::GetFrameCount();
    ImGui::SetNextItemWidth(width_before_item(key, def));
}

float combo_items_need(const char* const* items, int n) {
    float w = 0.0f;
    for (int k = 0; items && k < n; ++k)
        if (items[k]) w = std::max(w, ImGui::CalcTextSize(items[k]).x);
    return combo_fit_width(w);
}

float combo_items_need(const char* zero_separated) {
    float w = 0.0f;
    for (const char* p = zero_separated; p && *p; p += std::strlen(p) + 1)
        w = std::max(w, ImGui::CalcTextSize(p).x);
    return combo_fit_width(w);
}

// Редактируемое поле, которое расширилось под текст, остаётся прокрученным влево: InputText
// сдвигает прокрутку, только когда курсор уходит из вида. Если текст теперь влезает целиком —
// возвращаем прокрутку в ноль. Звать прямо перед InputText, после выставленной ширины.
static void unscroll_if_fits(const char* label, const std::string& text) {
    ImGuiContext& g = *GImGui;
    ImGuiWindow* win = g.CurrentWindow;
    const ImGuiID id = win->GetID(label);
    if (g.ActiveId != id) return;
    ImGuiInputTextState* st = ImGui::GetInputTextState(id);
    if (!st || st->Scroll.x <= 0.0f) return;
    const float w = (g.NextItemData.HasFlags & ImGuiNextItemDataFlags_HasWidth) ? g.NextItemData.Width
                                                                                 : ImGui::CalcItemWidth();
    if (w <= 0.0f) return;
    if (input_fit_width(ImGui::CalcTextSize(text.c_str()).x) <= w) st->Scroll.x = 0.0f;
}

// Ширина для общих полей ввода (InputTextStr, InputNumStr, ...): ручка и запись того, сколько
// нужно тексту. width <= 0 — ширину задали снаружи (item_w / SetNextItemWidth) или её нет.
static void helper_width(float width, const std::string& text, const char* file, const char* func) {
    const float text_need = input_fit_width(ImGui::CalcTextSize(text.c_str()).x);
    // Для автоподстройки — с запасом в один символ: ширина применяется со следующего кадра, и
    // без запаса введённый символ на кадр не влезал бы, InputText прокручивался влево и так и
    // оставался прокрученным в уже расширенном поле.
    const float grow_need = text_need + ImGui::CalcTextSize("0").x;
    if (width <= 0.0f) {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        if (g_pending_frame == ImGui::GetFrameCount() && p.x == g_pending_pos.x && p.y == g_pending_pos.y) {
            need_record(g_pending_key, text_need);
            grow_record(g_pending_key, grow_need);
            // Ширину item_w уже выставил — по прошлому кадру; текст этого кадра мог стать длиннее.
            ImGuiNextItemData& nd = GImGui->NextItemData;
            if ((nd.HasFlags & ImGuiNextItemDataFlags_HasWidth) && nd.Width > 0.0f)
                nd.Width = grown_width(g_pending_key, nd.Width);
        }
        g_pending_frame = -1;
        return;
    }
    g_pending_frame = -1;
    const std::string key = site_width_key(file, func, width);
    need_record(key, text_need);
    grow_record(key, grow_need);
    ImGui::SetNextItemWidth(width_before_item(key, width));
}
}


namespace ucuda::ui_detail {
// ПКМ по полю без своего меню: пункты конфигов вкладки и сброс (если поле в активном конфиге).
static void cfg_field_popup(std::string& str, const char* id_label) {
    AppModel* m = cfg_field_scope_model(&str);
    if (!m) return;                                 // поле не из конфига вкладки — меню нечего предложить
    if (!ImGui::BeginPopupContextItem()) return;   // id — от самого поля
    // Подпись поля — видимая часть label (до "##"); у безымянных полей её нет.
    std::string name = id_label ? id_label : "";
    if (const size_t hh = name.find("##"); hh != std::string::npos) name.erase(hh);
    if (name.empty()) name = "value";
    ImGui::TextDisabled("%s = %s", name.c_str(), str.empty() ? "(empty)" : str.c_str());
    ImGui::Separator();
    cfg_field_menu_items(&str, name);
    draw_undo_menu_item(m);
    ImGui::EndPopup();
}
}


namespace ucuda::ui_detail {
bool InputTextStr(const char* label, std::string& str, float width, const char* file, const char* func) {
    std::vector<char>& buf = input_scratch(str, 1024);
    helper_width(width, str, file, func);
    unscroll_if_fits(label, str);
    bool changed = ImGui::InputText(label, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
bool InputTextStrHint(const char* label, std::string& str, const char* hint,
                             float width, const char* file, const char* func) {
    std::vector<char>& buf = input_scratch(str, 1024);
    helper_width(width, str.empty() ? std::string(hint ? hint : "") : str, file, func);
    unscroll_if_fits(label, str);
    bool changed = ImGui::InputTextWithHint(label, hint, buf.data(), buf.size());
    if (changed) str = buf.data();
    return changed;
}
}


namespace ucuda::ui_detail {
int filter_comma_to_dot(ImGuiInputTextCallbackData* data) {
    if (data->EventChar == ',') data->EventChar = '.';
    return 0;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool is_numeric_string(const std::string& s) {
    if (s.empty()) return true;
    double v = 0.0;
    return parse_num_checked(s, v);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_axis_name(const std::vector<std::string>& params,
                                   const std::vector<std::string>& vars,
                                   int param_idx, bool sweep_over_var, int var_idx,
                                   bool sweep_over_h) {
    if (sweep_over_h) return "h";
    if (sweep_over_var) {
        if (var_idx >= 0 && var_idx < (int)vars.size()) return vars[var_idx] + " (IC)";
        return "var";
    }
    // -1 — цель "s": коэффициент симметрии живёт в a[0], а не в списке
    // параметров системы, поэтому имени для него в params[] нет.
    if (param_idx < 0) return "s";
    if (param_idx < (int)params.size()) return params[param_idx];
    return "param";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string sweep_range_label(const std::string& axis,
                                                   const std::string& lo_text,
                                                   const std::string& hi_text) {
    const std::string lo = lo_text.empty() ? std::string("?") : lo_text;
    const std::string hi = hi_text.empty() ? std::string("?") : hi_text;
    return axis + " [" + lo + ".." + hi + "]";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string label_identity(const std::string& s) {
    const size_t k = s.rfind(" [");
    return k == std::string::npos ? s : s.substr(0, k);
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_window(const ParametricPlotWindow& w) {
    const char* kn = w.kind == ParametricPlotWindow::Kind::Bifurcation ? "Bifurcation"
                    : w.kind == ParametricPlotWindow::Kind::LLE ? "LLE"
                    : w.kind == ParametricPlotWindow::Kind::LS  ? "LS" : "Metrics";
    const char* suffix = w.colored_1d ? "Colored 1D" : (w.mode_2d ? "2D" : "1D");
    return std::string(kn) + " " + suffix;
}
}


namespace ucuda::ui_detail {
[[nodiscard]] std::string auto_label_dft1d(const Dft1DConfig& c,
                                    const std::vector<std::string>& params,
                                    const std::vector<std::string>& vars) {
    return sweep_range_label(auto_axis_name(params, vars, c.param_index, c.sweep_over_var,
                                            c.var_sweep_index, c.sweep_over_h),
                             c.param_lo_text, c.param_hi_text);
}
}


namespace ucuda::ui_detail {
void refresh_auto_labels(AppModel& model) {
    for (auto& bd : model.bifurcation_session.diagrams)
        if (!bd.label_is_manual)
            bd.label = auto_label_sweep(bd, model.bifurcation_session.params, model.bifurcation_session.vars);
    for (auto& c : model.lle_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.lle_session.params, model.lle_session.vars);
    for (auto& c : model.ls_session.curves)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.ls_session.params, model.ls_session.vars);
    for (auto& c : model.metrics_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_sweep(c, model.metrics_session.params, model.metrics_session.vars);
    for (auto& c : model.dft1d_session.configs)
        if (!c.label_is_manual)
            c.label = auto_label_dft1d(c, model.dft1d_session.params, model.dft1d_session.vars);
    for (auto& w : model.parametric_plot_windows)
        if (!w.label_is_manual)
            w.label = auto_label_window(w);
    for (auto& w : model.dft1d_plot_windows)
        if (!w.label_is_manual)
            w.label = "DFT 1D";
}
}


namespace ucuda::ui_detail {
[[nodiscard]] bool parse_num_ok(const std::string& v, double& out) {
    const double a = parse_ratio_or(v, 0.0);
    const double b = parse_ratio_or(v, 1.0);
    if (a != b) return false;
    out = a;
    return true;
}
}


namespace ucuda::ui_detail {
bool InputNumStr(const char* label, std::string& str, float width,
                        const std::function<void()>& ctx_menu,
                        bool allow_empty, const char* file, const char* func) {
    std::vector<char>& buf = input_scratch(str, 1024);
    helper_width(width, str, file, func);
    unscroll_if_fits(label, str);
    // CallbackHistory — ↑/↓ в активном InputText, обрабатываем в digit_step_input_callback.
    // CallbackCharFilter — прежняя замена запятой на точку, тоже в нём.
    bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
        digit_step_input_callback);
    if (changed) str = buf.data();
    if (ctx_menu) {
        // Своё меню поля (рассылка по вкладкам) допишет пункты конфигов вкладки и сброс.
        std::string* prev = cfg_menu_field();
        set_cfg_menu_field(&str);
        ctx_menu();
        set_cfg_menu_field(prev);
    } else {
        cfg_field_popup(str, label);
    }

    // Inline-предупреждение, если содержимое не парсится как число.
    // Default из engine'а (0) всё равно применится, но пользователю
    // явно сигналим, что введённое значение игнорируется.
    if (!(allow_empty && str.empty()) && !is_numeric_string(str)) {
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.55f, 0.3f, 1.0f)),
            "  invalid number, using default");
    }
    return changed;
}
}


namespace ucuda::ui_detail {
int list_digit_step_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag != ImGuiInputTextFlags_CallbackHistory) return 0;
    int dir = 0;
    if (data->EventKey == ImGuiKey_UpArrow)   dir = +1;
    if (data->EventKey == ImGuiKey_DownArrow) dir = -1;
    if (dir == 0) return 0;
    const std::string text(data->Buf, data->Buf + data->BufTextLen);
    const int cur = std::clamp(data->CursorPos, 0, (int)text.size());
    int b = cur, e = cur;
    while (b > 0 && text[b - 1] != ',' && text[b - 1] != ';') --b;
    while (e < (int)text.size() && text[e] != ',' && text[e] != ';') ++e;
    while (b < e && text[b] == ' ') ++b;
    while (e > b && text[e - 1] == ' ') --e;
    if (b >= e) return 0;
    std::string new_tok;
    int new_cur = 0;
    if (!DigitInput::ComputeStep(text.substr(b, e - b), std::clamp(cur - b, 0, e - b), dir,
                                 new_tok, new_cur))
        return 0;
    data->DeleteChars(b, e - b);
    data->InsertChars(b, new_tok.c_str());
    const int pos = std::clamp(b + new_cur, 0, data->BufTextLen);
    data->CursorPos = data->SelectionStart = data->SelectionEnd = pos;
    return 0;
}
}


namespace ucuda::ui_detail {
bool InputNumListStr(const char* label, std::string& str, float width, const char* file, const char* func) {
    std::vector<char>& buf = input_scratch(str, 1024);
    helper_width(width, str, file, func);
    unscroll_if_fits(label, str);
    const bool changed = ImGui::InputText(label, buf.data(), buf.size(),
        ImGuiInputTextFlags_CallbackHistory, list_digit_step_callback);
    if (changed) str = buf.data();
    cfg_field_popup(str, label);
    return changed;
}
}


namespace ucuda::ui_detail {
bool InputNumStrCommit(const char* label, std::string& text, double& value,
                              float width, const char* file, const char* func) {
    std::vector<char>& buf = input_scratch(text, 1024);
    helper_width(width, text, file, func);
    unscroll_if_fits(label, text);
    if (ImGui::InputText(label, buf.data(), buf.size(),
                         ImGuiInputTextFlags_CallbackCharFilter | ImGuiInputTextFlags_CallbackHistory,
                         digit_step_input_callback))
        text = buf.data();

    // Снимаем ДО возможного TextColored: тот станет last item, и IsItem* уже
    // спрашивал бы про него.
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    const bool valid     = !text.empty() && is_numeric_string(text);
    if (!valid)
        ImGui::TextColored(readable_text(ImVec4(1.0f, 0.55f, 0.3f, 1.0f)),
            "  invalid number, keeping previous value");

    if (committed && valid) {
        value = parse_ratio_or(text, value);
        return true;
    }
    return false;
}
}


namespace ucuda::ui_detail {
void do_undo(AppModel& m) {
    if (m.undo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void do_redo(AppModel& m) {
    if (m.redo_last()) m.undo_note_time = ImGui::GetTime();
}
}


namespace ucuda::ui_detail {
void draw_undo_menu_item(AppModel* m) {
    if (!m) return;
    const std::string* u = m->undo_top();
    const std::string* r = m->redo_top();
    if (!u && !r) return;
    ImGui::Separator();
    if (u) {
        const std::string lbl = "Undo: " + *u;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Z")) do_undo(*m);
    }
    if (r) {
        const std::string lbl = "Redo: " + *r;
        if (ImGui::MenuItem(lbl.c_str(), "Ctrl+Shift+Z")) do_redo(*m);
    }
}
}


namespace ucuda::ui_detail {
std::string dot_latex(const std::string& name, const FieldLabels* labels ) {
    std::string tex = name_to_latex(name);
    if (labels) {
        auto it = labels->find(name);
        if (it != labels->end() && !it->second.empty()) tex = it->second;
    }
    const size_t us = tex.find("_{");
    return us == std::string::npos || us == 0 ? "\\dot{" + tex + "}" : "\\dot{" + tex.substr(0, us) + "}" + tex.substr(us);
}
}
