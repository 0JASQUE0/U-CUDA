#include "plot_axis.h"
#include "digit_input.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <string>
#include <vector>

double nice_step(double range, int target_count) {
    if (range <= 0) return 1.0;
    double raw = range / std::max(1, target_count);
    double mag = std::pow(10.0, std::floor(std::log10(raw)));
    double norm = raw / mag;
    double step;
    if (norm < 1.5) step = 1.0;
    else if (norm < 3.5) step = 2.0;
    else if (norm < 7.5) step = 5.0;
    else                 step = 10.0;
    return step * mag;
}

double nice_step_up(double step) {
    if (!(step > 0.0)) return step;
    const double mag  = std::pow(10.0, std::floor(std::log10(step) + 1e-9));
    const double norm = step / mag;
    if (norm < 1.5) return 2.0  * mag;
    if (norm < 3.5) return 5.0  * mag;
    return 10.0 * mag;
}

bool node_snap_visible(double step_node, double range, float span_px) {
    if (!(step_node > 0.0) || !(range > 0.0) || span_px <= 1.0f) return false;
    // 4 px — порог различимости узла. Ниже него притяжение к сетке двигает тик
    // меньше чем на толщину линии, а подпись портит целиком.
    return (double)span_px * step_node / range >= 4.0;
}

// Глобальное значение для fmt_tick. Менеджится set_tick_precision() (зовётся
// из app_main при загрузке config и из Settings UI при изменении слайдера).
static int g_tick_precision = 4;

// Plot palette state. set_plot_light_theme зовётся из gui.cpp (Settings)
// + app_main.cpp (startup, читает AppConfig::dark_theme). Геттеры
// возвращают цвета под текущую тему — без перекомпиляции/ссылок на ImGui
// стиль, чтобы plot-код не зависел от того, активен ли ImGui контекст.
static bool g_plot_light = false;

void set_plot_light_theme(bool light) { g_plot_light = light; }
bool plot_light_theme()               { return g_plot_light; }

// Screenshot-to-clipboard sink. См. plot_axis.h.
static std::function<void(ImVec2, ImVec2)> g_screenshot_sink;

void set_screenshot_request_sink(std::function<void(ImVec2, ImVec2)> sink) {
    g_screenshot_sink = std::move(sink);
}
void request_plot_screenshot(ImVec2 min, ImVec2 max) {
    if (g_screenshot_sink) g_screenshot_sink(min, max);
}

// Совмещённый callback числовых полей: запятая→точка (CallbackCharFilter) +
// digit-step на ↑/↓ (CallbackHistory). ImGui позволяет OR'ить флаги; здесь
// диспетчеризуем по EventFlag. CallbackHistory — специальный event, который
// ImGui шлёт когда в активном InputText нажали ↑/↓ (изначально сделан под
// REPL command history). Ровно то, что нам нужно: клавиша уже отфильтрована
// и передана нам через колбэк — не нужен ни IsKeyPressed, ни pending-cursor
// state. См. объявление в plot_axis.h.
int digit_step_input_callback(ImGuiInputTextCallbackData* data) {
    if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
        if (data->EventChar == ',') data->EventChar = '.';
        return 0;
    }
    if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
        int dir = 0;
        if (data->EventKey == ImGuiKey_UpArrow)   dir = +1;
        if (data->EventKey == ImGuiKey_DownArrow) dir = -1;
        if (dir == 0) return 0;

        std::string text(data->Buf, data->Buf + data->BufTextLen);
        std::string new_text;
        int new_cursor = 0;
        if (!DigitInput::ComputeStep(text, data->CursorPos, dir,
                                     new_text, new_cursor)) {
            return 0;  // Дробь / scientific / невалидный ввод — не трогаем.
        }

        // DeleteChars + InsertChars сами выставляют BufDirty=true, ImGui
        // подхватит новую длину и вернёт changed=true из InputText.
        data->DeleteChars(0, data->BufTextLen);
        data->InsertChars(0, new_text.c_str());
        if (new_cursor < 0) new_cursor = 0;
        if (new_cursor > data->BufTextLen) new_cursor = data->BufTextLen;
        data->CursorPos      = new_cursor;
        data->SelectionStart = new_cursor;
        data->SelectionEnd   = new_cursor;
    }
    return 0;
}

ImU32 plot_col_text() {
    return g_plot_light ? IM_COL32( 30,  30,  40, 255)
                        : IM_COL32(220, 220, 230, 255);
}
ImU32 plot_col_axis() {
    return g_plot_light ? IM_COL32( 60,  60,  80, 220)
                        : IM_COL32(200, 200, 210, 200);
}
ImU32 plot_col_grid() {
    return g_plot_light ? IM_COL32(170, 170, 180, 200)
                        : IM_COL32( 80,  80,  90, 120);
}
ImU32 plot_col_border() {
    return g_plot_light ? IM_COL32(100, 100, 110, 220)
                        : IM_COL32(120, 120, 130, 200);
}
void plot_bg_color(float& r, float& g, float& b, float& a) {
    if (g_plot_light) { r = 0.985f; g = 0.985f; b = 0.985f; a = 1.0f; }
    else              { r = 0.080f; g = 0.080f; b = 0.100f; a = 1.0f; }
}

// ---------------------------------------------------------------------------
// LaTeX-подписи (контракт — в plot_axis.h). Строка разбирается в список
// прогонов {текст, шрифт, кегль, сдвиг базовой линии} плюс накладки — точки
// \dot, линейки \bar и дробные черты. Рисуется всё обычным AddText, поэтому
// поворот Y-подписи на -90° (plot_view_2d/heatmap_view крутят вершины кадра
// вручную) продолжает работать.
// ---------------------------------------------------------------------------
static ImFont* g_math_roman  = nullptr;
static ImFont* g_math_italic = nullptr;
static bool    g_math_on     = false;
static float   g_plot_font_scale = 1.0f;

void set_plot_math_fonts(ImFont* roman, ImFont* italic) {
    g_math_roman  = roman;
    g_math_italic = italic;
}
void set_plot_math_enabled(bool on) { g_math_on = on; }
bool plot_math_enabled()            { return g_math_on; }

void set_plot_font_scale(float s) { g_plot_font_scale = std::clamp(s, 0.5f, 3.0f); }
float plot_font_scale()           { return g_plot_font_scale; }
float plot_text_line_height()     { return ImGui::GetTextLineHeight() * g_plot_font_scale; }

// Кегль подписей округляется до целых пикселей: 13px UI-шрифта на множителе
// 1.6 дали бы 20.8, а на дробном кегле em-сетка глифа не совпадает с
// пиксельной, и засечки размазываются ещё до всякого позиционирования.
static float plot_font_px() {
    return std::max(1.0f, std::floor(ImGui::GetFontSize() * g_plot_font_scale + 0.5f));
}

namespace {

struct MathRun {
    std::string text;
    ImFont*     font;
    float       size;
    float       x;    // от левого края строки
    float       dy;   // сдвиг базовой линии: степень вверх, индекс вниз
};
// r > 0 — точка (\dot/\ddot); r == 0 — линейка (\bar, дробная черта).
struct MathMark { float x0, x1, y, r; };
struct MathLayout {
    std::vector<MathRun>  runs;
    std::vector<MathMark> marks;
};
// Форсированное начертание внутри \mathrm{...} / \mathit{...}.
struct Style { bool upright = false; bool italic = false; };

ImFont* math_roman()  { return g_math_roman  ? g_math_roman  : ImGui::GetFont(); }
ImFont* math_italic() { return g_math_italic ? g_math_italic : math_roman(); }

float font_ascent(ImFont* f, float size) {
    ImFontBaked* b = f ? f->GetFontBaked(size) : nullptr;
    return (b && b->Ascent > 0.0f) ? b->Ascent : size * 0.8f;
}
float run_width(ImFont* f, float size, const char* b, const char* e) {
    return f ? f->CalcTextSizeA(size, FLT_MAX, 0.0f, b, e).x : 0.0f;
}
float snap_px(float v) { return std::floor(v + 0.5f); }
// Шрифт может не знать типографских знаков (ProggyClean) — тогда обходимся ASCII.
bool has_glyph(ImWchar c) {
    ImFont* f = math_roman();
    return f && f->IsGlyphInFont(c);
}

// Отдельный глиф может отсутствовать в серифной паре: у CM Serif нет ∞ и ∂ —
// они живут в отдельных математических начертаниях CM. Рисуем такой знак
// UI-шрифтом, иначе ImGui подставит '?'.
ImFont* font_with_glyph(ImFont* pref, const char* utf8) {
    const unsigned char* p = (const unsigned char*)utf8;
    unsigned cp;
    if (p[0] < 0x80) cp = p[0];
    else if ((p[0] & 0xE0) == 0xC0 && p[1]) cp = ((p[0] & 0x1Fu) << 6) | (p[1] & 0x3Fu);
    else if ((p[0] & 0xF0) == 0xE0 && p[1] && p[2])
        cp = ((p[0] & 0x0Fu) << 12) | ((p[1] & 0x3Fu) << 6) | (p[2] & 0x3Fu);
    else return pref;   // вне BMP у нас глифов нет
    if (!pref || pref->IsGlyphInFont((ImWchar)cp)) return pref;
    ImFont* ui = ImGui::GetFont();
    return (ui && ui->IsGlyphInFont((ImWchar)cp)) ? ui : pref;
}

struct NameGlyph { const char* name; const char* glyph; };

// Имена пишутся и со слешем (\sigma из LaTeX-поля), и без него: в alphabet_text
// системы параметры заданы словами — "x,y,z,sigma,rho,beta".
const NameGlyph kGreek[] = {
    {"alpha","α"}, {"beta","β"}, {"gamma","γ"}, {"delta","δ"}, {"epsilon","ε"},
    {"varepsilon","ε"}, {"zeta","ζ"}, {"eta","η"}, {"theta","θ"}, {"vartheta","ϑ"},
    {"iota","ι"}, {"kappa","κ"}, {"lambda","λ"}, {"mu","μ"}, {"nu","ν"}, {"xi","ξ"},
    {"omicron","ο"}, {"pi","π"}, {"varpi","ϖ"}, {"rho","ρ"}, {"varrho","ϱ"},
    {"sigma","σ"}, {"varsigma","ς"}, {"tau","τ"}, {"upsilon","υ"}, {"phi","φ"},
    {"varphi","ϕ"}, {"chi","χ"}, {"psi","ψ"}, {"omega","ω"},
    {"Gamma","Γ"}, {"Delta","Δ"}, {"Theta","Θ"}, {"Lambda","Λ"}, {"Xi","Ξ"},
    {"Pi","Π"}, {"Sigma","Σ"}, {"Upsilon","Υ"}, {"Phi","Φ"}, {"Psi","Ψ"}, {"Omega","Ω"},
};
// Только со слешем: слово "in" или "to" в подписи — это слово, а не оператор.
const NameGlyph kSymbols[] = {
    {"cdot","·"}, {"times","×"}, {"infty","∞"}, {"pm","±"}, {"mp","∓"},
    {"partial","∂"}, {"nabla","∇"}, {"approx","≈"}, {"neq","≠"}, {"leq","≤"},
    {"geq","≥"}, {"ll","≪"}, {"gg","≫"}, {"to","→"}, {"rightarrow","→"},
    {"leftarrow","←"}, {"ldots","…"}, {"cdots","⋯"}, {"prime","′"},
    {"circ","°"}, {"hbar","ℏ"}, {"propto","∝"}, {"in","∈"}, {"sum","∑"},
    {"int","∫"}, {"sqrt","√"}, {"langle","⟨"}, {"rangle","⟩"},
};

const char* lookup(const NameGlyph* tbl, int n, const std::string& s) {
    for (int i = 0; i < n; ++i)
        if (s == tbl[i].name) return tbl[i].glyph;
    return nullptr;
}

bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_lower(char c) { return c >= 'a' && c <= 'z'; }

const char* group_end(const char* b, const char* e) {
    int depth = 1;
    for (const char* p = b; p < e; ++p) {
        if (*p == '{') ++depth;
        else if (*p == '}' && --depth == 0) return p;
    }
    return e;
}

// Аргумент команды или индекса: {...} целиком, \cmd целиком либо один символ
// (UTF-8 — вместе с продолжающими байтами).
void take_arg(const char*& p, const char* e, const char*& ab, const char*& ae) {
    if (p >= e) { ab = ae = p; return; }
    if (*p == '{') {
        ab = p + 1;
        ae = group_end(ab, e);
        p  = (ae < e) ? ae + 1 : e;
        return;
    }
    ab = p;
    if (*p == '\\') { ++p; while (p < e && is_alpha(*p)) ++p; }
    else            { ++p; while (p < e && ((unsigned char)*p & 0xC0) == 0x80) ++p; }
    ae = p;
}

float typeset(MathLayout& L, const char* b, const char* e, float size, Style st);

void append_layout(MathLayout& dst, const MathLayout& src, float dx, float dy) {
    for (MathRun r : src.runs)   { r.x  += dx; r.dy += dy; dst.runs.push_back(r); }
    for (MathMark m : src.marks) { m.x0 += dx; m.x1 += dx; m.y += dy; dst.marks.push_back(m); }
}

void emit(MathLayout& L, float& x, ImFont* f, float size, const std::string& text) {
    if (text.empty()) return;
    L.runs.push_back({ text, f, size, x, 0.0f });
    x += run_width(f, size, text.c_str(), text.c_str() + text.size());
}

// Цифры вплотную к одиночной букве или греческому имени уходят в индекс:
// переменные системы объявляются как x1,x2,x3, а не как x_1.
void emit_trailing_digits(MathLayout& L, float& x, const char*& p, const char* e, float size) {
    const char* ds = p;
    while (p < e && is_digit(*p)) ++p;
    if (ds == p) return;
    MathLayout sub;
    const float w = typeset(sub, ds, p, size * 0.72f, Style{ true, false });
    append_layout(L, sub, x, size * 0.20f);
    x += w;
}

// Цифры, знаки и пробелы — прямым начертанием; дефис становится настоящим
// минусом, иначе подписи тиков рядом с 10^−5 выглядят разнобоем.
std::string upright_text(const char* b, const char* e) {
    const bool real_minus = has_glyph(0x2212);
    std::string t;
    t.reserve((size_t)(e - b));
    for (const char* p = b; p < e; ++p) {
        if (*p == '-' && real_minus) t += "\xE2\x88\x92";   // U+2212 MINUS SIGN
        else                         t += *p;
    }
    return t;
}

float typeset(MathLayout& L, const char* b, const char* e, float size, Style st) {
    float x = 0.0f;
    const char* p = b;
    while (p < e) {
        const char c = *p;

        if (c == '{') {
            const char* ge = group_end(p + 1, e);
            MathLayout sub;
            const float w = typeset(sub, p + 1, ge, size, st);
            append_layout(L, sub, x, 0.0f);
            x += w;
            p = (ge < e) ? ge + 1 : e;
            continue;
        }
        if (c == '}') { ++p; continue; }

        if (c == '_' || c == '^') {
            ++p;
            const char *ab, *ae;
            take_arg(p, e, ab, ae);
            MathLayout sub;
            const float w = typeset(sub, ab, ae, size * 0.72f, st);
            append_layout(L, sub, x, (c == '^') ? -size * 0.42f : size * 0.20f);
            x += w + size * 0.02f;
            continue;
        }

        if (c == '\\' && p + 1 < e) {
            ++p;
            if (!is_alpha(*p)) {
                const char cc = *p++;
                if      (cc == ',' || cc == ';' || cc == ' ') x += size * 0.17f;  // \, \; — тонкий пробел
                else if (cc == '!')                           x -= size * 0.10f;  // \! — отрицательный
                else emit(L, x, math_roman(), size, std::string(1, cc));          // \{ \} \% \&
                continue;
            }
            const char* cs = p;
            while (p < e && is_alpha(*p)) ++p;
            const std::string cmd(cs, p);

            if (const char* g = lookup(kGreek, IM_ARRAYSIZE(kGreek), cmd)) {
                const bool it = st.italic || (!st.upright && is_lower(cmd[0]));
                emit(L, x, font_with_glyph(it ? math_italic() : math_roman(), g), size, g);
                emit_trailing_digits(L, x, p, e, size);
                continue;
            }
            if (const char* g = lookup(kSymbols, IM_ARRAYSIZE(kSymbols), cmd)) {
                emit(L, x, font_with_glyph(math_roman(), g), size, g);
                continue;
            }
            if (cmd == "dot" || cmd == "ddot" || cmd == "bar") {
                const char *ab, *ae;
                take_arg(p, e, ab, ae);
                MathLayout sub;
                const float w = typeset(sub, ab, ae, size, st);
                append_layout(L, sub, x, 0.0f);
                const float top = -font_ascent(math_italic(), size) * 0.86f;
                const float cx  = x + w * 0.5f;
                if (cmd == "bar")      L.marks.push_back({ x + w * 0.05f, x + w * 0.95f, top, 0.0f });
                else if (cmd == "dot") L.marks.push_back({ cx, cx, top, size * 0.055f });
                else {
                    const float d = size * 0.12f;
                    L.marks.push_back({ cx - d, cx - d, top, size * 0.055f });
                    L.marks.push_back({ cx + d, cx + d, top, size * 0.055f });
                }
                x += w;
                continue;
            }
            if (cmd == "hat" || cmd == "tilde" || cmd == "vec" || cmd == "mathbf") {
                // Диакритику не рисуем: голый глиф честнее кривой шляпки.
                const char *ab, *ae;
                take_arg(p, e, ab, ae);
                MathLayout sub;
                const float w = typeset(sub, ab, ae, size, st);
                append_layout(L, sub, x, 0.0f);
                x += w;
                continue;
            }
            if (cmd == "frac") {
                const char *nb, *ne, *db, *de;
                take_arg(p, e, nb, ne);
                take_arg(p, e, db, de);
                MathLayout num, den;
                const float fs = size * 0.82f;
                const float wn = typeset(num, nb, ne, fs, st);
                const float wd = typeset(den, db, de, fs, st);
                const float w  = std::max(wn, wd) + size * 0.20f;
                const float axis = -size * 0.28f;   // высота дробной черты над базовой линией
                append_layout(L, num, x + (w - wn) * 0.5f, axis - fs * 0.45f);
                append_layout(L, den, x + (w - wd) * 0.5f, axis + fs * 0.78f);
                L.marks.push_back({ x + size * 0.05f, x + w - size * 0.05f, axis, 0.0f });
                x += w;
                continue;
            }
            if (cmd == "mathrm" || cmd == "text" || cmd == "operatorname" || cmd == "mathit") {
                const char *ab, *ae;
                take_arg(p, e, ab, ae);
                Style s2 = st;
                s2.italic  = (cmd == "mathit");
                s2.upright = !s2.italic;
                MathLayout sub;
                const float w = typeset(sub, ab, ae, size, s2);
                append_layout(L, sub, x, 0.0f);
                x += w;
                continue;
            }
            // Неизвестная команда: печатаем её имя прямым, без слеша.
            emit(L, x, math_roman(), size, cmd);
            continue;
        }

        if (is_alpha(c)) {
            const char* ws = p;
            while (p < e && is_alpha(*p)) ++p;
            const std::string w(ws, p);

            // 'e' между цифрами — это порядок числа, а не переменная: строку
            // целиком rewrite_scientific уже развернул бы в степень десятки,
            // но внутри подписи ("h=1e-3, RK4") число остаётся как есть.
            if (w.size() == 1 && (w[0] == 'e' || w[0] == 'E') &&
                ws > b && is_digit(ws[-1]) && p < e &&
                (is_digit(*p) || ((*p == '+' || *p == '-') && p + 1 < e && is_digit(p[1])))) {
                emit(L, x, math_roman(), size, w);
                continue;
            }

            const char* g = lookup(kGreek, IM_ARRAYSIZE(kGreek), w);
            // Слово из нескольких букв — это не произведение переменных, а
            // название ("parameter", "max", "IC"): TeX набирает такое прямым.
            bool it = g ? is_lower(w[0]) : (w.size() == 1);
            if (st.upright) it = false;
            if (st.italic)  it = true;
            ImFont* wf = it ? math_italic() : math_roman();
            emit(L, x, g ? font_with_glyph(wf, g) : wf, size, g ? g : w);
            if (g || w.size() == 1) emit_trailing_digits(L, x, p, e, size);
            continue;
        }

        const char* rs = p;
        while (p < e && !is_alpha(*p) && *p != '\\' && *p != '{' && *p != '}'
                     && *p != '_' && *p != '^') ++p;
        if (p == rs) ++p;   // одиночный спецсимвол в хвосте строки: не зациклиться
        emit(L, x, math_roman(), size, upright_text(rs, p));
    }
    return x;
}

// "1.5e-05" -> "1.5×10^{-5}". Только если ВСЯ строка — одно такое число:
// иначе пострадала бы подпись вида "h=1e-3, RK4".
bool rewrite_scientific(const char* s, std::string& out) {
    const char* p = s;
    const char* mant = p;
    if (*p == '+' || *p == '-') ++p;
    bool any = false;
    while (is_digit(*p)) { ++p; any = true; }
    if (*p == '.') { ++p; while (is_digit(*p)) { ++p; any = true; } }
    if (!any || (*p != 'e' && *p != 'E')) return false;
    const std::string mantissa(mant, p);

    ++p;
    bool neg = false;
    if      (*p == '+') ++p;
    else if (*p == '-') { neg = true; ++p; }
    const char* ds = p;
    while (is_digit(*p)) ++p;
    if (ds == p || *p != '\0') return false;
    std::string digits(ds, p);
    while (digits.size() > 1 && digits[0] == '0') digits.erase(0, 1);

    out.clear();
    // Мантисса "1" в TeX не пишется — остаётся чистая степень десяти.
    if (mantissa != "1") {
        out += mantissa;
        out += has_glyph(0x00D7) ? "×" : "*";
    }
    out += "10^{";
    if (neg) out += "-";
    out += digits;
    out += "}";
    return true;
}

}  // namespace

ImVec2 plot_text_size(const char* s) {
    if (!s || !*s) return ImGui::CalcTextSize(s ? s : "");
    if (!g_math_on) {
        // Обычный ImGui-текст, но кегль подписей работает и здесь: при 1.0
        // путь буквально прежний, иначе меряем тем же шрифтом другого размера.
        if (g_plot_font_scale == 1.0f) return ImGui::CalcTextSize(s);
        ImFont* f = ImGui::GetFont();
        const float sz = plot_font_px();
        return ImVec2(run_width(f, sz, s, s + std::strlen(s)), plot_text_line_height());
    }
    const float size = plot_font_px();
    std::string sci;
    const char* src = rewrite_scientific(s, sci) ? sci.c_str() : s;
    MathLayout L;
    const float w = typeset(L, src, src + std::strlen(src), size, Style{});
    return ImVec2(w, plot_text_line_height());
}

void plot_text(ImDrawList* dl, ImVec2 pos, ImU32 col, const char* s) {
    if (!dl || !s || !*s) return;
    if (!g_math_on) {
        if (g_plot_font_scale == 1.0f) { dl->AddText(pos, col, s); return; }
        dl->AddText(ImGui::GetFont(), plot_font_px(),
                    ImVec2(snap_px(pos.x), snap_px(pos.y)), col, s);
        return;
    }
    const float size = plot_font_px();
    std::string sci;
    const char* src = rewrite_scientific(s, sci) ? sci.c_str() : s;
    MathLayout L;
    typeset(L, src, src + std::strlen(src), size, Style{});

    // AddText кладёт pos в верх строки, значит базовая линия = pos.y + ascent
    // базового шрифта. Прогоны другого кегля выравниваются по ней же.
    //
    // Каждый прогон садится на ЦЕЛЫЙ пиксель. Дробная позиция глифа означает
    // билинейную выборку из атласа, то есть мыло: ascent прямого и курсивного
    // начертаний отличается на доли пикселя, а подписи тиков вдобавок
    // центрируются по px - ширина/2. Снап — по обеим осям и до поворота
    // Y-подписи: на -90° целые координаты остаются целыми.
    const float base = snap_px(pos.y + font_ascent(math_roman(), size));
    for (const MathRun& r : L.runs) {
        const ImVec2 rp(snap_px(pos.x + r.x),
                        snap_px(base + r.dy - font_ascent(r.font, r.size)));
        dl->AddText(r.font, r.size, rp, col, r.text.c_str());
    }
    for (const MathMark& m : L.marks) {
        const float my = snap_px(base + m.y);
        const ImVec2 a(snap_px(pos.x + m.x0), my);
        if (m.r > 0.0f) dl->AddCircleFilled(a, std::max(1.0f, m.r), col, 8);
        else dl->AddLine(a, ImVec2(snap_px(pos.x + m.x1), my), col,
                         std::max(1.0f, size * 0.055f));
    }
}

float plot_left_margin_for_width(float max_tick_w, bool has_axis_name) {
    // 4 — зазор тик/плот (draw_axis_y_grid), 6 — тик/имя оси, высота строки —
    // само повёрнутое имя, ещё 8 — воздух у левого края блока.
    float m = max_tick_w + (has_axis_name ? 4.0f + 6.0f + plot_text_line_height() + 8.0f
                                          : 8.0f);
    m = std::ceil(m / 8.0f) * 8.0f;
    return std::clamp(m, 32.0f, 400.0f);
}

float plot_y_axis_margin(const AxisInfo& y, const char* y_name) {
    double emin, emax;
    axis_effective(y, emin, emax);
    const double vry = emax - emin;
    float max_w = 0.0f;
    if (std::abs(vry) >= 1e-30) {
        const double lo = std::min(emin, emax);
        const double hi = std::max(emin, emax);
        if (y.log_scale) {
            // Лог-ось подписывает только границы диапазона — см. draw_axis_y_grid.
            max_w = std::max(plot_text_size(fmt_tick(lo).c_str()).x,
                             plot_text_size(fmt_tick(hi).c_str()).x);
        } else {
            const double sy = nice_step(std::abs(vry), 6);
            const double ystart = std::ceil(lo / sy) * sy;
            const int n = (int)std::floor((hi - ystart) / sy + 1e-9) + 1;
            // Ограничение сверху — страховка от вырожденного шага: рисуется всё
            // равно не больше десятка тиков, а мерить миллион строк тут нельзя.
            for (int i = 0; i < n && i < 64; ++i)
                max_w = std::max(max_w, plot_text_size(fmt_tick(ystart + i * sy).c_str()).x);
        }
    }
    return plot_left_margin_for_width(max_w, y_name && *y_name);
}

void set_tick_precision(int n) {
    g_tick_precision = std::clamp(n, 2, 10);
}

std::string fmt_tick(double v) {
    char buf[32];
    double a = std::abs(v);
    if (a < 1e-12) return "0";
    // %.Ne даёт N+1 значащих цифр (1 до точки + N после); чтобы согласовать
    // с %.Ng (N значащих), для научной нотации передаём prec-1.
    if (a != 0 && (a < 1e-3 || a >= 1e5))
        std::snprintf(buf, sizeof(buf), "%.*e", std::max(1, g_tick_precision - 1), v);
    else
        std::snprintf(buf, sizeof(buf), "%.*g", g_tick_precision, v);
    return buf;
}

double fit_tick_step_x(double step, double lo, double hi, float plot_w) {
    const double range = hi - lo;
    if (!(step > 0.0) || !(range > 0.0) || plot_w <= 1.0f) return step;
    for (int guard = 0; guard < 12; ++guard) {
        const double v0 = std::ceil(lo / step) * step;
        const int n = (int)std::floor((hi - v0) / step + 1e-9) + 1;
        if (n <= 1) return step;   // одна подпись ни на что не наезжает
        float w = 0.0f;
        for (int i = 0; i < n && i < 40; ++i)
            w = std::max(w, plot_text_size(fmt_tick(v0 + i * step).c_str()).x);
        if ((double)plot_w * step / range >= (double)w + 8.0) return step;
        const double up = nice_step_up(step);
        if (!(up > step)) return step;
        step = up;
    }
    return step;
}

double fit_tick_step_y(double step, double range, float plot_h) {
    if (!(step > 0.0) || !(range > 0.0) || plot_h <= 1.0f) return step;
    // Подписи по Y однострочные — меряем высотой строки с запасом в 60%.
    const double need = (double)plot_text_line_height() * 1.6;
    for (int guard = 0; guard < 12; ++guard) {
        if ((double)plot_h * step / range >= need) return step;
        const double up = nice_step_up(step);
        if (!(up > step)) return step;
        step = up;
    }
    return step;
}

void make_ortho_mvp(double xmin, double xmax, double ymin, double ymax, float out[16]) {
    double dx = xmax - xmin; if (std::abs(dx) < 1e-30) dx = 1.0;
    double dy = ymax - ymin; if (std::abs(dy) < 1e-30) dy = 1.0;
    float sx = (float)(2.0 / dx);
    float sy = (float)(2.0 / dy);
    float tx = (float)(-(xmax + xmin) / dx);
    float ty = (float)(-(ymax + ymin) / dy);
    out[0] = sx; out[1] = 0;  out[2] = 0; out[3] = 0;
    out[4] = 0;  out[5] = sy; out[6] = 0; out[7] = 0;
    out[8] = 0;  out[9] = 0;  out[10] = 1; out[11] = 0;
    out[12] = tx; out[13] = ty; out[14] = 0; out[15] = 1;
}

void draw_axis_x_grid(ImDrawList* dl, const AxisInfo& x,
    ImVec2 pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text,
    double snap_lo, double snap_hi, int snap_n)
{
    double emin, emax;
    axis_effective(x, emin, emax);
    double vrx = emax - emin;
    if (std::abs(vrx) < 1e-30) return;

    double lo = std::min(emin, emax);
    double hi = std::max(emin, emax);

    // Log-масштаб: маппинг у Plot2DView теперь действительно логарифмический
    // (см. AxisInfo::log_scale), но промежуточные отметки здесь по-прежнему НЕ
    // рисуем — только границы диапазона. Это решение по оформлению, а не
    // ограничение: "красивые" линейные тики подписывали бы значения, которые
    // не считались, а декадные загромождают узкие диапазоны.
    if (x.log_scale) {
        // Концы в линейном и логарифмическом отображении попадают в одни и те
        // же пиксели (lo -> 0, hi -> plot_w), поэтому формула ниже верна и
        // после перевода Plot2DView на лог-координату.
        auto draw_edge = [&](double xv) {
            float px = pos.x + (float)((xv - emin) / vrx) * plot_w;
            dl->AddLine(ImVec2(px, pos.y), ImVec2(px, pos.y + plot_h), col_grid, 1.0f);
            std::string lbl = fmt_tick(xv);
            ImVec2 ts = plot_text_size(lbl.c_str());
            plot_text(dl, ImVec2(px - ts.x * 0.5f, pos.y + plot_h + 2), col_text, lbl.c_str());
        };
        draw_edge(lo);
        draw_edge(hi);
        return;
    }

    double sx = nice_step(std::abs(vrx), 8);
    sx = fit_tick_step_x(sx, lo, hi, plot_w);
    // Snap-to-node: округляем шаг тиков к целому кратному step_node и стартовую
    // позицию тоже кратно этому шагу. Каждый тик тогда — узел параметрической
    // сетки (snap_lo + k*step_node), без "промежуточных" значений.
    //
    // Только пока узлы различимы на экране (node_snap_visible). На плотной
    // сетке шаг тика наследовал иррациональный шаг узлов: свип 4..20 из 500
    // точек давал подписи 4, 5.99, 7.98, ..., 19.9 вместо 4, 6, ..., 20.
    double xstart;
    double step_node = (snap_n > 1 && snap_hi > snap_lo)
                       ? (snap_hi - snap_lo) / (double)(snap_n - 1) : 0.0;
    if (!node_snap_visible(step_node, std::abs(vrx), plot_w)) step_node = 0.0;
    if (step_node > 0.0) {
        int mult = (int)std::lround(sx / step_node);
        if (mult < 1) mult = 1;
        sx = (double)mult * step_node;
        int k_lo = (int)std::ceil((lo - snap_lo) / step_node - 1e-9);
        int k_start = (int)std::ceil((double)k_lo / (double)mult - 1e-9) * mult;
        xstart = snap_lo + (double)k_start * step_node;
    } else {
        xstart = std::ceil(lo / sx) * sx;
    }
    // hi-xstart нормируется на sx → floor(...) + 1 даёт ровно столько тиков,
    // сколько помещается в [xstart, hi]. Эпсилон ловит floating-point случаи
    // когда xstart + k*sx должно совпадать с hi, но из-за accumulation lo чуть
    // меньше. Дополнительно: tick рисуем только если он в пределах view (с
    // запасом в полстепа в обе стороны) — это исключает overshoot на правом
    // краю при zoom, когда последний tick визуально выпадает за границу плота.
    int nx = (int)std::floor((hi - xstart) / sx + 1e-9) + 1;
    if (nx < 0) nx = 0;

    // Подпись центрирована по px — её половина уезжает в margin_left/right
    // (они для этого и оставлены в layout'е плота). Не клампим текст в
    // ширину плота, иначе крайние tick'и без подписей.
    auto draw_tick = [&](double xv) {
        float px = pos.x + (float)((xv - emin) / vrx) * plot_w;
        dl->AddLine(ImVec2(px, pos.y), ImVec2(px, pos.y + plot_h), col_grid, 1.0f);
        std::string lbl = fmt_tick(xv);
        ImVec2 ts = plot_text_size(lbl.c_str());
        plot_text(dl, ImVec2(px - ts.x * 0.5f, pos.y + plot_h + 2), col_text, lbl.c_str());
    };

    // Порог растёт вместе с tick precision (Settings): чем больше значащих
    // цифр в подписи (fmt_tick), тем она шире, и тем больший зазор нужен,
    // чтобы edge-tick не наезжал текстом на соседний регулярный тик.
    double edge_frac;
    if      (g_tick_precision <= 2) edge_frac = 0.25;
    else if (g_tick_precision == 3) edge_frac = 0.30;
    else if (g_tick_precision == 4) edge_frac = 0.35;
    else                            edge_frac = 0.40;
    const double edge_eps = sx * edge_frac;
    bool lo_covered = false, hi_covered = false;

    for (int ix = 0; ix < nx; ++ix) {
        double xv = xstart + ix * sx;
        if (xv > hi + sx * 1e-6 || xv < lo - sx * 1e-6) continue;
        if (ix == 0)      lo_covered = std::abs(xv - lo) <= edge_eps;
        if (ix == nx - 1) hi_covered = std::abs(xv - hi) <= edge_eps;
        draw_tick(xv);
    }

    // Границы свипа (snap_lo/snap_hi) должны быть видны всегда, даже если
    // регулярный шаг тиков на них не попадает — иначе крайняя точка расчёта
    // визуально теряется (напр. подписи доходят до -0.0996, а не до 0).
    // Только для 1D Bif/LLE/LS (snap активен); Phase/TimeDomain (snap_n==0)
    // не затрагиваются.
    if (snap_n > 1) {
        if (!lo_covered) draw_tick(lo);
        if (!hi_covered) draw_tick(hi);
    }
}

void draw_axis_y_grid(ImDrawList* dl, const AxisInfo& y,
    ImVec2 pos, float plot_w, float plot_h,
    ImU32 col_grid, ImU32 col_text)
{
    double emin, emax;
    axis_effective(y, emin, emax);
    double vry = emax - emin;
    if (std::abs(vry) < 1e-30) return;

    double lo = std::min(emin, emax);
    double hi = std::max(emin, emax);

    // См. draw_axis_x_grid -- log-масштаб рисует только границы диапазона.
    if (y.log_scale) {
        auto draw_edge = [&](double yv) {
            float py = pos.y + (float)((emax - yv) / vry) * plot_h;
            dl->AddLine(ImVec2(pos.x, py), ImVec2(pos.x + plot_w, py), col_grid, 1.0f);
            std::string lbl = fmt_tick(yv);
            ImVec2 ts = plot_text_size(lbl.c_str());
            plot_text(dl, ImVec2(pos.x - ts.x - 4, py - ts.y * 0.5f), col_text, lbl.c_str());
        };
        draw_edge(lo);
        draw_edge(hi);
        return;
    }

    double sy = nice_step(std::abs(vry), 6);
    sy = fit_tick_step_y(sy, std::abs(vry), plot_h);
    double ystart = std::ceil(lo / sy) * sy;
    // См. комментарий в draw_axis_x_grid про формулу и +1e-9 эпсилон.
    int ny = (int)std::floor((hi - ystart) / sy + 1e-9) + 1;
    if (ny < 0) ny = 0;

    for (int iy = 0; iy < ny; ++iy) {
        double yv = ystart + iy * sy;
        if (yv > hi + sy * 1e-6 || yv < lo - sy * 1e-6) continue;
        float py = pos.y + (float)((emax - yv) / vry) * plot_h;
        // Подпись центрирована по py — её половина уезжает в margin_top/bottom
        // (они для этого и оставлены в layout'е плота). Не клампим текст по
        // высоте плота, иначе крайние tick'и (на самой границе view) без
        // подписей.
        dl->AddLine(ImVec2(pos.x, py), ImVec2(pos.x + plot_w, py), col_grid, 1.0f);
        std::string lbl = fmt_tick(yv);
        ImVec2 ts = plot_text_size(lbl.c_str());
        plot_text(dl, ImVec2(pos.x - ts.x - 4, py - ts.y * 0.5f), col_text, lbl.c_str());
    }
}