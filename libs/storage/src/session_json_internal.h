#pragma once
#include "ucuda/storage/session_io.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <cctype>
#include <stdexcept>

namespace ucuda::storage_detail {








// сериализация


    std::string esc(const std::string& s) ;


    void jstr(std::ostringstream& o, const std::string& v) ;


    void jmap(std::ostringstream& o, const std::map<std::string, std::string>& m) ;


    // мини-парсер JSON
    struct JP {
        std::string s; size_t i = 0;
        JP(std::string src) :s(std::move(src)) {}
        void ws() { while (i < s.size() && std::isspace((unsigned char)s[i]))++i; }
        char peek() { ws(); return i < s.size() ? s[i] : '\0'; }
        void expect(char c) { ws(); if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("expected ") + c); ++i; }
        bool opt(char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; }

        std::string str() {
            ws(); if (s[i] != '"') throw std::runtime_error("expected string"); ++i;
            std::string o;
            while (i < s.size() && s[i] != '"') {
                char c = s[i++];
                if (c == '\\' && i < s.size()) {
                    char e = s[i++];
                    switch (e) {
                    case 'n':o += '\n'; break; case 'r':o += '\r'; break; case 't':o += '\t'; break;
                    case '"':o += '"'; break; case '\\':o += '\\'; break; case '/':o += '/'; break;
                    case 'u': if (i + 4 <= s.size()) { int cd = std::stoi(s.substr(i, 4), nullptr, 16); i += 4; if (cd < 0x80)o += (char)cd; } break;
                    default:o += e;
                    }
                }
                else o += c;
            }
            if (i < s.size())++i; return o;
        }
        bool boolean() { ws(); if (s.compare(i, 4, "true") == 0) { i += 4; return true; } if (s.compare(i, 5, "false") == 0) { i += 5; return false; } throw std::runtime_error("expected bool"); }
        // читает число (целое/дробное) и возвращает как строку; строку в кавычках — тоже,
        // иначе на "0.1" курсор застревает на кавычке, парсер видит не ',' и не '}' и
        // объявляет битым весь файл сессии.
        std::string str_or_num() {
            ws(); if (i < s.size() && s[i] == '"') return str();
            size_t st = i; if (i < s.size() && (s[i] == '-' || s[i] == '+'))++i;
            while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+'))++i;
            return s.substr(st, i - st);
        }
        std::map<std::string, std::string> map_ss() {
            std::map<std::string, std::string> m; expect('{'); ws();
            if (opt('}')) return m;
            while (true) { std::string k = str(); expect(':'); std::string v = str(); m[k] = v; ws(); if (opt(','))continue; expect('}'); break; }
            return m;
        }
        void skip_value() { // на случай неизвестных полей
            ws(); char c = peek();
            if (c == '"') { str(); }
            else if (c == '{') { expect('{'); if (!opt('}')) { while (true) { str(); expect(':'); skip_value(); if (opt(','))continue; expect('}'); break; } } }
            else if (c == '[') { expect('['); if (!opt(']')) { while (true) { skip_value(); if (opt(','))continue; expect(']'); break; } } }
            else if (c == 't' || c == 'f') { boolean(); }
            else { while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i; }
        }
    };



// ---- RQA-поля проекции ----
// Пишутся ТОЛЬКО когда источник реально выбран (rqa_source != None). Иначе файлы всех
// существующих сессий распухли бы на 17 ключей в каждой проекции ради настроек, которых нет.
// Читателю ключи не обязательны: неизвестные глотает skip_value, отсутствующие остаются на
// дефолтах Projection — поэтому старые сессии грузятся без единой правки.
void write_rqa_fields(std::ostringstream& o, const Projection& p) ;


// true, если ключ был RQA-шный и уже разобран.
bool read_rqa_field(const std::string& k, JP& p, Projection& pr) ;


// ---- Библиотека регуляторов шага ----
void write_str_array(std::ostringstream& o, const std::vector<std::string>& v) ;


void read_str_array(JP& p, std::vector<std::string>& v) ;


void write_user_ctrl(std::ostringstream& o, const AdaptiveUserCtrl& u) ;


void read_user_ctrl(JP& p, AdaptiveUserCtrl& u) ;


// Тот же ли регулятор считается (имя и описание не в счёт).
bool same_ctrl_definition(const AdaptiveUserCtrl& a, const AdaptiveUserCtrl& b) ;


// Регулятор из сессии (ctrl_def) -> библиотека; возвращает имя, под которым он там есть.
// Нет такого имени — добавляется как есть. Есть с тем же определением — оно и есть. Есть, но
// считает иначе (правили после сохранения сессии, или сессия с другой машины), — сессия
// получает своё: запись с тем же определением под любым именем или новая "<имя> (session)".
// Раньше библиотека молча побеждала, и сессия считалась не тем регулятором, что сохранён.
std::string import_session_ctrl(const AdaptiveUserCtrl& u) ;


// ---- Адаптивный шаг ----
// Вложенный объект "adaptive" пишется, только если настройки отличаются от
// умолчания: сессии без адаптивного шага не меняются ни на байт, а старые
// читаются как Fixed (ключа нет — остаются дефолты AdaptiveSettings).
void write_adaptive(std::ostringstream& o, const AdaptiveSettings& a,
                    const char* prefix, const char* suffix) ;


void read_adaptive(JP& p, AdaptiveSettings& a) ;


// Регулятор заменён (read_adaptive): ось свипа по параметру прежнего регулятора
// (kAdAxisCtrl + k) указывала бы теперь на другой параметр Хайрера или на c[k], которого
// он не читает, и свип молча считал бы не то. Такая ось возвращается к обычной цели свипа.
// Зовётся и после регулятора, и после осей: порядок ключей в файле любой.
void drop_foreign_ctrl_axes(const AdaptiveSettings& a, int& ax, int& ax2) ;

// BifurcationAnalysisSession

// Сериализует одну БД в JSON-объект (без обёртки фигурными). Используется
// внутри массива "diagrams".
void write_diagram(std::ostringstream& o, const BifurcationDiagramConfig& bd) ;


// Читает одно поле в БД. Возвращает true, если ключ распознан.
// Используется и при разборе нового формата (внутри "diagrams"), и при
// разборе legacy-формата (плоские ключи на верхнем уровне = одна БД).
bool read_diagram_field(JP& p, BifurcationDiagramConfig& bd, const std::string& key) ;


// LLEAnalysisSession — отдельный JSON-файл `_last_lle.json`. Структура та же
// что у parametric (массив объектов), отличаются только поля per-«прогон».



void write_lle_curve(std::ostringstream& o, const LLECurveConfig& c) ;


bool read_lle_curve_field(JP& p, LLECurveConfig& c, const std::string& key) ;


// LyapunovSpectrumAnalysisSession — `_last_ls.json`. Поля LSCurveConfig
// идентичны LLECurveConfig — копия LLE-сериализатора.



void write_ls_curve(std::ostringstream& o, const LSCurveConfig& c) ;


bool read_ls_curve_field(JP& p, LSCurveConfig& c, const std::string& key) ;


// Dft1DAnalysisSession — `_last_dft1d.json`. Multi-config layout как у
// Basins/FastSync (см. read_fastsync_field — тот же "чистый", без legacy
// fallback, шаблон, т.к. это новая фича без старых сохранений). Result не
// сохраняется, как и display_cache* (транзиентный, перестраивается лениво).
void write_dft1d_config(std::ostringstream& o, const Dft1DConfig& c) ;


bool read_dft1d_field(JP& p, Dft1DConfig& c, const std::string& key) ;


// BasinsAnalysisSession — `_last_basins.json`. Один config на сессию
// (без curves-vector). Result не сохраняется.

// Запись одного BasinsConfig в JSON (без внешних { } — пишет голый объект), используется внутри
// массива "configs". Проекции фазовых портретов по бассейнам: формат тот же, что у "projections" в
// session_to_json(PhaseAnalysisSession); дублируется, а не шарится, чтобы не трогать рабочий
// сериализатор Phase-сессии.
void write_basins_phase_projections(std::ostringstream& o,
                                           const std::vector<Projection>& projs) ;


void read_basins_phase_projections(JP& p, std::vector<Projection>& out) ;


void write_basins_config(std::ostringstream& o, const BasinsConfig& c,
                                const BasinsPhaseSlot* phase_slot) ;


// Чтение одного поля BasinsConfig. Возвращает true если ключ распознан и
// значение прочитано из p. Иначе false — caller должен сам сделать skip_value.
// Используется и при разборе нового формата (внутри "configs"), и при
// разборе legacy плоского формата.
bool read_basins_field(JP& p, BasinsConfig& c, const std::string& key) ;


// FastSyncAnalysisSession JSON — multi-config layout как у basins.
void write_fastsync_config(std::ostringstream& o, const FastSyncConfig& c) ;


bool read_fastsync_field(JP& p, FastSyncConfig& c, const std::string& key) ;


// ---------------------------------------------------------------------------
// NetworkSession JSON — multi-config layout как у basins/fastsync, плюс две
// вложенные коллекции: узлы (переопределения параметров и НУ, положение в
// редакторе) и рёбра. Результат расчёта не пишется — он на порядки больше
// самой сессии и восстанавливается одним Run.
// Переопределения узла: пустое значение значит «берём общее», и таблица UI
// заводит такой ключ на КАЖДЫЙ параметр каждого узла просто потому, что по нему
// кликнули. В файл они не идут — иначе сессия из 256 узлов распухает на
// тысячи пустых строк, не неся ничего.
void jmap_nonempty(std::ostringstream& o, const std::map<std::string, std::string>& m) ;


void write_network_config(std::ostringstream& o, const NetworkConfig& c) ;


bool read_network_field(JP& p, NetworkConfig& c, const std::string& key) ;


// SignalMetricsAnalysisSession — `_last_metrics.json`. Структура как у LLE:
// массив конфигов, результат не хранится (после загрузки — Run заново).



void write_metrics_config(std::ostringstream& o, const SignalMetricsConfig& c) ;


bool read_metrics_config_field(JP& p, SignalMetricsConfig& c, const std::string& key) ;


// DFT1D plot windows: same flat-array shape as parametric windows, minus
// kind/mode_2d/colored_1d (DFT1D has only one display kind).
// OrderAnalysisSession — `_last_order.json`. Multi-config layout как у
// DFT1D/Basins/FastSync. Настройки ОТОБРАЖЕНИЯ здесь не пишутся: они живут на
// окнах графика (`_last_order_windows.json`), а не на конфиге. Result, как
// везде, не сохраняется.
void write_order_config(std::ostringstream& o, const OrderConfig& c) ;


bool read_order_field(JP& p, OrderConfig& c, const std::string& key) ;


// Custom-tab bundle: shared config + five embedded sub-session objects.


void write_shared_config(std::ostringstream& o, const CustomTabSharedConfig& c) ;


// Workspace metadata: tabs, active/next id, splitter width. imgui.ini owns
// the per-window dock layout inside each tab (keyed off the tab-derived
// dockspace id in ws_dock_id).
void write_workspace(std::ostringstream& o, const CustomWorkspace& ws) ;


// ---- Библиотека регуляторов шага (step_controllers.json) ----
bool load_ctrl_library(const std::string& path, std::string* err) ;


bool save_ctrl_library(std::string* err) ;
}
