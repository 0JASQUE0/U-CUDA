#include "sysparse.hpp"
#include <vector>
#include <set>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace {

    const char BS = '\\';

    // Преобразует модуль |..| и \left|..\right| в fabs(..).
    std::string convert_abs(std::string s) {
        s = std::regex_replace(s, std::regex(R"(\\left\|)"), "fabs(");
        s = std::regex_replace(s, std::regex(R"(\\right\|)"), ")");
        std::string r; bool open = true;
        for (char ch : s) {
            if (ch == '|') { r += open ? "fabs(" : ")"; open = !open; }
            else r += ch;
        }
        return r;
    }

    std::string strip_env(std::string s) {
        // \mathrm{d} -> d (производные в "d/dt" часто пишутся через \mathrm)
        s = std::regex_replace(s, std::regex(R"(\\mathrm\s*\{([^}]*)\})"), "$1");
        // \operatorname{...} -> ... (на всякий случай)
        s = std::regex_replace(s, std::regex(R"(\\operatorname\s*\{([^}]*)\})"), "$1");
        // команды масштабирования скобок: \big( \bigl( \bigr) \Big \bigg ... -> убрать,
        // сама скобка остаётся. Также \! \, уже чистятся ниже.
        s = std::regex_replace(s, std::regex(R"(\\(?:bigg|Bigg|big|Big)[lr]?)"), "");

        // Переменные как функции времени: x(t), y ( t ) -> x, y.
        // Убираем "( t )" с пробелами. Делаем это до разбора, т.к. иначе x(t)
        // выглядит как вызов функции. Также чистим производную: d x(t) -> d x.
        s = std::regex_replace(s, std::regex(R"(\s*\(\s*t\s*\)\s*)"), " ");
        s = convert_abs(s);

        // нижний индекс в фигурных скобках: x_{m} -> x_m, a_{ij} -> a_ij
        s = std::regex_replace(s, std::regex(R"(_\s*\{([A-Za-z0-9]+)\})"), "_$1");
        // верхний индекс-степень в скобках уже обрабатывается основным парсером (^{...})

        // OCR-артефакт: пробелы вокруг десятичной точки между цифрами.
        // "0 . 5" -> "0.5", "1. 25" -> "1.25" и т.п. Применяем несколько раз
        // на случай цепочек.
        for (int pass = 0; pass < 2; ++pass)
            s = std::regex_replace(s, std::regex(R"((\d)\s*\.\s*(\d))"), "$1.$2");

        static const std::vector<std::string> kill = {
            "\\begin{cases}","\\end{cases}","\\begin{aligned}","\\end{aligned}",
            "\\begin{array}","\\end{array}","\\left\\{","\\right.","\\left.","&",
            "\\quad","\\;","\\,","\\!"," \\ "
        };
        for (auto& k : kill) {
            size_t p;
            while ((p = s.find(k)) != std::string::npos) s.erase(p, k.size());
        }
        s = std::regex_replace(s, std::regex(R"(\{[lcr|]+\})"), "");
        return s;
    }

    // Снимает обёртывающие фигурные скобки и пунктуацию уравнений с одной строки/части.
    std::string clean_part(std::string t) {
        // убрать ведущие/замыкающие пробелы
        auto trim = [](std::string& x) {
            size_t a = x.find_first_not_of(" \t\n\r");
            size_t b = x.find_last_not_of(" \t\n\r");
            if (a == std::string::npos) { x.clear(); return; }
            x = x.substr(a, b - a + 1);
            };
        trim(t);
        auto count = [](const std::string& x, char ch) { size_t n = 0; for (char c : x) if (c == ch) ++n; return n; };
        // Единый цикл: снимаем внешние пары {}, пустые {}, хвостовую пунктуацию и
        // непарные скобки. Объединено, чтобы после удаления ';' снова проверить
        // внешнюю пару (иначе "{...}=...}  ;" не очищается).
        bool changed = true;
        while (changed) {
            changed = false;
            // пустые группы {}
            size_t p;
            while ((p = t.find("{}")) != std::string::npos) { t.erase(p, 2); changed = true; }
            trim(t);
            // завершающая пунктуация , ; .
            while (!t.empty() && (t.back() == ',' || t.back() == ';' || t.back() == '.')) { t.pop_back(); changed = true; }
            trim(t);
            // внешняя пара { ... } охватывающая всё
            if (t.size() >= 2 && t.front() == '{' && t.back() == '}') {
                int depth = 0; bool wraps = true;
                for (size_t i = 0; i < t.size(); ++i) {
                    if (t[i] == '{') ++depth;
                    else if (t[i] == '}') { --depth; if (depth == 0 && i + 1 != t.size()) { wraps = false; break; } }
                }
                if (wraps) { t = t.substr(1, t.size() - 2); changed = true; trim(t); }
            }
            // лишняя закрывающая } в конце, если открывающих меньше
            if (!t.empty() && t.back() == '}' && count(t, '}') > count(t, '{')) { t.pop_back(); changed = true; }
            trim(t);
            // лишняя открывающая { в начале, если закрывающих меньше
            if (!t.empty() && t.front() == '{' && count(t, '{') > count(t, '}')) { t.erase(t.begin()); changed = true; }
            trim(t);
        }
        return t;
    }

    std::vector<std::string> split_rows(const std::string& s) {
        std::vector<std::string> rows;
        size_t start = 0;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == BS && s[i + 1] == BS) {
                rows.push_back(s.substr(start, i - start));
                i += 1; start = i + 1;
            }
        }
        rows.push_back(s.substr(start));
        std::vector<std::string> out;
        for (auto& r : rows) {
            if (r.find_first_not_of(" \t\n\r") != std::string::npos) out.push_back(r);
        }
        return out;
    }

    std::string lhs_var(const std::string& lhs) {
        std::smatch m;
        // \dot — извлекаем базовое имя, затем ищем индекс _idx в ОСТАТКЕ строки
        // (он может быть отделён скобками: {{\dot{x}}_{m}} -> base x, индекс _m)
        {
            std::smatch md;
            std::string base;
            size_t tail_pos = 0;
            if (std::regex_search(lhs, md, std::regex(R"(\\dot\s*\{\s*\\([A-Za-z]+)\s*\})"))) {
                // \dot{\theta} -> греческое имя theta
                base = md[1];
                tail_pos = md.position(0) + md.length(0);
            }
            else if (std::regex_search(lhs, md, std::regex(R"(\\dot\s*\\([A-Za-z]+))"))) {
                // \dot\theta (без скобок) -> theta
                base = md[1];
                tail_pos = md.position(0) + md.length(0);
            }
            else if (std::regex_search(lhs, md, std::regex(R"(\\dot\s*\{\s*([A-Za-z]\w*)\s*\})"))) {
                base = md[1];
                tail_pos = md.position(0) + md.length(0);
            }
            else if (std::regex_search(lhs, md, std::regex(R"(\\dot\s*([A-Za-z]\w*))"))) {
                base = md[1];
                tail_pos = md.position(0) + md.length(0);
            }
            if (!base.empty()) {
                // ищем _index в хвосте после \dot{...}
                std::string tail = lhs.substr(tail_pos);
                std::smatch mi;
                if (std::regex_search(tail, mi, std::regex(R"(_(\w+))")))
                    base += "_" + std::string(mi[1]);
                return base;
            }
        }
        // \frac{ d x }{ d t }  (знаменатель: t, \tau, или другая переменная)
        if (std::regex_search(lhs, m, std::regex(R"(\\frac\s*\{\s*d\s*([A-Za-z]\w*)\s*\}\s*\{\s*d\s*\\?[A-Za-z]\w*\s*\})")))
            return m[1];
        // \frac{ d x }{ d \tau }  с греческой переменной дифференцирования (числитель обычный)
        if (std::regex_search(lhs, m, std::regex(R"(\\frac\s*\{\s*d\s*([A-Za-z]\w*)\s*\}\s*\{\s*d\s*\\[A-Za-z]+\s*\})")))
            return m[1];
        // d x / d t   (знаменатель: t, \tau, ...)
        if (std::regex_search(lhs, m, std::regex(R"(d\s*([A-Za-z]\w*)\s*/\s*d\s*\\?[A-Za-z]\w*)")))
            return m[1];
        // x'
        if (std::regex_search(lhs, m, std::regex(R"(([A-Za-z]\w*)\s*')")))
            return m[1];
        return "";
    }

    // Токенизация правой части по известному алфавиту с вставкой '*'.
    // Алфавит: точные имена (греческие как \sigma и обычные как x).
    // Возвращает выражение с явными операторами, годное для основного парсера.
    std::string insert_mult(const std::string& rhs,
        const std::set<std::string>& alpha_plain,   // x, y, b ...
        const std::set<std::string>& alpha_greek) { // sigma, omega ...
        std::string out;
        size_t i = 0;
        // нужен ли '*' между предыдущим токеном-операндом и следующим операндом
        bool prev_operand = false;

        auto emit_op = [&](const std::string& tok) {
            if (prev_operand) out += " * ";
            out += tok; prev_operand = true;
            };

        while (i < rhs.size()) {
            char c = rhs[i];
            if (std::isspace((unsigned char)c)) { out += c; ++i; continue; }
            if (c == BS) {
                // команда \name
                size_t j = i + 1;
                while (j < rhs.size() && std::isalpha((unsigned char)rhs[j])) ++j;
                std::string name = rhs.substr(i + 1, j - i - 1);
                // \cdot / \times — это ОПЕРАТОР умножения, не операнд
                if (name == "cdot" || name == "times") {
                    out += " * "; prev_operand = false; i = j; continue;
                }
                if (alpha_greek.count(name)) { emit_op("\\" + name); i = j; continue; }
                // иначе функция (\sin, \cos, \tanh ...) или \left
                if (prev_operand) out += " * ";
                out += "\\" + name; prev_operand = false; i = j;
                // степень на функции: \sin^{...} — скопировать ^{...} как часть функции,
                // НЕ вставляя умножение между функцией и её аргументом
                if (i < rhs.size() && rhs[i] == '^') {
                    out += '^'; ++i;
                    if (i < rhs.size() && rhs[i] == '{') {
                        int depth = 0;
                        do { out += rhs[i]; if (rhs[i] == '{') ++depth; else if (rhs[i] == '}') --depth; ++i; } while (i < rhs.size() && depth > 0);
                    }
                    else if (i < rhs.size()) { out += rhs[i]; ++i; }
                }
                continue;
            }
            if (std::isalpha((unsigned char)c)) {
                // жадно режем по алфавиту обычных имён: ищем самое длинное совпадение
                std::string best;
                for (size_t L = 1; i + L <= rhs.size(); ++L) {
                    if (!std::isalpha((unsigned char)rhs[i + L - 1])) break;
                    std::string cand = rhs.substr(i, L);
                    if (alpha_plain.count(cand)) best = cand;
                }
                if (!best.empty()) {
                    // разбиваем подряд идущие односимвольные: bz -> b*z, если оба в алфавите
                    emit_op(best); i += best.size(); continue;
                }
                // неизвестная буква (функция типа sin распознается дальше) — соберём слово
                size_t j = i; while (j < rhs.size() && std::isalpha((unsigned char)rhs[j])) ++j;
                std::string word = rhs.substr(i, j - i);
                // константа pi — это операнд (как число), а не функция
                if (word == "pi") { emit_op(word); i = j; continue; }
                if (prev_operand) out += " * ";
                out += word; prev_operand = false; // вероятно функция; '(' даст вызов
                i = j; continue;
            }
            if (std::isdigit((unsigned char)c) || c == '.') {
                size_t j = i; while (j < rhs.size() && (std::isdigit((unsigned char)rhs[j]) || rhs[j] == '.')) ++j;
                emit_op(rhs.substr(i, j - i)); i = j; continue;
            }
            if (c == '(') { if (prev_operand) out += " * "; out += '('; prev_operand = false; ++i; continue; }
            if (c == ')') { out += ')'; prev_operand = true; ++i; continue; }
            if (c == '{') { out += '{'; prev_operand = false; ++i; continue; }
            if (c == '}') { out += '}'; prev_operand = true; ++i; continue; }
            if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == ',') { out += c; prev_operand = false; ++i; continue; }
            // прочее — копируем как есть
            out += c; ++i;
        }
        out = std::regex_replace(out, std::regex("  +"), " ");
        return out;
    }

} // namespace

// ---- Разбор определений функций: "h(x) = тело" ----
// Принимает LaTeX или plain. Снимает окружения (\begin{aligned}, \big, ...),
// разбивает по "\\", берёт части с '=' (пустые/служебные пропускает).
FuncDefs parse_func_defs(const std::string& text) {
    FuncDefs out;
    // прогоняем через ту же очистку, что и систему: окружения, \big, \mathrm,
    // индексы _{1}->_1, модуль |..|->fabs, (t)->убрать и т.п.
    std::string s = strip_env(text);

    // разбиваем по "\\" (как уравнения), плюс по переводам строк
    std::vector<std::string> parts;
    {
        size_t start = 0;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (s[i] == '\\' && s[i + 1] == '\\') {
                parts.push_back(s.substr(start, i - start));
                i += 1; start = i + 1;
            }
        }
        parts.push_back(s.substr(start));
    }
    // дополнительно режем каждую часть по переводам строк
    std::vector<std::string> lines;
    for (auto& p : parts) {
        std::istringstream in(p);
        std::string ln;
        while (std::getline(in, ln)) lines.push_back(ln);
    }

    for (auto& line : lines) {
        // пропускаем пустые/служебные (нет '=' и нет букв -> мусор)
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            bool has_alpha = false;
            for (char ch : line) if (std::isalpha((unsigned char)ch)) { has_alpha = true; break; }
            if (!has_alpha) continue;            // пустая строка/остаток окружения
            // строка с буквами, но без '=' — возможно ошибка ввода; но если это
            // остаток вроде "\end{aligned}" уже снят strip_env, пропустим тихо
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            continue; // мягко пропускаем, не валим весь разбор
        }
        std::string lhs = line.substr(0, eq);
        std::string rhs = line.substr(eq + 1);

        std::smatch m;
        if (!std::regex_search(lhs, m, std::regex(R"(\s*([A-Za-z]\w*)\s*\(\s*([^)]*)\s*\))")))
            continue; // не похоже на заголовок функции — пропускаем
        std::string name = m[1];
        std::string args_str = m[2];
        FuncDef def;
        std::string cur;
        for (char c : args_str) {
            if (c == ',') {
                size_t a = cur.find_first_not_of(" \t");
                size_t b = cur.find_last_not_of(" \t");
                if (a != std::string::npos) def.args.push_back(cur.substr(a, b - a + 1));
                cur.clear();
            }
            else cur += c;
        }
        {
            size_t a = cur.find_first_not_of(" \t"); size_t b = cur.find_last_not_of(" \t");
            if (a != std::string::npos) def.args.push_back(cur.substr(a, b - a + 1));
        }

        // тело — trim + завершающая пунктуация
        size_t a = rhs.find_first_not_of(" \t");
        if (a != std::string::npos) rhs = rhs.substr(a);
        while (!rhs.empty() && (rhs.back() == ',' || rhs.back() == ';' || rhs.back() == '.' ||
            rhs.back() == ' ' || rhs.back() == '\t')) rhs.pop_back();
        // индексы и модуль уже сняты strip_env, но подстрахуемся
        rhs = std::regex_replace(rhs, std::regex(R"(_\s*\{([A-Za-z0-9]+)\})"), "_$1");
        def.body = convert_abs(rhs);
        if (!def.body.empty()) out[name] = def;
    }
    if (out.empty())
        throw std::runtime_error("no function definitions found (expected name(args) = body)");
    return out;
}

namespace {
    // Заменяет в выражении вызовы пользовательских функций их телами (инлайнинг).
    // Поддержка одного уровня вложенности достаточна для типичных систем;
    // для функций, вызывающих другие функции, применяем повторно до фикс-точки.
    std::string inline_funcs_once(const std::string& expr, const FuncDefs& funcs, bool& changed) {
        std::string out;
        size_t i = 0;
        while (i < expr.size()) {
            // ищем идентификатор, за которым '('
            if (std::isalpha((unsigned char)expr[i]) || expr[i] == '_') {
                size_t j = i;
                while (j < expr.size() && (std::isalnum((unsigned char)expr[j]) || expr[j] == '_')) ++j;
                std::string name = expr.substr(i, j - i);
                // пропускаем пробелы до '('
                size_t k = j; while (k < expr.size() && std::isspace((unsigned char)expr[k])) ++k;
                auto it = funcs.find(name);
                if (it != funcs.end() && k < expr.size() && expr[k] == '(') {
                    // парсим аргумент(ы) с учётом вложенных скобок
                    int depth = 0; size_t argstart = k + 1; size_t p = k;
                    std::vector<std::string> actual;
                    std::string cur;
                    for (p = k; p < expr.size(); ++p) {
                        char c = expr[p];
                        if (c == '(') { ++depth; if (depth == 1) continue; }
                        if (c == ')') {
                            --depth; if (depth == 0) {
                                size_t a = cur.find_first_not_of(" \t"); size_t b = cur.find_last_not_of(" \t");
                                if (a != std::string::npos) actual.push_back(cur.substr(a, b - a + 1));
                                break;
                            }
                        }
                        if (c == ',' && depth == 1) {
                            size_t a = cur.find_first_not_of(" \t"); size_t b = cur.find_last_not_of(" \t");
                            if (a != std::string::npos) actual.push_back(cur.substr(a, b - a + 1));
                            cur.clear(); continue;
                        }
                        cur += c;
                    }
                    // p указывает на закрывающую ')'
                    const FuncDef& def = it->second;
                    if (actual.size() != def.args.size())
                        throw std::runtime_error("function " + name + ": argument count mismatch");
                    // подставляем тело, заменяя формальные аргументы на фактические
                    std::string body = def.body;
                    // замена идентификаторов-аргументов целиком (не подстрок)
                    for (size_t ai = 0; ai < def.args.size(); ++ai) {
                        const std::string& formal = def.args[ai];
                        const std::string& act = actual[ai];
                        std::string rep;
                        size_t q = 0;
                        while (q < body.size()) {
                            if ((std::isalpha((unsigned char)body[q]) || body[q] == '_')) {
                                size_t r = q;
                                while (r < body.size() && (std::isalnum((unsigned char)body[r]) || body[r] == '_')) ++r;
                                std::string id = body.substr(q, r - q);
                                if (id == formal) rep += "(" + act + ")"; // оборачиваем для безопасности
                                else rep += id;
                                q = r;
                            }
                            else { rep += body[q]; ++q; }
                        }
                        body = rep;
                    }
                    out += "(" + body + ")"; // оборачиваем подставленное тело
                    changed = true;
                    i = p + 1;
                    continue;
                }
                // не пользовательская функция — копируем имя как есть
                out += name; i = j; continue;
            }
            out += expr[i]; ++i;
        }
        return out;
    }

    std::string inline_all_funcs(const std::string& expr, const FuncDefs& funcs) {
        if (funcs.empty()) return expr;
        std::string cur = expr;
        for (int iter = 0; iter < 16; ++iter) { // защита от бесконечной рекурсии
            bool changed = false;
            cur = inline_funcs_once(cur, funcs, changed);
            if (!changed) break;
        }
        return cur;
    }
} // namespace

System parse_system_from_latex(const std::string& multiline_latex,
    const std::vector<std::string>& alphabet,
    ParamOrder order,
    const FuncDefs& funcs) {
    // разделяем алфавит на греческие (многобуквенные имена команд) и обычные
    std::set<std::string> alpha_plain, alpha_greek;
    for (auto& a : alphabet) {
        if (a.size() >= 2 && std::all_of(a.begin(), a.end(),
            [](char ch) { return std::isalpha((unsigned char)ch); })
            && std::islower((unsigned char)a[0]) == std::islower((unsigned char)a[0])) {
            // эвристика: если имя длиннее 1 символа и встречается как \name -> греческое
            alpha_greek.insert(a);
        }
        // также добавляем в plain для прямого совпадения (x, y, b и короткие)
        if (a.size() == 1) alpha_plain.insert(a);
        else alpha_plain.insert(a); // многобуквенные обычные тоже допустимы
    }

    std::string s = strip_env(multiline_latex);
    std::vector<std::string> rows = split_rows(s);
    if (rows.empty()) throw std::runtime_error("no equations found in OCR output");

    System sys; sys.latex = true;
    std::set<std::string> used_syms;

    for (auto& row : rows) {
        // СНАЧАЛА снимаем внешние обёртки {...} со всего уравнения,
        // иначе разрез по '=' внутри обёртки ломает баланс скобок.
        std::string eqn = clean_part(row);
        // пропускаем пустые/служебные строки (остатки \end{array}, пунктуация)
        if (eqn.find_first_of("=") == std::string::npos) {
            bool has_alpha = false;
            for (char ch : eqn) if (std::isalpha((unsigned char)ch)) { has_alpha = true; break; }
            if (!has_alpha) continue; // действительно пустая/служебная — пропускаем
            throw std::runtime_error("equation without '=' (LHS with derivative required): " + eqn);
        }
        size_t eq = eqn.find('=');
        std::string lhs = clean_part(eqn.substr(0, eq));
        std::string rhs = clean_part(eqn.substr(eq + 1));

        std::string var = lhs_var(lhs);
        if (var.empty())
            throw std::runtime_error("cannot detect state variable in LHS: '" + lhs + "'");
        sys.vars.push_back(var);

        // инлайним вспомогательные функции (если заданы), затем умножение
        std::string rhs_inlined = inline_all_funcs(rhs, funcs);
        std::string rhs2 = insert_mult(rhs_inlined, alpha_plain, alpha_greek);
        size_t a = rhs2.find_first_not_of(" \t");
        if (a != std::string::npos) rhs2 = rhs2.substr(a);
        sys.rhs.push_back(rhs2);
    }

    // параметры = символы алфавита, не являющиеся переменными И не константами
    static const std::set<std::string> constants = { "pi" };
    std::set<std::string> varset(sys.vars.begin(), sys.vars.end());

    // множество допустимых параметров (из алфавита, не переменные, не константы)
    std::set<std::string> param_set;
    for (auto& a : alphabet)
        if (!varset.count(a) && !constants.count(a)) param_set.insert(a);

    std::vector<std::string> ordered;
    if (order == ParamOrder::AsInAlphabet) {
        // порядок как в alphabet
        for (auto& a : alphabet)
            if (param_set.count(a)) ordered.push_back(a);
    }
    else {
        // AsInSystem: порядок первого появления в правых частях
        std::set<std::string> seen;
        for (const auto& rhs : sys.rhs) {
            // ищем имена параметров как подстроки-идентификаторы в rhs
            // (rhs уже содержит \greek и обычные имена)
            size_t i = 0;
            while (i < rhs.size()) {
                // читаем идентификатор: \name или буквенно-цифровой с _
                std::string tok;
                if (rhs[i] == '\\') {
                    size_t j = i + 1;
                    while (j < rhs.size() && std::isalpha((unsigned char)rhs[j])) ++j;
                    tok = rhs.substr(i + 1, j - i - 1); // имя без слеша
                    i = j;
                }
                else if (std::isalpha((unsigned char)rhs[i]) || rhs[i] == '_') {
                    size_t j = i;
                    while (j < rhs.size() && (std::isalnum((unsigned char)rhs[j]) || rhs[j] == '_')) ++j;
                    tok = rhs.substr(i, j - i);
                    i = j;
                }
                else { ++i; continue; }

                if (param_set.count(tok) && !seen.count(tok)) {
                    seen.insert(tok);
                    ordered.push_back(tok);
                }
            }
        }
        // если какой-то параметр объявлен в алфавите, но не встретился в rhs —
        // добавим его в конец (в порядке алфавита), чтобы не потерять
        for (auto& a : alphabet)
            if (param_set.count(a) && !seen.count(a)) { ordered.push_back(a); seen.insert(a); }
    }
    sys.params = ordered;

    return sys;
}