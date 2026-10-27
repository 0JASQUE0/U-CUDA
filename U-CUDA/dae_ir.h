#pragma once
// Промежуточное представление ДАУ (DAE IR) — этап 1 плана вкладки DAE.
//
// Полуявная форма:  x' = f(x, z, t),  0 = g(x, z, t).
//   x — дифференциальные переменные (vars_text записи), z — алгебраические
//   (alg_text). Связей ровно столько, сколько алгебраических переменных;
//   своей пары "связь — переменная" у связи нет, порядок связей — как в тексте.
//
// Разбор опирается на парсер ОДУ: строка "0 = g_k" временно пишется как
// "z_k' = g_k", и parse_system_from_latex нормализует все правые части одним
// путём (неявное умножение, |x|, ^, вспомогательные функции). Затем
// codegen_rhs проверяет каждое выражение до конца — неизвестный символ или
// синтаксическая ошибка ловятся здесь, а не при компиляции ядра.
//
// Индекс переменной (как nind1/2/3 в RADAU5) задаётся в записи текстом
// "x=1, lambda=3"; где не задан — дифференциальной 1, алгебраической 1, если
// структурный анализ нашёл ей связь (см. structural_index_hint), иначе 0
// ("не известен"). Структурный анализ — паросочетание "связь — алгебраическая
// переменная, входящая в неё": есть полное паросочетание — ∂g/∂z может быть
// невырождена (индекс 1 структурно), нет — индекс не меньше 2.

#include "sysparse.hpp"
#include "codegen.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct DaeSystem {
    std::vector<std::string> diff;     // x
    std::vector<std::string> alg;      // z
    std::vector<std::string> params;   // a[1..M], порядок как в params_text
    std::vector<std::string> f;        // f[i] — правая часть diff[i]' (нормализованная)
    std::vector<std::string> g;        // связи 0 = g[k]
    std::vector<int>         index;    // по переменным: diff, затем alg; 0 = не известен
    bool uses_t = false;               // в правых частях есть явное время t
    // 1 — у каждой связи нашлась своя алгебраическая переменная (индекс 1
    // структурно возможен); 2 — нет (индекс >= 2). У системы без связей — 0.
    int  structural_index_hint = 0;
    std::vector<std::string> unmatched_alg;   // z без связи в паросочетании

    size_t n_vars() const { return diff.size() + alg.size(); }
    // System для кодогена: vars = x, затем z; rhs = f, затем g. t, если есть,
    // — последний параметр (до этапа явного времени это единственный способ
    // провести выражение с t через codegen).
    System as_codegen_system() const {
        System s;
        s.vars = diff;
        s.vars.insert(s.vars.end(), alg.begin(), alg.end());
        s.params = params;
        if (uses_t) s.params.push_back("t");
        s.rhs = f;
        s.rhs.insert(s.rhs.end(), g.begin(), g.end());
        s.latex = true;
        return s;
    }
};

namespace dae_detail {

inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// "x, y; z" -> {"x","y","z"} (разделители: запятая, точка с запятой, пробел).
inline std::vector<std::string> split_names(const std::string& t) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&] {
        std::string n = trim(cur);
        if (!n.empty() && n[0] == '\\') n = n.substr(1);
        if (!n.empty() && std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
        cur.clear();
    };
    for (char ch : t) {
        if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') flush();
        else cur += ch;
    }
    flush();
    return out;
}

// Идентификаторы выражения: \name -> name, буквенно-цифровые с '_'. Имена,
// за которыми идёт '(' (вызовы функций), не берутся.
inline std::set<std::string> identifiers(const std::string& e) {
    std::set<std::string> out;
    size_t i = 0;
    while (i < e.size()) {
        const char ch = e[i];
        size_t j = i;
        std::string tok;
        if (ch == '\\') {
            j = i + 1;
            while (j < e.size() && std::isalpha((unsigned char)e[j])) ++j;
            tok = e.substr(i + 1, j - i - 1);
        } else if (std::isalpha((unsigned char)ch) || ch == '_') {
            while (j < e.size() && (std::isalnum((unsigned char)e[j]) || e[j] == '_')) ++j;
            tok = e.substr(i, j - i);
        } else if (std::isdigit((unsigned char)ch) || ch == '.') {
            // число целиком, вместе с показателем 1e-3, — чтобы "e" не стал именем
            while (j < e.size() && (std::isalnum((unsigned char)e[j]) || e[j] == '.')) ++j;
            if (j < e.size() && (e[j] == '+' || e[j] == '-') && (e[j - 1] == 'e' || e[j - 1] == 'E')) {
                ++j;
                while (j < e.size() && std::isdigit((unsigned char)e[j])) ++j;
            }
            i = j;
            continue;
        } else { ++i; continue; }
        size_t k = j;
        while (k < e.size() && (e[k] == ' ' || e[k] == '\t')) ++k;
        const bool call = k < e.size() && (e[k] == '(' || e[k] == '{');
        if (!tok.empty() && !call) out.insert(tok);
        i = j;
    }
    return out;
}

// Замена идентификатора целиком (не внутри других имён).
inline std::string replace_ident(const std::string& s, const std::string& from, const std::string& to) {
    auto is_id = [](char c) { return std::isalnum((unsigned char)c) || c == '_'; };
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0
            && (i == 0 || (!is_id(s[i - 1]) && s[i - 1] != '\\'))
            && (i + from.size() >= s.size() || !is_id(s[i + from.size()]))) {
            out += to;
            i += from.size();
        } else out += s[i++];
    }
    return out;
}

// Строки текста: по переводу строки и по LaTeX-разделителю "\\".
inline std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n' || (text[i] == '\\' && i + 1 < text.size() && text[i + 1] == '\\')) {
            out.push_back(cur);
            cur.clear();
            if (text[i] == '\\') ++i;
        } else cur += text[i];
    }
    out.push_back(cur);
    return out;
}

// Левая часть — ноль: "0", "0.0", "{0}".
inline bool is_zero_lhs(std::string l) {
    l.erase(std::remove_if(l.begin(), l.end(),
                           [](char c) { return c == ' ' || c == '\t' || c == '{' || c == '}'; }),
            l.end());
    if (l.empty()) return false;
    for (char c : l) if (c != '0' && c != '.') return false;
    return l.find('0') != std::string::npos;
}

} // namespace dae_detail

// "x=1, lambda=3" -> {x:1, lambda:3}. Мусор пропускается.
inline std::map<std::string, int> dae_parse_index_text(const std::string& t) {
    std::map<std::string, int> out;
    std::string item;
    auto flush = [&] {
        const size_t eq = item.find_first_of("=:");
        if (eq != std::string::npos) {
            std::string n = dae_detail::trim(item.substr(0, eq));
            if (!n.empty() && n[0] == '\\') n = n.substr(1);
            try { out[n] = std::stoi(dae_detail::trim(item.substr(eq + 1))); } catch (...) {}
        }
        item.clear();
    };
    for (char ch : t) { if (ch == ',' || ch == ';' || ch == '\n') flush(); else item += ch; }
    flush();
    return out;
}

// Разбор текста ДАУ. vars_text — x (пусто: берутся из левых частей по порядку),
// alg_text — z (обязательно, если есть связи), params_text — параметры.
// Бросает std::runtime_error с понятным сообщением.
inline DaeSystem parse_dae(const std::string& text,
                           const std::string& vars_text,
                           const std::string& alg_text,
                           const std::string& params_text,
                           const std::string& index_text = {},
                           const FuncDefs& funcs = {}) {
    using namespace dae_detail;
    DaeSystem d;
    d.alg    = split_names(alg_text);
    d.params = split_names(params_text);
    std::vector<std::string> diff_declared = split_names(vars_text);

    // 1. Строки: связи отдельно, уравнения с производной — как есть.
    std::vector<std::string> ode_rows, con_rows, ode_rhs;
    for (const std::string& raw : split_lines(text)) {
        std::string row = trim(raw);
        if (row.empty() || row[0] == '#' || row.rfind("//", 0) == 0 || row[0] == '%') continue;
        const size_t eq = row.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("equation without '=': " + row);
        const std::string lhs = row.substr(0, eq), rhs = trim(row.substr(eq + 1));
        if (rhs.empty()) throw std::runtime_error("empty right-hand side: " + row);
        if (is_zero_lhs(lhs)) con_rows.push_back(rhs);
        else { ode_rows.push_back(row); ode_rhs.push_back(rhs); }
    }
    if (ode_rows.empty() && con_rows.empty()) throw std::runtime_error("no equations");
    if (con_rows.size() != d.alg.size())
        throw std::runtime_error(std::to_string(con_rows.size()) + " constraint(s) '0 = ...' but "
            + std::to_string(d.alg.size()) + " algebraic variable(s): their numbers must match");

    // 2. Общий разбор: связь k пишется как "z_k' = g_k".
    // Разделитель строк — LaTeX "\\": split_rows парсера ОДУ режет по нему.
    std::string joined;
    for (const auto& r : ode_rows) joined += r + " \\\\\n";
    for (size_t k = 0; k < con_rows.size(); ++k) joined += d.alg[k] + "' = " + con_rows[k] + " \\\\\n";

    // Символы правых частей (левые "dx/dt" дали бы лишние dx, dt).
    static const std::set<std::string> latex_cmds = {
        "cdot", "times", "left", "right", "frac", "dot", "sqrt", "mathrm", "operatorname", "quad" };
    std::set<std::string> found;
    for (const auto* rows : { &ode_rhs, &con_rows })
        for (const auto& r : *rows)
            for (const auto& id : identifiers(r))
                if (!latex_cmds.count(id)) found.insert(id);
    d.uses_t = found.count("t") != 0;

    std::vector<std::string> alphabet = diff_declared;
    alphabet.insert(alphabet.end(), d.alg.begin(), d.alg.end());
    alphabet.insert(alphabet.end(), d.params.begin(), d.params.end());
    // Парсер ОДУ снимает "(t)" как запись x(t) и съел бы аргумент cos(t):
    // на время разбора t подменяется служебным именем.
    static const char* const kTimeAlias = "ucudaTimeT";
    if (d.uses_t) {
        alphabet.push_back(kTimeAlias);
        joined = replace_ident(joined, "t", kTimeAlias);
    }
    System parsed = parse_system_from_latex(joined, alphabet, ParamOrder::AsInAlphabet, funcs);
    if (d.uses_t)
        for (auto& r : parsed.rhs) r = replace_ident(r, kTimeAlias, "t");

    const size_t n_ode = ode_rows.size();
    if (parsed.vars.size() != n_ode + con_rows.size() || parsed.rhs.size() != parsed.vars.size())
        throw std::runtime_error("equation parser returned " + std::to_string(parsed.vars.size())
            + " rows for " + std::to_string(n_ode + con_rows.size()) + " equations");
    std::vector<std::string> lhs_vars(parsed.vars.begin(), parsed.vars.begin() + n_ode);
    for (size_t i = 0; i < n_ode; ++i) {
        const std::string& v = lhs_vars[i];
        if (std::find(d.alg.begin(), d.alg.end(), v) != d.alg.end())
            throw std::runtime_error("'" + v + "' is algebraic but has a derivative equation");
        if (std::find(lhs_vars.begin(), lhs_vars.begin() + i, v) != lhs_vars.begin() + i)
            throw std::runtime_error("two derivative equations for '" + v + "'");
    }
    if (diff_declared.empty()) diff_declared = lhs_vars;
    for (const auto& v : diff_declared)
        if (std::find(lhs_vars.begin(), lhs_vars.end(), v) == lhs_vars.end())
            throw std::runtime_error("no equation " + v + "' = ... for differential variable '" + v + "'");
    for (const auto& v : lhs_vars)
        if (std::find(diff_declared.begin(), diff_declared.end(), v) == diff_declared.end())
            throw std::runtime_error("'" + v + "' has a derivative but is not in the differential list");

    d.diff = diff_declared;
    for (const auto& v : d.diff) {
        const size_t i = std::find(lhs_vars.begin(), lhs_vars.end(), v) - lhs_vars.begin();
        d.f.push_back(parsed.rhs[i]);
    }
    d.g.assign(parsed.rhs.begin() + n_ode, parsed.rhs.end());

    // 3. Символы: всё неизвестное — ошибка с именем, а не "unknown symbol" из кодогена.
    {
        std::set<std::string> known(d.diff.begin(), d.diff.end());
        known.insert(d.alg.begin(), d.alg.end());
        known.insert(d.params.begin(), d.params.end());
        known.insert("t");
        known.insert("pi");
        known.insert("e");
        std::string unknown;
        for (const auto& id : found)
            if (!known.count(id)) unknown += (unknown.empty() ? "" : ", ") + id;
        if (!unknown.empty())
            throw std::runtime_error("unknown symbol(s): " + unknown + " (add them to Parameters)");
    }

    // 4. Полная проверка выражений тем же разбором, что пойдёт в ядро.
    codegen_rhs(d.as_codegen_system());

    // 5. Структурное паросочетание связей и алгебраических переменных.
    if (!d.g.empty()) {
        const size_t m = d.g.size();
        std::vector<std::vector<int>> adj(m);
        for (size_t k = 0; k < m; ++k) {
            const auto ids = identifiers(con_rows[k]);
            for (size_t j = 0; j < d.alg.size(); ++j)
                if (ids.count(d.alg[j])) adj[k].push_back((int)j);
        }
        std::vector<int> match_z(d.alg.size(), -1);
        std::vector<char> seen;
        std::function<bool(int)> augment = [&](int k) -> bool {
            for (int j : adj[k]) {
                if (seen[j]) continue;
                seen[j] = 1;
                if (match_z[j] < 0 || augment(match_z[j])) { match_z[j] = k; return true; }
            }
            return false;
        };
        size_t matched = 0;
        for (size_t k = 0; k < m; ++k) {
            seen.assign(d.alg.size(), 0);
            if (augment((int)k)) ++matched;
        }
        d.structural_index_hint = matched == m ? 1 : 2;
        for (size_t j = 0; j < d.alg.size(); ++j)
            if (match_z[j] < 0) d.unmatched_alg.push_back(d.alg[j]);
    }

    // 6. Индексы переменных.
    const auto given = dae_parse_index_text(index_text);
    for (const auto& v : d.diff) {
        auto it = given.find(v);
        d.index.push_back(it != given.end() ? it->second : 1);
    }
    for (const auto& v : d.alg) {
        auto it = given.find(v);
        const bool matched = std::find(d.unmatched_alg.begin(), d.unmatched_alg.end(), v)
                             == d.unmatched_alg.end();
        d.index.push_back(it != given.end() ? it->second
                          : (d.structural_index_hint == 1 && matched) ? 1 : 0);
    }
    for (const auto& kv : given) {
        if (std::find(d.diff.begin(), d.diff.end(), kv.first) == d.diff.end()
            && std::find(d.alg.begin(), d.alg.end(), kv.first) == d.alg.end())
            throw std::runtime_error("index given for unknown variable '" + kv.first + "'");
    }
    return d;
}
