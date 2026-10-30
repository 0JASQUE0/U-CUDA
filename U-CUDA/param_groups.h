#pragma once
// Группы параметров — только для интерфейса: как поля параметров разложены по раскрывающимся
// узлам. Порядок a[] и всё, что идёт в расчёт, от групп не зависят.
//
// Группа — путь "OPA2134/XU1" ('/' — вложенность) и список параметров. Промежуточные узлы
// пути отдельной записи не требуют. Параметр — не больше чем в одной группе; не попавшие ни в
// одну показываются в конце узлом "Other". Записи держатся отсортированными по пути.
//
// Импорт схемы раскладывает параметры сам (SpiceImportResult::param_groups): R / C / L /
// источники по видам, параметры моделей приборов — "Models/<модель>", всё из подсхемы —
// "<подсхема>/<экземпляр>". Пользователь правит группы в окне "Parameter groups".
//
// В system.json — одна строка "param_groups": строки "<путь>: p1, p2" (строкой, а не объектом:
// старые сборки пропускают неизвестный ключ через parse_string).

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct ParamGroup {
    std::string path;
    std::vector<std::string> params;
};
using ParamGroups = std::vector<ParamGroup>;

// Путь без пустых сегментов и пробелов по краям сегментов; ':' и переводы строк убираются
// (они — разделители формата хранения).
inline std::string pg_normalize_path(const std::string& s) {
    std::string out, seg;
    auto flush = [&] {
        size_t a = seg.find_first_not_of(" \t"), b = seg.find_last_not_of(" \t");
        if (a != std::string::npos) {
            if (!out.empty()) out += '/';
            out += seg.substr(a, b - a + 1);
        }
        seg.clear();
    };
    for (char c : s) {
        if (c == '/') flush();
        else if (c != ':' && c != '\n' && c != '\r') seg += c;
    }
    flush();
    return out;
}

// path — сам узел или лежит внутри prefix.
inline bool pg_in_subtree(const std::string& path, const std::string& prefix) {
    return path == prefix || (path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0
                              && path[prefix.size()] == '/');
}

inline ParamGroup* pg_find(ParamGroups& g, const std::string& path) {
    for (ParamGroup& x : g) if (x.path == path) return &x;
    return nullptr;
}

// Группа по пути; нет — создаётся на своём месте по сортировке.
inline ParamGroup& pg_get(ParamGroups& g, const std::string& path) {
    auto it = std::lower_bound(g.begin(), g.end(), path,
                               [](const ParamGroup& a, const std::string& p) { return a.path < p; });
    if (it != g.end() && it->path == path) return *it;
    return *g.insert(it, ParamGroup{ path, {} });
}

// Путь группы параметра, "" — без группы.
inline std::string pg_group_of(const ParamGroups& g, const std::string& param) {
    for (const ParamGroup& x : g)
        if (std::find(x.params.begin(), x.params.end(), param) != x.params.end()) return x.path;
    return {};
}

// Перенос параметра в группу path ("" — убрать из групп).
inline void pg_move(ParamGroups& g, const std::string& param, const std::string& path) {
    for (ParamGroup& x : g) x.params.erase(std::remove(x.params.begin(), x.params.end(), param), x.params.end());
    const std::string p = pg_normalize_path(path);
    if (!p.empty()) pg_get(g, p).params.push_back(param);
}

// Переименование узла вместе с поддеревом; совпавшие пути сливаются.
inline void pg_rename(ParamGroups& g, const std::string& from, const std::string& to_raw) {
    const std::string to = pg_normalize_path(to_raw);
    if (to.empty() || to == from || pg_in_subtree(to, from)) return;
    ParamGroups moved, kept;
    for (ParamGroup& x : g) (pg_in_subtree(x.path, from) ? moved : kept).push_back(std::move(x));
    g = std::move(kept);
    for (ParamGroup& x : moved) {
        ParamGroup& d = pg_get(g, to + x.path.substr(from.size()));
        d.params.insert(d.params.end(), x.params.begin(), x.params.end());
    }
}

// Удаление узла с поддеревом; его параметры остаются без группы.
inline void pg_remove(ParamGroups& g, const std::string& path) {
    g.erase(std::remove_if(g.begin(), g.end(), [&](const ParamGroup& x) { return pg_in_subtree(x.path, path); }),
            g.end());
}

// Оставляет в группах только параметры из names (по разу); пустые группы — если drop_empty.
inline void pg_prune(ParamGroups& g, const std::vector<std::string>& names, bool drop_empty) {
    const std::set<std::string> known(names.begin(), names.end());
    std::set<std::string> seen;
    for (ParamGroup& x : g)
        x.params.erase(std::remove_if(x.params.begin(), x.params.end(),
                                      [&](const std::string& p) { return !known.count(p) || !seen.insert(p).second; }),
                       x.params.end());
    if (drop_empty)
        g.erase(std::remove_if(g.begin(), g.end(), [](const ParamGroup& x) { return x.params.empty(); }), g.end());
}

// После импорта схемы: что пользователь уже разложил, остаётся на месте; новые параметры идут в
// группы импорта; исчезнувшие параметры и опустевшие группы убираются.
inline void pg_merge_auto(ParamGroups& g, const std::vector<std::string>& names,
                          const std::vector<std::pair<std::string, std::string>>& auto_groups) {
    pg_prune(g, names, false);
    std::set<std::string> placed;
    for (const ParamGroup& x : g) placed.insert(x.params.begin(), x.params.end());
    for (const auto& pa : auto_groups) {
        const std::string p = pg_normalize_path(pa.second);
        if (p.empty() || placed.count(pa.first)) continue;
        pg_get(g, p).params.push_back(pa.first);
        placed.insert(pa.first);
    }
    pg_prune(g, names, true);
}

inline std::string pg_to_text(const ParamGroups& g) {
    std::string s;
    for (const ParamGroup& x : g) {
        if (!s.empty()) s += '\n';
        s += x.path + ":";
        for (size_t i = 0; i < x.params.size(); ++i) s += (i ? ", " : " ") + x.params[i];
    }
    return s;
}

inline ParamGroups pg_from_text(const std::string& text) {
    ParamGroups g;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t e = text.find('\n', pos);
        if (e == std::string::npos) e = text.size();
        const std::string line = text.substr(pos, e - pos);
        pos = e + 1;
        const size_t c = line.find(':');
        if (c == std::string::npos) continue;
        const std::string path = pg_normalize_path(line.substr(0, c));
        if (path.empty()) continue;
        ParamGroup& x = pg_get(g, path);
        std::string cur;
        for (size_t i = c + 1; i <= line.size(); ++i) {
            const char ch = i < line.size() ? line[i] : ',';
            if (ch == ',') { if (!cur.empty()) x.params.push_back(cur); cur.clear(); }
            else if (ch != ' ' && ch != '\t' && ch != '\r') cur += ch;
        }
    }
    return g;
}
