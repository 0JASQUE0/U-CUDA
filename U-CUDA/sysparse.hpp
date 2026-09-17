#pragma once
#include "codegen.hpp"
#include <string>
#include <vector>
#include <set>
#include <map>

// Порядок индексации параметров в a[1..M].
enum class ParamOrder {
    AsInAlphabet,   // в порядке, заданном пользователем в алфавите
    AsInSystem      // в порядке первого появления в правых частях
};

// Определение вспомогательной функции: имя -> (формальные аргументы, тело).
// Напр. h(x) = m1*x + ...  ->  {"h", {{"x"}, "m1*x + ..."}}
struct FuncDef {
    std::vector<std::string> args;  // формальные параметры, напр. {"x"}
    std::string body;               // тело-выражение в LaTeX/обычном синтаксисе
};
using FuncDefs = std::map<std::string, FuncDef>;

// Разбирает многострочный LaTeX в System.
//
// alphabet — символы системы (переменные + параметры).
// order    — порядок индексации параметров.
// funcs    — определения вспомогательных функций для инлайнинга.
//            Если пусто (по умолчанию), инлайнинг не выполняется и
//            любой неизвестный вызов f(...) обрабатывается как раньше.
//            Если задано, вызовы f(...) в правых частях заменяются телом f
//            с подстановкой аргументов.
// discrete_map — the system is a map x_{n+1} = f(x_n): there is no derivative
//            in the LHS, the iteration index is stripped before parsing, and
//            the LHS itself is a bare variable name.
//
// Бросает std::runtime_error при ошибке разбора.
System parse_system_from_latex(const std::string& multiline_latex,
    const std::vector<std::string>& alphabet,
    ParamOrder order = ParamOrder::AsInAlphabet,
    const FuncDefs& funcs = {},
    bool discrete_map = false);

// Разбирает текст определений функций (по одному на строку):
//   h(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)
//   g(s) = ...
// Возвращает словарь определений. Бросает при синтаксической ошибке.
FuncDefs parse_func_defs(const std::string& text);

// Результат авто-распознавания алфавита по тексту LaTeX/Plain.
struct DetectedAlphabet {
    std::vector<std::string> vars;    // символы внутри производных
    std::vector<std::string> params;  // остальные идентификаторы
};

// Сканирует текст и классифицирует все идентификаторы:
//   - всё, что встречается внутри \dot{X} / \frac{dX}{dt} / X' / dX/dt → vars
//   - все остальные идентификаторы (буквы + греки типа \alpha) → params
// Из params исключаются: math-функции (sin/cos/exp/log/...) и LaTeX-команды
// (\frac/\dot/\left/...). Порядок в выводе — по первому появлению в тексте.
// Дубликаты не повторяются.
//
// discrete_map — map mode: there are no derivatives, so vars come from the LHS
// of each equation and the iteration symbol (n) is excluded from params.
DetectedAlphabet detect_alphabet(const std::string& text, bool discrete_map = false);