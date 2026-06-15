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
//
// Бросает std::runtime_error при ошибке разбора.
System parse_system_from_latex(const std::string& multiline_latex,
    const std::vector<std::string>& alphabet,
    ParamOrder order = ParamOrder::AsInAlphabet,
    const FuncDefs& funcs = {});

// Разбирает текст определений функций (по одному на строку):
//   h(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)
//   g(s) = ...
// Возвращает словарь определений. Бросает при синтаксической ошибке.
FuncDefs parse_func_defs(const std::string& text);