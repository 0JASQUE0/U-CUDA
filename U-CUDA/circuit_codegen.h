#pragma once
#include <string>
#include <vector>
#include "circuit_solver.h"

// Кодогенерация схемного решателя под NVRTC (фаза 3 плана).
//
// Топология известна на этапе кодогенерации, поэтому символьное исключение
// делается ОДИН РАЗ на хосте: порядок исключения и позиции fill-in считаются по
// структуре, без чисел. В ядро идёт прямолинейный код — матрицы как структуры
// данных там нет.
//
// Граница между тем, что запекается, и тем, что остаётся рантайм-ным: запекается
// СТРУКТУРА (порядок исключения, позиции fill-in, набор живых скаляров),
// номиналы и параметры моделей приходят массивом-аргументом. Иначе каждая точка
// бифуркационной диаграммы означала бы отдельную компиляцию.

// Позиция в матрице.
struct MatEntry {
    int row = 0, col = 0;
};

// Структурный портрет якобиана: какие клетки могут быть ненулевыми ХОТЬ КОГДА.
// Строится по раскладке, а не по одному снимку значений: у умножителя
// производная обращается в ноль при нулевом входе, но клетка остаётся живой.
void mna_pattern(const MnaLayout& lay, std::vector<MatEntry>& out);

struct EliminationStep {
    int pivot_row = 0;
    int pivot_col = 0;
    // Строки, которые этот шаг обновляет, и столбцы, по которым идёт обновление.
    std::vector<int> rows;
    std::vector<int> cols;
};

struct EliminationPlan {
    std::vector<EliminationStep> steps;
    int       n            = 0;
    long long nnz_original = 0;
    long long nnz_fill_in  = 0;
    long long nnz_factor   = 0;   // ненулей в L+U
    long long nnz_u        = 0;   // ненулей в U: её хранить обязаны, обратная подстановка читает
    long long nnz_l        = 0;   // ненулей в L: при исключении на [A|b] хранить НЕ надо
    long long peak_active  = 0;   // пик ненулей активной подматрицы
    long long mul_ops      = 0;   // умножений-сложений в факторизации
    int       max_row_len  = 0;
    bool      diagonal_free = false;  // есть ли строки без диагонального элемента
};

// Жадный Марковиц по структуре. Диагональным пивотингом обойтись нельзя:
// строка-ограничение идеального ОУ несёт клетки только на входах, и её
// диагональ структурно нулевая.
bool plan_elimination(const std::vector<MatEntry>& pattern, int n,
                      EliminationPlan& out, std::string* err);

std::string elimination_report(const EliminationPlan& p);

// Проверка структурного портрета против реальной сборки: портрет обязан быть
// НАДМНОЖЕСТВОМ фактических ненулей. Ловит расхождение между mna_pattern и
// CircuitSolver::assemble — то самое, из-за чего GPU и CPU считали бы разное.
std::string circuit_codegen_selftest(bool* ok);
