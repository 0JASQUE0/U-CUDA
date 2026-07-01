#include "digit_input.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>

namespace DigitInput {

// Общий сплит-хелпер для 'e'/'E' и '/'. sep_pos — индекс разделителя в text.
// Каретка слева от разделителя (cursor <= sep_pos) → работаем со «львой»
// частью, иначе с «правой». Рекурсивно вызываем ComputeStep для выделенного
// куска — там сработает обычная ветка (без разделителя → plain decimal).
// Части, которые остаются без изменений, склеиваются обратно, курсор
// пересчитывается с учётом сдвига.
static bool ComputeStepSplit(const std::string& text, int cursor, int direction,
                             int sep_pos, std::string& new_text, int& new_cursor);

bool ComputeStep(const std::string& text, int cursor, int direction,
                 std::string& new_text, int& new_cursor) {
    if (text.empty()) return false;
    if (direction != +1 && direction != -1) return false;

    // Диспатч: сначала научная нотация (e/E), затем дробь (/). После диспатча
    // управление ушло в ComputeStepSplit → рекурсия в ComputeStep для куска,
    // где 'e' / '/' уже отсутствует.
    size_t e_pos = text.find_first_of("eE");
    if (e_pos != std::string::npos)
        return ComputeStepSplit(text, cursor, direction, (int)e_pos,
                                new_text, new_cursor);

    size_t slash_pos = text.find('/');
    if (slash_pos != std::string::npos)
        return ComputeStepSplit(text, cursor, direction, (int)slash_pos,
                                new_text, new_cursor);

    // Валидация plain-decimal: только [+-]? digits и не более одной '.'.
    // Отсекаем пробелы и прочее — там семантика неоднозначная. 'e/E' и '/'
    // сюда уже не долетят — их обработал диспатч выше.
    int dot = -1;
    int first_digit = -1, last_digit = -1;
    int sign_at = -1;
    for (int i = 0; i < (int)text.size(); ++i) {
        char c = text[i];
        if (c == '+' || c == '-') {
            if (i != 0) return false;      // знак только в начале
            sign_at = i;
            continue;
        }
        if (c == '.') {
            if (dot >= 0) return false;    // только одна точка
            dot = i;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        if (first_digit < 0) first_digit = i;
        last_digit = i;
    }
    if (first_digit < 0) return false;

    // Если точки нет — виртуально ставим её сразу после последней цифры.
    const int effective_dot = (dot >= 0) ? dot : (last_digit + 1);

    // Клампим курсор.
    int c = cursor;
    if (c < 0) c = 0;
    if (c > (int)text.size()) c = (int)text.size();

    auto is_dig = [&](int i) -> bool {
        return i >= 0 && i < (int)text.size()
            && std::isdigit(static_cast<unsigned char>(text[i]));
    };

    // Выбираем "затронутую" цифру = ближайшая слева от каретки. Если слева
    // цифры нет (каретка в самом начале / слева знак / слева точка) — fallback
    // на ближайшую справа. Знак и точка цифрами не считаются.
    int affected = -1;
    for (int i = c - 1; i >= 0; --i) if (is_dig(i)) { affected = i; break; }
    if (affected < 0)
        for (int i = c; i < (int)text.size(); ++i)
            if (is_dig(i)) { affected = i; break; }
    if (affected < 0) return false;

    // Разряд (power of 10) выбранной цифры.
    int power;
    if (affected < effective_dot) power =  effective_dot - 1 - affected;
    else                          power = -(affected - effective_dot);

    // Парсим значение.
    double value;
    try { value = std::stod(text); }
    catch (...) { return false; }

    const double step = std::pow(10.0, power) * static_cast<double>(direction);
    const double new_value = value + step;

    // Форматирование: сохраняем число дробных знаков; если power отрицательный
    // и глубже, чем текущее число знаков — расширяем (иначе digit «исчезнет»).
    int frac_digits = 0;
    if (dot >= 0) frac_digits = (int)text.size() - dot - 1;
    if (power < 0 && -power > frac_digits) frac_digits = -power;

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", frac_digits, new_value);
    new_text = buf;

    // Курсор — СПРАВА от цифры того же разряда в новой строке. Это сохраняет
    // инвариант "цифра слева от каретки = редактируемая", поэтому повторные
    // ↑/↓ продолжают крутить тот же разряд без смещения по позициям.
    int new_dot = (int)new_text.find('.');
    if (new_dot < 0) new_dot = (int)new_text.size();
    if (power >= 0) new_cursor = new_dot - power;
    else            new_cursor = new_dot + (-power) + 1;
    if (new_cursor < 0) new_cursor = 0;
    if (new_cursor > (int)new_text.size()) new_cursor = (int)new_text.size();

    return true;
}

static bool ComputeStepSplit(const std::string& text, int cursor, int direction,
                             int sep_pos, std::string& new_text, int& new_cursor) {
    std::string sub;
    int sub_cursor;
    std::string new_sub;
    int new_sub_cursor = 0;

    if (cursor <= sep_pos) {
        // Работаем с левой частью (мантисса / числитель).
        sub = text.substr(0, static_cast<size_t>(sep_pos));
        sub_cursor = cursor;
        if (!ComputeStep(sub, sub_cursor, direction, new_sub, new_sub_cursor))
            return false;
        new_text = new_sub + text.substr(static_cast<size_t>(sep_pos));
        new_cursor = new_sub_cursor;
    } else {
        // Работаем с правой частью (экспонента / знаменатель).
        sub = text.substr(static_cast<size_t>(sep_pos + 1));
        sub_cursor = cursor - (sep_pos + 1);
        if (!ComputeStep(sub, sub_cursor, direction, new_sub, new_sub_cursor))
            return false;
        new_text = text.substr(0, static_cast<size_t>(sep_pos + 1)) + new_sub;
        new_cursor = (sep_pos + 1) + new_sub_cursor;
    }
    return true;
}

} // namespace DigitInput
