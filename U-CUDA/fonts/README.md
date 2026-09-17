# fonts\

Шрифт для LaTeX-подписей на графиках (Settings → Font → «LaTeX-style labels on plots»).

В репозитории лежит **Computer Modern Unicode** — тот самый шрифт, которым LaTeX
набирает текст по умолчанию:

| файл | начертание |
|---|---|
| `cmunrm.otf` | прямое — цифры тиков, названия функций, знаки |
| `cmunti.otf` | курсив — переменные и строчные греческие буквы |

Взяты с CTAN (`fonts/cm-unicode`, версия 0.7.0), лицензия — OFL, полный текст в
`OFL.txt`. Покрывают латиницу, греческий и кириллицу. Знаков ∞ и ∂ в них нет: в CM
они живут в отдельных математических начертаниях, поэтому такие символы приложение
рисует UI-шрифтом (см. `font_with_glyph` в `plot_axis.cpp`).

Каталог в сборку не входит, `.vcxproj` его не копирует — приложение ищет шрифты в
рантайме по той же схеме, что и `library\`: `U_CUDA_FONTS` → `fonts\` рядом с exe →
dev-раскладка через vcxproj → dev-раскладка через solution. Для сборки, которая
раздаётся отдельно от исходников, каталог нужно положить рядом с exe.

## Заменить на другой шрифт

Достаточно положить сюда одну из пар — ищутся в таком порядке:

| прямой | курсив | что это |
|---|---|---|
| `cmunrm.otf` | `cmunti.otf` | Computer Modern Unicode (по умолчанию) |
| `cmunrm.ttf` | `cmunti.ttf` | он же из TTF-сборки |
| `lmroman10-regular.otf` | `lmroman10-italic.otf` | Latin Modern (кириллицы нет) |
| `LatinModernRoman-Regular.ttf` | `LatinModernRoman-Italic.ttf` | Latin Modern, вариант именования |
| `texgyretermes-regular.otf` | `texgyretermes-italic.otf` | TeX Gyre Termes (клон Times) |

Если ни одной пары нет, режим работает и берёт Times New Roman из `C:\Windows\Fonts`.
Курсив не обязателен: без него переменные набираются прямым начертанием.
