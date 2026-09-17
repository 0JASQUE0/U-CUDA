# fonts\

Сюда кладётся шрифт для LaTeX-подписей на графиках (Settings → Font → «LaTeX-style
labels on plots»). Каталог нужен только ради самого шрифта: в сборку он не входит,
`.vcxproj` его не копирует, приложение ищет файлы в рантайме.

Без шрифта режим работает и берёт **Times New Roman** из `C:\Windows\Fonts` — засечки
те же по духу, но это не Computer Modern.

## Настоящий TeX-шрифт

Положить в этот каталог одну из пар (ищутся в таком порядке):

| прямой | курсив | что это |
|---|---|---|
| `cmunrm.ttf` | `cmunti.ttf` | Computer Modern Unicode — тот самый шрифт LaTeX |
| `lmroman10-regular.otf` | `lmroman10-italic.otf` | Latin Modern |
| `LatinModernRoman-Regular.ttf` | `LatinModernRoman-Italic.ttf` | Latin Modern, вариант именования |
| `texgyretermes-regular.otf` | `texgyretermes-italic.otf` | TeX Gyre Termes (клон Times) |

Computer Modern Unicode (OFL) покрывает латиницу, греческий и кириллицу — для подписей
вида σ, λ, x₁ этого достаточно. Latin Modern кириллицы не содержит.

Курсив не обязателен: без него переменные набираются прямым начертанием.

Другой каталог задаётся переменной окружения `U_CUDA_FONTS` (ищется первым).
