# ucuda_run: ключи job-файла и форматы выхода

Строка — `key = value`, `#` — комментарий до конца строки, ключи без учёта регистра.
Повтор ключа дописывает значение (нужно только для `rhs`); `key=value` в командной строке заменяет.
Числа — выражения: `8/3`, `2*pi`, `1e-6`, `(1+sqrt(5))/2`, `-inf`. Списки — через запятую.

## Система

| ключ | умолч. | смысл |
|---|---|---|
| `task` | — | `bif1d lle1d ls1d bif2d lle2d ls2d basins metrics dft1d` |
| `vars` | — | переменные состояния → X[0..N-1] |
| `params` | пусто | параметры → a[1..M], в этом порядке |
| `rhs` | — | правая часть, по строке на переменную в порядке `vars` (обычный синтаксис кодогена) |
| `values` | — | значения параметров, в порядке `params` |
| `ic` | — | начальные условия, в порядке `vars` |
| `s` | 0.5 | a[0], коэффициент симметрии (CD, SEMP, SIMP, Complex CD...) |
| `scheme` | RK4 | имя схемы (список ниже) |
| `map` | 0 | дискретное отображение: rhs = x_{n+1}, схема игнорируется |
| `body_file` | — | готовое тело `calculateDiscreteModel` вместо кодогена (`rhs` тогда не обязательны) |
| `dump_body` | 0 | записать тело в `<out>_body.cu` |
| `newton_tol` / `newton_max_iters` / `newton_full` | 1e-10 / 8 / 0 | Ньютон неявных схем |
| `cd_fp_iters` | 4 | итерации CD FP / CD fast FP |

Схемы (`scheme_from_name`): `Euler`, `Euler-Cromer`, `Explicit Midpoint`, `RK4`, `DOPRI78`,
`DOPRI78 (legacy)`, `RK45`, `DOP853`, `CD`, `CD fast`, `CD FP`, `CD fast FP`, `CD_{1-0}`,
`Complex CD`, `Complex CD4`, `CCD4 (o4s3)`, `CCD4 (o4s4)`, `CCD4 (s-split)_{0-1}`,
`CCD4 (s-split)_{1-0}`, `Complex CD_{1-0}`, `Complex CD4_{1-0}`, `CCD4 (o4s3)_{1-0}`,
`CCD4 (o4s4)_{1-0}`, `Implicit Euler`, `Implicit Midpoint`, `SEMP`, `SIMP`, `D`,
`Complex Implicit Euler`, `GBS (n=2)`, `GBS 2-4`, `GBS 2-4-6`, `GBS 2-4-6-8`, `GBS 2-4-6-8-10`,
`GBS 2-4-6-8-10-12`. Имя с пробелами в командной строке PowerShell — в кавычках: `scheme="CD fast"`.

## Интегрирование и свип

| ключ | умолч. | смысл |
|---|---|---|
| `h` | 0.01 | шаг |
| `transient` | 0 (basins: 1000) | время переходного процесса (отбрасывается) |
| `tmax` | 100 | время записи / усреднения после transient |
| `max_value` | 1e6 | порог расходимости |
| `sweep` | — | ось X: имя параметра, переменной (свип по НУ), `h` или `s` |
| `range` | — | `lo, hi` оси X |
| `log` | 0 | логарифмическая сетка оси X (lo, hi > 0) |
| `sweep_y`, `range_y`, `log_y` | — | ось Y для `*2d`, `basins`, 2D-`metrics` |
| `n` | 500 | точек по оси (2D: сетка n×n) |
| `out_var` | первая из `vars` | переменная для пиков / метрик / DFT |
| `prescaler` | 1 | писать каждый k-й шаг (bif, basins, metrics, dft) |
| `continuation` / `reverse` / `cpu` | 0 | продолжение по параметру (1D), обратный ход, счёт на CPU |
| `out` | — | префикс выходных файлов |
| `precision` | 12 | значащих цифр в CSV |
| `gui_csv` | 0 | дополнительно CSV движка в формате экспорта GUI (`<out>_gui...`) |

Задачи:
- `bif1d`, `bif2d`: `all_samples` (0; 1 — все итерации, для отображений); `bif2d` ещё `eps_dbscan` (0.1),
  `mult_peak`, `mult_interval` (из configCUDA.h; `mult_interval = 0` — кластеризация только по амплитуде).
- `lle1d`, `ls1d`, `lle2d`, `ls2d`: `nt` (1.0, время между ренормализациями), `eps` (1e-4, возмущение),
  `vector_transient` (0, разгон касательных векторов).
- `basins`: `sweep`/`sweep_y` — только переменные, `range`/`range_y`; `eps_dbscan` (0.5),
  `feature1`/`feature2` (0/1), `mult1`/`mult2`. Признаки: 0 avg peaks, 1 avg intervals, 2/3 RMS,
  4/5 stdev, 6–11 те же в log, 12 signed-log avg peaks.
- `metrics`: 1D или 2D (если задан `sweep_y`); `metric_mask` (все), `minmax_interp` (0).
- `dft1d`: `n_freq` (200), `freq_range` (`0, 10`), `freq_log` (0), `window` (0 нет, 1 Ханн, 2 Хэмминг,
  3 Блэкман, 4 Блэкман-Харрис).

Глобальные настройки (в GUI — Settings): `rdc` (1), `fmad` (1), `block` (32), `peaks_interpolate`,
`eps_fixed_point`, `eps_peak_delta`, `eps_interpeak_delta`, `peak_threshold`, `max_peaks` —
умолчания пиков из `configCUDA.h`.

## Выход

Все файлы — `<out>_<task>.csv` (первая строка — заголовок) и `<out>_run.txt`.
2D-сетки — построчно, ось X меняется быстрее: ячейка `c` → `ix = c % n`, `iy = c / n`.

| task | столбцы |
|---|---|
| `bif1d` | `<sweep>, flag, value, interval` — строка на пик; при `flag <= 0` одна строка с `nan` |
| `lle1d` | `<sweep>, lle, flag` |
| `ls1d` | `<sweep>, LE1..LEk, flag` |
| `bif2d` | `<sweep>, <sweep_y>, period, flag` |
| `lle2d` | `<sweep>, <sweep_y>, lle, flag` |
| `ls2d` | `<sweep>, <sweep_y>, LE1..LEk, flag` |
| `basins` | `<x>, <y>, basin, regime, avg_peak, avg_interval` |
| `metrics` | `x[, y], regime, Max, Min, ..., Complexity` (`signal_metrics_write_csv`) |
| `dft1d` | `<sweep>, freq, amplitude, ak_cos, bk_sin, flag` |

Флаги: БД — `-1` равновесие, `0` уход за `max_value`, `N > 0` число пиков (bif1d) / период (bif2d);
LLE/LS — `1` колебания, `0` расходимость (равновесие не детектируется); бассейны: `basin >= 1`
кластер колебаний, `<= -1` кластер равновесия, `0` уход; `regime` −1/0/1. Неизмеримые показатели
Ляпунова (флаги ядра 999/−999) — `nan`; `avg_peak = 999` у бассейнов — «нет пиков».
Координаты: `lo + (hi-lo)*i/(n-1)` (или log-сетка); при `reverse = 1` точки 1D идут от `hi` к `lo`.
