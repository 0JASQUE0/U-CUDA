# U-CUDA — Chaotic Dynamics on CUDA

Анализ нелинейных динамических систем на CUDA + ImGui. Пользователь задаёт систему ОДУ (LaTeX или текст, можно с фото через OCR), приложение генерирует CUDA-код, интегрирует на GPU и рисует фазовые портреты 2D/3D.

## CMake: Arch Linux и Windows

Общая сборка описана в `CMakeLists.txt`, профили — в `CMakePresets.json`.
ImGui (docking), ImPlot и ImPlot3D загружаются автоматически в закреплённых версиях.
GLFW берётся из системы либо собирается из исходников. При первой конфигурации нужен интернет.

В CMake **Debug и Release собирают графическое приложение** `U-CUDA`.
Старый расчётный сценарий включается отдельно: `-DU_CUDA_BUILD_LEGACY=ON`,
цель `U-CUDA-legacy`. Исходная сборка через `.sln` также сохранена;
в ней Debug по-прежнему означает старый сценарий.

### Arch Linux

```bash
sudo pacman -S --needed base-devel cmake ninja cuda glfw gdb zenity wl-clipboard xclip ttf-dejavu
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
./build/linux-debug/bin/U-CUDA
```

Для оптимизированной сборки замени `linux-debug` на `linux-release`.
Профили Arch используют `/opt/cuda/bin/nvcc` и `/usr/bin/g++-15`
(пакет `gcc15` устанавливается как зависимость CUDA). CUDA 13.3 проверена
с текущими заголовками Arch; более старый Toolkit может не поддерживать новую glibc.
Для другого расположения Toolkit/компилятора создай `CMakeUserPresets.json`
или передай `-DCMAKE_CUDA_COMPILER=... -DCMAKE_CUDA_HOST_COMPILER=... -DCUDAToolkit_ROOT=...`.

На этой рабочей копии дополнительно настроены локальные профили `linux-local`
и `linux-local-release`: CUDA/GCC и библиотеки лежат в игнорируемом `build/`,
системная установка для них не нужна. Запуск:

```bash
cmake --preset linux-local
cmake --build --preset linux-local
ctest --preset linux-local
./build/linux-local133/bin/U-CUDA
```

Эти локальные профили и Toolkit не входят в Git. Для новой копии используй
обычную установку Arch выше. `zenity` нужен для файловых диалогов,
`wl-clipboard` — для PNG в буфере обмена Wayland, `xclip` — для X11.
Графический запуск требует активной сессии Wayland/X11 и OpenGL 3.3;
GPU-расчёты требуют работающего драйвера NVIDIA (`nvidia-smi`).

### Windows через CMake

Нужны CMake 3.25+, Git, Visual Studio 2022 с C++ и CUDA Toolkit
с интеграцией Visual Studio. `CUDA_PATH` должен указывать на Toolkit,
а `%CUDA_PATH%\bin` — присутствовать в `PATH` для CUDA DLL.
vcpkg для этого способа не требуется.

```powershell
cmake --preset windows
cmake --build --preset windows-debug
.\build\windows\bin\Debug\U-CUDA.exe
# Оптимизированная версия:
cmake --build --preset windows-release
.\build\windows\bin\Release\U-CUDA.exe
```

Профили по умолчанию используют CUDA-архитектуру `75` (RTX 2080).
Для другой видеокарты переопредели `CMAKE_CUDA_ARCHITECTURES` при конфигурации.
На Windows конфигурация подготовлена, но её сборку следует проверить на Windows:
Linux-проверки не заменяют проверку MSVC.

### VS Code / Code OSS

1. Открой **корень репозитория**, установи рекомендованные расширения.
2. Выполни **CMake: Select Configure Preset**: на этой машине `linux-local`,
   на Arch с системной CUDA `linux-debug`, на Windows `windows`.
3. Выбери цель запуска **U-CUDA** через **CMake: Set Debug Target**.
4. **Ctrl+Shift+B** собирает приложение. В панели Run and Debug выбери
   **U-CUDA (Linux)** или **U-CUDA (Windows)** и нажми **F5**.

Linux использует Native Debug (`webfreak.debug`) и GDB, подсказки — clangd.
CMake Tools копирует `compile_commands.json` в `build/` для clangd.
Для Windows-профиля отладки нужен Microsoft C/C++ (`ms-vscode.cpptools`)
из каталога обычного VS Code; в каталоге Code OSS он может отсутствовать.
Для отладки устройства CUDA нужен отдельный CUDA-отладчик; F5 здесь отлаживает CPU-код.

Ресурсы `kernels/`, `library_template/` и `ocr_server.py` копируются рядом
с исполняемым файлом. При работе из исходников пользовательская библиотека
остаётся в `U-CUDA/library/`. `CUDA_PATH` может переопределить путь к заголовкам
NVRTC; без него CMake передаёт путь найденного Toolkit.

OCR опционален. На Linux можно создать `.venv` в корне проекта и установить
зависимости из `requirements.txt`; при необходимости задай `U_CUDA_PYTHON`
путём к Python с установленным pix2text. Windows-команды ниже относятся
к исходному способу установки. Без Python-зависимостей приложение запускается,
но распознавание изображений недоступно.

Тест `platform` проверяет Linux IPC, протокол OCR с тестовым процессом,
компиляцию и выполнение CPU-схем, ошибки компиляции. `gpu_smoke` проверяет
NVRTC, траекторию и параметрическую развёртку на GPU; без доступной CUDA
он помечается как пропущенный.

## Исходная сборка Windows через Visual Studio

## Обновление существующей копии

Если проект уже клонирован и нужно подтянуть новые правки:

```powershell
cd <путь-к-репо>
git checkout main
git pull

# почистить кэш VS и старые сборочные артефакты (gitignored, но мешают)
Remove-Item -Recurse -Force .vs, x64 -ErrorAction SilentlyContinue
```

Открой `U-CUDA.sln` в Visual Studio → **Build → Rebuild Solution**.

## Системные требования

- **Windows 10/11 x64** (для исходной сборки `.sln`; Linux — см. CMake выше)
- **Visual Studio 2022** (toolset v143, C++17)
- **NVIDIA GPU** + **CUDA Toolkit** (см. ниже про версию)
- **vcpkg** (manifest-режим — зависимости в [vcpkg.json](vcpkg.json))
- **Python 3.10** (для OCR-сервера, опционально — без него работает всё, кроме распознавания с фото)

## Установка

### 0. Получить исходники

```powershell
# выбери папку, в которой будет лежать проект, например D:\
cd D:\
git clone https://github.com/0JASQUE0/U-CUDA.git
cd U-CUDA
```

### 1. Зависимости C++ через vcpkg

```powershell
# если vcpkg ещё не установлен — клонируем в C:\vcpkg (можно в любое место):
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg integrate install
```

> **Важно:** перед `vcpkg integrate install` закрой Visual Studio, после — перезапусти. Иначе VS не подхватит интеграцию.

После `integrate install` Visual Studio автоматически подхватит [vcpkg.json](vcpkg.json) при сборке и установит `glfw3`, `imgui`, `implot`, `implot3d` (5-15 минут при первом ребилде).

**Если при сборке ловишь `cannot open source file "imgui.h"` или подобное** — интеграция vcpkg не сработала. Проверь:

```powershell
C:\vcpkg\vcpkg integrate install
# должно вывести: "Applied user-wide integration for this vcpkg root."
```

Закрой VS, открой `U-CUDA.sln` заново, **Build → Rebuild Solution**. В Output-окне сборки сначала должно появиться `vcpkg install ...` (скачивание/компиляция зависимостей), и только потом твой код. Если этого нет — `vcpkg integrate install` не отработал (запущен под другим пользователем, не от твоего имени, или vcpkg сломан).

### 2. CUDA Toolkit

Проект собран под **CUDA 12.8**, но работает на любой версии **12.x и 13.x** — используются только стандартные Runtime API + NVRTC.

> **CUDA 13:** в 13.0 у `cuCtxCreate` поменялась сигнатура (добавился параметр `CUctxCreateParams*`). В коде это разрулено через `#if CUDA_VERSION >= 13000` в [nvrtc_engine.cpp](U-CUDA/nvrtc_engine.cpp), так что собирается и на 12.x, и на 13.x без правок.

Если у тебя другая версия CUDA, поменяй в `U-CUDA\U-CUDA.vcxproj` две строки (или через Visual Studio: **ПКМ на проекте → Build Dependencies → Build Customizations…** — снять `CUDA 12.8`, поставить установленную):

```xml
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.props" />
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 12.8.targets" />
```

Проверить установленные версии:
```powershell
ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
```

### 3. Python OCR (опционально)

OCR-сервер распознаёт уравнения с фото через `pix2text` (около 100 пакетов, включая torch — ~5 ГБ). Если OCR не нужен — пропусти этот шаг, приложение запустится без него.

PowerShell по умолчанию запрещает выполнение локальных `.ps1` скриптов. Один раз разреши себе:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Согласись на изменение (`Y`). После этого:

```powershell
.\setup.ps1
```

Скрипт создаст `.venv` в корне репо, поставит [requirements.txt](requirements.txt) и пропишет переменную `U_CUDA_PYTHON` для текущего пользователя.

### 4. Сборка

Открой `U-CUDA.sln` в Visual Studio 2022 → выбери конфигурацию **Release | x64** → **Build**.

Исполняемый файл будет в `x64\Release\U-CUDA.exe`. Рядом с ним Post-Build Event автоматически положит `ocr_server.py` и `library\` — эта папка с .exe и сопутствующими файлами полностью переносима.

## Как приложение находит Python

В порядке приоритета:
1. Переменная окружения **`U_CUDA_PYTHON`** (полный путь к `python.exe`) — это ставит `setup.ps1`
2. **`<exe_dir>\.venv\Scripts\python.exe`** (если venv лежит рядом с .exe)
3. **`python`** из `PATH`

## Структура

- `U-CUDA\` — исходники C++/CUDA
- `U-CUDA\library\` — пресеты систем (Lorenz, Rossler, Chen, …)
- `U-CUDA\ocr_server.py` — Python-сервер распознавания LaTeX с фото
- `vcpkg.json` — манифест C++ зависимостей
- `requirements.txt` — Python-зависимости OCR
- `setup.ps1` — bootstrap Python venv + установка зависимостей
