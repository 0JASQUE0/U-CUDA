# U-CUDA — Chaotic Dynamics on CUDA

Анализ нелинейных динамических систем на CUDA + ImGui. Пользователь задаёт систему ОДУ (LaTeX или текст, можно с фото через OCR), приложение генерирует CUDA-код, интегрирует на GPU и рисует фазовые портреты 2D/3D.

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

- **Windows 10/11 x64**
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

Команда `git clone <url> <папка>` — это **одна команда**: первый аргумент — что клонировать, второй (опциональный) — куда. Без второго аргумента создастся папка с именем репо в текущей директории.

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
