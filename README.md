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

# если в requirements.txt появились новые пакеты — обновить venv
.\setup.ps1
```

Открой `U-CUDA.sln` в Visual Studio → **Build → Rebuild Solution**.

> **Раньше работал с этим проектом под именем FDS?** После переименования в U-CUDA измени привычки:
> - Запускай `U-CUDA.sln` (не `FDS.sln` — его больше нет)
> - Переменная окружения для Python — `U_CUDA_PYTHON` (не `FDS_PYTHON`); старую удали через **Параметры системы → Переменные среды**
> - Старая папка `.vs\FDS\` и артефакты в `x64\` от старой сборки не мешают, но место занимают — почисти

## Системные требования

- **Windows 10/11 x64**
- **Visual Studio 2022** (toolset v143, C++17)
- **NVIDIA GPU** + **CUDA Toolkit** (см. ниже про версию)
- **vcpkg** (manifest-режим — зависимости в [vcpkg.json](vcpkg.json))
- **Python 3.10** (для OCR-сервера, опционально — без него работает всё, кроме распознавания с фото)

## Установка

### 1. Зависимости C++ через vcpkg

```powershell
# если vcpkg ещё не установлен:
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg integrate install
```

После `integrate install` Visual Studio автоматически подхватит [vcpkg.json](vcpkg.json) при сборке и установит `glfw3`, `imgui`, `implot`, `implot3d`.

### 2. CUDA Toolkit

Проект собран под **CUDA 12.8**, но должен работать на любой версии 12.x — используются только стандартные Runtime API + NVRTC.

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
