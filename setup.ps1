# setup.ps1 — настройка Python-окружения для OCR-сервера U-CUDA.
# Создаёт .venv в корне репо, ставит requirements.txt, прописывает U_CUDA_PYTHON.
# CUDA и vcpkg не трогает — про них см. README.md.

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$venvDir  = Join-Path $repoRoot '.venv'
$venvPy   = Join-Path $venvDir  'Scripts\python.exe'
$reqFile  = Join-Path $repoRoot 'requirements.txt'

Write-Host "=== U-CUDA setup ===" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"

# 1. Найти Python 3.10+
function Find-Python {
    foreach ($cmd in @('py -3.10', 'py -3', 'python', 'python3')) {
        try {
            $parts = $cmd -split ' '
            $exe   = $parts[0]
            $args  = if ($parts.Length -gt 1) { $parts[1..($parts.Length-1)] } else { @() }
            $ver = & $exe @args -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $major, $minor = $ver.Trim().Split('.')
                if ([int]$major -eq 3 -and [int]$minor -ge 10) {
                    return ,@($exe, $args)
                }
            }
        } catch { }
    }
    return $null
}

$pyCmd = Find-Python
if (-not $pyCmd) {
    Write-Host "Python 3.10+ не найден." -ForegroundColor Red
    Write-Host "Установи: https://www.python.org/downloads/ (или 'winget install Python.Python.3.10')"
    exit 1
}
$pyExe, $pyArgs = $pyCmd
Write-Host "Python OK: $pyExe $pyArgs" -ForegroundColor Green

# 2. Создать venv
if (Test-Path $venvPy) {
    Write-Host "venv уже существует: $venvDir" -ForegroundColor Yellow
} else {
    Write-Host "Создаю venv в $venvDir ..."
    & $pyExe @pyArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv" }
}

# 3. Установить зависимости (тяжёлый шаг — torch и pix2text)
if (-not (Test-Path $reqFile)) {
    Write-Host "requirements.txt не найден: $reqFile" -ForegroundColor Red
    exit 1
}
Write-Host "Устанавливаю зависимости (это надолго, ~5 ГБ — torch + onnx) ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install упал" }

# 4. Прописать U_CUDA_PYTHON в окружение пользователя
Write-Host "Прописываю U_CUDA_PYTHON=$venvPy для текущего пользователя ..." -ForegroundColor Cyan
[Environment]::SetEnvironmentVariable('U_CUDA_PYTHON', $venvPy, 'User')
$env:U_CUDA_PYTHON = $venvPy

Write-Host ""
Write-Host "=== Готово ===" -ForegroundColor Green
Write-Host "U_CUDA_PYTHON = $venvPy"
Write-Host "Перезапусти Visual Studio / терминал, чтобы переменная подхватилась."
Write-Host "Дальше: открой U-CUDA.sln, собери Release | x64."
