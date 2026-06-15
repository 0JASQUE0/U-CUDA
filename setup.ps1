# setup.ps1 -- bootstrap Python environment for the U-CUDA OCR server.
# Creates .venv in the repo root, installs requirements.txt, sets U_CUDA_PYTHON.
# CUDA and vcpkg are not touched here -- see README.md.
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
# (cp1251 on Russian locale) unless there is a UTF-8 BOM, so non-ASCII text
# breaks the parser on a fresh machine.

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$venvDir  = Join-Path $repoRoot '.venv'
$venvPy   = Join-Path $venvDir  'Scripts\python.exe'
$reqFile  = Join-Path $repoRoot 'requirements.txt'

Write-Host "=== U-CUDA setup ===" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"

# 1. Locate Python 3.10+
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
    Write-Host "Python 3.10+ not found." -ForegroundColor Red
    Write-Host "Install it: https://www.python.org/downloads/ (or 'winget install Python.Python.3.10')"
    exit 1
}
$pyExe, $pyArgs = $pyCmd
Write-Host "Python OK: $pyExe $pyArgs" -ForegroundColor Green

# 2. Create venv
if (Test-Path $venvPy) {
    Write-Host "venv already exists: $venvDir" -ForegroundColor Yellow
} else {
    Write-Host "Creating venv at $venvDir ..."
    & $pyExe @pyArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
}

# 3. Install dependencies (heavy: torch + pix2text, ~5 GB)
if (-not (Test-Path $reqFile)) {
    Write-Host "requirements.txt not found at $reqFile" -ForegroundColor Red
    exit 1
}
Write-Host "Installing dependencies (this takes a while, ~5 GB: torch + onnx) ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r $reqFile
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4. Persist U_CUDA_PYTHON for the current user
Write-Host "Setting U_CUDA_PYTHON=$venvPy for current user ..." -ForegroundColor Cyan
[Environment]::SetEnvironmentVariable('U_CUDA_PYTHON', $venvPy, 'User')
$env:U_CUDA_PYTHON = $venvPy

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "U_CUDA_PYTHON = $venvPy"
Write-Host "Restart Visual Studio / terminal so the env var is picked up."
Write-Host "Next: open U-CUDA.sln in Visual Studio and build Release | x64."
