# Сборка ucuda_run.exe: раннер + объектники Release-сборки U-CUDA (U-CUDA\x64\Release\*.obj).
# kernels\ берётся из выхода Release (x64\Release\kernels) — ровно то, что грузит U-CUDA.exe, —
# и обновляется при каждом вызове. Перелинковка — только если объектники или раннер новее exe.
#   .\build.ps1 [-Repo <repository>] [-Out <dir>] [-Force]   (default: the repository holding this skill)
param(
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path,
    [string]$Out  = "$PSScriptRoot\bin",
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$objDir = "$Repo\U-CUDA\x64\Release"
$src    = "$Repo\U-CUDA"
$objs   = 'parametric_engine', 'data_export', 'krs_cpu', 'codegen', 'sysparse'
$runner = "$PSScriptRoot\runner\ucuda_run.cpp", "$PSScriptRoot\runner\stubs.cpp"

foreach ($o in $objs) {
    if (-not (Test-Path "$objDir\$o.obj")) { throw "нет $objDir\$o.obj — сначала собрать Release (U-CUDA.sln)" }
    if ((Get-Item "$src\$o.cpp").LastWriteTime -gt (Get-Item "$objDir\$o.obj").LastWriteTime) {
        Write-Warning "$o.cpp новее $o.obj — Release не пересобран, раннер возьмёт старый код"
    }
}
if (-not (Test-Path "$Repo\x64\Release\kernels")) { throw "нет $Repo\x64\Release\kernels — сначала собрать Release" }

New-Item -ItemType Directory -Force $Out | Out-Null
# kernels\ рядом с exe: движок ищет шаблоны в каталоге СВОЕГО exe
if (Test-Path "$Out\kernels") { Remove-Item -Recurse -Force "$Out\kernels" }
Copy-Item -Recurse "$Repo\x64\Release\kernels" "$Out\kernels"

$exe = "$Out\ucuda_run.exe"
$inputs = @($objs | ForEach-Object { "$objDir\$_.obj" }) + $runner
$newest = ($inputs | ForEach-Object { (Get-Item $_).LastWriteTime } | Measure-Object -Maximum).Maximum
if (-not $Force -and (Test-Path $exe) -and (Get-Item $exe).LastWriteTime -gt $newest) {
    "up to date: $exe (kernels обновлены)"
    exit 0
}

$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vsw) { & $vsw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath } else { $null }
if (-not $vs) { $vs = 'C:\Program Files\Microsoft Visual Studio\2022\Professional' }
$vc = "$vs\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vc)) { throw "не найден vcvars64.bat ($vc)" }

# Флаги как у ClCompile Release проекта: /MD, /utf-8, без /GL (WholeProgramOptimization = false)
$fl  = "/nologo /EHsc /std:c++17 /O2 /MD /utf-8 /DWIN32 /DWIN64 /DNDEBUG /I`"$src`" /I`"$src\include`" /I`"%CUDA_PATH%\include`""
$ob  = ($objs | ForEach-Object { "`"$objDir\$_.obj`"" }) -join ' '
$rs  = ($runner | ForEach-Object { "`"$_`"" }) -join ' '
$lk  = "/link /LIBPATH:`"%CUDA_PATH%\lib\x64`" cudart_static.lib nvrtc.lib cuda.lib"
New-Item -ItemType Directory -Force "$Out\obj" | Out-Null
$log = "$Out\build.log"
cmd /c "call `"$vc`" >nul 2>&1 && cd /d `"$Out\obj`" && cl $fl $rs $ob /Fe:`"$exe`" $lk" *> $log
if ($LASTEXITCODE -ne 0) {
    Select-String -Path $log -Pattern 'error|fatal' | Select-Object -First 20 | ForEach-Object { $_.Line }
    throw "сборка не удалась, лог: $log"
}
"built: $exe"
