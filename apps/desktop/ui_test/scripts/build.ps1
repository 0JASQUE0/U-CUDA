# Builds U-CUDA with Dear ImGui Test Engine into $env:TEMP\ucb-ui (CMake, Release).
# Usage (from anywhere): & <repo>\apps\desktop\ui_test\scripts\build.ps1 [-Reconfigure]
#
# cmake: $env:UCUDA_CMAKE if set, otherwise the CMake bundled with the newest Visual Studio that
# has the C++ tools (found with vswhere), otherwise cmake from PATH.
param([switch]$Reconfigure)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$b = Join-Path $env:TEMP 'ucb-ui'
$log = Join-Path $env:TEMP 'ucb-ui-build.log'

function Find-CMake {
    if ($env:UCUDA_CMAKE -and (Test-Path $env:UCUDA_CMAKE)) { return $env:UCUDA_CMAKE }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                         -property installationPath
        if ($vs) {
            $cm = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
            if (Test-Path $cm) { return $cm }
        }
    }
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'cmake not found: install Visual Studio 2022 with C++ CMake tools, or set UCUDA_CMAKE'
}
$cm = Find-CMake

Push-Location $repo
try {
    if ($Reconfigure -or -not (Test-Path (Join-Path $b 'CMakeCache.txt'))) {
        & $cm --preset windows -B $b -DU_CUDA_UI_TESTS=ON *> (Join-Path $env:TEMP 'ucb-ui-configure.log')
        if ($LASTEXITCODE -ne 0) { throw "configure failed, see $env:TEMP\ucb-ui-configure.log" }
    }
    & $cm --build $b --config Release --target U-CUDA -- /m /nr:false /v:minimal *> $log
    $code = $LASTEXITCODE
} finally { Pop-Location }

$errors = Select-String -Path $log -Pattern 'error C\d+|error LNK\d+|fatal error|error MSB' | Select-Object -First 20
if ($code -ne 0 -or $errors) {
    $errors | ForEach-Object { $_.Line }
    throw "build failed (exit $code), full log: $log"
}
$exe = Join-Path $b 'bin\Release\U-CUDA.exe'
if (-not (Test-Path $exe)) { throw "exe not found: $exe" }
"OK $exe"
