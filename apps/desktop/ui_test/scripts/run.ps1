# Runs the UI-test build of U-CUDA on a script or on compiled tests; prints the important lines.
# Usage:
#   & <repo>\apps\desktop\ui_test\scripts\run.ps1 -Script C:\path\check.uit [-Out C:\path\out] [-Fresh] [-Speed fast|normal|cinematic] [-Extra '--ui-size','1920x1080']
#   & <repo>\apps\desktop\ui_test\scripts\run.ps1 -Test all
param(
    [string]$Script = '',
    [string]$Test = '',
    [string]$Out = '',
    [switch]$Fresh,
    [string]$Speed = 'fast',
    [int]$TimeoutSec = 900,
    [string[]]$Extra = @()
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $env:TEMP 'ucb-ui\bin\Release\U-CUDA.exe'
if (-not (Test-Path $exe)) { throw "no UI-test build, run build.ps1 first ($exe)" }
if (-not $Script -and -not $Test) { throw 'give -Script or -Test' }
if (-not $Out) {
    $base = if ($Script) { [IO.Path]::GetFileNameWithoutExtension($Script) } else { 'tests' }
    $Out = Join-Path $env:TEMP "ucb-ui-out\$base"
}
New-Item -ItemType Directory -Force $Out | Out-Null
$argv = @('--ui-out', $Out, '--ui-speed', $Speed)
if ($Script) { $argv += @('--ui-script', (Resolve-Path $Script).Path) }
if ($Test)   { $argv += @('--ui-test', $Test) }
if ($Fresh)  { $argv += '--ui-fresh' }
$argv += $Extra

$log = Join-Path $Out 'log.txt'
$quoted = $argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
$p = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory (Split-Path $exe) `
    -RedirectStandardOutput $log -RedirectStandardError (Join-Path $Out 'stderr.txt') -PassThru -WindowStyle Normal
$null = $p.Handle   # PS 5.1: without a cached handle ExitCode stays empty
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Stop-Process -Id $p.Id -Force
    "TIMEOUT after $TimeoutSec s, process killed. Log: $log"
    Get-Content $log -Tail 30
    exit 3
}
$code = $p.ExitCode
# Command echo, command output, errors and the summary; the rest of the engine log stays in log.txt.
Get-Content $log | Where-Object { $_ -match '^\[ui|KO|Error|error|ERROR|stopped at' }
$err = Join-Path $Out 'stderr.txt'
if ((Test-Path $err) -and (Get-Item $err).Length -gt 0) { '--- stderr ---'; Get-Content $err -Tail 20 }
"exit $code; out: $Out"
exit $code
