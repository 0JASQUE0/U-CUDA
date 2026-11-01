#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$MainBuild = "$PSScriptRoot/../../build/main-parity",
    [string]$ModularBuild = "$PSScriptRoot/../../build/windows",
    [string]$MainSource = "$PSScriptRoot/../../build/main-parity-source",
    [string]$OutputRoot = "$PSScriptRoot/../../build/upstream-parity"
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath("$PSScriptRoot/../..")
$mainBin = [IO.Path]::GetFullPath("$MainBuild/bin/Release")
$modularBin = [IO.Path]::GetFullPath("$ModularBuild/bin/Release")
$suites = @(
    @{ Name='regression'; Main='main_regression'; Modular='ucuda_regression'; Args=@('--gtest_filter=-ArtifactPolicyCpu.*:ParityComparatorCpu.*') },
    @{ Name='dae'; Main='main_dae'; Modular='ucuda_dae_tests'; Args=@() },
    @{ Name='observations'; Main='main_observations'; Modular='ucuda_parity_observations'; Args=@() }
)
foreach ($name in @('krs_cpu','adaptive','architecture','sessions','tabs','network','library','spice','prune')) {
    $suites += @{ Name=$name; Main="main_$name"; Modular="upstream_$name"; Args=@() }
}
foreach ($suite in $suites) {
    foreach ($exe in @("$mainBin/$($suite.Main).exe", "$modularBin/$($suite.Modular).exe")) {
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Build the required executable first: $exe" }
    }
}
$comparator = "$modularBin/ucuda_parity_compare.exe"
if (-not (Test-Path -LiteralPath $comparator -PathType Leaf)) { throw "Build ucuda_parity_compare first" }
$run = Join-Path ([IO.Path]::GetFullPath($OutputRoot)) ([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds().ToString())
if (Test-Path -LiteralPath $run) { throw "Run directory already exists: $run" }
[void][IO.Directory]::CreateDirectory($run)
$utf8 = [Text.UTF8Encoding]::new($false)
function Write-Json([string]$Path, $Value) { [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 20), $utf8) }
$metadata = [ordered]@{ started_utc=[DateTimeOffset]::UtcNow.ToString('o'); status='incomplete'; suites=@() }
foreach ($variant in @('main','modular')) {
    $source = if ($variant -eq 'main') { [IO.Path]::GetFullPath($MainSource) } else { $repo }
    $revision = & git -C $source rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Cannot identify $variant source" }
    $tree = & git -C $source status --porcelain=v1
    if ($LASTEXITCODE -ne 0) { throw "Cannot capture $variant working tree" }
    $metadata[$variant] = @{ source=$source; revision=$revision; working_tree=@($tree) }
    & git -C $source diff HEAD --binary --no-ext-diff *> "$run/$variant-working-tree.patch"
    if ($LASTEXITCODE -ne 0) { throw "Cannot capture $variant source differences" }
    $build = if ($variant -eq 'main') { $MainBuild } else { $ModularBuild }
    Copy-Item -LiteralPath "$build/CMakeCache.txt" -Destination "$run/$variant-CMakeCache.txt"
    $bin = if ($variant -eq 'main') { $mainBin } else { $modularBin }
    $metadata[$variant]['runtime_sha256'] = @(Get-ChildItem -LiteralPath "$bin/kernels" -File | ForEach-Object {
        @{ name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
    })
    $metadata[$variant]['executables_sha256'] = @($suites | ForEach-Object {
        $name = if ($variant -eq 'main') { $_.Main } else { $_.Modular }
        @{ name=$name; sha256=(Get-FileHash -LiteralPath "$bin/$name.exe" -Algorithm SHA256).Hash }
    })
}
& nvidia-smi --query-gpu=name,uuid,driver_version --format=csv *> "$run/gpu.csv"
if ($LASTEXITCODE -ne 0) { throw "A real NVIDIA GPU is required" }
$metadata['cpu'] = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
$metadata['os'] = [Environment]::OSVersion.VersionString
$metadata['environment'] = @{}
foreach ($key in @('CUDA_VISIBLE_DEVICES','CUDA_PATH','CUDA_CACHE_DISABLE','UCUDA_NVRTC_PRUNE','UCUDA_NVRTC_LINEINFO','UCUDA_NVRTC_MAXRREG')) {
    $metadata.environment[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
}
$metadata['test_sources_sha256'] = @(Get-ChildItem -LiteralPath "$repo/tests" -Recurse -File | Where-Object { $_.Extension -in @('.cpp','.h','.json','.cir','.ps1','.cmake') } | ForEach-Object {
    @{ path=[IO.Path]::GetRelativePath($repo,$_.FullName); sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
})
Write-Json "$run/metadata.json" $metadata
Write-Output "Parity run: $run"
$saved = @{}
foreach ($key in @('TMP','TEMP','UCUDA_TEST_RESULT_DIR','U_CUDA_LIBRARY')) { $saved[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
try {
    foreach ($variant in @('main','modular')) {
        $stage = "$run/$variant"
        foreach ($directory in @('outputs/numeric','cache','user-library')) { [void][IO.Directory]::CreateDirectory("$stage/$directory") }
        $env:TMP = "$stage/cache"; $env:TEMP = "$stage/cache"
        $env:UCUDA_TEST_RESULT_DIR = "$stage/outputs/numeric"
        $env:U_CUDA_LIBRARY = "$stage/user-library"
        foreach ($suite in $suites) {
            $exe = if ($variant -eq 'main') { "$mainBin/$($suite.Main).exe" } else { "$modularBin/$($suite.Modular).exe" }
            $arguments = @($suite.Args)
            if ($suite.Name -eq 'observations') { $arguments += @('--compatibility-report', "$stage/compatibility.json") }
            if ($suite.Name -in @('regression','dae')) { $arguments += "--gtest_output=xml:$stage/$($suite.Name).xml" }
            Write-Output "Running $variant/$($suite.Name)"
            $watch = [Diagnostics.Stopwatch]::StartNew()
            & $exe @arguments *> "$stage/$($suite.Name).log"
            $result = $LASTEXITCODE; $watch.Stop()
            $metadata.suites += @{ variant=$variant; name=$suite.Name; exit_code=$result; elapsed_ms=$watch.ElapsedMilliseconds }
            Write-Json "$run/metadata.json" $metadata
            Write-Output "$variant/$($suite.Name): exit $result"
        }
    }
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key,$saved[$key],'Process') }
}
& $comparator "$run/main/outputs" "$run/modular/outputs" "$run/comparison-exact.json" *> "$run/comparison-exact.log"
$comparisonExit = $LASTEXITCODE
$metadata['comparison_exit_code'] = $comparisonExit
$metadata['finished_utc'] = [DateTimeOffset]::UtcNow.ToString('o')
$failed = @($metadata.suites | Where-Object { $_.exit_code -ne 0 }).Count -ne 0
$metadata.status = if ($failed -or $comparisonExit -ne 0) { 'failed_or_different' } else { 'identical' }
Write-Json "$run/metadata.json" $metadata
Get-Content -LiteralPath "$run/comparison-exact.log"
Write-Output "Status: $($metadata.status); report: $run"
if ($failed -or $comparisonExit -ne 0) { exit 1 }
