#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ParityRun,
    [string]$MainBuild = "$PSScriptRoot/../../build/main-parity",
    [string]$ModularBuild = "$PSScriptRoot/../../build/windows",
    [string]$Config = "$PSScriptRoot/../test_config.json"
)
$ErrorActionPreference = 'Stop'
$ParityRun = [IO.Path]::GetFullPath($ParityRun)
$correctness = Get-Content -Raw -LiteralPath "$ParityRun/metadata.json" | ConvertFrom-Json
if ($correctness.status -ne 'identical') { throw 'Complete exact correctness comparison before measuring speed' }
$expected = @()
foreach ($backend in @('cpu','gpu')) {
    foreach ($method in @('Euler','RK4','DOPRI78','ComplexCD')) { $expected += "${backend}_phase_$method/real_time" }
    $expected += "${backend}_solver_bdf_fixed_5/real_time"
    $expected += "${backend}_solver_DAE_Radau_IIA__5_/real_time"
}
$expected += @('gpu_analysis_network/real_time','gpu_network_kernel/manual_time')
$expected += @('cpu_cold_compile_RK4/real_time','gpu_cold_compile_RK4/real_time')
$pattern = '^(' + (($expected | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')$'
$configText = Get-Content -Raw -LiteralPath $Config
$configuration = $configText | ConvertFrom-Json
$p = $configuration.performance
if ($p.series -lt 3 -or $p.repetitions -lt 3 -or $p.minimum_seconds -le 0 -or $p.warmup_seconds -le 0) { throw 'Invalid measurement configuration' }
$bin = @{ main=[IO.Path]::GetFullPath("$MainBuild/bin/Release/main_benchmark.exe"); modular=[IO.Path]::GetFullPath("$ModularBuild/bin/Release/ucuda_benchmark.exe") }
$tool = [IO.Path]::GetFullPath("$ModularBuild/bin/Release/ucuda_test_tool.exe")
foreach ($file in @($bin.main,$bin.modular,$tool)) { if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Build $file first" } }
$directory = "$ParityRun/performance/$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"
if (Test-Path -LiteralPath $directory) { throw 'Measurement directory already exists' }
[void][IO.Directory]::CreateDirectory($directory)
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText("$directory/config.json", $configText, $utf8)
$metadata = [ordered]@{ status='incomplete'; started_utc=[DateTimeOffset]::UtcNow.ToString('o'); expected=$expected; protocol=$configuration; executables=@{} }
foreach ($variant in @('main','modular')) {
    $metadata.executables[$variant] = @{ path=$bin[$variant]; sha256=(Get-FileHash -LiteralPath $bin[$variant] -Algorithm SHA256).Hash }
    [void][IO.Directory]::CreateDirectory("$directory/$variant/cache")
}
function Save-Metadata { [IO.File]::WriteAllText("$directory/metadata.json", ($metadata | ConvertTo-Json -Depth 30), $utf8) }
Save-Metadata
$saved = @{}
foreach ($key in @('TMP','TEMP','UCUDA_BENCHMARK_BACKEND')) { $saved[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$cases = @{ main=@{}; modular=@{} }
try {
    $env:UCUDA_BENCHMARK_BACKEND = 'all'
    for ($series = 0; $series -lt $p.series; ++$series) {
        foreach ($variant in @('main','modular')) {
            $stage = "$directory/$variant"
            $env:TMP = "$stage/cache"; $env:TEMP = "$stage/cache"
            $output = "$stage/benchmark-$series.json"
            Write-Output "Measuring $variant series $($series + 1)/$($p.series): $directory"
            $warmup = ([double]$p.warmup_seconds).ToString([Globalization.CultureInfo]::InvariantCulture)
            $minimum = ([double]$p.minimum_seconds).ToString([Globalization.CultureInfo]::InvariantCulture)
            & $bin[$variant] "--benchmark_filter=$pattern" "--benchmark_min_warmup_time=$warmup" "--benchmark_min_time=${minimum}s" `
                "--benchmark_repetitions=$($p.repetitions)" "--benchmark_out=$output" --benchmark_out_format=json *> "$stage/benchmark-$series.log"
            if ($LASTEXITCODE -ne 0) { throw "Benchmark process failed: $variant series $series" }
            $report = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
            if (@($report.benchmarks | Where-Object { $_.error_occurred }).Count) { throw 'Independent benchmark correctness failed' }
            $rows = @($report.benchmarks | Where-Object { $_.run_type -eq 'iteration' })
            foreach ($name in $expected) {
                $samples = @($rows | Where-Object { $_.run_name -eq $name })
                if ($samples.Count -ne $p.repetitions) { throw "Incomplete timing repetitions: $variant/$name" }
                $id = "perf.$name"
                $inputs = @{ benchmark=$name; unit=$samples[0].time_unit }
                foreach ($field in $samples[0].PSObject.Properties | Where-Object { $_.Name.StartsWith('input_') }) { $inputs[$field.Name] = $field.Value }
                if (-not $cases[$variant].ContainsKey($id)) {
                    $cases[$variant][$id] = @{ id=$id; version=[int]$samples[0].input_workload_version; inputs=$inputs; values=@(); passed=$true; series=@() }
                }
                $timings = @($samples | ForEach-Object { [double]$_.real_time })
                if (@($timings | Where-Object { -not [double]::IsFinite($_) -or $_ -le 0 }).Count) { throw 'Invalid timing sample' }
                $cases[$variant][$id].series += ,$timings
            }
            if (@($rows | Select-Object -ExpandProperty run_name -Unique).Count -ne $expected.Count) { throw 'Unexpected benchmark coverage' }
        }
    }
    $environment = @{ cpu=$correctness.cpu; os=$correctness.os; gpu=(Get-Content -Raw -LiteralPath "$ParityRun/gpu.csv"); protocol='Same pinned Google Benchmark, Release, same CPU/GPU, FMA/RDC defaults, interleaved series' }
    foreach ($variant in @('main','modular')) {
        $run = @{ schema=1; checks_passed=$true; environment=$environment; configuration=$configuration; cases=$cases[$variant] }
        [IO.File]::WriteAllText("$directory/$variant/run.json", ($run | ConvertTo-Json -Depth 30), $utf8)
    }
    & $tool compare --run "$directory/modular" --baseline "$directory/main/run.json" --config "$directory/config.json" *> "$directory/comparison.json"
    $metadata['comparison_exit_code'] = $LASTEXITCODE
    $metadata.status = if ($LASTEXITCODE -eq 0) { 'pass' } else { 'regression_or_inconclusive' }
    $metadata['finished_utc'] = [DateTimeOffset]::UtcNow.ToString('o')
    Save-Metadata
    Write-Output "Performance status: $($metadata.status); report: $directory"
    if ($metadata.comparison_exit_code -ne 0) { exit 1 }
} catch {
    $metadata.status = 'failed'; $metadata['error'] = $_.Exception.Message; Save-Metadata
    throw
} finally {
    foreach ($key in $saved.Keys) { [Environment]::SetEnvironmentVariable($key,$saved[$key],'Process') }
}
