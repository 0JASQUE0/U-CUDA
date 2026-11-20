# Wrapper: the script lives in the repository (apps\desktop\ui_test\scripts\build.ps1).
# The skill folder may be reached through a junction (~/.claude/skills): resolve it to the repository.
$here = Get-Item $PSScriptRoot
if ($here.Target) { $here = Get-Item (@($here.Target)[0]) }
$repo = (Resolve-Path (Join-Path $here.FullName '..\..\..')).Path
& (Join-Path $repo 'apps\desktop\ui_test\scripts\build.ps1') @args
