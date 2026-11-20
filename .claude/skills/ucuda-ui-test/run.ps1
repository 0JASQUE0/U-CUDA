# Wrapper: the script lives in the repository (apps\desktop\ui_test\scripts\run.ps1).
& (Join-Path $PSScriptRoot '..\..\..\apps\desktop\ui_test\scripts\run.ps1') @args
exit $LASTEXITCODE
