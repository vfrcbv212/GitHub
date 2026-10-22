param([Parameter(Mandatory=$true)][string]$Project)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& '.\.venv\Scripts\python.exe' -m volleyvision.cli gui $Project
exit $LASTEXITCODE
