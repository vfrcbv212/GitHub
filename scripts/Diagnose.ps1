$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& '.\.venv\Scripts\python.exe' -m volleyvision.cli diagnose --out diagnostics.json
exit $LASTEXITCODE
