param([ValidateSet('cpu','cu126','cu130','cu132')][string]$Compute='cpu')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } }
if (-not (Test-Path '.venv\Scripts\python.exe')) { py -3.12 -m venv .venv; Check-Exit }
$Python=Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $Python -m pip install 'pip==26.2.2'; Check-Exit
& $Python -m pip install -c requirements-windows.txt 'torch==2.13.0' 'torchvision==0.28.0' --index-url "https://download.pytorch.org/whl/$Compute"; Check-Exit
& $Python -m pip install -c requirements-windows.txt -e '.[cv,dev]'; Check-Exit
& $Python -m pip check; Check-Exit
& $Python -m volleyvision.cli diagnose --out diagnostics.json; Check-Exit
Write-Output 'Environment ready. See README.md; CUDA still requires a working NVIDIA driver.'
