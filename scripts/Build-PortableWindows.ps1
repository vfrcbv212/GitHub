param()
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot
function Check-NativeExit {
    if ($LASTEXITCODE -ne 0) { throw "Command failed: exit $LASTEXITCODE" }
}
$buildRoot = Join-Path $repoRoot 'work\windows-native'
New-Item -ItemType Directory -Force $buildRoot | Out-Null
$pythonArchive = Join-Path $buildRoot 'python-3.12.10.zip'
Invoke-WebRequest 'https://api.nuget.org/v3-flatcontainer/python/3.12.10/python.3.12.10.nupkg' -OutFile $pythonArchive
Expand-Archive $pythonArchive (Join-Path $buildRoot 'cpython') -Force
$runtimeRoot = Join-Path $buildRoot 'cpython\tools'
$runtimePython = Join-Path $runtimeRoot 'python.exe'
$env:PYTHONPATH = $repoRoot
$env:VOLLEYVISION_BUNDLE_ROOT = $null
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $runtimePython -m pip install 'pip==26.2.2' 'setuptools==82.0.1' wheel
Check-NativeExit
& $runtimePython -m pip install --no-deps --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.13.0' 'torchvision==0.28.0'
Check-NativeExit
& $runtimePython -m pip install -c packaging/windows/portable-constraints.txt -r packaging/windows/runtime-requirements.txt 'pytest==9.1.1'
Check-NativeExit
& $runtimePython -m pip check
Check-NativeExit
& $runtimePython -m pytest -q --junitxml=work/native-windows-tests.xml 2>&1 | Tee-Object work/native-windows-tests.txt
Check-NativeExit
$modelProject = Join-Path $buildRoot 'model-project'
& $runtimePython -m volleyvision.cli init $modelProject --title 'Build models'
Check-NativeExit
& $runtimePython -m volleyvision.cli fetch-models $modelProject
Check-NativeExit
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Check-NativeExit
if (-not $vsRoot) { throw 'MSVC x64 build tools not found' }
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
$launcher = Join-Path $buildRoot 'VolleyVision.exe'
$batchPath = Join-Path $buildRoot 'compile-launcher.cmd'
$batch = @"
@echo off
call "$vcvars" >nul
if errorlevel 1 exit /b 1
cl /nologo /O2 /MT /utf-8 /DUNICODE /D_UNICODE "$repoRoot\packaging\windows\launcher.c" user32.lib /link /SUBSYSTEM:WINDOWS /ENTRY:wWinMainCRTStartup /OUT:"$launcher"
exit /b %errorlevel%
"@
[IO.File]::WriteAllText($batchPath, $batch, [Text.Encoding]::ASCII)
& cmd.exe /c $batchPath
Check-NativeExit
$outRoot = Join-Path $repoRoot 'work\portable\VolleyVision'
$modelsRoot = Join-Path $modelProject 'models'
& $runtimePython scripts/build_windows_portable.py --runtime $runtimeRoot --launcher $launcher --models $modelsRoot --out $outRoot
Check-NativeExit
$selfTestOut = Join-Path $repoRoot 'work\native-check'
$selfTestArguments = @('--self-test', '--out', ('"{0}"' -f $selfTestOut))
$process = Start-Process -FilePath (Join-Path $outRoot 'VolleyVision.exe') -ArgumentList $selfTestArguments -WorkingDirectory $outRoot -PassThru -Wait
if ($process.ExitCode -ne 0) { throw "EXE self-test failed: exit $($process.ExitCode)" }
$check = Get-Content (Join-Path $selfTestOut 'result.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $check.offline -or -not $check.bundled_weights_verified -or -not $check.gui_opened -or $check.synthetic_pipeline -ne 'complete') {
    throw 'EXE self-test did not confirm required checks'
}
$evidence = [ordered]@{
    runner_os = $env:RUNNER_OS
    windows_version = [Environment]::OSVersion.VersionString
    commit = $env:GITHUB_SHA
    exe_exit_code = $process.ExitCode
    native_windows_ci = $true
    target_user_pc_tested = $false
    nvidia_tested = $false
    self_test = $check
}
New-Item -ItemType Directory -Force 'docs/validation' | Out-Null
$evidence | ConvertTo-Json -Depth 20 | Set-Content docs/validation/native-windows-ci.json -Encoding utf8
$archive = Join-Path $repoRoot 'work\VolleyVision-0.3-Windows-x64.zip'
$resultText = & $runtimePython scripts/build_windows_portable.py --runtime $runtimeRoot --launcher $launcher --models $modelsRoot --out $outRoot --zip $archive
Check-NativeExit
$resultText | Set-Content work/windows-release-result.json -Encoding utf8
