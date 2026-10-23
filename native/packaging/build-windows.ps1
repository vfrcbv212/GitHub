param([Parameter(Mandatory=$true)][string]$QtRoot,[Parameter(Mandatory=$true)][string]$DependencyRoot)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $projectRoot
$ortRoot = Join-Path $DependencyRoot 'ort'
$destination = Join-Path $projectRoot 'native-dist\VolleyScope'
cmake -S native -B native-build -G 'Visual Studio 17 2022' -A x64 "-DCMAKE_PREFIX_PATH=$QtRoot" "-DONNXRUNTIME_ROOT=$ortRoot"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
cmake --build native-build --config Release --parallel 3
if ($LASTEXITCODE -ne 0) { throw 'Native C++ build failed' }
New-Item -ItemType Directory -Force $destination | Out-Null
Copy-Item native-build\Release\VolleyScope.exe $destination
Copy-Item (Join-Path $ortRoot 'runtimes\win-x64\native\*.dll') $destination
Copy-Item (Join-Path $DependencyRoot 'directml\bin\x64-win\DirectML.dll') $destination
& (Join-Path $QtRoot 'bin\windeployqt.exe') --release --compiler-runtime --no-translations (Join-Path $destination 'VolleyScope.exe')
if ($LASTEXITCODE -ne 0) { throw 'Qt deployment failed' }
$vsLocator = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsInstallation = & $vsLocator -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$crt = Get-ChildItem (Join-Path $vsInstallation 'VC\Redist\MSVC') -Directory -Recurse -Filter 'Microsoft.VC143.CRT' | Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending | Select-Object -First 1
if ($null -eq $crt) { throw 'Portable MSVC runtime not found' }
Copy-Item (Join-Path $crt.FullName '*.dll') $destination -Force
foreach ($plugin in @('platforms\qoffscreen.dll','sqldrivers\qsqlite.dll')) {
    $target = Join-Path $destination $plugin
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    Copy-Item (Join-Path $QtRoot ('plugins\' + $plugin)) $target -Force
}
$ffmpegRoot = (Get-ChildItem (Join-Path $DependencyRoot 'ffmpeg') -Directory | Select-Object -First 1).FullName
Copy-Item (Join-Path $ffmpegRoot 'bin') (Join-Path $destination 'ffmpeg') -Recurse
Copy-Item native\models (Join-Path $destination 'models') -Recurse
Copy-Item native\README.md,native\SOURCES.md,LICENSE $destination
Copy-Item native\LICENSES (Join-Path $destination 'LICENSES') -Recurse
foreach ($folder in @((Join-Path $QtRoot 'Licenses'),(Join-Path $DependencyRoot 'ort'),(Join-Path $DependencyRoot 'directml'),$ffmpegRoot)) {
    if (-not (Test-Path $folder)) { continue }
    Get-ChildItem $folder -File -Recurse | Where-Object { $_.Name -match '^(LICENSE|COPYING|NOTICE|LGPL|GPL)' -and $_.Length -lt 200000 } | ForEach-Object {
        $name = ($folder | Split-Path -Leaf) + '-' + $_.Name
        Copy-Item $_.FullName (Join-Path $destination ('LICENSES\' + $name)) -Force
    }
}
$source = Join-Path $destination 'source\native'
New-Item -ItemType Directory -Force $source | Out-Null
foreach ($name in @('app','core','tests','tools','packaging','LICENSES')) { Copy-Item (Join-Path 'native' $name) (Join-Path $source $name) -Recurse }
Copy-Item native\CMakeLists.txt,native\README.md,native\SOURCES.md $source
Copy-Item LICENSE (Split-Path $source)
@'
VolleyScope Native 0.2 — экспериментальная версия C++.
Распакуйте всю папку и запустите VolleyScope.exe. Python не требуется.
Первый шаг: Открыть видео → Площадка и состав → образцы основных и запасных.
Подробности: README.md. Исходники и условия моделей входят в папку.
30 FPS на целевой видеокарте пока не подтверждены.
'@ | Set-Content (Join-Path $destination 'START.txt') -Encoding UTF8
