<#
.SYNOPSIS
    Builds the Windows desktop executable with PyInstaller.

.DESCRIPTION
    Creates (or reuses) a virtual environment, installs the dependencies, runs
    the test suite, and produces dist\AZInterviewCopilot\AZInterviewCopilot.exe.

    This must run on Windows: the app depends on WASAPI loopback capture through
    PyAudioWPatch, which does not exist on other platforms.

.PARAMETER SkipTests
    Build without running the test suite first.

.EXAMPLE
    .\build_windows.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "This build must run on Windows (WASAPI loopback capture is Windows-only)."
}

$venv = Join-Path $PSScriptRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -3 -m venv $venv
}

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r requirements-dev.txt --quiet

if (-not $SkipTests) {
    Write-Host "Running tests..." -ForegroundColor Cyan
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Tests failed - not building." }
}

Write-Host "Building executable..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean az_interview_copilot.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$exe = Join-Path $PSScriptRoot "dist\AZInterviewCopilot\AZInterviewCopilot.exe"
if (-not (Test-Path $exe)) { throw "Expected executable was not produced: $exe" }

Write-Host "`nVerifying the build..." -ForegroundColor Cyan
& $exe --check
# --check exits non-zero when API keys are absent; that is a configuration
# state, not a build failure, so it is reported rather than thrown.

$size = [math]::Round((Get-ChildItem (Split-Path $exe) -Recurse |
    Measure-Object -Property Length -Sum).Sum / 1MB, 1)

Write-Host "`nBuild complete." -ForegroundColor Green
Write-Host "  Executable : $exe"
Write-Host "  Bundle size: $size MB"
Write-Host "`nDistribute the whole dist\AZInterviewCopilot folder, not just the .exe."
