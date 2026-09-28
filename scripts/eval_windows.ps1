# Evaluate the untrained model and the newest SFT adapter on the held-out test split, then write
# outputs\eval\report.md. Usage (from the repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\eval_windows.ps1
# Pass -Adapter to evaluate a specific run instead of the newest one.

param(
    [string]$Adapter = "",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step([string]$Name, [scriptblock]$Cmd) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "no .venv found; run scripts\train_windows.ps1 first" -ForegroundColor Red
    exit 1
}
$py = (Resolve-Path ".venv\Scripts\python.exe").Path

if (-not $SkipInstall) {
    Step "install evaluation extras" { & $py -m pip install -e ".[dev,train,eval]" }
}

if ($Adapter -eq "") {
    $run = Get-ChildItem outputs\sft -Directory -Filter "sft-r*" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "adapter") } |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($null -eq $run) {
        Write-Host "no finished SFT run with an adapter in outputs\sft" -ForegroundColor Red
        exit 1
    }
    $Adapter = Join-Path $run.FullName "adapter"
}
Write-Host "adapter: $Adapter"

Step "evaluate base model" { & $py -m echolm.cli eval run --model base --name base }
Step "evaluate SFT adapter" { & $py -m echolm.cli eval run --model $Adapter --name sft }
Step "write report" { & $py -m echolm.cli eval report }
