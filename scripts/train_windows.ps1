# Full SFT run on Windows: install, rebuild the dataset, preflight, smoke test, train.
# Usage (from the repo root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\train_windows.ps1 -Me "swapnil"
# Add -SkipInstall on later runs once the packages are in place.

param(
    [Parameter(Mandatory = $true)][string]$Me,
    [switch]$SkipInstall
)

# native tools write progress to stderr; "Stop" would turn that into a fatal error in PowerShell 5.1,
# so every step checks its exit code instead
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
    Step "create venv (python 3.12)" { py -3.12 -m venv .venv }
}
$py = (Resolve-Path ".venv\Scripts\python.exe").Path

if (-not $SkipInstall) {
    Step "upgrade pip" { & $py -m pip install --upgrade pip }
    Step "install CUDA torch" {
        & $py -m pip install "torch>=2.10,<2.13" torchvision --index-url https://download.pytorch.org/whl/cu128
    }
    Step "install echolm with training extras" { & $py -m pip install -e ".[dev,train]" }
    $cuda = & $py -c "import torch; print(torch.cuda.is_available())"
    if ($cuda -ne "True") {
        # unsloth's dependencies can pull a CPU-only torch from PyPI; put the CUDA build back
        $ver = & $py -c "import torch; print(torch.__version__.split('+')[0])"
        Step "restore CUDA torch $ver" {
            & $py -m pip install --force-reinstall --no-deps "torch==$ver" --index-url https://download.pytorch.org/whl/cu128
        }
    }
}

Write-Host "`n=== unit tests (not blocking) ===" -ForegroundColor Cyan
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "some tests failed, continuing to training" -ForegroundColor Yellow }

$exports = @(Get-ChildItem exports -File -Include *.txt, *.json -Recurse -ErrorAction SilentlyContinue)
if ($exports.Count -eq 0) {
    Write-Host "no .txt or .json exports in the exports folder" -ForegroundColor Red
    exit 1
}
foreach ($f in $exports) {
    Step "parse $($f.Name)" { & $py -m echolm.cli parse $f.FullName --me $Me }
}
Step "build dataset" { & $py -m echolm.cli format --config configs/default.yaml }
Step "preflight" { & $py -m echolm.cli train check }
Step "smoke test (10 steps)" { & $py -m echolm.cli train sft --max-steps 10 }
Step "full SFT run" { & $py -m echolm.cli train sft }

Write-Host "`nDone. Results are in the newest outputs\sft\sft-r16-* folder (train.log, run_info.json, adapter\)." -ForegroundColor Green
