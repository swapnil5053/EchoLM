# EchoLM on Windows, from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Me "your name in the export"
# or, on the public Ubuntu IRC logs instead of your own chats (no -Me needed):
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Dataset ubuntu
# Stages run in order: setup, data, sft, grpo, eval, card. Pick a subset with -Stages, e.g.
#   -Stages grpo,eval,card
# -Stages seeds trains GRPO again with several seeds (-Seeds 1,2,3) and adds them to the report.
# Add -SkipInstall once packages are in place, -RetrainSft to train SFT again even if a run exists.

param(
    [ValidateSet("chat", "ubuntu")][string]$Dataset = "chat",
    [string]$Me = "",
    [string]$IrcUser = "",
    [string]$Channel = "#ubuntu",
    [int]$Since = 2016,
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Stages = @("setup", "data", "sft", "grpo", "eval", "card"),
    [switch]$SkipInstall,
    [switch]$RetrainSft
)

# native tools write progress to stderr; "Stop" would turn that into a fatal error in PowerShell 5.1,
# so every step checks its exit code instead
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)
$Stages = @($Stages | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToLower() })
$started = Get-Date

# the two datasets never share files: your chats stay under data\ and outputs\,
# the public run lives under data\ubuntu\ and outputs\ubuntu\ and is the one that fills README.md
if ($Dataset -eq "ubuntu") {
    $parsed = "data/ubuntu/parsed"; $data = "data/ubuntu/processed"; $out = "outputs/ubuntu"
    $dataCfg = "configs/irc.yaml"; $Me = "Alex"
} else {
    $parsed = "data/parsed"; $data = "data/processed"; $out = "outputs"; $dataCfg = "configs/default.yaml"
}
$evalDir = "$out/eval"

function Step([string]$Name, [scriptblock]$Cmd) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

function Newest([string]$Dir, [string]$Filter, [string]$Inside) {
    Get-ChildItem $Dir -Directory -Filter $Filter -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName $Inside) } |
        Sort-Object LastWriteTime | Select-Object -Last 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Step "create venv (python 3.12)" { py -3.12 -m venv .venv }
}
$py = (Resolve-Path ".venv\Scripts\python.exe").Path

if ($Stages -contains "setup" -and -not $SkipInstall) {
    Step "upgrade pip" { & $py -m pip install --upgrade pip }
    Step "install CUDA torch" {
        & $py -m pip install "torch>=2.8,<2.13" torchvision --index-url https://download.pytorch.org/whl/cu128
    }
    Step "install echolm" { & $py -m pip install -e ".[dev,train,eval,demo,irc]" }
    $cuda = & $py -c "import torch; print(torch.cuda.is_available())"
    if ($cuda -ne "True") {
        # unsloth's dependencies can pull a CPU-only torch from PyPI; put the CUDA build back
        $ver = & $py -c "import torch; print(torch.__version__.split('+')[0])"
        Step "restore CUDA torch $ver" {
            & $py -m pip install --force-reinstall --no-deps "torch==$ver" --index-url https://download.pytorch.org/whl/cu128
        }
    }
    Write-Host "`n=== unit tests (not blocking) ===" -ForegroundColor Cyan
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { Write-Host "some tests failed, continuing" -ForegroundColor Yellow }
}

if ($Stages -contains "data" -and $Dataset -eq "ubuntu") {
    $logs = "data/irc/logs.jsonl"
    if (-not (Test-Path $logs)) {
        Step "download $Channel logs since $Since (streams the 6 GB dataset once)" {
            & $py -m echolm.cli irc fetch --channel $Channel --since $Since --out $logs
        }
    }
    Step "most active nicks" { & $py -m echolm.cli irc users --logs $logs }
    $pick = @()
    if ($IrcUser -ne "") { $pick = @("--user", $IrcUser) }
    Step "rebuild 1:1 threads" {
        & $py -m echolm.cli irc export --logs $logs --out data/ubuntu/ubuntu_irc.json --alias $Me @pick
    }
    Step "parse" { & $py -m echolm.cli parse data/ubuntu/ubuntu_irc.json --me $Me --out $parsed }
    Step "build dataset" { & $py -m echolm.cli format --parsed $parsed --out $data --config $dataCfg }
    # how well the thread rebuilding matches human reply labels; informative, never blocks the run
    $gold = "data/irc/irc-disentanglement"
    if (-not (Test-Path "$gold/data/test")) {
        # only the 3 MB test split, not the whole repository
        git clone --depth 1 -q --filter=blob:none --no-checkout https://github.com/jkkummerfeld/irc-disentanglement $gold
        git -C $gold sparse-checkout set --no-cone "/data/test/"
        git -C $gold checkout -q
    }
    if (Test-Path "$gold/data/test") {
        Write-Host "`n=== thread rebuilding vs. human labels (irc-disentanglement test split) ===" -ForegroundColor Cyan
        & $py -m echolm.cli irc validate --data "$gold/data/test" --skip 2008-07-14_18 --skip 2010-08-17_18
    }
}

if ($Stages -contains "data" -and $Dataset -eq "chat") {
    if ($Me -eq "") { Write-Host "-Me is required for the data stage" -ForegroundColor Red; exit 1 }
    $exports = @(Get-ChildItem exports -File -Include *.txt, *.json -Recurse -ErrorAction SilentlyContinue)
    if ($exports.Count -eq 0) { Write-Host "no .txt or .json exports in exports\" -ForegroundColor Red; exit 1 }
    foreach ($f in $exports) {
        Step "parse $($f.Name)" { & $py -m echolm.cli parse $f.FullName --me $Me --out $parsed }
    }
    Step "build dataset" { & $py -m echolm.cli format --parsed $parsed --out $data --config $dataCfg }
}

if ($Stages -contains "sft") {
    if ($RetrainSft -or -not (Newest "$out/sft" "sft-r*" "run_info.json")) {
        Step "preflight" { & $py -m echolm.cli train check --data $data }
        Write-Host "`n=== SFT smoke test (10 steps) ===" -ForegroundColor Cyan
        & $py -m echolm.cli train sft --data $data --out "$out/sft" --max-steps 10
        if ($LASTEXITCODE -ne 0) {
            # the usual Windows failure is Unsloth's triton compilation; uncompiled is slower but works
            Write-Host "retrying with UNSLOTH_COMPILE_DISABLE=1" -ForegroundColor Yellow
            $env:UNSLOTH_COMPILE_DISABLE = "1"
            Step "SFT smoke test, no compile" {
                & $py -m echolm.cli train sft --data $data --out "$out/sft" --max-steps 10
            }
        }
        Step "SFT" { & $py -m echolm.cli train sft --data $data --out "$out/sft" }
    } else {
        Write-Host "`n=== SFT: reusing the newest finished run (add -RetrainSft to train again) ===" -ForegroundColor Cyan
    }
}

$sftCkpt = (& $py -m echolm.cli train select --root "$out/sft" | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0 -and ($Stages -contains "grpo" -or $Stages -contains "eval" -or $Stages -contains "seeds")) {
    Write-Host "no SFT run to build on; run the sft stage first" -ForegroundColor Red
    exit 1
}
if ($sftCkpt) { Write-Host "SFT checkpoint: $sftCkpt" }

if ($Stages -contains "grpo") {
    $backend = "unsloth"
    Write-Host "`n=== GRPO smoke test (3 steps) ===" -ForegroundColor Cyan
    & $py -m echolm.cli train grpo --data $data --out "$out/grpo" --init $sftCkpt --max-steps 3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "unsloth backend failed, retrying with plain transformers + peft" -ForegroundColor Yellow
        $backend = "hf"
        Step "GRPO smoke test, hf backend" {
            & $py -m echolm.cli train grpo --data $data --out "$out/grpo" --init $sftCkpt --max-steps 3 --backend hf
        }
    }
    Step "GRPO ($backend backend)" {
        & $py -m echolm.cli train grpo --data $data --out "$out/grpo" --init $sftCkpt --backend $backend
    }
}

if ($Stages -contains "eval") {
    # the base model never changes, so its samples are kept and only re-scored
    $ev = @("--data", $data, "--out", $evalDir)
    if (Test-Path "$evalDir/base/generations.jsonl") {
        Step "score base model" { & $py -m echolm.cli eval run --model base --name base --score-only @ev }
    } else {
        Step "evaluate base model" { & $py -m echolm.cli eval run --model base --name base @ev }
    }
    Step "evaluate SFT" { & $py -m echolm.cli eval run --model $sftCkpt --name sft @ev }
    $grpo = Newest "$out/grpo" "grpo-run-*" "adapter"
    if ($grpo) {
        $adapter = Join-Path $grpo.FullName "adapter"
        Step "evaluate GRPO" { & $py -m echolm.cli eval run --model $adapter --name grpo @ev }
    }
    # older comparison rows (e.g. other SFT checkpoints) are re-scored with the current metrics
    Get-ChildItem $evalDir -Directory -ErrorAction SilentlyContinue | Where-Object {
        @("base", "sft", "grpo") -notcontains $_.Name -and (Test-Path (Join-Path $_.FullName "generations.jsonl"))
    } | ForEach-Object {
        # not $name: PowerShell variables ignore case and Step's own $Name would shadow it
        $row = $_.Name
        Step "re-score $row" { & $py -m echolm.cli eval run --model $row --name $row --score-only @ev }
    }
    # only the public dataset's numbers go into README.md; your own chat's report stays in outputs\
    $readme = @()
    if ($Dataset -eq "ubuntu") { $readme = @("--readme", "README.md") }
    Step "report" { & $py -m echolm.cli eval report --out $evalDir @readme }
    # figures of the public run go into the repo; your own chat's stay next to its outputs
    $figs = "$out/figures"
    if ($Dataset -eq "ubuntu") { $figs = "docs/figures" }
    Write-Host "`n=== training curves (not blocking) ===" -ForegroundColor Cyan
    & $py -m echolm.cli eval plot --outputs $out --out $figs
    if ($LASTEXITCODE -ne 0) { Write-Host "could not draw the curves, continuing" -ForegroundColor Yellow }
}

if ($Stages -contains "seeds") {
    # one GRPO run per seed from the same SFT checkpoint; finished seeds are skipped on a rerun
    $ev = @("--data", $data, "--out", $evalDir)
    foreach ($seed in $Seeds) {
        $root = "$out/grpo-seeds/s$seed"
        if (-not (Newest $root "grpo-run-*" "adapter")) {
            Step "GRPO seed $seed" {
                & $py -m echolm.cli train grpo --data $data --out $root --init $sftCkpt --seed $seed
            }
        }
        if (-not (Test-Path "$evalDir/grpo-s$seed/metrics.json")) {
            $adapter = Join-Path (Newest $root "grpo-run-*" "adapter").FullName "adapter"
            Step "evaluate GRPO seed $seed" { & $py -m echolm.cli eval run --model $adapter --name "grpo-s$seed" @ev }
        }
    }
    $readme = @()
    if ($Dataset -eq "ubuntu") { $readme = @("--readme", "README.md") }
    Step "report" { & $py -m echolm.cli eval report --out $evalDir @readme }
}

if ($Stages -contains "card") {
    if (Newest "$out/grpo" "grpo-run-*" "adapter") {
        $kind = @()
        if ($Dataset -eq "ubuntu") { $kind = @("--irc") }
        Step "model card" { & $py -m echolm.cli card --root "$out/grpo" --eval $evalDir --data $data @kind }
    }
}

$mins = [math]::Round(((Get-Date) - $started).TotalMinutes)
Write-Host "`nDone in $mins min. Report: $evalDir/report.md" -ForegroundColor Green
Write-Host "Chat with the models: .venv\Scripts\echolm demo --outputs $out --data $data" -ForegroundColor Green
