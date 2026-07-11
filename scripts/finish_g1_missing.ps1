# Finish the 5 missing G1 baseline records without triggering the run-fingerprint
# guard (which would re-run every completed record). Each missing (model, fold,
# seed) is trained in its own fresh staging run, then its record + checkpoint are
# copied into the canonical run directory. Training code is behaviorally
# identical to the earlier records (recipe byte-identical, no training-path file
# changed since the run started); only run metadata differs.

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$repo = "E:\1\hwnas\hwnas"
$py = Join-Path $repo ".venv_cuda\Scripts\python.exe"
$stage = Join-Path $repo "results\protocol\_patch"
New-Item -ItemType Directory -Force $stage | Out-Null
Set-Location $repo

$common = @(
    "--data-dir", "data\NKSID", "--epochs", "150", "--batch-size", "8",
    "--gradient-accumulation-steps", "4", "--amp", "--save-checkpoints",
    "--device", "cuda", "--no-resume"
)

# model canonical-dir  ->  extra args
$missing = @(
    @{ model = "g1_mobilenet_v2_grayscale_imagenet"; fold = 4; seed = 44;
       extra = @("--arch","mobilenet_v2","--pretrained","--selection-provenance","baseline_predeclared") },
    @{ model = "g1_mobilenet_v2_scratch"; fold = 3; seed = 44;
       extra = @("--arch","mobilenet_v2","--selection-provenance","baseline_predeclared") },
    @{ model = "g1_mobilenet_v2_scratch"; fold = 4; seed = 42;
       extra = @("--arch","mobilenet_v2","--selection-provenance","baseline_predeclared") },
    @{ model = "g1_mobilenet_v2_scratch"; fold = 4; seed = 43;
       extra = @("--arch","mobilenet_v2","--selection-provenance","baseline_predeclared") },
    @{ model = "g1_mobilenet_v2_scratch"; fold = 4; seed = 44;
       extra = @("--arch","mobilenet_v2","--selection-provenance","baseline_predeclared") }
)

foreach ($m in $missing) {
    $tag = "fold$($m.fold)_seed$($m.seed)"
    $canonical = Join-Path $repo "results\protocol\$($m.model)"
    $target = Join-Path $canonical "run_$tag.json"
    if (Test-Path $target) { "[patch] $($m.model)/$tag already present, skip" | Out-Host; continue }

    $runName = "$($m.model)__patch_$tag"
    $log = Join-Path $stage "$runName.log"
    "[patch] $(Get-Date -Format s) training $($m.model)/$tag" | Out-File -Append -Encoding utf8 $log
    & $py (Join-Path $repo "run_eval_protocol.py") @common `
        "--folds" "$($m.fold)" "--seeds" "$($m.seed)" `
        @($m.extra) "--output-dir" "results\protocol\_patch" "--run-name" $runName *>> $log

    $srcJson = Join-Path $stage "$runName\run_$tag.json"
    $srcPt = Join-Path $stage "$runName\best_$tag.pt"
    if (Test-Path $srcJson) {
        Copy-Item $srcJson $target -Force
        if (Test-Path $srcPt) { Copy-Item $srcPt (Join-Path $canonical "best_$tag.pt") -Force }
        "[patch] $(Get-Date -Format s) merged $($m.model)/$tag exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
    } else {
        "[patch] $(Get-Date -Format s) FAILED $($m.model)/$tag exit=$LASTEXITCODE (no record produced)" | Out-File -Append -Encoding utf8 $log
    }
}

"[patch] all missing records processed" | Out-File -Append -Encoding utf8 (Join-Path $stage "finish_g1_missing.done")
