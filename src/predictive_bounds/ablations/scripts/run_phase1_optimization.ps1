param(
    [string]$DatasetName = "dataset_toxicity",
    [string]$DatasetSetup = "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify",
    [double]$BudgetPerSample = 20,
    [int]$CalSize = 3000,
    [int]$SeedStart = 0,
    [int]$SeedEnd = 10,
    [int]$N1 = 200,
    [double]$TauPrior = 0.56,
    [string]$Projection = "platt",
    [string]$Score = "prob",
    [string]$Device = "cuda:0",
    [string]$OutputDir = "results/phase1_optimization_ablation",
    [switch]$DryRunFixture
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
Push-Location $repoRoot
try {
    $arguments = @(
        "-m", "src.predictive_bounds.ablations.phase1_optimization",
        "--dataset-name", $DatasetName,
        "--dataset-setup", $DatasetSetup,
        "--budget-per-sample", $BudgetPerSample,
        "--cal-size", $CalSize,
        "--seed-start", $SeedStart,
        "--seed-end", $SeedEnd,
        "--n1", $N1,
        "--tau-prior", $TauPrior,
        "--projection", $Projection,
        "--score", $Score,
        "--device", $Device,
        "--output-dir", $OutputDir
    )
    if ($DryRunFixture) {
        $arguments += "--dry-run-fixture"
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Phase-I optimization ablation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
