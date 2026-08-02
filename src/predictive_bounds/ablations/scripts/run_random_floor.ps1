param(
    [string]$DatasetName = "dataset_toxicity",
    [string]$DatasetSetup = "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify",
    [int]$CalSize = 3000,
    [int]$SeedStart = 0,
    [int]$SeedEnd = 10,
    [double]$BudgetPerSample = 20,
    [double]$TauPrior = 0.56,
    [double]$MUpperBound = 200,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
Push-Location $repoRoot
try {
    & python -m src.predictive_bounds.ablations.random_floor `
        --dataset-name $DatasetName `
        --dataset-setup $DatasetSetup `
        --cal-size $CalSize `
        --seed-start $SeedStart `
        --seed-end $SeedEnd `
        --budget-per-sample $BudgetPerSample `
        --tau-prior $TauPrior `
        --m-upper-bound $MUpperBound `
        --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Random-floor ablation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
