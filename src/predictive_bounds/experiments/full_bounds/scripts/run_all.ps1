# Construct, merge, plot, and tabulate the complete manuscript matrix.
param(
    [string]$PythonExe = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "cuda:0" }),
    [int]$SeedStart = $(if ($env:SEED_START) { $env:SEED_START } else { 0 }),
    [int]$SeedEnd = $(if ($env:SEED_END) { $env:SEED_END } else { 50 }),
    [string]$Suffix = $(if ($env:EXPERIMENT_SUFFIX) { $env:EXPERIMENT_SUFFIX } else { "full_bounds_v1" }),
    [ValidateSet("high", "low")]
    [string]$Quality = $(if ($env:FIGURE_QUALITY) { $env:FIGURE_QUALITY } else { "high" }),
    [switch]$AvailableOnly,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Arguments = @(
    "-m", "src.predictive_bounds.experiments.full_bounds.run_all",
    "--seed-start", "$SeedStart",
    "--seed-end", "$SeedEnd",
    "--device", $Device,
    "--suffix", $Suffix,
    "--quality", $Quality
)
if ($AvailableOnly) {
    $Arguments += "--available-only"
}
$Arguments += $ExtraArgs
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
