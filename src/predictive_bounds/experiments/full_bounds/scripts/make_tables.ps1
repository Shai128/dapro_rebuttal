# Regenerate the single copy-paste-ready LaTeX table file from merged results.
param(
    [string]$PythonExe = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [string]$Suffix = $(if ($env:EXPERIMENT_SUFFIX) { $env:EXPERIMENT_SUFFIX } else { "full_bounds_v4_soft_upb" }),
    [switch]$AvailableOnly,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$Arguments = @(
    "-m", "src.predictive_bounds.experiments.full_bounds.make_tables",
    "--suffix", $Suffix
)
if ($AvailableOnly) {
    $Arguments += "--available-only"
}
$Arguments += $ExtraArgs
& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
