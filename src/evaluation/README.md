# Fixed-benchmark metric estimation

This experiment estimates population metrics on a fixed cached benchmark under
limited interaction budgets.  For the unsafe-event-rate objective, Target-A is

```text
A_i = 1{T_i <= M},
```

where `M` is the benchmark horizon.  This is intentionally different from the
LPB construction event anchored at a candidate model quantile.

## Run the matrix

Edit the configuration block in `scripts/run.sh`, then run one of:

```bash
bash src/evaluation/scripts/run.sh --local
bash src/evaluation/scripts/run.sh --local --cpu --available-only
bash src/evaluation/scripts/run.sh --slurm --parallel-jobs 50
bash src/evaluation/scripts/run.sh --slurm --dry-run
```

The launcher runs every requested seed, performs a strict merge, and generates
figures.  Temporary seed files are stored under
`results/tmp_metric_calibration_results`; merged CSVs are stored under
`results/merged_metric_calibration_dfs`; figures and across-seed summaries are
stored under `figures/metric_estimation`.

The merged file has one row per method and seed.  It includes realized and
expected budget diagnostics, the estimated UER and RMTTU and their absolute
errors, inverse-probability weight diagnostics, metric-event-weighted weights,
observed-event and resolved-trajectory counts, effective sample size, and a
conditional estimator-variance proxy.

## Plot existing merged results

```bash
python -m src.evaluation.summarize
python -m src.evaluation.summarize --experiment NAME
```

Every boxplot excludes the full-budget oracle method and shows its value as a
red horizontal dashed line.
