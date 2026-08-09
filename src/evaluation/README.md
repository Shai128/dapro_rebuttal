# Fixed-benchmark metric estimation

This experiment estimates population metrics on a fixed cached benchmark under
limited interaction budgets. For the unsafe-event-rate objective, the target is

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

The launcher runs every requested seed and performs a strict merge. Figure
generation is disabled on the server by default; pass `--figures` to enable
it. Temporary seed files are stored under
`results/tmp_metric_calibration_results`; merged CSVs are stored under
`results/merged_metric_calibration_dfs`; figures and across-seed summaries are
stored under `figures/metric_estimation`.

The merged file has one row per method and seed.  It includes realized and
expected budget diagnostics, the estimated UER and RMTTU and their absolute
errors, inverse-probability weight diagnostics, metric-event-weighted weights,
observed-event and resolved-trajectory counts, and effective sample size.

Shared baselines and oracles are stored once per seed. N1-dependent methods
are stored separately, with explicit `configured_dapro_n1` and
`configured_crc_control_size` columns. The plotting code expands a shared
baseline only when making a like-for-like N1 panel; the merged CSV itself has
no duplicate baseline observations.

## Plot existing merged results

```bash
python -m src.evaluation.summarize
python -m src.evaluation.summarize --experiment NAME
python -m src.evaluation.summarize \
  --experiment-suffix generalized_dapro_metric_v1 --quality high
```

The figures include only the production metric methods. Variance bars are the
ordinary sample variance of the metric estimates across the 50 random
calibration/test splits. The fixed calibration+test oracle is shown as a red
horizontal dashed reference on metric-value figures.
