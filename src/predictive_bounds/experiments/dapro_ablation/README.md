# LPB DAPRO ablations

This workflow evaluates Static, soft-prefix Generalized DAPRO without CRC,
and the same DAPRO policy with CRC on the Toxicity/Qwen-to-Qwen setup.  Every
cell uses 50 paired random calibration/test splits and 90% target coverage.

Run on a Linux/Ubuntu Slurm server:

```bash
bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh \
  --slurm --parallel-jobs 8 --cpu
```

Omit `--cpu` to request the configured GPU resource.  Use `--dry-run` to print
all commands, or edit the documented configuration block at the top of
`scripts/run.sh`.

The launcher runs:

- `N1`: 50, 100, 200, 300, 400 at budget 20;
- score noise: lambda = 0, .1, .25, .5, .75, 1 at N1=100 and budget 20;
- budget: 5, 10, 20, 30, 40, 50 with N1 = 25, 50, 100, 150, 200, 250.

The score corruption is

```text
S_lambda = (1-lambda) S + lambda S_permuted,
```

where `S_permuted` is an independently permuted copy of the same time
column.  It therefore preserves the time-specific marginal scale and isolates
loss of row-level ranking information.  The permutation seed is fixed, so all
lambda values are paired.

After downloading the merged result directories, generate paper figures with:

```bash
python -m src.predictive_bounds.experiments.dapro_ablation.summarize_paper \
  --quality high --strict-seeds
```

Figures and their seed-level plotting data are written to
`figures/paper/ablations`.  In the budget panel, Static is charged its assigned
`sum(C_i)/n`, while DAPRO is charged actual event-stopped interactions.
