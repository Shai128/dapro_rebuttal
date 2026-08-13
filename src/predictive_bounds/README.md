# Predictive bounds package

This package contains the production LPB/UPB pipeline and the supporting
allocation, calibration, and survival utilities.

The two primary command-line entry points intentionally remain at package root:

```bash
python -m src.predictive_bounds.construct_calibrated_bound --help
python -m src.predictive_bounds.merge_bounds_results --help
```

Supporting code is grouped by responsibility:

- `budget_allocators/`: DAPRO, constant-probability, locally adaptive, and
  comparison allocation policies.
- `calibration/`: calibrated LPB/UPB estimators.
- `survival_utils/`: conditional distributions, quantiles, and coverage tools.
- `experiments/`: additional analyses and non-primary experiment entry points.
- `ablations/`: focused ablations with Linux and Windows runner scripts.

## Definitive public DAPRO

`budget_allocators.DAPRO.DAPRO` is the final variance-aligned allocator. It
minimizes a regularized empirical target-event variance objective, projects the
policy through two score bins, and reserves one expected interaction per
Phase-II row for projection transfer error. Its terminal propensity floor is
included in the budget correction, so the floor cannot silently add expected
cost after optimization. The former mean-inverse-probability objective remains
available only as `LegacyMeanWeightDAPRO` for ablation compatibility.

The frozen configuration, five-dataset runner, and reporting script are in
`experiments/definitive_dapro/`.

## Generalized DAPRO with soft prefix masses

`SoftTargetDAPRO` and `SoftTargetCRCDAPRO` use the same causal two-bin DAPRO
policy class, but replace the hard Phase-I endpoint target by model-integrated
event mass at every observed prefix. With no metric horizon configured, they
target the raw-alpha LPB miscoverage event. The first method uses the
projection-margin budget controller; the second uses an independent CRC fold
of size `N1 // 2` and a causal shared-PAV row cap of
`2 * budget_per_sample`.  The cap is evaluated prefix by prefix, so an early
continuation decision never depends on a future interaction.

The production LPB matrix, including both soft-prefix variants for
`N1={200,100,50}` and budgets `{5,10,20}`, can be constructed and merged on a
server with:

```bash
bash src/predictive_bounds/scripts/calibrate.sh --slurm
```

After downloading or extracting the merged `all_df.csv` files locally,
generate the complete budget/N1 figure tree with:

```bash
python -m src.predictive_bounds.experiments.full_bounds.summarize \
  --input-dir results/merged_calibration_dfs --quality low
```

## Timing contract

Tensor positions are zero-based, while event times and allocation horizons are
one-based interaction counts. At the start of turn `t`, `x[i, t]` is visible.
At the end of that turn, `y[i, t]` is revealed; after a negative label,
`x[i, t + 1]` becomes visible. A first event in `y[i, t]` has
`t_tilde[i] = t + 1`; no event in a sequence of length `T` has
`t_tilde[i] = T + 1`.

For real data, `x[i, t]` is the concatenation of the current prompt embedding
and the previous response embedding. The previous-response block is zero for
the first turn. This is causal: the response whose label is `y[i, t]` is never
included in `x[i, t]`.
