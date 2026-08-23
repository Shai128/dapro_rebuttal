# DAPRO LPB and metric-estimation ablations

Run the complete 50-split suite from the repository root:

```bash
bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh \
  --slurm --parallel-jobs 20 --cpu
```

Omit `--cpu` to request the configured GPU resource. `--local`, `--dry-run`,
`--available-only`, `--seed-start`, `--seed-end`, and `--experiment-suffix`
are also supported. A failed configuration is recorded and does not cancel the
remaining configurations. Each construction job is merged immediately.

The main LPB studies use Toxicity/Qwen2.5-14B, target coverage 90%, budget 20,
and `N1=50`, except for the pre-existing N1 and budget sweeps and the
attacker-shift studies, which use budget 10:

- hard realized Target-A versus soft model-integrated prefix mass;
- `K=1,2,4,8` quantile-bin maps and a continuous monotone four-knot rank map;
- current hazard, remaining-time quantile, causal target value, random rank,
  and a clearly tagged noncausal oracle remaining-time score;
- Gemma-to-Qwen attacker shifts for Red Team and Toxicity at `B=10`;
- the existing N1, score-noise, and budget sweeps.

Every cell contains Static, raw zero-margin DAPRO, and DAPRO+CRC. The oracle
score is an efficiency upper anchor only (`ablation_score_is_causal=0`) and
must not be presented as deployable.

The score study fixes `K=4` and the controller within each raw/CRC comparison.
“Remaining-time quantile” is the inverse conditional median number of turns
remaining (larger means a predicted earlier event). “Causal target value” is
the current-prefix probability
`P(t < T < q_alpha(X) | T > t, X_it)`. The random anchor independently
permutes the hazard ranks at every time. The oracle anchor uses the latent
target indicator divided by true remaining time and is therefore explicitly
noncausal.

The same launcher additionally runs two event-rate metric studies on the same
Toxicity/Qwen2.5-14B setup, with the recommended `B=20`, `N1=50`, and CRC
control size 25:

- the production K2 hazard score mixed with independently time-permuted score
  values at `lambda = 0, .1, .25, .5, .75, 1`;
- the five named score alternatives at fixed K4 and otherwise identical
  controllers.

For metrics, the DAPRO objective and causal target-value score both use the
event-rate target `A_i = 1{T_i <= 200}`. Thus the target-value score is
`P(t < T <= 200 | T > t, X_it)`, rather than the LPB-specific
`P(t < T < q_alpha(X) | T > t, X_it)`. The oracle remaining-time score uses
the latent realization and is retained only as a nondeployable upper-quality
anchor. Every metric factor cell again contains Static, raw zero-margin DAPRO,
and DAPRO+CRC.

Generate paper figures and machine-readable means/variances with:

```bash
python -m src.predictive_bounds.experiments.dapro_ablation.summarize_paper \
  --quality low
```

This generates both task families by default. Use `--tasks lpb` or
`--tasks metric` to generate only one. Metric outputs are
`dapro_metric_score_noise_ablation.jpg` and
`dapro_metric_score_ablation.jpg`; each reports the event-rate estimate,
absolute error, variance over the random calibration/test splits, budget,
observed events, and mean metric Target-A inverse-propensity weight.

The standard four-panel figure reports coverage, realized DAPRO budget
(assigned/static truncation budget for Static), absolute coverage deviation,
and the selected-target mean inverse-propensity weight. The representation
study additionally writes `dapro_representation_diagnostics.jpg`, containing
the Phase-I fitted objective, Phase-II hard target-weight objective, and
coverage variance across random splits. Exact plotted means, variances,
standard deviations, and sample counts are stored in
`dapro_ablation_mean_variance.csv`.

Ordered numerical sweeps are line plots with mean plus/minus one standard
deviation. The categorical hard-vs-soft and attacker-shift studies are shown
as boxplots over the 50 random splits.
