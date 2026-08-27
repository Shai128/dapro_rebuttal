# DAPRO LPB and metric-estimation ablations

Run the complete 50-split suite from the repository root:

```bash
bash src/predictive_bounds/experiments/dapro_ablation/scripts/run.sh \
  --slurm --parallel-jobs 20 --cpu
```

Omit `--cpu` to request the configured GPU resource. `--local`, `--dry-run`,
`--available-only`, `--seed-start`, `--seed-end`, `--experiment-suffix`, and
`--metric-experiment-suffix` are also supported. A failed configuration is
recorded and does not cancel the remaining configurations. Each construction
job is merged immediately.

The main LPB studies use Toxicity/Qwen2.5-14B, target coverage 90%, budget 20,
and total adaptive-calibration size
`|I_cal1|+|I_crc|=50`, except for the sample-size and budget sweeps and the
attacker-shift studies, which use budget 10:

- three distinct optimization objectives: soft-prefix, hard-terminal, and
  soft-terminal (hard-prefix is omitted because it equals hard-terminal);
- `K=1,2,4,8` quantile-bin maps and a continuous monotone four-knot rank map;
- estimated current event probability, estimated remaining-time quantile,
  random rank, and a clearly tagged noncausal oracle remaining-time score;
- Qwen-to-Gemma attacker shifts for Red Team and Toxicity at `B=10`;
- the existing sample-size, score-noise, and budget sweeps;
- the CRC row-cost cap at `0.1B, 0.5B, B, 2B, 5B, 10B`, capped at the
  executable horizon 200.

Every cell contains Static, raw zero-margin DAPRO, and DAPRO+CRC. The oracle
score is an efficiency upper anchor only (`ablation_score_is_causal=0`) and
must not be presented as deployable.
Because `Cmax` is a CRC candidate-family restriction, raw DAPRO is deliberately
unchanged and repeated as a horizontal no-CRC control across that sweep.

The score study fixes the paper-wide `K=2` representation and the controller
within each raw/CRC comparison.
“Est. remaining-time quantile” is the inverse conditional median number of
turns remaining (larger means a predicted earlier event). The random anchor independently
permutes the hazard ranks at every time. The oracle anchor uses the latent
target indicator divided by true remaining time and is therefore explicitly
noncausal.

The same launcher runs every applicable ablation for event-rate metric
estimation on the same Toxicity/Qwen2.5-14B setup. This includes sample size,
budget, score noise, the three distinct objectives, representation, score,
and Cmax. Attacker shift remains LPB-only. Default cells use the recommended
`B=20`, total adaptive-calibration size 50, and CRC control size 25; sample-
size and budget studies vary their named factor and use a half-size control
fold. Metric sample-size cells are written to one merged directory per size, preventing
parallel Slurm jobs from racing on a shared CSV; the summarizer combines those
directories automatically.

For metrics, the DAPRO objective uses the event-rate target
`A_i = 1{T_i <= 200}`. The oracle remaining-time score uses
the latent realization and is retained only as a nondeployable upper-quality
anchor. Every metric factor cell again contains Static, raw zero-margin DAPRO,
and DAPRO+CRC.

All non-factor DAPRO components use the same soft-prefix target,
current-hazard score, `K=2`, global regularization `0.001`, raw zero-margin
controller, and capped CRC controller unless the named ablation explicitly
changes one of those components. In particular, “Current hazard” in the
named-score study and `lambda=0` in the score-noise study now instantiate the
same allocator.
The policy-shape optimization uses its available per-sample budget directly;
there is no separate `B_ref = gamma * B_shape` reference-budget scale.

Generate the paper figures with:

```bash
python -m src.predictive_bounds.experiments.dapro_ablation.summarize_paper \
  --quality high
```

This generates both task families by default. Use `--tasks lpb` or
`--tasks metric` to generate only one. Metric outputs are
named `dapro_metric_<kind>_ablation.jpg`; each reports the event-rate estimate,
absolute error, variance over the random calibration/test splits, budget,
observed events, and mean weighted error.

The standard four-panel figure reports coverage, realized DAPRO budget
(assigned/static truncation budget for Static), absolute coverage deviation,
and the selected-target mean weighted error. The sample-
size, score-noise, budget, optimization-objective, representation, and
optimization-process figures additionally report coverage variance and the
number of observed events. The representation
study additionally writes LPB and metric representation-diagnostic figures
containing the Phase-I fitted objective, Phase-II target-weight objective, and
the relevant estimator variance across random splits. The summarizer writes
only figures below `figures/paper/ablations`.

Ordered numerical sweeps are line plots with mean plus/minus one standard
deviation. The categorical coefficient/support and attacker-shift studies are shown
as boxplots over the 50 random splits.
