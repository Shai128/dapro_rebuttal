# DAPRO score/value and variance-decomposition audit

## Scope and reproducibility

This audit is offline and leaves every production allocator unchanged.  It
uses the two priority Qwen setups, horizon 200, calibration/test sizes
3000/3000, budget 20, Phase-I size 50, terminal propensity floor 0.005, and
the no-CRC projection margin 1.0.  Phase-I cost is included in every total
expected and realized budget.

The current policy is soft-prefix Generalized DAPRO with the one-step hazard
score and two empirical quantile bins.  The principal alternative uses four
bins and the causal target-value score

\[
v_{it}(h_i)=\widehat{\Pr}(t<T_i<h_i\mid T_i>t,X_{it}),
\]

where `h_i=200+1` in the metric experiment (equivalently the event occurs by
200) and `h_i` is the frozen raw-alpha LPB candidate in the LPB experiment.
This changes the ordering score, not the generalized soft-prefix objective.
The four-way LPB screen also includes hazard/K4 and geometric interpolation of
the optimized target-value/K4 rank knots.

Metric CRNs follow the production metric convention: a PCG64 table aligned to
the 3000-row outer calibration set.  LPB CRNs follow the production LPB
convention: a population-level PCG64 table is generated first and indexed by
the selected calibration rows.  Data, policy, and acquisition seeds are equal
in outer-split experiments and separated explicitly in fixed-split acquisition
experiments.

The reproducible drivers are:

- `analysis/diagnostics/dapro_binning_audit.py` for metric policies and HT
  acquisition replicates;
- `analysis/diagnostics/lpb_dapro_binning_audit.py` for LPB calibration,
  coverage, fixed-candidate conditional variance, and candidate switching;
- `analysis/diagnostics/summarize_score_value_audit.py` for the final tables.

Representative PowerShell commands are:

```powershell
$seeds50 = (0..49) -join ','
python -m analysis.diagnostics.dapro_binning_audit --setup toxicity_qwen --seeds $seeds50 --n1 50 --budget 20 --bins 2 --scores hazard --objectives soft --projections hard_bin --output outputs\dapro_binning_audit\ht_toxicity_hazard_k2_50.csv
python -m analysis.diagnostics.dapro_binning_audit --setup toxicity_qwen --seeds $seeds50 --n1 50 --budget 20 --bins 4 --scores future_risk --objectives soft --projections smooth_rank --output outputs\dapro_binning_audit\ht_toxicity_future_smooth_k4_50.csv
python -m analysis.diagnostics.lpb_dapro_binning_audit --setup toxicity_qwen --mode outer --seeds $seeds50 --n1 50 --budget 20 --variants hazard_k2,target_value_k4 --output outputs\dapro_binning_audit\lpb_outer50_toxicity_rerun.csv
python -m analysis.diagnostics.lpb_dapro_binning_audit --setup toxicity_qwen --mode acquisition --fixed-data-seed 0 --fixed-policy-seed 0 --seeds $seeds50 --n1 50 --budget 20 --variants hazard_k2,target_value_k4 --output outputs\dapro_binning_audit\lpb_acq50_toxicity_seed0.csv
python -m analysis.diagnostics.summarize_score_value_audit
```

The checked-in raw runs split the LPB outer experiment into seeds 0--9 and
10--49.  The summarizer combines them without rerunning optimization.

## Metric estimation

Across 50 outer splits, the future-risk/smooth-K4 policy lowers the mean exact
conditional HT variance from 7.2653 to 7.0178 percentage-point squared on
toxicity (3.40%) and from 10.3238 to 10.1963 on red-team (1.23%).  Adding the
variance of each split's fully observed target rate gives law-of-total-variance
predictions of 7.6102 versus 7.3627 on toxicity and 10.5738 versus 10.4464 on
red-team.

The single-acquisition-draw variance observed over those same 50 outer splits
moves in the opposite direction: 7.9236 to 9.1744 on toxicity and 7.0958 to
7.5364 on red-team.  This is not evidence that the exact objective failed.
Each point combines a different finite-population split, a different fitted
policy, and one Bernoulli acquisition realization.  With only 50 such draws,
the sampled acquisition residual variance and its accidental covariance with
the split truth are noisy.  In particular, red-team's observed residual
variance is only 6.55 although its exact mean is 10.32.

The fixed-split replication verifies the formula.  Over 500 acquisition seeds
with data and policy seed fixed at zero, hazard/K2 has empirical versus exact
variance 8.5939 versus 8.6290 on toxicity and 11.2122 versus 11.2228 on
red-team.  Future-risk/smooth-K4 gives 8.0638 versus 8.5276 on toxicity and
11.3880 versus 11.2335 on red-team.  The remaining deviations are ordinary
Monte Carlo error; the current-policy ratios are 0.996 and 0.999.  The richer
policy helps toxicity on this split but is slightly worse on red seed zero,
consistent with its small average improvement and policy-fit heterogeneity.

Thus, richer scores/bins produce a real but modest metric-estimation gain.  A
50-split one-draw variance is too noisy to rank policies whose exact risks
differ by only one to three percent; the paired exact conditional variance is
the appropriate primary diagnostic, with repeated acquisition used as a
validation.

## LPB coverage and candidate switching

The four-way ten-split toxicity screen gave coverage variances 0.2277
(hazard/K2), 0.2277 (hazard/K4), 0.2455 (target-value/K4), and 0.2505
(target-value/smooth-K4) percentage-point squared.  Hard target-value/K4 was
retained because it had the best fixed-candidate diagnostics; smooth rank did
not help and was not extended.

Over 50 outer splits, toxicity coverage variance is indistinguishable:
0.491091 for hazard/K2 and 0.491131 for target-value/K4.  Nevertheless, the
mechanism-specific quantities improve sharply:

- selected-candidate exact conditional variance: 0.008258 to 0.001611 (80.5%);
- full-calibration-fixed candidate exact variance: 0.008441 to 0.001591
  (81.2%);
- frozen target-anchor exact variance: 0.25966 to 0.07536 (71.0%);
- switching away from the full-calibration oracle candidate: 60% to 32%;
- mean absolute candidate-index displacement: 1.46 to 0.58.

Across five fixed outer/policy splits and 50 acquisition replicates per split,
the mean within-split coverage variance falls from 0.004717 to 0.002428 (48.5%)
and the mean fixed-candidate HT variance falls from 0.006323 to 0.001437
(77.3%).  The apparent paradox is scale: outer split/test composition creates
about 0.49 percentage-point squared of coverage variance, roughly two orders
of magnitude more than acquisition and candidate switching.  Reducing the
latter cannot visibly change the former with 50 outer splits.

On red-team, all policies select the identical candidate on every one of the
50 outer splits.  The selected fixed-candidate conditional variance is about
`3.4e-13`, and five fixed-split acquisition experiments have zero candidate or
coverage movement to numerical precision.  Its 0.49355 outer coverage
variance is therefore entirely split/test variation; no acquisition score can
reduce it in this regime.

## Budget and deployment qualification

The metric alternatives use approximately the same total expected budget as
hazard/K2.  Across outer splits, toxicity expected cost is 19.385 versus 19.459
and red-team is 19.050 versus 19.098 turns per sample.

The LPB target-value/K4 screen is deliberately no-CRC so it isolates score
information.  Its 50-split mean expected costs are below 20 (19.296 toxicity,
18.992 red), but the projection-transfer assumption can fail on individual
splits.  In the five fixed toxicity splits used for acquisition decomposition,
target-value/K4 averages 20.062 expected turns and exceeds 20 on two splits;
hazard/K2 also exceeds 20 on one split.  Consequently target-value/K4 is an
algorithmic lead, not yet a budget-valid production replacement.  It should be
wrapped in the same independent causal shared-PAV CRC controller before any
claim of budget validity.

## Bottom line

Two-bin hazard DAPRO does discard useful information on toxicity.  Merely
raising the bin count does almost nothing; changing the score to the causal
remaining probability of the named target event is the important step.  That
score greatly improves LPB fixed-candidate risk and switching, and yields a
smaller but consistent average exact-risk improvement for metric estimation.
It does not lower total 50-split LPB coverage variance because that quantity is
dominated by calibration/test resampling rather than acquisition.  The next
production experiment should therefore combine target-value/K4 with the
existing causal CRC budget controller and evaluate both conditional risk and
outer-split coverage, without expecting the latter to move materially in the
current 3000/3000 regime.

Final machine-readable outputs:

- `outputs/dapro_binning_audit/score_value_metric_outer50_summary.csv`
- `outputs/dapro_binning_audit/score_value_metric_acq500_summary.csv`
- `outputs/dapro_binning_audit/score_value_lpb_outer50_summary.csv`
- `outputs/dapro_binning_audit/score_value_lpb_acquisition_summary.csv`

