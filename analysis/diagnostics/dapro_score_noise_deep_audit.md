# DAPRO score-noise and score-sufficiency audit

## Bottom line

The score-noise experiment is working: at `lambda=1` the corrupted score has
mean timewise correlation `-0.00019` with the original score, and only
`49.993%` of its K2 median-bin assignments agree with the original.  The flat
coverage curves therefore do **not** indicate an implementation failure.

They also do not show that scores are useless.  They show that final LPB
coverage and a post-calibration selected-target objective are poor isolated
diagnostics of the score's efficiency in this regime.  Calibration pins the
mean near 90%, split/test composition dominates the remaining variance, and
the selected target changes with the noisy calibration curve.  The current
instantaneous-hazard score is informative, but it is not the most
target-aligned LPB score, and K2 often prevents the optimizer from using the
information it contains.

The recommended next DAPRO variant keeps the soft-prefix target objective and
CRC, but replaces the instantaneous hazard by a causal remaining-target-value
or target-value-per-remaining-cost score and replaces forced median K2 by K4
or an optimized distinct-value split.  It should be evaluated first under the
same N1, CRC, common random numbers, and budget controller before replacing
the paper method.

## What the score-noise ablation actually shows

The production ablation uses

\[
S_{it}^{(\lambda)}=(1-\lambda)S_{it}+\lambda S_{\pi_t(i),t},
\]

where each time column is independently permuted.  This preserves the
time-specific marginal scale.  Fifty paired outer splits give:

| Method | Quantity | lambda=0 | lambda=1 | Change |
|---|---:|---:|---:|---:|
| DAPRO + CRC | coverage variance (squared pp) | 0.6164 | 0.6167 | +0.0004 |
| DAPRO raw | coverage variance (squared pp) | 0.5080 | 0.4844 | -0.0236 |
| DAPRO + CRC | fixed target-anchor conditional variance (squared pp) | 0.5006 | 0.5229 | +4.45% |
| DAPRO raw | fixed target-anchor conditional variance (squared pp) | 0.3932 | 0.4360 | +10.87% |
| DAPRO + CRC | fixed-anchor mean objective weight | 0.3133 | 0.3202 | +2.21% |
| DAPRO raw | fixed-anchor mean objective weight | 0.2800 | 0.2933 | +4.74% |
| DAPRO + CRC | selected-candidate conditional variance (squared pp) | 0.0601 | 0.0480 | -20.3% |
| DAPRO raw | selected-candidate conditional variance (squared pp) | 0.01135 | 0.00670 | -41.0% |

The fixed-anchor result is the clean score diagnostic: randomizing the score
worsens the objective DAPRO was fitted to.  The paired normal 95% intervals
for the lambda-one minus lambda-zero change are `[0.0046,0.0400]` squared pp
with CRC and `[0.0236,0.0619]` without CRC.  Thus the original score provides
a real, though modest, improvement for its stated target.

The selected-candidate result moves in the opposite direction because the
calibrated candidate is a noisy nonlinear output, not the fixed target used by
the allocator.  A random, more time-only policy can accidentally favor the
event set selected after observing the HT calibration curve.  This is target
mismatch/selection behavior, not evidence that random scores dominate useful
scores.  The selected mean bound changes between lambda zero and one on 74% of
CRC splits (mean absolute size change `0.177`) and 50% of raw splits (`0.073`),
yet neighboring candidates have very similar test coverage.  This is why the
selected conditional component can move while total coverage remains flat.

## Why the four headline curves are almost flat

1. **Coverage mean is constrained by calibration.** Every method chooses a
   candidate intended to achieve 90% coverage.  Allocation quality should not
   systematically move the mean away from 90%; it should reduce estimation
   noise and candidate switching.
2. **The plotted target weight uses a moving target.** The stored quantity is
   \(n^{-1}\sum_i A_i(q_{\hat\tau})/\pi_i\), computed from latent outcomes;
   it is not the realized HT estimator
   \(n^{-1}\sum_i A_iR_i/\pi_i\).  For a *fixed* event set,
   \[
   \frac1n\sum_i\frac{A_i}{\pi_i}
   =\bar A+\frac1n\sum_iA_i(1/\pi_i-1),
   \]
   so lower is genuinely better and the second term is proportional to exact
   conditional variance.  In this plot, however, `q_hat` changes with the
   acquisition/calibration result: lambda zero versus one changes the selected
   mean bound on 74% of CRC splits and 50% of raw splits.  It therefore mixes
   score efficiency with target selection.  The frozen raw-alpha target
   objective is the clean comparison.
3. **Outer-split variance dominates.** In the earlier paired 50-split LPB
   audit, total coverage variance is about `0.491` squared pp, whereas the
   selected-candidate acquisition component is only `0.0083` squared pp for
   hazard/K2.  Even a large relative reduction of the latter barely changes
   the former.
4. **K2 frequently collapses to a time-only response.** In the reproduced
   toxicity seed-zero policy, the low/high bins receive the same raw
   continuation probability at 178 of 200 turns.  Across nearby splits the
   mean number of distinct raw K2 probabilities is only 1.115 per turn.  If
   the two bin probabilities are pooled, changing bin membership has no
   effect.
5. **History still creates many terminal propensities.** K2 does not imply
   only two final inclusion probabilities: earlier bin paths, cumulative
   products, the shared envelope, and CRC contraction create many cumulative
   reaches.  Randomizing the current score leaves this time/history structure
   intact.
6. **CRC can attenuate score differences.** The shared causal envelope and
   selected affine contraction are necessary for the budget guarantee but can
   clip differences between score-conditioned paths.  Lambda one selects a
   slightly less contracted mean mixture (`0.886` versus `0.862`), partially
   offsetting its less informative grouping.

## Is the instantaneous hazard the right score?

It is a conventional and causal score, but it is not fully aligned with the
LPB target.  At time `t`, the current score is

\[
h_{it}=\widehat P(T_i=t\mid T_i\ge t,X_{it}),
\]

whereas the target is the row-specific cumulative event

\[
A_i=\mathbf 1\{T_i<q_{\alpha}(X_i)\}.
\]

A higher immediate hazard need not imply a higher chance of the named event
before that row's remaining target horizon.  The causal target-value score is

\[
r_{it}=\widehat P(t<T_i<q_{\alpha}(X_i)
                  \mid T_i>t,X_{it}),
\]

and a block-Neyman refinement ranks by
\(\sqrt{r_{it}/c_{it}}\), with `c_it` the predicted remaining acquisition
cost.

On the full aligned toxicity/Qwen cache, among rows whose target horizon has
not passed, hazard AUC versus the named LPB event is `0.629/0.706/0.710` at
turns 10/20/50.  Target-value AUC is `0.782/0.758/0.860`.  Thus the alternative
contains materially better late-prefix target information.

The comparison reverses early: at turns 1/2/5, hazard AUC is
`0.729/0.594/0.588`, while target-value gives `0.382/0.505/0.568`.  This is
understandable because `q_alpha(X_i)` is itself a quantile of the same initial
model, so its predicted probability of falling below its own quantile is close
to constant; discretization/model error can even reverse the residual ranking.
Consequently the most defensible replacement is not blindly “target value at
all times.”  It is a downstream benefit-per-cost/value score, or a small
causal basis that can use immediate hazard early and updated remaining target
risk later.

There is also a binning pathology.  Among all active rows, target-value is
zero for 81.5% at turn 20, 93.8% at turn 50, and 96.7% at turn 100 because the
row's target horizon has already passed.  A median split then puts the median
at zero and yields an all-active K2 AUC of exactly 0.5, despite raw AUCs of
`0.957/0.992/0.990`.  The correct fix is not to abandon the score, but to use
a zero-versus-positive/optimized distinct-value cutpoint, K4 with minimum cell
support, or a small continuous monotone map.

## Alternative-score validation

Existing paired 50-split experiments compare the current hazard/K2 policy to
hard target-value/K4 at nearly the same expected budget:

| Toxicity LPB diagnostic | Hazard/K2 | Target-value/K4 | Change |
|---|---:|---:|---:|
| total outer coverage variance (squared pp) | 0.491091 | 0.491131 | unchanged |
| selected-candidate conditional variance | 0.008258 | 0.001611 | -80.5% |
| fixed target-anchor variance | 0.25966 | 0.07536 | -71.0% |
| switch from full-calibration oracle | 60% | 32% | -28 pp |
| mean candidate-index displacement | 1.46 | 0.58 | -60.3% |

Across five fixed outer splits and 50 acquisition replicates per split,
within-split coverage variance falls from `0.004717` to `0.002428` squared pp
(-48.5%).  Total outer variance remains unchanged because it is roughly two
orders of magnitude larger.

The same score/representation change also improves exact metric-estimation
variance, but more modestly: toxicity `7.2653 -> 7.0178` squared pp (-3.40%)
and red-team `10.3238 -> 10.1963` (-1.23%).  A separate time-only comparison
on toxicity gives `8.0544` versus `7.1469` for hazard/K2, showing that the
current hazard score can help substantially relative to no score in a target
where its ranking is better aligned.

## Recommendation

Do not remove score adaptation and do not judge it by mean coverage or mean
HT target weight.  For the LPB paper method, the strongest next candidate is:

1. keep the soft-prefix Generalized-DAPRO objective;
2. use causal target value per remaining cost, or a small hazard-plus-updated-
   target-value basis that preserves the strong early hazard signal;
3. use K4 or an optimized distinct-value K2 split that handles zero atoms;
4. keep the independent CRC controller and causal shared envelope;
5. compare under identical N1, control size, common random numbers, and actual
   budget;
6. report fixed-anchor/selected-candidate conditional variance and repeated
   fixed-split acquisition variance alongside total outer coverage variance.

The no-CRC target-value/K4 result is a strong algorithmic lead, not yet a
production replacement: its individual-split expected budget can exceed 20.
An apples-to-apples CRC-controlled run is required before changing the public
DAPRO definition.

## Reproducible artifacts

- `analysis/diagnostics/dapro_score_noise_deep_audit.py`
- `outputs/dapro_binning_audit/score_noise_lpb_deep_summary.csv`
- `outputs/dapro_binning_audit/score_noise_lpb_endpoint_paired.csv`
- `outputs/dapro_binning_audit/score_noise_lpb_target_auc.csv`
- `outputs/dapro_binning_audit/score_value_lpb_outer50_summary.csv`
- `outputs/dapro_binning_audit/score_value_lpb_acquisition_summary.csv`
- `outputs/dapro_binning_audit/factorial_50split_summary.csv`
