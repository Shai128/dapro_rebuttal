# Unified audit of DAPRO for metric estimation and LPB construction

Date: 2026-08-12

## Executive conclusion

The two-bin representation does discard information from the current-prefix
score, but it does **not** reduce the deployed dynamic policy to two trajectory
propensities.  It supplies at most two raw continuation values at each turn;
the preceding bin path creates many cumulative-reach states, and the common
cumulative correction creates many final conditional probabilities.  On the
two priority Qwen setups, moving from the current hazard/K2 policy to a
target-aware future-value score and four smoothly interpolated bins lowered
the exact metric-HT acquisition variance by 3.40% (toxicity) and 1.23%
(red-team).  A low-dimensional continuous Basis-DAPRO did not improve this
variance and was much worse on red-team.  Thus K2 is a real approximation,
but not the dominant bottleneck in these experiments.

The statistical target matters more than the number of bins:

* For a fixed unsafe-event metric, the hard event-weighted DAPRO objective is
  the exact conditional Horvitz--Thompson (HT) acquisition variance.  The
  approximations are the soft/model coefficients, chosen score and policy
  class, finite Phase-I fit, and budget controller--not the variance formula.
* For an LPB, a fixed Target-A objective is exact only for one frozen candidate
  rate.  The reported coverage is a nonlinear candidate-selection functional.
  Its exact acquisition variance is governed by boundary-crossing
  probabilities and output jumps.  A one-anchor DAPRO objective is therefore
  a proxy for the final LPB statistic.
* For metric estimation, the largest validated improvement came from changing
  the estimator, not merely the allocator.  A sequential augmented HT (AHT)
  estimator uses every acquired prefix through prediction increments and
  remains design-unbiased for any frozen predictor.  At budget 20 it reduced
  exact conditional variance from 7.697 to 4.511 pp^2 on toxicity and from
  9.336 to 2.522 pp^2 on red-team under the same initial-PMF event schedule.
  Its relative advantage grows at budgets 10 and 5.

For LPB construction, a causal target-value score with K4 reduced the
acquisition component substantially on toxicity--about 80.5% for the selected
fixed candidate and 48.5% for the exact nonlinear selected coverage in
fixed-split repetitions--but the total 50-split coverage variance remained
unchanged because calibration/test composition was roughly one hundred times
larger.  On red-team the acquisition component was already numerically zero,
so no allocation refinement could improve the reported 50-split variance.

The recommended unified framework is therefore the common nonnegative
squared-influence/reach problem described below, with different target masses
and estimators for metric and LPB tasks.  The strongest currently supported
production directions are:

1. metric: sequential AHT plus a causal information-gain/residual-tail DAPRO
   allocation, with the existing shared-PAV CRC controller when model-budget
   transfer is not assumed;
2. LPB: corrected causal shared-PAV CRC, the target-value score, and K4 only
   where Phase-I support is adequate; retain K2 as the low-variance small-fit
   baseline;
3. do not productionize the tested analytic-margin, pivotal, or centered
   smooth-selection objectives--the real-data pilots failed;
4. treat continuous Basis-DAPRO as a useful compression/speed option, not as a
   demonstrated variance improvement.

## 1. What the two-bin code actually does

At turn `t`, current generalized DAPRO reads the diagonal conditional PMF
entry

```
score_it = conditional_grid[i, t, t]
```

which is the estimated instantaneous event hazard based on the current prefix
`X_it`.  It is not restricted to `X_i0`.  On the active Phase-I rows it forms
an empirical median threshold and ties log continuation probabilities inside
the low/high score groups.  The exact solver optimizes these tied values; this
is direct optimization of a deployable lookup table, not the legacy
oracle-policy regression projection.

The raw table has at most two values per time, but the deployed policy is
history-adaptive.  If `p_it` is the raw conditional probability, cumulative
reach is `rho_it=prod_{s<=t}p_is`; different earlier bin paths give different
`rho_it`.  The shared cumulative correction, positivity floor, causal shared
envelope and optional CRC contraction operate on these cumulative paths and
can therefore give many final conditional values even when the raw current
turn has one or two values.

For toxicity/Qwen, seed 0, N1=50 and B=20:

| turn | raw current-turn values | raw cumulative reaches | corrected conditional values |
|---:|---:|---:|---:|
| 0 | 2 | 2 | 2 |
| 20 | 2 | 84 | 84 |
| 50 | 1 | 255 | many |
| 199 | 2 | 474 | 474 |

Representative median thresholds and raw low/high probabilities include
`t=1: 0.002440 -> (0.2313,0.2746)`, `t=20: 0.006069 ->
(0.9562,0.9562)`, and `t=200: 0.000417 -> (0.6302,0.8248)`.
The solver pooled the two bins to the same raw value at 178 of 200 turns in
this split.  Thus the correct criticism is: K2 loses resolution in the new
score conditional on the existing history state.  It does not make the whole
policy static or two-valued.

The information loss is measurable.  The raw hazard had thousands of unique
values.  On toxicity its AUC for the eventual unsafe event at turns
1/2/5/10/20/50/100/150 was
`.749/.686/.709/.749/.740/.725/.718/.698`; after median binning it was
`.682/.645/.674/.693/.680/.660/.663/.643`.  K2 retains roughly 50% of
within-time score variance, K4 about 78%, and K8 about 90%.  There is also an
atom pathology: if the median is zero, `searchsorted(..., side="right")`
sends every zero to the high bin, can empty the low bin, and may collapse the
turn to a time-only policy.  A future implementation should put cutpoints
between distinct score values and enforce minimum cell support.

Any strictly increasing recalibration of the hazard--logit, square root, or
temperature scaling--leaves quantile assignments unchanged.  A better score
must change ranking, not merely calibrate values.

## 2. Unified acquisition design

For row `i` and active turn `t`, let `rho_it` be cumulative reach, `a_it>=0`
be target or squared-influence mass, and `d_it>=0` be acquisition-cost mass.
The common program is

\[
 \min_{\rho\in\mathcal P}
 \frac1n\sum_{i,t}a_{it}(\rho_{it}^{-1}-1),
 \qquad
 \frac1n\sum_{i,t}d_{it}\rho_{it}\le B.
\]

This separates five axes that should not be conflated:

1. the downstream target/influence that defines `a`;
2. hard outcomes versus a soft model used to estimate `a`;
3. the information/score exposed to the policy;
4. bins, splines, saturated PAV or another policy representation;
5. model-budget bisection, a projection margin, or CRC control.

Existing methods are instances of this program:

* Target-A DAPRO uses a hard event for one frozen LPB candidate.
* Definitive DAPRO uses a regularized hard endpoint mass, the current hazard,
  and K2.
* soft-prefix Generalized DAPRO uses model hazard mass over the named target
  horizon, the current hazard score, and K2.
* Metric-PMF uses `P(T=t|X_i0)` and `P(T>=t|X_i0)` in a saturated
  precommitted cumulative-reach class; antitonic PAV is its exact plug-in
  solution.
* a value-score DAPRO changes only the information map, using predicted
  remaining target risk per expected remaining cost.
* Basis-DAPRO changes only the representation, replacing time-by-bin cells by
  a low-dimensional monotone rank/time basis.

If fixed features satisfy `log rho_it=x_it^T theta`, the objective is a sum of
`a exp(-x^T theta)`, and the budget constraint is a sublevel set of a sum of
`d exp(x^T theta)`.  Both are convex.  Linear probability, monotonicity and
floor constraints preserve convexity.  This provides a genuine superclass of
the current bins, provided scores/knots/influence weights are frozen before
the solve.  Jointly learning cutpoints or target weights inside the same solve
need not remain convex.

## 3. Score sufficiency and quantization theory

In the one-stage unclipped problem, among policies measurable with respect to
a score sigma-field `G`, the optimum is

\[
 J_G^*=\frac{\{E\sqrt{E(a\mid G)E(d\mid G)}\}^2}{B}.
\]

It equals the full-information optimum only if the benefit/cost ratio `a/d`
is essentially constant after conditioning on the score.  Binning is a
further coarsening.  For cells `C`,

\[
 J_{\rm bins}^*=\frac{\left(\sum_C\sqrt{A_CD_C}\right)^2}{B},
 \quad A_C=E[a1_C],\quad D_C=E[d1_C].
\]

Refining a fixed population partition cannot worsen this optimum.  It can
worsen deployment performance because the fitted cell levels and cutpoints
have estimation error.  A simple rare-event example makes the possible loss
large: if a fraction `eta<1/2` of perfectly ranked rows contains all target
mass, a median split mixes those rows with negatives and has regret ratio
`1/(2 eta)`.  Conversely, if clipping/pooling collapses the unrestricted
optimum or the benefit/cost ratio is nearly two-level, K2 is sufficient.

The current instantaneous hazard is optimal only for a one-step target.  For
a block decision that either acquires the remaining trajectory or stops, the
Neyman rule is

\[
 q^*(H_{it})\propto
 \sqrt{R(H_{it})/C(H_{it})},
\]

where `R` is remaining squared-influence/target risk and `C` is expected
remaining acquisition cost.  This motivates the causal future-value score.
It uses the full current PMF row `P(T=u|H_it)`, not future prefixes.

## 4. Metric estimation: exact variance and better estimators

For `A_i=1{T_i<=M}`, endpoint resolution `R_i` and exact inclusion propensity
`pi_i`, ordinary HT is

\[
 \widehat\mu_{HT}=n^{-1}\sum_iR_iA_i/\pi_i,
\qquad
 \operatorname{Var}(\widehat\mu_{HT}\mid\mathcal D)
 =n^{-2}\sum_iA_i(\pi_i^{-1}-1).
\]

The hard A-weighted objective is therefore exact for the fixed metric.  The
soft hazard mass is a model-based Rao--Blackwell-style proxy.  In the priority
caches, pooled soft event mass was 0.446 versus a realized 0.493 on toxicity
and 0.709 versus 0.743 on red-team, with event-time total variation near 0.14.
Across broad policy changes the soft proxy correlated strongly with exact
risk, but within the close policies of one split its rank correlation was only
0.43 on toxicity and 0.10 on red-team in Phase I.  This is where the apparent
"exact objective" can cease to rank candidate policies well.

### Sequential augmented HT

Let a frozen predictor emit a predictable sequence
`m_i0,...,m_iL`, and force `m_iL=A_i` once the outcome is fully known.  With
increments `Delta_it=m_it-m_i,t-1` and reach indicator `R_it`, define

\[
 \widetilde A_i=m_{i0}+\sum_{t=1}^{L_i}
 \frac{R_{it}}{\rho_{it}}\Delta_{it},
 \qquad \widehat\mu_{AHT}=n^{-1}\sum_i\widetilde A_i.
\]

For any frozen, possibly misspecified predictor this is conditionally
design-unbiased, because each reached increment has expectation equal to the
full increment and the increments telescope to `A_i`.  Its exact row-level
design variance is

\[
 \sum_{t=1}^{L_i}
 (\rho_{it}^{-1}-\rho_{i,t-1}^{-1})
 (A_i-m_{i,t-1})^2,
 \qquad \rho_{i0}=1.
\]

If `m_it=E[A_i|H_it]` is the Doob martingale, the population acquisition
inflation is `sum_t E[Delta_it^2(1/rho_it-1)]`.  This yields an
information-gain DAPRO mass `a_it=E[Delta_it^2|H_i,t-1]`.  A conservative
misspecification-robust alternative uses residual-tail masses
`E[(A_i-m_i,t-1)^2|H_i,t-1]`.

Under the initial-PMF event allocation, 50-split exact conditional variances
were:

| budget | setup | ordinary HT | terminal AHT | sequential AHT |
|---:|---|---:|---:|---:|
| 20 | toxicity | 7.697 | 6.193 | 4.511 |
| 20 | red-team | 9.336 | 2.801 | 2.522 |
| 10 | toxicity | 17.046 | 13.066 | 9.702 |
| 10 | red-team | 21.150 | 6.214 | 5.655 |
| 5 | toxicity | 35.745 | 26.775 | 20.084 |
| 5 | red-team | 44.779 | 13.040 | 11.922 |

All values are squared percentage points.  At budgets 10 and 5 the event
schedule used only 8.60/4.30 turns on toxicity and 9.79/4.89 on red-team
because the plug-in model budget is not the true event-stopped cost.  This is
the same transfer issue as no-CRC DAPRO, not an unbiasedness failure.  The
residual/information-gain schedules often improved the AHT variance further,
but sometimes slightly exceeded nominal cost; they require the causal
shared-envelope CRC controller for a production budget guarantee.

### Score/bin experiments

At B=20 and N1=50 over the same 50 outer splits, hazard/K2 to
future-risk/smooth-K4 changed exact ordinary-HT variance from 7.2653 to 7.0178
on toxicity (-3.40%) and 10.3238 to 10.1963 on red-team (-1.23%).  Fixed-split
500-acquisition experiments verified the exact variance formula.  The
variance of the single realized estimate across only 50 outer splits can move
in the opposite direction; its relative Monte Carlo standard error is about
20%, and it mixes split, policy-fit and acquisition randomness.

A continuous low-rank Basis-DAPRO used 16 parameters rather than the current
400 time/bin cells and was roughly 10--19 times faster.  It tied current K2 on
toxicity (7.273 versus 7.265 exact HT variance) and was substantially worse on
red-team (12.221 versus 10.324).  It is valuable compression, not evidence
that continuous score resolution is intrinsically better.  Per-time policy
flexibility can be more valuable than within-bin resolution.

## 5. LPB construction: the exact nonlinear target

For nested candidates `f_0<=...<=f_J`, define
`A_ij=1{T_i<f_j(X_i)}` and the common-acquisition HT rates

\[
 M_j=n^{-1}\sum_i Z_iA_{ij}/\pi_i.
\]

Then

\[
 \operatorname{Cov}(M_j,M_k\mid\mathcal D)
 =n^{-2}\sum_iA_{i,\min(j,k)}(\pi_i^{-1}-1).
\]

Thus Target-A is exact for one fixed anchor.  Let the strict-prefix selector
be `K=max{j:M_j<alpha}`, and let `c_j` be the fixed test coverage of candidate
`j`, with `Delta c_j=c_j-c_j-1`.  Since

\[
 Y=c_K=c_0+\sum_j\Delta c_j I\{M_j<\alpha\},
\]

and these crossing indicators are nested, if
`F_j=P(M_j<alpha|D)` then exactly

\[
 \operatorname{Var}(Y\mid\mathcal D)
 =\sum_j\Delta c_j^2F_j(1-F_j)
 +2\sum_{j<k}\Delta c_j\Delta c_kF_k(1-F_j).
\]

This is the acquisition variance of the actual nonlinear LPB output.  It
depends on candidate margins, nested covariance and coverage jumps, none of
which appears in a one-anchor objective.  Adjacent candidate-band HT
increments use disjoint rows and are exactly independent; their Gaussian
limit is a time-changed Brownian path.

The executable diagnostic matches exact enumeration and Monte Carlo to
numerical precision.  Pivotal Efron--Stein finite-difference weights give a
valid nonnegative upper-bound target; a smooth selector gives ordinary
influence weights in regular regions.  However, the N1=50 real-data pilots
exposed severe nonregularity: 1,257 raw candidates collapsed to only about 26
distinct Phase-I event patterns, and many empirical rates lay exactly at
alpha.  Analytic margin weights discarded cross-candidate cancellation and
performed 10.7 times worse than the target-value reference in the bounded
three-split pilot.  Simulated pivotal weights were extremely sparse.  Neither
should be productionized from these results.

There is also a design-law distinction.  For a smooth LPB functional with
gradient `gamma`, superpopulation split influence is
`sum_j gamma_j(A_ij-m_j)`, whereas fixed-population ordinary-HT acquisition
influence is `sum_j gamma_j A_ij`.  Centering is correct for the split term but
wrong for conditional HT noise; a centered pilot spent budget on negative
safe-row influence and failed decisively.  Correcting the pilot to the
uncentered acquisition influence reduced its mean nonlinear variance by 91.9%
(0.05669 to 0.00461 pp^2 at the widest tested bandwidth), confirming the
distinction.  It nevertheless remained 6.5 times worse than the soft
target-value reference (0.000705 pp^2): the hard influence had only about
8 effective Phase-I rows and underused roughly 0.61 turns.  A future
smooth-selection method must use uncentered acquisition influence, then
Rao--Blackwellize `psi(T)^2` with the prefix model and estimate the candidate
curve from a larger independent or zero-cost `X_i0` prediction population.
Centered influence is appropriate only for the split component or for an
augmented/Hajek estimator whose residual design variance is separately
derived.

### LPB empirical result

On toxicity, changing hazard/K2 to causal target-value hard K4 left total
50-split coverage variance essentially unchanged (0.491091 versus 0.491131
pp^2), but reduced:

* selected-fixed-candidate conditional variance by 80.5%;
* full-calibration-oracle fixed-candidate variance by 81.2%;
* the nominal anchor risk by 71%;
* candidate switching from 60% to 32%;
* five-fixed-split nonlinear acquisition variance by 48.5%.

The total LPB variance did not move because calibration/test composition was
about 0.49 pp^2, while acquisition variance was roughly 0.002--0.005 pp^2.
On red-team all tested policies selected identical candidates in all 50
splits, with acquisition variance numerically zero; the 0.493553 pp^2 total
was entirely the data-split/test component.

## 6. Why LPB can appear to match its oracle while the metric cannot

The metric and LPB "oracle" comparisons are not like-for-like.  Full budget on
the fixed 6,000-row calibration+test union is a truth line with exactly zero
variance.  Full budget on each random 3,000-row calibration split retains
split variance: 0.3449 pp^2 toxicity and 0.2500 red-team.  A finite-budget
metric estimator adds acquisition noise directly to every reported mean.

An LPB calibration curve is intermediate.  If noise does not cross a candidate
boundary--or crosses only candidates with identical integer bounds--the final
test coverage is unchanged.  Its full-budget oracle itself retains about 0.49
pp^2 from calibration/test sampling, so a finite method can look oracle-like
once its much smaller crossing variance falls below that floor.

At B=20 even an impossible label-clairvoyant metric allocator cannot generally
match the split-full oracle.  Revealing every positive alone would cost 23.05
turns on toxicity and 44.90 on red-team.  The exact label-aware finite-budget
HT frontier predicts total variances near 0.432 pp^2 toxicity and 2.367
red-team after adding split variance.  The toxicity gap can nearly close with
strong outcome prediction; the red-team gap cannot close at B=20 with
ordinary HT.  Sequential AHT changes this frontier by extracting unbiased
information from partial trajectories, which is why it is the most promising
route toward the oracle.

## 7. Approximation and controller accounting

For a fixed target `a`, score and feasible classes give the decomposition

\[
 J(\widetilde\rho)-J(\rho^*)=
 \underbrace{J(\rho_S^*)-J(\rho^*)}_{\text{score sufficiency}}+
 \underbrace{J(\rho_R^*)-J(\rho_S^*)}_{\text{representation/binning}}+
 \underbrace{J(\widehat\rho_R)-J(\rho_R^*)}_{\text{model+finite fit}}+
 \underbrace{J(\widetilde\rho)-J(\widehat\rho_R)}_{\text{controller}}.
\]

The error between this fixed-target surrogate and the true nonlinear LPB
variance is a separate target error and need not be nonnegative.  This audit
locates the observed losses as follows:

* representation/binning: measurable but small on the priority metric setups;
* score sufficiency: future target value helps more than K alone, especially
  for LPB acquisition on toxicity;
* coefficient/model fit: substantial--soft masses are miscalibrated and rank
  close policies poorly with only 25--100 fit rows;
* target mismatch: decisive for LPB because fixed-anchor risk is not selected
  coverage variance;
* controller distortion: real and must be measured; CRC certifies marginal
  expected cost, not variance optimality or per-split cost;
* irreducible split/test variation: dominant in the reported LPB experiments.

The former deployment-row cap also used future score paths and was not causal.
It has been replaced by a Phase-I-only nonincreasing shared-PAV envelope
`e_t`, followed online by `rho_cap(t)=min(rho_base(t),e_t)`.  This guarantees
predictability and a pathwise row-cost envelope for CRC.  Corrected capped-CRC
names carry the `causal_shared_pav_v1` suffix so stale noncausal results cannot
silently mix with new runs.

## 8. Recommended next implementation

### Metric

1. Keep Metric-PMF/no-CRC as the inexpensive precommitted allocator when its
   model-budget transfer assumption is acceptable.
2. Add sequential AHT as the primary estimator.  It is exactly
   design-unbiased for a frozen model and uses partial prefixes that ordinary
   event-only HT throws away.
3. Fit a history-adaptive Generalized DAPRO variant with residual-tail or
   expected-squared-update masses and the value-per-cost score.  Use K2 at
   very small policy-fit sizes; test K4 only with adequate cell support.
4. Wrap the candidate family in the corrected shared-PAV CRC controller when
   a distribution-free marginal expected-budget guarantee is required.
5. Evaluate exact AHT design variance, ordinary HT variance, 50-split
   variance, and true expected cost separately.

### LPB

1. Use the corrected causal shared-PAV cap for every capped CRC DAPRO result.
2. Retain the current soft anchor at small N1, but replace instantaneous
   hazard ranking by causal target-value ranking; K4 is justified only where
   minimum support is adequate.
3. Report exact nonlinear fixed-split acquisition variance in addition to
   50-split coverage variance.  The latter can be insensitive to a 50--80%
   acquisition improvement.
4. Compress duplicate candidate event patterns before any selection-target
   calculation.
5. Do not deploy naive inverse-margin, sparse pivotal, or centered smooth
   influence weights.  A future selector-aware method must use uncentered HT
   acquisition influence and be stabilized on a larger independent or
   model-predicted population.

### Representation

Keep K2 as the robust small-fit baseline.  A safe next binning improvement is
an optimal distinct-value K2 cutpoint with a minimum cell size, followed by a
predeclared K4 test at N1=100/200.  Continuous Basis-DAPRO is attractive when
runtime/model compression matters, but the present variance results do not
justify replacing per-time K2.  A production convex basis implementation
should use an exponential-cone or specialized primal-dual solver rather than
prototype SLSQP.

## Reproducible artifacts

* `analysis/diagnostics/dapro_binning_audit.py`
* `analysis/diagnostics/lpb_dapro_binning_audit.py`
* `analysis/diagnostics/dapro_score_value_variance_audit.md`
* `analysis/diagnostics/lpb_selection_variance.py`
* `analysis/diagnostics/lpb_selection_dapro_audit.py`
* `analysis/diagnostics/metric_augmented_ht_audit.py`
* `analysis/theory/selection_influence_dapro.md`
* `src/predictive_bounds/experiments/basis_dapro_prototype.py`
* `src/predictive_bounds/experiments/basis_dapro_metric_diagnostic.py`
* `outputs/dapro_binning_audit/score_value_metric_outer50_summary.csv`
* `outputs/dapro_binning_audit/score_value_metric_acq500_summary.csv`
* `outputs/dapro_binning_audit/score_value_lpb_outer50_summary.csv`
* `outputs/dapro_binning_audit/score_value_lpb_acquisition_summary.csv`
* `outputs/dapro_binning_audit/metric_augmented_ht_50split_summary.csv`
* `outputs/dapro_binning_audit/SELECTION_DAPRO_PILOT.md`

The diagnostic methods remain isolated unless explicitly stated.  They do
not alter the production allocator registry.
