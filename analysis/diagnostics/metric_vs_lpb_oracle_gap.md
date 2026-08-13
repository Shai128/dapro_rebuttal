# Why finite-budget LPB calibration can match its oracle while metric estimation cannot

Date: 2026-08-12

## The two reported oracles answer different questions

The LPB oracle (`oracle_survival_calibration`) observes every trajectory in a
random calibration split, selects a candidate lower bound from that split, and
then evaluates the selected bound on the complementary, fully observed test
split.  Its reported variance is therefore the variance of a downstream
coverage statistic under both calibration and test sampling.  It is not zero.

The metric workflow exposes two different full-budget references:

1. `oracle_full_budget` observes the fixed 6,000-row calibration--test union.
   That union does not change with the split seed, so its variance is exactly
   zero.  It is a truth line, not a competing estimator.
2. `oracle_split_full_budget` observes all 3,000 rows in each random calibration
   split.  It has zero conditional acquisition variance, but it retains split
   variance: 0.3449 squared percentage points for toxicity and 0.2500 for
   red-team over the 50 experimental splits.

## Fixed-metric acquisition variance

Let

\[
 A_i=\mathbf 1\{T_i\le M\},\qquad
 \widehat\mu_{\rm HT}=\frac1n\sum_i\frac{R_iA_i}{\pi_i},
\]

where `R_i` says that the trajectory was followed far enough to reveal `A_i`
and `pi_i` is its logged inclusion propensity.  Conditional on the fixed
benchmark rows and a frozen predictable policy,

\[
 \operatorname{Var}(\widehat\mu_{\rm HT}\mid\mathcal D)
 =\frac1{n^2}\sum_i A_i\left(\frac1{\pi_i}-1\right).
\]

Thus every missed positive trajectory contributes acquisition noise directly
to the reported number.  There is no discrete decision plateau.

For an LPB, by contrast, the HT curve is an intermediate object.  If noisy and
full-budget calibration choose the same candidate index, their test bounds and
test coverages are identical.  Acquisition error matters mainly when it changes
the candidate crossing the target-miscoverage boundary.

## Why the budget regimes are structurally different

For the two priority caches (`M=200`, `n=3,000`, `B=20`):

| setup | event rate by 200 | full-observation cost | cost of all event rows only |
|---|---:|---:|---:|
| toxicity/Qwen | 49.433% | 124.183 | 23.049 |
| red-team/Qwen | 74.467% | 95.971 | 44.904 |

The last column is `E[T 1{T<=M}]`.  It grants the allocator impossible advance
knowledge of which rows are safe and charges zero to safe rows.  Even under
that favorable fiction, `B=20` is below the cost required to reveal every
positive: slightly below it for toxicity and less than half of it for red-team.

The LPB target is instead a boundary event with prevalence near 10%.  DAPRO
can concentrate reach near that boundary.  In the existing 50-split LPB runs,
the selected-fixed-candidate HT variance diagnostic for no-CRC Generalized
DAPRO (`N1=50`) is 0.0083 squared percentage points on toxicity and numerically
zero on red-team, against total downstream oracle coverage variances of 0.4978
and 0.4936.  Acquisition noise is therefore hidden beneath the ordinary
calibration/test sampling floor.  On red-team the three no-CRC Generalized
DAPRO `N1` configurations selected exactly the same test coverage values as
the full-observation oracle in the stored runs.

## Exact clairvoyant finite-budget HT frontier

For active length `L_i=min(T_i,M)`, the exact unrestricted label-aware solution
to the pure-HT problem is a row-level inclusion coin followed by full
continuation,

\[
 \pi_i=\operatorname{clip}\left(s\sqrt{A_i/L_i},\epsilon,1\right),
\]

where `s` meets the expected budget.  This uses latent `A_i` and `L_i` and is
not deployable; it is a lower bound on any real allocation method.  The table
reports its conditional variance for an `n=3,000` metric estimate.

| setup | B=5 | B=10 | B=20 | B=30 | B=40 | zero-variance saturation cost |
|---|---:|---:|---:|---:|---:|---:|
| toxicity/Qwen | 3.637 | 1.132 | 0.087 | 0.000 | 0.000 | 23.049 |
| red-team/Qwen | 15.631 | 6.581 | 2.117 | 0.726 | 0.138 | 44.904 |

All entries are squared percentage points.  Adding the empirical split floors
gives the best possible joint variance at `B=20`: approximately 0.432 for
toxicity and 2.367 for red-team.  Consequently:

- toxicity can in principle approach its 0.345 split-full oracle, but only if
  the policy nearly identifies positives and their lengths before paying to
  observe them;
- red-team cannot approach its 0.250 split-full oracle with the pure HT
  estimator at `B=20`, even with a clairvoyant allocation rule.  A budget near
  40--45 is needed before the pure-HT acquisition term becomes negligible.

The fixed-union truth has variance zero, so no random split estimator can match
it except through complete observation, a perfect model, or a change in the
target/experimental protocol.

## A model-assisted route that changes the attainable frontier

Let `m_i` be a frozen, zero-cost prediction of `A_i` available for every row,
for example the initial-PMF event probability by `M`.  The augmented HT
estimator

\[
 \widehat\mu_{\rm AHT}
 =\frac1n\sum_i\left[m_i+\frac{R_i}{\pi_i}(A_i-m_i)\right]
\]

is design-unbiased for the fixed-benchmark mean for any fixed `m_i`.  Its exact
conditional variance is

\[
 \frac1{n^2}\sum_i(A_i-m_i)^2\left(\frac1{\pi_i}-1\right).
\]

Model error now replaces the raw event indicator in the allocation objective.
An initial-PMF plug-in schedule can use event-time mass
`p_i(t)(1-m_i)^2` and put the safe-tail mass `p_i(T>M)m_i^2` at the horizon,
then solve the same antitonic-PAV/common-scale program as Metric-optimal PMF.

An offline exact-variance prototype on the full aligned caches gave:

| setup | schedule | true expected cost | conditional variance | plus split floor |
|---|---|---:|---:|---:|
| toxicity/Qwen | residual-PMF, model scale | 18.087 | 3.876 | 4.221 |
| toxicity/Qwen | residual-PMF, true-cost scalar diagnostic | 20.000 | 3.440 | 3.785 |
| red-team/Qwen | residual-PMF, model scale | 20.196 | 2.296 | 2.546 |
| red-team/Qwen | residual-PMF, true-cost scalar diagnostic | 20.000 | 2.324 | 2.574 |

The true-cost scalar is an oracle diagnostic; a real method would estimate it
on an independent pilot/CRC fold.  The initial predictor is materially biased
as a plug-in estimator (mean 45.306% versus 49.433% on toxicity and 79.495%
versus 74.467% on red-team), but augmentation removes that bias in expectation.
The prototype is not yet a production 50-split allocator; it establishes the
potential of changing the influence function rather than continuing to refine
pure-HT allocation alone.

## Recommended oracle hierarchy

Metric-estimation plots should distinguish four references:

1. **Fixed-union truth:** full calibration+test, variance zero; a horizontal
   truth line, never presented as a fair competitor.
2. **Split full-observation oracle:** full calibration only; the empirical
   sampling floor, with its actual cost shown.
3. **Budget-matched clairvoyant HT lower bound:** latent-label water filling;
   nondeployable and clearly separated from methods.
4. **Information oracle:** a budget-matched policy using the true conditional
   law of future outcomes but not the realized future.  On real data this can
   only be approximated; on synthetic data it is the cleanest deployable-policy
   efficiency benchmark.

For practical progress, the most promising next method is a residual/influence
Generalized DAPRO: use `(A_i-m_i)^2` for the hard fit objective or its causal
conditional expectation for soft prefix masses, use remaining residual risk
per expected remaining cost as the score, and report the augmented HT estimate.
It preserves the same general DAPRO allocation framework while attacking the
quantity that actually determines estimator variance.
