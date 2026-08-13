# Is there one generally best DAPRO variant?

Date: 2026-08-13

## Answer

Among the deployable variants tested here, the best single **DAPRO core** is
soft-prefix Generalized DAPRO with the task-appropriate target event, the
current-prefix hazard score, and two bins.  This conclusion holds separately
inside the no-CRC and corrected-CRC regimes at `N1=50`:

* it has the lowest exact metric-estimation acquisition variance on all eight
  real cached setups with no CRC, and on all eight with CRC;
* it has the best aggregate LPB coverage variance, MSE to 90% coverage, and
  selected-candidate acquisition variance in both controller regimes;
* at `N1=200`, it retains the lowest exact metric acquisition variance on all
  four setups tested, with and without CRC;
* in the broader historical LPB archive (32 setup/budget cells, 50 splits), it
  has the best aggregate LPB results at `N1=50` and `N1=100`, and remains the
  best corrected-CRC default.

This is strong empirical generality, not a universal mathematical dominance
theorem.  At `N1=200`, total LPB coverage rankings occasionally change because
the allocation component is already tiny and calibration/test composition
dominates.  A sufficiently misspecified conditional PMF could also make hard
Definitive DAPRO better than its soft counterpart.  Therefore the defensible
claim is **best generally supported default**, not **best for every possible
distribution**.

CRC is an orthogonal controller choice.  If a finite-sample marginal expected
budget guarantee is required, use soft Generalized DAPRO with the corrected
causal shared-PAV CRC controller.  If the Phase-I-to-deployment cost-transfer
assumption is accepted, the no-CRC version is usually less conservative and
has lower variance.  The target/coefficient choice can be unified; the
controller cannot be declared uniformly better because it trades efficiency
for a guarantee.

If the paper must name exactly one complete method rather than one coefficient
family, use

\[
\boxed{\text{soft-prefix Generalized DAPRO + corrected causal CRC}.}
\]

Keep its no-CRC counterpart as the efficiency ablation.  This choice is not
because CRC improves the soft coefficient; CRC is selected because it adds
the finite-sample marginal expected-budget guarantee.

## There are three distinct coefficient estimators, not four

“Generalized DAPRO with hard indicators” is not a fourth method once the
target, score, K2 representation, and controller are fixed.  In the common
solver, a hard row weight is simply converted into a one-hot endpoint mass:

\[
 a^{\rm hard}_{it}=A_i\mathbf 1\{t=L_i\}.
\]

Therefore:

* hard Generalized DAPRO with coefficient \(A_i\) is Target-A DAPRO;
* hard Generalized DAPRO with coefficient
  \((A_i+0.001)/1.001\) is Definitive DAPRO;
* soft Generalized DAPRO replaces that endpoint one-hot mass by causal prefix
  hazard mass and is the only genuinely different coefficient estimator.

Repeating \(A_i\) at every prefix would be a different objective, but not the
HT variance objective: it would count one terminal target event many times.
It is therefore not an appropriate “hard-prefix” competitor.

## Matched targets and methods

Every method uses the correct task-specific target:

### Metric estimation

\[
A_i^{\mathrm{metric}}=\mathbf 1\{T_i\le 200\}.
\]

### LPB construction

\[
A_i^{\mathrm{LPB}}
=\mathbf 1\{T_i<f_{\alpha}(X_i)\},
\qquad \alpha=0.10,
\]

where the raw-alpha candidate is frozen before Phase-II acquisition.

The coefficient estimators are:

1. **Target-A DAPRO:** the unregularized realized binary endpoint event.
2. **Definitive DAPRO:** the same realized event with global regularization
   `(A+0.001)/1.001`.
3. **soft Generalized DAPRO:** causal model-integrated event mass over the same
   named target horizon.

All three use the same current-prefix hazard score, two-bin deployable policy,
positivity floor, Phase-I split, budget, and CRNs.  Thus the main comparison
isolates coefficient estimation.  CRC variants use a corrected shared-PAV
envelope learned only from the policy-fit fold and an independent control
fold; no future-prefix row cap is used.

For metric estimation, the law of total variance makes the conditional
comparison especially strong.  Conditional on a calibration split and the
frozen fitted policy, HT is unbiased for that split's fully observed rate, so

\[
 \operatorname{Var}_{S,R}(\widehat\mu)
 =\operatorname{Var}_{S}(\mu_S)
  +\mathbb E_S\!\left[
    \operatorname{Var}_{R}(\widehat\mu\mid S)
  \right].
\]

The split-oracle term is identical for every allocator.  Consequently, a
method that lowers the exact conditional acquisition variance also lowers the
true total split-plus-acquisition variance, even when the empirical variance
of ten one-draw estimates happens to reverse because it is noisy.

## Corrected eight-setup experiment

The setups are toxicity/Qwen, toxicity/Phi, red-team/Qwen judge,
red-team/LlamaGuard, three hallucination target models, and AutoIF/Qwen.  All
use budget 20, horizon 200, calibration size 3,000, and `N1=50`.  The metric
and LPB runs use the same outer seeds and common acquisition randomness.

### Metric estimation: exact acquisition variance

Mean across the eight setups, in squared percentage points:

| Controller | Target-A | Definitive | soft Generalized | Soft setup wins |
|---|---:|---:|---:|---:|
| No CRC | 24.880 | 7.840 | **4.780** | **8/8** |
| Corrected CRC | 49.149 | 24.678 | **12.644** | **8/8** |

Relative to Definitive, soft Generalized lowers mean exact acquisition
variance by 39.0% without CRC and 48.8% with CRC.  Its geometric mean regret
to the best method is exactly 1.0 in both regimes; Definitive's is 1.70 and
2.06, while hard Target-A's is 6.63 and 4.19.

The raw variance of ten final metric estimates is much noisier and does not
rank methods consistently.  Ten outer splits provide only nine variance
degrees of freedom and mix split, policy-fit, and acquisition randomness.
The exact conditional variance is the stable paired diagnostic for allocation
efficiency; the final 50-split variance remains the correct reporting metric
when enough splits are available.

### Metric `N1=200` check

Four setups, five paired outer splits:

| Controller | Target-A | Definitive | soft Generalized |
|---|---:|---:|---:|
| No CRC | 16.955 | 11.260 | **10.072** |
| Corrected CRC | 26.036 | 14.748 | **10.024** |

Soft Generalized wins every setup in this smaller check as well.  Thus its
advantage is not solely a `N1=50` phenomenon, although the advantage narrows
without CRC as the hard coefficient becomes less noisy.

### LPB construction at `N1=50`

Means across eight setups:

| Controller | Method | Coverage variance | MSE to 90% | Acquisition variance | Cost |
|---|---|---:|---:|---:|---:|
| No CRC | Target-A | 1.041 | 1.070 | 0.677 | 17.953 |
| No CRC | Definitive | 0.978 | 1.051 | 0.292 | 17.421 |
| No CRC | soft Generalized | **0.830** | **0.873** | **0.153** | 17.243 |
| CRC | Target-A | 6.980 | 7.133 | 8.726 | 5.506 |
| CRC | Definitive | 0.898 | 1.041 | 0.339 | 15.784 |
| CRC | soft Generalized | **0.787** | **0.816** | **0.165** | 16.025 |

All variances and MSEs are squared percentage points.  Soft Generalized is a
winner or exact tie on all eight LPB setups under both controllers.  Several
hallucination/guard setups are ties because every serious method already has
zero selected-candidate acquisition variance.

Hard Target-A + CRC is particularly poor.  With only 25 policy-fit rows, its
sparse hard objective often gives the causal envelope and CRC family too
little useful target shape.  CRC then correctly selects a strongly contracted
policy.  The resulting mean cost is only 5.51, but the low cost is not a
virtue: coverage MSE and acquisition variance explode.  CRC guarantees budget
control; it does not repair a noisy coefficient estimator or force full use
of the available budget.

Every corrected CRC selector certificate is valid in these experiments.  The
per-split `total_expected_budget_valid` flag is not the CRC theorem: CRC gives
a marginal expected-budget guarantee, so some fixed splits can conditionally
exceed 20 even when the selector is valid.  Mean CRC cost is below 20 for all
families.

## The broader 32-cell LPB archive

The archive contains 32 dataset/model/budget cells, 50 random splits, and
`N1=50,100,200`.  The no-CRC rows are directly usable.  Its CRC rows used the
older future-path cap, so they are included only as a sensitivity check and
not as causal production evidence.

At `N1=50` and `N1=100`, soft Generalized has the lowest aggregate coverage
variance, MSE, and acquisition variance.  At `N1=200`, no-CRC aggregate total
coverage variance is mixed:

| Method | Coverage variance | MSE to 90% | Acquisition variance | Budget-valid split rate |
|---|---:|---:|---:|---:|
| Target-A | **0.974** | **1.000** | 0.564 | 0.554 |
| Definitive | 1.069 | 1.084 | **0.458** | 0.967 |
| soft Generalized | 1.166 | 1.185 | 0.520 | 0.972 |

This is the clearest reason not to say “universally best.”  Target-A happens
to have the smallest aggregate total coverage variance in this slice, but it
has worse acquisition variance, uses more cost, and satisfies the conditional
budget diagnostic on only 55.4% of split cells.  Total LPB coverage is mostly
split/test variance, so small acquisition differences need not determine its
ranking.  Definitive is the better `N1=200` allocation in this slice, while
soft Generalized remains the safer cross-task default.

## Why soft Generalized generally wins

The hard target is statistically exact but empirically sparse.  With a frozen
policy,

\[
\operatorname{Var}(\widehat\mu_{\mathrm{HT}}\mid\mathcal D)
=\frac1{n^2}\sum_iA_i(\pi_i^{-1}-1).
\]

Target-A therefore uses the right population coefficient.  But DAPRO must
learn a history-to-probability map from Phase I.  A hard binary endpoint gives
one noisy coefficient per full trajectory and zeros for most target-negative
rows.  The optimizer can saturate a handful of positives and learn a brittle
table.

Definitive adds a small global component, preventing complete indifference on
negative rows.  It is much more stable than unregularized Target-A, but the
regularization does not add target-specific information.

Soft Generalized replaces the endpoint with causal prefix event masses.  If
the conditional model is correct, this is a Rao--Blackwellization of the hard
coefficient: it preserves the population objective while reducing Phase-I
coefficient variance and supplies useful signal at many prefixes.  Even under
moderate misspecification, the variance reduction can dominate the modeling
bias.  That is what happens in the eight tested caches.

The CRC comparison sharpens this diagnosis.  At \(N_1=50\), CRC leaves only
25 rows for fitting the policy shape.  Hard target events are then extremely
sparse, whereas every active prefix supplies a model-integrated soft mass.
Accordingly, the soft improvement over Definitive grows from 39.0% without
CRC to 48.8% with CRC for metric estimation.  At \(N_1=200\), the no-CRC
advantage narrows to 10.6%, exactly as expected if coefficient-estimation
noise is the main mechanism.

There is no distribution-free dominance.  A badly misspecified conditional
PMF can rank prefixes incorrectly; with a sufficiently large Phase I,
Definitive can then be better.  This motivates monitoring soft-vs-hard
coefficient calibration rather than asserting a theorem that soft must win.

## Other invented DAPRO variants

### Target-value smooth K4

For metric estimation, replacing the current hazard/K2 response by causal
future target value and smooth K4 changes exact acquisition variance as
follows:

* improves five setups materially or slightly;
* is essentially tied on two;
* is 4.7% worse on hallucination/Llama;
* gives a large 26.2% improvement on hallucination/Phi;
* but violates the no-CRC cost target on many splits, most clearly
  hallucination/Qwen with mean cost 21.97.

For LPB it lowers fixed-anchor/acquisition diagnostics on toxicity but does
not improve actual ten-split coverage variance on any nontrivial setup; it is
equal or slightly worse.  It is therefore not a universal replacement for
hazard/K2.  It remains a valuable CRC-wrapped research direction.

### Continuous Basis-DAPRO

Basis-DAPRO compresses the policy from roughly 400 time/bin coefficients to
16 and is 10--19 times faster.  It ties K2 on toxicity metric estimation but
is substantially worse on red-team.  It is a computational compression, not
a superior universal allocator.

### Selection-/Influence-DAPRO

The theoretically aligned uncentered selector influence is preferable to the
centered split influence, but the hard pilot had only about eight effective
rows and lost to soft Generalized.  It needs Rao--Blackwellized causal
influence masses and a larger independent curve-estimation sample before it
can be considered a default.

### Sequential augmented HT

Sequential AHT produces much larger variance reductions for metric
estimation, but it changes the estimator rather than merely selecting a DAPRO
coefficient.  It should be combined with soft Generalized allocation, not
treated as a competing DAPRO family.

## Recommendation

Use one common DAPRO formulation in both tasks:

\[
\boxed{\text{soft-prefix Generalized DAPRO, appropriate target A, hazard K2}}
\]

with:

* `A = 1{T<=200}` for the unsafe-event metric;
* `A = 1{T<f_alpha(X)}` for the LPB boundary;
* corrected shared-PAV CRC when a finite-sample marginal expected-budget
  guarantee is required;
* projection/no-CRC when lower variance is preferred and cost transfer is
  empirically trusted.

Keep Definitive as the essential model-robust ablation and fallback.  Keep
unregularized Target-A as a theoretical/oracle-style diagnostic, not the
default at `N1<=200`.  Do not replace the default by value/K4 or Basis-DAPRO
until a CRC-wrapped version wins actual cross-split task metrics consistently.

The strongest future candidate for even broader robustness is a cross-fitted
shrinkage coefficient

\[
a^{(\eta)}=(1-\eta)a^{\mathrm{hard}}+\eta a^{\mathrm{soft}},
\]

where `eta` is selected on policy-fit data independently of the CRC control
labels.  It nests Definitive-like hard fitting and soft Generalized fitting,
and could adapt to model misspecification.  It has not yet been validated here
and should not displace the empirically supported soft default.

## Reproducibility

The generated comparison tables and exact method rankings are under
`outputs/dapro_universal_comparison/`.  The summarizer is
`analysis/diagnostics/compare_dapro_variants.py`.  The metric registry exposes
the extra hard comparison methods only when
`--include-dapro-comparison` is passed; the default production method set is
unchanged.
