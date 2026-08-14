# Endpoint/block versus history-adaptive DAPRO for UPBs

## Question

The production UPB allocator makes one randomized decision at the initial
prefix and, if selected, follows the trajectory through the executable block.
That is less adaptive than ordinary DAPRO, which can continue or stop at every
turn after observing the current prefix.  This audit asks whether the block
restriction loses efficiency and implements two genuinely history-adaptive
alternatives.

All results below use the same random calibration/test splits and the same
cached conditional-PMF predictions as the production UPB experiment.  The
full-budget oracle coverage agrees seed-for-seed with the production run, which
verifies split alignment.

## UPB target and the value 201

For a finite candidate upper bound `f_i` in `{1,...,200}`, the calibration
target is miscoverage

\[
  A_i(f_i)=\mathbf 1\{T_i>f_i\}.
\]

The value 201 denotes infinity/no event through the executable horizon.  Its
miscoverage is deterministically zero, so it needs no acquisition and is never
inverse-probability weighted.

Let

\[
  m_{it}(f)=\widehat{\Pr}(T_i>f\mid H_{it})
\]

be the frozen model prediction available after prefix `t`.  The method remains
design-valid even if these predictions are misspecified.

## Estimator 1: terminal-residual AHT and a block action

Draw `S_i ~ Bernoulli(p_i)` at `X_{i0}`.  If selected, follow the row until the
candidate is resolved; otherwise use only the initial prediction.  The row
contribution is

\[
  \widetilde A_i^{\rm block}(f)
  =m_{i0}(f)+\frac{S_i}{p_i}\{A_i(f)-m_{i0}(f)\}.
\]

Conditionally on the complete benchmark,

\[
 \mathbb E_S[\widetilde A_i^{\rm block}(f)]=A_i(f),\qquad
 \operatorname{Var}_S(\widetilde A_i^{\rm block}(f))
 =\{A_i(f)-m_{i0}(f)\}^2(p_i^{-1}-1).
\]

Under the fitted Bernoulli model, the soft risk is `m_i0(1-m_i0)`.  If `c_i`
is predicted reveal cost, the unrestricted one-stage Neyman rule is
`p_i proportional to sqrt(m_i0(1-m_i0)/c_i)`.  Production uses a deployable
median-K2 approximation to that score, a common budget scale, positivity, and
optionally CRC.  This is model-only because both its target risk and score are
computed from `X_i0`; it is not ordinary label-fitted DAPRO.

## Estimator 2: all-prefix sequential AHT

For a predictable turn-by-turn policy, let `R_it` indicate reaching update `t`
and let `rho_it` be its logged cumulative propensity.  Terminalize the model
path at the realized target, so that `m_iL=A_i`, and define

\[
 \widetilde A_i^{\rm seq}(f)
 =m_{i0}(f)+\sum_{t=1}^{L_i(f)}
   \frac{R_{it}}{\rho_{it}}\{m_{it}(f)-m_{i,t-1}(f)\}.
\]

For every frozen, possibly misspecified prediction path and every predictable
positive policy,

\[
 \mathbb E_R[\widetilde A_i^{\rm seq}(f)]=A_i(f).
\]

The exact conditional path variance is

\[
 \operatorname{Var}_R(\widetilde A_i^{\rm seq}(f))
 =\sum_{t=1}^{L_i(f)}
 \left(\rho_{it}^{-1}-\rho_{i,t-1}^{-1}\right)
 \{A_i(f)-m_{i,t-1}(f)\}^2,
 \qquad \rho_{i0}=1.
\]

If the prediction process is the true Doob martingale, its population
acquisition inflation is controlled by squared prediction updates
`E[(m_it-m_i,t-1)^2 | H_i,t-1]`.  This motivates the information-gain policy
below.  The terminal block estimator is recovered when the entire correction
is made in a single final update.

## The two history-adaptive competitors

Both competitors use the same median-K2 generalized-DAPRO optimizer, make a
random continuation decision at every reached turn, fit on fully observed
Phase-I trajectory labels, log the complete `N x 200` conditional-probability
matrix, and calibrate with the all-prefix estimator above.

1. **Dynamic endpoint mass.**  This is the direct conventional soft-prefix
   counterpart.  At the frozen finite UPB anchor it places soft survival mass
   `1-h_it` at the resolving endpoint and uses the current-prefix value score.

2. **Dynamic information gain.**  On Phase I it places mass
   `(m_it-m_i,t-1)^2` at each reached update, with the configured small global
   exploration component.  Its causal score is

   \[
      s_{it}=\sqrt{
        \frac{m_{it}(f_i)\{1-m_{it}(f_i)\}}
        {\widehat{\mathbb E}[\text{remaining turns to resolution}
          \mid H_{it}]}}
   \]

   and is split at the active Phase-I median into K=2 score bins at each turn.

The no-CRC versions use the existing fit-to-deployment projection-margin
controller.  Bisection enforces the fitted/model budget equation but provides
no finite-sample deployment-budget theorem.  The CRC versions use an
independent control half, a fixed nested contraction family, and the full
200-turn support bound.  They intentionally do not use the shared-PAV row cap.

## Why sequential adaptivity can lose

Sequential AHT can improve on a block action only when new prefixes provide
reliable information early enough to justify retaining the row.  It is not a
free improvement:

- terminal reach is a product `rho_iL=prod_t p_it`; moderate per-turn
  probabilities can create very small terminal propensities;
- every noisy or misspecified model update is separately inverse weighted;
- K2 policy fitting spends fully observed Phase-I rows and introduces policy
  estimation variation;
- CRC with the full 200-turn support is conservative at small control sizes.

In these caches, the initial prediction carries most of the useful UPB target
information.  Prefix updates are not strong enough to compensate for the long
propensity product.  The block policy keeps mean maximum weights near 5--32,
whereas the dynamic policies average roughly 125--165 at target coverage 70%.

## Fifty-split results at target coverage 70%

`Exact variance` is the mean exact conditional acquisition variance of the
coverage estimator, in percentage-point squared units.  `Split variance` is
the ordinary variance of the 50 reported coverage values and is noisier because
it also includes calibration/test composition.  `Oracle MSE` measures squared
deviation from the full-budget result on the same split.

### Toxicity/Qwen

| Method | Exact variance | Split variance | Oracle MSE | Expected cost |
|---|---:|---:|---:|---:|
| Static | 4.901 | 3.941 | 3.558 | 12.405 |
| Endpoint/block | **3.511** | 4.515 | **2.809** | 15.619 |
| Endpoint/block + CRC | **3.650** | 5.159 | 3.621 | 17.949 |
| Dynamic endpoint mass, N1=50 | 4.684 | 5.708 | 4.198 | 18.068 |
| Dynamic endpoint mass + CRC, N1=50 | 6.530 | 7.194 | 4.700 | 13.165 |
| Dynamic information gain, N1=100 | 3.425 | 5.756 | 4.665 | 18.305 |
| Dynamic information gain + CRC, N1=100 | 4.257 | **4.502** | **2.902** | 16.287 |
| Dynamic information gain, N1=200 | 4.077 | 6.166 | 4.105 | 18.773 |
| Dynamic information gain + CRC, N1=200 | 4.255 | 7.814 | 6.159 | 18.330 |
| Full-budget oracle | 0 | 1.409 | 0 | 124.053 |

The only exact-variance near-tie is dynamic information gain at N1=100 versus
the block method: paired difference -0.086 pp^2, bootstrap 95% interval
[-0.416, 0.252].  It is not a reliable improvement, it uses 2.69 more turns,
and its split-conditional expected cost exceeds 20 on 4/50 splits.

### Red-team/Qwen

| Method | Exact variance | Split variance | Oracle MSE | Expected cost |
|---|---:|---:|---:|---:|
| Static | 6.123 | 6.190 | 4.027 | 9.630 |
| Endpoint/block | **2.801** | 4.424 | **3.747** | 18.794 |
| Endpoint/block + CRC | **3.517** | **4.100** | **3.777** | 18.031 |
| Dynamic endpoint mass, N1=50 | 2.904 | 3.782 | 3.359 | 18.946 |
| Dynamic endpoint mass + CRC, N1=50 | 5.653 | 5.164 | 4.795 | 12.984 |
| Dynamic information gain, N1=100 | 3.544 | 4.511 | 4.138 | 19.130 |
| Dynamic information gain + CRC, N1=100 | 4.831 | 4.903 | 4.866 | 16.386 |
| Dynamic information gain, N1=200 | 3.979 | 5.310 | 4.683 | 19.095 |
| Dynamic information gain + CRC, N1=200 | 4.569 | 4.958 | 4.493 | 18.282 |
| Full-budget oracle | 0 | 1.259 | 0 | 96.300 |

Even the closest dynamic endpoint comparison is worse in exact variance:
+0.103 pp^2, paired bootstrap 95% interval [0.005, 0.205].

## Multi-target conclusion

Averaging exact conditional variance over target coverages 60%, 70%, 80%, and
90% gives:

| Method | Toxicity | Red-team |
|---|---:|---:|
| Endpoint/block | **2.674** | **2.281** |
| Endpoint/block + CRC | 2.812 | 2.864 |
| Dynamic endpoint mass | 3.693 | 3.296 |
| Dynamic endpoint mass + CRC | 5.065 | 5.590 |
| Dynamic information gain, N1=100 | 2.693 | 3.659 |
| Dynamic information gain + CRC, N1=100 | 3.449 | 4.605 |
| Dynamic information gain, N1=200 | 3.201 | 4.193 |
| Dynamic information gain + CRC, N1=200 | 3.473 | 4.583 |
| Static | 4.157 | 5.011 |

The recommended production UPB method therefore remains the endpoint/block
soft Generalized-DAPRO specialization.  The result is empirical rather than a
universal theorem: history adaptivity could win on a benchmark where prefix
updates deliver substantial early information.  The sequential-AHT code and
the two dynamic policies remain isolated experimental comparators, not members
of the production UPB registry.

## Reproduction

The experiment driver is
`analysis/diagnostics/upb_history_adaptive_comparison.py`.  Raw checkpointed
results are:

- `outputs/upb_dynamic_comparison_toxicity_n1_50.csv`
- `outputs/upb_dynamic_comparison_red_qwen_n1_50.csv`
- `outputs/upb_dynamic_comparison_toxicity_n1_100_info.csv`
- `outputs/upb_dynamic_comparison_red_qwen_n1_100_info.csv`
- `outputs/upb_dynamic_comparison_toxicity_n1_200_info.csv`
- `outputs/upb_dynamic_comparison_red_qwen_n1_200_info.csv`

The implementation tests are in `tests/test_upb_information_gain.py` and cover
finite-sample design unbiasedness by exact enumeration, the exact path-variance
identity, block-estimator telescoping, prefix causality, positivity, candidate
propensities, and end-to-end execution.
