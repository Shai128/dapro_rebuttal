# CRC-free budget control for soft-prefix hazard-K2 DAPRO

## Executive conclusion

Using two score bins removes the legacy oracle-to-regression projection error,
but it does **not**, by itself, turn the Phase-I empirical cost constraint into
a deployment-population expected-budget guarantee.  The remaining error is
the difference between the Phase-I and Phase-II distributions of score-bin
paths, prior horizons, and event-stopped lengths.  A counterexample already
exists for one bin, so no argument based only on `K=2` can close that gap.

There are, however, two ways to change the controller and obtain a genuine
CRC-free theorem without assuming a numerical projection-error bound.

1. **Static all-high K2 envelope.**  This retains independent rowwise
   acquisition and the current LPB sampling structure.  It is exact but, in
   the available experiments, uses only about half of the metric budget and
   materially worsens variance.
2. **Shared online predictable-compensator controller.**  This uses a common
   scalar logit adjustment among the currently active rows at each turn.  It
   gives exact conditional expected-budget control, almost completely uses the
   budget, and changes the learned K2 probabilities very little.  Metric HT
   validity and a design-unbiased variance estimator survive.  It induces
   cross-row dependence, so the manuscript's independent-row Bernstein proof
   does not transfer verbatim.  A finite-sample replacement follows from the
   preserved pairwise HT cross moments and Cantelli's inequality, under an
   explicit second-moment condition.  Its confidence dependence is looser.

The second controller is the promising algorithm.  It was the best aggregate
metric method in the initial four-setup comparison while satisfying the
budget on every tested split.  The first controller is the conservative
fallback if preserving independent calibration rows is non-negotiable.

## 1. Why K2 alone is not a budget proof

Let the Phase-I-fitted deployable table be

\[
  p_{t,0}\le p_{t,1}, \qquad t=1,\ldots,M,
\]

and let `b_i(t)` be the bin selected by the causal score of row `i` at turn
`t`.  Its cumulative reach is

\[
  \rho_i(t)=\prod_{s=1}^t p_{s,b_i(s)}.
\]

Direct K2 lookup means that there is no subsequent Platt or isotonic
regression.  Thus it eliminates one important source of error in Legacy
DAPRO.  Nevertheless, the raw optimizer enforces only

\[
  \frac{1}{n_1}\sum_{i\in\mathcal I_1}
  \sum_{t\le L_i}\rho_i(t)\le \bar B_2,
\]

on the rows used to learn the table.  The desired quantity is the analogous
cost under Phase-II score paths and lengths.  The cardinality of the table is
small, but its `2M` entries and the `M` learned cutpoints are data adaptive.
Nothing forces the Phase-II frequency of high-bin paths or long horizons to
equal its Phase-I frequency.  This distinction remains even for `K=1`.
Consequently, smaller projection error is empirical evidence for a smaller
budget gap, not a distribution-free expected-budget theorem.

## 2. Static all-high-path controller

Define

\[
  p_t^+=\max_{b\in\{0,1\}}p_{t,b}=p_{t,1},
  \qquad
  \rho^+(t)=\prod_{s=1}^t p_s^+.
\]

Every possible future K2 path obeys `rho_i(t) <= rho^+(t)`.  Let `g_gamma` be
the increasing cumulative-logit transform used by the current implementation,
including the terminal mixture floor `epsilon`:

\[
  g_\gamma(r)
  =\epsilon+(1-\epsilon)
    \operatorname{logit}^{-1}\{\operatorname{logit}(r)+\gamma\}.
\]

The deployment prior horizons `Q_i` are functions of `X_i0` and are known
before acquisition.  Select the largest scalar `gamma` satisfying

\[
  \sum_{i\in\mathcal I_2}\sum_{t=1}^{Q_i}
  g_\gamma\{\rho^+(t)\}\le B_2,
\]

where `B_2` is the total budget left after fully observing Phase I.  Apply the
same transform to each actual causal path.

For every fixed collection of complete trajectories,

\[
\begin{aligned}
 \mathbb E_R[C_2\mid\mathcal D_1,\mathcal Z]
 &=\sum_{i\in\mathcal I_2}\sum_{t=1}^{L_i}
   g_\gamma\{\rho_i(t)\}\\
 &\le\sum_{i\in\mathcal I_2}\sum_{t=1}^{Q_i}
   g_\gamma\{\rho^+(t)\}\le B_2.
\end{aligned}
\]

This result needs no iid assumption, correct hazard model, CRC split, or
projection-error bound.  It extends immediately to any finite `K` by taking
the maximum table entry at each time.

### X0-tightened version

The first K2 score is already known from `X_i0`.  Therefore a tighter valid
envelope is

\[
 \rho_i^{+,0}(t)=p_{1,b_i(1)}\prod_{s=2}^t p_s^+.
\]

The experiment called this `k2_x0_high`.  It was only slightly tighter for
metric estimation and identical for the tested LPB configurations, because
the first-bin information did not explain most of the worst-path slack.

## 3. Shared online predictable-compensator controller

Let `F_{t-1}` contain the Phase-I fit, all acquisition coins and observed
prefixes before turn `t`, and the active set `H_t`.  The raw K2 probabilities
`q_it` for active rows are `F_{t-1}`-predictable.  Freeze a positive time-spend
profile `w_1,...,w_M` from Phase I.  Initialize remaining credit `R_1=B_2`.
At time `t`, define

\[
 d_t=R_t\frac{w_t}{\sum_{s=t}^M w_s}.
\]

Choose one scalar `a_t` and set

\[
 p_{it}=\operatorname{logit}^{-1}
        \{\operatorname{logit}(q_{it})+a_t\},\qquad i\in H_t,
\]

so that

\[
 c_t:=\sum_{i\in H_t}p_{it}\le d_t.
\]

If `d_t >= |H_t|`, use `p_it=1` and carry the unused credit forward.  Otherwise
the scalar is found by bisection.  Update `R_{t+1}=R_t-c_t`.  Tiny positive
mass is included in every potentially active time of the frozen profile, so a
preterminal tranche is never exactly zero.  In exact arithmetic all
continuation probabilities are positive.

If `Z_it` is the indicator that active row `i` is acquired at turn `t`, then

\[
 \mathbb E[Z_{it}\mid\mathcal F_{t-1}]
 =\mathbf 1\{i\in H_t\}p_{it}.
\]

Hence

\[
 \mathbb E_R[C_2\mid\mathcal D_1,\mathcal Z]
 =\mathbb E_R\!\left[\sum_t c_t\mid\mathcal D_1,\mathcal Z\right]
 \le B_2.
\]

The guarantee is conditional on the complete benchmark trajectories and is
therefore stronger than an iid marginal guarantee.  It does not refer to a
projection error.  Importantly, stopped and event-resolved rows release
credit to the survivors, which is why this method avoids the conservatism of
the static all-high envelope.

### HT moments under the shared ledger

The common scalar creates cross-row dependence, but does not invalidate HT.
For a fixed target endpoint, let `R_i` be its observation indicator and let
`Pi_i` be the product of the sequential probabilities actually logged along
the observed path.  Then

\[
  W_i=R_i/\Pi_i
\]

is a terminal likelihood-ratio martingale and
`E_R(W_i | Z)=1`.  Same-round Bernoulli draws are conditionally independent.
The martingales for distinct rows therefore have zero cross-variation, so

\[
  \mathbb E_R(W_iW_j\mid\mathcal Z)=1,
  \qquad i\ne j,
\]

even though later probabilities depend on the joint past active set.  Thus
distinct rowwise HT contributions remain uncorrelated.

The frozen-policy formula `A_i(1/Pi_i-1)` cannot be evaluated from one
adaptive run because `Pi_i` is itself random.  The correct observed-only,
design-unbiased variance estimator is instead

\[
 \widehat V_R(\widehat\mu)
 =\frac{1}{n^2}\sum_i A_iR_i
   \frac{1-\Pi_i}{\Pi_i^2}.
\]

Indeed, its summand is `A_i(W_i^2-W_i)`, whose expectation is
`A_i Var_R(W_i)`.  This is the estimator that should be reported for the
online controller.

For LPB, every fixed candidate's HT miscoverage estimate remains unbiased.
The current independent-row MGF proof does not transfer verbatim, but row
independence is stronger than necessary.  Let

\[
 Y_{\tau i}=A_{\tau i}R_{\tau i}/\Pi_{\tau i}
\]

on Phase II and let `Y=A` on fully observed Phase I.  If
`E(Y_tau_i^2 | D1) <= vbar` for every candidate, the cross-moment identity
above gives zero covariance between distinct Phase-II HT rows and zero
covariance between the Phase-I and Phase-II sums.  Hence, for
`S_tau=sum_i Y_tau_i`,

\[
 \operatorname{Var}(S_\tau)
 \le N_1P_\tau(1-P_\tau)+N_2(\bar v-P_\tau^2)
 \le N\bar v.
\]

Cantelli's inequality and the same strict-initial-prefix selector then yield
the finite-sample coverage gap

\[
 \Delta_{\rm ledger}
 =\sqrt{\frac{(1-\delta)\bar v}{\delta N}}.
\]

With probability at least `1-delta`, the selected LPB has coverage at least
`1-alpha-Delta_ledger`.  This is a complete replacement theorem, but it is
weaker than the independent-row Bernstein result: its confidence dependence
is polynomial rather than logarithmic, and the online controller does not by
itself certify a useful numerical value of `vbar`.  The static all-high
controller preserves the original proof architecture.

## 4. Why independent per-row accounts are not the solution

An independent variant preassigns row credits `b_i` from `X_i0`, with
`sum_i b_i <= B_2`, and forces the predictable sum on every possible realized
row path to be at most `b_i`.  This preserves rowwise independence and proves
the budget, but it has an unavoidable long-horizon failure.

Along an all-continue history, suppose `sum_{t=1}^M p_it <= b_i`.  By AM--GM,

\[
  \Pi_i(M)=\prod_{t=1}^M p_{it}
  \le \left(\frac{b_i}{M}\right)^M.
\]

At `M=200` and `b_i=20`, this upper bound is `10^{-200}`.  Thus a pathwise
per-row account cannot simultaneously offer a modest budget and a useful
uniform terminal propensity.  X0-personalized credits can move this problem
between rows but cannot remove it under a fixed total budget.

The two-setup diagnostic confirmed the pathology.  With budget 20, the row
accounts used only 2.41 queries per sample for Toxicity metric estimation and
1.91 for Red Team; median minimum propensities were `1.37e-4` and `1.83e-3`,
and median maximum weights were 7,303 and 548.  The Toxicity estimate had an
MSE of 1,144 squared percentage points.  This variant should be rejected.

## 5. Experiments

The isolated diagnostic reused the production soft-prefix target masses,
hazard score, median K2 table, and exact DAPRO optimizer.  No production
registry was changed.  We used `N1=50`, four real setups, both metric and LPB
tasks, and five matched calibration--test splits.  Budgets were 20 for
Toxicity, Red Team, and AutoIF and 10 for Hallucination.  These are small
diagnostics rather than final paper-scale estimates.

### Metric estimation: aggregate over four setups

| Controller | Mean conditional expected cost | Valid-split rate | Mean across-split variance (pp^2) | Median minimum propensity | Median maximum weight |
|---|---:|---:|---:|---:|---:|
| Shared online compensator | 17.24 | 100% | **7.02** | 0.101 | 11.7 |
| Soft-prefix K2 + CRC | 15.62 | 85%* | 10.05 | 0.033 | 35.4 |
| Soft-prefix K2, current no-CRC margin | 17.11 | 70% | 10.39 | 0.073 | 15.5 |
| Static all-high envelope | 7.94 | 100% | 33.37 | 0.024 | 41.3 |
| X0 + all-high envelope | 8.20 | 100% | 34.42 | 0.025 | 40.4 |

`*` The CRC theorem is marginal over an independent control sample; it does
not require the exact expected cost of every realized benchmark split to be
below budget.  The displayed rate is the stronger fixed-split diagnostic and
should not be interpreted as a failure of the marginal CRC statement.

The online controller's mean absolute log-probability distortion was between
0.003 and 0.007 in the four metric setups.  It changed roughly 18--32% of the
active conditional probabilities; most changes were numerically very small.

### Metric estimation by setup

| Setup | Online variance | No-CRC variance | CRC variance | Online expected cost | Online min propensity (median) |
|---|---:|---:|---:|---:|---:|
| Toxicity | 7.24 | **6.70** | 16.31 | 19.90 | 0.056 |
| Red Team | 8.10 | 7.18 | **5.00** | 19.55 | 0.140 |
| Hallucination | **2.01** | 13.44 | 3.66 | 9.51 | 0.146 |
| AutoIF | **10.74** | 14.23 | 15.24 | 19.98 | 0.062 |

With only five splits, rankings within a setup are noisy.  The useful result
is that exact budget control did not require the severe performance loss of
the static envelope.

### Fifty acquisition repeats on fixed splits

We froze each of three fitted policies and repeated only acquisition 50 times
per split.  This is necessary for the cross-row adaptive design.

| Setup | Empirical within-split metric variance (pp^2) | Mean design-unbiased estimate (pp^2) | Within-split LPB coverage variance (pp^2) |
|---|---:|---:|---:|
| Toxicity | 6.58 | 6.97 | 0.000 |
| Red Team | 8.16 | 10.18 | 0.058 |
| AutoIF | 6.54 | not rerun with the final estimator column | 0.000 |

The observed-only variance estimator tracks the repeated-acquisition variance
at the expected Monte Carlo precision.  LPB candidate selection was extremely
stable to acquisition randomness in these fitted splits.

### LPB across-split behavior

Across the four setups, the online, current no-CRC, all-high, and X0-all-high
methods had the same mean five-split LPB coverage variance, 1.05 squared
percentage points.  CRC averaged 1.29.  This equality reflects identical
selected candidates in many splits, not a theoretical identity.  The online
controller was valid on every split and used 17.40 expected queries on
average; the all-high controller was also valid but used only 13.50.

## 6. Recommendation

1. Do not claim that median K2 makes the raw Phase-I empirical constraint a
   theorem.  It does not.
2. Add the shared online predictable-compensator controller as the serious
   CRC-free candidate.  It preserves the DAPRO score, bins, target objective,
   and optimized table; only a single causal scalar budget adjustment is added
   at each turn.
3. For metric estimation, its HT validity and variance theory are clean, and
   the current results justify a larger experiment.
4. For LPB, use the new pairwise-cross-moment/Cantelli theorem when its
   second-moment assumption is acceptable.  Retain CRC or the static envelope
   when the sharper independent-row Bernstein guarantee is required.  The
   empirical LPB results are promising, but the online theorem's gap is looser.
5. Reject independent per-row pathwise accounts and do not use the static
   all-high controller as the main method unless preserving row independence
   is worth roughly half the usable metric budget.

Reproducible code and outputs are in
`analysis/diagnostics/k2_budget_safe_experiment.py`,
`outputs/k2_budget_safe_main5`, `outputs/k2_online_reps50_v2`, and
`outputs/k2_row_accounts_main5`.
