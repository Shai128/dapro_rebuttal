# K2 cutpoint audit: is the median special?

## Scope

This is an isolated diagnostic of soft-prefix Generalized DAPRO.  Production
code was not changed.  The experiment keeps the hazard score, task-specific
soft objective, K2 table, positivity mixture, one-turn projection margin, and
cumulative budget correction fixed.  The sole intervention is to replace the
per-time empirical median by the empirical quantile

\[
q\in\{0.1,0.2,\ldots,0.9\}.
\]

The audit uses five common outer splits and common acquisition uniforms for
each of four real setups (toxicity/Qwen, red-team/Qwen,
hallucination/Qwen, and AutoIF/Qwen), at \(N_1=50\), \(B/n=20\), and
\(t_{\max}=200\).  There are 360 fitted policies.  The primary response is
the exact conditional Horvitz--Thompson target variance on Phase II.  For LPB
we additionally perform the complete weighted calibration and evaluate the
selected bound on the held-out test set.

## Why the production code uses a median

The median is a stability choice, not the solution of the DAPRO variance
objective.  If \(n_t\) Phase-I histories are active at time \(t\), an empirical
\(q\)-cut gives approximately \(n_tq\) low-bin and \(n_t(1-q)\) high-bin
observations.  The median maximizes

\[
\min\{n_tq,n_t(1-q)\},
\]

and hence maximizes the worst cell size.  This is particularly important for
\(N_1=50\), because \(n_t\) becomes small late in a trajectory.  It reduces
the sampling noise of both bin moments and fitted continuation probabilities.
It is also invariant to strictly monotone recalibrations of the score.

The median is not generally variance-optimal.  Consider a one-decision
version of the problem and a threshold \(c\).  Write

\[
A_b(c)=\E[a(S)\1\{B_c(S)=b\}],\qquad
D_b(c)=\E[d(S)\1\{B_c(S)=b\}],\qquad b\in\{0,1\}.
\]

For a fixed partition, the interior optimum satisfies

\[
p_b\ \propto\ \sqrt{A_b(c)/D_b(c)},
\]

and, after imposing a cost \(C\), its optimized objective is proportional to

\[
\frac{1}{C}
\left\{\sqrt{A_0(c)D_0(c)}+\sqrt{A_1(c)D_1(c)}\right\}^2.
\]

Thus the population-optimal threshold minimizes the expression in braces; it
need not have 50% mass on either side.  If the value-to-cost ratio
\(r(s)=a(s)/d(s)\) is continuous and monotone, an interior optimum obeys

\[
r(c)=\sqrt{r_0(c)r_1(c)},\qquad
r_b(c)=A_b(c)/D_b(c).
\]

The sequential DAPRO problem has suffix coupling, so the effective \(a\) and
\(d\) in this calculation depend on the other time decisions.  Nevertheless,
the calculation establishes the important point: balanced occupancy and
variance-optimal partitioning are distinct objectives.

## Fixed-cutpoint results

The table reports the matched-split ratio of exact target variance to the
median policy, averaged over the 20 setup/split pairs.  Values below one favor
the alternative.  Win/tie/loss counts exclude three LPB pairs for which both
variances were exactly zero.

| Task | Quantile | Mean variance ratio | Win/tie/loss vs median |
|---|---:|---:|---:|
| Metric | 0.1 | 1.0287 | 6/0/14 |
| Metric | 0.3 | 1.0044 | 9/0/11 |
| Metric | 0.4 | 1.0024 | 10/0/10 |
| Metric | **0.5** | **1.0000** | -- |
| Metric | 0.6 | **0.9980** | 11/0/9 |
| Metric | 0.7 | 0.9995 | 10/0/10 |
| Metric | 0.9 | 1.0153 | 6/0/14 |
| LPB | 0.1 | 1.0840 | 2/7/8 |
| LPB | 0.3 | 1.0331 | 5/7/5 |
| LPB | **0.5** | **1.0000** | -- |
| LPB | 0.6 | 0.9928 | 8/7/2 |
| LPB | 0.7 | **0.9835** | 9/6/2 |
| LPB | 0.8 | 0.9792 | 8/4/5 |
| LPB | 0.9 | **0.9728** | 7/5/5 |

For metric estimation, \(q=0.6\) improves the mean by only 0.20%, loses on
9 of 20 splits, and can be 1.24% worse on an individual split.  This is too
small and inconsistent to justify replacing the median.  The best setup-level
quantile is 0.5 for AutoIF, 0.8 for hallucination, 0.3 for red-team, and 0.7
for toxicity.  There is no universally best fixed quantile.

For LPB, upper quantiles are more promising because the target event is rare:
putting only the top 10--30% of hazard scores in the high bin can isolate the
valuable histories better than a 50/50 split.  However, \(q=0.9\)'s 2.72%
mean improvement is driven largely by toxicity.  It loses on five
nondegenerate splits and is as much as 4.19% worse on one split.  The more
stable \(q=0.7\) improves the mean by 1.65% and loses twice.  The setup-level
optima are 0.7 (AutoIF), 0.8 (hallucination), 0.8 (red-team), and 0.9
(toxicity).  The hallucination and red-team LPB anchor variances are essentially
zero, so their numerical improvements are practically irrelevant.

Across all 20 LPB outer splits, changing the cutpoint never changed the
selected LPB candidate or its test coverage under common randomness.  It only
changed acquisition variance and realized cost.  Consequently the observed
five-split coverage variance is identical for all nine cutpoints; five splits
are far too few to use that equality as a general claim.

## Choosing the quantile from Phase I

A natural attempt is finite-grid empirical risk minimization: fit all nine
policies and choose the \(q\) with the smallest Phase-I soft-prefix objective.
It did not work reliably here.

| Task | Mean selected q | Variance ratio to median | Variance ratio to deployment oracle | Mean within-split Spearman correlation, fit proxy vs exact variance |
|---|---:|---:|---:|---:|
| Metric | 0.475 | 1.0065 | 1.0296 | 0.102 |
| LPB | 0.375 | 1.0574 | 1.1105 | -0.006 |

For metric estimation, Phase-I ERM loses 0.65% to the median on average.  For
LPB it loses 5.74%, including a 19.1% mean loss on toxicity.  The fit proxy's
ranking has almost no out-of-sample association with the exact target
variance.  This is selection optimism: the same 50 trajectories determine
the threshold, bin probabilities, and reported training objective.  It does
not invalidate Horvitz--Thompson unbiasedness once the policy is frozen before
Phase II, but it gives poor policy selection and enlarges the class whose cost
must generalize.

The ex-post deployment oracle over the nine quantiles would reduce variance by
about 2.3% for metric estimation and 4.1% for LPB at the setup-aggregate level.
This shows that some cutpoint headroom exists, but also that the attainable
gain is modest and cannot be recovered by the current in-sample soft proxy.

## Cutpoints do not solve raw budget validity

K2 eliminates the old oracle-to-regression projection error, but it does not
eliminate the Phase-I-to-deployment cost gap.  At the median, expected total
cost for hallucination metric estimation averaged 23.49 and reached 30.34,
despite a target of 20; three of five splits exceeded the target.  Across all
nine cutpoints on that setup, mean cost lies between 23.29 and 23.57 and the
maximum lies between 29.57 and 30.41.  No quantile fixes the problem.

The same issue is milder but visible for LPB: the median reaches 20.23 on
hallucination and 21.04 on toxicity.  Therefore neither the median nor an
optimized quantile supplies an expected-budget proof for the raw empirical
K2 policy.

An exact pathwise/worst-prefix budget controller is compatible with any fixed
or Phase-I-learned cutpoint.  Its proof conditions on Phase I and dominates
every possible Phase-II bin path, so choosing \(q\) adaptively cannot break
the guarantee.  By contrast, if one tries to justify the raw policy by
empirical-to-population generalization, cutpoint selection makes the policy
class larger and the argument harder, not easier.

## Recommendation

1. Retain the median as the default for metric estimation.  The observed
   alternative gains are negligible and non-universal.
2. Treat a fixed global \(q=0.7\) LPB policy as a worthwhile ablation, not a
   replacement.  It is the best stability/efficiency compromise in this small
   study; \(q=0.9\) has a larger average gain but appreciably higher split
   risk.
3. Do not select \(q\) using the in-sample soft-prefix objective at
   \(N_1=50\).  If tuning is desired, use independent historical setups or a
   genuinely independent validation fold, impose minimum cell sizes at every
   relevant time, and then refit the K2 table after freezing \(q\).
4. Couple any adaptive cutpoint with the exact pathwise budget cap.  A
   cutpoint optimization changes statistical efficiency; it does not create
   budget validity.

Reproducible rows and summaries are in
`analysis/diagnostics/k2_cutpoint_summary/`; the driver is
`analysis/diagnostics/k2_cutpoint_audit.py`.
