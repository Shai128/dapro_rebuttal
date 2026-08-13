# Raw median-K2 DAPRO: what can and cannot be proved about budget validity

## Executive conclusion

Let a policy-fitting sample of size `n` be used both to form the per-time
median score cutpoints and to optimize the two continuation probabilities in
each time/bin cell.  Enforcing the expected-cost constraint on those same
trajectories does **not** imply a finite-sample conditional or marginal
expected-budget guarantee for a fresh trajectory.  Median binning removes the
separate oracle-to-regression projection used by Legacy DAPRO, but it does not
remove adaptive empirical-to-population transfer error for the full joint
trajectory `(L,S_1,...,S_M)`.

This is not merely a missing proof technique.  A one-step, continuous-score,
positive-propensity counterexample below shows that exact rank symmetry of the
sample median is insufficient.  Consequently, the raw current K2 method can
support:

1. an asymptotic budget theorem for fixed `M` and fixed `K=2`, via a uniform
   law of large numbers;
2. a finite-sample theorem after a *proved* complexity/stability reserve,
   although generic reserves are vacuous at `M=200`, `n<=200`; or
3. an exact finite-sample theorem after changing the controller--for example,
   a worst-score-path constraint, independent CRC, or a predictable online
   compensator.

The median is a sensible minimax default for cell occupancy, not a source of
budget validity.  Under an exact worst-path or CRC controller, the split
quantile can be optimized on the policy-fit data without endangering the
budget theorem.

## 1. The actual transfer problem

Write a complete latent trajectory as

```
Z = (L,Q,S(1),...,S(M), other target coefficients),    L <= Q <= M.
```

For cutpoints `c=(c_1,...,c_M)` and a monotone K2 table
`p_{t,0} <= p_{t,1}`, let

```
b_t(Z) = 1{S(t) >= c_t},
rho_pi(Z,t) = product_{s<=t} p_{s,b_s(Z)},
C_pi(Z) = sum_{t<=L} rho_pi(Z,t).
```

The current direct-bin solver produces a data-adaptive policy
`pi_D=(c_D,p_D)` and enforces

```
P_n C_{pi_D} <= b_fit
```

on the policy-fitting trajectories.  The needed deployment statement is

```
E_D P C_{pi_D} <= b_deploy
```

(marginal validity), or the stronger `P C_{pi_D} <= b_deploy` almost surely
conditional on `D`.  Direct bin lookup makes the fitted and deployed
*functional form* identical.  It does not make `P_n C` equal to `P C`.
The remaining gap includes:

- sampling error in active-length frequencies;
- sampling error in complete low/high prefix patterns;
- dependence of those patterns on event times and horizons;
- randomness of the median cutpoints; and
- adaptive selection of all `2M` table entries from the same paths used to
  evaluate the constraint.

The word "projection" in the legacy proof obscures this distinction.  In K2,
the within-fit oracle-to-regression residual is zero, while the relevant
policy-fit-to-deployment generalization error is not.

## 2. Rank symmetry of the median is not enough

### 2.1 What the order statistic really gives

Suppose `m=2r+1` active fitting scores at one time are iid from a continuous
distribution `F`, and use the order-statistic median
`c=S_(r+1)`.  With `high = 1{S >= c}`:

- the fitting high-bin fraction is `(r+1)/(2r+1)`;
- `F(c) ~ Beta(r+1,r+1)`; and
- for an independent active score, the *marginal* probability of the high bin
  is exactly `1/2`.

More generally, for threshold `S_(k)`, the fitting high fraction is
`(m-k+1)/m`, whereas the fresh high probability averaged over the threshold is
`(m-k+1)/(m+1)`.  Thus, if the two table values were fixed independently of
the quantile sample, the fitting bin average would be slightly conservative
for a monotone table.

This fact is only one-time and marginal.  It does not control a table chosen
adaptively from the same order statistics and concomitant target/cost labels,
nor products of bins across time.  Numpy's linearly interpolated median also
does not equal an order statistic when the active count is even.  In that
case the fresh high probability is distribution-dependent, lying between the
two adjacent order-statistic tail probabilities.  Ties require randomized
tie-breaking for the clean rank statement.

For clarity, there is a narrow positive result behind the intuition.  If
`p_L<=p_H` is fixed independently of the `m` scores used to form the exact
order-statistic threshold `S_(k)`, then

```
E[fitting bin-average - fresh bin-average]
 = (m-k+1)/[m(m+1)] * (p_H-p_L) >= 0.
```

Thus the self-including empirical rank average is conservative in that
one-step, externally fixed-table problem.  Current DAPRO violates the crucial
independence: both table entries are optimized from the same score paths and
their target/cost concomitants.  Sequential cost also contains interactions
among the bins through cumulative products, for which marginal rank balance
does not determine the joint path law.

### 2.2 A continuous-score, K2, positive-floor counterexample

The following counterexample rules out an exact marginal theorem based only
on median rank symmetry.

Take one acquisition decision and an odd fitting size `n=2r+1`.  Let
`S ~ Uniform(0,1)`, let `c` be the sample median, and put

```
w_H = (r+1)/(2r+1),    R(c) = P(S_new >= c | c) = 1-c.
```

Fix a floor `epsilon >= 0` and a target `b` satisfying

```
epsilon < b <= epsilon + w_H(1-epsilon).
```

Define the following data-adaptive monotone K2 table:

```
if R(c) <= w_H:
    p_L = p_H = b
else:
    p_L = epsilon
    p_H = epsilon + (b-epsilon)/w_H.
```

For every fitting sample, its empirical expected cost is exactly `b`.  In the
second branch,

```
(1-w_H)p_L + w_H p_H = b.
```

For a fresh score, however, its conditional population cost is `b` in the
first branch and

```
epsilon + R(c)(b-epsilon)/w_H > b
```

in the second.  The event `R(c)>w_H`, equivalently
`c < 1-w_H`, has strictly positive probability under the beta law of the
median.  Therefore

```
E_D P C_{pi_D} > b,
```

even though the score is continuous, the split is the exact sample median,
the table is monotone, every probability is at least `epsilon`, and the
empirical constraint holds with equality for every sample.

As a numerical check, with `epsilon=0.005` and `b=0.2`, direct integration
under the beta distribution of the median gives fresh expected costs
`0.20876`, `0.20669`, and `0.20497` for `n=51`, `101`, and `201`, respectively,
while the fitting cost remains exactly `0.2` in every sample.

This learner is deliberately simple; its purpose is to refute any theorem
whose only premises are iid sampling, median K2 binning, monotonicity,
positivity, and empirical feasibility.  The current DAPRO table is also
adaptive to target/cost concomitants, so rank symmetry cannot be invoked as
if its two values were independent of the quantile sample.

An even simpler sequential counterexample uses the one-bin subclass.  Let
`L=2` with probability `q` and `L=1` otherwise.  Set the second continuation
to one if no fitting path reaches time two and to zero otherwise.  Fitting
cost is always one, but fresh expected cost is
`1+q(1-q)^n`.  This isolates active-length and unseen-prefix error.

### 2.3 The exact one-step DAPRO objective can itself overrun

The preceding construction allows an arbitrary data-adaptive K2 table.  The
phenomenon also occurs for the KKT solution of DAPRO's inverse-reach
objective.  Take `n=5`, `S~Uniform(0,1)`, one acquisition decision, and target
mass

```
a(S)=1{S>1/5}.
```

Let `c=S_(3)` be the fitting median, so the low/high empirical cost masses are
`D_0=2/5` and `D_1=3/5`.  For a target cost `0<b<3/5`, solve the unsaturated
one-step DAPRO problem

```
minimize   A_0/p_0 + A_1/p_1
subject to (2/5)p_0+(3/5)p_1 <= b,   0<=p_0<=p_1<=1,
```

where `A_j` is the empirical target mass in bin `j`.  Use the natural uniform
budget-filling tie rule when all five target masses are zero.  The empirical
cost is `b` in every fitting sample.

Let `J` be the number of fitting scores at most `1/5`.  The six binomial
probabilities for `J=0,...,5` are

```
(1024,1280,640,160,20,1)/3125.
```

The numbers of target points in the low and high bins are, respectively,

```
(m_0(J),m_1(J))
 = (2,3),(1,3),(0,3),(0,2),(0,1),(0,0).
```

Since `A_0/D_0=m_0/2` and `A_1/D_1=m_1/3`, the nonsaturated KKT solution is

```
p_k = b sqrt(A_k/D_k)
      / [(2/5)sqrt(A_0/D_0)+(3/5)sqrt(A_1/D_1)].
```

If `A_0=0<A_1`, this is understood as the limiting solution
`p_0=0,p_1=5b/3`; a zero-mass objective term is defined to be zero.  If the
two ratios agree, the solution is `p_0=p_1=b`.

The conditional median means are

```
E[c | J=0,...,5] = (3/5,13/25,2/5,3/20,3/25,1/10).
```

For `J=0,1,2`, the median is the 3rd of 5, 2nd of 4, or 1st of 3 draws from
`Uniform(1/5,1)`.  For `J=3,4,5`, it is the 3rd of 3, 3rd of 4, or 3rd of 5
draws from `Uniform(0,1/5)`.  The displayed means therefore follow from
`E[U_(k)]=k/(m+1)` for `m` standard-uniform draws.

Conditional on `J`, the ratios of fresh population cost to `b` are

```
1, r_1, 1, 17/12, 22/15, 1,
```

where, writing `s=sqrt(1/2)`,

```
r_1 = (13s+12)/[5(2s+3)] = 0.9601886205... .
```

For example, when `J=1`, the bin value-to-cost ratios are `1/2` and `1`, so
`p_0/b=5s/(2s+3)` and `p_1/b=5/(2s+3)`.  Combining these with
`E[c|J=1]=13/25` gives `r_1`.  When `J=3`, only the high bin has target mass,
so its fresh cost ratio is
`(1-3/20)(5/3)=17/12`; the `J=4` calculation similarly gives
`(1-3/25)(5/3)=22/15`.

These values follow directly from the KKT rule
`p_j proportional to sqrt(A_j/D_j)` and the conditional means of the relevant
uniform order statistics.  Consequently,

```
E[fresh cost]/b
 = [1921+1280 r_1]/3125
 = 1.0080132589... > 1.
```

The strict inequality is fully algebraic.  It is equivalent to
`r_1>301/320`, which after cross-multiplication is equivalent to
`1/sqrt(2)>27/46`.

Thus even the exact DAPRO optimizer, not merely an adversarial K2 learner, can
match its median-bin fitting cost and overrun marginally.  Replacing `a` by
`(a+delta)/(1+delta)` and adding a small positive cumulative floor preserves
the strict inequality for all sufficiently small positive `delta` and floor
by continuity.  At the production values `delta=0.001`, floor `0.005`, and
`b=0.2`, including a common one-step logit correction back to empirical cost
`0.2`, direct evaluation of the six `J` cases gives fresh expected cost
`0.2012423238`.  Thus the production regularization and positivity machinery
do not rescue an exact rank-based theorem even in this one-step example.
The corrected low/high probabilities and fresh costs in the six cases are:

| `J` | probability | `E[c | J]` | corrected `p_L` | corrected `p_H` | fresh cost |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.32768 | 0.60 | 0.200000 | 0.200000 | 0.200000 |
| 1 | 0.40960 | 0.52 | 0.161059 | 0.225961 | 0.192212 |
| 2 | 0.20480 | 0.40 | 0.014969 | 0.323354 | 0.200000 |
| 3 | 0.05120 | 0.15 | 0.017152 | 0.321898 | 0.276186 |
| 4 | 0.00640 | 0.12 | 0.022004 | 0.318664 | 0.283065 |
| 5 | 0.00032 | 0.10 | 0.200000 | 0.200000 | 0.200000 |

## 3. Why the attached conformal/exchangeability proof cannot be reused raw

The valid single-parameter conformal argument has the following structure.
Conditional on an independently trained model/shape, there is a fixed nested
family of row losses `K_i(lambda)` that is right-continuous, non-increasing in
`lambda`, and bounded above by `M_K`.  The calibration selector is

```
hat lambda = inf{lambda:
    [sum_{i=1}^n K_i(lambda) + M_K]/(n+1) <= alpha}.
```

For an independent exchangeable row `n+1`, define the symmetric augmented
selector

```
tilde lambda = inf{lambda:
    sum_{i=1}^{n+1} K_i(lambda)/(n+1) <= alpha}.
```

Because `K_{n+1}(hat lambda) <= M_K`, `hat lambda` is feasible for the
augmented problem, so `tilde lambda <= hat lambda`.  Monotonicity gives
`K_{n+1}(hat lambda) <= K_{n+1}(tilde lambda)`.  The augmented selector is
permutation-symmetric, hence exchangeability gives

```
E K_{n+1}(tilde lambda)
 = E [sum_i K_i(tilde lambda)/(n+1)] <= alpha.
```

The proof is sound only because the *same fixed function* `K_i(lambda)` is
evaluated for every exchangeable row.  It applies to the attached local
single-parameter policy when its predictive model and lambda-indexed policy
family were trained independently of the calibration rows (subject also to
the stated transformed-loss envelope accounting for random pilot cost).

For current direct K2 DAPRO, both the median cutpoints and the `2M` probability
table are fitted on the same Phase-I paths.  If `g_D(lambda)` denotes that
shape, the losses are

```
K_i^D(lambda) = C(Z_i; g_D(lambda)) + pilot adjustment.
```

Appending the fresh row and fitting symmetrically produces `g_{D+}(lambda)`,
not `g_D(lambda)`.  If the proof keeps `g_D`, the new row is distinguished as
the unique row omitted from shape fitting, so the augmented selector is not
permutation-symmetric.  If the proof refits `g_{D+}`, the key monotonic
comparison fails: changing lambda is ordered within either family, but there
is no order between `g_D(hat lambda)` and `g_{D+}(tilde lambda)`.  The maximum
loss correction does not repair this broken symmetry.

Ways to make that conformal proof applicable are:

1. learn cutpoints and a nested K2 shape on an independent training set, then
   use all Phase-I rows only to select one scalar;
2. keep the current fit/control split (CRC); or
3. use cross-fitted CRC: fit a shape on fold A, control/deploy it on fold B's
   deployment stratum, and vice versa.  This reuses all pilot rows across
   policies but is still independent risk control, not a raw same-sample
   proof.

## 4. Audit of the legacy projection proof

The legacy theorem is valid only under assumptions that essentially posit the
missing population transfer equality.  They do not follow from K2.

1. The oracle assumption separately assumes a pathwise score-to-probability
   representation and a population budget constraint.  Empirical feasibility
   does not establish the second condition.
2. Treating `(S_i,P_i)` as iid is not coherent if `P_i` denotes the rowwise
   optimizer output: those probabilities are computed jointly from the whole
   Phase-I sample and from its random remaining budget.  They are coupled
   across rows.  The iid statement can only refer to a separately defined
   population oracle, in which case its budget feasibility is again an extra
   premise rather than a consequence of DAPRO.
3. The mixed-product cancellation assumption is much stronger than
   coordinate-wise unbiasedness or pairwise uncorrelated errors.  Every
   nonempty cross-time mixed moment must vanish conditional on the fitted
   sample.
4. Score coordinates, active lengths, and event times are intentionally
   dependent in a multi-turn trajectory.  Treating only marginal score laws
   omits the stopping mask in the budget functional.
5. A jointly optimized row policy need not admit the assumed coordinate-wise
   oracle representation.  Direct K2 does admit a lookup representation, but
   its population cost is still unknown.
6. Even for direct K2, the deployed conditional probability after the common
   cumulative-logit correction and positivity mixture is generally a
   function of the complete observed bin prefix: it is the ratio of two
   corrected cumulative reaches.  It is not a coordinate-wise map of the
   current score alone, as required by the legacy product-of-coordinate-errors
   expansion.
7. The displayed asymptotic inequality
   `|E product_j epsilon_j| <= product_j E|epsilon_j|` is false without
   independence.  Since all errors are bounded by one, the conclusion can be
   repaired using

   ```
   E product_{j in A}|epsilon_j| <= min_{k in A} E|epsilon_k|,
   ```

   but it then remains only an asymptotic consistency statement.

K2 makes consistency and generalization more plausible.  It does not imply
the finite-sample cancellation assumptions.

## 5. The strongest theorem for the unchanged raw K2 method

Let `C_2` contain all full-trajectory cost functions induced by monotone K2
tables and arbitrary per-time thresholds, with fixed finite horizon `M` and a
compact probability parameter space.  The class is a bounded finite-
dimensional VC-subgraph/Glivenko--Cantelli class.  Define its expected
one-sided gap

```
G_n = E sup_{pi in Pi_2} [(P-P_n) C_pi].
```

For *any* empirical K2 optimizer satisfying
`P_n C_{hat pi} <= b_n`, including adaptive medians and table values,

```
E P C_{hat pi} <= E b_n + G_n.
```

For fixed `M`, `G_n -> 0`; therefore raw K2 is asymptotically budget-valid per
deployment row (and a fixed positive fit margin eventually dominates the
gap).  This proof is cleaner than the legacy oracle/projection proof and needs
no oracle map, cross-time independence, or assumed numerical projection-error
bound.

With realized pilot cost `B_1`, `N_2=N-n`, and fitting target
`b_n=(N\bar B-B_1)/N_2-eta`, the same calculation gives

```
E[total cost] <= N bar B - N_2 eta + N_2 G_n.
```

Thus `eta>=G_n` is sufficient for an exact finite-sample marginal statement;
with `eta=0` one only gets an `N_2 G_n` allowance, and with any fixed positive
margin one gets eventual validity as `G_n->0`.  This is a proved learning-
complexity correction, not an assumed projection-error constant, but its
generic numerical value is the problem.

It is not a finite-sample `<= b` theorem.  Generic bounds scale with the
complexity of a sequential table having approximately `2M` probability
parameters, `M` thresholds, and products of depth up to `M`.  At `M=200` and
`n in {50,100,200}`, distribution-free VC/Rademacher reserves are at least of
the order of the full loss range and are practically vacuous.  Hard medians
and the constrained inverse-probability optimizer are also not uniformly
stable: changing one rare long path can create/remove a cell and move its
probability from the floor to one.

A strongly convex, smooth-gate, externally fixed-threshold redesign could
admit a proved uniform-stability reserve `beta_n` and hence
`E P C <= E P_n C + beta_n`.  That would be a genuine theorem rather than an
assumed projection bound, but it is a different algorithm and the worst-case
sequential Lipschitz constants make the reserve unattractive.  K2 alone does
not supply stability.

## 6. Exact no-CRC static guarantee: the worst-score-path constraint

For a monotone K2 table, the all-high future path dominates every possible
future score path.  If first bins and prompt horizons are known, define

```
rho_i^+(t) = p_{1,b_i(1)} product_{s=2}^t p_{s,1}.
```

After whatever isotone cumulative correction/floor map `Psi` is used, impose

```
sum_{i in phase II} sum_{t<=Q_i} Psi(rho_i^+)(t) <= B_remaining.
```

Then, conditional on the fitted sample and on every collection of complete
latent deployment trajectories,

```
E_randomization [total phase-II tokens] <= B_remaining.
```

The proof is pointwise: each realized cumulative reach is at most its all-high
counterpart and `L_i<=Q_i`.  It needs neither iid data, correct hazards, CRC,
nor a transfer-error bound.  In log probabilities the constraint is a convex
sum of exponentials, so it can be inserted directly into the Generalized
DAPRO optimization.

For a distribution-free guarantee conditional on the fitted data and with no
restrictions on future bin paths beyond monotonicity, this envelope is also
essentially necessary: the allowed latent configuration includes every row
remaining active to `Q_i` and taking the high bin at every future time.  A
controller that violates the all-high inequality cannot guarantee the budget
for that configuration.  Thus any less conservative static theorem must use
additional distributional information, an independent calibration device,
or information revealed online.

The necessity can be phrased as a no-free-lunch argument.  Fix any realized
fitting data `D` for which the selected policy has a latent path `z+` with
`C_{pi_D}(z+)>b`.  Form an iid discrete distribution that puts probability
`1-eta` on `z+` and small positive mass on every point appearing in `D`.
The exact fitting sample `D` then has positive probability, while for small
enough `eta` the conditional deployment cost of `pi_D` exceeds `b`.  Hence a
conditional guarantee uniform over iid laws forces a pointwise cost envelope;
for a monotone K2 table its maximizer is precisely the all-high, full-horizon
path (or the all-high future continuation conditional on an already revealed
first bin).

## 7. Exact no-CRC adaptive guarantee: a predictable compensator

There is a second exact route that can preserve the raw policy until an
overrun becomes imminent, but it changes deployment and introduces dependence
between rows.

Index all offered token acquisitions in a predictable scheduling order by
`k`.  Let `F_{k-1}` contain all information observed before decision `k`, let
`A_k` indicate that the decision is currently available, and choose a
predictable continuation probability `p_k`.  Draw

```
X_k = A_k Z_k,    Z_k | F_{k-1} ~ Bernoulli(p_k).
```

If the controller enforces, on every realized history,

```
sum_k A_k p_k <= B_remaining,
```

then

```
E sum_k X_k = E sum_k A_k p_k <= B_remaining.
```

A simple guard starts with a ledger `R_0=B_remaining` and replaces each raw
K2 probability `q_k` by `p_k=min(q_k,R_{k-1})`, updating
`R_k=R_{k-1}-A_k p_k`.  Batch/round versions use a common water-filling scale
so that the sum of the active raw probabilities fits the remaining ledger.

Let `D_k=X_k-A_k p_k`.  This is a martingale difference with increments at
most one and predictable quadratic variation

```
V <= sum_k A_k p_k <= B_remaining.
```

Freedman's inequality consequently gives, for `x=log(1/delta)`,

```
P{sum_k X_k > B_remaining + sqrt(2 B_remaining x) + x/3} <= delta
```

(or the equivalent Bernstein-root form, depending on the chosen Freedman
version).

Sequential HT unbiasedness survives cross-row adaptation if every required
probability is strictly positive and predictable.  For a fixed latent target
`Y_i`, repeated tower conditioning gives

```
E[ Y_i product_t Z_it/p_it | complete latent data ] = Y_i.
```

An arbitrary hard ledger can force later probabilities to zero.  The tested
controller avoids this by freezing a strictly positive time-spend profile and
allocating a positive tranche at every still-possible preterminal time.  A
positivity reserve remains necessary in any implementation.

The shared ledger makes row estimators dependent, but conditional
independence of same-turn coins implies a stronger useful identity.  If
`W_i=R_i/Pi_i` and `W_j=R_j/Pi_j` are terminal rowwise likelihood-ratio
martingales, then

```
E(W_i W_j | complete latent trajectories) = 1,  i != j.
```

Thus distinct fixed-target HT contributions are uncorrelated even though
they are not independent.  The one-run design-unbiased acquisition-variance
estimator is `sum_i A_i R_i(1-Pi_i)/Pi_i^2 / N^2`.

The original LPB product-MGF proof still cannot be reused verbatim.  However,
the pairwise identity gives a finite-sample Cantelli replacement.  If
`E(Y_tau_i^2 | D1) <= vbar` for every candidate, then
`Var(sum_i Y_tau_i) <= N vbar`, and the same strict-prefix selector has
coverage gap

```
Delta_ledger = sqrt((1-delta) vbar / (delta N)).
```

This is looser in `delta` than the independent-row Bernstein theorem and
requires an explicit second-moment condition.  The worst-path controller is
still the cleaner option when the original sharper LPB proof architecture is
required.

## 8. Why the median, and how to choose another quantile

The median maximizes the smaller expected cell count:

```
min(n q, n(1-q))
```

over split quantiles `q`.  It is therefore the minimax choice when no prior
information says where score discrimination lies.  It reduces empty cells,
table sensitivity, and the variance of cellwise coefficient estimates.  None
of these facts is a budget theorem.

For a one-stage unsaturated approximation, let

```
A_b(c) = E[a(S) 1{S in bin b(c)}],
D_b(c) = E[d(S) 1{S in bin b(c)}].
```

Minimizing `sum_b A_b/p_b` subject to `sum_b D_b p_b <= B` gives

```
p_b proportional to sqrt(A_b/D_b),
J*(c) = [sqrt(A_0 D_0)+sqrt(A_1 D_1)]^2/B
```

before clipping at one.  Hence a task-optimal threshold minimizes
`sqrt(A_0D_0)+sqrt(A_1D_1)`, not necessarily the row median.  If rare target
mass is concentrated in the upper score tail, a 0.75 or 0.9 cut can improve
discrimination, at the price of a smaller and less stable high cell.

For the sequential method, use the actual Generalized-DAPRO objective only on
independent historical/tuning data.  The four-setup experiment found that
choosing the quantile on the same `N1=50` policy-fit sample was worse than the
median: exact variance increased by 0.65% for metric estimation and 5.74% for
LPB, while the within-split rank correlations between fit objective and
deployment variance were only 0.102 and -0.006.  Thus:

1. use the median as the metric default;
2. optionally predeclare `q=0.7` as an LPB ablation;
3. if adapting `q`, use independent historical data or a separate tuning fold
   and enforce minimum active-cell sizes;
4. fit every candidate under the same final budget controller; and
5. validate target variance across outer splits.

With CRC, choose the quantile only on the policy-fit fold before freezing the
candidate family.  With the worst-path controller, even a data-adaptive
quantile choice cannot invalidate budget control because the final selected
table is checked pointwise.  With raw empirical cost matching, searching over
quantiles increases adaptivity and weakens the already unavailable transfer
guarantee; in that case the median is the safest pre-specified default.

Under a worst-path controller, high quantiles can also be counterproductive:
the small high-risk cell may receive probabilities near one, while the robust
constraint charges every unseen future prefix as high.  A lower threshold can
make the high-bin table closer to the population average and substantially
tighten the robust envelope.  The split must therefore be optimized jointly
with the chosen controller rather than in isolation.

## 9. Recommended claims and comparisons

The defensible manuscript claims are:

- **Raw median-K2:** no separate regression projection; empirically stable;
  asymptotically budget-valid for fixed horizon under iid sampling and a
  uniform law of large numbers.  Do not claim an exact finite-sample budget.
- **Worst-path K2:** finite-sample, distribution-free conditional expected
  budget validity without CRC or transfer assumptions.  It may be
  conservative.
- **K2+CRC:** finite-sample marginal expected-budget validity and typically
  less conservative because it calibrates average rather than worst-path
  cost.
- **Online-guard K2:** exact conditional expected budget, a Freedman
  realized-cost bound, positive predictable HT, and pairwise cross-moment
  preservation.  The LPB theorem uses the looser Cantelli gap above under a
  second-moment condition.

The key empirical comparison should hold the soft-prefix target, score,
positivity floor, and total budget fixed while varying only:

1. raw K2 plus the current empirical margin;
2. K2 with the all-high robust constraint;
3. K2+CRC; and optionally
4. K2 with a predictable online guard.

For each, report conditional expected total cost (not only realized cost), the
fraction of splits over budget, mean/maximum gap, target variance, LPB
coverage variance, and the fraction of the nominal budget actually used.
