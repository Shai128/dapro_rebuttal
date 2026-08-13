# Selection-, Influence-, and Residual-DAPRO

This note separates the statistical target, the coefficient estimator, the
policy representation, and the budget controller.  All statements concerning
acquisition variance condition on the complete benchmark trajectories and on
the policy-fitting data unless stated otherwise.

## 1. Master acquisition problem

For trajectory `i` and active prefix `t`, let `rho_i(t)` be its cumulative
reach probability.  Associate a nonnegative squared-influence or event mass
`a_i(t)` and a nonnegative at-risk cost mass `d_i(t)`.  The common design
problem is

\[
 \min_{\rho\in\mathcal P}
 \frac1n\sum_{i,t}a_i(t)\{\rho_i(t)^{-1}-1\},
 \qquad
 \frac1n\sum_{i,t}d_i(t)\rho_i(t)\leq \bar B .
\]

The target determines `a`; a hard endpoint, conditional event model, or
martingale model estimates it; the score and basis determine
`mathcal P`; and projection, model-budget bisection, or CRC controls cost.
These choices are logically orthogonal.

If fixed prefix features satisfy

\[
 \log\rho_i(t)=x_i(t)^\top\theta,
\]

then the objective is a sum of exponentials of affine functions and the
budget sublevel set is convex.  With a convex penalty `Omega`, the Lagrangian
Hessian is

\[
 \sum_{i,t}
 \{a_i(t)e^{-x_i(t)^\top\theta}
 +\lambda d_i(t)e^{x_i(t)^\top\theta}\}
 x_i(t)x_i(t)^\top+\nabla^2\Omega(\theta)\succeq0.
\]

Linear sign and probability constraints preserve convexity.  Strict
convexity requires feature rank or a strongly convex penalty.  This result
does not cover jointly selecting knots, cutpoints, scores, or influence
weights inside the optimization.

## 2. Exact fixed-candidate covariance

Let `f_0 <= ... <= f_J` be nested LPB candidates and

\[
 A_{ij}=\mathbf 1\{T_i<f_j(X_i)\}.
\]

Let `Z_i` say that the common event endpoint needed by all candidates is
resolved, with probability `pi_i`.  The HT calibration rate is

\[
 M_j=\frac1n\sum_i \frac{Z_iA_{ij}}{\pi_i}.
\]

For a frozen predictable policy and row-independent acquisition draws,

\[
 \operatorname{Cov}(M_j,M_k)
 =\frac1{n^2}\sum_i A_{i,\min(j,k)}
 \left(\frac1{\pi_i}-1\right).
\]

Thus Target-A's objective is exact for one fixed candidate.  It is a proxy
only for the candidate-selecting LPB pipeline.

Define event bands by the first candidate containing the event.  The
increments `M_j-M_{j-1}` use disjoint rows and are therefore exactly
independent.  Their Gaussian limit is a time-changed Brownian motion, but
independence itself is finite-sample and does not require a CLT.

## 3. Exact variance of hard-selected LPB coverage

Use the strict-prefix selector

\[
 K=\max\{j:M_j<\alpha\},
\]

with explicit end-point sentinels.  Let `c_j` be the fixed complete-test
coverage of candidate `j`, and set `d_j=c_j-c_{j-1}`.  Since `M_j` is
pathwise nondecreasing,

\[
 Y:=c_K=c_0+\sum_{j=1}^Jd_j I_j,
 \qquad I_j=\mathbf 1\{M_j<\alpha\},
\]

and `I_k <= I_j` whenever `j < k`.  Put

\[
 F_j=\Pr_R(M_j<\alpha).
\]

Then

\[
 \mathbb E_RY=c_0+\sum_jd_jF_j
\]

and, exactly,

\[
 \operatorname{Var}_R(Y)
 =\sum_jd_j^2F_j(1-F_j)
 +2\sum_{j<k}d_jd_kF_k(1-F_j).
\]

The identity follows from

\[
 \operatorname{Cov}(I_j,I_k)
 =F_{\max(j,k)}-F_jF_k.
\]

Consequently, the nonlinear variance requires only the marginal crossing
probabilities, not their full joint law.  Each `F_j` is the CDF at `alpha` of
a weighted Poisson-binomial sum.  Common-random-number Bernoulli simulation
is exact up to Monte Carlo error.  A Gaussian approximation uses

\[
 F_j\approx\Phi\!\left(
 \frac{\alpha-m_j}{\sqrt{V_j}}
 \right),\qquad
 V_j=\frac1{n^2}\sum_iA_{ij}(\pi_i^{-1}-1).
\]

If `H(F;c)` denotes the displayed variance and
`mu=c_0+sum_j d_j F_j`, then

\[
 \frac{\partial H}{\partial F_j}
 =(c_j-c_{j-1})(c_j+c_{j-1}-2\mu).
\]

For the Gaussian approximation, with
`z_j=(alpha-m_j)/sqrt(V_j)`,

\[
 \frac{\partial F_j}{\partial V_j}
 =-\frac{z_j\varphi(z_j)}{2V_j}.
\]

These derivatives permit direct optimization of the Gaussian crossing
functional, but it is generally nonconvex.  In particular, its local weights
for `V_j` can be negative because increasing noise can move a crossing
probability either toward or away from one half.  Such signed weights must not
be passed to the nonnegative DAPRO solver.  The pivotal or margin bounds below
are safer optimization targets.

A useful Gaussian diagnostic is the Berry--Esseen ratio

\[
 \mathrm{BE}_j=
 \frac{
  n^{-3}\sum_i A_{ij}
  \left\{(1-\pi_i)^3/\pi_i^2+(1-\pi_i)\right\}
 }{V_j^{3/2}}.
\]

Up to the universal Berry--Esseen constant, this is the standardized sum of
third absolute centered moments.  Very small propensities can make it large;
in that regime exact Bernoulli crossing simulation should replace the
Brownian approximation.

If test trajectories are random and independent rather than conditioned,
the law of total variance additionally contributes

\[
 \mathbb E_K\{c_K(1-c_K)/n_{\rm test}\}.
\]

Random calibration/test splits and policy fitting add their own between-split
term.  A fixed-selected-candidate HT diagnostic is not the nonlinear variance
above.

## 4. Pivotal Selection-DAPRO

For any square-integrable function `Y(Z_1,...,Z_n)` of independent Bernoulli
acquisitions, write `Y_i(z)` for the output after setting coordinate `i` to
`z`.  The first-order Hoeffding decomposition and Efron--Stein inequality give

\[
 \sum_i\pi_i(1-\pi_i)
 \{\mathbb E_{-i}[Y_i(1)-Y_i(0)]\}^2
 \leq\operatorname{Var}(Y)
\]

and

\[
 \operatorname{Var}(Y)
 \leq\sum_i\pi_i(1-\pi_i)
 \mathbb E_{-i}[\{Y_i(1)-Y_i(0)\}^2].
\]

For an event in band `k_i`, toggling its acquisition adds
`1/(n pi_i)` to every `M_j` with `j >= k_i`.  Its finite-difference effect is
therefore obtained by applying the exact selector to a perturbed crossing
path.

At a reference policy `pi_i^0`, estimate

\[
 q_i^0=\mathbb E_{-i}
 [\{Y_i(1)-Y_i(0)\}^2]
\]

by exact Bernoulli or independent-increment Gaussian simulation and freeze

\[
 w_i^0=(n\pi_i^0)^2q_i^0.
\]

Then

\[
 \frac1{n^2}w_i^0(1/\pi_i^0-1)
 =\pi_i^0(1-\pi_i^0)q_i^0,
\]

so an ordinary weighted DAPRO fit locally matches the Efron--Stein upper-bound
term.  One optional fixed-point update is reasonable; repeated unrestricted
iteration has no global guarantee.

If event time `t` belongs to band `k(t)`, a causal soft implementation uses

\[
 a_i(t)=h_i(t)w_{k(t)}^0,
\]

on active policy-fit prefixes.  All crossing weights, candidate gaps, and
reference propensities must be frozen from training/policy-fit information
before control or deployment outcomes are inspected.

For differentiable `g(M)`, this construction has the local limit

\[
 \phi_i=\nabla g(m)^\top A_i,
 \qquad w_i=\phi_i^2,
\]

which is Influence-DAPRO.  One-hot influence recovers Target-A.

## 5. Analytic margin-crossing weights

Pivotal simulation is unnecessary when every complete-data candidate rate is
separated from the target.  Let

\[
 I_j^*=\mathbf1\{m_j<\alpha\},\qquad
 \delta_j=|\alpha-m_j|>0,
 \qquad D_c=\sum_j|d_j|.
\]

Weighted Cauchy--Schwarz and one-sided Cantelli imply

\[
 \operatorname{Var}(Y)
 \leq\mathbb E[(Y-c_{K^*})^2]
 \leq D_c\sum_j|d_j|
 \frac{V_j}{V_j+\delta_j^2}
 \leq D_c\sum_j|d_j|\frac{V_j}{\delta_j^2}.
\]

The final expression is a standard DAPRO objective with row weight

\[
 w_i^{\rm margin}
 =\sum_j\frac{|d_j|A_{ij}}{\delta_j^2}.
\]

Equivalently, an event at time `t` receives
`w(t)=sum_{j:t<f_j}|d_j|/delta_j^2`; its soft prefix mass is `h_i(t)w(t)`.

For matched calibration and test populations, `c_j=1-m_j`, so
`|d_j|=m_j-m_{j-1}`.  With estimated rates, simultaneous pilot bounds give a
robust lower margin

\[
 \delta_j^L=(|\alpha-\widehat m_j|-r_j)_+.
\]

A zero robust margin correctly signals a nonregular boundary and should be
handled with pivotal simulation rather than an artificial theorem.

## 6. Soft, score, quantization, and controller error

Let `L_eps=eps^{-1}-1`.  If `rho_hat` minimizes the objective formed with
`a_hat` in the same feasible class as the true minimizer, then

\[
 J_a(\rho_{\widehat a})-\inf_{\rho\in\mathcal P}J_a(\rho)
 \leq 2L_\epsilon\|a-\widehat a\|_1/n,
\]

apart from optimization and regularization error.  Correct conditional soft
masses Rao--Blackwellize hard endpoints; misspecified hazards pay this bias
bound.  If `||d-d_hat||_1/n <= eta`, fitting to `B-eta` ensures true cost at
most `B`; numerical bisection against `d_hat` alone does not.

In the one-stage unclipped problem, policies measurable with respect to a
sigma-field `G` attain

\[
 J_G^*=\frac{
 [\mathbb E\sqrt{\mathbb E(a\mid G)\mathbb E(d\mid G)}]^2
 }{B}.
\]

Conditional Cauchy--Schwarz shows that this is no smaller than the
full-information optimum.  Equality requires `a/d` to be constant conditional
on `G`.  This is an exact score-sufficiency criterion.  A monotone transform
of a score leaves quantile bins unchanged and cannot restore lost ranking
information.

For score-partition cells `B`, define cell masses `A_B,D_B`.  Then

\[
 J_{\rm bins}^*=\frac{(\sum_B\sqrt{A_BD_B})^2}{B}.
\]

Partition refinement weakly improves the population optimum.  If
`log(a/d)` has range at most `R_B` in a cell, the Kantorovich inequality gives

\[
 \frac{J_{\rm bins}^*}{J_{\rm fine}^*}
 \leq\max_B\cosh^2(R_B/4).
\]

The bound becomes vacuous for sparse or zero event mass, exactly where a
forced median split can mix a rare high-value group with many zero-value
rows.

For CRC contraction

\[
 \rho_\gamma=\epsilon+\gamma(\rho-\epsilon),
\]

`rho_gamma >= gamma rho`, and therefore

\[
 J_a(\rho_\gamma)
 \leq\gamma^{-1}J_a(\rho)
 +(\gamma^{-1}-1)\sum a/n.
\]

The exact controller distortion is

\[
 \sum a(1/\rho_\gamma-1/\rho)/n.
\]

CRC validity does not imply this distortion is small.

## 7. Value score and deployable representations

The instantaneous hazard is the exact ordering only for a one-step
event-mass/unit-cost decision.  A useful causal approximation to the
downstream coordinate value is

\[
 v_i(t)=\log r_i(t)-\log c_i(t),
\]

where `r_i(t)` is remaining squared target-influence probability and `c_i(t)`
is expected remaining reached cost under the current prefix.  A block
reveal/stop problem has the exact solution `q proportional sqrt(r/c)`.
For a fully revisable history-adaptive policy this is a principled
approximation, not a universal sufficiency theorem.

### Continuous monotone basis

Rank-transform `v` to `u in [0,1]`, use monotone score bases and a small
time basis, and fit the resulting fixed-feature convex program.  Temporal
pooling permits 10--20 effective parameters rather than a separate map at
every turn.

### Adaptive optimal bins

During a DAPRO coordinate update, a tied contiguous score cell has loss

\[
 \alpha_Be^{-y}+\beta_Be^y,
 \qquad
 y_B^*=\operatorname{clip}
 \left\{\frac12\log(\alpha_B/\beta_B),\ell_B,0\right\}.
\]

Reduced-isotonic dynamic programming finds the exact best `K` contiguous
monotone cells for that coordinate.  This improves fitted loss over a forced
median split, but alternating cutpoint updates make the global multi-time
problem nonconvex.  Small fitted samples can make cutpoints unstable.

The safest progression for CRC policy-fit sizes 25, 50, and 100 is:

1. repair the causal row-cost cap;
2. test an optimal distinct-value two-cell split with minimum cell size;
3. test a strongly regularized 10--16 parameter linear-rank/time-basis model;
4. add two or three score knots only at the largest fit size.

## 8. Sequential augmented HT for metric estimation

Let `A_i=1{T_i<=M}`.  Along a complete trajectory define any predictable
sequence `m_i0,...,m_iL` with terminal value `m_iL=A_i`, and increments
`Delta_it=m_it-m_i,t-1`.  With reach indicator `R_it` and cumulative
propensity `rho_it`, define

\[
 \widetilde A_i=m_{i0}
 +\sum_{t=1}^{L_i}\frac{R_i(t)}{\rho_i(t)}\Delta_i(t).
\]

It is design-unbiased for `A_i` for any frozen model.  Its exact conditional
variance is

\[
 \operatorname{Var}_R(\widetilde A_i)
 =\sum_{u=1}^{L_i}
 \left\{\rho_i(u)^{-1}-\rho_i(u-1)^{-1}\right\}
 \{A_i-m_i(u-1)\}^2,
\]

where `rho_i(0)=1`.  Consequently,

\[
 \operatorname{Var}_R(\widetilde A_i)
 \leq\sum_u\{A_i-m_i(u-1)\}^2
 \{\rho_i(u)^{-1}-1\},
\]

which is a conservative master objective without model correctness.

If `m_it` is the true Doob martingale, its increments are orthogonal and the
population acquisition inflation is exactly

\[
 \sum_t\mathbb E[\Delta_i(t)^2
 \{\rho_i(t)^{-1}-1\}].
\]

Thus the efficient soft mass is expected squared prediction update, or
information gain, rather than the instantaneous event hazard.  A
misspecified model leaves design unbiasedness intact but loses this
population variance identity.

## 9. Causal budget cap

A cap coefficient computed from an evaluation row's entire future score path
is not predictable.  A causal alternative freezes a nonincreasing envelope

\[
 e_1\geq\cdots\geq e_M\geq\epsilon,
 \qquad\sum_te_t\leq C_{\max},
\]

using fit or initial-only information and deploys

\[
 \rho_i^{\rm cap}(t)=\min\{\rho_i^{\rm base}(t),e_t\}.
\]

This is prefix-measurable, preserves temporal monotonicity and positivity,
and supplies the pathwise cost support bound required by CRC.

## 10. What is and is not guaranteed

Logged predictable propensities and positivity give fixed-target HT or AHT
design validity even under coefficient-model misspecification.  Independent
CRC controls and a frozen nested bounded family give marginal expected-total
budget control.  They do not give realized-budget, per-split budget, or
variance superiority guarantees.

Influence-DAPRO is first-order exact only for a smooth downstream functional.
Pivotal Selection-DAPRO targets an Efron--Stein upper bound for the hard
selector.  Margin Selection-DAPRO targets a concentration upper bound.
Neither makes the selected LPB estimator unbiased.  Richer score maps improve
the population optimum in nested classes but may increase finite-fit policy
regret.

The identities and bounds in Sections 3--5 have an executable NumPy audit in
`analysis/diagnostics/lpb_selection_variance.py`, with regression tests in
`tests/test_lpb_selection_variance_diagnostic.py`.
