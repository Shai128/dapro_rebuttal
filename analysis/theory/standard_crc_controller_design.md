# Standard DAPRO CRC: validity, utilization, and design alternatives

## What is capped

For a complete trajectory with active length (L_i), the policy cost is

\[
c_i(P)=\sum_{t=1}^{L_i}\rho_i(t),\qquad
\rho_i(t)=\prod_{s=1}^t P_i(s).
\]

The canonical capped-CRC variants enforce (c_i(P)\le C_{\max}), with
(C_{\max}=\min\{t_{\max},2\bar B\}).  This is a cap on conditional expected
cost, not on realized turns.  A realized sampled trajectory may continue to
its full horizon.  Nevertheless, monotonicity implies

\[
\rho_i(q)\le C_{\max}/q,
\]

so the cap can lower the inclusion probability of late endpoint targets.
It is therefore a statistical/computational tradeoff, not an
efficiency-neutral technicality.

The current canonical CRC classes for LPB, metric estimation, and the current
v3 history-adaptive UPB method use the (2\bar B) cap.  Their no-CRC variants
do not.  The stored UPB `v2` result files instead record
`risk_budget_row_cost_cap_enabled=0`; poor performance in those old UPB
results cannot have been caused by the (2\bar B) cap.  In the stored
Toxicity/Qwen (B=20) runs where it was active, the cap changed about 29% of
LPB deployment policy rows and 57% of metric-estimation rows (aggregating the
available (N_1) settings and splits), so it is not innocuous in those tasks.

## Why the shape budget does not subtract realized CRC cost

After observing the policy-fit fold,

\[
B_{\rm rem}=B_{\rm tot}-\sum_{i\in I_1}L_i,
\qquad
\bar B_{\rm shape}=B_{\rm rem}/(n+m).
\]

The shape budget only constructs a frozen candidate family.  The final
budget guarantee comes from the composite CRC loss

\[
\ell_i(\lambda)=c_i(\lambda)+(n/m)b_i,
\qquad b_i=L_i.
\]

Using

\[
\frac{B_{\rm rem}-\sum_{i\in I_{\rm crc}}b_i}{m}
\]

to refit the family would expose CRC labels before the family is frozen and
invalidate the standard exchangeability proof.  It can be used only with an
additional independent control fold or with an aggressive residual-budget
theorem whose transformed-envelope condition is enforced.  The latter
condition failed empirically for metric and UPB tasks in the local audit.

## Exact contraction

Let \(\bar\rho_i(t)\) be the capped base cumulative reach and \(\epsilon\) the
minimum cumulative reach.  The nested family is

\[
\rho_i^\lambda(t)=\epsilon+lambda
  \{\bar\rho_i(t)-\epsilon\}.
\]

Thus

\[
P_i^\lambda(1)=\rho_i^\lambda(1),\qquad
P_i^\lambda(t)=
\frac{\epsilon+\lambda\{\bar\rho_i(t)-\epsilon\}}
     {\epsilon+\lambda\{\bar\rho_i(t-1)-\epsilon\}}.
\]

The conditional probabilities are ratios of affine functions of \(\lambda\),
whereas each row cost is exactly affine:

\[
c_i(\lambda)=\epsilon L_i+lambda
\left\{\sum_{t=1}^{L_i}\bar\rho_i(t)-\epsilon L_i\right\}.
\]

Consequently, the continuous CRC boundary can be solved in closed form.  The
current 401-point grid loses at most (1/400=0.0025) in the contraction scale,
so replacing it by the continuous solution improves utilization only
slightly.

## Validity versus exact utilization

The standard selector is

\[
\widehat\lambda=\sup\left\{\lambda:
\frac{\sum_{i\in I_{\rm crc}}\ell_i(\lambda)+L_{\max}}{n+1}
\le \frac{B_{\rm rem}}{m}\right\},
\qquad
L_{\max}=C_{\max}+(n/m)t_{\max}.
\]

It gives a marginal expected upper-budget guarantee under the usual frozen
family, exchangeability, boundedness, nesting, and feasibility conditions.
It cannot provide a distribution-free lower bound on budget use.  Even a
valid family containing only a very conservative policy can underuse the
budget arbitrarily.  The explicit support correction is

\[
\frac{L_{\max}}{n+1}
=\frac{C_{\max}}{n+1}
 +\frac{n}{m(n+1)}t_{\max},
\]

and can also make the selected policy conservative.  A continuous boundary
makes the empirical inequality tight when the solution is interior, but it
does not force the population deployment mean to equal the target.

## Most defensible improvements

1. **Use the continuous closed-form contraction.**  This preserves the exact
   proof and removes grid slack, but the gain will be small with (K=401).
2. **Select the cap and base-policy aggression using only the policy-fit
   fold.**  For example, compare
   (C_{\max}\in\{\bar B,2\bar B,4\bar B,t_{\max}\}) using a policy-fit
   variance proxy that includes the corresponding CRC correction, freeze the
   winner, and then apply the untouched independent CRC selector.  Conditional
   on the policy-fit fold, the selected cap and family are fixed, so the
   standard proof remains valid.  This directly addresses endpoint-heavy UPB
   and restricted-mean targets.
3. **Fit an overcomplete base family.**  The shape target is not part of the
   validity theorem.  A more aggressive base target, selected from the
   policy-fit fold and bounded by (C_{\max}), can prevent underutilization
   when the current selector repeatedly returns \(\lambda=1\).  CRC still
   performs the final contraction.
4. **Tune the CRC-fold size as a design parameter.**  Increasing (n) reduces
   the candidate-cost correction (C_{\max}/(n+1)), but fully observing more
   control rows consumes budget and leaves fewer policy-fit/deployment rows.
5. **Use a deterministic upper bound on the actual prior horizon.**  If the
   predictor is architecturally clipped to (Q_i\le Q_{\max}<t_{\max}), then
   (t_{\max}) in the pilot-cost envelope can be replaced by (Q_{\max}).
   An empirical maximum is not sufficient for a distribution-free claim.

Subtracting realized CRC cost and reusing the same fold for certification is
not recommended.  To do that cleanly, one needs a fourth role: policy fit,
cost pilot, independent CRC, and deployment.  That version is valid but spends
additional full labels and is unlikely to be the simplest primary method.

The recommended next ablation is therefore a policy-fit-selected
(C_{\max}\)/base-aggression sweep, followed by the same standard composite
CRC.  It preserves the clean theorem while testing the design choice most
likely to matter for UPB and restricted-mean metric estimation.
