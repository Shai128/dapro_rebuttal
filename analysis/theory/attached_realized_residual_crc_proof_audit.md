# Audit of the attached transformed-loss CRC proof

## Verdict

The conformal argument is correct **after** two substantive repairs:

1. the nested candidate family must be learned on a separate policy-fit fold
   and then frozen before the CRC fold is observed; and
2. the transformed envelope must hold uniformly for a fresh row and every
   policy that the CRC selector can return.

As written, the proposition uses the same `I_1` both as the DAPRO training
sample and as the exchangeable CRC sample.  For label-fitted DAPRO this makes
the augmented selector non-symmetric: the original rows helped learn the
candidate family whereas the augmented deployment row did not.  Conditioning
only on the random split does not fix this.  The production three-way split
does: condition on the independent policy-fit fold, call the CRC size (n),
the deployment size (m), and replace (B) by the budget remaining after
policy fitting.

## Corrected proposition

Conditional on a policy-fit fold, suppose the frozen nested costs
(c_i(\lambda)) and full-observation costs (b_i) are identical measurable
functionals of exchangeable CRC/deployment trajectories.  Put

\[
 \rho=\frac{n+1}{m},\qquad
 K_i(\lambda)=c_i(\lambda)+\rho(b_i-\mu_b),
 \qquad \mu_b=\mathbb E[b_i\mid\mathcal I_{\rm fit}].
\]

If the family is nested/right-continuous, the selector is feasible, and

\[
 K_z(\widehat\lambda)\le L_K
 \quad\text{for every possible fresh trajectory }z\quad\text{a.s.},
\]

then the selector

\[
 \frac{\sum_{i=1}^n c_i(\lambda)+L_K}{n+1}
 \le
 \frac{B_{\rm rem}-\sum_{i=1}^n b_i}{m}
\]

has the claimed marginal expected-total-budget guarantee.  The symmetric
augmented-selector proof in the attachment then goes through verbatim,
conditional on the policy-fit fold.

The attachment sets (L_K=\maxl=M).  This is valid only under its stated
envelope (K_z(\widehat\lambda)\le M).  A deterministic sufficient condition
using the production bounds is

\[
 c_z(\lambda)\le C_{\max},\quad b_z\le M,quad \mu_b\ge0,
 \qquad
 C_{\max}+\frac{n+1}{m}M\le M.
\]

The corrected causal shared-PAV cap supplies (c\le C_{\max}) in current LPB
and metric DAPRO runs.  The saved-run audit verifies that this sufficient
condition holds in all eligible LPB/metric rows.  It is not automatic for the
currently uncapped UPB family.

## Why this is not the earlier aggressive selector

The attachment uses

\[
 \frac{\sum_i c_i(\lambda)+M}{n+1}
 \le \frac{B_{\rm rem}-\sum_i b_i}{m}.
\]

The earlier proposed rule used (C_{\max}), not (M), in the numerator.  The
proof therefore does not validate that earlier rule unless one strengthens
the envelope all the way to

\[
 c_z(\widehat\lambda)+\rho(b_z-\mu_b)\le C_{\max},
\]

which is generally false.  The exact finite counterexample for the earlier
rule violates this stronger envelope.

Without separately assuming (K\le M), the generic deterministic bound is

\[
 L_K=C_{\max}+\frac{n+1}{m}M.
\]

The resulting valid realized-residual selector is always more conservative
than production composite CRC.  Indeed, writing both selectors on the same
per-deployment scale gives

\[
 U_{\rm transformed}-U_{\rm production}
 =\frac{\sum_i b_i+M}{m(n+1)}>0.
\]

Thus the transformed proof does not yield a tighter certified selector than
the current composite-loss construction.

## Saved-run reconstruction of the exact attached rule

The exact (+M) selector was reconstructed over 23,999 eligible saved rows:

| Task | Selector feasible | Production mean budget | Attached-proof mean budget | Change |
|---|---:|---:|---:|---:|
| LPB | 93.8% | 13.500 | 10.531 | -2.970 |
| Metric | 90.6% | 13.093 | 10.332 | -2.761 |
| UPB | 100% | 12.938 | 12.976 | +0.038 |

For budget 10, feasibility falls to 79.2% for LPB and 60.8% for metrics at
policy-fit/control sizes 25/25.  For the priority toxicity/Qwen LPB setup at
budget 20, the attached selector reduces mean expected budget as follows:

| Total configured (N_1) | Production | Attached proof |
|---:|---:|---:|
| 50 | 18.956 | 12.948 |
| 100 | 19.351 | 16.320 |
| 200 | 19.713 | 18.250 |

It therefore avoids empirical over-consumption by being substantially more
conservative, and it is sometimes infeasible even at the minimum-propensity
candidate.  This is the opposite of the intended motivation for replacing
production CRC.

## Recommendation

Do not make the attached selector the primary method.  Keep production
composite CRC in the code and main theorem.  The attached result is suitable
as an appendix proposition after adding the separate frozen policy-fit fold
and explicitly stating the uniform transformed envelope.  If simplicity is
the goal, omit the transformed variant entirely: it adds a strong condition
and, on the saved experiments, is less efficient.
