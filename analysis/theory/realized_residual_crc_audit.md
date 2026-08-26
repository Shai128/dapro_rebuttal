# Realized-residual CRC: theory and saved-run audit

> **Scope.** This note audits the earlier rule whose correction is only
> (C_{\max}).  The later proof sketch supplied by the author uses the larger
> correction (\maxl) and a transformed-loss envelope.  That is a different
> selector; see `attached_realized_residual_crc_proof_audit.md`.

## Two selectors

Condition on the policy-fit fold.  Let (n) be the CRC size, (m) the
deployment size, (b_iin[0,M]) the fully observed CRC cost, and
(c_i(k)in[0,C_{\max}]) the expected acquisition cost of frozen nested
candidate (k) on CRC trajectory (i).  Write (B_{\rm rem}) for the budget
remaining after the policy-fit fold.

The production composite-loss CRC selector uses

\[
 U_{\rm current}(k)=
 \frac{
   \sum_{i=1}^n c_i(k)+(n/m)\sum_{i=1}^n b_i
   +C_{\max}+(n/m)M
 }{n+1}
 \leq \frac{B_{\rm rem}}m .
\]

The proposed realized-residual selector is equivalently

\[
 U_{\rm new}(k)=
 \frac{\sum_{i=1}^n c_i(k)+C_{\max}}{n+1}
 +\frac{\sum_{i=1}^n b_i}{m}
 \leq \frac{B_{\rm rem}}m .
\]

Their difference is

\[
 U_{\rm current}(k)-U_{\rm new}(k)
 =\frac{nM-\sum_i b_i}{m(n+1)}\geq0.
\]

This is an identity under the existing support assumption (b_i\leq M), not
an additional validity assumption.  It proves only that the new rule is less
conservative.  A quantity smaller than a valid upper bound is not necessarily
an upper bound.

## The actual sufficient assumption

One sufficient condition for the new rule is

\[
 \mathbb E\!left[c_{n+1}(\widehat k)\mid\mathcal I_1\right]
 \leq
 \mathbb E\!\left[
   \frac{\sum_{i=1}^n c_i(\widehat k)+C_{\max}}{n+1}
   \middle|\mathcal I_1
 \right].
 \tag{A}
\]

Taking expectations in the proposed pointwise selector and using
exchangeability of (b_i) then gives

\[
 m\,\mathbb E[c_{n+1}(\widehat k)]+n\,\mathbb E[b_{n+1}]
 \leq B_{\rm rem}.
\]

But (A) is precisely the CRC generalization statement that can fail when the
same CRC outcomes both determine the random residual target and select
(\widehat k).  Bounding (b_i) or (f_\tau) does not establish (A).  A
deterministic constant pilot reservation makes the threshold fixed and
restores ordinary CRC, but that reduces to the conservative accounting rather
than the proposed realized-residual rule.

## Exact finite counterexample

Take one CRC row and one deployment row.  The population has three row types
with probabilities

\[
 (0.6819741,\ 0.1070096,\ 0.2110163),
\]

pilot costs (b=(0.4,0.4,0.8)), and three nested candidate-cost columns

\[
 c(1)=(1,0.8,1),\qquad
 c(2)=(1,0.2,0.2),\qquad
 c(3)=(0.4,0,0).
\]

Let (C_{\max}=1) and (B_{\rm rem}=1.4).  The proposed selector is feasible
for every possible CRC type.  It selects candidate 1 for types 1--2 and
candidate 2 for type 3.  Exact enumeration gives

\[
 \mathbb E[b_{\rm CRC}+c_{\rm deploy}(\widehat k)]
 =1.4138338>1.4.
\]

Thus bounded costs, nesting, exchangeability, feasibility, and
(nM-\sum b_i\geq0) do not imply the desired budget guarantee.

## Saved-run audit

`analysis/diagnostics/realized_residual_crc_audit.py` reconstructs the full
affine CRC candidate family from the saved selected scale and costs, changes
only the selector, and evaluates the exact conditional expected Phase-II cost.
It reproduces the production-selected scale on all 23,999 eligible saved rows
to maximum absolute error (1.1\times10^{-16}).

| Task | Split/method rows | Proposed scale increases | Current mean budget slack | Mean budget recovered | Splitwise valid |
|---|---:|---:|---:|---:|---:|
| LPB | 5,700 | 55.4% | 1.763 | 0.0217 | 87.0% |
| Metric | 5,699 | 43.6% | 2.171 | 0.0145 | 84.3% |
| UPB | 12,600 | 76.0% | 4.205 | 0.0376 | 96.7% |

Individual splitwise conditional expected-budget overruns are not by
themselves failures of a *marginal* CRC theorem.  More importantly, across the
50 saved splits per configuration, the proposed selector has three empirical
mean overruns among 480 configurations, compared with two for production CRC.
The additional failure is an LPB AutoIF/MiniPhi-4 configuration at budget 20:
production has mean 19.9849 turns and the proposed selector has mean 20.0154.
The two metric failures already fluctuate slightly above 20 under production
CRC and become slightly larger under the proposal.

The proposal therefore does not empirically satisfy the budget in every saved
configuration, and it recovers only about one percent of the observed slack.
The dominant slack comes from policy-fit/pilot expenditure, mismatch between
the fitted cost shape and deployment, candidate-family/cap restrictions, and
the finite candidate grid—not from the composite (b_i) correction alone.

## Recommendation

Keep the composite-loss CRC rule in the main theorem.  If a realized-residual
variant is reported, call it an assumption-based heuristic and state (A)
directly; do not cite (nM-\sum b_i\geq0) as sufficient.  Two valid ways to
reduce conservatism are:

1. replace (M) by a smaller **deterministic** pilot support (Q_{\max}) if
   every policy-fit/control/deployment horizon is actually capped at
   (Q_{\max}); or
2. use a simultaneous high-probability UCB for the fixed candidate costs and
   subtract the realized pilot cost.  This is conditionally valid but is often
   more conservative at long horizons.

Limiting (f_\tau) can implement the first option, but it may change the bound
class and coverage if scientifically relevant LPBs exceed the cap.
