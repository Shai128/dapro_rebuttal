# Empirical validation of realized-residual aggressive CRC

## Question and implementation

The experimental selector uses

\[
\frac{\sum_{i\in\mathcal I_{\rm crc}}c_i(\lambda)+C_{\max}}
     {n_{\rm crc}+1}
\le
\frac{B_{\rm rem}-\sum_{i\in\mathcal I_{\rm crc}}b_i}{m}.
\]

The executable candidate family, its policy-fit split, the K=2 score map, and
the row cap \(C_{\max}=2B=40\) are identical to standard CRC DAPRO.  The only
change is the selector above.  The proposed rowwise sufficient condition

\[
c_i(\lambda)\le C_{\max}
 -\frac{n_{\rm crc}+1}{m}(Q_i-\underline b)
\]

was **diagnosed but not enforced**.  We also diagnosed the less conservative
transformed-envelope condition actually used by the proof,

\[
c_i(\lambda)+\frac{n_{\rm crc}+1}{m}(b_i-\mu_b)\le C_{\max}.
\]

The runs use 10 matched calibration--test splits per configuration, a target
budget of 20, K=2, and common acquisition randomness.  LPB and metric runs use
\(N_1=50\), with 25 policy-fit and 25 CRC rows.  UPB runs use \(N_1=100\), with
50 policy-fit and 50 CRC rows, and the recommended 80% UPB policy anchor.  The
datasets are Toxicity and Red Team, with Qwen2.5-14B as attacker and target.

## Budget and variance results

All figures below are means across the 10 matched splits.  `Expected` is the
conditional expected acquisition cost logged by the frozen policy; `realized`
is the actual event-stopped acquisition cost.  LPB variances are in squared
percentage points, event-rate variances in squared percentage points, and
restricted-mean variances in squared turns.

| Task | Dataset | CRC | Expected | Realized | Primary variance | Restricted-mean variance |
|---|---|---:|---:|---:|---:|---:|
| LPB | Red Team | standard | 18.654 | 18.630 | 1.261 | -- |
| LPB | Red Team | aggressive | 18.697 | 18.670 | 1.296 | -- |
| LPB | Toxicity | standard | 19.045 | 19.205 | 0.489 | -- |
| LPB | Toxicity | aggressive | 19.063 | 19.231 | 0.489 | -- |
| Metric | Red Team | standard | 17.483 | 17.548 | 5.228 | 19.128 |
| Metric | Red Team | aggressive | 17.505 | 17.583 | 5.531 | 19.331 |
| Metric | Toxicity | standard | 18.374 | 18.437 | 15.433 | 25.522 |
| Metric | Toxicity | aggressive | 18.392 | 18.456 | 15.755 | 25.948 |
| UPB, 80% policy anchor | Red Team | standard | 17.484 | 17.700 | 5.410 | -- |
| UPB, 80% policy anchor | Red Team | aggressive | 17.496 | 17.719 | 5.349 | -- |
| UPB, 80% policy anchor | Toxicity | standard | 13.723 | 13.857 | 4.865 | -- |
| UPB, 80% policy anchor | Toxicity | aggressive | 13.723 | 13.857 | 4.865 | -- |

For UPB, `Primary variance` is coverage variance at 80%.  At Red Team 90%,
aggressive CRC instead increased coverage variance from 2.982 to 3.093; hence
the very small improvement at 80% is not consistent across coverage targets.

The aggressive rule increased mean realized spending by only 0.001--0.040
turns per sample and the selected mixture scale by only 0.00025--0.00225.  It
therefore leaves the policy almost unchanged.  It did not consistently lower
LPB coverage variance, event-rate variance, restricted-mean variance, or UPB
coverage variance.

## Does the unenforced requirement hold?

| Task/dataset | Selected transformed: splits with a violation | Family transformed | Selected sufficient cap | Family sufficient cap |
|---|---:|---:|---:|---:|
| LPB / Red Team | 0/10 | 0/10 | 0/10 | 0/10 |
| LPB / Toxicity | 0/10 | 0/10 | 0/10 | 0/10 |
| Metric / Red Team | 0/10 | 0/10 | 1/10 | 1/10 |
| Metric / Toxicity | 1/10 | 1/10 | 3/10 | 7/10 |
| UPB / Red Team, 80% anchor | 4/10 | 9/10 | 5/10 | 9/10 |
| UPB / Toxicity, 80% anchor | 9/10 | 10/10 | 9/10 | 10/10 |

For Toxicity metric estimation, the selected sufficient cap failed on 11.7%
of rows in one representative violating split, by as much as 1.104 turns; the
actual transformed envelope failed by only 0.033 turns.  For Toxicity UPB, the
selected sufficient cap failed on 8.9% of rows in a representative split, by
as much as 3.500 turns, while the transformed envelope failed by 1.340 turns.
Thus the sufficient condition is not merely failing because of numerical
tolerance, and the transformed condition needed by the proof is itself false
for UPB.

## Interpretation of split-level overages

The target is a **marginal expected** budget, not a conditional guarantee for
every calibration split or acquisition realization.  Consequently, individual
splits can exceed 20 even under standard CRC.  The aggressive mean expected and
realized budgets stayed below 20 in every configuration.  Approximate 95%
confidence intervals for the means also stayed below 20.  Nevertheless, the
largest individual-split overages were not always tiny: Toxicity LPB reached
6.17% in conditional expected cost and 10.77% in realized cost, while Toxicity
metric estimation reached 6.04% and 12.14%, respectively.  These observations
do not by themselves refute a marginal guarantee, but neither do ten splits
prove one.

## Decision

Do **not** replace standard CRC with this aggressive selector as the primary
method.  The empirical mean budget is acceptable, but the assumption required
by the proposed proof is not universally satisfied, the practical budget gain
is negligible, and variance is not consistently improved.  The implementation
therefore remains an explicitly named experimental mode, `crc_aggressive`; the
production/default DAPRO CRC remains unchanged.

