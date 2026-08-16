# UPB DAPRO versus Static: implementation and variance audit

This note audits the merged UPB experiment `upb_unified_aht_v1`.  The source
data are the 16 merged files below `results/merged_upb_calibration_dfs`.  The
matched comparison contains 144 experiment cells: four benchmark/model
setups, two budgets, three target coverages, and the applicable Phase-I/CRC
sizes, with 50 calibration--test splits per cell.

Machine-readable summaries are in:

- `analysis/diagnostics/upb_unified_aht_summary.csv`
- `analysis/diagnostics/upb_unified_aht_cell_audit.csv`

## Main result

The conclusion that no UPB method improves on Static is not supported by the
exact variance of the estimator actually used.  The table reports the median
ratio to Static of the exact conditional acquisition variance and the fraction
of matched cells in which the method is better than Static.

| UPB method | Median exact-variance ratio | Cells better than Static | Median 50-split coverage-variance ratio | Mean expected cost |
|---|---:|---:|---:|---:|
| Endpoint/block + terminal residual AHT | 0.483 | 100.0% | 0.636 | 16.41 |
| Information-gain + sequential AHT | 0.634 | 91.7% | 0.635 | 18.07 |
| Residual + sequential AHT | 0.758 | 68.8% | 0.756 | 17.95 |
| Endpoint/block + terminal residual AHT + CRC | 0.770 | 85.4% | 0.828 | 13.85 |
| Information-gain + sequential AHT + CRC | 0.987 | 51.4% | 0.985 | 13.98 |
| Residual + sequential AHT + CRC | 1.241 | 35.4% | 1.103 | 13.97 |
| Historical soft-prefix DAPRO | 1.712 | 18.8% | 1.915 | 17.85 |
| Historical soft-prefix DAPRO + CRC | 2.909 | 4.9% | 2.793 | 13.96 |

The endpoint/block method is the strongest overall UPB allocator in these
experiments.  Among genuinely history-adaptive methods, information-gain DAPRO
is the strongest.  The improvement is not confined to one setup: the endpoint
method's median exact-variance ratios are 0.588 (AutoIF), 0.231 (red-team with
LlamaGuard), 0.486 (red-team with the Qwen judge), and 0.722 (toxicity).

## The soft-prefix implementation defect

The historical soft-prefix class optimized a dynamic prefix policy but declared
`upb_estimator_kind = "ordinary_ht"`.  It therefore discarded every intermediate
model update and used ordinary terminal Horvitz--Thompson estimation.  Static,
by contrast, used terminal residual AHT; information-gain and residual DAPRO
used sequential AHT.  This was an estimator mismatch, not a fair comparison of
allocation policies.

The result metadata compounded the problem by writing
`upb_calibration_estimator=sequential_augmented_horvitz_thompson` for every
method, irrespective of the estimator that was actually run.

The correction does three things:

1. soft-prefix UPB DAPRO now declares the sequential estimator;
2. its allocator name contains `seq_estimator_v2`, preventing old and corrected
   result files from being mixed or skipped;
3. `upb_calibration_estimator` records the actual ordinary, terminal-residual,
   or sequential estimator.

On identical policies and five matched splits for each target coverage, merely
correcting the estimator reduced the soft-prefix exact variance by 50--87%.
For red-team/Qwen the corrected raw method beats Static at 70%, 80%, and 90%
coverage (ratios 0.488, 0.645, and 0.690).  For toxicity it beats Static at 90%
(0.637) but remains slightly worse at 70% and 80% (1.108 and 1.230).  Thus the
bug explains most of the catastrophic result, but not all remaining efficiency
differences.

## Why mean target weight can be misleading

`mean_a_weighted_inverse_probability` is the latent raw-HT diagnostic

\[
  n^{-1}\sum_i A_i/\pi_i.
\]

It is not the variance of a residual or sequential augmented estimator.  For a
terminal residual AHT estimator the acquisition-variance term is

\[
  n^{-2}\sum_i (A_i-m_{i0})^2(\pi_i^{-1}-1),
\]

and for sequential AHT the exact full-path term is

\[
  n^{-2}\sum_i\sum_t
  (\rho_{it}^{-1}-\rho_{i,t-1}^{-1})
  (A_i-m_{i,t-1})^2.
\]

Consequently, improving `A_i/pi_i` is neither necessary nor sufficient for
improving the estimator used by these UPB methods.  A striking example in the
merged results is information-gain DAPRO without CRC: its median raw target-
weight ratio to Static is 2.259, while its median exact estimator-variance ratio
is 0.634.  The summary plots now include
`estimated_conditional_variance_upb_coverage_estimator`, which is the appropriate
comparison.  The similarly named `mean_a_weighted_inverse_probability_minus_one`
diagnostic was also corrected: it had accidentally contained a residual-squared
variance proxy rather than the advertised raw \(A(1/\pi-1)\) quantity.

## Why corrected soft-prefix DAPRO is still not the best UPB design

The original soft-prefix objective places model event mass at the candidate UPB
endpoint.  That is closely aligned with ordinary endpoint HT, but sequential
AHT gains efficiency from prediction updates along the trajectory.  Its natural
coefficient is squared information gain, approximately
\((m_{it}-m_{i,t-1})^2\), not endpoint survival mass.  Information-gain DAPRO
therefore aligns the allocation objective with the estimator.

Other contributors are:

- terminal propensities are products of many continuation decisions and can
  approach the 0.005 floor;
- Phase I fully observes trajectories averaging about 98 turns, consuming a
  substantial fraction of the budget;
- roughly 152--155 of the 400 time/bin cells are empty in aggregate diagnostics,
  so late K=2 mappings have little direct fitting support;
- the CRC variant uses the full 200-turn support envelope and contracts the
  policy substantially.  Its average cost is about 14 rather than 18--20.

The raw non-CRC policy is more efficient but is only model/transfer controlled:
in the historical files its exact Phase-II expected-cost diagnostic is at or
below the target in only about 56--60% of split cells.  CRC supplies marginal
expected-budget validity, not a guarantee that every conditional split cost is
at most the budget.

## Recommendation

- **Primary UPB method when the model-budget assumption is acceptable:**
  endpoint/block allocation with terminal residual AHT.  It is the lowest-
  variance method in the full audit and beats Static in all 144 matched cells.
- **Primary genuinely history-adaptive UPB method:** information-gain DAPRO with
  sequential AHT.  It is task-aligned and beats Static in 91.7% of cells.
- **Primary finite-sample budget-controlled method among those tested:**
  endpoint/block + terminal residual AHT + CRC.
- **Ablation only:** corrected soft-prefix DAPRO v2.  It is now implemented
  consistently, but its endpoint-mass objective is not the best match for
  sequential AHT.

The UPB anchor itself is not the source of the failure.  Across a representative
file, initial model miscoverage at nominal 70/80/90% coverage is approximately
0.300/0.200/0.100; calibration moves it to 0.289/0.190/0.089 and test coverage
is approximately 0.701/0.808/0.908.  The issue was primarily estimator/objective
alignment and budget-controller distortion, not a reversed UPB event or a
misinterpreted infinity value.  The value 201 continues to represent no event
by time 200 and therefore has deterministic UPB miscoverage zero.
