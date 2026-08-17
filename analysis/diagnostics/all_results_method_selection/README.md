# Cross-task budget-allocation method selection

This audit uses every merged result currently present under `results`:

- 30 LPB files, 720 exact method configurations;
- 22 UPB files, 1,584 exact method/coverage configurations;
- 36 metric files, 773 exact method configurations.

Every selected configuration contains 50 random calibration--test splits.
LPB and UPB files are internally source-consistent (one source fingerprint per
task).  Hallucination directories have historical names but contain the same
current method registry as the suffixed files.

The reduction is reproduced by
`analysis/diagnostics/all_results_method_selection.py`.  The three
`*_config_summary.csv` files contain one row per matched task cell and exact
method/N1/controller configuration.

## Comparison criteria

- LPB: variance over 50 splits at 90% coverage, checked alongside squared
  error from 90%, mean bound value, and expected budget.
- UPB: the same quantities separately at 70%, 80%, and 90% coverage, plus the
  exact conditional variance of the estimator actually used.
- Metrics: MSE relative to the fixed full-benchmark value for both unsafe-event
  rate and the implemented event-time ratio.  Ratios are normalized to Static
  inside each dataset/model/budget cell.  The joint score is the worse of the
  two normalized MSE ratios, preventing one metric from hiding failure on the
  other.

The metric endpoint/block rows are not eligible for the final recommendation.
The current allocator may stop a selected metric trajectory at the tau-prior
quantile rather than at M, so it does not necessarily resolve its declared
terminal target.  Its apparently excellent MSE cannot be interpreted as a
valid design-based comparison until that implementation is corrected and
rerun.

## Best common tested family

The strongest existing cross-task compromise is **raw soft-prefix Generalized
DAPRO with N1=50**.

| Task/output | Median ratio to Static | Cells better than Static |
|---|---:|---:|
| LPB 50-split coverage variance | 0.280 | 26/30 |
| UPB 50-split coverage variance | 0.617 | 50/66 |
| Metric unsafe-event-rate MSE | 0.317 | 31/31 |
| Metric event-time-ratio MSE | 0.525 | 30/31 |
| Metric worst-of-two normalized MSE | 0.539 | 30/31 below one on both metrics |

Its task-wise median regret relative to the retrospectively best eligible
method in each cell is 1.061 (LPB), 1.325 (UPB), and 1.042 (joint metrics).
Thus it is especially close to the per-cell winner on LPB and metrics; UPB is
the principal weakness.

The result is not caused by one budget.  For raw soft-prefix N1=50:

| Task | Budget | Median ratio to Static |
|---|---:|---:|
| LPB variance | 10 | 0.222 |
| LPB variance | 20 | 0.291 |
| UPB variance | 10 | 0.572 |
| UPB variance | 20 | 0.674 |
| Joint metric MSE | 10 | 0.607 |
| Joint metric MSE | 20 | 0.428 |

The N1=50 method is not available in all budget-5 metric cells because fully
observing its Phase-I fold can itself exceed the total budget.

## UPB coverage-level result

Corrected sequential-AHT soft-prefix UPB DAPRO is useful, but is not the UPB
winner and weakens as the target event becomes rarer.

| Target coverage | Median variance ratio to Static | Wins |
|---:|---:|---:|
| 70% | 0.493 | 17/22 |
| 80% | 0.645 | 17/22 |
| 90% | 0.781 | 16/22 |

The model-only endpoint/block UPB allocator is best overall: its corresponding
ratios are 0.467, 0.509, and 0.665, with 60/66 total wins.  Information-gain
N1=100 is close and wins 59/66 cells.  Soft-prefix has meaningful failures,
including a 2.26x variance ratio on hallucination/Mini-Phi at 90% and roughly
1.3--1.4x ratios in several toxicity/low-coverage cells.  It is therefore
incorrect to call soft-prefix universally superior on UPB, but equally
incorrect to call it generally worse than Static.

The less noisy exact conditional acquisition-variance diagnostic gives the
same ordering: median ratios to Static are 0.380 for endpoint/block, 0.513 for
information-gain N1=100, 0.512 for soft-prefix N1=50, and 0.745 for
soft-prefix+CRC N1=100.  Thus the across-split result is not merely a lucky
50-split fluctuation.

## Best method by task

If different methods are allowed:

- **LPB:** information-gain N1=100 has the smallest median ratio (0.265), but
  soft-prefix+CRC N1=50 has the strongest win/tail profile (28/30 wins, median
  0.300, 90th percentile 0.932) and a finite-sample marginal budget
  certificate.  The differences among the leading prefix methods are small.
- **UPB:** model-only endpoint/block is best (median 0.517, 60/66 wins).  Its
  CRC N1=200 version is the strongest certified endpoint alternative (median
  0.624, 54/66 wins).
- **Unsafe-event rate only:** information-gain sequential AHT is the strongest
  eligible dynamic metric method (roughly 0.18 median MSE ratio at N1=100).
- **Both reported metrics jointly:** raw soft-prefix N1=50 is best among valid
  tested methods (median worst-metric ratio 0.539).

Static is therefore not the general winner for any of the three tasks, though
it wins individual cells and remains the simplest low-cost baseline.

## Budget validity

Raw soft-prefix spends nearly the nominal budget on average, but bisection on
model/Phase-I expected cost is not a finite-sample budget theorem.  Its median
expected-cost ratios are 0.999 (LPB), 0.979 (UPB), and 1.002 (metrics), while
only about 60%, 64%, and 55% of the underlying split diagnostics respectively
fall at or below the nominal budget.

The best single certified compromise is **soft-prefix+CRC with N1=100**:

| Task | Median ratio to Static |
|---|---:|
| LPB variance | 0.315 |
| UPB variance | 0.828 |
| Joint metric worst-MSE | 0.878 |

Every CRC selector reports a valid certificate, but CRC is marginal over the
exchangeable control draw and may underuse the budget.  This compromise is
less compelling on UPB at 90% coverage (median ratio 0.958; 12/22 wins) and on
hallucination event-time estimation.

## Metric interpretation limitation

The saved `estimated_rmttu` is not the classical restricted mean
E[min(T,M)].  It is E[T | T<=M], estimated as a ratio of event-time and event-
count HT totals.  Existing experiments therefore validate raw soft-prefix as
a useful shared allocation for unsafe-event rate and this conditional
event-time ratio.  They do not establish performance for classical restricted
mean survival/event time.

The event-target soft-prefix policy helps the ratio because its influence is
supported on event rows, A(T-r).  It ignores the magnitude |T-r| and is not
ratio-optimal, but preferentially resolving event rows is still useful.  A
classical restricted mean also depends on survivors through M, so the same
argument does not apply.

## Paper recommendation

For a paper that insists on one simple allocation idea, present
**soft-prefix Generalized DAPRO** as the core method and treat CRC as an
optional scalar budget controller, not as a second allocation algorithm.
Use raw N1=50 for the strongest efficiency result and the N1=100 CRC variant
for the budget-certified deployment result.  State explicitly that the soft
target changes with the scientific output and that the reported event-time
metric is secondary.

Do not claim one universally optimal allocation.  If the paper wants a single
policy theoretically aligned with event rate, conditional event time, and
classical restricted mean simultaneously, the next method must be a
multi-target/vector influence allocation with vector AHT.  That method has not
yet been evaluated in the saved results and should not be substituted into the
main empirical claim without a new experiment suite.
