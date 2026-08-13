# DAPRO score, binning, proxy, and causal-CRC audit

Date: 2026-08-11

This note separates exploratory score/bin ablations from the production
50-split CRC validation.  Every cost reported below is an expected number of
acquired turns per one of the 3,000 evaluation rows and **includes Phase I**.
For `N1=50, control=25`, this means 25 policy-fit rows and 25 CRC-control rows
are fully observed and their realized costs are included before adding the
Phase-II expected cost.

## Data and target

Priority configurations:

1. `dataset_toxicity` / `attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify`
2. `dataset_red_team` / `attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct`

All runs use `M=200`, `B=20`, `cal_size=3000`, and the real 6,000-row
`probability_est_cal_test.pt` cache.  Conditional variances are for the
Horvitz--Thompson unsafe-event-rate estimator and are expressed in squared
percentage points.

## What two bins actually do

The cache contains essentially continuous scores: on toxicity/Qwen there are
about 3,000--6,000 distinct active scores per inspected turn.  Median binning
reduces this to two score IDs at each turn, but the optimizer pools the two
IDs very frequently.  In the reproduced toxicity seed-0 policy, both bins
receive exactly the same raw continuation probability at 178/200 turns
(89%).  Across nine further splits, the mean number of distinct raw
probabilities is 1.115 per turn for `K=2`, 1.226 for `K=4`, and 1.314 for
`K=8`.

Selected toxicity seed-0 examples:

| Turn | Median hazard cut | Raw q, low | Raw q, high |
|---:|---:|---:|---:|
| 1 | 0.002440 | 0.2313 | 0.2746 |
| 2 | 0.016097 | 1.0000 | 1.0000 |
| 5 | 0.015396 | 0.9437 | 1.0000 |
| 20 | 0.006069 | 0.9562 | 0.9562 |
| 50 | 0.002039 | 0.9661 | 1.0000 |
| 100 | 0.003248 | 1.0000 | 1.0000 |
| 200 | 0.000417 | 0.6302 | 0.8248 |

This does not imply only two terminal inclusion propensities.  The entire
history of bin IDs is multiplied into cumulative reach, and the shared
cumulative-logit budget correction is path dependent.  For example, the
same seed has 49 corrected conditional probabilities at turn 20, 105 at turn
50, and 371 at turn 200.

Median binning nevertheless discards discrimination.  On toxicity, raw
one-step-hazard AUC versus the eventual unsafe event at turns
1/2/5/10/20/50/100/150 is
`.749/.686/.709/.749/.740/.725/.718/.698`; the binary-bin AUC is
`.682/.645/.674/.693/.680/.660/.663/.643`.

## Exploratory no-CRC ablations

These are policy reconstructions that reproduce the production no-CRC result
to numerical precision.  They are exploratory 9- or 10-split runs, not the
production 50-split result.

### Toxicity/Qwen, nine splits (seeds 1--9)

| Score | Bins | Exact variance | Expected total cost |
|---|---:|---:|---:|
| time only | 1 | 8.0544 | 19.4104 |
| one-step hazard | 2 | 7.1469 | 19.4722 |
| one-step hazard | 4 | 7.1265 | 19.3167 |
| one-step hazard | 8 | 7.1377 | 19.2369 |
| future event risk | 2 | 7.1343 | 19.4003 |
| future event risk | 4 | 7.0982 | 19.3263 |
| future event risk | 8 | 7.1481 | 19.2058 |

### Red-team/Qwen, ten splits (seeds 0--9)

| Score | Bins | Exact variance | Expected total cost |
|---|---:|---:|---:|
| one-step hazard | 2 | 10.9692 | 18.2598 |
| one-step hazard | 4 | 10.9006 | 18.2276 |
| future event risk | 2 | 10.9325 | 18.2743 |
| future event risk | 4 | 10.8758 | 18.2351 |

Thus four bins and a target-aligned future-value score help, but only by about
one percent on the two priority Qwen setups.  Eight bins begin to overfit.
The larger information loss is material on at least one other cache:
toxicity/Phi (five exploratory splits) improves from 6.6834 for hazard/K2 to
6.3141 for future-risk/smooth-K4, with cost changing from 19.3077 to 19.3821.

## Smooth-rank prototype

`smooth-rank DAPRO` uses the optimized four-bin probabilities as monotone
empirical-rank knots and geometrically interpolates between adjacent knots.
It adds no label-fitted regression and preserves score monotonicity.  Relative
to hard K4:

* Toxicity/Qwen seeds 1--9: hazard 7.1265 -> 7.0401 (9/9 wins), and
  future-risk 7.0982 -> 7.0351 (8/9 wins).
* Red-team/Qwen seeds 0--9: hazard 10.9006 -> 10.8772 (7/10 wins), and
  future-risk 10.8758 -> 10.8564 (8/10 wins).

This establishes measurable bin-quantization headroom, but also establishes
that it is not the dominant source of the large differences between complete
allocators.

## Full 50-split factorial: bin count, score, and interpolation

The exploratory screen was extended to all 50 paired calibration--test split
seeds.  Every row below uses the same outer split, the same `N1=50` policy-fit
split, and the same projection-margin controller.  The variance is the mean
exact fixed-split HT acquisition variance; cost includes the 50 fully observed
Phase-I rows.  No cost matching or rescaling was applied.

| Setup | Policy | Exact variance | Expected total cost | Change from hazard/K2 |
|---|---|---:|---:|---:|
| Toxicity | hazard, hard K2 | 7.2653 | 19.3851 | reference |
| Toxicity | hazard, hard K4 | 7.1456 | 19.4375 | -1.65% |
| Toxicity | future risk, hard K2 | 7.2080 | 19.3965 | -0.79% |
| Toxicity | future risk, hard K4 | 7.0965 | 19.4654 | -2.32% |
| Toxicity | hazard, smooth K4 | 7.0588 | 19.4626 | -2.84% |
| Toxicity | future risk, smooth K4 | 7.0178 | 19.4594 | -3.41% |
| Red-team | hazard, hard K2 | 10.3238 | 19.0497 | reference |
| Red-team | hazard, hard K4 | 10.2361 | 19.0648 | -0.85% |
| Red-team | future risk, hard K2 | 10.3017 | 19.0634 | -0.21% |
| Red-team | future risk, hard K4 | 10.2177 | 19.0702 | -1.03% |
| Red-team | hazard, smooth K4 | 10.2126 | 19.0954 | -1.08% |
| Red-team | future risk, smooth K4 | 10.1963 | 19.0982 | -1.23% |

The best policy beats the original policy on 42/50 toxicity splits and 43/50
red-team splits.  Paired 20,000-resample bootstrap intervals for the exact
variance difference are `[-0.315,-0.182]` and `[-0.165,-0.093]` squared
percentage points.  The comparison is therefore not driven by the initial
ten seeds.  The extra expected cost is only 0.074 turns on toxicity and 0.049
on red-team, although the comparison is not exactly cost matched.

The requested variance across random splits has two components.  The
empirical full-observation calibration-split variance is 0.3449 squared
percentage points for toxicity and 0.2500 for red-team.  Adding this common
component to the mean exact acquisition variance gives the design-expected
joint split/acquisition variance: 7.6102 to 7.3627 on toxicity and 10.5738 to
10.4464 on red-team.  This sum is more stable than the sample variance of only
50 one-shot HT acquisition realizations; the latter can fluctuate above or
below its expectation and is retained as the primary plotted empirical
quantity, not as the policy-mechanism diagnostic.

## Objective/proxy audit

For a fixed metric target and fixed benchmark rows,

`N^{-2} sum_i A_i (1 / pi_i - 1)`

is the exact conditional acquisition variance, not a proxy.  The stored hard
Phase-II objective equals it to machine precision.  Approximation enters when
DAPRO replaces the hard endpoint event with model-predicted causal prefix
event masses and estimates the policy from a small Phase-I fold.

On 108 toxicity policy/split points, the Phase-I soft training surrogate has
Pearson correlation 0.955 with exact variance but only 0.428 mean within-split
Spearman correlation for ranking nearby policies.  The Phase-II soft
surrogate has Pearson 0.978, within-split Spearman 0.711, and is 5.5% below the
hard objective on average.  Red-team gives 0.989 across-point Pearson but
only 0.10 within-split Spearman for the Phase-I surrogate; the Phase-II soft
surrogate has 0.985 Pearson, 0.78 within-split Spearman, and is 2.5% low on
average.  Therefore the variance theory is exact; the practical weakness is
coefficient/model generalization when selecting among close policies.

## Production causal shared-PAV CRC, 50 splits

Method:
`dapro_soft_prefix_bins_2_metric_horizon_200_global_0p001_budget_crc_control_25_row_cap_2p00x_budget_causal_shared_pav_v1_n1_50`.

| Setup | Method | Exact conditional variance | Across-split estimator variance | Expected total cost |
|---|---|---:|---:|---:|
| Toxicity | old noncausal CRC | 7.8527 | 9.5563 | 18.4612 |
| Toxicity | no CRC | 7.2652 | 8.0676 | 19.3852 |
| Toxicity | causal shared-PAV CRC | 7.6977 | 10.5840 | 18.3545 |
| Red-team | old noncausal CRC | 11.5923 | 9.5326 | 17.8869 |
| Red-team | no CRC | 10.3238 | 7.1591 | 19.0496 |
| Red-team | causal shared-PAV CRC | 11.3412 | 7.8126 | 17.9305 |

Against the old noncausal cap, the causal cap improves exact variance by
0.1550 pp^2 (1.97%) on toxicity and 0.2511 pp^2 (2.17%) on red-team.  Paired
bootstrap 95% intervals are [-0.237,-0.084] and [-0.351,-0.153].  Across-split
variance is much noisier: the paired bootstrap intervals for the difference
include zero on both setups.

All 100 CRC selectors are valid.  Metadata reports
`risk_budget_row_cost_cap_uses_future_prefixes=0`; the shared envelope sums to
40 turns, and it changes about 59% of toxicity and 89% of red-team deployment
rows.  Six toxicity splits and three red-team splits have conditional
expected cost above 20.  This does not contradict CRC: the guarantee is
marginal over the independent control data, while the mean expected costs
over 50 splits are 18.35 and 17.93.

Production merged files:

* `results/merged_metric_calibration_dfs/dataset_toxicity_attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify_20_m_causal_pav_crc_v2/all_df.csv`
* `results/merged_metric_calibration_dfs/dataset_red_team_attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct_20_m_causal_pav_crc_v2/all_df.csv`

## Reproduction commands

The exploratory driver is `analysis/diagnostics/dapro_binning_audit.py`.
Example:

```powershell
python analysis/diagnostics/dapro_binning_audit.py --setup red_qwen --seeds 0,1,2,3,4,5,6,7,8,9 --n1 50 --bins 2,4 --scores hazard,future_risk --objectives soft --projections hard_bin,smooth_rank --output tmp/dapro_audit_red_10seeds.csv
```

Production estimate command (replace dataset/setup for toxicity):

```powershell
python -m src.evaluation.estimate --seed-start 0 --seed-end 50 --data-type real --dataset-name dataset_red_team --dataset-setup attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct --budget-per-sample 20 --cal-size 3000 --device cpu --dapro-n1 50 --crc-control-size 25 --experiment-suffix causal_pav_crc_v2 --allocator-name dapro_soft_prefix_bins_2_metric_horizon_200_global_0p001_budget_crc_control_25_row_cap_2p00x_budget_causal_shared_pav_v1_n1_50 --overwrite
```

The matching merge changes only the module and omits `--overwrite`:

```powershell
python -m src.evaluation.merge_results --seed-start 0 --seed-end 50 --data-type real --dataset-name dataset_red_team --dataset-setup attack_default_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_llm-judge_qwen25_14b_instruct --budget-per-sample 20 --cal-size 3000 --device cpu --dapro-n1 50 --crc-control-size 25 --experiment-suffix causal_pav_crc_v2 --allocator-name dapro_soft_prefix_bins_2_metric_horizon_200_global_0p001_budget_crc_control_25_row_cap_2p00x_budget_causal_shared_pav_v1_n1_50
```
