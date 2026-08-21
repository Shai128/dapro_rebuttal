# Recommended follow-up ablations (not implemented)

The implemented study isolates Phase-I size, score quality, and budget.  The
following additions are the most likely to answer ICLR/NeurIPS reviewer
questions.  They are ranked by expected evidentiary value.

## Highest priority

1. **Target coefficient: hard versus soft prefix.** Compare Target-A/Definitive
   endpoint indicators against the Rao--Blackwellized soft-prefix masses with
   the same score, K2 representation, folds, controller, and common random
   numbers.  This isolates the central Generalized-DAPRO contribution.
2. **Representation capacity.** Compare K=1,2,4,8 bins and a small continuous
   monotone score map.  Report Phase-I objective, Phase-II target-weight
   objective, coverage variance, and runtime.  This answers whether K2 throws
   away useful score information or acts as needed regularization.
3. **Score semantics.** Compare instantaneous hazard, remaining target-event
   probability, and target-risk/remaining-cost (Neyman value).  Keep K and the
   budget controller fixed.  A random score and a full-information/oracle score
   provide lower and upper quality anchors.
4. **Variance decomposition.** For several fixed calibration/test splits, run
   200--500 acquisition replicates.  Separate acquisition variance from split,
   policy-fit, candidate-switching, and test-sample variance.  Fifty outer
   splits alone are too noisy to resolve modest allocation gains.
5. **CRC price and control-fold allocation.** Hold total N1 fixed and vary the
   CRC fraction (e.g. 20%, 33%, 50%, 67%), then also compare at equal policy-fit
   size.  Report selector validity, expected and realized budget, selected CRC
   contraction, and variance.  This prevents conflating CRC with less policy
   training data.

## Robustness and validity

6. **Probability-model misspecification.** Temperature perturb, recalibrate,
   or use a weaker hazard model.  HT validity should remain while efficiency
   changes; this empirically separates validity from model-assisted variance
   reduction.
7. **Distribution shift.** Fit the policy on one target model/domain and deploy
   on another compatible population.  Report both raw and CRC policies; this is
   the stress test for projection transfer.
8. **Positivity and row-cap sensitivity.** Vary terminal propensity floor and
   CRC row-cap multiplier.  Report maximum weights, effective sample size,
   budget utilization, and target-weight variance, not coverage alone.
9. **Calibration size separate from N1.** Vary the total calibration/test split
   size while fixing N1, then vary N1 at fixed total size.  This distinguishes
   allocation-learning error from downstream calibration sampling error.
10. **Target-anchor robustness.** Compare raw alpha, Phase-I calibrated anchor,
    a small boundary band, and a multi-target objective.  This directly tests
    sensitivity to the ultimately selected LPB candidate.

## Breadth and practicality

11. **Coverage levels and prior horizon.** Repeat at 70%, 80%, 90%, and 95%
    coverage and vary tau-prior.  Rare target events are where K2 quantization
    and inverse weights are most likely to fail.
12. **Cross-dataset/model replication.** Use at least one toxicity, red-team,
    hallucination, and instruction-following setup, with paired confidence
    intervals for DAPRO minus Static rather than only separate boxplots.
13. **Runtime and memory.** Report fit and deployment time, peak memory, and
    number of learned policy parameters versus N1, K, and horizon.
14. **Oracle efficiency gap.** Include full-budget, target-aware clairvoyant,
    random-score, and time-only policies to show how much of the attainable
    variance reduction is captured by the learned score and representation.

For every follow-up, preserve identical outer splits and acquisition uniforms
across methods.  Primary comparisons should use the exact conditional target
HT variance when available, with across-split coverage variance reported as a
separate downstream outcome.
