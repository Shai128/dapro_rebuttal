# Ablation coverage and recommended follow-ups

The launcher now covers Phase-I size, score quality, budget, the full
hard/soft-by-prefix/terminal comparison, representation capacity, named-score
semantics, both attacker-shift directions, and the CRC row-cost cap for LPB.
Every applicable factor is also run for event-rate metric estimation. The
remaining additions below are the most likely to answer ICLR/NeurIPS reviewer
questions and are ranked by expected evidentiary value.

## Highest priority

1. **Variance decomposition.** For several fixed calibration/test splits, run
   200--500 acquisition replicates.  Separate acquisition variance from split,
   policy-fit, candidate-switching, and test-sample variance.  Fifty outer
   splits alone are too noisy to resolve modest allocation gains.
2. **CRC price and control-fold allocation.** Hold total N1 fixed and vary the
   CRC fraction (e.g. 20%, 33%, 50%, 67%), then also compare at equal policy-fit
   size.  Report selector validity, expected and realized budget, selected CRC
   contraction, and variance.  This prevents conflating CRC with less policy
   training data.

## Robustness and validity

3. **Probability-model misspecification.** Temperature perturb, recalibrate,
   or use a weaker hazard model.  HT validity should remain while efficiency
   changes; this empirically separates validity from model-assisted variance
   reduction.
4. **Positivity sensitivity.** The CRC row-cap multiplier is now covered;
   separately vary the terminal propensity floor. Report maximum weights,
   effective sample size, budget utilization, and target-weight variance, not
   coverage alone.
5. **Calibration size separate from N1.** Vary the total calibration/test split
   size while fixing N1, then vary N1 at fixed total size.  This distinguishes
   allocation-learning error from downstream calibration sampling error.
6. **Target-anchor robustness.** Compare raw alpha, Phase-I calibrated anchor,
    a small boundary band, and a multi-target objective.  This directly tests
    sensitivity to the ultimately selected LPB candidate.

## Breadth and practicality

7. **Coverage levels and prior horizon.** Repeat at 70%, 80%, 90%, and 95%
    coverage and vary tau-prior.  Rare target events are where K2 quantization
    and inverse weights are most likely to fail.
8. **Cross-dataset/model replication.** Use at least one toxicity, red-team,
    hallucination, and instruction-following setup, with paired confidence
    intervals for DAPRO minus Static rather than only separate boxplots.
9. **Runtime and memory.** Report fit and deployment time, peak memory, and
    number of learned policy parameters versus N1, K, and horizon.
10. **Oracle efficiency gap.** Include full-budget, target-aware clairvoyant,
    random-score, and time-only policies to show how much of the attainable
    variance reduction is captured by the learned score and representation.

For every follow-up, preserve identical outer splits and acquisition uniforms
across methods.  Primary comparisons should use the exact conditional target
HT variance when available, with across-split coverage variance reported as a
separate downstream outcome.
