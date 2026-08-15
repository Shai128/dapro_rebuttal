# UPB pipeline and Soft-prefix Generalized-DAPRO validation

## Final design

The upper predictive bound (UPB) uses the convention

\[
    f(X)=201 \quad\Longleftrightarrow\quad
    \text{infinity / no event through the real horizon }1{:}200.
\]

Consequently, an event at turn 200 is real and is covered only by a bound at
least 200, whereas 201 covers every benchmark trajectory.  Calibration targets
the finite-candidate miscoverage indicator

\[
    A_i(f)=\mathbf 1\{T_i>f(X_i),\ f(X_i)<201\}.
\]

Every allocation method exposes the candidate-specific probability of reaching
`f`.  The production endpoint/block allocator uses the terminal-residual
augmented Horvitz--Thompson contribution

\[
  \widehat A_i(f)=m_i(f)+
  \frac{R_i(f)}{\pi_i(\min\{T_i,f_i\})}\{A_i(f)-m_i(f)\},
  \qquad
  m_i(f)=\widehat{\Pr}(T_i>f_i\mid X_{i0}).
\]

For every frozen prediction `m`, this is design-unbiased conditionally on the
complete benchmark; model misspecification affects efficiency, not validity.
It is the one-update special case of the all-prefix sequential AHT estimator.
The value 201 has `A=m=0` and is deterministic.

The production UPB registry uses the endpoint/block specialization of
Soft-prefix Generalized DAPRO.  It uses soft residual mass `m(1-m)`, expected
reveal cost, and a median K2 value/cost score.  The action is one Bernoulli
decision at `X_i0` followed by the complete candidate block.  This particular
specialization fits no trajectory labels; that statement does **not** apply to
ordinary history-adaptive DAPRO, which fits its policy on fully observed Phase-I
trajectories.  The CRC version uses 100 independent full controls and the full
200-turn support bound.  It deliberately has no row cap and no causal
shared-PAV envelope.

This block restriction is not assumed to be globally optimal.  A separate
comparison implements (i) conventional, label-fitted, per-turn endpoint-mass
DAPRO and (ii) per-turn information-gain DAPRO with the all-prefix sequential
AHT estimator.  Across 50 matched splits, two setups, and target coverages
60%, 70%, 80%, and 90%, neither dynamic method improves on the block method
generally.  The complete derivation and results are in
`analysis/diagnostics/upb_history_adaptive_comparison.md`.

## Final 50-split results (budget 20, target coverage 70%)

| Setup | Method | Mean coverage (%) | Coverage variance (pp²) | MSE to split full-budget oracle (pp²) | Mean UPB | UPB=201 (%) | Mean expected budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toxicity/Qwen | Static | 69.731 | 3.941 | 3.558 | 125.982 | 39.567 | 12.405 |
| Toxicity/Qwen | Soft DAPRO | 69.751 | 4.515 | **2.809** | 126.064 | 39.641 | 15.619 |
| Toxicity/Qwen | Soft DAPRO + CRC | 69.766 | 5.159 | 3.621 | 126.093 | 39.657 | 17.949 |
| Toxicity/Qwen | Full-budget oracle | 69.905 | 1.409 | 0 | 126.247 | 39.773 | 124.053 |
| Red-team/Qwen | Static | 70.447 | 6.190 | 4.027 | 144.538 | 5.531 | 9.630 |
| Red-team/Qwen | Soft DAPRO | 70.318 | 4.424 | **3.747** | 144.058 | 5.281 | 18.794 |
| Red-team/Qwen | Soft DAPRO + CRC | 70.181 | **4.100** | 3.777 | 143.613 | 5.133 | 18.031 |
| Red-team/Qwen | Full-budget oracle | 70.333 | 1.259 | 0 | 144.669 | 5.060 | 96.300 |

The stable residual acquisition-risk diagnostic is lower for DAPRO on both
setups: toxicity `1.053/1.095` (without/with CRC) versus Static `1.470`, and
red-team `0.840/1.055` versus Static `1.837`.  Red-team also has lower raw
50-split coverage variance.  Toxicity is the non-cherry-picked exception: its
raw variance is slightly larger, although no-CRC DAPRO is closer to the
full-budget split oracle.  Raw coverage variance alone rewards an estimator
that fails to track genuine split-to-split oracle changes, so oracle-relative
MSE and conditional residual risk should be reported alongside it.

All 100 CRC selectors passed their finite-sample selector certificate.  Mean
expected cost is below 20.  Individual split-conditional expected costs can
exceed 20 (maxima 22.27 and 21.07), as expected: the CRC theorem controls the
marginal expected total budget, not every realized calibration split.

## Reproduction

On Linux/Ubuntu, the complete UPB matrix is launched with:

```bash
bash src/predictive_bounds/scripts/calibrate_upb.sh --slurm --parallel-jobs 20 --cpu
```

The shared construct/merge engine continues past failed configurations and can
also be invoked directly:

```bash
bash src/predictive_bounds/scripts/calibrate.sh --bound-type upb --local
# or
bash src/predictive_bounds/scripts/calibrate.sh --bound-type upb --slurm --parallel-jobs 20 --cpu
```

The local validation used experiment suffix `upb_final_validation`.  Its compact
machine-readable summary is `outputs/upb_final_validation_summary.csv`; merged
CSVs are under `results/merged_upb_calibration_dfs`, and figures/tables are under
`outputs/upb_final_validation_figures`.
